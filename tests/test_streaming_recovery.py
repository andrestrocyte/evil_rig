"""Crash recovery must remain bounded, ordered and retryable after write failure."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from hf_dm.config import SessionConfig, config_to_dict
from hf_dm.data import recorder as recorder_module
from hf_dm.data import raw_stream
from hf_dm.data.raw_stream import BLOCK_ROWS, RAW_DTYPE, WAL_MAGIC, WAL_RECORD, ChunkRows, NpzRows, WalRows
from hf_dm.data.recorder import SessionRecorder, _reconcile_streamdio_recovery_rows, recover_session_journal
from hf_dm.data.session_export import session_bundle_dir


def crashed_recording(path, count=20, *, chunk_s=0, layout="flat", durable=True):
    config = SessionConfig(subject_id="recover_test", n_trials=0, results_dir=str(path))
    config.camera.required = False
    config.session_export.layout = layout
    config.session_export.streamdio_memory_mode = "bounded"
    config.session_export.streamdio_recent_events = 1
    r = SessionRecorder(path, config.subject_id, config.phase.value)
    r.set_export_options(config_to_dict(config)["session_export"])
    # Journal chunk settings without running a live chunk writer. Recovery is
    # responsible for the missing suffix after this simulated interruption.
    r.export_options["streamdio_chunk_s"] = chunk_s
    r._journal_append("export_options", r.export_options)
    r.set_metadata({"config": config_to_dict(config)})
    for i in range(count):
        r.log_streamdio_event("analog_input", 13, i % 4096, i * 1000, i / 1000,
                             host_perf_counter_s=100 + i / 1000)
    r._close_streamdio_wal(durable=durable)
    r.close_journal()
    return r


def write_prefix(r, rows):
    directory = session_bundle_dir(r.base_path, layout="flat") / "continuous_streams"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "streamdio_chunk_000.npz"
    np.savez_compressed(path, streamdio_events=np.array(rows, dtype=RAW_DTYPE))
    return path


def test_recovery_appends_bounded_suffix_without_loading_full_wal_or_overwriting_prefix(tmp_path, monkeypatch):
    count = BLOCK_ROWS * 2 + 19
    r = crashed_recording(tmp_path, count, chunk_s=1000000)
    wal = WalRows(r.streamdio_wal_path)
    prefix = write_prefix(r, wal[:3])
    original = prefix.read_bytes()
    writes = []
    write = recorder_module._atomic_write_npz

    def bounded_write(path, **arrays):
        if path.name.startswith("streamdio_chunk_"):
            assert isinstance(arrays["streamdio_events"], raw_stream.RawArray)
            writes.append(len(arrays["streamdio_events"].rows))
            assert writes[-1] <= BLOCK_ROWS
        return write(path, **arrays)

    monkeypatch.setattr(recorder_module, "_atomic_write_npz", bounded_write)
    monkeypatch.setattr(recorder_module, "load_streamdio_wal", lambda *args: pytest.fail("list loader forbidden"))
    monkeypatch.setattr(SessionRecorder, "_configure_streamdio_chunking", lambda *args: pytest.fail("live queue forbidden"))
    result = recover_session_journal(r.incomplete_marker_path)
    assert result["validation"]["valid"]
    assert result["recovered_streamdio_events"] == count
    assert writes == [BLOCK_ROWS, BLOCK_ROWS, 16]
    assert prefix.read_bytes() == original
    actual = NpzRows(Path(result["artifact"]))
    assert len(actual) == count
    assert [row[2] for row in actual] == [i % 4096 for i in range(count)]
    manifest = json.loads((Path(result["bundle_dir"]) / "manifest.json").read_text())
    assert manifest["continuous_streams"]["event_count"] == count
    assert manifest["recorder_stats"]["max_streamdio_events_in_memory"] == 0


def test_recovery_chunk_write_failure_keeps_sources_and_can_be_retried(tmp_path, monkeypatch):
    r = crashed_recording(tmp_path, chunk_s=.005)
    wal_before = r.streamdio_wal_path.read_bytes()
    prefix = write_prefix(r, WalRows(r.streamdio_wal_path)[:3])
    prefix_before = prefix.read_bytes()
    write = recorder_module._atomic_write_npz
    writes = 0

    def fail_second_chunk(path, **arrays):
        nonlocal writes
        if path.name.startswith("streamdio_chunk_"):
            writes += 1
            if writes == 2:
                raise OSError("injected disk failure")
        return write(path, **arrays)

    monkeypatch.setattr(recorder_module, "_atomic_write_npz", fail_second_chunk)
    with pytest.raises(OSError, match="injected disk failure"):
        recover_session_journal(r.incomplete_marker_path)
    assert r.incomplete_marker_path.exists()
    assert r.streamdio_wal_path.read_bytes() == wal_before
    assert prefix.read_bytes() == prefix_before
    monkeypatch.setattr(recorder_module, "_atomic_write_npz", write)
    result = recover_session_journal(r.incomplete_marker_path)
    assert result["validation"]["valid"]
    assert len(NpzRows(Path(result["artifact"]))) == 20
    assert prefix.read_bytes() == prefix_before


@pytest.mark.parametrize("damage", ["partial", "crc_tail", "crc_middle"])
def test_recovery_preserves_damaged_tail_bytes_and_refuses_interior_corruption(tmp_path, damage):
    # A CRC-damaged last record can be a crash tail only when no checkpoint
    # already promises it. Synced-record loss is tested in test_raw_checkpoints.
    r = crashed_recording(tmp_path, count=3, durable=damage != "crc_tail")
    raw = bytearray(r.streamdio_wal_path.read_bytes())
    if damage == "partial":
        raw.extend(b"partial")
    elif damage == "crc_tail":
        raw[-5] ^= 255
    else:
        raw[len(WAL_MAGIC) + WAL_RECORD.size + 30] ^= 255
    r.streamdio_wal_path.write_bytes(raw)
    if damage == "crc_middle":
        with pytest.raises(ValueError, match="interior checksum"):
            recover_session_journal(r.incomplete_marker_path)
        assert r.incomplete_marker_path.exists()
        assert r.streamdio_wal_path.read_bytes() == raw
        assert not r.base_path.with_suffix(".npz").exists()
    else:
        result = recover_session_journal(r.incomplete_marker_path)
        assert result["validation"]["valid"]
        assert result["recovered_streamdio_events"] == (2 if damage == "crc_tail" else 3)
        assert Path(result["streamdio_wal_recovery_backup"]).read_bytes() == raw
        manifest = json.loads(Path(result["manifest"]).read_text())
        assert any(item["path"].endswith(".wal.recovery-backup") for item in manifest["artifacts"])


@pytest.mark.parametrize("damage", ["missing_middle", "bad_zip", "divergent"])
def test_bad_chunk_source_never_becomes_silent_empty_stream(tmp_path, damage):
    r = crashed_recording(tmp_path)
    prefix = write_prefix(r, WalRows(r.streamdio_wal_path)[:3])
    if damage == "missing_middle":
        prefix.rename(prefix.with_name("streamdio_chunk_001.npz"))
    elif damage == "bad_zip":
        prefix.write_bytes(b"not a zip file")
    else:
        rows = WalRows(r.streamdio_wal_path)[:3]
        rows[1] = ("analog_input", 13, 777, *rows[1][3:])
        np.savez_compressed(prefix, streamdio_events=np.array(rows, dtype=RAW_DTYPE))
    with pytest.raises(Exception, match="gap|zip|diverge"):
        recover_session_journal(r.incomplete_marker_path)
    assert r.incomplete_marker_path.exists()
    assert r.streamdio_wal_path.exists()
    assert not r.base_path.with_suffix(".npz").exists()


def test_failed_recovery_validation_keeps_wal_and_incomplete_marker(tmp_path, monkeypatch):
    from hf_dm.data import session_export
    r = crashed_recording(tmp_path)
    with r.streamdio_wal_path.open("ab") as handle:
        handle.write(b"partial")
    before = r.streamdio_wal_path.read_bytes()
    validate = session_export.validate_session
    monkeypatch.setattr(session_export, "validate_session", lambda *args: {"valid": False, "errors": ["injected"]})
    result = recover_session_journal(r.incomplete_marker_path)
    assert result["validation"]["valid"] is False
    assert r.incomplete_marker_path.exists()
    assert r.streamdio_wal_path.read_bytes() == before
    monkeypatch.setattr(session_export, "validate_session", validate)
    retried = recover_session_journal(r.incomplete_marker_path)
    assert retried["validation"]["valid"]
    assert not r.incomplete_marker_path.exists()


def test_reconciliation_is_sequential_and_validates_longer_tail():
    class Sequential(raw_stream.FileRows):
        _count = 3

        def __getitem__(self, index):
            pytest.fail("recovery must not repeatedly index a file")

        def __iter__(self):
            yield ("camera", 0, 0, 0, 0., 0.)
            raise ValueError("corrupt longer tail")

    with pytest.raises(ValueError, match="corrupt longer tail"):
        _reconcile_streamdio_recovery_rows(Sequential(), [("camera", 0, 0, 0, 0., 0.)])


def test_failed_wal_cleanup_cannot_clear_incomplete_marker(tmp_path, monkeypatch):
    r = crashed_recording(tmp_path)
    unlink = Path.unlink

    def locked(path, *args, **kwargs):
        if path == r.streamdio_wal_path:
            raise PermissionError("WAL still in use")
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", locked)
    with pytest.raises(PermissionError, match="still in use"):
        recover_session_journal(r.incomplete_marker_path)
    assert r.incomplete_marker_path.exists()
    assert r.streamdio_wal_path.exists()


def interrupt_recovery_cleanup(r, monkeypatch, *, target="marker"):
    """Fail after validation, on either side of deleting the redundant WAL."""
    blocked = r.incomplete_marker_path if target == "marker" else r.streamdio_wal_path
    unlink = Path.unlink

    def fail(path, *args, **kwargs):
        if path == blocked:
            raise PermissionError("injected cleanup failure")
        return unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", fail)
        with pytest.raises(PermissionError, match="injected cleanup failure"):
            recover_session_journal(r.incomplete_marker_path)
    assert r.incomplete_marker_path.exists()


@pytest.mark.parametrize("target", ["marker", "wal"])
@pytest.mark.parametrize("layout", ["flat", "mm_hierarchy"])
def test_cleanup_retry_preserves_verified_npz_and_journal(tmp_path, monkeypatch, target, layout):
    r = crashed_recording(tmp_path, count=23, layout=layout)
    expected = list(WalRows(r.streamdio_wal_path))
    interrupt_recovery_cleanup(r, monkeypatch, target=target)
    assert r.streamdio_wal_path.exists() == (target == "wal")
    artifact = r.base_path.with_suffix(".npz")
    before = artifact.read_bytes(), r.journal_path.read_bytes()
    # Resuming must validate the finished files; it must not replay the journal.
    monkeypatch.setattr(recorder_module, "prepare_journal", lambda *a, **k: pytest.fail("unexpected replay"))
    result = recover_session_journal(r.incomplete_marker_path)
    assert result["resumed_completed_recovery"]
    assert result["validation"]["valid"]
    assert result["recovered_streamdio_events"] == len(expected)
    assert list(NpzRows(artifact)) == expected
    assert (artifact.read_bytes(), r.journal_path.read_bytes()) == before
    assert not r.incomplete_marker_path.exists()
    assert not r.streamdio_wal_path.exists()


def test_completion_checkpoint_write_failure_keeps_wal_for_retry(tmp_path, monkeypatch):
    r = crashed_recording(tmp_path)
    before = r.streamdio_wal_path.read_bytes()
    replace = recorder_module.os.replace

    def fail(source, destination):
        if Path(destination) == r.incomplete_marker_path:
            payload = json.loads(Path(source).read_text(encoding="utf-8"))
            if "recovery_completion" in payload:
                raise OSError("injected checkpoint replacement failure")
        return replace(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr(recorder_module.os, "replace", fail)
        with pytest.raises(OSError, match="checkpoint replacement failure"):
            recover_session_journal(r.incomplete_marker_path)
    assert "recovery_completion" not in json.loads(r.incomplete_marker_path.read_text())
    assert r.streamdio_wal_path.read_bytes() == before
    result = recover_session_journal(r.incomplete_marker_path)
    assert result["validation"]["valid"]
    assert result["recovered_streamdio_events"] == 20


@pytest.mark.parametrize("damage", ["manifest", "artifact", "journal", "missing_manifest", "outside", "schema", "result"])
def test_completed_recovery_damage_refuses_retry_without_mutating_sources(tmp_path, monkeypatch, damage):
    r = crashed_recording(tmp_path)
    interrupt_recovery_cleanup(r, monkeypatch, target="wal")
    marker = json.loads(r.incomplete_marker_path.read_text())
    manifest = r.journal_path.parent / marker["recovery_completion"]["manifest"]
    if damage == "manifest":
        with manifest.open("a") as handle:
            handle.write(" ")
    elif damage == "missing_manifest":
        manifest.unlink()
    elif damage in {"artifact", "journal"}:
        path = r.base_path.with_suffix(".npz") if damage == "artifact" else r.journal_path
        with path.open("ab") as handle:
            handle.write(b"damage")
    else:
        key, value = {"outside": ("manifest", "../manifest.json"),
                      "schema": ("schema_version", 99), "result": ("result", None)}[damage]
        marker["recovery_completion"][key] = value
        r.incomplete_marker_path.write_text(json.dumps(marker))
    paths = (r.incomplete_marker_path, r.streamdio_wal_path, r.journal_path, r.base_path.with_suffix(".npz"))
    before = [path.read_bytes() for path in paths]
    with pytest.raises(RuntimeError, match="checkpoint|manifest|no longer validates"):
        recover_session_journal(r.incomplete_marker_path)
    assert [path.read_bytes() for path in paths] == before


def test_completed_recovery_can_resume_from_a_copy_with_local_evidence(tmp_path, monkeypatch):
    source = tmp_path / "source"
    r = crashed_recording(source)
    with r.journal_path.open("ab") as handle:
        handle.write(b'{"unfinished":')
    with r.streamdio_wal_path.open("ab") as handle:
        handle.write(b"partial")
    interrupt_recovery_cleanup(r, monkeypatch)
    copied = tmp_path / "copy"
    shutil.copytree(source, copied)
    original_marker = r.incomplete_marker_path.read_bytes()
    # Both the CLI's relative input and copied recovery evidence must work.
    monkeypatch.chdir(tmp_path)
    result = recover_session_journal(Path("copy") / r.incomplete_marker_path.name)
    assert result["validation"]["valid"]
    assert result["recovered_streamdio_events"] == 20
    assert result["resumed_completed_recovery"]
    assert Path(result["streamdio_wal_recovery_backup"]).resolve().parent == copied
    assert all(Path(path).resolve().parent == copied for path in result["journal_recovery_backups"])
    assert r.incomplete_marker_path.read_bytes() == original_marker
    assert not (copied / r.incomplete_marker_path.name).exists()


@pytest.mark.parametrize("layout", ["flat", "mm_hierarchy"])
@pytest.mark.parametrize("damage", [False, True])
def test_legacy_cleanup_retry_never_replaces_finished_raw_data(tmp_path, monkeypatch, layout, damage):
    r = crashed_recording(tmp_path, count=7, layout=layout)
    interrupt_recovery_cleanup(r, monkeypatch)
    state = json.loads(r.incomplete_marker_path.read_text())
    del state["recovery_completion"]  # Recoveries before the durable checkpoint.
    r.incomplete_marker_path.write_text(json.dumps(state))
    if damage:
        with r.base_path.with_suffix(".npz").open("ab") as handle:
            handle.write(b"damage")
    before = r.base_path.with_suffix(".npz").read_bytes(), r.journal_path.read_bytes()
    if damage:
        with pytest.raises(RuntimeError, match="WAL is missing"):
            recover_session_journal(r.incomplete_marker_path)
        assert r.incomplete_marker_path.exists()
    else:
        result = recover_session_journal(r.incomplete_marker_path)
        assert result["validation"]["valid"]
        assert result["recovered_streamdio_events"] == 7
        assert result["streamdio_recovery_source"] == "verified_existing_bundle"
        assert not r.incomplete_marker_path.exists()
    assert (r.base_path.with_suffix(".npz").read_bytes(), r.journal_path.read_bytes()) == before


def test_recovery_completion_checkpoint_accepts_relative_cli_path(tmp_path, monkeypatch):
    r = crashed_recording(tmp_path / "session")
    monkeypatch.chdir(tmp_path)
    result = recover_session_journal(Path("session") / r.incomplete_marker_path.name)
    assert result["validation"]["valid"]
    assert result["recovered_streamdio_events"] == 20


def test_chunk_order_is_numeric_after_999(tmp_path, monkeypatch):
    # No SDK/filesystem load is relevant to index ordering. Exercise the same
    # reader with more than 1,000 one-row parts and out-of-order path discovery.
    monkeypatch.setattr(raw_stream, "NpzRows", lambda path: [("camera", 0, 0, raw_stream.chunk_number(path), 0., 0.)])
    paths = [tmp_path / f"streamdio_chunk_{i:03d}.npz" for i in reversed(range(1002))]
    rows = ChunkRows(paths)
    assert [row[3] for row in rows] == list(range(1002))


def test_continuous_manifest_keeps_numeric_chunk_order(tmp_path):
    from hf_dm.data.session_export import _continuous_stream_summary
    directory = tmp_path / "continuous_streams"
    directory.mkdir()
    paths = []
    for index in [1000, 999]:
        path = directory / f"streamdio_chunk_{index:03d}.npz"
        np.savez_compressed(path, streamdio_events=np.array(
            [("camera", 0, 0, index, 0., 0.)], dtype=RAW_DTYPE))
        paths.append(path)
    index = _continuous_stream_summary(tmp_path, paths)
    assert [item["first_module_timestamp_us"] for item in index["chunks"]] == [999, 1000]


def test_conflict_closes_both_recovery_readers_before_exception_escapes():
    closed = []

    class Reader(raw_stream.FileRows):
        _count = 2

        def __init__(self, value):
            self.value = value

        def __iter__(self):
            try:
                for _ in range(2):
                    yield ("camera", 0, self.value, 0, 0., 0.)
            finally:
                closed.append(self.value)

    with pytest.raises(RuntimeError, match="diverge") as error:
        _reconcile_streamdio_recovery_rows(Reader(1), Reader(2))
    assert error.value is not None  # retain traceback; don't rely on GC for cleanup
    assert sorted(closed) == [1, 2]


@pytest.mark.parametrize("prefix_count", [0, 3])
def test_recovery_validates_when_exports_disabled_and_completes_any_existing_chunk_prefix(tmp_path, prefix_count):
    r = crashed_recording(tmp_path, chunk_s=0)
    # These are supported lightweight export settings. A prefix can remain when
    # chunking was disabled later; it must not appear as a complete shorter stream.
    with r.journal_path.open("a", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in r.journal_path.read_text().splitlines()]
        handle.write(json.dumps({"schema_version": 1, "sequence": len(rows), "kind": "export_options",
            "recorded_wall_s": rows[-1]["recorded_wall_s"],
            "payload": {**r.export_options, "manifest": False, "config_snapshot": False,
                        "export_csv_on_finalize": False, "export_timeline": False}}) + "\n")
    if prefix_count:
        write_prefix(r, WalRows(r.streamdio_wal_path)[:prefix_count])
    result = recover_session_journal(r.incomplete_marker_path)
    assert result["validation"]["valid"], result["validation"]
    assert result["recovered_streamdio_events"] == 20
    manifest = json.loads(Path(result["manifest"]).read_text())
    if prefix_count:
        assert manifest["continuous_streams"]["event_count"] == 20
    assert not r.incomplete_marker_path.exists()
