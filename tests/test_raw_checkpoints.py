"""Raw count evidence must survive source loss without blocking acquisition."""

import json
import threading
import time

import numpy as np
import pytest

from hf_dm.data import recorder as recorder_module
from hf_dm.data.raw_stream import RAW_DTYPE, WAL_MAGIC, WAL_RECORD, NpzRows, WalRows
from hf_dm.data.recorder import SessionRecorder, SessionSummary, recover_session_journal
from hf_dm.data.session_export import _validate_jsonl_artifact, session_bundle_dir
from hf_dm.data.session_journal import scan_journal


def recording(path, count=20, *, layout="flat"):
    r = SessionRecorder(path, "raw_check", "test", base_name="session")
    r.set_export_options({"layout": layout, "streamdio_chunk_s": 0, "streamdio_wal_fsync_s": 3600})
    append(r, 0, count)
    return r


def append(r, start, count):
    for i in range(start, start + count):
        r.log_streamdio_event("analog_input", 13, i % 4096, i * 1000, i / 1000,
                             host_perf_counter_s=100 + i / 1000)


def close(r, *, durable=True):
    r._close_streamdio_wal(durable=durable)
    r.close_journal()


def checkpoints(r):
    return [row["payload"] for line in r.journal_path.read_text().splitlines()
            if (row := json.loads(line))["kind"] == "streamdio_checkpoint"]


@pytest.mark.parametrize("damage", ["missing", "empty", "short", "crc_tail"])
def test_lost_committed_raw_data_is_refused_before_any_journal_repair(tmp_path, damage):
    r = recording(tmp_path)
    close(r)
    raw = r.streamdio_wal_path.read_bytes()
    if damage == "missing":
        r.streamdio_wal_path.unlink()
    elif damage == "empty":
        r.streamdio_wal_path.write_bytes(WAL_MAGIC)
    elif damage == "short":
        r.streamdio_wal_path.write_bytes(raw[:len(WAL_MAGIC) + 10 * WAL_RECORD.size])
    else:
        r.streamdio_wal_path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 255]))
    with r.journal_path.open("ab") as handle:
        handle.write(b'{"torn":')
    before = r.journal_path.read_bytes()
    with pytest.raises(RuntimeError, match="Missing raw StreamDIO"):
        recover_session_journal(r.incomplete_marker_path)
    assert r.journal_path.read_bytes() == before
    assert r.incomplete_marker_path.exists()
    assert not list(tmp_path.glob("*.recovery-backup"))
    assert not r.base_path.with_suffix(".npz").exists()


@pytest.mark.parametrize("layout", ["flat", "mm_hierarchy"])
@pytest.mark.parametrize("prefix_count", [5, 20])
def test_wal_loss_requires_enough_verified_chunks(tmp_path, layout, prefix_count):
    r = recording(tmp_path, layout=layout)
    close(r)
    expected = list(WalRows(r.streamdio_wal_path))
    chunks = session_bundle_dir(r.base_path, layout=layout) / "continuous_streams"
    chunks.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(chunks / "streamdio_chunk_000.npz",
                        streamdio_events=np.array(expected[:prefix_count], dtype=RAW_DTYPE))
    r.streamdio_wal_path.unlink()
    if prefix_count == 20:
        result = recover_session_journal(r.incomplete_marker_path)
        assert result["validation"]["valid"]
        assert result["raw_checkpoint_verified"]
        assert result["raw_checkpoint_event_count"] == 20
        assert list(NpzRows(r.base_path.with_suffix(".npz"))) == expected
    else:
        before = r.journal_path.read_bytes()
        with pytest.raises(RuntimeError, match="requires at least 20"):
            recover_session_journal(r.incomplete_marker_path)
        assert r.journal_path.read_bytes() == before


def test_uncheckpointed_damaged_tail_remains_recoverable(tmp_path):
    r = recording(tmp_path, 2)
    r._flush_streamdio_wal(durable=True)
    append(r, 2, 1)
    close(r, durable=False)
    raw = r.streamdio_wal_path.read_bytes()
    r.streamdio_wal_path.write_bytes(raw[:-1])
    result = recover_session_journal(r.incomplete_marker_path)
    assert result["validation"]["valid"]
    assert result["raw_checkpoint_event_count"] == result["recovered_streamdio_events"] == 2
    assert result["streamdio_wal_trailing_bytes"] > 0


@pytest.mark.parametrize("missing", [False, True])
def test_empty_stream_has_an_explicit_source_even_before_first_sample(tmp_path, missing):
    r = recording(tmp_path, 0)
    close(r)
    assert r.streamdio_wal_path.read_bytes() == WAL_MAGIC
    if missing:
        r.streamdio_wal_path.unlink()
        with pytest.raises(RuntimeError, match="Missing raw StreamDIO sources"):
            recover_session_journal(r.incomplete_marker_path)
    else:
        result = recover_session_journal(r.incomplete_marker_path)
        assert result["validation"]["valid"] and result["recovered_streamdio_events"] == 0


def test_legacy_journal_reports_missing_independent_count_evidence(tmp_path):
    r = recording(tmp_path)
    close(r)
    rows = [json.loads(line) for line in r.journal_path.read_text().splitlines()]
    rows[0]["payload"].pop("streamdio_checkpoint_version")
    rows = [row for row in rows if row["kind"] != "streamdio_checkpoint"]
    for i, row in enumerate(rows):
        row["sequence"] = i
    r.journal_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = recover_session_journal(r.incomplete_marker_path)
    assert result["validation"]["valid"] and result["recovered_streamdio_events"] == 20
    assert result["raw_checkpoint_verified"] is False
    assert result["raw_checkpoint_event_count"] is None


@pytest.mark.parametrize("count", [-1, True, 1.5, "20", None])
def test_checkpoint_count_schema_cannot_be_coerced(tmp_path, count):
    r = recording(tmp_path, 0)
    r._journal_append("streamdio_checkpoint", {"event_count": count, "wal_fsync_succeeded": True})
    close(r, durable=False)
    with pytest.raises(ValueError, match="checkpoint event_count"):
        recover_session_journal(r.incomplete_marker_path)


def test_decreasing_checkpoint_count_is_not_accepted(tmp_path):
    r = recording(tmp_path, 10)
    r._flush_streamdio_wal(durable=True)
    r._journal_append("streamdio_checkpoint", {"event_count": 5, "wal_fsync_succeeded": True})
    close(r, durable=False)
    with pytest.raises(ValueError, match="event_count decreased"):
        scan_journal(r.journal_path)


@pytest.mark.parametrize("bad", [True, 2, "1"])
def test_unknown_or_coerced_checkpoint_protocol_is_refused(tmp_path, bad):
    r = recording(tmp_path, 0)
    close(r)
    rows = [json.loads(line) for line in r.journal_path.read_text().splitlines()]
    rows[0]["payload"]["streamdio_checkpoint_version"] = bad
    r.journal_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="checkpoint version"):
        recover_session_journal(r.incomplete_marker_path)
    assert any("checkpoint version" in error for error in
               _validate_jsonl_artifact(r.journal_path, r.journal_path.name, {}))


def test_invalid_checkpoint_sync_status_is_refused(tmp_path):
    r = recording(tmp_path, 0)
    r._journal_append("streamdio_checkpoint", {"event_count": 0, "wal_fsync_succeeded": "yes"})
    close(r, durable=False)
    with pytest.raises(ValueError, match="checkpoint fsync status"):
        recover_session_journal(r.incomplete_marker_path)


def test_periodic_worker_records_count_without_waiting_for_a_trial(tmp_path):
    r = recording(tmp_path, 0)
    r._streamdio_wal_fsync_s = .005
    append(r, 0, 1)
    deadline = time.perf_counter() + 1
    while r._streamdio_wal_fsync_count == 0 and time.perf_counter() < deadline:
        time.sleep(.005)
    close(r, durable=False)
    assert checkpoints(r)[-1]["event_count"] == 1


def test_slow_flush_allows_appends_and_orders_concurrent_checkpoints(tmp_path, monkeypatch):
    r = recording(tmp_path, 5)
    flushing = threading.Event()
    release = threading.Event()
    errors = []
    sync = recorder_module.fsync_best_effort

    def slow_sync(handle, *, path=None):
        if path == r.streamdio_wal_path and not flushing.is_set():
            flushing.set()
            assert release.wait(2), "test did not release the simulated disk flush"
        return sync(handle, path=path)

    def flush():
        try:
            r._flush_streamdio_wal(durable=True)
        except BaseException as exc:
            errors.append(exc)

    monkeypatch.setattr(recorder_module, "fsync_best_effort", slow_sync)
    first = threading.Thread(target=flush)
    second = threading.Thread(target=flush)
    writer = threading.Thread(target=append, args=(r, 5, 3))
    first.start()
    try:
        assert flushing.wait(1)
        writer.start()
        writer.join(.5)
        assert not writer.is_alive(), "WAL fsync blocked incoming samples"
        second.start()
    finally:
        release.set()
        first.join(2)
        if second.ident is not None:
            second.join(2)
        if writer.ident is not None:
            writer.join(2)
        close(r)
    assert not errors
    assert [item["event_count"] for item in checkpoints(r)] == [5, 8]
    assert scan_journal(r.journal_path).raw_minimum_count == 8


@pytest.mark.parametrize("failure", ["wal", "checkpoint"])
def test_failed_sync_or_checkpoint_retains_evidence_and_closes_wal(tmp_path, monkeypatch, failure):
    r = recording(tmp_path)
    if failure == "wal":
        sync = recorder_module.fsync_best_effort

        def fail(handle, *, path=None):
            if path == r.streamdio_wal_path:
                raise OSError("injected WAL flush failure")
            return sync(handle, path=path)

        monkeypatch.setattr(recorder_module, "fsync_best_effort", fail)
    else:
        write = r._journal_append

        def fail_write(kind, payload, **kwargs):
            if kind == "streamdio_checkpoint":
                raise OSError("injected checkpoint failure")
            return write(kind, payload, **kwargs)

        monkeypatch.setattr(r, "_journal_append", fail_write)
    with pytest.raises(OSError, match="injected"):
        r._close_streamdio_wal(durable=True)
    r.close_journal()
    assert not checkpoints(r)
    assert r._streamdio_wal_handle is None
    assert r.incomplete_marker_path.exists() and r.streamdio_wal_path.exists()


def test_unsupported_fsync_is_recorded_without_claiming_disk_durability(tmp_path, monkeypatch):
    r = recording(tmp_path)
    monkeypatch.setattr(recorder_module, "fsync_best_effort", lambda *a, **k: False)
    close(r)
    assert checkpoints(r) == [{"event_count": 20, "wal_fsync_succeeded": False}]
    result = recover_session_journal(r.incomplete_marker_path)
    assert result["validation"]["valid"]


def test_journal_fsync_failure_does_not_reuse_a_written_checkpoint_sequence(tmp_path, monkeypatch):
    r = recording(tmp_path)
    sync = recorder_module.fsync_best_effort

    def fail(handle, *, path=None):
        if path == r.journal_path:
            raise OSError("injected journal fsync failure")
        return sync(handle, path=path)

    with monkeypatch.context() as patch:
        patch.setattr(recorder_module, "fsync_best_effort", fail)
        with pytest.raises(OSError, match="journal fsync failure"):
            r._flush_streamdio_wal(durable=True)
    r.log_event("flush_failure_reported", {})
    close(r, durable=False)
    assert scan_journal(r.journal_path).raw_minimum_count == 20
    result = recover_session_journal(r.incomplete_marker_path)
    assert result["validation"]["valid"] and result["recovered_streamdio_events"] == 20


def test_periodic_checkpoint_failure_is_latched_for_session_abort(tmp_path, monkeypatch):
    r = recording(tmp_path, 0)
    r._streamdio_wal_fsync_s = .005
    write = r._journal_append

    def fail(kind, payload, **kwargs):
        if kind == "streamdio_checkpoint":
            raise OSError("injected periodic checkpoint failure")
        return write(kind, payload, **kwargs)

    monkeypatch.setattr(r, "_journal_append", fail)
    append(r, 0, 1)
    deadline = time.perf_counter() + 1
    while r._streamdio_wal_background_error is None and time.perf_counter() < deadline:
        time.sleep(.005)
    close(r, durable=False)
    with pytest.raises(RuntimeError, match="periodic checkpoint failure"):
        r.raise_if_background_error()


def test_finalization_keeps_final_checkpoint_in_validated_journal(tmp_path):
    r = recording(tmp_path, 12)
    r.finalize(SessionSummary("raw_check", "test", 0, 0, "completed", 0, 1, 2))
    assert checkpoints(r)[-1]["event_count"] == 12
    assert not r.streamdio_wal_path.exists()
    errors = _validate_jsonl_artifact(r.journal_path, r.journal_path.name,
                                    {"counts": {"streamdio_events": 11}})
    assert any("checkpoint exceeds exported raw count" in error for error in errors)
