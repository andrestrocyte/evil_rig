"""Crash boundaries, corruption refusal and exact recovery evidence."""

import json
import shutil
from pathlib import Path

import pytest

from hf_dm.data import session_journal
from hf_dm.data.recorder import SessionRecorder, SessionSummary, load_session_journal, recover_session_journal
from hf_dm.data.session_export import export_existing_session, validate_session


def _row(sequence, *, kind="event", payload=None):
    return {"schema_version": 1, "sequence": sequence, "kind": kind,
            "recorded_wall_s": 1.0, "payload": {} if payload is None else payload}


def _journal(tmp_path):
    path = tmp_path / "session.journal.jsonl"
    rows = [_row(0, kind="session_header"), _row(1, payload={"detail": "café\u2028note"})]
    path.write_bytes(b"\r\n".join(json.dumps(row, ensure_ascii=False).encode("utf-8") for row in rows) + b"\r\n")
    return path, rows


@pytest.mark.parametrize("tail", [b'{"schema_version":1', b'{"detail":"caf\xc3'])
def test_torn_tail_preserves_every_complete_unicode_row_and_original_bytes(tmp_path, tail):
    path, expected = _journal(tmp_path)
    original = path.read_bytes() + tail
    path.write_bytes(original)
    with pytest.raises(ValueError, match="JSON/UTF-8"):
        load_session_journal(path)
    rows, repaired = load_session_journal(path, repair_trailing=True)
    assert repaired and rows == expected
    backups = session_journal.journal_backups(path)
    assert len(backups) == 1 and backups[0].read_bytes() == original
    assert load_session_journal(path) == (expected, False)


@pytest.mark.parametrize("tail", [b'{"broken":\n', b'{"broken":\r\n\n', b'\xff',
                                  b'{"detail":"\xc3\n', b'{"broken":\n{}'])
def test_complete_or_interior_corruption_is_never_discarded_as_a_torn_tail(tmp_path, tail):
    path, _ = _journal(tmp_path)
    original = path.read_bytes() + tail
    path.write_bytes(original)
    with pytest.raises(ValueError, match="JSON/UTF-8"):
        load_session_journal(path, repair_trailing=True)
    assert path.read_bytes() == original
    assert session_journal.journal_backups(path) == []


@pytest.mark.parametrize("bad", [[], _row(7), _row(2, payload=[]), _row(2, kind="unknown"),
                                 _row(2, kind="session_header"), {**_row(2), "schema_version": 9}])
def test_valid_json_with_invalid_schema_is_not_tail_damage(tmp_path, bad):
    path, _ = _journal(tmp_path)
    original = path.read_bytes() + json.dumps(bad).encode()
    path.write_bytes(original)
    with pytest.raises(ValueError):
        load_session_journal(path, repair_trailing=True)
    assert path.read_bytes() == original
    assert session_journal.journal_backups(path) == []


def _crashed_recorder(tmp_path):
    recorder = SessionRecorder(tmp_path, "S1", "test", base_name="session")
    recorder.set_export_options({"layout": "flat"})
    recorder.log_event("test_note", {"detail": "keep this complete event"})
    recorder.log_streamdio_event("analog_input", 13, 17, 1000, .001)
    recorder.close_journal()
    recorder._close_streamdio_wal(durable=True)
    return recorder


def test_recovery_keeps_valid_last_row_when_only_the_newline_was_lost(tmp_path):
    recorder = _crashed_recorder(tmp_path)
    original = recorder.journal_path.read_bytes().rstrip(b"\r\n")
    recorder.journal_path.write_bytes(original)
    result = recover_session_journal(recorder.incomplete_marker_path)
    assert result["validation"]["valid"], result["validation"]["errors"]
    assert result["journal_newline_added"] and result["journal_trailing_bytes"] == 0
    rows, repaired = load_session_journal(recorder.journal_path)
    assert not repaired
    notes = [row["payload"] for row in rows if row["payload"].get("event_type") == "test_note"]
    assert len(notes) == 1 and notes[0]["payload"]["detail"] == "keep this complete event"
    backup = Path(result["journal_recovery_backups"][0])
    assert backup.read_bytes() == original
    backup.write_bytes(original + b"tampered")
    assert not validate_session(result["bundle_dir"])["valid"]
    with pytest.raises(RuntimeError, match="Recovery backup checksum mismatch"):
        export_existing_session(recorder.base_path.with_suffix(".json"))


def test_completed_manifest_free_session_is_not_repaired_before_refusal(tmp_path):
    recorder = SessionRecorder(tmp_path, "S1", "test", base_name="session")
    recorder.set_export_options({"manifest": False, "layout": "flat"})
    recorder.finalize(SessionSummary("S1", "test", 0, 0, "completed", 0, 1, 2))
    original = recorder.journal_path.read_bytes() + b'{"partial":'
    recorder.journal_path.write_bytes(original)
    with pytest.raises(RuntimeError, match="already contains a final summary"):
        recover_session_journal(recorder.journal_path)
    assert recorder.journal_path.read_bytes() == original
    assert session_journal.journal_backups(recorder.journal_path) == []


def test_repeated_repairs_preserve_distinct_backups_and_reuse_identical_bytes(tmp_path):
    path, expected = _journal(tmp_path)
    first = path.read_bytes() + b'{"first":'
    path.write_bytes(first)
    load_session_journal(path, repair_trailing=True)
    path.write_bytes(first)
    load_session_journal(path, repair_trailing=True)
    assert len(session_journal.journal_backups(path)) == 1
    second = path.read_bytes() + b'{"second":'
    path.write_bytes(second)
    assert load_session_journal(path, repair_trailing=True) == (expected, True)
    backups = session_journal.journal_backups(path)
    assert len(backups) == 2
    assert {item.read_bytes() for item in backups} == {first, second}


def test_failed_atomic_repair_keeps_source_and_backup_and_can_retry(tmp_path, monkeypatch):
    path, expected = _journal(tmp_path)
    original = path.read_bytes() + b'{"torn":'
    path.write_bytes(original)
    replace = session_journal.os.replace

    def fail_rewrite(source, target):
        if Path(target) == path:
            raise OSError("injected replace failure")
        replace(source, target)

    monkeypatch.setattr(session_journal.os, "replace", fail_rewrite)
    with pytest.raises(OSError, match="replace failure"):
        load_session_journal(path, repair_trailing=True)
    assert path.read_bytes() == original
    assert session_journal.journal_backups(path)[0].read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))
    monkeypatch.setattr(session_journal.os, "replace", replace)
    assert load_session_journal(path, repair_trailing=True) == (expected, True)


def test_failed_validation_retry_retains_each_original_journal_in_manifest(tmp_path, monkeypatch):
    from hf_dm.data import session_export

    recorder = _crashed_recorder(tmp_path)
    path = recorder.journal_path
    first = path.read_bytes() + b'{"first":'
    path.write_bytes(first)
    validator = session_export.validate_session
    monkeypatch.setattr(session_export, "validate_session", lambda *args: {"valid": False, "errors": ["injected"]})
    failed = recover_session_journal(recorder.incomplete_marker_path)
    assert not failed["validation"]["valid"] and recorder.incomplete_marker_path.exists()
    second = path.read_bytes() + b'{"detail":"\xc3'
    path.write_bytes(second)
    monkeypatch.setattr(session_export, "validate_session", validator)
    result = recover_session_journal(recorder.incomplete_marker_path)
    assert result["validation"]["valid"], result["validation"]["errors"]
    assert result["replaced_prior_summary"]
    assert {Path(name).read_bytes() for name in result["journal_recovery_backups"]} == {first, second}
    manifest_path = Path(result["manifest"])
    manifest = json.loads(manifest_path.read_text())
    artifacts = {(manifest_path.parent / item["path"]).resolve() for item in manifest["artifacts"]}
    assert all(Path(name).resolve() in artifacts for name in result["journal_recovery_backups"])


def test_recovery_scan_refuses_file_changes_before_replay_or_repair(tmp_path):
    path, _ = _journal(tmp_path)
    scan = session_journal.scan_journal(path)
    replay = session_journal.iter_journal(scan)
    next(replay)
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(RuntimeError, match="changed during recovery"):
        list(replay)
    with pytest.raises(RuntimeError, match="changed during recovery"):
        list(session_journal.iter_journal(scan))
    with pytest.raises(RuntimeError, match="changed during recovery"):
        session_journal.prepare_journal(scan)
    assert session_journal.journal_backups(path) == []


@pytest.mark.parametrize("damage", [None, "missing", "changed"])
def test_retry_in_a_copied_directory_requires_local_recovery_evidence(tmp_path, monkeypatch, damage):
    from hf_dm.data import session_export

    original_dir, copied_dir = tmp_path / "original", tmp_path / "copied"
    recorder = _crashed_recorder(original_dir)
    with recorder.journal_path.open("ab") as handle:
        handle.write(b'{"torn":')
    validator = session_export.validate_session
    monkeypatch.setattr(session_export, "validate_session", lambda *args: {"valid": False, "errors": ["injected"]})
    recover_session_journal(recorder.incomplete_marker_path)
    monkeypatch.setattr(session_export, "validate_session", validator)
    shutil.copytree(original_dir, copied_dir)
    if damage == "missing":
        (copied_dir / "session.journal.jsonl.recovery-backup").unlink()
    elif damage == "changed":
        (copied_dir / "session.journal.jsonl.recovery-backup").write_bytes(b"wrong evidence")
    if damage:
        original = (copied_dir / recorder.journal_path.name).read_bytes()
        with pytest.raises((FileNotFoundError, RuntimeError), match="Recovery backup missing|checksum mismatch"):
            recover_session_journal(copied_dir / recorder.incomplete_marker_path.name)
        assert (copied_dir / recorder.incomplete_marker_path.name).exists()
        assert (copied_dir / recorder.journal_path.name).read_bytes() == original
    else:
        result = recover_session_journal(copied_dir / recorder.incomplete_marker_path.name)
        assert result["validation"]["valid"], result["validation"]["errors"]
        manifest_path = Path(result["manifest"])
        manifest = json.loads(manifest_path.read_text())
        backups = [(manifest_path.parent / item["path"]).resolve() for item in manifest["artifacts"]
                   if item["path"].endswith(".recovery-backup")]
        assert backups and all(item.is_relative_to(copied_dir) for item in backups)
