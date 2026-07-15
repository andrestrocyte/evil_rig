import json
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

import hf_dm.data.recorder as recorder_module

from hf_dm.config import SessionConfig, config_to_dict
from hf_dm.data.recorder import (
    SessionRecorder,
    SessionSummary,
    TrialRecord,
    load_session_journal,
    load_streamdio_wal,
    recover_session_journal,
)
from hf_dm.data.session_export import session_bundle_dir, validate_session
from hf_dm.session_qc import evaluate_session_qc


def _trial(index=1):
    return TrialRecord(
        trial_index=index,
        phase="decision_making",
        force_direction="cw",
        choice="cw",
        matched=True,
        reward_duration_s=0.08,
        responded=True,
        response_time_s=0.2,
        net_pulses=3,
        cw_pulses=3,
        ccw_pulses=0,
        final_position_deg=35.0,
        force_samples=2,
        force_last=100.0,
        force_mean=90.0,
        force_min=80.0,
        force_max=100.0,
        trial_start_wall_s=10.0,
        trial_start_mono_s=20.0,
    )


def _summary(recorder, *, reason="completed"):
    return SessionSummary(
        subject_id="S1",
        phase="decision_making",
        n_trials_planned=1,
        n_trials_completed=1,
        end_reason=reason,
        consecutive_failures=0,
        session_start_wall_s=recorder._session_start_wall,
        session_end_wall_s=recorder._session_start_wall + 1.0,
    )


def _recorder(tmp_path):
    config = SessionConfig(subject_id="S1", n_trials=1, results_dir=str(tmp_path))
    config.session_export.layout = "flat"
    recorder = SessionRecorder(tmp_path, "S1", "decision_making")
    recorder.set_export_options(config_to_dict(config)["session_export"])
    recorder.set_metadata({"config": config_to_dict(config)})
    return recorder


def test_session_journal_is_incremental_and_manifest_validated(tmp_path):
    recorder = _recorder(tmp_path)
    recorder.log_bpod_trial(1, [{"type": "StateStart", "state": "ChoiceWindow"}])
    recorder.add_trial(_trial())

    rows, repaired = load_session_journal(recorder.journal_path)
    kinds = [row["kind"] for row in rows]

    assert not repaired
    assert kinds[0] == "session_header"
    assert "trial" in kinds
    assert "bpod_trial" in kinds
    assert recorder.incomplete_marker_path.exists()

    recorder.finalize(_summary(recorder))
    bundle = session_bundle_dir(recorder.base_path, layout="flat")
    result = validate_session(bundle)

    assert result["valid"]
    assert not recorder.incomplete_marker_path.exists()
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    artifact_names = {
        str(item["path"]).replace("\\", "/").split("/")[-1]
        for item in manifest["artifacts"]
    }
    assert recorder.journal_path.name in artifact_names


def test_recover_session_repairs_truncated_tail_and_finalizes(tmp_path):
    recorder = _recorder(tmp_path)
    recorder.log_bpod_trial(1, [{"type": "StateStart", "state": "ChoiceWindow"}])
    for index in range(1, 4):
        recorder.log_streamdio_event(
            "timing_heartbeat",
            None,
            None,
            index * 100_000,
            index * 0.1,
            host_perf_counter_s=100.0 + index * 0.1,
        )
    recorder.add_trial(_trial())
    recorder.close_journal()
    with recorder.journal_path.open("a", encoding="utf-8") as handle:
        handle.write('{"schema_version":1')

    result = recover_session_journal(recorder.incomplete_marker_path)

    assert result["repaired_trailing_record"]
    assert result["recovered_trials"] == 1
    assert result["recovered_bpod_trials"] == 1
    assert result["validation"]["valid"]
    assert not recorder.incomplete_marker_path.exists()
    assert recorder.journal_path.with_name(
        f"{recorder.journal_path.name}.recovery-backup"
    ).exists()
    sidecar = json.loads(recorder.base_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert sidecar["summary"]["end_reason"] == "recovered_after_crash"
    assert sidecar["summary"]["n_trials_completed"] == 1
    assert evaluate_session_qc(result["bundle_dir"], write=False)["overall_status"] == "PASS"


def test_recover_session_rebuilds_incomplete_stream_chunks_from_wal(tmp_path):
    config = SessionConfig(subject_id="S1", n_trials=1, results_dir=str(tmp_path))
    config.session_export.layout = "flat"
    config.session_export.streamdio_chunk_s = 0.01
    config.session_export.streamdio_memory_mode = "bounded"
    recorder = SessionRecorder(tmp_path, "S1", "decision_making")
    recorder.set_export_options(config_to_dict(config)["session_export"])
    recorder.set_metadata({"config": config_to_dict(config)})
    recorder.log_streamdio_event("analog_input", 13, 1, 1000, 0.000)
    recorder.log_streamdio_event("camera", None, None, 2000, 0.020)
    recorder.log_streamdio_event("analog_input", 13, 2, 3000, 0.030)

    chunk_dir = session_bundle_dir(recorder.base_path, layout="flat") / "continuous_streams"
    deadline = time.perf_counter() + 1.0
    while not list(chunk_dir.glob("streamdio_chunk_*.npz")) and time.perf_counter() < deadline:
        time.sleep(0.005)
    assert list(chunk_dir.glob("streamdio_chunk_*.npz"))

    assert recorder._streamdio_chunk_queue is not None
    recorder._streamdio_chunk_queue.put((None, None))
    assert recorder._streamdio_chunk_thread is not None
    recorder._streamdio_chunk_thread.join(timeout=1.0)
    recorder._streamdio_chunk_thread = None
    recorder._close_streamdio_wal(durable=True)
    recorder.close_journal()

    result = recover_session_journal(recorder.incomplete_marker_path)

    assert result["recovered_streamdio_events"] == 3
    assert result["validation"]["valid"]
    manifest = json.loads(
        (Path(result["bundle_dir"]) / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["continuous_streams"]["event_count"] == 3
    assert manifest["counts"]["streamdio_events"] == 3


def test_recovery_uses_longer_chunks_when_wal_tail_is_truncated(tmp_path):
    config = SessionConfig(subject_id="S1", n_trials=1, results_dir=str(tmp_path))
    config.session_export.layout = "flat"
    config.session_export.streamdio_chunk_s = 0.001
    config.session_export.streamdio_memory_mode = "bounded"
    recorder = SessionRecorder(tmp_path, "S1", "decision_making")
    recorder.set_export_options(config_to_dict(config)["session_export"])
    recorder.set_metadata({"config": config_to_dict(config)})
    for index in range(3):
        recorder.log_streamdio_event(
            "analog_input", 13, index, 1000 + index, index * 0.002
        )
    recorder._finalize_streamdio_chunks()
    recorder._close_streamdio_wal(durable=True)
    recorder.close_journal()

    wal = recorder.streamdio_wal_path.read_bytes()
    record_size = recorder_module._STREAMDIO_WAL_RECORD.size
    recorder.streamdio_wal_path.write_bytes(wal[: -record_size // 2])
    wal_rows, trailing = load_streamdio_wal(recorder.streamdio_wal_path)
    assert len(wal_rows) == 2
    assert trailing > 0

    result = recover_session_journal(recorder.incomplete_marker_path)

    assert result["recovered_streamdio_events"] == 3
    assert result["streamdio_recovery_source"] == "chunks_extend_wal"
    assert result["validation"]["valid"]


def test_session_journal_sequences_concurrent_event_writers(tmp_path):
    recorder = _recorder(tmp_path)
    barrier = threading.Barrier(5)

    def write_events(worker):
        barrier.wait()
        for index in range(50):
            recorder.log_event(
                "background_status",
                {"worker": worker, "index": index},
            )

    threads = [threading.Thread(target=write_events, args=(worker,)) for worker in range(4)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()
    recorder.close_journal()

    rows, repaired = load_session_journal(recorder.journal_path)

    assert not repaired
    assert [row["sequence"] for row in rows] == list(range(len(rows)))
    assert sum(row["kind"] == "event" for row in rows) == 202


def test_recover_session_refuses_to_rewrite_completed_session(tmp_path):
    recorder = _recorder(tmp_path)
    recorder.add_trial(_trial())
    recorder.finalize(_summary(recorder))

    try:
        recover_session_journal(recorder.journal_path)
    except RuntimeError as exc:
        assert "already finalized" in str(exc)
    else:
        raise AssertionError("completed session was accepted for crash recovery")


def test_recover_session_refuses_completed_session_without_manifest(tmp_path):
    recorder = _recorder(tmp_path)
    recorder.export_options.update(
        {
            "manifest": False,
            "config_snapshot": False,
            "export_csv_on_finalize": False,
            "export_timeline": False,
        }
    )
    recorder.add_trial(_trial())
    recorder.finalize(_summary(recorder))

    assert not recorder.incomplete_marker_path.exists()
    assert not (session_bundle_dir(recorder.base_path, layout="flat") / "manifest.json").exists()
    try:
        recover_session_journal(recorder.journal_path)
    except RuntimeError as exc:
        assert "already contains a final summary" in str(exc)
    else:
        raise AssertionError("completed manifest-free session was accepted for crash recovery")


def test_hard_process_exit_preserves_recoverable_completed_trial(tmp_path):
    script = r'''
import os
import sys
from pathlib import Path
from hf_dm.config import SessionConfig, config_to_dict
from hf_dm.data.recorder import SessionRecorder, TrialRecord

root = Path(sys.argv[1])
config = SessionConfig(subject_id="S1", n_trials=1, results_dir=str(root))
config.session_export.layout = "flat"
recorder = SessionRecorder(root, "S1", "decision_making", base_name="crashed")
recorder.set_export_options(config_to_dict(config)["session_export"])
recorder.set_metadata({"config": config_to_dict(config)})
recorder.log_bpod_trial(1, [{"type": "StateStart", "state": "ChoiceWindow"}])
recorder.log_streamdio_event("analog_input", 13, 1234, 1000, 0.001)
recorder.log_streamdio_event("camera", None, None, 2000, 0.002)
recorder.add_trial(TrialRecord(
    trial_index=1, phase="decision_making", force_direction="cw", choice="cw",
    matched=True, reward_duration_s=0.08, responded=True, response_time_s=0.2,
    net_pulses=3, cw_pulses=3, ccw_pulses=0, final_position_deg=35.0,
    force_samples=2, force_last=100.0, force_mean=90.0, force_min=80.0,
    force_max=100.0, trial_start_wall_s=10.0, trial_start_mono_s=20.0,
))
os._exit(23)
'''
    process = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        check=False,
    )

    assert process.returncode == 23
    marker = tmp_path / "crashed.incomplete.json"
    assert marker.exists()

    result = recover_session_journal(marker)

    assert result["recovered_trials"] == 1
    assert result["recovered_bpod_trials"] == 1
    assert result["recovered_streamdio_events"] == 2
    assert result["validation"]["valid"]
    assert not marker.exists()
    assert not (tmp_path / "crashed.streamdio.wal").exists()


def test_streamdio_wal_loader_ignores_only_a_truncated_final_record(tmp_path):
    recorder = _recorder(tmp_path)
    recorder.log_streamdio_event("analog_input", 13, 42, 1000, 0.001)
    recorder._close_streamdio_wal()
    with recorder.streamdio_wal_path.open("ab") as handle:
        handle.write(b"partial")

    rows, trailing_bytes = load_streamdio_wal(recorder.streamdio_wal_path)

    assert rows == [("analog_input", 13, 42, 1000, 0.001, rows[0][5])]
    assert trailing_bytes == len(b"partial")


def test_streamdio_wal_loader_remains_compatible_with_v1_records(tmp_path):
    path = tmp_path / "legacy.streamdio.wal"
    payload = struct.pack(
        "<24siiQdd",
        b"camera".ljust(24, b"\0"),
        -1,
        -1,
        1234,
        0.001234,
        42.0,
    )
    path.write_bytes(b"HFDMSTR1\n" + payload)

    rows, trailing_bytes = load_streamdio_wal(path)

    assert rows == [("camera", -1, -1, 1234, 0.001234, 42.0)]
    assert trailing_bytes == 0


def test_streamdio_wal_crc_rejects_corrupt_complete_tail_record(tmp_path):
    recorder = _recorder(tmp_path)
    recorder.log_streamdio_event("analog_input", 13, 1, 1000, 0.001)
    recorder.log_streamdio_event("analog_input", 13, 2, 2000, 0.002)
    recorder._close_streamdio_wal()
    with recorder.streamdio_wal_path.open("r+b") as handle:
        handle.seek(-8, 2)
        value = handle.read(1)
        handle.seek(-1, 1)
        handle.write(bytes([value[0] ^ 0xFF]))

    rows, trailing_bytes = load_streamdio_wal(recorder.streamdio_wal_path)

    assert len(rows) == 1
    assert rows[0][2] == 1
    assert trailing_bytes > 0


def test_streamdio_wal_is_periodically_fsynced_during_long_trial(tmp_path):
    config = SessionConfig(subject_id="S1", n_trials=1, results_dir=str(tmp_path))
    config.session_export.streamdio_wal_fsync_s = 0.01
    recorder = SessionRecorder(tmp_path, "S1", "decision_making")
    recorder.set_export_options(config_to_dict(config)["session_export"])
    recorder.log_streamdio_event("analog_input", 13, 1, 1000, 0.001)
    deadline = time.perf_counter() + 0.5
    while recorder._streamdio_wal_fsync_count == 0 and time.perf_counter() < deadline:
        time.sleep(0.005)

    recorder._close_streamdio_wal()

    assert recorder._streamdio_wal_fsync_count >= 1


def test_streamdio_wal_paths_survive_process_cwd_changes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    recorder = SessionRecorder(Path("relative-output"), "S1", "decision_making")
    recorder.log_streamdio_event("analog_input", 13, 1, 1000, 0.001)

    monkeypatch.chdir(tmp_path.parent)
    assert recorder._sync_streamdio_wal() is True
    recorder.log_streamdio_event("camera", None, None, 2000, 0.002)
    recorder._close_streamdio_wal(durable=True)

    rows, trailing_bytes = load_streamdio_wal(recorder.streamdio_wal_path)
    assert recorder.streamdio_wal_path.is_absolute()
    assert [row[0] for row in rows] == ["analog_input", "camera"]
    assert trailing_bytes == 0


def test_streamdio_wal_failure_is_latched_without_callback_error_storm(tmp_path):
    recorder = SessionRecorder(tmp_path, "S1", "decision_making")
    recorder._streamdio_wal_background_error = "OSError: disk full"

    for index in range(100):
        recorder.log_streamdio_event("analog_input", 13, index, index, index / 1000)

    assert recorder._streamdio_events_rejected_after_wal_error == 100
    assert recorder._streamdio_total_count == 0
    with pytest.raises(RuntimeError, match="disk full"):
        recorder.raise_if_background_error()


def test_streamdio_wal_background_disk_failure_is_prominent(tmp_path, monkeypatch):
    config = SessionConfig(subject_id="S1", n_trials=1, results_dir=str(tmp_path))
    config.session_export.streamdio_wal_fsync_s = 0.01
    recorder = SessionRecorder(tmp_path, "S1", "decision_making")
    recorder.set_export_options(config_to_dict(config)["session_export"])
    recorder.log_streamdio_event("analog_input", 13, 1, 1000, 0.001)

    def disk_full():
        raise OSError(28, "disk full")

    monkeypatch.setattr(recorder, "_sync_streamdio_wal", disk_full)
    deadline = time.perf_counter() + 0.5
    while recorder._streamdio_wal_background_error is None and time.perf_counter() < deadline:
        time.sleep(0.005)

    with pytest.raises(RuntimeError, match="disk full"):
        recorder.raise_if_background_error()
    recorder._close_streamdio_wal()
