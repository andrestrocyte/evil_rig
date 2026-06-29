import json
import subprocess
import sys
import threading

from hf_dm.config import SessionConfig, config_to_dict
from hf_dm.data.recorder import (
    SessionRecorder,
    SessionSummary,
    TrialRecord,
    load_session_journal,
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
    assert result["validation"]["valid"]
    assert not marker.exists()
