import json
import sys

from hf_dm.config import SessionConfig, config_to_dict
from hf_dm.data.recorder import SessionRecorder, SessionSummary, TrialRecord
from hf_dm.data.session_export import session_bundle_dir
from hf_dm.session_qc import QcThresholds, evaluate_session_qc
from hf_dm.ops_cli import main


def _trial(index, *, responded=True, matched=True, response_time_s=1.0, stalls=0):
    return TrialRecord(
        trial_index=index,
        phase="decision_making",
        force_direction="cw",
        choice="cw" if responded else "timeout",
        matched=matched,
        reward_duration_s=0.08 if responded else 0.0,
        responded=responded,
        response_time_s=response_time_s,
        net_pulses=3,
        cw_pulses=3,
        ccw_pulses=0,
        final_position_deg=35.0,
        force_samples=1,
        force_last=1.0,
        force_mean=1.0,
        force_min=1.0,
        force_max=1.0,
        trial_start_wall_s=float(index),
        trial_start_mono_s=float(index),
        choice_loop_max_frame_s=0.02,
        choice_loop_stall_count=stalls,
    )


def _finalized_session(tmp_path, trials):
    config = SessionConfig(subject_id="S1", n_trials=len(trials), results_dir=str(tmp_path))
    config.session_export.layout = "flat"
    recorder = SessionRecorder(tmp_path, "S1", "decision_making")
    recorder.set_export_options(config_to_dict(config)["session_export"])
    recorder.set_metadata({"config": config_to_dict(config)})
    for index in range(1, 4):
        recorder.log_streamdio_event(
            "timing_heartbeat",
            None,
            None,
            index * 100_000,
            index * 0.1,
            host_perf_counter_s=100.0 + index * 0.1,
        )
    for trial in trials:
        recorder.add_trial(trial)
    recorder.finalize(
        SessionSummary(
            subject_id="S1",
            phase="decision_making",
            n_trials_planned=len(trials),
            n_trials_completed=len(trials),
            end_reason="completed",
            consecutive_failures=0,
            session_start_wall_s=recorder._session_start_wall,
            session_end_wall_s=recorder._session_start_wall + 1.0,
        )
    )
    return recorder


def test_session_qc_writes_one_result_per_metric_for_bundle_and_npz(tmp_path):
    recorder = _finalized_session(tmp_path, [_trial(index) for index in range(1, 5)])
    bundle = session_bundle_dir(recorder.base_path, layout="flat")

    report = evaluate_session_qc(bundle)
    npz_report = evaluate_session_qc(recorder.base_path.with_suffix(".npz"), write=False)

    assert report["overall_status"] == "PASS"
    assert len(report["metrics"]) == 13
    assert {metric["name"] for metric in report["metrics"]} == {
        "responded_rate",
        "matched_rate_responded",
        "mean_response_time_s",
        "choice_loop_stall_count",
        "choice_loop_max_frame_s",
        "streamdio_heartbeat_count",
        "streamdio_estimated_missed_pulses",
        "streamdio_teensy_interval_max_abs_error_s",
        "streamdio_pc_interval_p99_abs_error_s",
        "streamdio_pc_interval_max_abs_error_s",
        "streamdio_pc_delivery_detrended_max_abs_s",
        "streamdio_teensy_nonmonotonic_intervals",
        "streamdio_pc_nonmonotonic_intervals",
    }
    output = bundle / "session.qc.json"
    assert output.exists()
    assert json.loads(output.read_text(encoding="utf-8"))["overall_status"] == "PASS"
    assert npz_report["overall_status"] == "PASS"


def test_session_qc_distinguishes_warn_and_fail(tmp_path):
    recorder = _finalized_session(
        tmp_path,
        [
            _trial(1, responded=True, matched=True, response_time_s=10.0, stalls=1),
            _trial(2, responded=False, matched=False, response_time_s=20.0),
        ],
    )

    warning = evaluate_session_qc(recorder.base_path.with_suffix(".json"), write=False)
    failing = evaluate_session_qc(
        recorder.base_path.with_suffix(".json"),
        thresholds=QcThresholds(min_responded_rate=0.80, warn_responded_rate=0.60),
        write=False,
    )

    assert warning["overall_status"] == "WARN"
    assert failing["overall_status"] == "FAIL"


def test_mean_response_time_excludes_timeouts(tmp_path):
    recorder = _finalized_session(
        tmp_path,
        [
            _trial(1, responded=True, response_time_s=2.0),
            _trial(2, responded=False, matched=False, response_time_s=20.0),
        ],
    )

    report = evaluate_session_qc(recorder.base_path.with_suffix(".json"), write=False)
    mean_rt = next(
        metric for metric in report["metrics"] if metric["name"] == "mean_response_time_s"
    )

    assert mean_rt["value"] == 2.0


def test_session_qc_cli_exits_nonzero_only_on_fail(tmp_path, monkeypatch, capsys):
    recorder = _finalized_session(
        tmp_path,
        [_trial(1, responded=False, matched=False, response_time_s=20.0, stalls=4)],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["hf-dm", "session-qc", str(recorder.base_path.with_suffix(".json"))],
    )

    try:
        main()
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("failing session QC returned a successful exit code")

    output = capsys.readouterr().out
    assert "Overall: FAIL" in output
    assert "responded_rate" in output
