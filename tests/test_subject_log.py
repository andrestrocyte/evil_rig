import json
import sys

from hf_dm.config import SessionConfig, config_to_dict
from hf_dm.data.recorder import (
    SessionRecorder,
    SessionSummary,
    TrialRecord,
    recover_session_journal,
)
from hf_dm.subject_log import (
    append_session_water_entry,
    append_weight_entry,
    load_subject_log,
    subject_log_path,
)
from hf_dm.ops_cli import main
from hf_dm.session_qc import evaluate_session_qc


def _trial(index, volume_ul):
    return TrialRecord(
        trial_index=index,
        phase="decision_making",
        force_direction="cw",
        choice="cw",
        matched=True,
        reward_duration_s=0.08,
        responded=True,
        response_time_s=1.0,
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
        reward_volume_ul=volume_ul,
    )


def test_weight_log_appends_valid_jsonl_without_rewriting_prior_rows(tmp_path):
    first = append_weight_entry(
        tmp_path / "subjects",
        "mouse01",
        weight_g=24.3,
        water_target_ml=1.2,
        entry_date="2026-06-29",
    )
    first_bytes = subject_log_path(tmp_path / "subjects", "mouse01").read_bytes()
    second = append_weight_entry(
        tmp_path / "subjects",
        "mouse01",
        weight_g=24.1,
        water_target_ml=1.2,
        entry_date="2026-06-30",
    )

    rows = load_subject_log(tmp_path / "subjects", "mouse01")

    assert first["appended"] and second["appended"]
    assert len(rows) == 2
    assert rows[0]["weight_g"] == 24.3
    assert rows[1]["weight_g"] == 24.1
    assert subject_log_path(tmp_path / "subjects", "mouse01").read_bytes().startswith(
        first_bytes
    )


def test_session_water_entry_is_idempotent_for_same_session(tmp_path):
    first = append_session_water_entry(
        tmp_path / "subjects",
        "mouse01",
        water_delivered_ul=120.0,
        session_id=str(tmp_path / "session"),
    )
    second = append_session_water_entry(
        tmp_path / "subjects",
        "mouse01",
        water_delivered_ul=120.0,
        session_id=str(tmp_path / "session"),
    )

    rows = load_subject_log(tmp_path / "subjects", "mouse01")

    assert first["appended"]
    assert not second["appended"]
    assert len(rows) == 1
    assert rows[0]["water_delivered_ml"] == 0.12


def test_session_finalize_appends_total_calibrated_water(tmp_path):
    subjects_dir = tmp_path / "subjects"
    config = SessionConfig(subject_id="mouse01", n_trials=2, results_dir=str(tmp_path))
    config.welfare.subjects_dir = str(subjects_dir)
    recorder = SessionRecorder(tmp_path, "mouse01", "decision_making")
    recorder.set_export_options(config_to_dict(config)["session_export"])
    recorder.set_metadata(
        {
            "config": config_to_dict(config),
            "reward_calibration": {"sha256": "abc"},
        }
    )
    recorder.add_trial(_trial(1, 4.0))
    recorder.add_trial(_trial(2, 2.0))

    recorder.finalize(
        SessionSummary(
            subject_id="mouse01",
            phase="decision_making",
            n_trials_planned=2,
            n_trials_completed=2,
            end_reason="completed",
            consecutive_failures=0,
            session_start_wall_s=recorder._session_start_wall,
            session_end_wall_s=recorder._session_start_wall + 1.0,
        )
    )

    rows = load_subject_log(subjects_dir, "mouse01")
    assert len(rows) == 1
    assert rows[0]["entry_type"] == "session_water"
    assert rows[0]["water_delivered_ml"] == 0.006
    event = next(
        item for item in recorder.events if item.event_type == "subject_water_logged"
    )
    assert event.payload["water_delivered_ul"] == 6.0
    assert event.payload["calibrated_trial_count"] == 2
    qc = evaluate_session_qc(recorder.base_path.with_suffix(".json"), write=False)
    water_metric = next(
        metric for metric in qc["metrics"] if metric["name"] == "subject_water_log"
    )
    assert water_metric["status"] == "PASS"


def test_log_weight_cli_appends_entry(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hf-dm",
            "log-weight",
            "mouse01",
            "--weight",
            "24.3",
            "--subjects-dir",
            str(tmp_path / "subjects"),
        ],
    )

    main()

    result = json.loads(capsys.readouterr().out)
    assert result["appended"]
    assert load_subject_log(tmp_path / "subjects", "mouse01")[0]["weight_g"] == 24.3


def test_recovered_calibrated_session_appends_water_once(tmp_path):
    subjects_dir = tmp_path / "subjects"
    config = SessionConfig(subject_id="mouse01", n_trials=1, results_dir=str(tmp_path))
    config.welfare.subjects_dir = str(subjects_dir)
    recorder = SessionRecorder(tmp_path, "mouse01", "decision_making", base_name="crashed")
    recorder.set_export_options(config_to_dict(config)["session_export"])
    recorder.set_metadata(
        {
            "config": config_to_dict(config),
            "reward_calibration": {"sha256": "abc"},
        }
    )
    recorder.add_trial(_trial(1, 4.0))
    recorder.close_journal()

    result = recover_session_journal(recorder.incomplete_marker_path)

    rows = load_subject_log(subjects_dir, "mouse01")
    assert result["validation"]["valid"]
    assert len(rows) == 1
    assert rows[0]["water_delivered_ml"] == 0.004
