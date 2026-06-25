import json
import os

import numpy as np

from hf_dm.config import SessionConfig, TrainingPhase
from hf_dm.hardware.bringup_constants import FRAME2TTL_INPUT_CHANNEL
from hf_dm.session.simulated import (
    SimulatedHFDecisionSession,
    SimulationOptions,
    shorten_for_simulation,
)


def test_simulated_session_runs_choice_timeout_and_cleanup(tmp_path):
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    config = SessionConfig(n_trials=3, results_dir=str(tmp_path))
    config.camera.save_dir = str(tmp_path / "videos")
    shorten_for_simulation(config)
    session = SimulatedHFDecisionSession(
        config,
        SimulationOptions(choice_mode="alternate", frames_to_choice=4),
    )

    session.connect()
    summary = session.run()

    choices = [trial.choice for trial in session.recorder.trials]
    rewards = [trial.reward_duration_s for trial in session.recorder.trials]
    event_types = [event.event_type for event in session.recorder.events]
    assert summary.n_trials_completed == 3
    assert choices[0] in ("cw", "ccw")
    assert choices[1] in ("cw", "ccw")
    assert choices[2] == "timeout"
    assert rewards[0] > 0
    assert rewards[1] > 0
    assert rewards[2] == 0
    assert session.recorder.trials[0].chosen_shape == "star"
    assert session.recorder.trials[1].chosen_shape == "circle"
    assert session.recorder.trials[2].chosen_shape is None
    assert "pump_start" in event_types
    assert "white_noise_start" in event_types
    assert "video_recording_stopped" in event_types
    assert "streamdio_camera_triggers_stopped" in event_types
    assert event_types.count("lick") == 3
    assert len(session.recorder.streamdio_events) > 0
    assert all(trial.lick_count >= 1 for trial in session.recorder.trials)
    assert all(trial.force_samples > 0 for trial in session.recorder.trials)
    assert session.recorder.trials[0].frame2ttl_choice_edge_count > 0
    assert session.recorder.trials[1].frame2ttl_choice_edge_count > 0
    assert session.recorder.trials[2].choice_loop_frames == 1
    assert session.recorder.trials[2].frame2ttl_choice_edge_count <= 2
    assert any(
        event[0] == "dio_input" and int(event[1]) == FRAME2TTL_INPUT_CHANNEL
        for event in session.recorder.streamdio_events
    )
    assert any(event[0] == "bpod_state" for event in session.recorder.streamdio_events)
    assert "bpod_state_marker" in event_types
    assert len(session.recorder.bpod_trials) == 3
    with np.load(session.recorder.base_path.with_suffix(".npz"), allow_pickle=True) as data:
        assert "trials_json" in data.files
        assert "events_json" in data.files
        first_trial = json.loads(str(data["trials_json"][0]))
        assert "lick_timestamps_us" in first_trial
        assert "camera_ttl_timestamps_us" in first_trial
    timeout_states = [
        row["state"]
        for row in session.recorder.bpod_trials[2]["events"]
        if row["type"] == "StateStart"
    ]
    assert timeout_states == [
        "Quiescence_Pre",
        "Quiescence",
        "ChoiceWindow_Pre",
        "ChoiceWindow",
        "TimeoutNoise_Pre",
        "TimeoutNoise",
        "PostTimeout_Pre",
        "PostTimeout",
    ]


def test_simulated_centering_requires_wheel_motion_and_rewards_at_center(tmp_path):
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    config = SessionConfig(
        phase=TrainingPhase.CENTERING,
        n_trials=1,
        random_seed=1,
        results_dir=str(tmp_path),
    )
    config.camera.save_dir = str(tmp_path / "videos")
    shorten_for_simulation(config)
    session = SimulatedHFDecisionSession(config, SimulationOptions(choice_mode="matched"))

    session.connect()
    summary = session.run()

    trial = session.recorder.trials[0]
    event_types = [event.event_type for event in session.recorder.events]
    assert summary.n_trials_completed == 1
    assert trial.choice == "centering"
    assert trial.responded
    assert trial.reward_duration_s == config.reward.matched_pump_s
    assert abs(trial.final_position_deg) <= config.centering_threshold_deg
    assert abs(trial.net_pulses) > 0
    assert "pump_start" in event_types


def test_simulated_frame2ttl_barcode_decodes_when_quiescence_allows_full_code(tmp_path):
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    config = SessionConfig(n_trials=1, results_dir=str(tmp_path))
    config.camera.save_dir = str(tmp_path / "videos")
    shorten_for_simulation(config)
    config.timing.quiescence_s = 3.0
    session = SimulatedHFDecisionSession(
        config,
        SimulationOptions(choice_mode="matched", frames_to_choice=4),
    )

    session.connect()
    summary = session.run()

    trial = session.recorder.trials[0]
    assert summary.n_trials_completed == 1
    assert trial.frame2ttl_barcode_complete
    assert trial.frame2ttl_barcode_decoded
    assert trial.frame2ttl_barcode_match
    assert trial.frame2ttl_first_ttl_timestamp_us is not None
    assert trial.frame2ttl_first_ttl_latency_from_bpod_s is not None


def test_simulated_frame2ttl_diagnostic_counts_sync_square_edges(tmp_path):
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    config = SessionConfig(n_trials=1, results_dir=str(tmp_path))
    config.camera.save_dir = str(tmp_path / "videos")
    shorten_for_simulation(config)
    session = SimulatedHFDecisionSession(
        config,
        SimulationOptions(choice_mode="matched", frames_to_choice=4),
    )

    session.connect()
    summary = session.frame2ttl_diagnostic(n_flips=4, hold_s=0.01, settle_s=0.0)

    assert summary.end_reason == "frame2ttl_diagnostic"
    diagnostic_end = [
        event
        for event in session.recorder.events
        if event.event_type == "frame2ttl_diagnostic_end"
    ][-1]
    assert diagnostic_end.payload["detected"]
    assert diagnostic_end.payload["edges_seen"] >= 4
    assert diagnostic_end.payload["states_seen"] == [0, 1]


def test_simulated_frame2ttl_barcode_diagnostic_decodes_all_cases(tmp_path):
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    config = SessionConfig(n_trials=1, results_dir=str(tmp_path))
    config.camera.save_dir = str(tmp_path / "videos")
    shorten_for_simulation(config)
    config.display.fps = 120
    session = SimulatedHFDecisionSession(
        config,
        SimulationOptions(choice_mode="matched", frames_to_choice=4),
    )

    session.connect()
    summary = session.frame2ttl_barcode_diagnostic(repetitions=1, settle_s=0.0)

    assert summary.end_reason == "frame2ttl_barcode_diagnostic"
    diagnostic_end = [
        event
        for event in session.recorder.events
        if event.event_type == "frame2ttl_barcode_diagnostic_end"
    ][-1]
    assert diagnostic_end.payload["cases_run"] == 6
    assert diagnostic_end.payload["decoded_count"] == 6
    assert diagnostic_end.payload["match_count"] == 6
    assert diagnostic_end.payload["all_match"]


def test_simulated_bpod_opcode_diagnostic_reports_marker_sequence(tmp_path):
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    config = SessionConfig(n_trials=1, results_dir=str(tmp_path))
    config.camera.save_dir = str(tmp_path / "videos")
    shorten_for_simulation(config)
    session = SimulatedHFDecisionSession(
        config,
        SimulationOptions(choice_mode="matched", frames_to_choice=4),
    )

    session.connect()
    summary = session.bpod_opcode_diagnostic(hold_s=0.001)

    assert summary.end_reason == "bpod_opcode_diagnostic"
    diagnostic_end = [
        event
        for event in session.recorder.events
        if event.event_type == "bpod_opcode_diagnostic_end"
    ][-1]
    assert diagnostic_end.payload["matched"]
    assert diagnostic_end.payload["seen_ids"] == diagnostic_end.payload["expected_ids"]
    assert diagnostic_end.payload["markers_seen"] == len(diagnostic_end.payload["expected_ids"])


def test_simulated_forced_risk_star_can_omit_pump_and_records_metadata(tmp_path):
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    config = SessionConfig(n_trials=1, random_seed=1, results_dir=str(tmp_path))
    config.camera.save_dir = str(tmp_path / "videos")
    config.task.modality = "risk"
    config.task.n_free_trials = 0
    config.task.n_forced_trials = 1
    config.task.risk_star_probability = 0.0
    shorten_for_simulation(config)
    session = SimulatedHFDecisionSession(
        config,
        SimulationOptions(choice_mode="matched", frames_to_choice=4),
    )

    session.connect()
    summary = session.run()

    trial = session.recorder.trials[0]
    event_types = [event.event_type for event in session.recorder.events]
    assert summary.n_trials_completed == 1
    assert trial.trial_mode == "forced"
    assert trial.forced_shape == "star"
    assert trial.modality == "risk"
    assert trial.chosen_shape == "star"
    assert trial.reward_duration_s == 0.0
    assert not trial.reward_delivered
    assert trial.risk_star_probability == 0.0
    assert "pump_omitted" in event_types
    assert "pump_start" not in event_types
    assert "force_on" not in event_types
    assert "force_skipped" in event_types
    assert trial.force_direction is None
    assert trial.force_output_direction is None
    assert all(op[0] != "force" for op in session.relay.ops)
    assert session.recorder.metadata["task_summary"]["symbols"]["star"]["effect"] == "risk"
    assert session.recorder.metadata["task_summary"]["symbols"]["star"]["reward_probability"] == 0.0


def test_simulated_force_modality_does_not_force_forced_circle(tmp_path):
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    config = SessionConfig(n_trials=2, random_seed=1, results_dir=str(tmp_path))
    config.camera.save_dir = str(tmp_path / "videos")
    config.task.modality = "force"
    config.task.n_free_trials = 0
    config.task.n_forced_trials = 2
    shorten_for_simulation(config)
    session = SimulatedHFDecisionSession(
        config,
        SimulationOptions(choice_mode="matched", frames_to_choice=4),
    )

    session.connect()
    summary = session.run()

    assert summary.n_trials_completed == 2
    by_shape = {trial.forced_shape: trial for trial in session.recorder.trials}
    star_trial = by_shape["star"]
    circle_trial = by_shape["circle"]
    assert star_trial.force_output_direction in ("cw", "ccw")
    assert star_trial.force_direction == star_trial.star_direction
    assert circle_trial.chosen_shape == "circle"
    assert circle_trial.force_direction is None
    assert circle_trial.force_output_direction is None
    assert circle_trial.choice_target_direction == circle_trial.circle_direction

    force_on_trials = [
        event.payload["trial_index"]
        for event in session.recorder.events
        if event.event_type == "force_on"
    ]
    skipped_trials = [
        event.payload["trial_index"]
        for event in session.recorder.events
        if event.event_type == "force_skipped"
    ]
    assert force_on_trials == [star_trial.trial_index]
    assert skipped_trials == [circle_trial.trial_index]
