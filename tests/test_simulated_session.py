import os

from hf_dm.config import SessionConfig, TrainingPhase
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
    assert len(session.recorder.bpod_trials) == 3
    timeout_states = [
        row["state"]
        for row in session.recorder.bpod_trials[2]["events"]
        if row["type"] == "StateStart"
    ]
    assert timeout_states == ["Quiescence", "ChoiceWindow", "TimeoutNoise", "PostTimeout"]


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
    assert session.recorder.metadata["symbols"]["star_effect"] == "risk"
