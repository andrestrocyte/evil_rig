import os

from hf_dm.config import DisplayConfig, SessionConfig
from hf_dm.display.pygame_display import TaskDisplay
from hf_dm.session.controller import HFDecisionSession
from hf_dm.session.simulated import (
    SimulatedHFDecisionSession,
    SimulationOptions,
    shorten_for_simulation,
)


def test_only_stimulus_opposite_wheel_direction_moves_toward_center():
    config = SessionConfig(display=DisplayConfig(choice_distance_deg=35.0))
    session = HFDecisionSession(config)

    assert session._decision_positions(35.0) == (0.0, 35.0)
    assert session._decision_positions(-35.0) == (-35.0, 0.0)
    assert session._decision_positions(10.0) == (-25.0, 35.0)
    assert session._decision_positions(-10.0) == (-35.0, 25.0)


def test_star_and_circle_have_exactly_equal_bright_pixels():
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    display = TaskDisplay(
        DisplayConfig(
            width=480,
            height=480,
            auto_size_to_display=False,
            display_index=None,
            noframe=False,
        )
    )
    try:
        assert display.shape_bright_pixel_counts["star"] > 0
        assert display.shape_bright_pixel_counts["star"] == display.shape_bright_pixel_counts["circle"]
    finally:
        display.close()


def test_bpod_fsm_choice_blanks_then_rewards_and_exits(tmp_path):
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    config = SessionConfig(n_trials=1, results_dir=str(tmp_path))
    config.camera.save_dir = str(tmp_path / "videos")
    shorten_for_simulation(config)
    session = SimulatedHFDecisionSession(
        config,
        SimulationOptions(choice_mode="matched", frames_to_choice=4),
    )

    session.connect()
    summary = session.run()

    assert summary.n_trials_completed == 1
    trial = session.recorder.trials[0]
    assert trial.chosen_shape == "star"
    assert trial.choice_side == trial.star_side
    assert trial.matched
    state_order = [
        row["state"]
        for row in session.recorder.bpod_trials[0]["events"]
        if row["type"] == "StateStart"
    ]
    assert state_order == [
        "Quiescence",
        "ChoiceWindow",
        "ChoiceBlankHigh",
        "RewardHigh",
        "PostRewardHigh",
    ]
    event_types = [event.event_type for event in session.recorder.events]
    assert event_types.index("choice_detected") < event_types.index("stimulus_off")
    assert event_types.index("stimulus_off") < event_types.index("pump_start")
    assert event_types.index("pump_start") < event_types.index("pump_end")
