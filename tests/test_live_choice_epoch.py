import os

from hf_dm.config import DisplayConfig, Frame2TTLConfig, SessionConfig
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


def test_frame2ttl_sync_square_draws_in_lower_right_corner():
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    display = TaskDisplay(
        DisplayConfig(
            width=240,
            height=180,
            auto_size_to_display=False,
            display_index=None,
            noframe=False,
        ),
        Frame2TTLConfig(
            enabled=True,
            patch_size_px=20,
            patch_margin_px=10,
            patch_corner="bottom_right",
        ),
    )
    try:
        display.draw_sync_square_frame(True)
        assert display.screen.get_at((219, 159))[:3] == (255, 255, 255)
        assert display.screen.get_at((5, 5))[:3] == (0, 0, 0)
        display.draw_sync_square_frame(False)
        assert display.screen.get_at((219, 159))[:3] == (0, 0, 0)
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
        "Quiescence_Pre",
        "Quiescence",
        "ChoiceWindow_Pre",
        "ChoiceWindow",
        "ChoiceBlankHigh_Pre",
        "ChoiceBlankHigh",
        "RewardHigh_Pre",
        "RewardHigh",
        "PostRewardHigh_Pre",
        "PostRewardHigh",
    ]
    event_types = [event.event_type for event in session.recorder.events]
    assert "bpod_state_marker" in event_types
    assert event_types.index("choice_detected") < event_types.index("stimulus_off")
    assert event_types.index("stimulus_off") < event_types.index("pump_start")
    assert event_types.index("pump_start") < event_types.index("pump_end")
