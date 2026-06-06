from hf_dm.config import SessionConfig
from hf_dm.hardware.bringup_constants import FORCE_ANALOG_CHANNEL, LICK_INPUT_CHANNEL, PUMP_OUTPUT_CHANNEL
from hf_dm.monitor import OperatorMonitorState
from hf_dm.monitor.gui import _frame_to_netpbm
from hf_dm.session.controller import HFDecisionSession


class FakeMonitor:
    def __init__(self) -> None:
        self.messages = []

    def publish(self, event_type: str, payload: dict | None = None) -> bool:
        self.messages.append((event_type, payload or {}))
        return True

    def publish_throttled(
        self,
        key: str,
        max_hz: float,
        event_type: str,
        payload: dict | None = None,
    ) -> bool:
        del key, max_hz
        return self.publish(event_type, payload)


def test_operator_monitor_state_tracks_trials_events_and_recent_frames():
    state = OperatorMonitorState(history_s=10.0, trial_history=3)

    state.apply({"type": "session_info", "payload": {"subject_id": "M1", "n_trials": 50}})
    state.apply(
        {
            "type": "trial_start",
            "payload": {
                "trial_index": 1,
                "phase": "decision_making",
                "star_side": "left",
                "circle_side": "right",
                "force_direction": "cw",
            },
        }
    )
    state.apply({"type": "force_sample", "payload": {"value": 123.0}})
    state.apply({"type": "force_sample", "payload": {"value": 20.0}})
    state.apply(
        {
            "type": "wheel_sample",
            "payload": {
                "position_deg": 7.5,
                "net_pulses": 10,
                "cw_pulses": 10,
                "ccw_pulses": 0,
            },
        }
    )
    state.apply(
        {
            "type": "fsm_states",
            "payload": {
                "states": [
                    {
                        "name": "ChoiceWindow",
                        "duration_s": 20.0,
                        "logic": "Tup -> TimeoutNoise",
                    }
                ]
            },
        }
    )
    state.apply({"type": "fsm_state", "payload": {"active_state": "ChoiceWindow", "phase": "choice"}})
    state.apply({"type": "lick_edge", "payload": {"channel": LICK_INPUT_CHANNEL, "value": 0}})
    state.apply({"type": "lick", "payload": {"channel": LICK_INPUT_CHANNEL}})
    state.apply({"type": "pump", "payload": {"active": True}})
    state.apply({"type": "white_noise", "payload": {"active": True}})
    state.apply(
        {
            "type": "camera_frame",
            "payload": {"frame_index": 2, "width": 2, "height": 2, "channels": 1, "data": b"abcd"},
        }
    )
    state.apply(
        {
            "type": "trial_result",
            "payload": {
                "trial_index": 1,
                "chosen_shape": "star",
                "matched": True,
                "responded": True,
            },
        }
    )

    assert state.session_info["subject_id"] == "M1"
    assert state.current_trial == 1
    assert state.current_star_side == "left"
    assert len(state.force_samples) == 2
    assert state.force_session_max == 123.0
    assert state.wheel_samples[-1][1:] == (7.5, 10, 10, 0)
    assert state.fsm_states[0]["name"] == "ChoiceWindow"
    assert state.active_fsm_state == "ChoiceWindow"
    assert state.lick_edge_count == 1
    assert state.last_lick_value == 0
    assert state.lick_count == 1
    assert state.pump_active
    assert state.noise_active
    assert state.latest_camera_frame["frame_index"] == 2
    assert state.outcomes[-1]["chosen_shape"] == "star"


def test_streamdio_callback_publishes_monitor_signals():
    session = HFDecisionSession(SessionConfig())
    monitor = FakeMonitor()
    session.monitor = monitor

    session._handle_streamdio_event("analog_input", FORCE_ANALOG_CHANNEL, 321, 1000, 0.001)
    session._handle_streamdio_event("dio_output", PUMP_OUTPUT_CHANNEL, 1, 2000, 0.002)
    session._handle_streamdio_event("dio_input", LICK_INPUT_CHANNEL, 0, 2500, 0.0025)
    session._handle_streamdio_event("dio_input", LICK_INPUT_CHANNEL, 1, 3000, 0.003)
    session._handle_streamdio_event("camera", None, None, 4000, 0.004)

    event_types = [event_type for event_type, _payload in monitor.messages]
    assert "force_sample" in event_types
    assert "pump" in event_types
    assert "lick_edge" in event_types
    assert "lick" in event_types
    assert "camera_ttl" in event_types
    assert event_types.count("lick") == 1


def test_monitor_frames_convert_to_tk_readable_netpbm_bytes():
    mono = _frame_to_netpbm(
        {"width": 2, "height": 2, "channels": 1, "data": bytes([0, 80, 160, 255])}
    )
    rgb = _frame_to_netpbm(
        {
            "width": 2,
            "height": 2,
            "channels": 3,
            "data": bytes(
                [
                    255,
                    0,
                    0,
                    0,
                    255,
                    0,
                    0,
                    0,
                    255,
                    255,
                    255,
                    255,
                ]
            ),
        }
    )

    assert mono is not None
    assert rgb is not None
    assert mono.startswith(b"P5\n2 2\n255\n")
    assert rgb.startswith(b"P6\n2 2\n255\n")
