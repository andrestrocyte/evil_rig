import queue

from hf_dm.config import MonitorConfig, SessionConfig
from hf_dm.hardware.bringup_constants import (
    CAP_TOUCH_INPUT_PIN,
    CAP_TOUCH_POWER_OUTPUT_CHANNEL,
    CAP_TOUCH_POWER_PIN,
    FORCE_ANALOG_CHANNEL,
    LICK_INPUT_CHANNEL,
    PUMP_OUTPUT_CHANNEL,
)
from hf_dm.monitor import OperatorMonitor, OperatorMonitorState
from hf_dm.monitor.gui import _frame_age_text, _frame_to_netpbm, _mono_frame_to_rgb_netpbm
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


class FakeBoundedQueue:
    def __init__(self, maxsize: int = 1) -> None:
        self.maxsize = maxsize
        self.items = []

    def put_nowait(self, item: dict) -> None:
        if len(self.items) >= self.maxsize:
            raise queue.Full
        self.items.append(item)

    def get_nowait(self) -> dict:
        if not self.items:
            raise queue.Empty
        return self.items.pop(0)


def test_operator_monitor_state_tracks_trials_events_and_recent_frames():
    state = OperatorMonitorState(history_s=10.0, trial_history=3)

    state.apply(
        {
            "type": "session_info",
            "payload": {
                "subject_id": "M1",
                "n_trials": 50,
                "lick_inputs": [
                    {
                        "channel": LICK_INPUT_CHANNEL,
                        "pin": CAP_TOUCH_INPUT_PIN,
                        "label": "cap touch",
                    }
                ],
                "cap_touch_power": {
                    "channel": CAP_TOUCH_POWER_OUTPUT_CHANNEL,
                    "pin": CAP_TOUCH_POWER_PIN,
                },
            },
        }
    )
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
    state.apply(
        {
            "type": "cap_touch_sample",
            "payload": {"channel": LICK_INPUT_CHANNEL, "pin": CAP_TOUCH_INPUT_PIN, "value": 740},
        }
    )
    state.apply({"type": "lick", "payload": {"channel": LICK_INPUT_CHANNEL}})
    state.apply(
        {
            "type": "cap_touch_power",
            "payload": {
                "active": True,
                "channel": CAP_TOUCH_POWER_OUTPUT_CHANNEL,
                "pin": CAP_TOUCH_POWER_PIN,
            },
        }
    )
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
    assert state.lick_inputs[0]["pin"] == CAP_TOUCH_INPUT_PIN
    assert state.lick_input_states[LICK_INPUT_CHANNEL]["value"] == 740
    assert state.lick_input_states[LICK_INPUT_CHANNEL]["source"] == "analog"
    assert state.cap_touch_power_active
    assert state.cap_touch_power["pin"] == CAP_TOUCH_POWER_PIN
    assert state.last_lick_value == 740
    assert state.lick_count == 1
    assert state.pump_active
    assert state.noise_active
    assert state.latest_camera_frame["frame_index"] == 2
    assert state.camera_frame_count == 1
    assert state.last_camera_frame_mono_s is not None
    assert state.outcomes[-1]["chosen_shape"] == "star"


def test_operator_monitor_state_tracks_latest_camera_and_task_frame_counts():
    state = OperatorMonitorState(history_s=10.0, trial_history=3)

    state.apply(
        {
            "type": "camera_frame",
            "payload": {"frame_index": 1, "width": 1, "height": 1, "channels": 1, "data": b"a"},
        }
    )
    state.apply(
        {
            "type": "camera_frame",
            "payload": {"frame_index": 2, "width": 1, "height": 1, "channels": 1, "data": b"b"},
        }
    )
    state.apply(
        {
            "type": "task_frame",
            "payload": {"frame_index": 5, "width": 1, "height": 1, "channels": 3, "data": b"abc"},
        }
    )

    assert state.camera_frame_count == 2
    assert state.task_frame_count == 1
    assert state.latest_camera_frame["frame_index"] == 2
    assert state.latest_task_frame["frame_index"] == 5
    assert state.last_camera_frame_mono_s is not None
    assert state.last_task_frame_mono_s is not None


def test_cap_touch_status_sample_does_not_create_lick_edge_indicator():
    state = OperatorMonitorState(history_s=10.0, trial_history=3)

    state.apply(
        {
            "type": "cap_touch_sample",
            "payload": {
                "channel": LICK_INPUT_CHANNEL,
                "pin": CAP_TOUCH_INPUT_PIN,
                "value": 900,
                "active": True,
                "baseline": 20,
                "rising_high_threshold": 700,
                "source": "firmware_digitizer_status",
            },
        }
    )

    touch_state = state.lick_input_states[LICK_INPUT_CHANNEL]
    assert state.lick_edge_count == 0
    assert state.lick_count == 0
    assert state.last_lick_edge_mono_s is None
    assert state.last_touch_sample_mono_s is not None
    assert state.last_lick_value == 900
    assert state.last_lick_source == "status"
    assert touch_state["source"] == "status"
    assert touch_state["active"] is True
    assert touch_state["baseline"] == 20
    assert touch_state["rising_high_threshold"] == 700
    assert touch_state["last_sample_mono_s"] == state.last_touch_sample_mono_s


def test_operator_monitor_replaces_stale_frames_per_source_queue():
    monitor = OperatorMonitor(MonitorConfig(enabled=True))
    camera_queue = FakeBoundedQueue(maxsize=1)
    task_queue = FakeBoundedQueue(maxsize=1)
    monitor._camera_frame_queue = camera_queue
    monitor._task_frame_queue = task_queue

    assert monitor.publish("camera_frame", {"frame_index": 1})
    assert monitor.publish("camera_frame", {"frame_index": 2})
    assert monitor.publish("task_frame", {"frame_index": 10})

    assert camera_queue.items[0]["payload"]["frame_index"] == 2
    assert task_queue.items[0]["payload"]["frame_index"] == 10
    assert monitor.published_counts["camera_frame"] == 2
    assert monitor.published_counts["task_frame"] == 1
    assert monitor.dropped_frames == 1


def test_streamdio_callback_publishes_monitor_signals():
    session = HFDecisionSession(SessionConfig())
    monitor = FakeMonitor()
    session.monitor = monitor

    session._handle_streamdio_event("analog_input", FORCE_ANALOG_CHANNEL, 321, 1000, 0.001)
    session._handle_streamdio_event("dio_output", PUMP_OUTPUT_CHANNEL, 1, 2000, 0.002)
    session._handle_streamdio_event(
        "dio_output",
        CAP_TOUCH_POWER_OUTPUT_CHANNEL,
        1,
        2200,
        0.0022,
    )
    session._handle_streamdio_event("analog_input", LICK_INPUT_CHANNEL, 700, 2400, 0.0024)
    session._handle_streamdio_event("dio_input", LICK_INPUT_CHANNEL, 0, 2500, 0.0025)
    session._handle_streamdio_event("dio_input", LICK_INPUT_CHANNEL, 1, 3000, 0.003)
    session._handle_streamdio_event("camera", None, None, 4000, 0.004)

    event_types = [event_type for event_type, _payload in monitor.messages]
    assert "force_sample" in event_types
    assert "pump" in event_types
    assert "cap_touch_power" in event_types
    assert "cap_touch_sample" in event_types
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


def test_mono_monitor_frames_can_fall_back_to_rgb_netpbm_bytes():
    rgb = _mono_frame_to_rgb_netpbm(
        {"width": 2, "height": 1, "channels": 1, "data": bytes([10, 200])}
    )

    assert rgb == b"P6\n2 1\n255\n" + bytes([10, 10, 10, 200, 200, 200])


def test_frame_age_text_formats_recent_and_stale_frames():
    assert _frame_age_text(None) == "age -"
    assert _frame_age_text(9.75, now=10.0) == "age 250 ms"
    assert _frame_age_text(8.0, now=10.0) == "age 2.0 s"
