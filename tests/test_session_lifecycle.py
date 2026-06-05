from unittest.mock import MagicMock

from hf_dm.config import SessionConfig
from hf_dm.data.recorder import SessionRecorder, TrialRecord
from hf_dm.hardware.bringup_constants import LICK_INPUT_CHANNEL
from hf_dm.session.controller import HFDecisionSession


class FakeRelay:
    def __init__(self) -> None:
        self.ops = []

    def task_outputs_off(self) -> dict:
        self.ops.append("task_outputs_off")
        return {"verified": True}

    def cap_touch_power(self, enabled: bool) -> dict:
        self.ops.append(("cap_touch_power", enabled))
        return {"enabled": enabled, "verified": True}

    def speaker_gate(self, enabled: bool) -> None:
        self.ops.append(("speaker_gate", enabled))


class FakeDIO:
    config_port = "COM8"
    stream_port = "COM7"

    def __init__(self) -> None:
        self.ops = []
        self.camera_events = [(1, 0.001), (2, 0.002)]

    def get_camera_events(self) -> list[tuple[int, float]]:
        return list(self.camera_events)

    def stop_camera(self) -> None:
        self.ops.append("stop_camera")

    def close(self) -> None:
        self.ops.append("close")


class FakeDisplay:
    def __init__(self) -> None:
        self.closed = False

    def show_black(self, _duration_s: float) -> None:
        return

    def close(self) -> None:
        self.closed = True


class FakeVideoRecorder:
    def stop(self) -> dict:
        return {"backend": "spinnaker", "frames_saved": 3, "capture_error": None}


class OneTrialSession(HFDecisionSession):
    def _run_bpod_dm_trial(self, trial_index: int) -> bool:
        assert self.recorder is not None
        self.recorder.add_trial(
            TrialRecord(
                trial_index=trial_index,
                phase=self.config.phase.value,
                force_direction="cw",
                choice="cw",
                matched=True,
                reward_duration_s=0.08,
                responded=True,
                response_time_s=0.01,
                net_pulses=1,
                cw_pulses=1,
                ccw_pulses=0,
                final_position_deg=1.0,
                force_samples=0,
                force_last=None,
                force_mean=None,
                force_min=None,
                force_max=None,
                trial_start_wall_s=1.0,
                trial_start_mono_s=1.0,
            )
        )
        return True


def test_streamdio_lick_events_are_recorded_on_configured_high_channel():
    session = HFDecisionSession(SessionConfig())

    session._handle_streamdio_event("dio_input", LICK_INPUT_CHANNEL, 1, 12345, 0.123)
    session._handle_streamdio_event("dio_input", LICK_INPUT_CHANNEL, 0, 12346, 0.124)
    session._handle_streamdio_event("dio_input", 1, 1, 12347, 0.125)

    assert len(session._lick_events) == 1
    assert session._lick_events[0]["channel"] == LICK_INPUT_CHANNEL
    assert session._lick_events[0]["timestamp_us"] == 12345


def test_raw_streamdio_events_accept_camera_none_fields_and_log_licks(tmp_path):
    session = HFDecisionSession(SessionConfig())
    session.recorder = SessionRecorder(tmp_path, "S1", "dm")

    session._handle_streamdio_event("camera", None, None, 12000, 0.120)
    session._handle_streamdio_event("analog_input", 13, 42, 13000, 0.130)
    session._handle_streamdio_event("dio_input", LICK_INPUT_CHANNEL, 1, 14000, 0.140)

    assert session.recorder.streamdio_events[0][:3] == ("camera", -1, -1)
    assert session.recorder.streamdio_events[1][:3] == ("analog_input", 13, 42)
    assert [event.event_type for event in session.recorder.events].count("lick") == 1


def test_run_finalizes_after_video_and_hardware_cleanup(tmp_path):
    cfg = SessionConfig(n_trials=1, results_dir=str(tmp_path))
    session = OneTrialSession(cfg)
    session.bpod = MagicMock()
    session.bpod.port = "COM5"
    session.relay = FakeRelay()
    session.dio = FakeDIO()
    session.display = FakeDisplay()
    session.recorder = SessionRecorder(tmp_path, "S1", "dm")
    session.camera_recorder = FakeVideoRecorder()

    summary = session.run()

    event_types = [event.event_type for event in session.recorder.events]
    assert summary.end_reason == "completed"
    assert event_types.index("video_recording_stopped") < event_types.index("session_end")
    assert event_types.index("task_outputs_cleanup_verified") < event_types.index(
        "video_recording_stopped"
    )
    assert event_types.index("video_recording_stopped") < event_types.index(
        "all_off_cleanup_verified"
    )
    assert event_types.index("all_off_cleanup_verified") < event_types.index("session_end")
    assert event_types.index("streamdio_camera_triggers_stopped") < event_types.index("session_end")
    assert session.display.closed
    assert session.dio.ops == ["stop_camera", "close"]
    assert session.relay.ops == ["task_outputs_off", ("cap_touch_power", False)]


def test_preflight_saves_stream_camera_and_cleanup_events(tmp_path):
    cfg = SessionConfig(n_trials=99, results_dir=str(tmp_path))
    session = HFDecisionSession(cfg)
    session.bpod = MagicMock()
    session.bpod.port = "COM5"
    session.relay = FakeRelay()
    session.dio = FakeDIO()
    session.display = FakeDisplay()
    session.recorder = SessionRecorder(tmp_path, "S1", "dm")
    session.camera_recorder = FakeVideoRecorder()

    summary = session.preflight(duration_s=0.0)

    event_types = [event.event_type for event in session.recorder.events]
    assert summary.end_reason == "preflight"
    assert summary.n_trials_planned == 0
    assert summary.n_trials_completed == 0
    assert "preflight_start" in event_types
    assert "preflight_end" in event_types
    assert event_types.index("preflight_end") < event_types.index("session_end")
    assert event_types.index("video_recording_stopped") < event_types.index("session_end")
    assert session.display.closed
