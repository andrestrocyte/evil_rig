import json
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from hf_dm.config import SessionConfig
from hf_dm.data.recorder import SessionEvent, SessionRecorder, TrialRecord
from hf_dm.hardware.bringup_constants import (
    CCW_FORCE_OUTPUT_CHANNEL,
    CW_FORCE_OUTPUT_CHANNEL,
    LICK_INPUT_CHANNEL,
    PUMP_OUTPUT_CHANNEL,
)
from hf_dm.session.controller import HFDecisionSession, HardwareCleanupError
from hf_dm.session.hardware_self_test import (
    HardwareSelfTestSession,
    shorten_for_hardware_self_test,
)
from hf_dm.safety_interlock import (
    HardwareSafetyInterlockError,
    arm_safety_interlock,
)


def test_live_session_rejects_quiescence_shorter_than_barcode():
    config = SessionConfig()
    config.timing.quiescence_s = 0.5
    session = HFDecisionSession(config)

    with pytest.raises(RuntimeError, match="cannot display the longest enabled Frame2TTL"):
        session._require_live_barcode_capacity()


def test_hardware_self_test_disables_barcode_for_shortened_quiescence():
    config = SessionConfig()

    shorten_for_hardware_self_test(config)

    assert config.timing.quiescence_s == 0.05
    assert config.frame2ttl.barcode_enabled is False
    HardwareSelfTestSession(config)._require_live_barcode_capacity()


def test_bpod_descriptor_records_device_and_firmware_identity():
    session = HFDecisionSession(SessionConfig())
    session.bpod = SimpleNamespace(
        port="COM5",
        serial_number="17032730",
        version=SimpleNamespace(
            firmware=(23, 10),
            machine=4,
            machine_str="State Machine 2+ r1.0",
            pcb=3,
            bpod_core="0.1.0a12",
        ),
    )

    assert session._bpod_descriptor() == {
        "port": "COM5",
        "serial_number": "17032730",
        "firmware": [23, 10],
        "machine": 4,
        "machine_name": "State Machine 2+ r1.0",
        "pcb_revision": 3,
        "bpod_core": "0.1.0a12",
    }


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


class PartialConnectDIO(FakeDIO):
    def __init__(self) -> None:
        super().__init__()
        self.raw = SimpleNamespace(module_info={"firmware_version": 0})

    def open(self, **_kwargs) -> None:
        self.ops.append("open")

    def configure_rotary_and_force(self, **_kwargs) -> None:
        self.ops.append("configure")


class FailingCleanupRelay(FakeRelay):
    def task_outputs_off(self) -> dict:
        self.ops.append("task_outputs_off")
        raise OSError("StreamDIO output USB disconnected")


class InterruptingCleanupRelay(FakeRelay):
    def task_outputs_off(self) -> dict:
        self.ops.append("task_outputs_off")
        raise KeyboardInterrupt("operator interrupted cleanup")


class FailingCameraDIO(FakeDIO):
    def stop_camera(self) -> None:
        self.ops.append("stop_camera")
        raise OSError("StreamDIO camera USB disconnected")


class FailingStateBpod:
    @property
    def is_running(self) -> bool:
        raise OSError("Bpod state unavailable")

    def set_softcode_handler(self, _handler) -> None:
        raise OSError("Bpod handler detach failed")

    def close(self) -> None:
        self.closed = True


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


class FailingVideoRecorder:
    def stop(self) -> dict:
        raise OSError("camera disappeared during shutdown")


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


class FailingTrialSession(HFDecisionSession):
    def _run_bpod_dm_trial(self, trial_index: int) -> bool:
        raise RuntimeError(f"trial {trial_index} failed")


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


def test_streamdio_status_snapshot_is_bounded_debug_data_and_stream_summary(tmp_path):
    session = HFDecisionSession(SessionConfig())
    session.recorder = SessionRecorder(tmp_path, "S1", "dm")
    session.recorder.register_stream("streamdio", {"type": "streamdio", "stream_port": "COM7"})
    session.dio = MagicMock()
    session.dio.get_module_status.return_value = {
        "streaming_active": True,
        "analog_input_status": {"channel": 13},
    }
    session._cap_touch_status_snapshot_count = 2
    session._latest_cap_touch_status = {"channel": LICK_INPUT_CHANNEL, "last_value": 812}

    session._capture_streamdio_module_status(12.5)
    session._update_streamdio_status_summary()

    assert [event.event_type for event in session.recorder.events] == ["session_start"]
    assert session.recorder.debug_event_counts == {"streamdio_module_status_snapshot": 1}
    status_poll = session.recorder.get_stream_summaries()["streamdio"]["status_poll"]
    assert status_poll["module_status_snapshot_count"] == 1
    assert status_poll["module_status_error_count"] == 0
    assert status_poll["cap_touch_status_snapshot_count"] == 2
    assert status_poll["latest_module_status"]["streaming_active"] is True
    assert status_poll["latest_cap_touch_status"]["last_value"] == 812


def test_prepare_bpod_run_records_discarded_idle_serial_bytes(tmp_path):
    session = HFDecisionSession(SessionConfig())
    session.recorder = SessionRecorder(tmp_path, "S1", "dm")
    session.speaker_relay = MagicMock()
    session.speaker_relay.prepare_bpod_run.return_value = 5

    session._prepare_bpod_run(3)

    event = session.recorder.events[-1]
    assert event.event_type == "bpod_serial_input_discarded"
    assert event.payload == {
        "trial_number": 3,
        "byte_count": 5,
        "reason": "idle_module_relay_bytes_before_fsm",
    }


def test_run_finalizes_after_video_and_hardware_cleanup(tmp_path):
    cfg = SessionConfig(n_trials=1, results_dir=str(tmp_path))
    session = OneTrialSession(cfg)
    session.bpod = MagicMock()
    session.bpod.port = "COM5"
    session.bpod.is_running = False
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


def test_run_finalizes_evidence_before_surfacing_cleanup_failure(tmp_path):
    cfg = SessionConfig(n_trials=1, results_dir=str(tmp_path))
    session = OneTrialSession(cfg)
    session.bpod = MagicMock(port="COM5", is_running=False)
    session.relay = FailingCleanupRelay()
    session.dio = FakeDIO()
    session.display = FakeDisplay()
    session.recorder = SessionRecorder(tmp_path, "S1", "dm")

    with pytest.raises(HardwareCleanupError, match="task_outputs_off"):
        session.run()

    assert session.display.closed
    assert "session_end" in [event.event_type for event in session.recorder.events]
    assert list(tmp_path.glob("*.npz"))


def test_run_preserves_trial_failure_after_cleanup_and_finalization(tmp_path):
    cfg = SessionConfig(n_trials=1, results_dir=str(tmp_path))
    session = FailingTrialSession(cfg)
    session.bpod = MagicMock(port="COM5", is_running=False)
    session.relay = FailingCleanupRelay()
    session.dio = FakeDIO()
    session.display = FakeDisplay()
    session.recorder = SessionRecorder(tmp_path, "S1", "dm")

    with pytest.raises(RuntimeError, match="trial 1 failed"):
        session.run()

    assert isinstance(session._last_cleanup_error, HardwareCleanupError)
    assert session.display.closed
    assert "session_end" in [event.event_type for event in session.recorder.events]
    assert list(tmp_path.glob("*.npz"))


def test_cleanup_continues_when_logging_and_audio_shutdown_fail():
    session = HFDecisionSession(SessionConfig())
    session.bpod = MagicMock()
    session.bpod.is_running = False
    session.relay = FakeRelay()
    session.speaker_relay = FakeRelay()
    session.white_noise_player = MagicMock()
    session.white_noise_player.stop.side_effect = OSError("audio unavailable")
    session.dio = FakeDIO()
    session.recorder = MagicMock()
    session.recorder.log_event.side_effect = OSError("disk unavailable")

    with pytest.raises(HardwareCleanupError) as caught:
        session.close()

    assert {item["stage"] for item in caught.value.errors} == {"white_noise_stop"}
    assert session.relay.ops == ["task_outputs_off", ("cap_touch_power", False)]
    assert session.speaker_relay.ops == [("speaker_gate", False)]
    assert session.dio.ops == ["stop_camera", "close"]
    session.bpod.close.assert_called_once_with()
    assert session._closed is True


def test_existing_safety_interlock_blocks_before_hardware_connection(tmp_path, monkeypatch):
    marker = tmp_path / "rig-interlock.json"
    arm_safety_interlock({"subject_id": "previous"}, marker)
    session = HFDecisionSession(SessionConfig())
    session._safety_interlock_path = marker
    connect_bpod = MagicMock()
    monkeypatch.setattr("hf_dm.session.controller.connect_bpod_core", connect_bpod)

    with pytest.raises(HardwareSafetyInterlockError, match="startup blocked"):
        session.connect()

    connect_bpod.assert_not_called()


def test_usb_cleanup_failures_latch_interlock_and_continue_all_off_attempts(tmp_path):
    marker = tmp_path / "rig-interlock.json"
    session = HFDecisionSession(SessionConfig())
    session._safety_interlock_path = marker
    arm_safety_interlock({"subject_id": "S1"}, marker)
    session._safety_interlock_armed = True
    session.bpod = MagicMock()
    session.bpod.is_running = False
    session.relay = FailingCleanupRelay()
    session.speaker_relay = FakeRelay()
    session.dio = FailingCameraDIO()

    with pytest.raises(HardwareCleanupError):
        session.close()

    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["state"] == "cleanup_failed"
    assert {item["stage"] for item in payload["cleanup_errors"]} == {
        "task_outputs_off",
        "camera_triggers_stop",
    }
    assert session.relay.ops == ["task_outputs_off", ("cap_touch_power", False)]
    assert session.speaker_relay.ops == [("speaker_gate", False)]
    assert session.dio.ops == ["stop_camera", "close"]
    session.bpod.close.assert_called_once_with()


def test_operator_abort_failure_latches_interlock_but_runs_fallback_cleanup(tmp_path):
    marker = tmp_path / "rig-interlock.json"
    session = HFDecisionSession(SessionConfig())
    session._safety_interlock_path = marker
    arm_safety_interlock({"subject_id": "S1"}, marker)
    session._safety_interlock_armed = True
    session.bpod = MagicMock()
    session.bpod.is_running = True
    session.bpod.send_softcode.side_effect = OSError("Bpod disconnected during abort")
    session.relay = FakeRelay()
    session.dio = FakeDIO()

    with pytest.raises(HardwareCleanupError):
        session.close()

    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert [item["stage"] for item in payload["cleanup_errors"]] == ["bpod_abort"]
    assert session.relay.ops == ["task_outputs_off", ("cap_touch_power", False)]
    assert session.dio.ops == ["stop_camera", "close"]


def test_verified_hardware_cleanup_clears_armed_interlock(tmp_path):
    marker = tmp_path / "rig-interlock.json"
    session = HFDecisionSession(SessionConfig())
    session._safety_interlock_path = marker
    arm_safety_interlock({"subject_id": "S1"}, marker)
    session._safety_interlock_armed = True
    session.bpod = MagicMock()
    session.bpod.is_running = False
    session.relay = FakeRelay()
    session.dio = FakeDIO()

    session.close()

    assert not marker.exists()
    status = json.loads(marker.with_name(f"{marker.name}.status").read_text(encoding="utf-8"))
    assert status["state"] == "verified_safe"


def test_verifier_cleanup_failure_cannot_skip_all_off_or_clear_interlock(tmp_path):
    marker = tmp_path / "rig-interlock.json"
    session = HFDecisionSession(SessionConfig())
    session._safety_interlock_path = marker
    arm_safety_interlock({"subject_id": "S1"}, marker)
    session._safety_interlock_armed = True
    session.bpod = MagicMock()
    session.bpod.is_running = False
    session.relay = FakeRelay()
    session.speaker_relay = FakeRelay()
    session.dio = FakeDIO()
    session._active_trial_runtime = MagicMock()
    session._drain_output_verifications = MagicMock(side_effect=RuntimeError("drain failed"))
    session._shutdown_output_verifier = MagicMock(side_effect=RuntimeError("shutdown failed"))

    with pytest.raises(HardwareCleanupError) as caught:
        session.close()

    assert session.relay.ops == ["task_outputs_off", ("cap_touch_power", False)]
    assert session.speaker_relay.ops == [("speaker_gate", False)]
    assert session.dio.ops == ["stop_camera", "close"]
    session.bpod.close.assert_called_once_with()
    assert session._closed is True
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["state"] == "cleanup_failed"
    assert {item["stage"] for item in payload["cleanup_errors"]} == {
        "output_verifier_drain",
        "output_verifier_shutdown",
    }
    assert caught.value.errors == payload["cleanup_errors"]


def test_verifier_drain_timeout_latches_interlock(tmp_path, monkeypatch):
    marker = tmp_path / "rig-interlock.json"
    session = HFDecisionSession(SessionConfig())
    session._safety_interlock_path = marker
    arm_safety_interlock({"subject_id": "S1"}, marker)
    session._safety_interlock_armed = True
    session.bpod = MagicMock()
    session.bpod.is_running = False
    session.relay = FakeRelay()
    session.dio = FakeDIO()
    runtime = MagicMock()
    pending = Future()
    session._active_trial_runtime = runtime
    session._output_verification_futures = [(runtime, pending)]
    monkeypatch.setattr(
        "hf_dm.session.controller.wait",
        lambda *_args, **_kwargs: (set(), {pending}),
    )

    with pytest.raises(HardwareCleanupError) as caught:
        session.close()

    assert "output_verifier_drain" in {
        item["stage"] for item in caught.value.errors
    }
    assert json.loads(marker.read_text(encoding="utf-8"))["state"] == "cleanup_failed"


def test_connect_failure_before_relay_keeps_interlock_latched(tmp_path, monkeypatch):
    marker = tmp_path / "rig-interlock.json"
    config = SessionConfig(results_dir=str(tmp_path / "sessions"))
    session = HFDecisionSession(config, run_mode="bench")
    session._safety_interlock_path = marker
    monkeypatch.setattr(
        "hf_dm.session.controller.connect_bpod_core",
        MagicMock(side_effect=OSError("Bpod unavailable")),
    )

    with pytest.raises(OSError, match="Bpod unavailable"):
        session.connect()
    with pytest.raises(HardwareCleanupError) as caught:
        session.close()

    assert {item["stage"] for item in caught.value.errors} == {
        "task_outputs_off_unavailable"
    }
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["state"] == "cleanup_failed"


def test_bpod_state_handler_and_monitor_failures_still_run_every_all_off_stage(tmp_path):
    marker = tmp_path / "rig-interlock.json"
    session = HFDecisionSession(SessionConfig())
    session._safety_interlock_path = marker
    arm_safety_interlock({"subject_id": "S1"}, marker)
    session._safety_interlock_armed = True
    session.bpod = FailingStateBpod()
    session.relay = FakeRelay()
    session.speaker_relay = FakeRelay()
    session.dio = FakeDIO()
    session._stop_operator_monitor = MagicMock(side_effect=RuntimeError("monitor stuck"))

    with pytest.raises(HardwareCleanupError) as caught:
        session.close()

    assert session.relay.ops == ["task_outputs_off", ("cap_touch_power", False)]
    assert session.speaker_relay.ops == [("speaker_gate", False)]
    assert session.dio.ops == ["stop_camera", "close"]
    assert session.bpod.closed is True
    assert session._closed is True
    assert {item["stage"] for item in caught.value.errors} == {
        "bpod_running_state",
        "bpod_softcode_handler_detach",
        "operator_monitor_stop",
    }
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["state"] == "cleanup_failed"


def test_partial_connect_camera_failure_retains_emergency_all_off_path(tmp_path, monkeypatch):
    config = SessionConfig(results_dir=str(tmp_path / "sessions"))
    session = HFDecisionSession(config, run_mode="bench")
    marker = tmp_path / "rig-interlock.json"
    session._safety_interlock_path = marker
    bpod = MagicMock()
    bpod.is_running = False
    bpod.port = "COM5"
    dio = PartialConnectDIO()
    relay = FakeRelay()
    monkeypatch.setattr("hf_dm.session.controller.connect_bpod_core", lambda _port: bpod)
    monkeypatch.setattr("hf_dm.session.controller.StreamDIOClient", lambda **_kwargs: dio)
    monkeypatch.setattr("hf_dm.session.controller.StreamDIOOutputs", lambda *_args, **_kwargs: relay)
    monkeypatch.setattr("hf_dm.session.controller.TaskDisplay", lambda *_args: FakeDisplay())
    monkeypatch.setattr(
        session,
        "_start_optional_video_recording",
        MagicMock(side_effect=RuntimeError("camera initialization failed")),
    )

    with pytest.raises(RuntimeError, match="camera initialization failed"):
        session.connect()
    session.close()

    assert relay.ops == ["task_outputs_off", ("cap_touch_power", False)]
    assert dio.ops == ["open", "configure", "stop_camera", "close"]
    bpod.close.assert_called_once_with()
    assert session._closed is True
    assert not marker.exists()
    status = json.loads(marker.with_name(f"{marker.name}.status").read_text(encoding="utf-8"))
    assert status["state"] == "verified_safe"


def test_operator_interrupt_during_cleanup_is_deferred_until_all_off_attempts_finish(tmp_path):
    marker = tmp_path / "rig-interlock.json"
    session = HFDecisionSession(SessionConfig())
    session._safety_interlock_path = marker
    arm_safety_interlock({"subject_id": "S1"}, marker)
    session._safety_interlock_armed = True
    session.bpod = MagicMock()
    session.bpod.is_running = False
    session.relay = InterruptingCleanupRelay()
    session.speaker_relay = FakeRelay()
    session.dio = FakeDIO()

    with pytest.raises(HardwareCleanupError) as caught:
        session.close()

    assert [item["stage"] for item in caught.value.errors] == ["task_outputs_off"]
    assert session.relay.ops == ["task_outputs_off", ("cap_touch_power", False)]
    assert session.speaker_relay.ops == [("speaker_gate", False)]
    assert session.dio.ops == ["stop_camera", "close"]
    session.bpod.close.assert_called_once_with()
    assert session._closed is True
    assert json.loads(marker.read_text(encoding="utf-8"))["state"] == "cleanup_failed"


def test_bpod_read_hang_and_camera_shutdown_failure_are_aggregated_after_all_off(tmp_path):
    marker = tmp_path / "rig-interlock.json"
    session = HFDecisionSession(SessionConfig())
    session._safety_interlock_path = marker
    arm_safety_interlock({"subject_id": "S1"}, marker)
    session._safety_interlock_armed = True
    session.bpod = MagicMock()
    session.bpod.is_running = True
    session._wait_for_bpod_trial_threads = MagicMock(
        side_effect=TimeoutError("Bpod read thread hung")
    )
    session.relay = FakeRelay()
    session.speaker_relay = FakeRelay()
    session.dio = FakeDIO()
    session.camera_recorder = FailingVideoRecorder()

    with pytest.raises(HardwareCleanupError) as caught:
        session.close()

    assert {item["stage"] for item in caught.value.errors} == {
        "bpod_abort",
        "camera_recording_stop",
    }
    assert session.relay.ops == ["task_outputs_off", ("cap_touch_power", False)]
    assert session.speaker_relay.ops == [("speaker_gate", False)]
    assert session.dio.ops == ["stop_camera", "close"]
    session.bpod.close.assert_called_once_with()
    assert session._closed is True


def test_preflight_saves_stream_camera_and_cleanup_events(tmp_path):
    cfg = SessionConfig(n_trials=99, results_dir=str(tmp_path))
    session = HFDecisionSession(cfg)
    session.bpod = MagicMock()
    session.bpod.port = "COM5"
    session.bpod.is_running = False
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


def test_preflight_display_wait_polls_recorder_durability_health(tmp_path):
    session = HFDecisionSession(SessionConfig())
    session.recorder = SessionRecorder(tmp_path, "S1", "dm")

    class FailingDisplay(FakeDisplay):
        def show_black(inner_self, duration_s):
            session.recorder._streamdio_wal_background_error = "OSError: disk full"

    session.display = FailingDisplay()

    with pytest.raises(RuntimeError, match="disk full"):
        session._show_black_with_recorder_health(1.0)


def test_hardware_self_test_returns_typed_validation_results(tmp_path):
    config = SessionConfig(n_trials=3)
    session = HardwareSelfTestSession(config)
    recorder = SessionRecorder(tmp_path, "S1", "decision_making")
    session.recorder = recorder
    shapes = ("star", "circle", None)
    choices = ("cw", "ccw", "timeout")
    rewards = (config.reward.matched_pump_s, config.reward.opposite_pump_s, 0.0)
    state_sequences = {
        "star": [
            "Quiescence_Pre", "Quiescence", "ChoiceWindow_Pre", "ChoiceWindow",
            "ChoiceBlankHigh_Pre", "ChoiceBlankHigh", "RewardHigh_Pre", "RewardHigh",
            "PostRewardHigh_Pre", "PostRewardHigh",
        ],
        "circle": [
            "Quiescence_Pre", "Quiescence", "ChoiceWindow_Pre", "ChoiceWindow",
            "ChoiceBlankLow_Pre", "ChoiceBlankLow", "RewardLow_Pre", "RewardLow",
            "PostRewardLow_Pre", "PostRewardLow",
        ],
        None: [
            "Quiescence_Pre", "Quiescence", "ChoiceWindow_Pre", "ChoiceWindow",
            "TimeoutForceOff", "TimeoutNoise_Pre", "TimeoutNoise",
            "PostTimeout_Pre", "PostTimeout",
        ],
    }
    for index, (shape, choice, reward) in enumerate(zip(shapes, choices, rewards), start=1):
        recorder.trials.append(
            TrialRecord(
                trial_index=index,
                phase="decision_making",
                force_direction="cw",
                choice=choice,
                matched=shape == "star",
                reward_duration_s=reward,
                responded=shape is not None,
                response_time_s=0.2,
                net_pulses=3,
                cw_pulses=3,
                ccw_pulses=0,
                final_position_deg=35.0,
                force_samples=2,
                force_last=1.0,
                force_mean=1.0,
                force_min=1.0,
                force_max=1.0,
                trial_start_wall_s=float(index),
                trial_start_mono_s=100.0 + index,
                chosen_shape=shape,
                force_output_direction="cw",
                camera_ttl_timestamps_us=[index * 1000, index * 1000 + 100],
            )
        )
        recorder.bpod_trials.append(
            {
                "trial_index": index,
                "events": [
                    {"type": "StateStart", "state": state}
                    for state in state_sequences[shape]
                ],
            }
        )
        if shape is not None:
            choice_mono = 10.0 + index
            recorder.events.extend(
                [
                    SessionEvent(1.0, choice_mono, "choice_detected", {"trial_index": index}),
                    SessionEvent(
                        1.0,
                        choice_mono + config.timing.choice_blank_s,
                        "pump_start",
                        {
                            "trial_index": index,
                            "actual_on_mono_s": choice_mono + config.timing.choice_blank_s,
                        },
                    ),
                    SessionEvent(
                        1.0,
                        choice_mono + config.timing.choice_blank_s + reward,
                        "pump_end",
                        {"trial_index": index, "actual_open_s": reward},
                    ),
                ]
            )
    for channel in (
        PUMP_OUTPUT_CHANNEL,
        CW_FORCE_OUTPUT_CHANNEL,
        CCW_FORCE_OUTPUT_CHANNEL,
        config.licks.power_output_channel,
    ):
        for state in (0, 1):
            recorder.events.append(
                SessionEvent(1.0, 1.0, "output_actual", {"channel": channel, "state": state})
            )
    recorder.events.extend(
        [
            SessionEvent(1.0, 1.0, "white_noise_start", {}),
            SessionEvent(1.0, 1.1, "white_noise_end", {}),
            SessionEvent(
                1.0,
                1.2,
                "video_recording_stopped",
                {"frames_saved": 6, "capture_error": None},
            ),
        ]
    )
    recorder.streamdio_events = [
        ("camera", -1, -1, index, index / 1000.0, index / 1000.0)
        for index in range(6)
    ]

    report = session.validate_results()

    assert report["status"] == "PASS"
    assert report["passed"]
    assert report["results"]
    assert all(
        set(result) == {"name", "status", "detail", "hint"}
        for result in report["results"]
    )

    # The maximum interval is an independent guard: a malformed or legacy
    # stall counter must not let a slow choice loop pass certification.
    recorder.trials[0].choice_loop_max_frame_s = 0.2
    recorder.trials[0].choice_loop_stall_count = 0
    with pytest.raises(RuntimeError, match="max frame=0.200000s"):
        session.validate_results()
