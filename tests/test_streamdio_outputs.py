import logging
import threading
from collections import defaultdict, deque

import pytest

import hf_dm.streamdio.client as client_module
from hf_dm.hardware.bringup_constants import (
    CAP_TOUCH_INPUT_PIN,
    FRAME2TTL_INPUT_CHANNEL,
    FORCE_ANALOG_CHANNEL,
    FORCE_ANALOG_PIN,
    LICK_INPUT_CHANNEL,
    SPEAKER_GATE_PIN,
    STREAMDIO_TIMING_HEARTBEAT_EVENT_CODE,
)
from hf_dm.streamdio.client import StreamDIOClient
from hf_dm.streamdio.outputs import StreamDIOOutputs


def test_streamdio_stop_retains_driver_buffer_stats():
    class StreamingProtocol:
        streaming_active = True
        _buffer_stats = {
            "bytes_read": 120,
            "events_processed": 10,
            "buffer_overruns": 0,
        }

        def stop_streaming(self):
            self.streaming_active = False

    client = object.__new__(StreamDIOClient)
    client._dio = StreamingProtocol()
    client._last_buffer_stats = {}

    client.stop_streaming()

    assert client.get_buffer_stats() == {
        "bytes_read": 120,
        "events_processed": 10,
        "buffer_overruns": 0,
    }


class FakeProtocol:
    streaming_active = True

    def __init__(self) -> None:
        self.client = None
        self.module_info = {
            "firmware_version": 5,
            "n_input_channels": 14,
            "n_output_channels": 13,
        }
        self.output_states = {}
        self.speaker_enabled = False
        self.speaker_samples = 0
        self.camera_enabled = False
        self.camera_frequency = 30
        self.camera_exposure_us = 10_000
        self.configured = None
        self._config_arcom = FakeArCOM()

    def configure_module(self, **kwargs) -> bool:
        self.configured = kwargs
        return True

    def get_analog_input_status(self) -> dict:
        analog = (self.configured or {}).get(
            "analog_input",
            {"channel": FORCE_ANALOG_CHANNEL, "enabled": True, "sample_interval_us": 1000},
        )
        channel = int(analog.get("channel", FORCE_ANALOG_CHANNEL))
        pin = CAP_TOUCH_INPUT_PIN if channel == LICK_INPUT_CHANNEL else FORCE_ANALOG_PIN
        return {
            "enabled": bool(analog.get("enabled", True)),
            "channel": channel,
            "pin": pin,
            "sample_interval_us": int(analog.get("sample_interval_us", 1000)),
            "last_value": 0,
        }

    def manual_override_output(self, channel: int, state: int) -> bool:
        self.output_states[channel] = state
        threading.Thread(
            target=self.client._dispatch_event,
            args=("dio_output", channel, state, 1234, 0.001234),
            daemon=True,
        ).start()
        return True

    def configure_speaker_noise(self, enabled: bool, sample_interval_us: int) -> bool:
        del sample_interval_us
        self.speaker_enabled = enabled
        return True

    def start_camera_triggers(self, frequency: int, exposure_us: int) -> bool:
        self.camera_enabled = True
        self.camera_frequency = frequency
        self.camera_exposure_us = exposure_us
        return True

    def stop_camera_triggers(self) -> bool:
        self.camera_enabled = False
        return True

    def _streamdio_get_equipment_status(self) -> dict:
        return {
            "camera_enabled": self.camera_enabled,
            "camera_frequency": self.camera_frequency,
            "camera_exposure_us": self.camera_exposure_us,
        }

    def get_speaker_status(self) -> dict:
        if self.speaker_enabled:
            self.speaker_samples += 10
        return {
            "enabled": self.speaker_enabled,
            "pin": SPEAKER_GATE_PIN,
            "sample_interval_us": 100,
            "samples_generated": self.speaker_samples,
            "current_state": 0,
        }


class FakeArCOM:
    def __init__(self, values: list[int] | None = None) -> None:
        self.writes = []
        self.values = iter(values or [1, 14, 25, 1, 1, 812, 123, 700, 283, 0, 0])

    def write(self, value, data_type) -> None:
        self.writes.append((value, data_type))

    def read(self, *_args):
        return next(self.values)


def _client() -> StreamDIOClient:
    client = object.__new__(StreamDIOClient)
    client.config_port = "COM_CONFIG"
    client.stream_port = "COM_STREAM"
    client.baudrate = 115200
    client._callback = None
    client._config_lock = threading.Lock()
    client._output_condition = threading.Condition()
    client._output_states = {channel: 0 for channel in range(1, 14)}
    client._output_sequences = defaultdict(int)
    client._output_events = {}
    client._output_event_history = defaultdict(lambda: deque(maxlen=64))
    client._dio = FakeProtocol()
    client._dio.client = client
    return client


def test_output_command_requires_matching_firmware_event():
    client = _client()

    transition = client.set_output_channel(1, True)

    assert transition["acknowledged"]
    assert transition["event_verified"]
    assert transition["channel"] == 1
    assert transition["state"] == 1
    assert transition["module_timestamp_us"] == 1234


def test_camera_start_and_stop_require_fresh_firmware_status():
    client = _client()

    client.start_camera_session(frequency_hz=30, exposure_us=10_000)
    client.stop_camera()

    assert client.raw.camera_enabled is False


class LegacyProtocolWithoutStatusProbe:
    """Driver builds predating _streamdio_get_equipment_status."""

    def start_camera_triggers(self, frequency: int, exposure_us: int) -> bool:
        return True

    def stop_camera_triggers(self) -> bool:
        return True


def test_camera_verification_fails_closed_without_status_probe():
    client = _client()
    client._dio = LegacyProtocolWithoutStatusProbe()

    with pytest.raises(RuntimeError, match="status probe"):
        client.start_camera_session(frequency_hz=30, exposure_us=10_000)
    with pytest.raises(RuntimeError, match="status probe"):
        client.stop_camera()


def test_camera_start_rejects_mismatched_firmware_status():
    client = _client()
    client.raw._streamdio_get_equipment_status = lambda: {
        "camera_enabled": True,
        "camera_frequency": 20,
        "camera_exposure_us": 10_000,
    }

    with pytest.raises(RuntimeError, match="Unexpected camera trigger status"):
        client.start_camera_session(frequency_hz=30, exposure_us=10_000)


def test_passive_output_observer_accepts_event_that_arrived_before_wait():
    client = _client()
    client._dispatch_event("dio_output", 1, 1, 4321, 0.004321)

    transition = client.wait_for_output_event(
        1,
        True,
        after_sequence=0,
        timeout_s=0.0,
    )

    assert transition["event_verified"]
    assert transition["actuator"] == "bpod_fsm_serial"
    assert transition["sequence"] == 1
    assert transition["module_timestamp_us"] == 4321


def test_passive_output_observer_uses_latest_event_after_history_rollover():
    client = _client()
    for sequence in range(70):
        client._dispatch_event("dio_output", 1, sequence % 2, 1000 + sequence, sequence / 1000)
    client._output_event_history[1].clear()

    transition = client.wait_for_output_event(
        1,
        True,
        after_sequence=1,
        timeout_s=0.0,
    )

    assert transition["event_verified"]
    assert transition["sequence"] > 64


def test_streamdio_close_surfaces_errors_after_attempting_every_stage():
    client = _client()
    calls = []
    client.stop_streaming = lambda: (_ for _ in ()).throw(OSError("stream stop"))
    client.stop_camera = lambda: calls.append("camera")
    client.raw.close = lambda: calls.append("driver")

    with pytest.raises(RuntimeError, match="stream stop"):
        client.close()

    assert calls == ["camera", "driver"]
    assert client.raw is None


def test_rotary_force_config_requires_cap_touch_digitizer_firmware(monkeypatch):
    client = _client()
    client.raw.module_info["firmware_version"] = 2
    monkeypatch.setattr(client_module, "STREAMDIO_TIMING_HEARTBEAT_FIRMWARE_VERSION", 2)

    try:
        client.configure_rotary_and_force(lick_channels=(LICK_INPUT_CHANNEL,))
    except RuntimeError as exc:
        assert "cap-touch digitizer" in str(exc)
    else:
        raise AssertionError("Expected old StreamDIO firmware to be rejected")


def test_rotary_force_config_accepts_digitizer_firmware():
    client = _client()

    client.configure_rotary_and_force(
        lick_channels=(LICK_INPUT_CHANNEL,),
        extra_input_channels=(FRAME2TTL_INPUT_CHANNEL,),
    )

    enabled = client.raw.configured["input_channels_enabled"]
    assert enabled[LICK_INPUT_CHANNEL - 1]
    assert enabled[FRAME2TTL_INPUT_CHANNEL - 1]
    assert client.raw.configured["analog_input"]["channel"] == FORCE_ANALOG_CHANNEL


def test_rotary_force_config_requires_timing_heartbeat_firmware():
    client = _client()
    client.raw.module_info["firmware_version"] = 4

    with pytest.raises(RuntimeError, match="100 ms timing heartbeat"):
        client.configure_rotary_and_force(lick_channels=())


def test_firmware_heartbeat_is_normalized_and_gets_pc_receive_timestamps():
    client = _client()
    received = []

    def callback(event_type, channel, value, timestamp_us, relative_time_s, **timestamps):
        received.append(
            (event_type, channel, value, timestamp_us, relative_time_s, timestamps)
        )

    client._callback = callback
    client._dispatch_event(
        "diagnostic",
        STREAMDIO_TIMING_HEARTBEAT_EVENT_CODE,
        None,
        100_000,
        0.1,
    )

    event = received[0]
    assert event[:5] == ("timing_heartbeat", None, None, 100_000, 0.1)
    assert event[5]["host_perf_counter_s"] > 0
    assert event[5]["host_wall_time_s"] > 0


def test_streamdio_driver_log_filter_suppresses_only_timing_heartbeat():
    heartbeat = logging.LogRecord(
        "pybpod_streamDIO_module.stream_dio_protocol",
        logging.WARNING,
        __file__,
        1,
        "Diagnostic event 1011 at 123456 us",
        (),
        None,
    )
    fault = logging.LogRecord(
        "pybpod_streamDIO_module.stream_dio_protocol",
        logging.WARNING,
        __file__,
        1,
        "Diagnostic event 1001 at 123456 us",
        (),
        None,
    )

    log_filter = client_module._TimingHeartbeatLogFilter()
    assert not log_filter.filter(heartbeat)
    assert log_filter.filter(fault)


def test_cap_touch_analog_diagnostic_streams_touch_channel_instead_of_force():
    client = _client()

    status = client.configure_cap_touch_analog_diagnostic(sample_interval_us=2000)

    enabled = client.raw.configured["input_channels_enabled"]
    assert not any(enabled)
    assert client.raw.configured["analog_input"]["channel"] == LICK_INPUT_CHANNEL
    assert client.raw.configured["analog_input"]["sample_interval_us"] == 2000
    assert status["channel"] == LICK_INPUT_CHANNEL
    assert status["pin"] == CAP_TOUCH_INPUT_PIN


def test_cap_touch_status_reads_digitizer_fields():
    client = _client()

    status = client.get_cap_touch_status()

    assert client.raw._config_arcom.writes == [(ord("l"), "uint8")]
    assert status["enabled"]
    assert status["channel"] == LICK_INPUT_CHANNEL
    assert status["pin"] == 25
    assert status["last_value"] == 812
    assert status["rising_high_threshold"] == 700


def test_module_status_collects_driver_status_snapshots():
    client = _client()
    client.raw.module_info["speaker_pin"] = 27  # Driver's legacy default.

    status = client.get_module_status()

    assert status["config_port"] == "COM_CONFIG"
    assert status["stream_port"] == "COM_STREAM"
    assert status["streaming_active"]
    assert status["module_info"]["firmware_version"] == 5
    assert status["analog_input_status"]["channel"] == FORCE_ANALOG_CHANNEL
    assert status["speaker_status"]["pin"] == SPEAKER_GATE_PIN
    assert status["module_info"]["speaker_pin"] == SPEAKER_GATE_PIN


def test_cap_touch_threshold_configuration_sends_runtime_firmware_command():
    client = _client()
    client.raw._config_arcom = FakeArCOM(
        [1, 1, 14, 25, 0, 0, 0, 0, 250, 90, 0, 0]
    )

    status = client.configure_cap_touch_digitizer(
        min_high_threshold=250,
        delta_high=120,
        delta_low=90,
    )

    assert client.raw._config_arcom.writes == [
        (ord("t"), "uint8"),
        (250, "uint16"),
        (120, "uint16"),
        (90, "uint16"),
        (ord("l"), "uint8"),
    ]
    assert status["channel"] == LICK_INPUT_CHANNEL
    assert status["rising_high_threshold"] == 250
    assert status["rising_low_threshold"] == 90


def test_cap_touch_threshold_configuration_requires_v4_firmware():
    client = _client()
    client.raw.module_info["firmware_version"] = 3

    try:
        client.configure_cap_touch_digitizer(min_high_threshold=250)
    except RuntimeError as exc:
        assert "firmware version 3" in str(exc)
        assert "expected 4" in str(exc)
    else:
        raise AssertionError("Expected firmware v3 to be rejected for threshold tuning")


def test_speaker_pulse_generates_samples_and_finishes_disabled():
    outputs = StreamDIOOutputs(_client())

    result = outputs.pulse_speaker_gate(0.001)

    assert result["samples_generated_during_pulse"] > 0
    assert result["on"]["status"]["enabled"]
    assert not result["off"]["status"]["enabled"]
