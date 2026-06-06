import threading
import time
from collections import defaultdict

from hf_dm.hardware.bringup_constants import FORCE_ANALOG_CHANNEL, FORCE_ANALOG_PIN, LICK_INPUT_CHANNEL
from hf_dm.streamdio.client import StreamDIOClient
from hf_dm.streamdio.outputs import StreamDIOOutputs


class FakeProtocol:
    streaming_active = True

    def __init__(self) -> None:
        self.client = None
        self.module_info = {
            "firmware_version": 3,
            "n_input_channels": 14,
            "n_output_channels": 13,
        }
        self.output_states = {}
        self.speaker_enabled = False
        self.speaker_samples = 0
        self.configured = None
        self._config_arcom = FakeArCOM()

    def configure_module(self, **kwargs) -> bool:
        self.configured = kwargs
        return True

    def get_analog_input_status(self) -> dict:
        return {
            "enabled": True,
            "channel": FORCE_ANALOG_CHANNEL,
            "pin": FORCE_ANALOG_PIN,
            "sample_interval_us": 1000,
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

    def get_speaker_status(self) -> dict:
        if self.speaker_enabled:
            self.speaker_samples += 10
        return {
            "enabled": self.speaker_enabled,
            "pin": 27,
            "sample_interval_us": 100,
            "samples_generated": self.speaker_samples,
            "current_state": 0,
        }


class FakeArCOM:
    def __init__(self) -> None:
        self.writes = []
        self.values = iter([1, 14, 25, 1, 1, 812, 123, 700, 283, 0, 0])

    def write(self, value, data_type) -> None:
        self.writes.append((value, data_type))

    def read(self, *_args):
        return next(self.values)


def _client() -> StreamDIOClient:
    client = object.__new__(StreamDIOClient)
    client._callback = None
    client._config_lock = threading.Lock()
    client._output_condition = threading.Condition()
    client._output_states = {channel: 0 for channel in range(1, 14)}
    client._output_sequences = defaultdict(int)
    client._output_events = {}
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


def test_rotary_force_config_requires_cap_touch_digitizer_firmware():
    client = _client()
    client.raw.module_info["firmware_version"] = 2

    try:
        client.configure_rotary_and_force(lick_channels=(LICK_INPUT_CHANNEL,))
    except RuntimeError as exc:
        assert "cap-touch digitizer" in str(exc)
    else:
        raise AssertionError("Expected old StreamDIO firmware to be rejected")


def test_rotary_force_config_accepts_digitizer_firmware():
    client = _client()

    client.configure_rotary_and_force(lick_channels=(LICK_INPUT_CHANNEL,))

    enabled = client.raw.configured["input_channels_enabled"]
    assert enabled[LICK_INPUT_CHANNEL - 1]
    assert client.raw.configured["analog_input"]["channel"] == FORCE_ANALOG_CHANNEL


def test_cap_touch_status_reads_digitizer_fields():
    client = _client()

    status = client.get_cap_touch_status()

    assert client.raw._config_arcom.writes == [(ord("l"), "uint8")]
    assert status["enabled"]
    assert status["channel"] == LICK_INPUT_CHANNEL
    assert status["pin"] == 25
    assert status["last_value"] == 812
    assert status["rising_high_threshold"] == 700


def test_speaker_pulse_generates_samples_and_finishes_disabled():
    outputs = StreamDIOOutputs(_client())

    result = outputs.pulse_speaker_gate(0.001)

    assert result["samples_generated_during_pulse"] > 0
    assert result["on"]["status"]["enabled"]
    assert not result["off"]["status"]["enabled"]
