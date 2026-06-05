import threading
import time
from collections import defaultdict

from hf_dm.streamdio.client import StreamDIOClient
from hf_dm.streamdio.outputs import StreamDIOOutputs


class FakeProtocol:
    streaming_active = True

    def __init__(self) -> None:
        self.client = None
        self.output_states = {}
        self.speaker_enabled = False
        self.speaker_samples = 0

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


def _client() -> StreamDIOClient:
    client = object.__new__(StreamDIOClient)
    client._callback = None
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


def test_speaker_pulse_generates_samples_and_finishes_disabled():
    outputs = StreamDIOOutputs(_client())

    result = outputs.pulse_speaker_gate(0.001)

    assert result["samples_generated_during_pulse"] > 0
    assert result["on"]["status"]["enabled"]
    assert not result["off"]["status"]["enabled"]
