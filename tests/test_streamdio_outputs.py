"""StreamDIOClient against streamdio's simulated module (the real driver underneath)."""

import time

import pytest
from streamdio import StreamDIOConfigError
from streamdio import protocol as p
from streamdio.testing import FakeModule, event, fake_streamdio

from hf_dm.hardware.bringup_constants import (
    CAP_TOUCH_INPUT_CHANNEL,
    CAP_TOUCH_INPUT_PIN,
    CAP_TOUCH_POWER_OUTPUT_CHANNEL,
    CCW_FORCE_OUTPUT_CHANNEL,
    CW_FORCE_OUTPUT_CHANNEL,
    DIRECTION_INPUT_CHANNEL,
    FORCE_ANALOG_CHANNEL,
    FRAME2TTL_INPUT_CHANNEL,
    PULSE_INPUT_CHANNEL,
    PUMP_OUTPUT_CHANNEL,
    SPEAKER_GATE_PIN,
    SPEAKER_OUTPUT_CHANNEL,
)
from hf_dm.streamdio.channel_map import MODULE_OUTPUT_NUMBERS, hfdm_channel_map
from hf_dm.streamdio.client import StreamDIOClient
from hf_dm.streamdio.outputs import StreamDIOOutputs
from hf_dm.timing_heartbeat import TIMING_HEARTBEAT_EVENT_TYPE


def wait_until(condition, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.005)
    pytest.fail("condition not met in time")


@pytest.fixture
def module():
    fake = FakeModule()
    with fake_streamdio(fake):
        yield fake


def connect(module, channel_map=None) -> StreamDIOClient:
    client = StreamDIOClient(module.config_port, module.stream_port, baudrate=1_312_500)
    client.open()
    client.apply_map(channel_map or hfdm_channel_map())
    return client


@pytest.fixture
def client(module):
    client = connect(module)
    yield client
    client.close()


def collect(client) -> list[tuple]:
    received: list[tuple] = []
    client.start_streaming(
        lambda event_type, channel, value, timestamp_us, relative_s, **_: received.append(
            (event_type, channel, value)
        )
    )
    return received


def test_the_map_puts_each_channel_on_its_wired_pin():
    board = FakeModule().board_info()
    cfg = hfdm_channel_map()
    cfg.validate(board)
    pins = {channel.name: board.pin(channel.pin).pin for channel in cfg}
    assert pins == {
        "pump": 23,
        "cwforce": 21,
        "ccwforce": 19,
        "cap_pwr": 15,
        "speaker": SPEAKER_GATE_PIN,
        "camera": 32,
        "wheel_a": 2,
        "wheel_b": 4,
        "f2ttl": 6,
        "force": 14,
        "lick": CAP_TOUCH_INPUT_PIN,
    }
    # The Bpod's serial messages rely on these, whatever the session leaves out.
    assert MODULE_OUTPUT_NUMBERS == {
        PUMP_OUTPUT_CHANNEL: 1,
        CW_FORCE_OUTPUT_CHANNEL: 2,
        CCW_FORCE_OUTPUT_CHANNEL: 3,
        CAP_TOUCH_POWER_OUTPUT_CHANNEL: 4,
        SPEAKER_OUTPUT_CHANNEL: 5,
    }
    lean = hfdm_channel_map(cap_touch=False, frame2ttl=False)
    assert [c.name for c in lean][:6] == [c.name for c in cfg][:6]


def test_map_is_applied_with_the_heartbeat_armed(module, client):
    assert module.config_source == p.CONFIG_SOURCE_HOST
    assert module.heartbeat_ms == 100
    assert client.map_info()["config_hash"] == module.config_hash
    assert client.map_info()["channel_map"] == hfdm_channel_map().to_dict()
    assert client.module_info["speaker_pin"] == SPEAKER_GATE_PIN


def test_module_identity_holds_only_what_survives_sessions_and_reboots(module):
    first = connect(module)
    identity = first.module_info
    first.close()
    module.reboot()
    second = connect(module, hfdm_channel_map(cap_touch=False, lick_raw=False))
    try:
        # A rig certificate binds this; the map and boot ID change per session.
        assert second.module_info == identity
        assert {"firmware_version", "build_id", "board"} <= set(identity)
        assert second.map_info()["boot_id"] != 0
    finally:
        second.close()


def test_invalid_lick_thresholds_are_refused_before_anything_changes(module):
    client = StreamDIOClient(module.config_port, module.stream_port, baudrate=0)
    client.open()
    try:
        with pytest.raises(StreamDIOConfigError, match="off_delta"):
            client.apply_map(hfdm_channel_map(lick_on_delta=100, lick_off_delta=150))
        assert module.config_source != p.CONFIG_SOURCE_HOST
    finally:
        client.close()


def test_output_command_requires_matching_firmware_event(module, client):
    collect(client)

    transition = client.set_output_channel(CAP_TOUCH_POWER_OUTPUT_CHANNEL, True)

    assert transition["acknowledged"]
    assert transition["event_verified"]
    assert transition["channel"] == CAP_TOUCH_POWER_OUTPUT_CHANNEL
    assert transition["state"] == 1
    assert "module_timestamp_us" in transition
    # evil_rig output 13 is the map's output 4.
    assert module.outputs == 1 << 3


def test_unknown_output_is_refused(client):
    with pytest.raises(RuntimeError, match="not in the StreamDIO map"):
        client.set_output_channel(7, True)


def test_events_reach_the_callback_in_evil_rig_numbers(module, client):
    received = collect(client)

    # Inputs in map order: wheel_a, wheel_b, f2ttl, lick.
    module.emit(module.input_state(0b0001))
    module.emit(module.input_state(0b0011))
    module.emit(module.input_state(0b0111))
    module.emit(module.input_state(0b1111))
    module.emit(module.analog_sample(0, 512))
    module.emit(event(p.DIAG_LATENCY_HEARTBEAT, 7 << p.DIO_STATE_SHIFT | module.now_us))

    expected = [
        ("dio_input", PULSE_INPUT_CHANNEL, 1),
        ("dio_input", DIRECTION_INPUT_CHANNEL, 1),
        ("dio_input", FRAME2TTL_INPUT_CHANNEL, 1),
        ("dio_input", CAP_TOUCH_INPUT_CHANNEL, 1),
        ("analog_input", FORCE_ANALOG_CHANNEL, 512),
        (TIMING_HEARTBEAT_EVENT_TYPE, None, None),
    ]
    wait_until(lambda: all(item in received for item in expected))


def test_each_camera_frame_is_one_camera_event(module, client):
    received = collect(client)
    camera = 1 << 5  # the camera trigger is the map's output 6

    for _ in range(3):
        module.emit(module.output_state(camera))
        module.emit(module.output_state(0))

    wait_until(lambda: len(client.get_camera_events()) == 3)
    assert [e for e in received if e[0] in {"camera", "dio_output"}] == [("camera", None, None)] * 3


def test_camera_start_and_stop_are_verified(module, client):
    client.start_camera_session(frequency_hz=60, exposure_us=2_000)
    status = client.raw.pulses["camera"].status()
    assert status["running"] and status["frequency_hz"] == 60 and status["width_us"] == 2_000

    client.stop_camera()

    assert not client.raw.pulses["camera"].status()["running"]


def test_camera_start_rejects_mismatched_firmware_status(client, monkeypatch):
    from streamdio.channels import Pulse

    monkeypatch.setattr(Pulse, "status", lambda self: {"running": True, "frequency_hz": 20.0})

    with pytest.raises(RuntimeError, match="Unexpected camera trigger status"):
        client.start_camera_session(frequency_hz=30, exposure_us=10_000)


def test_passive_output_observer_accepts_event_that_arrived_before_wait(client):
    client._dispatch_event("dio_output", 1, 1, 4321, 0.004321)

    transition = client.wait_for_output_event(
        PUMP_OUTPUT_CHANNEL,
        True,
        after_sequence=0,
        timeout_s=0.0,
    )

    assert transition["event_verified"]
    assert transition["actuator"] == "bpod_fsm_serial"
    assert transition["sequence"] == 1
    assert transition["module_timestamp_us"] == 4321


def test_passive_output_observer_uses_latest_event_after_history_rollover(client):
    for sequence in range(70):
        client._dispatch_event("dio_output", 1, sequence % 2, 1000 + sequence, sequence / 1000)
    client._output_event_history[PUMP_OUTPUT_CHANNEL].clear()

    transition = client.wait_for_output_event(
        PUMP_OUTPUT_CHANNEL,
        True,
        after_sequence=1,
        timeout_s=0.0,
    )

    assert transition["event_verified"]
    assert transition["sequence"] > 64


def test_streamdio_close_surfaces_errors_after_attempting_every_stage(module):
    client = connect(module)
    driver = client.raw
    calls = []
    client.stop_streaming = lambda: (_ for _ in ()).throw(OSError("stream stop"))
    client.stop_camera = lambda: calls.append("camera")
    driver_close = driver.close
    driver.close = lambda: calls.append("driver") or driver_close()

    with pytest.raises(RuntimeError, match="stream stop"):
        client.close()

    assert calls == ["camera", "driver"]
    assert client.raw is None


def test_cap_touch_status_reads_the_lick_detector(module):
    client = connect(
        module,
        hfdm_channel_map(lick_min_on_level=250, lick_on_delta=120, lick_off_delta=90),
    )
    try:
        for _ in range(4):
            module.threshold_sample("lick", 100)

        status = client.get_cap_touch_status()

        assert status["enabled"]
        assert status["channel"] == CAP_TOUCH_INPUT_CHANNEL
        assert status["pin"] == CAP_TOUCH_INPUT_PIN
        assert status["last_value"] == 100
        assert status["baseline"] == 100
        assert status["rising_high_threshold"] == 250  # max(baseline + 120, 250)
        assert status["rising_low_threshold"] == 190
        assert status["state"] == 0
    finally:
        client.close()


def test_lick_samples_stream_as_cap_touch_analog_when_asked(module):
    client = connect(module, hfdm_channel_map(lick_raw=True))
    try:
        received = collect(client)
        module.threshold_sample("lick", 300)
        wait_until(lambda: ("analog_input", CAP_TOUCH_INPUT_CHANNEL, 300) in received)
    finally:
        client.close()


def test_optical_sessions_leave_the_cap_touch_pin_alone(module):
    client = connect(module, hfdm_channel_map(cap_touch=False))
    try:
        assert "lick" not in {channel.name for channel in client.channel_map}
        assert client.get_cap_touch_status()["enabled"] is False
    finally:
        client.close()


def test_module_status_collects_driver_status_snapshots(client):
    status = client.get_module_status()

    assert status["streaming_active"] is False
    assert status["module_info"]["firmware_version"] == 1
    assert status["config_status"]["source"] == "host"
    assert status["analog_input_status"]["interval_us"] == 1000
    assert status["speaker_status"]["pin"] == SPEAKER_GATE_PIN
    assert "buffer_overruns" in status["buffer_stats"]


def test_speaker_gate_pulses_its_output_and_finishes_disabled(module, client):
    collect(client)
    outputs = StreamDIOOutputs(client)

    result = outputs.pulse_speaker_gate(0.001)

    assert result["on"]["event_verified"] and result["on"]["enabled"]
    assert result["off"]["event_verified"] and not result["off"]["enabled"]
    assert module.outputs == 0


def test_active_low_speaker_gate_idles_high(module):
    client = connect(module, hfdm_channel_map(speaker_active_high=False))
    try:
        collect(client)
        speaker = 1 << 4
        assert module.outputs == speaker  # initial level: amplifier disabled
        outputs = StreamDIOOutputs(client, speaker_gate_active_high=False)

        outputs.speaker_gate(True)
        assert module.outputs == 0
        outputs.speaker_gate(False)
        assert module.outputs == speaker
    finally:
        client.close()


def test_buffer_stats_outlive_the_stream(client):
    collect(client)
    client.stop_streaming()
    stats = client.get_buffer_stats()
    assert stats["frames"] >= 0 and "frames_lost" in stats


def test_no_input_reaches_the_bpod(client):
    # Nothing in the state machine uses them, and an event byte during the Bpod's
    # module-info read would corrupt that reply.
    assert client.map_info()["n_bpod_events"] == 0
