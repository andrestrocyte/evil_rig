"""
Rig wiring and evil_rig's StreamDIO channel numbers.

The channel numbers are evil_rig's own: recordings, exports and the monitor use them.
They are the numbers of the fixed-layout firmware the rig started on; the StreamDIO
channel map (``hf_dm.streamdio.channel_map``) assigns them to its named channels.
"""

from dataclasses import dataclass


# The module's delivery heartbeat (streamdio ``start_latency_heartbeat``): emitted by
# the firmware on its own clock, so host-side jitter is delivery jitter.
STREAMDIO_TIMING_HEARTBEAT_EVENT_CODE = 1012
STREAMDIO_TIMING_HEARTBEAT_PERIOD_US = 100_000

# Serial / USB defaults (all bringups)
DEFAULT_BAUDRATE = 1_312_500

# Rotary encoder + force
PULSE_INPUT_CHANNEL = 1          # DI01 / pin 2
DIRECTION_INPUT_CHANNEL = 3    # DI03 / pin 4
FORCE_ANALOG_CHANNEL = 13      # AI13 / Teensy pin 14
FORCE_ANALOG_PIN = 14
FORCE_ANALOG_SAMPLE_INTERVAL_US = 1000
INPUT_REFRACTORY_PERIOD_US = 300

# Capacitive touch
CAP_TOUCH_POWER_OUTPUT_CHANNEL = 13  # output channel 13 / physical pin 15
CAP_TOUCH_POWER_PIN = 15
CAP_TOUCH_INPUT_CHANNEL = 14         # input channel 14 / physical pin 25
CAP_TOUCH_INPUT_PIN = 25
LICK_INPUT_CHANNEL = CAP_TOUCH_INPUT_CHANNEL
CAP_TOUCH_DEFAULT_MIN_HIGH_THRESHOLD = 700
CAP_TOUCH_DEFAULT_DELTA_HIGH = 350
CAP_TOUCH_DEFAULT_DELTA_LOW = 160

# Force-direction + pump outputs
PUMP_OUTPUT_CHANNEL = 1        # pin 23
CW_FORCE_OUTPUT_CHANNEL = 3    # pin 21
CCW_FORCE_OUTPUT_CHANNEL = 5   # pin 19

# Speaker gate: Teensy pin 26 enables the amplifier that plays the PC sound card's
# white noise. Output 14 is evil_rig's number for it (the fixed-layout firmware had
# no output channel on this pin).
SPEAKER_GATE_PIN = 26
SPEAKER_OUTPUT_CHANNEL = 14
SERIAL_GATE_ON_MESSAGE_ID = 210
SERIAL_GATE_OFF_MESSAGE_ID = 211

# Bpod -> StreamDIO state markers.
# Each preloaded message (IDs 50+) sends [ord('S'), state_id] to StreamDIO so the
# module timestamps the state entry.
STREAMDIO_BPOD_STATE_OPCODE = 83  # ord('S')
STREAMDIO_STATE_MESSAGE_ID_BASE = 50
STREAMDIO_STATE_MARKERS = (
    ("Quiescence", 1),
    ("ChoiceWindow", 2),
    ("ChoiceBlankHigh", 3),
    ("ChoiceBlankLow", 4),
    ("RewardHigh", 5),
    ("RewardLow", 6),
    ("PostRewardHigh", 7),
    ("PostRewardLow", 8),
    ("TimeoutNoise", 9),
    ("PostTimeout", 10),
    ("Abort", 11),
)
STREAMDIO_STATE_MARKER_IDS = {name: state_id for name, state_id in STREAMDIO_STATE_MARKERS}
STREAMDIO_STATE_MARKER_NAMES = {state_id: name for name, state_id in STREAMDIO_STATE_MARKERS}
STREAMDIO_STATE_MARKER_MESSAGE_IDS = {
    name: STREAMDIO_STATE_MESSAGE_ID_BASE + index
    for index, (name, _state_id) in enumerate(STREAMDIO_STATE_MARKERS)
}
STREAMDIO_STATE_MARKER_IDS_BY_MESSAGE_ID = {
    STREAMDIO_STATE_MESSAGE_ID_BASE + index: state_id
    for index, (_name, state_id) in enumerate(STREAMDIO_STATE_MARKERS)
}

# Camera TTL via StreamDIO (Teensy pin 32)
CAMERA_TRIGGER_PIN = 32
DEFAULT_CAMERA_FREQUENCY_HZ = 30
DEFAULT_CAMERA_EXPOSURE_US = 10_000

# Frame2TTL v3 TTL input
FRAME2TTL_INPUT_CHANNEL = 5      # DI05 / Teensy pin 6
FRAME2TTL_INPUT_PIN = 6

# Wheel kinematics
DEFAULT_WHEEL_RADIUS_MM = 31.5
DEFAULT_ENCODER_TICKS_PER_REV = 1024
DEFAULT_GAIN_DEG_PER_MM = 4.0

# Bpod serial relay channel
DEFAULT_BPOD_SERIAL_CHANNEL = 1


@dataclass(frozen=True)
class BringupPorts:
    """COM port selection; AUTO finds the ports."""

    bpod_port: str = "AUTO"
    dio_config_port: str = "AUTO"
    dio_stream_port: str = "AUTO"
    baudrate: int = DEFAULT_BAUDRATE
    bpod_serial_channel: int = DEFAULT_BPOD_SERIAL_CHANNEL
