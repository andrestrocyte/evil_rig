"""
Hardware constants aligned with MM_PyBpod2 bringup notebooks.

References (repo: MM_PyBpod2/examples/):
  - rotary_encoder_bringup.ipynb
  - screen_bringup_shape_force_discount.ipynb
  - camera_bringup_pin32_ttl.ipynb
  - speaker_bringup_pin27_whitenoise.ipynb
"""

from dataclasses import dataclass


# StreamDIO discovery bytes (stream_dio_protocol.py)
STREAMDIO_CONFIG_DISCOVERY = 222
STREAMDIO_STREAM_DISCOVERY = 223
STREAMDIO_DIO_INPUT_EVENT_BASE = 1
STREAMDIO_DIO_OUTPUT_EVENT_BASE = 29
STREAMDIO_ANALOG_INPUT_EVENT_BASE = 55
STREAMDIO_BPOD_STATE_EVENT_BASE = 100
STREAMDIO_CAP_TOUCH_DIGITIZER_FIRMWARE_VERSION = 3
STREAMDIO_CAP_TOUCH_CONFIG_FIRMWARE_VERSION = 4
STREAMDIO_TIMING_HEARTBEAT_FIRMWARE_VERSION = 5
STREAMDIO_TIMING_HEARTBEAT_EVENT_CODE = 1011
STREAMDIO_TIMING_HEARTBEAT_PERIOD_US = 100_000

# Serial / USB defaults (all bringups)
DEFAULT_BAUDRATE = 1_312_500

# Rotary encoder + force (rotary_encoder_bringup.ipynb)
PULSE_INPUT_CHANNEL = 1          # DI01 / pin 2
DIRECTION_INPUT_CHANNEL = 3    # DI03 / pin 4
FORCE_ANALOG_CHANNEL = 13      # AI13 / Teensy pin 14
FORCE_ANALOG_PIN = 14
FORCE_ANALOG_SAMPLE_INTERVAL_US = 1000
INPUT_REFRACTORY_PERIOD_US = 300

# Capacitive touch (cap_touch_bringup_pin15_power_pin14_touch.ipynb)
CAP_TOUCH_POWER_OUTPUT_CHANNEL = 13  # output channel 13 / physical pin 15
CAP_TOUCH_POWER_PIN = 15
CAP_TOUCH_INPUT_CHANNEL = 14         # input channel 14 / physical pin 25
CAP_TOUCH_INPUT_PIN = 25
LICK_INPUT_CHANNEL = CAP_TOUCH_INPUT_CHANNEL
CAP_TOUCH_DEFAULT_MIN_HIGH_THRESHOLD = 700
CAP_TOUCH_DEFAULT_DELTA_HIGH = 350
CAP_TOUCH_DEFAULT_DELTA_LOW = 160

# Force-direction + pump outputs (screen_bringup_shape_force_discount.ipynb)
PUMP_OUTPUT_CHANNEL = 1        # pin 23
CW_FORCE_OUTPUT_CHANNEL = 3    # pin 21
CCW_FORCE_OUTPUT_CHANNEL = 5   # pin 19

# Speaker gate moved to Teensy pin 26; the original bringup notebook used pin 27.
SPEAKER_GATE_PIN = 26
SPEAKER_GATE_OPCODE = 89         # ord('Y')
SERIAL_GATE_ON_MESSAGE_ID = 210
SERIAL_GATE_OFF_MESSAGE_ID = 211
SPEAKER_NOISE_SAMPLE_INTERVAL_US = 100

# Bpod -> StreamDIO state markers.
# Message IDs 50+ match the MM_PyBpod2 examples; each preloaded message sends
# [ord('S'), state_id] to StreamDIO so the module timestamps the state entry.
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

# Camera TTL via StreamDIO (camera_bringup_pin32_ttl.ipynb defaults)
DEFAULT_CAMERA_FREQUENCY_HZ = 30
DEFAULT_CAMERA_EXPOSURE_US = 10_000

# Frame2TTL v3 TTL input (frame2ttl_v3_bringup_di05_pin6.ipynb)
FRAME2TTL_INPUT_CHANNEL = 5      # DI05 / Teensy pin 6
FRAME2TTL_INPUT_PIN = 6

# Wheel kinematics (screen_shape_force_discount_task.py)
DEFAULT_WHEEL_RADIUS_MM = 31.5
DEFAULT_ENCODER_TICKS_PER_REV = 1024
DEFAULT_GAIN_DEG_PER_MM = 4.0

# Bpod serial relay channel (all bringups)
DEFAULT_BPOD_SERIAL_CHANNEL = 1


@dataclass(frozen=True)
class BringupPorts:
    """COM port selection; use AUTO in the field like the notebooks."""

    bpod_port: str = "AUTO"
    dio_config_port: str = "AUTO"
    dio_stream_port: str = "AUTO"
    baudrate: int = DEFAULT_BAUDRATE
    bpod_serial_channel: int = DEFAULT_BPOD_SERIAL_CHANNEL
