"""The StreamDIO channel map this rig runs, and evil_rig's numbers for its channels.

The module numbers inputs, outputs and analog channels by their position in the map.
evil_rig keeps the channel numbers it has always recorded (``bringup_constants``), so
recordings, exports and the monitor do not depend on the map's order: the tables here
give each named channel its evil_rig number, and :class:`StreamDIOClient` translates
the module's numbers through them.

Outputs come first and in a fixed order, whatever the session uses: the Bpod state
machine addresses them by their module number (``[n, level]`` serial messages).
"""

from __future__ import annotations

from typing import Any

from streamdio import AnalogIn, AnalogThreshold, DigitalIn, DigitalOut, ModuleConfig, PulseOut

from hf_dm.hardware.bringup_constants import (
    CAP_TOUCH_INPUT_CHANNEL,
    CAP_TOUCH_POWER_OUTPUT_CHANNEL,
    CCW_FORCE_OUTPUT_CHANNEL,
    CW_FORCE_OUTPUT_CHANNEL,
    DEFAULT_CAMERA_EXPOSURE_US,
    DEFAULT_CAMERA_FREQUENCY_HZ,
    DIRECTION_INPUT_CHANNEL,
    FORCE_ANALOG_CHANNEL,
    FORCE_ANALOG_SAMPLE_INTERVAL_US,
    FRAME2TTL_INPUT_CHANNEL,
    INPUT_REFRACTORY_PERIOD_US,
    PULSE_INPUT_CHANNEL,
    PUMP_OUTPUT_CHANNEL,
    SPEAKER_OUTPUT_CHANNEL,
)

CAMERA = "camera"
LICK = "lick"
SPEAKER = "speaker"

# Channel name -> evil_rig channel number, per kind of event.
INPUT_CHANNELS = {
    "wheel_a": PULSE_INPUT_CHANNEL,
    "wheel_b": DIRECTION_INPUT_CHANNEL,
    "f2ttl": FRAME2TTL_INPUT_CHANNEL,  # Frame2TTL (names are at most 8 characters)
    LICK: CAP_TOUCH_INPUT_CHANNEL,
}
OUTPUT_CHANNELS = {
    "pump": PUMP_OUTPUT_CHANNEL,
    "cwforce": CW_FORCE_OUTPUT_CHANNEL,
    "ccwforce": CCW_FORCE_OUTPUT_CHANNEL,
    "cap_pwr": CAP_TOUCH_POWER_OUTPUT_CHANNEL,
    SPEAKER: SPEAKER_OUTPUT_CHANNEL,
}
ANALOG_CHANNELS = {
    "force": FORCE_ANALOG_CHANNEL,
    LICK: CAP_TOUCH_INPUT_CHANNEL,  # the lick detector's raw samples (lick_raw=True)
}

# Both analog channels sample as the fixed-layout firmware did (the Teensy core's
# analogRead defaults): the force calibration and the cap-touch thresholds are in
# these counts.
ANALOG_RESOLUTION = 10
ANALOG_AVERAGING = 4


def _outputs(speaker_active_high: bool) -> list:
    return [
        DigitalOut("pump", "DO01"),
        DigitalOut("cwforce", "DO03"),
        DigitalOut("ccwforce", "DO05"),
        DigitalOut("cap_pwr", "J28-9", bpod_control=False),  # Teensy pin 15
        # The gate of the amplifier playing the PC sound card's white noise (pin 26).
        DigitalOut(SPEAKER, "J27-14", initial=0 if speaker_active_high else 1),
        PulseOut(
            CAMERA,
            "J27-9",  # Teensy pin 32
            frequency_hz=DEFAULT_CAMERA_FREQUENCY_HZ,
            width_us=DEFAULT_CAMERA_EXPOSURE_US,
            bpod_control=False,
        ),
    ]


# Module output number (1-based) of each evil_rig output, for the Bpod's serial
# messages. Fixed: _outputs() does not depend on the session.
MODULE_OUTPUT_NUMBERS = {
    OUTPUT_CHANNELS[channel.name]: number
    for number, channel in enumerate(_outputs(True), start=1)
    if channel.name in OUTPUT_CHANNELS
}


def hfdm_channel_map(
    *,
    cap_touch: bool = True,
    frame2ttl: bool = True,
    lick_min_on_level: int | None = None,
    lick_on_delta: int | None = None,
    lick_off_delta: int | None = None,
    lick_raw: bool = False,
    speaker_active_high: bool = True,
) -> ModuleConfig:
    """The map for a session.

    :param cap_touch: the capacitive lick detector on pin 25 (``licks.source``
        ``capacitive``); off, the pin is not used
    :param frame2ttl: the Frame2TTL input on DI05
    :param lick_min_on_level: the lowest level a lick can fire at; ``None`` keeps
        the default (700)
    :param lick_on_delta: excursion from the baseline that fires a lick (350)
    :param lick_off_delta: excursion that releases it (160)
    :param lick_raw: stream the lick detector's samples too (cap-touch diagnostic)
    :param speaker_active_high: the speaker gate's level that enables the amplifier
    """
    channels: list = _outputs(speaker_active_high)
    # Inputs reach the stream only. The state machine uses none of them as events
    # (evil_rig decodes licks, the wheel and Frame2TTL from the stream), the wheel's
    # edges would exceed the Bpod's module-event rate (~2,000/s), and an event byte
    # sent while the Bpod reads the module's info would corrupt that reply.
    channels += [
        DigitalIn("wheel_a", "DI01", refractory_us=INPUT_REFRACTORY_PERIOD_US, to_bpod=False),
        DigitalIn("wheel_b", "DI03", refractory_us=INPUT_REFRACTORY_PERIOD_US, to_bpod=False),
    ]
    if frame2ttl:
        channels.append(
            DigitalIn("f2ttl", "DI05", refractory_us=INPUT_REFRACTORY_PERIOD_US, to_bpod=False)
        )
    channels.append(
        AnalogIn(
            "force",
            "J28-10",  # Teensy pin 14
            interval_us=FORCE_ANALOG_SAMPLE_INTERVAL_US,
            resolution=ANALOG_RESOLUTION,
            averaging=ANALOG_AVERAGING,
        )
    )
    if cap_touch:
        thresholds = {
            key: int(value)
            for key, value in (
                ("min_on_level", lick_min_on_level),
                ("on_delta", lick_on_delta),
                ("off_delta", lick_off_delta),
            )
            if value is not None
        }
        # A touch raises the sensor's level. Rising only: with "either", a baseline
        # caught during a touch (at start-up, or after the power-switch blanking)
        # would read every release as a falling "touch".
        channels.append(
            AnalogThreshold(
                LICK,
                "J27-15",  # Teensy pin 25
                polarity="rising",
                blank_output="cap_pwr",
                stream_raw=lick_raw,
                to_bpod=False,
                resolution=ANALOG_RESOLUTION,
                averaging=ANALOG_AVERAGING,
                **thresholds,
            )
        )
    return ModuleConfig(channels)


def session_channel_map(config: Any, *, lick_raw: bool = False) -> ModuleConfig:
    """The map for a :class:`~hf_dm.config.SessionConfig`."""
    licks = config.licks
    return hfdm_channel_map(
        cap_touch=licks.source == "capacitive",
        frame2ttl=bool(config.frame2ttl.enabled),
        lick_min_on_level=licks.cap_touch_min_high_threshold,
        lick_on_delta=licks.cap_touch_delta_high,
        lick_off_delta=licks.cap_touch_delta_low,
        lick_raw=lick_raw,
        speaker_active_high=bool(config.speaker.gate_active_high),
    )
