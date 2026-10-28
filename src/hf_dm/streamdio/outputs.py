"""Verified fast outputs through StreamDIO USB1."""

from __future__ import annotations

import time
from typing import Any

from hf_dm.fsm.relay import wait_until_perf_counter
from hf_dm.hardware.bringup_constants import (
    CAP_TOUCH_POWER_OUTPUT_CHANNEL,
    CCW_FORCE_OUTPUT_CHANNEL,
    CW_FORCE_OUTPUT_CHANNEL,
    PUMP_OUTPUT_CHANNEL,
    SPEAKER_OUTPUT_CHANNEL,
)
from hf_dm.streamdio.client import StreamDIOClient


class StreamDIOOutputs:
    """Actuate rig outputs and require firmware ACKs/physical output events."""

    def __init__(
        self,
        dio: StreamDIOClient,
        *,
        cap_touch_power_channel: int = CAP_TOUCH_POWER_OUTPUT_CHANNEL,
        speaker_gate_active_high: bool = True,
    ) -> None:
        self.dio = dio
        self.cap_touch_power_channel = int(cap_touch_power_channel)
        self.speaker_gate_active_high = bool(speaker_gate_active_high)

    def set_channel_state(self, channel: int, state: bool) -> dict[str, Any]:
        return self.dio.set_output_channel(channel, state)

    def set_force_direction(self, direction: str | None) -> list[dict[str, Any]]:
        if direction == "cw":
            return [
                self.set_channel_state(CCW_FORCE_OUTPUT_CHANNEL, False),
                self.set_channel_state(CW_FORCE_OUTPUT_CHANNEL, True),
            ]
        if direction == "ccw":
            return [
                self.set_channel_state(CW_FORCE_OUTPUT_CHANNEL, False),
                self.set_channel_state(CCW_FORCE_OUTPUT_CHANNEL, True),
            ]
        return self.force_off()

    def force_off(self) -> list[dict[str, Any]]:
        return [
            self.set_channel_state(CW_FORCE_OUTPUT_CHANNEL, False),
            self.set_channel_state(CCW_FORCE_OUTPUT_CHANNEL, False),
        ]

    def pump_off(self) -> dict[str, Any]:
        return self.set_channel_state(PUMP_OUTPUT_CHANNEL, False)

    def pump_on(self) -> dict[str, Any]:
        return self.set_channel_state(PUMP_OUTPUT_CHANNEL, True)

    def pulse_pump(self, duration_s: float) -> dict[str, Any]:
        on = self.pump_on()
        wait_until_perf_counter(time.perf_counter() + max(0.0, float(duration_s)))
        off = self.pump_off()
        return {"on": on, "off": off}

    def cap_touch_power(self, enabled: bool) -> dict[str, Any]:
        return self.set_channel_state(self.cap_touch_power_channel, enabled)

    def speaker_gate(self, enabled: bool) -> dict[str, Any]:
        """Enable or disable the amplifier playing the PC sound card's white noise."""
        level = bool(enabled) == self.speaker_gate_active_high
        result = self.set_channel_state(SPEAKER_OUTPUT_CHANNEL, level)
        result["enabled"] = bool(enabled)
        return result

    def pulse_speaker_gate(self, duration_s: float) -> dict[str, Any]:
        on = self.speaker_gate(True)
        wait_until_perf_counter(time.perf_counter() + max(0.0, float(duration_s)))
        off = self.speaker_gate(False)
        return {"on": on, "off": off}

    def all_off(self) -> dict[str, Any]:
        result = self.task_outputs_off()
        result["cap_touch_power"] = self.cap_touch_power(False)
        return result

    def task_outputs_off(self) -> dict[str, Any]:
        """Make reward/effort/noise safe while leaving cap-touch powered."""
        speaker = self.speaker_gate(False)
        pump = self.pump_off()
        force = self.force_off()
        return {
            "speaker": speaker,
            "pump": pump,
            "force": force,
        }
