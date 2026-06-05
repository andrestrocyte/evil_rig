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
    SPEAKER_NOISE_SAMPLE_INTERVAL_US,
)
from hf_dm.streamdio.client import StreamDIOClient


class StreamDIOOutputs:
    """Actuate rig outputs and require firmware ACKs/physical output events."""

    def __init__(
        self,
        dio: StreamDIOClient,
        *,
        cap_touch_power_channel: int = CAP_TOUCH_POWER_OUTPUT_CHANNEL,
        speaker_sample_interval_us: int = SPEAKER_NOISE_SAMPLE_INTERVAL_US,
    ) -> None:
        self.dio = dio
        self.cap_touch_power_channel = int(cap_touch_power_channel)
        self.speaker_sample_interval_us = int(speaker_sample_interval_us)

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
        return self.dio.set_speaker_noise(
            enabled,
            sample_interval_us=self.speaker_sample_interval_us,
        )

    def pulse_speaker_gate(self, duration_s: float) -> dict[str, Any]:
        on = self.speaker_gate(True)
        wait_until_perf_counter(time.perf_counter() + max(0.0, float(duration_s)))
        active_status = {
            str(key): value.item() if hasattr(value, "item") else value
            for key, value in dict(self.dio.raw.get_speaker_status()).items()
        }
        samples_before = int(on["status"].get("samples_generated", 0))
        samples_after = int(active_status.get("samples_generated", 0))
        if duration_s > 0.0 and samples_after <= samples_before:
            raise RuntimeError(
                "StreamDIO speaker was enabled but generated no white-noise samples"
            )
        off = self.speaker_gate(False)
        return {
            "on": on,
            "active_status": active_status,
            "samples_generated_during_pulse": samples_after - samples_before,
            "off": off,
        }

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
