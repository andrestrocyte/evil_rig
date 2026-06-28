"""Wheel pulse integration: same logic as screen_shape_force_discount_task."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Literal

from hf_dm.hardware.bringup_constants import (
    DEFAULT_ENCODER_TICKS_PER_REV,
    DEFAULT_GAIN_DEG_PER_MM,
    DEFAULT_WHEEL_RADIUS_MM,
    DIRECTION_INPUT_CHANNEL,
    FORCE_ANALOG_CHANNEL,
    PULSE_INPUT_CHANNEL,
)

Direction = Literal["cw", "ccw"]


@dataclass
class WheelKinematics:
    wheel_radius_mm: float = DEFAULT_WHEEL_RADIUS_MM
    encoder_ticks_per_rev: int = DEFAULT_ENCODER_TICKS_PER_REV
    gain_deg_per_mm: float = DEFAULT_GAIN_DEG_PER_MM
    pulse_channel: int = PULSE_INPUT_CHANNEL
    direction_channel: int = DIRECTION_INPUT_CHANNEL
    force_channel: int = FORCE_ANALOG_CHANNEL

    @property
    def mm_per_tick(self) -> float:
        return (2.0 * math.pi * self.wheel_radius_mm) / float(self.encoder_ticks_per_rev)

    @property
    def deg_per_pulse(self) -> float:
        return self.mm_per_tick * self.gain_deg_per_mm


class WheelAccumulator:
    def __init__(self, kinematics: WheelKinematics | None = None):
        self.cfg = kinematics or WheelKinematics()
        self.lock = threading.Lock()
        self.direction_is_high = False
        self.last_pulse_monotonic = time.perf_counter()
        self.reset_trial()

    def reset_trial(self) -> None:
        with self.lock:
            self.net_pulses = 0
            self.cw_pulses = 0
            self.ccw_pulses = 0
            self.force_samples = 0
            self.force_sum = 0.0
            self.force_last: float | None = None
            self.force_min: float | None = None
            self.force_max: float | None = None
            self.force_values: list[float] = []
            self.last_pulse_monotonic = time.perf_counter()

    def handle_event(
        self,
        event_type: str,
        channel: int,
        value: int,
        timestamp_us: int,
        relative_time: float,
    ) -> None:
        del timestamp_us, relative_time
        with self.lock:
            if event_type == "dio_input":
                if channel == self.cfg.direction_channel:
                    self.direction_is_high = bool(value)
                elif channel == self.cfg.pulse_channel and value == 1:
                    delta = 1 if self.direction_is_high else -1
                    self.net_pulses += delta
                    if delta > 0:
                        self.cw_pulses += 1
                    else:
                        self.ccw_pulses += 1
                    self.last_pulse_monotonic = time.perf_counter()
            elif event_type == "analog_input" and channel == self.cfg.force_channel:
                numeric_value = float(value)
                self.force_samples += 1
                self.force_sum += numeric_value
                self.force_last = numeric_value
                self.force_values.append(numeric_value)
                self.force_min = (
                    numeric_value if self.force_min is None else min(self.force_min, numeric_value)
                )
                self.force_max = (
                    numeric_value if self.force_max is None else max(self.force_max, numeric_value)
                )

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            force_mean = None
            force_p95 = None
            if self.force_samples:
                force_mean = self.force_sum / float(self.force_samples)
                force_p95 = _percentile(self.force_values, 95.0)
            return {
                "net_pulses": self.net_pulses,
                "cw_pulses": self.cw_pulses,
                "ccw_pulses": self.ccw_pulses,
                "force_samples": self.force_samples,
                "force_last": self.force_last,
                "force_mean": force_mean,
                "force_p95": force_p95,
                "force_min": self.force_min,
                "force_max": self.force_max,
                "quiescent_for_s": time.perf_counter() - self.last_pulse_monotonic,
            }

    def signed_position_deg(self) -> float:
        snap = self.snapshot()
        return snap["net_pulses"] * self.cfg.deg_per_pulse


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * (max(0.0, min(100.0, percentile)) / 100.0)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return float(ordered[lower])
    fraction = rank - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)
