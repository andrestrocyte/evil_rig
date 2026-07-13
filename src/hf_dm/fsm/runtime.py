"""Thread-safe host view of one active bpod-core trial."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from hf_dm.streamdio.wheel import Direction

Side = Literal["left", "right"]
Shape = Literal["star", "circle"]


@dataclass
class TrialRuntime:
    trial_index: int
    force_direction: Direction | None
    choice_target_direction: Direction
    star_direction: Direction
    circle_direction: Direction
    star_side: Side
    circle_side: Side
    trial_mode: str = "free"
    forced_shape: Shape | None = None
    modality: str = "force"
    high_reward_s: float = 0.08
    low_reward_s: float = 0.04
    high_delay_s: float = 0.0
    low_delay_s: float = 0.0
    star_reward_delivered: bool = True
    risk_star_probability: float | None = None
    task_kind: Literal["decision", "centering"] = "decision"
    centering_start_position_deg: float | None = None
    phase: str = "starting"
    phase_started_mono_s: float = field(default_factory=time.perf_counter)
    error: str | None = None
    chosen_shape: Shape | None = None
    choice_side: Side | None = None
    wheel_choice: Direction | None = None
    choice_detected_mono_s: float | None = None
    choice_closed: bool = False
    state_entries: list[dict[str, Any]] = field(default_factory=list)
    output_results: list[dict[str, Any]] = field(default_factory=list)
    pump_actual_on_mono_s: float | None = None
    _condition: threading.Condition = field(
        default_factory=threading.Condition,
        init=False,
        repr=False,
    )

    def enter(
        self,
        phase: str,
        *,
        softcode: int,
        output_result: Any = None,
    ) -> None:
        now = time.perf_counter()
        entry = {
            "phase": str(phase),
            "softcode": int(softcode),
            "host_mono_s": now,
        }
        with self._condition:
            self.phase = str(phase)
            self.phase_started_mono_s = now
            self.state_entries.append(entry)
            if output_result is not None:
                self.output_results.append(
                    {
                        **entry,
                        "result": output_result,
                    }
                )
            self._condition.notify_all()

    def commit_choice(
        self,
        *,
        shape: Shape,
        side: Side,
        wheel_choice: Direction,
        detected_mono_s: float,
    ) -> bool:
        with self._condition:
            if self.choice_closed or self.chosen_shape is not None:
                return False
            self.chosen_shape = shape
            self.choice_side = side
            self.wheel_choice = wheel_choice
            self.choice_detected_mono_s = float(detected_mono_s)
            return True

    def close_choice(self, *, clear: bool = False) -> None:
        with self._condition:
            self.choice_closed = True
            if clear:
                self.chosen_shape = None
                self.choice_side = None
                self.wheel_choice = None
                self.choice_detected_mono_s = None

    def fail(self, exc: BaseException) -> None:
        with self._condition:
            self.error = f"{type(exc).__name__}: {exc}"
            self._condition.notify_all()

    def record_pump_on(self, monotonic_s: float) -> None:
        """Record the verified pump-on edge under the runtime lock."""

        with self._condition:
            self.pump_actual_on_mono_s = float(monotonic_s)
            self._condition.notify_all()

    def pump_open_duration(self, off_monotonic_s: float) -> float | None:
        """Return verified pump-open time from a coherent runtime snapshot."""

        with self._condition:
            if self.pump_actual_on_mono_s is None:
                return None
            return float(off_monotonic_s) - self.pump_actual_on_mono_s

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            return {
                "phase": self.phase,
                "phase_started_mono_s": self.phase_started_mono_s,
                "error": self.error,
                "chosen_shape": self.chosen_shape,
                "choice_side": self.choice_side,
                "wheel_choice": self.wheel_choice,
                "choice_target_direction": self.choice_target_direction,
                "star_direction": self.star_direction,
                "circle_direction": self.circle_direction,
                "force_output_active": self.force_direction is not None,
                "force_output_direction": self.force_direction,
                "trial_mode": self.trial_mode,
                "forced_shape": self.forced_shape,
                "modality": self.modality,
                "high_reward_s": self.high_reward_s,
                "low_reward_s": self.low_reward_s,
                "high_delay_s": self.high_delay_s,
                "low_delay_s": self.low_delay_s,
                "star_reward_delivered": self.star_reward_delivered,
                "risk_star_probability": self.risk_star_probability,
                "task_kind": self.task_kind,
                "centering_start_position_deg": self.centering_start_position_deg,
                "choice_detected_mono_s": self.choice_detected_mono_s,
                "choice_closed": self.choice_closed,
                "state_entries": list(self.state_entries),
                "output_results": list(self.output_results),
                "pump_actual_on_mono_s": self.pump_actual_on_mono_s,
            }
