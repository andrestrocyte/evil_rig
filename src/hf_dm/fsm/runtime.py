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
    force_direction: Direction
    star_side: Side
    circle_side: Side
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

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            return {
                "phase": self.phase,
                "phase_started_mono_s": self.phase_started_mono_s,
                "error": self.error,
                "chosen_shape": self.chosen_shape,
                "choice_side": self.choice_side,
                "wheel_choice": self.wheel_choice,
                "choice_detected_mono_s": self.choice_detected_mono_s,
                "choice_closed": self.choice_closed,
                "state_entries": list(self.state_entries),
                "output_results": list(self.output_results),
                "pump_actual_on_mono_s": self.pump_actual_on_mono_s,
            }
