"""Small asynchronous bpod-core FSM simulator used by the task simulation."""

from __future__ import annotations

import queue
import threading
import time
from typing import Any, Callable


class SimulatedBpod:
    port = "SIM_BPOD"

    def __init__(self) -> None:
        self._handler: Callable[[int], None] | None = None
        self._serial_handler: Callable[[str, int], None] | None = None
        self._input: queue.Queue[int] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._completed: queue.Queue[list[dict[str, Any]]] = queue.Queue()

    def reset_session_clock(self) -> None:
        return

    def set_softcode_handler(self, handler: Callable[[int], None] | None) -> None:
        self._handler = handler

    def set_serial_handler(self, handler: Callable[[str, int], None] | None) -> None:
        self._serial_handler = handler

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def run(self, fsm, *, trial_number: int | None = None, validate: bool = True) -> None:
        del validate
        self.wait()
        while not self._input.empty():
            try:
                self._input.get_nowait()
            except queue.Empty:
                break
        self._thread = threading.Thread(
            target=self._run_fsm,
            args=(fsm, int(trial_number or 0)),
            name=f"sim-bpod-trial-{trial_number or 0}",
            daemon=True,
        )
        self._thread.start()

    def _run_fsm(self, fsm, trial_number: int) -> None:
        rows: list[dict[str, Any]] = []
        state_name = next(iter(fsm.states))
        trial_start = time.perf_counter()
        while not state_name.startswith(">"):
            state = fsm.states[state_name]
            entered = time.perf_counter()
            rows.append(
                {
                    "trial": trial_number,
                    "type": "StateStart",
                    "event": "StateStart",
                    "state": state_name,
                    "time_host_s": entered,
                    "time_trial_s": entered - trial_start,
                }
            )
            outgoing = state.actions.get("SoftCode")
            if self._serial_handler is not None:
                for output_name, message_id in state.actions.items():
                    if output_name == "SoftCode":
                        continue
                    self._serial_handler(str(output_name), int(message_id))
            if outgoing is not None and self._handler is not None:
                self._handler(int(outgoing))

            deadline = entered + float(state.timer)
            target: str | None = None
            while target is None:
                remaining = deadline - time.perf_counter()
                if remaining <= 0.0:
                    event_name = "Tup"
                    target = state.transitions.get(event_name, state_name)
                    rows.append(
                        {
                            "trial": trial_number,
                            "type": "InputEvent",
                            "event": event_name,
                            "state": state_name,
                            "time_host_s": time.perf_counter(),
                            "time_trial_s": time.perf_counter() - trial_start,
                        }
                    )
                    break
                try:
                    softcode = self._input.get(timeout=min(remaining, 0.01))
                except queue.Empty:
                    continue
                event_name = f"SoftCode{softcode}"
                if event_name not in state.transitions:
                    continue
                target = state.transitions[event_name]
                rows.append(
                    {
                        "trial": trial_number,
                        "type": "InputEvent",
                        "event": event_name,
                        "state": state_name,
                        "time_host_s": time.perf_counter(),
                        "time_trial_s": time.perf_counter() - trial_start,
                    }
                )
            rows.append(
                {
                    "trial": trial_number,
                    "type": "StateEnd",
                    "event": "StateEnd",
                    "state": state_name,
                    "time_host_s": time.perf_counter(),
                    "time_trial_s": time.perf_counter() - trial_start,
                }
            )
            state_name = str(target)
        self._completed.put(rows)

    def send_softcode(self, softcode: int) -> None:
        self._input.put(int(softcode))

    def wait(self) -> None:
        if self._thread is not None:
            self._thread.join()

    def get_data(self, *, concat: bool = True, rechunk: bool = False, lazy: bool = False):
        del concat, rechunk, lazy
        return self._completed.get()

    def close(self) -> None:
        self.wait()
        self._handler = None
