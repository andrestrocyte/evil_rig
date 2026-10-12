"""Cooperative operator stop and finite execution deadlines."""

from __future__ import annotations

import math
import os
from pathlib import Path
import shutil
import threading
import time
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")


class OperatorStop(KeyboardInterrupt):
    """A stop requested by the launcher; use the normal cleanup/finalization path."""


class RunControl:
    def __init__(self, *, storage_paths: Iterable[str | Path] = (), reserve_bytes: int = 2 * 1024**3) -> None:
        value = os.environ.get("HF_DM_STOP_REQUEST")
        self.stop_path = Path(value) if value else None
        seconds = float(os.environ.get("HF_DM_RUN_TIMEOUT_S", "0"))
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("HF_DM_RUN_TIMEOUT_S must be finite and non-negative")
        self.deadline = time.perf_counter() + seconds if seconds else None
        self._next_poll = 0.0
        self.reserve_bytes = reserve_bytes
        self._storage_paths = tuple(dict.fromkeys(Path(path).expanduser().resolve() for path in storage_paths))
        self._next_storage_check = 0.0

    def check(self) -> None:
        now = time.perf_counter()
        if self.deadline is not None and now >= self.deadline:
            raise TimeoutError("Run exceeded its configured time limit; stopping and saving data")
        if now >= self._next_poll:
            self._next_poll = now + 0.1
            if self.stop_path is not None and self.stop_path.exists():
                raise OperatorStop("Operator requested stop")
        if now >= self._next_storage_check:
            self._next_storage_check = now + 2.0
            checked_devices = set()
            for path in self._storage_paths:
                existing = path
                while not existing.exists() and existing.parent != existing:
                    existing = existing.parent
                device = existing.stat().st_dev
                if device in checked_devices:
                    continue
                checked_devices.add(device)
                free = shutil.disk_usage(existing).free
                if free < self.reserve_bytes:
                    raise OSError(
                        f"Low disk space at {path}: {free / 1024**3:.2f} GiB free; "
                        f"stopping to preserve {self.reserve_bytes / 1024**3:.2f} GiB for data finalization"
                    )


def bounded_close(callback: Callable[[], T], *, timeout_s: float = 3.0) -> T:
    """Limit a driver's blocking teardown; callers must retain the interlock on failure."""
    done = threading.Event()
    results: list[T] = []
    errors: list[BaseException] = []

    def close() -> None:
        try:
            results.append(callback())
        except BaseException as exc:
            errors.append(exc)
        finally:
            done.set()

    threading.Thread(target=close, name="bpod-close", daemon=True).start()
    if not done.wait(timeout_s):
        raise TimeoutError(f"Bpod cleanup did not finish within {timeout_s:g} seconds")
    if errors:
        raise errors[0]
    return results[0]
