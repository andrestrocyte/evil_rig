"""Persistent fail-closed latch for incomplete rig hardware cleanup."""

from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path
from typing import Any

from hf_dm.durable_io import fsync_best_effort


_INTERLOCK_ENV = "HF_DM_SAFETY_INTERLOCK_PATH"


class HardwareSafetyInterlockError(RuntimeError):
    """Raised when a previous process did not prove that rig outputs were safe."""


def safety_interlock_path() -> Path:
    override = os.environ.get(_INTERLOCK_ENV)
    if override:
        return Path(override).expanduser().resolve()
    return (Path.home() / ".hf_dm" / "hardware_cleanup_interlock.json").resolve()


def assert_safety_interlock_clear(path: Path | None = None) -> None:
    target = safety_interlock_path() if path is None else Path(path)
    if not target.exists():
        return
    try:
        detail = json.loads(target.read_text(encoding="utf-8"))
        state = str(detail.get("state", "unknown"))
        updated = detail.get("updated_wall_time_s", detail.get("armed_wall_time_s"))
    except Exception:
        state = "unreadable"
        updated = None
    raise HardwareSafetyInterlockError(
        "Rig startup blocked by the persistent hardware safety interlock at "
        f"{target} (state={state!r}, updated_wall_time_s={updated!r}). "
        "Verify pump, CW/CCW force, speaker gate, camera triggers, and cap-touch "
        "power are physically off; inspect the recorded cleanup errors; then remove "
        "this file manually before retrying."
    )


def arm_safety_interlock(payload: dict[str, Any], path: Path | None = None) -> Path:
    target = safety_interlock_path() if path is None else Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 1,
        "state": "session_active_cleanup_not_yet_verified",
        "armed_wall_time_s": time.time(),
        "updated_wall_time_s": time.time(),
        "host": platform.node(),
        "pid": os.getpid(),
        **payload,
    }
    encoded = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        assert_safety_interlock_clear(target)
        raise AssertionError("unreachable")
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        fsync_best_effort(handle, path=target)
    return target


def latch_safety_interlock(
    errors: list[dict[str, str]],
    *,
    path: Path | None = None,
) -> Path:
    target = safety_interlock_path() if path is None else Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    previous: dict[str, Any] = {}
    try:
        previous = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        pass
    record = {
        **previous,
        "schema_version": 1,
        "state": "cleanup_failed",
        "updated_wall_time_s": time.time(),
        "cleanup_errors": errors,
    }
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(record, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            fsync_best_effort(handle, path=temporary)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def clear_safety_interlock(path: Path | None = None) -> None:
    target = safety_interlock_path() if path is None else Path(path)
    target.unlink(missing_ok=True)
