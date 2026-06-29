"""Append-only local subject weight and water records."""

from __future__ import annotations

import json
import math
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

_LOG_LOCK = threading.RLock()


@dataclass(frozen=True)
class SubjectLogEntry:
    entry_id: str
    subject_id: str
    date: str
    entry_type: str
    recorded_wall_s: float
    weight_g: float | None
    water_target_ml: float | None
    water_delivered_ml: float | None
    session_id: str | None = None
    notes: str | None = None


def subject_log_path(subjects_dir: Path | str, subject_id: str) -> Path:
    subject = _validate_subject_id(subject_id)
    return Path(subjects_dir).expanduser().resolve() / subject / "water_log.jsonl"


def append_weight_entry(
    subjects_dir: Path | str,
    subject_id: str,
    *,
    weight_g: float,
    water_target_ml: float | None = None,
    entry_date: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    weight = _positive_float(weight_g, "weight_g")
    target = (
        None
        if water_target_ml is None
        else _non_negative_float(water_target_ml, "water_target_ml")
    )
    entry = SubjectLogEntry(
        entry_id=f"weight:{uuid.uuid4()}",
        subject_id=_validate_subject_id(subject_id),
        date=_validate_date(entry_date or date.today().isoformat()),
        entry_type="weight",
        recorded_wall_s=time.time(),
        weight_g=weight,
        water_target_ml=target,
        water_delivered_ml=None,
        notes=str(notes) if notes not in (None, "") else None,
    )
    return append_subject_entry(subjects_dir, entry)


def append_session_water_entry(
    subjects_dir: Path | str,
    subject_id: str,
    *,
    water_delivered_ul: float,
    session_id: str,
    water_target_ml: float | None = None,
    entry_date: str | None = None,
) -> dict[str, Any]:
    delivered_ul = _non_negative_float(water_delivered_ul, "water_delivered_ul")
    target = (
        None
        if water_target_ml is None
        else _non_negative_float(water_target_ml, "water_target_ml")
    )
    canonical_session = str(Path(session_id).expanduser().resolve())
    entry = SubjectLogEntry(
        entry_id=f"session_water:{canonical_session}",
        subject_id=_validate_subject_id(subject_id),
        date=_validate_date(entry_date or date.today().isoformat()),
        entry_type="session_water",
        recorded_wall_s=time.time(),
        weight_g=None,
        water_target_ml=target,
        water_delivered_ml=delivered_ul / 1000.0,
        session_id=canonical_session,
    )
    return append_subject_entry(subjects_dir, entry)


def append_subject_entry(
    subjects_dir: Path | str,
    entry: SubjectLogEntry,
) -> dict[str, Any]:
    path = subject_log_path(subjects_dir, entry.subject_id)
    row = asdict(entry)
    with _LOG_LOCK:
        existing_ids = {
            str(item.get("entry_id"))
            for item in load_subject_log(path.parent.parent, entry.subject_id)
            if item.get("entry_id")
        }
        if entry.entry_id in existing_ids:
            return {"path": str(path), "entry": row, "appended": False}
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    return {"path": str(path), "entry": row, "appended": True}


def load_subject_log(
    subjects_dir: Path | str,
    subject_id: str,
) -> list[dict[str, Any]]:
    path = subject_log_path(subjects_dir, subject_id)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid subject log JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Subject log row is not an object at {path}:{line_number}")
            rows.append(row)
    return rows


def _validate_subject_id(value: str) -> str:
    subject = str(value).strip()
    if not subject or Path(subject).name != subject or subject in {".", ".."}:
        raise ValueError("subject_id must be a non-empty path-safe name")
    return subject


def _validate_date(value: str) -> str:
    try:
        return date.fromisoformat(str(value)).isoformat()
    except ValueError as exc:
        raise ValueError("date must use YYYY-MM-DD format") from exc


def _positive_float(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and greater than zero")
    return number


def _non_negative_float(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number
