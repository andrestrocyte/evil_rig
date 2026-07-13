"""Persistence and command helpers for interactive animal-use remediation."""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import yaml

from hf_dm.certification import resolve_certification_path
from hf_dm.config import SessionConfig, config_to_dict, validate_session_config
from hf_dm.durable_io import fsync_best_effort, fsync_directory_best_effort

OPERATOR_STATE_SCHEMA_VERSION = 1


def certification_workspace(config: SessionConfig) -> Path:
    """Return the ignored workspace used by the configured certificate."""

    certificate = resolve_certification_path(config)
    if certificate is not None:
        return certificate.parent
    source = config.session_export.source_config_path
    if source:
        return Path(source).expanduser().resolve().parent.parent / "certifications"
    return Path("certifications").resolve()


def ensure_certification_path(config: SessionConfig) -> Path:
    certificate = resolve_certification_path(config)
    if certificate is None:
        certificate = certification_workspace(config) / "rig-certification.json"
        config.animal_use.certification_path = str(certificate)
    return certificate


def operator_state_path(config: SessionConfig) -> Path:
    certificate = resolve_certification_path(config)
    stem = certificate.stem if certificate is not None else "rig-certification"
    return certification_workspace(config) / f"{stem}.operator-state.json"


def audit_log_path(config: SessionConfig) -> Path:
    certificate = resolve_certification_path(config)
    stem = certificate.stem if certificate is not None else "rig-certification"
    return certification_workspace(config) / f"{stem}.gui-events.jsonl"


def effective_config_path(config: SessionConfig) -> Path:
    certificate = resolve_certification_path(config)
    stem = certificate.stem if certificate is not None else "rig-certification"
    return certification_workspace(config) / f".{stem}.effective.yaml"


def load_operator_state(config: SessionConfig) -> dict[str, Any]:
    path = operator_state_path(config)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Operator state must be a JSON object: {path}")
    if payload.get("schema_version") != OPERATOR_STATE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported operator state schema in {path}")
    values = payload.get("values")
    if not isinstance(values, dict):
        raise ValueError(f"Operator state values must be a JSON object: {path}")
    return dict(values)


def apply_operator_state(config: SessionConfig, values: dict[str, Any]) -> None:
    """Apply only the small, explicitly supported operational overlay."""

    calibration = values.get("reward_calibration_path")
    if calibration not in (None, ""):
        config.reward.calibration_path = str(Path(str(calibration)).expanduser().resolve())
    evidence = values.get("physical_evidence_path")
    if evidence not in (None, ""):
        # Evidence is not a SessionConfig field; retain it in the overlay only.
        values["physical_evidence_path"] = str(Path(str(evidence)).expanduser().resolve())
    revision = values.get("hardware_failsafe_revision")
    if revision not in (None, ""):
        config.animal_use.hardware_failsafe_revision = str(revision).strip()
    cap = values.get("max_cumulative_reward_ul")
    if cap not in (None, ""):
        config.welfare.max_cumulative_reward_ul = float(cap)
    subjects = values.get("subjects_dir")
    if subjects not in (None, ""):
        config.welfare.subjects_dir = str(Path(str(subjects)).expanduser().resolve())


def save_operator_state(config: SessionConfig, values: dict[str, Any]) -> Path:
    canonical = {
        "operator": str(values.get("operator") or "").strip(),
        "reward_calibration_path": _absolute_or_none(values.get("reward_calibration_path")),
        "physical_evidence_path": _absolute_or_none(values.get("physical_evidence_path")),
        "hardware_failsafe_revision": str(
            values.get("hardware_failsafe_revision") or ""
        ).strip(),
        "max_cumulative_reward_ul": _positive_or_none(
            values.get("max_cumulative_reward_ul")
        ),
        "subjects_dir": _absolute_or_none(values.get("subjects_dir")),
    }
    apply_operator_state(config, canonical)
    validate_session_config(config)
    path = operator_state_path(config)
    _atomic_write_json(
        path,
        {
            "schema_version": OPERATOR_STATE_SCHEMA_VERSION,
            "updated_wall_time_s": time.time(),
            "values": canonical,
        },
    )
    return path


def append_gui_audit(
    config: SessionConfig,
    event: str,
    *,
    operator: str | None = None,
    detail: Any = None,
) -> Path:
    path = audit_log_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "event_id": str(uuid.uuid4()),
        "wall_time_s": time.time(),
        "event": str(event),
        "operator": None if operator is None else str(operator).strip() or None,
        "detail": detail,
    }
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        handle.flush()
        fsync_best_effort(handle, path=path)
    return path


def write_json_artifact(path: Path | str, payload: dict[str, Any]) -> Path:
    """Durably replace a GUI-authored JSON artifact."""

    target = Path(path).expanduser().resolve()
    _atomic_write_json(target, payload)
    return target


def write_effective_config(config: SessionConfig) -> Path:
    """Write the exact in-memory configuration used by hardware workflows."""

    validate_session_config(config)
    raw = config_to_dict(config)
    raw["results_dir"] = str(Path(config.results_dir).expanduser().resolve())
    raw["camera"]["save_dir"] = str(Path(config.camera.save_dir).expanduser().resolve())
    raw["welfare"]["subjects_dir"] = str(
        Path(config.welfare.subjects_dir).expanduser().resolve()
    )
    if config.reward.calibration_path:
        raw["reward"]["calibration_path"] = str(
            Path(config.reward.calibration_path).expanduser().resolve()
        )
    certificate = resolve_certification_path(config)
    if certificate is None:
        certificate = certification_workspace(config) / "rig-certification.json"
        config.animal_use.certification_path = str(certificate)
    raw["animal_use"]["certification_path"] = str(certificate)
    raw["session_export"]["source_config_path"] = None
    raw["session_export"]["cli_overrides"] = []
    path = effective_config_path(config)
    _atomic_write_text(path, yaml.safe_dump(raw, sort_keys=False))
    return path


def certification_command(
    config: SessionConfig,
    *,
    physical_evidence: Path | str | None,
    confirmations: dict[str, bool],
) -> list[str]:
    import sys

    required = {
        "no_animal_present",
        "collection_vessel_installed",
        "force_mechanism_clear",
        "speaker_exposure_safe",
        "emergency_power_control_available",
    }
    missing = sorted(name for name in required if confirmations.get(name) is not True)
    if missing:
        raise ValueError("Missing hardware safety confirmations: " + ", ".join(missing))
    effective = write_effective_config(config)
    certificate = resolve_certification_path(config)
    assert certificate is not None
    command = [
        sys.executable,
        "-m",
        "hf_dm.ops_cli",
        "certify-rig",
        "--config",
        str(effective),
        "--output",
        str(certificate),
        "--confirm-no-animal",
        "--confirm-collection-vessel",
        "--confirm-force-clear",
        "--confirm-speaker-safe",
        "--confirm-emergency-stop",
    ]
    if physical_evidence not in (None, ""):
        command.extend(
            ["--physical-evidence", str(Path(str(physical_evidence)).expanduser().resolve())]
        )
    return command


def _absolute_or_none(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(Path(str(value)).expanduser().resolve())


def _positive_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    number = float(value)
    if number <= 0:
        raise ValueError("max_cumulative_reward_ul must be greater than zero")
    return number


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            fsync_best_effort(handle, path=temporary)
        os.replace(temporary, path)
        fsync_directory_best_effort(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
