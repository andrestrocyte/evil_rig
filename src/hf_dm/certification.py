"""Fail-closed release identity and animal-use certification checks."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable

from hf_dm.config import SessionConfig, config_to_dict
from hf_dm.data.recorder import capture_runtime_provenance
from hf_dm.durable_io import fsync_best_effort
from hf_dm.subject_log import load_subject_log
from hf_dm.valve_calibration import RewardCalibration, resolve_reward_calibration_path

CERTIFICATION_KIND = "hf_dm_rig_certification"
CERTIFICATION_SCHEMA_VERSION = 1

REQUIRED_CERTIFICATION_CHECKS = frozenset(
    {
        "all_off_failure_injection",
        "bpod_states",
        "camera_sync",
        "cap_touch",
        "force_direction_magnitude",
        "frame2ttl",
        "hardware_failsafe",
        "single_actuator_ownership",
        "soak",
        "speaker_spl",
        "storage_durability",
        "streamdio_no_overrun",
        "valve_timing_40ms",
        "valve_timing_80ms",
        "welfare_limits",
        "wheel_sign_geometry",
    }
)
REQUIRED_HUMAN_APPROVALS = frozenset(
    {
        "calibration",
        "emergency_stop",
        "physical_safety",
        "welfare_protocol",
    }
)


def required_certification_checks(config: SessionConfig) -> frozenset[str]:
    if config.licks.source == "optical":
        return (REQUIRED_CERTIFICATION_CHECKS - {"cap_touch"}) | {"optical_lick"}
    return REQUIRED_CERTIFICATION_CHECKS


class AnimalUseGateError(RuntimeError):
    """Raised before animal acquisition when any release requirement is unproven."""

    def __init__(self, report: dict[str, Any]):
        self.report = report
        failures = [
            str(item.get("name"))
            for item in report.get("checks", [])
            if item.get("status") != "PASS"
        ]
        super().__init__(
            "Animal-use startup blocked by "
            f"{len(failures)} failed release check(s): {', '.join(failures)}"
        )


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def certified_config_payload(config: SessionConfig) -> dict[str, Any]:
    """Return the effective safety-relevant config used for release matching."""

    payload = config_to_dict(config)
    # Subject/session identity and output locations change for every acquisition.
    # They are validated independently at startup and must not invalidate a rig
    # certificate that otherwise matches the exact task and hardware contract.
    payload["subject_id"] = None
    payload["results_dir"] = None
    # Session length is checked against calibrated reward limits on every run.
    # Reordering or choosing fewer trials does not change the hardware contract.
    payload["n_trials"] = None
    payload["random_seed"] = None
    reward = payload.get("reward")
    if isinstance(reward, dict):
        # The calibration artifact itself is bound separately by SHA-256.
        reward["calibration_path"] = None
    camera = payload.get("camera")
    if isinstance(camera, dict):
        camera["save_dir"] = None
    monitor = payload.get("monitor")
    if isinstance(monitor, dict) and monitor.get("record_path"):
        # 'auto' resolves to a new absolute path after hardware opens. Preserve
        # whether telemetry recording is enabled, not its per-session filename.
        monitor["record_path"] = "<session>"
    task = payload.get("task")
    if isinstance(task, dict):
        task["profile_name"] = None
        task["n_free_trials"] = None
        task["n_forced_trials"] = None
        task["selected_mouse_dir"] = None
        task["session_name"] = None
        task["session_dir"] = None
    welfare = payload.get("welfare")
    if isinstance(welfare, dict):
        welfare["subjects_dir"] = None
    animal_use = payload.get("animal_use")
    if isinstance(animal_use, dict):
        animal_use["certification_path"] = None
    export = payload.get("session_export")
    if isinstance(export, dict):
        # Absolute source locations are provenance, not experiment semantics.
        export["source_config_path"] = None
        # Overrides are an audit trail. Their effective values already appear in
        # the canonical config fields above, while setup timestamps and paths are
        # intentionally session-specific.
        export["cli_overrides"] = []
    return payload


def certified_config_sha256(config: SessionConfig) -> str:
    encoded = json.dumps(
        certified_config_payload(config),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve_certification_path(config: SessionConfig) -> Path | None:
    value = config.animal_use.certification_path
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    source = config.session_export.source_config_path
    if source:
        return (Path(source).expanduser().resolve().parent / path).resolve()
    return path.resolve()


def build_release_identity(
    config: SessionConfig,
    *,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    runtime = dict(provenance or capture_runtime_provenance())
    calibration_path = resolve_reward_calibration_path(config)
    calibration_sha256 = (
        sha256_file(calibration_path)
        if calibration_path is not None and calibration_path.is_file()
        else None
    )
    source_path = config.session_export.source_config_path
    source_sha256 = None
    if source_path:
        source = Path(source_path).expanduser()
        if source.is_file():
            source_sha256 = sha256_file(source)
    return {
        "git_commit": runtime.get("git_commit"),
        "git_exact_tag": runtime.get("git_exact_tag"),
        "git_dirty": runtime.get("git_dirty"),
        "runtime_code_identity_scheme": runtime.get("runtime_code_identity_scheme"),
        "runtime_code_sha256": runtime.get("runtime_code_sha256"),
        "requirements_lock_sha256": runtime.get("requirements_lock_sha256"),
        "package_versions": runtime.get("package_versions"),
        "python_version": runtime.get("python_version"),
        "streamdio_driver": runtime.get("streamdio_driver"),
        "streamdio_driver_sha256": (runtime.get("streamdio_driver") or {}).get("source_sha256"),
        "config_sha256": certified_config_sha256(config),
        "source_config_sha256": source_sha256,
        "calibration_sha256": calibration_sha256,
        "hardware_failsafe_revision": config.animal_use.hardware_failsafe_revision,
    }


def load_certification(path: Path | str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Rig certification must contain a JSON object")
    return payload


def probe_local_durable_storage(path: Path | str) -> dict[str, Any]:
    """Require a local fixed path with working file flush and atomic replace."""

    target = Path(path).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    local = _is_local_fixed_path(target)
    probe = target / f".hf_dm_durability_probe.{os.getpid()}.{time.time_ns()}"
    replacement = probe.with_suffix(".replaced")
    synced = False
    replaced = False
    error: str | None = None
    try:
        with probe.open("xb") as handle:
            handle.write(b"hf-dm-durability-probe\n")
            handle.flush()
            synced = fsync_best_effort(handle, path=probe)
        os.replace(probe, replacement)
        replaced = replacement.read_bytes() == b"hf-dm-durability-probe\n"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        probe.unlink(missing_ok=True)
        replacement.unlink(missing_ok=True)
    return {
        "path": str(target),
        "local_fixed_storage": local,
        "file_fsync": synced,
        "atomic_replace": replaced,
        "error": error,
        "pass": bool(local and synced and replaced and error is None),
    }


def evaluate_animal_use_gate(
    config: SessionConfig,
    *,
    operator_confirmed: bool,
    provenance: dict[str, Any] | None = None,
    actual_hardware: dict[str, Any] | None = None,
    now_s: float | None = None,
    storage_probe: Callable[[Path | str], dict[str, Any]] = probe_local_durable_storage,
) -> dict[str, Any]:
    """Evaluate all machine-verifiable prerequisites without opening hardware."""

    now = time.time() if now_s is None else float(now_s)
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append(
            {
                "name": name,
                "status": "PASS" if passed else "FAIL",
                "detail": detail,
            }
        )

    if not config.animal_use.enabled:
        check("animal_use_enabled", True, "disabled; bench/simulation behavior unchanged")
        return {"status": "PASS", "animal_use_enabled": False, "checks": checks}

    cert_path = resolve_certification_path(config)
    certificate: dict[str, Any] = {}
    certificate_error: str | None = None
    if cert_path is None:
        certificate_error = "animal_use.certification_path is not configured"
    else:
        try:
            certificate = load_certification(cert_path)
        except Exception as exc:
            certificate_error = f"{type(exc).__name__}: {exc}"
    check(
        "certification_readable",
        certificate_error is None,
        {"path": None if cert_path is None else str(cert_path), "error": certificate_error},
    )

    check(
        "operator_confirmation",
        bool(operator_confirmed) or not config.animal_use.require_operator_confirmation,
        "one-time confirmation supplied" if operator_confirmed else "confirmation missing",
    )

    identity = build_release_identity(config, provenance=provenance)
    check(
        "clean_git_provenance",
        not config.animal_use.require_clean_git
        or (identity.get("git_commit") is not None and identity.get("git_dirty") is False),
        {"git_commit": identity.get("git_commit"), "git_dirty": identity.get("git_dirty")},
    )
    check(
        "dependency_lock_present",
        bool(identity.get("requirements_lock_sha256")),
        identity.get("requirements_lock_sha256"),
    )

    storage_paths = {"results": config.results_dir}
    if config.camera.record_video or config.camera.required:
        storage_paths["camera"] = config.camera.save_dir
    storage: dict[str, Any] = {"paths": {}}
    for name, path in storage_paths.items():
        try:
            storage["paths"][name] = storage_probe(path)
        except Exception as exc:
            storage["paths"][name] = {
                "pass": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
    storage["pass"] = all(
        item.get("pass") is True for item in storage["paths"].values()
    )
    check(
        "local_durable_storage",
        not config.animal_use.require_local_storage or storage.get("pass") is True,
        storage,
    )

    check(
        "required_runtime_hardware",
        (not config.animal_use.require_camera or (config.camera.required and config.camera.record_video))
        and (not config.animal_use.require_speaker or config.speaker.required)
        and (not config.animal_use.require_licks
             or config.licks.source == "optical" or config.licks.power_required),
        {
            "camera_required": config.camera.required,
            "camera_record_video": config.camera.record_video,
            "speaker_required": config.speaker.required,
            "lick_power_required": config.licks.power_required,
            "lick_source": config.licks.source,
        },
    )

    calibration = _calibration_release_detail(config, now)
    check("reward_calibration", calibration.get("pass") is True, calibration)
    welfare = _welfare_release_detail(config, calibration, now)
    check("subject_welfare", welfare.get("pass") is True, welfare)

    schema_ok = (
        certificate.get("schema_version") == CERTIFICATION_SCHEMA_VERSION
        and certificate.get("kind") == CERTIFICATION_KIND
    )
    check(
        "certification_schema",
        schema_ok,
        {
            "schema_version": certificate.get("schema_version"),
            "kind": certificate.get("kind"),
        },
    )
    check("certification_status", certificate.get("status") == "PASS", certificate.get("status"))

    created = _optional_float(certificate.get("created_wall_time_s"))
    age_days = None if created is None else (now - created) / 86400.0
    expires = _optional_float(certificate.get("expires_wall_time_s"))
    age_ok = (
        created is not None
        and age_days is not None
        and 0 <= age_days <= config.animal_use.max_certification_age_days
        and (expires is None or now <= expires)
    )
    check(
        "certification_current",
        age_ok,
        {"created_wall_time_s": created, "age_days": age_days, "expires_wall_time_s": expires},
    )

    certified_release = certificate.get("release")
    if not isinstance(certified_release, dict):
        certified_release = {}
    release_keys = (
        "streamdio_driver_sha256",
        "requirements_lock_sha256",
        "package_versions",
        "config_sha256",
        "calibration_sha256",
        "hardware_failsafe_revision",
    )
    if identity.get("runtime_code_identity_scheme") or certified_release.get("runtime_code_identity_scheme"):
        # A clean documentation/test-only commit need not repeat commissioning.
        # Both sides must carry a matching installed-code hash; missing hashes
        # and old certificates fail closed against a newly fingerprinted runtime.
        release_keys += ("runtime_code_identity_scheme", "runtime_code_sha256", "python_version")
    else:
        # Historical provenance supplied during offline evaluation retains its
        # original exact-commit semantics. Live capture always includes a scheme.
        release_keys += ("git_commit",)
    mismatches = {
        key: {"certified": certified_release.get(key), "current": identity.get(key)}
        for key in release_keys
        if certified_release.get(key) != identity.get(key) or identity.get(key) in (None, "")
    }
    check("exact_release_identity", not mismatches, mismatches or identity)

    certified_checks = certificate.get("checks")
    if not isinstance(certified_checks, list):
        certified_checks = []
    check_status = {
        str(item.get("name")): str(item.get("status"))
        for item in certified_checks
        if isinstance(item, dict)
    }
    missing_or_failed = {
        name: check_status.get(name)
        for name in sorted(required_certification_checks(config))
        if check_status.get(name) != "PASS"
    }
    extra_nonpass = {
        name: status for name, status in check_status.items() if status != "PASS"
    }
    check(
        "certified_checks_all_pass",
        not missing_or_failed and not extra_nonpass,
        {"missing_or_failed": missing_or_failed, "other_nonpass": extra_nonpass},
    )

    approvals = certificate.get("human_approvals")
    if not isinstance(approvals, dict):
        approvals = {}
    missing_approvals = [
        name
        for name in sorted(REQUIRED_HUMAN_APPROVALS)
        if not _approval_passed(
            approvals.get(name),
            now_s=now,
            max_age_days=config.animal_use.max_certification_age_days,
        )
    ]
    check("human_approvals", not missing_approvals, missing_approvals)

    certified_hardware = certificate.get("hardware")
    if not isinstance(certified_hardware, dict):
        certified_hardware = {}
    identity_missing = _missing_hardware_identity(certified_hardware, certified_release)
    check("certified_hardware_identity_complete", not identity_missing, identity_missing)
    if actual_hardware is not None:
        hardware_mismatches = _subset_mismatches(certified_hardware, actual_hardware)
        check("connected_hardware_matches", not hardware_mismatches, hardware_mismatches)

    status = "PASS" if all(item["status"] == "PASS" for item in checks) else "FAIL"
    return {
        "status": status,
        "animal_use_enabled": True,
        "certification_path": None if cert_path is None else str(cert_path),
        "release_identity": identity,
        "checks": checks,
    }


def assert_animal_use_ready(config: SessionConfig, **kwargs: Any) -> dict[str, Any]:
    report = evaluate_animal_use_gate(config, **kwargs)
    if report["status"] != "PASS":
        raise AnimalUseGateError(report)
    return report


def _calibration_release_detail(config: SessionConfig, now_s: float) -> dict[str, Any]:
    path = resolve_reward_calibration_path(config)
    if path is None:
        return {"pass": False, "error": "reward.calibration_path is not configured"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        calibration = RewardCalibration.load(path)
        expected = {
            "matched_reward_ul": calibration.volume_for_duration(config.reward.matched_pump_s),
            "opposite_reward_ul": calibration.volume_for_duration(config.reward.opposite_pump_s),
        }
        created = _optional_float(payload.get("created_wall_s"))
        age_days = None if created is None else (now_s - created) / 86400.0
        fit = payload.get("linear_fit") if isinstance(payload.get("linear_fit"), dict) else {}
        r_squared = _optional_float(fit.get("r_squared"))
        max_residual = _optional_float(fit.get("max_abs_residual_ul"))
        current = (
            created is not None
            and age_days is not None
            and 0 <= age_days <= config.animal_use.max_calibration_age_days
        )
        return {
            "pass": bool(
                current
                and r_squared is not None
                and r_squared >= 0.95
                and max_residual is not None
                and max_residual <= config.animal_use.max_calibration_residual_ul
            ),
            "path": str(path),
            "sha256": calibration.sha256,
            "created_wall_s": created,
            "age_days": age_days,
            "r_squared": r_squared,
            "max_abs_residual_ul": max_residual,
            "approved_max_residual_ul": config.animal_use.max_calibration_residual_ul,
            **expected,
        }
    except Exception as exc:
        return {"pass": False, "path": str(path), "error": f"{type(exc).__name__}: {exc}"}


def _welfare_release_detail(
    config: SessionConfig,
    calibration: dict[str, Any],
    now_s: float,
) -> dict[str, Any]:
    reasons: list[str] = []
    cap = config.welfare.max_cumulative_reward_ul
    max_reward = max(
        float(calibration.get("matched_reward_ul") or 0.0),
        float(calibration.get("opposite_reward_ul") or 0.0),
    )
    worst_case = max_reward * config.n_trials
    if cap is None:
        reasons.append("welfare.max_cumulative_reward_ul is not configured")
    elif worst_case > cap:
        reasons.append(f"worst-case session reward {worst_case:g} uL exceeds cap {cap:g} uL")

    latest_weight: dict[str, Any] | None = None
    weight_age_days: int | None = None
    if config.welfare.require_current_weight:
        try:
            rows = load_subject_log(config.welfare.subjects_dir, config.subject_id)
            weight_rows = [row for row in rows if row.get("entry_type") == "weight"]
            if weight_rows:
                latest_weight = max(weight_rows, key=lambda row: str(row.get("date", "")))
                entry_date = date.fromisoformat(str(latest_weight["date"]))
                now_date = datetime.fromtimestamp(now_s, tz=timezone.utc).date()
                weight_age_days = (now_date - entry_date).days
            if latest_weight is None:
                reasons.append("current subject weight record is missing")
            elif weight_age_days is None or not 0 <= weight_age_days <= config.welfare.max_weight_age_days:
                reasons.append(f"subject weight age is {weight_age_days!r} days")
            elif (_optional_float(latest_weight.get("weight_g")) or 0.0) <= 0:
                reasons.append("current subject weight is missing or non-positive")
        except Exception as exc:
            reasons.append(f"subject log error: {type(exc).__name__}: {exc}")
    water_target = config.welfare.water_target_ml
    if water_target is None and latest_weight is not None:
        water_target = _optional_float(latest_weight.get("water_target_ml"))
    if water_target is None or water_target <= 0:
        reasons.append("positive water target is not configured in config or current weight record")
    return {
        "pass": not reasons and calibration.get("pass") is True,
        "max_cumulative_reward_ul": cap,
        "worst_case_reward_ul": worst_case,
        "weight_age_days": weight_age_days,
        "water_target_ml": water_target,
        "reasons": reasons,
    }


def _missing_hardware_identity(
    hardware: dict[str, Any],
    release: dict[str, Any],
) -> list[str]:
    required_paths = (
        ("bpod", "serial_number"),
        ("bpod", "firmware"),
        ("streamdio", "firmware_version"),
        ("streamdio", "build_id"),
        ("camera", "serial_number"),
        ("camera", "firmware_version"),
    )
    missing: list[str] = []
    for group, key in required_paths:
        section = hardware.get(group)
        if not isinstance(section, dict) or section.get(key) in (None, "", []):
            missing.append(f"hardware.{group}.{key}")
    driver = release.get("streamdio_driver")
    if not isinstance(driver, dict) or driver.get("module_path") in (None, ""):
        missing.append("release.streamdio_driver.module_path")
    return missing


def _subset_mismatches(expected: Any, actual: Any, prefix: str = "hardware") -> dict[str, Any]:
    mismatches: dict[str, Any] = {}
    if not isinstance(expected, dict) or not isinstance(actual, dict):
        if expected != actual:
            mismatches[prefix] = {"certified": expected, "actual": actual}
        return mismatches
    for key, expected_value in expected.items():
        if expected_value is None:
            continue
        path = f"{prefix}.{key}"
        if key not in actual:
            mismatches[path] = {"certified": expected_value, "actual": None}
        elif isinstance(expected_value, dict):
            mismatches.update(_subset_mismatches(expected_value, actual[key], path))
        elif actual[key] != expected_value:
            mismatches[path] = {"certified": expected_value, "actual": actual[key]}
    return mismatches


def _is_local_fixed_path(path: Path) -> bool:
    text = str(path)
    if text.startswith("\\\\") or text.startswith("//"):
        return False
    if os.name != "nt":
        return True
    root = Path(path.anchor)
    if not root.anchor:
        return False
    get_drive_type = ctypes.windll.kernel32.GetDriveTypeW
    get_drive_type.argtypes = [ctypes.c_wchar_p]
    get_drive_type.restype = ctypes.c_uint
    return int(get_drive_type(str(root))) == 3  # DRIVE_FIXED


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _approval_passed(
    value: Any,
    *,
    now_s: float | None = None,
    max_age_days: float | None = None,
) -> bool:
    signed = _optional_float(value.get("signed_wall_time_s")) if isinstance(value, dict) else None
    valid = (
        isinstance(value, dict)
        and value.get("approved") is True
        and bool(str(value.get("operator") or "").strip())
        and bool(str(value.get("signature") or "").strip())
        and (signed or 0.0) > 0
    )
    if not valid or now_s is None or max_age_days is None or signed is None:
        return valid
    age_days = (float(now_s) - signed) / 86400.0
    return 0 <= age_days <= float(max_age_days)
