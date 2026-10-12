"""Single-command no-animal rig certification orchestration."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from hf_dm.certification import (
    CERTIFICATION_KIND,
    CERTIFICATION_SCHEMA_VERSION,
    required_certification_checks,
    REQUIRED_HUMAN_APPROVALS,
    build_release_identity,
    probe_local_durable_storage,
    resolve_certification_path,
)
from hf_dm.config import SessionConfig, validate_session_config
from hf_dm.durable_io import fsync_best_effort, fsync_directory_best_effort

_SELF_TEST_PREFIX = "Hardware self-test validated: "


def run_rig_certification(
    config_path: Path | str,
    *,
    confirmations: dict[str, bool],
    output: Path | str | None = None,
    physical_evidence: Path | str | None = None,
    preflight_duration_s: float = 3.0,
    cap_touch_duration_s: float = 10.0,
    barcode_repetitions: int = 3,
    command_runner: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run all automated rig diagnostics and emit one gate-consumable report.

    Physical oscilloscope, force, SPL, emergency-stop, and welfare evidence is
    intentionally never inferred from software telemetry. It must be supplied as
    a separately signed JSON artifact and is merged without allowing it to
    override a failed machine check.
    """

    required_confirmations = {
        "no_animal_present",
        "collection_vessel_installed",
        "force_mechanism_clear",
        "speaker_exposure_safe",
        "emergency_power_control_available",
    }
    missing = sorted(name for name in required_confirmations if confirmations.get(name) is not True)
    if missing:
        raise ValueError(
            "Certification refused before hardware access; missing operator confirmations: "
            + ", ".join(missing)
        )

    source = Path(config_path).expanduser().resolve()
    config = validate_session_config(SessionConfig.from_yaml(source))
    config.session_export.source_config_path = str(source)
    output_path = Path(output).expanduser().resolve() if output is not None else resolve_certification_path(config)
    if output_path is None:
        raise ValueError("Set animal_use.certification_path or pass --output")

    started = time.time()
    storage = probe_local_durable_storage(config.results_dir)
    runner = command_runner or _run_cli_diagnostic
    runs: dict[str, dict[str, Any]] = {}
    if storage["pass"]:
        run_specs = [
            (
                "preflight",
                ["--preflight", "--preflight-duration", str(preflight_duration_s)],
            ),
            ("hardware_self_test", ["--hardware-self-test"]),
            ("frame2ttl", ["--frame2ttl-diagnostic"]),
            (
                "frame2ttl_barcode",
                [
                    "--frame2ttl-barcode-diagnostic",
                    "--frame2ttl-barcode-repetitions",
                    str(max(1, int(barcode_repetitions))),
                ],
            ),
            ("bpod_opcode", ["--bpod-opcode-diagnostic"]),
        ]
        for name, arguments in run_specs:
            runs[name] = runner(
                name=name,
                config_path=source,
                config=config,
                arguments=arguments,
            )
        if config.licks.source == "capacitive":
            cap_report = output_path.with_name(f".{output_path.stem}.cap-touch.json")
            runs["cap_touch"] = runner(
                name="cap_touch", config_path=source, config=config,
                arguments=[
                    "--cap-touch-diagnostic", "--cap-touch-duration",
                    str(preflight_duration_s if cap_touch_duration_s <= 0 else cap_touch_duration_s),
                    "--cap-touch-report", str(cap_report),
                ], report_path=cap_report,
            )
            cap_report.unlink(missing_ok=True)
    else:
        for name in (
            "preflight",
            "hardware_self_test",
            "frame2ttl",
            "frame2ttl_barcode",
            "bpod_opcode",
            "cap_touch",
        ):
            runs[name] = {
                "name": name,
                "status": "FAIL",
                "error": "skipped because the configured results storage failed durability checks",
            }

    machine_checks = _machine_checks(runs, storage, config)
    evidence = _load_physical_evidence(
        physical_evidence,
        now_s=started,
        max_age_days=config.animal_use.max_certification_age_days,
    )
    checks = _merge_certification_checks(config, machine_checks, evidence.get("checks"))
    approvals = evidence.get("human_approvals")
    if not isinstance(approvals, dict):
        approvals = {}
    release = build_release_identity(config)
    hardware = _hardware_identity_from_runs(runs)
    release_missing = _missing_release_identity(release)
    checks.append(
        {
            "name": "release_identity_complete",
            "status": "PASS" if not release_missing else "FAIL",
            "detail": {"missing": release_missing, "release": release},
        }
    )
    hardware_missing = _missing_hardware_identity(hardware)
    checks.append(
        {
            "name": "hardware_identity_complete",
            "status": "PASS" if not hardware_missing else "FAIL",
            "detail": {"missing": hardware_missing, "hardware": hardware},
        }
    )
    status = "PASS" if (
        all(item.get("status") == "PASS" for item in checks)
        and all(
            _approval_passed(
                approvals.get(name),
                now_s=started,
                max_age_days=config.animal_use.max_certification_age_days,
            )
            for name in REQUIRED_HUMAN_APPROVALS
        )
        and release.get("hardware_failsafe_revision") not in (None, "")
    ) else "FAIL"
    report = {
        "schema_version": CERTIFICATION_SCHEMA_VERSION,
        "kind": CERTIFICATION_KIND,
        "status": status,
        "created_wall_time_s": time.time(),
        "expires_wall_time_s": time.time()
        + config.animal_use.max_certification_age_days * 86400.0,
        "duration_s": time.time() - started,
        "command": "hf-dm certify-rig",
        "config_path": str(source),
        "release": release,
        "hardware": hardware,
        "operator_test_confirmations": dict(confirmations),
        "human_approvals": approvals,
        "checks": checks,
        "machine_runs": runs,
        "physical_evidence_path": (
            None if physical_evidence is None else str(Path(physical_evidence).expanduser().resolve())
        ),
    }
    _atomic_write_json(output_path, report)
    report["output"] = str(output_path)
    return report


def _run_cli_diagnostic(
    *,
    name: str,
    config_path: Path,
    config: SessionConfig,
    arguments: list[str],
    report_path: Path | None = None,
) -> dict[str, Any]:
    results_root = Path(config.results_dir).expanduser().resolve()
    before = {path.resolve() for path in results_root.rglob("session.json")} if results_root.exists() else set()
    command = [
        sys.executable,
        "-m",
        "hf_dm.cli",
        "--config",
        str(config_path),
        "--no-session-dialog",
        "--no-monitor",
        *arguments,
    ]
    started = time.time()
    completed = subprocess.run(
        command,
        cwd=str(Path(__file__).resolve().parents[2]),
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    after = {path.resolve() for path in results_root.rglob("session.json")} if results_root.exists() else set()
    new_sessions = sorted(after - before, key=lambda path: path.stat().st_mtime_ns)
    session_path = new_sessions[-1] if new_sessions else None
    session = _read_json(session_path) if session_path is not None else None
    report = _read_json(report_path) if report_path is not None and report_path.exists() else None
    self_test = None
    if name == "hardware_self_test":
        match = re.search(
            rf"^{re.escape(_SELF_TEST_PREFIX)}(.+)$",
            completed.stdout,
            flags=re.MULTILINE,
        )
        if match:
            try:
                self_test = json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
    return {
        "name": name,
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "returncode": completed.returncode,
        "duration_s": time.time() - started,
        "command": command,
        "session_path": None if session_path is None else str(session_path),
        "session": session,
        "report": report,
        "self_test": self_test,
        "stdout_tail": completed.stdout[-8000:],
        "stderr_tail": completed.stderr[-8000:],
    }


def _machine_checks(
    runs: dict[str, dict[str, Any]],
    storage: dict[str, Any],
    config: SessionConfig | None = None,
) -> list[dict[str, Any]]:
    self_run = runs.get("hardware_self_test", {})
    self_report = self_run.get("self_test")
    if not isinstance(self_report, dict):
        self_report = {}
    self_session = self_run.get("session")
    if not isinstance(self_session, dict):
        self_session = {}

    preflight = runs.get("preflight", {})
    preflight_end = _event_payload(preflight.get("session"), "preflight_end")
    frame = _event_payload(runs.get("frame2ttl", {}).get("session"), "frame2ttl_diagnostic_end")
    barcode = _event_payload(
        runs.get("frame2ttl_barcode", {}).get("session"),
        "frame2ttl_barcode_diagnostic_end",
    )
    opcode = _event_payload(runs.get("bpod_opcode", {}).get("session"), "bpod_opcode_diagnostic_end")
    cap = runs.get("cap_touch", {}).get("report")
    if not isinstance(cap, dict):
        cap = {}
    ownership = _single_ownership_detail(self_session)
    overruns = _recursive_values(self_session.get("stream_summaries"), "buffer_overruns")
    overrun_ok = bool(overruns) and all(int(value) == 0 for value in overruns)
    camera_exact = (
        self_report.get("stream_camera_ttls") is not None
        and self_report.get("stream_camera_ttls") == self_report.get("video_frames")
        and int(self_report.get("video_frames") or 0) > 0
    )

    def result(name: str, passed: bool, detail: Any) -> dict[str, Any]:
        return {"name": name, "status": "PASS" if passed else "FAIL", "detail": detail}

    lick_check = result(
        "cap_touch", cap.get("detected_touch") is True and _transition_verified_off(cap.get("power_off")), cap,
    )
    if config is not None and config.licks.source == "optical":
        events = (preflight.get("session") or {}).get("events", [])
        edges = [event.get("payload", {}) for event in events if event.get("event_type") == "lick_edge"]
        valid_edges = [edge for edge in edges if edge.get("clock") == config.licks.clock
                       and edge.get("source") == "bpod_optical"]
        lick_check = result("optical_lick", preflight.get("status") == "PASS"
                            and {edge.get("value") for edge in valid_edges} == {0, 1},
                            {"clock": config.licks.clock, "edges_seen": len(valid_edges), "preflight": preflight_end})

    return [
        result("preflight", preflight.get("status") == "PASS" and bool(preflight_end), preflight_end),
        result("storage_durability", storage.get("pass") is True, storage),
        result(
            "bpod_states",
            self_run.get("status") == "PASS"
            and self_report.get("overall_status") == "PASS"
            and opcode.get("matched") is True,
            {"self_test": self_report, "opcode": opcode},
        ),
        result("camera_sync", camera_exact, self_report),
        result(
            "frame2ttl",
            frame.get("detected") is True and barcode.get("all_match") is True,
            {"square": frame, "barcode": barcode},
        ),
        lick_check,
        result("single_actuator_ownership", ownership.get("pass") is True, ownership),
        result(
            "streamdio_no_overrun",
            self_run.get("status") == "PASS" and overrun_ok,
            {"buffer_overruns": overruns, "self_test": self_report},
        ),
    ]


def _single_ownership_detail(session: dict[str, Any]) -> dict[str, Any]:
    events = session.get("events")
    if not isinstance(events, list):
        return {"pass": False, "error": "hardware self-test events missing"}
    task_events = [
        event
        for event in events
        if isinstance(event, dict)
        and event.get("event_type") in {"force_on", "force_off", "pump_start", "pump_end"}
    ]
    counts: dict[str, int] = {}
    passive = True
    for event in task_events:
        name = str(event.get("event_type"))
        counts[name] = counts.get(name, 0) + 1
        payload = event.get("payload")
        passive = passive and isinstance(payload, dict)
        if isinstance(payload, dict):
            passive = passive and payload.get("actuator") == "bpod_fsm_serial"
            passive = passive and payload.get("verification") == "asynchronous_streamdio_event"
    expected = {"force_on": 3, "force_off": 3, "pump_start": 2, "pump_end": 2}
    physical_sequences: dict[str, list[int]] = {
        "pump": [],
        "cw_force": [],
        "ccw_force": [],
    }
    for event in events:
        if not isinstance(event, dict) or event.get("event_type") != "output_actual":
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        name = str(payload.get("name") or "")
        if name in physical_sequences:
            physical_sequences[name].append(int(payload.get("state", -1)))
    expected_physical = {
        "pump": [1, 0, 1, 0],
        "cw_force": [1, 0, 1, 0],
        "ccw_force": [1, 0],
    }
    return {
        "pass": passive and counts == expected and physical_sequences == expected_physical,
        "counts": counts,
        "expected": expected,
        "passive_host_verification": passive,
        "physical_output_sequences": physical_sequences,
        "expected_physical_output_sequences": expected_physical,
    }


def _merge_certification_checks(
    config: SessionConfig,
    machine: list[dict[str, Any]],
    physical: Any,
) -> list[dict[str, Any]]:
    by_name = {str(item["name"]): dict(item) for item in machine}
    if isinstance(physical, list):
        for item in physical:
            if not isinstance(item, dict) or not item.get("name"):
                continue
            name = str(item["name"])
            incoming = _validated_physical_check(config, dict(item))
            if name in by_name:
                if by_name[name].get("status") != "PASS" or incoming.get("status") != "PASS":
                    by_name[name] = {
                        "name": name,
                        "status": "FAIL",
                        "detail": {"machine": by_name[name], "physical": incoming},
                    }
                else:
                    by_name[name]["physical_evidence"] = incoming
            else:
                by_name[name] = incoming
    for name in required_certification_checks(config):
        if name not in by_name:
            by_name[name] = {
                "name": name,
                "status": "FAIL",
                "detail": "required physical or machine evidence is missing",
            }
    return [by_name[name] for name in sorted(by_name)]


def _validated_physical_check(
    config: SessionConfig,
    item: dict[str, Any],
) -> dict[str, Any]:
    name = str(item.get("name", ""))
    if item.get("status") != "PASS":
        return item
    evidence = item.get("evidence")
    failures: list[str] = []
    if not isinstance(evidence, dict):
        evidence = {}
        failures.append("structured evidence object missing")
    operator = evidence.get("operator")
    recorded = _number(evidence.get("recorded_wall_time_s"))
    artifact_sha256 = str(evidence.get("artifact_sha256") or "")
    if not operator:
        failures.append("operator identity missing")
    if recorded is None or recorded <= 0:
        failures.append("recorded_wall_time_s missing")
    else:
        age_days = (time.time() - recorded) / 86400.0
        if not 0 <= age_days <= config.animal_use.max_certification_age_days:
            failures.append("physical evidence is future-dated or stale")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", artifact_sha256):
        failures.append("artifact_sha256 must identify the immutable evidence")

    if name in {"valve_timing_40ms", "valve_timing_80ms"}:
        target = 40.0 if name.endswith("40ms") else 80.0
        measured = _number(evidence.get("measured_ms"))
        tolerance = _number(evidence.get("tolerance_ms"))
        if not evidence.get("instrument_id"):
            failures.append("oscilloscope/logic-analyzer instrument_id missing")
        if measured is None or tolerance is None or tolerance <= 0:
            failures.append("measured_ms and positive tolerance_ms are required")
        elif abs(measured - target) > tolerance:
            failures.append(
                f"measured pulse {measured:g} ms is outside {target:g} +/- {tolerance:g} ms"
            )
    elif name == "speaker_spl":
        measured = _number(evidence.get("measured_db_spl"))
        maximum = _number(evidence.get("approved_max_db_spl"))
        if not evidence.get("instrument_id"):
            failures.append("SPL meter instrument_id missing")
        if measured is None or maximum is None or measured > maximum:
            failures.append("measured SPL is missing or exceeds the approved maximum")
    elif name == "force_direction_magnitude":
        cw = _number(evidence.get("cw_magnitude"))
        ccw = _number(evidence.get("ccw_magnitude"))
        maximum = _number(evidence.get("approved_max_magnitude"))
        if not evidence.get("instrument_id"):
            failures.append("force measurement instrument_id missing")
        if evidence.get("cw_direction_verified") is not True:
            failures.append("CW force direction is not verified")
        if evidence.get("ccw_direction_verified") is not True:
            failures.append("CCW force direction is not verified")
        if (
            cw is None
            or ccw is None
            or maximum is None
            or min(cw, ccw) <= 0
            or max(cw, ccw) > maximum
        ):
            failures.append("force magnitudes are missing, non-positive, or above the safe limit")
    elif name == "wheel_sign_geometry":
        radius = _number(evidence.get("wheel_radius_mm"))
        gain = _number(evidence.get("gain_deg_per_mm"))
        if evidence.get("cw_sign_verified") is not True or evidence.get("ccw_sign_verified") is not True:
            failures.append("both wheel signs must be physically verified")
        if radius is None or abs(radius - config.wheel.wheel_radius_mm) > 0.1:
            failures.append("measured wheel radius does not match the configured radius")
        if gain is None or abs(gain - config.wheel.gain_deg_per_mm) > 1e-6:
            failures.append("verified wheel gain does not match the configured gain")
    elif name == "hardware_failsafe":
        if evidence.get("revision") != config.animal_use.hardware_failsafe_revision:
            failures.append("hardware fail-safe revision does not match the config")
        independent = set(evidence.get("independent_of") or [])
        required_independence = {"python", "usb", "bpod", "streamdio"}
        if not required_independence <= independent:
            failures.append("fail-safe independence from Python/USB/Bpod/StreamDIO is unproven")
        if not _failure_cases_pass(
            evidence,
            {"stuck_high", "unplugged_usb", "frozen_host", "bpod_reset", "streamdio_reset"},
        ):
            failures.append("hardware fail-safe failure cases are missing or failed")
    elif name == "all_off_failure_injection":
        if not _failure_cases_pass(
            evidence,
            {
                "stuck_high",
                "unplugged_usb",
                "frozen_host",
                "bpod_reset",
                "streamdio_reset",
                "emergency_stop",
            },
        ):
            failures.append("required physical all-off injection cases are missing or failed")
    elif name == "soak":
        duration = _number(evidence.get("duration_s"))
        required = 2.0 * _maximum_planned_session_s(config)
        if duration is None or duration < required:
            failures.append(f"soak duration must be at least {required:g} seconds")
        if int(evidence.get("streamdio_overruns", -1)) != 0:
            failures.append("StreamDIO overruns were nonzero or unreported")
        if int(evidence.get("dropped_packets", -1)) != 0:
            failures.append("dropped packets were nonzero or unreported")
        if evidence.get("session_validation_pass") is not True:
            failures.append("soak session validation did not pass")
        if evidence.get("qc_pass") is not True:
            failures.append("soak QC did not pass")
    elif name == "welfare_limits":
        if evidence.get("protocol_id") in (None, ""):
            failures.append("approved protocol_id missing")
        if evidence.get("trial_limit_verified") is not True:
            failures.append("trial limit is not approved")
        if evidence.get("iti_verified") is not True:
            failures.append("ITI is not approved")
        if evidence.get("engagement_stop_rules_verified") is not True:
            failures.append("engagement stop rules are not approved")
        cap = _number(evidence.get("max_cumulative_reward_ul"))
        if cap != config.welfare.max_cumulative_reward_ul:
            failures.append("approved cumulative reward cap does not match config")

    if failures:
        return {
            "name": name,
            "status": "FAIL",
            "detail": {
                "claimed": item,
                "evidence_validation_failures": failures,
            },
        }
    return item


def _failure_cases_pass(evidence: dict[str, Any], required: set[str]) -> bool:
    cases = evidence.get("cases")
    if not isinstance(cases, list):
        return False
    cases_by_name = {
        str(case.get("name")): case
        for case in cases
        if isinstance(case, dict)
    }
    for name in required:
        case = cases_by_name.get(name)
        if not isinstance(case, dict) or case.get("status") != "PASS":
            return False
        measured = _number(case.get("safe_deenergization_ms"))
        limit = _number(case.get("approved_limit_ms"))
        if measured is None or limit is None or limit <= 0 or measured > limit:
            return False
    return True


def _maximum_planned_session_s(config: SessionConfig) -> float:
    success = (
        config.timing.choice_blank_s
        + max(config.reward.matched_pump_s, config.reward.opposite_pump_s)
        + config.timing.post_reward_s
    )
    timeout = config.timing.timeout_noise_s + config.timing.post_timeout_s
    per_trial = (
        config.timing.quiescence_s
        + config.timing.choice_timeout_s
        + max(success, timeout)
        + config.timing.inter_trial_min_s
    )
    return config.n_trials * per_trial


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _hardware_identity_from_runs(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    session = runs.get("hardware_self_test", {}).get("session")
    if not isinstance(session, dict):
        return {"bpod": {}, "streamdio": {}, "camera": {}}
    metadata = session.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    camera: dict[str, Any] = {}
    video_end = _event_payload(session, "video_recording_stopped")
    if isinstance(video_end.get("device"), dict):
        camera = dict(video_end["device"])
    bpod = metadata.get("bpod")
    streamdio = metadata.get("streamdio_module_info")
    return {
        "bpod": dict(bpod) if isinstance(bpod, dict) else {},
        "streamdio": dict(streamdio) if isinstance(streamdio, dict) else {},
        "camera": camera,
    }


def _missing_release_identity(release: dict[str, Any]) -> list[str]:
    missing = [
        key
        for key in (
            "git_commit",
            "streamdio_driver_sha256",
            "requirements_lock_sha256",
            "package_versions",
            "config_sha256",
            "calibration_sha256",
            "hardware_failsafe_revision",
        )
        if release.get(key) in (None, "", {})
    ]
    if release.get("git_dirty") is not False:
        missing.append("git_clean")
    driver = release.get("streamdio_driver")
    if not isinstance(driver, dict) or driver.get("module_path") in (None, ""):
        missing.append("streamdio_driver.module_path")
    return missing


def _missing_hardware_identity(hardware: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for group, key in (
        ("bpod", "serial_number"),
        ("bpod", "firmware"),
        ("streamdio", "firmware_version"),
        ("camera", "serial_number"),
        ("camera", "firmware_version"),
    ):
        section = hardware.get(group)
        if not isinstance(section, dict) or section.get(key) in (None, "", []):
            missing.append(f"{group}.{key}")
    return missing


def _event_payload(session: Any, event_type: str) -> dict[str, Any]:
    if not isinstance(session, dict) or not isinstance(session.get("events"), list):
        return {}
    for event in reversed(session["events"]):
        if isinstance(event, dict) and event.get("event_type") == event_type:
            payload = event.get("payload")
            return dict(payload) if isinstance(payload, dict) else {}
    return {}


def _recursive_values(value: Any, key: str) -> list[int]:
    values: list[int] = []
    if isinstance(value, dict):
        for name, item in value.items():
            if name == key:
                try:
                    values.append(int(item))
                except (TypeError, ValueError):
                    pass
            else:
                values.extend(_recursive_values(item, key))
    elif isinstance(value, list):
        for item in value:
            values.extend(_recursive_values(item, key))
    return values


def _transition_verified_off(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    state = value.get("state", value.get("enabled", 1))
    if state is None:
        return False
    return (
        int(state) == 0
        and value.get("acknowledged", True) is True
        and value.get("event_verified", True) is True
    )


def _approval_passed(
    value: Any,
    *,
    now_s: float | None = None,
    max_age_days: float | None = None,
) -> bool:
    signed = _number(value.get("signed_wall_time_s")) if isinstance(value, dict) else None
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


def _load_physical_evidence(
    path: Path | str | None,
    *,
    now_s: float | None = None,
    max_age_days: float | None = None,
) -> dict[str, Any]:
    if path is None:
        return {}
    payload = _read_json(Path(path).expanduser().resolve())
    if not isinstance(payload, dict):
        raise ValueError("Physical certification evidence must be a JSON object")
    if payload.get("kind") != "hf_dm_physical_certification_evidence":
        raise ValueError("Unsupported physical certification evidence kind")
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported physical certification evidence schema_version")
    created = _number(payload.get("created_wall_time_s"))
    if (created or 0.0) <= 0:
        raise ValueError("Physical certification evidence created_wall_time_s is required")
    if now_s is not None and max_age_days is not None and created is not None:
        age_days = (float(now_s) - created) / 86400.0
        if not 0 <= age_days <= float(max_age_days):
            raise ValueError("Physical certification evidence is future-dated or stale")
    if not str(payload.get("rig_id") or "").strip():
        raise ValueError("Physical certification evidence rig_id is required")
    return payload


def _read_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            fsync_best_effort(handle, path=temporary)
        os.replace(temporary, path)
        fsync_directory_best_effort(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
