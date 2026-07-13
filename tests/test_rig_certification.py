from __future__ import annotations

import json
import time

import pytest

from hf_dm.certification import REQUIRED_CERTIFICATION_CHECKS
from hf_dm.rig_certification import run_rig_certification


CONFIRMATIONS = {
    "no_animal_present": True,
    "collection_vessel_installed": True,
    "force_mechanism_clear": True,
    "speaker_exposure_safe": True,
    "emergency_power_control_available": True,
}


def _write_config(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        "subject_id: mouse01\n"
        "results_dir: " + json.dumps(str(tmp_path / "sessions")) + "\n"
        "animal_use:\n"
        "  enabled: true\n"
        "  certification_path: rig.json\n"
        "  hardware_failsafe_revision: watchdog-r2\n"
        "welfare:\n"
        "  max_cumulative_reward_ul: 500\n",
        encoding="utf-8",
    )
    return config


def _session(event_type, payload):
    return {"events": [{"event_type": event_type, "payload": payload}]}


def _fake_runner(**kwargs):
    name = kwargs["name"]
    if name == "preflight":
        return {"name": name, "status": "PASS", "session": _session("preflight_end", {"ok": True})}
    if name == "hardware_self_test":
        events = []
        for event_type in ("force_on", "force_off"):
            for _ in range(3):
                events.append(
                    {
                        "event_type": event_type,
                        "payload": {
                            "actuator": "bpod_fsm_serial",
                            "verification": "asynchronous_streamdio_event",
                        },
                    }
                )
        for event_type in ("pump_start", "pump_end"):
            for _ in range(2):
                events.append(
                    {
                        "event_type": event_type,
                        "payload": {
                            "actuator": "bpod_fsm_serial",
                            "verification": "asynchronous_streamdio_event",
                        },
                    }
                )
        for name, states in {
            "pump": [1, 0, 1, 0],
            "cw_force": [1, 0, 1, 0],
            "ccw_force": [1, 0],
        }.items():
            for state in states:
                events.append(
                    {
                        "event_type": "output_actual",
                        "payload": {"name": name, "state": state},
                    }
                )
        events.append(
            {
                "event_type": "video_recording_stopped",
                "payload": {
                    "device": {
                        "serial_number": "20137635",
                        "firmware_version": "1707.0.125.0",
                    }
                },
            }
        )
        return {
            "name": name,
            "status": "PASS",
            "self_test": {
                "overall_status": "PASS",
                "stream_camera_ttls": 92,
                "video_frames": 92,
            },
            "session": {
                "events": events,
                "metadata": {
                    "bpod": {"serial_number": "17032730", "firmware": [23, 10]},
                    "streamdio_module_info": {"firmware_version": 4},
                },
                "stream_summaries": {"streamdio": {"buffer_overruns": 0}},
            },
        }
    if name == "frame2ttl":
        return {
            "name": name,
            "status": "PASS",
            "session": _session("frame2ttl_diagnostic_end", {"detected": True}),
        }
    if name == "frame2ttl_barcode":
        return {
            "name": name,
            "status": "PASS",
            "session": _session("frame2ttl_barcode_diagnostic_end", {"all_match": True}),
        }
    if name == "bpod_opcode":
        return {
            "name": name,
            "status": "PASS",
            "session": _session("bpod_opcode_diagnostic_end", {"matched": True}),
        }
    if name == "cap_touch":
        return {
            "name": name,
            "status": "PASS",
            "report": {
                "detected_touch": True,
                "power_off": {
                    "state": 0,
                    "acknowledged": True,
                    "event_verified": True,
                },
            },
        }
    raise AssertionError(name)


def _physical_checks():
    now = time.time()
    common = {
        "operator": "tester",
        "recorded_wall_time_s": now,
        "artifact_sha256": "e" * 64,
    }
    failure_cases = [
        {
            "name": name,
            "status": "PASS",
            "safe_deenergization_ms": 10.0,
            "approved_limit_ms": 50.0,
        }
        for name in (
            "stuck_high",
            "unplugged_usb",
            "frozen_host",
            "bpod_reset",
            "streamdio_reset",
            "emergency_stop",
        )
    ]
    return [
        {
            "name": "valve_timing_40ms",
            "status": "PASS",
            "evidence": {**common, "instrument_id": "scope-1", "measured_ms": 40.0, "tolerance_ms": 1.0},
        },
        {
            "name": "valve_timing_80ms",
            "status": "PASS",
            "evidence": {**common, "instrument_id": "scope-1", "measured_ms": 80.0, "tolerance_ms": 1.0},
        },
        {
            "name": "speaker_spl",
            "status": "PASS",
            "evidence": {**common, "instrument_id": "spl-1", "measured_db_spl": 70.0, "approved_max_db_spl": 75.0},
        },
        {
            "name": "force_direction_magnitude",
            "status": "PASS",
            "evidence": {
                **common,
                "instrument_id": "force-gauge-1",
                "cw_direction_verified": True,
                "ccw_direction_verified": True,
                "cw_magnitude": 1.0,
                "ccw_magnitude": 1.0,
                "approved_max_magnitude": 2.0,
            },
        },
        {
            "name": "wheel_sign_geometry",
            "status": "PASS",
            "evidence": {
                **common,
                "cw_sign_verified": True,
                "ccw_sign_verified": True,
                "wheel_radius_mm": 31.5,
                "gain_deg_per_mm": 4.0,
            },
        },
        {
            "name": "hardware_failsafe",
            "status": "PASS",
            "evidence": {
                **common,
                "revision": "watchdog-r2",
                "independent_of": ["python", "usb", "bpod", "streamdio"],
                "cases": failure_cases,
            },
        },
        {
            "name": "all_off_failure_injection",
            "status": "PASS",
            "evidence": {**common, "cases": failure_cases},
        },
        {
            "name": "soak",
            "status": "PASS",
            "evidence": {
                **common,
                "duration_s": 3000.0,
                "streamdio_overruns": 0,
                "dropped_packets": 0,
                "session_validation_pass": True,
                "qc_pass": True,
            },
        },
        {
            "name": "welfare_limits",
            "status": "PASS",
            "evidence": {
                **common,
                "protocol_id": "protocol-1",
                "trial_limit_verified": True,
                "iti_verified": True,
                "engagement_stop_rules_verified": True,
                "max_cumulative_reward_ul": 500.0,
            },
        },
    ]


def test_certify_rig_refuses_missing_pre_actuation_confirmations(tmp_path):
    config = _write_config(tmp_path)
    confirmations = dict(CONFIRMATIONS)
    confirmations["no_animal_present"] = False

    with pytest.raises(ValueError, match="no_animal_present"):
        run_rig_certification(
            config,
            confirmations=confirmations,
            command_runner=lambda **_kwargs: pytest.fail("hardware was touched"),
        )


def test_certify_rig_emits_gate_consumable_pass_only_with_complete_evidence(
    tmp_path, monkeypatch
):
    config = _write_config(tmp_path)
    evidence_path = tmp_path / "physical.json"
    evidence_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "hf_dm_physical_certification_evidence",
                "created_wall_time_s": time.time(),
                "rig_id": "rig1",
                "checks": _physical_checks(),
                "human_approvals": {
                    name: {
                        "approved": True,
                        "operator": "tester",
                        "signature": "tester-signature",
                        "signed_wall_time_s": time.time(),
                    }
                    for name in (
                        "calibration",
                        "emergency_stop",
                        "physical_safety",
                        "welfare_protocol",
                    )
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "hf_dm.rig_certification.build_release_identity",
        lambda _config: {
            "git_commit": "a" * 40,
            "git_exact_tag": "v0.1.0-animal-ready",
            "git_dirty": False,
            "requirements_lock_sha256": "b" * 64,
            "package_versions": {"bpod-core": "0.1.0a12"},
            "config_sha256": "c" * 64,
            "calibration_sha256": "d" * 64,
            "hardware_failsafe_revision": "watchdog-r2",
            "streamdio_driver": {"module_path": "C:/vendor/streamdio.py"},
        },
    )

    report = run_rig_certification(
        config,
        confirmations=CONFIRMATIONS,
        physical_evidence=evidence_path,
        command_runner=_fake_runner,
    )

    assert report["status"] == "PASS"
    assert report["hardware"]["camera"]["serial_number"] == "20137635"
    assert report["hardware"]["bpod"]["firmware"] == [23, 10]
    saved = json.loads((tmp_path / "rig.json").read_text(encoding="utf-8"))
    assert saved["status"] == "PASS"
    assert {item["name"] for item in saved["checks"]} >= REQUIRED_CERTIFICATION_CHECKS


def test_certify_rig_without_physical_evidence_writes_fail_report(tmp_path, monkeypatch):
    config = _write_config(tmp_path)
    monkeypatch.setattr(
        "hf_dm.rig_certification.build_release_identity",
        lambda _config: {
            "git_commit": "a" * 40,
            "git_dirty": False,
            "hardware_failsafe_revision": "watchdog-r2",
        },
    )

    report = run_rig_certification(
        config,
        confirmations=CONFIRMATIONS,
        command_runner=_fake_runner,
    )

    assert report["status"] == "FAIL"
    check = next(item for item in report["checks"] if item["name"] == "speaker_spl")
    assert check["status"] == "FAIL"
    assert (tmp_path / "rig.json").exists()


def test_certify_rig_rejects_stale_physical_evidence(tmp_path):
    config = _write_config(tmp_path)
    evidence_path = tmp_path / "stale-physical.json"
    evidence_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "hf_dm_physical_certification_evidence",
                "created_wall_time_s": time.time() - 31 * 86400,
                "rig_id": "rig1",
                "checks": [],
                "human_approvals": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="future-dated or stale"):
        run_rig_certification(
            config,
            confirmations=CONFIRMATIONS,
            physical_evidence=evidence_path,
            command_runner=_fake_runner,
        )
