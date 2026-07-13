from __future__ import annotations

import json
import time
from copy import deepcopy
from unittest.mock import MagicMock

import pytest

from hf_dm.certification import (
    REQUIRED_CERTIFICATION_CHECKS,
    AnimalUseGateError,
    assert_animal_use_ready,
    build_release_identity,
    certified_config_sha256,
    evaluate_animal_use_gate,
)
from hf_dm.config import SessionConfig
from hf_dm.session.controller import HFDecisionSession
from hf_dm.valve_calibration import (
    ValveMeasurement,
    fit_valve_calibration,
    save_valve_calibration,
)


def _release_fixture(tmp_path):
    now = time.time()
    calibration_payload = fit_valve_calibration(
        [
            ValveMeasurement(0.04, 20.0, 10),
            ValveMeasurement(0.08, 40.0, 10),
            ValveMeasurement(0.12, 60.0, 10),
        ]
    )
    calibration_payload["created_wall_s"] = now
    calibration = save_valve_calibration(calibration_payload, tmp_path / "valve.json")
    config = SessionConfig()
    config.results_dir = str(tmp_path / "sessions")
    config.reward.calibration_path = str(calibration.source_path)
    config.welfare.max_cumulative_reward_ul = 500.0
    config.welfare.water_target_ml = 1.0
    config.animal_use.enabled = True
    config.animal_use.certification_path = str(tmp_path / "rig-certification.json")
    config.animal_use.hardware_failsafe_revision = "watchdog-r2"
    provenance = {
        "git_commit": "a" * 40,
        "git_exact_tag": "v0.1.0-animal-ready",
        "git_dirty": False,
        "requirements_lock_sha256": "b" * 64,
        "package_versions": {"bpod-hf-dm": "0.1.0", "bpod-core": "0.1.0a12"},
        "python_version": "3.12.test",
        "streamdio_driver": {
            "module": "pybpod_streamDIO_module",
            "module_path": "C:/vendor/streamdio.py",
            "git_commit": "c" * 40,
            "git_dirty": False,
        },
    }
    release = build_release_identity(config, provenance=provenance)
    certificate = {
        "schema_version": 1,
        "kind": "hf_dm_rig_certification",
        "status": "PASS",
        "created_wall_time_s": now,
        "expires_wall_time_s": now + 7 * 86400,
        "release": release,
        "hardware": {
            "bpod": {"serial_number": "17032730", "firmware": [23, 10]},
            "streamdio": {"firmware_version": 6, "serial_number": "streamdio-rig1"},
            "camera": {"serial_number": "20137635", "firmware_version": "2112.0.125.0"},
        },
        "checks": [
            {"name": name, "status": "PASS", "evidence": {"fixture": True}}
            for name in sorted(REQUIRED_CERTIFICATION_CHECKS)
        ],
        "human_approvals": {
            name: {
                "approved": True,
                "operator": "tester",
                "signature": "tester-signature",
                "signed_wall_time_s": now,
            }
            for name in (
                "calibration",
                "emergency_stop",
                "physical_safety",
                "welfare_protocol",
            )
        },
    }
    (tmp_path / "rig-certification.json").write_text(
        json.dumps(certificate), encoding="utf-8"
    )
    return config, provenance, certificate, now


def test_exact_pass_certification_allows_animal_use(tmp_path):
    config, provenance, certificate, now = _release_fixture(tmp_path)

    report = assert_animal_use_ready(
        config,
        operator_confirmed=True,
        provenance=provenance,
        actual_hardware=certificate["hardware"],
        now_s=now,
        storage_probe=lambda _path: {"pass": True, "fixture": True},
    )

    assert report["status"] == "PASS"
    assert all(item["status"] == "PASS" for item in report["checks"])


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    [
        (lambda config, provenance, certificate, now: provenance.update(git_dirty=True), "clean_git_provenance"),
        (
            lambda config, provenance, certificate, now: certificate.update(
                created_wall_time_s=now - 31 * 86400
            ),
            "certification_current",
        ),
        (
            lambda config, provenance, certificate, now: certificate["checks"][0].update(
                status="WARN"
            ),
            "certified_checks_all_pass",
        ),
        (
            lambda config, provenance, certificate, now: certificate["release"].update(
                git_commit="d" * 40
            ),
            "exact_release_identity",
        ),
        (
            lambda config, provenance, certificate, now: certificate[
                "human_approvals"
            ]["calibration"].update(signed_wall_time_s=now - 31 * 86400),
            "human_approvals",
        ),
    ],
)
def test_release_mismatch_or_nonpass_evidence_blocks_animal_use(
    tmp_path, mutation, failed_check
):
    config, provenance, certificate, now = _release_fixture(tmp_path)
    mutation(config, provenance, certificate, now)
    (tmp_path / "rig-certification.json").write_text(
        json.dumps(certificate), encoding="utf-8"
    )

    report = evaluate_animal_use_gate(
        config,
        operator_confirmed=True,
        provenance=provenance,
        now_s=now,
        storage_probe=lambda _path: {"pass": True},
    )

    assert report["status"] == "FAIL"
    assert next(item for item in report["checks"] if item["name"] == failed_check)[
        "status"
    ] == "FAIL"
    with pytest.raises(AnimalUseGateError):
        assert_animal_use_ready(
            config,
            operator_confirmed=True,
            provenance=provenance,
            now_s=now,
            storage_probe=lambda _path: {"pass": True},
        )


def test_connected_device_identity_mismatch_blocks_animal_use(tmp_path):
    config, provenance, certificate, now = _release_fixture(tmp_path)
    actual = json.loads(json.dumps(certificate["hardware"]))
    actual["camera"]["serial_number"] = "wrong-camera"

    report = evaluate_animal_use_gate(
        config,
        operator_confirmed=True,
        provenance=provenance,
        actual_hardware=actual,
        now_s=now,
        storage_probe=lambda _path: {"pass": True},
    )

    assert report["status"] == "FAIL"
    identity_check = next(
        item for item in report["checks"] if item["name"] == "connected_hardware_matches"
    )
    assert identity_check["detail"]["hardware.camera.serial_number"]["actual"] == "wrong-camera"


def test_operator_confirmation_is_per_run_and_cannot_be_stored_in_certificate(tmp_path):
    config, provenance, _certificate, now = _release_fixture(tmp_path)

    report = evaluate_animal_use_gate(
        config,
        operator_confirmed=False,
        provenance=provenance,
        now_s=now,
        storage_probe=lambda _path: {"pass": True},
    )

    assert report["status"] == "FAIL"
    assert next(
        item for item in report["checks"] if item["name"] == "operator_confirmation"
    )["status"] == "FAIL"


def test_certificate_hash_ignores_per_session_identity_and_storage_paths(tmp_path):
    config, _provenance, _certificate, _now = _release_fixture(tmp_path)
    baseline = certified_config_sha256(config)
    changed = deepcopy(config)
    changed.subject_id = "different-mouse"
    changed.results_dir = str(tmp_path / "other-session")
    changed.camera.save_dir = str(tmp_path / "other-video")
    changed.reward.calibration_path = str(tmp_path / "relocated-calibration.json")
    changed.welfare.subjects_dir = str(tmp_path / "other-subject-records")
    changed.animal_use.certification_path = str(tmp_path / "relocated-certificate.json")
    changed.task.selected_mouse_dir = str(tmp_path / "different-mouse")
    changed.task.session_name = "different-session"
    changed.task.session_dir = str(tmp_path / "different-session")
    changed.session_export.source_config_path = str(tmp_path / "effective.yaml")
    changed.session_export.cli_overrides.append({"created_wall_time_s": time.time()})

    assert certified_config_sha256(changed) == baseline


def test_certificate_hash_still_binds_task_and_safety_values(tmp_path):
    config, _provenance, _certificate, _now = _release_fixture(tmp_path)
    baseline = certified_config_sha256(config)

    for mutate in (
        lambda value: setattr(value, "n_trials", value.n_trials + 1),
        lambda value: setattr(value.reward, "matched_pump_s", 0.09),
        lambda value: setattr(value.timing, "choice_timeout_s", 21.0),
        lambda value: setattr(value.animal_use, "hardware_failsafe_revision", "watchdog-r3"),
    ):
        changed = deepcopy(config)
        mutate(changed)
        assert certified_config_sha256(changed) != baseline


def test_gate_probes_results_and_camera_storage(tmp_path):
    config, provenance, _certificate, now = _release_fixture(tmp_path)
    config.camera.save_dir = str(tmp_path / "videos")
    calls = []

    def probe(path):
        calls.append(str(path))
        return {"pass": "videos" not in str(path)}

    report = evaluate_animal_use_gate(
        config,
        operator_confirmed=True,
        provenance=provenance,
        now_s=now,
        storage_probe=probe,
    )

    assert calls == [config.results_dir, config.camera.save_dir]
    storage = next(
        item for item in report["checks"] if item["name"] == "local_durable_storage"
    )
    assert storage["status"] == "FAIL"
    assert storage["detail"]["paths"]["results"]["pass"] is True
    assert storage["detail"]["paths"]["camera"]["pass"] is False


def test_animal_gate_blocks_before_bpod_or_streamdio_is_opened(tmp_path, monkeypatch):
    config = SessionConfig()
    config.results_dir = str(tmp_path / "sessions")
    config.animal_use.enabled = True
    config.animal_use.certification_path = str(tmp_path / "missing-certification.json")
    session = HFDecisionSession(config, animal_use_confirmed=True)
    session._safety_interlock_path = tmp_path / "interlock.json"
    connect_bpod = MagicMock()
    monkeypatch.setattr("hf_dm.session.controller.connect_bpod_core", connect_bpod)

    with pytest.raises(AnimalUseGateError):
        session.connect()

    connect_bpod.assert_not_called()
    assert not session._safety_interlock_path.exists()
