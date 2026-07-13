from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from hf_dm.certification import AnimalUseGateError, certified_config_sha256
from hf_dm.certification_gui import CertificationDialogResult
from hf_dm.certification_workflow import (
    append_gui_audit,
    apply_operator_state,
    certification_command,
    load_operator_state,
    save_operator_state,
    write_effective_config,
    write_json_artifact,
)
from hf_dm.cli import (
    _remediate_animal_use_gate,
    _validate_uncertified_bench_request,
    main as cli_main,
)
from hf_dm.config import SessionConfig


def _config(tmp_path: Path) -> SessionConfig:
    config = SessionConfig()
    config.results_dir = str(tmp_path / "sessions")
    config.camera.save_dir = str(tmp_path / "videos")
    config.animal_use.enabled = True
    config.animal_use.certification_path = str(tmp_path / "certifications" / "rig.json")
    config.session_export.source_config_path = str(tmp_path / "configs" / "task.yaml")
    return config


def test_operator_state_roundtrip_applies_supported_fields(tmp_path):
    config = _config(tmp_path)
    values = {
        "operator": "Operator A",
        "reward_calibration_path": tmp_path / "calibrations" / "valve.json",
        "physical_evidence_path": tmp_path / "evidence.json",
        "hardware_failsafe_revision": "watchdog-r2",
        "max_cumulative_reward_ul": "500",
        "subjects_dir": tmp_path / "subjects",
    }

    path = save_operator_state(config, values)
    loaded = load_operator_state(config)
    fresh = _config(tmp_path)
    apply_operator_state(fresh, loaded)

    assert path.is_file()
    assert loaded["operator"] == "Operator A"
    assert fresh.reward.calibration_path == str((tmp_path / "calibrations" / "valve.json").resolve())
    assert fresh.animal_use.hardware_failsafe_revision == "watchdog-r2"
    assert fresh.welfare.max_cumulative_reward_ul == 500.0
    assert fresh.welfare.subjects_dir == str((tmp_path / "subjects").resolve())


def test_gui_audit_is_append_only_jsonl(tmp_path):
    config = _config(tmp_path)

    path = append_gui_audit(config, "opened", operator="tester", detail={"status": "FAIL"})
    append_gui_audit(config, "cancelled", operator="tester")

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["event"] for row in rows] == ["opened", "cancelled"]
    assert all(row["event_id"] and row["wall_time_s"] > 0 for row in rows)


def test_gui_json_artifact_is_valid_and_replaceable(tmp_path):
    path = tmp_path / "physical-evidence.json"

    write_json_artifact(path, {"status": "draft"})
    write_json_artifact(path, {"status": "measured", "checks": []})

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "checks": [],
        "status": "measured",
    }
    assert not list(tmp_path.glob("*.tmp"))


def test_effective_config_preserves_certificate_hash_across_relocated_paths(tmp_path):
    config = _config(tmp_path)
    config.reward.calibration_path = str(tmp_path / "calibrations" / "valve.json")
    config.welfare.subjects_dir = str(tmp_path / "subjects")
    baseline = certified_config_sha256(config)

    path = write_effective_config(config)
    reloaded = SessionConfig.from_yaml(path)

    assert certified_config_sha256(reloaded) == baseline
    assert Path(reloaded.results_dir).is_absolute()
    assert Path(reloaded.camera.save_dir).is_absolute()


def test_certification_command_requires_every_live_safety_confirmation(tmp_path):
    config = _config(tmp_path)
    confirmations = {
        "no_animal_present": True,
        "collection_vessel_installed": True,
        "force_mechanism_clear": True,
        "speaker_exposure_safe": True,
        "emergency_power_control_available": False,
    }

    with pytest.raises(ValueError, match="emergency_power_control_available"):
        certification_command(config, physical_evidence=None, confirmations=confirmations)

    confirmations["emergency_power_control_available"] = True
    command = certification_command(
        config,
        physical_evidence=tmp_path / "evidence.json",
        confirmations=confirmations,
    )
    assert command[-2:] == ["--physical-evidence", str((tmp_path / "evidence.json").resolve())]
    assert "--confirm-emergency-stop" in command


def test_remediation_dialog_cannot_bypass_failed_gate(monkeypatch, tmp_path):
    config = _config(tmp_path)
    report = {"status": "FAIL", "checks": []}

    def always_block(_config, **_kwargs):
        raise AnimalUseGateError(report)

    monkeypatch.setattr("hf_dm.cli.assert_animal_use_ready", always_block)

    with pytest.raises(AnimalUseGateError):
        _remediate_animal_use_gate(
            config,
            operator_confirmed=False,
            dialog_enabled=True,
            dialog_fn=lambda _config, _report: CertificationDialogResult(True, True),
        )


def test_remediation_returns_only_after_final_gate_passes(monkeypatch, tmp_path):
    config = _config(tmp_path)
    report = {"status": "FAIL", "checks": []}
    calls = []

    def first_block_then_pass(_config, **kwargs):
        calls.append(kwargs["operator_confirmed"])
        if len(calls) == 1:
            raise AnimalUseGateError(report)
        return {"status": "PASS", "checks": []}

    monkeypatch.setattr("hf_dm.cli.assert_animal_use_ready", first_block_then_pass)

    confirmed = _remediate_animal_use_gate(
        config,
        operator_confirmed=False,
        dialog_enabled=True,
        dialog_fn=lambda _config, received: CertificationDialogResult(
            received is report,
            True,
        ),
    )

    assert confirmed is True
    assert calls == [False, True]


def test_uncertified_bench_requires_explicit_no_animal_confirmation():
    with pytest.raises(ValueError, match="requires --confirm-no-animal"):
        _validate_uncertified_bench_request(
            allow_uncertified=True,
            confirm_no_animal=False,
            conflicting_mode=False,
        )
    with pytest.raises(ValueError, match="requires --allow-uncertified"):
        _validate_uncertified_bench_request(
            allow_uncertified=False,
            confirm_no_animal=True,
            conflicting_mode=False,
        )

    assert _validate_uncertified_bench_request(
        allow_uncertified=True,
        confirm_no_animal=True,
        conflicting_mode=False,
    )


def test_uncertified_bench_cannot_mask_another_operational_mode():
    with pytest.raises(ValueError, match="only valid for a normal full-task run"):
        _validate_uncertified_bench_request(
            allow_uncertified=True,
            confirm_no_animal=True,
            conflicting_mode=True,
        )


def test_uncertified_full_task_is_labeled_bench_and_audited(
    tmp_path,
    monkeypatch,
    capsys,
):
    config_path = tmp_path / "task.yaml"
    config_path.write_text(
        "\n".join(
            [
                "subject_id: bench-mouse",
                "n_trials: 1",
                f"results_dir: {tmp_path.as_posix()}/sessions",
                "camera:",
                "  record_video: false",
                "  required: false",
                f"  save_dir: {tmp_path.as_posix()}/videos",
                "monitor:",
                "  enabled: false",
                "animal_use:",
                "  enabled: true",
                f"  certification_path: {tmp_path.as_posix()}/missing-certificate.json",
            ]
        ),
        encoding="utf-8",
    )
    captured = {}

    class FakeSession:
        def __init__(self, config, *, run_mode, animal_use_confirmed):
            captured["config"] = config
            captured["run_mode"] = run_mode
            captured["animal_use_confirmed"] = animal_use_confirmed

        def connect(self):
            captured["connected"] = True

        def run(self):
            return types.SimpleNamespace(
                end_reason="completed",
                n_trials_completed=1,
                n_trials_planned=1,
            )

        def close(self, **_kwargs):
            captured["closed"] = True

    monkeypatch.setattr("hf_dm.cli.HFDecisionSession", FakeSession)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hf-dm-run",
            "--config",
            str(config_path),
            "--no-session-dialog",
            "--allow-uncertified",
            "--confirm-no-animal",
        ],
    )

    cli_main()

    assert captured["connected"] is True
    assert captured["run_mode"] == "bench"
    assert captured["animal_use_confirmed"] is False
    overrides = captured["config"].session_export.cli_overrides
    bypass = next(
        item for item in overrides if item["source"].startswith("hf-dm-run --allow-uncertified")
    )
    assert bypass["values"] == {
        "run_mode": "bench",
        "animal_present": False,
        "certification_required": False,
    }
    assert "Session finished: completed" in capsys.readouterr().out


def test_disabled_gate_is_bench_labeled_and_session_name_selects_output_folder(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "task.yaml"
    config_path.write_text(
        "\n".join(
            [
                "subject_id: bench",
                "n_trials: 1",
                f"results_dir: {tmp_path.as_posix()}/sessions",
                "camera:",
                "  record_video: false",
                "  required: false",
                f"  save_dir: {tmp_path.as_posix()}/videos",
                "monitor:",
                "  enabled: false",
                "animal_use:",
                "  enabled: false",
            ]
        ),
        encoding="utf-8",
    )
    captured = {}

    class FakeSession:
        def __init__(self, config, *, run_mode, animal_use_confirmed):
            captured.update(config=config, run_mode=run_mode)

        def connect(self):
            pass

        def run(self):
            return types.SimpleNamespace(
                end_reason="completed", n_trials_completed=1, n_trials_planned=1
            )

        def close(self, **_kwargs):
            pass

    monkeypatch.setattr("hf_dm.cli.HFDecisionSession", FakeSession)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hf-dm-run",
            "--config",
            str(config_path),
            "--no-session-dialog",
            "--session-name",
            "named-run",
        ],
    )

    cli_main()

    expected = tmp_path / "sessions" / "named-run"
    assert captured["run_mode"] == "bench"
    assert Path(captured["config"].results_dir) == expected
    assert captured["config"].task.session_name == "named-run"
    assert captured["config"].task.session_dir == str(expected)
