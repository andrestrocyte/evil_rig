import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hf_dm.audio.white_noise import WhiteNoisePlayer


def test_failed_audio_start_releases_partially_started_stream(monkeypatch):
    control = SimpleNamespace(
        start_continuous_playback_from_protocol=lambda: True,
        get_status=lambda: {"continuous_playback_active": False},
        stop_and_cleanup=Mock(),
    )
    monkeypatch.setitem(sys.modules, "pybpod_soundcard_module", SimpleNamespace(
        SoundControl=SimpleNamespace(create_from_protocol=lambda **kwargs: control),
    ))
    player = WhiteNoisePlayer()
    assert player.start_continuous() is False
    control.stop_and_cleanup.assert_called_once()
    assert player._control is None


def test_audio_stop_failure_is_visible_to_cleanup_interlock():
    player = WhiteNoisePlayer()
    player._control = SimpleNamespace(stop_and_cleanup=Mock(side_effect=OSError("audio stuck")))
    with pytest.raises(OSError, match="audio stuck"):
        player.stop()
    assert player._control is not None


def test_wheel_driver_provenance_does_not_inherit_venv_parent_git(tmp_path, monkeypatch):
    from hf_dm.data import recorder
    module = tmp_path / ".venv/Lib/site-packages/pybpod_streamDIO_module/__init__.py"
    module.parent.mkdir(parents=True)
    module.write_text("# driver package\n")
    implementation = module.parent / "driver.py"
    implementation.write_text("REVISION = 1\n")
    monkeypatch.setattr(recorder.importlib.util, "find_spec", lambda name: SimpleNamespace(origin=str(module)))
    monkeypatch.setattr(recorder, "installed_distribution", lambda name: SimpleNamespace(
        read_text=lambda file: json.dumps({"url": "https://example.org/driver.git", "vcs_info": {"commit_id": "driver-revision"}}),
    ))
    git = Mock(return_value={"git_root": str(tmp_path), "git_commit": "task-revision"})
    monkeypatch.setattr(recorder, "_git_identity", git)
    result = recorder.capture_runtime_provenance(tmp_path)
    assert result["streamdio_driver"]["git_commit"] == "driver-revision"
    assert result["streamdio_driver"]["git_root"] is None
    git.assert_called_once_with(tmp_path)
    first_hash = result["streamdio_driver"]["source_sha256"]
    implementation.write_text("REVISION = 2\n")
    assert recorder.capture_runtime_provenance(tmp_path)["streamdio_driver"]["source_sha256"] != first_hash


def test_self_test_resets_selected_trial_counts_and_modality():
    from hf_dm.config import SessionConfig, validate_session_config, TrainingPhase
    from hf_dm.session.hardware_self_test import shorten_for_hardware_self_test
    config = SessionConfig()
    config.phase = TrainingPhase.CENTERING
    config.n_trials = 10
    config.task.n_free_trials = 5
    config.task.n_forced_trials = 5
    config.task.modality = "delay"
    shorten_for_hardware_self_test(config)
    validate_session_config(config)
    assert config.n_trials == config.task.n_free_trials == 3
    assert config.task.n_forced_trials == 0
    assert config.task.modality == "force"
