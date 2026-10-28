import json
import sys
from pathlib import Path
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
    module = tmp_path / ".venv/Lib/site-packages/streamdio/__init__.py"
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


@pytest.mark.parametrize("change", ["edit", "add", "delete", "rename", "asset"])
def test_runtime_identity_detects_package_code_and_asset_changes(tmp_path, change):
    from hf_dm.data.recorder import _runtime_code_identity

    package = tmp_path / "hf_dm"
    package.mkdir()
    source = package / "__init__.py"
    source.write_bytes(b"REVISION = 1\n")
    asset = package / "defaults.yaml"
    asset.write_bytes(b"speaker_pin: 26\n")
    original = _runtime_code_identity(package)["runtime_code_sha256"]
    assert original
    if change == "edit":
        source.write_bytes(b"REVISION = 2\n")
    elif change == "add":
        (package / "new_module.py").write_bytes(b"VALUE = 1\n")
    elif change == "delete":
        source.unlink()
    elif change == "rename":
        source.rename(package / "renamed.py")
    else:
        asset.write_bytes(b"speaker_pin: 27\n")
    assert _runtime_code_identity(package)["runtime_code_sha256"] != original


def test_runtime_identity_ignores_checkout_location_docs_and_bytecode(tmp_path):
    from hf_dm.data.recorder import _runtime_code_identity

    hashes = []
    for folder in ("first", "second"):
        root = tmp_path / folder
        package = root / "src/hf_dm"
        package.mkdir(parents=True)
        (package / "__init__.py").write_bytes(b"REVISION = 1\n")
        (root / "README.md").write_text(folder)
        (package / "__init__.pyc").write_text(folder)
        cache = package / "__pycache__"
        cache.mkdir()
        (cache / "cache-file").write_text(folder)
        hashes.append(_runtime_code_identity(package)["runtime_code_sha256"])
    assert hashes[0] and hashes[0] == hashes[1]


@pytest.mark.parametrize("failure", ["missing", "empty", "unreadable", "enumeration"])
def test_runtime_identity_failure_never_returns_partial_hash(tmp_path, monkeypatch, failure):
    from hf_dm.data import recorder

    package = tmp_path / "hf_dm"
    if failure != "missing":
        package.mkdir()
    if failure in {"unreadable", "enumeration"}:
        (package / "a.py").write_bytes(b"# readable\n")
        blocked = package / "z.py"
        blocked.write_bytes(b"# unreadable\n")
        if failure == "unreadable":
            original_open = Path.open

            def open_file(path, *args, **kwargs):
                if path == blocked:
                    raise PermissionError("fixture unreadable source")
                return original_open(path, *args, **kwargs)

            monkeypatch.setattr(Path, "open", open_file)
        else:
            def walk(path, *, onerror):
                onerror(PermissionError("fixture directory enumeration failure"))

            monkeypatch.setattr(recorder.os, "walk", walk)
    identity = recorder._runtime_code_identity(package)
    assert identity["runtime_code_identity_scheme"] == "hf_dm_package_tree_v1"
    assert identity["runtime_code_sha256"] is None
    assert identity["runtime_code_error"]


def test_runtime_provenance_hashes_installed_package_instead_of_git_cwd(tmp_path, monkeypatch):
    from hf_dm.data import recorder

    package = tmp_path / "installed/hf_dm"
    module = package / "data/recorder.py"
    module.parent.mkdir(parents=True)
    module.write_bytes(b"# actual runtime source\n")
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    monkeypatch.setattr(recorder, "__file__", str(module))
    monkeypatch.setattr(recorder, "_git_identity", lambda _: {"git_root": str(unrelated)})
    provenance = recorder.capture_runtime_provenance(unrelated)
    assert provenance["runtime_code_sha256"] == recorder._runtime_code_identity(package)["runtime_code_sha256"]
    (unrelated / "other.py").write_bytes(b"# unrelated checkout\n")
    assert recorder.capture_runtime_provenance(unrelated)["runtime_code_sha256"] == provenance["runtime_code_sha256"]
