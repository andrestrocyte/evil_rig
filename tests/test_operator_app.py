from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest
import yaml

from hf_dm.config import SessionConfig
from hf_dm.operator_app import LaunchOptions, command_steps, prepare_config, diagnostic_result
from hf_dm.session.controller import HFDecisionSession
from hf_dm.session.run_control import OperatorStop, RunControl, bounded_close


def test_stop_request_is_cooperative_and_deadline_is_finite(tmp_path, monkeypatch):
    stop = tmp_path / "stop.request"
    monkeypatch.setenv("HF_DM_STOP_REQUEST", str(stop))
    monkeypatch.setenv("HF_DM_RUN_TIMEOUT_S", "0.001")
    control = RunControl()
    time.sleep(0.005)
    with pytest.raises(TimeoutError, match="time limit"):
        control.check()
    monkeypatch.setenv("HF_DM_RUN_TIMEOUT_S", "0")
    control = RunControl()
    stop.touch()
    with pytest.raises(OperatorStop):
        control.check()


def test_hung_driver_close_returns_without_hanging_caller():
    release = threading.Event()
    try:
        with pytest.raises(TimeoutError, match="cleanup"):
            bounded_close(release.wait, timeout_s=0.01)
    finally:
        release.set()


def test_saved_diagnostic_failure_is_not_reported_as_pass(tmp_path):
    import json
    path = tmp_path / "session.json"
    for payload, passed in [({}, False), ({"all_match": False}, False), ({"all_match": True}, True)]:
        path.write_text(json.dumps({"events": [{
            "event_type": "frame2ttl_barcode_diagnostic_end", "payload": payload,
        }]}))
        assert diagnostic_result("screen_barcodes", str(path))["passed"] is passed


def test_saved_camera_count_mismatch_is_not_reported_as_pass(tmp_path):
    import json
    path = tmp_path / "session.json"
    path.write_text(json.dumps({"stream_summaries": {"spinnaker_video": {
        "trigger_alignment": {"status": "FAIL", "recorded_triggers": 30, "saved_frames": 29},
    }}}))
    assert diagnostic_result("session", str(path))["passed"] is False


def test_hung_bpod_reader_never_enters_unbounded_driver_wait():
    release = threading.Event()
    thread = threading.Thread(target=release.wait, daemon=True)
    thread.start()
    session = HFDecisionSession(SessionConfig())
    session.bpod = SimpleNamespace(_read_thread=thread)
    try:
        with pytest.raises(RuntimeError, match="state-machine reader"):
            session._wait_for_bpod_trial_threads(timeout_s=0.01)
    finally:
        release.set()
        thread.join(timeout=1)


def test_hung_softcode_handler_has_a_deadline():
    session = HFDecisionSession(SessionConfig())
    session.bpod = SimpleNamespace(_softcode_thread=SimpleNamespace(
        _idle=threading.Event(), queue=SimpleNamespace(empty=lambda: True), drain=lambda: None,
    ))
    with pytest.raises(RuntimeError, match="softcode handlers"):
        session._wait_for_bpod_trial_threads(timeout_s=0.01)


def test_launcher_preserves_profiles_and_records_absolute_output_paths(tmp_path):
    root = Path(__file__).resolve().parents[1]
    profile = root / "configs/dm_50.yaml"
    before = profile.read_bytes()
    options = LaunchOptions(subject="M001", data_dir=str(tmp_path / "data"), mode="animal")
    effective = prepare_config(root, tmp_path, options)
    raw = yaml.safe_load(effective.read_text())
    config = SessionConfig.from_yaml(effective)
    assert profile.read_bytes() == before
    assert config.n_trials == 10
    assert raw["licks"]["source"] == "optical"
    assert raw["animal_use"]["enabled"] is True
    assert Path(raw["animal_use"]["certification_path"]).parent == root / "certifications"
    assert Path(config.results_dir) == tmp_path / "data"
    assert command_steps("session", options) == [("session", ["--confirm-animal-use"])]
    with pytest.raises(ValueError, match="Bench"):
        command_steps("check", options)


@pytest.mark.parametrize("subject", ["../escape", "a/b", "..", "", "a:b"])
def test_launcher_rejects_subject_path_injection(subject, tmp_path):
    with pytest.raises(ValueError):
        LaunchOptions(subject=subject, data_dir=str(tmp_path)).validate()
