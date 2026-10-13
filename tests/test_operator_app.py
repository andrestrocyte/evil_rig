from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest
import yaml

from hf_dm.config import SessionConfig
from hf_dm.operator_app import LaunchOptions, RigSetupJob, command_steps, prepare_config, diagnostic_result
from hf_dm.session.controller import HFDecisionSession
from hf_dm.session.run_control import OperatorStop, RunControl, bounded_close


def test_verified_cleanup_allows_bounded_large_export_after_stop(tmp_path, monkeypatch):
    from hf_dm.operator_app import JobWatchdog, hardware_closed_signal
    phase = tmp_path / "phase.json"
    monkeypatch.setenv("HF_DM_PHASE_PATH", str(phase))
    monkeypatch.setenv("HF_DM_RUN_TOKEN", "this-launch")
    control = RunControl()
    watch = JobWatchdog(0, 60)
    assert watch.check(10, stop_requested=True, hardware_closed=False) is None
    assert not hardware_closed_signal(phase, "this-launch")
    control.mark_hardware_closed()
    assert not hardware_closed_signal(phase, "previous-launch")
    assert hardware_closed_signal(phase, "this-launch")
    assert watch.check(12, stop_requested=True, hardware_closed=True) is None
    assert watch.check(100, stop_requested=True, hardware_closed=True) is None
    assert not watch.timed_out
    assert "finalization timed out" in watch.check(313, stop_requested=True, hardware_closed=True)
    control.clear_hardware_closed()
    assert not hardware_closed_signal(phase, "this-launch")


def test_unverified_hardware_cleanup_keeps_short_stop_deadline():
    from hf_dm.operator_app import JobWatchdog
    watch = JobWatchdog(0, 20)
    assert watch.check(21, stop_requested=False, hardware_closed=False) is None
    assert watch.timed_out
    assert "failed to stop" in watch.check(52, stop_requested=True, hardware_closed=False)


def test_saved_data_validation_has_a_process_timeout(monkeypatch):
    import io
    import subprocess
    from hf_dm.operator_app import validate_saved_session
    def stalled(*args, **kwargs):
        assert kwargs["timeout"] == 300
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])
    monkeypatch.setattr(subprocess, "run", stalled)
    with pytest.raises(subprocess.TimeoutExpired):
        validate_saved_session("session.json", io.BytesIO())


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


@pytest.mark.parametrize("mode,backend,frames", [
    ("bench", "spinnaker", 0),
    ("bench", "spinnaker_sim", 0),
    ("simulation", "spinnaker", 0),
    ("simulation", "spinnaker_sim", 1),
])
def test_camera_not_applicable_cannot_hide_real_recording_failure(tmp_path, mode, backend, frames):
    import json
    path = tmp_path / "session.json"
    path.write_text(json.dumps({"metadata": {"run_mode": mode}, "stream_summaries": {
        "spinnaker_video": {"backend": backend, "frames_saved": frames,
                            "trigger_alignment": {"status": "NOT_APPLICABLE"}},
    }}))
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


def setup_job(tmp_path):
    import shutil
    root = tmp_path / "rig checkout"
    (root / "configs").mkdir(parents=True)
    shutil.copyfile(Path(__file__).resolve().parents[1] / "configs/dm_50.yaml", root / "configs/dm_50.yaml")
    options = LaunchOptions(subject="setup_test", lick_source="optical", data_dir=str(tmp_path / "recordings"))
    return RigSetupJob(root, options), options


def test_rig_setup_uses_selected_options_without_starting_a_task(tmp_path, monkeypatch):
    import json
    import subprocess
    job, options = setup_job(tmp_path)
    process = SimpleNamespace(pid=123, poll=lambda: None)
    calls = []

    def launch(command, **kwargs):
        calls.append((command, kwargs))
        return process

    monkeypatch.setattr(subprocess, "Popen", launch)
    try:
        job.start()
        assert job.running
        command, kwargs = calls[0]
        assert command[1:4] == ["-m", "hf_dm.ops_cli", "certification-gui"]
        assert "--confirm-animal-use" not in command
        assert "--hardware-self-test" not in command
        assert command[-1] == str(job.config)
        assert kwargs["stdout"] is job._log_handle
        assert kwargs["env"]["PYTHONUTF8"] == "1"
        config = SessionConfig.from_yaml(job.config)
        assert config.licks.source == "optical"
        assert config.welfare.append_session_water is True
        assert options.mode == "bench"
        assert Path(config.results_dir) == Path(options.data_dir)
        assert json.loads((job.run_dir / "result.json").read_text())["status"] == "running"
        with pytest.raises(RuntimeError, match="already started"):
            job.start()
        assert len(calls) == 1
    finally:
        process.poll = lambda: 0
        assert not job.running


@pytest.mark.parametrize("returncode, expected", [(0, "closed"), (1, "failed")])
def test_setup_close_releases_log_without_claiming_readiness(tmp_path, monkeypatch, returncode, expected):
    import json
    import subprocess
    job, _ = setup_job(tmp_path)
    process = SimpleNamespace(pid=123, poll=lambda: returncode)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: process)
    job.start()
    handle = job._log_handle
    assert not job.running
    assert handle.closed
    report = json.loads((job.run_dir / "result.json").read_text())
    assert report["status"] == expected and report["returncode"] == returncode
    assert "ready" not in report and "passed" not in report


def test_setup_launch_failure_closes_log_and_records_error(tmp_path, monkeypatch):
    import json
    import subprocess
    job, _ = setup_job(tmp_path)
    handles = []

    def fail(*args, **kwargs):
        handles.append(kwargs["stdout"])
        raise OSError("injected launch failure")

    monkeypatch.setattr(subprocess, "Popen", fail)
    with pytest.raises(OSError, match="injected launch failure"):
        job.start()
    assert handles[0].closed
    assert not job.running
    assert json.loads((job.run_dir / "result.json").read_text())["status"] == "failed"


def test_setup_result_write_failure_does_not_break_gui_polling(tmp_path, monkeypatch):
    import subprocess
    job, _ = setup_job(tmp_path)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: SimpleNamespace(pid=123, poll=lambda: 0))
    job.start()

    def fail():
        raise OSError("injected result write failure")

    monkeypatch.setattr(job, "_save_report", fail)
    assert not job.running
    assert "could not be saved" in job.status
    assert "injected result write failure" in job.report["report_error"]
