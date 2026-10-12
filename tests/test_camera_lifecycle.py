"""Resource ownership across camera startup failures and finite cleanup retries."""

import ctypes
import json
from types import SimpleNamespace
import weakref

import pytest

from hf_dm.camera import spinnaker


class UnstartedThread:
    def __init__(self, *args, **kwargs):
        pass

    def start(self):
        pass

    def is_alive(self):
        return False

    def join(self, *args, **kwargs):
        pytest.fail("must not join an unstarted/already stopped thread")


def pyspin_recorder(tmp_path, monkeypatch, failure=None, exception=RuntimeError):
    calls = []
    r = spinnaker.HeadlessSpinnakerRecorder(tmp_path, "S1", "test", serial_number="fake")

    def action(name):
        calls.append(name)
        if name == failure:
            raise exception(f"injected {name}")

    camera = SimpleNamespace(
        TLDevice=SimpleNamespace(DeviceSerialNumber=SimpleNamespace(GetValue=lambda: "fake")),
        Init=lambda: action("init"), DeInit=lambda: action("deinit"),
        BeginAcquisition=lambda: action("begin"), EndAcquisition=lambda: action("end"),
    )
    cameras = SimpleNamespace(
        GetSize=lambda: action("count") or 1,
        GetByIndex=lambda index: action("select") or camera,
        Clear=lambda: action("clear_list"),
    )
    system = SimpleNamespace(GetCameras=lambda: action("enumerate") or cameras,
                             ReleaseInstance=lambda: action("release_system"))
    monkeypatch.setattr(r, "_load_pyspin", lambda: SimpleNamespace(System=SimpleNamespace(GetInstance=lambda: system)))
    monkeypatch.setattr(r, "_configure_camera", lambda: action("configure"))

    class CaptureThread(UnstartedThread):
        def start(self):
            action("thread_start")

    monkeypatch.setattr(spinnaker.threading, "Thread", CaptureThread)
    return r, calls, camera, cameras, system


@pytest.mark.parametrize("failure", ["enumerate", "count", "select", "init", "configure", "begin", "thread_start"])
def test_pyspin_failed_start_releases_only_owned_resources_and_keeps_original_error(tmp_path, monkeypatch, failure):
    r, calls, _, _, _ = pyspin_recorder(tmp_path, monkeypatch, failure)
    with pytest.raises(RuntimeError, match=f"injected {failure}"):
        r.start()
    assert r._running is False
    assert r._camera is r._camera_list is r._system is None
    assert "release_system" in calls
    assert ("clear_list" in calls) == (failure != "enumerate")
    assert ("deinit" in calls) == (failure in {"configure", "begin", "thread_start"})
    assert ("end" in calls) == (failure == "thread_start")
    if r._timestamps is not None:
        assert r._timestamps._handle.closed
    calls_after_failure = list(calls)
    summary = r.stop()
    assert calls == calls_after_failure
    assert summary["capture_error"] == f"RuntimeError: injected {failure}"
    assert summary["cleanup_error"] is None


def test_pyspin_keyboard_interrupt_during_start_also_cleans_up(tmp_path, monkeypatch):
    r, calls, *_ = pyspin_recorder(tmp_path, monkeypatch, "configure", KeyboardInterrupt)
    with pytest.raises(KeyboardInterrupt, match="configure"):
        r.start()
    assert calls[-2:] == ["deinit", "release_system"]
    assert r._system is None


def test_camera_start_is_single_use_and_cannot_overwrite_live_files(tmp_path, monkeypatch):
    r, calls, *_ = pyspin_recorder(tmp_path, monkeypatch)
    r.start()
    before = list(calls)
    with pytest.raises(RuntimeError, match="single-use"):
        r.start()
    assert calls == before
    assert r._acquisition_started
    first = r.stop()
    assert first["cleanup_error"] is None
    with pytest.raises(RuntimeError, match="single-use"):
        r.start()
    assert r.stop()["cleanup_error"] is None
    assert calls.count("end") == calls.count("deinit") == calls.count("release_system") == 1


@pytest.mark.parametrize("stage", ["end", "deinit", "release_system"])
def test_pyspin_cleanup_retains_failed_resource_until_successful_retry(tmp_path, monkeypatch, stage):
    r, calls, camera, _, system = pyspin_recorder(tmp_path, monkeypatch)
    r.start()
    owner, method = {"end": (camera, "EndAcquisition"), "deinit": (camera, "DeInit"),
                     "release_system": (system, "ReleaseInstance")}[stage]
    original = getattr(owner, method)

    def fail():
        raise RuntimeError("cleanup injected")

    setattr(owner, method, fail)
    first = r.stop()
    assert "cleanup injected" in first["cleanup_error"]
    assert r._system is system
    if stage != "release_system":
        assert r._camera is camera
    setattr(owner, method, original)
    second = r.stop()
    assert second["cleanup_error"] is None
    assert second["cleanup_error_history"] == first["cleanup_error_history"]
    assert r._camera is r._system is None


def test_pyspin_serial_lookup_does_not_pin_unmatched_camera_in_exception_traceback(tmp_path, monkeypatch):
    r, calls, _, cameras, _ = pyspin_recorder(tmp_path, monkeypatch)
    seen = []

    class Camera:
        TLDevice = SimpleNamespace(DeviceSerialNumber=SimpleNamespace(GetValue=lambda: "different"))

    def get_camera(index):
        camera = Camera()
        seen.append(weakref.ref(camera))
        return camera

    cameras.GetByIndex = get_camera
    with pytest.raises(RuntimeError, match="serial not found") as captured:
        r.start()
    assert captured.value is not None
    assert seen and seen[0]() is None
    assert calls[-2:] == ["clear_list", "release_system"]


def c_recorder(tmp_path):
    r = spinnaker._SpinnakerCRecorder(tmp_path, serial_number=None, fps=30, exposure_us=10000,
                                    pixel_format="Mono8", gain_db=None, gamma=None, acquisition_timeout_ms=10)
    r._system = ctypes.c_void_p(1)
    r._camera_list = ctypes.c_void_p(2)
    r._camera = ctypes.c_void_p(3)
    r._camera_initialized = r._acquisition_started = True
    return r


@pytest.mark.parametrize("stage", ["spinCameraEndAcquisition", "spinCameraDeInit", "spinCameraRelease",
                                  "spinCameraListClear", "spinCameraListDestroy", "spinSystemReleaseInstance"])
def test_c_cleanup_retains_handles_and_stops_before_releasing_dependent_parents(tmp_path, stage):
    r = c_recorder(tmp_path)
    order = ["spinCameraEndAcquisition", "spinCameraDeInit", "spinCameraRelease",
             "spinCameraListClear", "spinCameraListDestroy", "spinSystemReleaseInstance"]
    calls = []

    def call(name):
        def invoke(handle):
            calls.append(name)
            return -1000 if name == stage else 0
        return invoke

    r._fn = {name: call(name) for name in order}
    first = r.stop()
    assert calls == order[:order.index(stage) + 1]
    assert stage in first["cleanup_error"]
    assert r._system.value == 1
    r._fn[stage] = lambda handle: 0
    second = r.stop()
    assert second["cleanup_error"] is None
    assert second["cleanup_error_history"] == first["cleanup_error_history"]
    assert not r._camera.value and not r._camera_list.value and not r._system.value
    assert json.loads((tmp_path / "recording_summary.json").read_text())["cleanup_error_history"]


def test_timestamp_flush_failure_still_closes_file_and_does_not_mask_startup_error(tmp_path, monkeypatch):
    r, calls, *_ = pyspin_recorder(tmp_path, monkeypatch, "thread_start")
    closed = []

    class BrokenHandle:
        closed = False

        def flush(self):
            raise OSError("flush failed")

        def close(self):
            self.closed = True
            closed.append(True)

    writer = object.__new__(spinnaker._FrameTimestampWriter)
    writer._handle = BrokenHandle()
    monkeypatch.setattr(spinnaker, "_FrameTimestampWriter", lambda path: writer)
    with pytest.raises(RuntimeError, match="thread_start") as captured:
        r.start()
    assert closed == [True]
    assert "flush failed" in " ".join(captured.value.__notes__)
    assert r._system is None
    assert calls[-3:] == ["end", "deinit", "release_system"]


def test_preview_stop_failure_does_not_skip_sdk_or_timestamp_cleanup(tmp_path, monkeypatch):
    r, calls, *_ = pyspin_recorder(tmp_path, monkeypatch)
    r.start()

    def fail():
        raise RuntimeError("preview stop failed")

    monkeypatch.setattr(r._preview_dispatcher, "stop", fail)
    summary = r.stop()
    assert "preview stop failed" in summary["cleanup_error"]
    assert calls[-3:] == ["end", "deinit", "release_system"]
    assert r._timestamps._handle.closed
