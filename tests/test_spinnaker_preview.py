import ctypes
import threading
import sys
import types

import pytest

from hf_dm.camera.spinnaker import (
    HeadlessSpinnakerRecorder,
    _PreviewDispatcher,
    _SpinnakerCRecorder,
    _downsample_pgm_file_preview,
)
from hf_dm.camera.spinnaker import _capture_stats
from hf_dm.data.video_export import export_preview_video


def test_preview_dispatcher_keeps_latest_frame_when_callback_is_busy():
    received: list[int] = []
    callback_started = threading.Event()
    release_callback = threading.Event()

    def callback(frame: dict) -> None:
        received.append(int(frame["frame_index"]))
        callback_started.set()
        release_callback.wait(timeout=1.0)

    dispatcher = _PreviewDispatcher(callback, thread_name="test-preview-dispatcher")
    dispatcher.start()
    try:
        assert dispatcher.submit({"frame_index": 1})
        assert callback_started.wait(timeout=1.0)
        for frame_index in range(2, 10):
            assert dispatcher.submit({"frame_index": frame_index})
        release_callback.set()
    finally:
        dispatcher.stop()

    assert received[0] == 1
    assert received[-1] == 9
    assert dispatcher.submitted_count == 9
    assert dispatcher.delivered_count == len(received)


def test_downsample_pgm_file_preview_reads_saved_spinnaker_mono_frame(tmp_path):
    path = tmp_path / "frame_000000001.pgm"
    pixels = bytes(range(12))
    path.write_bytes(b"P5\n# camera frame\n4 3\n255\n" + pixels)

    preview = _downsample_pgm_file_preview(path, max_width=2, max_height=2)

    assert preview == (2, 2, bytes([0, 2, 8, 10]))


def test_downsample_pgm_file_preview_handles_crlf_header_separator(tmp_path):
    path = tmp_path / "frame_000000002.pgm"
    pixels = bytes(range(12))
    path.write_bytes(b"P5\r\n4 3\r\n255\r\n" + pixels)

    preview = _downsample_pgm_file_preview(path, max_width=2, max_height=2)

    assert preview == (2, 2, bytes([0, 2, 8, 10]))


def test_export_preview_video_writes_saved_pgm_sequence(tmp_path, monkeypatch):
    video_dir = tmp_path / "video"
    video_dir.mkdir()
    (video_dir / "recording_summary.json").write_text(
        '{"target_trigger_hz": 20}',
        encoding="utf-8",
    )
    for index in range(2):
        (video_dir / f"frame_{index:09d}.pgm").write_bytes(
            b"P5\n2 2\n255\n" + bytes([index, index + 1, index + 2, index + 3])
        )

    written_shapes = []

    class FakeWriter:
        def __init__(self, *args):
            self.args = args

        def isOpened(self):
            return True

        def write(self, frame):
            written_shapes.append(tuple(frame.shape))

        def release(self):
            pass

    fake_cv2 = types.SimpleNamespace(
        VideoWriter_fourcc=lambda *codec: 0,
        VideoWriter=FakeWriter,
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    result = export_preview_video(video_dir)

    assert result["frames_written"] == 2
    assert result["fps"] == 20.0
    assert written_shapes == [(2, 2), (2, 2)]
    assert (video_dir / "preview_video_export.json").exists()


def test_capture_stats_counts_saved_skipped_and_observed_frames():
    stats = _capture_stats(
        frames_saved=10,
        incomplete_frames=2,
        acquisition_timeouts=3,
        save_errors=1,
    )

    assert stats == {
        "frames_saved": 10,
        "incomplete_frames": 2,
        "acquisition_timeouts": 3,
        "save_errors": 1,
        "frames_skipped": 3,
        "frames_observed": 13,
    }


def test_camera_wait_timeouts_do_not_invent_lost_frames():
    stats = _capture_stats(frames_saved=10, incomplete_frames=0, acquisition_timeouts=3)
    assert stats["frames_observed"] == 10
    assert stats["frames_skipped"] == 0
    assert stats["acquisition_timeouts"] == 3


def test_spinnaker_does_not_teardown_sdk_while_capture_thread_is_alive():
    class HungThread:
        def join(self, timeout):
            self.timeout = timeout

        def is_alive(self):
            return True

    recorder = object.__new__(HeadlessSpinnakerRecorder)
    recorder._c_backend = None
    recorder._running = True
    recorder._thread = HungThread()
    recorder._acquisition_timeout_ms = 10
    recorder._preview_dispatcher = types.SimpleNamespace(stop=lambda: None)
    recorder._error = None
    recorder._cleanup_pyspin = lambda: (_ for _ in ()).throw(
        AssertionError("SDK teardown must not run")
    )

    with pytest.raises(TimeoutError, match="capture thread did not stop"):
        recorder.stop()


@pytest.mark.parametrize("failure", ["node", "entry", "read", "set"])
def test_spinnaker_c_required_enums_fail_closed(failure):
    recorder = object.__new__(_SpinnakerCRecorder)
    recorder._get_node = lambda _node_map, _name: (
        None if failure == "node" else ctypes.c_void_p(1)
    )

    def get_entry(_node, _name, entry_pointer):
        if failure == "entry":
            return 1
        ctypes.cast(entry_pointer, ctypes.POINTER(ctypes.c_void_p)).contents.value = 2
        return 0

    def read_entry(_entry, value_pointer):
        if failure == "read":
            return 1
        ctypes.cast(value_pointer, ctypes.POINTER(ctypes.c_int64)).contents.value = 3
        return 0

    recorder._fn = {
        "spinEnumerationGetEntryByName": get_entry,
        "spinEnumerationEntryGetIntValue": read_entry,
        "spinEnumerationSetIntValue": lambda _node, _value: 1 if failure == "set" else 0,
    }

    with pytest.raises(RuntimeError, match="Required Spinnaker enum TriggerMode=On"):
        recorder._set_enum(ctypes.c_void_p(), "TriggerMode", "On", required=True)
    recorder._set_enum(ctypes.c_void_p(), "PixelFormat", "Mono8")


@pytest.mark.parametrize("failure", ["node", "entry", "read", "set"])
def test_pyspin_required_enums_fail_closed(failure):
    class Entry:
        available = failure != "entry"
        readable = failure != "entry"

        def GetValue(self):
            if failure == "read":
                raise RuntimeError("read failed")
            return 3

    class Node:
        available = failure != "node"
        writable = failure != "node"

        def GetEntryByName(self, _name):
            return Entry()

        def SetIntValue(self, _value):
            if failure == "set":
                raise RuntimeError("set failed")

    recorder = object.__new__(HeadlessSpinnakerRecorder)
    recorder._pyspin = types.SimpleNamespace(
        CEnumerationPtr=lambda node: node,
        IsAvailable=lambda node: node.available,
        IsWritable=lambda node: node.writable,
        IsReadable=lambda entry: entry.readable,
    )
    nodemap = types.SimpleNamespace(GetNode=lambda _name: Node())

    with pytest.raises(RuntimeError, match="Required PySpin enum TriggerMode=On"):
        recorder._set_pyspin_enum(nodemap, "TriggerMode", "On", required=True)
    recorder._set_pyspin_enum(nodemap, "PixelFormat", "Mono8")


def _bare_headless_recorder() -> HeadlessSpinnakerRecorder:
    recorder = object.__new__(HeadlessSpinnakerRecorder)
    recorder._error = None
    recorder._thread = None
    recorder._save_errors = 0
    recorder._c_backend = None
    return recorder


def test_health_reasons_reports_all_failure_modes():
    recorder = _bare_headless_recorder()
    assert recorder.health_reasons() == []

    recorder._error = "camera disconnected"
    recorder._thread = types.SimpleNamespace(is_alive=lambda: False)
    recorder._save_errors = 2

    assert recorder.health_reasons() == [
        "capture_error: camera disconnected",
        "capture_thread_not_alive",
        "frame_save_errors: 2",
    ]


def test_health_reasons_delegates_to_c_backend():
    backend = object.__new__(_SpinnakerCRecorder)
    backend._error = "spin error"
    backend._thread = None
    backend._save_errors = 0

    recorder = _bare_headless_recorder()
    recorder._error = "ignored because backend owns capture"
    recorder._c_backend = backend

    assert recorder.health_reasons() == ["capture_error: spin error"]


@pytest.mark.parametrize("save_fails", [False, True])
def test_c_capture_copies_mono_pixels_preserves_padding_and_releases_image(tmp_path, save_fails):
    import ctypes
    from hf_dm.camera.spinnaker import _SpinnakerCRecorder

    recorder = _SpinnakerCRecorder(
        tmp_path, serial_number=None, fps=30, exposure_us=10000,
        pixel_format="Mono8", gain_db=None, gamma=None, acquisition_timeout_ms=100,
        preview_callback=lambda frame: None, preview_min_interval_s=0,
    )
    raw = ctypes.create_string_buffer(bytes([1, 2, 3, 4, 99, 99, 5, 6, 7, 8, 99, 99]))
    previews, released, timestamps = [], [], []
    recorder._preview_dispatcher = types.SimpleNamespace(
        enabled=True, submit=lambda frame: previews.append(frame) or True,
    )
    recorder._timestamps = types.SimpleNamespace(write=lambda *args, **kw: timestamps.append((args, kw)))

    def assign(value):
        def call(_image, pointer):
            pointer._obj.value = value
            return 0
        return call

    def acquire(_camera, _timeout, pointer):
        pointer._obj.value = 1
        recorder._running = False
        return 0

    def release(_image):
        released.append(True)
        ctypes.memset(raw, 0, len(raw))
        return 0

    recorder._fn = {
        "spinCameraGetNextImageEx": acquire,
        "spinImageIsIncomplete": assign(0),
        "spinImageGetTimeStamp": assign(123456789),
        "spinImageGetWidth": assign(4),
        "spinImageGetHeight": assign(2),
        "spinImageGetStride": assign(6),
        "spinImageGetData": assign(ctypes.addressof(raw)),
        "spinImageRelease": release,
        "spinImageSaveFromExt": lambda *args: pytest.fail("Mono8 must bypass SDK encoder"),
    }
    target = tmp_path / "frame_000000000.pgm"
    if save_fails:
        target.mkdir()
    recorder._running = True
    recorder._capture_loop()
    assert released == [True]
    if save_fails:
        assert recorder.error is not None
        assert recorder._save_errors == 1
        assert recorder.frame_count == 0
        assert not timestamps
    else:
        assert recorder.error is None
        assert recorder.frame_count == 1
        assert target.read_bytes() == b"P5\n4 2\n255\n" + bytes(range(1, 9))
        assert timestamps[0][0] == (0, 123456789, target.name)
        assert timestamps[0][1]["received_mono_s"] == previews[0]["acquired_mono_s"]
        assert previews[0]["data"] == bytes(range(1, 9))
        assert previews[0]["acquired_mono_s"] <= previews[0]["monotonic_s"]


def test_preview_cadence_keeps_target_rate_under_jitter_without_catchup():
    from hf_dm.camera.spinnaker import _PreviewCadence

    cadence = _PreviewCadence(1 / 15)
    emitted = [index for index in range(300) if cadence.due(
        100 + index / 30 + (0.002 if index % 2 else 0)
    )]
    assert 149 <= len(emitted) <= 151
    assert cadence.due(200)
    assert not cadence.due(200.001)


def test_c_mono_fast_path_retains_sdk_fallback_for_other_pixel_formats(tmp_path):
    from hf_dm.camera.spinnaker import _SpinnakerCRecorder
    recorder = _SpinnakerCRecorder(
        tmp_path, serial_number=None, fps=30, exposure_us=10000,
        pixel_format="Mono16", gain_db=None, gamma=None, acquisition_timeout_ms=100,
    )
    assert recorder._read_mono_image(None) is None


def test_c_camera_configuration_unlocks_trigger_nodes_before_required_writes():
    from hf_dm.camera.spinnaker import _SpinnakerCRecorder
    recorder = object.__new__(_SpinnakerCRecorder)
    recorder._camera = None
    recorder._exposure_us = 10000
    recorder._pixel_format = "Mono8"
    recorder._gain_db = recorder._gamma = None
    recorder._check = lambda *args: None
    recorder._apply_roi = lambda *args: None
    recorder._record_optional_settings = lambda *args: None
    recorder._set_bool = recorder._set_float = lambda *args, **kwargs: None
    mode = "TriggerWidth"
    commands = []

    def set_enum(_map, name, value, **kwargs):
        nonlocal mode
        if name in {"TriggerMode", "ExposureAuto"} and mode == "TriggerWidth":
            raise RuntimeError("Node unavailable in TriggerWidth mode")
        commands.append((name, value))
        if name == "ExposureMode":
            mode = value

    recorder._set_enum = set_enum
    recorder._configure_camera()
    assert mode == "TriggerWidth"
    assert commands.index(("ExposureMode", "Timed")) < commands.index(("TriggerMode", "Off"))
    assert commands.index(("TriggerSource", "Line0")) < commands.index(("TriggerMode", "On"))
    assert commands[-2:] == [("TriggerMode", "On"), ("ExposureMode", "TriggerWidth")]


@pytest.mark.parametrize("save_fails", [False, True])
def test_pyspin_copies_mono_pixels_previews_before_save_and_releases(tmp_path, monkeypatch, save_fails):
    import csv
    import numpy as np
    from hf_dm.camera import spinnaker

    r = HeadlessSpinnakerRecorder(tmp_path, "test", "test", preview_callback=lambda frame: None)
    r.session_dir.mkdir()
    pixels = np.arange(12, dtype=np.uint8).reshape(2, 6)[:, :4]  # padded, noncontiguous view
    expected = pixels.tobytes()
    previews, actions = [], []
    r._preview_dispatcher = types.SimpleNamespace(enabled=True, submit=lambda frame: previews.append(frame) or True)
    r._timestamps = spinnaker._FrameTimestampWriter(r.session_dir)

    class Image:
        def IsIncomplete(self):
            return False

        def GetTimeStamp(self):
            return 123456789

        def GetNDArray(self):
            return pixels

        def Save(self, path):
            pytest.fail("Mono8 must bypass the SDK encoder")

        def Release(self):
            actions.append("release")
            pixels[:] = 0

    def next_image(timeout):
        r._running = False
        return Image()

    save = spinnaker._save_mono_pgm

    def checked_save(*args):
        assert previews[0]["data"] == expected
        actions.append("save")
        if save_fails:
            raise TimeoutError("disk timeout")
        save(*args)

    monkeypatch.setattr(spinnaker, "_save_mono_pgm", checked_save)
    r._camera = types.SimpleNamespace(GetNextImage=next_image)
    r._running = True
    r._capture_loop()
    r._timestamps.close()
    assert actions == ["save", "release"]
    assert previews[0]["data"] == expected
    assert r._acquisition_timeouts == 0
    with r._timestamps.path.open() as handle:
        rows = list(csv.DictReader(handle))
    if save_fails:
        assert r._save_errors == 1
        assert "disk timeout" in r.error
        assert rows == []
    else:
        assert r.error is None
        assert r.frame_count == 1
        assert (r.session_dir / rows[0]["filename"]).read_bytes() == b"P5\n4 2\n255\n" + expected
        assert int(rows[0]["camera_timestamp_ns"]) == 123456789
        assert float(rows[0]["received_monotonic_s"]) == previews[0]["acquired_mono_s"]
        assert float(rows[0]["received_monotonic_s"]) <= float(rows[0]["monotonic_s"])


def test_pyspin_timeout_after_stop_is_normal_end_of_trigger_stream(tmp_path):
    r = HeadlessSpinnakerRecorder(tmp_path, "test", "test")

    def next_image(timeout):
        r._running = False
        raise RuntimeError("Spinnaker timeout")

    r._timestamps = object()
    r._camera = types.SimpleNamespace(GetNextImage=next_image)
    r._running = True
    r._capture_loop()
    assert r.error is None
    assert r._acquisition_timeouts == 1
