"""Headless Spinnaker recorder using PySpin (optional)."""

from __future__ import annotations

import csv
import ctypes
import json
import logging
import math
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .settings import apply_roi

logger = logging.getLogger(__name__)


PreviewCallback = Callable[[dict[str, Any]], None]


def _cleanup_step(errors: list[str], name: str, action: Callable[[], Any]) -> bool:
    try:
        action()
        return True
    except Exception as exc:
        errors.append(f"{name}: {type(exc).__name__}: {exc}")
        return False


def _remember_cleanup_errors(history: list[str], errors: list[str]) -> None:
    history.extend(error for error in errors if error not in history)


def _join_capture_thread(thread: threading.Thread | None, timeout_ms: int) -> None:
    # Thread.start() can fail after assignment. Joining an unstarted thread raises
    # another exception and used to prevent release of the camera and timestamp file.
    if thread is not None and thread.is_alive():
        thread.join(timeout=max(5.0, timeout_ms / 1000.0 + 2.0))
        if thread.is_alive():
            raise TimeoutError("capture thread did not stop before SDK teardown")


class _PreviewCadence:
    """Keep preview cadence despite small acquisition jitter; never catch up stale frames."""

    def __init__(self, interval_s: float):
        self.interval_s = max(0.0, float(interval_s))
        self.next_s: float | None = None

    def due(self, now: float) -> bool:
        if self.interval_s == 0:
            return True
        if self.next_s is None:
            self.next_s = now + self.interval_s
            return True
        if now < self.next_s:
            return False
        self.next_s += (math.floor((now - self.next_s) / self.interval_s) + 1) * self.interval_s
        return True


def _duration_s(start_wall_s: float | None, stop_wall_s: float | None) -> float | None:
    if start_wall_s is None or stop_wall_s is None:
        return None
    return max(0.0, float(stop_wall_s) - float(start_wall_s))


def _mean_fps(frame_count: int, duration_s: float | None) -> float | None:
    if duration_s is None or duration_s <= 0.0:
        return None
    return float(frame_count) / duration_s


def _capture_stats(
    *,
    frames_saved: int,
    incomplete_frames: int,
    acquisition_timeouts: int,
    save_errors: int = 0,
) -> dict[str, int]:
    # A timed-out wait can occur before triggers start or after they stop; it
    # does not identify a missing frame. Keep it separate from observed losses.
    skipped = max(0, int(incomplete_frames)) + max(0, int(save_errors))
    return {
        "frames_saved": max(0, int(frames_saved)),
        "incomplete_frames": max(0, int(incomplete_frames)),
        "acquisition_timeouts": max(0, int(acquisition_timeouts)),
        "save_errors": max(0, int(save_errors)),
        "frames_skipped": skipped,
        "frames_observed": max(0, int(frames_saved)) + skipped,
    }


def _downsample_array_preview(
    array: Any,
    *,
    max_width: int,
    max_height: int,
) -> tuple[int, int, int, bytes] | None:
    shape = getattr(array, "shape", None)
    if not shape or len(shape) < 2:
        return None
    height = int(shape[0])
    width = int(shape[1])
    if width <= 0 or height <= 0:
        return None
    step = max(
        1,
        int(math.ceil(width / float(max(1, max_width)))),
        int(math.ceil(height / float(max(1, max_height)))),
    )
    try:
        if len(shape) == 2:
            view = array[::step, ::step].copy()
            return int(view.shape[1]), int(view.shape[0]), 1, view.tobytes()
        channels = min(3, int(shape[2]))
        view = array[::step, ::step, :channels].copy()
        return int(view.shape[1]), int(view.shape[0]), channels, view.tobytes()
    except Exception:
        return None


def _downsample_mono_bytes(
    raw: bytes,
    *,
    width: int,
    height: int,
    stride: int,
    max_width: int,
    max_height: int,
) -> tuple[int, int, bytes] | None:
    if width <= 0 or height <= 0 or stride < width:
        return None
    step = max(
        1,
        int(math.ceil(width / float(max(1, max_width)))),
        int(math.ceil(height / float(max(1, max_height)))),
    )
    rows: list[bytes] = []
    for y in range(0, height, step):
        row = raw[y * stride : y * stride + width]
        rows.append(row[::step])
    if not rows:
        return None
    return len(rows[0]), len(rows), b"".join(rows)


def _downsample_pgm_file_preview(
    path: Path,
    *,
    max_width: int,
    max_height: int,
) -> tuple[int, int, bytes] | None:
    try:
        raw = path.read_bytes()
        width, height, pixel_offset = _parse_pgm_header(raw)
    except Exception:
        return None
    pixels = width * height
    if pixels <= 0 or len(raw) < pixel_offset + pixels:
        return None
    return _downsample_mono_bytes(
        raw[pixel_offset : pixel_offset + pixels],
        width=width,
        height=height,
        stride=width,
        max_width=max_width,
        max_height=max_height,
    )


def _parse_pgm_header(raw: bytes) -> tuple[int, int, int]:
    magic, pos = _next_pgm_token(raw, 0)
    if magic != b"P5":
        raise ValueError("unsupported PGM magic")
    width_token, pos = _next_pgm_token(raw, pos)
    height_token, pos = _next_pgm_token(raw, pos)
    max_value_token, pos = _next_pgm_token(raw, pos)
    width = int(width_token)
    height = int(height_token)
    max_value = int(max_value_token)
    if width <= 0 or height <= 0 or max_value <= 0 or max_value > 255:
        raise ValueError("unsupported PGM dimensions or bit depth")
    if pos >= len(raw) or raw[pos] not in b" \t\r\n":
        raise ValueError("PGM header missing raster separator")
    if raw[pos : pos + 2] == b"\r\n":
        return width, height, pos + 2
    return width, height, pos + 1


def _next_pgm_token(raw: bytes, pos: int) -> tuple[bytes, int]:
    length = len(raw)
    while pos < length:
        value = raw[pos]
        if value in b" \t\r\n":
            pos += 1
            continue
        if value == ord("#"):
            while pos < length and raw[pos] not in b"\r\n":
                pos += 1
            continue
        break
    start = pos
    while pos < length and raw[pos] not in b" \t\r\n":
        pos += 1
    if start == pos:
        raise ValueError("PGM token missing")
    return raw[start:pos], pos


def _save_mono_pgm(path: Path, width: int, height: int, stride: int, raw: bytes) -> None:
    """Save every Mono8 pixel, excluding row padding, without an SDK encoder."""
    if min(width, height) <= 0 or stride < width or len(raw) != stride * height:
        raise ValueError("Invalid Mono8 dimensions or buffer size")
    with path.open("wb") as handle:
        handle.write(f"P5\n{width} {height}\n255\n".encode("ascii"))
        if stride == width:
            handle.write(raw)
        else:
            for row in range(height):
                handle.write(raw[row * stride : row * stride + width])


class _PreviewDispatcher:
    """Drop-stale dispatcher so preview publishing cannot stall acquisition."""

    def __init__(self, callback: PreviewCallback | None, *, thread_name: str) -> None:
        self._callback = callback
        self._thread_name = thread_name
        self._condition = threading.Condition()
        self._latest: dict[str, Any] | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self.submitted_count = 0
        self.delivered_count = 0

    @property
    def enabled(self) -> bool:
        return self._callback is not None

    def start(self) -> None:
        if self._callback is None or self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run,
            name=self._thread_name,
            daemon=True,
        )
        self._thread.start()

    def submit(self, frame: dict[str, Any]) -> bool:
        if self._callback is None:
            return False
        with self._condition:
            if not self._running:
                return False
            self._latest = frame
            self.submitted_count += 1
            self._condition.notify()
        return True

    def stop(self) -> None:
        with self._condition:
            self._running = False
            self._condition.notify()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

    def _run(self) -> None:
        assert self._callback is not None
        while True:
            with self._condition:
                while self._running and self._latest is None:
                    self._condition.wait(timeout=0.25)
                if not self._running and self._latest is None:
                    return
                frame = self._latest
                self._latest = None
            if frame is None:
                continue
            try:
                self._callback(frame)
                self.delivered_count += 1
            except Exception:
                logger.debug("Could not publish Spinnaker preview frame", exc_info=True)


class _FrameTimestampWriter:
    def __init__(self, session_dir: Path) -> None:
        self.path = session_dir / "frame_timestamps.csv"
        self._handle = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._handle)
        try:
            self._writer.writerow(
                ["frame_index", "camera_timestamp_ns", "wall_time_s", "monotonic_s", "filename",
                 "received_wall_time_s", "received_monotonic_s"]
            )
        except BaseException as exc:
            errors: list[str] = []
            _cleanup_step(errors, "timestamp_header_cleanup", self._handle.close)
            if errors:
                exc.add_note("; ".join(errors))
            raise

    def write(
        self, frame_index: int, camera_timestamp_ns: int | None, filename: str,
        *, received_wall_s: float | None = None, received_mono_s: float | None = None,
    ) -> None:
        self._writer.writerow(
            [
                int(frame_index),
                "" if camera_timestamp_ns is None else int(camera_timestamp_ns),
                time.time(),
                time.perf_counter(),
                filename,
                "" if received_wall_s is None else received_wall_s,
                "" if received_mono_s is None else received_mono_s,
            ]
        )
        if frame_index % 30 == 0:
            self._handle.flush()

    def close(self) -> None:
        if not self._handle.closed:
            try:
                self._handle.flush()
            finally:
                self._handle.close()


class _SpinnakerCRecorder:
    """Spinnaker C API fallback for rigs without a matching PySpin wheel."""

    _TIMEOUT_ERROR = -1011

    def __init__(
        self,
        session_dir: Path,
        *,
        serial_number: str | None,
        fps: int,
        exposure_us: int,
        pixel_format: str,
        roi_width: int | None = None,
        roi_height: int | None = None,
        roi_offset_x: int | None = None,
        roi_offset_y: int | None = None,
        gain_db: float | None,
        gamma: float | None,
        acquisition_timeout_ms: int,
        preview_callback: PreviewCallback | None = None,
        preview_max_width: int = 520,
        preview_max_height: int = 390,
        preview_min_interval_s: float = 1 / 15,
    ) -> None:
        self.session_dir = session_dir
        self._serial_number = serial_number
        self._fps = int(fps)
        self._exposure_us = int(exposure_us)
        self._pixel_format = str(pixel_format)
        self._roi_width = None if roi_width is None else int(roi_width)
        self._roi_height = None if roi_height is None else int(roi_height)
        self._roi_offset_x = None if roi_offset_x is None else int(roi_offset_x)
        self._roi_offset_y = None if roi_offset_y is None else int(roi_offset_y)
        self._gain_db = gain_db
        self._gamma = gamma
        self._actual_roi: dict[str, int] | None = None
        self._settings_readback: dict[str, Any] = {}
        self._acquisition_timeout_ms = int(acquisition_timeout_ms)
        self._preview_callback = preview_callback
        self._preview_max_width = int(preview_max_width)
        self._preview_max_height = int(preview_max_height)
        self._preview_min_interval_s = float(preview_min_interval_s)
        self._preview_cadence = _PreviewCadence(preview_min_interval_s)
        self._last_preview_mono_s = 0.0
        self._preview_count = 0
        self._preview_dispatcher = _PreviewDispatcher(
            preview_callback,
            thread_name="spinnaker-c-preview",
        )
        self._running = False
        self._thread: threading.Thread | None = None
        self._error: str | None = None
        self._frame_count = 0
        self._incomplete_frames = 0
        self._acquisition_timeouts = 0
        self._save_errors = 0
        self._start_wall_s: float | None = None
        self._stop_wall_s: float | None = None
        self._dll_dir_handle: Any = None
        self._dll: Any = None
        self._fn: dict[str, Any] = {}
        self._system = ctypes.c_void_p()
        self._camera_list = ctypes.c_void_p()
        self._camera = ctypes.c_void_p()
        self._device_info: dict[str, str | None] = {
            "serial_number": None,
            "model_name": None,
            "vendor_name": None,
            "device_version": None,
            "firmware_version": None,
        }
        self._dll_path: str | None = None
        self._camera_initialized = False
        self._acquisition_started = False
        self._camera_list_cleared = False
        self._start_attempted = False
        self._cleanup_errors: list[str] = []
        self._unreleased_images: list[ctypes.c_void_p] = []
        self._timestamps: _FrameTimestampWriter | None = None

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def error(self) -> str | None:
        return self._error

    def health_reasons(self) -> list[str]:
        """Return why the recorder is unhealthy; an empty list means healthy."""
        reasons: list[str] = []
        if self._error:
            reasons.append(f"capture_error: {self._error}")
        if self._thread is not None and not self._thread.is_alive():
            reasons.append("capture_thread_not_alive")
        if self._save_errors:
            reasons.append(f"frame_save_errors: {self._save_errors}")
        return reasons

    def _roi_summary(self) -> dict[str, int | None]:
        return {
            "width": self._roi_width,
            "height": self._roi_height,
            "offset_x": self._roi_offset_x,
            "offset_y": self._roi_offset_y,
        }

    def start(self) -> Path:
        if self._start_attempted:
            raise RuntimeError("Camera recorder is single-use; create a new recorder for another recording")
        self._start_attempted = True
        try:
            self._load_api()
            self._check("spinSystemGetInstance", ctypes.byref(self._system))
            self._check("spinCameraListCreateEmpty", ctypes.byref(self._camera_list))
            self._check("spinSystemGetCameras", self._system, self._camera_list)
            self._camera = self._select_camera()
            device_version = self._camera_string(self._camera, "DeviceVersion")
            firmware_version = self._camera_string(
                self._camera,
                "DeviceFirmwareVersion",
            )
            self._device_info = {
                "serial_number": self._camera_string(self._camera, "DeviceSerialNumber"),
                "model_name": self._camera_string(self._camera, "DeviceModelName"),
                "vendor_name": self._camera_string(self._camera, "DeviceVendorName"),
                "device_version": device_version,
                # Blackfly S firmware is exposed as DeviceVersion on some
                # Spinnaker/transport-layer combinations.
                "firmware_version": firmware_version or device_version,
            }
            self._check("spinCameraInit", self._camera)
            self._camera_initialized = True
            self._configure_camera()
            self._check("spinCameraBeginAcquisition", self._camera)
            self._acquisition_started = True
            self._timestamps = _FrameTimestampWriter(self.session_dir)
            self._preview_dispatcher.start()
            self._running = True
            self._start_wall_s = time.time()
            self._thread = threading.Thread(
                target=self._capture_loop,
                name="spinnaker-c-capture",
                daemon=True,
            )
            self._thread.start()
            logger.info("Spinnaker C API recording started: %s", self.session_dir)
            return self.session_dir
        except BaseException as exc:
            self._error = f"{type(exc).__name__}: {exc}"
            self._running = False
            cleanup_errors: list[str] = []
            stopped = _cleanup_step(cleanup_errors, "capture_thread_stop", lambda: _join_capture_thread(
                self._thread, self._acquisition_timeout_ms))
            _cleanup_step(cleanup_errors, "preview_stop", self._preview_dispatcher.stop)
            if stopped:
                cleanup_errors.extend(self._cleanup())
                if self._timestamps is not None:
                    _cleanup_step(cleanup_errors, "timestamp_close", self._timestamps.close)
            _remember_cleanup_errors(self._cleanup_errors, cleanup_errors)
            if cleanup_errors:
                logger.error("Spinnaker C startup cleanup errors: %s", "; ".join(cleanup_errors))
                exc.add_note("Camera startup cleanup: " + "; ".join(cleanup_errors))
            raise

    def stop(self) -> dict[str, Any]:
        self._running = False
        try:
            _join_capture_thread(self._thread, self._acquisition_timeout_ms)
        except TimeoutError as exc:
            self._error = self._error or str(exc)
            cleanup_errors = [f"capture_thread_stop: {exc}"]
            _cleanup_step(cleanup_errors, "preview_stop", self._preview_dispatcher.stop)
            _remember_cleanup_errors(self._cleanup_errors, cleanup_errors)
            if len(cleanup_errors) > 1:
                exc.add_note("; ".join(cleanup_errors[1:]))
            raise
        cleanup_errors: list[str] = []
        _cleanup_step(cleanup_errors, "preview_stop", self._preview_dispatcher.stop)
        self._stop_wall_s = time.time()
        cleanup_errors.extend(self._cleanup())
        if self._timestamps is not None:
            if not _cleanup_step(cleanup_errors, "timestamp_close", self._timestamps.close):
                self._error = self._error or cleanup_errors[-1]
        _remember_cleanup_errors(self._cleanup_errors, cleanup_errors)

        summary = {
            "backend": "spinnaker_c",
            "device": dict(self._device_info),
            "spinnaker_c_dll": self._dll_path,
            "session_dir": str(self.session_dir),
            "frames_saved": int(self._frame_count),
            "capture_stats": _capture_stats(
                frames_saved=self._frame_count,
                incomplete_frames=self._incomplete_frames,
                acquisition_timeouts=self._acquisition_timeouts,
                save_errors=self._save_errors,
            ),
            "target_trigger_hz": int(self._fps),
            "exposure_us": int(self._exposure_us),
            "pixel_format": self._pixel_format,
            "roi": self._actual_roi,
            "requested_roi": self._roi_summary(),
            "settings_readback": dict(self._settings_readback),
            "exposure_control": "StreamDIO trigger pulse width; ExposureTime is the timed-mode setting",
            "preview_frames": int(self._preview_count),
            "preview_frames_delivered": int(self._preview_dispatcher.delivered_count),
            "frame_timestamps_csv": str(self.session_dir / "frame_timestamps.csv"),
            "start_wall_s": self._start_wall_s,
            "stop_wall_s": self._stop_wall_s,
            "duration_s": _duration_s(self._start_wall_s, self._stop_wall_s),
            "mean_saved_fps": _mean_fps(
                self._frame_count,
                _duration_s(self._start_wall_s, self._stop_wall_s),
            ),
            # Recorder lifetime includes waiting before/after triggers. The
            # session compares frames with the recorded hardware trigger count.
            "estimated_missing_triggers": None,
            "capture_error": self._error,
            "cleanup_error": "; ".join(cleanup_errors) or None,
            "cleanup_error_history": list(self._cleanup_errors),
        }
        (self.session_dir / "recording_summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        logger.info(
            "Spinnaker C API recording stopped (frames=%d, error=%s)",
            self._frame_count,
            self._error,
        )
        return summary

    def _load_api(self) -> None:
        candidates: list[Path] = []
        if configured := os.environ.get("SPINNAKER_C_DLL"):
            candidates.append(Path(configured))
        candidates.extend(
            [
                Path(r"C:\Program Files\Teledyne\Spinnaker\bin64\vs2015\SpinnakerC_v140.dll"),
                Path(
                    r"C:\Program Files\FLIR Systems\Spinnaker\bin64\vs2015"
                    r"\SpinnakerC_v140.dll"
                ),
            ]
        )
        dll_path = next((path for path in candidates if path.exists()), None)
        if dll_path is None:
            raise RuntimeError("Spinnaker C API DLL not found")
        self._dll_path = str(dll_path.resolve())
        if hasattr(os, "add_dll_directory"):
            self._dll_dir_handle = os.add_dll_directory(str(dll_path.parent))
        self._dll = ctypes.CDLL(str(dll_path))

        void_p = ctypes.c_void_p
        size_p = ctypes.POINTER(ctypes.c_size_t)
        self._bind("spinSystemGetInstance", [ctypes.POINTER(void_p)])
        self._bind("spinSystemReleaseInstance", [void_p])
        self._bind("spinCameraListCreateEmpty", [ctypes.POINTER(void_p)])
        self._bind("spinSystemGetCameras", [void_p, void_p])
        self._bind("spinCameraListGetSize", [void_p, size_p])
        self._bind("spinCameraListGet", [void_p, ctypes.c_size_t, ctypes.POINTER(void_p)])
        self._bind("spinCameraListClear", [void_p])
        self._bind("spinCameraListDestroy", [void_p])
        self._bind("spinCameraInit", [void_p])
        self._bind("spinCameraDeInit", [void_p])
        self._bind("spinCameraRelease", [void_p])
        self._bind("spinCameraBeginAcquisition", [void_p])
        self._bind("spinCameraEndAcquisition", [void_p])
        self._bind(
            "spinCameraGetNextImageEx",
            [void_p, ctypes.c_uint64, ctypes.POINTER(void_p)],
        )
        self._bind("spinImageIsIncomplete", [void_p, ctypes.POINTER(ctypes.c_ubyte)])
        self._bind("spinImageGetTimeStamp", [void_p, ctypes.POINTER(ctypes.c_uint64)])
        self._bind("spinImageSave", [void_p, ctypes.c_char_p, ctypes.c_int])
        self._bind("spinImageSaveFromExt", [void_p, ctypes.c_char_p])
        self._bind_optional("spinImageGetWidth", [void_p, size_p])
        self._bind_optional("spinImageGetHeight", [void_p, size_p])
        self._bind_optional("spinImageGetStride", [void_p, size_p])
        self._bind_optional("spinImageGetData", [void_p, ctypes.POINTER(void_p)])
        self._bind("spinImageRelease", [void_p])
        self._bind("spinCameraGetNodeMap", [void_p, ctypes.POINTER(void_p)])
        self._bind("spinCameraGetTLDeviceNodeMap", [void_p, ctypes.POINTER(void_p)])
        self._bind("spinNodeMapGetNode", [void_p, ctypes.c_char_p, ctypes.POINTER(void_p)])
        self._bind(
            "spinEnumerationGetEntryByName",
            [void_p, ctypes.c_char_p, ctypes.POINTER(void_p)],
        )
        self._bind("spinEnumerationEntryGetIntValue", [void_p, ctypes.POINTER(ctypes.c_int64)])
        self._bind("spinEnumerationSetIntValue", [void_p, ctypes.c_int64])
        self._bind("spinEnumerationGetCurrentEntry", [void_p, ctypes.POINTER(void_p)])
        self._bind("spinEnumerationEntryGetSymbolic", [void_p, ctypes.c_char_p, size_p])
        self._bind("spinFloatSetValue", [void_p, ctypes.c_double])
        self._bind("spinFloatGetValue", [void_p, ctypes.POINTER(ctypes.c_double)])
        self._bind("spinIntegerSetValue", [void_p, ctypes.c_int64])
        self._bind("spinIntegerGetValue", [void_p, ctypes.POINTER(ctypes.c_int64)])
        self._bind("spinBooleanSetValue", [void_p, ctypes.c_ubyte])
        self._bind("spinBooleanGetValue", [void_p, ctypes.POINTER(ctypes.c_ubyte)])
        self._bind(
            "spinStringGetValue",
            [void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_size_t)],
        )

    def _bind(self, name: str, argtypes: list[Any]) -> None:
        func = getattr(self._dll, name)
        func.argtypes = argtypes
        func.restype = ctypes.c_int
        self._fn[name] = func

    def _bind_optional(self, name: str, argtypes: list[Any]) -> bool:
        func = getattr(self._dll, name, None)
        if func is None:
            return False
        func.argtypes = argtypes
        func.restype = ctypes.c_int
        self._fn[name] = func
        return True

    def _check(self, name: str, *args: Any) -> None:
        result = int(self._fn[name](*args))
        if result != 0:
            raise RuntimeError(f"{name} failed with Spinnaker error {result}")

    def _select_camera(self) -> ctypes.c_void_p:
        count = ctypes.c_size_t()
        self._check("spinCameraListGetSize", self._camera_list, ctypes.byref(count))
        if count.value == 0:
            raise RuntimeError("No Spinnaker cameras found")
        if self._serial_number is None:
            self._check("spinCameraListGet", self._camera_list, 0, ctypes.byref(self._camera))
            return self._camera

        for index in range(count.value):
            self._camera = ctypes.c_void_p()
            self._check(
                "spinCameraListGet",
                self._camera_list,
                index,
                ctypes.byref(self._camera),
            )
            serial = self._camera_string(self._camera, "DeviceSerialNumber")
            if serial == self._serial_number:
                return self._camera
            self._check("spinCameraRelease", self._camera)
            self._camera = ctypes.c_void_p()
        raise RuntimeError(f"Spinnaker camera serial not found: {self._serial_number}")

    def _camera_string(self, camera: ctypes.c_void_p, node_name: str) -> str | None:
        node_map = ctypes.c_void_p()
        node = ctypes.c_void_p()
        if self._fn["spinCameraGetTLDeviceNodeMap"](camera, ctypes.byref(node_map)) != 0:
            return None
        if self._fn["spinNodeMapGetNode"](
            node_map,
            node_name.encode(),
            ctypes.byref(node),
        ) != 0:
            return None
        size = ctypes.c_size_t(256)
        buffer = ctypes.create_string_buffer(size.value)
        if self._fn["spinStringGetValue"](node, buffer, ctypes.byref(size)) != 0:
            return None
        return buffer.value.decode(errors="replace")

    def _configure_camera(self) -> None:
        node_map = ctypes.c_void_p()
        self._check("spinCameraGetNodeMap", self._camera, ctypes.byref(node_map))
        # Match the verified bringup: StreamDIO drives Line0 FrameStart TTLs and
        # the TTL pulse width controls camera exposure.
        # TriggerMode is unavailable while this camera is in TriggerWidth mode.
        self._set_enum(node_map, "ExposureMode", "Timed", required=True)
        self._set_enum(node_map, "TriggerSelector", "FrameStart", required=True)
        self._set_enum(node_map, "TriggerMode", "Off", required=True)
        self._set_enum(node_map, "AcquisitionMode", "Continuous", required=True)
        self._set_bool(node_map, "AcquisitionFrameRateEnable", False, required=True)
        self._set_enum(node_map, "ExposureAuto", "Off", required=True)
        self._set_float(node_map, "ExposureTime", self._exposure_us)
        self._apply_roi(node_map)
        self._set_enum(node_map, "PixelFormat", self._pixel_format, required=True)
        if self._gain_db is not None:
            self._set_enum(node_map, "GainAuto", "Off", required=True)
            self._set_float(node_map, "Gain", self._gain_db)
        if self._gamma is not None:
            self._set_bool(node_map, "GammaEnable", True, required=True)
            self._set_float(node_map, "Gamma", self._gamma)
        self._record_optional_settings(node_map)
        self._set_enum(node_map, "TriggerSelector", "FrameStart", required=True)
        self._set_enum(node_map, "TriggerSource", "Line0", required=True)
        self._set_enum(node_map, "TriggerActivation", "RisingEdge", required=True)
        self._set_float(node_map, "TriggerDelay", 9.0)
        self._set_enum(node_map, "TriggerMode", "On", required=True)
        self._set_enum(node_map, "ExposureMode", "TriggerWidth", required=True)

    def _record_optional_settings(self, node_map: ctypes.c_void_p) -> None:
        for name, kind in (("Gain", "Float"), ("Gamma", "Float"), ("GammaEnable", "Boolean")):
            try:
                self._settings_readback[name] = self._read_value(node_map, name, kind)
            except RuntimeError:
                self._settings_readback[name] = None
        try:
            self._settings_readback["GainAuto"] = self._read_enum(node_map, "GainAuto")
        except RuntimeError:
            self._settings_readback["GainAuto"] = None

    def _read_value(self, node_map: ctypes.c_void_p, name: str, kind: str) -> Any:
        node = self._get_node(node_map, name)
        if node is None:
            raise RuntimeError(f"Spinnaker {name}: node unavailable")
        value = {"Integer": ctypes.c_int64, "Float": ctypes.c_double, "Boolean": ctypes.c_ubyte}[kind]()
        self._check(f"spin{kind}GetValue", node, ctypes.byref(value))
        return bool(value.value) if kind == "Boolean" else value.value

    def _read_enum(self, node_map: ctypes.c_void_p, name: str) -> str:
        node = self._get_node(node_map, name)
        if node is None:
            raise RuntimeError(f"Spinnaker {name}: node unavailable")
        entry = ctypes.c_void_p()
        self._check("spinEnumerationGetCurrentEntry", node, ctypes.byref(entry))
        size = ctypes.c_size_t(256)
        buffer = ctypes.create_string_buffer(size.value)
        self._check("spinEnumerationEntryGetSymbolic", entry, buffer, ctypes.byref(size))
        return buffer.value.decode("ascii")

    def _get_node(self, node_map: ctypes.c_void_p, name: str) -> ctypes.c_void_p | None:
        node = ctypes.c_void_p()
        result = self._fn["spinNodeMapGetNode"](node_map, name.encode(), ctypes.byref(node))
        return node if result == 0 and node.value else None

    def _set_enum(
        self,
        node_map: ctypes.c_void_p,
        name: str,
        entry_name: str,
        *,
        required: bool = False,
    ) -> None:
        def failed(reason: str) -> None:
            if required:
                raise RuntimeError(f"Required Spinnaker enum {name}={entry_name}: {reason}")

        node = self._get_node(node_map, name)
        if node is None:
            failed("node unavailable")
            return
        entry = ctypes.c_void_p()
        if self._fn["spinEnumerationGetEntryByName"](
            node,
            entry_name.encode(),
            ctypes.byref(entry),
        ) != 0 or not entry.value:
            failed("entry unavailable")
            return
        value = ctypes.c_int64()
        if self._fn["spinEnumerationEntryGetIntValue"](entry, ctypes.byref(value)) != 0:
            failed("entry unreadable")
            return
        result = self._fn["spinEnumerationSetIntValue"](node, value)
        if result != 0:
            failed(f"set failed with Spinnaker error {result}")
            logger.debug("Could not set Spinnaker enum %s=%s (%d)", name, entry_name, result)
            return
        try:
            actual = self._read_enum(node_map, name)
            if actual != entry_name:
                raise RuntimeError(f"readback mismatch: got {actual}")
        except RuntimeError as exc:
            failed(str(exc))
            return
        self._settings_readback[name] = actual

    def _set_float(self, node_map: ctypes.c_void_p, name: str, value: float) -> None:
        node = self._get_node(node_map, name)
        if node is None:
            raise RuntimeError(f"Required Spinnaker float {name}={value}: node unavailable")
        self._check("spinFloatSetValue", node, float(value))
        # Camera floats can be quantized. Retain the actual accepted value.
        self._settings_readback[name] = self._read_value(node_map, name, "Float")

    def _set_int(self, node_map: ctypes.c_void_p, name: str, value: int) -> None:
        node = self._get_node(node_map, name)
        if node is None:
            raise RuntimeError(f"Required Spinnaker integer {name}={value}: node unavailable")
        self._check("spinIntegerSetValue", node, int(value))
        actual = self._read_value(node_map, name, "Integer")
        if actual != value:
            raise RuntimeError(f"Spinnaker {name} readback mismatch: requested {value}, got {actual}")

    def _apply_roi(self, node_map: ctypes.c_void_p) -> None:
        self._actual_roi = apply_roi(
            self._roi_summary(),
            lambda name: self._read_value(node_map, name, "Integer"),
            lambda name, value: self._set_int(node_map, name, value),
        )

    def _set_bool(
        self,
        node_map: ctypes.c_void_p,
        name: str,
        value: bool,
        *,
        required: bool = False,
    ) -> None:
        node = self._get_node(node_map, name)
        if node is None:
            if required:
                raise RuntimeError(f"Required Spinnaker bool {name}={value}: node unavailable")
            return
        result = self._fn["spinBooleanSetValue"](node, 1 if value else 0)
        if result != 0:
            if required:
                raise RuntimeError(
                    f"Required Spinnaker bool {name}={value}: "
                    f"set failed with Spinnaker error {result}"
                )
            logger.debug("Could not set Spinnaker bool %s=%s (%d)", name, value, result)
            return
        try:
            actual = self._read_value(node_map, name, "Boolean")
            if actual != value:
                raise RuntimeError(f"readback mismatch: got {actual}")
        except RuntimeError as exc:
            if required:
                raise RuntimeError(f"Required Spinnaker bool {name}={value}: {exc}") from exc
            return
        self._settings_readback[name] = actual

    def _capture_loop(self) -> None:
        assert self._timestamps is not None
        while self._running:
            image = ctypes.c_void_p()
            result = int(
                self._fn["spinCameraGetNextImageEx"](
                    self._camera,
                    self._acquisition_timeout_ms,
                    ctypes.byref(image),
                )
            )
            if result == self._TIMEOUT_ERROR:
                self._acquisition_timeouts += 1
                continue
            if result != 0:
                self._error = f"spinCameraGetNextImageEx failed with Spinnaker error {result}"
                logger.error("Spinnaker C capture loop error: %s", self._error)
                break
            acquired_mono_s = time.perf_counter()
            acquired_wall_s = time.time()
            try:
                incomplete = ctypes.c_ubyte()
                if self._fn["spinImageIsIncomplete"](image, ctypes.byref(incomplete)) != 0:
                    self._incomplete_frames += 1
                    continue
                if incomplete.value:
                    self._incomplete_frames += 1
                    continue
                frame_index = self._frame_count
                filename = f"frame_{frame_index:09d}.pgm"
                frame_path = self.session_dir / filename
                camera_timestamp = ctypes.c_uint64()
                timestamp_result = self._fn["spinImageGetTimeStamp"](
                    image,
                    ctypes.byref(camera_timestamp),
                )
                camera_timestamp_ns = camera_timestamp.value if timestamp_result == 0 else None
                try:
                    mono_frame = self._read_mono_image(image)
                    if mono_frame is not None:
                        self._maybe_emit_saved_preview(
                            frame_path, frame_index=frame_index,
                            camera_timestamp_ns=camera_timestamp_ns,
                            mono_frame=mono_frame, acquired_mono_s=acquired_mono_s,
                        )
                        _save_mono_pgm(frame_path, *mono_frame)
                    else:
                        self._check("spinImageSaveFromExt", image, os.fsencode(frame_path))
                        self._maybe_emit_saved_preview(
                            frame_path, frame_index=frame_index,
                            camera_timestamp_ns=camera_timestamp_ns,
                            acquired_mono_s=acquired_mono_s,
                        )
                except Exception:
                    self._save_errors += 1
                    raise
                self._timestamps.write(
                    frame_index, camera_timestamp_ns, filename,
                    received_wall_s=acquired_wall_s, received_mono_s=acquired_mono_s,
                )
                self._frame_count += 1
            except Exception as exc:
                self._error = f"{type(exc).__name__}: {exc}"
                logger.error("Spinnaker C capture loop error: %s", self._error)
                break
            finally:
                try:
                    self._check("spinImageRelease", image)
                except Exception as exc:
                    error = f"Image release failed: {type(exc).__name__}: {exc}"
                    self._error = self._error or error
                    self._unreleased_images.append(image)
                    _remember_cleanup_errors(self._cleanup_errors, [error])
                    self._running = False
                    logger.error("Spinnaker image release failed: %s", error)

    def _read_mono_image(self, image: ctypes.c_void_p) -> tuple[int, int, int, bytes] | None:
        """Copy Mono8 pixels before release; avoid the SDK's slow image encoder."""
        functions = ("spinImageGetWidth", "spinImageGetHeight", "spinImageGetStride", "spinImageGetData")
        if self._pixel_format != "Mono8" or not all(name in self._fn for name in functions):
            return None
        width, height, stride = ctypes.c_size_t(), ctypes.c_size_t(), ctypes.c_size_t()
        pointer = ctypes.c_void_p()
        for name, target in zip(functions, (width, height, stride, pointer)):
            self._check(name, image, ctypes.byref(target))
        if not pointer.value or width.value <= 0 or height.value <= 0 or stride.value < width.value:
            raise RuntimeError("Invalid Spinnaker Mono8 image buffer")
        return width.value, height.value, stride.value, ctypes.string_at(pointer, stride.value * height.value)

    def _maybe_emit_saved_preview(
        self,
        frame_path: Path,
        *,
        frame_index: int,
        camera_timestamp_ns: int | None,
        mono_frame: tuple[int, int, int, bytes] | None = None,
        acquired_mono_s: float | None = None,
    ) -> None:
        if not self._preview_dispatcher.enabled:
            return
        if not self._preview_cadence.due(time.perf_counter()):
            return
        try:
            if mono_frame is not None:
                width, height, stride, raw = mono_frame
                preview = _downsample_mono_bytes(
                    raw, width=width, height=height, stride=stride,
                    max_width=self._preview_max_width, max_height=self._preview_max_height,
                )
            else:
                preview = _downsample_pgm_file_preview(
                    frame_path, max_width=self._preview_max_width,
                    max_height=self._preview_max_height,
                )
            if preview is None:
                return
            out_width, out_height, data = preview
            self._last_preview_mono_s = time.perf_counter()
            if self._preview_dispatcher.submit(
                {
                    "frame_index": int(frame_index),
                    "camera_timestamp_ns": camera_timestamp_ns,
                    "width": out_width,
                    "height": out_height,
                    "channels": 1,
                    "format": "L",
                    "data": data,
                    "monotonic_s": self._last_preview_mono_s,
                    "acquired_mono_s": acquired_mono_s,
                    "backend": "spinnaker_c",
                }
            ):
                self._preview_count += 1
        except Exception:
            logger.debug("Could not emit Spinnaker C preview frame", exc_info=True)

    def _cleanup(self) -> list[str]:
        errors: list[str] = []

        def release(name: str, handle: ctypes.c_void_p) -> bool:
            return _cleanup_step(errors, name, lambda: self._check(name, handle))

        for image in list(self._unreleased_images):
            if release("spinImageRelease", image):
                self._unreleased_images.remove(image)
        if self._unreleased_images:
            _remember_cleanup_errors(self._cleanup_errors, errors)
            return errors
        # Retain failed handles for a subsequent stop attempt. Parent resources
        # cannot be released while their camera/list is still in use.
        if self._acquisition_started and release("spinCameraEndAcquisition", self._camera):
            self._acquisition_started = False
        if not self._acquisition_started:
            if self._camera_initialized and release("spinCameraDeInit", self._camera):
                self._camera_initialized = False
            if not self._camera_initialized and self._camera.value:
                if release("spinCameraRelease", self._camera):
                    self._camera = ctypes.c_void_p()
        if not self._camera.value and self._camera_list.value:
            if not self._camera_list_cleared:
                self._camera_list_cleared = release("spinCameraListClear", self._camera_list)
            if self._camera_list_cleared and release("spinCameraListDestroy", self._camera_list):
                self._camera_list = ctypes.c_void_p()
        if not self._camera.value and not self._camera_list.value and self._system.value:
            if release("spinSystemReleaseInstance", self._system):
                self._system = ctypes.c_void_p()
        if not self._system.value and self._dll_dir_handle is not None:
            if _cleanup_step(errors, "dll_directory_close", self._dll_dir_handle.close):
                self._dll_dir_handle = None
        _remember_cleanup_errors(self._cleanup_errors, errors)
        return errors


class HeadlessSpinnakerRecorder:
    """Record camera frames in headless mode as an image sequence."""

    def __init__(
        self,
        out_dir: Path,
        subject_id: str,
        phase: str,
        *,
        serial_number: str | None = None,
        fps: int = 30,
        exposure_us: int = 10_000,
        pixel_format: str = "Mono8",
        roi_width: int | None = None,
        roi_height: int | None = None,
        roi_offset_x: int | None = None,
        roi_offset_y: int | None = None,
        gain_db: float | None = None,
        gamma: float | None = None,
        acquisition_timeout_ms: int = 1000,
        preview_callback: PreviewCallback | None = None,
        preview_max_width: int = 520,
        preview_max_height: int = 390,
        preview_min_interval_s: float = 1 / 15,
    ) -> None:
        # Native camera SDK calls run on a capture thread.  Use an absolute path
        # so a process-wide working-directory change cannot redirect frame data.
        self._out_dir = Path(out_dir).expanduser().resolve()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.session_dir = self._out_dir / f"{subject_id}_{phase}_{stamp}_spinnaker"
        self._serial_number = serial_number
        self._fps = int(fps)
        self._exposure_us = int(exposure_us)
        self._pixel_format = str(pixel_format)
        self._roi_width = None if roi_width is None else int(roi_width)
        self._roi_height = None if roi_height is None else int(roi_height)
        self._roi_offset_x = None if roi_offset_x is None else int(roi_offset_x)
        self._roi_offset_y = None if roi_offset_y is None else int(roi_offset_y)
        self._gain_db = gain_db
        self._gamma = gamma
        self._acquisition_timeout_ms = int(acquisition_timeout_ms)
        self._preview_callback = preview_callback
        self._preview_max_width = int(preview_max_width)
        self._preview_max_height = int(preview_max_height)
        self._preview_min_interval_s = float(preview_min_interval_s)
        self._preview_cadence = _PreviewCadence(preview_min_interval_s)
        self._last_preview_mono_s = 0.0
        self._preview_count = 0
        self._preview_dispatcher = _PreviewDispatcher(
            preview_callback,
            thread_name="spinnaker-preview",
        )
        self.backend = "spinnaker_pyspin"
        self._actual_roi: dict[str, int] | None = None
        self._settings_readback: dict[str, Any] = {}

        self._running = False
        self._thread: threading.Thread | None = None
        self._error: str | None = None
        self._frame_count = 0
        self._incomplete_frames = 0
        self._acquisition_timeouts = 0
        self._save_errors = 0
        self._start_wall_s: float | None = None
        self._stop_wall_s: float | None = None

        self._pyspin: Any = None
        self._system: Any = None
        self._camera: Any = None
        self._camera_list: Any = None
        self._camera_initialized = False
        self._acquisition_started = False
        self._start_attempted = False
        self._cleanup_errors: list[str] = []
        self._unreleased_images: list[Any] = []
        self._device_info: dict[str, str | None] = {
            "serial_number": None,
            "model_name": None,
            "vendor_name": None,
            "device_version": None,
            "firmware_version": None,
        }
        self._timestamps: _FrameTimestampWriter | None = None
        self._c_backend: _SpinnakerCRecorder | None = None

    @property
    def frame_count(self) -> int:
        if self._c_backend is not None:
            return self._c_backend.frame_count
        return self._frame_count

    @property
    def error(self) -> str | None:
        if self._c_backend is not None:
            return self._c_backend.error
        return self._error

    def health_reasons(self) -> list[str]:
        """Return why the recorder is unhealthy; an empty list means healthy."""
        if self._c_backend is not None:
            return self._c_backend.health_reasons()
        reasons: list[str] = []
        if self._error:
            reasons.append(f"capture_error: {self._error}")
        if self._thread is not None and not self._thread.is_alive():
            reasons.append("capture_thread_not_alive")
        if self._save_errors:
            reasons.append(f"frame_save_errors: {self._save_errors}")
        return reasons

    @property
    def device_info(self) -> dict[str, str | None]:
        """Identity captured from the connected camera before acquisition starts."""

        if self._c_backend is not None:
            return dict(self._c_backend._device_info)
        return dict(self._device_info)

    @property
    def camera_settings(self) -> dict[str, Any]:
        """Settings read back before acquisition, also available in crash journals."""
        source = self._c_backend if self._c_backend is not None else self
        return {
            "roi": None if source._actual_roi is None else dict(source._actual_roi),
            "requested_roi": source._roi_summary(),
            "settings_readback": dict(source._settings_readback),
            "exposure_control": "StreamDIO trigger pulse width; ExposureTime is the timed-mode setting",
        }

    def _roi_summary(self) -> dict[str, int | None]:
        return {
            "width": self._roi_width,
            "height": self._roi_height,
            "offset_x": self._roi_offset_x,
            "offset_y": self._roi_offset_y,
        }

    def start(self) -> Path:
        """Start camera acquisition and background frame saving."""
        if self._start_attempted:
            raise RuntimeError("Camera recorder is single-use; create a new recorder for another recording")
        self._start_attempted = True
        self.session_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._pyspin = self._load_pyspin()
        except RuntimeError as exc:
            logger.warning("%s Falling back to installed Spinnaker C API.", exc)
            self.backend = "spinnaker_c"
            self._c_backend = _SpinnakerCRecorder(
                self.session_dir,
                serial_number=self._serial_number,
                fps=self._fps,
                exposure_us=self._exposure_us,
                pixel_format=self._pixel_format,
                roi_width=self._roi_width,
                roi_height=self._roi_height,
                roi_offset_x=self._roi_offset_x,
                roi_offset_y=self._roi_offset_y,
                gain_db=self._gain_db,
                gamma=self._gamma,
                acquisition_timeout_ms=self._acquisition_timeout_ms,
                preview_callback=self._preview_callback,
                preview_max_width=self._preview_max_width,
                preview_max_height=self._preview_max_height,
                preview_min_interval_s=self._preview_min_interval_s,
            )
            return self._c_backend.start()

        try:
            self._system = self._pyspin.System.GetInstance()
            self._camera_list = self._system.GetCameras()
            if self._camera_list.GetSize() == 0:
                raise RuntimeError("No Spinnaker cameras found")

            self._camera = self._select_camera(self._camera_list)
            device_version = self._pyspin_device_value("DeviceVersion")
            firmware_version = self._pyspin_device_value("DeviceFirmwareVersion")
            self._device_info = {
                "serial_number": self._pyspin_device_value("DeviceSerialNumber"),
                "model_name": self._pyspin_device_value("DeviceModelName"),
                "vendor_name": self._pyspin_device_value("DeviceVendorName"),
                "device_version": device_version,
                "firmware_version": firmware_version or device_version,
            }
            self._camera_list.Clear()
            self._camera_list = None
            self._camera.Init()
            self._camera_initialized = True
            self._configure_camera()
            self._camera.BeginAcquisition()
            self._acquisition_started = True
            self._timestamps = _FrameTimestampWriter(self.session_dir)
            self._preview_dispatcher.start()
            self._running = True
            self._start_wall_s = time.time()
            self._thread = threading.Thread(
                target=self._capture_loop,
                name="spinnaker-capture",
                daemon=True,
            )
            self._thread.start()
            logger.info("Headless Spinnaker recording started: %s", self.session_dir)
            return self.session_dir
        except BaseException as exc:
            self._error = f"{type(exc).__name__}: {exc}"
            self._running = False
            cleanup_errors: list[str] = []
            stopped = _cleanup_step(cleanup_errors, "capture_thread_stop", lambda: _join_capture_thread(
                self._thread, self._acquisition_timeout_ms))
            _cleanup_step(cleanup_errors, "preview_stop", self._preview_dispatcher.stop)
            if stopped:
                cleanup_error = self._cleanup_pyspin()
                if cleanup_error:
                    cleanup_errors.append(cleanup_error)
                if self._timestamps is not None:
                    _cleanup_step(cleanup_errors, "timestamp_close", self._timestamps.close)
            _remember_cleanup_errors(self._cleanup_errors, cleanup_errors)
            if cleanup_errors:
                logger.error("PySpin startup cleanup errors: %s", "; ".join(cleanup_errors))
                exc.add_note("Camera startup cleanup: " + "; ".join(cleanup_errors))
            raise

    def stop(self) -> dict[str, Any]:
        """Stop acquisition and return recording summary."""
        if self._c_backend is not None:
            return self._c_backend.stop()

        self._running = False
        try:
            _join_capture_thread(self._thread, self._acquisition_timeout_ms)
        except TimeoutError as exc:
            self._error = self._error or str(exc)
            cleanup_errors = [f"capture_thread_stop: {exc}"]
            _cleanup_step(cleanup_errors, "preview_stop", self._preview_dispatcher.stop)
            _remember_cleanup_errors(self._cleanup_errors, cleanup_errors)
            if len(cleanup_errors) > 1:
                exc.add_note("; ".join(cleanup_errors[1:]))
            raise
        cleanup_errors: list[str] = []
        _cleanup_step(cleanup_errors, "preview_stop", self._preview_dispatcher.stop)
        self._stop_wall_s = time.time()
        cleanup_error = self._cleanup_pyspin()
        if cleanup_error:
            cleanup_errors.append(cleanup_error)
        if self._timestamps is not None:
            if not _cleanup_step(cleanup_errors, "timestamp_close", self._timestamps.close):
                self._error = self._error or cleanup_errors[-1]
        _remember_cleanup_errors(self._cleanup_errors, cleanup_errors)

        summary = {
            "backend": self.backend,
            "device": dict(self._device_info),
            "session_dir": str(self.session_dir),
            "frames_saved": int(self._frame_count),
            "capture_stats": _capture_stats(
                frames_saved=self._frame_count,
                incomplete_frames=self._incomplete_frames,
                acquisition_timeouts=self._acquisition_timeouts,
                save_errors=self._save_errors,
            ),
            "target_trigger_hz": int(self._fps),
            "exposure_us": int(self._exposure_us),
            "pixel_format": self._pixel_format,
            "roi": self._actual_roi,
            "requested_roi": self._roi_summary(),
            "settings_readback": dict(self._settings_readback),
            "exposure_control": "StreamDIO trigger pulse width; ExposureTime is the timed-mode setting",
            "preview_frames": int(self._preview_count),
            "preview_frames_delivered": int(self._preview_dispatcher.delivered_count),
            "frame_timestamps_csv": str(self.session_dir / "frame_timestamps.csv"),
            "start_wall_s": self._start_wall_s,
            "stop_wall_s": self._stop_wall_s,
            "duration_s": _duration_s(self._start_wall_s, self._stop_wall_s),
            "mean_saved_fps": _mean_fps(
                self._frame_count,
                _duration_s(self._start_wall_s, self._stop_wall_s),
            ),
            "estimated_missing_triggers": None,
            "capture_error": self._error,
            "cleanup_error": "; ".join(cleanup_errors) or None,
            "cleanup_error_history": list(self._cleanup_errors),
        }
        (self.session_dir / "recording_summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        logger.info(
            "Headless Spinnaker recording stopped (frames=%d, error=%s)",
            self._frame_count,
            self._error,
        )
        return summary

    def _cleanup_pyspin(self) -> str | None:
        cleanup_errors: list[str] = []
        for image in list(self._unreleased_images):
            if _cleanup_step(cleanup_errors, "Image.Release", image.Release):
                self._unreleased_images.remove(image)
        if self._unreleased_images:
            _remember_cleanup_errors(self._cleanup_errors, cleanup_errors)
            return "; ".join(cleanup_errors)
        if self._camera is not None:
            if self._acquisition_started and _cleanup_step(
                cleanup_errors, "EndAcquisition", self._camera.EndAcquisition,
            ):
                self._acquisition_started = False
            if not self._acquisition_started:
                if self._camera_initialized and _cleanup_step(
                    cleanup_errors, "DeInit", self._camera.DeInit,
                ):
                    self._camera_initialized = False
                if not self._camera_initialized:
                    self._camera = None
        if self._camera is None and self._camera_list is not None:
            if _cleanup_step(cleanup_errors, "CameraList.Clear", self._camera_list.Clear):
                self._camera_list = None
        if self._camera is None and self._camera_list is None and self._system is not None:
            if _cleanup_step(cleanup_errors, "ReleaseInstance", self._system.ReleaseInstance):
                self._system = None
        _remember_cleanup_errors(self._cleanup_errors, cleanup_errors)
        return "; ".join(cleanup_errors) or None

    def _load_pyspin(self) -> Any:
        try:
            import PySpin  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - import availability
            raise RuntimeError(
                "PySpin not available. Install FLIR Spinnaker SDK Python bindings to use "
                "camera.backend: spinnaker."
            ) from exc
        return PySpin

    def _select_camera(self, cam_list: Any) -> Any:
        if self._serial_number is None:
            return cam_list.GetByIndex(0)
        for idx in range(cam_list.GetSize()):
            cam = cam_list.GetByIndex(idx)
            try:
                if str(cam.TLDevice.DeviceSerialNumber.GetValue()) == self._serial_number:
                    return cam
            finally:
                # A retained exception traceback must not keep an unmatched
                # camera reference alive when the parent SDK is released.
                del cam
        raise RuntimeError(f"Spinnaker camera serial not found: {self._serial_number}")

    def _pyspin_device_value(self, name: str) -> str | None:
        try:
            value = getattr(self._camera.TLDevice, name).GetValue()
            return str(value) if value is not None else None
        except Exception:
            return None

    def _configure_camera(self) -> None:
        nodemap = self._camera.GetNodeMap()
        self._set_pyspin_enum(nodemap, "ExposureMode", "Timed", required=True)
        self._set_pyspin_enum(nodemap, "TriggerSelector", "FrameStart", required=True)
        self._set_pyspin_enum(nodemap, "TriggerMode", "Off", required=True)
        self._set_pyspin_enum(nodemap, "AcquisitionMode", "Continuous", required=True)
        self._set_pyspin_value(nodemap, "AcquisitionFrameRateEnable", False, "Boolean")
        self._set_pyspin_enum(nodemap, "ExposureAuto", "Off", required=True)
        self._set_pyspin_value(nodemap, "ExposureTime", self._exposure_us, "Float")
        self._apply_pyspin_roi(nodemap)
        self._set_pyspin_enum(nodemap, "PixelFormat", self._pixel_format, required=True)
        if self._gain_db is not None:
            self._set_pyspin_enum(nodemap, "GainAuto", "Off", required=True)
            self._set_pyspin_value(nodemap, "Gain", self._gain_db, "Float")
        if self._gamma is not None:
            self._set_pyspin_value(nodemap, "GammaEnable", True, "Boolean")
            self._set_pyspin_value(nodemap, "Gamma", self._gamma, "Float")
        for name, kind in (("Gain", "Float"), ("Gamma", "Float"),
                           ("GammaEnable", "Boolean"), ("GainAuto", "Enumeration")):
            try:
                self._settings_readback[name] = self._read_pyspin_value(nodemap, name, kind)
            except Exception:
                self._settings_readback[name] = None
        self._set_pyspin_enum(nodemap, "TriggerSelector", "FrameStart", required=True)
        self._set_pyspin_enum(nodemap, "TriggerSource", "Line0", required=True)
        self._set_pyspin_enum(nodemap, "TriggerActivation", "RisingEdge", required=True)
        self._set_pyspin_value(nodemap, "TriggerDelay", 9.0, "Float")
        self._set_pyspin_enum(nodemap, "TriggerMode", "On", required=True)
        self._set_pyspin_enum(nodemap, "ExposureMode", "TriggerWidth", required=True)

    def _read_pyspin_value(self, nodemap: Any, name: str, kind: str) -> Any:
        node = getattr(self._pyspin, f"C{kind}Ptr")(nodemap.GetNode(name))
        if not self._pyspin.IsAvailable(node) or not self._pyspin.IsReadable(node):
            raise RuntimeError(f"PySpin {name}: node unavailable or unreadable")
        if kind == "Enumeration":
            return str(node.GetCurrentEntry().GetSymbolic())
        return node.GetValue()

    def _set_pyspin_value(self, nodemap: Any, name: str, value: Any, kind: str) -> None:
        node = getattr(self._pyspin, f"C{kind}Ptr")(nodemap.GetNode(name))
        if not self._pyspin.IsAvailable(node) or not self._pyspin.IsWritable(node):
            raise RuntimeError(f"Required PySpin {name}={value}: node unavailable or unwritable")
        convert = {"Integer": int, "Float": float, "Boolean": bool}[kind]
        node.SetValue(convert(value))
        actual = self._read_pyspin_value(nodemap, name, kind)
        if kind != "Float" and actual != value:
            raise RuntimeError(f"PySpin {name} readback mismatch: requested {value}, got {actual}")
        self._settings_readback[name] = actual

    def _set_pyspin_enum(
        self,
        nodemap: Any,
        node_name: str,
        entry_name: str,
        *,
        required: bool = False,
    ) -> None:
        PySpin = self._pyspin
        try:
            node = PySpin.CEnumerationPtr(nodemap.GetNode(node_name))
            if not PySpin.IsAvailable(node) or not PySpin.IsWritable(node):
                raise RuntimeError("node unavailable or unwritable")
            entry = node.GetEntryByName(entry_name)
            if not PySpin.IsAvailable(entry) or not PySpin.IsReadable(entry):
                raise RuntimeError("entry unavailable or unreadable")
            node.SetIntValue(entry.GetValue())
            actual = self._read_pyspin_value(nodemap, node_name, "Enumeration")
            if actual != entry_name:
                raise RuntimeError(f"readback mismatch: got {actual}")
            self._settings_readback[node_name] = actual
        except Exception as exc:
            if required:
                raise RuntimeError(
                    f"Required PySpin enum {node_name}={entry_name}: {exc}"
                ) from exc
            return

    def _set_pyspin_int(self, nodemap: Any, node_name: str, value: int) -> None:
        self._set_pyspin_value(nodemap, node_name, value, "Integer")

    def _apply_pyspin_roi(self, nodemap: Any) -> None:
        self._actual_roi = apply_roi(
            self._roi_summary(),
            lambda name: self._read_pyspin_value(nodemap, name, "Integer"),
            lambda name, value: self._set_pyspin_int(nodemap, name, value),
        )

    def _capture_loop(self) -> None:
        assert self._camera is not None
        assert self._timestamps is not None
        while self._running:
            image = None
            try:
                image = self._camera.GetNextImage(int(self._acquisition_timeout_ms))
                acquired_mono_s = time.perf_counter()
                acquired_wall_s = time.time()
                if image.IsIncomplete():
                    self._incomplete_frames += 1
                    continue
                frame_index = self._frame_count
                # Mono8 uses the same direct, lossless writer as the C backend.
                filename = f"frame_{frame_index:09d}.{'pgm' if self._pixel_format == 'Mono8' else 'png'}"
                frame_path = self.session_dir / filename
                camera_timestamp = None
                try:
                    camera_timestamp = int(image.GetTimeStamp())
                except Exception:
                    pass
                try:
                    array = None
                    if self._pixel_format == "Mono8":
                        array = image.GetNDArray()
                        if len(array.shape) != 2 or str(array.dtype) != "uint8":
                            raise ValueError("Invalid PySpin Mono8 array")
                        array = array.copy()
                    self._maybe_emit_preview(
                        image, frame_index=frame_index, camera_timestamp_ns=camera_timestamp,
                        array=array, acquired_mono_s=acquired_mono_s,
                    )
                    if array is None:
                        image.Save(str(frame_path))
                    else:
                        height, width = array.shape
                        _save_mono_pgm(frame_path, width, height, width, array.tobytes())
                except Exception:
                    self._save_errors += 1
                    raise
                self._timestamps.write(
                    frame_index, camera_timestamp, filename,
                    received_wall_s=acquired_wall_s, received_mono_s=acquired_mono_s,
                )
                self._frame_count += 1
            except Exception as exc:  # pragma: no cover - device-specific
                # A disk/encoder timeout is a save failure, not an empty camera wait.
                if image is None and "timeout" in str(exc).lower():
                    self._acquisition_timeouts += 1
                    continue
                self._error = f"{type(exc).__name__}: {exc}"
                logger.error("Spinnaker capture loop error: %s", self._error)
                break
            finally:
                if image is not None:
                    try:
                        image.Release()
                    except Exception as exc:
                        error = f"Image release failed: {type(exc).__name__}: {exc}"
                        self._error = self._error or error
                        self._unreleased_images.append(image)
                        _remember_cleanup_errors(self._cleanup_errors, [error])
                        self._running = False
                        logger.error("PySpin image release failed: %s", error)

    def _maybe_emit_preview(
        self,
        image: Any,
        *,
        frame_index: int,
        camera_timestamp_ns: int | None,
        array: Any = None,
        acquired_mono_s: float | None = None,
    ) -> None:
        if not self._preview_dispatcher.enabled:
            return
        if not self._preview_cadence.due(time.perf_counter()):
            return
        try:
            if array is None:
                array = image.GetNDArray()
            preview = _downsample_array_preview(
                array,
                max_width=self._preview_max_width,
                max_height=self._preview_max_height,
            )
            if preview is None:
                return
            width, height, channels, data = preview
            self._last_preview_mono_s = time.perf_counter()
            if self._preview_dispatcher.submit(
                {
                    "frame_index": int(frame_index),
                    "camera_timestamp_ns": camera_timestamp_ns,
                    "width": width,
                    "height": height,
                    "channels": channels,
                    "format": "L" if channels == 1 else "RGB",
                    "data": data,
                    "monotonic_s": self._last_preview_mono_s,
                    "acquired_mono_s": acquired_mono_s,
                    "backend": "spinnaker_pyspin",
                }
            ):
                self._preview_count += 1
        except Exception:
            logger.debug("Could not emit PySpin preview frame", exc_info=True)
