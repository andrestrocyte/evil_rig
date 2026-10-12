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

logger = logging.getLogger(__name__)


PreviewCallback = Callable[[dict[str, Any]], None]


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
        self._writer.writerow(
            ["frame_index", "camera_timestamp_ns", "wall_time_s", "monotonic_s", "filename"]
        )

    def write(self, frame_index: int, camera_timestamp_ns: int | None, filename: str) -> None:
        self._writer.writerow(
            [
                int(frame_index),
                "" if camera_timestamp_ns is None else int(camera_timestamp_ns),
                time.time(),
                time.perf_counter(),
                filename,
            ]
        )
        if frame_index % 30 == 0:
            self._handle.flush()

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.flush()
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
        except Exception:
            self._running = False
            self._preview_dispatcher.stop()
            if self._timestamps is not None:
                self._timestamps.close()
            cleanup_errors = self._cleanup()
            if cleanup_errors:
                logger.error("Spinnaker C startup cleanup errors: %s", "; ".join(cleanup_errors))
            raise

    def stop(self) -> dict[str, Any]:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, self._acquisition_timeout_ms / 1000.0 + 2.0))
            if self._thread.is_alive():
                self._error = "capture thread did not stop before SDK teardown"
                self._preview_dispatcher.stop()
                raise TimeoutError(self._error)
        self._preview_dispatcher.stop()
        self._stop_wall_s = time.time()
        cleanup_errors = self._cleanup()
        if self._timestamps is not None:
            self._timestamps.close()

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
            "roi": self._roi_summary(),
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
        self._bind("spinFloatSetValue", [void_p, ctypes.c_double])
        self._bind_optional("spinIntegerSetValue", [void_p, ctypes.c_int64])
        self._bind("spinBooleanSetValue", [void_p, ctypes.c_ubyte])
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
            camera = ctypes.c_void_p()
            self._check("spinCameraListGet", self._camera_list, 0, ctypes.byref(camera))
            return camera

        for index in range(count.value):
            camera = ctypes.c_void_p()
            self._check(
                "spinCameraListGet",
                self._camera_list,
                index,
                ctypes.byref(camera),
            )
            serial = self._camera_string(camera, "DeviceSerialNumber")
            if serial == self._serial_number:
                return camera
            self._fn["spinCameraRelease"](camera)
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
            self._set_enum(node_map, "GainAuto", "Off")
            self._set_float(node_map, "Gain", self._gain_db)
        if self._gamma is not None:
            self._set_bool(node_map, "GammaEnable", True)
            self._set_float(node_map, "Gamma", self._gamma)
        self._set_enum(node_map, "TriggerSelector", "FrameStart", required=True)
        self._set_enum(node_map, "TriggerSource", "Line0", required=True)
        self._set_enum(node_map, "TriggerActivation", "RisingEdge", required=True)
        self._set_float(node_map, "TriggerDelay", 9.0)
        self._set_enum(node_map, "TriggerMode", "On", required=True)
        self._set_enum(node_map, "ExposureMode", "TriggerWidth", required=True)

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

    def _set_float(self, node_map: ctypes.c_void_p, name: str, value: float) -> None:
        node = self._get_node(node_map, name)
        if node is None:
            return
        result = self._fn["spinFloatSetValue"](node, float(value))
        if result != 0:
            logger.debug("Could not set Spinnaker float %s=%s (%d)", name, value, result)

    def _set_int(self, node_map: ctypes.c_void_p, name: str, value: int) -> None:
        if "spinIntegerSetValue" not in self._fn:
            return
        node = self._get_node(node_map, name)
        if node is None:
            return
        result = self._fn["spinIntegerSetValue"](node, int(value))
        if result != 0:
            logger.debug("Could not set Spinnaker integer %s=%s (%d)", name, value, result)

    def _apply_roi(self, node_map: ctypes.c_void_p) -> None:
        for node_name, value in (
            ("OffsetX", self._roi_offset_x),
            ("OffsetY", self._roi_offset_y),
            ("Width", self._roi_width),
            ("Height", self._roi_height),
        ):
            if value is not None:
                self._set_int(node_map, node_name, max(0, int(value)))

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
                acquired_mono_s = time.perf_counter()
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
                        width, height, stride, raw = mono_frame
                        with frame_path.open("wb") as handle:
                            handle.write(f"P5\n{width} {height}\n255\n".encode("ascii"))
                            if stride == width:
                                handle.write(raw)
                            else:
                                for row in range(height):
                                    handle.write(raw[row * stride : row * stride + width])
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
                self._timestamps.write(frame_index, camera_timestamp_ns, filename)
                self._frame_count += 1
            except Exception as exc:
                self._error = f"{type(exc).__name__}: {exc}"
                logger.error("Spinnaker C capture loop error: %s", self._error)
                break
            finally:
                self._fn["spinImageRelease"](image)

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
        steps = [
            ("spinCameraEndAcquisition", self._acquisition_started, self._camera),
            ("spinCameraDeInit", self._camera_initialized, self._camera),
            ("spinCameraRelease", bool(self._camera.value), self._camera),
            ("spinCameraListClear", bool(self._camera_list.value), self._camera_list),
            ("spinCameraListDestroy", bool(self._camera_list.value), self._camera_list),
            ("spinSystemReleaseInstance", bool(self._system.value), self._system),
        ]
        for name, enabled, handle in steps:
            if not enabled:
                continue
            try:
                result = int(self._fn[name](handle))
                if result != 0:
                    errors.append(f"{name}: {result}")
            except Exception as exc:
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
        self._acquisition_started = False
        self._camera_initialized = False
        self._camera = ctypes.c_void_p()
        self._camera_list = ctypes.c_void_p()
        self._system = ctypes.c_void_p()
        if self._dll_dir_handle is not None:
            self._dll_dir_handle.close()
            self._dll_dir_handle = None
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

    def _roi_summary(self) -> dict[str, int | None]:
        return {
            "width": self._roi_width,
            "height": self._roi_height,
            "offset_x": self._roi_offset_x,
            "offset_y": self._roi_offset_y,
        }

    def start(self) -> Path:
        """Start camera acquisition and background frame saving."""
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
            cam_list = self._system.GetCameras()
            if cam_list.GetSize() == 0:
                cam_list.Clear()
                raise RuntimeError("No Spinnaker cameras found")

            self._camera = self._select_camera(cam_list)
            device_version = self._pyspin_device_value("DeviceVersion")
            firmware_version = self._pyspin_device_value("DeviceFirmwareVersion")
            self._device_info = {
                "serial_number": self._pyspin_device_value("DeviceSerialNumber"),
                "model_name": self._pyspin_device_value("DeviceModelName"),
                "vendor_name": self._pyspin_device_value("DeviceVendorName"),
                "device_version": device_version,
                "firmware_version": firmware_version or device_version,
            }
            cam_list.Clear()
            self._camera.Init()
            self._configure_camera()
            self._camera.BeginAcquisition()
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
        except Exception:
            self._cleanup_pyspin()
            raise

    def stop(self) -> dict[str, Any]:
        """Stop acquisition and return recording summary."""
        if self._c_backend is not None:
            return self._c_backend.stop()

        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, self._acquisition_timeout_ms / 1000.0 + 2.0))
            if self._thread.is_alive():
                self._error = "capture thread did not stop before SDK teardown"
                self._preview_dispatcher.stop()
                raise TimeoutError(self._error)
        self._preview_dispatcher.stop()
        self._stop_wall_s = time.time()
        cleanup_error = self._cleanup_pyspin()
        if self._timestamps is not None:
            self._timestamps.close()

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
            "roi": self._roi_summary(),
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
            "cleanup_error": cleanup_error,
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
        try:
            if self._camera is not None:
                self._camera.EndAcquisition()
        except Exception as exc:  # pragma: no cover - device-specific
            cleanup_errors.append(f"EndAcquisition: {type(exc).__name__}: {exc}")
        try:
            if self._camera is not None:
                self._camera.DeInit()
        except Exception as exc:  # pragma: no cover - device-specific
            cleanup_errors.append(f"DeInit: {type(exc).__name__}: {exc}")
        finally:
            self._camera = None
        try:
            if self._system is not None:
                self._system.ReleaseInstance()
        except Exception as exc:  # pragma: no cover - device-specific
            cleanup_errors.append(f"ReleaseInstance: {type(exc).__name__}: {exc}")
        finally:
            self._system = None
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
            if str(cam.TLDevice.DeviceSerialNumber.GetValue()) == self._serial_number:
                return cam
        raise RuntimeError(f"Spinnaker camera serial not found: {self._serial_number}")

    def _pyspin_device_value(self, name: str) -> str | None:
        try:
            value = getattr(self._camera.TLDevice, name).GetValue()
            return str(value) if value is not None else None
        except Exception:
            return None

    def _configure_camera(self) -> None:
        PySpin = self._pyspin
        nodemap = self._camera.GetNodeMap()

        self._set_pyspin_enum(nodemap, "ExposureMode", "Timed", required=True)
        self._set_pyspin_enum(nodemap, "TriggerSelector", "FrameStart", required=True)
        self._set_pyspin_enum(nodemap, "TriggerMode", "Off", required=True)
        self._camera.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)

        if not PySpin.IsWritable(self._camera.AcquisitionFrameRateEnable):
            raise RuntimeError("Required PySpin AcquisitionFrameRateEnable is not writable")
        self._camera.AcquisitionFrameRateEnable.SetValue(False)

        if not PySpin.IsWritable(self._camera.ExposureAuto):
            raise RuntimeError("Required PySpin ExposureAuto is not writable")
        self._camera.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
        if PySpin.IsWritable(self._camera.ExposureTime):
            self._camera.ExposureTime.SetValue(float(self._exposure_us))
        self._apply_pyspin_roi(nodemap)

        if self._gain_db is not None and PySpin.IsWritable(self._camera.GainAuto):
            self._camera.GainAuto.SetValue(PySpin.GainAuto_Off)
        if self._gain_db is not None and PySpin.IsWritable(self._camera.Gain):
            self._camera.Gain.SetValue(float(self._gain_db))

        if self._gamma is not None and PySpin.IsWritable(self._camera.GammaEnable):
            self._camera.GammaEnable.SetValue(True)
        if self._gamma is not None and PySpin.IsWritable(self._camera.Gamma):
            self._camera.Gamma.SetValue(float(self._gamma))

        pixel_node = PySpin.CEnumerationPtr(nodemap.GetNode("PixelFormat"))
        if not PySpin.IsAvailable(pixel_node) or not PySpin.IsWritable(pixel_node):
            raise RuntimeError("Required PySpin PixelFormat node is unavailable or unwritable")
        pixel_entry = pixel_node.GetEntryByName(self._pixel_format)
        if not PySpin.IsAvailable(pixel_entry) or not PySpin.IsReadable(pixel_entry):
            raise RuntimeError(f"Required PySpin PixelFormat {self._pixel_format!r} is unavailable")
        pixel_node.SetIntValue(pixel_entry.GetValue())
        self._set_pyspin_enum(nodemap, "TriggerSelector", "FrameStart", required=True)
        self._set_pyspin_enum(nodemap, "TriggerSource", "Line0", required=True)
        self._set_pyspin_enum(nodemap, "TriggerActivation", "RisingEdge", required=True)
        trigger_delay = PySpin.CFloatPtr(nodemap.GetNode("TriggerDelay"))
        if PySpin.IsAvailable(trigger_delay) and PySpin.IsWritable(trigger_delay):
            trigger_delay.SetValue(9.0)
        self._set_pyspin_enum(nodemap, "TriggerMode", "On", required=True)
        self._set_pyspin_enum(nodemap, "ExposureMode", "TriggerWidth", required=True)

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
        except Exception as exc:
            if required:
                raise RuntimeError(
                    f"Required PySpin enum {node_name}={entry_name}: {exc}"
                ) from exc
            return

    def _set_pyspin_int(self, nodemap: Any, node_name: str, value: int) -> None:
        PySpin = self._pyspin
        node = PySpin.CIntegerPtr(nodemap.GetNode(node_name))
        if PySpin.IsAvailable(node) and PySpin.IsWritable(node):
            node.SetValue(max(0, int(value)))

    def _apply_pyspin_roi(self, nodemap: Any) -> None:
        for node_name, value in (
            ("OffsetX", self._roi_offset_x),
            ("OffsetY", self._roi_offset_y),
            ("Width", self._roi_width),
            ("Height", self._roi_height),
        ):
            if value is not None:
                self._set_pyspin_int(nodemap, node_name, value)

    def _capture_loop(self) -> None:
        assert self._camera is not None
        assert self._timestamps is not None
        while self._running:
            image = None
            try:
                image = self._camera.GetNextImage(int(self._acquisition_timeout_ms))
                if image.IsIncomplete():
                    self._incomplete_frames += 1
                    continue
                frame_index = self._frame_count
                filename = f"frame_{frame_index:09d}.png"
                frame_path = self.session_dir / filename
                try:
                    image.Save(str(frame_path))
                except Exception:
                    self._save_errors += 1
                    raise
                camera_timestamp = None
                try:
                    camera_timestamp = int(image.GetTimeStamp())
                except Exception:
                    pass
                self._timestamps.write(frame_index, camera_timestamp, filename)
                self._maybe_emit_preview(
                    image,
                    frame_index=frame_index,
                    camera_timestamp_ns=camera_timestamp,
                )
                self._frame_count += 1
            except Exception as exc:  # pragma: no cover - device-specific
                if "timeout" in str(exc).lower() and self._running:
                    self._acquisition_timeouts += 1
                    continue
                self._error = f"{type(exc).__name__}: {exc}"
                logger.error("Spinnaker capture loop error: %s", self._error)
                break
            finally:
                if image is not None:
                    image.Release()

    def _maybe_emit_preview(
        self,
        image: Any,
        *,
        frame_index: int,
        camera_timestamp_ns: int | None,
    ) -> None:
        if not self._preview_dispatcher.enabled:
            return
        if not self._preview_cadence.due(time.perf_counter()):
            return
        try:
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
                    "backend": "spinnaker_pyspin",
                }
            ):
                self._preview_count += 1
        except Exception:
            logger.debug("Could not emit PySpin preview frame", exc_info=True)
