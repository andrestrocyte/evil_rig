"""Headless Spinnaker recorder using PySpin (optional)."""

from __future__ import annotations

import csv
import ctypes
import json
import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


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
        gain_db: float | None,
        gamma: float | None,
        acquisition_timeout_ms: int,
    ) -> None:
        self.session_dir = session_dir
        self._serial_number = serial_number
        self._fps = int(fps)
        self._exposure_us = int(exposure_us)
        self._pixel_format = str(pixel_format)
        self._gain_db = gain_db
        self._gamma = gamma
        self._acquisition_timeout_ms = int(acquisition_timeout_ms)
        self._running = False
        self._thread: threading.Thread | None = None
        self._error: str | None = None
        self._frame_count = 0
        self._start_wall_s: float | None = None
        self._stop_wall_s: float | None = None
        self._dll_dir_handle: Any = None
        self._dll: Any = None
        self._fn: dict[str, Any] = {}
        self._system = ctypes.c_void_p()
        self._camera_list = ctypes.c_void_p()
        self._camera = ctypes.c_void_p()
        self._camera_initialized = False
        self._acquisition_started = False
        self._timestamps: _FrameTimestampWriter | None = None

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def error(self) -> str | None:
        return self._error

    def start(self) -> Path:
        try:
            self._load_api()
            self._check("spinSystemGetInstance", ctypes.byref(self._system))
            self._check("spinCameraListCreateEmpty", ctypes.byref(self._camera_list))
            self._check("spinSystemGetCameras", self._system, self._camera_list)
            self._camera = self._select_camera()
            self._check("spinCameraInit", self._camera)
            self._camera_initialized = True
            self._configure_camera()
            self._check("spinCameraBeginAcquisition", self._camera)
            self._acquisition_started = True
            self._timestamps = _FrameTimestampWriter(self.session_dir)
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
        self._stop_wall_s = time.time()
        cleanup_errors = self._cleanup()
        if self._timestamps is not None:
            self._timestamps.close()

        summary = {
            "backend": "spinnaker_c",
            "session_dir": str(self.session_dir),
            "frames_saved": int(self._frame_count),
            "frame_timestamps_csv": str(self.session_dir / "frame_timestamps.csv"),
            "start_wall_s": self._start_wall_s,
            "stop_wall_s": self._stop_wall_s,
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
        self._set_enum(node_map, "TriggerMode", "Off")
        self._set_enum(node_map, "AcquisitionMode", "Continuous")
        self._set_bool(node_map, "AcquisitionFrameRateEnable", False)
        self._set_enum(node_map, "ExposureAuto", "Off")
        self._set_float(node_map, "ExposureTime", self._exposure_us)
        self._set_enum(node_map, "ExposureMode", "TriggerWidth")
        self._set_enum(node_map, "PixelFormat", self._pixel_format)
        if self._gain_db is not None:
            self._set_enum(node_map, "GainAuto", "Off")
            self._set_float(node_map, "Gain", self._gain_db)
        if self._gamma is not None:
            self._set_bool(node_map, "GammaEnable", True)
            self._set_float(node_map, "Gamma", self._gamma)
        self._set_enum(node_map, "TriggerSelector", "FrameStart")
        self._set_enum(node_map, "TriggerSource", "Line0")
        self._set_enum(node_map, "TriggerActivation", "RisingEdge")
        self._set_float(node_map, "TriggerDelay", 9.0)
        self._set_enum(node_map, "TriggerMode", "On")

    def _get_node(self, node_map: ctypes.c_void_p, name: str) -> ctypes.c_void_p | None:
        node = ctypes.c_void_p()
        result = self._fn["spinNodeMapGetNode"](node_map, name.encode(), ctypes.byref(node))
        return node if result == 0 and node.value else None

    def _set_enum(self, node_map: ctypes.c_void_p, name: str, entry_name: str) -> None:
        node = self._get_node(node_map, name)
        if node is None:
            return
        entry = ctypes.c_void_p()
        if self._fn["spinEnumerationGetEntryByName"](
            node,
            entry_name.encode(),
            ctypes.byref(entry),
        ) != 0:
            return
        value = ctypes.c_int64()
        if self._fn["spinEnumerationEntryGetIntValue"](entry, ctypes.byref(value)) != 0:
            return
        result = self._fn["spinEnumerationSetIntValue"](node, value)
        if result != 0:
            logger.debug("Could not set Spinnaker enum %s=%s (%d)", name, entry_name, result)

    def _set_float(self, node_map: ctypes.c_void_p, name: str, value: float) -> None:
        node = self._get_node(node_map, name)
        if node is None:
            return
        result = self._fn["spinFloatSetValue"](node, float(value))
        if result != 0:
            logger.debug("Could not set Spinnaker float %s=%s (%d)", name, value, result)

    def _set_bool(self, node_map: ctypes.c_void_p, name: str, value: bool) -> None:
        node = self._get_node(node_map, name)
        if node is None:
            return
        result = self._fn["spinBooleanSetValue"](node, 1 if value else 0)
        if result != 0:
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
                continue
            if result != 0:
                self._error = f"spinCameraGetNextImageEx failed with Spinnaker error {result}"
                logger.error("Spinnaker C capture loop error: %s", self._error)
                break
            try:
                incomplete = ctypes.c_ubyte()
                if self._fn["spinImageIsIncomplete"](image, ctypes.byref(incomplete)) != 0:
                    continue
                if incomplete.value:
                    continue
                frame_index = self._frame_count
                filename = f"frame_{frame_index:09d}.pgm"
                frame_path = self.session_dir / filename
                result = int(self._fn["spinImageSaveFromExt"](image, os.fsencode(frame_path)))
                if result != 0:
                    self._error = f"spinImageSaveFromExt failed with Spinnaker error {result}"
                    logger.error("Spinnaker C capture loop error: %s", self._error)
                    break
                camera_timestamp = ctypes.c_uint64()
                timestamp_result = self._fn["spinImageGetTimeStamp"](
                    image,
                    ctypes.byref(camera_timestamp),
                )
                self._timestamps.write(
                    frame_index,
                    camera_timestamp.value if timestamp_result == 0 else None,
                    filename,
                )
                self._frame_count += 1
            finally:
                self._fn["spinImageRelease"](image)

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
        gain_db: float | None = None,
        gamma: float | None = None,
        acquisition_timeout_ms: int = 1000,
    ) -> None:
        self._out_dir = Path(out_dir)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.session_dir = self._out_dir / f"{subject_id}_{phase}_{stamp}_spinnaker"
        self._serial_number = serial_number
        self._fps = int(fps)
        self._exposure_us = int(exposure_us)
        self._pixel_format = str(pixel_format)
        self._gain_db = gain_db
        self._gamma = gamma
        self._acquisition_timeout_ms = int(acquisition_timeout_ms)
        self.backend = "spinnaker_pyspin"

        self._running = False
        self._thread: threading.Thread | None = None
        self._error: str | None = None
        self._frame_count = 0
        self._start_wall_s: float | None = None
        self._stop_wall_s: float | None = None

        self._pyspin: Any = None
        self._system: Any = None
        self._camera: Any = None
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
                gain_db=self._gain_db,
                gamma=self._gamma,
                acquisition_timeout_ms=self._acquisition_timeout_ms,
            )
            return self._c_backend.start()

        try:
            self._system = self._pyspin.System.GetInstance()
            cam_list = self._system.GetCameras()
            if cam_list.GetSize() == 0:
                cam_list.Clear()
                raise RuntimeError("No Spinnaker cameras found")

            self._camera = self._select_camera(cam_list)
            cam_list.Clear()
            self._camera.Init()
            self._configure_camera()
            self._camera.BeginAcquisition()
            self._timestamps = _FrameTimestampWriter(self.session_dir)
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
        self._stop_wall_s = time.time()
        cleanup_error = self._cleanup_pyspin()
        if self._timestamps is not None:
            self._timestamps.close()

        summary = {
            "backend": self.backend,
            "session_dir": str(self.session_dir),
            "frames_saved": int(self._frame_count),
            "frame_timestamps_csv": str(self.session_dir / "frame_timestamps.csv"),
            "start_wall_s": self._start_wall_s,
            "stop_wall_s": self._stop_wall_s,
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

    def _configure_camera(self) -> None:
        PySpin = self._pyspin
        nodemap = self._camera.GetNodeMap()

        self._set_pyspin_enum(nodemap, "TriggerMode", "Off")
        self._camera.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)

        if PySpin.IsWritable(self._camera.AcquisitionFrameRateEnable):
            self._camera.AcquisitionFrameRateEnable.SetValue(False)

        if PySpin.IsWritable(self._camera.ExposureAuto):
            self._camera.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
        if PySpin.IsWritable(self._camera.ExposureTime):
            self._camera.ExposureTime.SetValue(float(self._exposure_us))
        self._set_pyspin_enum(nodemap, "ExposureMode", "TriggerWidth")

        if self._gain_db is not None and PySpin.IsWritable(self._camera.GainAuto):
            self._camera.GainAuto.SetValue(PySpin.GainAuto_Off)
        if self._gain_db is not None and PySpin.IsWritable(self._camera.Gain):
            self._camera.Gain.SetValue(float(self._gain_db))

        if self._gamma is not None and PySpin.IsWritable(self._camera.GammaEnable):
            self._camera.GammaEnable.SetValue(True)
        if self._gamma is not None and PySpin.IsWritable(self._camera.Gamma):
            self._camera.Gamma.SetValue(float(self._gamma))

        pixel_node = PySpin.CEnumerationPtr(nodemap.GetNode("PixelFormat"))
        if PySpin.IsAvailable(pixel_node) and PySpin.IsWritable(pixel_node):
            pixel_entry = pixel_node.GetEntryByName(self._pixel_format)
            if PySpin.IsAvailable(pixel_entry) and PySpin.IsReadable(pixel_entry):
                pixel_node.SetIntValue(pixel_entry.GetValue())
            else:
                logger.warning("Pixel format '%s' unavailable, keeping camera default", self._pixel_format)
        self._set_pyspin_enum(nodemap, "TriggerSelector", "FrameStart")
        self._set_pyspin_enum(nodemap, "TriggerSource", "Line0")
        self._set_pyspin_enum(nodemap, "TriggerActivation", "RisingEdge")
        trigger_delay = PySpin.CFloatPtr(nodemap.GetNode("TriggerDelay"))
        if PySpin.IsAvailable(trigger_delay) and PySpin.IsWritable(trigger_delay):
            trigger_delay.SetValue(9.0)
        self._set_pyspin_enum(nodemap, "TriggerMode", "On")

    def _set_pyspin_enum(self, nodemap: Any, node_name: str, entry_name: str) -> None:
        PySpin = self._pyspin
        node = PySpin.CEnumerationPtr(nodemap.GetNode(node_name))
        if not PySpin.IsAvailable(node) or not PySpin.IsWritable(node):
            return
        entry = node.GetEntryByName(entry_name)
        if PySpin.IsAvailable(entry) and PySpin.IsReadable(entry):
            node.SetIntValue(entry.GetValue())

    def _capture_loop(self) -> None:
        assert self._camera is not None
        assert self._timestamps is not None
        while self._running:
            image = None
            try:
                image = self._camera.GetNextImage(int(self._acquisition_timeout_ms))
                if image.IsIncomplete():
                    continue
                frame_index = self._frame_count
                filename = f"frame_{frame_index:09d}.png"
                frame_path = self.session_dir / filename
                image.Save(str(frame_path))
                camera_timestamp = None
                try:
                    camera_timestamp = int(image.GetTimeStamp())
                except Exception:
                    pass
                self._timestamps.write(frame_index, camera_timestamp, filename)
                self._frame_count += 1
            except Exception as exc:  # pragma: no cover - device-specific
                if "timeout" in str(exc).lower() and self._running:
                    continue
                self._error = f"{type(exc).__name__}: {exc}"
                logger.error("Spinnaker capture loop error: %s", self._error)
                break
            finally:
                if image is not None:
                    image.Release()

