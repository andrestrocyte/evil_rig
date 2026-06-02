"""Headless Spinnaker recorder using PySpin (optional)."""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


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
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.session_dir = self._out_dir / f"{subject_id}_{phase}_{stamp}_spinnaker"
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

        self._pyspin: Any = None
        self._system: Any = None
        self._camera: Any = None

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def error(self) -> str | None:
        return self._error

    def start(self) -> Path:
        """Start camera acquisition and background frame saving."""
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._pyspin = self._load_pyspin()
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
        self._running = True
        self._start_wall_s = time.time()
        self._thread = threading.Thread(target=self._capture_loop, name="spinnaker-capture", daemon=True)
        self._thread.start()
        logger.info("Headless Spinnaker recording started: %s", self.session_dir)
        return self.session_dir

    def stop(self) -> dict[str, Any]:
        """Stop acquisition and return recording summary."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._stop_wall_s = time.time()

        cleanup_error: str | None = None
        try:
            if self._camera is not None:
                self._camera.EndAcquisition()
                self._camera.DeInit()
        except Exception as exc:  # pragma: no cover - device-specific
            cleanup_error = f"{type(exc).__name__}: {exc}"
        finally:
            self._camera = None
            if self._system is not None:
                self._system.ReleaseInstance()
                self._system = None

        summary = {
            "backend": "spinnaker",
            "session_dir": str(self.session_dir),
            "frames_saved": int(self._frame_count),
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

        self._camera.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)

        if PySpin.IsWritable(self._camera.AcquisitionFrameRateEnable):
            self._camera.AcquisitionFrameRateEnable.SetValue(True)
        if PySpin.IsWritable(self._camera.AcquisitionFrameRate):
            self._camera.AcquisitionFrameRate.SetValue(float(self._fps))

        if PySpin.IsWritable(self._camera.ExposureAuto):
            self._camera.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
        if PySpin.IsWritable(self._camera.ExposureTime):
            self._camera.ExposureTime.SetValue(float(self._exposure_us))

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

    def _capture_loop(self) -> None:
        assert self._camera is not None
        while self._running:
            image = None
            try:
                image = self._camera.GetNextImage(int(self._acquisition_timeout_ms))
                if image.IsIncomplete():
                    continue
                frame_path = self.session_dir / f"frame_{self._frame_count:09d}.png"
                image.Save(str(frame_path))
                self._frame_count += 1
            except Exception as exc:  # pragma: no cover - device-specific
                self._error = f"{type(exc).__name__}: {exc}"
                logger.error("Spinnaker capture loop error: %s", self._error)
                break
            finally:
                if image is not None:
                    image.Release()

