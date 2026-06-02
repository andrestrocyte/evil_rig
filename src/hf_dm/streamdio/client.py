"""StreamDIO wrapper — imports MM_PyBpod2 module (same as bringup notebooks)."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Callable

from hf_dm.hardware.bringup_constants import (
    DEFAULT_CAMERA_EXPOSURE_US,
    DEFAULT_CAMERA_FREQUENCY_HZ,
    DIRECTION_INPUT_CHANNEL,
    FORCE_ANALOG_CHANNEL,
    FORCE_ANALOG_PIN,
    FORCE_ANALOG_SAMPLE_INTERVAL_US,
    INPUT_REFRACTORY_PERIOD_US,
    PULSE_INPUT_CHANNEL,
)
from hf_dm.hardware.discovery import resolve_streamdio_ports


def _ensure_mm_pybpod2_src() -> None:
    """Add MM_PyBpod2/src when developing alongside the bringup repo."""
    candidates = [
        Path(__file__).resolve().parents[4] / "MM_PyBpod2" / "src",
        Path.home() / "MM_PyBpod2" / "src",
    ]
    for path in candidates:
        if path.is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


class StreamDIOClient:
    """Thin facade over pybpod_streamDIO_module.StreamDIOProtocol."""

    def __init__(self, config_port: str = "AUTO", stream_port: str = "AUTO", *, baudrate: int):
        _ensure_mm_pybpod2_src()
        from pybpod_streamDIO_module.stream_dio_protocol import StreamDIOProtocol

        self._protocol_cls = StreamDIOProtocol
        self.config_port = config_port
        self.stream_port = stream_port
        self.baudrate = baudrate
        self._dio: Any = None

    def open(self, excluded_bpod_port: str | None = None) -> None:
        cfg, stream = resolve_streamdio_ports(
            self.config_port,
            self.stream_port,
            excluded_port=excluded_bpod_port,
            baud=self.baudrate,
        )
        self.config_port = cfg
        self.stream_port = stream
        self._dio = self._protocol_cls(config_port=cfg, stream_port=stream, connect=False)
        self._dio.open()

    def configure_rotary_and_force(self) -> None:
        if self._dio is None:
            raise RuntimeError("StreamDIO not connected")
        enable_vector = [False] * int(self._dio.module_info["n_input_channels"])
        enable_vector[PULSE_INPUT_CHANNEL - 1] = True
        enable_vector[DIRECTION_INPUT_CHANNEL - 1] = True
        ok = self._dio.configure_module(
            refractory_period=INPUT_REFRACTORY_PERIOD_US,
            input_channels_enabled=enable_vector,
            analog_input={
                "channel": FORCE_ANALOG_CHANNEL,
                "enabled": True,
                "sample_interval_us": FORCE_ANALOG_SAMPLE_INTERVAL_US,
            },
        )
        if not ok:
            raise RuntimeError("StreamDIO configure_module failed")
        status = self._dio.get_analog_input_status()
        if not status.get("enabled"):
            raise RuntimeError(f"Analog force not enabled: {status}")
        if int(status.get("channel", -1)) != FORCE_ANALOG_CHANNEL:
            raise RuntimeError(f"Unexpected force channel: {status}")
        if int(status.get("pin", -1)) != FORCE_ANALOG_PIN:
            raise RuntimeError(f"Unexpected force pin: {status}")

    def start_camera_session(
        self,
        frequency_hz: int = DEFAULT_CAMERA_FREQUENCY_HZ,
        exposure_us: int = DEFAULT_CAMERA_EXPOSURE_US,
    ) -> None:
        if self._dio is None:
            raise RuntimeError("StreamDIO not connected")
        if not self._dio.start_camera_triggers(frequency=frequency_hz, exposure_us=exposure_us):
            raise RuntimeError("Failed to start camera triggers on StreamDIO")

    def stop_camera(self) -> None:
        if self._dio is not None:
            self._dio.stop_camera_triggers()

    def start_streaming(self, callback: Callable[..., None] | None = None) -> None:
        if self._dio is None:
            raise RuntimeError("StreamDIO not connected")
        if not self._dio.start_streaming(callback=callback):
            raise RuntimeError("StreamDIO start_streaming failed")

    def stop_streaming(self) -> None:
        if self._dio is not None and getattr(self._dio, "streaming_active", False):
            self._dio.stop_streaming()
            time.sleep(0.05)

    def get_camera_events(self) -> list[tuple[int, float]]:
        """Return (timestamp_us, relative_s) for camera TTL events in history."""
        if self._dio is None:
            return []
        events = self._dio.get_event_history(event_types=["camera"])
        return [(int(e[3]), float(e[-1])) for e in events]

    def close(self) -> None:
        if self._dio is None:
            return
        try:
            self.stop_streaming()
            self.stop_camera()
            self._dio.close()
        except Exception:
            pass
        self._dio = None

    @property
    def raw(self) -> Any:
        return self._dio
