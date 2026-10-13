"""Continuous PC white noise for the pin-26 hardware gate."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class WhiteNoisePlayer:
    """Bringup-exact SoundControl lifecycle for continuous gated white noise."""

    def __init__(self, *, frequency_hz: float = 6000.0, volume: float = 1.0) -> None:
        self.frequency_hz = float(frequency_hz)
        self.volume = float(volume)
        self._control = None
        self._status: dict = {}

    def start_continuous(self) -> bool:
        try:
            from pybpod_soundcard_module import SoundControl

            control = SoundControl.create_from_protocol(
                sound_frequency=self.frequency_hz,
                sound_volume=self.volume,
                sound_type="white_noise",
            )
            if control is None:
                raise RuntimeError("SoundControl.create_from_protocol returned None")
            self._control = control
            if not control.start_continuous_playback_from_protocol():
                raise RuntimeError("continuous white-noise playback did not start")
            self._status = dict(control.get_status())
            if not self._status.get("continuous_playback_active", False):
                raise RuntimeError(f"unexpected SoundControl status: {self._status}")
            return True
        except Exception as exc:
            logger.exception("White noise PC playback unavailable: %s", exc)
            self.stop()
            return False

    @property
    def status(self) -> dict:
        if self._control is not None:
            try:
                self._status = dict(self._control.get_status())
            except Exception:
                pass
        return dict(self._status)

    def stop(self) -> None:
        if self._control is not None:
            self._control.stop_and_cleanup()
            self._control = None
        self._status = {}
