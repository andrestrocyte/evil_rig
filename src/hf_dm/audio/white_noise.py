"""Continuous white noise + Bpod gate (speaker_bringup_pin27_whitenoise.ipynb pattern)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _ensure_sound_module() -> None:
    for root in (Path.home() / "MM_PyBpod2" / "src", Path(__file__).resolve().parents[4] / "MM_PyBpod2" / "src"):
        if root.is_dir() and str(root) not in sys.path:
            sys.path.insert(0, str(root))


class WhiteNoisePlayer:
    """Optional PC audio; FSM gates pin 27 via serial relay during timeout."""

    def __init__(self) -> None:
        self._control = None

    def start_continuous(self) -> bool:
        _ensure_sound_module()
        try:
            from pybpod_soundcard_module.soundcontrol import SoundControl

            self._control = SoundControl()
            return bool(self._control.start_continuous_white_noise(volume=1.0))
        except Exception as exc:
            logger.warning("White noise PC playback unavailable: %s", exc)
            return False

    def stop(self) -> None:
        if self._control is not None:
            try:
                self._control.stop_continuous_playback()
            except Exception:
                pass
            self._control = None
