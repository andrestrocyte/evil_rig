from pathlib import Path
from tempfile import TemporaryDirectory

from hf_dm.config import SessionConfig
from hf_dm.hardware.bringup_constants import FRAME2TTL_INPUT_CHANNEL, FRAME2TTL_INPUT_PIN


def test_lick_cap_touch_thresholds_load_from_yaml():
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "config.yaml"
        path.write_text(
            """
licks:
  cap_touch_min_high_threshold: 250
  cap_touch_delta_high: 120
  cap_touch_delta_low: 90
""",
            encoding="utf-8",
        )

        config = SessionConfig.from_yaml(path)

    assert config.licks.cap_touch_min_high_threshold == 250
    assert config.licks.cap_touch_delta_high == 120
    assert config.licks.cap_touch_delta_low == 90


def test_frame2ttl_defaults_and_yaml_load():
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "config.yaml"
        path.write_text(
            """
frame2ttl:
  enabled: true
  input_channel: 5
  input_pin: 6
  patch_size_px: 72
  patch_corner: bottom_right
""",
            encoding="utf-8",
        )

        config = SessionConfig.from_yaml(path)

    assert config.frame2ttl.enabled
    assert config.frame2ttl.input_channel == FRAME2TTL_INPUT_CHANNEL
    assert config.frame2ttl.input_pin == FRAME2TTL_INPUT_PIN
    assert config.frame2ttl.patch_size_px == 72
    assert config.frame2ttl.patch_corner == "bottom_right"
