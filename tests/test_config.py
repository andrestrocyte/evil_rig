from pathlib import Path
from tempfile import TemporaryDirectory

from hf_dm.config import SessionConfig


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
