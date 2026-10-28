from dataclasses import replace
from pathlib import Path

from streamdio.testing import FakeModule, fake_streamdio

from hf_dm.config import SessionConfig
from hf_dm.hardware.bringup_constants import (
    CAP_TOUCH_INPUT_CHANNEL,
    CAP_TOUCH_INPUT_PIN,
    FORCE_ANALOG_CHANNEL,
)
from hf_dm.session.cap_touch_diagnostic import CapTouchDiagnosticState
from hf_dm.session.cap_touch_diagnostic import _interpret_cap_touch_report
from hf_dm.session.cap_touch_diagnostic import run_cap_touch_diagnostic

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "dm_50.yaml"


def test_raw_analog_mode_still_reports_the_detector_status():
    module = FakeModule()
    config = SessionConfig.from_yaml(CONFIG)
    config.ports = replace(
        config.ports, dio_config_port=module.config_port, dio_stream_port=module.stream_port
    )
    with fake_streamdio(module):
        report = run_cap_touch_diagnostic(config, duration_s=0.3, raw_analog=True)

    assert report["mode"] == "raw_analog"
    assert report["status_count"] > 0
    assert report["final_status"]["enabled"] is True
    assert report["final_status"]["baseline"] is not None


def test_cap_touch_diagnostic_counts_lick_edges_status_and_force_separately():
    state = CapTouchDiagnosticState(lick_channels=(CAP_TOUCH_INPUT_CHANNEL,))

    state.handle_event("analog_input", FORCE_ANALOG_CHANNEL, 20, 1000, 0.001)
    state.handle_event("analog_input", FORCE_ANALOG_CHANNEL, 40, 2000, 0.002)
    state.handle_event("analog_input", CAP_TOUCH_INPUT_CHANNEL, 100, 2500, 0.0025)
    state.handle_event("analog_input", CAP_TOUCH_INPUT_CHANNEL, 850, 2600, 0.0026)
    state.handle_event("dio_input", CAP_TOUCH_INPUT_CHANNEL, 1, 3000, 0.003)
    state.handle_event("dio_input", CAP_TOUCH_INPUT_CHANNEL, 0, 4000, 0.004)
    state.add_status(
        {
            "enabled": True,
            "channel": CAP_TOUCH_INPUT_CHANNEL,
            "pin": CAP_TOUCH_INPUT_PIN,
            "state": 0,
            "last_value": 900,
            "baseline": 10,
            "rising_high_threshold": 700,
        }
    )

    report = state.snapshot()

    assert report["high_edge_count"] == 1
    assert report["low_edge_count"] == 1
    assert report["force_sample_count"] == 2
    assert report["force_min"] == 20
    assert report["force_max"] == 40
    assert report["cap_touch_analog_sample_count"] == 2
    assert report["cap_touch_analog_min"] == 100
    assert report["cap_touch_analog_max"] == 850
    assert report["cap_touch_analog_last"] == 850
    assert report["cap_touch_analog_span"] == 750
    assert report["adc_last"] == 900
    assert report["baseline_last"] == 10
    assert report["final_status"]["pin"] == CAP_TOUCH_INPUT_PIN

    state.clear_cap_touch_analog_samples()
    cleared = state.snapshot()
    assert cleared["cap_touch_analog_sample_count"] == 0
    assert cleared["cap_touch_analog_span"] is None


def test_cap_touch_diagnostic_interprets_raw_and_digitizer_reports():
    assert (
        _interpret_cap_touch_report(
            {
                "mode": "raw_analog",
                "detected_raw_change": True,
                "cap_touch_analog_sample_count": 10,
            }
        )
        == "raw_adc_swing_detected"
    )
    assert (
        _interpret_cap_touch_report(
            {
                "mode": "raw_analog",
                "detected_raw_change": False,
                "cap_touch_analog_sample_count": 10,
            }
        )
        == "raw_adc_below_threshold"
    )
    assert (
        _interpret_cap_touch_report(
            {
                "mode": "digitizer",
                "detected_touch": False,
                "status_adc_crossed_threshold": True,
                "adc_max": 500,
            }
        )
        == "adc_crossed_threshold_without_edge"
    )
    assert (
        _interpret_cap_touch_report(
            {
                "mode": "digitizer",
                "detected_touch": False,
                "status_adc_crossed_threshold": False,
                "adc_max": 10,
            }
        )
        == "digitizer_adc_below_threshold"
    )
