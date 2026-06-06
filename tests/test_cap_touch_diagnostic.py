from hf_dm.hardware.bringup_constants import (
    CAP_TOUCH_INPUT_CHANNEL,
    CAP_TOUCH_INPUT_PIN,
    FORCE_ANALOG_CHANNEL,
)
from hf_dm.session.cap_touch_diagnostic import CapTouchDiagnosticState


def test_cap_touch_diagnostic_counts_lick_edges_status_and_force_separately():
    state = CapTouchDiagnosticState(lick_channels=(CAP_TOUCH_INPUT_CHANNEL,))

    state.handle_event("analog_input", FORCE_ANALOG_CHANNEL, 20, 1000, 0.001)
    state.handle_event("analog_input", FORCE_ANALOG_CHANNEL, 40, 2000, 0.002)
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
    assert report["adc_last"] == 900
    assert report["baseline_last"] == 10
    assert report["final_status"]["pin"] == CAP_TOUCH_INPUT_PIN
