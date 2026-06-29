import json
import math
import sys

from hf_dm.ops_cli import main
from hf_dm.valve_calibration import (
    RewardCalibration,
    ValveMeasurement,
    collect_valve_measurements,
    fit_valve_calibration,
    save_valve_calibration,
)


def test_valve_calibration_fits_synthetic_masses_and_round_trips(tmp_path):
    payload = fit_valve_calibration(
        [
            ValveMeasurement(open_time_s=0.04, total_mass_mg=20.0, repetitions=10),
            ValveMeasurement(open_time_s=0.08, total_mass_mg=40.0, repetitions=10),
            ValveMeasurement(open_time_s=0.12, total_mass_mg=60.0, repetitions=10),
        ]
    )

    calibration = save_valve_calibration(payload, tmp_path / "valve.json")
    loaded = RewardCalibration.load(tmp_path / "valve.json")

    assert math.isclose(payload["linear_fit"]["slope_ul_per_s"], 50.0)
    assert math.isclose(payload["linear_fit"]["intercept_ul"], 0.0, abs_tol=1e-12)
    assert math.isclose(payload["linear_fit"]["r_squared"], 1.0)
    assert math.isclose(calibration.volume_for_duration(0.06), 3.0)
    assert loaded.sha256 == calibration.sha256
    assert [point.volume_ul for point in loaded.points] == [2.0, 4.0, 6.0]


def test_collect_valve_measurements_uses_mocked_hardware_and_operator_mass():
    pulses = []
    preparations = []

    measurements = collect_valve_measurements(
        [0.04, 0.08],
        repetitions=3,
        pulse_pump=pulses.append,
        read_mass_mg=lambda duration, repetitions: duration * 1000 * repetitions,
        before_duration=lambda duration, repetitions: preparations.append(
            (duration, repetitions)
        ),
        inter_pulse_s=0.0,
    )

    assert pulses == [0.04, 0.04, 0.04, 0.08, 0.08, 0.08]
    assert preparations == [(0.04, 3), (0.08, 3)]
    assert [measurement.total_mass_mg for measurement in measurements] == [120.0, 240.0]


def test_reward_calibration_rejects_out_of_range_duration(tmp_path):
    calibration = save_valve_calibration(
        fit_valve_calibration(
            [
                ValveMeasurement(0.04, 20.0, 10),
                ValveMeasurement(0.08, 40.0, 10),
            ]
        ),
        tmp_path / "valve.json",
    )

    try:
        calibration.volume_for_duration(0.10)
    except ValueError as exc:
        assert "outside calibration range" in str(exc)
    else:
        raise AssertionError("out-of-range reward duration was extrapolated")


def test_reward_calibration_rejects_decreasing_volume_curve():
    try:
        fit_valve_calibration(
            [
                ValveMeasurement(0.04, 40.0, 10),
                ValveMeasurement(0.08, 20.0, 10),
            ]
        )
    except ValueError as exc:
        assert "must not decrease" in str(exc)
    else:
        raise AssertionError("non-physical decreasing calibration curve was accepted")


def test_calibrate_valve_cli_dispatches_without_touching_hardware(
    tmp_path, monkeypatch, capsys
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("subject_id: mouse01\n", encoding="utf-8")
    called = {}

    def fake_run(config, **kwargs):
        called["config"] = config
        called.update(kwargs)
        return {"output": str(tmp_path / "valve.json"), "points": []}

    monkeypatch.setattr("hf_dm.ops_cli.run_guided_valve_calibration", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hf-dm",
            "calibrate-valve",
            "--config",
            str(config_path),
            "--durations-ms",
            "40,80",
            "--repetitions",
            "3",
            "--yes",
        ],
    )

    main()

    result = json.loads(capsys.readouterr().out)
    assert result["output"].endswith("valve.json")
    assert called["durations_s"] == [0.04, 0.08]
    assert called["repetitions"] == 3
    assert called["overwrite"] is False
