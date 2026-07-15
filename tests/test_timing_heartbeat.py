import math

from hf_dm.timing_heartbeat import analyze_timing_heartbeats


def _rows(
    count: int,
    *,
    teensy_period_s: float = 0.1,
    pc_period_s: float = 0.1,
):
    return [
        (
            "timing_heartbeat",
            -1,
            -1,
            int(round(index * teensy_period_s * 1_000_000)),
            index * teensy_period_s,
            1000.0 + index * pc_period_s,
        )
        for index in range(1, count + 1)
    ]


def _metric(report, name):
    return next(metric for metric in report["metrics"] if metric["name"] == name)


def test_perfect_100_ms_dual_clock_heartbeat_passes():
    report = analyze_timing_heartbeats(_rows(201))

    assert report["overall_status"] == "PASS"
    assert report["heartbeat_count"] == 201
    assert report["interval_count"] == 200
    assert report["estimated_missed_pulses"] == 0
    assert math.isclose(report["statistics"]["teensy_interval_s"]["mean"], 0.1)
    assert math.isclose(report["statistics"]["pc_interval_s"]["mean"], 0.1)
    assert abs(report["final_pc_minus_teensy_elapsed_s"]) < 1e-9
    assert _metric(report, "pc_vs_teensy_rate_error_ppm")["status"] == "PASS"


def test_accumulating_pc_clock_lag_is_reported_and_fails_rate_qc():
    report = analyze_timing_heartbeats(_rows(201, pc_period_s=0.1002))

    assert report["overall_status"] == "FAIL"
    assert math.isclose(
        report["final_pc_minus_teensy_elapsed_s"],
        0.04,
        rel_tol=0,
        abs_tol=1e-9,
    )
    assert math.isclose(
        report["pc_vs_teensy_rate_error_ppm"],
        2000.0,
        rel_tol=0,
        abs_tol=1e-5,
    )
    assert _metric(report, "pc_vs_teensy_rate_error_ppm")["status"] == "FAIL"
    assert _metric(report, "pc_interval_p99_abs_error_s")["status"] == "PASS"


def test_transient_pc_delivery_backlog_is_visible_in_interval_and_residual_qc():
    rows = _rows(201)
    delayed = []
    for index, row in enumerate(rows):
        values = list(row)
        if index >= 100:
            values[5] += 0.15
        delayed.append(tuple(values))

    report = analyze_timing_heartbeats(delayed)

    assert report["overall_status"] in {"WARN", "FAIL"}
    assert report["statistics"]["pc_interval_s"]["max"] > 0.24
    assert report["max_abs_pc_minus_teensy_elapsed_s"] >= 0.149
    assert _metric(report, "pc_interval_max_abs_error_s")["status"] == "WARN"
    assert _metric(report, "pc_delivery_detrended_max_abs_s")["status"] != "PASS"


def test_missing_firmware_beat_is_inferred_from_teensy_timestamp_gap():
    rows = _rows(20)
    del rows[9]

    report = analyze_timing_heartbeats(rows)

    assert report["overall_status"] == "FAIL"
    assert report["estimated_missed_pulses"] == 1
    assert _metric(report, "estimated_missed_pulses")["status"] == "WARN"
    assert _metric(report, "teensy_interval_max_abs_error_s")["status"] == "FAIL"


def test_nonmonotonic_pc_timestamp_fails_heartbeat_qc():
    rows = _rows(10)
    values = list(rows[5])
    values[5] = rows[4][5]
    rows[5] = tuple(values)

    report = analyze_timing_heartbeats(rows)

    assert report["overall_status"] == "FAIL"
    assert _metric(report, "pc_nonmonotonic_intervals")["value"] == 1
    assert _metric(report, "pc_nonmonotonic_intervals")["status"] == "FAIL"
