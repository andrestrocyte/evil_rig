"""Dual-clock QC for the firmware-origin StreamDIO 100 ms heartbeat."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from hf_dm.hardware.bringup_constants import (
    STREAMDIO_TIMING_HEARTBEAT_EVENT_CODE,
    STREAMDIO_TIMING_HEARTBEAT_PERIOD_US,
)

TIMING_HEARTBEAT_EVENT_TYPE = "timing_heartbeat"


@dataclass(frozen=True)
class TimingHeartbeatThresholds:
    """Conservative acquisition thresholds, separate from the 100 ms contract."""

    min_heartbeat_count: int = 3
    warn_heartbeat_count: int = 2
    module_interval_pass_error_s: float = 0.0005
    module_interval_warn_error_s: float = 0.002
    pc_interval_p99_pass_error_s: float = 0.010
    pc_interval_p99_warn_error_s: float = 0.050
    pc_interval_max_pass_error_s: float = 0.050
    pc_interval_max_warn_error_s: float = 0.200
    detrended_residual_pass_s: float = 0.020
    detrended_residual_warn_s: float = 0.100
    rate_min_span_s: float = 10.0
    rate_pass_error_ppm: float = 200.0
    rate_warn_error_ppm: float = 1000.0
    missed_pulses_pass: int = 0
    missed_pulses_warn: int = 1


def analyze_timing_heartbeats(
    streamdio_rows: Iterable[Any],
    *,
    session_events: Iterable[Any] | None = None,
    thresholds: TimingHeartbeatThresholds | None = None,
) -> dict[str, Any]:
    """Analyze ordered heartbeat rows without changing their acquisition order.

    The Teensy timestamp measures when firmware emitted the beat. The PC
    ``perf_counter`` timestamp measures when Python dispatched the same event.
    Their elapsed-time difference exposes accumulating clock/transport lag while
    detrended residuals expose transient USB or host-processing backlog.
    """

    limits = thresholds or TimingHeartbeatThresholds()
    expected_s = STREAMDIO_TIMING_HEARTBEAT_PERIOD_US / 1_000_000.0
    wall_by_module_us = _heartbeat_wall_times(session_events or ())
    source_rows = [
        row
        for raw in streamdio_rows
        if (row := _normalize_streamdio_row(raw)) is not None
        and row["event_type"] == TIMING_HEARTBEAT_EVENT_TYPE
    ]

    samples: list[dict[str, Any]] = []
    module_intervals: list[float] = []
    pc_intervals: list[float] = []
    interval_differences: list[float] = []
    cumulative_differences: list[float] = []
    module_nonmonotonic = 0
    pc_nonmonotonic = 0
    estimated_missed = 0

    first_module_s = source_rows[0]["module_timestamp_us"] / 1_000_000.0 if source_rows else None
    first_pc_s = source_rows[0]["host_perf_counter_s"] if source_rows else None
    prior_module_s: float | None = None
    prior_pc_s: float | None = None
    for index, row in enumerate(source_rows):
        module_us = int(row["module_timestamp_us"])
        module_s = module_us / 1_000_000.0
        pc_s = float(row["host_perf_counter_s"])
        module_elapsed_s = None if first_module_s is None else module_s - first_module_s
        pc_elapsed_s = None if first_pc_s is None else pc_s - first_pc_s
        module_interval_s = None if prior_module_s is None else module_s - prior_module_s
        pc_interval_s = None if prior_pc_s is None else pc_s - prior_pc_s
        interval_difference_s = (
            None
            if module_interval_s is None or pc_interval_s is None
            else pc_interval_s - module_interval_s
        )
        cumulative_difference_s = (
            None
            if module_elapsed_s is None or pc_elapsed_s is None
            else pc_elapsed_s - module_elapsed_s
        )
        if module_interval_s is not None:
            module_intervals.append(module_interval_s)
            if module_interval_s <= 0:
                module_nonmonotonic += 1
            else:
                periods = max(1, int(round(module_interval_s / expected_s)))
                if module_interval_s > expected_s * 1.5:
                    estimated_missed += max(0, periods - 1)
        if pc_interval_s is not None:
            pc_intervals.append(pc_interval_s)
            if pc_interval_s <= 0:
                pc_nonmonotonic += 1
        if interval_difference_s is not None:
            interval_differences.append(interval_difference_s)
        if cumulative_difference_s is not None:
            cumulative_differences.append(cumulative_difference_s)
        samples.append(
            {
                "sequence": index + 1,
                "teensy_timestamp_us": module_us,
                # MM_PyBpod2 derives this from host wall time at batch dispatch;
                # it is retained for provenance but is not a Teensy clock.
                "driver_relative_time_s": row["relative_time_s"],
                "teensy_elapsed_s": module_elapsed_s,
                "teensy_interval_s": module_interval_s,
                "pc_perf_counter_s": pc_s,
                "pc_wall_time_s": wall_by_module_us.get(module_us),
                "pc_elapsed_s": pc_elapsed_s,
                "pc_interval_s": pc_interval_s,
                "pc_minus_teensy_interval_s": interval_difference_s,
                "pc_minus_teensy_elapsed_s": cumulative_difference_s,
            }
        )
        prior_module_s = module_s
        prior_pc_s = pc_s

    module_errors = [abs(value - expected_s) for value in module_intervals]
    pc_errors = [abs(value - expected_s) for value in pc_intervals]
    module_elapsed = [float(sample["teensy_elapsed_s"] or 0.0) for sample in samples]
    pc_elapsed = [float(sample["pc_elapsed_s"] or 0.0) for sample in samples]
    clock_fit = _fit_elapsed_clocks(module_elapsed, pc_elapsed)
    module_span_s = module_elapsed[-1] if module_elapsed else 0.0
    rate_error_ppm = (
        None
        if clock_fit["slope"] is None
        else (float(clock_fit["slope"]) - 1.0) * 1_000_000.0
    )
    graded_rate_error_ppm = (
        abs(rate_error_ppm)
        if module_span_s >= limits.rate_min_span_s and rate_error_ppm is not None
        else None
    )
    residuals = [float(value) for value in clock_fit["residuals_s"]]

    metrics = [
        _higher_metric(
            "heartbeat_count",
            len(samples),
            limits.min_heartbeat_count,
            limits.warn_heartbeat_count,
            "At least three firmware beats are required to evaluate two intervals",
        ),
        _lower_metric(
            "estimated_missed_pulses",
            estimated_missed,
            limits.missed_pulses_pass,
            limits.missed_pulses_warn,
            "Missing firmware beats inferred from Teensy timestamp gaps",
        ),
        _lower_metric(
            "teensy_interval_max_abs_error_s",
            max(module_errors) if module_errors else None,
            limits.module_interval_pass_error_s,
            limits.module_interval_warn_error_s,
            "Worst absolute Teensy interval error from the 100 ms contract",
        ),
        _lower_metric(
            "pc_interval_p99_abs_error_s",
            _percentile(pc_errors, 99.0),
            limits.pc_interval_p99_pass_error_s,
            limits.pc_interval_p99_warn_error_s,
            "99th percentile PC dispatch interval error from 100 ms",
        ),
        _lower_metric(
            "pc_interval_max_abs_error_s",
            max(pc_errors) if pc_errors else None,
            limits.pc_interval_max_pass_error_s,
            limits.pc_interval_max_warn_error_s,
            "Worst PC dispatch interval error from 100 ms",
        ),
        _lower_metric(
            "pc_delivery_detrended_max_abs_s",
            max((abs(value) for value in residuals), default=None),
            limits.detrended_residual_pass_s,
            limits.detrended_residual_warn_s,
            "Worst transient PC delivery deviation after removing linear clock drift",
        ),
        _lower_metric(
            "pc_vs_teensy_rate_error_ppm",
            graded_rate_error_ppm,
            limits.rate_pass_error_ppm,
            limits.rate_warn_error_ppm,
            (
                "Absolute PC-versus-Teensy clock-rate error"
                if module_span_s >= limits.rate_min_span_s
                else f"Not graded until at least {limits.rate_min_span_s:g} s of heartbeat data"
            ),
            missing_status="N/A",
        ),
        _lower_metric(
            "teensy_nonmonotonic_intervals",
            module_nonmonotonic,
            0,
            0,
            "Non-positive Teensy timestamp intervals",
        ),
        _lower_metric(
            "pc_nonmonotonic_intervals",
            pc_nonmonotonic,
            0,
            0,
            "Non-positive PC perf-counter intervals",
        ),
    ]
    graded_statuses = {str(metric["status"]) for metric in metrics if metric["status"] != "N/A"}
    overall_status = (
        "FAIL"
        if "FAIL" in graded_statuses
        else "WARN" if "WARN" in graded_statuses else "PASS"
    )

    return {
        "schema_version": 1,
        "kind": "hf_dm_streamdio_timing_heartbeat_qc",
        "generated_wall_time_s": time.time(),
        "event_type": TIMING_HEARTBEAT_EVENT_TYPE,
        "event_code": STREAMDIO_TIMING_HEARTBEAT_EVENT_CODE,
        "expected_interval_s": expected_s,
        "overall_status": overall_status,
        "heartbeat_count": len(samples),
        "interval_count": max(0, len(samples) - 1),
        "estimated_missed_pulses": estimated_missed,
        "first_teensy_timestamp_us": (
            None if not samples else samples[0]["teensy_timestamp_us"]
        ),
        "last_teensy_timestamp_us": (
            None if not samples else samples[-1]["teensy_timestamp_us"]
        ),
        "teensy_span_s": module_span_s if samples else None,
        "first_pc_perf_counter_s": (
            None if not samples else samples[0]["pc_perf_counter_s"]
        ),
        "last_pc_perf_counter_s": (
            None if not samples else samples[-1]["pc_perf_counter_s"]
        ),
        "pc_span_s": pc_elapsed[-1] if pc_elapsed else None,
        "final_pc_minus_teensy_elapsed_s": (
            cumulative_differences[-1] if cumulative_differences else None
        ),
        "max_abs_pc_minus_teensy_elapsed_s": max(
            (abs(value) for value in cumulative_differences),
            default=None,
        ),
        "pc_vs_teensy_rate_error_ppm": rate_error_ppm,
        "statistics": {
            "teensy_interval_s": _statistics(module_intervals),
            "pc_interval_s": _statistics(pc_intervals),
            "pc_minus_teensy_interval_s": _statistics(interval_differences),
            "pc_minus_teensy_elapsed_s": _statistics(cumulative_differences),
            "pc_delivery_detrended_residual_s": _statistics(residuals),
        },
        "clock_fit": {
            key: value for key, value in clock_fit.items() if key != "residuals_s"
        },
        "thresholds": asdict(limits),
        "metrics": metrics,
        "samples": samples,
    }


def timing_heartbeat_qc_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Return the compact form embedded in primary JSON/NPZ and the manifest."""

    return {key: value for key, value in report.items() if key != "samples"}


def _normalize_streamdio_row(row: Any) -> dict[str, Any] | None:
    if isinstance(row, dict):
        event_type = row.get("event_type")
        module_timestamp_us = row.get("module_timestamp_us", row.get("timestamp_us"))
        relative_time_s = row.get("relative_time_s")
        host_perf_counter_s = row.get("host_perf_counter_s")
    else:
        try:
            values = list(row)
        except TypeError:
            return None
        if len(values) < 6:
            return None
        event_type = values[0]
        module_timestamp_us = values[3]
        relative_time_s = values[4]
        host_perf_counter_s = values[5]
    if (
        module_timestamp_us is None
        or relative_time_s is None
        or host_perf_counter_s is None
    ):
        return None
    try:
        return {
            "event_type": str(event_type),
            "module_timestamp_us": int(module_timestamp_us),
            "relative_time_s": float(relative_time_s),
            "host_perf_counter_s": float(host_perf_counter_s),
        }
    except (TypeError, ValueError, OverflowError):
        return None


def _heartbeat_wall_times(events: Iterable[Any]) -> dict[int, float]:
    wall_by_module_us: dict[int, float] = {}
    for event in events:
        if isinstance(event, dict):
            event_type = event.get("event_type")
            payload = event.get("payload") or {}
        else:
            event_type = getattr(event, "event_type", None)
            payload = getattr(event, "payload", {}) or {}
        if event_type != TIMING_HEARTBEAT_EVENT_TYPE or not isinstance(payload, dict):
            continue
        try:
            module_us = int(payload["module_timestamp_us"])
            wall_s = float(payload["host_wall_time_s"])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(wall_s):
            wall_by_module_us[module_us] = wall_s
    return wall_by_module_us


def _fit_elapsed_clocks(source: list[float], target: list[float]) -> dict[str, Any]:
    if len(source) < 2 or len(source) != len(target):
        return {
            "status": "insufficient_samples",
            "slope": None,
            "intercept_s": None,
            "rms_residual_s": None,
            "max_abs_residual_s": None,
            "residuals_s": [],
        }
    source_mean = sum(source) / len(source)
    target_mean = sum(target) / len(target)
    denominator = sum((value - source_mean) ** 2 for value in source)
    if denominator <= 0:
        return {
            "status": "degenerate_samples",
            "slope": None,
            "intercept_s": None,
            "rms_residual_s": None,
            "max_abs_residual_s": None,
            "residuals_s": [],
        }
    slope = sum(
        (x - source_mean) * (y - target_mean) for x, y in zip(source, target)
    ) / denominator
    intercept = target_mean - slope * source_mean
    residuals = [y - (slope * x + intercept) for x, y in zip(source, target)]
    return {
        "status": "ok",
        "slope": slope,
        "intercept_s": intercept,
        "rms_residual_s": math.sqrt(
            sum(value * value for value in residuals) / len(residuals)
        ),
        "max_abs_residual_s": max(abs(value) for value in residuals),
        "residuals_s": residuals,
    }


def _statistics(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "max": None,
            "std": None,
            "p50": None,
            "p95": None,
            "p99": None,
        }
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "count": len(values),
        "min": min(values),
        "mean": mean,
        "max": max(values),
        "std": math.sqrt(variance),
        "p50": _percentile(values, 50.0),
        "p95": _percentile(values, 95.0),
        "p99": _percentile(values, 99.0),
    }


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * max(0.0, min(100.0, percentile)) / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _higher_metric(
    name: str,
    value: int | float | None,
    pass_threshold: int | float,
    warn_threshold: int | float,
    detail: str,
) -> dict[str, Any]:
    if value is None or not math.isfinite(float(value)):
        status = "FAIL"
    elif value >= pass_threshold:
        status = "PASS"
    elif value >= warn_threshold:
        status = "WARN"
    else:
        status = "FAIL"
    return {
        "name": name,
        "status": status,
        "value": value,
        "pass_threshold": pass_threshold,
        "warn_threshold": warn_threshold,
        "comparison": ">=",
        "detail": detail,
    }


def _lower_metric(
    name: str,
    value: int | float | None,
    pass_threshold: int | float,
    warn_threshold: int | float,
    detail: str,
    *,
    missing_status: str = "FAIL",
) -> dict[str, Any]:
    if value is None or not math.isfinite(float(value)):
        status = missing_status
    elif value <= pass_threshold:
        status = "PASS"
    elif value <= warn_threshold:
        status = "WARN"
    else:
        status = "FAIL"
    return {
        "name": name,
        "status": status,
        "value": value,
        "pass_threshold": pass_threshold,
        "warn_threshold": warn_threshold,
        "comparison": "<=",
        "detail": detail,
    }
