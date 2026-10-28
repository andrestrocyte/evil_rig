"""Standalone reward-valve calibration and duration-to-volume interpolation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from bisect import bisect_left
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from hf_dm.config import SessionConfig
from hf_dm.durable_io import fsync_best_effort, fsync_directory_best_effort
from hf_dm.streamdio.channel_map import session_channel_map
from hf_dm.streamdio.client import StreamDIOClient
from hf_dm.streamdio.outputs import StreamDIOOutputs


@dataclass(frozen=True)
class ValveMeasurement:
    open_time_s: float
    total_mass_mg: float
    repetitions: int


@dataclass(frozen=True)
class CalibrationPoint:
    open_time_s: float
    volume_ul: float
    repetitions: int
    total_mass_mg: float


@dataclass(frozen=True)
class RewardCalibration:
    source_path: Path
    points: tuple[CalibrationPoint, ...]
    sha256: str
    fit_slope_ul_per_s: float | None = None
    fit_intercept_ul: float | None = None
    fit_r_squared: float | None = None
    fit_max_abs_residual_ul: float | None = None
    fit_rmse_ul: float | None = None

    @classmethod
    def load(cls, path: Path | str) -> "RewardCalibration":
        source = Path(path).expanduser().resolve()
        payload = json.loads(source.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1 or payload.get("kind") != "hf_dm_valve_calibration":
            raise ValueError(f"Unsupported valve calibration format: {source}")
        raw_points = payload.get("points")
        if not isinstance(raw_points, list) or len(raw_points) < 2:
            raise ValueError("Valve calibration requires at least two points")
        points = tuple(
            sorted(
                (
                    CalibrationPoint(
                        open_time_s=float(point["open_time_s"]),
                        volume_ul=float(point["volume_ul"]),
                        repetitions=int(point.get("repetitions", 0)),
                        total_mass_mg=float(point.get("total_mass_mg", 0.0)),
                    )
                    for point in raw_points
                ),
                key=lambda point: point.open_time_s,
            )
        )
        _validate_points(points)
        fit = payload.get("linear_fit") or {}
        return cls(
            source_path=source,
            points=points,
            sha256=_sha256(source),
            fit_slope_ul_per_s=_optional_float(fit.get("slope_ul_per_s")),
            fit_intercept_ul=_optional_float(fit.get("intercept_ul")),
            fit_r_squared=_optional_float(fit.get("r_squared")),
            fit_max_abs_residual_ul=_optional_float(fit.get("max_abs_residual_ul")),
            fit_rmse_ul=_optional_float(fit.get("rmse_ul")),
        )

    @property
    def min_open_time_s(self) -> float:
        return self.points[0].open_time_s

    @property
    def max_open_time_s(self) -> float:
        return self.points[-1].open_time_s

    def volume_for_duration(self, open_time_s: float) -> float:
        duration = float(open_time_s)
        if not math.isfinite(duration):
            raise ValueError("Reward duration must be finite")
        if duration < self.min_open_time_s or duration > self.max_open_time_s:
            raise ValueError(
                f"Reward duration {duration:g}s is outside calibration range "
                f"[{self.min_open_time_s:g}, {self.max_open_time_s:g}]s"
            )
        durations = [point.open_time_s for point in self.points]
        index = bisect_left(durations, duration)
        if index < len(self.points) and math.isclose(
            self.points[index].open_time_s,
            duration,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            return self.points[index].volume_ul
        low = self.points[index - 1]
        high = self.points[index]
        fraction = (duration - low.open_time_s) / (high.open_time_s - low.open_time_s)
        return low.volume_ul + fraction * (high.volume_ul - low.volume_ul)

    def metadata(self) -> dict[str, Any]:
        return {
            "path": str(self.source_path),
            "sha256": self.sha256,
            "min_open_time_s": self.min_open_time_s,
            "max_open_time_s": self.max_open_time_s,
            "points": [asdict(point) for point in self.points],
            "linear_fit": {
                "slope_ul_per_s": self.fit_slope_ul_per_s,
                "intercept_ul": self.fit_intercept_ul,
                "r_squared": self.fit_r_squared,
                "max_abs_residual_ul": self.fit_max_abs_residual_ul,
                "rmse_ul": self.fit_rmse_ul,
            },
        }


def fit_valve_calibration(
    measurements: Sequence[ValveMeasurement],
    *,
    water_density_mg_per_ul: float = 1.0,
) -> dict[str, Any]:
    if not math.isfinite(water_density_mg_per_ul) or water_density_mg_per_ul <= 0:
        raise ValueError("water_density_mg_per_ul must be finite and greater than zero")
    ordered = sorted(measurements, key=lambda item: item.open_time_s)
    if len(ordered) < 2:
        raise ValueError("At least two valve durations are required")
    points: list[CalibrationPoint] = []
    seen: set[float] = set()
    for measurement in ordered:
        if not math.isfinite(measurement.open_time_s) or measurement.open_time_s <= 0:
            raise ValueError("Valve open times must be finite and greater than zero")
        if measurement.open_time_s in seen:
            raise ValueError("Valve open times must be unique")
        if measurement.repetitions <= 0:
            raise ValueError("Valve repetitions must be greater than zero")
        if not math.isfinite(measurement.total_mass_mg) or measurement.total_mass_mg < 0:
            raise ValueError("Measured masses must be finite and non-negative")
        seen.add(measurement.open_time_s)
        points.append(
            CalibrationPoint(
                open_time_s=measurement.open_time_s,
                volume_ul=(measurement.total_mass_mg / water_density_mg_per_ul)
                / measurement.repetitions,
                repetitions=measurement.repetitions,
                total_mass_mg=measurement.total_mass_mg,
            )
        )
    _validate_points(tuple(points))
    x = np.asarray([point.open_time_s for point in points], dtype=float)
    y = np.asarray([point.volume_ul for point in points], dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    residual_sum = float(np.sum((y - predicted) ** 2))
    max_abs_residual_ul = float(np.max(np.abs(y - predicted)))
    rmse_ul = float(np.sqrt(np.mean((y - predicted) ** 2)))
    total_sum = float(np.sum((y - np.mean(y)) ** 2))
    if total_sum <= np.finfo(float).eps:
        r_squared = 1.0 if residual_sum <= np.finfo(float).eps else 0.0
    else:
        r_squared = 1.0 - residual_sum / total_sum
    return {
        "schema_version": 1,
        "kind": "hf_dm_valve_calibration",
        "created_wall_s": time.time(),
        "water_density_mg_per_ul": water_density_mg_per_ul,
        "points": [asdict(point) for point in points],
        "linear_fit": {
            "slope_ul_per_s": float(slope),
            "intercept_ul": float(intercept),
            "r_squared": float(r_squared),
            "max_abs_residual_ul": max_abs_residual_ul,
            "rmse_ul": rmse_ul,
        },
    }


def save_valve_calibration(
    payload: dict[str, Any],
    path: Path | str,
) -> RewardCalibration:
    output = Path(path)
    _atomic_write_json(output, payload)
    return RewardCalibration.load(output)


def collect_valve_measurements(
    durations_s: Sequence[float],
    *,
    repetitions: int,
    pulse_pump: Callable[[float], Any],
    read_mass_mg: Callable[[float, int], float],
    before_duration: Callable[[float, int], Any] | None = None,
    inter_pulse_s: float = 0.10,
) -> list[ValveMeasurement]:
    if repetitions <= 0:
        raise ValueError("repetitions must be greater than zero")
    measurements: list[ValveMeasurement] = []
    for duration in durations_s:
        duration_s = float(duration)
        if not math.isfinite(duration_s) or duration_s <= 0:
            raise ValueError("calibration durations must be finite and greater than zero")
        if before_duration is not None:
            before_duration(duration_s, repetitions)
        for pulse_index in range(repetitions):
            pulse_pump(duration_s)
            if inter_pulse_s > 0 and pulse_index + 1 < repetitions:
                time.sleep(inter_pulse_s)
        measurements.append(
            ValveMeasurement(
                open_time_s=duration_s,
                total_mass_mg=float(read_mass_mg(duration_s, repetitions)),
                repetitions=repetitions,
            )
        )
    return measurements


def run_guided_valve_calibration(
    config: SessionConfig,
    *,
    durations_s: Sequence[float],
    repetitions: int,
    output: Path | str | None = None,
    water_density_mg_per_ul: float = 1.0,
    inter_pulse_s: float = 0.10,
    input_fn: Callable[[str], str] = input,
    overwrite: bool = False,
) -> dict[str, Any]:
    normalized_durations = tuple(float(duration) for duration in durations_s)
    if len(normalized_durations) < 2 or len(set(normalized_durations)) != len(
        normalized_durations
    ):
        raise ValueError("At least two unique valve durations are required")
    if any(not math.isfinite(duration) or duration <= 0 for duration in normalized_durations):
        raise ValueError("Valve durations must be finite and greater than zero")
    if repetitions <= 0:
        raise ValueError("repetitions must be greater than zero")
    if not math.isfinite(water_density_mg_per_ul) or water_density_mg_per_ul <= 0:
        raise ValueError("water_density_mg_per_ul must be finite and greater than zero")
    if not math.isfinite(inter_pulse_s) or inter_pulse_s < 0:
        raise ValueError("inter_pulse_s must be finite and non-negative")
    default_name = f"valve_{datetime.now():%Y%m%d_%H%M%S}.json"
    output_path = Path(output or Path("calibrations") / default_name).expanduser().resolve()
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Calibration output already exists: {output_path}. Use --overwrite to replace it."
        )
    dio = StreamDIOClient(
        config_port=config.ports.dio_config_port,
        stream_port=config.ports.dio_stream_port,
        baudrate=config.ports.baudrate,
    )
    relay: StreamDIOOutputs | None = None
    try:
        dio.open()
        dio.apply_map(session_channel_map(config))
        dio.start_streaming()
        relay = StreamDIOOutputs(
            dio,
            cap_touch_power_channel=config.licks.power_output_channel,
            speaker_gate_active_high=config.speaker.gate_active_high,
        )
        relay.pump_off()

        def read_mass(duration_s: float, pulse_count: int) -> float:
            prompt = (
                f"Enter mass gain in mg after {pulse_count} openings at "
                f"{duration_s * 1000:g} ms: "
            )
            return float(input_fn(prompt).strip())

        def prepare_collection(duration_s: float, pulse_count: int) -> None:
            input_fn(
                f"Tare the vessel for {pulse_count} openings at "
                f"{duration_s * 1000:g} ms, then press Enter to actuate the pump: "
            )

        measurements = collect_valve_measurements(
            normalized_durations,
            repetitions=repetitions,
            pulse_pump=relay.pulse_pump,
            read_mass_mg=read_mass,
            before_duration=prepare_collection,
            inter_pulse_s=inter_pulse_s,
        )
        payload = fit_valve_calibration(
            measurements,
            water_density_mg_per_ul=water_density_mg_per_ul,
        )
        payload["hardware"] = {
            "config_port": dio.config_port,
            "stream_port": dio.stream_port,
            "module_info": _json_safe_dict(dio.module_info),
        }
        loaded = save_valve_calibration(payload, output_path)
        return {
            "output": str(output_path.resolve()),
            "sha256": loaded.sha256,
            "points": [asdict(point) for point in loaded.points],
            "linear_fit": payload["linear_fit"],
        }
    finally:
        if relay is not None:
            try:
                relay.pump_off()
            except Exception:
                pass
        dio.close()


def resolve_reward_calibration_path(config: SessionConfig) -> Path | None:
    value = config.reward.calibration_path
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    source_config = config.session_export.source_config_path
    if source_config:
        return Path(source_config).expanduser().resolve().parent / path
    return path.resolve()


def _validate_points(points: tuple[CalibrationPoint, ...]) -> None:
    prior_duration = -math.inf
    prior_volume = -math.inf
    for point in points:
        if not math.isfinite(point.open_time_s) or point.open_time_s <= prior_duration:
            raise ValueError("Calibration open times must be finite, unique, and increasing")
        if not math.isfinite(point.volume_ul) or point.volume_ul < 0:
            raise ValueError("Calibration volumes must be finite and non-negative")
        if point.volume_ul < prior_volume:
            raise ValueError("Calibration volume must not decrease as open time increases")
        prior_duration = point.open_time_s
        prior_volume = point.volume_ul


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _json_safe_dict(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): item.item() if hasattr(item, "item") else item
        for key, item in value.items()
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            fsync_best_effort(handle, path=temporary)
        os.replace(temporary, path)
        fsync_directory_best_effort(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
