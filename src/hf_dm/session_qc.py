"""Read-only quality-control evaluation for finalized HF-DM sessions."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from hf_dm.data.session_export import session_bundle_dir
from hf_dm.durable_io import fsync_best_effort

QcStatus = Literal["PASS", "WARN", "FAIL"]


@dataclass(frozen=True)
class QcThresholds:
    min_responded_rate: float = 0.50
    warn_responded_rate: float = 0.40
    min_matched_responded_rate: float = 0.55
    warn_matched_responded_rate: float = 0.45
    max_mean_response_time_s: float = 12.0
    warn_mean_response_time_s: float = 15.0
    max_choice_loop_stalls: int = 0
    warn_choice_loop_stalls: int = 2
    max_choice_loop_frame_s: float = 0.10
    warn_choice_loop_frame_s: float = 0.20

    def __post_init__(self) -> None:
        for name in (
            "min_responded_rate",
            "warn_responded_rate",
            "min_matched_responded_rate",
            "warn_matched_responded_rate",
        ):
            value = float(getattr(self, name))
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between zero and one")
        if self.warn_responded_rate > self.min_responded_rate:
            raise ValueError("warn_responded_rate cannot exceed min_responded_rate")
        if self.warn_matched_responded_rate > self.min_matched_responded_rate:
            raise ValueError("warn_matched_responded_rate cannot exceed min_matched_responded_rate")
        if self.max_mean_response_time_s <= 0:
            raise ValueError("max_mean_response_time_s must be greater than zero")
        if self.warn_mean_response_time_s < self.max_mean_response_time_s:
            raise ValueError("warn_mean_response_time_s cannot be below the pass threshold")
        if self.max_choice_loop_stalls < 0:
            raise ValueError("max_choice_loop_stalls must be non-negative")
        if self.warn_choice_loop_stalls < self.max_choice_loop_stalls:
            raise ValueError("warn_choice_loop_stalls cannot be below the pass threshold")
        if self.max_choice_loop_frame_s <= 0:
            raise ValueError("max_choice_loop_frame_s must be greater than zero")
        if self.warn_choice_loop_frame_s < self.max_choice_loop_frame_s:
            raise ValueError("warn_choice_loop_frame_s cannot be below the pass threshold")


@dataclass(frozen=True)
class QcMetric:
    name: str
    status: QcStatus
    value: float | int | None
    pass_threshold: float | int
    warn_threshold: float | int
    comparison: str
    detail: str
    numerator: int | None = None
    denominator: int | None = None


def evaluate_session_qc(
    path: Path | str,
    *,
    thresholds: QcThresholds | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Evaluate curriculum and acquisition metrics without modifying source data."""
    limits = thresholds or QcThresholds()
    source, session = load_session_for_qc(path)
    trials = [dict(trial) for trial in session.get("trials") or []]
    n_trials = len(trials)
    responded = [trial for trial in trials if bool(trial.get("responded"))]
    matched = [trial for trial in responded if bool(trial.get("matched"))]
    response_times = [
        float(trial["response_time_s"])
        for trial in responded
        if _finite_number(trial.get("response_time_s"))
    ]
    stall_values = [
        max(0, int(trial.get("choice_loop_stall_count") or 0))
        for trial in trials
        if trial.get("choice_loop_stall_count") is not None
    ]
    stall_count = sum(stall_values) if len(stall_values) == n_trials else None
    frame_times = [
        float(trial["choice_loop_max_frame_s"])
        for trial in trials
        if _finite_number(trial.get("choice_loop_max_frame_s"))
    ]
    events = [dict(event) for event in session.get("events") or []]

    metrics = [
        _higher_metric(
            "responded_rate",
            len(responded) / n_trials if n_trials else None,
            pass_threshold=limits.min_responded_rate,
            warn_threshold=limits.warn_responded_rate,
            detail="Fraction of completed trials with a response",
            numerator=len(responded),
            denominator=n_trials,
        ),
        _higher_metric(
            "matched_rate_responded",
            len(matched) / len(responded) if responded else None,
            pass_threshold=limits.min_matched_responded_rate,
            warn_threshold=limits.warn_matched_responded_rate,
            detail="Matched choices divided by responded trials",
            numerator=len(matched),
            denominator=len(responded),
        ),
        _lower_metric(
            "mean_response_time_s",
            (
                sum(response_times) / len(response_times)
                if response_times and len(response_times) == len(responded)
                else None
            ),
            pass_threshold=limits.max_mean_response_time_s,
            warn_threshold=limits.warn_mean_response_time_s,
            detail="Mean recorded response_time_s across responded trials",
            strict_pass=True,
        ),
        _lower_metric(
            "choice_loop_stall_count",
            stall_count,
            pass_threshold=limits.max_choice_loop_stalls,
            warn_threshold=limits.warn_choice_loop_stalls,
            detail="Total frames exceeding the configured choice-loop stall threshold",
        ),
        _lower_metric(
            "choice_loop_max_frame_s",
            max(frame_times) if len(frame_times) == n_trials and frame_times else None,
            pass_threshold=limits.max_choice_loop_frame_s,
            warn_threshold=limits.warn_choice_loop_frame_s,
            detail="Worst recorded choice-loop frame duration",
        ),
    ]
    heartbeat_qc = (
        (session.get("streamdio") or {}).get("timing_heartbeat_qc")
        if isinstance(session.get("streamdio"), dict)
        else None
    )
    if isinstance(heartbeat_qc, dict) and isinstance(heartbeat_qc.get("metrics"), list):
        for heartbeat_metric in heartbeat_qc["metrics"]:
            if not isinstance(heartbeat_metric, dict):
                continue
            status_value = str(heartbeat_metric.get("status"))
            if status_value not in {"PASS", "WARN", "FAIL"}:
                continue
            status = cast(QcStatus, status_value)
            metrics.append(
                QcMetric(
                    name=f"streamdio_{heartbeat_metric.get('name')}",
                    status=status,
                    value=(
                        heartbeat_metric.get("value")
                        if _finite_number(heartbeat_metric.get("value"))
                        else None
                    ),
                    pass_threshold=heartbeat_metric.get("pass_threshold", 0),
                    warn_threshold=heartbeat_metric.get("warn_threshold", 0),
                    comparison=str(heartbeat_metric.get("comparison") or "present"),
                    detail=str(heartbeat_metric.get("detail") or "StreamDIO timing heartbeat QC"),
                )
            )
    else:
        metrics.append(
            QcMetric(
                name="streamdio_timing_heartbeat_qc_present",
                status="FAIL",
                value=0,
                pass_threshold=1,
                warn_threshold=1,
                comparison="present",
                detail="Finalized session has no integrated StreamDIO timing heartbeat QC",
            )
        )
    metadata = session.get("metadata") or {}
    if isinstance(metadata, dict) and isinstance(metadata.get("reward_calibration"), dict):
        event_types = {str(event.get("event_type")) for event in events}
        water_logged = "subject_water_logged" in event_types
        water_error = "subject_water_log_error" in event_types
        metrics.append(
            QcMetric(
                name="subject_water_log",
                status="PASS" if water_logged and not water_error else "FAIL",
                value=1 if water_logged and not water_error else 0,
                pass_threshold=1,
                warn_threshold=1,
                comparison="present",
                detail=(
                    "Calibrated session water was appended to the subject log"
                    if water_logged and not water_error
                    else "Calibrated session has no successful subject-water log event"
                ),
            )
        )
    overall = _overall_status(metrics)
    manifest = _find_manifest_for_session(path, source, session)
    output_dir = manifest.parent if manifest is not None else source.parent
    output_path = output_dir / "session.qc.json"
    report = {
        "schema_version": 1,
        "kind": "hf_dm_session_qc",
        "generated_wall_s": time.time(),
        "source_session": str(source),
        "manifest": str(manifest) if manifest is not None else None,
        "overall_status": overall,
        "counts": {
            "trials": n_trials,
            "responded": len(responded),
            "matched_responded": len(matched),
        },
        "thresholds": asdict(limits),
        "metrics": [asdict(metric) for metric in metrics],
        "output": str(output_path),
    }
    if write:
        _atomic_write_json(output_path, report)
    return report


def load_session_for_qc(path: Path | str) -> tuple[Path, dict[str, Any]]:
    candidate = Path(path)
    if candidate.is_dir() or candidate.name == "manifest.json":
        manifest = _manifest_from_candidate(candidate)
        if manifest is not None:
            sidecar = _session_json_from_manifest(manifest)
            if sidecar is not None:
                return sidecar, json.loads(sidecar.read_text(encoding="utf-8"))
        if candidate.is_dir():
            sidecars = [
                item
                for item in sorted(candidate.glob("*.json"))
                if item.name not in {"manifest.json", "session.qc.json"}
                and not item.name.endswith("_export_error.json")
            ]
            for sidecar in sidecars:
                payload = json.loads(sidecar.read_text(encoding="utf-8"))
                if isinstance(payload.get("trials"), list):
                    return sidecar, payload
        raise FileNotFoundError(f"No finalized session data found for {candidate}")
    if candidate.suffix.lower() == ".json":
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        if isinstance(payload.get("trials"), list):
            return candidate, payload
        manifest = _manifest_from_candidate(candidate)
        if manifest is not None:
            sidecar = _session_json_from_manifest(manifest)
            if sidecar is not None:
                return sidecar, json.loads(sidecar.read_text(encoding="utf-8"))
        raise ValueError(f"JSON file is not a session sidecar or manifest: {candidate}")
    if candidate.suffix.lower() == ".npz":
        sidecar = candidate.with_suffix(".json")
        if sidecar.exists():
            return sidecar, json.loads(sidecar.read_text(encoding="utf-8"))
        return candidate, _load_npz_session(candidate)
    raise ValueError(f"Unsupported session QC input: {candidate}")


def format_qc_table(report: dict[str, Any]) -> str:
    rows = [
        "STATUS  METRIC                         VALUE",
        "------  -----------------------------  --------",
    ]
    for metric in report.get("metrics") or []:
        value = metric.get("value")
        value_text = "n/a" if value is None else f"{value:.4g}"
        rows.append(f"{metric['status']:<6}  {metric['name']:<29}  {value_text:>8}")
    rows.append(f"Overall: {report.get('overall_status', 'FAIL')}")
    rows.append(f"Report: {report.get('output')}")
    return "\n".join(rows)


def _higher_metric(
    name: str,
    value: float | None,
    *,
    pass_threshold: float,
    warn_threshold: float,
    detail: str,
    numerator: int | None = None,
    denominator: int | None = None,
) -> QcMetric:
    if value is None or not math.isfinite(value):
        status: QcStatus = "FAIL"
    elif value >= pass_threshold:
        status = "PASS"
    elif value >= warn_threshold:
        status = "WARN"
    else:
        status = "FAIL"
    return QcMetric(
        name=name,
        status=status,
        value=value,
        pass_threshold=pass_threshold,
        warn_threshold=warn_threshold,
        comparison=">=",
        detail=detail,
        numerator=numerator,
        denominator=denominator,
    )


def _lower_metric(
    name: str,
    value: float | int | None,
    *,
    pass_threshold: float | int,
    warn_threshold: float | int,
    detail: str,
    strict_pass: bool = False,
    missing_status: QcStatus = "FAIL",
) -> QcMetric:
    if value is None or not _finite_number(value):
        status: QcStatus = missing_status
    elif (value < pass_threshold) if strict_pass else (value <= pass_threshold):
        status = "PASS"
    elif value <= warn_threshold:
        status = "WARN"
    else:
        status = "FAIL"
    return QcMetric(
        name=name,
        status=status,
        value=value,
        pass_threshold=pass_threshold,
        warn_threshold=warn_threshold,
        comparison="<" if strict_pass else "<=",
        detail=detail,
    )


def _overall_status(metrics: list[QcMetric]) -> QcStatus:
    statuses = {metric.status for metric in metrics}
    if "FAIL" in statuses:
        return "FAIL"
    if "WARN" in statuses:
        return "WARN"
    return "PASS"


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(
        float(value)
    )


def _load_npz_session(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        if "trials_json" in data.files:
            trials = [json.loads(str(item)) for item in data["trials_json"]]
        elif "trials" in data.files and data["trials"].dtype.names:
            trials = [
                {
                    name: row[name].item() if hasattr(row[name], "item") else row[name]
                    for name in data["trials"].dtype.names or ()
                }
                for row in data["trials"]
            ]
        else:
            trials = []
        summary: dict[str, Any] = {}
        if "summary_json" in data.files and len(data["summary_json"]):
            summary = json.loads(str(data["summary_json"][0]))
        heartbeat_qc: dict[str, Any] | None = None
        if "timing_heartbeat_qc_json" in data.files and len(data["timing_heartbeat_qc_json"]):
            heartbeat_qc = json.loads(str(data["timing_heartbeat_qc_json"][0]))
    return {
        "trials": trials,
        "summary": summary,
        "metadata": {},
        "streamdio": {"timing_heartbeat_qc": heartbeat_qc},
    }


def _manifest_from_candidate(path: Path) -> Path | None:
    if path.name == "manifest.json" and path.exists():
        return path
    if path.is_dir():
        direct = path / "manifest.json"
        if direct.exists():
            return direct
        nested = sorted(path.glob("*_bundle/manifest.json"))
        return nested[0] if nested else None
    return None


def _session_json_from_manifest(manifest_path: Path) -> Path | None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in manifest.get("artifacts") or []:
        rel = item.get("path")
        if not rel or Path(str(rel)).suffix.lower() != ".json":
            continue
        candidate = (manifest_path.parent / str(rel)).resolve()
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload.get("trials"), list) and isinstance(payload.get("summary"), dict):
            return candidate
    return None


def _find_manifest_for_session(
    requested: Path | str,
    source: Path,
    session: dict[str, Any],
) -> Path | None:
    direct = _manifest_from_candidate(Path(requested))
    if direct is not None:
        return direct
    beside = source.parent / "manifest.json"
    if beside.exists():
        return beside
    layout = str(
        ((session.get("metadata") or {}).get("config") or {})
        .get("session_export", {})
        .get("layout", "flat")
    )
    candidate = session_bundle_dir(source.with_suffix(""), layout=layout) / "manifest.json"
    return candidate if candidate.exists() else None


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            fsync_best_effort(handle, path=temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
