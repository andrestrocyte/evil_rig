"""Cold-path session bundle, manifest, and CSV export helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import time
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import yaml

from hf_dm.hardware.bringup_constants import FORCE_ANALOG_CHANNEL, FRAME2TTL_INPUT_CHANNEL
from hf_dm.durable_io import fsync_best_effort, fsync_directory_best_effort
from hf_dm.timing_heartbeat import (
    TIMING_HEARTBEAT_EVENT_TYPE,
    analyze_timing_heartbeats,
    timing_heartbeat_qc_summary,
)


SEMANTIC_TIMELINE_EVENTS = {
    "session_start",
    "session_end",
    "trial_start",
    "trial_end",
    "centering_trial_start",
    "preflight_start",
    "preflight_end",
    "stimulus_on",
    "stimulus_off",
    "choice_detected",
    "choice_timeout",
    "feedback_on",
    "force_on",
    "force_off",
    "force_skipped",
    "pump_start",
    "pump_end",
    "pump_omitted",
    "output_actual",
    "lick",
    "white_noise_start",
    "white_noise_end",
    "bpod_state_marker",
    "frame2ttl_edge",
    TIMING_HEARTBEAT_EVENT_TYPE,
    "frame2ttl_barcode_start",
    "frame2ttl_barcode_end",
    "frame2ttl_barcode_decode",
    "frame2ttl_diagnostic_start",
    "frame2ttl_diagnostic_end",
    "frame2ttl_barcode_diagnostic_start",
    "frame2ttl_barcode_diagnostic_case_start",
    "frame2ttl_barcode_diagnostic_case_end",
    "frame2ttl_barcode_diagnostic_end",
    "bpod_opcode_diagnostic_start",
    "bpod_opcode_diagnostic_end",
    "video_recording_started",
    "video_recording_stopped",
}

# Diagnostic events can legitimately contain large nested JSON payloads (for
# example, all cases from a repeated Frame2TTL barcode qualification).  Python's
# csv module defaults to a roughly 128 KiB field limit, which is smaller than a
# valid recorder-generated detail_json field.  Keep validation bounded while
# allowing the largest payloads this application emits.
_CSV_VALIDATION_FIELD_LIMIT = 64 * 1024 * 1024


@contextmanager
def _atomic_text_handle(path: Path, *, newline: str | None = None):
    """Yield a same-directory temporary text file and atomically publish it."""

    target = Path(path)
    temporary = target.with_name(
        f".{target.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8", newline=newline) as handle:
            yield handle
            handle.flush()
            fsync_best_effort(handle, path=temporary)
        os.replace(temporary, target)
        fsync_directory_best_effort(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_text(path: Path, text: str) -> None:
    with _atomic_text_handle(path, newline="\n") as handle:
        handle.write(text)


STREAMDIO_COLUMNS = [
    "event_type",
    "channel",
    "value",
    "module_timestamp_us",
    "relative_time_s",
    "host_perf_counter_s",
]

TIMING_HEARTBEAT_COLUMNS = [
    "sequence",
    "teensy_timestamp_us",
    "driver_relative_time_s",
    "teensy_elapsed_s",
    "teensy_interval_s",
    "pc_perf_counter_s",
    "pc_wall_time_s",
    "pc_elapsed_s",
    "pc_interval_s",
    "pc_minus_teensy_interval_s",
    "pc_minus_teensy_elapsed_s",
]

TIMELINE_COLUMNS = [
    "session_mono_s",
    "wall_time_s",
    "module_timestamp_us",
    "source",
    "event_type",
    "trial_index",
    "detail_json",
]


def resolve_session_output(
    results_dir: Path,
    *,
    subject_id: str,
    phase: str,
    layout: str = "flat",
    now: datetime | None = None,
) -> tuple[Path, str | None]:
    """Return the recorder output directory and optional fixed base name."""
    if str(layout or "flat").lower() != "mm_hierarchy":
        return Path(results_dir), None
    stamp = now or datetime.now()
    day_dir = Path(results_dir) / str(phase) / str(subject_id) / stamp.strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    run_id = 1
    while True:
        candidate = day_dir / f"{run_id:03d}"
        try:
            candidate.mkdir()
            return candidate, "session"
        except FileExistsError:
            run_id += 1


def export_existing_session(path: Path | str) -> dict[str, Any]:
    """Regenerate cold exports for an already finalized JSON/NPZ session."""
    session_path = Path(path)
    if session_path.is_dir():
        candidates: list[Path] = []
        for candidate in sorted(session_path.glob("*.json")):
            if candidate.name == "manifest.json":
                continue
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if (
                isinstance(payload, dict)
                and isinstance(payload.get("summary"), dict)
                and isinstance(payload.get("trials"), list)
                and isinstance(payload.get("events"), list)
            ):
                candidates.append(candidate)
        if not candidates:
            raise FileNotFoundError(f"No session JSON found in {session_path}")
        if len(candidates) > 1:
            names = ", ".join(candidate.name for candidate in candidates)
            raise RuntimeError(f"Ambiguous session JSON files in {session_path}: {names}")
        json_path = candidates[0]
    elif session_path.suffix == ".npz":
        json_path = session_path.with_suffix(".json")
    else:
        json_path = session_path
    if not json_path.exists():
        raise FileNotFoundError(json_path)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    npz_name = data.get("npz_file")
    npz_path = json_path.with_name(npz_name) if npz_name else json_path.with_suffix(".npz")
    streamdio_events = _load_npz_streamdio(npz_path) if npz_path.exists() else []
    bundle = make_session_bundle(
        base_path=json_path.with_suffix(""),
        summary=data.get("summary") or {},
        metadata=data.get("metadata") or {},
        trials=data.get("trials") or [],
        events=data.get("events") or [],
        debug_events=data.get("debug_events_recent") or [],
        streamdio_events=streamdio_events,
        bpod_trials=data.get("bpod_trials") or [],
        fsm_definitions=data.get("fsm_definitions") or [],
        npz_path=npz_path if npz_path.exists() else None,
        json_path=json_path,
        options=_extract_export_options(data.get("metadata") or {}),
        recorder_stats={"offline_export": True},
    )
    return bundle


def validate_session(path: Path | str) -> dict[str, Any]:
    """Validate a session manifest and the files it points to."""
    root = Path(path)
    manifest_path = _find_manifest(root)
    if manifest_path is None:
        raise FileNotFoundError(f"No manifest.json found for {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    checked = 0
    npz_checked = 0
    json_checked = 0
    jsonl_checked = 0
    csv_checked = 0
    for item in manifest.get("artifacts", []):
        rel = item.get("path")
        if not rel:
            continue
        artifact = (manifest_path.parent / rel).resolve()
        if not artifact.exists():
            errors.append(f"missing: {rel}")
            continue
        checked += 1
        expected_bytes = item.get("bytes")
        if expected_bytes is not None:
            actual_bytes = artifact.stat().st_size
            try:
                declared_bytes = int(expected_bytes)
            except (TypeError, ValueError):
                errors.append(f"invalid byte count: {rel}: {expected_bytes}")
            else:
                if actual_bytes != declared_bytes:
                    errors.append(
                        f"byte size mismatch: {rel}: {actual_bytes} != {declared_bytes}"
                    )
        expected = item.get("sha256")
        if expected and sha256_file(artifact) != expected:
            errors.append(f"sha256 mismatch: {rel}")
        if artifact.suffix.lower() == ".npz":
            npz_checked += 1
            errors.extend(_validate_npz_artifact(artifact, str(rel), manifest))
        elif artifact.suffix.lower() == ".json":
            json_checked += 1
            errors.extend(_validate_json_artifact(artifact, str(rel), manifest))
        elif artifact.suffix.lower() == ".jsonl":
            jsonl_checked += 1
            errors.extend(_validate_jsonl_artifact(artifact, str(rel), manifest))
        elif artifact.suffix.lower() == ".csv":
            csv_checked += 1
            errors.extend(_validate_csv_artifact(artifact, str(rel), manifest))
    errors.extend(_validate_config_provenance(manifest, manifest_path.parent))
    errors.extend(_validate_manifest_flags(manifest, manifest_path.parent))
    errors.extend(_validate_manifest_counts(manifest, manifest_path.parent))
    errors.extend(_validate_timing_heartbeat_consistency(manifest, manifest_path.parent))
    errors.extend(_validate_continuous_stream_index(manifest, manifest_path.parent))
    errors.extend(_validate_recorder_stats(manifest))
    return {
        "manifest": str(manifest_path),
        "checked_artifacts": checked,
        "npz_files_checked": npz_checked,
        "json_files_checked": json_checked,
        "jsonl_files_checked": jsonl_checked,
        "csv_files_checked": csv_checked,
        "valid": not errors,
        "errors": errors,
    }


def make_session_bundle(
    *,
    base_path: Path,
    summary: Any,
    metadata: dict[str, Any],
    trials: list[Any],
    events: list[Any],
    streamdio_events: list[Any],
    bpod_trials: list[dict[str, Any]],
    fsm_definitions: dict[str, str] | list[dict[str, Any]],
    npz_path: Path | None,
    json_path: Path,
    options: dict[str, Any] | None = None,
    recorder_stats: dict[str, Any] | None = None,
    debug_events: list[Any] | None = None,
) -> dict[str, Any]:
    options = options or {}
    layout = str(options.get("layout", "flat")).lower()
    bundle_dir = session_bundle_dir(Path(base_path), layout=layout)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    exports_dir = bundle_dir / "exports"
    bpod_dir = bundle_dir / "bpod_exports"
    bpod_data_dir = bundle_dir / "bpod_data"
    continuous_dir = bundle_dir / "continuous_streams"
    videos_dir = bundle_dir / "videos"
    exports_dir.mkdir(exist_ok=True)
    bpod_dir.mkdir(exist_ok=True)
    bpod_data_dir.mkdir(exist_ok=True)
    continuous_dir.mkdir(exist_ok=True)
    videos_dir.mkdir(exist_ok=True)

    started = time.perf_counter()
    summary_dict = _to_dict(summary)
    trial_dicts = [_to_dict(trial) for trial in trials]
    event_dicts = [_event_to_dict(event) for event in events]
    debug_event_dicts = [_event_to_dict(event) for event in (debug_events or [])]
    streamdio_rows = [_streamdio_to_dict(row) for row in streamdio_events]

    written: list[Path] = []
    timing_heartbeat_report = analyze_timing_heartbeats(
        streamdio_rows,
        session_events=event_dicts,
    )
    timing_heartbeat_samples = list(timing_heartbeat_report.get("samples") or [])
    timing_heartbeat_summary = timing_heartbeat_qc_summary(timing_heartbeat_report)
    timing_heartbeat_csv = exports_dir / "streamdio_timing_heartbeat.csv"
    timing_heartbeat_qc_path = exports_dir / "streamdio_timing_qc.json"
    timing_heartbeat_summary.update(
        {
            "timestamps_artifact": "exports/streamdio_timing_heartbeat.csv",
            "qc_artifact": "exports/streamdio_timing_qc.json",
        }
    )
    _write_timing_heartbeat_csv(timing_heartbeat_csv, timing_heartbeat_samples)
    _atomic_write_text(
        timing_heartbeat_qc_path,
        json.dumps(timing_heartbeat_summary, indent=2),
    )
    written.extend([timing_heartbeat_csv, timing_heartbeat_qc_path])
    config_snapshot = None
    if bool(options.get("config_snapshot", True)):
        config_snapshot = bundle_dir / "config.snapshot.yaml"
        _write_config_snapshot(config_snapshot, metadata)
        written.append(config_snapshot)

    if bool(options.get("export_csv_on_finalize", True)):
        written.extend(
            [
                _write_dict_csv(exports_dir / "trials_summary.csv", trial_dicts),
                _write_dict_csv(exports_dir / "events.csv", event_dicts),
                _write_streamdio_csv(exports_dir / "streamdio_events.csv", streamdio_rows),
                _write_streamdio_subset_csv(
                    exports_dir / "force_trace.csv",
                    streamdio_rows,
                    lambda row: row["event_type"] == "analog_input"
                    and int(row["channel"]) == _force_channel(metadata),
                ),
                _write_streamdio_subset_csv(
                    exports_dir / "camera_ttl.csv",
                    streamdio_rows,
                    lambda row: row["event_type"] == "camera",
                ),
                _write_streamdio_subset_csv(
                    exports_dir / "frame2ttl_edges.csv",
                    streamdio_rows,
                    lambda row: row["event_type"] == "dio_input"
                    and int(row["channel"]) == _frame2ttl_channel(metadata),
                ),
                _write_licks_csv(exports_dir / "licks.csv", event_dicts),
                _write_bpod_flat_csv(bpod_dir / "bpod_trials_flat.csv", bpod_trials),
            ]
        )
        module_info = _streamdio_module_info(metadata, event_dicts)
        if module_info:
            path = exports_dir / "streamdio_module_info.json"
            _atomic_write_text(path, json.dumps(module_info, indent=2))
            written.append(path)
        stream_summaries = metadata.get("stream_summaries")
        if isinstance(stream_summaries, dict) and stream_summaries:
            path = exports_dir / "stream_summaries.json"
            _atomic_write_text(path, json.dumps(stream_summaries, indent=2))
            written.append(path)
        soundcard_status = _soundcard_status(event_dicts)
        if soundcard_status:
            path = exports_dir / "soundcard_status.json"
            _atomic_write_text(path, json.dumps(soundcard_status, indent=2))
            written.append(path)
        if debug_event_dicts:
            path = exports_dir / "debug_events_recent.json"
            _atomic_write_text(path, json.dumps(debug_event_dicts, indent=2))
            written.append(path)

    bpod_data_path = _write_bpod_data_json(
        bpod_data_dir / "bpod_trials.json",
        bpod_trials=bpod_trials,
        fsm_definitions=fsm_definitions,
    )
    if bpod_data_path is not None:
        written.append(bpod_data_path)

    for existing_export in (
        exports_dir / "monitor_telemetry.jsonl",
        exports_dir / "rig_preflight_report.json",
    ):
        if existing_export.exists():
            written.append(existing_export)

    if bool(options.get("export_timeline", True)):
        timeline = _timeline_rows(event_dicts, streamdio_rows, bundle_dir=bundle_dir)
        timeline_path = exports_dir / "synchronized_timeline.csv"
        written.append(_write_dict_csv(timeline_path, timeline))
        anchors_path = exports_dir / "alignment_anchors.json"
        _atomic_write_text(
            anchors_path,
            json.dumps(
                _alignment_anchors(event_dicts, streamdio_rows, bundle_dir=bundle_dir),
                indent=2,
            ),
        )
        written.append(anchors_path)

    video_pointer = _write_video_pointer(
        bundle_dir=bundle_dir,
        videos_dir=videos_dir,
        event_dicts=event_dicts,
    )
    if video_pointer is not None:
        written.append(video_pointer)

    preview_export = _maybe_export_preview_video(
        bundle_dir=bundle_dir,
        exports_dir=exports_dir,
        metadata=metadata,
        event_dicts=event_dicts,
        options=options,
    )
    if preview_export is not None:
        written.append(preview_export)

    manifest_path = bundle_dir / "manifest.json"
    chunk_paths = sorted(continuous_dir.glob("streamdio_chunk_*.npz"))
    journal_path = Path(base_path).with_suffix(".journal.jsonl")
    artifact_paths = [
        json_path,
        *([journal_path] if journal_path.exists() else []),
        *written,
        *_video_artifact_paths(bundle_dir, event_dicts),
        *chunk_paths,
    ]
    if npz_path is not None and npz_path.exists():
        artifact_paths.insert(0, npz_path)
    manifest = _manifest(
        manifest_path=manifest_path,
        base_path=base_path,
        summary=summary_dict,
        metadata=metadata,
        trials=trial_dicts,
        event_dicts=event_dicts,
        streamdio_rows=streamdio_rows,
        bpod_trials=bpod_trials,
        fsm_definitions=fsm_definitions,
        artifacts=artifact_paths,
        options=options,
        recorder_stats={
            **(recorder_stats or {}),
            "bundle_write_s": time.perf_counter() - started,
            "bundle_dir": str(bundle_dir),
            "bundle_artifact_count": _existing_artifact_count(artifact_paths),
            "bundle_artifact_bytes": _artifact_bytes(artifact_paths),
        },
        timing_heartbeat_qc=timing_heartbeat_summary,
    )
    if bool(options.get("manifest", True)):
        _atomic_write_text(manifest_path, json.dumps(manifest, indent=2))
    alyx_path = _maybe_write_alyx_registration(
        exports_dir=exports_dir,
        manifest_path=manifest_path,
        bundle_dir=bundle_dir,
        metadata=metadata,
        summary=summary_dict,
        artifacts=artifact_paths,
    )
    if alyx_path is not None:
        artifact_paths.append(alyx_path)
        manifest = _manifest(
            manifest_path=manifest_path,
            base_path=base_path,
            summary=summary_dict,
            metadata=metadata,
            trials=trial_dicts,
            event_dicts=event_dicts,
            streamdio_rows=streamdio_rows,
            bpod_trials=bpod_trials,
            fsm_definitions=fsm_definitions,
            artifacts=artifact_paths,
            options=options,
            recorder_stats={
                **(recorder_stats or {}),
                "bundle_write_s": time.perf_counter() - started,
                "bundle_dir": str(bundle_dir),
                "bundle_artifact_count": _existing_artifact_count(artifact_paths),
                "bundle_artifact_bytes": _artifact_bytes(artifact_paths),
            },
            timing_heartbeat_qc=timing_heartbeat_summary,
        )
        if bool(options.get("manifest", True)):
            _atomic_write_text(manifest_path, json.dumps(manifest, indent=2))
    return {
        "bundle_dir": str(bundle_dir),
        "manifest": str(manifest_path) if bool(options.get("manifest", True)) else None,
        "exports_dir": str(exports_dir),
        "artifacts": [str(path) for path in artifact_paths],
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _existing_artifact_count(paths: list[Path]) -> int:
    return sum(1 for path in paths if path.exists())


def _artifact_bytes(paths: list[Path]) -> int:
    return sum(path.stat().st_size for path in paths if path.exists())


def session_bundle_dir(base_path: Path, *, layout: str = "flat") -> Path:
    if str(layout or "flat").lower() == "mm_hierarchy" and base_path.name == "session":
        return base_path.parent
    return base_path.with_name(f"{base_path.name}_bundle")


def main_export(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Regenerate exports for a finalized HF-DM session.")
    parser.add_argument("session", type=Path, help="Session .json, .npz, or directory")
    args = parser.parse_args(argv)
    result = export_existing_session(args.session)
    print(json.dumps(result, indent=2))


def main_validate(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Validate a finalized HF-DM session manifest.")
    parser.add_argument("session", type=Path, help="Session bundle, manifest, .json, or .npz")
    args = parser.parse_args(argv)
    result = validate_session(args.session)
    print(json.dumps(result, indent=2))
    if not result["valid"]:
        raise SystemExit(1)


def _extract_export_options(metadata: dict[str, Any]) -> dict[str, Any]:
    return dict((metadata.get("config") or {}).get("session_export") or {})


def _to_dict(value: Any) -> dict[str, Any]:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, SimpleNamespace):
        return dict(value.__dict__)
    return {"value": value}


def _event_to_dict(event: Any) -> dict[str, Any]:
    out = _to_dict(event)
    if "payload" in out and not isinstance(out["payload"], dict):
        try:
            out["payload"] = json.loads(out["payload"])
        except Exception:
            out["payload"] = {"value": out["payload"]}
    out.setdefault("payload", {})
    return out


def _streamdio_to_dict(row: Any) -> dict[str, Any]:
    if isinstance(row, dict):
        return {name: row.get(name) for name in STREAMDIO_COLUMNS}
    if hasattr(row, "dtype") and getattr(row.dtype, "names", None):
        return {name: _json_value(row[name]) for name in STREAMDIO_COLUMNS}
    values = list(row)
    return {
        name: _json_value(values[index]) if index < len(values) else None
        for index, name in enumerate(STREAMDIO_COLUMNS)
    }


def _json_value(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def _write_config_snapshot(path: Path, metadata: dict[str, Any]) -> None:
    source_config_path = (metadata.get("config") or {}).get("session_export", {}).get(
        "source_config_path"
    )
    cli_overrides = _config_cli_overrides(metadata)
    snapshot = {
        "source_config_path": source_config_path,
        "cli_overrides": cli_overrides,
        "effective_config": metadata.get("config") or {},
        "task_summary": metadata.get("task_summary") or {},
        "trial_schedule": metadata.get("trial_schedule") or [],
    }
    if source_config_path:
        source_path = Path(source_config_path)
        try:
            source_bytes = source_path.read_bytes()
            snapshot["source_config_sha256"] = hashlib.sha256(source_bytes).hexdigest()
            snapshot["source_config_yaml"] = source_bytes.decode("utf-8", errors="replace")
        except OSError as exc:
            snapshot["source_config_error"] = f"{type(exc).__name__}: {exc}"
    with _atomic_text_handle(path, newline="\n") as handle:
        handle.write(yaml.safe_dump(snapshot, sort_keys=False))


def _config_provenance(metadata: dict[str, Any]) -> dict[str, Any]:
    source_config_path = (metadata.get("config") or {}).get("session_export", {}).get(
        "source_config_path"
    )
    provenance: dict[str, Any] = {
        "source_config_path": source_config_path,
        "snapshot_artifact": "config.snapshot.yaml",
    }
    cli_overrides = _config_cli_overrides(metadata)
    if cli_overrides:
        provenance["cli_overrides"] = cli_overrides
    if not source_config_path:
        return provenance
    source_path = Path(source_config_path)
    try:
        provenance["source_config_sha256"] = sha256_file(source_path)
    except OSError as exc:
        provenance["source_config_error"] = f"{type(exc).__name__}: {exc}"
    return provenance


def _config_cli_overrides(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    direct = metadata.get("config_overrides")
    if isinstance(direct, list):
        return [dict(item) for item in direct if isinstance(item, dict)]
    if isinstance(direct, dict):
        return [dict(direct)]
    session_export = (metadata.get("config") or {}).get("session_export") or {}
    from_config = session_export.get("cli_overrides")
    if isinstance(from_config, list):
        return [dict(item) for item in from_config if isinstance(item, dict)]
    if isinstance(from_config, dict):
        return [dict(from_config)]
    return []


def _write_dict_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["empty"]
    with _atomic_text_handle(path, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})
    return path


def _write_streamdio_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    with _atomic_text_handle(path, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STREAMDIO_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in STREAMDIO_COLUMNS})
    return path


def _write_timing_heartbeat_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    """Write the complete dual-clock heartbeat trace, including an empty header."""

    with _atomic_text_handle(path, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TIMING_HEARTBEAT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in TIMING_HEARTBEAT_COLUMNS})
    return path


def _write_streamdio_subset_csv(
    path: Path,
    rows: list[dict[str, Any]],
    predicate: Any,
) -> Path:
    return _write_streamdio_csv(path, [row for row in rows if predicate(row)])


def _write_licks_csv(path: Path, events: list[dict[str, Any]]) -> Path:
    rows = []
    for event in events:
        if event.get("event_type") not in {"lick", "lick_edge"}:
            continue
        payload = event.get("payload") or {}
        rows.append(
            {
                "wall_time_s": event.get("wall_time_s"),
                "monotonic_s": event.get("monotonic_s"),
                "event_type": event.get("event_type"),
                "trial_index": payload.get("trial_index"),
                "channel": payload.get("channel"),
                "state": payload.get("state"),
                "timestamp_us": payload.get("timestamp_us"),
                "relative_time_s": payload.get("relative_time_s"),
            }
        )
    return _write_dict_csv(path, rows)


def _write_bpod_flat_csv(path: Path, bpod_trials: list[dict[str, Any]]) -> Path:
    rows: list[dict[str, Any]] = []
    for trial in bpod_trials:
        trial_index = trial.get("trial_index")
        for row_index, row in enumerate(trial.get("events") or []):
            flat = {
                "trial_index": trial_index,
                "row_index": row_index,
                "row_json": json.dumps(row, default=str),
            }
            if isinstance(row, dict):
                for key, value in row.items():
                    if isinstance(value, (dict, list)):
                        flat[key] = json.dumps(value, default=str)
                    else:
                        flat[key] = value
            rows.append(flat)
    return _write_dict_csv(path, rows)


def _bpod_trial_row_count(bpod_trials: list[dict[str, Any]]) -> int:
    return sum(len(trial.get("events") or []) for trial in bpod_trials)


def _write_bpod_data_json(
    path: Path,
    *,
    bpod_trials: list[dict[str, Any]],
    fsm_definitions: dict[str, str] | list[dict[str, Any]],
) -> Path | None:
    if not bpod_trials and not fsm_definitions:
        return None
    if isinstance(fsm_definitions, dict):
        fsm_rows = [
            {"fsm_name": name, "fsm_json": fsm_json}
            for name, fsm_json in fsm_definitions.items()
        ]
    else:
        fsm_rows = list(fsm_definitions)
    payload = {
        "schema_version": 1,
        "source": "bpod_core_get_data_concat_false",
        "trial_count": len(bpod_trials),
        "fsm_definition_count": len(fsm_rows),
        "bpod_trials": bpod_trials,
        "fsm_definitions": fsm_rows,
    }
    _atomic_write_text(path, json.dumps(payload, indent=2))
    return path


def _timeline_rows(
    events: list[dict[str, Any]],
    streamdio_rows: list[dict[str, Any]],
    *,
    bundle_dir: Path | None = None,
) -> list[dict[str, Any]]:
    event_start = _first_mono(events)
    rows: list[dict[str, Any]] = []
    for event in events:
        event_type = str(event.get("event_type"))
        if event_type not in SEMANTIC_TIMELINE_EVENTS:
            continue
        payload = event.get("payload") or {}
        event_mono_s = payload.get(
            "host_perf_counter_s",
            payload.get("monotonic_s", event.get("monotonic_s")),
        )
        rows.append(
            {
                "session_mono_s": _relative(event_mono_s, event_start),
                "wall_time_s": payload.get(
                    "host_wall_time_s",
                    payload.get("wall_time_s", event.get("wall_time_s")),
                ),
                "module_timestamp_us": payload.get(
                    "module_timestamp_us", payload.get("timestamp_us")
                ),
                "source": _event_source(event_type),
                "event_type": event_type,
                "trial_index": payload.get("trial_index"),
                "detail_json": json.dumps(payload, default=str, sort_keys=True),
            }
        )
    for row in streamdio_rows:
        event_type = str(row.get("event_type"))
        if event_type not in {"camera", "bpod_state"}:
            continue
        rows.append(
            {
                # StreamDIO relative_time_s is firmware-relative and does not
                # share an origin with session event monotonic_s.  Co-sort only
                # on the captured host perf-counter clock.
                "session_mono_s": _relative(
                    row.get("host_perf_counter_s"), event_start
                ),
                "wall_time_s": "",
                "module_timestamp_us": row.get("module_timestamp_us"),
                "source": "streamdio",
                "event_type": event_type,
                "trial_index": "",
                "detail_json": json.dumps(row, default=str, sort_keys=True),
            }
        )
    first_frame = _first_spinnaker_frame(bundle_dir, events) if bundle_dir is not None else None
    if first_frame is not None:
        rows.append(
            {
                "session_mono_s": _relative(first_frame.get("monotonic_s"), event_start),
                "wall_time_s": first_frame.get("wall_time_s"),
                "module_timestamp_us": first_frame.get("camera_timestamp_us"),
                "source": "spinnaker",
                "event_type": "first_spinnaker_frame",
                "trial_index": "",
                "detail_json": json.dumps(first_frame, default=str, sort_keys=True),
            }
        )
    rows.sort(key=lambda item: _sort_key(item.get("session_mono_s")))
    return rows


def _alignment_anchors(
    events: list[dict[str, Any]],
    streamdio_rows: list[dict[str, Any]],
    *,
    bundle_dir: Path | None = None,
) -> dict[str, Any]:
    markers = [e for e in events if e.get("event_type") == "bpod_state_marker"]
    choices = [e for e in events if e.get("event_type") == "choice_detected"]
    pump_events = [
        e for e in events if e.get("event_type") in {"pump_start", "pump_end", "output_actual"}
    ]
    first_camera = next((row for row in streamdio_rows if row.get("event_type") == "camera"), None)
    first_frame2ttl = next(
        (
            row
            for row in streamdio_rows
            if row.get("event_type") == "dio_input"
            and int(row.get("channel") or -1) == FRAME2TTL_INPUT_CHANNEL
        ),
        None,
    )
    return {
        "bpod_state_markers": [_compact_event(event) for event in markers],
        "choice_events": [_compact_event(event) for event in choices],
        "pump_and_output_events": [_compact_event(event) for event in pump_events],
        "first_camera_ttl": first_camera,
        "first_frame2ttl_edge": first_frame2ttl,
        "first_spinnaker_frame": (
            _first_spinnaker_frame(bundle_dir, events)
            if bundle_dir is not None
            else None
        ),
    }


def _first_spinnaker_frame(
    bundle_dir: Path | None,
    events: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if bundle_dir is None:
        return None
    stopped = [event for event in events if event.get("event_type") == "video_recording_stopped"]
    if not stopped:
        return None
    payload = stopped[-1].get("payload") or {}
    timestamps_value = payload.get("frame_timestamps_csv")
    if not timestamps_value:
        return None
    timestamps_path = Path(timestamps_value)
    if not timestamps_path.is_absolute():
        timestamps_path = bundle_dir / timestamps_path
    if not timestamps_path.exists():
        return None
    try:
        with timestamps_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            row = next(reader, None)
    except Exception:
        return None
    if not row:
        return None
    camera_timestamp_ns = _optional_int(row.get("camera_timestamp_ns"))
    return {
        "frame_index": _optional_int(row.get("frame_index")),
        "camera_timestamp_ns": camera_timestamp_ns,
        "camera_timestamp_us": (
            None if camera_timestamp_ns is None else int(camera_timestamp_ns // 1000)
        ),
        "wall_time_s": _optional_float(row.get("wall_time_s") or row.get("host_wall_s")),
        "monotonic_s": _optional_float(row.get("monotonic_s") or row.get("host_mono_s")),
        "filename": row.get("filename") or "",
        "frame_timestamps_csv": _relative_path(timestamps_path, bundle_dir),
    }


def _soundcard_status(events: list[dict[str, Any]]) -> dict[str, Any]:
    snapshots = []
    for event in events:
        if event.get("event_type") not in {
            "white_noise_ready",
            "white_noise_start",
            "white_noise_end",
            "white_noise_shutdown",
        }:
            continue
        payload = event.get("payload") or {}
        status = payload.get("sound_status") or payload.get("status")
        result = payload.get("result")
        if not status and isinstance(result, dict):
            status = result.get("sound_status")
        if not status:
            continue
        snapshots.append(
            {
                "event_type": event.get("event_type"),
                "wall_time_s": event.get("wall_time_s"),
                "monotonic_s": event.get("monotonic_s"),
                "trial_index": payload.get("trial_index"),
                "status": status,
            }
        )
    latest_status = snapshots[-1]["status"] if snapshots else {}
    return {
        "snapshot_count": len(snapshots),
        "device_id": latest_status.get("device_id"),
        "volume": latest_status.get("volume"),
        "continuous_playback_active": latest_status.get("continuous_playback_active"),
        "latest_status": latest_status,
        "snapshots": snapshots,
    }


def _maybe_export_preview_video(
    *,
    bundle_dir: Path,
    exports_dir: Path,
    metadata: dict[str, Any],
    event_dicts: list[dict[str, Any]],
    options: dict[str, Any],
) -> Path | None:
    video_config = (metadata.get("config") or {}).get("video") or {}
    requested = bool(
        options.get("duplicate_preview_video", False)
        or video_config.get("duplicate_preview_video", False)
    )
    if not requested:
        return None
    video = _video_summary(bundle_dir, event_dicts)
    session_dir = video.get("session_dir")
    if not session_dir:
        path = exports_dir / "preview_video_error.json"
        _atomic_write_text(
            path,
            json.dumps({"error": "no video session_dir found"}, indent=2),
        )
        return path
    try:
        from hf_dm.data.video_export import export_preview_video

        result = export_preview_video(
            bundle_dir / str(session_dir),
            fps=video_config.get("preview_video_fps") or None,
            codec=str(video_config.get("preview_video_codec") or "FFV1"),
        )
        path = exports_dir / "preview_video_export.json"
        _atomic_write_text(path, json.dumps(result, indent=2))
        return path
    except Exception as exc:
        path = exports_dir / "preview_video_error.json"
        _atomic_write_text(
            path,
            json.dumps({"error": f"{type(exc).__name__}: {exc}"}, indent=2),
        )
        return path


def _write_video_pointer(
    *,
    bundle_dir: Path,
    videos_dir: Path,
    event_dicts: list[dict[str, Any]],
) -> Path | None:
    video = _video_summary(bundle_dir, event_dicts)
    if not video:
        return None
    artifacts = []
    for artifact in _video_artifact_paths(bundle_dir, event_dicts):
        if not artifact.exists():
            continue
        artifacts.append(
            {
                "path": _relative_path(artifact, bundle_dir),
                "bytes": artifact.stat().st_size,
                "sha256": sha256_file(artifact),
            }
        )
    pointer = {
        "schema_version": 1,
        "kind": "spinnaker_video_pointer",
        "video": video,
        "artifacts": artifacts,
    }
    path = videos_dir / "spinnaker_video.json"
    _atomic_write_text(path, json.dumps(pointer, indent=2))
    return path


def _maybe_write_alyx_registration(
    *,
    exports_dir: Path,
    manifest_path: Path,
    bundle_dir: Path,
    metadata: dict[str, Any],
    summary: dict[str, Any],
    artifacts: list[Path],
) -> Path | None:
    try:
        from hf_dm.data.alyx_registration import maybe_write_alyx_registration

        return maybe_write_alyx_registration(
            exports_dir=exports_dir,
            manifest_path=manifest_path,
            bundle_dir=bundle_dir,
            metadata=metadata,
            summary=summary,
            artifacts=artifacts,
        )
    except Exception as exc:
        path = exports_dir / "alyx_registration.json"
        _atomic_write_text(
            path,
            json.dumps({"status": "error", "error": f"{type(exc).__name__}: {exc}"}, indent=2),
        )
        return path


def _manifest(
    *,
    manifest_path: Path,
    base_path: Path,
    summary: dict[str, Any],
    metadata: dict[str, Any],
    trials: list[dict[str, Any]],
    event_dicts: list[dict[str, Any]],
    streamdio_rows: list[dict[str, Any]],
    bpod_trials: list[dict[str, Any]],
    fsm_definitions: dict[str, str] | list[dict[str, Any]],
    artifacts: list[Path],
    options: dict[str, Any],
    recorder_stats: dict[str, Any],
    timing_heartbeat_qc: dict[str, Any],
) -> dict[str, Any]:
    counts_by_type: dict[str, int] = {}
    for row in streamdio_rows:
        event_type = str(row.get("event_type"))
        counts_by_type[event_type] = counts_by_type.get(event_type, 0) + 1
    event_counts: dict[str, int] = {}
    for event in event_dicts:
        event_type = str(event.get("event_type"))
        event_counts[event_type] = event_counts.get(event_type, 0) + 1
    artifact_entries = []
    for path in artifacts:
        if not path.exists():
            continue
        artifact_entries.append(
            {
                "path": _relative_path(path, manifest_path.parent),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    streamdio_info = _streamdio_module_info(metadata, event_dicts)
    video_summary = _video_summary(manifest_path.parent, event_dicts)
    frame_count = _optional_int(video_summary.get("n_frames")) or 0
    force_channel = _force_channel(metadata)
    force_sample_count = sum(
        1
        for row in streamdio_rows
        if row.get("event_type") == "analog_input"
        and int(row.get("channel") or -1) == force_channel
    )
    runtime_provenance = metadata.get("runtime_provenance") or {}
    package_version = (
        runtime_provenance.get("package_version")
        if "package_version" in runtime_provenance
        else _package_version()
    )
    git_commit = (
        runtime_provenance.get("git_commit")
        if "git_commit" in runtime_provenance
        else _git_commit(base_path.parent)
    )
    manifest = {
        "schema_version": 1,
        "created_wall_time_s": time.time(),
        "session_base": str(base_path),
        "layout": options.get("layout", "flat"),
        "package": {
            "name": "bpod-hf-dm",
            "version": package_version,
            "git_commit": git_commit,
            "git_dirty": runtime_provenance.get("git_dirty"),
            "provenance_captured_wall_s": runtime_provenance.get(
                "provenance_captured_wall_s"
            ),
        },
        "config": _config_provenance(metadata),
        "summary": summary,
        "flags": _flags(metadata),
        "counts": {
            "trials": len(trials),
            "events": len(event_dicts),
            "streamdio_events": len(streamdio_rows),
            "streamdio_by_type": dict(sorted(counts_by_type.items())),
            "session_events_by_type": dict(sorted(event_counts.items())),
            "licks": event_counts.get("lick", 0),
            "camera_ttls": counts_by_type.get("camera", 0),
            "timing_heartbeats": counts_by_type.get(
                TIMING_HEARTBEAT_EVENT_TYPE,
                0,
            ),
            "frames": frame_count,
            "force_samples": force_sample_count,
            "frame2ttl_edges": sum(
                1
                for row in streamdio_rows
                if row.get("event_type") == "dio_input"
                and int(row.get("channel") or -1) == _frame2ttl_channel(metadata)
            ),
            "bpod_trials": len(bpod_trials),
            "bpod_trial_rows": _bpod_trial_row_count(bpod_trials),
        },
        "streamdio": {
            "module_info": streamdio_info,
            "firmware_version": (
                streamdio_info.get("firmware_version") if isinstance(streamdio_info, dict) else None
            ),
            "timing_heartbeat_qc": timing_heartbeat_qc,
        },
        "stream_summaries": metadata.get("stream_summaries") or {},
        "video": video_summary,
        "clock_alignment": _clock_alignment(event_dicts, streamdio_rows, metadata),
        "artifacts": artifact_entries,
        "fsm_definition_count": (
            len(fsm_definitions)
            if isinstance(fsm_definitions, dict)
            else len(list(fsm_definitions))
        ),
        "recorder_stats": recorder_stats,
    }
    monitor_telemetry = _monitor_telemetry_summary(manifest_path.parent, artifacts)
    if monitor_telemetry is not None:
        manifest["monitor_telemetry"] = monitor_telemetry
    continuous_streams = _continuous_stream_summary(manifest_path.parent, artifacts)
    if continuous_streams is not None:
        manifest["continuous_streams"] = continuous_streams
    return manifest


def _clock_alignment(
    events: list[dict[str, Any]],
    streamdio_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    wall_perf_anchors: list[tuple[float, float]] = []
    module_perf_anchors: list[tuple[float, float]] = []
    module_wall_anchors: list[tuple[float, float]] = []

    for event in events:
        event_type = str(event.get("event_type", ""))
        if event_type not in {
            "bpod_state_marker",
            "frame2ttl_edge",
            TIMING_HEARTBEAT_EVENT_TYPE,
        }:
            continue
        payload = event.get("payload") or {}
        record_perf_s = _finite_float(event.get("monotonic_s"))
        payload_perf_s = _finite_float(payload.get("host_perf_counter_s"))
        perf_s = payload_perf_s if payload_perf_s is not None else record_perf_s
        wall_s = _finite_float(
            payload.get(
                "host_wall_time_s",
                payload.get("wall_time_s", event.get("wall_time_s")),
            )
        )
        module_us = _finite_float(
            payload.get("module_timestamp_us", payload.get("timestamp_us"))
        )
        module_s = None if module_us is None else module_us / 1_000_000.0
        wall_perf_s = (
            payload_perf_s
            if payload.get("host_wall_time_s") is not None
            or payload.get("wall_time_s") is not None
            else record_perf_s
        )
        if wall_perf_s is not None and wall_s is not None:
            wall_perf_anchors.append((wall_perf_s, wall_s))
        if module_s is not None and perf_s is not None:
            module_perf_anchors.append((module_s, perf_s))
        if module_s is not None and wall_s is not None:
            module_wall_anchors.append((module_s, wall_s))

    frame2ttl_channel = _frame2ttl_channel(metadata)
    for row in streamdio_rows:
        row_event_type = row.get("event_type")
        if row_event_type == "dio_input":
            if _optional_int(row.get("channel")) != frame2ttl_channel:
                continue
        elif row_event_type != TIMING_HEARTBEAT_EVENT_TYPE:
            continue
        module_us = _finite_float(row.get("module_timestamp_us"))
        perf_s = _finite_float(row.get("host_perf_counter_s"))
        if module_us is not None and perf_s is not None:
            module_perf_anchors.append((module_us / 1_000_000.0, perf_s))

    return {
        "model": "target_s = slope * source_s + intercept",
        "pairs": {
            "perf_counter_to_wall": _fit_affine_clock_pair(
                wall_perf_anchors,
                source_clock="perf_counter_s",
                target_clock="wall_time_s",
            ),
            "streamdio_to_perf_counter": _fit_affine_clock_pair(
                module_perf_anchors,
                source_clock="streamdio_module_s",
                target_clock="perf_counter_s",
            ),
            "streamdio_to_wall": _fit_affine_clock_pair(
                module_wall_anchors,
                source_clock="streamdio_module_s",
                target_clock="wall_time_s",
            ),
        },
    }


def _fit_affine_clock_pair(
    anchors: list[tuple[float, float]],
    *,
    source_clock: str,
    target_clock: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "source_clock": source_clock,
        "target_clock": target_clock,
        "status": "insufficient_anchors",
        "slope": None,
        "intercept": None,
        "n_anchors": len(anchors),
        "max_abs_residual_s": None,
        "rms_residual_s": None,
    }
    if len(anchors) < 2:
        return result

    source: np.ndarray = np.asarray([anchor[0] for anchor in anchors], dtype=float)
    target: np.ndarray = np.asarray([anchor[1] for anchor in anchors], dtype=float)
    source_centered = source - float(np.mean(source))
    denominator = float(np.dot(source_centered, source_centered))
    if denominator <= np.finfo(float).eps:
        result["status"] = "degenerate_anchors"
        return result

    target_centered = target - float(np.mean(target))
    slope = float(np.dot(source_centered, target_centered) / denominator)
    intercept = float(np.mean(target) - slope * np.mean(source))
    residuals = target - (slope * source + intercept)
    result.update(
        {
            "status": "ok",
            "slope": slope,
            "intercept": intercept,
            "max_abs_residual_s": float(np.max(np.abs(residuals))),
            "rms_residual_s": float(np.sqrt(np.mean(np.square(residuals)))),
        }
    )
    return result


def _continuous_stream_summary(
    bundle_dir: Path,
    artifacts: list[Path],
) -> dict[str, Any] | None:
    chunks = []
    counts_by_type: dict[str, int] = {}
    event_count = 0
    for path in artifacts:
        if not path.exists() or path.suffix.lower() != ".npz":
            continue
        rel = _relative_path(path, bundle_dir).replace("\\", "/")
        if not rel.startswith("continuous_streams/streamdio_chunk_"):
            continue
        summary = _streamdio_chunk_summary(path, rel)
        chunks.append(summary)
        event_count += int(summary.get("event_count") or 0)
        for event_type, count in (summary.get("counts_by_type") or {}).items():
            counts_by_type[str(event_type)] = (
                counts_by_type.get(str(event_type), 0) + int(count)
            )
    if not chunks:
        return None
    chunks.sort(key=lambda item: str(item.get("path") or ""))
    return {
        "stream": "streamdio_events",
        "chunk_count": len(chunks),
        "event_count": event_count,
        "counts_by_type": dict(sorted(counts_by_type.items())),
        "chunks": chunks,
    }


def _streamdio_chunk_summary(path: Path, rel: str) -> dict[str, Any]:
    try:
        rows = _load_npz_streamdio(path)
    except Exception as exc:
        return {
            "path": rel,
            "event_count": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }
    counts_by_type: dict[str, int] = {}
    for row in rows:
        event_type = str(row.get("event_type"))
        counts_by_type[event_type] = counts_by_type.get(event_type, 0) + 1
    summary: dict[str, Any] = {
        "path": rel,
        "event_count": len(rows),
        "counts_by_type": dict(sorted(counts_by_type.items())),
    }
    if rows:
        first = rows[0]
        last = rows[-1]
        summary.update(
            {
                "first_module_timestamp_us": _optional_int(
                    first.get("module_timestamp_us")
                ),
                "last_module_timestamp_us": _optional_int(last.get("module_timestamp_us")),
                "first_relative_time_s": _optional_float(first.get("relative_time_s")),
                "last_relative_time_s": _optional_float(last.get("relative_time_s")),
            }
        )
    return summary


def _monitor_telemetry_summary(
    bundle_dir: Path,
    artifacts: list[Path],
) -> dict[str, Any] | None:
    for path in artifacts:
        if not path.exists():
            continue
        rel = _relative_path(path, bundle_dir).replace("\\", "/")
        if rel == "exports/monitor_telemetry.jsonl":
            return _monitor_telemetry_jsonl_summary(path, rel)
    return None


def _monitor_telemetry_jsonl_summary(path: Path, rel: str) -> dict[str, Any]:
    line_count = 0
    parse_error_count = 0
    counts_by_kind: dict[str, int] = {}
    counts_by_message_type: dict[str, int] = {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                line_count += 1
                try:
                    row = json.loads(text)
                except json.JSONDecodeError:
                    parse_error_count += 1
                    continue
                if not isinstance(row, dict):
                    continue
                kind = row.get("kind")
                if isinstance(kind, str) and kind:
                    counts_by_kind[kind] = counts_by_kind.get(kind, 0) + 1
                message = row.get("message")
                if isinstance(message, dict):
                    message_type = message.get("type")
                    if isinstance(message_type, str) and message_type:
                        counts_by_message_type[message_type] = (
                            counts_by_message_type.get(message_type, 0) + 1
                        )
    except OSError as exc:
        return {
            "path": rel,
            "line_count": line_count,
            "error": f"{type(exc).__name__}: {exc}",
        }
    summary = {
        "path": rel,
        "line_count": line_count,
        "counts_by_kind": dict(sorted(counts_by_kind.items())),
        "counts_by_message_type": dict(sorted(counts_by_message_type.items())),
    }
    if parse_error_count:
        summary["parse_error_count"] = parse_error_count
    return summary


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, default=str, sort_keys=True)
    return value


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _finite_float(value: Any) -> float | None:
    parsed = _optional_float(value)
    if parsed is None or not np.isfinite(parsed):
        return None
    return parsed


def _first_mono(events: list[dict[str, Any]]) -> float:
    for event in events:
        value = event.get("monotonic_s")
        if value not in (None, ""):
            return float(str(value))
    return 0.0


def _relative(value: Any, start: float) -> Any:
    if value in (None, ""):
        return ""
    return float(value) - start


def _sort_key(value: Any) -> float:
    if value in (None, ""):
        return float("inf")
    return float(value)


def _event_source(event_type: str) -> str:
    if event_type.startswith("frame2ttl"):
        return "frame2ttl"
    if event_type == "bpod_state_marker":
        return "bpod_streamdio_marker"
    if event_type == TIMING_HEARTBEAT_EVENT_TYPE:
        return "streamdio_teensy"
    if event_type in {"pump_start", "pump_end", "output_actual", "force_on", "force_off"}:
        return "outputs"
    if event_type.startswith("white_noise"):
        return "sound"
    if event_type.startswith("video"):
        return "spinnaker"
    return "session"


def _compact_event(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "wall_time_s": event.get("wall_time_s"),
        "monotonic_s": event.get("monotonic_s"),
        "event_type": event.get("event_type"),
        "payload": event.get("payload") or {},
    }


def _streamdio_module_info(
    metadata: dict[str, Any],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    direct = metadata.get("streamdio_module_info")
    if isinstance(direct, dict):
        return direct
    for event in events:
        if event.get("event_type") != "hardware_connected":
            continue
        payload = event.get("payload") or {}
        info = payload.get("streamdio_module_info")
        if isinstance(info, dict):
            return info
    return {}


def _flags(metadata: dict[str, Any]) -> dict[str, Any]:
    config = metadata.get("config") or {}
    return {
        "camera": bool((config.get("camera") or {}).get("record_video", False)),
        "frame2ttl": bool((config.get("frame2ttl") or {}).get("enabled", False)),
        "cap_touch": bool((config.get("licks") or {}).get("power_required", False)),
        "monitor": bool((config.get("monitor") or {}).get("enabled", False)),
    }


def _video_summary(bundle_dir: Path, events: list[dict[str, Any]]) -> dict[str, Any]:
    started = [e for e in events if e.get("event_type") == "video_recording_started"]
    stopped = [e for e in events if e.get("event_type") == "video_recording_stopped"]
    payload = dict((stopped[-1].get("payload") if stopped else {}) or {})
    if started:
        payload.setdefault("started", started[-1].get("payload") or {})
    session_dir_value = payload.get("session_dir")
    session_dir_path = Path(session_dir_value) if session_dir_value else None
    if session_dir_path is not None and not session_dir_path.is_absolute():
        session_dir_path = bundle_dir / session_dir_path
    if session_dir_path is not None:
        recording_summary = session_dir_path / "recording_summary.json"
        if recording_summary.exists():
            payload["primary"] = _relative_path(recording_summary, bundle_dir)
        if "n_frames" not in payload and payload.get("frames_saved") is not None:
            payload["n_frames"] = int(payload["frames_saved"])
        preview_meta_path = session_dir_path / "preview_video_export.json"
        if preview_meta_path.exists():
            try:
                preview = json.loads(preview_meta_path.read_text(encoding="utf-8"))
                if preview.get("output_path"):
                    preview["output_path"] = _relative_path(
                        Path(preview["output_path"]),
                        bundle_dir,
                    )
                preview["metadata_path"] = _relative_path(preview_meta_path, bundle_dir)
                payload["preview_video"] = preview
            except Exception:
                payload["preview_video"] = {
                    "metadata_path": _relative_path(preview_meta_path, bundle_dir),
                    "read_error": True,
                }
    for key in ("session_dir", "frame_timestamps_csv"):
        if payload.get(key):
            payload[key] = _relative_path(Path(payload[key]), bundle_dir)
    return payload


def _video_artifact_paths(bundle_dir: Path, events: list[dict[str, Any]]) -> list[Path]:
    stopped = [e for e in events if e.get("event_type") == "video_recording_stopped"]
    if not stopped:
        return []
    payload = dict((stopped[-1].get("payload") if stopped else {}) or {})
    paths: list[Path] = []
    session_dir_value = payload.get("session_dir")
    if session_dir_value:
        session_dir = Path(session_dir_value)
        if not session_dir.is_absolute():
            session_dir = bundle_dir / session_dir
        paths.append(session_dir / "recording_summary.json")
    if payload.get("frame_timestamps_csv"):
        timestamps = Path(payload["frame_timestamps_csv"])
        if not timestamps.is_absolute():
            timestamps = bundle_dir / timestamps
        paths.append(timestamps)
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen or not path.exists():
            continue
        seen.add(resolved)
        deduped.append(path)
    return deduped


def _relative_path(path: Path, start: Path) -> str:
    try:
        return os.path.relpath(path.resolve(), start.resolve())
    except ValueError:
        return str(path)


def _force_channel(metadata: dict[str, Any]) -> int:
    config = metadata.get("config") or {}
    wheel = config.get("wheel") or {}
    return int(wheel.get("force_channel", FORCE_ANALOG_CHANNEL))


def _frame2ttl_channel(metadata: dict[str, Any]) -> int:
    config = metadata.get("config") or {}
    frame2ttl = config.get("frame2ttl") or {}
    return int(frame2ttl.get("input_channel", FRAME2TTL_INPUT_CHANNEL))


def _package_version() -> str | None:
    try:
        return version("bpod-hf-dm")
    except PackageNotFoundError:
        return None


def _git_commit(cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd),
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _load_npz_streamdio(path: Path) -> list[dict[str, Any]]:
    with np.load(path, allow_pickle=True) as data:
        if "streamdio_events" not in data.files:
            return []
        return [_streamdio_to_dict(row) for row in data["streamdio_events"]]


def _validate_config_provenance(manifest: dict[str, Any], bundle_dir: Path) -> list[str]:
    config = manifest.get("config") or {}
    expected = config.get("source_config_sha256")
    if not expected:
        return []
    errors: list[str] = []
    snapshot_rel = config.get("snapshot_artifact") or "config.snapshot.yaml"
    snapshot_path = bundle_dir / str(snapshot_rel)
    if snapshot_path.exists():
        try:
            snapshot = yaml.safe_load(snapshot_path.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            errors.append(
                f"config snapshot unreadable: {snapshot_rel}: {type(exc).__name__}: {exc}"
            )
        else:
            source_yaml = snapshot.get("source_config_yaml")
            if source_yaml is not None:
                actual = hashlib.sha256(str(source_yaml).encode("utf-8")).hexdigest()
                if actual != str(expected):
                    errors.append(
                        f"config snapshot source sha256 mismatch: {snapshot_rel}: "
                        f"{actual} != {expected}"
                    )
                return errors
    source_path_value = config.get("source_config_path")
    if source_path_value:
        source_path = Path(str(source_path_value))
        if source_path.exists():
            actual = sha256_file(source_path)
            if actual != str(expected):
                errors.append(
                    f"source config sha256 mismatch: {source_path}: {actual} != {expected}"
                )
    return errors


def _validate_manifest_flags(manifest: dict[str, Any], bundle_dir: Path) -> list[str]:
    flags = manifest.get("flags")
    if not isinstance(flags, dict):
        return ["manifest missing flags object"]
    errors: list[str] = []
    for key in ("camera", "frame2ttl", "cap_touch", "monitor"):
        if not isinstance(flags.get(key), bool):
            errors.append(f"manifest flag {key} is not boolean")

    snapshot_rel = (manifest.get("config") or {}).get("snapshot_artifact")
    if not snapshot_rel:
        return errors
    snapshot_path = bundle_dir / str(snapshot_rel)
    if not snapshot_path.exists():
        return errors
    try:
        snapshot = yaml.safe_load(snapshot_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return errors
    effective_config = snapshot.get("effective_config")
    if not isinstance(effective_config, dict):
        return errors
    expected = {
        "camera": bool((effective_config.get("camera") or {}).get("record_video", False)),
        "frame2ttl": bool(
            (effective_config.get("frame2ttl") or {}).get("enabled", False)
        ),
        "cap_touch": bool(
            (effective_config.get("licks") or {}).get("power_required", False)
        ),
        "monitor": bool((effective_config.get("monitor") or {}).get("enabled", False)),
    }
    for key, expected_value in expected.items():
        actual = flags.get(key)
        if isinstance(actual, bool) and actual != expected_value:
            errors.append(
                f"manifest flag {key} mismatch: {actual} != {expected_value}"
            )
    return errors


def _validate_manifest_counts(manifest: dict[str, Any], bundle_dir: Path) -> list[str]:
    counts = manifest.get("counts")
    if not isinstance(counts, dict):
        return ["manifest missing counts object"]
    errors: list[str] = []
    payload, rel = _primary_session_json_payload(manifest, bundle_dir)
    metadata: dict[str, Any] = {}
    if payload is not None:
        raw_metadata = payload.get("metadata")
        metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
        trials = payload.get("trials")
        if isinstance(trials, list):
            _append_count_mismatch(
                errors,
                "manifest count trials",
                counts.get("trials"),
                len(trials),
                rel,
            )
        events = payload.get("events")
        if isinstance(events, list):
            _append_count_mismatch(
                errors,
                "manifest count events",
                counts.get("events"),
                len(events),
                rel,
            )
            event_counts = _count_by_key(events, "event_type")
            _append_mapping_mismatch(
                errors,
                "manifest session_events_by_type",
                counts.get("session_events_by_type"),
                event_counts,
                rel,
            )
            _append_count_mismatch(
                errors,
                "manifest count licks",
                counts.get("licks"),
                event_counts.get("lick", 0),
                rel,
            )
        bpod_trials = payload.get("bpod_trials")
        if isinstance(bpod_trials, list):
            _append_count_mismatch(
                errors,
                "manifest count bpod_trials",
                counts.get("bpod_trials"),
                len(bpod_trials),
                rel,
            )
            _append_count_mismatch(
                errors,
                "manifest count bpod_trial_rows",
                counts.get("bpod_trial_rows"),
                _bpod_trial_row_count(bpod_trials),
                rel,
            )
        streamdio = payload.get("streamdio")
        if isinstance(streamdio, dict):
            _append_count_mismatch(
                errors,
                "manifest count streamdio_events",
                counts.get("streamdio_events"),
                _optional_int(streamdio.get("event_count")),
                rel,
            )
            streamdio_counts = streamdio.get("counts_by_type")
            if isinstance(streamdio_counts, dict):
                _append_mapping_mismatch(
                    errors,
                    "manifest streamdio_by_type",
                    counts.get("streamdio_by_type"),
                    streamdio_counts,
                    rel,
                )

    npz_path, npz_rel = _primary_npz_path(manifest, bundle_dir)
    if npz_path is not None and npz_path.exists():
        streamdio_rows: list[dict[str, Any]] | None
        try:
            streamdio_rows = _load_npz_streamdio(npz_path)
        except Exception:
            streamdio_rows = None
        if streamdio_rows is not None:
            streamdio_counts = _count_by_key(streamdio_rows, "event_type")
            _append_count_mismatch(
                errors,
                "manifest count streamdio_events",
                counts.get("streamdio_events"),
                len(streamdio_rows),
                npz_rel,
            )
            _append_count_mismatch(
                errors,
                "manifest count timing_heartbeats",
                counts.get("timing_heartbeats"),
                streamdio_counts.get(TIMING_HEARTBEAT_EVENT_TYPE, 0),
                npz_rel,
            )
            _append_mapping_mismatch(
                errors,
                "manifest streamdio_by_type",
                counts.get("streamdio_by_type"),
                streamdio_counts,
                npz_rel,
            )
            _append_count_mismatch(
                errors,
                "manifest count camera_ttls",
                counts.get("camera_ttls"),
                streamdio_counts.get("camera", 0),
                npz_rel,
            )
            force_samples = sum(
                1
                for row in streamdio_rows
                if row.get("event_type") == "analog_input"
                and int(row.get("channel") or -1) == _force_channel(metadata)
            )
            _append_count_mismatch(
                errors,
                "manifest count force_samples",
                counts.get("force_samples"),
                force_samples,
                npz_rel,
            )
            frame2ttl_edges = sum(
                1
                for row in streamdio_rows
                if row.get("event_type") == "dio_input"
                and int(row.get("channel") or -1) == _frame2ttl_channel(metadata)
            )
            _append_count_mismatch(
                errors,
                "manifest count frame2ttl_edges",
                counts.get("frame2ttl_edges"),
                frame2ttl_edges,
                npz_rel,
            )
    raw_video = manifest.get("video")
    video: dict[str, Any] = dict(raw_video) if isinstance(raw_video, dict) else {}
    video_frames = _optional_int(video.get("n_frames")) or 0
    _append_count_mismatch(
        errors,
        "manifest count frames",
        counts.get("frames"),
        video_frames,
        "manifest video",
    )
    return errors


def _primary_session_json_payload(
    manifest: dict[str, Any],
    bundle_dir: Path,
) -> tuple[dict[str, Any] | None, str]:
    for item in manifest.get("artifacts") or []:
        if not isinstance(item, dict):
            continue
        rel = str(item.get("path") or "")
        if not rel.lower().endswith(".json"):
            continue
        path = bundle_dir / rel
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict) and {"summary", "trials", "events"}.issubset(payload):
            return payload, rel
    return None, "session json"


def _primary_npz_path(
    manifest: dict[str, Any],
    bundle_dir: Path,
) -> tuple[Path | None, str]:
    for item in manifest.get("artifacts") or []:
        if not isinstance(item, dict):
            continue
        rel = str(item.get("path") or "")
        normalized = rel.replace("\\", "/")
        if not rel.lower().endswith(".npz"):
            continue
        if normalized.startswith("continuous_streams/"):
            continue
        return bundle_dir / rel, rel
    return None, "session npz"


def _count_by_key(rows: list[Any], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        value = str(row.get(key))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _append_count_mismatch(
    errors: list[str],
    label: str,
    declared: Any,
    actual: int | None,
    source: str,
) -> None:
    expected = _optional_int(declared)
    if expected is None or actual is None:
        return
    if expected != int(actual):
        errors.append(f"{label} mismatch: {source}: {expected} != {int(actual)}")


def _append_mapping_mismatch(
    errors: list[str],
    label: str,
    declared: Any,
    actual: dict[str, Any],
    source: str,
) -> None:
    expected = _int_mapping(declared)
    actual_int = _int_mapping(actual)
    if expected is None or actual_int is None:
        return
    if expected != actual_int:
        errors.append(f"{label} mismatch: {source}: {expected} != {actual_int}")


def _int_mapping(value: Any) -> dict[str, int] | None:
    if not isinstance(value, dict):
        return None
    output: dict[str, int] = {}
    for key, raw in value.items():
        count = _optional_int(raw)
        if count is None:
            return None
        output[str(key)] = count
    return dict(sorted(output.items()))


def _validate_continuous_stream_index(
    manifest: dict[str, Any],
    bundle_dir: Path,
) -> list[str]:
    index = manifest.get("continuous_streams")
    if index is None:
        return []
    if not isinstance(index, dict):
        return ["continuous stream index is not an object"]
    errors: list[str] = []
    chunks = index.get("chunks")
    if not isinstance(chunks, list):
        return ["continuous stream index missing chunks list"]
    declared_chunk_count = _optional_int(index.get("chunk_count"))
    if declared_chunk_count is not None and declared_chunk_count != len(chunks):
        errors.append(
            f"continuous stream chunk count mismatch: {declared_chunk_count} != {len(chunks)}"
        )
    actual_event_count = 0
    actual_counts_by_type: dict[str, int] = {}
    for chunk_index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            errors.append(f"continuous stream chunk is not an object: {chunk_index}")
            continue
        rel = chunk.get("path")
        if not isinstance(rel, str) or not rel:
            errors.append(f"continuous stream chunk missing path: {chunk_index}")
            continue
        path = (bundle_dir / rel).resolve()
        if not path.exists():
            errors.append(f"continuous stream chunk missing: {rel}")
            continue
        actual = _streamdio_chunk_summary(path, rel.replace("\\", "/"))
        if actual.get("error"):
            errors.append(f"continuous stream chunk unreadable: {rel}: {actual['error']}")
            continue
        actual_event_count += int(actual.get("event_count") or 0)
        for event_type, count in (actual.get("counts_by_type") or {}).items():
            actual_counts_by_type[str(event_type)] = (
                actual_counts_by_type.get(str(event_type), 0) + int(count)
            )
        _compare_chunk_summary(chunk, actual, rel, errors)

    declared_event_count = _optional_int(index.get("event_count"))
    if declared_event_count is not None and declared_event_count != actual_event_count:
        errors.append(
            f"continuous stream event count mismatch: "
            f"{declared_event_count} != {actual_event_count}"
        )
    manifest_streamdio_count = _optional_int(
        (manifest.get("counts") or {}).get("streamdio_events")
    )
    if manifest_streamdio_count is not None and actual_event_count != manifest_streamdio_count:
        errors.append(
            f"continuous stream manifest event count mismatch: "
            f"{actual_event_count} != {manifest_streamdio_count}"
        )
    declared_counts = index.get("counts_by_type") or {}
    if declared_counts != dict(sorted(actual_counts_by_type.items())):
        errors.append("continuous stream counts_by_type mismatch")
    return errors


def _compare_chunk_summary(
    declared: dict[str, Any],
    actual: dict[str, Any],
    rel: str,
    errors: list[str],
) -> None:
    declared_count = _optional_int(declared.get("event_count"))
    actual_count = _optional_int(actual.get("event_count"))
    if (
        declared_count is not None
        and actual_count is not None
        and declared_count != actual_count
    ):
        errors.append(
            f"continuous stream chunk event count mismatch: {rel}: "
            f"{declared_count} != {actual_count}"
        )
    if (declared.get("counts_by_type") or {}) != (actual.get("counts_by_type") or {}):
        errors.append(f"continuous stream chunk counts_by_type mismatch: {rel}")
    for field in ("first_module_timestamp_us", "last_module_timestamp_us"):
        declared_value = _optional_int(declared.get(field))
        actual_value = _optional_int(actual.get(field))
        if declared_value != actual_value:
            errors.append(
                f"continuous stream chunk {field} mismatch: {rel}: "
                f"{declared_value} != {actual_value}"
            )
    for field in ("first_relative_time_s", "last_relative_time_s"):
        declared_time = _optional_float(declared.get(field))
        actual_time = _optional_float(actual.get(field))
        if declared_time is None and actual_time is None:
            continue
        if declared_time is None or actual_time is None:
            errors.append(
                f"continuous stream chunk {field} mismatch: {rel}: "
                f"{declared_time} != {actual_time}"
            )
            continue
        if abs(declared_time - actual_time) > 1e-9:
            errors.append(
                f"continuous stream chunk {field} mismatch: {rel}: "
                f"{declared_time} != {actual_time}"
            )


def _validate_recorder_stats(manifest: dict[str, Any]) -> list[str]:
    stats = manifest.get("recorder_stats")
    if stats is None:
        return []
    if not isinstance(stats, dict):
        return ["recorder_stats is not an object"]
    errors: list[str] = []
    artifacts = manifest.get("artifacts") or []
    if not isinstance(artifacts, list):
        artifacts = []

    expected_count = _optional_int(stats.get("bundle_artifact_count"))
    if expected_count is not None and expected_count != len(artifacts):
        errors.append(
            f"recorder_stats bundle_artifact_count mismatch: "
            f"{expected_count} != {len(artifacts)}"
        )

    if stats.get("bundle_artifact_bytes") is not None:
        expected_bytes = _optional_int(stats.get("bundle_artifact_bytes"))
        actual_bytes = 0
        bytes_complete = True
        for item in artifacts:
            if not isinstance(item, dict):
                bytes_complete = False
                continue
            item_bytes = _optional_int(item.get("bytes"))
            if item_bytes is None:
                bytes_complete = False
            else:
                actual_bytes += item_bytes
        if expected_bytes is None:
            errors.append(
                f"recorder_stats invalid bundle_artifact_bytes: "
                f"{stats.get('bundle_artifact_bytes')}"
            )
        elif bytes_complete and expected_bytes != actual_bytes:
            errors.append(
                f"recorder_stats bundle_artifact_bytes mismatch: "
                f"{expected_bytes} != {actual_bytes}"
            )

    counts = manifest.get("counts") or {}
    streamdio_total = _optional_int(stats.get("streamdio_total_count"))
    manifest_streamdio = _optional_int(counts.get("streamdio_events"))
    if (
        streamdio_total is not None
        and manifest_streamdio is not None
        and streamdio_total != manifest_streamdio
    ):
        errors.append(
            f"recorder_stats streamdio_total_count mismatch: "
            f"{streamdio_total} != {manifest_streamdio}"
        )

    recent = _optional_int(stats.get("debug_event_recent_count"))
    total = _optional_int(stats.get("debug_event_total_count"))
    evicted = _optional_int(stats.get("debug_event_evicted_count"))
    capacity = _optional_int(stats.get("debug_event_ring_capacity"))
    if recent is not None and capacity is not None and recent > capacity:
        errors.append(
            f"recorder_stats debug_event_recent_count exceeds capacity: "
            f"{recent} > {capacity}"
        )
    if recent is not None and total is not None and recent > total:
        errors.append(
            f"recorder_stats debug_event_recent_count exceeds total: "
            f"{recent} > {total}"
        )
    if recent is not None and total is not None and evicted is not None:
        if recent + evicted != total:
            errors.append(
                f"recorder_stats debug event accounting mismatch: "
                f"{recent} + {evicted} != {total}"
            )

    chunk_count = _optional_int(stats.get("streamdio_chunk_count"))
    continuous = manifest.get("continuous_streams")
    if chunk_count is not None and isinstance(continuous, dict):
        indexed_chunks = _optional_int(continuous.get("chunk_count"))
        if indexed_chunks is not None and chunk_count != indexed_chunks:
            errors.append(
                f"recorder_stats streamdio_chunk_count mismatch: "
                f"{chunk_count} != {indexed_chunks}"
            )
    return errors


def _validate_npz_artifact(path: Path, rel: str, manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required_primary = {
        "trials",
        "events",
        "streamdio_events",
        "bpod_trials_json",
        "trials_json",
        "events_json",
        "fsm_definitions_json",
        "summary_json",
        "session_metadata_json",
    }
    is_chunk = "continuous_streams" in rel.replace("\\", "/") or path.name.startswith(
        "streamdio_chunk_"
    )
    try:
        with np.load(path, allow_pickle=True) as data:
            files = set(data.files)
            if is_chunk:
                if "streamdio_events" not in files:
                    errors.append(f"chunk npz missing streamdio_events: {rel}")
                return errors
            missing = sorted(required_primary - files)
            if missing:
                errors.append(f"primary npz missing arrays: {rel}: {missing}")
                return errors
            if "timing_heartbeat_qc_json" in files:
                try:
                    npz_qc = json.loads(str(data["timing_heartbeat_qc_json"][0]))
                except Exception as exc:
                    errors.append(
                        f"primary npz invalid timing_heartbeat_qc_json: "
                        f"{rel}: {type(exc).__name__}: {exc}"
                    )
                else:
                    manifest_qc = (manifest.get("streamdio") or {}).get(
                        "timing_heartbeat_qc"
                    )
                    if isinstance(manifest_qc, dict):
                        for field in (
                            "overall_status",
                            "heartbeat_count",
                            "interval_count",
                            "estimated_missed_pulses",
                        ):
                            if npz_qc.get(field) != manifest_qc.get(field):
                                errors.append(
                                    f"primary npz timing heartbeat {field} mismatch: {rel}"
                                )
            counts = manifest.get("counts") or {}
            trial_count = counts.get("trials")
            if trial_count is not None and len(data["trials"]) != int(trial_count):
                errors.append(
                    f"primary npz trial count mismatch: {rel}: "
                    f"{len(data['trials'])} != {int(trial_count)}"
                )
            streamdio_count = counts.get("streamdio_events")
            if streamdio_count is not None and len(data["streamdio_events"]) != int(streamdio_count):
                errors.append(
                    f"primary npz streamdio count mismatch: {rel}: "
                    f"{len(data['streamdio_events'])} != {int(streamdio_count)}"
                )
            if len(data["summary_json"]) < 1:
                errors.append(f"primary npz missing summary row: {rel}")
    except Exception as exc:
        errors.append(f"npz unreadable: {rel}: {type(exc).__name__}: {exc}")
    return errors


def _validate_json_artifact(path: Path, rel: str, manifest: dict[str, Any]) -> list[str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"json unreadable: {rel}: {type(exc).__name__}: {exc}"]
    if _relative_matches(rel, "exports/debug_events_recent.json"):
        return _validate_debug_events_recent_json(payload, rel, manifest)
    if _relative_matches(rel, "exports/rig_preflight_report.json"):
        return _validate_rig_preflight_report_json(payload, rel, manifest)
    if _relative_matches(rel, "exports/streamdio_timing_qc.json"):
        return _validate_timing_heartbeat_qc_json(payload, rel, manifest)
    if not isinstance(payload, dict):
        return []
    if _relative_matches(rel, "bpod_data/bpod_trials.json"):
        return _validate_bpod_data_json(payload, rel, manifest)
    if _relative_matches(rel, "videos/spinnaker_video.json"):
        return _validate_spinnaker_video_pointer(payload, rel, manifest)
    if _relative_name(rel) == "recording_summary.json":
        return _validate_spinnaker_recording_summary(payload, rel, manifest)
    if _relative_matches(rel, "exports/streamdio_module_info.json"):
        return _validate_streamdio_module_info_json(payload, rel, manifest)
    if _relative_matches(rel, "exports/stream_summaries.json"):
        return _validate_stream_summaries_json(payload, rel, manifest)
    if _relative_matches(rel, "exports/soundcard_status.json"):
        return _validate_soundcard_status_json(payload, rel)
    if _relative_matches(rel, "exports/alignment_anchors.json"):
        return _validate_alignment_anchors_json(payload, rel, manifest)
    if _relative_matches(rel, "exports/alyx_registration.json"):
        return _validate_alyx_registration_json(payload, rel)
    if _relative_matches(rel, "exports/preview_video_export.json"):
        return _validate_preview_video_export_json(payload, rel, manifest)
    if _relative_matches(rel, "exports/preview_video_error.json"):
        return _validate_preview_video_error_json(payload, rel)
    if not {"summary", "trials", "events"}.issubset(payload):
        return []
    errors: list[str] = []
    counts = manifest.get("counts") or {}
    trial_count = counts.get("trials")
    trials = payload.get("trials")
    if trial_count is not None and isinstance(trials, list) and len(trials) != int(trial_count):
        errors.append(
            f"session json trial count mismatch: {rel}: {len(trials)} != {int(trial_count)}"
        )
    streamdio = payload.get("streamdio")
    if isinstance(streamdio, dict):
        streamdio_count = counts.get("streamdio_events")
        event_count = streamdio.get("event_count")
        if streamdio_count is not None and event_count is not None:
            if int(event_count) != int(streamdio_count):
                errors.append(
                    f"session json streamdio count mismatch: {rel}: "
                    f"{int(event_count)} != {int(streamdio_count)}"
                )
    return errors


def _validate_jsonl_artifact(path: Path, rel: str, manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    line_count = 0
    counts_by_kind: dict[str, int] = {}
    counts_by_message_type: dict[str, int] = {}
    is_monitor_telemetry = _relative_matches(rel, "exports/monitor_telemetry.jsonl")
    is_session_journal = path.name.endswith(".journal.jsonl")
    journal_counts: dict[str, int] = {}
    journal_summary: dict[str, Any] | None = None
    expected_sequence = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                line_count += 1
                try:
                    row = json.loads(text)
                except json.JSONDecodeError as exc:
                    errors.append(f"jsonl invalid json: {rel}:{line_number}: {exc.msg}")
                    continue
                if not isinstance(row, dict):
                    errors.append(f"jsonl row is not an object: {rel}:{line_number}")
                    continue
                if is_monitor_telemetry:
                    errors.extend(
                        _validate_monitor_telemetry_jsonl_row(row, rel, line_number)
                    )
                    kind = row.get("kind")
                    if isinstance(kind, str) and kind:
                        counts_by_kind[kind] = counts_by_kind.get(kind, 0) + 1
                    message = row.get("message")
                    if isinstance(message, dict):
                        message_type = message.get("type")
                        if isinstance(message_type, str) and message_type:
                            counts_by_message_type[message_type] = (
                                counts_by_message_type.get(message_type, 0) + 1
                            )
                if is_session_journal:
                    errors.extend(
                        _validate_session_journal_row(
                            row,
                            rel,
                            line_number,
                            expected_sequence=expected_sequence,
                        )
                    )
                    kind = row.get("kind")
                    if isinstance(kind, str):
                        journal_counts[kind] = journal_counts.get(kind, 0) + 1
                        if kind == "session_summary" and isinstance(row.get("payload"), dict):
                            journal_summary = dict(row["payload"])
                    expected_sequence += 1
    except Exception as exc:
        return [f"jsonl unreadable: {rel}: {type(exc).__name__}: {exc}"]
    if is_monitor_telemetry:
        telemetry = manifest.get("monitor_telemetry") or {}
        expected = _optional_int(telemetry.get("line_count"))
        if expected is not None and line_count != expected:
            errors.append(
                f"monitor telemetry row count mismatch: {rel}: {line_count} != {expected}"
            )
        _append_mapping_mismatch(
            errors,
            "monitor telemetry counts_by_kind",
            telemetry.get("counts_by_kind"),
            counts_by_kind,
            rel,
        )
        _append_mapping_mismatch(
            errors,
            "monitor telemetry counts_by_message_type",
            telemetry.get("counts_by_message_type"),
            counts_by_message_type,
            rel,
        )
    if is_session_journal:
        if journal_counts.get("session_header") != 1:
            errors.append(f"session journal must contain one header: {rel}")
        if journal_counts.get("session_summary") != 1:
            errors.append(f"session journal must contain one summary: {rel}")
        manifest_counts = manifest.get("counts") or {}
        for kind, manifest_key in (
            ("trial", "trials"),
            ("event", "events"),
            ("bpod_trial", "bpod_trials"),
        ):
            expected = _optional_int(manifest_counts.get(manifest_key))
            actual = journal_counts.get(kind, 0)
            if expected is not None and actual != expected:
                errors.append(
                    f"session journal {kind} count mismatch: {rel}: {actual} != {expected}"
                )
        manifest_summary = manifest.get("summary") or {}
        if journal_summary is not None:
            for field in ("subject_id", "phase", "n_trials_completed", "end_reason"):
                if field in manifest_summary and journal_summary.get(field) != manifest_summary.get(field):
                    errors.append(f"session journal summary mismatch for {field}: {rel}")
    return errors


def _validate_session_journal_row(
    row: dict[str, Any],
    rel: str,
    line_number: int,
    *,
    expected_sequence: int,
) -> list[str]:
    errors: list[str] = []
    if row.get("schema_version") != 1:
        errors.append(f"session journal schema mismatch: {rel}:{line_number}")
    if row.get("sequence") != expected_sequence:
        errors.append(
            f"session journal sequence mismatch: {rel}:{line_number}: "
            f"{row.get('sequence')} != {expected_sequence}"
        )
    kind = row.get("kind")
    allowed = {
        "session_header",
        "metadata",
        "export_options",
        "stream_summary",
        "fsm_definition",
        "event",
        "trial",
        "bpod_trial",
        "session_summary",
    }
    if kind not in allowed:
        errors.append(f"session journal invalid kind: {rel}:{line_number}: {kind}")
    if not isinstance(row.get("recorded_wall_s"), (int, float)):
        errors.append(f"session journal invalid recorded_wall_s: {rel}:{line_number}")
    if not isinstance(row.get("payload"), dict):
        errors.append(f"session journal missing payload object: {rel}:{line_number}")
    if expected_sequence == 0 and kind != "session_header":
        errors.append(f"session journal first row is not a header: {rel}:{line_number}")
    return errors


def _validate_monitor_telemetry_jsonl_row(
    row: dict[str, Any],
    rel: str,
    line_number: int,
) -> list[str]:
    errors: list[str] = []
    kind = row.get("kind")
    if not isinstance(kind, str) or not kind:
        errors.append(f"monitor telemetry missing kind: {rel}:{line_number}")
    for field in ("record_wall_s", "record_mono_s"):
        value = row.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(f"monitor telemetry invalid {field}: {rel}:{line_number}")
    message = row.get("message")
    if not isinstance(message, dict):
        errors.append(f"monitor telemetry missing message object: {rel}:{line_number}")
        return errors
    message_type = message.get("type")
    if not isinstance(message_type, str) or not message_type:
        errors.append(f"monitor telemetry missing message type: {rel}:{line_number}")
    payload = message.get("payload")
    if payload is not None and not isinstance(payload, dict):
        errors.append(f"monitor telemetry invalid payload: {rel}:{line_number}")
    return errors


def _validate_csv_artifact(path: Path, rel: str, manifest: dict[str, Any]) -> list[str]:
    try:
        row_count = _csv_row_count(path)
    except Exception as exc:
        return [f"csv unreadable: {rel}: {type(exc).__name__}: {exc}"]
    errors: list[str] = []
    if _relative_matches(rel, "exports/synchronized_timeline.csv"):
        errors.extend(_validate_timeline_csv(path, rel))
    counts = manifest.get("counts") or {}
    expected: int | None = None
    if _relative_matches(rel, "exports/trials_summary.csv"):
        expected = _optional_int(counts.get("trials"))
    elif _relative_matches(rel, "exports/events.csv"):
        expected = _optional_int(counts.get("events"))
    elif _relative_matches(rel, "exports/streamdio_events.csv"):
        expected = _optional_int(counts.get("streamdio_events"))
    elif _relative_matches(rel, "exports/force_trace.csv"):
        expected = _optional_int(counts.get("force_samples"))
    elif _relative_matches(rel, "exports/camera_ttl.csv"):
        expected = _optional_int(counts.get("camera_ttls"))
    elif _relative_matches(rel, "exports/frame2ttl_edges.csv"):
        expected = _optional_int(counts.get("frame2ttl_edges"))
    elif _relative_matches(rel, "exports/streamdio_timing_heartbeat.csv"):
        expected = _optional_int(counts.get("timing_heartbeats"))
        errors.extend(_validate_timing_heartbeat_csv_header(path, rel))
    elif _relative_matches(rel, "exports/licks.csv"):
        event_counts = counts.get("session_events_by_type") or {}
        expected = _optional_int(event_counts.get("lick") or 0) or 0
        expected += _optional_int(event_counts.get("lick_edge") or 0) or 0
    elif _relative_matches(rel, "bpod_exports/bpod_trials_flat.csv"):
        expected = _optional_int(counts.get("bpod_trial_rows"))
    elif _relative_name(rel) == "frame_timestamps.csv":
        expected = _optional_int((manifest.get("video") or {}).get("n_frames"))
    if expected is None:
        return errors
    if row_count != expected:
        errors.append(f"csv row count mismatch: {rel}: {row_count} != {expected}")
    return errors


def _csv_row_count(path: Path) -> int:
    csv.field_size_limit(max(csv.field_size_limit(), _CSV_VALIDATION_FIELD_LIMIT))
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return sum(1 for _ in reader)


def _validate_timeline_csv(path: Path, rel: str) -> list[str]:
    errors: list[str] = []
    csv.field_size_limit(max(csv.field_size_limit(), _CSV_VALIDATION_FIELD_LIMIT))
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [name for name in TIMELINE_COLUMNS if name not in fieldnames]
        if missing:
            return [f"timeline csv missing columns: {rel}: {missing}"]
        previous_time: float | None = None
        for row_index, row in enumerate(reader, start=1):
            session_mono = row.get("session_mono_s")
            if session_mono not in (None, ""):
                try:
                    current_time = float(str(session_mono))
                except ValueError:
                    errors.append(
                        f"timeline csv invalid session_mono_s: {rel}:{row_index}: "
                        f"{session_mono}"
                    )
                else:
                    if previous_time is not None and current_time < previous_time:
                        errors.append(
                            f"timeline csv not sorted by session_mono_s: {rel}:{row_index}"
                        )
                    previous_time = current_time
            detail_json = row.get("detail_json")
            if detail_json not in (None, ""):
                try:
                    json.loads(str(detail_json))
                except json.JSONDecodeError as exc:
                    errors.append(
                        f"timeline csv invalid detail_json: {rel}:{row_index}: {exc.msg}"
                    )
    return errors


def _validate_timing_heartbeat_consistency(
    manifest: dict[str, Any],
    bundle_dir: Path,
) -> list[str]:
    declared = (manifest.get("streamdio") or {}).get("timing_heartbeat_qc")
    if declared is None:
        return []
    if not isinstance(declared, dict):
        return ["manifest timing heartbeat QC is not an object"]
    npz_path, rel = _primary_npz_path(manifest, bundle_dir)
    if npz_path is None or not npz_path.exists():
        return ["timing heartbeat QC cannot be verified without primary NPZ"]
    try:
        rows = _load_npz_streamdio(npz_path)
    except Exception as exc:
        return [
            f"timing heartbeat QC source unreadable: {rel}: "
            f"{type(exc).__name__}: {exc}"
        ]
    actual = timing_heartbeat_qc_summary(analyze_timing_heartbeats(rows))
    errors: list[str] = []
    for field in (
        "overall_status",
        "heartbeat_count",
        "interval_count",
        "estimated_missed_pulses",
        "first_teensy_timestamp_us",
        "last_teensy_timestamp_us",
        "teensy_span_s",
        "first_pc_perf_counter_s",
        "last_pc_perf_counter_s",
        "pc_span_s",
        "final_pc_minus_teensy_elapsed_s",
        "max_abs_pc_minus_teensy_elapsed_s",
        "pc_vs_teensy_rate_error_ppm",
        "statistics",
        "clock_fit",
        "thresholds",
        "metrics",
    ):
        if declared.get(field) != actual.get(field):
            errors.append(f"timing heartbeat QC does not match raw StreamDIO data: {field}")
    return errors


def _validate_timing_heartbeat_csv_header(path: Path, rel: str) -> list[str]:
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            fieldnames = csv.DictReader(handle).fieldnames or []
    except Exception as exc:
        return [f"timing heartbeat csv unreadable: {rel}: {type(exc).__name__}: {exc}"]
    missing = [name for name in TIMING_HEARTBEAT_COLUMNS if name not in fieldnames]
    return [] if not missing else [f"timing heartbeat csv missing columns: {rel}: {missing}"]


def _validate_timing_heartbeat_qc_json(
    payload: Any,
    rel: str,
    manifest: dict[str, Any],
) -> list[str]:
    if not isinstance(payload, dict):
        return [f"timing heartbeat QC is not an object: {rel}"]
    errors: list[str] = []
    if payload.get("schema_version") != 1:
        errors.append(f"timing heartbeat QC schema mismatch: {rel}")
    if payload.get("kind") != "hf_dm_streamdio_timing_heartbeat_qc":
        errors.append(f"timing heartbeat QC kind mismatch: {rel}")
    if payload.get("event_type") != TIMING_HEARTBEAT_EVENT_TYPE:
        errors.append(f"timing heartbeat QC event type mismatch: {rel}")
    if payload.get("overall_status") not in {"PASS", "WARN", "FAIL"}:
        errors.append(f"timing heartbeat QC invalid overall_status: {rel}")
    if payload.get("timestamps_artifact") != "exports/streamdio_timing_heartbeat.csv":
        errors.append(f"timing heartbeat QC timestamps artifact mismatch: {rel}")
    metrics = payload.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        errors.append(f"timing heartbeat QC missing metrics: {rel}")
    else:
        required_metrics = {
            "heartbeat_count",
            "estimated_missed_pulses",
            "teensy_interval_max_abs_error_s",
            "pc_interval_p99_abs_error_s",
            "pc_interval_max_abs_error_s",
            "pc_delivery_detrended_max_abs_s",
            "pc_vs_teensy_rate_error_ppm",
            "teensy_nonmonotonic_intervals",
            "pc_nonmonotonic_intervals",
        }
        names = {
            str(metric.get("name"))
            for metric in metrics
            if isinstance(metric, dict)
        }
        missing = sorted(required_metrics - names)
        if missing:
            errors.append(f"timing heartbeat QC missing metric names: {rel}: {missing}")
        for index, metric in enumerate(metrics):
            if not isinstance(metric, dict):
                errors.append(f"timing heartbeat QC metric is not an object: {rel}:{index}")
                continue
            if metric.get("status") not in {"PASS", "WARN", "FAIL", "N/A"}:
                errors.append(f"timing heartbeat QC metric status invalid: {rel}:{index}")
    heartbeat_count = _optional_int(payload.get("heartbeat_count"))
    manifest_count = _optional_int((manifest.get("counts") or {}).get("timing_heartbeats"))
    if heartbeat_count is None:
        errors.append(f"timing heartbeat QC invalid heartbeat_count: {rel}")
    elif manifest_count is not None and heartbeat_count != manifest_count:
        errors.append(
            f"timing heartbeat QC count mismatch: {rel}: "
            f"{heartbeat_count} != {manifest_count}"
        )
    manifest_qc = (manifest.get("streamdio") or {}).get("timing_heartbeat_qc")
    if not isinstance(manifest_qc, dict):
        errors.append(f"manifest missing timing heartbeat QC object: {rel}")
    else:
        for field in (
            "overall_status",
            "heartbeat_count",
            "interval_count",
            "estimated_missed_pulses",
            "first_teensy_timestamp_us",
            "last_teensy_timestamp_us",
        ):
            if payload.get(field) != manifest_qc.get(field):
                errors.append(f"manifest timing heartbeat {field} mismatch: {rel}")
    return errors


def _validate_bpod_data_json(
    payload: dict[str, Any],
    rel: str,
    manifest: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    trials = payload.get("bpod_trials")
    if not isinstance(trials, list):
        errors.append(f"bpod data missing bpod_trials list: {rel}")
        return errors
    declared = payload.get("trial_count")
    if declared is not None and int(declared) != len(trials):
        errors.append(
            f"bpod data trial count mismatch: {rel}: {int(declared)} != {len(trials)}"
        )
    expected = (manifest.get("counts") or {}).get("bpod_trials")
    if expected is not None and len(trials) != int(expected):
        errors.append(
            f"bpod data manifest trial count mismatch: {rel}: "
            f"{len(trials)} != {int(expected)}"
        )
    row_count = _bpod_trial_row_count(trials)
    expected_rows = (manifest.get("counts") or {}).get("bpod_trial_rows")
    if expected_rows is not None and row_count != int(expected_rows):
        errors.append(
            f"bpod data manifest row count mismatch: {rel}: "
            f"{row_count} != {int(expected_rows)}"
        )
    fsm_rows = payload.get("fsm_definitions") or []
    fsm_declared = payload.get("fsm_definition_count")
    if fsm_declared is not None and int(fsm_declared) != len(fsm_rows):
        errors.append(
            f"bpod data fsm count mismatch: {rel}: "
            f"{int(fsm_declared)} != {len(fsm_rows)}"
        )
    return errors


def _validate_streamdio_module_info_json(
    payload: dict[str, Any],
    rel: str,
    manifest: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    manifest_streamdio = manifest.get("streamdio") or {}
    manifest_info = manifest_streamdio.get("module_info") or {}
    manifest_firmware = manifest_streamdio.get("firmware_version")
    payload_firmware = payload.get("firmware_version")
    if manifest_firmware is not None:
        if payload_firmware is None:
            errors.append(f"streamdio module info missing firmware_version: {rel}")
        elif str(payload_firmware) != str(manifest_firmware):
            errors.append(
                f"streamdio module info firmware mismatch: {rel}: "
                f"{payload_firmware} != {manifest_firmware}"
            )
    if manifest_info and payload != manifest_info:
        payload_without_fw = dict(payload)
        manifest_without_fw = dict(manifest_info)
        payload_without_fw.pop("firmware_version", None)
        manifest_without_fw.pop("firmware_version", None)
        if payload_without_fw != manifest_without_fw:
            errors.append(f"streamdio module info manifest mismatch: {rel}")
    return errors


def _validate_stream_summaries_json(
    payload: dict[str, Any],
    rel: str,
    manifest: dict[str, Any],
) -> list[str]:
    expected = manifest.get("stream_summaries") or {}
    if payload != expected:
        return [f"stream summaries manifest mismatch: {rel}"]
    return []


def _validate_soundcard_status_json(payload: dict[str, Any], rel: str) -> list[str]:
    errors: list[str] = []
    snapshots = payload.get("snapshots")
    if not isinstance(snapshots, list):
        return [f"soundcard status missing snapshots list: {rel}"]
    declared = payload.get("snapshot_count")
    if declared is None:
        errors.append(f"soundcard status missing snapshot_count: {rel}")
    else:
        try:
            declared_count = int(declared)
        except (TypeError, ValueError):
            errors.append(f"soundcard status invalid snapshot_count: {rel}: {declared}")
        else:
            if declared_count != len(snapshots):
                errors.append(
                    f"soundcard status snapshot count mismatch: {rel}: "
                    f"{declared_count} != {len(snapshots)}"
                )
    for index, snapshot in enumerate(snapshots):
        if not isinstance(snapshot, dict):
            errors.append(f"soundcard status snapshot is not an object: {rel}[{index}]")
            continue
        if not isinstance(snapshot.get("status"), dict):
            errors.append(
                f"soundcard status snapshot missing status object: {rel}[{index}]"
            )
    latest_status = payload.get("latest_status")
    if snapshots:
        last = snapshots[-1] if isinstance(snapshots[-1], dict) else {}
        last_status = last.get("status") if isinstance(last, dict) else None
        if isinstance(last_status, dict) and latest_status != last_status:
            errors.append(f"soundcard status latest_status mismatch: {rel}")
        if isinstance(last_status, dict):
            for key in ("device_id", "volume", "continuous_playback_active"):
                if key in last_status and payload.get(key) != last_status.get(key):
                    errors.append(f"soundcard status {key} mismatch: {rel}")
    elif latest_status not in ({}, None):
        errors.append(f"soundcard status latest_status should be empty: {rel}")
    return errors


def _validate_debug_events_recent_json(
    payload: Any,
    rel: str,
    manifest: dict[str, Any],
) -> list[str]:
    if not isinstance(payload, list):
        return [f"debug events recent is not a list: {rel}"]
    errors: list[str] = []
    stats = manifest.get("recorder_stats") or {}
    expected_recent = _optional_int(stats.get("debug_event_recent_count"))
    if expected_recent is not None and len(payload) != expected_recent:
        errors.append(
            f"debug events recent count mismatch: {rel}: "
            f"{len(payload)} != {expected_recent}"
        )
    ring_capacity = _optional_int(stats.get("debug_event_ring_capacity"))
    if ring_capacity is not None and len(payload) > ring_capacity:
        errors.append(
            f"debug events recent exceeds ring capacity: {rel}: "
            f"{len(payload)} > {ring_capacity}"
        )
    total_count = _optional_int(stats.get("debug_event_total_count"))
    if total_count is not None and len(payload) > total_count:
        errors.append(
            f"debug events recent exceeds total count: {rel}: "
            f"{len(payload)} > {total_count}"
        )
    for index, event in enumerate(payload):
        if not isinstance(event, dict):
            errors.append(f"debug event is not an object: {rel}[{index}]")
            continue
        event_type = event.get("event_type")
        if not isinstance(event_type, str) or not event_type:
            errors.append(f"debug event missing event_type: {rel}[{index}]")
        for field in ("wall_time_s", "monotonic_s"):
            value = event.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                errors.append(f"debug event invalid {field}: {rel}[{index}]")
        if not isinstance(event.get("payload"), dict):
            errors.append(f"debug event missing payload object: {rel}[{index}]")
    return errors


def _validate_rig_preflight_report_json(
    payload: Any,
    rel: str,
    manifest: dict[str, Any],
) -> list[str]:
    if not isinstance(payload, dict):
        return [f"rig preflight report is not an object: {rel}"]
    errors: list[str] = []
    if payload.get("schema_version") != 1:
        errors.append(f"rig preflight report schema_version mismatch: {rel}")
    if payload.get("command") != "hf-dm rig-preflight":
        errors.append(f"rig preflight report command mismatch: {rel}")
    if _optional_float(payload.get("generated_wall_time_s")) is None:
        errors.append(f"rig preflight report missing generated_wall_time_s: {rel}")
    duration = _optional_float(payload.get("duration_s"))
    if duration is None or duration < 0:
        errors.append(f"rig preflight report invalid duration_s: {rel}")
    if not isinstance(payload.get("config_path"), str) or not payload.get("config_path"):
        errors.append(f"rig preflight report missing config_path: {rel}")
    if not isinstance(payload.get("monitor_enabled"), bool):
        errors.append(f"rig preflight report invalid monitor_enabled: {rel}")

    summary = payload.get("summary")
    manifest_summary = manifest.get("summary") or {}
    if not isinstance(summary, dict):
        errors.append(f"rig preflight report missing summary object: {rel}")
    else:
        for field in ("subject_id", "phase", "end_reason", "n_trials_completed"):
            if (
                field in manifest_summary
                and summary.get(field) != manifest_summary.get(field)
            ):
                errors.append(f"rig preflight report summary {field} mismatch: {rel}")

    session = payload.get("session")
    if not isinstance(session, dict):
        errors.append(f"rig preflight report missing session object: {rel}")
    else:
        for field in ("base_path", "npz", "json", "bundle_dir"):
            if not isinstance(session.get(field), str) or not session.get(field):
                errors.append(f"rig preflight report missing session {field}: {rel}")

    preflight_events = payload.get("preflight_events")
    if not isinstance(preflight_events, list):
        errors.append(f"rig preflight report missing preflight_events list: {rel}")
    else:
        event_counts = (manifest.get("counts") or {}).get("session_events_by_type") or {}
        expected = sum(
            int(_optional_int(count) or 0)
            for event_type, count in event_counts.items()
            if str(event_type).startswith("preflight")
        )
        if expected != len(preflight_events):
            errors.append(
                f"rig preflight report event count mismatch: {rel}: "
                f"{len(preflight_events)} != {expected}"
            )
        for index, event in enumerate(preflight_events):
            if not isinstance(event, dict):
                errors.append(f"rig preflight event is not an object: {rel}[{index}]")
                continue
            event_type = event.get("event_type")
            if not isinstance(event_type, str) or not event_type.startswith("preflight"):
                errors.append(f"rig preflight event_type mismatch: {rel}[{index}]")
            if not isinstance(event.get("payload"), dict):
                errors.append(
                    f"rig preflight event missing payload object: {rel}[{index}]"
                )

    streamdio = payload.get("streamdio")
    if not isinstance(streamdio, dict):
        errors.append(f"rig preflight report missing streamdio object: {rel}")
    else:
        manifest_counts = manifest.get("counts") or {}
        expected_events = _optional_int(manifest_counts.get("streamdio_events"))
        report_events = _optional_int(streamdio.get("event_count"))
        if expected_events is not None and report_events != expected_events:
            errors.append(
                f"rig preflight streamdio event count mismatch: {rel}: "
                f"{report_events} != {expected_events}"
            )
        report_counts = streamdio.get("counts_by_type")
        expected_counts = manifest_counts.get("streamdio_by_type") or {}
        if isinstance(report_counts, dict) and report_counts != expected_counts:
            errors.append(f"rig preflight streamdio counts_by_type mismatch: {rel}")
        elif not isinstance(report_counts, dict):
            errors.append(f"rig preflight streamdio missing counts_by_type: {rel}")
        if not isinstance(streamdio.get("module_status"), dict):
            errors.append(f"rig preflight streamdio missing module_status object: {rel}")
    validation = payload.get("validation")
    if not isinstance(validation, dict):
        errors.append(f"rig preflight report missing validation object: {rel}")
    else:
        status = validation.get("status")
        if status not in {"PASS", "WARN", "FAIL"}:
            errors.append(f"rig preflight report invalid validation status: {rel}")
        if not isinstance(validation.get("passed"), bool):
            errors.append(f"rig preflight report invalid validation passed flag: {rel}")
        results = validation.get("results")
        if not isinstance(results, list) or not results:
            errors.append(f"rig preflight report missing validation results: {rel}")
        else:
            result_statuses = set()
            for index, result in enumerate(results):
                if not isinstance(result, dict):
                    errors.append(
                        f"rig preflight validation result is not an object: {rel}[{index}]"
                    )
                    continue
                if not isinstance(result.get("name"), str) or not result.get("name"):
                    errors.append(f"rig preflight validation missing name: {rel}[{index}]")
                result_status = result.get("status")
                if result_status not in {"PASS", "WARN", "FAIL"}:
                    errors.append(f"rig preflight validation invalid status: {rel}[{index}]")
                else:
                    result_statuses.add(result_status)
                if not isinstance(result.get("detail"), str):
                    errors.append(f"rig preflight validation missing detail: {rel}[{index}]")
            expected_status = (
                "FAIL"
                if "FAIL" in result_statuses
                else "WARN"
                if "WARN" in result_statuses
                else "PASS"
            )
            if status != expected_status:
                errors.append(f"rig preflight validation aggregate mismatch: {rel}")
    return errors


def _validate_alignment_anchors_json(
    payload: dict[str, Any],
    rel: str,
    manifest: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    required_lists = {
        "bpod_state_markers": _event_count(manifest, "bpod_state_marker"),
        "choice_events": _event_count(manifest, "choice_detected"),
        "pump_and_output_events": (
            _event_count(manifest, "pump_start")
            + _event_count(manifest, "pump_end")
            + _event_count(manifest, "output_actual")
        ),
    }
    for key, expected_count in required_lists.items():
        value = payload.get(key)
        if not isinstance(value, list):
            errors.append(f"alignment anchors missing {key} list: {rel}")
            continue
        if len(value) != expected_count:
            errors.append(
                f"alignment anchors {key} count mismatch: {rel}: "
                f"{len(value)} != {expected_count}"
            )
        for index, item in enumerate(value):
            if not isinstance(item, dict):
                errors.append(f"alignment anchor {key} is not an object: {rel}[{index}]")
                continue
            if "event_type" not in item or "payload" not in item:
                errors.append(
                    f"alignment anchor {key} missing compact event fields: {rel}[{index}]"
                )

    if _optional_int((manifest.get("counts") or {}).get("camera_ttls")):
        if not isinstance(payload.get("first_camera_ttl"), dict):
            errors.append(f"alignment anchors missing first_camera_ttl: {rel}")
    if _optional_int((manifest.get("counts") or {}).get("frame2ttl_edges")):
        if not isinstance(payload.get("first_frame2ttl_edge"), dict):
            errors.append(f"alignment anchors missing first_frame2ttl_edge: {rel}")
    video = manifest.get("video") or {}
    if _optional_int(video.get("n_frames")) and video.get("frame_timestamps_csv"):
        if not isinstance(payload.get("first_spinnaker_frame"), dict):
            errors.append(f"alignment anchors missing first_spinnaker_frame: {rel}")
    return errors


def _validate_alyx_registration_json(payload: dict[str, Any], rel: str) -> list[str]:
    errors: list[str] = []
    status = payload.get("status")
    allowed = {"skipped", "registered", "error"}
    if status not in allowed:
        return [f"alyx registration invalid status: {rel}: {status}"]
    if status == "skipped":
        if not isinstance(payload.get("reason"), str) or not payload.get("reason"):
            errors.append(f"alyx registration skipped without reason: {rel}")
    elif status == "registered":
        if not isinstance(payload.get("endpoint"), str) or not payload.get("endpoint"):
            errors.append(f"alyx registration registered without endpoint: {rel}")
        if not isinstance(payload.get("request_payload"), dict):
            errors.append(f"alyx registration missing request_payload: {rel}")
        status_code = payload.get("status_code")
        if status_code is not None and _optional_int(status_code) is None:
            errors.append(f"alyx registration invalid status_code: {rel}")
    elif status == "error":
        if not isinstance(payload.get("error"), str) or not payload.get("error"):
            errors.append(f"alyx registration error without message: {rel}")
    return errors


def _validate_preview_video_export_json(
    payload: dict[str, Any],
    rel: str,
    manifest: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    for key in ("output_path", "video_dir", "codec", "source_summary"):
        if not isinstance(payload.get(key), str) or not payload.get(key):
            errors.append(f"preview video export missing {key}: {rel}")
    for key in ("frames_seen", "frames_written", "frames_skipped"):
        value = _optional_int(payload.get(key))
        if value is None or value < 0:
            errors.append(f"preview video export invalid {key}: {rel}")
    frames_seen = _optional_int(payload.get("frames_seen"))
    frames_written = _optional_int(payload.get("frames_written"))
    frames_skipped = _optional_int(payload.get("frames_skipped"))
    if (
        frames_seen is not None
        and frames_written is not None
        and frames_skipped is not None
        and frames_written + frames_skipped != frames_seen
    ):
        errors.append(
            f"preview video export frame accounting mismatch: {rel}: "
            f"{frames_written} + {frames_skipped} != {frames_seen}"
        )
    for key in ("fps", "frame_width", "frame_height"):
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            errors.append(f"preview video export invalid {key}: {rel}")
    manifest_preview = (manifest.get("video") or {}).get("preview_video")
    if isinstance(manifest_preview, dict):
        manifest_frames = _optional_int(manifest_preview.get("frames_written"))
        if (
            manifest_frames is not None
            and frames_written is not None
            and manifest_frames != frames_written
        ):
            errors.append(
                f"preview video export manifest frame mismatch: {rel}: "
                f"{frames_written} != {manifest_frames}"
            )
    return errors


def _validate_preview_video_error_json(payload: dict[str, Any], rel: str) -> list[str]:
    error = payload.get("error")
    if not isinstance(error, str) or not error:
        return [f"preview video error missing message: {rel}"]
    return []


def _event_count(manifest: dict[str, Any], event_type: str) -> int:
    event_counts = (manifest.get("counts") or {}).get("session_events_by_type") or {}
    return int(_optional_int(event_counts.get(event_type)) or 0)


def _validate_spinnaker_video_pointer(
    payload: dict[str, Any],
    rel: str,
    manifest: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    if payload.get("kind") != "spinnaker_video_pointer":
        errors.append(f"spinnaker video pointer kind mismatch: {rel}")
    video = payload.get("video")
    if not isinstance(video, dict):
        errors.append(f"spinnaker video pointer missing video object: {rel}")
        return errors
    manifest_video = manifest.get("video") or {}
    if video.get("n_frames") is not None and manifest_video.get("n_frames") is not None:
        if int(video["n_frames"]) != int(manifest_video["n_frames"]):
            errors.append(
                f"spinnaker video pointer frame count mismatch: {rel}: "
                f"{int(video['n_frames'])} != {int(manifest_video['n_frames'])}"
            )
    return errors


def _validate_spinnaker_recording_summary(
    payload: dict[str, Any],
    rel: str,
    manifest: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    manifest_video = manifest.get("video") or {}
    payload_frames = payload.get("n_frames")
    if payload_frames is None:
        payload_frames = payload.get("frames_saved")
    manifest_frames = manifest_video.get("n_frames")
    if payload_frames is not None and manifest_frames is not None:
        if int(payload_frames) != int(manifest_frames):
            errors.append(
                f"spinnaker recording summary frame count mismatch: {rel}: "
                f"{int(payload_frames)} != {int(manifest_frames)}"
            )
    payload_backend = payload.get("backend")
    manifest_backend = manifest_video.get("backend")
    if payload_backend and manifest_backend and str(payload_backend) != str(manifest_backend):
        errors.append(
            f"spinnaker recording summary backend mismatch: {rel}: "
            f"{payload_backend} != {manifest_backend}"
        )
    return errors


def _relative_matches(rel: str, expected: str) -> bool:
    return str(rel).replace("\\", "/") == expected


def _relative_name(rel: str) -> str:
    return Path(str(rel).replace("\\", "/")).name


def _find_manifest(path: Path) -> Path | None:
    if path.name == "manifest.json" and path.exists():
        return path
    if path.is_dir():
        direct = path / "manifest.json"
        if direct.exists():
            return direct
        manifests = sorted(path.glob("*_bundle/manifest.json"))
        return manifests[0] if manifests else None
    if path.suffix in {".json", ".npz"}:
        direct = session_bundle_dir(path.with_suffix(""), layout="flat") / "manifest.json"
        if direct.exists():
            return direct
        mm_direct = path.with_name("manifest.json")
        if mm_direct.exists():
            return mm_direct
    return None
