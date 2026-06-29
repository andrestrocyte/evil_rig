"""Operational helper CLI for cold-path session tools and rig checks."""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import yaml

from hf_dm.config import (
    ConfigValidationError,
    SessionConfig,
    config_to_dict,
    validate_session_config,
)
from hf_dm.data.recorder import recover_session_journal
from hf_dm.session_qc import QcThresholds, evaluate_session_qc, format_qc_table
from hf_dm.subject_log import append_weight_entry
from hf_dm.data.session_export import (
    export_existing_session,
    session_bundle_dir,
    validate_session,
)
from hf_dm.data.video_export import export_preview_video
from hf_dm.session.controller import HFDecisionSession
from hf_dm.valve_calibration import run_guided_valve_calibration
from hf_dm.validation import ValidationResult, validation_report


def main() -> None:
    parser = argparse.ArgumentParser(description="HF-DM operational helpers")
    sub = parser.add_subparsers(dest="command", required=True)

    export_parser = sub.add_parser("export-session", help="Regenerate derived session exports")
    export_parser.add_argument("session", type=Path, help="Session .json, .npz, or directory")
    export_parser.add_argument(
        "--export-preview-video",
        action="store_true",
        help="Also create an offline FFV1 preview AVI from saved Spinnaker PGM frames",
    )
    export_parser.add_argument("--preview-output", type=Path, default=None)
    export_parser.add_argument("--preview-fps", type=float, default=None)
    export_parser.add_argument("--preview-codec", default="FFV1")
    export_parser.add_argument("--preview-max-frames", type=int, default=None)

    validate_parser = sub.add_parser("validate-session", help="Validate a session manifest")
    validate_parser.add_argument("session", type=Path, help="Session bundle, manifest, .json, or .npz")

    config_parser = sub.add_parser(
        "validate-config",
        help="Validate task settings without opening hardware",
    )
    config_parser.add_argument("config", type=Path, help="Session YAML configuration")

    recover_parser = sub.add_parser(
        "recover-session",
        help="Finalize an interrupted session from its durable journal",
    )
    recover_parser.add_argument(
        "session",
        type=Path,
        help="Session directory, incomplete marker, or .journal.jsonl file",
    )
    recover_parser.add_argument("--end-reason", default="recovered_after_crash")

    qc_parser = sub.add_parser(
        "session-qc",
        help="Evaluate curriculum and acquisition QC for a finalized session",
    )
    qc_parser.add_argument("session", type=Path, help="Session bundle, manifest, .json, or .npz")
    qc_parser.add_argument("--min-responded-rate", type=float, default=0.50)
    qc_parser.add_argument("--warn-responded-rate", type=float, default=0.40)
    qc_parser.add_argument("--min-matched-rate", type=float, default=0.55)
    qc_parser.add_argument("--warn-matched-rate", type=float, default=0.45)
    qc_parser.add_argument("--max-mean-response-time", type=float, default=12.0)
    qc_parser.add_argument("--warn-mean-response-time", type=float, default=15.0)
    qc_parser.add_argument("--max-stalls", type=int, default=0)
    qc_parser.add_argument("--warn-stalls", type=int, default=2)
    qc_parser.add_argument("--max-frame-time", type=float, default=0.10)
    qc_parser.add_argument("--warn-frame-time", type=float, default=0.20)

    valve_parser = sub.add_parser(
        "calibrate-valve",
        help="Run a guided StreamDIO reward-valve calibration",
    )
    valve_parser.add_argument("--config", type=Path, required=True)
    valve_parser.add_argument(
        "--durations-ms",
        default="20,40,60,80,100",
        help="Comma-separated valve opening durations in milliseconds",
    )
    valve_parser.add_argument("--repetitions", type=int, default=10)
    valve_parser.add_argument("--output", type=Path, default=None)
    valve_parser.add_argument("--water-density", type=float, default=1.0)
    valve_parser.add_argument("--inter-pulse", type=float, default=0.10)
    valve_parser.add_argument("--overwrite", action="store_true")
    valve_parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm the rig is clear and a collection vessel is in place",
    )

    weight_parser = sub.add_parser(
        "log-weight",
        help="Append a local subject weight/water-target record",
    )
    weight_parser.add_argument("subject")
    weight_parser.add_argument("--weight", type=float, required=True)
    weight_parser.add_argument("--water-target", type=float, default=None)
    weight_parser.add_argument("--date", default=None)
    weight_parser.add_argument("--subjects-dir", type=Path, default=Path("subjects"))
    weight_parser.add_argument("--notes", default=None)

    preview_parser = sub.add_parser(
        "export-preview-video",
        help="Create an offline FFV1 preview AVI from saved Spinnaker PGM frames",
    )
    preview_parser.add_argument("session", type=Path, help="Video dir, recording summary, manifest, or session JSON")
    preview_parser.add_argument("--output", type=Path, default=None)
    preview_parser.add_argument("--fps", type=float, default=None)
    preview_parser.add_argument("--codec", default="FFV1")
    preview_parser.add_argument("--max-frames", type=int, default=None)

    preflight_parser = sub.add_parser("rig-preflight", help="Run no-trial hardware preflight")
    preflight_parser.add_argument("--config", type=Path, required=True)
    preflight_parser.add_argument("--duration", type=float, default=2.0)
    preflight_parser.add_argument("--monitor", action="store_true")
    preflight_parser.add_argument("--no-monitor", action="store_true")
    preflight_parser.add_argument("-v", "--verbose", action="store_true")

    roi_parser = sub.add_parser("camera-roi-setup", help="Select or write Spinnaker ROI fields")
    roi_parser.add_argument("--config", type=Path, required=True)
    roi_parser.add_argument("--output", type=Path, default=None)
    roi_parser.add_argument("--width", type=int, default=None)
    roi_parser.add_argument("--height", type=int, default=None)
    roi_parser.add_argument("--offset-x", type=int, default=None)
    roi_parser.add_argument("--offset-y", type=int, default=None)
    roi_parser.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt for missing ROI fields using the current YAML values as defaults",
    )
    roi_parser.add_argument(
        "--visual",
        action="store_true",
        help="Acquire a full FLIR frame and select the ROI visually",
    )
    roi_parser.add_argument("--clear", action="store_true", help="Remove ROI fields from the YAML")

    args = parser.parse_args()
    if args.command == "export-session":
        result = export_existing_session(args.session)
        if args.export_preview_video:
            result["preview_video"] = export_preview_video(
                args.session,
                output=args.preview_output,
                fps=args.preview_fps,
                codec=args.preview_codec,
                max_frames=args.preview_max_frames,
            )
        print(json.dumps(result, indent=2))
        return
    if args.command == "validate-session":
        result = validate_session(args.session)
        print(json.dumps(result, indent=2))
        if not result["valid"]:
            raise SystemExit(1)
        return
    if args.command == "validate-config":
        try:
            config = validate_session_config(SessionConfig.from_yaml(args.config))
            result = {
                "valid": True,
                "config": str(args.config),
                "effective": config_to_dict(config),
                "errors": [],
            }
        except ConfigValidationError as exc:
            result = {
                "valid": False,
                "config": str(args.config),
                "errors": [asdict(issue) for issue in exc.issues],
            }
        except Exception as exc:
            result = {
                "valid": False,
                "config": str(args.config),
                "errors": [
                    {
                        "field": "config",
                        "message": f"{type(exc).__name__}: {exc}",
                        "value": None,
                    }
                ],
            }
        print(json.dumps(result, indent=2))
        if not result["valid"]:
            raise SystemExit(1)
        return
    if args.command == "recover-session":
        result = recover_session_journal(
            args.session,
            end_reason=args.end_reason,
        )
        print(json.dumps(result, indent=2))
        if not result["validation"]["valid"]:
            raise SystemExit(1)
        return
    if args.command == "session-qc":
        thresholds = QcThresholds(
            min_responded_rate=args.min_responded_rate,
            warn_responded_rate=args.warn_responded_rate,
            min_matched_responded_rate=args.min_matched_rate,
            warn_matched_responded_rate=args.warn_matched_rate,
            max_mean_response_time_s=args.max_mean_response_time,
            warn_mean_response_time_s=args.warn_mean_response_time,
            max_choice_loop_stalls=args.max_stalls,
            warn_choice_loop_stalls=args.warn_stalls,
            max_choice_loop_frame_s=args.max_frame_time,
            warn_choice_loop_frame_s=args.warn_frame_time,
        )
        result = evaluate_session_qc(args.session, thresholds=thresholds)
        print(format_qc_table(result))
        if result["overall_status"] == "FAIL":
            raise SystemExit(1)
        return
    if args.command == "calibrate-valve":
        if not args.yes:
            confirmation = input(
                "Calibration will actuate the reward pump. Clear the rig, place a tared "
                "collection vessel, and type CALIBRATE to continue: "
            )
            if confirmation.strip() != "CALIBRATE":
                raise SystemExit("Valve calibration cancelled")
        try:
            durations_s = [
                float(item.strip()) / 1000.0
                for item in str(args.durations_ms).split(",")
                if item.strip()
            ]
            if len(durations_s) < 2 or any(duration <= 0 for duration in durations_s):
                raise ValueError
        except ValueError:
            parser.error("--durations-ms requires at least two positive comma-separated values")
        if args.repetitions <= 0:
            parser.error("--repetitions must be greater than zero")
        if args.water_density <= 0:
            parser.error("--water-density must be greater than zero")
        if args.inter_pulse < 0:
            parser.error("--inter-pulse must be non-negative")
        config = SessionConfig.from_yaml(args.config)
        config.session_export.source_config_path = str(args.config.resolve())
        result = run_guided_valve_calibration(
            config,
            durations_s=durations_s,
            repetitions=args.repetitions,
            output=args.output,
            water_density_mg_per_ul=args.water_density,
            inter_pulse_s=args.inter_pulse,
            overwrite=args.overwrite,
        )
        print(json.dumps(result, indent=2))
        return
    if args.command == "log-weight":
        result = append_weight_entry(
            args.subjects_dir,
            args.subject,
            weight_g=args.weight,
            water_target_ml=args.water_target,
            entry_date=args.date,
            notes=args.notes,
        )
        print(json.dumps(result, indent=2))
        return
    if args.command == "export-preview-video":
        result = export_preview_video(
            args.session,
            output=args.output,
            fps=args.fps,
            codec=args.codec,
            max_frames=args.max_frames,
        )
        print(json.dumps(result, indent=2))
        return
    if args.command == "rig-preflight":
        if args.monitor and args.no_monitor:
            parser.error("--monitor cannot be combined with --no-monitor")
        logging.basicConfig(
            level=logging.DEBUG if args.verbose else logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        config = SessionConfig.from_yaml(args.config)
        validate_session_config(config)
        config.session_export.source_config_path = str(args.config)
        if args.monitor:
            config.monitor.enabled = True
            _record_cli_override(
                config,
                source="hf-dm rig-preflight --monitor",
                values={"monitor.enabled": True},
            )
        if args.no_monitor:
            config.monitor.enabled = False
            _record_cli_override(
                config,
                source="hf-dm rig-preflight --no-monitor",
                values={"monitor.enabled": False},
            )
        session = HFDecisionSession(config)
        try:
            session.connect()
            summary = session.preflight(duration_s=args.duration)
        except BaseException:
            session.close()
            raise
        report_path = _write_rig_preflight_report(
            session,
            duration_s=args.duration,
            config_path=args.config,
        )
        preflight_report = json.loads(report_path.read_text(encoding="utf-8"))
        json_path = session.recorder.base_path.with_suffix(".json")
        export_result = export_existing_session(json_path)
        validation = _validate_after_export(export_result)
        print(
            json.dumps(
                {
                    "end_reason": summary.end_reason,
                    "duration_s": args.duration,
                    "report": str(report_path),
                    "preflight_validation": preflight_report.get("validation"),
                    "bundle_dir": export_result.get("bundle_dir"),
                    "manifest": export_result.get("manifest"),
                    "validation": validation,
                },
                indent=2,
            )
        )
        return
    if args.command == "camera-roi-setup":
        result = _write_camera_roi(
            args.config,
            output=args.output,
            width=args.width,
            height=args.height,
            offset_x=args.offset_x,
            offset_y=args.offset_y,
            interactive=args.interactive,
            visual=args.visual,
            clear=args.clear,
        )
        print(json.dumps(result, indent=2))
        return
    parser.error(f"unknown command: {args.command}")


def _record_cli_override(
    config: SessionConfig,
    *,
    source: str,
    values: dict[str, object],
) -> None:
    if not values:
        return
    config.session_export.cli_overrides.append(
        {
            "source": source,
            "values": json.loads(json.dumps(values, default=str)),
            "created_wall_time_s": time.time(),
        }
    )


def _write_camera_roi(
    config_path: Path,
    *,
    output: Path | None,
    width: int | None,
    height: int | None,
    offset_x: int | None,
    offset_y: int | None,
    interactive: bool,
    visual: bool,
    clear: bool,
) -> dict[str, object]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    camera = dict(raw.get("camera") or {})
    integrations = dict(raw.get("integrations") or {})
    keys = {
        "roi_width": width,
        "roi_height": height,
        "roi_offset_x": offset_x,
        "roi_offset_y": offset_y,
    }
    supplied_coordinates = any(value is not None for value in keys.values())
    if clear and (visual or interactive or supplied_coordinates):
        raise SystemExit(
            "camera-roi-setup --clear cannot be combined with other ROI modes or numbers"
        )
    if visual and (interactive or supplied_coordinates):
        raise SystemExit(
            "camera-roi-setup --visual cannot be combined with --interactive or ROI numbers"
        )
    use_visual = bool(visual) or (
        bool(integrations.get("mm_flir_roi_gui", False))
        and not clear
        and not interactive
        and all(value is None for value in keys.values())
    )
    if clear:
        mode = "clear"
    elif use_visual:
        mode = "visual"
    elif interactive:
        mode = "interactive"
    else:
        mode = "numeric"
    if clear:
        for key in keys:
            camera.pop(key, None)
    else:
        if use_visual:
            from hf_dm.camera.roi_gui import select_spinnaker_roi

            selected = select_spinnaker_roi(
                serial_number=(
                    str(camera["serial_number"])
                    if camera.get("serial_number") not in (None, "")
                    else None
                ),
                exposure_us=int(camera.get("exposure_us", 10_000)),
                gain_db=(
                    float(camera["gain_db"])
                    if camera.get("gain_db") not in (None, "")
                    else None
                ),
                pixel_format=str(camera.get("pixel_format", "Mono8")),
            )
            if selected is None:
                return {
                    "config": str(config_path),
                    "output": None,
                    "mode": "visual",
                    "cancelled": True,
                    "roi": None,
                }
            keys = {
                "roi_width": selected["width"],
                "roi_height": selected["height"],
                "roi_offset_x": selected["offset_x"],
                "roi_offset_y": selected["offset_y"],
            }
        elif interactive:
            keys = {
                "roi_width": _prompt_roi_int("ROI width", width, camera.get("roi_width")),
                "roi_height": _prompt_roi_int("ROI height", height, camera.get("roi_height")),
                "roi_offset_x": _prompt_roi_int(
                    "ROI offset X",
                    offset_x,
                    camera.get("roi_offset_x"),
                ),
                "roi_offset_y": _prompt_roi_int(
                    "ROI offset Y",
                    offset_y,
                    camera.get("roi_offset_y"),
                ),
            }
        missing = [name for name, value in keys.items() if value is None]
        if missing:
            raise SystemExit(
                "camera-roi-setup requires --width, --height, --offset-x, and --offset-y "
                "unless --interactive or --clear is used"
            )
        for key, value in keys.items():
            assert value is not None
            camera[key] = max(0, int(value))
    raw["camera"] = camera
    target = output or config_path
    target.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return {
        "config": str(config_path),
        "output": str(target),
        "mode": mode,
        "cancelled": False,
        "roi": None
        if clear
        else {
            "width": camera.get("roi_width"),
            "height": camera.get("roi_height"),
            "offset_x": camera.get("roi_offset_x"),
            "offset_y": camera.get("roi_offset_y"),
        },
    }


def _prompt_roi_int(label: str, provided: int | None, current: object) -> int | None:
    if provided is not None:
        return max(0, int(provided))
    default = None if current in (None, "") else int(current)
    suffix = "" if default is None else f" [{default}]"
    raw = input(f"{label}{suffix}: ").strip()
    if raw == "":
        return default
    return max(0, int(raw))


def _write_rig_preflight_report(
    session: HFDecisionSession,
    *,
    duration_s: float,
    config_path: Path,
) -> Path:
    if session.recorder is None:
        raise RuntimeError("rig-preflight finished without a SessionRecorder")
    recorder = session.recorder
    layout = str((recorder.export_options or {}).get("layout", "flat"))
    bundle_dir = session_bundle_dir(recorder.base_path, layout=layout)
    exports_dir = bundle_dir / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    report_path = exports_dir / "rig_preflight_report.json"
    preflight_events = [
        asdict(event)
        for event in recorder.events
        if str(event.event_type).startswith("preflight")
    ]
    module_status = _streamdio_module_status(session)
    typed_validation = validation_report(
        _preflight_validation_results(
            session,
            duration_s=float(duration_s),
            preflight_events=preflight_events,
            module_status=module_status,
        )
    )
    report = {
        "schema_version": 1,
        "generated_wall_time_s": time.time(),
        "command": "hf-dm rig-preflight",
        "config_path": str(config_path),
        "duration_s": float(duration_s),
        "summary": asdict(recorder.summary) if recorder.summary is not None else None,
        "monitor_enabled": bool(session.config.monitor.enabled),
        "session": {
            "base_path": str(recorder.base_path),
            "npz": str(recorder.base_path.with_suffix(".npz")),
            "json": str(recorder.base_path.with_suffix(".json")),
            "bundle_dir": str(bundle_dir),
        },
        "config": config_to_dict(session.config),
        "preflight_events": preflight_events,
        "validation": typed_validation,
        "streamdio": {
            "event_count": int(getattr(recorder, "_streamdio_total_count", 0)),
            "recent_in_memory_count": len(recorder.streamdio_events),
            "counts_by_type": dict(sorted(getattr(recorder, "_streamdio_counts", {}).items())),
            "module_status": module_status,
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path


def _preflight_validation_results(
    session: HFDecisionSession,
    *,
    duration_s: float,
    preflight_events: list[dict[str, object]],
    module_status: dict[str, object],
) -> list[ValidationResult]:
    recorder = session.recorder
    assert recorder is not None
    end_payload: dict[str, object] = {}
    for event in reversed(preflight_events):
        if event.get("event_type") == "preflight_end" and isinstance(event.get("payload"), dict):
            end_payload = dict(event["payload"])
            break
    results = [
        ValidationResult(
            name="preflight_completed",
            status=(
                "PASS"
                if recorder.summary is not None and recorder.summary.end_reason == "preflight"
                else "FAIL"
            ),
            detail=(
                f"end_reason={recorder.summary.end_reason}"
                if recorder.summary is not None
                else "session summary missing"
            ),
            hint="Inspect preflight_error events and the session journal.",
        ),
        ValidationResult(
            name="streamdio_connection",
            status="PASS" if isinstance(module_status.get("module_info"), dict) else "FAIL",
            detail=(
                f"module={module_status.get('module_info')}"
                if module_status.get("module_info")
                else "StreamDIO module metadata missing"
            ),
            hint="Check COM assignments, cable ownership, and StreamDIO firmware.",
        ),
        ValidationResult(
            name="streamdio_events",
            status="PASS" if int(getattr(recorder, "_streamdio_total_count", 0)) > 0 else "FAIL",
            detail=f"events={int(getattr(recorder, '_streamdio_total_count', 0))}",
            hint="Confirm USB2 streaming is active and no other process owns the port.",
        ),
    ]
    overruns = _nested_int(module_status, "buffer_overruns")
    results.append(
        ValidationResult(
            name="streamdio_buffer_overruns",
            status="WARN" if overruns is None else ("PASS" if overruns == 0 else "FAIL"),
            detail="not reported" if overruns is None else f"overruns={overruns}",
            hint="Reduce host load or inspect StreamDIO USB streaming if overruns are non-zero.",
        )
    )
    force_samples = int(end_payload.get("force_samples_seen") or 0)
    results.append(
        ValidationResult(
            name="force_adc_stream",
            status=("PASS" if force_samples > 0 else ("WARN" if duration_s <= 0 else "FAIL")),
            detail=f"samples={force_samples}",
            hint="Check analog channel 13/pin 14 and the StreamDIO acquisition configuration.",
        )
    )
    camera_events = int(end_payload.get("camera_events_seen") or 0)
    camera_required = bool(session.config.camera.required and session.config.camera.record_video)
    camera_status = "PASS" if camera_events > 0 else ("FAIL" if camera_required else "WARN")
    results.append(
        ValidationResult(
            name="camera_trigger_stream",
            status=camera_status,
            detail=f"camera_ttls={camera_events}",
            hint="Check the camera trigger cable and StreamDIO camera output configuration.",
        )
    )
    preflight_errors = [
        event
        for event in preflight_events
        if str(event.get("event_type")) == "preflight_error"
    ]
    results.append(
        ValidationResult(
            name="preflight_runtime_errors",
            status="FAIL" if preflight_errors else "PASS",
            detail=f"errors={len(preflight_errors)}",
            hint="Inspect the preflight_error payload for the originating exception.",
        )
    )
    return results


def _nested_int(value: object, key: str) -> int | None:
    if isinstance(value, dict):
        if key in value:
            try:
                return int(value[key])
            except (TypeError, ValueError):
                return None
        for item in value.values():
            found = _nested_int(item, key)
            if found is not None:
                return found
    return None


def _streamdio_module_status(session: HFDecisionSession) -> dict[str, object]:
    dio = getattr(session, "dio", None)
    getter = getattr(dio, "get_module_status", None)
    if callable(getter):
        try:
            return dict(getter())
        except Exception:
            pass
    cached = getattr(session, "_latest_streamdio_module_status", None)
    if isinstance(cached, dict) and cached:
        return {"source": "cached_status_poll", **cached}
    raw = getattr(dio, "raw", None)
    module_info = getattr(raw, "module_info", None)
    if isinstance(module_info, dict) and module_info:
        return {"source": "connection_metadata", "module_info": dict(module_info)}
    recorder = getattr(session, "recorder", None)
    summaries = getattr(recorder, "stream_summaries", None)
    if isinstance(summaries, dict):
        streamdio = summaries.get("streamdio")
        if isinstance(streamdio, dict) and streamdio:
            return {"source": "session_stream_summary", **streamdio}
    return {}


def _validate_after_export(export_result: dict[str, object]) -> dict[str, object]:
    manifest = export_result.get("manifest")
    bundle_dir = export_result.get("bundle_dir")
    if not manifest:
        return {"valid": None, "skipped": "manifest disabled"}
    try:
        return validate_session(Path(str(bundle_dir or manifest)))
    except Exception as exc:
        return {"valid": False, "error": f"{type(exc).__name__}: {exc}"}


if __name__ == "__main__":
    main()
