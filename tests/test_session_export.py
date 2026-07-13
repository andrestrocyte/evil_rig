import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

import hf_dm.data.recorder as recorder_module
from hf_dm.config import SessionConfig, config_to_dict
from hf_dm.data.recorder import SessionRecorder, SessionSummary, TrialRecord
from hf_dm.data.session_bundle import SessionBundle
from hf_dm.data.session_export import (
    _clock_alignment,
    export_existing_session,
    resolve_session_output,
    sha256_file,
    validate_session,
)


def _trial() -> TrialRecord:
    return TrialRecord(
        trial_index=1,
        phase="decision_making",
        force_direction="cw",
        choice="cw",
        matched=True,
        reward_duration_s=0.08,
        responded=True,
        response_time_s=0.2,
        net_pulses=3,
        cw_pulses=3,
        ccw_pulses=0,
        final_position_deg=5.0,
        force_samples=2,
        force_last=42.0,
        force_mean=41.0,
        force_min=40.0,
        force_max=42.0,
        trial_start_wall_s=1.0,
        trial_start_mono_s=10.0,
        lick_count=1,
        lick_timestamps_us=[4000],
        camera_ttl_timestamps_us=[2000, 36000, 70000],
        frame2ttl_edge_count=1,
    )


def _summary() -> SessionSummary:
    return SessionSummary(
        subject_id="S1",
        phase="decision_making",
        n_trials_planned=1,
        n_trials_completed=1,
        end_reason="completed",
        consecutive_failures=0,
        session_start_wall_s=1.0,
        session_end_wall_s=2.0,
    )


def test_trial_record_derives_camera_ttl_interval_stats():
    trial = _trial()

    assert trial.camera_ttl_interval_count == 2
    assert np.isclose(trial.camera_ttl_mean_interval_s, 0.034)
    assert np.isclose(trial.camera_ttl_max_interval_s, 0.034)


def test_clock_alignment_fits_state_marker_and_frame2ttl_anchors():
    events = [
        {
            "event_type": "bpod_state_marker",
            "wall_time_s": 100.0 + 2.0 * index,
            "monotonic_s": 10.0 + index,
            "payload": {
                "module_timestamp_us": (5 + index) * 1_000_000,
                "host_perf_counter_s": 10.0 + index,
            },
        }
        for index in range(3)
    ]
    streamdio_rows = [
        {
            "event_type": "dio_input",
            "channel": 5,
            "module_timestamp_us": 8_000_000,
            "host_perf_counter_s": 13.0,
        },
        {
            "event_type": "dio_input",
            "channel": 5,
            "module_timestamp_us": 9_000_000,
            "host_perf_counter_s": 14.0,
        },
    ]

    alignment = _clock_alignment(
        events,
        streamdio_rows,
        {"config": {"frame2ttl": {"input_channel": 5}}},
    )

    pairs = alignment["pairs"]
    assert pairs["perf_counter_to_wall"] == {
        "source_clock": "perf_counter_s",
        "target_clock": "wall_time_s",
        "status": "ok",
        "slope": 2.0,
        "intercept": 80.0,
        "n_anchors": 3,
        "max_abs_residual_s": 0.0,
        "rms_residual_s": 0.0,
    }
    assert pairs["streamdio_to_perf_counter"]["status"] == "ok"
    assert pairs["streamdio_to_perf_counter"]["n_anchors"] == 5
    assert np.isclose(pairs["streamdio_to_perf_counter"]["slope"], 1.0)
    assert np.isclose(pairs["streamdio_to_perf_counter"]["intercept"], 5.0)
    assert pairs["streamdio_to_wall"]["status"] == "ok"
    assert np.isclose(pairs["streamdio_to_wall"]["slope"], 2.0)
    assert np.isclose(pairs["streamdio_to_wall"]["intercept"], 90.0)


def test_clock_alignment_degrades_gracefully_with_too_few_anchors():
    alignment = _clock_alignment([], [], {"config": {}})

    for pair in alignment["pairs"].values():
        assert pair["status"] == "insufficient_anchors"
        assert pair["n_anchors"] == 0
        assert pair["slope"] is None
        assert pair["intercept"] is None
        assert pair["max_abs_residual_s"] is None
        assert pair["rms_residual_s"] is None


def _refresh_manifest_artifact_sha(manifest_path, artifact_path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in manifest["artifacts"]:
        resolved = (manifest_path.parent / item["path"]).resolve()
        if resolved == artifact_path.resolve():
            item["sha256"] = sha256_file(artifact_path)
            item["bytes"] = artifact_path.stat().st_size
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def test_session_finalize_writes_manifest_snapshot_and_exports(tmp_path):
    cfg = SessionConfig(subject_id="S1", n_trials=1, results_dir=str(tmp_path))
    source_config = tmp_path / "source_config.yaml"
    source_config.write_text(
        "subject_id: S1\nn_trials: 1\nsession_export:\n  manifest: true\n",
        encoding="utf-8",
    )
    cfg.session_export.source_config_path = str(source_config)
    cfg.session_export.cli_overrides.append(
        {
            "source": "pytest --monitor",
            "values": {"monitor.enabled": True},
        }
    )
    recorder = SessionRecorder(tmp_path, cfg.subject_id, cfg.phase.value)
    recorder.set_export_options(config_to_dict(cfg)["session_export"])
    recorder.set_metadata(
        {
            "config": config_to_dict(cfg),
            "task_summary": {"modality": "force"},
            "streamdio_module_info": {"firmware_version": 4},
        }
    )
    recorder.register_stream("streamdio", {"type": "streamdio", "module_info": {"firmware_version": 4}})
    recorder.log_event("trial_start", {"trial_index": 1})
    recorder.log_event("bpod_state_marker", {"trial_index": 1, "state_name": "ChoiceWindow"})
    recorder.log_event("lick", {"trial_index": 1, "channel": 14, "timestamp_us": 4000})
    recorder.log_event(
        "white_noise_ready",
        {"status": {"device_id": "sim", "continuous_playback_active": True}},
    )
    recorder.log_streamdio_event("analog_input", 13, 42, 1000, 0.001)
    recorder.log_streamdio_event("camera", None, None, 2000, 0.002)
    recorder.log_streamdio_event("dio_input", 5, 1, 3000, 0.003)
    recorder.log_fsm_definition("test_fsm", {"states": ["ChoiceWindow"]})
    recorder.log_bpod_trial(1, [{"type": "StateStart", "state": "ChoiceWindow"}])
    recorder.add_trial(_trial())
    summaries = recorder.get_stream_summaries()
    summaries["streamdio"]["type"] = "mutated"
    assert recorder.get_stream_summaries()["streamdio"]["type"] == "streamdio"

    artifact = recorder.finalize(_summary())

    bundle = recorder.base_path.with_name(f"{recorder.base_path.name}_bundle")
    manifest_path = bundle / "manifest.json"
    assert artifact == recorder.base_path.with_suffix(".npz")
    assert manifest_path.exists()
    assert (bundle / "config.snapshot.yaml").exists()
    assert (bundle / "exports" / "streamdio_events.csv").exists()
    assert (bundle / "exports" / "force_trace.csv").exists()
    assert (bundle / "exports" / "camera_ttl.csv").exists()
    assert (bundle / "exports" / "frame2ttl_edges.csv").exists()
    assert (bundle / "exports" / "licks.csv").exists()
    assert (bundle / "exports" / "synchronized_timeline.csv").exists()
    assert (bundle / "exports" / "alignment_anchors.json").exists()
    assert (bundle / "exports" / "soundcard_status.json").exists()
    assert (bundle / "exports" / "stream_summaries.json").exists()
    assert (bundle / "bpod_exports" / "bpod_trials_flat.csv").exists()
    assert (bundle / "bpod_data" / "bpod_trials.json").exists()
    assert not recorder.base_path.with_name(f"{recorder.base_path.name}_export_error.json").exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_paths = {item["path"].replace("\\", "/") for item in manifest["artifacts"]}
    bpod_data = json.loads(
        (bundle / "bpod_data" / "bpod_trials.json").read_text(encoding="utf-8")
    )
    assert manifest["counts"]["trials"] == 1
    assert manifest["counts"]["streamdio_by_type"]["camera"] == 1
    assert manifest["counts"]["frames"] == 0
    assert manifest["counts"]["force_samples"] == 1
    assert manifest["counts"]["frame2ttl_edges"] == 1
    assert manifest["counts"]["bpod_trial_rows"] == 1
    assert manifest["clock_alignment"]["model"] == (
        "target_s = slope * source_s + intercept"
    )
    assert set(manifest["clock_alignment"]["pairs"]) == {
        "perf_counter_to_wall",
        "streamdio_to_perf_counter",
        "streamdio_to_wall",
    }
    for pair in manifest["clock_alignment"]["pairs"].values():
        assert {
            "slope",
            "intercept",
            "n_anchors",
            "max_abs_residual_s",
            "rms_residual_s",
        } <= set(pair)
    assert "bpod_data/bpod_trials.json" in artifact_paths
    assert manifest["config"]["source_config_path"] == str(source_config)
    assert manifest["config"]["source_config_sha256"] == sha256_file(source_config)
    assert manifest["config"]["snapshot_artifact"] == "config.snapshot.yaml"
    assert manifest["config"]["cli_overrides"][0]["source"] == "pytest --monitor"
    assert bpod_data["source"] == "bpod_core_get_data_concat_false"
    assert bpod_data["trial_count"] == 1
    assert bpod_data["fsm_definition_count"] == 1
    assert bpod_data["bpod_trials"][0]["events"][0]["state"] == "ChoiceWindow"
    soundcard = json.loads(
        (bundle / "exports" / "soundcard_status.json").read_text(encoding="utf-8")
    )
    assert soundcard["device_id"] == "sim"
    assert soundcard["continuous_playback_active"] is True
    assert manifest["streamdio"]["firmware_version"] == 4
    assert manifest["stream_summaries"]["streamdio"]["type"] == "streamdio"
    assert manifest["recorder_stats"]["max_streamdio_events_in_memory"] == 3
    assert manifest["recorder_stats"]["bundle_artifact_count"] == len(
        manifest["artifacts"]
    )
    assert manifest["recorder_stats"]["bundle_artifact_bytes"] == sum(
        item["bytes"] for item in manifest["artifacts"]
    )
    assert manifest["recorder_stats"]["bundle_artifact_bytes"] >= manifest[
        "recorder_stats"
    ]["primary_artifact_bytes"]
    with np.load(artifact, allow_pickle=True) as data:
        trial_rows = data["trials"]
        assert "camera_ttl_mean_interval_s" in trial_rows.dtype.names
        assert trial_rows["camera_ttl_interval_count"][0] == 2
        assert np.isclose(trial_rows["camera_ttl_mean_interval_s"][0], 0.034)
    with (bundle / "exports" / "trials_summary.csv").open(
        "r", newline="", encoding="utf-8"
    ) as handle:
        trial_summary = next(csv.DictReader(handle))
    assert trial_summary["camera_ttl_interval_count"] == "2"
    assert np.isclose(float(trial_summary["camera_ttl_mean_interval_s"]), 0.034)
    assert validate_session(bundle)["valid"]
    snapshot = yaml.safe_load((bundle / "config.snapshot.yaml").read_text(encoding="utf-8"))
    assert snapshot["source_config_path"] == str(source_config)
    assert snapshot["source_config_sha256"] == sha256_file(source_config)
    assert snapshot["source_config_yaml"] == source_config.read_bytes().decode("utf-8")
    assert snapshot["cli_overrides"][0]["values"]["monitor.enabled"] is True
    assert snapshot["effective_config"]["subject_id"] == "S1"


def test_export_existing_session_regenerates_bundle(tmp_path):
    cfg = SessionConfig(subject_id="S1", n_trials=1, results_dir=str(tmp_path))
    recorder = SessionRecorder(tmp_path, cfg.subject_id, cfg.phase.value)
    recorder.set_export_options(config_to_dict(cfg)["session_export"])
    recorder.set_metadata({"config": config_to_dict(cfg)})
    recorder.log_streamdio_event("camera", None, None, 2000, 0.002)
    recorder.add_trial(_trial())
    recorder.finalize(_summary())

    bundle = recorder.base_path.with_name(f"{recorder.base_path.name}_bundle")
    (bundle / "manifest.json").unlink()

    result = export_existing_session(recorder.base_path.with_suffix(".json"))

    assert result["manifest"].endswith("manifest.json")
    assert validate_session(bundle)["valid"]


def test_manifest_indexes_spinnaker_summary_and_frame_timestamps(tmp_path):
    cfg = SessionConfig(subject_id="S1", n_trials=1, results_dir=str(tmp_path))
    recorder = SessionRecorder(tmp_path, cfg.subject_id, cfg.phase.value)
    recorder.set_export_options(config_to_dict(cfg)["session_export"])
    recorder.set_metadata({"config": config_to_dict(cfg)})
    video_dir = tmp_path / "videos" / "S1_decision_spinnaker"
    video_dir.mkdir(parents=True)
    summary_path = video_dir / "recording_summary.json"
    timestamps_path = video_dir / "frame_timestamps.csv"
    summary_path.write_text(
        json.dumps({"backend": "spinnaker", "frames_saved": 7}),
        encoding="utf-8",
    )
    timestamps_path.write_text(
        "\n".join(
            [
                "frame_index,camera_timestamp_ns,wall_time_s,monotonic_s,filename",
                "0,123456789,10.5,2.5,frame_000000000.pgm",
                "1,123456790,10.6,2.6,frame_000000001.pgm",
                "2,123456791,10.7,2.7,frame_000000002.pgm",
                "3,123456792,10.8,2.8,frame_000000003.pgm",
                "4,123456793,10.9,2.9,frame_000000004.pgm",
                "5,123456794,11.0,3.0,frame_000000005.pgm",
                "6,123456795,11.1,3.1,frame_000000006.pgm",
            ]
        ),
        encoding="utf-8",
    )
    recorder.log_event(
        "video_recording_stopped",
        {
            "backend": "spinnaker",
            "session_dir": str(video_dir),
            "frames_saved": 7,
            "frame_timestamps_csv": str(timestamps_path),
        },
    )
    recorder.add_trial(_trial())

    recorder.finalize(_summary())

    bundle = recorder.base_path.with_name(f"{recorder.base_path.name}_bundle")
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    artifact_paths = {item["path"].replace("\\", "/") for item in manifest["artifacts"]}
    video = manifest["video"]
    pointer_path = bundle / "videos" / "spinnaker_video.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    timeline_path = bundle / "exports" / "synchronized_timeline.csv"
    with timeline_path.open("r", newline="", encoding="utf-8") as handle:
        timeline_rows = list(csv.DictReader(handle))
    anchors = json.loads(
        (bundle / "exports" / "alignment_anchors.json").read_text(encoding="utf-8")
    )
    first_frame_rows = [
        row for row in timeline_rows if row["event_type"] == "first_spinnaker_frame"
    ]
    assert video["n_frames"] == 7
    assert manifest["counts"]["frames"] == 7
    assert video["primary"].endswith("recording_summary.json")
    assert pointer["kind"] == "spinnaker_video_pointer"
    assert pointer["video"]["n_frames"] == 7
    assert pointer["video"]["primary"].endswith("recording_summary.json")
    assert any(item["path"].endswith("recording_summary.json") for item in pointer["artifacts"])
    assert first_frame_rows
    assert first_frame_rows[0]["source"] == "spinnaker"
    assert first_frame_rows[0]["module_timestamp_us"] == "123456"
    assert anchors["first_spinnaker_frame"]["filename"] == "frame_000000000.pgm"
    assert any(path.endswith("recording_summary.json") for path in artifact_paths)
    assert any(path.endswith("frame_timestamps.csv") for path in artifact_paths)
    assert "videos/spinnaker_video.json" in artifact_paths
    assert validate_session(bundle)["valid"]


def test_validate_session_detects_primary_npz_schema_corruption(tmp_path):
    cfg = SessionConfig(subject_id="S1", n_trials=1, results_dir=str(tmp_path))
    recorder = SessionRecorder(tmp_path, cfg.subject_id, cfg.phase.value)
    recorder.set_export_options(config_to_dict(cfg)["session_export"])
    recorder.set_metadata({"config": config_to_dict(cfg)})
    recorder.log_streamdio_event("camera", None, None, 2000, 0.002)
    recorder.add_trial(_trial())
    recorder.finalize(_summary())

    bundle = recorder.base_path.with_name(f"{recorder.base_path.name}_bundle")
    manifest_path = bundle / "manifest.json"
    npz_path = recorder.base_path.with_suffix(".npz")
    np.savez_compressed(npz_path, streamdio_events=np.array([], dtype=object))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in manifest["artifacts"]:
        resolved = (manifest_path.parent / item["path"]).resolve()
        if resolved == npz_path.resolve():
            item["sha256"] = sha256_file(npz_path)
            item["bytes"] = npz_path.stat().st_size
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    result = validate_session(bundle)

    assert not result["valid"]
    assert any("primary npz missing arrays" in error for error in result["errors"])


def test_validate_session_detects_artifact_byte_size_mismatch(tmp_path):
    cfg = SessionConfig(subject_id="S1", n_trials=1, results_dir=str(tmp_path))
    recorder = SessionRecorder(tmp_path, cfg.subject_id, cfg.phase.value)
    recorder.set_export_options(config_to_dict(cfg)["session_export"])
    recorder.set_metadata({"config": config_to_dict(cfg)})
    recorder.add_trial(_trial())
    recorder.finalize(_summary())

    bundle = recorder.base_path.with_name(f"{recorder.base_path.name}_bundle")
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][0]["bytes"] += 1
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    result = validate_session(bundle)

    assert not result["valid"]
    assert any("byte size mismatch" in error for error in result["errors"])


def test_validate_session_detects_recorder_stats_mismatch(tmp_path):
    cfg = SessionConfig(subject_id="S1", n_trials=1, results_dir=str(tmp_path))
    recorder = SessionRecorder(tmp_path, cfg.subject_id, cfg.phase.value)
    recorder.set_export_options(config_to_dict(cfg)["session_export"])
    recorder.set_metadata({"config": config_to_dict(cfg)})
    recorder.log_streamdio_event("camera", None, None, 2000, 0.002)
    recorder.add_trial(_trial())
    recorder.finalize(_summary())

    bundle = recorder.base_path.with_name(f"{recorder.base_path.name}_bundle")
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["recorder_stats"]["bundle_artifact_count"] += 1
    manifest["recorder_stats"]["streamdio_total_count"] += 1
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    result = validate_session(bundle)

    assert not result["valid"]
    assert any(
        "recorder_stats bundle_artifact_count mismatch" in error
        for error in result["errors"]
    )
    assert any(
        "recorder_stats streamdio_total_count mismatch" in error
        for error in result["errors"]
    )


def test_validate_session_detects_manifest_flag_mismatch(tmp_path):
    cfg = SessionConfig(subject_id="S1", n_trials=1, results_dir=str(tmp_path))
    cfg.monitor.enabled = True
    recorder = SessionRecorder(tmp_path, cfg.subject_id, cfg.phase.value)
    recorder.set_export_options(config_to_dict(cfg)["session_export"])
    recorder.set_metadata({"config": config_to_dict(cfg)})
    recorder.add_trial(_trial())
    recorder.finalize(_summary())

    bundle = recorder.base_path.with_name(f"{recorder.base_path.name}_bundle")
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["flags"]["monitor"] is True
    manifest["flags"]["monitor"] = False
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    result = validate_session(bundle)

    assert not result["valid"]
    assert any("manifest flag monitor mismatch" in error for error in result["errors"])


def test_validate_session_detects_manifest_count_mismatch(tmp_path):
    cfg = SessionConfig(subject_id="S1", n_trials=1, results_dir=str(tmp_path))
    recorder = SessionRecorder(tmp_path, cfg.subject_id, cfg.phase.value)
    recorder.set_export_options(config_to_dict(cfg)["session_export"])
    recorder.set_metadata({"config": config_to_dict(cfg)})
    recorder.log_event("lick", {"trial_index": 1, "channel": 14, "timestamp_us": 4000})
    recorder.log_streamdio_event("camera", None, None, 2000, 0.002)
    recorder.log_streamdio_event("dio_input", 5, 1, 3000, 0.003)
    recorder.log_bpod_trial(1, [{"type": "StateStart", "state": "ChoiceWindow"}])
    recorder.add_trial(_trial())
    recorder.finalize(_summary())

    bundle = recorder.base_path.with_name(f"{recorder.base_path.name}_bundle")
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["counts"]["events"] += 1
    manifest["counts"]["session_events_by_type"]["lick"] += 1
    manifest["counts"]["streamdio_by_type"]["camera"] += 1
    manifest["counts"]["camera_ttls"] += 1
    manifest["counts"]["frames"] += 1
    manifest["counts"]["force_samples"] += 1
    manifest["counts"]["frame2ttl_edges"] += 1
    manifest["counts"]["bpod_trial_rows"] += 1
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    result = validate_session(bundle)

    assert not result["valid"]
    assert any("manifest count events mismatch" in error for error in result["errors"])
    assert any(
        "manifest session_events_by_type mismatch" in error
        for error in result["errors"]
    )
    assert any(
        "manifest streamdio_by_type mismatch" in error for error in result["errors"]
    )
    assert any("manifest count camera_ttls mismatch" in error for error in result["errors"])
    assert any("manifest count frames mismatch" in error for error in result["errors"])
    assert any("manifest count force_samples mismatch" in error for error in result["errors"])
    assert any(
        "manifest count frame2ttl_edges mismatch" in error
        for error in result["errors"]
    )
    assert any(
        "manifest count bpod_trial_rows mismatch" in error
        for error in result["errors"]
    )


def test_validate_session_detects_config_snapshot_source_hash_mismatch(tmp_path):
    source_config = tmp_path / "source_config.yaml"
    source_config.write_text("subject_id: S1\nn_trials: 1\n", encoding="utf-8")
    cfg = SessionConfig(subject_id="S1", n_trials=1, results_dir=str(tmp_path))
    cfg.session_export.source_config_path = str(source_config)
    recorder = SessionRecorder(tmp_path, cfg.subject_id, cfg.phase.value)
    recorder.set_export_options(config_to_dict(cfg)["session_export"])
    recorder.set_metadata({"config": config_to_dict(cfg)})
    recorder.add_trial(_trial())
    recorder.finalize(_summary())

    bundle = recorder.base_path.with_name(f"{recorder.base_path.name}_bundle")
    manifest_path = bundle / "manifest.json"
    snapshot_path = bundle / "config.snapshot.yaml"
    snapshot = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    snapshot["source_config_yaml"] = "subject_id: tampered\n"
    snapshot_path.write_text(yaml.safe_dump(snapshot, sort_keys=False), encoding="utf-8")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in manifest["artifacts"]:
        resolved = (manifest_path.parent / item["path"]).resolve()
        if resolved == snapshot_path.resolve():
            item["sha256"] = sha256_file(snapshot_path)
            item["bytes"] = snapshot_path.stat().st_size
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    result = validate_session(bundle)

    assert not result["valid"]
    assert any(
        "config snapshot source sha256 mismatch" in error
        for error in result["errors"]
    )


def test_validate_session_detects_bpod_data_count_mismatch(tmp_path):
    cfg = SessionConfig(subject_id="S1", n_trials=1, results_dir=str(tmp_path))
    recorder = SessionRecorder(tmp_path, cfg.subject_id, cfg.phase.value)
    recorder.set_export_options(config_to_dict(cfg)["session_export"])
    recorder.set_metadata({"config": config_to_dict(cfg)})
    recorder.log_bpod_trial(1, [{"type": "StateStart", "state": "ChoiceWindow"}])
    recorder.add_trial(_trial())
    recorder.finalize(_summary())

    bundle = recorder.base_path.with_name(f"{recorder.base_path.name}_bundle")
    manifest_path = bundle / "manifest.json"
    bpod_path = bundle / "bpod_data" / "bpod_trials.json"
    bpod_data = json.loads(bpod_path.read_text(encoding="utf-8"))
    bpod_data["trial_count"] = 2
    bpod_path.write_text(json.dumps(bpod_data, indent=2), encoding="utf-8")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in manifest["artifacts"]:
        resolved = (manifest_path.parent / item["path"]).resolve()
        if resolved == bpod_path.resolve():
            item["sha256"] = sha256_file(bpod_path)
            item["bytes"] = bpod_path.stat().st_size
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    result = validate_session(bundle)

    assert not result["valid"]
    assert any("bpod data trial count mismatch" in error for error in result["errors"])


def test_validate_session_detects_bpod_data_row_count_mismatch(tmp_path):
    cfg = SessionConfig(subject_id="S1", n_trials=1, results_dir=str(tmp_path))
    recorder = SessionRecorder(tmp_path, cfg.subject_id, cfg.phase.value)
    recorder.set_export_options(config_to_dict(cfg)["session_export"])
    recorder.set_metadata({"config": config_to_dict(cfg)})
    recorder.log_bpod_trial(
        1,
        [
            {"type": "StateStart", "state": "ChoiceWindow"},
            {"type": "StateStart", "state": "Reward"},
        ],
    )
    recorder.add_trial(_trial())
    recorder.finalize(_summary())

    bundle = recorder.base_path.with_name(f"{recorder.base_path.name}_bundle")
    manifest_path = bundle / "manifest.json"
    bpod_path = bundle / "bpod_data" / "bpod_trials.json"
    bpod_data = json.loads(bpod_path.read_text(encoding="utf-8"))
    bpod_data["bpod_trials"][0]["events"].pop()
    bpod_path.write_text(json.dumps(bpod_data, indent=2), encoding="utf-8")
    _refresh_manifest_artifact_sha(manifest_path, bpod_path)

    result = validate_session(bundle)

    assert not result["valid"]
    assert any(
        "bpod data manifest row count mismatch" in error
        for error in result["errors"]
    )


def test_validate_session_detects_bpod_flat_csv_row_count_mismatch(tmp_path):
    cfg = SessionConfig(subject_id="S1", n_trials=1, results_dir=str(tmp_path))
    recorder = SessionRecorder(tmp_path, cfg.subject_id, cfg.phase.value)
    recorder.set_export_options(config_to_dict(cfg)["session_export"])
    recorder.set_metadata({"config": config_to_dict(cfg)})
    recorder.log_bpod_trial(
        1,
        [
            {"type": "StateStart", "state": "ChoiceWindow"},
            {"type": "StateStart", "state": "Reward"},
        ],
    )
    recorder.add_trial(_trial())
    recorder.finalize(_summary())

    bundle = recorder.base_path.with_name(f"{recorder.base_path.name}_bundle")
    manifest_path = bundle / "manifest.json"
    flat_csv = bundle / "bpod_exports" / "bpod_trials_flat.csv"
    header = flat_csv.read_text(encoding="utf-8").splitlines()[0]
    flat_csv.write_text(f"{header}\n", encoding="utf-8")
    _refresh_manifest_artifact_sha(manifest_path, flat_csv)

    result = validate_session(bundle)

    assert not result["valid"]
    assert any("bpod_trials_flat.csv" in error for error in result["errors"])
    assert any("csv row count mismatch" in error for error in result["errors"])


def test_validate_session_detects_csv_row_count_mismatch(tmp_path):
    cfg = SessionConfig(subject_id="S1", n_trials=1, results_dir=str(tmp_path))
    recorder = SessionRecorder(tmp_path, cfg.subject_id, cfg.phase.value)
    recorder.set_export_options(config_to_dict(cfg)["session_export"])
    recorder.set_metadata({"config": config_to_dict(cfg)})
    recorder.add_trial(_trial())
    recorder.finalize(_summary())

    bundle = recorder.base_path.with_name(f"{recorder.base_path.name}_bundle")
    manifest_path = bundle / "manifest.json"
    trials_csv = bundle / "exports" / "trials_summary.csv"
    header = trials_csv.read_text(encoding="utf-8").splitlines()[0]
    trials_csv.write_text(f"{header}\n", encoding="utf-8")
    _refresh_manifest_artifact_sha(manifest_path, trials_csv)

    result = validate_session(bundle)

    assert not result["valid"]
    assert result["csv_files_checked"] > 0
    assert any("csv row count mismatch" in error for error in result["errors"])


def test_validate_session_detects_force_trace_row_count_mismatch(tmp_path):
    cfg = SessionConfig(subject_id="S1", n_trials=1, results_dir=str(tmp_path))
    recorder = SessionRecorder(tmp_path, cfg.subject_id, cfg.phase.value)
    recorder.set_export_options(config_to_dict(cfg)["session_export"])
    recorder.set_metadata({"config": config_to_dict(cfg)})
    recorder.log_streamdio_event("analog_input", 13, 42, 1000, 0.001)
    recorder.add_trial(_trial())
    recorder.finalize(_summary())

    bundle = recorder.base_path.with_name(f"{recorder.base_path.name}_bundle")
    manifest_path = bundle / "manifest.json"
    force_csv = bundle / "exports" / "force_trace.csv"
    header = force_csv.read_text(encoding="utf-8").splitlines()[0]
    force_csv.write_text(f"{header}\n", encoding="utf-8")
    _refresh_manifest_artifact_sha(manifest_path, force_csv)

    result = validate_session(bundle)

    assert not result["valid"]
    assert any(
        "force_trace.csv" in error and "csv row count mismatch" in error
        for error in result["errors"]
    )


def test_validate_session_checks_monitor_telemetry_jsonl(tmp_path):
    cfg = SessionConfig(subject_id="S1", n_trials=1, results_dir=str(tmp_path))
    recorder = SessionRecorder(tmp_path, cfg.subject_id, cfg.phase.value)
    recorder.set_export_options(config_to_dict(cfg)["session_export"])
    recorder.set_metadata({"config": config_to_dict(cfg)})
    bundle = recorder.base_path.with_name(f"{recorder.base_path.name}_bundle")
    telemetry_path = bundle / "exports" / "monitor_telemetry.jsonl"
    telemetry_path.parent.mkdir(parents=True)
    telemetry_path.write_text(
        json.dumps(
            {
                "kind": "event",
                "record_wall_s": 1.0,
                "record_mono_s": 2.0,
                "message": {"type": "trial_phase", "payload": {"phase": "choice"}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    recorder.add_trial(_trial())
    recorder.finalize(_summary())

    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = validate_session(bundle)

    assert manifest["monitor_telemetry"] == {
        "path": "exports/monitor_telemetry.jsonl",
        "line_count": 1,
        "counts_by_kind": {"event": 1},
        "counts_by_message_type": {"trial_phase": 1},
    }
    assert result["valid"]
    assert result["jsonl_files_checked"] == 2

    telemetry_path.write_text("", encoding="utf-8")
    _refresh_manifest_artifact_sha(manifest_path, telemetry_path)

    result = validate_session(bundle)

    assert not result["valid"]
    assert any("monitor telemetry row count mismatch" in error for error in result["errors"])

    telemetry_path.write_text("{not-json}\n", encoding="utf-8")
    _refresh_manifest_artifact_sha(manifest_path, telemetry_path)

    result = validate_session(bundle)

    assert not result["valid"]
    assert result["jsonl_files_checked"] == 2
    assert any("jsonl invalid json" in error for error in result["errors"])


def test_validate_session_detects_monitor_telemetry_type_count_mismatch(tmp_path):
    cfg = SessionConfig(subject_id="S1", n_trials=1, results_dir=str(tmp_path))
    recorder = SessionRecorder(tmp_path, cfg.subject_id, cfg.phase.value)
    recorder.set_export_options(config_to_dict(cfg)["session_export"])
    recorder.set_metadata({"config": config_to_dict(cfg)})
    bundle = recorder.base_path.with_name(f"{recorder.base_path.name}_bundle")
    telemetry_path = bundle / "exports" / "monitor_telemetry.jsonl"
    telemetry_path.parent.mkdir(parents=True)
    row = {
        "kind": "event",
        "record_wall_s": 1.0,
        "record_mono_s": 2.0,
        "message": {"type": "trial_phase", "payload": {"phase": "choice"}},
    }
    telemetry_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    recorder.add_trial(_trial())
    recorder.finalize(_summary())

    manifest_path = bundle / "manifest.json"
    row["kind"] = "status"
    row["message"]["type"] = "monitor_status"
    telemetry_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    _refresh_manifest_artifact_sha(manifest_path, telemetry_path)

    result = validate_session(bundle)

    assert not result["valid"]
    assert any(
        "monitor telemetry counts_by_kind mismatch" in error
        for error in result["errors"]
    )
    assert any(
        "monitor telemetry counts_by_message_type mismatch" in error
        for error in result["errors"]
    )


def test_validate_session_detects_timeline_csv_detail_corruption(tmp_path):
    cfg = SessionConfig(subject_id="S1", n_trials=1, results_dir=str(tmp_path))
    recorder = SessionRecorder(tmp_path, cfg.subject_id, cfg.phase.value)
    recorder.set_export_options(config_to_dict(cfg)["session_export"])
    recorder.set_metadata({"config": config_to_dict(cfg)})
    recorder.log_event("trial_start", {"trial_index": 1})
    recorder.add_trial(_trial())
    recorder.finalize(_summary())

    bundle = recorder.base_path.with_name(f"{recorder.base_path.name}_bundle")
    manifest_path = bundle / "manifest.json"
    timeline_csv = bundle / "exports" / "synchronized_timeline.csv"
    with timeline_csv.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0].keys())
    rows[0]["detail_json"] = "{not-json"
    with timeline_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    _refresh_manifest_artifact_sha(manifest_path, timeline_csv)

    result = validate_session(bundle)

    assert not result["valid"]
    assert any("timeline csv invalid detail_json" in error for error in result["errors"])


def test_validate_session_detects_alignment_anchor_count_mismatch(tmp_path):
    cfg = SessionConfig(subject_id="S1", n_trials=1, results_dir=str(tmp_path))
    recorder = SessionRecorder(tmp_path, cfg.subject_id, cfg.phase.value)
    recorder.set_export_options(config_to_dict(cfg)["session_export"])
    recorder.set_metadata({"config": config_to_dict(cfg)})
    recorder.log_event("bpod_state_marker", {"trial_index": 1, "state_name": "ChoiceWindow"})
    recorder.add_trial(_trial())
    recorder.finalize(_summary())

    bundle = recorder.base_path.with_name(f"{recorder.base_path.name}_bundle")
    manifest_path = bundle / "manifest.json"
    anchors_path = bundle / "exports" / "alignment_anchors.json"
    anchors = json.loads(anchors_path.read_text(encoding="utf-8"))
    anchors["bpod_state_markers"] = []
    anchors_path.write_text(json.dumps(anchors, indent=2), encoding="utf-8")
    _refresh_manifest_artifact_sha(manifest_path, anchors_path)

    result = validate_session(bundle)

    assert not result["valid"]
    assert any(
        "alignment anchors bpod_state_markers count mismatch" in error
        for error in result["errors"]
    )


def test_validate_session_detects_streamdio_module_info_mismatch(tmp_path):
    cfg = SessionConfig(subject_id="S1", n_trials=1, results_dir=str(tmp_path))
    recorder = SessionRecorder(tmp_path, cfg.subject_id, cfg.phase.value)
    recorder.set_export_options(config_to_dict(cfg)["session_export"])
    recorder.set_metadata(
        {
            "config": config_to_dict(cfg),
            "streamdio_module_info": {
                "firmware_version": 4,
                "model": "StreamDIO",
            },
        }
    )
    recorder.add_trial(_trial())
    recorder.finalize(_summary())

    bundle = recorder.base_path.with_name(f"{recorder.base_path.name}_bundle")
    manifest_path = bundle / "manifest.json"
    module_info_path = bundle / "exports" / "streamdio_module_info.json"
    module_info = json.loads(module_info_path.read_text(encoding="utf-8"))
    module_info["firmware_version"] = 999
    module_info_path.write_text(json.dumps(module_info, indent=2), encoding="utf-8")
    _refresh_manifest_artifact_sha(manifest_path, module_info_path)

    result = validate_session(bundle)

    assert not result["valid"]
    assert any(
        "streamdio module info firmware mismatch" in error
        for error in result["errors"]
    )


def test_validate_session_detects_soundcard_status_inconsistency(tmp_path):
    cfg = SessionConfig(subject_id="S1", n_trials=1, results_dir=str(tmp_path))
    recorder = SessionRecorder(tmp_path, cfg.subject_id, cfg.phase.value)
    recorder.set_export_options(config_to_dict(cfg)["session_export"])
    recorder.set_metadata({"config": config_to_dict(cfg)})
    recorder.log_event(
        "white_noise_ready",
        {
            "status": {
                "device_id": "sim",
                "volume": 0.5,
                "continuous_playback_active": True,
            }
        },
    )
    recorder.add_trial(_trial())
    recorder.finalize(_summary())

    bundle = recorder.base_path.with_name(f"{recorder.base_path.name}_bundle")
    manifest_path = bundle / "manifest.json"
    soundcard_path = bundle / "exports" / "soundcard_status.json"
    soundcard = json.loads(soundcard_path.read_text(encoding="utf-8"))
    soundcard["latest_status"] = {"device_id": "stale"}
    soundcard_path.write_text(json.dumps(soundcard, indent=2), encoding="utf-8")
    _refresh_manifest_artifact_sha(manifest_path, soundcard_path)

    result = validate_session(bundle)

    assert not result["valid"]
    assert any(
        "soundcard status latest_status mismatch" in error
        for error in result["errors"]
    )


def test_validate_session_detects_soundcard_summary_field_mismatch(tmp_path):
    cfg = SessionConfig(subject_id="S1", n_trials=1, results_dir=str(tmp_path))
    recorder = SessionRecorder(tmp_path, cfg.subject_id, cfg.phase.value)
    recorder.set_export_options(config_to_dict(cfg)["session_export"])
    recorder.set_metadata({"config": config_to_dict(cfg)})
    recorder.log_event(
        "white_noise_ready",
        {
            "status": {
                "device_id": "sim",
                "volume": 0.5,
                "continuous_playback_active": True,
            }
        },
    )
    recorder.add_trial(_trial())
    recorder.finalize(_summary())

    bundle = recorder.base_path.with_name(f"{recorder.base_path.name}_bundle")
    soundcard_path = bundle / "exports" / "soundcard_status.json"
    soundcard = json.loads(soundcard_path.read_text(encoding="utf-8"))
    soundcard["continuous_playback_active"] = False
    soundcard_path.write_text(json.dumps(soundcard, indent=2), encoding="utf-8")
    _refresh_manifest_artifact_sha(bundle / "manifest.json", soundcard_path)

    result = validate_session(bundle)

    assert not result["valid"]
    assert any(
        "soundcard status continuous_playback_active mismatch" in error
        for error in result["errors"]
    )


def test_validate_session_detects_stream_summary_mismatch(tmp_path):
    cfg = SessionConfig(subject_id="S1", n_trials=1, results_dir=str(tmp_path))
    recorder = SessionRecorder(tmp_path, cfg.subject_id, cfg.phase.value)
    recorder.set_export_options(config_to_dict(cfg)["session_export"])
    recorder.set_metadata({"config": config_to_dict(cfg)})
    recorder.register_stream("streamdio", {"type": "streamdio", "stream_port": "COM1"})
    recorder.add_trial(_trial())
    recorder.finalize(_summary())

    bundle = recorder.base_path.with_name(f"{recorder.base_path.name}_bundle")
    manifest_path = bundle / "manifest.json"
    stream_summaries_path = bundle / "exports" / "stream_summaries.json"
    stream_summaries = json.loads(stream_summaries_path.read_text(encoding="utf-8"))
    stream_summaries["streamdio"]["stream_port"] = "COM2"
    stream_summaries_path.write_text(
        json.dumps(stream_summaries, indent=2), encoding="utf-8"
    )
    _refresh_manifest_artifact_sha(manifest_path, stream_summaries_path)

    result = validate_session(bundle)

    assert not result["valid"]
    assert any("stream summaries manifest mismatch" in error for error in result["errors"])


def test_validate_session_detects_spinnaker_pointer_frame_mismatch(tmp_path):
    cfg = SessionConfig(subject_id="S1", n_trials=1, results_dir=str(tmp_path))
    recorder = SessionRecorder(tmp_path, cfg.subject_id, cfg.phase.value)
    recorder.set_export_options(config_to_dict(cfg)["session_export"])
    recorder.set_metadata({"config": config_to_dict(cfg)})
    video_dir = tmp_path / "videos" / "S1_decision_spinnaker"
    video_dir.mkdir(parents=True)
    (video_dir / "recording_summary.json").write_text(
        json.dumps({"backend": "spinnaker", "frames_saved": 7}),
        encoding="utf-8",
    )
    timestamps_path = video_dir / "frame_timestamps.csv"
    timestamps_path.write_text(
        "frame_index,camera_timestamp_ns,wall_time_s,monotonic_s,filename\n",
        encoding="utf-8",
    )
    recorder.log_event(
        "video_recording_stopped",
        {
            "backend": "spinnaker",
            "session_dir": str(video_dir),
            "frames_saved": 7,
            "frame_timestamps_csv": str(timestamps_path),
        },
    )
    recorder.add_trial(_trial())
    recorder.finalize(_summary())

    bundle = recorder.base_path.with_name(f"{recorder.base_path.name}_bundle")
    manifest_path = bundle / "manifest.json"
    pointer_path = bundle / "videos" / "spinnaker_video.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    pointer["video"]["n_frames"] = 999
    pointer_path.write_text(json.dumps(pointer, indent=2), encoding="utf-8")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in manifest["artifacts"]:
        resolved = (manifest_path.parent / item["path"]).resolve()
        if resolved == pointer_path.resolve():
            item["sha256"] = sha256_file(pointer_path)
            item["bytes"] = pointer_path.stat().st_size
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    result = validate_session(bundle)

    assert not result["valid"]
    assert any(
        "spinnaker video pointer frame count mismatch" in error
        for error in result["errors"]
    )


def test_validate_session_detects_spinnaker_recording_summary_frame_mismatch(tmp_path):
    cfg = SessionConfig(subject_id="S1", n_trials=1, results_dir=str(tmp_path))
    recorder = SessionRecorder(tmp_path, cfg.subject_id, cfg.phase.value)
    recorder.set_export_options(config_to_dict(cfg)["session_export"])
    recorder.set_metadata({"config": config_to_dict(cfg)})
    video_dir = tmp_path / "videos" / "S1_decision_spinnaker"
    video_dir.mkdir(parents=True)
    summary_path = video_dir / "recording_summary.json"
    summary_path.write_text(
        json.dumps({"backend": "spinnaker", "frames_saved": 7}),
        encoding="utf-8",
    )
    timestamps_path = video_dir / "frame_timestamps.csv"
    timestamps_path.write_text(
        "frame_index,camera_timestamp_ns,wall_time_s,monotonic_s,filename\n",
        encoding="utf-8",
    )
    recorder.log_event(
        "video_recording_stopped",
        {
            "backend": "spinnaker",
            "session_dir": str(video_dir),
            "frames_saved": 7,
            "frame_timestamps_csv": str(timestamps_path),
        },
    )
    recorder.add_trial(_trial())
    recorder.finalize(_summary())

    bundle = recorder.base_path.with_name(f"{recorder.base_path.name}_bundle")
    manifest_path = bundle / "manifest.json"
    recording_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    recording_summary["frames_saved"] = 999
    summary_path.write_text(json.dumps(recording_summary, indent=2), encoding="utf-8")
    _refresh_manifest_artifact_sha(manifest_path, summary_path)

    result = validate_session(bundle)

    assert not result["valid"]
    assert any(
        "spinnaker recording summary frame count mismatch" in error
        for error in result["errors"]
    )


def test_validate_session_checks_preview_video_export_json(tmp_path, monkeypatch):
    cfg = SessionConfig(subject_id="S1", n_trials=1, results_dir=str(tmp_path))
    cfg.video.duplicate_preview_video = True
    cfg.video.preview_video_fps = 15.0
    recorder = SessionRecorder(tmp_path, cfg.subject_id, cfg.phase.value)
    recorder.set_export_options(config_to_dict(cfg)["session_export"])
    recorder.set_metadata({"config": config_to_dict(cfg)})
    video_dir = tmp_path / "videos" / "S1_decision_spinnaker"
    video_dir.mkdir(parents=True)
    (video_dir / "recording_summary.json").write_text(
        json.dumps({"backend": "spinnaker", "frames_saved": 3}),
        encoding="utf-8",
    )

    def fake_export_preview_video(session, *, fps=None, codec="FFV1", **_kwargs):
        session = str(session)
        return {
            "output_path": str(Path(session) / "preview_ffv1.avi"),
            "video_dir": session,
            "codec": codec,
            "fps": fps,
            "frame_width": 2,
            "frame_height": 2,
            "frames_seen": 3,
            "frames_written": 2,
            "frames_skipped": 1,
            "export_s": 0.01,
            "source_summary": str(Path(session) / "recording_summary.json"),
        }

    import hf_dm.data.video_export as video_export

    monkeypatch.setattr(video_export, "export_preview_video", fake_export_preview_video)
    recorder.log_event(
        "video_recording_stopped",
        {
            "backend": "spinnaker",
            "session_dir": str(video_dir),
            "frames_saved": 3,
        },
    )
    recorder.add_trial(_trial())
    recorder.finalize(_summary())

    bundle = recorder.base_path.with_name(f"{recorder.base_path.name}_bundle")
    manifest_path = bundle / "manifest.json"
    preview_path = bundle / "exports" / "preview_video_export.json"
    assert preview_path.exists()
    assert validate_session(bundle)["valid"]

    preview = json.loads(preview_path.read_text(encoding="utf-8"))
    preview["frames_skipped"] = 0
    preview_path.write_text(json.dumps(preview, indent=2), encoding="utf-8")
    _refresh_manifest_artifact_sha(manifest_path, preview_path)

    result = validate_session(bundle)

    assert not result["valid"]
    assert any(
        "preview video export frame accounting mismatch" in error
        for error in result["errors"]
    )


def test_optional_streamdio_chunk_writer_adds_manifest_artifacts(tmp_path):
    cfg = SessionConfig(subject_id="S1", n_trials=1, results_dir=str(tmp_path))
    cfg.session_export.streamdio_chunk_s = 0.001
    recorder = SessionRecorder(tmp_path, cfg.subject_id, cfg.phase.value)
    recorder.set_export_options(config_to_dict(cfg)["session_export"])
    recorder.set_metadata({"config": config_to_dict(cfg)})
    recorder.log_streamdio_event("analog_input", 13, 1, 1000, 0.000)
    recorder.log_streamdio_event("analog_input", 13, 2, 2000, 0.002)
    recorder.log_streamdio_event("camera", None, None, 3000, 0.004)
    recorder.add_trial(_trial())

    recorder.finalize(_summary())

    bundle = recorder.base_path.with_name(f"{recorder.base_path.name}_bundle")
    chunks = sorted((bundle / "continuous_streams").glob("streamdio_chunk_*.npz"))
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    artifact_paths = {item["path"].replace("\\", "/") for item in manifest["artifacts"]}
    assert chunks
    assert any("continuous_streams" in path for path in artifact_paths)
    continuous = manifest["continuous_streams"]
    assert continuous["stream"] == "streamdio_events"
    assert continuous["chunk_count"] == len(chunks)
    assert continuous["event_count"] == 3
    assert continuous["counts_by_type"] == {"analog_input": 2, "camera": 1}
    assert continuous["chunks"][0]["path"].startswith("continuous_streams/")
    assert continuous["chunks"][0]["first_module_timestamp_us"] == 1000
    assert continuous["chunks"][0]["first_relative_time_s"] == 0.0
    assert manifest["recorder_stats"]["streamdio_chunk_count"] == len(chunks)
    assert manifest["recorder_stats"]["bundle_artifact_count"] == len(
        manifest["artifacts"]
    )
    assert manifest["recorder_stats"]["bundle_artifact_bytes"] == sum(
        item["bytes"] for item in manifest["artifacts"]
    )
    assert validate_session(bundle)["valid"]

    manifest["continuous_streams"]["chunks"][0]["event_count"] += 1
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    result = validate_session(bundle)

    assert not result["valid"]
    assert any(
        "continuous stream chunk event count mismatch" in error
        for error in result["errors"]
    )


def test_streamdio_bounded_memory_reconstructs_full_npz_from_chunks(tmp_path):
    cfg = SessionConfig(subject_id="S1", n_trials=1, results_dir=str(tmp_path))
    cfg.session_export.streamdio_chunk_s = 0.001
    cfg.session_export.streamdio_memory_mode = "bounded"
    cfg.session_export.streamdio_recent_events = 2
    recorder = SessionRecorder(tmp_path, cfg.subject_id, cfg.phase.value)
    recorder.set_export_options(config_to_dict(cfg)["session_export"])
    recorder.set_metadata({"config": config_to_dict(cfg)})
    for index in range(6):
        recorder.log_streamdio_event("analog_input", 13, index, 1000 + index, index * 0.002)
    recorder.add_trial(_trial())

    recorder.finalize(_summary())

    assert len(recorder.streamdio_events) <= 2
    with np.load(recorder.base_path.with_suffix(".npz"), allow_pickle=True) as data:
        assert len(data["streamdio_events"]) == 6
    bundle = recorder.base_path.with_name(f"{recorder.base_path.name}_bundle")
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"]["streamdio_events"] == 6
    assert manifest["recorder_stats"]["streamdio_memory_mode"] == "bounded"
    assert manifest["recorder_stats"]["streamdio_recent_in_memory_count"] <= 2
    assert manifest["recorder_stats"]["max_streamdio_events_in_memory"] <= 2
    assert validate_session(bundle)["valid"]


def test_streamdio_chunk_write_retries_transient_failure(tmp_path, monkeypatch):
    cfg = SessionConfig(subject_id="S1", n_trials=1, results_dir=str(tmp_path))
    cfg.session_export.streamdio_chunk_s = 0.001
    cfg.session_export.streamdio_memory_mode = "bounded"
    recorder = SessionRecorder(tmp_path, cfg.subject_id, cfg.phase.value)
    recorder.set_export_options(config_to_dict(cfg)["session_export"])
    recorder.set_metadata({"config": config_to_dict(cfg)})
    original = recorder_module._atomic_write_npz
    attempts = 0

    def transient(path, **arrays):
        nonlocal attempts
        if "streamdio_chunk_" in str(path):
            attempts += 1
            if attempts == 1:
                raise OSError("temporary write failure")
        return original(path, **arrays)

    monkeypatch.setattr(recorder_module, "_atomic_write_npz", transient)
    recorder.log_streamdio_event("analog_input", 13, 1, 1000, 0.000)
    recorder.log_streamdio_event("analog_input", 13, 2, 2000, 0.002)
    recorder.add_trial(_trial())

    recorder.finalize(_summary())

    assert attempts >= 2
    assert recorder._streamdio_chunk_error is None
    with np.load(recorder.base_path.with_suffix(".npz"), allow_pickle=True) as data:
        assert len(data["streamdio_events"]) == 2


def test_streamdio_failed_chunk_rows_fall_back_to_primary_npz(tmp_path, monkeypatch):
    cfg = SessionConfig(subject_id="S1", n_trials=1, results_dir=str(tmp_path))
    cfg.session_export.streamdio_chunk_s = 0.001
    cfg.session_export.streamdio_memory_mode = "bounded"
    cfg.session_export.streamdio_recent_events = 1
    recorder = SessionRecorder(tmp_path, cfg.subject_id, cfg.phase.value)
    recorder.set_export_options(config_to_dict(cfg)["session_export"])
    recorder.set_metadata({"config": config_to_dict(cfg)})
    original = recorder_module._atomic_write_npz

    def fail_chunks(path, **arrays):
        if "streamdio_chunk_" in str(path):
            raise OSError("persistent chunk failure")
        return original(path, **arrays)

    monkeypatch.setattr(recorder_module, "_atomic_write_npz", fail_chunks)
    for index in range(4):
        recorder.log_streamdio_event(
            "analog_input", 13, index, 1000 + index, index * 0.002
        )
    recorder.add_trial(_trial())

    recorder.finalize(_summary())

    assert "persistent chunk failure" in str(recorder._streamdio_chunk_error)
    with np.load(recorder.base_path.with_suffix(".npz"), allow_pickle=True) as data:
        assert len(data["streamdio_events"]) == 4
    bundle = recorder.base_path.with_name(f"{recorder.base_path.name}_bundle")
    assert validate_session(bundle)["valid"]


def test_bounded_finalize_prefers_complete_wal_over_incomplete_chunks(
    tmp_path,
    monkeypatch,
):
    recorder = SessionRecorder(tmp_path, "S1", "decision_making")
    recorder.set_export_options(
        {
            "streamdio_chunk_s": 0.001,
            "streamdio_memory_mode": "bounded",
            "streamdio_recent_events": 1,
            "manifest": False,
            "config_snapshot": False,
            "export_csv_on_finalize": False,
            "export_timeline": False,
        }
    )
    for index in range(10):
        recorder.log_streamdio_event(
            "analog_input",
            13,
            index,
            index * 1_000,
            index * 0.002,
        )
    recorder._close_streamdio_wal(durable=True)
    recorder._finalize_streamdio_chunks()
    monkeypatch.setattr(recorder, "_load_streamdio_chunk_rows", lambda: [])

    rows = recorder._streamdio_rows_for_finalize()

    assert len(rows) == 10
    assert [row[2] for row in rows] == list(range(10))


def test_structured_event_payload_is_not_truncated(tmp_path):
    recorder = SessionRecorder(tmp_path, "S1", "decision_making")
    payload = {"long_value": "x" * 5000}
    recorder.log_event("long_payload", payload)

    events = recorder._events_to_structured()

    row = next(item for item in events if item["event_type"] == "long_payload")
    assert json.loads(str(row["payload_json"])) == payload


def test_debug_event_ring_exports_recent_events_without_polluting_semantic_events(tmp_path):
    cfg = SessionConfig(subject_id="S1", n_trials=1, results_dir=str(tmp_path))
    recorder = SessionRecorder(tmp_path, cfg.subject_id, cfg.phase.value)
    recorder.set_export_options(config_to_dict(cfg)["session_export"])
    recorder.set_metadata({"config": config_to_dict(cfg)})
    for index in range(2050):
        recorder.log_debug_event("choice_loop_stall", {"frame": index})
    recorder.add_trial(_trial())

    recorder.finalize(_summary())

    bundle = recorder.base_path.with_name(f"{recorder.base_path.name}_bundle")
    sidecar = json.loads(recorder.base_path.with_suffix(".json").read_text(encoding="utf-8"))
    debug_events = sidecar["debug_events"]
    assert "choice_loop_stall" not in [event["event_type"] for event in sidecar["events"]]
    assert debug_events["ring_capacity"] == 2048
    assert debug_events["recent_count"] == 2048
    assert debug_events["total_count"] == 2050
    assert debug_events["evicted_count"] == 2
    assert (bundle / "exports" / "debug_events_recent.json").exists()
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["recorder_stats"]["debug_event_total_count"] == 2050
    with np.load(recorder.base_path.with_suffix(".npz"), allow_pickle=True) as data:
        assert "debug_events_json" in data.files
        assert len(data["debug_events_json"]) == 2048
    assert validate_session(bundle)["valid"]

    debug_path = bundle / "exports" / "debug_events_recent.json"
    debug_rows = json.loads(debug_path.read_text(encoding="utf-8"))
    debug_path.write_text(json.dumps(debug_rows[:-1], indent=2), encoding="utf-8")
    _refresh_manifest_artifact_sha(bundle / "manifest.json", debug_path)

    result = validate_session(bundle)

    assert not result["valid"]
    assert any("debug events recent count mismatch" in error for error in result["errors"])


def test_optional_alyx_registration_writes_skipped_result_without_credentials(tmp_path, monkeypatch):
    monkeypatch.delenv("ALYX_API_URL", raising=False)
    monkeypatch.delenv("ALYX_TOKEN", raising=False)
    cfg = SessionConfig(subject_id="S1", n_trials=1, results_dir=str(tmp_path))
    cfg.integrations.mm_alyx_register = True
    recorder = SessionRecorder(tmp_path, cfg.subject_id, cfg.phase.value)
    recorder.set_export_options(config_to_dict(cfg)["session_export"])
    recorder.set_metadata({"config": config_to_dict(cfg)})
    recorder.add_trial(_trial())

    recorder.finalize(_summary())

    bundle = recorder.base_path.with_name(f"{recorder.base_path.name}_bundle")
    alyx_path = bundle / "exports" / "alyx_registration.json"
    assert alyx_path.exists()
    alyx_result = json.loads(alyx_path.read_text(encoding="utf-8"))
    assert alyx_result["status"] == "skipped"
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    artifact_paths = {item["path"].replace("\\", "/") for item in manifest["artifacts"]}
    assert "exports/alyx_registration.json" in artifact_paths
    assert validate_session(bundle)["valid"]

    alyx_result["status"] = "maybe"
    alyx_path.write_text(json.dumps(alyx_result, indent=2), encoding="utf-8")
    _refresh_manifest_artifact_sha(bundle / "manifest.json", alyx_path)

    result = validate_session(bundle)

    assert not result["valid"]
    assert any("alyx registration invalid status" in error for error in result["errors"])


def test_mm_hierarchy_output_uses_date_and_run_folder(tmp_path):
    first, base_name = resolve_session_output(
        tmp_path,
        subject_id="S1",
        phase="decision_making",
        layout="mm_hierarchy",
        now=datetime(2026, 6, 26, 12, 0, 0),
    )
    first.mkdir(parents=True)
    second, second_base_name = resolve_session_output(
        tmp_path,
        subject_id="S1",
        phase="decision_making",
        layout="mm_hierarchy",
        now=datetime(2026, 6, 26, 12, 1, 0),
    )

    assert first == tmp_path / "decision_making" / "S1" / "2026-06-26" / "001"
    assert second == tmp_path / "decision_making" / "S1" / "2026-06-26" / "002"
    assert base_name == "session"
    assert second_base_name == "session"


def test_session_bundle_facade_wraps_recorder_and_stream_registration(tmp_path):
    cfg = SessionConfig(subject_id="S1", n_trials=1, results_dir=str(tmp_path))
    bundle = SessionBundle.start(cfg)
    bundle.register_streamdio({"stream_port": "COM1"})
    bundle.register_camera({"backend": "spinnaker_sim", "frames_saved": 3})
    bundle.recorder.add_trial(_trial())
    assert bundle.get_stream_summaries()["streamdio"]["stream_port"] == "COM1"

    artifact = bundle.finalize(_summary())

    assert artifact.exists()
    manifest = json.loads((bundle.bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stream_summaries"]["streamdio"]["stream_port"] == "COM1"
    assert manifest["stream_summaries"]["spinnaker_video"]["frames_saved"] == 3


def test_session_bundle_context_finalizes_unfinalized_recorder(tmp_path):
    cfg = SessionConfig(subject_id="S1", n_trials=1, results_dir=str(tmp_path))

    with SessionBundle.start(cfg) as bundle:
        bundle.recorder.add_trial(_trial())

    assert bundle.final_artifact is not None
    assert bundle.final_artifact.exists()
    sidecar = json.loads(
        bundle.recorder.base_path.with_suffix(".json").read_text(encoding="utf-8")
    )
    assert sidecar["summary"]["end_reason"] == "context_exit"
    assert sidecar["summary"]["n_trials_completed"] == 1
    assert validate_session(bundle.bundle_dir)["valid"]


def test_session_bundle_context_finalizes_and_reraises_error(tmp_path):
    cfg = SessionConfig(subject_id="S1", n_trials=1, results_dir=str(tmp_path))

    try:
        with SessionBundle.start(cfg) as bundle:
            raise RuntimeError("test failure")
    except RuntimeError as exc:
        assert str(exc) == "test failure"
    else:
        raise AssertionError("SessionBundle suppressed the context error")

    assert bundle.final_artifact is not None
    sidecar = json.loads(
        bundle.recorder.base_path.with_suffix(".json").read_text(encoding="utf-8")
    )
    assert sidecar["summary"]["end_reason"] == "context_error"
    assert any(
        event["event_type"] == "session_bundle_context_error"
        for event in sidecar["events"]
    )
    assert validate_session(bundle.bundle_dir)["valid"]
