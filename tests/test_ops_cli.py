import json
import sys
import types
from pathlib import Path

from hf_dm.config import SessionConfig, config_to_dict
from hf_dm.data.recorder import SessionRecorder, SessionSummary, TrialRecord
from hf_dm.data.session_export import sha256_file, validate_session
from hf_dm.ops_cli import main


def _make_session(tmp_path):
    cfg = SessionConfig(subject_id="S1", n_trials=1, results_dir=str(tmp_path))
    recorder = SessionRecorder(tmp_path, cfg.subject_id, cfg.phase.value)
    recorder.set_export_options(config_to_dict(cfg)["session_export"])
    recorder.set_metadata({"config": config_to_dict(cfg)})
    recorder.add_trial(
        TrialRecord(
            trial_index=1,
            phase=cfg.phase.value,
            force_direction="cw",
            choice="cw",
            matched=True,
            reward_duration_s=0.08,
            responded=True,
            response_time_s=0.1,
            net_pulses=1,
            cw_pulses=1,
            ccw_pulses=0,
            final_position_deg=1.0,
            force_samples=0,
            force_last=None,
            force_mean=None,
            force_min=None,
            force_max=None,
            trial_start_wall_s=1.0,
            trial_start_mono_s=1.0,
        )
    )
    recorder.finalize(
        SessionSummary(
            subject_id="S1",
            phase=cfg.phase.value,
            n_trials_planned=1,
            n_trials_completed=1,
            end_reason="completed",
            consecutive_failures=0,
            session_start_wall_s=1.0,
            session_end_wall_s=2.0,
        )
    )
    return recorder


def test_ops_cli_export_and_validate_session(tmp_path, monkeypatch, capsys):
    recorder = _make_session(tmp_path)
    json_path = recorder.base_path.with_suffix(".json")

    monkeypatch.setattr(sys, "argv", ["hf-dm", "export-session", str(json_path)])
    main()
    export_result = json.loads(capsys.readouterr().out)
    assert export_result["manifest"].endswith("manifest.json")

    monkeypatch.setattr(sys, "argv", ["hf-dm", "validate-session", export_result["bundle_dir"]])
    main()
    validate_result = json.loads(capsys.readouterr().out)
    assert validate_result["valid"]


def test_ops_cli_camera_roi_setup_writes_yaml(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "config.yaml"
    output_path = tmp_path / "roi.yaml"
    config_path.write_text("camera:\n  frequency_hz: 30\n", encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hf-dm",
            "camera-roi-setup",
            "--config",
            str(config_path),
            "--output",
            str(output_path),
            "--width",
            "640",
            "--height",
            "480",
            "--offset-x",
            "12",
            "--offset-y",
            "8",
        ],
    )
    main()

    result = json.loads(capsys.readouterr().out)
    written = output_path.read_text(encoding="utf-8")
    assert result["roi"] == {"width": 640, "height": 480, "offset_x": 12, "offset_y": 8}
    assert "roi_width: 640" in written
    assert "roi_height: 480" in written


def test_ops_cli_camera_roi_setup_interactive_prompts(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "config.yaml"
    output_path = tmp_path / "roi.yaml"
    config_path.write_text("camera:\n  frequency_hz: 30\n", encoding="utf-8")
    answers = iter(["800", "600", "16", "24"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hf-dm",
            "camera-roi-setup",
            "--config",
            str(config_path),
            "--output",
            str(output_path),
            "--interactive",
        ],
    )

    main()

    result = json.loads(capsys.readouterr().out)
    written = output_path.read_text(encoding="utf-8")
    assert result["roi"] == {"width": 800, "height": 600, "offset_x": 16, "offset_y": 24}
    assert "roi_width: 800" in written
    assert "roi_offset_y: 24" in written


def test_ops_cli_camera_roi_setup_selects_visual_roi(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "config.yaml"
    output_path = tmp_path / "roi.yaml"
    config_path.write_text(
        "camera:\n  frequency_hz: 30\n  serial_number: CAM1\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "hf_dm.camera.roi_gui.select_spinnaker_roi",
        lambda **_kwargs: {"width": 640, "height": 480, "offset_x": 16, "offset_y": 8},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hf-dm",
            "camera-roi-setup",
            "--config",
            str(config_path),
            "--output",
            str(output_path),
            "--visual",
        ],
    )

    main()

    result = json.loads(capsys.readouterr().out)
    written = output_path.read_text(encoding="utf-8")
    assert result["mode"] == "visual"
    assert not result["cancelled"]
    assert result["roi"] == {"width": 640, "height": 480, "offset_x": 16, "offset_y": 8}
    assert "roi_width: 640" in written
    assert "roi_offset_x: 16" in written


def test_ops_cli_camera_roi_setup_cancel_preserves_config(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "config.yaml"
    original = "camera:\n  frequency_hz: 30\n"
    config_path.write_text(original, encoding="utf-8")
    monkeypatch.setattr(
        "hf_dm.camera.roi_gui.select_spinnaker_roi",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["hf-dm", "camera-roi-setup", "--config", str(config_path), "--visual"],
    )

    main()

    result = json.loads(capsys.readouterr().out)
    assert result["cancelled"]
    assert result["output"] is None
    assert config_path.read_text(encoding="utf-8") == original


def test_ops_cli_camera_roi_setup_rejects_visual_numeric_mix(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("camera:\n  frequency_hz: 30\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hf-dm",
            "camera-roi-setup",
            "--config",
            str(config_path),
            "--visual",
            "--width",
            "640",
        ],
    )

    try:
        main()
    except SystemExit as exc:
        assert "cannot be combined" in str(exc)
    else:
        raise AssertionError("visual and numeric ROI modes were accepted together")


def test_ops_cli_export_preview_video(tmp_path, monkeypatch, capsys):
    video_dir = tmp_path / "video"
    video_dir.mkdir()
    (video_dir / "recording_summary.json").write_text(
        '{"target_trigger_hz": 10}',
        encoding="utf-8",
    )
    (video_dir / "frame_000000000.pgm").write_bytes(b"P5\n2 2\n255\n\x00\x01\x02\x03")

    class FakeWriter:
        def isOpened(self):
            return True

        def write(self, frame):
            assert tuple(frame.shape) == (2, 2)

        def release(self):
            pass

    monkeypatch.setitem(
        sys.modules,
        "cv2",
        types.SimpleNamespace(
            VideoWriter_fourcc=lambda *codec: 0,
            VideoWriter=lambda *args: FakeWriter(),
        ),
    )
    monkeypatch.setattr(sys, "argv", ["hf-dm", "export-preview-video", str(video_dir)])

    main()

    result = json.loads(capsys.readouterr().out)
    assert result["frames_written"] == 1
    assert result["output_path"].endswith("preview_ffv1.avi")


def test_ops_cli_rig_preflight_writes_report_and_refreshes_manifest(
    tmp_path,
    monkeypatch,
    capsys,
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "subject_id: S1",
                f"results_dir: {tmp_path.as_posix()}",
                "camera:",
                "  record_video: false",
                "monitor:",
                "  enabled: false",
            ]
        ),
        encoding="utf-8",
    )

    class FakeSession:
        def __init__(self, config):
            self.config = config
            self.recorder = None
            self.closed = False
            self.dio = types.SimpleNamespace(
                get_module_status=lambda: {
                    "config_port": "COM_CONFIG",
                    "stream_port": "COM_STREAM",
                    "module_info": {"firmware_version": 4},
                }
            )

        def connect(self):
            self.recorder = SessionRecorder(
                tmp_path,
                self.config.subject_id,
                self.config.phase.value,
            )
            self.recorder.set_export_options(config_to_dict(self.config)["session_export"])
            self.recorder.set_metadata({"config": config_to_dict(self.config)})
            self.recorder.log_streamdio_event("camera", None, None, 1000, 0.1)

        def preflight(self, duration_s=2.0):
            assert self.recorder is not None
            self.recorder.log_event("preflight_start", {"duration_s": duration_s})
            self.recorder.log_event(
                "preflight_end",
                {
                    "duration_s": duration_s,
                    "camera_events_seen": 1,
                    "force_samples_seen": 0,
                },
            )
            summary = SessionSummary(
                subject_id=self.config.subject_id,
                phase=self.config.phase.value,
                n_trials_planned=0,
                n_trials_completed=0,
                end_reason="preflight",
                consecutive_failures=0,
                session_start_wall_s=1.0,
                session_end_wall_s=2.0,
            )
            self.recorder.finalize(summary)
            return summary

        def close(self):
            self.closed = True

    monkeypatch.setattr("hf_dm.ops_cli.HFDecisionSession", FakeSession)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hf-dm",
            "rig-preflight",
            "--config",
            str(config_path),
            "--duration",
            "0.25",
            "--no-monitor",
        ],
    )

    main()

    result = json.loads(capsys.readouterr().out)
    report_path = Path(result["report"])
    manifest_path = Path(result["manifest"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_paths = [
        str(item["path"]).replace("\\", "/")
        for item in manifest.get("artifacts", [])
    ]

    assert result["end_reason"] == "preflight"
    assert result["validation"]["valid"] is True
    assert report["summary"]["end_reason"] == "preflight"
    assert report["duration_s"] == 0.25
    assert report["streamdio"]["module_status"]["config_port"] == "COM_CONFIG"
    assert report["streamdio"]["module_status"]["module_info"]["firmware_version"] == 4
    assert any(
        path.endswith("exports/rig_preflight_report.json")
        for path in artifact_paths
    )

    report["streamdio"]["event_count"] = 999
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for item in manifest["artifacts"]:
        resolved = (manifest_path.parent / item["path"]).resolve()
        if resolved == report_path.resolve():
            item["sha256"] = sha256_file(report_path)
            item["bytes"] = report_path.stat().st_size
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    validation = validate_session(manifest_path.parent)

    assert not validation["valid"]
    assert any(
        "rig preflight streamdio event count mismatch" in error
        for error in validation["errors"]
    )
