from pathlib import Path
from tempfile import TemporaryDirectory

from hf_dm.config import (
    ConfigValidationError,
    SessionConfig,
    config_validation_issues,
    validate_session_config,
)
from hf_dm.hardware.bringup_constants import FRAME2TTL_INPUT_CHANNEL, FRAME2TTL_INPUT_PIN


def test_lick_cap_touch_thresholds_load_from_yaml():
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "config.yaml"
        path.write_text(
            """
licks:
  cap_touch_min_high_threshold: 250
  cap_touch_delta_high: 120
  cap_touch_delta_low: 90
""",
            encoding="utf-8",
        )

        config = SessionConfig.from_yaml(path)

    assert config.licks.cap_touch_min_high_threshold == 250
    assert config.licks.cap_touch_delta_high == 120
    assert config.licks.cap_touch_delta_low == 90


def test_frame2ttl_defaults_and_yaml_load():
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "config.yaml"
        path.write_text(
            """
frame2ttl:
  enabled: true
  input_channel: 5
  input_pin: 6
  patch_size_px: 72
  patch_corner: bottom_right
""",
            encoding="utf-8",
        )

        config = SessionConfig.from_yaml(path)

    assert config.frame2ttl.enabled
    assert config.frame2ttl.input_channel == FRAME2TTL_INPUT_CHANNEL
    assert config.frame2ttl.input_pin == FRAME2TTL_INPUT_PIN
    assert config.frame2ttl.patch_size_px == 72
    assert config.frame2ttl.patch_corner == "bottom_right"


def test_session_export_defaults_and_yaml_load():
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "config.yaml"
        path.write_text(
            """
session_export:
  layout: mm_hierarchy
  manifest: true
  config_snapshot: true
  export_csv_on_finalize: false
  export_timeline: false
  streamdio_chunk_s: 45
  streamdio_memory_mode: bounded
  streamdio_recent_events: 123
  streamdio_wal_fsync_s: 0.5
  cli_overrides:
    - source: pytest
      values:
        monitor.enabled: true
""",
            encoding="utf-8",
        )

        config = SessionConfig.from_yaml(path)

    assert config.session_export.layout == "mm_hierarchy"
    assert config.session_export.manifest
    assert config.session_export.config_snapshot
    assert not config.session_export.export_csv_on_finalize
    assert not config.session_export.export_timeline
    assert config.session_export.streamdio_chunk_s == 45
    assert config.session_export.streamdio_memory_mode == "bounded"
    assert config.session_export.streamdio_recent_events == 123
    assert config.session_export.streamdio_wal_fsync_s == 0.5
    assert config.session_export.cli_overrides == [
        {"source": "pytest", "values": {"monitor.enabled": True}}
    ]


def test_production_defaults_use_mm_layout_and_ten_choice_history():
    config = SessionConfig()

    assert config.session_export.layout == "mm_hierarchy"
    assert config.monitor.trial_history == 10


def test_dm_200_uses_bounded_streamdio_chunks():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "dm_200.yaml"
    config = SessionConfig.from_yaml(config_path)

    assert config.session_export.layout == "mm_hierarchy"
    assert config.session_export.streamdio_chunk_s == 45
    assert config.session_export.streamdio_memory_mode == "bounded"
    assert config.monitor.trial_history == 10
    assert config.integrations.mm_flir_roi_gui


def test_official_profiles_use_mm_layout_visual_roi_and_ten_choices():
    config_dir = Path(__file__).resolve().parents[1] / "configs"

    for filename in ("centering_50.yaml", "dm_50.yaml", "dm_100.yaml", "dm_200.yaml"):
        config = SessionConfig.from_yaml(config_dir / filename)
        assert config.session_export.layout == "mm_hierarchy", filename
        assert config.monitor.trial_history == 10, filename
        assert config.integrations.mm_flir_roi_gui, filename
        assert config.session_export.streamdio_wal_fsync_s == 1.0, filename


def test_semantic_config_validation_reports_all_dangerous_values():
    config = SessionConfig(n_trials=0)
    config.camera.frequency_hz = 30
    config.camera.exposure_us = 40_000
    config.session_export.streamdio_memory_mode = "bounded"
    config.session_export.streamdio_chunk_s = 0
    config.task.modality = "mystery"

    issues = config_validation_issues(config)
    fields = {issue.field for issue in issues}

    assert "n_trials" in fields
    assert "camera.exposure_us" in fields
    assert "session_export.streamdio_chunk_s" in fields
    assert "task.modality" in fields
    try:
        validate_session_config(config)
    except ConfigValidationError as exc:
        assert len(exc.issues) == len(issues)
    else:
        raise AssertionError("invalid hardware configuration was accepted")


def test_task_balancing_config_loads_and_validates(tmp_path):
    path = tmp_path / "balanced.yaml"
    path.write_text(
        "n_trials: 12\n"
        "task:\n"
        "  n_free_trials: 8\n"
        "  n_forced_trials: 4\n"
        "  side_balancing: blocked\n"
        "  max_consecutive_side: 3\n"
        "  max_consecutive_forced: 2\n",
        encoding="utf-8",
    )

    config = SessionConfig.from_yaml(path)

    assert config.task.side_balancing == "blocked"
    assert config.task.max_consecutive_side == 3
    assert config.task.max_consecutive_forced == 2


def test_reward_calibration_path_loads_without_changing_default(tmp_path):
    default = SessionConfig()
    path = tmp_path / "calibrated.yaml"
    path.write_text("reward:\n  calibration_path: calibrations/valve.json\n", encoding="utf-8")

    configured = SessionConfig.from_yaml(path)

    assert default.reward.calibration_path is None
    assert configured.reward.calibration_path == "calibrations/valve.json"
    assert configured.session_export.source_config_path == str(path.resolve())


def test_animal_use_and_welfare_release_fields_load(tmp_path):
    path = tmp_path / "animal.yaml"
    path.write_text(
        "animal_use:\n"
        "  enabled: true\n"
        "  certification_path: certifications/rig.json\n"
        "  max_certification_age_days: 14\n"
        "  hardware_failsafe_revision: watchdog-r2\n"
        "welfare:\n"
        "  max_cumulative_reward_ul: 750\n"
        "  require_current_weight: true\n"
        "  max_weight_age_days: 2\n",
        encoding="utf-8",
    )

    config = SessionConfig.from_yaml(path)

    assert config.animal_use.enabled is True
    assert config.animal_use.certification_path == "certifications/rig.json"
    assert config.animal_use.max_certification_age_days == 14
    assert config.animal_use.hardware_failsafe_revision == "watchdog-r2"
    assert config.welfare.max_cumulative_reward_ul == 750
    assert config.welfare.require_current_weight is True
    assert config.welfare.max_weight_age_days == 2


def test_welfare_log_config_loads(tmp_path):
    path = tmp_path / "welfare.yaml"
    path.write_text(
        "welfare:\n"
        "  subjects_dir: local_subjects\n"
        "  water_target_ml: 1.25\n"
        "  append_session_water: false\n",
        encoding="utf-8",
    )

    config = SessionConfig.from_yaml(path)

    assert config.welfare.subjects_dir == "local_subjects"
    assert config.welfare.water_target_ml == 1.25
    assert not config.welfare.append_session_water


def test_task_balancing_validation_rejects_impossible_forced_cap():
    config = SessionConfig(n_trials=10)
    config.task.n_free_trials = 1
    config.task.n_forced_trials = 9
    config.task.max_consecutive_forced = 2

    fields = {issue.field for issue in config_validation_issues(config)}

    assert "task.max_consecutive_forced" in fields


def test_monitor_record_path_loads_from_yaml():
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "config.yaml"
        path.write_text(
            """
monitor:
  streamdio_status_interval_s: 12.5
  record_path: auto
""",
            encoding="utf-8",
        )

        config = SessionConfig.from_yaml(path)

    assert config.monitor.record_path == "auto"
    assert config.monitor.streamdio_status_interval_s == 12.5


def test_camera_roi_loads_from_yaml():
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "config.yaml"
        path.write_text(
            """
camera:
  roi_width: 640
  roi_height: 480
  roi_offset_x: 32
  roi_offset_y: 16
""",
            encoding="utf-8",
        )

        config = SessionConfig.from_yaml(path)

    assert config.camera.roi_width == 640
    assert config.camera.roi_height == 480
    assert config.camera.roi_offset_x == 32
    assert config.camera.roi_offset_y == 16


def test_video_export_config_loads_from_yaml():
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "config.yaml"
        path.write_text(
            """
video:
  record_video: false
  export_spinnaker_summary: true
  duplicate_preview_video: true
  preview_video_codec: FFV1
  preview_video_fps: 12.5
""",
            encoding="utf-8",
        )

        config = SessionConfig.from_yaml(path)

    assert config.video.record_video is False
    assert not config.camera.record_video
    assert config.video.export_spinnaker_summary
    assert config.video.duplicate_preview_video
    assert config.video.preview_video_codec == "FFV1"
    assert config.video.preview_video_fps == 12.5


def test_integrations_config_loads_from_yaml():
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "config.yaml"
        path.write_text(
            """
integrations:
  mm_soundcard_status: true
  mm_alyx_register: true
  mm_flir_roi_gui: false
  alyx_api_url: https://alyx.example.test/api
  alyx_token: secret
  alyx_project: hf_dm
  alyx_user: rig
  alyx_timeout_s: 2.5
""",
            encoding="utf-8",
        )

        config = SessionConfig.from_yaml(path)

    assert config.integrations.mm_soundcard_status
    assert config.integrations.mm_alyx_register
    assert not config.integrations.mm_flir_roi_gui
    assert config.integrations.alyx_api_url == "https://alyx.example.test/api"
    assert config.integrations.alyx_token == "secret"
    assert config.integrations.alyx_project == "hf_dm"
    assert config.integrations.alyx_user == "rig"
    assert config.integrations.alyx_timeout_s == 2.5
