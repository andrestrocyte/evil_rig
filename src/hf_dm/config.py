"""Session and task configuration (YAML-loadable)."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from hf_dm.hardware.bringup_constants import (
    BringupPorts,
    CAP_TOUCH_POWER_OUTPUT_CHANNEL,
    DEFAULT_BAUDRATE,
    DIRECTION_INPUT_CHANNEL,
    FRAME2TTL_INPUT_CHANNEL,
    FRAME2TTL_INPUT_PIN,
    FORCE_ANALOG_CHANNEL,
    LICK_INPUT_CHANNEL,
    PULSE_INPUT_CHANNEL,
    SPEAKER_NOISE_SAMPLE_INTERVAL_US,
)
from hf_dm.streamdio.wheel import WheelKinematics


class TrainingPhase(str, Enum):
    CENTERING = "centering"
    DECISION_MAKING = "decision_making"


@dataclass
class DisplayConfig:
    width: int | None = None
    height: int | None = None
    distance_mm: float = 100.0
    pixel_size_mm: float = 0.114
    choice_distance_deg: float = 35.0
    choice_center_threshold_deg: float = 1.0
    rotated_choice_axis: bool = True
    display_index: int | None = 1
    auto_size_to_display: bool = True
    window_x: int | None = None
    window_y: int | None = None
    fullscreen: bool = False
    noframe: bool = True
    fps: int = 60
    pre_trial_black_s: float = 0.0
    feedback_hold_s: float = 0.6
    background_rgb: tuple[int, int, int] = (0, 0, 0)
    cue_rgb: tuple[int, int, int] = (245, 245, 245)
    target_rgb: tuple[int, int, int] = (80, 80, 80)
    matched_rgb: tuple[int, int, int] = (72, 198, 135)
    unmatched_rgb: tuple[int, int, int] = (88, 180, 255)
    timeout_rgb: tuple[int, int, int] = (235, 94, 94)
    shape_radius_px: int = 90
    target_radius_px: int = 120
    show_debug_overlay: bool = False

    @property
    def axis(self) -> str:
        return "vertical" if self.rotated_choice_axis else "horizontal"


@dataclass
class Frame2TTLConfig:
    enabled: bool = True
    input_channel: int = FRAME2TTL_INPUT_CHANNEL
    input_pin: int = FRAME2TTL_INPUT_PIN
    patch_size_px: int = 64
    patch_margin_px: int = 16
    patch_corner: str = "bottom_right"
    barcode_enabled: bool = True
    barcode_preamble_high_frames: int = 12
    barcode_preamble_low_frames: int = 12
    barcode_one_high_frames: int = 6
    barcode_zero_high_frames: int = 6
    barcode_inter_bit_low_frames: int = 6
    choice_flash_enabled: bool = True


@dataclass
class TimingConfig:
    quiescence_s: float = 3.0
    inter_trial_min_s: float = 3.0
    choice_timeout_s: float = 20.0
    choice_blank_s: float = 1.0
    post_reward_s: float = 1.0
    timeout_noise_s: float = 2.0
    post_timeout_s: float = 1.0
    choice_loop_stall_s: float = 0.1
    output_verification_timeout_s: float = 0.15


@dataclass
class RewardConfig:
    matched_pump_s: float = 0.08
    opposite_pump_s: float = 0.04
    calibration_path: str | None = None


@dataclass
class TaskDesignConfig:
    modality: str = "force"
    profile_name: str = "config"
    n_free_trials: int | None = None
    n_forced_trials: int = 0
    risk_star_probability: float = 0.66
    delay_star_extra_s: float = 2.0
    side_balancing: str = "bernoulli"
    max_consecutive_side: int = 0
    max_consecutive_forced: int = 0
    selected_mouse_dir: str | None = None
    session_name: str | None = None
    session_dir: str | None = None


@dataclass
class LickConfig:
    channels: tuple[int, ...] = (LICK_INPUT_CHANNEL,)
    high_only: bool = True
    power_output_channel: int = CAP_TOUCH_POWER_OUTPUT_CHANNEL
    power_required: bool = True
    cap_touch_min_high_threshold: int | None = None
    cap_touch_delta_high: int | None = None
    cap_touch_delta_low: int | None = None


@dataclass
class SpeakerConfig:
    sample_interval_us: int = SPEAKER_NOISE_SAMPLE_INTERVAL_US
    frequency_hz: float = 6000.0
    volume: float = 1.0
    gate_active_high: bool = True
    required: bool = True


@dataclass
class EngagementConfig:
    idle_timeout_s: float = 60.0
    max_consecutive_failures: int = 10


@dataclass
class CameraConfig:
    frequency_hz: int = 30
    exposure_us: int = 10_000
    record_video: bool = True
    required: bool = True
    video_notes: str = "Align external recorder to StreamDIO camera TTL; timestamps saved in session npz."
    backend: str = "spinnaker"
    serial_number: str | None = None
    gain_db: float | None = None
    gamma: float | None = None
    pixel_format: str = "Mono8"
    acquisition_timeout_ms: int = 1000
    save_dir: str = "data/videos"
    roi_width: int | None = None
    roi_height: int | None = None
    roi_offset_x: int | None = None
    roi_offset_y: int | None = None


@dataclass
class VideoExportConfig:
    record_video: bool | None = None
    export_spinnaker_summary: bool = True
    duplicate_preview_video: bool = False
    preview_video_codec: str = "FFV1"
    preview_video_fps: float | None = None


@dataclass
class IntegrationConfig:
    mm_soundcard_status: bool = True
    mm_alyx_register: bool = False
    mm_flir_roi_gui: bool = False
    alyx_api_url: str | None = None
    alyx_project: str | None = None
    alyx_user: str | None = None
    alyx_timeout_s: float = 5.0


@dataclass
class WelfareConfig:
    subjects_dir: str = "subjects"
    water_target_ml: float | None = None
    append_session_water: bool = True
    max_cumulative_reward_ul: float | None = None
    require_current_weight: bool = False
    max_weight_age_days: int = 1


@dataclass
class AnimalUseConfig:
    """Fail-closed release requirements applied only to real animal sessions."""

    enabled: bool = False
    certification_path: str | None = None
    max_certification_age_days: int = 30
    max_calibration_age_days: int = 30
    max_calibration_residual_ul: float = 0.5
    require_clean_git: bool = True
    require_local_storage: bool = True
    require_camera: bool = True
    require_speaker: bool = True
    require_licks: bool = True
    require_operator_confirmation: bool = True
    hardware_failsafe_revision: str | None = None


@dataclass
class MonitorConfig:
    enabled: bool = False
    display_index: int | None = 0
    window_x: int | None = None
    window_y: int | None = None
    width: int = 1500
    height: int = 940
    fps: int = 30
    history_s: float = 20.0
    trial_history: int = 10
    queue_size: int = 512
    force_sample_hz: float = 80.0
    wheel_sample_hz: float = 60.0
    screen_fps: float = 12.0
    camera_fps: float = 10.0
    camera_preview_width: int = 520
    camera_preview_height: int = 390
    task_preview_width: int = 220
    task_preview_height: int = 120
    task_preview_rotate_left: bool = True
    streamdio_status_interval_s: float = 30.0
    record_path: str | None = None


@dataclass
class SessionExportConfig:
    layout: str = "mm_hierarchy"
    manifest: bool = True
    config_snapshot: bool = True
    export_csv_on_finalize: bool = True
    export_timeline: bool = True
    streamdio_chunk_s: float = 0.0
    streamdio_memory_mode: str = "full"
    streamdio_recent_events: int = 50000
    streamdio_wal_fsync_s: float = 1.0
    source_config_path: str | None = None
    cli_overrides: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class SessionConfig:
    subject_id: str = "mouse01"
    phase: TrainingPhase = TrainingPhase.DECISION_MAKING
    n_trials: int = 50
    random_seed: int | None = None
    centering_threshold_deg: float = 8.0
    # Centering phase: reward when |position| is below this threshold.
    ports: BringupPorts = field(default_factory=BringupPorts)
    wheel: WheelKinematics = field(default_factory=WheelKinematics)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    frame2ttl: Frame2TTLConfig = field(default_factory=Frame2TTLConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    task: TaskDesignConfig = field(default_factory=TaskDesignConfig)
    licks: LickConfig = field(default_factory=LickConfig)
    speaker: SpeakerConfig = field(default_factory=SpeakerConfig)
    engagement: EngagementConfig = field(default_factory=EngagementConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    video: VideoExportConfig = field(default_factory=VideoExportConfig)
    integrations: IntegrationConfig = field(default_factory=IntegrationConfig)
    welfare: WelfareConfig = field(default_factory=WelfareConfig)
    animal_use: AnimalUseConfig = field(default_factory=AnimalUseConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    session_export: SessionExportConfig = field(default_factory=SessionExportConfig)
    results_dir: str = "data/sessions"

    @staticmethod
    def from_yaml(path: Path | str) -> SessionConfig:
        path = Path(path)
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if loaded is None:
            raw: dict[str, Any] = {}
        elif isinstance(loaded, dict):
            raw = loaded
        else:
            raise ConfigValidationError(
                [ConfigValidationIssue("<root>", "must be a YAML mapping", loaded)]
            )
        _validate_raw_config(raw)
        phase = TrainingPhase(raw.get("phase", TrainingPhase.DECISION_MAKING.value))
        ports_raw = raw.get("ports", {})
        ports = BringupPorts(
            bpod_port=str(ports_raw.get("bpod_port", "AUTO")),
            dio_config_port=str(ports_raw.get("dio_config_port", "AUTO")),
            dio_stream_port=str(ports_raw.get("dio_stream_port", "AUTO")),
            baudrate=int(ports_raw.get("baudrate", DEFAULT_BAUDRATE)),
            bpod_serial_channel=int(ports_raw.get("bpod_serial_channel", 1)),
        )
        timing_raw = raw.get("timing", {})
        reward_raw = raw.get("reward", {})
        task_raw = raw.get("task", raw.get("session_design", {}))
        lick_raw = raw.get("licks", raw.get("cap_touch", {}))
        speaker_raw = raw.get("speaker", {})
        display_raw = raw.get("display", {})
        engagement_raw = raw.get("engagement", {})
        camera_raw = raw.get("camera", {})
        video_raw = raw.get("video", {})
        integrations_raw = raw.get("integrations", {})
        welfare_raw = raw.get("welfare", {})
        animal_use_raw = raw.get("animal_use", {})
        monitor_raw = raw.get("monitor", {})
        session_export_raw = raw.get("session_export", {})
        wheel_raw = raw.get("wheel", {})
        frame2ttl_raw = raw.get("frame2ttl", {})
        video_record_video = video_raw.get("record_video", None)

        def channels(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
            value = lick_raw.get(name, default)
            if isinstance(value, str):
                value = [part.strip() for part in value.split(",") if part.strip()]
            if isinstance(value, int):
                value = [value]
            if not isinstance(value, (list, tuple)):
                return default
            return tuple(max(1, int(part)) for part in value)

        def rgb(name: str, default: tuple[int, int, int]) -> tuple[int, int, int]:
            value = display_raw.get(name, default)
            if isinstance(value, str):
                value = [part.strip() for part in value.split(",")]
            if not isinstance(value, (list, tuple)) or len(value) != 3:
                return default
            return (
                max(0, min(255, int(value[0]))),
                max(0, min(255, int(value[1]))),
                max(0, min(255, int(value[2]))),
            )

        def optional_lick_int(name: str) -> int | None:
            value = lick_raw.get(name)
            if value in (None, ""):
                return None
            return int(value)

        config = SessionConfig(
            subject_id=str(raw.get("subject_id", "mouse01")),
            phase=phase,
            n_trials=int(raw.get("n_trials", 50)),
            random_seed=raw.get("random_seed"),
            centering_threshold_deg=float(raw.get("centering_threshold_deg", 8.0)),
            ports=ports,
            wheel=WheelKinematics(
                wheel_radius_mm=float(wheel_raw.get("wheel_radius_mm", 31.5)),
                encoder_ticks_per_rev=int(wheel_raw.get("encoder_ticks_per_rev", 1024)),
                gain_deg_per_mm=float(wheel_raw.get("gain_deg_per_mm", 4.0)),
                pulse_channel=int(wheel_raw.get("pulse_channel", PULSE_INPUT_CHANNEL)),
                direction_channel=int(
                    wheel_raw.get("direction_channel", DIRECTION_INPUT_CHANNEL)
                ),
                force_channel=int(wheel_raw.get("force_channel", FORCE_ANALOG_CHANNEL)),
            ),
            display=DisplayConfig(
                width=(
                    int(display_raw["width"])
                    if display_raw.get("width") not in (None, "")
                    else None
                ),
                height=(
                    int(display_raw["height"])
                    if display_raw.get("height") not in (None, "")
                    else None
                ),
                distance_mm=float(display_raw.get("distance_mm", 100.0)),
                pixel_size_mm=float(display_raw.get("pixel_size_mm", 0.114)),
                choice_distance_deg=float(display_raw.get("choice_distance_deg", 35.0)),
                choice_center_threshold_deg=float(
                    display_raw.get("choice_center_threshold_deg", 1.0)
                ),
                rotated_choice_axis=bool(display_raw.get("rotated_choice_axis", True)),
                display_index=(
                    int(display_raw["display_index"])
                    if display_raw.get("display_index") not in (None, "")
                    else None
                ),
                auto_size_to_display=bool(display_raw.get("auto_size_to_display", True)),
                window_x=(
                    int(display_raw["window_x"])
                    if display_raw.get("window_x") not in (None, "")
                    else None
                ),
                window_y=(
                    int(display_raw["window_y"])
                    if display_raw.get("window_y") not in (None, "")
                    else None
                ),
                fullscreen=bool(display_raw.get("fullscreen", False)),
                noframe=bool(display_raw.get("noframe", True)),
                fps=int(display_raw.get("fps", 60)),
                pre_trial_black_s=float(display_raw.get("pre_trial_black_s", 0.0)),
                feedback_hold_s=float(display_raw.get("feedback_hold_s", 0.6)),
                background_rgb=rgb("background_rgb", (0, 0, 0)),
                cue_rgb=rgb("cue_rgb", (245, 245, 245)),
                target_rgb=rgb("target_rgb", (80, 80, 80)),
                matched_rgb=rgb("matched_rgb", (72, 198, 135)),
                unmatched_rgb=rgb("unmatched_rgb", (88, 180, 255)),
                timeout_rgb=rgb("timeout_rgb", (235, 94, 94)),
                shape_radius_px=int(display_raw.get("shape_radius_px", 90)),
                target_radius_px=int(display_raw.get("target_radius_px", 120)),
                show_debug_overlay=bool(display_raw.get("show_debug_overlay", False)),
            ),
            frame2ttl=Frame2TTLConfig(
                enabled=bool(frame2ttl_raw.get("enabled", True)),
                input_channel=int(
                    frame2ttl_raw.get("input_channel", FRAME2TTL_INPUT_CHANNEL)
                ),
                input_pin=int(frame2ttl_raw.get("input_pin", FRAME2TTL_INPUT_PIN)),
                patch_size_px=int(frame2ttl_raw.get("patch_size_px", 64)),
                patch_margin_px=int(frame2ttl_raw.get("patch_margin_px", 16)),
                patch_corner=str(frame2ttl_raw.get("patch_corner", "bottom_right")),
                barcode_enabled=bool(frame2ttl_raw.get("barcode_enabled", True)),
                barcode_preamble_high_frames=int(
                    frame2ttl_raw.get("barcode_preamble_high_frames", 12)
                ),
                barcode_preamble_low_frames=int(
                    frame2ttl_raw.get("barcode_preamble_low_frames", 12)
                ),
                barcode_one_high_frames=int(
                    frame2ttl_raw.get("barcode_one_high_frames", 6)
                ),
                barcode_zero_high_frames=int(
                    frame2ttl_raw.get("barcode_zero_high_frames", 6)
                ),
                barcode_inter_bit_low_frames=int(
                    frame2ttl_raw.get("barcode_inter_bit_low_frames", 6)
                ),
                choice_flash_enabled=bool(
                    frame2ttl_raw.get("choice_flash_enabled", True)
                ),
            ),
            timing=TimingConfig(
                quiescence_s=float(timing_raw.get("quiescence_s", 3.0)),
                inter_trial_min_s=float(timing_raw.get("inter_trial_min_s", 3.0)),
                choice_timeout_s=float(timing_raw.get("choice_timeout_s", 20.0)),
                choice_blank_s=float(timing_raw.get("choice_blank_s", 1.0)),
                post_reward_s=float(timing_raw.get("post_reward_s", 1.0)),
                timeout_noise_s=float(timing_raw.get("timeout_noise_s", 2.0)),
                post_timeout_s=float(timing_raw.get("post_timeout_s", 1.0)),
                choice_loop_stall_s=float(timing_raw.get("choice_loop_stall_s", 0.1)),
                output_verification_timeout_s=float(
                    timing_raw.get("output_verification_timeout_s", 0.15)
                ),
            ),
            reward=RewardConfig(
                matched_pump_s=float(reward_raw.get("matched_pump_s", 0.08)),
                opposite_pump_s=float(reward_raw.get("opposite_pump_s", 0.04)),
                calibration_path=(
                    str(reward_raw["calibration_path"])
                    if reward_raw.get("calibration_path") not in (None, "")
                    else None
                ),
            ),
            task=TaskDesignConfig(
                modality=str(task_raw.get("modality", "force")).lower(),
                profile_name=str(task_raw.get("profile_name", "config")),
                n_free_trials=(
                    int(task_raw["n_free_trials"])
                    if task_raw.get("n_free_trials") not in (None, "")
                    else None
                ),
                n_forced_trials=int(task_raw.get("n_forced_trials", 0)),
                risk_star_probability=float(task_raw.get("risk_star_probability", 0.66)),
                delay_star_extra_s=float(task_raw.get("delay_star_extra_s", 2.0)),
                side_balancing=str(task_raw.get("side_balancing", "bernoulli")).lower(),
                max_consecutive_side=int(task_raw.get("max_consecutive_side", 0)),
                max_consecutive_forced=int(task_raw.get("max_consecutive_forced", 0)),
                selected_mouse_dir=(
                    str(task_raw["selected_mouse_dir"])
                    if task_raw.get("selected_mouse_dir") not in (None, "")
                    else None
                ),
                session_name=(
                    str(task_raw["session_name"])
                    if task_raw.get("session_name") not in (None, "")
                    else None
                ),
                session_dir=(
                    str(task_raw["session_dir"])
                    if task_raw.get("session_dir") not in (None, "")
                    else None
                ),
            ),
            licks=LickConfig(
                channels=channels(
                    "channels",
                    channels("digital_channels", (LICK_INPUT_CHANNEL,)),
                ),
                high_only=bool(lick_raw.get("high_only", True)),
                power_output_channel=int(
                    lick_raw.get("power_output_channel", CAP_TOUCH_POWER_OUTPUT_CHANNEL)
                ),
                power_required=bool(lick_raw.get("power_required", True)),
                cap_touch_min_high_threshold=optional_lick_int(
                    "cap_touch_min_high_threshold"
                ),
                cap_touch_delta_high=optional_lick_int("cap_touch_delta_high"),
                cap_touch_delta_low=optional_lick_int("cap_touch_delta_low"),
            ),
            speaker=SpeakerConfig(
                sample_interval_us=int(
                    speaker_raw.get(
                        "sample_interval_us",
                        SPEAKER_NOISE_SAMPLE_INTERVAL_US,
                    )
                ),
                frequency_hz=float(speaker_raw.get("frequency_hz", 6000.0)),
                volume=float(speaker_raw.get("volume", 1.0)),
                gate_active_high=bool(speaker_raw.get("gate_active_high", True)),
                required=bool(speaker_raw.get("required", True)),
            ),
            engagement=EngagementConfig(
                idle_timeout_s=float(engagement_raw.get("idle_timeout_s", 60.0)),
                max_consecutive_failures=int(engagement_raw.get("max_consecutive_failures", 10)),
            ),
            camera=CameraConfig(
                frequency_hz=int(camera_raw.get("frequency_hz", 30)),
                exposure_us=int(camera_raw.get("exposure_us", 10_000)),
                record_video=bool(
                    camera_raw.get(
                        "record_video",
                        True if video_record_video is None else video_record_video,
                    )
                ),
                required=bool(camera_raw.get("required", True)),
                video_notes=str(
                    camera_raw.get(
                        "video_notes",
                        "Align external recorder to StreamDIO camera TTL; timestamps saved in session npz.",
                    )
                ),
                backend=str(camera_raw.get("backend", "spinnaker")),
                serial_number=(
                    str(camera_raw["serial_number"])
                    if camera_raw.get("serial_number") not in (None, "")
                    else None
                ),
                gain_db=(
                    float(camera_raw["gain_db"])
                    if camera_raw.get("gain_db") not in (None, "")
                    else None
                ),
                gamma=(
                    float(camera_raw["gamma"])
                    if camera_raw.get("gamma") not in (None, "")
                    else None
                ),
                pixel_format=str(camera_raw.get("pixel_format", "Mono8")),
                acquisition_timeout_ms=int(camera_raw.get("acquisition_timeout_ms", 1000)),
                save_dir=str(camera_raw.get("save_dir", "data/videos")),
                roi_width=(
                    int(camera_raw["roi_width"])
                    if camera_raw.get("roi_width") not in (None, "")
                    else None
                ),
                roi_height=(
                    int(camera_raw["roi_height"])
                    if camera_raw.get("roi_height") not in (None, "")
                    else None
                ),
                roi_offset_x=(
                    int(camera_raw["roi_offset_x"])
                    if camera_raw.get("roi_offset_x") not in (None, "")
                    else None
                ),
                roi_offset_y=(
                    int(camera_raw["roi_offset_y"])
                    if camera_raw.get("roi_offset_y") not in (None, "")
                    else None
                ),
            ),
            video=VideoExportConfig(
                record_video=(
                    bool(video_record_video)
                    if video_record_video not in (None, "")
                    else None
                ),
                export_spinnaker_summary=bool(
                    video_raw.get("export_spinnaker_summary", True)
                ),
                duplicate_preview_video=bool(
                    video_raw.get("duplicate_preview_video", False)
                ),
                preview_video_codec=str(video_raw.get("preview_video_codec", "FFV1")),
                preview_video_fps=(
                    float(video_raw["preview_video_fps"])
                    if video_raw.get("preview_video_fps") not in (None, "")
                    else None
                ),
            ),
            integrations=IntegrationConfig(
                mm_soundcard_status=bool(
                    integrations_raw.get("mm_soundcard_status", True)
                ),
                mm_alyx_register=bool(
                    integrations_raw.get("mm_alyx_register", False)
                ),
                mm_flir_roi_gui=bool(
                    integrations_raw.get("mm_flir_roi_gui", False)
                ),
                alyx_api_url=(
                    str(integrations_raw["alyx_api_url"])
                    if integrations_raw.get("alyx_api_url") not in (None, "")
                    else None
                ),
                alyx_project=(
                    str(integrations_raw["alyx_project"])
                    if integrations_raw.get("alyx_project") not in (None, "")
                    else None
                ),
                alyx_user=(
                    str(integrations_raw["alyx_user"])
                    if integrations_raw.get("alyx_user") not in (None, "")
                    else None
                ),
                alyx_timeout_s=float(integrations_raw.get("alyx_timeout_s", 5.0)),
            ),
            welfare=WelfareConfig(
                subjects_dir=str(welfare_raw.get("subjects_dir", "subjects")),
                water_target_ml=(
                    float(welfare_raw["water_target_ml"])
                    if welfare_raw.get("water_target_ml") not in (None, "")
                    else None
                ),
                append_session_water=bool(
                    welfare_raw.get("append_session_water", True)
                ),
                max_cumulative_reward_ul=(
                    float(welfare_raw["max_cumulative_reward_ul"])
                    if welfare_raw.get("max_cumulative_reward_ul") not in (None, "")
                    else None
                ),
                require_current_weight=bool(
                    welfare_raw.get("require_current_weight", False)
                ),
                max_weight_age_days=int(welfare_raw.get("max_weight_age_days", 1)),
            ),
            animal_use=AnimalUseConfig(
                enabled=bool(animal_use_raw.get("enabled", False)),
                certification_path=(
                    str(animal_use_raw["certification_path"])
                    if animal_use_raw.get("certification_path") not in (None, "")
                    else None
                ),
                max_certification_age_days=int(
                    animal_use_raw.get("max_certification_age_days", 30)
                ),
                max_calibration_age_days=int(
                    animal_use_raw.get("max_calibration_age_days", 30)
                ),
                max_calibration_residual_ul=float(
                    animal_use_raw.get("max_calibration_residual_ul", 0.5)
                ),
                require_clean_git=bool(
                    animal_use_raw.get("require_clean_git", True)
                ),
                require_local_storage=bool(
                    animal_use_raw.get("require_local_storage", True)
                ),
                require_camera=bool(animal_use_raw.get("require_camera", True)),
                require_speaker=bool(animal_use_raw.get("require_speaker", True)),
                require_licks=bool(animal_use_raw.get("require_licks", True)),
                require_operator_confirmation=bool(
                    animal_use_raw.get("require_operator_confirmation", True)
                ),
                hardware_failsafe_revision=(
                    str(animal_use_raw["hardware_failsafe_revision"])
                    if animal_use_raw.get("hardware_failsafe_revision") not in (None, "")
                    else None
                ),
            ),
            monitor=MonitorConfig(
                enabled=bool(monitor_raw.get("enabled", False)),
                display_index=(
                    int(monitor_raw["display_index"])
                    if monitor_raw.get("display_index") not in (None, "")
                    else None
                ),
                window_x=(
                    int(monitor_raw["window_x"])
                    if monitor_raw.get("window_x") not in (None, "")
                    else None
                ),
                window_y=(
                    int(monitor_raw["window_y"])
                    if monitor_raw.get("window_y") not in (None, "")
                    else None
                ),
                width=int(monitor_raw.get("width", 1500)),
                height=int(monitor_raw.get("height", 940)),
                fps=int(monitor_raw.get("fps", 30)),
                history_s=float(monitor_raw.get("history_s", 20.0)),
                trial_history=int(monitor_raw.get("trial_history", 10)),
                queue_size=int(monitor_raw.get("queue_size", 512)),
                force_sample_hz=float(monitor_raw.get("force_sample_hz", 80.0)),
                wheel_sample_hz=float(monitor_raw.get("wheel_sample_hz", 60.0)),
                screen_fps=float(monitor_raw.get("screen_fps", 12.0)),
                camera_fps=float(monitor_raw.get("camera_fps", 10.0)),
                camera_preview_width=int(monitor_raw.get("camera_preview_width", 520)),
                camera_preview_height=int(monitor_raw.get("camera_preview_height", 390)),
                task_preview_width=int(monitor_raw.get("task_preview_width", 220)),
                task_preview_height=int(monitor_raw.get("task_preview_height", 120)),
                task_preview_rotate_left=bool(
                    monitor_raw.get("task_preview_rotate_left", True)
                ),
                streamdio_status_interval_s=float(
                    monitor_raw.get("streamdio_status_interval_s", 30.0)
                ),
                record_path=(
                    str(monitor_raw["record_path"])
                    if monitor_raw.get("record_path") not in (None, "")
                    else None
                ),
            ),
            session_export=SessionExportConfig(
                layout=str(session_export_raw.get("layout", "mm_hierarchy")),
                manifest=bool(session_export_raw.get("manifest", True)),
                config_snapshot=bool(
                    session_export_raw.get("config_snapshot", True)
                ),
                export_csv_on_finalize=bool(
                    session_export_raw.get("export_csv_on_finalize", True)
                ),
                export_timeline=bool(
                    session_export_raw.get("export_timeline", True)
                ),
                streamdio_chunk_s=float(
                    session_export_raw.get("streamdio_chunk_s", 0.0)
                ),
                streamdio_memory_mode=str(
                    session_export_raw.get("streamdio_memory_mode", "full")
                ),
                streamdio_recent_events=int(
                    session_export_raw.get("streamdio_recent_events", 50000)
                ),
                streamdio_wal_fsync_s=float(
                    session_export_raw.get("streamdio_wal_fsync_s", 1.0)
                ),
                source_config_path=(
                    str(session_export_raw["source_config_path"])
                    if session_export_raw.get("source_config_path") not in (None, "")
                    else None
                ),
                cli_overrides=_normalize_cli_overrides(
                    session_export_raw.get("cli_overrides")
                ),
            ),
            results_dir=str(raw.get("results_dir", "data/sessions")),
        )
        if config.session_export.source_config_path is None:
            config.session_export.source_config_path = str(path.resolve())
        return validate_session_config(config)


@dataclass(frozen=True)
class ConfigValidationIssue:
    field: str
    message: str
    value: Any = None


class ConfigValidationError(ValueError):
    def __init__(self, issues: list[ConfigValidationIssue]):
        self.issues = tuple(issues)
        details = "; ".join(f"{issue.field}: {issue.message}" for issue in issues)
        super().__init__(f"Invalid session configuration: {details}")


def _validate_raw_config(raw: dict[str, Any]) -> None:
    """Reject misspelled fields and YAML values that Python would coerce silently."""

    aliases = {"session_design": "task", "cap_touch": "licks"}
    section_types = {
        "ports": BringupPorts,
        "wheel": WheelKinematics,
        "display": DisplayConfig,
        "frame2ttl": Frame2TTLConfig,
        "timing": TimingConfig,
        "reward": RewardConfig,
        "task": TaskDesignConfig,
        "licks": LickConfig,
        "speaker": SpeakerConfig,
        "engagement": EngagementConfig,
        "camera": CameraConfig,
        "video": VideoExportConfig,
        "integrations": IntegrationConfig,
        "welfare": WelfareConfig,
        "animal_use": AnimalUseConfig,
        "monitor": MonitorConfig,
        "session_export": SessionExportConfig,
    }
    issues: list[ConfigValidationIssue] = []
    allowed_top = {item.name for item in fields(SessionConfig)} | set(aliases)
    for key in sorted(set(raw) - allowed_top, key=str):
        issues.append(ConfigValidationIssue(str(key), "unknown field", raw[key]))

    for alias, canonical in aliases.items():
        if alias in raw and canonical in raw:
            issues.append(
                ConfigValidationIssue(
                    alias,
                    f"cannot be used together with {canonical}",
                    raw[alias],
                )
            )

    for source_name in (*section_types, *aliases):
        if source_name not in raw:
            continue
        value = raw[source_name]
        if not isinstance(value, dict):
            issues.append(
                ConfigValidationIssue(source_name, "must be a YAML mapping", value)
            )
            continue
        canonical = aliases.get(source_name, source_name)
        config_type = section_types[canonical]
        allowed = {item.name for item in fields(config_type)}
        if canonical == "licks":
            allowed.add("digital_channels")  # documented legacy alias
        for key in sorted(set(value) - allowed, key=str):
            secret = canonical == "integrations" and key == "alyx_token"
            issues.append(
                ConfigValidationIssue(
                    f"{source_name}.{key}",
                    "set ALYX_TOKEN in the environment" if secret else "unknown field",
                    "<redacted>" if secret else value[key],
                )
            )

        if canonical == "licks" and {"channels", "digital_channels"} <= set(value):
            issues.append(
                ConfigValidationIssue(
                    f"{source_name}.digital_channels",
                    "cannot be used together with channels",
                    value["digital_channels"],
                )
            )

        defaults = config_type()
        bool_fields = {
            item.name
            for item in fields(config_type)
            if isinstance(getattr(defaults, item.name), bool)
        }
        if canonical == "video":
            bool_fields.add("record_video")
        for key in sorted(bool_fields & set(value)):
            item = value[key]
            if item is not None and type(item) is not bool:
                issues.append(
                    ConfigValidationIssue(
                        f"{source_name}.{key}",
                        "must be a YAML boolean (true/false), not a quoted value",
                        item,
                    )
                )

    raw_camera = raw.get("camera")
    raw_video = raw.get("video")
    camera_record_video = raw_camera.get("record_video") if isinstance(raw_camera, dict) else None
    video_record_video = raw_video.get("record_video") if isinstance(raw_video, dict) else None
    if (
        camera_record_video is not None
        and video_record_video is not None
        and camera_record_video != video_record_video
    ):
        issues.append(
            ConfigValidationIssue(
                "video.record_video",
                "contradicts camera.record_video",
                video_record_video,
            )
        )

    if issues:
        raise ConfigValidationError(issues)


def config_validation_issues(config: SessionConfig) -> list[ConfigValidationIssue]:
    """Return semantic configuration failures before any hardware is opened."""
    issues: list[ConfigValidationIssue] = []

    def require(condition: bool, field_name: str, message: str, value: Any) -> None:
        if not condition:
            issues.append(ConfigValidationIssue(field_name, message, value))

    def finite(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

    subject_id = str(config.subject_id).strip()
    require(
        subject_id == str(config.subject_id)
        and bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", subject_id))
        and subject_id not in {".", ".."},
        "subject_id",
        "must be a non-empty path-safe identifier",
        config.subject_id,
    )
    require(config.n_trials > 0, "n_trials", "must be greater than zero", config.n_trials)
    require(bool(str(config.results_dir).strip()), "results_dir", "must not be empty", config.results_dir)
    require(
        config.random_seed is None
        or (isinstance(config.random_seed, int) and not isinstance(config.random_seed, bool)),
        "random_seed",
        "must be an integer or null",
        config.random_seed,
    )
    require(
        finite(config.centering_threshold_deg)
        and 0 <= config.centering_threshold_deg < config.display.choice_distance_deg,
        "centering_threshold_deg",
        "must be finite, non-negative, and below display.choice_distance_deg",
        config.centering_threshold_deg,
    )

    require(config.ports.baudrate > 0, "ports.baudrate", "must be greater than zero", config.ports.baudrate)
    require(
        config.ports.bpod_serial_channel > 0,
        "ports.bpod_serial_channel",
        "must be one-based and greater than zero",
        config.ports.bpod_serial_channel,
    )
    for field_name in ("bpod_port", "dio_config_port", "dio_stream_port"):
        value = getattr(config.ports, field_name)
        require(bool(str(value).strip()), f"ports.{field_name}", "must not be empty", value)

    wheel = config.wheel
    require(
        finite(wheel.wheel_radius_mm) and wheel.wheel_radius_mm > 0,
        "wheel.wheel_radius_mm",
        "must be finite and greater than zero",
        wheel.wheel_radius_mm,
    )
    require(
        wheel.encoder_ticks_per_rev > 0,
        "wheel.encoder_ticks_per_rev",
        "must be greater than zero",
        wheel.encoder_ticks_per_rev,
    )
    require(
        finite(wheel.gain_deg_per_mm) and wheel.gain_deg_per_mm > 0,
        "wheel.gain_deg_per_mm",
        "must be finite and greater than zero",
        wheel.gain_deg_per_mm,
    )
    for field_name in ("pulse_channel", "direction_channel", "force_channel"):
        value = getattr(wheel, field_name)
        require(value > 0, f"wheel.{field_name}", "must be greater than zero", value)

    display = config.display
    for field_name in ("distance_mm", "pixel_size_mm", "choice_distance_deg"):
        value = getattr(display, field_name)
        require(
            finite(value) and value > 0,
            f"display.{field_name}",
            "must be finite and greater than zero",
            value,
        )
    require(display.fps > 0, "display.fps", "must be greater than zero", display.fps)
    require(
        finite(display.choice_center_threshold_deg)
        and 0 <= display.choice_center_threshold_deg < display.choice_distance_deg,
        "display.choice_center_threshold_deg",
        "must be finite, non-negative, and below choice_distance_deg",
        display.choice_center_threshold_deg,
    )
    require(
        (display.width is None) == (display.height is None),
        "display.width/height",
        "must both be set or both be null",
        [display.width, display.height],
    )
    for field_name in ("width", "height"):
        value = getattr(display, field_name)
        if value is not None:
            require(value > 0, f"display.{field_name}", "must be greater than zero", value)

    frame2ttl = config.frame2ttl
    require(frame2ttl.input_channel > 0, "frame2ttl.input_channel", "must be greater than zero", frame2ttl.input_channel)
    require(frame2ttl.input_pin > 0, "frame2ttl.input_pin", "must be greater than zero", frame2ttl.input_pin)
    require(frame2ttl.patch_size_px > 0, "frame2ttl.patch_size_px", "must be greater than zero", frame2ttl.patch_size_px)
    require(frame2ttl.patch_margin_px >= 0, "frame2ttl.patch_margin_px", "must be non-negative", frame2ttl.patch_margin_px)
    require(
        frame2ttl.patch_corner in {"top_left", "top_right", "bottom_left", "bottom_right"},
        "frame2ttl.patch_corner",
        "must be a supported screen corner",
        frame2ttl.patch_corner,
    )
    if frame2ttl.barcode_enabled:
        for field_name in (
            "barcode_preamble_high_frames",
            "barcode_preamble_low_frames",
            "barcode_one_high_frames",
            "barcode_zero_high_frames",
            "barcode_inter_bit_low_frames",
        ):
            value = getattr(frame2ttl, field_name)
            require(value > 0, f"frame2ttl.{field_name}", "must be greater than zero", value)

    timing = config.timing
    for field_name in (
        "quiescence_s",
        "inter_trial_min_s",
        "choice_blank_s",
        "post_reward_s",
        "timeout_noise_s",
        "post_timeout_s",
    ):
        value = getattr(timing, field_name)
        require(finite(value) and value >= 0, f"timing.{field_name}", "must be finite and non-negative", value)
    for field_name in (
        "choice_timeout_s",
        "choice_loop_stall_s",
        "output_verification_timeout_s",
    ):
        value = getattr(timing, field_name)
        require(finite(value) and value > 0, f"timing.{field_name}", "must be finite and greater than zero", value)
    require(
        timing.output_verification_timeout_s <= 1.0,
        "timing.output_verification_timeout_s",
        "must be at most one second",
        timing.output_verification_timeout_s,
    )

    for field_name in ("matched_pump_s", "opposite_pump_s"):
        value = getattr(config.reward, field_name)
        require(
            finite(value) and 0 <= value <= 1.0,
            f"reward.{field_name}",
            "must be finite and between zero and one second",
            value,
        )
    if config.reward.calibration_path is not None:
        require(
            bool(str(config.reward.calibration_path).strip()),
            "reward.calibration_path",
            "must not be empty when configured",
            config.reward.calibration_path,
        )

    task = config.task
    require(task.modality in {"force", "risk", "delay"}, "task.modality", "must be force, risk, or delay", task.modality)
    require(task.n_forced_trials >= 0, "task.n_forced_trials", "must be non-negative", task.n_forced_trials)
    require(
        task.side_balancing in {"bernoulli", "blocked"},
        "task.side_balancing",
        "must be bernoulli or blocked",
        task.side_balancing,
    )
    require(
        task.max_consecutive_side >= 0,
        "task.max_consecutive_side",
        "must be non-negative",
        task.max_consecutive_side,
    )
    require(
        task.max_consecutive_forced >= 0,
        "task.max_consecutive_forced",
        "must be non-negative",
        task.max_consecutive_forced,
    )
    if task.n_free_trials is not None:
        require(task.n_free_trials >= 0, "task.n_free_trials", "must be non-negative", task.n_free_trials)
        require(
            task.n_free_trials + task.n_forced_trials == config.n_trials,
            "task.n_free_trials/task.n_forced_trials",
            "must sum to n_trials when n_free_trials is explicit",
            [task.n_free_trials, task.n_forced_trials, config.n_trials],
        )
    else:
        require(
            task.n_forced_trials <= config.n_trials,
            "task.n_forced_trials",
            "cannot exceed n_trials when n_free_trials is implicit",
            task.n_forced_trials,
        )
    effective_free = (
        max(0, config.n_trials - task.n_forced_trials)
        if task.n_free_trials is None
        else max(0, task.n_free_trials)
    )
    if task.max_consecutive_forced > 0:
        require(
            task.n_forced_trials
            <= task.max_consecutive_forced * (effective_free + 1),
            "task.max_consecutive_forced",
            "cannot satisfy this cap with the configured free/forced trial counts",
            task.max_consecutive_forced,
        )
    require(
        finite(task.risk_star_probability) and 0 <= task.risk_star_probability <= 1,
        "task.risk_star_probability",
        "must be between zero and one",
        task.risk_star_probability,
    )
    require(
        finite(task.delay_star_extra_s) and task.delay_star_extra_s >= 0,
        "task.delay_star_extra_s",
        "must be finite and non-negative",
        task.delay_star_extra_s,
    )

    lick_channels = tuple(config.licks.channels)
    require(bool(lick_channels), "licks.channels", "must contain at least one input", lick_channels)
    require(len(set(lick_channels)) == len(lick_channels), "licks.channels", "must not contain duplicates", lick_channels)
    for channel in lick_channels:
        require(channel > 0, "licks.channels", "all channels must be greater than zero", channel)
    require(
        config.licks.power_output_channel > 0,
        "licks.power_output_channel",
        "must be greater than zero",
        config.licks.power_output_channel,
    )

    # All digital inputs share one StreamDIO DI bank, so any overlap means two
    # subsystems would consume each other's edges (the analog force channel is
    # numbered in the separate AI bank and cannot collide).
    digital_inputs: list[tuple[str, int]] = [
        ("wheel.pulse_channel", wheel.pulse_channel),
        ("wheel.direction_channel", wheel.direction_channel),
        *((f"licks.channels[{i}]", ch) for i, ch in enumerate(lick_channels)),
    ]
    if frame2ttl.enabled:
        digital_inputs.append(("frame2ttl.input_channel", frame2ttl.input_channel))
    seen_inputs: dict[int, str] = {}
    for field_name, channel in digital_inputs:
        if channel in seen_inputs:
            require(
                False,
                field_name,
                f"conflicts with {seen_inputs[channel]} on StreamDIO digital input {channel}",
                channel,
            )
        else:
            seen_inputs[channel] = field_name
    speaker = config.speaker
    require(speaker.sample_interval_us > 0, "speaker.sample_interval_us", "must be greater than zero", speaker.sample_interval_us)
    require(finite(speaker.frequency_hz) and speaker.frequency_hz > 0, "speaker.frequency_hz", "must be finite and greater than zero", speaker.frequency_hz)
    require(finite(speaker.volume) and 0 <= speaker.volume <= 1, "speaker.volume", "must be between zero and one", speaker.volume)

    require(config.engagement.idle_timeout_s > 0, "engagement.idle_timeout_s", "must be greater than zero", config.engagement.idle_timeout_s)
    require(
        config.engagement.max_consecutive_failures > 0,
        "engagement.max_consecutive_failures",
        "must be greater than zero",
        config.engagement.max_consecutive_failures,
    )

    camera = config.camera
    require(camera.frequency_hz > 0, "camera.frequency_hz", "must be greater than zero", camera.frequency_hz)
    require(camera.exposure_us > 0, "camera.exposure_us", "must be greater than zero", camera.exposure_us)
    if camera.frequency_hz > 0:
        require(
            camera.exposure_us < 1_000_000 / camera.frequency_hz,
            "camera.exposure_us",
            "must be shorter than the camera trigger period",
            camera.exposure_us,
        )
    require(camera.acquisition_timeout_ms > 0, "camera.acquisition_timeout_ms", "must be greater than zero", camera.acquisition_timeout_ms)
    require(bool(str(camera.backend).strip()), "camera.backend", "must not be empty", camera.backend)
    require(bool(str(camera.save_dir).strip()), "camera.save_dir", "must not be empty", camera.save_dir)
    if camera.required:
        require(
            camera.record_video,
            "camera.record_video",
            "must be true when camera.required is true",
            camera.record_video,
        )
        require(
            camera.backend == "spinnaker",
            "camera.backend",
            "must be spinnaker when camera.required is true",
            camera.backend,
        )
    require(
        (camera.roi_width is None) == (camera.roi_height is None),
        "camera.roi_width/roi_height",
        "must both be set or both be null",
        [camera.roi_width, camera.roi_height],
    )
    require(
        (camera.roi_offset_x is None) == (camera.roi_offset_y is None),
        "camera.roi_offset_x/roi_offset_y",
        "must both be set or both be null",
        [camera.roi_offset_x, camera.roi_offset_y],
    )
    for field_name in ("roi_width", "roi_height"):
        value = getattr(camera, field_name)
        if value is not None:
            require(value > 0, f"camera.{field_name}", "must be greater than zero", value)
    for field_name in ("roi_offset_x", "roi_offset_y"):
        value = getattr(camera, field_name)
        if value is not None:
            require(value >= 0, f"camera.{field_name}", "must be non-negative", value)

    monitor = config.monitor
    require(monitor.fps > 0, "monitor.fps", "must be greater than zero", monitor.fps)
    require(1 <= monitor.trial_history <= 10, "monitor.trial_history", "must be between 1 and 10", monitor.trial_history)
    require(monitor.queue_size >= 16, "monitor.queue_size", "must be at least 16", monitor.queue_size)
    for field_name in ("force_sample_hz", "wheel_sample_hz", "screen_fps", "camera_fps"):
        value = getattr(monitor, field_name)
        require(finite(value) and value > 0, f"monitor.{field_name}", "must be finite and greater than zero", value)

    export = config.session_export
    require(export.layout in {"flat", "mm_hierarchy"}, "session_export.layout", "must be flat or mm_hierarchy", export.layout)
    require(export.streamdio_chunk_s >= 0, "session_export.streamdio_chunk_s", "must be non-negative", export.streamdio_chunk_s)
    require(
        export.streamdio_wal_fsync_s > 0,
        "session_export.streamdio_wal_fsync_s",
        "must be greater than zero",
        export.streamdio_wal_fsync_s,
    )
    require(
        export.streamdio_memory_mode in {"full", "bounded", "rolling", "chunked"},
        "session_export.streamdio_memory_mode",
        "must be full, bounded, rolling, or chunked",
        export.streamdio_memory_mode,
    )
    if export.streamdio_memory_mode in {"bounded", "rolling", "chunked"}:
        require(
            export.streamdio_chunk_s > 0,
            "session_export.streamdio_chunk_s",
            "must be greater than zero for bounded or chunked memory modes",
            export.streamdio_chunk_s,
        )
    require(export.streamdio_recent_events > 0, "session_export.streamdio_recent_events", "must be greater than zero", export.streamdio_recent_events)
    require(config.integrations.alyx_timeout_s > 0, "integrations.alyx_timeout_s", "must be greater than zero", config.integrations.alyx_timeout_s)
    require(
        bool(str(config.welfare.subjects_dir).strip()),
        "welfare.subjects_dir",
        "must not be empty",
        config.welfare.subjects_dir,
    )
    if config.welfare.water_target_ml is not None:
        require(
            finite(config.welfare.water_target_ml)
            and config.welfare.water_target_ml >= 0,
            "welfare.water_target_ml",
            "must be finite and non-negative",
            config.welfare.water_target_ml,
        )
    if config.welfare.max_cumulative_reward_ul is not None:
        require(
            finite(config.welfare.max_cumulative_reward_ul)
            and config.welfare.max_cumulative_reward_ul > 0,
            "welfare.max_cumulative_reward_ul",
            "must be finite and greater than zero",
            config.welfare.max_cumulative_reward_ul,
        )
    require(
        config.welfare.max_weight_age_days > 0,
        "welfare.max_weight_age_days",
        "must be greater than zero",
        config.welfare.max_weight_age_days,
    )

    animal = config.animal_use
    require(
        animal.max_certification_age_days > 0,
        "animal_use.max_certification_age_days",
        "must be greater than zero",
        animal.max_certification_age_days,
    )
    require(
        animal.max_calibration_age_days > 0,
        "animal_use.max_calibration_age_days",
        "must be greater than zero",
        animal.max_calibration_age_days,
    )
    require(
        finite(animal.max_calibration_residual_ul)
        and animal.max_calibration_residual_ul > 0,
        "animal_use.max_calibration_residual_ul",
        "must be finite and greater than zero",
        animal.max_calibration_residual_ul,
    )
    if animal.certification_path is not None:
        require(
            bool(str(animal.certification_path).strip()),
            "animal_use.certification_path",
            "must not be empty when configured",
            animal.certification_path,
        )
    if animal.hardware_failsafe_revision is not None:
        require(
            bool(str(animal.hardware_failsafe_revision).strip()),
            "animal_use.hardware_failsafe_revision",
            "must not be empty when configured",
            animal.hardware_failsafe_revision,
        )
    return issues


def validate_session_config(config: SessionConfig) -> SessionConfig:
    issues = config_validation_issues(config)
    if issues:
        raise ConfigValidationError(issues)
    return config


def _normalize_cli_overrides(value: Any) -> list[dict[str, Any]]:
    if value in (None, ""):
        return []
    if isinstance(value, dict):
        return [dict(value)]
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, dict)]
    return [{"value": str(value)}]


def config_to_dict(config: SessionConfig) -> dict[str, Any]:
    def convert(value: Any) -> Any:
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, tuple):
            return [convert(item) for item in value]
        if isinstance(value, list):
            return [convert(item) for item in value]
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items()}
        return value

    return convert(asdict(config))


def task_design_summary(config: SessionConfig) -> dict[str, Any]:
    task = config.task
    modality = str(task.modality or "force").lower()
    if modality not in {"force", "risk", "delay"}:
        modality = "force"
    n_forced = max(0, int(task.n_forced_trials))
    if task.n_free_trials is None:
        n_free = max(0, int(config.n_trials) - n_forced)
    else:
        n_free = max(0, int(task.n_free_trials))
    total = n_free + n_forced
    star_summary: dict[str, Any] = {
        "shape": "star",
        "effect": modality,
        "reward_duration_s": float(config.reward.matched_pump_s),
        "relative_reward": "long",
        "physical_force_output": modality == "force",
    }
    if modality == "risk":
        reward_probability = max(0.0, min(1.0, float(task.risk_star_probability)))
        star_summary["reward_probability"] = reward_probability
        star_summary["omission_probability"] = 1.0 - reward_probability
    if modality == "delay":
        star_summary["extra_delay_s"] = max(0.0, float(task.delay_star_extra_s))
    return {
        "modality": modality,
        "profile_name": task.profile_name,
        "n_trials": total,
        "n_free_trials": n_free,
        "n_forced_trials": n_forced,
        "side_balancing": task.side_balancing,
        "max_consecutive_side": task.max_consecutive_side,
        "max_consecutive_forced": task.max_consecutive_forced,
        "choice_modes": {
            "free": "star and circle both visible; either centered shape ends the trial",
            "forced": "only the forced shape is visible; only that side can end the trial before timeout",
        },
        "symbols": {
            "star": star_summary,
            "circle": {
                "shape": "circle",
                "effect": "baseline",
                "reward_duration_s": float(config.reward.opposite_pump_s),
                "reward_probability": 1.0,
                "extra_delay_s": 0.0,
                "relative_reward": "short",
                "physical_force_output": False,
            },
        },
    }
