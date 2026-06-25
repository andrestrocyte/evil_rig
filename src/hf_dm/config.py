"""Session and task configuration (YAML-loadable)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from hf_dm.hardware.bringup_constants import (
    BringupPorts,
    CAP_TOUCH_POWER_OUTPUT_CHANNEL,
    DEFAULT_BAUDRATE,
    FRAME2TTL_INPUT_CHANNEL,
    FRAME2TTL_INPUT_PIN,
    LICK_INPUT_CHANNEL,
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


@dataclass
class RewardConfig:
    matched_pump_s: float = 0.08
    opposite_pump_s: float = 0.04


@dataclass
class TaskDesignConfig:
    modality: str = "force"
    profile_name: str = "config"
    n_free_trials: int | None = None
    n_forced_trials: int = 0
    risk_star_probability: float = 0.66
    delay_star_extra_s: float = 2.0
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
    trial_history: int = 12
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
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    results_dir: str = "data/sessions"

    @staticmethod
    def from_yaml(path: Path | str) -> SessionConfig:
        path = Path(path)
        raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
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
        monitor_raw = raw.get("monitor", {})
        wheel_raw = raw.get("wheel", {})
        frame2ttl_raw = raw.get("frame2ttl", {})

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
            return tuple(max(0, min(255, int(part))) for part in value)

        def optional_lick_int(name: str) -> int | None:
            value = lick_raw.get(name)
            if value in (None, ""):
                return None
            return int(value)

        return SessionConfig(
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
            ),
            reward=RewardConfig(
                matched_pump_s=float(reward_raw.get("matched_pump_s", 0.08)),
                opposite_pump_s=float(reward_raw.get("opposite_pump_s", 0.04)),
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
                record_video=bool(camera_raw.get("record_video", True)),
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
                trial_history=int(monitor_raw.get("trial_history", 12)),
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
            ),
            results_dir=str(raw.get("results_dir", "data/sessions")),
        )


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
