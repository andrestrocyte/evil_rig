"""Session and task configuration (YAML-loadable)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from hf_dm.hardware.bringup_constants import BringupPorts, DEFAULT_BAUDRATE
from hf_dm.streamdio.wheel import WheelKinematics


class TrainingPhase(str, Enum):
    CENTERING = "centering"
    DECISION_MAKING = "decision_making"


@dataclass
class DisplayConfig:
    choice_distance_deg: float = 35.0
    rotated_choice_axis: bool = True
    display_index: int | None = 1
    fps: int = 60
    pre_trial_black_s: float = 2.0
    feedback_hold_s: float = 0.6
    matched_rgb: tuple[int, int, int] = (72, 198, 135)
    unmatched_rgb: tuple[int, int, int] = (88, 180, 255)
    timeout_rgb: tuple[int, int, int] = (235, 94, 94)


@dataclass
class TimingConfig:
    quiescence_s: float = 2.0
    inter_trial_min_s: float = 3.0
    choice_timeout_s: float = 15.0
    timeout_noise_s: float = 2.0


@dataclass
class RewardConfig:
    matched_pump_s: float = 0.08
    opposite_pump_s: float = 0.04


@dataclass
class EngagementConfig:
    idle_timeout_s: float = 60.0
    max_consecutive_failures: int = 10


@dataclass
class CameraConfig:
    frequency_hz: int = 30
    exposure_us: int = 10_000
    record_video: bool = True
    video_notes: str = "Align external recorder to StreamDIO camera TTL; timestamps saved in session npz."
    backend: str = "external"
    serial_number: str | None = None
    gain_db: float | None = None
    gamma: float | None = None
    pixel_format: str = "Mono8"
    acquisition_timeout_ms: int = 1000
    save_dir: str = "data/videos"


@dataclass
class SessionConfig:
    subject_id: str = "mouse01"
    phase: TrainingPhase = TrainingPhase.DECISION_MAKING
    n_trials: int = 50
    random_seed: int | None = None
    centering_threshold_deg: float = 8.0
  # centering phase: reward when |position| below this
    ports: BringupPorts = field(default_factory=BringupPorts)
    wheel: WheelKinematics = field(default_factory=WheelKinematics)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    engagement: EngagementConfig = field(default_factory=EngagementConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    results_dir: str = "data/sessions"
    white_noise_continuous: bool = True

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
        display_raw = raw.get("display", {})
        engagement_raw = raw.get("engagement", {})
        camera_raw = raw.get("camera", {})
        wheel_raw = raw.get("wheel", {})
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
                gain_deg_per_mm=float(wheel_raw.get("gain_deg_per_mm", 8.0)),
            ),
            display=DisplayConfig(
                choice_distance_deg=float(display_raw.get("choice_distance_deg", 35.0)),
                pre_trial_black_s=float(display_raw.get("pre_trial_black_s", 2.0)),
                feedback_hold_s=float(display_raw.get("feedback_hold_s", 0.6)),
            ),
            timing=TimingConfig(
                quiescence_s=float(timing_raw.get("quiescence_s", 2.0)),
                inter_trial_min_s=float(timing_raw.get("inter_trial_min_s", 3.0)),
                choice_timeout_s=float(timing_raw.get("choice_timeout_s", 15.0)),
                timeout_noise_s=float(timing_raw.get("timeout_noise_s", 2.0)),
            ),
            reward=RewardConfig(
                matched_pump_s=float(reward_raw.get("matched_pump_s", 0.08)),
                opposite_pump_s=float(reward_raw.get("opposite_pump_s", 0.04)),
            ),
            engagement=EngagementConfig(
                idle_timeout_s=float(engagement_raw.get("idle_timeout_s", 60.0)),
                max_consecutive_failures=int(engagement_raw.get("max_consecutive_failures", 10)),
            ),
            camera=CameraConfig(
                frequency_hz=int(camera_raw.get("frequency_hz", 30)),
                exposure_us=int(camera_raw.get("exposure_us", 10_000)),
                record_video=bool(camera_raw.get("record_video", True)),
                video_notes=str(
                    camera_raw.get(
                        "video_notes",
                        "Align external recorder to StreamDIO camera TTL; timestamps saved in session npz.",
                    )
                ),
                backend=str(camera_raw.get("backend", "external")),
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
            results_dir=str(raw.get("results_dir", "data/sessions")),
            white_noise_continuous=bool(raw.get("white_noise_continuous", True)),
        )
