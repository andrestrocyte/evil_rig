"""Hardware-free simulation for exercising the live task loop."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal

from hf_dm.config import SessionConfig
from hf_dm.data.recorder import SessionRecorder
from hf_dm.display.pygame_display import TaskDisplay
from hf_dm.fsm.relay import RelayMessageIds
from hf_dm.fsm.runtime import TrialRuntime
from hf_dm.hardware.bringup_constants import (
    CAP_TOUCH_POWER_OUTPUT_CHANNEL,
    CAP_TOUCH_POWER_PIN,
    LICK_INPUT_CHANNEL,
)
from hf_dm.session.controller import HFDecisionSession
from hf_dm.session.simulated_bpod import SimulatedBpod
from hf_dm.streamdio.wheel import Direction, WheelKinematics

SimChoiceMode = Literal["matched", "opposite", "timeout", "cw", "ccw", "alternate"]


@dataclass
class SimulationOptions:
    choice_mode: SimChoiceMode = "alternate"
    frames_to_choice: int = 18
    emit_licks: bool = True
    emit_force: bool = True


class SimulatedWheel:
    def __init__(self, kinematics: WheelKinematics, *, frames_to_choice: int, emit_force: bool) -> None:
        self.cfg = kinematics
        self._frames_to_choice = max(1, int(frames_to_choice))
        self._emit_force = emit_force
        self._planned_choice: Literal["cw", "ccw", "timeout"] = "timeout"
        self._choice_distance_deg = 35.0
        self.reset_trial()

    def plan_choice(
        self,
        choice: Literal["cw", "ccw", "timeout"],
        choice_distance_deg: float,
    ) -> None:
        self._planned_choice = choice
        self._choice_distance_deg = float(choice_distance_deg)
        self.reset_trial()

    def reset_trial(self) -> None:
        self.net_pulses = 0
        self.cw_pulses = 0
        self.ccw_pulses = 0
        self.force_samples = 0
        self.force_sum = 0.0
        self.force_last: float | None = None
        self.force_min: float | None = None
        self.force_max: float | None = None
        self.last_pulse_monotonic = time.perf_counter()

    def handle_event(
        self,
        event_type: str,
        channel: int,
        value: int,
        timestamp_us: int,
        relative_time: float,
    ) -> None:
        del event_type, channel, value, timestamp_us, relative_time

    def snapshot(self) -> dict:
        if self._planned_choice != "timeout":
            target_sign = 1 if self._planned_choice == "cw" else -1
            target_pulses = max(1, round(self._choice_distance_deg / self.cfg.deg_per_pulse))
            step = max(1, round(target_pulses / self._frames_to_choice))
            next_abs = min(target_pulses, abs(self.net_pulses) + step)
            self.net_pulses = target_sign * next_abs
            self.cw_pulses = max(0, self.net_pulses)
            self.ccw_pulses = max(0, -self.net_pulses)
            self.last_pulse_monotonic = time.perf_counter()

        if self._emit_force:
            self.force_samples += 1
            self.force_last = 1000.0 + abs(self.net_pulses) * 2.0
            self.force_sum += self.force_last
            self.force_min = self.force_last if self.force_min is None else min(self.force_min, self.force_last)
            self.force_max = self.force_last if self.force_max is None else max(self.force_max, self.force_last)

        force_mean = self.force_sum / self.force_samples if self.force_samples else None
        return {
            "net_pulses": self.net_pulses,
            "cw_pulses": self.cw_pulses,
            "ccw_pulses": self.ccw_pulses,
            "force_samples": self.force_samples,
            "force_last": self.force_last,
            "force_mean": force_mean,
            "force_min": self.force_min,
            "force_max": self.force_max,
            "quiescent_for_s": time.perf_counter() - self.last_pulse_monotonic,
        }

    def signed_position_deg(self) -> float:
        return self.net_pulses * self.cfg.deg_per_pulse


class SimulatedRelay:
    def __init__(self) -> None:
        self.ops: list[tuple] = []

    @staticmethod
    def _transition(channel: int, state: bool) -> dict:
        now = time.perf_counter()
        return {
            "channel": int(channel),
            "state": int(bool(state)),
            "acknowledged": True,
            "command_start_mono_s": now,
            "command_end_mono_s": now,
            "event_verified": True,
            "event_host_perf_counter_s": now,
        }

    def set_force_direction(self, direction: str | None) -> list[dict]:
        self.ops.append(("force", direction, time.perf_counter()))
        if direction == "cw":
            return [self._transition(5, False), self._transition(3, True)]
        if direction == "ccw":
            return [self._transition(3, False), self._transition(5, True)]
        return self.force_off()

    def force_off(self) -> list[dict]:
        self.ops.append(("force_off", time.perf_counter()))
        return [self._transition(3, False), self._transition(5, False)]

    def pump_on(self) -> dict:
        self.ops.append(("pump_on", time.perf_counter()))
        return self._transition(1, True)

    def pump_off(self) -> dict:
        self.ops.append(("pump_off", time.perf_counter()))
        return self._transition(1, False)

    def pulse_pump(self, duration_s: float) -> dict:
        on = self.pump_on()
        time.sleep(max(0.0, float(duration_s)))
        off = self.pump_off()
        return {"on": on, "off": off}

    def speaker_gate(self, enabled: bool) -> dict:
        self.ops.append(("speaker_gate", bool(enabled), time.perf_counter()))
        now = time.perf_counter()
        return {
            "enabled": bool(enabled),
            "acknowledged": True,
            "command_start_mono_s": now,
            "command_end_mono_s": now,
            "status": {
                "enabled": bool(enabled),
                "pin": 27,
                "sample_interval_us": 100,
                "samples_generated": 1 if enabled else 0,
                "current_state": 0,
            },
        }

    def pulse_speaker_gate(self, duration_s: float) -> dict:
        on = self.speaker_gate(True)
        time.sleep(max(0.0, float(duration_s)))
        off = self.speaker_gate(False)
        return {"on": on, "active_status": on["status"], "off": off}

    def cap_touch_power(self, enabled: bool) -> dict:
        self.ops.append(("cap_touch_power", bool(enabled), time.perf_counter()))
        return self._transition(13, enabled)

    def all_off(self) -> dict:
        result = self.task_outputs_off()
        result["cap_touch_power"] = self.cap_touch_power(False)
        return result

    def task_outputs_off(self) -> dict:
        return {
            "pump": self.pump_off(),
            "force": self.force_off(),
            "speaker": self.speaker_gate(False),
        }


class SimulatedSpeakerRelay:
    module_name = "SimSerial"

    def __init__(self) -> None:
        self.ids = RelayMessageIds()
        self.ops: list[tuple[str, bool, float]] = []

    def act(self, message_id: int) -> dict[str, int]:
        return {self.module_name: int(message_id)}

    def speaker_gate(self, enabled: bool) -> None:
        self.ops.append(("speaker_gate", bool(enabled), time.perf_counter()))


class SimulatedDIO:
    config_port = "SIM_CONFIG"
    stream_port = "SIM_STREAM"

    def __init__(self) -> None:
        self._camera_active = False
        self._camera_frequency_hz = 30
        self._camera_start_mono = time.perf_counter()
        self._camera_events: list[tuple[int, float]] = []
        self._callback = None

    def start_streaming(self, callback=None) -> None:
        self._callback = callback

    def start_camera_session(self, frequency_hz: int, exposure_us: int) -> None:
        del exposure_us
        self._camera_active = True
        self._camera_frequency_hz = max(1, int(frequency_hz))
        self._camera_start_mono = time.perf_counter()
        self._camera_events.clear()

    def _update_camera_events(self) -> None:
        if not self._camera_active:
            return
        elapsed = time.perf_counter() - self._camera_start_mono
        expected = int(elapsed * self._camera_frequency_hz)
        while len(self._camera_events) < expected:
            idx = len(self._camera_events) + 1
            relative_s = idx / float(self._camera_frequency_hz)
            timestamp_us = int(relative_s * 1_000_000)
            self._camera_events.append((timestamp_us, relative_s))
            if self._callback is not None:
                self._callback("camera", 0, 0, timestamp_us, relative_s)

    def get_camera_events(self) -> list[tuple[int, float]]:
        self._update_camera_events()
        return list(self._camera_events)

    def stop_camera(self) -> None:
        self._update_camera_events()
        self._camera_active = False

    def close(self) -> None:
        self.stop_camera()


class SimulatedVideoRecorder:
    def __init__(
        self,
        save_dir: Path,
        subject_id: str,
        phase: str,
        *,
        preview_callback: Callable[[dict[str, Any]], None] | None = None,
        preview_width: int = 320,
        preview_height: int = 240,
        fps: int = 10,
    ) -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.session_dir = Path(save_dir) / f"{subject_id}_{phase}_{stamp}_spinnaker_sim"
        self._start_wall_s = time.time()
        self._preview_callback = preview_callback
        self._preview_width = max(64, int(preview_width))
        self._preview_height = max(48, int(preview_height))
        self._fps = max(1, int(fps))
        self._running = False
        self._thread: threading.Thread | None = None
        self._frames = 0

    def start(self) -> Path:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        if self._preview_callback is not None:
            self._running = True
            self._thread = threading.Thread(
                target=self._preview_loop,
                name="spinnaker-sim-preview",
                daemon=True,
            )
            self._thread.start()
        return self.session_dir

    def stop(self) -> dict:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        return {
            "backend": "spinnaker_sim",
            "session_dir": str(self.session_dir),
            "frames_saved": 0,
            "preview_frames": self._frames,
            "start_wall_s": self._start_wall_s,
            "stop_wall_s": time.time(),
            "capture_error": None,
            "cleanup_error": None,
        }

    def _preview_loop(self) -> None:
        period_s = 1.0 / float(self._fps)
        while self._running:
            frame_index = self._frames
            data = bytearray(self._preview_width * self._preview_height)
            for y in range(self._preview_height):
                base = y * self._preview_width
                band = int(55 + 45 * math.sin((frame_index * 0.11) + y * 0.045))
                for x in range(self._preview_width):
                    data[base + x] = max(
                        0,
                        min(255, band + ((x + frame_index * 3) % 90)),
                    )
            now = time.perf_counter()
            self._preview_callback(
                {
                    "frame_index": frame_index,
                    "camera_timestamp_ns": int(now * 1_000_000_000),
                    "width": self._preview_width,
                    "height": self._preview_height,
                    "channels": 1,
                    "format": "L",
                    "data": bytes(data),
                    "monotonic_s": now,
                    "backend": "spinnaker_sim",
                }
            )
            self._frames += 1
            time.sleep(period_s)


class SimulatedHFDecisionSession(HFDecisionSession):
    def __init__(self, config: SessionConfig, options: SimulationOptions):
        super().__init__(config)
        self.options = options
        self.wheel = SimulatedWheel(
            config.wheel,
            frames_to_choice=options.frames_to_choice,
            emit_force=options.emit_force,
        )

    def connect(self) -> None:
        self._closed = False
        self.bpod = SimulatedBpod()
        self.bpod.set_softcode_handler(self._handle_bpod_softcode)
        self.relay = SimulatedRelay()
        self.speaker_relay = SimulatedSpeakerRelay()
        self.dio = SimulatedDIO()
        self.display = TaskDisplay(self.config.display)
        self.recorder = SessionRecorder(
            Path(self.config.results_dir),
            self.config.subject_id,
            self.config.phase.value,
        )
        self._start_operator_monitor()
        if self.config.camera.record_video:
            video = SimulatedVideoRecorder(
                Path(self.config.camera.save_dir),
                self.config.subject_id,
                self.config.phase.value,
                preview_callback=(
                    self._handle_camera_preview if self.monitor is not None else None
                ),
                preview_width=self.config.monitor.camera_preview_width,
                preview_height=self.config.monitor.camera_preview_height,
                fps=int(max(1.0, float(self.config.monitor.camera_fps))),
            )
            session_dir = video.start()
            self.camera_recorder = video
            self.recorder.log_event(
                "video_recording_started",
                {"backend": "spinnaker_sim", "save_dir": str(session_dir)},
            )
        self.dio.start_streaming(callback=self._handle_streamdio_event)
        self.recorder.log_event("streamdio_streaming_started", {"simulated": True})
        self.recorder.log_event("all_off_startup_verified", self.relay.all_off())
        self.recorder.log_event(
            "white_noise_ready",
            {
                "source": "simulated_bpod_pin27_gate",
                "started": True,
                "required": self.config.speaker.required,
                "simulated": True,
            },
        )
        cap_touch_transition = self.relay.cap_touch_power(True)
        self._publish_monitor(
            "cap_touch_power",
            {
                "active": True,
                "state": 1,
                "channel": self.config.licks.power_output_channel,
                "pin": (
                    CAP_TOUCH_POWER_PIN
                    if self.config.licks.power_output_channel == CAP_TOUCH_POWER_OUTPUT_CHANNEL
                    else None
                ),
                "source": "connect_command",
                "transition": cap_touch_transition,
            },
        )
        self.recorder.log_event(
            "cap_touch_power_on",
            {
                "output_channel": self.config.licks.power_output_channel,
                "input_channels": list(self.config.licks.channels),
                "transition": cap_touch_transition,
                "simulated": True,
            },
        )
        self.dio.start_camera_session(
            frequency_hz=self.config.camera.frequency_hz,
            exposure_us=self.config.camera.exposure_us,
        )
        self.recorder.log_event(
            "streamdio_camera_triggers_started",
            {
                "frequency_hz": self.config.camera.frequency_hz,
                "exposure_us": self.config.camera.exposure_us,
                "simulated": True,
            },
        )
        self.recorder.log_event(
            "hardware_connected",
            {
                "bpod_port": "SIM_BPOD",
                "dio_config": self.dio.config_port,
                "dio_stream": self.dio.stream_port,
                "camera_hz": self.config.camera.frequency_hz,
                "camera_backend": "spinnaker_sim",
                "lick_channels": list(self.config.licks.channels),
                "simulated": True,
            },
        )

    def _choice_for_trial(
        self,
        force_direction: Direction,
        trial_index: int,
    ) -> Literal["cw", "ccw", "timeout"]:
        mode = self.options.choice_mode
        if mode == "matched":
            return force_direction
        if mode == "opposite":
            return "ccw" if force_direction == "cw" else "cw"
        if mode in ("cw", "ccw", "timeout"):
            return mode
        cycle = ("matched", "opposite", "timeout")[(trial_index - 1) % 3]
        if cycle == "matched":
            return force_direction
        if cycle == "opposite":
            return "ccw" if force_direction == "cw" else "cw"
        return "timeout"

    def _prepare_decision_trial(self, runtime: TrialRuntime) -> None:
        runtime.simulated_choice = self._choice_for_trial(  # type: ignore[attr-defined]
            runtime.force_direction,
            runtime.trial_index,
        )

    def _on_choice_window_started(self, runtime: TrialRuntime) -> None:
        planned = runtime.simulated_choice  # type: ignore[attr-defined]
        self.wheel.plan_choice(planned, self.config.display.choice_distance_deg)
        if self.options.emit_licks:
            self._handle_streamdio_event(
                "dio_input",
                LICK_INPUT_CHANNEL,
                1,
                int((time.perf_counter() % 10_000) * 1_000_000),
                time.perf_counter(),
            )

    def _prepare_centering_trial(self, start_position_deg: float) -> None:
        movement = "cw" if start_position_deg < 0.0 else "ccw"
        self.wheel.plan_choice(movement, abs(start_position_deg))


def shorten_for_simulation(config: SessionConfig, *, n_trials: int | None = None) -> None:
    if n_trials is not None:
        config.n_trials = int(n_trials)
    config.timing.quiescence_s = 0.0
    config.timing.inter_trial_min_s = min(config.timing.inter_trial_min_s, 0.05)
    config.timing.choice_timeout_s = min(config.timing.choice_timeout_s, 0.5)
    config.timing.choice_blank_s = min(config.timing.choice_blank_s, 0.05)
    config.timing.post_reward_s = min(config.timing.post_reward_s, 0.05)
    config.timing.timeout_noise_s = min(config.timing.timeout_noise_s, 0.05)
    config.timing.post_timeout_s = min(config.timing.post_timeout_s, 0.05)
    config.display.pre_trial_black_s = min(config.display.pre_trial_black_s, 0.05)
    config.display.feedback_hold_s = min(config.display.feedback_hold_s, 0.05)
