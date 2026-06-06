"""Session orchestration: bpod-core session + verified StreamDIO I/O + display."""

from __future__ import annotations

import logging
import random
import threading
import time
from pathlib import Path
from typing import Literal

from bpod_core.bpod import Bpod

from hf_dm.audio.white_noise import WhiteNoisePlayer
from hf_dm.config import SessionConfig, TrainingPhase
from hf_dm.camera.spinnaker import HeadlessSpinnakerRecorder
from hf_dm.hardware.discovery import connect_bpod_core
from hf_dm.data.recorder import SessionRecorder, SessionSummary, TrialRecord
from hf_dm.display.pygame_display import TaskDisplay
from hf_dm.fsm import softcodes as sc
from hf_dm.fsm.builders import build_host_supervised_decision_fsm
from hf_dm.fsm.relay import SerialRelay, wait_until_perf_counter
from hf_dm.fsm.runtime import Side, Shape, TrialRuntime
from hf_dm.hardware.bringup_constants import (
    CAP_TOUCH_INPUT_CHANNEL,
    CAP_TOUCH_INPUT_PIN,
    CAP_TOUCH_POWER_OUTPUT_CHANNEL,
    CAP_TOUCH_POWER_PIN,
    CCW_FORCE_OUTPUT_CHANNEL,
    CW_FORCE_OUTPUT_CHANNEL,
    FORCE_ANALOG_CHANNEL,
    PUMP_OUTPUT_CHANNEL,
    STREAMDIO_CAP_TOUCH_CONFIG_FIRMWARE_VERSION,
)
from hf_dm.monitor import OperatorMonitor
from hf_dm.streamdio.client import StreamDIOClient
from hf_dm.streamdio.outputs import StreamDIOOutputs
from hf_dm.streamdio.wheel import Direction, WheelAccumulator

logger = logging.getLogger(__name__)


class HFDecisionSession:
    def __init__(self, config: SessionConfig):
        self.config = config
        self.rng = random.Random(config.random_seed)
        self.bpod: Bpod | None = None
        self.relay: StreamDIOOutputs | None = None
        self.speaker_relay: SerialRelay | None = None
        self.white_noise_player: WhiteNoisePlayer | None = None
        self.dio: StreamDIOClient | None = None
        self.wheel = WheelAccumulator(config.wheel)
        self.display: TaskDisplay | None = None
        self.recorder: SessionRecorder | None = None
        self.camera_recorder: HeadlessSpinnakerRecorder | None = None
        self.monitor: OperatorMonitor | None = None
        self._last_wheel_movement_mono = time.perf_counter()
        self._consecutive_failures = 0
        self._lick_lock = threading.Lock()
        self._lick_events: list[dict[str, float | int]] = []
        self._closed = False
        self._last_choice_loop_stats: dict[str, float | int | str | bool] = {}
        self._active_trial_runtime: TrialRuntime | None = None
        self._cap_touch_poll_stop = threading.Event()
        self._cap_touch_poll_thread: threading.Thread | None = None

    def connect(self) -> None:
        self._closed = False
        ports = self.config.ports
        self.bpod = connect_bpod_core(None if ports.bpod_port == "AUTO" else ports.bpod_port)
        self.bpod.reset_session_clock()
        self.bpod.set_softcode_handler(self._handle_bpod_softcode)

        self.dio = StreamDIOClient(
            config_port=ports.dio_config_port,
            stream_port=ports.dio_stream_port,
            baudrate=ports.baudrate,
        )
        excluded = self._bpod_serial_ports()
        self.dio.open(excluded_bpod_ports=excluded)
        self.dio.configure_rotary_and_force(lick_channels=self.config.licks.channels)
        self.display = TaskDisplay(self.config.display)
        out = Path(self.config.results_dir)
        self.recorder = SessionRecorder(out, self.config.subject_id, self.config.phase.value)
        self._configure_cap_touch_digitizer_from_config()
        self._start_operator_monitor()
        self._start_optional_video_recording()
        self.dio.start_streaming(callback=self._handle_streamdio_event)
        self.recorder.log_event("streamdio_streaming_started", {})
        self.relay = StreamDIOOutputs(
            self.dio,
            cap_touch_power_channel=self.config.licks.power_output_channel,
            speaker_sample_interval_us=self.config.speaker.sample_interval_us,
        )
        baseline = self._run_all_off()
        self.recorder.log_event("all_off_startup_verified", baseline)
        self.speaker_relay = SerialRelay(
            self.bpod,
            serial_channel_index=max(0, ports.bpod_serial_channel - 1),
            speaker_gate_active_high=self.config.speaker.gate_active_high,
        )
        self.speaker_relay.speaker_gate(False)
        self.white_noise_player = WhiteNoisePlayer(
            frequency_hz=self.config.speaker.frequency_hz,
            volume=self.config.speaker.volume,
        )
        sound_started = self.white_noise_player.start_continuous()
        self.speaker_relay.speaker_gate(False)
        sound_status = self.white_noise_player.status
        self.recorder.log_event(
            "white_noise_ready",
            {
                "source": "pc_soundcard_continuous_bpod_pin27_gate",
                "started": sound_started,
                "required": self.config.speaker.required,
                "gate_active_high": self.config.speaker.gate_active_high,
                "status": sound_status,
            },
        )
        if self.config.speaker.required and not sound_started:
            raise RuntimeError(
                "Required continuous white noise could not start; refusing to run "
                "without audible timeout punishment."
            )
        cap_touch_power = self.relay.cap_touch_power(True)
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
                "transition": cap_touch_power,
            },
        )
        self.recorder.log_event(
            "cap_touch_power_on",
            {
                "output_channel": self.config.licks.power_output_channel,
                "input_channels": list(self.config.licks.channels),
                "transition": cap_touch_power,
            },
        )
        self._start_cap_touch_status_poll()
        self.dio.start_camera_session(
            frequency_hz=self.config.camera.frequency_hz,
            exposure_us=self.config.camera.exposure_us,
        )
        self.recorder.log_event(
            "streamdio_camera_triggers_started",
            {
                "frequency_hz": self.config.camera.frequency_hz,
                "exposure_us": self.config.camera.exposure_us,
            },
        )
        self.recorder.log_event(
            "hardware_connected",
            {
                "bpod_port": self.bpod.port,
                "bpod_usb_ports": sorted(excluded),
                "bpod_event_allocation": getattr(
                    self.bpod,
                    "hf_dm_event_allocation",
                    None,
                ),
                "dio_config": self.dio.config_port,
                "dio_stream": self.dio.stream_port,
                "camera_hz": self.config.camera.frequency_hz,
                "camera_backend": self.config.camera.backend,
                "lick_channels": list(self.config.licks.channels),
                "cap_touch_power_output_channel": self.config.licks.power_output_channel,
                "output_control": "bpod_fsm_softcode_to_verified_streamdio_usb",
                "speaker_control": "bpod_fsm_serial_pin27_gate_plus_pc_white_noise",
            },
        )

    def close(self, *, stop_monitor: bool = True) -> None:
        if self._closed:
            return
        cleanup_result: dict = {}
        try:
            if self.bpod is not None and self.bpod.is_running:
                try:
                    self.bpod.send_softcode(sc.SESSION_ABORT)
                    self._wait_for_bpod_trial_threads()
                except Exception:
                    logger.exception("Failed to abort active Bpod trial during cleanup")
            self._stop_cap_touch_status_poll()
            if self.relay is not None:
                try:
                    result = self.relay.task_outputs_off()
                    cleanup_result.update(result)
                    if self.recorder is not None:
                        self.recorder.log_event("task_outputs_cleanup_verified", result)
                except Exception as exc:
                    logger.exception("Failed to make task outputs safe")
                    if self.recorder is not None:
                        self.recorder.log_event(
                            "task_outputs_cleanup_error",
                            {"error": f"{type(exc).__name__}: {exc}"},
                        )
            if self.speaker_relay is not None:
                try:
                    self.speaker_relay.speaker_gate(False)
                    cleanup_result["speaker_gate"] = "off"
                    if self.recorder is not None:
                        self.recorder.log_event("speaker_gate_cleanup", {"state": "off"})
                except Exception as exc:
                    logger.exception("Failed to turn off Bpod speaker gate")
                    if self.recorder is not None:
                        self.recorder.log_event(
                            "speaker_gate_cleanup_error",
                            {"error": f"{type(exc).__name__}: {exc}"},
                        )
            if self.white_noise_player is not None:
                self.white_noise_player.stop()
                self.white_noise_player = None
            if self.dio is not None:
                try:
                    self.dio.stop_camera()
                    if self.recorder is not None:
                        self.recorder.log_event("streamdio_camera_triggers_stopped", {})
                except Exception as exc:
                    logger.exception("Failed to stop StreamDIO camera triggers")
                    if self.recorder is not None:
                        self.recorder.log_event(
                            "streamdio_camera_stop_error",
                            {"error": f"{type(exc).__name__}: {exc}"},
                        )
            self._stop_optional_video_recording()
            if self.relay is not None:
                try:
                    cap_touch_off = self.relay.cap_touch_power(False)
                    self._publish_monitor(
                        "cap_touch_power",
                        {
                            "active": False,
                            "state": 0,
                            "channel": self.config.licks.power_output_channel,
                            "pin": (
                                CAP_TOUCH_POWER_PIN
                                if self.config.licks.power_output_channel
                                == CAP_TOUCH_POWER_OUTPUT_CHANNEL
                                else None
                            ),
                            "source": "cleanup_command",
                            "transition": cap_touch_off,
                        },
                    )
                    cleanup_result["cap_touch_power"] = cap_touch_off
                    if self.recorder is not None:
                        self.recorder.log_event("all_off_cleanup_verified", cleanup_result)
                except Exception as exc:
                    logger.exception("Failed to turn off cap-touch power")
                    if self.recorder is not None:
                        self.recorder.log_event(
                            "cap_touch_power_off_error",
                            {"error": f"{type(exc).__name__}: {exc}"},
                        )
            if self.dio is not None:
                try:
                    self.dio.close()
                except Exception as exc:
                    logger.exception("Failed to close StreamDIO")
                    if self.recorder is not None:
                        self.recorder.log_event(
                            "streamdio_close_error",
                            {"error": f"{type(exc).__name__}: {exc}"},
                        )
            if self.bpod is not None:
                try:
                    self.bpod.set_softcode_handler(None)
                    self.bpod.close()
                except Exception as exc:
                    logger.exception("Failed to close Bpod")
                    if self.recorder is not None:
                        self.recorder.log_event(
                            "bpod_close_error",
                            {"error": f"{type(exc).__name__}: {exc}"},
                        )
        finally:
            if stop_monitor:
                self._stop_operator_monitor()
            self._closed = True

    def _bpod_serial_ports(self) -> set[str]:
        if self.bpod is None:
            return set()
        ports = {str(self.bpod.port)}
        for serial_obj in (getattr(self.bpod, "serial1", None), getattr(self.bpod, "serial2", None)):
            port = getattr(serial_obj, "port", None)
            if port:
                ports.add(str(port))
        return ports

    def _start_operator_monitor(self) -> None:
        if not self.config.monitor.enabled:
            return
        self.monitor = OperatorMonitor(self.config.monitor)
        self.monitor.start()
        self._publish_monitor(
            "session_info",
            {
                "subject_id": self.config.subject_id,
                "phase": self.config.phase.value,
                "n_trials": self.config.n_trials,
                "display_index": self.config.display.display_index,
                "camera_hz": self.config.camera.frequency_hz,
                "camera_backend": self.config.camera.backend,
                "force_channel": self.config.wheel.force_channel,
                "lick_channels": list(self.config.licks.channels),
                "lick_high_only": self.config.licks.high_only,
                "lick_inputs": self._lick_input_descriptors(),
                "cap_touch_power": {
                    "channel": self.config.licks.power_output_channel,
                    "pin": (
                        CAP_TOUCH_POWER_PIN
                        if self.config.licks.power_output_channel
                        == CAP_TOUCH_POWER_OUTPUT_CHANNEL
                        else None
                    ),
                },
            },
        )
        self._publish_monitor(
            "fsm_states",
            {
                "controller": "host_supervised_decision",
                "states": self._fsm_monitor_states(),
            },
        )
        if self.display is not None:
            self.display.fill_black()
            self._publish_monitor("task_state", {"phase": "idle"})
            self._publish_task_screen(force=True)
        if self.recorder is not None:
            self.recorder.log_event(
                "operator_monitor_started",
                {
                    "enabled": True,
                    "display_index": self.config.monitor.display_index,
                    "size": [self.config.monitor.width, self.config.monitor.height],
                },
            )

    def _stop_operator_monitor(self, summary: SessionSummary | None = None) -> None:
        monitor = self.monitor
        if monitor is None:
            return
        published_counts = monitor.published_counts
        if summary is not None:
            self._publish_monitor(
                "session_summary",
                {
                    "end_reason": summary.end_reason,
                    "n_trials_completed": summary.n_trials_completed,
                    "n_trials_planned": summary.n_trials_planned,
                },
            )
        dropped = monitor.dropped_messages
        dropped_frames = monitor.dropped_frames
        monitor.stop()
        self.monitor = None
        if self.recorder is not None:
            self.recorder.log_event(
                "operator_monitor_stopped",
                {
                    "dropped_messages": int(dropped),
                    "dropped_frames": int(dropped_frames),
                    "published_counts": published_counts,
                },
        )

    def _configure_cap_touch_digitizer_from_config(self) -> None:
        if self.dio is None:
            return
        licks = self.config.licks
        threshold_requested = (
            licks.cap_touch_min_high_threshold is not None
            or licks.cap_touch_delta_high is not None
            or licks.cap_touch_delta_low is not None
        )
        firmware_version = int(self.dio.raw.module_info.get("firmware_version", 0))
        if (
            not threshold_requested
            and firmware_version < STREAMDIO_CAP_TOUCH_CONFIG_FIRMWARE_VERSION
        ):
            return
        status = self.dio.configure_cap_touch_digitizer(
            min_high_threshold=licks.cap_touch_min_high_threshold,
            delta_high=licks.cap_touch_delta_high,
            delta_low=licks.cap_touch_delta_low,
        )
        if self.recorder is not None:
            self.recorder.log_event(
                "cap_touch_digitizer_configured",
                {
                    "requested": {
                        "min_high_threshold": licks.cap_touch_min_high_threshold,
                        "delta_high": licks.cap_touch_delta_high,
                        "delta_low": licks.cap_touch_delta_low,
                    },
                    "status": status,
                },
            )

    def _start_cap_touch_status_poll(self) -> None:
        if self.monitor is None or self.dio is None:
            return
        if CAP_TOUCH_INPUT_CHANNEL not in {int(channel) for channel in self.config.licks.channels}:
            return
        if self._cap_touch_poll_thread is not None and self._cap_touch_poll_thread.is_alive():
            return

        max_hz = min(20.0, max(1.0, float(self.config.monitor.force_sample_hz)))
        interval_s = 1.0 / max_hz
        self._cap_touch_poll_stop.clear()

        def poll() -> None:
            while not self._cap_touch_poll_stop.wait(interval_s):
                if self.dio is None:
                    return
                try:
                    status = self.dio.get_cap_touch_status()
                except Exception as exc:
                    logger.warning("Stopping cap-touch status polling: %s", exc)
                    self._publish_monitor(
                        "cap_touch_sample",
                        {
                            "channel": CAP_TOUCH_INPUT_CHANNEL,
                            "pin": CAP_TOUCH_INPUT_PIN,
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                    )
                    return
                channel = int(status.get("channel", CAP_TOUCH_INPUT_CHANNEL))
                self._publish_monitor_throttled(
                    "cap_touch_status",
                    max_hz,
                    "cap_touch_sample",
                    {
                        "channel": channel,
                        "pin": int(status.get("pin", CAP_TOUCH_INPUT_PIN)),
                        "value": int(status.get("last_value", 0)),
                        "active": bool(status.get("state", 0)),
                        "polarity": int(status.get("polarity", 0)),
                        "baseline": int(status.get("baseline", 0)),
                        "rising_high_threshold": int(
                            status.get("rising_high_threshold", 0)
                        ),
                        "rising_low_threshold": int(status.get("rising_low_threshold", 0)),
                        "falling_high_threshold": int(
                            status.get("falling_high_threshold", 0)
                        ),
                        "falling_low_threshold": int(status.get("falling_low_threshold", 0)),
                        "source": "firmware_digitizer_status",
                    },
                )

        self._cap_touch_poll_thread = threading.Thread(
            target=poll,
            name="hf-dm-cap-touch-status",
            daemon=True,
        )
        self._cap_touch_poll_thread.start()

    def _stop_cap_touch_status_poll(self) -> None:
        self._cap_touch_poll_stop.set()
        thread = self._cap_touch_poll_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._cap_touch_poll_thread = None

    def _publish_monitor(self, event_type: str, payload: dict | None = None) -> None:
        if self.monitor is not None:
            self.monitor.publish(event_type, payload or {})

    def _publish_monitor_throttled(
        self,
        key: str,
        max_hz: float,
        event_type: str,
        payload: dict | None = None,
    ) -> None:
        if self.monitor is not None:
            self.monitor.publish_throttled(key, max_hz, event_type, payload or {})

    def _lick_input_descriptors(self) -> list[dict]:
        descriptors: list[dict] = []
        for channel in self.config.licks.channels:
            channel_i = int(channel)
            descriptors.append(
                {
                    "channel": channel_i,
                    "pin": CAP_TOUCH_INPUT_PIN if channel_i == CAP_TOUCH_INPUT_CHANNEL else None,
                    "label": "cap touch" if channel_i == CAP_TOUCH_INPUT_CHANNEL else "lick",
                    "digital_events_enabled": True,
                    "analog_capable": channel_i == CAP_TOUCH_INPUT_CHANNEL,
                }
            )
        return descriptors

    def _fsm_monitor_states(self) -> list[dict]:
        timing = self.config.timing
        reward = self.config.reward
        return [
            {
                "name": "Quiescence",
                "duration_s": timing.quiescence_s,
                "logic": "Tup -> ChoiceWindow; wheel motion -> QuiescenceReset",
            },
            {
                "name": "QuiescenceReset",
                "duration_s": 0.001,
                "logic": "Tup -> Quiescence",
            },
            {
                "name": "ChoiceWindow",
                "duration_s": timing.choice_timeout_s,
                "logic": "high choice -> ChoiceBlankHigh; low choice -> ChoiceBlankLow; Tup -> TimeoutNoise",
            },
            {
                "name": "ChoiceBlankHigh",
                "duration_s": timing.choice_blank_s,
                "logic": "Tup -> RewardHigh",
            },
            {
                "name": "ChoiceBlankLow",
                "duration_s": timing.choice_blank_s,
                "logic": "Tup -> RewardLow",
            },
            {
                "name": "RewardHigh",
                "duration_s": reward.matched_pump_s,
                "logic": "pump ON; Tup -> PostRewardHigh",
            },
            {
                "name": "RewardLow",
                "duration_s": reward.opposite_pump_s,
                "logic": "pump ON; Tup -> PostRewardLow",
            },
            {
                "name": "PostRewardHigh",
                "duration_s": timing.post_reward_s,
                "logic": "pump OFF; Tup -> exit",
            },
            {
                "name": "PostRewardLow",
                "duration_s": timing.post_reward_s,
                "logic": "pump OFF; Tup -> exit",
            },
            {
                "name": "TimeoutNoise",
                "duration_s": timing.timeout_noise_s,
                "logic": "force OFF + white noise ON; Tup -> PostTimeout",
            },
            {
                "name": "PostTimeout",
                "duration_s": timing.post_timeout_s,
                "logic": "white noise OFF; Tup -> exit",
            },
            {
                "name": "Abort",
                "duration_s": 0.001,
                "logic": "all task outputs OFF; Tup -> exit",
            },
        ]

    def _publish_fsm_state(
        self,
        state_name: str,
        *,
        runtime: TrialRuntime | None = None,
        phase: str | None = None,
    ) -> None:
        payload: dict = {"active_state": state_name}
        if phase is not None:
            payload["phase"] = phase
        if runtime is not None:
            payload.update(
                {
                    "trial_index": runtime.trial_index,
                    "force_direction": runtime.force_direction,
                    "star_side": runtime.star_side,
                    "circle_side": runtime.circle_side,
                    "chosen_shape": runtime.chosen_shape,
                    "choice_side": runtime.choice_side,
                }
            )
        self._publish_monitor("fsm_state", payload)

    def _publish_task_screen(self, *, force: bool = False) -> None:
        if self.monitor is None or self.display is None:
            return
        monitor_cfg = self.config.monitor
        frame = self.display.preview_frame(
            monitor_cfg.task_preview_width,
            monitor_cfg.task_preview_height,
            rotate_left=monitor_cfg.task_preview_rotate_left,
        )
        if frame is not None:
            if force:
                self.monitor.publish("task_frame", frame)
            else:
                self.monitor.publish_throttled(
                    "task_screen",
                    monitor_cfg.screen_fps,
                    "task_frame",
                    frame,
                )

    def _publish_wheel_snapshot(self, *, phase: str, trial_index: int | None = None) -> None:
        snap = self.wheel.snapshot()
        if snap["force_last"] is not None:
            self._publish_monitor_throttled(
                "force",
                self.config.monitor.force_sample_hz,
                "force_sample",
                {
                    "value": float(snap["force_last"]),
                    "source": "wheel_snapshot",
                },
            )
        self._publish_monitor_throttled(
            "wheel",
            self.config.monitor.wheel_sample_hz,
            "wheel_sample",
            {
                "trial_index": trial_index,
                "phase": phase,
                "position_deg": int(snap["net_pulses"]) * self.config.wheel.deg_per_pulse,
                "net_pulses": int(snap["net_pulses"]),
                "cw_pulses": int(snap["cw_pulses"]),
                "ccw_pulses": int(snap["ccw_pulses"]),
                "force_last": snap["force_last"],
            },
        )

    def _handle_camera_preview(self, frame: dict) -> None:
        self._publish_monitor("camera_frame", frame)

    def _add_trial_record(self, trial: TrialRecord) -> None:
        assert self.recorder is not None
        self.recorder.add_trial(trial)
        self._publish_monitor(
            "trial_result",
            {
                "trial_index": trial.trial_index,
                "phase": trial.phase,
                "choice": trial.choice,
                "choice_side": trial.choice_side,
                "chosen_shape": trial.chosen_shape,
                "matched": trial.matched,
                "responded": trial.responded,
                "reward_duration_s": trial.reward_duration_s,
                "response_time_s": trial.response_time_s,
                "final_position_deg": trial.final_position_deg,
                "star_side": trial.star_side,
                "circle_side": trial.circle_side,
                "lick_count": trial.lick_count,
            },
        )

    def _handle_streamdio_event(
        self,
        event_type: str,
        channel: int | None,
        value: int | None,
        timestamp_us: int,
        relative_time: float,
    ) -> None:
        if self.recorder is not None:
            self.recorder.log_streamdio_event(
                event_type,
                channel,
                value,
                timestamp_us,
                relative_time,
            )
        normalized_channel = -1 if channel is None else int(channel)
        normalized_value = -1 if value is None else int(value)
        if event_type == "dio_output" and self.recorder is not None:
            output_names = {
                PUMP_OUTPUT_CHANNEL: "pump",
                CW_FORCE_OUTPUT_CHANNEL: "cw_force",
                CCW_FORCE_OUTPUT_CHANNEL: "ccw_force",
                self.config.licks.power_output_channel: "cap_touch_power",
            }
            self.recorder.log_event(
                "output_actual",
                {
                    "channel": normalized_channel,
                    "name": output_names.get(normalized_channel, "other"),
                    "state": normalized_value,
                    "module_timestamp_us": int(timestamp_us),
                    "relative_time_s": float(relative_time),
                },
            )
        if event_type == "camera":
            self._publish_monitor(
                "camera_ttl",
                {
                    "timestamp_us": int(timestamp_us),
                    "relative_time_s": float(relative_time),
                },
            )
        self.wheel.handle_event(
            event_type,
            normalized_channel,
            normalized_value,
            timestamp_us,
            relative_time,
        )
        if event_type == "analog_input" and normalized_channel in self.config.licks.channels:
            self._publish_monitor_throttled(
                f"cap_touch_sample_{normalized_channel}",
                min(30.0, max(1.0, float(self.config.monitor.force_sample_hz))),
                "cap_touch_sample",
                {
                    "channel": normalized_channel,
                    "pin": (
                        CAP_TOUCH_INPUT_PIN
                        if normalized_channel == CAP_TOUCH_INPUT_CHANNEL
                        else None
                    ),
                    "value": float(normalized_value),
                    "timestamp_us": int(timestamp_us),
                    "relative_time_s": float(relative_time),
                },
            )
            return
        if event_type == "analog_input" and normalized_channel == FORCE_ANALOG_CHANNEL:
            self._publish_monitor_throttled(
                "force",
                self.config.monitor.force_sample_hz,
                "force_sample",
                {
                    "value": float(normalized_value),
                    "timestamp_us": int(timestamp_us),
                    "relative_time_s": float(relative_time),
                },
            )
            return
        if (
            event_type == "dio_input"
            and normalized_channel
            in (self.config.wheel.pulse_channel, self.config.wheel.direction_channel)
        ):
            self._publish_wheel_snapshot(phase="stream")
        if event_type == "dio_output" and normalized_channel == PUMP_OUTPUT_CHANNEL:
            self._publish_monitor(
                "pump",
                {
                    "active": bool(normalized_value),
                    "state": normalized_value,
                    "timestamp_us": int(timestamp_us),
                    "relative_time_s": float(relative_time),
                },
            )
        if (
            event_type == "dio_output"
            and normalized_channel == self.config.licks.power_output_channel
        ):
            self._publish_monitor(
                "cap_touch_power",
                {
                    "active": bool(normalized_value),
                    "state": normalized_value,
                    "channel": normalized_channel,
                    "pin": (
                        CAP_TOUCH_POWER_PIN
                        if normalized_channel == CAP_TOUCH_POWER_OUTPUT_CHANNEL
                        else None
                    ),
                    "timestamp_us": int(timestamp_us),
                    "relative_time_s": float(relative_time),
                },
            )
        if event_type != "dio_input":
            return
        if normalized_channel not in self.config.licks.channels:
            return
        counted = not self.config.licks.high_only or bool(normalized_value)
        base_event = {
            "channel": normalized_channel,
            "pin": CAP_TOUCH_INPUT_PIN if normalized_channel == CAP_TOUCH_INPUT_CHANNEL else None,
            "value": normalized_value,
            "timestamp_us": int(timestamp_us),
            "relative_time_s": float(relative_time),
            "wall_time_s": time.time(),
            "monotonic_s": time.perf_counter(),
        }
        edge_event = {**base_event, "counted": counted}
        self._publish_monitor("lick_edge", edge_event)
        if self.recorder is not None:
            self.recorder.log_event("lick_edge", edge_event)
        if not counted:
            return
        lick_event = {
            **base_event,
        }
        with self._lick_lock:
            self._lick_events.append(lick_event)
        self._publish_monitor("lick", lick_event)
        if self.recorder is not None:
            self.recorder.log_event("lick", lick_event)

    def _lick_event_count(self) -> int:
        with self._lick_lock:
            return len(self._lick_events)

    def _lick_events_since(self, start_index: int) -> list[dict[str, float | int]]:
        with self._lick_lock:
            return list(self._lick_events[start_index:])

    def _run_all_off(self) -> dict:
        assert self.relay is not None
        return self.relay.all_off()

    def _pulse_audible_white_noise(self, duration_s: float) -> dict:
        assert self.speaker_relay is not None
        started = time.perf_counter()
        self.speaker_relay.speaker_gate(True)
        on_done = time.perf_counter()
        self._publish_monitor(
            "white_noise",
            {"active": True, "duration_s": float(duration_s), "source": "host_pulse"},
        )
        wait_until_perf_counter(on_done + max(0.0, float(duration_s)))
        self.speaker_relay.speaker_gate(False)
        off_done = time.perf_counter()
        self._publish_monitor(
            "white_noise",
            {"active": False, "duration_s": float(duration_s), "source": "host_pulse"},
        )
        return {
            "source": "pc_soundcard_continuous_bpod_pin27_gate",
            "command_start_mono_s": started,
            "gate_on_mono_s": on_done,
            "gate_off_mono_s": off_done,
            "duration_s": off_done - on_done,
        }

    def _start_optional_video_recording(self) -> None:
        if self.recorder is None:
            return
        camera_cfg = self.config.camera
        if not camera_cfg.record_video:
            self.recorder.log_event("video_recording_skipped", {"reason": "record_video_false"})
            return
        if camera_cfg.backend != "spinnaker":
            self.recorder.log_event(
                "video_recording_external",
                {
                    "backend": camera_cfg.backend,
                    "notes": camera_cfg.video_notes,
                },
            )
            return

        try:
            recorder = HeadlessSpinnakerRecorder(
                out_dir=Path(camera_cfg.save_dir),
                subject_id=self.config.subject_id,
                phase=self.config.phase.value,
                serial_number=camera_cfg.serial_number,
                fps=camera_cfg.frequency_hz,
                exposure_us=camera_cfg.exposure_us,
                pixel_format=camera_cfg.pixel_format,
                gain_db=camera_cfg.gain_db,
                gamma=camera_cfg.gamma,
                acquisition_timeout_ms=camera_cfg.acquisition_timeout_ms,
                preview_callback=(
                    self._handle_camera_preview if self.monitor is not None else None
                ),
                preview_max_width=self.config.monitor.camera_preview_width,
                preview_max_height=self.config.monitor.camera_preview_height,
                preview_min_interval_s=(
                    1.0 / max(1.0, float(self.config.monitor.camera_fps))
                ),
            )
            session_dir = recorder.start()
            self.camera_recorder = recorder
            self.recorder.log_event(
                "video_recording_started",
                {
                    "backend": recorder.backend,
                    "save_dir": str(session_dir),
                    "serial_number": camera_cfg.serial_number,
                    "pixel_format": camera_cfg.pixel_format,
                },
            )
        except Exception as exc:
            logger.exception("Failed to start headless Spinnaker recording")
            self.recorder.log_event(
                "video_recording_error",
                {
                    "backend": "spinnaker",
                    "stage": "start",
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            if camera_cfg.required:
                raise RuntimeError(
                    "Required Spinnaker recording could not start; refusing to run "
                    "a session without video."
                ) from exc

    def _stop_optional_video_recording(self) -> None:
        if self.camera_recorder is None:
            return
        assert self.recorder is not None
        try:
            summary = self.camera_recorder.stop()
            self.recorder.log_event("video_recording_stopped", summary)
        except Exception as exc:
            logger.exception("Failed to stop headless Spinnaker recording")
            self.recorder.log_event(
                "video_recording_error",
                {
                    "backend": "spinnaker",
                    "stage": "stop",
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
        finally:
            self.camera_recorder = None

    def run(self) -> SessionSummary:
        assert self.bpod is not None and self.relay is not None and self.display is not None
        assert self.recorder is not None and self.dio is not None

        end_reason = "completed"
        try:
            for trial_index in range(1, self.config.n_trials + 1):
                if self._should_end_session():
                    end_reason = self._engagement_end_reason()
                    break

                if self.config.phase == TrainingPhase.CENTERING:
                    ok = self._run_centering_trial(trial_index)
                else:
                    ok = self._run_bpod_dm_trial(trial_index)

                if not ok:
                    self._consecutive_failures += 1
                else:
                    self._consecutive_failures = 0

                if self._should_end_session():
                    end_reason = self._engagement_end_reason()
                    break
        except BaseException as exc:
            end_reason = "keyboard_interrupt" if isinstance(exc, KeyboardInterrupt) else "error"
            self.recorder.log_event(
                "session_error",
                {"type": type(exc).__name__, "message": str(exc)},
            )
            raise
        finally:
            self.close(stop_monitor=False)
            if self.display is not None:
                self.display.close()
            summary = SessionSummary(
                subject_id=self.config.subject_id,
                phase=self.config.phase.value,
                n_trials_planned=self.config.n_trials,
                n_trials_completed=len(self.recorder.trials),
                end_reason=end_reason,
                consecutive_failures=self._consecutive_failures,
                session_start_wall_s=self.recorder._session_start_wall,
                session_end_wall_s=time.time(),
                engagement_idle_triggered=end_reason == "idle_timeout",
            )
            self._stop_operator_monitor(summary)
            path = self.recorder.finalize(summary)
            logger.info("Session data saved to %s", path)
        return summary

    def preflight(self, duration_s: float = 2.0) -> SessionSummary:
        """Exercise connected hardware streams without running a trial."""
        assert self.display is not None and self.recorder is not None and self.dio is not None
        duration_s = max(0.0, float(duration_s))
        start_camera_count = len(self.dio.get_camera_events())
        start_lick_count = self._lick_event_count()
        start_streamdio_count = len(self.recorder.streamdio_events)
        start_snap = self.wheel.snapshot()
        self.recorder.log_event(
            "preflight_start",
            {
                "duration_s": duration_s,
                "camera_events_start": start_camera_count,
                "lick_events_start": start_lick_count,
                "force_samples_start": start_snap["force_samples"],
            },
        )
        end_reason = "preflight"
        try:
            if duration_s > 0.0:
                self.display.show_black(duration_s)
            end_snap = self.wheel.snapshot()
            end_camera_count = len(self.dio.get_camera_events())
            end_lick_count = self._lick_event_count()
            raw_events = self.recorder.streamdio_events[start_streamdio_count:]
            digital_counts: dict[int, int] = {}
            for event_type, channel, _value, _timestamp_us, _relative_s, _host_s in raw_events:
                if event_type == "dio_input":
                    digital_counts[int(channel)] = digital_counts.get(int(channel), 0) + 1
            self.recorder.log_event(
                "preflight_end",
                {
                    "duration_s": duration_s,
                    "camera_events_seen": max(0, end_camera_count - start_camera_count),
                    "lick_events_seen": max(0, end_lick_count - start_lick_count),
                    "force_samples_seen": max(
                        0,
                        int(end_snap["force_samples"]) - int(start_snap["force_samples"]),
                    ),
                    "force_last": end_snap["force_last"],
                    "force_mean": end_snap["force_mean"],
                    "net_pulses": end_snap["net_pulses"],
                    "cw_pulses": end_snap["cw_pulses"],
                    "ccw_pulses": end_snap["ccw_pulses"],
                    "wheel_pulse_edges_seen": digital_counts.get(
                        self.config.wheel.pulse_channel,
                        0,
                    ),
                    "wheel_direction_edges_seen": digital_counts.get(
                        self.config.wheel.direction_channel,
                        0,
                    ),
                    "lick_input_edges_seen": sum(
                        digital_counts.get(channel, 0) for channel in self.config.licks.channels
                    ),
                },
            )
        except BaseException as exc:
            end_reason = "keyboard_interrupt" if isinstance(exc, KeyboardInterrupt) else "preflight_error"
            self.recorder.log_event(
                "preflight_error",
                {"type": type(exc).__name__, "message": str(exc)},
            )
            raise
        finally:
            self.close(stop_monitor=False)
            if self.display is not None:
                self.display.close()
            summary = SessionSummary(
                subject_id=self.config.subject_id,
                phase=self.config.phase.value,
                n_trials_planned=0,
                n_trials_completed=0,
                end_reason=end_reason,
                consecutive_failures=self._consecutive_failures,
                session_start_wall_s=self.recorder._session_start_wall,
                session_end_wall_s=time.time(),
                engagement_idle_triggered=False,
            )
            self._stop_operator_monitor(summary)
            path = self.recorder.finalize(summary)
            logger.info("Preflight data saved to %s", path)
        return summary

    def _should_end_session(self) -> bool:
        idle_s = time.perf_counter() - self._last_wheel_movement_mono
        if idle_s >= self.config.engagement.idle_timeout_s:
            return True
        if self._consecutive_failures >= self.config.engagement.max_consecutive_failures:
            return True
        return False

    def _engagement_end_reason(self) -> str:
        idle_s = time.perf_counter() - self._last_wheel_movement_mono
        if idle_s >= self.config.engagement.idle_timeout_s:
            return "idle_timeout"
        return "max_consecutive_failures"

    def _track_engagement_from_wheel(self) -> None:
        snap = self.wheel.snapshot()
        if snap["cw_pulses"] or snap["ccw_pulses"] or abs(snap["net_pulses"]) > 0:
            last_pulse_mono_s = time.perf_counter() - float(snap["quiescent_for_s"])
            self._last_wheel_movement_mono = max(
                self._last_wheel_movement_mono,
                last_pulse_mono_s,
            )

    def _wait_quiescence(self) -> None:
        assert self.recorder is not None
        self.recorder.log_event("quiescence_wait_start", {"required_s": self.config.timing.quiescence_s})
        self.wheel.reset_trial()
        while True:
            if self.display is not None:
                self.display.pump_events()
            self._track_engagement_from_wheel()
            snap = self.wheel.snapshot()
            if snap["quiescent_for_s"] >= self.config.timing.quiescence_s:
                self.recorder.log_event("quiescence_ok", {"still_s": snap["quiescent_for_s"]})
                return
            time.sleep(0.01)

    def _pre_trial_black(self) -> None:
        assert self.display is not None and self.recorder is not None
        self.recorder.log_event("pre_trial_black_start", {"duration_s": self.config.display.pre_trial_black_s})
        self.display.show_black(self.config.display.pre_trial_black_s)
        self.recorder.log_event("pre_trial_black_end", {})

    def _set_choice_loop_stats(
        self,
        *,
        trial_index: int,
        outcome: str,
        start_mono_s: float,
        end_mono_s: float,
        frame_count: int,
        frame_interval_sum_s: float,
        frame_interval_max_s: float,
        stall_count: int,
        reward_duration_s: float = 0.0,
    ) -> None:
        denominator = max(1, frame_count - 1)
        stats: dict[str, float | int | str | bool] = {
            "trial_index": int(trial_index),
            "outcome": outcome,
            "duration_s": max(0.0, end_mono_s - start_mono_s),
            "frames": int(frame_count),
            "mean_frame_s": frame_interval_sum_s / denominator,
            "max_frame_s": frame_interval_max_s,
            "stall_count": int(stall_count),
            "stall_threshold_s": float(self.config.timing.choice_loop_stall_s),
            "display_fps": int(self.config.display.fps),
            "reward_duration_s": float(reward_duration_s),
        }
        self._last_choice_loop_stats = stats
        if self.recorder is not None:
            self.recorder.log_event("choice_loop_stats", stats)

    @staticmethod
    def _actual_transition_time(transition: dict | list[dict]) -> float:
        transitions = transition if isinstance(transition, list) else [transition]
        return max(
            float(item.get("event_host_perf_counter_s", item["command_end_mono_s"]))
            for item in transitions
        )

    def _wait_for_bpod_trial_threads(self, timeout_s: float = 2.0) -> None:
        """Wait for bpod-core's read, event, and output-handler work to finish."""
        if self.bpod is None:
            return
        self.bpod.wait()

        event_thread = getattr(self.bpod, "_event_thread", None)
        if event_thread is not None and event_thread.is_alive():
            event_thread.join(timeout=max(0.0, float(timeout_s)))
            if event_thread.is_alive():
                raise RuntimeError("Timed out waiting for bpod-core trial data")

        softcode_thread = getattr(self.bpod, "_softcode_thread", None)
        if softcode_thread is not None and hasattr(softcode_thread, "drain"):
            softcode_thread.drain()

    def _handle_bpod_softcode(self, softcode: int) -> None:
        """Execute verified host outputs when the Bpod FSM enters a state."""
        runtime = self._active_trial_runtime
        if runtime is None:
            logger.warning("Ignoring Bpod output SoftCode %d without an active trial", softcode)
            return
        if self.relay is None or self.recorder is None:
            runtime.fail(RuntimeError("Bpod SoftCode arrived before task outputs were ready"))
            return

        try:
            if softcode == sc.STATE_QUIESCENCE:
                self._publish_fsm_state("Quiescence", runtime=runtime, phase="quiescence")
                runtime.enter("quiescence", softcode=softcode)
                return

            if softcode == sc.STATE_CHOICE:
                self.wheel.reset_trial()
                transition = self.relay.set_force_direction(runtime.force_direction)
                actual_on = self._actual_transition_time(transition)
                self.recorder.log_event(
                    "force_on",
                    {
                        "trial_index": runtime.trial_index,
                        "force_direction": runtime.force_direction,
                        "transitions": transition,
                        "actual_on_mono_s": actual_on,
                        "trigger": "bpod_state_entry",
                    },
                )
                self._on_choice_window_started(runtime)
                self._publish_fsm_state("ChoiceWindow", runtime=runtime, phase="choice")
                runtime.enter("choice", softcode=softcode, output_result=transition)
                return

            if softcode == sc.STATE_BLANK:
                runtime.close_choice()
                transition = self.relay.force_off()
                self.recorder.log_event(
                    "force_off",
                    {
                        "trial_index": runtime.trial_index,
                        "reason": "choice",
                        "transitions": transition,
                        "actual_off_mono_s": self._actual_transition_time(transition),
                        "trigger": "bpod_state_entry",
                    },
                )
                blank_state = (
                    "ChoiceBlankHigh"
                    if runtime.chosen_shape == "star"
                    else "ChoiceBlankLow"
                    if runtime.chosen_shape == "circle"
                    else "ChoiceBlank"
                )
                self._publish_fsm_state(blank_state, runtime=runtime, phase="blank")
                runtime.enter("blank", softcode=softcode, output_result=transition)
                return

            if softcode == sc.STATE_PUMP_ON:
                if runtime.chosen_shape not in ("star", "circle"):
                    raise RuntimeError("Bpod entered reward state before a host choice was committed")
                reward_s = (
                    self.config.reward.matched_pump_s
                    if runtime.chosen_shape == "star"
                    else self.config.reward.opposite_pump_s
                )
                transition = self.relay.pump_on()
                actual_on = self._actual_transition_time(transition)
                runtime.pump_actual_on_mono_s = actual_on
                self.recorder.log_event(
                    "pump_start",
                    {
                        "trial_index": runtime.trial_index,
                        "duration_s": reward_s,
                        "chosen_shape": runtime.chosen_shape,
                        "transition": transition,
                        "actual_on_mono_s": actual_on,
                        "trigger": "bpod_state_entry",
                    },
                )
                self._publish_monitor(
                    "pump",
                    {
                        "active": True,
                        "trial_index": runtime.trial_index,
                        "duration_s": reward_s,
                        "source": "bpod_state_entry",
                    },
                )
                self._publish_fsm_state(
                    "RewardHigh" if runtime.chosen_shape == "star" else "RewardLow",
                    runtime=runtime,
                    phase="reward",
                )
                runtime.enter("reward", softcode=softcode, output_result=transition)
                return

            if softcode == sc.STATE_PUMP_OFF:
                transition = self.relay.pump_off()
                actual_off = self._actual_transition_time(transition)
                actual_open = (
                    None
                    if runtime.pump_actual_on_mono_s is None
                    else actual_off - runtime.pump_actual_on_mono_s
                )
                self.recorder.log_event(
                    "pump_end",
                    {
                        "trial_index": runtime.trial_index,
                        "transition": transition,
                        "actual_off_mono_s": actual_off,
                        "actual_open_s": actual_open,
                        "trigger": "bpod_state_entry",
                    },
                )
                self._publish_monitor(
                    "pump",
                    {
                        "active": False,
                        "trial_index": runtime.trial_index,
                        "source": "bpod_state_entry",
                    },
                )
                self._publish_fsm_state(
                    "PostRewardHigh" if runtime.chosen_shape == "star" else "PostRewardLow",
                    runtime=runtime,
                    phase="post_reward",
                )
                runtime.enter("post_reward", softcode=softcode, output_result=transition)
                return

            if softcode == sc.STATE_TIMEOUT_ON:
                runtime.close_choice(clear=True)
                transition = self.relay.force_off()
                self.recorder.log_event(
                    "force_off",
                    {
                        "trial_index": runtime.trial_index,
                        "reason": "timeout",
                        "transitions": transition,
                        "actual_off_mono_s": self._actual_transition_time(transition),
                        "trigger": "bpod_state_entry",
                    },
                )
                self.recorder.log_event(
                    "white_noise_start",
                    {
                        "trial_index": runtime.trial_index,
                        "duration_s": self.config.timing.timeout_noise_s,
                        "source": "bpod_serial_pin27_gate",
                        "trigger": "bpod_state_entry",
                    },
                )
                self._publish_monitor(
                    "white_noise",
                    {
                        "active": True,
                        "trial_index": runtime.trial_index,
                        "duration_s": self.config.timing.timeout_noise_s,
                    },
                )
                self._publish_fsm_state("TimeoutNoise", runtime=runtime, phase="timeout")
                runtime.enter("timeout", softcode=softcode, output_result=transition)
                return

            if softcode == sc.STATE_TIMEOUT_OFF:
                self.recorder.log_event(
                    "white_noise_end",
                    {
                        "trial_index": runtime.trial_index,
                        "duration_s": self.config.timing.timeout_noise_s,
                        "source": "bpod_serial_pin27_gate",
                        "trigger": "bpod_state_entry",
                    },
                )
                self._publish_monitor(
                    "white_noise",
                    {
                        "active": False,
                        "trial_index": runtime.trial_index,
                        "duration_s": self.config.timing.timeout_noise_s,
                    },
                )
                self._publish_fsm_state("PostTimeout", runtime=runtime, phase="post_timeout")
                runtime.enter("post_timeout", softcode=softcode)
                return

            if softcode == sc.STATE_ABORT:
                result = self.relay.task_outputs_off()
                self._publish_fsm_state("Abort", runtime=runtime, phase="abort")
                runtime.enter("abort", softcode=softcode, output_result=result)
                return

            raise RuntimeError(f"Unknown Bpod output SoftCode {softcode}")
        except BaseException as exc:
            logger.exception("Bpod state output failed for SoftCode %d", softcode)
            runtime.fail(exc)

    @staticmethod
    def _opposite_side(side: Side) -> Side:
        return "right" if side == "left" else "left"

    @staticmethod
    def _wheel_direction_for_side(side: Side) -> Direction:
        return "cw" if side == "left" else "ccw"

    def _choose_star_side(self, trial_index: int) -> Side:
        del trial_index
        return self.rng.choice(["left", "right"])

    def _prepare_decision_trial(self, runtime: TrialRuntime) -> None:
        """Extension hook for the command-driven simulator and hardware self-test."""
        del runtime

    def _on_choice_window_started(self, runtime: TrialRuntime) -> None:
        """Extension hook called from the Bpod output handler after wheel reset."""
        del runtime

    def _after_decision_trial(self, runtime: TrialRuntime) -> None:
        """Extension hook called after the Bpod FSM exits."""
        del runtime

    def _decision_positions(self, net_position_deg: float) -> tuple[float, float]:
        distance = self.config.display.choice_distance_deg
        bounded = max(-distance, min(distance, float(net_position_deg)))
        left_position = -distance + max(0.0, bounded)
        right_position = distance + min(0.0, bounded)
        return left_position, right_position

    def _run_bpod_decision_epoch(
        self,
        *,
        star_side: Side,
        trial_index: int,
    ) -> dict:
        assert self.bpod is not None and self.display is not None and self.recorder is not None
        assert self.speaker_relay is not None

        circle_side = self._opposite_side(star_side)
        force_direction = self._wheel_direction_for_side(star_side)
        runtime = TrialRuntime(
            trial_index=trial_index,
            force_direction=force_direction,
            star_side=star_side,
            circle_side=circle_side,
        )
        self._active_trial_runtime = runtime
        self._prepare_decision_trial(runtime)
        fsm = build_host_supervised_decision_fsm(
            self.speaker_relay,
            quiescence_s=self.config.timing.quiescence_s,
            choice_timeout_s=self.config.timing.choice_timeout_s,
            choice_blank_s=self.config.timing.choice_blank_s,
            high_reward_s=self.config.reward.matched_pump_s,
            low_reward_s=self.config.reward.opposite_pump_s,
            post_reward_s=self.config.timing.post_reward_s,
            timeout_noise_s=self.config.timing.timeout_noise_s,
            post_timeout_s=self.config.timing.post_timeout_s,
        )
        self.recorder.log_fsm_definition("host_supervised_decision", fsm)
        self.wheel.reset_trial()
        self.display.fill_black()

        choice_start_mono_s: float | None = None
        final_left_deg = -self.config.display.choice_distance_deg
        final_right_deg = self.config.display.choice_distance_deg
        frame_count = 0
        frame_interval_sum_s = 0.0
        frame_interval_max_s = 0.0
        stall_count = 0
        previous_frame_mono_s: float | None = None
        stimulus_visible = False
        timeout_logged = False
        abort_sent = False
        last_phase = "starting"
        last_quiescence_edges = 0
        stall_threshold_s = max(
            float(self.config.timing.choice_loop_stall_s),
            (1.0 / max(1, self.config.display.fps)) * 3.0,
        )

        self.bpod.run(fsm, trial_number=trial_index)
        try:
            while self.bpod.is_running:
                state = runtime.snapshot()
                phase = str(state["phase"])
                if state["error"] and not abort_sent:
                    self.bpod.send_softcode(sc.SESSION_ABORT)
                    abort_sent = True

                if phase == "quiescence":
                    self._track_engagement_from_wheel()
                    wheel_snap = self.wheel.snapshot()
                    edges = int(wheel_snap["cw_pulses"]) + int(wheel_snap["ccw_pulses"])
                    if edges != last_quiescence_edges:
                        self.bpod.send_softcode(sc.QUIESCENCE_MOVEMENT)
                        self._publish_fsm_state(
                            "QuiescenceReset",
                            runtime=runtime,
                            phase="quiescence_reset",
                        )
                        last_quiescence_edges = edges
                    if last_phase != phase:
                        self.display.fill_black()
                        self._publish_monitor(
                            "task_state",
                            {"trial_index": trial_index, "phase": phase},
                        )
                        self._publish_task_screen()
                    else:
                        self.display.pump_events()
                    self._publish_wheel_snapshot(phase=phase, trial_index=trial_index)

                elif phase == "choice":
                    frame_mono_s = time.perf_counter()
                    if choice_start_mono_s is None:
                        choice_start_mono_s = float(state["phase_started_mono_s"])
                    if previous_frame_mono_s is not None:
                        interval_s = frame_mono_s - previous_frame_mono_s
                        frame_interval_sum_s += interval_s
                        frame_interval_max_s = max(frame_interval_max_s, interval_s)
                        if interval_s >= stall_threshold_s:
                            stall_count += 1
                            self.recorder.log_event(
                                "choice_loop_stall",
                                {
                                    "trial_index": trial_index,
                                    "interval_s": interval_s,
                                    "stall_threshold_s": stall_threshold_s,
                                    "frame": frame_count + 1,
                                },
                            )
                    previous_frame_mono_s = frame_mono_s
                    frame_count += 1
                    self._track_engagement_from_wheel()
                    wheel_snap = self.wheel.snapshot()
                    net_position_deg = (
                        int(wheel_snap["net_pulses"]) * self.config.wheel.deg_per_pulse
                    )
                    final_left_deg, final_right_deg = self._decision_positions(net_position_deg)
                    flip_mono_s = self.display.draw_decision(
                        star_side=star_side,
                        left_position_deg=final_left_deg,
                        right_position_deg=final_right_deg,
                        force_direction=force_direction,
                        trial_index=trial_index,
                    )
                    self._publish_monitor_throttled(
                        "task_state",
                        min(30.0, float(self.config.display.fps)),
                        "task_state",
                        {
                            "trial_index": trial_index,
                            "phase": phase,
                            "star_side": star_side,
                            "circle_side": circle_side,
                            "force_direction": force_direction,
                            "left_position_deg": final_left_deg,
                            "right_position_deg": final_right_deg,
                            "net_position_deg": net_position_deg,
                        },
                    )
                    self._publish_wheel_snapshot(phase=phase, trial_index=trial_index)
                    self._publish_task_screen()
                    if not stimulus_visible:
                        stimulus_visible = True
                        self.recorder.log_event(
                            "stimulus_on",
                            {
                                "trial_index": trial_index,
                                "star_side": star_side,
                                "circle_side": circle_side,
                                "left_position_deg": final_left_deg,
                                "right_position_deg": final_right_deg,
                                "display_flip_mono_s": flip_mono_s,
                                "bright_pixel_counts": self.display.shape_bright_pixel_counts,
                            },
                        )

                    threshold = self.config.display.choice_center_threshold_deg
                    selected_side: Side | None = None
                    if final_left_deg >= -threshold:
                        selected_side = "left"
                    elif final_right_deg <= threshold:
                        selected_side = "right"

                    if selected_side is not None:
                        chosen_shape: Shape = "star" if selected_side == star_side else "circle"
                        wheel_choice = self._wheel_direction_for_side(selected_side)
                        detected_mono_s = time.perf_counter()
                        if runtime.commit_choice(
                            shape=chosen_shape,
                            side=selected_side,
                            wheel_choice=wheel_choice,
                            detected_mono_s=detected_mono_s,
                        ):
                            response_time_s = max(
                                0.0,
                                detected_mono_s - float(choice_start_mono_s),
                            )
                            reward_s = (
                                self.config.reward.matched_pump_s
                                if chosen_shape == "star"
                                else self.config.reward.opposite_pump_s
                            )
                            self.recorder.log_event(
                                "choice_detected",
                                {
                                    "trial_index": trial_index,
                                    "choice": wheel_choice,
                                    "choice_side": selected_side,
                                    "chosen_shape": chosen_shape,
                                    "force_direction": force_direction,
                                    "matched": chosen_shape == "star",
                                    "response_time_s": response_time_s,
                                    "reward_duration_s": reward_s,
                                    "host_to_bpod_softcode": (
                                        sc.CHOICE_HIGH
                                        if chosen_shape == "star"
                                        else sc.CHOICE_LOW
                                    ),
                                },
                            )
                            self._publish_monitor(
                                "task_state",
                                {
                                    "trial_index": trial_index,
                                    "phase": "choice_detected",
                                    "choice": wheel_choice,
                                    "choice_side": selected_side,
                                    "chosen_shape": chosen_shape,
                                    "matched": chosen_shape == "star",
                                    "response_time_s": response_time_s,
                                    "reward_duration_s": reward_s,
                                },
                            )
                            self.bpod.send_softcode(
                                sc.CHOICE_HIGH if chosen_shape == "star" else sc.CHOICE_LOW
                            )

                else:
                    if stimulus_visible:
                        blank_flip_mono_s = self.display.fill_black()
                        stimulus_visible = False
                        self._publish_monitor(
                            "task_state",
                            {"trial_index": trial_index, "phase": phase},
                        )
                        self._publish_task_screen()
                        self.recorder.log_event(
                            "stimulus_off",
                            {
                                "trial_index": trial_index,
                                "bpod_phase": phase,
                                "display_flip_mono_s": blank_flip_mono_s,
                            },
                        )
                    else:
                        self.display.pump_events()
                        if last_phase != phase:
                            self._publish_monitor(
                                "trial_phase",
                                {"trial_index": trial_index, "phase": phase},
                            )
                    if phase == "timeout" and not timeout_logged:
                        timeout_logged = True
                        self.recorder.log_event(
                            "choice_timeout",
                            {
                                "trial_index": trial_index,
                                "force_direction": force_direction,
                                "response_time_s": self.config.timing.choice_timeout_s,
                                "trigger": "bpod_choice_window_tup",
                            },
                        )

                last_phase = phase
                self.display.clock.tick(self.config.display.fps)
        except BaseException:
            if self.bpod.is_running:
                try:
                    self.bpod.send_softcode(sc.SESSION_ABORT)
                    self._wait_for_bpod_trial_threads()
                except Exception:
                    logger.exception("Failed to abort Bpod trial after host exception")
            raise
        finally:
            self._wait_for_bpod_trial_threads()

        bpod_data = self.bpod.get_data(concat=False)
        self.recorder.log_bpod_trial(trial_index, bpod_data)
        self._after_decision_trial(runtime)
        state = runtime.snapshot()
        self._active_trial_runtime = None
        if state["error"]:
            raise RuntimeError(f"Bpod trial output failed: {state['error']}")

        chosen_shape = state["chosen_shape"]
        choice_side = state["choice_side"]
        wheel_choice = state["wheel_choice"] or "timeout"
        responded = chosen_shape in ("star", "circle")
        response_time_s = (
            self.config.timing.choice_timeout_s
            if state["choice_detected_mono_s"] is None or choice_start_mono_s is None
            else max(0.0, float(state["choice_detected_mono_s"]) - choice_start_mono_s)
        )
        reward_s = (
            self.config.reward.matched_pump_s
            if chosen_shape == "star"
            else self.config.reward.opposite_pump_s
            if chosen_shape == "circle"
            else 0.0
        )
        final_position_deg = (
            final_left_deg
            if choice_side == "left"
            else final_right_deg
            if choice_side == "right"
            else final_left_deg if abs(final_left_deg) < abs(final_right_deg) else final_right_deg
        )
        choice_end_mono_s = (
            float(state["choice_detected_mono_s"])
            if state["choice_detected_mono_s"] is not None
            else (choice_start_mono_s or time.perf_counter()) + self.config.timing.choice_timeout_s
        )
        self._set_choice_loop_stats(
            trial_index=trial_index,
            outcome=str(wheel_choice),
            start_mono_s=choice_start_mono_s or choice_end_mono_s,
            end_mono_s=choice_end_mono_s,
            frame_count=frame_count,
            frame_interval_sum_s=frame_interval_sum_s,
            frame_interval_max_s=frame_interval_max_s,
            stall_count=stall_count,
            reward_duration_s=reward_s,
        )
        return {
            "force_direction": force_direction,
            "star_side": star_side,
            "circle_side": circle_side,
            "choice": wheel_choice,
            "choice_side": choice_side,
            "chosen_shape": chosen_shape,
            "matched": chosen_shape == "star",
            "responded": responded,
            "response_time_s": response_time_s,
            "reward_duration_s": reward_s,
            "final_position_deg": final_position_deg,
            "runtime": state,
        }

    def _run_live_choice_epoch_legacy(
        self,
        force_direction: Direction,
        trial_index: int,
    ) -> tuple[Literal["cw", "ccw", "timeout"], float, float, float]:
        assert self.display is not None and self.relay is not None and self.recorder is not None
        choice_distance = self.config.display.choice_distance_deg
        self.wheel.reset_trial()
        force_transition = self.relay.set_force_direction(force_direction)
        self.recorder.log_event(
            "force_on",
            {
                "trial_index": trial_index,
                "force_direction": force_direction,
                "transitions": force_transition,
                "actual_on_mono_s": self._actual_transition_time(force_transition),
            },
        )
        stimulus_flip_mono_s = self.display.draw_trial(
            force_direction,
            0.0,
            trial_index=trial_index,
        )
        self.recorder.log_event(
            "stimulus_on",
            {
                "trial_index": trial_index,
                "force_direction": force_direction,
                "position_deg": 0.0,
                "display_flip_mono_s": stimulus_flip_mono_s,
            },
        )

        start = stimulus_flip_mono_s
        deadline = start + self.config.timing.choice_timeout_s
        final_pos_deg = 0.0
        response_time_s = self.config.timing.choice_timeout_s
        frame_count = 0
        frame_interval_sum_s = 0.0
        frame_interval_max_s = 0.0
        stall_count = 0
        previous_frame_mono_s: float | None = None
        stall_threshold_s = max(
            float(self.config.timing.choice_loop_stall_s),
            (1.0 / max(1, self.config.display.fps)) * 3.0,
        )

        while True:
            frame_mono_s = time.perf_counter()
            if previous_frame_mono_s is not None:
                interval_s = frame_mono_s - previous_frame_mono_s
                frame_interval_sum_s += interval_s
                frame_interval_max_s = max(frame_interval_max_s, interval_s)
                if interval_s >= stall_threshold_s:
                    stall_count += 1
                    self.recorder.log_event(
                        "choice_loop_stall",
                        {
                            "trial_index": trial_index,
                            "interval_s": interval_s,
                            "stall_threshold_s": stall_threshold_s,
                            "frame": frame_count + 1,
                        },
                    )
            previous_frame_mono_s = frame_mono_s
            frame_count += 1
            self._track_engagement_from_wheel()
            snap = self.wheel.snapshot()
            final_pos_deg = max(
                -choice_distance,
                min(choice_distance, snap["net_pulses"] * self.config.wheel.deg_per_pulse),
            )
            elapsed = time.perf_counter() - start
            self.display.draw_trial(force_direction, final_pos_deg, trial_index=trial_index)

            choice: Literal["cw", "ccw", "timeout"] | None = None
            if final_pos_deg >= choice_distance:
                choice = "cw"
            elif final_pos_deg <= -choice_distance:
                choice = "ccw"

            if choice is not None:
                response_time_s = elapsed
                matched = choice == force_direction
                reward_s = (
                    self.config.reward.matched_pump_s
                    if matched
                    else self.config.reward.opposite_pump_s
                )
                outcome = "matched" if matched else "opposite"
                self.recorder.log_event(
                    "choice_detected",
                    {
                        "trial_index": trial_index,
                        "choice": choice,
                        "force_direction": force_direction,
                        "matched": matched,
                        "response_time_s": response_time_s,
                        "final_position_deg": final_pos_deg,
                        "reward_duration_s": reward_s,
                    },
                )
                force_off_cmd_mono_s = time.perf_counter()
                force_off_transition = self.relay.force_off()
                force_off_done_mono_s = time.perf_counter()
                self.recorder.log_event(
                    "force_off",
                    {
                        "trial_index": trial_index,
                        "reason": "choice",
                        "command_start_mono_s": force_off_cmd_mono_s,
                        "command_end_mono_s": force_off_done_mono_s,
                        "transitions": force_off_transition,
                        "actual_off_mono_s": self._actual_transition_time(force_off_transition),
                    },
                )
                pump_cmd_start_mono_s = time.perf_counter()
                pump_on_transition = self.relay.pump_on()
                pump_on_done_mono_s = time.perf_counter()
                pump_actual_on_mono_s = self._actual_transition_time(pump_on_transition)
                self.recorder.log_event(
                    "pump_start",
                    {
                        "trial_index": trial_index,
                        "duration_s": reward_s,
                        "matched": matched,
                        "command_start_mono_s": pump_cmd_start_mono_s,
                        "command_end_mono_s": pump_on_done_mono_s,
                        "transition": pump_on_transition,
                        "actual_on_mono_s": pump_actual_on_mono_s,
                    },
                )
                self._publish_monitor(
                    "pump",
                    {
                        "active": True,
                        "trial_index": trial_index,
                        "duration_s": reward_s,
                        "source": "host_choice_epoch",
                    },
                )
                feedback_flip_mono_s = self.display.draw_trial(
                    force_direction,
                    final_pos_deg,
                    outcome=outcome,
                    trial_index=trial_index,
                    response_time_s=response_time_s,
                )
                self.recorder.log_event(
                    "feedback_on",
                    {
                        "trial_index": trial_index,
                        "outcome": outcome,
                        "position_deg": final_pos_deg,
                        "display_flip_mono_s": feedback_flip_mono_s,
                    },
                )
                self._publish_task_screen()
                wait_until_perf_counter(pump_actual_on_mono_s + reward_s)
                pump_off_cmd_start_mono_s = time.perf_counter()
                pump_off_transition = self.relay.pump_off()
                pump_cmd_end_mono_s = time.perf_counter()
                pump_actual_off_mono_s = self._actual_transition_time(pump_off_transition)
                self.recorder.log_event(
                    "pump_end",
                    {
                        "trial_index": trial_index,
                        "duration_s": reward_s,
                        "matched": matched,
                        "command_start_mono_s": pump_cmd_start_mono_s,
                        "pump_on_done_mono_s": pump_on_done_mono_s,
                        "pump_off_command_start_mono_s": pump_off_cmd_start_mono_s,
                        "command_end_mono_s": pump_cmd_end_mono_s,
                        "transition": pump_off_transition,
                        "actual_off_mono_s": pump_actual_off_mono_s,
                        "actual_open_s": pump_actual_off_mono_s - pump_actual_on_mono_s,
                    },
                )
                self._publish_monitor(
                    "pump",
                    {
                        "active": False,
                        "trial_index": trial_index,
                        "source": "host_choice_epoch",
                    },
                )
                self._set_choice_loop_stats(
                    trial_index=trial_index,
                    outcome=choice,
                    start_mono_s=start,
                    end_mono_s=force_off_cmd_mono_s,
                    frame_count=frame_count,
                    frame_interval_sum_s=frame_interval_sum_s,
                    frame_interval_max_s=frame_interval_max_s,
                    stall_count=stall_count,
                    reward_duration_s=reward_s,
                )
                return choice, response_time_s, final_pos_deg, reward_s

            if time.perf_counter() >= deadline:
                response_time_s = min(elapsed, self.config.timing.choice_timeout_s)
                self.recorder.log_event(
                    "choice_timeout",
                    {
                        "trial_index": trial_index,
                        "force_direction": force_direction,
                        "response_time_s": response_time_s,
                        "final_position_deg": final_pos_deg,
                    },
                )
                force_off_cmd_mono_s = time.perf_counter()
                force_off_transition = self.relay.force_off()
                force_off_done_mono_s = time.perf_counter()
                self.recorder.log_event(
                    "force_off",
                    {
                        "trial_index": trial_index,
                        "reason": "timeout",
                        "command_start_mono_s": force_off_cmd_mono_s,
                        "command_end_mono_s": force_off_done_mono_s,
                        "transitions": force_off_transition,
                        "actual_off_mono_s": self._actual_transition_time(force_off_transition),
                    },
                )
                feedback_flip_mono_s = self.display.draw_trial(
                    force_direction,
                    final_pos_deg,
                    outcome="timeout",
                    trial_index=trial_index,
                    response_time_s=response_time_s,
                )
                self.recorder.log_event(
                    "feedback_on",
                    {
                        "trial_index": trial_index,
                        "outcome": "timeout",
                        "position_deg": final_pos_deg,
                        "display_flip_mono_s": feedback_flip_mono_s,
                    },
                )
                noise_cmd_start_mono_s = time.perf_counter()
                self.recorder.log_event(
                    "white_noise_start",
                    {
                        "trial_index": trial_index,
                        "duration_s": self.config.timing.timeout_noise_s,
                        "command_start_mono_s": noise_cmd_start_mono_s,
                    },
                )
                noise_result = self._pulse_audible_white_noise(
                    self.config.timing.timeout_noise_s
                )
                noise_cmd_end_mono_s = time.perf_counter()
                self.recorder.log_event(
                    "white_noise_end",
                    {
                        "trial_index": trial_index,
                        "duration_s": self.config.timing.timeout_noise_s,
                        "command_start_mono_s": noise_cmd_start_mono_s,
                        "command_end_mono_s": noise_cmd_end_mono_s,
                        "result": noise_result,
                    },
                )
                self._set_choice_loop_stats(
                    trial_index=trial_index,
                    outcome="timeout",
                    start_mono_s=start,
                    end_mono_s=force_off_cmd_mono_s,
                    frame_count=frame_count,
                    frame_interval_sum_s=frame_interval_sum_s,
                    frame_interval_max_s=frame_interval_max_s,
                    stall_count=stall_count,
                )
                return "timeout", response_time_s, final_pos_deg, 0.0

            self.display.clock.tick(self.config.display.fps)

    def _run_iti(self) -> None:
        assert self.recorder is not None
        duration_s = self.config.timing.inter_trial_min_s
        self.recorder.log_event("iti_start", {"duration_s": duration_s})
        if self.display is not None:
            self.display.show_black(duration_s)
        else:
            time.sleep(duration_s)
        self.recorder.log_event("iti_end", {"duration_s": duration_s})

    def _run_dm_trial_legacy(self, trial_index: int) -> bool:
        assert (
            self.bpod is not None
            and self.relay is not None
            and self.recorder is not None
            and self.display is not None
            and self.dio is not None
        )
        force_direction = self._choose_force_direction(trial_index)
        trial_wall = time.time()
        trial_mono = time.perf_counter()
        camera_start_idx = len(self.dio.get_camera_events())
        lick_start_idx = self._lick_event_count()
        self.recorder.log_event("trial_start", {"trial_index": trial_index, "force_direction": force_direction})
        self._publish_monitor(
            "trial_start",
            {
                "trial_index": trial_index,
                "phase": self.config.phase.value,
                "force_direction": force_direction,
            },
        )

        self._pre_trial_black()
        self._wait_quiescence()

        choice, rt, pos_deg, reward_s = self._run_live_choice_epoch_legacy(
            force_direction,
            trial_index,
        )
        loop_stats = dict(self._last_choice_loop_stats)

        matched = False
        responded = choice in ("cw", "ccw")

        if responded:
            matched = choice == force_direction
            outcome = "matched" if matched else "opposite"
            self.display.hold_trial(
                self.config.display.feedback_hold_s,
                force_direction,
                pos_deg,
                outcome,
                trial_index=trial_index,
                response_time_s=rt,
            )
        else:
            self.display.hold_trial(
                self.config.display.feedback_hold_s,
                force_direction,
                pos_deg,
                "timeout",
                trial_index=trial_index,
                response_time_s=rt,
            )

        snap = self.wheel.snapshot()
        cam_events = self.dio.get_camera_events()[camera_start_idx:]
        lick_events = self._lick_events_since(lick_start_idx)
        self._add_trial_record(
            TrialRecord(
                trial_index=trial_index,
                phase=self.config.phase.value,
                force_direction=force_direction,
                choice=choice,
                matched=matched,
                reward_duration_s=reward_s,
                responded=responded,
                response_time_s=rt,
                net_pulses=snap["net_pulses"],
                cw_pulses=snap["cw_pulses"],
                ccw_pulses=snap["ccw_pulses"],
                final_position_deg=pos_deg,
                force_samples=snap["force_samples"],
                force_last=snap["force_last"],
                force_mean=snap["force_mean"],
                force_min=snap["force_min"],
                force_max=snap["force_max"],
                trial_start_wall_s=trial_wall,
                trial_start_mono_s=trial_mono,
                camera_ttl_timestamps_us=[ts for ts, _ in cam_events],
                lick_count=len(lick_events),
                lick_timestamps_us=[int(event["timestamp_us"]) for event in lick_events],
                lick_relative_times_s=[float(event["relative_time_s"]) for event in lick_events],
                lick_channels=[int(event["channel"]) for event in lick_events],
                choice_loop_frames=int(loop_stats.get("frames", 0)),
                choice_loop_duration_s=float(loop_stats.get("duration_s", 0.0)),
                choice_loop_mean_frame_s=float(loop_stats.get("mean_frame_s", 0.0)),
                choice_loop_max_frame_s=float(loop_stats.get("max_frame_s", 0.0)),
                choice_loop_stall_count=int(loop_stats.get("stall_count", 0)),
            )
        )
        self._run_iti()
        return responded

    def _run_bpod_dm_trial(self, trial_index: int) -> bool:
        assert self.recorder is not None and self.dio is not None
        trial_wall = time.time()
        trial_mono = time.perf_counter()
        camera_start_idx = len(self.dio.get_camera_events())
        lick_start_idx = self._lick_event_count()
        star_side = self._choose_star_side(trial_index)
        circle_side = self._opposite_side(star_side)
        force_direction = self._wheel_direction_for_side(star_side)
        self.recorder.log_event(
            "trial_start",
            {
                "trial_index": trial_index,
                "star_side": star_side,
                "circle_side": circle_side,
                "force_direction": force_direction,
                "trial_controller": "bpod_core_fsm",
            },
        )
        self._publish_monitor(
            "trial_start",
            {
                "trial_index": trial_index,
                "phase": self.config.phase.value,
                "star_side": star_side,
                "circle_side": circle_side,
                "force_direction": force_direction,
            },
        )

        result = self._run_bpod_decision_epoch(star_side=star_side, trial_index=trial_index)
        loop_stats = dict(self._last_choice_loop_stats)
        snap = self.wheel.snapshot()
        cam_events = self.dio.get_camera_events()[camera_start_idx:]
        lick_events = self._lick_events_since(lick_start_idx)
        self._add_trial_record(
            TrialRecord(
                trial_index=trial_index,
                phase=self.config.phase.value,
                force_direction=result["force_direction"],
                choice=result["choice"],
                matched=bool(result["matched"]),
                reward_duration_s=float(result["reward_duration_s"]),
                responded=bool(result["responded"]),
                response_time_s=float(result["response_time_s"]),
                net_pulses=snap["net_pulses"],
                cw_pulses=snap["cw_pulses"],
                ccw_pulses=snap["ccw_pulses"],
                final_position_deg=float(result["final_position_deg"]),
                force_samples=snap["force_samples"],
                force_last=snap["force_last"],
                force_mean=snap["force_mean"],
                force_min=snap["force_min"],
                force_max=snap["force_max"],
                trial_start_wall_s=trial_wall,
                trial_start_mono_s=trial_mono,
                star_side=result["star_side"],
                circle_side=result["circle_side"],
                choice_side=result["choice_side"],
                chosen_shape=result["chosen_shape"],
                bpod_state_entries=len(result["runtime"]["state_entries"]),
                camera_ttl_timestamps_us=[ts for ts, _ in cam_events],
                lick_count=len(lick_events),
                lick_timestamps_us=[int(event["timestamp_us"]) for event in lick_events],
                lick_relative_times_s=[float(event["relative_time_s"]) for event in lick_events],
                lick_channels=[int(event["channel"]) for event in lick_events],
                choice_loop_frames=int(loop_stats.get("frames", 0)),
                choice_loop_duration_s=float(loop_stats.get("duration_s", 0.0)),
                choice_loop_mean_frame_s=float(loop_stats.get("mean_frame_s", 0.0)),
                choice_loop_max_frame_s=float(loop_stats.get("max_frame_s", 0.0)),
                choice_loop_stall_count=int(loop_stats.get("stall_count", 0)),
            )
        )
        return bool(result["responded"])

    def _choose_force_direction(self, trial_index: int) -> Direction:
        del trial_index
        return self.rng.choice(["cw", "ccw"])

    def _prepare_centering_trial(self, start_position_deg: float) -> None:
        """Extension hook used by the hardware-free simulator."""
        del start_position_deg

    def _run_centering_trial(self, trial_index: int) -> bool:
        assert (
            self.relay is not None
            and self.recorder is not None
            and self.dio is not None
            and self.display is not None
        )
        trial_wall = time.time()
        trial_mono = time.perf_counter()
        camera_start_idx = len(self.dio.get_camera_events())
        lick_start_idx = self._lick_event_count()
        start_position_deg = self.rng.choice(
            [-self.config.display.choice_distance_deg, self.config.display.choice_distance_deg]
        )
        movement_direction: Direction = "cw" if start_position_deg < 0.0 else "ccw"
        self.recorder.log_event(
            "centering_trial_start",
            {
                "trial_index": trial_index,
                "start_position_deg": start_position_deg,
                "movement_direction": movement_direction,
                "threshold_deg": self.config.centering_threshold_deg,
            },
        )
        self._publish_monitor(
            "trial_start",
            {
                "trial_index": trial_index,
                "phase": self.config.phase.value,
                "force_direction": movement_direction,
            },
        )
        self._pre_trial_black()
        self._wait_quiescence()

        threshold = self.config.centering_threshold_deg
        centered = False
        self._prepare_centering_trial(start_position_deg)
        self.wheel.reset_trial()
        stimulus_flip_mono_s = self.display.draw_centering(
            movement_direction,
            start_position_deg,
            trial_index=trial_index,
        )
        self._publish_monitor(
            "task_state",
            {
                "trial_index": trial_index,
                "phase": "centering",
                "force_direction": movement_direction,
                "position_deg": start_position_deg,
            },
        )
        self._publish_task_screen()
        self.recorder.log_event(
            "stimulus_on",
            {
                "trial_index": trial_index,
                "phase": "centering",
                "movement_direction": movement_direction,
                "position_deg": start_position_deg,
                "display_flip_mono_s": stimulus_flip_mono_s,
            },
        )
        start = stimulus_flip_mono_s
        deadline = start + self.config.timing.choice_timeout_s
        position_deg = start_position_deg
        while time.perf_counter() < deadline:
            self._track_engagement_from_wheel()
            position_deg = max(
                -self.config.display.choice_distance_deg,
                min(
                    self.config.display.choice_distance_deg,
                    start_position_deg + self.wheel.signed_position_deg(),
                ),
            )
            self.display.draw_centering(
                movement_direction,
                position_deg,
                trial_index=trial_index,
            )
            self._publish_monitor_throttled(
                "task_state",
                min(30.0, float(self.config.display.fps)),
                "task_state",
                {
                    "trial_index": trial_index,
                    "phase": "centering",
                    "force_direction": movement_direction,
                    "position_deg": position_deg,
                },
            )
            self._publish_wheel_snapshot(phase="centering", trial_index=trial_index)
            self._publish_task_screen()
            if abs(position_deg) <= threshold:
                centered = True
                break
            self.display.clock.tick(self.config.display.fps)

        reward_s = 0.0
        response_time_s = min(
            time.perf_counter() - start,
            self.config.timing.choice_timeout_s,
        )
        if centered:
            reward_s = self.config.reward.matched_pump_s
            pump_command_start_mono_s = time.perf_counter()
            pump_on_transition = self.relay.pump_on()
            pump_on_done_mono_s = time.perf_counter()
            pump_actual_on_mono_s = self._actual_transition_time(pump_on_transition)
            self.recorder.log_event(
                "pump_start",
                {
                    "trial_index": trial_index,
                    "duration_s": reward_s,
                    "reason": "centering",
                    "command_start_mono_s": pump_command_start_mono_s,
                    "command_end_mono_s": pump_on_done_mono_s,
                    "transition": pump_on_transition,
                    "actual_on_mono_s": pump_actual_on_mono_s,
                },
            )
            self._publish_monitor(
                "pump",
                {
                    "active": True,
                    "trial_index": trial_index,
                    "duration_s": reward_s,
                    "source": "centering",
                },
            )
            feedback_flip_mono_s = self.display.draw_centering(
                movement_direction,
                position_deg,
                outcome="centered",
                trial_index=trial_index,
                response_time_s=response_time_s,
            )
            self.recorder.log_event(
                "feedback_on",
                {
                    "trial_index": trial_index,
                    "outcome": "centered",
                    "position_deg": position_deg,
                    "display_flip_mono_s": feedback_flip_mono_s,
                },
            )
            self._publish_task_screen()
            wait_until_perf_counter(pump_actual_on_mono_s + reward_s)
            pump_off_command_start_mono_s = time.perf_counter()
            pump_off_transition = self.relay.pump_off()
            pump_actual_off_mono_s = self._actual_transition_time(pump_off_transition)
            self.recorder.log_event(
                "pump_end",
                {
                    "trial_index": trial_index,
                    "duration_s": reward_s,
                    "reason": "centering",
                    "pump_on_done_mono_s": pump_on_done_mono_s,
                    "pump_off_command_start_mono_s": pump_off_command_start_mono_s,
                    "transition": pump_off_transition,
                    "actual_off_mono_s": pump_actual_off_mono_s,
                    "actual_open_s": pump_actual_off_mono_s - pump_actual_on_mono_s,
                    "command_end_mono_s": time.perf_counter(),
                },
            )
            self._publish_monitor(
                "pump",
                {
                    "active": False,
                    "trial_index": trial_index,
                    "source": "centering",
                },
            )
            outcome = "centered"
        else:
            feedback_flip_mono_s = self.display.draw_centering(
                movement_direction,
                position_deg,
                outcome="timeout",
                trial_index=trial_index,
                response_time_s=response_time_s,
            )
            self.recorder.log_event(
                "feedback_on",
                {
                    "trial_index": trial_index,
                    "outcome": "timeout",
                    "position_deg": position_deg,
                    "display_flip_mono_s": feedback_flip_mono_s,
                },
            )
            self._publish_task_screen()
            self.recorder.log_event(
                "white_noise_start",
                {
                    "trial_index": trial_index,
                    "duration_s": self.config.timing.timeout_noise_s,
                    "reason": "centering_timeout",
                },
            )
            noise_result = self._pulse_audible_white_noise(
                self.config.timing.timeout_noise_s
            )
            self.recorder.log_event(
                "white_noise_end",
                {
                    "trial_index": trial_index,
                    "duration_s": self.config.timing.timeout_noise_s,
                    "reason": "centering_timeout",
                    "result": noise_result,
                },
            )
            outcome = "timeout"

        self.display.hold_centering(
            self.config.display.feedback_hold_s,
            movement_direction,
            position_deg,
            outcome,
            trial_index=trial_index,
            response_time_s=response_time_s,
        )

        snap = self.wheel.snapshot()
        cam_events = self.dio.get_camera_events()[camera_start_idx:]
        lick_events = self._lick_events_since(lick_start_idx)
        self._add_trial_record(
            TrialRecord(
                trial_index=trial_index,
                phase="centering",
                force_direction=movement_direction,
                choice="centering" if centered else "timeout",
                matched=centered,
                reward_duration_s=reward_s,
                responded=centered,
                response_time_s=response_time_s,
                net_pulses=snap["net_pulses"],
                cw_pulses=snap["cw_pulses"],
                ccw_pulses=snap["ccw_pulses"],
                final_position_deg=position_deg,
                force_samples=snap["force_samples"],
                force_last=snap["force_last"],
                force_mean=snap["force_mean"],
                force_min=snap["force_min"],
                force_max=snap["force_max"],
                trial_start_wall_s=trial_wall,
                trial_start_mono_s=trial_mono,
                camera_ttl_timestamps_us=[ts for ts, _ in cam_events],
                lick_count=len(lick_events),
                lick_timestamps_us=[int(event["timestamp_us"]) for event in lick_events],
                lick_relative_times_s=[float(event["relative_time_s"]) for event in lick_events],
                lick_channels=[int(event["channel"]) for event in lick_events],
            )
        )
        self._run_iti()
        return centered
