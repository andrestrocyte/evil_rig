"""Session orchestration: bpod-core session + verified StreamDIO I/O + display."""

from __future__ import annotations

import logging
import random
import threading
import time
from pathlib import Path
from typing import Literal

from bpod_core.bpod import Bpod
from bpod_core.fsm import StateMachine

from hf_dm.audio.white_noise import WhiteNoisePlayer
from hf_dm.config import (
    SessionConfig,
    TrainingPhase,
    config_to_dict,
    task_design_summary,
    validate_session_config,
)
from hf_dm.camera.spinnaker import HeadlessSpinnakerRecorder
from hf_dm.data.session_export import resolve_session_output, session_bundle_dir
from hf_dm.hardware.discovery import connect_bpod_core
from hf_dm.data.recorder import SessionRecorder, SessionSummary, TrialRecord
from hf_dm.display.pygame_display import TaskDisplay
from hf_dm.frame2ttl import (
    TOTAL_BITS as FRAME2TTL_TOTAL_BITS,
    decode_frame2ttl_barcode_events,
    encode_frame2ttl_trial_code,
    frame2ttl_barcode_states,
)
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
    STREAMDIO_STATE_MARKER_IDS,
    STREAMDIO_STATE_MARKER_NAMES,
)
from hf_dm.monitor import OperatorMonitor
from hf_dm.streamdio.client import StreamDIOClient
from hf_dm.streamdio.outputs import StreamDIOOutputs
from hf_dm.streamdio.wheel import Direction, WheelAccumulator

logger = logging.getLogger(__name__)


class HFDecisionSession:
    def __init__(self, config: SessionConfig):
        self.config = validate_session_config(config)
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
        self._frame2ttl_lock = threading.Lock()
        self._frame2ttl_events: list[dict[str, float | int]] = []
        self._closed = False
        self._last_choice_loop_stats: dict[str, float | int | str | bool] = {}
        self._active_trial_runtime: TrialRuntime | None = None
        self._streamdio_status_poll_stop = threading.Event()
        self._streamdio_status_poll_thread: threading.Thread | None = None
        self._streamdio_module_status_snapshot_count = 0
        self._streamdio_module_status_error_count = 0
        self._latest_streamdio_module_status: dict | None = None
        self._cap_touch_status_snapshot_count = 0
        self._latest_cap_touch_status: dict | None = None
        self._trial_schedule = self._build_trial_schedule()

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
        self.dio.configure_rotary_and_force(
            lick_channels=self.config.licks.channels,
            extra_input_channels=self._frame2ttl_input_channels(),
        )
        self.display = TaskDisplay(self.config.display, self.config.frame2ttl)
        out, base_name = resolve_session_output(
            Path(self.config.results_dir),
            subject_id=self.config.subject_id,
            phase=self.config.phase.value,
            layout=self.config.session_export.layout,
        )
        self.recorder = SessionRecorder(
            out,
            self.config.subject_id,
            self.config.phase.value,
            base_name=base_name,
        )
        self.recorder.set_export_options(config_to_dict(self.config)["session_export"])
        self.recorder.set_metadata(self._session_metadata())
        self.recorder.register_stream(
            "streamdio",
            {
                "type": "streamdio",
                "config_port": self.dio.config_port,
                "stream_port": self.dio.stream_port,
                "module_info": self._streamdio_module_info(),
                "force_channel": FORCE_ANALOG_CHANNEL,
                "lick_channels": list(self.config.licks.channels),
                "frame2ttl_channels": self._frame2ttl_input_channels(),
            },
        )
        self._configure_monitor_record_path()
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
        self._start_streamdio_status_poll()
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
                "streamdio_module_info": self._streamdio_module_info(),
                "camera_hz": self.config.camera.frequency_hz,
                "camera_backend": self.config.camera.backend,
                "lick_channels": list(self.config.licks.channels),
                "frame2ttl": self._frame2ttl_descriptor(),
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
            self._stop_streamdio_status_poll()
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
                if self.recorder is not None:
                    self.recorder.log_event(
                        "white_noise_shutdown",
                        {"status": self._sound_status()},
                    )
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

    def _build_trial_schedule(self) -> list[dict[str, str | None]]:
        task = self.config.task
        n_forced = max(0, int(task.n_forced_trials))
        if task.n_free_trials is None:
            n_free = max(0, int(self.config.n_trials) - n_forced)
        else:
            n_free = max(0, int(task.n_free_trials))
        total = n_free + n_forced
        if total > 0:
            self.config.n_trials = total

        schedule: list[dict[str, str | None]] = [
            {"trial_mode": "free", "forced_shape": None} for _ in range(n_free)
        ]
        forced_shapes = ["star" if idx % 2 == 0 else "circle" for idx in range(n_forced)]
        self.rng.shuffle(forced_shapes)
        schedule.extend(
            {"trial_mode": "forced", "forced_shape": shape} for shape in forced_shapes
        )
        self.rng.shuffle(schedule)
        return schedule

    def _trial_spec(self, trial_index: int) -> dict[str, str | None]:
        if 1 <= trial_index <= len(self._trial_schedule):
            return dict(self._trial_schedule[trial_index - 1])
        return {"trial_mode": "free", "forced_shape": None}

    def _n_trials_by_mode(self) -> dict[str, int]:
        counts = {"free": 0, "forced": 0}
        for spec in self._trial_schedule:
            mode = str(spec.get("trial_mode") or "free")
            counts[mode] = counts.get(mode, 0) + 1
        return counts

    def _session_metadata(self) -> dict:
        return {
            "config": config_to_dict(self.config),
            "trial_schedule": list(self._trial_schedule),
            "task_summary": task_design_summary(self.config),
            "streamdio_module_info": self._streamdio_module_info(),
        }

    def _streamdio_module_info(self) -> dict:
        if self.dio is None:
            return {}
        raw = getattr(self.dio, "raw", None)
        module_info = getattr(raw, "module_info", None)
        if not isinstance(module_info, dict):
            return {}
        return {
            str(key): value.item() if hasattr(value, "item") else value
            for key, value in module_info.items()
        }

    def _configure_monitor_record_path(self) -> None:
        record_path = self.config.monitor.record_path
        if record_path is None or str(record_path).lower() != "auto":
            return
        if self.recorder is None:
            return
        bundle_dir = session_bundle_dir(
            self.recorder.base_path,
            layout=self.config.session_export.layout,
        )
        exports_dir = bundle_dir / "exports"
        exports_dir.mkdir(parents=True, exist_ok=True)
        self.config.monitor.record_path = str(exports_dir / "monitor_telemetry.jsonl")

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
                "task_modality": self.config.task.modality,
                "task_profile": self.config.task.profile_name,
                "n_free_trials": self._n_trials_by_mode().get("free", 0),
                "n_forced_trials": self._n_trials_by_mode().get("forced", 0),
                "display_index": self.config.display.display_index,
                "camera_hz": self.config.camera.frequency_hz,
                "camera_backend": self.config.camera.backend,
                "force_channel": self.config.wheel.force_channel,
                "lick_channels": list(self.config.licks.channels),
                "lick_high_only": self.config.licks.high_only,
                "lick_inputs": self._lick_input_descriptors(),
                "frame2ttl": self._frame2ttl_descriptor(),
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
        child_status = monitor.child_status
        self.monitor = None
        if self.recorder is not None:
            self.recorder.log_event(
                "operator_monitor_stopped",
                {
                    "dropped_messages": int(dropped),
                    "dropped_frames": int(dropped_frames),
                    "published_counts": published_counts,
                    "child_status": child_status,
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

    def _start_streamdio_status_poll(self) -> None:
        if self.monitor is None or self.dio is None:
            return
        if (
            self._streamdio_status_poll_thread is not None
            and self._streamdio_status_poll_thread.is_alive()
        ):
            return

        cap_touch_enabled = CAP_TOUCH_INPUT_CHANNEL in {
            int(channel) for channel in self.config.licks.channels
        }
        cap_touch_hz = min(20.0, max(1.0, float(self.config.monitor.force_sample_hz)))
        poll_interval_s = 1.0 / cap_touch_hz if cap_touch_enabled else 1.0
        module_status_interval_s = max(
            1.0,
            float(self.config.monitor.streamdio_status_interval_s),
        )
        self._streamdio_status_poll_stop.clear()
        self._streamdio_module_status_snapshot_count = 0
        self._streamdio_module_status_error_count = 0
        self._latest_streamdio_module_status = None
        self._cap_touch_status_snapshot_count = 0
        self._latest_cap_touch_status = None

        def poll() -> None:
            next_module_status_mono = 0.0
            while not self._streamdio_status_poll_stop.wait(poll_interval_s):
                if self.dio is None:
                    return
                now = time.perf_counter()
                if now >= next_module_status_mono:
                    self._capture_streamdio_module_status(module_status_interval_s)
                    next_module_status_mono = now + module_status_interval_s
                if not cap_touch_enabled:
                    continue
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
                self._cap_touch_status_snapshot_count += 1
                self._latest_cap_touch_status = dict(status)
                channel = int(status.get("channel", CAP_TOUCH_INPUT_CHANNEL))
                self._publish_monitor_throttled(
                    "cap_touch_status",
                    cap_touch_hz,
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

        self._streamdio_status_poll_thread = threading.Thread(
            target=poll,
            name="hf-dm-streamdio-status",
            daemon=True,
        )
        self._streamdio_status_poll_thread.start()

    def _capture_streamdio_module_status(self, interval_s: float) -> None:
        if self.dio is None:
            return
        try:
            status = self.dio.get_module_status()
        except Exception as exc:
            self._streamdio_module_status_error_count += 1
            payload = {
                "source": "monitor_thread",
                "interval_s": float(interval_s),
                "error": f"{type(exc).__name__}: {exc}",
            }
            logger.warning("StreamDIO module status snapshot failed: %s", exc)
            if self.recorder is not None:
                self.recorder.log_debug_event(
                    "streamdio_module_status_error",
                    payload,
                )
        else:
            self._streamdio_module_status_snapshot_count += 1
            self._latest_streamdio_module_status = dict(status)
            payload = {
                "source": "monitor_thread",
                "interval_s": float(interval_s),
                "status": status,
            }
            if self.recorder is not None:
                self.recorder.log_debug_event(
                    "streamdio_module_status_snapshot",
                    payload,
                )
        self._publish_monitor("streamdio_status", payload)

    def _stop_streamdio_status_poll(self) -> None:
        self._streamdio_status_poll_stop.set()
        thread = self._streamdio_status_poll_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._streamdio_status_poll_thread = None
        self._update_streamdio_status_summary()

    def _update_streamdio_status_summary(self) -> None:
        if self.recorder is None:
            return
        if not (
            self._streamdio_module_status_snapshot_count
            or self._streamdio_module_status_error_count
            or self._cap_touch_status_snapshot_count
        ):
            return
        streamdio = self.recorder.get_stream_summaries().get("streamdio", {})
        status_poll = {
            "source": "monitor_thread",
            "module_status_interval_s": max(
                1.0,
                float(self.config.monitor.streamdio_status_interval_s),
            ),
            "module_status_snapshot_count": self._streamdio_module_status_snapshot_count,
            "module_status_error_count": self._streamdio_module_status_error_count,
            "cap_touch_status_snapshot_count": self._cap_touch_status_snapshot_count,
        }
        if self._latest_streamdio_module_status is not None:
            status_poll["latest_module_status"] = self._latest_streamdio_module_status
        if self._latest_cap_touch_status is not None:
            status_poll["latest_cap_touch_status"] = self._latest_cap_touch_status
        streamdio["status_poll"] = status_poll
        self.recorder.register_stream("streamdio", streamdio)

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

    def _frame2ttl_input_channels(self) -> tuple[int, ...]:
        cfg = self.config.frame2ttl
        if not cfg.enabled:
            return ()
        channel = int(cfg.input_channel)
        if channel <= 0:
            return ()
        return (channel,)

    def _frame2ttl_descriptor(self) -> dict:
        cfg = self.config.frame2ttl
        return {
            "enabled": bool(cfg.enabled),
            "input_channel": int(cfg.input_channel),
            "input_pin": int(cfg.input_pin),
            "patch_size_px": int(cfg.patch_size_px),
            "patch_margin_px": int(cfg.patch_margin_px),
            "patch_corner": str(cfg.patch_corner),
            "barcode_enabled": bool(cfg.barcode_enabled),
            "choice_flash_enabled": bool(cfg.choice_flash_enabled),
        }

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
                    "force_output_direction": runtime.force_direction,
                    "force_output_active": runtime.force_direction is not None,
                    "choice_target_direction": runtime.choice_target_direction,
                    "star_direction": runtime.star_direction,
                    "circle_direction": runtime.circle_direction,
                    "star_side": runtime.star_side,
                    "circle_side": runtime.circle_side,
                    "chosen_shape": runtime.chosen_shape,
                    "choice_side": runtime.choice_side,
                    "trial_mode": runtime.trial_mode,
                    "forced_shape": runtime.forced_shape,
                    "modality": runtime.modality,
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
                "trial_mode": trial.trial_mode,
                "forced_shape": trial.forced_shape,
                "modality": trial.modality,
                "matched": trial.matched,
                "responded": trial.responded,
                "reward_duration_s": trial.reward_duration_s,
                "reward_delay_s": trial.reward_delay_s,
                "reward_delivered": trial.reward_delivered,
                "response_time_s": trial.response_time_s,
                "final_position_deg": trial.final_position_deg,
                "star_side": trial.star_side,
                "circle_side": trial.circle_side,
                "star_direction": trial.star_direction,
                "circle_direction": trial.circle_direction,
                "choice_target_direction": trial.choice_target_direction,
                "force_output_direction": trial.force_output_direction,
                "force_output_active": trial.force_output_direction is not None,
                "lick_count": trial.lick_count,
                "frame2ttl_code_word": trial.frame2ttl_code_word,
                "frame2ttl_trial_type_code": trial.frame2ttl_trial_type_code,
                "frame2ttl_barcode_complete": trial.frame2ttl_barcode_complete,
                "frame2ttl_barcode_decoded": trial.frame2ttl_barcode_decoded,
                "frame2ttl_barcode_match": trial.frame2ttl_barcode_match,
                "frame2ttl_edge_count": trial.frame2ttl_edge_count,
                "frame2ttl_choice_edge_count": trial.frame2ttl_choice_edge_count,
                "frame2ttl_first_ttl_latency_from_bpod_s": (
                    trial.frame2ttl_first_ttl_latency_from_bpod_s
                ),
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
        event_host_mono_s = time.perf_counter()
        if self._is_frame2ttl_edge(event_type, normalized_channel):
            self._record_frame2ttl_edge(
                normalized_value,
                timestamp_us,
                relative_time,
                event_host_mono_s,
            )
        if event_type == "bpod_state" and self.recorder is not None:
            state_id = normalized_channel
            state_name = STREAMDIO_STATE_MARKER_NAMES.get(state_id, f"state_{state_id}")
            trial_index = (
                None
                if self._active_trial_runtime is None
                else self._active_trial_runtime.trial_index
            )
            payload = {
                "trial_index": trial_index,
                "state_id": state_id,
                "state_name": state_name,
                "module_timestamp_us": int(timestamp_us),
                "relative_time_s": float(relative_time),
                "host_perf_counter_s": float(event_host_mono_s),
            }
            self.recorder.log_event("bpod_state_marker", payload)
            self._publish_monitor("bpod_state_marker", payload)
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

    def _is_frame2ttl_edge(self, event_type: str, channel: int) -> bool:
        cfg = self.config.frame2ttl
        return (
            bool(cfg.enabled)
            and event_type == "dio_input"
            and int(channel) == int(cfg.input_channel)
        )

    def _record_frame2ttl_edge(
        self,
        state: int,
        timestamp_us: int,
        relative_time: float,
        host_mono_s: float,
    ) -> None:
        cfg = self.config.frame2ttl
        event = {
            "channel": int(cfg.input_channel),
            "pin": int(cfg.input_pin),
            "state": int(state),
            "timestamp_us": int(timestamp_us),
            "relative_time_s": float(relative_time),
            "host_perf_counter_s": float(host_mono_s),
            "wall_time_s": time.time(),
        }
        with self._frame2ttl_lock:
            self._frame2ttl_events.append(event)
        self._publish_monitor("frame2ttl_edge", event)

    def _frame2ttl_events_between(self, start_mono_s: float, end_mono_s: float) -> list[dict]:
        with self._frame2ttl_lock:
            return [
                dict(event)
                for event in self._frame2ttl_events
                if start_mono_s <= float(event["host_perf_counter_s"]) <= end_mono_s
            ]

    def _frame2ttl_event_count(self) -> int:
        with self._frame2ttl_lock:
            return len(self._frame2ttl_events)

    def _frame2ttl_events_since(self, start_index: int) -> list[dict]:
        with self._frame2ttl_lock:
            return [dict(event) for event in self._frame2ttl_events[start_index:]]

    def _notify_frame2ttl_display_state(
        self,
        state: bool,
        flip_mono_s: float,
        *,
        trial_index: int,
        source: str,
        frame_index: int,
    ) -> None:
        del state, flip_mono_s, trial_index, source, frame_index

    def _build_frame2ttl_barcode(self, runtime: TrialRuntime) -> tuple[dict, list[bool]]:
        cfg = self.config.frame2ttl
        code = encode_frame2ttl_trial_code(
            trial_index=runtime.trial_index,
            star_side=runtime.star_side,
            trial_mode=runtime.trial_mode,
            forced_shape=runtime.forced_shape,
            modality=runtime.modality,
        )
        states = frame2ttl_barcode_states(
            code,
            preamble_high_frames=cfg.barcode_preamble_high_frames,
            preamble_low_frames=cfg.barcode_preamble_low_frames,
            one_high_frames=cfg.barcode_one_high_frames,
            zero_high_frames=cfg.barcode_zero_high_frames,
            inter_bit_low_frames=cfg.barcode_inter_bit_low_frames,
        )
        payload = code.to_dict()
        payload["barcode_frame_count"] = len(states)
        return payload, states

    def _finalize_frame2ttl_trial(
        self,
        *,
        trial_index: int,
        code: dict | None,
        barcode_flips: list[dict],
        bpod_choice_start_mono_s: float | None,
        choice_end_mono_s: float | None,
    ) -> dict:
        cfg = self.config.frame2ttl
        result: dict = {
            "enabled": bool(cfg.enabled),
            "code_word": None if code is None else int(code["word"]),
            "trial_type_code": None if code is None else int(code["trial_type_code"]),
            "barcode_complete": False,
            "barcode_decoded": False,
            "barcode_match": False,
            "edge_count": 0,
            "choice_edge_count": 0,
            "first_ttl_timestamp_us": None,
            "first_ttl_relative_time_s": None,
            "first_ttl_latency_from_bpod_s": None,
        }
        if not cfg.enabled:
            return result
        if bpod_choice_start_mono_s is not None and choice_end_mono_s is not None:
            result["choice_edge_count"] = len(
                self._frame2ttl_events_between(
                    float(bpod_choice_start_mono_s),
                    float(choice_end_mono_s) + 0.05,
                )
            )
        if code is None or not barcode_flips:
            self._publish_monitor("frame2ttl_decode", {"trial_index": trial_index, **result})
            return result

        complete = len(barcode_flips) >= int(code.get("barcode_frame_count", len(barcode_flips)))
        result["barcode_complete"] = complete
        start = float(barcode_flips[0]["display_flip_mono_s"]) - 0.05
        end = float(barcode_flips[-1]["display_flip_mono_s"]) + 0.25
        edges = self._frame2ttl_events_between(start, end)
        result["edge_count"] = len(edges)
        if edges:
            first = edges[0]
            result["first_ttl_timestamp_us"] = int(first["timestamp_us"])
            result["first_ttl_relative_time_s"] = float(first["relative_time_s"])
            bpod_start = barcode_flips[0].get("bpod_state_started_mono_s")
            if bpod_start is not None:
                result["first_ttl_latency_from_bpod_s"] = (
                    float(first["host_perf_counter_s"]) - float(bpod_start)
                )
        decoded = decode_frame2ttl_barcode_events(
            edges,
            frame_period_s=1.0 / max(1, int(self.config.display.fps)),
            expected_bits=FRAME2TTL_TOTAL_BITS,
            preamble_high_frames=cfg.barcode_preamble_high_frames,
            preamble_low_frames=cfg.barcode_preamble_low_frames,
            one_high_frames=cfg.barcode_one_high_frames,
            zero_high_frames=cfg.barcode_zero_high_frames,
            inter_bit_low_frames=cfg.barcode_inter_bit_low_frames,
        )
        result["decode"] = decoded
        result["barcode_decoded"] = bool(decoded.get("decoded"))
        result["barcode_match"] = (
            bool(decoded.get("decoded"))
            and int(decoded.get("word", -1)) == int(code["word"])
            and bool(decoded.get("checksum_ok", False))
        )
        payload = {
            "trial_index": trial_index,
            "expected": code,
            **result,
        }
        if self.recorder is not None:
            self.recorder.log_event("frame2ttl_barcode_decode", payload)
        self._publish_monitor("frame2ttl_decode", payload)
        return result

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
            "sound_status": self._sound_status(),
        }

    def _sound_status(self) -> dict:
        if self.white_noise_player is None:
            return {}
        return self.white_noise_player.status

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
                roi_width=camera_cfg.roi_width,
                roi_height=camera_cfg.roi_height,
                roi_offset_x=camera_cfg.roi_offset_x,
                roi_offset_y=camera_cfg.roi_offset_y,
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
            self.recorder.register_stream(
                "spinnaker_video",
                {
                    "type": "spinnaker",
                    "backend": recorder.backend,
                    "save_dir": str(session_dir),
                    "target_trigger_hz": camera_cfg.frequency_hz,
                    "exposure_us": camera_cfg.exposure_us,
                    "pixel_format": camera_cfg.pixel_format,
                    "roi": {
                        "width": camera_cfg.roi_width,
                        "height": camera_cfg.roi_height,
                        "offset_x": camera_cfg.roi_offset_x,
                        "offset_y": camera_cfg.roi_offset_y,
                    },
                },
            )
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
            self.recorder.register_stream(
                "spinnaker_video",
                {
                    "type": "spinnaker",
                    **summary,
                },
            )
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
                    "frame2ttl_input_edges_seen": digital_counts.get(
                        int(self.config.frame2ttl.input_channel),
                        0,
                    )
                    if self.config.frame2ttl.enabled
                    else 0,
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

    def frame2ttl_diagnostic(
        self,
        *,
        n_flips: int = 20,
        hold_s: float = 0.25,
        settle_s: float = 0.5,
    ) -> SessionSummary:
        """Flash the configured sync square and count Frame2TTL DI transitions."""
        assert self.display is not None and self.recorder is not None
        n_flips = max(1, int(n_flips))
        hold_s = max(0.01, float(hold_s))
        settle_s = max(0.0, float(settle_s))
        start_idx = self._frame2ttl_event_count()
        flip_log: list[dict] = []
        self.recorder.log_event(
            "frame2ttl_diagnostic_start",
            {
                "n_flips": n_flips,
                "hold_s": hold_s,
                "settle_s": settle_s,
                "frame2ttl": self._frame2ttl_descriptor(),
            },
        )
        end_reason = "frame2ttl_diagnostic"
        try:
            start_flip = self.display.draw_sync_square_frame(False)
            self._notify_frame2ttl_display_state(
                False,
                start_flip,
                trial_index=0,
                source="diagnostic",
                frame_index=-1,
            )
            self._publish_task_screen(force=True)
            self._wait_display(settle_s)
            for index in range(n_flips):
                state = index % 2 == 0
                flip_mono_s = self.display.draw_sync_square_frame(state)
                self._notify_frame2ttl_display_state(
                    state,
                    flip_mono_s,
                    trial_index=0,
                    source="diagnostic",
                    frame_index=index,
                )
                row = {
                    "flip_index": index + 1,
                    "state": int(state),
                    "display_flip_mono_s": flip_mono_s,
                }
                flip_log.append(row)
                self._publish_monitor(
                    "frame2ttl_diagnostic_flip",
                    {"n_flips": n_flips, **row},
                )
                self._publish_task_screen()
                self._wait_display(hold_s)
            final_flip = self.display.draw_sync_square_frame(False)
            self._notify_frame2ttl_display_state(
                False,
                final_flip,
                trial_index=0,
                source="diagnostic",
                frame_index=n_flips,
            )
            self._publish_task_screen(force=True)
            self._wait_display(settle_s)
            edges = self._frame2ttl_events_since(start_idx)
            states_seen = sorted({int(event["state"]) for event in edges})
            result = {
                "n_flips": n_flips,
                "hold_s": hold_s,
                "settle_s": settle_s,
                "edges_seen": len(edges),
                "states_seen": states_seen,
                "detected": len(edges) > 0 and len(states_seen) >= 2,
                "first_ttl_timestamp_us": (
                    None if not edges else int(edges[0]["timestamp_us"])
                ),
                "first_ttl_relative_time_s": (
                    None if not edges else float(edges[0]["relative_time_s"])
                ),
                "first_ttl_host_perf_counter_s": (
                    None if not edges else float(edges[0]["host_perf_counter_s"])
                ),
                "flips": flip_log,
            }
            self.recorder.log_event("frame2ttl_diagnostic_end", result)
            self._publish_monitor("frame2ttl_decode", {"trial_index": 0, **result})
        except BaseException as exc:
            end_reason = (
                "keyboard_interrupt"
                if isinstance(exc, KeyboardInterrupt)
                else "frame2ttl_diagnostic_error"
            )
            self.recorder.log_event(
                "frame2ttl_diagnostic_error",
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
            logger.info("Frame2TTL diagnostic data saved to %s", path)
        return summary

    def frame2ttl_barcode_diagnostic(
        self,
        *,
        repetitions: int = 1,
        settle_s: float = 0.25,
    ) -> SessionSummary:
        """Display representative trial-start barcodes and decode DI transitions."""
        assert self.display is not None and self.recorder is not None
        cfg = self.config.frame2ttl
        repetitions = max(1, int(repetitions))
        settle_s = max(0.0, float(settle_s))
        cases = self._frame2ttl_barcode_diagnostic_cases()
        case_results: list[dict] = []
        self.recorder.log_event(
            "frame2ttl_barcode_diagnostic_start",
            {
                "repetitions": repetitions,
                "settle_s": settle_s,
                "cases": cases,
                "frame2ttl": self._frame2ttl_descriptor(),
            },
        )
        end_reason = "frame2ttl_barcode_diagnostic"
        try:
            initial_flip = self.display.draw_sync_square_frame(False)
            self._notify_frame2ttl_display_state(
                False,
                initial_flip,
                trial_index=0,
                source="barcode_diagnostic",
                frame_index=-1,
            )
            self._publish_task_screen(force=True)
            self._wait_display(settle_s)
            for repeat_index in range(repetitions):
                for case_index, case in enumerate(cases, start=1):
                    trial_index = repeat_index * len(cases) + case_index
                    code = encode_frame2ttl_trial_code(
                        trial_index=trial_index,
                        star_side=case["star_side"],
                        trial_mode=case["trial_mode"],
                        forced_shape=case["forced_shape"],
                        modality=case["modality"],
                    )
                    states = frame2ttl_barcode_states(
                        code,
                        preamble_high_frames=cfg.barcode_preamble_high_frames,
                        preamble_low_frames=cfg.barcode_preamble_low_frames,
                        one_high_frames=cfg.barcode_one_high_frames,
                        zero_high_frames=cfg.barcode_zero_high_frames,
                        inter_bit_low_frames=cfg.barcode_inter_bit_low_frames,
                    )
                    start_idx = self._frame2ttl_event_count()
                    started_mono_s = time.perf_counter()
                    start_payload = {
                        "trial_index": trial_index,
                        "repeat_index": repeat_index + 1,
                        "case_index": case_index,
                        "expected": code.to_dict(),
                        "frame_count": len(states),
                    }
                    self.recorder.log_event(
                        "frame2ttl_barcode_diagnostic_case_start",
                        start_payload,
                    )
                    self._publish_monitor("frame2ttl_barcode", start_payload)
                    for frame_index, state in enumerate(states):
                        flip_mono_s = self.display.draw_sync_square_frame(bool(state))
                        self._notify_frame2ttl_display_state(
                            bool(state),
                            flip_mono_s,
                            trial_index=trial_index,
                            source="barcode_diagnostic",
                            frame_index=frame_index,
                        )
                        self.display.clock.tick(max(1, int(self.config.display.fps)))
                    final_flip = self.display.draw_sync_square_frame(False)
                    self._notify_frame2ttl_display_state(
                        False,
                        final_flip,
                        trial_index=trial_index,
                        source="barcode_diagnostic",
                        frame_index=len(states),
                    )
                    self._publish_task_screen(force=True)
                    self._wait_display(settle_s)
                    edges = self._frame2ttl_events_since(start_idx)
                    decoded = decode_frame2ttl_barcode_events(
                        edges,
                        frame_period_s=1.0 / max(1, int(self.config.display.fps)),
                        expected_bits=FRAME2TTL_TOTAL_BITS,
                        preamble_high_frames=cfg.barcode_preamble_high_frames,
                        preamble_low_frames=cfg.barcode_preamble_low_frames,
                        one_high_frames=cfg.barcode_one_high_frames,
                        zero_high_frames=cfg.barcode_zero_high_frames,
                        inter_bit_low_frames=cfg.barcode_inter_bit_low_frames,
                    )
                    match = (
                        bool(decoded.get("decoded"))
                        and int(decoded.get("word", -1)) == int(code.word)
                        and bool(decoded.get("checksum_ok", False))
                    )
                    case_result = {
                        **start_payload,
                        "edges_seen": len(edges),
                        "states_seen": sorted({int(event["state"]) for event in edges}),
                        "started_mono_s": started_mono_s,
                        "ended_mono_s": time.perf_counter(),
                        "decode": decoded,
                        "barcode_decoded": bool(decoded.get("decoded")),
                        "barcode_match": match,
                    }
                    case_results.append(case_result)
                    self.recorder.log_event(
                        "frame2ttl_barcode_diagnostic_case_end",
                        case_result,
                    )
                    self._publish_monitor(
                        "frame2ttl_decode",
                        {
                            "trial_index": trial_index,
                            "code_word": int(code.word),
                            **case_result,
                        },
                    )
            result = {
                "repetitions": repetitions,
                "cases_run": len(case_results),
                "decoded_count": sum(
                    1 for item in case_results if item["barcode_decoded"]
                ),
                "match_count": sum(1 for item in case_results if item["barcode_match"]),
                "all_match": bool(case_results)
                and all(item["barcode_match"] for item in case_results),
                "case_results": case_results,
            }
            self.recorder.log_event("frame2ttl_barcode_diagnostic_end", result)
        except BaseException as exc:
            end_reason = (
                "keyboard_interrupt"
                if isinstance(exc, KeyboardInterrupt)
                else "frame2ttl_barcode_diagnostic_error"
            )
            self.recorder.log_event(
                "frame2ttl_barcode_diagnostic_error",
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
            logger.info("Frame2TTL barcode diagnostic data saved to %s", path)
        return summary

    def bpod_opcode_diagnostic(self, *, hold_s: float = 0.01) -> SessionSummary:
        """Send StreamDIO state-marker opcodes from Bpod without task outputs."""
        assert self.bpod is not None and self.speaker_relay is not None
        assert self.recorder is not None
        hold_s = max(0.001, float(hold_s))
        sequence = [
            "Quiescence",
            "ChoiceWindow",
            "ChoiceBlankHigh",
            "RewardHigh",
            "PostRewardHigh",
            "TimeoutNoise",
            "PostTimeout",
            "Abort",
        ]
        expected_ids = [STREAMDIO_STATE_MARKER_IDS[name] for name in sequence]
        start_streamdio_idx = len(self.recorder.streamdio_events)
        self.recorder.log_event(
            "bpod_opcode_diagnostic_start",
            {"hold_s": hold_s, "sequence": sequence, "expected_ids": expected_ids},
        )
        end_reason = "bpod_opcode_diagnostic"
        try:
            fsm = StateMachine(name="BpodOpcodeDiagnostic")
            for index, state_name in enumerate(sequence):
                next_state = (
                    f"{sequence[index + 1]}_Opcode"
                    if index + 1 < len(sequence)
                    else ">exit"
                )
                fsm.add_state(
                    name=f"{state_name}_Opcode",
                    timer=hold_s,
                    transitions={"Tup": next_state},
                    actions=self.speaker_relay.bpod_state_marker_act(state_name),
                )
            self.recorder.log_fsm_definition("bpod_opcode_diagnostic", fsm)
            self._prepare_bpod_run(0)
            self.bpod.run(fsm, trial_number=0)
            while self.bpod.is_running:
                if self.display is not None:
                    self.display.pump_events()
                time.sleep(0.002)
            self._wait_for_bpod_trial_threads()
            self.recorder.log_bpod_trial(0, self.bpod.get_data(concat=False))
            events = [
                event
                for event in self.recorder.streamdio_events[start_streamdio_idx:]
                if event[0] == "bpod_state"
            ]
            seen_ids = [int(event[1]) for event in events]
            result = {
                "hold_s": hold_s,
                "sequence": sequence,
                "expected_ids": expected_ids,
                "seen_ids": seen_ids,
                "markers_seen": len(seen_ids),
                "matched": seen_ids == expected_ids,
                "streamdio_events": events,
            }
            self.recorder.log_event("bpod_opcode_diagnostic_end", result)
            self._publish_monitor("bpod_opcode_diagnostic", result)
        except BaseException as exc:
            end_reason = (
                "keyboard_interrupt"
                if isinstance(exc, KeyboardInterrupt)
                else "bpod_opcode_diagnostic_error"
            )
            self.recorder.log_event(
                "bpod_opcode_diagnostic_error",
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
            logger.info("Bpod opcode diagnostic data saved to %s", path)
        return summary

    @staticmethod
    def _frame2ttl_barcode_diagnostic_cases() -> list[dict[str, str | None]]:
        return [
            {
                "label": "force_free_left",
                "modality": "force",
                "trial_mode": "free",
                "forced_shape": None,
                "star_side": "left",
            },
            {
                "label": "force_free_right",
                "modality": "force",
                "trial_mode": "free",
                "forced_shape": None,
                "star_side": "right",
            },
            {
                "label": "risk_forced_star_left",
                "modality": "risk",
                "trial_mode": "forced",
                "forced_shape": "star",
                "star_side": "left",
            },
            {
                "label": "risk_forced_star_right",
                "modality": "risk",
                "trial_mode": "forced",
                "forced_shape": "star",
                "star_side": "right",
            },
            {
                "label": "delay_forced_circle_left",
                "modality": "delay",
                "trial_mode": "forced",
                "forced_shape": "circle",
                "star_side": "left",
            },
            {
                "label": "delay_forced_circle_right",
                "modality": "delay",
                "trial_mode": "forced",
                "forced_shape": "circle",
                "star_side": "right",
            },
        ]

    def _wait_display(self, duration_s: float) -> None:
        if self.display is None:
            time.sleep(max(0.0, float(duration_s)))
            return
        deadline = time.perf_counter() + max(0.0, float(duration_s))
        while time.perf_counter() < deadline:
            self.display.pump_events()
            self.display.clock.tick(self.config.display.fps)

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
    def _lick_rate_hz(lick_count: int, duration_s: float) -> float:
        duration = max(0.0, float(duration_s))
        if duration <= 0.0:
            return 0.0
        return float(max(0, int(lick_count))) / duration

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

    def _prepare_bpod_run(self, trial_number: int) -> None:
        if self.speaker_relay is None:
            return
        prepare = getattr(self.speaker_relay, "prepare_bpod_run", None)
        if not callable(prepare):
            return
        discarded = int(prepare())
        if discarded > 0 and self.recorder is not None:
            self.recorder.log_event(
                "bpod_serial_input_discarded",
                {
                    "trial_number": int(trial_number),
                    "byte_count": discarded,
                    "reason": "idle_module_relay_bytes_before_fsm",
                },
            )

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
                if runtime.force_direction is None:
                    transition = self.relay.force_off()
                    self.recorder.log_event(
                        "force_skipped",
                        {
                            "trial_index": runtime.trial_index,
                            "modality": runtime.modality,
                            "trial_mode": runtime.trial_mode,
                            "forced_shape": runtime.forced_shape,
                            "choice_target_direction": runtime.choice_target_direction,
                            "reason": self._force_skip_reason(runtime),
                            "transitions": transition,
                            "actual_off_mono_s": self._actual_transition_time(transition),
                            "trigger": "bpod_state_entry",
                        },
                    )
                else:
                    transition = self.relay.set_force_direction(runtime.force_direction)
                    actual_on = self._actual_transition_time(transition)
                    self.recorder.log_event(
                        "force_on",
                        {
                            "trial_index": runtime.trial_index,
                            "force_direction": runtime.force_direction,
                            "force_output_direction": runtime.force_direction,
                            "choice_target_direction": runtime.choice_target_direction,
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
                reward_s = self._runtime_reward_duration(runtime, runtime.chosen_shape)
                if reward_s <= 0.0:
                    self.recorder.log_event(
                        "pump_omitted",
                        {
                            "trial_index": runtime.trial_index,
                            "chosen_shape": runtime.chosen_shape,
                            "modality": runtime.modality,
                            "reason": "risk_no_reward",
                            "trigger": "bpod_state_entry",
                        },
                    )
                    self._publish_monitor(
                        "pump",
                        {
                            "active": False,
                            "trial_index": runtime.trial_index,
                            "duration_s": 0.0,
                            "source": "risk_no_reward",
                        },
                    )
                    self._publish_fsm_state(
                        "RewardHigh" if runtime.chosen_shape == "star" else "RewardLow",
                        runtime=runtime,
                        phase="reward",
                    )
                    runtime.enter(
                        "reward",
                        softcode=softcode,
                        output_result={"pump_skipped": True, "reason": "risk_no_reward"},
                    )
                    return
                transition = self.relay.pump_on()
                actual_on = self._actual_transition_time(transition)
                runtime.pump_actual_on_mono_s = actual_on
                self.recorder.log_event(
                    "pump_start",
                    {
                        "trial_index": runtime.trial_index,
                        "duration_s": reward_s,
                        "chosen_shape": runtime.chosen_shape,
                        "modality": runtime.modality,
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
                        "chosen_shape": runtime.chosen_shape,
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
                        "chosen_shape": runtime.chosen_shape,
                        "modality": runtime.modality,
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
                        "sound_status": self._sound_status(),
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
                        "sound_status": self._sound_status(),
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

    @staticmethod
    def _force_skip_reason(runtime: TrialRuntime) -> str:
        if runtime.modality != "force":
            return f"{runtime.modality}_modality_no_physical_force"
        if runtime.trial_mode == "forced" and runtime.forced_shape == "circle":
            return "forced_circle_baseline"
        return "no_star_force_effect"

    def _choose_star_side(self, trial_index: int) -> Side:
        del trial_index
        return self.rng.choice(["left", "right"])

    def _reward_plan_for_trial(self) -> dict[str, float | bool | str | None]:
        modality = str(self.config.task.modality or "force").lower()
        if modality not in {"force", "risk", "delay"}:
            modality = "force"
        star_delivered = True
        risk_probability: float | None = None
        if modality == "risk":
            risk_probability = max(0.0, min(1.0, float(self.config.task.risk_star_probability)))
            star_delivered = self.rng.random() < risk_probability
        high_reward_s = float(self.config.reward.matched_pump_s) if star_delivered else 0.0
        low_reward_s = float(self.config.reward.opposite_pump_s)
        low_delay_s = float(self.config.timing.choice_blank_s)
        high_delay_s = low_delay_s
        if modality == "delay":
            high_delay_s += max(0.0, float(self.config.task.delay_star_extra_s))
        return {
            "modality": modality,
            "high_reward_s": high_reward_s,
            "low_reward_s": low_reward_s,
            "high_delay_s": high_delay_s,
            "low_delay_s": low_delay_s,
            "star_reward_delivered": bool(star_delivered),
            "risk_star_probability": risk_probability,
        }

    @staticmethod
    def _reward_state_timer(reward_s: float) -> float:
        return max(0.001, float(reward_s))

    @staticmethod
    def _runtime_reward_duration(runtime: TrialRuntime, shape: Shape | None) -> float:
        if shape == "star":
            return float(runtime.high_reward_s)
        if shape == "circle":
            return float(runtime.low_reward_s)
        return 0.0

    @staticmethod
    def _runtime_reward_delay(runtime: TrialRuntime, shape: Shape | None) -> float:
        if shape == "star":
            return float(runtime.high_delay_s)
        if shape == "circle":
            return float(runtime.low_delay_s)
        return 0.0

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
        trial_mode: str = "free",
        forced_shape: Shape | None = None,
    ) -> dict:
        assert self.bpod is not None and self.display is not None and self.recorder is not None
        assert self.speaker_relay is not None

        circle_side = self._opposite_side(star_side)
        reward_plan = self._reward_plan_for_trial()
        star_direction = self._wheel_direction_for_side(star_side)
        circle_direction = self._wheel_direction_for_side(circle_side)
        choice_target_direction = (
            circle_direction if trial_mode == "forced" and forced_shape == "circle" else star_direction
        )
        force_output_direction: Direction | None = (
            star_direction
            if reward_plan["modality"] == "force"
            and not (trial_mode == "forced" and forced_shape == "circle")
            else None
        )
        runtime = TrialRuntime(
            trial_index=trial_index,
            force_direction=force_output_direction,
            choice_target_direction=choice_target_direction,
            star_direction=star_direction,
            circle_direction=circle_direction,
            star_side=star_side,
            circle_side=circle_side,
            trial_mode=str(trial_mode),
            forced_shape=forced_shape,
            modality=str(reward_plan["modality"]),
            high_reward_s=float(reward_plan["high_reward_s"]),
            low_reward_s=float(reward_plan["low_reward_s"]),
            high_delay_s=float(reward_plan["high_delay_s"]),
            low_delay_s=float(reward_plan["low_delay_s"]),
            star_reward_delivered=bool(reward_plan["star_reward_delivered"]),
            risk_star_probability=(
                None
                if reward_plan["risk_star_probability"] is None
                else float(reward_plan["risk_star_probability"])
            ),
        )
        self._active_trial_runtime = runtime
        self._prepare_decision_trial(runtime)
        fsm = build_host_supervised_decision_fsm(
            self.speaker_relay,
            quiescence_s=self.config.timing.quiescence_s,
            choice_timeout_s=self.config.timing.choice_timeout_s,
            choice_blank_s=self.config.timing.choice_blank_s,
            high_reward_s=self._reward_state_timer(runtime.high_reward_s),
            low_reward_s=self._reward_state_timer(runtime.low_reward_s),
            post_reward_s=self.config.timing.post_reward_s,
            timeout_noise_s=self.config.timing.timeout_noise_s,
            post_timeout_s=self.config.timing.post_timeout_s,
            high_delay_s=runtime.high_delay_s,
            low_delay_s=runtime.low_delay_s,
        )
        self.recorder.log_fsm_definition(f"host_supervised_decision_trial_{trial_index}", fsm)
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
        last_drawn_choice_signature: tuple[float, float, str | None] | None = None
        choice_sync_square_state = False
        timeout_logged = False
        abort_sent = False
        last_phase = "starting"
        last_quiescence_edges = 0
        frame2ttl_code: dict | None = None
        frame2ttl_barcode_states: list[bool] = []
        frame2ttl_barcode_flips: list[dict] = []
        frame2ttl_barcode_index = 0
        frame2ttl_barcode_started = False
        frame2ttl_barcode_incomplete_logged = False
        if self.config.frame2ttl.enabled and self.config.frame2ttl.barcode_enabled:
            frame2ttl_code, frame2ttl_barcode_states = self._build_frame2ttl_barcode(runtime)
        stall_threshold_s = max(
            float(self.config.timing.choice_loop_stall_s),
            (1.0 / max(1, self.config.display.fps)) * 3.0,
        )

        self._prepare_bpod_run(trial_index)
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
                    if frame2ttl_barcode_index < len(frame2ttl_barcode_states):
                        patch_state = frame2ttl_barcode_states[frame2ttl_barcode_index]
                        flip_mono_s = self.display.draw_sync_square_frame(patch_state)
                        self._notify_frame2ttl_display_state(
                            patch_state,
                            flip_mono_s,
                            trial_index=trial_index,
                            source="barcode",
                            frame_index=frame2ttl_barcode_index,
                        )
                        frame2ttl_barcode_flips.append(
                            {
                                "frame_index": frame2ttl_barcode_index,
                                "state": int(patch_state),
                                "display_flip_mono_s": flip_mono_s,
                                "bpod_state_started_mono_s": state["phase_started_mono_s"],
                            }
                        )
                        if not frame2ttl_barcode_started:
                            frame2ttl_barcode_started = True
                            payload = {
                                "trial_index": trial_index,
                                "expected": frame2ttl_code,
                                "frame_count": len(frame2ttl_barcode_states),
                                "first_display_flip_mono_s": flip_mono_s,
                                "bpod_state_started_mono_s": state["phase_started_mono_s"],
                            }
                            self.recorder.log_event("frame2ttl_barcode_start", payload)
                            self._publish_monitor("frame2ttl_barcode", payload)
                        frame2ttl_barcode_index += 1
                        if frame2ttl_barcode_index >= len(frame2ttl_barcode_states):
                            payload = {
                                "trial_index": trial_index,
                                "expected": frame2ttl_code,
                                "frame_count": len(frame2ttl_barcode_states),
                                "last_display_flip_mono_s": flip_mono_s,
                                "complete": True,
                            }
                            self.recorder.log_event("frame2ttl_barcode_end", payload)
                            self._publish_monitor("frame2ttl_barcode", payload)
                        self._publish_monitor_throttled(
                            "task_state",
                            min(30.0, float(self.config.display.fps)),
                            "task_state",
                            {
                                "trial_index": trial_index,
                                "phase": phase,
                                "frame2ttl_barcode_frame": frame2ttl_barcode_index,
                                "frame2ttl_barcode_frames": len(frame2ttl_barcode_states),
                            },
                        )
                        self._publish_task_screen()
                    elif last_phase != phase:
                        flip_mono_s = self.display.fill_black()
                        self._notify_frame2ttl_display_state(
                            False,
                            flip_mono_s,
                            trial_index=trial_index,
                            source="quiescence_black",
                            frame_index=frame2ttl_barcode_index,
                        )
                        self._publish_monitor(
                            "task_state",
                            {"trial_index": trial_index, "phase": phase},
                        )
                        self._publish_task_screen()
                    else:
                        self.display.pump_events()
                    self._publish_wheel_snapshot(phase=phase, trial_index=trial_index)

                elif phase == "choice":
                    if (
                        frame2ttl_barcode_states
                        and frame2ttl_barcode_index < len(frame2ttl_barcode_states)
                        and not frame2ttl_barcode_incomplete_logged
                    ):
                        frame2ttl_barcode_incomplete_logged = True
                        payload = {
                            "trial_index": trial_index,
                            "expected": frame2ttl_code,
                            "frames_drawn": frame2ttl_barcode_index,
                            "frame_count": len(frame2ttl_barcode_states),
                            "complete": False,
                            "reason": "choice_window_started",
                        }
                        self.recorder.log_event("frame2ttl_barcode_incomplete", payload)
                        self._publish_monitor("frame2ttl_barcode", payload)
                    if choice_start_mono_s is None:
                        choice_start_mono_s = float(state["phase_started_mono_s"])
                    self._track_engagement_from_wheel()
                    wheel_snap = self.wheel.snapshot()
                    net_position_deg = (
                        int(wheel_snap["net_pulses"]) * self.config.wheel.deg_per_pulse
                    )
                    final_left_deg, final_right_deg = self._decision_positions(net_position_deg)
                    visible_shape = forced_shape if trial_mode == "forced" else None
                    draw_signature = (
                        round(float(final_left_deg), 6),
                        round(float(final_right_deg), 6),
                        visible_shape,
                    )
                    should_draw = (not stimulus_visible) or (
                        draw_signature != last_drawn_choice_signature
                    )
                    if should_draw:
                        frame_mono_s = time.perf_counter()
                        if previous_frame_mono_s is not None:
                            interval_s = frame_mono_s - previous_frame_mono_s
                            frame_interval_sum_s += interval_s
                            frame_interval_max_s = max(frame_interval_max_s, interval_s)
                            if interval_s >= stall_threshold_s:
                                stall_count += 1
                                self.recorder.log_debug_event(
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
                        sync_square_state = None
                        if (
                            self.config.frame2ttl.enabled
                            and self.config.frame2ttl.choice_flash_enabled
                        ):
                            if stimulus_visible:
                                choice_sync_square_state = not choice_sync_square_state
                            sync_square_state = choice_sync_square_state
                        flip_mono_s = self.display.draw_decision(
                            star_side=star_side,
                            left_position_deg=final_left_deg,
                            right_position_deg=final_right_deg,
                            force_direction=force_output_direction,
                            visible_shape=visible_shape,
                            trial_index=trial_index,
                            sync_square_state=sync_square_state,
                        )
                        last_drawn_choice_signature = draw_signature
                        if sync_square_state is not None:
                            self._notify_frame2ttl_display_state(
                                sync_square_state,
                                flip_mono_s,
                                trial_index=trial_index,
                                source="choice",
                                frame_index=frame_count,
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
                                "force_direction": force_output_direction,
                                "force_output_direction": force_output_direction,
                                "force_output_active": force_output_direction is not None,
                                "choice_target_direction": choice_target_direction,
                                "star_direction": star_direction,
                                "circle_direction": circle_direction,
                                "trial_mode": trial_mode,
                                "forced_shape": forced_shape,
                                "modality": runtime.modality,
                                "left_position_deg": final_left_deg,
                                "right_position_deg": final_right_deg,
                                "net_position_deg": net_position_deg,
                                "frame2ttl_sync_state": sync_square_state,
                                "frame2ttl_choice_frame": frame_count,
                                "display_update_reason": (
                                    "stimulus_on" if not stimulus_visible else "wheel_motion"
                                ),
                            },
                        )
                        self._publish_task_screen()
                        if not stimulus_visible:
                            stimulus_visible = True
                            self.recorder.log_event(
                                "stimulus_on",
                                {
                                    "trial_index": trial_index,
                                    "star_side": star_side,
                                    "circle_side": circle_side,
                                    "trial_mode": trial_mode,
                                    "forced_shape": forced_shape,
                                    "modality": runtime.modality,
                                    "force_direction": force_output_direction,
                                    "force_output_direction": force_output_direction,
                                    "force_output_active": force_output_direction is not None,
                                    "choice_target_direction": choice_target_direction,
                                    "star_direction": star_direction,
                                    "circle_direction": circle_direction,
                                    "left_position_deg": final_left_deg,
                                    "right_position_deg": final_right_deg,
                                    "display_flip_mono_s": flip_mono_s,
                                    "bright_pixel_counts": self.display.shape_bright_pixel_counts,
                                },
                            )
                    else:
                        self.display.pump_events()
                    self._publish_wheel_snapshot(phase=phase, trial_index=trial_index)

                    threshold = self.config.display.choice_center_threshold_deg
                    selected_side: Side | None = None
                    if final_left_deg >= -threshold:
                        selected_side = "left"
                    elif final_right_deg <= threshold:
                        selected_side = "right"
                    if selected_side is not None and trial_mode == "forced":
                        forced_side = star_side if forced_shape == "star" else circle_side
                        if selected_side != forced_side:
                            selected_side = None

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
                            reward_s = self._runtime_reward_duration(runtime, chosen_shape)
                            reward_delay_s = self._runtime_reward_delay(runtime, chosen_shape)
                            self.recorder.log_event(
                                "choice_detected",
                                {
                                    "trial_index": trial_index,
                                    "choice": wheel_choice,
                                    "choice_side": selected_side,
                                    "chosen_shape": chosen_shape,
                                    "trial_mode": trial_mode,
                                    "forced_shape": forced_shape,
                                    "modality": runtime.modality,
                                    "force_direction": force_output_direction,
                                    "force_output_direction": force_output_direction,
                                    "force_output_active": force_output_direction is not None,
                                    "choice_target_direction": choice_target_direction,
                                    "star_direction": star_direction,
                                    "circle_direction": circle_direction,
                                    "matched": chosen_shape == "star",
                                    "response_time_s": response_time_s,
                                    "reward_duration_s": reward_s,
                                    "reward_delay_s": reward_delay_s,
                                    "reward_delivered": reward_s > 0.0,
                                    "risk_star_probability": runtime.risk_star_probability,
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
                                    "trial_mode": trial_mode,
                                    "forced_shape": forced_shape,
                                    "modality": runtime.modality,
                                    "matched": chosen_shape == "star",
                                    "response_time_s": response_time_s,
                                    "reward_duration_s": reward_s,
                                    "reward_delay_s": reward_delay_s,
                                    "reward_delivered": reward_s > 0.0,
                                },
                            )
                            self.bpod.send_softcode(
                                sc.CHOICE_HIGH if chosen_shape == "star" else sc.CHOICE_LOW
                            )

                else:
                    if stimulus_visible:
                        blank_flip_mono_s = self.display.fill_black()
                        self._notify_frame2ttl_display_state(
                            False,
                            blank_flip_mono_s,
                            trial_index=trial_index,
                            source="stimulus_off",
                            frame_index=frame_count,
                        )
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
                                "force_direction": force_output_direction,
                                "force_output_direction": force_output_direction,
                                "force_output_active": force_output_direction is not None,
                                "choice_target_direction": choice_target_direction,
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
        reward_s = self._runtime_reward_duration(runtime, chosen_shape)
        reward_delay_s = self._runtime_reward_delay(runtime, chosen_shape)
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
        frame2ttl_summary = self._finalize_frame2ttl_trial(
            trial_index=trial_index,
            code=frame2ttl_code,
            barcode_flips=frame2ttl_barcode_flips,
            bpod_choice_start_mono_s=choice_start_mono_s,
            choice_end_mono_s=choice_end_mono_s,
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
            "force_direction": force_output_direction,
            "force_output_direction": force_output_direction,
            "choice_target_direction": choice_target_direction,
            "star_direction": star_direction,
            "circle_direction": circle_direction,
            "star_side": star_side,
            "circle_side": circle_side,
            "trial_mode": trial_mode,
            "forced_shape": forced_shape,
            "modality": runtime.modality,
            "choice": wheel_choice,
            "choice_side": choice_side,
            "chosen_shape": chosen_shape,
            "matched": chosen_shape == "star",
            "responded": responded,
            "response_time_s": response_time_s,
            "reward_duration_s": reward_s,
            "reward_delay_s": reward_delay_s,
            "reward_delivered": reward_s > 0.0,
            "risk_star_probability": runtime.risk_star_probability,
            "final_position_deg": final_position_deg,
            "runtime": state,
            "frame2ttl": frame2ttl_summary,
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
                    self.recorder.log_debug_event(
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
                        "sound_status": self._sound_status(),
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
        frame2ttl_start_idx = self._frame2ttl_event_count()
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
                force_p95=snap.get("force_p95"),
                force_min=snap["force_min"],
                force_max=snap["force_max"],
                trial_start_wall_s=trial_wall,
                trial_start_mono_s=trial_mono,
                camera_ttl_timestamps_us=[ts for ts, _ in cam_events],
                lick_count=len(lick_events),
                lick_rate_hz=self._lick_rate_hz(
                    len(lick_events),
                    float(loop_stats.get("duration_s", rt)),
                ),
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
        frame2ttl_start_idx = self._frame2ttl_event_count()
        spec = self._trial_spec(trial_index)
        trial_mode = str(spec.get("trial_mode") or "free")
        forced_shape = spec.get("forced_shape")
        forced_shape = forced_shape if forced_shape in ("star", "circle") else None
        forced_shape_typed: Shape | None = forced_shape  # type: ignore[assignment]
        star_side = self._choose_star_side(trial_index)
        circle_side = self._opposite_side(star_side)
        star_direction = self._wheel_direction_for_side(star_side)
        circle_direction = self._wheel_direction_for_side(circle_side)
        choice_target_direction = (
            circle_direction
            if trial_mode == "forced" and forced_shape_typed == "circle"
            else star_direction
        )
        modality = str(self.config.task.modality or "force").lower()
        force_output_direction: Direction | None = (
            star_direction
            if modality == "force"
            and not (trial_mode == "forced" and forced_shape_typed == "circle")
            else None
        )
        self.recorder.log_event(
            "trial_start",
            {
                "trial_index": trial_index,
                "star_side": star_side,
                "circle_side": circle_side,
                "force_direction": force_output_direction,
                "force_output_direction": force_output_direction,
                "force_output_active": force_output_direction is not None,
                "choice_target_direction": choice_target_direction,
                "star_direction": star_direction,
                "circle_direction": circle_direction,
                "trial_mode": trial_mode,
                "forced_shape": forced_shape_typed,
                "modality": self.config.task.modality,
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
                "force_direction": force_output_direction,
                "force_output_direction": force_output_direction,
                "force_output_active": force_output_direction is not None,
                "choice_target_direction": choice_target_direction,
                "star_direction": star_direction,
                "circle_direction": circle_direction,
                "trial_mode": trial_mode,
                "forced_shape": forced_shape_typed,
                "modality": self.config.task.modality,
            },
        )

        result = self._run_bpod_decision_epoch(
            star_side=star_side,
            trial_index=trial_index,
            trial_mode=trial_mode,
            forced_shape=forced_shape_typed,
        )
        loop_stats = dict(self._last_choice_loop_stats)
        snap = self.wheel.snapshot()
        cam_events = self.dio.get_camera_events()[camera_start_idx:]
        lick_events = self._lick_events_since(lick_start_idx)
        frame2ttl_summary = dict(result.get("frame2ttl", {}))
        frame2ttl_edges_total = max(0, self._frame2ttl_event_count() - frame2ttl_start_idx)
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
                force_p95=snap.get("force_p95"),
                force_min=snap["force_min"],
                force_max=snap["force_max"],
                trial_start_wall_s=trial_wall,
                trial_start_mono_s=trial_mono,
                star_side=result["star_side"],
                circle_side=result["circle_side"],
                star_direction=result["star_direction"],
                circle_direction=result["circle_direction"],
                choice_target_direction=result["choice_target_direction"],
                force_output_direction=result["force_output_direction"],
                choice_side=result["choice_side"],
                chosen_shape=result["chosen_shape"],
                trial_mode=str(result["trial_mode"]),
                forced_shape=result["forced_shape"],
                modality=str(result["modality"]),
                reward_delay_s=float(result["reward_delay_s"]),
                reward_delivered=bool(result["reward_delivered"]),
                risk_star_probability=result["risk_star_probability"],
                bpod_state_entries=len(result["runtime"]["state_entries"]),
                camera_ttl_timestamps_us=[ts for ts, _ in cam_events],
                lick_count=len(lick_events),
                lick_rate_hz=self._lick_rate_hz(
                    len(lick_events),
                    float(loop_stats.get("duration_s", result["response_time_s"])),
                ),
                lick_timestamps_us=[int(event["timestamp_us"]) for event in lick_events],
                lick_relative_times_s=[float(event["relative_time_s"]) for event in lick_events],
                lick_channels=[int(event["channel"]) for event in lick_events],
                choice_loop_frames=int(loop_stats.get("frames", 0)),
                choice_loop_duration_s=float(loop_stats.get("duration_s", 0.0)),
                choice_loop_mean_frame_s=float(loop_stats.get("mean_frame_s", 0.0)),
                choice_loop_max_frame_s=float(loop_stats.get("max_frame_s", 0.0)),
                choice_loop_stall_count=int(loop_stats.get("stall_count", 0)),
                frame2ttl_code_word=frame2ttl_summary.get("code_word"),
                frame2ttl_trial_type_code=frame2ttl_summary.get("trial_type_code"),
                frame2ttl_barcode_complete=bool(
                    frame2ttl_summary.get("barcode_complete", False)
                ),
                frame2ttl_barcode_decoded=bool(
                    frame2ttl_summary.get("barcode_decoded", False)
                ),
                frame2ttl_barcode_match=bool(
                    frame2ttl_summary.get("barcode_match", False)
                ),
                frame2ttl_edge_count=frame2ttl_edges_total,
                frame2ttl_choice_edge_count=int(
                    frame2ttl_summary.get("choice_edge_count", 0) or 0
                ),
                frame2ttl_first_ttl_timestamp_us=frame2ttl_summary.get(
                    "first_ttl_timestamp_us"
                ),
                frame2ttl_first_ttl_relative_time_s=frame2ttl_summary.get(
                    "first_ttl_relative_time_s"
                ),
                frame2ttl_first_ttl_latency_from_bpod_s=frame2ttl_summary.get(
                    "first_ttl_latency_from_bpod_s"
                ),
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
                    "sound_status": self._sound_status(),
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
                force_p95=snap.get("force_p95"),
                force_min=snap["force_min"],
                force_max=snap["force_max"],
                trial_start_wall_s=trial_wall,
                trial_start_mono_s=trial_mono,
                camera_ttl_timestamps_us=[ts for ts, _ in cam_events],
                lick_count=len(lick_events),
                lick_rate_hz=self._lick_rate_hz(
                    len(lick_events),
                    response_time_s,
                ),
                lick_timestamps_us=[int(event["timestamp_us"]) for event in lick_events],
                lick_relative_times_s=[float(event["relative_time_s"]) for event in lick_events],
                lick_channels=[int(event["channel"]) for event in lick_events],
            )
        )
        self._run_iti()
        return centered
