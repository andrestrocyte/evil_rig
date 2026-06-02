"""Session orchestration: bpod-core FSM + StreamDIO wheel supervisor + display."""

from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Literal

from bpod_core.bpod import Bpod

from hf_dm.config import SessionConfig, TrainingPhase
from hf_dm.camera.spinnaker import HeadlessSpinnakerRecorder
from hf_dm.hardware.discovery import connect_bpod_core
from hf_dm.data.recorder import SessionRecorder, SessionSummary, TrialRecord
from hf_dm.display.pygame_display import TaskDisplay
from hf_dm.fsm import builders, softcodes as sc
from hf_dm.fsm.relay import SerialRelay
from hf_dm.streamdio.client import StreamDIOClient
from hf_dm.streamdio.wheel import Direction, WheelAccumulator

logger = logging.getLogger(__name__)


class HFDecisionSession:
    def __init__(self, config: SessionConfig):
        self.config = config
        self.rng = random.Random(config.random_seed)
        self.bpod: Bpod | None = None
        self.relay: SerialRelay | None = None
        self.dio: StreamDIOClient | None = None
        self.wheel = WheelAccumulator(config.wheel)
        self.display: TaskDisplay | None = None
        self.recorder: SessionRecorder | None = None
        self.camera_recorder: HeadlessSpinnakerRecorder | None = None
        self._last_wheel_movement_mono = time.monotonic()
        self._consecutive_failures = 0

    def connect(self) -> None:
        ports = self.config.ports
        self.bpod = connect_bpod_core(None if ports.bpod_port == "AUTO" else ports.bpod_port)
        self.bpod.reset_session_clock()
        self.relay = SerialRelay(self.bpod)

        self.dio = StreamDIOClient(
            config_port=ports.dio_config_port,
            stream_port=ports.dio_stream_port,
            baudrate=ports.baudrate,
        )
        excluded = self.bpod.port if self.bpod else None
        self.dio.open(excluded_bpod_port=excluded)
        self.dio.configure_rotary_and_force()
        self.dio.start_camera_session(
            frequency_hz=self.config.camera.frequency_hz,
            exposure_us=self.config.camera.exposure_us,
        )
        self.dio.start_streaming(callback=self.wheel.handle_event)

        self.display = TaskDisplay(self.config.display)
        out = Path(self.config.results_dir)
        self.recorder = SessionRecorder(out, self.config.subject_id, self.config.phase.value)
        self._start_optional_video_recording()
        self.recorder.log_event(
            "hardware_connected",
            {
                "bpod_port": self.bpod.port,
                "dio_config": self.dio.config_port,
                "dio_stream": self.dio.stream_port,
                "camera_hz": self.config.camera.frequency_hz,
                "camera_backend": self.config.camera.backend,
            },
        )

    def close(self) -> None:
        self._stop_optional_video_recording()
        if self.dio is not None:
            self.dio.close()
        if self.bpod is not None:
            self.bpod.close()

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
            )
            session_dir = recorder.start()
            self.camera_recorder = recorder
            self.recorder.log_event(
                "video_recording_started",
                {
                    "backend": "spinnaker",
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

        noise = __import__("hf_dm.audio.white_noise", fromlist=["WhiteNoisePlayer"]).WhiteNoisePlayer()
        if self.config.white_noise_continuous:
            noise.start_continuous()

        end_reason = "completed"
        try:
            for trial_index in range(1, self.config.n_trials + 1):
                if self._should_end_session():
                    end_reason = self._engagement_end_reason()
                    break

                if self.config.phase == TrainingPhase.CENTERING:
                    ok = self._run_centering_trial(trial_index)
                else:
                    ok = self._run_dm_trial(trial_index)

                if not ok:
                    self._consecutive_failures += 1
                else:
                    self._consecutive_failures = 0

                if self._should_end_session():
                    end_reason = self._engagement_end_reason()
                    break
        finally:
            noise.stop()
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
            path = self.recorder.finalize(summary)
            logger.info("Session data saved to %s", path)
            self.display.close()
            self.close()
            return summary

    def _should_end_session(self) -> bool:
        idle_s = time.monotonic() - self._last_wheel_movement_mono
        if idle_s >= self.config.engagement.idle_timeout_s:
            return True
        if self._consecutive_failures >= self.config.engagement.max_consecutive_failures:
            return True
        return False

    def _engagement_end_reason(self) -> str:
        idle_s = time.monotonic() - self._last_wheel_movement_mono
        if idle_s >= self.config.engagement.idle_timeout_s:
            return "idle_timeout"
        return "max_consecutive_failures"

    def _track_engagement_from_wheel(self) -> None:
        snap = self.wheel.snapshot()
        if snap["cw_pulses"] or snap["ccw_pulses"] or abs(snap["net_pulses"]) > 0:
            self._last_wheel_movement_mono = time.monotonic()

    def _wait_quiescence(self) -> None:
        assert self.recorder is not None
        self.recorder.log_event("quiescence_wait_start", {"required_s": self.config.timing.quiescence_s})
        self.wheel.reset_trial()
        while True:
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

    def _poll_choice(
        self,
        force_direction: Direction,
        deadline_s: float,
        trial_index: int,
    ) -> tuple[Literal["cw", "ccw", "timeout"], float, float]:
        assert self.display is not None and self.bpod is not None
        choice_distance = self.config.display.choice_distance_deg
        start = time.monotonic()
        while time.monotonic() < start + deadline_s:
            self._track_engagement_from_wheel()
            snap = self.wheel.snapshot()
            pos_deg = max(-choice_distance, min(choice_distance, snap["net_pulses"] * self.config.wheel.deg_per_pulse))
            elapsed = time.monotonic() - start
            self.display.draw_trial(force_direction, pos_deg, trial_index=trial_index)
            if pos_deg >= choice_distance:
                return "cw", elapsed, pos_deg
            if pos_deg <= -choice_distance:
                return "ccw", elapsed, pos_deg
            time.sleep(1.0 / self.config.display.fps)
        return "timeout", deadline_s, pos_deg

    def _run_fsm_choice_epoch(
        self,
        force_direction: Direction,
        trial_index: int,
    ) -> tuple[Literal["cw", "ccw", "timeout"], float, float]:
        assert self.bpod is not None and self.relay is not None and self.recorder is not None
        fsm = builders.build_choice_epoch_fsm(
            self.relay,
            force_direction,
            self.config.timing.choice_timeout_s,
        )
        fsm_name = f"choice_epoch_{force_direction}"
        self.recorder.log_fsm_definition(fsm_name, fsm)
        self.recorder.log_event(
            "fsm_run_start",
            {"fsm_name": fsm_name, "trial_index": trial_index, "force_direction": force_direction},
        )
        self.bpod.validate_state_machine(fsm)
        self.bpod.run(fsm)

        choice: Literal["cw", "ccw", "timeout"] = "timeout"
        response_time_s = self.config.timing.choice_timeout_s
        final_pos_deg = 0.0

        while self.bpod.is_running:
            choice, response_time_s, final_pos_deg = self._poll_choice(
                force_direction,
                self.config.timing.choice_timeout_s,
                trial_index,
            )
            if choice in ("cw", "ccw"):
                code = sc.CHOICE_CW if choice == "cw" else sc.CHOICE_CCW
                self.bpod.send_softcode(code)
                break
            time.sleep(1.0 / self.config.display.fps)

        self.bpod.wait()
        self.recorder.log_event(
            "fsm_run_end",
            {"fsm_name": fsm_name, "trial_index": trial_index, "ended_choice": choice},
        )
        if not self.bpod.is_running and choice == "timeout":
            # FSM ended via Tup → TimeoutPunish branch (white noise played on hardware)
            pass
        return choice, response_time_s, final_pos_deg

    def _run_reward_fsm(self, matched: bool) -> float:
        assert self.bpod is not None and self.relay is not None and self.recorder is not None
        duration = (
            self.config.reward.matched_pump_s if matched else self.config.reward.opposite_pump_s
        )
        fsm = builders.build_reward_fsm(self.relay, duration)
        fsm_name = f"reward_{'matched' if matched else 'opposite'}_{duration:.3f}s"
        self.recorder.log_fsm_definition(fsm_name, fsm)
        self.recorder.log_event("fsm_run_start", {"fsm_name": fsm_name, "matched": matched})
        self.bpod.validate_state_machine(fsm)
        self.bpod.run(fsm)
        self.bpod.wait()
        self.recorder.log_event("fsm_run_end", {"fsm_name": fsm_name, "matched": matched})
        return duration

    def _run_iti(self) -> None:
        assert self.bpod is not None and self.recorder is not None
        fsm = builders.build_iti_fsm(self.config.timing.inter_trial_min_s)
        duration_s = self.config.timing.inter_trial_min_s
        fsm_name = f"iti_{duration_s:.3f}s"
        self.recorder.log_fsm_definition(fsm_name, fsm)
        self.recorder.log_event("fsm_run_start", {"fsm_name": fsm_name})
        self.bpod.validate_state_machine(fsm)
        self.bpod.run(fsm)
        self.bpod.wait()
        self.recorder.log_event("fsm_run_end", {"fsm_name": fsm_name})

    def _run_dm_trial(self, trial_index: int) -> bool:
        assert (
            self.bpod is not None
            and self.relay is not None
            and self.recorder is not None
            and self.display is not None
            and self.dio is not None
        )
        force_direction: Direction = self.rng.choice(["cw", "ccw"])
        trial_wall = time.time()
        trial_mono = time.monotonic()
        self.recorder.log_event("trial_start", {"trial_index": trial_index, "force_direction": force_direction})

        self._pre_trial_black()
        self._wait_quiescence()

        choice, rt, pos_deg = self._run_fsm_choice_epoch(force_direction, trial_index)

        matched = False
        reward_s = 0.0
        responded = choice in ("cw", "ccw")

        if responded:
            matched = choice == force_direction
            reward_s = self._run_reward_fsm(matched)
            outcome = "matched" if matched else "opposite"
            self.display.draw_trial(force_direction, pos_deg, outcome=outcome, trial_index=trial_index)
            time.sleep(self.config.display.feedback_hold_s)
        else:
            self.display.draw_trial(force_direction, pos_deg, outcome="timeout", trial_index=trial_index)
            time.sleep(self.config.display.feedback_hold_s)

        snap = self.wheel.snapshot()
        cam_events = self.dio.get_camera_events()
        self.recorder.add_trial(
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
            )
        )
        self._run_iti()
        return responded

    def _run_centering_trial(self, trial_index: int) -> bool:
        assert self.bpod is not None and self.relay is not None and self.recorder is not None
        self.recorder.log_event("centering_trial_start", {"trial_index": trial_index})
        self._pre_trial_black()
        self._wait_quiescence()

        threshold = self.config.centering_threshold_deg
        centered = False
        start = time.monotonic()
        while time.monotonic() - start < 30.0:
            self._track_engagement_from_wheel()
            pos = abs(self.wheel.signed_position_deg())
            if pos <= threshold:
                centered = True
                break
            time.sleep(0.02)

        reward_s = 0.0
        if centered:
            fsm = builders.build_centering_reward_fsm(self.relay, self.config.reward.matched_pump_s)
            fsm_name = f"centering_reward_{self.config.reward.matched_pump_s:.3f}s"
            self.recorder.log_fsm_definition(fsm_name, fsm)
            self.recorder.log_event("fsm_run_start", {"fsm_name": fsm_name, "trial_index": trial_index})
            self.bpod.validate_state_machine(fsm)
            self.bpod.run(fsm)
            self.bpod.send_softcode(sc.QUIESCENCE_OK)
            self.bpod.wait()
            self.recorder.log_event("fsm_run_end", {"fsm_name": fsm_name, "trial_index": trial_index})
            reward_s = self.config.reward.matched_pump_s

        snap = self.wheel.snapshot()
        self.recorder.add_trial(
            TrialRecord(
                trial_index=trial_index,
                phase="centering",
                force_direction=None,
                choice="centering" if centered else "timeout",
                matched=centered,
                reward_duration_s=reward_s,
                responded=centered,
                response_time_s=time.monotonic() - start,
                net_pulses=snap["net_pulses"],
                cw_pulses=snap["cw_pulses"],
                ccw_pulses=snap["ccw_pulses"],
                final_position_deg=self.wheel.signed_position_deg(),
                force_samples=snap["force_samples"],
                force_last=snap["force_last"],
                force_mean=snap["force_mean"],
                force_min=snap["force_min"],
                force_max=snap["force_max"],
                trial_start_wall_s=time.time(),
                trial_start_mono_s=time.monotonic(),
            )
        )
        self._run_iti()
        return centered
