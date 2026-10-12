"""Bpod-owned actuator runner for centering trials."""

from __future__ import annotations

import logging
import time
from typing import Any

from hf_dm.data.recorder import TrialRecord
from hf_dm.fsm import softcodes as sc
from hf_dm.fsm.builders import build_host_supervised_decision_fsm
from hf_dm.fsm.runtime import TrialRuntime
from hf_dm.streamdio.wheel import Direction

logger = logging.getLogger(__name__)


class CenteringTrialRunner:
    @staticmethod
    def run(session: Any, trial_index: int) -> bool:
        self = session
        assert self.bpod is not None and self.speaker_relay is not None
        assert self.recorder is not None and self.dio is not None and self.display is not None

        trial_wall = time.time()
        trial_mono = time.perf_counter()
        camera_start_idx = len(self.dio.get_camera_events())
        lick_start_idx = self._lick_event_count()
        start_position_deg = self.rng.choice(
            [-self.config.display.choice_distance_deg, self.config.display.choice_distance_deg]
        )
        start_side = "left" if start_position_deg < 0.0 else "right"
        movement_direction: Direction = self._wheel_direction_for_side(start_side)
        opposite_direction: Direction = "ccw" if movement_direction == "cw" else "cw"
        runtime = TrialRuntime(
            trial_index=trial_index,
            force_direction=movement_direction,
            choice_target_direction=movement_direction,
            star_direction=movement_direction,
            circle_direction=opposite_direction,
            star_side=start_side,
            circle_side=self._opposite_side(start_side),
            high_reward_s=float(self.config.reward.matched_pump_s),
            low_reward_s=0.0,
            high_delay_s=float(self.config.timing.choice_blank_s),
            low_delay_s=float(self.config.timing.choice_blank_s),
            task_kind="centering",
            centering_start_position_deg=float(start_position_deg),
        )
        self._active_trial_runtime = runtime
        self.recorder.log_event(
            "centering_trial_start",
            {
                "trial_index": trial_index,
                "start_position_deg": start_position_deg,
                "movement_direction": movement_direction,
                "threshold_deg": self.config.centering_threshold_deg,
                "trial_controller": "bpod_core_fsm",
                "output_control": "bpod_fsm_serial_with_async_streamdio_observation",
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

        fsm = build_host_supervised_decision_fsm(
            self.speaker_relay,
            force_direction=movement_direction,
            quiescence_s=self.config.timing.quiescence_s,
            choice_timeout_s=self.config.timing.choice_timeout_s,
            choice_blank_s=self.config.timing.choice_blank_s,
            high_reward_s=self._reward_state_timer(runtime.high_reward_s),
            low_reward_s=self._reward_state_timer(0.0),
            high_reward_enabled=runtime.high_reward_s > 0.0,
            low_reward_enabled=False,
            post_reward_s=self.config.timing.post_reward_s,
            timeout_noise_s=self.config.timing.timeout_noise_s,
            post_timeout_s=self.config.timing.post_timeout_s,
            high_delay_s=runtime.high_delay_s,
            low_delay_s=runtime.low_delay_s,
        )
        self.recorder.log_fsm_definition(f"bpod_centering_trial_{trial_index}", fsm)
        self.wheel.reset_trial()
        self._prepare_centering_trial(start_position_deg)
        self.display.fill_black()

        threshold = self.config.centering_threshold_deg
        choice_start_mono_s: float | None = None
        position_deg = start_position_deg
        feedback_logged = False
        stimulus_logged = False
        candidate_payload: dict[str, Any] | None = None
        last_quiescence_edges = 0
        abort_sent = False
        self._prepare_bpod_run(trial_index)
        self.bpod.run(fsm, trial_number=trial_index)
        try:
            while self.bpod.is_running:
                self._check_recorder_health()
                state = runtime.snapshot()
                phase = str(state["phase"])
                if state["error"] and not abort_sent:
                    self.bpod.send_softcode(sc.SESSION_ABORT)
                    abort_sent = True

                if phase == "quiescence":
                    self._track_engagement_from_wheel()
                    snap = self.wheel.snapshot()
                    edges = int(snap["cw_pulses"]) + int(snap["ccw_pulses"])
                    if edges != last_quiescence_edges:
                        self.bpod.send_softcode(sc.QUIESCENCE_MOVEMENT)
                        last_quiescence_edges = edges
                    self.display.pump_events()
                elif phase == "choice":
                    if choice_start_mono_s is None:
                        choice_start_mono_s = float(state["phase_started_mono_s"])
                    self._track_engagement_from_wheel()
                    position_deg = max(
                        -self.config.display.choice_distance_deg,
                        min(
                            self.config.display.choice_distance_deg,
                            start_position_deg + self.config.display.wheel_to_screen_deg(
                                self.wheel.signed_position_deg()
                            ),
                        ),
                    )
                    flip_mono_s = self.display.draw_centering(
                        movement_direction,
                        position_deg,
                        trial_index=trial_index,
                    )
                    if not stimulus_logged:
                        stimulus_logged = True
                        self.recorder.log_event(
                            "stimulus_on",
                            {
                                "trial_index": trial_index,
                                "phase": "centering",
                                "movement_direction": movement_direction,
                                "position_deg": position_deg,
                                "display_flip_mono_s": flip_mono_s,
                            },
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
                    if candidate_payload is None and abs(position_deg) <= threshold:
                        detected_mono_s = time.perf_counter()
                        if runtime.commit_choice(
                            shape="star",
                            side=runtime.star_side,
                            wheel_choice=movement_direction,
                            detected_mono_s=detected_mono_s,
                        ):
                            candidate_payload = {
                                "trial_index": trial_index,
                                "position_deg": position_deg,
                                "response_time_s": max(
                                    0.0, detected_mono_s - float(choice_start_mono_s)
                                ),
                                "host_to_bpod_softcode": sc.CHOICE_HIGH,
                                "detected_mono_s": detected_mono_s,
                            }
                            self.recorder.log_debug_event(
                                "centering_candidate", candidate_payload
                            )
                            self.bpod.send_softcode(sc.CHOICE_HIGH)
                else:
                    if phase in {"blank", "reward", "post_reward"} and not feedback_logged:
                        feedback_logged = True
                        flip_mono_s = self.display.draw_centering(
                            movement_direction,
                            position_deg,
                            outcome="centered",
                            trial_index=trial_index,
                            response_time_s=(
                                None
                                if candidate_payload is None
                                else float(candidate_payload["response_time_s"])
                            ),
                        )
                        self.recorder.log_event(
                            "feedback_on",
                            {
                                "trial_index": trial_index,
                                "outcome": "centered",
                                "position_deg": position_deg,
                                "display_flip_mono_s": flip_mono_s,
                                "bpod_confirmed": True,
                            },
                        )
                        if candidate_payload is not None:
                            self.recorder.log_event(
                                "choice_detected",
                                {**candidate_payload, "bpod_confirmed": True},
                            )
                    elif phase in {"timeout", "post_timeout"} and not feedback_logged:
                        feedback_logged = True
                        flip_mono_s = self.display.draw_centering(
                            movement_direction,
                            position_deg,
                            outcome="timeout",
                            trial_index=trial_index,
                            response_time_s=self.config.timing.choice_timeout_s,
                        )
                        self.recorder.log_event(
                            "feedback_on",
                            {
                                "trial_index": trial_index,
                                "outcome": "timeout",
                                "position_deg": position_deg,
                                "display_flip_mono_s": flip_mono_s,
                                "trigger": "bpod_choice_window_tup",
                            },
                        )
                    else:
                        self.display.pump_events()
                self.display.clock.tick(self.config.display.fps)
        except BaseException:
            if self.bpod.is_running:
                try:
                    self.bpod.send_softcode(sc.SESSION_ABORT)
                    self._wait_for_bpod_trial_threads()
                except Exception:
                    logger.exception("Failed to abort Bpod centering trial after host exception")
            raise
        finally:
            self._wait_for_bpod_trial_threads()
            self._drain_output_verifications(runtime)

        self.recorder.log_bpod_trial(trial_index, self.bpod.get_data(concat=False))
        state = runtime.snapshot()
        self._active_trial_runtime = None
        if state["error"]:
            raise RuntimeError(f"Bpod centering output failed: {state['error']}")

        centered = state["chosen_shape"] == "star"
        if candidate_payload is not None and not centered:
            self.recorder.log_event(
                "centering_candidate_rejected",
                {
                    **candidate_payload,
                    "reason": "bpod_timeout_won_boundary_race",
                    "final_bpod_phase": str(state["phase"]),
                },
            )
        response_time_s = (
            float(candidate_payload["response_time_s"])
            if centered and candidate_payload is not None
            else self.config.timing.choice_timeout_s
        )
        outcome = "centered" if centered else "timeout"
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
        reward_s = self.config.reward.matched_pump_s if centered else 0.0
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
                lick_rate_hz=self._lick_rate_hz(len(lick_events), response_time_s),
                lick_timestamps_us=[int(event["timestamp_us"]) for event in lick_events],
                lick_relative_times_s=[float(event["relative_time_s"]) for event in lick_events],
                lick_channels=[int(event["channel"]) for event in lick_events],
            )
        )
        self._run_iti()
        return centered
