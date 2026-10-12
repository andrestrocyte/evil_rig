"""Behavior-preserving runners for production Bpod decision trials."""

from __future__ import annotations

import logging
import time
from typing import Any

from hf_dm.data.recorder import TrialRecord
from hf_dm.fsm import softcodes as sc
from hf_dm.fsm.builders import build_host_supervised_decision_fsm
from hf_dm.fsm.runtime import Side, Shape, TrialRuntime
from hf_dm.streamdio.wheel import Direction

logger = logging.getLogger(__name__)


class BpodDecisionEpochRunner:
    @staticmethod
    def run(
        session: Any,
        *,
        star_side: Side,
        trial_index: int,
        trial_mode: str = "free",
        forced_shape: Shape | None = None,
    ) -> dict:
        self = session
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
            force_direction=force_output_direction,
            quiescence_s=self.config.timing.quiescence_s,
            choice_timeout_s=self.config.timing.choice_timeout_s,
            choice_blank_s=self.config.timing.choice_blank_s,
            high_reward_s=self._reward_state_timer(runtime.high_reward_s),
            low_reward_s=self._reward_state_timer(runtime.low_reward_s),
            high_reward_enabled=runtime.high_reward_s > 0.0,
            low_reward_enabled=runtime.low_reward_s > 0.0,
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
        choice_loop_iteration_count = 0
        choice_loop_interval_sum_s = 0.0
        choice_loop_interval_max_s = 0.0
        stall_count = 0
        previous_choice_loop_mono_s: float | None = None
        stimulus_visible = False
        last_drawn_choice_signature: tuple[float, float, str | None] | None = None
        choice_sync_square_state = False
        timeout_logged = False
        choice_candidate_payload: dict[str, Any] | None = None
        choice_confirmed_logged = False
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
                self._check_recorder_health()
                state = runtime.snapshot()
                phase = str(state["phase"])
                if state["error"] and not abort_sent:
                    self.bpod.send_softcode(sc.SESSION_ABORT)
                    abort_sent = True

                if phase == "quiescence":
                    self._track_engagement_from_wheel()
                    wheel_snap = self.wheel.snapshot(include_percentile=False)
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
                    loop_mono_s = time.perf_counter()
                    if previous_choice_loop_mono_s is not None:
                        loop_interval_s = loop_mono_s - previous_choice_loop_mono_s
                        choice_loop_interval_sum_s += loop_interval_s
                        choice_loop_interval_max_s = max(
                            choice_loop_interval_max_s,
                            loop_interval_s,
                        )
                        if loop_interval_s >= stall_threshold_s:
                            stall_count += 1
                            self.recorder.log_debug_event(
                                "choice_loop_stall",
                                {
                                    "trial_index": trial_index,
                                    "interval_s": loop_interval_s,
                                    "stall_threshold_s": stall_threshold_s,
                                    "frame": frame_count,
                                },
                            )
                    previous_choice_loop_mono_s = loop_mono_s
                    choice_loop_iteration_count += 1
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
                    wheel_snap = self.wheel.snapshot(include_percentile=False)
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
                            choice_candidate_payload = {
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
                                "detected_mono_s": detected_mono_s,
                            }
                            self.recorder.log_debug_event(
                                "choice_candidate",
                                choice_candidate_payload,
                            )
                            self._publish_monitor(
                                "task_state",
                                {
                                    "trial_index": trial_index,
                                    "phase": "choice_candidate",
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
                    if (
                        choice_candidate_payload is not None
                        and not choice_confirmed_logged
                        and state["chosen_shape"] in {"star", "circle"}
                        and phase in {"blank", "reward", "post_reward"}
                    ):
                        confirmed_payload = {
                            **choice_candidate_payload,
                            "bpod_confirmed": True,
                            "bpod_confirmed_phase": phase,
                            "confirmed_mono_s": time.perf_counter(),
                        }
                        self.recorder.log_event("choice_detected", confirmed_payload)
                        self._publish_monitor(
                            "task_state",
                            {
                                "trial_index": trial_index,
                                "phase": "choice_detected",
                                "choice": confirmed_payload["choice"],
                                "choice_side": confirmed_payload["choice_side"],
                                "chosen_shape": confirmed_payload["chosen_shape"],
                                "matched": confirmed_payload["matched"],
                                "response_time_s": confirmed_payload["response_time_s"],
                            },
                        )
                        choice_confirmed_logged = True
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
            self._drain_output_verifications(runtime)

        bpod_data = self.bpod.get_data(concat=False)
        self.recorder.log_bpod_trial(trial_index, bpod_data)
        self._after_decision_trial(runtime)
        state = runtime.snapshot()
        self._active_trial_runtime = None
        if state["error"]:
            raise RuntimeError(f"Bpod trial output failed: {state['error']}")

        if state["chosen_shape"] in {"star", "circle"} and not choice_confirmed_logged:
            assert choice_candidate_payload is not None
            self.recorder.log_event(
                "choice_detected",
                {
                    **choice_candidate_payload,
                    "bpod_confirmed": True,
                    "bpod_confirmed_phase": str(state["phase"]),
                    "confirmed_mono_s": time.perf_counter(),
                },
            )
            choice_confirmed_logged = True
        elif choice_candidate_payload is not None and state["chosen_shape"] is None:
            self.recorder.log_event(
                "choice_candidate_rejected",
                {
                    **choice_candidate_payload,
                    "reason": "bpod_timeout_won_boundary_race",
                    "final_bpod_phase": str(state["phase"]),
                },
            )

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
            frame_count=choice_loop_iteration_count,
            frame_interval_sum_s=choice_loop_interval_sum_s,
            frame_interval_max_s=choice_loop_interval_max_s,
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

class BpodDecisionTrialRunner:
    @staticmethod
    def run(session: Any, trial_index: int) -> bool:
        self = session
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
        self._run_iti()
        return bool(result["responded"])
