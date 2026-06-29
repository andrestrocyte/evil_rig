"""Behavior-preserving runner for centering trials."""

from __future__ import annotations

import time
from typing import Any

from hf_dm.data.recorder import TrialRecord
from hf_dm.fsm.relay import wait_until_perf_counter
from hf_dm.streamdio.wheel import Direction


class CenteringTrialRunner:
    @staticmethod
    def run(session: Any, trial_index: int) -> bool:
        self = session
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
