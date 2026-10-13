"""Controlled live hardware check using the production task loop."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Literal

from hf_dm.config import SessionConfig, TrainingPhase
from hf_dm.hardware.bringup_constants import (
    CCW_FORCE_OUTPUT_CHANNEL,
    CW_FORCE_OUTPUT_CHANNEL,
    PUMP_OUTPUT_CHANNEL,
)
from hf_dm.fsm.runtime import TrialRuntime
from hf_dm.session.controller import HFDecisionSession
from hf_dm.streamdio.wheel import Direction
from hf_dm.validation import ValidationResult, validation_report

SelfTestChoice = Literal["matched", "opposite", "timeout"]


@dataclass(frozen=True)
class HardwareSelfTestOptions:
    injection_delay_s: float = 0.20
    pulse_interval_s: float = 0.004


class HardwareSelfTestSession(HFDecisionSession):
    """Exercise real outputs while command-driving wheel choices for three trials."""

    _sequence: tuple[SelfTestChoice, ...] = ("matched", "opposite", "timeout")
    _force_sequence: tuple[Direction, ...] = ("cw", "cw", "ccw")

    def __init__(self, config: SessionConfig, options: HardwareSelfTestOptions | None = None):
        super().__init__(config, run_mode="bench")
        self.options = options or HardwareSelfTestOptions()
        self._injectors: dict[int, threading.Thread] = {}

    def _choose_force_direction(self, trial_index: int) -> Direction:
        return self._force_sequence[(trial_index - 1) % len(self._force_sequence)]

    def _choose_star_side(self, trial_index: int):
        return "left" if self._choose_force_direction(trial_index) == "cw" else "right"

    def _on_choice_window_started(self, runtime: TrialRuntime) -> None:
        trial_index = runtime.trial_index
        force_direction = runtime.force_direction or runtime.choice_target_direction
        plan = self._sequence[(trial_index - 1) % len(self._sequence)]
        planned_choice: Literal["cw", "ccw", "timeout"]
        if plan == "matched":
            planned_choice = force_direction
        elif plan == "opposite":
            planned_choice = "ccw" if force_direction == "cw" else "cw"
        else:
            planned_choice = "timeout"

        assert self.recorder is not None
        self.recorder.log_event(
            "hardware_self_test_plan",
            {
                "trial_index": trial_index,
                "plan": plan,
                "force_direction": force_direction,
                "planned_choice": planned_choice,
                "synthetic_wheel_only": True,
                "real_outputs": True,
            },
        )
        if planned_choice != "timeout":
            injector = threading.Thread(
                target=self._inject_wheel_choice,
                args=(planned_choice, trial_index),
                name=f"hf-dm-self-test-wheel-{trial_index}",
                daemon=True,
            )
            self._injectors[trial_index] = injector
            injector.start()

    def _after_decision_trial(self, runtime: TrialRuntime) -> None:
        injector = self._injectors.pop(runtime.trial_index, None)
        if injector is not None:
            injector.join(timeout=2.0)

    def _run_live_choice_epoch_legacy(
        self,
        force_direction: Direction,
        trial_index: int,
    ) -> tuple[Literal["cw", "ccw", "timeout"], float, float, float]:
        plan = self._sequence[(trial_index - 1) % len(self._sequence)]
        planned_choice: Literal["cw", "ccw", "timeout"]
        if plan == "matched":
            planned_choice = force_direction
        elif plan == "opposite":
            planned_choice = "ccw" if force_direction == "cw" else "cw"
        else:
            planned_choice = "timeout"

        assert self.recorder is not None
        self.recorder.log_event(
            "hardware_self_test_plan",
            {
                "trial_index": trial_index,
                "plan": plan,
                "force_direction": force_direction,
                "planned_choice": planned_choice,
                "synthetic_wheel_only": True,
                "real_outputs": True,
            },
        )

        injector: threading.Thread | None = None
        if planned_choice != "timeout":
            injector = threading.Thread(
                target=self._inject_wheel_choice,
                args=(planned_choice, trial_index),
                name=f"hf-dm-self-test-wheel-{trial_index}",
                daemon=True,
            )
            injector.start()

        try:
            return super()._run_live_choice_epoch_legacy(force_direction, trial_index)
        finally:
            if injector is not None:
                injector.join(timeout=2.0)

    def _inject_wheel_choice(self, choice: Direction, trial_index: int) -> None:
        time.sleep(max(0.0, self.options.injection_delay_s))
        assert self.recorder is not None
        self.recorder.log_event(
            "hardware_self_test_injection_start",
            {"trial_index": trial_index, "choice": choice},
        )
        pulse_count = (
            int(math.ceil(self.config.display.choice_distance_deg / self.config.wheel.deg_per_pulse))
            + 2
        )
        direction_value = 1 if choice == "cw" else 0
        start = time.perf_counter()
        for _ in range(pulse_count):
            now = time.perf_counter()
            timestamp_us = int(now * 1_000_000)
            self.wheel.handle_event(
                "dio_input",
                self.config.wheel.direction_channel,
                direction_value,
                timestamp_us,
                now,
            )
            self.wheel.handle_event(
                "dio_input",
                self.config.wheel.pulse_channel,
                1,
                timestamp_us,
                now,
            )
            time.sleep(max(0.0, self.options.pulse_interval_s))
        self.recorder.log_event(
            "hardware_self_test_injection_end",
            {
                "trial_index": trial_index,
                "choice": choice,
                "pulses_injected": pulse_count,
                "duration_s": time.perf_counter() - start,
            },
        )

    def validate_results(self) -> dict:
        """Fail if the live production-loop diagnostic did not prove its contracts."""
        if self.recorder is None:
            raise RuntimeError("Hardware self-test has no session record")

        events = self.recorder.events
        event_by_trial: dict[int, dict[str, object]] = {}
        for event in events:
            trial_index = event.payload.get("trial_index")
            if trial_index is not None:
                event_by_trial.setdefault(int(trial_index), {})[event.event_type] = event

        failures: list[str] = []
        check_failures: dict[str, list[str]] = {}

        def fail(check: str, detail: str) -> None:
            failures.append(detail)
            check_failures.setdefault(check, []).append(detail)
        max_choice_to_pump_s = 0.0
        max_reward_error_s = 0.0
        bpod_states_by_trial = {
            int(item["trial_index"]): [
                str(row["state"])
                for row in item["events"]
                if row.get("type") == "StateStart" and row.get("state")
            ]
            for item in self.recorder.bpod_trials
        }
        for trial in self.recorder.trials:
            trial_events = event_by_trial.get(trial.trial_index, {})
            if trial.chosen_shape == "star":
                expected_states = [
                    "Quiescence_Pre",
                    "Quiescence",
                    "ChoiceWindow_Pre",
                    "ChoiceWindow",
                    "ChoiceBlankHigh_Pre",
                    "ChoiceBlankHigh",
                    "RewardHigh_Pre",
                    "RewardHigh",
                    "PostRewardHigh_Pre",
                    "PostRewardHigh",
                ]
            elif trial.chosen_shape == "circle":
                expected_states = [
                    "Quiescence_Pre",
                    "Quiescence",
                    "ChoiceWindow_Pre",
                    "ChoiceWindow",
                    "ChoiceBlankLow_Pre",
                    "ChoiceBlankLow",
                    "RewardLow_Pre",
                    "RewardLow",
                    "PostRewardLow_Pre",
                    "PostRewardLow",
                ]
            else:
                expected_states = [
                    "Quiescence_Pre",
                    "Quiescence",
                    "ChoiceWindow_Pre",
                    "ChoiceWindow",
                    "TimeoutNoise_Pre",
                    "TimeoutNoise",
                    "PostTimeout_Pre",
                    "PostTimeout",
                ]
                if trial.force_output_direction is not None:
                    expected_states.insert(4, "TimeoutForceOff")
            actual_states = bpod_states_by_trial.get(trial.trial_index, [])
            if actual_states != expected_states:
                fail(
                    "bpod_state_sequences",
                    f"trial {trial.trial_index}: decoded Bpod states "
                    f"{actual_states}, expected {expected_states}"
                )

            if trial.choice in ("cw", "ccw"):
                choice_event = trial_events.get("choice_detected")
                pump_start = trial_events.get("pump_start")
                pump_end = trial_events.get("pump_end")
                if choice_event is None or pump_start is None or pump_end is None:
                    fail(
                        "reward_event_chain",
                        f"trial {trial.trial_index}: missing choice/pump events",
                    )
                    continue
                choice_to_pump_s = (
                    float(pump_start.payload["actual_on_mono_s"])
                    - float(choice_event.monotonic_s)
                )
                actual_open_s = float(pump_end.payload["actual_open_s"])
                max_choice_to_pump_s = max(max_choice_to_pump_s, choice_to_pump_s)
                max_reward_error_s = max(
                    max_reward_error_s,
                    abs(actual_open_s - trial.reward_duration_s),
                )
                expected_choice_to_pump_s = self.config.timing.choice_blank_s
                if abs(choice_to_pump_s - expected_choice_to_pump_s) > 0.050:
                    fail(
                        "choice_to_pump_timing",
                        f"trial {trial.trial_index}: choice-to-pump error "
                        f"{choice_to_pump_s - expected_choice_to_pump_s:+.6f}s"
                    )
                if abs(actual_open_s - trial.reward_duration_s) > 0.010:
                    fail(
                        "reward_duration_timing",
                        f"trial {trial.trial_index}: pump duration error "
                        f"{actual_open_s - trial.reward_duration_s:+.6f}s"
                    )
            if trial.force_samples <= 0:
                fail("force_adc", f"trial {trial.trial_index}: no force ADC samples")
            if not trial.camera_ttl_timestamps_us:
                fail(
                    "camera_ttl_per_trial",
                    f"trial {trial.trial_index}: no camera TTL timestamps",
                )
            stall_threshold_s = max(
                float(self.config.timing.choice_loop_stall_s),
                (1.0 / max(1, self.config.display.fps)) * 3.0,
            )
            if (
                trial.choice_loop_stall_count
                or trial.choice_loop_max_frame_s >= stall_threshold_s
            ):
                fail(
                    "choice_loop_stalls",
                    f"trial {trial.trial_index}: {trial.choice_loop_stall_count} "
                    f"choice-loop stalls, max frame={trial.choice_loop_max_frame_s:.6f}s"
                )

        output_states = {
            (int(event.payload["channel"]), int(event.payload["state"]))
            for event in events
            if event.event_type == "output_actual"
        }
        required_output_states = {
            (PUMP_OUTPUT_CHANNEL, 1),
            (PUMP_OUTPUT_CHANNEL, 0),
            (CW_FORCE_OUTPUT_CHANNEL, 1),
            (CW_FORCE_OUTPUT_CHANNEL, 0),
            (CCW_FORCE_OUTPUT_CHANNEL, 1),
            (CCW_FORCE_OUTPUT_CHANNEL, 0),
        }
        if self.config.licks.source == "capacitive":
            required_output_states.update({
                (self.config.licks.power_output_channel, 1),
                (self.config.licks.power_output_channel, 0),
            })
        missing_outputs = sorted(required_output_states - output_states)
        if missing_outputs:
            fail(
                "firmware_output_events",
                f"missing firmware output events: {missing_outputs}",
            )

        noise_start_events = [event for event in events if event.event_type == "white_noise_start"]
        noise_end_events = [event for event in events if event.event_type == "white_noise_end"]
        if not noise_start_events or len(noise_start_events) != len(noise_end_events):
            fail("white_noise_gate", "StreamDIO pin-26 timeout gate did not complete")

        video_events = [event for event in events if event.event_type == "video_recording_stopped"]
        video_frames = sum(int(event.payload.get("frames_saved", 0)) for event in video_events)
        if video_frames <= 0:
            fail("spinnaker_recording", "Spinnaker recording saved no frames")
        if any(event.payload.get("capture_error") for event in video_events):
            fail("spinnaker_recording", "Spinnaker recording reported a capture error")

        stream_camera_ttls = sum(
            1 for event in self.recorder.streamdio_events if event[0] == "camera"
        )
        if stream_camera_ttls != video_frames:
            fail(
                "camera_ttl_frame_alignment",
                f"camera TTL/frame mismatch: {stream_camera_ttls} TTLs, {video_frames} frames"
            )
        first_trial_mono_s = min(
            (trial.trial_start_mono_s for trial in self.recorder.trials),
            default=math.inf,
        )
        diagnostic_events = [
            (int(event[1]), float(event[5]))
            for event in self.recorder.streamdio_events
            if event[0] == "diagnostic"
        ]
        startup_diagnostic_codes = [
            code for code, host_mono_s in diagnostic_events if host_mono_s < first_trial_mono_s
        ]
        trial_diagnostic_codes = [
            code for code, host_mono_s in diagnostic_events if host_mono_s >= first_trial_mono_s
        ]
        if trial_diagnostic_codes:
            fail(
                "streamdio_diagnostics",
                f"trial-time StreamDIO diagnostic events: {trial_diagnostic_codes}",
            )
        wheel_edges = sum(
            1
            for event in self.recorder.streamdio_events
            if event[0] == "dio_input"
            and int(event[1])
            in (self.config.wheel.pulse_channel, self.config.wheel.direction_channel)
        )

        check_specs = [
            (
                "bpod_state_sequences",
                f"decoded production state sequences for {len(self.recorder.trials)} trials",
                "Inspect exact Bpod rows and the generated FSM definition.",
            ),
            (
                "reward_event_chain",
                "choice, pump-start, and pump-end events were present",
                "Inspect output_actual events and pump channel wiring.",
            ),
            (
                "choice_to_pump_timing",
                f"maximum observed choice-to-pump delay={max_choice_to_pump_s * 1000.0:.3f} ms",
                "Check host load and Bpod ChoiceBlank timing.",
            ),
            (
                "reward_duration_timing",
                f"maximum pump duration error={max_reward_error_s * 1000.0:.3f} ms",
                "Check StreamDIO output acknowledgements and host scheduling load.",
            ),
            (
                "force_adc",
                f"force samples={sum(trial.force_samples for trial in self.recorder.trials)}",
                "Check force ADC channel 13/pin 14.",
            ),
            (
                "camera_ttl_per_trial",
                "camera TTL timestamps were present on every trial",
                "Check StreamDIO camera triggering and event streaming.",
            ),
            (
                "choice_loop_stalls",
                "no choice-loop stalls were recorded",
                "Reduce display/host load and inspect frame timing.",
            ),
            (
                "firmware_output_events",
                f"firmware output events={len(output_states)} distinct states",
                "Check StreamDIO USB1 output event reporting.",
            ),
            (
                "white_noise_gate",
                f"completed white-noise gate pulses={len(noise_end_events)}",
                "Check StreamDIO Teensy pin 26 and the continuous sound source.",
            ),
            (
                "spinnaker_recording",
                f"video frames={video_frames}",
                "Check FLIR trigger mode, exposure, and recording backend.",
            ),
            (
                "camera_ttl_frame_alignment",
                f"camera TTLs={stream_camera_ttls}, video frames={video_frames}",
                "Inspect FLIR frame timestamps and StreamDIO camera events.",
            ),
            (
                "streamdio_diagnostics",
                "no diagnostic codes occurred during trials",
                "Decode diagnostic IDs and inspect StreamDIO firmware state.",
            ),
        ]
        typed_results = []
        for name, pass_detail, hint in check_specs:
            details = check_failures.get(name, [])
            typed_results.append(
                ValidationResult(
                    name=name,
                    status="FAIL" if details else "PASS",
                    detail="; ".join(details) if details else pass_detail,
                    hint=hint,
                )
            )
        typed_results.append(
            ValidationResult(
                name="streamdio_startup_diagnostics",
                status="WARN" if startup_diagnostic_codes else "PASS",
                detail=(
                    f"startup diagnostic codes={startup_diagnostic_codes}"
                    if startup_diagnostic_codes
                    else "no startup diagnostic codes"
                ),
                hint="Startup-only codes may be benign; decode and document recurring values.",
            )
        )
        typed_report = validation_report(typed_results)

        report = {
            **typed_report,
            "failures": failures,
            "trials": len(self.recorder.trials),
            "max_choice_to_pump_ms": max_choice_to_pump_s * 1000.0,
            "max_reward_error_ms": max_reward_error_s * 1000.0,
            "firmware_output_events": len(
                [event for event in events if event.event_type == "output_actual"]
            ),
            "white_noise_gate_pulses": len(noise_end_events),
            "force_samples": sum(trial.force_samples for trial in self.recorder.trials),
            "camera_ttls": sum(
                len(trial.camera_ttl_timestamps_us) for trial in self.recorder.trials
            ),
            "stream_camera_ttls": stream_camera_ttls,
            "video_frames": video_frames,
            "startup_diagnostic_codes": startup_diagnostic_codes,
            "trial_diagnostic_codes": trial_diagnostic_codes,
            "physical_wheel_edges": wheel_edges,
            "lick_events": len([event for event in events if event.event_type == "lick"]),
            "decoded_bpod_states": bpod_states_by_trial,
        }
        if failures:
            raise RuntimeError(f"Hardware self-test validation failed: {failures}")
        return report


def shorten_for_hardware_self_test(config: SessionConfig) -> None:
    """Keep real reward durations while making the live diagnostic brief."""
    config.n_trials = 3
    config.phase = TrainingPhase.DECISION_MAKING
    config.task.modality = "force"
    config.task.n_free_trials = 3
    config.task.n_forced_trials = 0
    config.timing.quiescence_s = 0.05
    config.timing.inter_trial_min_s = 0.20
    config.timing.choice_timeout_s = 0.80
    config.timing.choice_blank_s = 0.05
    config.timing.post_reward_s = 0.05
    config.timing.timeout_noise_s = 0.20
    config.timing.post_timeout_s = 0.05
    config.display.pre_trial_black_s = 0.10
    config.display.feedback_hold_s = 0.20
    # Barcode integrity has its own live diagnostic. The shortened 50 ms
    # quiescence used here cannot carry a production Frame2TTL barcode.
    config.frame2ttl.barcode_enabled = False
    config.engagement.idle_timeout_s = 60.0
    config.engagement.max_consecutive_failures = 10
    config.camera.record_video = True
    config.camera.required = True
