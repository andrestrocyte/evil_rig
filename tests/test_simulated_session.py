import json
import os
import time
from itertools import groupby
from pathlib import Path

import numpy as np

from hf_dm.config import SessionConfig, TrainingPhase
from hf_dm.data.session_export import session_bundle_dir
from hf_dm.fsm import softcodes as sc
from hf_dm.fsm.runtime import TrialRuntime
from hf_dm.hardware.bringup_constants import FRAME2TTL_INPUT_CHANNEL
from hf_dm.session.controller import HFDecisionSession
from hf_dm.session.simulated import (
    SimulatedHFDecisionSession,
    SimulationOptions,
    shorten_for_simulation,
)
from hf_dm.valve_calibration import (
    ValveMeasurement,
    fit_valve_calibration,
    save_valve_calibration,
)


def _max_run(values):
    return max((sum(1 for _ in group) for _, group in groupby(values)), default=0)


def test_simulated_session_runs_choice_timeout_and_cleanup(tmp_path):
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    config = SessionConfig(n_trials=3, results_dir=str(tmp_path))
    config.camera.save_dir = str(tmp_path / "videos")
    shorten_for_simulation(config)
    session = SimulatedHFDecisionSession(
        config,
        SimulationOptions(choice_mode="alternate", frames_to_choice=4),
    )

    session.connect()
    summary = session.run()

    choices = [trial.choice for trial in session.recorder.trials]
    rewards = [trial.reward_duration_s for trial in session.recorder.trials]
    event_types = [event.event_type for event in session.recorder.events]
    assert summary.n_trials_completed == 3
    assert choices[0] in ("cw", "ccw")
    assert choices[1] in ("cw", "ccw")
    assert choices[2] == "timeout"
    assert rewards[0] > 0
    assert rewards[1] > 0
    assert rewards[2] == 0
    assert session.recorder.trials[0].chosen_shape == "star"
    assert session.recorder.trials[1].chosen_shape == "circle"
    assert session.recorder.trials[2].chosen_shape is None
    assert "pump_start" in event_types
    assert "white_noise_start" in event_types
    assert "video_recording_stopped" in event_types
    assert "streamdio_camera_triggers_stopped" in event_types
    assert event_types.count("lick") == 3
    assert len(session.recorder.streamdio_events) > 0
    assert all(trial.lick_count >= 1 for trial in session.recorder.trials)
    assert all(trial.force_samples > 0 for trial in session.recorder.trials)
    assert session.recorder.trials[0].frame2ttl_choice_edge_count > 0
    assert session.recorder.trials[1].frame2ttl_choice_edge_count > 0
    # A stationary timeout trial redraws once, but the choice-loop cadence must
    # still be measured on every host iteration rather than between redraws.
    assert session.recorder.trials[2].choice_loop_frames > 1
    assert session.recorder.trials[2].choice_loop_stall_count == 0
    assert session.recorder.trials[2].choice_loop_max_frame_s >= 0.0
    assert session.recorder.trials[2].frame2ttl_choice_edge_count <= 2
    assert any(
        event[0] == "dio_input" and int(event[1]) == FRAME2TTL_INPUT_CHANNEL
        for event in session.recorder.streamdio_events
    )
    assert any(event[0] == "bpod_state" for event in session.recorder.streamdio_events)
    heartbeat_rows = [
        event
        for event in session.recorder.streamdio_events
        if event[0] == "timing_heartbeat"
    ]
    assert len(heartbeat_rows) >= 3
    assert "bpod_state_marker" in event_types
    assert event_types.count("timing_heartbeat") == len(heartbeat_rows)
    assert event_types.count("iti_start") == 3
    assert event_types.count("iti_end") == 3
    speaker_states = [operation[1] for operation in session.speaker_relay.ops]
    assert True in speaker_states
    assert speaker_states.index(True) < speaker_states.index(False, speaker_states.index(True))
    assert len(session.recorder.bpod_trials) == 3
    with np.load(session.recorder.base_path.with_suffix(".npz"), allow_pickle=True) as data:
        assert "trials_json" in data.files
        assert "events_json" in data.files
        assert "timing_heartbeat_qc_json" in data.files
        first_trial = json.loads(str(data["trials_json"][0]))
        assert "lick_timestamps_us" in first_trial
        assert "camera_ttl_timestamps_us" in first_trial
        heartbeat_qc = json.loads(str(data["timing_heartbeat_qc_json"][0]))
        assert heartbeat_qc["overall_status"] == "PASS"
        assert heartbeat_qc["heartbeat_count"] == len(heartbeat_rows)
    bundle = session_bundle_dir(
        session.recorder.base_path,
        layout=config.session_export.layout,
    )
    heartbeat_csv = bundle / "exports" / "streamdio_timing_heartbeat.csv"
    heartbeat_qc_path = bundle / "exports" / "streamdio_timing_qc.json"
    assert heartbeat_csv.exists()
    assert json.loads(heartbeat_qc_path.read_text(encoding="utf-8"))["overall_status"] == "PASS"
    timeout_states = [
        row["state"]
        for row in session.recorder.bpod_trials[2]["events"]
        if row["type"] == "StateStart"
    ]
    assert timeout_states == [
        "Quiescence_Pre",
        "Quiescence",
        "ChoiceWindow_Pre",
        "ChoiceWindow",
        "TimeoutForceOff",
        "TimeoutNoise_Pre",
        "TimeoutNoise",
        "PostTimeout_Pre",
        "PostTimeout",
    ]


def test_choice_candidate_loses_cleanly_when_bpod_timeout_wins(tmp_path):
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    config = SessionConfig(n_trials=1, results_dir=str(tmp_path))
    config.camera.record_video = False
    config.camera.save_dir = str(tmp_path / "videos")
    shorten_for_simulation(config)
    config.timing.choice_timeout_s = 0.20
    session = SimulatedHFDecisionSession(
        config,
        SimulationOptions(choice_mode="matched", frames_to_choice=1),
    )
    session.connect()
    original_send = session.bpod.send_softcode

    def withhold_choice_until_timeout(softcode: int) -> None:
        if softcode in {sc.CHOICE_HIGH, sc.CHOICE_LOW}:
            while session.bpod.is_running:
                time.sleep(0.001)
            return
        original_send(softcode)

    session.bpod.send_softcode = withhold_choice_until_timeout

    summary = session.run()

    assert summary.n_trials_completed == 1
    assert session.recorder.trials[0].choice == "timeout"
    event_types = [event.event_type for event in session.recorder.events]
    assert "choice_detected" not in event_types
    assert "choice_candidate_rejected" in event_types


def test_softcode_output_observation_is_nonblocking_and_never_actuates(
    tmp_path,
    monkeypatch,
):
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    config = SessionConfig(n_trials=1, results_dir=str(tmp_path))
    config.camera.record_video = False
    shorten_for_simulation(config)
    session = SimulatedHFDecisionSession(config, SimulationOptions(choice_mode="timeout"))
    session.connect()
    runtime = TrialRuntime(
        trial_index=1,
        force_direction="cw",
        choice_target_direction="cw",
        star_direction="cw",
        circle_direction="ccw",
        star_side="left",
        circle_side="right",
    )
    runtime.simulated_choice = "timeout"  # type: ignore[attr-defined]
    session._active_trial_runtime = runtime
    session._reset_output_verification_sequences()

    def missing_event(*_args, **_kwargs):
        time.sleep(0.15)
        raise RuntimeError("synthetic missing output event")

    monkeypatch.setattr(session.dio, "wait_for_output_event", missing_event)
    task_ops_before = list(session.relay.ops)

    started = time.perf_counter()
    session._handle_bpod_softcode(sc.STATE_CHOICE)
    callback_elapsed_s = time.perf_counter() - started

    assert callback_elapsed_s < 0.05
    assert runtime.snapshot()["phase"] == "choice"
    assert session.relay.ops == task_ops_before
    session._drain_output_verifications(runtime)
    assert "synthetic missing output event" in str(runtime.snapshot()["error"])
    assert "output_verification_failed" in [
        event.event_type for event in session.recorder.events
    ]
    session._active_trial_runtime = None
    session.close()


def test_default_balancing_preserves_seeded_trial_record_snapshot(tmp_path):
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    config = SessionConfig(n_trials=6, random_seed=23, results_dir=str(tmp_path))
    config.camera.record_video = False
    config.camera.save_dir = str(tmp_path / "videos")
    shorten_for_simulation(config)
    session = SimulatedHFDecisionSession(
        config,
        SimulationOptions(choice_mode="alternate", frames_to_choice=2),
    )

    session.connect()
    session.run()

    snapshot = [
        (
            trial.trial_index,
            trial.star_side,
            trial.circle_side,
            trial.choice,
            trial.chosen_shape,
            trial.matched,
            trial.reward_duration_s,
            trial.trial_mode,
            trial.forced_shape,
            trial.modality,
            trial.reward_delivered,
            trial.risk_star_probability,
        )
        for trial in session.recorder.trials
    ]
    assert snapshot == [
        (1, "right", "left", "ccw", "star", True, 0.08, "free", None, "force", True, None),
        (2, "right", "left", "cw", "circle", False, 0.04, "free", None, "force", True, None),
        (3, "right", "left", "timeout", None, False, 0.0, "free", None, "force", False, None),
        (4, "left", "right", "cw", "star", True, 0.08, "free", None, "force", True, None),
        (5, "left", "right", "ccw", "circle", False, 0.04, "free", None, "force", True, None),
        (6, "right", "left", "timeout", None, False, 0.0, "free", None, "force", False, None),
    ]
    sidecar = json.loads(session.recorder.base_path.with_suffix(".json").read_text(encoding="utf-8"))
    provenance = sidecar["metadata"]["runtime_provenance"]
    header = json.loads(session.recorder.journal_path.read_text(encoding="utf-8").splitlines()[0])
    assert "git_commit" in provenance
    assert provenance["git_dirty"] in {True, False, None}
    assert provenance["package_version"] == "0.1.0"
    installed_bpod_core = provenance["package_versions"]["bpod-core"]
    lock_text = (Path(__file__).resolve().parents[1] / "requirements.lock").read_text(
        encoding="utf-8"
    )
    assert installed_bpod_core
    assert f"bpod-core=={installed_bpod_core}" in lock_text
    assert provenance["package_versions"]["numpy"]
    assert provenance["python_version"]
    assert provenance["python_executable"]
    assert provenance["requirements_lock_sha256"]
    assert provenance["streamdio_driver"]["module"] == "pybpod_streamDIO_module"
    assert header["payload"]["git_commit"] == provenance["git_commit"]
    assert header["payload"]["git_dirty"] == provenance["git_dirty"]
    assert all(trial.reward_volume_ul is None for trial in session.recorder.trials)


def test_extracted_runners_preserve_trial_records_and_event_stream(tmp_path):
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    config = SessionConfig(n_trials=3, random_seed=23, results_dir=str(tmp_path))
    config.camera.record_video = False
    config.frame2ttl.enabled = False
    shorten_for_simulation(config)
    session = SimulatedHFDecisionSession(
        config,
        SimulationOptions(choice_mode="alternate", frames_to_choice=2),
    )

    session.connect()
    session.run()

    trial_records = [
        (
            trial.trial_index,
            trial.star_side,
            trial.circle_side,
            trial.choice,
            trial.chosen_shape,
            trial.choice_side,
            trial.matched,
            trial.responded,
            trial.reward_duration_s,
            trial.trial_mode,
            trial.forced_shape,
            trial.modality,
            trial.reward_delivered,
            trial.bpod_state_entries,
        )
        for trial in session.recorder.trials
    ]
    assert trial_records == [
        (1, "left", "right", "cw", "star", "left", True, True, 0.08, "free", None, "force", True, 5),
        (2, "right", "left", "cw", "circle", "left", False, True, 0.04, "free", None, "force", True, 5),
        (3, "right", "left", "timeout", None, None, False, False, 0.0, "free", None, "force", False, 4),
    ]

    session_start = [
        "session_start",
        "session_metadata",
        "streamdio_streaming_started",
        "all_off_startup_verified",
        "white_noise_ready",
        "cap_touch_power_on",
        "streamdio_camera_triggers_started",
        "hardware_connected",
    ]
    responded_trial = [
        "trial_start",
        "fsm_definition_added",
        "bpod_state_marker",
        "bpod_state_marker",
        "stimulus_on",
        "bpod_state_marker",
        "choice_detected",
        "stimulus_off",
        "bpod_state_marker",
        "bpod_state_marker",
        "bpod_trial_complete",
        "choice_loop_stats",
        "trial_end",
        "iti_start",
        "iti_end",
    ]
    timeout_trial = [
        "trial_start",
        "fsm_definition_added",
        "bpod_state_marker",
        "bpod_state_marker",
        "stimulus_on",
        "bpod_state_marker",
        "white_noise_start",
        "stimulus_off",
        "choice_timeout",
        "bpod_state_marker",
        "white_noise_end",
        "bpod_trial_complete",
            "choice_loop_stats",
            "trial_end",
            "iti_start",
            "iti_end",
    ]
    session_end = [
        "task_outputs_cleanup_verified",
        "speaker_gate_cleanup",
        "streamdio_camera_triggers_stopped",
        "all_off_cleanup_verified",
        "session_end",
    ]
    asynchronously_observed_outputs = {
        "output_actual",
        "force_on",
        "force_off",
        "pump_start",
        "pump_end",
        # Simulated StreamDIO inputs arrive on the same callback thread as live
        # inputs. Their order relative to host-render/log events is deliberately
        # not a scheduler contract.
            "lick_edge",
            "lick",
            "timing_heartbeat",
        }
    stable_event_stream = [
        event.event_type
        for event in session.recorder.events
        if event.event_type not in asynchronously_observed_outputs
    ]
    assert stable_event_stream == (
        session_start
        + responded_trial
        + responded_trial
        + timeout_trial
        + session_end
    )
    event_types = [event.event_type for event in session.recorder.events]
    assert event_types.count("lick_edge") == 3
    assert event_types.count("lick") == 3
    bpod_outputs = [
        operation[1:4]
        for operation in session.relay.ops
        if operation[0] == "bpod_output"
    ]
    assert bpod_outputs == [
        ("cw_force_on", 3, 1),
        ("cw_force_off", 3, 0),
        ("pump_on", 1, 1),
        ("pump_off", 1, 0),
        ("ccw_force_on", 5, 1),
        ("ccw_force_off", 5, 0),
        ("pump_on", 1, 1),
        ("pump_off", 1, 0),
        ("ccw_force_on", 5, 1),
        ("ccw_force_off", 5, 0),
    ]
    host_task_ops = [
        operation[0]
        for operation in session.relay.ops
        if operation[0] != "bpod_output"
    ]
    assert "pump_on" not in host_task_ops
    assert "force" not in host_task_ops
    assert host_task_ops.count("pump_off") == 2
    assert host_task_ops.count("force_off") == 2


def test_blocked_side_schedule_is_balanced_reproducible_and_capped():
    config = SessionConfig(n_trials=200, random_seed=71)
    config.task.side_balancing = "blocked"
    config.task.max_consecutive_side = 4

    first = HFDecisionSession(config)
    second_config = SessionConfig(n_trials=200, random_seed=71)
    second_config.task.side_balancing = "blocked"
    second_config.task.max_consecutive_side = 4
    second = HFDecisionSession(second_config)
    first_sides = [spec["star_side"] for spec in first._trial_schedule]
    second_sides = [spec["star_side"] for spec in second._trial_schedule]

    assert first_sides == second_sides
    assert abs(first_sides.count("left") - first_sides.count("right")) <= 1
    assert _max_run(first_sides) <= 4
    assert [first._choose_star_side(index) for index in range(1, 201)] == first_sides


def test_bernoulli_side_schedule_honors_explicit_run_cap():
    config = SessionConfig(n_trials=200, random_seed=72)
    config.task.max_consecutive_side = 3

    session = HFDecisionSession(config)
    sides = [spec["star_side"] for spec in session._trial_schedule]

    assert _max_run(sides) <= 3
    assert [session._choose_star_side(index) for index in range(1, 201)] == sides


def test_forced_trial_schedule_respects_requested_run_cap():
    config = SessionConfig(n_trials=24, random_seed=19)
    config.task.n_free_trials = 12
    config.task.n_forced_trials = 12
    config.task.max_consecutive_forced = 2

    session = HFDecisionSession(config)
    modes = [spec["trial_mode"] for spec in session._trial_schedule]

    assert modes.count("free") == 12
    assert modes.count("forced") == 12
    forced_runs = [sum(1 for _ in group) for mode, group in groupby(modes) if mode == "forced"]
    assert max(forced_runs) <= 2


def test_simulated_session_annotates_reward_volume_without_changing_pump_time(tmp_path):
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    calibration_path = tmp_path / "valve.json"
    calibration = save_valve_calibration(
        fit_valve_calibration(
            [
                ValveMeasurement(0.04, 20.0, 10),
                ValveMeasurement(0.08, 40.0, 10),
            ]
        ),
        calibration_path,
    )
    config = SessionConfig(n_trials=3, random_seed=23, results_dir=str(tmp_path / "sessions"))
    config.camera.record_video = False
    config.reward.calibration_path = str(calibration_path)
    config.welfare.subjects_dir = str(tmp_path / "subjects")
    shorten_for_simulation(config)
    session = SimulatedHFDecisionSession(
        config,
        SimulationOptions(choice_mode="alternate", frames_to_choice=2),
    )

    session.connect()
    session.run()

    assert [trial.reward_duration_s for trial in session.recorder.trials] == [0.08, 0.04, 0.0]
    assert [trial.reward_volume_ul for trial in session.recorder.trials] == [4.0, 2.0, None]
    assert session.recorder.metadata["reward_calibration"]["sha256"] == calibration.sha256
    assert sum(
        event.event_type == "reward_volume_annotated"
        for event in session.recorder.events
    ) == 2
    with np.load(session.recorder.base_path.with_suffix(".npz"), allow_pickle=True) as data:
        volumes = data["trials"]["reward_volume_ul"]
        assert volumes[:2].tolist() == [4.0, 2.0]
        assert np.isnan(volumes[2])


def test_simulated_centering_requires_wheel_motion_and_rewards_at_center(tmp_path):
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    config = SessionConfig(
        phase=TrainingPhase.CENTERING,
        n_trials=1,
        random_seed=1,
        results_dir=str(tmp_path),
    )
    config.camera.save_dir = str(tmp_path / "videos")
    shorten_for_simulation(config)
    session = SimulatedHFDecisionSession(config, SimulationOptions(choice_mode="matched"))

    session.connect()
    summary = session.run()

    trial = session.recorder.trials[0]
    event_types = [event.event_type for event in session.recorder.events]
    assert summary.n_trials_completed == 1
    assert trial.choice == "centering"
    assert trial.responded
    assert trial.reward_duration_s == config.reward.matched_pump_s
    assert abs(trial.final_position_deg) <= config.centering_threshold_deg
    assert abs(trial.net_pulses) > 0
    assert "pump_start" in event_types
    bpod_outputs = [
        operation[1]
        for operation in session.relay.ops
        if operation[0] == "bpod_output"
    ]
    expected_force = "cw" if trial.force_direction == "cw" else "ccw"
    assert bpod_outputs == [
        f"{expected_force}_force_on",
        f"{expected_force}_force_off",
        "pump_on",
        "pump_off",
    ]
    host_task_ops = [
        operation[0]
        for operation in session.relay.ops
        if operation[0] != "bpod_output"
    ]
    assert "pump_on" not in host_task_ops
    assert "force" not in host_task_ops


def test_simulated_frame2ttl_barcode_decodes_when_quiescence_allows_full_code(tmp_path):
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    config = SessionConfig(n_trials=1, results_dir=str(tmp_path))
    config.camera.save_dir = str(tmp_path / "videos")
    shorten_for_simulation(config)
    config.timing.quiescence_s = 3.0
    session = SimulatedHFDecisionSession(
        config,
        SimulationOptions(choice_mode="matched", frames_to_choice=4),
    )

    session.connect()
    summary = session.run()

    trial = session.recorder.trials[0]
    assert summary.n_trials_completed == 1
    assert trial.frame2ttl_barcode_complete
    assert trial.frame2ttl_barcode_decoded
    assert trial.frame2ttl_barcode_match
    assert trial.frame2ttl_first_ttl_timestamp_us is not None
    assert trial.frame2ttl_first_ttl_latency_from_bpod_s is not None
    assert any(
        event.event_type == "frame2ttl_edge" for event in session.recorder.events
    )


def test_simulated_video_normalizes_relative_output_path(tmp_path, monkeypatch):
    from hf_dm.session.simulated import SimulatedVideoRecorder

    monkeypatch.chdir(tmp_path)
    recorder = SimulatedVideoRecorder(
        Path("relative-videos"), "mouse", "decision_making"
    )

    assert recorder.session_dir.is_absolute()
    assert recorder.start().is_dir()


def test_simulated_frame2ttl_diagnostic_counts_sync_square_edges(tmp_path):
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    config = SessionConfig(n_trials=1, results_dir=str(tmp_path))
    config.camera.save_dir = str(tmp_path / "videos")
    shorten_for_simulation(config)
    session = SimulatedHFDecisionSession(
        config,
        SimulationOptions(choice_mode="matched", frames_to_choice=4),
    )

    session.connect()
    summary = session.frame2ttl_diagnostic(n_flips=4, hold_s=0.01, settle_s=0.0)

    assert summary.end_reason == "frame2ttl_diagnostic"
    diagnostic_end = [
        event
        for event in session.recorder.events
        if event.event_type == "frame2ttl_diagnostic_end"
    ][-1]
    assert diagnostic_end.payload["detected"]
    assert diagnostic_end.payload["edges_seen"] >= 4
    assert diagnostic_end.payload["states_seen"] == [0, 1]


def test_simulated_frame2ttl_barcode_diagnostic_decodes_all_cases(tmp_path):
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    config = SessionConfig(n_trials=1, results_dir=str(tmp_path))
    config.camera.save_dir = str(tmp_path / "videos")
    shorten_for_simulation(config)
    config.display.fps = 120
    session = SimulatedHFDecisionSession(
        config,
        SimulationOptions(choice_mode="matched", frames_to_choice=4),
    )

    session.connect()
    summary = session.frame2ttl_barcode_diagnostic(repetitions=1, settle_s=0.0)

    assert summary.end_reason == "frame2ttl_barcode_diagnostic"
    diagnostic_end = [
        event
        for event in session.recorder.events
        if event.event_type == "frame2ttl_barcode_diagnostic_end"
    ][-1]
    assert diagnostic_end.payload["cases_run"] == 6
    assert diagnostic_end.payload["decoded_count"] == 6
    assert diagnostic_end.payload["match_count"] == 6
    assert diagnostic_end.payload["all_match"]


def test_simulated_bpod_opcode_diagnostic_reports_marker_sequence(tmp_path):
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    config = SessionConfig(n_trials=1, results_dir=str(tmp_path))
    config.camera.save_dir = str(tmp_path / "videos")
    shorten_for_simulation(config)
    session = SimulatedHFDecisionSession(
        config,
        SimulationOptions(choice_mode="matched", frames_to_choice=4),
    )

    session.connect()
    summary = session.bpod_opcode_diagnostic(hold_s=0.001)

    assert summary.end_reason == "bpod_opcode_diagnostic"
    diagnostic_end = [
        event
        for event in session.recorder.events
        if event.event_type == "bpod_opcode_diagnostic_end"
    ][-1]
    assert diagnostic_end.payload["matched"]
    assert diagnostic_end.payload["seen_ids"] == diagnostic_end.payload["expected_ids"]
    assert diagnostic_end.payload["markers_seen"] == len(diagnostic_end.payload["expected_ids"])


def test_simulated_forced_risk_star_can_omit_pump_and_records_metadata(tmp_path):
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    config = SessionConfig(n_trials=1, random_seed=1, results_dir=str(tmp_path))
    config.camera.save_dir = str(tmp_path / "videos")
    config.task.modality = "risk"
    config.task.n_free_trials = 0
    config.task.n_forced_trials = 1
    config.task.risk_star_probability = 0.0
    shorten_for_simulation(config)
    session = SimulatedHFDecisionSession(
        config,
        SimulationOptions(choice_mode="matched", frames_to_choice=4),
    )

    session.connect()
    summary = session.run()

    trial = session.recorder.trials[0]
    event_types = [event.event_type for event in session.recorder.events]
    assert summary.n_trials_completed == 1
    assert trial.trial_mode == "forced"
    assert trial.forced_shape == "star"
    assert trial.modality == "risk"
    assert trial.chosen_shape == "star"
    assert trial.reward_duration_s == 0.0
    assert not trial.reward_delivered
    assert trial.risk_star_probability == 0.0
    assert "pump_omitted" in event_types
    assert "pump_start" not in event_types
    assert "force_on" not in event_types
    assert "force_skipped" in event_types
    assert trial.force_direction is None
    assert trial.force_output_direction is None
    assert all(op[0] != "force" for op in session.relay.ops)
    assert session.recorder.metadata["task_summary"]["symbols"]["star"]["effect"] == "risk"
    assert session.recorder.metadata["task_summary"]["symbols"]["star"]["reward_probability"] == 0.0


def test_simulated_force_modality_does_not_force_forced_circle(tmp_path):
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    config = SessionConfig(n_trials=2, random_seed=1, results_dir=str(tmp_path))
    config.camera.save_dir = str(tmp_path / "videos")
    config.task.modality = "force"
    config.task.n_free_trials = 0
    config.task.n_forced_trials = 2
    shorten_for_simulation(config)
    session = SimulatedHFDecisionSession(
        config,
        SimulationOptions(choice_mode="matched", frames_to_choice=4),
    )

    session.connect()
    summary = session.run()

    assert summary.n_trials_completed == 2
    by_shape = {trial.forced_shape: trial for trial in session.recorder.trials}
    star_trial = by_shape["star"]
    circle_trial = by_shape["circle"]
    assert star_trial.force_output_direction in ("cw", "ccw")
    assert star_trial.force_direction == star_trial.star_direction
    assert circle_trial.chosen_shape == "circle"
    assert circle_trial.force_direction is None
    assert circle_trial.force_output_direction is None
    assert circle_trial.choice_target_direction == circle_trial.circle_direction

    force_on_trials = [
        event.payload["trial_index"]
        for event in session.recorder.events
        if event.event_type == "force_on"
    ]
    skipped_trials = [
        event.payload["trial_index"]
        for event in session.recorder.events
        if event.event_type == "force_skipped"
    ]
    assert force_on_trials == [star_trial.trial_index]
    assert skipped_trials == [circle_trial.trial_index]
