"""Command-line entry: hf-dm-run --config configs/dm_50.yaml"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable

from hf_dm.certification import AnimalUseGateError, assert_animal_use_ready
from hf_dm.certification_gui import CertificationDialogResult, run_certification_dialog
from hf_dm.certification_workflow import (
    append_gui_audit,
    apply_operator_state,
    load_operator_state,
)
from hf_dm.config import (
    SessionConfig,
    config_to_dict,
    task_design_summary,
    validate_session_config,
)
from hf_dm.session.cap_touch_diagnostic import run_cap_touch_diagnostic
from hf_dm.session.controller import HFDecisionSession
from hf_dm.session.hardware_self_test import HardwareSelfTestSession, shorten_for_hardware_self_test
from hf_dm.session.simulated import (
    SimulatedHFDecisionSession,
    SimulationOptions,
    shorten_for_simulation,
)


PROFILE_COUNTS: dict[str, tuple[int, int]] = {
    "custom": (50, 0),
    "50 free": (50, 0),
    "100 free": (100, 0),
    "200 free": (200, 0),
    "50 forced": (0, 50),
    "100 forced": (0, 100),
    "200 forced": (0, 200),
    "50 50/50": (25, 25),
    "100 50/50": (50, 50),
    "200 50/50": (100, 100),
}


@dataclass
class SessionSetup:
    selected_mouse_dir: str
    modality: str
    profile_name: str
    n_free_trials: int
    n_forced_trials: int
    session_name: str
    session_dir: str
    lick_source: str = "capacitive"
    optical_clock: str = "bpod"


def _default_session_name(modality: str, profile_name: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_profile = profile_name.replace("/", "-").replace(" ", "_")
    return f"{stamp}_{modality}_{safe_profile}"


def _validated_session_name(value: str) -> str:
    name = str(value).strip()
    if not name or name in {".", ".."} or Path(name).name != name:
        raise ValueError("session name must be a non-empty folder name, not a path")
    return name


def _setup_from_values(
    *,
    mouse_dir: Path,
    modality: str = "force",
    profile_name: str = "custom",
    n_free_trials: int | None = None,
    n_forced_trials: int | None = None,
    session_name: str | None = None,
    lick_source: str = "capacitive",
    optical_clock: str = "bpod",
) -> SessionSetup:
    if lick_source not in {"capacitive", "optical"}:
        raise ValueError("lick source must be capacitive or optical")
    if optical_clock not in {"bpod", "streamdio"}:
        raise ValueError("optical lick clock must be bpod or streamdio")
    modality = _normalize_modality(modality)
    profile_name = profile_name if profile_name in PROFILE_COUNTS else "custom"
    profile_free, profile_forced = PROFILE_COUNTS[profile_name]
    free = profile_free if n_free_trials is None else max(0, int(n_free_trials))
    forced = profile_forced if n_forced_trials is None else max(0, int(n_forced_trials))
    name = _validated_session_name(
        session_name or _default_session_name(modality, profile_name)
    )
    selected_mouse_dir = Path(mouse_dir)
    return SessionSetup(
        selected_mouse_dir=str(selected_mouse_dir),
        modality=modality,
        profile_name=profile_name,
        n_free_trials=free,
        n_forced_trials=forced,
        session_name=name,
        session_dir=str(selected_mouse_dir / name),
        lick_source=lick_source,
        optical_clock=optical_clock,
    )


def _run_session_setup_dialog(*, lick_source: str = "capacitive", optical_clock: str = "bpod") -> SessionSetup:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.withdraw()
    mouse_dir = filedialog.askdirectory(
        parent=root,
        title="Select mouse folder for this session",
        mustexist=True,
    )
    if not mouse_dir:
        root.destroy()
        raise RuntimeError("Session setup cancelled: no mouse folder selected")

    result: dict[str, SessionSetup] = {}
    root.deiconify()
    root.title("HF DM Session Setup")
    root.geometry("460x410")
    root.resizable(False, False)

    modality_var = tk.StringVar(value="force")
    lick_source_var = tk.StringVar(value=lick_source)
    optical_clock_var = tk.StringVar(value=optical_clock)
    profile_var = tk.StringVar(value="50 free")
    free_var = tk.StringVar(value="50")
    forced_var = tk.StringVar(value="0")
    session_name_var = tk.StringVar(value=_default_session_name("force", "50 free"))

    def update_profile(*_args) -> None:
        profile = profile_var.get()
        free, forced = PROFILE_COUNTS.get(profile, PROFILE_COUNTS["custom"])
        if profile != "custom":
            free_var.set(str(free))
            forced_var.set(str(forced))
        session_name_var.set(_default_session_name(modality_var.get(), profile))

    def update_modality(*_args) -> None:
        session_name_var.set(_default_session_name(modality_var.get(), profile_var.get()))

    def accept() -> None:
        try:
            result["setup"] = _setup_from_values(
                mouse_dir=Path(mouse_dir),
                modality=modality_var.get(),
                profile_name=profile_var.get(),
                n_free_trials=int(free_var.get()),
                n_forced_trials=int(forced_var.get()),
                session_name=session_name_var.get().strip() or None,
                lick_source=lick_source_var.get(),
                optical_clock=optical_clock_var.get(),
            )
        except Exception as exc:
            messagebox.showerror("Invalid session setup", str(exc), parent=root)
            return
        root.quit()

    def cancel() -> None:
        root.quit()

    root.columnconfigure(1, weight=1)
    pad = {"padx": 12, "pady": 6}
    ttk.Label(root, text="Mouse folder").grid(row=0, column=0, sticky="w", **pad)
    ttk.Label(root, text=str(mouse_dir)).grid(row=0, column=1, sticky="w", **pad)
    ttk.Label(root, text="Modality").grid(row=1, column=0, sticky="w", **pad)
    ttk.Combobox(
        root,
        textvariable=modality_var,
        values=("force", "risk", "delay"),
        state="readonly",
    ).grid(row=1, column=1, sticky="ew", **pad)
    ttk.Label(root, text="Profile").grid(row=2, column=0, sticky="w", **pad)
    ttk.Combobox(
        root,
        textvariable=profile_var,
        values=tuple(PROFILE_COUNTS),
        state="readonly",
    ).grid(row=2, column=1, sticky="ew", **pad)
    ttk.Label(root, text="Free trials").grid(row=3, column=0, sticky="w", **pad)
    ttk.Entry(root, textvariable=free_var).grid(row=3, column=1, sticky="ew", **pad)
    ttk.Label(root, text="Forced trials").grid(row=4, column=0, sticky="w", **pad)
    ttk.Entry(root, textvariable=forced_var).grid(row=4, column=1, sticky="ew", **pad)
    ttk.Label(root, text="Session name").grid(row=5, column=0, sticky="w", **pad)
    ttk.Entry(root, textvariable=session_name_var).grid(row=5, column=1, sticky="ew", **pad)

    buttons = ttk.Frame(root)
    ttk.Label(root, text="Lick sensor").grid(row=6, column=0, sticky="w", **pad)
    ttk.Combobox(
        root, textvariable=lick_source_var, values=("capacitive", "optical"),
        state="readonly",
    ).grid(row=6, column=1, sticky="ew", **pad)
    ttk.Label(root, text="Optical timestamp clock").grid(row=7, column=0, sticky="w", **pad)
    ttk.Combobox(
        root, textvariable=optical_clock_var, values=("bpod", "streamdio"), state="readonly",
    ).grid(row=7, column=1, sticky="ew", **pad)
    buttons.grid(row=8, column=0, columnspan=2, sticky="e", padx=12, pady=18)
    ttk.Button(buttons, text="Cancel", command=cancel).pack(side="right", padx=(8, 0))
    ttk.Button(buttons, text="Start", command=accept).pack(side="right")

    profile_var.trace_add("write", update_profile)
    modality_var.trace_add("write", update_modality)
    root.protocol("WM_DELETE_WINDOW", cancel)
    root.mainloop()
    root.destroy()

    if "setup" not in result:
        raise RuntimeError("Session setup cancelled")
    return result["setup"]


def _apply_session_setup(config: SessionConfig, setup: SessionSetup) -> None:
    config.licks.source = setup.lick_source
    config.licks.optical_clock = setup.optical_clock
    session_dir = Path(setup.session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "videos").mkdir(parents=True, exist_ok=True)
    config.subject_id = Path(setup.selected_mouse_dir).name or config.subject_id
    config.task.modality = setup.modality
    config.task.profile_name = setup.profile_name
    config.task.n_free_trials = int(setup.n_free_trials)
    config.task.n_forced_trials = int(setup.n_forced_trials)
    config.task.selected_mouse_dir = setup.selected_mouse_dir
    config.task.session_name = setup.session_name
    config.task.session_dir = str(session_dir)
    config.n_trials = int(setup.n_free_trials) + int(setup.n_forced_trials)
    config.results_dir = str(session_dir)
    config.camera.save_dir = str(session_dir / "videos")
    metadata = {
        "setup": asdict(setup),
        "task_summary": task_design_summary(config),
        "config": config_to_dict(config),
        "created_wall_time_s": datetime.now().timestamp(),
    }
    (session_dir / "session_info.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def _record_cli_override(
    config: SessionConfig,
    *,
    source: str,
    values: dict[str, object],
) -> None:
    if not values:
        return
    config.session_export.cli_overrides.append(
        {
            "source": source,
            "values": json.loads(json.dumps(values, default=str)),
            "created_wall_time_s": datetime.now().timestamp(),
        }
    )


def _normalize_modality(value: str) -> str:
    modality = str(value or "force").lower()
    if modality not in {"force", "risk", "delay"}:
        raise ValueError("modality must be force, risk, or delay")
    return modality


def _load_saved_operator_state(config: SessionConfig) -> None:
    if not config.animal_use.enabled:
        return
    try:
        values = load_operator_state(config)
        apply_operator_state(config, values)
    except Exception as exc:
        # A corrupt overlay must never prevent the remediation window from
        # opening. It will display the same error and require corrected values.
        append_gui_audit(
            config,
            "operator_state_load_failed",
            detail=f"{type(exc).__name__}: {exc}",
        )


def _remediate_animal_use_gate(
    config: SessionConfig,
    *,
    operator_confirmed: bool,
    dialog_enabled: bool,
    report: dict[str, Any] | None = None,
    dialog_fn: Callable[
        [SessionConfig, dict[str, Any]], CertificationDialogResult
    ] = run_certification_dialog,
) -> bool:
    """Return confirmation only after the production gate passes."""

    try:
        assert_animal_use_ready(config, operator_confirmed=operator_confirmed)
        return operator_confirmed
    except AnimalUseGateError as exc:
        blocked_report: dict[str, Any] = report or exc.report
        if not dialog_enabled:
            raise
    try:
        outcome = dialog_fn(config, blocked_report)
    except Exception:
        # GUI/display failures preserve the original fail-closed behavior.
        raise AnimalUseGateError(blocked_report)
    if not outcome.continue_session or not outcome.operator_confirmed:
        raise AnimalUseGateError(blocked_report)
    assert_animal_use_ready(config, operator_confirmed=True)
    return True


def _validate_uncertified_bench_request(
    *,
    allow_uncertified: bool,
    confirm_no_animal: bool,
    conflicting_mode: bool,
) -> bool:
    """Validate and identify an explicitly labeled no-animal full-task run."""

    if confirm_no_animal and not allow_uncertified:
        raise ValueError("--confirm-no-animal requires --allow-uncertified")
    if not allow_uncertified:
        return False
    if not confirm_no_animal:
        raise ValueError("--allow-uncertified requires --confirm-no-animal")
    if conflicting_mode:
        raise ValueError(
            "--allow-uncertified is only valid for a normal full-task run, not another "
            "simulation, preflight, self-test, or diagnostic mode"
        )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HF decision-making session (bpod-core + StreamDIO)")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML session config (see configs/)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--confirm-animal-use",
        action="store_true",
        help=(
            "One-run confirmation that the operator completed the animal safety and "
            "welfare checklist. Required when animal_use.enabled is true."
        ),
    )
    parser.add_argument(
        "--allow-uncertified",
        action="store_true",
        help=(
            "Run the complete task in explicitly labeled bench mode without a rig certificate. "
            "Requires --confirm-no-animal and must never be used for animal acquisition."
        ),
    )
    parser.add_argument(
        "--confirm-no-animal",
        action="store_true",
        help="Confirm no animal is present for --allow-uncertified bench operation.",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run a hardware-free simulation that exercises the live display/task loop.",
    )
    parser.add_argument(
        "--lick-source", choices=("capacitive", "optical"),
        help="Lick sensor for this session: StreamDIO capacitive touch or Bpod optical port.",
    )
    parser.add_argument(
        "--lick-clock", choices=("bpod", "streamdio"),
        help="Optical timestamps: original Bpod edges or StreamDIO receipt of forwarded port events.",
    )
    parser.add_argument(
        "--lick-port", type=int,
        help="Bpod behavior port for the optical lickometer (default: configured port, normally 1).",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Connect streams/camera and save a short no-trial hardware preflight record.",
    )
    parser.add_argument(
        "--hardware-self-test",
        action="store_true",
        help=(
            "Run three production Bpod-FSM trials with command-driven wheel choices. "
            "WARNING: activates real force, pump, audible speaker noise, camera, and outputs."
        ),
    )
    parser.add_argument(
        "--cap-touch-diagnostic",
        action="store_true",
        help=(
            "Run a StreamDIO-only cap-touch check: powers ch13/pin15, polls "
            "digitizer status on ch14/pin25, and reports lick edges."
        ),
    )
    parser.add_argument(
        "--frame2ttl-diagnostic",
        action="store_true",
        help=(
            "Run a no-trial Frame2TTL check: flashes the configured sync square "
            "and records DI05/pin6 TTL edges."
        ),
    )
    parser.add_argument(
        "--frame2ttl-barcode-diagnostic",
        action="store_true",
        help=(
            "Run a no-trial Frame2TTL barcode check: displays representative "
            "trial-start count barcodes and reports DI decode/match accuracy."
        ),
    )
    parser.add_argument(
        "--frame2ttl-barcode-repetitions",
        type=int,
        default=1,
        help="Number of times to repeat all barcode cases for --frame2ttl-barcode-diagnostic.",
    )
    parser.add_argument(
        "--bpod-opcode-diagnostic",
        action="store_true",
        help=(
            "Run a no-output Bpod serial opcode check: sends StreamDIO state "
            "marker opcodes and reports bpod_state timestamps."
        ),
    )
    parser.add_argument(
        "--bpod-opcode-hold-s",
        type=float,
        default=0.01,
        help="Seconds to hold each marker state for --bpod-opcode-diagnostic.",
    )
    parser.add_argument(
        "--frame2ttl-flips",
        type=int,
        default=20,
        help="Number of square flips for --frame2ttl-diagnostic.",
    )
    parser.add_argument(
        "--frame2ttl-hold-s",
        type=float,
        default=0.25,
        help="Seconds to hold each square state for --frame2ttl-diagnostic.",
    )
    parser.add_argument(
        "--frame2ttl-settle-s",
        type=float,
        default=0.5,
        help="Seconds to hold black before/after --frame2ttl-diagnostic flashes.",
    )
    parser.add_argument(
        "--cap-touch-duration",
        type=float,
        default=15.0,
        help="Seconds to run --cap-touch-diagnostic.",
    )
    parser.add_argument(
        "--cap-touch-poll-hz",
        type=float,
        default=10.0,
        help="Status polling rate for --cap-touch-diagnostic.",
    )
    parser.add_argument(
        "--cap-touch-raw-analog",
        action="store_true",
        help=(
            "For --cap-touch-diagnostic, temporarily stream cap-touch ch14/pin25 "
            "as the single analog input instead of force."
        ),
    )
    parser.add_argument(
        "--cap-touch-min-high-threshold",
        type=int,
        default=None,
        help="Runtime cap-touch minimum high threshold for --cap-touch-diagnostic.",
    )
    parser.add_argument(
        "--cap-touch-delta-high",
        type=int,
        default=None,
        help="Runtime cap-touch high delta for --cap-touch-diagnostic.",
    )
    parser.add_argument(
        "--cap-touch-delta-low",
        type=int,
        default=None,
        help="Runtime cap-touch release delta for --cap-touch-diagnostic.",
    )
    parser.add_argument(
        "--cap-touch-report",
        type=Path,
        default=None,
        help="Write the cap-touch diagnostic JSON report to this path.",
    )
    parser.add_argument(
        "--preflight-duration",
        type=float,
        default=2.0,
        help="Seconds to hold the preflight streaming/camera check.",
    )
    parser.add_argument(
        "--simulate-headless",
        action="store_true",
        help="Use SDL dummy video driver for simulation checks without opening a window.",
    )
    parser.add_argument(
        "--simulate-choice-mode",
        choices=["matched", "opposite", "timeout", "cw", "ccw", "alternate"],
        default="alternate",
        help="Choice pattern for --simulate.",
    )
    parser.add_argument(
        "--simulate-trials",
        type=int,
        default=None,
        help="Override n_trials and shorten timing for simulation.",
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Force-enable the operator monitor GUI, overriding the YAML monitor block.",
    )
    parser.add_argument(
        "--no-monitor",
        action="store_true",
        help="Disable the operator monitor GUI, overriding the YAML monitor block.",
    )
    parser.add_argument(
        "--no-session-dialog",
        action="store_true",
        help="Skip mouse/session setup dialogs and use YAML or explicit session args.",
    )
    parser.add_argument(
        "--no-certification-dialog",
        action="store_true",
        help=(
            "Keep animal-use certificate failures non-interactive. Intended for automation; "
            "normal operator sessions open the remediation window."
        ),
    )
    parser.add_argument(
        "--session-mouse-dir",
        type=Path,
        default=None,
        help="Mouse folder to save this session under; bypasses folder selection.",
    )
    parser.add_argument(
        "--session-modality",
        choices=["force", "risk", "delay"],
        default=None,
        help="Task modality for this session.",
    )
    parser.add_argument(
        "--session-profile",
        choices=tuple(PROFILE_COUNTS),
        default=None,
        help="Predefined free/forced profile.",
    )
    parser.add_argument(
        "--free-trials",
        type=int,
        default=None,
        help="Manual number of free-choice trials.",
    )
    parser.add_argument(
        "--forced-trials",
        type=int,
        default=None,
        help="Manual number of forced-choice trials.",
    )
    parser.add_argument(
        "--session-name",
        type=str,
        default=None,
        help="Override generated session folder name.",
    )
    args = parser.parse_args()
    if args.hardware_self_test and args.simulate:
        parser.error("--hardware-self-test cannot be combined with --simulate")
    if args.hardware_self_test and args.preflight:
        parser.error("--hardware-self-test cannot be combined with --preflight")
    frame2ttl_modes = [
        args.frame2ttl_diagnostic,
        args.frame2ttl_barcode_diagnostic,
        args.bpod_opcode_diagnostic,
    ]
    if sum(1 for enabled in frame2ttl_modes if enabled) > 1:
        parser.error(
            "--frame2ttl-diagnostic, --frame2ttl-barcode-diagnostic, and "
            "--bpod-opcode-diagnostic are mutually exclusive"
        )
    if args.cap_touch_diagnostic and (
        args.simulate
        or args.preflight
        or args.hardware_self_test
        or any(frame2ttl_modes)
    ):
        parser.error(
            "--cap-touch-diagnostic cannot be combined with --simulate, "
            "--preflight, --hardware-self-test, or sync diagnostics"
        )
    if any(frame2ttl_modes) and (args.preflight or args.hardware_self_test):
        parser.error(
            "sync diagnostics cannot be combined with --preflight or "
            "--hardware-self-test"
        )
    try:
        uncertified_bench = _validate_uncertified_bench_request(
            allow_uncertified=args.allow_uncertified,
            confirm_no_animal=args.confirm_no_animal,
            conflicting_mode=bool(
                args.simulate
                or args.preflight
                or args.hardware_self_test
                or args.cap_touch_diagnostic
                or any(frame2ttl_modes)
            ),
        )
    except ValueError as exc:
        parser.error(str(exc))

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = SessionConfig.from_yaml(args.config)
    if args.lick_source is not None:
        config.licks.source = args.lick_source
    if args.lick_port is not None:
        config.licks.bpod_port = args.lick_port
    if args.lick_clock is not None:
        config.licks.optical_clock = args.lick_clock
    if args.lick_source is not None or args.lick_port is not None or args.lick_clock is not None:
        _record_cli_override(config, source="hf-dm-run lick sensor arguments", values={
            "licks.source": config.licks.source, "licks.bpod_port": config.licks.bpod_port,
            "licks.optical_clock": config.licks.optical_clock,
        })
    config.session_export.source_config_path = str(args.config)
    if args.monitor and args.no_monitor:
        parser.error("--monitor cannot be combined with --no-monitor")
    if args.monitor:
        config.monitor.enabled = True
        _record_cli_override(
            config,
            source="hf-dm-run --monitor",
            values={"monitor.enabled": True},
        )
    if args.no_monitor:
        config.monitor.enabled = False
        _record_cli_override(
            config,
            source="hf-dm-run --no-monitor",
            values={"monitor.enabled": False},
        )
    if any(frame2ttl_modes):
        config.camera.record_video = False
        config.camera.required = False
        config.speaker.required = False
        mode = (
            "frame2ttl_diagnostic"
            if args.frame2ttl_diagnostic
            else "frame2ttl_barcode_diagnostic"
            if args.frame2ttl_barcode_diagnostic
            else "bpod_opcode_diagnostic"
        )
        _record_cli_override(
            config,
            source=f"hf-dm-run --{mode.replace('_', '-')}",
            values={
                "diagnostic_mode": mode,
                "camera.record_video": False,
                "camera.required": False,
                "speaker.required": False,
            },
        )
    setup_requested = (
        args.session_mouse_dir is not None
        or args.session_modality is not None
        or args.session_profile is not None
        or args.free_trials is not None
        or args.forced_trials is not None
        or args.session_name is not None
    )
    should_show_setup = (
        not args.no_session_dialog
        and not args.simulate
        and not args.preflight
        and not args.hardware_self_test
        and not args.cap_touch_diagnostic
        and not any(frame2ttl_modes)
    )
    if should_show_setup:
        setup = _run_session_setup_dialog(lick_source=config.licks.source, optical_clock=config.licks.optical_clock)
        _apply_session_setup(config, setup)
        _record_cli_override(
            config,
            source="hf-dm-run session setup dialog",
            values={"setup": asdict(setup)},
        )
    elif setup_requested:
        if args.session_mouse_dir is not None:
            setup = _setup_from_values(
                mouse_dir=args.session_mouse_dir,
                modality=args.session_modality or config.task.modality,
                profile_name=args.session_profile or config.task.profile_name,
                n_free_trials=args.free_trials,
                n_forced_trials=args.forced_trials,
                session_name=args.session_name,
                lick_source=config.licks.source,
                optical_clock=config.licks.optical_clock,
            )
            _apply_session_setup(config, setup)
            _record_cli_override(
                config,
                source="hf-dm-run session setup arguments",
                values={"setup": asdict(setup)},
            )
        else:
            session_arg_overrides: dict[str, object] = {}
            if args.session_modality is not None:
                config.task.modality = args.session_modality
                session_arg_overrides["task.modality"] = args.session_modality
            if args.session_profile is not None:
                free, forced = PROFILE_COUNTS[args.session_profile]
                config.task.profile_name = args.session_profile
                config.task.n_free_trials = free
                config.task.n_forced_trials = forced
                config.n_trials = free + forced
                session_arg_overrides["task.profile_name"] = args.session_profile
                session_arg_overrides["task.n_free_trials"] = free
                session_arg_overrides["task.n_forced_trials"] = forced
                session_arg_overrides["n_trials"] = config.n_trials
            if args.free_trials is not None:
                config.task.n_free_trials = max(0, int(args.free_trials))
                session_arg_overrides["task.n_free_trials"] = config.task.n_free_trials
            if args.forced_trials is not None:
                config.task.n_forced_trials = max(0, int(args.forced_trials))
                session_arg_overrides["task.n_forced_trials"] = config.task.n_forced_trials
            if args.free_trials is not None or args.forced_trials is not None:
                free = (
                    config.task.n_free_trials
                    if config.task.n_free_trials is not None
                    else config.n_trials
                )
                config.n_trials = int(free) + int(config.task.n_forced_trials)
                session_arg_overrides["n_trials"] = config.n_trials
            if args.session_name is not None:
                session_name = _validated_session_name(args.session_name)
                session_dir = Path(config.results_dir) / session_name
                session_dir.mkdir(parents=True, exist_ok=True)
                (session_dir / "videos").mkdir(parents=True, exist_ok=True)
                config.task.session_name = session_name
                config.task.session_dir = str(session_dir)
                config.results_dir = str(session_dir)
                config.camera.save_dir = str(session_dir / "videos")
                session_arg_overrides.update(
                    {
                        "task.session_name": session_name,
                        "task.session_dir": str(session_dir),
                        "results_dir": str(session_dir),
                        "camera.save_dir": str(session_dir / "videos"),
                    }
                )
            _record_cli_override(
                config,
                source="hf-dm-run session arguments",
                values=session_arg_overrides,
            )
    if uncertified_bench:
        logging.warning(
            "UNCERTIFIED BENCH RUN: no animal confirmed; certification gate is not applied"
        )
        _record_cli_override(
            config,
            source="hf-dm-run --allow-uncertified --confirm-no-animal",
            values={
                "run_mode": "bench",
                "animal_present": False,
                "certification_required": False,
            },
        )
    elif (
        not config.animal_use.enabled
        and not args.simulate
        and not args.hardware_self_test
        and not args.preflight
        and not args.cap_touch_diagnostic
        and not any(frame2ttl_modes)
    ):
        logging.warning(
            "ANIMAL-USE CERTIFICATION IS DISABLED: this acquisition is labeled bench mode"
        )
        _record_cli_override(
            config,
            source="animal_use.enabled=false",
            values={"run_mode": "bench", "certification_required": False},
        )
    _load_saved_operator_state(config)
    validate_session_config(config)
    if args.cap_touch_diagnostic:
        report = run_cap_touch_diagnostic(
            config,
            duration_s=args.cap_touch_duration,
            poll_hz=args.cap_touch_poll_hz,
            min_high_threshold=args.cap_touch_min_high_threshold,
            delta_high=args.cap_touch_delta_high,
            delta_low=args.cap_touch_delta_low,
            raw_analog=args.cap_touch_raw_analog,
        )
        if args.cap_touch_report is not None:
            args.cap_touch_report.parent.mkdir(parents=True, exist_ok=True)
            args.cap_touch_report.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        return
    if args.simulate:
        if args.simulate_headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            config.monitor.enabled = False
            _record_cli_override(
                config,
                source="hf-dm-run --simulate-headless",
                values={"monitor.enabled": False, "SDL_VIDEODRIVER": "dummy"},
            )
        shorten_for_simulation(config, n_trials=args.simulate_trials)
        _record_cli_override(
            config,
            source="hf-dm-run --simulate",
            values={
                "simulate_choice_mode": args.simulate_choice_mode,
                "simulate_trials": args.simulate_trials,
                "shortened_timing": True,
                "n_trials": config.n_trials,
            },
        )
        session = SimulatedHFDecisionSession(
            config,
            SimulationOptions(choice_mode=args.simulate_choice_mode),
        )
    elif args.hardware_self_test:
        shorten_for_hardware_self_test(config)
        _record_cli_override(
            config,
            source="hf-dm-run --hardware-self-test",
            values={
                "shortened_timing": True,
                "n_trials": config.n_trials,
                "camera.record_video": config.camera.record_video,
                "camera.required": config.camera.required,
            },
        )
        logging.warning(
            "HARDWARE SELF-TEST: activating real force, pump, audible speaker, camera, and outputs"
        )
        session = HardwareSelfTestSession(config)
    else:
        bench_mode = bool(
            args.preflight
            or any(frame2ttl_modes)
            or uncertified_bench
            or not config.animal_use.enabled
        )
        animal_use_confirmed = args.confirm_animal_use
        if not bench_mode and config.animal_use.enabled:
            animal_use_confirmed = _remediate_animal_use_gate(
                config,
                operator_confirmed=animal_use_confirmed,
                dialog_enabled=not args.no_certification_dialog,
            )
        session = HFDecisionSession(
            config,
            run_mode="bench" if bench_mode else "animal",
            animal_use_confirmed=animal_use_confirmed,
        )
    connected = False
    while True:
        try:
            session.connect()
            connected = True
            break
        except AnimalUseGateError as exc:
            session.close(raise_on_error=False)
            is_interactive_animal = (
                not args.simulate
                and not args.hardware_self_test
                and not args.preflight
                and not uncertified_bench
                and not any(frame2ttl_modes)
                and config.animal_use.enabled
            )
            if not is_interactive_animal:
                raise
            animal_use_confirmed = _remediate_animal_use_gate(
                config,
                operator_confirmed=False,
                dialog_enabled=not args.no_certification_dialog,
                report=exc.report,
            )
            session = HFDecisionSession(
                config,
                run_mode="animal",
                animal_use_confirmed=animal_use_confirmed,
            )
        except BaseException:
            session.close(raise_on_error=False)
            raise
    try:
        assert connected
        if args.frame2ttl_diagnostic:
            summary = session.frame2ttl_diagnostic(
                n_flips=args.frame2ttl_flips,
                hold_s=args.frame2ttl_hold_s,
                settle_s=args.frame2ttl_settle_s,
            )
        elif args.frame2ttl_barcode_diagnostic:
            summary = session.frame2ttl_barcode_diagnostic(
                repetitions=args.frame2ttl_barcode_repetitions,
                settle_s=args.frame2ttl_settle_s,
            )
        elif args.bpod_opcode_diagnostic:
            summary = session.bpod_opcode_diagnostic(hold_s=args.bpod_opcode_hold_s)
        elif args.preflight:
            summary = session.preflight(duration_s=args.preflight_duration)
        else:
            summary = session.run()
            if isinstance(session, HardwareSelfTestSession):
                report = session.validate_results()
                print(f"Hardware self-test validated: {json.dumps(report, sort_keys=True)}")
    except BaseException:
        # Preserve the initiating failure; close records and interlocks any
        # independent cleanup failure without replacing the original cause.
        session.close(raise_on_error=False)
        raise
    print(f"Session finished: {summary.end_reason}")
    print(f"Trials completed: {summary.n_trials_completed}/{summary.n_trials_planned}")


if __name__ == "__main__":
    main()
