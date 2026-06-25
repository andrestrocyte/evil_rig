"""Command-line entry: hf-dm-run --config configs/dm_50.yaml"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import logging
import os
from pathlib import Path

from hf_dm.config import SessionConfig, config_to_dict, task_design_summary
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


def _default_session_name(modality: str, profile_name: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_profile = profile_name.replace("/", "-").replace(" ", "_")
    return f"{stamp}_{modality}_{safe_profile}"


def _setup_from_values(
    *,
    mouse_dir: Path,
    modality: str = "force",
    profile_name: str = "custom",
    n_free_trials: int | None = None,
    n_forced_trials: int | None = None,
    session_name: str | None = None,
) -> SessionSetup:
    modality = _normalize_modality(modality)
    profile_name = profile_name if profile_name in PROFILE_COUNTS else "custom"
    profile_free, profile_forced = PROFILE_COUNTS[profile_name]
    free = profile_free if n_free_trials is None else max(0, int(n_free_trials))
    forced = profile_forced if n_forced_trials is None else max(0, int(n_forced_trials))
    name = session_name or _default_session_name(modality, profile_name)
    selected_mouse_dir = Path(mouse_dir)
    return SessionSetup(
        selected_mouse_dir=str(selected_mouse_dir),
        modality=modality,
        profile_name=profile_name,
        n_free_trials=free,
        n_forced_trials=forced,
        session_name=name,
        session_dir=str(selected_mouse_dir / name),
    )


def _run_session_setup_dialog() -> SessionSetup:
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
    root.geometry("420x310")
    root.resizable(False, False)

    modality_var = tk.StringVar(value="force")
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
    buttons.grid(row=6, column=0, columnspan=2, sticky="e", padx=12, pady=18)
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


def _normalize_modality(value: str) -> str:
    modality = str(value or "force").lower()
    if modality not in {"force", "risk", "delay"}:
        raise ValueError("modality must be force, risk, or delay")
    return modality


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
        "--simulate",
        action="store_true",
        help="Run a hardware-free simulation that exercises the live display/task loop.",
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

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = SessionConfig.from_yaml(args.config)
    if args.monitor and args.no_monitor:
        parser.error("--monitor cannot be combined with --no-monitor")
    if args.monitor:
        config.monitor.enabled = True
    if args.no_monitor:
        config.monitor.enabled = False
    if any(frame2ttl_modes):
        config.camera.record_video = False
        config.camera.required = False
        config.speaker.required = False
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
        _apply_session_setup(config, _run_session_setup_dialog())
    elif setup_requested:
        if args.session_mouse_dir is not None:
            setup = _setup_from_values(
                mouse_dir=args.session_mouse_dir,
                modality=args.session_modality or config.task.modality,
                profile_name=args.session_profile or config.task.profile_name,
                n_free_trials=args.free_trials,
                n_forced_trials=args.forced_trials,
                session_name=args.session_name,
            )
            _apply_session_setup(config, setup)
        else:
            if args.session_modality is not None:
                config.task.modality = args.session_modality
            if args.session_profile is not None:
                free, forced = PROFILE_COUNTS[args.session_profile]
                config.task.profile_name = args.session_profile
                config.task.n_free_trials = free
                config.task.n_forced_trials = forced
                config.n_trials = free + forced
            if args.free_trials is not None:
                config.task.n_free_trials = max(0, int(args.free_trials))
            if args.forced_trials is not None:
                config.task.n_forced_trials = max(0, int(args.forced_trials))
            if args.free_trials is not None or args.forced_trials is not None:
                free = (
                    config.task.n_free_trials
                    if config.task.n_free_trials is not None
                    else config.n_trials
                )
                config.n_trials = int(free) + int(config.task.n_forced_trials)
    if args.cap_touch_diagnostic:
        run_cap_touch_diagnostic(
            config,
            duration_s=args.cap_touch_duration,
            poll_hz=args.cap_touch_poll_hz,
            min_high_threshold=args.cap_touch_min_high_threshold,
            delta_high=args.cap_touch_delta_high,
            delta_low=args.cap_touch_delta_low,
            raw_analog=args.cap_touch_raw_analog,
        )
        return
    if args.simulate:
        if args.simulate_headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            config.monitor.enabled = False
        shorten_for_simulation(config, n_trials=args.simulate_trials)
        session = SimulatedHFDecisionSession(
            config,
            SimulationOptions(choice_mode=args.simulate_choice_mode),
        )
    elif args.hardware_self_test:
        shorten_for_hardware_self_test(config)
        logging.warning(
            "HARDWARE SELF-TEST: activating real force, pump, audible speaker, camera, and outputs"
        )
        session = HardwareSelfTestSession(config)
    else:
        session = HFDecisionSession(config)
    try:
        session.connect()
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
        session.close()
        raise
    print(f"Session finished: {summary.end_reason}")
    print(f"Trials completed: {summary.n_trials_completed}/{summary.n_trials_planned}")


if __name__ == "__main__":
    main()
