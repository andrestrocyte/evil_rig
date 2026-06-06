"""Command-line entry: hf-dm-run --config configs/dm_50.yaml"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from hf_dm.config import SessionConfig
from hf_dm.session.cap_touch_diagnostic import run_cap_touch_diagnostic
from hf_dm.session.controller import HFDecisionSession
from hf_dm.session.hardware_self_test import HardwareSelfTestSession, shorten_for_hardware_self_test
from hf_dm.session.simulated import (
    SimulatedHFDecisionSession,
    SimulationOptions,
    shorten_for_simulation,
)


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
    args = parser.parse_args()
    if args.hardware_self_test and args.simulate:
        parser.error("--hardware-self-test cannot be combined with --simulate")
    if args.hardware_self_test and args.preflight:
        parser.error("--hardware-self-test cannot be combined with --preflight")
    if args.cap_touch_diagnostic and (
        args.simulate or args.preflight or args.hardware_self_test
    ):
        parser.error(
            "--cap-touch-diagnostic cannot be combined with --simulate, "
            "--preflight, or --hardware-self-test"
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
    if args.cap_touch_diagnostic:
        run_cap_touch_diagnostic(
            config,
            duration_s=args.cap_touch_duration,
            poll_hz=args.cap_touch_poll_hz,
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
        if args.preflight:
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
