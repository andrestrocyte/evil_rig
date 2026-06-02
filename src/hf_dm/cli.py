"""Command-line entry: hf-dm-run --config configs/dm_50.yaml"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from hf_dm.config import SessionConfig
from hf_dm.session.controller import HFDecisionSession


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HF decision-making session (bpod-core + StreamDIO)")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML session config (see configs/)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = SessionConfig.from_yaml(args.config)
    session = HFDecisionSession(config)
    session.connect()
    summary = session.run()
    print(f"Session finished: {summary.end_reason}")
    print(f"Trials completed: {summary.n_trials_completed}/{summary.n_trials_planned}")


if __name__ == "__main__":
    main()
