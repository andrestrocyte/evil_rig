"""Neuroscience-grade session logging (NPZ + JSON sidecar)."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

Direction = Literal["cw", "ccw"]
Choice = Literal["cw", "ccw", "timeout", "centering"]


@dataclass
class SessionEvent:
    wall_time_s: float
    monotonic_s: float
    event_type: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrialRecord:
    trial_index: int
    phase: str
    force_direction: Direction | None
    choice: Choice
    matched: bool
    reward_duration_s: float
    responded: bool
    response_time_s: float
    net_pulses: int
    cw_pulses: int
    ccw_pulses: int
    final_position_deg: float
    force_samples: int
    force_last: float | None
    force_mean: float | None
    force_min: float | None
    force_max: float | None
    trial_start_wall_s: float
    trial_start_mono_s: float
    camera_ttl_timestamps_us: list[int] = field(default_factory=list)


@dataclass
class SessionSummary:
    subject_id: str
    phase: str
    n_trials_planned: int
    n_trials_completed: int
    end_reason: str
    consecutive_failures: int
    session_start_wall_s: float
    session_end_wall_s: float
    engagement_idle_triggered: bool = False


class SessionRecorder:
    def __init__(self, out_dir: Path, subject_id: str, phase: str):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.base_path = self.out_dir / f"{subject_id}_{phase}_{stamp}"
        self.events: list[SessionEvent] = []
        self.trials: list[TrialRecord] = []
        # bpod-core FSM definitions captured for reproducibility.
        # Key: an experiment-chosen identifier (e.g. "choice_epoch_cw")
        # Value: fsm.to_json() output (string)
        self.fsm_definitions: dict[str, str] = {}
        self.summary: SessionSummary | None = None
        self._session_start_wall = time.time()
        self._session_start_mono = time.monotonic()
        self.log_event("session_start", {"subject_id": subject_id, "phase": phase})

    def log_fsm_definition(self, fsm_name: str, fsm: Any) -> None:
        """Store the exact bpod-core FSM we executed (JSON serialization)."""
        if fsm_name in self.fsm_definitions:
            return

        fsm_json: str
        if hasattr(fsm, "to_json"):
            fsm_json = fsm.to_json()
        elif hasattr(fsm, "to_dict"):
            fsm_json = json.dumps(fsm.to_dict())
        else:
            # Fallback: at least persist something usable for debugging.
            fsm_json = json.dumps(fsm, default=str)

        self.fsm_definitions[fsm_name] = fsm_json
        self.log_event("fsm_definition_added", {"fsm_name": fsm_name})

    def log_event(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
        self.events.append(
            SessionEvent(
                wall_time_s=time.time(),
                monotonic_s=time.monotonic(),
                event_type=event_type,
                payload=payload or {},
            )
        )

    def add_trial(self, trial: TrialRecord) -> None:
        self.trials.append(trial)
        self.log_event(
            "trial_end",
            {
                "trial_index": trial.trial_index,
                "choice": trial.choice,
                "matched": trial.matched,
                "response_time_s": trial.response_time_s,
            },
        )

    def finalize(self, summary: SessionSummary) -> Path:
        self.summary = summary
        self.log_event("session_end", {"end_reason": summary.end_reason})

        npz_path = self.base_path.with_suffix(".npz")
        json_path = self.base_path.with_suffix(".json")

        trials_arr = self._trials_to_structured()
        events_arr = self._events_to_structured()

        np.savez_compressed(
            npz_path,
            trials=trials_arr,
            events=events_arr,
            summary_json=np.array([json.dumps(asdict(summary))], dtype=object),
        )

        meta = {
            "summary": asdict(summary),
            "trials": [asdict(t) for t in self.trials],
            "events": [asdict(e) for e in self.events],
            "fsm_definitions": [
                {"fsm_name": name, "fsm_json": fsm_json}
                for name, fsm_json in self.fsm_definitions.items()
            ],
            "npz_file": str(npz_path.name),
        }
        json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return npz_path

    def _trials_to_structured(self) -> np.ndarray:
        if not self.trials:
            return np.array([], dtype=object)
        dtype = [
            ("trial_index", "i4"),
            ("phase", "U24"),
            ("force_direction", "U8"),
            ("choice", "U8"),
            ("matched", "?"),
            ("reward_duration_s", "f8"),
            ("responded", "?"),
            ("response_time_s", "f8"),
            ("net_pulses", "i4"),
            ("cw_pulses", "i4"),
            ("ccw_pulses", "i4"),
            ("final_position_deg", "f8"),
            ("force_samples", "i4"),
            ("force_last", "f8"),
            ("force_mean", "f8"),
            ("force_min", "f8"),
            ("force_max", "f8"),
            ("trial_start_wall_s", "f8"),
            ("trial_start_mono_s", "f8"),
        ]
        rows = []
        for t in self.trials:
            rows.append(
                (
                    t.trial_index,
                    t.phase,
                    t.force_direction or "",
                    t.choice,
                    t.matched,
                    t.reward_duration_s,
                    t.responded,
                    t.response_time_s,
                    t.net_pulses,
                    t.cw_pulses,
                    t.ccw_pulses,
                    t.final_position_deg,
                    t.force_samples,
                    t.force_last if t.force_last is not None else np.nan,
                    t.force_mean if t.force_mean is not None else np.nan,
                    t.force_min if t.force_min is not None else np.nan,
                    t.force_max if t.force_max is not None else np.nan,
                    t.trial_start_wall_s,
                    t.trial_start_mono_s,
                )
            )
        return np.array(rows, dtype=dtype)

    def _events_to_structured(self) -> np.ndarray:
        dtype = [("wall_time_s", "f8"), ("monotonic_s", "f8"), ("event_type", "U32"), ("payload_json", "U256")]
        rows = []
        for e in self.events:
            payload = json.dumps(e.payload)[:250]
            rows.append((e.wall_time_s, e.monotonic_s, e.event_type, payload))
        return np.array(rows, dtype=dtype)
