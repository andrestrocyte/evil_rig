"""Neuroscience-grade session logging (NPZ + JSON sidecar)."""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
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
    star_side: str | None = None
    circle_side: str | None = None
    choice_side: str | None = None
    chosen_shape: str | None = None
    trial_mode: str = "free"
    forced_shape: str | None = None
    modality: str = "force"
    reward_delay_s: float = 0.0
    reward_delivered: bool = False
    risk_star_probability: float | None = None
    bpod_state_entries: int = 0
    camera_ttl_timestamps_us: list[int] = field(default_factory=list)
    lick_count: int = 0
    lick_timestamps_us: list[int] = field(default_factory=list)
    lick_relative_times_s: list[float] = field(default_factory=list)
    lick_channels: list[int] = field(default_factory=list)
    choice_loop_frames: int = 0
    choice_loop_duration_s: float = 0.0
    choice_loop_mean_frame_s: float = 0.0
    choice_loop_max_frame_s: float = 0.0
    choice_loop_stall_count: int = 0


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
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.base_path = self.out_dir / f"{subject_id}_{phase}_{stamp}"
        self.events: list[SessionEvent] = []
        self.trials: list[TrialRecord] = []
        self.streamdio_events: list[tuple[str, int, int, int, float, float]] = []
        self.bpod_trials: list[dict[str, Any]] = []
        self.fsm_definitions: dict[str, str] = {}
        self.metadata: dict[str, Any] = {}
        self.summary: SessionSummary | None = None
        self._session_start_wall = time.time()
        self._session_start_mono = time.perf_counter()
        self.log_event("session_start", {"subject_id": subject_id, "phase": phase})

    def set_metadata(self, metadata: dict[str, Any]) -> None:
        self.metadata.update(json.loads(json.dumps(metadata, default=str)))
        self.log_event("session_metadata", self.metadata)

    def log_fsm_definition(self, fsm_name: str, fsm: Any) -> None:
        """Store an optional bpod-core FSM template (JSON serialization)."""
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
                monotonic_s=time.perf_counter(),
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

    def log_bpod_trial(self, trial_index: int, data: Any) -> None:
        """Persist the exact bpod-core state/event record for one trial."""
        if hasattr(data, "to_dicts"):
            rows = data.to_dicts()
        elif isinstance(data, list):
            rows = data
        else:
            rows = [{"value": str(data)}]
        serializable_rows = json.loads(json.dumps(rows, default=str))
        self.bpod_trials.append(
            {
                "trial_index": int(trial_index),
                "events": serializable_rows,
            }
        )
        states = [
            row.get("state")
            for row in serializable_rows
            if row.get("type") == "StateStart" or row.get("event") == "StateStart"
        ]
        self.log_event(
            "bpod_trial_complete",
            {
                "trial_index": int(trial_index),
                "event_count": len(serializable_rows),
                "states": [state for state in states if state],
            },
        )

    def log_streamdio_event(
        self,
        event_type: str,
        channel: int | None,
        value: int | None,
        timestamp_us: int,
        relative_time_s: float,
    ) -> None:
        """Append one raw StreamDIO event without expensive per-sample object logging."""
        self.streamdio_events.append(
            (
                str(event_type),
                -1 if channel is None else int(channel),
                -1 if value is None else int(value),
                int(timestamp_us),
                float(relative_time_s),
                time.perf_counter(),
            )
        )

    def finalize(self, summary: SessionSummary) -> Path:
        self.summary = summary
        self.log_event("session_end", {"end_reason": summary.end_reason})

        npz_path = self.base_path.with_suffix(".npz")
        json_path = self.base_path.with_suffix(".json")

        trials_arr = self._trials_to_structured()
        events_arr = self._events_to_structured()
        streamdio_arr = self._streamdio_to_structured()

        np.savez_compressed(
            npz_path,
            trials=trials_arr,
            events=events_arr,
            streamdio_events=streamdio_arr,
            bpod_trials_json=np.array(
                [json.dumps(trial) for trial in self.bpod_trials],
                dtype=object,
            ),
            summary_json=np.array([json.dumps(asdict(summary))], dtype=object),
            session_metadata_json=np.array(
                [json.dumps(self.metadata, default=str)],
                dtype=object,
            ),
        )

        streamdio_counts = Counter(event[0] for event in self.streamdio_events)
        meta = {
            "summary": asdict(summary),
            "metadata": self.metadata,
            "trials": [asdict(t) for t in self.trials],
            "events": [asdict(e) for e in self.events],
            "streamdio": {
                "event_count": len(self.streamdio_events),
                "counts_by_type": dict(sorted(streamdio_counts.items())),
                "data_location": "NPZ array: streamdio_events",
            },
            "fsm_definitions": [
                {"fsm_name": name, "fsm_json": fsm_json}
                for name, fsm_json in self.fsm_definitions.items()
            ],
            "bpod_trials": self.bpod_trials,
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
            ("star_side", "U8"),
            ("circle_side", "U8"),
            ("choice_side", "U8"),
            ("chosen_shape", "U8"),
            ("trial_mode", "U12"),
            ("forced_shape", "U8"),
            ("modality", "U12"),
            ("reward_delay_s", "f8"),
            ("reward_delivered", "?"),
            ("risk_star_probability", "f8"),
            ("bpod_state_entries", "i4"),
            ("lick_count", "i4"),
            ("choice_loop_frames", "i4"),
            ("choice_loop_duration_s", "f8"),
            ("choice_loop_mean_frame_s", "f8"),
            ("choice_loop_max_frame_s", "f8"),
            ("choice_loop_stall_count", "i4"),
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
                    t.star_side or "",
                    t.circle_side or "",
                    t.choice_side or "",
                    t.chosen_shape or "",
                    t.trial_mode,
                    t.forced_shape or "",
                    t.modality,
                    t.reward_delay_s,
                    t.reward_delivered,
                    (
                        t.risk_star_probability
                        if t.risk_star_probability is not None
                        else np.nan
                    ),
                    t.bpod_state_entries,
                    t.lick_count,
                    t.choice_loop_frames,
                    t.choice_loop_duration_s,
                    t.choice_loop_mean_frame_s,
                    t.choice_loop_max_frame_s,
                    t.choice_loop_stall_count,
                )
            )
        return np.array(rows, dtype=dtype)

    def _events_to_structured(self) -> np.ndarray:
        dtype = [("wall_time_s", "f8"), ("monotonic_s", "f8"), ("event_type", "U64"), ("payload_json", "U1024")]
        rows = []
        for e in self.events:
            payload = json.dumps(e.payload)[:1000]
            rows.append((e.wall_time_s, e.monotonic_s, e.event_type, payload))
        return np.array(rows, dtype=dtype)

    def _streamdio_to_structured(self) -> np.ndarray:
        dtype = [
            ("event_type", "U24"),
            ("channel", "i2"),
            ("value", "i4"),
            ("module_timestamp_us", "u8"),
            ("relative_time_s", "f8"),
            ("host_perf_counter_s", "f8"),
        ]
        return np.array(self.streamdio_events, dtype=dtype)
