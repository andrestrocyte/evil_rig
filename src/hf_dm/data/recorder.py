"""Neuroscience-grade session logging (NPZ + JSON sidecar)."""

from __future__ import annotations

import json
import queue
import threading
import time
from collections import Counter, deque
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
    force_p95: float | None = None
    star_side: str | None = None
    circle_side: str | None = None
    star_direction: str | None = None
    circle_direction: str | None = None
    choice_target_direction: str | None = None
    force_output_direction: str | None = None
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
    camera_ttl_interval_count: int = 0
    camera_ttl_mean_interval_s: float = 0.0
    camera_ttl_max_interval_s: float = 0.0
    lick_count: int = 0
    lick_rate_hz: float = 0.0
    lick_timestamps_us: list[int] = field(default_factory=list)
    lick_relative_times_s: list[float] = field(default_factory=list)
    lick_channels: list[int] = field(default_factory=list)
    choice_loop_frames: int = 0
    choice_loop_duration_s: float = 0.0
    choice_loop_mean_frame_s: float = 0.0
    choice_loop_max_frame_s: float = 0.0
    choice_loop_stall_count: int = 0
    frame2ttl_code_word: int | None = None
    frame2ttl_trial_type_code: int | None = None
    frame2ttl_barcode_complete: bool = False
    frame2ttl_barcode_decoded: bool = False
    frame2ttl_barcode_match: bool = False
    frame2ttl_edge_count: int = 0
    frame2ttl_choice_edge_count: int = 0
    frame2ttl_first_ttl_timestamp_us: int | None = None
    frame2ttl_first_ttl_relative_time_s: float | None = None
    frame2ttl_first_ttl_latency_from_bpod_s: float | None = None

    def __post_init__(self) -> None:
        if self.camera_ttl_timestamps_us:
            self.camera_ttl_timestamps_us = [
                int(timestamp_us) for timestamp_us in self.camera_ttl_timestamps_us
            ]
        if (
            self.camera_ttl_interval_count
            or self.camera_ttl_mean_interval_s
            or self.camera_ttl_max_interval_s
        ):
            return
        intervals_s = [
            (next_us - previous_us) / 1_000_000.0
            for previous_us, next_us in zip(
                self.camera_ttl_timestamps_us,
                self.camera_ttl_timestamps_us[1:],
            )
            if next_us >= previous_us
        ]
        if not intervals_s:
            return
        self.camera_ttl_interval_count = len(intervals_s)
        self.camera_ttl_mean_interval_s = sum(intervals_s) / len(intervals_s)
        self.camera_ttl_max_interval_s = max(intervals_s)


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
    def __init__(
        self,
        out_dir: Path,
        subject_id: str,
        phase: str,
        *,
        base_name: str | None = None,
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.base_path = self.out_dir / (base_name or f"{subject_id}_{phase}_{stamp}")
        self.events: list[SessionEvent] = []
        self.debug_events: deque[SessionEvent] = deque(maxlen=2048)
        self.debug_event_counts: Counter[str] = Counter()
        self.debug_events_evicted = 0
        self.trials: list[TrialRecord] = []
        self.streamdio_events: list[tuple[str, int, int, int, float, float]] = []
        self.bpod_trials: list[dict[str, Any]] = []
        self.fsm_definitions: dict[str, str] = {}
        self.metadata: dict[str, Any] = {}
        self.stream_summaries: dict[str, dict[str, Any]] = {}
        self.export_options: dict[str, Any] = {}
        self._streamdio_chunk_s = 0.0
        self._streamdio_chunk_dir: Path | None = None
        self._streamdio_chunk_buffer: list[tuple[str, int, int, int, float, float]] = []
        self._streamdio_chunk_start_s: float | None = None
        self._streamdio_chunk_index = 0
        self._streamdio_chunk_queue: queue.Queue[
            tuple[int | None, list[tuple[str, int, int, int, float, float]] | None]
        ] | None = None
        self._streamdio_chunk_thread: threading.Thread | None = None
        self._streamdio_chunk_files: list[str] = []
        self._streamdio_chunk_error: str | None = None
        self._streamdio_memory_mode = "full"
        self._streamdio_memory_bounded = False
        self._streamdio_recent_max_events = 0
        self._streamdio_total_count = 0
        self._streamdio_counts: Counter[str] = Counter()
        self._max_event_count = 0
        self._max_trial_count = 0
        self._max_streamdio_event_count = 0
        self._max_bpod_trial_count = 0
        self.summary: SessionSummary | None = None
        self._session_start_wall = time.time()
        self._session_start_mono = time.perf_counter()
        self.log_event("session_start", {"subject_id": subject_id, "phase": phase})

    def set_metadata(self, metadata: dict[str, Any]) -> None:
        self.metadata.update(json.loads(json.dumps(metadata, default=str)))
        if self.stream_summaries:
            self.metadata["stream_summaries"] = dict(self.stream_summaries)
        self.log_event("session_metadata", self.metadata)

    def set_export_options(self, options: dict[str, Any]) -> None:
        self.export_options = json.loads(json.dumps(options, default=str))
        self._configure_streamdio_chunking()

    def register_stream(self, name: str, summary: dict[str, Any]) -> None:
        self.stream_summaries[str(name)] = json.loads(json.dumps(summary, default=str))
        self.metadata["stream_summaries"] = dict(self.stream_summaries)

    def get_stream_summaries(self) -> dict[str, dict[str, Any]]:
        return json.loads(json.dumps(self.stream_summaries, default=str))

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
        self._max_event_count = max(self._max_event_count, len(self.events))

    def log_debug_event(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
        if len(self.debug_events) == self.debug_events.maxlen:
            self.debug_events_evicted += 1
        self.debug_events.append(
            SessionEvent(
                wall_time_s=time.time(),
                monotonic_s=time.perf_counter(),
                event_type=event_type,
                payload=payload or {},
            )
        )
        self.debug_event_counts[str(event_type)] += 1

    def add_trial(self, trial: TrialRecord) -> None:
        self.trials.append(trial)
        self._max_trial_count = max(self._max_trial_count, len(self.trials))
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
        self._max_bpod_trial_count = max(self._max_bpod_trial_count, len(self.bpod_trials))
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
        row = (
            str(event_type),
            -1 if channel is None else int(channel),
            -1 if value is None else int(value),
            int(timestamp_us),
            float(relative_time_s),
            time.perf_counter(),
        )
        self._streamdio_total_count += 1
        self._streamdio_counts[row[0]] += 1
        self.streamdio_events.append(row)
        self._prune_streamdio_events_if_needed()
        self._max_streamdio_event_count = max(
            self._max_streamdio_event_count,
            len(self.streamdio_events),
        )
        self._buffer_streamdio_chunk(row)

    def finalize(self, summary: SessionSummary) -> Path:
        finalize_start = time.perf_counter()
        self.summary = summary
        self.log_event("session_end", {"end_reason": summary.end_reason})
        self._finalize_streamdio_chunks()

        npz_path = self.base_path.with_suffix(".npz")
        json_path = self.base_path.with_suffix(".json")

        trials_arr = self._trials_to_structured()
        events_arr = self._events_to_structured()
        final_streamdio_events = self._streamdio_rows_for_finalize()
        streamdio_arr = self._streamdio_rows_to_structured(final_streamdio_events)

        npz_error: str | None = None
        try:
            np.savez_compressed(
                npz_path,
                trials=trials_arr,
                events=events_arr,
                streamdio_events=streamdio_arr,
                bpod_trials_json=np.array(
                    [json.dumps(trial) for trial in self.bpod_trials],
                    dtype=object,
                ),
                trials_json=np.array(
                    [json.dumps(asdict(trial), default=str) for trial in self.trials],
                    dtype=object,
                ),
                events_json=np.array(
                    [json.dumps(asdict(event), default=str) for event in self.events],
                    dtype=object,
                ),
                debug_events_json=np.array(
                    [json.dumps(asdict(event), default=str) for event in self.debug_events],
                    dtype=object,
                ),
                fsm_definitions_json=np.array(
                    [
                        json.dumps({"fsm_name": name, "fsm_json": fsm_json}, default=str)
                        for name, fsm_json in self.fsm_definitions.items()
                    ],
                    dtype=object,
                ),
                summary_json=np.array([json.dumps(asdict(summary))], dtype=object),
                session_metadata_json=np.array(
                    [json.dumps(self.metadata, default=str)],
                    dtype=object,
                ),
            )
        except OSError as exc:
            npz_error = f"{type(exc).__name__}: {exc}"
            try:
                if npz_path.exists():
                    npz_path.unlink()
            except OSError:
                pass

        streamdio_counts = Counter(event[0] for event in final_streamdio_events)
        if not streamdio_counts and self._streamdio_counts:
            streamdio_counts = Counter(self._streamdio_counts)
        meta = {
            "summary": asdict(summary),
            "metadata": self.metadata,
            "trials": [asdict(t) for t in self.trials],
            "events": [asdict(e) for e in self.events],
            "debug_events_recent": [asdict(e) for e in self.debug_events],
            "debug_events": {
                "ring_capacity": self.debug_events.maxlen,
                "recent_count": len(self.debug_events),
                "total_count": sum(self.debug_event_counts.values()),
                "evicted_count": self.debug_events_evicted,
                "counts_by_type": dict(sorted(self.debug_event_counts.items())),
                "data_location": "JSON sidecar debug_events_recent and NPZ debug_events_json",
            },
            "streamdio": {
                "event_count": self._streamdio_total_count,
                "recent_in_memory_count": len(self.streamdio_events),
                "memory_mode": self._streamdio_memory_mode,
                "counts_by_type": dict(sorted(streamdio_counts.items())),
                "data_location": "NPZ array: streamdio_events",
            },
            "fsm_definitions": [
                {"fsm_name": name, "fsm_json": fsm_json}
                for name, fsm_json in self.fsm_definitions.items()
            ],
            "bpod_trials": self.bpod_trials,
            "stream_summaries": dict(self.stream_summaries),
            "npz_file": None if npz_error is not None else str(npz_path.name),
        }
        if npz_error is not None:
            meta["npz_error"] = npz_error
        json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        artifact_bytes = 0
        for path in (npz_path, json_path):
            if path.exists():
                artifact_bytes += path.stat().st_size
        options = self.export_options or (
            self.metadata.get("config", {}).get("session_export", {})
            if isinstance(self.metadata.get("config"), dict)
            else {}
        )
        if any(
            bool(options.get(name, True))
            for name in ("manifest", "config_snapshot", "export_csv_on_finalize", "export_timeline")
        ):
            try:
                from hf_dm.data.session_export import make_session_bundle

                make_session_bundle(
                    base_path=self.base_path,
                    summary=summary,
                    metadata=self.metadata,
                    trials=[asdict(trial) for trial in self.trials],
                    events=[asdict(event) for event in self.events],
                    debug_events=[asdict(event) for event in self.debug_events],
                    streamdio_events=final_streamdio_events,
                    bpod_trials=self.bpod_trials,
                    fsm_definitions=self.fsm_definitions,
                    npz_path=None if npz_error is not None else npz_path,
                    json_path=json_path,
                    options=options,
                    recorder_stats={
                        "primary_artifact_bytes": artifact_bytes,
                        "finalize_primary_write_s": time.perf_counter() - finalize_start,
                        "npz_error": npz_error,
                        "streamdio_chunk_s": self._streamdio_chunk_s,
                        "streamdio_chunk_files": list(self._streamdio_chunk_files),
                        "streamdio_chunk_count": len(self._streamdio_chunk_files),
                        "streamdio_chunk_error": self._streamdio_chunk_error,
                        "streamdio_memory_mode": self._streamdio_memory_mode,
                        "streamdio_total_count": self._streamdio_total_count,
                        "streamdio_recent_in_memory_count": len(self.streamdio_events),
                        "streamdio_recent_max_events": self._streamdio_recent_max_events,
                        "max_session_events_in_memory": self._max_event_count,
                        "max_trials_in_memory": self._max_trial_count,
                        "max_streamdio_events_in_memory": self._max_streamdio_event_count,
                        "max_bpod_trials_in_memory": self._max_bpod_trial_count,
                        "debug_event_ring_capacity": self.debug_events.maxlen,
                        "debug_event_recent_count": len(self.debug_events),
                        "debug_event_total_count": sum(self.debug_event_counts.values()),
                        "debug_event_evicted_count": self.debug_events_evicted,
                    },
                )
            except Exception as exc:
                # Export duplication must never invalidate the primary NPZ/JSON data.
                export_error_path = self.base_path.with_name(
                    f"{self.base_path.name}_export_error.json"
                )
                export_error_path.write_text(
                    json.dumps({"error": f"{type(exc).__name__}: {exc}"}, indent=2),
                    encoding="utf-8",
                )
        return json_path if npz_error is not None else npz_path

    def _configure_streamdio_chunking(self) -> None:
        if self._streamdio_chunk_thread is not None:
            return
        chunk_s = float(self.export_options.get("streamdio_chunk_s", 0.0) or 0.0)
        self._streamdio_chunk_s = max(0.0, chunk_s)
        requested_mode = str(self.export_options.get("streamdio_memory_mode", "full")).lower()
        self._streamdio_memory_mode = requested_mode if requested_mode else "full"
        self._streamdio_recent_max_events = max(
            0,
            int(self.export_options.get("streamdio_recent_events", 50000) or 0),
        )
        if self._streamdio_chunk_s <= 0.0:
            self._streamdio_memory_bounded = False
            self._streamdio_memory_mode = "full"
            return
        self._streamdio_memory_bounded = requested_mode in {"bounded", "rolling", "chunked"}
        if self._streamdio_memory_bounded and self._streamdio_recent_max_events <= 0:
            self._streamdio_recent_max_events = 50000
        try:
            from hf_dm.data.session_export import session_bundle_dir

            layout = str(self.export_options.get("layout", "flat"))
            self._streamdio_chunk_dir = (
                session_bundle_dir(self.base_path, layout=layout) / "continuous_streams"
            )
            self._streamdio_chunk_dir.mkdir(parents=True, exist_ok=True)
            self._streamdio_chunk_queue = queue.Queue()
            self._streamdio_chunk_thread = threading.Thread(
                target=self._streamdio_chunk_writer,
                name="hf-dm-streamdio-chunk-writer",
                daemon=True,
            )
            self._streamdio_chunk_thread.start()
        except Exception as exc:
            self._streamdio_chunk_error = f"{type(exc).__name__}: {exc}"
            self._streamdio_chunk_s = 0.0
            self._streamdio_memory_bounded = False

    def _prune_streamdio_events_if_needed(self) -> None:
        if not self._streamdio_memory_bounded or self._streamdio_recent_max_events <= 0:
            return
        excess = len(self.streamdio_events) - self._streamdio_recent_max_events
        if excess <= 0:
            return
        if self._streamdio_recent_max_events >= 1024 and excess < 1024:
            return
        del self.streamdio_events[:excess]

    def _buffer_streamdio_chunk(
        self,
        row: tuple[str, int, int, int, float, float],
    ) -> None:
        if self._streamdio_chunk_s <= 0.0 or self._streamdio_chunk_queue is None:
            return
        relative_s = float(row[4])
        if self._streamdio_chunk_start_s is None:
            self._streamdio_chunk_start_s = relative_s
        self._streamdio_chunk_buffer.append(row)
        if relative_s - self._streamdio_chunk_start_s < self._streamdio_chunk_s:
            return
        self._enqueue_streamdio_chunk()

    def _enqueue_streamdio_chunk(self) -> None:
        if not self._streamdio_chunk_buffer or self._streamdio_chunk_queue is None:
            return
        rows = self._streamdio_chunk_buffer
        self._streamdio_chunk_buffer = []
        self._streamdio_chunk_start_s = None
        self._streamdio_chunk_queue.put((self._streamdio_chunk_index, rows))
        self._streamdio_chunk_index += 1

    def _finalize_streamdio_chunks(self) -> None:
        if self._streamdio_chunk_queue is None:
            return
        self._enqueue_streamdio_chunk()
        self._streamdio_chunk_queue.put((None, None))
        if self._streamdio_chunk_thread is not None:
            self._streamdio_chunk_thread.join(timeout=10.0)
            if self._streamdio_chunk_thread.is_alive():
                self._streamdio_chunk_error = "chunk writer did not stop within timeout"
        self._streamdio_chunk_thread = None

    def _streamdio_chunk_writer(self) -> None:
        assert self._streamdio_chunk_queue is not None
        assert self._streamdio_chunk_dir is not None
        while True:
            index, rows = self._streamdio_chunk_queue.get()
            if index is None:
                return
            try:
                path = self._streamdio_chunk_dir / f"streamdio_chunk_{int(index):03d}.npz"
                np.savez_compressed(
                    path,
                    streamdio_events=self._streamdio_rows_to_structured(rows or []),
                )
                self._streamdio_chunk_files.append(str(path))
            except Exception as exc:
                self._streamdio_chunk_error = f"{type(exc).__name__}: {exc}"

    def _streamdio_rows_for_finalize(self) -> list[tuple[str, int, int, int, float, float]]:
        if not self._streamdio_memory_bounded:
            return list(self.streamdio_events)
        rows = self._load_streamdio_chunk_rows()
        return rows if rows else list(self.streamdio_events)

    def _load_streamdio_chunk_rows(self) -> list[tuple[str, int, int, int, float, float]]:
        rows: list[tuple[str, int, int, int, float, float]] = []
        for value in sorted(self._streamdio_chunk_files):
            path = Path(value)
            if not path.exists():
                continue
            try:
                with np.load(path, allow_pickle=True) as data:
                    if "streamdio_events" not in data.files:
                        continue
                    for row in data["streamdio_events"]:
                        rows.append(
                            (
                                str(row["event_type"]),
                                int(row["channel"]),
                                int(row["value"]),
                                int(row["module_timestamp_us"]),
                                float(row["relative_time_s"]),
                                float(row["host_perf_counter_s"]),
                            )
                        )
            except Exception as exc:
                self._streamdio_chunk_error = f"{type(exc).__name__}: {exc}"
                return []
        return rows

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
            ("force_p95", "f8"),
            ("force_min", "f8"),
            ("force_max", "f8"),
            ("trial_start_wall_s", "f8"),
            ("trial_start_mono_s", "f8"),
            ("star_side", "U8"),
            ("circle_side", "U8"),
            ("star_direction", "U8"),
            ("circle_direction", "U8"),
            ("choice_target_direction", "U8"),
            ("force_output_direction", "U8"),
            ("choice_side", "U8"),
            ("chosen_shape", "U8"),
            ("trial_mode", "U12"),
            ("forced_shape", "U8"),
            ("modality", "U12"),
            ("reward_delay_s", "f8"),
            ("reward_delivered", "?"),
            ("risk_star_probability", "f8"),
            ("bpod_state_entries", "i4"),
            ("camera_ttl_interval_count", "i4"),
            ("camera_ttl_mean_interval_s", "f8"),
            ("camera_ttl_max_interval_s", "f8"),
            ("lick_count", "i4"),
            ("lick_rate_hz", "f8"),
            ("choice_loop_frames", "i4"),
            ("choice_loop_duration_s", "f8"),
            ("choice_loop_mean_frame_s", "f8"),
            ("choice_loop_max_frame_s", "f8"),
            ("choice_loop_stall_count", "i4"),
            ("frame2ttl_code_word", "i8"),
            ("frame2ttl_trial_type_code", "i4"),
            ("frame2ttl_barcode_complete", "?"),
            ("frame2ttl_barcode_decoded", "?"),
            ("frame2ttl_barcode_match", "?"),
            ("frame2ttl_edge_count", "i4"),
            ("frame2ttl_choice_edge_count", "i4"),
            ("frame2ttl_first_ttl_timestamp_us", "i8"),
            ("frame2ttl_first_ttl_relative_time_s", "f8"),
            ("frame2ttl_first_ttl_latency_from_bpod_s", "f8"),
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
                    t.force_p95 if t.force_p95 is not None else np.nan,
                    t.force_min if t.force_min is not None else np.nan,
                    t.force_max if t.force_max is not None else np.nan,
                    t.trial_start_wall_s,
                    t.trial_start_mono_s,
                    t.star_side or "",
                    t.circle_side or "",
                    t.star_direction or "",
                    t.circle_direction or "",
                    t.choice_target_direction or "",
                    t.force_output_direction or "",
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
                    t.camera_ttl_interval_count,
                    t.camera_ttl_mean_interval_s,
                    t.camera_ttl_max_interval_s,
                    t.lick_count,
                    t.lick_rate_hz,
                    t.choice_loop_frames,
                    t.choice_loop_duration_s,
                    t.choice_loop_mean_frame_s,
                    t.choice_loop_max_frame_s,
                    t.choice_loop_stall_count,
                    -1 if t.frame2ttl_code_word is None else int(t.frame2ttl_code_word),
                    (
                        -1
                        if t.frame2ttl_trial_type_code is None
                        else int(t.frame2ttl_trial_type_code)
                    ),
                    t.frame2ttl_barcode_complete,
                    t.frame2ttl_barcode_decoded,
                    t.frame2ttl_barcode_match,
                    t.frame2ttl_edge_count,
                    t.frame2ttl_choice_edge_count,
                    (
                        -1
                        if t.frame2ttl_first_ttl_timestamp_us is None
                        else int(t.frame2ttl_first_ttl_timestamp_us)
                    ),
                    (
                        np.nan
                        if t.frame2ttl_first_ttl_relative_time_s is None
                        else float(t.frame2ttl_first_ttl_relative_time_s)
                    ),
                    (
                        np.nan
                        if t.frame2ttl_first_ttl_latency_from_bpod_s is None
                        else float(t.frame2ttl_first_ttl_latency_from_bpod_s)
                    ),
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
        return self._streamdio_rows_to_structured(self.streamdio_events)

    @staticmethod
    def _streamdio_rows_to_structured(
        rows: list[tuple[str, int, int, int, float, float]],
    ) -> np.ndarray:
        dtype = [
            ("event_type", "U24"),
            ("channel", "i2"),
            ("value", "i4"),
            ("module_timestamp_us", "u8"),
            ("relative_time_s", "f8"),
            ("host_perf_counter_s", "f8"),
        ]
        return np.array(rows, dtype=dtype)
