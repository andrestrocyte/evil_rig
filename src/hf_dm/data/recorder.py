"""Neuroscience-grade session logging (NPZ + JSON sidecar)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import queue
import struct
import subprocess
import sys
import threading
import time
import zlib
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

import numpy as np

from hf_dm.durable_io import fsync_best_effort, fsync_directory_best_effort
from hf_dm.timing_heartbeat import (
    analyze_timing_heartbeats,
    timing_heartbeat_qc_summary,
)

Direction = Literal["cw", "ccw"]
Choice = Literal["cw", "ccw", "timeout", "centering"]
JournalMode = Literal["new", "resume", "disabled"]

_STREAMDIO_WAL_MAGIC_V1 = b"HFDMSTR1\n"
_STREAMDIO_WAL_MAGIC = b"HFDMSTR2\n"
_STREAMDIO_WAL_PAYLOAD = struct.Struct("<24siiQdd")
_STREAMDIO_WAL_RECORD_V1 = _STREAMDIO_WAL_PAYLOAD
_STREAMDIO_WAL_RECORD = struct.Struct("<24siiQddI")


def _git_identity(candidate: Path) -> dict[str, str | bool | None]:
    identity: dict[str, str | bool | None] = {
        "git_commit": None,
        "git_dirty": None,
        "git_root": None,
        "git_exact_tag": None,
    }
    try:
        root_result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(candidate),
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        if root_result.returncode != 0:
            return identity
        root = Path(root_result.stdout.strip()).resolve()
        identity["git_root"] = str(root)
        commit_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        if commit_result.returncode == 0:
            identity["git_commit"] = commit_result.stdout.strip() or None
        tag_result = subprocess.run(
            ["git", "describe", "--tags", "--exact-match", "HEAD"],
            cwd=str(root),
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        if tag_result.returncode == 0:
            identity["git_exact_tag"] = tag_result.stdout.strip() or None
        dirty_result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=str(root),
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        if dirty_result.returncode == 0:
            identity["git_dirty"] = bool(dirty_result.stdout.strip())
    except Exception:
        pass
    return identity


def capture_runtime_provenance(cwd: Path | str | None = None) -> dict[str, Any]:
    """Capture code identity once, before acquisition artifacts are created."""
    try:
        package_version: str | None = version("bpod-hf-dm")
    except PackageNotFoundError:
        package_version = None

    candidates = (
        [Path(cwd)]
        if cwd is not None
        else [Path(__file__).resolve().parents[3], Path.cwd()]
    )
    task_git: dict[str, str | bool | None] = {
        "git_commit": None,
        "git_dirty": None,
        "git_root": None,
        "git_exact_tag": None,
    }
    for candidate in candidates:
        task_git = _git_identity(candidate)
        if task_git["git_root"] is not None:
            break

    package_versions: dict[str, str | None] = {}
    for distribution in ("bpod-hf-dm", "bpod-core", "numpy", "PyYAML", "pyserial", "pygame"):
        try:
            package_versions[distribution] = version(distribution)
        except PackageNotFoundError:
            package_versions[distribution] = None

    streamdio_driver: dict[str, Any] = {
        "module": "pybpod_streamDIO_module",
        "module_path": None,
        "git_commit": None,
        "git_dirty": None,
        "git_root": None,
    }
    try:
        driver_spec = importlib.util.find_spec("pybpod_streamDIO_module")
        if driver_spec is not None and driver_spec.origin:
            driver_path = Path(driver_spec.origin).resolve()
            streamdio_driver["module_path"] = str(driver_path)
            streamdio_driver.update(_git_identity(driver_path.parent))
    except Exception:
        pass

    lock_sha256: str | None = None
    task_root_value = task_git.get("git_root")
    if isinstance(task_root_value, str):
        try:
            lock_sha256 = hashlib.sha256(
                (Path(task_root_value) / "requirements.lock").read_bytes()
            ).hexdigest()
        except OSError:
            pass
    return {
        **task_git,
        "package_version": package_version,
        "package_versions": package_versions,
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "requirements_lock_sha256": lock_sha256,
        "streamdio_driver": streamdio_driver,
        "provenance_captured_wall_s": time.time(),
    }


def _atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            fsync_best_effort(handle, path=temporary)
        os.replace(temporary, path)
        fsync_directory_best_effort(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_npz(path: Path, **arrays: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            fsync_best_effort(handle, path=temporary)
        os.replace(temporary, path)
        fsync_directory_best_effort(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


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
    reward_volume_ul: float | None = None
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
        journal_mode: JournalMode = "new",
        runtime_provenance: dict[str, Any] | None = None,
    ):
        # Acquisition runs alongside native camera/audio libraries.  Some vendor
        # libraries temporarily change the process working directory, which is
        # process-wide on Windows and can affect unrelated Python threads.  Keep
        # every durable recorder path absolute so periodic WAL fsyncs cannot race
        # those native calls and resolve the session path somewhere else.
        self.out_dir = Path(out_dir).expanduser().resolve()
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
        self._streamdio_failed_chunk_rows: list[
            tuple[str, int, int, int, float, float]
        ] = []
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
        self._journal_path = self.base_path.with_suffix(".journal.jsonl")
        self._incomplete_marker_path = self.base_path.with_suffix(".incomplete.json")
        self._streamdio_wal_path = self.base_path.with_suffix(".streamdio.wal")
        self._streamdio_wal_handle: Any = None
        self._streamdio_wal_lock = threading.Lock()
        self._streamdio_wal_fsync_s = 1.0
        self._streamdio_wal_fsync_stop = threading.Event()
        self._streamdio_wal_fsync_thread: threading.Thread | None = None
        self._streamdio_wal_fsync_count = 0
        self._streamdio_wal_background_error: str | None = None
        self._streamdio_events_rejected_after_wal_error = 0
        self._journal_handle: Any = None
        self._journal_lock = threading.RLock()
        self._journal_sequence = 0
        self._journal_mode: JournalMode = journal_mode
        self.runtime_provenance = json.loads(
            json.dumps(runtime_provenance or {}, default=str)
        )
        self._finalized = False
        self._open_journal(subject_id=subject_id, phase=phase)
        if journal_mode != "resume":
            self.log_event("session_start", {"subject_id": subject_id, "phase": phase})

    @property
    def journal_path(self) -> Path:
        return self._journal_path

    @property
    def incomplete_marker_path(self) -> Path:
        return self._incomplete_marker_path

    @property
    def streamdio_wal_path(self) -> Path:
        return self._streamdio_wal_path

    def _open_journal(self, *, subject_id: str, phase: str) -> None:
        if self._journal_mode == "disabled":
            return
        if self._journal_mode == "resume":
            if not self._journal_path.exists():
                raise FileNotFoundError(self._journal_path)
            self._journal_sequence = sum(
                1
                for line in self._journal_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
            self._journal_handle = self._journal_path.open(
                "a",
                encoding="utf-8",
                newline="\n",
                buffering=1,
            )
            return
        self._journal_handle = self._journal_path.open(
            "w",
            encoding="utf-8",
            newline="\n",
            buffering=1,
        )
        header = {
            "subject_id": str(subject_id),
            "phase": str(phase),
            "base_path": str(self.base_path),
            "session_start_wall_s": self._session_start_wall,
            "session_start_mono_s": self._session_start_mono,
            **self.runtime_provenance,
        }
        self._journal_append("session_header", header, durable=True)
        _atomic_write_text(
            self._incomplete_marker_path,
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "incomplete",
                    "journal": self._journal_path.name,
                    "streamdio_wal": self._streamdio_wal_path.name,
                    **header,
                },
                indent=2,
            ),
        )

    def _journal_append(
        self,
        kind: str,
        payload: dict[str, Any],
        *,
        durable: bool = False,
    ) -> None:
        with self._journal_lock:
            if self._journal_handle is None:
                return
            row = {
                "schema_version": 1,
                "sequence": self._journal_sequence,
                "kind": str(kind),
                "recorded_wall_s": time.time(),
                "payload": json.loads(json.dumps(payload, default=str)),
            }
            self._journal_handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            self._journal_handle.flush()
            if durable:
                fsync_best_effort(self._journal_handle, path=self._journal_path)
            self._journal_sequence += 1

    def close_journal(self) -> None:
        with self._journal_lock:
            if self._journal_handle is None:
                return
            self._journal_handle.flush()
            self._journal_handle.close()
            self._journal_handle = None

    def set_metadata(self, metadata: dict[str, Any]) -> None:
        self.metadata.update(json.loads(json.dumps(metadata, default=str)))
        if self.stream_summaries:
            self.metadata["stream_summaries"] = dict(self.stream_summaries)
        self._journal_append("metadata", self.metadata)
        self.log_event("session_metadata", self.metadata)

    def set_export_options(self, options: dict[str, Any]) -> None:
        self.export_options = json.loads(json.dumps(options, default=str))
        self._journal_append("export_options", self.export_options)
        self._streamdio_wal_fsync_s = max(
            0.001,
            float(self.export_options.get("streamdio_wal_fsync_s", 1.0)),
        )
        self._configure_streamdio_chunking()

    def register_stream(self, name: str, summary: dict[str, Any]) -> None:
        self.stream_summaries[str(name)] = json.loads(json.dumps(summary, default=str))
        self.metadata["stream_summaries"] = dict(self.stream_summaries)
        self._journal_append(
            "stream_summary",
            {"name": str(name), "summary": self.stream_summaries[str(name)]},
        )

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
        self._journal_append(
            "fsm_definition",
            {"fsm_name": fsm_name, "fsm_json": fsm_json},
        )
        self.log_event("fsm_definition_added", {"fsm_name": fsm_name})

    def log_event(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
        payload_snapshot = json.loads(json.dumps(payload or {}, default=str))
        event = SessionEvent(
            wall_time_s=time.time(),
            monotonic_s=time.perf_counter(),
            event_type=event_type,
            payload=payload_snapshot,
        )
        self.events.append(event)
        self._journal_append("event", asdict(event))
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
        self._flush_streamdio_wal(durable=True)
        self._journal_append("trial", asdict(trial), durable=True)
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
        self._journal_append("bpod_trial", self.bpod_trials[-1])
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
        *,
        host_perf_counter_s: float | None = None,
    ) -> None:
        """Append one raw StreamDIO event without expensive per-sample object logging."""
        # The first WAL failure is raised to the StreamDIO callback and latched.
        # Further high-rate samples must not produce thousands of callback
        # exceptions while the main task thread is arranging a safe abort.
        if self._streamdio_wal_background_error is not None:
            self._streamdio_events_rejected_after_wal_error += 1
            return
        row = (
            str(event_type),
            -1 if channel is None else int(channel),
            -1 if value is None else int(value),
            int(timestamp_us),
            float(relative_time_s),
            (
                time.perf_counter()
                if host_perf_counter_s is None
                else float(host_perf_counter_s)
            ),
        )
        try:
            self._append_streamdio_wal(row)
        except BaseException as exc:
            self._set_streamdio_wal_background_error(exc)
            self.raise_if_background_error()
            raise AssertionError("unreachable")
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
        if self._finalized:
            raise RuntimeError(f"SessionRecorder already finalized: {self.base_path}")
        self.raise_if_background_error()
        finalize_start = time.perf_counter()
        self.summary = summary
        self._append_session_water_log(summary)
        self.log_event("session_end", {"end_reason": summary.end_reason})
        self._journal_append("session_summary", asdict(summary), durable=True)
        self.close_journal()
        self._close_streamdio_wal(durable=True)
        self.raise_if_background_error()
        self._finalize_streamdio_chunks()

        npz_path = self.base_path.with_suffix(".npz")
        json_path = self.base_path.with_suffix(".json")

        trials_arr = self._trials_to_structured()
        events_arr = self._events_to_structured()
        final_streamdio_events = self._streamdio_rows_for_finalize()
        self._discard_partial_streamdio_chunks()
        streamdio_arr = self._streamdio_rows_to_structured(final_streamdio_events)
        timing_heartbeat_qc = timing_heartbeat_qc_summary(
            analyze_timing_heartbeats(
                final_streamdio_events,
                session_events=self.events,
            )
        )

        npz_error: str | None = None
        try:
            _atomic_write_npz(
                npz_path,
                trials=trials_arr,
                events=events_arr,
                streamdio_events=streamdio_arr,
                bpod_trials_json=np.array(
                    [json.dumps(trial) for trial in self.bpod_trials],
                    dtype=str,
                ),
                trials_json=np.array(
                    [json.dumps(asdict(trial), default=str) for trial in self.trials],
                    dtype=str,
                ),
                events_json=np.array(
                    [json.dumps(asdict(event), default=str) for event in self.events],
                    dtype=str,
                ),
                debug_events_json=np.array(
                    [json.dumps(asdict(event), default=str) for event in self.debug_events],
                    dtype=str,
                ),
                fsm_definitions_json=np.array(
                    [
                        json.dumps({"fsm_name": name, "fsm_json": fsm_json}, default=str)
                        for name, fsm_json in self.fsm_definitions.items()
                    ],
                    dtype=str,
                ),
                summary_json=np.array([json.dumps(asdict(summary))], dtype=str),
                session_metadata_json=np.array(
                    [json.dumps(self.metadata, default=str)],
                    dtype=str,
                ),
                timing_heartbeat_qc_json=np.array(
                    [json.dumps(timing_heartbeat_qc, default=str)],
                    dtype=str,
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
                "timing_heartbeat_qc": timing_heartbeat_qc,
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
        _atomic_write_text(json_path, json.dumps(meta, indent=2))
        artifact_bytes = 0
        for path in (npz_path, json_path):
            if path.exists():
                artifact_bytes += path.stat().st_size
        options = self.export_options or (
            self.metadata.get("config", {}).get("session_export", {})
            if isinstance(self.metadata.get("config"), dict)
            else {}
        )
        export_error: str | None = None
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
                        "streamdio_wal_fsync_s": self._streamdio_wal_fsync_s,
                        "streamdio_wal_fsync_count": self._streamdio_wal_fsync_count,
                        "streamdio_wal_background_error": (
                            self._streamdio_wal_background_error
                        ),
                        "streamdio_events_rejected_after_wal_error": (
                            self._streamdio_events_rejected_after_wal_error
                        ),
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
                export_error = f"{type(exc).__name__}: {exc}"
                export_error_path = self.base_path.with_name(
                    f"{self.base_path.name}_export_error.json"
                )
                _atomic_write_text(
                    export_error_path,
                    json.dumps({"error": export_error}, indent=2),
                )
        if json_path.exists() and npz_error is None and export_error is None:
            try:
                self._incomplete_marker_path.unlink(missing_ok=True)
                self._streamdio_wal_path.unlink(missing_ok=True)
                fsync_directory_best_effort(self.base_path.parent)
            except OSError:
                pass
        self._finalized = True
        if npz_error is not None:
            raise RuntimeError(
                "Primary NPZ finalization failed; recovery journal and StreamDIO WAL "
                f"were retained: {npz_error}"
            )
        return npz_path

    def _append_session_water_log(self, summary: SessionSummary) -> None:
        config = self.metadata.get("config")
        if not isinstance(config, dict):
            return
        welfare = config.get("welfare") or {}
        if not isinstance(welfare, dict) or not bool(
            welfare.get("append_session_water", True)
        ):
            return
        if not isinstance(self.metadata.get("reward_calibration"), dict):
            return
        volumes = [
            float(trial.reward_volume_ul)
            for trial in self.trials
            if trial.reward_volume_ul is not None
        ]
        subjects_dir = Path(str(welfare.get("subjects_dir") or "subjects"))
        if not subjects_dir.is_absolute():
            runtime = self.metadata.get("runtime_provenance") or {}
            git_root = runtime.get("git_root") if isinstance(runtime, dict) else None
            subjects_dir = (Path(str(git_root)) if git_root else Path.cwd()) / subjects_dir
        try:
            from hf_dm.subject_log import append_session_water_entry

            result = append_session_water_entry(
                subjects_dir,
                summary.subject_id,
                water_delivered_ul=sum(volumes),
                water_target_ml=welfare.get("water_target_ml"),
                session_id=str(self.base_path),
                # Attribute all water to the behavioral session date.  A run
                # crossing midnight must not move its rewards into the next
                # day's welfare ledger.
                entry_date=datetime.fromtimestamp(
                    summary.session_start_wall_s
                ).date().isoformat(),
            )
            self.log_event(
                "subject_water_logged",
                {
                    "path": result["path"],
                    "appended": result["appended"],
                    "water_delivered_ul": sum(volumes),
                    "calibrated_trial_count": len(volumes),
                },
            )
        except Exception as exc:
            self.log_event(
                "subject_water_log_error",
                {"error": f"{type(exc).__name__}: {exc}"},
            )

    @staticmethod
    def recover(
        path: Path | str,
        *,
        end_reason: str = "recovered_after_crash",
    ) -> dict[str, Any]:
        return recover_session_journal(path, end_reason=end_reason)

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

    def _append_streamdio_wal(
        self,
        row: tuple[str, int, int, int, float, float],
    ) -> None:
        self._start_streamdio_wal_fsync_worker()
        event_type = row[0].encode("utf-8")
        if len(event_type) > 24:
            raise ValueError(f"StreamDIO event type exceeds WAL limit: {row[0]!r}")
        payload = _STREAMDIO_WAL_PAYLOAD.pack(
            event_type.ljust(24, b"\0"),
            int(row[1]),
            int(row[2]),
            int(row[3]),
            float(row[4]),
            float(row[5]),
        )
        record = payload + struct.pack("<I", zlib.crc32(payload) & 0xFFFFFFFF)
        with self._streamdio_wal_lock:
            if self._streamdio_wal_handle is None:
                new_file = not self._streamdio_wal_path.exists()
                self._streamdio_wal_handle = self._streamdio_wal_path.open(
                    "ab",
                    buffering=0,
                )
                if new_file or self._streamdio_wal_path.stat().st_size == 0:
                    self._streamdio_wal_handle.write(_STREAMDIO_WAL_MAGIC)
            written = self._streamdio_wal_handle.write(record)
            if written != len(record):
                raise OSError(
                    f"Short StreamDIO WAL write: {written} of {len(record)} bytes"
                )

    def _start_streamdio_wal_fsync_worker(self) -> None:
        with self._streamdio_wal_lock:
            if self._streamdio_wal_fsync_thread is not None:
                return
            self._streamdio_wal_fsync_stop.clear()
            self._streamdio_wal_fsync_thread = threading.Thread(
                target=self._streamdio_wal_fsync_loop,
                name="hf-dm-streamdio-wal-fsync",
                daemon=True,
            )
            self._streamdio_wal_fsync_thread.start()

    def _streamdio_wal_fsync_loop(self) -> None:
        while not self._streamdio_wal_fsync_stop.wait(self._streamdio_wal_fsync_s):
            if not self._streamdio_wal_path.exists():
                continue
            try:
                self._sync_streamdio_wal()
                self._streamdio_wal_fsync_count += 1
            except BaseException as exc:
                self._set_streamdio_wal_background_error(exc)
                return

    def _stop_streamdio_wal_fsync_worker(self) -> None:
        self._streamdio_wal_fsync_stop.set()
        thread = self._streamdio_wal_fsync_thread
        self._streamdio_wal_fsync_thread = None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(1.0, self._streamdio_wal_fsync_s + 0.5))
            if thread.is_alive() and self._streamdio_wal_background_error is None:
                self._streamdio_wal_background_error = (
                    "RuntimeError: StreamDIO WAL fsync worker did not stop"
                )

    def _sync_streamdio_wal(self) -> bool:
        # Duplicate the open descriptor while holding the append lock, then do
        # the potentially slow filesystem flush after releasing that lock.  The
        # high-rate StreamDIO callback is therefore never held up by fsync.  The
        # absolute path remains as a fallback during recovery/finalization.
        sync_fd: int | None = None
        with self._streamdio_wal_lock:
            if self._streamdio_wal_handle is not None:
                sync_fd = os.dup(self._streamdio_wal_handle.fileno())
        if sync_fd is not None:
            with os.fdopen(sync_fd, "rb", buffering=0) as sync_handle:
                return fsync_best_effort(sync_handle, path=self._streamdio_wal_path)
        if not self._streamdio_wal_path.exists():
            return True
        with self._streamdio_wal_path.open("r+b", buffering=0) as sync_handle:
            return fsync_best_effort(sync_handle, path=self._streamdio_wal_path)

    def _set_streamdio_wal_background_error(self, exc: BaseException) -> None:
        if self._streamdio_wal_background_error is None:
            self._streamdio_wal_background_error = f"{type(exc).__name__}: {exc}"

    def raise_if_background_error(self) -> None:
        if self._streamdio_wal_background_error is not None:
            raise RuntimeError(
                "StreamDIO WAL durability failed: "
                f"{self._streamdio_wal_background_error}"
            )

    def _flush_streamdio_wal(self, *, durable: bool = False) -> None:
        with self._streamdio_wal_lock:
            if self._streamdio_wal_handle is None:
                return
            self._streamdio_wal_handle.flush()
        if durable:
            self._sync_streamdio_wal()

    def _close_streamdio_wal(self, *, durable: bool = False) -> None:
        self._stop_streamdio_wal_fsync_worker()
        with self._streamdio_wal_lock:
            if self._streamdio_wal_handle is None:
                return
            self._streamdio_wal_handle.flush()
            if durable:
                fsync_best_effort(
                    self._streamdio_wal_handle,
                    path=self._streamdio_wal_path,
                )
            self._streamdio_wal_handle.close()
            self._streamdio_wal_handle = None

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
            path = self._streamdio_chunk_dir / f"streamdio_chunk_{int(index):03d}.npz"
            error: Exception | None = None
            for attempt in range(3):
                try:
                    _atomic_write_npz(
                        path,
                        streamdio_events=self._streamdio_rows_to_structured(rows or []),
                    )
                    self._streamdio_chunk_files.append(str(path))
                    error = None
                    break
                except Exception as exc:
                    error = exc
                    if attempt < 2:
                        time.sleep(0.05 * (attempt + 1))
            if error is not None:
                self._streamdio_chunk_error = f"{type(error).__name__}: {error}"
                self._streamdio_failed_chunk_rows.extend(rows or [])

    def _streamdio_rows_for_finalize(self) -> list[tuple[str, int, int, int, float, float]]:
        if not self._streamdio_memory_bounded:
            return list(self.streamdio_events)
        if self._streamdio_wal_path.exists():
            try:
                wal_rows, trailing_bytes = load_streamdio_wal(self._streamdio_wal_path)
                if trailing_bytes == 0 and len(wal_rows) == self._streamdio_total_count:
                    return wal_rows
                self._streamdio_chunk_error = (
                    "StreamDIO WAL incomplete at finalize: "
                    f"{len(wal_rows)} of {self._streamdio_total_count} events, "
                    f"{trailing_bytes} trailing bytes"
                )
            except Exception as exc:
                self._streamdio_chunk_error = (
                    f"StreamDIO WAL read failed: {type(exc).__name__}: {exc}"
                )
        rows = self._load_streamdio_chunk_rows()
        if rows or self._streamdio_failed_chunk_rows:
            return [*rows, *self._streamdio_failed_chunk_rows]
        return list(self.streamdio_events)

    def _discard_partial_streamdio_chunks(self) -> None:
        """Remove a misleading partial chunk set after a terminal writer error.

        The complete WAL/primary NPZ remains authoritative.  Publishing only the
        chunks that happened to succeed would make the continuous-stream index
        disagree with the complete primary stream.
        """

        if self._streamdio_chunk_error is None:
            return
        for value in list(self._streamdio_chunk_files):
            try:
                Path(value).unlink(missing_ok=True)
            except OSError as exc:
                self._streamdio_chunk_error += (
                    f"; partial chunk cleanup: {type(exc).__name__}: {exc}"
                )
        self._streamdio_chunk_files.clear()

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
        dtype = [
            ("trial_index", "i4"),
            ("phase", "U24"),
            ("force_direction", "U8"),
            ("choice", "U8"),
            ("matched", "?"),
            ("reward_duration_s", "f8"),
            ("reward_volume_ul", "f8"),
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
        if not self.trials:
            return np.array([], dtype=dtype)
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
                    t.reward_volume_ul if t.reward_volume_ul is not None else np.nan,
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
        payloads = [json.dumps(event.payload) for event in self.events]
        event_type_chars = max(1, *(len(event.event_type) for event in self.events))
        payload_chars = max(1, *(len(payload) for payload in payloads))
        dtype = [
            ("wall_time_s", "f8"),
            ("monotonic_s", "f8"),
            ("event_type", f"U{event_type_chars}"),
            ("payload_json", f"U{payload_chars}"),
        ]
        rows = [
            (event.wall_time_s, event.monotonic_s, event.event_type, payload)
            for event, payload in zip(self.events, payloads, strict=True)
        ]
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


_JOURNAL_SUFFIX = ".journal.jsonl"


def load_streamdio_wal(
    path: Path | str,
) -> tuple[list[tuple[str, int, int, int, float, float]], int]:
    """Load complete raw StreamDIO WAL records and report a truncated tail."""

    wal_path = Path(path)
    rows: list[tuple[str, int, int, int, float, float]] = []
    with wal_path.open("rb") as handle:
        magic = handle.read(len(_STREAMDIO_WAL_MAGIC))
        if magic == _STREAMDIO_WAL_MAGIC:
            record_struct = _STREAMDIO_WAL_RECORD
            has_checksum = True
        elif magic == _STREAMDIO_WAL_MAGIC_V1:
            record_struct = _STREAMDIO_WAL_RECORD_V1
            has_checksum = False
        else:
            raise ValueError(f"Invalid StreamDIO WAL header: {wal_path}")
        trailing_bytes = 0
        while True:
            record = handle.read(record_struct.size)
            if not record:
                break
            if len(record) != record_struct.size:
                trailing_bytes = len(record)
                break
            if has_checksum:
                payload = record[:-4]
                expected_crc = struct.unpack("<I", record[-4:])[0]
                if (zlib.crc32(payload) & 0xFFFFFFFF) != expected_crc:
                    trailing_bytes = len(record) + len(handle.read())
                    break
                unpacked = _STREAMDIO_WAL_PAYLOAD.unpack(payload)
            else:
                unpacked = _STREAMDIO_WAL_RECORD_V1.unpack(record)
            event_type, channel, value, timestamp_us, relative_s, host_s = unpacked
            rows.append(
                (
                    event_type.rstrip(b"\0").decode("utf-8"),
                    int(channel),
                    int(value),
                    int(timestamp_us),
                    float(relative_s),
                    float(host_s),
                )
            )
    return rows, trailing_bytes


def resolve_session_journal(path: Path | str) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        incomplete = sorted(candidate.glob("*.incomplete.json"))
        if len(incomplete) == 1:
            candidate = incomplete[0]
        else:
            journals = sorted(candidate.glob(f"*{_JOURNAL_SUFFIX}"))
            if len(journals) != 1:
                raise FileNotFoundError(
                    f"Expected one session journal in {candidate}, found {len(journals)}"
                )
            return journals[0]
    if candidate.name.endswith(".incomplete.json"):
        marker = json.loads(candidate.read_text(encoding="utf-8"))
        journal_name = marker.get("journal")
        if not journal_name:
            raise ValueError(f"Incomplete marker has no journal field: {candidate}")
        candidate = candidate.parent / str(journal_name)
    elif not candidate.name.endswith(_JOURNAL_SUFFIX):
        candidate = candidate.with_suffix(_JOURNAL_SUFFIX)
    if not candidate.exists():
        raise FileNotFoundError(candidate)
    return candidate


def load_session_journal(
    path: Path | str,
    *,
    repair_trailing: bool = False,
) -> tuple[list[dict[str, Any]], bool]:
    journal_path = resolve_session_journal(path)
    original = journal_path.read_text(encoding="utf-8")
    lines = original.splitlines()
    nonempty = [index for index, line in enumerate(lines) if line.strip()]
    last_nonempty = nonempty[-1] if nonempty else -1
    rows: list[dict[str, Any]] = []
    repaired = False
    valid_lines: list[str] = []
    for index, line in enumerate(lines):
        text = line.strip()
        if not text:
            continue
        try:
            row = json.loads(text)
        except json.JSONDecodeError as exc:
            if repair_trailing and index == last_nonempty:
                repaired = True
                break
            raise ValueError(
                f"Invalid session journal JSON at {journal_path}:{index + 1}: {exc.msg}"
            ) from exc
        if not isinstance(row, dict):
            raise ValueError(f"Session journal row is not an object: {journal_path}:{index + 1}")
        if row.get("schema_version") != 1:
            raise ValueError(f"Unsupported session journal schema: {journal_path}:{index + 1}")
        if row.get("sequence") != len(rows):
            raise ValueError(
                f"Session journal sequence mismatch: {journal_path}:{index + 1}: "
                f"{row.get('sequence')} != {len(rows)}"
            )
        if not isinstance(row.get("payload"), dict):
            raise ValueError(f"Session journal payload is not an object: {journal_path}:{index + 1}")
        rows.append(row)
        valid_lines.append(json.dumps(row, separators=(",", ":")))
    if not rows or rows[0].get("kind") != "session_header":
        raise ValueError(f"Session journal is missing its header: {journal_path}")
    if repaired:
        _backup_recovery_journal(journal_path, original)
        _atomic_write_text(journal_path, "\n".join(valid_lines) + "\n")
    return rows, repaired


def _backup_recovery_journal(path: Path, original: str) -> Path:
    backup = path.with_name(f"{path.name}.recovery-backup")
    if not backup.exists():
        _atomic_write_text(backup, original)
    return backup


def _remove_prior_journal_summaries(
    path: Path,
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    filtered = [row for row in rows if row.get("kind") != "session_summary"]
    if len(filtered) == len(rows):
        return rows, False
    original = path.read_text(encoding="utf-8")
    _backup_recovery_journal(path, original)
    for sequence, row in enumerate(filtered):
        row["sequence"] = sequence
    _atomic_write_text(
        path,
        "\n".join(json.dumps(row, separators=(",", ":")) for row in filtered) + "\n",
    )
    return filtered, True


def _reconcile_streamdio_recovery_rows(
    wal_rows: list[tuple[str, int, int, int, float, float]],
    chunk_rows: list[tuple[str, int, int, int, float, float]],
) -> tuple[list[tuple[str, int, int, int, float, float]], str]:
    """Choose the complete prefix-compatible StreamDIO recovery source.

    Both stores originate from the same ordered callback stream.  A crash may
    leave either one with a shorter durable tail, but they must agree for their
    common prefix.  Divergence is treated as corruption rather than silently
    publishing a plausible-looking merged stream.
    """

    if not wal_rows and not chunk_rows:
        return [], "none"
    if not wal_rows:
        return list(chunk_rows), "chunks_only"
    if not chunk_rows:
        return list(wal_rows), "wal_only"
    common_length = min(len(wal_rows), len(chunk_rows))
    mismatch = next(
        (
            index
            for index in range(common_length)
            if wal_rows[index] != chunk_rows[index]
        ),
        None,
    )
    if mismatch is not None:
        raise RuntimeError(
            "StreamDIO recovery sources diverge at event "
            f"{mismatch}; refusing silent WAL/chunk data reconciliation"
        )
    if len(wal_rows) > len(chunk_rows):
        return list(wal_rows), "wal_extends_chunks"
    if len(chunk_rows) > len(wal_rows):
        return list(chunk_rows), "chunks_extend_wal"
    return list(wal_rows), "wal_chunks_identical"


def recover_session_journal(
    path: Path | str,
    *,
    end_reason: str = "recovered_after_crash",
) -> dict[str, Any]:
    journal_path = resolve_session_journal(path)
    base_name = journal_path.name[: -len(_JOURNAL_SUFFIX)]
    incomplete_marker = journal_path.with_name(f"{base_name}.incomplete.json")
    manifest_candidates = (
        journal_path.parent / "manifest.json",
        journal_path.parent / f"{base_name}_bundle" / "manifest.json",
    )
    if not incomplete_marker.exists() and any(item.exists() for item in manifest_candidates):
        raise RuntimeError(
            f"Session is already finalized; refusing recovery without an incomplete marker: "
            f"{journal_path.parent}"
        )
    rows, repaired_tail = load_session_journal(journal_path, repair_trailing=True)
    if not incomplete_marker.exists() and any(
        row.get("kind") == "session_summary" for row in rows
    ):
        raise RuntimeError(
            f"Session journal already contains a final summary; refusing recovery without an "
            f"incomplete marker: {journal_path.parent}"
        )
    rows, removed_summary = _remove_prior_journal_summaries(journal_path, rows)
    header = dict(rows[0]["payload"])
    subject_id = str(header.get("subject_id") or "unknown")
    phase = str(header.get("phase") or "unknown")
    wal_name = str(header.get("streamdio_wal") or f"{base_name}.streamdio.wal")
    wal_path = journal_path.parent / wal_name
    wal_rows: list[tuple[str, int, int, int, float, float]] = []
    wal_trailing_bytes = 0
    if wal_path.exists():
        wal_rows, wal_trailing_bytes = load_streamdio_wal(wal_path)

    recorder = SessionRecorder(
        journal_path.parent,
        subject_id,
        phase,
        base_name=base_name,
        journal_mode="resume",
    )
    recorder.events.clear()
    recorder.trials.clear()
    recorder.bpod_trials.clear()
    recorder.fsm_definitions.clear()
    recorder.metadata.clear()
    recorder.stream_summaries.clear()
    recorder.export_options.clear()
    recorder._session_start_wall = float(
        header.get("session_start_wall_s", recorder._session_start_wall)
    )
    recorder._session_start_mono = float(
        header.get("session_start_mono_s", recorder._session_start_mono)
    )

    for row in rows[1:]:
        kind = row.get("kind")
        payload = dict(row["payload"])
        if kind == "event":
            recorder.events.append(SessionEvent(**payload))
        elif kind == "trial":
            recorder.trials.append(TrialRecord(**payload))
        elif kind == "bpod_trial":
            recorder.bpod_trials.append(payload)
        elif kind == "metadata":
            recorder.metadata.update(payload)
        elif kind == "export_options":
            recorder.export_options.update(payload)
        elif kind == "stream_summary":
            name = str(payload.get("name") or "unknown")
            stream_summary = payload.get("summary")
            if isinstance(stream_summary, dict):
                recorder.stream_summaries[name] = dict(stream_summary)
        elif kind == "fsm_definition":
            fsm_name = payload.get("fsm_name")
            fsm_json = payload.get("fsm_json")
            if fsm_name is not None and fsm_json is not None:
                recorder.fsm_definitions[str(fsm_name)] = str(fsm_json)

    if recorder.stream_summaries:
        recorder.metadata["stream_summaries"] = dict(recorder.stream_summaries)
    recorder._max_event_count = len(recorder.events)
    recorder._max_trial_count = len(recorder.trials)
    recorder._max_bpod_trial_count = len(recorder.bpod_trials)

    from hf_dm.data.session_export import session_bundle_dir, validate_session

    layout = str(recorder.export_options.get("layout", "mm_hierarchy"))
    chunk_dir = session_bundle_dir(recorder.base_path, layout=layout) / "continuous_streams"
    recorder._streamdio_chunk_files = [str(item) for item in sorted(chunk_dir.glob("streamdio_chunk_*.npz"))]
    chunk_streamdio = recorder._load_streamdio_chunk_rows()
    recovered_streamdio, streamdio_recovery_source = _reconcile_streamdio_recovery_rows(
        wal_rows,
        chunk_streamdio,
    )
    recorder.streamdio_events = recovered_streamdio
    recorder._streamdio_total_count = len(recovered_streamdio)
    recorder._streamdio_counts = Counter(row[0] for row in recovered_streamdio)
    recorder._max_streamdio_event_count = len(recovered_streamdio)
    recorder._streamdio_chunk_s = float(recorder.export_options.get("streamdio_chunk_s", 0.0) or 0.0)
    recorder._streamdio_memory_mode = str(
        recorder.export_options.get("streamdio_memory_mode", "full")
    )
    recorder._streamdio_memory_bounded = bool(recorder._streamdio_chunk_files)

    # Either durable store can contain the longer crash-safe tail.  Rebuild every
    # chunk from the reconciled stream instead of publishing a partial index.
    if recovered_streamdio and recorder._streamdio_chunk_s > 0.0:
        for prior_chunk in chunk_dir.glob("streamdio_chunk_*.npz"):
            prior_chunk.unlink()
        recorder._streamdio_chunk_files.clear()
        recorder._streamdio_chunk_index = 0
        recorder._configure_streamdio_chunking()
        for stream_row in recovered_streamdio:
            recorder._buffer_streamdio_chunk(stream_row)

    recorder.log_event(
        "session_recovered",
        {
            "journal": str(journal_path),
            "repaired_trailing_record": repaired_tail,
            "replaced_prior_summary": removed_summary,
            "recovered_trials": len(recorder.trials),
            "recovered_bpod_trials": len(recorder.bpod_trials),
            "recovered_streamdio_events": len(recovered_streamdio),
            "streamdio_wal_events": len(wal_rows),
            "streamdio_chunk_events": len(chunk_streamdio),
            "streamdio_recovery_source": streamdio_recovery_source,
            "streamdio_wal_trailing_bytes": wal_trailing_bytes,
        },
    )
    config_metadata = recorder.metadata.get("config")
    planned = len(recorder.trials)
    if isinstance(config_metadata, dict):
        try:
            planned = int(config_metadata.get("n_trials", planned))
        except (TypeError, ValueError):
            pass
    summary = SessionSummary(
        subject_id=subject_id,
        phase=phase,
        n_trials_planned=max(planned, len(recorder.trials)),
        n_trials_completed=len(recorder.trials),
        end_reason=str(end_reason),
        consecutive_failures=0,
        session_start_wall_s=recorder._session_start_wall,
        session_end_wall_s=time.time(),
        engagement_idle_triggered=False,
    )
    artifact = recorder.finalize(summary)
    bundle_dir = session_bundle_dir(recorder.base_path, layout=layout)
    validation = validate_session(bundle_dir)
    return {
        "journal": str(journal_path),
        "artifact": str(artifact),
        "bundle_dir": str(bundle_dir),
        "manifest": validation.get("manifest"),
        "validation": validation,
        "repaired_trailing_record": repaired_tail,
        "replaced_prior_summary": removed_summary,
        "recovered_trials": len(recorder.trials),
        "recovered_bpod_trials": len(recorder.bpod_trials),
        "recovered_streamdio_events": len(recovered_streamdio),
        "streamdio_wal_events": len(wal_rows),
        "streamdio_chunk_events": len(chunk_streamdio),
        "streamdio_recovery_source": streamdio_recovery_source,
        "streamdio_wal_trailing_bytes": wal_trailing_bytes,
    }
