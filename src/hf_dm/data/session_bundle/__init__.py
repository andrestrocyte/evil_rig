"""Thin MM-style session bundle facade over the existing SessionRecorder."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hf_dm.config import SessionConfig, config_to_dict, validate_session_config
from hf_dm.data.recorder import SessionRecorder, SessionSummary
from hf_dm.data.session_export import resolve_session_output, session_bundle_dir


@dataclass
class SessionBundle:
    """Small orchestration wrapper; SessionRecorder remains the source of truth."""

    recorder: SessionRecorder
    config: SessionConfig
    final_artifact: Path | None = field(default=None, init=False)

    @classmethod
    def start(
        cls,
        config: SessionConfig,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> "SessionBundle":
        validate_session_config(config)
        out, base_name = resolve_session_output(
            Path(config.results_dir),
            subject_id=config.subject_id,
            phase=config.phase.value,
            layout=config.session_export.layout,
        )
        recorder = SessionRecorder(
            out,
            config.subject_id,
            config.phase.value,
            base_name=base_name,
        )
        config_dict = config_to_dict(config)
        recorder.set_export_options(config_dict["session_export"])
        recorder.set_metadata(metadata or {"config": config_dict})
        return cls(recorder=recorder, config=config)

    @property
    def bundle_dir(self) -> Path:
        return session_bundle_dir(
            self.recorder.base_path,
            layout=self.config.session_export.layout,
        )

    def register_streamdio(self, summary: dict[str, Any]) -> None:
        self.recorder.register_stream("streamdio", {"type": "streamdio", **summary})

    def register_camera(self, summary: dict[str, Any]) -> None:
        self.recorder.register_stream("spinnaker_video", {"type": "video", **summary})

    def get_stream_summaries(self) -> dict[str, dict[str, Any]]:
        return self.recorder.get_stream_summaries()

    def finalize(self, summary: SessionSummary) -> Path:
        if self.final_artifact is None:
            self.final_artifact = self.recorder.finalize(summary)
        return self.final_artifact

    def __enter__(self) -> "SessionBundle":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        if exc is not None:
            self.recorder.log_event(
                "session_bundle_context_error",
                {"error": f"{type(exc).__name__}: {exc}"},
            )
        if self.final_artifact is None:
            self.finalize(
                SessionSummary(
                    subject_id=self.config.subject_id,
                    phase=self.config.phase.value,
                    n_trials_planned=self.config.n_trials,
                    n_trials_completed=len(self.recorder.trials),
                    end_reason="context_error" if exc is not None else "context_exit",
                    consecutive_failures=0,
                    session_start_wall_s=self.recorder._session_start_wall,
                    session_end_wall_s=time.time(),
                    engagement_idle_triggered=False,
                )
            )
        return False
