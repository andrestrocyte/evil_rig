"""Offline video export helpers for saved Spinnaker PGM frame sequences."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from hf_dm.camera.spinnaker import _parse_pgm_header


def export_preview_video(
    session: Path | str,
    *,
    output: Path | str | None = None,
    fps: float | None = None,
    codec: str = "FFV1",
    max_frames: int | None = None,
) -> dict[str, Any]:
    """Write an offline AVI preview from saved Mono8 PGM frames.

    OpenCV is imported only inside this function so the live task and ordinary
    analysis path do not depend on it.
    """
    try:
        import cv2  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError(
            "OpenCV is required for preview video export. Install opencv-python "
            "or MM_PyBpod2, then rerun the offline export."
        ) from exc

    video_dir, summary = resolve_video_session_dir(Path(session))
    frames = sorted(video_dir.glob("frame_*.pgm"))
    if max_frames is not None:
        frames = frames[: max(0, int(max_frames))]
    if not frames:
        raise FileNotFoundError(f"No frame_*.pgm files found in {video_dir}")

    target_fps = float(
        fps
        if fps is not None
        else summary.get("target_trigger_hz")
        or summary.get("mean_saved_fps")
        or 10.0
    )
    target_fps = max(0.1, target_fps)
    output_path = Path(output) if output is not None else video_dir / "preview_ffv1.avi"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    first = _read_pgm_mono8(frames[0])
    height, width = first.shape
    fourcc = cv2.VideoWriter_fourcc(*_fourcc(codec))
    writer = cv2.VideoWriter(str(output_path), fourcc, target_fps, (width, height), False)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open preview video writer for {output_path}")

    started = time.perf_counter()
    written = 0
    skipped = 0
    try:
        for frame_path in frames:
            try:
                frame = _read_pgm_mono8(frame_path)
            except Exception:
                skipped += 1
                continue
            if frame.shape != (height, width):
                skipped += 1
                continue
            writer.write(frame)
            written += 1
    finally:
        writer.release()

    result = {
        "output_path": str(output_path),
        "video_dir": str(video_dir),
        "codec": _fourcc(codec),
        "fps": target_fps,
        "frame_width": width,
        "frame_height": height,
        "frames_seen": len(frames),
        "frames_written": written,
        "frames_skipped": skipped,
        "export_s": time.perf_counter() - started,
        "source_summary": str(video_dir / "recording_summary.json"),
    }
    (video_dir / "preview_video_export.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    return result


def resolve_video_session_dir(session: Path) -> tuple[Path, dict[str, Any]]:
    """Resolve a saved video directory from common session artifact paths."""
    path = Path(session)
    if path.is_file() and path.name == "recording_summary.json":
        return path.parent, _load_json(path)
    if path.is_dir() and (path / "recording_summary.json").exists():
        return path, _load_json(path / "recording_summary.json")

    manifest = _manifest_path(path)
    if manifest is not None:
        data = _load_json(manifest)
        video = data.get("video") or {}
        session_dir = video.get("session_dir") or video.get("primary")
        if session_dir:
            resolved = (manifest.parent / session_dir).resolve()
            summary_path = resolved / "recording_summary.json"
            return resolved, _load_json(summary_path) if summary_path.exists() else video

    if path.is_file() and path.suffix == ".json":
        data = _load_json(path)
        for event in reversed(data.get("events") or []):
            if event.get("event_type") != "video_recording_stopped":
                continue
            payload = event.get("payload") or {}
            if payload.get("session_dir"):
                resolved = Path(payload["session_dir"])
                summary_path = resolved / "recording_summary.json"
                return resolved, _load_json(summary_path) if summary_path.exists() else payload

    if path.is_dir():
        matches = sorted(path.glob("**/recording_summary.json"))
        if matches:
            return matches[-1].parent, _load_json(matches[-1])

    raise FileNotFoundError(f"Could not resolve a Spinnaker video directory from {session}")


def _manifest_path(path: Path) -> Path | None:
    if path.is_file() and path.name == "manifest.json":
        return path
    if path.is_dir() and (path / "manifest.json").exists():
        return path / "manifest.json"
    if path.is_dir():
        matches = sorted(path.glob("*_bundle/manifest.json"))
        return matches[-1] if matches else None
    return None


def _read_pgm_mono8(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    width, height, pixel_offset = _parse_pgm_header(raw)
    pixels = width * height
    if len(raw) < pixel_offset + pixels:
        raise ValueError(f"PGM raster is shorter than expected: {path}")
    frame = np.frombuffer(raw[pixel_offset : pixel_offset + pixels], dtype=np.uint8)
    return frame.reshape((height, width))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fourcc(codec: str) -> str:
    value = str(codec or "FFV1").upper()
    if len(value) < 4:
        value = value.ljust(4)
    return value[:4]
