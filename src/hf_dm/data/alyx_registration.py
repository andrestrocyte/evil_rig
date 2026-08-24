"""Optional cold-path Alyx registration for completed session bundles."""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from hf_dm.durable_io import fsync_best_effort, fsync_directory_best_effort


def maybe_write_alyx_registration(
    *,
    exports_dir: Path,
    manifest_path: Path,
    bundle_dir: Path,
    metadata: dict[str, Any],
    summary: dict[str, Any],
    artifacts: list[Path],
) -> Path | None:
    config = (metadata.get("config") or {}).get("integrations") or {}
    if not bool(config.get("mm_alyx_register", False)):
        return None
    path = exports_dir / "alyx_registration.json"
    if path.exists():
        try:
            prior = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            prior = None
        if (
            isinstance(prior, dict)
            and prior.get("status") == "registered"
            and _same_session(prior.get("request_payload"), metadata, summary)
        ):
            return path
    result = register_alyx_session(
        manifest_path=manifest_path,
        bundle_dir=bundle_dir,
        metadata=metadata,
        summary=summary,
        artifacts=artifacts,
        config=config,
    )
    _atomic_write_json(path, result)
    return path


def register_alyx_session(
    *,
    manifest_path: Path,
    bundle_dir: Path,
    metadata: dict[str, Any],
    summary: dict[str, Any],
    artifacts: list[Path],
    config: dict[str, Any],
) -> dict[str, Any]:
    api_url = str(config.get("alyx_api_url") or os.environ.get("ALYX_API_URL") or "")
    token = str(os.environ.get("ALYX_TOKEN") or "")
    project = config.get("alyx_project") or os.environ.get("ALYX_PROJECT")
    user = config.get("alyx_user") or os.environ.get("ALYX_USER")
    timeout_s = float(config.get("alyx_timeout_s") or 5.0)
    if not api_url or not token:
        return {
            "status": "skipped",
            "reason": "missing ALYX_API_URL/ALYX_TOKEN (tokens are environment-only)",
            "created_wall_time_s": time.time(),
        }

    endpoint = _sessions_endpoint(api_url)
    if not endpoint.lower().startswith("https://"):
        return {
            "status": "error",
            "endpoint": endpoint,
            "error": "Alyx credentials require an HTTPS endpoint",
            "created_wall_time_s": time.time(),
        }
    payload = {
        "subject": metadata.get("subject_id") or summary.get("subject_id"),
        "phase": metadata.get("phase") or summary.get("phase"),
        "project": project,
        "created_by": user,
        "session_start_wall_s": summary.get("session_start_wall_s"),
        "session_end_wall_s": summary.get("session_end_wall_s"),
        "end_reason": summary.get("end_reason"),
        "manifest": _relative(manifest_path, bundle_dir),
        "artifacts": [
            _relative(path, bundle_dir)
            for path in artifacts
            if Path(path).exists()
        ],
        "metadata": {
            "package": "bpod-hf-dm",
            "task_profile": (metadata.get("config") or {}).get("task", {}).get("profile_name"),
        },
    }
    body = json.dumps(payload).encode("utf-8")
    idempotency_key = hashlib.sha256(
        json.dumps(
            {
                "subject": payload["subject"],
                "phase": payload["phase"],
                "session_start_wall_s": payload["session_start_wall_s"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Token {token}",
            "Idempotency-Key": idempotency_key,
        },
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            response_body = response.read(4096).decode("utf-8", errors="replace")
            parsed_body: Any
            try:
                parsed_body = json.loads(response_body) if response_body else {}
            except json.JSONDecodeError:
                parsed_body = response_body
            return {
                "status": "registered",
                "endpoint": endpoint,
                "status_code": int(getattr(response, "status", 200)),
                "response": parsed_body,
                "request_payload": payload,
                "idempotency_key": idempotency_key,
                "duration_s": time.perf_counter() - started,
                "created_wall_time_s": time.time(),
            }
    except (OSError, urllib.error.URLError, urllib.error.HTTPError) as exc:
        return {
            "status": "error",
            "endpoint": endpoint,
            "error": f"{type(exc).__name__}: {exc}",
            "request_payload": payload,
            "idempotency_key": idempotency_key,
            "duration_s": time.perf_counter() - started,
            "created_wall_time_s": time.time(),
        }


def _sessions_endpoint(api_url: str) -> str:
    base = api_url.rstrip("/")
    return base if base.endswith("/sessions") else f"{base}/sessions"


def _same_session(
    request_payload: Any,
    metadata: dict[str, Any],
    summary: dict[str, Any],
) -> bool:
    if not isinstance(request_payload, dict):
        return False
    return (
        request_payload.get("subject")
        == (metadata.get("subject_id") or summary.get("subject_id"))
        and request_payload.get("phase")
        == (metadata.get("phase") or summary.get("phase"))
        and request_payload.get("session_start_wall_s")
        == summary.get("session_start_wall_s")
    )


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, indent=2))
            handle.flush()
            fsync_best_effort(handle, path=temporary)
        os.replace(temporary, path)
        fsync_directory_best_effort(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _relative(path: Path, start: Path) -> str:
    try:
        return os.path.relpath(Path(path).resolve(), Path(start).resolve())
    except ValueError:
        return str(path)
