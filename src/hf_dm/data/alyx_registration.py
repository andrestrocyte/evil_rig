"""Optional cold-path Alyx registration for completed session bundles."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


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
    result = register_alyx_session(
        manifest_path=manifest_path,
        bundle_dir=bundle_dir,
        metadata=metadata,
        summary=summary,
        artifacts=artifacts,
        config=config,
    )
    path = exports_dir / "alyx_registration.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
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
    token = str(config.get("alyx_token") or os.environ.get("ALYX_TOKEN") or "")
    project = config.get("alyx_project") or os.environ.get("ALYX_PROJECT")
    user = config.get("alyx_user") or os.environ.get("ALYX_USER")
    timeout_s = float(config.get("alyx_timeout_s") or 5.0)
    if not api_url or not token:
        return {
            "status": "skipped",
            "reason": "missing ALYX_API_URL/ALYX_TOKEN or integrations.alyx_* config",
            "created_wall_time_s": time.time(),
        }

    endpoint = _sessions_endpoint(api_url)
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
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Token {token}",
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
                "duration_s": time.perf_counter() - started,
                "created_wall_time_s": time.time(),
            }
    except (OSError, urllib.error.URLError, urllib.error.HTTPError) as exc:
        return {
            "status": "error",
            "endpoint": endpoint,
            "error": f"{type(exc).__name__}: {exc}",
            "request_payload": payload,
            "duration_s": time.perf_counter() - started,
            "created_wall_time_s": time.time(),
        }


def _sessions_endpoint(api_url: str) -> str:
    base = api_url.rstrip("/")
    return base if base.endswith("/sessions") else f"{base}/sessions"


def _relative(path: Path, start: Path) -> str:
    try:
        return os.path.relpath(Path(path).resolve(), Path(start).resolve())
    except ValueError:
        return str(path)
