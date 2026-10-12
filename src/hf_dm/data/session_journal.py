"""Streaming inspection and conservative repair of the session JSONL journal."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Generator
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hf_dm.durable_io import fsync_best_effort, fsync_directory_best_effort


def _identity(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


@dataclass(frozen=True)
class JournalScan:
    path: Path
    identity: tuple[int, int, int, int]
    row_count: int
    header: dict[str, Any]
    summary_count: int
    valid_bytes: int
    trailing_bytes: int
    needs_newline: bool
    recovery_backups: dict[str, str | None]
    raw_minimum_count: int | None = None
    export_layout: str = "mm_hierarchy"

    def check_unchanged(self) -> None:
        if _identity(self.path) != self.identity:
            raise RuntimeError(f"Session journal changed during recovery: {self.path}")


def _backup_references(row: dict[str, Any], references: dict[str, str | None]) -> None:
    event = row["payload"]
    if row["kind"] != "event" or event.get("event_type") != "session_recovered":
        return
    payload = event.get("payload") or {}
    checksums = payload.get("recovery_backup_sha256") or {}
    paths = list(payload.get("journal_recovery_backups") or [])
    if wal_backup := payload.get("streamdio_wal_recovery_backup"):
        paths.append(wal_backup)
    for path in paths:
        name = Path(str(path).replace("\\", "/")).name
        expected = checksums.get(name)
        prior = references.get(name)
        if prior is not None and expected is not None and prior != expected:
            raise ValueError(f"Conflicting recovery backup checksums: {name}")
        references[name] = expected or prior


def backup_checksums(paths: list[Path]) -> dict[str, str]:
    result = {}
    for path in paths:
        with path.open("rb") as source:
            result[path.name] = hashlib.file_digest(source, "sha256").hexdigest()
    return result


def streamdio_checkpoint_count(payload: dict[str, Any]) -> int:
    """A saved checkpoint is a lower bound, never an estimate of the crash tail."""
    count = payload.get("event_count")
    if type(count) is not int or count < 0:
        raise ValueError("Invalid StreamDIO checkpoint event_count")
    if type(payload.get("wal_fsync_succeeded")) is not bool:
        raise ValueError("Invalid StreamDIO checkpoint fsync status")
    return count


def raw_count_lower_bound(row: dict[str, Any], previous: int | None) -> int | None:
    """Share the independent raw-count checks between recovery and validation."""
    kind, payload = row["kind"], row["payload"]
    if kind == "session_header" and "streamdio_checkpoint_version" in payload:
        version = payload["streamdio_checkpoint_version"]
        if type(version) is not int or version != 1:
            raise ValueError("Unsupported StreamDIO checkpoint version")
        return 0
    if kind == "streamdio_checkpoint":
        count = streamdio_checkpoint_count(payload)
        if previous is not None and count < previous:
            raise ValueError("StreamDIO checkpoint event_count decreased")
        return count
    if kind == "event" and payload.get("event_type") == "session_recovered":
        prior_count = (payload.get("payload") or {}).get("recovered_streamdio_events")
        if prior_count is not None:
            if type(prior_count) is not int or prior_count < 0:
                raise ValueError("Invalid prior recovery raw event count")
            return max(previous or 0, prior_count)
    return previous


def _validate_row(row: Any, path: Path, line_number: int, sequence: int) -> dict[str, Any]:
    location = f"{path}:{line_number}"
    if not isinstance(row, dict):
        raise ValueError(f"Session journal row is not an object: {location}")
    if row.get("schema_version") != 1:
        raise ValueError(f"Unsupported session journal schema: {location}")
    if row.get("sequence") != sequence:
        raise ValueError(f"Session journal sequence mismatch: {location}: {row.get('sequence')} != {sequence}")
    if not isinstance(row.get("payload"), dict):
        raise ValueError(f"Session journal payload is not an object: {location}")
    kind = row.get("kind")
    if sequence == 0 and kind != "session_header":
        raise ValueError(f"Session journal is missing its header: {path}")
    if sequence > 0 and kind == "session_header":
        raise ValueError(f"Session journal has a duplicate header: {location}")
    if kind not in {"session_header", "metadata", "export_options", "stream_summary",
                    "fsm_definition", "event", "trial", "bpod_trial", "session_summary", "streamdio_checkpoint"}:
        raise ValueError(f"Session journal invalid kind: {location}: {kind}")
    if kind == "streamdio_checkpoint":
        streamdio_checkpoint_count(row["payload"])
    return row


def scan_journal(path: Path, *, tolerate_tail: bool = False) -> JournalScan:
    """Validate all rows without retaining them; only an unterminated tail can tear.

    A newline-terminated malformed row is corruption, even at EOF. UTF-8 damage is
    repairable only when an otherwise valid byte sequence ends mid-character.
    """
    identity = _identity(path)
    count = summaries = offset = trailing = 0
    header: dict[str, Any] = {}
    recovery_backups: dict[str, str | None] = {}
    raw_minimum_count: int | None = None
    export_layout = "mm_hierarchy"
    last_line = b""
    with path.open("rb") as handle:
        line_number = 0
        while offset < identity[2]:
            raw = handle.readline(identity[2] - offset)
            if not raw:
                raise RuntimeError(f"Session journal truncated during recovery: {path}")
            line_number += 1
            offset += len(raw)
            last_line = raw
            if not raw.strip():
                continue
            try:
                row = json.loads(raw.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                partial_utf8 = not isinstance(exc, UnicodeDecodeError) or exc.reason == "unexpected end of data"
                if tolerate_tail and partial_utf8 and not raw.endswith(b"\n") and offset == identity[2]:
                    trailing = len(raw)
                    break
                raise ValueError(f"Invalid session journal JSON/UTF-8 at {path}:{line_number}: {exc}") from exc
            row = _validate_row(row, path, line_number, count)
            _backup_references(row, recovery_backups)
            raw_minimum_count = raw_count_lower_bound(row, raw_minimum_count)
            if count == 0:
                header = row["payload"]
            if row["kind"] == "export_options" and "layout" in row["payload"]:
                export_layout = str(row["payload"]["layout"])
            summaries += row["kind"] == "session_summary"
            count += 1
    if not count:
        raise ValueError(f"Session journal is missing its header: {path}")
    scan = JournalScan(path, identity, count, header, summaries, offset - trailing,
                       trailing, bool(last_line and not last_line.endswith(b"\n") and not trailing),
                       recovery_backups, raw_minimum_count, export_layout)
    scan.check_unchanged()
    return scan


def iter_journal(scan: JournalScan) -> Generator[dict[str, Any], None, None]:
    """Replay the validated prefix, bounded by the largest individual JSON row."""
    scan.check_unchanged()
    offset = count = line_number = 0
    with scan.path.open("rb") as handle:
        while offset < scan.valid_bytes:
            raw = handle.readline(scan.valid_bytes - offset)
            if not raw:
                raise RuntimeError(f"Session journal truncated during recovery: {scan.path}")
            offset += len(raw)
            line_number += 1
            if not raw.strip():
                continue
            row = _validate_row(json.loads(raw.decode("utf-8")), scan.path, line_number, count)
            count += 1
            yield row
    scan.check_unchanged()
    if count != scan.row_count:
        raise RuntimeError(f"Session journal row count changed during recovery: {scan.path}")


def journal_backups(path: Path) -> list[Path]:
    return sorted(item for item in path.parent.iterdir()
                  if item.name.startswith(f"{path.name}.") and item.name.endswith(".recovery-backup")
                  and item.is_file())


def _backup_journal(scan: JournalScan) -> Path:
    """Preserve exact bytes; retries with different contents retain another backup."""
    scan.check_unchanged()
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=scan.path.parent, prefix=f".{scan.path.name}.",
                                         suffix=".tmp", delete=False) as target:
            temporary = Path(target.name)
            digest = hashlib.sha256()
            with scan.path.open("rb") as source:
                remaining = scan.identity[2]
                while remaining:
                    block = source.read(min(1024 * 1024, remaining))
                    if not block:
                        raise RuntimeError(f"Session journal truncated during recovery: {scan.path}")
                    target.write(block)
                    digest.update(block)
                    remaining -= len(block)
            target.flush()
            fsync_best_effort(target, path=temporary)
        scan.check_unchanged()
        backup = scan.path.with_name(f"{scan.path.name}.recovery-backup")
        if backup.exists():
            with backup.open("rb") as prior:
                if hashlib.file_digest(prior, "sha256").digest() == digest.digest():
                    return backup
            backup = scan.path.with_name(f"{scan.path.name}.{digest.hexdigest()}.recovery-backup")
            if backup.exists():
                with backup.open("rb") as prior:
                    if hashlib.file_digest(prior, "sha256").digest() != digest.digest():
                        raise RuntimeError(f"Existing journal recovery backup differs: {backup}")
                return backup
        os.replace(temporary, backup)
        fsync_directory_best_effort(backup.parent)
        return backup
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prepare_journal(scan: JournalScan, *, remove_summaries: bool = False) -> JournalScan:
    """Back up, repair and atomically rewrite only when append/recovery needs it."""
    scan.check_unchanged()
    # Check previous evidence before choosing a backup name or changing the log.
    # Historical absolute paths remain provenance; moved sessions use local files.
    for name, expected in scan.recovery_backups.items():
        path = scan.path.parent / name
        if not path.is_file():
            raise FileNotFoundError(f"Recovery backup missing: {path}")
        if expected is not None and backup_checksums([path])[name] != expected:
            raise RuntimeError(f"Recovery backup checksum mismatch: {path}")
    if not (scan.trailing_bytes or scan.needs_newline or (remove_summaries and scan.summary_count)):
        return scan
    _backup_journal(scan)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=scan.path.parent, prefix=f".{scan.path.name}.",
                                         suffix=".tmp", delete=False) as target:
            temporary = Path(target.name)
            sequence = 0
            with closing(iter_journal(scan)) as rows:
                for row in rows:
                    if remove_summaries and row["kind"] == "session_summary":
                        continue
                    row["sequence"] = sequence
                    target.write(json.dumps(row, separators=(",", ":")) + "\n")
                    sequence += 1
            target.flush()
            fsync_best_effort(target, path=temporary)
        scan.check_unchanged()
        os.replace(temporary, scan.path)
        fsync_directory_best_effort(scan.path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return scan_journal(scan.path)
