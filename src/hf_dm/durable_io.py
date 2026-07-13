"""Small cross-platform helpers for best-effort durable file writes."""

from __future__ import annotations

import errno
import os
import threading
import warnings
from pathlib import Path
from typing import IO, Any


_UNSUPPORTED_FSYNC_ERRNOS = {
    errno.EBADF,
    errno.EINVAL,
    getattr(errno, "ENOSYS", errno.EINVAL),
    getattr(errno, "ENOTSUP", errno.EINVAL),
    getattr(errno, "EOPNOTSUPP", errno.EINVAL),
}
_warned_locations: set[str] = set()
_warning_lock = threading.Lock()


def fsync_best_effort(handle: IO[Any], *, path: Path | str | None = None) -> bool:
    """Flush *handle* to stable storage when the backing filesystem supports it.

    Some Windows-mounted and virtual filesystems accept normal writes but reject
    ``fsync`` with ``EBADF`` or ``EINVAL``. A valid, open handle on one of those
    filesystems should not make an otherwise recoverable session impossible to
    start. Unexpected I/O errors still propagate.

    Returns ``True`` when ``fsync`` succeeds and ``False`` when it is unsupported.
    """

    if handle.closed:
        raise ValueError("cannot fsync a closed file")
    file_descriptor = handle.fileno()
    if file_descriptor < 0:
        raise OSError(errno.EBADF, "invalid file descriptor")
    location = str(Path(path).resolve().anchor if path is not None else "unknown filesystem")
    try:
        os.fsync(file_descriptor)
    except OSError as exc:
        if exc.errno not in _UNSUPPORTED_FSYNC_ERRNOS:
            raise
        with _warning_lock:
            should_warn = location not in _warned_locations
            _warned_locations.add(location)
        if should_warn:
            warnings.warn(
                f"Filesystem at {location!r} does not support fsync; continuing after flush. "
                "Crash/power-loss durability is reduced for files on this filesystem.",
                RuntimeWarning,
                stacklevel=2,
            )
        return False
    return True
