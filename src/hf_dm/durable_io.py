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


def _warn_unsupported(location: str) -> None:
    with _warning_lock:
        should_warn = location not in _warned_locations
        _warned_locations.add(location)
    if should_warn:
        warnings.warn(
            f"Filesystem directory at {location!r} does not support fsync; "
            "continuing after flush. Crash/power-loss durability is reduced for "
            "files in this directory.",
            RuntimeWarning,
            stacklevel=3,
        )


def _sync_descriptor(file_descriptor: int) -> None:
    """Flush without poisoning file descriptors on unsupported Windows volumes."""

    if os.name != "nt":
        os.fsync(file_descriptor)
        return

    import ctypes
    import msvcrt

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    flush_file_buffers = kernel32.FlushFileBuffers
    flush_file_buffers.argtypes = [ctypes.c_void_p]
    flush_file_buffers.restype = ctypes.c_int
    native_handle = msvcrt.get_osfhandle(file_descriptor)
    ctypes.set_last_error(0)
    if not flush_file_buffers(ctypes.c_void_p(native_handle)):
        winerror = ctypes.get_last_error()
        error = OSError(errno.EINVAL, f"FlushFileBuffers failed with WinError {winerror}")
        error.winerror = winerror  # type: ignore[attr-defined]
        raise error


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
    location = "unknown filesystem"
    if path is not None:
        directory = Path(path).resolve().parent
        location = str(directory)
    try:
        _sync_descriptor(file_descriptor)
    except OSError as exc:
        if exc.errno not in _UNSUPPORTED_FSYNC_ERRNOS:
            raise
        _warn_unsupported(location)
        return False
    return True
