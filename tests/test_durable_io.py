from __future__ import annotations

import errno

import pytest

from hf_dm import durable_io
from hf_dm.data.recorder import SessionRecorder


def test_fsync_best_effort_degrades_for_unsupported_filesystem(tmp_path, monkeypatch):
    path = tmp_path / "journal.jsonl"
    durable_io._warned_locations.clear()
    attempts = 0

    def unsupported(_file_descriptor: int) -> None:
        nonlocal attempts
        attempts += 1
        raise OSError(errno.EBADF, "unsupported by mounted filesystem")

    monkeypatch.setattr(durable_io, "_sync_descriptor", unsupported)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("record\n")
        handle.flush()
        with pytest.warns(RuntimeWarning, match="durability is reduced"):
            assert durable_io.fsync_best_effort(handle, path=path) is False
        assert durable_io.fsync_best_effort(handle, path=path) is False
    assert attempts == 2


def test_fsync_best_effort_propagates_real_io_errors(tmp_path, monkeypatch):
    path = tmp_path / "journal.jsonl"
    durable_io._warned_locations.clear()

    def failed(_file_descriptor: int) -> None:
        raise OSError(errno.EIO, "device failure")

    monkeypatch.setattr(durable_io, "_sync_descriptor", failed)
    with path.open("w", encoding="utf-8") as handle:
        handle.flush()
        with pytest.raises(OSError, match="device failure"):
            durable_io.fsync_best_effort(handle, path=path)


def test_session_journal_keeps_writing_after_unsupported_fsync(tmp_path, monkeypatch):
    durable_io._warned_locations.clear()

    def unsupported(_file_descriptor: int) -> None:
        raise OSError(errno.EBADF, "unsupported by mounted filesystem")

    monkeypatch.setattr(durable_io, "_sync_descriptor", unsupported)
    with pytest.warns(RuntimeWarning, match="durability is reduced"):
        recorder = SessionRecorder(tmp_path, "S1", "decision")
    recorder.log_event("after_header", {"value": 1})
    recorder.close_journal()

    rows = recorder.journal_path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 3


def test_unsupported_directory_does_not_disable_fsync_for_sibling(tmp_path, monkeypatch):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    durable_io._warned_locations.clear()
    attempts = 0

    def first_fails(_file_descriptor: int) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError(errno.EINVAL, "unsupported here")

    monkeypatch.setattr(durable_io, "_sync_descriptor", first_fails)
    with (first / "a").open("w") as first_handle, (second / "b").open("w") as second_handle:
        with pytest.warns(RuntimeWarning, match="durability is reduced"):
            assert durable_io.fsync_best_effort(first_handle, path=first / "a") is False
        assert durable_io.fsync_best_effort(second_handle, path=second / "b") is True

    assert attempts == 2
