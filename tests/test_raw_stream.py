import io
import threading
import zipfile

import numpy as np
import pytest

from hf_dm.data.raw_stream import BLOCK_ROWS, RAW_DTYPE, NpzRows, RawArray, WalRows, write_npz
from hf_dm.data.recorder import SessionRecorder, SessionSummary


def summary():
    return SessionSummary("S1", "decision_making", 0, 0, "test", 0, 0, 1)


def recorder_at(path):
    recorder = SessionRecorder(path, "S1", "decision_making")
    recorder.set_export_options({
        "streamdio_memory_mode": "bounded", "streamdio_recent_events": 1,
        "manifest": False, "config_snapshot": False,
        "export_csv_on_finalize": False, "export_timeline": False,
    })
    return recorder


def test_block_serialization_retains_exact_dtype_order_values_and_repeatable_reads(tmp_path):
    recorder = recorder_at(tmp_path)
    rows = [("analog_input", 13, i % 4096, i * 1000, i / 1000, 100 + i / 1000)
            for i in range(BLOCK_ROWS * 2 + 3)]
    for kind, channel, value, timestamp, relative, host in rows:
        recorder.log_streamdio_event(kind, channel, value, timestamp, relative, host_perf_counter_s=host)
    recorder._close_streamdio_wal()
    wal = WalRows(recorder.streamdio_wal_path)
    assert list(wal) == rows
    assert wal[-1] == rows[-1]
    assert wal[BLOCK_ROWS - 1:BLOCK_ROWS + 1] == rows[BLOCK_ROWS - 1:BLOCK_ROWS + 1]
    with pytest.raises(IndexError):
        _ = wal[len(wal)]
    target = tmp_path / "stream.npz"
    with target.open("wb") as handle:
        write_npz(handle, {"streamdio_events": RawArray(wal), "other": np.arange(4)})
    with np.load(target, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["streamdio_events"], np.array(rows, dtype=RAW_DTYPE))
        np.testing.assert_array_equal(data["other"], np.arange(4))
    loaded = NpzRows(target)
    assert list(loaded) == list(loaded) == rows
    recorder.finalize(summary())


def test_new_reader_preserves_existing_numpy_archives_and_rejects_truncation(tmp_path):
    array = np.array([("camera", 0, 1, 123, .001, 34.)], dtype=RAW_DTYPE)
    path = tmp_path / "legacy.npz"
    np.savez_compressed(path, streamdio_events=array)
    assert list(NpzRows(path)) == array.tolist()
    buffer = io.BytesIO()
    np.lib.format.write_array_header_2_0(buffer, {
        "descr": np.lib.format.dtype_to_descr(RAW_DTYPE), "fortran_order": False, "shape": (2,),
    })
    buffer.write(array.tobytes())  # valid ZIP, incomplete NPY payload
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("streamdio_events.npy", buffer.getvalue())
    with pytest.raises(ValueError, match="Truncated"):
        list(NpzRows(path))


def test_validator_reports_lazy_raw_stream_corruption_without_crashing(tmp_path):
    import json
    from hf_dm.config import SessionConfig, config_to_dict
    from hf_dm.data.session_export import validate_session
    recorder = recorder_at(tmp_path)
    config = SessionConfig(results_dir=str(tmp_path))
    recorder.set_export_options(config_to_dict(config)["session_export"])
    recorder.set_metadata({"config": config_to_dict(config)})
    recorder.log_streamdio_event("camera", 0, 0, 1, .001)
    path = recorder.finalize(summary())
    with zipfile.ZipFile(path) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    entries["streamdio_events.npy"] = entries["streamdio_events.npy"][:-1]
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    result = validate_session(path)
    assert result["valid"] is False
    assert "raw-event stream unreadable" in json.dumps(result["errors"])


@pytest.mark.parametrize("damage", ["crc", "truncated", "missing"])
def test_damaged_bounded_wal_never_finalizes_partial_cache_as_complete(tmp_path, damage):
    recorder = recorder_at(tmp_path)
    for i in range(3):
        recorder.log_streamdio_event("analog_input", 13, i, i, i / 1000)
    recorder._close_streamdio_wal()
    path = recorder.streamdio_wal_path
    if damage == "missing":
        path.unlink()
    else:
        original = path.read_bytes()
        if damage == "crc":
            changed = bytearray(original)
            changed[-8] ^= 255
            path.write_bytes(changed)
        else:
            path.write_bytes(original[:-3])
    with pytest.raises(RuntimeError, match="WAL"):
        recorder.finalize(summary())
    assert recorder.incomplete_marker_path.exists()
    assert not recorder.base_path.with_suffix(".npz").exists()
    if damage != "missing":
        assert path.exists()


def test_raw_array_never_passes_a_full_session_copy_to_writer():
    class Rows:
        def __len__(self):
            return BLOCK_ROWS * 4 + 1
        def __iter__(self):
            for index in range(len(self)):
                yield ("camera", 0, 0, index, index / 1000, index / 1000)
    class Sink:
        def __init__(self):
            self.maximum_write = 0
        def write(self, value):
            self.maximum_write = max(self.maximum_write, len(value))
    sink = Sink()
    RawArray(Rows()).write(sink)
    assert sink.maximum_write <= BLOCK_ROWS * RAW_DTYPE.itemsize


def test_stalled_optional_chunk_writer_has_bounded_queue_and_preserves_recovery(tmp_path, monkeypatch):
    from hf_dm.data import recorder as module
    recorder = recorder_at(tmp_path)
    recorder.set_export_options({**recorder.export_options, "streamdio_chunk_s": .001})
    writing, release = threading.Event(), threading.Event()
    original = module._atomic_write_npz
    def stalled(path, **arrays):
        writing.set()
        assert release.wait(5)
        original(path, **arrays)
    monkeypatch.setattr(module, "_atomic_write_npz", stalled)
    monkeypatch.setattr(module, "CHUNK_SHUTDOWN_TIMEOUT_S", .02)
    try:
        for i in range(2):
            recorder.log_streamdio_event("analog_input", 13, i, i, i * .002)
        assert writing.wait(1)
        for i in range(2, 40):
            recorder.log_streamdio_event("analog_input", 13, i, i, i * .002)
        assert recorder._streamdio_chunk_queue.qsize() <= 2
        assert "backlog" in recorder._streamdio_chunk_error
        with pytest.raises(RuntimeError, match="writer did not stop"):
            recorder.finalize(summary())
        assert recorder.incomplete_marker_path.exists()
        assert recorder.streamdio_wal_path.exists()
        assert len(list(WalRows(recorder.streamdio_wal_path))) == 40
        assert not recorder.base_path.with_suffix(".npz").exists()
    finally:
        release.set()
        recorder._streamdio_chunk_thread.join(timeout=2)
        assert not recorder._streamdio_chunk_thread.is_alive()
