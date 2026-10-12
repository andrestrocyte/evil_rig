import io
import json
from dataclasses import asdict
from pathlib import Path
import zipfile

import numpy as np
import pytest

from hf_dm.data.array_stream import BLOCK_BYTES, BlockArray, NpzArray, StringArray
from hf_dm.data.raw_stream import write_npz
from hf_dm.data.recorder import SessionRecorder, SessionSummary, _atomic_write_npz, _atomic_write_text
from hf_dm.data.session_export import _validate_npz_artifact, validate_session


@pytest.mark.parametrize("count", [0, 1, 17001])
def test_block_arrays_match_numpy_dtypes_bytes_and_values_across_blocks(tmp_path, count):
    dtype = np.dtype([("kind", "U8"), ("tick", ">i8"), ("note", "U64")])
    rows = [("café", index, f"雪 {index}") for index in range(count)]
    expected = np.array(rows, dtype=dtype)
    path = tmp_path / "block.npz"
    _atomic_write_npz(path, values=BlockArray(count, dtype, lambda: iter(rows)))
    with np.load(path, allow_pickle=False) as archive:
        actual = archive["values"]
        assert actual.dtype == expected.dtype
        assert actual.tobytes() == expected.tobytes()
    source = NpzArray(path, "values")
    blocks = list(source.blocks())
    assert sum(len(block) for block in blocks) == count
    assert b"".join(block.tobytes() for block in blocks) == expected.tobytes()


@pytest.mark.parametrize("rows", [[], [""], ["é", "雪", "a\x00b", "end😀"]])
def test_string_arrays_preserve_numpy_unicode_width_and_encoding(rows):
    target = io.BytesIO()
    write_npz(target, {"values": StringArray(rows, str)})
    target.seek(0)
    with np.load(target, allow_pickle=False) as archive:
        expected = np.array(rows, dtype=str)
        assert archive["values"].dtype == expected.dtype
        assert archive["values"].tobytes() == expected.tobytes()


def test_wide_rows_bound_both_writer_and_reader_blocks_in_bytes(tmp_path):
    class Sink:
        maximum = 0
        def write(self, data):
            self.maximum = max(self.maximum, len(data))

    rows = range(700)
    array = StringArray(rows, lambda value: str(value) + "x" * 4096)
    sink = Sink()
    array.write(sink)
    assert sink.maximum <= BLOCK_BYTES
    path = tmp_path / "wide.npz"
    _atomic_write_npz(path, values=array)
    count = 0
    for block in NpzArray(path, "values").blocks():
        assert block.nbytes <= BLOCK_BYTES
        count += len(block)
    assert count == len(rows)


@pytest.mark.parametrize("extra", [-1, 1])
def test_source_count_changes_close_iterator_and_leave_existing_artifact_intact(tmp_path, extra):
    closed = []
    def rows():
        try:
            yield from range(10 + extra)
        finally:
            closed.append(True)
    path = tmp_path / "existing.npz"
    path.write_bytes(b"prior artifact")
    with pytest.raises(ValueError, match="row count changed"):
        _atomic_write_npz(path, values=BlockArray(10, np.dtype("i8"), rows))
    assert closed == [True]
    assert path.read_bytes() == b"prior artifact"
    assert not list(tmp_path.glob("*.tmp"))


def test_string_growth_after_sizing_fails_instead_of_silently_truncating():
    rows = ["small"]
    array = StringArray(rows, str)
    rows[0] = "a much larger value"
    with pytest.raises(ValueError, match="refusing truncation"):
        array.write(io.BytesIO())


def _recording(tmp_path):
    recorder = SessionRecorder(tmp_path, "S1", "test", runtime_provenance={})
    recorder.set_export_options({"layout": "flat"})
    recorder.log_event("unicode_note", {"text": "café 雪 😀", "embedded_null": "a\x00b"})
    recorder.log_event("large_payload", {"trace": "x" * 12000, "nested": [1, {"value": None}]})
    recorder.log_bpod_trial(1, [{"state": "Choice", "timestamp": 123}])
    recorder.log_debug_event("diagnostic", {"note": "kept"})
    recorder.fsm_definitions["example"] = '{"state":"Choice"}'
    return recorder


def test_finalized_streamed_semantic_arrays_equal_the_original_numpy_representation(tmp_path):
    recorder = _recording(tmp_path)
    path = recorder.finalize(SessionSummary("S1", "test", 0, 0, "test", 0, 0, 1))
    payloads = [json.dumps(event.payload) for event in recorder.events]
    dtype = np.dtype([("wall_time_s", "f8"), ("monotonic_s", "f8"),
                      ("event_type", f"U{max(map(len, [event.event_type for event in recorder.events]))}"),
                      ("payload_json", f"U{max(map(len, payloads))}")])
    expected = {"events": np.array([(event.wall_time_s, event.monotonic_s, event.event_type, payload)
                for event, payload in zip(recorder.events, payloads, strict=True)], dtype=dtype),
                "events_json": np.array([json.dumps(asdict(event), default=str) for event in recorder.events], dtype=str),
                "bpod_trials_json": np.array([json.dumps(row) for row in recorder.bpod_trials], dtype=str),
                "debug_events_json": np.array([json.dumps(asdict(row), default=str) for row in recorder.debug_events], dtype=str),
                "trials_json": np.array([], dtype=str),
                "fsm_definitions_json": np.array([json.dumps({"fsm_name": "example", "fsm_json": recorder.fsm_definitions["example"]}, default=str)], dtype=str)}
    with np.load(path, allow_pickle=False) as archive:
        for name, value in expected.items():
            assert archive[name].dtype == value.dtype
            assert archive[name].tobytes() == value.tobytes(), name
    assert validate_session(path)["valid"]
    sidecar = path.with_suffix(".json")
    assert sidecar.read_bytes() == json.dumps(json.loads(sidecar.read_text()), indent=2).encode("utf-8")


@pytest.mark.parametrize("field", ["payload", "event_type"])
def test_structured_event_growth_fails_without_truncation(tmp_path, field):
    recorder = _recording(tmp_path)
    recorder.close_journal()
    recorder._close_streamdio_wal()
    array = recorder._event_array()
    if field == "payload":
        recorder.events[0].payload["longer"] = "x" * 20000
    else:
        recorder.events[0].event_type = "a" * 200
    with pytest.raises(ValueError, match="refusing truncation"):
        array.write(io.BytesIO())


@pytest.mark.parametrize("damage", ["truncated", "trailing", "count", "json", "object"])
def test_validator_consumes_entire_semantic_members_and_rejects_corrupt_arrays(tmp_path, damage):
    recorder = _recording(tmp_path)
    path = recorder.finalize(SessionSummary("S1", "test", 0, 0, "test", 0, 0, 1))
    manifest = json.loads((path.parent / (path.stem + "_bundle") / "manifest.json").read_text())
    with zipfile.ZipFile(path) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    name = "events_json.npy"
    if damage == "truncated":
        entries[name] = entries[name][:-1]
    elif damage == "trailing":
        entries[name] += b"unexpected"
    else:
        values = np.load(io.BytesIO(entries[name]), allow_pickle=False)
        if damage == "count":
            values = values[:-1]
        elif damage == "json":
            values[-1] = "not json"
        else:
            values = np.array([{"invalid": "object array"}], dtype=object)
        handle = io.BytesIO()
        np.lib.format.write_array(handle, values)
        entries[name] = handle.getvalue()
    with zipfile.ZipFile(path, "w") as archive:
        for key, value in entries.items():
            archive.writestr(key, value)
    # Bypass the outer manifest hash check: the array validator itself must fail.
    errors = _validate_npz_artifact(path, path.name, manifest)
    assert any("events_json" in error for error in errors), errors


def test_failed_streamed_json_write_keeps_existing_file_and_closes_generator(tmp_path):
    path = tmp_path / "sidecar.json"
    path.write_text('{"original":true}')
    closed = []
    def chunks():
        try:
            yield '{"partial":'
            raise ValueError("encoding failed")
        finally:
            closed.append(True)
    with pytest.raises(ValueError, match="encoding failed"):
        _atomic_write_text(path, chunks())
    assert path.read_text() == '{"original":true}'
    assert closed == [True]
    assert not list(tmp_path.glob("*.tmp"))


def test_disk_failure_closes_suspended_text_generator_even_with_retained_traceback(tmp_path, monkeypatch):
    closed = []
    def chunks():
        try:
            yield "first"
            yield "second"
        finally:
            closed.append(True)

    class FailingFile:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def writelines(self, values):
            next(values)
            raise OSError("disk full")

    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: FailingFile())
    with pytest.raises(OSError, match="disk full") as captured:
        _atomic_write_text(tmp_path / "sidecar.json", chunks())
    assert captured.value is not None
    assert closed == [True]
