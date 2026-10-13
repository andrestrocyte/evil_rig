"""Repeatable, bounded-memory readers and NumPy serialization for raw events."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from itertools import islice
from pathlib import Path
import struct
from typing import Any, IO, overload
import zipfile
import zlib

import numpy as np

from hf_dm.data.array_stream import BlockArray

WAL_MAGIC_V1 = b"HFDMSTR1\n"
WAL_MAGIC = b"HFDMSTR2\n"
WAL_PAYLOAD = struct.Struct("<24siiQdd")
WAL_RECORD = struct.Struct("<24siiQddI")
RAW_DTYPE = np.dtype([
    ("event_type", "U24"), ("channel", "i2"), ("value", "i4"),
    ("module_timestamp_us", "u8"), ("relative_time_s", "f8"),
    ("host_perf_counter_s", "f8"),
])
BLOCK_ROWS = 8192
RawRow = tuple[str, int, int, int, float, float]


class FileRows(Sequence[RawRow]):
    """Open each pass independently; no persistent file handles or full copies."""

    _count: int

    def __len__(self) -> int:
        return self._count

    @overload
    def __getitem__(self, index: int) -> RawRow: ...

    @overload
    def __getitem__(self, index: slice) -> list[RawRow]: ...

    def __getitem__(self, index: int | slice) -> RawRow | list[RawRow]:
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            if step < 0:
                return [self[i] for i in range(start, stop, step)]
            return list(islice(self, start, stop, step))
        normalized = index + len(self) if index < 0 else index
        if not 0 <= normalized < len(self):
            raise IndexError(index)
        return next(islice(self, normalized, normalized + 1))

    def __iter__(self) -> Iterator[RawRow]:
        raise NotImplementedError


class WalRows(FileRows):
    """Repeatable closed-WAL reader; optionally accept only a damaged final record.

    Interior CRC damage is never treated as an ordinary crash tail. Recovery
    retains the original bytes and exposes the discarded byte count explicitly.
    """

    def __init__(self, path: Path, *, recover_tail: bool = False):
        self.path = path
        with path.open("rb") as handle:
            self.magic = handle.read(len(WAL_MAGIC))
        if self.magic not in {WAL_MAGIC, WAL_MAGIC_V1}:
            raise ValueError(f"Invalid StreamDIO WAL header: {path}")
        self.record = WAL_RECORD if self.magic == WAL_MAGIC else WAL_PAYLOAD
        self._count, tail = divmod(path.stat().st_size - len(self.magic), self.record.size)
        if tail and not recover_tail:
            raise ValueError(f"StreamDIO WAL has {tail} trailing bytes: {path}")
        self.trailing_bytes = tail
        if recover_tail and self.magic == WAL_MAGIC:
            seen = 0
            with path.open("rb") as handle:
                handle.seek(len(self.magic))
                while seen < self._count:
                    n = min(BLOCK_ROWS, self._count - seen)
                    block = handle.read(n * self.record.size)
                    if len(block) != n * self.record.size:
                        raise ValueError("StreamDIO WAL changed during recovery scan")
                    view = memoryview(block)
                    for index, raw in enumerate(self.record.iter_unpack(block)):
                        start = index * self.record.size
                        if zlib.crc32(view[start:start + WAL_PAYLOAD.size]) & 0xFFFFFFFF != raw[6]:
                            if seen != self._count - 1:
                                raise ValueError(f"StreamDIO WAL interior checksum mismatch at record {seen}")
                            self.trailing_bytes += self.record.size
                            self._count -= 1
                            break
                        seen += 1
        with path.open("rb") as handle:
            handle.seek(len(self.magic) + self._count * self.record.size)
            self._tail = handle.read(self.trailing_bytes + 1)
        if len(self._tail) != self.trailing_bytes:
            raise ValueError("StreamDIO WAL size changed during initialization")

    def __iter__(self) -> Iterator[RawRow]:
        names: dict[bytes, str] = {}
        seen = 0
        with self.path.open("rb") as handle:
            if handle.read(len(self.magic)) != self.magic:
                raise ValueError("StreamDIO WAL header changed during export")
            while seen < self._count:
                n = min(BLOCK_ROWS, self._count - seen)
                block = handle.read(self.record.size * n)
                if len(block) != self.record.size * n:
                    raise ValueError("StreamDIO WAL truncated during export")
                view = memoryview(block)
                for index, raw in enumerate(self.record.iter_unpack(block)):
                    if self.magic == WAL_MAGIC:
                        start = index * self.record.size
                        if zlib.crc32(view[start:start + WAL_PAYLOAD.size]) & 0xFFFFFFFF != raw[6]:
                            raise ValueError(f"StreamDIO WAL checksum mismatch at record {seen}")
                    name = names.get(raw[0])
                    if name is None:
                        name = raw[0].rstrip(b"\0").decode("utf-8")
                        names[raw[0]] = name
                    yield (name, raw[1], raw[2], raw[3], raw[4], raw[5])
                    seen += 1
            if handle.read(self.trailing_bytes + 1) != self._tail:
                raise ValueError("StreamDIO WAL tail or count changed during export")
        if seen != self._count:
            raise ValueError("StreamDIO WAL count changed during export")


def _npy_header(handle: IO[bytes]) -> tuple[int, np.dtype]:
    version = np.lib.format.read_magic(handle)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
    elif version == (2, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
    else:
        raise ValueError(f"Unsupported raw-event NPY version: {version}")
    if len(shape) != 1 or fortran or dtype.hasobject or dtype.names != RAW_DTYPE.names:
        raise ValueError("Invalid raw-event array layout")
    return int(shape[0]), dtype


class NpzRows(FileRows):
    """Stream an existing compressed raw-event array, including ZIP CRC checks."""

    def __init__(self, path: Path):
        self.path = path
        with zipfile.ZipFile(path) as archive, archive.open("streamdio_events.npy") as handle:
            self._count, self.dtype = _npy_header(handle)

    def __iter__(self) -> Iterator[RawRow]:
        with zipfile.ZipFile(self.path) as archive, archive.open("streamdio_events.npy") as handle:
            count, dtype = _npy_header(handle)
            if count != self._count or dtype != self.dtype:
                raise ValueError("Raw-event array header changed during export")
            remaining = count
            while remaining:
                n = min(BLOCK_ROWS, remaining)
                block = handle.read(n * dtype.itemsize)
                if len(block) != n * dtype.itemsize:
                    raise ValueError("Truncated raw-event array")
                yield from np.frombuffer(block, dtype=dtype).tolist()
                remaining -= n
            if handle.read(1):
                raise ValueError("Unexpected trailing raw-event array bytes")


def chunk_number(path: Path) -> int:
    """Numeric order remains correct when a recording exceeds 999 chunks."""
    return int(path.stem.removeprefix("streamdio_chunk_"))


class ChunkRows(FileRows):
    """Read the whole contiguous chunk prefix, without loading entire arrays."""

    def __init__(self, paths: Sequence[Path]):
        self.paths = sorted(paths, key=chunk_number)
        if [chunk_number(path) for path in self.paths] != list(range(len(self.paths))):
            raise ValueError("StreamDIO chunk sequence has a gap or duplicate index")
        self.parts = [NpzRows(path) for path in self.paths]
        self._count = sum(map(len, self.parts))

    def __iter__(self) -> Iterator[RawRow]:
        for part in self.parts:
            yield from part


class RawArray:
    """Write the unchanged structured array format in bounded blocks."""

    def __init__(self, rows: Sequence[RawRow]):
        self.rows = rows

    def write(self, handle: IO[bytes]) -> None:
        np.lib.format.write_array_header_2_0(handle, {
            "descr": np.lib.format.dtype_to_descr(RAW_DTYPE),
            "fortran_order": False, "shape": (len(self.rows),),
        })
        iterator = iter(self.rows)
        written = 0
        while block := list(islice(iterator, BLOCK_ROWS)):
            handle.write(np.asarray(block, dtype=RAW_DTYPE).tobytes())
            written += len(block)
        if written != len(self.rows):
            raise ValueError("Raw event count changed during serialization")


def write_npz(handle: IO[bytes], arrays: dict[str, Any]) -> None:
    with zipfile.ZipFile(handle, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for name, array in arrays.items():
            with archive.open(f"{name}.npy", "w", force_zip64=True) as target:
                if isinstance(array, (RawArray, BlockArray)):
                    array.write(target)
                else:
                    np.lib.format.write_array(target, np.asanyarray(array), allow_pickle=True)
