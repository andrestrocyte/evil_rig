"""One-dimensional NumPy members with bounded buffers and unchanged dtypes.

NPY stores a header followed by contiguous array bytes; NPZ is a ZIP of NPY members.
See https://numpy.org/doc/1.26/reference/generated/numpy.lib.format.html.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Generator, Iterator
from contextlib import ExitStack
from itertools import islice
from pathlib import Path
from typing import Any, Generic, IO, TypeVar
import zipfile

import numpy as np

BLOCK_BYTES = 4 * 1024 * 1024
MAX_BLOCK_ROWS = 8192
T = TypeVar("T")


def block_rows(dtype: np.dtype) -> int:
    if dtype.hasobject or dtype.itemsize <= 0:
        raise ValueError("Block arrays require a non-object, nonzero-width dtype")
    return max(1, min(MAX_BLOCK_ROWS, BLOCK_BYTES // dtype.itemsize))


class BlockArray:
    """A repeatable row source; allocate at most 4 MiB or one unusually wide row."""

    def __init__(self, count: int, dtype: np.dtype, rows: Callable[[], Iterator[Any]]):
        self.count = int(count)
        self.dtype = np.dtype(dtype)
        self.iter_rows = rows
        self.block_rows = block_rows(self.dtype)
        if self.count < 0:
            raise ValueError("Negative block-array row count")

    def write(self, handle: IO[bytes]) -> None:
        np.lib.format.write_array_header_2_0(handle, {
            "descr": np.lib.format.dtype_to_descr(self.dtype),
            "fortran_order": False, "shape": (self.count,),
        })
        with ExitStack() as stack:
            iterator = self.iter_rows()
            if close := getattr(iterator, "close", None):
                stack.callback(close)
            written = 0
            while written < self.count:
                n = min(self.block_rows, self.count - written)
                block = list(islice(iterator, n))
                if len(block) != n:
                    raise ValueError("Array row count changed during serialization")
                array = np.asarray(block, dtype=self.dtype)
                if array.shape != (n,):
                    raise ValueError("Block source produced a non-vector array")
                handle.write(memoryview(array).cast("B"))
                written += n
            sentinel = object()
            if next(iterator, sentinel) is not sentinel:
                raise ValueError("Array row count changed during serialization")


class StringArray(BlockArray, Generic[T]):
    """Match np.array(encoded_rows, dtype=str) without retaining all strings."""

    def __init__(self, rows: Collection[T], encode: Callable[[T], str]):
        count = len(rows)
        width = max(1, max((len(encode(row)) for row in rows), default=0))
        if len(rows) != count:
            raise ValueError("String-array count changed during sizing")

        def encoded() -> Iterator[str]:
            for row in rows:
                value = encode(row)
                if len(value) > width:
                    raise ValueError("String-array value grew after sizing; refusing truncation")
                yield value

        super().__init__(count, np.dtype(f"U{width}"), encoded)


def _read_header(handle: IO[bytes]) -> tuple[int, np.dtype]:
    version = np.lib.format.read_magic(handle)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
    elif version == (2, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
    else:
        raise ValueError(f"Unsupported NPY version: {version}")
    if len(shape) != 1 or fortran or int(shape[0]) < 0:
        raise ValueError("Expected a one-dimensional C-order array")
    block_rows(dtype)
    return int(shape[0]), dtype


class NpzArray:
    """Read all bytes/ZIP checksums without loading an entire semantic array."""

    def __init__(self, path: Path, name: str):
        self.path = path
        self.name = name
        with zipfile.ZipFile(path) as archive, archive.open(f"{name}.npy") as source:
            self.count, self.dtype = _read_header(source)

    def blocks(self) -> Generator[np.ndarray, None, None]:
        with zipfile.ZipFile(self.path) as archive, archive.open(f"{self.name}.npy") as source:
            if _read_header(source) != (self.count, self.dtype):
                raise ValueError("Array header changed during validation")
            remaining = self.count
            while remaining:
                n = min(block_rows(self.dtype), remaining)
                raw = source.read(n * self.dtype.itemsize)
                if len(raw) != n * self.dtype.itemsize:
                    raise ValueError(f"Truncated NPY member: {self.name}")
                yield np.frombuffer(raw, dtype=self.dtype)
                remaining -= n
            if source.read(1):
                raise ValueError(f"Unexpected trailing NPY bytes: {self.name}")
