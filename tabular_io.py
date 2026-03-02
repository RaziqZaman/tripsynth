#!/usr/bin/env python3
"""Helpers for reading and writing CSV/Parquet tabular data."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterator, List, Sequence


def is_parquet_path(path: Path) -> bool:
    return path.suffix.lower() == ".parquet"


def _require_pyarrow() -> tuple[object, object]:
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except ModuleNotFoundError as exc:
        missing = exc.name or "pyarrow"
        raise ModuleNotFoundError(
            f"Missing Python dependency: {missing}. "
            "Install parquet support with: pip install pyarrow"
        ) from exc
    return pa, pq


def read_rows(path: Path) -> tuple[List[str], List[Dict[str, object]]]:
    if is_parquet_path(path):
        _, pq = _require_pyarrow()
        table = pq.read_table(path)
        return list(table.column_names), list(table.to_pylist())

    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        columns = list(reader.fieldnames or [])
        return columns, list(reader)


def write_rows(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, object]]) -> None:
    cols = list(fieldnames)
    if is_parquet_path(path):
        pa, pq = _require_pyarrow()
        schema = pa.schema([(c, pa.string()) for c in cols])
        normalized = [
            {c: "" if row.get(c) is None else str(row.get(c)) for c in cols}
            for row in rows
        ]
        table = pa.Table.from_pylist(normalized, schema=schema)
        pq.write_table(table, path, compression="zstd")
        return

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def get_fieldnames(path: Path) -> List[str]:
    if is_parquet_path(path):
        _, pq = _require_pyarrow()
        pf = pq.ParquetFile(path)
        return list(pf.schema.names)

    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or [])


def iter_rows(path: Path, batch_size: int = 8192) -> Iterator[Dict[str, object]]:
    if is_parquet_path(path):
        _, pq = _require_pyarrow()
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=batch_size):
            for row in batch.to_pylist():
                yield row
        return

    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def count_rows(path: Path) -> int:
    if is_parquet_path(path):
        _, pq = _require_pyarrow()
        pf = pq.ParquetFile(path)
        return int(pf.metadata.num_rows)

    n = 0
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for _ in reader:
            n += 1
    return n


class StringRowWriter:
    def __init__(self, path: Path, fieldnames: Sequence[str], buffer_size: int = 8192) -> None:
        self.path = path
        self.fieldnames = list(fieldnames)
        self.buffer_size = max(1, int(buffer_size))
        self._csv_file = None
        self._csv_writer = None
        self._pa = None
        self._pq_writer = None
        self._schema = None
        self._buffer: List[Dict[str, str]] = []

        if is_parquet_path(path):
            pa, pq = _require_pyarrow()
            self._pa = pa
            self._schema = pa.schema([(c, pa.string()) for c in self.fieldnames])
            self._pq_writer = pq.ParquetWriter(path, self._schema, compression="zstd")
        else:
            self._csv_file = path.open("w", newline="", encoding="utf-8")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self.fieldnames)
            self._csv_writer.writeheader()

    def write(self, row: Dict[str, object]) -> None:
        normalized = {
            c: "" if row.get(c) is None else str(row.get(c))
            for c in self.fieldnames
        }
        if self._csv_writer is not None:
            self._csv_writer.writerow(normalized)
            return

        self._buffer.append(normalized)
        if len(self._buffer) >= self.buffer_size:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        if self._pq_writer is None or self._pa is None:
            return

        table = self._pa.Table.from_pylist(self._buffer, schema=self._schema)
        self._pq_writer.write_table(table)
        self._buffer.clear()

    def close(self) -> None:
        self.flush()
        if self._pq_writer is not None:
            self._pq_writer.close()
            self._pq_writer = None
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None

    def __enter__(self) -> "StringRowWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
