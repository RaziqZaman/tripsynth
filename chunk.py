#!/usr/bin/env python3
"""Partition a parquet file into near-equal row-count chunks."""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm


def chunk_sizes(total_rows: int, num_chunks: int) -> list[int]:
    base = total_rows // num_chunks
    remainder = total_rows % num_chunks
    return [base + (1 if i < remainder else 0) for i in range(num_chunks)]


def split_parquet(input_path: Path, output_dir: Path, num_chunks: int) -> None:
    parquet_file = pq.ParquetFile(input_path)
    total_rows = parquet_file.metadata.num_rows
    schema = parquet_file.schema_arrow
    sizes = chunk_sizes(total_rows, num_chunks)

    output_dir.mkdir(parents=True, exist_ok=True)

    chunk_idx = 0
    rows_in_chunk = 0
    target = sizes[chunk_idx] if sizes else 0
    writer = None

    def open_writer(idx: int):
        out_path = output_dir / f"synthetic_trips_chunk_{idx + 1:02d}.parquet"
        return pq.ParquetWriter(out_path, schema=schema)

    if num_chunks > 0:
        writer = open_writer(chunk_idx)

    with tqdm(total=total_rows, unit="rows", desc="Chunking parquet") as pbar:
        for rg_idx in range(parquet_file.num_row_groups):
            table = parquet_file.read_row_group(rg_idx)
            cursor = 0

            while cursor < table.num_rows and chunk_idx < num_chunks:
                remaining = target - rows_in_chunk
                take = min(remaining, table.num_rows - cursor)
                writer.write_table(table.slice(cursor, take))
                cursor += take
                rows_in_chunk += take
                pbar.update(take)

                if rows_in_chunk == target:
                    writer.close()
                    chunk_idx += 1
                    rows_in_chunk = 0
                    if chunk_idx < num_chunks:
                        target = sizes[chunk_idx]
                        writer = open_writer(chunk_idx)
                    else:
                        writer = None

    # If there are trailing zero-row chunks, create valid empty parquet files.
    while chunk_idx < num_chunks:
        if writer is None:
            writer = open_writer(chunk_idx)
        writer.write_table(schema.empty_table())
        writer.close()
        chunk_idx += 1
        if chunk_idx < num_chunks:
            writer = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split a parquet file into N near-equal chunks by row count."
        )
    )
    parser.add_argument(
        "--input",
        default="synthetic_trips.parquet",
        help="Input parquet path (default: synthetic_trips.parquet).",
    )
    parser.add_argument(
        "--output-dir",
        default="chunked_synthetic_trips",
        help="Output directory for chunk files (default: chunked_synthetic_trips).",
    )
    parser.add_argument(
        "--chunks",
        type=int,
        default=12,
        help="Number of output chunks (default: 12).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.chunks <= 0:
        raise ValueError("--chunks must be a positive integer")

    split_parquet(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        num_chunks=args.chunks,
    )


if __name__ == "__main__":
    main()
