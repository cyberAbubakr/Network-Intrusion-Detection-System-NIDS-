"""
Dataset discovery and memory-conscious loading for CIC-IDS2017.

Design goals (Day 1 scope):
    * Discover whatever files actually exist under data/raw/CIC-IDS2017/
      (parquet/ and csv/ subfolders) - never assume a fixed file list.
    * Prefer Parquet when available (smaller, typed, faster).
    * Fall back to chunked CSV reading when only CSV files are present.
    * NEVER load the full dataset into memory as a single DataFrame.
      All public loading functions are generators that yield chunks.

This module does not download, extract, or otherwise fetch any data.
It only operates on files the user has already placed on disk.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 250_000  # rows per CSV chunk, per project resource constraint

PARQUET_SUFFIXES = {".parquet", ".pq"}
CSV_SUFFIXES = {".csv"}


@dataclass
class DiscoveredDataset:
    """Result of scanning data/raw/CIC-IDS2017 for usable files."""

    parquet_files: list[Path] = field(default_factory=list)
    csv_files: list[Path] = field(default_factory=list)

    @property
    def has_parquet(self) -> bool:
        return len(self.parquet_files) > 0

    @property
    def has_csv(self) -> bool:
        return len(self.csv_files) > 0

    @property
    def is_empty(self) -> bool:
        return not self.has_parquet and not self.has_csv

    @property
    def preferred_format(self) -> Optional[str]:
        if self.has_parquet:
            return "parquet"
        if self.has_csv:
            return "csv"
        return None


def discover_dataset_files(raw_dir: Path | str) -> DiscoveredDataset:
    """
    Scan the raw dataset directory for Parquet and CSV files.

    Expects (but does not require) the layout:
        data/raw/CIC-IDS2017/parquet/*.parquet
        data/raw/CIC-IDS2017/csv/*.csv

    Also tolerates files dropped directly in the CIC-IDS2017/ root, in case
    the user extracts the archive without sorting it into subfolders.

    Parameters
    ----------
    raw_dir:
        Path to data/raw/CIC-IDS2017 (or an equivalent root).

    Returns
    -------
    DiscoveredDataset
        Sorted lists of discovered parquet and csv files. Sorting is by
        filename so that downstream chunked processing is reproducible.

    Raises
    ------
    FileNotFoundError
        If raw_dir does not exist at all.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw dataset directory does not exist: {raw_dir}. "
            "This function only discovers files already on disk; it does "
            "not download anything."
        )

    parquet_files: list[Path] = []
    csv_files: list[Path] = []

    for path in raw_dir.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in PARQUET_SUFFIXES:
            parquet_files.append(path)
        elif suffix in CSV_SUFFIXES:
            csv_files.append(path)

    parquet_files.sort(key=lambda p: p.name)
    csv_files.sort(key=lambda p: p.name)

    dataset = DiscoveredDataset(parquet_files=parquet_files, csv_files=csv_files)

    logger.info(
        "Discovered %d parquet file(s) and %d csv file(s) under %s",
        len(dataset.parquet_files),
        len(dataset.csv_files),
        raw_dir,
    )
    return dataset


def iter_parquet_chunks(
    files: Sequence[Path],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    columns: Optional[Sequence[str]] = None,
) -> Iterator[pd.DataFrame]:
    """
    Yield row-group-sized batches from a list of Parquet files without
    materializing the full dataset in memory.

    Uses pyarrow's batch iteration under the hood so memory use stays
    bounded regardless of total dataset size.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment issue
        raise ImportError(
            "pyarrow is required to read Parquet files. Install it via "
            "requirements.txt (pyarrow>=14.0)."
        ) from exc

    for file_path in files:
        logger.info("Reading parquet file: %s", file_path)
        parquet_file = pq.ParquetFile(str(file_path))
        for batch in parquet_file.iter_batches(
            batch_size=chunk_size, columns=list(columns) if columns else None
        ):
            chunk = batch.to_pandas()
            chunk.attrs["source_file"] = str(file_path)
            yield chunk


def iter_csv_chunks(
    files: Sequence[Path],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    usecols: Optional[Sequence[str]] = None,
    encoding: str = "utf-8",
) -> Iterator[pd.DataFrame]:
    """
    Yield fixed-size chunks from a list of CSV files using pandas'
    native chunked reader (constant memory regardless of file size).

    CIC-IDS2017 CSVs are known to sometimes use latin-1 encoding and to
    have leading/trailing whitespace in column names; both are handled
    defensively here without silently dropping any columns.
    """
    for file_path in files:
        logger.info("Reading csv file (chunked, chunksize=%d): %s", chunk_size, file_path)
        try:
            reader = pd.read_csv(
                file_path,
                chunksize=chunk_size,
                usecols=usecols,
                encoding=encoding,
                low_memory=False,
                skipinitialspace=True,
            )
            for chunk in reader:
                chunk.columns = [str(c).strip() for c in chunk.columns]
                chunk.attrs["source_file"] = str(file_path)
                yield chunk
        except UnicodeDecodeError:
            logger.warning(
                "UTF-8 decoding failed for %s, retrying with latin-1", file_path
            )
            reader = pd.read_csv(
                file_path,
                chunksize=chunk_size,
                usecols=usecols,
                encoding="latin-1",
                low_memory=False,
                skipinitialspace=True,
            )
            for chunk in reader:
                chunk.columns = [str(c).strip() for c in chunk.columns]
                chunk.attrs["source_file"] = str(file_path)
                yield chunk


def iter_dataset_chunks(
    raw_dir: Path | str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    columns: Optional[Sequence[str]] = None,
    prefer: Optional[str] = None,
) -> Iterator[pd.DataFrame]:
    """
    High-level, memory-conscious dataset iterator.

    Prefers Parquet when present; falls back to chunked CSV reading
    otherwise. Never loads the full dataset into a single DataFrame.

    Parameters
    ----------
    raw_dir:
        Root directory to scan (data/raw/CIC-IDS2017).
    chunk_size:
        Rows per chunk (CSV) or approx. rows per batch (Parquet).
    columns:
        Optional column subset to read (applied identically to both formats
        when supported). If None, all columns are read.
    prefer:
        Force "parquet" or "csv" instead of auto-detecting. Raises if the
        preferred format has no files.

    Yields
    ------
    pd.DataFrame
        Successive chunks of the dataset, each independently manageable
        in memory.

    Raises
    ------
    FileNotFoundError
        If no parquet or csv files are found at all.
    """
    dataset = discover_dataset_files(raw_dir)

    if dataset.is_empty:
        raise FileNotFoundError(
            f"No parquet or csv files found under {raw_dir}. "
            "Place the CIC-IDS2017 files there before running this script "
            "(this project never downloads data automatically)."
        )

    fmt = prefer or dataset.preferred_format

    if fmt == "parquet":
        if not dataset.has_parquet:
            raise FileNotFoundError(f"No parquet files found under {raw_dir}.")
        logger.info("Loading dataset using PARQUET path (preferred).")
        yield from iter_parquet_chunks(dataset.parquet_files, chunk_size, columns)
    elif fmt == "csv":
        if not dataset.has_csv:
            raise FileNotFoundError(f"No csv files found under {raw_dir}.")
        logger.info("Loading dataset using CHUNKED CSV fallback path.")
        yield from iter_csv_chunks(dataset.csv_files, chunk_size, columns)
    else:
        raise ValueError(f"Unknown format preference: {fmt!r}")


def peek_schema(raw_dir: Path | str, n_rows: int = 5) -> pd.DataFrame:
    """
    Read only a small sample of the dataset (first chunk, truncated to
    n_rows) purely to inspect column names/dtypes. Used by
    scripts/inspect_dataset.py so schema checks never require loading the
    whole dataset.
    """
    for chunk in iter_dataset_chunks(raw_dir, chunk_size=max(n_rows, 1000)):
        return chunk.head(n_rows)
    return pd.DataFrame()
