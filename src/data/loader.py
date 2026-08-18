"""
Dataset discovery and memory-conscious loading for CIC-IDS2017.

Design goals (Day 1 scope):
    * Discover files under data/raw/CIC-IDS2017/.
    * Prefer the dedicated parquet/ directory when it exists.
    * Prefer Parquet over CSV when both are available.
    * Avoid processing duplicate copies of the same dataset.
    * Fall back to chunked CSV reading when Parquet is unavailable.
    * NEVER load the full raw dataset into memory as a single DataFrame.
      Public loading functions yield manageable chunks.

Expected layout:

    data/raw/CIC-IDS2017/
        parquet/
            *.parquet
        csv/
            *.csv

Some distributions may instead place files directly under:

    data/raw/CIC-IDS2017/
        *.parquet
        *.csv

This module supports both layouts.

IMPORTANT:
    If the same Parquet files exist both directly under CIC-IDS2017/
    and inside CIC-IDS2017/parquet/, only the files inside parquet/
    are used. This prevents accidental duplicate processing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional, Sequence

import pandas as pd


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CHUNK_SIZE = 250_000

PARQUET_SUFFIXES = {".parquet", ".pq"}
CSV_SUFFIXES = {".csv"}


# ---------------------------------------------------------------------------
# Dataset discovery result
# ---------------------------------------------------------------------------


@dataclass
class DiscoveredDataset:
    """Result of scanning the CIC-IDS2017 raw dataset directory."""

    parquet_files: list[Path] = field(default_factory=list)
    csv_files: list[Path] = field(default_factory=list)

    @property
    def has_parquet(self) -> bool:
        """Return True when at least one Parquet file was discovered."""
        return len(self.parquet_files) > 0

    @property
    def has_csv(self) -> bool:
        """Return True when at least one CSV file was discovered."""
        return len(self.csv_files) > 0

    @property
    def is_empty(self) -> bool:
        """Return True when no usable dataset files were discovered."""
        return not self.has_parquet and not self.has_csv

    @property
    def preferred_format(self) -> Optional[str]:
        """
        Return the preferred dataset format.

        Parquet is preferred whenever it exists. CSV is used only when
        no Parquet files are available.
        """
        if self.has_parquet:
            return "parquet"

        if self.has_csv:
            return "csv"

        return None


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------


def discover_dataset_files(raw_dir: Path | str) -> DiscoveredDataset:
    """
    Discover Parquet and CSV files under the CIC-IDS2017 raw directory.

    Discovery rules
    ---------------

    Parquet:
        1. If raw_dir/parquet/ exists and contains Parquet files,
           ONLY those files are used.
        2. Otherwise, Parquet files directly under raw_dir are used.

    CSV:
        1. If raw_dir/csv/ exists and contains CSV files,
           ONLY those files are used.
        2. Otherwise, CSV files directly under raw_dir are used.

    The function intentionally does NOT use rglob("*").

    Why?
    ----
    Some CIC-IDS2017 distributions contain duplicate copies of the
    dataset, for example:

        CIC-IDS2017/
            Benign-Monday-no-metadata.parquet
            ...
            parquet/
                Benign-Monday-no-metadata.parquet
                ...

    A recursive rglob("*") would discover both copies and process
    the dataset twice.

    This implementation treats parquet/ and csv/ as canonical
    directories when they exist.

    Parameters
    ----------
    raw_dir:
        Path to data/raw/CIC-IDS2017.

    Returns
    -------
    DiscoveredDataset
        Sorted lists of discovered Parquet and CSV files.

    Raises
    ------
    FileNotFoundError
        If raw_dir does not exist.
    """

    raw_dir = Path(raw_dir)

    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw dataset directory does not exist: {raw_dir}. "
            "This function only discovers files already on disk; "
            "it does not download anything."
        )

    if not raw_dir.is_dir():
        raise FileNotFoundError(
            f"Raw dataset path is not a directory: {raw_dir}"
        )

    parquet_files: list[Path] = []
    csv_files: list[Path] = []

    # ------------------------------------------------------------------
    # Parquet discovery
    # ------------------------------------------------------------------

    parquet_dir = raw_dir / "parquet"

    if parquet_dir.is_dir():
        parquet_files = [
            path
            for path in parquet_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in PARQUET_SUFFIXES
        ]

        if parquet_files:
            logger.info(
                "Using canonical Parquet directory: %s",
                parquet_dir,
            )

    # If parquet/ is missing or empty, look only at files directly
    # inside raw_dir. Do NOT recursively search.
    if not parquet_files:
        parquet_files = [
            path
            for path in raw_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in PARQUET_SUFFIXES
        ]

        if parquet_files:
            logger.info(
                "Using root-level Parquet files under: %s",
                raw_dir,
            )

    # ------------------------------------------------------------------
    # CSV discovery
    # ------------------------------------------------------------------

    csv_dir = raw_dir / "csv"

    if csv_dir.is_dir():
        csv_files = [
            path
            for path in csv_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in CSV_SUFFIXES
        ]

        if csv_files:
            logger.info(
                "Using canonical CSV directory: %s",
                csv_dir,
            )

    # If csv/ is missing or empty, look only at files directly inside
    # raw_dir. Again, do NOT recursively search.
    if not csv_files:
        csv_files = [
            path
            for path in raw_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in CSV_SUFFIXES
        ]

        if csv_files:
            logger.info(
                "Using root-level CSV files under: %s",
                raw_dir,
            )

    # ------------------------------------------------------------------
    # Remove duplicate paths and sort deterministically
    # ------------------------------------------------------------------

    parquet_files = sorted(
        set(parquet_files),
        key=lambda path: path.name.lower(),
    )

    csv_files = sorted(
        set(csv_files),
        key=lambda path: path.name.lower(),
    )

    dataset = DiscoveredDataset(
        parquet_files=parquet_files,
        csv_files=csv_files,
    )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    logger.info(
        "Discovered %d Parquet file(s) and %d CSV file(s) under %s",
        len(dataset.parquet_files),
        len(dataset.csv_files),
        raw_dir,
    )

    for path in dataset.parquet_files:
        logger.info(
            "Selected Parquet file: %s",
            path,
        )

    for path in dataset.csv_files:
        logger.info(
            "Selected CSV file: %s",
            path,
        )

    return dataset


# ---------------------------------------------------------------------------
# Parquet loading
# ---------------------------------------------------------------------------


def iter_parquet_chunks(
    files: Sequence[Path],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    columns: Optional[Sequence[str]] = None,
) -> Iterator[pd.DataFrame]:
    """
    Yield manageable batches from Parquet files.

    PyArrow may expose some CIC-IDS2017 integer columns as int32 even
    though the actual dataset contains values outside the int32 range.

    Before yielding each chunk, integer columns are promoted to int64.
    This prevents overflow when the cleaned data is subsequently written
    back to Parquet.

    No values are clipped, replaced, or otherwise modified.
    """

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "pyarrow is required to read Parquet files. "
            "Install it with: pip install pyarrow"
        ) from exc

    for file_path in files:
        logger.info(
            "Reading Parquet file: %s",
            file_path,
        )

        parquet_file = pq.ParquetFile(str(file_path))

        for batch in parquet_file.iter_batches(
            batch_size=chunk_size,
            columns=list(columns) if columns else None,
        ):
            chunk = batch.to_pandas()

            # ---------------------------------------------------------
            # Promote integer columns to int64.
            #
            # CIC-IDS2017 contains some values in columns such as
            # "Fwd Header Length" that exceed the signed int32 range.
            #
            # We deliberately preserve the original values rather
            # than clipping or replacing them.
            # ---------------------------------------------------------
            integer_columns = chunk.select_dtypes(
                include=["integer"]
            ).columns

            for column in integer_columns:
                if chunk[column].isna().any():
                    # Nullable integer columns need pandas' nullable
                    # Int64 dtype.
                    chunk[column] = chunk[column].astype("Int64")
                else:
                    chunk[column] = chunk[column].astype("int64")

            # Preserve source-file information for downstream
            # capture-day detection and auditing.
            chunk.attrs["source_file"] = str(file_path)

            yield chunk

# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------


def iter_csv_chunks(
    files: Sequence[Path],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    usecols: Optional[Sequence[str]] = None,
    encoding: str = "utf-8",
) -> Iterator[pd.DataFrame]:
    """
    Yield fixed-size chunks from CSV files.

    Uses pandas' chunked CSV reader so memory consumption is bounded.

    CIC-IDS2017 CSV files may use UTF-8 or Latin-1 encoding. If UTF-8
    decoding fails, the reader automatically retries with Latin-1.

    Parameters
    ----------
    files:
        CSV files to process.

    chunk_size:
        Number of rows per chunk.

    usecols:
        Optional subset of columns.

    encoding:
        Initial encoding to attempt.

    Yields
    ------
    pandas.DataFrame
        One DataFrame per CSV chunk.
    """

    for file_path in files:
        logger.info(
            "Reading CSV file in chunks "
            "(chunksize=%d): %s",
            chunk_size,
            file_path,
        )

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
                # Normalize whitespace around column names.
                chunk.columns = [
                    str(column).strip()
                    for column in chunk.columns
                ]

                chunk.attrs["source_file"] = str(file_path)

                yield chunk

        except UnicodeDecodeError:
            logger.warning(
                "UTF-8 decoding failed for %s. "
                "Retrying with latin-1.",
                file_path,
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
                chunk.columns = [
                    str(column).strip()
                    for column in chunk.columns
                ]

                chunk.attrs["source_file"] = str(file_path)

                yield chunk


# ---------------------------------------------------------------------------
# High-level dataset iterator
# ---------------------------------------------------------------------------


def iter_dataset_chunks(
    raw_dir: Path | str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    columns: Optional[Sequence[str]] = None,
    prefer: Optional[str] = None,
) -> Iterator[pd.DataFrame]:
    """
    Iterate through the dataset in memory-conscious chunks.

    Format selection:
        * Parquet is preferred by default.
        * CSV is used when no Parquet files are available.
        * The caller can explicitly force "parquet" or "csv".

    Parameters
    ----------
    raw_dir:
        Root directory containing the CIC-IDS2017 dataset.

    chunk_size:
        Number of rows per chunk/batch.

    columns:
        Optional subset of columns.

    prefer:
        Optional explicit format:
            "parquet"
            "csv"

    Yields
    ------
    pandas.DataFrame
        Successive chunks of the dataset.

    Raises
    ------
    FileNotFoundError
        If no usable dataset files exist.

    ValueError
        If an unsupported format is requested.
    """

    dataset = discover_dataset_files(raw_dir)

    if dataset.is_empty:
        raise FileNotFoundError(
            f"No Parquet or CSV files found under {raw_dir}. "
            "Place the CIC-IDS2017 files there before running "
            "the preprocessing pipeline."
        )

    # ---------------------------------------------------------------
    # Determine format
    # ---------------------------------------------------------------

    if prefer is None:
        fmt = dataset.preferred_format
    else:
        fmt = prefer.lower().strip()

    # ---------------------------------------------------------------
    # Parquet path
    # ---------------------------------------------------------------

    if fmt == "parquet":
        if not dataset.has_parquet:
            raise FileNotFoundError(
                f"No Parquet files found under {raw_dir}."
            )

        logger.info(
            "Loading dataset using PARQUET path."
        )

        yield from iter_parquet_chunks(
            dataset.parquet_files,
            chunk_size=chunk_size,
            columns=columns,
        )

        return

    # ---------------------------------------------------------------
    # CSV path
    # ---------------------------------------------------------------

    if fmt == "csv":
        if not dataset.has_csv:
            raise FileNotFoundError(
                f"No CSV files found under {raw_dir}."
            )

        logger.info(
            "Loading dataset using CHUNKED CSV path."
        )

        yield from iter_csv_chunks(
            dataset.csv_files,
            chunk_size=chunk_size,
            usecols=columns,
        )

        return

    # ---------------------------------------------------------------
    # Invalid format
    # ---------------------------------------------------------------

    raise ValueError(
        f"Unknown format preference: {prefer!r}. "
        "Expected 'parquet', 'csv', or None."
    )


# ---------------------------------------------------------------------------
# Schema inspection
# ---------------------------------------------------------------------------


def peek_schema(
    raw_dir: Path | str,
    n_rows: int = 5,
) -> pd.DataFrame:
    """
    Read a small sample of the dataset.

    This function is intended for schema inspection and does not
    materialize the complete dataset.

    Parameters
    ----------
    raw_dir:
        Dataset root directory.

    n_rows:
        Number of rows to return.

    Returns
    -------
    pandas.DataFrame
        First n_rows from the first available dataset chunk.
    """

    if n_rows <= 0:
        raise ValueError(
            f"n_rows must be greater than zero; got {n_rows}"
        )

    for chunk in iter_dataset_chunks(
        raw_dir,
        chunk_size=max(n_rows, 1000),
    ):
        return chunk.head(n_rows)

    return pd.DataFrame()