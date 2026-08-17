"""Lightweight unit tests for src.data.loader using tiny synthetic data.

No real dataset files are read anywhere in this file.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.loader import (
    discover_dataset_files,
    iter_csv_chunks,
    iter_dataset_chunks,
    iter_parquet_chunks,
)


def _make_synthetic_df(n_rows: int = 20) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Flow Duration": range(n_rows),
            "Total Fwd Packets": [i % 5 for i in range(n_rows)],
            " Label": ["BENIGN" if i % 2 == 0 else "DoS Hulk" for i in range(n_rows)],
        }
    )


def test_discover_dataset_files_empty_dir(tmp_path):
    result = discover_dataset_files(tmp_path)
    assert result.is_empty
    assert result.preferred_format is None


def test_discover_dataset_files_missing_dir(tmp_path):
    missing = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        discover_dataset_files(missing)


def test_discover_dataset_files_finds_csv_and_parquet(tmp_path):
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
    parquet_dir = tmp_path / "parquet"
    parquet_dir.mkdir()

    df = _make_synthetic_df(10)
    df.to_csv(csv_dir / "tiny.csv", index=False)
    df.to_parquet(parquet_dir / "tiny.parquet", index=False)

    result = discover_dataset_files(tmp_path)
    assert result.has_csv
    assert result.has_parquet
    assert result.preferred_format == "parquet"  # parquet preferred


def test_iter_csv_chunks_respects_chunk_size(tmp_path):
    df = _make_synthetic_df(23)
    csv_path = tmp_path / "tiny.csv"
    df.to_csv(csv_path, index=False)

    chunks = list(iter_csv_chunks([csv_path], chunk_size=10))
    sizes = [len(c) for c in chunks]

    assert sizes == [10, 10, 3]
    assert sum(sizes) == 23


def test_iter_csv_chunks_strips_column_whitespace(tmp_path):
    df = _make_synthetic_df(5)
    csv_path = tmp_path / "tiny.csv"
    df.to_csv(csv_path, index=False)

    chunk = next(iter_csv_chunks([csv_path], chunk_size=100))
    assert "Label" in chunk.columns  # " Label" stripped to "Label"
    assert " Label" not in chunk.columns


def test_iter_parquet_chunks_reads_all_rows(tmp_path):
    df = _make_synthetic_df(17)
    parquet_path = tmp_path / "tiny.parquet"
    df.to_parquet(parquet_path, index=False)

    chunks = list(iter_parquet_chunks([parquet_path], chunk_size=8))
    total_rows = sum(len(c) for c in chunks)
    assert total_rows == 17


def test_iter_dataset_chunks_prefers_parquet(tmp_path):
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
    parquet_dir = tmp_path / "parquet"
    parquet_dir.mkdir()

    csv_df = _make_synthetic_df(5)
    csv_df.to_csv(csv_dir / "tiny.csv", index=False)

    parquet_df = _make_synthetic_df(9)
    parquet_df.to_parquet(parquet_dir / "tiny.parquet", index=False)

    chunk = next(iter_dataset_chunks(tmp_path, chunk_size=100))
    # Should have read the parquet (9 rows), not the csv (5 rows).
    assert len(chunk) == 9


def test_iter_dataset_chunks_raises_when_no_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        next(iter_dataset_chunks(tmp_path))
