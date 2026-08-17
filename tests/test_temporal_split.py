"""Lightweight unit tests for src.data.temporal_split using tiny synthetic data."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.temporal_split import (
    chronological_train_test_split,
    detect_timestamp_column,
    verify_no_temporal_leakage,
)


def _make_synthetic_timeseries(n_rows: int = 100) -> pd.DataFrame:
    timestamps = pd.date_range("2017-07-01", periods=n_rows, freq="min")
    return pd.DataFrame(
        {
            "Timestamp": timestamps,
            "Feature A": range(n_rows),
            "Label": ["BENIGN" if i % 3 else "DoS Hulk" for i in range(n_rows)],
        }
    )


def test_detect_timestamp_column_by_name():
    df = _make_synthetic_timeseries(10)
    result = detect_timestamp_column(df)
    assert result.column == "Timestamp"
    assert result.parsed_successfully


def test_detect_timestamp_column_explicit_override():
    df = _make_synthetic_timeseries(10)
    df = df.rename(columns={"Timestamp": "flow_start_time"})
    result = detect_timestamp_column(df, explicit_column="flow_start_time")
    assert result.column == "flow_start_time"


def test_detect_timestamp_column_none_found():
    df = pd.DataFrame({"A": [1, 2, 3]})
    result = detect_timestamp_column(df)
    assert result.column is None
    assert not result.parsed_successfully


def test_detect_timestamp_column_missing_explicit_raises():
    df = pd.DataFrame({"A": [1, 2, 3]})
    with pytest.raises(KeyError):
        detect_timestamp_column(df, explicit_column="does_not_exist")


def test_chronological_split_produces_strict_time_order():
    df = _make_synthetic_timeseries(100)
    split = chronological_train_test_split(df, timestamp_column="Timestamp", test_size=0.2)

    assert len(split.train_df) + len(split.test_df) == 100
    assert len(split.test_df) == 20
    assert split.train_end <= split.test_start


def test_chronological_split_invalid_test_size_raises():
    df = _make_synthetic_timeseries(10)
    with pytest.raises(ValueError):
        chronological_train_test_split(df, timestamp_column="Timestamp", test_size=1.5)


def test_chronological_split_missing_column_raises():
    df = _make_synthetic_timeseries(10).drop(columns=["Timestamp"])
    with pytest.raises(KeyError):
        chronological_train_test_split(df, timestamp_column="Timestamp", test_size=0.2)


def test_chronological_split_unparseable_timestamps_raise():
    df = _make_synthetic_timeseries(10)
    # Cast to object dtype first so a bad string can be assigned; a native
    # datetime64 column would reject the assignment outright.
    df["Timestamp"] = df["Timestamp"].astype(object)
    df.loc[3, "Timestamp"] = "not-a-real-date"
    with pytest.raises(ValueError):
        chronological_train_test_split(df, timestamp_column="Timestamp", test_size=0.2)


def test_verify_no_temporal_leakage_passes_for_valid_split():
    df = _make_synthetic_timeseries(100)
    split = chronological_train_test_split(df, timestamp_column="Timestamp", test_size=0.2)
    check = verify_no_temporal_leakage(split)
    assert check.passed
    assert check.n_overlapping_rows == 0


def test_verify_no_temporal_leakage_fails_for_shuffled_split():
    df = _make_synthetic_timeseries(100)
    split = chronological_train_test_split(df, timestamp_column="Timestamp", test_size=0.2)

    # Deliberately corrupt the split to simulate a leakage bug: swap one
    # late train-row's timestamp with an early test-row's timestamp so
    # they overlap.
    bad_train = split.train_df.copy()
    bad_test = split.test_df.copy()
    bad_train.iloc[-1, bad_train.columns.get_loc("Timestamp")] = bad_test.iloc[0]["Timestamp"]

    split.train_df = bad_train
    check = verify_no_temporal_leakage(split)
    assert not check.passed
    assert check.n_overlapping_rows > 0
