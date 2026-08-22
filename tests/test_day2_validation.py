"""Lightweight unit tests for src.day2.validation using tiny synthetic data."""

from __future__ import annotations

import pandas as pd
import pytest

from src.day2.validation import build_validation_split


def _make_synthetic_train_df(rows_per_day: int = 20) -> pd.DataFrame:
    days = ["monday", "tuesday", "wednesday", "thursday"]
    rows = []
    for day in days:
        for i in range(rows_per_day):
            rows.append(
                {
                    "capture_day": day,
                    "feature_a": i,
                    "label_binary": 0 if i % 4 else 1,
                }
            )
    return pd.DataFrame(rows)


def test_validation_split_has_no_day_overlap_with_subtrain():
    train_df = _make_synthetic_train_df()
    result = build_validation_split(train_df)

    subtrain_days = set(result.subtrain_df["capture_day"].unique())
    val_days = set(result.val_df["capture_day"].unique())

    assert subtrain_days.isdisjoint(val_days)
    assert val_days == {"thursday"}
    assert subtrain_days == {"monday", "tuesday", "wednesday"}


def test_validation_split_row_counts_match_days():
    train_df = _make_synthetic_train_df(rows_per_day=20)
    result = build_validation_split(train_df)

    assert len(result.val_df) == 20
    assert len(result.subtrain_df) == 60
    assert len(result.val_df) + len(result.subtrain_df) == len(train_df)


def test_validation_split_never_produces_friday():
    # Friday is never present in train_df in the first place (Day 1's
    # split already isolated it into test.parquet), so it cannot leak
    # into either side of this split.
    train_df = _make_synthetic_train_df()
    result = build_validation_split(train_df)

    all_days = set(result.subtrain_df["capture_day"]).union(
        set(result.val_df["capture_day"])
    )
    assert "friday" not in all_days


def test_validation_split_missing_day_column_raises():
    df = pd.DataFrame({"feature_a": [1, 2, 3], "label_binary": [0, 1, 0]})
    with pytest.raises(KeyError):
        build_validation_split(df)


def test_validation_split_summary_is_serializable_dict():
    train_df = _make_synthetic_train_df()
    result = build_validation_split(train_df)
    summary = result.summary()

    assert summary["validation_days"] == ["thursday"]
    assert summary["subtrain_days"] == ["monday", "tuesday", "wednesday"]
    assert summary["leakage_check_passed"] is True
