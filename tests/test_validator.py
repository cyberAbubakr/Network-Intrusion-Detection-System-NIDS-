"""Lightweight unit tests for src.data.validator using tiny synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.validator import (
    detect_label_column,
    merge_schema_reports,
    validate_nan_inf,
    validate_schema,
)


def test_detect_label_column_exact_match():
    assert detect_label_column(["A", "B", "Label"]) == "Label"


def test_detect_label_column_whitespace_variant():
    assert detect_label_column(["A", "B", " Label"]) == " Label"


def test_detect_label_column_none_found():
    assert detect_label_column(["A", "B", "C"]) is None


def test_validate_schema_detects_duplicates_and_unnamed():
    df = pd.DataFrame(
        [[1, 2, 3, "BENIGN"]],
        columns=["A", "A", "Unnamed: 0", "Label"],
    )
    report = validate_schema(df)
    assert report.detected_label_column == "Label"
    assert "A" in report.duplicate_columns
    assert "Unnamed: 0" in report.unnamed_columns
    assert report.n_rows == 1
    assert report.n_columns == 4


def test_validate_schema_no_label_column():
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    report = validate_schema(df)
    assert report.detected_label_column is None
    assert any("No label column" in note for note in report.notes)


def test_validate_nan_inf_counts_correctly():
    df = pd.DataFrame(
        {
            "num": [1.0, np.nan, np.inf, -np.inf, 5.0],
            "cat": ["a", "b", None, "d", "e"],
        }
    )
    report = validate_nan_inf(df)
    assert report.nan_counts["num"] == 1
    assert report.nan_counts["cat"] == 1
    assert report.inf_counts["num"] == 2
    assert report.inf_counts["cat"] == 0
    assert report.has_nan
    assert report.has_inf


def test_validate_nan_inf_flags_all_nan_column():
    df = pd.DataFrame({"good": [1, 2, 3], "bad": [np.nan, np.nan, np.nan]})
    report = validate_nan_inf(df)
    assert "bad" in report.columns_all_nan
    assert "good" not in report.columns_all_nan


def test_validate_nan_inf_clean_data_has_no_issues():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    report = validate_nan_inf(df)
    assert not report.has_nan
    assert not report.has_inf


def test_merge_schema_reports_detects_drift():
    df1 = pd.DataFrame({"A": [1], "Label": ["BENIGN"]})
    df2 = pd.DataFrame({"A": [1], "B": [2], "Label": ["BENIGN"]})
    report1 = validate_schema(df1)
    report2 = validate_schema(df2)

    merged = merge_schema_reports([report1, report2])
    assert any("drift" in note.lower() for note in merged.notes)
    assert merged.n_rows == 2
