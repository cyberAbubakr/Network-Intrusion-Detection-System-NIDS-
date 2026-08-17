"""Lightweight unit tests for src.data.cleaner and src.data.feature_engineering
using tiny synthetic data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.cleaner import handle_nan_inf, normalize_labels
from src.data.feature_engineering import (
    identify_leakage_prone_features,
    select_features,
)


# ---------------------------------------------------------------------------
# cleaner.py
# ---------------------------------------------------------------------------


def test_normalize_labels_strips_whitespace_and_derives_binary():
    df = pd.DataFrame({"Label": [" BENIGN", "BENIGN", "DoS Hulk ", "PortScan"]})
    out, log = normalize_labels(df, label_col="Label")

    assert out["label_multiclass"].tolist() == ["BENIGN", "BENIGN", "DoS Hulk", "PortScan"]
    assert out["label_binary"].tolist() == [0, 0, 1, 1]
    assert "label_multiclass_raw" in out.columns
    assert len(log.entries) > 0


def test_normalize_labels_missing_column_raises():
    df = pd.DataFrame({"A": [1, 2]})
    with pytest.raises(KeyError):
        normalize_labels(df, label_col="Label")


def test_normalize_labels_custom_benign_aliases():
    df = pd.DataFrame({"Label": ["Normal", "Attack", "normal"]})
    out, _ = normalize_labels(df, label_col="Label", benign_aliases={"normal"})
    assert out["label_binary"].tolist() == [0, 1, 0]


def test_handle_nan_inf_replaces_inf_with_nan():
    df = pd.DataFrame({"a": [1.0, np.inf, -np.inf, 4.0]})
    out, log = handle_nan_inf(df)
    assert out["a"].isna().sum() == 2
    assert not np.isinf(out["a"]).any()
    assert len(log.entries) > 0


def test_handle_nan_inf_does_not_drop_or_impute_nan():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
    out, _ = handle_nan_inf(df)
    assert len(out) == len(df)  # no rows dropped
    assert out["a"].isna().sum() == 1  # NaN left as-is


def test_handle_nan_inf_keep_strategy_leaves_inf():
    df = pd.DataFrame({"a": [1.0, np.inf]})
    out, _ = handle_nan_inf(df, inf_strategy="keep")
    assert np.isinf(out["a"]).any()


# ---------------------------------------------------------------------------
# feature_engineering.py
# ---------------------------------------------------------------------------


def test_identify_leakage_prone_features_flags_identifier_columns():
    df = pd.DataFrame(
        {
            "Flow ID": ["1-2-3"] * 5,
            "Src IP": ["10.0.0.1"] * 5,
            "Dst Port": [80] * 5,
            "Feature A": [1, 2, 3, 4, 5],
            "Label": ["BENIGN"] * 5,
        }
    )
    report = identify_leakage_prone_features(df, primary_label_col="Label")
    assert "Flow ID" in report.identifier_like_columns
    assert "Src IP" in report.identifier_like_columns
    assert "Dst Port" in report.identifier_like_columns
    assert "Feature A" not in report.flagged_columns


def test_identify_leakage_prone_features_flags_near_constant():
    df = pd.DataFrame(
        {
            "Feature A": [1] * 999 + [2],
            "Label": ["BENIGN"] * 1000,
        }
    )
    report = identify_leakage_prone_features(df, near_constant_threshold=0.99)
    assert "Feature A" in report.near_constant_columns


def test_identify_leakage_prone_features_flags_duplicate_label_like_column():
    df = pd.DataFrame(
        {
            "Label": ["BENIGN", "DoS"],
            "Attack Category": ["none", "dos"],
        }
    )
    report = identify_leakage_prone_features(df, primary_label_col="Label")
    assert "Attack Category" in report.label_like_columns


def test_select_features_excludes_and_drops_correctly():
    df = pd.DataFrame(
        {
            "Label": ["BENIGN", "DoS", "BENIGN", "DoS"],
            "Flow ID": ["a", "b", "c", "d"],
            "Feature A": [1, 2, 3, 4],
            "Feature A Copy": [1, 2, 3, 4],  # exact duplicate of Feature A
            "Constant Feature": [7, 7, 7, 7],
        }
    )
    out, selected, log = select_features(
        df,
        exclude_columns=["Label", "Flow ID"],
        drop_near_constant=True,
        near_constant_threshold=0.75,
    )

    assert "Label" not in selected
    assert "Flow ID" not in selected
    assert "Feature A" in selected
    # one of the two identical columns should be dropped
    assert not ("Feature A" in selected and "Feature A Copy" in selected)
    assert "Constant Feature" not in selected
    assert len(log.entries) > 0


def test_select_features_never_drops_excluded_columns():
    df = pd.DataFrame(
        {
            "Label": ["BENIGN"] * 5,  # constant, but excluded -> must survive
            "Feature A": [1, 2, 3, 4, 5],
        }
    )
    out, selected, _ = select_features(
        df, exclude_columns=["Label"], near_constant_threshold=0.99
    )
    assert "Label" in out.columns
    assert "Label" not in selected
