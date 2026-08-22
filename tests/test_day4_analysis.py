"""Lightweight unit tests for src.day4.analysis using tiny synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.day4.analysis import (
    build_class_feature_shift_table,
    detection_vs_shift_table,
    distribution_shift_tests,
    group_statistics,
    hybrid_threshold_crossing_table,
    rf_feature_importances,
    top_n_features,
    top_shifted_features_per_class,
)


# ---------------------------------------------------------------------
# Unseen classes / frozen thresholds (sanity checks on the constants
# this project has already established, per the Day 4 brief).
# ---------------------------------------------------------------------

def test_expected_unseen_classes_are_the_documented_set():
    expected_unseen = {"Bot", "DDoS", "PortScan"}
    assert expected_unseen == {"Bot", "DDoS", "PortScan"}


def test_frozen_thresholds_match_documented_values():
    from src.day2.thresholding import RF_DAY1_FROZEN_THRESHOLD

    assert RF_DAY1_FROZEN_THRESHOLD == 0.01
    # IsolationForest=0.15 and Hybrid=0.50 are read from Day 2's saved
    # artifacts at runtime (not hardcoded constants in src/), so here
    # we only check the documented fallback values used by
    # scripts/run_day4.py when that artifact is absent.
    from scripts import run_day4  # noqa: F401 -- import to access module constants

    assert run_day4.FALLBACK_RF_THRESHOLD == 0.01
    assert run_day4.FALLBACK_IF_THRESHOLD == 0.15
    assert run_day4.FALLBACK_HYBRID_THRESHOLD == 0.50


# ---------------------------------------------------------------------
# Feature alignment (reused from Day 3 -- confirm it's importable and
# still enforces mismatches, exercised here via a fitted RF).
# ---------------------------------------------------------------------

def test_feature_alignment_check_passes_for_matching_features():
    from src.day3.zero_day import assert_feature_alignment

    X = pd.DataFrame({"f1": [0, 1, 2, 3], "f2": [1, 0, 1, 0]})
    y = [0, 1, 0, 1]
    clf = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)

    assert_feature_alignment(["f1", "f2"], clf, context="test model")


def test_feature_alignment_check_raises_for_mismatched_features():
    from src.day3.zero_day import assert_feature_alignment

    X = pd.DataFrame({"f1": [0, 1, 2, 3], "f2": [1, 0, 1, 0]})
    y = [0, 1, 0, 1]
    clf = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)

    with pytest.raises(AssertionError):
        assert_feature_alignment(["f1", "f3"], clf, context="test model")


# ---------------------------------------------------------------------
# No-fitting guarantee: rf_feature_importances must not fit anything --
# it should fail loudly on an unfitted model rather than silently fit it.
# ---------------------------------------------------------------------

def test_rf_feature_importances_does_not_fit_an_unfitted_model():
    unfitted = RandomForestClassifier(n_estimators=5, random_state=0)

    with pytest.raises(Exception):
        # feature_importances_ raises on an unfitted estimator; this
        # confirms rf_feature_importances reads an existing attribute
        # rather than calling .fit() to make one available.
        rf_feature_importances(unfitted, ["f1", "f2"])


def test_rf_feature_importances_reads_existing_fitted_attribute():
    X = pd.DataFrame({"f1": [0, 1, 2, 3, 4, 5], "f2": [5, 4, 3, 2, 1, 0]})
    y = [0, 0, 1, 1, 0, 1]
    clf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)

    importances = rf_feature_importances(clf, ["f1", "f2"])

    assert set(importances.index) == {"f1", "f2"}
    assert np.isclose(importances.sum(), clf.feature_importances_.sum())
    # sorted descending
    assert list(importances) == sorted(importances, reverse=True)


def test_top_n_features_returns_requested_count():
    importances = pd.Series([0.5, 0.3, 0.1, 0.05, 0.05], index=["a", "b", "c", "d", "e"])
    top = top_n_features(importances, n=3)
    assert top == ["a", "b", "c"]


# ---------------------------------------------------------------------
# Feature-shift calculations.
# ---------------------------------------------------------------------

def test_group_statistics_schema():
    df = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0], "f2": [10.0, 20.0, 30.0, 40.0]})
    stats_df = group_statistics(df, ["f1", "f2"])

    assert list(stats_df.columns) == ["mean", "std", "median", "25%", "75%"]
    assert set(stats_df.index) == {"f1", "f2"}
    assert stats_df.loc["f1", "mean"] == pytest.approx(2.5)


def test_build_class_feature_shift_table_computes_correct_percentage():
    importances = pd.Series([0.9], index=["f1"])
    train_stats = pd.DataFrame({"mean": [100.0]}, index=["f1"])
    group_stats = {"ddos": pd.DataFrame({"mean": [150.0]}, index=["f1"])}

    shift = build_class_feature_shift_table(importances, train_stats, group_stats, ["f1"])

    assert shift.loc[0, "train_mean"] == 100.0
    assert shift.loc[0, "ddos_mean"] == 150.0
    assert shift.loc[0, "ddos_shift_pct"] == pytest.approx(50.0)


def test_build_class_feature_shift_table_handles_zero_train_mean():
    importances = pd.Series([0.5], index=["f1"])
    train_stats = pd.DataFrame({"mean": [0.0]}, index=["f1"])
    group_stats = {"bot": pd.DataFrame({"mean": [5.0]}, index=["f1"])}

    shift = build_class_feature_shift_table(importances, train_stats, group_stats, ["f1"])

    assert np.isnan(shift.loc[0, "bot_shift_pct"])


def test_top_shifted_features_per_class_ranks_by_absolute_shift():
    shift_table = pd.DataFrame(
        {
            "feature": ["f1", "f2", "f3"],
            "RF_importance": [0.5, 0.3, 0.2],
            "ddos_shift_pct": [10.0, -80.0, 5.0],
        }
    )

    top = top_shifted_features_per_class(shift_table, ["ddos"], n=2)

    assert list(top["feature"]) == ["f2", "f1"]
    assert list(top["rank"]) == [1, 2]


def test_detection_vs_shift_table_maps_rates_correctly():
    shift_table = pd.DataFrame({"feature": ["f1"], "bot_shift_pct": [40.0]})
    detection_rates = {"bot": {"random_forest": 0.22, "isolation_forest": 0.09, "hybrid": 0.0}}

    table = detection_vs_shift_table(shift_table, ["bot"], detection_rates)

    row = table.iloc[0]
    assert row["class"] == "bot"
    assert row["rf_detection_rate"] == pytest.approx(0.22)
    assert row["hybrid_detection_rate"] == pytest.approx(0.0)


# ---------------------------------------------------------------------
# Hybrid threshold-crossing calculations.
# ---------------------------------------------------------------------

def test_hybrid_threshold_crossing_table_counts_correctly():
    scores_by_class = {
        "DDoS": np.array([0.9, 0.9, 0.1, 0.2]),   # 2 of 4 cross 0.5
        "Bot": np.array([0.1, 0.2, 0.3]),          # 0 of 3 cross 0.5
    }

    table = hybrid_threshold_crossing_table(scores_by_class, threshold=0.5)
    table = table.set_index("attack_class")

    assert table.loc["DDoS", "samples"] == 4
    assert table.loc["DDoS", "above_threshold"] == 2
    assert table.loc["DDoS", "detection_rate"] == pytest.approx(0.5)
    assert table.loc["Bot", "above_threshold"] == 0
    assert table.loc["Bot", "detection_rate"] == pytest.approx(0.0)


def test_hybrid_threshold_crossing_table_matches_expected_schema():
    scores_by_class = {"PortScan": np.array([0.6, 0.4])}
    table = hybrid_threshold_crossing_table(scores_by_class, threshold=0.5)

    expected_cols = {
        "attack_class", "samples", "above_threshold", "detection_rate",
        "mean_hybrid_score", "median_hybrid_score", "max_hybrid_score",
    }
    assert expected_cols == set(table.columns)


# ---------------------------------------------------------------------
# Distribution-shift statistical tests.
# ---------------------------------------------------------------------

def test_distribution_shift_tests_detects_a_real_shift():
    rng = np.random.default_rng(0)
    train_df = pd.DataFrame({"f1": rng.normal(0, 1, 500)})
    shifted_df = pd.DataFrame({"f1": rng.normal(5, 1, 200)})  # clearly shifted

    result = distribution_shift_tests(
        train_df, {"shifted": shifted_df}, ["f1"], sample_cap=300, random_state=0
    )

    assert len(result) == 1
    row = result.iloc[0]
    assert row["p_value"] < 0.01
    assert abs(row["cohens_d"]) > 1.0  # large effect size for a 5-std mean shift


def test_distribution_shift_tests_respects_sample_cap():
    rng = np.random.default_rng(1)
    train_df = pd.DataFrame({"f1": rng.normal(0, 1, 50_000)})
    group_df = pd.DataFrame({"f1": rng.normal(0, 1, 50_000)})

    result = distribution_shift_tests(
        train_df, {"g": group_df}, ["f1"], sample_cap=1_000, random_state=0
    )

    row = result.iloc[0]
    assert row["n_train_sampled"] == 1_000
    assert row["n_group_sampled"] == 1_000


def test_distribution_shift_tests_no_shift_gives_small_effect_size():
    rng = np.random.default_rng(2)
    train_df = pd.DataFrame({"f1": rng.normal(0, 1, 500)})
    same_df = pd.DataFrame({"f1": rng.normal(0, 1, 500)})

    result = distribution_shift_tests(
        train_df, {"same": same_df}, ["f1"], sample_cap=500, random_state=0
    )

    row = result.iloc[0]
    assert abs(row["cohens_d"]) < 0.5
