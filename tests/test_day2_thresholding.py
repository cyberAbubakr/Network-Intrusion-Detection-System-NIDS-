"""Lightweight unit tests for src.day2.thresholding using tiny synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.day2.thresholding import (
    DEFAULT_THRESHOLD_GRID,
    RF_DAY1_FROZEN_THRESHOLD,
    evaluate_frozen_threshold,
    evaluate_threshold_grid,
    identify_unseen_classes,
    per_class_frozen_results,
    select_threshold_from_validation,
)


def _make_synthetic_scores(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, size=n)
    # Probabilities correlated with the true label, plus noise, so the
    # threshold grid actually differentiates.
    y_proba = np.clip(y_true * 0.6 + rng.normal(0, 0.2, size=n) + 0.2, 0, 1)
    return y_true, y_proba


def test_evaluate_threshold_grid_returns_one_row_per_threshold():
    y_true, y_proba = _make_synthetic_scores()
    grid = evaluate_threshold_grid(y_true, y_proba)
    assert len(grid) == len(DEFAULT_THRESHOLD_GRID)
    assert set(grid["threshold"]) == set(DEFAULT_THRESHOLD_GRID)


def test_select_threshold_from_validation_picks_grid_member():
    y_true, y_proba = _make_synthetic_scores()
    selection = select_threshold_from_validation(y_true, y_proba)
    assert selection.threshold in DEFAULT_THRESHOLD_GRID
    assert selection.selection_metric == "f1"
    assert 0.0 <= selection.validation_metrics["f1"] <= 1.0


def test_selected_threshold_maximizes_f1_on_grid():
    y_true, y_proba = _make_synthetic_scores()
    selection = select_threshold_from_validation(y_true, y_proba)
    best_f1_in_grid = selection.grid["f1"].max()
    assert selection.validation_metrics["f1"] == best_f1_in_grid


def test_evaluate_frozen_threshold_does_not_change_input_threshold():
    y_true, y_proba = _make_synthetic_scores()
    frozen = evaluate_frozen_threshold(y_true, y_proba, threshold=0.3)
    assert frozen["threshold"] == 0.3
    assert "roc_auc" in frozen
    assert "pr_auc" in frozen


def test_per_class_frozen_results_groups_by_class():
    y_proba = np.array([0.9, 0.1, 0.8, 0.05])
    y_pred = np.array([1, 0, 1, 0])
    classes = pd.Series(["DDoS", "Benign", "DDoS", "Benign"])

    result = per_class_frozen_results(y_proba, y_pred, classes)

    assert set(result["class"]) == {"DDoS", "Benign"}
    ddos_row = result.loc[result["class"] == "DDoS"].iloc[0]
    assert ddos_row["samples"] == 2
    assert ddos_row["predicted_attack"] == 2


def test_identify_unseen_classes():
    train_classes = ["Benign", "DoS Hulk", "FTP-Patator"]
    test_classes = ["Benign", "DDoS", "PortScan", "Bot"]

    unseen = identify_unseen_classes(train_classes, test_classes)

    assert unseen == sorted(["DDoS", "PortScan", "Bot"])
    assert "Benign" not in unseen


def test_rf_day1_frozen_threshold_is_fixed_at_point_zero_one():
    # This is a fixed, documented constant (Day 1's established RF
    # threshold), not something re-derived at runtime.
    assert RF_DAY1_FROZEN_THRESHOLD == 0.01
    assert RF_DAY1_FROZEN_THRESHOLD in DEFAULT_THRESHOLD_GRID


def test_independent_threshold_selection_for_different_score_scales():
    # Regression test: RF, IsolationForest, and Hybrid scores live on
    # different scales and must each get their own threshold selected
    # from validation, rather than one selection being reused across
    # detectors.
    y_val = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    # RF-like scores: well-separated, best threshold should sit low.
    rf_scores = np.array([0.02, 0.03, 0.01, 0.04, 0.9, 0.95, 0.85, 0.92])

    # A different score distribution (e.g. hybrid/anomaly-like) where
    # the best separating threshold on the grid is higher.
    other_scores = np.array([0.1, 0.15, 0.05, 0.2, 0.45, 0.5, 0.4, 0.48])

    rf_selection = select_threshold_from_validation(y_val, rf_scores)
    other_selection = select_threshold_from_validation(y_val, other_scores)

    # Selections are computed independently and need not match.
    assert rf_selection.threshold != other_selection.threshold
    # Each selection's frozen validation metrics correspond to its own
    # threshold and its own scores, not a shared/reused threshold.
    assert rf_selection.validation_metrics["threshold"] == rf_selection.threshold
    assert other_selection.validation_metrics["threshold"] == other_selection.threshold
