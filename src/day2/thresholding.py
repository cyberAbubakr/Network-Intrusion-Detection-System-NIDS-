"""
Day 2 - Steps 2-4: Threshold selection (validation-only), frozen Friday
evaluation, and per-class / unseen-attack breakdown.

Rules encoded here:
    * ``select_threshold_from_validation`` must only ever be called with
      validation data (e.g. Thursday), never with Friday.
    * ``evaluate_frozen_threshold`` applies an already-chosen threshold
      unchanged -- it never searches for a "better" threshold.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

# Predefined, small threshold grid (per the Day 2 brief -- no broad
# hyperparameter search).
DEFAULT_THRESHOLD_GRID: tuple[float, ...] = (
    0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50,
)

# Day 1's Random Forest baseline threshold, established during Day 1's
# exploratory threshold analysis on Friday (best tested F1 in that
# sweep). It is adopted here as the Random Forest's frozen, reported
# production threshold for Day 2 comparisons, rather than the
# dataclass/library default of 0.5. It is a fixed constant -- it is
# never re-derived from Friday inside Day 2 code, and Day 2's own
# validation-based threshold selection (see
# ``select_threshold_from_validation``) is computed and reported
# independently alongside it for transparency.
RF_DAY1_FROZEN_THRESHOLD: float = 0.01


def compute_threshold_metrics(
    y_true: Sequence[int],
    y_proba: Sequence[float],
    threshold: float,
) -> dict:
    """Precision/recall/F1/FPR/FNR/confusion matrix at a single threshold."""

    y_true_arr = np.asarray(y_true)
    y_proba_arr = np.asarray(y_proba)
    y_pred = (y_proba_arr >= threshold).astype(int)

    precision = precision_score(y_true_arr, y_pred, zero_division=0)
    recall = recall_score(y_true_arr, y_pred, zero_division=0)
    f1 = f1_score(y_true_arr, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(
        y_true_arr, y_pred, labels=[0, 1]
    ).ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    fnr = fn / (fn + tp) if (fn + tp) > 0 else float("nan")

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def evaluate_threshold_grid(
    y_true: Sequence[int],
    y_proba: Sequence[float],
    thresholds: Sequence[float] = DEFAULT_THRESHOLD_GRID,
) -> pd.DataFrame:
    """Evaluate a predefined threshold grid and return one row per threshold."""

    rows = [
        compute_threshold_metrics(y_true, y_proba, t) for t in thresholds
    ]
    return pd.DataFrame(rows)


@dataclass
class ThresholdSelection:
    """Result of selecting a threshold from validation data only."""

    threshold: float
    selection_metric: str
    validation_metrics: dict
    grid: pd.DataFrame

    def summary(self) -> dict:
        return {
            "selected_threshold": self.threshold,
            "selection_metric": self.selection_metric,
            "validation_metrics": self.validation_metrics,
        }


def select_threshold_from_validation(
    y_val_true: Sequence[int],
    y_val_proba: Sequence[float],
    thresholds: Sequence[float] = DEFAULT_THRESHOLD_GRID,
    selection_metric: str = "f1",
) -> ThresholdSelection:
    """
    Select a classification threshold using ONLY validation data.

    Callers must pass Thursday (or another validation split) here --
    never Friday. See ``src.day2.validation`` for how the validation
    split is constructed.

    Parameters
    ----------
    y_val_true, y_val_proba:
        True binary labels and predicted attack probabilities on the
        validation set.

    thresholds:
        Predefined threshold grid to evaluate. Default
        ``DEFAULT_THRESHOLD_GRID``.

    selection_metric:
        Column in the grid to maximize. Default ``"f1"``.

    Returns
    -------
    ThresholdSelection
    """

    grid = evaluate_threshold_grid(y_val_true, y_val_proba, thresholds)

    best_idx = grid[selection_metric].idxmax()
    threshold = float(grid.loc[best_idx, "threshold"])

    validation_metrics = compute_threshold_metrics(
        y_val_true, y_val_proba, threshold
    )

    logger.info(
        "Selected threshold=%.4f from VALIDATION ONLY using metric=%s "
        "(f1=%.4f, precision=%.4f, recall=%.4f, fpr=%.4f)",
        threshold,
        selection_metric,
        validation_metrics["f1"],
        validation_metrics["precision"],
        validation_metrics["recall"],
        validation_metrics["fpr"],
    )

    return ThresholdSelection(
        threshold=threshold,
        selection_metric=selection_metric,
        validation_metrics=validation_metrics,
        grid=grid,
    )


def evaluate_frozen_threshold(
    y_true: Sequence[int],
    y_proba: Sequence[float],
    threshold: float,
) -> dict:
    """
    Apply an already-selected (frozen) threshold to a test set (e.g.
    Friday) and report full metrics, including threshold-independent
    ROC-AUC / PR-AUC. The threshold itself is never adjusted here.
    """

    metrics = compute_threshold_metrics(y_true, y_proba, threshold)

    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    except ValueError as exc:
        metrics["roc_auc"] = float("nan")
        logger.warning("ROC-AUC could not be computed (%s).", exc)

    try:
        metrics["pr_auc"] = float(average_precision_score(y_true, y_proba))
    except ValueError as exc:
        metrics["pr_auc"] = float("nan")
        logger.warning("PR-AUC could not be computed (%s).", exc)

    return metrics


def per_class_frozen_results(
    y_proba: Sequence[float],
    y_pred: Sequence[int],
    class_labels: pd.Series,
) -> pd.DataFrame:
    """
    Per multiclass-label breakdown of frozen-threshold predictions
    (e.g. Friday's Benign / DDoS / PortScan / Bot classes).

    Mirrors the shape of Day 1's
    ``results/day1/baseline_per_class_analysis.csv`` so the two are
    directly comparable.
    """

    df = pd.DataFrame(
        {
            "class": pd.Series(class_labels).astype(str).to_numpy(),
            "proba": np.asarray(y_proba),
            "pred": np.asarray(y_pred),
        }
    )

    rows = []
    for cls, group in df.groupby("class"):
        rows.append(
            {
                "class": cls,
                "samples": int(len(group)),
                "predicted_attack": int(group["pred"].sum()),
                "detection_rate": float(group["pred"].mean()),
                "mean_attack_probability": float(group["proba"].mean()),
                "median_attack_probability": float(group["proba"].median()),
                "max_attack_probability": float(group["proba"].max()),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("samples", ascending=False)
        .reset_index(drop=True)
    )


def identify_unseen_classes(
    train_classes: Sequence[str],
    test_classes: Sequence[str],
) -> list[str]:
    """
    Return the sorted list of multiclass labels present in
    ``test_classes`` (e.g. Friday) but absent from ``train_classes``
    (e.g. Monday-Thursday).

    Used for Day 2 Step 4 (unseen-attack analysis) so that "the model
    detected DDoS well" claims are not made without checking whether
    DDoS was actually present during training.
    """

    train_set = {str(c) for c in train_classes}
    test_set = {str(c) for c in test_classes}
    return sorted(test_set - train_set)
