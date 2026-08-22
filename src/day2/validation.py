"""
Day 2 - Step 1: Validation split.

Builds a validation set for threshold selection *entirely inside* the
Day 1 training period (Monday-Thursday), so that Friday -- the frozen
cross-temporal test set -- is never touched during threshold selection.

Default strategy
-----------------
Monday-Wednesday -> "sub-train" period (documented only; the existing
                     Day 1 Random Forest baseline is reused as-is and is
                     NOT retrained on this subset -- see module docstring
                     caveat below).
Thursday         -> validation period. The existing baseline's
                     predict_proba on Thursday is used for threshold
                     selection.

This module is a thin wrapper around the already-verified Day 1
temporal-splitting utilities in ``src.data.temporal_split``
(``day_based_train_test_split`` / ``verify_no_day_leakage``); it does not
reimplement splitting or leakage-checking logic.

Known limitation (documented, not hidden)
------------------------------------------
The Day 2 brief requires reusing the *existing* trained Random Forest
model/pipeline rather than retraining it (retraining is explicitly out
of scope: "Do not replace the existing Random Forest baseline"). That
existing model was fit on the full Monday-Thursday training set, which
includes Thursday. Consequently, Thursday is not fully "unseen" by the
model itself -- only Friday is a genuine, never-touched holdout.
Threshold selection on Thursday is still valid and useful (it avoids
picking a threshold by looking at Friday), but it should be understood
as "in-sample validation for thresholding" rather than a fully
independent validation set. This is reported explicitly in the Day 2
results rather than glossed over.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

import pandas as pd

from src.data.temporal_split import (
    DaySplitResult,
    day_based_train_test_split,
    verify_no_day_leakage,
)

logger = logging.getLogger(__name__)

DEFAULT_SUBTRAIN_DAYS: tuple[str, ...] = ("monday", "tuesday", "wednesday")
DEFAULT_VALIDATION_DAYS: tuple[str, ...] = ("thursday",)


@dataclass
class ValidationSplit:
    """Result of building the Day 2 validation split."""

    subtrain_df: pd.DataFrame
    val_df: pd.DataFrame
    split: DaySplitResult
    leakage_passed: bool
    leakage_message: str

    def summary(self) -> dict:
        return {
            "subtrain_days": self.split.train_days,
            "validation_days": self.split.test_days,
            "subtrain_rows": len(self.subtrain_df),
            "validation_rows": len(self.val_df),
            "leakage_check_passed": self.leakage_passed,
            "leakage_check_message": self.leakage_message,
            "note": (
                "Friday is not part of train_df and is therefore "
                "untouched by this split."
            ),
        }


def build_validation_split(
    train_df: pd.DataFrame,
    day_column: str = "capture_day",
    validation_days: Sequence[str] = DEFAULT_VALIDATION_DAYS,
    subtrain_days: Optional[Sequence[str]] = DEFAULT_SUBTRAIN_DAYS,
) -> ValidationSplit:
    """
    Split the Day 1 training DataFrame (Monday-Thursday) into a
    documented sub-training period and a validation period, using the
    same capture-day methodology already verified in Day 1.

    Parameters
    ----------
    train_df:
        The Day 1 training DataFrame, i.e. the contents of
        ``data/processed/day1/train.parquet``. Must still contain the
        ``day_column`` (Day 1's ``prepare_data.py`` excludes it from
        model *features* but keeps it as a column in the parquet file).

    day_column:
        Name of the capture-day column. Default ``"capture_day"``.

    validation_days:
        Capture day(s) held out for validation / threshold selection.
        Default ``("thursday",)``.

    subtrain_days:
        Capture day(s) documented as the sub-training period. Only used
        for bookkeeping/leakage verification here -- the existing Day 1
        model is reused, not retrained (see module docstring).

    Returns
    -------
    ValidationSplit

    Raises
    ------
    KeyError
        If ``day_column`` is missing from ``train_df``.

    RuntimeError
        If the leakage check on the resulting split fails.
    """

    if day_column not in train_df.columns:
        raise KeyError(
            f"{day_column!r} not found in train_df columns "
            f"({list(train_df.columns)[:10]}...). "
            "build_validation_split expects the raw Day 1 "
            "data/processed/day1/train.parquet, which retains "
            "capture_day as a column even though it is excluded from "
            "the model's feature_names."
        )

    split = day_based_train_test_split(
        train_df,
        day_column=day_column,
        test_days=list(validation_days),
        train_days=list(subtrain_days) if subtrain_days else None,
    )

    leakage_check = verify_no_day_leakage(split)

    if not leakage_check.passed:
        raise RuntimeError(
            f"Validation split leakage check failed: {leakage_check.message}"
        )

    logger.info(
        "Day 2 validation split built: subtrain_days=%s (%d rows), "
        "validation_days=%s (%d rows). Friday is not present in "
        "train_df and remains untouched.",
        split.train_days,
        len(split.train_df),
        split.test_days,
        len(split.test_df),
    )

    return ValidationSplit(
        subtrain_df=split.train_df,
        val_df=split.test_df,
        split=split,
        leakage_passed=leakage_check.passed,
        leakage_message=leakage_check.message,
    )
