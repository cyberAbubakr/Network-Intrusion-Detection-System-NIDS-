"""
Day 3 - Zero-day / unseen-attack class inventory and population-building
utilities.

Scope: identify which Friday attack classes were genuinely absent from
the full Monday-Thursday supervised training period (not Day 2's
internal Monday-Wednesday sub-train, which is a smaller, different
period than what the Random Forest was actually fit on), and build the
benign+unseen-class evaluation subsets used by ``scripts/run_day3.py``.

This module does NOT retrain, refit, or modify anything from Day 1 or
Day 2. It only reads label_multiclass values and filters rows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import pandas as pd

from src.day2.thresholding import identify_unseen_classes

logger = logging.getLogger(__name__)

DEFAULT_LABEL_COL = "label_multiclass"
DEFAULT_BENIGN_LABEL = "Benign"


@dataclass
class ClassInventory:
    """
    Explicit seen/unseen/benign bookkeeping for the Day 3 zero-day
    experiment.

    ``train_classes`` / ``test_classes`` are the FULL set of
    ``label_multiclass`` values present during the actual Random Forest
    training period (Monday-Thursday) and on Friday, respectively --
    deliberately NOT Day 2's internal Monday-Wednesday validation
    sub-split, which the Random Forest was not exclusively trained on.
    """

    label_col: str
    benign_label: str
    train_classes: list[str]
    test_classes: list[str]
    seen_attack_classes: list[str]
    unseen_attack_classes: list[str]

    def summary(self) -> dict[str, Any]:
        return {
            "label_col": self.label_col,
            "benign_label": self.benign_label,
            "train_classes": self.train_classes,
            "test_classes": self.test_classes,
            "seen_attack_classes": self.seen_attack_classes,
            "unseen_attack_classes": self.unseen_attack_classes,
            "n_seen_attack_classes": len(self.seen_attack_classes),
            "n_unseen_attack_classes": len(self.unseen_attack_classes),
            "note": (
                "train_classes/seen_attack_classes are computed from the "
                "FULL Monday-Thursday training period (the data the "
                "Random Forest was actually fit on), not Day 2's smaller "
                "internal Monday-Wednesday validation sub-train."
            ),
        }


def class_inventory(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str = DEFAULT_LABEL_COL,
    benign_label: str = DEFAULT_BENIGN_LABEL,
) -> ClassInventory:
    """
    Build the seen/unseen/benign class inventory for Day 3.

    Parameters
    ----------
    train_df:
        The FULL Day 1 training DataFrame (Monday-Thursday), i.e. the
        contents of ``data/processed/day1/train.parquet`` -- the same
        data the Random Forest baseline was actually fit on.

    test_df:
        The frozen Day 1 Friday test DataFrame.

    Raises
    ------
    KeyError
        If ``label_col`` is missing from either DataFrame.
    """

    if label_col not in train_df.columns or label_col not in test_df.columns:
        raise KeyError(
            f"{label_col!r} missing from train_df/test_df. Day 3 zero-day "
            "analysis requires the multiclass label column produced by "
            "Day 1's src.data.cleaner.normalize_labels."
        )

    train_classes = sorted(str(c) for c in train_df[label_col].dropna().unique())
    test_classes = sorted(str(c) for c in test_df[label_col].dropna().unique())

    seen_attack_classes = sorted(c for c in train_classes if c != benign_label)

    unseen_attack_classes = identify_unseen_classes(train_classes, test_classes)
    # identify_unseen_classes already excludes anything present in
    # train_classes; explicitly drop benign_label too, defensively, so
    # it can never be reported as an "unseen attack class".
    unseen_attack_classes = [c for c in unseen_attack_classes if c != benign_label]

    logger.info(
        "Class inventory: %d train classes, %d Friday classes, "
        "%d seen attack classes, %d unseen attack classes: %s",
        len(train_classes), len(test_classes),
        len(seen_attack_classes), len(unseen_attack_classes), unseen_attack_classes,
    )

    return ClassInventory(
        label_col=label_col,
        benign_label=benign_label,
        train_classes=train_classes,
        test_classes=test_classes,
        seen_attack_classes=seen_attack_classes,
        unseen_attack_classes=unseen_attack_classes,
    )


def assert_no_unseen_leakage(
    train_df: pd.DataFrame,
    unseen_attack_classes: Sequence[str],
    label_col: str = DEFAULT_LABEL_COL,
) -> None:
    """
    Research-integrity check: raise if any class flagged as "unseen"
    actually appears anywhere in the supervised training data.
    """

    train_classes = {str(c) for c in train_df[label_col].dropna().unique()}
    leaked = [c for c in unseen_attack_classes if c in train_classes]
    if leaked:
        raise AssertionError(
            "Zero-day integrity check failed: class(es) flagged as unseen "
            f"actually appear in train_df: {leaked}"
        )


def assert_feature_alignment(
    feature_names: Sequence[str],
    model: Any,
    context: str = "model",
) -> None:
    """
    Research-integrity check: confirm the feature list used to build
    evaluation inputs matches what a loaded model was actually fit on,
    when the model exposes ``feature_names_in_`` (as scikit-learn
    estimators fit on a DataFrame do).
    """

    expected = getattr(model, "feature_names_in_", None)
    if expected is None:
        logger.warning(
            "%s does not expose feature_names_in_; skipping strict "
            "feature-alignment check.",
            context,
        )
        return

    expected_list = list(expected)
    if list(feature_names) != expected_list:
        raise AssertionError(
            f"Feature alignment check failed for {context}: evaluation "
            "features do not match the features the model was fit on.\n"
            f"Expected ({len(expected_list)}): {expected_list}\n"
            f"Got ({len(feature_names)}): {list(feature_names)}"
        )


def build_population(
    df: pd.DataFrame,
    classes: Sequence[str],
    benign_label: str = DEFAULT_BENIGN_LABEL,
    label_col: str = DEFAULT_LABEL_COL,
) -> pd.DataFrame:
    """
    Return the subset of ``df`` whose ``label_col`` value is either
    ``benign_label`` or in ``classes``.

    Used to build the "unseen-attacks + benign" and "single unseen
    class + benign" evaluation populations. Never returns an
    attack-only subset -- precision/FPR/ROC-AUC/PR-AUC require negative
    (benign) examples to be meaningful, so benign rows are always
    included alongside the attack class(es) of interest.
    """

    mask = df[label_col].isin(list(classes) + [benign_label])
    return df.loc[mask].copy()
