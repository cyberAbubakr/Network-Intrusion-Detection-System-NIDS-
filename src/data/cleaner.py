"""
Cleaning utilities: label normalization and NaN/Inf handling.

Project rule: "Do not silently remove features. Log every preprocessing
decision." Every function here returns a CleaningLog alongside the cleaned
data, recording exactly what was changed and why, so the decision trail is
auditable and reproducible.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CleaningLog:
    """Accumulates a human-readable, ordered log of cleaning decisions."""

    entries: list[str] = field(default_factory=list)

    def add(self, message: str) -> None:
        logger.info(message)
        self.entries.append(message)

    def to_dict(self) -> dict:
        return {"entries": list(self.entries)}


def normalize_labels(
    df: pd.DataFrame,
    label_col: str,
    benign_aliases: Optional[set[str]] = None,
    log: Optional[CleaningLog] = None,
) -> tuple[pd.DataFrame, CleaningLog]:
    """
    Normalize the label column in place (on a copy):
        1. Strip leading/trailing whitespace from label strings
           (CIC-IDS2017 CSVs are notorious for " BENIGN" vs "BENIGN").
        2. Preserve the ORIGINAL multiclass label in a new column
           `label_multiclass_raw` (nothing is discarded).
        3. Add a `label_binary` column: 0 for benign traffic, 1 for any
           attack, based on case-insensitive matching against a
           configurable benign-alias set (default: {"benign"}).

    This function does not decide the modeling target for the caller -- it
    exposes both the raw multiclass label and a derived binary label, and
    logs exactly how the binary label was derived.

    Raises
    ------
    KeyError
        If label_col is not present in df.
    """
    if label_col not in df.columns:
        raise KeyError(
            f"Label column '{label_col}' not found in DataFrame columns: "
            f"{list(df.columns)}"
        )

    if log is None:
        log = CleaningLog()

    if benign_aliases is None:
        benign_aliases = {"benign"}

    out = df.copy()

    raw_values = out[label_col].astype(str)
    stripped_values = raw_values.str.strip()

    n_changed = int((raw_values != stripped_values).sum())
    if n_changed > 0:
        log.add(
            f"normalize_labels: stripped whitespace from {n_changed} label "
            f"value(s) in column '{label_col}'."
        )

    out["label_multiclass_raw"] = out[label_col]
    out["label_multiclass"] = stripped_values

    normalized_lower = stripped_values.str.lower()
    is_benign = normalized_lower.isin({a.lower() for a in benign_aliases})
    out["label_binary"] = (~is_benign).astype(int)

    n_benign = int(is_benign.sum())
    n_attack = int((~is_benign).sum())
    log.add(
        f"normalize_labels: derived 'label_binary' from '{label_col}' using "
        f"benign_aliases={sorted(benign_aliases)} -> "
        f"{n_benign} benign row(s), {n_attack} attack row(s)."
    )

    unique_classes = sorted(stripped_values.unique().tolist())
    log.add(
        f"normalize_labels: {len(unique_classes)} distinct raw label value(s) "
        f"observed in this chunk: {unique_classes}"
    )

    return out, log


def handle_nan_inf(
    df: pd.DataFrame,
    numeric_only: bool = True,
    inf_strategy: str = "to_nan",
    nan_strategy: str = "flag_only",
    log: Optional[CleaningLog] = None,
) -> tuple[pd.DataFrame, CleaningLog]:
    """
    Handle NaN/Inf values with an explicit, logged strategy. Nothing is
    dropped or imputed silently.

    Parameters
    ----------
    numeric_only:
        Only touch columns pandas already treats as numeric.
    inf_strategy:
        "to_nan" (default): replace +/-Inf with NaN so downstream steps
        have one consistent "missing" representation. "keep": leave Inf
        values untouched (not recommended, but explicit).
    nan_strategy:
        "flag_only" (default): do not modify NaN values at all; only
        record counts. Actual imputation/row-dropping is deliberately
        NOT performed in Day 1 -- that decision belongs to a modeling-
        specific step so it can be tuned/justified per-model, not baked
        into a generic cleaner.

    Returns
    -------
    (cleaned_df, log)
    """
    if log is None:
        log = CleaningLog()

    out = df.copy()

    numeric_cols = out.select_dtypes(include=[np.number]).columns
    target_cols = numeric_cols if numeric_only else out.columns

    if inf_strategy == "to_nan":
        inf_mask = out[target_cols].isin([np.inf, -np.inf])
        n_inf = int(inf_mask.to_numpy().sum())
        if n_inf > 0:
            out[target_cols] = out[target_cols].replace([np.inf, -np.inf], np.nan)
            log.add(
                f"handle_nan_inf: replaced {n_inf} Inf/-Inf value(s) with NaN "
                f"across {len(target_cols)} numeric column(s)."
            )
    elif inf_strategy == "keep":
        log.add("handle_nan_inf: inf_strategy='keep' -> Inf values left untouched.")
    else:
        raise ValueError(f"Unknown inf_strategy: {inf_strategy!r}")

    n_nan_total = int(out[target_cols].isna().to_numpy().sum())
    if nan_strategy == "flag_only":
        if n_nan_total > 0:
            log.add(
                f"handle_nan_inf: {n_nan_total} NaN value(s) present after Inf "
                f"handling; NOT imputed or dropped (nan_strategy='flag_only'). "
                f"Imputation is deferred to model-specific pipeline steps."
            )
    else:
        raise ValueError(
            f"Unknown nan_strategy: {nan_strategy!r} (only 'flag_only' is "
            "implemented in Day 1 scope)."
        )

    return out, log
