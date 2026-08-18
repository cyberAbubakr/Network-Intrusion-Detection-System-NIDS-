
#!/usr/bin/env python
"""
Dataset Cleaning Utilities
==========================

Memory-conscious cleaning utilities for the CIC-IDS2017 preprocessing
pipeline.

Project rule
------------
"Do not silently remove features. Log every preprocessing decision."

Every cleaning function returns a CleaningLog alongside the cleaned data.
The log records what changed, why it changed, and whether rows/features
were retained.

Day 1 scope
-----------
1. Normalize labels.
2. Preserve the original multiclass label.
3. Canonicalize common CIC-IDS2017 label spelling/encoding variants.
4. Keep the primary label column synchronized with the normalized label.
5. Derive a binary benign/attack label.
6. Convert +/-Inf to NaN.
7. Detect physically impossible negative values in explicitly configured
   non-negative features.
8. Never silently drop rows or features.
9. Never perform model-specific imputation.

Important
---------
Raw input files are never modified by these functions.

Invalid physical measurements are converted to NaN rather than replaced
with guessed values. Model-specific preprocessing may later decide how
to handle those NaNs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CIC-IDS2017 label canonicalization
# ---------------------------------------------------------------------------

# Canonical labels used by the project.
#
# CIC-IDS2017 has appeared in multiple CSV/parquet releases with small
# spelling differences, whitespace differences, and encoding corruption.
# These aliases are normalized BEFORE label_binary is derived.
#
# IMPORTANT:
# We do not collapse genuinely different attack classes.
# These mappings only correct known representations of the same class.
LABEL_ALIASES: dict[str, str] = {
    # Benign
    "benign": "Benign",

    # DoS
    "dos hulk": "DoS Hulk",
    "doS hulk": "DoS Hulk",
    "dos goldeneye": "DoS GoldenEye",
    "dos slowloris": "DoS slowloris",
    "dos slowhttptest": "DoS Slowhttptest",
    "dosslowhttptest": "DoS Slowhttptest",

    # Brute force
    "ftp-patator": "FTP-Patator",
    "ftp-patator ": "FTP-Patator",
    "ssh-patator": "SSH-Patator",

    # Web attacks
    "web attack � brute force": "Web Attack � Brute Force",
    "web attack � xss": "Web Attack � XSS",
    "web attack � sql injection": "Web Attack � Sql Injection",

    # Other attacks
    "infiltration": "Infiltration",
    "heartbleed": "Heartbleed",
    "bot": "Bot",
    "ddos": "DDoS",
    "portscan": "PortScan",
}


def _canonicalize_label(value: object) -> str:
    """
    Convert one raw label value into the project's canonical representation.

    The comparison is case-insensitive and whitespace-insensitive.

    Unknown labels are NOT deleted or silently changed. They are returned
    after whitespace normalization so new/unexpected classes remain visible.
    """
    if pd.isna(value):
        return "NaN"

    text = str(value).strip()

    if not text:
        return ""

    lookup_key = text.lower()

    # Direct known alias.
    if lookup_key in LABEL_ALIASES:
        return LABEL_ALIASES[lookup_key]

    # Handle common UTF-8/Windows-1252 mojibake patterns without assuming
    # that every non-ASCII label is corrupt.
    #
    # Example:
    #   "Web Attack â€“ XSS"
    # may appear instead of:
    #   "Web Attack � XSS"
    if "web attack" in lookup_key:
        if "brute" in lookup_key:
            return "Web Attack � Brute Force"
        if "xss" in lookup_key:
            return "Web Attack � XSS"
        if "sql" in lookup_key or "injection" in lookup_key:
            return "Web Attack � Sql Injection"

    # Preserve unknown labels rather than inventing a mapping.
    return text


def _canonicalize_labels(series: pd.Series) -> pd.Series:
    """Vectorized wrapper around _canonicalize_label()."""
    return series.map(_canonicalize_label).astype("string")


# ---------------------------------------------------------------------------
# Cleaning audit log
# ---------------------------------------------------------------------------


@dataclass
class CleaningLog:
    """
    Accumulates a human-readable, ordered log of cleaning decisions.

    The log is intentionally simple so it can be serialized into JSON,
    attached to preprocessing reports, or inspected during experiments.
    """

    entries: list[str] = field(default_factory=list)

    def add(self, message: str) -> None:
        """Add an entry to the cleaning log and Python logger."""
        logger.info(message)
        self.entries.append(message)

    def to_dict(self) -> dict:
        """Return the log in a JSON-serializable structure."""
        return {"entries": list(self.entries)}


# ---------------------------------------------------------------------------
# Label normalization
# ---------------------------------------------------------------------------


def normalize_labels(
    df: pd.DataFrame,
    label_col: str,
    benign_aliases: Optional[set[str]] = None,
    log: Optional[CleaningLog] = None,
) -> tuple[pd.DataFrame, CleaningLog]:
    """
    Normalize labels and derive binary attack/benign targets.

    Operations
    ----------
    1. Validate that label_col exists.
    2. Preserve the original source label in label_multiclass_raw.
    3. Strip leading/trailing whitespace.
    4. Canonicalize known CIC-IDS2017 label variants.
    5. Write the canonical label BACK into label_col.
    6. Store the canonical multiclass label in label_multiclass.
    7. Create label_binary:
           0 = benign
           1 = attack
    8. Log all transformations.

    This is important because downstream metadata normally identifies
    the original label column, e.g. "Label". If we only normalized
    label_multiclass while leaving "Label" untouched, downstream training
    would still receive inconsistent raw labels.

    No rows are removed.

    Parameters
    ----------
    df:
        Input DataFrame.

    label_col:
        Name of the source label column.

    benign_aliases:
        Case-insensitive labels considered benign.
        Defaults to {"benign"}.

    log:
        Existing CleaningLog to append to.

    Returns
    -------
    tuple[pd.DataFrame, CleaningLog]
        Cleaned DataFrame and audit log.

    Raises
    ------
    KeyError
        If label_col does not exist.
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

    # ------------------------------------------------------------------
    # Preserve the exact original labels.
    # ------------------------------------------------------------------

    out["label_multiclass_raw"] = out[label_col]

    raw_values = out[label_col].astype("string")
    stripped_values = raw_values.str.strip()

    whitespace_changed = (
        raw_values.fillna("<NA>") != stripped_values.fillna("<NA>")
    )

    n_whitespace_changed = int(whitespace_changed.sum())

    if n_whitespace_changed:
        log.add(
            f"normalize_labels: stripped leading/trailing whitespace from "
            f"{n_whitespace_changed} label value(s) in '{label_col}'."
        )
    else:
        log.add(
            f"normalize_labels: no leading/trailing whitespace changes "
            f"detected in '{label_col}'."
        )

    # ------------------------------------------------------------------
    # Canonicalize known CIC-IDS2017 label variants.
    # ------------------------------------------------------------------

    canonical_values = _canonicalize_labels(stripped_values)

    changed_mask = (
        stripped_values.fillna("<NA>") != canonical_values.fillna("<NA>")
    )

    n_canonicalized = int(changed_mask.sum())

    if n_canonicalized:
        changed_pairs: dict[str, str] = {}

        for old, new in zip(
            stripped_values[changed_mask],
            canonical_values[changed_mask],
        ):
            old_text = str(old)
            new_text = str(new)

            if old_text != new_text:
                changed_pairs[old_text] = new_text

        log.add(
            f"normalize_labels: canonicalized {n_canonicalized} "
            f"label value(s) in '{label_col}'. "
            f"Mappings applied: {changed_pairs}"
        )
    else:
        log.add(
            f"normalize_labels: no known label aliases required "
            f"canonicalization in '{label_col}'."
        )

    # ------------------------------------------------------------------
    # IMPORTANT:
    # Keep the original label column synchronized with the normalized
    # multiclass label.
    #
    # This prevents metadata such as:
    #     "label_col": "Label"
    #
    # from pointing at an unnormalized column.
    # ------------------------------------------------------------------

    out[label_col] = canonical_values
    out["label_multiclass"] = canonical_values

    log.add(
        f"normalize_labels: synchronized primary label column "
        f"'{label_col}' with canonical multiclass labels."
    )

    # ------------------------------------------------------------------
    # Derive binary label.
    # ------------------------------------------------------------------

    normalized_benign_aliases = {
        str(alias).strip().lower()
        for alias in benign_aliases
    }

    normalized_lower = canonical_values.str.lower()

    is_benign = normalized_lower.isin(normalized_benign_aliases)

    # Treat missing labels as unknown rather than automatically calling
    # them attacks. This prevents missing target values from silently
    # becoming valid attack samples.
    missing_mask = canonical_values.isna() | canonical_values.eq("NaN")

    binary_values = pd.Series(
        np.nan,
        index=out.index,
        dtype="float64",
    )

    binary_values.loc[is_benign & ~missing_mask] = 0
    binary_values.loc[~is_benign & ~missing_mask] = 1

    out["label_binary"] = binary_values.astype("Int64")

    n_missing = int(missing_mask.sum())
    n_benign = int((binary_values == 0).sum())
    n_attack = int((binary_values == 1).sum())

    log.add(
        f"normalize_labels: derived 'label_binary' from '{label_col}' "
        f"using benign_aliases={sorted(normalized_benign_aliases)} -> "
        f"{n_benign} benign row(s), {n_attack} attack row(s), "
        f"{n_missing} missing/unknown label row(s)."
    )

    # ------------------------------------------------------------------
    # Record observed canonical multiclass labels.
    # ------------------------------------------------------------------

    unique_classes = sorted(
        canonical_values.dropna().unique().tolist()
    )

    log.add(
        f"normalize_labels: observed {len(unique_classes)} distinct "
        f"canonical label value(s): {unique_classes}"
    )

    return out, log


# ---------------------------------------------------------------------------
# Non-negative feature validation
# ---------------------------------------------------------------------------


def validate_nonnegative_features(
    df: pd.DataFrame,
    columns: Sequence[str],
    log: Optional[CleaningLog] = None,
) -> tuple[pd.DataFrame, CleaningLog]:
    """
    Validate explicitly configured non-negative numerical features.

    Negative measurements are considered physically invalid and are
    converted to NaN.

    Rows and columns are retained.

    No arbitrary upper bound is imposed.

    No imputation is performed.

    Parameters
    ----------
    df:
        Input DataFrame.

    columns:
        Feature names that are physically constrained to be non-negative.

    log:
        Existing CleaningLog to append to.

    Returns
    -------
    tuple[pd.DataFrame, CleaningLog]
        Cleaned DataFrame and audit log.

    Raises
    ------
    TypeError
        If a configured feature contains non-numeric non-null values.
    """

    if log is None:
        log = CleaningLog()

    out = df.copy()

    for column in columns:
        if column not in out.columns:
            log.add(
                f"validate_nonnegative_features: column '{column}' "
                f"not present; validation skipped."
            )
            continue

        numeric = pd.to_numeric(
            out[column],
            errors="coerce",
        )

        original_non_null = out[column].notna()
        conversion_failed = original_non_null & numeric.isna()

        n_conversion_failed = int(conversion_failed.sum())

        if n_conversion_failed:
            raise TypeError(
                f"Column '{column}' contains "
                f"{n_conversion_failed} non-numeric value(s); "
                f"cannot validate it as a non-negative feature."
            )

        invalid_mask = numeric < 0
        n_invalid = int(invalid_mask.sum())

        if n_invalid == 0:
            log.add(
                f"validate_nonnegative_features: '{column}' passed "
                f"non-negative validation; 0 invalid value(s) found."
            )
            continue

        out.loc[invalid_mask, column] = np.nan

        log.add(
            f"validate_nonnegative_features: found {n_invalid} "
            f"physically impossible negative value(s) in '{column}'; "
            f"replaced those measurements with NaN. Rows retained; "
            f"feature retained; no imputation performed."
        )

    return out, log


# ---------------------------------------------------------------------------
# NaN / Inf handling
# ---------------------------------------------------------------------------


def handle_nan_inf(
    df: pd.DataFrame,
    numeric_only: bool = True,
    inf_strategy: str = "to_nan",
    nan_strategy: str = "flag_only",
    log: Optional[CleaningLog] = None,
) -> tuple[pd.DataFrame, CleaningLog]:
    """
    Handle NaN and Inf values using an explicit, logged strategy.

    No rows or features are silently removed.

    Parameters
    ----------
    df:
        Input DataFrame.

    numeric_only:
        If True, only pandas numeric columns are inspected.

    inf_strategy:
        "to_nan":
            Replace +/-Inf with NaN.

        "keep":
            Leave Inf values untouched.

    nan_strategy:
        "flag_only":
            Existing NaN values are not imputed or dropped.

    Returns
    -------
    tuple[pd.DataFrame, CleaningLog]
        Cleaned DataFrame and audit log.

    Raises
    ------
    ValueError
        If an unsupported strategy is requested.
    """

    if log is None:
        log = CleaningLog()

    out = df.copy()

    if inf_strategy not in {"to_nan", "keep"}:
        raise ValueError(
            f"Unknown inf_strategy: {inf_strategy!r}. "
            f"Expected 'to_nan' or 'keep'."
        )

    if nan_strategy != "flag_only":
        raise ValueError(
            f"Unknown nan_strategy: {nan_strategy!r}. "
            f"Only 'flag_only' is implemented in Day 1."
        )

    if numeric_only:
        target_cols = out.select_dtypes(
            include=[np.number]
        ).columns
    else:
        target_cols = out.columns

    # ------------------------------------------------------------------
    # Replace +/-Inf with NaN.
    # ------------------------------------------------------------------

    if inf_strategy == "to_nan":
        if len(target_cols):
            inf_mask = out[target_cols].isin(
                [np.inf, -np.inf]
            )

            n_inf = int(
                inf_mask.to_numpy().sum()
            )
        else:
            n_inf = 0

        if n_inf:
            out[target_cols] = out[target_cols].replace(
                [np.inf, -np.inf],
                np.nan,
            )

            log.add(
                f"handle_nan_inf: replaced {n_inf} Inf/-Inf "
                f"value(s) with NaN across "
                f"{len(target_cols)} inspected numeric column(s)."
            )
        else:
            log.add(
                "handle_nan_inf: no Inf/-Inf values detected."
            )

    else:
        log.add(
            "handle_nan_inf: inf_strategy='keep' -> "
            "Inf/-Inf values left untouched."
        )

    # ------------------------------------------------------------------
    # Count NaN values.
    # ------------------------------------------------------------------

    if len(target_cols):
        n_nan_total = int(
            out[target_cols].isna().to_numpy().sum()
        )
    else:
        n_nan_total = 0

    if n_nan_total:
        log.add(
            f"handle_nan_inf: {n_nan_total} NaN value(s) present "
            f"after Inf handling; NOT imputed or dropped. "
            f"nan_strategy='flag_only'. Imputation is deferred "
            f"to model-specific preprocessing."
        )
    else:
        log.add(
            "handle_nan_inf: no NaN values present after "
            "Inf handling."
        )

    return out, log

