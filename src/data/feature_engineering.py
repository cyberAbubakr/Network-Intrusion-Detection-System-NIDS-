
"""
Leakage-prone feature identification and lightweight feature selection.

Two distinct concerns are intentionally kept separate:

1. Leakage identification (`identify_leakage_prone_features`)
   ------------------------------------------------------------
   Flags columns that are likely to cause leakage or trivial memorization,
   such as Flow IDs, raw IP addresses, ports, duplicate label-like columns,
   and near-constant columns.

   This function is advisory only. It NEVER mutates the DataFrame.

2. Lightweight feature selection (`select_features`)
   ---------------------------------------------------
   Applies cheap, deterministic feature-selection rules suitable for the
   Day 1 CPU/RAM constraints.

   It may remove:
       - exact duplicate feature columns
       - near-constant feature columns

   Explicitly excluded columns (labels, identifiers, capture-day metadata,
   etc.) are retained in the DataFrame but are excluded from the returned
   model-feature list.

Project rule:
    "Do not silently remove features. Log every preprocessing decision."

Every actual feature-removal decision is recorded through CleaningLog.

Important:
    Physical-value validation belongs in src.data.cleaner.
    Label normalization belongs in src.data.cleaner.
    NaN/Inf handling belongs in src.data.cleaner.

This module should therefore remain focused on leakage detection and
lightweight feature selection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence

import pandas as pd

from src.data.cleaner import CleaningLog


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Column-name rules
# ---------------------------------------------------------------------------

# These names commonly identify fields that can allow a model to memorize
# flows or hosts rather than learn network behavior.
#
# Matching is case-insensitive and substring-based because CIC-IDS2017
# derivatives sometimes use slightly different column naming conventions.
LEAKAGE_PRONE_NAME_HINTS = (
    "flow id",
    "flow_id",
    "src ip",
    "source ip",
    "src_ip",
    "dst ip",
    "destination ip",
    "dst_ip",
    "src port",
    "source port",
    "src_port",
    "dst port",
    "destination port",
    "dst_port",
    "unnamed:",
)


# A second label/target column alongside the primary label is dangerous
# because it can directly expose the target to the model.
LABEL_LIKE_NAME_HINTS = (
    "label",
    "attack",
    "class",
    "category",
)


# ---------------------------------------------------------------------------
# Leakage report
# ---------------------------------------------------------------------------

@dataclass
class LeakageReport:
    """
    Rule-based leakage-risk assessment.

    This object is advisory. Creating a LeakageReport never changes the
    input DataFrame.
    """

    identifier_like_columns: list[str] = field(default_factory=list)
    label_like_columns: list[str] = field(default_factory=list)
    near_constant_columns: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def flagged_columns(self) -> list[str]:
        """
        Return all flagged columns once, preserving discovery order.
        """
        seen: list[str] = []

        for column in (
            self.identifier_like_columns
            + self.label_like_columns
            + self.near_constant_columns
        ):
            if column not in seen:
                seen.append(column)

        return seen

    def to_dict(self) -> dict:
        """
        Return a JSON-serializable representation.
        """
        return {
            "identifier_like_columns": list(
                self.identifier_like_columns
            ),
            "label_like_columns": list(
                self.label_like_columns
            ),
            "near_constant_columns": list(
                self.near_constant_columns
            ),
            "flagged_columns": self.flagged_columns,
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# Leakage identification
# ---------------------------------------------------------------------------

def identify_leakage_prone_features(
    df: pd.DataFrame,
    primary_label_col: Optional[str] = None,
    near_constant_threshold: float = 0.999,
) -> LeakageReport:
    """
    Identify columns that may cause leakage or trivial memorization.

    The following checks are performed:

    1. Identifier-like names:
       Flow ID, IP addresses, ports, and stray index columns.

    2. Label-like names:
       Columns containing label/attack/class/category terminology other
       than the primary detected label.

    3. Near-constant columns:
       Columns where at least `near_constant_threshold` of the rows share
       one value.

    This function NEVER drops columns and NEVER modifies `df`.

    Parameters
    ----------
    df:
        Input DataFrame.

    primary_label_col:
        Name of the actual target/label column. This column is not flagged
        merely because its name contains "label".

    near_constant_threshold:
        Fraction of rows sharing the most common value required to flag a
        column. Default is 0.999, i.e. 99.9%.

    Returns
    -------
    LeakageReport
        Advisory report describing potentially problematic columns.

    Raises
    ------
    ValueError
        If `near_constant_threshold` is outside [0, 1].
    """
    if not 0.0 <= near_constant_threshold <= 1.0:
        raise ValueError(
            "near_constant_threshold must be between 0 and 1; "
            f"got {near_constant_threshold!r}"
        )

    identifier_like: list[str] = []
    label_like: list[str] = []
    near_constant: list[str] = []
    notes: list[str] = []

    # ------------------------------------------------------------------
    # Name-based leakage checks.
    # ------------------------------------------------------------------

    for column in df.columns:
        column_lower = str(column).strip().lower()

        if any(
            hint in column_lower
            for hint in LEAKAGE_PRONE_NAME_HINTS
        ):
            identifier_like.append(column)
            continue

        if column != primary_label_col and any(
            hint == column_lower or hint in column_lower
            for hint in LABEL_LIKE_NAME_HINTS
        ):
            label_like.append(column)

    # ------------------------------------------------------------------
    # Near-constant check.
    #
    # This is intentionally performed per chunk. The Day 1 pipeline is
    # memory-conscious and therefore does not build a full-dataset
    # correlation matrix or other expensive structure here.
    # ------------------------------------------------------------------

    if len(df) > 0:
        for column in df.columns:
            if column in identifier_like:
                continue

            if column in label_like:
                continue

            try:
                frequencies = df[column].value_counts(
                    normalize=True,
                    dropna=False,
                )
            except (TypeError, ValueError):
                # If a column cannot be evaluated safely, leave it alone.
                # The goal here is detection, not destructive coercion.
                logger.debug(
                    "Could not evaluate near-constant status for %r.",
                    column,
                    exc_info=True,
                )
                continue

            if frequencies.empty:
                continue

            top_frequency = float(frequencies.iloc[0])

            if top_frequency >= near_constant_threshold:
                near_constant.append(column)

    # ------------------------------------------------------------------
    # Human-readable notes.
    # ------------------------------------------------------------------

    if identifier_like:
        notes.append(
            f"{len(identifier_like)} column(s) flagged as "
            f"identifier-like: {identifier_like}"
        )

    if label_like:
        notes.append(
            f"{len(label_like)} column(s) flagged as label-like but "
            f"distinct from the primary label column "
            f"{primary_label_col!r}: {label_like}"
        )

    if near_constant:
        notes.append(
            f"{len(near_constant)} column(s) flagged as near-constant "
            f"(threshold={near_constant_threshold:.1%}): "
            f"{near_constant}"
        )

    for note in notes:
        logger.warning("identify_leakage_prone_features: %s", note)

    return LeakageReport(
        identifier_like_columns=identifier_like,
        label_like_columns=label_like,
        near_constant_columns=near_constant,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Lightweight feature selection
# ---------------------------------------------------------------------------

def select_features(
    df: pd.DataFrame,
    exclude_columns: Sequence[str],
    drop_near_constant: bool = True,
    near_constant_threshold: float = 0.999,
    drop_exact_duplicate_columns: bool = True,
    log: Optional[CleaningLog] = None,
) -> tuple[pd.DataFrame, list[str], CleaningLog]:
    """
    Perform lightweight, deterministic feature selection.

    The function is designed for the Day 1 CPU/RAM constraints.

    Rules:

        1. `exclude_columns` are excluded from the model-feature list.
           They remain in the returned DataFrame.

        2. Exact duplicate feature columns may be removed.

        3. Near-constant feature columns may be removed.

    Identifier and label columns therefore survive in the DataFrame when
    explicitly excluded, allowing the rest of the preprocessing pipeline
    to retain labels and metadata.

    Every actual removal is recorded in CleaningLog.

    Parameters
    ----------
    df:
        Input DataFrame.

    exclude_columns:
        Columns that must not be included in model features.

    drop_near_constant:
        Whether to remove near-constant feature columns.

    near_constant_threshold:
        Threshold used for near-constant detection.

    drop_exact_duplicate_columns:
        Whether to remove exact duplicate feature columns.

    log:
        Existing CleaningLog. A new one is created when omitted.

    Returns
    -------
    tuple[pd.DataFrame, list[str], CleaningLog]
        Modified DataFrame, selected feature names, and audit log.
    """
    if log is None:
        log = CleaningLog()

    if not 0.0 <= near_constant_threshold <= 1.0:
        raise ValueError(
            "near_constant_threshold must be between 0 and 1; "
            f"got {near_constant_threshold!r}"
        )

    out = df.copy()

    # Preserve ordering while removing duplicates from the exclusion list.
    exclude_set = set(exclude_columns)

    # Only columns that can actually become model features are candidates
    # for removal.
    candidate_columns = [
        column
        for column in out.columns
        if column not in exclude_set
    ]

    dropped_columns: list[str] = []

    # ------------------------------------------------------------------
    # Exact duplicate columns
    # ------------------------------------------------------------------

    if drop_exact_duplicate_columns:
        seen_signatures: dict[tuple, str] = {}
        duplicate_columns: list[str] = []

        for column in candidate_columns:
            try:
                # Hash the complete column. This is more reliable than
                # comparing only the first 1000 values because duplicate
                # values later in a chunk must not be missed.
                signature_values = pd.util.hash_pandas_object(
                    out[column],
                    index=False,
                )

                signature = (
                    len(signature_values),
                    int(signature_values.sum()),
                    int(signature_values.iloc[0])
                    if len(signature_values)
                    else 0,
                    int(signature_values.iloc[-1])
                    if len(signature_values)
                    else 0,
                )

            except (TypeError, ValueError):
                logger.debug(
                    "Could not hash column %r for duplicate detection.",
                    column,
                    exc_info=True,
                )
                continue

            if signature in seen_signatures:
                original_column = seen_signatures[signature]

                if out[column].equals(out[original_column]):
                    duplicate_columns.append(column)
                    continue

            seen_signatures[signature] = column

        if duplicate_columns:
            log.add(
                "select_features: dropping "
                f"{len(duplicate_columns)} exact-duplicate "
                f"feature column(s): {duplicate_columns}"
            )

            out = out.drop(columns=duplicate_columns)
            dropped_columns.extend(duplicate_columns)

    # ------------------------------------------------------------------
    # Near-constant columns
    # ------------------------------------------------------------------

    if drop_near_constant:
        leakage_report = identify_leakage_prone_features(
            out,
            primary_label_col=None,
            near_constant_threshold=near_constant_threshold,
        )

        near_constant_columns = [
            column
            for column in leakage_report.near_constant_columns
            if column not in exclude_set
            and column not in dropped_columns
        ]

        if near_constant_columns:
            log.add(
                "select_features: dropping "
                f"{len(near_constant_columns)} near-constant "
                f"feature column(s) "
                f"(threshold={near_constant_threshold:.1%}): "
                f"{near_constant_columns}"
            )

            out = out.drop(columns=near_constant_columns)
            dropped_columns.extend(near_constant_columns)

    # ------------------------------------------------------------------
    # Final model-feature list
    # ------------------------------------------------------------------

    selected_feature_names = [
        column
        for column in out.columns
        if column not in exclude_set
        and column not in dropped_columns
    ]

    excluded_present = [
        column
        for column in exclude_columns
        if column in out.columns
    ]

    missing_excluded = [
        column
        for column in exclude_columns
        if column not in out.columns
    ]

    if missing_excluded:
        log.add(
            "select_features: requested exclusion column(s) not present "
            f"in DataFrame: {missing_excluded}"
        )

    log.add(
        "select_features: final feature count = "
        f"{len(selected_feature_names)}; "
        f"original columns = {len(df.columns)}; "
        f"dropped columns = {len(dropped_columns)}; "
        f"excluded model-only columns present = "
        f"{len(excluded_present)}."
    )

    if dropped_columns:
        log.add(
            "select_features: final dropped feature column(s): "
            f"{dropped_columns}"
        )
    else:
        log.add(
            "select_features: no feature columns were physically "
            "dropped."
        )

    return out, selected_feature_names, log

