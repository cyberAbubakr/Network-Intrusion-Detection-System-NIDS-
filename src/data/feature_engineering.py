"""
Leakage-prone feature identification and lightweight feature selection.

Two distinct concerns, kept separate:

1. Leakage identification (`identify_leakage_prone_features`) -- flags
   columns that are identifiers, near-constant, or otherwise likely to let
   a model "cheat" (e.g. Flow ID, raw IPs/ports, or the label itself
   duplicated under another name). This is a STATIC, rule-based check.
   It only recommends; it never drops columns for you.

2. Lightweight feature selection (`select_features`) -- a cheap,
   CPU/RAM-friendly filter (constant/near-constant columns, exact
   duplicate columns) suitable for an 8GB-RAM CPU-only machine. No
   iterative wrapper methods, no SHAP, no heavy correlation matrices on
   the full dataset.

Every decision is logged via cleaner.CleaningLog so nothing is silently
removed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.data.cleaner import CleaningLog

logger = logging.getLogger(__name__)

# Column-name substrings that commonly indicate identifier / leakage-prone
# fields in CIC-IDS2017-derived datasets. Matching is case-insensitive and
# substring-based since exact column names vary by release.
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

# Label-like columns other than the primary detected label; if present
# alongside the primary label they are almost certainly leakage.
LABEL_LIKE_NAME_HINTS = ("label", "attack", "class", "category")


@dataclass
class LeakageReport:
    """Rule-based leakage-risk assessment. Advisory only -- never mutates data."""

    identifier_like_columns: list[str] = field(default_factory=list)
    label_like_columns: list[str] = field(default_factory=list)
    near_constant_columns: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def flagged_columns(self) -> list[str]:
        seen: list[str] = []
        for col in (
            self.identifier_like_columns
            + self.label_like_columns
            + self.near_constant_columns
        ):
            if col not in seen:
                seen.append(col)
        return seen

    def to_dict(self) -> dict:
        return {
            "identifier_like_columns": self.identifier_like_columns,
            "label_like_columns": self.label_like_columns,
            "near_constant_columns": self.near_constant_columns,
            "flagged_columns": self.flagged_columns,
            "notes": self.notes,
        }


def identify_leakage_prone_features(
    df: pd.DataFrame,
    primary_label_col: Optional[str] = None,
    near_constant_threshold: float = 0.999,
) -> LeakageReport:
    """
    Rule-based (cheap) scan for columns likely to cause data leakage or
    provide the model with a "free" answer.

    Checks performed (all O(n) or O(n log n), safe for 8GB RAM):
        * Column name matches known identifier hints (Flow ID, raw IPs,
          ports, stray "Unnamed:" index columns).
        * Column name looks label-like but is NOT the primary detected
          label column (duplicate/near-duplicate label columns).
        * Column is near-constant (>= near_constant_threshold fraction of
          rows share the single most common value) -- not leakage per se,
          but flagged here since it is also a "free"/uninformative feature
          risk worth surfacing at the same review step.

    This function NEVER drops or renames columns. It returns a report for
    a human (or select_features, explicitly) to act on.
    """
    identifier_like: list[str] = []
    label_like: list[str] = []
    near_constant: list[str] = []
    notes: list[str] = []

    for col in df.columns:
        col_lower = str(col).strip().lower()

        if any(hint in col_lower for hint in LEAKAGE_PRONE_NAME_HINTS):
            identifier_like.append(col)
            continue

        if col != primary_label_col and any(
            hint == col_lower or hint in col_lower for hint in LABEL_LIKE_NAME_HINTS
        ):
            label_like.append(col)

    for col in df.columns:
        if col in identifier_like or col in label_like:
            continue
        if len(df) == 0:
            continue
        try:
            top_freq = df[col].value_counts(normalize=True, dropna=False).iloc[0]
        except IndexError:
            continue
        if top_freq >= near_constant_threshold:
            near_constant.append(col)

    if identifier_like:
        notes.append(
            f"{len(identifier_like)} column(s) flagged as identifier-like "
            f"(risk of leakage / trivial memorization): {identifier_like}"
        )
    if label_like:
        notes.append(
            f"{len(label_like)} column(s) flagged as label-like but distinct "
            f"from the primary label column "
            f"({primary_label_col!r}): {label_like}. These would leak the "
            "target if included as features."
        )
    if near_constant:
        notes.append(
            f"{len(near_constant)} column(s) are near-constant "
            f"(>= {near_constant_threshold:.1%} single value): {near_constant}"
        )

    for note in notes:
        logger.warning(note)

    return LeakageReport(
        identifier_like_columns=identifier_like,
        label_like_columns=label_like,
        near_constant_columns=near_constant,
        notes=notes,
    )


def select_features(
    df: pd.DataFrame,
    exclude_columns: Sequence[str],
    drop_near_constant: bool = True,
    near_constant_threshold: float = 0.999,
    drop_exact_duplicate_columns: bool = True,
    log: Optional[CleaningLog] = None,
) -> tuple[pd.DataFrame, list[str], CleaningLog]:
    """
    Lightweight, CPU/RAM-cheap feature selection.

    Deliberately excludes anything requiring the full dataset in memory
    (e.g. full pairwise correlation matrices, mutual information over all
    rows, recursive feature elimination). Suitable for an 8GB RAM CPU-only
    machine and for per-chunk or per-sample use.

    Steps (each optional, each logged):
        1. Always exclude `exclude_columns` (labels, identifiers, etc.)
           from the *feature* set returned -- but they are NOT dropped
           from the returned DataFrame, only from `selected_feature_names`.
        2. Optionally flag exact-duplicate columns (identical values across
           all rows) and drop all but the first occurrence.
        3. Optionally flag near-constant columns (see
           identify_leakage_prone_features) and drop them.

    Returns
    -------
    (df_with_decisions_applied, selected_feature_names, log)
        df_with_decisions_applied: copy of df with only the DROPPED
            columns removed (duplicates / near-constant, if enabled).
            exclude_columns are kept in the DataFrame (e.g. so the label
            survives) but excluded from selected_feature_names.
        selected_feature_names: final list of feature columns to use for
            modeling (excludes exclude_columns and any dropped columns).
        log: CleaningLog recording every drop decision.
    """
    if log is None:
        log = CleaningLog()

    out = df.copy()
    exclude_set = set(exclude_columns)

    dropped_columns: list[str] = []

    if drop_exact_duplicate_columns:
        seen_signatures: dict[tuple, str] = {}
        duplicate_cols: list[str] = []
        candidate_cols = [c for c in out.columns if c not in exclude_set]
        for col in candidate_cols:
            # Hash a sample-based signature for cheap duplicate detection;
            # fall back to full comparison only on signature collision.
            try:
                signature = tuple(pd.util.hash_pandas_object(out[col], index=False).values[:1000])
            except TypeError:
                continue
            if signature in seen_signatures:
                original = seen_signatures[signature]
                if out[col].equals(out[original]):
                    duplicate_cols.append(col)
                    continue
            seen_signatures[signature] = col

        if duplicate_cols:
            log.add(
                f"select_features: dropping {len(duplicate_cols)} exact-duplicate "
                f"column(s): {duplicate_cols}"
            )
            out = out.drop(columns=duplicate_cols)
            dropped_columns.extend(duplicate_cols)

    if drop_near_constant:
        leakage_report = identify_leakage_prone_features(
            out, near_constant_threshold=near_constant_threshold
        )
        near_constant = [
            c for c in leakage_report.near_constant_columns if c not in exclude_set
        ]
        if near_constant:
            log.add(
                f"select_features: dropping {len(near_constant)} near-constant "
                f"column(s) (threshold={near_constant_threshold:.1%}): "
                f"{near_constant}"
            )
            out = out.drop(columns=near_constant)
            dropped_columns.extend(near_constant)

    selected_feature_names = [
        c for c in out.columns if c not in exclude_set and c not in dropped_columns
    ]

    log.add(
        f"select_features: final feature count = {len(selected_feature_names)} "
        f"(from {len(df.columns)} original columns, "
        f"{len(dropped_columns)} dropped, {len(exclude_set)} excluded as "
        "label/identifier)."
    )

    return out, selected_feature_names, log
