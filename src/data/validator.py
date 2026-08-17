"""
Dataset and schema validation for CIC-IDS2017 (Day 1 scope).

Everything here operates on a single in-memory chunk (a pandas DataFrame),
so it can be safely used inside the chunked loading loop from loader.py
without ever requiring the full dataset in RAM.

CIC-IDS2017's actual column names vary slightly between the original
UNB release and various re-hosted / re-processed copies (extra spaces,
different casing, "Label" vs "label", etc.). This module never hard-codes
an assumed column list as ground truth; it detects what is actually present
and reports discrepancies rather than silently fixing or dropping anything.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Columns that are commonly expected in CIC-IDS2017-derived files. This is
# used only to WARN about naming drift, never to filter or rename columns
# automatically.
COMMONLY_EXPECTED_LABEL_NAMES = {"label", "Label", " Label", "attack", "Attack"}


@dataclass
class SchemaReport:
    """Result of validating one chunk's schema."""

    n_rows: int
    n_columns: int
    columns: list[str]
    dtypes: dict[str, str]
    detected_label_column: Optional[str]
    duplicate_columns: list[str] = field(default_factory=list)
    unnamed_columns: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "columns": self.columns,
            "dtypes": self.dtypes,
            "detected_label_column": self.detected_label_column,
            "duplicate_columns": self.duplicate_columns,
            "unnamed_columns": self.unnamed_columns,
            "notes": self.notes,
        }


@dataclass
class NanInfReport:
    """Result of validating NaN / Inf content of one chunk."""

    n_rows: int
    nan_counts: dict[str, int]
    inf_counts: dict[str, int]
    columns_all_nan: list[str] = field(default_factory=list)

    @property
    def has_nan(self) -> bool:
        return any(v > 0 for v in self.nan_counts.values())

    @property
    def has_inf(self) -> bool:
        return any(v > 0 for v in self.inf_counts.values())

    def to_dict(self) -> dict:
        return {
            "n_rows": self.n_rows,
            "nan_counts": self.nan_counts,
            "inf_counts": self.inf_counts,
            "columns_all_nan": self.columns_all_nan,
        }


def detect_label_column(columns: Sequence[str]) -> Optional[str]:
    """
    Detect the most likely label/target column by exact and
    case-insensitive matching against common CIC-IDS2017 naming variants.
    Returns None if nothing matches -- callers must handle that explicitly
    rather than assuming a default.
    """
    stripped = {c.strip(): c for c in columns}

    for candidate in COMMONLY_EXPECTED_LABEL_NAMES:
        if candidate in columns:
            return candidate
        if candidate.strip() in stripped:
            return stripped[candidate.strip()]

    lowered = {c.strip().lower(): c for c in columns}
    if "label" in lowered:
        return lowered["label"]

    return None


def validate_schema(df: pd.DataFrame) -> SchemaReport:
    """
    Inspect a chunk's columns/dtypes and flag structural issues:
    duplicate column names, unnamed ("Unnamed: N") columns typically
    produced by stray index columns in CSV exports, and an unresolved
    label column.

    This function does NOT mutate df and does NOT drop anything -- it only
    reports, per the "no silent feature removal" project rule.
    """
    columns = list(df.columns)
    # df.dtypes is positional/aligned with df.columns even when column
    # names repeat, whereas df[c] on a duplicate name returns a DataFrame
    # (not a Series) and has no .dtype attribute -- so we zip against
    # df.dtypes.values instead of indexing by name.
    dtypes = {str(c): str(dt) for c, dt in zip(columns, df.dtypes.values)}

    seen = set()
    duplicates = []
    for c in columns:
        if c in seen:
            duplicates.append(c)
        seen.add(c)

    unnamed = [c for c in columns if str(c).lower().startswith("unnamed:")]

    label_col = detect_label_column(columns)

    notes = []
    if label_col is None:
        notes.append(
            "No label column detected via known naming variants. "
            "Downstream steps that require labels must fail loudly, not guess."
        )
    if duplicates:
        notes.append(f"Duplicate column names found: {duplicates}")
    if unnamed:
        notes.append(
            f"Unnamed columns found (likely stray CSV index columns): {unnamed}"
        )

    report = SchemaReport(
        n_rows=len(df),
        n_columns=len(columns),
        columns=columns,
        dtypes=dtypes,
        detected_label_column=label_col,
        duplicate_columns=duplicates,
        unnamed_columns=unnamed,
        notes=notes,
    )

    for note in notes:
        logger.warning(note)

    return report


def validate_nan_inf(df: pd.DataFrame) -> NanInfReport:
    """
    Count NaN and +/-Inf occurrences per numeric-capable column.

    Inf detection is done only on columns pandas already treats as numeric,
    since np.isinf on object/string columns raises. Non-numeric columns
    report an inf_count of 0 rather than being silently skipped from the
    output (they still appear in the dict with 0).
    """
    nan_counts = df.isna().sum().to_dict()
    nan_counts = {str(k): int(v) for k, v in nan_counts.items()}

    inf_counts: dict[str, int] = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in df.columns:
        if col in numeric_cols:
            col_values = df[col].to_numpy()
            inf_counts[str(col)] = int(np.isinf(col_values).sum())
        else:
            inf_counts[str(col)] = 0

    columns_all_nan = [c for c, v in nan_counts.items() if v == len(df) and len(df) > 0]

    report = NanInfReport(
        n_rows=len(df),
        nan_counts=nan_counts,
        inf_counts=inf_counts,
        columns_all_nan=columns_all_nan,
    )

    if report.has_nan:
        offenders = {k: v for k, v in nan_counts.items() if v > 0}
        logger.warning("NaN values found in columns: %s", offenders)
    if report.has_inf:
        offenders = {k: v for k, v in inf_counts.items() if v > 0}
        logger.warning("Inf/-Inf values found in columns: %s", offenders)
    if columns_all_nan:
        logger.warning("Columns that are entirely NaN in this chunk: %s", columns_all_nan)

    return report


def merge_schema_reports(reports: Sequence[SchemaReport]) -> SchemaReport:
    """
    Combine multiple per-chunk SchemaReports (e.g. across CSV chunks or
    multiple files) into one summary, checking for schema drift between
    chunks (a real risk with CIC-IDS2017's multi-file CSV releases).
    """
    if not reports:
        raise ValueError("Cannot merge an empty sequence of SchemaReports.")

    first = reports[0]
    notes = list(first.notes)
    all_columns_sets = {tuple(r.columns) for r in reports}
    if len(all_columns_sets) > 1:
        notes.append(
            f"Schema drift detected across chunks/files: "
            f"{len(all_columns_sets)} distinct column sets observed."
        )
        logger.warning(notes[-1])

    total_rows = sum(r.n_rows for r in reports)

    return SchemaReport(
        n_rows=total_rows,
        n_columns=first.n_columns,
        columns=first.columns,
        dtypes=first.dtypes,
        detected_label_column=first.detected_label_column,
        duplicate_columns=first.duplicate_columns,
        unnamed_columns=first.unnamed_columns,
        notes=notes,
    )
