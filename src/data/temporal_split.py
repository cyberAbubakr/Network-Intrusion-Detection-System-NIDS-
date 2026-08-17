"""
Timestamp detection, chronological train/test splitting, and temporal
leakage verification.

This is the core of the "Cross-Temporal" part of the project: models must
be evaluated on traffic that occurs strictly AFTER the training window,
never on a random shuffle. Every split produced here is verified before
being trusted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

# Column-name substrings that commonly indicate a timestamp field in
# CIC-IDS2017-derived datasets. Case-insensitive substring match.
TIMESTAMP_NAME_HINTS = ("timestamp", "time stamp", "flow start", "start time")


@dataclass
class TimestampDetectionResult:
    column: Optional[str]
    parsed_successfully: bool
    n_unparseable: int
    notes: list[str] = field(default_factory=list)


def detect_timestamp_column(
    df: pd.DataFrame, explicit_column: Optional[str] = None
) -> TimestampDetectionResult:
    """
    Detect the most likely timestamp column.

    If `explicit_column` is given, it is validated and used directly
    (still parsed/checked, never blindly trusted). Otherwise columns are
    scanned by name for known hints, in the order they appear.

    This function does not mutate df. It only inspects.
    """
    notes: list[str] = []

    candidates: list[str]
    if explicit_column is not None:
        if explicit_column not in df.columns:
            raise KeyError(
                f"explicit_column={explicit_column!r} not found in DataFrame "
                f"columns: {list(df.columns)}"
            )
        candidates = [explicit_column]
    else:
        candidates = [
            c
            for c in df.columns
            if any(hint in str(c).strip().lower() for hint in TIMESTAMP_NAME_HINTS)
        ]

    if not candidates:
        notes.append(
            "No timestamp-like column detected by name. Chronological "
            "splitting cannot proceed without an explicit column."
        )
        logger.warning(notes[-1])
        return TimestampDetectionResult(
            column=None, parsed_successfully=False, n_unparseable=0, notes=notes
        )

    column = candidates[0]
    if len(candidates) > 1:
        notes.append(
            f"Multiple timestamp-like columns found: {candidates}. "
            f"Using the first: {column!r}. Pass explicit_column= to override."
        )
        logger.warning(notes[-1])

    parsed = pd.to_datetime(df[column], errors="coerce")
    n_unparseable = int(parsed.isna().sum() - df[column].isna().sum())
    n_unparseable = max(n_unparseable, 0)

    if n_unparseable > 0:
        notes.append(
            f"{n_unparseable} value(s) in column {column!r} could not be "
            "parsed as timestamps and were coerced to NaT for detection "
            "purposes only (source data left untouched)."
        )
        logger.warning(notes[-1])

    parsed_successfully = n_unparseable < len(df)

    return TimestampDetectionResult(
        column=column,
        parsed_successfully=parsed_successfully,
        n_unparseable=n_unparseable,
        notes=notes,
    )


@dataclass
class TemporalSplitResult:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    timestamp_column: str
    split_timestamp: pd.Timestamp
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def summary(self) -> dict:
        return {
            "timestamp_column": self.timestamp_column,
            "split_timestamp": str(self.split_timestamp),
            "train_rows": len(self.train_df),
            "test_rows": len(self.test_df),
            "train_start": str(self.train_start),
            "train_end": str(self.train_end),
            "test_start": str(self.test_start),
            "test_end": str(self.test_end),
        }


def chronological_train_test_split(
    df: pd.DataFrame,
    timestamp_column: str,
    test_size: float = 0.2,
) -> TemporalSplitResult:
    """
    Split a DataFrame into train/test sets STRICTLY by time: every row in
    train occurs before every row in test. No shuffling, no stratified
    sampling, no random_state -- ordering is the only thing that matters
    here.

    Parameters
    ----------
    df:
        Must contain `timestamp_column`, parseable as datetimes.
    timestamp_column:
        Name of the column to sort and split on.
    test_size:
        Fraction (0, 1) of rows to place in the test set (the
        chronologically LATEST test_size fraction of rows).

    Raises
    ------
    KeyError
        If timestamp_column is missing.
    ValueError
        If test_size is not in (0, 1), df is empty, or timestamps cannot
        be parsed for every row.
    """
    if timestamp_column not in df.columns:
        raise KeyError(f"timestamp_column={timestamp_column!r} not in DataFrame.")
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")
    if len(df) == 0:
        raise ValueError("Cannot split an empty DataFrame.")

    working = df.copy()
    working["_parsed_ts"] = pd.to_datetime(working[timestamp_column], errors="coerce")

    n_bad = int(working["_parsed_ts"].isna().sum())
    if n_bad > 0:
        raise ValueError(
            f"{n_bad} row(s) have unparseable timestamps in "
            f"{timestamp_column!r}. Resolve/clean before splitting -- "
            "chronological_train_test_split refuses to silently drop rows."
        )

    working = working.sort_values("_parsed_ts", kind="mergesort").reset_index(drop=True)

    split_idx = int(round(len(working) * (1 - test_size)))
    split_idx = min(max(split_idx, 1), len(working) - 1)

    train = working.iloc[:split_idx].drop(columns=["_parsed_ts"])
    test = working.iloc[split_idx:].drop(columns=["_parsed_ts"])

    train_ts = pd.to_datetime(train[timestamp_column], errors="coerce")
    test_ts = pd.to_datetime(test[timestamp_column], errors="coerce")
    split_timestamp = test_ts.min()

    result = TemporalSplitResult(
        train_df=train,
        test_df=test,
        timestamp_column=timestamp_column,
        split_timestamp=split_timestamp,
        train_start=train_ts.min(),
        train_end=train_ts.max(),
        test_start=test_ts.min(),
        test_end=test_ts.max(),
    )

    logger.info(
        "Chronological split: train=%d rows [%s -> %s], test=%d rows [%s -> %s]",
        len(train),
        result.train_start,
        result.train_end,
        len(test),
        result.test_start,
        result.test_end,
    )

    return result


@dataclass
class TemporalLeakageCheck:
    passed: bool
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    n_overlapping_rows: int
    message: str


def verify_no_temporal_leakage(split: TemporalSplitResult) -> TemporalLeakageCheck:
    """
    Explicitly verify that the training set ends strictly before (or
    exactly at, for boundary ties) the test set begins, and that no
    individual timestamp value appears on both sides.

    This is a hard gate: callers should treat `passed=False` as a fatal
    error, not a warning, since it means the "cross-temporal" premise of
    the whole project has been violated.
    """
    train_end = split.train_end
    test_start = split.test_start

    train_ts = pd.to_datetime(split.train_df[split.timestamp_column], errors="coerce")
    test_ts = pd.to_datetime(split.test_df[split.timestamp_column], errors="coerce")

    overlapping_values = set(train_ts.unique()).intersection(set(test_ts.unique()))
    n_overlapping_rows = int(
        train_ts.isin(overlapping_values).sum() + test_ts.isin(overlapping_values).sum()
    )

    order_ok = train_end <= test_start

    passed = order_ok and n_overlapping_rows == 0

    if passed:
        message = (
            f"Temporal leakage check PASSED: train ends {train_end}, "
            f"test starts {test_start}, no overlapping timestamp values."
        )
        logger.info(message)
    else:
        message = (
            f"Temporal leakage check FAILED: train_end={train_end}, "
            f"test_start={test_start}, order_ok={order_ok}, "
            f"n_overlapping_rows={n_overlapping_rows} "
            f"({len(overlapping_values)} distinct overlapping timestamp "
            "value(s)). Do not proceed to training with this split."
        )
        logger.error(message)

    return TemporalLeakageCheck(
        passed=passed,
        train_end=train_end,
        test_start=test_start,
        n_overlapping_rows=n_overlapping_rows,
        message=message,
    )
