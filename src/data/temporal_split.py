"""
Temporal splitting and leakage verification for CIC-IDS2017.

This module provides two temporal splitting strategies:

1. Row-level chronological splitting
   ----------------------------------
   Used when the dataset contains a genuine timestamp column.

   The data is sorted chronologically and the latest portion is held out
   for testing. No random shuffling is performed.

2. Capture-day splitting
   ----------------------
   Used as a fallback when the dataset does not contain a usable
   row-level timestamp column.

   CIC-IDS2017 files commonly identify their capture day in the filename
   (Monday, Tuesday, Wednesday, Thursday, Friday). Files are therefore
   ordered by capture day and complete days can be held out for testing.

Both strategies include explicit leakage verification. A failed leakage
check should be treated as a hard failure by downstream training code.

Important:
    This module does not download, modify, or silently discard dataset
    records. It only performs inspection, labeling, splitting, and
    verification on data supplied by the caller.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# Timestamp detection
# ============================================================================

# Common timestamp-related column-name fragments found in CIC-IDS2017
# distributions and reprocessed versions.
#
# Matching is case-insensitive and substring-based because different
# distributions may use names such as:
#
#   Timestamp
#   Flow Start Time
#   StartTime
#   flow start
#
TIMESTAMP_NAME_HINTS = (
    "timestamp",
    "time stamp",
    "flow start",
    "starttime",
    "start time",
)


@dataclass
class TimestampDetectionResult:
    """
    Result of timestamp-column detection.

    Attributes
    ----------
    column:
        Name of the detected timestamp column, or None if none was found.

    parsed_successfully:
        True when at least some non-null timestamp values were successfully
        parsed.

    n_unparseable:
        Number of non-null values that could not be parsed as timestamps.

    notes:
        Human-readable diagnostic messages.
    """

    column: Optional[str]
    parsed_successfully: bool
    n_unparseable: int
    notes: list[str] = field(default_factory=list)


def detect_timestamp_column(
    df: pd.DataFrame,
    explicit_column: Optional[str] = None,
) -> TimestampDetectionResult:
    """
    Detect and inspect a likely timestamp column.

    If ``explicit_column`` is provided, that exact column is used after
    validating that it exists.

    Otherwise, columns are searched using ``TIMESTAMP_NAME_HINTS``.

    This function does not mutate the DataFrame.

    Parameters
    ----------
    df:
        DataFrame to inspect.

    explicit_column:
        Optional explicit timestamp column name.

    Returns
    -------
    TimestampDetectionResult
        Detection and parsing information.

    Raises
    ------
    KeyError
        If an explicitly requested timestamp column does not exist.
    """

    notes: list[str] = []

    if explicit_column is not None:
        if explicit_column not in df.columns:
            raise KeyError(
                f"explicit_column={explicit_column!r} not found in DataFrame "
                f"columns: {list(df.columns)}"
            )

        candidates = [explicit_column]

    else:
        candidates = [
            column
            for column in df.columns
            if any(
                hint in str(column).strip().lower()
                for hint in TIMESTAMP_NAME_HINTS
            )
        ]

    if not candidates:
        message = (
            "No timestamp-like column detected by name. "
            "Row-level chronological splitting cannot proceed without "
            "an explicit timestamp column."
        )

        notes.append(message)
        logger.warning(message)

        return TimestampDetectionResult(
            column=None,
            parsed_successfully=False,
            n_unparseable=0,
            notes=notes,
        )

    column = candidates[0]

    if len(candidates) > 1:
        message = (
            f"Multiple timestamp-like columns found: {candidates}. "
            f"Using the first detected column {column!r}. "
            "Pass explicit_column= to override this behavior."
        )

        notes.append(message)
        logger.warning(message)

    parsed = pd.to_datetime(df[column], errors="coerce")

    original_nulls = df[column].isna()
    parsed_nulls = parsed.isna()

    n_unparseable = int((parsed_nulls & ~original_nulls).sum())
    n_unparseable = max(n_unparseable, 0)

    if n_unparseable > 0:
        message = (
            f"{n_unparseable} value(s) in timestamp column {column!r} "
            "could not be parsed. Values were coerced to NaT for "
            "inspection only; the source DataFrame was not modified."
        )

        notes.append(message)
        logger.warning(message)

    non_null_count = int((~original_nulls).sum())

    if non_null_count == 0:
        parsed_successfully = False

        message = (
            f"Timestamp column {column!r} contains no non-null values. "
            "It cannot be used for chronological splitting."
        )

        notes.append(message)
        logger.warning(message)

    else:
        parsed_successfully = n_unparseable < non_null_count

    return TimestampDetectionResult(
        column=column,
        parsed_successfully=parsed_successfully,
        n_unparseable=n_unparseable,
        notes=notes,
    )


# ============================================================================
# Row-level chronological split
# ============================================================================


@dataclass
class TemporalSplitResult:
    """
    Result of a chronological train/test split.
    """

    train_df: pd.DataFrame
    test_df: pd.DataFrame

    timestamp_column: str
    split_timestamp: pd.Timestamp

    train_start: pd.Timestamp
    train_end: pd.Timestamp

    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def summary(self) -> dict:
        """Return a serializable summary of the temporal split."""

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
    Split a DataFrame chronologically into train and test sets.

    The earliest ``1 - test_size`` fraction becomes training data.
    The latest ``test_size`` fraction becomes test data.

    No random shuffling or stratification is performed.

    Parameters
    ----------
    df:
        Input DataFrame.

    timestamp_column:
        Column containing timestamps.

    test_size:
        Fraction of rows to reserve for the chronologically latest
        test set. Must be strictly between 0 and 1.

    Returns
    -------
    TemporalSplitResult
        Chronological train/test split and temporal metadata.

    Raises
    ------
    KeyError
        If timestamp_column is missing.

    ValueError
        If the DataFrame is empty, test_size is invalid, or one or more
        timestamps cannot be parsed.
    """

    if timestamp_column not in df.columns:
        raise KeyError(
            f"timestamp_column={timestamp_column!r} not in DataFrame."
        )

    if not 0.0 < test_size < 1.0:
        raise ValueError(
            f"test_size must be strictly between 0 and 1, got {test_size}"
        )

    if len(df) == 0:
        raise ValueError("Cannot split an empty DataFrame.")

    working = df.copy()

    # Parse timestamps into a temporary column.
    working["_parsed_ts"] = pd.to_datetime(
        working[timestamp_column],
        errors="coerce",
    )

    n_bad = int(working["_parsed_ts"].isna().sum())

    if n_bad > 0:
        raise ValueError(
            f"{n_bad} row(s) have unparseable timestamps in "
            f"{timestamp_column!r}. Resolve the timestamp problem before "
            "splitting. The chronological splitter refuses to silently "
            "drop invalid rows."
        )

    # Stable sorting ensures deterministic ordering when multiple rows
    # have identical timestamps.
    working = (
        working
        .sort_values("_parsed_ts", kind="mergesort")
        .reset_index(drop=True)
    )

    # Calculate the number of training rows.
    split_idx = int(round(len(working) * (1.0 - test_size)))

    # Guarantee that both partitions contain at least one row.
    split_idx = min(
        max(split_idx, 1),
        len(working) - 1,
    )

    train = (
        working
        .iloc[:split_idx]
        .drop(columns=["_parsed_ts"])
        .reset_index(drop=True)
    )

    test = (
        working
        .iloc[split_idx:]
        .drop(columns=["_parsed_ts"])
        .reset_index(drop=True)
    )

    train_ts = pd.to_datetime(
        train[timestamp_column],
        errors="coerce",
    )

    test_ts = pd.to_datetime(
        test[timestamp_column],
        errors="coerce",
    )

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
        "Chronological split: "
        "train=%d rows [%s -> %s], "
        "test=%d rows [%s -> %s]",
        len(train),
        result.train_start,
        result.train_end,
        len(test),
        result.test_start,
        result.test_end,
    )

    return result


# ============================================================================
# Row-level temporal leakage verification
# ============================================================================


@dataclass
class TemporalLeakageCheck:
    """
    Result of verifying a row-level temporal split.
    """

    passed: bool
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    n_overlapping_rows: int
    message: str


def verify_no_temporal_leakage(
    split: TemporalSplitResult,
) -> TemporalLeakageCheck:
    """
    Verify that a chronological train/test split contains no temporal
    leakage.

    Two conditions must hold:

    1. The latest training timestamp must be before or equal to the
       earliest testing timestamp.

    2. No exact timestamp value may occur on both sides of the split.

    Because this project treats temporal leakage as a hard failure,
    callers should stop training/evaluation when ``passed`` is False.

    Parameters
    ----------
    split:
        A TemporalSplitResult produced by chronological_train_test_split.

    Returns
    -------
    TemporalLeakageCheck
        Verification result.
    """

    train_end = split.train_end
    test_start = split.test_start

    train_ts = pd.to_datetime(
        split.train_df[split.timestamp_column],
        errors="coerce",
    )

    test_ts = pd.to_datetime(
        split.test_df[split.timestamp_column],
        errors="coerce",
    )

    train_unique = set(train_ts.dropna().unique())
    test_unique = set(test_ts.dropna().unique())

    overlapping_values = train_unique.intersection(test_unique)

    n_overlapping_rows = int(
        train_ts.isin(overlapping_values).sum()
        + test_ts.isin(overlapping_values).sum()
    )

    order_ok = train_end <= test_start

    passed = order_ok and n_overlapping_rows == 0

    if passed:
        message = (
            f"Temporal leakage check PASSED: "
            f"train ends {train_end}, "
            f"test starts {test_start}, "
            "and no timestamp values overlap."
        )

        logger.info(message)

    else:
        message = (
            f"Temporal leakage check FAILED: "
            f"train_end={train_end}, "
            f"test_start={test_start}, "
            f"order_ok={order_ok}, "
            f"n_overlapping_rows={n_overlapping_rows} "
            f"({len(overlapping_values)} distinct overlapping "
            "timestamp value(s)). "
            "Do not proceed to training with this split."
        )

        logger.error(message)

    return TemporalLeakageCheck(
        passed=passed,
        train_end=train_end,
        test_start=test_start,
        n_overlapping_rows=n_overlapping_rows,
        message=message,
    )


# ============================================================================
# Capture-day temporal ordering
# ============================================================================

# CIC-IDS2017 was collected across five weekday capture periods.
#
# This ordering is used only when row-level timestamps are unavailable.
CIC_IDS2017_DAY_ORDER = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
)


_DAY_NAME_PATTERN = re.compile(
    r"(?<![a-zA-Z])("
    + "|".join(CIC_IDS2017_DAY_ORDER)
    + r")(?![a-zA-Z])",
    re.IGNORECASE,
)


def detect_capture_day(
    filename: Path | str,
) -> Optional[str]:
    """
    Detect the CIC-IDS2017 capture day from a filename.

    Examples
    --------
    ``Tuesday-WorkingHours.pcap_ISCX.csv`` -> ``"tuesday"``

    ``Friday-WorkingHours-Afternoon-DDos.parquet`` -> ``"friday"``

    If no recognized day is present, None is returned.

    The function never guesses a day.
    """

    name = Path(filename).name

    match = _DAY_NAME_PATTERN.search(name)

    if match is None:
        return None

    return match.group(1).lower()


@dataclass
class DayAssignment:
    """
    Mapping between dataset files and CIC-IDS2017 capture days.
    """

    file_days: dict[str, Optional[str]] = field(default_factory=dict)
    undetected_files: list[str] = field(default_factory=list)

    @property
    def all_detected(self) -> bool:
        """Return True if every file has a detected capture day."""

        return len(self.undetected_files) == 0

    @property
    def distinct_days(self) -> list[str]:
        """Return detected days in chronological CIC-IDS2017 order."""

        days = {
            day
            for day in self.file_days.values()
            if day is not None
        }

        return sorted(
            days,
            key=CIC_IDS2017_DAY_ORDER.index,
        )


def assign_capture_days(
    files: Sequence[Path],
) -> DayAssignment:
    """
    Assign capture days to dataset files using filenames.

    No file contents are read.

    Files without a recognizable day are returned in
    ``undetected_files`` rather than being silently dropped or guessed.
    """

    file_days: dict[str, Optional[str]] = {}
    undetected: list[str] = []

    for file_path in files:
        day = detect_capture_day(file_path)

        file_days[str(file_path)] = day

        if day is None:
            undetected.append(str(file_path))

    assignment = DayAssignment(
        file_days=file_days,
        undetected_files=undetected,
    )

    if undetected:
        logger.warning(
            "Could not detect a capture day from %d filename(s): %s. "
            "Day-based splitting requires every file to be identifiable "
            "by filename. Rename the file(s) to include the day name, "
            "or use row-level timestamp splitting if a genuine timestamp "
            "column is available.",
            len(undetected),
            undetected,
        )

    else:
        logger.info(
            "Detected capture days for %d file(s): %s",
            len(files),
            {
                Path(path).name: day
                for path, day in file_days.items()
            },
        )

    return assignment


def label_chunk_with_capture_day(
    chunk: pd.DataFrame,
    day_column: str = "capture_day",
) -> pd.DataFrame:
    """
    Add the CIC-IDS2017 capture day to a loaded chunk.

    The source filename is expected in:

        chunk.attrs["source_file"]

    This attribute is populated by the chunk iterators in
    ``src.data.loader``.

    Parameters
    ----------
    chunk:
        DataFrame produced by one of the dataset loader iterators.

    day_column:
        Name of the new capture-day column.

    Returns
    -------
    pd.DataFrame
        Copy of the input DataFrame with the capture day added.

    Raises
    ------
    KeyError
        If source_file metadata is unavailable.

    ValueError
        If the source filename does not contain a recognizable capture day.
    """

    if "source_file" not in chunk.attrs:
        raise KeyError(
            "chunk.attrs['source_file'] is missing. "
            "label_chunk_with_capture_day requires chunks produced by "
            "src.data.loader's iterators "
            "(iter_parquet_chunks / iter_csv_chunks / "
            "iter_dataset_chunks)."
        )

    source_file = chunk.attrs["source_file"]

    day = detect_capture_day(source_file)

    if day is None:
        raise ValueError(
            f"Could not detect a CIC-IDS2017 capture day from filename "
            f"{source_file!r}. Day-based splitting refuses to guess. "
            "Rename the file to include monday..friday, or use row-level "
            "timestamp splitting when a genuine timestamp column exists."
        )

    out = chunk.copy()
    out[day_column] = day

    return out


# ============================================================================
# Day-based train/test split
# ============================================================================


@dataclass
class DaySplitResult:
    """
    Result of a capture-day-based train/test split.
    """

    train_df: pd.DataFrame
    test_df: pd.DataFrame

    day_column: str

    train_days: list[str]
    test_days: list[str]

    def summary(self) -> dict:
        """Return a serializable split summary."""

        return {
            "day_column": self.day_column,
            "train_days": self.train_days,
            "test_days": self.test_days,
            "train_rows": len(self.train_df),
            "test_rows": len(self.test_df),
        }


def day_based_train_test_split(
    df: pd.DataFrame,
    day_column: str,
    test_days: Sequence[str],
    train_days: Optional[Sequence[str]] = None,
) -> DaySplitResult:
    """
    Split a DataFrame by complete CIC-IDS2017 capture days.

    This is the fallback temporal strategy when row-level timestamps are
    unavailable.

    Parameters
    ----------
    df:
        Input DataFrame containing the capture-day column.

    day_column:
        Name of the capture-day column.

    test_days:
        Day(s) to hold out for testing.

    train_days:
        Day(s) to use for training. If None, every present day not listed
        in test_days is used.

    Returns
    -------
    DaySplitResult
        Training and testing DataFrames plus split metadata.

    Raises
    ------
    KeyError
        If day_column is missing.

    ValueError
        If day names are invalid, train/test sets overlap, either side
        is empty, requested days have no rows, or chronological ordering
        is violated.
    """

    if day_column not in df.columns:
        raise KeyError(
            f"day_column={day_column!r} not in DataFrame."
        )

    # Normalize test-day names.
    normalized_test_days = [
        str(day).strip().lower()
        for day in test_days
    ]

    for day in normalized_test_days:
        if day not in CIC_IDS2017_DAY_ORDER:
            raise ValueError(
                f"Unknown day name in test_days: {day!r}. "
                f"Must be one of {CIC_IDS2017_DAY_ORDER}."
            )

    test_days = normalized_test_days

    # Determine which recognized days are actually present.
    present_days = sorted(
        {
            str(day).strip().lower()
            for day in df[day_column].dropna().unique()
            if str(day).strip().lower() in CIC_IDS2017_DAY_ORDER
        },
        key=CIC_IDS2017_DAY_ORDER.index,
    )

    if train_days is None:
        normalized_train_days = [
            day
            for day in present_days
            if day not in test_days
        ]

    else:
        normalized_train_days = [
            str(day).strip().lower()
            for day in train_days
        ]

        for day in normalized_train_days:
            if day not in CIC_IDS2017_DAY_ORDER:
                raise ValueError(
                    f"Unknown day name in train_days: {day!r}. "
                    f"Must be one of {CIC_IDS2017_DAY_ORDER}."
                )

    train_days = normalized_train_days

    # Check train/test overlap.
    overlap = set(train_days).intersection(test_days)

    if overlap:
        raise ValueError(
            "train_days and test_days overlap: "
            f"{sorted(overlap)}"
        )

    if not train_days or not test_days:
        raise ValueError(
            "Both train_days and test_days must be non-empty. "
            f"Got train_days={train_days}, "
            f"test_days={test_days}."
        )

    # Enforce chronological ordering.
    max_train_idx = max(
        CIC_IDS2017_DAY_ORDER.index(day)
        for day in train_days
    )

    min_test_idx = min(
        CIC_IDS2017_DAY_ORDER.index(day)
        for day in test_days
    )

    if min_test_idx <= max_train_idx:
        raise ValueError(
            "Every test day must chronologically follow every train day "
            f"in CIC_IDS2017_DAY_ORDER={CIC_IDS2017_DAY_ORDER}. "
            f"Got train_days={train_days} "
            f"(max index={max_train_idx}), "
            f"test_days={test_days} "
            f"(min index={min_test_idx})."
        )

    # Create partitions.
    normalized_day_series = (
        df[day_column]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    train_df = (
        df[normalized_day_series.isin(train_days)]
        .reset_index(drop=True)
    )

    test_df = (
        df[normalized_day_series.isin(test_days)]
        .reset_index(drop=True)
    )

    if len(train_df) == 0:
        raise ValueError(
            f"No rows found for train_days={train_days}."
        )

    if len(test_df) == 0:
        raise ValueError(
            f"No rows found for test_days={test_days}."
        )

    result = DaySplitResult(
        train_df=train_df,
        test_df=test_df,
        day_column=day_column,
        train_days=list(train_days),
        test_days=list(test_days),
    )

    logger.info(
        "Day-based split: "
        "train_days=%s (%d rows), "
        "test_days=%s (%d rows)",
        train_days,
        len(train_df),
        test_days,
        len(test_df),
    )

    return result


# ============================================================================
# Day-based leakage verification
# ============================================================================


@dataclass
class DayLeakageCheck:
    """
    Result of verifying a capture-day train/test split.
    """

    passed: bool
    train_days: list[str]
    test_days: list[str]
    message: str


def verify_no_day_leakage(
    split: DaySplitResult,
) -> DayLeakageCheck:
    """
    Verify that train and test capture days do not overlap and that the
    test days occur strictly after the training days.

    This is a hard gate for day-based temporal evaluation.
    """

    train_days = [
        str(day).strip().lower()
        for day in split.train_days
    ]

    test_days = [
        str(day).strip().lower()
        for day in split.test_days
    ]

    overlap = set(train_days).intersection(test_days)

    if train_days and test_days:
        max_train_idx = max(
            CIC_IDS2017_DAY_ORDER.index(day)
            for day in train_days
        )

        min_test_idx = min(
            CIC_IDS2017_DAY_ORDER.index(day)
            for day in test_days
        )

        order_ok = min_test_idx > max_train_idx

    else:
        order_ok = False

    passed = order_ok and not overlap

    if passed:
        message = (
            "Day leakage check PASSED: "
            f"train_days={train_days}, "
            f"test_days={test_days}, "
            "no overlap and chronological order preserved."
        )

        logger.info(message)

    else:
        message = (
            "Day leakage check FAILED: "
            f"train_days={train_days}, "
            f"test_days={test_days}, "
            f"overlap={sorted(overlap)}, "
            f"order_ok={order_ok}. "
            "Do not proceed to training with this split."
        )

        logger.error(message)

    return DayLeakageCheck(
        passed=passed,
        train_days=train_days,
        test_days=test_days,
        message=message,
    )