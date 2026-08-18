
#!/usr/bin/env python
"""
Day 1 - Step 2: Full preprocessing pipeline (memory-conscious).

Pipeline:
    1. Discover CIC-IDS2017 files.
    2/3. Load via loader.iter_dataset_chunks.
    4/5. Validate schema + NaN/Inf per chunk.
    6/7. Identify leakage-prone features + lightweight feature selection.
    8. Normalize labels.
    9. Validate physically constrained non-negative features.
    10. Detect timestamp column.
    11. Chronological train/test split.
    12. Verify no temporal leakage.

If no row-level timestamp exists, the pipeline falls back to a
capture-day-based temporal split using the original CIC-IDS2017
file identity.

Memory strategy:
    Pass 1 processes the raw dataset chunk-by-chunk and writes cleaned,
    feature-selected chunks to data/processed/day1/parts/.

    Pass 2 loads only the cleaned/reduced parts to perform the global
    temporal split.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cleaner import (  # noqa: E402
    CleaningLog,
    handle_nan_inf,
    normalize_labels,
    validate_nonnegative_features,
)

from src.data.feature_engineering import (  # noqa: E402
    identify_leakage_prone_features,
    select_features,
)

from src.data.loader import DEFAULT_CHUNK_SIZE, iter_dataset_chunks  # noqa: E402

from src.data.temporal_split import (  # noqa: E402
    chronological_train_test_split,
    day_based_train_test_split,
    detect_timestamp_column,
    label_chunk_with_capture_day,
    verify_no_day_leakage,
    verify_no_temporal_leakage,
)

from src.data.validator import validate_nan_inf, validate_schema  # noqa: E402


LOG_DIR = PROJECT_ROOT / "logs"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "CIC-IDS2017",
    )

    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "day1",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
    )

    parser.add_argument(
        "--timestamp-column",
        type=str,
        default=None,
        help="Force a specific timestamp column instead of auto-detecting.",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--test-days",
        type=str,
        default="friday",
        help=(
            "Comma-separated capture day(s) to hold out as the test set "
            "when no row-level timestamp column is found. Default: friday."
        ),
    )

    parser.add_argument(
        "--train-days",
        type=str,
        default=None,
        help=(
            "Comma-separated capture day(s) to use for training in the "
            "day-based fallback. Default: every day not in --test-days."
        ),
    )

    parser.add_argument(
        "--overwrite-parts",
        action="store_true",
        help=(
            "Delete and recreate data/processed/day1/parts/ "
            "before running."
        ),
    )

    return parser


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_DIR / "prepare_data.log"),
        ],
    )

    return logging.getLogger("prepare_data")


def pass_one_clean_chunks(
    raw_dir: Path,
    parts_dir: Path,
    chunk_size: int,
    logger: logging.Logger,
) -> tuple[str, list[str], bool]:
    """
    Stream raw chunks -> validate -> clean -> feature-select -> write parts.

    Returns:
        primary_label_col
        selected_feature_names
        day_labeling_ok
    """

    parts_dir.mkdir(parents=True, exist_ok=True)

    all_decisions = CleaningLog()

    primary_label_col: str | None = None
    selected_feature_names: list[str] | None = None

    day_labeling_ok = True

    for i, chunk in enumerate(
        iter_dataset_chunks(raw_dir, chunk_size=chunk_size)
    ):
        logger.info(
            "Processing raw chunk %d: %d rows, %d columns",
            i,
            len(chunk),
            len(chunk.columns),
        )

        # ---------------------------------------------------------------
        # Step 1: Validate schema
        # ---------------------------------------------------------------

        schema_report = validate_schema(chunk)

        # ---------------------------------------------------------------
        # Step 2: Validate NaN / Inf
        # ---------------------------------------------------------------

        validate_nan_inf(chunk)

        # ---------------------------------------------------------------
        # Detect primary label column
        # ---------------------------------------------------------------

        if primary_label_col is None:
            primary_label_col = schema_report.detected_label_column

            if primary_label_col is None:
                raise ValueError(
                    "No label column could be detected in the first chunk. "
                    "Cannot proceed with supervised preprocessing. "
                    "Inspect the dataset with "
                    "scripts/inspect_dataset.py first."
                )

            logger.info(
                "Using detected primary label column: %r",
                primary_label_col,
            )

        # ---------------------------------------------------------------
        # Step 3: Normalize labels
        # ---------------------------------------------------------------

        cleaned, all_decisions = normalize_labels(
            chunk,
            label_col=primary_label_col,
            log=all_decisions,
        )

        # ---------------------------------------------------------------
        # Step 4: Handle NaN / Inf
        # ---------------------------------------------------------------

        cleaned, all_decisions = handle_nan_inf(
            cleaned,
            log=all_decisions,
        )

        # ---------------------------------------------------------------
        # Step 5: Validate non-negative network-flow features
        # ---------------------------------------------------------------
        #
        # These CIC-IDS2017 measurements should not be negative.
        # If invalid negative values exist, cleaner.validate_nonnegative_features
        # handles them according to the project's cleaning policy.
        #
        # We only validate columns that actually exist in the dataset.
        # This keeps the pipeline compatible with slightly different
        # CIC-IDS2017 parquet releases.

        nonnegative_features = [
            "Flow Duration",
            "Total Fwd Packets",
            "Total Backward Packets",
            "Fwd Packets Length Total",
            "Bwd Packets Length Total",
            "Fwd Header Length",
            "Bwd Header Length",
        ]

        available_nonnegative_features = [
            column
            for column in nonnegative_features
            if column in cleaned.columns
        ]

        if available_nonnegative_features:
            cleaned, all_decisions = validate_nonnegative_features(
                cleaned,
                columns=available_nonnegative_features,
                log=all_decisions,
            )

        # ---------------------------------------------------------------
        # Step 6: Capture-day labeling
        # ---------------------------------------------------------------

        if day_labeling_ok:
            try:
                cleaned = label_chunk_with_capture_day(cleaned)

            except (KeyError, ValueError) as exc:
                day_labeling_ok = False

                logger.warning(
                    "Capture-day labeling unavailable for chunk %d (%s). "
                    "The day-based temporal-split fallback will not be "
                    "available; a timestamp column will be required instead.",
                    i,
                    exc,
                )

        # ---------------------------------------------------------------
        # Step 7: Identify leakage-prone features
        # ---------------------------------------------------------------

        leakage_report = identify_leakage_prone_features(
            cleaned,
            primary_label_col=primary_label_col,
        )

        exclude_columns = list(
            set(
                [
                    primary_label_col,
                    "label_multiclass_raw",
                    "label_multiclass",
                    "label_binary",
                    "capture_day",
                ]
                + leakage_report.flagged_columns
            )
        )

        # ---------------------------------------------------------------
        # Step 8: Lightweight feature selection
        # ---------------------------------------------------------------

        cleaned, chunk_features, all_decisions = select_features(
            cleaned,
            exclude_columns=exclude_columns,
            log=all_decisions,
        )

        # ---------------------------------------------------------------
        # Keep feature schema consistent across chunks
        # ---------------------------------------------------------------

        if selected_feature_names is None:
            selected_feature_names = chunk_features

        else:
            missing = set(selected_feature_names) - set(chunk_features)

            if missing:
                logger.warning(
                    "Chunk %d is missing previously-selected feature(s): %s. "
                    "Intersecting feature set to keep schema consistent.",
                    i,
                    sorted(missing),
                )

                selected_feature_names = [
                    column
                    for column in selected_feature_names
                    if column in chunk_features
                ]

        # ---------------------------------------------------------------
        # Write cleaned chunk
        # ---------------------------------------------------------------

        part_path = parts_dir / f"part_{i:05d}.parquet"

        cleaned.to_parquet(
            part_path,
            index=False,
        )

        logger.info(
            "Wrote cleaned chunk %d (%d rows) -> %s",
            i,
            len(cleaned),
            part_path,
        )

    if primary_label_col is None or selected_feature_names is None:
        raise ValueError(
            "No data chunks were found or processed under raw_dir."
        )

    # ---------------------------------------------------------------
    # Save preprocessing decision log
    # ---------------------------------------------------------------

    decisions_path = LOG_DIR / "prepare_data_decisions.json"

    decisions_path.write_text(
        json.dumps(
            all_decisions.to_dict(),
            indent=2,
        )
    )

    logger.info(
        "Wrote full preprocessing decision log -> %s",
        decisions_path,
    )

    return (
        primary_label_col,
        selected_feature_names,
        day_labeling_ok,
    )


def pass_two_split(
    parts_dir: Path,
    processed_dir: Path,
    label_col: str,
    feature_names: list[str],
    explicit_timestamp_col: str | None,
    test_size: float,
    day_labeling_ok: bool,
    test_days: list[str],
    train_days: list[str] | None,
    logger: logging.Logger,
) -> dict:

    import pyarrow.dataset as ds

    # ---------------------------------------------------------------
    # Load cleaned/reduced parts
    # ---------------------------------------------------------------

    dataset = ds.dataset(
        str(parts_dir),
        format="parquet",
    )

    table = dataset.to_table()

    df = table.to_pandas()

    logger.info(
        "Loaded cleaned/reduced dataset for splitting: %d rows, %d columns",
        *df.shape,
    )

    # ---------------------------------------------------------------
    # Detect timestamp
    # ---------------------------------------------------------------

    ts_result = detect_timestamp_column(
        df,
        explicit_column=explicit_timestamp_col,
    )

    # ===============================================================
    # Primary path: row-level timestamp split
    # ===============================================================

    if ts_result.column is not None:

        split_method = "row_timestamp"

        timestamp_column = ts_result.column

        logger.info(
            "Using timestamp column %r for chronological split.",
            timestamp_column,
        )

        # Timestamp is used only for ordering/splitting.
        # It must not be used as a model feature.

        if timestamp_column in feature_names:
            logger.info(
                "Excluding timestamp column %r from model features.",
                timestamp_column,
            )

            feature_names = [
                column
                for column in feature_names
                if column != timestamp_column
            ]

        split = chronological_train_test_split(
            df,
            timestamp_column=timestamp_column,
            test_size=test_size,
        )

        leakage_check = verify_no_temporal_leakage(split)

        if not leakage_check.passed:
            raise RuntimeError(
                f"Temporal leakage check failed: "
                f"{leakage_check.message}"
            )

        split_summary = split.summary()

        train_df = split.train_df
        test_df = split.test_df

    # ===============================================================
    # Fallback: capture-day split
    # ===============================================================

    else:

        logger.warning(
            "No row-level timestamp column detected/provided. "
            "Falling back to capture-day temporal splitting."
        )

        if not day_labeling_ok or "capture_day" not in df.columns:
            raise ValueError(
                "Day-based fallback is unavailable: capture-day labeling "
                "failed for one or more source files. Either rename raw "
                "files to include the CIC-IDS2017 day name "
                "(monday..friday) and re-run, or provide a real row-level "
                "timestamp column via --timestamp-column."
            )

        split_method = "capture_day"

        split = day_based_train_test_split(
            df,
            day_column="capture_day",
            test_days=test_days,
            train_days=train_days,
        )

        day_leakage_check = verify_no_day_leakage(split)

        if not day_leakage_check.passed:
            raise RuntimeError(
                f"Day leakage check failed: "
                f"{day_leakage_check.message}"
            )

        leakage_check = day_leakage_check

        split_summary = split.summary()

        train_df = split.train_df
        test_df = split.test_df

        if "capture_day" in feature_names:
            feature_names = [
                column
                for column in feature_names
                if column != "capture_day"
            ]

    # ---------------------------------------------------------------
    # Write final train/test datasets
    # ---------------------------------------------------------------

    processed_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_path = processed_dir / "train.parquet"
    test_path = processed_dir / "test.parquet"

    train_df.to_parquet(
        train_path,
        index=False,
    )

    test_df.to_parquet(
        test_path,
        index=False,
    )

    logger.info(
        "Wrote train split -> %s (%d rows)",
        train_path,
        len(train_df),
    )

    logger.info(
        "Wrote test split -> %s (%d rows)",
        test_path,
        len(test_df),
    )

    # ---------------------------------------------------------------
    # Save metadata
    # ---------------------------------------------------------------

    metadata = {
        "label_col": label_col,
        "feature_names": feature_names,
        "split_method": split_method,
        "timestamp_column": ts_result.column,
        "split_summary": split_summary,
        "leakage_check": {
            "passed": leakage_check.passed,
            "message": leakage_check.message,
        },
    }

    metadata_path = processed_dir / "split_metadata.json"

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            default=str,
        )
    )

    logger.info(
        "Wrote split metadata -> %s",
        metadata_path,
    )

    return metadata


def main() -> int:

    args = build_arg_parser().parse_args()

    logger = setup_logging()

    parts_dir = args.processed_dir / "parts"

    # ---------------------------------------------------------------
    # Remove old cleaned parts when requested
    # ---------------------------------------------------------------

    if args.overwrite_parts and parts_dir.exists():

        logger.info(
            "Removing existing processed parts: %s",
            parts_dir,
        )

        shutil.rmtree(parts_dir)

    # ---------------------------------------------------------------
    # Parse day arguments
    # ---------------------------------------------------------------

    test_days = [
        day.strip().lower()
        for day in args.test_days.split(",")
        if day.strip()
    ]

    train_days = (
        [
            day.strip().lower()
            for day in args.train_days.split(",")
            if day.strip()
        ]
        if args.train_days
        else None
    )

    # ---------------------------------------------------------------
    # Run preprocessing
    # ---------------------------------------------------------------

    try:

        label_col, feature_names, day_labeling_ok = pass_one_clean_chunks(
            args.raw_dir,
            parts_dir,
            args.chunk_size,
            logger,
        )

        pass_two_split(
            parts_dir,
            args.processed_dir,
            label_col,
            feature_names,
            args.timestamp_column,
            args.test_size,
            day_labeling_ok,
            test_days,
            train_days,
            logger,
        )

    except (
        FileNotFoundError,
        ValueError,
        RuntimeError,
        KeyError,
    ) as exc:

        logger.error(
            "prepare_data failed: %s",
            exc,
        )

        return 1

    logger.info(
        "Day 1 data preparation complete."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

