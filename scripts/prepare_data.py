#!/usr/bin/env python
"""
Day 1 - Step 2: Full preprocessing pipeline (memory-conscious).

Pipeline (per project's numbered Day 1 scope, items 1-11):
    1. Discover CIC-IDS2017 files (parquet-first, chunked-CSV fallback).
    2/3. Load via loader.iter_dataset_chunks (bounded memory per chunk).
    4/5. Validate schema + NaN/Inf per chunk.
    6/7. Identify leakage-prone features + lightweight feature selection.
    8.   Normalize labels.
    9.   Detect timestamp column.
    10.  Chronological train/test split.
    11.  Verify no temporal leakage.

Memory strategy
----------------
Pass 1 processes the raw dataset strictly chunk-by-chunk (never holding
more than one chunk in memory) and writes each cleaned/feature-selected
chunk immediately to data/processed/day1/parts/*.parquet.

Pass 2 reads the already-cleaned, already-reduced parts back in with
pyarrow (typically far smaller than the raw dataset after dropping
identifier/near-constant columns) to perform the single global
chronological sort that a correct temporal split requires. This is a
deliberate, documented exception to "never concatenate the complete
dataset into RAM": the *raw* dataset is never fully materialized, only
the cleaned/reduced one, and only because a correct chronological split
is inherently a global operation. If this still does not fit in 8GB for
the machine in use, increase chunking granularity or subsample before
re-running.

This script is NOT run automatically. Run it yourself after placing the
real dataset under data/raw/CIC-IDS2017/.
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

from src.data.cleaner import CleaningLog, handle_nan_inf, normalize_labels  # noqa: E402
from src.data.feature_engineering import (  # noqa: E402
    identify_leakage_prone_features,
    select_features,
)
from src.data.loader import DEFAULT_CHUNK_SIZE, iter_dataset_chunks  # noqa: E402
from src.data.temporal_split import (  # noqa: E402
    chronological_train_test_split,
    detect_timestamp_column,
    verify_no_temporal_leakage,
)
from src.data.validator import validate_nan_inf, validate_schema  # noqa: E402

LOG_DIR = PROJECT_ROOT / "logs"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir", type=Path, default=PROJECT_ROOT / "data" / "raw" / "CIC-IDS2017"
    )
    parser.add_argument(
        "--processed-dir", type=Path, default=PROJECT_ROOT / "data" / "processed" / "day1"
    )
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument(
        "--timestamp-column",
        type=str,
        default=None,
        help="Force a specific timestamp column instead of auto-detecting.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--overwrite-parts",
        action="store_true",
        help="Delete and recreate data/processed/day1/parts/ before running.",
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
    raw_dir: Path, parts_dir: Path, chunk_size: int, logger: logging.Logger
) -> tuple[str, list[str]]:
    """
    Stream raw chunks -> validate -> clean -> feature-select -> write parts.
    Returns (primary_label_col, selected_feature_names) from the first
    chunk (schema is assumed consistent across chunks; drift is logged by
    validate_schema/merge_schema_reports if it occurs).
    """
    parts_dir.mkdir(parents=True, exist_ok=True)
    all_decisions = CleaningLog()

    primary_label_col: str | None = None
    selected_feature_names: list[str] | None = None
    exclude_columns: list[str] = []

    for i, chunk in enumerate(iter_dataset_chunks(raw_dir, chunk_size=chunk_size)):
        schema_report = validate_schema(chunk)
        validate_nan_inf(chunk)

        if primary_label_col is None:
            primary_label_col = schema_report.detected_label_column
            if primary_label_col is None:
                raise ValueError(
                    "No label column could be detected in the first chunk. "
                    "Cannot proceed with label normalization / supervised "
                    "baseline training. Inspect the dataset with "
                    "scripts/inspect_dataset.py first."
                )
            logger.info("Using detected primary label column: %r", primary_label_col)

        cleaned, all_decisions = normalize_labels(
            chunk, label_col=primary_label_col, log=all_decisions
        )
        cleaned, all_decisions = handle_nan_inf(cleaned, log=all_decisions)

        leakage_report = identify_leakage_prone_features(
            cleaned, primary_label_col=primary_label_col
        )
        exclude_columns = list(
            set(
                [primary_label_col, "label_multiclass_raw", "label_multiclass", "label_binary"]
                + leakage_report.flagged_columns
            )
        )

        cleaned, chunk_features, all_decisions = select_features(
            cleaned, exclude_columns=exclude_columns, log=all_decisions
        )

        if selected_feature_names is None:
            selected_feature_names = chunk_features
        else:
            # Keep only features present in every chunk so downstream
            # concatenation has a consistent schema; log any drift instead
            # of silently reindexing.
            missing = set(selected_feature_names) - set(chunk_features)
            if missing:
                logger.warning(
                    "Chunk %d is missing previously-selected feature(s): %s. "
                    "Intersecting feature set to keep schema consistent.",
                    i,
                    missing,
                )
                selected_feature_names = [c for c in selected_feature_names if c in chunk_features]

        part_path = parts_dir / f"part_{i:05d}.parquet"
        cleaned.to_parquet(part_path, index=False)
        logger.info("Wrote cleaned chunk %d (%d rows) -> %s", i, len(cleaned), part_path)

    if primary_label_col is None or selected_feature_names is None:
        raise ValueError("No data chunks were found/processed under raw_dir.")

    decisions_path = LOG_DIR / "prepare_data_decisions.json"
    decisions_path.write_text(json.dumps(all_decisions.to_dict(), indent=2))
    logger.info("Wrote full preprocessing decision log -> %s", decisions_path)

    return primary_label_col, selected_feature_names


def pass_two_split(
    parts_dir: Path,
    processed_dir: Path,
    label_col: str,
    feature_names: list[str],
    explicit_timestamp_col: str | None,
    test_size: float,
    logger: logging.Logger,
) -> dict:
    import pyarrow.dataset as ds

    dataset = ds.dataset(str(parts_dir), format="parquet")
    table = dataset.to_table()
    df = table.to_pandas()
    logger.info("Loaded cleaned/reduced dataset for splitting: %d rows, %d columns", *df.shape)

    ts_result = detect_timestamp_column(df, explicit_column=explicit_timestamp_col)
    if ts_result.column is None:
        raise ValueError(
            "No timestamp column detected/provided. Pass --timestamp-column "
            "explicitly once you know the real column name (run "
            "scripts/inspect_dataset.py to see available columns)."
        )

    # The timestamp column is used to order/split the data, not as a model
    # feature (it is also non-numeric as a raw string/datetime and would
    # otherwise leak the exact split boundary to the model). Exclude it
    # from the feature set here rather than in select_features, since only
    # this step knows which column was actually used for the split.
    if ts_result.column in feature_names:
        logger.info(
            "Excluding timestamp column %r from model features (used for "
            "splitting only).",
            ts_result.column,
        )
        feature_names = [c for c in feature_names if c != ts_result.column]

    split = chronological_train_test_split(
        df, timestamp_column=ts_result.column, test_size=test_size
    )
    leakage_check = verify_no_temporal_leakage(split)
    if not leakage_check.passed:
        raise RuntimeError(f"Temporal leakage check failed: {leakage_check.message}")

    processed_dir.mkdir(parents=True, exist_ok=True)
    train_path = processed_dir / "train.parquet"
    test_path = processed_dir / "test.parquet"
    split.train_df.to_parquet(train_path, index=False)
    split.test_df.to_parquet(test_path, index=False)
    logger.info("Wrote train split -> %s (%d rows)", train_path, len(split.train_df))
    logger.info("Wrote test split -> %s (%d rows)", test_path, len(split.test_df))

    metadata = {
        "label_col": label_col,
        "feature_names": feature_names,
        "timestamp_column": ts_result.column,
        "split_summary": split.summary(),
        "leakage_check": {
            "passed": leakage_check.passed,
            "message": leakage_check.message,
        },
    }
    metadata_path = processed_dir / "split_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str))
    logger.info("Wrote split metadata -> %s", metadata_path)

    return metadata


def main() -> int:
    args = build_arg_parser().parse_args()
    logger = setup_logging()

    parts_dir = args.processed_dir / "parts"
    if args.overwrite_parts and parts_dir.exists():
        shutil.rmtree(parts_dir)

    try:
        label_col, feature_names = pass_one_clean_chunks(
            args.raw_dir, parts_dir, args.chunk_size, logger
        )
        pass_two_split(
            parts_dir,
            args.processed_dir,
            label_col,
            feature_names,
            args.timestamp_column,
            args.test_size,
            logger,
        )
    except (FileNotFoundError, ValueError, RuntimeError, KeyError) as exc:
        logger.error("prepare_data failed: %s", exc)
        return 1

    logger.info("Day 1 data preparation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
