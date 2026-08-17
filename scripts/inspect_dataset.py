#!/usr/bin/env python
"""
Day 1 - Step 1: Dataset discovery + lightweight schema/NaN-Inf inspection.

Does NOT train anything and does NOT process the full dataset in one shot.
Reads only:
    * the file listing under data/raw/CIC-IDS2017/
    * the first chunk (default: 5 rows) for a schema peek
    * one full chunk (default: chunk-size rows) for a NaN/Inf spot check

Usage
-----
    python scripts/prepare_data.py --raw-dir data/raw/CIC-IDS2017 ...  (later)
    python scripts/inspect_dataset.py \
        --raw-dir data/raw/CIC-IDS2017 \
        --chunk-size 250000

This script is intentionally NOT run automatically. Run it yourself after
placing the real dataset under data/raw/CIC-IDS2017/.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import (  # noqa: E402
    DEFAULT_CHUNK_SIZE,
    discover_dataset_files,
    iter_dataset_chunks,
    peek_schema,
)
from src.data.validator import validate_nan_inf, validate_schema  # noqa: E402

LOG_DIR = PROJECT_ROOT / "logs"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "CIC-IDS2017",
        help="Root directory containing parquet/ and/or csv/ subfolders.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Rows per chunk for the NaN/Inf spot check (default: %(default)s).",
    )
    parser.add_argument(
        "--peek-rows",
        type=int,
        default=5,
        help="Rows to show in the schema peek (default: %(default)s).",
    )
    return parser


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_DIR / "inspect_dataset.log"),
        ],
    )


def main() -> int:
    args = build_arg_parser().parse_args()
    setup_logging()
    logger = logging.getLogger("inspect_dataset")

    logger.info("Discovering dataset files under: %s", args.raw_dir)
    try:
        dataset = discover_dataset_files(args.raw_dir)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return 1

    if dataset.is_empty:
        logger.error(
            "No parquet or csv files found. Place CIC-IDS2017 files under "
            "%s before running this script.",
            args.raw_dir,
        )
        return 1

    logger.info("Preferred format: %s", dataset.preferred_format)
    logger.info("Parquet files (%d): %s", len(dataset.parquet_files), dataset.parquet_files)
    logger.info("CSV files (%d): %s", len(dataset.csv_files), dataset.csv_files)

    logger.info("Peeking schema (%d rows)...", args.peek_rows)
    sample = peek_schema(args.raw_dir, n_rows=args.peek_rows)
    if sample.empty:
        logger.error("Could not read any rows from the discovered files.")
        return 1

    schema_report = validate_schema(sample)
    logger.info("Schema report:\n%s", json.dumps(schema_report.to_dict(), indent=2, default=str))

    logger.info("Reading one full chunk (size=%d) for a NaN/Inf spot check...", args.chunk_size)
    first_chunk = next(iter_dataset_chunks(args.raw_dir, chunk_size=args.chunk_size))
    nan_inf_report = validate_nan_inf(first_chunk)
    logger.info(
        "NaN/Inf spot check (first %d-row chunk):\n%s",
        len(first_chunk),
        json.dumps(nan_inf_report.to_dict(), indent=2, default=str),
    )

    report_path = LOG_DIR / "inspect_dataset_report.json"
    report_path.write_text(
        json.dumps(
            {
                "raw_dir": str(args.raw_dir),
                "preferred_format": dataset.preferred_format,
                "n_parquet_files": len(dataset.parquet_files),
                "n_csv_files": len(dataset.csv_files),
                "schema": schema_report.to_dict(),
                "nan_inf_spot_check": nan_inf_report.to_dict(),
            },
            indent=2,
            default=str,
        )
    )
    logger.info("Wrote report to %s", report_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
