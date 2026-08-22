#!/usr/bin/env python
"""
Day 2 - End-to-end script: validation -> threshold selection -> frozen
Friday evaluation -> unseen-attack analysis -> IsolationForest ->
hybrid detector -> alert-structure demo.

Reuses the Day 1 artifacts as-is:
    data/processed/day1/train.parquet   (Monday-Thursday)
    data/processed/day1/test.parquet    (Friday, frozen)
    data/processed/day1/split_metadata.json
    models/day1/random_forest_baseline.joblib

Does NOT retrain or modify the Day 1 Random Forest baseline.

Outputs:
    results/day2/validation_threshold_selection.json   (RF/IsolationForest/Hybrid, each selected independently on validation only)
    results/day2/validation_threshold_grid_rf.csv
    results/day2/validation_threshold_grid_isolation_forest.csv
    results/day2/validation_threshold_grid_hybrid.csv
    results/day2/frozen_friday_metrics.json             (Random Forest, frozen at Day 1's threshold: RF_DAY1_FROZEN_THRESHOLD = 0.01)
    results/day2/frozen_friday_per_class.csv
    results/day2/isolation_forest_config.json
    results/day2/comparison_table.csv
    results/day2/sample_alerts.json
    models/day2/isolation_forest.joblib
    models/day2/day2_metadata.json

This script is NOT run automatically -- run it yourself after Day 1's
scripts/prepare_data.py and scripts/train_baseline.py have produced
their artifacts.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

LOG_DIR = PROJECT_ROOT / "logs"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-dir", type=Path, default=PROJECT_ROOT / "data" / "processed" / "day1"
    )
    parser.add_argument(
        "--day1-models-dir", type=Path, default=PROJECT_ROOT / "models" / "day1"
    )
    parser.add_argument(
        "--day2-models-dir", type=Path, default=PROJECT_ROOT / "models" / "day2"
    )
    parser.add_argument(
        "--results-dir", type=Path, default=PROJECT_ROOT / "results" / "day2"
    )
    parser.add_argument(
        "--anomaly-n-samples", type=int, default=100_000,
        help="Rows sampled from the training set to fit IsolationForest.",
    )
    parser.add_argument("--anomaly-contamination", type=float, default=0.05)
    parser.add_argument("--anomaly-benign-fraction", type=float, default=0.9)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--hybrid-rf-weight", type=float, default=0.7)
    parser.add_argument("--hybrid-anomaly-weight", type=float, default=0.3)
    parser.add_argument("--n-sample-alerts", type=int, default=5)
    return parser


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_DIR / "run_day2.log"),
        ],
    )
    return logging.getLogger("run_day2")


def main() -> int:
    args = build_arg_parser().parse_args()
    logger = setup_logging()

    import joblib
    import pandas as pd

    from src.day2.alerts import build_alert
    from src.day2.anomaly import (
        anomaly_scores,
        fit_isolation_forest,
        sample_training_data,
    )
    from src.day2.hybrid import HybridConfig, combine_scores, hybrid_predict
    from src.day2.thresholding import (
        RF_DAY1_FROZEN_THRESHOLD,
        evaluate_frozen_threshold,
        identify_unseen_classes,
        per_class_frozen_results,
        select_threshold_from_validation,
    )
    from src.day2.validation import build_validation_split

    train_path = args.processed_dir / "train.parquet"
    test_path = args.processed_dir / "test.parquet"
    metadata_path = args.processed_dir / "split_metadata.json"
    rf_model_path = args.day1_models_dir / "random_forest_baseline.joblib"

    for path in (train_path, test_path, metadata_path, rf_model_path):
        if not path.exists():
            logger.error(
                "%s not found. Run Day 1's scripts/prepare_data.py and "
                "scripts/train_baseline.py first.",
                path,
            )
            return 1

    metadata = json.loads(metadata_path.read_text())
    feature_names = metadata["feature_names"]

    logger.info("Loading Day 1 train/test parquet and Random Forest baseline...")
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)  # Friday -- frozen, only read once below
    rf_model = joblib.load(rf_model_path)

    # -----------------------------------------------------------------
    # Step 1: Validation split (never touches Friday / test_df)
    # -----------------------------------------------------------------
    val_split = build_validation_split(train_df)
    logger.info("Validation split summary: %s", val_split.summary())

    def _impute(df: pd.DataFrame, medians: pd.Series) -> pd.DataFrame:
        return df[feature_names].fillna(medians)

    train_medians = train_df[feature_names].median(numeric_only=True)

    X_val = _impute(val_split.val_df, train_medians)
    y_val = val_split.val_df["label_binary"].astype(int)

    val_proba = rf_model.predict_proba(X_val)[:, 1]

    # -----------------------------------------------------------------
    # Step 2: Threshold selection -- VALIDATION ONLY
    # -----------------------------------------------------------------
    # Day 2's own validation-based selection for the RF, computed and
    # reported for transparency/comparison. It is NOT what gets frozen
    # and applied below -- see rf_threshold immediately after.
    rf_validation_selection = select_threshold_from_validation(y_val, val_proba)
    logger.info(
        "Day 2 validation-selected RF threshold=%.4f (F1=%.4f) -- "
        "reported for comparison only.",
        rf_validation_selection.threshold,
        rf_validation_selection.validation_metrics["f1"],
    )

    # The Random Forest's frozen, reported production threshold is
    # Day 1's already-established value (best tested F1 in Day 1's
    # threshold sweep), not Day 2's validation grid pick and not the
    # 0.5 library default.
    rf_threshold = RF_DAY1_FROZEN_THRESHOLD
    logger.info("Using RF_DAY1_FROZEN_THRESHOLD=%.4f for RF Friday evaluation.", rf_threshold)

    # -----------------------------------------------------------------
    # Step 3: Frozen Friday evaluation -- threshold applied unchanged
    # -----------------------------------------------------------------
    X_test = _impute(test_df, train_medians)
    y_test = test_df["label_binary"].astype(int)
    test_proba = rf_model.predict_proba(X_test)[:, 1]

    frozen_metrics = evaluate_frozen_threshold(
        y_test, test_proba, rf_threshold
    )
    logger.info("Frozen Friday metrics (RF @ %.4f): %s", rf_threshold, frozen_metrics)

    test_pred = (test_proba >= rf_threshold).astype(int)

    per_class = None
    if "label_multiclass" in test_df.columns:
        per_class = per_class_frozen_results(
            test_proba, test_pred, test_df["label_multiclass"]
        )

        # Step 4: unseen-attack analysis
        train_classes = (
            val_split.subtrain_df["label_multiclass"].unique()
            if "label_multiclass" in val_split.subtrain_df.columns
            else train_df.get("label_multiclass", pd.Series(dtype=str)).unique()
        )
        unseen = identify_unseen_classes(train_classes, test_df["label_multiclass"].unique())
        logger.info("Friday classes absent from training: %s", unseen)

    # -----------------------------------------------------------------
    # Step 5: Lightweight anomaly detector (IsolationForest)
    # -----------------------------------------------------------------
    anomaly_sample = sample_training_data(
        train_df,
        feature_names,
        n_samples=args.anomaly_n_samples,
        benign_fraction=args.anomaly_benign_fraction,
        random_state=args.random_state,
    )
    anomaly_model = fit_isolation_forest(
        anomaly_sample,
        feature_names,
        contamination=args.anomaly_contamination,
        random_state=args.random_state,
    )

    val_anomaly = anomaly_scores(anomaly_model, val_split.val_df)
    test_anomaly = anomaly_scores(anomaly_model, test_df)

    # IsolationForest threshold selection -- VALIDATION ONLY, on the
    # anomaly detector's own score scale (not the RF's). Small
    # predefined grid, same as Step 2; no hyperparameter search.
    anomaly_selection = select_threshold_from_validation(y_val, val_anomaly)
    logger.info(
        "Selected IsolationForest threshold=%.4f from validation (F1=%.4f)",
        anomaly_selection.threshold,
        anomaly_selection.validation_metrics["f1"],
    )

    frozen_anomaly_metrics = evaluate_frozen_threshold(
        y_test, test_anomaly, anomaly_selection.threshold
    )

    # -----------------------------------------------------------------
    # Step 6: Hybrid detector
    # -----------------------------------------------------------------
    hybrid_config = HybridConfig(
        rf_weight=args.hybrid_rf_weight,
        anomaly_weight=args.hybrid_anomaly_weight,
    )

    val_hybrid = combine_scores(val_proba, val_anomaly, hybrid_config)
    test_hybrid = combine_scores(test_proba, test_anomaly, hybrid_config)

    # Hybrid threshold selection -- VALIDATION ONLY, on the hybrid
    # score's own scale. Small predefined grid; frozen before Friday.
    hybrid_selection = select_threshold_from_validation(y_val, val_hybrid)
    hybrid_config.threshold = hybrid_selection.threshold
    logger.info(
        "Selected hybrid threshold=%.4f from validation (F1=%.4f)",
        hybrid_selection.threshold,
        hybrid_selection.validation_metrics["f1"],
    )

    test_hybrid_pred = hybrid_predict(test_hybrid, hybrid_config.threshold)

    frozen_hybrid_metrics = evaluate_frozen_threshold(
        y_test, test_hybrid, hybrid_config.threshold
    )

    comparison = pd.DataFrame(
        [
            {"detector": "random_forest", **frozen_metrics},
            {"detector": "isolation_forest", **frozen_anomaly_metrics},
            {"detector": "hybrid", **frozen_hybrid_metrics},
        ]
    )

    # -----------------------------------------------------------------
    # Step 7: Structured alert demo (no LLM call)
    # -----------------------------------------------------------------
    n_alerts = min(args.n_sample_alerts, int(test_hybrid_pred.sum()))
    alert_indices = test_df.index[test_hybrid_pred.astype(bool)][:n_alerts]

    sample_alerts = []
    for idx in alert_indices:
        pos = test_df.index.get_loc(idx)
        sample_alerts.append(
            build_alert(
                row=X_test.iloc[pos],
                feature_names=feature_names,
                rf_probability=float(test_proba[pos]),
                anomaly_score=float(test_anomaly[pos]),
                hybrid_score=float(test_hybrid[pos]),
                rf_threshold=rf_threshold,
                hybrid_threshold=hybrid_config.threshold,
                predicted_label=int(test_hybrid_pred[pos]),
            )
        )

    # -----------------------------------------------------------------
    # Write outputs
    # -----------------------------------------------------------------
    args.day2_models_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(anomaly_model.model, args.day2_models_dir / "isolation_forest.joblib")

    day2_metadata = {
        "validation_split": val_split.summary(),
        "random_forest": {
            "day1_frozen_threshold": rf_threshold,
            "day2_validation_selection": rf_validation_selection.summary(),
            "note": (
                "Friday evaluation uses the Day 1 frozen threshold "
                "(RF_DAY1_FROZEN_THRESHOLD), not the Day 2 "
                "validation-selected value above, which is reported "
                "for comparison only."
            ),
        },
        "isolation_forest_threshold_selection": anomaly_selection.summary(),
        "hybrid_threshold_selection": hybrid_selection.summary(),
        "isolation_forest_config": anomaly_model.config_summary(),
        "isolation_forest_sample": anomaly_sample.summary(),
        "hybrid_config": hybrid_config.summary(),
    }
    (args.day2_models_dir / "day2_metadata.json").write_text(
        json.dumps(day2_metadata, indent=2, default=str)
    )

    (args.results_dir / "validation_threshold_selection.json").write_text(
        json.dumps(
            {
                "random_forest": {
                    "day1_frozen_threshold_used": rf_threshold,
                    "day2_validation_selection": rf_validation_selection.summary(),
                },
                "isolation_forest": anomaly_selection.summary(),
                "hybrid": hybrid_selection.summary(),
            },
            indent=2,
            default=str,
        )
    )
    rf_validation_selection.grid.to_csv(
        args.results_dir / "validation_threshold_grid_rf.csv", index=False
    )
    anomaly_selection.grid.to_csv(
        args.results_dir / "validation_threshold_grid_isolation_forest.csv", index=False
    )
    hybrid_selection.grid.to_csv(
        args.results_dir / "validation_threshold_grid_hybrid.csv", index=False
    )
    (args.results_dir / "frozen_friday_metrics.json").write_text(
        json.dumps(frozen_metrics, indent=2, default=str)
    )
    if per_class is not None:
        per_class.to_csv(
            args.results_dir / "frozen_friday_per_class.csv", index=False
        )
    (args.results_dir / "isolation_forest_config.json").write_text(
        json.dumps(anomaly_model.config_summary(), indent=2, default=str)
    )
    comparison.to_csv(args.results_dir / "comparison_table.csv", index=False)
    (args.results_dir / "sample_alerts.json").write_text(
        json.dumps(sample_alerts, indent=2, default=str)
    )

    logger.info("Day 2 pipeline complete. Outputs written to %s", args.results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
