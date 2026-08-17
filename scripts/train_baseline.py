#!/usr/bin/env python
"""
Day 1 - Step 3: Lightweight Random Forest baseline + evaluation.

Trains a single, deliberately modest RandomForestClassifier on the
chronologically-split train set produced by scripts/prepare_data.py, and
evaluates it on the (strictly later) test set.

No hyperparameter tuning, no cross-validation grid search, no SHAP/
explainability -- those are out of Day 1 scope. This is a baseline only.

Metrics computed (binary label_binary: 0=benign, 1=attack):
    * Precision, Recall, F1 (positive class = attack = 1)
    * ROC-AUC, PR-AUC (using predicted probabilities)
    * False Positive Rate (FPR), False Negative Rate (FNR)

Outputs:
    models/day1/random_forest_baseline.joblib
    models/day1/random_forest_baseline_metadata.json
    results/day1/baseline_metrics.json

This script is NOT run automatically. Run it yourself after
scripts/prepare_data.py has produced data/processed/day1/{train,test}.parquet.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

LOG_DIR = PROJECT_ROOT / "logs"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-dir", type=Path, default=PROJECT_ROOT / "data" / "processed" / "day1"
    )
    parser.add_argument("--models-dir", type=Path, default=PROJECT_ROOT / "models" / "day1")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results" / "day1")
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Kept modest by default for CPU-only, 8GB RAM machines.",
    )
    parser.add_argument("--max-depth", type=int, default=20)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_DIR / "train_baseline.log"),
        ],
    )
    return logging.getLogger("train_baseline")


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    from sklearn.metrics import (
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
        average_precision_score,
    )

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        roc_auc = roc_auc_score(y_true, y_proba)
    except ValueError as exc:
        roc_auc = float("nan")
        logging.getLogger("train_baseline").warning(
            "ROC-AUC could not be computed (%s). Likely only one class "
            "present in y_true for this split.",
            exc,
        )

    try:
        pr_auc = average_precision_score(y_true, y_proba)
    except ValueError:
        pr_auc = float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    fnr = fn / (fn + tp) if (fn + tp) > 0 else float("nan")

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fnr),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def main() -> int:
    args = build_arg_parser().parse_args()
    logger = setup_logging()

    import joblib
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier

    train_path = args.processed_dir / "train.parquet"
    test_path = args.processed_dir / "test.parquet"
    metadata_path = args.processed_dir / "split_metadata.json"

    for path in (train_path, test_path, metadata_path):
        if not path.exists():
            logger.error(
                "%s not found. Run scripts/prepare_data.py first.", path
            )
            return 1

    metadata = json.loads(metadata_path.read_text())
    feature_names = metadata["feature_names"]

    logger.info("Loading train/test parquet splits...")
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    missing_train = [c for c in feature_names if c not in train_df.columns]
    missing_test = [c for c in feature_names if c not in test_df.columns]
    if missing_train or missing_test:
        logger.error(
            "Feature columns from metadata missing at train/test time. "
            "train missing=%s, test missing=%s",
            missing_train,
            missing_test,
        )
        return 1

    X_train = train_df[feature_names]
    y_train = train_df["label_binary"]
    X_test = test_df[feature_names]
    y_test = test_df["label_binary"]

    # NaNs are intentionally left as NaN through cleaning (see cleaner.py);
    # RandomForestClassifier cannot handle NaN, so impute here at the
    # model-boundary with a simple, logged strategy (median), rather than
    # inside the generic cleaner.
    n_nan_train = int(X_train.isna().to_numpy().sum())
    n_nan_test = int(X_test.isna().to_numpy().sum())
    if n_nan_train or n_nan_test:
        logger.info(
            "Imputing NaNs with train-set column medians before fitting "
            "(train NaNs=%d, test NaNs=%d).",
            n_nan_train,
            n_nan_test,
        )
        medians = X_train.median(numeric_only=True)
        X_train = X_train.fillna(medians)
        X_test = X_test.fillna(medians)

    logger.info(
        "Training RandomForestClassifier: n_estimators=%d, max_depth=%s, "
        "n_jobs=%d, random_state=%d on %d rows x %d features",
        args.n_estimators,
        args.max_depth,
        args.n_jobs,
        args.random_state,
        len(X_train),
        len(feature_names),
    )

    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        n_jobs=args.n_jobs,
        random_state=args.random_state,
        class_weight="balanced",
    )

    start = time.time()
    clf.fit(X_train, y_train)
    train_seconds = time.time() - start
    logger.info("Training complete in %.1fs", train_seconds)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_proba)
    logger.info("Evaluation metrics:\n%s", json.dumps(metrics, indent=2))

    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    model_path = args.models_dir / "random_forest_baseline.joblib"
    joblib.dump(clf, model_path)
    logger.info("Saved model -> %s", model_path)

    model_metadata = {
        "model_type": "RandomForestClassifier",
        "hyperparameters": {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "random_state": args.random_state,
            "class_weight": "balanced",
        },
        "feature_names": feature_names,
        "label_col": "label_binary",
        "timestamp_column": metadata.get("timestamp_column"),
        "split_summary": metadata.get("split_summary"),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "train_seconds": train_seconds,
        "nan_imputation": {
            "strategy": "train-median",
            "train_nans_before_impute": n_nan_train,
            "test_nans_before_impute": n_nan_test,
        },
    }
    model_metadata_path = args.models_dir / "random_forest_baseline_metadata.json"
    model_metadata_path.write_text(json.dumps(model_metadata, indent=2, default=str))
    logger.info("Saved model metadata -> %s", model_metadata_path)

    results_path = args.results_dir / "baseline_metrics.json"
    results_path.write_text(json.dumps(metrics, indent=2, default=str))
    logger.info("Saved evaluation metrics -> %s", results_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
