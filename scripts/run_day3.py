#!/usr/bin/env python
"""
Day 3 - End-to-end script: zero-day / unseen-attack-class detection
analysis.

Central question: "Can an anomaly-based or hybrid detector identify
attack behavior belonging to an attack class that was not available to
the supervised classifier during training?"

Reuses the Day 1 and Day 2 artifacts AS-IS -- nothing is retrained or
refit here:
    data/processed/day1/train.parquet             (Monday-Thursday, full)
    data/processed/day1/test.parquet               (Friday, frozen)
    data/processed/day1/split_metadata.json
    models/day1/random_forest_baseline.joblib      (Day 1 RF, loaded only)
    models/day2/isolation_forest.joblib            (Day 2 IsolationForest, loaded only)
    results/day2/validation_threshold_selection.json (frozen IF/Hybrid thresholds)

Friday is NOT automatically "zero-day": Day 3 explicitly determines
which Friday attack classes are absent from the FULL Monday-Thursday
supervised training period (see src.day3.zero_day.class_inventory) and
analyzes those separately from classes the Random Forest already saw.

No threshold is selected or tuned in this script. All three detectors
(Random Forest, IsolationForest, Hybrid) are evaluated at their
already-frozen Day 1/Day 2 thresholds. Unseen-attack rows are never
used for threshold selection -- this script does not import or call
select_threshold_from_validation at all.

Outputs:
    results/day3/zero_day_summary.json
    results/day3/zero_day_comparison.csv
    results/day3/zero_day_per_class.csv
    results/day3/zero_day_detection_rates.csv
    results/day3/day3_metadata.json

Does NOT write to data/, models/day1/, models/day2/, or results/day2/.

This script is NOT run automatically -- run it yourself, from the
project root, after Day 1's and Day 2's scripts have produced their
artifacts:

    python scripts/run_day3.py
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

# Fallback frozen IsolationForest / Hybrid thresholds, used ONLY if
# results/day2/validation_threshold_selection.json is not found. These
# mirror the already-established Day 2 frozen values and are NOT
# re-derived here (this script performs no threshold selection).
FALLBACK_IF_THRESHOLD = 0.15
FALLBACK_HYBRID_THRESHOLD = 0.50


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
        "--day2-results-dir", type=Path, default=PROJECT_ROOT / "results" / "day2"
    )
    parser.add_argument(
        "--results-dir", type=Path, default=PROJECT_ROOT / "results" / "day3"
    )
    parser.add_argument(
        "--if-threshold", type=float, default=None,
        help="Override the frozen IsolationForest threshold instead of "
             "reading it from results/day2/validation_threshold_selection.json.",
    )
    parser.add_argument(
        "--hybrid-threshold", type=float, default=None,
        help="Override the frozen Hybrid threshold instead of reading it "
             "from results/day2/validation_threshold_selection.json.",
    )
    return parser


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_DIR / "run_day3.log"),
        ],
    )
    return logging.getLogger("run_day3")


def main() -> int:
    args = build_arg_parser().parse_args()
    logger = setup_logging()

    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.utils.validation import check_is_fitted

    from src.data.validator import validate_nan_inf
    from src.day2.anomaly import AnomalyModel, anomaly_scores
    from src.day2.hybrid import HybridConfig, combine_scores, hybrid_predict
    from src.day2.thresholding import (
        RF_DAY1_FROZEN_THRESHOLD,
        evaluate_frozen_threshold,
        per_class_frozen_results,
    )
    from src.day3.zero_day import (
        assert_feature_alignment,
        assert_no_unseen_leakage,
        build_population,
        class_inventory,
    )

    # -----------------------------------------------------------------
    # Locate and validate required artifacts -- STOP if anything needed
    # for a defensible zero-day experiment is missing, rather than
    # fabricating one.
    # -----------------------------------------------------------------
    train_path = args.processed_dir / "train.parquet"
    test_path = args.processed_dir / "test.parquet"
    metadata_path = args.processed_dir / "split_metadata.json"
    rf_model_path = args.day1_models_dir / "random_forest_baseline.joblib"
    if_model_path = args.day2_models_dir / "isolation_forest.joblib"
    day2_threshold_path = args.day2_results_dir / "validation_threshold_selection.json"
    day2_metadata_path = args.day2_models_dir / "day2_metadata.json"

    required = [train_path, test_path, metadata_path, rf_model_path, if_model_path]
    missing = [p for p in required if not p.exists()]
    if missing:
        logger.error(
            "Day 3 requires existing Day 1 and Day 2 artifacts. Missing: %s. "
            "Run Day 1's scripts/prepare_data.py + scripts/train_baseline.py "
            "and Day 2's scripts/run_day2.py first. STOPPING rather than "
            "fabricating a zero-day experiment.",
            [str(p) for p in missing],
        )
        return 1

    metadata = json.loads(metadata_path.read_text())
    feature_names = metadata["feature_names"]

    logger.info("Loading Day 1 train/test parquet, Day 1 RF, and Day 2 IsolationForest...")
    train_df = pd.read_parquet(train_path)   # Monday-Thursday, full supervised training period
    test_df = pd.read_parquet(test_path)     # Friday -- frozen, read-only
    rf_model = joblib.load(rf_model_path)
    if_raw_model = joblib.load(if_model_path)

    # -----------------------------------------------------------------
    # Research-integrity check: models are pre-fitted (loaded), never
    # fit here. check_is_fitted raises on an unfitted estimator, so
    # this also proves Friday could not have been used to fit them in
    # this script (no .fit() call exists below, and these checks run
    # before test_df is ever passed to either model).
    # -----------------------------------------------------------------
    check_is_fitted(rf_model)
    check_is_fitted(if_raw_model)
    logger.info("Integrity check passed: RF and IsolationForest are pre-fitted, loaded models.")

    # Research-integrity check: evaluation features match what the RF
    # was actually fit on.
    assert_feature_alignment(feature_names, rf_model, context="Random Forest baseline")
    assert_feature_alignment(feature_names, if_raw_model, context="IsolationForest")
    logger.info("Integrity check passed: feature columns match the fitted models.")

    # -----------------------------------------------------------------
    # Class inventory -- computed from the FULL Monday-Thursday
    # training period (what the RF actually saw), not Day 2's smaller
    # internal validation sub-train.
    # -----------------------------------------------------------------
    inventory = class_inventory(train_df, test_df)
    logger.info("Class inventory: %s", inventory.summary())

    assert_no_unseen_leakage(train_df, inventory.unseen_attack_classes)
    logger.info(
        "Integrity check passed: no unseen-flagged class appears in training data."
    )

    if not inventory.unseen_attack_classes:
        logger.error(
            "No Friday attack class is absent from the Monday-Thursday "
            "training period -- there is no genuine zero-day population "
            "to evaluate. STOPPING rather than fabricating one."
        )
        return 1

    # -----------------------------------------------------------------
    # NaN/Inf check on the raw evaluation data (reused from Day 1's
    # validator, not reimplemented).
    # -----------------------------------------------------------------
    nan_inf_report = validate_nan_inf(test_df[feature_names])
    logger.info(
        "Friday NaN/Inf check: has_nan=%s, has_inf=%s (imputed with training "
        "medians below, same strategy as Day 1/Day 2).",
        nan_inf_report.has_nan, nan_inf_report.has_inf,
    )

    # -----------------------------------------------------------------
    # Reconstruct the Day 2 anomaly-scoring wrapper around the loaded,
    # already-fitted IsolationForest. Training medians are recomputed
    # from the full Monday-Thursday training data (the same strategy
    # Day 1's RF baseline and Day 2's imputation already use) -- Day
    # 2's exact anomaly-training-SAMPLE medians were not persisted to
    # disk, and this script does not touch or refit Day 2's sampling.
    # NaNs are rare (Day 1: 124/1.8M train rows), so this is a
    # negligible, documented approximation, not a methodology change.
    # -----------------------------------------------------------------
    train_medians = train_df[feature_names].median(numeric_only=True)

    if_config = {}
    if day2_metadata_path.exists():
        day2_metadata = json.loads(day2_metadata_path.read_text())
        if_config = day2_metadata.get("isolation_forest_config", {})

    anomaly_model = AnomalyModel(
        model=if_raw_model,
        feature_names=list(feature_names),
        train_medians=train_medians,
        contamination=if_config.get("contamination", if_raw_model.get_params().get("contamination")),
        n_estimators=if_config.get("n_estimators", if_raw_model.get_params().get("n_estimators")),
        random_state=if_config.get("random_state", if_raw_model.get_params().get("random_state")),
        n_training_samples=if_config.get("n_training_samples", -1),
    )

    # -----------------------------------------------------------------
    # Frozen thresholds -- read from Day 2's saved artifacts where
    # possible. NOT re-selected here (this script never touches
    # select_threshold_from_validation).
    # -----------------------------------------------------------------
    rf_threshold = RF_DAY1_FROZEN_THRESHOLD

    if_threshold = args.if_threshold
    hybrid_threshold = args.hybrid_threshold
    threshold_source = "cli_override"

    if if_threshold is None or hybrid_threshold is None:
        if day2_threshold_path.exists():
            day2_thresholds = json.loads(day2_threshold_path.read_text())
            if if_threshold is None:
                if_threshold = day2_thresholds["isolation_forest"]["selected_threshold"]
            if hybrid_threshold is None:
                hybrid_threshold = day2_thresholds["hybrid"]["selected_threshold"]
            threshold_source = str(day2_threshold_path)
        else:
            logger.warning(
                "%s not found; falling back to documented Day 2 status "
                "thresholds (IsolationForest=%.2f, Hybrid=%.2f) instead of "
                "re-selecting anything here.",
                day2_threshold_path, FALLBACK_IF_THRESHOLD, FALLBACK_HYBRID_THRESHOLD,
            )
            if if_threshold is None:
                if_threshold = FALLBACK_IF_THRESHOLD
            if hybrid_threshold is None:
                hybrid_threshold = FALLBACK_HYBRID_THRESHOLD
            threshold_source = "fallback_default"

    logger.info(
        "Frozen thresholds in use -- RF=%.4f (Day 1), IsolationForest=%.4f, "
        "Hybrid=%.4f (source: %s). None selected/tuned in this script.",
        rf_threshold, if_threshold, hybrid_threshold, threshold_source,
    )

    hybrid_config = HybridConfig(threshold=hybrid_threshold)  # rf_weight/anomaly_weight left at Day 2 defaults (0.7/0.3)

    # -----------------------------------------------------------------
    # Score Friday ONCE with each detector.
    # -----------------------------------------------------------------
    X_test = test_df[feature_names].fillna(train_medians)
    assert not X_test.isna().any().any(), "NaNs remain after imputation -- aborting."
    assert not np.isinf(X_test.to_numpy()).any(), "Inf values present after imputation -- aborting."

    y_test = test_df["label_binary"].astype(int)
    test_proba = rf_model.predict_proba(X_test)[:, 1]
    test_anomaly = anomaly_scores(anomaly_model, test_df)
    test_hybrid = combine_scores(test_proba, test_anomaly, hybrid_config)

    test_pred_rf = (test_proba >= rf_threshold).astype(int)
    test_pred_if = (test_anomaly >= if_threshold).astype(int)
    test_pred_hybrid = hybrid_predict(test_hybrid, hybrid_threshold)

    # -----------------------------------------------------------------
    # Populations -- kept strictly separate, never mixed.
    #   A. all_friday               -- full Friday (seen + unseen + benign)
    #   B. unseen_attacks_vs_benign -- unseen classes + benign only
    #   C. one row per unseen class -- that class + benign only
    # -----------------------------------------------------------------
    populations: dict[str, pd.DataFrame] = {"all_friday": test_df}
    populations["unseen_attacks_vs_benign"] = build_population(
        test_df, inventory.unseen_attack_classes, inventory.benign_label
    )
    for cls in inventory.unseen_attack_classes:
        populations[f"unseen__{cls}"] = build_population(
            test_df, [cls], inventory.benign_label
        )

    detector_scores = {
        "random_forest": (test_proba, rf_threshold),
        "isolation_forest": (test_anomaly, if_threshold),
        "hybrid": (test_hybrid, hybrid_threshold),
    }

    comparison_rows = []
    for pop_name, pop_df in populations.items():
        pop_positions = test_df.index.get_indexer(pop_df.index)
        pop_y = y_test.to_numpy()[pop_positions]
        for detector_name, (scores, threshold) in detector_scores.items():
            pop_scores = np.asarray(scores)[pop_positions]
            metrics = evaluate_frozen_threshold(pop_y, pop_scores, threshold)
            comparison_rows.append(
                {
                    "detector": detector_name,
                    "population": pop_name,
                    "n_samples": len(pop_df),
                    "threshold_used": threshold,
                    **metrics,
                }
            )
    comparison = pd.DataFrame(comparison_rows)

    # -----------------------------------------------------------------
    # Per-class detection-rate table -- unseen classes ONLY, one row
    # per class per detector, no benign rows mixed in (recall-only
    # population by construction -- precision/FPR are not meaningful
    # here and are intentionally not reported in this table).
    # -----------------------------------------------------------------
    unseen_mask = test_df["label_multiclass"].isin(inventory.unseen_attack_classes)
    unseen_classes_series = test_df.loc[unseen_mask, "label_multiclass"]

    per_class_tables = {}
    for detector_name, (scores, threshold) in detector_scores.items():
        scores_arr = np.asarray(scores)[unseen_mask.to_numpy()]
        preds_arr = (scores_arr >= threshold).astype(int)
        table = per_class_frozen_results(scores_arr, preds_arr, unseen_classes_series)
        table["pct_detected"] = table["detection_rate"] * 100.0
        table["pct_missed"] = (1 - table["detection_rate"]) * 100.0
        per_class_tables[detector_name] = table

    # Wide comparison table: attack_class | samples | RF_detected | RF_detection_rate | ...
    per_class_wide = per_class_tables["random_forest"][["class", "samples", "predicted_attack", "detection_rate"]].rename(
        columns={"predicted_attack": "RF_detected", "detection_rate": "RF_detection_rate", "class": "attack_class"}
    )
    for detector_name, prefix in [("isolation_forest", "IF"), ("hybrid", "hybrid")]:
        t = per_class_tables[detector_name][["class", "predicted_attack", "detection_rate"]].rename(
            columns={
                "predicted_attack": f"{prefix}_detected",
                "detection_rate": f"{prefix}_detection_rate",
                "class": "attack_class",
            }
        )
        per_class_wide = per_class_wide.merge(t, on="attack_class", how="left")

    # Long format (class, detector, samples, detected, detection_rate,
    # pct_detected, pct_missed) -- convenient for plotting.
    detection_rates_long = pd.concat(
        [
            table.assign(detector=detector_name)[
                ["detector", "class", "samples", "predicted_attack", "detection_rate", "pct_detected", "pct_missed"]
            ]
            for detector_name, table in per_class_tables.items()
        ],
        ignore_index=True,
    ).rename(columns={"class": "attack_class", "predicted_attack": "detected"})

    # -----------------------------------------------------------------
    # Automatic, honest research interpretation -- rule-based on the
    # actual numbers, no forced positive framing.
    # -----------------------------------------------------------------
    def _recall_for(detector_name: str) -> float:
        row = comparison[
            (comparison["detector"] == detector_name)
            & (comparison["population"] == "unseen_attacks_vs_benign")
        ]
        return float(row["recall"].iloc[0]) if len(row) else float("nan")

    rf_recall = _recall_for("random_forest")
    if_recall = _recall_for("isolation_forest")
    hybrid_recall = _recall_for("hybrid")

    interpretation_lines = [
        f"Unseen attack classes identified: {inventory.unseen_attack_classes} "
        f"(absent from the full Monday-Thursday training period).",
        f"Random Forest (binary, frozen threshold={rf_threshold}) recall on "
        f"unseen attacks + benign: {rf_recall:.4f}.",
        f"IsolationForest (frozen threshold={if_threshold}) recall on unseen "
        f"attacks + benign: {if_recall:.4f}.",
        f"Hybrid (frozen threshold={hybrid_threshold}) recall on unseen "
        f"attacks + benign: {hybrid_recall:.4f}.",
    ]

    if not any(np.isnan([rf_recall, if_recall, hybrid_recall])):
        best_name, best_recall = max(
            [("Random Forest", rf_recall), ("IsolationForest", if_recall), ("Hybrid", hybrid_recall)],
            key=lambda pair: pair[1],
        )
        interpretation_lines.append(f"Highest unseen-attack recall: {best_name} ({best_recall:.4f}).")

        if if_recall > rf_recall:
            interpretation_lines.append(
                "The anomaly detector (IsolationForest) detects unseen attack "
                "traffic at a higher rate than the binary Random Forest here, "
                "consistent with the hypothesis that anomaly-based detection "
                "can generalize better to attack behavior absent from "
                "supervised training."
            )
        else:
            interpretation_lines.append(
                "The Random Forest's binary attack/benign signal still detects "
                "unseen attack traffic at least as well as IsolationForest here "
                "-- i.e. binary 'is this attack-like' generalization from RF "
                "extends to some unseen classes, even though RF cannot name "
                "them (see multiclass caveat below)."
            )

        if hybrid_recall < min(rf_recall, if_recall) - 0.05:
            interpretation_lines.append(
                "The hybrid detector performs worse than BOTH individual "
                "detectors on unseen attacks (not just worse than one). This "
                "is a negative result and is reported as-is: at its current "
                "frozen threshold and 0.7/0.3 weighting, combining the two "
                "signals does not help, and may hurt, zero-day detection -- "
                "consistent with the RF-dominated hybrid formula inheriting "
                "the RF's degraded confidence on temporally-shifted traffic "
                "(see Day 2 findings)."
            )
        elif hybrid_recall > max(rf_recall, if_recall):
            interpretation_lines.append(
                "The hybrid detector improves unseen-attack recall over both "
                "individual detectors here."
            )
        else:
            interpretation_lines.append(
                "The hybrid detector's unseen-attack recall falls between the "
                "two individual detectors -- it neither clearly helps nor "
                "clearly fails here."
            )

    interpretation_lines.append(
        "Multiclass caveat: the Random Forest was trained with a BINARY "
        "(benign/attack) label, so even if it flags unseen-class traffic as "
        "'attack', it was never trained to name that class -- multiclass "
        "identification of a truly unseen class is not possible for a "
        "classifier that never saw any example of it, by definition."
    )

    interpretation_text = "\n".join(f"- {line}" for line in interpretation_lines)
    logger.info("Research interpretation:\n%s", interpretation_text)
    print("\n=== Day 3 Research Interpretation ===\n" + interpretation_text + "\n")

    # -----------------------------------------------------------------
    # Write outputs (results/day3 only -- never results/day2, models/day1,
    # models/day2, or data/).
    # -----------------------------------------------------------------
    args.results_dir.mkdir(parents=True, exist_ok=True)

    zero_day_summary = {
        "class_inventory": inventory.summary(),
        "thresholds_used": {
            "random_forest": rf_threshold,
            "isolation_forest": if_threshold,
            "hybrid": hybrid_threshold,
            "source": threshold_source,
            "note": "All thresholds reused from Day 1/Day 2 frozen values; none selected or tuned in Day 3.",
        },
        "hybrid_config": hybrid_config.summary(),
        "nan_inf_report_friday_raw": nan_inf_report.to_dict(),
        "integrity_checks_passed": [
            "rf_and_if_pre_fitted_before_test_access",
            "feature_alignment_rf",
            "feature_alignment_isolation_forest",
            "no_unseen_class_in_training_data",
            "no_nan_or_inf_after_imputation",
        ],
        "research_interpretation": interpretation_lines,
    }
    (args.results_dir / "zero_day_summary.json").write_text(
        json.dumps(zero_day_summary, indent=2, default=str)
    )
    comparison.to_csv(args.results_dir / "zero_day_comparison.csv", index=False)
    per_class_wide.to_csv(args.results_dir / "zero_day_per_class.csv", index=False)
    detection_rates_long.to_csv(args.results_dir / "zero_day_detection_rates.csv", index=False)

    day3_metadata = {
        "inputs": {
            "train_path": str(train_path),
            "test_path": str(test_path),
            "rf_model_path": str(rf_model_path),
            "if_model_path": str(if_model_path),
            "day2_threshold_source": threshold_source,
        },
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "class_inventory": inventory.summary(),
        "populations_evaluated": list(populations.keys()),
    }
    (args.results_dir / "day3_metadata.json").write_text(
        json.dumps(day3_metadata, indent=2, default=str)
    )

    logger.info("Day 3 outputs written to: %s", args.results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
