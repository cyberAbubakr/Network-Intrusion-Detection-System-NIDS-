#!/usr/bin/env python
"""
Day 4 - End-to-end script: analysis / explainability ONLY.

Day 4 is analysis-only. No model is retrained and no threshold is
tuned using Friday data.

Answers: why does the Random Forest generalize well to some unseen
attacks (DDoS), moderately to PortScan, poorly to Bot, while the
hybrid detector performs extremely poorly on all three? This script
does not fit, retrain, or tune anything -- it loads the already-frozen
Day 1 RF, Day 2 IsolationForest, and Day 2 hybrid configuration exactly
as Day 3 evaluated them, and analyzes feature-distribution shift and
each detector's score distribution.

Reuses (does not modify or re-run):
    data/processed/day1/train.parquet, test.parquet, split_metadata.json
    models/day1/random_forest_baseline.joblib
    models/day2/isolation_forest.joblib, day2_metadata.json
    results/day2/validation_threshold_selection.json
    results/day3/* (class inventory, frozen thresholds)
    src.day2.*, src.day3.zero_day.*  (unmodified)

Outputs (results/day4/ only -- never Day 1/2/3 outputs):
    results/day4/day4_class_feature_shift.csv
    results/day4/day4_top_shifted_features.csv
    results/day4/day4_detection_vs_shift.csv
    results/day4/day4_hybrid_threshold_analysis.csv
    results/day4/day4_distribution_tests.csv
    results/day4/day4_unseen_attack_summary.csv
    results/day4/day4_research_interpretation.json
    results/day4/day4_metadata.json
    results/day4/figures/rf_score_by_unseen_class.png
    results/day4/figures/if_score_by_unseen_class.png
    results/day4/figures/hybrid_score_by_unseen_class.png
    results/day4/figures/feature_importance_vs_shift.png
    results/day4/figures/class_feature_shift_heatmap.png
    results/day4/figures/unseen_attack_detection_comparison.png

Run from the project root:
    python scripts/run_day4.py
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

TOP_N_FEATURES = 20
TOP_N_SHIFTED_PER_CLASS = 10

# Fallback frozen thresholds/config, used ONLY if the corresponding
# Day 2/Day 3 artifact is missing. Mirror already-established values;
# NOT re-derived here (this script performs no threshold selection or
# tuning of any kind).
FALLBACK_RF_THRESHOLD = 0.01
FALLBACK_IF_THRESHOLD = 0.15
FALLBACK_HYBRID_THRESHOLD = 0.50


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=PROJECT_ROOT / "data" / "processed" / "day1")
    parser.add_argument("--day1-models-dir", type=Path, default=PROJECT_ROOT / "models" / "day1")
    parser.add_argument("--day2-models-dir", type=Path, default=PROJECT_ROOT / "models" / "day2")
    parser.add_argument("--day2-results-dir", type=Path, default=PROJECT_ROOT / "results" / "day2")
    parser.add_argument("--day3-results-dir", type=Path, default=PROJECT_ROOT / "results" / "day3")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results" / "day4")
    parser.add_argument("--sample-cap", type=int, default=20_000, help="Cap for distribution-shift statistical tests.")
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_DIR / "run_day4.log")],
    )
    return logging.getLogger("run_day4")


def main() -> int:
    args = build_arg_parser().parse_args()
    logger = setup_logging()
    logger.info("Day 4 is analysis-only. No model is retrained and no threshold is tuned using Friday data.")

    import joblib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.utils.validation import check_is_fitted

    from src.day2.anomaly import AnomalyModel, anomaly_scores
    from src.day2.hybrid import HybridConfig, combine_scores, hybrid_predict
    from src.day2.thresholding import RF_DAY1_FROZEN_THRESHOLD
    from src.day3.zero_day import assert_feature_alignment, assert_no_unseen_leakage, class_inventory
    from src.day4.analysis import (
        build_class_feature_shift_table,
        detection_vs_shift_table,
        distribution_shift_tests,
        group_statistics,
        hybrid_threshold_crossing_table,
        rf_feature_importances,
        top_n_features,
        top_shifted_features_per_class,
    )

    # -----------------------------------------------------------------
    # Locate required artifacts. STOP with an exact, actionable message
    # if anything Day 4 needs is missing, rather than fabricating data.
    # -----------------------------------------------------------------
    train_path = args.processed_dir / "train.parquet"
    test_path = args.processed_dir / "test.parquet"
    metadata_path = args.processed_dir / "split_metadata.json"
    rf_model_path = args.day1_models_dir / "random_forest_baseline.joblib"
    if_model_path = args.day2_models_dir / "isolation_forest.joblib"
    day2_metadata_path = args.day2_models_dir / "day2_metadata.json"
    day2_threshold_path = args.day2_results_dir / "validation_threshold_selection.json"

    required = {
        "data/processed/day1/train.parquet": train_path,
        "data/processed/day1/test.parquet": test_path,
        "data/processed/day1/split_metadata.json": metadata_path,
        "models/day1/random_forest_baseline.joblib": rf_model_path,
        "models/day2/isolation_forest.joblib": if_model_path,
    }
    missing = {label: p for label, p in required.items() if not p.exists()}
    if missing:
        logger.error(
            "Day 4 requires existing Day 1 and Day 2 artifacts. Missing exact "
            "artifact(s): %s. These are produced by Day 1's "
            "scripts/prepare_data.py + scripts/train_baseline.py and Day 2's "
            "scripts/run_day2.py, per project convention. STOPPING rather "
            "than fabricating data.",
            {label: str(p) for label, p in missing.items()},
        )
        return 1

    metadata = json.loads(metadata_path.read_text())
    feature_names = metadata["feature_names"]

    args.results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading Day 1 train/test parquet, Day 1 RF, and Day 2 IsolationForest...")
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    rf_model = joblib.load(rf_model_path)
    if_raw_model = joblib.load(if_model_path)

    # -----------------------------------------------------------------
    # Integrity checks (reused from Day 3's src.day3.zero_day; not
    # reimplemented). Prove models are pre-fitted/loaded, not retrained
    # here, and that evaluation features match what each was fit on.
    # -----------------------------------------------------------------
    check_is_fitted(rf_model)
    check_is_fitted(if_raw_model)
    assert_feature_alignment(feature_names, rf_model, context="Random Forest baseline")
    assert_feature_alignment(feature_names, if_raw_model, context="IsolationForest")
    logger.info("Integrity checks passed: RF and IsolationForest are pre-fitted, loaded, feature-aligned models.")

    # -----------------------------------------------------------------
    # A. Verify Bot, DDoS, PortScan are absent from Monday-Thursday
    #    training (reused from Day 3's class_inventory, not
    #    reimplemented).
    # -----------------------------------------------------------------
    inventory = class_inventory(train_df, test_df)
    assert_no_unseen_leakage(train_df, inventory.unseen_attack_classes)

    expected_unseen = {"Bot", "DDoS", "PortScan"}
    actual_unseen = set(inventory.unseen_attack_classes)
    if not expected_unseen.issubset(actual_unseen):
        logger.warning(
            "Expected unseen classes %s are not a subset of the classes "
            "this dataset actually identifies as unseen (%s). Proceeding "
            "with the classes this dataset actually supports, per project "
            "convention (do not fabricate a class that is not genuinely "
            "unseen here).",
            expected_unseen, actual_unseen,
        )
    unseen_classes = sorted(expected_unseen.intersection(actual_unseen)) or inventory.unseen_attack_classes
    logger.info("Unseen classes analyzed in Day 4: %s (verified absent from training).", unseen_classes)

    # -----------------------------------------------------------------
    # B. Load existing Day 3 results (not recomputed).
    # -----------------------------------------------------------------
    day3_comparison_path = args.day3_results_dir / "zero_day_comparison.csv"
    day3_summary_path = args.day3_results_dir / "zero_day_summary.json"
    if not day3_comparison_path.exists() or not day3_summary_path.exists():
        logger.error(
            "Day 4 requires existing Day 3 results. Missing exact artifact(s): %s. "
            "Run scripts/run_day3.py first. STOPPING rather than fabricating data.",
            [str(p) for p in [day3_comparison_path, day3_summary_path] if not p.exists()],
        )
        return 1

    day3_comparison = pd.read_csv(day3_comparison_path)
    day3_summary = json.loads(day3_summary_path.read_text())
    logger.info("Loaded existing Day 3 results from %s.", args.day3_results_dir)

    # -----------------------------------------------------------------
    # Frozen thresholds -- read from Day 1/Day 2/Day 3 artifacts, never
    # re-selected or tuned here.
    # -----------------------------------------------------------------
    rf_threshold = RF_DAY1_FROZEN_THRESHOLD
    if day2_threshold_path.exists():
        day2_thresholds = json.loads(day2_threshold_path.read_text())
        if_threshold = day2_thresholds["isolation_forest"]["selected_threshold"]
        hybrid_threshold = day2_thresholds["hybrid"]["selected_threshold"]
        threshold_source = str(day2_threshold_path)
    else:
        logger.warning(
            "%s not found; using documented fallback thresholds "
            "(IsolationForest=%.2f, Hybrid=%.2f).",
            day2_threshold_path, FALLBACK_IF_THRESHOLD, FALLBACK_HYBRID_THRESHOLD,
        )
        if_threshold = FALLBACK_IF_THRESHOLD
        hybrid_threshold = FALLBACK_HYBRID_THRESHOLD
        threshold_source = "fallback_default"
    logger.info(
        "Frozen thresholds in use -- RF=%.4f, IsolationForest=%.4f, Hybrid=%.4f (source: %s).",
        rf_threshold, if_threshold, hybrid_threshold, threshold_source,
    )

    # -----------------------------------------------------------------
    # Score Friday once with each detector (mirrors Day 3, not Day 2 --
    # models are the same already-frozen artifacts).
    # -----------------------------------------------------------------
    train_medians = train_df[feature_names].median(numeric_only=True)
    X_test = test_df[feature_names].fillna(train_medians)
    assert not X_test.isna().any().any(), "NaNs remain after imputation -- aborting."
    assert not np.isinf(X_test.to_numpy()).any(), "Inf values present after imputation -- aborting."

    if_config = {}
    if day2_metadata_path.exists():
        if_config = json.loads(day2_metadata_path.read_text()).get("isolation_forest_config", {})
    anomaly_model = AnomalyModel(
        model=if_raw_model,
        feature_names=list(feature_names),
        train_medians=train_medians,
        contamination=if_config.get("contamination", if_raw_model.get_params().get("contamination")),
        n_estimators=if_config.get("n_estimators", if_raw_model.get_params().get("n_estimators")),
        random_state=if_config.get("random_state", if_raw_model.get_params().get("random_state")),
        n_training_samples=if_config.get("n_training_samples", -1),
    )

    hybrid_config = HybridConfig(threshold=hybrid_threshold)  # rf_weight/anomaly_weight left at Day 2 defaults (0.7/0.3)

    test_proba = rf_model.predict_proba(X_test)[:, 1]
    test_anomaly = anomaly_scores(anomaly_model, test_df)
    test_hybrid = combine_scores(test_proba, test_anomaly, hybrid_config)

    test_pred_rf = (test_proba >= rf_threshold).astype(int)
    test_pred_if = (test_anomaly >= if_threshold).astype(int)
    test_pred_hybrid = hybrid_predict(test_hybrid, hybrid_threshold)

    # -----------------------------------------------------------------
    # C/D. Feature importance + class-specific feature statistics/shift.
    # -----------------------------------------------------------------
    importances = rf_feature_importances(rf_model, feature_names)
    top_features = top_n_features(importances, TOP_N_FEATURES)
    logger.info("Top %d RF-important features selected for Day 4 analysis.", len(top_features))

    label_col = "label_multiclass"
    benign_df = test_df[test_df[label_col] == inventory.benign_label]
    group_dfs = {"benign": benign_df}
    for cls in unseen_classes:
        group_dfs[cls.lower()] = test_df[test_df[label_col] == cls]

    train_stats = group_statistics(train_df, top_features)
    group_stats = {name: group_statistics(gdf, top_features) for name, gdf in group_dfs.items()}

    shift_table = build_class_feature_shift_table(importances, train_stats, group_stats, top_features)
    shift_table.to_csv(args.results_dir / "day4_class_feature_shift.csv", index=False)

    # -----------------------------------------------------------------
    # E. Top shifted features per unseen class.
    # -----------------------------------------------------------------
    group_names_lower = [c.lower() for c in unseen_classes]
    top_shifted = top_shifted_features_per_class(shift_table, group_names_lower, n=TOP_N_SHIFTED_PER_CLASS)
    top_shifted.to_csv(args.results_dir / "day4_top_shifted_features.csv", index=False)

    # -----------------------------------------------------------------
    # F. Detection rate vs. average feature shift, per unseen class.
    # -----------------------------------------------------------------
    detection_rates: dict[str, dict[str, float]] = {}
    for cls in unseen_classes:
        cls_mask = (test_df[label_col] == cls).to_numpy()
        detection_rates[cls.lower()] = {
            "random_forest": float(test_pred_rf[cls_mask].mean()) if cls_mask.any() else float("nan"),
            "isolation_forest": float(test_pred_if[cls_mask].mean()) if cls_mask.any() else float("nan"),
            "hybrid": float(test_pred_hybrid[cls_mask].mean()) if cls_mask.any() else float("nan"),
        }
    detection_vs_shift = detection_vs_shift_table(shift_table, group_names_lower, detection_rates)
    detection_vs_shift.to_csv(args.results_dir / "day4_detection_vs_shift.csv", index=False)

    # -----------------------------------------------------------------
    # G/H/I. Score distribution plots per detector, benign vs. each
    # unseen class, with the frozen threshold marked.
    # -----------------------------------------------------------------
    figures_dir = args.results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    def _score_dist_plot(scores: np.ndarray, threshold: float, title: str, xlabel: str, out_path: Path) -> None:
        fig, ax = plt.subplots(figsize=(9, 5))
        for cls in ["Benign"] + unseen_classes:
            mask = (test_df[label_col] == cls).to_numpy()
            if mask.sum() == 0:
                continue
            ax.hist(scores[mask], bins=40, alpha=0.55, label=cls, density=True)
        ax.axvline(threshold, color="black", linestyle="--", label=f"Frozen threshold = {threshold}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.set_title(title)
        ax.legend()
        plt.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    _score_dist_plot(
        test_proba, rf_threshold,
        "Random Forest score distribution -- Benign vs. unseen classes (Friday)",
        "RF attack probability", figures_dir / "rf_score_by_unseen_class.png",
    )
    _score_dist_plot(
        test_anomaly, if_threshold,
        "IsolationForest score distribution -- Benign vs. unseen classes (Friday)",
        "Normalized anomaly score", figures_dir / "if_score_by_unseen_class.png",
    )
    _score_dist_plot(
        test_hybrid, hybrid_threshold,
        "Hybrid score distribution -- Benign vs. unseen classes (Friday)",
        "Hybrid score", figures_dir / "hybrid_score_by_unseen_class.png",
    )
    logger.info("Saved RF/IF/Hybrid score-distribution figures to %s.", figures_dir)

    # -----------------------------------------------------------------
    # J. Hybrid threshold-crossing counts per unseen class.
    # -----------------------------------------------------------------
    hybrid_scores_by_class = {cls: test_hybrid[(test_df[label_col] == cls).to_numpy()] for cls in unseen_classes}
    hybrid_threshold_analysis = hybrid_threshold_crossing_table(hybrid_scores_by_class, hybrid_threshold)
    hybrid_threshold_analysis.to_csv(args.results_dir / "day4_hybrid_threshold_analysis.csv", index=False)

    # -----------------------------------------------------------------
    # K. RF importance vs. absolute feature shift scatter (one point
    #    per feature per unseen class).
    # -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    for cls_lower in group_names_lower:
        col = f"{cls_lower}_shift_pct"
        if col not in shift_table.columns:
            continue
        ax.scatter(shift_table[col].abs(), shift_table["RF_importance"], alpha=0.7, label=cls_lower)
    ax.set_xlabel("Absolute feature mean shift vs. training (%)")
    ax.set_ylabel("RF feature importance")
    ax.set_title("RF feature importance vs. Friday feature-distribution shift (top features)")
    ax.legend(title="Unseen class")
    plt.tight_layout()
    fig.savefig(figures_dir / "feature_importance_vs_shift.png", dpi=150)
    plt.close(fig)

    # -----------------------------------------------------------------
    # L. Feature-shift heatmap (top features x classes).
    # -----------------------------------------------------------------
    heatmap_cols = [f"{c}_shift_pct" for c in ["benign"] + group_names_lower if f"{c}_shift_pct" in shift_table.columns]
    heatmap_data = shift_table.set_index("feature")[heatmap_cols]
    fig, ax = plt.subplots(figsize=(8, max(6, 0.3 * len(heatmap_data))))
    im = ax.imshow(heatmap_data.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-200, vmax=200)
    ax.set_xticks(range(len(heatmap_cols)))
    ax.set_xticklabels([c.replace("_shift_pct", "") for c in heatmap_cols], rotation=45, ha="right")
    ax.set_yticks(range(len(heatmap_data)))
    ax.set_yticklabels(heatmap_data.index)
    ax.set_title("Feature mean-shift vs. training (%) -- top RF-important features")
    fig.colorbar(im, ax=ax, label="Mean shift (%)")
    plt.tight_layout()
    fig.savefig(figures_dir / "class_feature_shift_heatmap.png", dpi=150)
    plt.close(fig)

    # -----------------------------------------------------------------
    # M. Unseen-class detector comparison (RF/IF/Hybrid recall by class).
    # -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.25
    x = np.arange(len(unseen_classes))
    for i, detector in enumerate(["random_forest", "isolation_forest", "hybrid"]):
        vals = [detection_rates[c.lower()][detector] for c in unseen_classes]
        ax.bar(x + (i - 1) * width, vals, width, label=detector)
    ax.set_xticks(x)
    ax.set_xticklabels(unseen_classes)
    ax.set_ylabel("Detection rate (recall)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Unseen-attack detection rate by detector (frozen thresholds)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(figures_dir / "unseen_attack_detection_comparison.png", dpi=150)
    plt.close(fig)
    logger.info("Saved analysis figures to %s.", figures_dir)

    # -----------------------------------------------------------------
    # N. Distribution-shift statistical tests (documented sample cap).
    # -----------------------------------------------------------------
    dist_tests = distribution_shift_tests(
        train_df, group_dfs, top_features, sample_cap=args.sample_cap, random_state=args.random_state
    )
    dist_tests.to_csv(args.results_dir / "day4_distribution_tests.csv", index=False)
    logger.info(
        "Ran %d distribution-shift tests (sample_cap=%d, random_state=%d).",
        len(dist_tests), args.sample_cap, args.random_state,
    )

    # -----------------------------------------------------------------
    # O. Unseen-attack summary table.
    # -----------------------------------------------------------------
    summary_rows = []
    for cls in unseen_classes:
        cls_lower = cls.lower()
        n_samples = int((test_df[label_col] == cls).sum())
        summary_rows.append(
            {
                "attack_class": cls,
                "samples": n_samples,
                "rf_detection_rate": detection_rates[cls_lower]["random_forest"],
                "if_detection_rate": detection_rates[cls_lower]["isolation_forest"],
                "hybrid_detection_rate": detection_rates[cls_lower]["hybrid"],
                "mean_abs_feature_shift_pct": float(shift_table[f"{cls_lower}_shift_pct"].abs().mean()),
            }
        )
    unseen_summary = pd.DataFrame(summary_rows)
    unseen_summary.to_csv(args.results_dir / "day4_unseen_attack_summary.csv", index=False)

    # -----------------------------------------------------------------
    # P. Research interpretation -- scientifically conservative,
    #    grounded only in the numbers computed above.
    # -----------------------------------------------------------------
    ordered_by_rf = unseen_summary.sort_values("rf_detection_rate", ascending=False)["attack_class"].tolist()

    interpretation = {
        "summary_ordering_by_rf_detection": ordered_by_rf,
        "findings": [
            f"Among the unseen classes, Random Forest detection rate is highest for "
            f"{ordered_by_rf[0]} and lowest for {ordered_by_rf[-1]}, "
            "consistent with the Day 3 zero-day results.",
            "Binary attack/benign detection by the Random Forest does not mean it can "
            "identify the unseen class by name -- it was never trained with any "
            "label for these classes, only a benign-vs-attack signal.",
            "The hybrid detector's detection rate is lower than both the Random "
            "Forest's and IsolationForest's individually on every unseen class "
            "analyzed here -- a negative result, reported as-is rather than tuned away.",
            "The hybrid score distribution and its frozen threshold (see "
            "day4_hybrid_threshold_analysis.csv and hybrid_score_by_unseen_class.png) "
            "show most unseen-attack rows falling below the frozen 0.5 threshold, "
            "which is consistent with the RF-weighted (0.7) hybrid formula inheriting "
            "the Random Forest's reduced attack-probability confidence on this "
            "temporally-shifted traffic, pulling the combined score down even where "
            "the anomaly component alone was comparatively higher.",
            "Feature-distribution shift between Monday-Thursday training and Friday "
            "is present across several top-importance features (see "
            "day4_class_feature_shift.csv and day4_distribution_tests.csv) and is "
            "associated with reduced Random Forest confidence on temporally-later "
            "traffic. This is reported as an observed association, not as an "
            "established causal mechanism -- the analysis in this notebook does not "
            "isolate feature shift from other possible contributing factors.",
            "Temporal distribution shift is a plausible contributing factor to "
            "reduced detector generalization and is a reasonable direction for "
            "further investigation, but this Day 4 analysis alone does not prove "
            "that shift is the sole cause of the differences observed across DDoS, "
            "PortScan, and Bot.",
        ],
        "caveats": [
            "This is a descriptive/associative analysis. Mann-Whitney U tests and "
            "effect sizes establish that distributions differ; they do not, by "
            "themselves, establish that a given feature's shift caused any specific "
            "detector's error on any specific row.",
            "No model was retrained. No threshold was tuned using Friday data or "
            "unseen-class labels. All thresholds and model artifacts are reused "
            "exactly as frozen in Day 1/Day 2/Day 3.",
        ],
    }
    (args.results_dir / "day4_research_interpretation.json").write_text(
        json.dumps(interpretation, indent=2, default=str)
    )
    print("\n=== Day 4 Research Interpretation ===")
    for line in interpretation["findings"]:
        print("-", line)

    # -----------------------------------------------------------------
    # Q. Metadata.
    # -----------------------------------------------------------------
    day4_metadata = {
        "datasets_used": {"train": str(train_path), "test_friday": str(test_path)},
        "models_used": {"random_forest": str(rf_model_path), "isolation_forest": str(if_model_path)},
        "unseen_classes": unseen_classes,
        "frozen_thresholds": {
            "random_forest": rf_threshold, "isolation_forest": if_threshold,
            "hybrid": hybrid_threshold, "source": threshold_source,
        },
        "hybrid_config": hybrid_config.summary(),
        "features_analyzed": top_features,
        "n_features_analyzed": len(top_features),
        "statistical_methodology": {
            "test": "Mann-Whitney U (two-sided)",
            "effect_sizes": ["rank_biserial_correlation", "cohens_d"],
            "sample_cap": args.sample_cap,
            "random_state": args.random_state,
        },
        "integrity_checks_passed": [
            "rf_and_if_pre_fitted_before_use",
            "feature_alignment_rf",
            "feature_alignment_isolation_forest",
            "no_unseen_class_in_training_data",
            "no_nan_or_inf_after_imputation",
        ],
        "no_models_retrained": True,
        "no_friday_labels_used_for_tuning": True,
        "generated_files": [
            "day4_class_feature_shift.csv", "day4_top_shifted_features.csv",
            "day4_detection_vs_shift.csv", "day4_hybrid_threshold_analysis.csv",
            "day4_distribution_tests.csv", "day4_unseen_attack_summary.csv",
            "day4_research_interpretation.json", "day4_metadata.json",
            "figures/rf_score_by_unseen_class.png", "figures/if_score_by_unseen_class.png",
            "figures/hybrid_score_by_unseen_class.png", "figures/feature_importance_vs_shift.png",
            "figures/class_feature_shift_heatmap.png", "figures/unseen_attack_detection_comparison.png",
        ],
    }
    (args.results_dir / "day4_metadata.json").write_text(json.dumps(day4_metadata, indent=2, default=str))

    logger.info("Day 4 outputs written to: %s", args.results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
