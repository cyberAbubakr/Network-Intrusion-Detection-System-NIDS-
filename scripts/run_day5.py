#!/usr/bin/env python
"""
Day 5 - End-to-end script: cross-dataset generalization evaluation.

Research question: does the NIDS generalize when evaluated on traffic
from a different dataset/distribution?

Reuses (does not modify or re-run):
    data/processed/day1/train.parquet, split_metadata.json
    models/day1/random_forest_baseline.joblib
    models/day2/isolation_forest.joblib, day2_metadata.json
    results/day2/validation_threshold_selection.json
    src.data.cleaner.normalize_labels        (label mapping)
    src.data.validator.detect_label_column   (label-column detection)
    src.day2.anomaly / hybrid / thresholding (scoring + metrics)
    src.day4.analysis.distribution_shift_tests (shift testing)

New Day 5 code (this script + src/day5/*):
    feature-name mapping between CIC-IDS2017 and the external dataset
    external label mapping (thin wrapper around normalize_labels)

External dataset (NOT bundled -- see EXTERNAL_DATASET_DIR below):
    Default expected location: data/external/cse_cic_ids2018/*.csv
    (any number of CSV files; all are concatenated). No proposal
    document was available to this script/assistant, so CSE-CIC-IDS2018
    was assumed as the standard follow-up dataset to CIC-IDS2017 for
    cross-dataset NIDS generalization studies -- adjust
    --external-dir/--dataset-name if your proposal specifies a
    different one.

If the external dataset is not found, this script does NOT fabricate
metrics. It writes results/day5/cross_dataset_metrics.json with
status="pipeline_ready_dataset_required", documents exactly where to
place the data and the exact command to re-run, and exits 0 (this is
an expected, valid state -- not a script error).

CICFlowMeter-derived CSVs (including CSE-CIC-IDS2018) commonly contain
+inf/-inf values (e.g. Flow Bytes/s or Flow Packets/s when Flow
Duration is 0). These are replaced with NaN and then imputed with
CIC-IDS2017 TRAINING medians -- pandas' fillna() does NOT treat inf as
missing, so the replace-then-fillna order below is required; fillna()
alone would leave inf values in place.

Memory note: the external CSV(s) can be very large (CSE-CIC-IDS2018 day
files run into the millions of rows). CSV reading is chunked
(chunksize=50_000) to reduce peak memory during parsing. All rows are
still evaluated -- --sample-cap only limits the distribution-shift
statistical tests (Mann-Whitney U), never the detector metrics, which
are always computed over every successfully processed external row.

Critical scientific rules enforced structurally:
    - This script never imports select_threshold_from_validation --
      thresholds are read from Day 1/Day 2's frozen artifacts only.
    - rf_model.fit(...) / if_raw_model.fit(...) are never called --
      check_is_fitted() runs on both before any external data is
      touched, proving they are pre-fitted, loaded artifacts.
    - The external dataset is used only for evaluation and
      distribution-shift comparison, never for fitting or threshold
      selection.
    - Imputation (for unmapped features AND for inf/NaN cleanup) uses
      only CIC-IDS2017 Day 1 training medians -- never a statistic
      computed from the external data itself.

Outputs (results/day5/ only -- never Day 1-4 outputs):
    results/day5/feature_mapping.json
    results/day5/label_mapping.json
    results/day5/label_mapping_table.csv
    results/day5/comparison_table.csv
    results/day5/attack_family_results.csv   (if attack-family labels available)
    results/day5/distribution_shift.csv
    results/day5/cross_dataset_metrics.json
    results/day5/day5_metadata.json
    results/day5/figures/*.png

Run from the project root:
    python scripts/run_day5.py
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

# Fallback frozen thresholds, used ONLY if the corresponding Day 2
# artifact is missing. Mirror already-established values; NOT
# re-derived here (this script performs no threshold selection).
RF_FROZEN_THRESHOLD = 0.01
FALLBACK_IF_THRESHOLD = 0.15
FALLBACK_HYBRID_THRESHOLD = 0.50

TOP_N_FEATURES_FOR_SHIFT = 20
DEFAULT_SAMPLE_CAP = 20_000
DEFAULT_CSV_CHUNK_SIZE = 50_000


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=PROJECT_ROOT / "data" / "processed" / "day1")
    parser.add_argument("--day1-models-dir", type=Path, default=PROJECT_ROOT / "models" / "day1")
    parser.add_argument("--day2-models-dir", type=Path, default=PROJECT_ROOT / "models" / "day2")
    parser.add_argument("--day2-results-dir", type=Path, default=PROJECT_ROOT / "results" / "day2")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results" / "day5")
    parser.add_argument(
        "--external-dir", type=Path, default=PROJECT_ROOT / "data" / "external" / "cse_cic_ids2018",
        help="Directory containing external-dataset CSV file(s). All *.csv files found are concatenated.",
    )
    parser.add_argument("--dataset-name", type=str, default="CSE-CIC-IDS2018")
    parser.add_argument("--external-label-col", type=str, default=None, help="Override auto-detected label column.")
    parser.add_argument(
        "--sample-cap", type=int, default=DEFAULT_SAMPLE_CAP,
        help="Caps ONLY the distribution-shift statistical tests (Mann-Whitney U). "
             "Detector evaluation (comparison_table.csv etc.) always uses every "
             "successfully processed external row, never a sample.",
    )
    parser.add_argument(
        "--csv-chunk-size", type=int, default=DEFAULT_CSV_CHUNK_SIZE,
        help="Row-chunk size used when reading external CSV file(s), to reduce peak memory during parsing.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_DIR / "run_day5.log")],
    )
    return logging.getLogger("run_day5")


def _load_external_csvs_chunked(csv_paths, chunk_size: int, logger: logging.Logger):
    """
    Read one or more CSV files in row-chunks (reduces peak memory during
    parsing compared to reading each whole file at once) and concatenate
    the result into a single DataFrame. All rows from all files are
    included -- this is a memory-friendlier read, not a sample.
    """
    import pandas as pd

    chunks = []
    total_rows = 0
    for csv_path in csv_paths:
        file_rows = 0
        for chunk in pd.read_csv(csv_path, low_memory=False, chunksize=chunk_size):
            chunk.columns = [c.strip() for c in chunk.columns]
            chunks.append(chunk)
            file_rows += len(chunk)
        total_rows += file_rows
        logger.info("Read %d row(s) from %s (chunksize=%d).", file_rows, csv_path, chunk_size)

    external_df = pd.concat(chunks, ignore_index=True)
    del chunks
    return external_df


def main() -> int:
    args = build_arg_parser().parse_args()
    logger = setup_logging()
    logger.info(
        "Day 5: cross-dataset generalization evaluation against %s. "
        "No model is retrained and no threshold is tuned using external data.",
        args.dataset_name,
    )

    import joblib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.metrics import precision_recall_curve, roc_curve
    from sklearn.utils.validation import check_is_fitted

    from src.data.validator import detect_label_column
    from src.day2.anomaly import AnomalyModel, anomaly_scores
    from src.day2.hybrid import HybridConfig, combine_scores, hybrid_predict
    from src.day2.thresholding import evaluate_frozen_threshold, per_class_frozen_results
    from src.day4.analysis import distribution_shift_tests, rf_feature_importances, top_n_features
    from src.day5.feature_mapping import apply_feature_mapping, build_feature_mapping
    from src.day5.label_mapping import map_external_labels

    args.results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # Required Day 1/Day 2 artifacts. STOP with the exact missing
    # artifact rather than fabricating anything.
    # -----------------------------------------------------------------
    train_path = args.processed_dir / "train.parquet"
    metadata_path = args.processed_dir / "split_metadata.json"
    rf_model_path = args.day1_models_dir / "random_forest_baseline.joblib"
    if_model_path = args.day2_models_dir / "isolation_forest.joblib"
    day2_metadata_path = args.day2_models_dir / "day2_metadata.json"
    day2_threshold_path = args.day2_results_dir / "validation_threshold_selection.json"

    required = {
        "data/processed/day1/train.parquet": train_path,
        "data/processed/day1/split_metadata.json": metadata_path,
        "models/day1/random_forest_baseline.joblib": rf_model_path,
        "models/day2/isolation_forest.joblib": if_model_path,
    }
    missing = {label: p for label, p in required.items() if not p.exists()}
    if missing:
        logger.error(
            "Day 5 requires existing Day 1 and Day 2 artifacts. Missing exact "
            "artifact(s): %s. Run Day 1's scripts/prepare_data.py + "
            "scripts/train_baseline.py and Day 2's scripts/run_day2.py first. "
            "STOPPING rather than fabricating data.",
            {k: str(v) for k, v in missing.items()},
        )
        return 1

    metadata = json.loads(metadata_path.read_text())
    feature_names = metadata["feature_names"]

    logger.info("Loading Day 1 training data, Day 1 RF, and Day 2 IsolationForest...")
    train_df = pd.read_parquet(train_path)
    rf_model = joblib.load(rf_model_path)
    if_raw_model = joblib.load(if_model_path)

    # Integrity check: prove both models are pre-fitted, loaded
    # artifacts BEFORE any external data is touched. No .fit() call
    # exists anywhere in this script.
    check_is_fitted(rf_model)
    check_is_fitted(if_raw_model)
    logger.info("Integrity check passed: RF and IsolationForest are pre-fitted, loaded models.")

    train_medians = train_df[feature_names].median(numeric_only=True)

    # -----------------------------------------------------------------
    # Frozen thresholds -- read from Day 1/Day 2 artifacts, never
    # re-selected or tuned on external data. This script never imports
    # select_threshold_from_validation.
    # -----------------------------------------------------------------
    rf_threshold = RF_FROZEN_THRESHOLD
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
    hybrid_config = HybridConfig(threshold=hybrid_threshold)  # rf_weight/anomaly_weight left at Day 2 defaults (0.7/0.3)
    logger.info(
        "Frozen thresholds in use (unchanged from Day 1/Day 2) -- RF=%.4f, "
        "IsolationForest=%.4f, Hybrid=%.4f (source: %s).",
        rf_threshold, if_threshold, hybrid_threshold, threshold_source,
    )

    # -----------------------------------------------------------------
    # External dataset discovery. If unavailable, write a documented
    # "pipeline_ready_dataset_required" status and stop -- no fake
    # metrics.
    # -----------------------------------------------------------------
    external_csvs = sorted(args.external_dir.glob("*.csv")) if args.external_dir.exists() else []

    if not external_csvs:
        logger.warning(
            "External dataset not found. Expected one or more *.csv files in: %s",
            args.external_dir,
        )
        status = {
            "status": "pipeline_ready_dataset_required",
            "dataset_name": args.dataset_name,
            "expected_location": str(args.external_dir),
            "expected_format": (
                "One or more CICFlowMeter-style CSV files (as distributed for "
                "CSE-CIC-IDS2018), each with a label column (commonly 'Label'). "
                "All *.csv files found in the directory are concatenated."
            ),
            "how_to_obtain": (
                "Download CSE-CIC-IDS2018 (e.g. via the AWS CLI instructions on "
                "the Canadian Institute for Cybersecurity's website) and place "
                "the day-wise CSV file(s) in the expected_location above."
            ),
            "exact_command_to_rerun": (
                f"python scripts/run_day5.py --external-dir {args.external_dir}"
            ),
            "pipeline_components_ready": [
                "feature mapping (src.day5.feature_mapping)",
                "label mapping (src.day5.label_mapping, reuses src.data.cleaner.normalize_labels)",
                "frozen-threshold evaluation (reuses src.day2.thresholding.evaluate_frozen_threshold)",
                "distribution-shift testing (reuses src.day4.analysis.distribution_shift_tests)",
                "inf/NaN cleanup using CIC-IDS2017 training medians (never external statistics)",
            ],
            "note": (
                "feature_mapping.json and label_mapping.json cannot be produced "
                "without the actual external dataset's column/label values -- "
                "they are not fabricated here."
            ),
        }
        (args.results_dir / "cross_dataset_metrics.json").write_text(json.dumps(status, indent=2, default=str))

        day5_metadata = {
            "status": "pipeline_ready_dataset_required",
            "dataset_name": args.dataset_name,
            "expected_external_dir": str(args.external_dir),
            "frozen_thresholds": {
                "random_forest": rf_threshold, "isolation_forest": if_threshold,
                "hybrid": hybrid_threshold, "source": threshold_source,
            },
            "no_models_retrained": True,
            "no_external_data_used_for_threshold_selection": True,
            "generated_files": ["cross_dataset_metrics.json", "day5_metadata.json"],
        }
        (args.results_dir / "day5_metadata.json").write_text(json.dumps(day5_metadata, indent=2, default=str))

        logger.info(
            "Day 5 pipeline is implemented and ready, but the experiment was NOT "
            "experimentally completed -- %s was not found. Wrote status to %s.",
            args.dataset_name, args.results_dir / "cross_dataset_metrics.json",
        )
        return 0

    # -----------------------------------------------------------------
    # Load external CSV(s) in row-chunks to reduce peak memory during
    # parsing. All rows are kept -- this is not a sample.
    # -----------------------------------------------------------------
    logger.info(
        "Found %d external CSV file(s) in %s. Loading (chunked, chunksize=%d)...",
        len(external_csvs), args.external_dir, args.csv_chunk_size,
    )
    external_df = _load_external_csvs_chunked(external_csvs, args.csv_chunk_size, logger)
    logger.info("External dataset loaded: %d rows, %d columns.", len(external_df), external_df.shape[1])

    # -----------------------------------------------------------------
    # Label mapping -- reuses src.data.cleaner.normalize_labels.
    # -----------------------------------------------------------------
    label_col = args.external_label_col or detect_label_column(list(external_df.columns))
    if label_col is None:
        logger.error("Could not detect a label column in the external dataset. STOPPING.")
        return 1

    external_df, label_mapping_table, _log = map_external_labels(external_df, label_col=label_col)
    label_mapping_table.to_csv(args.results_dir / "label_mapping_table.csv", index=False)
    (args.results_dir / "label_mapping.json").write_text(
        json.dumps(
            {
                "external_label_column_detected": label_col,
                "benign_aliases": ["benign"],
                "mapping_table": label_mapping_table.to_dict(orient="records"),
                "note": "Original external label strings are preserved in label_multiclass; only benign-vs-attack is standardized to label_binary.",
            },
            indent=2, default=str,
        )
    )
    logger.info("Label mapping complete: %d distinct original labels found.", len(label_mapping_table))

    # -----------------------------------------------------------------
    # Feature mapping -- never silently drops a CIC-IDS2017 feature.
    # Unchanged from the existing approach: unmapped features are
    # imputed with training medians, never invented mappings.
    # -----------------------------------------------------------------
    mapping = build_feature_mapping(feature_names, external_df.columns)
    (args.results_dir / "feature_mapping.json").write_text(json.dumps(mapping.summary(), indent=2, default=str))

    if not mapping.mapped:
        logger.error(
            "Zero CIC-IDS2017 features could be matched to external columns -- "
            "the datasets are not comparable with this mapping approach. STOPPING."
        )
        return 1
    if len(mapping.unmapped_cic2017_features) > 0:
        logger.warning(
            "%d/%d CIC-IDS2017 features could not be matched to the external "
            "dataset and will be imputed with CIC-IDS2017 training medians: %s",
            len(mapping.unmapped_cic2017_features), len(feature_names), mapping.unmapped_cic2017_features,
        )

    external_mapped = apply_feature_mapping(external_df, mapping, train_medians)

    # -----------------------------------------------------------------
    # Inf/NaN cleanup. CICFlowMeter-derived CSVs (including
    # CSE-CIC-IDS2018) commonly contain +inf/-inf (e.g. rate features
    # divided by a zero Flow Duration). pandas' fillna() does NOT treat
    # inf as missing, so +inf/-inf must be replaced with NaN FIRST,
    # then imputed -- using CIC-IDS2017 TRAINING medians only, never a
    # statistic derived from the external data.
    # -----------------------------------------------------------------
    raw_values = external_mapped.to_numpy(dtype=float)
    n_pos_inf = int(np.isposinf(raw_values).sum())
    n_neg_inf = int(np.isneginf(raw_values).sum())
    n_inf_total = n_pos_inf + n_neg_inf
    n_nan_before = int(external_mapped.isna().sum().sum())

    logger.info(
        "Inf/NaN scan on mapped external features: %d +inf, %d -inf (%d total inf), "
        "%d pre-existing NaN. Replacing all with NaN, then imputing with CIC-IDS2017 "
        "training medians.",
        n_pos_inf, n_neg_inf, n_inf_total, n_nan_before,
    )

    external_mapped = external_mapped.replace([np.inf, -np.inf], np.nan)
    n_nan_after_inf_replacement = int(external_mapped.isna().sum().sum())
    external_mapped = external_mapped.fillna(train_medians)

    logger.info(
        "Imputation complete: %d value(s) (%d inf + %d pre-existing NaN) filled with "
        "CIC-IDS2017 training medians.",
        n_nan_after_inf_replacement, n_inf_total, n_nan_before,
    )

    assert not external_mapped.isna().any().any(), "NaNs remain in mapped external features after imputation."
    assert not np.isinf(external_mapped.to_numpy()).any(), "Inf values present in mapped external features."
    logger.info("Integrity check passed: no NaN/inf remain in mapped external features after imputation.")

    # -----------------------------------------------------------------
    # Score the external dataset once with each detector. Every
    # successfully processed row is scored -- --sample-cap does not
    # apply here (only to the distribution-shift tests below).
    # -----------------------------------------------------------------
    if_config = {}
    if day2_metadata_path.exists():
        if_config = json.loads(day2_metadata_path.read_text()).get("isolation_forest_config", {})
    anomaly_model = AnomalyModel(
        model=if_raw_model, feature_names=list(feature_names), train_medians=train_medians,
        contamination=if_config.get("contamination", if_raw_model.get_params().get("contamination")),
        n_estimators=if_config.get("n_estimators", if_raw_model.get_params().get("n_estimators")),
        random_state=if_config.get("random_state", if_raw_model.get_params().get("random_state")),
        n_training_samples=if_config.get("n_training_samples", -1),
    )

    ext_proba = rf_model.predict_proba(external_mapped[feature_names])[:, 1]
    ext_anomaly = anomaly_scores(anomaly_model, external_mapped)
    ext_hybrid = combine_scores(ext_proba, ext_anomaly, hybrid_config)

    y_ext = external_df["label_binary"].astype(int).to_numpy()

    detector_scores = {
        "random_forest": (ext_proba, rf_threshold),
        "isolation_forest": (ext_anomaly, if_threshold),
        "hybrid": (ext_hybrid, hybrid_threshold),
    }

    n_total = len(external_df)
    n_benign = int((y_ext == 0).sum())
    n_attack = int((y_ext == 1).sum())

    comparison_rows = []
    for detector_name, (scores, threshold) in detector_scores.items():
        metrics = evaluate_frozen_threshold(y_ext, scores, threshold)
        comparison_rows.append(
            {
                "detector": detector_name,
                "dataset": args.dataset_name,
                "threshold_used": threshold,
                "total_samples": n_total,
                "benign_samples": n_benign,
                "attack_samples": n_attack,
                "attack_detection_rate": metrics["recall"],
                **metrics,
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(args.results_dir / "comparison_table.csv", index=False)
    logger.info("Cross-dataset comparison (all %d external rows):\n%s", n_total, comparison.to_string())

    # -----------------------------------------------------------------
    # Per attack-family results, if the external dataset has more than
    # just benign/attack in label_multiclass.
    # -----------------------------------------------------------------
    family_labels = external_df["label_multiclass"]
    attack_family_results = None
    if family_labels[family_labels != "Benign"].nunique() >= 1:
        family_tables = []
        for detector_name, (scores, threshold) in detector_scores.items():
            preds = (np.asarray(scores) >= threshold).astype(int)
            table = per_class_frozen_results(scores, preds, family_labels)
            table.insert(0, "detector", detector_name)
            family_tables.append(table)
        attack_family_results = pd.concat(family_tables, ignore_index=True)
        attack_family_results.to_csv(args.results_dir / "attack_family_results.csv", index=False)

    # -----------------------------------------------------------------
    # Distribution-shift analysis (reuses Day 4's implementation
    # unmodified) -- restricted to features that were actually mapped
    # from real external columns (imputed placeholder features would
    # produce a degenerate, meaningless "shift" of ~0%).
    #
    # --sample-cap applies ONLY here (Mann-Whitney U on a capped random
    # sample per group, for computational feasibility) -- it never
    # limits the detector evaluation above, which always uses all
    # n_total rows.
    # -----------------------------------------------------------------
    importances = rf_feature_importances(rf_model, feature_names)
    top_features = top_n_features(importances, TOP_N_FEATURES_FOR_SHIFT)
    shiftable_features = [f for f in top_features if f in mapping.mapped]
    excluded_from_shift = [f for f in top_features if f not in mapping.mapped]
    if excluded_from_shift:
        logger.warning(
            "%d top-importance feature(s) excluded from distribution-shift "
            "testing because they could not be mapped to a real external "
            "column: %s",
            len(excluded_from_shift), excluded_from_shift,
        )

    dist_tests = pd.DataFrame()
    if shiftable_features:
        dist_tests = distribution_shift_tests(
            train_df, {"external": external_mapped}, shiftable_features,
            sample_cap=args.sample_cap, random_state=args.random_state,
        )
    dist_tests.to_csv(args.results_dir / "distribution_shift.csv", index=False)

    # -----------------------------------------------------------------
    # Visualizations.
    # -----------------------------------------------------------------
    def _score_hist(scores, threshold, title, xlabel, out_name):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(np.asarray(scores)[y_ext == 0], bins=40, alpha=0.6, label="Benign", density=True)
        ax.hist(np.asarray(scores)[y_ext == 1], bins=40, alpha=0.6, label="Attack", density=True)
        ax.axvline(threshold, color="black", linestyle="--", label=f"Frozen threshold = {threshold}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.set_title(title)
        ax.legend()
        plt.tight_layout()
        fig.savefig(figures_dir / out_name, dpi=150)
        plt.close(fig)

    _score_hist(ext_proba, rf_threshold, f"RF score distribution -- {args.dataset_name}", "RF attack probability", "rf_score_distribution.png")
    _score_hist(ext_anomaly, if_threshold, f"IsolationForest score distribution -- {args.dataset_name}", "Normalized anomaly score", "if_score_distribution.png")
    _score_hist(ext_hybrid, hybrid_threshold, f"Hybrid score distribution -- {args.dataset_name}", "Hybrid score", "hybrid_score_distribution.png")

    # Detector comparison bar chart.
    fig, ax = plt.subplots(figsize=(8, 5))
    comparison.set_index("detector")[["precision", "recall", "f1"]].plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"Detector comparison on {args.dataset_name} (frozen thresholds)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    fig.savefig(figures_dir / "detector_comparison.png", dpi=150)
    plt.close(fig)

    # ROC and PR curves.
    fig, ax = plt.subplots(figsize=(7, 6))
    for detector_name, (scores, _threshold) in detector_scores.items():
        fpr, tpr, _ = roc_curve(y_ext, scores)
        ax.plot(fpr, tpr, label=detector_name)
    ax.plot([0, 1], [0, 1], linestyle=":", color="gray", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC curves -- {args.dataset_name}")
    ax.legend()
    plt.tight_layout()
    fig.savefig(figures_dir / "roc_curves.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    for detector_name, (scores, _threshold) in detector_scores.items():
        prec, rec, _ = precision_recall_curve(y_ext, scores)
        ax.plot(rec, prec, label=detector_name)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall curves -- {args.dataset_name}")
    ax.legend()
    plt.tight_layout()
    fig.savefig(figures_dir / "pr_curves.png", dpi=150)
    plt.close(fig)

    # Confusion matrices.
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (detector_name, row) in zip(axes, comparison.set_index("detector").iterrows()):
        cm = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]])
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred Benign", "Pred Attack"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["True Benign", "True Attack"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center")
        ax.set_title(detector_name)
    fig.suptitle(f"Confusion matrices -- {args.dataset_name} (frozen thresholds)")
    plt.tight_layout()
    fig.savefig(figures_dir / "confusion_matrices.png", dpi=150)
    plt.close(fig)

    if not dist_tests.empty:
        fig, ax = plt.subplots(figsize=(9, max(5, 0.3 * len(dist_tests))))
        sorted_tests = dist_tests.sort_values("cohens_d", key=abs, ascending=True)
        ax.barh(sorted_tests["feature"], sorted_tests["cohens_d"])
        ax.set_xlabel("Cohen's d (train vs. external)")
        ax.set_title(f"Feature-distribution shift effect size -- {args.dataset_name} vs. CIC-IDS2017 training")
        plt.tight_layout()
        fig.savefig(figures_dir / "feature_distribution_shift.png", dpi=150)
        plt.close(fig)

    logger.info("Saved figures to %s.", figures_dir)

    # -----------------------------------------------------------------
    # Cross-dataset metrics summary + research interpretation.
    # -----------------------------------------------------------------
    rf_row = comparison[comparison["detector"] == "random_forest"].iloc[0]
    if_row = comparison[comparison["detector"] == "isolation_forest"].iloc[0]
    hybrid_row = comparison[comparison["detector"] == "hybrid"].iloc[0]

    findings = [
        f"Random Forest F1 on {args.dataset_name}: {rf_row['f1']:.4f} (recall={rf_row['recall']:.4f}, precision={rf_row['precision']:.4f}), "
        f"using the frozen CIC-IDS2017 threshold {rf_threshold} without any external-data tuning.",
        f"IsolationForest F1 on {args.dataset_name}: {if_row['f1']:.4f} (recall={if_row['recall']:.4f}).",
        f"Hybrid F1 on {args.dataset_name}: {hybrid_row['f1']:.4f} (recall={hybrid_row['recall']:.4f}).",
    ]
    if hybrid_row["f1"] < min(rf_row["f1"], if_row["f1"]):
        findings.append(
            "The hybrid detector performs worse than both individual detectors on this "
            "external dataset, consistent with the Day 4 finding that its RF-heavy "
            "weighting inherits reduced RF confidence under distribution shift -- this is "
            "reported honestly rather than tuned away."
        )
    elif hybrid_row["f1"] > max(rf_row["f1"], if_row["f1"]):
        findings.append("The hybrid detector improves over both individual detectors on this external dataset.")
    else:
        findings.append("The hybrid detector's performance falls between the two individual detectors on this external dataset.")

    if not dist_tests.empty:
        n_significant = int((dist_tests["p_value"] < 0.01).sum())
        findings.append(
            f"{n_significant}/{len(dist_tests)} tested top-importance features show a "
            f"statistically significant distribution difference (p<0.01) between "
            f"CIC-IDS2017 training and {args.dataset_name}. This is reported as an "
            "association between distribution differences and detector performance, "
            "not as evidence that the shift causes any specific performance change."
        )
    if mapping.unmapped_cic2017_features:
        findings.append(
            f"{len(mapping.unmapped_cic2017_features)} of {len(feature_names)} "
            "CIC-IDS2017 features could not be matched to a real external column and "
            "were imputed with training medians for evaluation -- this is a limitation "
            "of the common-feature approach, not a property of the external traffic "
            "itself, and should be considered when interpreting RF/IsolationForest "
            "scores on this dataset."
        )
    if n_inf_total:
        findings.append(
            f"{n_inf_total} +inf/-inf value(s) were found in the mapped external "
            "features (common in CICFlowMeter-derived rate features such as "
            "Flow Bytes/s when Flow Duration is 0) and were replaced with NaN, then "
            "imputed with CIC-IDS2017 training medians before scoring."
        )

    interpretation = {
        "research_question": "Does the NIDS generalize when evaluated on traffic from a different dataset/distribution?",
        "dataset": args.dataset_name,
        "findings": findings,
        "caveats": [
            "Cross-dataset evaluation compares two different capture environments, time "
            "periods, and (often) traffic-generation methodologies -- differences in "
            "detector performance may reflect any of these, not solely 'generalization' "
            "in the abstract.",
            f"{len(mapping.unmapped_cic2017_features)}/{len(feature_names)} features required "
            "imputation rather than being genuinely observed on the external dataset.",
            f"{n_inf_total} inf value(s) and {n_nan_before} pre-existing NaN value(s) in the "
            "mapped features were imputed with training medians rather than being genuinely "
            "observed finite values.",
            "No model was retrained. No threshold was tuned using external data. All "
            "thresholds and model artifacts are reused exactly as frozen in Day 1/Day 2.",
        ],
    }
    (args.results_dir / "cross_dataset_metrics.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "dataset_name": args.dataset_name,
                "n_external_rows_processed": n_total,
                "n_inf_values_found_and_replaced": n_inf_total,
                "comparison_table": comparison.to_dict(orient="records"),
                "interpretation": interpretation,
            },
            indent=2, default=str,
        )
    )
    print("\n=== Day 5 Research Interpretation ===")
    for line in findings:
        print("-", line)

    # -----------------------------------------------------------------
    # Metadata.
    # -----------------------------------------------------------------
    day5_metadata = {
        "status": "completed",
        "dataset_name": args.dataset_name,
        "external_files_used": [str(p) for p in external_csvs],
        "csv_chunk_size": args.csv_chunk_size,
        "n_external_rows": n_total,
        "inf_nan_handling": {
            "n_pos_inf_found": n_pos_inf,
            "n_neg_inf_found": n_neg_inf,
            "n_inf_total_found": n_inf_total,
            "n_pre_existing_nan_found": n_nan_before,
            "n_values_imputed_with_training_medians": n_nan_after_inf_replacement,
            "imputation_source": "CIC-IDS2017 Day 1 training medians only -- never external statistics",
        },
        "frozen_thresholds": {
            "random_forest": rf_threshold, "isolation_forest": if_threshold,
            "hybrid": hybrid_threshold, "source": threshold_source,
        },
        "hybrid_config": hybrid_config.summary(),
        "feature_mapping_summary": mapping.summary(),
        "features_analyzed_for_shift": shiftable_features,
        "features_excluded_from_shift": excluded_from_shift,
        "statistical_methodology": {
            "test": "Mann-Whitney U (two-sided)", "effect_sizes": ["rank_biserial_correlation", "cohens_d"],
            "sample_cap": args.sample_cap,
            "sample_cap_scope": "distribution_shift_tests ONLY -- detector evaluation always uses all n_external_rows",
            "random_state": args.random_state,
        },
        "integrity_checks_passed": [
            "rf_and_if_pre_fitted_before_external_use",
            "no_select_threshold_from_validation_imported",
            "no_nan_or_inf_after_imputation",
        ],
        "no_models_retrained": True,
        "no_external_data_used_for_threshold_selection": True,
        "generated_files": [
            "feature_mapping.json", "label_mapping.json", "label_mapping_table.csv",
            "comparison_table.csv", "distribution_shift.csv", "cross_dataset_metrics.json",
            "day5_metadata.json",
        ] + (["attack_family_results.csv"] if attack_family_results is not None else []),
    }
    (args.results_dir / "day5_metadata.json").write_text(json.dumps(day5_metadata, indent=2, default=str))

    logger.info("Day 5 outputs written to: %s", args.results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())