#!/usr/bin/env python
"""
Day 6 - End-to-end script: CTU-IDSEVAL-6 frozen external evaluation.

This is a FROZEN evaluation, structured identically to Day 5's
CSE-CIC-IDS2018 evaluation. CTU-IDSEVAL-6 is treated strictly as an
unseen external dataset:

    - No model is retrained (Random Forest, IsolationForest).
    - No preprocessing statistic (medians, mappings) is computed from
      CTU-IDSEVAL-6 -- only from CIC-IDS2017 Day 1 training data.
    - No threshold is selected or tuned on CTU-IDSEVAL-6.
    - The hybrid formula and weights (0.7 RF + 0.3 anomaly) are
      unchanged from Day 2.

Reuses (does not modify or duplicate the logic of):
    data/processed/day1/train.parquet, split_metadata.json
    models/day1/random_forest_baseline.joblib
    models/day2/isolation_forest.joblib, day2_metadata.json
    results/day2/validation_threshold_selection.json
    src.data.validator.detect_label_column   (label-column detection)
    src.day2.thresholding.RF_DAY1_FROZEN_THRESHOLD, evaluate_frozen_threshold, per_class_frozen_results
    src.day2.anomaly.AnomalyModel / anomaly_scores
    src.day2.hybrid.HybridConfig / combine_scores / hybrid_predict
    src.day3.zero_day.assert_feature_alignment
    src.day5.feature_mapping.build_feature_mapping / apply_feature_mapping
        (owns +inf/-inf -> NaN -> CIC-IDS2017-training-median imputation,
        with an internal no-NaN/no-inf guarantee -- see that module)
    src.day5.label_mapping.map_external_labels (reuses src.data.cleaner.normalize_labels)

New Day 6 code (this script + src/day6/ctu_idseval.py):
    CTU-IDSEVAL-6 Zeek .conn-labeled.log discovery/parsing (the only piece Day 5
    does not already expose as an importable, dataset-agnostic function)

External dataset (NOT bundled):
    Default expected location: data/external/ctu_idseval6/ (recursive *.conn-labeled.log) (any
    number of CSV files; all are concatenated). No assumption is made
    about CTU-IDSEVAL-6's exact column names or label spelling beyond
    "one or more CSV files, each with a label column somewhere" --
    src.day5.feature_mapping matches columns by normalized-token name,
    and src.data.validator.detect_label_column auto-detects the label
    column. If CTU-IDSEVAL-6 uses a benign-label spelling other than
    "Benign"/"benign" (e.g. "Normal", "Background"), pass
    --benign-aliases to override.

If the external dataset is not found, this script does NOT fabricate
metrics. It writes results/day6/ctu_idseval6_metrics.json with
status="pipeline_ready_dataset_required", documents exactly where to
place the data and the exact command to re-run, and exits 0 (this is
an expected, valid state -- not a script error).

Outputs (results/day6/ only -- never Day 1-5 outputs):
    results/day6/feature_mapping.json
    results/day6/label_mapping.json
    results/day6/label_mapping_table.csv
    results/day6/comparison_table.csv
    results/day6/attack_family_results.csv   (if attack-family labels available)
    results/day6/ctu_idseval6_metrics.json
    results/day6/day6_metadata.json

Run from the project root:
    python scripts/run_day6.py
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
# Hybrid weights -- unchanged from Day 2. Recorded here only for
# reporting; src.day2.hybrid.HybridConfig's own defaults are what is
# actually used (this script never overrides them).
HYBRID_RF_WEIGHT = 0.7
HYBRID_ANOMALY_WEIGHT = 0.3


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=PROJECT_ROOT / "data" / "processed" / "day1")
    parser.add_argument("--day1-models-dir", type=Path, default=PROJECT_ROOT / "models" / "day1")
    parser.add_argument("--day2-models-dir", type=Path, default=PROJECT_ROOT / "models" / "day2")
    parser.add_argument("--day2-results-dir", type=Path, default=PROJECT_ROOT / "results" / "day2")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results" / "day6")
    parser.add_argument(
        "--external-dir", type=Path, default=PROJECT_ROOT / "data" / "external" / "ctu_idseval6",
        help="Directory containing CTU-IDSEVAL-6 Zeek *.conn-labeled.log files. Searched recursively.",
    )
    parser.add_argument("--dataset-name", type=str, default="CTU-IDSEVAL-6")
    parser.add_argument("--external-label-col", type=str, default=None, help="Override auto-detected label column.")
    parser.add_argument(
        "--benign-aliases", type=str, default="benign",
        help="Comma-separated, case-insensitive benign-label spellings (e.g. 'benign,normal') "
             "if CTU-IDSEVAL-6 does not use 'Benign'/'benign'. Do NOT add 'background' here -- "
             "Background is handled separately via --background-policy, not as a benign alias.",
    )
    parser.add_argument(
        "--background-policy", type=str, default="exclude",
        choices=["exclude", "treat_as_benign", "treat_as_malicious"],
        help="How to treat rows labeled 'Background' for binary metrics. Default 'exclude' "
             "(CTU/Stratosphere-family convention: Background is unverified traffic, not a "
             "synonym for attack). See src.day6.ctu_idseval.apply_background_policy.",
    )
    parser.add_argument("--csv-chunk-size", type=int, default=50_000)
    return parser


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_DIR / "run_day6.log")],
    )
    return logging.getLogger("run_day6")



def discover_ctu_zeek_logs(external_dir: Path) -> list[Path]:
    """Discover CTU-IDSEVAL-6 Zeek connection logs recursively."""
    if not external_dir.exists():
        return []
    return sorted(
        p for p in external_dir.rglob("*.conn-labeled.log")
        if p.is_file() and "_macros" not in [part.lower() for part in p.parts]
    )


def _decode_zeek_separator(raw: str) -> str:
    raw = raw.strip()
    try:
        return bytes(raw, "utf-8").decode("unicode_escape")
    except Exception:
        return "\t"


def load_ctu_zeek_logs(log_files: list[Path], chunk_size: int = 50_000):
    """Parse Zeek .conn-labeled.log files using #fields and tab-separated rows."""
    import pandas as pd

    frames = []
    rows_read = 0
    malformed_rows = 0

    for path in log_files:
        separator = "\t"
        fields = None
        batch = []

        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line_no, raw_line in enumerate(fh, start=1):
                line = raw_line.rstrip("\r\n")
                if not line:
                    continue

                if line.startswith("#separator"):
                    separator = _decode_zeek_separator(
                        line[len("#separator"):].strip()
                    )
                    continue

                if line.startswith("#fields"):
                    payload = line[len("#fields"):]
                    if payload.startswith(separator):
                        payload = payload[len(separator):]
                    else:
                        payload = payload.lstrip()
                    fields = payload.split(separator)
                    continue

                if line.startswith("#"):
                    continue

                if fields is None:
                    raise ValueError(
                        f"{path}: data encountered before #fields at line {line_no}"
                    )

                values = line.split(separator)
                if len(values) != len(fields):
                    malformed_rows += 1
                    raise ValueError(
                        f"{path}: malformed row at line {line_no}; "
                        f"expected {len(fields)} fields, got {len(values)}"
                    )

                batch.append(values)
                rows_read += 1

                if len(batch) >= chunk_size:
                    frames.append(pd.DataFrame(batch, columns=fields))
                    batch = []

        if batch:
            frames.append(pd.DataFrame(batch, columns=fields))

    if not frames:
        return pd.DataFrame(), {
            "files_parsed": len(log_files),
            "rows_read": 0,
            "malformed_rows": malformed_rows,
        }

    df = pd.concat(frames, ignore_index=True)
    df = df.replace({"-": pd.NA, "(empty)": pd.NA})

    return df, {
        "files_parsed": len(log_files),
        "rows_read": rows_read,
        "malformed_rows": malformed_rows,
    }



def main() -> int:
    args = build_arg_parser().parse_args()
    logger = setup_logging()
    logger.info(
        "Day 6: CTU-IDSEVAL-6 frozen external evaluation. This is a FROZEN "
        "evaluation -- no model is retrained, no preprocessing statistic is "
        "computed from CTU-IDSEVAL-6, no threshold is tuned on it.",
    )

    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.utils.validation import check_is_fitted

    from src.data.validator import detect_label_column
    from src.day2.anomaly import AnomalyModel, anomaly_scores
    from src.day2.hybrid import HybridConfig, combine_scores, hybrid_predict
    from src.day2.thresholding import evaluate_frozen_threshold, per_class_frozen_results
    from src.day3.zero_day import assert_feature_alignment
    from src.day5.feature_mapping import apply_feature_mapping, build_feature_mapping
    from src.day5.label_mapping import map_external_labels
    from src.day6.ctu_idseval import apply_background_policy, derive_cic_features_from_zeek

    args.results_dir.mkdir(parents=True, exist_ok=True)

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
            "Day 6 requires existing Day 1 and Day 2 artifacts (unchanged, "
            "loaded not retrained). Missing exact artifact(s): %s. Run Day 1's "
            "scripts/prepare_data.py + scripts/train_baseline.py and Day 2's "
            "scripts/run_day2.py first. STOPPING rather than fabricating data.",
            {k: str(v) for k, v in missing.items()},
        )
        return 1

    metadata = json.loads(metadata_path.read_text())
    feature_names = metadata["feature_names"]

    logger.info("Loading Day 1 training data (medians only), Day 1 RF, and Day 2 IsolationForest...")
    train_df = pd.read_parquet(train_path)
    rf_model = joblib.load(rf_model_path)
    if_raw_model = joblib.load(if_model_path)

    # Integrity check: prove both models are pre-fitted, loaded
    # artifacts BEFORE any CTU-IDSEVAL-6 data is touched. No .fit()
    # call exists anywhere in this script.
    check_is_fitted(rf_model)
    check_is_fitted(if_raw_model)
    assert_feature_alignment(feature_names, rf_model, context="Random Forest baseline")
    assert_feature_alignment(feature_names, if_raw_model, context="IsolationForest")
    logger.info("Integrity check passed: RF and IsolationForest are pre-fitted, loaded, feature-aligned models.")

    # The ONLY preprocessing statistic used anywhere below: CIC-IDS2017
    # Day 1 TRAINING medians. Never recomputed from CTU-IDSEVAL-6.
    train_medians = train_df[feature_names].median(numeric_only=True)
    logger.info(
        "Preprocessing statistic source: CIC-IDS2017 Day 1 training medians "
        "(%d features) -- no statistic is computed from CTU-IDSEVAL-6.",
        len(train_medians),
    )

    # -----------------------------------------------------------------
    # Frozen thresholds -- read from Day 1/Day 2 artifacts, never
    # re-selected or tuned on CTU-IDSEVAL-6.
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
    assert hybrid_config.rf_weight == HYBRID_RF_WEIGHT
    assert hybrid_config.anomaly_weight == HYBRID_ANOMALY_WEIGHT
    logger.info(
        "Frozen configuration in use (unchanged from Day 1/Day 2) -- RF=%.4f, "
        "IsolationForest=%.4f, Hybrid=%.4f (source: %s); hybrid weights RF=%.1f/anomaly=%.1f.",
        rf_threshold, if_threshold, hybrid_threshold, threshold_source,
        hybrid_config.rf_weight, hybrid_config.anomaly_weight,
    )

    # -----------------------------------------------------------------
    # CTU-IDSEVAL-6 discovery. If unavailable, write a documented
    # "pipeline_ready_dataset_required" status and stop -- no fake
    # metrics.
    # -----------------------------------------------------------------
    external_logs = discover_ctu_zeek_logs(args.external_dir)

    if not external_logs:
        logger.warning("CTU-IDSEVAL-6 not found. Expected one or more *.conn-labeled.log files recursively under: %s", args.external_dir)
        status = {
            "status": "pipeline_ready_dataset_required",
            "dataset_name": args.dataset_name,
            "expected_location": str(args.external_dir),
            "expected_format": (
                "One or more Zeek *.conn-labeled.log files containing flow-level features and a "
                "label column (auto-detected via src.data.validator.detect_label_column, "
                "or pass --external-label-col to override). Column names do not need to "
                "match CIC-IDS2017 exactly -- src.day5.feature_mapping matches by "
                "normalized token name. Zeek files are discovered recursively and concatenated."
            ),
            "exact_command_to_rerun": f"python scripts/run_day6.py --external-dir {args.external_dir}",
            "pipeline_components_ready": [
                "Zeek .conn-labeled.log recursive discovery and parsing",
                "feature mapping (src.day5.feature_mapping, unmodified, reused as-is)",
                "label mapping (src.day5.label_mapping, unmodified, reused as-is)",
                "+inf/-inf/NaN sanitization via CIC-IDS2017 training medians (src.day5.feature_mapping.apply_feature_mapping)",
                "frozen-threshold evaluation (src.day2.thresholding.evaluate_frozen_threshold, unmodified)",
            ],
            "note": (
                "feature_mapping.json and label_mapping.json cannot be produced "
                "without the actual CTU-IDSEVAL-6 column/label values -- they are "
                "not fabricated here."
            ),
        }
        (args.results_dir / "ctu_idseval6_metrics.json").write_text(json.dumps(status, indent=2, default=str))

        day6_metadata = {
            "status": "pipeline_ready_dataset_required",
            "dataset_name": args.dataset_name,
            "expected_external_dir": str(args.external_dir),
            "frozen_thresholds": {
                "random_forest": rf_threshold, "isolation_forest": if_threshold,
                "hybrid": hybrid_threshold, "source": threshold_source,
            },
            "hybrid_weights": {"rf_weight": hybrid_config.rf_weight, "anomaly_weight": hybrid_config.anomaly_weight},
            "no_models_retrained": True,
            "no_preprocessing_statistics_from_external_data": True,
            "no_threshold_tuning_on_external_data": True,
            "generated_files": ["ctu_idseval6_metrics.json", "day6_metadata.json"],
        }
        (args.results_dir / "day6_metadata.json").write_text(json.dumps(day6_metadata, indent=2, default=str))

        logger.info(
            "Day 6 pipeline is implemented and ready, but the experiment was NOT "
            "experimentally completed -- %s was not found. Wrote status to %s.",
            args.dataset_name, args.results_dir / "ctu_idseval6_metrics.json",
        )
        return 0

    # -----------------------------------------------------------------
    # Load CTU-IDSEVAL-6 (chunked; all rows, not a sample).
    # -----------------------------------------------------------------
    logger.info(
        "Found %d CTU-IDSEVAL-6 Zeek .conn-labeled.log file(s) under %s.",
        len(external_logs), args.external_dir,
    )
    external_df, zeek_quality = load_ctu_zeek_logs(
        external_logs, chunk_size=args.csv_chunk_size
    )
    logger.info("Zeek parsing summary: %s", json.dumps(zeek_quality))
    if external_df.empty:
        logger.error("Zeek files were found but no data rows were parsed. STOPPING.")
        return 1

    # -----------------------------------------------------------------
    # Zeek -> CIC-IDS2017 feature adapter (Day 6 only; NEW).
    #
    # Runs strictly before build_feature_mapping/apply_feature_mapping,
    # which are otherwise unmodified. Adds columns spelled with exact
    # standard CIC-IDS2017/CICFlowMeter feature names, computed only from
    # Zeek fields that were actually present in this export -- no
    # fabricated packet-level statistics, no CTU-derived preprocessing
    # statistics (this is per-row arithmetic on CTU's own field values,
    # not a fitted statistic). See src.day6.ctu_idseval for the exact
    # mapping table, unit-conversion caveat, and what is intentionally
    # left unmapped/imputed.
    # -----------------------------------------------------------------
    external_df, zeek_cic_report = derive_cic_features_from_zeek(external_df)
    logger.info("Zeek->CIC adapter report: %s", json.dumps(zeek_cic_report.summary(), default=str))

    # -----------------------------------------------------------------
    # Label mapping -- reuses src.day5.label_mapping (which reuses
    # src.data.cleaner.normalize_labels), unmodified.
    # -----------------------------------------------------------------
    label_col = args.external_label_col or ("label" if "label" in external_df.columns else detect_label_column(list(external_df.columns)))
    if label_col is None:
        logger.error("Could not detect a label column in CTU-IDSEVAL-6. Pass --external-label-col. STOPPING.")
        return 1

    benign_aliases = {a.strip().lower() for a in args.benign_aliases.split(",") if a.strip()}
    if "background" in benign_aliases:
        logger.error(
            "'background' must not be passed via --benign-aliases; Background is handled "
            "explicitly via --background-policy (default: exclude from binary metrics), not "
            "folded into 'benign'. STOPPING rather than silently mislabeling Background."
        )
        return 1

    # Keep the ORIGINAL raw label string around under a stable name before
    # map_external_labels runs, so the Background policy below can key off
    # the true source string rather than whatever label_binary Day 1/5's
    # normalize_labels (a two-class Benign/Attack function) assigned it.
    external_df["_day6_original_label"] = external_df[label_col]

    external_df, label_mapping_table, _log = map_external_labels(
        external_df, label_col=label_col, benign_aliases=benign_aliases
    )

    # -----------------------------------------------------------------
    # Background/Benign/Malicious policy (Day 6 only; NEW).
    #
    # normalize_labels (Day 1/5, unmodified, out of scope for Day 6) was
    # designed for a two-class Benign/Attack world, so absent this step
    # "Background" would silently fall through to "attack" purely because
    # it doesn't match a benign alias string. This makes that decision
    # explicit, documented, and overridable via --background-policy
    # instead. See src.day6.ctu_idseval.apply_background_policy for the
    # rationale (CTU/Stratosphere-family convention).
    # -----------------------------------------------------------------
    n_rows_before_background_policy = len(external_df)
    external_df, background_policy_report = apply_background_policy(
        external_df,
        original_label_col="_day6_original_label",
        label_binary_col="label_binary",
        policy=args.background_policy,
    )
    logger.info("Background label policy applied: %s", json.dumps(background_policy_report, default=str))
    external_df = external_df.drop(columns=["_day6_original_label"])
    label_mapping_table.to_csv(args.results_dir / "label_mapping_table.csv", index=False)
    (args.results_dir / "label_mapping.json").write_text(
        json.dumps(
            {
                "external_label_column_detected": label_col,
                "benign_aliases": sorted(benign_aliases),
                "mapping_table": label_mapping_table.to_dict(orient="records"),
                "note": "Original CTU-IDSEVAL-6 label strings are preserved in label_multiclass; only benign-vs-attack is standardized to label_binary.",
            },
            indent=2, default=str,
        )
    )
    logger.info("Label mapping complete: %d distinct original labels found.", len(label_mapping_table))

    # -----------------------------------------------------------------
    # Feature mapping + sanitization -- reuses src.day5.feature_mapping
    # unmodified. apply_feature_mapping guarantees no NaN/inf remain
    # and imputes exclusively with CIC-IDS2017 training medians.
    # -----------------------------------------------------------------
    mapping = build_feature_mapping(feature_names, external_df.columns)
    feature_mapping_output = mapping.summary()
    feature_mapping_output["zeek_to_cic_adapter"] = zeek_cic_report.summary()
    (args.results_dir / "feature_mapping.json").write_text(json.dumps(feature_mapping_output, indent=2, default=str))

    if not mapping.mapped:
        logger.error(
            "Zero CIC-IDS2017 features could be matched to CTU-IDSEVAL-6 columns -- "
            "the datasets are not comparable with this mapping approach. STOPPING."
        )
        return 1
    if mapping.unmapped_cic2017_features:
        logger.warning(
            "%d/%d CIC-IDS2017 features could not be matched to CTU-IDSEVAL-6 and "
            "will be imputed with CIC-IDS2017 training medians: %s",
            len(mapping.unmapped_cic2017_features), len(feature_names), mapping.unmapped_cic2017_features,
        )

    external_mapped = apply_feature_mapping(external_df, mapping, train_medians)
    sanitization = external_mapped.attrs.get("sanitization", {})
    logger.info(
        "CTU-IDSEVAL-6 feature sanitization (via apply_feature_mapping): %s",
        json.dumps(sanitization),
    )

    # Final integrity check (belt-and-suspenders on top of
    # apply_feature_mapping's own internal verification).
    assert not external_mapped.isna().any().any(), "NaNs remain in mapped external features after imputation."
    assert not np.isinf(external_mapped.to_numpy(dtype=float)).any(), "Inf values remain in mapped external features after imputation."
    logger.info("Integrity check passed: no NaN/inf remain in mapped CTU-IDSEVAL-6 features after imputation.")

    # -----------------------------------------------------------------
    # Score CTU-IDSEVAL-6 once with each detector, using the frozen
    # models and the frozen hybrid formula/weights.
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

    ctu_proba = rf_model.predict_proba(external_mapped[feature_names])[:, 1]
    ctu_anomaly = anomaly_scores(anomaly_model, external_mapped)
    ctu_hybrid = combine_scores(ctu_proba, ctu_anomaly, hybrid_config)  # 0.7 * RF + 0.3 * anomaly, unchanged

    y_ctu = external_df["label_binary"].astype(int).to_numpy()

    detector_scores = {
        "random_forest": (ctu_proba, rf_threshold),
        "isolation_forest": (ctu_anomaly, if_threshold),
        "hybrid": (ctu_hybrid, hybrid_threshold),
    }

    n_total = len(external_df)
    n_benign = int((y_ctu == 0).sum())
    n_attack = int((y_ctu == 1).sum())

    comparison_rows = []
    for detector_name, (scores, threshold) in detector_scores.items():
        metrics = evaluate_frozen_threshold(y_ctu, scores, threshold)
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
    logger.info("CTU-IDSEVAL-6 comparison (all %d rows, frozen config):\n%s", n_total, comparison.to_string())

    # -----------------------------------------------------------------
    # Attack-family breakdown, if labels support it. Never invented.
    # -----------------------------------------------------------------
    family_labels = (external_df["detailedlabel"].fillna(external_df["label_multiclass"]).astype(str) if "detailedlabel" in external_df.columns else external_df["label_multiclass"])
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
    # Metrics + reproducibility summary.
    # -----------------------------------------------------------------
    (args.results_dir / "ctu_idseval6_metrics.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "dataset_name": args.dataset_name,
                "n_samples": n_total,
                "n_benign_samples": n_benign,
                "n_attack_samples": n_attack,
                "n_features": len(feature_names),
                "comparison_table": comparison.to_dict(orient="records"),
                "background_label_policy": background_policy_report,
                "zeek_to_cic_feature_adapter": zeek_cic_report.summary(),
                "note": (
                    "This is a frozen evaluation: no model retrained, no "
                    "preprocessing statistic computed from CTU-IDSEVAL-6, no "
                    "threshold tuned on it. n_samples/n_benign_samples/"
                    "n_attack_samples reflect rows remaining AFTER the "
                    "Background policy (see background_label_policy)."
                ),
            },
            indent=2, default=str,
        )
    )

    day6_metadata = {
        "status": "completed",
        "dataset_name": args.dataset_name,
        "external_files_used": [str(p) for p in external_logs],
        "zeek_chunk_size": args.csv_chunk_size,
        "zeek_parsing": zeek_quality,
        "n_samples": n_total,
        "label_distribution": {"benign": n_benign, "attack": n_attack},
        "sanitization": sanitization,
        "background_label_policy": background_policy_report,
        "rows_before_background_policy": n_rows_before_background_policy,
        "zeek_to_cic_feature_adapter": zeek_cic_report.summary(),
        "frozen_thresholds": {
            "random_forest": rf_threshold, "isolation_forest": if_threshold,
            "hybrid": hybrid_threshold, "source": threshold_source,
        },
        "hybrid_config": hybrid_config.summary(),
        "hybrid_weights_confirmed_unchanged": {
            "rf_weight": hybrid_config.rf_weight, "anomaly_weight": hybrid_config.anomaly_weight,
        },
        "feature_mapping_summary": mapping.summary(),
        "model_artifact_paths": {"random_forest": str(rf_model_path), "isolation_forest": str(if_model_path)},
        "preprocessing_artifact_paths": {"train_medians_source": str(train_path)},
        "integrity_checks_passed": [
            "rf_and_if_pre_fitted_before_ctu_use",
            "feature_alignment_rf",
            "feature_alignment_isolation_forest",
            "no_nan_or_inf_after_imputation",
            "preprocessing_statistics_from_cic_ids2017_training_only",
        ],
        "no_models_retrained": True,
        "no_preprocessing_statistics_from_external_data": True,
        "no_threshold_tuning_on_external_data": True,
        "generated_files": [
            "feature_mapping.json", "label_mapping.json", "label_mapping_table.csv",
            "comparison_table.csv", "ctu_idseval6_metrics.json", "day6_metadata.json",
        ] + (["attack_family_results.csv"] if attack_family_results is not None else []),
    }
    (args.results_dir / "day6_metadata.json").write_text(json.dumps(day6_metadata, indent=2, default=str))

    logger.info("Day 6 outputs written to: %s", args.results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())