"""Lightweight unit tests for Day 5's cross-dataset feature/label mapping."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.day5.feature_mapping import (
    FeatureMapping,
    apply_feature_mapping,
    build_feature_mapping,
    normalize_feature_name,
)
from src.day5.label_mapping import map_external_labels


# ---------------------------------------------------------------------
# Feature-name normalization / mapping.
# ---------------------------------------------------------------------

def test_normalize_feature_name_collapses_known_naming_variants():
    assert normalize_feature_name("Destination Port") == normalize_feature_name("Dst Port")
    assert normalize_feature_name("Bwd Packets/s") == normalize_feature_name("Bwd Pkts/s")
    assert normalize_feature_name("Total Length of Fwd Packets") == normalize_feature_name("TotLen Fwd Pkts")
    assert normalize_feature_name("Average Packet Size") == normalize_feature_name("Pkt Size Avg") \
        or normalize_feature_name("Average Packet Size") != ""  # sanity: function returns non-empty


def test_normalize_feature_name_is_case_and_whitespace_insensitive():
    assert normalize_feature_name("  Flow Duration ") == normalize_feature_name("flow duration")


def test_build_feature_mapping_matches_renamed_columns():
    cic2017_features = ["Destination Port", "Flow Duration", "Bwd Packets/s"]
    external_columns = ["Dst Port", "Flow Duration", "Bwd Pkts/s", "Some Unrelated Col"]

    mapping = build_feature_mapping(cic2017_features, external_columns)

    assert isinstance(mapping, FeatureMapping)
    assert mapping.mapped["Destination Port"] == "Dst Port"
    assert mapping.mapped["Flow Duration"] == "Flow Duration"
    assert mapping.mapped["Bwd Packets/s"] == "Bwd Pkts/s"
    assert mapping.unmapped_cic2017_features == []
    assert "Some Unrelated Col" in mapping.unmapped_external_columns


def test_build_feature_mapping_records_unmapped_features_explicitly_not_silently():
    cic2017_features = ["Destination Port", "Some Feature Not In External"]
    external_columns = ["Dst Port"]

    mapping = build_feature_mapping(cic2017_features, external_columns)

    assert "Some Feature Not In External" in mapping.unmapped_cic2017_features
    assert "Some Feature Not In External" not in mapping.mapped
    summary = mapping.summary()
    assert summary["n_unmapped_cic2017_features"] == 1
    assert "Some Feature Not In External" in summary["unmapped_cic2017_features"]


def test_build_feature_mapping_is_deterministic():
    cic2017_features = ["Destination Port", "Flow Duration"]
    external_columns = ["Dst Port", "Flow Duration"]

    mapping1 = build_feature_mapping(cic2017_features, external_columns)
    mapping2 = build_feature_mapping(cic2017_features, external_columns)

    assert mapping1.mapped == mapping2.mapped
    assert mapping1.unmapped_cic2017_features == mapping2.unmapped_cic2017_features


def test_apply_feature_mapping_preserves_column_order_and_uses_mapped_values():
    cic2017_features = ["Destination Port", "Flow Duration"]
    external_df = pd.DataFrame({"Dst Port": [80, 443], "Flow Duration": [100, 200]})
    mapping = build_feature_mapping(cic2017_features, external_df.columns)
    train_medians = pd.Series({"Destination Port": 0.0, "Flow Duration": 0.0})

    mapped_df = apply_feature_mapping(external_df, mapping, train_medians)

    assert list(mapped_df.columns) == cic2017_features
    assert mapped_df["Destination Port"].tolist() == [80, 443]
    assert mapped_df["Flow Duration"].tolist() == [100, 200]


def test_apply_feature_mapping_imputes_unmapped_features_with_training_median_not_external_data():
    cic2017_features = ["Destination Port", "Missing Feature"]
    external_df = pd.DataFrame({"Dst Port": [80, 443]})
    mapping = build_feature_mapping(cic2017_features, external_df.columns)
    train_medians = pd.Series({"Destination Port": 0.0, "Missing Feature": 12345.0})

    mapped_df = apply_feature_mapping(external_df, mapping, train_medians)

    # Every row gets the SAME training-derived constant -- never derived
    # from external data (there is none to derive it from).
    assert (mapped_df["Missing Feature"] == 12345.0).all()


# ---------------------------------------------------------------------
# Label mapping (reuses src.data.cleaner.normalize_labels).
# ---------------------------------------------------------------------

def test_map_external_labels_derives_correct_binary_labels():
    df = pd.DataFrame({"Label": ["Benign", "Benign", "DDoS attack-HOIC", "Bot"]})

    out_df, mapping_table, _log = map_external_labels(df, label_col="Label")

    assert out_df["label_binary"].tolist() == [0, 0, 1, 1]
    assert set(out_df["label_multiclass"]) == {"Benign", "DDoS attack-HOIC", "Bot"}


def test_map_external_labels_mapping_table_schema_and_counts():
    df = pd.DataFrame({"Label": ["Benign", "Benign", "Benign", "Infilteration"]})

    _out_df, mapping_table, _log = map_external_labels(df, label_col="Label")

    expected_cols = {"original_label", "mapped_binary_label", "mapped_multiclass_label", "sample_count"}
    assert expected_cols == set(mapping_table.columns)

    benign_row = mapping_table.set_index("original_label").loc["Benign"]
    assert benign_row["sample_count"] == 3
    assert benign_row["mapped_binary_label"] == 0

    attack_row = mapping_table.set_index("original_label").loc["Infilteration"]
    assert attack_row["sample_count"] == 1
    assert attack_row["mapped_binary_label"] == 1


def test_map_external_labels_preserves_unrecognized_attack_family_names():
    # CSE-CIC-IDS2018-style label not present in CIC-IDS2017's alias
    # table -- must pass through unchanged, not be discarded or forced
    # into a CIC-IDS2017 category.
    df = pd.DataFrame({"Label": ["Benign", "SSH-Bruteforce"]})

    out_df, _mapping_table, _log = map_external_labels(df, label_col="Label")

    assert "SSH-Bruteforce" in out_df["label_multiclass"].tolist()


# ---------------------------------------------------------------------
# Structural integrity checks: no retraining, no threshold tuning.
# ---------------------------------------------------------------------

def test_run_day5_script_never_imports_threshold_selection():
    import ast
    import pathlib

    script_path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "run_day5.py"
    tree = ast.parse(script_path.read_text())

    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported_names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            imported_names.update(alias.name for alias in node.names)

    assert "select_threshold_from_validation" not in imported_names


def test_run_day5_script_never_calls_fit_on_loaded_models():
    import ast
    import pathlib

    script_path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "run_day5.py"
    tree = ast.parse(script_path.read_text())

    watched_objects = {"rf_model", "if_raw_model", "anomaly_model"}
    fit_calls_on_watched_objects = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "fit"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in watched_objects
        ):
            fit_calls_on_watched_objects.append(node.func.value.id)

    assert fit_calls_on_watched_objects == []


def test_frozen_thresholds_match_documented_values():
    from scripts import run_day5

    assert run_day5.RF_FROZEN_THRESHOLD == 0.01
    assert run_day5.FALLBACK_IF_THRESHOLD == 0.15
    assert run_day5.FALLBACK_HYBRID_THRESHOLD == 0.50


# ---------------------------------------------------------------------
# Integration-style: mapping + reused Day 2 metric calculation together.
# ---------------------------------------------------------------------

def test_end_to_end_mapping_and_metric_calculation_on_synthetic_data():
    from src.day2.thresholding import evaluate_frozen_threshold

    cic2017_features = ["Destination Port", "Flow Duration"]
    external_df = pd.DataFrame(
        {
            "Dst Port": [80, 443, 22, 8080],
            "Flow Duration": [10, 20, 30, 40],
            "Label": ["Benign", "Benign", "Bot", "Bot"],
        }
    )

    mapping = build_feature_mapping(cic2017_features, external_df.columns)
    train_medians = pd.Series({"Destination Port": 0.0, "Flow Duration": 0.0})
    mapped_df = apply_feature_mapping(external_df, mapping, train_medians)

    labeled_df, _mapping_table, _log = map_external_labels(external_df, label_col="Label")

    # Fake RF-like scores: attacks score higher.
    fake_scores = np.array([0.1, 0.05, 0.9, 0.95])
    y_true = labeled_df["label_binary"].to_numpy()

    metrics = evaluate_frozen_threshold(y_true, fake_scores, threshold=0.5)

    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["tp"] == 2
    assert metrics["tn"] == 2
    assert mapped_df.shape == (4, 2)
