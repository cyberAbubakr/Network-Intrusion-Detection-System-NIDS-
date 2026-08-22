"""Lightweight unit tests for src.day3.zero_day using tiny synthetic data."""

from __future__ import annotations

import pandas as pd
import pytest

from src.day3.zero_day import (
    ClassInventory,
    assert_feature_alignment,
    assert_no_unseen_leakage,
    build_population,
    class_inventory,
)


def _make_train_test():
    train_df = pd.DataFrame(
        {
            "label_multiclass": (
                ["Benign"] * 4 + ["DoS Hulk"] * 2 + ["FTP-Patator"] * 2
            ),
        }
    )
    test_df = pd.DataFrame(
        {
            "label_multiclass": (
                ["Benign"] * 4 + ["DoS Hulk"] * 2 + ["DDoS"] * 2 + ["PortScan"] * 1
            ),
        }
    )
    return train_df, test_df


def test_class_inventory_identifies_unseen_classes_only():
    train_df, test_df = _make_train_test()

    inventory = class_inventory(train_df, test_df)

    assert isinstance(inventory, ClassInventory)
    assert "DoS Hulk" in inventory.seen_attack_classes
    assert "FTP-Patator" in inventory.seen_attack_classes
    assert set(inventory.unseen_attack_classes) == {"DDoS", "PortScan"}
    # DoS Hulk appears on Friday too, but it is NOT unseen -- it was in training.
    assert "DoS Hulk" not in inventory.unseen_attack_classes
    assert "Benign" not in inventory.unseen_attack_classes


def test_class_inventory_uses_full_training_period_not_a_subset():
    # A class present ONLY in a later part of "training" (simulating
    # Thursday) must still count as seen, since the RF is fit on the
    # full Monday-Thursday period, not a Monday-Wednesday sub-split.
    train_df = pd.DataFrame({"label_multiclass": ["Benign", "DoS GoldenEye"]})
    test_df = pd.DataFrame({"label_multiclass": ["Benign", "DoS GoldenEye", "Bot"]})

    inventory = class_inventory(train_df, test_df)

    assert "DoS GoldenEye" not in inventory.unseen_attack_classes
    assert inventory.unseen_attack_classes == ["Bot"]


def test_assert_no_unseen_leakage_passes_for_genuinely_unseen_classes():
    train_df, test_df = _make_train_test()
    inventory = class_inventory(train_df, test_df)

    # Should not raise.
    assert_no_unseen_leakage(train_df, inventory.unseen_attack_classes)


def test_assert_no_unseen_leakage_raises_if_class_actually_in_training():
    train_df, test_df = _make_train_test()

    with pytest.raises(AssertionError):
        assert_no_unseen_leakage(train_df, ["DoS Hulk"])  # DoS Hulk IS in training


def test_build_population_includes_only_benign_and_requested_classes():
    _, test_df = _make_train_test()

    pop = build_population(test_df, ["DDoS"], benign_label="Benign")

    assert set(pop["label_multiclass"].unique()) == {"Benign", "DDoS"}
    assert "PortScan" not in pop["label_multiclass"].values
    assert "DoS Hulk" not in pop["label_multiclass"].values


def test_assert_feature_alignment_passes_when_model_lacks_feature_names_in(caplog):
    class DummyModelNoFeatureNames:
        pass

    # Should not raise -- just warns and skips the strict check.
    assert_feature_alignment(["f1", "f2"], DummyModelNoFeatureNames(), context="dummy")


def test_assert_feature_alignment_raises_on_mismatch():
    class DummyModel:
        feature_names_in_ = ["f1", "f2", "f3"]

    with pytest.raises(AssertionError):
        assert_feature_alignment(["f1", "f2"], DummyModel(), context="dummy")
