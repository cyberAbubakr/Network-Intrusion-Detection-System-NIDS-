"""Lightweight unit tests for src.day2.hybrid using tiny synthetic data."""

from __future__ import annotations

import numpy as np
import pytest

from src.day2.hybrid import HybridConfig, combine_scores, hybrid_predict


def test_combine_scores_is_weighted_average():
    rf = np.array([1.0, 0.0])
    anomaly = np.array([0.0, 1.0])
    config = HybridConfig(rf_weight=0.7, anomaly_weight=0.3)

    hybrid = combine_scores(rf, anomaly, config)

    assert hybrid[0] == pytest.approx(0.7)
    assert hybrid[1] == pytest.approx(0.3)


def test_combine_scores_equal_weights_averages():
    rf = np.array([0.8])
    anomaly = np.array([0.2])
    config = HybridConfig(rf_weight=1.0, anomaly_weight=1.0)

    hybrid = combine_scores(rf, anomaly, config)

    assert hybrid[0] == pytest.approx(0.5)


def test_combine_scores_rejects_zero_total_weight():
    rf = np.array([0.5])
    anomaly = np.array([0.5])
    config = HybridConfig(rf_weight=0.0, anomaly_weight=0.0)

    with pytest.raises(ValueError):
        combine_scores(rf, anomaly, config)


def test_hybrid_predict_thresholds_correctly():
    scores = np.array([0.1, 0.5, 0.9])
    preds = hybrid_predict(scores, threshold=0.5)
    assert list(preds) == [0, 1, 1]


def test_hybrid_config_summary_is_serializable():
    config = HybridConfig(rf_weight=0.7, anomaly_weight=0.3, threshold=0.2)
    summary = config.summary()
    assert summary["rf_weight"] == 0.7
    assert summary["anomaly_weight"] == 0.3
    assert summary["threshold"] == 0.2
    assert "formula" in summary
