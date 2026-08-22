"""Lightweight unit tests for src.day2.alerts using tiny synthetic data."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.day2.alerts import build_alert

FEATURES = ["f1", "f2", "f3"]


def _make_row() -> pd.Series:
    return pd.Series({"f1": np.float64(3.2), "f2": np.float64(-1.0), "f3": np.nan})


def test_build_alert_is_json_serializable():
    row = _make_row()
    alert = build_alert(
        row=row,
        feature_names=FEATURES,
        rf_probability=0.9,
        anomaly_score=0.8,
        hybrid_score=0.87,
        rf_threshold=0.2,
        hybrid_threshold=0.3,
        predicted_label=1,
    )
    # Should not raise.
    dumped = json.dumps(alert)
    assert isinstance(dumped, str)


def test_build_alert_contains_expected_fields():
    row = _make_row()
    alert = build_alert(
        row=row,
        feature_names=FEATURES,
        rf_probability=0.9,
        anomaly_score=0.8,
        hybrid_score=0.87,
        rf_threshold=0.2,
        hybrid_threshold=0.3,
        predicted_label=1,
        timestamp="2017-07-07T09:00:00",
    )

    assert alert["timestamp"] == "2017-07-07T09:00:00"
    assert alert["predicted_label"] == 1
    assert alert["rf_attack_probability"] == 0.9
    assert alert["detector_decisions"]["random_forest"] is True
    assert alert["detector_decisions"]["hybrid"] is True
    assert "top_feature_values" in alert


def test_build_alert_top_features_respects_top_n():
    row = _make_row()
    alert = build_alert(
        row=row,
        feature_names=FEATURES,
        rf_probability=0.5,
        anomaly_score=0.5,
        hybrid_score=0.5,
        rf_threshold=0.5,
        hybrid_threshold=0.5,
        predicted_label=0,
        top_n_features=2,
    )
    assert len(alert["top_feature_values"]) <= 2


def test_build_alert_handles_nan_feature_gracefully():
    row = _make_row()  # f3 is NaN
    alert = build_alert(
        row=row,
        feature_names=FEATURES,
        rf_probability=0.1,
        anomaly_score=0.1,
        hybrid_score=0.1,
        rf_threshold=0.5,
        hybrid_threshold=0.5,
        predicted_label=0,
    )
    # NaN should have been coerced to None, not a raw float NaN
    # (which is technically not valid JSON).
    assert alert["top_feature_values"].get("f3") is None or "f3" not in alert["top_feature_values"]
    json.dumps(alert)
