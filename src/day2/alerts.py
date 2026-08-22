"""
Day 2 - Step 7: LLM-ready structured alert.

This module does NOT call any LLM or external API. It only converts a
single detection into a deterministic, JSON-serializable evidence dict
that a future explanation layer could consume.
"""

from __future__ import annotations

import json
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd


def _to_jsonable(value: Any) -> Any:
    """Coerce common numpy scalar types into plain JSON-serializable types."""

    if value is None:
        return None
    if isinstance(value, np.floating):
        value = float(value)
        return None if np.isnan(value) else value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def build_alert(
    row: pd.Series,
    feature_names: Sequence[str],
    rf_probability: float,
    anomaly_score: float,
    hybrid_score: float,
    rf_threshold: float,
    hybrid_threshold: float,
    predicted_label: int,
    top_n_features: int = 5,
    timestamp: Optional[str] = None,
) -> dict[str, Any]:
    """
    Convert a single detection into a structured, JSON-serializable
    evidence dict.

    Parameters
    ----------
    row:
        A single row of feature data (e.g. ``X_test.iloc[i]``).

    feature_names:
        The full feature set used by the detectors.

    rf_probability, anomaly_score, hybrid_score:
        The three signals produced by the Day 1/Day 2 detectors for this
        row.

    rf_threshold, hybrid_threshold:
        The frozen thresholds used to make each detector's decision.

    predicted_label:
        Final predicted binary label (0=benign, 1=attack) for this row.

    top_n_features:
        Number of "top relevant" feature values to include. Selected by
        largest absolute magnitude -- a simple, deterministic,
        explainable proxy for relevance (SHAP is out of scope for Day 2).

    timestamp:
        Optional timestamp string, if available for this row.

    Returns
    -------
    dict
        A JSON-serializable evidence object. ``json.dumps`` is called on
        it before returning, purely to fail loudly here (not downstream)
        if something non-serializable slipped in.
    """

    feature_values = {
        name: _to_jsonable(row.get(name)) for name in feature_names
    }

    ranked = sorted(
        (
            (name, abs(val))
            for name, val in feature_values.items()
            if isinstance(val, (int, float))
        ),
        key=lambda pair: pair[1],
        reverse=True,
    )
    top_features = {name: feature_values[name] for name, _ in ranked[:top_n_features]}

    alert = {
        "timestamp": timestamp,
        "predicted_label": int(predicted_label),
        "rf_attack_probability": float(rf_probability),
        "anomaly_score": float(anomaly_score),
        "hybrid_score": float(hybrid_score),
        "thresholds_used": {
            "rf_threshold": float(rf_threshold),
            "hybrid_threshold": float(hybrid_threshold),
        },
        "detector_decisions": {
            "random_forest": bool(rf_probability >= rf_threshold),
            "hybrid": bool(hybrid_score >= hybrid_threshold),
        },
        "top_feature_values": top_features,
    }

    # Fail loudly here if something isn't serializable, rather than
    # downstream when this is handed to a future LLM layer.
    json.dumps(alert)

    return alert
