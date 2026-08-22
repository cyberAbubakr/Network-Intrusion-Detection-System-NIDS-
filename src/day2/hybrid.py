"""
Day 2 - Step 6: Transparent hybrid detector.

Combines two already-normalized [0, 1] signals:
    1. The Day 1 Random Forest's attack probability.
    2. The Day 2 IsolationForest's normalized anomaly score.

The combination is a simple weighted average -- deliberately not a
neural ensemble or a black-box stacker, per the Day 2 brief ("The design
must be explainable").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_RF_WEIGHT = 0.7
DEFAULT_ANOMALY_WEIGHT = 0.3
DEFAULT_HYBRID_THRESHOLD = 0.5


@dataclass
class HybridConfig:
    """Configuration for the transparent hybrid score."""

    rf_weight: float = DEFAULT_RF_WEIGHT
    anomaly_weight: float = DEFAULT_ANOMALY_WEIGHT
    threshold: float = DEFAULT_HYBRID_THRESHOLD

    def summary(self) -> dict:
        return {
            "rf_weight": self.rf_weight,
            "anomaly_weight": self.anomaly_weight,
            "threshold": self.threshold,
            "formula": (
                "hybrid_score = (rf_weight * rf_probability + "
                "anomaly_weight * anomaly_score) / "
                "(rf_weight + anomaly_weight)"
            ),
        }


def combine_scores(
    rf_proba: np.ndarray,
    anomaly_score: np.ndarray,
    config: HybridConfig,
) -> np.ndarray:
    """
    Weighted linear combination of the RF probability and the
    normalized IsolationForest anomaly score. Both inputs are assumed
    to already be in [0, 1].
    """

    rf_proba_arr = np.asarray(rf_proba)
    anomaly_score_arr = np.asarray(anomaly_score)

    weight_sum = config.rf_weight + config.anomaly_weight
    if weight_sum <= 0:
        raise ValueError("rf_weight + anomaly_weight must be > 0.")

    return (
        config.rf_weight * rf_proba_arr
        + config.anomaly_weight * anomaly_score_arr
    ) / weight_sum


def hybrid_predict(hybrid_score: np.ndarray, threshold: float) -> np.ndarray:
    """Binary prediction from the hybrid score at a given threshold."""

    return (np.asarray(hybrid_score) >= threshold).astype(int)
