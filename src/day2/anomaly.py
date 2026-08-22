"""
Day 2 - Step 5: Lightweight anomaly detector.

Uses scikit-learn's IsolationForest (already a project dependency via
scikit-learn -- no new packages required) trained on a small, controlled,
mostly-benign sample of the training data, per the Day 2 brief:

    "Do NOT train it on the entire raw dataset if unnecessary.
     Use a controlled training sample from the training data, preferably
     emphasizing benign traffic because anomaly detection is intended to
     model normal behavior."

This module keeps a single sensible default configuration plus room for
one documented alternative -- no broad hyperparameter search.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)

DEFAULT_N_SAMPLES = 100_000
DEFAULT_BENIGN_FRACTION = 0.9
DEFAULT_CONTAMINATION = 0.05
DEFAULT_N_ESTIMATORS = 100
DEFAULT_RANDOM_STATE = 42


@dataclass
class AnomalyTrainingSample:
    """A controlled, documented sample used to fit IsolationForest."""

    X: pd.DataFrame
    n_samples: int
    n_benign: int
    n_attack: int
    benign_fraction_requested: float
    random_state: int

    def summary(self) -> dict:
        return {
            "n_samples": self.n_samples,
            "n_benign": self.n_benign,
            "n_attack": self.n_attack,
            "benign_fraction_requested": self.benign_fraction_requested,
            "benign_fraction_actual": (
                self.n_benign / self.n_samples if self.n_samples else 0.0
            ),
            "random_state": self.random_state,
            "n_features": self.X.shape[1],
        }


def sample_training_data(
    train_df: pd.DataFrame,
    feature_names: Sequence[str],
    n_samples: int = DEFAULT_N_SAMPLES,
    benign_fraction: float = DEFAULT_BENIGN_FRACTION,
    random_state: int = DEFAULT_RANDOM_STATE,
    label_col: str = "label_binary",
) -> AnomalyTrainingSample:
    """
    Draw a controlled, mostly-benign sample from the training data for
    fitting IsolationForest.

    A small amount of attack traffic is intentionally included (10% by
    default) because IsolationForest is unsupervised and does not need a
    pure-benign set; keeping a small attack minority makes the "normal
    behavior" boundary slightly more realistic without turning this into
    a supervised model. The dominant signal is still benign traffic.

    Parameters
    ----------
    train_df:
        Day 1 training DataFrame (or the Day 2 sub-train subset).

    feature_names:
        Feature columns to use (reuses the Day 1 RF's ``feature_names``).

    n_samples:
        Target sample size. Capped at ``len(train_df)``. Default 100,000
        rows -- small enough to fit comfortably on an 8GB CPU-only
        machine.

    benign_fraction:
        Target fraction of the sample that is benign. Default 0.9.

    random_state:
        Seed for reproducible sampling.

    label_col:
        Binary label column used only to stratify the sample (0=benign,
        1=attack); it is not included in ``X``.
    """

    n_samples = min(n_samples, len(train_df))

    n_benign_target = int(round(n_samples * benign_fraction))
    n_attack_target = n_samples - n_benign_target

    benign_pool = train_df[train_df[label_col] == 0]
    attack_pool = train_df[train_df[label_col] == 1]

    n_benign = min(n_benign_target, len(benign_pool))
    n_attack = min(n_attack_target, len(attack_pool))

    benign_sample = benign_pool.sample(n=n_benign, random_state=random_state)
    attack_sample = (
        attack_pool.sample(n=n_attack, random_state=random_state)
        if n_attack > 0
        else attack_pool.iloc[0:0]
    )

    sample = (
        pd.concat([benign_sample, attack_sample])
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )

    X = sample[list(feature_names)].copy()
    medians = X.median(numeric_only=True)
    X = X.fillna(medians)

    result = AnomalyTrainingSample(
        X=X,
        n_samples=len(sample),
        n_benign=n_benign,
        n_attack=n_attack,
        benign_fraction_requested=benign_fraction,
        random_state=random_state,
    )

    logger.info(
        "IsolationForest training sample: %d rows (%d benign, %d attack, "
        "benign_fraction_requested=%.2f), random_state=%d, %d features",
        result.n_samples,
        result.n_benign,
        result.n_attack,
        benign_fraction,
        random_state,
        X.shape[1],
    )

    return result


@dataclass
class AnomalyModel:
    """A fitted IsolationForest plus the metadata needed to score new data."""

    model: IsolationForest
    feature_names: list[str]
    train_medians: pd.Series
    contamination: float
    n_estimators: int
    random_state: int
    n_training_samples: int

    def config_summary(self) -> dict:
        return {
            "model_type": "IsolationForest",
            "contamination": self.contamination,
            "n_estimators": self.n_estimators,
            "random_state": self.random_state,
            "n_training_samples": self.n_training_samples,
            "n_features": len(self.feature_names),
        }


def fit_isolation_forest(
    sample: AnomalyTrainingSample,
    feature_names: Sequence[str],
    contamination: float = DEFAULT_CONTAMINATION,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> AnomalyModel:
    """Fit IsolationForest on a pre-sampled, mostly-benign training set."""

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(sample.X)

    logger.info(
        "Fitted IsolationForest: n_estimators=%d, contamination=%.3f, "
        "random_state=%d, trained on %d rows.",
        n_estimators,
        contamination,
        random_state,
        sample.n_samples,
    )

    return AnomalyModel(
        model=model,
        feature_names=list(feature_names),
        train_medians=sample.X.median(numeric_only=True),
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=random_state,
        n_training_samples=sample.n_samples,
    )


def anomaly_scores(anomaly_model: AnomalyModel, df: pd.DataFrame) -> np.ndarray:
    """
    Return normalized anomaly scores in [0, 1] for ``df``, where higher
    means more anomalous (more attack-like).

    scikit-learn's ``IsolationForest.score_samples`` returns higher =
    more normal, so scores are inverted and then min-max normalized
    over the batch being scored, purely for interpretability /
    combination with the RF probability downstream (see
    ``src.day2.hybrid``). Min-max normalization is fit fresh on each
    call, which is fine here because both validation and Friday are
    scored as whole batches, not streamed row by row.
    """

    X = df[anomaly_model.feature_names].copy()
    X = X.fillna(anomaly_model.train_medians)

    raw = anomaly_model.model.score_samples(X)  # higher = more normal
    inverted = -raw  # higher = more anomalous

    lo, hi = float(inverted.min()), float(inverted.max())
    if hi - lo < 1e-12:
        return np.zeros_like(inverted)

    return (inverted - lo) / (hi - lo)
