"""Lightweight unit tests for src.day2.anomaly using tiny synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.day2.anomaly import (
    anomaly_scores,
    fit_isolation_forest,
    sample_training_data,
)

FEATURES = ["f1", "f2", "f3"]


def _make_synthetic_train_df(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_attack = n // 5
    n_benign = n - n_attack

    benign = pd.DataFrame(
        rng.normal(0, 1, size=(n_benign, 3)), columns=FEATURES
    )
    benign["label_binary"] = 0

    attack = pd.DataFrame(
        rng.normal(5, 1, size=(n_attack, 3)), columns=FEATURES
    )
    attack["label_binary"] = 1

    return pd.concat([benign, attack], ignore_index=True)


def test_sample_training_data_respects_benign_fraction():
    train_df = _make_synthetic_train_df()
    sample = sample_training_data(
        train_df, FEATURES, n_samples=100, benign_fraction=0.9, random_state=1
    )
    assert sample.n_samples == 100
    assert sample.n_benign == 90
    assert sample.n_attack == 10
    assert list(sample.X.columns) == FEATURES


def test_sample_training_data_caps_at_available_rows():
    train_df = _make_synthetic_train_df(n=50)
    sample = sample_training_data(train_df, FEATURES, n_samples=10_000, random_state=1)
    assert sample.n_samples <= 50


def test_isolation_forest_fits_and_scores_run_successfully():
    train_df = _make_synthetic_train_df()
    sample = sample_training_data(train_df, FEATURES, n_samples=200, random_state=1)
    model = fit_isolation_forest(sample, FEATURES, contamination=0.1, random_state=1)

    scores = anomaly_scores(model, train_df)

    assert len(scores) == len(train_df)
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0)


def test_isolation_forest_scores_attacks_more_anomalous_on_average():
    # Attacks were generated far from the benign cluster, so a
    # correctly-behaving IsolationForest should score them as more
    # anomalous on average.
    train_df = _make_synthetic_train_df()
    sample = sample_training_data(
        train_df, FEATURES, n_samples=200, benign_fraction=0.95, random_state=1
    )
    model = fit_isolation_forest(sample, FEATURES, contamination=0.1, random_state=1)

    scores = anomaly_scores(model, train_df)
    mean_benign = scores[train_df["label_binary"] == 0].mean()
    mean_attack = scores[train_df["label_binary"] == 1].mean()

    assert mean_attack > mean_benign


def test_anomaly_scores_handles_nan_via_median_imputation():
    train_df = _make_synthetic_train_df()
    sample = sample_training_data(train_df, FEATURES, n_samples=200, random_state=1)
    model = fit_isolation_forest(sample, FEATURES, random_state=1)

    df_with_nan = train_df.copy()
    df_with_nan.loc[0, "f1"] = np.nan

    scores = anomaly_scores(model, df_with_nan)
    assert not np.isnan(scores).any()
