"""
Day 4 - Analysis/explainability utilities.

Pure analysis: descriptive statistics, feature-distribution shift,
threshold-crossing counts, and distribution-shift significance tests.
Nothing here fits, retrains, or tunes anything -- every function takes
already-computed scores/predictions or already-loaded models and
produces a table.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_CAP = 20_000
DEFAULT_RANDOM_STATE = 42


def rf_feature_importances(rf_model, feature_names: Sequence[str]) -> pd.Series:
    """
    Return the Random Forest's already-fitted feature_importances_ as a
    Series indexed by feature name, sorted descending. Does not refit
    the model.
    """
    importances = pd.Series(
        rf_model.feature_importances_, index=list(feature_names), name="RF_importance"
    )
    return importances.sort_values(ascending=False)


def top_n_features(importances: pd.Series, n: int = 20) -> list[str]:
    """Top-n most important feature names (by RF importance)."""
    return list(importances.head(n).index)


def group_statistics(df: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    """
    Descriptive statistics (mean, std, median, 25%, 75%) for each
    feature in ``features``, computed on ``df``.
    """
    stats_df = df[list(features)].agg(
        ["mean", "std", "median", lambda s: s.quantile(0.25), lambda s: s.quantile(0.75)]
    )
    stats_df.index = ["mean", "std", "median", "25%", "75%"]
    return stats_df.T


def build_class_feature_shift_table(
    importances: pd.Series,
    train_stats: pd.DataFrame,
    group_stats: Mapping[str, pd.DataFrame],
    features: Sequence[str],
) -> pd.DataFrame:
    """
    Build the ``day4_class_feature_shift.csv`` table: one row per
    feature, with RF importance, the training mean, each group's mean,
    and each group's percentage mean-shift relative to training.

    ``group_stats`` keys are expected to include (at minimum) "benign",
    "ddos", "portscan", "bot" -- callers control the exact group set.
    """
    rows = []
    for feature in features:
        train_mean = float(train_stats.loc[feature, "mean"])
        row: dict = {
            "feature": feature,
            "RF_importance": float(importances.get(feature, float("nan"))),
            "train_mean": train_mean,
        }
        for group_name, gstats in group_stats.items():
            group_mean = float(gstats.loc[feature, "mean"])
            row[f"{group_name}_mean"] = group_mean
            if train_mean == 0 or not np.isfinite(train_mean):
                shift_pct = float("nan")
            else:
                shift_pct = (group_mean - train_mean) / abs(train_mean) * 100.0
            row[f"{group_name}_shift_pct"] = shift_pct
        rows.append(row)

    return pd.DataFrame(rows).sort_values("RF_importance", ascending=False).reset_index(drop=True)


def top_shifted_features_per_class(
    shift_table: pd.DataFrame,
    class_names: Sequence[str],
    n: int = 10,
) -> pd.DataFrame:
    """
    Long-format table: for each class, the top-n features by absolute
    shift percentage. Columns: class, feature, RF_importance,
    shift_pct, abs_shift_pct, rank.
    """
    rows = []
    for cls in class_names:
        col = f"{cls}_shift_pct"
        if col not in shift_table.columns:
            continue
        sub = shift_table[["feature", "RF_importance", col]].copy()
        sub["abs_shift_pct"] = sub[col].abs()
        sub = sub.sort_values("abs_shift_pct", ascending=False).head(n)
        sub = sub.rename(columns={col: "shift_pct"})
        sub.insert(0, "class", cls)
        sub["rank"] = range(1, len(sub) + 1)
        rows.append(sub)

    if not rows:
        return pd.DataFrame(columns=["class", "feature", "RF_importance", "shift_pct", "abs_shift_pct", "rank"])

    return pd.concat(rows, ignore_index=True)


def detection_vs_shift_table(
    shift_table: pd.DataFrame,
    class_names: Sequence[str],
    detection_rates: Mapping[str, Mapping[str, float]],
) -> pd.DataFrame:
    """
    One row per unseen class: mean absolute feature shift (across the
    analyzed top-N features) vs. each detector's detection rate on that
    class. Purely descriptive -- this is a correlational comparison,
    not a statistical test of causality.

    ``detection_rates`` maps class -> {"random_forest": rate,
    "isolation_forest": rate, "hybrid": rate}.
    """
    rows = []
    for cls in class_names:
        col = f"{cls}_shift_pct"
        mean_abs_shift = float(shift_table[col].abs().mean()) if col in shift_table.columns else float("nan")
        rates = detection_rates.get(cls, {})
        rows.append(
            {
                "class": cls,
                "mean_abs_feature_shift_pct": mean_abs_shift,
                "rf_detection_rate": rates.get("random_forest", float("nan")),
                "if_detection_rate": rates.get("isolation_forest", float("nan")),
                "hybrid_detection_rate": rates.get("hybrid", float("nan")),
            }
        )
    return pd.DataFrame(rows)


def hybrid_threshold_crossing_table(
    scores_by_class: Mapping[str, np.ndarray],
    threshold: float,
) -> pd.DataFrame:
    """
    For each class's hybrid scores, count how many rows cross
    ``threshold`` and summarize the score distribution. Matches the
    ``day4_hybrid_threshold_analysis.csv`` schema:
    attack_class | samples | above_threshold | detection_rate |
    mean_hybrid_score | median_hybrid_score | max_hybrid_score
    """
    rows = []
    for cls, scores in scores_by_class.items():
        scores = np.asarray(scores)
        above = int((scores >= threshold).sum())
        n = len(scores)
        rows.append(
            {
                "attack_class": cls,
                "samples": n,
                "above_threshold": above,
                "detection_rate": above / n if n else float("nan"),
                "mean_hybrid_score": float(scores.mean()) if n else float("nan"),
                "median_hybrid_score": float(np.median(scores)) if n else float("nan"),
                "max_hybrid_score": float(scores.max()) if n else float("nan"),
            }
        )
    return pd.DataFrame(rows)


@dataclass
class DistributionTestResult:
    feature: str
    group: str
    n_train_sampled: int
    n_group_sampled: int
    mannwhitney_u: float
    p_value: float
    rank_biserial_effect_size: float
    cohens_d: float


def _cap_sample(series: pd.Series, cap: int, random_state: int) -> pd.Series:
    series = series.dropna()
    if len(series) > cap:
        return series.sample(n=cap, random_state=random_state)
    return series


def distribution_shift_tests(
    train_df: pd.DataFrame,
    group_dfs: Mapping[str, pd.DataFrame],
    features: Sequence[str],
    sample_cap: int = DEFAULT_SAMPLE_CAP,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    """
    Mann-Whitney U test (training vs. each group) for each feature,
    reporting both a p-value and an effect size (rank-biserial
    correlation and Cohen's d), computed on a capped random sample of
    each population for computational feasibility.

    This tests whether the two distributions differ -- it does NOT
    establish that any resulting difference *causes* detector
    behavior; that inference is left to the (conservatively worded)
    research interpretation, not to this function.
    """
    rows = []
    for group_name, gdf in group_dfs.items():
        for feature in features:
            train_sample = _cap_sample(train_df[feature], sample_cap, random_state)
            group_sample = _cap_sample(gdf[feature], sample_cap, random_state)

            if len(train_sample) < 2 or len(group_sample) < 2:
                logger.warning(
                    "Skipping distribution test for feature=%s group=%s: "
                    "insufficient samples after capping/dropna.",
                    feature, group_name,
                )
                continue

            u_stat, p_value = stats.mannwhitneyu(
                train_sample, group_sample, alternative="two-sided"
            )

            n1, n2 = len(train_sample), len(group_sample)
            # Rank-biserial correlation effect size, in [-1, 1].
            rank_biserial = 1.0 - (2.0 * u_stat) / (n1 * n2)

            pooled_std = np.sqrt(
                ((n1 - 1) * train_sample.std(ddof=1) ** 2 + (n2 - 1) * group_sample.std(ddof=1) ** 2)
                / (n1 + n2 - 2)
            )
            cohens_d = (
                float((group_sample.mean() - train_sample.mean()) / pooled_std)
                if pooled_std > 0
                else float("nan")
            )

            rows.append(
                DistributionTestResult(
                    feature=feature,
                    group=group_name,
                    n_train_sampled=n1,
                    n_group_sampled=n2,
                    mannwhitney_u=float(u_stat),
                    p_value=float(p_value),
                    rank_biserial_effect_size=float(rank_biserial),
                    cohens_d=cohens_d,
                ).__dict__
            )

    return pd.DataFrame(rows)
