from __future__ import annotations

from typing import Any, Mapping, Sequence
import json

import numpy as np
import pandas as pd


def reconstruct_day2_anomaly_model(
    raw_isolation_forest: Any,
    train_df: pd.DataFrame,
    feature_names: Sequence[str],
    *,
    n_samples: int = 100_000,
    benign_fraction: float = 0.9,
    contamination: float = 0.05,
    n_estimators: int = 100,
    random_state: int = 42,
):
    """Reconstruct Day 2 AnomalyModel metadata around the frozen IF.

    This does not call fit() and does not retrain the model.
    It reproduces the deterministic Day 2 sample only to recover the
    sample medians that the original AnomalyModel wrapper stored.
    """
    from src.day2.anomaly import AnomalyModel, sample_training_data

    feature_names = list(feature_names)
    missing = [c for c in feature_names if c not in train_df.columns]
    if missing:
        raise KeyError(
            f"Training data is missing {len(missing)} required feature(s): {missing[:10]}"
        )

    anomaly_sample = sample_training_data(
        train_df,
        feature_names,
        n_samples=n_samples,
        benign_fraction=benign_fraction,
        random_state=random_state,
    )

    return AnomalyModel(
        model=raw_isolation_forest,
        feature_names=feature_names,
        train_medians=anomaly_sample.X.median(numeric_only=True),
        contamination=float(contamination),
        n_estimators=int(n_estimators),
        random_state=int(random_state),
        n_training_samples=int(anomaly_sample.n_samples),
    )


def unwrap_isolation_forest(model_or_wrapper: Any) -> Any:
    """Accept raw sklearn IsolationForest or project AnomalyModel wrapper."""
    if hasattr(model_or_wrapper, "model") and hasattr(model_or_wrapper, "feature_names"):
        return model_or_wrapper.model
    return model_or_wrapper


def _as_series(y: Any, index=None) -> pd.Series:
    if isinstance(y, pd.Series):
        return y.copy()
    arr = np.asarray(y)
    if index is None:
        index = np.arange(len(arr))
    return pd.Series(arr, index=index)


def deterministic_sample(y: Any, n_samples: int, random_state: int = 42) -> list[Any]:
    """Return deterministic row labels suitable for pandas .loc."""
    ys = _as_series(y)
    if n_samples <= 0 or len(ys) == 0:
        return []
    if n_samples >= len(ys):
        return ys.index.tolist()

    rng = np.random.default_rng(random_state)
    classes = list(pd.unique(ys))

    if len(classes) < 2:
        pos = rng.choice(np.arange(len(ys)), size=n_samples, replace=False)
        return ys.index[np.sort(pos)].tolist()

    selected: set[int] = set()
    per_class = n_samples // len(classes)
    remainder = n_samples % len(classes)

    for i, cls in enumerate(sorted(classes, key=lambda x: str(x))):
        positions = np.flatnonzero(ys.to_numpy() == cls)
        take = min(len(positions), per_class + (1 if i < remainder else 0))
        if take:
            picked = rng.choice(positions, size=take, replace=False)
            selected.update(int(p) for p in picked)

    if len(selected) < n_samples:
        remaining = np.asarray([i for i in range(len(ys)) if i not in selected], dtype=int)
        need = min(n_samples - len(selected), len(remaining))
        if need:
            picked = rng.choice(remaining, size=need, replace=False)
            selected.update(int(p) for p in picked)

    return ys.index[sorted(selected)].tolist()


def classify_binary_outcomes(y_true: Any, y_pred: Any, index=None) -> dict[str, list[Any]]:
    """Return original row indices for TP, TN, FP and FN."""
    yt = np.asarray(y_true).astype(int)
    yp = np.asarray(y_pred).astype(int)
    if yt.shape != yp.shape:
        raise ValueError("y_true and y_pred must have identical shape.")
    if index is None:
        index = np.arange(len(yt))
    if len(index) != len(yt):
        raise ValueError("index length must match labels.")
    idx = np.asarray(index, dtype=object)
    return {
        "TP": idx[(yt == 1) & (yp == 1)].tolist(),
        "TN": idx[(yt == 0) & (yp == 0)].tolist(),
        "FP": idx[(yt == 0) & (yp == 1)].tolist(),
        "FN": idx[(yt == 1) & (yp == 0)].tolist(),
    }


def _positive_class_shap_array(values: Any, n_rows: int, n_features: int) -> np.ndarray:
    """Normalize common binary-classifier SHAP outputs to (rows, features)."""
    if hasattr(values, "values"):
        values = values.values

    if isinstance(values, list):
        if not values:
            raise ValueError("Empty SHAP values list.")
        arr = np.asarray(values[1] if len(values) > 1 else values[0])
    else:
        arr = np.asarray(values)

    if arr.ndim == 2:
        out = arr
    elif arr.ndim == 3:
        if arr.shape[0] == n_rows and arr.shape[1] == n_features:
            out = arr[:, :, 1 if arr.shape[2] > 1 else 0]
        elif arr.shape[1] == n_rows and arr.shape[2] == n_features:
            out = arr[1 if arr.shape[0] > 1 else 0, :, :]
        elif arr.shape[0] == n_rows and arr.shape[2] == n_features:
            out = arr[:, 1 if arr.shape[1] > 1 else 0, :]
        else:
            raise ValueError(f"Unsupported SHAP shape: {arr.shape}")
    else:
        raise ValueError(f"Unsupported SHAP ndim: {arr.ndim}")

    out = np.asarray(out, dtype=float)
    if out.shape != (n_rows, n_features):
        raise ValueError(f"SHAP shape {out.shape} != {(n_rows, n_features)}")
    return out


def _tree_shap_values(model: Any, X: pd.DataFrame) -> np.ndarray:
    import shap
    explainer = shap.TreeExplainer(model)
    try:
        raw = explainer(X)
    except Exception:
        raw = explainer.shap_values(X)
    return _positive_class_shap_array(raw, len(X), X.shape[1])


def rf_global_shap_importance(model: Any, X: pd.DataFrame, feature_names: Sequence[str]):
    """Return global positive-class TreeSHAP importance and raw 2-D SHAP values."""
    feature_names = list(feature_names)
    Xdf = pd.DataFrame(X, columns=feature_names, index=getattr(X, "index", None))
    shap_values = _tree_shap_values(model, Xdf)
    result = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": np.mean(np.abs(shap_values), axis=0),
    }).sort_values(["mean_abs_shap", "feature"], ascending=[False, True], ignore_index=True)
    result["rank"] = np.arange(1, len(result) + 1)
    result["shap_rank"] = result["rank"]
    return result, shap_values


def _json_scalar(v):
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    return v


def _contributors(feature_names, values, contributions, top_k=10):
    rows = [
        {"feature": str(f), "value": _json_scalar(v), "contribution": float(c)}
        for f, v, c in zip(feature_names, values, contributions)
    ]
    pos = sorted([r for r in rows if r["contribution"] > 0], key=lambda r: r["contribution"], reverse=True)[:top_k]
    neg = sorted([r for r in rows if r["contribution"] < 0], key=lambda r: r["contribution"])[:top_k]
    return pos, neg


def rf_local_shap_explanations(
    model: Any,
    X: pd.DataFrame,
    y_true: Any,
    y_pred: Any,
    probabilities: Any,
    threshold: float,
    feature_names: Sequence[str],
    outcomes: Mapping[str, Sequence[Any]],
    max_per_category: int = 1,
    dataset: str = "CIC-IDS2017",
    top_k: int = 10,
):
    """Return deterministic TP/TN/FP/FN local TreeSHAP explanations."""
    feature_names = list(feature_names)
    Xdf = pd.DataFrame(X, columns=feature_names, index=getattr(X, "index", None))
    yt = _as_series(y_true, Xdf.index)
    yp = _as_series(y_pred, Xdf.index)
    prob = _as_series(probabilities, Xdf.index)

    selected = []
    for outcome in ("TP", "TN", "FP", "FN"):
        for idx in list(outcomes.get(outcome, []))[:max_per_category]:
            selected.append((outcome, idx))
    if not selected:
        return []

    idxs = [idx for _, idx in selected]
    Xsel = Xdf.loc[idxs]
    sv = _tree_shap_values(model, Xsel)

    records = []
    for row_i, (outcome, idx) in enumerate(selected):
        row = Xsel.loc[idx]
        pos, neg = _contributors(feature_names, row.to_numpy(), sv[row_i], top_k)
        pred = int(yp.loc[idx])
        records.append({
            "detector": "random_forest",
            "dataset": dataset,
            "sample_index": _json_scalar(idx),
            "outcome": outcome,
            "true_label": int(yt.loc[idx]),
            "prediction": "attack" if pred else "benign",
            "prediction_binary": pred,
            "score": float(prob.loc[idx]),
            "score_semantics": "random_forest_attack_probability",
            "threshold": float(threshold),
            "evidence_type": "shap",
            "top_positive_contributors": pos,
            "top_negative_contributors": neg,
            "limitations": ["SHAP attribution is not causal evidence."],
        })
    return records


def _safe_normalize(s: pd.Series) -> pd.Series:
    arr = pd.to_numeric(s, errors="coerce").fillna(0.0).astype(float)
    denom = float(np.abs(arr).sum())
    if denom == 0 or not np.isfinite(denom):
        return pd.Series(np.zeros(len(arr)), index=arr.index)
    return np.abs(arr) / denom


def build_importance_comparison(shap_importance, builtin_importance, permutation_importance_df):
    """Merge SHAP, RF built-in and permutation importance by feature."""
    s = shap_importance.copy()
    b = builtin_importance.copy()
    p = permutation_importance_df.copy()

    if "shap_rank" not in s:
        s["shap_rank"] = s.get("rank", s["mean_abs_shap"].rank(ascending=False, method="min"))
    if "builtin_rank" not in b:
        b["builtin_rank"] = b["rf_builtin_importance"].rank(ascending=False, method="min")
    if "permutation_rank" not in p:
        p["permutation_rank"] = p["permutation_importance_mean"].rank(ascending=False, method="min")

    merged = (
        s[[c for c in ["feature", "mean_abs_shap", "shap_rank"] if c in s]]
        .merge(b[[c for c in ["feature", "rf_builtin_importance", "builtin_rank"] if c in b]], on="feature", how="outer")
        .merge(p[[c for c in ["feature", "permutation_importance_mean", "permutation_importance_std", "permutation_rank"] if c in p]], on="feature", how="outer")
    )
    merged["shap_normalized"] = _safe_normalize(merged["mean_abs_shap"])
    merged["builtin_normalized"] = _safe_normalize(merged["rf_builtin_importance"])
    merged["permutation_normalized"] = _safe_normalize(merged["permutation_importance_mean"])
    return merged.sort_values(["shap_rank", "builtin_rank", "permutation_rank", "feature"], na_position="last", ignore_index=True)


def _if_score(model_or_wrapper, X):
    """Higher-is-more-anomalous raw IF score for attribution only."""
    model = unwrap_isolation_forest(model_or_wrapper)
    if hasattr(model, "decision_function"):
        return -np.asarray(model.decision_function(X), dtype=float)
    if hasattr(model, "score_samples"):
        return -np.asarray(model.score_samples(X), dtype=float)
    raise TypeError("Isolation Forest lacks decision_function/score_samples.")


def _if_reference(X):
    return X.median(numeric_only=True).reindex(X.columns)


def _if_perturb_contrib(model_or_wrapper, X, reference):
    """One-feature-at-a-time raw anomaly-score perturbation attribution."""
    base = _if_score(model_or_wrapper, X)
    out = np.zeros((len(X), X.shape[1]), dtype=float)
    for j, col in enumerate(X.columns):
        pert = X.copy()
        pert[col] = reference[col]
        out[:, j] = base - _if_score(model_or_wrapper, pert)
    return out


def isolation_forest_global_explanation(model, X, feature_names, random_state=42):
    """Global IF importance via deterministic one-feature perturbation; not DIFFI."""
    feature_names = list(feature_names)
    Xdf = pd.DataFrame(X, columns=feature_names, index=getattr(X, "index", None))
    contrib = _if_perturb_contrib(model, Xdf, _if_reference(Xdf))
    out = pd.DataFrame({
        "feature": feature_names,
        "importance": np.mean(np.abs(contrib), axis=0),
        "method": "one_feature_at_a_time_raw_anomaly_score_perturbation",
    }).sort_values(["importance", "feature"], ascending=[False, True], ignore_index=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def isolation_forest_local_explanations(
    model,
    X,
    y_true,
    predictions,
    anomaly_score,
    threshold,
    feature_names,
    outcomes,
    max_per_category=1,
    random_state=42,
    dataset="CIC-IDS2017",
    top_k=10,
):
    """Return deterministic local IF explanations; anomaly score is not a probability."""
    feature_names = list(feature_names)
    Xdf = pd.DataFrame(X, columns=feature_names, index=getattr(X, "index", None))
    yt = _as_series(y_true, Xdf.index)
    yp = _as_series(predictions, Xdf.index)
    scores = _as_series(anomaly_score, Xdf.index)

    selected = []
    for outcome in ("TP", "TN", "FP", "FN"):
        for idx in list(outcomes.get(outcome, []))[:max_per_category]:
            selected.append((outcome, idx))
    if not selected:
        return []

    idxs = [idx for _, idx in selected]
    Xsel = Xdf.loc[idxs]
    contrib = _if_perturb_contrib(model, Xsel, _if_reference(Xdf))

    records = []
    for i, (outcome, idx) in enumerate(selected):
        row = Xsel.loc[idx]
        pos, neg = _contributors(feature_names, row.to_numpy(), contrib[i], top_k)
        top = sorted(pos + neg, key=lambda r: abs(r["contribution"]), reverse=True)[:top_k]
        pred = int(yp.loc[idx])
        records.append({
            "detector": "isolation_forest",
            "dataset": dataset,
            "sample_index": _json_scalar(idx),
            "outcome": outcome,
            "true_label": int(yt.loc[idx]),
            "prediction": "attack" if pred else "benign",
            "prediction_binary": pred,
            "anomaly_score": float(scores.loc[idx]),
            "score_semantics": "normalized_anomaly_score_not_probability",
            "threshold": float(threshold),
            "evidence_type": "one_feature_at_a_time_raw_anomaly_score_perturbation",
            "top_contributors": top,
            "top_positive_contributors": pos,
            "top_negative_contributors": neg,
            "limitations": [
                "Isolation Forest anomaly score is not a probability.",
                "Perturbation attribution uses the underlying frozen IsolationForest raw score.",
                "One-feature perturbation is model-agnostic and non-causal.",
                "Feature interactions may not be fully captured.",
            ],
        })
    return records


def _mark_day6_sources(record, source_map=None):
    source_map = source_map or {}
    for key in ("top_contributors", "top_positive_contributors", "top_negative_contributors"):
        for c in record.get(key, []):
            c["feature_source"] = source_map.get(c.get("feature"), "unknown_day6_source")
    return record


def build_llm_evidence(
    rf_local_explanations,
    if_local_explanations,
    rf_threshold,
    if_threshold,
    hybrid_threshold,
    hybrid_weights,
    day6_summary=None,
    day6_mapped_feature_count=9,
    day6_imputed_feature_count=49,
):
    """Build evidence for LLM narration; ML detector remains authoritative."""
    day6_summary = dict(day6_summary or {})
    source_map = day6_summary.get("feature_source_map") or day6_summary.get("feature_provenance") or {}

    def convert(item):
        rec = dict(item)
        rec["decision_authority"] = "machine_learning_detector"
        rec["llm_role"] = "narration_only_no_detection_override"
        rec["frozen_configuration"] = {
            "random_forest_threshold": float(rf_threshold),
            "isolation_forest_threshold": float(if_threshold),
            "hybrid_threshold": float(hybrid_threshold),
            "hybrid_weights": {
                "rf": float(hybrid_weights.get("rf", hybrid_weights.get("rf_weight", 0.7))),
                "anomaly": float(hybrid_weights.get("anomaly", hybrid_weights.get("anomaly_weight", 0.3))),
            },
        }
        limits = list(rec.get("limitations", []))
        if str(rec.get("dataset", "")).lower() in {"ctu-idseval-6", "ctu_idseval6", "day6"}:
            rec = _mark_day6_sources(rec, source_map)
            rec["day6_representation"] = {
                "mapped_feature_count": int(day6_mapped_feature_count),
                "training_median_imputed_feature_count": int(day6_imputed_feature_count),
            }
            limits.extend([
                f"CTU-IDSEVAL-6 limitation: {day6_mapped_feature_count}/58 mapped; {day6_imputed_feature_count}/58 training-median imputed.",
                "unknown_day6_source must not be described as directly observed.",
            ])
        rec["limitations"] = list(dict.fromkeys(limits))
        return rec

    records = [convert(x) for x in rf_local_explanations] + [convert(x) for x in if_local_explanations]
    json.dumps(records, default=_json_scalar)
    return records
