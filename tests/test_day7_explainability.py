import json
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest

from src.day7.explainability import (
    deterministic_sample,
    classify_binary_outcomes,
    build_importance_comparison,
    isolation_forest_global_explanation,
    isolation_forest_local_explanations,
    build_llm_evidence,
)


def _data():
    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.normal(size=(100, 4)), columns=["f1", "f2", "f3", "f4"], index=range(1000, 1100))
    y = pd.Series(((X["f1"] + X["f2"]) > 0).astype(int), index=X.index)
    return X, y


def test_deterministic_sample():
    _, y = _data()
    assert deterministic_sample(y, 20, 42) == deterministic_sample(y, 20, 42)


def test_outcomes():
    out = classify_binary_outcomes([1,0,0,1], [1,0,1,0], [10,11,12,13])
    assert out == {"TP":[10], "TN":[11], "FP":[12], "FN":[13]}


def test_importance_merge():
    s = pd.DataFrame({"feature":["a","b"], "mean_abs_shap":[2.,1.], "rank":[1,2]})
    b = pd.DataFrame({"feature":["a","b"], "rf_builtin_importance":[.7,.3]})
    p = pd.DataFrame({"feature":["a","b"], "permutation_importance_mean":[.2,-.1], "permutation_importance_std":[.01,.01]})
    out = build_importance_comparison(s,b,p)
    assert np.isclose(out["shap_normalized"].sum(), 1.0)


def test_if_explainability():
    X, y = _data()
    m = IsolationForest(random_state=0).fit(X)
    glob = isolation_forest_global_explanation(m, X.iloc[:15], X.columns)
    assert len(glob) == 4
    score = -m.decision_function(X)
    pred = (score >= np.median(score)).astype(int)
    outcomes = classify_binary_outcomes(y, pred, X.index)
    local = isolation_forest_local_explanations(m, X, y, pred, score, float(np.median(score)), X.columns, outcomes)
    assert isinstance(local, list)
    if local:
        assert local[0]["score_semantics"] == "anomaly_score_not_probability"


def test_llm_schema():
    rec = [{
        "detector":"random_forest",
        "dataset":"CTU-IDSEVAL-6",
        "prediction":"attack",
        "top_positive_contributors":[{"feature":"Flow Duration","value":1,"contribution":.2}],
        "top_negative_contributors":[],
        "limitations":[]
    }]
    out = build_llm_evidence(rec, [], .01, .15, .50, {"rf":.7,"anomaly":.3}, {}, 9, 49)
    json.dumps(out)
    assert out[0]["llm_role"] == "narration_only_no_detection_override"
    assert out[0]["day6_representation"]["mapped_feature_count"] == 9
