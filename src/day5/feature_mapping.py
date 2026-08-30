"""
Day 5 - Feature-name mapping between the CIC-IDS2017 canonical feature
set (as already frozen in data/processed/day1/split_metadata.json) and
an external dataset's raw CICFlowMeter-style column names (e.g.
CSE-CIC-IDS2018).

Two datasets built with different CICFlowMeter versions commonly use
different tokens for the same measurement (e.g. "Packets" vs "Pkts",
"Destination" vs "Dst"). This module normalizes both column-name sets
to a common token vocabulary and matches on that, rather than assuming
either dataset's exact spelling. Nothing is silently dropped -- every
CIC-IDS2017 feature that cannot be matched is recorded explicitly, and
is imputed with a CIC-IDS2017 TRAINING statistic (never derived from
external data) when the mapped DataFrame is built.

``apply_feature_mapping`` additionally guarantees its output contains
no NaN or +inf/-inf values: CICFlowMeter-derived CSVs (including
CSE-CIC-IDS2018) commonly contain +inf/-inf in rate-style features
(e.g. Flow Bytes/s when Flow Duration is 0), and pandas' ``fillna()``
does not treat inf as missing. Infinities are replaced with NaN first,
then all NaN (from inf-replacement, coercion failures, or genuinely
missing external values) is imputed using CIC-IDS2017 training medians
only -- never a statistic computed from the external data.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Known long-form <-> short-form token substitutions seen across
# CICFlowMeter versions/datasets (CIC-IDS2017 vs CSE-CIC-IDS2018 in
# particular, e.g. "Bwd Packets/s" vs "Bwd Pkts/s", "Destination Port"
# vs "Dst Port"). Applied during normalization, in addition to generic
# punctuation/case stripping, so equivalent columns collapse to the
# same normalized key even when spelled differently.
_TOKEN_REPLACEMENTS = {
    "packets": "pkts", "packet": "pkt",
    "bytes": "byts", "byte": "byt",
    "length": "len",
    "total": "tot",
    "destination": "dst",
    "source": "src",
    "average": "avg",
    "variance": "var",
    "count": "cnt",
    "forward": "fwd",
    "backward": "bwd",
    "minimum": "min",
    "maximum": "max",
    "segment": "seg",
    "bulk": "blk",
    "header": "hdr",
}


def normalize_feature_name(name: str) -> str:
    """
    Normalize a CICFlowMeter-style column name to a version-independent
    token key: lowercase, punctuation/whitespace stripped, common
    filler words dropped, and known long-form/short-form token pairs
    collapsed to the same spelling.
    """
    key = str(name).strip().lower()
    key = re.sub(r"[\s/\\_\-\.]+", " ", key)
    words = [w for w in key.split(" ") if w and w not in {"of", "the"}]
    words = [_TOKEN_REPLACEMENTS.get(w, w) for w in words]
    return "".join(words)


@dataclass
class FeatureMapping:
    """Result of matching CIC-IDS2017 canonical features against an external dataset's raw columns."""

    mapped: dict
    unmapped_cic2017_features: list
    unmapped_external_columns: list
    cic2017_features: list

    def summary(self) -> dict:
        return {
            "n_cic2017_features": len(self.cic2017_features),
            "n_mapped": len(self.mapped),
            "n_unmapped_cic2017_features": len(self.unmapped_cic2017_features),
            "unmapped_cic2017_features": self.unmapped_cic2017_features,
            "n_unmapped_external_columns": len(self.unmapped_external_columns),
            "unmapped_external_columns": self.unmapped_external_columns,
            "mapping": self.mapped,
            "note": (
                "unmapped_cic2017_features are imputed with CIC-IDS2017 "
                "training medians when evaluating the external dataset -- "
                "never silently dropped, never derived from external data."
            ),
        }


def build_feature_mapping(
    cic2017_features: Sequence[str],
    external_columns: Sequence[str],
) -> FeatureMapping:
    """
    Match each CIC-IDS2017 feature to an external-dataset column by
    normalized name. One-to-one: each external column is used at most
    once (first normalized match wins).
    """

    cic2017_features = list(cic2017_features)
    external_columns = list(external_columns)

    external_by_norm: dict[str, str] = {}
    for col in external_columns:
        norm = normalize_feature_name(col)
        if norm not in external_by_norm:
            external_by_norm[norm] = col

    mapped: dict[str, str] = {}
    used_external_cols: set[str] = set()
    for feature in cic2017_features:
        norm = normalize_feature_name(feature)
        ext_col = external_by_norm.get(norm)
        if ext_col is not None and ext_col not in used_external_cols:
            mapped[feature] = ext_col
            used_external_cols.add(ext_col)

    unmapped_cic2017 = [f for f in cic2017_features if f not in mapped]
    unmapped_external = [c for c in external_columns if c not in used_external_cols]

    logger.info(
        "Feature mapping: %d/%d CIC-IDS2017 features mapped to external columns; "
        "%d unmapped CIC-IDS2017 features; %d unmapped external columns.",
        len(mapped), len(cic2017_features), len(unmapped_cic2017), len(unmapped_external),
    )

    return FeatureMapping(
        mapped=mapped,
        unmapped_cic2017_features=unmapped_cic2017,
        unmapped_external_columns=unmapped_external,
        cic2017_features=cic2017_features,
    )


def apply_feature_mapping(
    external_df: pd.DataFrame,
    mapping: FeatureMapping,
    train_medians: pd.Series,
) -> pd.DataFrame:
    """
    Build a DataFrame with exactly ``mapping.cic2017_features`` as
    columns, in that order, using the external dataset's mapped columns
    where available, fully sanitized for downstream model scoring.

    Pipeline:
        1. Mapped columns are coerced to numeric (``errors="coerce"``);
           any CIC-IDS2017 feature that could not be mapped is filled
           with the CIC-IDS2017 TRAINING median for that feature.
        2. ``+inf``/``-inf`` (common in CICFlowMeter-derived rate
           features, e.g. Flow Bytes/s when Flow Duration is 0) are
           detected, logged, and replaced with NaN.
        3. All resulting NaN (coercion failures, inf-replacement, and
           genuinely missing external values alike) are imputed using
           ``train_medians`` ONLY -- never a statistic computed from
           the external data itself.
        4. The result is verified to contain no remaining NaN or
           infinite values; violations raise ``ValueError`` rather
           than silently returning bad data.

    Callers can rely on the returned DataFrame being fully finite and
    ready for ``model.predict_proba`` / ``score_samples`` without
    further cleanup.
    """

    out = pd.DataFrame(index=external_df.index)
    for feature in mapping.cic2017_features:
        if feature in mapping.mapped:
            out[feature] = pd.to_numeric(external_df[mapping.mapped[feature]], errors="coerce")
        else:
            out[feature] = float(train_medians.get(feature, np.nan))

    # --- Detect +inf / -inf before touching anything -------------------
    raw_values = out.to_numpy(dtype=float)
    n_pos_inf = int(np.isposinf(raw_values).sum())
    n_neg_inf = int(np.isneginf(raw_values).sum())
    n_inf_total = n_pos_inf + n_neg_inf
    n_nan_before = int(out.isna().sum().sum())

    logger.info(
        "apply_feature_mapping: pre-sanitization scan -- %d +inf, %d -inf "
        "(%d total inf), %d NaN across %d rows x %d features.",
        n_pos_inf, n_neg_inf, n_inf_total, n_nan_before, len(out), len(mapping.cic2017_features),
    )

    # --- +inf/-inf -> NaN (pandas' fillna() does NOT treat inf as missing) ---
    out = out.replace([np.inf, -np.inf], np.nan)
    n_nan_after_inf_replacement = int(out.isna().sum().sum())

    # --- Impute using CIC-IDS2017 TRAINING medians only -----------------
    out = out.fillna(train_medians)

    n_remaining_nan = int(out.isna().sum().sum())
    remaining_values = out.to_numpy(dtype=float)
    n_remaining_inf = int(np.isinf(remaining_values).sum())

    logger.info(
        "apply_feature_mapping: sanitization complete -- %d value(s) imputed "
        "with CIC-IDS2017 training medians (%d from inf-replacement, %d "
        "pre-existing NaN); %d NaN and %d inf remain.",
        n_nan_after_inf_replacement, n_inf_total, n_nan_before,
        n_remaining_nan, n_remaining_inf,
    )

    if n_remaining_nan > 0:
        raise ValueError(
            f"apply_feature_mapping: {n_remaining_nan} NaN value(s) remain after "
            "imputation with CIC-IDS2017 training medians. This likely means a "
            "CIC-IDS2017 feature's training median is itself NaN/missing -- check "
            "train_medians for the affected feature(s) before proceeding."
        )
    if n_remaining_inf > 0:
        raise ValueError(
            f"apply_feature_mapping: {n_remaining_inf} infinite value(s) remain "
            "after sanitization. This likely means a CIC-IDS2017 training median "
            "used for imputation is itself +inf/-inf -- check train_medians for "
            "the affected feature(s) before proceeding."
        )

    # Attach sanitization stats to the DataFrame so callers (scripts,
    # notebooks) can report them without recomputing/duplicating this
    # logic themselves.
    out.attrs["sanitization"] = {
        "n_pos_inf_found": n_pos_inf,
        "n_neg_inf_found": n_neg_inf,
        "n_inf_total_found": n_inf_total,
        "n_nan_found_before_imputation": n_nan_before,
        "n_values_imputed_with_training_medians": n_nan_after_inf_replacement,
        "n_remaining_nan": n_remaining_nan,
        "n_remaining_inf": n_remaining_inf,
        "imputation_source": "CIC-IDS2017 Day 1 training medians only -- never external statistics",
    }

    return out