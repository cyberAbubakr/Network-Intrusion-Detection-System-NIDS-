"""
Day 5 - External-dataset label mapping.

Reuses ``src.data.cleaner.normalize_labels`` UNMODIFIED to derive a
binary benign/attack label and a canonical multiclass label from an
external dataset's raw label column, exactly as Day 1 already does for
CIC-IDS2017. Unrecognized label strings (attack-family names specific
to the external dataset) pass through unchanged into
``label_multiclass`` rather than being silently discarded or forced
into a CIC-IDS2017 category.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from src.data.cleaner import normalize_labels


def map_external_labels(
    df: pd.DataFrame,
    label_col: str,
    benign_aliases: Optional[set] = None,
):
    """
    Apply Day 1's existing label-normalization logic to an external
    dataset.

    Returns
    -------
    (df_with_labels, mapping_table, log)

    ``mapping_table`` has one row per distinct ORIGINAL label value
    actually observed in the external data: original_label,
    mapped_binary_label, mapped_multiclass_label, sample_count.
    """

    out_df, log = normalize_labels(df, label_col=label_col, benign_aliases=benign_aliases)

    mapping_table = (
        out_df.groupby(label_col, dropna=False)
        .agg(
            mapped_binary_label=("label_binary", "first"),
            mapped_multiclass_label=("label_multiclass", "first"),
            sample_count=("label_binary", "size"),
        )
        .reset_index()
        .rename(columns={label_col: "original_label"})
        .sort_values("sample_count", ascending=False)
        .reset_index(drop=True)
    )

    return out_df, mapping_table, log
