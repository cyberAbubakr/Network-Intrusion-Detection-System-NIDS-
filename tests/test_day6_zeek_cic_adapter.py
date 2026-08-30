"""
Day 6 - Tests for the Zeek -> CIC-IDS2017 adapter and Background label
policy added in ctu_idseval.py.

These are NEW tests covering only the new Day 6 adapter code. They do
not touch, import, or require Day 1-5 code, model artifacts, or the
real CTU-IDSEVAL-6 dataset, so they can run standalone. They are
separate from (and additive to) tests/test_day6_ctu_idseval.py, which
covers the rest of the Day 6 pipeline and was not modified here.

NOTE ON PROVENANCE: this file was authored and executed against the
adapter code in this same change, using synthetic Zeek-conn-log-shaped
rows (not the real CTU-IDSEVAL-6 dataset, which was not available in
the environment this was authored in). All assertions below passed
when run. This does NOT substitute for running it against the real
tests/test_day6_ctu_idseval.py and the real dataset -- do that too.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.day6.ctu_idseval import (
    BACKGROUND_POLICY_CHOICES,
    apply_background_policy,
    derive_cic_features_from_zeek,
)


@pytest.fixture
def zeek_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "duration": [1.5, 0.0, 2.0],
            "orig_bytes": [1000, 500, 0],
            "resp_bytes": [2000, 0, 100],
            "orig_pkts": [10, 5, 0],
            "resp_pkts": [8, 0, 2],
            "proto": ["tcp", "udp", "tcp"],
            "orig_label": ["Benign", "Malicious", "Background"],
        }
    )


def test_direct_mappings_use_correct_zeek_fields(zeek_df):
    out, report = derive_cic_features_from_zeek(zeek_df)
    assert out.loc[0, "Total Fwd Packets"] == 10
    assert out.loc[0, "Total Backward Packets"] == 8
    assert out.loc[0, "Total Length of Fwd Packets"] == 1000
    assert out.loc[0, "Total Length of Bwd Packets"] == 2000
    assert set(report.directly_mapped) == {
        "Flow Duration",
        "Total Fwd Packets",
        "Total Backward Packets",
        "Total Length of Fwd Packets",
        "Total Length of Bwd Packets",
    }


def test_flow_duration_unit_conversion_is_explicit_and_documented(zeek_df):
    out, report = derive_cic_features_from_zeek(zeek_df, duration_multiplier=1_000_000.0)
    assert out.loc[0, "Flow Duration"] == 1.5 * 1_000_000.0
    assert report.duration_unit_conversion["verified_against_this_projects_actual_data"] is False
    # Multiplier is overridable, e.g. once the real unit is confirmed against
    # data/processed/day1/split_metadata.json, without touching adapter logic.
    out_seconds, _ = derive_cic_features_from_zeek(zeek_df, duration_multiplier=1.0)
    assert out_seconds.loc[0, "Flow Duration"] == 1.5


def test_derived_rate_features_use_raw_seconds_duration(zeek_df):
    out, report = derive_cic_features_from_zeek(zeek_df)
    # row 0: (1000+2000)/1.5
    assert out.loc[0, "Flow Bytes/s"] == pytest.approx((1000 + 2000) / 1.5)
    assert out.loc[0, "Flow Packets/s"] == pytest.approx((10 + 8) / 1.5)
    assert out.loc[0, "Fwd Packets/s"] == pytest.approx(10 / 1.5)
    assert out.loc[0, "Bwd Packets/s"] == pytest.approx(8 / 1.5)
    assert out.loc[0, "Fwd Packet Length Mean"] == pytest.approx(1000 / 10)
    assert out.loc[0, "Bwd Packet Length Mean"] == pytest.approx(2000 / 8)


def test_zero_duration_produces_inf_not_a_guess(zeek_df):
    # Row 1 has duration == 0.0. The adapter must NOT silently zero-fill
    # or guess a value -- it must produce +inf so the existing, frozen
    # src.day5.feature_mapping.apply_feature_mapping sanitizer (unchanged)
    # is the single place that decides how to handle it (impute with the
    # CIC-IDS2017 training median).
    out, _ = derive_cic_features_from_zeek(zeek_df)
    assert math.isinf(out.loc[1, "Flow Bytes/s"])
    assert math.isinf(out.loc[1, "Flow Packets/s"])


def test_no_disallowed_packet_statistics_are_fabricated(zeek_df):
    out, report = derive_cic_features_from_zeek(zeek_df)
    disallowed = {
        "Fwd Packet Length Std", "Fwd Packet Length Max", "Fwd Packet Length Min",
        "Bwd Packet Length Std", "Bwd Packet Length Max", "Bwd Packet Length Min",
        "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
        "SYN Flag Count", "FIN Flag Count", "URG Flag Count",
    }
    assert disallowed.isdisjoint(out.columns)
    assert disallowed.isdisjoint(set(report.directly_mapped) | set(report.derived))


def test_missing_zeek_field_is_skipped_and_reported_not_fabricated():
    df = pd.DataFrame({"duration": [1.0], "orig_pkts": [5]})  # no *_bytes fields at all
    out, report = derive_cic_features_from_zeek(df)
    assert "Total Length of Fwd Packets" not in out.columns
    assert "Total Length of Fwd Packets" in report.skipped_missing_zeek_fields
    assert report.skipped_missing_zeek_fields["Total Length of Fwd Packets"] == ["orig_bytes"]


def test_background_excluded_by_default(zeek_df):
    df = zeek_df.copy()
    # Simulate what a naive two-class normalize_labels would have done:
    # Background incorrectly folded into attack (label_binary=1) purely
    # because it isn't "benign".
    df["label_binary"] = [0, 1, 1]
    out, policy_report = apply_background_policy(df, original_label_col="orig_label")
    assert len(out) == 2
    assert "Background" not in out["orig_label"].values
    assert policy_report["policy_applied"] == "exclude"
    assert policy_report["n_background_rows_found"] == 1


@pytest.mark.parametrize("policy,expected_label", [
    ("treat_as_benign", 0),
    ("treat_as_malicious", 1),
])
def test_background_policy_alternatives_are_explicit_not_silent(zeek_df, policy, expected_label):
    df = zeek_df.copy()
    df["label_binary"] = [0, 1, 1]
    out, policy_report = apply_background_policy(df, original_label_col="orig_label", policy=policy)
    assert len(out) == 3  # kept, not dropped
    bg_row = out[out["orig_label"] == "Background"]
    assert bg_row["label_binary"].iloc[0] == expected_label
    assert policy_report["policy_applied"] == policy


def test_unknown_background_policy_rejected(zeek_df):
    df = zeek_df.copy()
    df["label_binary"] = [0, 1, 1]
    with pytest.raises(ValueError):
        apply_background_policy(df, original_label_col="orig_label", policy="not_a_real_policy")


def test_background_policy_choices_constant_matches_cli_choices():
    # Guards against ctu_idseval.py and run_day6.py's argparse --background-policy
    # choices silently drifting apart.
    assert set(BACKGROUND_POLICY_CHOICES) == {"exclude", "treat_as_benign", "treat_as_malicious"}