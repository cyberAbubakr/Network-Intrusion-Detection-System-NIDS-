"""
Day 6 - CTU-IDSEVAL-6 dataset discovery and loading.

This module intentionally contains ONLY the minimum CTU-specific glue
that Day 5's implementation does not already expose as an importable,
dataset-agnostic function (Day 5's chunked CSV reader is a private
helper inside scripts/run_day5.py, which this project's Day 5 code is
not to be modified to expose). Everything else -- feature mapping,
label mapping, frozen-model scoring, frozen-threshold metrics -- is
imported unchanged from src.day2, src.day3, and src.day5 in
scripts/run_day6.py.

No schema is assumed or invented here beyond "one or more CSV files,
each with a label column somewhere". CTU-IDSEVAL-6's actual column
names are matched to the frozen CIC-IDS2017 feature space by
src.day5.feature_mapping's normalized-token matching, which does not
require knowing the external schema in advance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_CTU_DIR_NAME = "ctu_idseval6"
DEFAULT_CSV_CHUNK_SIZE = 50_000

# ---------------------------------------------------------------------------
# Day 6 - Zeek conn.log -> CIC-IDS2017 feature-name adapter.
#
# CTU-IDSEVAL-6's Zeek `.conn-labeled.log` files use Zeek's own conn.log
# schema (duration, orig_bytes, resp_bytes, orig_pkts, resp_pkts, ...),
# not CICFlowMeter's CIC-IDS2017 column names. src.day5.feature_mapping's
# normalized-token matcher (built for two CICFlowMeter-family datasets
# that spell the same measurement differently, e.g. "Packets" vs "Pkts")
# has nothing to match against Zeek names at all, hence 0/58 mapped.
#
# This adapter does NOT change src.day5.feature_mapping. It runs strictly
# BEFORE build_feature_mapping/apply_feature_mapping and adds a small set
# of extra columns to the external DataFrame, spelled with the exact
# standard CICFlowMeter/CIC-IDS2017 feature names, so that the existing,
# unmodified normalized-token matcher in src.day5.feature_mapping can pick
# them up on its own. Any of these 11 candidate names that do not happen
# to be among the frozen 58 are simply ignored downstream (extra columns
# are harmless); any of the frozen 58 NOT covered by this list continue
# to be imputed with CIC-IDS2017 training medians exactly as the existing
# frozen methodology already does for wholly-unmapped features.
#
# Only measurements Zeek's conn.log genuinely supports are derived here.
# Zeek conn.log has no packet-length distribution (std/var/min/max), no
# inter-arrival-time distribution, and no TCP flag counts, so none of the
# corresponding CIC-IDS2017 features (e.g. "Fwd Packet Length Std",
# "Flow IAT Mean", "SYN Flag Count") are attempted here -- they remain
# imputed, per the "do not guess packet-level statistics that Zeek
# conn.log cannot provide" requirement.
# ---------------------------------------------------------------------------

# Zeek `duration` is documented as seconds (Zeek `interval` type).
# CICFlowMeter's own C++/Java implementation records "Flow Duration" in
# MICROSECONDS (this is CICFlowMeter's actual, if non-obvious, unit
# convention, independent of any one dataset's exported CSV). Absent
# access to this project's own data/processed/day1 prepare_data.py /
# split_metadata.json to directly confirm the exact unit this frozen
# model was trained on, this adapter defaults to microseconds -- the
# documented CICFlowMeter convention -- and makes the conversion an
# explicit, named constant rather than a silent multiplication, so it is
# trivially auditable and correctable.
#
# ACTION REQUIRED before trusting "Flow Duration" results: confirm this
# against data/processed/day1/split_metadata.json / prepare_data.py (e.g.
# check whether Monday-Friday CIC-IDS2017 "Flow Duration" values are on
# the order of 1e5-1e7 (microseconds) or 0-a few hundred (seconds)) and
# update ZEEK_DURATION_TO_CIC_DURATION_MULTIPLIER below if this project's
# CIC-IDS2017 data uses seconds instead.
ZEEK_DURATION_UNIT = "seconds"                 # Zeek's own documented unit for `duration`
CIC_FLOW_DURATION_UNIT_ASSUMED = "microseconds"  # CICFlowMeter's documented unit; UNVERIFIED against this project's data
ZEEK_DURATION_TO_CIC_DURATION_MULTIPLIER = 1_000_000.0  # seconds -> microseconds

# CIC-IDS2017 feature names this adapter can populate directly from a
# single Zeek field, with no arithmetic beyond a unit conversion.
_DIRECT_ZEEK_TO_CIC: dict[str, str] = {
    "Flow Duration": "duration",              # unit-converted, see above
    "Total Fwd Packets": "orig_pkts",
    "Total Backward Packets": "resp_pkts",
    "Total Length of Fwd Packets": "orig_bytes",
    "Total Length of Bwd Packets": "resp_bytes",
}

# CIC-IDS2017 feature names this adapter derives via simple, semantically
# defensible arithmetic over Zeek fields (rates and per-packet means).
# Rate features use RAW Zeek `duration` (seconds), not the unit-converted
# "Flow Duration" column, so rates are expressed per-second regardless of
# CIC_FLOW_DURATION_UNIT_ASSUMED.
_DERIVED_CIC_FEATURES: tuple[str, ...] = (
    "Flow Bytes/s",
    "Flow Packets/s",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Mean",
)

# Zeek fields this adapter reads. Missing fields are handled gracefully:
# any candidate CIC feature that needs a field not present in this
# CTU-IDSEVAL-6 export is simply skipped (and reported as skipped), never
# fabricated from an absent column.
_REQUIRED_ZEEK_FIELDS_BY_CIC_FEATURE: dict[str, tuple[str, ...]] = {
    "Flow Duration": ("duration",),
    "Total Fwd Packets": ("orig_pkts",),
    "Total Backward Packets": ("resp_pkts",),
    "Total Length of Fwd Packets": ("orig_bytes",),
    "Total Length of Bwd Packets": ("resp_bytes",),
    "Flow Bytes/s": ("orig_bytes", "resp_bytes", "duration"),
    "Flow Packets/s": ("orig_pkts", "resp_pkts", "duration"),
    "Fwd Packets/s": ("orig_pkts", "duration"),
    "Bwd Packets/s": ("resp_pkts", "duration"),
    "Fwd Packet Length Mean": ("orig_bytes", "orig_pkts"),
    "Bwd Packet Length Mean": ("resp_bytes", "resp_pkts"),
}


@dataclass
class ZeekCicAdapterReport:
    """Record of exactly what this adapter did and did not populate."""

    directly_mapped: list = field(default_factory=list)
    derived: list = field(default_factory=list)
    skipped_missing_zeek_fields: dict = field(default_factory=dict)
    zeek_fields_used: list = field(default_factory=list)
    duration_unit_conversion: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "directly_mapped_cic_features": sorted(self.directly_mapped),
            "n_directly_mapped": len(self.directly_mapped),
            "derived_cic_features": sorted(self.derived),
            "n_derived": len(self.derived),
            "skipped_missing_zeek_fields": self.skipped_missing_zeek_fields,
            "n_skipped": len(self.skipped_missing_zeek_fields),
            "zeek_fields_used": sorted(set(self.zeek_fields_used)),
            "duration_unit_conversion": self.duration_unit_conversion,
            "note": (
                "This is the full set of CIC-IDS2017 features Zeek conn.log "
                "can legitimately support (direct field or simple rate/mean "
                "arithmetic). All remaining frozen CIC-IDS2017 features "
                "(packet-length std/var/min/max, IAT distributions, TCP flag "
                "counts, etc.) are NOT reconstructable from Zeek conn.log and "
                "are imputed downstream with CIC-IDS2017 Day 1 training "
                "medians by src.day5.feature_mapping.apply_feature_mapping, "
                "unchanged."
            ),
        }


def derive_cic_features_from_zeek(
    external_df: pd.DataFrame,
    duration_multiplier: float = ZEEK_DURATION_TO_CIC_DURATION_MULTIPLIER,
) -> tuple[pd.DataFrame, ZeekCicAdapterReport]:
    """
    Add CIC-IDS2017-named columns to a Zeek-derived external DataFrame,
    computed only from fields Zeek conn.log genuinely provides.

    Does not modify or remove any existing column, does not touch
    src.day5.feature_mapping, and does not compute anything from
    CTU-IDSEVAL-6 that is used as a preprocessing STATISTIC (this is
    per-row arithmetic on CTU-IDSEVAL-6's own already-observed field
    values, not a fitted statistic borrowed from CTU-IDSEVAL-6 -- no
    CTU-derived medians/means/scales are introduced here).

    Returns the augmented DataFrame plus a report of exactly which CIC
    features were populated directly, which were derived, and which were
    skipped because a required Zeek field was absent from this export.
    """

    out = external_df.copy()
    report = ZeekCicAdapterReport(
        duration_unit_conversion={
            "zeek_duration_unit": ZEEK_DURATION_UNIT,
            "cic_flow_duration_unit_assumed": CIC_FLOW_DURATION_UNIT_ASSUMED,
            "multiplier_applied_seconds_to_target": duration_multiplier,
            "verified_against_this_projects_actual_data": False,
            "action_required": (
                "Confirm CIC-IDS2017 'Flow Duration' unit against "
                "data/processed/day1/split_metadata.json / prepare_data.py "
                "and adjust ZEEK_DURATION_TO_CIC_DURATION_MULTIPLIER if this "
                "project's CIC-IDS2017 data is in seconds rather than "
                "microseconds."
            ),
        },
    )

    available = set(external_df.columns)

    def _numeric(field_name: str) -> pd.Series:
        return pd.to_numeric(external_df[field_name], errors="coerce")

    def _has_fields(cic_feature: str) -> bool:
        needed = _REQUIRED_ZEEK_FIELDS_BY_CIC_FEATURE[cic_feature]
        missing = [f for f in needed if f not in available]
        if missing:
            report.skipped_missing_zeek_fields[cic_feature] = missing
            return False
        report.zeek_fields_used.extend(needed)
        return True

    # --- Direct mappings (single Zeek field, unit conversion only) -----
    if _has_fields("Flow Duration"):
        out["Flow Duration"] = _numeric("duration") * duration_multiplier
        report.directly_mapped.append("Flow Duration")

    if _has_fields("Total Fwd Packets"):
        out["Total Fwd Packets"] = _numeric("orig_pkts")
        report.directly_mapped.append("Total Fwd Packets")

    if _has_fields("Total Backward Packets"):
        out["Total Backward Packets"] = _numeric("resp_pkts")
        report.directly_mapped.append("Total Backward Packets")

    if _has_fields("Total Length of Fwd Packets"):
        out["Total Length of Fwd Packets"] = _numeric("orig_bytes")
        report.directly_mapped.append("Total Length of Fwd Packets")

    if _has_fields("Total Length of Bwd Packets"):
        out["Total Length of Bwd Packets"] = _numeric("resp_bytes")
        report.directly_mapped.append("Total Length of Bwd Packets")

    # --- Derived (rates use RAW seconds-duration; means use raw counts) -
    # 0-duration / 0-packet flows legitimately produce +inf via division;
    # this is intentionally left as +inf here rather than guarded, because
    # src.day5.feature_mapping.apply_feature_mapping already detects and
    # sanitizes +inf -> CIC-IDS2017 training median downstream (unchanged,
    # frozen behavior) -- duplicating that guard here would diverge from
    # the single frozen sanitization path.
    duration_s = _numeric("duration") if "duration" in available else None

    if _has_fields("Flow Bytes/s"):
        out["Flow Bytes/s"] = (_numeric("orig_bytes") + _numeric("resp_bytes")) / duration_s
        report.derived.append("Flow Bytes/s")

    if _has_fields("Flow Packets/s"):
        out["Flow Packets/s"] = (_numeric("orig_pkts") + _numeric("resp_pkts")) / duration_s
        report.derived.append("Flow Packets/s")

    if _has_fields("Fwd Packets/s"):
        out["Fwd Packets/s"] = _numeric("orig_pkts") / duration_s
        report.derived.append("Fwd Packets/s")

    if _has_fields("Bwd Packets/s"):
        out["Bwd Packets/s"] = _numeric("resp_pkts") / duration_s
        report.derived.append("Bwd Packets/s")

    if _has_fields("Fwd Packet Length Mean"):
        out["Fwd Packet Length Mean"] = _numeric("orig_bytes") / _numeric("orig_pkts")
        report.derived.append("Fwd Packet Length Mean")

    if _has_fields("Bwd Packet Length Mean"):
        out["Bwd Packet Length Mean"] = _numeric("resp_bytes") / _numeric("resp_pkts")
        report.derived.append("Bwd Packet Length Mean")

    logger.info(
        "Zeek->CIC adapter: %d feature(s) directly mapped, %d derived, %d "
        "skipped (missing Zeek field(s)). See ZeekCicAdapterReport.summary() "
        "for exact names.",
        len(report.directly_mapped), len(report.derived), len(report.skipped_missing_zeek_fields),
    )

    return out, report


# ---------------------------------------------------------------------------
# Day 6 - Background/Benign/Malicious label policy.
#
# CTU-IDSEVAL-6 follows the CTU/Stratosphere-family (CTU-13, CTU-SME-11,
# etc.) three-way labeling convention: Benign, Malicious, and Background.
# In that family's documented convention, "Background" denotes traffic
# not confirmed as either an attack against a monitored victim or benign
# activity BY a monitored victim -- e.g. third-party/incidental traffic
# transiting the capture that was never manually verified either way. It
# is emphatically NOT a synonym for "not benign" / "attack", and treating
# it as attack by default (e.g. "anything != 'Benign' counts as attack")
# would inject unverified traffic into the attack class and bias
# precision/recall in an unknown direction.
#
# This project's own official CTU-IDSEVAL-6 documentation was not
# available in this environment (no dataset README/data dictionary was
# uploaded), so the policy below follows the documented CTU/Stratosphere
# family convention rather than inventing a rule. If CTU-IDSEVAL-6 ships
# its own README/data-dictionary defining Background differently, that
# document should override this default -- see EXCLUDE_BACKGROUND_DEFAULT.
#
# Default policy: EXCLUDE Background rows from binary attack/benign
# metrics (they are neither a verified positive nor a verified negative),
# while still reporting their count and, if per-class breakdowns are
# produced, their raw detector scores separately. This is the most
# defensible default given unverified labels, and is applied explicitly
# and loggably -- never via silent string matching against "Benign".
# ---------------------------------------------------------------------------

BACKGROUND_LABEL_ALIASES = {"background"}

BACKGROUND_POLICY_CHOICES = ("exclude", "treat_as_benign", "treat_as_malicious")
DEFAULT_BACKGROUND_POLICY = "exclude"


def apply_background_policy(
    df: pd.DataFrame,
    original_label_col: str,
    label_binary_col: str = "label_binary",
    policy: str = DEFAULT_BACKGROUND_POLICY,
    background_aliases: set = BACKGROUND_LABEL_ALIASES,
) -> tuple[pd.DataFrame, dict]:
    """
    Apply an explicit Background/Benign/Malicious label policy for binary
    evaluation, instead of relying on whatever src.data.cleaner.normalize_labels
    happened to assign to a "Background" string (which -- being written for
    a two-class Benign/Attack world -- would otherwise fall through to
    "not benign" = attack).

    Parameters
    ----------
    df:
        DataFrame after src.day5.label_mapping.map_external_labels has
        already run (has ``label_binary`` populated by the existing,
        unmodified Day 1 logic).

    original_label_col:
        The ORIGINAL raw CTU-IDSEVAL-6 label column (pre-normalization),
        so Background rows can be identified from the source string
        rather than from ``label_binary`` (which may have already
        (mis)classified them as attack).

    policy:
        One of ``BACKGROUND_POLICY_CHOICES``:
          - "exclude": drop Background rows from the binary-metrics
            DataFrame entirely (default; recommended).
          - "treat_as_benign": keep Background rows, force
            ``label_binary=0``.
          - "treat_as_malicious": keep Background rows, force
            ``label_binary=1``.
        Regardless of policy, this function is the ONLY place Background
        rows are reassigned -- there is no other, implicit path.

    Returns
    -------
    (df_for_binary_metrics, policy_report)
        ``df_for_binary_metrics`` has the chosen policy applied.
        ``policy_report`` documents the policy, counts, and rationale for
        inclusion verbatim in Day 6 output artifacts.
    """

    if policy not in BACKGROUND_POLICY_CHOICES:
        raise ValueError(f"Unknown background policy {policy!r}; choose one of {BACKGROUND_POLICY_CHOICES}")

    is_background = (
        df[original_label_col].astype(str).str.strip().str.lower().isin(background_aliases)
    )
    n_background = int(is_background.sum())
    n_total = len(df)

    if policy == "exclude":
        out = df.loc[~is_background].copy()
    elif policy == "treat_as_benign":
        out = df.copy()
        out.loc[is_background, label_binary_col] = 0
    else:  # treat_as_malicious
        out = df.copy()
        out.loc[is_background, label_binary_col] = 1

    report = {
        "policy_applied": policy,
        "background_label_aliases_matched": sorted(background_aliases),
        "n_total_rows_before_policy": n_total,
        "n_background_rows_found": n_background,
        "n_rows_used_for_binary_metrics_after_policy": len(out),
        "rationale": (
            "CTU/Stratosphere-family convention: Background denotes traffic "
            "not confirmed as attack-on-victim or benign-from-victim, i.e. "
            "unverified either way -- not a synonym for 'attack'. Default "
            "policy excludes it from binary precision/recall/F1 rather than "
            "silently folding it into the attack class via string "
            "mismatch against 'Benign'."
        ),
    }

    logger.info(
        "Background label policy: %s. %d/%d rows identified as Background; "
        "%d rows remain for binary metrics.",
        policy, n_background, n_total, len(out),
    )

    return out, report


def discover_ctu_csvs(external_dir: Path) -> list[Path]:
    """Return all *.csv files in ``external_dir``, sorted; empty list if the directory doesn't exist or has none."""
    if not external_dir.exists():
        return []
    return sorted(external_dir.glob("*.csv"))


def load_ctu_csvs_chunked(
    csv_paths: Sequence[Path],
    chunk_size: int = DEFAULT_CSV_CHUNK_SIZE,
) -> pd.DataFrame:
    """
    Read one or more CTU-IDSEVAL-6 CSV files in row-chunks (reduces
    peak memory during parsing for potentially large files) and
    concatenate into a single DataFrame. All rows from all files are
    included -- this is a memory-friendlier read, not a sample.
    Mirrors the approach already used for CSE-CIC-IDS2018 in
    scripts/run_day5.py.
    """

    chunks = []
    total_rows = 0
    for csv_path in csv_paths:
        file_rows = 0
        for chunk in pd.read_csv(csv_path, low_memory=False, chunksize=chunk_size):
            chunk.columns = [c.strip() for c in chunk.columns]
            chunks.append(chunk)
            file_rows += len(chunk)
        total_rows += file_rows
        logger.info("Read %d row(s) from %s (chunksize=%d).", file_rows, csv_path, chunk_size)

    df = pd.concat(chunks, ignore_index=True)
    del chunks
    logger.info("CTU-IDSEVAL-6 dataset loaded: %d rows, %d columns.", len(df), df.shape[1])
    return df