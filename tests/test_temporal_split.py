"""Lightweight unit tests for src.data.temporal_split using tiny synthetic data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.temporal_split import (
    chronological_train_test_split,
    detect_timestamp_column,
    verify_no_temporal_leakage,
)


def _make_synthetic_timeseries(n_rows: int = 100) -> pd.DataFrame:
    timestamps = pd.date_range("2017-07-01", periods=n_rows, freq="min")
    return pd.DataFrame(
        {
            "Timestamp": timestamps,
            "Feature A": range(n_rows),
            "Label": ["BENIGN" if i % 3 else "DoS Hulk" for i in range(n_rows)],
        }
    )


def test_detect_timestamp_column_by_name():
    df = _make_synthetic_timeseries(10)
    result = detect_timestamp_column(df)
    assert result.column == "Timestamp"
    assert result.parsed_successfully


def test_detect_timestamp_column_explicit_override():
    df = _make_synthetic_timeseries(10)
    df = df.rename(columns={"Timestamp": "flow_start_time"})
    result = detect_timestamp_column(df, explicit_column="flow_start_time")
    assert result.column == "flow_start_time"


def test_detect_timestamp_column_none_found():
    df = pd.DataFrame({"A": [1, 2, 3]})
    result = detect_timestamp_column(df)
    assert result.column is None
    assert not result.parsed_successfully


def test_detect_timestamp_column_missing_explicit_raises():
    df = pd.DataFrame({"A": [1, 2, 3]})
    with pytest.raises(KeyError):
        detect_timestamp_column(df, explicit_column="does_not_exist")


def test_chronological_split_produces_strict_time_order():
    df = _make_synthetic_timeseries(100)
    split = chronological_train_test_split(df, timestamp_column="Timestamp", test_size=0.2)

    assert len(split.train_df) + len(split.test_df) == 100
    assert len(split.test_df) == 20
    assert split.train_end <= split.test_start


def test_chronological_split_invalid_test_size_raises():
    df = _make_synthetic_timeseries(10)
    with pytest.raises(ValueError):
        chronological_train_test_split(df, timestamp_column="Timestamp", test_size=1.5)


def test_chronological_split_missing_column_raises():
    df = _make_synthetic_timeseries(10).drop(columns=["Timestamp"])
    with pytest.raises(KeyError):
        chronological_train_test_split(df, timestamp_column="Timestamp", test_size=0.2)


def test_chronological_split_unparseable_timestamps_raise():
    df = _make_synthetic_timeseries(10)
    # Cast to object dtype first so a bad string can be assigned; a native
    # datetime64 column would reject the assignment outright.
    df["Timestamp"] = df["Timestamp"].astype(object)
    df.loc[3, "Timestamp"] = "not-a-real-date"
    with pytest.raises(ValueError):
        chronological_train_test_split(df, timestamp_column="Timestamp", test_size=0.2)


def test_verify_no_temporal_leakage_passes_for_valid_split():
    df = _make_synthetic_timeseries(100)
    split = chronological_train_test_split(df, timestamp_column="Timestamp", test_size=0.2)
    check = verify_no_temporal_leakage(split)
    assert check.passed
    assert check.n_overlapping_rows == 0


def test_verify_no_temporal_leakage_fails_for_shuffled_split():
    df = _make_synthetic_timeseries(100)
    split = chronological_train_test_split(df, timestamp_column="Timestamp", test_size=0.2)

    # Deliberately corrupt the split to simulate a leakage bug: swap one
    # late train-row's timestamp with an early test-row's timestamp so
    # they overlap.
    bad_train = split.train_df.copy()
    bad_test = split.test_df.copy()
    bad_train.iloc[-1, bad_train.columns.get_loc("Timestamp")] = bad_test.iloc[0]["Timestamp"]

    split.train_df = bad_train
    check = verify_no_temporal_leakage(split)
    assert not check.passed
    assert check.n_overlapping_rows > 0


# ---------------------------------------------------------------------
# Day-based (file-identity) split tests
# ---------------------------------------------------------------------

from src.data.temporal_split import (  # noqa: E402
    CIC_IDS2017_DAY_ORDER,
    assign_capture_days,
    day_based_train_test_split,
    detect_capture_day,
    label_chunk_with_capture_day,
    verify_no_day_leakage,
)


def test_detect_capture_day_matches_known_filenames():
    assert detect_capture_day("Monday-WorkingHours.pcap_ISCX.csv") == "monday"
    assert detect_capture_day("Tuesday-WorkingHours.pcap_ISCX.csv") == "tuesday"
    assert detect_capture_day("Wednesday-workingHours.pcap_ISCX.csv") == "wednesday"
    assert detect_capture_day("Thursday-WorkingHours-Morning-WebAttacks.csv") == "thursday"
    assert detect_capture_day("Friday-WorkingHours-Afternoon-DDos.parquet") == "friday"


def test_detect_capture_day_matches_actual_project_filenames():
    # The exact 8 CIC-IDS2017 Parquet filenames used in this project's
    # real dataset (verified against the maintainer's handoff notes).
    # Kept as a literal, additive regression test so a future change to
    # the day-detection regex cannot silently break these specific files
    # without a test failing.
    expected = {
        "Benign-Monday-no-metadata.parquet": "monday",
        "Bruteforce-Tuesday-no-metadata.parquet": "tuesday",
        "DoS-Wednesday-no-metadata.parquet": "wednesday",
        "Infiltration-Thursday-no-metadata.parquet": "thursday",
        "WebAttacks-Thursday-no-metadata.parquet": "thursday",
        "Botnet-Friday-no-metadata.parquet": "friday",
        "DDoS-Friday-no-metadata.parquet": "friday",
        "Portscan-Friday-no-metadata.parquet": "friday",
    }
    for filename, expected_day in expected.items():
        assert detect_capture_day(filename) == expected_day, (
            f"{filename!r} should map to {expected_day!r}"
        )


def test_assign_capture_days_matches_actual_project_filenames():
    # Same 8 real filenames, run through assign_capture_days (the
    # higher-level function actually used by prepare_data.py /
    # notebook 03) to confirm the full-batch day assignment, distinct
    # days present, and that every file is detected -- none left in
    # undetected_files.
    files = [
        Path("Benign-Monday-no-metadata.parquet"),
        Path("Bruteforce-Tuesday-no-metadata.parquet"),
        Path("DoS-Wednesday-no-metadata.parquet"),
        Path("Infiltration-Thursday-no-metadata.parquet"),
        Path("WebAttacks-Thursday-no-metadata.parquet"),
        Path("Botnet-Friday-no-metadata.parquet"),
        Path("DDoS-Friday-no-metadata.parquet"),
        Path("Portscan-Friday-no-metadata.parquet"),
    ]

    assignment = assign_capture_days(files)

    assert assignment.all_detected
    assert assignment.undetected_files == []
    assert assignment.distinct_days == ["monday", "tuesday", "wednesday", "thursday", "friday"]

    expected_by_name = {
        "Benign-Monday-no-metadata.parquet": "monday",
        "Bruteforce-Tuesday-no-metadata.parquet": "tuesday",
        "DoS-Wednesday-no-metadata.parquet": "wednesday",
        "Infiltration-Thursday-no-metadata.parquet": "thursday",
        "WebAttacks-Thursday-no-metadata.parquet": "thursday",
        "Botnet-Friday-no-metadata.parquet": "friday",
        "DDoS-Friday-no-metadata.parquet": "friday",
        "Portscan-Friday-no-metadata.parquet": "friday",
    }
    for f in files:
        assert assignment.file_days[str(f)] == expected_by_name[f.name]


def test_detect_capture_day_returns_none_when_absent():
    assert detect_capture_day("merged_dataset_part_00001.parquet") is None
    assert detect_capture_day("flow_data.csv") is None


def test_detect_capture_day_does_not_false_positive_on_substrings():
    # "mon" inside another word should not match "monday"; only a real,
    # word-bounded day name should match.
    assert detect_capture_day("money_flow_stats.csv") is None


def test_assign_capture_days_all_detected():
    files = [Path("Monday-WorkingHours.csv"), Path("Friday-Afternoon-DDos.csv")]
    result = assign_capture_days(files)
    assert result.all_detected
    assert result.distinct_days == ["monday", "friday"]


def test_assign_capture_days_reports_undetected_without_guessing():
    files = [Path("Monday-WorkingHours.csv"), Path("mystery_file.csv")]
    result = assign_capture_days(files)
    assert not result.all_detected
    assert str(files[1]) in result.undetected_files


def test_label_chunk_with_capture_day_success():
    chunk = pd.DataFrame({"a": [1, 2, 3]})
    chunk.attrs["source_file"] = "Tuesday-WorkingHours.pcap_ISCX.csv"
    labeled = label_chunk_with_capture_day(chunk)
    assert (labeled["capture_day"] == "tuesday").all()


def test_label_chunk_with_capture_day_missing_source_file_raises():
    chunk = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(KeyError):
        label_chunk_with_capture_day(chunk)


def test_label_chunk_with_capture_day_undetectable_day_raises():
    chunk = pd.DataFrame({"a": [1, 2, 3]})
    chunk.attrs["source_file"] = "merged_all_days.parquet"
    with pytest.raises(ValueError):
        label_chunk_with_capture_day(chunk)


def _make_day_labeled_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": range(10),
            "capture_day": ["monday"] * 4 + ["wednesday"] * 3 + ["friday"] * 3,
        }
    )


def test_day_based_split_default_train_days():
    df = _make_day_labeled_df()
    split = day_based_train_test_split(df, day_column="capture_day", test_days=["friday"])
    assert split.train_days == ["monday", "wednesday"]
    assert split.test_days == ["friday"]
    assert len(split.train_df) == 7
    assert len(split.test_df) == 3


def test_day_based_split_missing_day_column_raises():
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(KeyError):
        day_based_train_test_split(df, day_column="capture_day", test_days=["friday"])


def test_day_based_split_unknown_day_name_raises():
    df = _make_day_labeled_df()
    with pytest.raises(ValueError):
        day_based_train_test_split(df, day_column="capture_day", test_days=["notaday"])


def test_day_based_split_overlapping_days_raise():
    df = _make_day_labeled_df()
    with pytest.raises(ValueError):
        day_based_train_test_split(
            df, day_column="capture_day", test_days=["friday"], train_days=["friday"]
        )


def test_day_based_split_rejects_out_of_order_days():
    df = _make_day_labeled_df()
    with pytest.raises(ValueError):
        day_based_train_test_split(
            df, day_column="capture_day", test_days=["monday"], train_days=["friday"]
        )


def test_verify_no_day_leakage_passes_for_valid_split():
    df = _make_day_labeled_df()
    split = day_based_train_test_split(df, day_column="capture_day", test_days=["friday"])
    check = verify_no_day_leakage(split)
    assert check.passed


def test_cic_ids2017_day_order_is_monday_to_friday():
    assert CIC_IDS2017_DAY_ORDER == ("monday", "tuesday", "wednesday", "thursday", "friday")
