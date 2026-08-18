# Cross-Temporal Hybrid Network Intrusion Detection

**Day 1 scope:** data pipeline (discovery -> load -> validate -> clean ->
feature engineering -> temporal split -> leakage verification) + a
lightweight Random Forest baseline with standard evaluation metrics.

Day 2+ (Isolation Forest, hybrid fusion, SHAP explainability, LLM/RAG
reporting, CSE-CIC-IDS2018 / CTU-IDSEVAL-6 / BigFlow-NIDS datasets) is
**out of scope for this codebase** and is not implemented here.

## Temporal splitting: two mechanisms, chosen automatically

Not every CIC-IDS2017 distribution retains a row-level `Timestamp`
column — some cleaned/re-hosted copies drop it along with Flow ID/IPs/
ports. `scripts/prepare_data.py` handles both cases explicitly, and
records which one was actually used in `split_metadata.json`
(`"split_method"`):

* **`row_timestamp`** (primary path) — used automatically when a real
  timestamp column is detected. Row-level chronological split; see
  `src/data/temporal_split.py`'s `chronological_train_test_split` /
  `verify_no_temporal_leakage`.
* **`capture_day`** (fallback) — used automatically when no timestamp
  column exists. Splits by the dataset's original per-capture-day file
  identity (Monday..Friday), inferred **only** from filenames, never
  guessed. Because CIC-IDS2017 attacks are day-specific by design, train
  and test days will not share the same class distribution — this is an
  inherent property of the dataset and is documented in
  `notebooks/03_temporal_split.ipynb` / `04_baseline_model.ipynb`, not
  concealed. If day identity can't be recovered from filenames either,
  the pipeline fails loudly rather than fabricating an ordering. See
  `src/data/temporal_split.py`'s `detect_capture_day` /
  `day_based_train_test_split` / `verify_no_day_leakage`.

Neither mechanism ever infers time from flow-statistical features (e.g.
`Flow IAT Mean`/`Flow IAT Min`) — those describe inter-arrival timing
within a flow, not when the flow occurred, and are never used as a
timestamp substitute.

## Status

This repository currently contains **code only** — no dataset has been
downloaded or processed, and no model has been trained on real data.
Every script that touches real data must be run manually by you; nothing
runs automatically.

## Project layout

```
cross-temporal-hybrid-nids/
├── data/
│   ├── raw/CIC-IDS2017/{parquet,csv}/   # put the real dataset files here
│   └── processed/day1/                  # pipeline output (parts, train/test splits)
├── src/
│   ├── data/
│   │   ├── loader.py              # file discovery + memory-bounded chunked loading
│   │   ├── validator.py           # schema + NaN/Inf validation
│   │   ├── cleaner.py             # label normalization + NaN/Inf handling (logged)
│   │   ├── feature_engineering.py # leakage-prone feature ID + lightweight selection
│   │   └── temporal_split.py      # timestamp OR capture-day detection, split, leakage check
├── scripts/
│   ├── inspect_dataset.py   # schema/NaN-Inf spot check (cheap, run first)
│   ├── prepare_data.py      # full Day 1 preprocessing pipeline (auto timestamp/day fallback)
│   └── train_baseline.py    # Random Forest baseline + evaluation
├── notebooks/
│   ├── 01_dataset_inspection.ipynb    # discovery, schema, labels, missing/inf, timestamps
│   ├── 02_data_preprocessing.ipynb    # loading, validation, leakage ID, feature selection
│   ├── 03_temporal_split.ipynb        # day-based split (this dataset has no row timestamp)
│   └── 04_baseline_model.ipynb        # Random Forest baseline + full evaluation, split-aware
├── models/day1/              # trained model + metadata (created by train_baseline.py)
├── results/day1/             # evaluation metrics (created by train_baseline.py)
├── tests/                    # unit tests using tiny synthetic data only
└── logs/                     # run logs from the scripts above
```

### `src/` vs `scripts/` vs `notebooks/`

* **`src/`** contains the reusable implementation code. This is the only
  place the actual logic lives — everything else calls into it rather
  than duplicating it.
* **`scripts/`** provides command-line execution of that logic end to
  end, meant to be run non-interactively.
* **`notebooks/`** provides research/academic demonstrations of the same
  pipeline — exploration, visualization, step-by-step walkthroughs.
  Notebooks call functions from `src/`; they do not re-implement any
  pipeline logic themselves.

**Status of the notebooks:** they contain properly written cells calling
into `src/`, but have **not** been executed against the real dataset —
no outputs are stored. Run them yourself, in order, after placing the
real CIC-IDS2017 files under `data/raw/CIC-IDS2017/` (with their
original per-day filenames preserved, e.g.
`Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`) and, for
`04_baseline_model.ipynb`, after running `scripts/prepare_data.py`.

## Getting the dataset

**This project never downloads data automatically.** Get CIC-IDS2017
yourself (e.g. from the University of New Brunswick's official
distribution) and place the files under:

```
data/raw/CIC-IDS2017/parquet/*.parquet   # preferred, if you have/convert to parquet
data/raw/CIC-IDS2017/csv/*.csv           # fallback, read in memory-safe chunks
```

You do not need both — the pipeline prefers Parquet automatically and
falls back to chunked CSV reading when only CSV is present.

## Running the pipeline (after you've placed the dataset)

Run these yourself, in order, from the project root:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Cheap sanity check: discovers files, peeks schema, spot-checks NaN/Inf
python scripts/inspect_dataset.py --raw-dir data/raw/CIC-IDS2017

# 2. Full Day 1 preprocessing: clean, feature-select, chronological split
python scripts/prepare_data.py --raw-dir data/raw/CIC-IDS2017 --chunk-size 250000

#    If timestamp auto-detection doesn't find the right column, re-run with:
# python scripts/prepare_data.py --timestamp-column "<exact column name>"

#    If your dataset has NO timestamp column at all, prepare_data.py
#    automatically falls back to day-based splitting (test day defaults
#    to Friday). To choose different days:
# python scripts/prepare_data.py --test-days friday --train-days monday,tuesday,wednesday,thursday

# 3. Train + evaluate the Random Forest baseline
python scripts/train_baseline.py
```

## Resource constraints this codebase is designed for

* Linux, CPU-only, 8 GB RAM, no GPU.
* CSV files are read in configurable chunks (default 250,000 rows);
  the raw dataset is never concatenated into a single in-memory
  DataFrame. See `src/data/loader.py` and the memory-strategy note at
  the top of `scripts/prepare_data.py` for the one documented exception
  (the already-cleaned/reduced dataset is loaded once to perform the
  single global chronological sort a correct temporal split requires).

## Design principles enforced in this codebase

* **No invented columns.** Column names/schema are detected at runtime;
  nothing is hard-coded as "the" CIC-IDS2017 schema. Naming drift across
  files/chunks is logged, not silently patched.
* **No silent feature removal.** Every drop (duplicate columns,
  near-constant columns, leakage-prone columns excluded from the
  feature set) is recorded in a `CleaningLog` and written to
  `logs/prepare_data_decisions.json`.
* **Explicit temporal-leakage verification.** Every chronological split
  is checked (`verify_no_temporal_leakage`) before it is allowed to feed
  into training; a failed check raises, it does not warn-and-continue.

## Tests

All tests use tiny, in-memory synthetic data — no real dataset files are
read anywhere in `tests/`.

```bash
pytest -q
```
