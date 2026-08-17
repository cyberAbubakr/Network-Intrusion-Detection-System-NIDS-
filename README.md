# Cross-Temporal Hybrid Network Intrusion Detection

**Day 1 scope:** data pipeline (discovery -> load -> validate -> clean ->
feature engineering -> chronological split -> leakage verification) +
a lightweight Random Forest baseline with standard evaluation metrics.

Day 2+ (Isolation Forest, hybrid fusion, SHAP explainability, LLM/RAG
reporting, CSE-CIC-IDS2018 / CTU-IDSEVAL-6 / BigFlow-NIDS datasets) is
**out of scope for this codebase** and is not implemented here.

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
│   │   └── temporal_split.py      # timestamp detection, chronological split, leakage check
├── scripts/
│   ├── inspect_dataset.py   # schema/NaN-Inf spot check (cheap, run first)
│   ├── prepare_data.py      # full Day 1 preprocessing pipeline
│   └── train_baseline.py    # Random Forest baseline + evaluation
├── notebooks/
│   ├── 01_dataset_inspection.ipynb    # discovery, schema, labels, missing/inf, timestamps
│   ├── 02_data_preprocessing.ipynb    # loading, validation, leakage ID, feature selection
│   ├── 03_temporal_split.ipynb        # chronological split + explicit leakage verification
│   └── 04_baseline_model.ipynb        # Random Forest baseline training + full evaluation
├── models/day1/              # trained model + metadata (created by train_baseline.py)
├── results/day1/             # evaluation metrics (created by train_baseline.py)
├── tests/                    # unit tests using tiny synthetic data only
└── logs/                     # run logs from the scripts above
```

### `src/` vs `scripts/` vs `notebooks/`

* **`src/`** contains the reusable implementation code (dataset discovery,
  chunked loading, validation, cleaning, feature engineering, temporal
  splitting). This is the only place the actual logic lives — everything
  else calls into it rather than duplicating it.
* **`scripts/`** provides command-line execution of that logic end to
  end (`inspect_dataset.py`, `prepare_data.py`, `train_baseline.py`),
  meant to be run non-interactively, e.g. in a terminal or CI.
* **`notebooks/`** provides research/academic demonstrations of the same
  pipeline — exploration, visualization, and step-by-step walkthroughs
  for a reader or reviewer. Notebooks call functions from `src/`; they do
  not re-implement any pipeline logic themselves.

## Notebooks

The `notebooks/` directory mirrors the Day 1 pipeline stage by stage:

| Notebook | Demonstrates |
|---|---|
| `01_dataset_inspection.ipynb` | file discovery, schema peek, columns/dtypes, label distribution, missing/infinite values, timestamp discovery, basic plots |
| `02_data_preprocessing.ipynb` | Parquet-preferred/CSV-fallback loading, schema/NaN-Inf validation, leakage-prone feature identification, feature selection, label normalization |
| `03_temporal_split.ipynb` | timestamp distribution, chronological ordering, the chronological train/test split (**not** a random split), explicit temporal-leakage verification, and a visualization of the split |
| `04_baseline_model.ipynb` | loading the prepared split, training the Day 1 Random Forest baseline, confusion matrix, Precision/Recall/F1, ROC-AUC/PR-AUC, FPR/FNR, feature importance, and saving the model |

**Status:** the notebooks contain properly written cells calling into
`src/`, but have **not** been executed against the real dataset — no
outputs are stored. Run them yourself, in order, after placing the real
CIC-IDS2017 files under `data/raw/CIC-IDS2017/` and (for
`04_baseline_model.ipynb`) after running `scripts/prepare_data.py`.

```bash
pip install -r requirements.txt   # includes jupyter/notebook + matplotlib
jupyter notebook notebooks/
```

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
