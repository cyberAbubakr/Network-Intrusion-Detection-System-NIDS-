
```Cross-Temporal Hybrid Network Intrusion Detection System

A research-oriented Network Intrusion Detection System (NIDS) that evaluates whether machine-learning detectors trained on earlier network traffic remain reliable across time, unseen attack classes, distribution shift, and a different dataset.

The project combines a supervised Random Forest, an unsupervised Isolation Forest, and a weighted hybrid detector. It is designed for CPU-only systems with limited memory and uses leakage-aware temporal evaluation rather than a random train/test split.

Project Status

Stage| Work| Status
Day 1| CIC-IDS2017 preprocessing, feature selection, temporal split, Random Forest baseline| Complete
Day 2| Validation, threshold selection, Isolation Forest, hybrid fusion, alerts| Complete
Day 3| Simulated zero-day evaluation using unseen attack classes| Complete
Day 4| Distribution-shift and unseen-attack analysis| Complete
Day 5| Cross-dataset validation on CSE-CIC-IDS2018| Complete
Day 6| SHAP model explainability| Planned
Day 7| LLM/RAG-assisted alert explanation and reporting| Planned

Current state: the detection, validation, zero-day, distribution-shift, and cross-dataset stages are implemented. SHAP explainability and LLM/RAG reporting are the remaining stages.

Research Objective

Many intrusion-detection experiments use random splitting, allowing similar traffic from the same collection period to appear in both training and testing. This can produce optimistic results.

This project instead asks:

«Can a detector trained on earlier traffic continue to identify attacks in later traffic, unseen attack classes, and an independently collected dataset without retraining or tuning on test data?»

Implemented Pipeline

Day 1 — Temporal Random Forest Baseline

- Discovers CIC-IDS2017 CSV or Parquet files.
- Normalizes labels, validates schemas, and handles invalid numeric values.
- Removes label-like, duplicate, and near-constant features.
- Produces 58 numeric model features.
- Uses Monday–Thursday traffic for training and Friday traffic for temporal testing when row-level timestamps are unavailable.
- Verifies that capture days do not leak between training and testing.
- Trains a lightweight Random Forest and freezes its selected threshold.

Latest recorded frozen Friday Random Forest results:

Metric| Value
Precision| 0.8235
Recall| 0.8819
F1-score| 0.8517
ROC-AUC| 0.9695
PR-AUC| 0.9101
False-positive rate| 0.0645
False-negative rate| 0.1181

Day 2 — Anomaly Detection and Hybrid Fusion

- Creates a validation partition without tuning on the final Friday test set.
- Trains an Isolation Forest and normalizes its anomaly scores.
- Selects thresholds using validation data only.
- Combines Random Forest and anomaly scores using weighted fusion.
- Freezes thresholds before final temporal evaluation.
- Produces comparison tables, per-class metrics, and sample alerts.

Component| Frozen configuration
Random Forest threshold| 0.01
Isolation Forest threshold| 0.15
Hybrid threshold| 0.50
Random Forest weight| 0.70
Anomaly weight| 0.30

Day 3 — Simulated Zero-Day Detection

The supervised model is trained while deliberately excluding:

- Bot
- DDoS
- PortScan

These excluded classes are treated as unseen attacks during evaluation. The experiment compares the Random Forest, Isolation Forest, and hybrid detector on attack types unavailable during supervised training.

The results preserve an important negative finding: the hybrid model does not automatically outperform both component models on every unseen class.

Day 4 — Distribution-Shift Analysis

- Compares feature distributions between Monday–Thursday training traffic and Friday traffic.
- Measures changes in Random Forest confidence.
- Relates important feature drift to temporal performance degradation.
- Runs distribution-shift tests across the selected features.
- Separately analyzes known and unseen attack behaviour.
- Preserves negative findings instead of presenting the hybrid detector as universally superior.

Important drifting features include:

- Packet Length Max
- Bwd Packet Length Std
- Packet Length Variance
- Packet Length Std
- Packet Length Mean
- Bwd Packet Length Mean
- Bwd Packet Length Max
- Flow Packets/s

Day 5 — Cross-Dataset Validation

The frozen CIC-IDS2017 models and thresholds are evaluated on CSE-CIC-IDS2018 without retraining or tuning on external data.

The cross-dataset pipeline:

- Maps CICFlowMeter feature-name variants.
- Coerces mapped values to numeric.
- Replaces positive and negative infinity with missing values.
- Imputes missing values using CIC-IDS2017 training medians only.
- Maps external labels to a consistent binary target.
- Records unmatched features.
- Verifies that external data is not used for training or threshold selection.
- Compares Random Forest, Isolation Forest, and hybrid generalization.

Experimental Safeguards

- No random train/test leakage: evaluation is separated by capture day when timestamps are unavailable.
- No threshold tuning on the final test set: thresholds are selected on validation data and frozen.
- No external-data retraining: CSE-CIC-IDS2018 is used only for evaluation.
- Training-only imputation: external missing values use CIC-IDS2017 training medians.
- No invented timestamps: flow timing statistics are never treated as capture timestamps.
- Negative results are retained: the hybrid model is not claimed to be universally better.
- Resource-aware processing: CSV input is processed in chunks and Parquet is preferred.

Repository Structure

.
├── data/
├── models/
├── results/
├── notebooks/
│   ├── 01_dataset_inspection.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_temporal_split.ipynb
│   ├── 04_baseline_model.ipynb
│   ├── 06_day2_validation_anomaly_hybrid.ipynb
│   ├── 07_day3_zero_day_detection.ipynb
│   ├── 08_day4_explainability_unseen_attack_analysis.ipynb
│   └── 09_day5_cross_dataset_validation.ipynb
├── scripts/
│   ├── inspect_dataset.py
│   ├── prepare_data.py
│   ├── train_baseline.py
│   ├── run_day2.py
│   ├── run_day3.py
│   ├── run_day4.py
│   └── run_day5.py
├── src/
│   ├── data/
│   ├── day2/
│   ├── day3/
│   ├── day4/
│   └── day5/
├── tests/
├── requirements.txt
└── README.md

Large datasets, generated models, and most experiment outputs are intentionally excluded from Git.

Datasets

CIC-IDS2017

CIC-IDS2017 is used for:

- Data preprocessing
- Model training
- Validation
- Temporal Friday testing
- Simulated zero-day evaluation
- Distribution-shift analysis

Place the original files under:

data/raw/CIC-IDS2017/

Keep their original filenames because capture-day detection uses file identity when a reliable row-level timestamp is unavailable.

CSE-CIC-IDS2018

CSE-CIC-IDS2018 is used only for cross-dataset evaluation. It must never be included in training or threshold selection.

Installation

Python 3.12 is recommended.

Linux/macOS

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

Windows PowerShell

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

Running the Project

Run the following commands from the repository root:

# Inspect CIC-IDS2017 files
python scripts/inspect_dataset.py --raw-dir data/raw/CIC-IDS2017

# Prepare the leakage-aware Day 1 dataset
python scripts/prepare_data.py --raw-dir data/raw/CIC-IDS2017 --chunk-size 250000

# Train and evaluate the Random Forest baseline
python scripts/train_baseline.py

# Run validation, Isolation Forest, hybrid fusion, and frozen evaluation
python scripts/run_day2.py

# Run simulated zero-day evaluation
python scripts/run_day3.py

# Run distribution-shift and unseen-attack analysis
python scripts/run_day4.py

# Run cross-dataset validation
python scripts/run_day5.py

The notebooks demonstrate the same research stages while calling reusable functions from "src/".

Tests

Tests use small synthetic inputs and do not require the complete datasets.

pytest -q

The test suite covers:

- Data preprocessing
- Temporal splitting
- Leakage verification
- Threshold selection
- Anomaly scoring
- Hybrid fusion
- Zero-day evaluation logic
- Distribution-shift analysis
- Feature and label mapping
- Infinity and missing-value handling
- Cross-dataset leakage safeguards

Key Finding

A hybrid detector is not automatically more robust simply because it combines supervised and unsupervised models.

Its performance depends on:

- Score calibration
- Fusion weights
- Decision thresholds
- Attack type
- Temporal distribution shift
- Cross-dataset differences

Weaker hybrid results are therefore treated as valid research findings rather than hidden.

Remaining Work

SHAP Explainability

The next stage will explain Random Forest predictions at global and individual-alert levels, including the features that raise or lower predicted attack risk.

LLM/RAG Reporting

The final planned stage will convert structured model outputs and SHAP evidence into readable analyst-facing alert explanations.

The language model will explain existing evidence; it will not replace the detector or invent unsupported causes.

Author

Syed Muhammad Abubakr
BS Computer Science
GitHub: "cyberAbubakr" (https://github.com/cyberAbubakr)

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
