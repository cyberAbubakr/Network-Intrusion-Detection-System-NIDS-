
# Cross-Temporal Hybrid Network Intrusion Detection System

A research-focused Network Intrusion Detection System that evaluates whether machine-learning models trained on earlier network traffic remain reliable against later traffic, unseen attack classes, distribution shift, and an external dataset.

The system combines:

- A supervised **Random Forest classifier**
- An unsupervised **Isolation Forest anomaly detector**
- A weighted **hybrid detection model**

The pipeline is designed for CPU-only systems with limited memory and uses temporal evaluation instead of a random train/test split.

## Project Status

| Stage | Description | Status |
|---|---|---|
| Day 1 | Data preprocessing, temporal splitting, and Random Forest baseline | Complete |
| Day 2 | Validation, Isolation Forest, hybrid fusion, and threshold freezing | Complete |
| Day 3 | Simulated zero-day and unseen-attack evaluation | Complete |
| Day 4 | Distribution-shift and performance-degradation analysis | Complete |
| Day 5 | Cross-dataset validation using CSE-CIC-IDS2018 | Complete |
| Day 6 | SHAP model explainability | Remaining |
| Day 7 | LLM/RAG-assisted alert explanations and reporting | Remaining |

> **Current state:** The complete detection and evaluation pipeline is implemented through cross-dataset validation. SHAP explainability and LLM/RAG reporting are the only major stages remaining.

## Research Objective

Many intrusion-detection experiments use random train/test splitting. This may place highly similar traffic from the same collection period in both sets, producing overly optimistic results.

This project instead investigates:

> Can an intrusion-detection model trained on earlier network traffic identify attacks in later traffic, unseen attack classes, and an independently collected dataset without retraining or tuning on the test data?

## System Architecture

The system contains three detection approaches:

### Random Forest

The Random Forest is the supervised component. It learns the difference between benign and malicious traffic using labelled CIC-IDS2017 training data.

### Isolation Forest

The Isolation Forest is the unsupervised component. It identifies unusual traffic patterns without depending directly on attack labels.

### Hybrid Detector

The hybrid detector combines the Random Forest probability and normalized Isolation Forest anomaly score:

```text
Hybrid Score = 0.70 × Random Forest Score
             + 0.30 × Isolation Forest Score
```

The hybrid model is evaluated honestly and is not assumed to outperform both individual models automatically.

## Completed Experimental Stages

### Day 1 — Data Pipeline and Temporal Baseline

The first stage implements:

- Dataset discovery
- CSV and Parquet loading
- Schema validation
- Label normalization
- NaN and infinity handling
- Leakage-prone feature removal
- Duplicate-feature removal
- Near-constant feature removal
- Temporal train/test splitting
- Leakage verification
- Random Forest training and evaluation

The final model uses **58 numeric network-flow features**.

Because the available CIC-IDS2017 files do not contain a reliable row-level timestamp, the project uses their original capture-day identities:

- **Training:** Monday to Thursday
- **Final temporal test:** Friday

No capture day is allowed to appear in both training and testing.

### Latest Random Forest Results

The latest recorded results on the frozen Friday temporal test set are:

| Metric | Value |
|---|---:|
| Precision | 0.8235 |
| Recall | 0.8819 |
| F1-score | 0.8517 |
| ROC-AUC | 0.9695 |
| PR-AUC | 0.9101 |
| False-positive rate | 0.0645 |
| False-negative rate | 0.1181 |
| True negatives | 360,575 |
| False positives | 24,842 |
| False negatives | 15,525 |
| True positives | 115,882 |

### Day 2 — Validation, Anomaly Detection, and Hybrid Fusion

The second stage adds:

- A validation partition
- Isolation Forest training
- Anomaly-score normalization
- Validation-only threshold selection
- Random Forest and Isolation Forest score fusion
- Frozen final thresholds
- Per-class evaluation
- Model-comparison tables
- Sample alert generation

The final test set is not used to select thresholds or model weights.

### Frozen Model Configuration

| Component | Frozen value |
|---|---:|
| Random Forest threshold | 0.01 |
| Isolation Forest threshold | 0.15 |
| Hybrid threshold | 0.50 |
| Random Forest weight | 0.70 |
| Isolation Forest weight | 0.30 |

After selection, these values remain fixed during later experiments.

### Day 3 — Simulated Zero-Day Detection

The zero-day experiment removes selected attack classes from supervised training:

- Bot
- DDoS
- PortScan

These excluded classes are then treated as unseen attacks during evaluation.

This stage compares how the following models respond to attacks they did not observe during supervised training:

- Random Forest
- Isolation Forest
- Hybrid detector

The experiment produces detection rates for each unseen attack class.

An important negative result is preserved: the hybrid model does **not** automatically outperform both component models on every unseen attack.

### Day 4 — Distribution-Shift Analysis

This stage investigates why performance changes between earlier training traffic and later Friday traffic.

It includes:

- Feature-distribution comparisons
- Random Forest feature-importance analysis
- Model-confidence analysis
- Statistical distribution-shift testing
- Known-versus-unseen attack comparison
- Association between feature drift and confidence degradation
- Analysis of hybrid-model weaknesses

Important drifting features include:

- Packet Length Max
- Bwd Packet Length Std
- Packet Length Variance
- Packet Length Std
- Packet Length Mean
- Bwd Packet Length Mean
- Bwd Packet Length Max
- Flow Packets/s

The analysis shows that changes in important network-flow features can reduce model confidence and affect generalization.

### Day 5 — Cross-Dataset Validation

The fifth stage evaluates the frozen CIC-IDS2017 models using **CSE-CIC-IDS2018**.

The external dataset is used only for evaluation.

The following are not allowed during this stage:

- Model retraining
- Threshold reselection
- Hybrid-weight modification
- Test-data-based feature selection
- External-data-based imputation statistics

The cross-dataset pipeline:

- Maps CICFlowMeter feature-name variations
- Converts mapped values to numeric
- Replaces positive and negative infinity with missing values
- Imputes missing values using CIC-IDS2017 training medians
- Maps external labels into a consistent binary target
- Records unmatched features
- Applies the frozen models and thresholds
- Compares Random Forest, Isolation Forest, and hybrid performance
- Verifies that external data never enters training

This stage measures whether the system generalizes beyond the dataset on which it was developed.

## Experimental Safeguards

The project enforces the following research safeguards:

- **Temporal separation:** earlier capture days are used for training and Friday traffic is used for final testing.
- **Leakage verification:** training and testing capture days cannot overlap.
- **Validation-only tuning:** thresholds are selected using validation data.
- **Frozen evaluation:** final thresholds and hybrid weights remain unchanged during later experiments.
- **No external retraining:** CSE-CIC-IDS2018 is used only for evaluation.
- **Training-only imputation:** external missing values use CIC-IDS2017 training medians.
- **No invented timestamps:** flow timing features are never treated as capture timestamps.
- **Transparent preprocessing:** removed and unmatched features are logged.
- **Negative results retained:** weaker hybrid performance is reported rather than hidden.
- **Resource-aware processing:** chunked CSV loading and Parquet support reduce memory usage.

## Repository Structure

```text
Network-Intrusion-Detection-System-NIDS-/
├── data/
│   ├── raw/
│   │   └── CIC-IDS2017/
│   └── processed/
│       └── day1/
├── models/
│   └── day1/
├── results/
│   └── day1/
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
│   │   ├── loader.py
│   │   ├── validator.py
│   │   ├── cleaner.py
│   │   ├── feature_engineering.py
│   │   └── temporal_split.py
│   ├── day2/
│   │   ├── alerts.py
│   │   ├── anomaly.py
│   │   ├── hybrid.py
│   │   ├── thresholding.py
│   │   └── validation.py
│   ├── day3/
│   │   └── zero_day.py
│   ├── day4/
│   │   └── analysis.py
│   └── day5/
│       ├── feature_mapping.py
│       └── label_mapping.py
├── tests/
├── requirements.txt
├── pytest.ini
└── README.md
```

Large datasets, trained models, and generated result files are intentionally excluded from GitHub and must be generated locally.

## Datasets

### CIC-IDS2017

CIC-IDS2017 is used for:

- Data preprocessing
- Supervised training
- Validation
- Temporal Friday evaluation
- Simulated zero-day evaluation
- Distribution-shift analysis

Place the files under:

```text
data/raw/CIC-IDS2017/
```

The original filenames should be preserved because the pipeline uses them to identify capture days when row-level timestamps are unavailable.

### CSE-CIC-IDS2018

CSE-CIC-IDS2018 is used only for cross-dataset evaluation.

It must not be included in:

- Model training
- Threshold selection
- Hybrid-weight selection
- Feature selection
- Training-median calculation

## Installation

Python 3.12 is recommended.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Linux or macOS

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the Project

Run all commands from the repository root.

### 1. Inspect the Dataset

```bash
python scripts/inspect_dataset.py --raw-dir data/raw/CIC-IDS2017
```

### 2. Prepare the Day 1 Data

```bash
python scripts/prepare_data.py --raw-dir data/raw/CIC-IDS2017 --chunk-size 250000
```

### 3. Train the Random Forest Baseline

```bash
python scripts/train_baseline.py
```

### 4. Run Validation and Hybrid Evaluation

```bash
python scripts/run_day2.py
```

### 5. Run the Simulated Zero-Day Experiment

```bash
python scripts/run_day3.py
```

### 6. Run Distribution-Shift Analysis

```bash
python scripts/run_day4.py
```

### 7. Run Cross-Dataset Validation

```bash
python scripts/run_day5.py
```

The notebooks provide step-by-step research demonstrations of the same stages and call reusable functions from `src/`.

## Testing

The automated tests use small synthetic inputs and do not require the complete datasets.

Run:

```bash
pytest -q
```

The test suite covers:

- Dataset loading
- Preprocessing
- Schema validation
- Temporal splitting
- Leakage detection
- Threshold selection
- Anomaly scoring
- Hybrid-score calculation
- Alert generation
- Zero-day evaluation
- Distribution-shift analysis
- Feature mapping
- Label mapping
- Infinity handling
- Missing-value imputation
- Cross-dataset leakage safeguards

## Key Research Finding

A hybrid model is not automatically more robust simply because it combines a supervised classifier and an unsupervised anomaly detector.

Its performance depends on:

- Score calibration
- Fusion weights
- Decision thresholds
- Attack class
- Temporal distribution shift
- Cross-dataset feature differences

In some experiments, the Random Forest or Isolation Forest performs better than the hybrid model on unseen attacks.

This is retained as a valid negative research result rather than being hidden or modified.

## Remaining Work

### SHAP Explainability

The next stage will use SHAP to explain Random Forest predictions.

The planned outputs include:

- Global feature importance
- SHAP summary plots
- Feature-contribution analysis
- Individual alert explanations
- Comparison of explanations across known and unseen attacks
- Explanation-stability analysis under distribution shift

### LLM/RAG-Assisted Reporting

The final stage will convert structured detection outputs and SHAP evidence into readable security-analyst reports.

The language model will:

- Summarize detected activity
- Explain the strongest supporting features
- Describe model confidence
- Identify whether the traffic resembles known or unseen behaviour
- Retrieve supporting cybersecurity knowledge
- Produce analyst-friendly alert narratives

The language model will explain evidence produced by the detection pipeline. It will not replace the models or invent unsupported attack causes.

## Limitations

- CIC-IDS2017 attacks are tied to specific capture days.
- Day-based splitting creates a harder and less balanced evaluation scenario.
- Dataset-specific patterns may not transfer reliably to another dataset.
- The Isolation Forest may produce a high false-positive rate.
- Weighted score fusion does not guarantee better unseen-attack detection.
- Cross-dataset feature names and distributions require careful mapping.
- SHAP and LLM/RAG explanation stages have not yet been implemented.
- The system is an offline research prototype, not a live production IDS.

## Technologies

- Python
- pandas
- NumPy
- scikit-learn
- PyArrow
- Jupyter Notebook
- pytest
- joblib
- Matplotlib
- Seaborn

## Author

**Syed Muhammad Abubakr**  
BS Computer Science  
GitHub: [cyberAbubakr](https://github.com/cyberAbubakr)
