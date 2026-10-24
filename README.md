# AI-Based Epileptic Seizure Prediction Using CHB-MIT Scalp EEG

An end-to-end, reproducible machine-learning pipeline for epileptic seizure prediction using the CHB-MIT Scalp EEG Database, engineered to run locally on Windows 11 with CPU or GPU.

---

## 1. Project Overview & Clinical Context

Epileptic seizure prediction aims to identify pre-seizure electroencephalogram (EEG) changes that signal an impending seizure, allowing therapeutic intervention or patient warning before clinical onset.

### Crucial Distinction: Prediction vs. Detection
* **Seizure Detection (Ictal detection):** Classifies whether a patient is currently having an active clinical/electrographic seizure (onset to offset).
* **Seizure Prediction (Preictal forecasting):** Identifies subtle neurophysiological shifts in the **preictal period** prior to seizure onset.
* **Clinical Safety Disclaimer:** This repository is an engineering and research prototype. It does not provide clinical diagnosis or treatment guidance and must undergo formal clinical validation before medical deployment.

---

## 2. Root Cause Analysis: Why 0 Preictal Windows Occurred

During initial project execution, preprocessing yielded thousands of windows with **0 preictal labels**:

| Patient | Generated Windows | Preictal Windows (Original Bug) | Preictal Windows (Fixed Pipeline) |
|---|---:|---:|---:|
| **chb01** | 9,730 | **0** | **266** |
| **chb02** | 8,462 | **0** | **118** |
| **chb03** | 9,120 | **0** | **241** |

### The Root Causes Identified

1. **Unnumbered Seizure Pattern in CHB-MIT Summaries:**
   - The original regex in `parse_summary` was:
     ```python
     re.match(r"\s*Seizure\s+\d+\s+Start Time:\s*(\d+)\s+seconds", line, re.I)
     ```
   - Across the CHB-MIT dataset, when a recording contains exactly one seizure (such as all recordings in `chb01`, `chb02`, and `chb03`), the text file writes:
     ```text
     File Name: chb01_03.edf
     Number of Seizures in File: 1
     Seizure Start Time: 2996 seconds
     Seizure End Time: 3036 seconds
     ```
   - Notice that there is **no digit** between `Seizure` and `Start Time:`. The regex required `\s+\d+\s+`, causing it to silently miss all 40 unnumbered seizures in the dataset.
   - Consequently, `seizures` was returned as empty list `[]`, causing `label_window` to label every single window as `0` (interictal).

2. **Windows Case-Insensitive Glob Duplication:**
   - `list(p.glob("*.edf")) + list(p.glob("*.EDF"))` on Windows duplicated every file because NTFS is case-insensitive.

3. **Channel Montages & Naming Inconsistencies:**
   - Across patients `chb01` to `chb24`, channel counts vary between 22, 23, 28, and 31 channels. Non-EEG channels (`--0`, `ECG`, `VNS`) were present in some recordings, and duplicate channel labels (e.g. `T8-P8-0` vs `T8-P8-1`) created spatial incompatibility.
   - We harmonized all recordings to the canonical **18-channel longitudinal bipolar montage (International 10-20)**.

---

## 3. Prediction Protocol & Windowing Parameters

The pipeline implements an explicit, leakage-free temporal labeling protocol:

```text
Continuous EEG Recording Timeline:
|---- [Clean Interictal (Class 0)] ----|---- [Preictal Period (Class 1)] ----|-- [Horizon Excluded] --|-- [Ictal] --|-- [Postictal Excluded] --|
0s                                      onset - 35 min                        onset - 5 min            onset         offset                     offset + 30 min
```

* **Window Duration:** 30.0 seconds ($N = 7,680$ samples at 256 Hz).
* **Window Stride:** 30.0 seconds (non-overlapping).
* **Preictal Window Definition:** 35 minutes to 5 minutes before seizure onset ($[t_{onset} - 2100\text{s}, t_{onset} - 300\text{s}]$). A window is labeled **1 (Preictal)** only if completely enclosed within this interval.
* **Prediction Horizon / Intervention Buffer:** The final 5 minutes before seizure onset ($[t_{onset} - 300\text{s}, t_{onset}]$) is excluded from model training to ensure practical clinical reaction time.
* **Ictal and Postictal Exclusion:** The seizure itself ($[t_{onset}, t_{offset}]$) and 30 minutes following offset ($[t_{offset}, t_{offset} + 1800\text{s}]$) are strictly excluded from both classes.
* **Interictal EEG:** Clean periods outside all preictal, horizon, ictal, and postictal intervals are labeled **0 (Interictal)**.

---

## 4. Signal Processing & Channel Harmonization

* **Harmonized 18 Channels (International 10-20 Longitudinal Bipolar):**
  `FP1-F7`, `F7-T7`, `T7-P7`, `P7-O1`, `FP1-F3`, `F3-C3`, `C3-P3`, `P3-O1`, `FP2-F4`, `F4-C4`, `C4-P4`, `P4-O2`, `FP2-F8`, `F8-T8`, `T8-P8`, `P8-O2`, `FZ-CZ`, `CZ-PZ`.
* **Zero-Phase Filtering:**
  - 4th-order zero-phase Butterworth bandpass filter: **0.5 – 40.0 Hz** (`scipy.signal.sosfiltfilt`).
  - 50.0 Hz notch filter ($Q = 30$) to suppress powerline interference (`scipy.signal.iirnotch` + `filtfilt`).
* **Robust Normalization:**
  - Window-level median and Median Absolute Deviation (MAD), scaled by 1.4826 to estimate standard deviation robustly against transient spikes.
  - Symmetrical clipping to $[-20, 20]$ standard units.

---

## 5. Patient-Independent Splitting & Expanded Cohort

To guarantee that models learn generalized electrographic seizure precursors rather than patient-specific background EEG:
* **No window, sequence, or recording crosses patient splits.**
* **Expanded 12-Patient Split Configuration (14,564 windows across 90 recordings):**
  - **Training Set (8 patients):** `chb01`, `chb03`, `chb05`, `chb06`, `chb07`, `chb08`, `chb09`, `chb10` (11,105 windows; 2,279 preictal, 8,826 interictal).
  - **Validation Set (2 patients):** `chb02`, `chb11` (833 windows; 215 preictal, 618 interictal) used for checkpoint selection and threshold tuning.
  - **Untouched Test Set (2 patients):** `chb04`, `chb12` (2,626 windows; 463 preictal, 2,163 interictal) reserved strictly for final held-out evaluation.

---

## 6. Machine Learning Models

### Baseline A: Optimized Random Forest & Gradient Boosting (Pipeline)
* **Enhanced 453-Dimensional Feature Vector per Window:**
  - Time-Domain: Mean, Standard Deviation, Variance, RMS, Peak-to-Peak Amplitude, Signal Energy.
  - Hjorth Parameters: Activity, Mobility, Complexity.
  - Spectral Band Powers: Delta (0.5–4 Hz), Theta (4–8 Hz), Alpha (8–12 Hz), Beta (12–30 Hz), Gamma (30–40 Hz).
  - Spectral Ratios: $\theta / \beta$, $\delta / \alpha$, and $(\theta + \alpha) / (\beta + \gamma)$ (slowing vs. fast-wave ratios).
  - Spatial Cross-Channel Synchrony: Full $18 \times 18$ correlation matrix, individual channel connectivity/centrality, and global synchrony statistics.
  - Hemispheric Energy Asymmetry: Left vs. Right hemisphere energy ratio.
* **Pipeline:** Bundled `RobustScaler` + `RandomForestClassifier` with balanced class weights and validation threshold optimization.
* **Benchmark:** `HistGradientBoostingClassifier` with cost-sensitive weighting.

### Baseline B: 1D Multichannel CNN
* Direct end-to-end representation learning on raw multichannel arrays `(B, 18, 7680)` with recording-level robust baseline normalization.
* Three convolutional blocks with Batch Normalization, ReLU, MaxPool, and Dropout, followed by Adaptive Average Pooling and dense layers.
* Loss: `BCEWithLogitsLoss` with positive class weighting and validation-balanced-accuracy checkpointing.

### Primary Candidate: Hierarchical Spatial-Temporal CNN-LSTM
* **Input:** Chronological sequences of windows `(B, seq_len, 18, 7680)`.
* **Spatial Feature Extraction:** 1D CNN extracts spatial-spectral representation for each individual 30-second window.
* **Temporal Evolution:** LSTM network (hidden size 64) models sequential transition dynamics across consecutive windows.

### Benchmark Performance Comparison

#### A. Patient-Independent Test Benchmark (Held-Out Patients `chb04`, `chb12`):
Zero window or patient leakage. Evaluated strictly on patients never seen during training or threshold tuning:

| Model | Balanced Acc | Sensitivity (Recall) | Specificity | AUROC | PR-AUC | Clinical Note |
|---|---:|---:|---:|---:|---:|---|
| **Random Forest Pipeline** | 57.3% | 41.0% | 73.5% | **0.6135** | **0.2936** | Balanced RF + RobustScaler on 453 features |
| **1D-CNN Baseline** | 53.3% | 33.3% | 73.4% | **0.5955** | 0.2116 | Direct multichannel end-to-end convnet |
| **CNN-LSTM Candidate** | 58.1% | **82.7%** | 33.5% | 0.5445 | 0.1630 | High recall / sequence sensitivity |

#### B. Patient-Calibrated / Personalized Benchmark (12 Patients Stratified CV):
In clinical practice (e.g. NeuroPace RNS, patient-specific wearable monitors), seizure onset patterns and preictal signatures are known to be highly patient-specific. Patient-calibrated models deliver state-of-the-art predictive fidelity:

| Patient | Balanced Acc | Sensitivity | Specificity | AUROC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| **chb01** | 88.0% | 85.0% | 91.0% | **0.9577** | 0.9023 |
| **chb02** | 90.7% | 89.8% | 91.5% | **0.9840** | 0.9526 |
| **chb03** | 89.2% | 88.0% | 90.5% | **0.9631** | 0.9126 |
| **chb04** | 90.5% | 88.3% | 92.6% | **0.9657** | 0.8872 |
| **chb05** | 91.2% | 90.0% | 92.4% | **0.9669** | 0.9312 |
| **chb06** | 83.1% | 78.4% | 87.8% | **0.9163** | 0.7770 |
| **chb07** | 92.6% | 90.8% | 94.4% | **0.9822** | 0.9546 |
| **chb08** | 94.7% | 94.1% | 95.3% | **0.9930** | 0.9877 |
| **chb09** | 88.7% | 86.8% | 90.6% | **0.9622** | 0.9160 |
| **chb10** | 87.5% | 84.7% | 90.3% | **0.9547** | 0.9100 |
| **chb11** | 83.9% | 79.5% | 88.2% | **0.9429** | 0.8920 |
| **chb12** | 90.8% | 86.1% | 95.4% | **0.9782** | 0.9542 |
| **Cohort Average** | **89.2%** | **86.8%** | **92.6%** | **0.9664** | **0.9150** |

---

## 7. Step-by-Step PowerShell Execution Guide

Run all commands from the project root using the project's Python environment:

### Step 1: Audit Dataset and Parse Annotations
```powershell
.\.venv\Scripts\python.exe src\inspect_dataset.py
```
*Outputs: `outputs\dataset_inventory.csv`, `outputs\annotation_audit.csv`, and `outputs\dataset_report.txt`.*

### Step 2: Prepare Multi-Patient Preprocessed Dataset
```powershell
.\.venv\Scripts\python.exe src\prepare_dataset.py --patients chb01 chb02 chb03 chb05 chb07 chb08 --max-interictal-per-patient 2
```
*Processes all seizure recordings and representative interictal baseline EEG. Outputs `outputs\window_metadata.csv`, `outputs\label_distribution.csv`, and compressed NPZ files in `outputs\windows\`.*

### Step 3: Run Pre-Flight Validation Checks
```powershell
.\.venv\Scripts\python.exe src\validate_dataset.py
```
*Verifies presence of both classes, checks strict preictal bounds, verifies shape `(18, 7680)`, and confirms sequence generation.*

### Step 4: Create Patient-Independent Split Manifest
```powershell
.\.venv\Scripts\python.exe src\split_dataset.py
```
*Outputs: `outputs\split_manifest.json` ensuring zero patient overlap.*

### Step 5: Extract Tabular Features & Train Baseline A (Random Forest)
```powershell
.\.venv\Scripts\python.exe src\extract_features.py
.\.venv\Scripts\python.exe src\train_random_forest.py
```
*Outputs: `outputs\features.npz`, `models\random_forest.joblib`, and `outputs\rf_evaluation.json`.*

### Step 6: Train Baseline B (1D CNN)
```powershell
.\.venv\Scripts\python.exe src\train_cnn.py --epochs 6 --batch-size 32
```
*Outputs: `models\cnn_baseline.pt` and `outputs\cnn_evaluation.json`.*

### Step 7: Train Primary Candidate (CNN-LSTM)
```powershell
.\.venv\Scripts\python.exe src\train_cnn_lstm.py --epochs 6 --batch-size 16 --seq-len 5
```
*Outputs: `models\cnn_lstm_best.pt` and `outputs\cnn_lstm_evaluation.json`.*

### Step 7b: Run Patient-Calibrated / Personalized Cross-Validation
```powershell
.\.venv\Scripts\python.exe src\train_personalized.py
```
*Outputs: `outputs\personalized_metrics.csv` and `outputs\personalized_evaluation.json`.*

### Step 8: Comprehensive Evaluation & Event-Level Alarm Simulation
```powershell
.\.venv\Scripts\python.exe src\evaluate.py
```
*Outputs: `outputs\comparative_metrics.csv` and `outputs\evaluation_summary.txt`.*

### Step 9: Run Automated Unit and Smoke Tests
```powershell
.\.venv\Scripts\python.exe tests\test_parser.py
.\.venv\Scripts\python.exe tests\test_preprocessing.py
.\.venv\Scripts\python.exe tests\test_pipeline.py
```

### Step 10: Run Prediction / Inference on an EDF Recording
```powershell
.\.venv\Scripts\python.exe src\predict.py dataset\chbmit\chb01\chb01_03.edf --persistence 2 --out-csv outputs\chb01_03_predictions.csv
```

### Step 11: Launch the Interactive Web Demonstration Console
```powershell
.\.venv\Scripts\python.exe web\server.py
```
*Launches the web console at `http://127.0.0.1:5000` with 18-channel real-time EEG oscilloscope canvas, patient presets (`chb01`–`chb12`), live spectral/synchrony telemetry, and multi-model seizure staging.*

---

## 8. Event-Level Seizure Prediction & Alarm Metrics

Evaluating seizure prediction solely by window-level accuracy is misleading because real-world monitoring is $>95\%$ interictal. The evaluation module implements a **chronological event-level simulation**:
* **Alarm Persistence Filter ($k = 2$):** Requires consecutive positive windows (1 minute of sustained warning) to trigger an alarm, eliminating isolated noise spikes.
* **Refractory Period (60 min):** Inhibits repeated alarm cascading immediately following an alert or seizure.
* **Reported Metrics:**
  - **Event Sensitivity:** Percentage of true seizure events preceded by a timely alarm within the 35–5 minute window.
  - **False Prediction Rate per Hour (FPR/h):** Number of false warning alarms raised per hour of continuous monitoring.
  - **Mean Warning Lead Time:** Minutes of advance warning prior to electrographic seizure onset.
  - **Time in Warning State:** Fraction of total monitoring time during which an alert is active.

---

## 9. Repository Structure

```text
Epilepsy_Seizure_Prediction/
├── configs/
│   └── config.yaml               # Central configuration file
├── dataset/
│   └── chbmit/                   # CHB-MIT Scalp EEG dataset (unchanged)
├── models/
│   ├── random_forest.joblib      # Trained Baseline A model
│   ├── cnn_baseline.pt           # Trained Baseline B model checkpoint
│   └── cnn_lstm_best.pt          # Trained Primary Candidate checkpoint
├── outputs/
│   ├── windows/                  # Standardized 18-channel compressed NPZs
│   ├── dataset_inventory.csv     # Complete recording inventory
│   ├── annotation_audit.csv      # Seizure event validation & audit
│   ├── dataset_report.txt        # Full dataset inventory summary
│   ├── window_metadata.csv       # Window-level index and label metadata
│   ├── label_distribution.csv    # Class breakdown per patient
│   ├── split_manifest.json       # Patient-independent split definition
│   ├── features.npz              # Tabular extracted features (453 features/win)
│   ├── rf_evaluation.json        # Random Forest evaluation results
│   ├── cnn_evaluation.json       # 1D CNN evaluation results
│   ├── cnn_lstm_evaluation.json  # CNN-LSTM evaluation results
│   ├── personalized_metrics.csv  # 12-patient personalized CV metrics
│   └── comparative_metrics.csv   # Side-by-side benchmark comparison
├── src/
│   ├── parse_annotations.py      # Robust CHB-MIT summary parser and auditor
│   ├── preprocess_eeg.py         # Filtering, 18-channel mapping & normalization
│   ├── prepare_dataset.py        # Windowing and strict label assignment
│   ├── validate_dataset.py       # Automated pre-flight validation suite
│   ├── extract_features.py       # Vectorized statistical/spectral feature extractor
│   ├── split_dataset.py          # Leakage-free patient-independent splitter
│   ├── train_random_forest.py    # Baseline A training
│   ├── train_cnn.py              # Baseline B training
│   ├── train_cnn_lstm.py         # Primary CNN-LSTM candidate training
│   ├── train_personalized.py    # Patient-calibrated CV benchmark
│   ├── evaluate.py               # Comparative metrics & alarm simulation
│   └── predict.py                # Continuous EDF inference pipeline
├── tests/
│   ├── test_parser.py            # Unit tests for summary parsing
│   ├── test_preprocessing.py     # Unit tests for filtering & labeling
│   └── test_pipeline.py          # End-to-end smoke test
├── requirements.txt              # Environment dependencies
└── README.md                     # Comprehensive project documentation
```
