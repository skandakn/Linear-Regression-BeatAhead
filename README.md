# BeatAhead: Linear Regression & Regularized Logistic Baseline for Intraoperative Myocardial Ischemia Early Warning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository implements the **Linear Modeling and Regularized Logistic Regression Baseline** for **BeatAhead**—an intraoperative early-warning system designed to predict impending sustained myocardial ischemia 5 to 15 minutes before manifest clinical onset.

---

## 📋 Clinical & Modeling Overview

Intraoperative myocardial ischemia often develops silently under general anesthesia. Early hemodynamic and electrophysiological cues—such as subtle ST-segment deviations, heart rate variability (HRV) dampening, and pulse arrival time (PAT) shifts—precede sustained ischemic events.

* **Clinical Goal**: Detect impending sustained ischemia ($\ge 10\,\text{s}$ ST depression $\le -0.10\,\text{mV}$ or elevation $\ge +0.10\,\text{mV}$) within a 5–15 minute prediction horizon.
* **Challenge**: Severe clinical class imbalance ($\approx 230:1$ negative-to-positive ratio in surgical monitoring).
* **Linear Modeling Role**: Serves as the transparent, clinically interpretable baseline benchmark against complex non-linear ensembles (Random Forest, XGBoost).

---

## 🛡️ Zero-Leakage Subject Partitioning

To avoid patient-level data leakage, partitioning is performed strictly at the **subject (caseid) level**:

* **Development Set (80 cases)**: 40 high-risk surgical cases (Cohort A) + 40 representative surgical cases (Cohort B). Used exclusively for median imputation, feature scaling, 5-Fold `StratifiedGroupKFold` cross-validation, and decision threshold calibration.
* **Locked Test Set (20 cases)**: 10 Cohort A + 10 Cohort B cases completely withheld during development and evaluation until final evaluation.

---

## 📊 Feature Matrix A (26 Multimodal Features)

| Physiological Modality | Extracted Features |
| :--- | :--- |
| **ECG / HRV** | `ecg_hr_mean`, `ecg_hr_std`, `ecg_rr_sdnn`, `ecg_rr_rmssd`, `ecg_pnn50`, `ecg_r_amp_mv`, `ecg_qrs_width_ms`, `ecg_sqi` |
| **Vascular / PAT** | `pat_median_ms`, `pat_iqr_ms`, `pat_valid_fraction`, `pat_valid` |
| **PPG Morphology** | `ppg_pulse_amp`, `ppg_perfusion_index`, `ppg_crest_time_ms`, `ppg_sqi` |
| **Oximetry (SpO2)** | `spo2_mean`, `spo2_min`, `spo2_std`, `spo2_desat_count` |
| **Pre-Event ST Trends** | `st_obs_mean`, `st_obs_median`, `st_obs_min`, `st_obs_std`, `st_delta_baseline`, `st_slope_mm_min` |

---

## 📈 Performance Summary

### Cross-Validation (80 Development Cases, 5-Fold StratifiedGroupKFold)
* **Out-of-Fold PR-AUC**: `0.1213` (vs $\approx 0.004$ random baseline prevalence)
* **Out-of-Fold ROC-AUC**: `0.9007`
* **Calibrated Decision Threshold**: `0.7821` (maximizes out-of-fold F1-score)
* **Out-of-Fold F1-Score**: `0.2431`

### Locked Out-of-Sample Test Evaluation (20 Cases)
* **Test Windows Evaluated**: 2,770 windows (11 positive event windows)
* **True Negatives**: 2,737
* **False Positives**: 22
* **False Negatives**: 11
* **True Positives**: 0 (highlights the inherent limitation of purely linear decision boundaries under extreme non-linear physiology, motivating non-linear ensembles like Random Forest and XGBoost)

---

## 🗂️ Repository Structure

```
Linear-Regression-BeatAhead/
├── .gitignore
├── README.md
├── requirements.txt
├── train_linear_regression.py     # Main end-to-end training and evaluation script
├── predict.py                     # Standalone CLI and programmatic inference tool
├── data/
│   ├── vitaldb_100_windows.csv.gz # Filtered sliding window dataset
│   ├── cohort_a_final_verified.csv
│   └── cohort_b_final_verified.csv
├── src/
│   ├── __init__.py
│   ├── dataset.py                 # Ingestion & strict subject-level partitioning
│   ├── feature_pipeline.py        # Median imputation & standardization
│   ├── models.py                  # Linear & regularized logistic classifiers
│   ├── metrics.py                 # PR-AUC, ROC-AUC, Brier score, threshold optimization
│   └── plotting.py                # Curves, confusion matrix & coefficient charts
├── tests/
│   └── test_linear_regression.py  # Unit test suite verifying zero leakage & reproducibility
├── artifacts/
│   ├── linear_regression_model.joblib
│   ├── feature_schema.json
│   └── model_metadata.json
└── outputs/
    ├── evaluation_metrics.json
    ├── test_predictions.csv
    ├── coefficients.csv
    ├── pr_roc_curves.png
    ├── confusion_matrix.png
    └── coefficients.png
```

---

## 🚀 Quickstart

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run Training & Evaluation Pipeline
```bash
python train_linear_regression.py
```

### 3. Run Inference on a Sample Window
```bash
python predict.py --test-sample
```

### 4. Run Unit Tests
```bash
python tests/test_linear_regression.py
```

---

## ⚖️ Clinical Interpretability: Top Linear Coefficients

Standardized logistic coefficients reveal directional physiological drivers:
* **Positive Risk Drivers**: Elevated `st_slope_mm_min`, increased `ecg_hr_mean`, and wider `ecg_qrs_width_ms`.
* **Protective / Baseline Stability**: Higher `spo2_mean`, stable `pat_median_ms`, and elevated `ppg_perfusion_index`.
