# Heart Disease Risk Prediction

An end-to-end, production-oriented machine learning pipeline that predicts the presence and severity of heart disease from clinical measurements, built on the UCI Heart Disease dataset.

> **Status:** Baseline pipeline (preprocessing → training → evaluation) complete and validated on a held-out test set. Model comparison and hyperparameter tuning are the next phase — see [Roadmap](#roadmap).

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Project Architecture](#project-architecture)
- [Key Design Decisions](#key-design-decisions)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Baseline Results](#baseline-results)
- [Roadmap](#roadmap)
- [Tech Stack](#tech-stack)
- [License](#license)

---

## Problem Statement

Cardiovascular disease is one of the leading causes of death worldwide, and early risk stratification from routine clinical measurements can support faster, better-informed decisions. This project builds a reproducible machine learning pipeline that classifies patients into one of five heart disease severity levels (0 = no disease, 1–4 = increasing severity) based on 15 clinical features such as age, resting blood pressure, cholesterol, and exercise-induced measurements.

The project is built with an emphasis on **reproducibility, data integrity, and evaluation rigor** — not just model accuracy — since this reflects how ML systems are actually built and reviewed in production settings.

## Dataset

- **Source:** UCI Machine Learning Repository — Heart Disease dataset (combined from four medical centers).
- **Size:** 920 patient records, 16 columns (15 features + target).
- **Target:** `num` — 5 classes, imbalanced (44.7% class 0 down to 3.0% class 4).
- **Key challenges identified during EDA:**
  - High missingness in `ca` (66.4%), `thal` (52.8%), and `slope` (33.6%) — plausibly non-random (MNAR), not just noise.
  - Clinically invalid values: zeros in `trestbps`/`chol`, negative values in `oldpeak`.
  - Class imbalance requiring stratified splitting and macro-averaged metrics.

Full exploratory analysis is available in [`notebooks/`](notebooks/).

*Note: raw data is not committed to this repository to keep it lightweight; it is available publicly from the UCI Machine Learning Repository.*

## Project Architecture

```
Raw CSV
   │
   ▼
load_data()  ──────────────────────────────────────────  src/preprocessing.py
   │
   ▼
split_features_target()   (drops id + target; single source of truth)
   │
   ▼
split_data()  ───────────────────────────────────────────  src/data_split.py
   │  (stratified train/test split)
   ▼
build_base_pipeline_steps()  ────────────────────────────  src/preprocessing.py
   │  ├─ domain_cleaning   (invalid zeros/negatives → NaN)
   │  └─ preprocessor      (impute + missing-flag + scale/encode)
   ▼
build_model_pipeline()  ─────────────────────────────────  src/modeling.py
   │  (shared steps + model)
   ▼
train_model() → evaluate_model() → save_model()  ────────  src/train.py (entry point)
```

The preprocessing steps are defined **once**, in `build_base_pipeline_steps()`, and reused by every pipeline in the project (training, evaluation, future inference) — no logic is duplicated across files.

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Macro F1 / precision / recall over accuracy | Target is heavily imbalanced (class 4 is 3% of the data); accuracy alone would hide poor performance on minority classes. |
| Stratified train/test split + stratified K-fold CV | Preserves class proportions in every split, avoiding evaluation on unrepresentative folds. |
| Domain-aware cleaning before imputation | Zero values in `chol`/`trestbps` and negative `oldpeak` are not statistically missing but are clinically impossible — treating them as ordinary numbers would silently corrupt the model. |
| Missing-value indicator flags | Missingness in `ca`/`thal` is unusually high and may be informative (e.g., certain source hospitals not running certain tests) rather than random. |
| Centralized configuration (`src/config.py`) | Paths, random seeds, and column names are defined once and imported everywhere, preventing silent inconsistencies between modules. |
| `Pipeline`-based preprocessing, fit only on training data | Prevents data leakage from test data into imputation statistics or scaling parameters. |
| `class_weight="balanced"` on the baseline model | Forces the model to pay attention to minority classes instead of defaulting to the majority class — see the trade-off this creates in [Baseline Results](#baseline-results). |

## Project Structure

```
heart-disease-ml/
├── data/
│   └── raw/                  # not committed — download from UCI (see Dataset section)
├── models/                   # trained pipeline artifacts (.joblib)
├── notebooks/                # EDA and exploratory training notebooks
├── reports/                  # generated evaluation artifacts (confusion matrix, etc.)
├── src/
│   ├── config.py             # centralized paths, constants, random seeds
│   ├── preprocessing.py      # data loading, domain cleaning, shared pipeline steps
│   ├── data_split.py         # stratified train/test splitting
│   ├── modeling.py           # model pipeline construction and training
│   ├── baseline.py           # stratified cross-validation baseline evaluation
│   ├── evaluation.py         # test-set metrics and confusion matrix plotting
│   └── train.py              # end-to-end entry point
├── tests/                    # unit tests (in progress)
├── .gitignore
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites
- Python 3.10+
- The Heart Disease dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease), saved locally as `data/raw/data.csv`

### Installation

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Usage

**Run the full training pipeline** (load → split → train → evaluate → save):

```bash
python -m src.train
```

This trains the model, prints test-set metrics, saves a confusion matrix to `reports/confusion_matrix.png`, and saves the trained pipeline to `models/baseline_model.joblib`.

**Run cross-validation only** (stability check across 5 folds, no held-out test set):

```bash
python -m src.baseline
```

**Run tests** *(test suite in progress — see Roadmap)*:

```bash
pytest tests/
```

## Baseline Results

These are the results of **Phase 1**: a single `LogisticRegression` model, deliberately evaluated first to establish an honest reference point before trying more complex models. The numbers below are modest by design — they exist to be improved upon, and the analysis of *why* they look this way is the actual deliverable of this phase.

Evaluated on a stratified held-out test set (184 samples, 20% of the data):

| Metric | Score |
|---|---|
| Accuracy | 0.5489 |
| Precision (macro) | 0.4376 |
| Recall (macro) | 0.4946 |
| F1-score (macro) | 0.4252 |

**Per-class breakdown:**

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| 0 (no disease) | 0.90 | 0.74 | 0.81 | 82 |
| 1 | 0.61 | 0.42 | 0.49 | 53 |
| 2 | 0.33 | 0.41 | 0.37 | 22 |
| 3 | 0.21 | 0.24 | 0.22 | 21 |
| 4 (most severe) | 0.14 | 0.67 | 0.23 | 6 |

**What this tells us:**

- The model separates "no disease" (class 0) from everything else reasonably well (F1 = 0.81), which is the most clinically actionable distinction in this dataset.
- Performance degrades sharply as severity increases and support shrinks. Class 4 has only 6 test samples, so its recall (0.67 → 4 out of 6 caught) is encouraging but statistically unstable, and its precision (0.14) shows the model is over-predicting this class.
- This pattern is a direct, expected consequence of `class_weight="balanced"`: the model is explicitly penalized for ignoring minority classes, so it now over-predicts them (high recall, low precision) rather than ignoring them entirely (which a default, unweighted model would do). This is a deliberate trade-off, not a bug — for a screening use case, catching severe cases (recall) is arguably more important than avoiding false alarms (precision), but this trade-off should be revisited once more model families are compared.
- The confusion matrix (`reports/confusion_matrix.png`, generated by `python -m src.train`) shows most misclassifications land on *adjacent* severity classes (e.g., 1 predicted as 2) rather than distant ones — the model's errors are "close," not random.

This baseline is the reference point the next phase (model comparison and tuning) will be measured against.

## Roadmap

- [ ] Compare multiple model families (Random Forest, Gradient Boosting) under identical stratified cross-validation
- [ ] Revisit the precision/recall trade-off above with class-weighting strategies tuned per model
- [ ] Hyperparameter tuning for the selected model (GridSearchCV / Optuna)
- [ ] Unit test suite (`pytest`) covering data splitting, preprocessing, and domain cleaning
- [ ] Model serving API (FastAPI) with input validation
- [ ] Containerization (Docker)
- [ ] CI pipeline (GitHub Actions) running tests and linting on every push

## Tech Stack

- **Language:** Python 3.10+
- **ML:** scikit-learn
- **Data handling:** pandas
- **Evaluation/visualization:** matplotlib
- **Model persistence:** joblib

## License

This project is licensed under the MIT License.

---

*Built as part of an ongoing effort to apply production-grade software engineering practices to machine learning projects.*