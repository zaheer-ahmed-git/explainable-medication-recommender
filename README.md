## ResearchModule: Explainable Medication Recommendation Pipeline

End-to-end research prototype for building a medication recommendation system from synthetic EHR data.  
The current codebase includes data preprocessing, a baseline ML ranking model, and an XGBoost-based ranker.

## Project Goal

Build a model that uses patient context (condition, diagnoses, labs, demographics, comorbidity features) to rank likely medications for each patient-condition pair, with a path toward explainable and conversational clinical decision support.

## Current Status

- Data preprocessing pipeline implemented
- Patient-level train/valid/test splitting implemented
- Baseline medication ranker implemented (`SGDClassifier` logistic loss)
- Advanced medication ranker implemented (`XGBoost`)
- Ranking metrics and artifact export implemented

## Repository Structure

- `data_preprocessing.py`  
  Builds model-ready artifacts from raw CSVs.
- `medication_ranking_model.py`  
  Baseline ML medication ranking model (linear SGD logistic classifier).
- `xgboost_medication_ranker.py`  
  Two-stage recommendation pipeline with XGBoost ranking.
- `DataAnalysis.ipynb`  
  EDA and visualization notebook.
- `Datasets/`  
  Raw synthetic EHR source files and generated images.
- `Datasets/processed/`  
  Generated training tables and preprocessing report.
- `Documentation/`  
  Research notes and architecture docs.

## Data Flow

Raw tables:

- `patients.csv`
- `diagnoses.csv`
- `lab_results.csv`
- `medications.csv`
- `outcomes.csv`

Preprocessing outputs:

- `Datasets/processed/patient_features.csv`
- `Datasets/processed/patient_condition_medication.csv`
- `Datasets/processed/preprocessing_report.json`

Model outputs:

- Baseline model output directory (default): `Models/medication_ranker/`
- XGBoost model output directory (default): `Models/xgboost_medication_ranker/`

## Modeling Approach

### 1) Candidate training table

`data_preprocessing.py` constructs rows of:

`patient_id + condition + medication -> label_prescribed`

- Positive label (`1`): medication observed for that patient-condition
- Negative label (`0`): medication is condition-plausible but not observed for that patient

### 2) Baseline ranker

`medication_ranking_model.py`:

- Uses sklearn pipeline (imputation + encoding + scaling + SGD classifier)
- Predicts prescription probability per candidate row
- Ranks medications within each `patient_id + condition` group

### 3) XGBoost ranker

`xgboost_medication_ranker.py`:

- Builds candidate catalog from observed positives
- Trains `XGBClassifier` for candidate scoring
- Produces top-k recommendations and scored candidate outputs

## Evaluation Metrics

Implemented metrics include:

- Binary metrics:
  - `average_precision`
  - `roc_auc`
- Ranking metrics (per patient-condition group):
  - `precision@k`
  - `recall@k`
  - `hit_rate@k`
  - `ndcg@k`
  - `mrr@k`

## Environment Setup (uv)

This project uses `uv` for dependency management and execution.

1. Install dependencies:

```powershell
uv sync
```

2. Run scripts with:

```powershell
uv run <script>.py
```

## How to Run

### Step 1: Build processed training data

```powershell
uv run data_preprocessing.py --data-dir Datasets --output-dir Datasets/processed
```

Optional controls:

- `--max-candidates-per-condition` (default: `20`; use `0` for all)
- `--min-candidate-count` (default: `5`)

### Step 2: Train baseline medication ranker

Quick experiment:

```powershell
uv run medication_ranking_model.py --training-table Datasets/processed/patient_condition_medication.csv --output-dir Models/medication_ranker_quick --max-rows 50000
```

Full run:

```powershell
uv run medication_ranking_model.py --training-table Datasets/processed/patient_condition_medication.csv --output-dir Models/medication_ranker
```

### Step 3: Train XGBoost medication ranker

```powershell
uv run xgboost_medication_ranker.py --training-table Datasets/processed/patient_condition_medication.csv --output-dir Models/xgboost_medication_ranker
```

## Leakage Controls

By default, modeling scripts exclude potentially leaky features:

- `medication_*` history features
- `outcome_*` features
- `candidate_*` popularity features

You can enable them explicitly via CLI flags for ablation studies, but keep defaults for cleaner baseline reporting.

## Example Artifacts

- `Datasets/processed/preprocessing_report.json`: table-level stats and split counts
- `metrics.json`: current baseline metric snapshot
- `Models/.../model.pkl`: serialized model pipeline and metadata
- `Models/.../scored_candidates.csv`: candidate-level scores
- `Models/.../top_recommendations.csv` (XGBoost): top-k recommendations per patient-condition

## Research Direction

This repository currently covers the structured-data recommendation core.  
Planned work includes:

- stronger benchmark comparisons (random/popularity baselines)
- per-condition evaluation and calibration
- explainability layer integration (feature attribution + evidence logging)
- conversational/LLM integration for clinician-facing interaction

## Notes

- Dataset is synthetic and intended for research/education.
- This system is decision-support oriented and not a replacement for clinical judgment.
