# Agent Memory

This file contains stable, versioned project facts. It is not a substitute for
task context, source-code inspection, or local agent memory.

## Stable Facts

- ULCO Calculco HPC access uses username `zahmed` (lab `lisic`)
 Server reference:
  `Documentation/CalculcoSetup.md`.
- The research topic is an explainable conversational medication recommender
  for clinician-facing decision support.
- Recommendation generation and explanation generation are separate modules.
- The target recommender combines Transformer context modeling and
  heterogeneous GNN relation modeling.
- Explanations should combine attribution, graph evidence, rule checks,
  uncertainty, contradiction handling, and provenance.
- The main structured task is ranking medication candidates for a
  patient/stay-condition context.
- MIMIC-IV v3.1, MIMIC-IV-Note v2.2, and eICU-CRD v2.0 are present locally
  under `Dataset/`.
- Raw datasets are licensed, de-identified, ignored, and non-redistributable.
- `uv` is the only supported Python environment and dependency tool.
- Python 3.13 is the declared runtime.

## Current Repository State

- The active Milestone 1 pipeline skeleton and focused tests are present as of
  2026-06-17.
- `pipeline/source_inventory.py` generates metadata-only
  `reports/source_inventory.json`; `reports/` remains ignored.
- `pipeline/cohort.py` generates adult MIMIC-IV/eICU cohort artifacts under
  ignored `Dataset/processed/cohorts/` and aggregate
  `reports/cohort_manifest.json`.
- `pipeline/profile_tables.py` generates aggregate-only
  `reports/quality_profile.json`; the latest default run completed 18 of 24
  configured structured tables and recorded scan failures for several large or
  malformed local source files.
- `pipeline/eda_summary.py` synthesizes aggregate inventory, cohort, and
  quality reports into ignored `reports/eda_dataset_understanding.json`,
  `reports/eda_dataset_understanding.md`, and figures under `reports/figures/`.
- `pipeline/source_integrity.py` checks the six profiling-blocked files against
  local `SHA256SUMS.txt` manifests and gzip integrity. The latest run found all
  six mismatched and gzip-failed, so those source files need re-transfer or
  re-download before feature work.
- Sepsis sub-cohort extraction, harmonization, EDA notebooks, feature tables,
  and models are not yet implemented.
- `DepreciatedCode/` contains the ignored synthetic prototype.
- The prototype includes preprocessing, deterministic patient splitting,
  linear and XGBoost ranking, and ranking metrics.
- `Documentation/ResearchDetail.md` is the current research framing.
- `Documentation/OldResearchDetail.md` is historical.
- `Documentation/DataFoundationRoadmap.md` is the implementation roadmap.
- `FinalPosterCDS.pdf` is an architectural research poster, not proof of a
  completed clinical system.

## Known Pitfalls

- The dataset directory is singular: `Dataset/`, not `Datasets/`.
- The legacy directory is spelled `DepreciatedCode/`; preserve the path until a
  deliberate migration.
- Older notes incorrectly state that MIMIC-IV-Note is absent.
- Older README content described active modules that are no longer in the
  working tree.
- MIMIC timestamps are shifted and are not real calendar dates.
- eICU is multi-center while MIMIC-IV is single-center; source differences are
  meaningful, not noise to erase.
- Observed prescriptions are not equivalent to optimal treatment labels.
- Unobserved candidate medications are not guaranteed clinical negatives.
- Outcome and medication-history features can leak future or target
  information.

## Do Not Do

- Do not commit or quote patient-level data.
- Do not load multi-gigabyte source tables into pandas.
- Do not pool MIMIC and eICU before measuring mapping and cohort compatibility.
- Do not claim clinical validity from synthetic or poster examples.
- Do not revive deleted code based only on stale documentation.
- Do not add dependencies outside `uv`.
