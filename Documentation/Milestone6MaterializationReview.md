# Milestone 6 Protected-Data Materialization Review

**Date:** 2026-07-06  
**Status:** completed (protected-data artifacts materialized)  
**Structured summary:** `reports/milestone6_materialization_review.json`

This document records the review of Milestone 6 implementation and
protected-data runs on Calculco. It complements the execution plan in
`Documentation/Milestone6.md` and the artifact dictionary in
`Documentation/Milestone6FeatureLabelDictionary.md`.

## Scope

Milestone 6 converts Milestone 5 harmonized tables into leakage-controlled
feature and observed-label artifacts. It does **not** train baselines, build
graph edges, or run recommendation models.

## Implementation (code)

The following were already implemented and verified with synthetic tests before
the protected-data runs:

| Component | Location |
|-----------|----------|
| Temporal features | `pipeline/features.py` |
| Training table and labels | `pipeline/build_training_table.py` |
| Path and version constants | `pipeline/config.py` |
| OAR wrappers | `scripts/calculco/features.sh`, `build_training_table.sh`, `milestone6.sh` |
| Tests | `tests/test_features.py`, `tests/test_build_training_table.py` |

Verification at review time: `uv run pytest tests/test_config.py tests/test_features.py tests/test_build_training_table.py` — 20 passed.

## OAR materialization

Both jobs were submitted from `ritchie.univ-littoral.fr` during the 2026
Calculco migration. Earlier `oarsub` attempts on the `calculco` login host
failed with **“not enough resources”** (no matching CPU nodes); `ritchie`
scheduling succeeded.

| Job ID | Script | Node | Started (UTC) | Exit | Approx. walltime |
|--------|--------|------|---------------|------|------------------|
| 830 | `features.sh` | chimay01 | 2026-07-05 20:11 | 0 | ~5 min |
| 1055 | `build_training_table.sh` | chimay08 | 2026-07-06 11:16 | 0 | ~23 s |

Logs: `scripts/calculco/logs/rm_features_830.{out,err}` and
`scripts/calculco/logs/rm_training_table_1055.{out,err}` (stderr empty for both).

## Artifacts produced

### Feature artifacts (`$DATASET_ROOT/processed/features/`)

| Table | Row count | Manifest |
|-------|-----------|----------|
| `cohort_decision_times.parquet` | 285,476 | `reports/milestone6_feature_manifest.json` |
| `patient_stay_features.parquet` | 285,476 | same |
| `event_sequences.parquet` | 212,570,568 | same |

Temporal contract: `t_pred = t0 + 24h`; label window 24h; split seed `20260617`;
pre-decision medications excluded from default event sequences.

Primary-eligible stays: MIMIC 40,740; eICU 77,718 (censored stays reported
separately in the feature manifest).

### Training artifacts (`$DATASET_ROOT/processed/training/`)

| Table | Row count | Manifest |
|-------|-----------|----------|
| `split_manifest.parquet` | 204,234 | `reports/training_table_manifest.json` |
| `candidate_catalog.parquet` | 32,547 | same |
| `patient_condition_medication.parquet` | 45,773,122 | same |

Candidate catalog: 713 conditions, top 50 candidates per condition from MIMIC
train positives only. Split integrity: 204,234 patients, **0** with multiple
splits.

## Exit gates

| Gate | Result |
|------|--------|
| Synthetic temporal/split/candidate tests | Pass |
| Protected feature artifacts | Complete |
| Protected training artifacts | Complete |
| Train-only candidate catalog | Pass |
| Default leakage controls | Pass |
| Aggregate-only manifests | Pass |

## Findings to review before Milestone 7

1. **eICU external evaluation:** `positive_row_count = 0` for the external
   split in the training table. This is expected when catalogs are MIMIC-train-only;
   eICU rows are weak observational negatives in the catalog cross-product.

2. **Out-of-catalog positives:** Observed label-window medications not in the
   train-derived catalog are substantial on MIMIC (~50% of positives on train)
   and 100% on eICU. Review before interpreting baseline recall or coverage.

3. **eICU condition mapping:** 431,397 eICU condition rows lack
   `index_condition_token` (source-native text). Ranking-group coverage on
   external data may be limited.

4. **Intermediate feature files:** `_event_sequences_predecision.parquet` and
   `_event_sequence_parts/` remain under `processed/features/` (~3.7 GB). Safe
   to delete after verifying `event_sequences.parquet`.

5. **Milestone 5 gates:** Harmonization completed 2026-07-05. Optional
   condition mapping resource files are still listed as missing in
   `reports/condition_normalization_coverage.json`; B1 (all CCS/CCSR categories)
   roll-up policy should be consciously accepted before pooled MIMIC+eICU training.

## Explicitly out of scope

- Milestone 7 baselines (random, popularity, linear, XGBoost)
- Graph artifacts (`graph_edges`, `patient_subgraphs`)
- Transformer-GNN training
- Pooled MIMIC+eICU training

## Next steps

1. Review `reports/training_table_manifest.json` (coverage losses, catalog
   counts, out-of-catalog positives).
2. Optionally reclaim disk from feature build intermediates.
3. Begin Milestone 7 transparent baselines after the catalog-coverage review.
