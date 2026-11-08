# CodexPLAN Step 10: Graph and Hybrid-Model Readiness

Status: **complete as a readiness gate review** (2026-07-18).  
Scope stack: Phase 8 P0 (`temporal-features-v2`) under
`$DATASET_ROOT/processed/phase8_p0/`.  
Clinical claim boundary: observational prescribing research only; not a
validated recommendation system and not Transformer-GNN training approval.

## Verdict

| Gate | Result | Meaning |
| --- | --- | --- |
| Graph structure suitability (Milestone 8) | **pass_for_graph_ablation** | Concept graph is connected, train-fit, leakage-audited, and usable for ablation |
| Hybrid complexity over transparent baselines (Milestone 8B) | **fail / not justified** | Frozen tabular XGBoost retained; no graph-aware candidate cleared +0.005 NDCG@10 |
| Phase 8 P0 feature promotion | **reject_inconclusive** | Isolated P0 stack did not clear the same lift gate vs canonical 8B reference |
| Proceed to neural Transformer-GNN | **no** | Keep transparent baselines; deepen structured features / mappings before hybrid neural work |

Primary sources (aggregate-only):

- `reports/phase8_p0_milestone8_graph_suitability.json`
- `reports/phase8_p0_milestone8_graph_schema.json`
- `reports/phase8_p0_milestone8_ablation_plan.json`
- `reports/phase8_p0_milestone8b_frozen_selection.json`
- `reports/phase8_p0_milestone8b_graph_feature_manifest.json`
- `reports/phase8_p0_patient_subgraphs_manifest.json`
- `reports/phase8_p0_model_ready_manifest.json`
- `reports/phase8_p0_feature_gate_review.json`
- `reports/milestone7_frozen_selection.json`
- Machine-readable summary: `reports/codexplan_step10_graph_hybrid_readiness.json`

## 1. Graph quantification (Milestone 8 on Phase 8 P0)

Generated: 2026-07-17. Graph version: `graph-suitability-v1`.  
Fit scope: MIMIC train only (`391416` edges).

### Nodes (concept-level global graph)

| Node type | Count | Mean degree | Min / max degree |
| --- | ---: | ---: | --- |
| condition | 714 | 514.1 | 10 / 2292 |
| medication | 751 | 108.2 | 1 / 1096 |
| lab | 639 | 309.8 | 7 / 713 |
| vital | 10 | 704.0 | 688 / 714 |
| intervention | 2365 | 54.7 | 2 / 685 |
| **total** | **4479** | — | — |

Connected components: **1** component covering all **4479** nodes; **0** singletons.

### Edges and relation coverage

| Relation | Edges | Density | Mean support |
| --- | ---: | ---: | ---: |
| `condition_medication_train_positive` | 32,597 | 0.061 | 63.3 | means only 6.1% of all possible condition–medication pairs actually occurred in your training data | Each condition–medication edge was observed in 63 patient encounters.

| `condition_lab_predecision` | 197,953 | 0.434 | 134.0 |
| `condition_vital_predecision` | 7,040 | 0.986 | 576.3 |
| `condition_intervention_predecision` | 129,482 | 0.077 | 23.0 |
| `medication_medication_train_coprescribed` | 24,344 | 0.086 | 223.6 | Meaning about 8.6% of all possible medication pairs are ever co-prescribed.
| **total** | **391,416** | — | — |

Sparsity reading:

- Condition–medication indication edges are sparse enough to carry selective
  ranking signal (~6% of possible condition×medication pairs in the train
  vocabulary).
- Condition–vital edges are nearly complete for the 10 core vitals (expected
  for a small curated vital set).
- Condition–lab edges are dense relative to the other families (~43%).
- Coprescription edges exist but are not a dense drug–drug clique.

Deferred relation families (schema-explicit, not in this graph): external DDI,
ontology, note embeddings, clinical-rule edges.

### Cold-start and positive coverage

| Source / split | Candidate cold-start | Condition cold-start | Positive graph coverage |
| --- | ---: | ---: | ---: |
| MIMIC train | 0.0 | 0.0 | 1.0 (2,062,027 positives) |
| MIMIC validation | 0.0 | 0.0 | 1.0 (245,803 positives) |
| MIMIC test | 0.0 | 0.0 | 1.0 (251,984 positives) |
| eICU external (default `rxnorm_or_atc`) | 0.0 | 0.0 | n/a (0 in-catalog positives) |

Candidate and condition tokens seen in evaluation splits are covered by the
train-fit graph vocabulary. eICU under the default token strategy remains
coverage-only for performance claims (0 positive ranking groups in the default
training table). ATC-3-first sensitivity is separately marked externally
evaluable in the model-ready package (see §5).

### Leakage audit

- Status: **pass**
- `train_only_graph_fit`: true
- Fit rows: MIMIC train only
- Blocked identifier columns in graph edges: none
- Validation / test / eICU used for coverage reporting only

Gate review criteria all true: edges present, condition–medication edges
present, connected component present, leakage audit passed →
`pass_for_graph_ablation`.

## 2. Per-stay subgraph readiness (GNN batch inputs)

`pipeline.patient_subgraphs` completed 2026-07-17 on the Phase 8 P0 stack.
Unit: one subgraph per `stay + condition` ranking group. Edges come only from
the train-fit global graph.

| Artifact | Rows |
| --- | ---: |
| `subgraph_index` | 916,107 |
| `subgraph_nodes` | 94,813,196 |
| `subgraph_edges` | 1,213,018,765 |
| `subgraph_candidates` | 45,794,222 |

| Source / split | Subgraphs | Nodes (sum) | Edges (sum) | Candidates | Cold candidates | Positives |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MIMIC train | 514,083 | 59,771,817 | 687,122,253 | 25,696,360 | 0 | 2,062,027 |
| MIMIC validation | 62,750 | 7,280,350 | 83,843,485 | 3,136,364 | 0 | 245,803 |
| MIMIC test | 65,062 | 7,567,559 | 86,929,216 | 3,251,466 | 0 | 251,984 |
| eICU external | 274,212 | 20,193,470 | 355,123,811 | 13,710,032 | 0 | 0 |

These artifacts are loadable GNN inputs for a later modeling milestone. Their
existence does **not** authorize neural training while the hybrid lift gate
fails.

## 3. Transformer inputs (available today)

| Input | Artifact | Status | Notes |
| --- | --- | --- | --- |
| Tabular stay features | `patient_stay_features` (285,476 stays) | Ready | Phase 8 P0: **215** columns (12 demo/context, 40 condition, 82 lab, 59 vital, 60 trend, 12 missingness, 5 allergy/intervention) |
| Event sequences | `event_sequences` (212,570,568 events) | Ready | Pre-decision only (`<= 24h`); default excludes medications |
| Ranking groups | `patient_condition_medication` (45,794,222 rows) | Ready | Score each candidate inside `stay + condition` |
| Note embeddings | — | Deferred P1 | Explicitly out of Step 10 / schema deferred list |
| Pre-decision medication history in stay features | — | Deferred / gated | Default exclusion remains |

Event-type volume (Phase 8 P0 feature manifest aggregates):

- MIMIC: lab 7.6M, vital 13.6M, intervention 0.47M (conditions untimed → excluded from sequences)
- eICU: lab 8.7M, vital 179.1M, intervention 2.9M, allergy 0.17M

## 4. GNN inputs (available today)

| Input | Status | Notes |
| --- | --- | --- |
| Patient–condition–medication–lab/vital/intervention concept graph | Ready | Five node types |
| Train-positive indication + predecision physiology/intervention + train coprescription | Ready | Five relation types |
| Per-stay subgraphs | Ready | §2 |
| Graph-derived tabular features (17 columns) | Ready | Milestone 8B ablation surface |
| External DDI / ontology / rule edges | Deferred P1 | Not required for this gate |

## 5. Fusion target and baseline gate

Fusion target (unchanged): score each candidate medication inside a
`stay + condition` ranking group (`ranking_group_id`), using observed
label-window prescriptions as historical labels.

Transparent baselines already exist (Milestone 7 frozen selection):

- Selected headline: **xgboost**
- MIMIC validation (k=10): NDCG 0.3695, MRR 0.4879, Hit 0.8483
- Comparators: random, global popularity, condition popularity, linear

Phase 8 P0 Milestone 8B validation (k=10, 57,949 positive ranking groups):

| Experiment | NDCG@10 | MRR@10 | Hit@10 | Δ NDCG vs frozen XGB |
| --- | ---: | ---: | ---: | ---: |
| `xgboost_frozen_reference` | 0.3749 | 0.4952 | 0.8532 | 0.0000 |
| `xgboost_graph_augmented` | 0.3768 | 0.4904 | 0.8540 | **+0.0019** |
| `late_fusion_validation_weighted` | 0.3749 | 0.4952 | 0.8532 | 0.0000 |
| `simple_ensemble_mean` | 0.3469 | 0.4666 | 0.8357 | −0.0280 |
| `graph_only_xgboost` | 0.2985 | 0.3975 | 0.8045 | −0.0764 |

Lift rule: require **≥ +0.005** absolute NDCG@10 without dropping MRR/Hit by
more than 0.01. Best graph-aware candidate (+0.0019) fails the rule.
Late-fusion selected graph weight: **0.0** (pure XGBoost).

Frozen 8B selection reason: *Frozen XGBoost retained because no graph-aware
candidate cleared the validation lift gate.*

eICU readiness nuance from `phase8_p0_model_ready_manifest.json`:

- Default `rxnorm_or_atc`: coverage-only (0 positive ranking groups)
- Sensitivity `atc3_or_rxnorm`: externally evaluable (148,916 positive ranking
  groups) for future external checks only; does not change the MIMIC hybrid
  lift failure

## 6. Leakage and safety checklist

| Control | Status |
| --- | --- |
| Patient-level MIMIC split (0 multi-split patients) | Pass |
| Feature cutoff `t_pred = t0 + 24h` | Pass |
| Label window `(24h, 48h]` observed meds | Pass |
| Graph edges fit on MIMIC train only | Pass |
| Subgraphs use train-fit edges only | Pass |
| Candidate catalog from MIMIC train positives | Pass |
| Reports aggregate-only / no row samples | Pass |
| Observed labels ≠ optimal treatment | Stated in manifests |
| eICU performance under default tokens | Blocked (0 positives) |

## 7. Gate decision and next actions

**Step 10 is complete as evidence.** The graph is structurally ready; hybrid
neural complexity is **not** justified yet.

Do next:

1. Keep frozen tabular XGBoost as the development reference.
2. Do not start PyTorch/PyG Transformer-GNN training on this stack.
3. Prefer cheaper lifts first: medication-mapping coverage (especially eICU /
   ATC-3 strategy review), condition-feature design beyond the rejected P0
   promotion, and targeted feature ablations under the same NDCG@10 gate.
4. Revisit learned GNN / Transformer branches only after a reviewed ablation
   clears the +0.005 NDCG@10 lift (or a newly documented gate with the same
   honesty about baselines).
5. If external validation is needed, use the ATC-3-first evaluable path
   explicitly; do not claim eICU performance under default RxNorm-first
   labels.

Related planning: `Documentation/HybridModelFeatureStrategy.md`,
`Documentation/Milestone8.md`, `Documentation/Milestone8B.md`.
