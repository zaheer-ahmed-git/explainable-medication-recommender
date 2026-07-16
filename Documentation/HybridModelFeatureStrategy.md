# Hybrid Model Feature Strategy

## Purpose and status

This document records a **reviewed planning direction** for feature families,
branch boundaries, and selection gates that should guide a future hybrid
Transformer and heterogeneous GNN medication ranker.

It does **not** change the current repository scope. As of 2026-07-11:

- Milestone 6 feature and label artifacts are implemented.
- Milestone 7 learned baselines (frozen `xgboost` reference) are implemented.
- Milestone 8 train-only concept graphs and Milestone 8B graph-aware ablations
  are implemented.
- Phase 8 P0 stay-level condition, trend, and explicit missingness features are
  implemented behind `pipeline.features --feature-set phase8_p0` for isolated
  ablation roots. They are not the canonical default until protected-data
  reruns and `phase8_p0_feature_gate_review.json` justify promotion.
- Full Transformer-GNN neural training, note embeddings, and external DDI/ontology
  edges are **not** implemented.

Treat every recommendation here as **planned work** until it appears in
`pipeline/`, manifests, tests, and milestone exit gates. This document must not
be read as proof of clinical validity or as a completed recommender.

Canonical alignment:

- Research framing: `Documentation/ResearchDetail.md`
- System boundaries: `ARCHITECTURE.md`
- Implemented feature contract: `Documentation/Milestone6FeatureLabelDictionary.md`
- Implemented graph contract: `Documentation/Milestone8.md`
- Graph ablation gate: `Documentation/Milestone8B.md`
- Phased roadmap: `Documentation/DataFoundationRoadmap.md`

## Task contract (unchanged)

The primary structured task remains **ranking condition-appropriate medication
candidates** for an ICU/unit stay, not visit-level multi-label drug-combination
classification on raw MIMIC-III-style benchmarks.

Locked defaults from Milestone 6:

- Unit of analysis: ICU/unit stay with source-qualified IDs.
- `t0`: ICU/unit admission.
- Feature cutoff: `t_pred = t0 + 24h`.
- Label window: medication starts with `24h < start_time <= 48h`.
- Observed prescriptions are **historical labels**, not optimal treatment.
- Unobserved candidates are weak observational negatives only.
- Candidate catalogs are learned from MIMIC train positives only.
- Patient-level splits are deterministic; eICU is external until coverage gates
  pass.

Any feature work must preserve this temporal contract and leakage boundaries.

## What is implemented today

### Stay-level tabular features (`patient_stay_features.parquet`)

Built by `pipeline.features` (see
`Documentation/Milestone6FeatureLabelDictionary.md`):

- Demographics and admission context: `age_years`, `sex`,
  `race_or_ethnicity`, `admission_type`, `admission_source`, `unit_type`,
  `last_unit_type`, `stay_type`, `stay_sequence`, `hospital_id`, `ward_id`.
- Aggregate pre-24h lab summaries for core tokens: creatinine, lactate, wbc,
  platelets, sodium, potassium, glucose (count, observed flag, min, mean, max
  per token).
- Aggregate pre-24h vital summaries for core tokens: heart_rate,
  mean_arterial_pressure, spo2, temperature, respiratory_rate.
- Allergy and intervention aggregates: `allergy_constraint_present_24h`,
  `predecision_intervention_count_24h`, and related counts.

**Default exclusions (implemented):** medication-history features in stay
features; pre-decision medication events in default event sequences.

**Known gap (documented, not yet implemented):** stay-level diagnosis or
comorbidity multi-hot features are **not** in `patient_stay_features.parquet`.
Conditions enter the ranking table as `index_condition_token` per row. Untimed
condition rows in `temporal_events` are excluded from default event sequences.

### Event sequences (`event_sequences.parquet`)

Pre-decision events with `event_time_hours_from_admit <= 24h`. Default event
types: condition, lab, vital, allergy, intervention (medication excluded unless
a reviewed CLI flag is enabled).

### Row-level ranking features (Milestone 7 / 8B)

From `pipeline.learned_baselines` and `pipeline.graph_ablation`:

- `index_condition_token`, `candidate_medication_token`, `candidate_rank`
- Milestone 8B graph-derived numeric columns (17 features; see below)

### Graph schema (Milestone 8, implemented)

Node types: `condition`, `medication`, `lab`, `vital`, `intervention`.

Relation types (train-fit, concept-level):

- `condition_medication_train_positive`
- `condition_lab_predecision`
- `condition_vital_predecision`
- `condition_intervention_predecision`
- `medication_medication_train_coprescribed`

Graph edges must not use validation, test, or eICU labels for fitting.

## Bottom-line recommendations

These are the planning defaults for hybrid work. They do not authorize skipping
Milestone 8B or baseline gates.

1. **Do not use every dataset column.** Target roughly **100–150 curated
   stay-level dimensions** plus a **bounded pre-decision event sequence** for the
   Transformer branch. For the GNN branch, keep the **five implemented node
   types** and expand relation families from **five implemented types** to at
   most **five to eight** only after curated sources and leakage review.

2. **Close the condition-feature gap before scaling model complexity.** Literature
   medication-recommendation models treat diagnoses as primary encoder input.
   The largest gap versus that literature and the current
   `patient_stay_features.parquet` is **stay-level condition/comorbidity
   context**. Add this in a reviewed Milestone 6 extension before PyTorch/PyG
   investment.

3. **Keep the lab/vital core physiology-focused.** The seven labs and five
   vitals in Milestone 6 match sepsis and ICU medication literature (renal,
   perfusion, infection, acid–base, oxygenation). Expand selectively (for
   example bicarbonate, BUN, bilirubin, hemoglobin) only through ablation, not
   by importing all harmonized lab tokens.

4. **Separate branch responsibilities.**
   - **Transformer:** temporal within-stay context, cross-feature interactions,
     optional text later.
   - **GNN:** relational medication signal (indication, coprescription; DDI and
     ontology later).

5. **Select features with existing project gates.** Use validation **NDCG@10**
   as the primary metric, **+0.005 absolute lift** over the frozen reference
   where applicable (Milestone 8B precedent), tie-breakers **MRR@10**, **Hit@10**,
   then simpler models. Always run leakage audits. Do not rely on offline
   univariate ranking alone.

6. **Defer high-cost modalities until structured hybrid beats XGBoost + graph
   ablation.** Notes, patient–patient similarity edges, and external DDI/ontology
   graphs remain **Phase 2+** per Milestone 8/8B scope.

## Branch-specific essential features

Status labels: **Implemented**, **Implemented (P0 ablation)**,
**Planned (P1+)**. P0 ablation means code and synthetic tests exist, but the
default canonical artifact roots are unchanged until the promotion gate passes.

### Transformer branch

| Feature family | Status | Rationale |
| --- | --- | --- |
| Demographics and admission context | Implemented | Baseline patient and care-setting context |
| Core lab summaries (7 tokens) | Implemented | Renal, perfusion, infection proxies |
| Core vital summaries (5 tokens) | Implemented | Hemodynamic and respiratory state |
| Allergy constraint presence | Implemented | Safety constraint signal |
| Pre-decision intervention counts | Implemented | Acuity and care-pathway proxy |
| Pre-decision event sequence | Implemented | Temporal interactions among labs, vitals, interventions |
| Index + comorbid condition multi-hot | Implemented (P0 ablation) | Train-only condition vocabulary in `phase8_p0`; protected-data lift pending |
| Last-value and trend lab/vital summaries | Implemented (P0 ablation) | First/last/delta/slope/hours-since-last for core labs/vitals; protected-data lift pending |
| Per-token missingness indicators | Implemented (P0 ablation) | Explicit complementary missingness flags for core labs/vitals; protected-data lift pending |
| Severity scores (SOFA/APACHE components) | Planned (P1+) | Common in sepsis medication prediction; requires explicit extraction contract |
| Expanded lab panel (selective) | Planned (P1+) | Add only if validation ablation shows lift |
| Pre-decision medication history | Planned (P1+, gated) | Leakage-sensitive; only meds strictly before `t_pred`, never label-window overlap |
| Note embeddings | Planned (P1+) | Optional; governance and leakage review required |

**Transformer should not receive (default):** raw hospital/ward identifiers as
learned features, discharge outcomes, label-window medications, global
medication popularity, unbounded ICD or itemid vocabularies.

### GNN branch

| Graph element | Status | Rationale |
| --- | --- | --- |
| Five node types (condition, medication, lab, vital, intervention) | Implemented | Matches Milestone 8 heterogeneous concept graph |
| Five train-fit relation types | Implemented | Indication, physiology linkage, coprescription |
| Graph-derived candidate features (17 columns) | Implemented | Milestone 8B ablation surface |
| DDI / contraindication edges | Planned (P1+) | GAMENet, SafeDrug, AMGNet; requires curated external source |
| ATC hierarchy edges | Planned (P1+) | Cold-start generalization for medications |
| Condition–condition comorbidity edges | Planned (P1+) | Train co-occurrence among roll-ups |
| Patient–patient similarity edges | Planned (P1+, discouraged early) | Privacy and leakage risk; defer |

**GNN should not receive initially:** patient nodes in the global graph,
test-set or validation positives for edge fitting, outcome-labeled similarity
links.

### Fusion head (future hybrid)

| Input | Status | Notes |
| --- | --- | --- |
| Transformer stay embedding | Planned | After branch ablations |
| GNN subgraph embedding | Planned | Per `(stay context, candidate medication)` |
| Candidate token identity | Implemented (row-level) | Via ranking table |
| `candidate_rank` weak prior | Implemented | Ablate early in neural fusion |
| Milestone 8B graph numeric features | Implemented | Replace with learned GNN embeddings if 8B lift is positive |

## Implemented graph-derived feature columns (Milestone 8B)

These columns in `pipeline.graph_ablation.GRAPH_NUMERIC_FEATURE_COLUMNS` are
the current relational feature surface:

- `graph_condition_medication_support_count`
- `graph_condition_medication_log_support`
- `graph_condition_medication_support_share`
- `graph_condition_total_medication_support`
- `graph_condition_medication_degree`
- `graph_condition_lab_degree`
- `graph_condition_vital_degree`
- `graph_condition_intervention_degree`
- `graph_condition_total_degree`
- `graph_condition_total_support`
- `graph_candidate_medication_degree`
- `graph_candidate_medication_support`
- `graph_candidate_coprescription_degree`
- `graph_candidate_coprescription_support`
- `graph_condition_in_graph`
- `graph_candidate_in_graph`
- `graph_direct_edge_present`

A future heterogeneous GNN should learn over the **same subgraph schema**, not
an unrelated edge dump.

## P0 starter cardinalities

These are design targets for Phase 8 P0 and later hybrid work. Actual
protected-data column counts must be read from
`reports/phase8_p0_milestone6_feature_manifest.json` after materialization:

**Transformer P0**

- Static/context: ~10–15 columns (demographics, admission, allergy/intervention
  flags).
- Condition multi-hot: ~20–40 roll-up tokens (train-fitted vocabulary).
- Core lab summaries: ~35 columns (7 labs × count/observed/min/mean/max/last/
  missing).
- Core vital summaries: ~30 columns (5 vitals × same pattern).
- Event sequence: cap 128–256 events; types
  `condition`, `lab`, `vital`, `intervention`, `allergy`; include `event_token`,
  `value_numeric`, `event_time_hours_from_admit`.

**GNN P0**

- Nodes: five implemented types.
- Edges: five implemented relation types with train support thresholds.
- Query subgraph: index condition + candidate medication + one-hop lab/vital/
  intervention neighbors + coprescribing medication neighbors from train.

## Phase 8 P0 ablation surface

The implemented Phase 8 P0 switch keeps the default `temporal-features-v1`
artifacts unchanged and writes `temporal-features-v2` only when
`--feature-set phase8_p0` is selected. It adds:

- train-fit condition presence columns from MIMIC train stay frequency, capped
  by `--condition-feature-top-n` and preferring `normalized_condition_token`
  over `project_condition_token`;
- trend columns for the seven core labs and five core vitals:
  first, last, delta, least-squares slope, and hours since last value inside
  the 0-24h window;
- explicit `*_missing_24h` indicators for the same core lab/vital tokens;
- aggregate-only manifest fields for feature-family counts, condition columns,
  and OOV counts without listing OOV token values.

Promotion requires rebuilding isolated Phase 8 P0 training/preprocessing
artifacts, rerunning Milestone 7, Milestone 8, and Milestone 8B on those roots,
then passing `pipeline.feature_gate_review` against the current canonical
Milestone 8B reference. No neural framework or clinical recommendation claim is
introduced by this ablation.

## Feature-selection strategy

Use the same evaluation machinery already in the repository. This is a **process**
for future milestones, not a new CLI unless separately implemented.

### Phase 0 — Domain prior and leakage denylist

Maintain a feature dictionary with: name, source artifact, window, branch,
leakage risk, cardinality, status (implemented/planned).

Hard exclude regardless of univariate score:

- post-`t_pred` events and discharge-only fields;
- label-window medications and outcomes used as predictors;
- global medication popularity fit outside train;
- patient identifiers in model inputs or public reports.

### Phase 1 — Train-only univariate screen

On MIMIC train only:

- prevalence, missing rate, and near-zero variance checks;
- mutual information with `label_prescribed` **within condition strata**;
- mRMR-style redundancy reduction within families (labs, vitals).

### Phase 2 — Wrapper ablation with Milestone 7 / 8B tooling

Sequential backward elimination on **MIMIC validation** using **NDCG@10**:

1. demographics/context block;
2. lab summary block;
3. vital summary block;
4. allergy/intervention block;
5. graph-derived block (Milestone 8B experiments).

Drop a block only if validation NDCG@10 falls by less than 0.005 when removed
(Milestone 8B selection delta as precedent). Use SHAP on XGBoost as a **stability
check**, not the sole selector.

Existing commands (implemented):

```bash
uv run python -m pipeline.evaluate_baselines
uv run python -m pipeline.graph_ablation
```

### Phase 3 — Transformer-specific ablation (future)

- Vocabulary caps for `event_token` by type (train frequency).
- Sequence length ablation {64, 128, 256}.
- Sequence-only vs sequence + static summaries vs summaries-only.

### Phase 4 — GNN-specific ablation (future)

- One relation family at a time beyond Milestone 8 five types.
- Subgraph radius (1-hop vs 2-hop).
- Cold-start reporting from Milestone 8 suitability metrics.

### Phase 5 — Fusion gate

Proceed to neural hybrid only if:

1. Milestone 8B frozen selection shows graph-aware lift over frozen XGBoost, or
   documents inconclusive evidence with a reviewed reason to continue; and
2. Phase 2–4 ablations show branch features beat their tabular/graph-only
   baselines under the same metrics and leakage audits.

Late fusion weights in Milestone 8B (`late_fusion_validation_weighted`) are the
precedent for a simple fusion baseline before end-to-end training.

## Phase 2+ deferrals (explicit)

Do not add these to the active pipeline until structured features beat the
Milestone 7 + 8B reference stack:

- MIMIC-IV-Note embeddings or dialogue text in the ranker;
- external DDI, indication, or ontology knowledge graphs;
- patient–patient similarity graphs;
- pooled MIMIC+eICU training before harmonization gates pass.

These remain aligned with the **research goal** in `ResearchDetail.md` but are
out of scope for the current data-foundation and ablation milestones.

## Sepsis deep-dive notes (planned conditioning)

When the sepsis cohort and index-condition policy in
`Documentation/SepsisCohortAndIndexConditionPolicy.md` is approved, prioritize
the same physiology core (lactate, MAP, creatinine, platelets, WBC) and
intervention flags (ventilation, vasopressors, dialysis) in ablations. This does
not change the general ICU feature contract for all conditions.

## Governance

- No raw or patient-level rows in Git, prompts, or aggregate reports.
- Every feature manifest must record: cohort version, feature version, split
  seed, window, mapping version, and branch assignment.
- Do not present observational ranking performance as clinical recommendation.
- Explanation layer consumes model evidence later (Milestone 9); the LLM does
  not invent prescription rationale.

## Next repository actions (ordered)

1. Materialize Phase 8 P0 artifacts in isolated roots and review
   `reports/phase8_p0_milestone6_feature_manifest.json`.
2. Rebuild Phase 8 P0 training/preprocessing artifacts, rerun Milestone 7 and
   Milestone 8/8B on the isolated roots, and write
   `reports/phase8_p0_feature_gate_review.json`.
3. Promote Phase 8 P0 only if validation NDCG@10 improves by the reviewed gate
   threshold without secondary metric regressions; otherwise record it as a
   negative or inconclusive ablation.
4. Only then author a hybrid modeling milestone plan (separate from this
   document) for Transformer and heterogeneous GNN training.

## Related literature (context only)

Medication-recommendation and ICU feature patterns were reviewed from GAMENet,
COGNet, AMGNet, MedGCN, DIAGNN, and recent sepsis ICU prediction work. Those
benchmarks use visit-level combination labels; this project adapts the **feature
families**, not the task definition, to the Milestone 6 stay-level ranking
contract.

See also `Documentation/SimilarPapers.md` for project-maintained related work.
