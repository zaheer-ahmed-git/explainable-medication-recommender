# Dataset Foundation and Hybrid Recommendation Preparation Plan

## Summary

This plan prepares MIMIC-IV v3.1, MIMIC-IV-Note v2.2, and eICU-CRD v2.0 for a clinician-facing Hybrid Medical Recommendation System. The immediate goal is not to train the Transformer-GNN model yet, but to create a reproducible, explainable data foundation: dataset inventory, EDA, quality reports, preprocessing decisions, model-ready schemas, feature strategy, graph-readiness analysis, and stakeholder-ready findings.

Default direction: analyze all ICU stays across MIMIC and eICU, use sepsis as the first deep-dive condition, develop on MIMIC, and reserve eICU for external validation until harmonization quality is proven.

## Execution Roadmap

1. **Governance and Source Inventory**
   - Verify source versions, file locations, licenses, checksums, and table sizes without printing patient-level rows.
   - Resolve current doc discrepancy: MIMIC-IV-Note is present locally under `Dataset/2.2/note`; older notes saying it is absent should be treated as stale.
   - Deliverables: source inventory manifest, table-size summary, data-use constraints, meeting slide on “available data and safe-use rules.”

2. **Cohort Definition**
   - Define ICU-stay-level cohorts with source-qualified IDs: `source`, `patient_uid`, `encounter_uid`, `stay_uid`.
   - MIMIC: adult ICU stays from `patients`, `admissions`, `icustays`; decide first ICU stay per admission as the default.
   - eICU: one row per `patientunitstayid`; document handling of multiple stays and age values such as `> 89`.
   - Sepsis: create the first reproducible deep-dive cohort after approving diagnosis criteria.
   - Deliverables: cohort manifest, attrition funnel, key-integrity report.

3. **Schema and Data Quality Profiling**
   - Profile required tables only: demographics, admissions/stays, diagnoses, procedures/treatments, labs, vitals, medications, allergies, severity, optional notes metadata.
   - Compute row counts, key uniqueness, null rates, duplicate rates, timestamp coverage, unit consistency, categorical cardinality, and plausible value ranges.
   - Use DuckDB projection/chunked reads for large tables such as MIMIC `labevents`, `chartevents`, `emar`, `pharmacy`, and eICU `vitalPeriodic`, `nurseCharting`, `lab`.
   - Deliverables: `quality_profile.json`, schema-quality notebook, aggregate figures only.

4. **EDA and Dataset Understanding**
   - Produce descriptive statistics for age, sex, race/ethnicity, ICU type, LOS, admission context, diagnoses, medications, labs, vitals, allergies, severity, and source coverage.
   - Compare MIMIC versus eICU distributions side by side to identify domain shift.
   - Analyze medication frequency by condition, diagnosis-medication co-occurrence, lab/vital missingness, and candidate-medication coverage.
   - Deliverables: EDA notebook set, figure pack, stakeholder summary of major dataset patterns.

5. **Harmonization**
   - Map both sources into common source-tagged schemas for patients, stays, conditions, medications, labs, vitals, allergies, interventions, and temporal events.
   - Conditions: normalize ICD-9/10 and eICU diagnosis strings into shared condition tokens or roll-ups.
   - Medications: normalize free-text and coded medication fields to ingredient/class level, with RxNorm/ATC mapping where available.
   - Labs/vitals: harmonize core concepts such as creatinine, lactate, WBC, platelets, sodium, potassium, glucose, HR, MAP, SpO2, and temperature.
   - Deliverables: mapping coverage tables, unmapped-concept report, harmonization-overlap notebook.

6. **Temporal Contract and Label Construction**
   - Default contract: `t0 = ICU admission`, `t_pred = t0 + 24h`, feature window ends at `t_pred`, label window covers medications after `t_pred` through `t_pred + 24h`.
   - Use observed prescriptions/orders as historical labels, not proof of optimal treatment.
   - Exclude future outcomes, discharge-time-only features, full-corpus popularity, and target-leaking medication history by default.
   - Generate one row per `patient/stay + condition + candidate medication` with `label_prescribed`.
   - Build candidate catalogs from training patients only.

7. **Preprocessing Pipeline**
   - Mandatory: clean invalid IDs, deduplicate, validate joins, preserve original source fields, normalize categorical values, parse timestamps/offsets, handle cancelled orders, and record provenance.
   - Mandatory: add missingness indicators for important labs/vitals; fit imputation, scaling, encoding, vocabularies, and candidate catalogs on training data only.
   - Mandatory: patient-level train/validation/test split for MIMIC and external split for eICU.
   - Optional: note embeddings, molecular drug features, curated DDI/rule sources, pooled MIMIC+eICU training after coverage gates pass.

8. **Feature Engineering**
   - Static/context: age, sex, race/ethnicity, admission type, ICU unit, source, LOS context.
   - Conditions: diagnosis flags, condition roll-ups, comorbidity counts, sepsis indicators, ICD ontology features.
   - Labs/vitals: last/min/max/mean/trend, abnormal flags, missingness rates.
   - Medications/interventions: prior medication classes before cutoff, procedures/treatments, ventilation/dialysis indicators where temporally valid.
   - Safety/constraints: allergy flags, renal impairment proxy, candidate-vs-current-med DDI risk where supported.
   - Deliverable: feature dictionary with rationale, source, window, leakage risk, and intended model branch.

9. **Model-Ready Artifacts**
   - `cohort_stays`: stay manifest, demographics, source, split, index time, prediction time.
   - `patient_stay_features`: one row per stay with pre-decision tabular features.
   - `patient_condition_medication`: primary ranking table with candidate medication, condition, label, split, and rank group.
   - `event_sequences`: ordered pre-decision events for Transformer inputs.
   - `graph_edges`: global heterogeneous graph edges with relation type and provenance.
   - `patient_subgraphs`: per-stay graph batches for GNN scoring.
   - Supporting artifacts: vocabularies, candidate catalog, split manifest, preprocessing report, data dictionary.

10. **Graph and Hybrid-Model Readiness**
   - Quantify node counts, edge counts, degree distributions, connected components, sparsity, cold-start rates, relation coverage, and leakage risk.
   - Transformer inputs: tabular stay features, event sequences, optional note embeddings.
   - GNN inputs: patient-condition-medication-lab/intervention graph, DDI/ontology/co-occurrence edges, per-stay subgraphs.
   - Fusion target: score each candidate medication within a `stay + condition` ranking group.
   - Gate: build baselines before the Transformer-GNN model; hybrid complexity must improve over transparent baselines.

## EDA and Visualization Plan

- Dataset inventory: table-size heatmap, source-domain coverage matrix.
- Cohort: attrition funnel, MIMIC/eICU cohort comparison, ICU stay counts by source.
- Quality: null-rate bar charts, missingness matrices, duplicate/key-integrity summaries, timestamp coverage plots.
- Distributions: age, LOS, labs, vitals, medication counts, diagnosis counts.
- Clinical patterns: top diagnoses, top medications by condition, medication co-occurrence network, diagnosis-medication bipartite view.
- Harmonization: MIMIC/eICU concept-overlap charts, medication mapping coverage, lab/vital unit compatibility.
- Modeling readiness: class balance by condition, candidate coverage@N, feature missingness by split/source, graph density/connectivity plots.
- Meeting visuals: 8-10 slide figure pack emphasizing findings, risks, preprocessing rationale, and next actions.

## Testing and Acceptance Criteria

- Use only synthetic fixtures in tests.
- Required checks: path/source validation, bounded reads, schema contracts, key uniqueness, patient split integrity, temporal cutoff enforcement, candidate catalogs from training only, unmapped concept reporting, no patient-level output in reports.
- Verification commands: `uv run pytest`, `uv run ruff check .`, `uv run ruff format --check .`.
- Acceptance: every artifact records source, cohort version, split seed, feature window, label window, mapping version, and preprocessing version.

## Risks and Mitigations

- **Temporal leakage:** enforce cutoff tests and exclude outcome/discharge-only fields by default.
- **Patient leakage:** split by patient, never row or stay alone.
- **Observed-label bias:** present labels as historical prescribing behavior, not optimal recommendations.
- **Domain shift:** treat MIMIC and eICU differences as reportable findings; use eICU as external validation first.
- **Missing-not-at-random labs:** include missingness indicators and discuss clinical ordering bias.
- **Medication normalization errors:** report unmapped and ambiguous drugs; keep original source values in restricted artifacts.
- **Sparse graph relations:** quantify graph suitability before committing to GNN architecture.
- **Clinical overclaiming:** stakeholder materials must label outputs as research decision support requiring clinician review.

## Stakeholder Meeting Briefing Structure

1. Project objective and current status: data-foundation stage, not validated clinical recommender.
2. Available datasets and safe-use constraints.
3. Dataset structure: key tables, source differences, and unit of analysis.
4. Data quality findings: missingness, duplicates, key integrity, timestamp/unit issues.
5. EDA findings: population, conditions, medications, labs/vitals, source comparisons.
6. Preprocessing decisions: cohort, temporal windows, harmonization, splitting, leakage controls.
7. Recommendation signals: candidate medications, condition-medication patterns, labs/vitals, comorbidities, graph relations.
8. Model-ready schema: ranking table, features, sequences, graph artifacts.
9. Risks and limitations: bias, leakage, missingness, external validity, observational labels.
10. Next milestones: pipeline skeleton, cohort/profile reports, harmonization, training table, baselines, graph-readiness gate.

## Assumptions and Defaults

- Use ICU stay as the primary analysis unit.
- Analyze all conditions broadly, with sepsis as the first deep-dive.
- Use MIMIC for development and eICU for external validation unless harmonization gates justify pooled training.
- Use `t_pred = ICU admission + 24h` and a 24-hour post-prediction label window as the first reviewed temporal contract.
- Keep clinical text optional for this phase; use note metadata or derived embeddings only after safety and leakage review.
- Do not train the final Transformer-GNN model during this dataset-understanding phase; prepare evidence, baselines, and model-ready artifacts first.
