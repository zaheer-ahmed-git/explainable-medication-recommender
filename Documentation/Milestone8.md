# Milestone 8 Execution Plan: Graph Suitability

## Summary

Milestone 8 is a gate-first graph-readiness stage. It builds train-only,
concept-level graph artifacts from completed Milestone 6 outputs and produces
aggregate reports that decide whether a heterogeneous graph branch is worth a
later modeling milestone.

It does not train the Transformer-GNN model, add external DDI or ontology
sources, run pooled MIMIC/eICU training, or make clinical recommendation claims.

## Implementation Status

Implemented for code, synthetic tests, and protected-data graph-readiness:

- `pipeline.graph_suitability` builds local ignored graph edges under
  `$DATASET_ROOT/processed/graph/milestone8/`.
- Aggregate reports are written to `reports/milestone8_graph_schema.json`,
  `reports/milestone8_graph_suitability.json`, and
  `reports/milestone8_ablation_plan.json`.
- `scripts/calculco/graph_suitability.sh` runs the protected-data job through
  OAR after Milestone 6 and Milestone 7 frozen-selection preflight checks.
- `notebooks/04_graph_suitability.ipynb` reads only aggregate JSON reports.
- `tests/test_graph_suitability.py` covers train-only graph fitting, leakage
  boundaries, cold-start reporting, sparse graphs, and report safety.

Protected-data materialization completed through Calculco job 5608. The gate
result is `pass_for_graph_ablation`, which authorizes Milestone 8B ablation
work but does not validate a clinical recommendation model.

## Graph Contract

Node types:

- `condition`
- `medication`
- `lab`
- `vital`
- `intervention`

Relation types:

- `condition_medication_train_positive`
- `condition_lab_predecision`
- `condition_vital_predecision`
- `condition_intervention_predecision`
- `medication_medication_train_coprescribed`

All learned graph statistics are fit from MIMIC train rows only. Validation,
test, and eICU rows are used only for aggregate coverage and cold-start
analysis. eICU remains coverage-only until in-catalog positive groups are
available.

## Commands

Lightweight synthetic verification:

```bash
uv run pytest tests/test_config.py tests/test_graph_suitability.py
```

Protected-data materialization on Calculco:

```bash
oarsub -O "$PROJECT_HOME/scripts/calculco/logs/rm_graph_%jobid%.out" \
       -E "$PROJECT_HOME/scripts/calculco/logs/rm_graph_%jobid%.err" \
       -S "$PROJECT_HOME/scripts/calculco/graph_suitability.sh"
```

## Acceptance Gates

- Milestone 6 feature and training artifacts are complete.
- Milestone 7 frozen selection exists; final test evaluation may still be
  pending, but graph output remains readiness-only.
- Graph edges are fit from MIMIC train only.
- Reports contain aggregate counts only and no patient identifiers or row
  samples.
- `milestone8_graph_suitability.json` records node/edge counts, degree
  summaries, connected components, sparsity, split coverage, cold-start rates,
  leakage audit, and a graph-readiness gate result.
- Any later GNN or hybrid model must first pass the separate Milestone 8B
  graph-aware ablation gate.
