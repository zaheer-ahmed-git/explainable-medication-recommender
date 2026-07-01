# Coding Standards

## Python

- Target Python 3.13.
- Use `pathlib.Path` for filesystem paths.
- Add type annotations to public functions and non-obvious internal functions.
- Prefer small pure functions around data contracts and isolate I/O.
- Use descriptive `snake_case` names and uppercase constants.
- Keep command-line entry points thin; put logic in importable functions.
- Use concise docstrings for contracts, assumptions, and leakage-sensitive
  behavior.
- Do not suppress exceptions that indicate invalid schemas or unsafe data.

## Data Processing

- Use DuckDB projection, filters, and joins for large compressed tables.
- Use pandas only after data is cohort-filtered and bounded.
- Select required columns explicitly.
- Validate required columns before processing.
- Preserve source, identifiers, units, timestamps, and mapping provenance.
- Avoid implicit type conversion for identifiers and clinical codes.
- Normalize missing values deliberately; do not silently convert missingness to
  a clinically meaningful zero.
- Record counts before and after filters, joins, and deduplication.

## Time and Leakage

- Make index time, feature window, and label window explicit.
- Fit vocabularies, candidate catalogs, imputers, scalers, and encoders on the
  training partition only.
- Split by patient before model selection.
- Do not use future notes, future labs, post-treatment outcomes, or test-set
  graph edges as features.
- Keep leakage-sensitive feature switches off by default.

## Reproducibility

- Centralize random seeds and cohort parameters.
- Prefer deterministic hash-based patient assignment when practical.
- Write machine-readable manifests beside derived artifacts.
- Include source paths by logical name and environment variables (`DATASET_ROOT`,
  `PROJECT_HOME`), not machine-specific absolute paths in committed code.
- Record software versions through `uv.lock`.

## Models and Metrics

- Implement simple baselines before complex models.
- Evaluate ranking by patient/stay and condition, not only row-level
  classification.
- Report precision, recall, hit rate, NDCG, and MRR at clinically motivated K
  values.
- Add calibration, safety, coverage, subgroup, and external-validation metrics
  when the data foundation supports them.
- Distinguish model score from calibrated probability.

## Tests

- Use synthetic, non-identifying fixtures.
- Test schema validation, key integrity, temporal cutoffs, patient separation,
  candidate generation, and leakage exclusions.
- Add a regression test for every fixed behavioral bug when feasible.

## Documentation

- Use ASCII unless a document requires names or terms with established
  non-ASCII spelling.
- Keep status language precise: planned, implemented, evaluated, deprecated.
- Link to canonical documents instead of duplicating large sections.
- Do not present illustrative medication examples as clinical guidance.
