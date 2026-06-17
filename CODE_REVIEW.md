# Code Review

Review findings should be ordered by severity and cite the affected file and
line. Focus on correctness and risk before style.

## Data Security

- Could raw or row-level clinical data enter Git, logs, prompts, screenshots,
  fixtures, reports, or model artifacts?
- Are secrets, credentials, access tokens, or local paths exposed?
- Are PhysioNet access and redistribution restrictions preserved?

## Cohort and Schema Correctness

- Is the unit of analysis explicit and stable?
- Are source identifiers joined at the correct level?
- Can joins multiply rows or lose patients unexpectedly?
- Are nulls, duplicates, units, timestamps, and source conventions handled?
- Are source-specific differences hidden by premature harmonization?

## Leakage and Evaluation

- Does any feature occur after the medication decision?
- Can the same patient appear across splits?
- Were candidate catalogs, vocabularies, preprocessing, and graph edges fitted
  using validation or test data?
- Are outcomes, medication history, popularity, or labels leaking into inputs?
- Are metrics grouped at the correct patient/stay-condition level?

## Model and Research Claims

- Is a complex method compared against meaningful baselines?
- Are scores described accurately?
- Are uncertainty, calibration, safety, coverage, and external validation
  considered?
- Does the conclusion exceed the evidence?
- Are illustrative examples clearly marked?

## Reproducibility

- Are seeds, cohort parameters, windows, mappings, feature lists, and versions
  recorded?
- Can generated artifacts be traced to code and configuration?
- Are absolute machine-specific paths avoided?

## Performance

- Are large files projected and filtered before materialization?
- Is pandas being asked to load an unbounded multi-gigabyte table?
- Are repeated scans, Cartesian joins, or dense graph operations justified?

## Tests

- Do tests use synthetic data?
- Are edge cases and regressions covered?
- Are temporal boundaries, split integrity, and mapping coverage tested?
- Are pre-existing failures distinguished from regressions?

## Documentation

- Do `README.md`, architecture, roadmap, commands, and changelog match the tree?
- Are implemented and planned capabilities clearly separated?

## Review Result

A review should end with:

- blocking findings;
- non-blocking risks;
- open assumptions;
- checks run and gaps; and
- a concise recommendation.
