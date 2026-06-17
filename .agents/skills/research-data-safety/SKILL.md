---
name: research-data-safety
description: Plan and review any task that accesses clinical datasets, derived patient-level artifacts, cohorts, features, labels, notes, or model outputs.
---

# Research Data Safety

## Before Access

1. Read `SECURITY.md`, `ARCHITECTURE.md`, and the data rules in `AGENTS.md`.
2. Confirm the minimum tables, columns, population, and aggregation needed.
3. Prefer schemas, headers, metadata, and aggregate counts over rows.
4. Keep network access disabled unless the user explicitly authorizes a
   necessary trusted source.

## During Processing

1. Use DuckDB projection and cohort filters for large compressed tables.
2. Avoid printing records, note text, identifiers, or free-text fields.
3. Preserve source and transformation provenance.
4. Write derived artifacts only under ignored local paths.
5. Apply patient-level splitting and temporal cutoffs before fitting any
   learned preprocessing or model.
6. Use synthetic fixtures for debugging and tests.

## Before Reporting

1. Confirm no restricted data appears in Git status or the diff.
2. Report only aggregate, disclosure-conscious results.
3. State cohort, source version, time window, limitations, and leakage controls.
4. Treat observed prescribing as historical behavior, not clinical optimality.
5. Surface uncertainty and source differences.

## Stop Conditions

Stop and report the issue if a task would require redistributing data, exposing
patient-level records, bypassing a data-use agreement, or making an unsupported
clinical recommendation.
