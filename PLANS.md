# Execution Plans

Use an execution plan for multi-file changes, cohort changes, schema changes,
modeling work, migrations, or tasks expected to take more than one focused
iteration.

The active research sequence lives in
`Documentation/DataFoundationRoadmap.md`. Task plans should reference that
roadmap rather than duplicating it.

## Plan Template

```markdown
# Plan: <short title>

## Goal

What observable result should exist when the work is complete?

## Context

- Relevant files and modules
- Current behavior
- Research or engineering motivation

## Constraints

- Data-access limits
- Public schema compatibility
- Dependency limits
- Compute limits
- Clinical and leakage constraints

## Out of Scope

What will deliberately not be changed?

## Assumptions

What is believed to be true, and how will it be verified?

## Milestones

1. <bounded milestone>
2. <bounded milestone>
3. <bounded milestone>

## Impacted Files

- `path/to/file`

## Risks

- Data leakage
- Join or schema risk
- Performance risk
- Reproducibility risk
- Rollback risk

## Verification

- Commands
- Data invariants
- Expected artifacts
- Manual review

## Rollback

How can the change be reverted without losing data or provenance?

## Results

- Commands run
- Outcomes
- Deviations from the plan
```

## Planning Rules

- Verify the current tree before planning from historical documentation.
- Split data foundation, modeling, and explanation work into separate
  milestones.
- Include a data-safety and leakage review.
- Define acceptance gates with measurable checks.
- Update the results section as milestones complete.
