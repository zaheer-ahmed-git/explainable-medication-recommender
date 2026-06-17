# Explainable Conversational Medication Recommender

Research foundation for a clinician-facing medication recommendation system
that combines conversational clinical understanding, hybrid
Transformer-GNN ranking, and grounded explanations.

The central research question is:

> How can a medical conversational recommender generate medication rankings
> that are accurate, faithful, grounded, and clinically understandable?

This repository is research software. It is not a medical device, does not
prescribe medication, and must not replace professional clinical judgment.

## Current Status

The repository is currently in the data-foundation and architecture stage.

- The research objective and target architecture are documented.
- Licensed MIMIC-IV v3.1, MIMIC-IV-Note v2.2, and eICU-CRD v2.0 data are
  available locally under the ignored `Dataset/` directory.
- A previous synthetic preprocessing and ranking prototype is retained in the
  ignored `DepreciatedCode/` directory for reference.
- Python dependencies for analysis, DuckDB processing, baseline modeling, and
  testing are declared in `pyproject.toml`.
- The active `pipeline/`, `tests/`, and notebook implementation described in
  the roadmap has not yet been rebuilt in the current working tree.

Do not interpret the poster's illustrative medication table or planned system
diagram as a clinically validated implementation.

## Target System

The proposed system has four cooperating layers:

1. **Conversational understanding:** an LLM extracts a structured patient
   profile from clinical dialogue, notes, and structured EHR fields.
2. **Hybrid recommendation:** a Transformer models patient-context and temporal
   interactions while a heterogeneous GNN models patient, diagnosis,
   medication, laboratory, and knowledge relations.
3. **Grounded explainability:** local attribution, graph evidence, clinical
   rules, uncertainty, and provenance are assembled before an LLM verbalizes
   the evidence.
4. **Clinician review:** ranked medication candidates are presented with
   rationale, warnings, uncertainty, and an audit trail.

Recommendation generation and explanation generation remain separate so that
the explanation is tied to actual model evidence instead of a plausible
free-form narrative.

See [ARCHITECTURE.md](ARCHITECTURE.md) for boundaries and
[Documentation/DataFoundationRoadmap.md](Documentation/DataFoundationRoadmap.md)
for the implementation sequence.

## Data Strategy

The project uses:

- **MIMIC-IV v3.1** for deep single-center hospital and ICU development data.
- **MIMIC-IV-Note v2.2** for de-identified discharge and radiology text.
- **eICU-CRD v2.0** for multi-center ICU data and external validation.

The preferred evaluation design is development on MIMIC-IV and external
validation on eICU after careful schema and vocabulary harmonization. Pooled
training is optional and must not be attempted until concept coverage, units,
cohort definitions, and label semantics are demonstrably compatible.

Raw PhysioNet data is licensed and must never be committed or redistributed.

## Planned Data Product

The main modeling artifact is a candidate-ranking table with one row per:

```text
patient/stay + condition + candidate medication
```

Each row should include a prescription label, pre-decision patient context,
source and cohort provenance, a deterministic patient-level split, and explicit
temporal boundaries. Observed prescribing is a historical label, not proof that
a medication is clinically optimal.

## Repository Guide

- `Documentation/ResearchDetail.md`: current research framing and contribution.
- `Documentation/DataFoundationRoadmap.md`: phased implementation plan.
- `Documentation/PosterPresentationGuide.md`: poster explanation and Q&A.
- `Documentation/SimilarPapers.md`: related-work notes.
- `FinalPosterCDS.pdf`: research poster.
- `ARCHITECTURE.md`: system and data architecture.
- `WORKFLOWS.md`: repeatable development and research workflows.
- `TESTING.md`: verification strategy.
- `CODE_REVIEW.md`: review checklist.
- `AGENTS.md`: durable instructions for coding agents.
- `.agents/skills/`: reusable repository workflows.

Local-only directories:

- `Dataset/`: licensed source data and generated data artifacts.
- `DepreciatedCode/`: synthetic prototype and historical artifacts.

## Environment

This project uses Python 3.13 and `uv` exclusively.

```powershell
uv sync
uv run ruff check .
```

When active tests exist:

```powershell
uv run pytest
```

Do not use `pip`, Poetry, Conda, global Python, or system site-packages.

## Working Principles

- Plan before risky or multi-file changes.
- Use bounded DuckDB queries or chunked readers for large compressed tables.
- Split by patient and apply temporal cutoffs before modeling.
- Keep source-specific extraction separate from harmonization.
- Log cohort, feature, label, split, and model provenance.
- Use synthetic fixtures in tests.
- Review the diff, verification results, data-safety implications, and research
  claims before merging.

## Research References

The project direction and related-work notes are maintained in
`Documentation/ResearchDetail.md` and `Documentation/SimilarPapers.md`.
Dataset users must also cite the official MIMIC-IV, MIMIC-IV-Note, and eICU-CRD
publications and comply with their PhysioNet data-use agreements.

## Security and Contributions

Read [SECURITY.md](SECURITY.md) before handling clinical data or reporting a
vulnerability. Contribution expectations are in
[CONTRIBUTING.md](CONTRIBUTING.md).

No open-source license has been selected for this repository. Unless a license
is added by the maintainers, no permission to reuse or redistribute the code is
granted.
