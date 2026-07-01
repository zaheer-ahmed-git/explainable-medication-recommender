# Project Instructions

## Execution Environment

Development runs on **ULCO Calculco**. Git tracks shared code only;
machine-specific paths live in **gitignored** files (see
`Documentation/Environment.md`).

### Calculco layout

Code lives in home storage; licensed data lives on protected NFS. Export path
variables before pipeline work:

```bash
export PROJECT_HOME=$HOME/ResearchModule
export DATASET_ROOT=$DATA_PROTECTED/Dataset   # or set DATA_PROTECTED parent
export WORK_SCRATCH=/workdir/<lab>/<username>   # for OAR job I/O
```

Account-specific server details are in gitignored
`Documentation/CalculcoSetup.local.md` (template:
`Documentation/CalculcoSetup.example.md`). Platform notes:
`Documentation/CalculcoSetup.md`.

### Shared rules

- Resolve paths through `pipeline/config.py` and environment variables.
- Licensed clinical data is never committed; aggregate reports only in `reports/`.
- Run heavy extraction via OAR on Calculco (`scripts/calculco/`); lightweight
  checks (`uv run pytest`, `uv run ruff check .`) on the login node.
- Do not run destructive commands or delete raw data unless explicitly requested.
- Ask when paths or credentials are missing instead of guessing.

## Repository Overview

- This is a research repository for an explainable conversational medication
  recommender for clinician-facing decision support.
- The target system separates clinical information extraction, medication
  ranking, grounded explanation, and clinician review.
- The planned recommender combines Transformer patient-context modeling with
  heterogeneous GNN relation modeling.
- The active repository is currently in the data-foundation and design stage.
  Do not claim that the production pipeline or Transformer-GNN model exists
  unless the corresponding code and tests are present.
- `DepreciatedCode/` contains the former synthetic-data prototype. It is
  reference material, not the active architecture.

## Canonical References

Read only the documents needed for the task:

- `README.md`: project entry point and current status.
- `ARCHITECTURE.md`: system boundaries and target architecture.
- `Documentation/DataFoundationRoadmap.md`: implementation sequence and gates.
- `Documentation/ResearchDetail.md`: research problem and contribution.
- `WORKFLOWS.md`: repeatable engineering and research procedures.
- `Documentation/Environment.md`: runtime detection and gitignored local files.
- `TESTING.md`: verification strategy.
- `CODE_REVIEW.md`: review priorities.
- `AGENT-MEMORY.md`: stable repository facts and known pitfalls.

Treat `Documentation/OldResearchDetail.md` as historical context. Do not use it
as the source of current project status.

## Python Environment

- Use `uv` exclusively.
- Never use `pip`, `pip3`, `python -m pip`, `poetry`, or `conda`.
- Run commands from the project root unless a workflow says otherwise.
- Sync dependencies with `uv sync`.
- Add runtime dependencies with `uv add <package>`.
- Add development dependencies with `uv add --dev <package>`.
- Remove dependencies with `uv remove <package>`.
- Before adding a package, explain why it is needed.

## Commands

- Run a script: `uv run <script>.py`
- Run a module: `uv run python -m <module>`
- Run tests: `uv run pytest`
- Run lint: `uv run ruff check .`
- Format: `uv run ruff format .`

Prefer the smallest relevant verification command. If no active tests cover a
change, say so explicitly and add focused tests when behavior is introduced.

## Path Configuration

Runtime paths are resolved in `pipeline/config.py`:

| Variable | Purpose | Default |
|----------|---------|---------|
| `PROJECT_HOME` or `RESEARCHMODULE_ROOT` | Repository root for code and default reports | directory containing `pipeline/` |
| `DATASET_ROOT` | Licensed clinical data root | `Dataset/` under project root |
| `DATA_PROTECTED` | Protected storage parent used when `DATASET_ROOT` is unset | none |
| `REPORTS_ROOT` | Aggregate report output directory | `reports/` under project root |
| `WORK_SCRATCH` | Ephemeral job I/O on Calculco (used by OAR scripts, not `config.py`) | none |

On Calculco, export `PROJECT_HOME`, `DATASET_ROOT`, and optionally
`WORK_SCRATCH` in `~/.bashrc` so agents and pipeline CLIs find protected
clinical data while code stays in home storage. See `Documentation/CalculcoSetup.md`
and `.env.example`. Do not hard-code user-specific absolute paths in code;
prefer these variables and `pathlib.Path` resolution in `pipeline/config.py`.

## Verification Tiers

Choose the smallest check that answers the question:

| Tier | When | Examples |
|------|------|----------|
| Lightweight | Config, docs, unit tests, lint | `uv run pytest tests/test_config.py`, `uv run ruff check .` |
| Pipeline (login node) | Small bounded CLIs on aggregate fixtures | `uv run pytest`, metadata-only inventory |
| HPC (OAR) | Cohort-filtered extraction, large-table scans | `oarsub -S scripts/calculco/extract_mimic.sh` |

Do not submit GPU jobs, full-dataset extraction, or long walltime OAR jobs unless
the user explicitly requests them. Tests that need protected data should document
required environment variables and skip when `DATASET_ROOT` is unset or paths are
missing.

## Data Safety

- `Dataset/` contains licensed, de-identified clinical data from MIMIC-IV,
  MIMIC-IV-Note, and eICU-CRD. Never commit, redistribute, upload, or paste raw
  rows from these datasets.
- Do not expose patient-level records in logs, fixtures, screenshots, prompts,
  reports, or error messages.
- Inspect schemas, counts, and aggregate statistics before inspecting rows.
- Use DuckDB, projection, filters, limits, and chunked reads for large files.
  Do not load multi-gigabyte CSV files into pandas.
- Tests must use synthetic or heavily minimized non-identifying fixtures.
- Keep dataset licenses and PhysioNet access conditions intact.

## Clinical Research Rules

- This project supports research and clinician review; it does not prescribe
  treatment or replace clinical judgment.
- Never present illustrative poster examples as validated clinical advice.
- Distinguish observed prescriptions from optimal or recommended treatment.
- Enforce patient-level splits and temporal cutoffs before reporting model
  performance.
- Treat medication history, post-treatment outcomes, candidate popularity, and
  future events as leakage risks unless an experiment explicitly justifies
  them.
- Record source dataset, cohort definition, extraction version, random seed,
  feature window, label window, and model version for reproducible results.

## Engineering Rules

- Use Python 3.13 conventions already declared in `pyproject.toml`.
- Prefer typed, testable functions and `pathlib.Path`.
- Keep source-specific extraction separate from cross-source harmonization.
- Preserve source identifiers and provenance through derived artifacts.
- Write generated data and model artifacts only under ignored output paths.
- Do not edit ignored legacy code or raw datasets unless the user explicitly
  asks for that surface.
- Do not silently change public schemas, cohort definitions, label semantics,
  split logic, or medication normalization rules.
- Keep changes scoped. Do not revive deleted modules merely because historical
  documentation mentions them.

## Documentation Rules

- Update `README.md` when current capabilities or entry points change.
- Update `ARCHITECTURE.md` for boundary or data-flow changes.
- Update `Documentation/DataFoundationRoadmap.md` when milestone status changes.
- Update `TESTING.md`, `WORKFLOWS.md`, and `CHANGELOG.md` when commands or
  operating procedures change.
- Mark planned, implemented, evaluated, and deprecated work clearly.

## Review Guidelines

Prioritize:

1. Patient-data exposure or licensing violations.
2. Temporal leakage, patient overlap, target leakage, or invalid evaluation.
3. Incorrect cohort joins, identifiers, units, timestamps, or medication maps.
4. Non-reproducible experiments and missing provenance.
5. Unbounded reads of large clinical tables.
6. Missing tests, stale documentation, and misleading research claims.

## Definition of Done

- The requested behavior or document is complete and internally consistent.
- Relevant tests and lint checks pass, or unavailable checks are reported.
- Data-safety and leakage implications were reviewed.
- Generated artifacts and raw data remain untracked.
- Documentation and changelog entries match the actual repository state.
- The final response lists the important files changed and verification run.
