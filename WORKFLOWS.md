# Workflows

## Significant Change

1. Read the closest instructions and canonical architecture documents.
2. State the goal, constraints, assumptions, and definition of done.
3. Create a milestone plan using `PLANS.md`.
4. Implement one milestone.
5. Run focused verification.
6. Review the diff using `CODE_REVIEW.md`.
7. Update affected docs and `CHANGELOG.md`.

## Build or Change a Cohort

1. Define source population, inclusion criteria, exclusion criteria, unit of
   analysis, index time, and deduplication rule.
2. Implement source-specific logic before unified logic.
3. Materialize only identifiers and necessary cohort fields first.
4. Produce a manifest with source counts and filter attrition.
5. Test key uniqueness, referential integrity, patient overlap, and boundary
   conditions.
6. Compare counts with documented dataset scales without treating approximate
   public counts as exact acceptance criteria.
7. Update `Documentation/DataFoundationRoadmap.md`.

Current broad adult cohort command:

```bash
uv run python -m pipeline.cohort
```

The generated cohort Parquet files are local ignored artifacts under
`Dataset/processed/cohorts/`; only aggregate manifest counts should be reported
outside the local environment.

## Profile a Large Table

1. Inspect the header and metadata without reading rows into logs.
2. Run a bounded schema profile.
3. Select required columns.
4. Use DuckDB or chunked streaming for complete scans.
5. Save aggregate results under `reports/`.
6. Review results for identifiers, units, nulls, duplicates, impossible values,
   and source-specific conventions.

Current aggregate profile command:

```bash
uv run python -m pipeline.profile_tables
```

The generated `reports/quality_profile.json` is an ignored local artifact. It
must contain aggregate counts and column metrics only, never row samples or note
text.

## Build EDA Brief

1. Confirm `reports/source_inventory.json`, `reports/cohort_manifest.json`, and
   `reports/quality_profile.json` exist.
2. Run `uv run python -m pipeline.eda_summary`.
3. Review `reports/eda_dataset_understanding.md` for stakeholder-facing
   messages, quality blockers, and next actions.
4. Review `reports/figures/` for aggregate charts.
5. Do not add row examples, note text, identifiers, or clinical
   recommendations to EDA outputs.

## Check Source Integrity

1. Run `uv run python -m pipeline.source_integrity` for profiling-blocked files.
2. Review `reports/source_integrity_failed_tables.json`.
3. Run `uv run python -m pipeline.source_integrity --all-manifest-files` for a
   complete checksum/gzip audit of all files listed in configured
   `SHA256SUMS.txt` manifests.
4. Use `--source mimiciv`, `--source eicu_crd`, or `--source mimiciv_note` to
   split the full audit into smaller source-specific runs.
5. Treat checksum mismatches, truly missing local files, or gzip failures as
   source-integrity blockers. If a configured uncompressed local file exists
   for a manifest `.csv.gz` entry, document it as a source-layout
   reconciliation before using the table downstream.
6. Re-transfer or re-download affected files before extraction or feature
   engineering.
7. Consider CSV parser fallbacks only after checksum and gzip validation pass.

## Build Source Inventory

1. Inspect only file metadata and CSV headers.
2. Run `uv run python -m pipeline.source_inventory`.
3. Confirm `reports/source_inventory.json` remains ignored.
4. Check missing expected files and checksum-file presence.
5. Do not print or paste clinical rows, note text, identifiers, or free-text
   values from the source files.

## Add an Extraction Module

1. Write the output schema and provenance fields first.
2. Validate required source columns.
3. Filter to cohort identifiers at the source query.
4. Normalize names only after preserving original values.
5. Add synthetic contract tests.
6. Record row counts and mapping coverage.

Current extraction commands:

```bash
uv run python -m pipeline.mimic_extract
uv run python -m pipeline.eicu_extract
```

Extraction commands depend on `reports/quality_profile.json`,
`reports/source_integrity_failed_tables.json`, and
`Dataset/processed/cohorts/cohort_stays.parquet`. Blocked tables are skipped
with aggregate manifest entries rather than forced through parser or integrity
failures.

## Run Milestone 5 Harmonization

1. Confirm source-specific extraction manifests are available and aggregate-only.
2. Place reviewed medication mapping files under
   `Dataset/mappings/medications/`:
   `mimic_ndc_rxnorm_atc.csv` and `eicu_drug_rxnorm_atc.csv`. These are a hard
   gate; harmonization fails without them.
3. Optionally add condition mapping files under `Dataset/mappings/conditions/`
   (`icd10_ccsr.csv`, `icd9_ccs.csv`, `icd9_to_icd10_gem.csv`,
   `icd_chapters.csv`, `eicu_diagnosis_text_condition_map.csv`,
   `project_condition_groups.csv`). These are optional; missing files degrade to
   structural ICD categories and source-native tokens without failing.
   - Run `uv run python scripts/fetch_condition_reference_files.py` (needs
     network) to download authoritative AHRQ CCSR/CCS and CDC GEM sources and
     write `icd10_ccsr.csv`, `icd9_ccs.csv`, `icd9_to_icd10_gem.csv`, and a
     derived `icd_chapters.csv`.
   - Run `uv run python scripts/build_condition_mappings.py` to inventory
     distinct diagnosis concepts and emit review-ready templates for the
     curation-only files (eICU text map, project condition groups).
4. Run `uv run python -m pipeline.harmonize`.
5. Review `reports/harmonization_manifest.json`,
   `reports/harmonization_coverage.json`, `reports/unmapped_concepts.json`,
   `reports/condition_normalization_coverage.json`, and
   `reports/eicu_diagnosis_text_mapping_review.csv`.
6. Confirm the manifest lists `cohort_stays`, `demographics`, `conditions`,
   `medications`, `labs`, `vitals`, `allergies`, `interventions`, and
   `temporal_events`, and inspect `condition_mapping_resources`.
7. Do not enable pooled training from harmonized artifacts until coverage
   thresholds (see `Documentation/ConditionNormalization.md`) and
   source-specific semantic differences are reviewed.

## Build a Training Table

1. Confirm Milestone 5 harmonization artifacts and aggregate coverage reports
   have been reviewed.
2. Freeze cohort, index time, feature window, label window, and patient split.
3. Run `uv run python -m pipeline.features` to build decision times,
   patient-stay features, and event sequences under
   `Dataset/processed/features/`.
4. Run `uv run python -m pipeline.build_training_table` to build the split
   manifest, train-only candidate catalog, and patient-condition-medication
   table under `Dataset/processed/training/`.
5. Build candidates from the training partition only.
6. Create observed-positive and sampled/implicit-negative labels with explicit
   caveats.
7. Exclude future and leakage-prone features by default.
8. Validate one patient belongs to one split.
9. Review `reports/milestone6_feature_manifest.json` and
   `reports/training_table_manifest.json` for censoring, temporal exclusions,
   candidate coverage, out-of-catalog positives, and aggregate-only contents.

## Run an Experiment

1. Name the hypothesis and baseline.
2. Freeze data and configuration identifiers.
3. Run on validation data while developing.
4. Use test or external-validation data only for the final locked evaluation.
5. Save configuration, metrics, seed, feature list, and model version.
6. Report failures, uncertainty, subgroup limitations, and negative results.
7. Do not promote a poster illustration or exploratory result to a validated
   clinical claim.

## Fix a Bug

1. Reproduce the behavior on synthetic or minimized data.
2. Add a regression test that fails.
3. Apply the smallest correction.
4. Rerun the reproduction and targeted tests.
5. Check whether the bug affected prior artifacts or reported metrics.
6. Update documentation or changelog when behavior changed.

## Documentation-Only Change

1. Verify claims against the current tree.
2. Keep one canonical source for each fact.
3. Fix links and commands.
4. Distinguish current state from roadmap.
5. Run Markdown/link checks available in the repository and inspect the diff.

## Dependency Change

1. Explain the concrete need.
2. Prefer the standard library or an existing dependency when reasonable.
3. Use `uv add`, `uv add --dev`, or `uv remove`.
4. Inspect `pyproject.toml` and `uv.lock`.
5. Run focused tests and lint.
6. Note security, licensing, and reproducibility implications.

## Calculco HPC (ULCO)

Use the cluster for heavy extraction, modeling, and long-running jobs.

1. Confirm `PROJECT_HOME` and `DATASET_ROOT` are exported.
2. Read account-specific paths in gitignored `Documentation/CalculcoSetup.local.md`
   (create from `Documentation/CalculcoSetup.example.md`). Generic platform notes:
   `Documentation/CalculcoSetup.md`.
3. Export runtime paths via `.env.calculco` or `scripts/calculco/common.local.sh`
   (both gitignored) before pipeline work.
4. Use `$WORK_SCRATCH/runs` for temporary job I/O; copy results back to protected
   storage after jobs complete.
5. Transfer large files with `rsync` via `pcsdata.univ-littoral.fr`.
6. Submit compute with OAR (`oarsub`), not on the login node.
7. Load software with `module load` or `uv` as documented on the Calculco website.

### Run Source Extraction on Calculco

Repository OAR scripts live under `scripts/calculco/`. `common.sh` loads
`.env.calculco` / `common.local.sh`, requires `DATASET_ROOT`, and sets scratch
for `TMPDIR` and `UV_CACHE_DIR` when `WORK_SCRATCH` is set. Outputs go to
`$DATASET_ROOT/processed/extracts/`; manifests to `$PROJECT_HOME/reports/`.

Preflight on the login node:

```bash
test -f "$DATASET_ROOT/processed/cohorts/cohort_stays.parquet"
test -f "$PROJECT_HOME/reports/quality_profile.json"
test -f "$PROJECT_HOME/reports/source_integrity_failed_tables.json"
cd "$PROJECT_HOME" && uv run python -V
```

Submit (pass log paths at submit time — not hard-coded in Git):

```bash
chmod +x scripts/calculco/*.sh
oarsub -O "$PROJECT_HOME/scripts/calculco/logs/rm_extract_%jobid%.out" \
       -E "$PROJECT_HOME/scripts/calculco/logs/rm_extract_%jobid%.err" \
       -S "$PROJECT_HOME/scripts/calculco/extract_mimic_eicu.sh"

# or run MIMIC and eICU in parallel:
oarsub -O "$PROJECT_HOME/scripts/calculco/logs/rm_mimic_%jobid%.out" \
       -E "$PROJECT_HOME/scripts/calculco/logs/rm_mimic_%jobid%.err" \
       -S "$PROJECT_HOME/scripts/calculco/extract_mimic.sh"
oarsub -O "$PROJECT_HOME/scripts/calculco/logs/rm_eicu_%jobid%.out" \
       -E "$PROJECT_HOME/scripts/calculco/logs/rm_eicu_%jobid%.err" \
       -S "$PROJECT_HOME/scripts/calculco/extract_eicu.sh"
```

Monitor with `oarstatmon.py` (overview) or `oarstat`. During the 2026 platform
migration, see `Documentation/CalculcoSetup.md` for `ritchie` login and node
property changes. Review only aggregate manifests after completion:

### Run Harmonization on Calculco

Outputs go to `$DATASET_ROOT/processed/harmonized/`; reports to
`$PROJECT_HOME/reports/`. Medication mapping files are a **hard gate**;
condition roll-up files under `$DATASET_ROOT/mappings/conditions/` are optional
(fetch with `uv run python scripts/fetch_condition_reference_files.py` on the
login node before submit).

Preflight on the login node:

```bash
test -f "$DATASET_ROOT/processed/cohorts/cohort_stays.parquet"
test -f "$DATASET_ROOT/processed/extracts/mimiciv/diagnoses_icd.parquet"
test -f "$DATASET_ROOT/processed/extracts/eicu_crd/diagnosis.parquet"
test -f "$DATASET_ROOT/mappings/medications/mimic_ndc_rxnorm_atc.csv"
test -f "$DATASET_ROOT/mappings/medications/eicu_drug_rxnorm_atc.csv"
cd "$PROJECT_HOME" && uv run python -V
```

Submit:

```bash
oarsub -O "$PROJECT_HOME/scripts/calculco/logs/rm_harmonize_%jobid%.out" \
       -E "$PROJECT_HOME/scripts/calculco/logs/rm_harmonize_%jobid%.err" \
       -S "$PROJECT_HOME/scripts/calculco/harmonize.sh"
```

If submission fails with **"There are not enough resources"**, run
`oarstatmon.py`. On the legacy `calculco` front-end during the 2026 migration,
most CPU nodes may be **Dead** while `gpudevice='-1'` excludes the only **Alive**
GPU nodes — see [OAR troubleshooting](#oar-troubleshooting-calculco-migration)
below. `harmonize.sh` omits `gpudevice='-1'` by default during migration.

Review only aggregate manifests after completion:

- `reports/harmonization_manifest.json`
- `reports/harmonization_coverage.json`
- `reports/condition_normalization_coverage.json`
- `reports/unmapped_concepts.json`

### Re-profile source tables on Calculco

Re-run the **full** aggregate quality profile after correcting local
`chartevents` / `inputevents` source files so their `scan_failed` entries
refresh and the extraction gates can materialize `inputevents`. A full run is
required because `pipeline.profile_tables` rewrites the whole
`reports/quality_profile.json`; a `--table` subset would drop the other tables'
gate entries.

```bash
oarsub -O "$PROJECT_HOME/scripts/calculco/logs/rm_profile_%jobid%.out" \
       -E "$PROJECT_HOME/scripts/calculco/logs/rm_profile_%jobid%.err" \
       -S "$PROJECT_HOME/scripts/calculco/profile_tables.sh"
```

Confirm `mimic_chartevents` and `mimic_inputevents` are `completed` in
`reports/quality_profile.json`, then re-run the MIMIC extractor so
`inputevents` materializes past its gate.

### Run Milestone 6 feature and training builds on Calculco

Run only after Milestone 5 harmonization and its aggregate coverage reports are
reviewed. Outputs go to `$DATASET_ROOT/processed/features/` and
`$DATASET_ROOT/processed/training/`; manifests to `$PROJECT_HOME/reports/`.

Preflight on the login node:

```bash
for t in cohort_stays demographics conditions medications labs vitals \
  allergies interventions temporal_events; do
  test -f "$DATASET_ROOT/processed/harmonized/$t.parquet" || echo "MISSING $t"
done
```

Submit the full Milestone 6 chain (features then training table) in one job:

```bash
oarsub -O "$PROJECT_HOME/scripts/calculco/logs/rm_milestone6_%jobid%.out" \
       -E "$PROJECT_HOME/scripts/calculco/logs/rm_milestone6_%jobid%.err" \
       -S "$PROJECT_HOME/scripts/calculco/milestone6.sh"
```

Or run the stages as separate jobs (`build_training_table.sh` depends on the
feature artifacts from `features.sh`):

```bash
oarsub -O "$PROJECT_HOME/scripts/calculco/logs/rm_features_%jobid%.out" \
       -E "$PROJECT_HOME/scripts/calculco/logs/rm_features_%jobid%.err" \
       -S "$PROJECT_HOME/scripts/calculco/features.sh"
oarsub -O "$PROJECT_HOME/scripts/calculco/logs/rm_training_table_%jobid%.out" \
       -E "$PROJECT_HOME/scripts/calculco/logs/rm_training_table_%jobid%.err" \
       -S "$PROJECT_HOME/scripts/calculco/build_training_table.sh"
```

Review only aggregate manifests after completion:

- `reports/milestone6_feature_manifest.json` (eligibility, splits, temporal
  exclusions)
- `reports/training_table_manifest.json` (split integrity, candidate counts,
  training rows by split, out-of-catalog positives, coverage losses)

### OAR troubleshooting (Calculco migration)

```bash
oarstatmon.py
```

| Symptom | Likely cause | Fix |
|---------|----------------|-----|
| `not enough resources` + type constraints | CPU nodes Dead; `gpudevice='-1'` leaves no nodes | SSH to `ritchie.univ-littoral.fr`, or omit `gpudevice='-1'` in the script |
| Job waits (`W`) a long time | Cluster busy | Try `-t besteffort` or fewer cores |

Probe that scheduling works (then `oardel <jobid>`):

```bash
oarsub -l /nodes=1/core=4,walltime=1:00:00 -t besteffort \
       -O /tmp/oar_probe_%jobid%.out -E /tmp/oar_probe_%jobid%.err "echo ok"
```

Job stdout/stderr are written under `scripts/calculco/logs/` and are gitignored.
Do not paste patient-level extract rows into chat, docs, or version control.
