# Condition Normalization

## Purpose

This note documents the semantic condition normalization layer added to
`pipeline/harmonize.py`. It records the frozen condition contract, the staged
mapping tiers, the schema, and the acceptance gates. It resolves the
`ARCHITECTURE.md` open decision "Shared condition vocabulary and roll-up level"
for the data-foundation stage.

## Behavior Before This Layer

Condition harmonization was **schema-level only**, not semantic-level:

- MIMIC `diagnoses_icd` rows preserved `icd_code` / `icd_version`, set
  `condition_system` to `ICD9CM` / `ICD10CM` / `ICD`, and emitted a
  source-native `condition_token` of the form `icd{version}:{code}` (for
  example `icd10:a419`) with `mapping_status = source_native_code`.
- eICU `diagnosis` rows emitted `icd9:{code}` when `icd9code` was present,
  otherwise a slugified `diagnosisstring` token, with `mapping_status` of
  `source_native_code`, `source_native_text`, or `unmapped_source_concept`.
- There was **no shared cross-source vocabulary, no ICD-9→ICD-10 bridge, no
  CCS/CCSR roll-up, no ICD chapter grouping, and no eICU text→concept mapping.**

This blocked Milestone 6 condition-specific labels
(`patient_condition_medication`), sepsis cohort finalization, and fair
MIMIC↔eICU overlap analysis, because `icd10:*` tokens and slugified eICU strings
are not comparable.

Non-condition domains (labs, vitals, allergies, interventions, temporal events)
were already schema-harmonized with `normalized_*_token` source-native columns.
Conditions now follow the same "keep source-native, add normalized" pattern.

## Frozen Condition Contract (v1)

- **Roll-up level:** CCSR (ICD-10-CM) and CCS (ICD-9-CM) are the preferred
  shared roll-ups. ICD chapter (from a chapters file) and a structural ICD
  **category** (first three characters of the code) are fallbacks. eICU
  text-only rows use a conservative curated exact-match dictionary only.
- **Canonical modeling token:** `normalized_condition_token`. Its granularity is
  recorded in `condition_rollup_level`
  (`ccsr` | `ccs` | `chapter` | `category` | `text_mapped`).
- **Backward compatibility:** `condition_token` is unchanged (source-native).
  Shared semantics live only in the new normalized columns.
- **Sepsis (first research target):** applied as a **separate** project-group
  layer via `project_condition_groups.csv`, not by overwriting the roll-up.
  Sepsis is only populated from a curated, clinically reviewed mapping file; the
  bootstrap script writes an illustrative template that must be reviewed before
  use. No sepsis definition is fabricated in code.
- **Provenance:** condition rows carry `mapping_version = condition-rollup-v1`
  (`CONDITION_MAPPING_VERSION`) plus the standard cohort/extraction/
  harmonization/timestamp fields. `mapping_source` and `mapping_confidence`
  record which file and match quality produced each roll-up.
- **Precedence (highest first):** `mapped_ccsr` → `mapped_ccs` →
  `mapped_icd_crosswalk` (GEM→CCSR) → `mapped_chapter` → `mapped_text_to_condition`
  → structural `category` (`source_native_code`) → `source_native_text` →
  `unmapped_condition`. Project groups are orthogonal and coexist with any tier.

## Optional / Degrading Mapping (unlike medications)

Medication mapping is a **hard gate**: missing RxNorm/ATC files fail the
harmonization CLI. Condition mapping is a **soft gate**:

- If an authoritative mapping file exists and validates, it is LEFT JOINed and
  used.
- If a file is missing, harmonization still succeeds. Coded rows degrade to a
  structural ICD category roll-up; text rows stay source-native. Missing files
  are recorded in the manifest and coverage reports.
- **No diagnosis row is ever dropped**, and no ICD/CCS/CCSR/GEM/project mapping
  is fabricated.

## Staged Tiers

| Tier | Scope | Status |
|------|-------|--------|
| A | Preserve source-native + add normalized columns; never crash without files | Implemented |
| B | ICD roll-up: CCSR (ICD-10), CCS (ICD-9), ICD chapter, or structural category fallback | Implemented (files optional) |
| C | ICD-9→ICD-10 GEM crosswalk then CCSR, lower confidence | Implemented (files optional) |
| D | eICU `diagnosisstring` exact normalized-string curated dictionary only | Implemented (files optional) |
| E | Project condition groups (sepsis first) from curated file | Implemented (files optional) |
| — | UMLS/SNOMED/ontology embeddings | Deferred |

## Schema (`conditions.parquet`)

Preserved columns (unchanged): `condition_system`, `condition_code`,
`condition_text`, `condition_token`, `mapping_status`, and all `PROVENANCE_SCHEMA`
fields.

Added columns:

| Column | Meaning |
|--------|---------|
| `source_condition_system` / `source_condition_code` / `source_condition_text` / `source_condition_token` | Redundant aliases of the source-native fields |
| `normalized_condition_system` | `CCSR`, `CCS`, `ICD10CM_CHAPTER`, `ICD10CM_CATEGORY`, `TEXT_CONDITION`, … |
| `normalized_condition_code` | Roll-up identifier |
| `normalized_condition_name` | Human-readable roll-up label |
| `normalized_condition_token` | Canonical shared modeling token (`ccsr:…`, `ccs:…`, `icd10cat:…`, curated token) |
| `condition_rollup_level` | `ccsr` / `ccs` / `chapter` / `category` / `text_mapped` |
| `mapping_source` | File/logic used, e.g. `icd10_ccsr.csv`, `structural_icd_category` |
| `mapping_confidence` | `exact` / `authoritative_crosswalk` / `approximate` / `curated_text` / `fallback_native` |
| `project_condition_group` / `project_condition_token` | Curated research group (sepsis first), orthogonal to the roll-up |

`mapping_status` enum: `mapped_ccsr`, `mapped_ccs`, `mapped_icd_crosswalk`,
`mapped_chapter`, `mapped_text_to_condition`, `source_native_code`,
`source_native_text`, `unmapped_condition`.

## Mapping Files (`$DATASET_ROOT/mappings/conditions/`)

All optional; defined in `CONDITION_MAPPING_SPECS` in `pipeline/config.py`.

| File | Required columns | Role |
|------|------------------|------|
| `icd10_ccsr.csv` | `icd_code`, `ccsr_category`, `ccsr_category_description` | MIMIC ICD-10 roll-up |
| `icd9_ccs.csv` | `icd_code`, `ccs_category`, `ccs_category_description` | ICD-9 roll-up (MIMIC + eICU coded) |
| `icd9_to_icd10_gem.csv` | `icd9_code`, `icd10_code`, `approximate_flag` | ICD-9→ICD-10 bridge (then CCSR) |
| `icd_chapters.csv` | `icd_version`, `category_code`, `chapter_code`, `chapter_name` | Broad chapter grouping (direct category lookup) |
| `eicu_diagnosis_text_condition_map.csv` | `diagnosisstring_normalized`, `condition_rollup_token`, `condition_name` | Curated exact eICU text → concept |
| `project_condition_groups.csv` | `match_type`, `match_value`, `project_condition_group`, `project_condition_token` | Sepsis/other research groups (`match_type` in `icd_code`, `icd_prefix`, `text_token`) |

Join keys are normalized inside `pipeline.harmonize` (punctuation stripped,
lowercased) so `A41.9`, `A419`, and `a419` match identically. eICU `icd9code`
may contain multiple comma-separated or mixed ICD-9/ICD-10 codes; only the first
code is used as a join key, and both CCS and CCSR joins are attempted (a mapping
is applied only if the code exists in the authoritative file).

## Commands

```bash
# Optional: inventory distinct diagnosis concepts and write review templates.
uv run python scripts/build_condition_mappings.py

# Optional: also merge the approved A1/B3 sepsis mappings into active local CSVs.
uv run python scripts/build_condition_mappings.py --write-curated-sepsis

# Fetch authoritative CCSR/CCS/GEM sources and derive icd_chapters.csv.
# Requires outbound network; raw downloads cache under an ignored directory.
uv run python scripts/fetch_condition_reference_files.py

# Harmonize (condition normalization runs automatically; degrades gracefully).
uv run python -m pipeline.harmonize
```

### Authoritative reference sources (public domain)

`scripts/fetch_condition_reference_files.py` downloads and transforms:

| Output | Source |
|--------|--------|
| `icd10_ccsr.csv` | AHRQ HCUP CCSR for ICD-10-CM (`DXCCSR-v2026-1.zip`), default inpatient category |
| `icd9_ccs.csv` | AHRQ HCUP single-level CCS for ICD-9-CM (`Single_Level_CCS_2015.zip`, `$dxref`) |
| `icd9_to_icd10_gem.csv` | CDC/NCHS 2018 ICD-9→ICD-10 GEM (`2018_I9gem.txt`); no-map rows dropped, approximate flag preserved |
| `icd_chapters.csv` | Derived from published ICD-9/ICD-10-CM chapter ranges, grounded in the CCSR/CCS code lists |

The fetch script does **not** produce `eicu_diagnosis_text_condition_map.csv` or
`project_condition_groups.csv`; those require clinical curation (see below). It
never fabricates mappings and reads no patient data.

## Reports (aggregate only)

- `reports/harmonization_manifest.json` — `condition_mapping_resources`
  (found/missing/ready) and `condition_mapping_optional: true`.
- `reports/harmonization_coverage.json` — `condition_rollup_coverage`
  (source × roll-up level × mapping status).
- `reports/condition_normalization_coverage.json` — per-source summary with
  roll-up and project-group coverage percentages and acceptance-gate targets.
- `reports/eicu_diagnosis_text_mapping_review.csv` — concept-level distinct
  eICU diagnosis strings, counts, and applied mapping (no patient rows).
- `reports/unmapped_concepts.json` — includes `unmapped_condition`.
- `reports/condition_mapping_build_report.json` — bootstrap template inventory.

## Acceptance Gates (initial; tune after first real run)

| Gate | Target |
|------|--------|
| MIMIC roll-up coverage (`normalized_condition_token` non-null) | ≥ 95% |
| eICU roll-up coverage | ≥ 85% |
| Sepsis cohort reproducibility | Stable stay count per mapping version |
| Cross-source overlap | Top roll-ups present in both sources with documented prevalence |
| Unmapped reporting | Aggregate only; no patient-level samples |

Structural ICD category satisfies coverage trivially for coded rows; review
`ccs_ccsr_percent` for authoritative roll-up depth and cross-source
comparability **before** enabling pooled MIMIC+eICU training.

## Remaining Work

- **Done:** authoritative CCSR (ICD-10-CM), CCS (ICD-9-CM), ICD-9→ICD-10 GEM,
  and derived ICD chapter files are produced by
  `scripts/fetch_condition_reference_files.py` and validated against
  `CONDITION_MAPPING_SPECS` (CCSR ≈75.7k codes / 496 categories, CCS ≈15.1k
  codes / 283 categories, GEM ≈24.4k crosswalk rows, ≈2.96k chapter rows).
- Re-run `uv run python -m pipeline.harmonize` to rebuild `conditions.parquet`
  with populated CCSR/CCS/GEM/chapter roll-ups, then review
  `reports/condition_normalization_coverage.json` against the gates above.
- Active A1/B3 sepsis mapping support is implemented in
  `scripts/build_condition_mappings.py --write-curated-sepsis`: it merges the
  approved exact ICD codes, `A40`/`A41` ICD prefixes, and discovered eICU sepsis
  text tokens into local active mapping CSVs while preserving existing curator
  rows. Re-run harmonization after writing those local files.
- Broader non-sepsis eICU diagnosis-text mapping still requires clinical
  curation before pooled MIMIC/eICU training or external performance claims.
- Build the cross-source condition overlap notebook after real coverage exists.
