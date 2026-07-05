"""Source-tagged harmonization entry point for Milestone 5."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import duckdb

from pipeline.config import (
    COHORTS_ROOT,
    COHORT_VERSION,
    CONDITION_MAPPING_SPECS,
    CONDITION_MAPPING_VERSION,
    DUCKDB_MEMORY_LIMIT,
    DUCKDB_TEMP_DIR,
    DUCKDB_THREADS,
    EXTRACTS_ROOT,
    HARMONIZATION_VERSION,
    HARMONIZED_ROOT,
    MAPPING_ROOT,
    MEDICATION_MAPPING_SPECS,
    MEDICATION_MAPPING_VERSION,
    MIMIC_CHARTEVENTS_VITAL_ITEMIDS,
    REPORTS_ROOT,
    MappingFileSpec,
)
from pipeline.extract_utils import (
    configure_duckdb_connection,
    parquet_scan,
    safe_error_message,
    sql_string,
)
from pipeline.io_utils import inspect_header


SCHEMA_VERSION = "harmonization-manifest-v1"
COVERAGE_SCHEMA_VERSION = "harmonization-coverage-v1"
UNMAPPED_SCHEMA_VERSION = "unmapped-concepts-v1"

DEFAULT_COHORT_PATH = COHORTS_ROOT / "cohort_stays.parquet"
DEFAULT_MANIFEST_PATH = REPORTS_ROOT / "harmonization_manifest.json"
DEFAULT_COVERAGE_PATH = REPORTS_ROOT / "harmonization_coverage.json"
DEFAULT_UNMAPPED_PATH = REPORTS_ROOT / "unmapped_concepts.json"
DEFAULT_CONDITION_COVERAGE_PATH = REPORTS_ROOT / "condition_normalization_coverage.json"
DEFAULT_TEXT_REVIEW_PATH = REPORTS_ROOT / "eicu_diagnosis_text_mapping_review.csv"

CONDITION_COVERAGE_SCHEMA_VERSION = "condition-normalization-coverage-v1"

SOURCE_NATIVE_MAPPING_VERSION = "source_native"
NOT_APPLICABLE_VERSION = "not_applicable"
UNKNOWN_EXTRACTION_VERSION = "not_available"

REQUIRED_HARMONIZED_TABLES = (
    "cohort_stays",
    "demographics",
    "conditions",
    "medications",
    "labs",
    "vitals",
    "allergies",
    "interventions",
    "temporal_events",
)
PROVENANCE_COLUMNS = (
    "cohort_version",
    "extraction_version",
    "mapping_version",
    "harmonization_version",
    "generated_at",
)


@dataclass(frozen=True)
class HarmonizationBuildConfig:
    """Configuration for source-tagged harmonized artifacts."""

    cohort_path: Path = DEFAULT_COHORT_PATH
    extracts_root: Path = EXTRACTS_ROOT
    harmonized_root: Path = HARMONIZED_ROOT
    mapping_root: Path = MAPPING_ROOT
    manifest_path: Path = DEFAULT_MANIFEST_PATH
    coverage_path: Path = DEFAULT_COVERAGE_PATH
    unmapped_path: Path = DEFAULT_UNMAPPED_PATH
    condition_coverage_path: Path = DEFAULT_CONDITION_COVERAGE_PATH
    text_review_path: Path = DEFAULT_TEXT_REVIEW_PATH
    harmonization_version: str = HARMONIZATION_VERSION
    cohort_version: str = COHORT_VERSION
    duckdb_temp_directory: Path | None = DUCKDB_TEMP_DIR
    duckdb_memory_limit: str | None = DUCKDB_MEMORY_LIMIT
    duckdb_threads: int | None = DUCKDB_THREADS


def mapping_resource_status(
    spec: MappingFileSpec,
    *,
    mapping_root: Path,
) -> dict[str, Any]:
    """Inspect one expected mapping file without reading source data."""

    path = mapping_root / spec.relative_path
    if not path.exists():
        return {
            "name": spec.name,
            "relative_path": spec.relative_path.as_posix(),
            "version": spec.version,
            "status": "missing",
            "missing_required_columns": list(spec.required_columns),
        }
    header = inspect_header(path) or ()
    missing = [column for column in spec.required_columns if column not in header]
    return {
        "name": spec.name,
        "relative_path": spec.relative_path.as_posix(),
        "version": spec.version,
        "status": "ready" if not missing else "missing_required_columns",
        "missing_required_columns": missing,
    }


def mapping_resources_ready(
    *,
    mapping_root: Path,
    specs: Sequence[MappingFileSpec] = MEDICATION_MAPPING_SPECS,
) -> tuple[bool, list[dict[str, Any]]]:
    """Return whether all required medication mapping files are usable."""

    statuses = [
        mapping_resource_status(spec, mapping_root=mapping_root) for spec in specs
    ]
    return all(status["status"] == "ready" for status in statuses), statuses


def condition_mapping_resource_statuses(
    *,
    mapping_root: Path,
    specs: Sequence[MappingFileSpec] = CONDITION_MAPPING_SPECS,
) -> list[dict[str, Any]]:
    """Inspect optional condition mapping files without gating harmonization.

    Unlike medication mapping, condition semantic mapping is optional: missing
    files degrade to source-native and structural roll-ups rather than failing
    the harmonization CLI.
    """

    return [mapping_resource_status(spec, mapping_root=mapping_root) for spec in specs]


def available_condition_mappings(
    *,
    mapping_root: Path,
    specs: Sequence[MappingFileSpec] = CONDITION_MAPPING_SPECS,
) -> dict[str, Path]:
    """Return {spec name: path} for condition mapping files that are usable."""

    available: dict[str, Path] = {}
    for spec in specs:
        status = mapping_resource_status(spec, mapping_root=mapping_root)
        if status["status"] == "ready":
            available[spec.name] = mapping_root / spec.relative_path
    return available


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a stable JSON artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def source_extract_path(
    config: HarmonizationBuildConfig, source: str, name: str
) -> Path:
    """Return a source extract path under the configured extracts root."""

    return config.extracts_root / source / name


def harmonized_path(config: HarmonizationBuildConfig, name: str) -> Path:
    """Return a harmonized artifact path."""

    return config.harmonized_root / name


def normalized_text_token_sql(expression: str) -> str:
    """Return SQL that turns source text into a coarse stable token."""

    return (
        "NULLIF(REGEXP_REPLACE(LOWER(TRIM(CAST("
        f"{expression} AS VARCHAR))), '[^a-z0-9]+', '_', 'g'), '')"
    )


def condition_code_key_sql(expression: str) -> str:
    """Return SQL that normalizes an ICD code into a compact join key.

    Removes punctuation/whitespace and lowercases so that ``A41.9`` and
    ``a419`` join identically. Returns NULL for blank codes.
    """

    return (
        "NULLIF(LOWER(REGEXP_REPLACE(TRIM(CAST("
        f"{expression} AS VARCHAR)), '[^A-Za-z0-9]+', '', 'g')), '')"
    )


def normalized_unit_sql(expression: str) -> str:
    """Return conservative unit normalization without unit conversion."""

    return (
        "NULLIF(REGEXP_REPLACE(LOWER(TRIM(CAST("
        f"{expression} AS VARCHAR))), '\\s+', ' ', 'g'), '')"
    )


def provenance_sql(
    config: HarmonizationBuildConfig,
    *,
    generated_at: str,
    extraction_version: str,
    mapping_version: str,
) -> str:
    """Return common provenance columns for harmonized artifacts."""

    return f"""
    {sql_string(config.cohort_version)} AS cohort_version,
    {extraction_version} AS extraction_version,
    {mapping_version} AS mapping_version,
    {sql_string(config.harmonization_version)} AS harmonization_version,
    {sql_string(generated_at)} AS generated_at"""


def configure_connection(
    config: HarmonizationBuildConfig, connection: duckdb.DuckDBPyConnection
) -> None:
    """Apply the memory-safe DuckDB settings used by every harmonization scan."""

    configure_duckdb_connection(
        connection,
        temp_directory=config.duckdb_temp_directory,
        memory_limit=config.duckdb_memory_limit,
        threads=config.duckdb_threads,
    )


def copy_query_to_parquet(
    connection: duckdb.DuckDBPyConnection,
    query: str,
    output_path: Path,
) -> int:
    """Materialize a harmonization query and return its row count."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    connection.execute(f"COPY ({query}) TO {sql_string(output_path)} (FORMAT PARQUET)")
    row = connection.execute(
        f"SELECT COUNT(*) FROM {parquet_scan(output_path)}"
    ).fetchone()
    return int(row[0]) if row is not None else 0


def union_queries(queries: Sequence[str]) -> str:
    """Union one or more SELECT queries by name."""

    return "\nUNION ALL BY NAME\n".join(f"SELECT * FROM ({query})" for query in queries)


def empty_select(schema: Sequence[tuple[str, str]]) -> str:
    """Return an empty SELECT with the requested schema."""

    columns = ",\n    ".join(
        f"CAST(NULL AS {column_type}) AS {column_name}"
        for column_name, column_type in schema
    )
    return f"SELECT\n    {columns}\nWHERE FALSE"


BASE_ID_SCHEMA: tuple[tuple[str, str], ...] = (
    ("source", "VARCHAR"),
    ("source_version", "VARCHAR"),
    ("patient_uid", "VARCHAR"),
    ("encounter_uid", "VARCHAR"),
    ("stay_uid", "VARCHAR"),
    ("source_patient_id", "VARCHAR"),
    ("source_encounter_id", "VARCHAR"),
    ("source_stay_id", "VARCHAR"),
)

PROVENANCE_SCHEMA: tuple[tuple[str, str], ...] = (
    ("cohort_version", "VARCHAR"),
    ("extraction_version", "VARCHAR"),
    ("mapping_version", "VARCHAR"),
    ("harmonization_version", "VARCHAR"),
    ("generated_at", "VARCHAR"),
)

DEMOGRAPHICS_SCHEMA = (
    *BASE_ID_SCHEMA,
    ("source_table", "VARCHAR"),
    ("age_years", "DOUBLE"),
    ("age_topcoded", "BOOLEAN"),
    ("sex", "VARCHAR"),
    ("race_or_ethnicity", "VARCHAR"),
    ("hospital_id", "VARCHAR"),
    ("ward_id", "VARCHAR"),
    ("admission_type", "VARCHAR"),
    ("admission_source", "VARCHAR"),
    ("unit_type", "VARCHAR"),
    ("last_unit_type", "VARCHAR"),
    ("stay_type", "VARCHAR"),
    ("stay_sequence", "BIGINT"),
    ("cohort_rule", "VARCHAR"),
    ("mapping_status", "VARCHAR"),
    *PROVENANCE_SCHEMA,
)

CONDITIONS_SCHEMA = (
    *BASE_ID_SCHEMA,
    ("source_table", "VARCHAR"),
    ("source_sequence", "VARCHAR"),
    ("condition_system", "VARCHAR"),
    ("condition_code", "VARCHAR"),
    ("condition_text", "VARCHAR"),
    ("condition_token", "VARCHAR"),
    ("source_condition_system", "VARCHAR"),
    ("source_condition_code", "VARCHAR"),
    ("source_condition_text", "VARCHAR"),
    ("source_condition_token", "VARCHAR"),
    ("normalized_condition_system", "VARCHAR"),
    ("normalized_condition_code", "VARCHAR"),
    ("normalized_condition_name", "VARCHAR"),
    ("normalized_condition_token", "VARCHAR"),
    ("condition_rollup_level", "VARCHAR"),
    ("mapping_source", "VARCHAR"),
    ("mapping_confidence", "VARCHAR"),
    ("project_condition_group", "VARCHAR"),
    ("project_condition_token", "VARCHAR"),
    ("mapping_status", "VARCHAR"),
    *PROVENANCE_SCHEMA,
)

MEDICATIONS_SCHEMA = (
    *BASE_ID_SCHEMA,
    ("source_table", "VARCHAR"),
    ("source_event_id", "VARCHAR"),
    ("event_start_time", "VARCHAR"),
    ("event_end_time", "VARCHAR"),
    ("medication_source_name", "VARCHAR"),
    ("source_code", "VARCHAR"),
    ("source_code_type", "VARCHAR"),
    ("route", "VARCHAR"),
    ("dose_value", "VARCHAR"),
    ("dose_unit", "VARCHAR"),
    ("rxcui", "VARCHAR"),
    ("ingredient_name", "VARCHAR"),
    ("rxnorm_name", "VARCHAR"),
    ("atc_code", "VARCHAR"),
    ("atc_level", "VARCHAR"),
    ("mapping_status", "VARCHAR"),
    *PROVENANCE_SCHEMA,
)

LABS_SCHEMA = (
    *BASE_ID_SCHEMA,
    ("source_table", "VARCHAR"),
    ("source_event_id", "VARCHAR"),
    ("event_time", "VARCHAR"),
    ("event_time_offset", "VARCHAR"),
    ("source_lab_code", "VARCHAR"),
    ("source_lab_name", "VARCHAR"),
    ("normalized_lab_token", "VARCHAR"),
    ("lab_value_numeric", "DOUBLE"),
    ("lab_value_text", "VARCHAR"),
    ("unit", "VARCHAR"),
    ("normalized_unit", "VARCHAR"),
    ("reference_range_lower", "VARCHAR"),
    ("reference_range_upper", "VARCHAR"),
    ("abnormal_flag", "VARCHAR"),
    ("mapping_status", "VARCHAR"),
    *PROVENANCE_SCHEMA,
)

VITALS_SCHEMA = (
    *BASE_ID_SCHEMA,
    ("source_table", "VARCHAR"),
    ("source_event_id", "VARCHAR"),
    ("event_time", "VARCHAR"),
    ("event_time_offset", "VARCHAR"),
    ("source_vital_code", "VARCHAR"),
    ("source_vital_name", "VARCHAR"),
    ("normalized_vital_token", "VARCHAR"),
    ("value_numeric", "DOUBLE"),
    ("unit", "VARCHAR"),
    ("normalized_unit", "VARCHAR"),
    ("mapping_status", "VARCHAR"),
    *PROVENANCE_SCHEMA,
)

ALLERGIES_SCHEMA = (
    *BASE_ID_SCHEMA,
    ("source_table", "VARCHAR"),
    ("source_event_id", "VARCHAR"),
    ("event_time", "VARCHAR"),
    ("event_entered_time", "VARCHAR"),
    ("allergen_source_name", "VARCHAR"),
    ("allergen_source_code", "VARCHAR"),
    ("normalized_allergen_token", "VARCHAR"),
    ("allergy_type", "VARCHAR"),
    ("reaction_text", "VARCHAR"),
    ("severity", "VARCHAR"),
    ("status", "VARCHAR"),
    ("mapping_status", "VARCHAR"),
    *PROVENANCE_SCHEMA,
)

INTERVENTIONS_SCHEMA = (
    *BASE_ID_SCHEMA,
    ("source_table", "VARCHAR"),
    ("source_event_id", "VARCHAR"),
    ("event_start_time", "VARCHAR"),
    ("event_end_time", "VARCHAR"),
    ("event_start_offset", "VARCHAR"),
    ("event_end_offset", "VARCHAR"),
    ("source_intervention_code", "VARCHAR"),
    ("source_intervention_code_type", "VARCHAR"),
    ("source_intervention_text", "VARCHAR"),
    ("normalized_intervention_token", "VARCHAR"),
    ("value_text", "VARCHAR"),
    ("value_numeric", "DOUBLE"),
    ("unit", "VARCHAR"),
    ("normalized_unit", "VARCHAR"),
    ("mapping_status", "VARCHAR"),
    *PROVENANCE_SCHEMA,
)

TEMPORAL_EVENTS_SCHEMA = (
    *BASE_ID_SCHEMA,
    ("event_type", "VARCHAR"),
    ("source_domain", "VARCHAR"),
    ("source_table", "VARCHAR"),
    ("source_event_id", "VARCHAR"),
    ("event_start_time", "VARCHAR"),
    ("event_end_time", "VARCHAR"),
    ("event_start_offset", "VARCHAR"),
    ("event_end_offset", "VARCHAR"),
    ("event_token", "VARCHAR"),
    ("source_code", "VARCHAR"),
    ("source_text", "VARCHAR"),
    ("value_numeric", "DOUBLE"),
    ("value_text", "VARCHAR"),
    ("unit", "VARCHAR"),
    ("normalized_unit", "VARCHAR"),
    ("mapping_status", "VARCHAR"),
    *PROVENANCE_SCHEMA,
)


EMPTY_SCHEMAS: dict[str, Sequence[tuple[str, str]]] = {
    "demographics": DEMOGRAPHICS_SCHEMA,
    "conditions": CONDITIONS_SCHEMA,
    "medications": MEDICATIONS_SCHEMA,
    "labs": LABS_SCHEMA,
    "vitals": VITALS_SCHEMA,
    "allergies": ALLERGIES_SCHEMA,
    "interventions": INTERVENTIONS_SCHEMA,
    "temporal_events": TEMPORAL_EVENTS_SCHEMA,
}


def base_manifest(
    config: HarmonizationBuildConfig,
    *,
    status: str,
    mapping_resources: list[dict[str, Any]],
    generated_at: str,
    condition_mapping_resources: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the common manifest shell."""

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": generated_at,
        "status": status,
        "data_safety": {
            "contains_patient_rows": False,
            "reporting_level": "aggregate harmonization statuses only",
            "patient_level_artifacts_are_local_ignored": True,
        },
        "configuration": {
            "cohort_path": str(config.cohort_path),
            "extracts_root": str(config.extracts_root),
            "harmonized_root": str(config.harmonized_root),
            "mapping_root": str(config.mapping_root),
            "harmonization_version": config.harmonization_version,
            "cohort_version": config.cohort_version,
        },
        "versions": {
            "cohort_version": config.cohort_version,
            "harmonization_version": config.harmonization_version,
            "medication_mapping_version": MEDICATION_MAPPING_VERSION,
            "condition_mapping_version": CONDITION_MAPPING_VERSION,
            "source_native_mapping_version": SOURCE_NATIVE_MAPPING_VERSION,
        },
        "required_tables": list(REQUIRED_HARMONIZED_TABLES),
        "mapping_resources": mapping_resources,
        "condition_mapping_resources": (
            condition_mapping_resources
            if condition_mapping_resources is not None
            else []
        ),
        "condition_mapping_optional": True,
        "artifacts": {},
        "tables": [],
    }


def cohort_query(config: HarmonizationBuildConfig, *, generated_at: str) -> str:
    """Build the harmonized cohort-stay query."""

    return f"""
SELECT
    *,
    'not_applicable' AS mapping_status,
{
        provenance_sql(
            config,
            generated_at=generated_at,
            extraction_version=sql_string(NOT_APPLICABLE_VERSION),
            mapping_version=sql_string(NOT_APPLICABLE_VERSION),
        )
    }
FROM {parquet_scan(config.cohort_path)}
"""


def demographics_query(config: HarmonizationBuildConfig, *, generated_at: str) -> str:
    """Build one harmonized demographic row per cohort stay."""

    return f"""
SELECT
    source,
    source_version,
    patient_uid,
    encounter_uid,
    stay_uid,
    source_patient_id,
    source_encounter_id,
    source_stay_id,
    'cohort_stays' AS source_table,
    TRY_CAST(age_years AS DOUBLE) AS age_years,
    TRY_CAST(age_topcoded AS BOOLEAN) AS age_topcoded,
    sex,
    race_or_ethnicity,
    hospital_id,
    ward_id,
    admission_type,
    admission_source,
    unit_type,
    last_unit_type,
    stay_type,
    stay_sequence,
    cohort_rule,
    'source_native_cohort' AS mapping_status,
{
        provenance_sql(
            config,
            generated_at=generated_at,
            extraction_version=sql_string(NOT_APPLICABLE_VERSION),
            mapping_version=sql_string(NOT_APPLICABLE_VERSION),
        )
    }
FROM {parquet_scan(config.cohort_path)}
"""


def _read_csv_all_varchar(path: Path) -> str:
    """Return a DuckDB CSV scan expression that keeps values as strings."""

    return f"read_csv_auto({sql_string(path)}, header = true, all_varchar = true)"


def _case_expr(branches: Sequence[tuple[str, str]], default: str) -> str:
    """Assemble a SQL CASE from ordered (condition, value) branches."""

    if not branches:
        return default
    whens = "\n        ".join(
        f"WHEN {condition} THEN {value}" for condition, value in branches
    )
    return f"""CASE
        {whens}
        ELSE {default}
    END"""


def _condition_mapping_ctes_and_joins(
    available: Mapping[str, Path],
) -> tuple[list[str], list[str], dict[str, str]]:
    """Build optional mapping CTEs, LEFT JOINs, and availability booleans.

    Only files that exist and validate produce CTEs/JOINs, so the query never
    references an absent mapping table. This is what lets condition
    normalization degrade gracefully instead of failing like medications.
    """

    code_key = condition_code_key_sql
    text_key = normalized_text_token_sql
    ctes: list[str] = []
    joins: list[str] = []
    has: dict[str, str] = {
        "ccsr": "FALSE",
        "ccs": "FALSE",
        "gem_ccsr": "FALSE",
        "chapter": "FALSE",
        "text": "FALSE",
        "project": False,  # type: ignore[dict-item]
    }

    if "icd10_ccsr" in available:
        path = available["icd10_ccsr"]
        ctes.append(
            f"""ccsr AS (
    SELECT code_key, ccsr_category, ccsr_category_description
    FROM (
        SELECT
            {code_key("icd_code")} AS code_key,
            CAST(ccsr_category AS VARCHAR) AS ccsr_category,
            CAST(ccsr_category_description AS VARCHAR) AS ccsr_category_description,
            ROW_NUMBER() OVER (
                PARTITION BY {code_key("icd_code")}
                ORDER BY CAST(ccsr_category AS VARCHAR),
                    CAST(ccsr_category_description AS VARCHAR)
            ) AS rn
        FROM {_read_csv_all_varchar(path)}
        WHERE {code_key("icd_code")} IS NOT NULL
    )
    WHERE rn = 1
)"""
        )
        joins.append(
            "LEFT JOIN ccsr AS m_ccsr\n"
            "    ON m_ccsr.code_key = b.icd_code_key\n"
            "    AND b.icd_version_key IN ('10', 'eicu_mixed')"
        )
        has["ccsr"] = "m_ccsr.ccsr_category IS NOT NULL"

    if "icd9_ccs" in available:
        path = available["icd9_ccs"]
        ctes.append(
            f"""ccs AS (
    SELECT code_key, ccs_category, ccs_category_description
    FROM (
        SELECT
            {code_key("icd_code")} AS code_key,
            CAST(ccs_category AS VARCHAR) AS ccs_category,
            CAST(ccs_category_description AS VARCHAR) AS ccs_category_description,
            ROW_NUMBER() OVER (
                PARTITION BY {code_key("icd_code")}
                ORDER BY CAST(ccs_category AS VARCHAR),
                    CAST(ccs_category_description AS VARCHAR)
            ) AS rn
        FROM {_read_csv_all_varchar(path)}
        WHERE {code_key("icd_code")} IS NOT NULL
    )
    WHERE rn = 1
)"""
        )
        joins.append(
            "LEFT JOIN ccs AS m_ccs\n"
            "    ON m_ccs.code_key = b.icd_code_key\n"
            "    AND b.icd_version_key IN ('9', 'eicu_mixed')"
        )
        has["ccs"] = "m_ccs.ccs_category IS NOT NULL"

    if "icd9_to_icd10_gem" in available and "icd10_ccsr" in available:
        path = available["icd9_to_icd10_gem"]
        ctes.append(
            f"""gem AS (
    SELECT icd9_key, icd10_key, approx
    FROM (
        SELECT
            {code_key("icd9_code")} AS icd9_key,
            {code_key("icd10_code")} AS icd10_key,
            LOWER(TRIM(CAST(approximate_flag AS VARCHAR))) AS approx,
            ROW_NUMBER() OVER (
                PARTITION BY {code_key("icd9_code")}
                ORDER BY {code_key("icd10_code")}
            ) AS rn
        FROM {_read_csv_all_varchar(path)}
        WHERE {code_key("icd9_code")} IS NOT NULL
            AND {code_key("icd10_code")} IS NOT NULL
    )
    WHERE rn = 1
)"""
        )
        joins.append(
            "LEFT JOIN gem AS m_gem\n"
            "    ON m_gem.icd9_key = b.icd_code_key\n"
            "    AND b.icd_version_key IN ('9', 'eicu_mixed')"
        )
        joins.append(
            "LEFT JOIN ccsr AS m_ccsr_gem\n    ON m_ccsr_gem.code_key = m_gem.icd10_key"
        )
        has["gem_ccsr"] = "m_ccsr_gem.ccsr_category IS NOT NULL"

    if "icd_chapters" in available:
        path = available["icd_chapters"]
        ctes.append(
            f"""chapters AS (
    SELECT category_key, icd_version_key, chapter_code, chapter_name
    FROM (
        SELECT
            {code_key("category_code")} AS category_key,
            CASE
                WHEN TRIM(CAST(icd_version AS VARCHAR)) IN ('10', 'ICD10', 'icd10')
                    THEN '10'
                WHEN TRIM(CAST(icd_version AS VARCHAR)) IN ('9', 'ICD9', 'icd9')
                    THEN '9'
                ELSE LOWER(TRIM(CAST(icd_version AS VARCHAR)))
            END AS icd_version_key,
            CAST(chapter_code AS VARCHAR) AS chapter_code,
            CAST(chapter_name AS VARCHAR) AS chapter_name,
            ROW_NUMBER() OVER (
                PARTITION BY {code_key("category_code")}, CAST(icd_version AS VARCHAR)
                ORDER BY CAST(chapter_code AS VARCHAR)
            ) AS rn
        FROM {_read_csv_all_varchar(path)}
        WHERE {code_key("category_code")} IS NOT NULL
    )
    WHERE rn = 1
)"""
        )
        joins.append(
            "LEFT JOIN chapters AS m_chap\n"
            "    ON m_chap.category_key = SUBSTR(b.icd_code_key, 1, 3)\n"
            "    AND m_chap.icd_version_key = b.icd_version_key"
        )
        has["chapter"] = "m_chap.chapter_code IS NOT NULL"

    if "eicu_diagnosis_text_condition_map" in available:
        path = available["eicu_diagnosis_text_condition_map"]
        ctes.append(
            f"""text_map AS (
    SELECT text_key, condition_rollup_token, condition_name
    FROM (
        SELECT
            {text_key("diagnosisstring_normalized")} AS text_key,
            CAST(condition_rollup_token AS VARCHAR) AS condition_rollup_token,
            CAST(condition_name AS VARCHAR) AS condition_name,
            ROW_NUMBER() OVER (
                PARTITION BY {text_key("diagnosisstring_normalized")}
                ORDER BY CAST(condition_rollup_token AS VARCHAR)
            ) AS rn
        FROM {_read_csv_all_varchar(path)}
        WHERE {text_key("diagnosisstring_normalized")} IS NOT NULL
    )
    WHERE rn = 1
)"""
        )
        joins.append(
            "LEFT JOIN text_map AS m_text\n    ON m_text.text_key = b.diag_text_key"
        )
        has["text"] = "m_text.condition_rollup_token IS NOT NULL"

    if "project_condition_groups" in available:
        path = available["project_condition_groups"]
        ctes.append(
            f"""project_code AS (
    SELECT match_key, project_condition_group, project_condition_token
    FROM (
        SELECT
            {code_key("match_value")} AS match_key,
            CAST(project_condition_group AS VARCHAR) AS project_condition_group,
            CAST(project_condition_token AS VARCHAR) AS project_condition_token,
            ROW_NUMBER() OVER (
                PARTITION BY {code_key("match_value")}
                ORDER BY CAST(project_condition_token AS VARCHAR)
            ) AS rn
        FROM {_read_csv_all_varchar(path)}
        WHERE LOWER(TRIM(CAST(match_type AS VARCHAR))) = 'icd_code'
            AND {code_key("match_value")} IS NOT NULL
    )
    WHERE rn = 1
)"""
        )
        ctes.append(
            f"""project_text AS (
    SELECT match_key, project_condition_group, project_condition_token
    FROM (
        SELECT
            {text_key("match_value")} AS match_key,
            CAST(project_condition_group AS VARCHAR) AS project_condition_group,
            CAST(project_condition_token AS VARCHAR) AS project_condition_token,
            ROW_NUMBER() OVER (
                PARTITION BY {text_key("match_value")}
                ORDER BY CAST(project_condition_token AS VARCHAR)
            ) AS rn
        FROM {_read_csv_all_varchar(path)}
        WHERE LOWER(TRIM(CAST(match_type AS VARCHAR))) = 'text_token'
            AND {text_key("match_value")} IS NOT NULL
    )
    WHERE rn = 1
)"""
        )
        joins.append(
            "LEFT JOIN project_code AS m_pcode\n"
            "    ON m_pcode.match_key = b.icd_code_key"
        )
        joins.append(
            "LEFT JOIN project_text AS m_ptext\n"
            "    ON m_ptext.match_key = b.diag_text_key"
        )
        has["project"] = True  # type: ignore[assignment]

    return ctes, joins, has


def condition_normalized_query(
    config: HarmonizationBuildConfig,
    *,
    base_cte_sql: str,
    available: Mapping[str, Path],
    generated_at: str,
) -> str:
    """Wrap a source base query with optional shared-roll-up normalization."""

    ctes, joins, has = _condition_mapping_ctes_and_joins(available)

    cat_code = "SUBSTR(b.icd_code_key, 1, 3)"
    cat_token = (
        "CASE b.icd_version_key "
        "WHEN '10' THEN 'icd10cat:' WHEN '9' THEN 'icd9cat:' "
        f"ELSE 'icdcat:' END || {cat_code}"
    )
    cat_system = (
        "CASE b.icd_version_key "
        "WHEN '10' THEN 'ICD10CM_CATEGORY' WHEN '9' THEN 'ICD9CM_CATEGORY' "
        "ELSE 'ICD_CATEGORY' END"
    )
    chap_prefix = (
        "CASE b.icd_version_key "
        "WHEN '10' THEN 'icd10chap:' WHEN '9' THEN 'icd9chap:' "
        "ELSE 'icdchap:' END"
    )
    chap_system = (
        "CASE b.icd_version_key "
        "WHEN '10' THEN 'ICD10CM_CHAPTER' WHEN '9' THEN 'ICD9CM_CHAPTER' "
        "ELSE 'ICD_CHAPTER' END"
    )
    gem_confidence = (
        "CASE WHEN m_gem.approx IN ('1', 'a', 'approximate', 'true', 'yes') "
        "THEN 'approximate' ELSE 'authoritative_crosswalk' END"
    )

    # (availability key, condition, value-per-field...) in precedence order.
    tiers: list[dict[str, str]] = []
    if has["ccsr"] != "FALSE":
        tiers.append(
            {
                "cond": has["ccsr"],
                "system": "'CCSR'",
                "code": "m_ccsr.ccsr_category",
                "name": "m_ccsr.ccsr_category_description",
                "token": "'ccsr:' || m_ccsr.ccsr_category",
                "level": "'ccsr'",
                "source": "'icd10_ccsr.csv'",
                "confidence": "'exact'",
                "status": "'mapped_ccsr'",
            }
        )
    if has["ccs"] != "FALSE":
        tiers.append(
            {
                "cond": has["ccs"],
                "system": "'CCS'",
                "code": "m_ccs.ccs_category",
                "name": "m_ccs.ccs_category_description",
                "token": "'ccs:' || m_ccs.ccs_category",
                "level": "'ccs'",
                "source": "'icd9_ccs.csv'",
                "confidence": "'exact'",
                "status": "'mapped_ccs'",
            }
        )
    if has["gem_ccsr"] != "FALSE":
        tiers.append(
            {
                "cond": has["gem_ccsr"],
                "system": "'CCSR'",
                "code": "m_ccsr_gem.ccsr_category",
                "name": "m_ccsr_gem.ccsr_category_description",
                "token": "'ccsr:' || m_ccsr_gem.ccsr_category",
                "level": "'ccsr'",
                "source": "'icd9_to_icd10_gem.csv+icd10_ccsr.csv'",
                "confidence": gem_confidence,
                "status": "'mapped_icd_crosswalk'",
            }
        )
    if has["chapter"] != "FALSE":
        tiers.append(
            {
                "cond": has["chapter"],
                "system": chap_system,
                "code": "m_chap.chapter_code",
                "name": "m_chap.chapter_name",
                "token": f"{chap_prefix} || m_chap.chapter_code",
                "level": "'chapter'",
                "source": "'icd_chapters.csv'",
                "confidence": "'exact'",
                "status": "'mapped_chapter'",
            }
        )
    if has["text"] != "FALSE":
        tiers.append(
            {
                "cond": has["text"],
                "system": "'TEXT_CONDITION'",
                "code": "m_text.condition_rollup_token",
                "name": "m_text.condition_name",
                "token": "m_text.condition_rollup_token",
                "level": "'text_mapped'",
                "source": "'eicu_diagnosis_text_condition_map.csv'",
                "confidence": "'curated_text'",
                "status": "'mapped_text_to_condition'",
            }
        )
    # Structural ICD category is always available for a coded row (no file).
    tiers.append(
        {
            "cond": "b.icd_code_key IS NOT NULL",
            "system": cat_system,
            "code": cat_code,
            "name": "CAST(NULL AS VARCHAR)",
            "token": cat_token,
            "level": "'category'",
            "source": "'structural_icd_category'",
            "confidence": "'fallback_native'",
            "status": "'source_native_code'",
        }
    )

    def field_case(field: str) -> str:
        return _case_expr(
            [(tier["cond"], tier[field]) for tier in tiers],
            "CAST(NULL AS VARCHAR)",
        )

    status_branches = [(tier["cond"], tier["status"]) for tier in tiers]
    status_branches.append(
        (
            "NULLIF(TRIM(CAST(b.condition_text AS VARCHAR)), '') IS NOT NULL",
            "'source_native_text'",
        )
    )
    mapping_status_case = _case_expr(status_branches, "'unmapped_condition'")

    if has["project"]:
        project_group = (
            "COALESCE(m_pcode.project_condition_group, m_ptext.project_condition_group)"
        )
        project_token = (
            "COALESCE(m_pcode.project_condition_token, m_ptext.project_condition_token)"
        )
    else:
        project_group = "CAST(NULL AS VARCHAR)"
        project_token = "CAST(NULL AS VARCHAR)"

    with_clause = ",\n".join([f"base AS (\n{base_cte_sql}\n)", *ctes])
    join_clause = "\n".join(joins)
    provenance = provenance_sql(
        config,
        generated_at=generated_at,
        extraction_version="b.extraction_version",
        mapping_version=sql_string(CONDITION_MAPPING_VERSION),
    )

    return f"""
WITH {with_clause}
SELECT
    b.source,
    b.source_version,
    b.patient_uid,
    b.encounter_uid,
    b.stay_uid,
    b.source_patient_id,
    b.source_encounter_id,
    b.source_stay_id,
    b.source_table,
    b.source_sequence,
    b.condition_system,
    b.condition_code,
    b.condition_text,
    b.condition_token,
    b.condition_system AS source_condition_system,
    b.condition_code AS source_condition_code,
    b.condition_text AS source_condition_text,
    b.condition_token AS source_condition_token,
    {field_case("system")} AS normalized_condition_system,
    {field_case("code")} AS normalized_condition_code,
    {field_case("name")} AS normalized_condition_name,
    {field_case("token")} AS normalized_condition_token,
    {field_case("level")} AS condition_rollup_level,
    {field_case("source")} AS mapping_source,
    {field_case("confidence")} AS mapping_confidence,
    {project_group} AS project_condition_group,
    {project_token} AS project_condition_token,
    {mapping_status_case} AS mapping_status,
{provenance}
FROM base AS b
{join_clause}
"""


def condition_queries(
    config: HarmonizationBuildConfig, *, generated_at: str
) -> list[str]:
    """Build condition harmonization queries with optional semantic roll-ups.

    Source-native code/text/token/system fields are always preserved. Shared
    normalized roll-up columns are added when authoritative mapping files exist
    under ``$DATASET_ROOT/mappings/conditions/``; missing files degrade to
    structural ICD categories and source-native tokens without failing.
    """

    available = available_condition_mappings(mapping_root=config.mapping_root)
    queries: list[str] = []

    mimic_diagnoses = source_extract_path(config, "mimiciv", "diagnoses_icd.parquet")
    if mimic_diagnoses.exists():
        mimic_base = f"""
SELECT
    source,
    source_version,
    patient_uid,
    encounter_uid,
    stay_uid,
    source_patient_id,
    source_encounter_id,
    source_stay_id,
    'mimic_diagnoses_icd' AS source_table,
    CAST(seq_num AS VARCHAR) AS source_sequence,
    CASE
        WHEN icd_version = '10' THEN 'ICD10CM'
        WHEN icd_version = '9' THEN 'ICD9CM'
        ELSE 'ICD'
    END AS condition_system,
    icd_code AS condition_code,
    CAST(NULL AS VARCHAR) AS condition_text,
    LOWER('icd' || COALESCE(icd_version, '') || ':' || COALESCE(icd_code, '')) AS condition_token,
    extraction_version AS extraction_version,
    CASE
        WHEN icd_version = '10' THEN '10'
        WHEN icd_version = '9' THEN '9'
        ELSE NULL
    END AS icd_version_key,
    {condition_code_key_sql("icd_code")} AS icd_code_key,
    CAST(NULL AS VARCHAR) AS diag_text_key
FROM {parquet_scan(mimic_diagnoses)}
"""
        queries.append(
            condition_normalized_query(
                config,
                base_cte_sql=mimic_base,
                available=available,
                generated_at=generated_at,
            )
        )

    eicu_diagnosis = source_extract_path(config, "eicu_crd", "diagnosis.parquet")
    if eicu_diagnosis.exists():
        diagnosis_token = normalized_text_token_sql("diagnosisstring")
        first_code = "SPLIT_PART(CAST(icd9code AS VARCHAR), ',', 1)"
        eicu_base = f"""
SELECT
    source,
    source_version,
    patient_uid,
    encounter_uid,
    stay_uid,
    source_patient_id,
    source_encounter_id,
    source_stay_id,
    'eicu_diagnosis' AS source_table,
    diagnosispriority AS source_sequence,
    CASE
        WHEN NULLIF(TRIM(CAST(icd9code AS VARCHAR)), '') IS NOT NULL THEN 'ICD9CM'
        ELSE 'eICU diagnosis string'
    END AS condition_system,
    icd9code AS condition_code,
    diagnosisstring AS condition_text,
    COALESCE('icd9:' || NULLIF(TRIM(CAST(icd9code AS VARCHAR)), ''), {diagnosis_token}) AS condition_token,
    extraction_version AS extraction_version,
    CASE
        WHEN NULLIF(TRIM(CAST(icd9code AS VARCHAR)), '') IS NOT NULL THEN 'eicu_mixed'
        ELSE NULL
    END AS icd_version_key,
    {condition_code_key_sql(first_code)} AS icd_code_key,
    {diagnosis_token} AS diag_text_key
FROM {parquet_scan(eicu_diagnosis)}
"""
        queries.append(
            condition_normalized_query(
                config,
                base_cte_sql=eicu_base,
                available=available,
                generated_at=generated_at,
            )
        )
    return queries


def medication_queries(
    config: HarmonizationBuildConfig, *, generated_at: str
) -> list[str]:
    """Build medication harmonization queries for available extracts."""

    queries: list[str] = []
    mimic_prescriptions = source_extract_path(
        config, "mimiciv", "prescriptions.parquet"
    )
    mimic_mapping = config.mapping_root / "medications" / "mimic_ndc_rxnorm_atc.csv"
    if mimic_prescriptions.exists():
        queries.append(
            f"""
WITH mapping AS (
    SELECT *
    FROM read_csv_auto({sql_string(mimic_mapping)}, header = true, all_varchar = true)
)
SELECT
    s.source,
    s.source_version,
    s.patient_uid,
    s.encounter_uid,
    s.stay_uid,
    s.source_patient_id,
    s.source_encounter_id,
    s.source_stay_id,
    'mimic_prescriptions' AS source_table,
    pharmacy_id AS source_event_id,
    starttime AS event_start_time,
    stoptime AS event_end_time,
    drug AS medication_source_name,
    s.ndc AS source_code,
    'ndc' AS source_code_type,
    route,
    dose_val_rx AS dose_value,
    dose_unit_rx AS dose_unit,
    m.rxcui,
    m.ingredient_name,
    m.rxnorm_name,
    m.atc_code,
    m.atc_level,
    CASE
        WHEN NULLIF(TRIM(CAST(m.rxcui AS VARCHAR)), '') IS NOT NULL
            OR NULLIF(TRIM(CAST(m.atc_code AS VARCHAR)), '') IS NOT NULL
        THEN 'mapped_rxnorm_or_atc'
        ELSE 'unmapped'
    END AS mapping_status,
{
                provenance_sql(
                    config,
                    generated_at=generated_at,
                    extraction_version="s.extraction_version",
                    mapping_version=sql_string(MEDICATION_MAPPING_VERSION),
                )
            }
FROM {parquet_scan(mimic_prescriptions)} AS s
LEFT JOIN mapping AS m
    ON NULLIF(TRIM(CAST(s.ndc AS VARCHAR)), '') = NULLIF(TRIM(CAST(m.ndc AS VARCHAR)), '')
"""
        )

    eicu_medication = source_extract_path(config, "eicu_crd", "medication.parquet")
    eicu_mapping = config.mapping_root / "medications" / "eicu_drug_rxnorm_atc.csv"
    if eicu_medication.exists():
        drug_name_token = "LOWER(TRIM(CAST(drugname AS VARCHAR)))"
        mapping_name_token = "LOWER(TRIM(CAST(drug_name AS VARCHAR)))"
        queries.append(
            f"""
WITH source_rows AS (
    SELECT ROW_NUMBER() OVER () AS extraction_row_id, *
    FROM {parquet_scan(eicu_medication)}
),
mapping AS (
    SELECT *
    FROM read_csv_auto({sql_string(eicu_mapping)}, header = true, all_varchar = true)
),
mapping_keyed AS (
    SELECT
        *,
        NULLIF(TRIM(CAST(drughiclseqno AS VARCHAR)), '') AS drughiclseqno_key,
        NULLIF(TRIM(CAST(gtc AS VARCHAR)), '') AS gtc_key,
        NULLIF({mapping_name_token}, '') AS drug_name_key
    FROM mapping
),
mapped_only AS (
    SELECT *
    FROM mapping_keyed
    WHERE
        NULLIF(TRIM(CAST(rxcui AS VARCHAR)), '') IS NOT NULL
        OR NULLIF(TRIM(CAST(atc_code AS VARCHAR)), '') IS NOT NULL
),
concept_mapping AS (
    SELECT *
    FROM mapping_keyed
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY drughiclseqno_key, gtc_key, drug_name_key
        ORDER BY
            CASE
                WHEN NULLIF(TRIM(CAST(rxcui AS VARCHAR)), '') IS NOT NULL
                    OR NULLIF(TRIM(CAST(atc_code AS VARCHAR)), '') IS NOT NULL
                THEN 0
                ELSE 1
            END,
            rxcui,
            atc_code
    ) = 1
),
hicl_mapping AS (
    SELECT *
    FROM mapped_only
    WHERE drughiclseqno_key IS NOT NULL
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY drughiclseqno_key
        ORDER BY rxcui, atc_code, ingredient_name, rxnorm_name
    ) = 1
),
gtc_mapping AS (
    SELECT *
    FROM mapped_only
    WHERE gtc_key IS NOT NULL
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY gtc_key
        ORDER BY rxcui, atc_code, ingredient_name, rxnorm_name
    ) = 1
),
name_mapping AS (
    SELECT *
    FROM mapped_only
    WHERE drug_name_key IS NOT NULL
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY drug_name_key
        ORDER BY rxcui, atc_code, ingredient_name, rxnorm_name
    ) = 1
),
candidate_mapping AS (
    SELECT s.extraction_row_id, m.*, 1 AS match_priority
    FROM source_rows AS s
    INNER JOIN concept_mapping AS m
        ON NULLIF(TRIM(CAST(s.drughiclseqno AS VARCHAR)), '') IS NOT DISTINCT FROM m.drughiclseqno_key
        AND NULLIF(TRIM(CAST(s.gtc AS VARCHAR)), '') IS NOT DISTINCT FROM m.gtc_key
        AND NULLIF({drug_name_token}, '') IS NOT DISTINCT FROM m.drug_name_key
    UNION ALL
    SELECT s.extraction_row_id, m.*, 2 AS match_priority
    FROM source_rows AS s
    INNER JOIN hicl_mapping AS m
        ON NULLIF(TRIM(CAST(s.drughiclseqno AS VARCHAR)), '') = NULLIF(TRIM(CAST(m.drughiclseqno AS VARCHAR)), '')
    UNION ALL
    SELECT s.extraction_row_id, m.*, 3 AS match_priority
    FROM source_rows AS s
    INNER JOIN gtc_mapping AS m
        ON NULLIF(TRIM(CAST(s.gtc AS VARCHAR)), '') = NULLIF(TRIM(CAST(m.gtc AS VARCHAR)), '')
    UNION ALL
    SELECT s.extraction_row_id, m.*, 4 AS match_priority
    FROM source_rows AS s
    INNER JOIN name_mapping AS m
        ON NULLIF({drug_name_token}, '') = NULLIF({mapping_name_token}, '')
),
best_mapping AS (
    SELECT *
    FROM candidate_mapping
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY extraction_row_id
        ORDER BY match_priority
    ) = 1
)
SELECT
    s.source,
    s.source_version,
    s.patient_uid,
    s.encounter_uid,
    s.stay_uid,
    s.source_patient_id,
    s.source_encounter_id,
    s.source_stay_id,
    'eicu_medication' AS source_table,
    medicationid AS source_event_id,
    drugstartoffset AS event_start_time,
    drugstopoffset AS event_end_time,
    drugname AS medication_source_name,
    COALESCE(s.drughiclseqno, s.gtc) AS source_code,
    CASE
        WHEN NULLIF(TRIM(CAST(s.drughiclseqno AS VARCHAR)), '') IS NOT NULL THEN 'drughiclseqno'
        WHEN NULLIF(TRIM(CAST(s.gtc AS VARCHAR)), '') IS NOT NULL THEN 'gtc'
        ELSE 'drug_name'
    END AS source_code_type,
    routeadmin AS route,
    dosage AS dose_value,
    CAST(NULL AS VARCHAR) AS dose_unit,
    b.rxcui,
    b.ingredient_name,
    b.rxnorm_name,
    b.atc_code,
    b.atc_level,
    CASE
        WHEN NULLIF(TRIM(CAST(b.rxcui AS VARCHAR)), '') IS NOT NULL
            OR NULLIF(TRIM(CAST(b.atc_code AS VARCHAR)), '') IS NOT NULL
        THEN 'mapped_rxnorm_or_atc'
        ELSE 'unmapped'
    END AS mapping_status,
{
                provenance_sql(
                    config,
                    generated_at=generated_at,
                    extraction_version="s.extraction_version",
                    mapping_version=sql_string(MEDICATION_MAPPING_VERSION),
                )
            }
FROM source_rows AS s
LEFT JOIN best_mapping AS b
    ON s.extraction_row_id = b.extraction_row_id
"""
        )
    return queries


def lab_queries(config: HarmonizationBuildConfig, *, generated_at: str) -> list[str]:
    """Build conservative source-native lab harmonization queries."""

    queries: list[str] = []
    mimic_labs = source_extract_path(config, "mimiciv", "labevents.parquet")
    mimic_lab_items = source_extract_path(config, "mimiciv", "d_labitems.parquet")
    if mimic_labs.exists():
        if mimic_lab_items.exists():
            source_lab_name = "d.label"
            source_lab_join = f"""
LEFT JOIN {parquet_scan(mimic_lab_items)} AS d
    ON NULLIF(TRIM(CAST(l.itemid AS VARCHAR)), '') = NULLIF(TRIM(CAST(d.itemid AS VARCHAR)), '')
"""
        else:
            source_lab_name = "CAST(NULL AS VARCHAR)"
            source_lab_join = ""
        lab_token = normalized_text_token_sql(f"COALESCE({source_lab_name}, l.itemid)")
        normalized_unit = normalized_unit_sql("l.valueuom")
        queries.append(
            f"""
SELECT
    l.source,
    l.source_version,
    l.patient_uid,
    l.encounter_uid,
    l.stay_uid,
    l.source_patient_id,
    l.source_encounter_id,
    l.source_stay_id,
    'mimic_labevents' AS source_table,
    l.labevent_id AS source_event_id,
    l.charttime AS event_time,
    CAST(NULL AS VARCHAR) AS event_time_offset,
    l.itemid AS source_lab_code,
    {source_lab_name} AS source_lab_name,
    {lab_token} AS normalized_lab_token,
    TRY_CAST(l.valuenum AS DOUBLE) AS lab_value_numeric,
    l.value AS lab_value_text,
    l.valueuom AS unit,
    {normalized_unit} AS normalized_unit,
    l.ref_range_lower AS reference_range_lower,
    l.ref_range_upper AS reference_range_upper,
    l.flag AS abnormal_flag,
    CASE
        WHEN {lab_token} IS NOT NULL THEN 'source_native_token'
        ELSE 'unmapped_source_concept'
    END AS mapping_status,
{
                provenance_sql(
                    config,
                    generated_at=generated_at,
                    extraction_version="l.extraction_version",
                    mapping_version=sql_string(SOURCE_NATIVE_MAPPING_VERSION),
                )
            }
FROM {parquet_scan(mimic_labs)} AS l
{source_lab_join}
"""
        )

    eicu_labs = source_extract_path(config, "eicu_crd", "lab.parquet")
    if eicu_labs.exists():
        lab_token = normalized_text_token_sql("labname")
        unit_expr = "COALESCE(labmeasurenamesystem, labmeasurenameinterface)"
        normalized_unit = normalized_unit_sql(unit_expr)
        queries.append(
            f"""
SELECT
    source,
    source_version,
    patient_uid,
    encounter_uid,
    stay_uid,
    source_patient_id,
    source_encounter_id,
    source_stay_id,
    'eicu_lab' AS source_table,
    labid AS source_event_id,
    CAST(NULL AS VARCHAR) AS event_time,
    labresultoffset AS event_time_offset,
    labtypeid AS source_lab_code,
    labname AS source_lab_name,
    {lab_token} AS normalized_lab_token,
    TRY_CAST(labresult AS DOUBLE) AS lab_value_numeric,
    labresulttext AS lab_value_text,
    {unit_expr} AS unit,
    {normalized_unit} AS normalized_unit,
    CAST(NULL AS VARCHAR) AS reference_range_lower,
    CAST(NULL AS VARCHAR) AS reference_range_upper,
    CAST(NULL AS VARCHAR) AS abnormal_flag,
    CASE
        WHEN {lab_token} IS NOT NULL THEN 'source_native_token'
        ELSE 'unmapped_source_concept'
    END AS mapping_status,
{
                provenance_sql(
                    config,
                    generated_at=generated_at,
                    extraction_version="extraction_version",
                    mapping_version=sql_string(SOURCE_NATIVE_MAPPING_VERSION),
                )
            }
FROM {parquet_scan(eicu_labs)}
"""
        )
    return queries


def eicu_vital_column_query(
    config: HarmonizationBuildConfig,
    *,
    path: Path,
    source_table: str,
    event_id_column: str,
    value_column: str,
    normalized_token: str,
    generated_at: str,
) -> str:
    """Build one eICU source-native vital column projection."""

    return f"""
SELECT
    source,
    source_version,
    patient_uid,
    encounter_uid,
    stay_uid,
    source_patient_id,
    source_encounter_id,
    source_stay_id,
    {sql_string(source_table)} AS source_table,
    {event_id_column} AS source_event_id,
    CAST(NULL AS VARCHAR) AS event_time,
    observationoffset AS event_time_offset,
    {sql_string(value_column)} AS source_vital_code,
    {sql_string(value_column)} AS source_vital_name,
    {sql_string(normalized_token)} AS normalized_vital_token,
    TRY_CAST({value_column} AS DOUBLE) AS value_numeric,
    CAST(NULL AS VARCHAR) AS unit,
    CAST(NULL AS VARCHAR) AS normalized_unit,
    'source_native_column' AS mapping_status,
{
        provenance_sql(
            config,
            generated_at=generated_at,
            extraction_version="extraction_version",
            mapping_version=sql_string(SOURCE_NATIVE_MAPPING_VERSION),
        )
    }
FROM {parquet_scan(path)}
WHERE NULLIF(TRIM(CAST({value_column} AS VARCHAR)), '') IS NOT NULL
"""


def mimic_chartevents_vital_query(
    config: HarmonizationBuildConfig,
    *,
    chartevents_path: Path,
    items_path: Path,
    generated_at: str,
) -> str:
    """Build the MIMIC chartevents charted-vital projection.

    Maps the curated MIMIC-IV chartevents vital itemids to the shared
    ``normalized_vital_token`` vocabulary (see ``MIMIC_CHARTEVENTS_VITAL_ITEMIDS``),
    preserving the source itemid, label, value, and unit. Defensively restricts
    to the mapped itemids even though the extract is already itemid-filtered.
    """

    token_branches = "\n".join(
        f"        WHEN {sql_string(itemid)} THEN {sql_string(token)}"
        for itemid, token in sorted(MIMIC_CHARTEVENTS_VITAL_ITEMIDS.items())
    )
    token_case = (
        "CASE NULLIF(TRIM(CAST(c.itemid AS VARCHAR)), '')\n"
        f"{token_branches}\n        ELSE NULL\n    END"
    )
    itemid_list = ", ".join(
        sql_string(itemid) for itemid in sorted(MIMIC_CHARTEVENTS_VITAL_ITEMIDS)
    )
    if items_path.exists():
        item_label = "d.label"
        item_join = f"""
LEFT JOIN {parquet_scan(items_path)} AS d
    ON NULLIF(TRIM(CAST(c.itemid AS VARCHAR)), '') = NULLIF(TRIM(CAST(d.itemid AS VARCHAR)), '')
"""
    else:
        item_label = "CAST(NULL AS VARCHAR)"
        item_join = ""
    normalized_unit = normalized_unit_sql("c.valueuom")
    return f"""
SELECT
    c.source,
    c.source_version,
    c.patient_uid,
    c.encounter_uid,
    c.stay_uid,
    c.source_patient_id,
    c.source_encounter_id,
    c.source_stay_id,
    'mimic_chartevents' AS source_table,
    CAST(NULL AS VARCHAR) AS source_event_id,
    c.charttime AS event_time,
    CAST(NULL AS VARCHAR) AS event_time_offset,
    c.itemid AS source_vital_code,
    COALESCE({item_label}, c.itemid) AS source_vital_name,
    {token_case} AS normalized_vital_token,
    TRY_CAST(c.valuenum AS DOUBLE) AS value_numeric,
    c.valueuom AS unit,
    {normalized_unit} AS normalized_unit,
    'mapped_vital_itemid' AS mapping_status,
{
        provenance_sql(
            config,
            generated_at=generated_at,
            extraction_version="c.extraction_version",
            mapping_version=sql_string(SOURCE_NATIVE_MAPPING_VERSION),
        )
    }
FROM {parquet_scan(chartevents_path)} AS c
{item_join}
WHERE NULLIF(TRIM(CAST(c.itemid AS VARCHAR)), '') IN ({itemid_list})
"""


def vital_queries(config: HarmonizationBuildConfig, *, generated_at: str) -> list[str]:
    """Build source-native vital harmonization queries for available extracts."""

    queries: list[str] = []
    mimic_chartevents = source_extract_path(config, "mimiciv", "chartevents.parquet")
    mimic_items = source_extract_path(config, "mimiciv", "d_items.parquet")
    if mimic_chartevents.exists():
        queries.append(
            mimic_chartevents_vital_query(
                config,
                chartevents_path=mimic_chartevents,
                items_path=mimic_items,
                generated_at=generated_at,
            )
        )
    periodic = source_extract_path(config, "eicu_crd", "vital_periodic.parquet")
    if periodic.exists():
        for value_column, normalized_token in (
            ("temperature", "temperature"),
            ("sao2", "spo2"),
            ("heartrate", "heart_rate"),
            ("respiration", "respiratory_rate"),
            ("systemicsystolic", "systolic_blood_pressure"),
            ("systemicdiastolic", "diastolic_blood_pressure"),
            ("systemicmean", "mean_arterial_pressure"),
        ):
            queries.append(
                eicu_vital_column_query(
                    config,
                    path=periodic,
                    source_table="eicu_vital_periodic",
                    event_id_column="vitalperiodicid",
                    value_column=value_column,
                    normalized_token=normalized_token,
                    generated_at=generated_at,
                )
            )

    aperiodic = source_extract_path(config, "eicu_crd", "vital_aperiodic.parquet")
    if aperiodic.exists():
        for value_column, normalized_token in (
            ("noninvasivesystolic", "noninvasive_systolic_blood_pressure"),
            ("noninvasivediastolic", "noninvasive_diastolic_blood_pressure"),
            ("noninvasivemean", "noninvasive_mean_arterial_pressure"),
        ):
            queries.append(
                eicu_vital_column_query(
                    config,
                    path=aperiodic,
                    source_table="eicu_vital_aperiodic",
                    event_id_column="vitalaperiodicid",
                    value_column=value_column,
                    normalized_token=normalized_token,
                    generated_at=generated_at,
                )
            )
    return queries


def allergy_queries(
    config: HarmonizationBuildConfig, *, generated_at: str
) -> list[str]:
    """Build allergy harmonization queries for available extracts."""

    queries: list[str] = []
    eicu_allergy = source_extract_path(config, "eicu_crd", "allergy.parquet")
    if eicu_allergy.exists():
        allergen_name = "COALESCE(allergyname, drugname)"
        allergen_token = normalized_text_token_sql(allergen_name)
        queries.append(
            f"""
SELECT
    source,
    source_version,
    patient_uid,
    encounter_uid,
    stay_uid,
    source_patient_id,
    source_encounter_id,
    source_stay_id,
    'eicu_allergy' AS source_table,
    allergyid AS source_event_id,
    allergyoffset AS event_time,
    allergyenteredoffset AS event_entered_time,
    {allergen_name} AS allergen_source_name,
    drughiclseqno AS allergen_source_code,
    {allergen_token} AS normalized_allergen_token,
    allergytype AS allergy_type,
    CAST(NULL AS VARCHAR) AS reaction_text,
    CAST(NULL AS VARCHAR) AS severity,
    CASE
        WHEN rxincluded = 'True' OR rxincluded = '1' THEN 'rx_included'
        WHEN writtenineicu = 'True' OR writtenineicu = '1' THEN 'written_in_icu'
        ELSE CAST(NULL AS VARCHAR)
    END AS status,
    CASE
        WHEN {allergen_token} IS NOT NULL THEN 'source_native_text'
        ELSE 'unmapped_source_concept'
    END AS mapping_status,
{
                provenance_sql(
                    config,
                    generated_at=generated_at,
                    extraction_version="extraction_version",
                    mapping_version=sql_string(SOURCE_NATIVE_MAPPING_VERSION),
                )
            }
FROM {parquet_scan(eicu_allergy)}
"""
        )
    return queries


def intervention_queries(
    config: HarmonizationBuildConfig, *, generated_at: str
) -> list[str]:
    """Build procedure, treatment, infusion, and support-event harmonization."""

    queries: list[str] = []
    mimic_procedures = source_extract_path(config, "mimiciv", "procedures_icd.parquet")
    if mimic_procedures.exists():
        queries.append(
            f"""
SELECT
    source,
    source_version,
    patient_uid,
    encounter_uid,
    stay_uid,
    source_patient_id,
    source_encounter_id,
    source_stay_id,
    'mimic_procedures_icd' AS source_table,
    seq_num AS source_event_id,
    chartdate AS event_start_time,
    CAST(NULL AS VARCHAR) AS event_end_time,
    CAST(NULL AS VARCHAR) AS event_start_offset,
    CAST(NULL AS VARCHAR) AS event_end_offset,
    icd_code AS source_intervention_code,
    CASE
        WHEN icd_version = '10' THEN 'ICD10PCS'
        WHEN icd_version = '9' THEN 'ICD9PROC'
        ELSE 'ICD'
    END AS source_intervention_code_type,
    CAST(NULL AS VARCHAR) AS source_intervention_text,
    LOWER('icd' || COALESCE(icd_version, '') || ':procedure:' || COALESCE(icd_code, '')) AS normalized_intervention_token,
    CAST(NULL AS VARCHAR) AS value_text,
    CAST(NULL AS DOUBLE) AS value_numeric,
    CAST(NULL AS VARCHAR) AS unit,
    CAST(NULL AS VARCHAR) AS normalized_unit,
    'source_native_code' AS mapping_status,
{
                provenance_sql(
                    config,
                    generated_at=generated_at,
                    extraction_version="extraction_version",
                    mapping_version=sql_string(SOURCE_NATIVE_MAPPING_VERSION),
                )
            }
FROM {parquet_scan(mimic_procedures)}
"""
        )

    mimic_procedureevents = source_extract_path(
        config, "mimiciv", "procedureevents.parquet"
    )
    mimic_items = source_extract_path(config, "mimiciv", "d_items.parquet")
    if mimic_procedureevents.exists():
        if mimic_items.exists():
            item_label = "d.label"
            item_join = f"""
LEFT JOIN {parquet_scan(mimic_items)} AS d
    ON NULLIF(TRIM(CAST(p.itemid AS VARCHAR)), '') = NULLIF(TRIM(CAST(d.itemid AS VARCHAR)), '')
"""
        else:
            item_label = "CAST(NULL AS VARCHAR)"
            item_join = ""
        text_expr = (
            f"COALESCE({item_label}, p.ordercategoryname, "
            "p.ordercategorydescription, p.itemid)"
        )
        token_expr = normalized_text_token_sql(text_expr)
        normalized_unit = normalized_unit_sql("p.valueuom")
        queries.append(
            f"""
SELECT
    p.source,
    p.source_version,
    p.patient_uid,
    p.encounter_uid,
    p.stay_uid,
    p.source_patient_id,
    p.source_encounter_id,
    p.source_stay_id,
    'mimic_procedureevents' AS source_table,
    p.orderid AS source_event_id,
    p.starttime AS event_start_time,
    p.endtime AS event_end_time,
    CAST(NULL AS VARCHAR) AS event_start_offset,
    CAST(NULL AS VARCHAR) AS event_end_offset,
    p.itemid AS source_intervention_code,
    'itemid' AS source_intervention_code_type,
    {text_expr} AS source_intervention_text,
    {token_expr} AS normalized_intervention_token,
    p.statusdescription AS value_text,
    TRY_CAST(p.value AS DOUBLE) AS value_numeric,
    p.valueuom AS unit,
    {normalized_unit} AS normalized_unit,
    CASE
        WHEN {token_expr} IS NOT NULL THEN 'source_native_token'
        ELSE 'unmapped_source_concept'
    END AS mapping_status,
{
                provenance_sql(
                    config,
                    generated_at=generated_at,
                    extraction_version="p.extraction_version",
                    mapping_version=sql_string(SOURCE_NATIVE_MAPPING_VERSION),
                )
            }
FROM {parquet_scan(mimic_procedureevents)} AS p
{item_join}
"""
        )

    eicu_treatment = source_extract_path(config, "eicu_crd", "treatment.parquet")
    if eicu_treatment.exists():
        token_expr = normalized_text_token_sql("treatmentstring")
        queries.append(
            f"""
SELECT
    source,
    source_version,
    patient_uid,
    encounter_uid,
    stay_uid,
    source_patient_id,
    source_encounter_id,
    source_stay_id,
    'eicu_treatment' AS source_table,
    treatmentid AS source_event_id,
    CAST(NULL AS VARCHAR) AS event_start_time,
    CAST(NULL AS VARCHAR) AS event_end_time,
    treatmentoffset AS event_start_offset,
    CAST(NULL AS VARCHAR) AS event_end_offset,
    CAST(NULL AS VARCHAR) AS source_intervention_code,
    CAST(NULL AS VARCHAR) AS source_intervention_code_type,
    treatmentstring AS source_intervention_text,
    {token_expr} AS normalized_intervention_token,
    activeupondischarge AS value_text,
    CAST(NULL AS DOUBLE) AS value_numeric,
    CAST(NULL AS VARCHAR) AS unit,
    CAST(NULL AS VARCHAR) AS normalized_unit,
    CASE
        WHEN {token_expr} IS NOT NULL THEN 'source_native_text'
        ELSE 'unmapped_source_concept'
    END AS mapping_status,
{
                provenance_sql(
                    config,
                    generated_at=generated_at,
                    extraction_version="extraction_version",
                    mapping_version=sql_string(SOURCE_NATIVE_MAPPING_VERSION),
                )
            }
FROM {parquet_scan(eicu_treatment)}
"""
        )

    eicu_infusion = source_extract_path(config, "eicu_crd", "infusion_drug.parquet")
    if eicu_infusion.exists():
        token_expr = normalized_text_token_sql("drugname")
        queries.append(
            f"""
SELECT
    source,
    source_version,
    patient_uid,
    encounter_uid,
    stay_uid,
    source_patient_id,
    source_encounter_id,
    source_stay_id,
    'eicu_infusion_drug' AS source_table,
    infusiondrugid AS source_event_id,
    CAST(NULL AS VARCHAR) AS event_start_time,
    CAST(NULL AS VARCHAR) AS event_end_time,
    infusionoffset AS event_start_offset,
    CAST(NULL AS VARCHAR) AS event_end_offset,
    CAST(NULL AS VARCHAR) AS source_intervention_code,
    CAST(NULL AS VARCHAR) AS source_intervention_code_type,
    drugname AS source_intervention_text,
    {token_expr} AS normalized_intervention_token,
    COALESCE(drugrate, infusionrate) AS value_text,
    TRY_CAST(drugamount AS DOUBLE) AS value_numeric,
    CAST(NULL AS VARCHAR) AS unit,
    CAST(NULL AS VARCHAR) AS normalized_unit,
    CASE
        WHEN {token_expr} IS NOT NULL THEN 'source_native_text'
        ELSE 'unmapped_source_concept'
    END AS mapping_status,
{
                provenance_sql(
                    config,
                    generated_at=generated_at,
                    extraction_version="extraction_version",
                    mapping_version=sql_string(SOURCE_NATIVE_MAPPING_VERSION),
                )
            }
FROM {parquet_scan(eicu_infusion)}
"""
        )

    eicu_apache_aps = source_extract_path(config, "eicu_crd", "apache_aps_var.parquet")
    if eicu_apache_aps.exists():
        for column in ("intubated", "vent", "dialysis"):
            queries.append(
                f"""
SELECT
    source,
    source_version,
    patient_uid,
    encounter_uid,
    stay_uid,
    source_patient_id,
    source_encounter_id,
    source_stay_id,
    'eicu_apache_aps_var' AS source_table,
    apacheapsvarid AS source_event_id,
    CAST(NULL AS VARCHAR) AS event_start_time,
    CAST(NULL AS VARCHAR) AS event_end_time,
    CAST(NULL AS VARCHAR) AS event_start_offset,
    CAST(NULL AS VARCHAR) AS event_end_offset,
    {sql_string(column)} AS source_intervention_code,
    'apache_aps_var' AS source_intervention_code_type,
    {sql_string(column)} AS source_intervention_text,
    {sql_string(column)} AS normalized_intervention_token,
    {column} AS value_text,
    TRY_CAST({column} AS DOUBLE) AS value_numeric,
    CAST(NULL AS VARCHAR) AS unit,
    CAST(NULL AS VARCHAR) AS normalized_unit,
    'source_native_column' AS mapping_status,
{
                    provenance_sql(
                        config,
                        generated_at=generated_at,
                        extraction_version="extraction_version",
                        mapping_version=sql_string(SOURCE_NATIVE_MAPPING_VERSION),
                    )
                }
FROM {parquet_scan(eicu_apache_aps)}
WHERE NULLIF(TRIM(CAST({column} AS VARCHAR)), '') IS NOT NULL
"""
            )
    return queries


def materialize_table(
    connection: duckdb.DuckDBPyConnection,
    *,
    table_name: str,
    query: str,
    output_path: Path,
    reason: str | None = None,
) -> dict[str, Any]:
    """Materialize one harmonized table and return its manifest record."""

    try:
        row_count = copy_query_to_parquet(connection, query, output_path)
    except Exception as error:
        return {
            "table_name": table_name,
            "output_path": str(output_path),
            "status": "failed",
            "row_count": None,
            "reason": safe_error_message(error),
        }
    return {
        "table_name": table_name,
        "output_path": str(output_path),
        "status": "completed",
        "row_count": row_count,
        "reason": reason,
    }


def materialize_domain_table(
    connection: duckdb.DuckDBPyConnection,
    *,
    table_name: str,
    queries: Sequence[str],
    output_path: Path,
    empty_schema: Sequence[tuple[str, str]],
) -> dict[str, Any]:
    """Materialize a domain table, writing an empty schema if extracts are absent."""

    if queries:
        return materialize_table(
            connection,
            table_name=table_name,
            query=union_queries(queries),
            output_path=output_path,
        )
    return materialize_table(
        connection,
        table_name=table_name,
        query=empty_select(empty_schema),
        output_path=output_path,
        reason="no source extracts found for this domain",
    )


def table_has_column(
    connection: duckdb.DuckDBPyConnection, path: Path, column_name: str
) -> bool:
    """Return whether a Parquet artifact has a column."""

    rows = connection.execute(f"DESCRIBE SELECT * FROM {parquet_scan(path)}").fetchall()
    return any(row[0] == column_name for row in rows)


def aggregate_counts_query(path: Path, *, domain: str) -> str:
    """Build an aggregate coverage count query for one harmonized table."""

    return f"""
SELECT
    {sql_string(domain)} AS domain,
    source,
    mapping_status,
    COUNT(*) AS row_count,
    {sql_string(path)} AS artifact_path
FROM {parquet_scan(path)}
GROUP BY source, mapping_status
"""


def condition_rollup_counts_query(path: Path) -> str:
    """Aggregate condition rows by source, roll-up level, and mapping status."""

    return f"""
SELECT
    source,
    COALESCE(condition_rollup_level, 'none') AS condition_rollup_level,
    mapping_status,
    COUNT(*) AS row_count
FROM {parquet_scan(path)}
GROUP BY source, condition_rollup_level, mapping_status
ORDER BY source, condition_rollup_level, mapping_status
"""


def condition_summary_query(path: Path) -> str:
    """Aggregate per-source condition normalization coverage counts."""

    return f"""
SELECT
    source,
    COUNT(*) AS total_rows,
    SUM(CASE WHEN normalized_condition_token IS NOT NULL THEN 1 ELSE 0 END)
        AS rollup_mapped_rows,
    SUM(CASE WHEN condition_rollup_level IN ('ccsr', 'ccs') THEN 1 ELSE 0 END)
        AS ccs_ccsr_rows,
    SUM(CASE WHEN mapping_status = 'mapped_icd_crosswalk' THEN 1 ELSE 0 END)
        AS icd_crosswalk_rows,
    SUM(CASE WHEN mapping_status = 'mapped_chapter' THEN 1 ELSE 0 END)
        AS chapter_rows,
    SUM(CASE WHEN condition_rollup_level = 'category' THEN 1 ELSE 0 END)
        AS category_rows,
    SUM(CASE WHEN mapping_status = 'mapped_text_to_condition' THEN 1 ELSE 0 END)
        AS text_mapped_rows,
    SUM(CASE WHEN mapping_status = 'source_native_code' THEN 1 ELSE 0 END)
        AS source_native_code_rows,
    SUM(CASE WHEN mapping_status = 'source_native_text' THEN 1 ELSE 0 END)
        AS source_native_text_rows,
    SUM(CASE WHEN project_condition_token IS NOT NULL THEN 1 ELSE 0 END)
        AS project_group_rows,
    SUM(CASE WHEN mapping_status = 'unmapped_condition' THEN 1 ELSE 0 END)
        AS unmapped_rows
FROM {parquet_scan(path)}
GROUP BY source
ORDER BY source
"""


def eicu_text_review_query(path: Path) -> str:
    """Aggregate distinct eICU diagnosis strings and their mapping status.

    Returns concept-level counts only (no patient rows, no identifiers).
    """

    text_key = normalized_text_token_sql("source_condition_text")
    return f"""
SELECT
    {text_key} AS diagnosisstring_normalized,
    COUNT(*) AS row_count,
    MAX(CASE WHEN condition_rollup_level = 'text_mapped'
        THEN normalized_condition_token END) AS mapped_condition_rollup_token,
    MAX(CASE WHEN condition_rollup_level = 'text_mapped'
        THEN normalized_condition_name END) AS mapped_condition_name,
    MAX(CASE WHEN condition_rollup_level = 'text_mapped'
        THEN mapping_source END) AS mapping_source,
    MAX(CASE WHEN condition_rollup_level = 'text_mapped'
        THEN mapping_confidence END) AS mapping_confidence,
    CASE
        WHEN MAX(CASE WHEN condition_rollup_level = 'text_mapped' THEN 1 ELSE 0 END) = 1
            THEN 'mapped_by_curated_text'
        ELSE 'unmapped_text_concept'
    END AS notes
FROM {parquet_scan(path)}
WHERE source = 'eicu_crd'
    AND {text_key} IS NOT NULL
GROUP BY {text_key}
ORDER BY row_count DESC, diagnosisstring_normalized
"""


def unit_summary_query(
    path: Path,
    *,
    domain: str,
    token_column: str,
    value_column: str,
) -> str:
    """Build aggregate unit-availability and compatibility query."""

    return f"""
WITH source_summary AS (
    SELECT
        source,
        COUNT(*) AS total_rows,
        SUM(CASE WHEN NULLIF(TRIM(CAST(unit AS VARCHAR)), '') IS NOT NULL THEN 1 ELSE 0 END) AS unit_available_rows,
        SUM(CASE WHEN normalized_unit IS NOT NULL THEN 1 ELSE 0 END) AS normalized_unit_available_rows,
        SUM(CASE WHEN normalized_unit IS NULL THEN 1 ELSE 0 END) AS unknown_unit_rows,
        SUM(CASE WHEN {value_column} IS NOT NULL THEN 1 ELSE 0 END) AS numeric_value_rows,
        COUNT(DISTINCT {token_column}) AS concept_count
    FROM {parquet_scan(path)}
    GROUP BY source
),
concept_units AS (
    SELECT
        source,
        {token_column} AS concept_token,
        COUNT(DISTINCT normalized_unit) AS normalized_unit_count
    FROM {parquet_scan(path)}
    WHERE {token_column} IS NOT NULL
        AND normalized_unit IS NOT NULL
    GROUP BY source, {token_column}
),
multi_unit AS (
    SELECT
        source,
        SUM(CASE WHEN normalized_unit_count > 1 THEN 1 ELSE 0 END) AS multi_unit_concept_count
    FROM concept_units
    GROUP BY source
)
SELECT
    {sql_string(domain)} AS domain,
    s.source,
    s.total_rows,
    s.unit_available_rows,
    s.normalized_unit_available_rows,
    s.unknown_unit_rows,
    s.numeric_value_rows,
    s.concept_count,
    COALESCE(m.multi_unit_concept_count, 0) AS multi_unit_concept_count,
    {sql_string(path)} AS artifact_path
FROM source_summary AS s
LEFT JOIN multi_unit AS m
    ON s.source = m.source
"""


def fetch_dict_rows(
    connection: duckdb.DuckDBPyConnection, query: str
) -> list[dict[str, Any]]:
    """Run a query and return dictionaries."""

    cursor = connection.execute(query)
    columns = [description[0] for description in cursor.description]
    return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]


def source_availability(config: HarmonizationBuildConfig) -> list[dict[str, Any]]:
    """Return source extract availability metadata without row content."""

    expected = {
        "demographics": {
            "mimiciv": ("patients.parquet", "admissions.parquet", "icustays.parquet"),
            "eicu_crd": ("patient.parquet",),
        },
        "conditions": {
            "mimiciv": ("diagnoses_icd.parquet",),
            "eicu_crd": ("diagnosis.parquet",),
        },
        "medications": {
            "mimiciv": ("prescriptions.parquet",),
            "eicu_crd": ("medication.parquet",),
        },
        "labs": {
            "mimiciv": ("labevents.parquet", "d_labitems.parquet"),
            "eicu_crd": ("lab.parquet",),
        },
        "vitals": {
            "mimiciv": ("chartevents.parquet",),
            "eicu_crd": ("vital_periodic.parquet", "vital_aperiodic.parquet"),
        },
        "allergies": {
            "mimiciv": (),
            "eicu_crd": ("allergy.parquet",),
        },
        "interventions": {
            "mimiciv": ("procedures_icd.parquet", "procedureevents.parquet"),
            "eicu_crd": (
                "treatment.parquet",
                "infusion_drug.parquet",
                "apache_aps_var.parquet",
            ),
        },
    }
    rows: list[dict[str, Any]] = []
    for domain, sources in expected.items():
        for source, names in sources.items():
            paths = [source_extract_path(config, source, name) for name in names]
            rows.append(
                {
                    "domain": domain,
                    "source": source,
                    "configured_extracts": [path.name for path in paths],
                    "available_extract_count": sum(path.exists() for path in paths),
                    "expected_extract_count": len(paths),
                    "status": (
                        "not_configured_for_source"
                        if not paths
                        else "available"
                        if all(path.exists() for path in paths)
                        else "partially_available"
                        if any(path.exists() for path in paths)
                        else "unavailable"
                    ),
                }
            )
    return rows


def write_coverage_reports(
    config: HarmonizationBuildConfig,
    *,
    table_records: Sequence[dict[str, Any]],
    generated_at: str,
) -> None:
    """Write aggregate coverage, unit compatibility, and unmapped reports."""

    artifact_by_domain = {
        record["table_name"]: Path(str(record["output_path"]))
        for record in table_records
        if record.get("status") == "completed"
    }
    coverage_rows: list[dict[str, Any]] = []
    unit_rows: list[dict[str, Any]] = []
    condition_rollup_rows: list[dict[str, Any]] = []

    with duckdb.connect(database=":memory:") as connection:
        configure_connection(config, connection)
        conditions_path = artifact_by_domain.get("conditions")
        if conditions_path is not None and conditions_path.exists():
            if table_has_column(connection, conditions_path, "condition_rollup_level"):
                condition_rollup_rows = fetch_dict_rows(
                    connection, condition_rollup_counts_query(conditions_path)
                )

        for domain, path in artifact_by_domain.items():
            if not path.exists() or not table_has_column(
                connection, path, "mapping_status"
            ):
                continue
            rows = fetch_dict_rows(
                connection, aggregate_counts_query(path, domain=domain)
            )
            if rows:
                coverage_rows.extend(rows)
            else:
                coverage_rows.append(
                    {
                        "domain": domain,
                        "source": "all",
                        "mapping_status": "no_rows",
                        "row_count": 0,
                        "artifact_path": str(path),
                    }
                )

        unit_specs = (
            ("labs", "normalized_lab_token", "lab_value_numeric"),
            ("vitals", "normalized_vital_token", "value_numeric"),
        )
        for domain, token_column, value_column in unit_specs:
            path = artifact_by_domain.get(domain)
            if path is None or not path.exists():
                continue
            rows = fetch_dict_rows(
                connection,
                unit_summary_query(
                    path,
                    domain=domain,
                    token_column=token_column,
                    value_column=value_column,
                ),
            )
            if rows:
                unit_rows.extend(rows)
            else:
                unit_rows.append(
                    {
                        "domain": domain,
                        "source": "all",
                        "total_rows": 0,
                        "unit_available_rows": 0,
                        "normalized_unit_available_rows": 0,
                        "unknown_unit_rows": 0,
                        "numeric_value_rows": 0,
                        "concept_count": 0,
                        "multi_unit_concept_count": 0,
                        "artifact_path": str(path),
                    }
                )

    artifact_summary = [
        {
            "domain": record["table_name"],
            "artifact_path": record["output_path"],
            "status": record["status"],
            "row_count": record["row_count"],
            "reason": record["reason"],
        }
        for record in table_records
    ]
    covered_domains = {row["domain"] for row in coverage_rows}
    for record in table_records:
        domain = str(record["table_name"])
        if domain not in covered_domains:
            coverage_rows.append(
                {
                    "domain": domain,
                    "source": "all",
                    "mapping_status": "not_applicable",
                    "row_count": int(record["row_count"] or 0),
                    "artifact_path": record["output_path"],
                }
            )

    coverage = {
        "schema_version": COVERAGE_SCHEMA_VERSION,
        "generated_at_utc": generated_at,
        "data_safety": {
            "contains_patient_rows": False,
            "reporting_level": (
                "aggregate row counts by domain/source/status and unit availability"
            ),
            "no_source_value_samples": True,
        },
        "versions": {
            "cohort_version": config.cohort_version,
            "harmonization_version": config.harmonization_version,
            "medication_mapping_version": MEDICATION_MAPPING_VERSION,
            "condition_mapping_version": CONDITION_MAPPING_VERSION,
            "source_native_mapping_version": SOURCE_NATIVE_MAPPING_VERSION,
        },
        "artifacts": artifact_summary,
        "coverage": coverage_rows,
        "condition_rollup_coverage": condition_rollup_rows,
        "unit_compatibility": unit_rows,
        "source_availability": source_availability(config),
    }
    write_json(config.coverage_path, coverage)

    unmapped = {
        "schema_version": UNMAPPED_SCHEMA_VERSION,
        "generated_at_utc": generated_at,
        "data_safety": {
            "contains_patient_rows": False,
            "reporting_level": "aggregate unmapped counts only",
            "no_source_value_samples": True,
        },
        "unmapped": [
            row
            for row in coverage_rows
            if str(row.get("mapping_status", "")).startswith("unmapped")
        ],
    }
    write_json(config.unmapped_path, unmapped)


def _percentage(numerator: Any, denominator: Any) -> float:
    """Return a rounded percentage with zero-denominator safety."""

    numerator = int(numerator or 0)
    denominator = int(denominator or 0)
    if not denominator:
        return 0.0
    return round(numerator * 100.0 / denominator, 2)


def write_condition_reports(
    config: HarmonizationBuildConfig,
    *,
    table_records: Sequence[dict[str, Any]],
    generated_at: str,
) -> None:
    """Write the dedicated condition normalization coverage and text review.

    Both artifacts are aggregate-only: per-source counts and concept-level
    diagnosis-string counts, never patient rows or identifiers.
    """

    conditions_path: Path | None = None
    for record in table_records:
        if (
            record.get("table_name") == "conditions"
            and record.get("status") == "completed"
        ):
            conditions_path = Path(str(record["output_path"]))

    available = available_condition_mappings(mapping_root=config.mapping_root)
    summary_rows: list[dict[str, Any]] = []
    rollup_rows: list[dict[str, Any]] = []
    text_review_rows: list[dict[str, Any]] = []

    if conditions_path is not None and conditions_path.exists():
        with duckdb.connect(database=":memory:") as connection:
            configure_connection(config, connection)
            if table_has_column(
                connection, conditions_path, "normalized_condition_token"
            ):
                summary_rows = fetch_dict_rows(
                    connection, condition_summary_query(conditions_path)
                )
                rollup_rows = fetch_dict_rows(
                    connection, condition_rollup_counts_query(conditions_path)
                )
                if table_has_column(
                    connection, conditions_path, "source_condition_text"
                ):
                    text_review_rows = fetch_dict_rows(
                        connection, eicu_text_review_query(conditions_path)
                    )

    for row in summary_rows:
        total = row.get("total_rows", 0)
        row["rollup_mapped_percent"] = _percentage(row.get("rollup_mapped_rows"), total)
        row["ccs_ccsr_percent"] = _percentage(row.get("ccs_ccsr_rows"), total)
        row["unmapped_percent"] = _percentage(row.get("unmapped_rows"), total)
        row["project_group_percent"] = _percentage(row.get("project_group_rows"), total)

    report = {
        "schema_version": CONDITION_COVERAGE_SCHEMA_VERSION,
        "generated_at_utc": generated_at,
        "data_safety": {
            "contains_patient_rows": False,
            "reporting_level": (
                "aggregate condition roll-up counts and concept-level "
                "diagnosis-string counts only"
            ),
            "no_source_value_samples": True,
        },
        "versions": {
            "cohort_version": config.cohort_version,
            "harmonization_version": config.harmonization_version,
            "condition_mapping_version": CONDITION_MAPPING_VERSION,
            "source_native_mapping_version": SOURCE_NATIVE_MAPPING_VERSION,
        },
        "mapping_files_used": sorted(available),
        "condition_mapping_resources": condition_mapping_resource_statuses(
            mapping_root=config.mapping_root
        ),
        "per_source_summary": summary_rows,
        "rollup_breakdown": rollup_rows,
        "acceptance_gates": {
            "mimic_rollup_coverage_target_percent": 95.0,
            "eicu_rollup_coverage_target_percent": 85.0,
            "note": (
                "Roll-up coverage counts rows with any non-null "
                "normalized_condition_token, including structural ICD "
                "categories. Review the ccs_ccsr_percent for authoritative "
                "roll-up depth before pooled MIMIC+eICU training."
            ),
        },
    }
    write_json(config.condition_coverage_path, report)

    _write_text_review_csv(config.text_review_path, text_review_rows)


def _write_text_review_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Write the aggregate eICU diagnosis-string mapping review CSV."""

    columns = [
        "diagnosisstring_normalized",
        "row_count",
        "mapped_condition_rollup_token",
        "mapped_condition_name",
        "mapping_source",
        "mapping_confidence",
        "notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    column: ("" if row.get(column) is None else row[column])
                    for column in columns
                }
            )


def temporal_event_queries(config: HarmonizationBuildConfig) -> list[str]:
    """Build temporal event references from harmonized domain artifacts."""

    queries: list[str] = []

    conditions = harmonized_path(config, "conditions.parquet")
    if conditions.exists():
        queries.append(
            f"""
SELECT
    source,
    source_version,
    patient_uid,
    encounter_uid,
    stay_uid,
    source_patient_id,
    source_encounter_id,
    source_stay_id,
    'condition' AS event_type,
    'conditions' AS source_domain,
    source_table,
    source_sequence AS source_event_id,
    CAST(NULL AS VARCHAR) AS event_start_time,
    CAST(NULL AS VARCHAR) AS event_end_time,
    CAST(NULL AS VARCHAR) AS event_start_offset,
    CAST(NULL AS VARCHAR) AS event_end_offset,
    condition_token AS event_token,
    condition_code AS source_code,
    condition_text AS source_text,
    CAST(NULL AS DOUBLE) AS value_numeric,
    CAST(NULL AS VARCHAR) AS value_text,
    CAST(NULL AS VARCHAR) AS unit,
    CAST(NULL AS VARCHAR) AS normalized_unit,
    mapping_status,
    cohort_version,
    extraction_version,
    mapping_version,
    harmonization_version,
    generated_at
FROM {parquet_scan(conditions)}
"""
        )

    medications = harmonized_path(config, "medications.parquet")
    if medications.exists():
        queries.append(
            f"""
SELECT
    source,
    source_version,
    patient_uid,
    encounter_uid,
    stay_uid,
    source_patient_id,
    source_encounter_id,
    source_stay_id,
    'medication' AS event_type,
    'medications' AS source_domain,
    source_table,
    source_event_id,
    event_start_time,
    event_end_time,
    CAST(NULL AS VARCHAR) AS event_start_offset,
    CAST(NULL AS VARCHAR) AS event_end_offset,
    COALESCE(ingredient_name, rxnorm_name, atc_code, medication_source_name) AS event_token,
    source_code,
    medication_source_name AS source_text,
    CAST(NULL AS DOUBLE) AS value_numeric,
    dose_value AS value_text,
    dose_unit AS unit,
    {normalized_unit_sql("dose_unit")} AS normalized_unit,
    mapping_status,
    cohort_version,
    extraction_version,
    mapping_version,
    harmonization_version,
    generated_at
FROM {parquet_scan(medications)}
"""
        )

    labs = harmonized_path(config, "labs.parquet")
    if labs.exists():
        queries.append(
            f"""
SELECT
    source,
    source_version,
    patient_uid,
    encounter_uid,
    stay_uid,
    source_patient_id,
    source_encounter_id,
    source_stay_id,
    'lab' AS event_type,
    'labs' AS source_domain,
    source_table,
    source_event_id,
    event_time AS event_start_time,
    CAST(NULL AS VARCHAR) AS event_end_time,
    event_time_offset AS event_start_offset,
    CAST(NULL AS VARCHAR) AS event_end_offset,
    normalized_lab_token AS event_token,
    source_lab_code AS source_code,
    source_lab_name AS source_text,
    lab_value_numeric AS value_numeric,
    lab_value_text AS value_text,
    unit,
    normalized_unit,
    mapping_status,
    cohort_version,
    extraction_version,
    mapping_version,
    harmonization_version,
    generated_at
FROM {parquet_scan(labs)}
"""
        )

    vitals = harmonized_path(config, "vitals.parquet")
    if vitals.exists():
        queries.append(
            f"""
SELECT
    source,
    source_version,
    patient_uid,
    encounter_uid,
    stay_uid,
    source_patient_id,
    source_encounter_id,
    source_stay_id,
    'vital' AS event_type,
    'vitals' AS source_domain,
    source_table,
    source_event_id,
    event_time AS event_start_time,
    CAST(NULL AS VARCHAR) AS event_end_time,
    event_time_offset AS event_start_offset,
    CAST(NULL AS VARCHAR) AS event_end_offset,
    normalized_vital_token AS event_token,
    source_vital_code AS source_code,
    source_vital_name AS source_text,
    value_numeric,
    CAST(NULL AS VARCHAR) AS value_text,
    unit,
    normalized_unit,
    mapping_status,
    cohort_version,
    extraction_version,
    mapping_version,
    harmonization_version,
    generated_at
FROM {parquet_scan(vitals)}
"""
        )

    allergies = harmonized_path(config, "allergies.parquet")
    if allergies.exists():
        queries.append(
            f"""
SELECT
    source,
    source_version,
    patient_uid,
    encounter_uid,
    stay_uid,
    source_patient_id,
    source_encounter_id,
    source_stay_id,
    'allergy' AS event_type,
    'allergies' AS source_domain,
    source_table,
    source_event_id,
    event_time AS event_start_time,
    CAST(NULL AS VARCHAR) AS event_end_time,
    event_entered_time AS event_start_offset,
    CAST(NULL AS VARCHAR) AS event_end_offset,
    normalized_allergen_token AS event_token,
    allergen_source_code AS source_code,
    allergen_source_name AS source_text,
    CAST(NULL AS DOUBLE) AS value_numeric,
    allergy_type AS value_text,
    CAST(NULL AS VARCHAR) AS unit,
    CAST(NULL AS VARCHAR) AS normalized_unit,
    mapping_status,
    cohort_version,
    extraction_version,
    mapping_version,
    harmonization_version,
    generated_at
FROM {parquet_scan(allergies)}
"""
        )

    interventions = harmonized_path(config, "interventions.parquet")
    if interventions.exists():
        queries.append(
            f"""
SELECT
    source,
    source_version,
    patient_uid,
    encounter_uid,
    stay_uid,
    source_patient_id,
    source_encounter_id,
    source_stay_id,
    'intervention' AS event_type,
    'interventions' AS source_domain,
    source_table,
    source_event_id,
    event_start_time,
    event_end_time,
    event_start_offset,
    event_end_offset,
    normalized_intervention_token AS event_token,
    source_intervention_code AS source_code,
    source_intervention_text AS source_text,
    value_numeric,
    value_text,
    unit,
    normalized_unit,
    mapping_status,
    cohort_version,
    extraction_version,
    mapping_version,
    harmonization_version,
    generated_at
FROM {parquet_scan(interventions)}
"""
        )

    return queries


def build_harmonized_artifacts(
    config: HarmonizationBuildConfig = HarmonizationBuildConfig(),
) -> dict[str, Any]:
    """Build source-tagged harmonized artifacts from extracted Parquet tables."""

    generated_at = datetime.now(UTC).isoformat()
    config.harmonized_root.mkdir(parents=True, exist_ok=True)
    mapping_ready, mapping_resources = mapping_resources_ready(
        mapping_root=config.mapping_root
    )
    condition_mapping_resources = condition_mapping_resource_statuses(
        mapping_root=config.mapping_root
    )
    if not mapping_ready:
        manifest = base_manifest(
            config,
            status="failed_missing_mapping_resources",
            mapping_resources=mapping_resources,
            generated_at=generated_at,
            condition_mapping_resources=condition_mapping_resources,
        )
        manifest["failure_reason"] = (
            "RxNorm/ATC medication mapping resources are required before "
            "medication harmonization"
        )
        manifest["coverage_path"] = str(config.coverage_path)
        manifest["unmapped_path"] = str(config.unmapped_path)
        write_json(config.manifest_path, manifest)
        return manifest

    manifest = base_manifest(
        config,
        status="completed",
        mapping_resources=mapping_resources,
        generated_at=generated_at,
        condition_mapping_resources=condition_mapping_resources,
    )
    tables: list[dict[str, Any]] = []

    with duckdb.connect(database=":memory:") as connection:
        configure_connection(config, connection)

        cohort_output = harmonized_path(config, "cohort_stays.parquet")
        cohort_record = materialize_table(
            connection,
            table_name="cohort_stays",
            query=cohort_query(config, generated_at=generated_at),
            output_path=cohort_output,
        )
        tables.append(cohort_record)
        if cohort_record["status"] == "completed":
            manifest["artifacts"]["cohort_stays"] = str(cohort_output)

        demographics_output = harmonized_path(config, "demographics.parquet")
        demographics_record = materialize_table(
            connection,
            table_name="demographics",
            query=demographics_query(config, generated_at=generated_at),
            output_path=demographics_output,
        )
        tables.append(demographics_record)
        if demographics_record["status"] == "completed":
            manifest["artifacts"]["demographics"] = str(demographics_output)

        domain_specs = (
            (
                "conditions",
                condition_queries(config, generated_at=generated_at),
                CONDITIONS_SCHEMA,
            ),
            (
                "medications",
                medication_queries(config, generated_at=generated_at),
                MEDICATIONS_SCHEMA,
            ),
            ("labs", lab_queries(config, generated_at=generated_at), LABS_SCHEMA),
            ("vitals", vital_queries(config, generated_at=generated_at), VITALS_SCHEMA),
            (
                "allergies",
                allergy_queries(config, generated_at=generated_at),
                ALLERGIES_SCHEMA,
            ),
            (
                "interventions",
                intervention_queries(config, generated_at=generated_at),
                INTERVENTIONS_SCHEMA,
            ),
        )
        for table_name, queries, schema in domain_specs:
            output = harmonized_path(config, f"{table_name}.parquet")
            record = materialize_domain_table(
                connection,
                table_name=table_name,
                queries=queries,
                output_path=output,
                empty_schema=schema,
            )
            tables.append(record)
            if record["status"] == "completed":
                manifest["artifacts"][table_name] = str(output)

        temporal_output = harmonized_path(config, "temporal_events.parquet")
        temporal_record = materialize_domain_table(
            connection,
            table_name="temporal_events",
            queries=temporal_event_queries(config),
            output_path=temporal_output,
            empty_schema=TEMPORAL_EVENTS_SCHEMA,
        )
        tables.append(temporal_record)
        if temporal_record["status"] == "completed":
            manifest["artifacts"]["temporal_events"] = str(temporal_output)

    if any(table["status"] == "failed" for table in tables):
        manifest["status"] = "failed"

    manifest["tables"] = tables
    manifest["coverage_path"] = str(config.coverage_path)
    manifest["unmapped_path"] = str(config.unmapped_path)
    manifest["condition_coverage_path"] = str(config.condition_coverage_path)
    manifest["text_review_path"] = str(config.text_review_path)
    write_json(config.manifest_path, manifest)
    write_coverage_reports(
        config,
        table_records=tables,
        generated_at=generated_at,
    )
    write_condition_reports(
        config,
        table_records=tables,
        generated_at=generated_at,
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build source-tagged harmonized artifacts from local extracts.",
    )
    parser.add_argument(
        "--cohort-path",
        type=Path,
        default=DEFAULT_COHORT_PATH,
        help="Unified cohort Parquet artifact from pipeline.cohort.",
    )
    parser.add_argument(
        "--extracts-root",
        type=Path,
        default=EXTRACTS_ROOT,
        help="Root directory containing source-specific extract folders.",
    )
    parser.add_argument(
        "--harmonized-root",
        type=Path,
        default=HARMONIZED_ROOT,
        help="Output directory for harmonized local Parquet artifacts.",
    )
    parser.add_argument(
        "--mapping-root",
        type=Path,
        default=MAPPING_ROOT,
        help="Root directory for local ignored mapping resources.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Output path for the harmonization manifest.",
    )
    parser.add_argument(
        "--coverage",
        type=Path,
        default=DEFAULT_COVERAGE_PATH,
        help="Output path for aggregate harmonization coverage.",
    )
    parser.add_argument(
        "--unmapped",
        type=Path,
        default=DEFAULT_UNMAPPED_PATH,
        help="Output path for aggregate unmapped concept counts.",
    )
    parser.add_argument(
        "--condition-coverage",
        type=Path,
        default=DEFAULT_CONDITION_COVERAGE_PATH,
        help="Output path for the dedicated condition normalization coverage.",
    )
    parser.add_argument(
        "--text-review",
        type=Path,
        default=DEFAULT_TEXT_REVIEW_PATH,
        help="Output path for the aggregate eICU diagnosis text mapping review.",
    )
    parser.add_argument(
        "--duckdb-temp-dir",
        type=Path,
        default=DUCKDB_TEMP_DIR,
        help=(
            "Directory DuckDB may use to spill larger-than-memory operators. "
            "Defaults to $DUCKDB_TEMP_DIR, then $TMPDIR, then the system temp dir."
        ),
    )
    parser.add_argument(
        "--duckdb-memory-limit",
        default=DUCKDB_MEMORY_LIMIT,
        help=(
            "Optional DuckDB memory ceiling (e.g. '24GB'). Bound this below the "
            "OAR cgroup allocation so DuckDB spills instead of being OOM-killed."
        ),
    )
    parser.add_argument(
        "--duckdb-threads",
        type=int,
        default=DUCKDB_THREADS,
        help="Optional DuckDB thread cap; fewer threads lower peak COPY memory.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_harmonized_artifacts(
        HarmonizationBuildConfig(
            cohort_path=args.cohort_path,
            extracts_root=args.extracts_root,
            harmonized_root=args.harmonized_root,
            mapping_root=args.mapping_root,
            manifest_path=args.manifest,
            coverage_path=args.coverage,
            unmapped_path=args.unmapped,
            condition_coverage_path=args.condition_coverage,
            text_review_path=args.text_review,
            duckdb_temp_directory=args.duckdb_temp_dir,
            duckdb_memory_limit=args.duckdb_memory_limit,
            duckdb_threads=args.duckdb_threads,
        )
    )
    print(
        "Wrote harmonization manifest: "
        f"status={manifest['status']}, "
        f"tables={len(manifest.get('tables', []))}"
    )
    if manifest["status"] == "failed_missing_mapping_resources":
        return 2
    return 1 if manifest["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
