"""Aggregate-only schema and data quality profiling for source tables."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Sequence

import duckdb

from pipeline.config import (
    DATASET_ROOT,
    REPORTS_ROOT,
    ensure_local_directories,
)
from pipeline.io_utils import (
    inspect_header,
    quote_identifier,
    requires_bounded_query,
    resolve_dataset_path,
)


SCHEMA_VERSION = "quality-profile-v1"
DEFAULT_OUTPUT_PATH = REPORTS_ROOT / "quality_profile.json"


@dataclass(frozen=True)
class NumericCheck:
    """Numeric parse and plausibility rule for one source column."""

    column: str
    minimum: float | None = None
    maximum: float | None = None


@dataclass(frozen=True)
class ReferentialCheck:
    """Aggregate foreign-key style check between source tables."""

    name: str
    child_columns: tuple[str, ...]
    parent_relative_path: Path
    parent_columns: tuple[str, ...]


@dataclass(frozen=True)
class TableProfileSpec:
    """Profiling configuration for one source table."""

    source: str
    source_version: str
    table_name: str
    relative_path: Path
    required_columns: tuple[str, ...]
    key_columns: tuple[str, ...] = ()
    profile_columns: tuple[str, ...] = ()
    numeric_checks: tuple[NumericCheck, ...] = ()
    timestamp_columns: tuple[str, ...] = ()
    categorical_columns: tuple[str, ...] = ()
    referential_checks: tuple[ReferentialCheck, ...] = ()
    notes: tuple[str, ...] = ()


def sql_string(value: str | Path) -> str:
    """Return a SQL string literal."""

    return "'" + str(value).replace("'", "''") + "'"


def csv_scan(path: Path) -> str:
    """Return a DuckDB CSV scan expression for aggregate profiling."""

    return f"read_csv_auto({sql_string(path)}, header = true, all_varchar = true)"


def nonblank_sql(column: str) -> str:
    """Return a SQL condition for non-null, non-empty string content."""

    quoted = quote_identifier(column)
    return f"{quoted} IS NOT NULL AND TRIM(CAST({quoted} AS VARCHAR)) <> ''"


def blank_sql(column: str) -> str:
    """Return a SQL condition for null or empty string content."""

    return f"NOT ({nonblank_sql(column)})"


def alias_for(prefix: str, column: str) -> str:
    safe = "".join(character if character.isalnum() else "_" for character in column)
    return f"{prefix}__{safe}"


def to_jsonable(value: Any) -> Any:
    """Convert DuckDB/Python scalar values to JSON-safe objects."""

    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: to_jsonable(nested_value) for key, nested_value in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(item) for item in value]
    return value


def fetch_single_row(
    connection: duckdb.DuckDBPyConnection, query: str
) -> dict[str, Any]:
    """Fetch one aggregate SQL row as a dictionary."""

    cursor = connection.execute(query)
    row = cursor.fetchone()
    if row is None:
        return {}
    column_names = [description[0] for description in cursor.description]
    return {
        column_name: to_jsonable(value)
        for column_name, value in zip(column_names, row, strict=True)
    }


def infer_column_types(
    connection: duckdb.DuckDBPyConnection,
    path: Path,
) -> list[dict[str, str]]:
    """Infer DuckDB column types without returning rows."""

    cursor = connection.execute(
        f"DESCRIBE SELECT * FROM read_csv_auto({sql_string(path)}, header = true)"
    )
    return [{"column": row[0], "duckdb_type": row[1]} for row in cursor.fetchall()]


def ordered_unique_columns(
    spec: TableProfileSpec, header: Sequence[str]
) -> tuple[str, ...]:
    """Return configured columns present in the table header, preserving order."""

    requested = (
        spec.key_columns
        + spec.profile_columns
        + spec.timestamp_columns
        + spec.categorical_columns
        + tuple(check.column for check in spec.numeric_checks)
    )
    seen: set[str] = set()
    available = set(header)
    columns: list[str] = []
    for column in requested:
        if column in available and column not in seen:
            columns.append(column)
            seen.add(column)
    return tuple(columns)


def aggregate_table_query(
    spec: TableProfileSpec, path: Path, header: Sequence[str]
) -> str:
    """Build a single-pass aggregate query for the configured profile columns."""

    select_parts = ["COUNT(*) AS row_count"]
    for column in ordered_unique_columns(spec, header):
        null_alias = alias_for("null_count", column)
        distinct_alias = alias_for("approx_distinct", column)
        select_parts.append(
            f"SUM(CASE WHEN {blank_sql(column)} THEN 1 ELSE 0 END) AS {quote_identifier(null_alias)}"
        )
        select_parts.append(
            "APPROX_COUNT_DISTINCT("
            f"CASE WHEN {nonblank_sql(column)} THEN TRIM(CAST({quote_identifier(column)} AS VARCHAR)) ELSE NULL END"
            f") AS {quote_identifier(distinct_alias)}"
        )
    for check in spec.numeric_checks:
        if check.column not in header:
            continue
        quoted = quote_identifier(check.column)
        parsed = f"TRY_CAST({quoted} AS DOUBLE)"
        prefix = alias_for("numeric", check.column)
        select_parts.extend(
            [
                f"SUM(CASE WHEN {nonblank_sql(check.column)} AND {parsed} IS NULL THEN 1 ELSE 0 END) AS {quote_identifier(prefix + '__parse_failure_count')}",
                f"SUM(CASE WHEN {parsed} IS NOT NULL THEN 1 ELSE 0 END) AS {quote_identifier(prefix + '__parseable_count')}",
                f"MIN({parsed}) AS {quote_identifier(prefix + '__min')}",
                f"MAX({parsed}) AS {quote_identifier(prefix + '__max')}",
                f"AVG({parsed}) AS {quote_identifier(prefix + '__mean')}",
            ]
        )
        bounds_conditions: list[str] = []
        if check.minimum is not None:
            bounds_conditions.append(f"{parsed} < {check.minimum}")
        if check.maximum is not None:
            bounds_conditions.append(f"{parsed} > {check.maximum}")
        if bounds_conditions:
            select_parts.append(
                f"SUM(CASE WHEN {parsed} IS NOT NULL AND ({' OR '.join(bounds_conditions)}) THEN 1 ELSE 0 END) AS {quote_identifier(prefix + '__out_of_bounds_count')}"
            )
    for column in spec.timestamp_columns:
        if column not in header:
            continue
        quoted = quote_identifier(column)
        parsed = f"TRY_CAST({quoted} AS TIMESTAMP)"
        prefix = alias_for("timestamp", column)
        select_parts.extend(
            [
                f"SUM(CASE WHEN {nonblank_sql(column)} AND {parsed} IS NULL THEN 1 ELSE 0 END) AS {quote_identifier(prefix + '__parse_failure_count')}",
                f"SUM(CASE WHEN {parsed} IS NOT NULL THEN 1 ELSE 0 END) AS {quote_identifier(prefix + '__parseable_count')}",
                f"MIN({parsed}) AS {quote_identifier(prefix + '__min')}",
                f"MAX({parsed}) AS {quote_identifier(prefix + '__max')}",
            ]
        )

    return f"SELECT {', '.join(select_parts)} FROM {csv_scan(path)}"


def profile_duplicate_keys(
    connection: duckdb.DuckDBPyConnection,
    spec: TableProfileSpec,
    path: Path,
    header: Sequence[str],
) -> dict[str, Any] | None:
    """Profile duplicate key groups and rows for the configured key."""

    if not spec.key_columns:
        return None
    missing = [column for column in spec.key_columns if column not in header]
    if missing:
        return {"status": "skipped_missing_key_columns", "missing_columns": missing}

    key_select = ", ".join(quote_identifier(column) for column in spec.key_columns)
    null_condition = " OR ".join(blank_sql(column) for column in spec.key_columns)
    query = f"""
WITH key_counts AS (
    SELECT {key_select}, COUNT(*) AS row_count
    FROM {csv_scan(path)}
    GROUP BY {key_select}
)
SELECT
    (SELECT COUNT(*) FROM {csv_scan(path)} WHERE {null_condition}) AS key_null_rows,
    COUNT(*) FILTER (WHERE row_count > 1) AS duplicate_key_groups,
    COALESCE(SUM(row_count) FILTER (WHERE row_count > 1), 0) AS duplicate_key_rows,
    COALESCE(SUM(row_count - 1) FILTER (WHERE row_count > 1), 0) AS duplicate_excess_rows
FROM key_counts
"""
    return fetch_single_row(connection, query)


def profile_referential_check(
    connection: duckdb.DuckDBPyConnection,
    check: ReferentialCheck,
    child_path: Path,
    child_header: Sequence[str],
    *,
    dataset_root: Path,
) -> dict[str, Any]:
    """Run one aggregate referential-integrity check."""

    parent_path = resolve_dataset_path(
        check.parent_relative_path,
        dataset_root=dataset_root,
    )
    parent_header = inspect_header(parent_path) or ()
    missing_child = [
        column for column in check.child_columns if column not in child_header
    ]
    missing_parent = [
        column for column in check.parent_columns if column not in parent_header
    ]
    if missing_child or missing_parent:
        return {
            "name": check.name,
            "status": "skipped_missing_columns",
            "missing_child_columns": missing_child,
            "missing_parent_columns": missing_parent,
        }

    child_present = " AND ".join(
        f"c.{nonblank_sql(column)}" for column in check.child_columns
    )
    parent_present = " AND ".join(
        f"{nonblank_sql(column)}" for column in check.parent_columns
    )
    join_condition = " AND ".join(
        f"c.{quote_identifier(child_column)} = p.{quote_identifier(parent_column)}"
        for child_column, parent_column in zip(
            check.child_columns,
            check.parent_columns,
            strict=True,
        )
    )
    parent_null = f"p.{quote_identifier(check.parent_columns[0])} IS NULL"
    parent_select = ", ".join(
        quote_identifier(column) for column in check.parent_columns
    )
    query = f"""
WITH parent_keys AS (
    SELECT DISTINCT {parent_select}
    FROM {csv_scan(parent_path)}
    WHERE {parent_present}
),
child_rows AS (
    SELECT *
    FROM {csv_scan(child_path)} AS c
    WHERE {child_present}
)
SELECT
    COUNT(*) AS checked_rows,
    SUM(CASE WHEN {parent_null} THEN 1 ELSE 0 END) AS orphan_rows
FROM child_rows AS c
LEFT JOIN parent_keys AS p
    ON {join_condition}
"""
    result = fetch_single_row(connection, query)
    result["name"] = check.name
    result["status"] = "completed"
    result["parent_relative_path"] = check.parent_relative_path.as_posix()
    return result


def parse_column_profiles(
    aggregate_row: dict[str, Any],
    spec: TableProfileSpec,
    header: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Extract per-column null and approximate cardinality metrics."""

    row_count = int(aggregate_row.get("row_count") or 0)
    profiles: dict[str, dict[str, Any]] = {}
    for column in ordered_unique_columns(spec, header):
        null_count = int(aggregate_row.get(alias_for("null_count", column)) or 0)
        approx_distinct = int(
            aggregate_row.get(alias_for("approx_distinct", column)) or 0
        )
        profiles[column] = {
            "null_count": null_count,
            "null_rate": null_count / row_count if row_count else None,
            "approx_distinct_count": approx_distinct,
        }
    return profiles


def parse_numeric_profiles(
    aggregate_row: dict[str, Any],
    spec: TableProfileSpec,
    header: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Extract numeric parse and plausibility metrics."""

    profiles: dict[str, dict[str, Any]] = {}
    for check in spec.numeric_checks:
        if check.column not in header:
            continue
        prefix = alias_for("numeric", check.column)
        profile = {
            "parse_failure_count": aggregate_row.get(
                prefix + "__parse_failure_count", 0
            ),
            "parseable_count": aggregate_row.get(prefix + "__parseable_count", 0),
            "min": aggregate_row.get(prefix + "__min"),
            "max": aggregate_row.get(prefix + "__max"),
            "mean": aggregate_row.get(prefix + "__mean"),
            "minimum_allowed": check.minimum,
            "maximum_allowed": check.maximum,
        }
        if check.minimum is not None or check.maximum is not None:
            profile["out_of_bounds_count"] = aggregate_row.get(
                prefix + "__out_of_bounds_count", 0
            )
        profiles[check.column] = profile
    return profiles


def parse_timestamp_profiles(
    aggregate_row: dict[str, Any],
    spec: TableProfileSpec,
    header: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Extract timestamp parse and coverage metrics."""

    profiles: dict[str, dict[str, Any]] = {}
    for column in spec.timestamp_columns:
        if column not in header:
            continue
        prefix = alias_for("timestamp", column)
        profiles[column] = {
            "parse_failure_count": aggregate_row.get(
                prefix + "__parse_failure_count", 0
            ),
            "parseable_count": aggregate_row.get(prefix + "__parseable_count", 0),
            "min": aggregate_row.get(prefix + "__min"),
            "max": aggregate_row.get(prefix + "__max"),
        }
    return profiles


def profile_table(
    connection: duckdb.DuckDBPyConnection,
    spec: TableProfileSpec,
    *,
    dataset_root: Path = DATASET_ROOT,
) -> dict[str, Any]:
    """Profile one configured source table with aggregate-only output."""

    path = resolve_dataset_path(spec.relative_path, dataset_root=dataset_root)
    header = inspect_header(path) or ()
    missing_required = [
        column for column in spec.required_columns if column not in header
    ]
    metadata = {
        "source": spec.source,
        "source_version": spec.source_version,
        "table_name": spec.table_name,
        "relative_path": spec.relative_path.as_posix(),
        "size_bytes": path.stat().st_size,
        "column_count": len(header),
        "requires_bounded_query": requires_bounded_query(
            spec.source,
            source_local_relative_path(spec),
            size_bytes=path.stat().st_size,
        ),
        "notes": list(spec.notes),
    }
    if missing_required:
        return {
            **metadata,
            "status": "skipped_missing_required_columns",
            "missing_required_columns": missing_required,
            "header_columns": list(header),
        }

    inferred_types = infer_column_types(connection, path)
    aggregate_row = fetch_single_row(
        connection,
        aggregate_table_query(spec, path, header),
    )
    duplicate_profile = profile_duplicate_keys(connection, spec, path, header)
    referential_profiles = [
        profile_referential_check(
            connection,
            check,
            path,
            header,
            dataset_root=dataset_root,
        )
        for check in spec.referential_checks
    ]

    return {
        **metadata,
        "status": "completed",
        "missing_required_columns": [],
        "row_count": aggregate_row.get("row_count", 0),
        "header_columns": list(header),
        "inferred_types": inferred_types,
        "key_columns": list(spec.key_columns),
        "column_profiles": parse_column_profiles(aggregate_row, spec, header),
        "numeric_profiles": parse_numeric_profiles(aggregate_row, spec, header),
        "timestamp_profiles": parse_timestamp_profiles(aggregate_row, spec, header),
        "duplicate_key_profile": duplicate_profile,
        "referential_integrity": referential_profiles,
    }


def source_local_relative_path(spec: TableProfileSpec) -> Path:
    """Return a path relative to the source-version root for large-table checks."""

    parts = spec.relative_path.parts
    if spec.source == "mimiciv" and len(parts) >= 3 and parts[:2] == ("mimiciv", "3.1"):
        return Path(*parts[2:])
    if (
        spec.source == "eicu_crd"
        and len(parts) >= 3
        and parts[:2] == ("eicu-crd", "2.0")
    ):
        return Path(*parts[2:])
    if spec.source == "mimiciv_note" and len(parts) >= 2 and parts[0] == "2.2":
        return Path(*parts[1:])
    return spec.relative_path


def failed_table_profile(
    spec: TableProfileSpec,
    error: Exception,
    *,
    dataset_root: Path,
) -> dict[str, Any]:
    """Create a safe failed-table report without raising."""

    path = resolve_dataset_path(
        spec.relative_path,
        dataset_root=dataset_root,
        must_exist=False,
    )
    size_bytes = path.stat().st_size if path.exists() else None
    try:
        header = list(inspect_header(path) or ()) if path.exists() else []
    except Exception:
        header = []
    return {
        "source": spec.source,
        "source_version": spec.source_version,
        "table_name": spec.table_name,
        "relative_path": spec.relative_path.as_posix(),
        "size_bytes": size_bytes,
        "column_count": len(header),
        "requires_bounded_query": (
            requires_bounded_query(
                spec.source,
                source_local_relative_path(spec),
                size_bytes=size_bytes or 0,
            )
            if size_bytes is not None
            else None
        ),
        "notes": list(spec.notes),
        "status": "scan_failed",
        "error_type": type(error).__name__,
        "error_message": safe_error_message(error),
        "header_columns": header,
    }


def safe_error_message(error: Exception) -> str:
    """Return a parser-error summary without source line content."""

    message = str(error).splitlines()[0].strip()
    if not message:
        return "scan failed"
    return message[:240]


def profile_quality(
    specs: Sequence[TableProfileSpec],
    *,
    dataset_root: Path = DATASET_ROOT,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    """Build a quality profile report for configured source tables."""

    ensure_local_directories()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(database=":memory:") as connection:
        connection.execute("PRAGMA enable_progress_bar=false")
        tables = []
        for spec in specs:
            try:
                tables.append(
                    profile_table(connection, spec, dataset_root=dataset_root)
                )
            except Exception as error:
                tables.append(
                    failed_table_profile(
                        spec,
                        error,
                        dataset_root=dataset_root,
                    )
                )
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "data_safety": {
            "contains_patient_rows": False,
            "reporting_level": "aggregate table and column metrics only",
            "no_value_samples": True,
        },
        "profile_policy": {
            "csv_reader": "duckdb read_csv_auto(all_varchar=true) for aggregate scans",
            "note_text_tables": "not included in default full profiling; use source inventory for headers",
        },
        "table_count": len(tables),
        "tables": tables,
    }
    output_path.write_text(
        json.dumps(to_jsonable(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def ref(
    name: str, parent: Path, child_columns: Sequence[str], parent_columns: Sequence[str]
) -> ReferentialCheck:
    return ReferentialCheck(
        name=name,
        child_columns=tuple(child_columns),
        parent_relative_path=parent,
        parent_columns=tuple(parent_columns),
    )


def table(
    source: str,
    version: str,
    table_name: str,
    relative_path: Path,
    required: Sequence[str],
    *,
    key: Sequence[str] = (),
    profile: Sequence[str] = (),
    numeric: Sequence[NumericCheck] = (),
    timestamp: Sequence[str] = (),
    categorical: Sequence[str] = (),
    referential: Sequence[ReferentialCheck] = (),
    notes: Sequence[str] = (),
) -> TableProfileSpec:
    return TableProfileSpec(
        source=source,
        source_version=version,
        table_name=table_name,
        relative_path=relative_path,
        required_columns=tuple(required),
        key_columns=tuple(key),
        profile_columns=tuple(profile),
        numeric_checks=tuple(numeric),
        timestamp_columns=tuple(timestamp),
        categorical_columns=tuple(categorical),
        referential_checks=tuple(referential),
        notes=tuple(notes),
    )


MIMIC_PATIENTS = Path("mimiciv/3.1/hosp/patients.csv.gz")
MIMIC_ADMISSIONS = Path("mimiciv/3.1/hosp/admissions.csv.gz")
MIMIC_ICUSTAYS = Path("mimiciv/3.1/icu/icustays.csv.gz")
MIMIC_DIAGNOSES = Path("mimiciv/3.1/hosp/diagnoses_icd.csv.gz")
MIMIC_PROCEDURES = Path("mimiciv/3.1/hosp/procedures_icd.csv.gz")
MIMIC_PRESCRIPTIONS = Path("mimiciv/3.1/hosp/prescriptions.csv.gz")
MIMIC_LABEVENTS = Path("mimiciv/3.1/hosp/labevents.csv.gz")
MIMIC_D_LABITEMS = Path("mimiciv/3.1/hosp/d_labitems.csv.gz")
MIMIC_CHARTEVENTS = Path("mimiciv/3.1/icu/chartevents.csv.gz")
MIMIC_D_ITEMS = Path("mimiciv/3.1/icu/d_items.csv.gz")
MIMIC_INPUTEVENTS = Path("mimiciv/3.1/icu/inputevents.csv.gz")
MIMIC_PROCEDUREEVENTS = Path("mimiciv/3.1/icu/procedureevents.csv.gz")

EICU_PATIENT = Path("eicu-crd/2.0/patient.csv.gz")
EICU_DIAGNOSIS = Path("eicu-crd/2.0/diagnosis.csv.gz")
EICU_LAB = Path("eicu-crd/2.0/lab.csv.gz")
EICU_MEDICATION = Path("eicu-crd/2.0/medication.csv.gz")
EICU_INFUSION = Path("eicu-crd/2.0/infusionDrug.csv.gz")
EICU_ALLERGY = Path("eicu-crd/2.0/allergy.csv.gz")
EICU_TREATMENT = Path("eicu-crd/2.0/treatment.csv.gz")
EICU_VITAL_PERIODIC = Path("eicu-crd/2.0/vitalPeriodic.csv.gz")
EICU_VITAL_APERIODIC = Path("eicu-crd/2.0/vitalAperiodic.csv.gz")
EICU_APACHE_RESULT = Path("eicu-crd/2.0/apachePatientResult.csv.gz")
EICU_APACHE_APS = Path("eicu-crd/2.0/apacheApsVar.csv.gz")
EICU_APACHE_PRED = Path("eicu-crd/2.0/apachePredVar.csv.gz")


DEFAULT_PROFILE_SPECS: tuple[TableProfileSpec, ...] = (
    table(
        "mimiciv",
        "3.1",
        "mimic_patients",
        MIMIC_PATIENTS,
        ("subject_id", "gender", "anchor_age"),
        key=("subject_id",),
        profile=("subject_id", "gender", "anchor_year_group"),
        numeric=(NumericCheck("anchor_age", 0, 120), NumericCheck("anchor_year")),
        categorical=("gender", "anchor_year_group"),
    ),
    table(
        "mimiciv",
        "3.1",
        "mimic_admissions",
        MIMIC_ADMISSIONS,
        ("subject_id", "hadm_id", "admittime", "dischtime"),
        key=("hadm_id",),
        profile=("subject_id", "hadm_id", "insurance", "language"),
        timestamp=("admittime", "dischtime", "deathtime", "edregtime", "edouttime"),
        numeric=(NumericCheck("hospital_expire_flag", 0, 1),),
        categorical=(
            "admission_type",
            "admission_location",
            "discharge_location",
            "race",
        ),
        referential=(
            ref(
                "admissions_subject_to_patients",
                MIMIC_PATIENTS,
                ("subject_id",),
                ("subject_id",),
            ),
        ),
    ),
    table(
        "mimiciv",
        "3.1",
        "mimic_icustays",
        MIMIC_ICUSTAYS,
        ("subject_id", "hadm_id", "stay_id", "intime", "outtime"),
        key=("stay_id",),
        profile=("subject_id", "hadm_id", "stay_id"),
        timestamp=("intime", "outtime"),
        numeric=(NumericCheck("los", 0, 365),),
        categorical=("first_careunit", "last_careunit"),
        referential=(
            ref(
                "icustays_subject_to_patients",
                MIMIC_PATIENTS,
                ("subject_id",),
                ("subject_id",),
            ),
            ref(
                "icustays_admission_to_admissions",
                MIMIC_ADMISSIONS,
                ("subject_id", "hadm_id"),
                ("subject_id", "hadm_id"),
            ),
        ),
    ),
    table(
        "mimiciv",
        "3.1",
        "mimic_diagnoses_icd",
        MIMIC_DIAGNOSES,
        ("subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"),
        key=("subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"),
        profile=("subject_id", "hadm_id", "icd_code"),
        numeric=(NumericCheck("seq_num", 1, 200), NumericCheck("icd_version", 9, 10)),
        categorical=("icd_version",),
        referential=(
            ref(
                "diagnoses_admission_to_admissions",
                MIMIC_ADMISSIONS,
                ("subject_id", "hadm_id"),
                ("subject_id", "hadm_id"),
            ),
        ),
    ),
    table(
        "mimiciv",
        "3.1",
        "mimic_procedures_icd",
        MIMIC_PROCEDURES,
        ("subject_id", "hadm_id", "seq_num", "chartdate", "icd_code", "icd_version"),
        key=("subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"),
        profile=("subject_id", "hadm_id", "icd_code"),
        timestamp=("chartdate",),
        numeric=(NumericCheck("seq_num", 1, 200), NumericCheck("icd_version", 9, 10)),
        categorical=("icd_version",),
        referential=(
            ref(
                "procedures_admission_to_admissions",
                MIMIC_ADMISSIONS,
                ("subject_id", "hadm_id"),
                ("subject_id", "hadm_id"),
            ),
        ),
    ),
    table(
        "mimiciv",
        "3.1",
        "mimic_prescriptions",
        MIMIC_PRESCRIPTIONS,
        ("subject_id", "hadm_id", "pharmacy_id", "drug", "starttime"),
        key=("pharmacy_id",),
        profile=("subject_id", "hadm_id", "pharmacy_id", "poe_id", "ndc", "drug"),
        timestamp=("starttime", "stoptime"),
        numeric=(NumericCheck("doses_per_24_hrs", 0, 48),),
        categorical=("drug_type", "dose_unit_rx", "route"),
        referential=(
            ref(
                "prescriptions_admission_to_admissions",
                MIMIC_ADMISSIONS,
                ("subject_id", "hadm_id"),
                ("subject_id", "hadm_id"),
            ),
        ),
        notes=(
            "Medication names are profiled only by null/cardinality counts, not values.",
        ),
    ),
    table(
        "mimiciv",
        "3.1",
        "mimic_labevents",
        MIMIC_LABEVENTS,
        ("labevent_id", "subject_id", "itemid", "charttime"),
        key=("labevent_id",),
        profile=("subject_id", "hadm_id", "itemid", "valueuom", "flag", "priority"),
        timestamp=("charttime", "storetime"),
        numeric=(
            NumericCheck("valuenum"),
            NumericCheck("ref_range_lower"),
            NumericCheck("ref_range_upper"),
        ),
        categorical=("valueuom", "flag", "priority"),
        referential=(
            ref(
                "labevents_item_to_d_labitems",
                MIMIC_D_LABITEMS,
                ("itemid",),
                ("itemid",),
            ),
            ref(
                "labevents_admission_to_admissions",
                MIMIC_ADMISSIONS,
                ("subject_id", "hadm_id"),
                ("subject_id", "hadm_id"),
            ),
        ),
    ),
    table(
        "mimiciv",
        "3.1",
        "mimic_d_labitems",
        MIMIC_D_LABITEMS,
        ("itemid", "label", "fluid", "category"),
        key=("itemid",),
        profile=("itemid", "label", "fluid", "category"),
        categorical=("fluid", "category"),
    ),
    table(
        "mimiciv",
        "3.1",
        "mimic_chartevents",
        MIMIC_CHARTEVENTS,
        ("subject_id", "stay_id", "charttime", "itemid"),
        profile=("subject_id", "hadm_id", "stay_id", "itemid", "valueuom", "warning"),
        timestamp=("charttime", "storetime"),
        numeric=(NumericCheck("valuenum"), NumericCheck("warning", 0, 1)),
        categorical=("valueuom", "warning"),
        referential=(
            ref(
                "chartevents_stay_to_icustays",
                MIMIC_ICUSTAYS,
                ("stay_id",),
                ("stay_id",),
            ),
            ref("chartevents_item_to_d_items", MIMIC_D_ITEMS, ("itemid",), ("itemid",)),
        ),
    ),
    table(
        "mimiciv",
        "3.1",
        "mimic_d_items",
        MIMIC_D_ITEMS,
        ("itemid", "label", "linksto", "category"),
        key=("itemid",),
        profile=("itemid", "linksto", "category", "unitname", "param_type"),
        numeric=(NumericCheck("lownormalvalue"), NumericCheck("highnormalvalue")),
        categorical=("linksto", "category", "unitname", "param_type"),
    ),
    table(
        "mimiciv",
        "3.1",
        "mimic_inputevents",
        MIMIC_INPUTEVENTS,
        ("subject_id", "stay_id", "starttime", "itemid", "orderid"),
        profile=("subject_id", "hadm_id", "stay_id", "itemid", "orderid"),
        timestamp=("starttime", "endtime", "storetime"),
        numeric=(
            NumericCheck("amount"),
            NumericCheck("rate"),
            NumericCheck("patientweight", 0, 500),
        ),
        categorical=("amountuom", "rateuom", "statusdescription", "ordercategoryname"),
        referential=(
            ref(
                "inputevents_stay_to_icustays",
                MIMIC_ICUSTAYS,
                ("stay_id",),
                ("stay_id",),
            ),
            ref("inputevents_item_to_d_items", MIMIC_D_ITEMS, ("itemid",), ("itemid",)),
        ),
    ),
    table(
        "mimiciv",
        "3.1",
        "mimic_procedureevents",
        MIMIC_PROCEDUREEVENTS,
        ("subject_id", "stay_id", "starttime", "itemid", "orderid"),
        profile=("subject_id", "hadm_id", "stay_id", "itemid", "orderid"),
        timestamp=("starttime", "endtime", "storetime"),
        numeric=(NumericCheck("value"), NumericCheck("patientweight", 0, 500)),
        categorical=(
            "valueuom",
            "locationcategory",
            "statusdescription",
            "ordercategoryname",
        ),
        referential=(
            ref(
                "procedureevents_stay_to_icustays",
                MIMIC_ICUSTAYS,
                ("stay_id",),
                ("stay_id",),
            ),
            ref(
                "procedureevents_item_to_d_items",
                MIMIC_D_ITEMS,
                ("itemid",),
                ("itemid",),
            ),
        ),
    ),
    table(
        "eicu_crd",
        "2.0",
        "eicu_patient",
        EICU_PATIENT,
        ("patientunitstayid", "uniquepid", "age", "gender"),
        key=("patientunitstayid",),
        profile=(
            "patientunitstayid",
            "patienthealthsystemstayid",
            "uniquepid",
            "hospitalid",
            "wardid",
        ),
        numeric=(
            NumericCheck("unitvisitnumber", 0, 100),
            NumericCheck("unitdischargeoffset"),
            NumericCheck("admissionheight", 0, 300),
            NumericCheck("admissionweight", 0, 500),
        ),
        categorical=(
            "gender",
            "age",
            "ethnicity",
            "unittype",
            "unitstaytype",
            "unitadmitsource",
        ),
        notes=(
            "Age may contain top-coded text such as '> 89'; cohort builder handles this separately.",
        ),
    ),
    table(
        "eicu_crd",
        "2.0",
        "eicu_diagnosis",
        EICU_DIAGNOSIS,
        ("diagnosisid", "patientunitstayid", "diagnosisoffset", "diagnosisstring"),
        key=("diagnosisid",),
        profile=("diagnosisid", "patientunitstayid", "icd9code"),
        numeric=(NumericCheck("diagnosisoffset"),),
        categorical=("activeupondischarge", "diagnosispriority"),
        referential=(
            ref(
                "diagnosis_stay_to_patient",
                EICU_PATIENT,
                ("patientunitstayid",),
                ("patientunitstayid",),
            ),
        ),
    ),
    table(
        "eicu_crd",
        "2.0",
        "eicu_lab",
        EICU_LAB,
        ("labid", "patientunitstayid", "labresultoffset", "labname"),
        key=("labid",),
        profile=(
            "labid",
            "patientunitstayid",
            "labtypeid",
            "labname",
            "labmeasurenamesystem",
            "labmeasurenameinterface",
        ),
        numeric=(
            NumericCheck("labresult"),
            NumericCheck("labresultoffset"),
            NumericCheck("labresultrevisedoffset"),
        ),
        categorical=("labname", "labmeasurenamesystem", "labmeasurenameinterface"),
        referential=(
            ref(
                "lab_stay_to_patient",
                EICU_PATIENT,
                ("patientunitstayid",),
                ("patientunitstayid",),
            ),
        ),
    ),
    table(
        "eicu_crd",
        "2.0",
        "eicu_medication",
        EICU_MEDICATION,
        ("medicationid", "patientunitstayid", "drugstartoffset", "drugname"),
        key=("medicationid",),
        profile=(
            "medicationid",
            "patientunitstayid",
            "drugname",
            "drughiclseqno",
            "gtc",
        ),
        numeric=(
            NumericCheck("drugorderoffset"),
            NumericCheck("drugstartoffset"),
            NumericCheck("drugstopoffset"),
            NumericCheck("gtc"),
        ),
        categorical=(
            "drugivadmixture",
            "drugordercancelled",
            "routeadmin",
            "frequency",
            "prn",
        ),
        referential=(
            ref(
                "medication_stay_to_patient",
                EICU_PATIENT,
                ("patientunitstayid",),
                ("patientunitstayid",),
            ),
        ),
        notes=(
            "Drug names and dosage text are profiled only by null/cardinality counts, not values.",
        ),
    ),
    table(
        "eicu_crd",
        "2.0",
        "eicu_infusion_drug",
        EICU_INFUSION,
        ("infusiondrugid", "patientunitstayid", "infusionoffset", "drugname"),
        key=("infusiondrugid",),
        profile=("infusiondrugid", "patientunitstayid", "drugname"),
        numeric=(
            NumericCheck("infusionoffset"),
            NumericCheck("infusionrate"),
            NumericCheck("drugamount"),
            NumericCheck("volumeoffluid"),
            NumericCheck("patientweight", 0, 500),
        ),
        referential=(
            ref(
                "infusion_stay_to_patient",
                EICU_PATIENT,
                ("patientunitstayid",),
                ("patientunitstayid",),
            ),
        ),
    ),
    table(
        "eicu_crd",
        "2.0",
        "eicu_allergy",
        EICU_ALLERGY,
        ("allergyid", "patientunitstayid", "allergyoffset"),
        key=("allergyid",),
        profile=("allergyid", "patientunitstayid", "drughiclseqno"),
        numeric=(NumericCheck("allergyoffset"), NumericCheck("allergyenteredoffset")),
        categorical=(
            "allergynotetype",
            "specialtytype",
            "usertype",
            "rxincluded",
            "writtenineicu",
            "allergytype",
        ),
        referential=(
            ref(
                "allergy_stay_to_patient",
                EICU_PATIENT,
                ("patientunitstayid",),
                ("patientunitstayid",),
            ),
        ),
    ),
    table(
        "eicu_crd",
        "2.0",
        "eicu_treatment",
        EICU_TREATMENT,
        ("treatmentid", "patientunitstayid", "treatmentoffset", "treatmentstring"),
        key=("treatmentid",),
        profile=("treatmentid", "patientunitstayid"),
        numeric=(NumericCheck("treatmentoffset"),),
        categorical=("activeupondischarge",),
        referential=(
            ref(
                "treatment_stay_to_patient",
                EICU_PATIENT,
                ("patientunitstayid",),
                ("patientunitstayid",),
            ),
        ),
    ),
    table(
        "eicu_crd",
        "2.0",
        "eicu_vital_periodic",
        EICU_VITAL_PERIODIC,
        ("vitalperiodicid", "patientunitstayid", "observationoffset"),
        key=("vitalperiodicid",),
        profile=("vitalperiodicid", "patientunitstayid"),
        numeric=(
            NumericCheck("observationoffset"),
            NumericCheck("temperature", 20, 45),
            NumericCheck("sao2", 0, 100),
            NumericCheck("heartrate", 0, 350),
            NumericCheck("respiration", 0, 120),
            NumericCheck("systemicmean", 0, 300),
        ),
        referential=(
            ref(
                "vital_periodic_stay_to_patient",
                EICU_PATIENT,
                ("patientunitstayid",),
                ("patientunitstayid",),
            ),
        ),
    ),
    table(
        "eicu_crd",
        "2.0",
        "eicu_vital_aperiodic",
        EICU_VITAL_APERIODIC,
        ("vitalaperiodicid", "patientunitstayid", "observationoffset"),
        key=("vitalaperiodicid",),
        profile=("vitalaperiodicid", "patientunitstayid"),
        numeric=(
            NumericCheck("observationoffset"),
            NumericCheck("noninvasivemean", 0, 300),
            NumericCheck("cardiacoutput"),
            NumericCheck("paop"),
        ),
        referential=(
            ref(
                "vital_aperiodic_stay_to_patient",
                EICU_PATIENT,
                ("patientunitstayid",),
                ("patientunitstayid",),
            ),
        ),
    ),
    table(
        "eicu_crd",
        "2.0",
        "eicu_apache_patient_result",
        EICU_APACHE_RESULT,
        ("apachepatientresultsid", "patientunitstayid", "apachescore"),
        key=("apachepatientresultsid",),
        profile=("apachepatientresultsid", "patientunitstayid", "apacheversion"),
        numeric=(
            NumericCheck("acutephysiologyscore"),
            NumericCheck("apachescore", 0, 300),
            NumericCheck("predictedicumortality", 0, 1),
            NumericCheck("predictediculos", 0, 365),
        ),
        categorical=("apacheversion", "actualicumortality", "actualhospitalmortality"),
        referential=(
            ref(
                "apache_result_stay_to_patient",
                EICU_PATIENT,
                ("patientunitstayid",),
                ("patientunitstayid",),
            ),
        ),
    ),
    table(
        "eicu_crd",
        "2.0",
        "eicu_apache_aps_var",
        EICU_APACHE_APS,
        ("apacheapsvarid", "patientunitstayid"),
        key=("apacheapsvarid",),
        profile=("apacheapsvarid", "patientunitstayid"),
        numeric=(
            NumericCheck("wbc"),
            NumericCheck("temperature", 20, 45),
            NumericCheck("sodium"),
            NumericCheck("heartrate", 0, 350),
            NumericCheck("meanbp", 0, 300),
            NumericCheck("creatinine"),
            NumericCheck("glucose"),
        ),
        categorical=("intubated", "vent", "dialysis"),
        referential=(
            ref(
                "apache_aps_stay_to_patient",
                EICU_PATIENT,
                ("patientunitstayid",),
                ("patientunitstayid",),
            ),
        ),
    ),
    table(
        "eicu_crd",
        "2.0",
        "eicu_apache_pred_var",
        EICU_APACHE_PRED,
        ("apachepredvarid", "patientunitstayid"),
        key=("apachepredvarid",),
        profile=("apachepredvarid", "patientunitstayid"),
        numeric=(
            NumericCheck("age", 0, 120),
            NumericCheck("creatinine"),
            NumericCheck("pao2"),
            NumericCheck("fio2"),
        ),
        categorical=(
            "gender",
            "region",
            "admitsource",
            "diedinhospital",
            "diabetes",
            "readmit",
        ),
        referential=(
            ref(
                "apache_pred_stay_to_patient",
                EICU_PATIENT,
                ("patientunitstayid",),
                ("patientunitstayid",),
            ),
        ),
    ),
)


def selected_specs(
    table_names: Sequence[str] | None = None,
) -> tuple[TableProfileSpec, ...]:
    """Return default specs, optionally filtered by table name."""

    if not table_names:
        return DEFAULT_PROFILE_SPECS
    requested = set(table_names)
    specs = tuple(
        spec for spec in DEFAULT_PROFILE_SPECS if spec.table_name in requested
    )
    missing = sorted(requested.difference(spec.table_name for spec in specs))
    if missing:
        raise ValueError(f"Unknown table profile names: {', '.join(missing)}")
    return specs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build aggregate-only schema and quality profiles for source tables.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DATASET_ROOT,
        help="Dataset root. Defaults to the repository Dataset directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output JSON report path. Defaults to reports/quality_profile.json.",
    )
    parser.add_argument(
        "--table",
        action="append",
        dest="tables",
        help="Profile one configured table name. May be passed multiple times.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = profile_quality(
        selected_specs(args.tables),
        dataset_root=args.dataset_root,
        output_path=args.output,
    )
    completed = sum(
        1 for table_report in report["tables"] if table_report["status"] == "completed"
    )
    print(
        f"Wrote aggregate quality profile for {completed}/{report['table_count']} tables to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
