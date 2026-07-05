"""Shared source-specific extraction helpers for data-foundation milestones."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import duckdb

from pipeline.config import (
    DATASET_ROOT,
    EXTRACTION_VERSION,
    REPORTS_ROOT,
)
from pipeline.io_utils import (
    DatasetPathError,
    inspect_header,
    quote_identifier,
    resolve_dataset_path,
)


SCHEMA_VERSION = "source-extraction-manifest-v1"
DEFAULT_QUALITY_PROFILE_PATH = REPORTS_ROOT / "quality_profile.json"
DEFAULT_INTEGRITY_REPORT_PATH = REPORTS_ROOT / "source_integrity_failed_tables.json"
LOCAL_PATIENT_ARTIFACT_MESSAGE = (
    "patient-level extracts are local ignored artifacts under Dataset/processed"
)


@dataclass(frozen=True)
class ExtractionTableSpec:
    """One cohort-filtered source table extraction contract."""

    table_name: str
    source: str
    source_version: str
    relative_path: Path
    output_name: str
    required_columns: tuple[str, ...]
    selected_columns: tuple[str, ...]
    join_columns: tuple[tuple[str, str], ...]
    profile_table_name: str
    requires_integrity_gate: bool = False
    lookup_table: bool = False
    notes: tuple[str, ...] = ()
    # Optional raw-source WHERE predicate applied before the cohort join, e.g. an
    # itemid allow-list for very large event tables. Referenced source columns
    # must be listed in ``required_columns``/``selected_columns`` so header
    # validation still covers them.
    source_row_filter: str | None = None


@dataclass(frozen=True)
class ExtractionBuildConfig:
    """Configuration for one source-specific extraction run."""

    source: str
    source_version: str
    dataset_root: Path
    cohort_path: Path
    output_root: Path
    manifest_path: Path
    quality_profile_path: Path = DEFAULT_QUALITY_PROFILE_PATH
    integrity_report_path: Path = DEFAULT_INTEGRITY_REPORT_PATH
    enforce_table_gates: bool = True
    extraction_version: str = EXTRACTION_VERSION


def sql_string(value: str | Path) -> str:
    """Return a SQL string literal."""

    return "'" + str(value).replace("'", "''") + "'"


def csv_scan(path: Path) -> str:
    """Return a DuckDB CSV scan expression that keeps source values as strings."""

    return f"read_csv_auto({sql_string(path)}, header = true, all_varchar = true)"


def parquet_scan(path: Path) -> str:
    """Return a DuckDB Parquet scan expression."""

    return f"read_parquet({sql_string(path)})"


def safe_error_message(error: Exception) -> str:
    """Return a short exception summary without source-row context."""

    message = str(error).splitlines()[0].strip()
    return (message or "extraction failed")[:240]


def load_json_if_present(path: Path) -> dict[str, Any] | None:
    """Load a JSON report if it exists."""

    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def table_status_from_quality(
    spec: ExtractionTableSpec,
    quality_report: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return the quality-profile gate status for a table."""

    if quality_report is None:
        return {
            "status": "missing_quality_profile",
            "ready": False,
            "quality_status": None,
        }
    for table in quality_report.get("tables", []):
        if table.get("table_name") == spec.profile_table_name:
            quality_status = table.get("status")
            return {
                "status": "ready" if quality_status == "completed" else "blocked",
                "ready": quality_status == "completed",
                "quality_status": quality_status,
            }
    return {
        "status": "missing_quality_profile_entry",
        "ready": False,
        "quality_status": None,
    }


def matching_integrity_result(
    spec: ExtractionTableSpec,
    integrity_report: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Find a byte-integrity result for the source table, if reported."""

    if integrity_report is None:
        return None
    relative_path = spec.relative_path.as_posix()
    for result in integrity_report.get("results", []):
        if (
            result.get("relative_path") == relative_path
            or result.get("table_name") == spec.profile_table_name
        ):
            return result
    return None


def integrity_status_from_report(
    spec: ExtractionTableSpec,
    integrity_report: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return the checksum/gzip gate status for a table."""

    result = matching_integrity_result(spec, integrity_report)
    if result is None:
        if spec.requires_integrity_gate:
            return {
                "status": "missing_integrity_result",
                "ready": False,
                "checksum_status": None,
                "gzip_status": None,
            }
        return {
            "status": "not_required",
            "ready": True,
            "checksum_status": None,
            "gzip_status": None,
        }

    checksum_status = result.get("checksum_status")
    gzip_status = (result.get("gzip_integrity") or {}).get("status")
    ready = checksum_status == "matched" and gzip_status in {"passed", "not_applicable"}
    return {
        "status": "ready" if ready else "blocked",
        "ready": ready,
        "checksum_status": checksum_status,
        "gzip_status": gzip_status,
    }


def evaluate_table_gate(
    spec: ExtractionTableSpec,
    *,
    quality_report: dict[str, Any] | None,
    integrity_report: dict[str, Any] | None,
    enforce_table_gates: bool,
) -> dict[str, Any]:
    """Evaluate quality and integrity gates for one extraction table."""

    if not enforce_table_gates:
        return {
            "ready": True,
            "status": "not_enforced",
            "quality": {"status": "not_enforced", "ready": True},
            "integrity": {"status": "not_enforced", "ready": True},
        }

    quality = table_status_from_quality(spec, quality_report)
    integrity = integrity_status_from_report(spec, integrity_report)
    ready = bool(quality["ready"] and integrity["ready"])
    return {
        "ready": ready,
        "status": "ready" if ready else "blocked",
        "quality": quality,
        "integrity": integrity,
    }


def validate_required_columns(
    spec: ExtractionTableSpec,
    *,
    dataset_root: Path = DATASET_ROOT,
) -> tuple[str, ...]:
    """Return missing required columns for a source file."""

    try:
        path = resolve_dataset_path(
            spec.relative_path,
            dataset_root=dataset_root,
            must_exist=True,
        )
    except DatasetPathError:
        return tuple(spec.required_columns)
    header = inspect_header(path)
    if header is None:
        return tuple(spec.required_columns)
    return tuple(column for column in spec.required_columns if column not in header)


def ordered_unique(values: Sequence[str]) -> tuple[str, ...]:
    """Return values in first-seen order without duplicates."""

    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            ordered.append(value)
            seen.add(value)
    return tuple(ordered)


def extraction_query(
    spec: ExtractionTableSpec,
    *,
    source_path: Path,
    cohort_path: Path,
    extraction_version: str,
) -> str:
    """Build the DuckDB SQL for one source extraction."""

    join_source_columns = tuple(source for source, _ in spec.join_columns)
    source_columns = ordered_unique(spec.selected_columns + join_source_columns)
    source_select = ", ".join(quote_identifier(column) for column in source_columns)
    selected_source_columns = ",\n        ".join(
        f"s.{quote_identifier(column)} AS {quote_identifier(column)}"
        for column in spec.selected_columns
    )
    source_where = (
        f"\n    WHERE {spec.source_row_filter}" if spec.source_row_filter else ""
    )
    constant_columns = f"""
        {sql_string(spec.source)} AS source,
        {sql_string(spec.source_version)} AS source_version,
        {sql_string(extraction_version)} AS extraction_version,
        {sql_string(spec.table_name)} AS extraction_table"""

    if spec.lookup_table:
        return f"""
WITH source_rows AS (
    SELECT {source_select}
    FROM {csv_scan(source_path)}{source_where}
)
SELECT
{constant_columns},
        {selected_source_columns}
FROM source_rows AS s
"""

    join_condition = " AND ".join(
        "NULLIF(TRIM(CAST(s.{source_column} AS VARCHAR)), '') = "
        "NULLIF(TRIM(CAST(c.{cohort_column} AS VARCHAR)), '')".format(
            source_column=quote_identifier(source_column),
            cohort_column=quote_identifier(cohort_column),
        )
        for source_column, cohort_column in spec.join_columns
    )
    return f"""
WITH cohort AS (
    SELECT *
    FROM {parquet_scan(cohort_path)}
    WHERE source = {sql_string(spec.source)}
),
source_rows AS (
    SELECT {source_select}
    FROM {csv_scan(source_path)}{source_where}
)
SELECT
        c.source,
        c.source_version,
        c.patient_uid,
        c.encounter_uid,
        c.stay_uid,
        c.source_patient_id,
        c.source_encounter_id,
        c.source_stay_id,
        {sql_string(extraction_version)} AS extraction_version,
        {sql_string(spec.table_name)} AS extraction_table,
        {selected_source_columns}
FROM source_rows AS s
INNER JOIN cohort AS c
    ON {join_condition}
"""


def configure_duckdb_connection(
    connection: duckdb.DuckDBPyConnection,
    *,
    temp_directory: Path | str | None = None,
    memory_limit: str | None = None,
    threads: int | None = None,
    preserve_insertion_order: bool = False,
) -> None:
    """Apply memory-safe runtime settings to a DuckDB connection.

    An in-memory DuckDB database keeps ``preserve_insertion_order`` on and has no
    spill directory by default, so a large ``COPY ... TO parquet`` over a wide
    ``UNION ALL`` (e.g. the eICU vital fan-out) buffers the whole ordered result
    in RAM and is OOM-killed by the OAR cgroup before DuckDB's own limit engages.
    Relaxing insertion order, enabling a spill ``temp_directory`` (DuckDB does not
    read the OS ``TMPDIR`` itself), and bounding memory/threads make the same
    query stream and offload to disk instead of dying.
    """

    connection.execute("PRAGMA enable_progress_bar=false")
    connection.execute(
        "SET preserve_insertion_order = "
        f"{'true' if preserve_insertion_order else 'false'}"
    )
    if temp_directory is not None:
        temp_path = Path(temp_directory)
        temp_path.mkdir(parents=True, exist_ok=True)
        connection.execute(f"SET temp_directory = {sql_string(temp_path)}")
    if memory_limit:
        connection.execute(f"SET memory_limit = {sql_string(memory_limit)}")
    if threads and threads > 0:
        connection.execute(f"SET threads = {int(threads)}")


def copy_query_to_parquet(
    connection: duckdb.DuckDBPyConnection,
    query: str,
    output_path: Path,
) -> None:
    """Materialize a query as a local Parquet artifact."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    connection.execute(f"COPY ({query}) TO {sql_string(output_path)} (FORMAT PARQUET)")


def parquet_row_count(connection: duckdb.DuckDBPyConnection, path: Path) -> int:
    """Count rows in a local Parquet artifact."""

    row = connection.execute(f"SELECT COUNT(*) FROM {parquet_scan(path)}").fetchone()
    return int(row[0]) if row is not None else 0


def source_artifact_path(
    config: ExtractionBuildConfig, spec: ExtractionTableSpec
) -> Path:
    """Return the output path for one extraction table."""

    return config.output_root / spec.output_name


def skipped_table_record(
    spec: ExtractionTableSpec,
    *,
    status: str,
    reason: str,
    output_path: Path,
    gate: dict[str, Any] | None = None,
    missing_columns: Sequence[str] = (),
) -> dict[str, Any]:
    """Build a manifest entry for a skipped extraction table."""

    return {
        "table_name": spec.table_name,
        "source": spec.source,
        "source_version": spec.source_version,
        "relative_path": spec.relative_path.as_posix(),
        "output_path": str(output_path),
        "status": status,
        "reason": reason,
        "row_count": None,
        "lookup_table": spec.lookup_table,
        "filtered_by_cohort_artifact": not spec.lookup_table,
        "missing_required_columns": list(missing_columns),
        "gate": gate,
        "notes": list(spec.notes),
    }


def build_source_extracts(
    config: ExtractionBuildConfig,
    table_specs: Sequence[ExtractionTableSpec],
) -> dict[str, Any]:
    """Build source-specific cohort-filtered extracts and an aggregate manifest."""

    config.output_root.mkdir(parents=True, exist_ok=True)
    config.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if not config.cohort_path.exists():
        raise DatasetPathError(f"Cohort artifact does not exist: {config.cohort_path}")

    quality_report = load_json_if_present(config.quality_profile_path)
    integrity_report = load_json_if_present(config.integrity_report_path)
    tables: list[dict[str, Any]] = []
    artifacts: dict[str, str] = {}

    with duckdb.connect(database=":memory:") as connection:
        connection.execute("PRAGMA enable_progress_bar=false")
        for spec in table_specs:
            output_path = source_artifact_path(config, spec)
            gate = evaluate_table_gate(
                spec,
                quality_report=quality_report,
                integrity_report=integrity_report,
                enforce_table_gates=config.enforce_table_gates,
            )
            if not gate["ready"]:
                tables.append(
                    skipped_table_record(
                        spec,
                        status="skipped_table_gate",
                        reason="quality_or_integrity_gate_not_ready",
                        output_path=output_path,
                        gate=gate,
                    )
                )
                continue

            missing_columns = validate_required_columns(
                spec,
                dataset_root=config.dataset_root,
            )
            if missing_columns:
                tables.append(
                    skipped_table_record(
                        spec,
                        status="skipped_missing_required_columns",
                        reason="source_header_missing_required_columns",
                        output_path=output_path,
                        gate=gate,
                        missing_columns=missing_columns,
                    )
                )
                continue

            try:
                source_path = resolve_dataset_path(
                    spec.relative_path,
                    dataset_root=config.dataset_root,
                )
                query = extraction_query(
                    spec,
                    source_path=source_path,
                    cohort_path=config.cohort_path,
                    extraction_version=config.extraction_version,
                )
                copy_query_to_parquet(connection, query, output_path)
                row_count = parquet_row_count(connection, output_path)
                artifacts[spec.table_name] = str(output_path)
                tables.append(
                    {
                        "table_name": spec.table_name,
                        "source": spec.source,
                        "source_version": spec.source_version,
                        "relative_path": spec.relative_path.as_posix(),
                        "output_path": str(output_path),
                        "status": "completed",
                        "reason": None,
                        "row_count": row_count,
                        "lookup_table": spec.lookup_table,
                        "filtered_by_cohort_artifact": not spec.lookup_table,
                        "missing_required_columns": [],
                        "gate": gate,
                        "notes": list(spec.notes),
                    }
                )
            except Exception as error:
                tables.append(
                    {
                        "table_name": spec.table_name,
                        "source": spec.source,
                        "source_version": spec.source_version,
                        "relative_path": spec.relative_path.as_posix(),
                        "output_path": str(output_path),
                        "status": "failed",
                        "reason": safe_error_message(error),
                        "row_count": None,
                        "lookup_table": spec.lookup_table,
                        "filtered_by_cohort_artifact": not spec.lookup_table,
                        "missing_required_columns": [],
                        "gate": gate,
                        "notes": list(spec.notes),
                    }
                )

    status_counts: dict[str, int] = {}
    for table in tables:
        status = str(table["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    completed_rows = sum(
        int(table["row_count"] or 0)
        for table in tables
        if table["status"] == "completed"
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source": config.source,
        "source_version": config.source_version,
        "data_safety": {
            "contains_patient_rows": False,
            "reporting_level": "aggregate extraction counts and statuses only",
            "identifier_artifacts_are_local_ignored": True,
            "patient_level_artifact_policy": LOCAL_PATIENT_ARTIFACT_MESSAGE,
        },
        "configuration": {
            "cohort_path": str(config.cohort_path),
            "output_root": str(config.output_root),
            "extraction_version": config.extraction_version,
            "quality_profile_path": str(config.quality_profile_path),
            "integrity_report_path": str(config.integrity_report_path),
            "enforce_table_gates": config.enforce_table_gates,
        },
        "summary": {
            "table_count": len(tables),
            "status_counts": status_counts,
            "completed_row_count": completed_rows,
        },
        "artifacts": artifacts,
        "tables": tables,
    }
    config.manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def manifest_has_failures(manifest: dict[str, Any]) -> bool:
    """Return whether a manifest includes attempted extraction failures."""

    return any(table.get("status") == "failed" for table in manifest.get("tables", []))
