"""Read aggregate metadata from local derived Parquet artifacts."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import duckdb

from pipeline.extract_utils import parquet_scan


def parquet_columns(
    connection: duckdb.DuckDBPyConnection,
    path: Path,
) -> tuple[str, ...]:
    """Return Parquet column names without reading patient rows."""

    cursor = connection.execute(f"DESCRIBE SELECT * FROM {parquet_scan(path)}")
    return tuple(str(row[0]) for row in cursor.fetchall())


def infer_consistent_version(
    paths: Sequence[Path],
    *,
    column_name: str,
    declared_version: str | None = None,
    fallback_version: str,
) -> str:
    """Infer one version stamp from input schemas and aggregate distinct values.

    Inputs without the requested column are tolerated for backward-compatible
    synthetic fixtures. Conflicting artifact values, or a conflicting explicit
    declaration, fail closed before downstream artifacts are materialized.
    """

    versions: set[str] = set()
    with duckdb.connect(database=":memory:") as connection:
        for path in paths:
            if not path.exists() or column_name not in parquet_columns(
                connection, path
            ):
                continue
            rows = connection.execute(
                f"""
SELECT DISTINCT TRIM(CAST({column_name} AS VARCHAR)) AS version_value
FROM {parquet_scan(path)}
WHERE {column_name} IS NOT NULL
    AND TRIM(CAST({column_name} AS VARCHAR)) <> ''
LIMIT 2
"""
            ).fetchall()
            versions.update(str(row[0]) for row in rows)

    if len(versions) > 1:
        joined = ", ".join(sorted(versions))
        raise ValueError(f"Conflicting {column_name} values across inputs: {joined}")

    inferred = next(iter(versions), None)
    if declared_version is not None and inferred not in {None, declared_version}:
        raise ValueError(
            f"Declared {column_name} {declared_version!r} does not match "
            f"input value {inferred!r}"
        )
    return declared_version or inferred or fallback_version
