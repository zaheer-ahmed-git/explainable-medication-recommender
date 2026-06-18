"""Safe file inspection helpers for licensed clinical source data."""

from __future__ import annotations

import csv
import gzip
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import duckdb

from pipeline.config import (
    DATASET_ROOT,
    DEFAULT_LARGE_TABLE_BYTES,
    LARGE_TABLES,
    SourceSpec,
)


class DatasetPathError(ValueError):
    """Raised when a requested dataset path is missing or outside Dataset/."""


@dataclass(frozen=True)
class FileMetadata:
    """Metadata-only inventory record for a source file."""

    source: str
    source_version: str
    relative_path: str
    size_bytes: int
    suffix: str
    compressed: bool
    header_columns: tuple[str, ...] | None
    requires_bounded_query: bool

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.header_columns is not None:
            payload["header_columns"] = list(self.header_columns)
        return payload


def resolve_dataset_path(
    relative_path: str | Path,
    *,
    dataset_root: Path = DATASET_ROOT,
    must_exist: bool = True,
) -> Path:
    """Resolve a path and ensure it stays under the configured Dataset root."""

    root = dataset_root.resolve()
    candidate = Path(relative_path)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root):
        raise DatasetPathError(f"Dataset path escapes root: {relative_path}")
    if must_exist and not resolved.exists():
        raise DatasetPathError(f"Dataset path does not exist: {resolved}")
    return resolved


def is_csv_like(path: Path) -> bool:
    """Return whether the path is a CSV file, including gzip-compressed CSV."""

    return path.name.endswith(".csv") or path.name.endswith(".csv.gz")


def inspect_header(path: Path) -> tuple[str, ...] | None:
    """Read only the header row for CSV/CSV.GZ files."""

    if not is_csv_like(path):
        return None

    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, mode="rt", encoding="utf-8", newline="") as file_obj:
        reader = csv.reader(file_obj)
        try:
            return tuple(next(reader))
        except StopIteration:
            return tuple()


def requires_bounded_query(
    source: str,
    relative_path: Path,
    *,
    size_bytes: int,
    large_table_bytes: int = DEFAULT_LARGE_TABLE_BYTES,
) -> bool:
    """Mark configured or large CSV-like tables as DuckDB/chunked-read only."""

    normalized_path = Path(relative_path)
    return (source, normalized_path) in LARGE_TABLES or size_bytes >= large_table_bytes


def source_root(source_spec: SourceSpec, *, dataset_root: Path = DATASET_ROOT) -> Path:
    """Return a source root under the provided dataset root."""

    return dataset_root / source_spec.root_relative_path


def inspect_file_metadata(
    source_spec: SourceSpec,
    file_path: Path,
    *,
    dataset_root: Path = DATASET_ROOT,
) -> FileMetadata:
    """Inspect safe metadata for a source file without reading data rows."""

    resolved = resolve_dataset_path(file_path, dataset_root=dataset_root)
    relative_to_source = resolved.relative_to(
        source_root(source_spec, dataset_root=dataset_root).resolve()
    )
    header_columns = inspect_header(resolved)
    size_bytes = resolved.stat().st_size
    return FileMetadata(
        source=source_spec.name,
        source_version=source_spec.version,
        relative_path=relative_to_source.as_posix(),
        size_bytes=size_bytes,
        suffix="".join(resolved.suffixes),
        compressed=resolved.name.endswith(".gz"),
        header_columns=header_columns,
        requires_bounded_query=requires_bounded_query(
            source_spec.name,
            relative_to_source,
            size_bytes=size_bytes,
        ),
    )


def iter_source_files(
    source_spec: SourceSpec,
    *,
    dataset_root: Path = DATASET_ROOT,
) -> Iterable[Path]:
    """Yield files under a source root in stable order."""

    root = source_root(source_spec, dataset_root=dataset_root)
    if not root.exists():
        return
    yield from sorted(path for path in root.rglob("*") if path.is_file())


def quote_identifier(identifier: str) -> str:
    """Quote a DuckDB identifier."""

    return '"' + identifier.replace('"', '""') + '"'


def read_csv_bounded(
    path: str | Path,
    *,
    columns: Iterable[str] | None = None,
    limit: int = 5,
    dataset_root: Path = DATASET_ROOT,
) -> list[dict[str, Any]]:
    """Run a bounded DuckDB CSV query for explicit, limited inspection.

    This helper is for controlled development checks. Callers must avoid logging
    returned rows from licensed clinical sources.
    """

    if limit <= 0:
        raise ValueError("limit must be positive for bounded CSV reads")

    resolved = resolve_dataset_path(path, dataset_root=dataset_root)
    select_clause = "*"
    if columns is not None:
        selected_columns = tuple(columns)
        if not selected_columns:
            raise ValueError("columns must contain at least one column")
        select_clause = ", ".join(
            quote_identifier(column) for column in selected_columns
        )

    escaped_path = str(resolved).replace("'", "''")
    query = (
        f"SELECT {select_clause} "
        f"FROM read_csv_auto('{escaped_path}', header = true) "
        f"LIMIT {int(limit)}"
    )
    with duckdb.connect(database=":memory:") as connection:
        cursor = connection.execute(query)
        column_names = [description[0] for description in cursor.description]
        return [dict(zip(column_names, row, strict=True)) for row in cursor.fetchall()]


def build_source_inventory(
    source_specs: Iterable[SourceSpec],
    *,
    dataset_root: Path = DATASET_ROOT,
) -> dict[str, Any]:
    """Build a metadata-only inventory for configured source datasets."""

    sources: list[dict[str, Any]] = []
    for source_spec in source_specs:
        root = source_root(source_spec, dataset_root=dataset_root)
        source_present = root.exists()
        expected_missing = [
            expected.as_posix()
            for expected in source_spec.expected_files
            if not (root / expected).exists()
        ]
        checksum_presence = {
            checksum.as_posix(): (root / checksum).exists()
            for checksum in source_spec.checksum_files
        }
        files = []
        if source_present:
            files = [
                inspect_file_metadata(
                    source_spec,
                    path,
                    dataset_root=dataset_root,
                ).to_json_dict()
                for path in iter_source_files(source_spec, dataset_root=dataset_root)
            ]
        sources.append(
            {
                "name": source_spec.name,
                "version": source_spec.version,
                "root_relative_path": source_spec.root_relative_path.as_posix(),
                "present": source_present,
                "expected_file_count": len(source_spec.expected_files),
                "missing_expected_files": expected_missing,
                "checksum_files_present": checksum_presence,
                "file_count": len(files),
                "files": files,
            }
        )

    return {
        "schema_version": "source-inventory-v1",
        "data_safety": {
            "contains_patient_rows": False,
            "inspection_level": "file metadata and CSV headers only",
        },
        "sources": sources,
    }
