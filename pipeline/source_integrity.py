"""Checksum and gzip integrity checks for source files."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from pipeline.config import (
    DATASET_ROOT,
    REPORTS_ROOT,
    SOURCE_SPECS,
    ensure_local_directories,
)
from pipeline.io_utils import resolve_dataset_path


SCHEMA_VERSION = "source-integrity-v1"
DEFAULT_OUTPUT_PATH = REPORTS_ROOT / "source_integrity_failed_tables.json"
FULL_MANIFEST_OUTPUT_PATH = REPORTS_ROOT / "source_integrity_all_manifest_files.json"
CHUNK_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class IntegrityTarget:
    """One source file that needs byte-level integrity checks."""

    source: str
    source_version: str
    table_name: str
    source_root_relative: Path
    file_relative: Path

    @property
    def dataset_relative_path(self) -> Path:
        return self.source_root_relative / self.file_relative


FAILED_PROFILE_TARGETS: tuple[IntegrityTarget, ...] = (
    IntegrityTarget(
        "mimiciv",
        "3.1",
        "mimic_prescriptions",
        Path("mimiciv") / "3.1",
        Path("hosp") / "prescriptions.csv.gz",
    ),
    IntegrityTarget(
        "mimiciv",
        "3.1",
        "mimic_labevents",
        Path("mimiciv") / "3.1",
        Path("hosp") / "labevents.csv.gz",
    ),
    IntegrityTarget(
        "mimiciv",
        "3.1",
        "mimic_chartevents",
        Path("mimiciv") / "3.1",
        Path("icu") / "chartevents.csv.gz",
    ),
    IntegrityTarget(
        "mimiciv",
        "3.1",
        "mimic_inputevents",
        Path("mimiciv") / "3.1",
        Path("icu") / "inputevents.csv.gz",
    ),
    IntegrityTarget(
        "eicu_crd",
        "2.0",
        "eicu_medication",
        Path("eicu-crd") / "2.0",
        Path("medication.csv.gz"),
    ),
    IntegrityTarget(
        "eicu_crd",
        "2.0",
        "eicu_apache_patient_result",
        Path("eicu-crd") / "2.0",
        Path("apachePatientResult.csv.gz"),
    ),
)


def safe_error_message(error: Exception) -> str:
    """Return an exception summary without any source content context."""

    message = str(error).splitlines()[0].strip()
    return (message or "integrity check failed")[:240]


def load_sha256_manifest(manifest_path: Path) -> dict[str, str]:
    """Parse a SHA256SUMS file into relative path -> expected digest."""

    expected: dict[str, str] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            expected[parts[-1].replace("\\", "/")] = parts[0].lower()
    return expected


def sha256_file(path: Path, *, chunk_bytes: int = CHUNK_BYTES) -> str:
    """Compute SHA-256 while streaming bytes."""

    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        while chunk := file_obj.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def gzip_integrity(path: Path, *, chunk_bytes: int = CHUNK_BYTES) -> dict[str, Any]:
    """Read a gzip file to EOF without retaining or printing decompressed data."""

    if not path.name.endswith(".gz"):
        return {
            "status": "not_applicable",
            "reason": "file is not gzip-compressed",
        }
    uncompressed_bytes = 0
    try:
        with gzip.open(path, mode="rb") as file_obj:
            while chunk := file_obj.read(chunk_bytes):
                uncompressed_bytes += len(chunk)
    except Exception as error:
        return {
            "status": "failed",
            "error_type": type(error).__name__,
            "error_message": safe_error_message(error),
            "uncompressed_bytes_before_error": uncompressed_bytes,
        }
    return {
        "status": "passed",
        "uncompressed_bytes": uncompressed_bytes,
    }


def target_from_manifest_entry(
    *,
    source: str,
    source_version: str,
    source_root_relative: Path,
    file_relative: str,
) -> IntegrityTarget:
    """Build an integrity target from one checksum manifest entry."""

    relative_path = Path(file_relative)
    table_name = (
        f"{source}_{relative_path.as_posix()}".replace("/", "_")
        .replace("\\", "_")
        .replace(".csv.gz", "")
        .replace(".csv", "")
        .replace(".txt", "")
    )
    return IntegrityTarget(
        source=source,
        source_version=source_version,
        table_name=table_name,
        source_root_relative=source_root_relative,
        file_relative=relative_path,
    )


def manifest_targets(
    *,
    dataset_root: Path = DATASET_ROOT,
    source_names: set[str] | None = None,
) -> tuple[IntegrityTarget, ...]:
    """Return targets for every file listed in each configured SHA256SUMS file."""

    targets: list[IntegrityTarget] = []
    for source_spec in SOURCE_SPECS:
        if source_names is not None and source_spec.name not in source_names:
            continue
        source_root = resolve_dataset_path(
            source_spec.root_relative_path, dataset_root=dataset_root
        )
        for checksum_file in source_spec.checksum_files:
            manifest_path = source_root / checksum_file
            if not manifest_path.exists():
                continue
            for file_relative in load_sha256_manifest(manifest_path):
                targets.append(
                    target_from_manifest_entry(
                        source=source_spec.name,
                        source_version=source_spec.version,
                        source_root_relative=source_spec.root_relative_path,
                        file_relative=file_relative,
                    )
                )
    return tuple(targets)


def check_target(
    target: IntegrityTarget, *, dataset_root: Path = DATASET_ROOT
) -> dict[str, Any]:
    """Run checksum and gzip checks for one target file."""

    file_path = resolve_dataset_path(
        target.dataset_relative_path, dataset_root=dataset_root, must_exist=False
    )
    source_root = resolve_dataset_path(
        target.source_root_relative, dataset_root=dataset_root
    )
    manifest_path = source_root / "SHA256SUMS.txt"
    expected_manifest = (
        load_sha256_manifest(manifest_path) if manifest_path.exists() else {}
    )
    manifest_key = target.file_relative.as_posix()
    expected_sha256 = expected_manifest.get(manifest_key)
    if not file_path.exists():
        alternate_local_file = find_alternate_configured_local_file(
            target, dataset_root=dataset_root
        )
        checksum_status = (
            "missing_manifest_file_but_configured_alternate_present"
            if alternate_local_file is not None
            else "missing_local_file"
        )
        return {
            "source": target.source,
            "source_version": target.source_version,
            "table_name": target.table_name,
            "relative_path": target.dataset_relative_path.as_posix(),
            "file_size_bytes": None,
            "expected_sha256": expected_sha256,
            "actual_sha256": None,
            "checksum_status": checksum_status,
            "gzip_integrity": {
                "status": "not_checked",
                "reason": "local file is missing",
            },
            "alternate_local_file": alternate_local_file,
        }
    actual_sha256 = sha256_file(file_path)
    if expected_sha256 is None:
        checksum_status = "missing_expected_checksum"
    elif actual_sha256 == expected_sha256:
        checksum_status = "matched"
    else:
        checksum_status = "mismatched"
    return {
        "source": target.source,
        "source_version": target.source_version,
        "table_name": target.table_name,
        "relative_path": target.dataset_relative_path.as_posix(),
        "file_size_bytes": file_path.stat().st_size,
        "expected_sha256": expected_sha256,
        "actual_sha256": actual_sha256,
        "checksum_status": checksum_status,
        "gzip_integrity": gzip_integrity(file_path),
    }


def find_alternate_configured_local_file(
    target: IntegrityTarget, *, dataset_root: Path = DATASET_ROOT
) -> dict[str, Any] | None:
    """Find configured local files that appear to be uncompressed variants."""

    if not target.file_relative.name.endswith(".csv.gz"):
        return None
    alternate_relative = target.file_relative.with_name(
        target.file_relative.name.removesuffix(".gz")
    )
    for source_spec in SOURCE_SPECS:
        if (
            source_spec.name == target.source
            and source_spec.root_relative_path == target.source_root_relative
            and alternate_relative in source_spec.expected_files
        ):
            alternate_path = resolve_dataset_path(
                target.source_root_relative / alternate_relative,
                dataset_root=dataset_root,
                must_exist=False,
            )
            if alternate_path.exists():
                return {
                    "relative_path": (
                        target.source_root_relative / alternate_relative
                    ).as_posix(),
                    "file_size_bytes": alternate_path.stat().st_size,
                    "reason": (
                        "configured source layout expects this table as an "
                        "uncompressed CSV rather than the manifest .csv.gz entry"
                    ),
                }
    return None


def summarize_results(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Build aggregate counts for integrity results."""

    checksum_counts: dict[str, int] = {}
    gzip_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    for result in results:
        checksum_status = result["checksum_status"]
        gzip_status = result["gzip_integrity"]["status"]
        source = result["source"]
        checksum_counts[checksum_status] = checksum_counts.get(checksum_status, 0) + 1
        gzip_counts[gzip_status] = gzip_counts.get(gzip_status, 0) + 1
        source_counts[source] = source_counts.get(source, 0) + 1
    return {
        "file_count": len(results),
        "source_counts": source_counts,
        "checksum_status_counts": checksum_counts,
        "gzip_status_counts": gzip_counts,
    }


def build_integrity_report(
    *,
    targets: Sequence[IntegrityTarget] = FAILED_PROFILE_TARGETS,
    dataset_root: Path = DATASET_ROOT,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    """Run integrity checks and write a safe aggregate/local report."""

    ensure_local_directories()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = [check_target(target, dataset_root=dataset_root) for target in targets]
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "data_safety": {
            "contains_patient_rows": False,
            "inspection_level": "file hashes and gzip stream integrity only",
            "no_decompressed_content_written": True,
        },
        "summary": summarize_results(results),
        "results": results,
        "recommended_interpretation": [
            "checksum mismatch means the local file differs from the manifest and should be re-transferred or re-downloaded",
            "gzip failure means the local compressed stream cannot be trusted for downstream features",
            "checksum match plus gzip pass shifts attention to CSV parser settings or source-specific malformed-row handling",
        ],
    }
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify checksum and gzip integrity for source files.",
    )
    parser.add_argument(
        "--all-manifest-files",
        action="store_true",
        help=(
            "Check every local file listed in configured SHA256SUMS manifests. "
            "Defaults to only the profiling-blocked files."
        ),
    )
    parser.add_argument(
        "--source",
        action="append",
        choices=tuple(source_spec.name for source_spec in SOURCE_SPECS),
        help="Limit --all-manifest-files to one source. May be provided multiple times.",
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
        default=None,
        help="Output JSON report path.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_path = args.output
    targets: Sequence[IntegrityTarget] = FAILED_PROFILE_TARGETS
    if args.all_manifest_files:
        targets = manifest_targets(
            dataset_root=args.dataset_root,
            source_names=set(args.source) if args.source else None,
        )
        output_path = output_path or FULL_MANIFEST_OUTPUT_PATH
    else:
        output_path = output_path or DEFAULT_OUTPUT_PATH
    report = build_integrity_report(
        targets=targets,
        dataset_root=args.dataset_root,
        output_path=output_path,
    )
    summary = report["summary"]
    print(
        "Wrote source integrity report: "
        f"{summary['file_count']} files, "
        f"checksum={summary['checksum_status_counts']}, "
        f"gzip={summary['gzip_status_counts']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
