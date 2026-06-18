import gzip
import hashlib
from pathlib import Path

from pipeline.config import SourceSpec
from pipeline.source_integrity import (
    IntegrityTarget,
    build_integrity_report,
    gzip_integrity,
    load_sha256_manifest,
    manifest_targets,
)


def write_gzip(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, mode="wb") as file_obj:
        file_obj.write(content)


def test_load_sha256_manifest_parses_paths(tmp_path: Path) -> None:
    manifest = tmp_path / "SHA256SUMS.txt"
    manifest.write_text("abc123  folder/file.csv.gz\n", encoding="utf-8")

    assert load_sha256_manifest(manifest) == {"folder/file.csv.gz": "abc123"}


def test_gzip_integrity_passes_and_fails_without_content(tmp_path: Path) -> None:
    good = tmp_path / "good.csv.gz"
    bad = tmp_path / "bad.csv.gz"
    write_gzip(good, b"col\nvalue\n")
    bad.write_bytes(b"not gzip")

    assert gzip_integrity(good)["status"] == "passed"
    assert gzip_integrity(tmp_path / "plain.csv")["status"] == "not_applicable"
    failed = gzip_integrity(bad)
    assert failed["status"] == "failed"
    assert "not gzip" not in failed["error_message"]


def test_build_integrity_report_detects_checksum_and_gzip_status(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "Dataset"
    source_root = dataset_root / "source" / "1.0"
    good_relative = Path("good.csv.gz")
    bad_relative = Path("bad.csv.gz")
    good_path = source_root / good_relative
    bad_path = source_root / bad_relative
    write_gzip(good_path, b"col\nvalue\n")
    bad_path.write_bytes(b"not gzip")
    good_hash = hashlib.sha256(good_path.read_bytes()).hexdigest()
    source_root.joinpath("SHA256SUMS.txt").write_text(
        f"{good_hash}  {good_relative.as_posix()}\n0000  {bad_relative.as_posix()}\n",
        encoding="utf-8",
    )

    report = build_integrity_report(
        targets=(
            IntegrityTarget(
                "synthetic",
                "1.0",
                "good",
                Path("source") / "1.0",
                good_relative,
            ),
            IntegrityTarget(
                "synthetic",
                "1.0",
                "bad",
                Path("source") / "1.0",
                bad_relative,
            ),
        ),
        dataset_root=dataset_root,
        output_path=tmp_path / "reports" / "integrity.json",
    )

    assert report["data_safety"]["contains_patient_rows"] is False
    assert report["summary"]["checksum_status_counts"] == {
        "matched": 1,
        "mismatched": 1,
    }
    assert report["summary"]["gzip_status_counts"] == {
        "passed": 1,
        "failed": 1,
    }


def test_build_integrity_report_supports_manifest_targets_and_plain_files(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_root = tmp_path / "Dataset"
    mimic_root = dataset_root / "mimiciv" / "3.1"
    plain_relative = Path("LICENSE.txt")
    gzip_relative = Path("hosp") / "admissions.csv.gz"
    plain_path = mimic_root / plain_relative
    gzip_path = mimic_root / gzip_relative
    plain_path.parent.mkdir(parents=True, exist_ok=True)
    plain_path.write_text("synthetic license text\n", encoding="utf-8")
    write_gzip(gzip_path, b"col\nvalue\n")
    plain_hash = hashlib.sha256(plain_path.read_bytes()).hexdigest()
    gzip_hash = hashlib.sha256(gzip_path.read_bytes()).hexdigest()
    mimic_root.joinpath("SHA256SUMS.txt").write_text(
        "\n".join(
            (
                f"{plain_hash}  {plain_relative.as_posix()}",
                f"{gzip_hash}  {gzip_relative.as_posix()}",
                "0000  hosp/missing.csv.gz",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "pipeline.source_integrity.SOURCE_SPECS",
        (
            SourceSpec(
                name="mimiciv",
                version="3.1",
                root_relative_path=Path("mimiciv") / "3.1",
                expected_files=(),
            ),
        ),
    )

    targets = manifest_targets(dataset_root=dataset_root)
    assert {target.file_relative.as_posix() for target in targets} == {
        "LICENSE.txt",
        "hosp/admissions.csv.gz",
        "hosp/missing.csv.gz",
    }
    assert {target.table_name for target in targets} == {
        "mimiciv_LICENSE",
        "mimiciv_hosp_admissions",
        "mimiciv_hosp_missing",
    }
    report = build_integrity_report(
        targets=targets,
        dataset_root=dataset_root,
        output_path=tmp_path / "reports" / "full_integrity.json",
    )

    assert report["summary"]["checksum_status_counts"] == {
        "matched": 2,
        "missing_local_file": 1,
    }
    assert report["summary"]["gzip_status_counts"] == {
        "not_applicable": 1,
        "passed": 1,
        "not_checked": 1,
    }


def test_missing_manifest_gzip_can_report_configured_plain_csv_alternate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_root = tmp_path / "Dataset"
    source_root = dataset_root / "notes" / "2.2"
    plain_relative = Path("note") / "detail.csv"
    manifest_relative = Path("note") / "detail.csv.gz"
    plain_path = source_root / plain_relative
    plain_path.parent.mkdir(parents=True, exist_ok=True)
    plain_path.write_text("note_id,subject_id\n", encoding="utf-8")
    source_root.joinpath("SHA256SUMS.txt").write_text(
        f"0000  {manifest_relative.as_posix()}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "pipeline.source_integrity.SOURCE_SPECS",
        (
            SourceSpec(
                name="mimiciv_note",
                version="2.2",
                root_relative_path=Path("notes") / "2.2",
                expected_files=(plain_relative,),
            ),
        ),
    )

    report = build_integrity_report(
        targets=manifest_targets(dataset_root=dataset_root),
        dataset_root=dataset_root,
        output_path=tmp_path / "reports" / "integrity.json",
    )

    result = report["results"][0]
    assert (
        result["checksum_status"]
        == "missing_manifest_file_but_configured_alternate_present"
    )
    assert result["alternate_local_file"] == {
        "relative_path": "notes/2.2/note/detail.csv",
        "file_size_bytes": plain_path.stat().st_size,
        "reason": (
            "configured source layout expects this table as an uncompressed CSV "
            "rather than the manifest .csv.gz entry"
        ),
    }
