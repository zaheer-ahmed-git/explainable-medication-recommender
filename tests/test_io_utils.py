import gzip
import json
from pathlib import Path

import pytest

from pipeline.config import SourceSpec
from pipeline.io_utils import (
    DatasetPathError,
    build_source_inventory,
    inspect_header,
    read_csv_bounded,
    requires_bounded_query,
    resolve_dataset_path,
)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_gzip_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, mode="wt", encoding="utf-8", newline="") as file_obj:
        file_obj.write(text)


def test_resolve_dataset_path_accepts_paths_inside_root(tmp_path: Path) -> None:
    dataset_root = tmp_path / "Dataset"
    target = dataset_root / "source" / "table.csv"
    write_text(target, "id,value\n1,alpha\n")

    resolved = resolve_dataset_path(
        Path("source") / "table.csv",
        dataset_root=dataset_root,
    )

    assert resolved == target.resolve()


def test_resolve_dataset_path_rejects_paths_outside_root(tmp_path: Path) -> None:
    dataset_root = tmp_path / "Dataset"
    dataset_root.mkdir()

    with pytest.raises(DatasetPathError):
        resolve_dataset_path(Path("..") / "outside.csv", dataset_root=dataset_root)


def test_resolve_dataset_path_reports_missing_files(tmp_path: Path) -> None:
    dataset_root = tmp_path / "Dataset"
    dataset_root.mkdir()

    with pytest.raises(DatasetPathError):
        resolve_dataset_path("missing.csv", dataset_root=dataset_root)


def test_inspect_header_supports_plain_and_gzipped_csv(tmp_path: Path) -> None:
    plain_csv = tmp_path / "plain.csv"
    gz_csv = tmp_path / "compressed.csv.gz"
    write_text(plain_csv, "stay_uid,feature\ns1,10\n")
    write_gzip_text(gz_csv, "patient_uid,value\np1,20\n")

    assert inspect_header(plain_csv) == ("stay_uid", "feature")
    assert inspect_header(gz_csv) == ("patient_uid", "value")


def test_inventory_contains_metadata_without_rows(tmp_path: Path) -> None:
    dataset_root = tmp_path / "Dataset"
    source_root = dataset_root / "synthetic" / "1.0"
    write_text(source_root / "table.csv", "patient_uid,value\npatient_1,secret_value\n")
    write_text(source_root / "README.txt", "supporting metadata\n")
    write_text(source_root / "SHA256SUMS.txt", "checksum placeholder\n")

    source_spec = SourceSpec(
        name="synthetic",
        version="1.0",
        root_relative_path=Path("synthetic") / "1.0",
        expected_files=(Path("table.csv"),),
    )

    inventory = build_source_inventory((source_spec,), dataset_root=dataset_root)
    payload = json.dumps(inventory)

    assert "patient_uid" in payload
    assert "patient_1" not in payload
    assert "secret_value" not in payload
    assert inventory["data_safety"]["contains_patient_rows"] is False
    assert inventory["sources"][0]["missing_expected_files"] == []


def test_requires_bounded_query_uses_configured_and_size_rules() -> None:
    assert requires_bounded_query(
        "mimiciv",
        Path("hosp") / "labevents.csv.gz",
        size_bytes=1,
    )
    assert requires_bounded_query(
        "synthetic",
        Path("large.csv.gz"),
        size_bytes=101_000_000,
    )
    assert not requires_bounded_query(
        "synthetic",
        Path("small.csv"),
        size_bytes=10,
    )


def test_read_csv_bounded_projects_columns_and_limits_rows(tmp_path: Path) -> None:
    dataset_root = tmp_path / "Dataset"
    table_path = dataset_root / "table.csv"
    write_text(table_path, "id,value,extra\n1,a,x\n2,b,y\n")

    rows = read_csv_bounded(
        "table.csv",
        columns=("id", "value"),
        limit=1,
        dataset_root=dataset_root,
    )

    assert rows == [{"id": 1, "value": "a"}]
