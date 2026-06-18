import json
from pathlib import Path

import pytest

from pipeline.profile_tables import (
    NumericCheck,
    ReferentialCheck,
    TableProfileSpec,
    profile_quality,
    selected_specs,
)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_profile_quality_reports_aggregate_metrics_without_values(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "Dataset"
    output_path = tmp_path / "reports" / "quality_profile.json"
    write_text(
        dataset_root / "source" / "patients.csv",
        "\n".join(
            [
                "patient_id,age,admit_time,sex",
                "p1,45,2020-01-01 00:00:00,F",
                "p2,not_numeric,not_a_timestamp,M",
            ]
        )
        + "\n",
    )
    write_text(
        dataset_root / "source" / "events.csv",
        "\n".join(
            [
                "event_id,patient_id,value,category",
                "e1,p1,10,A",
                "e1,p3,200,B",
                "e2,,bad,C",
            ]
        )
        + "\n",
    )
    specs = (
        TableProfileSpec(
            source="synthetic",
            source_version="1.0",
            table_name="patients",
            relative_path=Path("source") / "patients.csv",
            required_columns=("patient_id", "age", "admit_time"),
            key_columns=("patient_id",),
            profile_columns=("patient_id", "sex"),
            numeric_checks=(NumericCheck("age", 18, 120),),
            timestamp_columns=("admit_time",),
            categorical_columns=("sex",),
        ),
        TableProfileSpec(
            source="synthetic",
            source_version="1.0",
            table_name="events",
            relative_path=Path("source") / "events.csv",
            required_columns=("event_id", "patient_id", "value"),
            key_columns=("event_id",),
            profile_columns=("event_id", "patient_id", "category"),
            numeric_checks=(NumericCheck("value", 0, 100),),
            categorical_columns=("category",),
            referential_checks=(
                ReferentialCheck(
                    name="events_patient_to_patients",
                    child_columns=("patient_id",),
                    parent_relative_path=Path("source") / "patients.csv",
                    parent_columns=("patient_id",),
                ),
            ),
        ),
    )

    report = profile_quality(specs, dataset_root=dataset_root, output_path=output_path)
    report_text = json.dumps(report)
    patients = report["tables"][0]
    events = report["tables"][1]

    assert report["data_safety"]["contains_patient_rows"] is False
    assert patients["row_count"] == 2
    assert patients["numeric_profiles"]["age"]["parse_failure_count"] == 1
    assert patients["timestamp_profiles"]["admit_time"]["parse_failure_count"] == 1
    assert events["row_count"] == 3
    assert events["column_profiles"]["patient_id"]["null_count"] == 1
    assert events["numeric_profiles"]["value"]["parse_failure_count"] == 1
    assert events["numeric_profiles"]["value"]["out_of_bounds_count"] == 1
    assert events["duplicate_key_profile"]["duplicate_excess_rows"] == 1
    assert events["referential_integrity"][0]["orphan_rows"] == 1
    assert "not_numeric" not in report_text
    assert "not_a_timestamp" not in report_text
    assert "p1" not in report_text


def test_profile_quality_skips_tables_with_missing_required_columns(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "Dataset"
    write_text(dataset_root / "source" / "bad.csv", "only_column\nvalue\n")
    spec = TableProfileSpec(
        source="synthetic",
        source_version="1.0",
        table_name="bad",
        relative_path=Path("source") / "bad.csv",
        required_columns=("required_column",),
    )

    report = profile_quality(
        (spec,),
        dataset_root=dataset_root,
        output_path=tmp_path / "reports" / "quality_profile.json",
    )

    assert report["tables"][0]["status"] == "skipped_missing_required_columns"
    assert report["tables"][0]["missing_required_columns"] == ["required_column"]


def test_selected_specs_rejects_unknown_table_names() -> None:
    with pytest.raises(ValueError, match="Unknown table profile names"):
        selected_specs(("no_such_table",))
