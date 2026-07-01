import importlib.util
import json
import sys
from pathlib import Path
from typing import Sequence

import duckdb


def load_builder_module():
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "build_condition_mappings.py"
    )
    spec = importlib.util.spec_from_file_location(
        "build_condition_mappings", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


builder = load_builder_module()


def sql_string(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def write_parquet_rows(
    path: Path,
    columns: Sequence[str],
    rows: Sequence[Sequence[str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = ", ".join(
        "(" + ", ".join(sql_string(value) for value in row) + ")" for row in rows
    )
    column_sql = ", ".join(columns)
    with duckdb.connect(database=":memory:") as connection:
        connection.execute(
            f"CREATE TABLE rows AS SELECT * FROM (VALUES {values}) AS t({column_sql})"
        )
        connection.execute(f"COPY rows TO {sql_string(path)} (FORMAT PARQUET)")


def read_csv_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_build_condition_mappings_writes_aggregate_templates(tmp_path: Path) -> None:
    extracts_root = tmp_path / "Dataset" / "processed" / "extracts"
    mapping_root = tmp_path / "Dataset" / "mappings"
    report_path = tmp_path / "reports" / "condition_mapping_build_report.json"

    write_parquet_rows(
        extracts_root / "mimiciv" / "diagnoses_icd.parquet",
        (
            "source",
            "source_version",
            "patient_uid",
            "stay_uid",
            "extraction_version",
            "seq_num",
            "icd_code",
            "icd_version",
        ),
        (
            ("mimiciv", "3.1", "mimiciv:10", "mimiciv:30", "test", "1", "A419", "10"),
            ("mimiciv", "3.1", "mimiciv:10", "mimiciv:30", "test", "2", "0389", "9"),
        ),
    )
    write_parquet_rows(
        extracts_root / "eicu_crd" / "diagnosis.parquet",
        (
            "source",
            "source_version",
            "patient_uid",
            "stay_uid",
            "extraction_version",
            "diagnosisid",
            "diagnosisstring",
            "icd9code",
            "diagnosispriority",
        ),
        (
            (
                "eicu_crd",
                "2.0",
                "eicu_crd:patient-a",
                "eicu_crd:400",
                "test",
                "d1",
                "infectious|sepsis",
                "0389",
                "1",
            ),
            (
                "eicu_crd",
                "2.0",
                "eicu_crd:patient-a",
                "eicu_crd:400",
                "test",
                "d2",
                "pulmonary|edema",
                "",
                "2",
            ),
        ),
    )

    report = builder.build_condition_mappings(
        builder.ConditionMappingBuildConfig(
            dataset_root=tmp_path / "Dataset",
            extracts_root=extracts_root,
            mapping_root=mapping_root,
            report_path=report_path,
        )
    )

    conditions_dir = mapping_root / "conditions"
    assert (conditions_dir / "mimic_distinct_icd_codes_for_mapping.csv").exists()
    assert (conditions_dir / "eicu_distinct_icd9_codes_for_mapping.csv").exists()
    assert (conditions_dir / "eicu_distinct_diagnosis_text_for_mapping.csv").exists()
    assert (conditions_dir / "eicu_diagnosis_text_condition_map_template.csv").exists()
    assert (conditions_dir / "project_condition_groups_template.csv").exists()

    assert report["mimic"]["distinct_icd_code_count"] == 2
    assert report["eicu"]["distinct_diagnosis_text_count"] == 2
    assert report["data_safety"]["contains_patient_rows"] is False

    # The active harmonization file is never written by the bootstrap script.
    assert not (conditions_dir / "project_condition_groups.csv").exists()

    report_text = json.dumps(report)
    assert "patient-a" not in report_text
    assert "mimiciv:10" not in report_text
    text_template = read_csv_text(
        conditions_dir / "eicu_distinct_diagnosis_text_for_mapping.csv"
    )
    assert "infectious_sepsis" in text_template
    assert "patient-a" not in text_template


def test_build_condition_mappings_degrades_without_extracts(tmp_path: Path) -> None:
    mapping_root = tmp_path / "Dataset" / "mappings"
    report = builder.build_condition_mappings(
        builder.ConditionMappingBuildConfig(
            dataset_root=tmp_path / "Dataset",
            extracts_root=tmp_path / "Dataset" / "processed" / "extracts",
            mapping_root=mapping_root,
            report_path=tmp_path / "reports" / "condition_mapping_build_report.json",
        )
    )

    assert report["mimic"]["status"] == "missing_extract"
    assert report["eicu"]["status"] == "missing_extract"
    # Templates still created (empty) so the folder contract is stable.
    assert (
        mapping_root / "conditions" / "project_condition_groups_template.csv"
    ).exists()
