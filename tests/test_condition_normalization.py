"""Synthetic tests for the optional semantic condition normalization layer.

All fixtures are synthetic. No licensed clinical rows are used. The tests assert
that source-native diagnosis fields are always preserved, shared roll-up tokens
are added only from authoritative mapping fixtures, missing mapping files never
fail harmonization or drop rows, and reports stay aggregate-only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import duckdb

from pipeline.harmonize import (
    HarmonizationBuildConfig,
    build_harmonized_artifacts,
)


def sql_string(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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


def read_parquet_rows(path: Path) -> list[dict[str, object]]:
    with duckdb.connect(database=":memory:") as connection:
        cursor = connection.execute(f"SELECT * FROM read_parquet({sql_string(path)})")
        columns = [description[0] for description in cursor.description]
        return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]


COHORT_COLUMNS = (
    "source",
    "source_version",
    "patient_uid",
    "encounter_uid",
    "stay_uid",
    "source_patient_id",
    "source_encounter_id",
    "source_stay_id",
    "age_years",
    "age_topcoded",
    "sex",
    "race_or_ethnicity",
    "hospital_id",
    "ward_id",
    "admission_type",
    "admission_source",
    "unit_type",
    "last_unit_type",
    "stay_type",
    "stay_sequence",
    "cohort_rule",
)

COHORT_ROWS = (
    (
        "mimiciv",
        "3.1",
        "mimiciv:10",
        "mimiciv:20",
        "mimiciv:30",
        "10",
        "20",
        "30",
        "65",
        "false",
        "F",
        "synthetic race",
        "",
        "",
        "urgent",
        "transfer",
        "MICU",
        "MICU",
        "first_icu_stay_per_admission",
        "1",
        "adult_icu",
    ),
    (
        "eicu_crd",
        "2.0",
        "eicu_crd:patient-a",
        "eicu_crd:500",
        "eicu_crd:400",
        "patient-a",
        "500",
        "400",
        "70",
        "false",
        "M",
        "synthetic ethnicity",
        "hospital-1",
        "ward-1",
        "unit admit",
        "emergency",
        "Med-Surg ICU",
        "",
        "admit",
        "1",
        "adult_unit_stay",
    ),
)

MIMIC_DX_COLUMNS = (
    "source",
    "source_version",
    "patient_uid",
    "encounter_uid",
    "stay_uid",
    "source_patient_id",
    "source_encounter_id",
    "source_stay_id",
    "extraction_version",
    "seq_num",
    "icd_code",
    "icd_version",
)

EICU_DX_COLUMNS = (
    "source",
    "source_version",
    "patient_uid",
    "encounter_uid",
    "stay_uid",
    "source_patient_id",
    "source_encounter_id",
    "source_stay_id",
    "extraction_version",
    "diagnosisid",
    "patientunitstayid",
    "activeupondischarge",
    "diagnosisoffset",
    "diagnosisstring",
    "icd9code",
    "diagnosispriority",
)


def _mimic_ids() -> tuple[str, ...]:
    return (
        "mimiciv",
        "3.1",
        "mimiciv:10",
        "mimiciv:20",
        "mimiciv:30",
        "10",
        "20",
        "30",
        "test-extract",
    )


def _eicu_ids() -> tuple[str, ...]:
    return (
        "eicu_crd",
        "2.0",
        "eicu_crd:patient-a",
        "eicu_crd:500",
        "eicu_crd:400",
        "patient-a",
        "500",
        "400",
        "test-extract",
    )


def _write_medication_gate(mapping_root: Path) -> None:
    """Write empty medication mapping CSVs so the hard medication gate passes."""

    write_text(
        mapping_root / "medications" / "mimic_ndc_rxnorm_atc.csv",
        "ndc,rxcui,ingredient_name,rxnorm_name,atc_code,atc_level\n",
    )
    write_text(
        mapping_root / "medications" / "eicu_drug_rxnorm_atc.csv",
        "drughiclseqno,gtc,drug_name,rxcui,ingredient_name,rxnorm_name,atc_code,atc_level\n",
    )


def _run(
    tmp_path: Path,
    *,
    mimic_dx: Sequence[Sequence[str]] = (),
    eicu_dx: Sequence[Sequence[str]] = (),
    condition_files: dict[str, str] | None = None,
) -> tuple[dict[str, object], Path, Path]:
    cohort_path = tmp_path / "cohorts" / "cohort_stays.parquet"
    extracts_root = tmp_path / "Dataset" / "processed" / "extracts"
    mapping_root = tmp_path / "Dataset" / "mappings"
    harmonized_root = tmp_path / "Dataset" / "processed" / "harmonized"
    reports_root = tmp_path / "reports"

    write_parquet_rows(cohort_path, COHORT_COLUMNS, COHORT_ROWS)
    _write_medication_gate(mapping_root)
    if mimic_dx:
        write_parquet_rows(
            extracts_root / "mimiciv" / "diagnoses_icd.parquet",
            MIMIC_DX_COLUMNS,
            mimic_dx,
        )
    if eicu_dx:
        write_parquet_rows(
            extracts_root / "eicu_crd" / "diagnosis.parquet",
            EICU_DX_COLUMNS,
            eicu_dx,
        )
    for name, content in (condition_files or {}).items():
        write_text(mapping_root / "conditions" / name, content)

    manifest = build_harmonized_artifacts(
        HarmonizationBuildConfig(
            cohort_path=cohort_path,
            extracts_root=extracts_root,
            harmonized_root=harmonized_root,
            mapping_root=mapping_root,
            manifest_path=reports_root / "harmonization_manifest.json",
            coverage_path=reports_root / "harmonization_coverage.json",
            unmapped_path=reports_root / "unmapped_concepts.json",
            condition_coverage_path=(
                reports_root / "condition_normalization_coverage.json"
            ),
            text_review_path=reports_root / "eicu_diagnosis_text_mapping_review.csv",
        )
    )
    return manifest, harmonized_root, reports_root


def _mimic_dx_row(seq: str, icd_code: str, icd_version: str) -> tuple[str, ...]:
    return (*_mimic_ids(), seq, icd_code, icd_version)


def _eicu_dx_row(
    diagnosis_id: str, diagnosis_string: str, icd9code: str, priority: str = "1"
) -> tuple[str, ...]:
    return (
        *_eicu_ids(),
        diagnosis_id,
        "400",
        "False",
        "120",
        diagnosis_string,
        icd9code,
        priority,
    )


def _conditions_by_token(harmonized_root: Path) -> dict[str, dict[str, object]]:
    rows = read_parquet_rows(harmonized_root / "conditions.parquet")
    return {str(row["condition_token"]): row for row in rows}


def test_mimic_icd10_maps_to_ccsr_when_fixture_present(tmp_path: Path) -> None:
    manifest, harmonized_root, _ = _run(
        tmp_path,
        mimic_dx=[_mimic_dx_row("1", "A419", "10")],
        condition_files={
            "icd10_ccsr.csv": (
                "icd_code,ccsr_category,ccsr_category_description\n"
                "A41.9,SEP,Sepsis unspecified organism\n"
            ),
        },
    )

    assert manifest["status"] == "completed"
    row = _conditions_by_token(harmonized_root)["icd10:a419"]
    assert row["condition_token"] == "icd10:a419"
    assert row["source_condition_code"] == "A419"
    assert row["normalized_condition_token"] == "ccsr:SEP"
    assert row["normalized_condition_system"] == "CCSR"
    assert row["condition_rollup_level"] == "ccsr"
    assert row["mapping_status"] == "mapped_ccsr"
    assert row["mapping_confidence"] == "exact"


def test_mimic_icd9_maps_to_ccs_when_fixture_present(tmp_path: Path) -> None:
    _, harmonized_root, _ = _run(
        tmp_path,
        mimic_dx=[_mimic_dx_row("1", "0389", "9")],
        condition_files={
            "icd9_ccs.csv": (
                "icd_code,ccs_category,ccs_category_description\n"
                "038.9,CCS2,Septicemia\n"
            ),
        },
    )

    row = _conditions_by_token(harmonized_root)["icd9:0389"]
    assert row["normalized_condition_token"] == "ccs:CCS2"
    assert row["condition_rollup_level"] == "ccs"
    assert row["mapping_status"] == "mapped_ccs"


def test_eicu_icd9_maps_to_ccs_when_fixture_present(tmp_path: Path) -> None:
    _, harmonized_root, _ = _run(
        tmp_path,
        eicu_dx=[_eicu_dx_row("d1", "infectious|sepsis", "0389")],
        condition_files={
            "icd9_ccs.csv": (
                "icd_code,ccs_category,ccs_category_description\n"
                "038.9,CCS2,Septicemia\n"
            ),
        },
    )

    row = _conditions_by_token(harmonized_root)["icd9:0389"]
    assert row["normalized_condition_token"] == "ccs:CCS2"
    assert row["mapping_status"] == "mapped_ccs"


def test_eicu_text_only_stays_source_native_without_text_map(tmp_path: Path) -> None:
    _, harmonized_root, _ = _run(
        tmp_path,
        eicu_dx=[_eicu_dx_row("d1", "pulmonary|acute pulmonary edema", "")],
    )

    rows = read_parquet_rows(harmonized_root / "conditions.parquet")
    assert len(rows) == 1
    row = rows[0]
    assert row["condition_token"] == "pulmonary_acute_pulmonary_edema"
    assert row["normalized_condition_token"] is None
    assert row["condition_rollup_level"] is None
    assert row["mapping_status"] == "source_native_text"


def test_eicu_text_only_maps_on_exact_curated_match(tmp_path: Path) -> None:
    _, harmonized_root, reports_root = _run(
        tmp_path,
        eicu_dx=[
            _eicu_dx_row("d1", "infectious diseases|sepsis|severe", ""),
            _eicu_dx_row("d2", "pulmonary|acute pulmonary edema", ""),
        ],
        condition_files={
            "eicu_diagnosis_text_condition_map.csv": (
                "diagnosisstring_normalized,condition_rollup_token,condition_name\n"
                "infectious_diseases_sepsis_severe,condition:sepsis,Sepsis\n"
            ),
        },
    )

    by_token = _conditions_by_token(harmonized_root)
    mapped = by_token["infectious_diseases_sepsis_severe"]
    assert mapped["normalized_condition_token"] == "condition:sepsis"
    assert mapped["mapping_status"] == "mapped_text_to_condition"
    assert mapped["condition_rollup_level"] == "text_mapped"
    unmapped = by_token["pulmonary_acute_pulmonary_edema"]
    assert unmapped["mapping_status"] == "source_native_text"

    review_path = reports_root / "eicu_diagnosis_text_mapping_review.csv"
    review_text = review_path.read_text(encoding="utf-8")
    assert "diagnosisstring_normalized" in review_text
    assert "condition:sepsis" in review_text
    assert "patient-a" not in review_text


def test_missing_condition_maps_degrade_to_structural_category(tmp_path: Path) -> None:
    manifest, harmonized_root, _ = _run(
        tmp_path,
        mimic_dx=[_mimic_dx_row("1", "A419", "10")],
        eicu_dx=[_eicu_dx_row("d1", "pulmonary|acute pulmonary edema", "")],
    )

    assert manifest["status"] == "completed"
    assert all(
        resource["status"] == "missing"
        for resource in manifest["condition_mapping_resources"]
    )
    by_token = _conditions_by_token(harmonized_root)
    mimic = by_token["icd10:a419"]
    assert mimic["mapping_status"] == "source_native_code"
    assert mimic["condition_rollup_level"] == "category"
    assert mimic["normalized_condition_token"] == "icd10cat:a41"
    assert mimic["normalized_condition_system"] == "ICD10CM_CATEGORY"
    eicu = by_token["pulmonary_acute_pulmonary_edema"]
    assert eicu["mapping_status"] == "source_native_text"


def test_no_rows_dropped_when_mappings_missing(tmp_path: Path) -> None:
    mimic_dx = [
        _mimic_dx_row("1", "A419", "10"),
        _mimic_dx_row("2", "0389", "9"),
        _mimic_dx_row("3", "", ""),
    ]
    eicu_dx = [
        _eicu_dx_row("d1", "infectious|sepsis", "0389"),
        _eicu_dx_row("d2", "pulmonary|edema", ""),
        _eicu_dx_row("d3", "", ""),
    ]
    _, harmonized_root, _ = _run(tmp_path, mimic_dx=mimic_dx, eicu_dx=eicu_dx)

    rows = read_parquet_rows(harmonized_root / "conditions.parquet")
    assert len(rows) == len(mimic_dx) + len(eicu_dx)


def test_condition_token_backward_compatible(tmp_path: Path) -> None:
    _, harmonized_root, _ = _run(
        tmp_path,
        mimic_dx=[_mimic_dx_row("1", "A41", "10")],
        condition_files={
            "icd10_ccsr.csv": (
                "icd_code,ccsr_category,ccsr_category_description\nA41,SEP,Sepsis\n"
            ),
        },
    )

    rows = read_parquet_rows(harmonized_root / "conditions.parquet")
    assert rows[0]["condition_token"] == "icd10:a41"
    assert rows[0]["source_condition_token"] == "icd10:a41"


def test_mapping_status_values_are_stable(tmp_path: Path) -> None:
    _, harmonized_root, _ = _run(
        tmp_path,
        mimic_dx=[_mimic_dx_row("1", "A419", "10")],
        eicu_dx=[_eicu_dx_row("d1", "pulmonary|edema", "")],
    )

    allowed = {
        "mapped_ccsr",
        "mapped_ccs",
        "mapped_icd_crosswalk",
        "mapped_chapter",
        "mapped_text_to_condition",
        "source_native_code",
        "source_native_text",
        "unmapped_condition",
    }
    rows = read_parquet_rows(harmonized_root / "conditions.parquet")
    assert {str(row["mapping_status"]) for row in rows} <= allowed


def test_gem_crosswalk_bridges_icd9_to_ccsr(tmp_path: Path) -> None:
    _, harmonized_root, _ = _run(
        tmp_path,
        mimic_dx=[_mimic_dx_row("1", "0389", "9")],
        condition_files={
            "icd10_ccsr.csv": (
                "icd_code,ccsr_category,ccsr_category_description\nA41.9,SEP,Sepsis\n"
            ),
            "icd9_to_icd10_gem.csv": (
                "icd9_code,icd10_code,approximate_flag\n038.9,A41.9,1\n"
            ),
        },
    )

    row = _conditions_by_token(harmonized_root)["icd9:0389"]
    assert row["normalized_condition_token"] == "ccsr:SEP"
    assert row["mapping_status"] == "mapped_icd_crosswalk"
    assert row["mapping_confidence"] == "approximate"


def test_chapter_fallback_when_no_ccs_ccsr(tmp_path: Path) -> None:
    _, harmonized_root, _ = _run(
        tmp_path,
        mimic_dx=[_mimic_dx_row("1", "A419", "10")],
        condition_files={
            "icd_chapters.csv": (
                "icd_version,category_code,chapter_code,chapter_name\n"
                "10,A41,I,Certain infectious and parasitic diseases\n"
            ),
        },
    )

    row = _conditions_by_token(harmonized_root)["icd10:a419"]
    assert row["normalized_condition_token"] == "icd10chap:I"
    assert row["mapping_status"] == "mapped_chapter"
    assert row["condition_rollup_level"] == "chapter"


def test_project_condition_group_applied_as_separate_layer(tmp_path: Path) -> None:
    _, harmonized_root, reports_root = _run(
        tmp_path,
        mimic_dx=[_mimic_dx_row("1", "A419", "10")],
        condition_files={
            "icd10_ccsr.csv": (
                "icd_code,ccsr_category,ccsr_category_description\nA41.9,SEP,Sepsis\n"
            ),
            "project_condition_groups.csv": (
                "match_type,match_value,project_condition_group,project_condition_token\n"
                "icd_code,A41.9,sepsis,condition:sepsis\n"
            ),
        },
    )

    row = _conditions_by_token(harmonized_root)["icd10:a419"]
    # Roll-up and project group coexist without overwriting each other.
    assert row["normalized_condition_token"] == "ccsr:SEP"
    assert row["project_condition_group"] == "sepsis"
    assert row["project_condition_token"] == "condition:sepsis"

    coverage = json.loads(
        (reports_root / "condition_normalization_coverage.json").read_text(
            encoding="utf-8"
        )
    )
    assert coverage["data_safety"]["contains_patient_rows"] is False


def test_project_condition_group_supports_icd_prefix(tmp_path: Path) -> None:
    _, harmonized_root, reports_root = _run(
        tmp_path,
        mimic_dx=[_mimic_dx_row("1", "A419", "10")],
        condition_files={
            "icd10_ccsr.csv": (
                "icd_code,ccsr_category,ccsr_category_description\nA41.9,SEP,Sepsis\n"
            ),
            "project_condition_groups.csv": (
                "match_type,match_value,project_condition_group,project_condition_token\n"
                "icd_prefix,A41,sepsis,condition:sepsis\n"
            ),
        },
    )

    row = _conditions_by_token(harmonized_root)["icd10:a419"]
    assert row["normalized_condition_token"] == "ccsr:SEP"
    assert row["project_condition_group"] == "sepsis"
    assert row["project_condition_token"] == "condition:sepsis"
    coverage = json.loads(
        (reports_root / "condition_normalization_coverage.json").read_text(
            encoding="utf-8"
        )
    )
    summary = {row["source"]: row for row in coverage["per_source_summary"]}
    assert summary["mimiciv"]["project_group_rows"] == 1


def test_reports_contain_no_patient_level_strings(tmp_path: Path) -> None:
    _, _, reports_root = _run(
        tmp_path,
        mimic_dx=[_mimic_dx_row("1", "A419", "10")],
        eicu_dx=[_eicu_dx_row("d1", "infectious|sepsis", "0389")],
        condition_files={
            "icd9_ccs.csv": (
                "icd_code,ccs_category,ccs_category_description\n"
                "038.9,CCS2,Septicemia\n"
            ),
        },
    )

    for name in (
        "harmonization_manifest.json",
        "harmonization_coverage.json",
        "condition_normalization_coverage.json",
        "unmapped_concepts.json",
    ):
        text = (reports_root / name).read_text(encoding="utf-8")
        assert "patient-a" not in text
        assert "mimiciv:10" not in text
