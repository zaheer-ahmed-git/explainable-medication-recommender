import gzip
import json
from pathlib import Path

import duckdb
import pytest

from pipeline.cohort import (
    CohortBuildConfig,
    build_cohort_artifacts,
    validate_required_columns,
)
from pipeline.io_utils import DatasetPathError


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_gzip_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, mode="wt", encoding="utf-8", newline="") as file_obj:
        file_obj.write(text)


def create_synthetic_dataset(dataset_root: Path) -> None:
    write_gzip_text(
        dataset_root / "mimiciv" / "3.1" / "hosp" / "patients.csv.gz",
        "\n".join(
            [
                "subject_id,gender,anchor_age,anchor_year,anchor_year_group,dod",
                "100,F,42,2150,2017 - 2019,",
                "101,M,17,2150,2017 - 2019,",
                "102,F,65,2150,2017 - 2019,",
            ]
        )
        + "\n",
    )
    write_gzip_text(
        dataset_root / "mimiciv" / "3.1" / "hosp" / "admissions.csv.gz",
        "\n".join(
            [
                "subject_id,hadm_id,admittime,dischtime,deathtime,admission_type,admit_provider_id,admission_location,discharge_location,insurance,language,marital_status,race,edregtime,edouttime,hospital_expire_flag",
                "100,200,2150-01-01 00:00:00,2150-01-05 00:00:00,,URGENT,P1,EMERGENCY ROOM,HOME,Private,English,MARRIED,WHITE,,,0",
                "101,201,2150-02-01 00:00:00,2150-02-03 00:00:00,,URGENT,P1,EMERGENCY ROOM,HOME,Private,English,SINGLE,WHITE,,,0",
                "102,202,2150-03-01 00:00:00,2150-03-06 00:00:00,,ELECTIVE,P2,PHYSICIAN REFERRAL,HOME,Medicare,English,WIDOWED,BLACK,,,0",
            ]
        )
        + "\n",
    )
    write_gzip_text(
        dataset_root / "mimiciv" / "3.1" / "icu" / "icustays.csv.gz",
        "\n".join(
            [
                "subject_id,hadm_id,stay_id,first_careunit,last_careunit,intime,outtime,los",
                "100,200,300,MICU,MICU,2150-01-01 02:00:00,2150-01-02 02:00:00,1.0",
                "100,200,301,SICU,SICU,2150-01-02 03:00:00,2150-01-03 03:00:00,1.0",
                "101,201,302,MICU,MICU,2150-02-01 02:00:00,2150-02-02 02:00:00,1.0",
                "102,202,303,CCU,CCU,2150-03-01 02:00:00,2150-03-02 14:00:00,1.5",
            ]
        )
        + "\n",
    )
    write_gzip_text(
        dataset_root / "eicu-crd" / "2.0" / "patient.csv.gz",
        "\n".join(
            [
                "patientunitstayid,patienthealthsystemstayid,gender,age,ethnicity,hospitalid,wardid,apacheadmissiondx,admissionheight,hospitaladmittime24,hospitaladmitoffset,hospitaladmitsource,hospitaldischargeyear,hospitaldischargetime24,hospitaldischargeoffset,hospitaldischargelocation,hospitaldischargestatus,unittype,unitadmittime24,unitadmitsource,unitvisitnumber,unitstaytype,admissionweight,dischargeweight,unitdischargetime24,unitdischargeoffset,unitdischargelocation,unitdischargestatus,uniquepid",
                "400,500,Female,> 89,Caucasian,10,20,Sepsis,,,,Emergency Department,,,,,,MICU,,ED,1,admit,,,,1440,,,patient-a",
                "401,501,Male,17,Caucasian,10,20,Observation,,,,Emergency Department,,,,,,MICU,,ED,1,admit,,,,500,,,patient-b",
                "402,502,Female,40,Black,11,21,Respiratory failure,,,,Transfer,,,,,,SICU,,Floor,2,readmit,,,,2880,,,",
            ]
        )
        + "\n",
    )


def read_parquet_rows(path: Path) -> list[dict[str, object]]:
    escaped_path = str(path).replace("'", "''")
    with duckdb.connect(database=":memory:") as connection:
        cursor = connection.execute(f"SELECT * FROM read_parquet('{escaped_path}')")
        column_names = [description[0] for description in cursor.description]
        return [dict(zip(column_names, row, strict=True)) for row in cursor.fetchall()]


def test_build_cohort_artifacts_creates_source_and_unified_outputs(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "Dataset"
    cohorts_root = dataset_root / "processed" / "cohorts"
    manifest_path = tmp_path / "reports" / "cohort_manifest.json"
    create_synthetic_dataset(dataset_root)

    manifest = build_cohort_artifacts(
        CohortBuildConfig(
            dataset_root=dataset_root,
            cohorts_root=cohorts_root,
            manifest_path=manifest_path,
        )
    )

    mimic_rows = read_parquet_rows(cohorts_root / "mimic_icu_stays.parquet")
    eicu_rows = read_parquet_rows(cohorts_root / "eicu_unit_stays.parquet")
    unified_rows = read_parquet_rows(cohorts_root / "cohort_stays.parquet")

    assert {row["stay_uid"] for row in mimic_rows} == {"mimiciv:300", "mimiciv:303"}
    assert {row["stay_uid"] for row in eicu_rows} == {
        "eicu_crd:400",
        "eicu_crd:402",
    }
    assert len(unified_rows) == 4
    assert all(str(row["stay_uid"]).startswith(row["source"]) for row in unified_rows)
    assert manifest["sources"]["mimiciv"]["excluded_by_first_stay_rule"] == 1
    assert manifest["sources"]["eicu_crd"]["topcoded_age_stays"] == 1


def test_manifest_is_aggregate_only(tmp_path: Path) -> None:
    dataset_root = tmp_path / "Dataset"
    manifest_path = tmp_path / "reports" / "cohort_manifest.json"
    create_synthetic_dataset(dataset_root)

    build_cohort_artifacts(
        CohortBuildConfig(
            dataset_root=dataset_root,
            cohorts_root=dataset_root / "processed" / "cohorts",
            manifest_path=manifest_path,
        )
    )
    manifest_text = manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)

    assert manifest["data_safety"]["contains_patient_rows"] is False
    assert "mimiciv:300" not in manifest_text
    assert "eicu_crd:400" not in manifest_text
    assert "patient-a" not in manifest_text


def test_include_all_mimic_icu_stays_option_keeps_repeated_stays(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "Dataset"
    cohorts_root = dataset_root / "processed" / "cohorts"
    create_synthetic_dataset(dataset_root)

    build_cohort_artifacts(
        CohortBuildConfig(
            dataset_root=dataset_root,
            cohorts_root=cohorts_root,
            manifest_path=tmp_path / "reports" / "cohort_manifest.json",
            mimic_first_icu_stay_per_admission=False,
        )
    )

    mimic_rows = read_parquet_rows(cohorts_root / "mimic_icu_stays.parquet")

    assert {row["stay_uid"] for row in mimic_rows} == {
        "mimiciv:300",
        "mimiciv:301",
        "mimiciv:303",
    }


def test_validate_required_columns_reports_missing_columns(tmp_path: Path) -> None:
    dataset_root = tmp_path / "Dataset"
    bad_source = Path("bad.csv")
    write_text(dataset_root / bad_source, "only_column\nvalue\n")

    with pytest.raises(DatasetPathError, match="missing required columns"):
        validate_required_columns(dataset_root, {bad_source: {"required_column"}})
