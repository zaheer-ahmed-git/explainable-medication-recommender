import gzip
import json
from pathlib import Path
from typing import Sequence

import duckdb

from pipeline.eicu_extract import EICU_EXTRACTION_TABLES, build_eicu_extracts
from pipeline.extract_utils import ExtractionBuildConfig
from pipeline.harmonize import (
    PROVENANCE_COLUMNS,
    REQUIRED_HARMONIZED_TABLES,
    HarmonizationBuildConfig,
    build_harmonized_artifacts,
)
from pipeline.mimic_extract import MIMIC_EXTRACTION_TABLES, build_mimic_extracts


def sql_string(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def write_gzip_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, mode="wt", encoding="utf-8", newline="") as file_obj:
        file_obj.write(text)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


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


def parquet_columns(path: Path) -> set[str]:
    with duckdb.connect(database=":memory:") as connection:
        rows = connection.execute(
            f"DESCRIBE SELECT * FROM read_parquet({sql_string(path)})"
        ).fetchall()
    return {str(row[0]) for row in rows}


def write_unified_cohort(path: Path) -> None:
    write_parquet_rows(
        path,
        (
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
        ),
        (
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
        ),
    )


def test_mimic_extract_filters_rows_to_cohort_before_writing(tmp_path: Path) -> None:
    dataset_root = tmp_path / "Dataset"
    cohort_path = tmp_path / "cohorts" / "cohort_stays.parquet"
    output_root = tmp_path / "Dataset" / "processed" / "extracts" / "mimiciv"
    manifest_path = tmp_path / "reports" / "mimic_manifest.json"
    write_unified_cohort(cohort_path)
    write_gzip_text(
        dataset_root / "mimiciv" / "3.1" / "hosp" / "diagnoses_icd.csv.gz",
        "\n".join(
            [
                "subject_id,hadm_id,seq_num,icd_code,icd_version",
                "10,20,1,A41,10",
                "99,999,1,ZZZ,10",
            ]
        )
        + "\n",
    )
    spec = next(
        table
        for table in MIMIC_EXTRACTION_TABLES
        if table.table_name == "mimic_diagnoses_icd"
    )

    manifest = build_mimic_extracts(
        ExtractionBuildConfig(
            source="mimiciv",
            source_version="3.1",
            dataset_root=dataset_root,
            cohort_path=cohort_path,
            output_root=output_root,
            manifest_path=manifest_path,
            enforce_table_gates=False,
        ),
        table_specs=(spec,),
    )

    rows = read_parquet_rows(output_root / "diagnoses_icd.parquet")
    manifest_text = manifest_path.read_text(encoding="utf-8")

    assert len(rows) == 1
    assert rows[0]["stay_uid"] == "mimiciv:30"
    assert rows[0]["icd_code"] == "A41"
    assert manifest["data_safety"]["contains_patient_rows"] is False
    assert "mimiciv:30" not in manifest_text
    assert "A41" not in manifest_text


def test_mimic_chartevents_extract_filters_to_vital_itemids(tmp_path: Path) -> None:
    dataset_root = tmp_path / "Dataset"
    cohort_path = tmp_path / "cohorts" / "cohort_stays.parquet"
    output_root = tmp_path / "Dataset" / "processed" / "extracts" / "mimiciv"
    manifest_path = tmp_path / "reports" / "mimic_manifest.json"
    write_unified_cohort(cohort_path)
    write_gzip_text(
        dataset_root / "mimiciv" / "3.1" / "icu" / "chartevents.csv.gz",
        "\n".join(
            [
                "subject_id,hadm_id,stay_id,charttime,storetime,itemid,value,valuenum,valueuom,warning",
                # cohort stay, vital itemid -> kept
                "10,20,30,2026-01-01 01:00:00,2026-01-01 01:05:00,220045,80,80,bpm,0",
                # cohort stay, non-vital itemid -> dropped by source_row_filter
                "10,20,30,2026-01-01 01:00:00,2026-01-01 01:05:00,999999,5,5,unit,0",
                # non-cohort stay, vital itemid -> dropped by cohort join
                "99,999,999,2026-01-01 01:00:00,2026-01-01 01:05:00,220045,90,90,bpm,0",
            ]
        )
        + "\n",
    )
    spec = next(
        table
        for table in MIMIC_EXTRACTION_TABLES
        if table.table_name == "mimic_chartevents"
    )

    build_mimic_extracts(
        ExtractionBuildConfig(
            source="mimiciv",
            source_version="3.1",
            dataset_root=dataset_root,
            cohort_path=cohort_path,
            output_root=output_root,
            manifest_path=manifest_path,
            enforce_table_gates=False,
        ),
        table_specs=(spec,),
    )

    rows = read_parquet_rows(output_root / "chartevents.parquet")

    assert len(rows) == 1
    assert rows[0]["stay_uid"] == "mimiciv:30"
    assert rows[0]["itemid"] == "220045"


def test_eicu_extract_filters_rows_to_unit_stay_before_writing(tmp_path: Path) -> None:
    dataset_root = tmp_path / "Dataset"
    cohort_path = tmp_path / "cohorts" / "cohort_stays.parquet"
    output_root = tmp_path / "Dataset" / "processed" / "extracts" / "eicu_crd"
    manifest_path = tmp_path / "reports" / "eicu_manifest.json"
    write_unified_cohort(cohort_path)
    write_gzip_text(
        dataset_root / "eicu-crd" / "2.0" / "medication.csv.gz",
        "\n".join(
            [
                "medicationid,patientunitstayid,drugorderoffset,drugstartoffset,drugivadmixture,drugordercancelled,drugname,drughiclseqno,dosage,routeadmin,frequency,loadingdose,prn,drugstopoffset,gtc",
                "700,400,10,20,No,No,synthetic med,123,5 mg,IV,q24h,No,No,60,GTC1",
                "701,999,10,20,No,No,other med,999,5 mg,IV,q24h,No,No,60,GTC9",
            ]
        )
        + "\n",
    )
    spec = next(
        table
        for table in EICU_EXTRACTION_TABLES
        if table.table_name == "eicu_medication"
    )

    build_eicu_extracts(
        ExtractionBuildConfig(
            source="eicu_crd",
            source_version="2.0",
            dataset_root=dataset_root,
            cohort_path=cohort_path,
            output_root=output_root,
            manifest_path=manifest_path,
            enforce_table_gates=False,
        ),
        table_specs=(spec,),
    )

    rows = read_parquet_rows(output_root / "medication.parquet")

    assert len(rows) == 1
    assert rows[0]["stay_uid"] == "eicu_crd:400"
    assert rows[0]["medicationid"] == "700"


def test_extraction_skips_blocked_tables_from_gate_reports(tmp_path: Path) -> None:
    dataset_root = tmp_path / "Dataset"
    cohort_path = tmp_path / "cohorts" / "cohort_stays.parquet"
    output_root = tmp_path / "Dataset" / "processed" / "extracts" / "mimiciv"
    manifest_path = tmp_path / "reports" / "mimic_manifest.json"
    quality_path = tmp_path / "reports" / "quality_profile.json"
    integrity_path = tmp_path / "reports" / "source_integrity_failed_tables.json"
    write_unified_cohort(cohort_path)
    write_json(
        quality_path,
        {"tables": [{"table_name": "mimic_inputevents", "status": "scan_failed"}]},
    )
    write_json(
        integrity_path,
        {
            "results": [
                {
                    "table_name": "mimic_inputevents",
                    "relative_path": "mimiciv/3.1/icu/inputevents.csv.gz",
                    "checksum_status": "mismatched",
                    "gzip_integrity": {"status": "failed"},
                }
            ]
        },
    )
    spec = next(
        table
        for table in MIMIC_EXTRACTION_TABLES
        if table.table_name == "mimic_inputevents"
    )

    manifest = build_mimic_extracts(
        ExtractionBuildConfig(
            source="mimiciv",
            source_version="3.1",
            dataset_root=dataset_root,
            cohort_path=cohort_path,
            output_root=output_root,
            manifest_path=manifest_path,
            quality_profile_path=quality_path,
            integrity_report_path=integrity_path,
            enforce_table_gates=True,
        ),
        table_specs=(spec,),
    )

    table = manifest["tables"][0]
    assert table["status"] == "skipped_table_gate"
    assert table["gate"]["quality"]["quality_status"] == "scan_failed"
    assert table["gate"]["integrity"]["checksum_status"] == "mismatched"
    assert not (output_root / "inputevents.parquet").exists()


def test_harmonize_missing_mapping_resources_writes_failure_manifest(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "reports" / "harmonization_manifest.json"

    manifest = build_harmonized_artifacts(
        HarmonizationBuildConfig(
            cohort_path=tmp_path / "cohorts" / "cohort_stays.parquet",
            extracts_root=tmp_path / "Dataset" / "processed" / "extracts",
            harmonized_root=tmp_path / "Dataset" / "processed" / "harmonized",
            mapping_root=tmp_path / "Dataset" / "mappings",
            manifest_path=manifest_path,
            coverage_path=tmp_path / "reports" / "coverage.json",
            unmapped_path=tmp_path / "reports" / "unmapped.json",
        )
    )
    manifest_text = manifest_path.read_text(encoding="utf-8")

    assert manifest["status"] == "failed_missing_mapping_resources"
    assert all(
        resource["status"] == "missing" for resource in manifest["mapping_resources"]
    )
    assert "cohort_stays.parquet" in manifest_text
    assert "patient-a" not in manifest_text


def test_harmonize_builds_conditions_and_mapped_medications(tmp_path: Path) -> None:
    cohort_path = tmp_path / "cohorts" / "cohort_stays.parquet"
    extracts_root = tmp_path / "Dataset" / "processed" / "extracts"
    mapping_root = tmp_path / "Dataset" / "mappings"
    harmonized_root = tmp_path / "Dataset" / "processed" / "harmonized"
    reports_root = tmp_path / "reports"
    write_unified_cohort(cohort_path)
    write_parquet_rows(
        extracts_root / "mimiciv" / "diagnoses_icd.parquet",
        (
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
        ),
        (
            (
                "mimiciv",
                "3.1",
                "mimiciv:10",
                "mimiciv:20",
                "mimiciv:30",
                "10",
                "20",
                "30",
                "test",
                "1",
                "A41",
                "10",
            ),
        ),
    )
    write_parquet_rows(
        extracts_root / "eicu_crd" / "medication.parquet",
        (
            "source",
            "source_version",
            "patient_uid",
            "encounter_uid",
            "stay_uid",
            "source_patient_id",
            "source_encounter_id",
            "source_stay_id",
            "extraction_version",
            "medicationid",
            "drugstartoffset",
            "drugstopoffset",
            "drugordercancelled",
            "drugname",
            "drughiclseqno",
            "gtc",
            "routeadmin",
            "dosage",
        ),
        (
            (
                "eicu_crd",
                "2.0",
                "eicu_crd:patient-a",
                "eicu_crd:500",
                "eicu_crd:400",
                "patient-a",
                "500",
                "400",
                "test",
                "700",
                "20",
                "60",
                "No",
                "synthetic med",
                "123",
                "GTC1",
                "IV",
                "5 mg",
            ),
        ),
    )
    write_text(
        mapping_root / "medications" / "mimic_ndc_rxnorm_atc.csv",
        "ndc,rxcui,ingredient_name,rxnorm_name,atc_code,atc_level\n",
    )
    write_text(
        mapping_root / "medications" / "eicu_drug_rxnorm_atc.csv",
        "\n".join(
            [
                "drughiclseqno,gtc,drug_name,rxcui,ingredient_name,rxnorm_name,atc_code,atc_level",
                "123,GTC1,synthetic med,RX1,synthetic ingredient,synthetic rx,ATC1,4",
            ]
        )
        + "\n",
    )

    manifest = build_harmonized_artifacts(
        HarmonizationBuildConfig(
            cohort_path=cohort_path,
            extracts_root=extracts_root,
            harmonized_root=harmonized_root,
            mapping_root=mapping_root,
            manifest_path=reports_root / "harmonization_manifest.json",
            coverage_path=reports_root / "harmonization_coverage.json",
            unmapped_path=reports_root / "unmapped_concepts.json",
            domain_materialization_batches=2,
        )
    )

    conditions = read_parquet_rows(harmonized_root / "conditions.parquet")
    medications = read_parquet_rows(harmonized_root / "medications.parquet")
    coverage = json.loads(
        (reports_root / "harmonization_coverage.json").read_text(encoding="utf-8")
    )

    assert manifest["status"] == "completed"
    assert conditions[0]["condition_token"] == "icd10:A41".lower()
    assert medications[0]["mapping_status"] == "mapped_rxnorm_or_atc"
    assert medications[0]["rxcui"] == "RX1"
    assert coverage["data_safety"]["contains_patient_rows"] is False


def test_harmonize_filters_cancelled_eicu_orders_and_deduplicates_events(
    tmp_path: Path,
) -> None:
    cohort_path = tmp_path / "cohorts" / "cohort_stays.parquet"
    extracts_root = tmp_path / "Dataset" / "processed" / "extracts"
    mapping_root = tmp_path / "Dataset" / "mappings"
    harmonized_root = tmp_path / "Dataset" / "processed" / "harmonized"
    reports_root = tmp_path / "reports"
    write_unified_cohort(cohort_path)
    write_parquet_rows(
        extracts_root / "eicu_crd" / "medication.parquet",
        (
            "source",
            "source_version",
            "patient_uid",
            "encounter_uid",
            "stay_uid",
            "source_patient_id",
            "source_encounter_id",
            "source_stay_id",
            "extraction_version",
            "medicationid",
            "drugstartoffset",
            "drugstopoffset",
            "drugordercancelled",
            "drugname",
            "drughiclseqno",
            "gtc",
            "routeadmin",
            "dosage",
        ),
        (
            (
                "eicu_crd",
                "2.0",
                "eicu_crd:patient-a",
                "eicu_crd:500",
                "eicu_crd:400",
                "patient-a",
                "500",
                "400",
                "test",
                "700",
                "20",
                "60",
                "No",
                "exact med",
                "123",
                "GTC1",
                "IV",
                "5 mg",
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
                "test",
                "700",
                "20",
                "60",
                "No",
                "exact med",
                "123",
                "GTC1",
                "IV",
                "5 mg",
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
                "test",
                "701",
                "25",
                "70",
                "Yes",
                "cancelled med",
                "123",
                "GTC1",
                "IV",
                "5 mg",
            ),
        ),
    )
    write_text(
        mapping_root / "medications" / "mimic_ndc_rxnorm_atc.csv",
        "ndc,rxcui,ingredient_name,rxnorm_name,atc_code,atc_level\n",
    )
    write_text(
        mapping_root / "medications" / "eicu_drug_rxnorm_atc.csv",
        "\n".join(
            [
                "drughiclseqno,gtc,drug_name,rxcui,ingredient_name,rxnorm_name,atc_code,atc_level",
                "123,GTC1,exact med,RX_EXACT,exact ingredient,exact rx,ATC_EXACT,3",
                "123,GTC1,cancelled med,RX_CANCEL,cancel ingredient,cancel rx,ATC_CANCEL,3",
            ]
        )
        + "\n",
    )

    manifest = build_harmonized_artifacts(
        HarmonizationBuildConfig(
            cohort_path=cohort_path,
            extracts_root=extracts_root,
            harmonized_root=harmonized_root,
            mapping_root=mapping_root,
            manifest_path=reports_root / "harmonization_manifest.json",
            coverage_path=reports_root / "harmonization_coverage.json",
            unmapped_path=reports_root / "unmapped_concepts.json",
            domain_materialization_batches=2,
        )
    )

    medications = read_parquet_rows(harmonized_root / "medications.parquet")
    medication_dedup = [
        row
        for row in manifest["cleanup"]["deduplication"]
        if row["domain"] == "medications" and row["source"] == "eicu_crd"
    ][0]
    cancellation = manifest["cleanup"]["cancelled_medication_orders"][0]

    assert manifest["status"] == "completed"
    assert len(medications) == 1
    assert medications[0]["source_event_id"] == "700"
    assert medications[0]["rxcui"] == "RX_EXACT"
    assert cancellation["medication_extract_row_count"] == 3
    assert cancellation["cancelled_order_row_count"] == 1
    assert medication_dedup["input_row_count"] == 2
    assert medication_dedup["deduplicated_row_count"] == 1
    assert medication_dedup["duplicate_excess_rows"] == 1


def test_harmonize_writes_all_milestone5_artifacts_and_reports(
    tmp_path: Path,
) -> None:
    cohort_path = tmp_path / "cohorts" / "cohort_stays.parquet"
    extracts_root = tmp_path / "Dataset" / "processed" / "extracts"
    mapping_root = tmp_path / "Dataset" / "mappings"
    harmonized_root = tmp_path / "Dataset" / "processed" / "harmonized"
    reports_root = tmp_path / "reports"
    write_unified_cohort(cohort_path)
    write_parquet_rows(
        extracts_root / "mimiciv" / "diagnoses_icd.parquet",
        (
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
        ),
        (
            (
                "mimiciv",
                "3.1",
                "mimiciv:10",
                "mimiciv:20",
                "mimiciv:30",
                "10",
                "20",
                "30",
                "test-extract",
                "1",
                "A41",
                "10",
            ),
        ),
    )
    write_parquet_rows(
        extracts_root / "mimiciv" / "prescriptions.parquet",
        (
            "source",
            "source_version",
            "patient_uid",
            "encounter_uid",
            "stay_uid",
            "source_patient_id",
            "source_encounter_id",
            "source_stay_id",
            "extraction_version",
            "pharmacy_id",
            "starttime",
            "stoptime",
            "drug",
            "ndc",
            "route",
            "dose_val_rx",
            "dose_unit_rx",
        ),
        (
            (
                "mimiciv",
                "3.1",
                "mimiciv:10",
                "mimiciv:20",
                "mimiciv:30",
                "10",
                "20",
                "30",
                "test-extract",
                "900",
                "2026-01-01",
                "2026-01-02",
                "synthetic mimic med",
                "111",
                "PO",
                "1",
                "tab",
            ),
        ),
    )
    write_parquet_rows(
        extracts_root / "mimiciv" / "labevents.parquet",
        (
            "source",
            "source_version",
            "patient_uid",
            "encounter_uid",
            "stay_uid",
            "source_patient_id",
            "source_encounter_id",
            "source_stay_id",
            "extraction_version",
            "labevent_id",
            "itemid",
            "charttime",
            "value",
            "valuenum",
            "valueuom",
            "ref_range_lower",
            "ref_range_upper",
            "flag",
        ),
        (
            (
                "mimiciv",
                "3.1",
                "mimiciv:10",
                "mimiciv:20",
                "mimiciv:30",
                "10",
                "20",
                "30",
                "test-extract",
                "lab-1",
                "501",
                "2026-01-01",
                "1.2",
                "1.2",
                "mg/dL",
                "0.5",
                "1.4",
                "normal",
            ),
        ),
    )
    write_parquet_rows(
        extracts_root / "mimiciv" / "d_labitems.parquet",
        ("source", "source_version", "extraction_version", "itemid", "label"),
        (("mimiciv", "3.1", "test-extract", "501", "Creatinine"),),
    )
    write_parquet_rows(
        extracts_root / "mimiciv" / "procedures_icd.parquet",
        (
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
            "chartdate",
            "icd_code",
            "icd_version",
        ),
        (
            (
                "mimiciv",
                "3.1",
                "mimiciv:10",
                "mimiciv:20",
                "mimiciv:30",
                "10",
                "20",
                "30",
                "test-extract",
                "1",
                "2026-01-01",
                "5A1955Z",
                "10",
            ),
        ),
    )
    write_parquet_rows(
        extracts_root / "mimiciv" / "procedureevents.parquet",
        (
            "source",
            "source_version",
            "patient_uid",
            "encounter_uid",
            "stay_uid",
            "source_patient_id",
            "source_encounter_id",
            "source_stay_id",
            "extraction_version",
            "orderid",
            "starttime",
            "endtime",
            "itemid",
            "value",
            "valueuom",
            "ordercategoryname",
            "ordercategorydescription",
            "statusdescription",
        ),
        (
            (
                "mimiciv",
                "3.1",
                "mimiciv:10",
                "mimiciv:20",
                "mimiciv:30",
                "10",
                "20",
                "30",
                "test-extract",
                "proc-order-1",
                "2026-01-01",
                "2026-01-02",
                "800",
                "2",
                "hour",
                "synthetic procedure",
                "synthetic procedure detail",
                "finished",
            ),
        ),
    )
    write_parquet_rows(
        extracts_root / "mimiciv" / "d_items.parquet",
        ("source", "source_version", "extraction_version", "itemid", "label"),
        (("mimiciv", "3.1", "test-extract", "800", "Synthetic procedure"),),
    )
    write_parquet_rows(
        extracts_root / "eicu_crd" / "lab.parquet",
        (
            "source",
            "source_version",
            "patient_uid",
            "encounter_uid",
            "stay_uid",
            "source_patient_id",
            "source_encounter_id",
            "source_stay_id",
            "extraction_version",
            "labid",
            "labresultoffset",
            "labtypeid",
            "labname",
            "labresult",
            "labresulttext",
            "labmeasurenamesystem",
            "labmeasurenameinterface",
        ),
        (
            (
                "eicu_crd",
                "2.0",
                "eicu_crd:patient-a",
                "eicu_crd:500",
                "eicu_crd:400",
                "patient-a",
                "500",
                "400",
                "test-extract",
                "lab-2",
                "60",
                "chem",
                "creatinine",
                "1.1",
                "1.1",
                "mg/dL",
                "",
            ),
        ),
    )
    write_parquet_rows(
        extracts_root / "eicu_crd" / "vital_periodic.parquet",
        (
            "source",
            "source_version",
            "patient_uid",
            "encounter_uid",
            "stay_uid",
            "source_patient_id",
            "source_encounter_id",
            "source_stay_id",
            "extraction_version",
            "vitalperiodicid",
            "observationoffset",
            "temperature",
            "sao2",
            "heartrate",
            "respiration",
            "systemicsystolic",
            "systemicdiastolic",
            "systemicmean",
        ),
        (
            (
                "eicu_crd",
                "2.0",
                "eicu_crd:patient-a",
                "eicu_crd:500",
                "eicu_crd:400",
                "patient-a",
                "500",
                "400",
                "test-extract",
                "vp-1",
                "30",
                "37.0",
                "98",
                "80",
                "18",
                "120",
                "70",
                "86",
            ),
        ),
    )
    write_parquet_rows(
        extracts_root / "eicu_crd" / "vital_aperiodic.parquet",
        (
            "source",
            "source_version",
            "patient_uid",
            "encounter_uid",
            "stay_uid",
            "source_patient_id",
            "source_encounter_id",
            "source_stay_id",
            "extraction_version",
            "vitalaperiodicid",
            "observationoffset",
            "noninvasivesystolic",
            "noninvasivediastolic",
            "noninvasivemean",
        ),
        (
            (
                "eicu_crd",
                "2.0",
                "eicu_crd:patient-a",
                "eicu_crd:500",
                "eicu_crd:400",
                "patient-a",
                "500",
                "400",
                "test-extract",
                "va-1",
                "45",
                "118",
                "68",
                "84",
            ),
        ),
    )
    write_parquet_rows(
        extracts_root / "eicu_crd" / "allergy.parquet",
        (
            "source",
            "source_version",
            "patient_uid",
            "encounter_uid",
            "stay_uid",
            "source_patient_id",
            "source_encounter_id",
            "source_stay_id",
            "extraction_version",
            "allergyid",
            "allergyoffset",
            "allergyenteredoffset",
            "rxincluded",
            "writtenineicu",
            "drugname",
            "allergytype",
            "allergyname",
            "drughiclseqno",
        ),
        (
            (
                "eicu_crd",
                "2.0",
                "eicu_crd:patient-a",
                "eicu_crd:500",
                "eicu_crd:400",
                "patient-a",
                "500",
                "400",
                "test-extract",
                "allergy-1",
                "10",
                "5",
                "1",
                "1",
                "synthetic allergen",
                "drug",
                "synthetic allergen",
                "321",
            ),
        ),
    )
    write_parquet_rows(
        extracts_root / "eicu_crd" / "treatment.parquet",
        (
            "source",
            "source_version",
            "patient_uid",
            "encounter_uid",
            "stay_uid",
            "source_patient_id",
            "source_encounter_id",
            "source_stay_id",
            "extraction_version",
            "treatmentid",
            "treatmentoffset",
            "treatmentstring",
            "activeupondischarge",
        ),
        (
            (
                "eicu_crd",
                "2.0",
                "eicu_crd:patient-a",
                "eicu_crd:500",
                "eicu_crd:400",
                "patient-a",
                "500",
                "400",
                "test-extract",
                "treat-1",
                "120",
                "synthetic treatment",
                "False",
            ),
        ),
    )
    write_parquet_rows(
        extracts_root / "eicu_crd" / "infusion_drug.parquet",
        (
            "source",
            "source_version",
            "patient_uid",
            "encounter_uid",
            "stay_uid",
            "source_patient_id",
            "source_encounter_id",
            "source_stay_id",
            "extraction_version",
            "infusiondrugid",
            "infusionoffset",
            "drugname",
            "drugrate",
            "infusionrate",
            "drugamount",
            "volumeoffluid",
            "patientweight",
        ),
        (
            (
                "eicu_crd",
                "2.0",
                "eicu_crd:patient-a",
                "eicu_crd:500",
                "eicu_crd:400",
                "patient-a",
                "500",
                "400",
                "test-extract",
                "inf-1",
                "130",
                "synthetic infusion",
                "2",
                "",
                "10",
                "50",
                "70",
            ),
        ),
    )
    write_parquet_rows(
        extracts_root / "eicu_crd" / "apache_aps_var.parquet",
        (
            "source",
            "source_version",
            "patient_uid",
            "encounter_uid",
            "stay_uid",
            "source_patient_id",
            "source_encounter_id",
            "source_stay_id",
            "extraction_version",
            "apacheapsvarid",
            "intubated",
            "vent",
            "dialysis",
        ),
        (
            (
                "eicu_crd",
                "2.0",
                "eicu_crd:patient-a",
                "eicu_crd:500",
                "eicu_crd:400",
                "patient-a",
                "500",
                "400",
                "test-extract",
                "apache-1",
                "1",
                "0",
                "1",
            ),
        ),
    )
    write_text(
        mapping_root / "medications" / "mimic_ndc_rxnorm_atc.csv",
        "\n".join(
            [
                "ndc,rxcui,ingredient_name,rxnorm_name,atc_code,atc_level",
                "111,RXMIMIC,mimic ingredient,mimic rx,ATC_MIMIC,3",
            ]
        )
        + "\n",
    )
    write_text(
        mapping_root / "medications" / "eicu_drug_rxnorm_atc.csv",
        "drughiclseqno,gtc,drug_name,rxcui,ingredient_name,rxnorm_name,atc_code,atc_level\n",
    )

    manifest = build_harmonized_artifacts(
        HarmonizationBuildConfig(
            cohort_path=cohort_path,
            extracts_root=extracts_root,
            harmonized_root=harmonized_root,
            mapping_root=mapping_root,
            manifest_path=reports_root / "harmonization_manifest.json",
            coverage_path=reports_root / "harmonization_coverage.json",
            unmapped_path=reports_root / "unmapped_concepts.json",
            domain_materialization_batches=2,
        )
    )

    required = set(REQUIRED_HARMONIZED_TABLES)
    assert manifest["status"] == "completed"
    assert required == set(manifest["artifacts"])
    table_records = {row["table_name"]: row for row in manifest["tables"]}
    assert table_records["labs"]["build_strategy"] == "split_query_hash_batches"
    assert table_records["labs"]["batch_count"] == 2
    assert table_records["vitals"]["build_strategy"] == "split_query_hash_batches"
    assert table_records["vitals"]["batch_count"] == 2
    assert manifest["coverage_path"] == str(
        reports_root / "harmonization_coverage.json"
    )
    assert manifest["unmapped_path"] == str(reports_root / "unmapped_concepts.json")
    for table_name in required:
        artifact_path = Path(manifest["artifacts"][table_name])
        assert artifact_path.exists()
        columns = parquet_columns(artifact_path)
        assert "source" in columns
        assert set(PROVENANCE_COLUMNS) <= columns

    coverage = json.loads(
        (reports_root / "harmonization_coverage.json").read_text(encoding="utf-8")
    )
    unmapped = json.loads(
        (reports_root / "unmapped_concepts.json").read_text(encoding="utf-8")
    )
    coverage_domains = {row["domain"] for row in coverage["coverage"]}
    artifact_domains = {row["domain"] for row in coverage["artifacts"]}
    unit_domains = {row["domain"] for row in coverage["unit_compatibility"]}
    event_types = {
        row["event_type"]
        for row in read_parquet_rows(harmonized_root / "temporal_events.parquet")
    }

    assert required <= coverage_domains
    assert required == artifact_domains
    assert {"labs", "vitals"} <= unit_domains
    assert {
        "condition",
        "medication",
        "lab",
        "vital",
        "allergy",
        "intervention",
    } <= event_types
    assert unmapped["data_safety"]["contains_patient_rows"] is False
    assert unmapped["data_safety"]["no_source_value_samples"] is True
    assert all("sample" not in row for row in unmapped["unmapped"])


def test_harmonize_maps_mimic_chartevents_vitals(tmp_path: Path) -> None:
    cohort_path = tmp_path / "cohorts" / "cohort_stays.parquet"
    extracts_root = tmp_path / "Dataset" / "processed" / "extracts"
    mapping_root = tmp_path / "Dataset" / "mappings"
    harmonized_root = tmp_path / "Dataset" / "processed" / "harmonized"
    reports_root = tmp_path / "reports"
    write_unified_cohort(cohort_path)
    write_parquet_rows(
        extracts_root / "mimiciv" / "chartevents.parquet",
        (
            "source",
            "source_version",
            "patient_uid",
            "encounter_uid",
            "stay_uid",
            "source_patient_id",
            "source_encounter_id",
            "source_stay_id",
            "extraction_version",
            "charttime",
            "storetime",
            "itemid",
            "value",
            "valuenum",
            "valueuom",
            "warning",
        ),
        (
            (
                "mimiciv",
                "3.1",
                "mimiciv:10",
                "mimiciv:20",
                "mimiciv:30",
                "10",
                "20",
                "30",
                "test-extract",
                "2026-01-01 01:00:00",
                "2026-01-01 01:05:00",
                "220045",
                "80",
                "80",
                "bpm",
                "0",
            ),
        ),
    )
    write_parquet_rows(
        extracts_root / "mimiciv" / "d_items.parquet",
        ("source", "source_version", "extraction_version", "itemid", "label"),
        (("mimiciv", "3.1", "test-extract", "220045", "Heart Rate"),),
    )
    write_text(
        mapping_root / "medications" / "mimic_ndc_rxnorm_atc.csv",
        "ndc,rxcui,ingredient_name,rxnorm_name,atc_code,atc_level\n",
    )
    write_text(
        mapping_root / "medications" / "eicu_drug_rxnorm_atc.csv",
        "drughiclseqno,gtc,drug_name,rxcui,ingredient_name,rxnorm_name,atc_code,atc_level\n",
    )

    manifest = build_harmonized_artifacts(
        HarmonizationBuildConfig(
            cohort_path=cohort_path,
            extracts_root=extracts_root,
            harmonized_root=harmonized_root,
            mapping_root=mapping_root,
            manifest_path=reports_root / "harmonization_manifest.json",
            coverage_path=reports_root / "harmonization_coverage.json",
            unmapped_path=reports_root / "unmapped_concepts.json",
        )
    )

    assert manifest["status"].startswith("completed")
    vitals = read_parquet_rows(harmonized_root / "vitals.parquet")
    mimic_vitals = [row for row in vitals if row["source"] == "mimiciv"]
    assert len(mimic_vitals) == 1
    vital = mimic_vitals[0]
    assert vital["normalized_vital_token"] == "heart_rate"
    assert vital["source_vital_name"] == "Heart Rate"
    assert vital["value_numeric"] == 80.0
    assert vital["mapping_status"] == "mapped_vital_itemid"
    assert vital["source_table"] == "mimic_chartevents"


def test_harmonize_prefers_exact_eicu_mapping_concept(tmp_path: Path) -> None:
    cohort_path = tmp_path / "cohorts" / "cohort_stays.parquet"
    extracts_root = tmp_path / "Dataset" / "processed" / "extracts"
    mapping_root = tmp_path / "Dataset" / "mappings"
    harmonized_root = tmp_path / "Dataset" / "processed" / "harmonized"
    reports_root = tmp_path / "reports"
    write_unified_cohort(cohort_path)
    write_parquet_rows(
        extracts_root / "eicu_crd" / "medication.parquet",
        (
            "source",
            "source_version",
            "patient_uid",
            "encounter_uid",
            "stay_uid",
            "source_patient_id",
            "source_encounter_id",
            "source_stay_id",
            "extraction_version",
            "medicationid",
            "drugstartoffset",
            "drugstopoffset",
            "drugordercancelled",
            "drugname",
            "drughiclseqno",
            "gtc",
            "routeadmin",
            "dosage",
        ),
        (
            (
                "eicu_crd",
                "2.0",
                "eicu_crd:patient-a",
                "eicu_crd:500",
                "eicu_crd:400",
                "patient-a",
                "500",
                "400",
                "test",
                "700",
                "20",
                "60",
                "No",
                "exact med",
                "123",
                "GTC1",
                "IV",
                "5 mg",
            ),
        ),
    )
    write_text(
        mapping_root / "medications" / "mimic_ndc_rxnorm_atc.csv",
        "ndc,rxcui,ingredient_name,rxnorm_name,atc_code,atc_level\n",
    )
    write_text(
        mapping_root / "medications" / "eicu_drug_rxnorm_atc.csv",
        "\n".join(
            [
                "drughiclseqno,gtc,drug_name,rxcui,ingredient_name,rxnorm_name,atc_code,atc_level",
                "123,GTC1,exact med,RX_EXACT,exact ingredient,exact rx,ATC_EXACT,3",
                "123,GTC9,other med,RX_HICL,hicl ingredient,hicl rx,ATC_HICL,3",
                ",GTC1,gtc med,RX_GTC,gtc ingredient,gtc rx,ATC_GTC,3",
            ]
        )
        + "\n",
    )

    manifest = build_harmonized_artifacts(
        HarmonizationBuildConfig(
            cohort_path=cohort_path,
            extracts_root=extracts_root,
            harmonized_root=harmonized_root,
            mapping_root=mapping_root,
            manifest_path=reports_root / "harmonization_manifest.json",
            coverage_path=reports_root / "harmonization_coverage.json",
            unmapped_path=reports_root / "unmapped_concepts.json",
        )
    )

    medications = read_parquet_rows(harmonized_root / "medications.parquet")

    assert manifest["status"].startswith("completed")
    assert len(medications) == 1
    assert medications[0]["rxcui"] == "RX_EXACT"


def test_harmonize_maps_mimic_prescription_ndc_without_column_ambiguity(
    tmp_path: Path,
) -> None:
    cohort_path = tmp_path / "cohorts" / "cohort_stays.parquet"
    extracts_root = tmp_path / "Dataset" / "processed" / "extracts"
    mapping_root = tmp_path / "Dataset" / "mappings"
    harmonized_root = tmp_path / "Dataset" / "processed" / "harmonized"
    reports_root = tmp_path / "reports"
    write_unified_cohort(cohort_path)
    write_parquet_rows(
        extracts_root / "mimiciv" / "prescriptions.parquet",
        (
            "source",
            "source_version",
            "patient_uid",
            "encounter_uid",
            "stay_uid",
            "source_patient_id",
            "source_encounter_id",
            "source_stay_id",
            "extraction_version",
            "pharmacy_id",
            "starttime",
            "stoptime",
            "drug",
            "ndc",
            "route",
            "dose_val_rx",
            "dose_unit_rx",
        ),
        (
            (
                "mimiciv",
                "3.1",
                "mimiciv:10",
                "mimiciv:20",
                "mimiciv:30",
                "10",
                "20",
                "30",
                "test",
                "900",
                "2026-01-01",
                "2026-01-02",
                "synthetic mimic med",
                "111",
                "PO",
                "1",
                "tab",
            ),
        ),
    )
    write_text(
        mapping_root / "medications" / "mimic_ndc_rxnorm_atc.csv",
        "\n".join(
            [
                "ndc,rxcui,ingredient_name,rxnorm_name,atc_code,atc_level",
                "111,RXMIMIC,mimic ingredient,mimic rx,ATC_MIMIC,3",
            ]
        )
        + "\n",
    )
    write_text(
        mapping_root / "medications" / "eicu_drug_rxnorm_atc.csv",
        "drughiclseqno,gtc,drug_name,rxcui,ingredient_name,rxnorm_name,atc_code,atc_level\n",
    )

    manifest = build_harmonized_artifacts(
        HarmonizationBuildConfig(
            cohort_path=cohort_path,
            extracts_root=extracts_root,
            harmonized_root=harmonized_root,
            mapping_root=mapping_root,
            manifest_path=reports_root / "harmonization_manifest.json",
            coverage_path=reports_root / "harmonization_coverage.json",
            unmapped_path=reports_root / "unmapped_concepts.json",
        )
    )

    medications = read_parquet_rows(harmonized_root / "medications.parquet")

    assert manifest["status"].startswith("completed")
    assert len(medications) == 1
    assert medications[0]["source_code"] == "111"
    assert medications[0]["rxcui"] == "RXMIMIC"
