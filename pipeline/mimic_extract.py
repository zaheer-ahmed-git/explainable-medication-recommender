"""Cohort-filtered MIMIC-IV source extraction for Milestone 5 preparation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from pipeline.config import (
    COHORTS_ROOT,
    DATASET_ROOT,
    EXTRACTS_ROOT,
    MIMIC_CHARTEVENTS_VITAL_ITEMIDS,
    REPORTS_ROOT,
)
from pipeline.extract_utils import (
    DEFAULT_INTEGRITY_REPORT_PATH,
    DEFAULT_QUALITY_PROFILE_PATH,
    ExtractionBuildConfig,
    ExtractionTableSpec,
    build_source_extracts,
    manifest_has_failures,
)


DEFAULT_MIMIC_OUTPUT_ROOT = EXTRACTS_ROOT / "mimiciv"
DEFAULT_MIMIC_MANIFEST_PATH = REPORTS_ROOT / "mimic_extraction_manifest.json"
DEFAULT_COHORT_PATH = COHORTS_ROOT / "cohort_stays.parquet"

# chartevents is the largest ICU table (~330M rows); restrict the extract to the
# curated core-vital itemids so the cohort-filtered artifact stays bounded.
_CHARTEVENTS_ITEMID_LIST = ", ".join(
    f"'{itemid}'" for itemid in sorted(MIMIC_CHARTEVENTS_VITAL_ITEMIDS)
)
MIMIC_CHARTEVENTS_VITAL_ITEMID_FILTER = (
    f"NULLIF(TRIM(CAST(itemid AS VARCHAR)), '') IN ({_CHARTEVENTS_ITEMID_LIST})"
)

MIMIC_EXTRACTION_TABLES: tuple[ExtractionTableSpec, ...] = (
    ExtractionTableSpec(
        table_name="mimic_patients",
        source="mimiciv",
        source_version="3.1",
        relative_path=Path("mimiciv/3.1/hosp/patients.csv.gz"),
        output_name="patients.parquet",
        required_columns=(
            "subject_id",
            "gender",
            "anchor_age",
            "anchor_year",
            "anchor_year_group",
            "dod",
        ),
        selected_columns=(
            "subject_id",
            "gender",
            "anchor_age",
            "anchor_year",
            "anchor_year_group",
            "dod",
        ),
        join_columns=(("subject_id", "source_patient_id"),),
        profile_table_name="mimic_patients",
    ),
    ExtractionTableSpec(
        table_name="mimic_admissions",
        source="mimiciv",
        source_version="3.1",
        relative_path=Path("mimiciv/3.1/hosp/admissions.csv.gz"),
        output_name="admissions.parquet",
        required_columns=(
            "subject_id",
            "hadm_id",
            "admittime",
            "dischtime",
            "deathtime",
            "admission_type",
            "admission_location",
            "discharge_location",
            "insurance",
            "language",
            "marital_status",
            "race",
            "edregtime",
            "edouttime",
            "hospital_expire_flag",
        ),
        selected_columns=(
            "subject_id",
            "hadm_id",
            "admittime",
            "dischtime",
            "deathtime",
            "admission_type",
            "admission_location",
            "discharge_location",
            "insurance",
            "language",
            "marital_status",
            "race",
            "edregtime",
            "edouttime",
            "hospital_expire_flag",
        ),
        join_columns=(
            ("subject_id", "source_patient_id"),
            ("hadm_id", "source_encounter_id"),
        ),
        profile_table_name="mimic_admissions",
    ),
    ExtractionTableSpec(
        table_name="mimic_icustays",
        source="mimiciv",
        source_version="3.1",
        relative_path=Path("mimiciv/3.1/icu/icustays.csv.gz"),
        output_name="icustays.parquet",
        required_columns=(
            "subject_id",
            "hadm_id",
            "stay_id",
            "first_careunit",
            "last_careunit",
            "intime",
            "outtime",
            "los",
        ),
        selected_columns=(
            "subject_id",
            "hadm_id",
            "stay_id",
            "first_careunit",
            "last_careunit",
            "intime",
            "outtime",
            "los",
        ),
        join_columns=(
            ("subject_id", "source_patient_id"),
            ("hadm_id", "source_encounter_id"),
            ("stay_id", "source_stay_id"),
        ),
        profile_table_name="mimic_icustays",
    ),
    ExtractionTableSpec(
        table_name="mimic_diagnoses_icd",
        source="mimiciv",
        source_version="3.1",
        relative_path=Path("mimiciv/3.1/hosp/diagnoses_icd.csv.gz"),
        output_name="diagnoses_icd.parquet",
        required_columns=(
            "subject_id",
            "hadm_id",
            "seq_num",
            "icd_code",
            "icd_version",
        ),
        selected_columns=(
            "subject_id",
            "hadm_id",
            "seq_num",
            "icd_code",
            "icd_version",
        ),
        join_columns=(
            ("subject_id", "source_patient_id"),
            ("hadm_id", "source_encounter_id"),
        ),
        profile_table_name="mimic_diagnoses_icd",
    ),
    ExtractionTableSpec(
        table_name="mimic_procedures_icd",
        source="mimiciv",
        source_version="3.1",
        relative_path=Path("mimiciv/3.1/hosp/procedures_icd.csv.gz"),
        output_name="procedures_icd.parquet",
        required_columns=(
            "subject_id",
            "hadm_id",
            "seq_num",
            "chartdate",
            "icd_code",
            "icd_version",
        ),
        selected_columns=(
            "subject_id",
            "hadm_id",
            "seq_num",
            "chartdate",
            "icd_code",
            "icd_version",
        ),
        join_columns=(
            ("subject_id", "source_patient_id"),
            ("hadm_id", "source_encounter_id"),
        ),
        profile_table_name="mimic_procedures_icd",
    ),
    ExtractionTableSpec(
        table_name="mimic_prescriptions",
        source="mimiciv",
        source_version="3.1",
        relative_path=Path("mimiciv/3.1/hosp/prescriptions.csv.gz"),
        output_name="prescriptions.parquet",
        required_columns=(
            "subject_id",
            "hadm_id",
            "pharmacy_id",
            "poe_id",
            "poe_seq",
            "starttime",
            "stoptime",
            "drug_type",
            "drug",
            "formulary_drug_cd",
            "gsn",
            "ndc",
            "prod_strength",
            "form_rx",
            "dose_val_rx",
            "dose_unit_rx",
            "form_val_disp",
            "form_unit_disp",
            "doses_per_24_hrs",
            "route",
        ),
        selected_columns=(
            "subject_id",
            "hadm_id",
            "pharmacy_id",
            "poe_id",
            "poe_seq",
            "starttime",
            "stoptime",
            "drug_type",
            "drug",
            "formulary_drug_cd",
            "gsn",
            "ndc",
            "prod_strength",
            "form_rx",
            "dose_val_rx",
            "dose_unit_rx",
            "form_val_disp",
            "form_unit_disp",
            "doses_per_24_hrs",
            "route",
        ),
        join_columns=(
            ("subject_id", "source_patient_id"),
            ("hadm_id", "source_encounter_id"),
        ),
        profile_table_name="mimic_prescriptions",
        requires_integrity_gate=True,
        notes=("provider identifiers and free-text comments are excluded",),
    ),
    ExtractionTableSpec(
        table_name="mimic_labevents",
        source="mimiciv",
        source_version="3.1",
        relative_path=Path("mimiciv/3.1/hosp/labevents.csv.gz"),
        output_name="labevents.parquet",
        required_columns=(
            "labevent_id",
            "subject_id",
            "hadm_id",
            "specimen_id",
            "itemid",
            "charttime",
            "storetime",
            "value",
            "valuenum",
            "valueuom",
            "ref_range_lower",
            "ref_range_upper",
            "flag",
            "priority",
        ),
        selected_columns=(
            "labevent_id",
            "subject_id",
            "hadm_id",
            "specimen_id",
            "itemid",
            "charttime",
            "storetime",
            "value",
            "valuenum",
            "valueuom",
            "ref_range_lower",
            "ref_range_upper",
            "flag",
            "priority",
        ),
        join_columns=(
            ("subject_id", "source_patient_id"),
            ("hadm_id", "source_encounter_id"),
        ),
        profile_table_name="mimic_labevents",
        requires_integrity_gate=True,
        notes=("source comments are excluded from the extract",),
    ),
    ExtractionTableSpec(
        table_name="mimic_d_labitems",
        source="mimiciv",
        source_version="3.1",
        relative_path=Path("mimiciv/3.1/hosp/d_labitems.csv.gz"),
        output_name="d_labitems.parquet",
        required_columns=("itemid", "label", "fluid", "category"),
        selected_columns=("itemid", "label", "fluid", "category"),
        join_columns=(),
        profile_table_name="mimic_d_labitems",
        lookup_table=True,
        notes=("non-patient lookup table",),
    ),
    ExtractionTableSpec(
        table_name="mimic_procedureevents",
        source="mimiciv",
        source_version="3.1",
        relative_path=Path("mimiciv/3.1/icu/procedureevents.csv.gz"),
        output_name="procedureevents.parquet",
        required_columns=(
            "subject_id",
            "hadm_id",
            "stay_id",
            "starttime",
            "endtime",
            "itemid",
            "value",
            "valueuom",
            "orderid",
            "ordercategoryname",
            "ordercategorydescription",
            "statusdescription",
        ),
        selected_columns=(
            "subject_id",
            "hadm_id",
            "stay_id",
            "starttime",
            "endtime",
            "itemid",
            "value",
            "valueuom",
            "orderid",
            "ordercategoryname",
            "ordercategorydescription",
            "statusdescription",
        ),
        join_columns=(
            ("subject_id", "source_patient_id"),
            ("hadm_id", "source_encounter_id"),
            ("stay_id", "source_stay_id"),
        ),
        profile_table_name="mimic_procedureevents",
    ),
    ExtractionTableSpec(
        table_name="mimic_inputevents",
        source="mimiciv",
        source_version="3.1",
        relative_path=Path("mimiciv/3.1/icu/inputevents.csv.gz"),
        output_name="inputevents.parquet",
        required_columns=(
            "subject_id",
            "hadm_id",
            "stay_id",
            "starttime",
            "endtime",
            "itemid",
            "amount",
            "amountuom",
            "rate",
            "rateuom",
            "orderid",
            "ordercategoryname",
            "statusdescription",
        ),
        selected_columns=(
            "subject_id",
            "hadm_id",
            "stay_id",
            "starttime",
            "endtime",
            "itemid",
            "amount",
            "amountuom",
            "rate",
            "rateuom",
            "orderid",
            "ordercategoryname",
            "statusdescription",
        ),
        join_columns=(
            ("subject_id", "source_patient_id"),
            ("hadm_id", "source_encounter_id"),
            ("stay_id", "source_stay_id"),
        ),
        profile_table_name="mimic_inputevents",
        requires_integrity_gate=True,
        notes=("currently skipped unless quality and integrity gates pass",),
    ),
    ExtractionTableSpec(
        table_name="mimic_chartevents",
        source="mimiciv",
        source_version="3.1",
        relative_path=Path("mimiciv/3.1/icu/chartevents.csv.gz"),
        output_name="chartevents.parquet",
        required_columns=(
            "subject_id",
            "hadm_id",
            "stay_id",
            "charttime",
            "storetime",
            "itemid",
            "value",
            "valuenum",
            "valueuom",
            "warning",
        ),
        selected_columns=(
            "charttime",
            "storetime",
            "itemid",
            "value",
            "valuenum",
            "valueuom",
            "warning",
        ),
        join_columns=(
            ("subject_id", "source_patient_id"),
            ("hadm_id", "source_encounter_id"),
            ("stay_id", "source_stay_id"),
        ),
        profile_table_name="mimic_chartevents",
        requires_integrity_gate=True,
        source_row_filter=MIMIC_CHARTEVENTS_VITAL_ITEMID_FILTER,
        notes=(
            "charted vitals only; restricted to curated core-vital itemids",
            "skipped unless quality and integrity gates pass",
        ),
    ),
    ExtractionTableSpec(
        table_name="mimic_d_items",
        source="mimiciv",
        source_version="3.1",
        relative_path=Path("mimiciv/3.1/icu/d_items.csv.gz"),
        output_name="d_items.parquet",
        required_columns=(
            "itemid",
            "label",
            "abbreviation",
            "linksto",
            "category",
            "unitname",
            "param_type",
        ),
        selected_columns=(
            "itemid",
            "label",
            "abbreviation",
            "linksto",
            "category",
            "unitname",
            "param_type",
        ),
        join_columns=(),
        profile_table_name="mimic_d_items",
        lookup_table=True,
        notes=("non-patient lookup table",),
    ),
)


def build_mimic_extracts(
    config: ExtractionBuildConfig,
    table_specs: Sequence[ExtractionTableSpec] = MIMIC_EXTRACTION_TABLES,
) -> dict[str, object]:
    """Build MIMIC-IV cohort-filtered extraction artifacts."""

    return build_source_extracts(config, table_specs)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build cohort-filtered MIMIC-IV extraction artifacts.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DATASET_ROOT,
        help="Dataset root. Defaults to the configured Dataset directory.",
    )
    parser.add_argument(
        "--cohort-path",
        type=Path,
        default=DEFAULT_COHORT_PATH,
        help="Unified cohort Parquet artifact from pipeline.cohort.",
    )
    parser.add_argument(
        "--extracts-root",
        type=Path,
        default=DEFAULT_MIMIC_OUTPUT_ROOT,
        help="Output directory for local MIMIC extract Parquet files.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MIMIC_MANIFEST_PATH,
        help="Output path for the aggregate extraction manifest.",
    )
    parser.add_argument(
        "--quality-profile",
        type=Path,
        default=DEFAULT_QUALITY_PROFILE_PATH,
        help="Aggregate quality profile report used for table gates.",
    )
    parser.add_argument(
        "--integrity-report",
        type=Path,
        default=DEFAULT_INTEGRITY_REPORT_PATH,
        help="Source integrity report used for table gates.",
    )
    parser.add_argument(
        "--no-table-gates",
        action="store_true",
        help="Disable quality/integrity gates; intended only for synthetic tests.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_mimic_extracts(
        ExtractionBuildConfig(
            source="mimiciv",
            source_version="3.1",
            dataset_root=args.dataset_root,
            cohort_path=args.cohort_path,
            output_root=args.extracts_root,
            manifest_path=args.manifest,
            quality_profile_path=args.quality_profile,
            integrity_report_path=args.integrity_report,
            enforce_table_gates=not args.no_table_gates,
        )
    )
    summary = manifest["summary"]
    print(
        "Wrote MIMIC extraction manifest: "
        f"tables={summary['table_count']}, "
        f"status={summary['status_counts']}, "
        f"completed_rows={summary['completed_row_count']}"
    )
    return 1 if manifest_has_failures(manifest) else 0


if __name__ == "__main__":
    raise SystemExit(main())
