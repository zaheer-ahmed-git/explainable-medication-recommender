"""Cohort-filtered eICU-CRD source extraction for Milestone 5 preparation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from pipeline.config import (
    COHORTS_ROOT,
    DATASET_ROOT,
    EXTRACTS_ROOT,
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


DEFAULT_EICU_OUTPUT_ROOT = EXTRACTS_ROOT / "eicu_crd"
DEFAULT_EICU_MANIFEST_PATH = REPORTS_ROOT / "eicu_extraction_manifest.json"
DEFAULT_COHORT_PATH = COHORTS_ROOT / "cohort_stays.parquet"

_STAY_JOIN = (("patientunitstayid", "source_stay_id"),)

EICU_EXTRACTION_TABLES: tuple[ExtractionTableSpec, ...] = (
    ExtractionTableSpec(
        table_name="eicu_patient",
        source="eicu_crd",
        source_version="2.0",
        relative_path=Path("eicu-crd/2.0/patient.csv.gz"),
        output_name="patient.parquet",
        required_columns=(
            "patientunitstayid",
            "patienthealthsystemstayid",
            "gender",
            "age",
            "ethnicity",
            "hospitalid",
            "wardid",
            "apacheadmissiondx",
            "hospitaladmitsource",
            "unitadmitsource",
            "unitvisitnumber",
            "unitstaytype",
            "unitdischargeoffset",
            "uniquepid",
        ),
        selected_columns=(
            "patientunitstayid",
            "patienthealthsystemstayid",
            "gender",
            "age",
            "ethnicity",
            "hospitalid",
            "wardid",
            "apacheadmissiondx",
            "hospitaladmitsource",
            "unitadmitsource",
            "unitvisitnumber",
            "unitstaytype",
            "unitdischargeoffset",
            "uniquepid",
        ),
        join_columns=_STAY_JOIN,
        profile_table_name="eicu_patient",
    ),
    ExtractionTableSpec(
        table_name="eicu_diagnosis",
        source="eicu_crd",
        source_version="2.0",
        relative_path=Path("eicu-crd/2.0/diagnosis.csv.gz"),
        output_name="diagnosis.parquet",
        required_columns=(
            "diagnosisid",
            "patientunitstayid",
            "activeupondischarge",
            "diagnosisoffset",
            "diagnosisstring",
            "icd9code",
            "diagnosispriority",
        ),
        selected_columns=(
            "diagnosisid",
            "patientunitstayid",
            "activeupondischarge",
            "diagnosisoffset",
            "diagnosisstring",
            "icd9code",
            "diagnosispriority",
        ),
        join_columns=_STAY_JOIN,
        profile_table_name="eicu_diagnosis",
    ),
    ExtractionTableSpec(
        table_name="eicu_lab",
        source="eicu_crd",
        source_version="2.0",
        relative_path=Path("eicu-crd/2.0/lab.csv.gz"),
        output_name="lab.parquet",
        required_columns=(
            "labid",
            "patientunitstayid",
            "labresultoffset",
            "labtypeid",
            "labname",
            "labresult",
            "labresulttext",
            "labmeasurenamesystem",
            "labmeasurenameinterface",
            "labresultrevisedoffset",
        ),
        selected_columns=(
            "labid",
            "patientunitstayid",
            "labresultoffset",
            "labtypeid",
            "labname",
            "labresult",
            "labresulttext",
            "labmeasurenamesystem",
            "labmeasurenameinterface",
            "labresultrevisedoffset",
        ),
        join_columns=_STAY_JOIN,
        profile_table_name="eicu_lab",
    ),
    ExtractionTableSpec(
        table_name="eicu_medication",
        source="eicu_crd",
        source_version="2.0",
        relative_path=Path("eicu-crd/2.0/medication.csv.gz"),
        output_name="medication.parquet",
        required_columns=(
            "medicationid",
            "patientunitstayid",
            "drugorderoffset",
            "drugstartoffset",
            "drugivadmixture",
            "drugordercancelled",
            "drugname",
            "drughiclseqno",
            "dosage",
            "routeadmin",
            "frequency",
            "loadingdose",
            "prn",
            "drugstopoffset",
            "gtc",
        ),
        selected_columns=(
            "medicationid",
            "patientunitstayid",
            "drugorderoffset",
            "drugstartoffset",
            "drugivadmixture",
            "drugordercancelled",
            "drugname",
            "drughiclseqno",
            "dosage",
            "routeadmin",
            "frequency",
            "loadingdose",
            "prn",
            "drugstopoffset",
            "gtc",
        ),
        join_columns=_STAY_JOIN,
        profile_table_name="eicu_medication",
        requires_integrity_gate=True,
    ),
    ExtractionTableSpec(
        table_name="eicu_infusion_drug",
        source="eicu_crd",
        source_version="2.0",
        relative_path=Path("eicu-crd/2.0/infusionDrug.csv.gz"),
        output_name="infusion_drug.parquet",
        required_columns=(
            "infusiondrugid",
            "patientunitstayid",
            "infusionoffset",
            "drugname",
            "drugrate",
            "infusionrate",
            "drugamount",
            "volumeoffluid",
            "patientweight",
        ),
        selected_columns=(
            "infusiondrugid",
            "patientunitstayid",
            "infusionoffset",
            "drugname",
            "drugrate",
            "infusionrate",
            "drugamount",
            "volumeoffluid",
            "patientweight",
        ),
        join_columns=_STAY_JOIN,
        profile_table_name="eicu_infusion_drug",
    ),
    ExtractionTableSpec(
        table_name="eicu_allergy",
        source="eicu_crd",
        source_version="2.0",
        relative_path=Path("eicu-crd/2.0/allergy.csv.gz"),
        output_name="allergy.parquet",
        required_columns=(
            "allergyid",
            "patientunitstayid",
            "allergyoffset",
            "allergyenteredoffset",
            "allergynotetype",
            "specialtytype",
            "usertype",
            "rxincluded",
            "writtenineicu",
            "drugname",
            "allergytype",
            "allergyname",
            "drughiclseqno",
        ),
        selected_columns=(
            "allergyid",
            "patientunitstayid",
            "allergyoffset",
            "allergyenteredoffset",
            "allergynotetype",
            "specialtytype",
            "usertype",
            "rxincluded",
            "writtenineicu",
            "drugname",
            "allergytype",
            "allergyname",
            "drughiclseqno",
        ),
        join_columns=_STAY_JOIN,
        profile_table_name="eicu_allergy",
    ),
    ExtractionTableSpec(
        table_name="eicu_treatment",
        source="eicu_crd",
        source_version="2.0",
        relative_path=Path("eicu-crd/2.0/treatment.csv.gz"),
        output_name="treatment.parquet",
        required_columns=(
            "treatmentid",
            "patientunitstayid",
            "treatmentoffset",
            "treatmentstring",
            "activeupondischarge",
        ),
        selected_columns=(
            "treatmentid",
            "patientunitstayid",
            "treatmentoffset",
            "treatmentstring",
            "activeupondischarge",
        ),
        join_columns=_STAY_JOIN,
        profile_table_name="eicu_treatment",
    ),
    ExtractionTableSpec(
        table_name="eicu_vital_periodic",
        source="eicu_crd",
        source_version="2.0",
        relative_path=Path("eicu-crd/2.0/vitalPeriodic.csv.gz"),
        output_name="vital_periodic.parquet",
        required_columns=(
            "vitalperiodicid",
            "patientunitstayid",
            "observationoffset",
            "temperature",
            "sao2",
            "heartrate",
            "respiration",
            "systemicsystolic",
            "systemicdiastolic",
            "systemicmean",
        ),
        selected_columns=(
            "vitalperiodicid",
            "patientunitstayid",
            "observationoffset",
            "temperature",
            "sao2",
            "heartrate",
            "respiration",
            "systemicsystolic",
            "systemicdiastolic",
            "systemicmean",
        ),
        join_columns=_STAY_JOIN,
        profile_table_name="eicu_vital_periodic",
    ),
    ExtractionTableSpec(
        table_name="eicu_vital_aperiodic",
        source="eicu_crd",
        source_version="2.0",
        relative_path=Path("eicu-crd/2.0/vitalAperiodic.csv.gz"),
        output_name="vital_aperiodic.parquet",
        required_columns=(
            "vitalaperiodicid",
            "patientunitstayid",
            "observationoffset",
            "noninvasivesystolic",
            "noninvasivediastolic",
            "noninvasivemean",
        ),
        selected_columns=(
            "vitalaperiodicid",
            "patientunitstayid",
            "observationoffset",
            "noninvasivesystolic",
            "noninvasivediastolic",
            "noninvasivemean",
        ),
        join_columns=_STAY_JOIN,
        profile_table_name="eicu_vital_aperiodic",
    ),
    ExtractionTableSpec(
        table_name="eicu_apache_patient_result",
        source="eicu_crd",
        source_version="2.0",
        relative_path=Path("eicu-crd/2.0/apachePatientResult.csv.gz"),
        output_name="apache_patient_result.parquet",
        required_columns=(
            "apachepatientresultsid",
            "patientunitstayid",
            "acutephysiologyscore",
            "apachescore",
            "apacheversion",
            "predictedicumortality",
            "actualicumortality",
            "predictediculos",
            "actualiculos",
            "predictedhospitalmortality",
            "actualhospitalmortality",
        ),
        selected_columns=(
            "apachepatientresultsid",
            "patientunitstayid",
            "acutephysiologyscore",
            "apachescore",
            "apacheversion",
            "predictedicumortality",
            "actualicumortality",
            "predictediculos",
            "actualiculos",
            "predictedhospitalmortality",
            "actualhospitalmortality",
        ),
        join_columns=_STAY_JOIN,
        profile_table_name="eicu_apache_patient_result",
        requires_integrity_gate=True,
        notes=(
            "outcome columns are extracted for provenance, not as default features",
        ),
    ),
    ExtractionTableSpec(
        table_name="eicu_apache_aps_var",
        source="eicu_crd",
        source_version="2.0",
        relative_path=Path("eicu-crd/2.0/apacheApsVar.csv.gz"),
        output_name="apache_aps_var.parquet",
        required_columns=(
            "apacheapsvarid",
            "patientunitstayid",
            "intubated",
            "vent",
            "dialysis",
            "wbc",
            "temperature",
            "respiratoryrate",
            "sodium",
            "heartrate",
            "meanbp",
            "creatinine",
            "glucose",
        ),
        selected_columns=(
            "apacheapsvarid",
            "patientunitstayid",
            "intubated",
            "vent",
            "dialysis",
            "wbc",
            "temperature",
            "respiratoryrate",
            "sodium",
            "heartrate",
            "meanbp",
            "creatinine",
            "glucose",
        ),
        join_columns=_STAY_JOIN,
        profile_table_name="eicu_apache_aps_var",
    ),
    ExtractionTableSpec(
        table_name="eicu_apache_pred_var",
        source="eicu_crd",
        source_version="2.0",
        relative_path=Path("eicu-crd/2.0/apachePredVar.csv.gz"),
        output_name="apache_pred_var.parquet",
        required_columns=(
            "apachepredvarid",
            "patientunitstayid",
            "gender",
            "age",
            "admitdiagnosis",
            "creatinine",
            "diabetes",
            "ventday1",
        ),
        selected_columns=(
            "apachepredvarid",
            "patientunitstayid",
            "gender",
            "age",
            "admitdiagnosis",
            "creatinine",
            "diabetes",
            "ventday1",
        ),
        join_columns=_STAY_JOIN,
        profile_table_name="eicu_apache_pred_var",
    ),
)


def build_eicu_extracts(
    config: ExtractionBuildConfig,
    table_specs: Sequence[ExtractionTableSpec] = EICU_EXTRACTION_TABLES,
) -> dict[str, object]:
    """Build eICU-CRD cohort-filtered extraction artifacts."""

    return build_source_extracts(config, table_specs)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build cohort-filtered eICU-CRD extraction artifacts.",
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
        default=DEFAULT_EICU_OUTPUT_ROOT,
        help="Output directory for local eICU extract Parquet files.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_EICU_MANIFEST_PATH,
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
    manifest = build_eicu_extracts(
        ExtractionBuildConfig(
            source="eicu_crd",
            source_version="2.0",
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
        "Wrote eICU extraction manifest: "
        f"tables={summary['table_count']}, "
        f"status={summary['status_counts']}, "
        f"completed_rows={summary['completed_row_count']}"
    )
    return 1 if manifest_has_failures(manifest) else 0


if __name__ == "__main__":
    raise SystemExit(main())
