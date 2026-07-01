"""Build source-specific ICU stay cohorts from licensed local datasets."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import duckdb

from pipeline.config import (
    COHORTS_ROOT,
    DATASET_ROOT,
    DEFAULT_COHORT_PARAMETERS,
    REPORTS_ROOT,
)
from pipeline.io_utils import DatasetPathError, inspect_header, resolve_dataset_path


SCHEMA_VERSION = "cohort-manifest-v1"
DEFAULT_MANIFEST_PATH = REPORTS_ROOT / "cohort_manifest.json"

MIMIC_PATIENTS = Path("mimiciv") / "3.1" / "hosp" / "patients.csv.gz"
MIMIC_ADMISSIONS = Path("mimiciv") / "3.1" / "hosp" / "admissions.csv.gz"
MIMIC_ICUSTAYS = Path("mimiciv") / "3.1" / "icu" / "icustays.csv.gz"
EICU_PATIENT = Path("eicu-crd") / "2.0" / "patient.csv.gz"

MIMIC_REQUIRED_COLUMNS = {
    MIMIC_PATIENTS: {"subject_id", "gender", "anchor_age"},
    MIMIC_ADMISSIONS: {
        "subject_id",
        "hadm_id",
        "admittime",
        "dischtime",
        "admission_type",
        "admission_location",
        "race",
        "insurance",
        "language",
        "marital_status",
        "hospital_expire_flag",
    },
    MIMIC_ICUSTAYS: {
        "subject_id",
        "hadm_id",
        "stay_id",
        "first_careunit",
        "last_careunit",
        "intime",
        "outtime",
        "los",
    },
}

EICU_REQUIRED_COLUMNS = {
    EICU_PATIENT: {
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
        "unittype",
        "unitstaytype",
        "unitvisitnumber",
        "unitdischargeoffset",
        "uniquepid",
    }
}


@dataclass(frozen=True)
class CohortArtifactPaths:
    """Local output paths for source-specific and unified cohort artifacts."""

    mimic_cohort: Path
    eicu_cohort: Path
    unified_cohort: Path
    manifest: Path


@dataclass(frozen=True)
class CohortBuildConfig:
    """Configuration for Milestone 2 cohort materialization."""

    dataset_root: Path = DATASET_ROOT
    cohorts_root: Path = COHORTS_ROOT
    manifest_path: Path = DEFAULT_MANIFEST_PATH
    adult_age_minimum: int = int(DEFAULT_COHORT_PARAMETERS["adult_age_minimum"])
    mimic_first_icu_stay_per_admission: bool = bool(
        DEFAULT_COHORT_PARAMETERS["mimic_first_icu_stay_per_admission"]
    )

    @property
    def artifact_paths(self) -> CohortArtifactPaths:
        return CohortArtifactPaths(
            mimic_cohort=self.cohorts_root / "mimic_icu_stays.parquet",
            eicu_cohort=self.cohorts_root / "eicu_unit_stays.parquet",
            unified_cohort=self.cohorts_root / "cohort_stays.parquet",
            manifest=self.manifest_path,
        )


def sql_string(value: str | Path) -> str:
    """Return a SQL string literal."""

    return "'" + str(value).replace("'", "''") + "'"


def csv_scan(path: Path) -> str:
    """Return a DuckDB CSV scan expression that avoids eager typed loading."""

    return f"read_csv_auto({sql_string(path)}, header = true, all_varchar = true)"


def validate_required_columns(
    dataset_root: Path,
    requirements: dict[Path, set[str]],
) -> None:
    """Validate source headers before running cohort SQL."""

    for relative_path, required_columns in requirements.items():
        path = resolve_dataset_path(relative_path, dataset_root=dataset_root)
        header = inspect_header(path)
        if header is None:
            raise DatasetPathError(f"Expected CSV-like source file: {path}")
        missing = sorted(required_columns.difference(header))
        if missing:
            raise DatasetPathError(
                f"{path} is missing required columns: {', '.join(missing)}"
            )


def source_paths(dataset_root: Path) -> dict[str, Path]:
    """Resolve source files used by the cohort milestone."""

    return {
        "mimic_patients": resolve_dataset_path(
            MIMIC_PATIENTS, dataset_root=dataset_root
        ),
        "mimic_admissions": resolve_dataset_path(
            MIMIC_ADMISSIONS, dataset_root=dataset_root
        ),
        "mimic_icustays": resolve_dataset_path(
            MIMIC_ICUSTAYS, dataset_root=dataset_root
        ),
        "eicu_patient": resolve_dataset_path(EICU_PATIENT, dataset_root=dataset_root),
    }


def mimic_common_ctes(paths: dict[str, Path]) -> str:
    """CTEs shared by MIMIC cohort and attrition queries."""

    return f"""
WITH
patients_raw AS (
    SELECT
        subject_id,
        gender,
        TRY_CAST(anchor_age AS DOUBLE) AS age_years
    FROM {csv_scan(paths["mimic_patients"])}
),
admissions_raw AS (
    SELECT
        subject_id,
        hadm_id,
        admittime,
        dischtime,
        admission_type,
        admission_location,
        race,
        insurance,
        language,
        marital_status,
        hospital_expire_flag
    FROM {csv_scan(paths["mimic_admissions"])}
),
icustays_raw AS (
    SELECT
        subject_id,
        hadm_id,
        stay_id,
        first_careunit,
        last_careunit,
        intime,
        outtime,
        TRY_CAST(los AS DOUBLE) AS los_days
    FROM {csv_scan(paths["mimic_icustays"])}
),
joined AS (
    SELECT
        i.subject_id,
        i.hadm_id,
        i.stay_id,
        p.gender,
        p.age_years,
        a.admittime,
        a.dischtime,
        a.admission_type,
        a.admission_location,
        a.race,
        a.insurance,
        a.language,
        a.marital_status,
        TRY_CAST(a.hospital_expire_flag AS INTEGER) AS hospital_expire_flag,
        i.first_careunit,
        i.last_careunit,
        i.intime,
        i.outtime,
        i.los_days,
        ROW_NUMBER() OVER (
            PARTITION BY i.hadm_id
            ORDER BY TRY_CAST(i.intime AS TIMESTAMP), i.stay_id
        ) AS icu_stay_number_in_admission
    FROM icustays_raw AS i
    INNER JOIN patients_raw AS p
        ON i.subject_id = p.subject_id
    INNER JOIN admissions_raw AS a
        ON i.subject_id = a.subject_id
        AND i.hadm_id = a.hadm_id
),
adult_joined AS (
    SELECT *
    FROM joined
    WHERE age_years >= {{adult_age_minimum}}
)
"""


def mimic_selected_query(
    paths: dict[str, Path],
    *,
    adult_age_minimum: int,
    first_icu_stay_per_admission: bool,
) -> str:
    """Build the MIMIC adult ICU stay cohort query."""

    first_stay_predicate = (
        "icu_stay_number_in_admission = 1" if first_icu_stay_per_admission else "TRUE"
    )
    return (
        mimic_common_ctes(paths).format(adult_age_minimum=adult_age_minimum)
        + f"""
SELECT
    'mimiciv' AS source,
    '3.1' AS source_version,
    'mimiciv:' || CAST(subject_id AS VARCHAR) AS patient_uid,
    'mimiciv:' || CAST(hadm_id AS VARCHAR) AS encounter_uid,
    'mimiciv:' || CAST(stay_id AS VARCHAR) AS stay_uid,
    CAST(subject_id AS VARCHAR) AS source_patient_id,
    CAST(hadm_id AS VARCHAR) AS source_encounter_id,
    CAST(stay_id AS VARCHAR) AS source_stay_id,
    age_years,
    FALSE AS age_topcoded,
    gender AS sex,
    race AS race_or_ethnicity,
    CAST(NULL AS VARCHAR) AS hospital_id,
    CAST(NULL AS VARCHAR) AS ward_id,
    admission_type,
    admission_location AS admission_source,
    first_careunit AS unit_type,
    last_careunit AS last_unit_type,
    CAST(NULL AS VARCHAR) AS stay_type,
    icu_stay_number_in_admission AS stay_sequence,
    TRY_CAST(admittime AS TIMESTAMP) AS encounter_start_time,
    TRY_CAST(dischtime AS TIMESTAMP) AS encounter_end_time,
    TRY_CAST(intime AS TIMESTAMP) AS stay_start_time,
    TRY_CAST(outtime AS TIMESTAMP) AS stay_end_time,
    CAST(NULL AS INTEGER) AS stay_start_offset_minutes,
    CAST(NULL AS INTEGER) AS stay_end_offset_minutes,
    los_days * 24.0 AS los_hours,
    hospital_expire_flag,
    'adult_icu_stay_first_per_admission=' || CAST({str(first_icu_stay_per_admission).lower()} AS VARCHAR) AS cohort_rule
FROM adult_joined
WHERE {first_stay_predicate}
"""
    )


def mimic_attrition_query(
    paths: dict[str, Path],
    *,
    adult_age_minimum: int,
    first_icu_stay_per_admission: bool,
) -> str:
    selected = mimic_selected_query(
        paths,
        adult_age_minimum=adult_age_minimum,
        first_icu_stay_per_admission=first_icu_stay_per_admission,
    )
    return (
        mimic_common_ctes(paths).format(adult_age_minimum=adult_age_minimum)
        + f"""
, selected AS (
    SELECT * FROM ({selected})
)
SELECT
    (SELECT COUNT(*) FROM icustays_raw) AS source_icu_stays,
    (SELECT COUNT(*) FROM joined) AS stays_joined_to_patient_admission,
    (SELECT COUNT(*) FROM adult_joined) AS adult_icu_stays,
    (SELECT COUNT(*) FROM selected) AS selected_stays,
    (SELECT COUNT(DISTINCT patient_uid) FROM selected) AS selected_patients,
    (SELECT COUNT(*) - COUNT(DISTINCT stay_uid) FROM selected) AS duplicate_stay_uid_count,
    (SELECT COUNT(*) FROM adult_joined)
        - (SELECT COUNT(*) FROM selected) AS excluded_by_first_stay_rule
"""
    )


def eicu_common_ctes(paths: dict[str, Path]) -> str:
    """CTEs shared by eICU cohort and attrition queries."""

    return f"""
WITH
patient_raw AS (
    SELECT
        patientunitstayid,
        patienthealthsystemstayid,
        gender,
        age,
        ethnicity,
        hospitalid,
        wardid,
        apacheadmissiondx,
        hospitaladmitsource,
        unitadmitsource,
        unittype,
        unitstaytype,
        unitvisitnumber,
        unitdischargeoffset,
        uniquepid
    FROM {csv_scan(paths["eicu_patient"])}
),
normalized AS (
    SELECT
        patientunitstayid,
        patienthealthsystemstayid,
        gender,
        TRIM(CAST(age AS VARCHAR)) AS age_text,
        ethnicity,
        hospitalid,
        wardid,
        apacheadmissiondx,
        hospitaladmitsource,
        unitadmitsource,
        unittype,
        unitstaytype,
        TRY_CAST(unitvisitnumber AS INTEGER) AS unitvisitnumber,
        TRY_CAST(unitdischargeoffset AS INTEGER) AS unitdischargeoffset,
        NULLIF(TRIM(CAST(uniquepid AS VARCHAR)), '') AS uniquepid
    FROM patient_raw
),
age_normalized AS (
    SELECT
        *,
        STARTS_WITH(age_text, '>') AS age_topcoded,
        CASE
            WHEN STARTS_WITH(age_text, '>') THEN 90.0
            ELSE TRY_CAST(age_text AS DOUBLE)
        END AS age_years
    FROM normalized
),
adult_unit_stays AS (
    SELECT *
    FROM age_normalized
    WHERE age_years >= {{adult_age_minimum}}
)
"""


def eicu_selected_query(paths: dict[str, Path], *, adult_age_minimum: int) -> str:
    """Build the eICU adult unit-stay cohort query."""

    return (
        eicu_common_ctes(paths).format(adult_age_minimum=adult_age_minimum)
        + """
SELECT
    'eicu_crd' AS source,
    '2.0' AS source_version,
    'eicu_crd:' || COALESCE(uniquepid, 'unitstay:' || CAST(patientunitstayid AS VARCHAR)) AS patient_uid,
    'eicu_crd:' || COALESCE(
        CAST(patienthealthsystemstayid AS VARCHAR),
        CAST(patientunitstayid AS VARCHAR)
    ) AS encounter_uid,
    'eicu_crd:' || CAST(patientunitstayid AS VARCHAR) AS stay_uid,
    COALESCE(uniquepid, 'unitstay:' || CAST(patientunitstayid AS VARCHAR)) AS source_patient_id,
    COALESCE(
        CAST(patienthealthsystemstayid AS VARCHAR),
        CAST(patientunitstayid AS VARCHAR)
    ) AS source_encounter_id,
    CAST(patientunitstayid AS VARCHAR) AS source_stay_id,
    age_years,
    age_topcoded,
    gender AS sex,
    ethnicity AS race_or_ethnicity,
    CAST(hospitalid AS VARCHAR) AS hospital_id,
    CAST(wardid AS VARCHAR) AS ward_id,
    apacheadmissiondx AS admission_type,
    COALESCE(unitadmitsource, hospitaladmitsource) AS admission_source,
    unittype AS unit_type,
    CAST(NULL AS VARCHAR) AS last_unit_type,
    unitstaytype AS stay_type,
    unitvisitnumber AS stay_sequence,
    CAST(NULL AS TIMESTAMP) AS encounter_start_time,
    CAST(NULL AS TIMESTAMP) AS encounter_end_time,
    CAST(NULL AS TIMESTAMP) AS stay_start_time,
    CAST(NULL AS TIMESTAMP) AS stay_end_time,
    0 AS stay_start_offset_minutes,
    unitdischargeoffset AS stay_end_offset_minutes,
    unitdischargeoffset / 60.0 AS los_hours,
    CAST(NULL AS INTEGER) AS hospital_expire_flag,
    'adult_unit_stay_age_topcoded_to_90' AS cohort_rule
FROM adult_unit_stays
"""
    )


def eicu_attrition_query(paths: dict[str, Path], *, adult_age_minimum: int) -> str:
    selected = eicu_selected_query(paths, adult_age_minimum=adult_age_minimum)
    return (
        eicu_common_ctes(paths).format(adult_age_minimum=adult_age_minimum)
        + f"""
, selected AS (
    SELECT * FROM ({selected})
)
SELECT
    (SELECT COUNT(*) FROM patient_raw) AS source_unit_stays,
    (SELECT COUNT(*) FROM age_normalized WHERE age_years IS NULL) AS missing_or_unparseable_age_stays,
    (SELECT COUNT(*) FROM adult_unit_stays) AS adult_unit_stays,
    (SELECT COUNT(*) FROM selected) AS selected_stays,
    (SELECT COUNT(DISTINCT patient_uid) FROM selected) AS selected_patients,
    (SELECT COUNT(*) - COUNT(DISTINCT stay_uid) FROM selected) AS duplicate_stay_uid_count,
    (SELECT COUNT(*) FROM selected WHERE age_topcoded) AS topcoded_age_stays
"""
    )


def fetch_single_row(
    connection: duckdb.DuckDBPyConnection, query: str
) -> dict[str, Any]:
    """Fetch one aggregate row as a dictionary."""

    cursor = connection.execute(query)
    row = cursor.fetchone()
    if row is None:
        return {}
    column_names = [description[0] for description in cursor.description]
    return dict(zip(column_names, row, strict=True))


def copy_query_to_parquet(
    connection: duckdb.DuckDBPyConnection,
    query: str,
    output_path: Path,
) -> None:
    """Materialize a query as a local Parquet artifact."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    connection.execute(f"COPY ({query}) TO {sql_string(output_path)} (FORMAT PARQUET)")


def build_cohort_artifacts(
    config: CohortBuildConfig = CohortBuildConfig(),
) -> dict[str, Any]:
    """Build source-specific and unified cohort artifacts plus a safe manifest."""

    config.cohorts_root.mkdir(parents=True, exist_ok=True)
    config.manifest_path.parent.mkdir(parents=True, exist_ok=True)

    validate_required_columns(config.dataset_root, MIMIC_REQUIRED_COLUMNS)
    validate_required_columns(config.dataset_root, EICU_REQUIRED_COLUMNS)
    paths = source_paths(config.dataset_root)

    artifact_paths = config.artifact_paths
    mimic_query = mimic_selected_query(
        paths,
        adult_age_minimum=config.adult_age_minimum,
        first_icu_stay_per_admission=config.mimic_first_icu_stay_per_admission,
    )
    eicu_query = eicu_selected_query(
        paths,
        adult_age_minimum=config.adult_age_minimum,
    )

    with duckdb.connect(database=":memory:") as connection:
        copy_query_to_parquet(connection, mimic_query, artifact_paths.mimic_cohort)
        copy_query_to_parquet(connection, eicu_query, artifact_paths.eicu_cohort)
        copy_query_to_parquet(
            connection,
            f"""
            SELECT * FROM read_parquet({sql_string(artifact_paths.mimic_cohort)})
            UNION ALL BY NAME
            SELECT * FROM read_parquet({sql_string(artifact_paths.eicu_cohort)})
            """,
            artifact_paths.unified_cohort,
        )
        mimic_attrition = fetch_single_row(
            connection,
            mimic_attrition_query(
                paths,
                adult_age_minimum=config.adult_age_minimum,
                first_icu_stay_per_admission=config.mimic_first_icu_stay_per_admission,
            ),
        )
        eicu_attrition = fetch_single_row(
            connection,
            eicu_attrition_query(
                paths,
                adult_age_minimum=config.adult_age_minimum,
            ),
        )
        unified_summary = fetch_single_row(
            connection,
            f"""
            SELECT
                COUNT(*) AS selected_stays,
                COUNT(DISTINCT patient_uid) AS selected_patients,
                COUNT(*) - COUNT(DISTINCT stay_uid) AS duplicate_stay_uid_count
            FROM read_parquet({sql_string(artifact_paths.unified_cohort)})
            """,
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "data_safety": {
            "contains_patient_rows": False,
            "identifier_artifacts_are_local_ignored": True,
            "reporting_level": "aggregate counts only",
        },
        "configuration": {
            "adult_age_minimum": config.adult_age_minimum,
            "mimic_first_icu_stay_per_admission": config.mimic_first_icu_stay_per_admission,
            "sepsis_subcohort_status": "not_implemented_pending_definition_approval",
            "split_status": "not_created_milestone_2",
        },
        "artifacts": {key: str(value) for key, value in asdict(artifact_paths).items()},
        "sources": {
            "mimiciv": mimic_attrition,
            "eicu_crd": eicu_attrition,
            "unified": unified_summary,
        },
    }
    config.manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build adult ICU/unit-stay cohort artifacts and aggregate manifest.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DATASET_ROOT,
        help="Dataset root. Defaults to the repository Dataset directory.",
    )
    parser.add_argument(
        "--cohorts-root",
        type=Path,
        default=COHORTS_ROOT,
        help="Output directory for local cohort Parquet files.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Output path for the aggregate JSON manifest.",
    )
    parser.add_argument(
        "--adult-age-minimum",
        type=int,
        default=int(DEFAULT_COHORT_PARAMETERS["adult_age_minimum"]),
        help="Minimum age in years for adult cohorts.",
    )
    parser.add_argument(
        "--include-all-mimic-icu-stays",
        action="store_true",
        help="Keep all adult MIMIC ICU stays instead of first ICU stay per admission.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_cohort_artifacts(
        CohortBuildConfig(
            dataset_root=args.dataset_root,
            cohorts_root=args.cohorts_root,
            manifest_path=args.manifest,
            adult_age_minimum=args.adult_age_minimum,
            mimic_first_icu_stay_per_admission=not args.include_all_mimic_icu_stays,
        )
    )
    sources = manifest["sources"]
    print(
        "Wrote cohort artifacts with aggregate counts: "
        f"MIMIC selected_stays={sources['mimiciv']['selected_stays']}, "
        f"eICU selected_stays={sources['eicu_crd']['selected_stays']}, "
        f"unified selected_stays={sources['unified']['selected_stays']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
