from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


PATIENT_ID = "patient_id"
ADHERENCE_TARGET = 80.0


def normalize_token(value: object) -> str:
    """Normalize clinical labels to stable snake_case tokens."""
    if pd.isna(value):
        return ""
    token = str(value).strip().lower()
    token = re.sub(r"[^a-z0-9]+", "_", token)
    token = re.sub(r"_+", "_", token).strip("_")
    return token


def normalize_series(series: pd.Series) -> pd.Series:
    return series.map(normalize_token).astype("string")


def read_csv(path: Path, parse_dates: Iterable[str] = ()) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    for column in parse_dates:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")
    return df


def coerce_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def flatten_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [f"{prefix}_{normalize_token(col)}" for col in df.columns]
    return df


def stable_patient_split(patient_id: object) -> str:
    """Deterministic patient-level split to prevent row-level leakage."""
    digest = hashlib.md5(str(patient_id).encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "valid"
    return "test"


def load_source_tables(data_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        "patients": read_csv(data_dir / "patients.csv"),
        "diagnoses": read_csv(data_dir / "diagnoses.csv", parse_dates=["visit_date"]),
        "labs": read_csv(data_dir / "lab_results.csv", parse_dates=["test_date"]),
        "medications": read_csv(data_dir / "medications.csv", parse_dates=["start_date"]),
        "outcomes": read_csv(
            data_dir / "outcomes.csv",
            parse_dates=["admission_date", "discharge_date"],
        ),
    }


def clean_patients(patients: pd.DataFrame) -> pd.DataFrame:
    patients = patients.copy()

    categorical_columns = [
        "sex",
        "smoking_status",
        "alcohol_use",
        "exercise_level",
        "insurance_type",
    ]
    for column in categorical_columns:
        if column in patients.columns:
            patients[column] = normalize_series(patients[column])

    numeric_columns = [
        "age",
        "bmi",
        "systolic_bp",
        "diastolic_bp",
        "heart_rate",
        "temperature_f",
        "charlson_index",
    ]
    patients = coerce_numeric(patients, numeric_columns)

    dx_columns = [column for column in patients.columns if column.startswith("dx_")]
    patients = coerce_numeric(patients, dx_columns)
    patients["condition_count"] = patients[dx_columns].fillna(0).sum(axis=1)
    patients["split"] = patients[PATIENT_ID].map(stable_patient_split)

    return patients


def build_condition_events(
    patients: pd.DataFrame,
    diagnoses: pd.DataFrame,
    medications: pd.DataFrame,
) -> pd.DataFrame:
    """Build one row per patient-condition from flags, visits, and indications."""
    dx_columns = [column for column in patients.columns if column.startswith("dx_")]

    from_patient_flags = (
        patients[[PATIENT_ID, *dx_columns]]
        .melt(id_vars=PATIENT_ID, var_name="condition", value_name="has_patient_dx_flag")
        .assign(
            condition=lambda df: df["condition"].str.replace("dx_", "", regex=False),
            has_patient_dx_flag=lambda df: pd.to_numeric(
                df["has_patient_dx_flag"], errors="coerce"
            ).fillna(0),
            diagnosis_visit_count=0,
            medication_record_count_for_condition=0,
        )
    )
    from_patient_flags = from_patient_flags[
        from_patient_flags["has_patient_dx_flag"].eq(1)
    ]

    primary_events = pd.DataFrame()
    secondary_events = pd.DataFrame()
    if not diagnoses.empty:
        primary_events = diagnoses[[PATIENT_ID, "primary_diagnosis"]].rename(
            columns={"primary_diagnosis": "condition"}
        )
        primary_events["condition"] = normalize_series(primary_events["condition"])
        primary_events = primary_events[primary_events["condition"].ne("")]

        secondary_source = diagnoses[[PATIENT_ID, "secondary_diagnoses"]].dropna()
        if not secondary_source.empty:
            secondary_events = secondary_source.assign(
                condition=secondary_source["secondary_diagnoses"]
                .astype(str)
                .str.split("|")
            ).explode("condition")[[PATIENT_ID, "condition"]]
            secondary_events["condition"] = normalize_series(secondary_events["condition"])
            secondary_events = secondary_events[secondary_events["condition"].ne("")]

    diagnosis_events = pd.concat([primary_events, secondary_events], ignore_index=True)
    if diagnosis_events.empty:
        diagnosis_counts = pd.DataFrame(
            columns=[PATIENT_ID, "condition", "diagnosis_visit_count"]
        )
    else:
        diagnosis_counts = (
            diagnosis_events.groupby([PATIENT_ID, "condition"], as_index=False)
            .size()
            .rename(columns={"size": "diagnosis_visit_count"})
        )
        diagnosis_counts["has_patient_dx_flag"] = 0
        diagnosis_counts["medication_record_count_for_condition"] = 0

    medication_conditions = pd.DataFrame(
        columns=[PATIENT_ID, "condition", "medication_record_count_for_condition"]
    )
    if not medications.empty and "indication" in medications.columns:
        medication_conditions = medications[[PATIENT_ID, "indication"]].copy()
        medication_conditions["condition"] = normalize_series(
            medication_conditions["indication"]
        )
        medication_conditions = medication_conditions[
            medication_conditions["condition"].ne("")
        ]
        medication_conditions = (
            medication_conditions.groupby([PATIENT_ID, "condition"], as_index=False)
            .size()
            .rename(columns={"size": "medication_record_count_for_condition"})
        )
        medication_conditions["has_patient_dx_flag"] = 0
        medication_conditions["diagnosis_visit_count"] = 0

    condition_events = pd.concat(
        [
            from_patient_flags[
                [
                    PATIENT_ID,
                    "condition",
                    "has_patient_dx_flag",
                    "diagnosis_visit_count",
                    "medication_record_count_for_condition",
                ]
            ],
            diagnosis_counts[
                [
                    PATIENT_ID,
                    "condition",
                    "has_patient_dx_flag",
                    "diagnosis_visit_count",
                    "medication_record_count_for_condition",
                ]
            ],
            medication_conditions[
                [
                    PATIENT_ID,
                    "condition",
                    "has_patient_dx_flag",
                    "diagnosis_visit_count",
                    "medication_record_count_for_condition",
                ]
            ],
        ],
        ignore_index=True,
    )

    return (
        condition_events.groupby([PATIENT_ID, "condition"], as_index=False)
        .agg(
            has_patient_dx_flag=("has_patient_dx_flag", "max"),
            diagnosis_visit_count=("diagnosis_visit_count", "sum"),
            medication_record_count_for_condition=(
                "medication_record_count_for_condition",
                "sum",
            ),
        )
        .sort_values([PATIENT_ID, "condition"])
    )


def aggregate_diagnoses(diagnoses: pd.DataFrame) -> pd.DataFrame:
    if diagnoses.empty:
        return pd.DataFrame(columns=[PATIENT_ID])

    diagnoses = diagnoses.copy()
    diagnoses["primary_diagnosis_norm"] = normalize_series(diagnoses["primary_diagnosis"])
    diagnoses["provider_specialty_norm"] = normalize_series(diagnoses["provider_specialty"])
    diagnoses["visit_type_norm"] = normalize_series(diagnoses["visit_type"])

    base = (
        diagnoses.groupby(PATIENT_ID)
        .agg(
            diagnosis_total_visits=("visit_date", "size"),
            diagnosis_unique_primary_conditions=("primary_diagnosis_norm", "nunique"),
            diagnosis_unique_provider_specialties=("provider_specialty_norm", "nunique"),
            diagnosis_first_visit_date=("visit_date", "min"),
            diagnosis_last_visit_date=("visit_date", "max"),
        )
        .reset_index()
    )

    visit_types = pd.crosstab(diagnoses[PATIENT_ID], diagnoses["visit_type_norm"])
    visit_types = flatten_columns(visit_types, "diagnosis_visit_type_count").reset_index()

    provider_specialties = pd.crosstab(
        diagnoses[PATIENT_ID], diagnoses["provider_specialty_norm"]
    )
    provider_specialties = flatten_columns(
        provider_specialties, "diagnosis_provider_count"
    ).reset_index()

    return base.merge(visit_types, on=PATIENT_ID, how="left").merge(
        provider_specialties, on=PATIENT_ID, how="left"
    )


def aggregate_labs(labs: pd.DataFrame) -> pd.DataFrame:
    if labs.empty:
        return pd.DataFrame(columns=[PATIENT_ID])

    labs = labs.copy()
    labs["test_name_norm"] = normalize_series(labs["test_name"])
    labs = coerce_numeric(
        labs,
        ["value", "reference_low", "reference_high", "is_abnormal", "delta_from_normal"],
    )

    grouped = labs.groupby([PATIENT_ID, "test_name_norm"])
    stats = grouped.agg(
        value_mean=("value", "mean"),
        value_min=("value", "min"),
        value_max=("value", "max"),
        test_count=("value", "size"),
        abnormal_count=("is_abnormal", "sum"),
        delta_from_normal_mean=("delta_from_normal", "mean"),
    )
    stats_wide = stats.unstack("test_name_norm")
    stats_wide.columns = [
        f"lab_{test}_{metric}" for metric, test in stats_wide.columns.to_flat_index()
    ]
    stats_wide = stats_wide.reset_index()

    latest = (
        labs.sort_values([PATIENT_ID, "test_name_norm", "test_date"])
        .drop_duplicates([PATIENT_ID, "test_name_norm"], keep="last")
        .copy()
    )
    latest_value = latest.pivot(
        index=PATIENT_ID, columns="test_name_norm", values="value"
    )
    latest_value = flatten_columns(latest_value, "lab_latest_value").reset_index()

    latest_abnormal = latest.pivot(
        index=PATIENT_ID, columns="test_name_norm", values="is_abnormal"
    )
    latest_abnormal = flatten_columns(
        latest_abnormal, "lab_latest_is_abnormal"
    ).reset_index()

    latest_delta = latest.pivot(
        index=PATIENT_ID, columns="test_name_norm", values="delta_from_normal"
    )
    latest_delta = flatten_columns(latest_delta, "lab_latest_delta").reset_index()

    return (
        stats_wide.merge(latest_value, on=PATIENT_ID, how="left")
        .merge(latest_abnormal, on=PATIENT_ID, how="left")
        .merge(latest_delta, on=PATIENT_ID, how="left")
    )


def aggregate_medications(medications: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    medications = medications.copy()
    medications["condition"] = normalize_series(medications["indication"])
    medications["medication_norm"] = normalize_series(medications["medication"])
    medications["frequency_norm"] = normalize_series(medications["frequency"])
    medications = medications[
        medications["condition"].ne("") & medications["medication_norm"].ne("")
    ]
    medications = coerce_numeric(
        medications,
        ["dose", "duration_days", "is_generic", "adherence_pct"],
    )

    patient_summary = (
        medications.groupby(PATIENT_ID)
        .agg(
            medication_total_records=("medication_norm", "size"),
            medication_unique_count=("medication_norm", "nunique"),
            medication_indication_count=("condition", "nunique"),
            medication_mean_adherence_pct=("adherence_pct", "mean"),
            medication_min_adherence_pct=("adherence_pct", "min"),
            medication_low_adherence_record_count=(
                "adherence_pct",
                lambda s: (s < ADHERENCE_TARGET).sum(),
            ),
            medication_total_duration_days=("duration_days", "sum"),
        )
        .reset_index()
    )

    positive_labels = (
        medications.groupby([PATIENT_ID, "condition", "medication_norm"], as_index=False)
        .agg(
            prescription_count=("medication_norm", "size"),
            dose_mean=("dose", "mean"),
            duration_days_total=("duration_days", "sum"),
            adherence_pct_mean=("adherence_pct", "mean"),
            adherence_pct_min=("adherence_pct", "min"),
            adherence_pct_max=("adherence_pct", "max"),
            is_generic_any=("is_generic", "max"),
            first_medication_start_date=("start_date", "min"),
            last_medication_start_date=("start_date", "max"),
        )
        .rename(columns={"medication_norm": "medication"})
    )
    positive_labels["label_prescribed"] = 1
    positive_labels["label_high_adherence"] = (
        positive_labels["adherence_pct_mean"] >= ADHERENCE_TARGET
    ).astype("Int64")

    return patient_summary, positive_labels


def aggregate_outcomes(outcomes: pd.DataFrame) -> pd.DataFrame:
    if outcomes.empty:
        return pd.DataFrame(columns=[PATIENT_ID])

    outcomes = outcomes.copy()
    outcomes = coerce_numeric(
        outcomes,
        [
            "length_of_stay_days",
            "icu_admission",
            "icu_days",
            "in_hospital_death",
            "readmitted_30d",
            "days_to_readmission",
            "total_charges_usd",
        ],
    )

    return (
        outcomes.groupby(PATIENT_ID)
        .agg(
            outcome_hospitalization_count=("admission_date", "size"),
            outcome_first_admission_date=("admission_date", "min"),
            outcome_last_admission_date=("admission_date", "max"),
            outcome_length_of_stay_mean=("length_of_stay_days", "mean"),
            outcome_length_of_stay_max=("length_of_stay_days", "max"),
            outcome_icu_admission_any=("icu_admission", "max"),
            outcome_icu_days_total=("icu_days", "sum"),
            outcome_in_hospital_death_any=("in_hospital_death", "max"),
            outcome_readmitted_30d_any=("readmitted_30d", "max"),
            outcome_readmitted_30d_count=("readmitted_30d", "sum"),
            outcome_days_to_readmission_min=("days_to_readmission", "min"),
            outcome_total_charges_usd_sum=("total_charges_usd", "sum"),
            outcome_total_charges_usd_mean=("total_charges_usd", "mean"),
        )
        .reset_index()
    )


def build_patient_features(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    patients = clean_patients(tables["patients"])
    diagnosis_summary = aggregate_diagnoses(tables["diagnoses"])
    lab_summary = aggregate_labs(tables["labs"])
    medication_summary, _ = aggregate_medications(tables["medications"])
    outcome_summary = aggregate_outcomes(tables["outcomes"])

    features = (
        patients.merge(diagnosis_summary, on=PATIENT_ID, how="left")
        .merge(lab_summary, on=PATIENT_ID, how="left")
        .merge(medication_summary, on=PATIENT_ID, how="left")
        .merge(outcome_summary, on=PATIENT_ID, how="left")
    )

    count_columns = [
        column
        for column in features.columns
        if column.endswith("_count")
        or column.endswith("_records")
        or column.endswith("_total")
        or column.endswith("_sum")
        or column.endswith("_any")
    ]
    features[count_columns] = features[count_columns].fillna(0)

    return features


def build_candidate_medications(
    positive_labels: pd.DataFrame,
    max_candidates_per_condition: int,
    min_candidate_count: int,
) -> pd.DataFrame:
    candidates = (
        positive_labels.groupby(["condition", "medication"], as_index=False)
        .agg(
            candidate_prescription_count=("prescription_count", "sum"),
            candidate_patient_count=(PATIENT_ID, "nunique"),
            candidate_mean_adherence_pct=("adherence_pct_mean", "mean"),
        )
        .sort_values(
            ["condition", "candidate_prescription_count", "candidate_patient_count"],
            ascending=[True, False, False],
        )
    )

    candidates = candidates[
        candidates["candidate_prescription_count"].ge(min_candidate_count)
    ]
    candidates["candidate_rank_within_condition"] = (
        candidates.groupby("condition").cumcount() + 1
    )

    if max_candidates_per_condition > 0:
        candidates = candidates[
            candidates["candidate_rank_within_condition"].le(max_candidates_per_condition)
        ]

    return candidates


def build_training_table(
    tables: dict[str, pd.DataFrame],
    patient_features: pd.DataFrame,
    max_candidates_per_condition: int,
    min_candidate_count: int,
) -> pd.DataFrame:
    patients = clean_patients(tables["patients"])
    _, positive_labels = aggregate_medications(tables["medications"])
    condition_events = build_condition_events(
        patients, tables["diagnoses"], tables["medications"]
    )
    candidates = build_candidate_medications(
        positive_labels,
        max_candidates_per_condition=max_candidates_per_condition,
        min_candidate_count=min_candidate_count,
    )

    base = condition_events.merge(candidates, on="condition", how="inner")
    training = base.merge(
        positive_labels,
        on=[PATIENT_ID, "condition", "medication"],
        how="left",
        suffixes=("", "_observed"),
    )

    training["label_prescribed"] = training["label_prescribed"].fillna(0).astype("int8")
    training["label_high_adherence"] = training["label_high_adherence"].where(
        training["label_prescribed"].eq(1), pd.NA
    )
    training["prescription_count"] = training["prescription_count"].fillna(0).astype("int16")

    feature_columns_to_drop = {
        "diagnosis_first_visit_date",
        "diagnosis_last_visit_date",
        "outcome_first_admission_date",
        "outcome_last_admission_date",
    }
    patient_feature_values = patient_features.drop(
        columns=[col for col in feature_columns_to_drop if col in patient_features.columns]
    )

    training = training.merge(patient_feature_values, on=PATIENT_ID, how="left")

    first_columns = [
        PATIENT_ID,
        "split",
        "condition",
        "medication",
        "label_prescribed",
        "label_high_adherence",
        "prescription_count",
        "adherence_pct_mean",
        "candidate_rank_within_condition",
        "candidate_prescription_count",
        "candidate_patient_count",
        "candidate_mean_adherence_pct",
        "has_patient_dx_flag",
        "diagnosis_visit_count",
        "medication_record_count_for_condition",
    ]
    ordered_columns = [
        column for column in first_columns if column in training.columns
    ] + [
        column for column in training.columns if column not in first_columns
    ]

    return training[ordered_columns].sort_values(
        [PATIENT_ID, "condition", "candidate_rank_within_condition"]
    )


def write_report(
    output_dir: Path,
    tables: dict[str, pd.DataFrame],
    patient_features: pd.DataFrame,
    training_table: pd.DataFrame,
) -> None:
    report = {
        "source_rows": {name: int(len(df)) for name, df in tables.items()},
        "patient_features_rows": int(len(patient_features)),
        "training_rows": int(len(training_table)),
        "positive_rows": int(training_table["label_prescribed"].sum()),
        "negative_rows": int((training_table["label_prescribed"] == 0).sum()),
        "unique_patients": int(training_table[PATIENT_ID].nunique()),
        "unique_conditions": int(training_table["condition"].nunique()),
        "unique_medications": int(training_table["medication"].nunique()),
        "split_counts": training_table["split"].value_counts(dropna=False).to_dict(),
        "notes": [
            "Rows are patient-condition-medication candidates.",
            "label_prescribed=1 means the medication was observed for that patient and indication.",
            "label_prescribed=0 means the medication is an observed candidate for that condition but not prescribed to that patient in the source data.",
            "Outcome columns are merged as patient-level summaries; avoid using them as predictors unless a temporal cutoff is added.",
        ],
    }

    with (output_dir / "preprocessing_report.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, default=str)


def run_preprocessing(
    data_dir: Path,
    output_dir: Path,
    max_candidates_per_condition: int,
    min_candidate_count: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = load_source_tables(data_dir)
    patient_features = build_patient_features(tables)
    training_table = build_training_table(
        tables=tables,
        patient_features=patient_features,
        max_candidates_per_condition=max_candidates_per_condition,
        min_candidate_count=min_candidate_count,
    )

    patient_features.to_csv(output_dir / "patient_features.csv", index=False)
    training_table.to_csv(
        output_dir / "patient_condition_medication.csv", index=False
    )
    write_report(output_dir, tables, patient_features, training_table)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create patient feature and patient-condition-medication tables."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("Datasets"),
        help="Directory containing the raw CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Datasets") / "processed",
        help="Directory where processed artifacts will be written.",
    )
    parser.add_argument(
        "--max-candidates-per-condition",
        type=int,
        default=20,
        help="Keep the top N medication candidates per condition. Use 0 for all.",
    )
    parser.add_argument(
        "--min-candidate-count",
        type=int,
        default=5,
        help="Minimum observed prescriptions required for a medication candidate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_preprocessing(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_candidates_per_condition=args.max_candidates_per_condition,
        min_candidate_count=args.min_candidate_count,
    )


if __name__ == "__main__":
    main()
