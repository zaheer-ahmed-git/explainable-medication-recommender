"""Build Milestone 6 candidate catalogs and observed-label training tables."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import duckdb

from pipeline.config import (
    COHORT_VERSION,
    DEFAULT_MODELING_PARAMETERS,
    DUCKDB_MEMORY_LIMIT,
    DUCKDB_TEMP_DIR,
    DUCKDB_THREADS,
    FEATURE_VERSION,
    FEATURES_ROOT,
    HARMONIZATION_VERSION,
    HARMONIZED_ROOT,
    LABEL_VERSION,
    MEDICATION_MAPPING_VERSION,
    REPORTS_ROOT,
    SPLIT_VERSION,
    TRAINING_ROOT,
)
from pipeline.extract_utils import (
    configure_duckdb_connection,
    parquet_scan,
    safe_error_message,
    sql_string,
)
from pipeline.features import (
    copy_query_to_parquet,
    event_hours_sql,
    fetch_dict_rows,
)


SCHEMA_VERSION = "milestone6-training-table-manifest-v1"
DEFAULT_MANIFEST_PATH = REPORTS_ROOT / "training_table_manifest.json"
CANDIDATE_TOKEN_STRATEGIES = ("rxnorm_or_atc", "atc3_or_rxnorm")

REQUIRED_HARMONIZED_TABLES = ("conditions", "medications")
REQUIRED_FEATURE_TABLES = ("cohort_decision_times", "patient_stay_features")


@dataclass(frozen=True)
class TrainingTableBuildConfig:
    """Configuration for candidate catalog and training-table construction."""

    harmonized_root: Path = HARMONIZED_ROOT
    features_root: Path = FEATURES_ROOT
    training_root: Path = TRAINING_ROOT
    manifest_path: Path = DEFAULT_MANIFEST_PATH
    candidate_top_n_per_condition: int = int(
        DEFAULT_MODELING_PARAMETERS["candidate_top_n_per_condition"]
    )
    prediction_offset_hours: int = int(
        DEFAULT_MODELING_PARAMETERS["prediction_offset_hours"]
    )
    label_window_hours: int = int(DEFAULT_MODELING_PARAMETERS["label_window_hours"])
    split_seed: int = int(DEFAULT_MODELING_PARAMETERS["split_seed"])
    cohort_version: str = COHORT_VERSION
    harmonization_version: str = HARMONIZATION_VERSION
    medication_mapping_version: str = MEDICATION_MAPPING_VERSION
    feature_version: str = FEATURE_VERSION
    label_version: str = LABEL_VERSION
    split_version: str = SPLIT_VERSION
    candidate_token_strategy: str = str(
        DEFAULT_MODELING_PARAMETERS.get("candidate_token_strategy", "rxnorm_or_atc")
    )
    duckdb_temp_directory: Path | None = DUCKDB_TEMP_DIR
    duckdb_memory_limit: str | None = DUCKDB_MEMORY_LIMIT
    duckdb_threads: int | None = DUCKDB_THREADS

    @property
    def cohort_decision_times_path(self) -> Path:
        return self.features_root / "cohort_decision_times.parquet"

    @property
    def patient_stay_features_path(self) -> Path:
        return self.features_root / "patient_stay_features.parquet"

    @property
    def split_manifest_path(self) -> Path:
        return self.training_root / "split_manifest.parquet"

    @property
    def candidate_catalog_path(self) -> Path:
        return self.training_root / "candidate_catalog.parquet"

    @property
    def patient_condition_medication_path(self) -> Path:
        return self.training_root / "patient_condition_medication.parquet"

    @property
    def label_window_end_hours(self) -> int:
        return self.prediction_offset_hours + self.label_window_hours


def harmonized_path(config: TrainingTableBuildConfig, table_name: str) -> Path:
    """Return a harmonized table path."""

    return config.harmonized_root / f"{table_name}.parquet"


def feature_path(config: TrainingTableBuildConfig, table_name: str) -> Path:
    """Return a feature table path."""

    return config.features_root / f"{table_name}.parquet"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write stable JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def configure_connection(
    config: TrainingTableBuildConfig,
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Apply shared memory-safe DuckDB settings."""

    configure_duckdb_connection(
        connection,
        temp_directory=config.duckdb_temp_directory,
        memory_limit=config.duckdb_memory_limit,
        threads=config.duckdb_threads,
    )


def missing_input_tables(config: TrainingTableBuildConfig) -> list[dict[str, str]]:
    """Return missing harmonized or feature inputs."""

    missing: list[dict[str, str]] = []
    for table_name in REQUIRED_HARMONIZED_TABLES:
        path = harmonized_path(config, table_name)
        if not path.exists():
            missing.append({"table_name": table_name, "path": str(path)})
    for table_name in REQUIRED_FEATURE_TABLES:
        path = feature_path(config, table_name)
        if not path.exists():
            missing.append({"table_name": table_name, "path": str(path)})
    return missing


def validate_config(config: TrainingTableBuildConfig) -> None:
    """Validate training-table options before materialization."""

    if config.candidate_token_strategy not in CANDIDATE_TOKEN_STRATEGIES:
        raise ValueError(
            "candidate_token_strategy must be one of "
            + ", ".join(CANDIDATE_TOKEN_STRATEGIES)
        )


def input_join_integrity_query(config: TrainingTableBuildConfig) -> str:
    """Return aggregate join-integrity checks for modeling inputs."""

    conditions = harmonized_path(config, "conditions")
    medications = harmonized_path(config, "medications")
    features = config.patient_stay_features_path
    decision = config.cohort_decision_times_path

    def table_check(table_name: str, table_path: Path) -> str:
        return f"""
SELECT
    {sql_string(table_name)} AS table_name,
    COUNT(*) AS row_count,
    SUM(CASE
        WHEN t.source IS NULL OR t.stay_uid IS NULL THEN 1 ELSE 0
    END) AS invalid_join_key_row_count,
    SUM(CASE
        WHEN t.source IS NOT NULL
            AND t.stay_uid IS NOT NULL
            AND d.stay_uid IS NULL
        THEN 1 ELSE 0
    END) AS orphan_row_count
FROM {parquet_scan(table_path)} AS t
LEFT JOIN {parquet_scan(decision)} AS d
    ON t.source = d.source
    AND t.stay_uid = d.stay_uid
"""

    return "\nUNION ALL\n".join(
        (
            table_check("conditions", conditions),
            table_check("medications", medications),
            table_check("patient_stay_features", features),
        )
    )


def input_join_integrity_has_failures(rows: Sequence[dict[str, Any]]) -> bool:
    """Return whether any aggregate join-integrity check failed."""

    return any(
        int(row.get("invalid_join_key_row_count") or 0) > 0
        or int(row.get("orphan_row_count") or 0) > 0
        for row in rows
    )


def condition_token_sql(alias: str = "c") -> str:
    """Return the index condition token expression."""

    return (
        f"COALESCE(NULLIF(TRIM(CAST({alias}.project_condition_token AS VARCHAR)), ''), "
        f"NULLIF(TRIM(CAST({alias}.normalized_condition_token AS VARCHAR)), ''))"
    )


def atc3_key_sql(alias: str = "m") -> str:
    """Return normalized ATC-3 key SQL from an ATC code column."""

    return (
        "NULLIF(SUBSTR(REGEXP_REPLACE(UPPER(TRIM(CAST("
        f"{alias}.atc_code AS VARCHAR))), '[^A-Z0-9]+', '', 'g'), 1, 4), '')"
    )


def rxcui_key_sql(alias: str = "m") -> str:
    """Return normalized RxCUI key SQL."""

    return f"NULLIF(TRIM(CAST({alias}.rxcui AS VARCHAR)), '')"


def medication_token_sql(alias: str = "m", *, strategy: str = "rxnorm_or_atc") -> str:
    """Return the canonical candidate medication token expression."""

    rxcui = rxcui_key_sql(alias)
    atc3 = atc3_key_sql(alias)
    if strategy == "rxnorm_or_atc":
        return f"""CASE
        WHEN {rxcui} IS NOT NULL THEN 'rxnorm:' || {rxcui}
        WHEN {atc3} IS NOT NULL THEN 'atc:' || {atc3}
        ELSE NULL
    END"""
    if strategy == "atc3_or_rxnorm":
        return f"""CASE
        WHEN {atc3} IS NOT NULL THEN 'atc:' || {atc3}
        WHEN {rxcui} IS NOT NULL THEN 'rxnorm:' || {rxcui}
        ELSE NULL
    END"""
    raise ValueError(
        "candidate_token_strategy must be one of "
        + ", ".join(CANDIDATE_TOKEN_STRATEGIES)
    )


def medication_display_name_sql(alias: str = "m") -> str:
    """Return a non-key medication display expression for local artifacts."""

    return (
        f"COALESCE(NULLIF(TRIM(CAST({alias}.ingredient_name AS VARCHAR)), ''), "
        f"NULLIF(TRIM(CAST({alias}.rxnorm_name AS VARCHAR)), ''), "
        f"NULLIF(TRIM(CAST({alias}.atc_code AS VARCHAR)), ''))"
    )


def stay_conditions_cte(config: TrainingTableBuildConfig) -> str:
    """Return the common stay-condition CTE SQL."""

    conditions = harmonized_path(config, "conditions")
    token = condition_token_sql("c")
    return f"""
stay_conditions AS (
    SELECT DISTINCT
        c.source,
        c.patient_uid,
        c.encounter_uid,
        c.stay_uid,
        {token} AS index_condition_token,
        COALESCE(
            NULLIF(TRIM(CAST(c.project_condition_group AS VARCHAR)), ''),
            NULLIF(TRIM(CAST(c.normalized_condition_name AS VARCHAR)), ''),
            NULLIF(TRIM(CAST(c.condition_text AS VARCHAR)), ''),
            {token}
        ) AS index_condition_name,
        c.condition_rollup_level,
        c.mapping_status AS condition_mapping_status
    FROM {parquet_scan(conditions)} AS c
)"""


def medication_label_events_cte(config: TrainingTableBuildConfig) -> str:
    """Return label-window medication event CTEs."""

    medications = harmonized_path(config, "medications")
    decision = config.cohort_decision_times_path
    event_hours = event_hours_sql(
        source_expr="m.source",
        event_time_expr="m.event_start_time",
        event_offset_expr="CAST(NULL AS VARCHAR)",
        stay_start_expr="d.stay_start_timestamp",
    )
    token = medication_token_sql("m", strategy=config.candidate_token_strategy)
    display = medication_display_name_sql("m")
    prediction = config.prediction_offset_hours
    label_end = config.label_window_end_hours
    return f"""
medication_events AS (
    SELECT
        m.source,
        m.source_version,
        m.patient_uid,
        m.encounter_uid,
        m.stay_uid,
        d.split,
        d.primary_training_eligible,
        {event_hours} AS medication_start_hours_from_admit,
        {token} AS candidate_medication_token,
        {display} AS candidate_medication_name,
        m.rxcui,
        m.atc_code,
        m.mapping_status,
        m.source_event_id
    FROM {parquet_scan(medications)} AS m
    INNER JOIN {parquet_scan(decision)} AS d
        ON m.stay_uid = d.stay_uid
        AND m.source = d.source
),
label_window_medications AS (
    SELECT *
    FROM medication_events
    WHERE primary_training_eligible
        AND medication_start_hours_from_admit > {prediction}
        AND medication_start_hours_from_admit <= {label_end}
)"""


def split_manifest_query(config: TrainingTableBuildConfig) -> str:
    """Build one patient-level split manifest row per patient."""

    decision = config.cohort_decision_times_path
    return f"""
SELECT
    source,
    patient_uid,
    split,
    COUNT(*) AS stay_count,
    {sql_string(config.cohort_version)} AS cohort_version,
    {sql_string(config.split_version)} AS split_version,
    {config.split_seed} AS split_seed
FROM {parquet_scan(decision)}
GROUP BY source, patient_uid, split
"""


def candidate_catalog_query(
    config: TrainingTableBuildConfig,
    *,
    generated_at: str,
) -> str:
    """Build train-only condition-specific candidate medication catalogs."""

    return f"""
WITH
{stay_conditions_cte(config)},
{medication_label_events_cte(config)},
train_positive_condition_meds AS (
    SELECT
        sc.index_condition_token,
        MIN(sc.index_condition_name) AS index_condition_name,
        lm.candidate_medication_token,
        MIN(lm.candidate_medication_name) AS candidate_medication_name,
        MIN(lm.rxcui) AS rxcui,
        MIN(lm.atc_code) AS atc_code,
        COUNT(*) AS positive_train_event_count,
        COUNT(DISTINCT lm.stay_uid) AS positive_train_stay_count
    FROM label_window_medications AS lm
    INNER JOIN stay_conditions AS sc
        ON lm.stay_uid = sc.stay_uid
        AND lm.source = sc.source
    WHERE lm.source = 'mimiciv'
        AND lm.split = 'train'
        AND sc.index_condition_token IS NOT NULL
        AND lm.candidate_medication_token IS NOT NULL
    GROUP BY
        sc.index_condition_token,
        lm.candidate_medication_token
),
ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY index_condition_token
            ORDER BY positive_train_stay_count DESC,
                positive_train_event_count DESC,
                candidate_medication_token
        ) AS candidate_rank
    FROM train_positive_condition_meds
)
SELECT
    index_condition_token,
    index_condition_name,
    candidate_medication_token,
    candidate_medication_name,
    rxcui,
    atc_code,
    candidate_rank,
    positive_train_stay_count,
    positive_train_event_count,
    {sql_string(config.medication_mapping_version)} AS medication_mapping_version,
    {sql_string(config.label_version)} AS label_version,
    {sql_string(config.split_version)} AS split_version,
    {config.split_seed} AS split_seed,
    {sql_string(generated_at)} AS generated_at
FROM ranked
WHERE candidate_rank <= {config.candidate_top_n_per_condition}
"""


def patient_condition_medication_query(
    config: TrainingTableBuildConfig,
    *,
    generated_at: str,
) -> str:
    """Build one row per eligible stay, condition, and catalog candidate."""

    decision = config.cohort_decision_times_path
    catalog = config.candidate_catalog_path
    return f"""
WITH
decision AS (
    SELECT *
    FROM {parquet_scan(decision)}
    WHERE primary_training_eligible
),
{stay_conditions_cte(config)},
{medication_label_events_cte(config)},
positive_labels AS (
    SELECT
        lm.source,
        lm.patient_uid,
        lm.encounter_uid,
        lm.stay_uid,
        sc.index_condition_token,
        lm.candidate_medication_token,
        MIN(lm.medication_start_hours_from_admit)
            AS label_first_observed_hours_from_admit,
        COUNT(*) AS label_event_count
    FROM label_window_medications AS lm
    INNER JOIN stay_conditions AS sc
        ON lm.stay_uid = sc.stay_uid
        AND lm.source = sc.source
    WHERE sc.index_condition_token IS NOT NULL
        AND lm.candidate_medication_token IS NOT NULL
    GROUP BY
        lm.source,
        lm.patient_uid,
        lm.encounter_uid,
        lm.stay_uid,
        sc.index_condition_token,
        lm.candidate_medication_token
)
SELECT
    d.source,
    d.source_version,
    d.patient_uid,
    d.encounter_uid,
    d.stay_uid,
    d.split,
    sc.index_condition_token,
    sc.index_condition_name,
    catalog.candidate_medication_token,
    catalog.candidate_medication_name,
    catalog.candidate_rank,
    d.stay_uid || '|' || sc.index_condition_token AS ranking_group_id,
    COALESCE(labels.label_event_count, 0) > 0 AS label_prescribed,
    labels.label_first_observed_hours_from_admit,
    COALESCE(labels.label_event_count, 0) AS label_event_count,
    {sql_string("observed_prescription_positive_or_weak_unobserved_negative")}
        AS label_semantics,
    d.prediction_time_hours_from_admit,
    d.label_window_end_hours_from_admit,
    {sql_string(config.cohort_version)} AS cohort_version,
    {sql_string(config.harmonization_version)} AS harmonization_version,
    {sql_string(config.feature_version)} AS feature_version,
    {sql_string(config.label_version)} AS label_version,
    {sql_string(config.split_version)} AS split_version,
    {config.split_seed} AS split_seed,
    {sql_string(generated_at)} AS generated_at
FROM decision AS d
INNER JOIN stay_conditions AS sc
    ON d.stay_uid = sc.stay_uid
    AND d.source = sc.source
INNER JOIN {parquet_scan(catalog)} AS catalog
    ON sc.index_condition_token = catalog.index_condition_token
LEFT JOIN positive_labels AS labels
    ON d.stay_uid = labels.stay_uid
    AND d.source = labels.source
    AND sc.index_condition_token = labels.index_condition_token
    AND catalog.candidate_medication_token = labels.candidate_medication_token
WHERE sc.index_condition_token IS NOT NULL
"""


def base_manifest(
    config: TrainingTableBuildConfig,
    *,
    status: str,
    generated_at: str,
) -> dict[str, Any]:
    """Build the common training-table manifest shell."""

    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "generated_at": generated_at,
        "data_safety": {
            "manifest_contains_patient_rows": False,
            "local_artifacts_contain_patient_level_rows": True,
            "artifact_storage": "ignored Dataset/processed/training",
        },
        "parameters": {
            "candidate_top_n_per_condition": config.candidate_top_n_per_condition,
            "candidate_token_strategy": config.candidate_token_strategy,
            "prediction_offset_hours": config.prediction_offset_hours,
            "label_window_hours": config.label_window_hours,
            "split_seed": config.split_seed,
            "development_source": "mimiciv",
            "external_validation_source": "eicu_crd",
        },
        "label_caveat": (
            "Labels are observed historical medication starts in the label "
            "window. Unobserved catalog candidates are weak observational "
            "negatives, not proof of clinical non-indication."
        ),
        "versions": {
            "cohort_version": config.cohort_version,
            "harmonization_version": config.harmonization_version,
            "feature_version": config.feature_version,
            "label_version": config.label_version,
            "split_version": config.split_version,
            "medication_mapping_version": config.medication_mapping_version,
        },
        "artifacts": {},
        "tables": [],
    }


def artifact_record(
    connection: duckdb.DuckDBPyConnection,
    *,
    table_name: str,
    query: str,
    output_path: Path,
) -> dict[str, Any]:
    """Materialize one training artifact and return an aggregate record."""

    try:
        row_count = copy_query_to_parquet(connection, query, output_path)
    except Exception as error:
        return {
            "table_name": table_name,
            "output_path": str(output_path),
            "status": "failed",
            "row_count": None,
            "reason": safe_error_message(error),
        }
    return {
        "table_name": table_name,
        "output_path": str(output_path),
        "status": "completed",
        "row_count": row_count,
    }


def split_integrity_query(config: TrainingTableBuildConfig) -> str:
    """Return aggregate patient split-integrity counts."""

    split_manifest = config.split_manifest_path
    return f"""
WITH per_patient AS (
    SELECT patient_uid, COUNT(DISTINCT split) AS split_count
    FROM {parquet_scan(split_manifest)}
    GROUP BY patient_uid
)
SELECT
    COUNT(*) AS patient_count,
    SUM(CASE WHEN split_count > 1 THEN 1 ELSE 0 END) AS patients_with_multiple_splits
FROM per_patient
"""


def condition_coverage_loss_query(config: TrainingTableBuildConfig) -> str:
    """Aggregate condition rows unavailable for catalog expansion."""

    conditions = harmonized_path(config, "conditions")
    token = condition_token_sql("c")
    return f"""
SELECT
    c.source,
    COUNT(*) AS condition_row_count,
    SUM(CASE WHEN {token} IS NULL THEN 1 ELSE 0 END)
        AS missing_index_condition_token_rows,
    COUNT(DISTINCT {token}) AS index_condition_count
FROM {parquet_scan(conditions)} AS c
GROUP BY c.source
ORDER BY c.source
"""


def medication_label_loss_query(config: TrainingTableBuildConfig) -> str:
    """Aggregate label-window medication coverage losses."""

    prediction = config.prediction_offset_hours
    label_end = config.label_window_end_hours
    return f"""
WITH
{medication_label_events_cte(config)}
SELECT
    source,
    split,
    SUM(CASE
        WHEN medication_start_hours_from_admit > {prediction}
            AND medication_start_hours_from_admit <= {label_end}
        THEN 1 ELSE 0
    END) AS label_window_medication_event_count,
    SUM(CASE
        WHEN medication_start_hours_from_admit > {prediction}
            AND medication_start_hours_from_admit <= {label_end}
            AND candidate_medication_token IS NULL
        THEN 1 ELSE 0
    END)
        AS unmapped_candidate_token_events,
    SUM(CASE WHEN medication_start_hours_from_admit IS NULL THEN 1 ELSE 0 END)
        AS missing_medication_start_time_events
FROM medication_events
WHERE primary_training_eligible
GROUP BY source, split
ORDER BY source, split
"""


def out_of_catalog_positive_query(config: TrainingTableBuildConfig) -> str:
    """Aggregate positives that are absent from train-only candidate catalogs."""

    catalog = config.candidate_catalog_path
    return f"""
WITH
{stay_conditions_cte(config)},
{medication_label_events_cte(config)},
positive_condition_meds AS (
    SELECT DISTINCT
        lm.source,
        lm.split,
        sc.index_condition_token,
        lm.candidate_medication_token,
        lm.stay_uid
    FROM label_window_medications AS lm
    INNER JOIN stay_conditions AS sc
        ON lm.stay_uid = sc.stay_uid
        AND lm.source = sc.source
    WHERE sc.index_condition_token IS NOT NULL
        AND lm.candidate_medication_token IS NOT NULL
)
SELECT
    pcm.source,
    pcm.split,
    COUNT(*) AS positive_condition_medication_stay_count,
    SUM(CASE WHEN catalog.candidate_medication_token IS NULL THEN 1 ELSE 0 END)
        AS out_of_catalog_positive_stay_count
FROM positive_condition_meds AS pcm
LEFT JOIN {parquet_scan(catalog)} AS catalog
    ON pcm.index_condition_token = catalog.index_condition_token
    AND pcm.candidate_medication_token = catalog.candidate_medication_token
GROUP BY pcm.source, pcm.split
ORDER BY pcm.source, pcm.split
"""


def append_manifest_summaries(
    connection: duckdb.DuckDBPyConnection,
    config: TrainingTableBuildConfig,
    manifest: dict[str, Any],
) -> None:
    """Attach aggregate-only summaries to a completed manifest."""

    split_manifest = config.split_manifest_path
    catalog = config.candidate_catalog_path
    table = config.patient_condition_medication_path
    manifest["split_counts"] = fetch_dict_rows(
        connection,
        f"""
SELECT source, split, COUNT(*) AS patient_count, SUM(stay_count) AS stay_count
FROM {parquet_scan(split_manifest)}
GROUP BY source, split
ORDER BY source, split
""",
    )
    manifest["split_integrity"] = fetch_dict_rows(
        connection,
        split_integrity_query(config),
    )[0]
    manifest["candidate_catalog_counts"] = fetch_dict_rows(
        connection,
        f"""
SELECT
    COUNT(DISTINCT index_condition_token) AS condition_count,
    COUNT(*) AS candidate_count,
    MAX(candidate_rank) AS max_candidate_rank
FROM {parquet_scan(catalog)}
""",
    )[0]
    manifest["training_rows_by_source_split"] = fetch_dict_rows(
        connection,
        f"""
SELECT
    source,
    split,
    COUNT(*) AS row_count,
    SUM(CASE WHEN label_prescribed THEN 1 ELSE 0 END) AS positive_row_count,
    COUNT(DISTINCT ranking_group_id) AS ranking_group_count
FROM {parquet_scan(table)}
GROUP BY source, split
ORDER BY source, split
""",
    )
    manifest["condition_coverage_loss"] = fetch_dict_rows(
        connection,
        condition_coverage_loss_query(config),
    )
    manifest["medication_label_loss"] = fetch_dict_rows(
        connection,
        medication_label_loss_query(config),
    )
    manifest["out_of_catalog_positives"] = fetch_dict_rows(
        connection,
        out_of_catalog_positive_query(config),
    )


def build_training_artifacts(
    config: TrainingTableBuildConfig = TrainingTableBuildConfig(),
) -> dict[str, Any]:
    """Build local Milestone 6 training artifacts and aggregate manifest."""

    generated_at = datetime.now(UTC).isoformat()
    config.training_root.mkdir(parents=True, exist_ok=True)
    validate_config(config)
    missing = missing_input_tables(config)
    if missing:
        manifest = base_manifest(
            config,
            status="failed_missing_inputs",
            generated_at=generated_at,
        )
        manifest["missing_inputs"] = missing
        write_json(config.manifest_path, manifest)
        return manifest

    manifest = base_manifest(config, status="completed", generated_at=generated_at)
    tables: list[dict[str, Any]] = []
    with duckdb.connect(database=":memory:") as connection:
        configure_connection(config, connection)
        manifest["join_integrity"] = fetch_dict_rows(
            connection,
            input_join_integrity_query(config),
        )
        if input_join_integrity_has_failures(manifest["join_integrity"]):
            manifest["status"] = "failed_join_integrity"
            manifest["tables"] = tables
            write_json(config.manifest_path, manifest)
            return manifest

        build_specs = (
            (
                "split_manifest",
                split_manifest_query(config),
                config.split_manifest_path,
            ),
            (
                "candidate_catalog",
                candidate_catalog_query(config, generated_at=generated_at),
                config.candidate_catalog_path,
            ),
            (
                "patient_condition_medication",
                patient_condition_medication_query(config, generated_at=generated_at),
                config.patient_condition_medication_path,
            ),
        )
        for table_name, query, output_path in build_specs:
            record = artifact_record(
                connection,
                table_name=table_name,
                query=query,
                output_path=output_path,
            )
            tables.append(record)
            if record["status"] == "completed":
                manifest["artifacts"][table_name] = str(output_path)

        if any(table["status"] == "failed" for table in tables):
            manifest["status"] = "failed"
        else:
            append_manifest_summaries(connection, config, manifest)
            if (
                int(
                    manifest["split_integrity"].get("patients_with_multiple_splits")
                    or 0
                )
                > 0
            ):
                manifest["status"] = "failed_split_integrity"

    manifest["tables"] = tables
    write_json(config.manifest_path, manifest)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Milestone 6 candidate catalogs and training table.",
    )
    parser.add_argument(
        "--harmonized-root",
        type=Path,
        default=HARMONIZED_ROOT,
        help="Root directory containing Milestone 5 harmonized Parquet tables.",
    )
    parser.add_argument(
        "--features-root",
        type=Path,
        default=FEATURES_ROOT,
        help="Root directory containing Milestone 6 feature artifacts.",
    )
    parser.add_argument(
        "--training-root",
        type=Path,
        default=TRAINING_ROOT,
        help="Output directory for local ignored training artifacts.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Output path for the aggregate training-table manifest.",
    )
    parser.add_argument(
        "--candidate-top-n-per-condition",
        type=int,
        default=int(DEFAULT_MODELING_PARAMETERS["candidate_top_n_per_condition"]),
        help="Maximum number of train-derived candidates per condition.",
    )
    parser.add_argument(
        "--candidate-token-strategy",
        choices=CANDIDATE_TOKEN_STRATEGIES,
        default=str(
            DEFAULT_MODELING_PARAMETERS.get("candidate_token_strategy", "rxnorm_or_atc")
        ),
        help=(
            "Medication token granularity for labels and candidates. "
            "rxnorm_or_atc preserves ingredient-first behavior; "
            "atc3_or_rxnorm builds an ATC-class-first catalog for coverage "
            "sensitivity analyses."
        ),
    )
    parser.add_argument(
        "--prediction-offset-hours",
        type=int,
        default=int(DEFAULT_MODELING_PARAMETERS["prediction_offset_hours"]),
        help="Feature cutoff in hours from ICU/unit admission.",
    )
    parser.add_argument(
        "--label-window-hours",
        type=int,
        default=int(DEFAULT_MODELING_PARAMETERS["label_window_hours"]),
        help="Label window duration after prediction time.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=int(DEFAULT_MODELING_PARAMETERS["split_seed"]),
        help="Seed recorded with split artifacts.",
    )
    parser.add_argument(
        "--duckdb-temp-dir",
        type=Path,
        default=DUCKDB_TEMP_DIR,
        help="Directory DuckDB may use to spill larger-than-memory operators.",
    )
    parser.add_argument(
        "--duckdb-memory-limit",
        default=DUCKDB_MEMORY_LIMIT,
        help="Optional DuckDB memory ceiling, e.g. '24GB'.",
    )
    parser.add_argument(
        "--duckdb-threads",
        type=int,
        default=DUCKDB_THREADS,
        help="Optional DuckDB thread cap.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_training_artifacts(
        TrainingTableBuildConfig(
            harmonized_root=args.harmonized_root,
            features_root=args.features_root,
            training_root=args.training_root,
            manifest_path=args.manifest,
            candidate_top_n_per_condition=args.candidate_top_n_per_condition,
            candidate_token_strategy=args.candidate_token_strategy,
            prediction_offset_hours=args.prediction_offset_hours,
            label_window_hours=args.label_window_hours,
            split_seed=args.split_seed,
            duckdb_temp_directory=args.duckdb_temp_dir,
            duckdb_memory_limit=args.duckdb_memory_limit,
            duckdb_threads=args.duckdb_threads,
        )
    )
    print(
        "Wrote Milestone 6 training manifest: "
        f"status={manifest['status']}, "
        f"tables={len(manifest.get('tables', []))}"
    )
    if manifest["status"] == "failed_missing_inputs":
        return 2
    return 1 if manifest["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
