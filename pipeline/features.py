"""Build Milestone 6 temporal feature artifacts from harmonized tables."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, replace
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
    REPORTS_ROOT,
    SPLIT_VERSION,
)
from pipeline.extract_utils import (
    configure_duckdb_connection,
    parquet_scan,
    safe_error_message,
    sql_string,
)


SCHEMA_VERSION = "milestone6-feature-manifest-v1"
DEFAULT_MANIFEST_PATH = REPORTS_ROOT / "milestone6_feature_manifest.json"
BASELINE_FEATURE_SET = "baseline"
PHASE8_P0_FEATURE_SET = "phase8_p0"
FEATURE_SETS = (BASELINE_FEATURE_SET, PHASE8_P0_FEATURE_SET)
PHASE8_P0_FEATURE_VERSION = "temporal-features-v2"

REQUIRED_HARMONIZED_TABLES = (
    "cohort_stays",
    "demographics",
    "labs",
    "vitals",
    "allergies",
    "interventions",
    "temporal_events",
)

DEFAULT_CORE_LAB_TOKENS = (
    "creatinine",
    "lactate",
    "wbc",
    "platelets",
    "sodium",
    "potassium",
    "glucose",
)
DEFAULT_CORE_VITAL_TOKENS = (
    "heart_rate",
    "mean_arterial_pressure",
    "spo2",
    "temperature",
    "respiratory_rate",
)
DEFAULT_STAY_FEATURE_BATCHES = 1
DEFAULT_EVENT_SEQUENCE_BATCHES = 8
DEFAULT_CONDITION_FEATURE_TOP_N = 40
DEFAULT_TREND_MIN_EVENTS = 2
TREND_SUMMARY_SUFFIXES = (
    "first_24h",
    "last_24h",
    "delta_24h",
    "slope_24h",
    "hours_since_last_24h",
)


@dataclass(frozen=True)
class FeatureBuildConfig:
    """Configuration for temporal feature artifact construction."""

    harmonized_root: Path = HARMONIZED_ROOT
    features_root: Path = FEATURES_ROOT
    manifest_path: Path = DEFAULT_MANIFEST_PATH
    prediction_offset_hours: int = int(
        DEFAULT_MODELING_PARAMETERS["prediction_offset_hours"]
    )
    label_window_hours: int = int(DEFAULT_MODELING_PARAMETERS["label_window_hours"])
    split_seed: int = int(DEFAULT_MODELING_PARAMETERS["split_seed"])
    feature_set: str = BASELINE_FEATURE_SET
    feature_version: str = FEATURE_VERSION
    split_version: str = SPLIT_VERSION
    cohort_version: str = COHORT_VERSION
    harmonization_version: str = HARMONIZATION_VERSION
    core_lab_tokens: tuple[str, ...] = DEFAULT_CORE_LAB_TOKENS
    core_vital_tokens: tuple[str, ...] = DEFAULT_CORE_VITAL_TOKENS
    condition_feature_top_n: int = DEFAULT_CONDITION_FEATURE_TOP_N
    trend_min_events: int = DEFAULT_TREND_MIN_EVENTS
    condition_vocabulary: tuple[str, ...] = ()
    include_predecision_medications: bool = False
    stay_feature_batches: int = DEFAULT_STAY_FEATURE_BATCHES
    event_sequence_batches: int = DEFAULT_EVENT_SEQUENCE_BATCHES
    duckdb_temp_directory: Path | None = DUCKDB_TEMP_DIR
    duckdb_memory_limit: str | None = DUCKDB_MEMORY_LIMIT
    duckdb_threads: int | None = DUCKDB_THREADS

    def __post_init__(self) -> None:
        if self.feature_set not in FEATURE_SETS:
            allowed = ", ".join(FEATURE_SETS)
            raise ValueError(f"feature_set must be one of: {allowed}")
        if self.condition_feature_top_n < 0:
            raise ValueError("condition_feature_top_n must be non-negative")
        if self.trend_min_events < 2:
            raise ValueError("trend_min_events must be at least 2")
        if self.feature_set == PHASE8_P0_FEATURE_SET:
            object.__setattr__(self, "feature_version", PHASE8_P0_FEATURE_VERSION)

    @property
    def cohort_decision_times_path(self) -> Path:
        return self.features_root / "cohort_decision_times.parquet"

    @property
    def patient_stay_features_path(self) -> Path:
        return self.features_root / "patient_stay_features.parquet"

    @property
    def event_sequences_path(self) -> Path:
        return self.features_root / "event_sequences.parquet"

    @property
    def label_window_end_hours(self) -> int:
        return self.prediction_offset_hours + self.label_window_hours


def harmonized_path(config: FeatureBuildConfig, table_name: str) -> Path:
    """Return a harmonized table path."""

    return config.harmonized_root / f"{table_name}.parquet"


def event_sequence_batch_count(config: FeatureBuildConfig) -> int:
    """Return a positive event-sequence batch count."""

    return max(1, int(config.event_sequence_batches))


def stay_feature_batch_count(config: FeatureBuildConfig) -> int:
    """Return a positive stay-feature batch count."""

    return max(1, int(config.stay_feature_batches))


def stay_feature_part_path(config: FeatureBuildConfig, batch_index: int) -> Path:
    """Return a generated stay-feature part path."""

    return (
        config.features_root
        / "_patient_stay_feature_parts"
        / (f"patient_stay_features_part_{batch_index:04d}.parquet")
    )


def event_sequence_staging_path(config: FeatureBuildConfig) -> Path:
    """Return the generated pre-decision event-sequence staging path."""

    return config.features_root / "_event_sequences_predecision.parquet"


def event_sequence_part_path(config: FeatureBuildConfig, batch_index: int) -> Path:
    """Return a generated event-sequence part path."""

    return (
        config.features_root
        / "_event_sequence_parts"
        / (f"event_sequences_part_{batch_index:04d}.parquet")
    )


def parquet_scan_paths(paths: Sequence[Path]) -> str:
    """Return a DuckDB Parquet scan expression for multiple paths."""

    path_list = ", ".join(sql_string(path) for path in paths)
    return f"read_parquet([{path_list}])"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write stable JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def configure_connection(
    config: FeatureBuildConfig,
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Apply shared memory-safe DuckDB settings."""

    configure_duckdb_connection(
        connection,
        temp_directory=config.duckdb_temp_directory,
        memory_limit=config.duckdb_memory_limit,
        threads=config.duckdb_threads,
    )


def copy_query_to_parquet(
    connection: duckdb.DuckDBPyConnection,
    query: str,
    output_path: Path,
) -> int:
    """Materialize a query to Parquet and return its row count."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S%f")
    temporary_path = output_path.with_name(
        f".{output_path.stem}.tmp-{timestamp}{output_path.suffix}"
    )
    try:
        connection.execute(
            f"COPY ({query}) TO {sql_string(temporary_path)} (FORMAT PARQUET)"
        )
        temporary_path.replace(output_path)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise
    row = connection.execute(
        f"SELECT COUNT(*) FROM {parquet_scan(output_path)}"
    ).fetchone()
    return int(row[0]) if row is not None else 0


def fetch_dict_rows(
    connection: duckdb.DuckDBPyConnection,
    query: str,
) -> list[dict[str, Any]]:
    """Run a query and return dictionaries."""

    cursor = connection.execute(query)
    columns = [description[0] for description in cursor.description]
    return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]


def split_bucket_sql(*, seed: int, patient_expr: str = "patient_uid") -> str:
    """Return deterministic patient split bucket SQL."""

    return (
        f"(HASH(COALESCE(CAST({patient_expr} AS VARCHAR), '') || "
        f"'|' || {sql_string(str(seed))}) % 10000)"
    )


def split_case_sql(*, seed: int, source_expr: str = "source") -> str:
    """Return the Milestone 6 deterministic split expression."""

    bucket = split_bucket_sql(seed=seed)
    return f"""CASE
        WHEN {source_expr} = 'eicu_crd' THEN 'external'
        WHEN {bucket} < 8000 THEN 'train'
        WHEN {bucket} < 9000 THEN 'validation'
        ELSE 'test'
    END"""


def stay_end_hours_sql() -> str:
    """Return SQL that normalizes stay observation duration to hours."""

    return """
CASE
    WHEN TRY_CAST(los_hours AS DOUBLE) IS NOT NULL THEN TRY_CAST(los_hours AS DOUBLE)
    WHEN TRY_CAST(stay_end_offset_minutes AS DOUBLE) IS NOT NULL
        THEN TRY_CAST(stay_end_offset_minutes AS DOUBLE) / 60.0
    WHEN TRY_CAST(stay_start_time AS TIMESTAMP) IS NOT NULL
        AND TRY_CAST(stay_end_time AS TIMESTAMP) IS NOT NULL
        THEN DATE_DIFF(
            'second',
            TRY_CAST(stay_start_time AS TIMESTAMP),
            TRY_CAST(stay_end_time AS TIMESTAMP)
        ) / 3600.0
    ELSE NULL
END"""


def event_hours_sql(
    *,
    source_expr: str,
    event_time_expr: str,
    event_offset_expr: str,
    stay_start_expr: str,
) -> str:
    """Return SQL normalizing event time to hours from ICU/unit admission."""

    return f"""
CASE
    WHEN TRY_CAST({event_offset_expr} AS DOUBLE) IS NOT NULL
        THEN TRY_CAST({event_offset_expr} AS DOUBLE) / 60.0
    WHEN {source_expr} = 'eicu_crd'
        AND TRY_CAST({event_time_expr} AS DOUBLE) IS NOT NULL
        THEN TRY_CAST({event_time_expr} AS DOUBLE) / 60.0
    WHEN TRY_CAST({event_time_expr} AS TIMESTAMP) IS NOT NULL
        AND {stay_start_expr} IS NOT NULL
        THEN DATE_DIFF(
            'second',
            {stay_start_expr},
            TRY_CAST({event_time_expr} AS TIMESTAMP)
        ) / 3600.0
    ELSE NULL
END"""


def _safe_feature_suffix(token: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in token)


def is_phase8_p0(config: FeatureBuildConfig) -> bool:
    """Return whether Phase 8 P0 feature families are enabled."""

    return config.feature_set == PHASE8_P0_FEATURE_SET


def required_harmonized_tables(config: FeatureBuildConfig) -> tuple[str, ...]:
    """Return harmonized tables required by this feature set."""

    if is_phase8_p0(config):
        return (*REQUIRED_HARMONIZED_TABLES, "conditions")
    return REQUIRED_HARMONIZED_TABLES


def _feature_suffixes(tokens: Sequence[str]) -> dict[str, str]:
    """Return stable SQL-safe suffixes, adding a short hash on collisions."""

    seen: set[str] = set()
    suffixes: dict[str, str] = {}
    for token in tokens:
        suffix = _safe_feature_suffix(token)
        if not suffix:
            suffix = "unknown"
        if suffix in seen:
            digest = hashlib.sha1(token.encode("utf-8")).hexdigest()[:8]
            suffix = f"{suffix}_{digest}"
        seen.add(suffix)
        suffixes[token] = suffix
    return suffixes


def condition_feature_columns(tokens: Sequence[str]) -> dict[str, str]:
    """Return Phase 8 condition-token feature column names."""

    return {
        token: f"condition_{suffix}_present_24h"
        for token, suffix in _feature_suffixes(tokens).items()
    }


def trend_column_names(*, tokens: Sequence[str], prefix: str) -> tuple[str, ...]:
    """Return Phase 8 trend column names for a lab or vital token family."""

    return tuple(
        f"{prefix}_{suffix}_{trend_suffix}"
        for suffix in _feature_suffixes(tokens).values()
        for trend_suffix in TREND_SUMMARY_SUFFIXES
    )


def missingness_column_names(*, tokens: Sequence[str], prefix: str) -> tuple[str, ...]:
    """Return Phase 8 explicit missingness column names."""

    return tuple(
        f"{prefix}_{suffix}_missing_24h"
        for suffix in _feature_suffixes(tokens).values()
    )


def _token_summary_columns(
    *,
    tokens: Sequence[str],
    token_column: str,
    value_column: str,
    prefix: str,
    include_phase8_p0: bool = False,
    event_time_column: str = "event_time_hours_from_admit",
    prediction_offset_hours: int = int(
        DEFAULT_MODELING_PARAMETERS["prediction_offset_hours"]
    ),
    trend_min_events: int = DEFAULT_TREND_MIN_EVENTS,
) -> str:
    columns: list[str] = []
    for token, suffix in _feature_suffixes(tokens).items():
        token_value = sql_string(token)
        predicate = f"{token_column} = {token_value}"
        numeric_predicate = f"{predicate} AND {value_column} IS NOT NULL"
        first_value = (
            "ARG_MIN("
            f"CASE WHEN {numeric_predicate} THEN {value_column} END, "
            f"CASE WHEN {numeric_predicate} THEN {event_time_column} END)"
        )
        last_value = (
            "ARG_MAX("
            f"CASE WHEN {numeric_predicate} THEN {value_column} END, "
            f"CASE WHEN {numeric_predicate} THEN {event_time_column} END)"
        )
        numeric_count = f"COUNT(CASE WHEN {numeric_predicate} THEN 1 END)"
        slope = (
            "REGR_SLOPE("
            f"CASE WHEN {predicate} THEN {value_column} END, "
            f"CASE WHEN {predicate} THEN {event_time_column} END)"
        )
        columns.extend(
            [
                (
                    "SUM(CASE WHEN "
                    f"{predicate} THEN 1 ELSE 0 END) "
                    f"AS {prefix}_{suffix}_count_24h"
                ),
                (
                    "MAX(CASE WHEN "
                    f"{predicate} THEN 1 ELSE 0 END) "
                    f"AS {prefix}_{suffix}_observed_24h"
                ),
                (
                    "MIN(CASE WHEN "
                    f"{predicate} THEN {value_column} END) "
                    f"AS {prefix}_{suffix}_min_24h"
                ),
                (
                    "AVG(CASE WHEN "
                    f"{predicate} THEN {value_column} END) "
                    f"AS {prefix}_{suffix}_mean_24h"
                ),
                (
                    "MAX(CASE WHEN "
                    f"{predicate} THEN {value_column} END) "
                    f"AS {prefix}_{suffix}_max_24h"
                ),
            ]
        )
        if include_phase8_p0:
            columns.extend(
                [
                    (
                        f"CASE WHEN MAX(CASE WHEN {predicate} THEN 1 ELSE 0 END) = 1 "
                        f"THEN 0 ELSE 1 END AS {prefix}_{suffix}_missing_24h"
                    ),
                    (f"{first_value} AS {prefix}_{suffix}_first_24h"),
                    (f"{last_value} AS {prefix}_{suffix}_last_24h"),
                    (
                        f"({last_value}) - ({first_value}) "
                        f"AS {prefix}_{suffix}_delta_24h"
                    ),
                    (
                        f"CASE WHEN {numeric_count} >= {trend_min_events} "
                        f"THEN {slope} ELSE NULL END "
                        f"AS {prefix}_{suffix}_slope_24h"
                    ),
                    (
                        "CASE WHEN "
                        f"MAX(CASE WHEN {numeric_predicate} THEN {event_time_column} END) "
                        "IS NOT NULL THEN "
                        f"{prediction_offset_hours}.0 - "
                        f"MAX(CASE WHEN {numeric_predicate} THEN {event_time_column} END) "
                        f"END AS {prefix}_{suffix}_hours_since_last_24h"
                    ),
                ]
            )
    return ",\n        ".join(columns)


def condition_token_sql(*, table_alias: str) -> str:
    """Return normalized/project condition token precedence SQL."""

    return f"""
COALESCE(
    NULLIF(TRIM(CAST({table_alias}.normalized_condition_token AS VARCHAR)), ''),
    NULLIF(TRIM(CAST({table_alias}.project_condition_token AS VARCHAR)), '')
)
"""


def condition_vocabulary_query(config: FeatureBuildConfig) -> str:
    """Return the train-only Phase 8 condition vocabulary query."""

    conditions = harmonized_path(config, "conditions")
    decision = config.cohort_decision_times_path
    token_sql = condition_token_sql(table_alias="c")
    limit_sql = f"LIMIT {config.condition_feature_top_n}"
    return f"""
WITH condition_tokens AS (
    SELECT
        d.stay_uid,
        {token_sql} AS condition_token
    FROM {parquet_scan(conditions)} AS c
    INNER JOIN {parquet_scan(decision)} AS d
        ON c.stay_uid = d.stay_uid
        AND c.source = d.source
    WHERE d.source = 'mimiciv'
        AND d.split = 'train'
        AND d.primary_training_eligible
)
SELECT
    condition_token
FROM condition_tokens
WHERE condition_token IS NOT NULL
GROUP BY condition_token
ORDER BY COUNT(DISTINCT stay_uid) DESC, condition_token
{limit_sql}
"""


def resolve_condition_vocabulary(
    connection: duckdb.DuckDBPyConnection,
    config: FeatureBuildConfig,
) -> tuple[str, ...]:
    """Fit the Phase 8 condition vocabulary from MIMIC train stays only."""

    if not is_phase8_p0(config) or config.condition_feature_top_n == 0:
        return ()
    rows = fetch_dict_rows(connection, condition_vocabulary_query(config))
    return tuple(str(row["condition_token"]) for row in rows)


def condition_present_columns(config: FeatureBuildConfig) -> str:
    """Return Phase 8 condition-present aggregate columns."""

    columns: list[str] = []
    for token, column_name in condition_feature_columns(
        config.condition_vocabulary
    ).items():
        columns.append(
            "MAX(CASE WHEN condition_token = "
            f"{sql_string(token)} THEN 1 ELSE 0 END) AS {column_name}"
        )
    return ",\n        ".join(columns)


def condition_cte_sql(config: FeatureBuildConfig) -> str:
    """Return optional Phase 8 condition CTEs."""

    if not is_phase8_p0(config) or not config.condition_vocabulary:
        return ""
    conditions = harmonized_path(config, "conditions")
    token_sql = condition_token_sql(table_alias="c")
    columns = condition_present_columns(config)
    return f""",
condition_events AS (
    SELECT
        c.stay_uid,
        {token_sql} AS condition_token
    FROM {parquet_scan(conditions)} AS c
    INNER JOIN decision AS d
        ON c.stay_uid = d.stay_uid
        AND c.source = d.source
),
condition_pre AS (
    SELECT *
    FROM condition_events
    WHERE condition_token IS NOT NULL
),
condition_agg AS (
    SELECT
        stay_uid,
        {columns}
    FROM condition_pre
    GROUP BY stay_uid
)
"""


def condition_select_columns(config: FeatureBuildConfig) -> str:
    """Return optional Phase 8 condition columns for the final feature SELECT."""

    if not is_phase8_p0(config) or not config.condition_vocabulary:
        return ""
    columns = [
        f"COALESCE(condition_agg.{column_name}, 0) AS {column_name}"
        for column_name in condition_feature_columns(
            config.condition_vocabulary
        ).values()
    ]
    return ",\n    " + ",\n    ".join(columns)


def condition_join_sql(config: FeatureBuildConfig) -> str:
    """Return optional Phase 8 condition aggregate join."""

    if not is_phase8_p0(config) or not config.condition_vocabulary:
        return ""
    return """
LEFT JOIN condition_agg
    ON d.stay_uid = condition_agg.stay_uid
"""


def coalesced_missingness_selects(*, tokens: Sequence[str], prefix: str) -> str:
    """Return final SELECT expressions for Phase 8 missingness indicators."""

    columns = [
        f"COALESCE({prefix}_agg.{column}, 1) AS {column}"
        for column in missingness_column_names(tokens=tokens, prefix=prefix)
    ]
    return ",\n    " + ",\n    ".join(columns) if columns else ""


def exclude_columns_sql(columns: Sequence[str]) -> str:
    """Return a DuckDB EXCLUDE list."""

    return ", ".join(columns)


def missing_input_tables(config: FeatureBuildConfig) -> list[dict[str, str]]:
    """Return missing required harmonized inputs."""

    missing: list[dict[str, str]] = []
    for table_name in required_harmonized_tables(config):
        path = harmonized_path(config, table_name)
        if not path.exists():
            missing.append({"table_name": table_name, "path": str(path)})
    return missing


def decision_times_query(config: FeatureBuildConfig, *, generated_at: str) -> str:
    """Build the cohort decision-time and eligibility query."""

    cohort = harmonized_path(config, "cohort_stays")
    split_case = split_case_sql(seed=config.split_seed)
    end_hours = stay_end_hours_sql()
    prediction = config.prediction_offset_hours
    label_end = config.label_window_end_hours
    return f"""
WITH normalized AS (
    SELECT
        source,
        source_version,
        patient_uid,
        encounter_uid,
        stay_uid,
        source_patient_id,
        source_encounter_id,
        source_stay_id,
        TRY_CAST(stay_start_time AS TIMESTAMP) AS stay_start_timestamp,
        TRY_CAST(stay_end_time AS TIMESTAMP) AS stay_end_timestamp,
        TRY_CAST(stay_start_offset_minutes AS DOUBLE) AS stay_start_offset_minutes,
        TRY_CAST(stay_end_offset_minutes AS DOUBLE) AS stay_end_offset_minutes,
        {end_hours} AS stay_end_hours_from_admit,
        TRY_CAST(los_hours AS DOUBLE) AS los_hours,
        cohort_version,
        harmonization_version
    FROM {parquet_scan(cohort)}
),
eligible AS (
    SELECT
        *,
        CASE
            WHEN stay_end_hours_from_admit IS NULL THEN 'missing_observation_end'
            WHEN stay_end_hours_from_admit < {prediction}
                THEN 'censored_before_prediction'
            WHEN stay_end_hours_from_admit < {label_end}
                THEN 'censored_before_label_window'
            ELSE 'eligible_primary'
        END AS eligibility_status
    FROM normalized
)
SELECT
    source,
    source_version,
    patient_uid,
    encounter_uid,
    stay_uid,
    source_patient_id,
    source_encounter_id,
    source_stay_id,
    {split_case} AS split,
    0.0 AS t0_hours_from_admit,
    {prediction}.0 AS prediction_time_hours_from_admit,
    {label_end}.0 AS label_window_end_hours_from_admit,
    stay_start_timestamp AS t0_timestamp,
    CASE
        WHEN stay_start_timestamp IS NOT NULL
            THEN stay_start_timestamp + INTERVAL '{prediction} hours'
        ELSE NULL
    END AS prediction_timestamp,
    CASE
        WHEN stay_start_timestamp IS NOT NULL
            THEN stay_start_timestamp + INTERVAL '{label_end} hours'
        ELSE NULL
    END AS label_window_end_timestamp,
    stay_start_timestamp,
    stay_end_timestamp,
    stay_start_offset_minutes,
    stay_end_offset_minutes,
    stay_end_hours_from_admit,
    los_hours,
    eligibility_status,
    eligibility_status = 'eligible_primary' AS primary_training_eligible,
    {sql_string(config.cohort_version)} AS cohort_version,
    {sql_string(config.harmonization_version)} AS harmonization_version,
    {sql_string(config.feature_version)} AS feature_version,
    {sql_string(config.split_version)} AS split_version,
    {sql_string(generated_at)} AS generated_at
FROM eligible
"""


def stay_hash_batch_filter(
    *,
    batch_index: int,
    batch_count: int,
    column: str = "stay_uid",
) -> str:
    """Return a deterministic stay-hash filter for one batch."""

    return (
        f"HASH(COALESCE(CAST({column} AS VARCHAR), '')) % {batch_count} = {batch_index}"
    )


def patient_stay_features_query(
    config: FeatureBuildConfig,
    *,
    decision_filter_sql: str = "TRUE",
) -> str:
    """Build the patient-stay feature query."""

    decision = config.cohort_decision_times_path
    demographics = harmonized_path(config, "demographics")
    labs = harmonized_path(config, "labs")
    vitals = harmonized_path(config, "vitals")
    allergies = harmonized_path(config, "allergies")
    interventions = harmonized_path(config, "interventions")
    prediction = config.prediction_offset_hours
    phase8_enabled = is_phase8_p0(config)

    lab_event_hours = event_hours_sql(
        source_expr="l.source",
        event_time_expr="l.event_time",
        event_offset_expr="l.event_time_offset",
        stay_start_expr="d.stay_start_timestamp",
    )
    vital_event_hours = event_hours_sql(
        source_expr="v.source",
        event_time_expr="v.event_time",
        event_offset_expr="v.event_time_offset",
        stay_start_expr="d.stay_start_timestamp",
    )
    allergy_event_hours = event_hours_sql(
        source_expr="a.source",
        event_time_expr="a.event_time",
        event_offset_expr="a.event_entered_time",
        stay_start_expr="d.stay_start_timestamp",
    )
    intervention_event_hours = event_hours_sql(
        source_expr="i.source",
        event_time_expr="i.event_start_time",
        event_offset_expr="i.event_start_offset",
        stay_start_expr="d.stay_start_timestamp",
    )
    lab_token_columns = _token_summary_columns(
        tokens=config.core_lab_tokens,
        token_column="normalized_lab_token",
        value_column="lab_value_numeric",
        prefix="lab",
        include_phase8_p0=phase8_enabled,
        prediction_offset_hours=prediction,
        trend_min_events=config.trend_min_events,
    )
    vital_token_columns = _token_summary_columns(
        tokens=config.core_vital_tokens,
        token_column="normalized_vital_token",
        value_column="value_numeric",
        prefix="vital",
        include_phase8_p0=phase8_enabled,
        prediction_offset_hours=prediction,
        trend_min_events=config.trend_min_events,
    )
    lab_excludes = [
        "stay_uid",
        "lab_event_count_24h",
        "lab_concept_count_24h",
        "lab_numeric_count_24h",
        "lab_missing_numeric_count_24h",
        "lab_abnormal_count_24h",
    ]
    vital_excludes = [
        "stay_uid",
        "vital_event_count_24h",
        "vital_concept_count_24h",
        "vital_numeric_count_24h",
        "vital_missing_numeric_count_24h",
    ]
    lab_missingness_select = ""
    vital_missingness_select = ""
    if phase8_enabled:
        lab_excludes.extend(
            missingness_column_names(tokens=config.core_lab_tokens, prefix="lab")
        )
        vital_excludes.extend(
            missingness_column_names(tokens=config.core_vital_tokens, prefix="vital")
        )
        lab_missingness_select = coalesced_missingness_selects(
            tokens=config.core_lab_tokens,
            prefix="lab",
        )
        vital_missingness_select = coalesced_missingness_selects(
            tokens=config.core_vital_tokens,
            prefix="vital",
        )

    return f"""
WITH decision AS (
    SELECT *
    FROM {parquet_scan(decision)}
    WHERE {decision_filter_sql}
)
{condition_cte_sql(config)},
lab_events AS (
    SELECT
        l.stay_uid,
        l.normalized_lab_token,
        l.lab_value_numeric,
        l.abnormal_flag,
        {lab_event_hours} AS event_time_hours_from_admit
    FROM {parquet_scan(labs)} AS l
    INNER JOIN decision AS d
        ON l.stay_uid = d.stay_uid
        AND l.source = d.source
),
lab_pre AS (
    SELECT *
    FROM lab_events
    WHERE event_time_hours_from_admit >= 0
        AND event_time_hours_from_admit <= {prediction}
),
lab_agg AS (
    SELECT
        stay_uid,
        COUNT(*) AS lab_event_count_24h,
        COUNT(DISTINCT normalized_lab_token) AS lab_concept_count_24h,
        SUM(CASE WHEN lab_value_numeric IS NOT NULL THEN 1 ELSE 0 END)
            AS lab_numeric_count_24h,
        SUM(CASE WHEN lab_value_numeric IS NULL THEN 1 ELSE 0 END)
            AS lab_missing_numeric_count_24h,
        SUM(CASE WHEN NULLIF(TRIM(CAST(abnormal_flag AS VARCHAR)), '') IS NOT NULL
            THEN 1 ELSE 0 END) AS lab_abnormal_count_24h,
        {lab_token_columns}
    FROM lab_pre
    GROUP BY stay_uid
),
vital_events AS (
    SELECT
        v.stay_uid,
        v.normalized_vital_token,
        v.value_numeric,
        {vital_event_hours} AS event_time_hours_from_admit
    FROM {parquet_scan(vitals)} AS v
    INNER JOIN decision AS d
        ON v.stay_uid = d.stay_uid
        AND v.source = d.source
),
vital_pre AS (
    SELECT *
    FROM vital_events
    WHERE event_time_hours_from_admit >= 0
        AND event_time_hours_from_admit <= {prediction}
),
vital_agg AS (
    SELECT
        stay_uid,
        COUNT(*) AS vital_event_count_24h,
        COUNT(DISTINCT normalized_vital_token) AS vital_concept_count_24h,
        SUM(CASE WHEN value_numeric IS NOT NULL THEN 1 ELSE 0 END)
            AS vital_numeric_count_24h,
        SUM(CASE WHEN value_numeric IS NULL THEN 1 ELSE 0 END)
            AS vital_missing_numeric_count_24h,
        {vital_token_columns}
    FROM vital_pre
    GROUP BY stay_uid
),
allergy_events AS (
    SELECT
        a.stay_uid,
        a.normalized_allergen_token,
        {allergy_event_hours} AS event_time_hours_from_admit
    FROM {parquet_scan(allergies)} AS a
    INNER JOIN decision AS d
        ON a.stay_uid = d.stay_uid
        AND a.source = d.source
),
allergy_agg AS (
    SELECT
        stay_uid,
        COUNT(*) AS allergy_event_count_24h,
        COUNT(DISTINCT normalized_allergen_token) AS allergy_concept_count_24h,
        COUNT(*) > 0 AS allergy_constraint_present_24h
    FROM allergy_events
    WHERE event_time_hours_from_admit >= 0
        AND event_time_hours_from_admit <= {prediction}
    GROUP BY stay_uid
),
intervention_events AS (
    SELECT
        i.stay_uid,
        i.normalized_intervention_token,
        {intervention_event_hours} AS event_time_hours_from_admit
    FROM {parquet_scan(interventions)} AS i
    INNER JOIN decision AS d
        ON i.stay_uid = d.stay_uid
        AND i.source = d.source
),
intervention_agg AS (
    SELECT
        stay_uid,
        COUNT(*) AS predecision_intervention_count_24h,
        COUNT(DISTINCT normalized_intervention_token)
            AS predecision_intervention_concept_count_24h
    FROM intervention_events
    WHERE event_time_hours_from_admit >= 0
        AND event_time_hours_from_admit <= {prediction}
    GROUP BY stay_uid
)
SELECT
    d.source,
    d.source_version,
    d.patient_uid,
    d.encounter_uid,
    d.stay_uid,
    d.split,
    d.eligibility_status,
    d.primary_training_eligible,
    d.t0_hours_from_admit,
    d.prediction_time_hours_from_admit,
    d.label_window_end_hours_from_admit,
    d.stay_end_hours_from_admit,
    demo.age_years,
    demo.age_topcoded,
    demo.sex,
    demo.race_or_ethnicity,
    demo.hospital_id,
    demo.ward_id,
    demo.admission_type,
    demo.admission_source,
    demo.unit_type,
    demo.last_unit_type,
    demo.stay_type,
    demo.stay_sequence{condition_select_columns(config)},
    COALESCE(lab_agg.lab_event_count_24h, 0) AS lab_event_count_24h,
    COALESCE(lab_agg.lab_concept_count_24h, 0) AS lab_concept_count_24h,
    COALESCE(lab_agg.lab_numeric_count_24h, 0) AS lab_numeric_count_24h,
    COALESCE(lab_agg.lab_missing_numeric_count_24h, 0)
        AS lab_missing_numeric_count_24h,
    COALESCE(lab_agg.lab_abnormal_count_24h, 0) AS lab_abnormal_count_24h,
    lab_agg.* EXCLUDE ({exclude_columns_sql(lab_excludes)}){lab_missingness_select},
    COALESCE(vital_agg.vital_event_count_24h, 0) AS vital_event_count_24h,
    COALESCE(vital_agg.vital_concept_count_24h, 0) AS vital_concept_count_24h,
    COALESCE(vital_agg.vital_numeric_count_24h, 0) AS vital_numeric_count_24h,
    COALESCE(vital_agg.vital_missing_numeric_count_24h, 0)
        AS vital_missing_numeric_count_24h,
    vital_agg.* EXCLUDE ({exclude_columns_sql(vital_excludes)})
        {vital_missingness_select},
    COALESCE(allergy_agg.allergy_event_count_24h, 0) AS allergy_event_count_24h,
    COALESCE(allergy_agg.allergy_concept_count_24h, 0)
        AS allergy_concept_count_24h,
    COALESCE(allergy_agg.allergy_constraint_present_24h, FALSE)
        AS allergy_constraint_present_24h,
    COALESCE(intervention_agg.predecision_intervention_count_24h, 0)
        AS predecision_intervention_count_24h,
    COALESCE(intervention_agg.predecision_intervention_concept_count_24h, 0)
        AS predecision_intervention_concept_count_24h,
    d.cohort_version,
    d.harmonization_version,
    d.feature_version,
    d.split_version,
    d.generated_at
FROM decision AS d
LEFT JOIN {parquet_scan(demographics)} AS demo
    ON d.stay_uid = demo.stay_uid
    AND d.source = demo.source
LEFT JOIN lab_agg
    ON d.stay_uid = lab_agg.stay_uid
LEFT JOIN vital_agg
    ON d.stay_uid = vital_agg.stay_uid
LEFT JOIN allergy_agg
    ON d.stay_uid = allergy_agg.stay_uid
LEFT JOIN intervention_agg
    ON d.stay_uid = intervention_agg.stay_uid
{condition_join_sql(config)}
"""


def patient_stay_features_finalize_query(*, part_paths: Sequence[Path]) -> str:
    """Combine patient-stay feature parts into the public single-file artifact."""

    return f"SELECT * FROM {parquet_scan_paths(part_paths)}"


def event_sequence_staging_query(config: FeatureBuildConfig) -> str:
    """Build reduced pre-decision events before sequence numbering."""

    decision = config.cohort_decision_times_path
    temporal = harmonized_path(config, "temporal_events")
    prediction = config.prediction_offset_hours
    event_hours = event_hours_sql(
        source_expr="e.source",
        event_time_expr="e.event_start_time",
        event_offset_expr="e.event_start_offset",
        stay_start_expr="d.stay_start_timestamp",
    )
    medication_filter = (
        "TRUE"
        if config.include_predecision_medications
        else "e.event_type <> 'medication'"
    )
    return f"""
WITH decision AS (
    SELECT *
    FROM {parquet_scan(decision)}
),
events AS (
    SELECT
        e.source,
        e.source_version,
        e.patient_uid,
        e.encounter_uid,
        e.stay_uid,
        e.event_type,
        e.source_domain,
        e.source_table,
        e.source_event_id,
        e.event_token,
        e.source_code,
        e.source_text,
        e.value_numeric,
        e.value_text,
        e.unit,
        e.normalized_unit,
        e.mapping_status,
        e.cohort_version,
        e.extraction_version,
        e.mapping_version,
        e.harmonization_version,
        d.split,
        {event_hours} AS event_time_hours_from_admit
    FROM {parquet_scan(temporal)} AS e
    INNER JOIN decision AS d
        ON e.stay_uid = d.stay_uid
        AND e.source = d.source
    WHERE {medication_filter}
),
predecision AS (
    SELECT *
    FROM events
    WHERE event_time_hours_from_admit >= 0
        AND event_time_hours_from_admit <= {prediction}
)
SELECT
    source,
    source_version,
    patient_uid,
    encounter_uid,
    stay_uid,
    split,
    event_type,
    source_domain,
    source_table,
    source_event_id,
    event_time_hours_from_admit,
    event_token,
    source_code,
    source_text,
    value_numeric,
    value_text,
    unit,
    normalized_unit,
    mapping_status,
    cohort_version,
    extraction_version,
    mapping_version,
    harmonization_version
FROM predecision
"""


def event_sequence_batch_query(
    staging_path: Path,
    *,
    batch_index: int,
    batch_count: int,
) -> str:
    """Build sequence-numbered events for one stay-hash batch."""

    return f"""
WITH staged AS (
    SELECT *
    FROM {parquet_scan(staging_path)}
    WHERE HASH(COALESCE(CAST(stay_uid AS VARCHAR), '')) % {batch_count}
        = {batch_index}
)
SELECT
    source,
    source_version,
    patient_uid,
    encounter_uid,
    stay_uid,
    split,
    ROW_NUMBER() OVER (
        PARTITION BY stay_uid
        ORDER BY event_time_hours_from_admit, event_type, source_event_id
    ) AS event_sequence_position,
    event_type,
    source_domain,
    source_table,
    source_event_id,
    event_time_hours_from_admit,
    event_token,
    source_code,
    source_text,
    value_numeric,
    value_text,
    unit,
    normalized_unit,
    mapping_status,
    cohort_version,
    extraction_version,
    mapping_version,
    harmonization_version
FROM staged
"""


def event_sequence_finalize_query(
    config: FeatureBuildConfig,
    *,
    part_paths: Sequence[Path],
) -> str:
    """Combine sequence-numbered parts into the public single-file artifact."""

    return f"""
SELECT
    source,
    source_version,
    patient_uid,
    encounter_uid,
    stay_uid,
    split,
    event_sequence_position,
    event_type,
    source_domain,
    source_table,
    source_event_id,
    event_time_hours_from_admit,
    event_token,
    source_code,
    source_text,
    value_numeric,
    value_text,
    unit,
    normalized_unit,
    mapping_status,
    cohort_version,
    extraction_version,
    mapping_version,
    harmonization_version,
    {sql_string(config.feature_version)} AS feature_version
FROM {parquet_scan_paths(part_paths)}
"""


def temporal_event_exclusion_query(config: FeatureBuildConfig) -> str:
    """Aggregate temporal events excluded from pre-decision sequences."""

    decision = config.cohort_decision_times_path
    temporal = harmonized_path(config, "temporal_events")
    prediction = config.prediction_offset_hours
    event_hours = event_hours_sql(
        source_expr="e.source",
        event_time_expr="e.event_start_time",
        event_offset_expr="e.event_start_offset",
        stay_start_expr="d.stay_start_timestamp",
    )
    medication_filter = (
        "FALSE"
        if config.include_predecision_medications
        else "e.event_type = 'medication'"
    )
    return f"""
WITH events AS (
    SELECT
        e.source,
        e.event_type,
        {event_hours} AS event_time_hours_from_admit,
        {medication_filter} AS excluded_by_default_medication_rule
    FROM {parquet_scan(temporal)} AS e
    INNER JOIN {parquet_scan(decision)} AS d
        ON e.stay_uid = d.stay_uid
        AND e.source = d.source
)
SELECT
    source,
    event_type,
    SUM(CASE WHEN event_time_hours_from_admit IS NULL THEN 1 ELSE 0 END)
        AS missing_time_rows,
    SUM(CASE WHEN event_time_hours_from_admit > {prediction} THEN 1 ELSE 0 END)
        AS after_prediction_rows,
    SUM(CASE WHEN event_time_hours_from_admit < 0 THEN 1 ELSE 0 END)
        AS before_admission_rows,
    SUM(CASE WHEN excluded_by_default_medication_rule THEN 1 ELSE 0 END)
        AS default_medication_exclusion_rows
FROM events
GROUP BY source, event_type
ORDER BY source, event_type
"""


def base_manifest(
    config: FeatureBuildConfig,
    *,
    status: str,
    generated_at: str,
) -> dict[str, Any]:
    """Build the common feature manifest shell."""

    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "generated_at": generated_at,
        "feature_set": config.feature_set,
        "feature_version": config.feature_version,
        "data_safety": {
            "manifest_contains_patient_rows": False,
            "local_artifacts_contain_patient_level_rows": True,
            "artifact_storage": "ignored Dataset/processed/features",
        },
        "parameters": {
            "feature_set": config.feature_set,
            "prediction_offset_hours": config.prediction_offset_hours,
            "label_window_hours": config.label_window_hours,
            "split_seed": config.split_seed,
            "condition_feature_top_n": config.condition_feature_top_n,
            "trend_min_events": config.trend_min_events,
            "include_predecision_medications": config.include_predecision_medications,
            "stay_feature_batches": stay_feature_batch_count(config),
            "event_sequence_batches": event_sequence_batch_count(config),
            "core_lab_tokens": list(config.core_lab_tokens),
            "core_vital_tokens": list(config.core_vital_tokens),
        },
        "versions": {
            "cohort_version": config.cohort_version,
            "harmonization_version": config.harmonization_version,
            "feature_version": config.feature_version,
            "split_version": config.split_version,
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
    """Materialize an artifact and return an aggregate record."""

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


def patient_stay_features_artifact_record(
    connection: duckdb.DuckDBPyConnection,
    config: FeatureBuildConfig,
) -> dict[str, Any]:
    """Materialize patient-stay features with bounded-memory stay-hash batches."""

    output_path = config.patient_stay_features_path
    batch_count = stay_feature_batch_count(config)
    if batch_count == 1:
        record = artifact_record(
            connection,
            table_name="patient_stay_features",
            query=patient_stay_features_query(config),
            output_path=output_path,
        )
        if record["status"] == "completed":
            record["build_strategy"] = "single_query"
            record["batch_count"] = batch_count
        return record

    part_counts: list[int] = []
    part_paths = [
        stay_feature_part_path(config, batch_index)
        for batch_index in range(batch_count)
    ]
    try:
        for part_path in part_paths:
            if part_path.exists():
                part_path.unlink()
        for batch_index, part_path in enumerate(part_paths):
            part_counts.append(
                copy_query_to_parquet(
                    connection,
                    patient_stay_features_query(
                        config,
                        decision_filter_sql=stay_hash_batch_filter(
                            batch_index=batch_index,
                            batch_count=batch_count,
                        ),
                    ),
                    part_path,
                )
            )
        row_count = copy_query_to_parquet(
            connection,
            patient_stay_features_finalize_query(part_paths=part_paths),
            output_path,
        )
    except Exception as error:
        return {
            "table_name": "patient_stay_features",
            "output_path": str(output_path),
            "status": "failed",
            "row_count": None,
            "reason": safe_error_message(error),
            "build_strategy": "stay_hash_batches",
            "batch_count": batch_count,
            "stale_output_exists": output_path.exists(),
        }

    return {
        "table_name": "patient_stay_features",
        "output_path": str(output_path),
        "status": "completed",
        "row_count": row_count,
        "build_strategy": "stay_hash_batches",
        "batch_count": batch_count,
        "batch_row_counts": part_counts,
    }


def event_sequence_artifact_record(
    connection: duckdb.DuckDBPyConnection,
    config: FeatureBuildConfig,
) -> dict[str, Any]:
    """Materialize event sequences with bounded-memory stay-hash batches."""

    output_path = config.event_sequences_path
    staging_path = event_sequence_staging_path(config)
    batch_count = event_sequence_batch_count(config)
    part_counts: list[int] = []
    part_paths = [
        event_sequence_part_path(config, batch_index)
        for batch_index in range(batch_count)
    ]
    try:
        staged_row_count = copy_query_to_parquet(
            connection,
            event_sequence_staging_query(config),
            staging_path,
        )
        for batch_index, part_path in enumerate(part_paths):
            part_counts.append(
                copy_query_to_parquet(
                    connection,
                    event_sequence_batch_query(
                        staging_path=staging_path,
                        batch_index=batch_index,
                        batch_count=batch_count,
                    ),
                    part_path,
                )
            )
        row_count = copy_query_to_parquet(
            connection,
            event_sequence_finalize_query(config, part_paths=part_paths),
            output_path,
        )
    except Exception as error:
        return {
            "table_name": "event_sequences",
            "output_path": str(output_path),
            "status": "failed",
            "row_count": None,
            "reason": safe_error_message(error),
            "build_strategy": "staged_hash_batches",
            "batch_count": batch_count,
        }

    return {
        "table_name": "event_sequences",
        "output_path": str(output_path),
        "status": "completed",
        "row_count": row_count,
        "build_strategy": "staged_hash_batches",
        "batch_count": batch_count,
        "staged_row_count": staged_row_count,
        "batch_row_counts": part_counts,
    }


def condition_oov_counts_query(config: FeatureBuildConfig) -> str:
    """Return aggregate Phase 8 condition OOV counts without token values."""

    conditions = harmonized_path(config, "conditions")
    decision = config.cohort_decision_times_path
    token_sql = condition_token_sql(table_alias="c")
    if config.condition_vocabulary:
        vocab = ", ".join(sql_string(token) for token in config.condition_vocabulary)
        in_vocab = f"condition_token IN ({vocab})"
    else:
        in_vocab = "FALSE"
    return f"""
WITH condition_tokens AS (
    SELECT
        d.source,
        d.split,
        d.stay_uid,
        {token_sql} AS condition_token
    FROM {parquet_scan(conditions)} AS c
    INNER JOIN {parquet_scan(decision)} AS d
        ON c.stay_uid = d.stay_uid
        AND c.source = d.source
),
classified AS (
    SELECT
        source,
        split,
        stay_uid,
        condition_token,
        ({in_vocab}) AS in_train_vocabulary
    FROM condition_tokens
    WHERE condition_token IS NOT NULL
)
SELECT
    source,
    split,
    COUNT(*) AS condition_token_row_count,
    COUNT(DISTINCT condition_token) AS distinct_condition_token_count,
    SUM(CASE WHEN NOT in_train_vocabulary THEN 1 ELSE 0 END)
        AS oov_condition_token_row_count,
    COUNT(DISTINCT CASE WHEN NOT in_train_vocabulary THEN condition_token END)
        AS distinct_oov_condition_token_count,
    COUNT(DISTINCT CASE WHEN NOT in_train_vocabulary THEN stay_uid END)
        AS stays_with_oov_condition_token_count
FROM classified
GROUP BY source, split
ORDER BY source, split
"""


def feature_column_counts_by_family(
    connection: duckdb.DuckDBPyConnection,
    config: FeatureBuildConfig,
) -> dict[str, int]:
    """Return aggregate feature-column counts grouped by coarse family."""

    describe = connection.execute(
        f"DESCRIBE SELECT * FROM {parquet_scan(config.patient_stay_features_path)}"
    ).fetchall()
    columns = [str(row[0]) for row in describe]
    condition_columns = set(
        condition_feature_columns(config.condition_vocabulary).values()
    )
    trend_columns = {
        *trend_column_names(tokens=config.core_lab_tokens, prefix="lab"),
        *trend_column_names(tokens=config.core_vital_tokens, prefix="vital"),
    }
    missing_columns = {
        *missingness_column_names(tokens=config.core_lab_tokens, prefix="lab"),
        *missingness_column_names(tokens=config.core_vital_tokens, prefix="vital"),
    }
    return {
        "total_columns": len(columns),
        "condition_columns": sum(column in condition_columns for column in columns),
        "trend_columns": sum(column in trend_columns for column in columns),
        "missingness_columns": sum(column in missing_columns for column in columns),
        "lab_columns": sum(column.startswith("lab_") for column in columns),
        "vital_columns": sum(column.startswith("vital_") for column in columns),
        "demographic_context_columns": sum(
            column
            in {
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
            }
            for column in columns
        ),
        "allergy_intervention_columns": sum(
            column.startswith("allergy_")
            or column.startswith("predecision_intervention_")
            for column in columns
        ),
    }


def append_phase8_manifest_summaries(
    connection: duckdb.DuckDBPyConnection,
    config: FeatureBuildConfig,
    manifest: dict[str, Any],
) -> None:
    """Attach aggregate-only Phase 8 feature-family summaries."""

    condition_columns = list(
        condition_feature_columns(config.condition_vocabulary).values()
    )
    lab_trend_columns = list(
        trend_column_names(tokens=config.core_lab_tokens, prefix="lab")
    )
    vital_trend_columns = list(
        trend_column_names(tokens=config.core_vital_tokens, prefix="vital")
    )
    lab_missing_columns = list(
        missingness_column_names(tokens=config.core_lab_tokens, prefix="lab")
    )
    vital_missing_columns = list(
        missingness_column_names(tokens=config.core_vital_tokens, prefix="vital")
    )
    manifest["feature_set"] = config.feature_set
    manifest["feature_version"] = config.feature_version
    manifest["condition_token_precedence"] = [
        "normalized_condition_token",
        "project_condition_token",
    ]
    manifest["condition_vocabulary_fit_scope"] = {
        "source": "mimiciv",
        "split": "train",
        "primary_training_eligible": True,
        "ranking": "distinct stay frequency desc, token asc",
    }
    manifest["condition_vocabulary_size"] = len(config.condition_vocabulary)
    manifest["condition_columns_added"] = condition_columns
    manifest["trend_columns_added"] = [*lab_trend_columns, *vital_trend_columns]
    manifest["missingness_columns_added"] = [
        *lab_missing_columns,
        *vital_missing_columns,
    ]
    manifest["feature_column_counts_by_family"] = feature_column_counts_by_family(
        connection,
        config,
    )
    manifest["condition_oov_counts"] = fetch_dict_rows(
        connection,
        condition_oov_counts_query(config),
    )


def append_manifest_summaries(
    connection: duckdb.DuckDBPyConnection,
    config: FeatureBuildConfig,
    manifest: dict[str, Any],
) -> None:
    """Attach aggregate-only summaries to a completed manifest."""

    decision = config.cohort_decision_times_path
    patient_features = config.patient_stay_features_path
    event_sequences = config.event_sequences_path
    manifest["eligibility_counts"] = fetch_dict_rows(
        connection,
        f"""
SELECT source, eligibility_status, COUNT(*) AS stay_count
FROM {parquet_scan(decision)}
GROUP BY source, eligibility_status
ORDER BY source, eligibility_status
""",
    )
    manifest["split_counts"] = fetch_dict_rows(
        connection,
        f"""
SELECT source, split, COUNT(DISTINCT patient_uid) AS patient_count,
    COUNT(*) AS stay_count
FROM {parquet_scan(decision)}
GROUP BY source, split
ORDER BY source, split
""",
    )
    manifest["feature_rows_by_source"] = fetch_dict_rows(
        connection,
        f"""
SELECT source, split, COUNT(*) AS row_count
FROM {parquet_scan(patient_features)}
GROUP BY source, split
ORDER BY source, split
""",
    )
    manifest["event_sequence_counts"] = fetch_dict_rows(
        connection,
        f"""
SELECT source, event_type, COUNT(*) AS row_count
FROM {parquet_scan(event_sequences)}
GROUP BY source, event_type
ORDER BY source, event_type
""",
    )
    manifest["temporal_event_exclusions"] = fetch_dict_rows(
        connection,
        temporal_event_exclusion_query(config),
    )
    if is_phase8_p0(config):
        append_phase8_manifest_summaries(connection, config, manifest)


def build_feature_artifacts(
    config: FeatureBuildConfig = FeatureBuildConfig(),
) -> dict[str, Any]:
    """Build local Milestone 6 feature artifacts and aggregate manifest."""

    generated_at = datetime.now(UTC).isoformat()
    config.features_root.mkdir(parents=True, exist_ok=True)
    missing = missing_input_tables(config)
    if missing:
        manifest = base_manifest(
            config,
            status="failed_missing_harmonized_inputs",
            generated_at=generated_at,
        )
        manifest["missing_inputs"] = missing
        write_json(config.manifest_path, manifest)
        return manifest

    tables: list[dict[str, Any]] = []
    artifacts: dict[str, str] = {}
    with duckdb.connect(database=":memory:") as connection:
        configure_connection(config, connection)
        decision_record = artifact_record(
            connection,
            table_name="cohort_decision_times",
            query=decision_times_query(config, generated_at=generated_at),
            output_path=config.cohort_decision_times_path,
        )
        tables.append(decision_record)
        if decision_record["status"] == "completed":
            artifacts["cohort_decision_times"] = str(config.cohort_decision_times_path)
            if is_phase8_p0(config):
                config = replace(
                    config,
                    condition_vocabulary=resolve_condition_vocabulary(
                        connection,
                        config,
                    ),
                )

        manifest = base_manifest(
            config,
            status="completed",
            generated_at=generated_at,
        )
        manifest["artifacts"].update(artifacts)
        if is_phase8_p0(config):
            manifest["condition_vocabulary_fit_status"] = (
                "completed" if decision_record["status"] == "completed" else "skipped"
            )

        record = patient_stay_features_artifact_record(connection, config)
        tables.append(record)
        if record["status"] == "completed":
            manifest["artifacts"]["patient_stay_features"] = str(
                config.patient_stay_features_path
            )

        record = event_sequence_artifact_record(connection, config)
        tables.append(record)
        if record["status"] == "completed":
            manifest["artifacts"]["event_sequences"] = str(config.event_sequences_path)

        if any(table["status"] == "failed" for table in tables):
            manifest["status"] = "failed"
        else:
            append_manifest_summaries(connection, config, manifest)

    manifest["tables"] = tables
    write_json(config.manifest_path, manifest)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Milestone 6 temporal feature artifacts.",
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
        help="Output directory for local ignored feature artifacts.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Output path for the aggregate feature manifest.",
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
        help="Seed used in deterministic patient split assignment.",
    )
    parser.add_argument(
        "--feature-set",
        choices=FEATURE_SETS,
        default=BASELINE_FEATURE_SET,
        help=(
            "Feature family set to build. baseline preserves temporal-features-v1; "
            "phase8_p0 adds reviewed condition, trend, and missingness columns "
            "with temporal-features-v2."
        ),
    )
    parser.add_argument(
        "--condition-feature-top-n",
        type=int,
        default=DEFAULT_CONDITION_FEATURE_TOP_N,
        help=(
            "Number of train-fit condition tokens to widen when "
            "--feature-set phase8_p0 is selected."
        ),
    )
    parser.add_argument(
        "--trend-min-events",
        type=int,
        default=DEFAULT_TREND_MIN_EVENTS,
        help=(
            "Minimum numeric events required before lab/vital trend slope is emitted."
        ),
    )
    parser.add_argument(
        "--include-predecision-medications",
        action="store_true",
        help=(
            "Include pre-decision medication events in event_sequences. "
            "Default excludes them to avoid target-proxy medication-history leakage."
        ),
    )
    parser.add_argument(
        "--stay-feature-batches",
        type=int,
        default=DEFAULT_STAY_FEATURE_BATCHES,
        help=(
            "Number of stay-hash batches used for patient_stay_features. "
            "Use more batches to lower peak memory on large lab/vital aggregates."
        ),
    )
    parser.add_argument(
        "--event-sequence-batches",
        type=int,
        default=DEFAULT_EVENT_SEQUENCE_BATCHES,
        help=(
            "Number of stay-hash batches used for event_sequences windowing. "
            "Use more batches to lower peak memory on large temporal_events."
        ),
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
    manifest = build_feature_artifacts(
        FeatureBuildConfig(
            harmonized_root=args.harmonized_root,
            features_root=args.features_root,
            manifest_path=args.manifest,
            prediction_offset_hours=args.prediction_offset_hours,
            label_window_hours=args.label_window_hours,
            split_seed=args.split_seed,
            feature_set=args.feature_set,
            condition_feature_top_n=args.condition_feature_top_n,
            trend_min_events=args.trend_min_events,
            include_predecision_medications=args.include_predecision_medications,
            stay_feature_batches=args.stay_feature_batches,
            event_sequence_batches=args.event_sequence_batches,
            duckdb_temp_directory=args.duckdb_temp_dir,
            duckdb_memory_limit=args.duckdb_memory_limit,
            duckdb_threads=args.duckdb_threads,
        )
    )
    print(
        "Wrote Milestone 6 feature manifest: "
        f"status={manifest['status']}, "
        f"tables={len(manifest.get('tables', []))}"
    )
    if manifest["status"] == "failed_missing_harmonized_inputs":
        return 2
    return 1 if manifest["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
