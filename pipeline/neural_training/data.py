"""Prepare leakage-safe, stay-grouped tensor caches for the neural branch.

All heavy work runs in DuckDB and streams to hash-sharded Parquet so peak memory
stays bounded and no patient rows enter Python. The step:

* resolves the approved (leakage-reviewed) stay feature projection from the
  training-contract rules;
* fits train-only vocabularies (``PAD=0``, ``UNK=1``, train indexes offset by
  two) and train-only numeric/event-value normalization statistics;
* materializes three flat caches per split - stay context features, truncated
  event token sequences, and per-candidate ranking rows - sharded by
  ``hash(source, stay_uid)`` so every ranking group's context, events, and
  candidates land in the same shard;
* excludes zero-positive ranking groups from the training cache (reporting them
  as coverage exclusions) while keeping every candidate of positive groups; and
* keeps all groups for the evaluation splits so scoring covers the full
  candidate universe.

This module never imports PyTorch.
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import duckdb

from pipeline.extract_utils import (
    configure_duckdb_connection,
    parquet_scan,
    safe_error_message,
    sql_string,
)
from pipeline.features import copy_query_to_parquet, fetch_dict_rows
from pipeline.io_utils import quote_identifier
from pipeline.neural_training.config import (
    FEATURE_LAYOUT_VERSION,
    PREDICTION_OFFSET_HOURS,
    PREPARE_SCHEMA_VERSION,
    PRIOR_SMOOTHING_ALPHA,
    RESERVED_TOKEN_COUNT,
    UNK_INDEX,
    NeuralTrainingConfig,
)
from pipeline.training_contract import (
    LOW_CARDINALITY_STAY_CATEGORICAL,
    approved_model_projection,
    schema_columns,
)

DEVELOPMENT_SOURCE = "mimiciv"


@dataclass(frozen=True)
class FeatureLayout:
    """Resolved, ordered model-input layout persisted alongside the caches."""

    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    max_sequence_length: int
    prediction_offset_hours: int
    feature_version: str | None

    def as_dict(self, *, vocab_sizes: dict[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": FEATURE_LAYOUT_VERSION,
            "numeric_columns": list(self.numeric_columns),
            "categorical_columns": list(self.categorical_columns),
            "max_sequence_length": self.max_sequence_length,
            "prediction_offset_hours": self.prediction_offset_hours,
            "feature_version": self.feature_version,
            "reserved_token_count": RESERVED_TOKEN_COUNT,
            "pad_index": 0,
            "unk_index": UNK_INDEX,
            "vocab_sizes": vocab_sizes,
        }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write stable aggregate JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def configure_connection(
    config: NeuralTrainingConfig,
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Apply shared memory-safe DuckDB settings."""

    configure_duckdb_connection(
        connection,
        temp_directory=config.duckdb_temp_directory,
        memory_limit=config.duckdb_memory_limit,
        threads=config.duckdb_threads,
    )


def resolve_feature_layout(
    connection: duckdb.DuckDBPyConnection,
    config: NeuralTrainingConfig,
) -> FeatureLayout:
    """Return the approved numeric and categorical stay feature layout."""

    columns = schema_columns(connection, config.patient_stay_features_path)
    approved = approved_model_projection("patient_stay_features", columns)
    categorical = tuple(
        name for name in approved if name in LOW_CARDINALITY_STAY_CATEGORICAL
    )
    numeric = tuple(name for name in approved if name not in categorical)
    # ``feature_version`` is a provenance column, not a model feature; record its
    # declared value from the schema when present for reproducibility.
    declared_version: str | None = None
    if any(name == "feature_version" for name, _ in columns):
        declared_version = _scalar(
            connection,
            f"SELECT DISTINCT feature_version "
            f"FROM {parquet_scan(config.patient_stay_features_path)} LIMIT 1",
        )
    return FeatureLayout(
        numeric_columns=numeric,
        categorical_columns=categorical,
        max_sequence_length=int(config.max_sequence_length),
        prediction_offset_hours=PREDICTION_OFFSET_HOURS,
        feature_version=declared_version,
    )


def _scalar(connection: duckdb.DuckDBPyConnection, query: str) -> Any:
    row = connection.execute(query).fetchone()
    return None if row is None else row[0]


def _train_features_scope(config: NeuralTrainingConfig) -> str:
    return (
        f"SELECT * FROM {parquet_scan(config.patient_stay_features_path)} "
        f"WHERE source = {sql_string(DEVELOPMENT_SOURCE)} AND split = 'train'"
    )


# DuckDB 1.5.x still raises ``STDDEV_SAMP is out of range`` on extreme finite
# values whose squared deviations overflow float64 (for example a pathological
# ``REGR_SLOPE``). Keep only values whose magnitude is safe to square.
NORMALIZATION_ABS_BOUND = 1e100


def _safe_mean_std(mean: Any, std: Any) -> tuple[float, float]:
    """Coerce raw AVG/STDDEV values into a finite ``(mean, std>0)`` pair."""

    mean_value = float(mean) if mean is not None else 0.0
    if not math.isfinite(mean_value):
        mean_value = 0.0
    std_value = float(std) if std is not None else 1.0
    if not math.isfinite(std_value) or std_value <= 0.0:
        std_value = 1.0
    return mean_value, std_value


def _finite_bounded_expr(raw_sql: str) -> str:
    """SQL expression that keeps finite values inside ``NORMALIZATION_ABS_BOUND``."""

    return (
        f"CASE WHEN isfinite({raw_sql}) "
        f"AND abs({raw_sql}) <= {NORMALIZATION_ABS_BOUND!r} "
        f"THEN {raw_sql} END"
    )


def _excluded_from_normalization_count_expr(raw_sql: str) -> str:
    """Count non-null values excluded as non-finite or magnitude-overflow risks."""

    return (
        f"SUM(CASE WHEN {raw_sql} IS NOT NULL AND NOT ("
        f"isfinite({raw_sql}) AND abs({raw_sql}) <= {NORMALIZATION_ABS_BOUND!r}"
        f") THEN 1 ELSE 0 END)"
    )


def numeric_normalization_stats(
    connection: duckdb.DuckDBPyConnection,
    config: NeuralTrainingConfig,
    numeric_columns: Sequence[str],
) -> tuple[dict[str, tuple[float, float]], dict[str, int]]:
    """Return train-only mean/std per numeric column and excluded-value counts.

    Std is floored at ``1.0``. Values that are non-finite (``inf``/``-inf``/
    ``NaN``) or whose absolute magnitude exceeds
    :data:`NORMALIZATION_ABS_BOUND` are excluded from the mean/std so DuckDB's
    ``STDDEV_SAMP`` cannot raise ``Out of Range`` on degenerate upstream
    features (for example a zero-time-variance or explosively large
    ``REGR_SLOPE`` trend column). The per-column count of excluded values is
    returned for the aggregate manifest.
    """

    if not numeric_columns:
        return {}, {}
    aggregates: list[str] = []
    for index, name in enumerate(numeric_columns):
        raw = f"CAST({quote_identifier(name)} AS DOUBLE)"
        usable = _finite_bounded_expr(raw)
        aggregates.append(f"AVG({usable}) AS m{index}")
        aggregates.append(f"STDDEV_SAMP({usable}) AS s{index}")
        aggregates.append(f"{_excluded_from_normalization_count_expr(raw)} AS n{index}")
    query = (
        f"SELECT {', '.join(aggregates)} "
        f"FROM ({_train_features_scope(config)}) AS train_features"
    )
    row = connection.execute(query).fetchone()
    values = list(row) if row is not None else []
    stats: dict[str, tuple[float, float]] = {}
    excluded_counts: dict[str, int] = {}
    for index, name in enumerate(numeric_columns):
        base = 3 * index
        mean = values[base] if base < len(values) else None
        std = values[base + 1] if base + 1 < len(values) else None
        count = values[base + 2] if base + 2 < len(values) else None
        stats[name] = _safe_mean_std(mean, std)
        excluded = int(count) if count is not None else 0
        if excluded:
            excluded_counts[name] = excluded
    return stats, excluded_counts


def event_value_stats(
    connection: duckdb.DuckDBPyConnection,
    config: NeuralTrainingConfig,
) -> tuple[float, float]:
    """Return train-only mean/std for finite, magnitude-bounded event values."""

    query = f"""
SELECT
    AVG(value_usable) AS mean_value,
    STDDEV_SAMP(value_usable) AS std_value
FROM (
    SELECT {_finite_bounded_expr("CAST(value_numeric AS DOUBLE)")} AS value_usable
    FROM {parquet_scan(config.event_sequences_path)}
    WHERE source = {sql_string(DEVELOPMENT_SOURCE)}
        AND split = 'train'
        AND event_type <> 'medication'
        AND value_numeric IS NOT NULL
        AND event_time_hours_from_admit >= 0
        AND event_time_hours_from_admit <= {PREDICTION_OFFSET_HOURS}
) AS usable_event_values
WHERE value_usable IS NOT NULL
"""
    row = connection.execute(query).fetchone()
    mean, std = _safe_mean_std(
        row[0] if row else None,
        row[1] if row else None,
    )
    return mean, std


def write_normalization_stats(
    connection: duckdb.DuckDBPyConnection,
    config: NeuralTrainingConfig,
    *,
    numeric_stats: dict[str, tuple[float, float]],
    event_mean: float,
    event_std: float,
) -> int:
    """Persist normalization statistics as a small aggregate Parquet table."""

    rows: list[str] = []
    for name, (mean, std) in numeric_stats.items():
        rows.append(
            f"('stay_numeric', {sql_string(name)}, {mean!r}::DOUBLE, {std!r}::DOUBLE)"
        )
    rows.append(
        f"('event_value', 'value_numeric', {event_mean!r}::DOUBLE, "
        f"{event_std!r}::DOUBLE)"
    )
    values = ",\n        ".join(rows)
    query = f"""
SELECT * FROM (
    VALUES
        {values}
) AS stats(feature_kind, column_name, mean, std)
"""
    return copy_query_to_parquet(connection, query, config.normalization_path)


# ---------------------------------------------------------------------------
# Vocabulary construction (train-only, PAD=0 / UNK=1, offset by two)
# ---------------------------------------------------------------------------
def _token_vocab_query(
    *,
    source_scan: str,
    token_expr: str,
    predicate: str,
) -> str:
    return f"""
WITH tokens AS (
    SELECT DISTINCT {token_expr} AS token
    FROM {source_scan}
    WHERE {predicate}
)
SELECT
    CAST(ROW_NUMBER() OVER (ORDER BY token) - 1 + {RESERVED_TOKEN_COUNT} AS BIGINT)
        AS token_index,
    token
FROM tokens
"""


def build_event_vocabulary(
    connection: duckdb.DuckDBPyConnection,
    config: NeuralTrainingConfig,
) -> int:
    query = _token_vocab_query(
        source_scan=parquet_scan(config.event_sequences_path),
        token_expr="event_type || '|' || event_token",
        predicate=(
            f"source = {sql_string(DEVELOPMENT_SOURCE)} AND split = 'train' "
            "AND event_type <> 'medication' AND event_token IS NOT NULL "
            "AND event_time_hours_from_admit >= 0 "
            f"AND event_time_hours_from_admit <= {PREDICTION_OFFSET_HOURS}"
        ),
    )
    return copy_query_to_parquet(connection, query, config.event_vocabulary_path)


def build_condition_vocabulary(
    connection: duckdb.DuckDBPyConnection,
    config: NeuralTrainingConfig,
) -> int:
    query = _token_vocab_query(
        source_scan=parquet_scan(config.candidate_catalog_path),
        token_expr="index_condition_token",
        predicate="index_condition_token IS NOT NULL",
    )
    return copy_query_to_parquet(connection, query, config.condition_vocabulary_path)


def build_candidate_vocabulary(
    connection: duckdb.DuckDBPyConnection,
    config: NeuralTrainingConfig,
) -> int:
    query = _token_vocab_query(
        source_scan=parquet_scan(config.candidate_catalog_path),
        token_expr="candidate_medication_token",
        predicate="candidate_medication_token IS NOT NULL",
    )
    return copy_query_to_parquet(connection, query, config.candidate_vocabulary_path)


def build_categorical_vocabulary(
    connection: duckdb.DuckDBPyConnection,
    config: NeuralTrainingConfig,
    categorical_columns: Sequence[str],
) -> int:
    """Build a per-column categorical vocabulary from MIMIC-train values."""

    if not categorical_columns:
        # Write an empty but typed table so downstream scans never fail.
        query = """
SELECT
    CAST(NULL AS VARCHAR) AS column_name,
    CAST(NULL AS VARCHAR) AS value,
    CAST(NULL AS BIGINT) AS token_index
WHERE FALSE
"""
        return copy_query_to_parquet(
            connection, query, config.categorical_vocabulary_path
        )
    selects = [
        f"""
SELECT DISTINCT
    {sql_string(name)} AS column_name,
    CAST({quote_identifier(name)} AS VARCHAR) AS value
FROM ({_train_features_scope(config)}) AS train_features
WHERE {quote_identifier(name)} IS NOT NULL
"""
        for name in categorical_columns
    ]
    unioned = "\nUNION ALL\n".join(selects)
    query = f"""
WITH values AS (
{unioned}
)
SELECT
    column_name,
    value,
    CAST(
        ROW_NUMBER() OVER (PARTITION BY column_name ORDER BY value)
        - 1 + {RESERVED_TOKEN_COUNT} AS BIGINT
    ) AS token_index
FROM values
"""
    return copy_query_to_parquet(connection, query, config.categorical_vocabulary_path)


def build_global_candidate_prior(
    connection: duckdb.DuckDBPyConnection,
    config: NeuralTrainingConfig,
) -> int:
    """Fit train-only global candidate log-odds priors with additive smoothing.

    Counts are taken from MIMIC-train ranking rows only. Validation and test
    joins reuse this table; unknown candidates fall back to the catalog-mean
    prior via ``COALESCE`` in :func:`groups_query`.
    """

    alpha = float(PRIOR_SMOOTHING_ALPHA)
    pcm = parquet_scan(config.patient_condition_medication_path)
    catalog = parquet_scan(config.candidate_catalog_path)
    query = f"""
WITH train_rows AS (
    SELECT candidate_medication_token, label_prescribed
    FROM {pcm}
    WHERE source = {sql_string(DEVELOPMENT_SOURCE)}
        AND split = 'train'
        AND ranking_group_id IS NOT NULL
        AND candidate_medication_token IS NOT NULL
),
totals AS (
    SELECT
        COUNT(*)::DOUBLE AS n_total,
        SUM(CASE WHEN label_prescribed THEN 1 ELSE 0 END)::DOUBLE AS n_pos
    FROM train_rows
),
per_candidate AS (
    SELECT
        candidate_medication_token AS token,
        COUNT(*)::DOUBLE AS n_total,
        SUM(CASE WHEN label_prescribed THEN 1 ELSE 0 END)::DOUBLE AS n_pos
    FROM train_rows
    GROUP BY candidate_medication_token
),
catalog_tokens AS (
    SELECT DISTINCT candidate_medication_token AS token
    FROM {catalog}
    WHERE candidate_medication_token IS NOT NULL
),
background AS (
    SELECT
        LN(({alpha} + COALESCE(totals.n_pos, 0.0))
            / ({alpha} + GREATEST(COALESCE(totals.n_total, 0.0)
                - COALESCE(totals.n_pos, 0.0), 0.0))) AS background_log_odds
    FROM totals
)
SELECT
    catalog_tokens.token,
    LN(
        ({alpha} + COALESCE(per_candidate.n_pos, 0.0))
        / ({alpha} + GREATEST(
            COALESCE(per_candidate.n_total, 0.0) - COALESCE(per_candidate.n_pos, 0.0),
            0.0
        ))
    ) AS log_odds,
    COALESCE(per_candidate.n_pos, 0.0) AS positive_count,
    COALESCE(per_candidate.n_total, 0.0) AS row_count,
    background.background_log_odds
FROM catalog_tokens
CROSS JOIN background
LEFT JOIN per_candidate
    ON catalog_tokens.token = per_candidate.token
"""
    return copy_query_to_parquet(connection, query, config.global_candidate_prior_path)


def build_condition_candidate_prior(
    connection: duckdb.DuckDBPyConnection,
    config: NeuralTrainingConfig,
) -> int:
    """Fit train-only condition×candidate log-odds priors with additive smoothing."""

    alpha = float(PRIOR_SMOOTHING_ALPHA)
    pcm = parquet_scan(config.patient_condition_medication_path)
    query = f"""
WITH train_rows AS (
    SELECT
        index_condition_token,
        candidate_medication_token,
        label_prescribed
    FROM {pcm}
    WHERE source = {sql_string(DEVELOPMENT_SOURCE)}
        AND split = 'train'
        AND ranking_group_id IS NOT NULL
        AND index_condition_token IS NOT NULL
        AND candidate_medication_token IS NOT NULL
),
per_pair AS (
    SELECT
        index_condition_token,
        candidate_medication_token,
        COUNT(*)::DOUBLE AS n_total,
        SUM(CASE WHEN label_prescribed THEN 1 ELSE 0 END)::DOUBLE AS n_pos
    FROM train_rows
    GROUP BY index_condition_token, candidate_medication_token
)
SELECT
    index_condition_token,
    candidate_medication_token,
    LN(
        ({alpha} + n_pos)
        / ({alpha} + GREATEST(n_total - n_pos, 0.0))
    ) AS log_odds,
    n_pos AS positive_count,
    n_total AS row_count
FROM per_pair
"""
    return copy_query_to_parquet(
        connection, query, config.condition_candidate_prior_path
    )


def vocab_embedding_size(
    connection: duckdb.DuckDBPyConnection,
    path: Path,
) -> int:
    """Return ``max(token_index) + 1`` (embedding rows including reserved)."""

    value = _scalar(
        connection,
        f"SELECT MAX(token_index) FROM {parquet_scan(path)}",
    )
    if value is None:
        return RESERVED_TOKEN_COUNT
    return int(value) + 1


def categorical_embedding_sizes(
    connection: duckdb.DuckDBPyConnection,
    config: NeuralTrainingConfig,
) -> dict[str, int]:
    rows = fetch_dict_rows(
        connection,
        f"""
SELECT column_name, MAX(token_index) AS max_index
FROM {parquet_scan(config.categorical_vocabulary_path)}
GROUP BY column_name
""",
    )
    return {
        str(row["column_name"]): int(row["max_index"]) + 1
        for row in rows
        if row["column_name"] is not None and row["max_index"] is not None
    }


# ---------------------------------------------------------------------------
# Shard filters and cache queries
# ---------------------------------------------------------------------------
def stay_shard_filter(*, alias: str, shard_index: int, shard_count: int) -> str:
    """Return a deterministic ``hash(source, stay_uid)`` shard predicate."""

    expression = (
        f"COALESCE({alias}.source, '') || '|' || "
        f"COALESCE(CAST({alias}.stay_uid AS VARCHAR), '')"
    )
    return f"(HASH({expression}) % {int(shard_count)}) = {int(shard_index)}"


def _numeric_projection(
    numeric_stats: dict[str, tuple[float, float]],
    numeric_columns: Sequence[str],
    *,
    alias: str,
) -> str:
    projections: list[str] = []
    for name in numeric_columns:
        mean, std = numeric_stats.get(name, (0.0, 1.0))
        column = f"CAST({alias}.{quote_identifier(name)} AS DOUBLE)"
        usable = _finite_bounded_expr(column)
        normalized = f"({usable} - {mean!r}) / {std!r}"
        # Map NULL, non-finite, and magnitude-overflow feature values (for
        # example a degenerate or explosively large ``REGR_SLOPE``) to the
        # normalized mean (0.0) so the tensor cache never carries ``inf``/
        # ``NaN`` or pathological magnitudes into training.
        projections.append(
            f"CASE WHEN {usable} IS NOT NULL AND isfinite({normalized}) "
            f"THEN {normalized} ELSE 0.0 END AS {quote_identifier(name)}"
        )
    return ",\n        ".join(projections)


def context_features_query(
    config: NeuralTrainingConfig,
    layout: FeatureLayout,
    numeric_stats: dict[str, tuple[float, float]],
    *,
    split: str,
    shard_index: int,
    positive_groups_only: bool,
) -> str:
    """Return one shard of normalized stay-context feature rows."""

    numeric_sql = _numeric_projection(
        numeric_stats, layout.numeric_columns, alias="psf"
    )
    numeric_clause = f",\n        {numeric_sql}" if numeric_sql else ""
    categorical_joins: list[str] = []
    categorical_selects: list[str] = []
    for position, name in enumerate(layout.categorical_columns):
        join_alias = f"cat{position}"
        categorical_joins.append(
            f"""
LEFT JOIN (
    SELECT value, token_index
    FROM {parquet_scan(config.categorical_vocabulary_path)}
    WHERE column_name = {sql_string(name)}
) AS {join_alias}
    ON CAST(psf.{quote_identifier(name)} AS VARCHAR) = {join_alias}.value
"""
        )
        categorical_selects.append(
            f"CAST(COALESCE({join_alias}.token_index, {UNK_INDEX}) AS BIGINT) "
            f"AS {quote_identifier(name + '_index')}"
        )
    categorical_clause = (
        ",\n        " + ",\n        ".join(categorical_selects)
        if categorical_selects
        else ""
    )
    stay_scope = _stay_scope_cte(
        config,
        split=split,
        positive_groups_only=positive_groups_only,
    )
    shard = stay_shard_filter(
        alias="psf", shard_index=shard_index, shard_count=config.shard_count
    )
    return f"""
WITH stay_scope AS (
{stay_scope}
)
SELECT
        psf.source,
        psf.split,
        psf.stay_uid{numeric_clause}{categorical_clause}
FROM {parquet_scan(config.patient_stay_features_path)} AS psf
INNER JOIN stay_scope
    ON psf.source = stay_scope.source
    AND psf.stay_uid = stay_scope.stay_uid
{"".join(categorical_joins)}
WHERE psf.source = {sql_string(DEVELOPMENT_SOURCE)}
    AND psf.split = {sql_string(split)}
    AND {shard}
"""


def _stay_scope_cte(
    config: NeuralTrainingConfig,
    *,
    split: str,
    positive_groups_only: bool,
) -> str:
    """Return distinct stays contributing at least one kept ranking group."""

    positive_filter = ""
    if positive_groups_only:
        positive_filter = """
        AND ranking_group_id IN (
            SELECT ranking_group_id
            FROM group_rows
            GROUP BY ranking_group_id
            HAVING SUM(CASE WHEN label_prescribed THEN 1 ELSE 0 END) > 0
        )"""
    return f"""
    WITH group_rows AS (
        SELECT source, split, stay_uid, ranking_group_id, label_prescribed
        FROM {parquet_scan(config.patient_condition_medication_path)}
        WHERE source = {sql_string(DEVELOPMENT_SOURCE)}
            AND split = {sql_string(split)}
            AND ranking_group_id IS NOT NULL
    )
    SELECT DISTINCT source, stay_uid
    FROM group_rows
    WHERE TRUE{positive_filter}
"""


def context_events_query(
    config: NeuralTrainingConfig,
    layout: FeatureLayout,
    *,
    event_mean: float,
    event_std: float,
    split: str,
    shard_index: int,
    positive_groups_only: bool,
) -> str:
    """Return one shard of truncated, normalized event token rows.

    The most recent ``max_sequence_length`` in-window events per stay are kept
    (ties broken by sequence position). ``recency_rank`` is ``1`` for the most
    recent event so the dataset can order oldest-first deterministically.
    """

    stay_scope = _stay_scope_cte(
        config,
        split=split,
        positive_groups_only=positive_groups_only,
    )
    shard = stay_shard_filter(
        alias="events", shard_index=shard_index, shard_count=config.shard_count
    )
    offset = PREDICTION_OFFSET_HOURS
    return f"""
WITH stay_scope AS (
{stay_scope}
),
in_window AS (
    SELECT
        events.source,
        events.split,
        events.stay_uid,
        events.event_type,
        events.event_token,
        events.event_time_hours_from_admit,
        events.event_sequence_position,
        events.value_numeric
    FROM {parquet_scan(config.event_sequences_path)} AS events
    INNER JOIN stay_scope
        ON events.source = stay_scope.source
        AND events.stay_uid = stay_scope.stay_uid
    WHERE events.source = {sql_string(DEVELOPMENT_SOURCE)}
        AND events.split = {sql_string(split)}
        AND events.event_type <> 'medication'
        AND events.event_token IS NOT NULL
        AND events.event_time_hours_from_admit >= 0
        AND events.event_time_hours_from_admit <= {offset}
        AND {shard}
),
ranked AS (
    SELECT
        in_window.*,
        ROW_NUMBER() OVER (
            PARTITION BY source, stay_uid
            ORDER BY
                event_time_hours_from_admit DESC,
                event_sequence_position DESC
        ) AS recency_rank
    FROM in_window
)
SELECT
    ranked.source,
    ranked.split,
    ranked.stay_uid,
    CAST(ranked.recency_rank AS BIGINT) AS recency_rank,
    CAST(COALESCE(vocab.token_index, {UNK_INDEX}) AS BIGINT) AS event_index,
    ranked.event_time_hours_from_admit / {offset}.0 AS event_time_norm,
    CASE
        WHEN {_finite_bounded_expr("CAST(ranked.value_numeric AS DOUBLE)")} IS NOT NULL
            AND isfinite(
                (
                    {_finite_bounded_expr("CAST(ranked.value_numeric AS DOUBLE)")}
                    - {event_mean!r}
                ) / {event_std!r}
            )
        THEN (
            {_finite_bounded_expr("CAST(ranked.value_numeric AS DOUBLE)")}
            - {event_mean!r}
        ) / {event_std!r}
        ELSE 0.0
    END AS event_value_norm,
    CASE
        WHEN {_finite_bounded_expr("CAST(ranked.value_numeric AS DOUBLE)")} IS NULL
        THEN 0.0 ELSE 1.0
    END AS event_value_mask
FROM ranked
LEFT JOIN {parquet_scan(config.event_vocabulary_path)} AS vocab
    ON ranked.event_type || '|' || ranked.event_token = vocab.token
WHERE ranked.recency_rank <= {int(layout.max_sequence_length)}
"""


def groups_query(
    config: NeuralTrainingConfig,
    *,
    split: str,
    shard_index: int,
    positive_groups_only: bool,
) -> str:
    """Return one shard of per-candidate ranking rows with indexes and priors.

    Candidate-side features are train-fit only:
    ``log1p(candidate_rank)``, global candidate log-odds, and
    condition×candidate log-odds (falling back to the global prior).
    """

    shard = stay_shard_filter(
        alias="pcm", shard_index=shard_index, shard_count=config.shard_count
    )
    positive_filter = ""
    if positive_groups_only:
        positive_filter = """
    AND pcm.ranking_group_id IN (
        SELECT ranking_group_id
        FROM (
            SELECT ranking_group_id, label_prescribed
            FROM scoped_rows
        )
        GROUP BY ranking_group_id
        HAVING SUM(CASE WHEN label_prescribed THEN 1 ELSE 0 END) > 0
    )"""
    return f"""
WITH scoped_rows AS (
    SELECT
        source,
        split,
        stay_uid,
        ranking_group_id,
        index_condition_token,
        candidate_medication_token,
        candidate_rank,
        label_prescribed
    FROM {parquet_scan(config.patient_condition_medication_path)}
    WHERE source = {sql_string(DEVELOPMENT_SOURCE)}
        AND split = {sql_string(split)}
        AND ranking_group_id IS NOT NULL
        AND candidate_medication_token IS NOT NULL
),
background AS (
    SELECT MAX(background_log_odds) AS background_log_odds
    FROM {parquet_scan(config.global_candidate_prior_path)}
)
SELECT
    pcm.source,
    pcm.split,
    pcm.stay_uid,
    pcm.ranking_group_id,
    pcm.index_condition_token,
    pcm.candidate_medication_token,
    pcm.candidate_rank,
    pcm.label_prescribed,
    CAST(COALESCE(cond.token_index, {UNK_INDEX}) AS BIGINT) AS condition_index,
    CAST(COALESCE(med.token_index, {UNK_INDEX}) AS BIGINT) AS candidate_index,
    LN(1.0 + CAST(GREATEST(COALESCE(pcm.candidate_rank, 0), 0) AS DOUBLE))
        AS candidate_rank_feat,
    CAST(
        COALESCE(global_prior.log_odds, background.background_log_odds, 0.0)
        AS DOUBLE
    ) AS global_prior,
    CAST(
        COALESCE(
            pair_prior.log_odds,
            global_prior.log_odds,
            background.background_log_odds,
            0.0
        ) AS DOUBLE
    ) AS condition_candidate_prior
FROM scoped_rows AS pcm
CROSS JOIN background
LEFT JOIN {parquet_scan(config.condition_vocabulary_path)} AS cond
    ON pcm.index_condition_token = cond.token
LEFT JOIN {parquet_scan(config.candidate_vocabulary_path)} AS med
    ON pcm.candidate_medication_token = med.token
LEFT JOIN {parquet_scan(config.global_candidate_prior_path)} AS global_prior
    ON pcm.candidate_medication_token = global_prior.token
LEFT JOIN {parquet_scan(config.condition_candidate_prior_path)} AS pair_prior
    ON pcm.index_condition_token = pair_prior.index_condition_token
    AND pcm.candidate_medication_token = pair_prior.candidate_medication_token
WHERE {shard}{positive_filter}
"""


def ranking_group_coverage(
    connection: duckdb.DuckDBPyConnection,
    config: NeuralTrainingConfig,
) -> list[dict[str, Any]]:
    """Report aggregate positive/zero-positive group counts per split."""

    splits = ", ".join(sql_string(split) for split in config.evaluation_splits())
    return fetch_dict_rows(
        connection,
        f"""
WITH group_labels AS (
    SELECT
        source,
        split,
        ranking_group_id,
        MAX(CASE WHEN label_prescribed THEN 1 ELSE 0 END) AS has_positive
    FROM {parquet_scan(config.patient_condition_medication_path)}
    WHERE source = {sql_string(DEVELOPMENT_SOURCE)} AND split IN ({splits})
    GROUP BY source, split, ranking_group_id
)
SELECT
    source,
    split,
    COUNT(*) AS ranking_group_count,
    SUM(has_positive) AS positive_group_count,
    SUM(CASE WHEN has_positive = 0 THEN 1 ELSE 0 END) AS zero_positive_group_count
FROM group_labels
GROUP BY source, split
ORDER BY source, split
""",
    )


def _materialize_split(
    connection: duckdb.DuckDBPyConnection,
    config: NeuralTrainingConfig,
    layout: FeatureLayout,
    numeric_stats: dict[str, tuple[float, float]],
    *,
    event_mean: float,
    event_std: float,
    split: str,
) -> dict[str, Any]:
    """Materialize context, event, and group caches for one split."""

    positive_groups_only = split == "train"
    context_dir = config.context_features_dir(split)
    events_dir = config.context_events_dir(split)
    groups_dir = config.groups_dir(split)
    for directory in (context_dir, events_dir, groups_dir):
        shutil.rmtree(directory, ignore_errors=True)
        directory.mkdir(parents=True, exist_ok=True)

    context_rows = 0
    event_rows = 0
    group_rows = 0
    for shard_index in range(config.shard_count):
        context_rows += copy_query_to_parquet(
            connection,
            context_features_query(
                config,
                layout,
                numeric_stats,
                split=split,
                shard_index=shard_index,
                positive_groups_only=positive_groups_only,
            ),
            context_dir / f"shard_{shard_index:04d}.parquet",
        )
        event_rows += copy_query_to_parquet(
            connection,
            context_events_query(
                config,
                layout,
                event_mean=event_mean,
                event_std=event_std,
                split=split,
                shard_index=shard_index,
                positive_groups_only=positive_groups_only,
            ),
            events_dir / f"shard_{shard_index:04d}.parquet",
        )
        group_rows += copy_query_to_parquet(
            connection,
            groups_query(
                config,
                split=split,
                shard_index=shard_index,
                positive_groups_only=positive_groups_only,
            ),
            groups_dir / f"shard_{shard_index:04d}.parquet",
        )
    return {
        "split": split,
        "positive_groups_only": positive_groups_only,
        "shard_count": config.shard_count,
        "stay_context_row_count": context_rows,
        "event_row_count": event_rows,
        "candidate_row_count": group_rows,
    }


def base_manifest(
    config: NeuralTrainingConfig,
    *,
    status: str,
    generated_at: str,
) -> dict[str, Any]:
    """Return the aggregate-only prepare manifest shell."""

    return {
        "schema_version": PREPARE_SCHEMA_VERSION,
        "status": status,
        "stage": "prepare",
        "mode": config.mode,
        "generated_at": generated_at,
        "parameters": {
            "development_source": DEVELOPMENT_SOURCE,
            "prediction_offset_hours": PREDICTION_OFFSET_HOURS,
            "max_sequence_length": config.max_sequence_length,
            "shard_count": config.shard_count,
            "reserved_token_count": RESERVED_TOKEN_COUNT,
            "seed": config.seed,
            "splits": list(config.evaluation_splits()),
        },
        "leakage_policy": {
            "vocabulary_fit_scope": "mimiciv_train",
            "normalization_fit_scope": "mimiciv_train",
            "prior_fit_scope": "mimiciv_train",
            "event_window": f"0 <= event_time_hours_from_admit <= "
            f"{PREDICTION_OFFSET_HOURS}",
            "medication_events_excluded": True,
            "train_cache_excludes_zero_positive_groups": True,
        },
        "data_safety": {
            "manifest_contains_patient_rows": False,
            "manifest_contains_row_samples": False,
            "local_caches_contain_patient_level_rows": True,
            "cache_storage": str(config.cache_root),
        },
        "artifacts": {},
        "splits": [],
    }


def prepare_neural_caches(config: NeuralTrainingConfig) -> dict[str, Any]:
    """Build vocabularies, normalization, feature layout, and sharded caches."""

    from pipeline.neural_training.contract import (
        blocked_report,
        preflight_errors,
    )

    generated_at = datetime.now(UTC).isoformat()
    errors = preflight_errors(config, stage="prepare")
    if errors:
        report = blocked_report(
            schema_version=PREPARE_SCHEMA_VERSION,
            stage="prepare",
            mode=config.mode,
            generated_at=generated_at,
            errors=errors,
        )
        write_json(config.prepare_manifest_path, report)
        return report

    config.vocab_root.mkdir(parents=True, exist_ok=True)
    config.cache_root.mkdir(parents=True, exist_ok=True)
    manifest = base_manifest(config, status="completed", generated_at=generated_at)
    try:
        with duckdb.connect(database=":memory:") as connection:
            configure_connection(config, connection)
            layout = resolve_feature_layout(connection, config)
            numeric_stats, numeric_nonfinite = numeric_normalization_stats(
                connection, config, layout.numeric_columns
            )
            event_mean, event_std = event_value_stats(connection, config)
            write_normalization_stats(
                connection,
                config,
                numeric_stats=numeric_stats,
                event_mean=event_mean,
                event_std=event_std,
            )
            build_event_vocabulary(connection, config)
            build_condition_vocabulary(connection, config)
            build_candidate_vocabulary(connection, config)
            build_categorical_vocabulary(connection, config, layout.categorical_columns)
            global_prior_rows = build_global_candidate_prior(connection, config)
            pair_prior_rows = build_condition_candidate_prior(connection, config)
            vocab_sizes = {
                "event": vocab_embedding_size(connection, config.event_vocabulary_path),
                "condition": vocab_embedding_size(
                    connection, config.condition_vocabulary_path
                ),
                "candidate": vocab_embedding_size(
                    connection, config.candidate_vocabulary_path
                ),
                "categorical": categorical_embedding_sizes(connection, config),
            }
            layout_payload = layout.as_dict(vocab_sizes=vocab_sizes)
            layout_payload["candidate_side_features"] = [
                "candidate_rank_feat",
                "global_prior",
                "condition_candidate_prior",
            ]
            write_json(config.feature_layout_path, layout_payload)
            manifest["feature_layout"] = {
                "numeric_column_count": len(layout.numeric_columns),
                "categorical_column_count": len(layout.categorical_columns),
                "max_sequence_length": layout.max_sequence_length,
                "vocab_sizes": vocab_sizes,
                "feature_version": layout.feature_version,
                "candidate_side_features": layout_payload["candidate_side_features"],
                "global_candidate_prior_rows": global_prior_rows,
                "condition_candidate_prior_rows": pair_prior_rows,
            }
            # Aggregate-only data-quality note: some upstream trend columns can
            # carry non-finite or explosively large finite values (for example
            # a pathological ``REGR_SLOPE``). These are excluded from train
            # normalization and mapped to the normalized mean in the caches;
            # record the counts so the exclusion is auditable without exposing
            # patient rows.
            manifest["data_quality"] = {
                "nonfinite_or_extreme_values_excluded_from_normalization": True,
                "normalization_abs_bound": NORMALIZATION_ABS_BOUND,
                "excluded_numeric_value_counts": numeric_nonfinite,
                # Backward-compatible alias for earlier smoke-test consumers.
                "nonfinite_numeric_value_counts": numeric_nonfinite,
            }
            for split in config.evaluation_splits():
                manifest["splits"].append(
                    _materialize_split(
                        connection,
                        config,
                        layout,
                        numeric_stats,
                        event_mean=event_mean,
                        event_std=event_std,
                        split=split,
                    )
                )
            manifest["ranking_group_coverage"] = ranking_group_coverage(
                connection, config
            )
            manifest["artifacts"] = {
                "feature_layout": str(config.feature_layout_path),
                "normalization_stats": str(config.normalization_path),
                "event_vocabulary": str(config.event_vocabulary_path),
                "condition_vocabulary": str(config.condition_vocabulary_path),
                "candidate_medication_vocabulary": str(
                    config.candidate_vocabulary_path
                ),
                "categorical_vocabulary": str(config.categorical_vocabulary_path),
                "global_candidate_prior": str(config.global_candidate_prior_path),
                "condition_candidate_prior": str(config.condition_candidate_prior_path),
                "cache_root": str(config.cache_root),
            }
            manifest["leakage_policy"]["prior_fit_scope"] = "mimiciv_train"
    except Exception as error:  # noqa: BLE001 - reported as aggregate status
        manifest["status"] = "failed"
        manifest["reason"] = safe_error_message(error)

    write_json(config.prepare_manifest_path, manifest)
    return manifest
