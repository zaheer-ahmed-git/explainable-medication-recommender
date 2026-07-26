"""Tests for the torch-free neural cache preparation and preflight gate."""

from __future__ import annotations

import json
import math
from pathlib import Path

import duckdb

from pipeline.neural_training.config import RESERVED_TOKEN_COUNT
from pipeline.neural_training.contract import preflight_errors
from pipeline.neural_training.data import prepare_neural_caches
from tests.milestone6_helpers import sql_string
from tests.neural_training_helpers import CANDIDATE_TOKENS, write_neural_fixture


def _read_rows(path: Path) -> list[dict[str, object]]:
    with duckdb.connect(database=":memory:") as connection:
        cursor = connection.execute(f"SELECT * FROM read_parquet({sql_string(path)})")
        columns = [description[0] for description in cursor.description]
        return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]


def test_prepare_builds_vocab_layout_and_caches(tmp_path: Path) -> None:
    config = write_neural_fixture(tmp_path)

    manifest = prepare_neural_caches(config)

    assert manifest["status"] == "completed"
    assert config.feature_layout_path.exists()
    layout = json.loads(config.feature_layout_path.read_text(encoding="utf-8"))

    # Approved categoricals surface as low-cardinality columns; identifiers and
    # provenance columns must be excluded from the numeric projection.
    assert set(layout["categorical_columns"]) == {"sex", "admission_type"}
    assert "stay_uid" not in layout["numeric_columns"]
    assert "patient_uid" not in layout["numeric_columns"]
    assert "feature_version" not in layout["numeric_columns"]
    assert {"age_years", "heart_rate_last"} <= set(layout["numeric_columns"])
    assert layout["feature_version"] == "temporal-features-v2"

    # PAD and UNK stay reserved; the event vocabulary excludes medications and
    # out-of-window tokens, keeping only lactate/heart_rate/creatinine... but
    # creatinine is out of the 24h window, so it is dropped.
    event_vocab = _read_rows(config.event_vocabulary_path)
    tokens = {row["token"] for row in event_vocab}
    assert tokens == {"lab|lactate", "vital|heart_rate"}
    assert min(int(row["token_index"]) for row in event_vocab) == RESERVED_TOKEN_COUNT

    candidate_vocab = _read_rows(config.candidate_vocabulary_path)
    assert {row["token"] for row in candidate_vocab} == set(CANDIDATE_TOKENS)

    # Train cache excludes the zero-positive validation group only from train;
    # validation keeps every group for scoring coverage.
    coverage = {row["split"]: row for row in manifest["ranking_group_coverage"]}
    assert coverage["validation"]["zero_positive_group_count"] == 1
    assert coverage["train"]["zero_positive_group_count"] == 0

    # Four train stays, each a positive group of four candidates, all retained.
    train_candidate_rows = sum(
        len(_read_rows(path)) for path in config.groups_dir("train").glob("*.parquet")
    )
    assert train_candidate_rows == 4 * len(CANDIDATE_TOKENS)

    # Train-only priors are persisted and joined into group caches.
    assert config.global_candidate_prior_path.exists()
    assert config.condition_candidate_prior_path.exists()
    assert layout["candidate_side_features"] == [
        "candidate_rank_feat",
        "global_prior",
        "condition_candidate_prior",
    ]
    assert manifest["leakage_policy"]["prior_fit_scope"] == "mimiciv_train"
    group_row = _read_rows(next(config.groups_dir("train").glob("*.parquet")))[0]
    assert "candidate_rank_feat" in group_row
    assert "global_prior" in group_row
    assert "condition_candidate_prior" in group_row
    assert math.isfinite(float(group_row["global_prior"]))


def test_prepare_handles_nonfinite_numeric_features(tmp_path: Path) -> None:
    # Degenerate upstream trend columns can produce inf/NaN (e.g. a
    # zero-time-variance REGR_SLOPE). DuckDB's STDDEV_SAMP raises "Out of Range"
    # on any non-finite input, so prepare must exclude them from normalization
    # and keep the caches finite.
    config = write_neural_fixture(tmp_path, nonfinite_numeric=True)

    manifest = prepare_neural_caches(config)

    assert manifest["status"] == "completed"

    # The excluded non-finite values are reported for auditability.
    counts = manifest["data_quality"]["excluded_numeric_value_counts"]
    assert counts.get("heart_rate_last") == 2

    # Persisted normalization stats are finite and std stays positive.
    stats = _read_rows(config.normalization_path)
    hr_stat = next(row for row in stats if row["column_name"] == "heart_rate_last")
    assert math.isfinite(float(hr_stat["mean"]))
    assert float(hr_stat["std"]) > 0.0

    # No inf/NaN leaks into the materialized context feature caches.
    for path in config.context_features_dir("train").glob("*.parquet"):
        for row in _read_rows(path):
            value = row["heart_rate_last"]
            assert value is not None and math.isfinite(float(value))


def test_prepare_handles_extreme_finite_numeric_features(tmp_path: Path) -> None:
    # Job 28346 failed with STDDEV_SAMP out of range on extreme finite trend
    # values (DuckDB 1.5.x). Magnitude-bounded exclusion must keep prepare
    # alive and map those cells to the normalized mean in the caches.
    config = write_neural_fixture(tmp_path, extreme_numeric=True)

    manifest = prepare_neural_caches(config)

    assert manifest["status"] == "completed"
    assert (
        manifest["data_quality"]["excluded_numeric_value_counts"].get("heart_rate_last")
        == 1
    )

    stats = _read_rows(config.normalization_path)
    hr_stat = next(row for row in stats if row["column_name"] == "heart_rate_last")
    assert math.isfinite(float(hr_stat["mean"]))
    assert float(hr_stat["std"]) > 0.0

    for path in config.context_features_dir("train").glob("*.parquet"):
        for row in _read_rows(path):
            value = float(row["heart_rate_last"])
            assert math.isfinite(value)
            assert abs(value) < 1e6


def test_prepare_is_blocked_when_gate_not_authorized(tmp_path: Path) -> None:
    config = write_neural_fixture(tmp_path, gate_authorized=False)

    manifest = prepare_neural_caches(config)

    assert manifest["status"] == "blocked_preflight"
    codes = {error["code"] for error in manifest["errors"]}
    assert "neural_gate_not_passed" in codes


def test_preflight_missing_gate_selection_blocks(tmp_path: Path) -> None:
    config = write_neural_fixture(tmp_path, write_gate_selection=False)

    errors = preflight_errors(config, stage="prepare")

    assert any(error["code"] == "missing_gate_selection" for error in errors)


def test_prepare_runs_ungated_for_smoke(tmp_path: Path) -> None:
    config = write_neural_fixture(
        tmp_path, gate_authorized=False, require_neural_gate=False
    )

    manifest = prepare_neural_caches(config)

    assert manifest["status"] == "completed"
