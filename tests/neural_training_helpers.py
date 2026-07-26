"""Synthetic model-ready fixtures for the neural training tests.

These builders write tiny, non-identifying Parquet tables that mirror the Phase 8
P0 model-ready schema (stay features, event sequences, ranking rows, candidate
catalog) plus a training-contract lock and a neural gate-selection report. They
never contain real clinical data.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from pipeline.neural_training.config import NeuralTrainingConfig
from tests.milestone6_helpers import write_parquet_rows

FEATURE_VERSION = "temporal-features-v2"
CONTRACT_DIGEST = "synthetic-contract-digest"
INDEX_CONDITION = "cond:sepsis"
CANDIDATE_TOKENS = ("rxnorm:1", "rxnorm:2", "rxnorm:3", "rxnorm:4")

_STAY_COLUMNS = (
    "source",
    "split",
    "patient_uid",
    "stay_uid",
    "feature_version",
    "age_years",
    "heart_rate_last",
    "sex",
    "admission_type",
)

_EVENT_COLUMNS = (
    "source",
    "split",
    "stay_uid",
    "event_sequence_position",
    "event_type",
    "event_time_hours_from_admit",
    "event_token",
    "value_numeric",
    "normalized_unit",
)

_PCM_COLUMNS = (
    "source",
    "split",
    "patient_uid",
    "stay_uid",
    "ranking_group_id",
    "index_condition_token",
    "candidate_medication_token",
    "candidate_rank",
    "label_prescribed",
)

_CATALOG_COLUMNS = (
    "index_condition_token",
    "candidate_medication_token",
    "candidate_rank",
    "positive_train_stay_count",
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _stay_rows(
    split: str,
    count: int,
    *,
    nonfinite_numeric: bool = False,
    extreme_numeric: bool = False,
) -> list[tuple[Any, ...]]:
    rows: list[tuple[Any, ...]] = []
    for index in range(count):
        heart_rate = 80.0 + index
        # Inject non-finite train features (mirroring a degenerate REGR_SLOPE
        # trend) so the normalization path is exercised against inf/NaN.
        if nonfinite_numeric and split == "train":
            if index == 0:
                heart_rate = float("inf")
            elif index == 1:
                heart_rate = float("nan")
        # DuckDB 1.5.x STDDEV_SAMP still overflows on extreme finite values
        # even after isfinite filtering (job 28346 failure mode).
        if extreme_numeric and split == "train" and index == 0:
            heart_rate = 1e200
        rows.append(
            (
                "mimiciv",
                split,
                f"mimiciv:patient-{split}-{index}",
                f"mimiciv:stay-{split}-{index}",
                FEATURE_VERSION,
                60.0 + index,
                heart_rate,
                "F" if index % 2 == 0 else "M",
                "urgent" if index % 2 == 0 else "elective",
            )
        )
    return rows


def _event_rows(split: str, count: int) -> list[tuple[Any, ...]]:
    rows: list[tuple[Any, ...]] = []
    for index in range(count):
        stay = f"mimiciv:stay-{split}-{index}"
        in_window = [
            ("lab", "lactate", 2.0, 1.5),
            ("lab", "lactate", 6.0, 2.4),
            ("vital", "heart_rate", 10.0, 88.0),
            # A medication event and an out-of-window event that must be dropped.
            ("medication", "rxnorm:1", 3.0, None),
            ("lab", "creatinine", 30.0, 1.1),
        ]
        for position, (event_type, token, hours, value) in enumerate(in_window):
            rows.append(
                (
                    "mimiciv",
                    split,
                    stay,
                    position,
                    event_type,
                    hours,
                    token,
                    value,
                    "unit" if value is not None else None,
                )
            )
    return rows


def _pcm_rows(split: str, count: int) -> list[tuple[Any, ...]]:
    rows: list[tuple[Any, ...]] = []
    for index in range(count):
        stay = f"mimiciv:stay-{split}-{index}"
        group = f"grp-{split}-{index}"
        # The last validation stay is a zero-positive group (coverage only).
        zero_positive = split == "validation" and index == count - 1
        for rank, token in enumerate(CANDIDATE_TOKENS, start=1):
            label = (rank == 1) and not zero_positive
            rows.append(
                (
                    "mimiciv",
                    split,
                    f"mimiciv:patient-{split}-{index}",
                    stay,
                    group,
                    INDEX_CONDITION,
                    token,
                    rank,
                    label,
                )
            )
    return rows


def _catalog_rows() -> list[tuple[Any, ...]]:
    return [
        (INDEX_CONDITION, token, rank, 10 - rank)
        for rank, token in enumerate(CANDIDATE_TOKENS, start=1)
    ]


def write_neural_fixture(
    root: Path,
    *,
    train_stays: int = 4,
    validation_stays: int = 4,
    gate_authorized: bool = True,
    write_gate_selection: bool = True,
    max_sequence_length: int = 8,
    shard_count: int = 2,
    require_neural_gate: bool = True,
    mode: str = "development",
    nonfinite_numeric: bool = False,
    extreme_numeric: bool = False,
) -> NeuralTrainingConfig:
    """Write synthetic model-ready inputs and return a matching config."""

    features_root = root / "features"
    training_root = root / "training"
    neural_root = root / "neural"

    write_parquet_rows(
        features_root / "patient_stay_features.parquet",
        _STAY_COLUMNS,
        tuple(
            _stay_rows(
                "train",
                train_stays,
                nonfinite_numeric=nonfinite_numeric,
                extreme_numeric=extreme_numeric,
            )
            + _stay_rows("validation", validation_stays)
        ),
    )
    write_parquet_rows(
        features_root / "event_sequences.parquet",
        _EVENT_COLUMNS,
        tuple(
            _event_rows("train", train_stays)
            + _event_rows("validation", validation_stays)
        ),
    )
    write_parquet_rows(
        training_root / "patient_condition_medication.parquet",
        _PCM_COLUMNS,
        tuple(
            _pcm_rows("train", train_stays) + _pcm_rows("validation", validation_stays)
        ),
    )
    write_parquet_rows(
        training_root / "candidate_catalog.parquet",
        _CATALOG_COLUMNS,
        tuple(_catalog_rows()),
    )

    contract_lock = root / "reports" / "training_contract_lock.json"
    _write_json(
        contract_lock,
        {"status": "completed", "contract_digest": CONTRACT_DIGEST},
    )
    gate_selection = root / "reports" / "gate_recovery_selection.json"
    if write_gate_selection:
        _write_json(
            gate_selection,
            {
                "status": "frozen",
                "neural_training_authorized": gate_authorized,
                "contract_digest": CONTRACT_DIGEST,
                "selected_experiment": "xgboost_rank_ndcg_oof_late_fusion",
                "candidate_ndcg_at_10": 0.3946069881534658,
                "reference_ndcg_at_10": 0.3748994692628306,
                "selection_basis": {
                    "candidate_metrics": {
                        "baseline_name": "xgboost_rank_ndcg_oof_late_fusion",
                        "ndcg_at_k": 0.3946069881534658,
                        "mrr_at_k": 0.5110430572914852,
                        "hit_rate_at_k": 0.8637422561217623,
                    }
                },
            },
        )

    return replace(
        NeuralTrainingConfig(),
        features_root=features_root,
        training_root=training_root,
        neural_root=neural_root,
        contract_lock_path=contract_lock,
        gate_selection_path=gate_selection,
        prepare_manifest_path=root / "reports" / "prepare_manifest.json",
        training_report_path=root / "reports" / "training_evaluation.json",
        score_report_path=root / "reports" / "score_evaluation.json",
        selection_report_path=root / "reports" / "neural_selection.json",
        reference_scores_path=root / "reports" / "_absent_reference.parquet",
        mode=mode,
        max_sequence_length=max_sequence_length,
        shard_count=shard_count,
        require_neural_gate=require_neural_gate,
        duckdb_temp_directory=None,
        duckdb_memory_limit=None,
        duckdb_threads=None,
    )
