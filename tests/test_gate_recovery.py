from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from pipeline.evaluate_baselines import BaselineEvaluationConfig, ranking_metric_query
from pipeline.gate_recovery import (
    GateRecoveryConfig,
    RankerHyperparameters,
    RecoveryExperiment,
    cross_validate_ranker,
    gate_decision,
    patient_fold_sql,
    preflight_errors,
    ranking_metrics_at_k,
    recovery_frame_query,
    resolve_recovery_feature_spec,
    select_oof_fusion_weight,
)
from pipeline.learned_baselines import LearnedFeatureSpec
from tests.milestone6_helpers import write_parquet_rows


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_patient_fold_expression_keeps_one_patient_together() -> None:
    expression = patient_fold_sql(seed=20260617, fold_count=3, alias="rows")
    with duckdb.connect(database=":memory:") as connection:
        result = connection.execute(
            f"""
SELECT patient_uid, {expression} AS fold_id
FROM (
    VALUES
        ('patient-a'),
        ('patient-a'),
        ('patient-b')
) AS rows(patient_uid)
"""
        ).fetchall()

    assert result[0][1] == result[1][1]
    assert all(0 <= row[1] < 3 for row in result)


def test_recovery_query_preserves_group_and_excludes_zero_positive_groups(
    tmp_path: Path,
) -> None:
    features_root = tmp_path / "features"
    training_root = tmp_path / "training"
    write_parquet_rows(
        features_root / "patient_stay_features.parquet",
        (
            "source",
            "patient_uid",
            "stay_uid",
            "split",
            "age_years",
            "condition_a_present_24h",
        ),
        (
            ("mimiciv", "patient-a", "stay-a", "train", 60.0, 1),
            ("mimiciv", "patient-b", "stay-b", "train", 70.0, 0),
        ),
    )
    candidates = []
    for rank in range(1, 16):
        candidates.append(
            (
                "mimiciv",
                "patient-a",
                "stay-a",
                "train",
                "group-positive",
                "condition:a",
                f"rxnorm:{rank}",
                rank,
                rank == 1,
            )
        )
    for rank in range(1, 4):
        candidates.append(
            (
                "mimiciv",
                "patient-b",
                "stay-b",
                "train",
                "group-zero",
                "condition:a",
                f"rxnorm:z{rank}",
                rank,
                False,
            )
        )
    write_parquet_rows(
        training_root / "patient_condition_medication.parquet",
        (
            "source",
            "patient_uid",
            "stay_uid",
            "split",
            "ranking_group_id",
            "index_condition_token",
            "candidate_medication_token",
            "candidate_rank",
            "label_prescribed",
        ),
        tuple(candidates),
    )
    manifest = tmp_path / "feature_manifest.json"
    _write_json(
        manifest,
        {"status": "completed", "condition_columns_added": ["condition_a_present_24h"]},
    )
    config = GateRecoveryConfig(
        features_root=features_root,
        training_root=training_root,
        feature_manifest_path=manifest,
        fold_count=3,
    )
    experiment = RecoveryExperiment(1, 1, "none", True)
    with duckdb.connect(database=":memory:") as connection:
        feature_spec = resolve_recovery_feature_spec(connection, config, experiment)
        frame = connection.execute(
            recovery_frame_query(
                config,
                experiment,
                feature_spec,
                split="train",
                sampled=True,
            )
        ).fetchdf()

    assert len(frame) == 11
    assert frame["label_prescribed"].sum() == 1
    assert set(frame["ranking_group_id"]) == {"group-positive"}
    assert "condition_a_present_24h" in frame.columns


def test_ranking_metrics_and_gate_decision() -> None:
    frame = pd.DataFrame(
        {
            "ranking_group_id": ["g1", "g1", "g1", "g2", "g2"],
            "candidate_medication_token": ["a", "b", "c", "d", "e"],
            "candidate_rank": [1, 2, 3, 1, 2],
            "label_prescribed": [True, False, True, False, True],
            "score": [0.9, 0.2, 0.8, 0.1, 0.7],
        }
    )

    metrics = ranking_metrics_at_k(frame, k=2)
    decision = gate_decision(
        candidate={
            "ndcg_at_k": 0.381,
            "mrr_at_k": 0.49,
            "hit_rate_at_k": 0.85,
        },
        reference={
            "ndcg_at_k": 0.374899,
            "mrr_at_k": 0.495,
            "hit_rate_at_k": 0.853,
        },
    )

    assert metrics["positive_ranking_group_count"] == 2
    assert metrics["hit_rate_at_k"] == 1.0
    assert decision["neural_training_authorized"] is True
    assert decision["required_candidate_ndcg_at_10"] == 0.379899


def test_train_fold_metric_matches_authoritative_ranking_query(tmp_path: Path) -> None:
    rows = (
        ("mimiciv", "train", "g1", "condition:a", "a", 1, True, "candidate", 0.9),
        ("mimiciv", "train", "g1", "condition:a", "b", 2, False, "candidate", 0.2),
        ("mimiciv", "train", "g1", "condition:a", "c", 3, True, "candidate", 0.8),
        ("mimiciv", "train", "g2", "condition:a", "d", 1, False, "candidate", 0.1),
        ("mimiciv", "train", "g2", "condition:a", "e", 2, True, "candidate", 0.7),
    )
    evaluation_root = tmp_path / "evaluation"
    score_path = evaluation_root / "baseline_scores.parquet"
    write_parquet_rows(
        score_path,
        (
            "source",
            "split",
            "ranking_group_id",
            "index_condition_token",
            "candidate_medication_token",
            "candidate_rank",
            "label_prescribed",
            "baseline_name",
            "score",
        ),
        rows,
    )
    config = BaselineEvaluationConfig(
        features_root=tmp_path,
        training_root=tmp_path,
        evaluation_root=evaluation_root,
        top_k=(2,),
    )
    frame = pd.DataFrame(
        rows,
        columns=(
            "source",
            "split",
            "ranking_group_id",
            "index_condition_token",
            "candidate_medication_token",
            "candidate_rank",
            "label_prescribed",
            "baseline_name",
            "score",
        ),
    )
    expected = ranking_metrics_at_k(frame, k=2)

    with duckdb.connect(database=":memory:") as connection:
        authoritative = (
            connection.execute(ranking_metric_query(config, k=2)).fetchdf().iloc[0]
        )

    for metric in (
        "precision_at_k",
        "recall_at_k",
        "hit_rate_at_k",
        "ndcg_at_k",
        "mrr_at_k",
    ):
        assert expected[metric] == pytest.approx(authoritative[metric])


def test_cross_validation_keeps_train_fold_selection_internal() -> None:
    rows = []
    for fold_id in range(3):
        for group_offset in range(2):
            group_id = f"fold-{fold_id}-group-{group_offset}"
            for candidate_rank in range(1, 4):
                rows.append(
                    {
                        "source": "mimiciv",
                        "split": "train",
                        "ranking_group_id": group_id,
                        "index_condition_token": "condition:a",
                        "candidate_medication_token": f"rxnorm:{candidate_rank}",
                        "candidate_rank": candidate_rank,
                        "label_prescribed": candidate_rank == group_offset + 1,
                        "patient_fold_id": fold_id,
                        "age_years": 40.0 + fold_id,
                    }
                )
    frame = pd.DataFrame(rows)
    feature_spec = LearnedFeatureSpec(
        stay_numeric=("age_years",),
        stay_categorical=(),
        row_numeric=("candidate_rank",),
        row_categorical=("index_condition_token", "candidate_medication_token"),
    )
    experiment = RecoveryExperiment(0, 1, "none", True)
    hyperparameters = RankerHyperparameters(
        max_depth=2,
        learning_rate=0.1,
        min_child_weight=1.0,
        max_boost_rounds=8,
        early_stopping_rounds=2,
    )

    result, scores = cross_validate_ranker(
        frame,
        frame,
        feature_spec,
        experiment,
        hyperparameters,
        seed=20260617,
        fold_count=3,
        capture_scores=True,
    )

    assert len(result["folds"]) == 3
    assert scores is not None
    assert len(scores) == len(frame)
    assert set(scores["split"]) == {"train"}


def test_oof_fusion_prefers_candidate_when_candidate_is_better() -> None:
    metadata = {
        "source": ["mimiciv"] * 4,
        "split": ["train"] * 4,
        "ranking_group_id": ["g1", "g1", "g2", "g2"],
        "index_condition_token": ["condition:a"] * 4,
        "candidate_medication_token": ["a", "b", "a", "b"],
        "candidate_rank": [1, 2, 1, 2],
        "label_prescribed": [True, False, True, False],
    }
    candidate = pd.DataFrame({**metadata, "score": [0.9, 0.1, 0.8, 0.2]})
    reference = pd.DataFrame({**metadata, "score": [0.1, 0.9, 0.2, 0.8]})

    selection = select_oof_fusion_weight(candidate, reference)

    assert selection["status"] == "selected_from_mimic_train_oof"
    assert selection["selected_candidate_weight"] > 0.5


def test_final_mode_is_blocked_without_passing_frozen_selection(
    tmp_path: Path,
) -> None:
    contract = tmp_path / "contract.json"
    reference_selection = tmp_path / "reference.json"
    selection = tmp_path / "selection.json"
    _write_json(contract, {"status": "completed", "contract_digest": "abc"})
    _write_json(
        reference_selection, {"selected_experiment": "xgboost_frozen_reference"}
    )
    _write_json(selection, {"neural_training_authorized": False})
    data_paths = [
        tmp_path / "features" / "patient_stay_features.parquet",
        tmp_path / "training" / "patient_condition_medication.parquet",
        tmp_path / "training" / "candidate_catalog.parquet",
        tmp_path / "graph" / "graph_edges.parquet",
        tmp_path / "reference_scores.parquet",
        tmp_path / "feature_manifest.json",
    ]
    for path in data_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    config = GateRecoveryConfig(
        features_root=tmp_path / "features",
        training_root=tmp_path / "training",
        graph_root=tmp_path / "graph",
        reference_scores_path=tmp_path / "reference_scores.parquet",
        reference_selection_path=reference_selection,
        contract_lock_path=contract,
        feature_manifest_path=tmp_path / "feature_manifest.json",
        selection_report_path=selection,
        mode="final",
        frozen_selection=True,
    )

    errors = preflight_errors(config)

    assert "recovery_gate_not_passed" in {row["code"] for row in errors}
