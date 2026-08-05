"""Synthetic CPU smoke test for the complete GNN and fusion workflow."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from pipeline.gnn_training.config import (
    GRAPH_REFERENCE_BASELINE_NAME,
    GNNArchitecture,
    GNNOptimization,
    GNNTrainingConfig,
)
from pipeline.gnn_training.data import prepare_gnn_caches, write_json
from pipeline.gnn_training.crossfit import crossfit_artifact_lock_errors
from pipeline.gnn_training.model import ABLATION_VARIANTS
from pipeline.neural_training.config import NeuralArchitecture, NeuralOptimization
from pipeline.neural_training.data import prepare_neural_caches
from tests.milestone6_helpers import write_parquet_rows
from tests.neural_training_helpers import (
    CANDIDATE_TOKENS,
    INDEX_CONDITION,
    write_neural_fixture,
)

pytest.importorskip("torch")

from pipeline.gnn_training.score_fusion import score_fusion  # noqa: E402
from pipeline.gnn_training.score_gnn import score_gnn  # noqa: E402
from pipeline.gnn_training.train_fusion import train_fusion  # noqa: E402
from pipeline.gnn_training.train_gnn import (  # noqa: E402
    refit_selected_gnn,
    select_gnn,
    train_gnn,
)
from pipeline.neural_training.train import train_transformer  # noqa: E402


def _write_patient_subgraphs(
    config: GNNTrainingConfig,
    *,
    train_stays: int,
    validation_stays: int,
) -> None:
    index_rows: list[tuple[object, ...]] = []
    node_rows: list[tuple[object, ...]] = []
    edge_rows: list[tuple[object, ...]] = []
    candidate_rows: list[tuple[object, ...]] = []
    for split, count in (
        ("train", train_stays),
        ("validation", validation_stays),
    ):
        for stay_index in range(count):
            group_id = f"grp-{split}-{stay_index}"
            positive_count = int(
                not (split == "validation" and stay_index == count - 1)
            )
            index_rows.append(
                (
                    "mimiciv",
                    split,
                    group_id,
                    INDEX_CONDITION,
                    7,
                    3,
                    len(CANDIDATE_TOKENS),
                    positive_count,
                )
            )
            node_rows.append(
                (
                    "mimiciv",
                    split,
                    group_id,
                    0,
                    f"condition|{INDEX_CONDITION}",
                    "condition",
                    "query_condition",
                    False,
                    False,
                )
            )
            for rank, token in enumerate(CANDIDATE_TOKENS, start=1):
                node_rows.append(
                    (
                        "mimiciv",
                        split,
                        group_id,
                        rank,
                        f"medication|{token}",
                        "medication",
                        "candidate_medication",
                        False,
                        False,
                    )
                )
                candidate_rows.append(
                    (
                        "mimiciv",
                        split,
                        group_id,
                        INDEX_CONDITION,
                        token,
                        rank,
                        rank,
                        rank == 1 and positive_count == 1,
                        False,
                    )
                )
            node_rows.extend(
                (
                    (
                        "mimiciv",
                        split,
                        group_id,
                        5,
                        "lab|lactate",
                        "lab",
                        "observed_context",
                        True,
                        False,
                    ),
                    (
                        "mimiciv",
                        split,
                        group_id,
                        6,
                        "vital|heart_rate",
                        "vital",
                        "observed_context",
                        True,
                        False,
                    ),
                )
            )
            support = train_stays
            edge_rows.extend(
                (
                    (
                        "mimiciv",
                        split,
                        group_id,
                        0,
                        1,
                        f"condition|{INDEX_CONDITION}",
                        f"medication|{CANDIDATE_TOKENS[0]}",
                        "condition",
                        "medication",
                        "condition_medication_train_positive",
                        support,
                    ),
                    (
                        "mimiciv",
                        split,
                        group_id,
                        0,
                        5,
                        f"condition|{INDEX_CONDITION}",
                        "lab|lactate",
                        "condition",
                        "lab",
                        "condition_lab_predecision",
                        support,
                    ),
                    (
                        "mimiciv",
                        split,
                        group_id,
                        0,
                        6,
                        f"condition|{INDEX_CONDITION}",
                        "vital|heart_rate",
                        "condition",
                        "vital",
                        "condition_vital_predecision",
                        support,
                    ),
                )
            )

    write_parquet_rows(
        config.subgraph_index_path,
        (
            "source",
            "split",
            "subgraph_id",
            "index_condition_token",
            "node_count",
            "edge_count",
            "candidate_count",
            "positive_count",
        ),
        tuple(index_rows),
    )
    write_parquet_rows(
        config.subgraph_nodes_path,
        (
            "source",
            "split",
            "subgraph_id",
            "node_index",
            "node_id",
            "node_type",
            "node_role",
            "observed_predecision",
            "cold_start",
        ),
        tuple(node_rows),
    )
    write_parquet_rows(
        config.subgraph_edges_path,
        (
            "source",
            "split",
            "subgraph_id",
            "src_node_index",
            "dst_node_index",
            "src_id",
            "dst_id",
            "src_type",
            "dst_type",
            "relation_type",
            "support_count",
        ),
        tuple(edge_rows),
    )
    write_parquet_rows(
        config.subgraph_candidates_path,
        (
            "source",
            "split",
            "subgraph_id",
            "index_condition_token",
            "candidate_medication_token",
            "candidate_node_index",
            "candidate_rank",
            "label_prescribed",
            "cold_start",
        ),
        tuple(candidate_rows),
    )


def _fast_configs(
    tmp_path: Path,
) -> tuple[object, GNNTrainingConfig]:
    train_stays = 8
    validation_stays = 4
    dataset_root = tmp_path / "synthetic_dataset"
    phase_root = dataset_root / "processed" / "phase8_p0"
    reports_root = phase_root / "reports"
    neural = write_neural_fixture(
        phase_root,
        train_stays=train_stays,
        validation_stays=validation_stays,
        require_neural_gate=False,
        gate_authorized=False,
        shard_count=2,
    )
    neural = replace(
        neural,
        device="cpu",
        architecture=NeuralArchitecture(
            event_embedding_dim=16,
            encoder_layers=1,
            attention_heads=2,
            feedforward_dim=32,
            dropout=0.0,
            feature_dropout=0.0,
            categorical_embedding_dim=4,
            condition_embedding_dim=8,
            candidate_embedding_dim=8,
            context_hidden_dim=16,
            scorer_hidden_dim=16,
            candidate_side_hidden_dim=8,
        ),
        optimization=replace(
            NeuralOptimization(),
            max_epochs=1,
            early_stopping_patience=1,
            batch_ranking_groups=4,
            mixed_precision=False,
        ),
    )
    graph_root = phase_root / "graph" / "milestone8"
    gnn = GNNTrainingConfig(
        dataset_root=dataset_root,
        reports_root=reports_root,
        graph_root=graph_root,
        subgraphs_root=graph_root / "patient_subgraphs",
        features_root=neural.features_root,
        training_root=neural.training_root,
        neural_root=neural.neural_root,
        gnn_root=phase_root / "gnn",
        graph_reference_scores_path=(
            phase_root / "evaluation" / "milestone8b" / "graph_ablation_scores.parquet"
        ),
        graph_reference_report_path=reports_root / "graph_reference.json",
        contract_lock_path=neural.contract_lock_path,
        subgraphs_manifest_path=reports_root / "subgraphs.json",
        neural_selection_path=neural.selection_report_path,
        prepare_manifest_path=reports_root / "gnn_prepare.json",
        crossfit_graph_manifest_path=reports_root / "gnn_crossfit.json",
        gnn_training_report_path=reports_root / "gnn_training.json",
        gnn_score_report_path=reports_root / "gnn_score.json",
        gnn_selection_report_path=reports_root / "gnn_selection.json",
        fusion_training_report_path=reports_root / "fusion_training.json",
        fusion_score_report_path=reports_root / "fusion_score.json",
        fusion_selection_report_path=reports_root / "fusion_selection.json",
        mode="development",
        allow_ungated=True,
        device="cpu",
        fold_count=2,
        shard_count=2,
        architecture=GNNArchitecture(
            concept_embedding_dim=8,
            node_type_embedding_dim=4,
            node_role_embedding_dim=4,
            hidden_dim=8,
            relation_layers=1,
            dropout=0.0,
            scorer_hidden_dim=8,
            transformer_context_dim=16,
            fusion_hidden_dim=8,
        ),
        optimization=GNNOptimization(
            max_epochs=1,
            early_stopping_patience=1,
            batch_ranking_groups=4,
            mixed_precision=False,
        ),
        duckdb_temp_directory=None,
        duckdb_memory_limit=None,
        duckdb_threads=None,
    )
    _write_patient_subgraphs(
        gnn,
        train_stays=train_stays,
        validation_stays=validation_stays,
    )
    write_json(
        gnn.graph_reference_report_path,
        {
            "status": "completed",
            "ranking_metrics": [
                {
                    "source": "mimiciv",
                    "split": "validation",
                    "baseline_name": GRAPH_REFERENCE_BASELINE_NAME,
                    "k": 10,
                    "ndcg_at_k": 0.0,
                    "mrr_at_k": 0.0,
                    "hit_rate_at_k": 0.0,
                    "positive_ranking_group_count": 3,
                }
            ],
        },
    )
    return neural, gnn


def test_prepare_train_score_and_fuse_smoke(tmp_path: Path) -> None:
    neural, config = _fast_configs(tmp_path)
    assert prepare_neural_caches(neural)["status"] == "completed"
    assert train_transformer(neural)["status"] == "completed"

    preparation = prepare_gnn_caches(config)
    assert preparation["status"] == "completed", preparation
    assert preparation["components"] == {
        "graph_cache": "completed",
        "crossfit_graph_caches": "completed",
        "frozen_transformer_cache": "completed",
    }

    gnn_training = train_gnn(config)
    assert gnn_training["status"] == "completed"
    assert config.gnn_checkpoint_path.is_file()
    assert config.gnn_oof_predictions_path.is_file()
    assert crossfit_artifact_lock_errors(config) == []

    first_variant = ABLATION_VARIANTS[0]
    assert config.fold_resume_path(0, first_variant).is_file()
    assert config.fold_completion_manifest_path(0, first_variant).is_file()
    config.fold_checkpoint_path(0, first_variant).unlink()
    resumed_gnn_training = train_gnn(config)
    assert resumed_gnn_training["status"] == "completed"
    assert resumed_gnn_training["fold_results"] == gnn_training["fold_results"]

    selection_only = select_gnn(config)
    assert selection_only["status"] == "completed"
    assert config.gnn_crossfit_selection_path.is_file()
    refit_only = refit_selected_gnn(config)
    assert refit_only["status"] == "completed"

    gnn_scoring = score_gnn(config)
    assert gnn_scoring["status"] == "completed"
    assert config.gnn_selection_report_path.is_file()

    fusion_training = train_fusion(config)
    assert fusion_training["status"] == "completed"
    assert config.fusion_checkpoint_path.is_file()

    fusion_scoring = score_fusion(config)
    assert fusion_scoring["status"] == "completed"
    assert config.fusion_selection_report_path.is_file()

    public_payload = json.dumps(
        {
            "preparation": preparation,
            "gnn_training": gnn_training,
            "gnn_selection_only": selection_only,
            "gnn_refit_only": refit_only,
            "gnn_scoring": gnn_scoring,
            "fusion_training": fusion_training,
            "fusion_scoring": fusion_scoring,
        },
        sort_keys=True,
    )
    assert "mimiciv:patient-" not in public_payload
    assert "mimiciv:stay-" not in public_payload
    assert "grp-train-" not in public_payload
