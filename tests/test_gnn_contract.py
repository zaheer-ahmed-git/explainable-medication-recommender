"""Synthetic tests for the torch-free GNN/fusion contract foundation."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from pipeline.gnn_training.__main__ import build_config, parse_args
from pipeline.gnn_training.config import (
    CROSS_FIT_SCHEMA_VERSION,
    DEFAULT_FOLD_COUNT,
    DEFAULT_SHARD_COUNT,
    FORWARD_RELATION_TYPES,
    PRIMARY_SEED,
    RELATION_TO_INDEX,
    RELATION_TYPES,
    REVERSE_RELATION_TYPES,
    SELF_LOOP_RELATION,
    GNNTrainingConfig,
)
from pipeline.gnn_training.contract import (
    blocked_report,
    preflight_errors,
)
from pipeline.gnn_training.data import (
    GRAPH_CACHE_ARTIFACT_LOCK_VERSION,
    artifact_tree_digest,
    graph_cache_artifact_hashes,
)
from pipeline.training_contract import PINNED_VERSIONS, sha256_file

CONTRACT_DIGEST = "c" * 64


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_artifact(path: Path, content: bytes = b"synthetic-artifact") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _synthetic_config(root: Path, *, allow_ungated: bool = False) -> GNNTrainingConfig:
    dataset_root = root / "dataset"
    reports_root = root / "reports"
    phase_root = dataset_root / "processed" / "phase8_p0"
    graph_root = phase_root / "graph" / "milestone8"
    subgraphs_root = graph_root / "patient_subgraphs"
    training_root = phase_root / "training"
    neural_root = phase_root / "neural"
    gnn_root = phase_root / "gnn"
    return GNNTrainingConfig(
        dataset_root=dataset_root,
        reports_root=reports_root,
        graph_root=graph_root,
        subgraphs_root=subgraphs_root,
        features_root=phase_root / "features",
        training_root=training_root,
        neural_root=neural_root,
        gnn_root=gnn_root,
        graph_reference_scores_path=(
            phase_root / "evaluation" / "milestone8b" / "graph_ablation_scores.parquet"
        ),
        graph_reference_report_path=reports_root / "graph_reference.json",
        contract_lock_path=reports_root / "training_contract_lock.json",
        subgraphs_manifest_path=reports_root / "patient_subgraphs_manifest.json",
        neural_selection_path=reports_root / "neural_selection.json",
        prepare_manifest_path=reports_root / "gnn_prepare_manifest.json",
        crossfit_graph_manifest_path=reports_root / "crossfit_graph_manifest.json",
        gnn_training_report_path=reports_root / "gnn_training.json",
        gnn_score_report_path=reports_root / "gnn_score.json",
        gnn_selection_report_path=reports_root / "gnn_selection.json",
        fusion_training_report_path=reports_root / "fusion_training.json",
        fusion_score_report_path=reports_root / "fusion_score.json",
        fusion_selection_report_path=reports_root / "fusion_selection.json",
        allow_ungated=allow_ungated,
    )


def _write_upstream_contract(config: GNNTrainingConfig) -> None:
    for path in (
        config.subgraph_index_path,
        config.subgraph_nodes_path,
        config.subgraph_edges_path,
        config.subgraph_candidates_path,
        config.patient_condition_medication_path,
        config.event_sequences_path,
    ):
        _write_artifact(path)

    _write_artifact(config.graph_reference_scores_path)

    _write_json(
        config.subgraphs_manifest_path,
        {
            "status": "completed",
            "versions": dict(PINNED_VERSIONS),
            "graph_fit_scope": [
                {
                    "fit_source": "mimiciv",
                    "fit_split": "train",
                    "edge_count": 7,
                }
            ],
            "data_safety": {
                "manifest_contains_patient_rows": False,
                "manifest_contains_row_samples": False,
            },
        },
    )
    _write_json(
        config.contract_lock_path,
        {
            "status": "completed",
            "versions": dict(PINNED_VERSIONS),
            "contract_digest": CONTRACT_DIGEST,
            "contract": {
                "manifests": {
                    "patient_subgraphs_manifest": {
                        "path": str(config.subgraphs_manifest_path),
                        "sha256": sha256_file(config.subgraphs_manifest_path),
                    }
                }
            },
        },
    )

    for path in (
        config.neural_checkpoint_path,
        config.neural_calibration_path,
        config.neural_feature_layout_path,
    ):
        _write_artifact(path)
    _write_json(
        config.neural_selection_path,
        {
            "status": "frozen",
            "model_frozen": True,
            "contract_digest": CONTRACT_DIGEST,
            "frozen_artifacts": {
                "checkpoint": {
                    "path": str(config.neural_checkpoint_path),
                    "sha256": sha256_file(config.neural_checkpoint_path),
                },
                "calibration": {
                    "path": str(config.neural_calibration_path),
                    "sha256": sha256_file(config.neural_calibration_path),
                },
                "feature_layout": {
                    "path": str(config.neural_feature_layout_path),
                    "sha256": sha256_file(config.neural_feature_layout_path),
                },
            },
        },
    )
    _write_json(config.graph_reference_report_path, {"status": "completed"})


def _write_prepared_graph(config: GNNTrainingConfig) -> None:
    for path in (
        config.feature_layout_path,
        config.graph_node_vocabulary_path,
        config.node_type_vocabulary_path,
        config.node_role_vocabulary_path,
    ):
        _write_artifact(path)
    _write_json(
        config.relation_vocabulary_path,
        {
            "schema_version": CROSS_FIT_SCHEMA_VERSION,
            "status": "completed",
            "relations": list(RELATION_TYPES),
            "relation_to_index": RELATION_TO_INDEX,
        },
    )
    config.shards_root.mkdir(parents=True, exist_ok=True)
    _write_artifact(config.shards_root / "shard_0000.parquet")
    artifact_hashes = graph_cache_artifact_hashes(config.gnn_root)
    _write_json(
        config.cache_manifest_path,
        {
            "status": "completed",
            "artifact_lock_version": GRAPH_CACHE_ARTIFACT_LOCK_VERSION,
            "shard_count": config.shard_count,
            "scope": "full_train_refit_only",
            "selection_eligible": False,
            "cached_splits": ["test", "train", "validation"],
            "artifact_hashes": artifact_hashes,
            "artifact_tree_digest": artifact_tree_digest(artifact_hashes),
        },
    )
    _write_json(
        config.prepare_manifest_path,
        {
            "status": "pending_required_component",
            "scope": "full_train_refit_only",
            "selection_eligible": False,
            "preparation_complete": False,
            "components": {
                "graph_cache": "completed",
                "frozen_transformer_cache": "pending",
            },
            "data_safety": {
                "report_contains_patient_rows": False,
                "report_contains_row_samples": False,
            },
        },
    )


def _write_crossfit_contract(config: GNNTrainingConfig) -> None:
    folds = []
    all_indices = set(range(config.fold_count))
    for fold_index in range(config.fold_count):
        folds.append(
            {
                "fold_index": fold_index,
                "held_out_fold_index": fold_index,
                "fit_fold_indices": sorted(all_indices - {fold_index}),
                "fit_source": "mimiciv",
                "fit_split": "train",
                "fit_patient_count": 80,
                "held_out_patient_count": 20,
                "patient_overlap_count": 0,
                "graph_fit_excludes_held_out_fold": True,
                "vocabulary_fit_excludes_held_out_fold": True,
                "support_fit_excludes_held_out_fold": True,
            }
        )
    _write_json(
        config.crossfit_graph_manifest_path,
        {
            "schema_version": CROSS_FIT_SCHEMA_VERSION,
            "status": "completed",
            "contract_digest": CONTRACT_DIGEST,
            "seed": config.seed,
            "fold_count": config.fold_count,
            "patient_grouped": True,
            "fit_scope": {
                "source": "mimiciv",
                "split": "train",
                "grouping_unit": "patient_uid",
            },
            "folds": folds,
            "data_safety": {
                "report_contains_patient_rows": False,
                "report_contains_identifier_values": False,
            },
        },
    )


def _write_gnn_checkpoint_artifacts(config: GNNTrainingConfig) -> None:
    for path in (
        config.gnn_checkpoint_path,
        config.gnn_calibration_path,
        config.gnn_training_state_path,
        config.gnn_oof_predictions_path,
    ):
        _write_artifact(path)


def _write_gnn_selection(config: GNNTrainingConfig) -> None:
    _write_json(
        config.gnn_selection_report_path,
        {
            "status": "frozen",
            "model_frozen": True,
            "standalone_qualified": True,
            "contract_digest": CONTRACT_DIGEST,
            "frozen_artifacts": {
                name: {"path": str(path), "sha256": sha256_file(path)}
                for name, path in {
                    "checkpoint": config.gnn_checkpoint_path,
                    "calibration": config.gnn_calibration_path,
                    "feature_layout": config.feature_layout_path,
                    "training_state": config.gnn_training_state_path,
                    "oof_predictions": config.gnn_oof_predictions_path,
                    "crossfit_manifest": config.crossfit_graph_manifest_path,
                    "cache_manifest": config.cache_manifest_path,
                    "graph_reference_report": config.graph_reference_report_path,
                    "graph_reference_scores": config.graph_reference_scores_path,
                }.items()
            },
        },
    )


def _complete_transformer_cache(config: GNNTrainingConfig) -> None:
    config.frozen_transformer_cache_root.mkdir(parents=True, exist_ok=True)
    _write_artifact(config.frozen_transformer_cache_root / "shard_0000.parquet")
    _write_json(
        config.transformer_cache_manifest_path,
        {"status": "completed", "shard_count": config.shard_count},
    )
    prepare = json.loads(config.prepare_manifest_path.read_text(encoding="utf-8"))
    prepare["status"] = "completed"
    prepare["preparation_complete"] = True
    prepare["components"]["frozen_transformer_cache"] = "completed"
    _write_json(config.prepare_manifest_path, prepare)


def _write_fusion_checkpoint_artifacts(config: GNNTrainingConfig) -> None:
    for path in (
        config.fusion_checkpoint_path,
        config.fusion_calibration_path,
        config.fusion_training_state_path,
    ):
        _write_artifact(path)


def _write_fusion_selection(config: GNNTrainingConfig) -> None:
    _write_json(
        config.fusion_selection_report_path,
        {
            "status": "frozen",
            "model_frozen": True,
            "hybrid_qualified": True,
            "contract_digest": CONTRACT_DIGEST,
            "frozen_artifacts": {
                name: {"path": str(path), "sha256": sha256_file(path)}
                for name, path in {
                    "checkpoint": config.fusion_checkpoint_path,
                    "calibration": config.fusion_calibration_path,
                    "feature_layout": config.feature_layout_path,
                    "training_state": config.fusion_training_state_path,
                    "gnn_checkpoint": config.gnn_checkpoint_path,
                    "gnn_selection": config.gnn_selection_report_path,
                    "transformer_checkpoint": config.neural_checkpoint_path,
                    "transformer_calibration": config.neural_calibration_path,
                    "transformer_feature_layout": (config.neural_feature_layout_path),
                    "transformer_cache_manifest": (
                        config.transformer_cache_manifest_path
                    ),
                    "crossfit_manifest": config.crossfit_graph_manifest_path,
                }.items()
            },
        },
    )


def test_defaults_and_relation_vocabulary_are_stable() -> None:
    config = GNNTrainingConfig()

    assert config.shard_count == DEFAULT_SHARD_COUNT == 256
    assert config.fold_count == DEFAULT_FOLD_COUNT == 5
    assert config.seed == PRIMARY_SEED == 20260617
    assert config.architecture.relation_layers == 2
    assert config.architecture.hidden_dim == 128
    assert config.architecture.dropout == 0.2
    assert config.optimization.optimizer_name == "AdamW"
    assert config.optimization.learning_rate == 3e-4
    assert config.optimization.weight_decay == 1e-4
    assert config.optimization.gradient_clip_norm == 1.0
    assert config.optimization.max_epochs == 30
    assert config.optimization.early_stopping_patience == 3
    assert config.optimization.mixed_precision is True
    assert len(FORWARD_RELATION_TYPES) == 5
    assert REVERSE_RELATION_TYPES == tuple(
        f"reverse_{relation}" for relation in FORWARD_RELATION_TYPES
    )
    assert RELATION_TYPES[-1] == SELF_LOOP_RELATION == "self_loop"
    assert len(RELATION_TO_INDEX) == 11


def test_cli_builds_config_without_importing_stage_modules() -> None:
    args = parse_args(
        [
            "train-gnn",
            "--shard-count",
            "32",
            "--fold-count",
            "5",
            "--learning-rate",
            "0.0003",
            "--allow-ungated",
        ]
    )

    config = build_config(args)

    assert config.shard_count == 32
    assert config.fold_count == 5
    assert config.optimization.learning_rate == 3e-4
    assert config.allow_ungated is True


def test_cli_can_disable_mixed_precision_for_numerical_fallback() -> None:
    args = parse_args(["train-gnn", "--no-mixed-precision"])

    config = build_config(args)

    assert config.optimization.mixed_precision is False


def test_prepare_accepts_exact_upstream_locks(tmp_path: Path) -> None:
    config = _synthetic_config(tmp_path)
    _write_upstream_contract(config)

    assert preflight_errors(config, stage="prepare") == []


def test_frozen_transformer_artifact_drift_and_path_change_are_blocked(
    tmp_path: Path,
) -> None:
    config = _synthetic_config(tmp_path)
    _write_upstream_contract(config)
    config.neural_checkpoint_path.write_bytes(b"changed")

    errors = preflight_errors(config, stage="prepare")

    assert "frozen_neural_artifact_changed" in {error["code"] for error in errors}

    alternate = config.neural_root / "checkpoints" / "alternate.pt"
    _write_artifact(alternate)
    selection = json.loads(config.neural_selection_path.read_text(encoding="utf-8"))
    selection["frozen_artifacts"]["checkpoint"] = {
        "path": str(alternate),
        "sha256": sha256_file(alternate),
    }
    _write_json(config.neural_selection_path, selection)

    errors = preflight_errors(config, stage="prepare")
    assert "frozen_neural_artifact_path_mismatch" in {error["code"] for error in errors}


def test_unsafe_write_root_and_ungated_protected_root_are_blocked(
    tmp_path: Path,
) -> None:
    config = _synthetic_config(tmp_path, allow_ungated=True)
    unsafe = replace(config, gnn_root=config.neural_root)

    errors = preflight_errors(unsafe, stage="prepare")

    codes = {error["code"] for error in errors}
    assert "unsafe_gnn_root" in codes
    assert "gnn_neural_root_overlap" in codes

    protected = _synthetic_config(tmp_path / "protected", allow_ungated=True)
    errors = preflight_errors(protected, stage="prepare")
    assert "ungated_production_path" in {error["code"] for error in errors}


def test_both_training_stages_require_crossfit_contract_not_full_train_cache(
    tmp_path: Path,
) -> None:
    config = _synthetic_config(tmp_path)
    _write_upstream_contract(config)
    _write_prepared_graph(config)

    for stage in ("train-gnn", "train-fusion"):
        errors = preflight_errors(config, stage=stage)
        assert "missing_crossfit_graph_contract" in {error["code"] for error in errors}


def test_crossfit_scope_requires_every_held_out_patient_fold_excluded(
    tmp_path: Path,
) -> None:
    config = _synthetic_config(tmp_path)
    _write_upstream_contract(config)
    _write_prepared_graph(config)
    _write_crossfit_contract(config)
    payload = json.loads(
        config.crossfit_graph_manifest_path.read_text(encoding="utf-8")
    )
    payload["folds"][2]["support_fit_excludes_held_out_fold"] = False
    _write_json(config.crossfit_graph_manifest_path, payload)

    errors = preflight_errors(config, stage="train-gnn")

    matching = [
        error for error in errors if error["code"] == "crossfit_heldout_not_excluded"
    ]
    assert matching == [
        {
            "code": "crossfit_heldout_not_excluded",
            "detail": (
                "held-out patient fold must be excluded from graph, "
                "vocabulary, and support fitting"
            ),
            "artifact_name": "support",
            "fold_index": 2,
        }
    ]


def test_final_gnn_scoring_requires_flag_selection_and_exact_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(_synthetic_config(tmp_path), mode="final")
    _write_upstream_contract(config)
    _write_prepared_graph(config)
    _write_crossfit_contract(config)
    monkeypatch.setattr(
        "pipeline.gnn_training.crossfit.crossfit_artifact_lock_errors",
        lambda _config: [],
    )
    _write_gnn_checkpoint_artifacts(config)

    errors = preflight_errors(config, stage="score-gnn")

    codes = {error["code"] for error in errors}
    assert "final_requires_frozen_selection" in codes
    assert "missing_gnn_selection" in codes

    _write_gnn_selection(config)
    config = replace(config, frozen_selection=True)
    assert preflight_errors(config, stage="score-gnn") == []

    config.gnn_checkpoint_path.write_bytes(b"drift")
    errors = preflight_errors(config, stage="score-gnn")
    assert "frozen_gnn_artifact_changed" in {error["code"] for error in errors}


def test_final_fusion_scoring_requires_hybrid_selection_and_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(_synthetic_config(tmp_path), mode="final")
    _write_upstream_contract(config)
    _write_prepared_graph(config)
    _complete_transformer_cache(config)
    _write_crossfit_contract(config)
    monkeypatch.setattr(
        "pipeline.gnn_training.crossfit.crossfit_artifact_lock_errors",
        lambda _config: [],
    )
    monkeypatch.setattr(
        "pipeline.gnn_training.data._transformer_cache_status",
        lambda _config: "completed",
    )
    _write_gnn_checkpoint_artifacts(config)
    _write_gnn_selection(config)
    _write_fusion_checkpoint_artifacts(config)

    errors = preflight_errors(config, stage="score-fusion")
    codes = {error["code"] for error in errors}
    assert "final_requires_frozen_selection" in codes
    assert "missing_hybrid_selection" in codes

    _write_fusion_selection(config)
    config = replace(config, frozen_selection=True)
    assert preflight_errors(config, stage="score-fusion") == []

    config.fusion_calibration_path.write_bytes(b"drift")
    errors = preflight_errors(config, stage="score-fusion")
    assert "frozen_hybrid_artifact_changed" in {error["code"] for error in errors}


def test_allow_ungated_is_explicit_and_reports_no_identifier_rows(
    tmp_path: Path,
) -> None:
    config = _synthetic_config(tmp_path, allow_ungated=True)

    assert preflight_errors(config, stage="prepare") == []
    report = blocked_report(
        config=config,
        schema_version="synthetic-v1",
        stage="prepare",
        generated_at="2026-01-01T00:00:00+00:00",
        errors=[],
    )

    assert report["gate_policy"] == {
        "upstream_artifact_gates_enforced": False,
        "allow_ungated": True,
        "allow_ungated_scope": "synthetic_unit_tests_only",
    }
    assert report["data_safety"]["report_contains_identifier_values"] is False
