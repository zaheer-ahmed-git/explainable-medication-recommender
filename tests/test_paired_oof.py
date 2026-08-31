"""Synthetic safety and selection tests for paired-OOF late fusion."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import duckdb
import pytest

from pipeline.extract_utils import parquet_scan
from pipeline.gnn_training.config import GNNTrainingConfig
from pipeline.gnn_training.model import ABLATION_VARIANTS
from pipeline.gnn_training.paired_oof import select_paired_oof_late_fusion
from pipeline.gnn_training.runtime import resolve_device as resolve_gnn_device
from pipeline.late_fusion_protocol import (
    FROZEN_GATE_SCHEMA_VERSION,
    PAIRED_OOF_PROTOCOL_VERSION,
    alpha_grid,
    transformer_oof_predictions_path,
)
from pipeline.neural_training.config import NeuralOptimization
from pipeline.neural_training.train import resolve_device as resolve_neural_device
from pipeline.training_contract import sha256_file, write_json
from tests.milestone6_helpers import write_parquet_rows
from tests.neural_training_helpers import write_neural_fixture


def test_alpha_grid_is_exact_and_pre_registered() -> None:
    values = alpha_grid()
    assert len(values) == 51
    assert values[0] == 0.0
    assert values[-1] == 0.25
    assert values[1] - values[0] == pytest.approx(0.005)


def test_explicit_cuda_device_fails_closed_when_cuda_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch")
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    gnn_config = replace(_selection_config(tmp_path), device="cuda")
    neural_config = replace(
        write_neural_fixture(
            tmp_path / "neural",
            train_stays=4,
            validation_stays=2,
            require_neural_gate=False,
            shard_count=1,
        ),
        device="cuda",
    )
    for resolver, config in (
        (resolve_gnn_device, gnn_config),
        (resolve_neural_device, neural_config),
    ):
        with pytest.raises(RuntimeError, match="CUDA.*unavailable"):
            resolver(config)


@pytest.mark.filterwarnings("ignore:.*GradScaler.*:FutureWarning")
def test_transformer_oof_fold_is_fit_isolated(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from pipeline.neural_training.oof import run_transformer_oof_fold

    config = write_neural_fixture(
        tmp_path,
        train_stays=10,
        validation_stays=2,
        require_neural_gate=False,
        shard_count=2,
    )
    config = replace(
        config,
        device="cpu",
        optimization=replace(
            NeuralOptimization(),
            max_epochs=1,
            batch_ranking_groups=4,
            mixed_precision=False,
        ),
    )
    report = run_transformer_oof_fold(
        config,
        gnn_root=tmp_path / "gnn",
        held_out_fold=0,
        fold_count=2,
        fixed_epochs=1,
    )
    assert report["status"] == "completed", report.get("reason")
    assert report["selection_policy"] == "fixed_epoch_no_heldout_selection"
    assert report["leakage_policy"]["held_out_used_for_early_stopping"] is False
    assert report["input_aggregate_counts"]["fit_patient_count"] > 0
    assert report["input_aggregate_counts"]["held_out_patient_count"] > 0

    output = transformer_oof_predictions_path(config.neural_root, 0)
    with duckdb.connect(database=":memory:") as connection:
        row = connection.execute(
            f"""
SELECT
    COUNT(*),
    COUNT(DISTINCT patient_fold_id),
    MIN(patient_fold_id),
    COUNT(*) FILTER (WHERE split <> 'train' OR source <> 'mimiciv')
FROM {parquet_scan(output)}
"""
        ).fetchone()
    assert row is not None
    assert int(row[0]) > 0
    assert tuple(int(value) for value in row[1:]) == (1, 0, 0)


def _selection_config(tmp_path: Path) -> GNNTrainingConfig:
    dataset_root = tmp_path / "dataset"
    phase_root = dataset_root / "processed" / "phase8_p0"
    reports_root = tmp_path / "reports"
    return GNNTrainingConfig(
        dataset_root=dataset_root,
        reports_root=reports_root,
        gnn_root=phase_root / "gnn",
        neural_root=phase_root / "neural",
        graph_root=phase_root / "graph" / "milestone8",
        subgraphs_root=phase_root / "graph" / "milestone8" / "patient_subgraphs",
        features_root=phase_root / "features",
        training_root=phase_root / "training",
        graph_reference_scores_path=(
            phase_root / "evaluation" / "milestone8b" / "graph_ablation_scores.parquet"
        ),
        graph_reference_report_path=reports_root / "graph_reference.json",
        contract_lock_path=reports_root / "contract.json",
        subgraphs_manifest_path=reports_root / "subgraphs.json",
        neural_selection_path=reports_root / "neural_selection.json",
        prepare_manifest_path=reports_root / "prepare.json",
        crossfit_graph_manifest_path=reports_root / "crossfit.json",
        gnn_training_report_path=reports_root / "gnn_training.json",
        gnn_score_report_path=reports_root / "gnn_score.json",
        gnn_selection_report_path=reports_root / "gnn_selection.json",
        fusion_training_report_path=reports_root / "fusion_training.json",
        fusion_score_report_path=reports_root / "fusion_score.json",
        fusion_selection_report_path=reports_root / "fusion_selection.json",
        allow_ungated=True,
        fold_count=2,
        shard_count=2,
        duckdb_temp_directory=None,
        duckdb_memory_limit=None,
        duckdb_threads=None,
    )


_OOF_COLUMNS = (
    "source",
    "split",
    "ranking_group_id",
    "index_condition_token",
    "candidate_medication_token",
    "candidate_rank",
    "label_prescribed",
    "patient_fold_id",
    "transformer_logit",
)


def _prediction_rows(*, gnn: bool, strong: bool) -> list[tuple[object, ...]]:
    rows: list[tuple[object, ...]] = []
    for group_index in range(4):
        fold = group_index % 2
        transformer_scores = (0.1, 0.2, -10.0) if group_index >= 2 else (1.0, 0.0, -1.0)
        if strong:
            gnn_scores = (2.0, -1.0, -2.0)
        else:
            gnn_scores = (-1.0, 1.0, 0.0)
        scores = gnn_scores if gnn else transformer_scores
        for rank, score in enumerate(scores, start=1):
            rows.append(
                (
                    "mimiciv",
                    "train",
                    f"group-{group_index}",
                    "condition-a",
                    f"med-{rank}",
                    rank,
                    rank == 1,
                    fold,
                    score,
                )
            )
    return rows


def _write_selection_artifacts(config: GNNTrainingConfig) -> None:
    for path in (
        config.neural_checkpoint_path,
        config.transformer_cache_manifest_path,
        config.cache_manifest_path,
    ):
        write_json(path, {"status": "synthetic"})
    write_json(config.neural_training_state_path, {"best_epoch": 0})
    transformer_rows = _prediction_rows(gnn=False, strong=False)
    for fold_index in range(config.fold_count):
        path = transformer_oof_predictions_path(config.neural_root, fold_index)
        fold_rows = tuple(row for row in transformer_rows if row[7] == fold_index)
        write_parquet_rows(path, _OOF_COLUMNS, fold_rows)
        write_json(
            config.reports_root
            / f"phase8_p0_transformer_paired_oof_fold_{fold_index:02d}_training.json",
            {
                "status": "completed",
                "protocol_version": PAIRED_OOF_PROTOCOL_VERSION,
                "held_out_fold_index": fold_index,
                "fold_count": config.fold_count,
                "fit_fold_indices": [
                    index for index in range(config.fold_count) if index != fold_index
                ],
                "fixed_epochs": 1,
                "selection_policy": "fixed_epoch_no_heldout_selection",
                "artifacts": {"oof_predictions_sha256": sha256_file(path)},
            },
        )
    for variant in ABLATION_VARIANTS:
        path = config.variant_oof_predictions_path(variant)
        rows = _prediction_rows(gnn=True, strong=variant == "full")
        gnn_columns = (*_OOF_COLUMNS[:-1], "gnn_logit")
        write_parquet_rows(path, gnn_columns, tuple(rows))
        write_json(
            config.variant_oof_report_path(variant),
            {
                "status": "completed",
                "protocol_version": PAIRED_OOF_PROTOCOL_VERSION,
                "ablation_variant": variant,
                "fold_count": config.fold_count,
                "artifacts": {"oof_predictions_sha256": sha256_file(path)},
            },
        )
        for fold_index in range(config.fold_count):
            write_json(
                config.fold_completion_manifest_path(fold_index, variant),
                {
                    "status": "completed",
                    "ablation_variant": variant,
                    "held_out_fold_index": fold_index,
                    "best_epoch": 0,
                },
            )


def test_joint_selection_uses_only_exact_paired_oof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _selection_config(tmp_path)
    _write_selection_artifacts(config)
    monkeypatch.setattr(
        "pipeline.gnn_training.paired_oof.preflight_errors",
        lambda _config, stage: [],
    )
    report = select_paired_oof_late_fusion(config)
    assert report["status"] == "completed", report.get("reason")
    assert report["selected_gnn_variant"] == "full"
    assert report["selected_alpha"] in alpha_grid()
    assert report["selected_alpha"] > 0.0
    assert report["leakage_policy"]["validation_gate_used_for_selection"] is False
    assert config.paired_late_checkpoint_path.is_file()


def test_joint_selection_fails_closed_on_label_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _selection_config(tmp_path)
    _write_selection_artifacts(config)
    path = config.variant_oof_predictions_path("rank_only")
    rows = _prediction_rows(gnn=True, strong=False)
    first = list(rows[0])
    first[6] = False
    rows[0] = tuple(first)
    write_parquet_rows(path, (*_OOF_COLUMNS[:-1], "gnn_logit"), tuple(rows))
    write_json(
        config.variant_oof_report_path("rank_only"),
        {
            "status": "completed",
            "protocol_version": PAIRED_OOF_PROTOCOL_VERSION,
            "ablation_variant": "rank_only",
            "fold_count": config.fold_count,
            "artifacts": {"oof_predictions_sha256": sha256_file(path)},
        },
    )
    monkeypatch.setattr(
        "pipeline.gnn_training.paired_oof.preflight_errors",
        lambda _config, stage: [],
    )
    report = select_paired_oof_late_fusion(config)
    assert report["status"] == "failed"
    assert "not exactly fold- and label-aligned" in report["reason"]


def test_frozen_gate_contract_is_explicit_and_fail_closed(tmp_path: Path) -> None:
    from pipeline.gnn_training.paired_late import _validate_frozen_gate

    config = replace(
        _selection_config(tmp_path),
        paired_frozen_gate_manifest_path=(
            tmp_path / "reports" / "phase8_p0_paired_oof_frozen_gate.json"
        ),
    )
    write_json(config.paired_oof_selection_path, {"status": "synthetic-frozen"})
    write_json(config.cache_manifest_path, {"status": "synthetic"})
    write_json(config.transformer_cache_manifest_path, {"status": "synthetic"})
    selection_hash = sha256_file(config.paired_oof_selection_path)
    checkpoint = {
        "paired_oof_selection_sha256": selection_hash,
        "generated_at": "2026-08-27T10:00:00+00:00",
    }
    gate = {
        "schema_version": FROZEN_GATE_SCHEMA_VERSION,
        "protocol_version": PAIRED_OOF_PROTOCOL_VERSION,
        "status": "frozen",
        "gate_id": "synthetic-new-gate",
        "source": "mimiciv",
        "split": "synthetic_gate",
        "frozen_at": "2026-08-27T11:00:00+00:00",
        "selection_completed_before_gate_opened": True,
        "one_shot_scoring_authorized": True,
        "used_for_model_selection": False,
        "used_for_gnn_variant_selection": False,
        "used_for_alpha_selection": False,
        "previously_scored_by_hybrid": False,
        "patient_overlap_with_train_count": 0,
        "paired_oof_selection_sha256": selection_hash,
        "gnn_cache_manifest_sha256": sha256_file(config.cache_manifest_path),
        "transformer_cache_manifest_sha256": sha256_file(
            config.transformer_cache_manifest_path
        ),
    }
    write_json(config.paired_frozen_gate_manifest_path, gate)

    validated, split = _validate_frozen_gate(config, checkpoint)
    assert validated["gate_id"] == "synthetic-new-gate"
    assert split == "synthetic_gate"

    gate["used_for_alpha_selection"] = True
    write_json(config.paired_frozen_gate_manifest_path, gate)
    with pytest.raises(ValueError, match="violates the paired OOF protocol"):
        _validate_frozen_gate(config, checkpoint)
