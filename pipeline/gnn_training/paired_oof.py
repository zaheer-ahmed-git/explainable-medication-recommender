"""Joint GNN-variant and fine-grid alpha selection on paired OOF logits."""

from __future__ import annotations

from datetime import UTC, datetime
import statistics
from pathlib import Path
from time import perf_counter
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from pipeline.extract_utils import parquet_scan, safe_error_message
from pipeline.gnn_training.config import (
    MINIMUM_NDCG_LIFT,
    SELECTION_K,
    GNNTrainingConfig,
)
from pipeline.gnn_training.contract import (
    blocked_report,
    contract_digest_or_none,
    preflight_errors,
)
from pipeline.gnn_training.data import configure_connection, write_json
from pipeline.gnn_training.model import ABLATION_VARIANTS
from pipeline.late_fusion_protocol import (
    PAIRED_OOF_PROTOCOL_VERSION,
    PAIRED_OOF_SELECTION_SCHEMA_VERSION,
    PRODUCTION_FOLD_COUNT,
    alpha_grid,
    transformer_oof_predictions_path,
)
from pipeline.neural_training.metrics import RankingMetricAccumulator
from pipeline.training_contract import load_json, sha256_file


def _zscore(values: np.ndarray) -> np.ndarray:
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("paired OOF logits are empty or non-finite")
    scale = float(values.std())
    if scale < 1e-6:
        return np.zeros_like(values, dtype=np.float64)
    return (values.astype(np.float64) - float(values.mean())) / scale


def _completed_transformer_artifacts(
    config: GNNTrainingConfig,
) -> list[Path]:
    paths: list[Path] = []
    if not config.neural_training_state_path.is_file():
        raise FileNotFoundError("frozen Transformer training state is missing")
    neural_state = load_json(config.neural_training_state_path)
    best_epoch = neural_state.get("best_epoch")
    if not isinstance(best_epoch, int) or best_epoch < 0:
        raise ValueError("frozen Transformer training state has no valid best_epoch")
    expected_fixed_epochs = best_epoch + 1
    for fold_index in range(config.fold_count):
        report_path = (
            config.reports_root
            / f"phase8_p0_transformer_paired_oof_fold_{fold_index:02d}_training.json"
        )
        prediction_path = transformer_oof_predictions_path(
            config.neural_root,
            fold_index,
        )
        if not report_path.is_file() or not prediction_path.is_file():
            raise FileNotFoundError(
                f"completed Transformer OOF artifact is missing for fold-{fold_index}"
            )
        report = load_json(report_path)
        artifacts = report.get("artifacts")
        if (
            report.get("status") != "completed"
            or report.get("protocol_version") != PAIRED_OOF_PROTOCOL_VERSION
            or report.get("held_out_fold_index") != fold_index
            or report.get("fold_count") != config.fold_count
            or report.get("fixed_epochs") != expected_fixed_epochs
            or report.get("selection_policy") != "fixed_epoch_no_heldout_selection"
            or report.get("fit_fold_indices")
            != [index for index in range(config.fold_count) if index != fold_index]
            or not isinstance(artifacts, dict)
            or artifacts.get("oof_predictions_sha256") != sha256_file(prediction_path)
        ):
            raise ValueError(
                f"Transformer OOF completion lock is incompatible for fold-{fold_index}"
            )
        paths.append(prediction_path)
    return paths


def _completed_gnn_artifact(config: GNNTrainingConfig, variant: str) -> Path:
    report_path = config.variant_oof_report_path(variant)
    prediction_path = config.variant_oof_predictions_path(variant)
    if not report_path.is_file() or not prediction_path.is_file():
        raise FileNotFoundError(
            f"completed GNN OOF artifact is missing for variant {variant}"
        )
    report = load_json(report_path)
    artifacts = report.get("artifacts")
    if (
        report.get("status") != "completed"
        or report.get("protocol_version") != PAIRED_OOF_PROTOCOL_VERSION
        or report.get("ablation_variant") != variant
        or report.get("fold_count") != config.fold_count
        or not isinstance(artifacts, dict)
        or artifacts.get("oof_predictions_sha256") != sha256_file(prediction_path)
    ):
        raise ValueError(f"GNN OOF completion lock is incompatible for {variant}")
    return prediction_path


def _scan_paths(paths: list[Path]) -> str:
    escaped = ", ".join("'" + str(path).replace("'", "''") + "'" for path in paths)
    return f"read_parquet([{escaped}])"


def _validate_pairing(
    connection: duckdb.DuckDBPyConnection,
    *,
    transformer_scan: str,
    gnn_path: Path,
    fold_count: int,
) -> int:
    gnn_scan = parquet_scan(gnn_path)
    row = connection.execute(
        f"""
WITH transformer AS (
    SELECT
        source,
        split,
        ranking_group_id,
        index_condition_token,
        candidate_medication_token,
        CAST(candidate_rank AS BIGINT) AS candidate_rank,
        CAST(label_prescribed AS BOOLEAN) AS label_prescribed,
        CAST(patient_fold_id AS INTEGER) AS patient_fold_id
    FROM {transformer_scan}
),
gnn AS (
    SELECT
        source,
        split,
        ranking_group_id,
        index_condition_token,
        candidate_medication_token,
        CAST(candidate_rank AS BIGINT) AS candidate_rank,
        CAST(label_prescribed AS BOOLEAN) AS label_prescribed,
        CAST(patient_fold_id AS INTEGER) AS patient_fold_id
    FROM {gnn_scan}
),
transformer_only AS (
    SELECT * FROM transformer EXCEPT ALL SELECT * FROM gnn
),
gnn_only AS (
    SELECT * FROM gnn EXCEPT ALL SELECT * FROM transformer
),
transformer_duplicate AS (
    SELECT
        source, split, ranking_group_id, candidate_medication_token, COUNT(*) AS n
    FROM transformer
    GROUP BY source, split, ranking_group_id, candidate_medication_token
    HAVING n <> 1
),
gnn_duplicate AS (
    SELECT
        source, split, ranking_group_id, candidate_medication_token, COUNT(*) AS n
    FROM gnn
    GROUP BY source, split, ranking_group_id, candidate_medication_token
    HAVING n <> 1
)
SELECT
    (SELECT COUNT(*) FROM transformer) AS transformer_count,
    (SELECT COUNT(*) FROM gnn) AS gnn_count,
    (SELECT COUNT(*) FROM transformer_only)
        + (SELECT COUNT(*) FROM gnn_only) AS mismatch_count,
    (SELECT COUNT(*) FROM transformer_duplicate)
        + (SELECT COUNT(*) FROM gnn_duplicate) AS duplicate_count,
    (
        SELECT COUNT(*) FROM transformer
        WHERE source <> 'mimiciv'
            OR split <> 'train'
            OR patient_fold_id < 0
            OR patient_fold_id >= {fold_count}
    ) AS invalid_scope_count
"""
    ).fetchone()
    if (
        row is None
        or int(row[0]) <= 0
        or int(row[0]) != int(row[1])
        or int(row[2]) != 0
        or int(row[3]) != 0
        or int(row[4]) != 0
    ):
        raise ValueError(
            "Transformer/GNN OOF candidates are not exactly fold- and label-aligned"
        )
    return int(row[0])


def _evaluate_variant(
    connection: duckdb.DuckDBPyConnection,
    *,
    transformer_scan: str,
    gnn_path: Path,
) -> dict[str, Any]:
    weights = alpha_grid()
    accumulators = {
        weight: RankingMetricAccumulator(k=SELECTION_K) for weight in weights
    }
    transformer_accumulator = RankingMetricAccumulator(k=SELECTION_K)
    gnn_accumulator = RankingMetricAccumulator(k=SELECTION_K)
    joined_rows = 0

    def update_group(group: pd.DataFrame) -> None:
        nonlocal joined_rows
        labels = group["label_prescribed"].to_numpy(dtype=np.float32)
        ranks = group["candidate_rank"].to_numpy(dtype=np.float64)
        if len(set(ranks.tolist())) != len(group):
            raise ValueError("paired OOF ranking group has duplicate candidate ranks")
        transformer_values = group["transformer_logit"].to_numpy(dtype=np.float64)
        gnn_values = group["gnn_logit"].to_numpy(dtype=np.float64)
        transformer_z = _zscore(transformer_values)
        gnn_z = _zscore(gnn_values)
        transformer_accumulator.update(
            labels=labels,
            scores=transformer_values,
            tie_breaker=ranks,
        )
        gnn_accumulator.update(
            labels=labels,
            scores=gnn_values,
            tie_breaker=ranks,
        )
        for weight, accumulator in accumulators.items():
            accumulator.update(
                labels=labels,
                scores=(1.0 - weight) * transformer_z + weight * gnn_z,
                tie_breaker=ranks,
            )
        joined_rows += len(group)

    reader = connection.sql(
        f"""
SELECT
    transformer.ranking_group_id,
    transformer.candidate_rank,
    transformer.label_prescribed,
    transformer.transformer_logit,
    gnn.gnn_logit
FROM {transformer_scan} AS transformer
INNER JOIN {parquet_scan(gnn_path)} AS gnn
    USING (
        source,
        split,
        ranking_group_id,
        index_condition_token,
        candidate_medication_token,
        candidate_rank,
        label_prescribed,
        patient_fold_id
    )
ORDER BY transformer.ranking_group_id, transformer.candidate_rank
"""
    ).to_arrow_reader(batch_size=100_000)
    pending = pd.DataFrame()
    for record_batch in reader:
        frame = record_batch.to_pandas()
        if not pending.empty:
            frame = pd.concat((pending, frame), ignore_index=True)
            pending = pd.DataFrame()
        if frame.empty:
            continue
        last_group = frame["ranking_group_id"].iloc[-1]
        complete = frame[frame["ranking_group_id"] != last_group]
        pending = frame[frame["ranking_group_id"] == last_group]
        for _group_id, group in complete.groupby("ranking_group_id", sort=False):
            update_group(group)
    if not pending.empty:
        update_group(pending)
    if joined_rows <= 0:
        raise ValueError("paired OOF join produced no candidates")
    summaries = {
        f"{weight:.3f}": accumulator.summary()
        for weight, accumulator in accumulators.items()
    }
    selected_weight = max(
        weights,
        key=lambda weight: (
            float(summaries[f"{weight:.3f}"]["ndcg_at_k"]),
            float(summaries[f"{weight:.3f}"]["mrr_at_k"]),
            float(summaries[f"{weight:.3f}"]["hit_rate_at_k"]),
            -weight,
        ),
    )
    return {
        "joined_candidate_row_count": joined_rows,
        "transformer_metrics": transformer_accumulator.summary(),
        "gnn_metrics": gnn_accumulator.summary(),
        "candidate_weights": summaries,
        "selected_alpha": selected_weight,
        "selected_metrics": summaries[f"{selected_weight:.3f}"],
    }


def select_paired_oof_late_fusion(config: GNNTrainingConfig) -> dict[str, Any]:
    """Jointly freeze GNN variant and alpha using paired train OOF only."""

    generated_at = datetime.now(UTC).isoformat()
    started = perf_counter()
    report: dict[str, Any] = {
        "schema_version": PAIRED_OOF_SELECTION_SCHEMA_VERSION,
        "protocol_version": PAIRED_OOF_PROTOCOL_VERSION,
        "status": "running",
        "stage": "select-paired-oof",
        "generated_at": generated_at,
        "mode": config.mode,
        "fold_count": config.fold_count,
        "alpha_grid": {
            "minimum": min(alpha_grid()),
            "maximum": max(alpha_grid()),
            "step": alpha_grid()[1] - alpha_grid()[0],
            "values": list(alpha_grid()),
        },
        "selection_scope": "mimiciv_train_paired_patient_grouped_oof_only",
        "leakage_policy": {
            "validation_gate_used_for_selection": False,
            "gnn_variant_selected_on_paired_oof": True,
            "alpha_selected_on_paired_oof": True,
            "adaptive_fusion_is_separate_escalation": True,
        },
        "data_safety": {
            "report_contains_patient_rows": False,
            "report_contains_row_samples": False,
            "report_contains_identifier_values": False,
        },
    }
    errors = preflight_errors(config, stage="select-paired-oof")
    if errors:
        report.update(
            blocked_report(
                config=config,
                schema_version=PAIRED_OOF_SELECTION_SCHEMA_VERSION,
                stage="select-paired-oof",
                generated_at=generated_at,
                errors=errors,
            )
        )
        write_json(config.paired_oof_selection_report_path, report)
        return report
    try:
        if config.mode != "development":
            raise ValueError("paired OOF selection must run in development mode")
        if (
            "protected" in config.dataset_root.resolve().parts
            and config.fold_count != PRODUCTION_FOLD_COUNT
        ):
            raise ValueError("protected paired OOF selection requires five folds")
        transformer_paths = _completed_transformer_artifacts(config)
        transformer_scan = _scan_paths(transformer_paths)
        variant_results: dict[str, dict[str, Any]] = {}
        with duckdb.connect(database=":memory:") as connection:
            configure_connection(config, connection)
            expected_rows: int | None = None
            for variant in ABLATION_VARIANTS:
                gnn_path = _completed_gnn_artifact(config, variant)
                row_count = _validate_pairing(
                    connection,
                    transformer_scan=transformer_scan,
                    gnn_path=gnn_path,
                    fold_count=config.fold_count,
                )
                if expected_rows is None:
                    expected_rows = row_count
                elif expected_rows != row_count:
                    raise ValueError("GNN variants have inconsistent OOF row counts")
                variant_results[variant] = _evaluate_variant(
                    connection,
                    transformer_scan=transformer_scan,
                    gnn_path=gnn_path,
                )

        selected_variant = max(
            ABLATION_VARIANTS,
            key=lambda variant: (
                float(variant_results[variant]["selected_metrics"]["ndcg_at_k"]),
                float(variant_results[variant]["selected_metrics"]["mrr_at_k"]),
                float(variant_results[variant]["selected_metrics"]["hit_rate_at_k"]),
                -float(variant_results[variant]["selected_alpha"]),
                -ABLATION_VARIANTS.index(variant),
            ),
        )
        selected = variant_results[selected_variant]
        selected_alpha = float(selected["selected_alpha"])
        transformer_metrics = selected["transformer_metrics"]
        best_gnn_variant = max(
            ABLATION_VARIANTS,
            key=lambda variant: (
                float(variant_results[variant]["gnn_metrics"]["ndcg_at_k"]),
                float(variant_results[variant]["gnn_metrics"]["mrr_at_k"]),
                float(variant_results[variant]["gnn_metrics"]["hit_rate_at_k"]),
                -ABLATION_VARIANTS.index(variant),
            ),
        )
        best_gnn_metrics = variant_results[best_gnn_variant]["gnn_metrics"]
        best_base_ndcg = max(
            float(transformer_metrics["ndcg_at_k"]),
            float(best_gnn_metrics["ndcg_at_k"]),
        )
        oof_lift = float(selected["selected_metrics"]["ndcg_at_k"]) - best_base_ndcg

        selected_refit_epochs = max(
            1,
            int(
                round(
                    statistics.median(
                        int(
                            load_json(
                                config.fold_completion_manifest_path(
                                    fold_index,
                                    selected_variant,
                                )
                            )["best_epoch"]
                        )
                        + 1
                        for fold_index in range(config.fold_count)
                    )
                )
            ),
        )
        refit_variant: str | None = None
        available_refit_epochs: int | None = None
        if config.gnn_training_state_path.is_file():
            state = load_json(config.gnn_training_state_path)
            value = state.get("selected_variant")
            refit_variant = value if isinstance(value, str) else None
            raw_epochs = state.get("refit_epochs")
            available_refit_epochs = raw_epochs if isinstance(raw_epochs, int) else None
        refit_required = (
            refit_variant != selected_variant
            or available_refit_epochs != selected_refit_epochs
            or not config.gnn_checkpoint_path.is_file()
        )
        checkpoint = {
            "schema_version": PAIRED_OOF_SELECTION_SCHEMA_VERSION,
            "protocol_version": PAIRED_OOF_PROTOCOL_VERSION,
            "status": "selected_pending_gnn_refit" if refit_required else "completed",
            "selection_frozen": True,
            "selected_gnn_variant": selected_variant,
            "selected_alpha": selected_alpha,
            "alpha_semantics": "(1-alpha)*zscore(transformer)+alpha*zscore(gnn)",
            "selection_scope": report["selection_scope"],
            "contract_digest": contract_digest_or_none(config),
            "inference_ready": not refit_required,
            "refit_required": refit_required,
            "available_full_refit_variant": refit_variant,
            "selected_gnn_refit_epochs": selected_refit_epochs,
            "available_full_refit_epochs": available_refit_epochs,
            "transformer_oof_sha256": [sha256_file(path) for path in transformer_paths],
            "gnn_oof_sha256": sha256_file(
                config.variant_oof_predictions_path(selected_variant)
            ),
            "transformer_checkpoint_sha256": sha256_file(config.neural_checkpoint_path),
            "transformer_cache_manifest_sha256": sha256_file(
                config.transformer_cache_manifest_path
            ),
            "gnn_cache_manifest_sha256": sha256_file(config.cache_manifest_path),
            "gnn_checkpoint": (
                {
                    "path": str(config.gnn_checkpoint_path),
                    "sha256": sha256_file(config.gnn_checkpoint_path),
                }
                if not refit_required and config.gnn_checkpoint_path.is_file()
                else None
            ),
            "generated_at": generated_at,
        }
        write_json(config.paired_oof_selection_path, checkpoint)
        write_json(config.paired_late_checkpoint_path, checkpoint)
        report.update(
            {
                "status": "completed",
                "selected_gnn_variant": selected_variant,
                "selected_alpha": selected_alpha,
                "selected_metrics": selected["selected_metrics"],
                "transformer_oof_metrics": transformer_metrics,
                "best_standalone_gnn_variant": best_gnn_variant,
                "best_standalone_gnn_oof_metrics": best_gnn_metrics,
                "best_base_oof_ndcg_at_10": best_base_ndcg,
                "selected_oof_ndcg_lift_over_best_base": oof_lift,
                "minimum_material_lift": MINIMUM_NDCG_LIFT,
                "global_blend_is_marginal": oof_lift < MINIMUM_NDCG_LIFT,
                "adaptive_fusion_authorized": oof_lift < MINIMUM_NDCG_LIFT,
                "gnn_refit_required": refit_required,
                "inference_ready": not refit_required,
                "variant_results": variant_results,
                "contract_digest": contract_digest_or_none(config),
                "artifacts": {
                    "protected_selection": str(config.paired_oof_selection_path),
                    "late_only_checkpoint": str(config.paired_late_checkpoint_path),
                },
            }
        )
    except Exception as error:  # noqa: BLE001 - aggregate fail-closed report
        report["status"] = "failed"
        report["reason"] = safe_error_message(error)
    report["wall_time_seconds"] = perf_counter() - started
    write_json(config.paired_oof_selection_report_path, report)
    return report
