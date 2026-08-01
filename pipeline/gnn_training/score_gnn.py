"""Canonical standalone GNN scoring and graph-only qualification gate."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from pipeline.config import (
    GRAPH_ABLATION_VERSION,
    GRAPH_VERSION,
    LABEL_VERSION,
    MILESTONE8B_REPORT_VERSION,
    SPLIT_VERSION,
)
from pipeline.extract_utils import parquet_scan, safe_error_message, sql_string
from pipeline.features import copy_query_to_parquet
from pipeline.gnn_training.config import (
    GNN_BASELINE_NAME,
    GNN_EXPERIMENT_VERSION,
    GNN_SELECTION_SCHEMA_VERSION,
    GNN_TRAINING_SCHEMA_VERSION,
    GRAPH_REFERENCE_BASELINE_NAME,
    MAXIMUM_SECONDARY_DROP,
    MINIMUM_NDCG_LIFT,
    GNNTrainingConfig,
)
from pipeline.gnn_training.contract import (
    blocked_report,
    contract_digest_or_none,
    preflight_errors,
)
from pipeline.gnn_training.data import write_json, write_json_exclusive
from pipeline.gnn_training.runtime import (
    frozen_artifact_locks,
    load_feature_spec,
    load_gnn_checkpoint,
    read_positive_temperature,
    require_finite_tensor,
    resolve_device,
    validate_gnn_training_state,
)
from pipeline.gnn_training.scoring import (
    COMPACT_PREDICTION_SCHEMA,
    AtomicParquetWriter,
    append_authoritative_metrics,
    base_score_report,
    configured_duckdb,
    materialize_canonical_scores,
    metric_row,
    qualification_decision,
    write_compact_batch,
)
from pipeline.gnn_training.dataset import iter_batches
from pipeline.training_contract import load_json, sha256_file


def _reference_metric_from_report(
    config: GNNTrainingConfig,
    *,
    split: str,
) -> dict[str, Any] | None:
    if not config.graph_reference_report_path.is_file():
        return None
    try:
        payload = load_json(config.graph_reference_report_path)
    except (OSError, TypeError, ValueError):
        return None
    if not config.allow_ungated:
        versions = payload.get("versions")
        frozen = payload.get("frozen_selection")
        if (
            payload.get("status") != "completed"
            or payload.get("schema_version") != MILESTONE8B_REPORT_VERSION
            or not isinstance(versions, dict)
            or versions.get("graph_ablation_version") != GRAPH_ABLATION_VERSION
            or versions.get("graph_version") != GRAPH_VERSION
            or versions.get("label_version") != LABEL_VERSION
            or versions.get("split_version") != SPLIT_VERSION
            or not isinstance(frozen, dict)
            or frozen.get("status") != "frozen"
        ):
            return None
    metric = metric_row(
        payload,
        baseline_name=GRAPH_REFERENCE_BASELINE_NAME,
        split=split,
    )
    if metric is None:
        return None
    try:
        valid = (
            int(metric["positive_ranking_group_count"]) > 0
            and math.isfinite(float(metric["ndcg_at_k"]))
            and math.isfinite(float(metric["mrr_at_k"]))
            and math.isfinite(float(metric["hit_rate_at_k"]))
        )
    except (KeyError, TypeError, ValueError):
        valid = False
    return metric if valid else None


@torch.no_grad()
def _write_predictions(
    config: GNNTrainingConfig,
    *,
    split: str,
    temperature: float,
    output_path: Path,
) -> tuple[int, str]:
    spec = load_feature_spec(config)
    device = resolve_device(config)
    model, payload = load_gnn_checkpoint(
        config.gnn_checkpoint_path,
        spec,
        device=device,
        expected_seed=config.seed,
    )
    validate_gnn_training_state(config, payload)
    model.eval()
    with AtomicParquetWriter(output_path, COMPACT_PREDICTION_SCHEMA) as writer:
        for batch in iter_batches(
            config,
            spec,
            split=split,
            batch_groups=config.optimization.batch_ranking_groups,
            shuffle=False,
            seed=config.seed,
            shards_root=config.shards_root,
        ):
            moved = batch.to(device)
            logits = model.forward_batch(moved).logits
            require_finite_tensor(
                logits,
                name="GNN scoring logits",
                mask=moved.candidate_mask,
            )
            probabilities = torch.sigmoid(logits / temperature)
            require_finite_tensor(
                probabilities,
                name="GNN scoring probabilities",
                mask=moved.candidate_mask,
            )
            write_compact_batch(writer, batch, probabilities)
        rows = writer.commit()
    return rows, str(device)


def _append_graph_reference(
    connection: Any,
    config: GNNTrainingConfig,
    *,
    candidate_path: Path,
    split: str,
) -> tuple[int, bool]:
    config.gnn_score_root.mkdir(parents=True, exist_ok=True)
    if not config.graph_reference_scores_path.is_file():
        rows = copy_query_to_parquet(
            connection,
            f"SELECT * FROM {parquet_scan(candidate_path)}",
            config.gnn_score_output_path,
        )
        return rows, False
    locked = parquet_scan(config.patient_condition_medication_path)
    reference = parquet_scan(config.graph_reference_scores_path)
    reference_check = connection.execute(
        f"""
WITH locked_candidates AS (
    SELECT
        source,
        split,
        ranking_group_id,
        index_condition_token,
        candidate_medication_token,
        CAST(candidate_rank AS BIGINT) AS candidate_rank,
        CAST(label_prescribed AS BOOLEAN) AS label_prescribed
    FROM {locked}
    WHERE source = 'mimiciv' AND split = {sql_string(split)}
),
reference_candidates AS (
    SELECT
        source,
        split,
        ranking_group_id,
        index_condition_token,
        candidate_medication_token,
        CAST(candidate_rank AS BIGINT) AS candidate_rank,
        CAST(label_prescribed AS BOOLEAN) AS label_prescribed
    FROM {reference}
    WHERE source = 'mimiciv'
        AND split = {sql_string(split)}
        AND baseline_name = {sql_string(GRAPH_REFERENCE_BASELINE_NAME)}
),
locked_only AS (
    SELECT * FROM locked_candidates
    EXCEPT ALL
    SELECT * FROM reference_candidates
),
reference_only AS (
    SELECT * FROM reference_candidates
    EXCEPT ALL
    SELECT * FROM locked_candidates
)
SELECT
    (SELECT COUNT(*) FROM reference_candidates) AS reference_count,
    (SELECT COUNT(*) FROM locked_only)
        + (SELECT COUNT(*) FROM reference_only) AS mismatch_count,
    (
        SELECT COUNT(*)
        FROM {reference}
        WHERE source = 'mimiciv'
            AND split = {sql_string(split)}
            AND baseline_name = {sql_string(GRAPH_REFERENCE_BASELINE_NAME)}
            AND (
                score IS NULL
                OR NOT isfinite(score)
                OR seed IS DISTINCT FROM {int(config.seed)}
                OR baseline_version IS DISTINCT FROM
                    {sql_string(GRAPH_ABLATION_VERSION)}
                OR evaluation_version IS DISTINCT FROM
                    {sql_string(MILESTONE8B_REPORT_VERSION)}
            )
    ) AS invalid_metadata_count
"""
    ).fetchone()
    if reference_check is None or int(reference_check[0]) == 0:
        if not config.allow_ungated:
            raise ValueError(
                "graph-only reference score table has no locked evaluation rows"
            )
        rows = copy_query_to_parquet(
            connection,
            f"SELECT * FROM {parquet_scan(candidate_path)}",
            config.gnn_score_output_path,
        )
        return rows, False
    if int(reference_check[1]) != 0 or int(reference_check[2]) != 0:
        raise ValueError(
            "graph-only reference scores do not match the locked evaluation scope"
        )
    query = f"""
SELECT * FROM {parquet_scan(candidate_path)}
UNION ALL BY NAME
SELECT
    source,
    split,
    ranking_group_id,
    index_condition_token,
    candidate_medication_token,
    candidate_rank,
    label_prescribed,
    baseline_name,
    score,
    seed,
    baseline_version,
    evaluation_version,
    generated_at
FROM {parquet_scan(config.graph_reference_scores_path)}
WHERE source = 'mimiciv'
    AND split = {sql_string(split)}
    AND baseline_name = {sql_string(GRAPH_REFERENCE_BASELINE_NAME)}
"""
    rows = copy_query_to_parquet(
        connection,
        query,
        config.gnn_score_output_path,
    )
    return rows, True


def _selection_report(
    config: GNNTrainingConfig,
    report: dict[str, Any],
    *,
    split: str,
    reference_appended: bool,
) -> dict[str, Any]:
    candidate = metric_row(report, baseline_name=GNN_BASELINE_NAME, split=split)
    reference = (
        metric_row(
            report,
            baseline_name=GRAPH_REFERENCE_BASELINE_NAME,
            split=split,
        )
        if reference_appended
        else _reference_metric_from_report(config, split=split)
    )
    if candidate is None or reference is None:
        return {
            "schema_version": GNN_SELECTION_SCHEMA_VERSION,
            "status": "failed_missing_validation_metrics",
            "model_frozen": False,
            "standalone_qualified": False,
            "generated_at": report["generated_at"],
            "data_safety": {
                "report_contains_patient_rows": False,
                "report_contains_row_samples": False,
                "report_contains_identifier_values": False,
            },
        }
    decision = qualification_decision(
        candidate=candidate,
        reference=reference,
        minimum_ndcg_lift=MINIMUM_NDCG_LIFT,
        maximum_secondary_drop=MAXIMUM_SECONDARY_DROP,
    )
    qualified = bool(decision["qualified"])
    return {
        "schema_version": GNN_SELECTION_SCHEMA_VERSION,
        "status": "frozen",
        # The relation checkpoint is immutable for fusion even when it does not
        # independently clear the graph-only reference gate.
        "model_frozen": True,
        "standalone_qualified": qualified,
        "final_scoring_authorized": qualified,
        "decision": (
            "qualify_standalone_gnn"
            if qualified
            else "retain_graph_only_xgboost_for_standalone"
        ),
        "generated_at": report["generated_at"],
        "selected_experiment": GNN_BASELINE_NAME,
        "reference_experiment": GRAPH_REFERENCE_BASELINE_NAME,
        "selection_basis": {
            "model_selection": "patient_grouped_mimic_train_crossfit",
            "gate_split": f"mimiciv_{split}",
            "k": 10,
            "candidate_metrics": candidate,
            "reference_metrics": reference,
        },
        **decision,
        "contract_digest": contract_digest_or_none(config),
        "seed": config.seed,
        "frozen_artifacts": frozen_artifact_locks(
            {
                "checkpoint": config.gnn_checkpoint_path,
                "calibration": config.gnn_calibration_path,
                "feature_layout": config.feature_layout_path,
                "training_state": config.gnn_training_state_path,
                "oof_predictions": config.gnn_oof_predictions_path,
                "crossfit_manifest": config.crossfit_graph_manifest_path,
                "cache_manifest": config.cache_manifest_path,
                "graph_reference_report": config.graph_reference_report_path,
                "graph_reference_scores": config.graph_reference_scores_path,
            }
        ),
        "fusion_policy": (
            "A frozen but non-qualifying standalone relation branch may enter "
            "the independently gated hybrid experiment; it is not promoted "
            "as a standalone scorer."
        ),
        "data_safety": {
            "report_contains_patient_rows": False,
            "report_contains_row_samples": False,
            "report_contains_identifier_values": False,
        },
    }


def _final_attempt_exists(config: GNNTrainingConfig) -> bool:
    return config.mode == "final" and config.gnn_final_score_completion_path.is_file()


def score_gnn(config: GNNTrainingConfig) -> dict[str, Any]:
    """Score validation/test and freeze the standalone qualification result."""

    generated_at = datetime.now(UTC).isoformat()
    split = "test" if config.mode == "final" else "validation"
    errors = preflight_errors(config, stage="score-gnn")
    if _final_attempt_exists(config):
        errors.append(
            {
                "code": "final_score_already_attempted",
                "detail": (
                    "one-shot final standalone GNN scoring was already claimed; "
                    "existing artifacts were left unchanged"
                ),
            }
        )
    if errors:
        report = blocked_report(
            config=config,
            schema_version=GNN_TRAINING_SCHEMA_VERSION,
            stage="score-gnn",
            generated_at=generated_at,
            errors=errors,
        )
        if not _final_attempt_exists(config):
            write_json(config.active_gnn_score_report_path, report)
        return report

    final_claimed = False
    if config.mode == "final":
        try:
            write_json_exclusive(
                config.gnn_final_score_completion_path,
                {
                    "schema_version": GNN_TRAINING_SCHEMA_VERSION,
                    "status": "running",
                    "mode": "final",
                    "scored_split": "test",
                    "started_at": generated_at,
                    "one_shot_claim": True,
                    "data_safety": {
                        "contains_patient_rows": False,
                        "contains_identifier_values": False,
                    },
                },
            )
            final_claimed = True
        except FileExistsError:
            report = blocked_report(
                config=config,
                schema_version=GNN_TRAINING_SCHEMA_VERSION,
                stage="score-gnn",
                generated_at=generated_at,
                errors=[
                    {
                        "code": "final_score_claim_race",
                        "detail": (
                            "another process claimed the one-shot final GNN run"
                        ),
                    }
                ],
            )
            return report

    report = base_score_report(
        config,
        schema_version=GNN_TRAINING_SCHEMA_VERSION,
        stage="score-gnn",
        split=split,
        baseline_name=GNN_BASELINE_NAME,
        output_path=config.gnn_score_output_path,
    )
    try:
        if (
            not config.allow_ungated
            and _reference_metric_from_report(config, split=split) is None
        ):
            raise ValueError(
                "graph-only aggregate reference report is not completed, "
                "version-compatible, frozen, and evaluable"
            )
        temperature = read_positive_temperature(
            config.gnn_calibration_path,
            expected_schema_version=GNN_TRAINING_SCHEMA_VERSION,
            allowed_methods=frozenset({"bounded_log_grid_single_temperature_bce"}),
            training_state_path=config.gnn_training_state_path,
        )
        prediction_path = config.gnn_score_root / "_gnn_predictions.parquet"
        candidate_path = config.gnn_score_root / "_gnn_candidate_scores.parquet"
        prediction_rows, device = _write_predictions(
            config,
            split=split,
            temperature=temperature,
            output_path=prediction_path,
        )
        with configured_duckdb(config) as connection:
            candidate_rows = materialize_canonical_scores(
                connection,
                config,
                predictions_path=prediction_path,
                output_path=candidate_path,
                split=split,
                baseline_name=GNN_BASELINE_NAME,
                baseline_version=GNN_EXPERIMENT_VERSION,
                evaluation_version=GNN_TRAINING_SCHEMA_VERSION,
                generated_at=report["generated_at"],
            )
            score_rows, reference_appended = _append_graph_reference(
                connection,
                config,
                candidate_path=candidate_path,
                split=split,
            )
            append_authoritative_metrics(
                connection,
                config,
                report,
                evaluation_root=config.gnn_score_root,
            )
        if (
            prediction_rows <= 0
            or candidate_rows <= 0
            or metric_row(
                report,
                baseline_name=GNN_BASELINE_NAME,
                split=split,
            )
            is None
        ):
            raise ValueError("standalone GNN scoring produced no authoritative metrics")
        report.update(
            {
                "temperature": temperature,
                "device": device,
                "candidate_prediction_row_count": prediction_rows,
                "candidate_score_row_count": candidate_rows,
                "score_row_count": score_rows,
                "reference_appended": reference_appended,
                "reference_baseline": GRAPH_REFERENCE_BASELINE_NAME,
            }
        )
        if config.mode == "development":
            selection = _selection_report(
                config,
                report,
                split=split,
                reference_appended=reference_appended,
            )
            write_json(config.gnn_selection_report_path, selection)
            report["selection_report"] = str(config.gnn_selection_report_path)
            report["standalone_qualified"] = selection.get(
                "standalone_qualified", False
            )
            report["decision"] = selection.get("decision")
            if selection.get("status") != "frozen":
                report["status"] = "failed"
                report["reason"] = (
                    "standalone GNN selection could not freeze comparable "
                    "validation metrics"
                )
        else:
            report["frozen_selection_reference"] = str(config.gnn_selection_report_path)
    except Exception as error:  # noqa: BLE001 - aggregate fail-closed report
        report["status"] = "failed"
        report["reason"] = safe_error_message(error)

    write_json(config.active_gnn_score_report_path, report)
    if final_claimed:
        marker: dict[str, Any] = {
            "schema_version": GNN_TRAINING_SCHEMA_VERSION,
            "status": report.get("status", "failed"),
            "mode": "final",
            "scored_split": "test",
            "started_at": generated_at,
            "completed_at": datetime.now(UTC).isoformat(),
            "one_shot_claim": True,
            "data_safety": {
                "contains_patient_rows": False,
                "contains_identifier_values": False,
            },
        }
        if report.get("status") == "completed":
            marker.update(
                {
                    "score_output_sha256": sha256_file(config.gnn_score_output_path),
                    "selection_report_sha256": sha256_file(
                        config.gnn_selection_report_path
                    ),
                }
            )
        write_json(config.gnn_final_score_completion_path, marker)
    return report
