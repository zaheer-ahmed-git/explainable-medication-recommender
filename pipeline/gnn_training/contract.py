"""Fail-closed, aggregate-only preflight checks for GNN and fusion stages.

The checks in this module never import PyTorch and never inspect or report
clinical rows.  They validate immutable upstream contracts, the frozen
Transformer artifact locks, path containment, stage-local artifacts, final
selection locks, and the patient-grouped cross-fit graph contract required for
model selection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from pipeline.gnn_training.config import (
    CROSS_FIT_SCHEMA_VERSION,
    FULL_TRAIN_REFIT_SCOPE,
    PREPARE_PENDING_STATUS,
    RELATION_TO_INDEX,
    RELATION_TYPES,
    GNNTrainingConfig,
)
from pipeline.training_contract import (
    PINNED_VERSIONS,
    load_json,
    sha256_file,
)

STAGES = (
    "prepare",
    "train-gnn",
    "train-gnn-fold",
    "materialize-gnn-oof",
    "select-paired-oof",
    "refit-paired-gnn",
    "score-paired-late",
    "select-gnn",
    "refit-gnn",
    "score-gnn",
    "train-fusion",
    "score-fusion",
)
GNN_STAGES = frozenset(
    {
        "train-gnn",
        "train-gnn-fold",
        "materialize-gnn-oof",
        "select-paired-oof",
        "refit-paired-gnn",
        "score-paired-late",
        "select-gnn",
        "refit-gnn",
        "score-gnn",
    }
)
FUSION_STAGES = frozenset({"train-fusion", "score-fusion"})
CROSS_FIT_REQUIRED_STAGES = GNN_STAGES | FUSION_STAGES
SCORING_STAGES = frozenset({"score-gnn", "score-fusion", "score-paired-late"})


def _error(
    code: str,
    detail: str,
    *,
    artifact_name: str | None = None,
    **extra: str | int | bool,
) -> dict[str, Any]:
    row: dict[str, Any] = {"code": code, "detail": detail}
    if artifact_name is not None:
        row["artifact_name"] = artifact_name
    row.update(extra)
    return row


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.expanduser().resolve().relative_to(root.expanduser().resolve())
    except (OSError, RuntimeError, ValueError):
        return False
    return True


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.expanduser().resolve() == right.expanduser().resolve()
    except (OSError, RuntimeError):
        return False


def _is_protected(path: Path) -> bool:
    try:
        return "protected" in path.expanduser().resolve().parts
    except (OSError, RuntimeError):
        return False


def path_safety_errors(config: GNNTrainingConfig) -> list[dict[str, Any]]:
    """Validate the immutable input roots and every configured output path."""

    errors: list[dict[str, Any]] = []
    if config.mode not in {"development", "final"}:
        errors.append(
            _error(
                "invalid_config",
                "mode must be development or final",
                artifact_name="mode",
            )
        )
    if config.fold_count < 2:
        errors.append(
            _error(
                "invalid_config",
                "fold_count must be at least two",
                artifact_name="fold_count",
            )
        )
    if config.shard_count <= 0:
        errors.append(
            _error(
                "invalid_config",
                "shard_count must be positive",
                artifact_name="shard_count",
            )
        )
    if not config.top_k or any(
        not isinstance(value, int) or value <= 0 for value in config.top_k
    ):
        errors.append(
            _error(
                "invalid_config",
                "top_k must contain positive integers",
                artifact_name="top_k",
            )
        )
    elif 10 not in config.top_k:
        errors.append(
            _error(
                "invalid_config",
                "top_k must include 10 because all frozen selection gates use @10",
                artifact_name="top_k",
            )
        )
    if config.architecture.relation_count != len(RELATION_TYPES):
        errors.append(
            _error(
                "invalid_config",
                "architecture must use the stable relation vocabulary size",
                artifact_name="relation_count",
            )
        )
    if not _same_path(config.gnn_root, config.expected_gnn_root):
        errors.append(
            _error(
                "unsafe_gnn_root",
                "gnn_root must equal DATASET_ROOT/processed/phase8_p0/gnn",
                artifact_name="gnn_root",
            )
        )
    if not _same_path(config.neural_root, config.expected_neural_root):
        errors.append(
            _error(
                "unsafe_neural_root",
                "neural_root must equal the immutable Phase 8 P0 neural root",
                artifact_name="neural_root",
            )
        )

    expected_phase_root = config.dataset_root / "processed" / "phase8_p0"
    expected_graph_root = expected_phase_root / "graph" / "milestone8"
    expected_features_root = expected_phase_root / "features"
    expected_training_root = expected_phase_root / "training"
    if not _same_path(config.graph_root, expected_graph_root):
        errors.append(
            _error(
                "unsafe_graph_input_root",
                "graph_root must equal the locked Phase 8 P0 graph root",
                artifact_name="graph_root",
            )
        )
    if not _same_path(config.subgraphs_root, expected_graph_root / "patient_subgraphs"):
        errors.append(
            _error(
                "unsafe_subgraph_input_root",
                "subgraphs_root must equal the locked patient-subgraph root",
                artifact_name="subgraphs_root",
            )
        )
    if not _same_path(config.training_root, expected_training_root):
        errors.append(
            _error(
                "unsafe_training_input_root",
                "training_root must equal the locked Phase 8 P0 training root",
                artifact_name="training_root",
            )
        )
    if not _same_path(config.features_root, expected_features_root):
        errors.append(
            _error(
                "unsafe_features_input_root",
                "features_root must equal the locked Phase 8 P0 features root",
                artifact_name="features_root",
            )
        )
    expected_reference = (
        expected_phase_root
        / "evaluation"
        / "milestone8b"
        / "graph_ablation_scores.parquet"
    )
    if not _same_path(config.graph_reference_scores_path, expected_reference):
        errors.append(
            _error(
                "unsafe_reference_score_path",
                "graph-only reference scores must use the locked Phase 8 P0 path",
                artifact_name="graph_reference_scores",
            )
        )

    roots_overlap = _is_under(config.gnn_root, config.neural_root) or _is_under(
        config.neural_root, config.gnn_root
    )
    if roots_overlap:
        errors.append(
            _error(
                "gnn_neural_root_overlap",
                "GNN writes and frozen neural inputs must use disjoint roots",
            )
        )

    allowed_restricted_roots = (config.gnn_root, config.paired_protocol_root)
    for index, path in enumerate(config.restricted_write_paths()):
        if not any(
            _is_under(path, root) for root in allowed_restricted_roots
        ) or _is_under(path, config.neural_root):
            errors.append(
                _error(
                    "unsafe_artifact_write_path",
                    "restricted GNN artifact writes must stay under gnn_root or "
                    "the versioned paired-protocol root, and outside neural_root",
                    artifact_name=f"restricted_write_{index}",
                )
            )

    aggregate_paths = (
        config.contract_lock_path,
        config.subgraphs_manifest_path,
        config.neural_selection_path,
        config.graph_reference_report_path,
        *config.aggregate_report_paths(),
    )
    for index, path in enumerate(aggregate_paths):
        if not _is_under(path, config.reports_root):
            errors.append(
                _error(
                    "unsafe_report_path",
                    "aggregate manifests and reports must stay under REPORTS_ROOT",
                    artifact_name=f"aggregate_report_{index}",
                )
            )

    if config.allow_ungated and _is_protected(config.dataset_root):
        errors.append(
            _error(
                "ungated_production_path",
                "--allow-ungated is reserved for synthetic unit-test roots and "
                "cannot authorize protected production paths",
            )
        )
    return errors


def required_inputs(config: GNNTrainingConfig) -> dict[str, Path]:
    """Return immutable upstream inputs required when gates are enforced."""

    return {
        "training_contract_lock": config.contract_lock_path,
        "patient_subgraphs_manifest": config.subgraphs_manifest_path,
        "neural_selection": config.neural_selection_path,
        "subgraph_index": config.subgraph_index_path,
        "subgraph_nodes": config.subgraph_nodes_path,
        "subgraph_edges": config.subgraph_edges_path,
        "subgraph_candidates": config.subgraph_candidates_path,
        "patient_condition_medication": config.patient_condition_medication_path,
        "event_sequences": config.event_sequences_path,
        "neural_checkpoint": config.neural_checkpoint_path,
        "neural_calibration": config.neural_calibration_path,
        "neural_feature_layout": config.neural_feature_layout_path,
    }


def graph_cache_artifacts(config: GNNTrainingConfig) -> dict[str, Path]:
    """Return prepared graph artifacts required by standalone GNN stages."""

    return {
        "feature_layout": config.feature_layout_path,
        "graph_node_vocabulary": config.graph_node_vocabulary_path,
        "node_type_vocabulary": config.node_type_vocabulary_path,
        "node_role_vocabulary": config.node_role_vocabulary_path,
        "relation_vocabulary": config.relation_vocabulary_path,
        "cache_manifest": config.cache_manifest_path,
        "graph_shards": config.shards_root,
    }


def transformer_cache_artifacts(config: GNNTrainingConfig) -> dict[str, Path]:
    """Return frozen Transformer cache artifacts required by fusion stages."""

    return {
        "frozen_transformer_cache": config.frozen_transformer_cache_root,
        "frozen_transformer_cache_manifest": config.transformer_cache_manifest_path,
    }


def checkpoint_artifacts(
    config: GNNTrainingConfig,
    *,
    branch: str,
) -> dict[str, Path]:
    """Return stage-local checkpoint artifacts for one scoring branch."""

    if branch == "gnn":
        return {
            "gnn_checkpoint": config.gnn_checkpoint_path,
            "gnn_calibration": config.gnn_calibration_path,
            "gnn_training_state": config.gnn_training_state_path,
        }
    if branch == "fusion":
        return {
            "fusion_checkpoint": config.fusion_checkpoint_path,
            "fusion_calibration": config.fusion_calibration_path,
            "fusion_training_state": config.fusion_training_state_path,
        }
    raise ValueError(f"unknown checkpoint branch: {branch!r}")


def required_stage_artifacts(
    config: GNNTrainingConfig,
    *,
    stage: str,
) -> dict[str, Path]:
    """Return local artifacts that must pre-exist for a stage."""

    if stage not in STAGES:
        raise ValueError(f"unknown stage: {stage!r}")
    if stage == "prepare":
        return {}

    artifacts = graph_cache_artifacts(config)
    if stage in FUSION_STAGES:
        artifacts.update(transformer_cache_artifacts(config))
        artifacts.update(checkpoint_artifacts(config, branch="gnn"))
    if stage == "score-paired-late":
        artifacts.update(transformer_cache_artifacts(config))
    if stage == "score-gnn":
        artifacts.update(checkpoint_artifacts(config, branch="gnn"))
    if stage == "score-fusion":
        artifacts.update(checkpoint_artifacts(config, branch="fusion"))
    return artifacts


def _load_document(
    path: Path,
    *,
    invalid_code: str,
    artifact_name: str,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    try:
        payload = load_json(path)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None, [
            _error(
                invalid_code,
                "aggregate JSON metadata is missing, unreadable, or malformed",
                artifact_name=artifact_name,
            )
        ]
    if not isinstance(payload, dict):
        return None, [
            _error(
                invalid_code,
                "aggregate JSON metadata must be an object",
                artifact_name=artifact_name,
            )
        ]
    return payload, []


def _training_contract_errors(
    config: GNNTrainingConfig,
) -> tuple[str | None, dict[str, Any] | None, list[dict[str, Any]]]:
    payload, errors = _load_document(
        config.contract_lock_path,
        invalid_code="invalid_contract_lock",
        artifact_name="training_contract_lock",
    )
    if payload is None:
        return None, None, errors

    digest = payload.get("contract_digest")
    if (
        payload.get("status") != "completed"
        or not isinstance(digest, str)
        or not digest
    ):
        errors.append(
            _error(
                "invalid_contract_lock",
                "training contract lock must be completed and contain a digest",
                artifact_name="training_contract_lock",
            )
        )
        digest = None

    versions = payload.get("versions")
    if not isinstance(versions, dict):
        errors.append(
            _error(
                "missing_pinned_versions",
                "training contract lock must declare all pinned versions",
                artifact_name="training_contract_lock",
            )
        )
    else:
        for name, expected in PINNED_VERSIONS.items():
            if versions.get(name) != expected:
                errors.append(
                    _error(
                        "contract_version_mismatch",
                        "training contract pinned version does not match the "
                        "Phase 8 P0 contract",
                        artifact_name=name,
                    )
                )
    return digest, payload, errors


def _locked_subgraph_manifest_errors(
    config: GNNTrainingConfig,
    contract_lock: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """Verify the current subgraph manifest is exactly the contract-locked file."""

    if contract_lock is None:
        return []
    try:
        lock = contract_lock["contract"]["manifests"]["patient_subgraphs_manifest"]
        locked_path = Path(lock["path"])
        locked_hash = str(lock["sha256"])
    except (KeyError, TypeError, ValueError):
        return [
            _error(
                "missing_locked_subgraph_manifest",
                "training contract does not lock the patient-subgraph manifest",
                artifact_name="patient_subgraphs_manifest",
            )
        ]

    errors: list[dict[str, Any]] = []
    if not _same_path(locked_path, config.subgraphs_manifest_path):
        errors.append(
            _error(
                "subgraph_manifest_path_mismatch",
                "configured patient-subgraph manifest differs from the contract lock",
                artifact_name="patient_subgraphs_manifest",
            )
        )
        return errors
    try:
        matches = (
            config.subgraphs_manifest_path.exists()
            and sha256_file(config.subgraphs_manifest_path) == locked_hash
        )
    except OSError:
        matches = False
    if not matches:
        errors.append(
            _error(
                "subgraph_manifest_changed",
                "patient-subgraph manifest is missing or changed from its lock",
                artifact_name="patient_subgraphs_manifest",
            )
        )
    return errors


def _subgraph_manifest_errors(
    config: GNNTrainingConfig,
) -> list[dict[str, Any]]:
    payload, errors = _load_document(
        config.subgraphs_manifest_path,
        invalid_code="invalid_subgraph_manifest",
        artifact_name="patient_subgraphs_manifest",
    )
    if payload is None:
        return errors
    if payload.get("status") != "completed":
        errors.append(
            _error(
                "subgraph_manifest_not_completed",
                "patient-subgraph manifest status must be completed",
                artifact_name="patient_subgraphs_manifest",
            )
        )
    scopes = payload.get("graph_fit_scope")
    scope_valid = (
        isinstance(scopes, list)
        and bool(scopes)
        and all(
            isinstance(row, dict)
            and row.get("fit_source") == "mimiciv"
            and row.get("fit_split") == "train"
            for row in scopes
        )
    )
    if not scope_valid:
        errors.append(
            _error(
                "subgraph_fit_scope_mismatch",
                "patient-subgraph graph fit scope must be exclusively MIMIC-IV train",
                artifact_name="patient_subgraphs_manifest",
            )
        )

    versions = payload.get("versions")
    if not isinstance(versions, dict):
        errors.append(
            _error(
                "subgraph_versions_missing",
                "patient-subgraph manifest must declare pinned versions",
                artifact_name="patient_subgraphs_manifest",
            )
        )
    else:
        for name, expected in PINNED_VERSIONS.items():
            if versions.get(name) != expected:
                errors.append(
                    _error(
                        "subgraph_version_mismatch",
                        "patient-subgraph version does not match the training lock",
                        artifact_name=name,
                    )
                )
    return errors


def _artifact_lock_errors(
    *,
    selection: Mapping[str, Any],
    expected_paths: Mapping[str, Path],
    branch_name: str,
) -> list[dict[str, Any]]:
    """Verify an exact selection path and SHA-256 lock for each artifact."""

    locks = selection.get("frozen_artifacts")
    if not isinstance(locks, dict):
        return [
            _error(
                f"missing_frozen_{branch_name}_artifacts",
                "frozen selection does not contain artifact locks",
            )
        ]

    errors: list[dict[str, Any]] = []
    for name, expected_path in expected_paths.items():
        lock = locks.get(name)
        if not isinstance(lock, dict):
            errors.append(
                _error(
                    f"missing_frozen_{branch_name}_artifact",
                    "required frozen artifact lock is missing",
                    artifact_name=name,
                )
            )
            continue
        try:
            locked_path = Path(lock["path"])
            locked_hash = str(lock["sha256"])
        except (KeyError, TypeError, ValueError):
            errors.append(
                _error(
                    f"invalid_frozen_{branch_name}_artifact",
                    "frozen artifact lock must contain path and sha256",
                    artifact_name=name,
                )
            )
            continue
        if not _same_path(locked_path, expected_path):
            errors.append(
                _error(
                    f"frozen_{branch_name}_artifact_path_mismatch",
                    "frozen artifact lock path does not match the configured path",
                    artifact_name=name,
                )
            )
            continue
        try:
            matches = (
                expected_path.exists() and sha256_file(expected_path) == locked_hash
            )
        except OSError:
            matches = False
        if not matches:
            errors.append(
                _error(
                    f"frozen_{branch_name}_artifact_changed",
                    "frozen artifact is missing or its SHA-256 digest changed",
                    artifact_name=name,
                )
            )
    return errors


def _neural_selection_errors(
    config: GNNTrainingConfig,
    *,
    contract_digest: str | None,
) -> list[dict[str, Any]]:
    payload, errors = _load_document(
        config.neural_selection_path,
        invalid_code="invalid_neural_selection",
        artifact_name="neural_selection",
    )
    if payload is None:
        return errors
    if payload.get("status") != "frozen" or payload.get("model_frozen") is not True:
        errors.append(
            _error(
                "neural_selection_not_frozen",
                "GNN work requires status=frozen and model_frozen=true for the "
                "Transformer selection",
                artifact_name="neural_selection",
            )
        )
    if (
        contract_digest is not None
        and payload.get("contract_digest") != contract_digest
    ):
        errors.append(
            _error(
                "neural_selection_contract_mismatch",
                "frozen Transformer selection does not match the training contract",
                artifact_name="neural_selection",
            )
        )
    errors.extend(
        _artifact_lock_errors(
            selection=payload,
            expected_paths={
                "checkpoint": config.neural_checkpoint_path,
                "calibration": config.neural_calibration_path,
                "feature_layout": config.neural_feature_layout_path,
            },
            branch_name="neural",
        )
    )
    return errors


def _component_excluded(fold: Mapping[str, Any], component: str) -> bool:
    direct = fold.get(f"{component}_fit_excludes_held_out_fold")
    if direct is True:
        return True
    proof = fold.get("exclusion_proof")
    if isinstance(proof, dict) and proof.get(component) is True:
        return True
    scopes = fold.get("fit_scopes")
    if isinstance(scopes, dict):
        component_scope = scopes.get(component)
        if (
            isinstance(component_scope, dict)
            and component_scope.get("held_out_fold_excluded") is True
        ):
            return True
    return False


def _crossfit_graph_errors(
    config: GNNTrainingConfig,
    *,
    contract_digest: str | None,
) -> list[dict[str, Any]]:
    """Require fold-specific graph/vocabulary/support exclusion evidence."""

    if not config.crossfit_graph_manifest_path.exists():
        return [
            _error(
                "missing_crossfit_graph_contract",
                "model selection requires a completed patient-grouped cross-fit "
                "graph contract; the full-train graph alone is invalid",
                artifact_name="crossfit_graph_manifest",
            )
        ]
    payload, errors = _load_document(
        config.crossfit_graph_manifest_path,
        invalid_code="invalid_crossfit_graph_contract",
        artifact_name="crossfit_graph_manifest",
    )
    if payload is None:
        return errors

    if payload.get("status") != "completed":
        errors.append(
            _error(
                "invalid_crossfit_graph_contract",
                "cross-fit graph contract status must be completed",
                artifact_name="crossfit_graph_manifest",
            )
        )
    if payload.get("schema_version") != CROSS_FIT_SCHEMA_VERSION:
        errors.append(
            _error(
                "invalid_crossfit_graph_contract",
                "cross-fit graph contract schema version does not match",
                artifact_name="crossfit_graph_manifest",
            )
        )
    if (
        contract_digest is not None
        and payload.get("contract_digest") != contract_digest
    ):
        errors.append(
            _error(
                "crossfit_contract_digest_mismatch",
                "cross-fit graph contract does not match the training contract",
                artifact_name="crossfit_graph_manifest",
            )
        )
    if (
        payload.get("seed") != config.seed
        or payload.get("fold_count") != config.fold_count
    ):
        errors.append(
            _error(
                "crossfit_graph_scope_mismatch",
                "cross-fit graph contract seed/fold count differs from configuration",
                artifact_name="crossfit_graph_manifest",
            )
        )
    scope = payload.get("fit_scope")
    scope_valid = (
        isinstance(scope, dict)
        and scope.get("source") == "mimiciv"
        and scope.get("split") == "train"
        and scope.get("grouping_unit") == "patient_uid"
        and payload.get("patient_grouped") is True
    )
    if not scope_valid:
        errors.append(
            _error(
                "crossfit_graph_scope_mismatch",
                "cross-fit graph/vocabulary/support fitting must use "
                "patient-grouped MIMIC-IV train folds",
                artifact_name="crossfit_graph_manifest",
            )
        )

    folds = payload.get("folds")
    if not isinstance(folds, list):
        errors.append(
            _error(
                "invalid_crossfit_graph_contract",
                "cross-fit graph contract must contain aggregate fold records",
                artifact_name="crossfit_graph_manifest",
            )
        )
        return errors

    expected_indices = set(range(config.fold_count))
    observed_indices = {
        row.get("fold_index")
        for row in folds
        if isinstance(row, dict) and isinstance(row.get("fold_index"), int)
    }
    if len(folds) != config.fold_count or observed_indices != expected_indices:
        errors.append(
            _error(
                "crossfit_fold_coverage_mismatch",
                "cross-fit graph contract must cover every configured held-out fold "
                "exactly once",
                artifact_name="crossfit_graph_manifest",
            )
        )

    for fold in folds:
        if not isinstance(fold, dict) or not isinstance(fold.get("fold_index"), int):
            continue
        fold_index = int(fold["fold_index"])
        expected_fit_folds = sorted(expected_indices - {fold_index})
        held_out_index = fold.get("held_out_fold_index", fold.get("held_out_fold"))
        fit_fold_indices = fold.get("fit_fold_indices")
        fit_fold_indices_valid = isinstance(fit_fold_indices, list) and all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in fit_fold_indices
        )
        scope_matches = (
            held_out_index == fold_index
            and fit_fold_indices_valid
            and sorted(fit_fold_indices) == expected_fit_folds
            and fold.get("fit_source") == "mimiciv"
            and fold.get("fit_split") == "train"
        )
        if not scope_matches:
            errors.append(
                _error(
                    "crossfit_graph_scope_mismatch",
                    "held-out and fit-fold scope is incomplete or inconsistent",
                    artifact_name="crossfit_graph_manifest",
                    fold_index=fold_index,
                )
            )
        try:
            overlap_count = int(fold.get("patient_overlap_count"))
            held_out_count = int(fold.get("held_out_patient_count"))
            fit_count = int(fold.get("fit_patient_count"))
        except (TypeError, ValueError):
            overlap_count = -1
            held_out_count = 0
            fit_count = 0
        if overlap_count != 0 or held_out_count <= 0 or fit_count <= 0:
            errors.append(
                _error(
                    "crossfit_patient_overlap",
                    "each fold must prove non-empty patient groups and zero "
                    "fit/held-out patient overlap",
                    artifact_name="crossfit_graph_manifest",
                    fold_index=fold_index,
                )
            )
        for component in ("graph", "vocabulary", "support"):
            if not _component_excluded(fold, component):
                errors.append(
                    _error(
                        "crossfit_heldout_not_excluded",
                        "held-out patient fold must be excluded from graph, "
                        "vocabulary, and support fitting",
                        artifact_name=component,
                        fold_index=fold_index,
                    )
                )
    if not errors:
        from pipeline.gnn_training.crossfit import crossfit_artifact_lock_errors

        errors.extend(crossfit_artifact_lock_errors(config))
    return errors


def _relation_vocabulary_errors(
    config: GNNTrainingConfig,
) -> list[dict[str, Any]]:
    if not config.relation_vocabulary_path.exists():
        return []
    payload, errors = _load_document(
        config.relation_vocabulary_path,
        invalid_code="invalid_relation_vocabulary",
        artifact_name="relation_vocabulary",
    )
    if payload is None:
        return errors
    relations = payload.get("relations")
    mapping = payload.get("relation_to_index")
    relations_match = relations == list(RELATION_TYPES)
    mapping_match = mapping == RELATION_TO_INDEX
    if not relations_match or not mapping_match:
        errors.append(
            _error(
                "relation_vocabulary_mismatch",
                "relation vocabulary must use the stable five forward, five "
                "reverse, and self-loop ordering",
                artifact_name="relation_vocabulary",
            )
        )
    return errors


def _prepare_manifest_errors(
    config: GNNTrainingConfig,
    *,
    stage: str,
) -> list[dict[str, Any]]:
    payload, errors = _load_document(
        config.prepare_manifest_path,
        invalid_code="invalid_prepare_manifest",
        artifact_name="prepare_manifest",
    )
    if payload is None:
        return errors

    status = payload.get("status")
    components = payload.get("components")
    graph_complete = (
        isinstance(components, dict) and components.get("graph_cache") == "completed"
    )
    transformer_complete = (
        isinstance(components, dict)
        and components.get("frozen_transformer_cache") == "completed"
    )
    if status not in {"completed", PREPARE_PENDING_STATUS} or not graph_complete:
        errors.append(
            _error(
                "prepared_graph_cache_not_completed",
                "stage requires a completed prepared graph-cache component",
                artifact_name="prepare_manifest",
            )
        )
    if stage in FUSION_STAGES and (
        status != "completed"
        or payload.get("preparation_complete") is not True
        or not transformer_complete
    ):
        errors.append(
            _error(
                "frozen_transformer_cache_not_completed",
                "fusion requires completed frozen Transformer cache extraction",
                artifact_name="prepare_manifest",
            )
        )

    if payload.get("scope") != FULL_TRAIN_REFIT_SCOPE:
        errors.append(
            _error(
                "prepare_scope_mismatch",
                "full prepared graph cache must be labeled full_train_refit_only",
                artifact_name="prepare_manifest",
            )
        )
    if payload.get("selection_eligible") is not False:
        errors.append(
            _error(
                "unsafe_full_train_selection_claim",
                "the full-train graph cache must explicitly record "
                "selection_eligible=false",
                artifact_name="prepare_manifest",
            )
        )
    return errors


def _cache_manifest_errors(
    config: GNNTrainingConfig,
) -> list[dict[str, Any]]:
    payload, errors = _load_document(
        config.cache_manifest_path,
        invalid_code="invalid_cache_manifest",
        artifact_name="cache_manifest",
    )
    if payload is None:
        return errors
    if (
        payload.get("status") != "completed"
        or payload.get("shard_count") != config.shard_count
        or payload.get("scope") != FULL_TRAIN_REFIT_SCOPE
        or payload.get("selection_eligible") is not False
    ):
        errors.append(
            _error(
                "invalid_cache_manifest",
                "cache manifest must be completed, use the configured shard count, "
                "and remain full-train-refit-only/non-selection-eligible",
                artifact_name="cache_manifest",
            )
        )
    from pipeline.gnn_training.data import (
        GRAPH_CACHE_ARTIFACT_LOCK_VERSION,
        artifact_tree_digest,
        graph_cache_artifact_hashes,
    )

    actual_hashes = graph_cache_artifact_hashes(config.gnn_root)
    if (
        payload.get("artifact_lock_version") != GRAPH_CACHE_ARTIFACT_LOCK_VERSION
        or payload.get("artifact_hashes") != actual_hashes
        or payload.get("artifact_tree_digest") != artifact_tree_digest(actual_hashes)
    ):
        errors.append(
            _error(
                "invalid_graph_cache_artifact_lock",
                "full-refit graph cache files do not match their exact content lock",
                artifact_name="cache_manifest",
            )
        )
    cached_splits = payload.get("cached_splits")
    if not config.allow_ungated and cached_splits != sorted(config.evaluation_splits()):
        errors.append(
            _error(
                "incomplete_graph_cache_splits",
                "production graph preparation must cache train, validation, and "
                "test before model selection",
                artifact_name="cache_manifest",
            )
        )
    return errors


def _selection_errors(
    config: GNNTrainingConfig,
    *,
    selection_path: Path,
    branch_name: str,
    contract_digest: str | None,
    expected_paths: Mapping[str, Path],
    require_cli_flag: bool,
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    if require_cli_flag and not config.frozen_selection:
        errors.append(
            _error(
                "final_requires_frozen_selection",
                "final scoring requires --frozen-selection",
            )
        )
    if not selection_path.exists():
        errors.append(
            _error(
                f"missing_{branch_name}_selection",
                "required frozen development selection report is missing",
                artifact_name=f"{branch_name}_selection",
            )
        )
        return errors
    payload, load_errors = _load_document(
        selection_path,
        invalid_code=f"invalid_{branch_name}_selection",
        artifact_name=f"{branch_name}_selection",
    )
    errors.extend(load_errors)
    if payload is None:
        return errors
    if payload.get("status") != "frozen" or payload.get("model_frozen") is not True:
        errors.append(
            _error(
                f"{branch_name}_selection_not_frozen",
                "selection must record status=frozen and model_frozen=true",
                artifact_name=f"{branch_name}_selection",
            )
        )
    if (
        contract_digest is not None
        and payload.get("contract_digest") != contract_digest
    ):
        errors.append(
            _error(
                f"{branch_name}_selection_contract_mismatch",
                "frozen selection does not match the training contract",
                artifact_name=f"{branch_name}_selection",
            )
        )
    errors.extend(
        _artifact_lock_errors(
            selection=payload,
            expected_paths=expected_paths,
            branch_name=branch_name,
        )
    )
    return errors


def preflight_errors(
    config: GNNTrainingConfig,
    *,
    stage: str,
) -> list[dict[str, Any]]:
    """Return all aggregate fail-closed errors for one pipeline stage."""

    if stage not in STAGES:
        raise ValueError(f"unknown stage: {stage!r}")

    errors = path_safety_errors(config)
    contract_digest: str | None = None
    contract_lock: dict[str, Any] | None = None

    if not config.allow_ungated:
        for name, path in required_inputs(config).items():
            if not path.exists():
                errors.append(
                    _error(
                        "missing_input",
                        "required upstream artifact is missing",
                        artifact_name=name,
                    )
                )

        if config.contract_lock_path.exists():
            contract_digest, contract_lock, contract_errors = _training_contract_errors(
                config
            )
            errors.extend(contract_errors)
        if config.subgraphs_manifest_path.exists():
            errors.extend(_subgraph_manifest_errors(config))
        errors.extend(_locked_subgraph_manifest_errors(config, contract_lock))
        if config.neural_selection_path.exists():
            errors.extend(
                _neural_selection_errors(
                    config,
                    contract_digest=contract_digest,
                )
            )
        if stage in CROSS_FIT_REQUIRED_STAGES:
            errors.extend(
                _crossfit_graph_errors(
                    config,
                    contract_digest=contract_digest,
                )
            )

    if stage != "prepare":
        errors.extend(_prepare_manifest_errors(config, stage=stage))
        for name, path in required_stage_artifacts(config, stage=stage).items():
            if not path.exists():
                errors.append(
                    _error(
                        "missing_stage_artifact",
                        "required prepared/checkpoint artifact is missing",
                        artifact_name=name,
                    )
                )
        errors.extend(_relation_vocabulary_errors(config))
        if config.cache_manifest_path.exists():
            errors.extend(_cache_manifest_errors(config))
    if (
        stage == "score-gnn"
        and not config.allow_ungated
        and (
            not config.graph_reference_report_path.is_file()
            or not config.graph_reference_scores_path.is_file()
        )
    ):
        errors.append(
            _error(
                "missing_graph_only_reference",
                "standalone GNN qualification requires both the completed "
                "aggregate graph-only evaluation report and its exact score table",
                artifact_name="graph_only_xgboost",
            )
        )

    gnn_selection_paths = {
        "checkpoint": config.gnn_checkpoint_path,
        "calibration": config.gnn_calibration_path,
        "feature_layout": config.feature_layout_path,
        "training_state": config.gnn_training_state_path,
        "oof_predictions": config.gnn_oof_predictions_path,
        "crossfit_manifest": config.crossfit_graph_manifest_path,
        "cache_manifest": config.cache_manifest_path,
        "graph_reference_report": config.graph_reference_report_path,
    }
    gnn_selection_paths["graph_reference_scores"] = config.graph_reference_scores_path
    fusion_selection_paths = {
        "checkpoint": config.fusion_checkpoint_path,
        "calibration": config.fusion_calibration_path,
        "feature_layout": config.feature_layout_path,
        "training_state": config.fusion_training_state_path,
        "gnn_checkpoint": config.gnn_checkpoint_path,
        "gnn_selection": config.gnn_selection_report_path,
        "transformer_checkpoint": config.neural_checkpoint_path,
        "transformer_calibration": config.neural_calibration_path,
        "transformer_feature_layout": config.neural_feature_layout_path,
        "transformer_cache_manifest": config.transformer_cache_manifest_path,
        "crossfit_manifest": config.crossfit_graph_manifest_path,
    }

    if stage in FUSION_STAGES and not config.allow_ungated:
        errors.extend(
            _selection_errors(
                config,
                selection_path=config.gnn_selection_report_path,
                branch_name="gnn",
                contract_digest=contract_digest,
                expected_paths=gnn_selection_paths,
                require_cli_flag=False,
            )
        )
        from pipeline.gnn_training.data import _transformer_cache_status

        if _transformer_cache_status(config) != "completed":
            errors.append(
                _error(
                    "invalid_frozen_transformer_cache",
                    "fusion requires a physically reconciled frozen Transformer "
                    "cache with unchanged hashes",
                    artifact_name="frozen_transformer_cache",
                )
            )

    if config.mode == "final" and stage == "score-gnn":
        errors.extend(
            _selection_errors(
                config,
                selection_path=config.gnn_selection_report_path,
                branch_name="gnn",
                contract_digest=contract_digest,
                expected_paths=gnn_selection_paths,
                require_cli_flag=True,
            )
        )
        if config.gnn_selection_report_path.exists():
            selection, _selection_load_errors = _load_document(
                config.gnn_selection_report_path,
                invalid_code="invalid_gnn_selection",
                artifact_name="gnn_selection",
            )
            if (
                selection is not None
                and selection.get("standalone_qualified") is not True
            ):
                errors.append(
                    _error(
                        "standalone_gnn_not_qualified",
                        "final standalone GNN scoring is blocked because the "
                        "development gate did not qualify it",
                        artifact_name="gnn_selection",
                    )
                )
    if config.mode == "final" and stage == "score-fusion":
        errors.extend(
            _selection_errors(
                config,
                selection_path=config.fusion_selection_report_path,
                branch_name="hybrid",
                contract_digest=contract_digest,
                expected_paths=fusion_selection_paths,
                require_cli_flag=True,
            )
        )
        if config.fusion_selection_report_path.exists():
            selection, _selection_load_errors = _load_document(
                config.fusion_selection_report_path,
                invalid_code="invalid_hybrid_selection",
                artifact_name="hybrid_selection",
            )
            if selection is not None and selection.get("hybrid_qualified") is not True:
                errors.append(
                    _error(
                        "hybrid_not_qualified",
                        "final hybrid scoring is blocked because the development "
                        "gate did not qualify it",
                        artifact_name="hybrid_selection",
                    )
                )

    if stage in {"prepare", "train-gnn"} and config.gnn_selection_report_path.exists():
        selection, _load_errors = _load_document(
            config.gnn_selection_report_path,
            invalid_code="invalid_gnn_selection",
            artifact_name="gnn_selection",
        )
        if selection is not None and selection.get("status") == "frozen":
            errors.append(
                _error(
                    "frozen_gnn_overwrite_blocked",
                    "prepare/refit cannot overwrite a frozen GNN selection",
                    artifact_name="gnn_selection",
                )
            )
    if (
        stage == "score-gnn"
        and config.mode == "development"
        and config.gnn_selection_report_path.exists()
    ):
        selection, _load_errors = _load_document(
            config.gnn_selection_report_path,
            invalid_code="invalid_gnn_selection",
            artifact_name="gnn_selection",
        )
        if selection is not None and selection.get("status") == "frozen":
            errors.append(
                _error(
                    "frozen_gnn_selection_overwrite_blocked",
                    "development scoring cannot overwrite a frozen GNN selection",
                    artifact_name="gnn_selection",
                )
            )
    if (
        stage in {"prepare", "train-fusion"}
        and config.fusion_selection_report_path.exists()
    ):
        selection, _load_errors = _load_document(
            config.fusion_selection_report_path,
            invalid_code="invalid_hybrid_selection",
            artifact_name="hybrid_selection",
        )
        if selection is not None and selection.get("status") == "frozen":
            errors.append(
                _error(
                    "frozen_hybrid_overwrite_blocked",
                    "prepare/refit cannot overwrite a frozen hybrid selection",
                    artifact_name="hybrid_selection",
                )
            )
    if (
        stage == "score-fusion"
        and config.mode == "development"
        and config.fusion_selection_report_path.exists()
    ):
        selection, _load_errors = _load_document(
            config.fusion_selection_report_path,
            invalid_code="invalid_hybrid_selection",
            artifact_name="hybrid_selection",
        )
        if selection is not None and selection.get("status") == "frozen":
            errors.append(
                _error(
                    "frozen_hybrid_selection_overwrite_blocked",
                    "development scoring cannot overwrite a frozen hybrid selection",
                    artifact_name="hybrid_selection",
                )
            )
    return errors


def contract_digest_or_none(config: GNNTrainingConfig) -> str | None:
    """Return a completed, correctly version-pinned contract digest."""

    if not config.contract_lock_path.exists():
        return None
    digest, _payload, errors = _training_contract_errors(config)
    return digest if not errors else None


def blocked_report(
    *,
    config: GNNTrainingConfig,
    schema_version: str,
    stage: str,
    generated_at: str,
    errors: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return a standard aggregate-only blocked-preflight report body."""

    return {
        "schema_version": schema_version,
        "status": "blocked_preflight",
        "stage": stage,
        "mode": config.mode,
        "generated_at": generated_at,
        "errors": [dict(error) for error in errors],
        "gate_policy": config.gate_policy(),
        "data_safety": {
            "report_contains_patient_rows": False,
            "report_contains_row_samples": False,
            "report_contains_identifier_values": False,
        },
    }
