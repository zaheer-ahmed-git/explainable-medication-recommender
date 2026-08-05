"""Torch-free configuration for the Phase 8 P0 GNN and fusion stages.

Restricted caches, checkpoints, and predictions are rooted under
``$DATASET_ROOT/processed/phase8_p0/gnn``.  Only aggregate reports are rooted
under ``REPORTS_ROOT``.  PyTorch is intentionally not imported here so CLI
parsing, preparation, and contract checks work without the optional ``neural``
dependency group.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pipeline.config import (
    DATASET_ROOT,
    DUCKDB_MEMORY_LIMIT,
    DUCKDB_TEMP_DIR,
    DUCKDB_THREADS,
    PROCESSED_DATA_ROOT,
    RANDOM_SEED,
    REPORTS_ROOT,
)

# ---------------------------------------------------------------------------
# Stable schemas, experiments, and typed graph vocabulary
# ---------------------------------------------------------------------------
PREPARE_SCHEMA_VERSION = "phase8-p0-gnn-prepare-manifest-v1"
CROSS_FIT_SCHEMA_VERSION = "phase8-p0-gnn-crossfit-graph-manifest-v1"
GNN_TRAINING_SCHEMA_VERSION = "phase8-p0-gnn-training-evaluation-v1"
GNN_SELECTION_SCHEMA_VERSION = "phase8-p0-gnn-training-selection-v1"
FUSION_TRAINING_SCHEMA_VERSION = "phase8-p0-fusion-training-evaluation-v1"
FUSION_SELECTION_SCHEMA_VERSION = "phase8-p0-fusion-training-selection-v1"
FEATURE_LAYOUT_VERSION = "phase8-p1-gnn-feature-layout-v2"
RELATION_VOCABULARY_VERSION = "phase8-p1-gnn-relation-vocabulary-v2"
GNN_EXPERIMENT_VERSION = "phase8-p1-native-rgcn-v2"
FUSION_EXPERIMENT_VERSION = "phase8-p0-frozen-transformer-fusion-v1"
PREPARE_PENDING_STATUS = "pending_required_component"
FULL_TRAIN_REFIT_SCOPE = "full_train_refit_only"
CROSS_FIT_SELECTION_SCOPE = "patient_grouped_crossfit_selection"

GNN_BASELINE_NAME = "gnn_relation_rgcn"
TRANSFORMER_BASELINE_NAME = "transformer_patient_context"
LATE_FUSION_BASELINE_NAME = "hybrid_late_fusion"
RESIDUAL_FUSION_BASELINE_NAME = "hybrid_residual"
GRAPH_REFERENCE_BASELINE_NAME = "graph_only_xgboost"

NODE_TYPES = ("condition", "medication", "lab", "vital", "intervention", "stay")
NODE_TYPE_TO_INDEX = {name: index for index, name in enumerate(NODE_TYPES)}
NODE_ROLES = (
    "query_condition",
    "candidate_medication",
    "observed_context",
    "stay_query",
)
NODE_ROLE_TO_INDEX = {name: index for index, name in enumerate(NODE_ROLES)}
PAD_INDEX = 0
UNK_INDEX = 1
RESERVED_CONCEPT_TOKEN_COUNT = 2

FORWARD_RELATION_TYPES = (
    "condition_medication_train_positive",
    "condition_lab_predecision",
    "condition_vital_predecision",
    "condition_intervention_predecision",
    "medication_medication_train_coprescribed",
    "stay_index_condition",
    "stay_context_observed",
)
REVERSE_RELATION_TYPES = tuple(
    f"reverse_{relation}" for relation in FORWARD_RELATION_TYPES
)
SELF_LOOP_RELATION = "self_loop"
RELATION_TYPES = (
    *FORWARD_RELATION_TYPES,
    *REVERSE_RELATION_TYPES,
    SELF_LOOP_RELATION,
)
RELATION_TO_INDEX = {relation: index for index, relation in enumerate(RELATION_TYPES)}

PRIMARY_SEED = RANDOM_SEED
DEFAULT_SHARD_COUNT = 256
DEFAULT_FOLD_COUNT = 5
DEFAULT_TRANSFORMER_CONTEXT_DIM = 256
SELECTION_K = 10
MINIMUM_NDCG_LIFT = 0.005
MAXIMUM_SECONDARY_DROP = 0.01

PHASE8_P0_ROOT = PROCESSED_DATA_ROOT / "phase8_p0"
DEFAULT_GNN_ROOT = PHASE8_P0_ROOT / "gnn"
DEFAULT_NEURAL_ROOT = PHASE8_P0_ROOT / "neural"
DEFAULT_GRAPH_ROOT = PHASE8_P0_ROOT / "graph" / "milestone8"
DEFAULT_SUBGRAPHS_ROOT = DEFAULT_GRAPH_ROOT / "patient_subgraphs"
DEFAULT_TRAINING_ROOT = PHASE8_P0_ROOT / "training"
DEFAULT_FEATURES_ROOT = PHASE8_P0_ROOT / "features"
DEFAULT_GRAPH_REFERENCE_SCORES = (
    PHASE8_P0_ROOT / "evaluation" / "milestone8b" / "graph_ablation_scores.parquet"
)

DEFAULT_CONTRACT_LOCK = REPORTS_ROOT / "phase8_p0_training_contract_lock.json"
DEFAULT_SUBGRAPHS_MANIFEST = REPORTS_ROOT / "phase8_p0_patient_subgraphs_manifest.json"
DEFAULT_NEURAL_SELECTION = REPORTS_ROOT / "phase8_p0_neural_training_selection.json"
DEFAULT_PREPARE_MANIFEST = REPORTS_ROOT / "phase8_p0_gnn_prepare_manifest.json"
DEFAULT_CROSS_FIT_GRAPH_MANIFEST = (
    REPORTS_ROOT / "phase8_p0_gnn_crossfit_graph_manifest.json"
)
DEFAULT_GNN_TRAINING_REPORT = REPORTS_ROOT / "phase8_p0_gnn_training_evaluation.json"
DEFAULT_GNN_SCORE_REPORT = REPORTS_ROOT / "phase8_p0_gnn_score_evaluation.json"
DEFAULT_GNN_SELECTION_REPORT = REPORTS_ROOT / "phase8_p0_gnn_training_selection.json"
DEFAULT_FUSION_TRAINING_REPORT = (
    REPORTS_ROOT / "phase8_p0_fusion_training_evaluation.json"
)
DEFAULT_FUSION_SCORE_REPORT = REPORTS_ROOT / "phase8_p0_fusion_score_evaluation.json"
DEFAULT_FUSION_SELECTION_REPORT = (
    REPORTS_ROOT / "phase8_p0_fusion_training_selection.json"
)
DEFAULT_GRAPH_REFERENCE_REPORT = (
    REPORTS_ROOT / "phase8_p0_milestone8b_ablation_evaluation.json"
)


@dataclass(frozen=True)
class GNNArchitecture:
    """Small native-PyTorch relation-aware encoder defaults."""

    concept_embedding_dim: int = 128
    node_type_embedding_dim: int = 16
    node_role_embedding_dim: int = 16
    time_bin_embedding_dim: int = 8
    node_continuous_dim: int = 5
    time_bin_count: int = 5
    hidden_dim: int = 128
    relation_layers: int = 2
    relation_count: int = len(RELATION_TYPES)
    dropout: float = 0.2
    relation_dropout: float = 0.1
    scorer_hidden_dim: int = 128
    transformer_context_dim: int = DEFAULT_TRANSFORMER_CONTEXT_DIM
    fusion_hidden_dim: int = 128

    @property
    def num_relations(self) -> int:
        """Return the stable forward/reverse/self-loop relation count."""

        return self.relation_count


@dataclass(frozen=True)
class GNNOptimization:
    """Pre-registered GNN/fusion optimization defaults."""

    optimizer_name: str = "AdamW"
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    max_epochs: int = 30
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 1e-4
    batch_ranking_groups: int = 32
    gradient_accumulation_groups: int = 32
    max_edges_per_batch: int | None = 100_000
    max_nodes_per_batch: int | None = 8_192
    progress_interval_batches: int = 100
    primary_positive_weight: float = 0.5
    auxiliary_bce_weight: float = 0.05
    mixed_precision: bool = True
    precision: str = "bf16"


@dataclass(frozen=True)
class GNNTrainingConfig:
    """Configuration shared by prepare, standalone GNN, and fusion stages."""

    dataset_root: Path = DATASET_ROOT
    reports_root: Path = REPORTS_ROOT
    graph_root: Path = DEFAULT_GRAPH_ROOT
    subgraphs_root: Path = DEFAULT_SUBGRAPHS_ROOT
    features_root: Path = DEFAULT_FEATURES_ROOT
    training_root: Path = DEFAULT_TRAINING_ROOT
    neural_root: Path = DEFAULT_NEURAL_ROOT
    gnn_root: Path = DEFAULT_GNN_ROOT
    graph_reference_scores_path: Path = DEFAULT_GRAPH_REFERENCE_SCORES
    graph_reference_report_path: Path = DEFAULT_GRAPH_REFERENCE_REPORT

    contract_lock_path: Path = DEFAULT_CONTRACT_LOCK
    subgraphs_manifest_path: Path = DEFAULT_SUBGRAPHS_MANIFEST
    neural_selection_path: Path = DEFAULT_NEURAL_SELECTION
    prepare_manifest_path: Path = DEFAULT_PREPARE_MANIFEST
    crossfit_graph_manifest_path: Path = DEFAULT_CROSS_FIT_GRAPH_MANIFEST
    gnn_training_report_path: Path = DEFAULT_GNN_TRAINING_REPORT
    gnn_score_report_path: Path = DEFAULT_GNN_SCORE_REPORT
    gnn_selection_report_path: Path = DEFAULT_GNN_SELECTION_REPORT
    fusion_training_report_path: Path = DEFAULT_FUSION_TRAINING_REPORT
    fusion_score_report_path: Path = DEFAULT_FUSION_SCORE_REPORT
    fusion_selection_report_path: Path = DEFAULT_FUSION_SELECTION_REPORT

    mode: str = "development"
    frozen_selection: bool = False
    allow_ungated: bool = False
    seed: int = PRIMARY_SEED
    fold_count: int = DEFAULT_FOLD_COUNT
    shard_count: int = DEFAULT_SHARD_COUNT
    top_k: tuple[int, ...] = (1, 3, 5, 10)
    device: str | None = None

    architecture: GNNArchitecture = field(default_factory=GNNArchitecture)
    optimization: GNNOptimization = field(default_factory=GNNOptimization)

    duckdb_temp_directory: Path | None = DUCKDB_TEMP_DIR
    duckdb_memory_limit: str | None = DUCKDB_MEMORY_LIMIT
    duckdb_max_temp_directory_size: str | None = None
    duckdb_threads: int | None = DUCKDB_THREADS

    @property
    def expected_gnn_root(self) -> Path:
        """Return the sole allowed restricted-output root."""

        return self.dataset_root / "processed" / "phase8_p0" / "gnn"

    @property
    def expected_neural_root(self) -> Path:
        """Return the immutable Transformer artifact root."""

        return self.dataset_root / "processed" / "phase8_p0" / "neural"

    # ---- immutable upstream patient-subgraph inputs -------------------------
    @property
    def subgraph_index_path(self) -> Path:
        return self.subgraphs_root / "subgraph_index.parquet"

    @property
    def subgraph_nodes_path(self) -> Path:
        return self.subgraphs_root / "subgraph_nodes.parquet"

    @property
    def subgraph_edges_path(self) -> Path:
        return self.subgraphs_root / "subgraph_edges.parquet"

    @property
    def subgraph_candidates_path(self) -> Path:
        return self.subgraphs_root / "subgraph_candidates.parquet"

    @property
    def patient_condition_medication_path(self) -> Path:
        return self.training_root / "patient_condition_medication.parquet"

    @property
    def event_sequences_path(self) -> Path:
        return self.features_root / "event_sequences.parquet"

    # ---- immutable frozen Transformer inputs -------------------------------
    @property
    def neural_checkpoint_path(self) -> Path:
        return self.neural_root / "checkpoints" / "transformer_recommender.pt"

    @property
    def neural_calibration_path(self) -> Path:
        return self.neural_root / "checkpoints" / "temperature_calibration.json"

    @property
    def neural_feature_layout_path(self) -> Path:
        return self.neural_root / "feature_layout.json"

    # ---- prepared GNN cache and vocabulary layout ---------------------------
    @property
    def cache_root(self) -> Path:
        return self.gnn_root / "cache"

    @property
    def shards_root(self) -> Path:
        return self.cache_root / "shards"

    @property
    def frozen_transformer_cache_root(self) -> Path:
        return self.cache_root / "frozen_transformer"

    @property
    def cache_manifest_path(self) -> Path:
        return self.cache_root / "cache_manifest.json"

    @property
    def transformer_cache_manifest_path(self) -> Path:
        return self.frozen_transformer_cache_root / "cache_manifest.json"

    @property
    def vocab_root(self) -> Path:
        return self.gnn_root / "vocab"

    @property
    def graph_node_vocabulary_path(self) -> Path:
        return self.vocab_root / "graph_node_vocabulary.parquet"

    @property
    def node_type_vocabulary_path(self) -> Path:
        return self.vocab_root / "node_type_vocabulary.json"

    @property
    def node_role_vocabulary_path(self) -> Path:
        return self.vocab_root / "node_role_vocabulary.json"

    @property
    def relation_vocabulary_path(self) -> Path:
        return self.vocab_root / "relation_vocabulary.json"

    @property
    def feature_layout_path(self) -> Path:
        return self.gnn_root / "feature_layout.json"

    @property
    def crossfit_root(self) -> Path:
        return self.gnn_root / "crossfit"

    def fold_graph_root(self, fold_index: int) -> Path:
        """Return one held-out-fold-specific graph fitting directory."""

        return self.crossfit_root / f"fold_{fold_index:02d}"

    def fold_feature_layout_path(self, fold_index: int) -> Path:
        return self.fold_graph_root(fold_index) / "feature_layout.json"

    def fold_cache_manifest_path(self, fold_index: int) -> Path:
        return self.fold_graph_root(fold_index) / "cache" / "cache_manifest.json"

    def fold_graph_edges_path(self, fold_index: int) -> Path:
        return self.fold_graph_root(fold_index) / "graph_edges.parquet"

    def fold_concept_vocabulary_path(self, fold_index: int) -> Path:
        return (
            self.fold_graph_root(fold_index) / "vocab" / "graph_node_vocabulary.parquet"
        )

    def fold_checkpoint_path(self, fold_index: int, variant: str) -> Path:
        return (
            self.checkpoints_root
            / "crossfit"
            / f"fold_{fold_index:02d}"
            / f"{variant}.pt"
        )

    def fold_completion_manifest_path(self, fold_index: int, variant: str) -> Path:
        """Return the aggregate-safe completed-fold sidecar manifest."""

        return self.fold_checkpoint_path(fold_index, variant).with_suffix(".json")

    def fold_residual_checkpoint_path(self, fold_index: int) -> Path:
        return (
            self.checkpoints_root
            / "crossfit"
            / f"fold_{fold_index:02d}"
            / "residual_fusion.pt"
        )

    def fold_resume_path(self, fold_index: int, variant: str) -> Path:
        """Return the mutable epoch-resume state outside immutable caches."""

        return (
            self.checkpoints_root
            / "resume"
            / f"fold_{fold_index:02d}"
            / f"{variant}.pt"
        )

    # ---- checkpoints --------------------------------------------------------
    @property
    def checkpoints_root(self) -> Path:
        return self.gnn_root / "checkpoints"

    @property
    def gnn_checkpoint_path(self) -> Path:
        return self.checkpoints_root / "gnn_relation_branch.pt"

    @property
    def gnn_calibration_path(self) -> Path:
        return self.checkpoints_root / "gnn_temperature_calibration.json"

    @property
    def gnn_training_state_path(self) -> Path:
        return self.checkpoints_root / "gnn_training_state.json"

    @property
    def gnn_crossfit_selection_path(self) -> Path:
        return self.checkpoints_root / "gnn_crossfit_selection.json"

    @property
    def gnn_final_score_completion_path(self) -> Path:
        return self.checkpoints_root / "gnn_final_score_completion.json"

    @property
    def fusion_checkpoint_path(self) -> Path:
        return self.checkpoints_root / "fusion_ranker.pt"

    @property
    def fusion_calibration_path(self) -> Path:
        return self.checkpoints_root / "fusion_temperature_calibration.json"

    @property
    def fusion_training_state_path(self) -> Path:
        return self.checkpoints_root / "fusion_training_state.json"

    @property
    def fusion_final_score_completion_path(self) -> Path:
        return self.checkpoints_root / "fusion_final_score_completion.json"

    # ---- restricted canonical-score outputs --------------------------------
    @property
    def predictions_root(self) -> Path:
        return self.gnn_root / "predictions"

    @property
    def oof_predictions_root(self) -> Path:
        return self.predictions_root / "oof"

    @property
    def gnn_oof_predictions_path(self) -> Path:
        return self.oof_predictions_root / "gnn_selected_variant.parquet"

    @property
    def residual_oof_predictions_path(self) -> Path:
        return self.oof_predictions_root / "residual_fusion.parquet"

    @property
    def gnn_score_root(self) -> Path:
        return self.predictions_root / "gnn" / self.mode

    @property
    def gnn_score_output_path(self) -> Path:
        return self.gnn_score_root / "baseline_scores.parquet"

    @property
    def active_gnn_score_report_path(self) -> Path:
        if self.mode == "development":
            return self.gnn_score_report_path
        return self.gnn_score_report_path.with_name(
            f"{self.gnn_score_report_path.stem}_final"
            f"{self.gnn_score_report_path.suffix}"
        )

    @property
    def fusion_score_root(self) -> Path:
        return self.predictions_root / "fusion" / self.mode

    @property
    def fusion_score_output_path(self) -> Path:
        return self.fusion_score_root / "baseline_scores.parquet"

    @property
    def active_fusion_score_report_path(self) -> Path:
        if self.mode == "development":
            return self.fusion_score_report_path
        return self.fusion_score_report_path.with_name(
            f"{self.fusion_score_report_path.stem}_final"
            f"{self.fusion_score_report_path.suffix}"
        )

    @property
    def neural_score_output_path(self) -> Path:
        mode_name = "development" if self.mode == "development" else "final"
        return self.neural_root / "predictions" / mode_name / "baseline_scores.parquet"

    def evaluation_splits(self) -> tuple[str, ...]:
        """Return immutable cache scopes; scoring still selects one mode split.

        Test rows are transformed with train-fit vocabularies and the already
        frozen Transformer during preparation, but no test metric or selection
        decision is computed before an authorized final scoring command.  This
        avoids an unsafe post-freeze cache rebuild.
        """

        return ("train", "validation", "test")

    def restricted_write_paths(self) -> tuple[Path, ...]:
        """Return representative local artifact writes validated by preflight."""

        return (
            self.cache_root,
            self.shards_root,
            self.frozen_transformer_cache_root,
            self.cache_manifest_path,
            self.transformer_cache_manifest_path,
            self.vocab_root,
            self.graph_node_vocabulary_path,
            self.node_type_vocabulary_path,
            self.node_role_vocabulary_path,
            self.relation_vocabulary_path,
            self.feature_layout_path,
            self.crossfit_root,
            self.checkpoints_root,
            self.gnn_checkpoint_path,
            self.gnn_calibration_path,
            self.gnn_training_state_path,
            self.gnn_crossfit_selection_path,
            self.gnn_final_score_completion_path,
            self.fusion_checkpoint_path,
            self.fusion_calibration_path,
            self.fusion_training_state_path,
            self.fusion_final_score_completion_path,
            self.predictions_root,
            self.oof_predictions_root,
            self.gnn_oof_predictions_path,
            self.residual_oof_predictions_path,
            self.gnn_score_output_path,
            self.fusion_score_output_path,
        )

    def aggregate_report_paths(self) -> tuple[Path, ...]:
        """Return every public aggregate report written by this package."""

        return (
            self.prepare_manifest_path,
            self.crossfit_graph_manifest_path,
            self.gnn_training_report_path,
            self.gnn_score_report_path,
            self.gnn_score_report_path.with_name(
                f"{self.gnn_score_report_path.stem}_final"
                f"{self.gnn_score_report_path.suffix}"
            ),
            self.gnn_selection_report_path,
            self.fusion_training_report_path,
            self.fusion_score_report_path,
            self.fusion_score_report_path.with_name(
                f"{self.fusion_score_report_path.stem}_final"
                f"{self.fusion_score_report_path.suffix}"
            ),
            self.fusion_selection_report_path,
        )

    def gate_policy(self) -> dict[str, object]:
        """Return explicit report metadata for the synthetic bypass policy."""

        return {
            "upstream_artifact_gates_enforced": not self.allow_ungated,
            "allow_ungated": self.allow_ungated,
            "allow_ungated_scope": (
                "synthetic_unit_tests_only" if self.allow_ungated else "not_enabled"
            ),
        }
