"""Configuration, paths, and fixed hyperparameters for neural training.

This module is deliberately free of any PyTorch import so it can be loaded for
CLI parsing, cache preparation, and preflight checks without the optional
``neural`` dependency group.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pipeline.config import (
    DUCKDB_MEMORY_LIMIT,
    DUCKDB_TEMP_DIR,
    DUCKDB_THREADS,
    PROCESSED_DATA_ROOT,
    RANDOM_SEED,
    REPORTS_ROOT,
)

# ---------------------------------------------------------------------------
# Schema and experiment identifiers
# ---------------------------------------------------------------------------
PREPARE_SCHEMA_VERSION = "phase8-p0-neural-prepare-manifest-v1"
TRAINING_SCHEMA_VERSION = "phase8-p0-neural-training-evaluation-v1"
SELECTION_SCHEMA_VERSION = "phase8-p0-neural-training-selection-v1"
EXPERIMENT_VERSION = "phase8-p0-neural-transformer-v3"
FEATURE_LAYOUT_VERSION = "phase8-p0-neural-feature-layout-v3"
# Base candidate-side features (always present). Stage-1-matched train-fit
# graph tabular columns are appended at prepare time (see GRAPH_SIDE_FEATURES).
PRIOR_SIDE_FEATURES = (
    "candidate_rank_feat",
    "global_prior",
    "condition_candidate_prior",
)
# Match the frozen Stage 1 recovery lock: context graph family at support 5,
# plus the direct-edge summaries that late fusion's graph-only branch carries.
# This is train-fit tabular graph signal only — not GNN message passing.
GRAPH_SUPPORT_THRESHOLD = 5
GRAPH_SIDE_FEATURES = (
    "graph_condition_medication_support_count",
    "graph_condition_medication_log_support",
    "graph_condition_medication_support_share",
    "graph_condition_total_medication_support",
    "graph_condition_medication_degree",
    "graph_condition_lab_degree",
    "graph_condition_vital_degree",
    "graph_condition_intervention_degree",
    "graph_condition_total_degree",
    "graph_condition_total_support",
    "graph_candidate_medication_degree",
    "graph_candidate_medication_support",
    "graph_candidate_coprescription_degree",
    "graph_candidate_coprescription_support",
    "graph_condition_in_graph",
    "graph_candidate_in_graph",
    "graph_direct_edge_present",
)
CANDIDATE_SIDE_FEATURES = PRIOR_SIDE_FEATURES + GRAPH_SIDE_FEATURES
CANDIDATE_SIDE_FEATURE_COUNT = len(CANDIDATE_SIDE_FEATURES)
PRIOR_SMOOTHING_ALPHA = 1.0

# The neural gate compares against the Stage 1 structured recovery winner
# recorded in ``phase8_p0_gate_recovery_selection.json`` (currently
# ``xgboost_rank_ndcg_oof_late_fusion``). The Milestone 8B XGBoost anchor
# (0.374899) remains the Stage 1 comparison baseline only; it is no longer the
# bar the Transformer must beat.
DEFAULT_STRUCTURED_REFERENCE_BASELINE = "xgboost_rank_ndcg_oof_late_fusion"
LEGACY_MILESTONE8B_XGBOOST_NDCG_AT_10 = 0.374899
MINIMUM_NDCG_LIFT = 0.005
MAXIMUM_SECONDARY_DROP = 0.01
SELECTION_K = 10

# Reserved token indexes. Train-derived vocabulary indexes are offset by two.
PAD_INDEX = 0
UNK_INDEX = 1
RESERVED_TOKEN_COUNT = 2

PRIMARY_SEED = RANDOM_SEED  # 20260617, per the plan's preselected primary run.

# Candidate sequence-length grid the plan permits; the default is selected on
# MIMIC-train folds during preparation but exposed here for reproducibility.
SEQUENCE_LENGTH_GRID = (64, 128, 256)

PHASE8_P0_ROOT = PROCESSED_DATA_ROOT / "phase8_p0"
DEFAULT_FEATURES_ROOT = PHASE8_P0_ROOT / "features"
DEFAULT_TRAINING_ROOT = PHASE8_P0_ROOT / "training"
DEFAULT_GRAPH_ROOT = PHASE8_P0_ROOT / "graph" / "milestone8"
DEFAULT_NEURAL_ROOT = PHASE8_P0_ROOT / "neural"
DEFAULT_GATE_RECOVERY_EVAL_ROOT = PHASE8_P0_ROOT / "evaluation" / "gate_recovery"
DEFAULT_REFERENCE_SCORES = DEFAULT_GATE_RECOVERY_EVAL_ROOT / "baseline_scores.parquet"
# Legacy Milestone 8B reference scores (kept for diagnostics / migration only).
DEFAULT_LEGACY_MILESTONE8B_SCORES = (
    PHASE8_P0_ROOT / "evaluation" / "milestone8b" / "_scores_reference.parquet"
)

DEFAULT_CONTRACT_LOCK = REPORTS_ROOT / "phase8_p0_training_contract_lock.json"
DEFAULT_GATE_SELECTION = REPORTS_ROOT / "phase8_p0_gate_recovery_selection.json"
DEFAULT_PREPARE_MANIFEST = REPORTS_ROOT / "phase8_p0_neural_prepare_manifest.json"
DEFAULT_TRAINING_REPORT = REPORTS_ROOT / "phase8_p0_neural_training_evaluation.json"
DEFAULT_SCORE_REPORT = REPORTS_ROOT / "phase8_p0_neural_score_evaluation.json"
DEFAULT_SELECTION_REPORT = REPORTS_ROOT / "phase8_p0_neural_training_selection.json"

DEFAULT_SHARD_COUNT = 8
DEFAULT_MAX_SEQUENCE_LENGTH = 128
PREDICTION_OFFSET_HOURS = 24


@dataclass(frozen=True)
class NeuralArchitecture:
    """Transformer patient/context architecture (v3 gap-recovery defaults).

    The GNN branch and joint fusion head remain documented extension points;
    this pipeline still trains the Transformer-only branch (plan Phase C).
    Tabular stay features use a residual MLP with feature dropout; candidate
    scoring fuses projected Stage-1-matched side features.
    """

    event_embedding_dim: int = 128
    encoder_layers: int = 2
    attention_heads: int = 4
    feedforward_dim: int = 256
    dropout: float = 0.3
    feature_dropout: float = 0.15
    categorical_embedding_dim: int = 16
    condition_embedding_dim: int = 64
    candidate_embedding_dim: int = 128
    context_hidden_dim: int = 256
    scorer_hidden_dim: int = 256
    candidate_side_hidden_dim: int = 64


@dataclass(frozen=True)
class NeuralOptimization:
    """Optimization schedule for the neural branch (v3 gap-recovery defaults)."""

    learning_rate: float = 5e-5
    weight_decay: float = 2e-3
    gradient_clip_norm: float = 1.0
    max_epochs: int = 30
    early_stopping_patience: int = 2
    early_stopping_min_delta: float = 1e-4
    batch_ranking_groups: int = 64
    auxiliary_bce_weight: float = 0.05
    # Extra listwise CE on the catalog-primary positive (lowest candidate_rank
    # among labeled positives) to push MRR without changing label semantics.
    primary_positive_weight: float = 0.5
    mixed_precision: bool = True
    # Linear warmup over the first ``warmup_epochs`` of optimizer steps, then
    # cosine decay to ``min_lr_ratio * learning_rate`` across ``max_epochs``.
    warmup_epochs: float = 0.5
    min_lr_ratio: float = 0.05
    # EMA of weights used for validation selection and exported checkpoint.
    ema_decay: float = 0.995


@dataclass(frozen=True)
class NeuralTrainingConfig:
    """Configuration for one neural prepare/train/score invocation."""

    features_root: Path = DEFAULT_FEATURES_ROOT
    training_root: Path = DEFAULT_TRAINING_ROOT
    graph_root: Path = DEFAULT_GRAPH_ROOT
    neural_root: Path = DEFAULT_NEURAL_ROOT
    reference_scores_path: Path = DEFAULT_REFERENCE_SCORES
    contract_lock_path: Path = DEFAULT_CONTRACT_LOCK
    gate_selection_path: Path = DEFAULT_GATE_SELECTION
    prepare_manifest_path: Path = DEFAULT_PREPARE_MANIFEST
    training_report_path: Path = DEFAULT_TRAINING_REPORT
    score_report_path: Path = DEFAULT_SCORE_REPORT
    selection_report_path: Path = DEFAULT_SELECTION_REPORT

    mode: str = "development"
    frozen_selection: bool = False
    seed: int = PRIMARY_SEED
    top_k: tuple[int, ...] = (1, 3, 5, 10)
    max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH
    shard_count: int = DEFAULT_SHARD_COUNT
    # When None, resolved from the Stage 1 gate selection's selected_experiment.
    reference_baseline_name: str | None = None

    architecture: NeuralArchitecture = field(default_factory=NeuralArchitecture)
    optimization: NeuralOptimization = field(default_factory=NeuralOptimization)

    # Setting this to False is reserved for synthetic smoke tests; production
    # runs must keep the neural-readiness gate enforced.
    require_neural_gate: bool = True
    # Optional torch device override ("cpu", "cuda", "cuda:0"). ``None`` selects
    # CUDA automatically when available and falls back to CPU otherwise.
    device: str | None = None

    duckdb_temp_directory: Path | None = DUCKDB_TEMP_DIR
    duckdb_memory_limit: str | None = DUCKDB_MEMORY_LIMIT
    duckdb_threads: int | None = DUCKDB_THREADS

    # ---- input artifact paths -------------------------------------------------
    @property
    def patient_stay_features_path(self) -> Path:
        return self.features_root / "patient_stay_features.parquet"

    @property
    def event_sequences_path(self) -> Path:
        return self.features_root / "event_sequences.parquet"

    @property
    def patient_condition_medication_path(self) -> Path:
        return self.training_root / "patient_condition_medication.parquet"

    @property
    def candidate_catalog_path(self) -> Path:
        return self.training_root / "candidate_catalog.parquet"

    # ---- cache layout ---------------------------------------------------------
    @property
    def cache_root(self) -> Path:
        return self.neural_root / "cache"

    @property
    def vocab_root(self) -> Path:
        return self.neural_root / "vocab"

    @property
    def feature_layout_path(self) -> Path:
        return self.neural_root / "feature_layout.json"

    @property
    def normalization_path(self) -> Path:
        return self.neural_root / "normalization_stats.parquet"

    @property
    def event_vocabulary_path(self) -> Path:
        return self.vocab_root / "event_vocabulary.parquet"

    @property
    def condition_vocabulary_path(self) -> Path:
        return self.vocab_root / "condition_vocabulary.parquet"

    @property
    def candidate_vocabulary_path(self) -> Path:
        return self.vocab_root / "candidate_medication_vocabulary.parquet"

    @property
    def categorical_vocabulary_path(self) -> Path:
        return self.vocab_root / "categorical_vocabulary.parquet"

    @property
    def global_candidate_prior_path(self) -> Path:
        return self.vocab_root / "global_candidate_prior.parquet"

    @property
    def condition_candidate_prior_path(self) -> Path:
        return self.vocab_root / "condition_candidate_prior.parquet"

    @property
    def graph_edges_path(self) -> Path:
        return self.graph_root / "graph_edges.parquet"

    @property
    def graph_features_path(self) -> Path:
        return self.cache_root / "graph_features.parquet"

    @property
    def checkpoints_root(self) -> Path:
        return self.neural_root / "checkpoints"

    @property
    def checkpoint_path(self) -> Path:
        return self.checkpoints_root / "transformer_recommender.pt"

    @property
    def calibration_path(self) -> Path:
        return self.checkpoints_root / "temperature_calibration.json"

    @property
    def training_state_path(self) -> Path:
        return self.checkpoints_root / "training_state.json"

    @property
    def predictions_root(self) -> Path:
        return self.neural_root / "predictions"

    @property
    def score_root(self) -> Path:
        if self.mode == "development":
            return self.predictions_root / "development"
        return self.predictions_root / "final"

    @property
    def score_output_path(self) -> Path:
        return self.score_root / "baseline_scores.parquet"

    def context_features_dir(self, split: str) -> Path:
        return self.cache_root / "context_features" / split

    def context_events_dir(self, split: str) -> Path:
        return self.cache_root / "context_events" / split

    def groups_dir(self, split: str) -> Path:
        return self.cache_root / "groups" / split

    def evaluation_splits(self) -> tuple[str, ...]:
        """Return the splits that are cached and scored for the current mode."""

        if self.mode == "development":
            return ("train", "validation")
        return ("train", "validation", "test")
