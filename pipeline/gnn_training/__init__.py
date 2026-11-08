"""Gate-first native-PyTorch GNN and frozen-Transformer fusion pipeline.

The configuration and contract surfaces are importable without PyTorch.  Model,
dataset, training, and scoring modules load the optional ``neural`` dependency
only when their corresponding stage runs.
"""

from __future__ import annotations

from pipeline.gnn_training.config import (
    FORWARD_RELATION_TYPES,
    NODE_ROLES,
    NODE_ROLE_TO_INDEX,
    NODE_TYPES,
    NODE_TYPE_TO_INDEX,
    RELATION_TO_INDEX,
    RELATION_TYPES,
    REVERSE_RELATION_TYPES,
    SELF_LOOP_RELATION,
    GNNArchitecture,
    GNNOptimization,
    GNNTrainingConfig,
)

__all__ = [
    "FORWARD_RELATION_TYPES",
    "GNNArchitecture",
    "GNNOptimization",
    "GNNTrainingConfig",
    "NODE_ROLES",
    "NODE_ROLE_TO_INDEX",
    "NODE_TYPES",
    "NODE_TYPE_TO_INDEX",
    "RELATION_TO_INDEX",
    "RELATION_TYPES",
    "REVERSE_RELATION_TYPES",
    "SELF_LOOP_RELATION",
]
