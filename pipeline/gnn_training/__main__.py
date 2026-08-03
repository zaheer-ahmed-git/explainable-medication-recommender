"""Lazy command-line entry point for GNN and frozen-Transformer fusion.

``prepare`` remains PyTorch-free.  Each training/scoring implementation is
imported only after its subcommand is selected, so package import and
``--help`` do not require the optional ``neural`` dependency group.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from pipeline.config import DUCKDB_MEMORY_LIMIT, DUCKDB_TEMP_DIR, DUCKDB_THREADS
from pipeline.gnn_training.config import (
    DEFAULT_FOLD_COUNT,
    DEFAULT_SHARD_COUNT,
    PRIMARY_SEED,
    GNNTrainingConfig,
)


def _parse_top_k(raw: str) -> tuple[int, ...]:
    try:
        values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError("top-k values must be positive integers") from error
    if not values or any(value <= 0 for value in values):
        raise ValueError("top-k values must be positive integers")
    return tuple(sorted(set(values)))


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--gnn-root", type=Path, default=None)
    parser.add_argument("--neural-root", type=Path, default=None)
    parser.add_argument("--graph-root", type=Path, default=None)
    parser.add_argument("--subgraphs-root", type=Path, default=None)
    parser.add_argument("--features-root", type=Path, default=None)
    parser.add_argument("--training-root", type=Path, default=None)
    parser.add_argument("--graph-reference-scores", type=Path, default=None)
    parser.add_argument("--graph-reference-report", type=Path, default=None)
    parser.add_argument("--contract-lock", type=Path, default=None)
    parser.add_argument("--subgraphs-manifest", type=Path, default=None)
    parser.add_argument("--neural-selection", type=Path, default=None)
    parser.add_argument("--crossfit-graph-manifest", type=Path, default=None)
    parser.add_argument("--gnn-selection", type=Path, default=None)
    parser.add_argument("--fusion-selection", type=Path, default=None)
    parser.add_argument(
        "--mode",
        choices=("development", "final"),
        default="development",
    )
    parser.add_argument("--seed", type=int, default=PRIMARY_SEED)
    parser.add_argument("--fold-count", type=int, default=DEFAULT_FOLD_COUNT)
    parser.add_argument("--shard-count", type=int, default=DEFAULT_SHARD_COUNT)
    parser.add_argument("--top-k", default="1,3,5,10")
    parser.add_argument("--frozen-selection", action="store_true")
    parser.add_argument(
        "--allow-ungated",
        action="store_true",
        help=(
            "Bypass upstream artifact gates only for synthetic unit tests. "
            "Protected production paths remain blocked."
        ),
    )
    parser.add_argument("--device", default=None, help="Torch device override.")

    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--relation-layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--batch-ranking-groups", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--gradient-clip-norm", type=float, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--auxiliary-bce-weight", type=float, default=None)
    parser.add_argument("--primary-positive-weight", type=float, default=None)
    parser.add_argument(
        "--no-mixed-precision",
        action="store_true",
        help=(
            "Disable CUDA mixed precision as a reviewed numerical-debugging "
            "fallback; the pre-registered default remains enabled."
        ),
    )

    parser.add_argument("--duckdb-temp-dir", type=Path, default=DUCKDB_TEMP_DIR)
    parser.add_argument("--duckdb-memory-limit", default=DUCKDB_MEMORY_LIMIT)
    parser.add_argument(
        "--duckdb-max-temp-directory-size",
        default=None,
        help=(
            "DuckDB spill ceiling on --duckdb-temp-dir. Set explicitly when "
            "spilling to WORK_SCRATCH so a small /tmp quota cannot block joins."
        ),
    )
    parser.add_argument("--duckdb-threads", type=int, default=DUCKDB_THREADS)


def _validate_positive(name: str, value: int | float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def build_config(args: argparse.Namespace) -> GNNTrainingConfig:
    """Construct a :class:`GNNTrainingConfig` from CLI arguments."""

    _validate_positive("fold-count", args.fold_count)
    if args.fold_count < 2:
        raise ValueError("fold-count must be at least 2")
    _validate_positive("shard-count", args.shard_count)

    defaults = GNNTrainingConfig()
    architecture_overrides: dict[str, Any] = {}
    if args.hidden_dim is not None:
        _validate_positive("hidden-dim", args.hidden_dim)
        architecture_overrides["hidden_dim"] = args.hidden_dim
    if args.relation_layers is not None:
        _validate_positive("relation-layers", args.relation_layers)
        architecture_overrides["relation_layers"] = args.relation_layers
    if args.dropout is not None:
        if not 0.0 <= args.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        architecture_overrides["dropout"] = args.dropout
    architecture = (
        replace(defaults.architecture, **architecture_overrides)
        if architecture_overrides
        else defaults.architecture
    )

    optimization_overrides: dict[str, Any] = {}
    if args.no_mixed_precision:
        optimization_overrides["mixed_precision"] = False
    for argument_name, field_name in (
        ("batch_ranking_groups", "batch_ranking_groups"),
        ("max_epochs", "max_epochs"),
        ("learning_rate", "learning_rate"),
        ("weight_decay", "weight_decay"),
        ("gradient_clip_norm", "gradient_clip_norm"),
        ("early_stopping_patience", "early_stopping_patience"),
        ("auxiliary_bce_weight", "auxiliary_bce_weight"),
        ("primary_positive_weight", "primary_positive_weight"),
    ):
        value = getattr(args, argument_name)
        if value is not None:
            if field_name in {
                "batch_ranking_groups",
                "max_epochs",
                "learning_rate",
                "gradient_clip_norm",
                "early_stopping_patience",
            }:
                _validate_positive(argument_name.replace("_", "-"), value)
            elif value < 0:
                raise ValueError(
                    f"{argument_name.replace('_', '-')} must be non-negative"
                )
            optimization_overrides[field_name] = value
    optimization = (
        replace(defaults.optimization, **optimization_overrides)
        if optimization_overrides
        else defaults.optimization
    )

    kwargs: dict[str, Any] = {
        "mode": args.mode,
        "frozen_selection": args.frozen_selection,
        "allow_ungated": args.allow_ungated,
        "seed": args.seed,
        "fold_count": args.fold_count,
        "shard_count": args.shard_count,
        "top_k": _parse_top_k(args.top_k),
        "device": args.device,
        "architecture": architecture,
        "optimization": optimization,
        "duckdb_temp_directory": args.duckdb_temp_dir,
        "duckdb_memory_limit": args.duckdb_memory_limit,
        "duckdb_max_temp_directory_size": args.duckdb_max_temp_directory_size,
        "duckdb_threads": args.duckdb_threads,
    }
    for argument_name, field_name in (
        ("gnn_root", "gnn_root"),
        ("neural_root", "neural_root"),
        ("graph_root", "graph_root"),
        ("subgraphs_root", "subgraphs_root"),
        ("features_root", "features_root"),
        ("training_root", "training_root"),
        ("graph_reference_scores", "graph_reference_scores_path"),
        ("graph_reference_report", "graph_reference_report_path"),
        ("contract_lock", "contract_lock_path"),
        ("subgraphs_manifest", "subgraphs_manifest_path"),
        ("neural_selection", "neural_selection_path"),
        ("crossfit_graph_manifest", "crossfit_graph_manifest_path"),
        ("gnn_selection", "gnn_selection_report_path"),
        ("fusion_selection", "fusion_selection_report_path"),
    ):
        value = getattr(args, argument_name)
        if value is not None:
            kwargs[field_name] = value
    return replace(defaults, **kwargs)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m pipeline.gnn_training",
        description=(
            "Phase 8 P0 relation-aware GNN and frozen-Transformer fusion pipeline."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        (
            "prepare",
            "Audit locks and build bounded graph/Transformer caches.",
        ),
        (
            "train-gnn",
            "Train the relation branch with patient-grouped cross-fitting.",
        ),
        (
            "score-gnn",
            "Score and qualify the standalone relation branch.",
        ),
        (
            "train-fusion",
            "Train late and residual frozen-Transformer fusion candidates.",
        ),
        (
            "score-fusion",
            "Score and qualify the selected hybrid candidate.",
        ),
    ):
        subparser = subparsers.add_parser(name, help=help_text)
        _add_common_arguments(subparser)
    return parser.parse_args(argv)


def _run(command: str, config: GNNTrainingConfig) -> dict[str, Any]:
    if command == "prepare":
        from pipeline.gnn_training.data import prepare_gnn_caches

        return prepare_gnn_caches(config)
    if command == "train-gnn":
        from pipeline.gnn_training.train_gnn import train_gnn

        return train_gnn(config)
    if command == "score-gnn":
        from pipeline.gnn_training.score_gnn import score_gnn

        return score_gnn(config)
    if command == "train-fusion":
        from pipeline.gnn_training.train_fusion import train_fusion

        return train_fusion(config)
    if command == "score-fusion":
        from pipeline.gnn_training.score_fusion import score_fusion

        return score_fusion(config)
    raise ValueError(f"unknown command: {command!r}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = build_config(args)
    except ValueError as error:
        print(f"Invalid GNN training arguments: {error}")
        return 2
    report = _run(args.command, config)
    status = report.get("status", "unknown")
    print(f"GNN {args.command} finished: status={status}, mode={config.mode}")
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
