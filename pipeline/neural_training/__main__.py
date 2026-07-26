"""Command-line entry point for the Stage 2 neural Transformer pipeline.

Three subcommands map to the pipeline stages:

* ``prepare`` - build vocabularies, normalization, layout, and sharded caches
  (DuckDB only; no PyTorch required);
* ``train`` - fit the Transformer branch with early stopping and calibration;
* ``score`` - score the evaluation split, compute metrics, and record the gate.

``train`` and ``score`` import PyTorch lazily so ``prepare`` and ``--help`` work
without the optional ``neural`` dependency group installed. Real runs stay
fail-closed behind the structured recovery gate unless ``--allow-ungated`` is
passed (reserved for synthetic smoke tests).
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from pipeline.config import DUCKDB_MEMORY_LIMIT, DUCKDB_TEMP_DIR, DUCKDB_THREADS
from pipeline.neural_training.config import (
    DEFAULT_MAX_SEQUENCE_LENGTH,
    DEFAULT_SHARD_COUNT,
    PRIMARY_SEED,
    NeuralTrainingConfig,
)


def _parse_top_k(raw: str) -> tuple[int, ...]:
    values = [int(item) for item in raw.split(",") if item.strip()]
    if not values or any(value <= 0 for value in values):
        raise ValueError("top-k values must be positive integers")
    return tuple(sorted(set(values)))


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--neural-root", type=Path, default=None)
    parser.add_argument("--features-root", type=Path, default=None)
    parser.add_argument("--training-root", type=Path, default=None)
    parser.add_argument("--reference-scores", type=Path, default=None)
    parser.add_argument("--contract-lock", type=Path, default=None)
    parser.add_argument("--gate-selection", type=Path, default=None)
    parser.add_argument(
        "--mode", choices=("development", "final"), default="development"
    )
    parser.add_argument("--seed", type=int, default=PRIMARY_SEED)
    parser.add_argument(
        "--max-sequence-length", type=int, default=DEFAULT_MAX_SEQUENCE_LENGTH
    )
    parser.add_argument("--shard-count", type=int, default=DEFAULT_SHARD_COUNT)
    parser.add_argument("--top-k", default="1,3,5,10")
    parser.add_argument("--frozen-selection", action="store_true")
    parser.add_argument(
        "--allow-ungated",
        action="store_true",
        help="Disable the neural-readiness gate (synthetic smoke tests only).",
    )
    parser.add_argument("--device", default=None, help="Torch device override.")
    parser.add_argument("--batch-ranking-groups", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--auxiliary-bce-weight", type=float, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--warmup-epochs", type=float, default=None)
    parser.add_argument("--duckdb-temp-dir", type=Path, default=DUCKDB_TEMP_DIR)
    parser.add_argument("--duckdb-memory-limit", default=DUCKDB_MEMORY_LIMIT)
    parser.add_argument("--duckdb-threads", type=int, default=DUCKDB_THREADS)


def build_config(args: argparse.Namespace) -> NeuralTrainingConfig:
    """Construct a :class:`NeuralTrainingConfig` from parsed CLI arguments."""

    defaults = NeuralTrainingConfig()
    optimization = defaults.optimization
    overrides: dict[str, Any] = {}
    if args.batch_ranking_groups is not None:
        overrides["batch_ranking_groups"] = args.batch_ranking_groups
    if args.max_epochs is not None:
        overrides["max_epochs"] = args.max_epochs
    if args.learning_rate is not None:
        overrides["learning_rate"] = args.learning_rate
    if args.weight_decay is not None:
        overrides["weight_decay"] = args.weight_decay
    if args.auxiliary_bce_weight is not None:
        overrides["auxiliary_bce_weight"] = args.auxiliary_bce_weight
    if args.early_stopping_patience is not None:
        overrides["early_stopping_patience"] = args.early_stopping_patience
    if args.warmup_epochs is not None:
        overrides["warmup_epochs"] = args.warmup_epochs
    if overrides:
        optimization = replace(optimization, **overrides)

    kwargs: dict[str, Any] = {
        "mode": args.mode,
        "seed": args.seed,
        "max_sequence_length": args.max_sequence_length,
        "shard_count": args.shard_count,
        "top_k": _parse_top_k(args.top_k),
        "frozen_selection": args.frozen_selection,
        "require_neural_gate": not args.allow_ungated,
        "device": args.device,
        "optimization": optimization,
        "duckdb_temp_directory": args.duckdb_temp_dir,
        "duckdb_memory_limit": args.duckdb_memory_limit,
        "duckdb_threads": args.duckdb_threads,
    }
    if args.neural_root is not None:
        kwargs["neural_root"] = args.neural_root
    if args.features_root is not None:
        kwargs["features_root"] = args.features_root
    if args.training_root is not None:
        kwargs["training_root"] = args.training_root
    if args.reference_scores is not None:
        kwargs["reference_scores_path"] = args.reference_scores
    if args.contract_lock is not None:
        kwargs["contract_lock_path"] = args.contract_lock
    if args.gate_selection is not None:
        kwargs["gate_selection_path"] = args.gate_selection
    return replace(defaults, **kwargs)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m pipeline.neural_training",
        description="Stage 2 conditional neural Transformer training pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("prepare", "Build vocabularies, normalization, and sharded caches."),
        ("train", "Train the Transformer branch with early stopping."),
        ("score", "Score the evaluation split and record the neural gate."),
    ):
        subparser = subparsers.add_parser(name, help=help_text)
        _add_common_arguments(subparser)
    return parser.parse_args(argv)


def _run(command: str, config: NeuralTrainingConfig) -> dict[str, Any]:
    if command == "prepare":
        from pipeline.neural_training.data import prepare_neural_caches

        return prepare_neural_caches(config)
    if command == "train":
        from pipeline.neural_training.train import train_transformer

        return train_transformer(config)
    if command == "score":
        from pipeline.neural_training.score import score_transformer

        return score_transformer(config)
    raise ValueError(f"unknown command: {command!r}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = build_config(args)
    except ValueError as error:
        print(f"Invalid neural training arguments: {error}")
        return 2
    report = _run(args.command, config)
    status = report.get("status", "unknown")
    print(f"Neural {args.command} finished: status={status}, mode={config.mode}")
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
