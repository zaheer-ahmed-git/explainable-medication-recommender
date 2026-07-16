"""Fit train-only preprocessing artifacts for model-ready tabular rows."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import duckdb
import joblib

from pipeline.config import (
    DUCKDB_MEMORY_LIMIT,
    DUCKDB_TEMP_DIR,
    DUCKDB_THREADS,
    FEATURE_VERSION,
    FEATURES_ROOT,
    LABEL_VERSION,
    RANDOM_SEED,
    REPORTS_ROOT,
    SPLIT_VERSION,
    TRAINING_ROOT,
)
from pipeline.extract_utils import (
    configure_duckdb_connection,
    safe_error_message,
    sql_string,
)
from pipeline.learned_baselines import (
    NEGATIVE_TO_POSITIVE_RATIO,
    LearnedFeatureSpec,
    fit_preprocessor,
    resolve_feature_spec,
    training_sample_query,
    write_training_sample,
)


SCHEMA_VERSION = "train-fitted-preprocessing-manifest-v1"
DEFAULT_MANIFEST_PATH = REPORTS_ROOT / "preprocessing_manifest.json"


@dataclass(frozen=True)
class PreprocessingBuildConfig:
    """Configuration for train-fitted preprocessing artifacts."""

    features_root: Path = FEATURES_ROOT
    training_root: Path = TRAINING_ROOT
    preprocessing_root: Path = TRAINING_ROOT / "preprocessing"
    manifest_path: Path = DEFAULT_MANIFEST_PATH
    seed: int = RANDOM_SEED
    condition_tokens: tuple[str, ...] = ()
    feature_version: str = FEATURE_VERSION
    label_version: str = LABEL_VERSION
    split_version: str = SPLIT_VERSION
    duckdb_temp_directory: Path | None = DUCKDB_TEMP_DIR
    duckdb_memory_limit: str | None = DUCKDB_MEMORY_LIMIT
    duckdb_threads: int | None = DUCKDB_THREADS

    @property
    def patient_stay_features_path(self) -> Path:
        return self.features_root / "patient_stay_features.parquet"

    @property
    def patient_condition_medication_path(self) -> Path:
        return self.training_root / "patient_condition_medication.parquet"

    @property
    def training_sample_path(self) -> Path:
        return self.preprocessing_root / "train_preprocessing_sample.parquet"

    @property
    def preprocessor_path(self) -> Path:
        return self.preprocessing_root / "train_fitted_preprocessor.joblib"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write stable JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def configure_connection(
    config: PreprocessingBuildConfig,
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Apply shared memory-safe DuckDB settings."""

    configure_duckdb_connection(
        connection,
        temp_directory=config.duckdb_temp_directory,
        memory_limit=config.duckdb_memory_limit,
        threads=config.duckdb_threads,
    )


def missing_input_tables(config: PreprocessingBuildConfig) -> list[dict[str, str]]:
    """Return missing inputs for train-fitted preprocessing."""

    required = (
        ("patient_stay_features", config.patient_stay_features_path),
        ("patient_condition_medication", config.patient_condition_medication_path),
    )
    return [
        {"table_name": table_name, "path": str(path)}
        for table_name, path in required
        if not path.exists()
    ]


def condition_filter_sql(config: PreprocessingBuildConfig) -> str:
    """Return optional condition filtering for the training sample."""

    if not config.condition_tokens:
        return "TRUE"
    tokens = ", ".join(sql_string(token) for token in config.condition_tokens)
    return f"pcm.index_condition_token IN ({tokens})"


def categorical_vocabulary_summary(
    preprocessor: Any,
    feature_spec: LearnedFeatureSpec,
) -> list[dict[str, Any]]:
    """Return aggregate category counts without category values."""

    categorical_columns = [
        *feature_spec.stay_categorical,
        *feature_spec.row_categorical,
    ]
    if not categorical_columns:
        return []

    def is_missing_statistic(value: object) -> bool:
        try:
            return bool(math.isnan(value))  # type: ignore[arg-type]
        except TypeError:
            return False

    categorical_pipeline = preprocessor.named_transformers_["categorical"]
    imputer = categorical_pipeline.named_steps["imputer"]
    encoder = categorical_pipeline.named_steps["encoder"]
    retained_columns = [
        column_name
        for column_name, statistic in zip(
            categorical_columns,
            imputer.statistics_,
            strict=True,
        )
        if not is_missing_statistic(statistic)
    ]
    return [
        {
            "column_name": column_name,
            "category_count": int(len(categories)),
        }
        for column_name, categories in zip(
            retained_columns,
            encoder.categories_,
            strict=True,
        )
    ]


def base_manifest(
    config: PreprocessingBuildConfig,
    *,
    status: str,
    generated_at: str,
) -> dict[str, Any]:
    """Return the aggregate manifest shell."""

    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "generated_at": generated_at,
        "data_safety": {
            "manifest_contains_patient_rows": False,
            "local_artifacts_contain_patient_level_rows": True,
            "artifact_storage": "ignored Dataset/processed/training/preprocessing",
            "vocabulary_values_are_local_only": True,
        },
        "fit_scope": {
            "source": "mimiciv",
            "split": "train",
            "condition_tokens": list(config.condition_tokens),
            "negative_to_positive_ratio": NEGATIVE_TO_POSITIVE_RATIO,
            "seed": config.seed,
        },
        "versions": {
            "feature_version": config.feature_version,
            "label_version": config.label_version,
            "split_version": config.split_version,
        },
        "artifacts": {},
    }


def save_preprocessor_artifact(
    config: PreprocessingBuildConfig,
    *,
    preprocessor: Any,
    feature_spec: LearnedFeatureSpec,
    category_summary: Sequence[dict[str, Any]],
    training_counts: dict[str, int],
    generated_at: str,
) -> None:
    """Persist the local train-fitted preprocessing artifact."""

    config.preprocessing_root.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "schema_version": SCHEMA_VERSION,
            "generated_at": generated_at,
            "fit_scope": {
                "source": "mimiciv",
                "split": "train",
                "condition_tokens": config.condition_tokens,
                "negative_to_positive_ratio": NEGATIVE_TO_POSITIVE_RATIO,
                "seed": config.seed,
            },
            "feature_spec": feature_spec,
            "preprocessor": preprocessor,
            "category_summary": list(category_summary),
            "training_counts": training_counts,
            "versions": {
                "feature_version": config.feature_version,
                "label_version": config.label_version,
                "split_version": config.split_version,
            },
        },
        config.preprocessor_path,
    )


def build_preprocessing_artifacts(
    config: PreprocessingBuildConfig = PreprocessingBuildConfig(),
) -> dict[str, Any]:
    """Fit imputation, scaling, encoding, and vocabularies on MIMIC train rows."""

    generated_at = datetime.now(UTC).isoformat()
    config.preprocessing_root.mkdir(parents=True, exist_ok=True)
    missing = missing_input_tables(config)
    if missing:
        manifest = base_manifest(
            config,
            status="failed_missing_inputs",
            generated_at=generated_at,
        )
        manifest["missing_inputs"] = missing
        write_json(config.manifest_path, manifest)
        return manifest

    manifest = base_manifest(config, status="completed", generated_at=generated_at)
    try:
        with duckdb.connect(database=":memory:") as connection:
            configure_connection(config, connection)
            feature_spec = resolve_feature_spec(
                connection,
                config.patient_stay_features_path,
            )
            sample_query = training_sample_query(
                patient_condition_medication_path=(
                    config.patient_condition_medication_path
                ),
                patient_stay_features_path=config.patient_stay_features_path,
                feature_spec=feature_spec,
                condition_filter_sql=condition_filter_sql(config),
                seed=config.seed,
            )
            sample_row_count = write_training_sample(
                connection,
                query=sample_query,
                output_path=config.training_sample_path,
            )
    except Exception as error:
        manifest["status"] = "failed"
        manifest["reason"] = safe_error_message(error)
        write_json(config.manifest_path, manifest)
        return manifest

    try:
        preprocessor, training_counts = fit_preprocessor(
            config.training_sample_path,
            feature_spec,
        )
    except ValueError as error:
        manifest["status"] = "failed_empty_training_sample"
        manifest["reason"] = safe_error_message(error)
        manifest["training_sample"] = {"row_count": sample_row_count}
        write_json(config.manifest_path, manifest)
        return manifest

    category_summary = categorical_vocabulary_summary(preprocessor, feature_spec)
    save_preprocessor_artifact(
        config,
        preprocessor=preprocessor,
        feature_spec=feature_spec,
        category_summary=category_summary,
        training_counts=training_counts,
        generated_at=generated_at,
    )

    manifest["artifacts"] = {
        "training_sample": str(config.training_sample_path),
        "preprocessor": str(config.preprocessor_path),
    }
    manifest["training_sample"] = {
        **training_counts,
        "materialized_row_count": sample_row_count,
    }
    manifest["feature_columns"] = list(feature_spec.model_columns)
    manifest["stay_numeric_columns"] = list(feature_spec.stay_numeric)
    manifest["stay_categorical_columns"] = list(feature_spec.stay_categorical)
    manifest["row_numeric_columns"] = list(feature_spec.row_numeric)
    manifest["row_categorical_columns"] = list(feature_spec.row_categorical)
    manifest["categorical_vocabulary_summary"] = category_summary
    write_json(config.manifest_path, manifest)
    return manifest


def parse_repeated_csv(values: Sequence[str] | None) -> tuple[str, ...]:
    """Parse repeated CLI values that may contain comma-separated tokens."""

    if not values:
        return ()
    parsed: list[str] = []
    for value in values:
        parsed.extend(token.strip() for token in value.split(",") if token.strip())
    return tuple(dict.fromkeys(parsed))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit train-only imputation, scaling, encoding, and categorical "
            "vocabulary artifacts for model-ready Milestone 6 rows."
        ),
    )
    parser.add_argument(
        "--features-root",
        type=Path,
        default=FEATURES_ROOT,
        help="Root directory containing Milestone 6 feature artifacts.",
    )
    parser.add_argument(
        "--training-root",
        type=Path,
        default=TRAINING_ROOT,
        help="Root directory containing Milestone 6 training artifacts.",
    )
    parser.add_argument(
        "--preprocessing-root",
        type=Path,
        default=TRAINING_ROOT / "preprocessing",
        help="Output directory for local ignored preprocessing artifacts.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Output path for the aggregate preprocessing manifest.",
    )
    parser.add_argument(
        "--condition-token",
        action="append",
        default=[],
        help="Optional condition token filter. Repeat or pass comma-separated values.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Seed used for deterministic weak-negative sampling.",
    )
    parser.add_argument(
        "--duckdb-temp-dir",
        type=Path,
        default=DUCKDB_TEMP_DIR,
        help="Directory DuckDB may use to spill larger-than-memory operators.",
    )
    parser.add_argument(
        "--duckdb-memory-limit",
        default=DUCKDB_MEMORY_LIMIT,
        help="Optional DuckDB memory ceiling, e.g. '24GB'.",
    )
    parser.add_argument(
        "--duckdb-threads",
        type=int,
        default=DUCKDB_THREADS,
        help="Optional DuckDB thread cap.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_preprocessing_artifacts(
        PreprocessingBuildConfig(
            features_root=args.features_root,
            training_root=args.training_root,
            preprocessing_root=args.preprocessing_root,
            manifest_path=args.manifest,
            seed=args.seed,
            condition_tokens=parse_repeated_csv(args.condition_token),
            duckdb_temp_directory=args.duckdb_temp_dir,
            duckdb_memory_limit=args.duckdb_memory_limit,
            duckdb_threads=args.duckdb_threads,
        )
    )
    print(
        "Wrote preprocessing manifest: "
        f"status={manifest['status']}, "
        f"training_rows={manifest.get('training_sample', {}).get('row_count', 0)}"
    )
    return 0 if manifest["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
