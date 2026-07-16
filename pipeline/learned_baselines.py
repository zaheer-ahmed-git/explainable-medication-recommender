"""Train and score Milestone 7 learned medication-ranking baselines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import duckdb
import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse as sp
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pipeline.extract_utils import parquet_scan, sql_string


LEARNED_BASELINES = ("linear", "xgboost")
NEGATIVE_TO_POSITIVE_RATIO = 5
TRAINING_FETCH_BATCH_SIZE = 100_000
SCORING_BATCH_COUNT = 32

LINEAR_V1_HYPERPARAMETERS: dict[str, Any] = {
    "loss": "log_loss",
    "max_iter": 1000,
    "tol": 1e-3,
}
XGBOOST_V1_HYPERPARAMETERS: dict[str, Any] = {
    "objective": "binary:logistic",
    "max_depth": 6,
    "n_estimators": 200,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
}

ROW_LEVEL_FEATURE_COLUMNS = (
    "index_condition_token",
    "candidate_medication_token",
    "candidate_rank",
)
STAY_FEATURE_DENYLIST = frozenset(
    {
        "source_version",
        "patient_uid",
        "encounter_uid",
        "stay_uid",
        "split",
        "eligibility_status",
        "primary_training_eligible",
        "t0_hours_from_admit",
        "prediction_time_hours_from_admit",
        "label_window_end_hours_from_admit",
        "stay_end_hours_from_admit",
        "cohort_version",
        "harmonization_version",
        "feature_version",
        "split_version",
        "generated_at",
    }
)
STAY_CATEGORICAL_COLUMNS = frozenset(
    {
        "source",
        "sex",
        "race_or_ethnicity",
        "admission_type",
        "admission_source",
        "unit_type",
        "last_unit_type",
        "stay_type",
        "hospital_id",
        "ward_id",
    }
)
SCORE_OUTPUT_COLUMNS = (
    "source",
    "split",
    "ranking_group_id",
    "index_condition_token",
    "candidate_medication_token",
    "candidate_rank",
    "label_prescribed",
    "baseline_name",
    "score",
    "seed",
    "baseline_version",
    "evaluation_version",
    "generated_at",
)


@dataclass(frozen=True)
class LearnedFeatureSpec:
    """Resolved model feature columns."""

    stay_numeric: tuple[str, ...]
    stay_categorical: tuple[str, ...]
    row_numeric: tuple[str, ...]
    row_categorical: tuple[str, ...]

    @property
    def model_columns(self) -> tuple[str, ...]:
        return (
            *self.stay_numeric,
            *self.stay_categorical,
            *self.row_numeric,
            *self.row_categorical,
        )


@dataclass
class LearnedBaselineArtifacts:
    """Local ignored artifacts for learned baselines."""

    models_root: Path
    training_sample_path: Path
    preprocessor_path: Path
    linear_model_path: Path
    xgboost_model_path: Path


def models_root_for(evaluation_root: Path) -> Path:
    return evaluation_root / "models"


def artifact_paths(evaluation_root: Path) -> LearnedBaselineArtifacts:
    models_root = models_root_for(evaluation_root)
    return LearnedBaselineArtifacts(
        models_root=models_root,
        training_sample_path=evaluation_root / "learned_training_sample.parquet",
        preprocessor_path=models_root / "learned_preprocessor.joblib",
        linear_model_path=models_root / "linear_sgd_model.joblib",
        xgboost_model_path=models_root / "xgboost_model.json",
    )


def resolve_feature_spec(
    connection: duckdb.DuckDBPyConnection,
    patient_stay_features_path: Path,
) -> LearnedFeatureSpec:
    """Resolve stay-level model columns from the feature parquet schema."""

    describe = connection.execute(
        f"DESCRIBE SELECT * FROM {parquet_scan(patient_stay_features_path)}"
    ).fetchall()
    stay_numeric: list[str] = []
    stay_categorical: list[str] = []
    for column_name, column_type, *_rest in describe:
        name = str(column_name)
        if name in STAY_FEATURE_DENYLIST:
            continue
        dtype = str(column_type).upper()
        if name in STAY_CATEGORICAL_COLUMNS:
            stay_categorical.append(name)
        elif "BOOL" in dtype:
            stay_numeric.append(name)
        elif any(
            token in dtype for token in ("INT", "DOUBLE", "FLOAT", "DECIMAL", "HUGEINT")
        ):
            stay_numeric.append(name)
        else:
            stay_categorical.append(name)
    return LearnedFeatureSpec(
        stay_numeric=tuple(stay_numeric),
        stay_categorical=tuple(stay_categorical),
        row_numeric=("candidate_rank",),
        row_categorical=ROW_LEVEL_FEATURE_COLUMNS[:2],
    )


def _feature_select_sql(feature_spec: LearnedFeatureSpec, *, stay_alias: str) -> str:
    columns: list[str] = []
    for column in feature_spec.stay_numeric:
        columns.append(f"{stay_alias}.{column}")
    for column in feature_spec.stay_categorical:
        columns.append(f"{stay_alias}.{column}")
    for column in feature_spec.row_numeric:
        columns.append(f"pcm.{column}")
    for column in feature_spec.row_categorical:
        columns.append(f"pcm.{column}")
    return ",\n        ".join(columns)


def training_sample_query(
    *,
    patient_condition_medication_path: Path,
    patient_stay_features_path: Path,
    feature_spec: LearnedFeatureSpec,
    condition_filter_sql: str,
    seed: int,
) -> str:
    """Build a deterministic train-only positive + 5:1 negative sample."""

    feature_sql = _feature_select_sql(feature_spec, stay_alias="psf")
    hash_uniform = (
        "CAST("
        "HASH("
        f"{sql_string(str(seed))} || '|' || train_rows.ranking_group_id "
        "|| '|' || train_rows.candidate_medication_token"
        ") AS DOUBLE"
        ") / 18446744073709551615.0"
    )
    return f"""
WITH train_rows AS (
    SELECT
        pcm.source,
        pcm.split,
        pcm.stay_uid,
        pcm.ranking_group_id,
        pcm.index_condition_token,
        pcm.candidate_medication_token,
        pcm.candidate_rank,
        pcm.label_prescribed
    FROM {parquet_scan(patient_condition_medication_path)} AS pcm
    WHERE pcm.source = 'mimiciv'
        AND pcm.split = 'train'
        AND {condition_filter_sql}
),
condition_counts AS (
    SELECT
        index_condition_token,
        SUM(CASE WHEN label_prescribed THEN 1 ELSE 0 END)
            AS positive_row_count,
        SUM(CASE WHEN NOT label_prescribed THEN 1 ELSE 0 END)
            AS negative_row_count
    FROM train_rows
    GROUP BY index_condition_token
),
selected_rows AS (
    SELECT train_rows.*
    FROM train_rows
    INNER JOIN condition_counts
        ON train_rows.index_condition_token = condition_counts.index_condition_token
    WHERE train_rows.label_prescribed
        OR (
            NOT train_rows.label_prescribed
            AND condition_counts.positive_row_count > 0
            AND {hash_uniform} < LEAST(
                1.0,
                (
                    condition_counts.positive_row_count
                    * {NEGATIVE_TO_POSITIVE_RATIO}
                )::DOUBLE
                / NULLIF(condition_counts.negative_row_count, 0)
            )
        )
)
SELECT
    pcm.source,
    pcm.split,
    pcm.ranking_group_id,
    pcm.index_condition_token,
    pcm.candidate_medication_token,
    pcm.candidate_rank,
    pcm.label_prescribed,
    {feature_sql}
FROM selected_rows AS pcm
INNER JOIN {parquet_scan(patient_stay_features_path)} AS psf
    ON pcm.stay_uid = psf.stay_uid
    AND pcm.source = psf.source
"""


def scoring_rows_query(
    *,
    patient_condition_medication_path: Path,
    patient_stay_features_path: Path,
    feature_spec: LearnedFeatureSpec,
    condition_filter_sql: str,
    scoring_scope_sql: str,
    batch_id: int,
    batch_count: int,
) -> str:
    """Return one scoring batch with model features and output metadata."""

    feature_sql = _feature_select_sql(feature_spec, stay_alias="psf")
    return f"""
SELECT
    pcm.source,
    pcm.split,
    pcm.ranking_group_id,
    pcm.index_condition_token,
    pcm.candidate_medication_token,
    pcm.candidate_rank,
    pcm.label_prescribed,
    {feature_sql}
FROM {parquet_scan(patient_condition_medication_path)} AS pcm
INNER JOIN {parquet_scan(patient_stay_features_path)} AS psf
    ON pcm.stay_uid = psf.stay_uid
    AND pcm.source = psf.source
WHERE {condition_filter_sql}
    AND {scoring_scope_sql}
    AND ABS(HASH(pcm.ranking_group_id)) % {batch_count} = {batch_id}
"""


def build_preprocessor(feature_spec: LearnedFeatureSpec) -> ColumnTransformer:
    """Build a sparse-friendly preprocessing pipeline."""

    numeric_columns = [*feature_spec.stay_numeric, *feature_spec.row_numeric]
    categorical_columns = [
        *feature_spec.stay_categorical,
        *feature_spec.row_categorical,
    ]
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(strategy="constant", fill_value="missing"),
                        ),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=True,
                            ),
                        ),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )


def _prepare_matrix(
    frame: pd.DataFrame,
    feature_spec: LearnedFeatureSpec,
) -> pd.DataFrame:
    return frame.loc[:, list(feature_spec.model_columns)]


def _training_counts(frame: pd.DataFrame) -> dict[str, int]:
    labels = frame["label_prescribed"].astype(bool)
    return {
        "row_count": int(len(frame)),
        "positive_row_count": int(labels.sum()),
        "negative_row_count": int((~labels).sum()),
        "index_condition_count": int(frame["index_condition_token"].nunique()),
    }


def iter_training_batches(
    training_sample_path: Path,
    *,
    batch_size: int = TRAINING_FETCH_BATCH_SIZE,
) -> Iterator[pd.DataFrame]:
    """Iterate over a materialized training sample without loading it all."""

    parquet_file = pq.ParquetFile(training_sample_path)
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        yield batch.to_pandas()


def fit_preprocessor(
    training_sample_path: Path,
    feature_spec: LearnedFeatureSpec,
) -> tuple[ColumnTransformer, dict[str, int]]:
    """Fit preprocessing on the bounded training sample."""

    frame = pd.read_parquet(training_sample_path)
    if frame.empty:
        raise ValueError("learned baseline training sample is empty")
    preprocessor = build_preprocessor(feature_spec)
    preprocessor.fit(_prepare_matrix(frame, feature_spec))
    return preprocessor, _training_counts(frame)


def fit_linear_model(
    training_sample_path: Path,
    feature_spec: LearnedFeatureSpec,
    preprocessor: ColumnTransformer,
    seed: int,
) -> SGDClassifier:
    """Fit a linear baseline with chunked partial updates."""

    model = SGDClassifier(random_state=seed, **LINEAR_V1_HYPERPARAMETERS)
    fitted = False
    for frame in iter_training_batches(training_sample_path):
        features = preprocessor.transform(_prepare_matrix(frame, feature_spec))
        labels = frame["label_prescribed"].astype(int).to_numpy()
        if not fitted:
            model.partial_fit(features, labels, classes=np.array([0, 1], dtype=int))
            fitted = True
        else:
            model.partial_fit(features, labels)
    if not fitted:
        raise ValueError("linear baseline training sample did not contain any rows")
    return model


def fit_xgboost_model(
    training_sample_path: Path,
    feature_spec: LearnedFeatureSpec,
    preprocessor: ColumnTransformer,
    seed: int,
) -> xgb.Booster:
    """Fit an XGBoost baseline on the bounded training sample."""

    feature_batches: list[sp.spmatrix] = []
    label_frames: list[np.ndarray] = []
    for frame in iter_training_batches(training_sample_path):
        transformed = preprocessor.transform(_prepare_matrix(frame, feature_spec))
        if sp.issparse(transformed):
            feature_batches.append(transformed.tocsr())
        else:
            feature_batches.append(sp.csr_matrix(transformed))
        label_frames.append(frame["label_prescribed"].astype(int).to_numpy())
    if not feature_batches:
        raise ValueError("learned baseline training sample is empty")
    features = sp.vstack(feature_batches, format="csr")
    labels = np.concatenate(label_frames)
    dmatrix = xgb.DMatrix(features, label=labels)
    params = {
        **XGBOOST_V1_HYPERPARAMETERS,
        "seed": seed,
    }
    num_boost_round = int(params.pop("n_estimators"))
    return xgb.train(params, dmatrix, num_boost_round=num_boost_round)


def predict_scores(
    model_name: str,
    *,
    model: SGDClassifier | xgb.Booster,
    preprocessor: ColumnTransformer,
    frame: pd.DataFrame,
    feature_spec: LearnedFeatureSpec,
) -> np.ndarray:
    """Return positive-class scores for one batch."""

    features = preprocessor.transform(_prepare_matrix(frame, feature_spec))
    if model_name == "linear":
        assert isinstance(model, SGDClassifier)
        return model.predict_proba(features)[:, 1]
    assert isinstance(model, xgb.Booster)
    return model.predict(xgb.DMatrix(features))


def _score_batch_frame(
    frame: pd.DataFrame,
    *,
    baseline_name: str,
    model: SGDClassifier | xgb.Booster,
    preprocessor: ColumnTransformer,
    feature_spec: LearnedFeatureSpec,
    seed: int,
    baseline_version: str,
    evaluation_version: str,
    generated_at: str,
) -> pd.DataFrame:
    scores = predict_scores(
        baseline_name,
        model=model,
        preprocessor=preprocessor,
        frame=frame,
        feature_spec=feature_spec,
    )
    return pd.DataFrame(
        {
            "source": frame["source"],
            "split": frame["split"],
            "ranking_group_id": frame["ranking_group_id"],
            "index_condition_token": frame["index_condition_token"],
            "candidate_medication_token": frame["candidate_medication_token"],
            "candidate_rank": frame["candidate_rank"],
            "label_prescribed": frame["label_prescribed"],
            "baseline_name": baseline_name,
            "score": scores,
            "seed": seed,
            "baseline_version": baseline_version,
            "evaluation_version": evaluation_version,
            "generated_at": generated_at,
        }
    )


def materialize_learned_scores(
    connection: duckdb.DuckDBPyConnection,
    *,
    output_path: Path,
    patient_condition_medication_path: Path,
    patient_stay_features_path: Path,
    feature_spec: LearnedFeatureSpec,
    condition_filter_sql: str,
    scoring_scope_sql: str,
    baselines: Sequence[str],
    preprocessor: ColumnTransformer,
    models: dict[str, SGDClassifier | xgb.Booster],
    seed: int,
    baseline_version: str,
    evaluation_version: str,
    generated_at: str,
    batch_count: int = SCORING_BATCH_COUNT,
) -> int:
    """Score learned baselines in hash batches and write one parquet file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    writer: pq.ParquetWriter | None = None
    row_count = 0
    try:
        for batch_id in range(batch_count):
            batch_query = scoring_rows_query(
                patient_condition_medication_path=patient_condition_medication_path,
                patient_stay_features_path=patient_stay_features_path,
                feature_spec=feature_spec,
                condition_filter_sql=condition_filter_sql,
                scoring_scope_sql=scoring_scope_sql,
                batch_id=batch_id,
                batch_count=batch_count,
            )
            frame = connection.execute(batch_query).fetchdf()
            if frame.empty:
                continue
            batch_frames: list[pd.DataFrame] = []
            for baseline_name in baselines:
                scored = _score_batch_frame(
                    frame,
                    baseline_name=baseline_name,
                    model=models[baseline_name],
                    preprocessor=preprocessor,
                    feature_spec=feature_spec,
                    seed=seed,
                    baseline_version=baseline_version,
                    evaluation_version=evaluation_version,
                    generated_at=generated_at,
                )
                batch_frames.append(scored)
            batch_output = pd.concat(batch_frames, ignore_index=True)
            table = pa.Table.from_pandas(batch_output, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)
            writer.write_table(table)
            row_count += len(batch_output)
    finally:
        if writer is not None:
            writer.close()
    return row_count


def save_learned_artifacts(
    artifacts: LearnedBaselineArtifacts,
    *,
    preprocessor: ColumnTransformer,
    models: dict[str, SGDClassifier | xgb.Booster],
    feature_spec: LearnedFeatureSpec,
) -> None:
    artifacts.models_root.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "feature_spec": feature_spec,
            "preprocessor": preprocessor,
        },
        artifacts.preprocessor_path,
    )
    if "linear" in models:
        joblib.dump(models["linear"], artifacts.linear_model_path)
    if "xgboost" in models:
        booster = models["xgboost"]
        assert isinstance(booster, xgb.Booster)
        booster.save_model(artifacts.xgboost_model_path)


def learned_manifest_section(
    *,
    status: str,
    feature_spec: LearnedFeatureSpec,
    training_counts: dict[str, int],
    artifacts: LearnedBaselineArtifacts,
    baselines: Sequence[str],
    seed: int,
    reason: str | None = None,
) -> dict[str, Any]:
    """Build aggregate learned-baseline metadata for the evaluation report."""

    payload: dict[str, Any] = {
        "status": status,
        "negative_to_positive_ratio": NEGATIVE_TO_POSITIVE_RATIO,
        "training_sample": training_counts,
        "feature_columns": list(feature_spec.model_columns),
        "stay_numeric_columns": list(feature_spec.stay_numeric),
        "stay_categorical_columns": list(feature_spec.stay_categorical),
        "row_level_columns": list(ROW_LEVEL_FEATURE_COLUMNS),
        "hyperparameters": {
            "linear": {
                **LINEAR_V1_HYPERPARAMETERS,
                "random_state": seed,
            },
            "xgboost": {
                **XGBOOST_V1_HYPERPARAMETERS,
                "seed": seed,
            },
        },
        "artifacts": {
            "training_sample": str(artifacts.training_sample_path),
            "preprocessor": str(artifacts.preprocessor_path),
            "linear_model": str(artifacts.linear_model_path),
            "xgboost_model": str(artifacts.xgboost_model_path),
        },
        "baselines": list(baselines),
    }
    if reason is not None:
        payload["reason"] = reason
    return payload


def write_training_sample(
    connection: duckdb.DuckDBPyConnection,
    *,
    query: str,
    output_path: Path,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    connection.execute(f"COPY ({query}) TO {sql_string(output_path)} (FORMAT PARQUET)")
    row = connection.execute(
        f"SELECT COUNT(*) FROM {parquet_scan(output_path)}"
    ).fetchone()
    assert row is not None
    return int(row[0])


def combine_score_tables(
    connection: duckdb.DuckDBPyConnection,
    *,
    score_paths: Sequence[Path],
    output_path: Path,
) -> int:
    """Union score parquet tables into the canonical baseline_scores artifact."""

    existing_paths = [path for path in score_paths if path.exists()]
    if not existing_paths:
        raise ValueError("no baseline score tables were produced")
    if len(existing_paths) == 1:
        connection.execute(
            f"COPY (SELECT * FROM {parquet_scan(existing_paths[0])}) "
            f"TO {sql_string(output_path)} (FORMAT PARQUET)"
        )
    else:
        unions = " UNION ALL ".join(
            f"SELECT * FROM {parquet_scan(path)}" for path in existing_paths
        )
        connection.execute(
            f"COPY ({unions}) TO {sql_string(output_path)} (FORMAT PARQUET)"
        )
    row = connection.execute(
        f"SELECT COUNT(*) FROM {parquet_scan(output_path)}"
    ).fetchone()
    assert row is not None
    return int(row[0])
