from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


TARGET = "label_prescribed"
PATIENT_ID = "patient_id"
CONDITION = "condition"
MEDICATION = "medication"
SPLIT = "split"
GROUP_COLUMNS = [PATIENT_ID, CONDITION]


# These are labels or directly observed medication-outcome details.
ALWAYS_DROP_COLUMNS = {
    PATIENT_ID,
    TARGET,
    "label_high_adherence",
    "prescription_count",
    "adherence_pct_mean",
    "adherence_pct_min",
    "adherence_pct_max",
    "dose_mean",
    "duration_days_total",
    "is_generic_any",
    "first_medication_start_date",
    "last_medication_start_date",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an XGBoost two-stage medication recommender."
    )
    parser.add_argument(
        "--training-table",
        type=Path,
        default=Path("Datasets") / "processed" / "patient_condition_medication.csv",
        help="Existing patient-condition-medication training table.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Models") / "xgboost_medication_ranker",
        help="Directory for model, metrics, candidate catalog, and recommendations.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        nargs="+",
        default=[1, 3, 5],
        help="K values for ranking metrics.",
    )
    parser.add_argument(
        "--recommendation-k",
        type=int,
        default=5,
        help="Number of top recommendations to save per patient-condition.",
    )
    parser.add_argument(
        "--candidate-top-n",
        type=int,
        default=20,
        help="Top N frequent medication candidates per condition for the catalog. Use 0 for all.",
    )
    parser.add_argument(
        "--min-candidate-positives",
        type=int,
        default=5,
        help="Minimum positive prescriptions needed for a condition-medication candidate.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Optional row limit for quick experiments. Use 0 for all rows.",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=1.0,
        help="Optional random sample fraction for quick experiments.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=300,
        help="Number of boosted trees.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=4,
        help="Maximum tree depth.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="Boosting learning rate.",
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=0.9,
        help="Row subsampling ratio.",
    )
    parser.add_argument(
        "--colsample-bytree",
        type=float,
        default=0.9,
        help="Column subsampling ratio.",
    )
    parser.add_argument(
        "--allow-medication-history-features",
        action="store_true",
        help="Keep medication_* features. Off by default to reduce target leakage.",
    )
    parser.add_argument(
        "--allow-outcome-features",
        action="store_true",
        help="Keep outcome_* features. Off by default because temporal leakage is possible.",
    )
    parser.add_argument(
        "--allow-candidate-popularity-features",
        action="store_true",
        help="Keep candidate_* features. Off by default for a stricter patient-personalized model.",
    )
    return parser.parse_args()


def load_table(path: Path, max_rows: int, sample_frac: float, seed: int) -> pd.DataFrame:
    nrows = max_rows if max_rows > 0 else None
    df = pd.read_csv(path, nrows=nrows, low_memory=False)

    if not 0 < sample_frac <= 1:
        raise ValueError("--sample-frac must be within (0, 1].")
    if sample_frac < 1:
        df = df.sample(frac=sample_frac, random_state=seed).reset_index(drop=True)

    required = {PATIENT_ID, CONDITION, MEDICATION, SPLIT, TARGET}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Training table is missing required columns: {missing}")

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").fillna(0).astype(int)
    return df


def starts_with_any(column: str, prefixes: Iterable[str]) -> bool:
    return any(column.startswith(prefix) for prefix in prefixes)


def select_feature_columns(args: argparse.Namespace, df: pd.DataFrame) -> list[str]:
    drop_columns = set(ALWAYS_DROP_COLUMNS)

    if not args.allow_medication_history_features:
        drop_columns |= {column for column in df.columns if column.startswith("medication_")}
        drop_columns.add("medication_record_count_for_condition")

    if not args.allow_outcome_features:
        drop_columns |= {column for column in df.columns if column.startswith("outcome_")}

    if not args.allow_candidate_popularity_features:
        drop_columns |= {column for column in df.columns if column.startswith("candidate_")}

    drop_columns |= {column for column in df.columns if column.endswith("_date")}

    return [
        column
        for column in df.columns
        if column not in drop_columns and column != SPLIT
    ]


def split_feature_types(df: pd.DataFrame, feature_columns: list[str]) -> tuple[list[str], list[str]]:
    numeric_columns = [
        column
        for column in feature_columns
        if pd.api.types.is_numeric_dtype(df[column])
    ]
    categorical_columns = [
        column for column in feature_columns if column not in numeric_columns
    ]
    return numeric_columns, categorical_columns


def build_candidate_catalog(
    df: pd.DataFrame,
    candidate_top_n: int,
    min_candidate_positives: int,
) -> pd.DataFrame:
    positives = df[df[TARGET].eq(1)]
    catalog = (
        positives.groupby([CONDITION, MEDICATION], as_index=False)
        .agg(
            positive_prescription_count=(TARGET, "size"),
            positive_patient_count=(PATIENT_ID, "nunique"),
            mean_adherence_pct=("adherence_pct_mean", "mean"),
        )
        .sort_values(
            [CONDITION, "positive_prescription_count", "positive_patient_count"],
            ascending=[True, False, False],
        )
    )
    catalog = catalog[
        catalog["positive_prescription_count"].ge(min_candidate_positives)
    ].copy()
    catalog["candidate_rank"] = catalog.groupby(CONDITION).cumcount() + 1

    if candidate_top_n > 0:
        catalog = catalog[catalog["candidate_rank"].le(candidate_top_n)]

    return catalog


def build_pipeline(
    numeric_columns: list[str],
    categorical_columns: list[str],
    args: argparse.Namespace,
    scale_pos_weight: float,
) -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=1.0,
        min_child_weight=5,
        tree_method="hist",
        random_state=args.random_state,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def predict_scores(model: Pipeline, df: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    return model.predict_proba(df[feature_columns])[:, 1]


def binary_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float | None]:
    if len(np.unique(y_true)) < 2:
        return {
            "average_precision": None,
            "roc_auc": None,
        }

    return {
        "average_precision": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
    }


def dcg_at_k(labels: np.ndarray, k: int) -> float:
    labels = labels[:k]
    if labels.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, labels.size + 2))
    return float(np.sum(labels * discounts))


def ranking_metrics(df: pd.DataFrame, score_column: str, top_k_values: list[int]) -> dict[str, float | int]:
    totals = {}
    for k in top_k_values:
        totals[f"precision_at_{k}"] = 0.0
        totals[f"recall_at_{k}"] = 0.0
        totals[f"hit_rate_at_{k}"] = 0.0
        totals[f"ndcg_at_{k}"] = 0.0
        totals[f"mrr_at_{k}"] = 0.0

    group_count = 0
    groups_with_positive = 0

    for _, group in df.groupby(GROUP_COLUMNS, sort=False):
        labels = group[TARGET].to_numpy()
        positive_count = int(labels.sum())
        group_count += 1
        if positive_count == 0:
            continue

        groups_with_positive += 1
        ranked = group.sort_values(score_column, ascending=False)
        ranked_labels = ranked[TARGET].to_numpy()
        positive_positions = np.flatnonzero(ranked_labels == 1)

        for k in top_k_values:
            top_labels = ranked_labels[:k]
            hits = int(top_labels.sum())
            ideal_labels = np.sort(labels)[::-1]
            ideal_dcg = dcg_at_k(ideal_labels, k)

            totals[f"precision_at_{k}"] += hits / k
            totals[f"recall_at_{k}"] += hits / positive_count
            totals[f"hit_rate_at_{k}"] += float(hits > 0)
            totals[f"ndcg_at_{k}"] += (
                dcg_at_k(ranked_labels, k) / ideal_dcg if ideal_dcg > 0 else 0.0
            )

            if positive_positions.size and int(positive_positions[0]) + 1 <= k:
                totals[f"mrr_at_{k}"] += 1.0 / (int(positive_positions[0]) + 1)

    denominator = max(groups_with_positive, 1)
    metrics: dict[str, float | int] = {
        "group_count": int(group_count),
        "groups_with_positive": int(groups_with_positive),
    }
    metrics.update({name: value / denominator for name, value in totals.items()})
    return metrics


def evaluate_split(
    model: Pipeline,
    df: pd.DataFrame,
    feature_columns: list[str],
    split_name: str,
    top_k_values: list[int],
) -> tuple[dict[str, float | int | None], pd.DataFrame]:
    split_df = df[df[SPLIT].eq(split_name)].copy()
    if split_df.empty:
        return {"rows": 0}, split_df

    split_df["score"] = predict_scores(model, split_df, feature_columns)
    y_true = split_df[TARGET].to_numpy()
    y_score = split_df["score"].to_numpy()

    metrics: dict[str, float | int | None] = {
        "rows": int(len(split_df)),
        "positives": int(split_df[TARGET].sum()),
        "positive_rate": float(split_df[TARGET].mean()),
    }
    metrics.update(binary_metrics(y_true, y_score))
    metrics.update(ranking_metrics(split_df, "score", top_k_values))
    return metrics, split_df


def save_top_recommendations(
    scored_df: pd.DataFrame,
    output_path: Path,
    recommendation_k: int,
) -> None:
    columns = [
        PATIENT_ID,
        CONDITION,
        MEDICATION,
        SPLIT,
        TARGET,
        "score",
    ]
    optional_columns = [
        "has_patient_dx_flag",
        "diagnosis_visit_count",
        "age",
        "sex",
        "charlson_index",
    ]
    columns += [column for column in optional_columns if column in scored_df.columns]

    top_recommendations = (
        scored_df.sort_values([PATIENT_ID, CONDITION, "score"], ascending=[True, True, False])
        .groupby(GROUP_COLUMNS, as_index=False)
        .head(recommendation_k)[columns]
    )
    top_recommendations.to_csv(output_path, index=False)


def train_and_evaluate(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_table(
        args.training_table,
        max_rows=args.max_rows,
        sample_frac=args.sample_frac,
        seed=args.random_state,
    )
    train_df = df[df[SPLIT].eq("train")].copy()
    if train_df.empty:
        raise ValueError("No rows found where split == 'train'.")

    candidate_catalog = build_candidate_catalog(
        df,
        candidate_top_n=args.candidate_top_n,
        min_candidate_positives=args.min_candidate_positives,
    )
    candidate_catalog.to_csv(args.output_dir / "candidate_catalog.csv", index=False)

    feature_columns = select_feature_columns(args, df)
    numeric_columns, categorical_columns = split_feature_types(df, feature_columns)

    positive_count = int(train_df[TARGET].sum())
    negative_count = int((train_df[TARGET] == 0).sum())
    scale_pos_weight = negative_count / max(positive_count, 1)

    pipeline = build_pipeline(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        args=args,
        scale_pos_weight=scale_pos_weight,
    )
    pipeline.fit(train_df[feature_columns], train_df[TARGET])

    metrics: dict[str, object] = {
        "training_table": str(args.training_table),
        "rows_loaded": int(len(df)),
        "candidate_catalog_rows": int(len(candidate_catalog)),
        "feature_count": len(feature_columns),
        "numeric_feature_count": len(numeric_columns),
        "categorical_feature_count": len(categorical_columns),
        "features_used": feature_columns,
        "xgboost_params": {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "scale_pos_weight": scale_pos_weight,
        },
        "leakage_controls": {
            "allow_medication_history_features": args.allow_medication_history_features,
            "allow_outcome_features": args.allow_outcome_features,
            "allow_candidate_popularity_features": args.allow_candidate_popularity_features,
        },
        "splits": {},
    }

    scored_frames = []
    for split_name in ["train", "valid", "test"]:
        split_metrics, scored_split = evaluate_split(
            model=pipeline,
            df=df,
            feature_columns=feature_columns,
            split_name=split_name,
            top_k_values=args.top_k,
        )
        metrics["splits"][split_name] = split_metrics
        if not scored_split.empty:
            scored_frames.append(scored_split)

    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    with (args.output_dir / "model.pkl").open("wb") as file:
        pickle.dump(
            {
                "pipeline": pipeline,
                "feature_columns": feature_columns,
                "numeric_columns": numeric_columns,
                "categorical_columns": categorical_columns,
                "candidate_catalog": candidate_catalog,
            },
            file,
        )

    if scored_frames:
        scored_df = pd.concat(scored_frames, ignore_index=True)
        save_top_recommendations(
            scored_df,
            output_path=args.output_dir / "top_recommendations.csv",
            recommendation_k=args.recommendation_k,
        )
        scored_df[
            [PATIENT_ID, CONDITION, MEDICATION, SPLIT, TARGET, "score"]
        ].to_csv(args.output_dir / "scored_candidates.csv", index=False)


def main() -> None:
    train_and_evaluate(parse_args())


if __name__ == "__main__":
    main()
