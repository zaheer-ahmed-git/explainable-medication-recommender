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
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "label_prescribed"
GROUP_COLUMNS = ["patient_id", "condition"]
ITEM_COLUMN = "medication"
SPLIT_COLUMN = "split"


ALWAYS_DROP_COLUMNS = {
    "patient_id",
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


LEAKY_PREFIXES = (
    "medication_",
    "outcome_",
)


POPULARITY_PREFIXES = (
    "candidate_",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a baseline medication ranking model."
    )
    parser.add_argument(
        "--training-table",
        type=Path,
        default=Path("Datasets") / "processed" / "patient_condition_medication.csv",
        help="Path to patient_condition_medication.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Models") / "medication_ranker",
        help="Directory for model and metric outputs.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        nargs="+",
        default=[1, 3, 5],
        help="K values for ranking metrics.",
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
        "--allow-medication-history-features",
        action="store_true",
        help="Keep medication_* features. Off by default to avoid target leakage.",
    )
    parser.add_argument(
        "--allow-outcome-features",
        action="store_true",
        help="Keep outcome_* features. Off by default because temporal leakage is possible.",
    )
    parser.add_argument(
        "--allow-popularity-features",
        action="store_true",
        help="Keep candidate_* features. Off by default because they were computed globally.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0001,
        help="SGDClassifier L2 regularization strength.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=25,
        help="Maximum SGD epochs.",
    )
    return parser.parse_args()


def load_training_table(path: Path, max_rows: int, sample_frac: float, seed: int) -> pd.DataFrame:
    nrows = max_rows if max_rows > 0 else None
    df = pd.read_csv(path, nrows=nrows, low_memory=False)

    if not 0 < sample_frac <= 1:
        raise ValueError("--sample-frac must be within (0, 1].")
    if sample_frac < 1:
        df = df.sample(frac=sample_frac, random_state=seed).reset_index(drop=True)

    required = {TARGET, SPLIT_COLUMN, ITEM_COLUMN, *GROUP_COLUMNS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Training table is missing required columns: {missing}")

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").fillna(0).astype(int)
    return df


def columns_with_prefix(columns: Iterable[str], prefixes: Iterable[str]) -> set[str]:
    return {
        column
        for column in columns
        if any(column.startswith(prefix) for prefix in prefixes)
    }


def select_feature_columns(
    df: pd.DataFrame,
    allow_medication_history_features: bool,
    allow_outcome_features: bool,
    allow_popularity_features: bool,
) -> list[str]:
    drop_columns = set(ALWAYS_DROP_COLUMNS)

    if not allow_medication_history_features:
        drop_columns |= columns_with_prefix(df.columns, ["medication_"])
        drop_columns.add("medication_record_count_for_condition")

    if not allow_outcome_features:
        drop_columns |= columns_with_prefix(df.columns, ["outcome_"])

    if not allow_popularity_features:
        drop_columns |= columns_with_prefix(df.columns, POPULARITY_PREFIXES)

    # These dates are awkward for a first linear baseline and are not needed yet.
    drop_columns |= {column for column in df.columns if column.endswith("_date")}

    feature_columns = [
        column
        for column in df.columns
        if column not in drop_columns and column != SPLIT_COLUMN
    ]
    return feature_columns


def split_features_by_type(df: pd.DataFrame, feature_columns: list[str]) -> tuple[list[str], list[str]]:
    numeric_columns = [
        column
        for column in feature_columns
        if pd.api.types.is_numeric_dtype(df[column])
    ]
    categorical_columns = [
        column
        for column in feature_columns
        if column not in numeric_columns
    ]
    return numeric_columns, categorical_columns


def build_pipeline(
    numeric_columns: list[str],
    categorical_columns: list[str],
    alpha: float,
    max_iter: int,
    random_state: int,
) -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
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

    classifier = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=alpha,
        class_weight="balanced",
        max_iter=max_iter,
        tol=1e-3,
        random_state=random_state,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def predict_scores(model: Pipeline, df: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(df[feature_columns])[:, 1]

    decision = model.decision_function(df[feature_columns])
    return 1.0 / (1.0 + np.exp(-decision))


def binary_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {
        "average_precision": None,
        "roc_auc": None,
    }
    if len(np.unique(y_true)) < 2:
        return metrics

    metrics["average_precision"] = float(average_precision_score(y_true, y_score))
    metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    return metrics


def dcg_at_k(labels: np.ndarray, k: int) -> float:
    labels = labels[:k]
    if labels.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, labels.size + 2))
    return float(np.sum(labels * discounts))


def evaluate_ranking(
    df: pd.DataFrame,
    score_column: str,
    top_k_values: list[int],
) -> dict[str, float | int]:
    grouped = df.groupby(GROUP_COLUMNS, sort=False)
    group_count = 0
    groups_with_positive = 0

    totals: dict[str, float] = {}
    for k in top_k_values:
        totals[f"precision_at_{k}"] = 0.0
        totals[f"recall_at_{k}"] = 0.0
        totals[f"hit_rate_at_{k}"] = 0.0
        totals[f"ndcg_at_{k}"] = 0.0
        totals[f"mrr_at_{k}"] = 0.0

    for _, group in grouped:
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

            first_positive_rank = (
                int(positive_positions[0]) + 1 if positive_positions.size else None
            )
            if first_positive_rank is not None and first_positive_rank <= k:
                totals[f"mrr_at_{k}"] += 1.0 / first_positive_rank

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
    split_df = df[df[SPLIT_COLUMN].eq(split_name)].copy()
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
    metrics.update(evaluate_ranking(split_df, "score", top_k_values))
    return metrics, split_df


def train_and_evaluate(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_training_table(
        path=args.training_table,
        max_rows=args.max_rows,
        sample_frac=args.sample_frac,
        seed=args.random_state,
    )

    train_df = df[df[SPLIT_COLUMN].eq("train")].copy()
    if train_df.empty:
        raise ValueError("No training rows found where split == 'train'.")

    feature_columns = select_feature_columns(
        df,
        allow_medication_history_features=args.allow_medication_history_features,
        allow_outcome_features=args.allow_outcome_features,
        allow_popularity_features=args.allow_popularity_features,
    )
    numeric_columns, categorical_columns = split_features_by_type(df, feature_columns)

    model = build_pipeline(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        alpha=args.alpha,
        max_iter=args.max_iter,
        random_state=args.random_state,
    )
    model.fit(train_df[feature_columns], train_df[TARGET])

    all_metrics: dict[str, object] = {
        "training_table": str(args.training_table),
        "rows_loaded": int(len(df)),
        "features_used": feature_columns,
        "numeric_feature_count": len(numeric_columns),
        "categorical_feature_count": len(categorical_columns),
        "top_k": args.top_k,
        "leakage_controls": {
            "allow_medication_history_features": args.allow_medication_history_features,
            "allow_outcome_features": args.allow_outcome_features,
            "allow_popularity_features": args.allow_popularity_features,
        },
        "splits": {},
    }

    scored_frames = []
    for split_name in ["train", "valid", "test"]:
        split_metrics, scored_split = evaluate_split(
            model=model,
            df=df,
            feature_columns=feature_columns,
            split_name=split_name,
            top_k_values=args.top_k,
        )
        all_metrics["splits"][split_name] = split_metrics
        if not scored_split.empty:
            scored_frames.append(
                scored_split[
                    [
                        "patient_id",
                        "condition",
                        "medication",
                        "split",
                        TARGET,
                        "score",
                    ]
                ]
            )

    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(all_metrics, file, indent=2)

    with (args.output_dir / "model.pkl").open("wb") as file:
        pickle.dump(
            {
                "model": model,
                "feature_columns": feature_columns,
                "numeric_columns": numeric_columns,
                "categorical_columns": categorical_columns,
            },
            file,
        )

    if scored_frames:
        scored = pd.concat(scored_frames, ignore_index=True)
        scored.to_csv(args.output_dir / "scored_candidates.csv", index=False)


def main() -> None:
    train_and_evaluate(parse_args())


if __name__ == "__main__":
    main()
