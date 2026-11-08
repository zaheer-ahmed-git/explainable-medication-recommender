"""Leakage-safe probability calibration for the frozen paired-OOF ranker.

The ranker and its selected ``(GNN variant, alpha)`` remain immutable.  This
module learns monotone mappings from raw branch rank scores to the probability
of an observed historical prescription using only patient-grouped MIMIC-train
out-of-fold predictions.  Restricted calibration parameters stay below the
versioned paired-protocol root; public reports contain aggregates only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
import math
from pathlib import Path
from time import perf_counter
from typing import Any

import duckdb
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit

from pipeline.extract_utils import parquet_scan, safe_error_message
from pipeline.gnn_training.config import (
    LATE_FUSION_BASELINE_NAME,
    TRANSFORMER_BASELINE_NAME,
    GNNTrainingConfig,
)
from pipeline.gnn_training.contract import (
    blocked_report,
    contract_digest_or_none,
    preflight_errors,
)
from pipeline.gnn_training.data import configure_connection, write_json
from pipeline.gnn_training.paired_oof import (
    _completed_gnn_artifact,
    _completed_transformer_artifacts,
    _scan_paths,
    _validate_pairing,
)
from pipeline.late_fusion_protocol import (
    PAIRED_CALIBRATION_PROTOCOL_VERSION,
    PAIRED_CALIBRATION_REPORT_SCHEMA_VERSION,
    PAIRED_CALIBRATION_SCHEMA_VERSION,
    PAIRED_OOF_PROTOCOL_VERSION,
    PAIRED_OOF_SELECTION_SCHEMA_VERSION,
)
from pipeline.training_contract import load_json, sha256_file

TEMPERATURE_METHOD = "temperature"
AFFINE_SIGMOID_METHOD = "affine_sigmoid"
ISOTONIC_METHOD = "isotonic"
DEFAULT_METHODS = (TEMPERATURE_METHOD, AFFINE_SIGMOID_METHOD)
METHOD_TIE_PRIORITY = {
    AFFINE_SIGMOID_METHOD: 0,
    TEMPERATURE_METHOD: 1,
    ISOTONIC_METHOD: 2,
}
TEMPERATURE_BOUNDS = (0.05, 10.0)
AFFINE_SLOPE_BOUNDS = (1e-4, 100.0)
PROBABILITY_EPSILON = 1e-12
RELIABILITY_BIN_COUNT = 10
MIN_REPORTING_BIN_ROWS = 25
LOCKED_GNN_VARIANT = "full"
LOCKED_GNN_ALPHA = 0.25
LOCKED_ALPHA_SEMANTICS = "(1-alpha)*zscore(transformer)+alpha*zscore(gnn)"


def _as_score_array(scores: Sequence[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("calibration scores must be a finite, non-empty vector")
    return values


def _as_label_array(labels: Sequence[bool | int] | np.ndarray) -> np.ndarray:
    values = np.asarray(labels, dtype=np.float64)
    if (
        values.ndim != 1
        or values.size == 0
        or not np.isfinite(values).all()
        or not np.isin(values, (0.0, 1.0)).all()
    ):
        raise ValueError("calibration labels must be a non-empty binary vector")
    return values


def _validate_fit_inputs(
    scores: Sequence[float] | np.ndarray,
    labels: Sequence[bool | int] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    score_array = _as_score_array(scores)
    label_array = _as_label_array(labels)
    if score_array.shape != label_array.shape:
        raise ValueError("calibration scores and labels must have the same shape")
    if np.unique(label_array).size != 2:
        raise ValueError("calibration fitting requires both label classes")
    return score_array, label_array


def _mean_log_loss_from_logit(logits: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(np.logaddexp(0.0, logits) - labels * logits))


def _fit_temperature(scores: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    lower, upper = (math.log(value) for value in TEMPERATURE_BOUNDS)

    def objective(log_temperature: float) -> float:
        return _mean_log_loss_from_logit(
            scores / math.exp(float(log_temperature)),
            labels,
        )

    result = minimize_scalar(
        objective,
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": 1e-8},
    )
    if not result.success or not math.isfinite(float(result.x)):
        raise RuntimeError("temperature calibration did not converge")
    return {
        "method": TEMPERATURE_METHOD,
        "parameters": {"temperature": math.exp(float(result.x))},
        "fit_objective": "unweighted_candidate_row_binary_log_loss",
        "strictly_order_preserving": True,
    }


def _fit_affine_sigmoid(scores: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    prevalence = float(labels.mean())
    initial_intercept = math.log(prevalence / (1.0 - prevalence))
    log_slope_bounds = tuple(math.log(value) for value in AFFINE_SLOPE_BOUNDS)

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        log_slope, intercept = (float(value) for value in parameters)
        slope = math.exp(log_slope)
        logits = intercept + slope * scores
        probabilities = expit(logits)
        residual = probabilities - labels
        gradient = np.array(
            [
                float(np.mean(residual * slope * scores)),
                float(np.mean(residual)),
            ],
            dtype=np.float64,
        )
        return _mean_log_loss_from_logit(logits, labels), gradient

    result = minimize(
        objective,
        np.array([0.0, initial_intercept], dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        bounds=(log_slope_bounds, (-20.0, 20.0)),
        options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 200},
    )
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError("affine-sigmoid calibration did not converge")
    return {
        "method": AFFINE_SIGMOID_METHOD,
        "parameters": {
            "slope": math.exp(float(result.x[0])),
            "intercept": float(result.x[1]),
        },
        "fit_objective": "unweighted_candidate_row_binary_log_loss",
        "strictly_order_preserving": True,
    }


def _fit_isotonic(scores: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    from sklearn.isotonic import IsotonicRegression

    model = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        increasing=True,
        out_of_bounds="clip",
    )
    model.fit(scores, labels)
    x_thresholds = np.asarray(model.X_thresholds_, dtype=np.float64)
    y_thresholds = np.asarray(model.y_thresholds_, dtype=np.float64)
    if (
        x_thresholds.size < 2
        or x_thresholds.shape != y_thresholds.shape
        or not np.isfinite(x_thresholds).all()
        or not np.isfinite(y_thresholds).all()
    ):
        raise RuntimeError("isotonic calibration produced invalid thresholds")
    return {
        "method": ISOTONIC_METHOD,
        "parameters": {
            "x_thresholds": x_thresholds.tolist(),
            "y_thresholds": y_thresholds.tolist(),
        },
        "fit_objective": "unweighted_candidate_row_brier_score",
        "strictly_order_preserving": False,
        "non_decreasing": True,
    }


def fit_calibrator(
    method: str,
    scores: Sequence[float] | np.ndarray,
    labels: Sequence[bool | int] | np.ndarray,
) -> dict[str, Any]:
    """Fit one registered monotone calibration mapping."""

    score_array, label_array = _validate_fit_inputs(scores, labels)
    if method == TEMPERATURE_METHOD:
        return _fit_temperature(score_array, label_array)
    if method == AFFINE_SIGMOID_METHOD:
        return _fit_affine_sigmoid(score_array, label_array)
    if method == ISOTONIC_METHOD:
        return _fit_isotonic(score_array, label_array)
    raise ValueError(f"unsupported calibration method: {method!r}")


def apply_calibrator(
    scores: Sequence[float] | np.ndarray,
    calibrator: Mapping[str, Any],
) -> np.ndarray:
    """Apply a validated calibration mapping without changing the rank score."""

    score_array = _as_score_array(scores)
    method = calibrator.get("method")
    parameters = calibrator.get("parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError("calibration parameters must be an object")
    if method == TEMPERATURE_METHOD:
        temperature = float(parameters.get("temperature", 0.0))
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("calibration temperature must be positive")
        probabilities = expit(score_array / temperature)
    elif method == AFFINE_SIGMOID_METHOD:
        slope = float(parameters.get("slope", 0.0))
        intercept = float(parameters.get("intercept", math.nan))
        if not math.isfinite(slope) or slope <= 0.0 or not math.isfinite(intercept):
            raise ValueError("affine calibration requires a positive slope")
        probabilities = expit(intercept + slope * score_array)
    elif method == ISOTONIC_METHOD:
        x_thresholds = np.asarray(parameters.get("x_thresholds"), dtype=np.float64)
        y_thresholds = np.asarray(parameters.get("y_thresholds"), dtype=np.float64)
        if (
            x_thresholds.ndim != 1
            or x_thresholds.size < 2
            or x_thresholds.shape != y_thresholds.shape
            or not np.isfinite(x_thresholds).all()
            or not np.isfinite(y_thresholds).all()
            or np.any(np.diff(x_thresholds) <= 0.0)
            or np.any(np.diff(y_thresholds) < 0.0)
        ):
            raise ValueError("isotonic calibration thresholds are invalid")
        probabilities = np.interp(
            score_array,
            x_thresholds,
            y_thresholds,
            left=float(y_thresholds[0]),
            right=float(y_thresholds[-1]),
        )
    else:
        raise ValueError("calibration method is missing or unsupported")
    if not np.isfinite(probabilities).all():
        raise ValueError("calibration produced non-finite probabilities")
    return np.clip(probabilities, 0.0, 1.0)


def calibrated_output_fields(
    rank_scores: Sequence[float] | np.ndarray,
    calibrator: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Return explicitly separated ranking and probability output fields."""

    rank_score = _as_score_array(rank_scores).copy()
    return {
        "rank_score": rank_score,
        "probability_observed_prescription": apply_calibrator(
            rank_score,
            calibrator,
        ),
    }


def probability_metrics(
    labels: Sequence[bool | int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
) -> dict[str, Any]:
    """Return candidate-row calibration metrics for one held-out score vector."""

    label_array = _as_label_array(labels)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    if (
        probability_array.shape != label_array.shape
        or not np.isfinite(probability_array).all()
        or np.any((probability_array < 0.0) | (probability_array > 1.0))
    ):
        raise ValueError("calibrated probabilities must be finite and in [0, 1]")
    clipped = np.clip(
        probability_array,
        PROBABILITY_EPSILON,
        1.0 - PROBABILITY_EPSILON,
    )
    bin_ids = np.minimum(
        RELIABILITY_BIN_COUNT - 1,
        np.floor(probability_array * RELIABILITY_BIN_COUNT).astype(np.int64),
    )
    ece = 0.0
    for bin_index in range(RELIABILITY_BIN_COUNT):
        mask = bin_ids == bin_index
        if np.any(mask):
            ece += float(mask.mean()) * abs(
                float(label_array[mask].mean()) - float(probability_array[mask].mean())
            )
    return {
        "row_count": int(label_array.size),
        "positive_row_count": int(label_array.sum()),
        "negative_row_count": int(label_array.size - label_array.sum()),
        "positive_rate": float(label_array.mean()),
        "mean_predicted_probability": float(probability_array.mean()),
        "brier_score": float(np.mean((probability_array - label_array) ** 2)),
        "log_loss": float(
            -np.mean(
                label_array * np.log(clipped) + (1.0 - label_array) * np.log1p(-clipped)
            )
        ),
        "expected_calibration_error_10bin": ece,
    }


def _reliability_bins(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> list[dict[str, Any]]:
    bin_ids = np.minimum(
        RELIABILITY_BIN_COUNT - 1,
        np.floor(probabilities * RELIABILITY_BIN_COUNT).astype(np.int64),
    )
    rows: list[dict[str, Any]] = []
    for bin_index in range(RELIABILITY_BIN_COUNT):
        mask = bin_ids == bin_index
        count = int(mask.sum())
        if count == 0:
            continue
        positive_count = int(labels[mask].sum())
        negative_count = count - positive_count
        reportable = (
            count >= MIN_REPORTING_BIN_ROWS
            and positive_count >= MIN_REPORTING_BIN_ROWS
            and negative_count >= MIN_REPORTING_BIN_ROWS
        )
        rows.append(
            {
                "bin_index": bin_index,
                "lower_bound_inclusive": bin_index / RELIABILITY_BIN_COUNT,
                "upper_bound_exclusive": (
                    (bin_index + 1) / RELIABILITY_BIN_COUNT
                    if bin_index < RELIABILITY_BIN_COUNT - 1
                    else None
                ),
                "row_count": count if reportable else None,
                "positive_row_count": positive_count if reportable else None,
                "negative_row_count": negative_count if reportable else None,
                "mean_predicted_probability": (
                    float(probabilities[mask].mean()) if reportable else None
                ),
                "observed_positive_rate": (
                    float(labels[mask].mean()) if reportable else None
                ),
                "reporting_status": (
                    "reported" if reportable else "suppressed_low_count"
                ),
            }
        )
    return rows


def cross_validate_calibrators(
    scores: Sequence[float] | np.ndarray,
    labels: Sequence[bool | int] | np.ndarray,
    patient_fold_ids: Sequence[int] | np.ndarray,
    *,
    fold_count: int,
    methods: Sequence[str] = DEFAULT_METHODS,
) -> dict[str, Any]:
    """Select one mapping using patient-fold-held-out calibration predictions."""

    score_array, label_array = _validate_fit_inputs(scores, labels)
    fold_array = np.asarray(patient_fold_ids, dtype=np.int64)
    if fold_array.shape != score_array.shape:
        raise ValueError("patient fold ids must match calibration scores")
    expected_folds = set(range(fold_count))
    if set(int(value) for value in np.unique(fold_array)) != expected_folds:
        raise ValueError("calibration requires every registered patient fold")
    registered_methods = tuple(dict.fromkeys(str(method) for method in methods))
    if not registered_methods or any(
        method not in METHOD_TIE_PRIORITY for method in registered_methods
    ):
        raise ValueError("calibration methods are empty or unsupported")

    method_results: dict[str, dict[str, Any]] = {}
    for method in registered_methods:
        cross_fitted = np.empty_like(score_array)
        fold_metrics: list[dict[str, Any]] = []
        for held_out_fold in range(fold_count):
            fit_mask = fold_array != held_out_fold
            held_out_mask = fold_array == held_out_fold
            calibrator = fit_calibrator(
                method,
                score_array[fit_mask],
                label_array[fit_mask],
            )
            probabilities = apply_calibrator(
                score_array[held_out_mask],
                calibrator,
            )
            cross_fitted[held_out_mask] = probabilities
            fold_metrics.append(
                {
                    "held_out_patient_fold": held_out_fold,
                    "fit_fold_indices": sorted(expected_folds - {held_out_fold}),
                    **probability_metrics(label_array[held_out_mask], probabilities),
                }
            )
        metrics = probability_metrics(label_array, cross_fitted)
        method_results[method] = {
            "method": method,
            "cross_fitted_metrics": metrics,
            "fold_metrics": fold_metrics,
            "reliability_bins": _reliability_bins(label_array, cross_fitted),
            "strictly_order_preserving": method != ISOTONIC_METHOD,
        }

    selected_method = min(
        registered_methods,
        key=lambda method: (
            float(method_results[method]["cross_fitted_metrics"]["brier_score"]),
            float(method_results[method]["cross_fitted_metrics"]["log_loss"]),
            METHOD_TIE_PRIORITY[method],
        ),
    )
    final_calibrator = fit_calibrator(selected_method, score_array, label_array)
    final_calibrator.update(
        {
            "fit_scope": "mimiciv_train_paired_patient_grouped_oof_only",
            "fit_row_count": int(label_array.size),
            "fit_positive_row_count": int(label_array.sum()),
            "fit_positive_rate": float(label_array.mean()),
        }
    )
    return {
        "selection_policy": "minimum_cross_fitted_brier_then_log_loss",
        "selected_method": selected_method,
        "method_results": method_results,
        "final_calibrator": final_calibrator,
    }


def _selection_and_model_provenance(
    config: GNNTrainingConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    selection = load_json(config.paired_oof_selection_path)
    late_checkpoint = load_json(config.paired_late_checkpoint_path)
    if (
        selection.get("schema_version") != PAIRED_OOF_SELECTION_SCHEMA_VERSION
        or selection.get("protocol_version") != PAIRED_OOF_PROTOCOL_VERSION
        or selection.get("selection_frozen") is not True
        or late_checkpoint.get("status") != "completed"
        or late_checkpoint.get("inference_ready") is not True
        or late_checkpoint.get("selected_gnn_variant")
        != selection.get("selected_gnn_variant")
        or float(late_checkpoint.get("selected_alpha", -1.0))
        != float(selection.get("selected_alpha", -2.0))
        or selection.get("selected_gnn_variant") != LOCKED_GNN_VARIANT
        or float(selection.get("selected_alpha", -1.0)) != LOCKED_GNN_ALPHA
        or selection.get("alpha_semantics") != LOCKED_ALPHA_SEMANTICS
    ):
        raise ValueError("paired selection/refit metadata is not calibration-ready")
    gnn_lock = late_checkpoint.get("gnn_checkpoint")
    if not isinstance(gnn_lock, Mapping):
        raise ValueError("paired GNN refit lock is missing")
    gnn_checkpoint = Path(str(gnn_lock.get("path", "")))
    allowed_gnn_checkpoints = {
        config.gnn_checkpoint_path.resolve(),
        config.paired_gnn_checkpoint_path.resolve(),
    }
    if (
        not gnn_checkpoint.is_file()
        or gnn_lock.get("sha256") != sha256_file(gnn_checkpoint)
        or gnn_checkpoint.resolve() not in allowed_gnn_checkpoints
    ):
        raise ValueError("paired GNN refit lock does not match its checkpoint")
    transformer_hash = sha256_file(config.neural_checkpoint_path)
    if selection.get("transformer_checkpoint_sha256") != transformer_hash:
        raise ValueError("paired Transformer checkpoint changed after selection")
    return selection, {
        "paired_oof_selection_sha256": sha256_file(config.paired_oof_selection_path),
        "paired_late_checkpoint_sha256": sha256_file(
            config.paired_late_checkpoint_path
        ),
        "transformer_checkpoint_sha256": transformer_hash,
        "gnn_refit_checkpoint_sha256": sha256_file(gnn_checkpoint),
    }


def load_paired_branch_calibrator(
    config: GNNTrainingConfig,
    *,
    baseline_name: str,
) -> dict[str, Any]:
    """Load one frozen mapping and verify its current model provenance."""

    if baseline_name not in {LATE_FUSION_BASELINE_NAME, TRANSFORMER_BASELINE_NAME}:
        raise ValueError("paired calibration baseline is unsupported")
    artifact = load_json(config.paired_calibration_path)
    provenance = artifact.get("provenance")
    mappings = artifact.get("branch_mappings")
    if (
        artifact.get("schema_version") != PAIRED_CALIBRATION_SCHEMA_VERSION
        or artifact.get("calibration_protocol_version")
        != PAIRED_CALIBRATION_PROTOCOL_VERSION
        or artifact.get("ranking_protocol_version") != PAIRED_OOF_PROTOCOL_VERSION
        or artifact.get("status") != "completed"
        or artifact.get("calibration_frozen") is not True
        or not isinstance(provenance, Mapping)
        or not isinstance(mappings, Mapping)
    ):
        raise ValueError("paired calibration artifact violates its contract")
    selection, model_provenance = _selection_and_model_provenance(config)
    expected = {
        "seed": config.seed,
        "fold_count": config.fold_count,
        "selected_gnn_variant": selection["selected_gnn_variant"],
        "selected_alpha": float(selection["selected_alpha"]),
        "alpha_semantics": selection["alpha_semantics"],
        "transformer_oof_sha256": selection["transformer_oof_sha256"],
        "selected_gnn_oof_sha256": selection["gnn_oof_sha256"],
        **model_provenance,
    }
    if any(provenance.get(name) != value for name, value in expected.items()):
        raise ValueError("paired calibration model provenance has changed")
    branch_mapping = mappings.get(baseline_name)
    if not isinstance(branch_mapping, Mapping):
        raise ValueError("paired calibration branch mapping is missing")
    calibrator = branch_mapping.get("final_calibrator")
    if not isinstance(calibrator, Mapping):
        raise ValueError("paired final calibrator is missing")
    # Validate parameters eagerly with a finite sentinel without exposing rows.
    apply_calibrator(np.array([0.0], dtype=np.float64), calibrator)
    return dict(calibrator)


def _load_paired_oof_scores(
    config: GNNTrainingConfig,
    *,
    selection: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    variant = str(selection["selected_gnn_variant"])
    alpha = float(selection["selected_alpha"])
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("selected paired alpha is outside [0, 1]")
    transformer_paths = _completed_transformer_artifacts(config)
    gnn_path = _completed_gnn_artifact(config, variant)
    transformer_hashes = [sha256_file(path) for path in transformer_paths]
    gnn_hash = sha256_file(gnn_path)
    if (
        selection.get("transformer_oof_sha256") != transformer_hashes
        or selection.get("gnn_oof_sha256") != gnn_hash
    ):
        raise ValueError("paired OOF prediction hashes changed after selection")
    transformer_scan = _scan_paths(transformer_paths)
    with duckdb.connect(database=":memory:") as connection:
        configure_connection(config, connection)
        row_count = _validate_pairing(
            connection,
            transformer_scan=transformer_scan,
            gnn_path=gnn_path,
            fold_count=config.fold_count,
        )
        invalid_group_count = connection.execute(
            f"""
SELECT COUNT(*)
FROM (
    SELECT ranking_group_id
    FROM {transformer_scan}
    GROUP BY ranking_group_id
    HAVING COUNT(DISTINCT patient_fold_id) <> 1
) AS invalid_groups
"""
        ).fetchone()
        if invalid_group_count is None or int(invalid_group_count[0]) != 0:
            raise ValueError("paired OOF ranking groups cross patient folds")
        values = connection.execute(
            f"""
WITH paired AS (
    SELECT
        transformer.source,
        transformer.split,
        transformer.ranking_group_id,
        transformer.patient_fold_id,
        CAST(transformer.label_prescribed AS DOUBLE) AS label_prescribed,
        CAST(transformer.transformer_logit AS DOUBLE) AS transformer_logit,
        CAST(gnn.gnn_logit AS DOUBLE) AS gnn_logit
    FROM {transformer_scan} AS transformer
    INNER JOIN {parquet_scan(gnn_path)} AS gnn
        USING (
            source,
            split,
            ranking_group_id,
            index_condition_token,
            candidate_medication_token,
            candidate_rank,
            label_prescribed,
            patient_fold_id
        )
), moments AS (
    SELECT
        *,
        AVG(transformer_logit) OVER score_group AS transformer_mean,
        STDDEV_POP(transformer_logit) OVER score_group AS transformer_std,
        AVG(gnn_logit) OVER score_group AS gnn_mean,
        STDDEV_POP(gnn_logit) OVER score_group AS gnn_std
    FROM paired
    WINDOW score_group AS (
        PARTITION BY source, split, ranking_group_id
    )
)
SELECT
    CAST(patient_fold_id AS INTEGER) AS patient_fold_id,
    label_prescribed,
    transformer_logit,
    (1.0 - {alpha})
        * CASE
            WHEN transformer_std < 1e-6 THEN 0.0
            ELSE (transformer_logit - transformer_mean) / transformer_std
          END
        + {alpha}
        * CASE
            WHEN gnn_std < 1e-6 THEN 0.0
            ELSE (gnn_logit - gnn_mean) / gnn_std
          END AS hybrid_rank_score
FROM moments
"""
        ).fetchnumpy()
    folds = np.asarray(values["patient_fold_id"], dtype=np.int64)
    labels = np.asarray(values["label_prescribed"], dtype=np.float64)
    transformer_scores = np.asarray(values["transformer_logit"], dtype=np.float64)
    hybrid_scores = np.asarray(values["hybrid_rank_score"], dtype=np.float64)
    if (
        labels.size != row_count
        or folds.shape != labels.shape
        or transformer_scores.shape != labels.shape
        or hybrid_scores.shape != labels.shape
        or not np.isfinite(transformer_scores).all()
        or not np.isfinite(hybrid_scores).all()
    ):
        raise ValueError("paired OOF calibration projection is invalid")
    return (
        {
            "labels": labels,
            "patient_fold_ids": folds,
            LATE_FUSION_BASELINE_NAME: hybrid_scores,
            TRANSFORMER_BASELINE_NAME: transformer_scores,
        },
        {
            "transformer_oof_sha256": transformer_hashes,
            "selected_gnn_oof_sha256": gnn_hash,
        },
    )


def _mapping_report(mapping: Mapping[str, Any]) -> dict[str, Any]:
    final = mapping["final_calibrator"]
    parameters = final["parameters"]
    final_summary: dict[str, Any] = {
        "method": final["method"],
        "strictly_order_preserving": final["strictly_order_preserving"],
        "fit_row_count": final["fit_row_count"],
        "fit_positive_row_count": final["fit_positive_row_count"],
        "fit_positive_rate": final["fit_positive_rate"],
    }
    final_negative_count = int(final["fit_row_count"]) - int(
        final["fit_positive_row_count"]
    )
    if (
        int(final["fit_row_count"]) < MIN_REPORTING_BIN_ROWS
        or int(final["fit_positive_row_count"]) < MIN_REPORTING_BIN_ROWS
        or final_negative_count < MIN_REPORTING_BIN_ROWS
    ):
        final_summary.update(
            {
                "fit_row_count": None,
                "fit_positive_row_count": None,
                "fit_positive_rate": None,
                "reporting_status": "suppressed_low_count",
            }
        )
    else:
        final_summary["reporting_status"] = "reported"
    if final["method"] == TEMPERATURE_METHOD:
        final_summary["temperature"] = parameters["temperature"]
    elif final["method"] == AFFINE_SIGMOID_METHOD:
        final_summary.update(
            {
                "slope": parameters["slope"],
                "intercept": parameters["intercept"],
            }
        )
    else:
        final_summary["threshold_count"] = len(parameters["x_thresholds"])

    def public_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
        row_count = int(metrics["row_count"])
        positive_count = int(metrics["positive_row_count"])
        negative_count = int(metrics["negative_row_count"])
        if (
            row_count >= MIN_REPORTING_BIN_ROWS
            and positive_count >= MIN_REPORTING_BIN_ROWS
            and negative_count >= MIN_REPORTING_BIN_ROWS
        ):
            return {**metrics, "reporting_status": "reported"}
        return {
            "row_count": None,
            "positive_row_count": None,
            "negative_row_count": None,
            "positive_rate": None,
            "mean_predicted_probability": None,
            "brier_score": None,
            "log_loss": None,
            "expected_calibration_error_10bin": None,
            "reporting_status": "suppressed_low_count",
        }

    return {
        "selection_policy": mapping["selection_policy"],
        "selected_method": mapping["selected_method"],
        "candidate_methods": {
            method: {
                "cross_fitted_metrics": public_metrics(result["cross_fitted_metrics"]),
                "fold_metrics": [
                    {
                        "held_out_patient_fold": fold["held_out_patient_fold"],
                        "fit_fold_indices": fold["fit_fold_indices"],
                        **public_metrics(fold),
                    }
                    for fold in result["fold_metrics"]
                ],
                "reliability_bins": result["reliability_bins"],
                "strictly_order_preserving": result["strictly_order_preserving"],
            }
            for method, result in mapping["method_results"].items()
        },
        "final_calibrator": final_summary,
    }


def calibrate_paired_oof(
    config: GNNTrainingConfig,
    *,
    include_isotonic: bool = False,
) -> dict[str, Any]:
    """Fit and freeze branch-specific mappings on paired MIMIC-train OOF only."""

    generated_at = datetime.now(UTC).isoformat()
    started = perf_counter()
    methods = (*DEFAULT_METHODS, *((ISOTONIC_METHOD,) if include_isotonic else ()))
    errors = preflight_errors(config, stage="calibrate-paired-oof")
    if config.paired_calibration_path.exists():
        errors.append(
            {
                "code": "paired_calibration_already_frozen",
                "detail": (
                    "the versioned paired-OOF calibration artifact already exists"
                ),
                "artifact_name": "paired_calibration",
            }
        )
    if errors:
        report = blocked_report(
            config=config,
            schema_version=PAIRED_CALIBRATION_REPORT_SCHEMA_VERSION,
            stage="calibrate-paired-oof",
            generated_at=generated_at,
            errors=errors,
        )
        write_json(config.paired_calibration_report_path, report)
        return report

    report: dict[str, Any] = {
        "schema_version": PAIRED_CALIBRATION_REPORT_SCHEMA_VERSION,
        "calibration_protocol_version": PAIRED_CALIBRATION_PROTOCOL_VERSION,
        "ranking_protocol_version": PAIRED_OOF_PROTOCOL_VERSION,
        "status": "running",
        "stage": "calibrate-paired-oof",
        "mode": config.mode,
        "generated_at": generated_at,
        "candidate_methods": list(methods),
        "data_safety": {
            "fit_scope": "mimiciv_train_paired_patient_grouped_oof_only",
            "report_contains_patient_rows": False,
            "report_contains_row_samples": False,
            "report_contains_identifier_values": False,
            "restricted_parameters_stay_on_protected_storage": True,
        },
    }
    try:
        if config.mode != "development":
            raise ValueError("paired OOF calibration must run in development mode")
        if (
            "protected" in config.dataset_root.resolve().parts
            and config.fold_count != 5
        ):
            raise ValueError("protected paired OOF calibration requires five folds")
        selection, model_provenance = _selection_and_model_provenance(config)
        scores, oof_provenance = _load_paired_oof_scores(
            config,
            selection=selection,
        )
        labels = scores["labels"]
        patient_folds = scores["patient_fold_ids"]
        mappings = {
            baseline_name: cross_validate_calibrators(
                scores[baseline_name],
                labels,
                patient_folds,
                fold_count=config.fold_count,
                methods=methods,
            )
            for baseline_name in (
                LATE_FUSION_BASELINE_NAME,
                TRANSFORMER_BASELINE_NAME,
            )
        }
        provenance = {
            "seed": config.seed,
            "fold_count": config.fold_count,
            "contract_digest": contract_digest_or_none(config),
            "selected_gnn_variant": selection["selected_gnn_variant"],
            "selected_alpha": float(selection["selected_alpha"]),
            "alpha_semantics": selection["alpha_semantics"],
            **model_provenance,
            **oof_provenance,
        }
        artifact = {
            "schema_version": PAIRED_CALIBRATION_SCHEMA_VERSION,
            "calibration_protocol_version": PAIRED_CALIBRATION_PROTOCOL_VERSION,
            "ranking_protocol_version": PAIRED_OOF_PROTOCOL_VERSION,
            "status": "completed",
            "calibration_frozen": True,
            "generated_at": generated_at,
            "fit_scope": "mimiciv_train_paired_patient_grouped_oof_only",
            "patient_grouping_basis": (
                "the deterministic patient_fold_id locked by the paired-OOF protocol"
            ),
            "selection_policy": "minimum_cross_fitted_brier_then_log_loss",
            "candidate_row_weighting": "unweighted_micro_average",
            "ranking_contract": {
                "rank_score": (
                    "unchanged raw branch score; paired hybrid uses the frozen "
                    "within-group z-score blend"
                ),
                "ranking_metrics_use_calibrated_probability": False,
                "selected_gnn_variant": selection["selected_gnn_variant"],
                "selected_alpha": float(selection["selected_alpha"]),
                "alpha_semantics": selection["alpha_semantics"],
            },
            "probability_contract": {
                "field": "probability_observed_prescription",
                "target": (
                    "historically observed medication in the (24h, 48h] label "
                    "window under the locked candidate policy"
                ),
                "clinical_optimality_probability": False,
                "unobserved_candidates_are_strong_negatives": False,
            },
            "provenance": provenance,
            "branch_mappings": mappings,
            "data_safety": report["data_safety"],
        }
        write_json(config.paired_calibration_path, artifact)
        report.update(
            {
                "status": "completed",
                "selection_policy": artifact["selection_policy"],
                "patient_grouping_basis": artifact["patient_grouping_basis"],
                "candidate_row_weighting": artifact["candidate_row_weighting"],
                "provenance": provenance,
                "branch_results": {
                    baseline_name: _mapping_report(mapping)
                    for baseline_name, mapping in mappings.items()
                },
                "artifacts": {
                    "paired_calibration_sha256": sha256_file(
                        config.paired_calibration_path
                    )
                },
            }
        )
    except Exception as error:  # noqa: BLE001 - aggregate fail-closed report
        report["status"] = "failed"
        report["reason"] = safe_error_message(error)
    report["wall_time_seconds"] = perf_counter() - started
    write_json(config.paired_calibration_report_path, report)
    return report
