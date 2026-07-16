"""Generate aggregate-only Phase 4-9 meeting visualizations."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "researchmodule-matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

from pipeline.config import PROJECT_ROOT, REPORTS_ROOT


SCHEMA_VERSION = "phase4-to-9-visualization-v1"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "visualization"
DEFAULT_FIGURES_ROOT = DEFAULT_OUTPUT_ROOT / "figures"
DEFAULT_MARKDOWN_PATH = DEFAULT_OUTPUT_ROOT / "meeting_figure_pack.md"
DEFAULT_SUMMARY_PATH = DEFAULT_OUTPUT_ROOT / "meeting_figure_pack.json"

REPORT_FILES = {
    "harmonization_coverage": "harmonization_coverage.json",
    "milestone6_features": "milestone6_feature_manifest.json",
    "phase8_p0_features": "phase8_p0_milestone6_feature_manifest.json",
    "training_table": "training_table_manifest.json",
    "preprocessing": "preprocessing_manifest.json",
    "milestone7_coverage": "milestone7_coverage_report.json",
    "milestone7_evaluation": "milestone7_baseline_evaluation.json",
    "milestone7_frozen": "milestone7_frozen_selection.json",
    "milestone8_suitability": "milestone8_graph_suitability.json",
    "milestone8b_features": "milestone8b_graph_feature_manifest.json",
    "milestone8b_evaluation": "milestone8b_ablation_evaluation.json",
    "milestone8b_frozen": "milestone8b_frozen_selection.json",
}

SOURCE_LABELS = {
    "mimiciv": "MIMIC-IV",
    "eicu_crd": "eICU-CRD",
}
SPLIT_LABELS = {
    "train": "train",
    "validation": "validation",
    "test": "test",
    "external": "external",
}
BASELINE_ORDER = (
    "random",
    "global_popularity",
    "condition_popularity",
    "linear",
    "xgboost",
)
ABLATION_ORDER = (
    "xgboost_frozen_reference",
    "graph_only_xgboost",
    "xgboost_graph_augmented",
    "late_fusion_validation_weighted",
    "simple_ensemble_mean",
)
PALETTE = {
    "blue": "#31688E",
    "teal": "#21918C",
    "green": "#35A853",
    "orange": "#E17C05",
    "red": "#C83E4D",
    "yellow": "#F6C141",
    "gray": "#6B7280",
    "light_gray": "#D1D5DB",
    "dark": "#1F2937",
}


@dataclass(frozen=True)
class FigureRecord:
    """Metadata for one generated meeting figure."""

    title: str
    filename: str
    caption: str

    @property
    def path(self) -> str:
        return f"figures/{self.filename}"


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""

    return json.loads(path.read_text(encoding="utf-8"))


def load_reports(reports_root: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Load available aggregate reports by logical name."""

    reports: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for name, filename in REPORT_FILES.items():
        path = reports_root / filename
        if path.exists():
            reports[name] = load_json(path)
        else:
            missing.append(filename)
    return reports, missing


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write stable JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def source_split_label(row: dict[str, Any]) -> str:
    """Return a compact source/split display label."""

    source = SOURCE_LABELS.get(str(row.get("source")), str(row.get("source")))
    split = SPLIT_LABELS.get(str(row.get("split")), str(row.get("split")))
    return f"{source}\n{split}"


def pretty_token(value: str) -> str:
    """Convert snake_case report tokens into compact chart labels."""

    replacements = {
        "xgboost": "XGBoost",
        "ndcg": "NDCG",
        "mrr": "MRR",
        "eicu": "eICU",
        "mimiciv": "MIMIC-IV",
    }
    pieces = value.replace("_", " ").split()
    return " ".join(replacements.get(piece.lower(), piece.title()) for piece in pieces)


def as_number(value: Any, default: float = 0.0) -> float:
    """Return a finite numeric value for plotting."""

    if value is None:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def format_int(value: Any) -> str:
    """Format a count-like value."""

    if value is None:
        return "n/a"
    return f"{int(as_number(value)):,}"


def format_rate(value: Any) -> str:
    """Format a 0-1 rate as a percent."""

    if value is None:
        return "n/a"
    return f"{as_number(value) * 100:.1f}%"


def setup_axis(ax: plt.Axes, *, title: str | None = None) -> None:
    """Apply consistent non-interactive chart styling."""

    if title:
        ax.set_title(title, fontsize=12, weight="bold", loc="left", pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.22, linewidth=0.8)
    ax.tick_params(axis="both", labelsize=9)


def save_figure(fig: plt.Figure, path: Path) -> None:
    """Save and close a figure."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def completed(report: dict[str, Any] | None) -> bool:
    """Return whether a report records completion."""

    return bool(report and report.get("status") in {"completed", "frozen"})


def phase_statuses(reports: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    """Derive implementation status for the Phase 4-9 meeting summary."""

    training = reports.get("training_table", {})
    preprocessing = reports.get("preprocessing", {})
    graph = reports.get("milestone8_suitability", {})
    split_integrity = training.get("split_integrity", {})
    leakage_audit = graph.get("leakage_audit", {})
    graph_gate = graph.get("gate_review", {})
    phase8b_selection = reports.get("milestone8b_frozen", {})
    return [
        {
            "phase": "Phase 4",
            "label": "Feature engineering",
            "status": "completed"
            if completed(reports.get("milestone6_features"))
            else "pending",
            "note": "Temporal stay features and event sequences materialized.",
        },
        {
            "phase": "Phase 5",
            "label": "Feature relevance",
            "status": "completed"
            if completed(reports.get("milestone7_evaluation"))
            and reports.get("milestone7_frozen", {}).get("status") == "frozen"
            else "partial",
            "note": "XGBoost selected from validation baselines.",
        },
        {
            "phase": "Phase 6",
            "label": "Preprocessing",
            "status": "completed"
            if completed(reports.get("training_table")) and completed(preprocessing)
            else "partial",
            "note": "Train-fitted preprocessing and candidate table are available.",
        },
        {
            "phase": "Phase 7",
            "label": "Bias/leakage controls",
            "status": "completed"
            if split_integrity.get("patients_with_multiple_splits") == 0
            and leakage_audit.get("status") == "pass"
            else "partial",
            "note": "Patient splits, temporal cutoffs, and train-only fitting are gated.",
        },
        {
            "phase": "Phase 8",
            "label": "Graph suitability",
            "status": "completed"
            if graph_gate.get("result") == "pass_for_graph_ablation"
            and phase8b_selection.get("status") == "frozen"
            else "partial",
            "note": "Graph gate passed; graph-aware ablation is frozen.",
        },
        {
            "phase": "Phase 9",
            "label": "Grounded explanation",
            "status": "planned",
            "note": "Roadmap is documented; implementation has not started.",
        },
    ]


def write_phase_status_figure(
    reports: dict[str, dict[str, Any]], figures_root: Path
) -> FigureRecord:
    """Plot Phase 4-9 status and claim boundaries."""

    rows = phase_statuses(reports)
    color_for_status = {
        "completed": PALETTE["green"],
        "partial": PALETTE["orange"],
        "planned": PALETTE["gray"],
        "pending": PALETTE["red"],
    }
    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.8, len(rows) - 0.2)
    ax.axis("off")
    ax.set_title(
        "Phase 4-9 Current State",
        fontsize=15,
        weight="bold",
        loc="left",
        pad=14,
    )
    for index, row in enumerate(reversed(rows)):
        y = index
        color = color_for_status[row["status"]]
        ax.add_patch(Rectangle((0.02, y - 0.3), 0.16, 0.6, color=color, alpha=0.95))
        ax.text(
            0.1,
            y,
            row["phase"],
            ha="center",
            va="center",
            color="white",
            fontsize=10,
            weight="bold",
        )
        ax.text(
            0.22,
            y + 0.11,
            row["label"],
            ha="left",
            va="center",
            fontsize=11,
            weight="bold",
            color=PALETTE["dark"],
        )
        ax.text(
            0.22,
            y - 0.14,
            textwrap.fill(row["note"], width=94),
            ha="left",
            va="center",
            fontsize=9.3,
            color=PALETTE["gray"],
        )
        ax.text(
            0.93,
            y,
            row["status"].replace("_", " "),
            ha="center",
            va="center",
            fontsize=9,
            color=color,
            weight="bold",
        )
    legend = [
        Patch(color=color_for_status["completed"], label="completed/frozen"),
        Patch(color=color_for_status["partial"], label="partial"),
        Patch(color=color_for_status["planned"], label="planned"),
    ]
    ax.legend(handles=legend, loc="lower left", bbox_to_anchor=(0.02, -0.08), ncol=3)
    filename = "01_phase4_to_9_status.png"
    save_figure(fig, figures_root / filename)
    return FigureRecord(
        title="Phase 4-9 Current State",
        filename=filename,
        caption=(
            "Use this opener to anchor the meeting: Phases 4-8 have concrete "
            "aggregate artifacts and gates; Phase 9 remains planned."
        ),
    )


def write_harmonization_coverage_figure(
    reports: dict[str, dict[str, Any]], figures_root: Path
) -> FigureRecord | None:
    """Plot harmonized rows by domain and source as source-combination context."""

    coverage = reports.get("harmonization_coverage", {}).get("coverage", [])
    rows = [
        row
        for row in coverage
        if row.get("domain")
        and row.get("source") in {"mimiciv", "eicu_crd"}
        and row.get("row_count") is not None
    ]
    if not rows:
        return None
    domains = sorted({str(row["domain"]) for row in rows})
    sources = ("mimiciv", "eicu_crd")
    values = {
        (str(row["domain"]), str(row["source"])): as_number(row.get("row_count"))
        for row in rows
    }
    x_positions = range(len(domains))
    width = 0.38
    fig, ax = plt.subplots(figsize=(12, 6.2))
    for offset, source in enumerate(sources):
        positions = [x + (offset - 0.5) * width for x in x_positions]
        ax.bar(
            positions,
            [values.get((domain, source), 0) for domain in domains],
            width=width,
            label=SOURCE_LABELS[source],
            color=PALETTE["blue"] if source == "mimiciv" else PALETTE["teal"],
        )
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylabel("Harmonized rows, symlog scale")
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(
        [pretty_token(domain) for domain in domains], rotation=30, ha="right"
    )
    setup_axis(ax, title="MIMIC/eICU Harmonized Domain Coverage")
    ax.legend(frameon=False)
    filename = "02_harmonization_domain_coverage.png"
    save_figure(fig, figures_root / filename)
    return FigureRecord(
        title="Harmonized Domain Coverage",
        filename=filename,
        caption=(
            "Shows the source-domain scale behind the Phase 4-9 work. The y-axis "
            "uses a symmetric log scale because labs/vitals dwarf smaller domains."
        ),
    )


def write_feature_rows_figure(
    reports: dict[str, dict[str, Any]], figures_root: Path
) -> FigureRecord | None:
    """Plot Milestone 6 feature rows and patient split counts."""

    feature_report = reports.get("milestone6_features", {})
    rows = feature_report.get("feature_rows_by_source", [])
    split_rows = feature_report.get("split_counts", [])
    if not rows:
        return None
    labels = [source_split_label(row) for row in rows]
    row_counts = [as_number(row.get("row_count")) for row in rows]
    patient_counts = [
        as_number(
            next(
                (
                    split_row.get("patient_count")
                    for split_row in split_rows
                    if split_row.get("source") == row.get("source")
                    and split_row.get("split") == row.get("split")
                ),
                0,
            )
        )
        for row in rows
    ]
    x_positions = range(len(rows))
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    ax.bar(
        [x - 0.18 for x in x_positions],
        row_counts,
        width=0.36,
        color=PALETTE["blue"],
        label="feature rows",
    )
    ax.bar(
        [x + 0.18 for x in x_positions],
        patient_counts,
        width=0.36,
        color=PALETTE["green"],
        label="patients",
    )
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Count")
    setup_axis(ax, title="Phase 4 Feature Rows and Patient Splits")
    ax.legend(frameon=False)
    filename = "03_feature_rows_by_source_split.png"
    save_figure(fig, figures_root / filename)
    return FigureRecord(
        title="Feature Rows and Splits",
        filename=filename,
        caption=(
            "Summarizes the Milestone 6 feature surface by source and split, "
            "including eICU as the external split."
        ),
    )


def write_feature_family_figure(
    reports: dict[str, dict[str, Any]], figures_root: Path
) -> FigureRecord | None:
    """Plot optional Phase 8 P0 feature-family counts."""

    report = reports.get("phase8_p0_features") or reports.get("milestone6_features")
    counts = report.get("feature_column_counts_by_family")
    if not isinstance(counts, dict):
        preprocessing = reports.get("preprocessing", {})
        counts = {
            "stay_numeric_columns": len(preprocessing.get("stay_numeric_columns", [])),
            "stay_categorical_columns": len(
                preprocessing.get("stay_categorical_columns", [])
            ),
            "row_numeric_columns": len(preprocessing.get("row_numeric_columns", [])),
            "row_categorical_columns": len(
                preprocessing.get("row_categorical_columns", [])
            ),
        }
    rows = [
        (key, as_number(value))
        for key, value in counts.items()
        if key != "total_columns" and as_number(value) > 0
    ]
    if not rows:
        return None
    rows = sorted(rows, key=lambda item: item[1])
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(
        [pretty_token(name.replace("_columns", "")) for name, _ in rows],
        [value for _, value in rows],
        color=[
            PALETTE["teal"],
            PALETTE["blue"],
            PALETTE["green"],
            PALETTE["orange"],
            PALETTE["yellow"],
            PALETTE["red"],
        ]
        * 3,
    )
    ax.set_xlabel("Feature columns")
    setup_axis(ax, title="Feature Families Available for Model-Ready Tables")
    filename = "04_feature_family_counts.png"
    save_figure(fig, figures_root / filename)
    caption = (
        "Counts feature families from the isolated Phase 8 P0 manifest when present."
    )
    return FigureRecord("Feature Family Counts", filename, caption)


def write_training_balance_figure(
    reports: dict[str, dict[str, Any]], figures_root: Path
) -> FigureRecord | None:
    """Plot candidate rows, observed positives, and positive rate by split."""

    rows = reports.get("training_table", {}).get("training_rows_by_source_split", [])
    if not rows:
        rows = reports.get("milestone7_coverage", {}).get("source_split_coverage", [])
    if not rows:
        return None
    labels = [source_split_label(row) for row in rows]
    candidate_counts = [
        as_number(row.get("row_count") or row.get("candidate_row_count"))
        for row in rows
    ]
    positive_counts = [
        as_number(
            row.get("positive_row_count") or row.get("in_catalog_positive_row_count")
        )
        for row in rows
    ]
    positive_rates = [
        positive / total if total > 0 else 0
        for positive, total in zip(positive_counts, candidate_counts, strict=True)
    ]
    fig, axes = plt.subplots(
        1, 2, figsize=(12, 5.6), gridspec_kw={"width_ratios": [1.3, 1]}
    )
    axes[0].bar(labels, candidate_counts, color=PALETTE["blue"], label="candidate rows")
    axes[0].bar(
        labels, positive_counts, color=PALETTE["orange"], label="observed positives"
    )
    axes[0].set_yscale("symlog", linthresh=1)
    axes[0].set_ylabel("Rows, symlog scale")
    setup_axis(axes[0], title="Candidate Scale")
    axes[0].legend(frameon=False)
    axes[1].bar(labels, positive_rates, color=PALETTE["green"])
    axes[1].set_ylim(0, max(positive_rates + [0.01]) * 1.25)
    axes[1].set_ylabel("Observed positive rate")
    axes[1].yaxis.set_major_formatter(lambda value, _: f"{value * 100:.1f}%")
    setup_axis(axes[1], title="Observed-Label Balance")
    filename = "05_training_table_balance.png"
    save_figure(fig, figures_root / filename)
    return FigureRecord(
        title="Candidate Table Balance",
        filename=filename,
        caption=(
            "Separates candidate-row scale from observed-label balance. Observed "
            "positives are historical labels, not clinical optimality."
        ),
    )


def gate_rows(reports: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    """Build leakage/readiness gate rows from aggregate manifests."""

    training = reports.get("training_table", {})
    preprocessing = reports.get("preprocessing", {})
    m7 = reports.get("milestone7_frozen", {})
    graph = reports.get("milestone8_suitability", {})
    m8b = reports.get("milestone8b_frozen", {})
    split_integrity = training.get("split_integrity", {})
    fit_scope = preprocessing.get("fit_scope", {})
    leakage_audit = graph.get("leakage_audit", {})
    eicu_statuses = {
        row.get("performance_status")
        for row in reports.get("milestone7_coverage", {}).get(
            "source_split_coverage", []
        )
        if row.get("source") == "eicu_crd"
    }
    return [
        {
            "gate": "patient split integrity",
            "status": "pass"
            if split_integrity.get("patients_with_multiple_splits") == 0
            else "review",
            "note": "0 patients in multiple splits"
            if split_integrity.get("patients_with_multiple_splits") == 0
            else "review split manifest",
        },
        {
            "gate": "train-only preprocessing",
            "status": "pass"
            if fit_scope.get("source") == "mimiciv"
            and fit_scope.get("split") == "train"
            else "review",
            "note": "fit on MIMIC train rows",
        },
        {
            "gate": "train-only candidate catalog",
            "status": "pass"
            if training.get("candidate_catalog_counts", {}).get("candidate_count")
            else "review",
            "note": "catalog derived from development train positives",
        },
        {
            "gate": "Milestone 7 frozen selection",
            "status": "frozen" if m7.get("status") == "frozen" else "review",
            "note": pretty_token(
                str(m7.get("selected_headline_baseline", "not frozen"))
            ),
        },
        {
            "gate": "train-fit graph construction",
            "status": "pass"
            if leakage_audit.get("status") == "pass"
            and leakage_audit.get("train_only_graph_fit") is True
            else "review",
            "note": "graph fit from MIMIC train only",
        },
        {
            "gate": "eICU interpretation boundary",
            "status": "coverage"
            if "coverage_only_no_in_catalog_positive_groups" in eicu_statuses
            else "review",
            "note": "coverage-only, no performance claim",
        },
        {
            "gate": "Milestone 8B frozen selection",
            "status": "frozen" if m8b.get("status") == "frozen" else "review",
            "note": pretty_token(str(m8b.get("selected_experiment", "not frozen"))),
        },
        {
            "gate": "grounded explanation layer",
            "status": "pending",
            "note": "Roadmap Milestone 9 not started",
        },
    ]


def write_gate_figure(
    reports: dict[str, dict[str, Any]], figures_root: Path
) -> FigureRecord:
    """Plot leakage/readiness gates as a compact status board."""

    rows = gate_rows(reports)
    color_for_status = {
        "pass": PALETTE["green"],
        "frozen": PALETTE["teal"],
        "coverage": PALETTE["yellow"],
        "pending": PALETTE["gray"],
        "review": PALETTE["orange"],
    }
    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(rows) - 0.3)
    ax.axis("off")
    ax.set_title(
        "Phase 7 Safety, Leakage, and Readiness Gates",
        fontsize=14,
        weight="bold",
        loc="left",
    )
    for index, row in enumerate(reversed(rows)):
        y = index
        color = color_for_status[row["status"]]
        ax.add_patch(Rectangle((0.02, y - 0.24), 0.16, 0.48, color=color, alpha=0.95))
        ax.text(
            0.1,
            y,
            row["status"],
            ha="center",
            va="center",
            fontsize=9,
            color="white",
            weight="bold",
        )
        ax.text(
            0.22,
            y + 0.09,
            pretty_token(row["gate"]),
            ha="left",
            va="center",
            fontsize=10.5,
            weight="bold",
        )
        ax.text(
            0.22,
            y - 0.14,
            row["note"],
            ha="left",
            va="center",
            fontsize=9.2,
            color=PALETTE["gray"],
        )
    legend = [
        Patch(color=PALETTE["green"], label="pass"),
        Patch(color=PALETTE["teal"], label="frozen"),
        Patch(color=PALETTE["yellow"], label="coverage-only"),
        Patch(color=PALETTE["gray"], label="pending"),
    ]
    ax.legend(handles=legend, loc="lower left", bbox_to_anchor=(0.02, -0.08), ncol=4)
    filename = "06_leakage_and_readiness_gates.png"
    save_figure(fig, figures_root / filename)
    return FigureRecord(
        title="Safety and Readiness Gates",
        filename=filename,
        caption=(
            "Meeting slide for leakage controls, frozen selection gates, and what "
            "must still be treated as pending or coverage-only."
        ),
    )


def ranking_metric_rows(
    report: dict[str, Any],
    *,
    k: int = 10,
    source: str = "mimiciv",
    splits: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Return aggregate ranking metrics for plotting."""

    selected_splits = {"validation", "test"} if splits is None else splits
    return [
        row
        for row in report.get("ranking_metrics", [])
        if row.get("k") == k
        and row.get("source") == source
        and row.get("split") in selected_splits
    ]


def write_metric_bar_figure(
    reports: dict[str, dict[str, Any]],
    figures_root: Path,
    *,
    report_name: str,
    filename: str,
    title: str,
    baseline_order: Sequence[str],
    caption: str,
) -> FigureRecord | None:
    """Plot validation/test NDCG@10 by baseline-like experiment."""

    report = reports.get(report_name, {})
    rows = ranking_metric_rows(report)
    if not rows:
        return None
    present = {str(row.get("baseline_name")) for row in rows}
    baselines = [baseline for baseline in baseline_order if baseline in present]
    baselines.extend(sorted(present - set(baselines)))
    value = {
        (str(row.get("baseline_name")), str(row.get("split"))): row.get("ndcg_at_k")
        for row in rows
    }
    x_positions = list(range(len(baselines)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(12, 5.8))
    for offset, split in enumerate(("validation", "test")):
        ax.bar(
            [x + (offset - 0.5) * width for x in x_positions],
            [as_number(value.get((baseline, split))) for baseline in baselines],
            width=width,
            label=split,
            color=PALETTE["blue"] if split == "validation" else PALETTE["orange"],
        )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [pretty_token(baseline) for baseline in baselines], rotation=20, ha="right"
    )
    ax.set_ylabel("NDCG@10")
    ax.set_ylim(
        0, max([as_number(row.get("ndcg_at_k")) for row in rows] + [0.1]) * 1.22
    )
    setup_axis(ax, title=title)
    ax.legend(frameon=False)
    save_figure(fig, figures_root / filename)
    return FigureRecord(title=title, filename=filename, caption=caption)


def write_graph_structure_figure(
    reports: dict[str, dict[str, Any]], figures_root: Path
) -> FigureRecord | None:
    """Plot graph node and edge counts."""

    report = reports.get("milestone8_suitability", {})
    node_rows = report.get("node_counts", [])
    edge_rows = report.get("edge_counts_by_relation", [])
    if not node_rows and not edge_rows:
        return None
    fig, axes = plt.subplots(
        1, 2, figsize=(13, 5.8), gridspec_kw={"width_ratios": [0.85, 1.35]}
    )
    if node_rows:
        nodes = sorted(node_rows, key=lambda row: as_number(row.get("node_count")))
        axes[0].barh(
            [pretty_token(str(row.get("node_type"))) for row in nodes],
            [as_number(row.get("node_count")) for row in nodes],
            color=PALETTE["teal"],
        )
        axes[0].set_xlabel("Nodes")
        setup_axis(axes[0], title="Concept Nodes")
    if edge_rows:
        edges = sorted(edge_rows, key=lambda row: as_number(row.get("edge_count")))
        axes[1].barh(
            [pretty_token(str(row.get("relation_type"))) for row in edges],
            [as_number(row.get("edge_count")) for row in edges],
            color=PALETTE["green"],
        )
        axes[1].set_xscale("symlog", linthresh=1)
        axes[1].set_xlabel("Edges, symlog scale")
        setup_axis(axes[1], title="Train-Fit Relations")
    filename = "08_graph_structure_summary.png"
    save_figure(fig, figures_root / filename)
    gate = report.get("gate_review", {}).get("result", "unknown")
    return FigureRecord(
        title="Graph Structure Summary",
        filename=filename,
        caption=f"Milestone 8 graph gate result: `{gate}`.",
    )


def write_fusion_curve_figure(
    reports: dict[str, dict[str, Any]], figures_root: Path
) -> FigureRecord | None:
    """Plot Milestone 8B late-fusion validation curve."""

    fusion = reports.get("milestone8b_frozen", {}).get("fusion_weight", {})
    candidates = fusion.get("candidates", [])
    if not candidates:
        candidates = (
            reports.get("milestone8b_evaluation", {})
            .get("fusion", {})
            .get("candidates", [])
        )
    if not candidates:
        return None
    weights = [as_number(row.get("graph_weight")) for row in candidates]
    ndcg = [as_number(row.get("ndcg_at_k")) for row in candidates]
    mrr = [as_number(row.get("mrr_at_k")) for row in candidates]
    selected = as_number(fusion.get("selected_graph_weight"))
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.plot(weights, ndcg, marker="o", color=PALETTE["blue"], label="NDCG@10")
    ax.plot(weights, mrr, marker="s", color=PALETTE["orange"], label="MRR@10")
    ax.axvline(
        selected,
        color=PALETTE["green"],
        linestyle="--",
        linewidth=1.8,
        label=f"selected weight {selected:g}",
    )
    ax.set_xlabel("Graph score weight")
    ax.set_ylabel("Validation metric")
    ax.set_ylim(0, max(ndcg + mrr + [0.1]) * 1.12)
    setup_axis(ax, title="Milestone 8B Fusion Sweep")
    ax.legend(frameon=False)
    filename = "10_milestone8b_fusion_weight_curve.png"
    save_figure(fig, figures_root / filename)
    return FigureRecord(
        title="Fusion Weight Sweep",
        filename=filename,
        caption=(
            "Validation sweep for late fusion. In the current run the selected "
            "late-fusion graph weight is 0, so graph-aware gain comes from the "
            "graph-augmented XGBoost comparison rather than late fusion."
        ),
    )


def write_markdown_pack(
    figures: Sequence[FigureRecord],
    summary: dict[str, Any],
    output_path: Path,
) -> None:
    """Write a compact meeting figure-pack markdown file."""

    lines = [
        "# Phase 4-9 Meeting Figure Pack",
        "",
        "Aggregate-only visual pack generated from local `reports/*.json` manifests.",
        "",
        "## Data Safety",
        "",
        "- Contains no patient-level rows, note text, raw identifiers, or clinical recommendations.",
        "- Observed prescriptions are historical labels, not proof of optimal treatment.",
        "- eICU remains coverage-only when in-catalog positive groups are absent.",
        "",
        "## Slide Order",
        "",
    ]
    for index, figure in enumerate(figures, start=1):
        lines.extend(
            [
                f"### {index}. {figure.title}",
                "",
                f"![{figure.title}]({figure.path})",
                "",
                figure.caption,
                "",
            ]
        )
    if summary.get("missing_reports"):
        lines.extend(
            [
                "## Missing Inputs",
                "",
                "The generator skipped any figures requiring these missing aggregate reports:",
                "",
            ]
        )
        lines.extend(f"- `{name}`" for name in summary["missing_reports"])
        lines.append("")
    lines.extend(
        [
            "## Suggested Close",
            "",
            "- Keep claims bounded to data-foundation, baseline, and graph-ablation evidence.",
            "- Review out-of-catalog positives and eICU coverage before external-performance claims.",
            "- Start Roadmap Milestone 9 only after explanation evidence sources and clinical-rule boundaries are reviewed.",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_summary(
    reports: dict[str, dict[str, Any]],
    missing_reports: list[str],
    figures: Sequence[FigureRecord],
) -> dict[str, Any]:
    """Build a JSON summary for the generated visual pack."""

    m7_frozen = reports.get("milestone7_frozen", {})
    m8 = reports.get("milestone8_suitability", {})
    m8b = reports.get("milestone8b_frozen", {})
    training = reports.get("training_table", {})
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "data_safety": {
            "contains_patient_rows": False,
            "contains_raw_note_text": False,
            "source": "aggregate JSON reports only",
        },
        "phase_status": phase_statuses(reports),
        "headline_facts": {
            "candidate_count": training.get("candidate_catalog_counts", {}).get(
                "candidate_count"
            ),
            "condition_count": training.get("candidate_catalog_counts", {}).get(
                "condition_count"
            ),
            "milestone7_selected_baseline": m7_frozen.get("selected_headline_baseline"),
            "milestone8_graph_gate": m8.get("gate_review", {}).get("result"),
            "milestone8b_selected_experiment": m8b.get("selected_experiment"),
            "milestone9_status": "planned_not_started",
        },
        "missing_reports": missing_reports,
        "figures": [
            figure.__dict__ | {"relative_path": figure.path} for figure in figures
        ],
    }


def generate_visualizations(
    *,
    reports_root: Path = REPORTS_ROOT,
    figures_root: Path = DEFAULT_FIGURES_ROOT,
    markdown_path: Path = DEFAULT_MARKDOWN_PATH,
    summary_path: Path = DEFAULT_SUMMARY_PATH,
) -> dict[str, Any]:
    """Generate Phase 4-9 aggregate figures and meeting pack files."""

    reports, missing_reports = load_reports(reports_root)
    figure_builders = (
        write_phase_status_figure,
        write_harmonization_coverage_figure,
        write_feature_rows_figure,
        write_feature_family_figure,
        write_training_balance_figure,
        write_gate_figure,
        lambda loaded, root: write_metric_bar_figure(
            loaded,
            root,
            report_name="milestone7_evaluation",
            filename="07_milestone7_baseline_ndcg_at_10.png",
            title="Milestone 7 Baseline NDCG@10",
            baseline_order=BASELINE_ORDER,
            caption=(
                "Validation drove model selection; held-out MIMIC test metrics "
                "are shown only after the frozen-selection gate."
            ),
        ),
        write_graph_structure_figure,
        lambda loaded, root: write_metric_bar_figure(
            loaded,
            root,
            report_name="milestone8b_evaluation",
            filename="09_milestone8b_ablation_ndcg_at_10.png",
            title="Milestone 8B Graph-Aware Ablation NDCG@10",
            baseline_order=ABLATION_ORDER,
            caption=(
                "Compares graph-only, graph-augmented, fusion, and ensemble "
                "ablations against the frozen XGBoost reference."
            ),
        ),
        write_fusion_curve_figure,
    )
    figures: list[FigureRecord] = []
    for builder in figure_builders:
        record = builder(reports, figures_root)
        if record is not None:
            figures.append(record)

    summary = build_summary(reports, missing_reports, figures)
    write_json(summary_path, summary)
    write_markdown_pack(figures, summary, markdown_path)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Generate aggregate-only Phase 4-9 meeting visualizations."
    )
    parser.add_argument("--reports-root", type=Path, default=REPORTS_ROOT)
    parser.add_argument("--figures-root", type=Path, default=DEFAULT_FIGURES_ROOT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN_PATH)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    args = parse_args(argv)
    summary = generate_visualizations(
        reports_root=args.reports_root,
        figures_root=args.figures_root,
        markdown_path=args.markdown,
        summary_path=args.summary,
    )
    print(
        "Wrote Phase 4-9 visualization pack: "
        f"{len(summary['figures'])} figures, "
        f"{len(summary['missing_reports'])} missing aggregate inputs"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
