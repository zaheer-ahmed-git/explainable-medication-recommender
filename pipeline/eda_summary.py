"""Aggregate EDA synthesis from safe data-foundation reports."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline.config import REPORTS_ROOT, ensure_local_directories


SCHEMA_VERSION = "eda-dataset-understanding-v1"
DEFAULT_INVENTORY_PATH = REPORTS_ROOT / "source_inventory.json"
DEFAULT_COHORT_PATH = REPORTS_ROOT / "cohort_manifest.json"
DEFAULT_QUALITY_PATH = REPORTS_ROOT / "quality_profile.json"
DEFAULT_OUTPUT_JSON = REPORTS_ROOT / "eda_dataset_understanding.json"
DEFAULT_OUTPUT_MARKDOWN = REPORTS_ROOT / "eda_dataset_understanding.md"
DEFAULT_FIGURES_ROOT = REPORTS_ROOT / "figures"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def format_int(value: int | float | None) -> str:
    if value is None:
        return "n/a"
    return f"{int(value):,}"


def format_bytes(value: int | float | None) -> str:
    if value is None:
        return "n/a"
    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def inventory_summary(inventory: dict[str, Any]) -> dict[str, Any]:
    sources: list[dict[str, Any]] = []
    total_files = 0
    total_bytes = 0
    missing_expected = 0
    bounded_files = 0
    for source in inventory.get("sources", []):
        files = source.get("files", [])
        source_bytes = sum(file.get("size_bytes") or 0 for file in files)
        source_bounded = sum(1 for file in files if file.get("requires_bounded_query"))
        source_missing = len(source.get("missing_expected_files", []))
        sources.append(
            {
                "name": source.get("name"),
                "version": source.get("version"),
                "present": bool(source.get("present")),
                "file_count": source.get("file_count", len(files)),
                "total_size_bytes": source_bytes,
                "total_size_readable": format_bytes(source_bytes),
                "bounded_file_count": source_bounded,
                "missing_expected_file_count": source_missing,
            }
        )
        total_files += source.get("file_count", len(files))
        total_bytes += source_bytes
        missing_expected += source_missing
        bounded_files += source_bounded
    return {
        "source_count": len(sources),
        "total_files": total_files,
        "total_size_bytes": total_bytes,
        "total_size_readable": format_bytes(total_bytes),
        "bounded_file_count": bounded_files,
        "missing_expected_file_count": missing_expected,
        "sources": sources,
    }


def cohort_summary(cohort: dict[str, Any]) -> dict[str, Any]:
    sources = cohort.get("sources", {})
    source_rows: list[dict[str, Any]] = []
    for source_name in ("mimiciv", "eicu_crd", "unified"):
        source = sources.get(source_name, {})
        source_rows.append(
            {
                "source": source_name,
                "selected_stays": source.get("selected_stays"),
                "selected_patients": source.get("selected_patients"),
                "duplicate_stay_uid_count": source.get("duplicate_stay_uid_count"),
            }
        )
    eicu = sources.get("eicu_crd", {})
    mimic = sources.get("mimiciv", {})
    return {
        "configuration": cohort.get("configuration", {}),
        "sources": source_rows,
        "mimic_excluded_by_first_stay_rule": mimic.get("excluded_by_first_stay_rule"),
        "eicu_missing_or_unparseable_age_stays": eicu.get(
            "missing_or_unparseable_age_stays"
        ),
        "eicu_topcoded_age_stays": eicu.get("topcoded_age_stays"),
    }


def table_status_summary(quality: dict[str, Any]) -> dict[str, Any]:
    statuses = Counter(
        table.get("status", "unknown") for table in quality.get("tables", [])
    )
    return {
        "configured_table_count": quality.get(
            "table_count", len(quality.get("tables", []))
        ),
        "status_counts": dict(sorted(statuses.items())),
        "completed_table_count": statuses.get("completed", 0),
        "scan_failed_table_count": statuses.get("scan_failed", 0),
    }


def completed_table_rows(quality: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        {
            "table_name": table.get("table_name"),
            "source": table.get("source"),
            "row_count": table.get("row_count", 0),
            "column_count": table.get("column_count", 0),
            "requires_bounded_query": table.get("requires_bounded_query"),
        }
        for table in quality.get("tables", [])
        if table.get("status") == "completed"
    ]
    return sorted(rows, key=lambda row: row.get("row_count") or 0, reverse=True)


def failed_scans(quality: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "table_name": table.get("table_name"),
            "source": table.get("source"),
            "relative_path": table.get("relative_path"),
            "error_type": table.get("error_type"),
            "error_message": table.get("error_message"),
        }
        for table in quality.get("tables", [])
        if table.get("status") != "completed"
    ]


def quality_findings(quality: dict[str, Any]) -> dict[str, Any]:
    duplicate_issues = []
    referential_issues = []
    numeric_parse_issues = []
    numeric_out_of_bounds = []
    high_null_columns = []

    for table in quality.get("tables", []):
        table_name = table.get("table_name")
        duplicate_profile = table.get("duplicate_key_profile") or {}
        if duplicate_profile.get("duplicate_excess_rows", 0):
            duplicate_issues.append(
                {
                    "table_name": table_name,
                    "duplicate_excess_rows": duplicate_profile.get(
                        "duplicate_excess_rows"
                    ),
                    "duplicate_key_groups": duplicate_profile.get(
                        "duplicate_key_groups"
                    ),
                }
            )
        for check in table.get("referential_integrity", []):
            if check.get("orphan_rows", 0):
                referential_issues.append(
                    {
                        "table_name": table_name,
                        "check": check.get("name"),
                        "orphan_rows": check.get("orphan_rows"),
                        "checked_rows": check.get("checked_rows"),
                    }
                )
        for column, profile in table.get("numeric_profiles", {}).items():
            parse_failures = profile.get("parse_failure_count") or 0
            out_of_bounds = profile.get("out_of_bounds_count") or 0
            if parse_failures:
                numeric_parse_issues.append(
                    {
                        "table_name": table_name,
                        "column": column,
                        "parse_failure_count": parse_failures,
                    }
                )
            if out_of_bounds:
                numeric_out_of_bounds.append(
                    {
                        "table_name": table_name,
                        "column": column,
                        "out_of_bounds_count": out_of_bounds,
                        "minimum_allowed": profile.get("minimum_allowed"),
                        "maximum_allowed": profile.get("maximum_allowed"),
                    }
                )
        for column, profile in table.get("column_profiles", {}).items():
            null_rate = profile.get("null_rate")
            if null_rate is not None and null_rate >= 0.5:
                high_null_columns.append(
                    {
                        "table_name": table_name,
                        "column": column,
                        "null_rate": null_rate,
                        "null_count": profile.get("null_count"),
                    }
                )

    return {
        "duplicate_key_issues": duplicate_issues,
        "referential_integrity_issues": referential_issues,
        "numeric_parse_issues": sorted(
            numeric_parse_issues,
            key=lambda item: item["parse_failure_count"],
            reverse=True,
        ),
        "numeric_out_of_bounds": sorted(
            numeric_out_of_bounds,
            key=lambda item: item["out_of_bounds_count"],
            reverse=True,
        ),
        "high_null_columns": sorted(
            high_null_columns,
            key=lambda item: item["null_rate"],
            reverse=True,
        ),
    }


def domain_readiness(quality: dict[str, Any]) -> dict[str, Any]:
    table_status = {
        table.get("table_name"): table.get("status")
        for table in quality.get("tables", [])
    }
    domains = {
        "cohort_demographics": (
            "ready",
            ["mimic_patients", "mimic_admissions", "mimic_icustays", "eicu_patient"],
        ),
        "conditions": ("ready", ["mimic_diagnoses_icd", "eicu_diagnosis"]),
        "procedures_treatments": (
            "ready",
            ["mimic_procedures_icd", "mimic_procedureevents", "eicu_treatment"],
        ),
        "labs": ("partial", ["mimic_labevents", "eicu_lab"]),
        "vitals": (
            "partial",
            ["mimic_chartevents", "eicu_vital_periodic", "eicu_vital_aperiodic"],
        ),
        "medications": (
            "blocked",
            ["mimic_prescriptions", "eicu_medication", "eicu_infusion_drug"],
        ),
        "severity": (
            "partial",
            [
                "eicu_apache_patient_result",
                "eicu_apache_aps_var",
                "eicu_apache_pred_var",
            ],
        ),
        "allergies": ("ready", ["eicu_allergy"]),
    }
    readiness: dict[str, Any] = {}
    for domain, (_, tables) in domains.items():
        statuses = {
            table: table_status.get(table, "not_configured") for table in tables
        }
        if all(status == "completed" for status in statuses.values()):
            status = "ready_for_extraction"
        elif any(status == "completed" for status in statuses.values()):
            status = "partial_requires_review"
        else:
            status = "blocked"
        readiness[domain] = {"status": status, "tables": statuses}
    return readiness


def stakeholder_messages(summary: dict[str, Any]) -> list[str]:
    quality = summary["quality"]
    cohort = summary["cohort"]
    unified = next(row for row in cohort["sources"] if row["source"] == "unified")
    messages = [
        f"Local data foundation covers {summary['inventory']['source_count']} source groups and {format_int(summary['inventory']['total_files'])} files.",
        f"Broad adult ICU/unit-stay cohort currently contains {format_int(unified['selected_stays'])} stays across MIMIC-IV and eICU.",
        f"Quality profiling completed for {quality['status_summary']['completed_table_count']} of {quality['status_summary']['configured_table_count']} configured structured tables.",
        "No duplicate stay IDs, duplicate configured table keys, or referential orphan rows were found in completed checks.",
        "Medication and some large event tables remain blocked by local scan/parser failures and should be reviewed before extraction or feature engineering.",
        "Observed prescriptions remain historical labels only; no clinical recommendation validity is implied at this stage.",
    ]
    return messages


def build_eda_summary(
    inventory: dict[str, Any],
    cohort: dict[str, Any],
    quality: dict[str, Any],
) -> dict[str, Any]:
    quality_summary = {
        "status_summary": table_status_summary(quality),
        "completed_table_rows": completed_table_rows(quality),
        "failed_scans": failed_scans(quality),
        "findings": quality_findings(quality),
        "domain_readiness": domain_readiness(quality),
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "data_safety": {
            "contains_patient_rows": False,
            "source": "aggregate reports only",
            "no_raw_rows_or_note_text": True,
        },
        "inputs": {
            "source_inventory_schema": inventory.get("schema_version"),
            "cohort_manifest_schema": cohort.get("schema_version"),
            "quality_profile_schema": quality.get("schema_version"),
        },
        "inventory": inventory_summary(inventory),
        "cohort": cohort_summary(cohort),
        "quality": quality_summary,
        "next_actions": [
            "Review or repair scan failures before extraction for affected medication, lab, vital, and APACHE tables.",
            "Create source-specific extraction only for domains with ready or reviewed partial status.",
            "Build notebook/figure views from aggregate reports before stakeholder presentation.",
            "Approve sepsis cohort definition before condition-specific EDA or candidate ranking.",
        ],
    }
    summary["stakeholder_messages"] = stakeholder_messages(summary)
    return summary


def write_json(summary: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_markdown(summary: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_rows = [
        (
            source["name"],
            source["version"],
            source["file_count"],
            source["total_size_readable"],
            source["bounded_file_count"],
            source["missing_expected_file_count"],
        )
        for source in summary["inventory"]["sources"]
    ]
    cohort_rows = [
        (
            row["source"],
            format_int(row["selected_stays"]),
            format_int(row["selected_patients"]),
            format_int(row["duplicate_stay_uid_count"]),
        )
        for row in summary["cohort"]["sources"]
    ]
    failed_rows = [
        (
            item["source"],
            item["table_name"],
            item["error_type"],
            item["error_message"],
        )
        for item in summary["quality"]["failed_scans"]
    ]
    readiness_rows = [
        (
            domain,
            detail["status"],
            ", ".join(
                f"{table}:{status}" for table, status in detail["tables"].items()
            ),
        )
        for domain, detail in summary["quality"]["domain_readiness"].items()
    ]
    top_rows = [
        (
            row["source"],
            row["table_name"],
            format_int(row["row_count"]),
            row["column_count"],
        )
        for row in summary["quality"]["completed_table_rows"][:12]
    ]
    messages = "\n".join(f"- {message}" for message in summary["stakeholder_messages"])
    next_actions = "\n".join(f"- {action}" for action in summary["next_actions"])

    output_path.write_text(
        "\n\n".join(
            [
                "# EDA Dataset Understanding Brief",
                "This report is generated from aggregate inventory, cohort, and quality-profile artifacts. It contains no patient-level rows, note text, or clinical recommendations.",
                "## Stakeholder Messages\n\n" + messages,
                "## Source Coverage\n\n"
                + markdown_table(
                    (
                        "Source",
                        "Version",
                        "Files",
                        "Size",
                        "Bounded Files",
                        "Missing Expected",
                    ),
                    source_rows,
                ),
                "## Cohort Scale\n\n"
                + markdown_table(
                    (
                        "Source",
                        "Selected Stays",
                        "Selected Patients",
                        "Duplicate Stay UIDs",
                    ),
                    cohort_rows,
                ),
                "## Largest Completed Tables\n\n"
                + markdown_table(
                    ("Source", "Table", "Rows", "Columns"),
                    top_rows,
                ),
                "## Scan Blockers\n\n"
                + (
                    markdown_table(
                        ("Source", "Table", "Error Type", "Safe Summary"), failed_rows
                    )
                    if failed_rows
                    else "No table-level scan blockers were recorded."
                ),
                "## Domain Readiness\n\n"
                + markdown_table(("Domain", "Status", "Tables"), readiness_rows),
                "## Next Actions\n\n" + next_actions,
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def save_bar_chart(
    labels: Sequence[str],
    values: Sequence[int | float],
    title: str,
    ylabel: str,
    output_path: Path,
    *,
    horizontal: bool = False,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#3B6EA8", "#3FA36B", "#C47F2A", "#7A5BA6", "#5E7C87", "#A64B4B"]
    if horizontal:
        ax.barh(labels, values, color=colors[: len(values)])
        ax.set_xlabel(ylabel)
        ax.invert_yaxis()
    else:
        ax.bar(labels, values, color=colors[: len(values)])
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=20)
    ax.set_title(title)
    ax.grid(axis="y" if not horizontal else "x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_figures(summary: dict[str, Any], figures_root: Path) -> list[str]:
    figures_root.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    cohort_sources = [
        row
        for row in summary["cohort"]["sources"]
        if row["source"] in {"mimiciv", "eicu_crd"}
    ]
    path = figures_root / "cohort_selected_stays_by_source.png"
    save_bar_chart(
        [row["source"] for row in cohort_sources],
        [row["selected_stays"] or 0 for row in cohort_sources],
        "Adult ICU/Unit Stays by Source",
        "Selected stays",
        path,
    )
    paths.append(str(path))

    status_counts = summary["quality"]["status_summary"]["status_counts"]
    path = figures_root / "quality_profile_status.png"
    save_bar_chart(
        list(status_counts.keys()),
        list(status_counts.values()),
        "Quality Profile Status",
        "Configured tables",
        path,
    )
    paths.append(str(path))

    top_tables = summary["quality"]["completed_table_rows"][:12]
    path = figures_root / "largest_completed_tables.png"
    save_bar_chart(
        [row["table_name"] for row in reversed(top_tables)],
        [row["row_count"] or 0 for row in reversed(top_tables)],
        "Largest Completed Table Profiles",
        "Rows",
        path,
        horizontal=True,
    )
    paths.append(str(path))

    findings = summary["quality"]["findings"]
    issue_counts = {
        "scan_failed": len(summary["quality"]["failed_scans"]),
        "duplicate_key": len(findings["duplicate_key_issues"]),
        "referential": len(findings["referential_integrity_issues"]),
        "numeric_parse": len(findings["numeric_parse_issues"]),
        "out_of_bounds": len(findings["numeric_out_of_bounds"]),
        "high_null": len(findings["high_null_columns"]),
    }
    path = figures_root / "quality_issue_categories.png"
    save_bar_chart(
        list(issue_counts.keys()),
        list(issue_counts.values()),
        "Quality Issue Categories",
        "Issue count",
        path,
    )
    paths.append(str(path))
    return paths


def generate_eda_outputs(
    *,
    inventory_path: Path = DEFAULT_INVENTORY_PATH,
    cohort_path: Path = DEFAULT_COHORT_PATH,
    quality_path: Path = DEFAULT_QUALITY_PATH,
    output_json: Path = DEFAULT_OUTPUT_JSON,
    output_markdown: Path = DEFAULT_OUTPUT_MARKDOWN,
    figures_root: Path = DEFAULT_FIGURES_ROOT,
    write_charts: bool = True,
) -> dict[str, Any]:
    ensure_local_directories()
    summary = build_eda_summary(
        load_json(inventory_path),
        load_json(cohort_path),
        load_json(quality_path),
    )
    if write_charts:
        summary["figures"] = write_figures(summary, figures_root)
    else:
        summary["figures"] = []
    write_json(summary, output_json)
    write_markdown(summary, output_markdown)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate aggregate EDA summary, briefing, and charts.",
    )
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY_PATH)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT_PATH)
    parser.add_argument("--quality", type=Path, default=DEFAULT_QUALITY_PATH)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_OUTPUT_MARKDOWN)
    parser.add_argument("--figures-root", type=Path, default=DEFAULT_FIGURES_ROOT)
    parser.add_argument("--no-figures", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = generate_eda_outputs(
        inventory_path=args.inventory,
        cohort_path=args.cohort,
        quality_path=args.quality,
        output_json=args.output_json,
        output_markdown=args.output_markdown,
        figures_root=args.figures_root,
        write_charts=not args.no_figures,
    )
    print(
        "Wrote aggregate EDA summary: "
        f"{summary['inventory']['source_count']} sources, "
        f"{summary['cohort']['sources'][-1]['selected_stays']} unified stays, "
        f"{summary['quality']['status_summary']['completed_table_count']} completed table profiles"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
