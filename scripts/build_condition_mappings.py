"""Bootstrap aggregate condition mapping templates for harmonization.

Unlike ``build_medication_mappings.py`` (a hard gate), condition semantic
mapping is optional. This script does not fabricate ICD/CCS/CCSR/SNOMED or
project-condition mappings. It only inventories the distinct diagnosis concepts
present in the cohort-filtered extracts and writes empty, review-ready template
CSVs plus an aggregate build report under ``$DATASET_ROOT/mappings/conditions``.

Curators then either drop authoritative CCS/CCSR/GEM/chapter reference files into
that folder (see ``fetch_condition_reference_files.py``) or fill the templates by
hand before re-running ``pipeline.harmonize``. Every output is aggregate-only:
distinct concept keys and row counts, never patient-level rows or identifiers.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import duckdb

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.config import (  # noqa: E402
    CONDITION_MAPPING_VERSION,
    DATASET_ROOT,
    EXTRACTS_ROOT,
    MAPPING_ROOT,
    REPORTS_ROOT,
)

SCHEMA_VERSION = "condition-mapping-build-report-v1"

# Normalization mirrors pipeline.harmonize so template keys match harmonization
# join keys exactly.
_CODE_KEY = (
    "NULLIF(LOWER(REGEXP_REPLACE(TRIM(CAST({expr} AS VARCHAR)), "
    "'[^A-Za-z0-9]+', '', 'g')), '')"
)
_TEXT_KEY = (
    "NULLIF(REGEXP_REPLACE(LOWER(TRIM(CAST({expr} AS VARCHAR))), "
    "'[^a-z0-9]+', '_', 'g'), '')"
)


def code_key(expr: str) -> str:
    return _CODE_KEY.format(expr=expr)


def text_key(expr: str) -> str:
    return _TEXT_KEY.format(expr=expr)


def sql_string(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


@dataclass(frozen=True)
class ConditionMappingBuildConfig:
    """Input and output locations for condition mapping template construction."""

    dataset_root: Path = DATASET_ROOT
    extracts_root: Path = EXTRACTS_ROOT
    mapping_root: Path = MAPPING_ROOT
    report_path: Path = REPORTS_ROOT / "condition_mapping_build_report.json"
    mimic_diagnoses_path: Path | None = None
    eicu_diagnosis_path: Path | None = None

    @property
    def condition_mapping_root(self) -> Path:
        return self.mapping_root / "conditions"

    @property
    def resolved_mimic_diagnoses_path(self) -> Path:
        return (
            self.mimic_diagnoses_path
            or self.extracts_root / "mimiciv" / "diagnoses_icd.parquet"
        )

    @property
    def resolved_eicu_diagnosis_path(self) -> Path:
        return (
            self.eicu_diagnosis_path
            or self.extracts_root / "eicu_crd" / "diagnosis.parquet"
        )


def write_csv(
    path: Path, columns: Sequence[str], rows: Sequence[Sequence[Any]]
) -> None:
    """Write a small aggregate template CSV without pandas."""

    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow(["" if value is None else value for value in row])


def _fetch(connection: duckdb.DuckDBPyConnection, query: str) -> list[tuple[Any, ...]]:
    return connection.execute(query).fetchall()


def build_mimic_templates(
    config: ConditionMappingBuildConfig,
    connection: duckdb.DuckDBPyConnection,
) -> dict[str, Any]:
    """Inventory distinct MIMIC ICD codes for review-ready mapping templates."""

    path = config.resolved_mimic_diagnoses_path
    if not path.exists():
        write_csv(
            config.condition_mapping_root / "mimic_distinct_icd_codes_for_mapping.csv",
            ("icd_version", "icd_code", "icd_code_key", "row_count"),
            (),
        )
        return {"status": "missing_extract", "distinct_icd_code_count": 0}

    rows = _fetch(
        connection,
        f"""
        SELECT
            CAST(icd_version AS VARCHAR) AS icd_version,
            CAST(icd_code AS VARCHAR) AS icd_code,
            {code_key("icd_code")} AS icd_code_key,
            COUNT(*) AS row_count
        FROM read_parquet({sql_string(path)})
        WHERE {code_key("icd_code")} IS NOT NULL
        GROUP BY icd_version, icd_code, {code_key("icd_code")}
        ORDER BY row_count DESC, icd_version, icd_code
        """,
    )
    write_csv(
        config.condition_mapping_root / "mimic_distinct_icd_codes_for_mapping.csv",
        ("icd_version", "icd_code", "icd_code_key", "row_count"),
        rows,
    )
    return {"status": "inventoried", "distinct_icd_code_count": len(rows)}


def build_eicu_templates(
    config: ConditionMappingBuildConfig,
    connection: duckdb.DuckDBPyConnection,
) -> dict[str, Any]:
    """Inventory distinct eICU codes and diagnosis strings for review templates."""

    path = config.resolved_eicu_diagnosis_path
    if not path.exists():
        for name, columns in (
            (
                "eicu_distinct_icd9_codes_for_mapping.csv",
                ("icd9code_first", "icd_code_key", "row_count"),
            ),
            (
                "eicu_distinct_diagnosis_text_for_mapping.csv",
                ("diagnosisstring_normalized", "row_count"),
            ),
            (
                "eicu_diagnosis_text_condition_map_template.csv",
                (
                    "diagnosisstring_normalized",
                    "condition_rollup_token",
                    "condition_name",
                    "row_count",
                ),
            ),
        ):
            write_csv(config.condition_mapping_root / name, columns, ())
        return {
            "status": "missing_extract",
            "distinct_icd9_code_count": 0,
            "distinct_diagnosis_text_count": 0,
        }

    first_code = "SPLIT_PART(CAST(icd9code AS VARCHAR), ',', 1)"
    icd_rows = _fetch(
        connection,
        f"""
        SELECT
            TRIM({first_code}) AS icd9code_first,
            {code_key(first_code)} AS icd_code_key,
            COUNT(*) AS row_count
        FROM read_parquet({sql_string(path)})
        WHERE {code_key(first_code)} IS NOT NULL
        GROUP BY TRIM({first_code}), {code_key(first_code)}
        ORDER BY row_count DESC, icd9code_first
        """,
    )
    write_csv(
        config.condition_mapping_root / "eicu_distinct_icd9_codes_for_mapping.csv",
        ("icd9code_first", "icd_code_key", "row_count"),
        icd_rows,
    )

    text_rows = _fetch(
        connection,
        f"""
        SELECT
            {text_key("diagnosisstring")} AS diagnosisstring_normalized,
            COUNT(*) AS row_count
        FROM read_parquet({sql_string(path)})
        WHERE {text_key("diagnosisstring")} IS NOT NULL
        GROUP BY {text_key("diagnosisstring")}
        ORDER BY row_count DESC, diagnosisstring_normalized
        """,
    )
    write_csv(
        config.condition_mapping_root / "eicu_distinct_diagnosis_text_for_mapping.csv",
        ("diagnosisstring_normalized", "row_count"),
        text_rows,
    )
    # Review-ready template: normalized string + blank curator columns + count.
    write_csv(
        config.condition_mapping_root
        / "eicu_diagnosis_text_condition_map_template.csv",
        (
            "diagnosisstring_normalized",
            "condition_rollup_token",
            "condition_name",
            "row_count",
        ),
        [(row[0], "", "", row[1]) for row in text_rows],
    )
    return {
        "status": "inventoried",
        "distinct_icd9_code_count": len(icd_rows),
        "distinct_diagnosis_text_count": len(text_rows),
    }


def write_project_group_template(config: ConditionMappingBuildConfig) -> Path:
    """Write a sepsis-first project-group template (illustrative, not truth).

    Rows are TEMPLATES that require clinical review before use. They are written
    only to the ``*_template.csv`` file, never to the active
    ``project_condition_groups.csv`` consumed by harmonization.
    """

    path = config.condition_mapping_root / "project_condition_groups_template.csv"
    write_csv(
        path,
        (
            "match_type",
            "match_value",
            "project_condition_group",
            "project_condition_token",
            "note",
        ),
        (
            (
                "icd_code",
                "A41.9",
                "sepsis",
                "condition:sepsis",
                "template_requires_clinical_review",
            ),
            (
                "text_token",
                "sepsis",
                "sepsis",
                "condition:sepsis",
                "template_requires_clinical_review",
            ),
            (
                "text_token",
                "septic_shock",
                "sepsis",
                "condition:sepsis",
                "template_requires_clinical_review",
            ),
        ),
    )
    return path


def build_condition_mappings(
    config: ConditionMappingBuildConfig = ConditionMappingBuildConfig(),
) -> dict[str, Any]:
    """Write aggregate condition mapping templates and a build report."""

    config.condition_mapping_root.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(database=":memory:") as connection:
        connection.execute("PRAGMA enable_progress_bar=false")
        mimic = build_mimic_templates(config, connection)
        eicu = build_eicu_templates(config, connection)
    project_template_path = write_project_group_template(config)

    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "condition_mapping_version": CONDITION_MAPPING_VERSION,
        "data_safety": {
            "contains_patient_rows": False,
            "reporting_level": "distinct diagnosis concepts and aggregate counts",
            "no_source_value_samples": True,
        },
        "configuration": {
            "dataset_root": str(config.dataset_root),
            "extracts_root": str(config.extracts_root),
            "mapping_root": str(config.mapping_root),
            "mimic_diagnoses_path": str(config.resolved_mimic_diagnoses_path),
            "eicu_diagnosis_path": str(config.resolved_eicu_diagnosis_path),
        },
        "mimic": mimic,
        "eicu": eicu,
        "templates_written": [
            "mimic_distinct_icd_codes_for_mapping.csv",
            "eicu_distinct_icd9_codes_for_mapping.csv",
            "eicu_distinct_diagnosis_text_for_mapping.csv",
            "eicu_diagnosis_text_condition_map_template.csv",
            project_template_path.name,
        ],
        "authoritative_reference_files_expected": [
            "icd10_ccsr.csv",
            "icd9_ccs.csv",
            "icd9_to_icd10_gem.csv",
            "icd_chapters.csv",
            "eicu_diagnosis_text_condition_map.csv",
            "project_condition_groups.csv",
        ],
        "notes": [
            "This script fabricates no clinical mappings; it inventories concepts.",
            "Project-group template rows require clinical review before use.",
            "Drop authoritative CCS/CCSR/GEM/chapter files into mappings/conditions/.",
        ],
    }
    config.report_path.parent.mkdir(parents=True, exist_ok=True)
    config.report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logging.info(
        "Condition mapping templates: MIMIC=%s codes, eICU=%s codes / %s text concepts",
        mimic.get("distinct_icd_code_count", 0),
        eicu.get("distinct_icd9_code_count", 0),
        eicu.get("distinct_diagnosis_text_count", 0),
    )
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build aggregate condition mapping templates for harmonization.",
    )
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--extracts-root", type=Path, default=EXTRACTS_ROOT)
    parser.add_argument("--mapping-root", type=Path, default=MAPPING_ROOT)
    parser.add_argument(
        "--report",
        type=Path,
        default=REPORTS_ROOT / "condition_mapping_build_report.json",
    )
    parser.add_argument("--mimic-diagnoses", type=Path, default=None)
    parser.add_argument("--eicu-diagnosis", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args(argv)
    report = build_condition_mappings(
        ConditionMappingBuildConfig(
            dataset_root=args.dataset_root,
            extracts_root=args.extracts_root,
            mapping_root=args.mapping_root,
            report_path=args.report,
            mimic_diagnoses_path=args.mimic_diagnoses,
            eicu_diagnosis_path=args.eicu_diagnosis,
        )
    )
    print(
        "Wrote condition mapping templates: "
        f"mimic_codes={report['mimic'].get('distinct_icd_code_count', 0)}, "
        f"eicu_codes={report['eicu'].get('distinct_icd9_code_count', 0)}, "
        f"eicu_text={report['eicu'].get('distinct_diagnosis_text_count', 0)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
