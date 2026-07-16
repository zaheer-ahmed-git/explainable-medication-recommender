"""Bootstrap aggregate condition mapping templates for harmonization.

Unlike ``build_medication_mappings.py`` (a hard gate), condition semantic
mapping is optional. By default, this script does not fabricate ICD/CCS/CCSR,
SNOMED, diagnosis-text, or project-condition mappings. It inventories the
distinct diagnosis concepts present in the cohort-filtered extracts and writes
empty, review-ready template CSVs plus an aggregate build report under
``$DATASET_ROOT/mappings/conditions``. With ``--write-curated-sepsis``, it also
merges the approved A1/B3 sepsis index-condition policy into the active local
mapping files consumed by ``pipeline.harmonize``.

Curators then either drop authoritative CCS/CCSR/GEM/chapter reference files into
that folder (see ``fetch_condition_reference_files.py``) or fill the templates by
hand before re-running ``pipeline.harmonize``. Every output is aggregate-only:
distinct concept keys and row counts, never patient-level rows or identifiers.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

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
SEPSIS_PROJECT_GROUP = "sepsis"
SEPSIS_PROJECT_TOKEN = "condition:sepsis"
SEPSIS_CONDITION_NAME = "Sepsis"
SEPSIS_EXACT_ICD_CODES = ("995.91", "995.92", "785.52", "R65.20", "R65.21")
SEPSIS_ICD_PREFIXES = ("A40", "A41")
SEPSIS_BASE_TEXT_TOKENS = ("sepsis", "severe_sepsis", "septic_shock")

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


def normalize_text_value(value: str) -> str:
    """Normalize text using the same token shape as SQL ``text_key``."""

    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def normalize_code_value(value: str) -> str:
    """Normalize codes using the same token shape as SQL ``code_key``."""

    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def is_sepsis_text_token(token: str) -> bool:
    """Return whether a normalized eICU diagnosis text token matches A1 sepsis."""

    normalized = normalize_text_value(token)
    if not normalized:
        return False
    parts = {part for part in normalized.split("_") if part}
    if "sepsis" in parts or "septicemia" in parts:
        return True
    return "septic" in parts and "shock" in parts


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
    write_curated_sepsis: bool = False

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


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    """Read a small mapping CSV if present."""

    import csv

    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_dict_csv(
    path: Path,
    columns: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """Write mapping dictionaries with a stable column order."""

    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


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


def curated_sepsis_text_tokens(config: ConditionMappingBuildConfig) -> list[str]:
    """Return discovered and policy-default eICU text tokens for sepsis mapping."""

    tokens = {normalize_text_value(token) for token in SEPSIS_BASE_TEXT_TOKENS}
    distinct_path = (
        config.condition_mapping_root / "eicu_distinct_diagnosis_text_for_mapping.csv"
    )
    for row in read_csv_dicts(distinct_path):
        token = normalize_text_value(row.get("diagnosisstring_normalized", ""))
        if is_sepsis_text_token(token):
            tokens.add(token)
    return sorted(token for token in tokens if token)


def mapping_row_key(
    row: Mapping[str, Any],
    key_columns: Sequence[str],
) -> tuple[str, ...]:
    """Return a normalized key for merging small curator mapping CSVs."""

    if tuple(key_columns) == ("match_type", "match_value"):
        match_type = normalize_text_value(str(row.get("match_type", "")))
        match_value = str(row.get("match_value", ""))
        if match_type in {"icd_code", "icd_prefix", "icd_code_prefix"}:
            normalized_value = normalize_code_value(match_value)
        else:
            normalized_value = normalize_text_value(match_value)
        return (match_type, normalized_value)
    return tuple(
        normalize_text_value(str(row.get(column, ""))) for column in key_columns
    )


def merge_rows_by_key(
    *,
    existing_rows: Sequence[dict[str, str]],
    new_rows: Sequence[Mapping[str, Any]],
    key_columns: Sequence[str],
) -> tuple[list[dict[str, str]], int]:
    """Append rows not already present, preserving existing curator edits."""

    merged = [dict(row) for row in existing_rows]
    seen = {mapping_row_key(row, key_columns) for row in merged}
    added = 0
    for row in new_rows:
        key = mapping_row_key(row, key_columns)
        if key in seen:
            continue
        merged.append(
            {
                str(column): "" if value is None else str(value)
                for column, value in row.items()
            }
        )
        seen.add(key)
        added += 1
    return merged, added


def write_curated_sepsis_mappings(
    config: ConditionMappingBuildConfig,
) -> dict[str, Any]:
    """Write active A1/B3 sepsis mapping CSVs from the approved policy."""

    condition_root = config.condition_mapping_root
    text_tokens = curated_sepsis_text_tokens(config)

    text_path = condition_root / "eicu_diagnosis_text_condition_map.csv"
    text_columns = (
        "diagnosisstring_normalized",
        "condition_rollup_token",
        "condition_name",
        "mapping_basis",
        "review_status",
        "mapping_version",
    )
    text_rows = [
        {
            "diagnosisstring_normalized": token,
            "condition_rollup_token": SEPSIS_PROJECT_TOKEN,
            "condition_name": SEPSIS_CONDITION_NAME,
            "mapping_basis": "approved_a1_b3_sepsis_text",
            "review_status": "approved_policy_2026-07-04",
            "mapping_version": CONDITION_MAPPING_VERSION,
        }
        for token in text_tokens
    ]
    merged_text, added_text = merge_rows_by_key(
        existing_rows=read_csv_dicts(text_path),
        new_rows=text_rows,
        key_columns=("diagnosisstring_normalized",),
    )
    write_dict_csv(text_path, text_columns, merged_text)

    project_path = condition_root / "project_condition_groups.csv"
    project_columns = (
        "match_type",
        "match_value",
        "project_condition_group",
        "project_condition_token",
        "mapping_basis",
        "review_status",
        "mapping_version",
    )
    project_rows: list[dict[str, str]] = []
    project_rows.extend(
        {
            "match_type": "icd_code",
            "match_value": code,
            "project_condition_group": SEPSIS_PROJECT_GROUP,
            "project_condition_token": SEPSIS_PROJECT_TOKEN,
            "mapping_basis": "approved_a1_exact_icd_code",
            "review_status": "approved_policy_2026-07-04",
            "mapping_version": CONDITION_MAPPING_VERSION,
        }
        for code in SEPSIS_EXACT_ICD_CODES
    )
    project_rows.extend(
        {
            "match_type": "icd_prefix",
            "match_value": prefix,
            "project_condition_group": SEPSIS_PROJECT_GROUP,
            "project_condition_token": SEPSIS_PROJECT_TOKEN,
            "mapping_basis": "approved_a1_icd_prefix",
            "review_status": "approved_policy_2026-07-04",
            "mapping_version": CONDITION_MAPPING_VERSION,
        }
        for prefix in SEPSIS_ICD_PREFIXES
    )
    project_rows.extend(
        {
            "match_type": "text_token",
            "match_value": token,
            "project_condition_group": SEPSIS_PROJECT_GROUP,
            "project_condition_token": SEPSIS_PROJECT_TOKEN,
            "mapping_basis": "approved_a1_b3_sepsis_text",
            "review_status": "approved_policy_2026-07-04",
            "mapping_version": CONDITION_MAPPING_VERSION,
        }
        for token in text_tokens
    )

    merged_project, added_project = merge_rows_by_key(
        existing_rows=read_csv_dicts(project_path),
        new_rows=project_rows,
        key_columns=("match_type", "match_value"),
    )
    write_dict_csv(project_path, project_columns, merged_project)

    return {
        "status": "written",
        "policy": "A1 coded sepsis + B3 project group",
        "eicu_text_tokens_considered": len(text_tokens),
        "eicu_text_rows_added": added_text,
        "project_group_rows_added": added_project,
        "active_files": [
            "eicu_diagnosis_text_condition_map.csv",
            "project_condition_groups.csv",
        ],
    }


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
    curated_sepsis = (
        write_curated_sepsis_mappings(config)
        if config.write_curated_sepsis
        else {
            "status": "not_requested",
            "policy": "A1 coded sepsis + B3 project group",
            "active_files": [],
        }
    )

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
        "curated_sepsis": curated_sepsis,
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
    parser.add_argument(
        "--write-curated-sepsis",
        action="store_true",
        help=(
            "Merge the approved A1/B3 sepsis code/text mappings into the active "
            "condition mapping CSVs consumed by harmonization."
        ),
    )
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
            write_curated_sepsis=args.write_curated_sepsis,
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
