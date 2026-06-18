import json
from pathlib import Path

from pipeline.eda_summary import build_eda_summary, generate_eda_outputs


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def synthetic_inventory() -> dict:
    return {
        "schema_version": "source-inventory-v1",
        "sources": [
            {
                "name": "source_a",
                "version": "1.0",
                "present": True,
                "file_count": 2,
                "missing_expected_files": [],
                "files": [
                    {
                        "relative_path": "table.csv",
                        "size_bytes": 100,
                        "requires_bounded_query": False,
                    },
                    {
                        "relative_path": "large.csv.gz",
                        "size_bytes": 200,
                        "requires_bounded_query": True,
                    },
                ],
            }
        ],
    }


def synthetic_cohort() -> dict:
    return {
        "schema_version": "cohort-manifest-v1",
        "configuration": {"adult_age_minimum": 18},
        "sources": {
            "mimiciv": {
                "selected_stays": 10,
                "selected_patients": 8,
                "duplicate_stay_uid_count": 0,
                "excluded_by_first_stay_rule": 1,
            },
            "eicu_crd": {
                "selected_stays": 20,
                "selected_patients": 15,
                "duplicate_stay_uid_count": 0,
                "missing_or_unparseable_age_stays": 2,
                "topcoded_age_stays": 3,
            },
            "unified": {
                "selected_stays": 30,
                "selected_patients": 23,
                "duplicate_stay_uid_count": 0,
            },
        },
    }


def synthetic_quality() -> dict:
    return {
        "schema_version": "quality-profile-v1",
        "table_count": 4,
        "tables": [
            {
                "status": "completed",
                "source": "mimiciv",
                "table_name": "mimic_patients",
                "row_count": 2,
                "column_count": 3,
                "requires_bounded_query": False,
                "duplicate_key_profile": {"duplicate_excess_rows": 0},
                "referential_integrity": [],
                "column_profiles": {
                    "patient_uid": {
                        "null_rate": 0.0,
                        "null_count": 0,
                    }
                },
                "numeric_profiles": {},
            },
            {
                "status": "completed",
                "source": "eicu_crd",
                "table_name": "eicu_vital_periodic",
                "row_count": 5,
                "column_count": 4,
                "requires_bounded_query": True,
                "duplicate_key_profile": {"duplicate_excess_rows": 0},
                "referential_integrity": [],
                "column_profiles": {
                    "temperature": {
                        "null_rate": 0.6,
                        "null_count": 3,
                    }
                },
                "numeric_profiles": {
                    "temperature": {
                        "parse_failure_count": 0,
                        "out_of_bounds_count": 1,
                        "minimum_allowed": 20,
                        "maximum_allowed": 45,
                    }
                },
            },
            {
                "status": "completed",
                "source": "eicu_crd",
                "table_name": "eicu_infusion_drug",
                "row_count": 4,
                "column_count": 4,
                "requires_bounded_query": False,
                "duplicate_key_profile": {"duplicate_excess_rows": 0},
                "referential_integrity": [],
                "column_profiles": {},
                "numeric_profiles": {},
            },
            {
                "status": "scan_failed",
                "source": "eicu_crd",
                "table_name": "eicu_medication",
                "relative_path": "medication.csv.gz",
                "error_type": "IOException",
                "error_message": "safe aggregate error",
            },
        ],
    }


def test_build_eda_summary_synthesizes_readiness_and_findings() -> None:
    summary = build_eda_summary(
        synthetic_inventory(),
        synthetic_cohort(),
        synthetic_quality(),
    )

    assert summary["data_safety"]["contains_patient_rows"] is False
    assert summary["inventory"]["source_count"] == 1
    assert summary["cohort"]["sources"][-1]["selected_stays"] == 30
    assert summary["quality"]["status_summary"]["completed_table_count"] == 3
    assert summary["quality"]["status_summary"]["scan_failed_table_count"] == 1
    assert (
        summary["quality"]["findings"]["numeric_out_of_bounds"][0]["column"]
        == "temperature"
    )
    assert (
        summary["quality"]["domain_readiness"]["medications"]["status"]
        == "partial_requires_review"
    )


def test_generate_eda_outputs_writes_safe_reports_and_figures(tmp_path: Path) -> None:
    inventory_path = tmp_path / "source_inventory.json"
    cohort_path = tmp_path / "cohort_manifest.json"
    quality_path = tmp_path / "quality_profile.json"
    output_json = tmp_path / "eda.json"
    output_markdown = tmp_path / "eda.md"
    figures_root = tmp_path / "figures"
    write_json(inventory_path, synthetic_inventory())
    write_json(cohort_path, synthetic_cohort())
    write_json(quality_path, synthetic_quality())

    summary = generate_eda_outputs(
        inventory_path=inventory_path,
        cohort_path=cohort_path,
        quality_path=quality_path,
        output_json=output_json,
        output_markdown=output_markdown,
        figures_root=figures_root,
    )
    json_text = output_json.read_text(encoding="utf-8")
    markdown_text = output_markdown.read_text(encoding="utf-8")

    assert output_json.exists()
    assert output_markdown.exists()
    assert len(summary["figures"]) == 4
    assert all(
        Path(path).exists() and Path(path).stat().st_size > 0
        for path in summary["figures"]
    )
    assert "EDA Dataset Understanding Brief" in markdown_text
    assert "contains_patient_rows" in json_text
