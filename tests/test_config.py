from pathlib import Path

from pipeline import config


def test_configured_source_roots_are_under_dataset_root() -> None:
    dataset_root = config.DATASET_ROOT.resolve()

    for source_spec in config.SOURCE_SPECS:
        assert source_spec.root.resolve().is_relative_to(dataset_root)


def test_resolve_project_root_prefers_project_home(tmp_path: Path) -> None:
    custom_root = tmp_path / "repo"
    custom_root.mkdir()

    resolved = config.resolve_project_root(
        code_root=tmp_path / "ignored",
        environ={"PROJECT_HOME": str(custom_root)},
    )

    assert resolved == custom_root.resolve()


def test_resolve_dataset_root_prefers_dataset_root_env(tmp_path: Path) -> None:
    project_root = tmp_path / "repo"
    dataset_root = tmp_path / "protected" / "Dataset"
    project_root.mkdir()
    dataset_root.mkdir(parents=True)

    resolved = config.resolve_dataset_root(
        project_root,
        environ={"DATASET_ROOT": str(dataset_root)},
    )

    assert resolved == dataset_root.resolve()


def test_resolve_dataset_root_uses_data_protected_fallback(tmp_path: Path) -> None:
    project_root = tmp_path / "repo"
    data_protected = tmp_path / "protected"
    project_root.mkdir()
    data_protected.mkdir()

    resolved = config.resolve_dataset_root(
        project_root,
        environ={"DATA_PROTECTED": str(data_protected)},
    )

    assert resolved == (data_protected / "Dataset").resolve()


def test_resolve_dataset_root_defaults_to_project_dataset(tmp_path: Path) -> None:
    project_root = tmp_path / "repo"
    project_root.mkdir()

    resolved = config.resolve_dataset_root(project_root, environ={})

    assert resolved == (project_root / "Dataset").resolve()


def test_resolve_reports_root_prefers_env_override(tmp_path: Path) -> None:
    project_root = tmp_path / "repo"
    reports_root = tmp_path / "scratch" / "reports"
    project_root.mkdir()
    reports_root.mkdir(parents=True)

    resolved = config.resolve_reports_root(
        project_root,
        environ={"REPORTS_ROOT": str(reports_root)},
    )

    assert resolved == reports_root.resolve()


def test_resolve_duckdb_temp_dir_prefers_explicit_env(tmp_path: Path) -> None:
    resolved = config.resolve_duckdb_temp_dir(
        environ={"DUCKDB_TEMP_DIR": str(tmp_path), "TMPDIR": "/should/not/win"},
    )

    assert resolved == tmp_path / "duckdb_spill"


def test_resolve_duckdb_temp_dir_falls_back_to_tmpdir(tmp_path: Path) -> None:
    resolved = config.resolve_duckdb_temp_dir(environ={"TMPDIR": str(tmp_path)})

    assert resolved == tmp_path / "duckdb_spill"


def test_resolve_duckdb_temp_dir_defaults_to_system_temp() -> None:
    resolved = config.resolve_duckdb_temp_dir(environ={})

    assert resolved.name == "duckdb_spill"
    assert resolved.parent.exists()


def test_resolve_duckdb_memory_limit_reads_env() -> None:
    assert (
        config.resolve_duckdb_memory_limit(environ={"DUCKDB_MEMORY_LIMIT": "24GB"})
        == "24GB"
    )
    assert (
        config.resolve_duckdb_memory_limit(environ={"DUCKDB_MEMORY_LIMIT": "  "})
        is None
    )
    assert config.resolve_duckdb_memory_limit(environ={}) is None


def test_resolve_duckdb_max_temp_dir_size_reads_env() -> None:
    assert (
        config.resolve_duckdb_max_temp_dir_size(
            environ={"DUCKDB_MAX_TEMP_DIR_SIZE": "150GB"}
        )
        == "150GB"
    )
    assert (
        config.resolve_duckdb_max_temp_dir_size(
            environ={"DUCKDB_MAX_TEMP_DIR_SIZE": "  "}
        )
        is None
    )
    assert config.resolve_duckdb_max_temp_dir_size(environ={}) is None


def test_resolve_duckdb_threads_reads_positive_int() -> None:
    assert config.resolve_duckdb_threads(environ={"DUCKDB_THREADS": "8"}) == 8
    assert config.resolve_duckdb_threads(environ={"DUCKDB_THREADS": "0"}) is None
    assert (
        config.resolve_duckdb_threads(environ={"DUCKDB_THREADS": "not-a-number"})
        is None
    )
    assert config.resolve_duckdb_threads(environ={}) is None


def test_expected_sources_include_mimic_note_and_eicu() -> None:
    source_names = {source_spec.name for source_spec in config.SOURCE_SPECS}

    assert {"mimiciv", "mimiciv_note", "eicu_crd"} <= source_names


def test_large_tables_include_known_multi_gb_sources() -> None:
    assert ("mimiciv", Path("hosp") / "labevents.csv.gz") in config.LARGE_TABLES
    assert ("mimiciv", Path("icu") / "chartevents.csv.gz") in config.LARGE_TABLES
    assert ("eicu_crd", Path("vitalPeriodic.csv.gz")) in config.LARGE_TABLES


def test_default_modeling_parameters_are_reproducible() -> None:
    assert config.COHORT_VERSION == "cohort-manifest-v1"
    assert config.DEFAULT_MODELING_PARAMETERS["split_seed"] == config.RANDOM_SEED
    assert config.DEFAULT_MODELING_PARAMETERS["prediction_offset_hours"] == 24
    assert config.DEFAULT_MODELING_PARAMETERS["label_window_hours"] == 24


def test_milestone5_output_roots_are_under_processed_dataset() -> None:
    assert config.EXTRACTS_ROOT == config.PROCESSED_DATA_ROOT / "extracts"
    assert config.HARMONIZED_ROOT == config.PROCESSED_DATA_ROOT / "harmonized"
    assert config.MAPPING_ROOT == config.DATASET_ROOT / "mappings"


def test_milestone6_output_roots_and_versions_are_configured() -> None:
    assert config.FEATURES_ROOT == config.PROCESSED_DATA_ROOT / "features"
    assert config.TRAINING_ROOT == config.PROCESSED_DATA_ROOT / "training"
    assert config.FEATURE_VERSION == "temporal-features-v1"
    assert config.LABEL_VERSION == "observed-medication-label-v1"
    assert config.SPLIT_VERSION == "patient-split-v1"


def test_milestone7_output_roots_and_versions_are_configured() -> None:
    assert config.EVALUATION_ROOT == config.PROCESSED_DATA_ROOT / "evaluation"
    assert config.MILESTONE7_EVALUATION_ROOT == config.EVALUATION_ROOT / "milestone7"
    assert config.BASELINE_VERSION == "baseline-ranking-v1"
    assert config.EVALUATION_VERSION == "milestone7-evaluation-v1"


def test_milestone8_graph_roots_and_versions_are_configured() -> None:
    assert config.GRAPH_ROOT == config.PROCESSED_DATA_ROOT / "graph"
    assert config.MILESTONE8_GRAPH_ROOT == config.GRAPH_ROOT / "milestone8"
    assert config.GRAPH_VERSION == "graph-suitability-v1"
    assert config.MILESTONE8_REPORT_VERSION == "milestone8-graph-suitability-v1"


def test_milestone8b_graph_ablation_roots_and_versions_are_configured() -> None:
    assert config.MILESTONE8B_EVALUATION_ROOT == (
        config.EVALUATION_ROOT / "milestone8b"
    )
    assert config.GRAPH_ABLATION_VERSION == "milestone8b-graph-ablation-v1"
    assert config.MILESTONE8B_REPORT_VERSION == ("milestone8b-ablation-evaluation-v1")


def test_medication_mapping_specs_define_required_rxnorm_atc_columns() -> None:
    specs = {spec.name: spec for spec in config.MEDICATION_MAPPING_SPECS}

    assert {"mimic_ndc_rxnorm_atc", "eicu_drug_rxnorm_atc"} == set(specs)
    assert {"rxcui", "ingredient_name", "atc_code"} <= set(
        specs["mimic_ndc_rxnorm_atc"].required_columns
    )
    assert {"drughiclseqno", "gtc", "drug_name", "rxcui", "atc_code"} <= set(
        specs["eicu_drug_rxnorm_atc"].required_columns
    )


def test_condition_mapping_specs_are_optional_shared_rollup_resources() -> None:
    specs = {spec.name: spec for spec in config.CONDITION_MAPPING_SPECS}

    assert {
        "icd10_ccsr",
        "icd9_ccs",
        "icd9_to_icd10_gem",
        "icd_chapters",
        "eicu_diagnosis_text_condition_map",
        "project_condition_groups",
    } == set(specs)
    assert config.CONDITION_MAPPING_VERSION == "condition-rollup-v1"
    for spec in specs.values():
        assert spec.relative_path.parts[0] == "conditions"
        assert spec.version == config.CONDITION_MAPPING_VERSION
    assert {"icd_code", "ccsr_category"} <= set(specs["icd10_ccsr"].required_columns)
    assert {"match_type", "match_value", "project_condition_token"} <= set(
        specs["project_condition_groups"].required_columns
    )
