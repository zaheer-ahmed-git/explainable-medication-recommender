from pathlib import Path

from pipeline import config


def test_configured_source_roots_are_under_dataset_root() -> None:
    dataset_root = config.DATASET_ROOT.resolve()

    for source_spec in config.SOURCE_SPECS:
        assert source_spec.root.resolve().is_relative_to(dataset_root)


def test_expected_sources_include_mimic_note_and_eicu() -> None:
    source_names = {source_spec.name for source_spec in config.SOURCE_SPECS}

    assert {"mimiciv", "mimiciv_note", "eicu_crd"} <= source_names


def test_large_tables_include_known_multi_gb_sources() -> None:
    assert ("mimiciv", Path("hosp") / "labevents.csv.gz") in config.LARGE_TABLES
    assert ("mimiciv", Path("icu") / "chartevents.csv.gz") in config.LARGE_TABLES
    assert ("eicu_crd", Path("vitalPeriodic.csv.gz")) in config.LARGE_TABLES


def test_default_modeling_parameters_are_reproducible() -> None:
    assert config.DEFAULT_MODELING_PARAMETERS["split_seed"] == config.RANDOM_SEED
    assert config.DEFAULT_MODELING_PARAMETERS["prediction_offset_hours"] == 24
    assert config.DEFAULT_MODELING_PARAMETERS["label_window_hours"] == 24
