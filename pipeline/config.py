"""Configuration for safe source inventory and future data-foundation stages."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

_CODE_ROOT = Path(__file__).resolve().parents[1]


def _path_from_env(name: str, environ: Mapping[str, str]) -> Path | None:
    raw = environ.get(name)
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def resolve_project_root(
    *,
    code_root: Path = _CODE_ROOT,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Resolve the repository root for code, reports, and default local paths."""

    env = os.environ if environ is None else environ
    return (
        _path_from_env("PROJECT_HOME", env)
        or _path_from_env("RESEARCHMODULE_ROOT", env)
        or code_root.resolve()
    )


def resolve_dataset_root(
    project_root: Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Resolve licensed clinical data root (protected NFS on Calculco)."""

    env = os.environ if environ is None else environ
    dataset_root = _path_from_env("DATASET_ROOT", env)
    if dataset_root is not None:
        return dataset_root

    data_protected = _path_from_env("DATA_PROTECTED", env)
    if data_protected is not None:
        return data_protected / "Dataset"

    return project_root / "Dataset"


def resolve_reports_root(
    project_root: Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Resolve aggregate report output directory."""

    env = os.environ if environ is None else environ
    return _path_from_env("REPORTS_ROOT", env) or (project_root / "reports")


def resolve_duckdb_temp_dir(
    *,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Resolve a writable directory DuckDB may use to spill to disk.

    An in-memory DuckDB database has no spill directory by default and cannot
    offload larger-than-memory operators, which makes big ``COPY ... TO parquet``
    unions (notably the eICU ``vital_periodic`` fan-out) get OOM-killed by the
    OAR cgroup before DuckDB's own memory limit engages. DuckDB does not read the
    OS ``TMPDIR`` on its own, so this value is passed explicitly via
    ``SET temp_directory``.

    Resolution order: ``DUCKDB_TEMP_DIR`` -> ``TMPDIR`` -> the system temp dir
    (typically node-local ``/tmp``). The result is not required to exist yet.
    """

    env = os.environ if environ is None else environ
    for name in ("DUCKDB_TEMP_DIR", "TMPDIR"):
        raw = env.get(name)
        if raw:
            return Path(raw).expanduser() / "duckdb_spill"
    return Path(tempfile.gettempdir()) / "duckdb_spill"


def resolve_duckdb_memory_limit(
    *,
    environ: Mapping[str, str] | None = None,
) -> str | None:
    """Resolve an optional DuckDB memory ceiling (e.g. ``"24GB"``).

    Bounding DuckDB below the OAR cgroup allocation lets it spill gracefully
    instead of being SIGKILLed. ``None`` keeps DuckDB's own default (80% of
    physical RAM), which is unsafe under a cgroup smaller than the machine.
    """

    env = os.environ if environ is None else environ
    raw = env.get("DUCKDB_MEMORY_LIMIT")
    value = raw.strip() if raw else ""
    return value or None


def resolve_duckdb_max_temp_dir_size(
    *,
    environ: Mapping[str, str] | None = None,
) -> str | None:
    """Resolve an optional cap on DuckDB's spill (``temp_directory``) size.

    DuckDB defaults ``max_temp_directory_size`` to ~90% of the free space on the
    drive holding ``temp_directory``. When node-local scratch falls back to a
    small ``/tmp`` (e.g. ~12 GiB on some Calculco compute nodes), larger-than-
    memory joins fail with ``failed to offload data block ... (X GiB/X GiB
    used)`` even though ``memory_limit`` is respected. Setting this explicitly
    lets operators grant more spill when ``DUCKDB_TEMP_DIR`` points at a larger
    volume. ``None`` keeps DuckDB's disk-based default.
    """

    env = os.environ if environ is None else environ
    raw = env.get("DUCKDB_MAX_TEMP_DIR_SIZE")
    value = raw.strip() if raw else ""
    return value or None


def resolve_duckdb_threads(
    *,
    environ: Mapping[str, str] | None = None,
) -> int | None:
    """Resolve an optional DuckDB thread cap.

    Fewer threads lower the peak buffered memory of parallel ``COPY`` pipelines.
    ``None`` keeps DuckDB's default (one thread per core).
    """

    env = os.environ if environ is None else environ
    raw = env.get("DUCKDB_THREADS")
    if not raw:
        return None
    try:
        threads = int(raw)
    except ValueError:
        return None
    return threads if threads > 0 else None


PROJECT_ROOT = resolve_project_root()
DATASET_ROOT = resolve_dataset_root(PROJECT_ROOT)
PROCESSED_DATA_ROOT = DATASET_ROOT / "processed"
COHORTS_ROOT = PROCESSED_DATA_ROOT / "cohorts"
EXTRACTS_ROOT = PROCESSED_DATA_ROOT / "extracts"
HARMONIZED_ROOT = PROCESSED_DATA_ROOT / "harmonized"
FEATURES_ROOT = PROCESSED_DATA_ROOT / "features"
TRAINING_ROOT = PROCESSED_DATA_ROOT / "training"
EVALUATION_ROOT = PROCESSED_DATA_ROOT / "evaluation"
MILESTONE7_EVALUATION_ROOT = EVALUATION_ROOT / "milestone7"
MILESTONE8B_EVALUATION_ROOT = EVALUATION_ROOT / "milestone8b"
GRAPH_ROOT = PROCESSED_DATA_ROOT / "graph"
MILESTONE8_GRAPH_ROOT = GRAPH_ROOT / "milestone8"
MAPPING_ROOT = DATASET_ROOT / "mappings"
REPORTS_ROOT = resolve_reports_root(PROJECT_ROOT)

DUCKDB_TEMP_DIR = resolve_duckdb_temp_dir()
DUCKDB_MEMORY_LIMIT = resolve_duckdb_memory_limit()
DUCKDB_MAX_TEMP_DIR_SIZE = resolve_duckdb_max_temp_dir_size()
DUCKDB_THREADS = resolve_duckdb_threads()

RANDOM_SEED = 20260617
COHORT_VERSION = "cohort-manifest-v1"
EXTRACTION_VERSION = "source-extraction-v1"
HARMONIZATION_VERSION = "harmonization-v1"
MEDICATION_MAPPING_VERSION = "medication-rxnorm-atc-v1"
CONDITION_MAPPING_VERSION = "condition-rollup-v1"
FEATURE_VERSION = "temporal-features-v1"
LABEL_VERSION = "observed-medication-label-v1"
SPLIT_VERSION = "patient-split-v1"
BASELINE_VERSION = "baseline-ranking-v1"
EVALUATION_VERSION = "milestone7-evaluation-v1"
GRAPH_VERSION = "graph-suitability-v1"
MILESTONE8_REPORT_VERSION = "milestone8-graph-suitability-v1"
GRAPH_ABLATION_VERSION = "milestone8b-graph-ablation-v1"
MILESTONE8B_REPORT_VERSION = "milestone8b-ablation-evaluation-v1"
DEFAULT_COHORT_PARAMETERS = {
    "unit_of_analysis": "icu_stay",
    "adult_age_minimum": 18,
    "mimic_first_icu_stay_per_admission": True,
    "initial_deep_dive_condition": "sepsis",
}
DEFAULT_MODELING_PARAMETERS = {
    "candidate_top_n_per_condition": 50,
    "candidate_token_strategy": "rxnorm_or_atc",
    "prediction_offset_hours": 24,
    "label_window_hours": 24,
    "split_seed": RANDOM_SEED,
}

# Standard MIMIC-IV (MetaVision) chartevents itemids for core charted vitals,
# mapped to the harmonized normalized_vital_token vocabulary shared with eICU.
# Used both to bound the chartevents extraction (itemid filter) and to build the
# harmonization vital projection so the two stay in sync. Flagged for clinical
# review before pooled analysis; itemids are stable in MIMIC-IV v3.1 d_items.
MIMIC_CHARTEVENTS_VITAL_ITEMIDS = {
    "220045": "heart_rate",
    "220210": "respiratory_rate",
    "220277": "spo2",
    "223761": "temperature",  # Fahrenheit
    "223762": "temperature",  # Celsius
    "220052": "mean_arterial_pressure",  # arterial line, invasive
    "220181": "noninvasive_mean_arterial_pressure",
    "220050": "systolic_blood_pressure",
    "220179": "noninvasive_systolic_blood_pressure",
    "220051": "diastolic_blood_pressure",
    "220180": "noninvasive_diastolic_blood_pressure",
}


@dataclass(frozen=True)
class SourceSpec:
    """Local source dataset layout and expected metadata-only files."""

    name: str
    version: str
    root_relative_path: Path
    expected_files: tuple[Path, ...]
    checksum_files: tuple[Path, ...] = (Path("SHA256SUMS.txt"),)

    @property
    def root(self) -> Path:
        return DATASET_ROOT / self.root_relative_path


@dataclass(frozen=True)
class MappingFileSpec:
    """Expected local mapping resource for concept harmonization."""

    name: str
    relative_path: Path
    required_columns: tuple[str, ...]
    version: str


MEDICATION_MAPPING_SPECS: tuple[MappingFileSpec, ...] = (
    MappingFileSpec(
        name="mimic_ndc_rxnorm_atc",
        relative_path=Path("medications") / "mimic_ndc_rxnorm_atc.csv",
        required_columns=(
            "ndc",
            "rxcui",
            "ingredient_name",
            "rxnorm_name",
            "atc_code",
            "atc_level",
        ),
        version=MEDICATION_MAPPING_VERSION,
    ),
    MappingFileSpec(
        name="eicu_drug_rxnorm_atc",
        relative_path=Path("medications") / "eicu_drug_rxnorm_atc.csv",
        required_columns=(
            "drughiclseqno",
            "gtc",
            "drug_name",
            "rxcui",
            "ingredient_name",
            "rxnorm_name",
            "atc_code",
            "atc_level",
        ),
        version=MEDICATION_MAPPING_VERSION,
    ),
)


CONDITION_MAPPING_SPECS: tuple[MappingFileSpec, ...] = (
    MappingFileSpec(
        name="icd10_ccsr",
        relative_path=Path("conditions") / "icd10_ccsr.csv",
        required_columns=(
            "icd_code",
            "ccsr_category",
            "ccsr_category_description",
        ),
        version=CONDITION_MAPPING_VERSION,
    ),
    MappingFileSpec(
        name="icd9_ccs",
        relative_path=Path("conditions") / "icd9_ccs.csv",
        required_columns=(
            "icd_code",
            "ccs_category",
            "ccs_category_description",
        ),
        version=CONDITION_MAPPING_VERSION,
    ),
    MappingFileSpec(
        name="icd9_to_icd10_gem",
        relative_path=Path("conditions") / "icd9_to_icd10_gem.csv",
        required_columns=(
            "icd9_code",
            "icd10_code",
            "approximate_flag",
        ),
        version=CONDITION_MAPPING_VERSION,
    ),
    MappingFileSpec(
        name="icd_chapters",
        relative_path=Path("conditions") / "icd_chapters.csv",
        required_columns=(
            "icd_version",
            "category_code",
            "chapter_code",
            "chapter_name",
        ),
        version=CONDITION_MAPPING_VERSION,
    ),
    MappingFileSpec(
        name="eicu_diagnosis_text_condition_map",
        relative_path=Path("conditions") / "eicu_diagnosis_text_condition_map.csv",
        required_columns=(
            "diagnosisstring_normalized",
            "condition_rollup_token",
            "condition_name",
        ),
        version=CONDITION_MAPPING_VERSION,
    ),
    MappingFileSpec(
        name="project_condition_groups",
        relative_path=Path("conditions") / "project_condition_groups.csv",
        required_columns=(
            "match_type",
            "match_value",
            "project_condition_group",
            "project_condition_token",
        ),
        version=CONDITION_MAPPING_VERSION,
    ),
)


SOURCE_SPECS: tuple[SourceSpec, ...] = (
    SourceSpec(
        name="mimiciv",
        version="3.1",
        root_relative_path=Path("mimiciv") / "3.1",
        expected_files=(
            Path("hosp") / "admissions.csv.gz",
            Path("hosp") / "patients.csv.gz",
            Path("hosp") / "diagnoses_icd.csv.gz",
            Path("hosp") / "d_icd_diagnoses.csv.gz",
            Path("hosp") / "procedures_icd.csv.gz",
            Path("hosp") / "d_icd_procedures.csv.gz",
            Path("hosp") / "labevents.csv.gz",
            Path("hosp") / "d_labitems.csv.gz",
            Path("hosp") / "prescriptions.csv.gz",
            Path("hosp") / "pharmacy.csv.gz",
            Path("hosp") / "emar.csv.gz",
            Path("hosp") / "emar_detail.csv.gz",
            Path("icu") / "icustays.csv.gz",
            Path("icu") / "chartevents.csv.gz",
            Path("icu") / "d_items.csv.gz",
            Path("icu") / "inputevents.csv.gz",
            Path("icu") / "procedureevents.csv.gz",
        ),
    ),
    SourceSpec(
        name="mimiciv_note",
        version="2.2",
        root_relative_path=Path("2.2"),
        expected_files=(
            Path("note") / "discharge.csv.gz",
            Path("note") / "discharge_detail.csv",
            Path("note") / "radiology.csv.gz",
            Path("note") / "radiology_detail.csv",
        ),
    ),
    SourceSpec(
        name="eicu_crd",
        version="2.0",
        root_relative_path=Path("eicu-crd") / "2.0",
        expected_files=(
            Path("patient.csv.gz"),
            Path("diagnosis.csv.gz"),
            Path("lab.csv.gz"),
            Path("medication.csv.gz"),
            Path("infusionDrug.csv.gz"),
            Path("allergy.csv.gz"),
            Path("treatment.csv.gz"),
            Path("vitalPeriodic.csv.gz"),
            Path("vitalAperiodic.csv.gz"),
            Path("apachePatientResult.csv.gz"),
            Path("apacheApsVar.csv.gz"),
            Path("apachePredVar.csv.gz"),
            Path("note.csv.gz"),
        ),
    ),
)

LARGE_TABLES: frozenset[tuple[str, Path]] = frozenset(
    {
        ("mimiciv", Path("hosp") / "labevents.csv.gz"),
        ("mimiciv", Path("hosp") / "prescriptions.csv.gz"),
        ("mimiciv", Path("hosp") / "pharmacy.csv.gz"),
        ("mimiciv", Path("hosp") / "emar.csv.gz"),
        ("mimiciv", Path("hosp") / "emar_detail.csv.gz"),
        ("mimiciv", Path("hosp") / "poe.csv.gz"),
        ("mimiciv", Path("icu") / "chartevents.csv.gz"),
        ("mimiciv", Path("icu") / "inputevents.csv.gz"),
        ("mimiciv", Path("icu") / "ingredientevents.csv.gz"),
        ("eicu_crd", Path("lab.csv.gz")),
        ("eicu_crd", Path("medication.csv.gz")),
        ("eicu_crd", Path("nurseCharting.csv.gz")),
        ("eicu_crd", Path("vitalPeriodic.csv.gz")),
        ("eicu_crd", Path("vitalAperiodic.csv.gz")),
        ("eicu_crd", Path("intakeOutput.csv.gz")),
    }
)
DEFAULT_LARGE_TABLE_BYTES = 100_000_000


def ensure_local_directories() -> None:
    """Create ignored local output directories used by data-foundation scripts."""

    PROCESSED_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    COHORTS_ROOT.mkdir(parents=True, exist_ok=True)
    EXTRACTS_ROOT.mkdir(parents=True, exist_ok=True)
    HARMONIZED_ROOT.mkdir(parents=True, exist_ok=True)
    FEATURES_ROOT.mkdir(parents=True, exist_ok=True)
    TRAINING_ROOT.mkdir(parents=True, exist_ok=True)
    EVALUATION_ROOT.mkdir(parents=True, exist_ok=True)
    MILESTONE7_EVALUATION_ROOT.mkdir(parents=True, exist_ok=True)
    GRAPH_ROOT.mkdir(parents=True, exist_ok=True)
    MILESTONE8_GRAPH_ROOT.mkdir(parents=True, exist_ok=True)
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
