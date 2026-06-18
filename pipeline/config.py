"""Configuration for safe source inventory and future data-foundation stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "Dataset"
PROCESSED_DATA_ROOT = DATASET_ROOT / "processed"
COHORTS_ROOT = PROCESSED_DATA_ROOT / "cohorts"
REPORTS_ROOT = PROJECT_ROOT / "reports"

RANDOM_SEED = 20260617
DEFAULT_COHORT_PARAMETERS = {
    "unit_of_analysis": "icu_stay",
    "adult_age_minimum": 18,
    "mimic_first_icu_stay_per_admission": True,
    "initial_deep_dive_condition": "sepsis",
}
DEFAULT_MODELING_PARAMETERS = {
    "candidate_top_n_per_condition": 50,
    "prediction_offset_hours": 24,
    "label_window_hours": 24,
    "split_seed": RANDOM_SEED,
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
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
