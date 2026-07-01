import csv
import importlib.util
import sys
import zipfile
from pathlib import Path


def load_fetch_module():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "fetch_condition_reference_files.py"
    )
    spec = importlib.util.spec_from_file_location(
        "fetch_condition_reference_files", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fetch = load_fetch_module()


# Minimal synthetic fixtures mirroring the authoritative source layouts.
# Single quotes are AHRQ text qualifiers and must be stripped by the parsers.
DXCCSR_HEADER = (
    "'ICD-10-CM CODE','ICD-10-CM CODE DESCRIPTION',"
    "'Default CCSR CATEGORY IP','Default CCSR CATEGORY DESCRIPTION IP',"
    "'Default CCSR CATEGORY OP','Default CCSR CATEGORY DESCRIPTION OP',"
    "'CCSR CATEGORY 1','CCSR CATEGORY 1 DESCRIPTION',"
    "'CCSR CATEGORY 2','CCSR CATEGORY 2 DESCRIPTION','Rationale'"
)
DXCCSR_ROWS = [
    "'A419',\"Sepsis, unspecified organism\",'INF002',Septicemia,"
    "'INF002',Septicemia,'INF002',Septicemia,' ',,06 Infectious",
    "'I10',\"Essential (primary) hypertension\",'CIR007',"
    "Essential hypertension,'CIR007',Essential hypertension,"
    "'CIR007',Essential hypertension,' ',,07 Circulatory",
]

DXREF_LINES = [
    "NOTE: New codes are introduced in October.",
    "'ICD-9-CM CODE','CCS CATEGORY','CCS CATEGORY DESCRIPTION',"
    "'ICD-9-CM CODE DESCRIPTION','OPTIONAL CCS CATEGORY',"
    "'OPTIONAL CCS CATEGORY DESCRIPTION'",
    "'     ','0    ','No DX',\"INVALID CODES IN USER DATA\",' ',' '",
    "'0389','2    ','Septicemia',\"SEPTICEMIA NOS\",' ',' '",
    "'4019','98   ','HTN',\"HYPERTENSION NOS\",' ',' '",
]

GEM_LINES = [
    "0389  A419    10000",
    "0010  A000    00000",
    "9999  NoDx    10001",  # no-map row must be skipped
]


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_zip(path: Path, member_name: str, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(member_name, text)


def test_parse_ccsr_uses_default_category(tmp_path: Path) -> None:
    csv_path = tmp_path / "DXCCSR_v2026-1.csv"
    _write(csv_path, "\n".join([DXCCSR_HEADER, *DXCCSR_ROWS]) + "\n")
    rows = fetch.parse_ccsr(csv_path)
    by_code = {row["icd_code"]: row for row in rows}
    assert by_code["A419"]["ccsr_category"] == "INF002"
    assert by_code["A419"]["ccsr_category_description"] == "Septicemia"
    assert by_code["I10"]["ccsr_category"] == "CIR007"
    # No stray single quotes survive.
    assert "'" not in by_code["A419"]["icd_code"]


def test_parse_ccs_skips_note_and_invalid(tmp_path: Path) -> None:
    csv_path = tmp_path / "dxref 2015.csv"
    _write(csv_path, "\n".join(DXREF_LINES) + "\n")
    rows = fetch.parse_ccs(csv_path)
    by_code = {row["icd_code"]: row for row in rows}
    assert "     " not in by_code  # invalid No-DX row dropped
    assert by_code["0389"]["ccs_category"] == "2"
    assert by_code["0389"]["ccs_category_description"] == "Septicemia"
    assert by_code["4019"]["ccs_category"] == "98"


def test_parse_gem_skips_no_map_and_sets_flag(tmp_path: Path) -> None:
    txt_path = tmp_path / "2018_I9gem.txt"
    _write(txt_path, "\n".join(GEM_LINES) + "\n")
    rows = fetch.parse_gem(txt_path)
    by_source = {row["icd9_code"]: row for row in rows}
    assert "9999" not in by_source  # no-map row skipped
    assert by_source["0389"]["icd10_code"] == "A419"
    assert by_source["0389"]["approximate_flag"] == "1"
    assert by_source["0010"]["approximate_flag"] == "0"


def test_chapter_lookup_boundaries() -> None:
    assert fetch.icd10_chapter_for("A41")[0] == "01"
    assert fetch.icd10_chapter_for("I10")[0] == "09"
    # Within-letter split: C00-D49 neoplasms vs D50-D89 blood.
    assert fetch.icd10_chapter_for("D49")[0] == "02"
    assert fetch.icd10_chapter_for("D50")[0] == "03"
    assert fetch.icd9_chapter_for("038")[0] == "01"
    assert fetch.icd9_chapter_for("E95")[0] == "18"
    assert fetch.icd9_chapter_for("V30")[0] == "19"


def test_build_chapter_rows_grounded_in_code_lists() -> None:
    ccsr_rows = [
        {
            "icd_code": "A419",
            "ccsr_category": "INF002",
            "ccsr_category_description": "",
        },
        {"icd_code": "I10", "ccsr_category": "CIR007", "ccsr_category_description": ""},
    ]
    ccs_rows = [
        {"icd_code": "0389", "ccs_category": "2", "ccs_category_description": ""},
    ]
    rows = fetch.build_chapter_rows(ccsr_rows, ccs_rows)
    icd10 = {r["category_code"]: r for r in rows if r["icd_version"] == "10"}
    icd9 = {r["category_code"]: r for r in rows if r["icd_version"] == "9"}
    assert icd10["A41"]["chapter_code"] == "01"
    assert icd10["I10"]["chapter_code"] == "09"
    assert icd9["038"]["chapter_code"] == "01"


def test_fetch_end_to_end_offline_matches_specs(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    _write_zip(
        cache_dir / "DXCCSR.zip",
        "DXCCSR-v2026-1/DXCCSR_v2026-1.csv",
        "\n".join([DXCCSR_HEADER, *DXCCSR_ROWS]) + "\n",
    )
    _write_zip(
        cache_dir / "Single_Level_CCS.zip",
        "$dxref 2015.csv",
        "\n".join(DXREF_LINES) + "\n",
    )
    _write(cache_dir / "2018_I9gem.txt", "\n".join(GEM_LINES) + "\n")

    config = fetch.ConditionReferenceConfig(
        dataset_root=tmp_path / "Dataset",
        mapping_root=tmp_path / "Dataset" / "mappings",
        reports_root=tmp_path / "reports",
        cache_dir=cache_dir,
        skip_download=True,
    )
    report = fetch.fetch_condition_reference_files(config)

    conditions = config.conditions_root
    assert (conditions / "icd10_ccsr.csv").exists()
    assert (conditions / "icd9_ccs.csv").exists()
    assert (conditions / "icd9_to_icd10_gem.csv").exists()
    assert (conditions / "icd_chapters.csv").exists()

    # Written headers must satisfy the config contract exactly.
    from pipeline.config import CONDITION_MAPPING_SPECS

    specs = {spec.name: list(spec.required_columns) for spec in CONDITION_MAPPING_SPECS}
    for name, filename in (
        ("icd10_ccsr", "icd10_ccsr.csv"),
        ("icd9_ccs", "icd9_ccs.csv"),
        ("icd9_to_icd10_gem", "icd9_to_icd10_gem.csv"),
        ("icd_chapters", "icd_chapters.csv"),
    ):
        with (conditions / filename).open(newline="", encoding="utf-8") as handle:
            header = next(csv.reader(handle))
        assert set(specs[name]).issubset(set(header)), (name, header)

    assert report["data_safety"]["contains_patient_rows"] is False
    assert report["counts"]["icd9_to_icd10_gem_rows"] == 2  # no-map skipped
