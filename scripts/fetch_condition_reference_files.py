"""Download and transform authoritative condition reference files.

Sources (aggregate classification vocabulary only; no patient data):

- AHRQ HCUP Clinical Classifications Software Refined (CCSR) for ICD-10-CM
  diagnoses (public domain). https://hcup-us.ahrq.gov/toolssoftware/ccsr/dxccsr.jsp
- AHRQ HCUP single-level Clinical Classifications Software (CCS) for ICD-9-CM
  diagnoses (public domain). https://hcup-us.ahrq.gov/toolssoftware/ccs/ccs.jsp
- CDC/NCHS ICD-9-CM to ICD-10-CM General Equivalence Mappings (GEM, public
  domain). https://ftp.cdc.gov/pub/health_statistics/nchs/Publications/ICD10CM/2018/

Outputs under ``$DATASET_ROOT/mappings/conditions/`` matching
``CONDITION_MAPPING_SPECS`` in ``pipeline/config.py``:

- ``icd10_ccsr.csv`` — ICD-10-CM code to default CCSR category (+ description)
- ``icd9_ccs.csv`` — ICD-9-CM code to single-level CCS category (+ description)
- ``icd9_to_icd10_gem.csv`` — ICD-9-CM to ICD-10-CM GEM with approximate flag
- ``icd_chapters.csv`` — ICD-9/ICD-10 three-character category to chapter,
  derived from the published structural chapter ranges (fallback roll-up)

This script only reproduces public classification vocabularies. It does not
fabricate mappings and does not read or emit any patient-level data. Raw
downloads are cached under an ignored directory and are never committed.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import shutil
import sys
import urllib.request
import zipfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.config import (  # noqa: E402
    CONDITION_MAPPING_SPECS,
    CONDITION_MAPPING_VERSION,
    DATASET_ROOT,
    MAPPING_ROOT,
    REPORTS_ROOT,
)

SCHEMA_VERSION = "condition-reference-fetch-v1"

# Authoritative default source URLs (fiscal-year pinned; override via CLI).
CCSR_URL = "https://hcup-us.ahrq.gov/toolssoftware/ccsr/DXCCSR-v2026-1.zip"
CCS_URL = "https://hcup-us.ahrq.gov/toolssoftware/ccs/Single_Level_CCS_2015.zip"
GEM_I9_URL = (
    "https://ftp.cdc.gov/pub/health_statistics/nchs/Publications/"
    "ICD10CM/2018/2018_I9gem.txt"
)

_USER_AGENT = "ResearchModule-ConditionReferenceFetch/1.0 (+research use)"

# ICD-10-CM chapter ranges keyed on the three-character category (FY2018+).
# Each tuple is (start_category, end_category, chapter_code, chapter_name).
# Comparison is lexical on the upper-cased three-character category, which is
# valid because ICD-10-CM categories are letter + two alphanumeric characters.
ICD10_CHAPTER_RANGES: tuple[tuple[str, str, str, str], ...] = (
    ("A00", "B99", "01", "Certain infectious and parasitic diseases"),
    ("C00", "D49", "02", "Neoplasms"),
    (
        "D50",
        "D89",
        "03",
        "Diseases of the blood and blood-forming organs and certain "
        "disorders involving the immune mechanism",
    ),
    ("E00", "E89", "04", "Endocrine, nutritional and metabolic diseases"),
    ("F01", "F99", "05", "Mental, behavioral and neurodevelopmental disorders"),
    ("G00", "G99", "06", "Diseases of the nervous system"),
    ("H00", "H59", "07", "Diseases of the eye and adnexa"),
    ("H60", "H95", "08", "Diseases of the ear and mastoid process"),
    ("I00", "I99", "09", "Diseases of the circulatory system"),
    ("J00", "J99", "10", "Diseases of the respiratory system"),
    ("K00", "K95", "11", "Diseases of the digestive system"),
    ("L00", "L99", "12", "Diseases of the skin and subcutaneous tissue"),
    (
        "M00",
        "M99",
        "13",
        "Diseases of the musculoskeletal system and connective tissue",
    ),
    ("N00", "N99", "14", "Diseases of the genitourinary system"),
    ("O00", "O9A", "15", "Pregnancy, childbirth and the puerperium"),
    ("P00", "P96", "16", "Certain conditions originating in the perinatal period"),
    (
        "Q00",
        "Q99",
        "17",
        "Congenital malformations, deformations and chromosomal abnormalities",
    ),
    (
        "R00",
        "R99",
        "18",
        "Symptoms, signs and abnormal clinical and laboratory findings, "
        "not elsewhere classified",
    ),
    (
        "S00",
        "T88",
        "19",
        "Injury, poisoning and certain other consequences of external causes",
    ),
    ("U00", "U85", "22", "Codes for special purposes"),
    ("V00", "Y99", "20", "External causes of morbidity"),
    (
        "Z00",
        "Z99",
        "21",
        "Factors influencing health status and contact with health services",
    ),
)

# ICD-9-CM numeric chapter ranges keyed on the three-digit category value.
ICD9_NUMERIC_CHAPTER_RANGES: tuple[tuple[int, int, str, str], ...] = (
    (1, 139, "01", "Infectious and parasitic diseases"),
    (140, 239, "02", "Neoplasms"),
    (
        240,
        279,
        "03",
        "Endocrine, nutritional and metabolic diseases, and immunity disorders",
    ),
    (280, 289, "04", "Diseases of the blood and blood-forming organs"),
    (290, 319, "05", "Mental disorders"),
    (320, 389, "06", "Diseases of the nervous system and sense organs"),
    (390, 459, "07", "Diseases of the circulatory system"),
    (460, 519, "08", "Diseases of the respiratory system"),
    (520, 579, "09", "Diseases of the digestive system"),
    (580, 629, "10", "Diseases of the genitourinary system"),
    (630, 679, "11", "Complications of pregnancy, childbirth, and the puerperium"),
    (680, 709, "12", "Diseases of the skin and subcutaneous tissue"),
    (
        710,
        739,
        "13",
        "Diseases of the musculoskeletal system and connective tissue",
    ),
    (740, 759, "14", "Congenital anomalies"),
    (760, 779, "15", "Certain conditions originating in the perinatal period"),
    (780, 799, "16", "Symptoms, signs, and ill-defined conditions"),
    (800, 999, "17", "Injury and poisoning"),
)

ICD9_E_CHAPTER = (
    "18",
    "Supplementary classification of external causes of injury and poisoning",
)
ICD9_V_CHAPTER = (
    "19",
    "Supplementary classification of factors influencing health status and "
    "contact with health services",
)


@dataclass(frozen=True)
class ConditionReferenceConfig:
    dataset_root: Path = DATASET_ROOT
    mapping_root: Path = MAPPING_ROOT
    reports_root: Path = REPORTS_ROOT
    cache_dir: Path | None = None
    skip_download: bool = False
    ccsr_url: str = CCSR_URL
    ccs_url: str = CCS_URL
    gem_url: str = GEM_I9_URL
    sources: dict[str, str] = field(default_factory=dict)

    @property
    def conditions_root(self) -> Path:
        return self.mapping_root / "conditions"

    @property
    def resolved_cache_dir(self) -> Path:
        return self.cache_dir or (self.conditions_root / "_reference_cache")

    @property
    def report_path(self) -> Path:
        return self.reports_root / "condition_reference_build_report.json"


def normalize_code_key(value: str) -> str:
    """Match ``pipeline.harmonize`` code-key normalization (strip, lower)."""

    cleaned = re.sub(r"[^A-Za-z0-9]+", "", (value or "").strip())
    return cleaned.lower()


def download_file(url: str, destination: Path, *, skip_download: bool) -> bool:
    """Download ``url`` to ``destination``; return True if a download ran."""

    if skip_download and destination.exists():
        logging.info("Reusing cached %s", destination.name)
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Downloading %s", url)
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})  # noqa: S310
    with urllib.request.urlopen(request, timeout=180) as response:  # noqa: S310
        with destination.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    return True


def _extract_first(archive_path: Path, extract_dir: Path, suffix_match: str) -> Path:
    """Extract the first archive member whose name matches ``suffix_match``."""

    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        members = [
            name
            for name in archive.namelist()
            if suffix_match.lower() in name.lower() and not name.endswith("/")
        ]
        if not members:
            raise FileNotFoundError(
                f"No member matching '{suffix_match}' in {archive_path.name}"
            )
        member = sorted(members)[0]
        archive.extract(member, extract_dir)
        return extract_dir / member


def parse_ccsr(csv_path: Path) -> list[dict[str, str]]:
    """Parse the horizontal DXCCSR CSV into default-category rows.

    Uses the default inpatient CCSR category, falling back to the first listed
    CCSR category when the default is blank. Single-quote text qualifiers used
    by AHRQ are stripped.
    """

    def clean(cell: str) -> str:
        return (cell or "").strip().strip("'").strip()

    rows: list[dict[str, str]] = []
    with csv_path.open(newline="", encoding="latin-1") as handle:
        reader = csv.reader(handle)
        next(reader, None)  # header
        for record in reader:
            if len(record) < 8:
                continue
            code = clean(record[0])
            default_ip = clean(record[2])
            default_ip_desc = clean(record[3])
            category_1 = clean(record[6])
            category_1_desc = clean(record[7])
            category = default_ip or category_1
            description = default_ip_desc or category_1_desc
            if not code or not category:
                continue
            rows.append(
                {
                    "icd_code": code,
                    "ccsr_category": category,
                    "ccsr_category_description": description,
                }
            )
    return rows


def parse_ccs(csv_path: Path) -> list[dict[str, str]]:
    """Parse the single-level ``$dxref`` CCS file (skips the leading NOTE)."""

    def clean(cell: str) -> str:
        return (cell or "").strip().strip("'").strip()

    lines = csv_path.read_text(encoding="latin-1").splitlines()
    reader = csv.reader(lines[1:])  # drop the NOTE line
    next(reader, None)  # header
    rows: list[dict[str, str]] = []
    for record in reader:
        if len(record) < 3:
            continue
        code = clean(record[0])
        category = clean(record[1])
        description = clean(record[2])
        if not code or not category or category == "0":
            continue
        rows.append(
            {
                "icd_code": code,
                "ccs_category": category,
                "ccs_category_description": description,
            }
        )
    return rows


def parse_gem(txt_path: Path) -> list[dict[str, str]]:
    """Parse the CDC I9 GEM (icd9, icd10, 5-flag string); skip no-map rows."""

    rows: list[dict[str, str]] = []
    for raw in txt_path.read_text(encoding="latin-1").splitlines():
        parts = raw.split()
        if len(parts) < 3:
            continue
        icd9, icd10, flags = parts[0], parts[1], parts[2]
        if len(flags) < 2:
            continue
        no_map = flags[1] == "1"
        if no_map or icd10.lower() == "nodx":
            continue
        approximate = "1" if flags[0] == "1" else "0"
        rows.append(
            {
                "icd9_code": icd9,
                "icd10_code": icd10,
                "approximate_flag": approximate,
            }
        )
    return rows


def icd10_chapter_for(category: str) -> tuple[str, str] | None:
    """Return (chapter_code, chapter_name) for a 3-char ICD-10-CM category."""

    cat = category.upper()
    if len(cat) < 3:
        return None
    for start, end, code, name in ICD10_CHAPTER_RANGES:
        if start <= cat <= end:
            return code, name
    return None


def icd9_chapter_for(category: str) -> tuple[str, str] | None:
    """Return (chapter_code, chapter_name) for a 3-char ICD-9-CM category."""

    cat = category.upper()
    if not cat:
        return None
    if cat.startswith("E"):
        return ICD9_E_CHAPTER
    if cat.startswith("V"):
        return ICD9_V_CHAPTER
    digits = cat[:3]
    if not digits.isdigit():
        return None
    value = int(digits)
    for start, end, code, name in ICD9_NUMERIC_CHAPTER_RANGES:
        if start <= value <= end:
            return code, name
    return None


def build_chapter_rows(
    ccsr_rows: list[dict[str, str]],
    ccs_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Derive category-to-chapter rows grounded in the authoritative code lists."""

    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    icd10_categories: set[str] = set()
    for row in ccsr_rows:
        key = normalize_code_key(row["icd_code"])
        if len(key) >= 3:
            icd10_categories.add(key[:3].upper())
    for category in sorted(icd10_categories):
        chapter = icd10_chapter_for(category)
        if chapter is None:
            continue
        marker = ("10", category)
        if marker in seen:
            continue
        seen.add(marker)
        rows.append(
            {
                "icd_version": "10",
                "category_code": category,
                "chapter_code": chapter[0],
                "chapter_name": chapter[1],
            }
        )

    icd9_categories: set[str] = set()
    for row in ccs_rows:
        key = normalize_code_key(row["icd_code"])
        if len(key) >= 3:
            icd9_categories.add(key[:3].upper())
    for category in sorted(icd9_categories):
        chapter = icd9_chapter_for(category)
        if chapter is None:
            continue
        marker = ("9", category)
        if marker in seen:
            continue
        seen.add(marker)
        rows.append(
            {
                "icd_version": "9",
                "category_code": category,
                "chapter_code": chapter[0],
                "chapter_name": chapter[1],
            }
        )
    return rows


def write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _spec_columns(name: str) -> list[str]:
    for spec in CONDITION_MAPPING_SPECS:
        if spec.name == name:
            return list(spec.required_columns)
    raise KeyError(f"No CONDITION_MAPPING_SPECS entry named '{name}'")


def _validate_columns(name: str, columns: list[str]) -> None:
    required = set(_spec_columns(name))
    missing = required.difference(columns)
    if missing:
        raise ValueError(
            f"{name}: written columns {columns} missing required {sorted(missing)}"
        )


def fetch_condition_reference_files(
    config: ConditionReferenceConfig = ConditionReferenceConfig(),
) -> dict[str, Any]:
    """Download authoritative sources and write condition mapping inputs."""

    output_root = config.conditions_root
    output_root.mkdir(parents=True, exist_ok=True)
    cache_dir = config.resolved_cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    downloads: dict[str, bool] = {}

    ccsr_zip = cache_dir / "DXCCSR.zip"
    downloads["ccsr"] = download_file(
        config.ccsr_url, ccsr_zip, skip_download=config.skip_download
    )
    ccsr_csv = _extract_first(ccsr_zip, cache_dir / "ccsr", "DXCCSR_v")

    ccs_zip = cache_dir / "Single_Level_CCS.zip"
    downloads["ccs"] = download_file(
        config.ccs_url, ccs_zip, skip_download=config.skip_download
    )
    ccs_csv = _extract_first(ccs_zip, cache_dir / "ccs", "dxref")

    gem_txt = cache_dir / "2018_I9gem.txt"
    downloads["gem"] = download_file(
        config.gem_url, gem_txt, skip_download=config.skip_download
    )

    ccsr_rows = parse_ccsr(ccsr_csv)
    ccs_rows = parse_ccs(ccs_csv)
    gem_rows = parse_gem(gem_txt)
    chapter_rows = build_chapter_rows(ccsr_rows, ccs_rows)

    icd10_ccsr_columns = _spec_columns("icd10_ccsr")
    icd9_ccs_columns = _spec_columns("icd9_ccs")
    gem_columns = _spec_columns("icd9_to_icd10_gem")
    chapter_columns = _spec_columns("icd_chapters")

    _validate_columns("icd10_ccsr", icd10_ccsr_columns)
    _validate_columns("icd9_ccs", icd9_ccs_columns)
    _validate_columns("icd9_to_icd10_gem", gem_columns)
    _validate_columns("icd_chapters", chapter_columns)

    ccsr_out = output_root / "icd10_ccsr.csv"
    ccs_out = output_root / "icd9_ccs.csv"
    gem_out = output_root / "icd9_to_icd10_gem.csv"
    chapters_out = output_root / "icd_chapters.csv"

    write_csv(ccsr_out, icd10_ccsr_columns, ccsr_rows)
    write_csv(ccs_out, icd9_ccs_columns, ccs_rows)
    write_csv(gem_out, gem_columns, gem_rows)
    write_csv(chapters_out, chapter_columns, chapter_rows)

    distinct_ccsr_categories = len({row["ccsr_category"] for row in ccsr_rows})
    distinct_ccs_categories = len({row["ccs_category"] for row in ccs_rows})
    approximate_gem = sum(1 for row in gem_rows if row["approximate_flag"] == "1")

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "condition_mapping_version": CONDITION_MAPPING_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "output_root": str(output_root),
        "sources": {
            "icd10_ccsr": config.ccsr_url,
            "icd9_ccs": config.ccs_url,
            "icd9_to_icd10_gem": config.gem_url,
            "icd_chapters": "Derived from published ICD-9/ICD-10-CM chapter ranges.",
        },
        "downloaded_this_run": downloads,
        "counts": {
            "icd10_ccsr_rows": len(ccsr_rows),
            "icd10_ccsr_distinct_categories": distinct_ccsr_categories,
            "icd9_ccs_rows": len(ccs_rows),
            "icd9_ccs_distinct_categories": distinct_ccs_categories,
            "icd9_to_icd10_gem_rows": len(gem_rows),
            "icd9_to_icd10_gem_approximate_rows": approximate_gem,
            "icd_chapters_rows": len(chapter_rows),
        },
        "output_files": {
            "icd10_ccsr.csv": str(ccsr_out),
            "icd9_ccs.csv": str(ccs_out),
            "icd9_to_icd10_gem.csv": str(gem_out),
            "icd_chapters.csv": str(chapters_out),
        },
        "data_safety": {
            "contains_patient_rows": False,
            "reporting_level": "public classification vocabulary and aggregate counts",
        },
        "notes": [
            "CCSR uses the default inpatient category (fallback to CCSR category 1).",
            "GEM no-map rows are excluded; approximate_flag preserves crosswalk "
            "confidence for lower-confidence bridge mappings.",
            "icd_chapters is a structural fallback derived from published chapter "
            "ranges, grounded in the authoritative CCSR/CCS code lists.",
            "eICU text and project condition groups (e.g. sepsis) still require "
            "clinical curation and are not produced by this fetch.",
            "Raw downloads are cached under an ignored directory; do not commit them.",
        ],
    }

    config.report_path.parent.mkdir(parents=True, exist_ok=True)
    config.report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    logging.info(
        "Wrote condition reference files to %s (CCSR=%d, CCS=%d, GEM=%d, chapters=%d)",
        output_root,
        len(ccsr_rows),
        len(ccs_rows),
        len(gem_rows),
        len(chapter_rows),
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download authoritative CCSR/CCS/GEM sources and write condition "
            "mapping reference files for harmonization."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--mapping-root", type=Path, default=MAPPING_ROOT)
    parser.add_argument("--reports-root", type=Path, default=REPORTS_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--ccsr-url", default=CCSR_URL)
    parser.add_argument("--ccs-url", default=CCS_URL)
    parser.add_argument("--gem-url", default=GEM_I9_URL)
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Reuse files already present in the cache directory.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))
    config = ConditionReferenceConfig(
        dataset_root=args.dataset_root,
        mapping_root=args.mapping_root,
        reports_root=args.reports_root,
        cache_dir=args.cache_dir,
        skip_download=args.skip_download,
        ccsr_url=args.ccsr_url,
        ccs_url=args.ccs_url,
        gem_url=args.gem_url,
    )
    report = fetch_condition_reference_files(config)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
