"""Download and prepare public medication reference files for mapping builds.

Sources (aggregate vocabulary only; no patient data):

- GAMENet mapping bundle (NDC→RxCUI dict, NDC→RxCUI→ATC4 table, DrugBank CID→ATC)
  https://github.com/sjy1203/GAMENet/tree/master/data
- RxNorm Current Prescribable Content (public domain, no UMLS license)
  https://www.nlm.nih.gov/research/umls/rxnorm/docs/prescribe.html

Outputs under ``$DATASET_ROOT/mappings/medications/``:

- ``ndc2RXCUI.txt`` — NDC to RxCUI (GAMENet + RxNorm RXNSAT NDC attributes)
- ``RXCUI2atc4.csv`` — RxCUI to ATC-4 with ingredient / RxNorm names
- ``drug-atc.csv`` — normalized drug name to ATC (for eICU name fallback)
- ``eicu_hicl_rxnorm_atc.csv`` — cohort eICU concepts mapped by exact normalized name

HICL/GTC codes in eICU are Medi-Span proprietary; this script maps by drug name and
writes concept-level rows for the builder's reviewed eICU reference slot. Unmapped
concepts remain reported explicitly.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import logging
import shutil
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.config import DATASET_ROOT, EXTRACTS_ROOT, MAPPING_ROOT  # noqa: E402

_builder_spec = importlib.util.spec_from_file_location(
    "build_medication_mappings",
    REPO_ROOT / "scripts" / "build_medication_mappings.py",
)
assert _builder_spec is not None and _builder_spec.loader is not None
_builder = importlib.util.module_from_spec(_builder_spec)
sys.modules[_builder_spec.name] = _builder
_builder_spec.loader.exec_module(_builder)
atc3_from_code = _builder.atc3_from_code
normalize_atc_code = _builder.normalize_atc_code
normalize_drug_name = _builder.normalize_drug_name
normalize_ndc = _builder.normalize_ndc
read_distinct_eicu_concepts = _builder.read_distinct_eicu_concepts

GAMENET_BASE = "https://raw.githubusercontent.com/sjy1203/GAMENet/master/data"
RXNORM_PRESCRIBE_URL = (
    "https://download.nlm.nih.gov/rxnorm/RxNorm_full_prescribe_current.zip"
)

SCHEMA_VERSION = "medication-reference-fetch-v1"


@dataclass(frozen=True)
class FetchConfig:
    dataset_root: Path = DATASET_ROOT
    mapping_root: Path = MAPPING_ROOT
    extracts_root: Path = EXTRACTS_ROOT
    work_dir: Path | None = None
    skip_download: bool = False

    @property
    def medication_mapping_root(self) -> Path:
        return self.mapping_root / "medications"


def download_file(url: str, destination: Path) -> None:
    """Download a URL to a local path."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Downloading %s", url)
    with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310
        with destination.open("wb") as handle:
            shutil.copyfileobj(response, handle)


def parse_gamenet_ndc_dict(path: Path) -> pd.DataFrame:
    """Parse GAMENet's Python-dict NDC→RxCUI file into a dataframe."""

    text = path.read_text(encoding="utf-8").strip()
    mapping = ast.literal_eval(text)
    rows = [(str(ndc), str(rxcui)) for ndc, rxcui in mapping.items()]
    frame = pd.DataFrame(rows, columns=["ndc", "rxcui"])
    frame["ndc_key"] = frame["ndc"].map(normalize_ndc)
    return frame[(frame["ndc_key"] != "") & (frame["rxcui"] != "")].drop_duplicates(
        subset=["ndc_key", "rxcui"]
    )


def load_rxnorm_prescribe(work_dir: Path, *, skip_download: bool) -> tuple[Path, Path]:
    """Download and extract RXNCONSO.RRF and RXNSAT.RRF from prescribable RxNorm."""

    zip_path = work_dir / "RxNorm_full_prescribe_current.zip"
    extract_dir = work_dir / "rxnorm_prescribe"
    rxnconso = extract_dir / "rrf" / "RXNCONSO.RRF"
    rxnsat = extract_dir / "rrf" / "RXNSAT.RRF"
    if rxnconso.exists() and rxnsat.exists():
        return rxnconso, rxnsat

    if not skip_download:
        download_file(RXNORM_PRESCRIBE_URL, zip_path)
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.namelist():
                if member.endswith(("RXNCONSO.RRF", "RXNSAT.RRF")):
                    archive.extract(member, extract_dir)
    if not rxnconso.exists() or not rxnsat.exists():
        raise FileNotFoundError(
            "RxNorm prescribable RXNCONSO/RXNSAT not found. "
            f"Expected under {extract_dir}"
        )
    return rxnconso, rxnsat


def _read_rxnorm_rrf(path: Path, columns: list[str]) -> pd.DataFrame:
    """Read a pipe-delimited RxNorm RRF file with stable column alignment."""

    frame = pd.read_csv(
        path,
        sep="|",
        header=None,
        names=columns,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8",
        index_col=False,
    )
    if len(frame.columns) > len(columns):
        frame = frame.iloc[:, : len(columns)]
    return frame.fillna("")


def read_rxnconso_concept_labels(path: Path) -> pd.DataFrame:
    """Map each RxCUI to ingredient and display names from RXNCONSO."""

    columns = [
        "RXCUI",
        "LAT",
        "TS",
        "LUI",
        "STT",
        "SUI",
        "ISPREF",
        "RXAUI",
        "SAUI",
        "SCUI",
        "SDUI",
        "SAB",
        "TTY",
        "CODE",
        "STR",
        "SRL",
        "SUPPRESS",
        "CVF",
    ]
    frame = _read_rxnorm_rrf(path, columns)
    frame = frame[frame["SAB"] == "RXNORM"]
    frame["rxcui"] = frame["RXCUI"].str.strip()
    frame["rxnorm_name"] = frame["STR"].str.strip()
    ingredient_priority = {"IN": 0, "PIN": 1, "MIN": 2}
    display_priority = {"SCD": 0, "SBD": 1, "GPCK": 2, "BPCK": 3, "IN": 4, "BN": 5}
    frame["ingredient_priority"] = frame["TTY"].map(ingredient_priority).fillna(99)
    frame["display_priority"] = frame["TTY"].map(display_priority).fillna(99)
    ingredients = (
        frame[frame["TTY"].isin(["IN", "PIN", "MIN"])]
        .sort_values(["rxcui", "ingredient_priority", "rxnorm_name"])
        .drop_duplicates(subset=["rxcui"], keep="first")
        .loc[:, ["rxcui", "rxnorm_name"]]
        .rename(columns={"rxnorm_name": "ingredient_name"})
    )
    display = (
        frame.sort_values(["rxcui", "display_priority", "rxnorm_name"])
        .drop_duplicates(subset=["rxcui"], keep="first")
        .loc[:, ["rxcui", "rxnorm_name"]]
    )
    labels = display.merge(ingredients, how="left", on="rxcui")
    labels["ingredient_name"] = labels["ingredient_name"].fillna("")
    labels.loc[labels["ingredient_name"] == "", "ingredient_name"] = labels[
        "rxnorm_name"
    ]
    return labels.reset_index(drop=True)


def read_rxnconso_names(path: Path) -> pd.DataFrame:
    """Load RxNorm concept names used for exact normalized drug-name matching."""

    columns = [
        "RXCUI",
        "LAT",
        "TS",
        "LUI",
        "STT",
        "SUI",
        "ISPREF",
        "RXAUI",
        "SAUI",
        "SCUI",
        "SDUI",
        "SAB",
        "TTY",
        "CODE",
        "STR",
        "SRL",
        "SUPPRESS",
        "CVF",
    ]
    frame = _read_rxnorm_rrf(path, columns)
    frame = frame[frame["SAB"] == "RXNORM"]
    frame = frame[frame["TTY"].isin(["IN", "PIN", "BN"])]
    frame["drug_name_key"] = frame["STR"].map(normalize_drug_name)
    frame["rxcui"] = frame["RXCUI"].str.strip()
    frame["rxnorm_name"] = frame["STR"].str.strip()
    frame["ingredient_name"] = frame["rxnorm_name"].where(
        frame["TTY"].isin(["IN", "PIN"]), ""
    )
    priority = {"IN": 0, "PIN": 1, "BN": 2}
    frame["priority"] = frame["TTY"].map(priority).fillna(9).astype(int)
    frame = frame[(frame["drug_name_key"] != "") & (frame["rxcui"] != "")]
    return (
        frame.sort_values(["drug_name_key", "priority", "rxcui", "rxnorm_name"])
        .drop_duplicates(subset=["drug_name_key"], keep="first")
        .loc[:, ["drug_name_key", "rxcui", "ingredient_name", "rxnorm_name", "TTY"]]
        .reset_index(drop=True)
    )


def read_rxnsat_ndc(path: Path) -> pd.DataFrame:
    """Load NDC attributes from RxNorm RXNSAT."""

    columns = [
        "RXCUI",
        "LUI",
        "SUI",
        "RXAUI",
        "STYPE",
        "CODE",
        "ATUI",
        "SAT",
        "ATN",
        "SAB",
        "ATV",
        "SUPPRESS",
        "CVF",
    ]
    frame = _read_rxnorm_rrf(path, columns)
    frame = frame[frame["ATN"] == "NDC"]
    frame = frame.rename(columns={"ATV": "ndc", "RXCUI": "rxcui"})
    frame["ndc_key"] = frame["ndc"].map(normalize_ndc)
    frame["rxcui"] = frame["rxcui"].str.strip()
    return frame[(frame["ndc_key"] != "") & (frame["rxcui"] != "")].drop_duplicates(
        subset=["ndc_key", "rxcui"]
    )


def build_rxcui_atc4(gamenet_atc_path: Path) -> pd.DataFrame:
    """Deduplicate GAMENet NDC/RxCUI/ATC4 rows to one row per RxCUI."""

    frame = pd.read_csv(gamenet_atc_path, dtype=str, keep_default_na=False)
    frame = frame.rename(columns={col: col.upper() for col in frame.columns})
    frame["rxcui"] = frame["RXCUI"].str.strip()
    frame["atc4_code"] = frame["ATC4"].map(normalize_atc_code)
    frame = frame[(frame["rxcui"] != "") & (frame["atc4_code"] != "")]
    return (
        frame.sort_values(["rxcui", "atc4_code"])
        .drop_duplicates(subset=["rxcui"], keep="first")
        .loc[:, ["rxcui", "atc4_code"]]
        .reset_index(drop=True)
    )


def enrich_rxcui_atc4(
    rxcui_atc4: pd.DataFrame, concept_labels: pd.DataFrame
) -> pd.DataFrame:
    """Attach ingredient / RxNorm names to RxCUI rows."""

    merged = rxcui_atc4.merge(concept_labels, how="left", on="rxcui")
    merged["ingredient_name"] = merged["ingredient_name"].fillna("")
    merged["rxnorm_name"] = merged["rxnorm_name"].fillna("")
    return merged


def build_drug_atc_table(rxcui_atc4: pd.DataFrame) -> pd.DataFrame:
    """Build drug-name→ATC rows for eICU name fallback."""

    rows: list[dict[str, str]] = []
    for record in rxcui_atc4.to_dict("records"):
        atc4 = normalize_atc_code(record.get("atc4_code", ""))
        atc3 = atc3_from_code(atc4)
        for name_column in ("ingredient_name", "rxnorm_name"):
            key = normalize_drug_name(record.get(name_column, ""))
            if key and atc4:
                rows.append(
                    {
                        "drug_name": key,
                        "atc_code": atc4,
                        "atc3": atc3,
                    }
                )
    if not rows:
        return pd.DataFrame(columns=["drug_name", "atc_code", "atc3"])
    frame = pd.DataFrame(rows).drop_duplicates(subset=["drug_name"], keep="first")
    return frame.sort_values("drug_name").reset_index(drop=True)


def build_eicu_reference(
    concepts: pd.DataFrame,
    rxnorm_names: pd.DataFrame,
    rxcui_atc4: pd.DataFrame,
    drug_atc: pd.DataFrame,
) -> pd.DataFrame:
    """Map cohort eICU concepts by exact normalized drug name."""

    atc_by_rxcui = rxcui_atc4.copy()
    atc_by_rxcui["atc_code"] = atc_by_rxcui["atc4_code"].map(atc3_from_code)
    atc_by_rxcui["atc_level"] = "3"
    atc_by_rxcui.loc[atc_by_rxcui["atc_code"] == "", "atc_code"] = atc_by_rxcui[
        "atc4_code"
    ]
    atc_by_rxcui.loc[
        (atc_by_rxcui["atc_level"] == "3") & (atc_by_rxcui["atc_code"].str.len() > 4),
        "atc_level",
    ] = "4"
    atc_by_name = drug_atc.rename(columns={"drug_name": "lookup_key"}).copy()
    atc_by_name["lookup_key"] = atc_by_name["lookup_key"].map(normalize_drug_name)
    atc_by_name["atc_level"] = atc_by_name["atc3"].map(
        lambda value: "3" if normalize_atc_code(value) else "4"
    )
    atc_by_name = atc_by_name[
        (atc_by_name["lookup_key"] != "") & (atc_by_name["atc_code"] != "")
    ]

    name_map = rxnorm_names.rename(columns={"drug_name_key": "lookup_key"})
    merged = concepts.merge(
        name_map, left_on="drug_name_key", right_on="lookup_key", how="left"
    )
    merged = merged.merge(
        atc_by_rxcui.loc[
            :, ["rxcui", "atc4_code", "atc_code", "atc_level"]
        ].drop_duplicates(subset=["rxcui"]),
        how="left",
        on="rxcui",
    )
    merged = merged.merge(
        atc_by_name.loc[:, ["lookup_key", "atc_code", "atc_level"]].rename(
            columns={
                "atc_code": "name_atc_code",
                "atc_level": "name_atc_level",
                "lookup_key": "drug_name_key",
            }
        ),
        how="left",
        on="drug_name_key",
    )
    merged["atc_code"] = (
        merged["atc_code"]
        .fillna("")
        .where(
            merged["atc_code"].fillna("") != "",
            merged["name_atc_code"].fillna(""),
        )
    )
    merged["atc_level"] = (
        merged["atc_level"]
        .fillna("")
        .where(
            merged["atc_level"].fillna("") != "",
            merged["name_atc_level"].fillna(""),
        )
    )
    merged["ingredient_name"] = merged["ingredient_name"].fillna("")
    merged["rxnorm_name"] = merged["rxnorm_name"].fillna("")
    merged["atc_code"] = merged["atc_code"].fillna("")
    merged["atc_level"] = merged["atc_level"].fillna("")
    merged.loc[merged["ingredient_name"] == "", "ingredient_name"] = merged[
        "rxnorm_name"
    ]

    output = pd.DataFrame(
        {
            "drughiclseqno": merged["drughiclseqno"],
            "gtc": merged["gtc"],
            "drug_name": merged["drug_name"],
            "rxcui": merged["rxcui"].fillna(""),
            "ingredient_name": merged["ingredient_name"],
            "rxnorm_name": merged["rxnorm_name"],
            "atc_code": merged["atc_code"].map(normalize_atc_code),
            "atc_level": merged["atc_level"],
        }
    )
    mapped = output[(output["rxcui"] != "") | (output["atc_code"] != "")]
    return mapped.sort_values(["drughiclseqno", "gtc", "drug_name"]).reset_index(
        drop=True
    )


def write_ndc_rxcui(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = frame.loc[:, ["ndc_key", "rxcui"]].rename(columns={"ndc_key": "ndc"})
    out = out.drop_duplicates().sort_values(["ndc", "rxcui"])
    out.to_csv(path, index=False)


def write_rxcui_atc4(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = frame.rename(columns={"atc4_code": "ATC4"})
    out.to_csv(
        path,
        index=False,
        columns=["rxcui", "ATC4", "ingredient_name", "rxnorm_name"],
    )


def write_drug_atc(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def write_eicu_reference(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def fetch_medication_reference_files(
    config: FetchConfig = FetchConfig(),
) -> dict[str, Any]:
    """Download public references and write mapping inputs."""

    output_root = config.medication_mapping_root
    output_root.mkdir(parents=True, exist_ok=True)

    work_dir = config.work_dir or (config.mapping_root / "_medication_ref_cache")
    work_dir.mkdir(parents=True, exist_ok=True)

    gamenet_ndc = work_dir / "ndc2rxnorm_mapping.txt"
    gamenet_atc = work_dir / "ndc2atc_level4.csv"
    gamenet_drug_atc = work_dir / "drug-atc.csv"

    gamenet_sources = {
        gamenet_ndc: f"{GAMENET_BASE}/ndc2rxnorm_mapping.txt",
        gamenet_atc: f"{GAMENET_BASE}/ndc2atc_level4.csv",
        gamenet_drug_atc: f"{GAMENET_BASE}/drug-atc.csv",
    }
    for path, url in gamenet_sources.items():
        if not config.skip_download or not path.exists():
            download_file(url, path)

    gamenet_ndc_frame = parse_gamenet_ndc_dict(gamenet_ndc)
    rxnconso_path, rxnsat_path = load_rxnorm_prescribe(
        work_dir, skip_download=config.skip_download
    )
    rxnorm_names = read_rxnconso_names(rxnconso_path)
    concept_labels = read_rxnconso_concept_labels(rxnconso_path)
    rxnorm_ndc = read_rxnsat_ndc(rxnsat_path)

    ndc_rxcui = pd.concat(
        [
            gamenet_ndc_frame.loc[:, ["ndc_key", "rxcui"]],
            rxnorm_ndc.loc[:, ["ndc_key", "rxcui"]],
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["ndc_key", "rxcui"])

    rxcui_atc4 = build_rxcui_atc4(gamenet_atc)
    rxcui_atc4 = enrich_rxcui_atc4(rxcui_atc4, concept_labels)
    drug_atc = build_drug_atc_table(rxcui_atc4)

    eicu_medication_path = config.extracts_root / "eicu_crd" / "medication.parquet"
    eicu_reference = pd.DataFrame(
        columns=[
            "drughiclseqno",
            "gtc",
            "drug_name",
            "rxcui",
            "ingredient_name",
            "rxnorm_name",
            "atc_code",
            "atc_level",
        ]
    )
    eicu_concept_count = 0
    if eicu_medication_path.exists():
        concepts = read_distinct_eicu_concepts(eicu_medication_path)
        eicu_concept_count = len(concepts)
        eicu_reference = build_eicu_reference(
            concepts, rxnorm_names, rxcui_atc4, drug_atc
        )

    ndc_out = output_root / "ndc2RXCUI.txt"
    atc_out = output_root / "RXCUI2atc4.csv"
    drug_atc_out = output_root / "drug-atc.csv"
    eicu_out = output_root / "eicu_hicl_rxnorm_atc.csv"
    provenance_out = output_root / "reference_provenance.json"

    write_ndc_rxcui(ndc_out, ndc_rxcui)
    write_rxcui_atc4(atc_out, rxcui_atc4)
    write_drug_atc(drug_atc_out, drug_atc)
    write_eicu_reference(eicu_out, eicu_reference)

    if gamenet_drug_atc.exists():
        shutil.copy2(
            gamenet_drug_atc,
            output_root / "gamenet_drugbank_cid_atc_source.csv",
        )

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "output_root": str(output_root),
        "sources": {
            "gamenet": GAMENET_BASE,
            "rxnorm_prescribe": RXNORM_PRESCRIBE_URL,
        },
        "counts": {
            "ndc_rxcui_rows": int(len(ndc_rxcui)),
            "gamenet_ndc_rows": int(len(gamenet_ndc_frame)),
            "rxnorm_ndc_rows": int(len(rxnorm_ndc)),
            "rxcui_atc4_rows": int(len(rxcui_atc4)),
            "drug_atc_name_rows": int(len(drug_atc)),
            "rxnorm_name_rows": int(len(rxnorm_names)),
            "eicu_distinct_concepts": eicu_concept_count,
            "eicu_reference_mapped_rows": int(len(eicu_reference)),
        },
        "output_files": {
            "ndc2RXCUI.txt": str(ndc_out),
            "RXCUI2atc4.csv": str(atc_out),
            "drug-atc.csv": str(drug_atc_out),
            "eicu_hicl_rxnorm_atc.csv": str(eicu_out),
        },
        "notes": [
            "GAMENet ndc2rxnorm_mapping.txt is converted from a Python dict to CSV.",
            "eICU HICL/GTC codes are not publicly crosswalked; name-based mapping only.",
            "Review high-impact unmapped concepts before pooled MIMIC/eICU training.",
        ],
    }
    provenance_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    logging.info("Wrote medication reference files to %s", output_root)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download public medication reference files for harmonization."
    )
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--mapping-root", type=Path, default=MAPPING_ROOT)
    parser.add_argument("--extracts-root", type=Path, default=EXTRACTS_ROOT)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Reuse files already present in --work-dir.",
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
    config = FetchConfig(
        dataset_root=args.dataset_root,
        mapping_root=args.mapping_root,
        extracts_root=args.extracts_root,
        work_dir=args.work_dir,
        skip_download=args.skip_download,
    )
    report = fetch_medication_reference_files(config)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
