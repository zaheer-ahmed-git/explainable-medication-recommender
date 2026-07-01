"""Build local medication mapping resources for harmonization.

The harmonization CLI expects merged, local CSVs under
``$DATASET_ROOT/mappings/medications``. This script builds those files from the
cohort-filtered medication extracts and any reviewed RxNorm/ATC reference files
present in the mapping folder.
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
from typing import Any, Sequence

import duckdb
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.config import (  # noqa: E402
    DATASET_ROOT,
    EXTRACTS_ROOT,
    MAPPING_ROOT,
    REPORTS_ROOT,
)


SCHEMA_VERSION = "medication-mapping-build-report-v1"
MIMIC_OUTPUT_COLUMNS = (
    "ndc",
    "rxcui",
    "ingredient_name",
    "rxnorm_name",
    "atc_code",
    "atc_level",
)
EICU_OUTPUT_COLUMNS = (
    "drughiclseqno",
    "gtc",
    "drug_name",
    "rxcui",
    "ingredient_name",
    "rxnorm_name",
    "atc_code",
    "atc_level",
)

NDC_RXCUI_NAMES = ("ndc2RXCUI.txt", "ndc2rxcui.txt", "ndc_rxnorm.csv")
RXCUI_ATC4_NAMES = ("RXCUI2atc4.csv", "rxcui2atc4.csv", "rxcui_atc4.csv")
DRUG_ATC_NAMES = ("drug-atc.csv", "drug_atc.csv", "drugbank_atc.csv")
EICU_REFERENCE_NAMES = (
    "eicu_drug_rxnorm_atc.csv",
    "eicu_hicl_rxnorm_atc.csv",
    "eicu_gtc_rxnorm_atc.csv",
)

NDC_COLUMNS = ("ndc", "ndc_code", "ndc11", "ndcnum")
RXCUI_COLUMNS = ("rxcui", "rx_cui", "rxnorm_cui", "rxnorm_id", "rxnorm")
ATC_COLUMNS = ("atc_code", "atc", "atc4", "atc4_code", "atc_level4")
ATC3_COLUMNS = ("atc3", "atc3_code", "atc_level3", "atc_3")
INGREDIENT_COLUMNS = (
    "ingredient_name",
    "ingredient",
    "rxnorm_ingredient",
    "rxnorm_ingredient_name",
)
RXNORM_NAME_COLUMNS = (
    "rxnorm_name",
    "rxnorm",
    "rxnorm_concept_name",
    "concept_name",
    "drug_name",
    "drugname",
    "name",
)
HICL_COLUMNS = ("drughiclseqno", "drug_hicl_seqno", "hicl", "hiclseqno")
GTC_COLUMNS = ("gtc", "drug_gtc", "generic_therapeutic_class")
DRUG_NAME_COLUMNS = ("drug_name", "drugname", "name", "rxnorm_name")

FORM_ROUTE_TOKENS = {
    "po",
    "iv",
    "im",
    "sc",
    "sq",
    "subq",
    "oral",
    "intravenous",
    "injection",
    "inj",
    "tablet",
    "tab",
    "tabs",
    "capsule",
    "cap",
    "caps",
    "solution",
    "soln",
    "suspension",
    "susp",
    "syrup",
    "elixir",
    "cream",
    "ointment",
    "patch",
    "drops",
    "drop",
    "infusion",
    "premix",
    "powder",
}


@dataclass(frozen=True)
class MedicationMappingBuildConfig:
    """Input and output locations for medication mapping construction."""

    dataset_root: Path = DATASET_ROOT
    extracts_root: Path = EXTRACTS_ROOT
    mapping_root: Path = MAPPING_ROOT
    report_path: Path = REPORTS_ROOT / "medication_mapping_build_report.json"
    mimic_prescriptions_path: Path | None = None
    eicu_medication_path: Path | None = None
    ndc_rxcui_path: Path | None = None
    rxcui_atc4_path: Path | None = None
    drug_atc_path: Path | None = None
    eicu_reference_path: Path | None = None

    @property
    def medication_mapping_root(self) -> Path:
        return self.mapping_root / "medications"

    @property
    def mimic_output_path(self) -> Path:
        return self.medication_mapping_root / "mimic_ndc_rxnorm_atc.csv"

    @property
    def eicu_output_path(self) -> Path:
        return self.medication_mapping_root / "eicu_drug_rxnorm_atc.csv"

    @property
    def resolved_mimic_prescriptions_path(self) -> Path:
        return (
            self.mimic_prescriptions_path
            or self.extracts_root / "mimiciv" / "prescriptions.parquet"
        )

    @property
    def resolved_eicu_medication_path(self) -> Path:
        return (
            self.eicu_medication_path
            or self.extracts_root / "eicu_crd" / "medication.parquet"
        )


def sql_string(value: str | Path) -> str:
    """Return a DuckDB SQL string literal."""

    return "'" + str(value).replace("'", "''") + "'"


def clean_string(value: object) -> str:
    """Return a stable empty-string-aware representation for mapping keys."""

    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def normalize_column_name(value: object) -> str:
    """Normalize source column names for flexible reference-file loading."""

    text = clean_string(value).lower()
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", text)).strip("_")


def normalize_ndc(value: object) -> str:
    """Normalize NDC only for matching while preserving source NDC for output."""

    text = clean_string(value)
    if not text:
        return ""
    if re.fullmatch(r"\d+\.0+", text):
        text = text.split(".", maxsplit=1)[0]
    if re.fullmatch(r"[0-9\-\s.]+", text):
        return re.sub(r"[^0-9]", "", text)
    return text


def normalize_code(value: object) -> str:
    """Normalize coded eICU/RxNorm keys without changing semantic content."""

    return clean_string(value).lower()


def normalize_atc_code(value: object) -> str:
    """Normalize ATC codes to uppercase compact strings."""

    return re.sub(r"[^A-Za-z0-9]", "", clean_string(value)).upper()


def atc3_from_code(value: object) -> str:
    """Roll an ATC code up to level 3 using the ATC hierarchy prefix."""

    code = normalize_atc_code(value)
    return code[:4] if len(code) >= 4 else ""


def normalize_drug_name(value: object) -> str:
    """Conservative exact-match key for eICU drug-name fallback mapping."""

    text = clean_string(value).lower()
    if not text:
        return ""
    text = re.sub(r"\([^)]*\d+[^)]*\)", " ", text)
    text = re.sub(
        r"\b\d+(?:\.\d+)?\s*(?:mcg|microgram|micrograms|mg|gm|g|kg|ml|l|meq|iu|units?|%)\b",
        " ",
        text,
    )
    text = re.sub(
        r"\b(?:q\d+h|q\d+hr|q\d+hrs|qday|daily|bid|tid|qid|prn|once|twice)\b",
        " ",
        text,
    )
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = [token for token in text.split() if token not in FORM_ROUTE_TOKENS]
    return " ".join(tokens)


def existing_path_or_none(path: Path | None) -> Path | None:
    """Return the path if it points to an existing regular file."""

    if path and path.exists() and path.is_file():
        return path
    return None


def find_reference_file(
    *,
    explicit_path: Path | None,
    search_roots: Sequence[Path],
    candidate_names: Sequence[str],
    exclude_paths: Sequence[Path] = (),
) -> Path | None:
    """Find a reference file by explicit path or common filenames."""

    explicit = existing_path_or_none(explicit_path)
    if explicit is not None:
        return explicit

    excluded = {path.resolve() for path in exclude_paths if path.exists()}
    wanted = {name.lower() for name in candidate_names}
    for root in search_roots:
        if not root.exists():
            continue
        direct = [path for path in sorted(root.iterdir()) if path.is_file()]
        nested = [
            path
            for path in sorted(root.rglob("*"))
            if path.is_file() and path.parent != root
        ]
        for path in [*direct, *nested]:
            if path.resolve() in excluded:
                continue
            if path.name.lower() in wanted:
                return path
    return None


def read_flexible_table(path: Path, fallback_columns: Sequence[str]) -> pd.DataFrame:
    """Read a small/medium reference CSV/TXT with flexible delimiters."""

    read_kwargs: dict[str, Any] = {
        "dtype": str,
        "keep_default_na": False,
        "encoding": "utf-8",
    }
    try:
        frame = pd.read_csv(path, sep=None, engine="python", **read_kwargs)
    except UnicodeDecodeError:
        frame = pd.read_csv(
            path,
            sep=None,
            engine="python",
            encoding="latin-1",
            dtype=str,
            keep_default_na=False,
        )
    if len(frame.columns) <= 1:
        for sep in ("\t", ",", "|", r"\s+"):
            frame = pd.read_csv(path, sep=sep, engine="python", **read_kwargs)
            if len(frame.columns) > 1:
                break

    normalized_columns = [normalize_column_name(column) for column in frame.columns]
    if not set(normalized_columns).intersection(
        set(NDC_COLUMNS + RXCUI_COLUMNS + ATC_COLUMNS + HICL_COLUMNS + GTC_COLUMNS)
    ):
        frame = pd.read_csv(
            path,
            sep=None,
            engine="python",
            header=None,
            dtype=str,
            keep_default_na=False,
            encoding="utf-8",
        )
        frame = frame.iloc[:, : len(fallback_columns)]
        frame.columns = list(fallback_columns[: len(frame.columns)])
    else:
        frame.columns = normalized_columns

    return frame.fillna("").map(clean_string)


def find_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    """Find the first candidate column present in a reference table."""

    normalized = {normalize_column_name(column): column for column in frame.columns}
    for candidate in candidates:
        column = normalized.get(normalize_column_name(candidate))
        if column is not None:
            return column
    return None


def empty_frame(columns: Sequence[str]) -> pd.DataFrame:
    """Return an empty string-typed frame with named columns."""

    return pd.DataFrame(columns=list(columns), dtype=str)


def load_ndc_rxcui(path: Path | None) -> pd.DataFrame:
    """Load NDC-to-RxCUI mappings if available."""

    if path is None:
        return empty_frame(("ndc_key", "rxcui"))
    frame = read_flexible_table(path, ("ndc", "rxcui"))
    ndc_column = find_column(frame, NDC_COLUMNS)
    rxcui_column = find_column(frame, RXCUI_COLUMNS)
    if ndc_column is None or rxcui_column is None:
        logging.warning("Could not identify NDC/RxCUI columns in %s", path)
        return empty_frame(("ndc_key", "rxcui"))
    result = pd.DataFrame(
        {
            "ndc_key": frame[ndc_column].map(normalize_ndc),
            "rxcui": frame[rxcui_column].map(clean_string),
        }
    )
    return (
        result[(result["ndc_key"] != "") & (result["rxcui"] != "")]
        .drop_duplicates()
        .sort_values(["ndc_key", "rxcui"])
        .reset_index(drop=True)
    )


def load_rxcui_atc4(path: Path | None) -> pd.DataFrame:
    """Load RxCUI-to-ATC4/name mappings if available."""

    if path is None:
        return empty_frame(
            ("rxcui", "ingredient_name", "rxnorm_name", "atc4_code", "atc3_code")
        )
    frame = read_flexible_table(
        path,
        ("rxcui", "atc_code", "ingredient_name", "rxnorm_name"),
    )
    rxcui_column = find_column(frame, RXCUI_COLUMNS)
    atc_column = find_column(frame, ATC_COLUMNS)
    ingredient_column = find_column(frame, INGREDIENT_COLUMNS)
    rxnorm_name_column = find_column(frame, RXNORM_NAME_COLUMNS)
    if rxcui_column is None or atc_column is None:
        logging.warning("Could not identify RxCUI/ATC columns in %s", path)
        return empty_frame(
            ("rxcui", "ingredient_name", "rxnorm_name", "atc4_code", "atc3_code")
        )
    result = pd.DataFrame(
        {
            "rxcui": frame[rxcui_column].map(clean_string),
            "ingredient_name": (
                frame[ingredient_column].map(clean_string)
                if ingredient_column is not None
                else ""
            ),
            "rxnorm_name": (
                frame[rxnorm_name_column].map(clean_string)
                if rxnorm_name_column is not None
                else ""
            ),
            "atc4_code": frame[atc_column].map(normalize_atc_code),
        }
    )
    result["atc3_code"] = result["atc4_code"].map(atc3_from_code)
    return (
        result[(result["rxcui"] != "") & (result["atc4_code"] != "")]
        .drop_duplicates()
        .sort_values(["rxcui", "atc4_code"])
        .reset_index(drop=True)
    )


def load_drug_atc(path: Path | None) -> pd.DataFrame:
    """Load drug-name-to-ATC mappings if available."""

    if path is None:
        return empty_frame(("drug_name_key", "atc4_code", "atc3_code"))
    frame = read_flexible_table(path, ("drug_name", "atc_code"))
    name_column = find_column(frame, DRUG_NAME_COLUMNS)
    atc_column = find_column(frame, ATC_COLUMNS)
    atc3_column = find_column(frame, ATC3_COLUMNS)
    if name_column is None or atc_column is None:
        logging.warning("Could not identify drug-name/ATC columns in %s", path)
        return empty_frame(("drug_name_key", "atc4_code", "atc3_code"))
    result = pd.DataFrame(
        {
            "drug_name_key": frame[name_column].map(normalize_drug_name),
            "atc4_code": frame[atc_column].map(normalize_atc_code),
        }
    )
    result["atc3_code"] = (
        frame[atc3_column].map(normalize_atc_code)
        if atc3_column is not None
        else result["atc4_code"].map(atc3_from_code)
    )
    return (
        result[(result["drug_name_key"] != "") & (result["atc4_code"] != "")]
        .drop_duplicates()
        .sort_values(["drug_name_key", "atc4_code"])
        .reset_index(drop=True)
    )


def load_existing_mimic_mapping(path: Path) -> pd.DataFrame:
    """Load an existing MIMIC output map to preserve reviewed rows on rebuild."""

    if not path.exists():
        return empty_frame(("ndc_key", *MIMIC_OUTPUT_COLUMNS))
    frame = pd.read_csv(path, dtype=str, keep_default_na=False).fillna("")
    missing = [column for column in MIMIC_OUTPUT_COLUMNS if column not in frame.columns]
    if missing:
        logging.warning("Ignoring existing MIMIC map with missing columns: %s", missing)
        return empty_frame(("ndc_key", *MIMIC_OUTPUT_COLUMNS))
    frame = frame.loc[:, list(MIMIC_OUTPUT_COLUMNS)].map(clean_string)
    frame["ndc_key"] = frame["ndc"].map(normalize_ndc)
    return frame


def load_eicu_reference(path: Path | None) -> pd.DataFrame:
    """Load reviewed eICU code/name-to-RxNorm/ATC mappings if available."""

    if path is None:
        return empty_frame((*EICU_OUTPUT_COLUMNS, "drug_name_key"))
    frame = read_flexible_table(path, EICU_OUTPUT_COLUMNS)
    columns = {
        "drughiclseqno": find_column(frame, HICL_COLUMNS),
        "gtc": find_column(frame, GTC_COLUMNS),
        "drug_name": find_column(frame, DRUG_NAME_COLUMNS),
        "rxcui": find_column(frame, RXCUI_COLUMNS),
        "ingredient_name": find_column(frame, INGREDIENT_COLUMNS),
        "rxnorm_name": find_column(frame, RXNORM_NAME_COLUMNS),
        "atc_code": find_column(frame, ATC_COLUMNS),
        "atc_level": find_column(frame, ("atc_level", "level")),
    }
    if columns["rxcui"] is None and columns["atc_code"] is None:
        logging.warning("Ignoring eICU reference with no RxCUI/ATC columns: %s", path)
        return empty_frame((*EICU_OUTPUT_COLUMNS, "drug_name_key"))
    result = pd.DataFrame()
    for column in EICU_OUTPUT_COLUMNS:
        source_column = columns[column]
        result[column] = (
            frame[source_column].map(clean_string) if source_column is not None else ""
        )
    result["atc_code"] = result["atc_code"].map(normalize_atc_code)
    result["drug_name_key"] = result["drug_name"].map(normalize_drug_name)
    return result


def read_distinct_mimic_ndcs(path: Path) -> tuple[pd.DataFrame, int]:
    """Read distinct nonblank NDCs and aggregate null-row count from extract."""

    if not path.exists():
        raise FileNotFoundError(f"MIMIC prescription extract not found: {path}")
    with duckdb.connect(database=":memory:") as connection:
        rows = connection.execute(
            f"""
            SELECT DISTINCT TRIM(CAST(ndc AS VARCHAR)) AS ndc
            FROM read_parquet({sql_string(path)})
            WHERE NULLIF(TRIM(CAST(ndc AS VARCHAR)), '') IS NOT NULL
            ORDER BY ndc
            """
        ).fetchdf()
        null_rows = connection.execute(
            f"""
            SELECT COUNT(*) AS null_rows
            FROM read_parquet({sql_string(path)})
            WHERE NULLIF(TRIM(CAST(ndc AS VARCHAR)), '') IS NULL
            """
        ).fetchone()[0]
    rows = rows.fillna("").map(clean_string)
    rows["ndc_key"] = rows["ndc"].map(normalize_ndc)
    return rows, int(null_rows)


def read_distinct_eicu_concepts(path: Path) -> pd.DataFrame:
    """Read distinct eICU medication concepts from the cohort extract."""

    if not path.exists():
        raise FileNotFoundError(f"eICU medication extract not found: {path}")
    with duckdb.connect(database=":memory:") as connection:
        rows = connection.execute(
            f"""
            SELECT DISTINCT
                COALESCE(NULLIF(TRIM(CAST(drughiclseqno AS VARCHAR)), ''), '') AS drughiclseqno,
                COALESCE(NULLIF(TRIM(CAST(gtc AS VARCHAR)), ''), '') AS gtc,
                COALESCE(NULLIF(TRIM(CAST(drugname AS VARCHAR)), ''), '') AS drug_name
            FROM read_parquet({sql_string(path)})
            WHERE
                NULLIF(TRIM(CAST(drughiclseqno AS VARCHAR)), '') IS NOT NULL
                OR NULLIF(TRIM(CAST(gtc AS VARCHAR)), '') IS NOT NULL
                OR NULLIF(TRIM(CAST(drugname AS VARCHAR)), '') IS NOT NULL
            ORDER BY drughiclseqno, gtc, drug_name
            """
        ).fetchdf()
    rows = rows.fillna("").map(clean_string)
    rows["hicl_key"] = rows["drughiclseqno"].map(normalize_code)
    rows["gtc_key"] = rows["gtc"].map(normalize_code)
    rows["drug_name_key"] = rows["drug_name"].map(normalize_drug_name)
    return rows


def first_mapping_per_key(
    frame: pd.DataFrame,
    *,
    key: str,
    output_columns: Sequence[str],
) -> pd.DataFrame:
    """Return one deterministic mapped row per nonblank key."""

    if frame.empty or key not in frame.columns:
        return empty_frame((key, *output_columns))
    mapped = frame[
        (frame[key] != "") & ((frame["rxcui"] != "") | (frame["atc_code"] != ""))
    ].copy()
    if mapped.empty:
        return empty_frame((key, *output_columns))
    mapped["atc_code"] = mapped["atc_code"].map(normalize_atc_code)
    return (
        mapped.sort_values(
            [key, "rxcui", "atc_level", "atc_code", "ingredient_name", "rxnorm_name"]
        )
        .drop_duplicates(subset=[key], keep="first")
        .loc[:, [key, *output_columns]]
        .reset_index(drop=True)
    )


def choose_mimic_rows(candidates: pd.DataFrame) -> pd.DataFrame:
    """Collapse candidate mappings to one row per source NDC."""

    if candidates.empty:
        return empty_frame(MIMIC_OUTPUT_COLUMNS)
    frame = candidates.copy().fillna("").map(clean_string)
    for column in MIMIC_OUTPUT_COLUMNS:
        if column not in frame:
            frame[column] = ""
    frame["has_rxcui"] = frame["rxcui"] != ""
    frame["has_atc3"] = frame["atc_level"] == "3"
    frame["has_atc"] = frame["atc_code"] != ""
    chosen = (
        frame.sort_values(
            [
                "ndc",
                "has_rxcui",
                "has_atc3",
                "has_atc",
                "rxcui",
                "atc_code",
                "ingredient_name",
                "rxnorm_name",
            ],
            ascending=[True, False, False, False, True, True, True, True],
        )
        .drop_duplicates(subset=["ndc"], keep="first")
        .loc[:, list(MIMIC_OUTPUT_COLUMNS)]
        .reset_index(drop=True)
    )
    return chosen


def build_mimic_mapping(
    config: MedicationMappingBuildConfig,
    *,
    ndc_rxcui: pd.DataFrame,
    rxcui_atc4: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the MIMIC NDC-to-RxNorm/ATC output mapping."""

    source_ndcs, null_ndc_rows = read_distinct_mimic_ndcs(
        config.resolved_mimic_prescriptions_path
    )
    existing = load_existing_mimic_mapping(config.mimic_output_path)

    candidates = source_ndcs.merge(ndc_rxcui, how="left", on="ndc_key")
    candidates = candidates.merge(rxcui_atc4, how="left", on="rxcui")
    candidates["atc_code"] = candidates["atc3_code"].where(
        candidates["atc3_code"].fillna("") != "",
        candidates["atc4_code"].fillna(""),
    )
    candidates["atc_level"] = candidates["atc3_code"].map(
        lambda value: "3" if clean_string(value) else ""
    )
    candidates.loc[
        (candidates["atc_level"] == "") & (candidates["atc4_code"].fillna("") != ""),
        "atc_level",
    ] = "4"
    for column in ("ingredient_name", "rxnorm_name"):
        if column not in candidates:
            candidates[column] = ""

    if not existing.empty:
        existing_subset = existing.loc[:, ["ndc_key", *MIMIC_OUTPUT_COLUMNS]].rename(
            columns={
                "rxcui": "existing_rxcui",
                "ingredient_name": "existing_ingredient_name",
                "rxnorm_name": "existing_rxnorm_name",
                "atc_code": "existing_atc_code",
                "atc_level": "existing_atc_level",
                "ndc": "existing_ndc",
            }
        )
        candidates = candidates.merge(existing_subset, how="left", on="ndc_key")
        for column in (
            "rxcui",
            "ingredient_name",
            "rxnorm_name",
            "atc_code",
            "atc_level",
        ):
            existing_column = f"existing_{column}"
            candidates[column] = candidates[column].where(
                candidates[column].fillna("") != "",
                candidates[existing_column].fillna(""),
            )

    output = choose_mimic_rows(candidates)
    output = output.sort_values("ndc").reset_index(drop=True)

    distinct_count = len(output)
    rx_count = int((output["rxcui"] != "").sum())
    atc_count = int((output["atc_code"] != "").sum())
    atc3_count = int((output["atc_level"] == "3").sum())
    unmapped = output[(output["rxcui"] == "") & (output["atc_code"] == "")]
    report = {
        "distinct_ndc_count": distinct_count,
        "null_or_blank_ndc_source_row_count": null_ndc_rows,
        "rxcui_mapped_count": rx_count,
        "rxcui_mapped_percent": percentage(rx_count, distinct_count),
        "atc_mapped_count": atc_count,
        "atc_mapped_percent": percentage(atc_count, distinct_count),
        "atc3_mapped_count": atc3_count,
        "atc3_mapped_percent": percentage(atc3_count, distinct_count),
        "unmapped_count": int(len(unmapped)),
        "unmapped_percent": percentage(len(unmapped), distinct_count),
    }
    return output, report


def rxnorm_name_reference(rxcui_atc4: pd.DataFrame) -> pd.DataFrame:
    """Build exact normalized-name mappings from RxNorm/ATC data."""

    if rxcui_atc4.empty:
        return empty_frame(
            (
                "drug_name_key",
                "rxcui",
                "ingredient_name",
                "rxnorm_name",
                "atc_code",
                "atc_level",
            )
        )
    rows: list[dict[str, str]] = []
    for record in rxcui_atc4.to_dict("records"):
        for name_column in ("ingredient_name", "rxnorm_name"):
            key = normalize_drug_name(record.get(name_column, ""))
            if key:
                rows.append(
                    {
                        "drug_name_key": key,
                        "rxcui": clean_string(record.get("rxcui", "")),
                        "ingredient_name": clean_string(
                            record.get("ingredient_name", "")
                        ),
                        "rxnorm_name": clean_string(record.get("rxnorm_name", "")),
                        "atc_code": atc3_from_code(record.get("atc4_code", "")),
                        "atc_level": "3",
                    }
                )
    if not rows:
        return empty_frame(
            (
                "drug_name_key",
                "rxcui",
                "ingredient_name",
                "rxnorm_name",
                "atc_code",
                "atc_level",
            )
        )
    return (
        pd.DataFrame(rows)
        .drop_duplicates()
        .sort_values(["drug_name_key", "rxcui", "atc_code"])
        .drop_duplicates(subset=["drug_name_key"], keep="first")
        .reset_index(drop=True)
    )


def drug_atc_name_reference(drug_atc: pd.DataFrame) -> pd.DataFrame:
    """Build exact normalized-name ATC fallback mappings."""

    if drug_atc.empty:
        return empty_frame(
            (
                "drug_name_key",
                "rxcui",
                "ingredient_name",
                "rxnorm_name",
                "atc_code",
                "atc_level",
            )
        )
    rows = drug_atc.copy()
    rows["rxcui"] = ""
    rows["ingredient_name"] = ""
    rows["rxnorm_name"] = ""
    rows["atc_code"] = rows["atc3_code"].where(
        rows["atc3_code"] != "", rows["atc4_code"]
    )
    rows["atc_level"] = rows["atc3_code"].map(lambda value: "3" if value else "4")
    return (
        rows.loc[
            :,
            [
                "drug_name_key",
                "rxcui",
                "ingredient_name",
                "rxnorm_name",
                "atc_code",
                "atc_level",
            ],
        ]
        .drop_duplicates()
        .sort_values(["drug_name_key", "atc_code"])
        .drop_duplicates(subset=["drug_name_key"], keep="first")
        .reset_index(drop=True)
    )


def build_eicu_mapping(
    config: MedicationMappingBuildConfig,
    *,
    eicu_reference: pd.DataFrame,
    rxcui_atc4: pd.DataFrame,
    drug_atc: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the eICU code/name-to-RxNorm/ATC output mapping."""

    concepts = read_distinct_eicu_concepts(config.resolved_eicu_medication_path)
    name_reference = pd.concat(
        [rxnorm_name_reference(rxcui_atc4), drug_atc_name_reference(drug_atc)],
        ignore_index=True,
    )
    if not name_reference.empty:
        name_reference = (
            name_reference.sort_values(["drug_name_key", "rxcui", "atc_code"])
            .drop_duplicates(subset=["drug_name_key"], keep="first")
            .reset_index(drop=True)
        )

    reference = eicu_reference.copy()
    if config.eicu_output_path.exists():
        reference = pd.concat(
            [reference, load_eicu_reference(config.eicu_output_path)],
            ignore_index=True,
        )
    if not reference.empty:
        reference["hicl_key"] = reference["drughiclseqno"].map(normalize_code)
        reference["gtc_key"] = reference["gtc"].map(normalize_code)
        reference["drug_name_key"] = reference["drug_name"].map(normalize_drug_name)

    by_hicl = first_mapping_per_key(
        reference,
        key="hicl_key",
        output_columns=(
            "rxcui",
            "ingredient_name",
            "rxnorm_name",
            "atc_code",
            "atc_level",
        ),
    )
    by_gtc = first_mapping_per_key(
        reference,
        key="gtc_key",
        output_columns=(
            "rxcui",
            "ingredient_name",
            "rxnorm_name",
            "atc_code",
            "atc_level",
        ),
    )
    by_existing_name = first_mapping_per_key(
        reference,
        key="drug_name_key",
        output_columns=(
            "rxcui",
            "ingredient_name",
            "rxnorm_name",
            "atc_code",
            "atc_level",
        ),
    )
    by_name = pd.concat([by_existing_name, name_reference], ignore_index=True)
    if not by_name.empty:
        by_name = (
            by_name[
                (by_name["drug_name_key"] != "")
                & ((by_name["rxcui"] != "") | (by_name["atc_code"] != ""))
            ]
            .sort_values(["drug_name_key", "rxcui", "atc_code"])
            .drop_duplicates(subset=["drug_name_key"], keep="first")
            .reset_index(drop=True)
        )

    output = concepts.copy()
    for column in ("rxcui", "ingredient_name", "rxnorm_name", "atc_code", "atc_level"):
        output[column] = ""
    output["mapping_method"] = ""

    output = fill_eicu_mapping(output, by_hicl, key="hicl_key", method="drughiclseqno")
    output = fill_eicu_mapping(output, by_gtc, key="gtc_key", method="gtc")
    output = fill_eicu_mapping(
        output,
        by_name,
        key="drug_name_key",
        method="normalized_name",
    )
    output["atc_code"] = output["atc_code"].map(normalize_atc_code)
    output = output.sort_values(["drughiclseqno", "gtc", "drug_name"]).reset_index(
        drop=True
    )

    final_output = output.loc[:, list(EICU_OUTPUT_COLUMNS)].copy()
    distinct_count = len(final_output)
    rx_count = int((final_output["rxcui"] != "").sum())
    atc_count = int((final_output["atc_code"] != "").sum())
    unmapped = output[(output["rxcui"] == "") & (output["atc_code"] == "")]
    report = {
        "distinct_eicu_medication_concept_count": distinct_count,
        "mapped_by_hicl_count": int(
            (output["mapping_method"] == "drughiclseqno").sum()
        ),
        "mapped_by_gtc_count": int((output["mapping_method"] == "gtc").sum()),
        "mapped_by_normalized_name_count": int(
            (output["mapping_method"] == "normalized_name").sum()
        ),
        "rxcui_mapped_count": rx_count,
        "rxcui_mapped_percent": percentage(rx_count, distinct_count),
        "atc_mapped_count": atc_count,
        "atc_mapped_percent": percentage(atc_count, distinct_count),
        "unmapped_count": int(len(unmapped)),
        "unmapped_percent": percentage(len(unmapped), distinct_count),
    }
    return final_output, report


def fill_eicu_mapping(
    output: pd.DataFrame,
    mapping: pd.DataFrame,
    *,
    key: str,
    method: str,
) -> pd.DataFrame:
    """Fill unmapped eICU rows using one mapping priority."""

    if mapping.empty:
        return output
    merged = output.merge(mapping, how="left", on=key, suffixes=("", "_mapped"))
    still_unmapped = (merged["rxcui"] == "") & (merged["atc_code"] == "")
    has_mapping = (merged["rxcui_mapped"].fillna("") != "") | (
        merged["atc_code_mapped"].fillna("") != ""
    )
    fill_mask = still_unmapped & has_mapping
    for column in ("rxcui", "ingredient_name", "rxnorm_name", "atc_code", "atc_level"):
        mapped_column = f"{column}_mapped"
        merged.loc[fill_mask, column] = merged.loc[fill_mask, mapped_column].fillna("")
    merged.loc[fill_mask, "mapping_method"] = method
    return merged.loc[:, output.columns]


def percentage(numerator: int | float, denominator: int | float) -> float:
    """Return a rounded percentage with zero-denominator safety."""

    if not denominator:
        return 0.0
    return round(float(numerator) * 100.0 / float(denominator), 2)


def write_mapping_csv(path: Path, frame: pd.DataFrame, columns: Sequence[str]) -> None:
    """Write one mapping CSV with exact required columns."""

    path.parent.mkdir(parents=True, exist_ok=True)
    output = frame.loc[:, list(columns)].fillna("").map(clean_string)
    output.to_csv(path, index=False)


def reference_status(path: Path | None, expected_name: str) -> dict[str, Any]:
    """Return aggregate status metadata for one reference resource."""

    return {
        "expected": expected_name,
        "status": "found" if path is not None else "missing",
        "path": str(path) if path is not None else None,
    }


def build_medication_mappings(
    config: MedicationMappingBuildConfig = MedicationMappingBuildConfig(),
) -> dict[str, Any]:
    """Build MIMIC and eICU medication mapping CSVs plus an aggregate report."""

    search_roots = (
        config.medication_mapping_root,
        config.mapping_root,
        config.dataset_root,
    )
    ndc_rxcui_path = find_reference_file(
        explicit_path=config.ndc_rxcui_path,
        search_roots=search_roots,
        candidate_names=NDC_RXCUI_NAMES,
    )
    rxcui_atc4_path = find_reference_file(
        explicit_path=config.rxcui_atc4_path,
        search_roots=search_roots,
        candidate_names=RXCUI_ATC4_NAMES,
    )
    drug_atc_path = find_reference_file(
        explicit_path=config.drug_atc_path,
        search_roots=search_roots,
        candidate_names=DRUG_ATC_NAMES,
    )
    eicu_reference_path = find_reference_file(
        explicit_path=config.eicu_reference_path,
        search_roots=search_roots,
        candidate_names=EICU_REFERENCE_NAMES,
        exclude_paths=(config.eicu_output_path,),
    )

    ndc_rxcui = load_ndc_rxcui(ndc_rxcui_path)
    rxcui_atc4 = load_rxcui_atc4(rxcui_atc4_path)
    drug_atc = load_drug_atc(drug_atc_path)
    eicu_reference = load_eicu_reference(eicu_reference_path)

    mimic_output, mimic_report = build_mimic_mapping(
        config,
        ndc_rxcui=ndc_rxcui,
        rxcui_atc4=rxcui_atc4,
    )
    eicu_output, eicu_report = build_eicu_mapping(
        config,
        eicu_reference=eicu_reference,
        rxcui_atc4=rxcui_atc4,
        drug_atc=drug_atc,
    )

    write_mapping_csv(config.mimic_output_path, mimic_output, MIMIC_OUTPUT_COLUMNS)
    write_mapping_csv(config.eicu_output_path, eicu_output, EICU_OUTPUT_COLUMNS)

    missing_references = [
        name
        for name, path in (
            ("ndc2RXCUI.txt or equivalent", ndc_rxcui_path),
            ("RXCUI2atc4.csv or equivalent", rxcui_atc4_path),
            ("drug-atc.csv or equivalent", drug_atc_path),
            ("reviewed eICU HICL/GTC-to-RxNorm map", eicu_reference_path),
        )
        if path is None
    ]
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "data_safety": {
            "contains_patient_rows": False,
            "reporting_level": "distinct medication concepts and aggregate mapping coverage",
            "raw_clinical_rows_logged": False,
        },
        "configuration": {
            "dataset_root": str(config.dataset_root),
            "extracts_root": str(config.extracts_root),
            "mapping_root": str(config.mapping_root),
            "mimic_prescriptions_path": str(config.resolved_mimic_prescriptions_path),
            "eicu_medication_path": str(config.resolved_eicu_medication_path),
            "mimic_output_path": str(config.mimic_output_path),
            "eicu_output_path": str(config.eicu_output_path),
        },
        "reference_files": {
            "ndc_rxcui": reference_status(ndc_rxcui_path, "ndc2RXCUI.txt"),
            "rxcui_atc4": reference_status(rxcui_atc4_path, "RXCUI2atc4.csv"),
            "drug_atc": reference_status(drug_atc_path, "drug-atc.csv"),
            "eicu_reference": reference_status(
                eicu_reference_path,
                "reviewed eICU HICL/GTC/name RxNorm map",
            ),
        },
        "missing_reference_files": missing_references,
        "mimic": mimic_report,
        "eicu": eicu_report,
    }
    config.report_path.parent.mkdir(parents=True, exist_ok=True)
    config.report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    log_report(report)
    return report


def log_report(report: dict[str, Any]) -> None:
    """Emit concise aggregate logs for the mapping build."""

    mimic = report["mimic"]
    eicu = report["eicu"]
    logging.info(
        "MIMIC NDCs: %s distinct, %s RxCUI mapped, %s ATC-4/ATC mapped, %s ATC-3 mapped, %s unmapped",
        mimic["distinct_ndc_count"],
        mimic["rxcui_mapped_count"],
        mimic["atc_mapped_count"],
        mimic["atc3_mapped_count"],
        mimic["unmapped_count"],
    )
    logging.info(
        "eICU concepts: %s distinct, %s HICL mapped, %s GTC mapped, %s name mapped, %s RxCUI mapped, %s ATC mapped, %s unmapped",
        eicu["distinct_eicu_medication_concept_count"],
        eicu["mapped_by_hicl_count"],
        eicu["mapped_by_gtc_count"],
        eicu["mapped_by_normalized_name_count"],
        eicu["rxcui_mapped_count"],
        eicu["atc_mapped_count"],
        eicu["unmapped_count"],
    )
    if report["missing_reference_files"]:
        logging.warning(
            "Missing reference resources: %s",
            ", ".join(report["missing_reference_files"]),
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Build MIMIC/eICU medication mapping CSVs for harmonization.",
    )
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--extracts-root", type=Path, default=EXTRACTS_ROOT)
    parser.add_argument("--mapping-root", type=Path, default=MAPPING_ROOT)
    parser.add_argument(
        "--report",
        type=Path,
        default=REPORTS_ROOT / "medication_mapping_build_report.json",
    )
    parser.add_argument("--mimic-prescriptions", type=Path, default=None)
    parser.add_argument("--eicu-medication", type=Path, default=None)
    parser.add_argument("--ndc-rxcui", type=Path, default=None)
    parser.add_argument("--rxcui-atc4", type=Path, default=None)
    parser.add_argument("--drug-atc", type=Path, default=None)
    parser.add_argument("--eicu-reference", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args(argv)
    report = build_medication_mappings(
        MedicationMappingBuildConfig(
            dataset_root=args.dataset_root,
            extracts_root=args.extracts_root,
            mapping_root=args.mapping_root,
            report_path=args.report,
            mimic_prescriptions_path=args.mimic_prescriptions,
            eicu_medication_path=args.eicu_medication,
            ndc_rxcui_path=args.ndc_rxcui,
            rxcui_atc4_path=args.rxcui_atc4,
            drug_atc_path=args.drug_atc,
            eicu_reference_path=args.eicu_reference,
        )
    )
    print(
        "Wrote medication mappings: "
        f"mimic_ndc={report['mimic']['distinct_ndc_count']}, "
        f"eicu_concepts={report['eicu']['distinct_eicu_medication_concept_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
