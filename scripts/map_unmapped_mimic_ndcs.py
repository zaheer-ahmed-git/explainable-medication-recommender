"""Map unmapped MIMIC NDC codes to RxCUI / RxNorm name / ATC using safe sources.

Pipeline per NDC (no fuzzy matching, no fabricated codes):

1. Normalize the NDC to lookup variants while preserving the original string.
2. Local reference lookup first:
   - ``ndc2RXCUI.txt``  (NDC -> RxCUI)
   - ``RXCUI2atc4.csv``  (RxCUI -> ATC-4 + names)
3. If local lookup is incomplete, query RxNav (NLM, trusted):
   - ``/REST/ndcstatus.json``           NDC -> RxCUI + concept name (handles obsolete NDCs)
   - ``/REST/rxclass/class/byRxcui``    RxCUI -> ingredient (TTY=IN) + ATC-4
4. Roll ATC-4 up to ATC-3 (first four characters) for the project convention.

Outputs (range-scoped, never overwrites the main mapping file):

- ``$DATASET_ROOT/mappings/medications/mimic_ndc_rxnorm_atc_patch_rows_<a>_<b>.csv``
  with the pipeline schema: ``ndc, rxcui, ingredient_name, rxnorm_name, atc_code, atc_level``
- ``reports/mimic_ndc_mapping_review_rows_<a>_<b>.csv`` with full diagnostics.

The patch keeps exactly one row per NDC (the harmonization join is 1:1 on NDC),
so multi-ATC ambiguity is collapsed deterministically and recorded in ``notes``.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.config import MAPPING_ROOT, REPORTS_ROOT  # noqa: E402

RXNAV_BASE = "https://rxnav.nlm.nih.gov/REST"
PATCH_COLUMNS = (
    "ndc",
    "rxcui",
    "ingredient_name",
    "rxnorm_name",
    "atc_code",
    "atc_level",
)
REVIEW_COLUMNS = (
    "ndc",
    "original_drug_name",
    "count",
    "normalized_ndc_variants",
    "rxcui",
    "ingredient_name",
    "rxnorm_name",
    "atc_code",
    "atc_level",
    "mapping_status",
    "mapping_source",
    "confidence",
    "notes",
)


def normalize_digits(value: str) -> str:
    """Strip every non-digit character from an NDC-like string."""

    return "".join(ch for ch in str(value) if ch.isdigit())


def ndc_lookup_variants(original: str) -> list[str]:
    """Return deduplicated NDC variants for matching, preserving the original.

    MIMIC stores 11-digit NDCs. We also derive a plain 10-digit form by dropping
    a single leading zero only when it is safe (length 11 with a leading zero).
    """

    variants: list[str] = []

    def add(candidate: str) -> None:
        if candidate and candidate not in variants:
            variants.append(candidate)

    add(str(original).strip())
    digits = normalize_digits(original)
    add(digits)
    if len(digits) == 11:
        add(digits.zfill(11))
        if digits[0] == "0":
            add(digits[1:])  # 11 -> 10 by dropping one leading zero
    if len(digits) == 10:
        add(digits.zfill(11))  # 10 -> 11 by left padding
    return variants


def atc3_from_code(atc4: str) -> str:
    """Roll an ATC-4 (e.g. ``C03AA``) up to ATC-3 (``C03A``)."""

    code = "".join(ch for ch in str(atc4) if ch.isalnum()).upper()
    return code[:4] if len(code) >= 4 else ""


@dataclass
class LocalReferences:
    ndc_to_rxcui: dict[str, str] = field(default_factory=dict)
    rxcui_to_atc4: dict[str, str] = field(default_factory=dict)
    rxcui_to_ingredient: dict[str, str] = field(default_factory=dict)
    rxcui_to_rxnorm_name: dict[str, str] = field(default_factory=dict)


def load_local_references(mapping_root: Path) -> LocalReferences:
    """Load local NDC->RxCUI and RxCUI->ATC4/name references."""

    refs = LocalReferences()
    ndc_path = mapping_root / "medications" / "ndc2RXCUI.txt"
    atc_path = mapping_root / "medications" / "RXCUI2atc4.csv"

    if ndc_path.exists():
        with ndc_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                ndc = normalize_digits(row.get("ndc", ""))
                rxcui = str(row.get("rxcui", "")).strip()
                if ndc and rxcui:
                    refs.ndc_to_rxcui.setdefault(ndc, rxcui)

    if atc_path.exists():
        with atc_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rxcui = str(row.get("rxcui", "")).strip()
                atc4 = str(row.get("ATC4", row.get("atc4", ""))).strip()
                name = str(row.get("rxnorm_name", "")).strip()
                ingredient = str(row.get("ingredient_name", "")).strip()
                if rxcui and atc4:
                    refs.rxcui_to_atc4.setdefault(rxcui, atc4)
                if rxcui and name:
                    refs.rxcui_to_rxnorm_name.setdefault(rxcui, name)
                if rxcui and ingredient:
                    refs.rxcui_to_ingredient.setdefault(rxcui, ingredient)
    return refs


class RxNavClient:
    """Minimal cached RxNav client for NDC/RxCUI/ATC lookups."""

    def __init__(self, cache_path: Path, *, enabled: bool, pause_s: float = 0.05):
        self.cache_path = cache_path
        self.enabled = enabled
        self.pause_s = pause_s
        self.cache: dict[str, Any] = {}
        self.calls = 0
        if cache_path.exists():
            try:
                self.cache = json.loads(cache_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self.cache = {}

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache), encoding="utf-8")

    def _get(self, url: str) -> dict[str, Any] | None:
        if url in self.cache:
            return self.cache[url]
        if not self.enabled:
            return None
        try:
            self.calls += 1
            with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - network failures are recorded, not raised
            logging.debug("RxNav error for %s: %s", url, exc)
            payload = {"_error": str(exc)}
        self.cache[url] = payload
        if self.pause_s:
            time.sleep(self.pause_s)
        return payload

    def ndc_status(self, ndc11: str) -> dict[str, str]:
        """Return rxcui + concept name for an NDC (including obsolete NDCs)."""

        url = f"{RXNAV_BASE}/ndcstatus.json?ndc={urllib.parse.quote(ndc11)}"
        payload = self._get(url) or {}
        status = payload.get("ndcStatus", {}) if isinstance(payload, dict) else {}
        rxcui = str(status.get("rxcui", "") or "").strip()
        name = str(status.get("conceptName", "") or "").strip()
        return {
            "rxcui": rxcui,
            "rxnorm_name": name,
            "status": str(status.get("status", "") or ""),
            "rxnormNdc": str(status.get("rxnormNdc", "") or ""),
        }

    def atc_for_rxcui(self, rxcui: str) -> list[dict[str, str]]:
        """Return ATC-4 classes and ingredient for an RxCUI via RxClass."""

        url = (
            f"{RXNAV_BASE}/rxclass/class/byRxcui.json?"
            f"rxcui={urllib.parse.quote(rxcui)}&relaSource=ATC"
        )
        payload = self._get(url) or {}
        info_list = (
            payload.get("rxclassDrugInfoList", {}).get("rxclassDrugInfo", [])
            if isinstance(payload, dict)
            else []
        )
        results: list[dict[str, str]] = []
        for info in info_list:
            item = info.get("rxclassMinConceptItem", {})
            concept = info.get("minConcept", {})
            atc4 = str(item.get("classId", "") or "").strip().upper()
            if atc4:
                results.append(
                    {
                        "atc4": atc4,
                        "atc_class_name": str(item.get("className", "") or ""),
                        "ingredient_name": str(concept.get("name", "") or ""),
                        "ingredient_rxcui": str(concept.get("rxcui", "") or ""),
                    }
                )
        return results

    def ingredients_for_rxcui(self, rxcui: str) -> list[dict[str, str]]:
        """Return ingredient (TTY=IN) concepts (rxcui + name) for an RxCUI."""

        url = (
            f"{RXNAV_BASE}/rxcui/{urllib.parse.quote(rxcui)}"
            "/related.json?tty=IN+PIN+MIN"
        )
        payload = self._get(url) or {}
        groups = (
            payload.get("relatedGroup", {}).get("conceptGroup", [])
            if isinstance(payload, dict)
            else []
        )
        results: list[dict[str, str]] = []
        for group in groups:
            for prop in group.get("conceptProperties", []) or []:
                name = str(prop.get("name", "") or "").strip()
                ing_rxcui = str(prop.get("rxcui", "") or "").strip()
                if name:
                    results.append({"rxcui": ing_rxcui, "name": name})
        return results


@dataclass
class MappedNdc:
    ndc: str
    original_drug_name: str
    count: int
    variants: list[str]
    rxcui: str = ""
    ingredient_name: str = ""
    rxnorm_name: str = ""
    atc_code: str = ""
    atc_level: str = ""
    mapping_status: str = "unmapped"
    mapping_source: str = ""
    confidence: str = "unmapped"
    notes: str = ""


def map_single_ndc(
    ndc: str,
    *,
    original_drug_name: str,
    count: int,
    refs: LocalReferences,
    rxnav: RxNavClient,
) -> MappedNdc:
    """Resolve one NDC to RxCUI / ingredient / ATC using local refs then RxNav."""

    variants = ndc_lookup_variants(ndc)
    result = MappedNdc(
        ndc=str(ndc),
        original_drug_name=original_drug_name,
        count=count,
        variants=variants,
    )
    notes: list[str] = []
    sources: list[str] = []

    # --- Step: NDC -> RxCUI (local first) ---
    rxcui = ""
    for variant in variants:
        key = normalize_digits(variant)
        if key in refs.ndc_to_rxcui:
            rxcui = refs.ndc_to_rxcui[key]
            sources.append("local:ndc2RXCUI.txt")
            notes.append(f"ndc->rxcui via local variant {key}")
            break

    if not rxcui:
        for variant in variants:
            key = normalize_digits(variant)
            if len(key) != 11:
                continue
            status = rxnav.ndc_status(key)
            if status.get("rxcui"):
                rxcui = status["rxcui"]
                if status.get("rxnorm_name"):
                    result.rxnorm_name = status["rxnorm_name"]
                sources.append("rxnav:ndcstatus")
                notes.append(
                    f"ndc->rxcui via RxNav ndcstatus ({status.get('status', '?')},"
                    f" variant {key})"
                )
                break

    if not rxcui:
        result.mapping_status = "unmapped_ndc_to_rxcui"
        result.confidence = "unmapped"
        result.notes = "; ".join(
            notes + [f"no RxCUI for variants {variants} in local refs or RxNav"]
        )
        result.mapping_source = ";".join(dict.fromkeys(sources)) or "none"
        return result

    result.rxcui = rxcui

    # --- Step: RxCUI -> ingredient + RxNorm name ---
    if rxcui in refs.rxcui_to_ingredient:
        result.ingredient_name = refs.rxcui_to_ingredient[rxcui]
    if not result.rxnorm_name and rxcui in refs.rxcui_to_rxnorm_name:
        result.rxnorm_name = refs.rxcui_to_rxnorm_name[rxcui]

    # --- Step: resolve ingredient concept (name + rxcui) for naming + ATC fallback ---
    ingredient_rxcui = ""
    ingredients = rxnav.ingredients_for_rxcui(rxcui)
    if ingredients:
        ingredient_rxcui = ingredients[0]["rxcui"]
        if not result.ingredient_name:
            result.ingredient_name = ingredients[0]["name"]
            sources.append("rxnav:related-IN")

    # --- Step: RxCUI -> ATC-4 (local first, then RxNav on drug then ingredient) ---
    atc4 = ""
    atc4_values: list[str] = []
    if rxcui in refs.rxcui_to_atc4:
        atc4 = refs.rxcui_to_atc4[rxcui]
        sources.append("local:RXCUI2atc4.csv")
        notes.append(f"rxcui->atc4 {atc4} via local")
    elif ingredient_rxcui and ingredient_rxcui in refs.rxcui_to_atc4:
        atc4 = refs.rxcui_to_atc4[ingredient_rxcui]
        sources.append("local:RXCUI2atc4.csv(ingredient)")
        notes.append(f"ingredient rxcui {ingredient_rxcui}->atc4 {atc4} via local")
    else:
        lookups = [(rxcui, "rxnav:rxclass")]
        for ing in ingredients:
            if ing.get("rxcui"):
                lookups.append((ing["rxcui"], "rxnav:rxclass(ingredient)"))
        for lookup_rxcui, tag in lookups:
            if not lookup_rxcui:
                continue
            atc_options = rxnav.atc_for_rxcui(lookup_rxcui)
            if atc_options:
                atc4_values = sorted(
                    {opt["atc4"] for opt in atc_options if opt["atc4"]}
                )
                atc4 = atc4_values[0]
                chosen = next(opt for opt in atc_options if opt["atc4"] == atc4)
                if not result.ingredient_name and chosen.get("ingredient_name"):
                    result.ingredient_name = chosen["ingredient_name"]
                sources.append(tag)
                notes.append(f"{lookup_rxcui}->atc4 {atc4} via {tag}")
                if len(atc4_values) > 1:
                    notes.append(
                        "multiple ATC classes; kept first for 1:1 patch, all="
                        + ",".join(atc4_values)
                    )
                break

    if not result.rxnorm_name:
        result.rxnorm_name = result.ingredient_name

    # --- Confidence + status ---
    used_rxnav_ndc = "rxnav:ndcstatus" in sources
    if atc4:
        result.atc_code = atc3_from_code(atc4)
        result.atc_level = "3"
        result.mapping_status = "mapped_rxnorm_atc"
        multi = any("multiple ATC classes" in n for n in notes)
        result.confidence = "medium" if (used_rxnav_ndc and multi) else "high"
        if multi and result.confidence == "high":
            result.confidence = "medium"
    else:
        result.mapping_status = "mapped_rxcui_missing_atc"
        result.confidence = "medium"
        notes.append("RxCUI resolved but no ATC mapping found")

    result.mapping_source = ";".join(dict.fromkeys(sources)) or "none"
    result.notes = "; ".join(notes)
    return result


def select_range(
    records: list[dict[str, Any]], start: int, end: int
) -> list[dict[str, Any]]:
    """Return 1-indexed inclusive slice [start, end] of the aggregate records."""

    return records[start - 1 : end]


def run(
    *,
    aggregate_path: Path,
    start: int,
    end: int,
    mapping_root: Path,
    reports_root: Path,
    use_network: bool,
) -> dict[str, Any]:
    data = json.loads(aggregate_path.read_text(encoding="utf-8"))
    records = data.get("top_unmapped_ndcs_by_row_count", [])
    if len(records) < end:
        raise ValueError(
            f"Aggregate file lists {len(records)} NDCs but range needs >= {end}. "
            "Regenerate the aggregate with more rows."
        )
    selected = select_range(records, start, end)

    refs = load_local_references(mapping_root)
    cache_path = mapping_root / "medications" / "_rxnav_cache.json"
    rxnav = RxNavClient(cache_path, enabled=use_network)

    mapped: list[MappedNdc] = []
    for rec in selected:
        ndc = str(rec.get("ndc", ""))
        if ndc in {"0", "(blank_ndc)", ""} or not normalize_digits(ndc):
            row = MappedNdc(
                ndc=ndc,
                original_drug_name=str(rec.get("drug_name", "") or ""),
                count=int(rec.get("unmapped_row_count", rec.get("count", 0)) or 0),
                variants=[ndc],
                mapping_status="invalid_ndc_placeholder",
                confidence="unmapped",
                mapping_source="none",
                notes="placeholder/blank NDC; not a real drug code",
            )
            mapped.append(row)
            continue
        row = map_single_ndc(
            ndc,
            original_drug_name=str(rec.get("drug_name", "") or ""),
            count=int(rec.get("unmapped_row_count", rec.get("count", 0)) or 0),
            refs=refs,
            rxnav=rxnav,
        )
        mapped.append(row)

    rxnav.save()

    # --- Write patch (one row per NDC, pipeline schema) ---
    patch_path = (
        mapping_root
        / "medications"
        / f"mimic_ndc_rxnorm_atc_patch_rows_{start}_{end}.csv"
    )
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    with patch_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(PATCH_COLUMNS)
        for row in mapped:
            writer.writerow(
                [
                    row.ndc,
                    row.rxcui,
                    row.ingredient_name,
                    row.rxnorm_name,
                    row.atc_code,
                    row.atc_level,
                ]
            )

    # --- Write review (full diagnostics) ---
    review_path = reports_root / f"mimic_ndc_mapping_review_rows_{start}_{end}.csv"
    review_path.parent.mkdir(parents=True, exist_ok=True)
    with review_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(REVIEW_COLUMNS)
        for row in mapped:
            writer.writerow(
                [
                    row.ndc,
                    row.original_drug_name,
                    row.count,
                    "|".join(row.variants),
                    row.rxcui,
                    row.ingredient_name,
                    row.rxnorm_name,
                    row.atc_code,
                    row.atc_level,
                    row.mapping_status,
                    row.mapping_source,
                    row.confidence,
                    row.notes,
                ]
            )

    total = len(mapped)
    rx = sum(1 for r in mapped if r.rxcui)
    atc = sum(1 for r in mapped if r.atc_code)
    atc3 = sum(1 for r in mapped if r.atc_level == "3")
    atc4_only = sum(1 for r in mapped if r.atc_code and r.atc_level == "4")
    missing_atc = sum(
        1 for r in mapped if r.mapping_status == "mapped_rxcui_missing_atc"
    )
    unmapped = sum(1 for r in mapped if not r.rxcui)
    ambiguous = sum(1 for r in mapped if "multiple ATC classes" in r.notes)
    by_conf: dict[str, int] = {}
    for r in mapped:
        by_conf[r.confidence] = by_conf.get(r.confidence, 0) + 1

    summary = {
        "schema_version": "mimic-ndc-mapping-summary-v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "range": {"start": start, "end": end, "count": total},
        "rxnav_network_used": use_network,
        "rxnav_live_calls": rxnav.calls,
        "coverage": {
            "total": total,
            "rxcui_mapped": rx,
            "rxcui_mapped_percent": round(100 * rx / total, 2) if total else 0.0,
            "atc_mapped": atc,
            "atc_mapped_percent": round(100 * atc / total, 2) if total else 0.0,
            "atc3_count": atc3,
            "atc4_only_count": atc4_only,
            "mapped_rxcui_missing_atc": missing_atc,
            "unmapped": unmapped,
            "ambiguous_multi_atc": ambiguous,
        },
        "confidence_breakdown": by_conf,
        "outputs": {"patch": str(patch_path), "review": str(review_path)},
    }
    summary_path = reports_root / f"mimic_ndc_mapping_summary_rows_{start}_{end}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aggregate",
        type=Path,
        default=REPORTS_ROOT / "unmapped_mimic_ndc_aggregate.json",
    )
    parser.add_argument("--start", type=int, default=34, help="1-indexed inclusive")
    parser.add_argument("--end", type=int, default=229, help="1-indexed inclusive")
    parser.add_argument("--mapping-root", type=Path, default=MAPPING_ROOT)
    parser.add_argument("--reports-root", type=Path, default=REPORTS_ROOT)
    parser.add_argument(
        "--no-network",
        action="store_true",
        help="Use only local references and the RxNav cache.",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))
    summary = run(
        aggregate_path=args.aggregate,
        start=args.start,
        end=args.end,
        mapping_root=args.mapping_root,
        reports_root=args.reports_root,
        use_network=not args.no_network,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
