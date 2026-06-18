"""CLI for generating a metadata-only clinical source inventory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from pipeline.config import REPORTS_ROOT, SOURCE_SPECS, ensure_local_directories
from pipeline.io_utils import build_source_inventory


DEFAULT_OUTPUT_PATH = REPORTS_ROOT / "source_inventory.json"


def write_source_inventory(
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, object]:
    """Write source metadata and return the generated inventory."""

    ensure_local_directories()
    inventory = build_source_inventory(SOURCE_SPECS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return inventory


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a metadata-only inventory of local clinical sources.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path for the JSON inventory. Defaults to reports/source_inventory.json.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    inventory = write_source_inventory(args.output)
    source_count = len(inventory["sources"])
    file_count = sum(len(source["files"]) for source in inventory["sources"])
    print(
        f"Wrote metadata for {source_count} sources and {file_count} files to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
