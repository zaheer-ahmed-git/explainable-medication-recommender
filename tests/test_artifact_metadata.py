from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.artifact_metadata import infer_consistent_version
from tests.milestone6_helpers import write_parquet_rows


def test_infer_consistent_version_uses_inputs_and_rejects_conflicts(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    write_parquet_rows(first, ("feature_version",), (("temporal-features-v2",),))
    write_parquet_rows(second, ("feature_version",), (("temporal-features-v2",),))

    assert (
        infer_consistent_version(
            (first, second),
            column_name="feature_version",
            fallback_version="temporal-features-v1",
        )
        == "temporal-features-v2"
    )

    write_parquet_rows(second, ("feature_version",), (("temporal-features-v1",),))
    with pytest.raises(ValueError, match="Conflicting feature_version"):
        infer_consistent_version(
            (first, second),
            column_name="feature_version",
            fallback_version="temporal-features-v1",
        )
