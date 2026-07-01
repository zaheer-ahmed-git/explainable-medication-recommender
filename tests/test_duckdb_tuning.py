"""Regression tests for memory-safe DuckDB configuration.

These guard the fix for the harmonization ``vitals`` OOM-kill: an in-memory
DuckDB database cannot spill unless ``temp_directory`` is set, and it buffers
large ordered ``COPY`` results unless ``preserve_insertion_order`` is disabled.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from pipeline.extract_utils import configure_duckdb_connection, sql_string


def _setting(connection: duckdb.DuckDBPyConnection, name: str) -> str:
    row = connection.execute(f"SELECT current_setting('{name}')").fetchone()
    assert row is not None
    return str(row[0])


def test_in_memory_duckdb_defaults_are_memory_risky(tmp_path: Path) -> None:
    """Document the unsafe defaults the fix overrides.

    Insertion order is preserved (forces buffering large COPY results) and the
    spill directory is not our bounded node-local path, so the raw connection is
    prone to being OOM-killed under a cgroup smaller than physical RAM.
    """

    with duckdb.connect(database=":memory:") as connection:
        assert _setting(connection, "preserve_insertion_order").lower() == "true"
        assert _setting(connection, "temp_directory") != str(tmp_path / "spill")


def test_configure_enables_spilling_and_relaxes_order(tmp_path: Path) -> None:
    spill_dir = tmp_path / "spill"
    with duckdb.connect(database=":memory:") as connection:
        configure_duckdb_connection(
            connection,
            temp_directory=spill_dir,
            memory_limit="256MB",
            threads=2,
        )

        assert spill_dir.exists()
        assert _setting(connection, "temp_directory") == str(spill_dir)
        assert _setting(connection, "preserve_insertion_order").lower() == "false"
        assert _setting(connection, "threads") == "2"
        assert _setting(connection, "memory_limit") != ""


def test_configured_connection_copies_wide_union_to_parquet(tmp_path: Path) -> None:
    """A multi-branch UNION ``COPY`` (the vitals pattern) completes and spills."""

    output = tmp_path / "vitals_like.parquet"
    per_branch = 50_000
    branches = 7
    with duckdb.connect(database=":memory:") as connection:
        configure_duckdb_connection(
            connection,
            temp_directory=tmp_path / "spill",
            memory_limit="256MB",
            threads=4,
        )
        branch_sql = " UNION ALL BY NAME ".join(
            [
                "SELECT i AS id, CAST(i AS DOUBLE) AS value_numeric "
                f"FROM range(0, {per_branch}) AS t(i)"
            ]
            * branches
        )
        connection.execute(
            f"COPY ({branch_sql}) TO {sql_string(output)} (FORMAT PARQUET)"
        )
        count = connection.execute(
            f"SELECT COUNT(*) FROM read_parquet({sql_string(output)})"
        ).fetchone()

    assert output.exists()
    assert count is not None
    assert int(count[0]) == per_branch * branches
