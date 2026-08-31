"""Shared fixtures for the pghybrid unit suite.

Nothing here opens a connection or reads the environment. The unit tests have to run
on a laptop with no Postgres installed, because that is the machine most first-time
contributors have, and a suite that needs a database to start is a suite that stops
being run.
"""

from __future__ import annotations

from typing import Any, Callable

import pytest

from pghybrid.config import Config


@pytest.fixture
def make_config() -> Callable[..., Config]:
    """Factory for the canonical Config used across the unit tests.

    The defaults here are deliberately richer than the library's own: a schema-qualified
    table, a non-default id column, a stored tsvector, filterable columns and passthrough
    columns each switch on a branch of the SQL builder that a bare three-argument Config
    would leave untested. Pass keyword overrides to narrow it back down.
    """

    def _make(**overrides: Any) -> Config:
        kwargs = {
            "table": "public.chunks",
            "text_column": "content",
            "vector_column": "embedding",
            "id_column": "chunk_id",
            "tsvector_column": "content_tsv",
            "filter_columns": ["tenant_id", "lang"],
            "extra_columns": ["title", "url"],
        }
        kwargs.update(overrides)
        return Config(**kwargs)

    return _make


@pytest.fixture
def config(make_config: Callable[..., Config]) -> Config:
    """The canonical Config, for the many tests that never need to vary it."""
    return make_config()


@pytest.fixture(scope="session")
def demo_table():
    """Create and seed the demo corpus once per session.

    It used to be created by a module-scoped fixture inside test_integration.py, which
    made every other module that touched it depend on collection order — the CLI tests
    passed locally against a database left seeded by an earlier run, and failed in CI
    where the ordering put them first. A shared session fixture is the fix: no module
    relies on another's side effects.
    """
    import os
    import pathlib as _pathlib
    import sys as _sys

    psycopg = pytest.importorskip("psycopg")
    dsn = os.environ.get(
        "PGHYBRID_TEST_DSN", "postgresql://postgres:pghybrid@localhost:55432/pghybrid"
    )
    try:
        connection = psycopg.connect(dsn, autocommit=True)
    except Exception:  # pragma: no cover - depends on the environment
        pytest.skip(f"no Postgres at {dsn}")

    _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2] / "scripts"))
    from seed_demo import DOCUMENTS, SCHEMA, to_pgvector, unit_vector

    with connection:
        connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        connection.execute(SCHEMA)
        for angle, title, content in DOCUMENTS:
            connection.execute(
                "INSERT INTO chunks (title, content, embedding) VALUES (%s, %s, %s)",
                (title, content, to_pgvector(unit_vector(angle))),
            )
    yield dsn


@pytest.fixture
def connection_for_cli(demo_table):
    """A separate connection for CLI tests to inspect the database with.

    The CLI opens its own, so the tests need one of their own to check that a command
    left the schema alone.
    """
    import os

    psycopg = pytest.importorskip("psycopg")
    dsn = os.environ.get(
        "PGHYBRID_TEST_DSN", "postgresql://postgres:pghybrid@localhost:55432/pghybrid"
    )
    with psycopg.connect(dsn, autocommit=True, row_factory=psycopg.rows.dict_row) as conn:
        yield conn
