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


@pytest.fixture
def connection_for_cli():
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
