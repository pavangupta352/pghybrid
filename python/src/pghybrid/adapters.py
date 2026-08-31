"""One-line wiring for the drivers people actually have.

:class:`~pghybrid.search.HybridSearch` takes an ``execute`` callable, which keeps the
package free of driver dependencies but leaves every user writing the same lambda, and
choosing a placeholder style, which is the single thing most likely to be got wrong.
``$1`` and ``%s`` are not interchangeable, and picking the wrong one fails with a message
about parameter counts that says nothing about the cause.

So each helper here does both: it wires the executor for one driver *and* sets the
placeholder style that driver needs. Nothing is imported at module scope, so the package
keeps its promise of zero runtime dependencies whichever of these you never use.

    from pghybrid.adapters import for_psycopg

    search = for_psycopg(conn, table="chunks", text_column="content",
                         vector_column="embedding", tsvector_column="fts")
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any, Optional

from .config import Config, ParamStyle
from .search import AsyncHybridSearch, HybridSearch

__all__ = [
    "for_psycopg",
    "for_asyncpg",
    "for_sqlalchemy",
    "for_django",
    "psycopg_executor",
    "asyncpg_executor",
    "sqlalchemy_executor",
    "django_executor",
]


def _as_mappings(cursor: Any) -> list[dict[str, Any]]:
    """Read a DBAPI cursor as dicts, whatever row factory it was configured with.

    A connection may already be returning mappings (``psycopg.rows.dict_row`` is
    common), and zipping column names against one of those pairs the names with the
    dict's keys instead of its values, so every column comes back holding its own name.
    """
    if cursor.description is None:
        return []
    columns = [column[0] for column in cursor.description]
    rows = cursor.fetchall()
    if rows and isinstance(rows[0], Mapping):
        return [dict(row) for row in rows]
    return [dict(zip(columns, row)) for row in rows]


def _resolve(config: Optional[Config], paramstyle: ParamStyle, **overrides: Any) -> Config:
    """Build or adjust a Config, forcing the placeholder style the driver requires.

    Passing ``paramstyle`` yourself is not an error worth respecting: a psycopg
    connection cannot execute ``$1`` no matter how firmly it was requested.
    """
    if config is None:
        if not overrides:
            raise TypeError(
                "pass a Config, or the keyword arguments to build one "
                "(table=..., text_column=..., vector_column=...)"
            )
        overrides.pop("paramstyle", None)
        return Config(paramstyle=paramstyle, **overrides)

    if overrides:
        overrides.pop("paramstyle", None)
        config = replace(config, **overrides)
    return replace(config, paramstyle=paramstyle)


# ------------------------------------------------------------------------- psycopg


def psycopg_executor(connection: Any) -> Any:
    """An executor over a psycopg 2 or 3 connection, cursor factory notwithstanding.

    Rows are returned as mappings whatever the connection was configured for, because
    the caller asked for search results rather than for tuples they then have to zip
    against a column list.
    """

    def execute(sql: str, params: list[Any]) -> list[dict[str, Any]]:
        with connection.cursor() as cursor:
            cursor.execute(sql, params)
            return _as_mappings(cursor)

    return execute


def for_psycopg(connection: Any, config: Optional[Config] = None, **kwargs: Any) -> HybridSearch:
    """Wire a psycopg connection. psycopg speaks ``%s``, so the style is set for you."""
    return HybridSearch(_resolve(config, "pyformat", **kwargs), psycopg_executor(connection))


# ------------------------------------------------------------------------- asyncpg


def asyncpg_executor(connection: Any) -> Any:
    """An executor over an asyncpg connection or pool.

    Both expose ``fetch``, so one function covers them, and asyncpg's ``Record`` is
    already a mapping.
    """

    async def execute(sql: str, params: list[Any]) -> list[Any]:
        return list(await connection.fetch(sql, *params))

    return execute


def for_asyncpg(
    connection: Any, config: Optional[Config] = None, **kwargs: Any
) -> AsyncHybridSearch:
    """Wire an asyncpg connection or pool. asyncpg speaks ``$1``."""
    return AsyncHybridSearch(_resolve(config, "numeric", **kwargs), asyncpg_executor(connection))


# ---------------------------------------------------------------------- SQLAlchemy


def sqlalchemy_executor(bind: Any) -> Any:
    """An executor over a SQLAlchemy Session, Connection or Engine.

    Uses ``exec_driver_sql``, which hands the statement to the DBAPI untouched, rather
    than ``text()``. ``text()`` would have to bind ``:p0`` style parameters, and in
    Postgres ``:`` is ambiguous with the cast operator: SQLAlchemy silently declines to
    bind ``:p0`` in ``:p0::vector`` and the server then reports a syntax error at a colon
    that looks nothing like the cause. The alternative, emitting ``CAST(x AS vector)``
    throughout, would make the generated SQL less idiomatic for every user in order to
    suit one ORM.
    """

    def execute(sql: str, params: list[Any]) -> list[dict[str, Any]]:
        # A Session has .connection(); an Engine has .connect(); a Connection is already
        # what we need. Engines are handled last because a Session also exposes .connect
        # on some versions.
        if hasattr(bind, "connection") and callable(bind.connection):
            connection = bind.connection()
            result = connection.exec_driver_sql(sql, tuple(params))
            return [dict(row) for row in result.mappings()]
        if hasattr(bind, "exec_driver_sql"):
            result = bind.exec_driver_sql(sql, tuple(params))
            return [dict(row) for row in result.mappings()]
        with bind.connect() as connection:
            result = connection.exec_driver_sql(sql, tuple(params))
            return [dict(row) for row in result.mappings()]

    return execute


def for_sqlalchemy(bind: Any, config: Optional[Config] = None, **kwargs: Any) -> HybridSearch:
    """Wire a SQLAlchemy Session, Connection or Engine."""
    return HybridSearch(_resolve(config, "pyformat", **kwargs), sqlalchemy_executor(bind))


# -------------------------------------------------------------------------- Django


def django_executor(using: str = "default") -> Any:
    """An executor over a Django database connection.

    Django's Postgres backend is psycopg, so the placeholder style is ``%s`` and the
    connection is taken from ``django.db.connections`` at call time rather than at
    import time, the alias may not be configured yet when the module is imported.
    """

    def execute(sql: str, params: list[Any]) -> list[dict[str, Any]]:
        from django.db import connections

        with connections[using].cursor() as cursor:
            cursor.execute(sql, params)
            return _as_mappings(cursor)

    return execute


def for_django(
    config: Optional[Config] = None, using: str = "default", **kwargs: Any
) -> HybridSearch:
    """Wire a Django database connection by alias."""
    return HybridSearch(_resolve(config, "pyformat", **kwargs), django_executor(using))
