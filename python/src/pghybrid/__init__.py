"""Hybrid search on the Postgres you already have.

Vector similarity and full-text search, combined by Reciprocal Rank Fusion, over plain
``pgvector``. No extension beyond ``pgvector`` is required, which is the entire point:
``pg_search`` and its peers are unavailable on most managed Postgres.

    from pghybrid import Config, HybridSearch

    search = HybridSearch(
        Config(table="chunks", text_column="content", vector_column="embedding",
               tsvector_column="fts", paramstyle="pyformat"),
        execute=lambda sql, params: conn.execute(sql, params).fetchall(),
    )
    rows = search.search("renewal notice period", embedding=vector, limit=10)

``execute`` is whatever driver you already have. This package opens no connections and
calls no embedding model.
"""

from .config import (
    COSINE,
    INNER_PRODUCT,
    L1,
    L2,
    METRICS,
    Config,
    Metric,
    Recency,
    Weights,
)
from .explain import ExplainReport, ExplainRow, FindReport, explain, measure_weights
from .schema import (
    Statement,
    TableInfo,
    TableNotFound,
    build_migration,
    dbapi_executor,
    introspect,
    suggest_config,
)
from .search import AsyncHybridSearch, HybridSearch, SearchResult
from .sql import IdentifierError, build_search_sql, quote_ident
from .textquery import ParsedQuery, parse_query

__version__ = "0.1.1"

__all__ = [
    # Configuration
    "Config",
    "Metric",
    "Recency",
    "Weights",
    "METRICS",
    "COSINE",
    "L2",
    "INNER_PRODUCT",
    "L1",
    # Searching
    "HybridSearch",
    "AsyncHybridSearch",
    "SearchResult",
    # Diagnosing
    "explain",
    "measure_weights",
    "ExplainReport",
    "ExplainRow",
    "FindReport",
    # Schema
    "introspect",
    "suggest_config",
    "build_migration",
    "dbapi_executor",
    "TableInfo",
    "TableNotFound",
    "Statement",
    # The SQL itself, for people who want to read or steal it
    "build_search_sql",
    "quote_ident",
    "IdentifierError",
    "parse_query",
    "ParsedQuery",
]


def __getattr__(name: str) -> object:
    """Expose ``doctor`` lazily.

    It is the heaviest module in the package and the only one most callers never touch,
    so importing it eagerly would make ``import pghybrid`` pay for a feature reserved
    for the moment something has already gone wrong.
    """
    if name == "doctor":
        from .doctor import doctor

        return doctor
    if name == "DoctorReport":
        from .doctor import DoctorReport

        return DoctorReport
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
