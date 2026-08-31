"""Running the generated statement, and shaping what comes back.

pghybrid never opens a connection. A :class:`HybridSearch` is built around an
``execute`` callable that takes ``(sql, params)`` and returns rows, which is what lets
the package work with psycopg, asyncpg, SQLAlchemy or node-postgres without importing
any of them and without holding an opinion about pooling, transactions or retries::

    search = HybridSearch(cfg, lambda sql, args: conn.execute(sql, args).fetchall())
    search = AsyncHybridSearch(cfg, lambda sql, args: conn.fetch(sql, *args))

The row shaping is the part worth being careful about. A row found by only one signal
has a NULL rank for the other, and the natural implementation — ``float(row["text_rank"])``
— raises on exactly the rows hybrid search exists to surface. Every conversion here
tolerates NULL and says so.
"""

from __future__ import annotations

from collections.abc import Awaitable, Iterable, Mapping
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Union

from .config import Config
from .sql import FusionMethod, build_search_sql

#: What the caller's ``execute`` is expected to be. The row type is deliberately loose:
#: anything dict-like survives :func:`row_mapping`.
Executor = Callable[[str, list[Any]], Iterable[Any]]
AsyncExecutor = Callable[[str, list[Any]], Awaitable[Iterable[Any]]]

#: Columns the fused query always produces. Everything else in a row came from the
#: user's table and is handed back untouched in :attr:`SearchResult.row`.
_SIGNAL_COLUMNS = frozenset(
    {
        "id",
        "score",
        "fused_score",
        "vector_rank",
        "vector_distance",
        "vector_contribution",
        "text_rank",
        "text_score",
        "text_contribution",
        "recency_factor",
        "highlight",
    }
)


@dataclass(frozen=True)
class SearchResult:
    """One ranked row, with the arithmetic that produced its position kept intact.

    The decomposition is not decoration. When a search result looks wrong the only
    useful question is which signal put it there, and a bare ``(id, score)`` tuple
    cannot answer it. Both ranks, both raw scores and both fused contributions travel
    with every row so the answer is always one attribute away.

    ``vector_rank`` and ``text_rank`` are None when that signal did not retrieve the
    row at all, which is different from retrieving it last.
    """

    id: Any
    score: float
    fused_score: float
    vector_rank: int | None
    vector_distance: float | None
    vector_contribution: float
    text_rank: int | None
    text_score: float | None
    text_contribution: float
    recency_factor: float | None = None
    highlight: str | None = None
    #: The columns copied through from the table (``text_column`` plus
    #: ``extra_columns``). Excluded from equality and repr so that comparing two
    #: results compares their ranking, and printing one does not print a whole chunk.
    row: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def matched_by(self) -> str:
        """Which signals retrieved this row: ``both``, ``vector``, ``text`` or ``none``."""
        if self.vector_rank is not None and self.text_rank is not None:
            return "both"
        if self.vector_rank is not None:
            return "vector"
        if self.text_rank is not None:
            return "text"
        return "none"

    def get(self, name: str, default: Any = None) -> Any:
        """Read a passthrough column, falling back to this result's own fields.

        ``result.get("title")`` reaches the table column and ``result.get("score")``
        the ranking, so calling code does not have to know which is which.
        """
        if name in self.row:
            return self.row[name]
        if name in _RESULT_FIELDS:
            return getattr(self, name)
        return default


_RESULT_FIELDS = frozenset(f.name for f in fields(SearchResult))


def row_mapping(row: Any) -> Mapping[str, Any]:
    """Coerce one driver row into a mapping.

    Drivers disagree about what a row is: psycopg returns tuples unless told
    otherwise, asyncpg returns a Record, SQLAlchemy a Row with a ``_mapping`` view.
    Accepting all of them here is what keeps the ``execute`` callable a one-liner
    instead of an adapter the user has to write.
    """
    if isinstance(row, Mapping):
        return row

    mapping = getattr(row, "_mapping", None)  # SQLAlchemy Row
    if isinstance(mapping, Mapping):
        return mapping

    as_dict = getattr(row, "_asdict", None)  # namedtuple, and several ORMs
    if callable(as_dict):
        result = as_dict()
        if isinstance(result, Mapping):
            return result

    keys = getattr(row, "keys", None)  # asyncpg Record, sqlite3.Row
    if callable(keys):
        return {str(key): row[key] for key in keys()}

    raise TypeError(
        f"execute() returned {type(row).__name__} rows, which are not dict-like. "
        "Return mappings instead: psycopg wants row_factory=dict_row, sqlite3 wants "
        "conn.row_factory = sqlite3.Row, asyncpg and SQLAlchemy already comply."
    )


def as_float(value: Any) -> float | None:
    """Float conversion that passes NULL through instead of raising on it.

    Also normalises the ``Decimal`` some drivers return for ``numeric``, so a caller
    never has to think about which driver produced a score.
    """
    if value is None:
        return None
    if isinstance(value, float):
        return value
    return float(value)


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _as_score(value: Any) -> float:
    """A contribution that must be a number: absent signals contribute zero, not NULL."""
    converted = as_float(value)
    return 0.0 if converted is None else converted


def result_from_row(row: Any) -> SearchResult:
    """Build a :class:`SearchResult` from one driver row.

    Shared by the sync and async clients. Anything the query added beyond the ranking
    columns is passed through in ``row`` rather than dropped, because the caller
    usually needs the title next to the score.
    """
    mapping = row_mapping(row)
    try:
        identifier = mapping["id"]
    except KeyError:
        raise KeyError(
            "no 'id' column in the result row. This happens when execute() runs a "
            "statement pghybrid did not build; pass the sql and params from "
            "HybridSearch.build_sql() through unchanged."
        ) from None

    return SearchResult(
        id=identifier,
        score=_as_score(mapping.get("score")),
        fused_score=_as_score(mapping.get("fused_score")),
        vector_rank=_as_int(mapping.get("vector_rank")),
        vector_distance=as_float(mapping.get("vector_distance")),
        vector_contribution=_as_score(mapping.get("vector_contribution")),
        text_rank=_as_int(mapping.get("text_rank")),
        text_score=as_float(mapping.get("text_score")),
        text_contribution=_as_score(mapping.get("text_contribution")),
        recency_factor=as_float(mapping.get("recency_factor")),
        highlight=mapping.get("highlight"),
        row={key: value for key, value in mapping.items() if key not in _SIGNAL_COLUMNS},
    )


def results_from_rows(rows: Iterable[Any]) -> list[SearchResult]:
    """Shape a whole result set. The ordering the database produced is preserved."""
    if rows is None:
        return []
    return [result_from_row(row) for row in rows]


def _normalise_text(text: str | None) -> str | None:
    """Treat a blank search box as no text signal at all.

    An empty tsquery matches nothing, so passing ``""`` through would build a text CTE
    that can only ever be empty and a ts_headline call over it. Dropping the signal
    gives the same rows for less work, and makes a blank query with no embedding fail
    loudly rather than return an empty list that looks like a relevance problem.
    """
    if text is None:
        return None
    return text if text.strip() else None


class _SearchBase:
    """State and statement building shared by the sync and async clients."""

    def __init__(self, config: Config, execute: Any) -> None:
        if not isinstance(config, Config):
            raise TypeError(f"config must be a pghybrid.Config, got {type(config).__name__}")
        if not callable(execute):
            raise TypeError(
                "execute must be callable as execute(sql, params). For psycopg: "
                "lambda sql, args: conn.execute(sql, args).fetchall()"
            )
        self.config = config
        self.execute = execute

    def build_sql(
        self,
        text: str | None = None,
        embedding: list[float] | None = None,
        *,
        limit: int = 10,
        offset: int = 0,
        filters: dict[str, Any] | None = None,
        candidate_limit: int | None = None,
        near_miss: int = 0,
        highlight: bool = False,
        fusion: FusionMethod | None = None,
    ) -> tuple[str, list[Any]]:
        """The statement :meth:`search` would run, without running it.

        Worth exposing: the fastest way to debug a ranking is to paste the query into
        psql and edit it, and the fastest way to trust a library is to read what it
        sends.
        """
        return build_search_sql(
            self.config,
            embedding=embedding,
            text=_normalise_text(text),
            limit=limit,
            offset=offset,
            filters=filters,
            candidate_limit=candidate_limit,
            near_miss=near_miss,
            highlight=highlight,
            fusion=fusion,
        )


class HybridSearch(_SearchBase):
    """Hybrid search over one table, driven by a synchronous ``execute`` callable.

    ``execute(sql, params)`` must run the statement and return the rows as mappings.
    Everything else — connecting, pooling, retrying, tracing — stays in the caller's
    code where it belongs.
    """

    def __init__(self, config: Config, execute: Executor) -> None:
        super().__init__(config, execute)

    def search(
        self,
        text: str | None = None,
        embedding: list[float] | None = None,
        *,
        limit: int = 10,
        offset: int = 0,
        filters: dict[str, Any] | None = None,
        candidate_limit: int | None = None,
        highlight: bool = False,
        fusion: FusionMethod | None = None,
    ) -> list[SearchResult]:
        """Rank rows by both signals at once.

        Passing only ``text`` runs a pure full-text search and only ``embedding`` a
        pure vector search, both returning the same shape, which is what makes the
        three-way comparison in :mod:`pghybrid.explain` an apples-to-apples one.
        """
        sql, params = self.build_sql(
            text,
            embedding,
            limit=limit,
            offset=offset,
            filters=filters,
            candidate_limit=candidate_limit,
            highlight=highlight,
            fusion=fusion,
        )
        return results_from_rows(self.execute(sql, params))

    def explain(
        self, text: str | None = None, embedding: list[float] | None = None, **kwargs: Any
    ) -> Any:
        """Diagnose one query. See :func:`pghybrid.explain.explain` for the arguments."""
        # Imported here rather than at module scope so the dependency runs one way:
        # the diagnostic layer is allowed to know about the runtime, not the reverse.
        from .explain import explain as _explain

        return _explain(self, text, embedding, **kwargs)

    def effective_weights(
        self,
        text: str | None = None,
        embedding: list[float] | None = None,
        *,
        fusion: FusionMethod | None = None,
        **kwargs: Any,
    ) -> Any:
        """Measure what the configured weights actually do to this query.

        Returns the :class:`~pghybrid.explain.WeightReport` for one fusion method:
        nominal weight share against the share of the score range each signal really
        controls. See :func:`pghybrid.explain.explain` for the full report.
        """
        from .explain import measure_weights

        return measure_weights(self, text, embedding, fusion=fusion, **kwargs)


class AsyncHybridSearch(_SearchBase):
    """Hybrid search driven by an awaitable ``execute`` callable.

    Identical to :class:`HybridSearch` apart from the awaits; the statement building
    and the row shaping are the same code, so the two clients cannot drift.
    """

    def __init__(self, config: Config, execute: AsyncExecutor) -> None:
        super().__init__(config, execute)

    async def search(
        self,
        text: str | None = None,
        embedding: list[float] | None = None,
        *,
        limit: int = 10,
        offset: int = 0,
        filters: dict[str, Any] | None = None,
        candidate_limit: int | None = None,
        highlight: bool = False,
        fusion: FusionMethod | None = None,
    ) -> list[SearchResult]:
        """Await the query and return the same :class:`SearchResult` list as the sync client."""
        sql, params = self.build_sql(
            text,
            embedding,
            limit=limit,
            offset=offset,
            filters=filters,
            candidate_limit=candidate_limit,
            highlight=highlight,
            fusion=fusion,
        )
        return results_from_rows(await self.execute(sql, params))

    async def explain(
        self, text: str | None = None, embedding: list[float] | None = None, **kwargs: Any
    ) -> Any:
        """Diagnose one query. See :func:`pghybrid.explain.explain_async`."""
        from .explain import explain_async

        return await explain_async(self, text, embedding, **kwargs)

    async def effective_weights(
        self,
        text: str | None = None,
        embedding: list[float] | None = None,
        *,
        fusion: FusionMethod | None = None,
        **kwargs: Any,
    ) -> Any:
        """Await the weight measurement described in :meth:`HybridSearch.effective_weights`."""
        from .explain import measure_weights_async

        return await measure_weights_async(self, text, embedding, fusion=fusion, **kwargs)


#: Either client, for code that accepts whichever one the caller built.
AnySearch = Union[HybridSearch, AsyncHybridSearch]

__all__ = [
    "AnySearch",
    "AsyncExecutor",
    "AsyncHybridSearch",
    "Executor",
    "HybridSearch",
    "SearchResult",
    "as_float",
    "result_from_row",
    "results_from_rows",
    "row_mapping",
]
