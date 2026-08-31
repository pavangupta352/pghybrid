"""Why this row and not that one.

Hybrid search fails quietly. The query runs, ten rows come back, and the one the user
wanted sits at position eleven where nobody looks. This module answers the two
questions that follow, both of which are hard to answer from a result set alone:

**Where did my chunk go?** The report ranks past the cut-off and marks the near-miss
band — the rows between ``limit`` and ``limit + near_miss``. That band is where a
retrieval bug usually lives, and it is invisible in production because the application
throws it away. If the expected chunk is not there either, :func:`explain` can look it
up directly and say whether it is outranked or simply not indexed. Those are different
bugs with different fixes, and confusing them costs days.

**Do my weights mean what I think?** Under ``weighted`` fusion people write
``0.7 * (1 - cosine_distance) + 0.3 * ts_rank`` and believe they configured 70/30. They
did not. Cosine distance is bounded and clusters tightly — the top candidates often all
sit within a few hundredths of each other — while ``ts_rank_cd`` is unbounded and tiny.
What decides the ordering is not each term's weight but the *range* each weighted term
covers across the candidates, and those ranges are not comparable. The report measures
both, for both fusion methods, so the gap is a number instead of an argument.

Everything here is read-only and driver-agnostic: it runs through the same ``execute``
callable as :class:`~pghybrid.search.HybridSearch`.
"""

from __future__ import annotations

import textwrap
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .config import Config
from .search import SearchResult, _normalise_text, as_float, results_from_rows, row_mapping
from .sql import (
    FusionMethod,
    Params,
    _distance_expr,
    _filter_sql,
    _format_vector,
    _tsquery_expr,
    _tsvector_expr,
    quote_ident,
)

if TYPE_CHECKING:  # pragma: no cover - import cycle exists only for type checkers
    from .search import AsyncHybridSearch, HybridSearch

FUSION_METHODS: tuple[FusionMethod, ...] = ("rrf", "weighted")

#: Columns worth using as a row label in the report, in order of preference. A report
#: that prints 400 characters of chunk body per row is unreadable, and every corpus
#: this library is aimed at has a short human name for a row somewhere.
_LABEL_CANDIDATES = ("title", "name", "heading", "subject", "path", "url")


# ======================================================================================
# The report
# ======================================================================================


@dataclass(frozen=True)
class ExplainRow:
    """One fused candidate, with both signals kept separate.

    ``vector_rank`` and ``text_rank`` are None when that signal never retrieved the
    row. A row that only one signal found is the normal case in hybrid search, not an
    error, and it is usually the interesting one.
    """

    position: int
    id: Any
    label: str
    score: float
    fused_score: float
    vector_rank: int | None
    vector_distance: float | None
    vector_contribution: float
    text_rank: int | None
    text_score: float | None
    text_contribution: float
    recency_factor: float | None
    #: True for rows ranked below ``limit`` but inside the near-miss band.
    near_miss: bool
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


@dataclass(frozen=True)
class SignalWeights:
    """What one signal's weighted term actually did to the ranking.

    ``span`` is the distance between the largest and smallest contribution this term
    made across the rows it retrieved. It is the number that matters for ordering: a
    term that is the same for every row cannot change anyone's position no matter how
    large its weight, and a term that swings widely decides the ranking no matter how
    small its weight.
    """

    signal: str
    weight: float
    #: ``weight / (vector weight + text weight)`` — the split the user thinks they set.
    nominal_share: float
    #: How many of the fused candidates this signal retrieved, and therefore how many
    #: rows the span was measured over. The rest scored zero from this signal, which is
    #: a larger jump than any difference within the span.
    matched: int
    low: float
    high: float
    span: float
    #: ``span / total span`` — the split they actually got. None when no candidate
    #: separated from any other, which makes the question meaningless rather than 50/50.
    effective_share: float | None


@dataclass(frozen=True)
class WeightReport:
    """The nominal-versus-effective comparison for one fusion method."""

    fusion: str
    vector: SignalWeights
    text: SignalWeights
    #: How many fused candidates the spans were measured over.
    candidates: int

    @property
    def distortion(self) -> float | None:
        """How far the effective split sits from the configured one, 0.0 to 1.0.

        Zero means the weights do what they say. Under RRF this is small by
        construction, because both terms are ``weight / (k + rank)`` and therefore
        share a scale. Under ``weighted`` fusion it routinely exceeds 0.3.
        """
        if self.vector.effective_share is None:
            return None
        return abs(self.vector.effective_share - self.vector.nominal_share)


@dataclass(frozen=True)
class CandidateStats:
    """Coverage of the candidate set, before fusion arithmetic gets involved."""

    fused: int
    vector_matched: int
    text_matched: int
    both_matched: int
    vector_distance_low: float | None = None
    vector_distance_high: float | None = None
    text_score_low: float | None = None
    text_score_high: float | None = None


@dataclass(frozen=True)
class FindReport:
    """Where a specific piece of text ended up, and why.

    ``found`` answers "is this text in the table at all". ``position`` answers "did the
    query retrieve it". A row with ``found=True`` and ``position=None`` was outranked
    or filtered out; ``found=False`` means the chunk was never indexed. The first is a
    tuning problem, the second is an ingestion bug, and the whole point of this report
    is that you no longer have to guess which one you have.
    """

    query: str
    found: bool
    id: Any = None
    label: str | None = None
    #: Position in the fused candidate list, or None if the query never retrieved it.
    position: int | None = None
    returned: bool = False
    near_miss: bool = False
    vector_rank: int | None = None
    vector_distance: float | None = None
    text_rank: int | None = None
    text_score: float | None = None
    #: True when the row's vector column is NULL, which no amount of tuning will fix.
    embedding_missing: bool = False
    text_matches: bool = False
    passes_filters: bool = True
    #: How many rows in the table contain the searched text.
    match_count: int = 0
    reason: str = ""
    remedy: str | None = None


@dataclass(frozen=True)
class ExplainReport:
    """The full diagnosis of one query.

    ``rows`` is the visible window: the returned rows plus the near-miss band.
    ``candidates`` is everything the fusion considered, which is what the weight
    measurement and :attr:`find` are computed over.
    """

    config: Config
    query: str | None
    embedding_dimensions: int | None
    fusion: str
    limit: int
    near_miss: int
    candidate_limit: int
    rows: list[ExplainRow]
    candidates: list[ExplainRow]
    stats: CandidateStats
    weights: Mapping[str, WeightReport]
    find: FindReport | None
    label_column: str
    #: The statement that produced ``candidates``, ready to paste into psql.
    sql: str
    params: list[Any]

    @property
    def returned(self) -> list[ExplainRow]:
        """The rows the query would actually have given the application."""
        return [row for row in self.rows if not row.near_miss]

    @property
    def near_miss_rows(self) -> list[ExplainRow]:
        """The rows just below the cut-off — the ones production never shows anyone."""
        return [row for row in self.rows if row.near_miss]

    @property
    def effective_weights(self) -> WeightReport:
        """The weight measurement for the fusion method this query actually used."""
        return self.weights[self.fusion]

    def row_for(self, identifier: Any) -> ExplainRow | None:
        """The candidate with this id, or None if the query never retrieved it."""
        for row in self.candidates:
            if row.id == identifier:
                return row
        return None

    def to_text(self, *, width: int = 104, ascii_only: bool = False) -> str:
        """Render the report as a fixed-width table.

        No colour codes and no dependencies, so the output survives a pipe, a log
        aggregator, a pull request comment and a screenshot equally well.
        """
        return _render(self, width=width, ascii_only=ascii_only)

    def __str__(self) -> str:
        return self.to_text()


# ======================================================================================
# Planning and execution
#
# The database work is two statements: the same search run under each fusion method,
# both to the full candidate depth. A third statement runs only when a `find` target is
# missing from the candidate set, which is exactly when its cost is worth paying.
# ======================================================================================


@dataclass(frozen=True)
class _Plan:
    config: Config
    text: str | None
    embedding: list[float] | None
    filters: dict[str, Any] | None
    limit: int
    near_miss: int
    #: The candidate depth the search would really use, after the widening below.
    candidate_limit: int
    fusion: str
    label_column: str
    find: str | None
    queries: Mapping[str, tuple[str, list[Any]]]


def _plan(
    search: HybridSearch | AsyncHybridSearch,
    text: str | None,
    embedding: list[float] | None,
    *,
    limit: int,
    near_miss: int,
    filters: dict[str, Any] | None,
    candidate_limit: int | None,
    fusion: FusionMethod | None,
    label_column: str | None,
    find: str | None,
    methods: Sequence[FusionMethod] = FUSION_METHODS,
) -> _Plan:
    """Resolve the arguments and build the statements, without running anything."""
    cfg: Config = search.config
    # Normalised here as well as in the client so that the lookup query, the report
    # header and the search all agree on whether there is a text signal at all.
    text = _normalise_text(text)
    if limit < 1:
        raise ValueError("limit must be >= 1")
    if near_miss < 0:
        raise ValueError("near_miss must be >= 0")

    chosen = fusion or cfg.fusion
    if chosen not in FUSION_METHODS:
        raise ValueError(f"unknown fusion method {chosen!r}; expected 'rrf' or 'weighted'")

    resolved_candidates = candidate_limit or cfg.candidate_limit
    # The search itself raises candidate_limit when the window is wider than it, so the
    # report has to do the same or it would describe a query nobody runs.
    depth = max(resolved_candidates, limit + near_miss)

    # Fetching the whole candidate set rather than the visible window costs nothing
    # extra in the database — the candidate CTEs are built either way — and it is what
    # lets `find` distinguish "ranked 40th" from "never retrieved".
    queries: dict[str, tuple[str, list[Any]]] = {
        method: search.build_sql(
            text,
            embedding,
            limit=depth,
            candidate_limit=depth,
            filters=filters,
            fusion=method,
        )
        for method in methods
    }

    return _Plan(
        config=cfg,
        text=text,
        embedding=list(embedding) if embedding is not None else None,
        filters=filters,
        limit=limit,
        near_miss=near_miss,
        candidate_limit=depth,
        fusion=chosen,
        label_column=label_column or _label_column(cfg),
        find=find,
        queries=queries,
    )


def explain(
    search: HybridSearch,
    text: str | None = None,
    embedding: list[float] | None = None,
    *,
    limit: int = 10,
    near_miss: int = 10,
    filters: dict[str, Any] | None = None,
    candidate_limit: int | None = None,
    fusion: FusionMethod | None = None,
    label_column: str | None = None,
    find: str | None = None,
) -> ExplainReport:
    """Diagnose one hybrid search.

    ``near_miss`` extends the report past ``limit`` so the rows that just missed the
    cut are visible; they are the usual home of a "search is broken" report.

    ``find`` takes the text the caller expected to be retrieved — a title, or any
    distinctive phrase from the chunk. If it is not among the candidates, one extra
    statement locates it in the table and reports its rank under each signal. That
    statement scans, because the whole question is about a row the indexes did not
    return, so pass ``find`` when you are debugging rather than on every request.

    Costs two statements — the same search under each fusion method, so the weight
    comparison is measured rather than modelled — plus that lookup when it is needed.
    """
    plan = _plan(
        search,
        text,
        embedding,
        limit=limit,
        near_miss=near_miss,
        filters=filters,
        candidate_limit=candidate_limit,
        fusion=fusion,
        label_column=label_column,
        find=find,
    )
    raw = {
        method: results_from_rows(search.execute(sql, params))
        for method, (sql, params) in plan.queries.items()
    }
    report = _assemble(plan, raw)

    if plan.find is not None and (report.find is None or not report.find.found):
        sql, params = _locate_sql(plan)
        report = _with_find(report, plan, search.execute(sql, params))
    return report


async def explain_async(
    search: AsyncHybridSearch,
    text: str | None = None,
    embedding: list[float] | None = None,
    *,
    limit: int = 10,
    near_miss: int = 10,
    filters: dict[str, Any] | None = None,
    candidate_limit: int | None = None,
    fusion: FusionMethod | None = None,
    label_column: str | None = None,
    find: str | None = None,
) -> ExplainReport:
    """:func:`explain` for an :class:`~pghybrid.search.AsyncHybridSearch`.

    Only the awaits differ; the planning, measurement and rendering are the same code.
    """
    plan = _plan(
        search,
        text,
        embedding,
        limit=limit,
        near_miss=near_miss,
        filters=filters,
        candidate_limit=candidate_limit,
        fusion=fusion,
        label_column=label_column,
        find=find,
    )
    raw = {}
    for method, (sql, params) in plan.queries.items():
        raw[method] = results_from_rows(await search.execute(sql, params))
    report = _assemble(plan, raw)

    if plan.find is not None and (report.find is None or not report.find.found):
        sql, params = _locate_sql(plan)
        report = _with_find(report, plan, await search.execute(sql, params))
    return report


def measure_weights(
    search: HybridSearch,
    text: str | None = None,
    embedding: list[float] | None = None,
    *,
    fusion: FusionMethod | None = None,
    filters: dict[str, Any] | None = None,
    candidate_limit: int | None = None,
) -> WeightReport:
    """Measure nominal against effective weights for one fusion method.

    A single statement, for callers who want the number without the table.
    """
    method = fusion or search.config.fusion
    plan = _plan(
        search,
        text,
        embedding,
        limit=1,
        near_miss=0,
        filters=filters,
        candidate_limit=candidate_limit,
        fusion=method,
        label_column=None,
        find=None,
        methods=(method,),
    )
    sql, params = plan.queries[plan.fusion]
    rows = results_from_rows(search.execute(sql, params))
    return _measure(plan.config, plan.fusion, rows)


async def measure_weights_async(
    search: AsyncHybridSearch,
    text: str | None = None,
    embedding: list[float] | None = None,
    *,
    fusion: FusionMethod | None = None,
    filters: dict[str, Any] | None = None,
    candidate_limit: int | None = None,
) -> WeightReport:
    """:func:`measure_weights` for an :class:`~pghybrid.search.AsyncHybridSearch`."""
    method = fusion or search.config.fusion
    plan = _plan(
        search,
        text,
        embedding,
        limit=1,
        near_miss=0,
        filters=filters,
        candidate_limit=candidate_limit,
        fusion=method,
        label_column=None,
        find=None,
        methods=(method,),
    )
    sql, params = plan.queries[plan.fusion]
    rows = results_from_rows(await search.execute(sql, params))
    return _measure(plan.config, plan.fusion, rows)


# ======================================================================================
# Assembly — pure functions over rows the caller already fetched
# ======================================================================================


def _label_column(cfg: Config) -> str:
    """The column to print as a row's name.

    Falls back to the searched text column, which is always present and always
    readable even when it is long.
    """
    for column in _LABEL_CANDIDATES:
        if column in cfg.extra_columns:
            return column
    return cfg.text_column


def _label(result: SearchResult, column: str) -> str:
    value = result.row.get(column)
    if value is None:
        value = result.row.get("title") or result.id
    return " ".join(str(value).split())


def _explain_rows(plan: _Plan, results: Sequence[SearchResult]) -> list[ExplainRow]:
    window = plan.limit + plan.near_miss
    rows = []
    for index, result in enumerate(results):
        position = index + 1
        rows.append(
            ExplainRow(
                position=position,
                id=result.id,
                label=_label(result, plan.label_column),
                score=result.score,
                fused_score=result.fused_score,
                vector_rank=result.vector_rank,
                vector_distance=result.vector_distance,
                vector_contribution=result.vector_contribution,
                text_rank=result.text_rank,
                text_score=result.text_score,
                text_contribution=result.text_contribution,
                recency_factor=result.recency_factor,
                near_miss=plan.limit < position <= window,
                row=result.row,
            )
        )
    return rows


def _stats(rows: Sequence[ExplainRow]) -> CandidateStats:
    distances = [row.vector_distance for row in rows if row.vector_distance is not None]
    scores = [row.text_score for row in rows if row.text_score is not None]
    return CandidateStats(
        fused=len(rows),
        vector_matched=sum(1 for row in rows if row.vector_rank is not None),
        text_matched=sum(1 for row in rows if row.text_rank is not None),
        both_matched=sum(1 for row in rows if row.matched_by == "both"),
        vector_distance_low=min(distances) if distances else None,
        vector_distance_high=max(distances) if distances else None,
        text_score_low=min(scores) if scores else None,
        text_score_high=max(scores) if scores else None,
    )


def _bounds(values: Sequence[float]) -> tuple[float, float]:
    """Low and high of a signal's contributions; a signal that never fired spans nothing."""
    return (min(values), max(values)) if values else (0.0, 0.0)


def _measure(cfg: Config, fusion: str, results: Sequence[SearchResult]) -> WeightReport:
    """Measure each signal's real influence over the candidate set.

    The measurement is deliberately taken on the contribution columns the database
    computed, not on a reconstruction of the formula in Python: the point is to report
    what the query did, and a second implementation of the arithmetic would eventually
    disagree with the first.

    Each span covers the rows that signal actually retrieved. Rows it did not retrieve
    contribute a coalesced zero, and folding those zeros in would break the measurement
    twice over: under RRF every span would collapse to ``weight / (k + 1)``, making the
    effective share identically equal to the nominal one for every query ever run — a
    tautology dressed up as a result — and under ``weighted`` fusion the reported range
    would be the magnitude of the signal rather than its spread, which is the opposite
    of the thing that decides an ordering. The zeros are still worth knowing about, so
    :attr:`SignalWeights.matched` reports how many rows each span was measured over.
    """
    vector_values = [
        result.vector_contribution for result in results if result.vector_rank is not None
    ]
    text_values = [result.text_contribution for result in results if result.text_rank is not None]

    vector_weight = float(cfg.weights.vector)
    text_weight = float(cfg.weights.text)
    total_weight = vector_weight + text_weight

    vector_low, vector_high = _bounds(vector_values)
    text_low, text_high = _bounds(text_values)
    vector_span = vector_high - vector_low
    text_span = text_high - text_low
    total_span = vector_span + text_span

    def share(span: float) -> float | None:
        # Every candidate scoring identically is a real state (one candidate, or a
        # degenerate query) and it has no answer, so say so instead of inventing 50/50.
        return None if total_span <= 0 else span / total_span

    return WeightReport(
        fusion=fusion,
        vector=SignalWeights(
            signal="vector",
            weight=vector_weight,
            nominal_share=vector_weight / total_weight if total_weight else 0.0,
            matched=len(vector_values),
            low=vector_low,
            high=vector_high,
            span=vector_span,
            effective_share=share(vector_span),
        ),
        text=SignalWeights(
            signal="text",
            weight=text_weight,
            nominal_share=text_weight / total_weight if total_weight else 0.0,
            matched=len(text_values),
            low=text_low,
            high=text_high,
            span=text_span,
            effective_share=share(text_span),
        ),
        candidates=len(results),
    )


def _assemble(plan: _Plan, raw: Mapping[str, list[SearchResult]]) -> ExplainReport:
    active = raw[plan.fusion]
    candidates = _explain_rows(plan, active)
    window = plan.limit + plan.near_miss
    sql, params = plan.queries[plan.fusion]

    report = ExplainReport(
        config=plan.config,
        query=plan.text,
        embedding_dimensions=len(plan.embedding) if plan.embedding is not None else None,
        fusion=plan.fusion,
        limit=plan.limit,
        near_miss=plan.near_miss,
        candidate_limit=plan.candidate_limit,
        rows=candidates[:window],
        candidates=candidates,
        stats=_stats(candidates),
        weights={method: _measure(plan.config, method, results) for method, results in raw.items()},
        find=None,
        label_column=plan.label_column,
        sql=sql,
        params=params,
    )
    if plan.find is None:
        return report
    return _replace_find(report, _find_in_candidates(report, plan.find))


def _replace_find(report: ExplainReport, finding: FindReport | None) -> ExplainReport:
    """Rebuild the frozen report with a find result attached."""
    return ExplainReport(
        config=report.config,
        query=report.query,
        embedding_dimensions=report.embedding_dimensions,
        fusion=report.fusion,
        limit=report.limit,
        near_miss=report.near_miss,
        candidate_limit=report.candidate_limit,
        rows=report.rows,
        candidates=report.candidates,
        stats=report.stats,
        weights=report.weights,
        find=finding,
        label_column=report.label_column,
        sql=report.sql,
        params=report.params,
    )


# ======================================================================================
# find — "I know the answer is in there; where did it go?"
# ======================================================================================


def _matches_text(row: ExplainRow, needle: str) -> bool:
    lowered = needle.lower()
    if row.label.lower() == lowered:
        return True
    return any(isinstance(value, str) and lowered in value.lower() for value in row.row.values())


def _find_in_candidates(report: ExplainReport, needle: str) -> FindReport | None:
    """Locate the expected text among the rows the query already returned.

    Checked before going back to the database, because the common case — the chunk was
    retrieved and simply ranked too low — needs no further query, and because ranks
    inside the candidate set are already the global ranks: the candidate CTE ranks the
    top ``candidate_limit`` rows of the whole table.
    """
    target = needle.strip()
    if not target:
        return None

    exact = [row for row in report.candidates if row.label.strip().lower() == target.lower()]
    matched = (
        exact[0]
        if exact
        else next((row for row in report.candidates if _matches_text(row, target)), None)
    )
    if matched is None:
        return None

    if matched.position <= report.limit:
        reason = f"returned at #{matched.position} — the query found it"
        remedy = None
    elif matched.near_miss:
        reason = f"#{matched.position} in the near-miss band, just outside the top {report.limit}"
        remedy = f"raising limit to {matched.position} would return it"
    else:
        reason = (
            f"fused at #{matched.position} of {len(report.candidates)} candidates, "
            f"below the near-miss band"
        )
        remedy = (
            f"widen the report with near_miss={matched.position - report.limit} to see "
            "what outranks it"
        )

    return FindReport(
        query=needle,
        found=True,
        id=matched.id,
        label=matched.label,
        position=matched.position,
        returned=matched.position <= report.limit,
        near_miss=matched.near_miss,
        vector_rank=matched.vector_rank,
        vector_distance=matched.vector_distance,
        text_rank=matched.text_rank,
        text_score=matched.text_score,
        embedding_missing=False,
        text_matches=matched.text_rank is not None,
        passes_filters=True,
        match_count=1,
        reason=reason,
        remedy=remedy,
    )


def _searchable_columns(cfg: Config) -> list[str]:
    """Columns the lookup will search for the expected text."""
    columns: list[str] = []
    for column in [cfg.text_column, *cfg.extra_columns]:
        if column and column not in columns:
            columns.append(column)
    return columns


def _locate_sql(plan: _Plan) -> tuple[str, list[Any]]:
    """Find the expected row in the table, ignoring both indexes and both cut-offs.

    This statement answers the question the search query cannot: does the row exist, is
    it excluded by a filter, does it have an embedding at all, and where would it rank
    globally under each signal. The global ranks are counts of better rows rather than
    an ORDER BY, which keeps it to one pass per signal, but it is still a scan — it runs
    only when the row is missing from the candidate set.

    The tsquery and distance expressions are borrowed from the SQL builder rather than
    rewritten. A lookup that parsed the query differently from the search would report
    ranks the search could never produce, which is worse than no report at all.
    """
    cfg = plan.config
    params = Params()
    table = quote_ident(cfg.table)
    id_col = quote_ident(cfg.id_column)
    needle = params.add(plan.find)

    columns = _searchable_columns(cfg)
    contains = " OR ".join(
        f"strpos(lower(coalesce({quote_ident(column)}::text, '')), lower({needle})) > 0"
        for column in columns
    )
    exact = " OR ".join(
        f"coalesce({quote_ident(column)}::text, '') = {needle}" for column in columns
    )

    # Filters are reported, not applied: a row hidden by a filter is a distinct and very
    # common failure, and silently returning "not found" for it would hide the cause.
    filter_clause = _filter_sql(cfg, plan.filters, params)
    filter_expr = filter_clause[len(" AND ") :] if filter_clause else "TRUE"
    # Postgres reads a bare constant in ORDER BY as a column ordinal, so the tie-break
    # on the filter can only be added when it is a real expression.
    filter_order = f", ({filter_expr}) DESC" if filter_clause else ""

    # Every CTE here is prefixed: an unprefixed name like `matches` or `target` would
    # shadow a real table of that name, because a CTE outranks a table in the same
    # statement whether or not the table name is quoted.
    ctes: list[str] = []
    target_from = table

    if plan.text is not None:
        ctes.append(f"_find_q AS (\n    SELECT {_tsquery_expr(cfg, plan.text, params)} AS tsq\n)")
        target_from = f"{table}, _find_q"
        tsv = _tsvector_expr(cfg)
        text_matches = f"({tsv} @@ _find_q.tsq)"
        text_score = f"{cfg.rank_function}({tsv}, _find_q.tsq)"
    else:
        text_matches = "NULL::boolean"
        text_score = "NULL::double precision"

    if plan.embedding is not None:
        vector = params.add_cast(_format_vector(plan.embedding), cfg.vector_type)
        distance = _distance_expr(cfg, vector)
    else:
        distance = "NULL::double precision"

    label = quote_ident(plan.label_column)
    ctes.append(
        "_find_target AS (\n"
        f"    SELECT {id_col} AS id,\n"
        f"           left(coalesce({label}::text, ''), 200) AS label,\n"
        f"           ({quote_ident(cfg.vector_column)} IS NULL) AS embedding_missing,\n"
        f"           {distance} AS distance,\n"
        f"           {text_matches} AS text_matches,\n"
        f"           {text_score} AS text_score,\n"
        f"           ({filter_expr}) AS passes_filters\n"
        f"    FROM {target_from}\n"
        f"    WHERE {contains}\n"
        f"    ORDER BY ({exact}) DESC{filter_order}, {id_col}\n"
        "    LIMIT 1\n"
        ")"
    )
    ctes.append(
        f"_find_matches AS (\n    SELECT count(*) AS n\n    FROM {table}\n    WHERE {contains}\n)"
    )

    if plan.embedding is not None:
        ctes.append(
            "_find_vector_rank AS (\n"
            "    SELECT count(*) + 1 AS rank\n"
            f"    FROM {table}\n"
            f"    WHERE {quote_ident(cfg.vector_column)} IS NOT NULL\n"
            f"      AND {distance} < (SELECT distance FROM _find_target){filter_clause}\n"
            ")"
        )
    else:
        ctes.append("_find_vector_rank AS (\n    SELECT NULL::bigint AS rank\n)")

    if plan.text is not None:
        tsv = _tsvector_expr(cfg)
        ctes.append(
            "_find_text_rank AS (\n"
            "    SELECT count(*) + 1 AS rank\n"
            f"    FROM {table}, _find_q\n"
            f"    WHERE {tsv} @@ _find_q.tsq\n"
            f"      AND {cfg.rank_function}({tsv}, _find_q.tsq) > "
            f"(SELECT text_score FROM _find_target){filter_clause}\n"
            ")"
        )
    else:
        ctes.append("_find_text_rank AS (\n    SELECT NULL::bigint AS rank\n)")

    sql = (
        "WITH " + ",\n".join(ctes) + "\n"
        "SELECT t.id,\n"
        "       t.label,\n"
        "       t.embedding_missing,\n"
        "       t.distance,\n"
        "       t.text_matches,\n"
        "       t.text_score,\n"
        "       t.passes_filters,\n"
        "       (SELECT n FROM _find_matches) AS match_count,\n"
        "       (SELECT rank FROM _find_vector_rank) AS global_vector_rank,\n"
        "       (SELECT rank FROM _find_text_rank) AS global_text_rank\n"
        "FROM _find_target t"
    )
    return params.render(sql, cfg.paramstyle)


def _with_find(report: ExplainReport, plan: _Plan, rows: Any) -> ExplainReport:
    """Turn the lookup's answer into a diagnosis."""
    needle = plan.find or ""
    materialised = list(rows) if rows is not None else []
    if not materialised:
        columns = ", ".join(_searchable_columns(plan.config))
        return _replace_find(
            report,
            FindReport(
                query=needle,
                found=False,
                match_count=0,
                reason=f"no row in {plan.config.table} contains that text (searched {columns})",
                remedy=(
                    "the chunk is not in the table, so no amount of ranking will "
                    "retrieve it — check ingestion and chunking, not the weights"
                ),
            ),
        )

    row = row_mapping(materialised[0])
    identifier = row.get("id")
    embedding_missing = bool(row.get("embedding_missing"))
    text_matches = bool(row.get("text_matches"))
    passes_filters = bool(row.get("passes_filters"))
    match_count = int(row.get("match_count") or 1)

    # A rank is only meaningful when the signal could produce one, and when the row is
    # part of the population the rank was counted over. Both global ranks are counts of
    # better rows *within the filters*, so a row the filters exclude would come back as
    # rank 1 — the exact opposite of the truth — as would a row whose distance is NULL
    # because it has no embedding.
    rankable = passes_filters
    raw_vector_rank = row.get("global_vector_rank")
    vector_rank = (
        int(raw_vector_rank)
        if raw_vector_rank is not None and rankable and not embedding_missing
        else None
    )
    raw_text_rank = row.get("global_text_rank")
    text_rank = (
        int(raw_text_rank) if raw_text_rank is not None and rankable and text_matches else None
    )

    reason, remedy = _diagnose(
        report,
        identifier=identifier,
        embedding_missing=embedding_missing,
        text_matches=text_matches,
        passes_filters=passes_filters,
        vector_rank=vector_rank,
        text_rank=text_rank,
    )

    return _replace_find(
        report,
        FindReport(
            query=needle,
            found=True,
            id=identifier,
            label=" ".join(str(row.get("label") or "").split()) or None,
            position=None,
            returned=False,
            near_miss=False,
            vector_rank=vector_rank,
            vector_distance=as_float(row.get("distance")),
            text_rank=text_rank,
            text_score=as_float(row.get("text_score")),
            embedding_missing=embedding_missing,
            text_matches=text_matches,
            passes_filters=passes_filters,
            match_count=match_count,
            reason=reason,
            remedy=remedy,
        ),
    )


def _diagnose(
    report: ExplainReport,
    *,
    identifier: Any,
    embedding_missing: bool,
    text_matches: bool,
    passes_filters: bool,
    vector_rank: int | None,
    text_rank: int | None,
) -> tuple[str, str | None]:
    """Name the one reason the row is not in the result, and what to do about it.

    The distinction that matters is whether any signal could reach the row at all. A
    row that no signal can retrieve is a data problem and no amount of tuning will
    help; a row that both signals rank, just too low, is a tuning problem and the
    cut-off is usually the thing to move.
    """
    cut = report.candidate_limit
    row = f"row {identifier}"

    if not passes_filters:
        return (
            f"{row} exists, but the filters on this query exclude it",
            "the ranking never saw it — check the filter, not the weights",
        )

    queried_vector = report.embedding_dimensions is not None
    queried_text = report.query is not None
    reachable = (queried_vector and not embedding_missing) or (queried_text and text_matches)

    if not reachable:
        causes = []
        if not queried_vector:
            causes.append("this query passed no embedding")
        elif embedding_missing:
            causes.append("the row has no embedding")
        if not queried_text:
            causes.append("this query passed no text")
        elif not text_matches:
            causes.append("it matches no term in the query text")

        if queried_vector and embedding_missing:
            remedy = "backfill the vector column for this row"
        elif not queried_vector:
            remedy = "search with an embedding too; keywords alone cannot reach this row"
        else:
            remedy = "the row shares nothing with this query"
        return f"{row} is unreachable here: {' and '.join(causes)}", remedy

    ranks = [rank for rank in (vector_rank, text_rank) if rank is not None]
    if any(rank <= cut for rank in ranks):
        # It made a candidate list and then lost the fused ordering before this depth.
        return (
            f"{row} entered the candidate set but fell outside the {cut} fused rows",
            f"raise candidate_limit above {cut} to see where it lands",
        )
    return (
        f"{row} ranks past the candidate cut-off of {cut}, so it never entered the fusion",
        f"raise candidate_limit to at least {min(ranks)}",
    )


# ======================================================================================
# Rendering
#
# One fixed-width table, no colour, no dependencies. The layout is built from column
# widths rather than f-string padding so that adding a column cannot silently shift the
# rules and headers out of alignment.
# ======================================================================================

_GLYPHS = {
    "rule": "─",
    "band": "╌",
    "dot": "·",
    "dash": "–",
    "arrow": "→",
    "null": "–",
    "mark": "▸",
    "ellipsis": "…",
    "range": "…",
}
_ASCII_GLYPHS = {
    "rule": "-",
    "band": "-",
    "dot": "*",
    "dash": "-",
    "arrow": "->",
    "null": "-",
    "mark": ">",
    "ellipsis": "...",
    "range": "..",
}

_INDENT = "  "
_GAP = "  "


@dataclass(frozen=True)
class _Col:
    key: str
    header: str
    width: int
    align: str = ">"
    group: str = ""


def _num(value: float | None, null: str = "-") -> str:
    """Fixed-width-friendly numbers.

    Five decimal places for the ordinary case, because RRF contributions differ in the
    fourth, and scientific notation at the extremes so a tiny ts_rank never renders as
    a column of zeroes.
    """
    if value is None:
        return null
    number = float(value)
    if number == 0:
        return "0"
    magnitude = abs(number)
    if magnitude < 1e-4 or magnitude >= 1e5:
        return f"{number:.1e}"
    if magnitude < 10:
        return f"{number:.5f}"
    if magnitude < 1000:
        return f"{number:.3f}"
    return f"{number:.1f}"


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _weight(value: float) -> str:
    """A weight as a person would write it: 1.0, 0.7, 2.5."""
    text = f"{value:g}"
    return f"{text}.0" if "." not in text and "e" not in text else text


def _wrap(text: str | None, width: int, indent: str, hanging: str) -> list[str]:
    """Wrap one prose line of the report.

    Only the diagnosis prose is wrapped, never the table: row ids, table names and
    labels all vary in length, so a sentence built around them cannot be sized by
    hand. Long words are left to overflow rather than split, because the long word is
    usually an identifier and a broken identifier cannot be searched for.
    """
    if not text:
        return []
    return textwrap.wrap(
        text,
        width=max(40, width),
        initial_indent=indent,
        subsequent_indent=hanging,
        break_long_words=False,
        break_on_hyphens=False,
    )


def _fit(text: str, width: int, align: str, ellipsis: str) -> str:
    text = str(text)
    if len(text) > width:
        text = text[: max(0, width - len(ellipsis))] + ellipsis
    return text.rjust(width) if align == ">" else text.ljust(width)


def _line(cells: Sequence[str], cols: Sequence[_Col], ellipsis: str) -> str:
    return (
        _INDENT
        + _GAP.join(
            _fit(cell, col.width, col.align, ellipsis) for cell, col in zip(cells, cols)
        ).rstrip()
    )


def _table_width(cols: Sequence[_Col]) -> int:
    return sum(col.width for col in cols) + len(_GAP) * (len(cols) - 1)


def _group_line(cols: Sequence[_Col]) -> str:
    """Centre each group's name over the columns it spans."""
    parts: list[str] = []
    index = 0
    while index < len(cols):
        group = cols[index].group
        end = index
        while end + 1 < len(cols) and cols[end + 1].group == group:
            end += 1
        span = _table_width(cols[index : end + 1])
        parts.append(group.center(span) if group else " " * span)
        index = end + 1
    return (_INDENT + _GAP.join(parts)).rstrip()


def _rule(width: int, char: str, label: str = "") -> str:
    if not label:
        return _INDENT + char * width
    text = f" {label} "
    left = max(0, (width - len(text)) // 2)
    right = max(0, width - len(text) - left)
    return _INDENT + char * left + text + char * right


def _columns(report: ExplainReport, width: int) -> list[_Col]:
    id_width = max(2, min(10, max((len(str(row.id)) for row in report.rows), default=2)))
    recency = any(row.recency_factor is not None for row in report.rows)

    fixed = [
        _Col("mark", "", 1, "<"),
        _Col("position", "#", 3),
        _Col("id", "id", id_width),
        _Col("label", report.label_column, 0, "<"),
        _Col("vector_rank", "rank", 4, ">", "vector"),
        _Col("vector_distance", "distance", 8, ">", "vector"),
        _Col("vector_contribution", "contrib", 8, ">", "vector"),
        _Col("text_rank", "rank", 4, ">", "text"),
        # Both rank functions are named ts_rank*; the header names the family and the
        # meta block above says which one is configured.
        _Col("text_score", "ts_rank", 8, ">", "text"),
        _Col("text_contribution", "contrib", 8, ">", "text"),
    ]
    if recency:
        fixed.append(_Col("recency_factor", "decay", 7, ">", "recency"))
    fixed.append(_Col("score", "score", 8, ">", "final"))

    overhead = _table_width(fixed) + len(_INDENT)
    label_width = max(20, min(44, width - overhead))
    return [
        _Col(col.key, col.header, label_width, col.align, col.group) if col.key == "label" else col
        for col in fixed
    ]


def _render(report: ExplainReport, *, width: int, ascii_only: bool) -> str:
    glyphs = _ASCII_GLYPHS if ascii_only else _GLYPHS
    lines: list[str] = []
    lines.extend(_render_header(report, glyphs))
    lines.append("")
    lines.extend(_render_table(report, glyphs, width))
    lines.append("")
    lines.extend(_render_weights(report, glyphs))
    if report.find is not None:
        lines.append("")
        lines.extend(_render_find(report, report.find, glyphs, width))
    return "\n".join(lines)


def _meta(label: str, value: str) -> str:
    return f"{_INDENT}{label:<11} {value}"


def _render_header(report: ExplainReport, glyphs: Mapping[str, str]) -> list[str]:
    cfg = report.config
    dot = f" {glyphs['dot']} "
    arrow = glyphs["arrow"]
    stats = report.stats

    query_bits = []
    if report.query is not None:
        query_bits.append(f'"{report.query}"')
    if report.embedding_dimensions is not None:
        query_bits.append(f"embedding {report.embedding_dimensions} dims")
    if not query_bits:
        query_bits.append("(no signals)")

    fusion_bits = [report.fusion]
    if report.fusion == "rrf":
        fusion_bits.append(f"k {cfg.k}")
    fusion_bits.append(
        f"weights vector {_weight(cfg.weights.vector)} / text {_weight(cfg.weights.text)}"
    )
    if cfg.recency is not None:
        fusion_bits.append(f"half-life {cfg.recency.half_life_days:g}d")

    coverage = [
        f"{report.candidate_limit} per signal {arrow} {stats.fused} fused",
        f"{stats.vector_matched} by vector",
        f"{stats.text_matched} by text",
        f"{stats.both_matched} by both",
    ]

    lines = [
        f"{_INDENT}pghybrid explain {glyphs['dot']} {cfg.table}",
        "",
        _meta("query", dot.join(query_bits)),
        _meta("fusion", dot.join(fusion_bits)),
        _meta("candidates", dot.join(coverage)),
    ]

    ranges = []
    if stats.vector_distance_low is not None:
        ranges.append(
            f"{cfg.metric.name} distance {_num(stats.vector_distance_low)} "
            f"{glyphs['range']} {_num(stats.vector_distance_high)}"
        )
    if stats.text_score_low is not None:
        ranges.append(
            f"{cfg.rank_function} {_num(stats.text_score_low)} "
            f"{glyphs['range']} {_num(stats.text_score_high)}"
        )
    if ranges:
        lines.append(_meta("signals", dot.join(ranges)))

    window = f"top {report.limit}"
    if report.near_miss:
        window += f" {glyphs['dot']} near-miss band of {report.near_miss}"
    lines.append(_meta("window", window))
    return lines


def _render_table(report: ExplainReport, glyphs: Mapping[str, str], width: int) -> list[str]:
    if not report.rows:
        # Column headers over nothing read as a rendering fault rather than a result.
        return [f"{_INDENT}no candidates: neither signal retrieved a row for this query"]

    cols = _columns(report, width)
    table_width = _table_width(cols)
    ellipsis = glyphs["ellipsis"]
    null = glyphs["null"]
    found_id = report.find.id if report.find is not None and report.find.position else None

    lines = [
        _group_line(cols),
        _line([col.header for col in cols], cols, ellipsis),
        _rule(table_width, glyphs["rule"]),
    ]

    last_band_rank = min(report.limit + report.near_miss, len(report.candidates))
    band = (
        f"ranks {report.limit + 1}{glyphs['dash']}{last_band_rank}"
        if last_band_rank > report.limit + 1
        else f"rank {report.limit + 1}"
    )
    for row in report.rows:
        if row.near_miss and (row.position == report.limit + 1):
            lines.append(_rule(table_width, glyphs["band"], f"near miss {glyphs['dot']} {band}"))
        cells = {
            "mark": glyphs["mark"] if found_id is not None and row.id == found_id else "",
            "position": str(row.position),
            "id": str(row.id),
            "label": row.label,
            "vector_rank": str(row.vector_rank) if row.vector_rank is not None else null,
            "vector_distance": _num(row.vector_distance, null),
            "vector_contribution": _num(row.vector_contribution, null),
            "text_rank": str(row.text_rank) if row.text_rank is not None else null,
            "text_score": _num(row.text_score, null),
            "text_contribution": _num(row.text_contribution, null),
            "recency_factor": _num(row.recency_factor, null),
            "score": _num(row.score, null),
        }
        lines.append(_line([cells[col.key] for col in cols], cols, ellipsis))

    remaining = len(report.candidates) - len(report.rows)
    if remaining > 0:
        lines.append(_rule(table_width, glyphs["rule"]))
        lines.append(
            f"{_INDENT}{remaining} further candidate{'s' if remaining != 1 else ''} "
            "fused below this window"
        )
    return lines


_WEIGHT_COLS = [
    _Col("fusion", "fusion", 10, "<"),
    _Col("signal", "signal", 6, "<"),
    _Col("rows", "rows", 6),
    _Col("weight", "weight", 6),
    _Col("nominal", "nominal", 8),
    _Col("range", "contribution range", 21),
    _Col("span", "span", 9),
    _Col("effective", "effective", 9),
]
#: Where the verdict line starts, so it sits under the numbers it summarises.
_VERDICT_INDENT = _WEIGHT_COLS[0].width + len(_GAP)


def _render_weights(report: ExplainReport, glyphs: Mapping[str, str]) -> list[str]:
    ellipsis = glyphs["ellipsis"]
    table_width = _table_width(_WEIGHT_COLS)
    heading = (
        f"{_INDENT}effective weights {glyphs['dot']} the share of the score range each "
        "signal controls"
    )
    if not report.candidates:
        return [heading, "", f"{_INDENT * 2}nothing was retrieved, so there is nothing to measure"]

    lines = [
        heading,
        "",
        _line([col.header for col in _WEIGHT_COLS], _WEIGHT_COLS, ellipsis),
        _rule(table_width, glyphs["rule"]),
    ]

    for method in FUSION_METHODS:
        measurement = report.weights.get(method)
        if measurement is None:
            continue
        active = f" {glyphs['mark']}" if method == report.fusion else ""
        for index, signal in enumerate((measurement.vector, measurement.text)):
            lines.append(
                _line(
                    [
                        f"{method}{active}" if index == 0 else "",
                        signal.signal,
                        f"{signal.matched}/{measurement.candidates}",
                        _weight(signal.weight),
                        _pct(signal.nominal_share),
                        f"{_num(signal.low)} {glyphs['range']} {_num(signal.high)}",
                        _num(signal.span),
                        _pct(signal.effective_share),
                    ],
                    _WEIGHT_COLS,
                    ellipsis,
                )
            )
        lines.append(f"{_INDENT}{' ' * _VERDICT_INDENT}{_verdict(report, measurement, glyphs)}")
    return lines


def _verdict(report: ExplainReport, measurement: WeightReport, glyphs: Mapping[str, str]) -> str:
    """One line stating the gap between the configured split and the measured one."""
    missing = []
    if report.embedding_dimensions is None:
        missing.append("no embedding")
    if report.query is None:
        missing.append("no query text")
    if missing:
        # A single-signal query owns 100% of the range by definition, which says nothing
        # about the weights and should not be read as if it did.
        return f"single-signal query ({' and '.join(missing)}): the weights never competed"

    vector, text = measurement.vector, measurement.text
    nominal = f"{vector.nominal_share * 100:.0f}/{text.nominal_share * 100:.0f}"
    if vector.effective_share is None or text.effective_share is None:
        return (
            f"configured {nominal} {glyphs['arrow']} not measurable: every candidate scored alike"
        )

    effective = f"{vector.effective_share * 100:.0f}/{text.effective_share * 100:.0f}"
    line = f"configured {nominal} {glyphs['arrow']} measured {effective}"

    spans = sorted(((vector.span, "vector"), (text.span, "text")), reverse=True)
    (wide, wide_name), (narrow, narrow_name) = spans
    if narrow > 0 and wide / narrow >= 1.5:
        line += (
            f"  {glyphs['dot']}  {wide_name} moves the score {wide / narrow:.1f}x "
            f"further than {narrow_name}"
        )
    elif narrow == 0 and wide > 0:
        line += f"  {glyphs['dot']}  {narrow_name} cannot change any position"
    return line


def _render_find(
    report: ExplainReport, finding: FindReport, glyphs: Mapping[str, str], width: int
) -> list[str]:
    dot = f" {glyphs['dot']} "
    arrow = glyphs["arrow"]
    lines = [f'{_INDENT}find {glyphs["dot"]} "{finding.query}"', ""]
    body = _INDENT * 2
    hanging = body + " " * (len(arrow) + 1)

    if not finding.found:
        lines += _wrap(finding.reason, width, body, body)
        lines += _wrap(finding.remedy, width, f"{body}{arrow} ", hanging)
        return lines

    identity = [f"id {finding.id}"]
    if finding.label:
        identity.append(finding.label)
    lines += _wrap(dot.join(identity), width, body, body)
    lines += _wrap(finding.reason, width, body, body)

    signals = []
    if finding.vector_rank is not None:
        detail = f"#{finding.vector_rank} by vector"
        if finding.vector_distance is not None:
            detail += f" (distance {_num(finding.vector_distance)})"
        signals.append(detail)
    elif finding.embedding_missing:
        signals.append("no embedding")
    elif finding.vector_distance is not None:
        # Reached when the row is unranked for a reason that makes a rank meaningless,
        # such as a filter excluding it; the distance is still worth seeing.
        signals.append(f"distance {_num(finding.vector_distance)}")
    if finding.text_rank is not None:
        detail = f"#{finding.text_rank} by text"
        if finding.text_score is not None:
            detail += f" ({report.config.rank_function} {_num(finding.text_score)})"
        signals.append(detail)
    elif not finding.text_matches:
        signals.append("no text match")
    elif finding.text_score is not None:
        signals.append(f"{report.config.rank_function} {_num(finding.text_score)}")
    if signals:
        lines += _wrap(dot.join(signals), width, body, body)
    if finding.match_count > 1:
        lines += _wrap(
            f"{finding.match_count} rows contain that text; the closest is shown",
            width,
            body,
            body,
        )
    lines += _wrap(finding.remedy, width, f"{body}{arrow} ", hanging)
    return lines


__all__ = [
    "CandidateStats",
    "ExplainReport",
    "ExplainRow",
    "FindReport",
    "SignalWeights",
    "WeightReport",
    "explain",
    "explain_async",
    "measure_weights",
    "measure_weights_async",
]
