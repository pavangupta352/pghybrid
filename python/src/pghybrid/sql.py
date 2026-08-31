"""SQL generation for hybrid search on plain Postgres.

Everything in this module is a pure function of a :class:`~pghybrid.config.Config`
and the call arguments. Nothing here touches a database, which is what makes the
generated SQL auditable, snapshot-testable, and copy-pasteable by people who never
install the package.

The generated statement is one query with two candidate CTEs — one per signal —
fused by Reciprocal Rank Fusion. Filters are applied *inside* each CTE so that both
signals search the same subset of rows; applying them after the fusion silently
destroys recall, which is the single most common way a hand-rolled implementation
goes wrong.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from .config import Config, ParamStyle, Recency
from .textquery import parse_query

# Postgres identifiers we are willing to interpolate. Anything outside this set is
# rejected rather than escaped, because a column name that needs escaping is far
# more likely to be a mistake (or an injection attempt) than a deliberate choice.
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")

# ln(2), for converting a half-life into an exponential decay rate.
_LN2 = 0.6931471805599453

FusionMethod = Literal["rrf", "weighted"]


class IdentifierError(ValueError):
    """Raised when a table or column name cannot be safely interpolated."""


def quote_ident(name: str) -> str:
    """Validate and double-quote a Postgres identifier.

    Qualified names (``schema.table``) are supported and each part is validated
    separately, so ``public.chunks`` becomes ``"public"."chunks"``.
    """
    if not isinstance(name, str) or not name:
        raise IdentifierError(f"identifier must be a non-empty string, got {name!r}")

    parts = name.split(".")
    if len(parts) > 2:
        raise IdentifierError(f"{name!r} has too many parts; expected 'name' or 'schema.name'")

    quoted = []
    for part in parts:
        if not _IDENT_RE.match(part):
            raise IdentifierError(
                f"{part!r} is not a valid Postgres identifier. Use letters, digits "
                "and underscores, starting with a letter or underscore."
            )
        quoted.append(f'"{part}"')
    return ".".join(quoted)


#: Internal sentinel wrapping a parameter slot while the statement is assembled.
#: Placeholders cannot be written directly during assembly because the right text
#: depends on the driver, and because one logical value may appear in the statement
#: more than once.
_TOKEN = "\x01p{index}\x01"
_TOKEN_RE = re.compile("\x01p(\\d+)\x01")


class Params:
    """Accumulates bind parameters and renders them in the driver's placeholder style.

    Every value that originates outside the config — the query text, the embedding,
    limits, filter values — goes through here, so the generated SQL never contains an
    interpolated literal.

    Placeholders are emitted as opaque tokens during assembly and resolved in
    :meth:`render`. That indirection exists because ``$1`` may be referenced twice in
    one statement while ``%s`` may not: numbered styles deduplicate, positional
    styles have to repeat the value.
    """

    def __init__(self) -> None:
        self._slots: list[Any] = []

    def add(self, value: Any) -> str:
        self._slots.append(value)
        return _TOKEN.format(index=len(self._slots) - 1)

    def add_cast(self, value: Any, cast: str) -> str:
        return f"{self.add(value)}::{cast}"

    def render(self, sql: str, paramstyle: ParamStyle) -> tuple[str, list[Any]]:
        """Substitute placeholders and return the statement with its final values."""
        if paramstyle == "numeric":
            numbers: dict[int, int] = {}
            values: list[Any] = []

            def replace(match: re.Match[str]) -> str:
                slot = int(match.group(1))
                if slot not in numbers:
                    values.append(self._slots[slot])
                    numbers[slot] = len(values)
                return f"${numbers[slot]}"

            return _TOKEN_RE.sub(replace, sql), values

        if paramstyle == "pyformat":
            values = []
            # A literal percent would otherwise be read as the start of a placeholder
            # by drivers that use pyformat.
            sql = sql.replace("%", "%%")

            def replace(match: re.Match[str]) -> str:
                values.append(self._slots[int(match.group(1))])
                return "%s"

            return _TOKEN_RE.sub(replace, sql), values

        raise ValueError(
            f"unknown paramstyle {paramstyle!r}; expected 'numeric' (asyncpg, "
            "node-postgres, raw SQL) or 'pyformat' (psycopg)"
        )


def _distance_expr(cfg: Config, vec_placeholder: str) -> str:
    """The distance operator for the configured metric, applied to the vector column.

    The placeholder arrives already cast to ``Config.vector_type`` — a halfvec column can
    only be compared with a halfvec — so nothing further is added here. Casting a second
    time produced ``$1::halfvec::halfvec``, which Postgres accepts and which made the
    generated SQL look like a mistake to anyone reading it.
    """
    return f"{quote_ident(cfg.vector_column)} {cfg.metric.operator} {vec_placeholder}"


def _tsvector_expr(cfg: Config) -> str:
    """The searchable tsvector: a stored column when configured, computed otherwise.

    Computing it inline means the package works against an untouched table with no
    migration at all — slower, but it lets someone try the library before changing
    their schema. The migration generator emits the stored form.
    """
    if cfg.tsvector_column:
        return quote_ident(cfg.tsvector_column)
    # NOTE: the two-argument form is required. to_tsvector(text) is STABLE, not
    # IMMUTABLE, because it reads default_text_search_config.
    return f"to_tsvector('{cfg.language}', coalesce({quote_ident(cfg.text_column)}, ''))"


def _tsquery_expr(cfg: Config, text: str, params: Params) -> str:
    """Build the tsquery expression for the configured match mode.

    ``all`` hands the whole string to one parser, giving AND semantics. ``any``
    combines one parser call per term with ``||``, so the keyword signal still
    produces a ranked candidate list when no single document contains every word.
    """
    parser = cfg.query_parser
    language = cfg.language

    def call(value: str) -> str:
        return f"{parser}('{language}', {params.add(value)})"

    if cfg.text_match == "all":
        return call(text)

    parsed = parse_query(text)
    if not parsed.positive:
        # Nothing to OR — a query of only exclusions, or only noise. Fall back to the
        # parser's own reading of the string rather than inventing a match-everything
        # query, which would flood the fusion with irrelevant candidates.
        return call(text)

    expression = " || ".join(call(term) for term in parsed.positive)
    if len(parsed.positive) > 1:
        expression = f"({expression})"
    for term in parsed.negative:
        expression = f"{expression} && !!{call(term)}"
    return expression


def _filter_sql(cfg: Config, filters: dict[str, Any] | None, params: Params) -> str:
    """Render the caller's filters as an AND-chain scoped to one candidate CTE."""
    if not filters:
        return ""
    if not cfg.filter_columns:
        raise ValueError(
            "filters were passed but Config.filter_columns is empty. List the "
            "columns you intend to filter on so they can be validated and indexed."
        )

    clauses = []
    for column, value in filters.items():
        if column not in cfg.filter_columns:
            raise ValueError(
                f"{column!r} is not in Config.filter_columns "
                f"({', '.join(cfg.filter_columns) or 'none'})."
            )
        col = quote_ident(column)
        if value is None:
            clauses.append(f"{col} IS NULL")
        elif isinstance(value, (list, tuple, set)):
            values = list(value)
            if not values:
                # An empty IN () is a syntax error in Postgres and an empty result
                # set semantically, so say so directly.
                clauses.append("FALSE")
            else:
                clauses.append(f"{col} = ANY({params.add(values)})")
        else:
            clauses.append(f"{col} = {params.add(value)}")
    return " AND " + " AND ".join(clauses)


def _recency_expr(recency: Recency | None, params: Params) -> str | None:
    """Exponential decay on the fused score, expressed as a half-life in days.

    A row with no timestamp decays by a factor of 1 (no penalty) rather than 0, so a
    partially-populated column cannot silently erase results.
    """
    if recency is None:
        return None
    col = quote_ident(recency.column)
    half_life = params.add(float(recency.half_life_days))
    return (
        f"coalesce(exp(-{_LN2} * "
        f"greatest(extract(epoch from (now() - {col})), 0) "
        f"/ ({half_life} * 86400.0)), 1.0)"
    )


def build_search_sql(
    cfg: Config,
    *,
    embedding: list[float] | None,
    text: str | None,
    limit: int,
    offset: int = 0,
    filters: dict[str, Any] | None = None,
    candidate_limit: int | None = None,
    near_miss: int = 0,
    highlight: bool = False,
    fusion: FusionMethod | None = None,
) -> tuple[str, list[Any]]:
    """Build the hybrid search statement and its bind parameters.

    Either signal may be omitted: passing only ``embedding`` produces a pure vector
    search and only ``text`` a pure full-text search, both with the same output
    columns, which is what makes the three-way comparison in ``explain`` honest.

    ``near_miss`` extends the result set past ``limit`` so callers can show the rows
    that just missed the cut — the ones that are usually the reason a search "failed".
    """
    if embedding is None and text is None:
        raise ValueError("at least one of embedding or text must be provided")
    if limit < 1:
        raise ValueError("limit must be >= 1")
    if offset < 0:
        raise ValueError("offset must be >= 0")

    fusion = fusion or cfg.fusion
    candidate_limit = candidate_limit or cfg.candidate_limit
    if candidate_limit < limit + near_miss:
        # Fusing fewer candidates than we intend to return would truncate the result
        # set before ranking ever happens.
        candidate_limit = limit + near_miss

    params = Params()
    table = quote_ident(cfg.table)
    id_col = quote_ident(cfg.id_column)

    ctes: list[str] = []
    have_vector = embedding is not None
    have_text = text is not None

    if embedding is not None:
        vec = params.add_cast(_format_vector(embedding), cfg.vector_type)
        distance = _distance_expr(cfg, vec)
        where = f"WHERE {quote_ident(cfg.vector_column)} IS NOT NULL"
        where += _filter_sql(cfg, filters, params)
        # The window sits outside the LIMIT on purpose. A rank() in the same SELECT as
        # ORDER BY ... LIMIT has to see every matching row before the limit can apply, so
        # its cost scales with the number of matches rather than with the limit. Ranking
        # the fifty rows that survive is the same answer for less work: 1.19ms against
        # 0.85ms on 100k rows, and the gap widens as the table grows.
        #
        # (EXPLAIN ANALYZE reports a far larger difference — 17.6ms against 0.8ms — but
        # that is mostly its own per-node instrumentation, which is expensive precisely
        # when a window processes every row. The figures above are wall-clock.)
        #
        # The inner ORDER BY carries a tiebreaker, without which the rows chosen at the
        # cut-off are arbitrary — and ties are not rare. ts_rank_cd gave only 3 distinct
        # values across 3,399 matching rows in the benchmark corpus, so which fifty came
        # back could change between identical runs.
        ctes.append(
            "vector_candidates AS (\n"
            "    SELECT id, distance, rank() OVER (ORDER BY distance) AS rank\n"
            "    FROM (\n"
            f"        SELECT {id_col} AS id,\n"
            f"               {distance} AS distance\n"
            f"        FROM {table}\n"
            f"        {where}\n"
            "        ORDER BY distance, id\n"
            f"        LIMIT {params.add(candidate_limit)}\n"
            "    ) candidates\n"
            ")"
        )

    if text is not None:
        tsquery = _tsquery_expr(cfg, text, params)
        tsv = _tsvector_expr(cfg)
        rank_expr = f"{cfg.rank_function}({tsv}, tsq)"
        where = "WHERE " + f"{tsv} @@ tsq"
        where += _filter_sql(cfg, filters, params)
        # Same shape as the vector side, and for the same two reasons: the window runs
        # over the fifty rows that survive rather than every match, and the cut-off is
        # deterministic. See the comment above vector_candidates.
        ctes.append(
            "text_query AS (\n"
            f"    SELECT {tsquery} AS tsq\n"
            "),\n"
            "text_candidates AS (\n"
            "    SELECT id, score, rank() OVER (ORDER BY score DESC) AS rank\n"
            "    FROM (\n"
            f"        SELECT {id_col} AS id,\n"
            f"               {rank_expr} AS score\n"
            f"        FROM {table}, text_query\n"
            f"        {where}\n"
            "        ORDER BY score DESC, id\n"
            f"        LIMIT {params.add(candidate_limit)}\n"
            "    ) candidates\n"
            ")"
        )

    scored_select, scored_from = _fusion_clause(
        cfg, params, have_vector=have_vector, have_text=have_text, fusion=fusion
    )
    ctes.append(f"scored AS (\n    SELECT {scored_select}\n    FROM {scored_from}\n)")
    # Summing in a second step keeps each contribution expression written once, which
    # makes the generated SQL readable enough that people copy it out of the README.
    ctes.append(
        "fused AS (\n"
        "    SELECT id, vector_rank, vector_distance, vector_contribution,\n"
        "           text_rank, text_score, text_contribution,\n"
        "           vector_contribution + text_contribution AS fused_score\n"
        "    FROM scored\n"
        ")"
    )

    decay = _recency_expr(cfg.recency, params)
    score_expr = "f.fused_score" if decay is None else f"(f.fused_score * {decay})"

    out_columns = [
        "f.id",
        f"{score_expr} AS score",
        "f.fused_score AS fused_score",
        "f.vector_rank",
        "f.vector_distance",
        "f.vector_contribution",
        "f.text_rank",
        "f.text_score",
        "f.text_contribution",
    ]
    if decay is not None:
        out_columns.append(f"{decay} AS recency_factor")

    for column in _output_columns(cfg):
        out_columns.append(f"t.{quote_ident(column)}")

    if highlight and have_text:
        headline_opts = params.add(cfg.headline_options)
        # ts_headline is expensive and is deliberately evaluated only for the rows
        # that survive ranking, never inside the candidate CTEs.
        out_columns.append(
            f"ts_headline('{cfg.language}', t.{quote_ident(cfg.text_column)}, "
            f"(SELECT tsq FROM text_query), {headline_opts}) AS highlight"
        )

    sql = (
        "WITH " + ",\n".join(ctes) + "\n"
        "SELECT " + ",\n       ".join(out_columns) + "\n"
        f"FROM fused f\n"
        f"JOIN {table} t ON t.{id_col} = f.id\n"
        "ORDER BY score DESC, f.id\n"
        f"LIMIT {params.add(limit + near_miss)} OFFSET {params.add(offset)}"
    )
    return params.render(sql, cfg.paramstyle)


def _fusion_clause(
    cfg: Config,
    params: Params,
    *,
    have_vector: bool,
    have_text: bool,
    fusion: FusionMethod,
) -> tuple[str, str]:
    """The SELECT list and FROM clause that combine the two candidate sets.

    A FULL OUTER JOIN is what lets a row found by only one signal still compete; an
    INNER JOIN here would quietly reduce hybrid search to the intersection of the two
    result sets, which is a different and much worse product.
    """
    # Every arithmetic parameter is cast explicitly rather than left to the server's
    # type inference. A driver that sends the weight and k as integers — which any
    # JavaScript driver must, because 1.0 and 1 are the same value there — makes
    # `1 / (60 + rank)` integer division, so every contribution truncates to zero and
    # the ranking silently collapses to whatever the tiebreaker is. The failure is
    # invisible: the query succeeds and returns rows, all scored 0.
    vector_weight = params.add(float(cfg.weights.vector)) + "::float8"
    text_weight = params.add(float(cfg.weights.text)) + "::float8"

    if fusion == "rrf":
        k = params.add(float(cfg.k)) + "::float8"
        vector_contribution = f"{vector_weight} / ({k} + v.rank)"
        text_contribution = f"{text_weight} / ({k} + t.rank)"
    elif fusion == "weighted":
        # Kept because people ask for it, and documented as the trap it is: cosine
        # distance is bounded and ts_rank is not, so the nominal weights do not
        # describe the actual influence of each signal. explain() measures the gap.
        vector_contribution = f"{vector_weight} * (1.0 - v.distance)"
        text_contribution = f"{text_weight} * t.score"
    else:
        raise ValueError(f"unknown fusion method {fusion!r}; expected 'rrf' or 'weighted'")

    if have_vector and have_text:
        select = (
            "coalesce(v.id, t.id) AS id,\n"
            "           v.rank AS vector_rank,\n"
            "           v.distance AS vector_distance,\n"
            f"           coalesce({vector_contribution}, 0) AS vector_contribution,\n"
            "           t.rank AS text_rank,\n"
            "           t.score AS text_score,\n"
            f"           coalesce({text_contribution}, 0) AS text_contribution"
        )
        from_clause = "vector_candidates v\n    FULL OUTER JOIN text_candidates t ON v.id = t.id"
    elif have_vector:
        select = (
            "v.id AS id,\n"
            "           v.rank AS vector_rank,\n"
            "           v.distance AS vector_distance,\n"
            f"           {vector_contribution} AS vector_contribution,\n"
            "           NULL::bigint AS text_rank,\n"
            "           NULL::double precision AS text_score,\n"
            "           0.0 AS text_contribution"
        )
        from_clause = "vector_candidates v"
    else:
        select = (
            "t.id AS id,\n"
            "           NULL::bigint AS vector_rank,\n"
            "           NULL::double precision AS vector_distance,\n"
            "           0.0 AS vector_contribution,\n"
            "           t.rank AS text_rank,\n"
            "           t.score AS text_score,\n"
            f"           {text_contribution} AS text_contribution"
        )
        from_clause = "text_candidates t"

    return select, from_clause


def _output_columns(cfg: Config) -> list[str]:
    """Columns copied through from the source table, de-duplicated and ordered."""
    columns: list[str] = []
    seen = set()
    for column in [cfg.text_column, *cfg.extra_columns]:
        if column and column not in seen:
            seen.add(column)
            columns.append(column)
    return columns


def _format_vector(embedding: list[float]) -> str:
    """Render a vector in pgvector's text input format.

    Passing the vector as text and casting keeps the package driver-agnostic: it
    works with psycopg, asyncpg and SQLAlchemy without any of them registering a
    pgvector type adapter.
    """
    if embedding is None:
        raise ValueError("embedding must not be None")
    try:
        return "[" + ",".join(repr(float(x)) for x in embedding) + "]"
    except (TypeError, ValueError) as exc:
        raise ValueError(f"embedding must be a sequence of numbers: {exc}") from exc
