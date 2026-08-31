"""Unit tests for the SQL builder.

Every test here is a pure function of a Config and some arguments: no connection, no
fixtures with side effects, no network. That is the whole point of keeping SQL
generation separate from execution, the statement can be asserted on directly, and a
regression in the query shape is caught before anyone has to notice bad search results.

The assertions favour naming the failure mode over matching the text. A test called
``test_hybrid_fuses_with_a_full_outer_join`` that fails tells you the library just
started returning the intersection of the two signals; a test called ``test_sql_shape``
tells you nothing.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import pytest

from pghybrid.config import RESERVED_OUTPUT_NAMES, Config, Recency, Weights
from pghybrid.sql import (
    IdentifierError,
    Params,
    _output_columns,
    build_search_sql,
    quote_ident,
)

GOLDEN = Path(__file__).parent / "golden" / "canonical_search.sql"


# --------------------------------------------------------------------------------------
# Helpers for reading the generated statement.
#
# The builder emits every CTE-internal SELECT indented and the final SELECT at column
# zero, which is what lets these split the statement without a SQL parser.
# --------------------------------------------------------------------------------------


def cte(sql: str, name: str) -> str:
    """The body of one named CTE."""
    match = re.search(rf"{re.escape(name)} AS \((.*?)\n\)", sql, re.DOTALL)
    assert match is not None, f"no {name!r} CTE in:\n{sql}"
    return match.group(1)


def cte_block(sql: str) -> str:
    """Everything before the final SELECT: the WITH clause and nothing else."""
    head, separator, _ = sql.partition("\nSELECT ")
    assert separator, f"no top-level SELECT in:\n{sql}"
    return head


def final_select(sql: str) -> str:
    """The final SELECT onwards, with the CTEs stripped off."""
    _, separator, tail = sql.partition("\nSELECT ")
    assert separator, f"no top-level SELECT in:\n{sql}"
    return "SELECT " + tail


def output_aliases(sql: str) -> list[str]:
    """The column names a caller actually receives, in order."""
    select_list = final_select(sql).split("\nFROM fused f")[0][len("SELECT ") :]
    aliases = []
    for column in select_list.split(",\n       "):
        column = column.strip()
        alias = column.rsplit(" AS ", 1)[1] if " AS " in column else column.rsplit(".", 1)[-1]
        aliases.append(alias.strip('"'))
    return aliases


def bound_limit(fragment: str, params: list[Any]) -> Any:
    """The value bound to the LIMIT in a fragment. Numeric paramstyle only."""
    match = re.search(r"LIMIT \$(\d+)", fragment)
    assert match is not None, f"no LIMIT in:\n{fragment}"
    return params[int(match.group(1)) - 1]


# --------------------------------------------------------------------------------------
# quote_ident
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("chunks", '"chunks"'),
        ("public.chunks", '"public"."chunks"'),
        ("_internal", '"_internal"'),
        ("col$1", '"col$1"'),
        ("MixedCase", '"MixedCase"'),
        ("analytics.Events2024", '"analytics"."Events2024"'),
    ],
)
def test_quote_ident_accepts_plain_and_qualified_names(name: str, expected: str) -> None:
    """Quoting is unconditional so a name that collides with a keyword still works."""
    assert quote_ident(name) == expected


@pytest.mark.parametrize(
    "name",
    [
        'a"; DROP TABLE users; --',
        "chunks; DELETE FROM chunks",
        "chunks'",
        "content, (SELECT secret FROM keys)",
        "tab-le",
        "1abc",
        "two words",
        "public.chunks; --",
        "  ",
    ],
)
def test_quote_ident_rejects_injection_attempts(name: str) -> None:
    """Names outside the identifier alphabet are refused, not escaped.

    Escaping would work, but a column name that needs escaping is far more likely to be
    an injected string than a deliberate choice, so the failure is loud.
    """
    with pytest.raises(IdentifierError):
        quote_ident(name)


@pytest.mark.parametrize("name", ["", None, 0, [], "a."])
def test_quote_ident_rejects_empty_and_missing_names(name: Any) -> None:
    with pytest.raises(IdentifierError):
        quote_ident(name)


@pytest.mark.parametrize("name", ["a.b.c", "db.public.chunks", "w.x.y.z"])
def test_quote_ident_rejects_three_part_names(name: str) -> None:
    """Postgres has no database-qualified references in a query; catch the confusion."""
    with pytest.raises(IdentifierError) as excinfo:
        quote_ident(name)
    assert "schema.name" in str(excinfo.value)


def test_identifier_error_is_a_value_error() -> None:
    """Callers that only catch ValueError must not leak an IdentifierError."""
    assert issubclass(IdentifierError, ValueError)
    with pytest.raises(ValueError):
        quote_ident("no spaces allowed")


def test_config_identifiers_are_validated_at_build_time(make_config: Any) -> None:
    """A bad name in the Config fails when the SQL is built, not when it is executed."""
    cfg = make_config(table="chunks; DROP TABLE users")
    with pytest.raises(IdentifierError):
        build_search_sql(cfg, embedding=[0.1], text=None, limit=5)


# --------------------------------------------------------------------------------------
# Placeholder rendering
# --------------------------------------------------------------------------------------


def test_numeric_style_reuses_one_placeholder_for_a_repeated_value(config: Config) -> None:
    """$1 can be referenced as many times as the statement needs it.

    The RRF constant k appears in both contribution expressions, and numbered
    placeholders let both point at one copy. The embedding is now mentioned only once,
    since ranking happens outside the LIMIT, so k is what exercises the deduplication.
    """
    sql, params = build_search_sql(config, embedding=[0.1, 0.2, 0.3], text="renewal", limit=5)
    scored = cte(sql, "scored")
    k_placeholder = re.search(r"\((\$\d+)::float8 \+ v\.rank\)", scored).group(1)
    assert scored.count(f"{k_placeholder}::float8") == 2
    assert params.count(60.0) == 1
    assert params.count("[0.1,0.2,0.3]") == 1


def test_pyformat_style_repeats_the_value_for_every_mention(make_config: Any) -> None:
    """%s is positional, so a value used twice has to be sent twice."""
    cfg = make_config(paramstyle="pyformat")
    sql, params = build_search_sql(cfg, embedding=[0.1, 0.2, 0.3], text="renewal", limit=5)
    scored = cte(sql, "scored")
    # k is referenced by both contributions and so must be sent twice.
    assert scored.count("%s::float8") == 4
    assert params.count(60.0) == 2
    assert "$1" not in sql


def test_parameter_count_differs_between_styles_for_the_same_query(make_config: Any) -> None:
    """The two styles must disagree on parameter count, and that is correct.

    This is the property the whole Params indirection exists for. If both styles ever
    produce the same count, one of them is wrong: either numeric stopped deduplicating
    repeated references, or pyformat stopped repeating them and the driver is about to
    receive fewer values than the statement has placeholders.
    """
    kwargs = dict(
        embedding=[0.1, 0.2, 0.3],
        text="renewal notice",
        limit=5,
        filters={"tenant_id": 7},
        highlight=True,
    )
    numeric_sql, numeric_params = build_search_sql(make_config(), **kwargs)
    pyformat_sql, pyformat_params = build_search_sql(make_config(paramstyle="pyformat"), **kwargs)

    assert len(pyformat_params) > len(numeric_params)
    assert len(numeric_params) == len(set(re.findall(r"\$\d+", numeric_sql)))
    assert len(pyformat_params) == pyformat_sql.count("%s")
    # Same statement, different placeholder syntax: normalising one to the other has to
    # produce the same skeleton.
    assert re.sub(r"\$\d+", "?", numeric_sql) == pyformat_sql.replace("%s", "?")


def test_pyformat_escapes_a_literal_percent() -> None:
    """A bare % is read as the start of a placeholder by psycopg, so it is doubled.

    No current code path emits a literal percent, but the renderer is the only place
    that could, and the escaping has to survive the next feature that needs a LIKE
    pattern or a modulo.
    """
    params = Params()
    slot = params.add("acme")
    sql, values = params.render(f"SELECT {slot} WHERE tenant LIKE 'a%b'", "pyformat")
    assert sql == "SELECT %s WHERE tenant LIKE 'a%%b'"
    assert values == ["acme"]


def test_numeric_style_leaves_a_literal_percent_alone() -> None:
    """Doubling the percent for a driver that does not use pyformat would corrupt it."""
    params = Params()
    slot = params.add("acme")
    sql, values = params.render(f"SELECT {slot} WHERE tenant LIKE 'a%b'", "numeric")
    assert sql == "SELECT $1 WHERE tenant LIKE 'a%b'"
    assert values == ["acme"]


def test_unknown_paramstyle_is_rejected_by_the_renderer() -> None:
    """Config catches this earlier; the renderer is the backstop for direct callers."""
    params = Params()
    with pytest.raises(ValueError) as excinfo:
        params.render("SELECT 1", "qmark")
    message = str(excinfo.value)
    assert "numeric" in message and "pyformat" in message


def test_no_caller_value_is_ever_interpolated_into_the_statement(make_config: Any) -> None:
    """Every value that came from outside the Config must arrive as a bind parameter."""
    cfg = make_config()
    sql, params = build_search_sql(
        cfg,
        embedding=[0.123456789],
        text="quarterly renewal",
        limit=5,
        filters={"tenant_id": "acme-tenant-7f3"},
        highlight=True,
    )
    for secret in ("0.123456789", "quarterly", "renewal", "acme-tenant-7f3"):
        assert secret not in sql
    assert "acme-tenant-7f3" in params


# --------------------------------------------------------------------------------------
# The three query shapes
# --------------------------------------------------------------------------------------


def test_each_signal_combination_builds_a_distinct_statement(config: Config) -> None:
    vector_only, _ = build_search_sql(config, embedding=[0.1], text=None, limit=5)
    text_only, _ = build_search_sql(config, embedding=None, text="renewal", limit=5)
    hybrid, _ = build_search_sql(config, embedding=[0.1], text="renewal", limit=5)

    assert len({vector_only, text_only, hybrid}) == 3

    assert "vector_candidates AS (" in vector_only
    assert "text_candidates AS (" not in vector_only

    assert "text_candidates AS (" in text_only
    assert "vector_candidates AS (" not in text_only

    assert "vector_candidates AS (" in hybrid
    assert "text_candidates AS (" in hybrid


def test_all_three_query_shapes_expose_the_same_output_columns(config: Config) -> None:
    """A single-signal search returns the same row shape as a hybrid one.

    ``explain`` compares vector-only, text-only and hybrid results side by side. That
    comparison is only honest if the three statements are interchangeable to the caller,
    so the columns for the missing signal are selected as typed NULLs rather than
    dropped.
    """
    expected = [
        "id",
        "score",
        "fused_score",
        "vector_rank",
        "vector_distance",
        "vector_contribution",
        "text_rank",
        "text_score",
        "text_contribution",
        "content",
        "title",
        "url",
    ]
    for embedding, text in ([0.1], None), (None, "renewal"), ([0.1], "renewal"):
        sql, _ = build_search_sql(config, embedding=embedding, text=text, limit=5)
        assert output_aliases(sql) == expected


def test_missing_signal_columns_are_typed_nulls(config: Config) -> None:
    """An untyped NULL makes Postgres guess the column type, which breaks drivers."""
    vector_only, _ = build_search_sql(config, embedding=[0.1], text=None, limit=5)
    assert "NULL::bigint AS text_rank" in vector_only
    assert "NULL::double precision AS text_score" in vector_only

    text_only, _ = build_search_sql(config, embedding=None, text="renewal", limit=5)
    assert "NULL::bigint AS vector_rank" in text_only
    assert "NULL::double precision AS vector_distance" in text_only


def test_passthrough_columns_are_deduplicated(make_config: Any) -> None:
    """Naming the text column again in extra_columns must not select it twice."""
    cfg = make_config(extra_columns=["content", "title", "title"])
    sql, _ = build_search_sql(cfg, embedding=[0.1], text=None, limit=5)
    aliases = output_aliases(sql)
    assert aliases.count("content") == 1
    assert aliases.count("title") == 1


# --------------------------------------------------------------------------------------
# Fusion
# --------------------------------------------------------------------------------------


def test_hybrid_fuses_with_a_full_outer_join_never_an_inner_join(config: Config) -> None:
    """An INNER JOIN here would silently reduce hybrid search to the intersection.

    That is the single most damaging regression this file can catch: the query still
    runs, still returns rows, and still looks plausible, it just quietly drops every
    document that only one of the two signals found, which is most of the ones hybrid
    search exists to surface.
    """
    sql, _ = build_search_sql(config, embedding=[0.1], text="renewal", limit=5)
    scored = cte(sql, "scored")
    assert "FULL OUTER JOIN text_candidates t ON v.id = t.id" in scored
    joins = re.findall(r"(?:\w+ )*JOIN", scored)
    assert joins == ["FULL OUTER JOIN"], f"unexpected join in the fusion: {joins}"
    # A row found by one signal has no rank in the other, so its missing contribution
    # has to fall back to zero rather than NULL, which would null out the whole sum.
    assert scored.count("coalesce(") == 3


def test_rrf_contribution_is_weight_over_k_plus_rank(config: Config) -> None:
    sql, params = build_search_sql(config, embedding=[0.1], text="renewal", limit=5)
    scored = cte(sql, "scored")
    # The ::float8 casts are load-bearing, not decoration. Without them a driver
    # that sends the weight and k as integers turns this into integer division and
    # every contribution truncates to zero, silently.
    assert re.search(
        r"coalesce\(\$\d+::float8 / \(\$\d+::float8 \+ v\.rank\), 0\) AS vector_contribution",
        scored,
    )
    assert re.search(
        r"coalesce\(\$\d+::float8 / \(\$\d+::float8 \+ t\.rank\), 0\) AS text_contribution",
        scored,
    )
    # k is bound once and referenced by both contributions.
    assert params.count(60.0) == 1


def test_weighted_fusion_scores_on_the_raw_signals(config: Config) -> None:
    """Kept because people ask for it, and asserted so its trap stays visible."""
    sql, _ = build_search_sql(config, embedding=[0.1], text="renewal", limit=5, fusion="weighted")
    scored = cte(sql, "scored")
    assert re.search(
        r"coalesce\(\$\d+::float8 \* \(1\.0 - v\.distance\), 0\) AS vector_contribution",
        scored,
    )
    assert re.search(r"coalesce\(\$\d+::float8 \* t\.score, 0\) AS text_contribution", scored)
    # Cosine distance is bounded and ts_rank is not, so the nominal weights do not
    # describe the actual influence of each signal. That is the trap; explain() measures
    # it. Nothing here should quietly start normalising the two scales.
    assert "rank)" not in scored


def test_fusion_argument_overrides_the_config(make_config: Any) -> None:
    cfg = make_config(fusion="weighted")
    default_sql, _ = build_search_sql(cfg, embedding=[0.1], text="renewal", limit=5)
    override_sql, _ = build_search_sql(cfg, embedding=[0.1], text="renewal", limit=5, fusion="rrf")
    assert "1.0 - v.distance" in default_sql
    assert "1.0 - v.distance" not in override_sql


def test_weights_reach_the_statement_as_bind_parameters(make_config: Any) -> None:
    """Tuning weights must not require regenerating the SQL, or the plan cache is lost."""
    cfg = make_config(weights=Weights(vector=2.5, text=0.25))
    _, params = build_search_sql(cfg, embedding=[0.1], text="renewal", limit=5)
    assert 2.5 in params
    assert 0.25 in params


# --------------------------------------------------------------------------------------
# Filters
# --------------------------------------------------------------------------------------


def test_filters_are_applied_inside_both_candidate_ctes(config: Config) -> None:
    """Filtering after the fusion is the classic way to destroy recall.

    If the filter runs after fusion, each signal spends its candidate budget on rows the
    caller has already excluded, and a tenant with few documents gets an empty result set
    from a query that matches plenty of its rows.
    """
    sql, _ = build_search_sql(
        config, embedding=[0.1], text="renewal", limit=5, filters={"tenant_id": 7}
    )
    assert sql.count('"tenant_id" = ') == 2
    assert '"tenant_id" = ' in cte(sql, "vector_candidates")
    assert '"tenant_id" = ' in cte(sql, "text_candidates")


def test_filters_do_not_appear_after_the_fusion(config: Config) -> None:
    sql, _ = build_search_sql(
        config, embedding=[0.1], text="renewal", limit=5, filters={"tenant_id": 7}
    )
    assert "tenant_id" not in cte(sql, "scored")
    assert "tenant_id" not in cte(sql, "fused")
    assert "tenant_id" not in final_select(sql)


def test_a_single_signal_query_applies_the_filter_once(config: Config) -> None:
    sql, _ = build_search_sql(config, embedding=[0.1], text=None, limit=5, filters={"tenant_id": 7})
    assert sql.count('"tenant_id" = ') == 1


def test_filters_are_anded_onto_the_existing_where_clause(config: Config) -> None:
    """The vector CTE already excludes NULL embeddings; a filter extends that clause."""
    sql, _ = build_search_sql(
        config, embedding=[0.1], text=None, limit=5, filters={"tenant_id": 7, "lang": "en"}
    )
    candidates = cte(sql, "vector_candidates")
    assert re.search(
        r'WHERE "embedding" IS NOT NULL AND "tenant_id" = \$\d+ AND "lang" = \$\d+',
        candidates,
    )


def test_unknown_filter_column_is_rejected(config: Config) -> None:
    """Filter columns are declared up front so they can be validated and indexed."""
    with pytest.raises(ValueError) as excinfo:
        build_search_sql(config, embedding=[0.1], text=None, limit=5, filters={"deleted_at": None})
    message = str(excinfo.value)
    assert "deleted_at" in message
    assert "tenant_id" in message and "lang" in message


def test_filters_without_declared_filter_columns_are_rejected(make_config: Any) -> None:
    cfg = make_config(filter_columns=[])
    with pytest.raises(ValueError) as excinfo:
        build_search_sql(cfg, embedding=[0.1], text=None, limit=5, filters={"tenant_id": 7})
    assert "filter_columns" in str(excinfo.value)


def test_empty_list_filter_becomes_false_rather_than_invalid_sql(config: Config) -> None:
    """``IN ()`` is a syntax error in Postgres; an empty set means no rows, so say so."""
    sql, params = build_search_sql(
        config, embedding=[0.1], text=None, limit=5, filters={"lang": []}
    )
    assert '"embedding" IS NOT NULL AND FALSE' in sql
    assert "IN ()" not in sql
    assert "ANY()" not in sql
    assert [] not in params


def test_none_filter_becomes_is_null(config: Config) -> None:
    """``= NULL`` is never true; the caller meant IS NULL."""
    sql, params = build_search_sql(
        config, embedding=[0.1], text=None, limit=5, filters={"tenant_id": None}
    )
    assert '"tenant_id" IS NULL' in sql
    assert None not in params


@pytest.mark.parametrize("value", [["en", "de"], ("en", "de"), {"en"}])
def test_collection_filters_bind_one_array_parameter(config: Config, value: Any) -> None:
    """= ANY($n) keeps the placeholder count independent of the number of values.

    Expanding to IN ($1, $2, $3) would generate a different statement for every list
    length and defeat the server's prepared-statement cache.
    """
    sql, params = build_search_sql(
        config, embedding=[0.1], text=None, limit=5, filters={"lang": value}
    )
    assert re.search(r'"lang" = ANY\(\$\d+\)', sql)
    assert any(isinstance(param, list) for param in params)


# --------------------------------------------------------------------------------------
# Text side
# --------------------------------------------------------------------------------------


def test_stored_tsvector_column_is_used_when_configured(config: Config) -> None:
    sql, _ = build_search_sql(config, embedding=None, text="renewal", limit=5)
    assert '"content_tsv" @@ tsq' in sql
    assert "to_tsvector(" not in sql


def test_inline_tsvector_uses_the_two_argument_form(make_config: Any) -> None:
    """``to_tsvector(text)`` is STABLE, not IMMUTABLE: it reads default_text_search_config.

    The one-argument form cannot be indexed, and a session that sets a different search
    configuration silently changes what the query means. Naming the language keeps both
    problems away.
    """
    cfg = make_config(tsvector_column=None)
    sql, _ = build_search_sql(cfg, embedding=None, text="renewal", limit=5)
    assert "to_tsvector('english', coalesce(\"content\", ''))" in sql


def test_ts_headline_is_evaluated_only_after_ranking(config: Config) -> None:
    """ts_headline re-parses the document text; running it per candidate is ruinous.

    Inside a candidate CTE it would run for every row the signal considered. In the final
    SELECT it runs only for the page being returned.
    """
    sql, _ = build_search_sql(config, embedding=[0.1], text="renewal", limit=5, highlight=True)
    assert sql.count("ts_headline(") == 1
    assert "ts_headline(" in final_select(sql)
    assert "ts_headline(" not in cte_block(sql)
    assert "highlight" in output_aliases(sql)


def test_headline_reuses_the_parsed_tsquery(config: Config) -> None:
    """Re-parsing the query string for the headline could highlight different terms."""
    sql, _ = build_search_sql(config, embedding=[0.1], text="renewal", limit=5, highlight=True)
    assert "(SELECT tsq FROM text_query)" in final_select(sql)


def test_highlight_is_ignored_without_a_text_signal(config: Config) -> None:
    """There is no tsquery to highlight against, so asking for one is a no-op."""
    sql, _ = build_search_sql(config, embedding=[0.1], text=None, limit=5, highlight=True)
    assert "ts_headline" not in sql
    assert "highlight" not in output_aliases(sql)


def test_headline_options_are_bound_not_interpolated(config: Config) -> None:
    sql, params = build_search_sql(config, embedding=[0.1], text="renewal", limit=5, highlight=True)
    assert "StartSel" not in sql
    assert any(isinstance(param, str) and "StartSel" in param for param in params)


# --------------------------------------------------------------------------------------
# Vector type and metric
# --------------------------------------------------------------------------------------


def test_halfvec_config_casts_the_query_vector_and_picks_the_halfvec_opclass(
    make_config: Any,
) -> None:
    """A halfvec column can only be compared with a halfvec, and indexed by halfvec ops.

    Getting either half wrong gives a query that still returns rows: the cast makes
    Postgres fall back to a sequential scan, and the wrong operator class makes the index
    unusable without saying so.
    """
    cfg = make_config(vector_type="halfvec", metric="l2")
    sql, _ = build_search_sql(cfg, embedding=[0.5, 0.25], text=None, limit=5)
    candidates = cte(sql, "vector_candidates")

    assert "::halfvec" in candidates
    assert "::vector" not in candidates
    # Exactly one cast. The placeholder arrives already cast to the configured vector
    # type, and adding another produced $1::halfvec::halfvec.
    assert candidates.count("::halfvec") == 1
    assert cfg.ops_class == "halfvec_l2_ops"

    # Was "$1::halfvec::halfvec": Params.add_cast already casts to Config.vector_type
    # and _distance_expr cast again. Postgres accepted it and the plan was identical,
    # but it read as a mistake in SQL people copy out of the README.
    assert "::halfvec::halfvec" not in candidates


def test_default_vector_type_casts_to_vector(config: Config) -> None:
    sql, _ = build_search_sql(config, embedding=[0.5], text=None, limit=5)
    assert "$1::vector" in sql
    assert "halfvec" not in sql


@pytest.mark.parametrize(
    ("metric", "operator", "ops_class"),
    [
        ("cosine", "<=>", "vector_cosine_ops"),
        ("l2", "<->", "vector_l2_ops"),
        ("euclidean", "<->", "vector_l2_ops"),
        ("inner_product", "<#>", "vector_ip_ops"),
        ("ip", "<#>", "vector_ip_ops"),
        ("l1", "<+>", "vector_l1_ops"),
        ("manhattan", "<+>", "vector_l1_ops"),
    ],
)
def test_each_metric_maps_to_its_operator(
    make_config: Any, metric: str, operator: str, ops_class: str
) -> None:
    """Using the wrong operator ranks by the wrong distance and skips the index.

    The failure is silent, results come back, they are just subtly worse, so the
    mapping is pinned here rather than trusted.
    """
    cfg = make_config(metric=metric)
    sql, _ = build_search_sql(cfg, embedding=[0.5], text=None, limit=5)
    assert f'"embedding" {operator} $1::vector' in sql
    assert cfg.ops_class == ops_class


def test_vector_candidates_exclude_null_embeddings(config: Config) -> None:
    """A NULL embedding sorts as an unknown distance and would pollute the candidates."""
    sql, _ = build_search_sql(config, embedding=[0.1], text=None, limit=5)
    assert '"embedding" IS NOT NULL' in cte(sql, "vector_candidates")


def test_vector_is_sent_as_text_for_the_server_to_cast(config: Config) -> None:
    """Passing pgvector's text format keeps the package driver-agnostic.

    No psycopg or asyncpg type adapter has to be registered, which is what lets the
    library have zero runtime dependencies.
    """
    _, params = build_search_sql(config, embedding=[0.1, -0.2], text=None, limit=5)
    assert params[0] == "[0.1,-0.2]"


# --------------------------------------------------------------------------------------
# Recency
# --------------------------------------------------------------------------------------


def test_recency_expression_is_absent_when_not_configured(config: Config) -> None:
    sql, _ = build_search_sql(config, embedding=[0.1], text="renewal", limit=5)
    assert "exp(" not in sql
    assert "recency_factor" not in output_aliases(sql)
    assert "f.fused_score AS score" in sql


def test_recency_decays_the_score_and_reports_the_factor(make_config: Any) -> None:
    """The factor is returned as its own column so a surprising ranking is explainable."""
    cfg = make_config(recency=Recency(column="published_at", half_life_days=30))
    sql, params = build_search_sql(cfg, embedding=[0.1], text="renewal", limit=5)

    assert "(f.fused_score * coalesce(exp(" in sql
    assert "recency_factor" in output_aliases(sql)
    assert 30.0 in params
    # One bind slot, two references: the score and the reported factor must not be able
    # to disagree about the half-life.
    assert sql.count("AS recency_factor") == 1
    assert params.count(30.0) == 1


def test_recency_leaves_rows_with_no_timestamp_undecayed(make_config: Any) -> None:
    """A half-populated timestamp column must not silently erase results.

    Decaying a NULL to zero would drop every backfilled row out of the ranking, which
    looks exactly like the search being broken.
    """
    cfg = make_config(recency=Recency(column="published_at", half_life_days=30))
    sql, _ = build_search_sql(cfg, embedding=[0.1], text=None, limit=5)
    match = re.search(r"(coalesce\(exp\(.+?\), 1\.0\)) AS recency_factor", sql)
    assert match is not None, sql
    decay = match.group(1)

    assert decay.startswith("coalesce(")
    assert decay.endswith(", 1.0)")
    # A future timestamp must not amplify the score past 1.0 either.
    assert 'greatest(extract(epoch from (now() - "published_at")), 0)' in decay
    # Half-life is expressed in days, so the epoch seconds are scaled by a day.
    assert "* 86400.0" in decay


def test_recency_column_is_quoted(make_config: Any) -> None:
    cfg = make_config(recency=Recency(column="published_at", half_life_days=7))
    sql, _ = build_search_sql(cfg, embedding=[0.1], text=None, limit=5)
    assert 'now() - "published_at"' in sql


# --------------------------------------------------------------------------------------
# Limits, offsets and the candidate budget
# --------------------------------------------------------------------------------------


def test_near_miss_extends_the_final_limit(config: Config) -> None:
    """The rows that just missed the cut are usually why a search "failed"."""
    sql, params = build_search_sql(
        config, embedding=[0.1], text=None, limit=10, near_miss=5, offset=20
    )
    match = re.search(r"LIMIT \$(\d+) OFFSET \$(\d+)", sql)
    assert match is not None, sql
    assert params[int(match.group(1)) - 1] == 15
    assert params[int(match.group(2)) - 1] == 20


def test_candidate_limit_is_raised_to_cover_limit_plus_near_miss(make_config: Any) -> None:
    """Fusing fewer candidates than we return truncates the result before ranking."""
    cfg = make_config(candidate_limit=5)
    sql, params = build_search_sql(cfg, embedding=[0.1], text="renewal", limit=10, near_miss=3)
    assert bound_limit(cte(sql, "vector_candidates"), params) == 13
    assert bound_limit(cte(sql, "text_candidates"), params) == 13


def test_reserved_names_are_exactly_what_the_statement_returns(make_config: Any) -> None:
    """The guard that keeps the reserved list honest as the query grows.

    RESERVED_OUTPUT_NAMES exists to stop a table column shadowing a computed one. It is a
    hand-written list, so it rots the moment someone adds an output column and does not
    think of it, and the failure it guards against is silent, which is exactly the kind
    nobody notices. This reads the aliases back out of a statement with every optional
    output turned on and insists the two agree.
    """
    cfg = make_config(recency=Recency(column="created_at", half_life_days=30))
    sql, _ = build_search_sql(
        cfg, embedding=[0.1], text="renewal", limit=5, highlight=True, near_miss=2
    )
    # The final SELECT: everything between the last "SELECT" at column 0 and its FROM.
    final = sql[sql.rindex("\nSELECT ") :]
    projection = final[: final.index("\nFROM ")]

    emitted = []
    for item in projection.replace("\nSELECT ", "").split(",\n"):
        item = item.strip().rstrip(",")
        if not item:
            continue
        # "expr AS name", or "f.name" / 't."name"' when nothing renames it.
        name = item.rsplit(" AS ", 1)[-1] if " AS " in item else item.rsplit(".", 1)[-1]
        emitted.append(name.strip().strip('"'))

    computed = [name for name in emitted if name not in _output_columns(cfg)]
    assert set(computed) == set(RESERVED_OUTPUT_NAMES), (
        "the statement's own output columns and RESERVED_OUTPUT_NAMES have diverged.\n"
        f"  emitted but not reserved: {sorted(set(computed) - set(RESERVED_OUTPUT_NAMES))}\n"
        f"  reserved but not emitted: {sorted(set(RESERVED_OUTPUT_NAMES) - set(computed))}"
    )


def test_the_candidate_pool_does_not_depend_on_the_offset(make_config: Any) -> None:
    """Ranks are assigned inside the pool, so a pool that varies per page reorders pages.

    This is the property that makes pagination usable at all. Widening the pool to cover
    the offset looks like the obvious fix for an empty page and is worse than the empty
    page: rows enter the text candidates as the pool grows, gain a text contribution and
    jump the fused ordering, so paging 8x10 returned 71 distinct rows instead of 80 with
    9 that a single limit=80 query returns never shown at all.
    """
    cfg = make_config(candidate_limit=100)
    pools = set()
    # Every legal offset, including the last one that fits.
    for offset in (0, 10, 50, 89, 90):
        sql, params = build_search_sql(
            cfg, embedding=[0.1], text="renewal", limit=10, offset=offset
        )
        pools.add(bound_limit(cte(sql, "vector_candidates"), params))
        pools.add(bound_limit(cte(sql, "text_candidates"), params))
    assert pools == {100}, f"the pool changed with the offset: {sorted(pools)}"


def test_a_page_outside_the_candidate_pool_is_an_error(make_config: Any) -> None:
    """An empty page is indistinguishable from having reached the end of the results."""
    cfg = make_config(candidate_limit=50)
    with pytest.raises(ValueError) as excinfo:
        build_search_sql(cfg, embedding=[0.1], text="renewal", limit=10, offset=50)
    message = str(excinfo.value)
    # The numbers a caller needs to act, not just a complaint.
    assert "at least 60" in message and "it is 50" in message
    assert "candidate_limit" in message

    # The last page that fits is still fine.
    assert build_search_sql(cfg, embedding=[0.1], text="renewal", limit=10, offset=40)


def test_near_miss_counts_against_the_pool_too(make_config: Any) -> None:
    """The near-miss band is rows we return, so it occupies the pool like any other."""
    cfg = make_config(candidate_limit=50)
    assert build_search_sql(cfg, embedding=[0.1], text="renewal", limit=10, offset=30, near_miss=10)
    with pytest.raises(ValueError, match="at least 61"):
        build_search_sql(cfg, embedding=[0.1], text="renewal", limit=10, offset=40, near_miss=11)


def test_a_large_limit_at_the_first_page_still_widens_the_pool(make_config: Any) -> None:
    """There is only one page, so there is no ordering to keep stable between pages."""
    cfg = make_config(candidate_limit=50)
    sql, params = build_search_sql(cfg, embedding=[0.1], text="renewal", limit=500)
    assert bound_limit(cte(sql, "vector_candidates"), params) == 500


def test_candidate_limit_is_left_alone_when_already_large_enough(make_config: Any) -> None:
    cfg = make_config(candidate_limit=200)
    sql, params = build_search_sql(cfg, embedding=[0.1], text=None, limit=10, near_miss=3)
    assert bound_limit(cte(sql, "vector_candidates"), params) == 200


def test_candidate_limit_argument_overrides_the_config(make_config: Any) -> None:
    cfg = make_config(candidate_limit=50)
    sql, params = build_search_sql(cfg, embedding=[0.1], text=None, limit=10, candidate_limit=120)
    assert bound_limit(cte(sql, "vector_candidates"), params) == 120


def test_candidate_limit_of_zero_falls_back_to_the_config(make_config: Any) -> None:
    """Documenting current behaviour, not endorsing it.

    ``candidate_limit or cfg.candidate_limit`` treats 0 as "not supplied" rather than as
    an out-of-range value.
    """
    # TODO: candidate_limit=0 is silently replaced by the config value instead of raising
    # the way limit=0 does. Harmless today (a zero budget is meaningless) but the two
    # arguments validate inconsistently.
    cfg = make_config(candidate_limit=50)
    sql, params = build_search_sql(cfg, embedding=[0.1], text=None, limit=10, candidate_limit=0)
    assert bound_limit(cte(sql, "vector_candidates"), params) == 50


def test_each_candidate_cte_orders_and_limits_independently(config: Config) -> None:
    """Both signals must contribute a full candidate list, ranked on their own terms.

    Each CTE limits first and ranks the survivors. Ranking inside the limited SELECT
    would force the window over every matching row before the limit could apply, which
    stops an index scan from finishing early.
    """
    sql, _ = build_search_sql(config, embedding=[0.1], text="renewal", limit=5)
    vector = cte(sql, "vector_candidates")
    text = cte(sql, "text_candidates")

    assert "rank() OVER (ORDER BY distance) AS rank" in vector
    assert "rank() OVER (ORDER BY score DESC) AS rank" in text

    # The tiebreaker is what makes the cut-off reproducible; ts_rank_cd ties heavily.
    assert "ORDER BY distance, id" in vector
    assert "ORDER BY score DESC, id" in text

    # The window must sit outside the subquery that carries the LIMIT.
    for candidates in (vector, text):
        assert candidates.index("rank() OVER") < candidates.index("LIMIT")


def test_results_are_ordered_by_the_final_score_with_a_stable_tiebreak(config: Config) -> None:
    """Without the id tiebreak, paging through equal scores can repeat or skip rows."""
    sql, _ = build_search_sql(config, embedding=[0.1], text="renewal", limit=5)
    assert "ORDER BY score DESC, f.id" in sql


# --------------------------------------------------------------------------------------
# Argument validation
# --------------------------------------------------------------------------------------


def test_at_least_one_signal_is_required(config: Config) -> None:
    with pytest.raises(ValueError) as excinfo:
        build_search_sql(config, embedding=None, text=None, limit=5)
    assert "embedding or text" in str(excinfo.value)


@pytest.mark.parametrize("limit", [0, -1, -100])
def test_limit_must_be_positive(config: Config, limit: int) -> None:
    with pytest.raises(ValueError) as excinfo:
        build_search_sql(config, embedding=[0.1], text=None, limit=limit)
    assert "limit must be >= 1" in str(excinfo.value)


@pytest.mark.parametrize("offset", [-1, -50])
def test_offset_must_be_non_negative(config: Config, offset: int) -> None:
    with pytest.raises(ValueError) as excinfo:
        build_search_sql(config, embedding=[0.1], text=None, limit=5, offset=offset)
    assert "offset must be >= 0" in str(excinfo.value)


def test_unknown_fusion_method_is_rejected(config: Config) -> None:
    with pytest.raises(ValueError) as excinfo:
        build_search_sql(config, embedding=[0.1], text=None, limit=5, fusion="borda")
    message = str(excinfo.value)
    assert "borda" in message
    assert "rrf" in message and "weighted" in message


@pytest.mark.parametrize("embedding", [["a"], [None], [{}]])
def test_non_numeric_embedding_is_rejected(config: Config, embedding: Any) -> None:
    """Catching this here beats a driver-specific cast error from the server."""
    with pytest.raises(ValueError) as excinfo:
        build_search_sql(config, embedding=embedding, text=None, limit=5)
    assert "sequence of numbers" in str(excinfo.value)


def test_empty_embedding_builds_but_pgvector_will_reject_it(config: Config) -> None:
    """Documenting current behaviour: dimensionality is the server's business.

    The builder has no idea how many dimensions the column has, so it does not pretend
    to validate the length.
    """
    _, params = build_search_sql(config, embedding=[], text=None, limit=5)
    assert params[0] == "[]"


def test_the_interpolated_names_are_validated_before_they_reach_the_statement(
    make_config: Any,
) -> None:
    """language, query_parser and rank_function are parts of the query, not values.

    They cannot be bound, so they are validated in Config instead, and they used not to
    be, which made a Config built from user input an injection surface unlike every other
    field. The check lives in Config, so the builder can interpolate them without
    thinking about it; this asserts the two halves stay connected.
    """
    with pytest.raises(ValueError, match="text search configuration"):
        make_config(language="english', 'injected")

    cfg = make_config(tsvector_column=None, language="french")
    sql, _ = build_search_sql(cfg, embedding=None, text="renewal", limit=5)
    assert "to_tsvector('french'" in sql
    assert sql.count("'") % 2 == 0, "an odd number of quotes means one of them escaped"


# --------------------------------------------------------------------------------------
# Golden snapshot
# --------------------------------------------------------------------------------------


def _canonical_query() -> tuple[str, list[Any]]:
    """One fully-featured query, spelled out rather than built from a fixture.

    A snapshot that depends on a shared fixture changes meaning whenever somebody tunes
    the fixture for an unrelated test, so this one owns its inputs.
    """
    cfg = Config(
        table="public.chunks",
        text_column="content",
        vector_column="embedding",
        id_column="chunk_id",
        tsvector_column="content_tsv",
        language="english",
        vector_type="vector",
        metric="cosine",
        fusion="rrf",
        k=60,
        weights=Weights(vector=1.5, text=1.0),
        candidate_limit=80,
        filter_columns=["tenant_id", "lang"],
        extra_columns=["title", "url"],
        recency=Recency(column="published_at", half_life_days=30.0),
        paramstyle="numeric",
        text_match="any",
    )
    return build_search_sql(
        cfg,
        embedding=[0.25, -0.5, 0.75],
        text='renewal "notice period" -pricing',
        limit=10,
        offset=20,
        filters={"tenant_id": 42, "lang": ["en", "de"]},
        near_miss=3,
        highlight=True,
    )


def test_canonical_sql_matches_the_golden_snapshot() -> None:
    """Pin the exact statement for one config that exercises every branch.

    This snapshot is the contract the TypeScript port has to reproduce byte for byte.
    Two ports that agree on the public API but disagree on the SQL they emit are two
    different libraries with one name, and the difference would only ever surface as
    "the JS results are a bit worse", which nobody can debug.

    A diff here is either a deliberate change to the query, regenerate with
    PGHYBRID_UPDATE_GOLDEN=1 and read the diff line by line before committing it, or a
    bug that just escaped every other test in this file.
    """
    sql, _ = _canonical_query()
    expected = sql + "\n"

    updating = bool(os.environ.get("PGHYBRID_UPDATE_GOLDEN"))
    if updating or not GOLDEN.exists():
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(expected, encoding="utf-8")
        if not updating:
            pytest.fail(f"golden snapshot was missing. Wrote {GOLDEN}; review it and re-run.")

    assert GOLDEN.read_text(encoding="utf-8") == expected


def test_canonical_parameters_match_the_snapshot() -> None:
    """The bind values are half the contract; SQL alone would not catch a reordering."""
    _, params = _canonical_query()
    assert params == [
        "[0.25,-0.5,0.75]",
        42,
        ["en", "de"],
        # The exclusion is bound in the vector CTE, which comes first, and the text CTE
        # references the same placeholder rather than binding a second copy.
        "pricing",
        80,
        "renewal",
        "notice period",
        42,
        ["en", "de"],
        80,
        1.5,
        60.0,
        1.0,
        30.0,
        "StartSel=<mark>, StopSel=</mark>, MaxFragments=2, MinWords=8, MaxWords=30",
        13,
        20,
    ]


def test_canonical_query_is_deterministic() -> None:
    """A snapshot test is worthless if the builder emits set-ordered output."""
    first_sql, first_params = _canonical_query()
    second_sql, second_params = _canonical_query()
    assert first_sql == second_sql
    assert first_params == second_params
