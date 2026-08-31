"""Unit tests for query parsing and the tsquery it produces.

Two layers are covered here. The first is ``parse_query``, which splits a search box
into terms. The second is what the SQL builder does with those terms, because the
tokenising only matters insofar as it produces a tsquery that ranks the way a person
expects, and because the most damaging bug in this area is invisible at the
``ParsedQuery`` level and only appears in the generated expression.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from pghybrid import Config
from pghybrid.sql import build_search_sql
from pghybrid.textquery import ParsedQuery, parse_query


def tsquery_expression(sql: str) -> str:
    """The expression the text_query CTE assigns to tsq."""
    match = re.search(r"text_query AS \(\n    SELECT (.*) AS tsq\n\)", sql)
    assert match is not None, f"no text_query CTE in:\n{sql}"
    return match.group(1)


# --------------------------------------------------------------------------------------
# parse_query
# --------------------------------------------------------------------------------------


def test_plain_terms_are_split_on_whitespace() -> None:
    assert parse_query("renewal notice period") == ParsedQuery(
        positive=["renewal", "notice", "period"], negative=[]
    )


def test_quoted_phrases_are_kept_whole() -> None:
    """A phrase is one term, so it can be handed to the parser as a phrase."""
    assert parse_query('renewal "notice period" clause') == ParsedQuery(
        positive=["renewal", "notice period", "clause"], negative=[]
    )


def test_leading_dash_excludes_a_term() -> None:
    assert parse_query("renewal -pricing") == ParsedQuery(
        positive=["renewal"], negative=["pricing"]
    )


def test_quoted_negation_excludes_a_whole_phrase() -> None:
    assert parse_query('renewal -"legacy plan"') == ParsedQuery(
        positive=["renewal"], negative=["legacy plan"]
    )


def test_the_documented_example_parses_as_documented() -> None:
    """The docstring of parse_query is the API contract most people will read."""
    assert parse_query('renewal "notice period" -pricing') == ParsedQuery(
        positive=["renewal", "notice period"], negative=["pricing"]
    )


def test_dash_inside_a_word_is_not_a_negation() -> None:
    """Hyphenated words are common; only a leading dash excludes."""
    assert parse_query("multi-tenant end-of-life") == ParsedQuery(
        positive=["multi-tenant", "end-of-life"], negative=[]
    )


@pytest.mark.parametrize("noise", ["or", "and", "OR", "And", "aNd"])
def test_bare_boolean_words_are_dropped_as_noise(noise: str) -> None:
    """Under ANY semantics the OR is already implied.

    Searching for the literal word "or" would also pollute the ranking, since it is a
    stopword in most configurations and contributes nothing but a wasted parser call.
    """
    parsed = parse_query(f"renewal {noise} termination")
    assert parsed == ParsedQuery(positive=["renewal", "termination"], negative=[])


def test_negated_boolean_word_is_still_a_negation() -> None:
    """ "-or" is an explicit instruction, not the noise word the tokeniser drops.

    The noise filter has to run only on bare terms, or a user excluding a literal word
    silently gets no exclusion at all.
    """
    assert parse_query("renewal -or") == ParsedQuery(positive=["renewal"], negative=["or"])
    assert parse_query("-and") == ParsedQuery(positive=[], negative=["and"])


def test_quoted_boolean_word_is_dropped_too() -> None:
    """Documenting current behaviour."""
    # TODO: a user who types "or" in quotes is asking for the literal word, so the noise
    # filter arguably should not apply to a quoted term. Marginal, but the two syntaxes
    # currently disagree with each other rather than with the user.
    assert parse_query('"or"') == ParsedQuery(positive=[], negative=[])
    assert parse_query('"and or"') == ParsedQuery(positive=["and or"], negative=[])


@pytest.mark.parametrize("text", ["", "   ", "\t\n ", None])
def test_empty_and_whitespace_only_input_yields_nothing(text: Any) -> None:
    """None is accepted because callers pass whatever the search box gave them."""
    parsed = parse_query(text)
    assert parsed == ParsedQuery(positive=[], negative=[])
    assert parsed.is_empty


def test_empty_quotes_contribute_no_term() -> None:
    assert parse_query('renewal ""') == ParsedQuery(positive=["renewal"], negative=[])
    assert parse_query('-"" renewal') == ParsedQuery(positive=["renewal"], negative=[])


def test_punctuation_only_input_survives_as_a_term() -> None:
    """Documenting current behaviour: the tokeniser does not judge term content.

    Postgres' parser turns punctuation into an empty tsquery, so the term matches nothing
    and costs one parser call. Filtering it here would mean deciding what counts as a
    word in every language the library supports, which is the parser's job.
    """
    assert parse_query("!!! ???") == ParsedQuery(positive=["!!!", "???"], negative=[])


def test_a_query_of_only_exclusions_has_no_positive_terms() -> None:
    parsed = parse_query("-pricing -legacy")
    assert parsed == ParsedQuery(positive=[], negative=["pricing", "legacy"])
    assert not parsed.is_empty


def test_unicode_terms_survive_intact() -> None:
    """Accents, CJK and emoji all pass through untouched.

    Any normalisation belongs to the text search configuration, not to a tokeniser that
    has no idea which language it is looking at.
    """
    parsed = parse_query('café 日本語 "kündigung frist" -naïve 🙂')
    assert parsed == ParsedQuery(
        positive=["café", "日本語", "kündigung frist", "🙂"], negative=["naïve"]
    )


def test_parsed_query_is_hashable_and_comparable() -> None:
    """A frozen value object, so it can be cached or compared without surprises."""
    assert parse_query("renewal") == parse_query("renewal")
    assert parse_query("renewal") != parse_query("termination")


# --------------------------------------------------------------------------------------
# The tsquery the builder generates
# --------------------------------------------------------------------------------------


def test_any_mode_ors_one_parser_call_per_term(make_config: Any) -> None:
    """OR semantics keep the keyword signal alive for multi-word queries.

    With Postgres' native AND, a four-word query usually matches nothing, the text
    candidate list comes back empty, and the fusion degrades to vector-only search
    without reporting that anything went wrong. Precision comes back through ranking:
    ts_rank_cd already scores a document matching three terms above one matching one.
    """
    cfg = make_config(text_match="any")
    sql, params = build_search_sql(cfg, embedding=None, text="renewal notice period", limit=5)
    tsq = tsquery_expression(sql)

    assert tsq.count("websearch_to_tsquery('english', $") == 3
    assert tsq.count(" || ") == 2
    assert " && " not in tsq
    assert params[:3] == ["renewal", "notice", "period"]


def test_any_mode_binds_each_term_separately(make_config: Any) -> None:
    """One term per placeholder, so no term can be smuggled in as query syntax."""
    cfg = make_config(text_match="any")
    sql, params = build_search_sql(cfg, embedding=None, text="renewal termination", limit=5)
    tsq = tsquery_expression(sql)
    assert re.fullmatch(
        r"\(websearch_to_tsquery\('english', \$1\) \|\| "
        r"websearch_to_tsquery\('english', \$2\)\)",
        tsq,
    )
    assert params[:2] == ["renewal", "termination"]


def test_any_mode_with_one_term_needs_no_parentheses(make_config: Any) -> None:
    cfg = make_config(text_match="any")
    sql, params = build_search_sql(cfg, embedding=None, text="renewal", limit=5)
    assert tsquery_expression(sql) == "websearch_to_tsquery('english', $1)"
    assert params[0] == "renewal"


def test_all_mode_makes_exactly_one_parser_call_with_the_whole_string(
    make_config: Any,
) -> None:
    """AND semantics are the right default for a filter, and are still one call away."""
    cfg = make_config(text_match="all")
    sql, params = build_search_sql(cfg, embedding=None, text="renewal notice -pricing", limit=5)
    tsq = tsquery_expression(sql)

    assert tsq == "websearch_to_tsquery('english', $1)"
    assert params[0] == "renewal notice -pricing"


def test_an_exclusion_constrains_both_signals_not_just_the_keyword_one(make_config: Any) -> None:
    """A leading dash is a statement about the answer, so both halves have to honour it.

    The tempting implementation puts the exclusion in the tsquery and stops, because that
    is where the parser already understands it. The vector half then never hears about it
    and happily returns the excluded rows: they drop out of the text candidates, so they
    arrive with a vector rank and no text rank, and RRF pays the top vector hit 1/(k+1),
    the largest single contribution it can award. The row someone typed "-pricing" to be
    rid of comes back near the top, ranked by half a search.

    So the predicate is rendered once and applied inside both candidate CTEs. Inside,
    because filtering the fused output would return fewer rows than the caller asked for.
    """
    cfg = make_config(text_match="any")
    sql, params = build_search_sql(
        cfg, embedding=[0.1] * 8, text="renewal notice -pricing", limit=5
    )
    vector_cte = sql[sql.index("vector_candidates AS (") : sql.index("text_query AS (")]
    text_cte = sql[sql.index("text_candidates AS (") : sql.index("scored AS (")]

    exclusion = "NOT coalesce(\"content_tsv\" @@ websearch_to_tsquery('english', $2), false)"
    assert exclusion in vector_cte, "the vector half can still return the excluded rows"
    assert exclusion in text_cte

    # Inside the candidate subquery and before its LIMIT, not wrapped around the fusion
    # afterwards: excluding rows after the cut-off returns fewer than the caller asked for.
    assert vector_cte.index(exclusion) < vector_cte.index("LIMIT")

    # The tsquery itself carries only what the user asked to find.
    assert "!!" not in tsquery_expression(sql)
    # Negation is an operator in the statement, never a character left inside a bound
    # value: the naive form hands the raw string to a single parser call.
    assert "pricing" in params
    assert "-pricing" not in params
    assert "renewal notice -pricing" not in params


def test_every_exclusion_is_applied(make_config: Any) -> None:
    cfg = make_config(text_match="any")
    sql, params = build_search_sql(
        cfg, embedding=[0.1] * 8, text="renewal -pricing -legacy", limit=5
    )
    vector_cte = sql[sql.index("vector_candidates AS (") : sql.index("text_query AS (")]
    assert "$2) || websearch_to_tsquery('english', $3)" in vector_cte
    assert params[1:4] == ["pricing", "legacy", 50]


def test_the_exclusion_is_rendered_once_and_referenced_twice(make_config: Any) -> None:
    """Numbered styles reuse the placeholder; positional styles have to repeat the value.

    The predicate appears in two CTEs, so this is the one place in the statement where a
    single logical value is genuinely referenced twice. Both paramstyles have to be right.
    """
    cfg = make_config(text_match="any")
    numeric, params = build_search_sql(cfg, embedding=[0.1] * 8, text="renewal -pricing", limit=5)
    assert numeric.count("websebogus") == 0
    assert numeric.count("$2") == 2, "the same placeholder should serve both CTEs"
    assert params.count("pricing") == 1

    pyformat, params = build_search_sql(
        make_config(text_match="any", paramstyle="pyformat"),
        embedding=[0.1] * 8,
        text="renewal -pricing",
        limit=5,
    )
    assert params.count("pricing") == 2, "pyformat cannot reuse a placeholder"


def test_a_null_tsvector_is_not_excluded_by_a_term_it_cannot_contain(make_config: Any) -> None:
    """``NULL @@ q`` is NULL and ``NOT NULL`` is NULL, which excludes the row.

    A row with no tsvector contains no words, so it contains no excluded word either. It
    has to survive the predicate, or a partially-populated column silently deletes rows
    from the vector half of every query carrying an exclusion.
    """
    cfg = make_config(text_match="any")
    sql, _ = build_search_sql(cfg, embedding=[0.1] * 8, text="renewal -pricing", limit=5)
    assert "NOT coalesce(" in sql
    assert ", false)" in sql


def test_a_query_of_only_exclusions_has_no_keyword_signal(make_config: Any) -> None:
    """There is nothing to rank by, so the text CTE is dropped rather than inverted.

    Handing "-pricing" to the parser yields ``!'pricing'``, which matches almost the whole
    table. ts_rank_cd scores a pure negation identically for every row, so the keyword
    half would contribute an arbitrary order, at full weight, that reshuffles the vector
    results for no reason. The exclusion still applies to what remains.
    """
    cfg = make_config(text_match="any")
    sql, params = build_search_sql(cfg, embedding=[0.1] * 8, text="-pricing", limit=5)
    assert "text_candidates" not in sql
    assert "NOT coalesce(" in sql
    assert params[1] == "pricing"
    assert "-pricing" not in params


def test_only_exclusions_and_no_embedding_says_what_is_missing(make_config: Any) -> None:
    cfg = make_config(text_match="any")
    with pytest.raises(ValueError, match="only excludes terms"):
        build_search_sql(cfg, embedding=None, text="-pricing", limit=5)


def test_a_noise_only_query_fails_loudly_rather_than_returning_nothing(make_config: Any) -> None:
    """ "and or" has no searchable words, and an empty result would look like relevance.

    Same choice _normalise_text makes for a blank search box: say what is wrong, because
    a caller handed [] goes looking in the ranking rather than at the query.
    """
    cfg = make_config(text_match="any")
    with pytest.raises(ValueError, match="no searchable terms"):
        build_search_sql(cfg, embedding=None, text="and or", limit=5)

    # With an embedding there is still a search to run; it is simply vector-only.
    sql, _ = build_search_sql(cfg, embedding=[0.1] * 8, text="and or", limit=5)
    assert "text_candidates" not in sql


def test_the_configured_parser_and_language_are_used(make_config: Any) -> None:
    cfg = make_config(text_match="any", query_parser="plainto_tsquery", language="german")
    sql, _ = build_search_sql(cfg, embedding=None, text="kündigung frist", limit=5)
    tsq = tsquery_expression(sql)
    assert tsq.count("plainto_tsquery('german', $") == 2
    assert "websearch_to_tsquery" not in tsq


def test_quoted_phrase_becomes_one_parser_call_but_loses_its_adjacency(
    make_config: Any,
) -> None:
    """Documenting a real gap, not endorsing it.

    The phrase survives as a single term and a single bound value, so it is never split
    across two OR-ed calls. But parse_query strips the quotes, and
    ``websearch_to_tsquery('english', 'notice period')`` is ``'notic' & 'period'``,
    an AND, not the ``'notic' <-> 'period'`` adjacency the user asked for by quoting.
    Under text_match="all" the whole raw string reaches the parser and the phrase does
    work, so the two modes disagree about what quotes mean.
    """
    # TODO: re-quote a multi-word term before binding it (or bind it to
    # phraseto_tsquery) so an ANY-mode phrase keeps its adjacency.
    cfg = make_config(text_match="any")
    sql, params = build_search_sql(cfg, embedding=None, text='renewal "notice period"', limit=5)
    assert tsquery_expression(sql).count("websearch_to_tsquery('english', $") == 2
    assert params[:2] == ["renewal", "notice period"]


class TestRepeatsAndLongQueries:
    """Two failure modes that only appear when someone pastes text into a search box."""

    def test_repeated_terms_are_collapsed(self) -> None:
        """`a | a` is `a`, so a repeat only makes the statement bigger."""
        parsed = parse_query("renewal Renewal RENEWAL notice renewal")
        assert parsed.positive == ["renewal", "notice"]

    def test_the_first_spelling_of_a_repeated_term_is_kept(self) -> None:
        """Deduplication folds case to compare but must not rewrite what it keeps."""
        assert parse_query("Renewal renewal").positive == ["Renewal"]

    def test_positive_and_negative_terms_deduplicate_separately(self) -> None:
        parsed = parse_query("renewal renewal -pricing -pricing")
        assert parsed.positive == ["renewal"]
        assert parsed.negative == ["pricing"]

    def test_a_pasted_document_does_not_blow_the_parser_stack(self, make_config: Any) -> None:
        """Past roughly 4,200 OR-ed parser calls Postgres reports a stack depth limit.

        The message reads like an internal error rather than "that query was too long",
        so the terms are capped before the statement is built. ts_rank_cd over hundreds
        of terms has long since stopped discriminating, so nothing of value is lost.
        """
        cfg = make_config(tsvector_column="fts", max_query_terms=200)
        query = " ".join(f"term{i}" for i in range(10_000))
        sql, params = build_search_sql(cfg, embedding=None, text=query, limit=5)
        assert sql.count("websearch_to_tsquery") == 200
        assert len([p for p in params if isinstance(p, str) and p.startswith("term")]) == 200

    def test_the_cap_is_configurable_and_validated(self) -> None:
        cfg = Config(table="c", text_column="content", vector_column="e", max_query_terms=3)
        sql, _ = build_search_sql(cfg, embedding=None, text="a b c d e f", limit=5)
        assert sql.count("websearch_to_tsquery") == 3
        with pytest.raises(ValueError, match="max_query_terms"):
            Config(table="c", text_column="content", vector_column="e", max_query_terms=0)
