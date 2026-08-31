"""Unit tests for query parsing and the tsquery it produces.

Two layers are covered here. The first is ``parse_query``, which splits a search box
into terms. The second is what the SQL builder does with those terms, because the
tokenising only matters insofar as it produces a tsquery that ranks the way a person
expects — and because the most damaging bug in this area is invisible at the
``ParsedQuery`` level and only appears in the generated expression.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

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
    """"-or" is an explicit instruction, not the noise word the tokeniser drops.

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


def test_any_mode_does_not_rewrite_and_not_into_or_not(make_config: Any) -> None:
    """The exclusion has to stay conjunctive, or it stops excluding anything.

    The tempting implementation of ANY is to parse the whole string once and swap the
    operators inside the resulting tsquery. That is wrong in a way that is easy to miss:
    ``'a' & !'b'`` becomes ``'a' | !'b'``, which is true for every document that merely
    lacks b — so a query of "renewal -pricing" matches the entire corpus except the
    pricing pages, ranked by nothing.

    So the OR is built from separate parser calls, the positives are parenthesised as a
    group, and each exclusion is attached with && rather than ||.
    """
    cfg = make_config(text_match="any")
    sql, params = build_search_sql(
        cfg, embedding=None, text="renewal notice -pricing", limit=5
    )
    tsq = tsquery_expression(sql)

    assert " && !!" in tsq
    assert " || !!" not in tsq
    # The exclusion applies to the whole disjunction, not just the last positive term.
    assert tsq.startswith("(")
    assert tsq.index(")") < tsq.index("&&")
    # Negation is an operator in the statement, never a character left inside a bound
    # value: the naive form would hand the raw string to a single parser call.
    assert params[:3] == ["renewal", "notice", "pricing"]
    assert "renewal notice -pricing" not in params
    assert "-pricing" not in params


def test_any_mode_attaches_every_exclusion(make_config: Any) -> None:
    cfg = make_config(text_match="any")
    sql, params = build_search_sql(
        cfg, embedding=None, text="renewal -pricing -legacy", limit=5
    )
    tsq = tsquery_expression(sql)
    assert tsq.count("&& !!") == 2
    assert params[:3] == ["renewal", "pricing", "legacy"]


def test_a_single_positive_with_an_exclusion_needs_no_group(make_config: Any) -> None:
    cfg = make_config(text_match="any")
    sql, _ = build_search_sql(cfg, embedding=None, text="renewal -pricing", limit=5)
    tsq = tsquery_expression(sql)
    assert tsq == (
        "websearch_to_tsquery('english', $1) && !!websearch_to_tsquery('english', $2)"
    )


def test_a_query_of_only_exclusions_falls_back_to_the_parser(make_config: Any) -> None:
    """There is nothing to OR, and inventing a match-everything query would be worse.

    A tsquery of pure negation matches almost the whole table, which would flood the
    fusion with candidates that no signal actually ranked. Handing the raw string to
    websearch_to_tsquery keeps the behaviour the user's syntax already implies.
    """
    cfg = make_config(text_match="any")
    sql, params = build_search_sql(cfg, embedding=None, text="-pricing", limit=5)
    assert tsquery_expression(sql) == "websearch_to_tsquery('english', $1)"
    assert params[0] == "-pricing"


def test_a_noise_only_query_falls_back_to_the_parser(make_config: Any) -> None:
    """websearch_to_tsquery reads "and or" as an empty tsquery, which matches nothing.

    So the text signal contributes no candidates and the search is vector-only, which is
    the honest answer for a query with no searchable words in it.
    """
    cfg = make_config(text_match="any")
    sql, params = build_search_sql(cfg, embedding=None, text="and or", limit=5)
    assert tsquery_expression(sql) == "websearch_to_tsquery('english', $1)"
    assert params[0] == "and or"


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
    ``websearch_to_tsquery('english', 'notice period')`` is ``'notic' & 'period'`` —
    an AND, not the ``'notic' <-> 'period'`` adjacency the user asked for by quoting.
    Under text_match="all" the whole raw string reaches the parser and the phrase does
    work, so the two modes disagree about what quotes mean.
    """
    # TODO: re-quote a multi-word term before binding it (or bind it to
    # phraseto_tsquery) so an ANY-mode phrase keeps its adjacency.
    cfg = make_config(text_match="any")
    sql, params = build_search_sql(
        cfg, embedding=None, text='renewal "notice period"', limit=5
    )
    assert tsquery_expression(sql).count("websearch_to_tsquery('english', $") == 2
    assert params[:2] == ["renewal", "notice period"]
