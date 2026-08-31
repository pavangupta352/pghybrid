"""Unit tests for the configuration value objects.

Config is the whole public surface of the library: everything else is a pure function
of it. So the validation here is the only place a mistake can be caught before it turns
into a confusing runtime error from a driver, or worse, into search results that are
quietly a little bit wrong.

Each rejection test asserts that the message names the valid options. A ValueError that
only says "invalid" costs the reader a trip to the source.
"""

from __future__ import annotations

import dataclasses

import pytest

from pghybrid.config import (
    COSINE,
    DEFAULT_RRF_K,
    INNER_PRODUCT,
    L1,
    L2,
    METRICS,
    Config,
    Metric,
    Recency,
    Weights,
)

REQUIRED = dict(table="chunks", text_column="content", vector_column="embedding")


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("cosine", COSINE),
        ("l2", L2),
        ("euclidean", L2),
        ("ip", INNER_PRODUCT),
        ("inner_product", INNER_PRODUCT),
        ("l1", L1),
        ("manhattan", L1),
    ],
)
def test_metric_aliases_resolve_to_the_shared_singleton(alias: str, expected: Metric) -> None:
    """People name these distances differently depending on where they learned them."""
    assert Config(metric=alias, **REQUIRED).metric is expected


def test_every_registered_alias_resolves() -> None:
    """Guards against an alias being added to METRICS but not to the coercion path."""
    for alias in METRICS:
        assert isinstance(Config(metric=alias, **REQUIRED).metric, Metric)


def test_a_metric_object_passes_through_untouched() -> None:
    assert Config(metric=L2, **REQUIRED).metric is L2


def test_unknown_metric_names_the_valid_options() -> None:
    with pytest.raises(ValueError) as excinfo:
        Config(metric="dot", **REQUIRED)
    message = str(excinfo.value)
    assert "dot" in message
    for alias in ("cosine", "euclidean", "inner_product", "manhattan"):
        assert alias in message


def test_metrics_are_frozen_value_objects() -> None:
    """Mutating a module-level metric would silently reconfigure every other Config."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        COSINE.operator = "<->"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("metric", "vector_ops", "halfvec_ops"),
    [
        (COSINE, "vector_cosine_ops", "halfvec_cosine_ops"),
        (L2, "vector_l2_ops", "halfvec_l2_ops"),
        (INNER_PRODUCT, "vector_ip_ops", "halfvec_ip_ops"),
        (L1, "vector_l1_ops", "halfvec_l1_ops"),
    ],
)
def test_ops_class_follows_the_vector_type(
    metric: Metric, vector_ops: str, halfvec_ops: str
) -> None:
    """An index built with the wrong operator class is simply never used.

    Postgres does not complain: it plans a sequential scan and the query gets slower by
    orders of magnitude with no error to explain why.
    """
    assert Config(metric=metric, vector_type="vector", **REQUIRED).ops_class == vector_ops
    assert Config(metric=metric, vector_type="halfvec", **REQUIRED).ops_class == halfvec_ops
    assert metric.ops_for("halfvec") == halfvec_ops


def test_every_metric_is_a_distance() -> None:
    """The ranking code assumes smaller is closer; pgvector has no similarity operators."""
    assert all(metric.ascending for metric in METRICS.values())


# --------------------------------------------------------------------------------------
# Enumerated fields
# --------------------------------------------------------------------------------------


def test_invalid_vector_type_names_the_valid_options() -> None:
    with pytest.raises(ValueError) as excinfo:
        Config(vector_type="float16", **REQUIRED)
    message = str(excinfo.value)
    assert "float16" in message
    assert "vector" in message and "halfvec" in message


def test_invalid_paramstyle_names_the_valid_options() -> None:
    """Getting the placeholder style wrong is the first thing that breaks for a new user.

    asyncpg raises a syntax error on %s and psycopg raises one on $1, and neither message
    mentions this library, so the error has to be raised here instead.
    """
    with pytest.raises(ValueError) as excinfo:
        Config(paramstyle="qmark", **REQUIRED)
    message = str(excinfo.value)
    assert "qmark" in message
    assert "numeric" in message and "pyformat" in message


def test_invalid_text_match_names_the_valid_options() -> None:
    with pytest.raises(ValueError) as excinfo:
        Config(text_match="either", **REQUIRED)
    message = str(excinfo.value)
    assert "either" in message
    assert "'any'" in message and "'all'" in message


@pytest.mark.parametrize("vector_type", ["vector", "halfvec"])
def test_valid_vector_types_are_accepted(vector_type: str) -> None:
    assert Config(vector_type=vector_type, **REQUIRED).vector_type == vector_type


@pytest.mark.parametrize("paramstyle", ["numeric", "pyformat"])
def test_valid_paramstyles_are_accepted(paramstyle: str) -> None:
    assert Config(paramstyle=paramstyle, **REQUIRED).paramstyle == paramstyle


@pytest.mark.parametrize("text_match", ["any", "all"])
def test_valid_text_match_modes_are_accepted(text_match: str) -> None:
    assert Config(text_match=text_match, **REQUIRED).text_match == text_match


# --------------------------------------------------------------------------------------
# Weights
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(("vector", "text"), [(-0.1, 1.0), (1.0, -0.1), (-1.0, -1.0), (0.0, -1.0)])
def test_negative_weights_are_rejected(vector: float, text: float) -> None:
    """A negative weight would rank a signal's best results last, which is never meant."""
    with pytest.raises(ValueError) as excinfo:
        Weights(vector=vector, text=text)
    assert "non-negative" in str(excinfo.value)


def test_both_weights_zero_is_rejected() -> None:
    """Every row would score zero and the ordering would collapse to the id tiebreak."""
    with pytest.raises(ValueError) as excinfo:
        Weights(vector=0, text=0)
    assert "greater than zero" in str(excinfo.value)


@pytest.mark.parametrize(("vector", "text"), [(1.0, 0.0), (0.0, 1.0)])
def test_zeroing_one_weight_is_allowed(vector: float, text: float) -> None:
    """Turning one signal off is a legitimate way to measure what the other contributes."""
    weights = Weights(vector=vector, text=text)
    assert (weights.vector, weights.text) == (vector, text)


def test_weights_default_to_parity() -> None:
    assert Weights() == Weights(vector=1.0, text=1.0)
    assert Config(**REQUIRED).weights == Weights()


def test_each_config_gets_its_own_weights() -> None:
    """A shared mutable default would leak a tuning change into every other Config."""
    assert Config(**REQUIRED).weights is not Config(**REQUIRED).weights


def test_dict_weights_are_coerced() -> None:
    """Configs routinely arrive from JSON, YAML or a settings file."""
    cfg = Config(weights={"vector": 2.0, "text": 0.5}, **REQUIRED)
    assert cfg.weights == Weights(vector=2.0, text=0.5)
    assert isinstance(cfg.weights, Weights)


def test_partial_dict_weights_take_the_defaults() -> None:
    assert Config(weights={"vector": 3.0}, **REQUIRED).weights == Weights(vector=3.0, text=1.0)


def test_dict_weights_are_still_validated() -> None:
    """Coercion must not become a way around the constructor's checks."""
    with pytest.raises(ValueError):
        Config(weights={"vector": -1.0}, **REQUIRED)


def test_unknown_weight_key_is_rejected() -> None:
    with pytest.raises(TypeError):
        Config(weights={"vektor": 2.0}, **REQUIRED)


# --------------------------------------------------------------------------------------
# Recency
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("half_life", [0, 0.0, -1, -30.5])
def test_half_life_must_be_positive(half_life: float) -> None:
    """A zero or negative half-life makes the decay term explode or invert."""
    with pytest.raises(ValueError) as excinfo:
        Recency(column="published_at", half_life_days=half_life)
    assert "half_life_days" in str(excinfo.value)


def test_a_valid_recency_is_accepted() -> None:
    recency = Recency(column="published_at", half_life_days=30)
    assert (recency.column, recency.half_life_days) == ("published_at", 30)


def test_recency_defaults_to_off() -> None:
    """Decay changes ranking, so it is never applied unless it was asked for."""
    assert Config(**REQUIRED).recency is None


def test_dict_recency_is_coerced() -> None:
    cfg = Config(recency={"column": "published_at", "half_life_days": 14}, **REQUIRED)
    assert cfg.recency == Recency(column="published_at", half_life_days=14)
    assert isinstance(cfg.recency, Recency)


def test_dict_recency_is_still_validated() -> None:
    with pytest.raises(ValueError):
        Config(recency={"column": "published_at", "half_life_days": 0}, **REQUIRED)


def test_recency_is_a_frozen_value_object() -> None:
    recency = Recency(column="published_at", half_life_days=30)
    with pytest.raises(dataclasses.FrozenInstanceError):
        recency.half_life_days = 60  # type: ignore[misc]


# --------------------------------------------------------------------------------------
# Numeric bounds and defaults
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("k", [-1, -60])
def test_rrf_k_must_be_non_negative(k: int) -> None:
    with pytest.raises(ValueError) as excinfo:
        Config(k=k, **REQUIRED)
    assert "k must be non-negative" in str(excinfo.value)


def test_rrf_k_of_zero_is_allowed() -> None:
    """k=0 makes RRF the plain reciprocal of the rank, which is a defensible choice."""
    assert Config(k=0, **REQUIRED).k == 0


@pytest.mark.parametrize("candidate_limit", [0, -1])
def test_candidate_limit_must_be_positive(candidate_limit: int) -> None:
    """A zero candidate budget would fuse two empty lists and return nothing."""
    with pytest.raises(ValueError) as excinfo:
        Config(candidate_limit=candidate_limit, **REQUIRED)
    assert "candidate_limit" in str(excinfo.value)


def test_the_defaults_are_the_ones_the_readme_documents() -> None:
    """These defaults are the library's argument, so pin them.

    text_match="any" in particular is the whole thesis: Postgres' native AND semantics
    make the keyword half of a hybrid search return nothing for most multi-word queries,
    which silently degrades the system to vector-only search.
    """
    cfg = Config(**REQUIRED)
    assert cfg.id_column == "id"
    assert cfg.tsvector_column is None
    assert cfg.language == "english"
    assert cfg.vector_type == "vector"
    assert cfg.metric is COSINE
    assert cfg.fusion == "rrf"
    assert cfg.k == DEFAULT_RRF_K == 60
    assert cfg.candidate_limit == 50
    assert cfg.text_match == "any"
    assert cfg.paramstyle == "numeric"
    assert cfg.query_parser == "websearch_to_tsquery"
    assert cfg.rank_function == "ts_rank_cd"
    assert cfg.filter_columns == []
    assert cfg.extra_columns == []


def test_each_config_gets_its_own_column_lists() -> None:
    first = Config(**REQUIRED)
    first.filter_columns.append("tenant_id")
    assert Config(**REQUIRED).filter_columns == []


# ----------------------------------------------------------- interpolated fields


class TestFieldsThatAreInterpolatedIntoTheStatement:
    """Three config fields cannot be bound, so they are validated instead.

    Everything a caller supplies is either a bind parameter or an identifier passed
    through quote_ident — except the text search configuration and the two function
    names, which are parts of the query rather than values. Those were interpolated
    unchecked, so a language string could close the quote it sits inside and append
    whatever it liked. An application that lets a user pick a search language is not a
    strange thing to build.
    """

    INJECTIONS = [
        "english'), (SELECT 1)) AS x FROM chunks; DROP TABLE users; --",
        "english' || (SELECT current_setting('is_superuser')) || '",
        "english'; --",
        "english\\\\",
        "",
        "pg catalog",
        "english; DROP TABLE t",
    ]

    @pytest.mark.parametrize("value", INJECTIONS)
    def test_language_rejects_anything_not_identifier_shaped(self, value: str) -> None:
        with pytest.raises(ValueError, match="text search configuration"):
            Config(table="c", text_column="content", vector_column="e", language=value)

    @pytest.mark.parametrize(
        "language", ["english", "simple", "french", "german", "pg_catalog.english", "_custom1"]
    )
    def test_real_configuration_names_are_accepted(self, language: str) -> None:
        assert (
            Config(table="c", text_column="content", vector_column="e", language=language).language
            == language
        )

    @pytest.mark.parametrize(
        "value", ["websearch_to_tsquery'); DROP TABLE t; --", "to_tsquery", "", "eval"]
    )
    def test_query_parser_is_a_closed_set(self, value: str) -> None:
        with pytest.raises(ValueError, match="query_parser"):
            Config(table="c", text_column="content", vector_column="e", query_parser=value)

    @pytest.mark.parametrize("value", ["ts_rank_cd'); DROP TABLE t; --", "ts_rank_bad", ""])
    def test_rank_function_is_a_closed_set(self, value: str) -> None:
        with pytest.raises(ValueError, match="rank_function"):
            Config(table="c", text_column="content", vector_column="e", rank_function=value)

    def test_headline_options_may_be_anything_because_it_is_bound(self) -> None:
        """The contrast that makes the rule clear: a value gets bound, not validated."""
        from pghybrid.sql import build_search_sql

        hostile = "x'); DROP TABLE t; --"
        cfg = Config(
            table="c",
            text_column="content",
            vector_column="e",
            tsvector_column="fts",
            headline_options=hostile,
        )
        sql, params = build_search_sql(cfg, embedding=None, text="hi", limit=5, highlight=True)
        assert "DROP TABLE" not in sql
        assert hostile in params
