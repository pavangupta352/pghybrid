"""Configuration objects describing the table being searched.

A :class:`Config` is the only thing the SQL builder needs. It is deliberately
declarative and free of connection details so that the same object can generate a
migration, a search query and a diagnostic report.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

#: Text search configuration names are Postgres identifiers, optionally schema-qualified
#: (``pg_catalog.english``). The value is interpolated into the statement rather than
#: bound — a text search configuration is not a value, it is part of the query — so it is
#: validated to exactly that shape. Without this, a language string could close the quote
#: it sits inside and append arbitrary SQL.
_LANGUAGE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?$")

#: Function names are interpolated for the same reason, so only these are accepted.
#: Anything outside the set is rejected rather than escaped.
QUERY_PARSERS = frozenset({"websearch_to_tsquery", "plainto_tsquery", "phraseto_tsquery"})
RANK_FUNCTIONS = frozenset({"ts_rank_cd", "ts_rank"})

VectorType = Literal["vector", "halfvec"]
FusionMethod = Literal["rrf", "weighted"]
#: "numeric" produces $1, $2 (asyncpg, node-postgres, raw SQL, Supabase).
#: "pyformat" produces %s (psycopg 2 and 3).
ParamStyle = Literal["numeric", "pyformat"]
#: "any" OR-combines the query terms so the keyword signal still ranks when no
#: document contains every word; "all" keeps Postgres' native AND semantics.
TextMatch = Literal["any", "all"]

# The RRF constant from Cormack, Clarke & Buettcher (2009), "Reciprocal Rank Fusion
# outperforms Condorcet and individual rank learning methods". 60 is their reported
# value and remains the sane default: it flattens the difference between the top few
# ranks so neither signal can dominate on its first result alone.
DEFAULT_RRF_K = 60


@dataclass(frozen=True)
class Metric:
    """A pgvector distance metric and the index operator classes that match it."""

    name: str
    operator: str
    ops_vector: str
    ops_halfvec: str
    #: Whether a smaller value means a closer match. Every pgvector operator is a
    #: distance, so this is always true; it exists to keep the ranking code honest.
    ascending: bool = True

    def ops_for(self, vector_type: VectorType) -> str:
        return self.ops_vector if vector_type == "vector" else self.ops_halfvec


COSINE = Metric("cosine", "<=>", "vector_cosine_ops", "halfvec_cosine_ops")
L2 = Metric("l2", "<->", "vector_l2_ops", "halfvec_l2_ops")
INNER_PRODUCT = Metric("inner_product", "<#>", "vector_ip_ops", "halfvec_ip_ops")
L1 = Metric("l1", "<+>", "vector_l1_ops", "halfvec_l1_ops")

METRICS: dict[str, Metric] = {
    "cosine": COSINE,
    "l2": L2,
    "euclidean": L2,
    "inner_product": INNER_PRODUCT,
    "ip": INNER_PRODUCT,
    "l1": L1,
    "manhattan": L1,
}


@dataclass(frozen=True)
class Weights:
    """Relative influence of each signal.

    Under RRF these behave the way they read, because both terms are computed from
    ranks and therefore share a scale. Under ``weighted`` fusion they do not, which
    is what :meth:`pghybrid.HybridSearch.effective_weights` measures.
    """

    vector: float = 1.0
    text: float = 1.0

    def __post_init__(self) -> None:
        if self.vector < 0 or self.text < 0:
            raise ValueError("weights must be non-negative")
        if self.vector == 0 and self.text == 0:
            raise ValueError("at least one weight must be greater than zero")


@dataclass(frozen=True)
class Recency:
    """Exponential decay applied to the fused score.

    ``half_life_days`` is the age at which a row's score is halved. Rows with a NULL
    timestamp are left undecayed rather than dropped.
    """

    column: str
    half_life_days: float

    def __post_init__(self) -> None:
        if self.half_life_days <= 0:
            raise ValueError("half_life_days must be greater than zero")


@dataclass
class Config:
    """Describes one searchable table.

    Only ``table``, ``text_column`` and ``vector_column`` are required. Everything
    else has a defensible default, and every default is stated in the README so the
    behaviour is never a surprise.
    """

    table: str
    text_column: str
    vector_column: str

    id_column: str = "id"
    #: A stored tsvector column. Leave it as None to have the query compute the
    #: tsvector inline, which needs no migration but cannot use a GIN index.
    tsvector_column: str | None = None

    language: str = "english"
    vector_type: VectorType = "vector"
    metric: Metric = COSINE

    fusion: FusionMethod = "rrf"
    k: int = DEFAULT_RRF_K
    weights: Weights = field(default_factory=Weights)

    #: How many rows each signal contributes to the fusion. Larger values find more
    #: rows that one signal ranked poorly, at a proportional cost per query.
    candidate_limit: int = 50

    filter_columns: list[str] = field(default_factory=list)
    extra_columns: list[str] = field(default_factory=list)
    recency: Recency | None = None

    query_parser: Literal["websearch_to_tsquery", "plainto_tsquery", "phraseto_tsquery"] = (
        "websearch_to_tsquery"
    )
    rank_function: Literal["ts_rank_cd", "ts_rank"] = "ts_rank_cd"
    #: Placeholder style for the driver you use. Getting this wrong is the first
    #: thing that breaks for a new user, so it is explicit rather than guessed.
    paramstyle: ParamStyle = "numeric"
    #: Upper bound on the terms taken from one query under "any" matching.
    #:
    #: Each term becomes another parser call OR-ed into the statement, and past roughly
    #: 4,200 of them Postgres gives up with "stack depth limit exceeded" — an alarming
    #: message for what is really "you pasted a document into the search box". Terms
    #: beyond the limit are dropped, which costs nothing real: ts_rank_cd over hundreds
    #: of terms has stopped discriminating long before this.
    max_query_terms: int = 200

    #: See TextMatch. "any" is the default because AND semantics make the keyword
    #: half of a hybrid search return nothing for most multi-word queries, which
    #: silently degrades the whole system to vector-only search.
    text_match: TextMatch = "any"
    headline_options: str = (
        "StartSel=<mark>, StopSel=</mark>, MaxFragments=2, MinWords=8, MaxWords=30"
    )

    def __post_init__(self) -> None:
        if isinstance(self.metric, str):
            try:
                self.metric = METRICS[self.metric]
            except KeyError:
                raise ValueError(
                    f"unknown metric {self.metric!r}; expected one of {', '.join(sorted(METRICS))}"
                ) from None
        if not isinstance(self.language, str) or not _LANGUAGE_RE.match(self.language):
            raise ValueError(
                f"language must be a Postgres text search configuration name such as "
                f"'english', 'simple' or 'pg_catalog.french', got {self.language!r}. "
                "It is interpolated into the statement rather than bound, so only "
                "identifier-shaped values are accepted."
            )
        if self.query_parser not in QUERY_PARSERS:
            raise ValueError(
                f"query_parser must be one of {', '.join(sorted(QUERY_PARSERS))}, "
                f"got {self.query_parser!r}"
            )
        if self.rank_function not in RANK_FUNCTIONS:
            raise ValueError(
                f"rank_function must be one of {', '.join(sorted(RANK_FUNCTIONS))}, "
                f"got {self.rank_function!r}"
            )
        if self.text_match not in ("any", "all"):
            raise ValueError(f"text_match must be 'any' or 'all', got {self.text_match!r}")
        if self.paramstyle not in ("numeric", "pyformat"):
            raise ValueError(f"paramstyle must be 'numeric' or 'pyformat', got {self.paramstyle!r}")
        if self.vector_type not in ("vector", "halfvec"):
            raise ValueError(f"vector_type must be 'vector' or 'halfvec', got {self.vector_type!r}")
        if self.k < 0:
            raise ValueError("k must be non-negative")
        if self.candidate_limit < 1:
            raise ValueError("candidate_limit must be >= 1")
        if self.max_query_terms < 1:
            raise ValueError("max_query_terms must be >= 1")
        if self.query_parser == "phraseto_tsquery" and self.rank_function == "ts_rank":
            # Not an error, but the combination reliably surprises people.
            pass
        if isinstance(self.weights, dict):
            self.weights = Weights(**self.weights)
        if isinstance(self.recency, dict):
            self.recency = Recency(**self.recency)

    @property
    def ops_class(self) -> str:
        """The operator class an index on the vector column must use."""
        return self.metric.ops_for(self.vector_type)
