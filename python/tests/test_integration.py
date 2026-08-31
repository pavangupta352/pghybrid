"""Tests that run against a real Postgres.

The unit suite proves the generated SQL says what it should. These prove the server
agrees, which is a different question: a statement can be perfectly formed and still
be rejected, use the wrong index, or rank rows in an order nobody intended.

Skipped automatically when no database is reachable, so ``pytest`` stays useful on a
laptop with nothing running. CI always has one.

    docker compose up -d
    PGHYBRID_TEST_DSN=postgresql://postgres:pghybrid@localhost:55432/pghybrid pytest
"""

from __future__ import annotations

import os
import pathlib
import sys

import pytest

from pghybrid import Config, Recency, Weights
from pghybrid.doctor import doctor
from pghybrid.explain import explain
from pghybrid.schema import build_migration, dbapi_executor, introspect, suggest_config
from pghybrid.search import HybridSearch

psycopg = pytest.importorskip("psycopg")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))
from seed_demo import (  # noqa: E402
    DEMO_QUERY,
    DOCUMENTS,
    PLANTED_TITLE,
    SCHEMA,
    query_vector,
    to_pgvector,
    unit_vector,
)

DSN = os.environ.get("PGHYBRID_TEST_DSN", "postgresql://postgres:pghybrid@localhost:55432/pghybrid")

pytestmark = pytest.mark.integration


def _reachable() -> bool:
    try:
        with psycopg.connect(DSN, connect_timeout=3):
            return True
    except Exception:
        return False


if not _reachable():  # pragma: no cover - depends on the environment
    pytest.skip(f"no Postgres at {DSN}", allow_module_level=True)


@pytest.fixture(scope="module")
def connection():
    conn = psycopg.connect(DSN, autocommit=True, row_factory=psycopg.rows.dict_row)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.execute(SCHEMA)
    for angle, title, content in DOCUMENTS:
        conn.execute(
            "INSERT INTO chunks (title, content, embedding) VALUES (%s, %s, %s)",
            (title, content, to_pgvector(unit_vector(angle))),
        )
    yield conn
    conn.close()


@pytest.fixture
def config() -> Config:
    return Config(
        table="chunks",
        text_column="content",
        vector_column="embedding",
        tsvector_column="fts",
        extra_columns=["title"],
        filter_columns=["tenant_id"],
        paramstyle="pyformat",
    )


@pytest.fixture
def search(connection, config) -> HybridSearch:
    return HybridSearch(
        config, execute=lambda sql, params: connection.execute(sql, params).fetchall()
    )


def titles(results) -> list[str]:
    return [str(r.get("title")) for r in results]


# --------------------------------------------------------------- the central claim


def test_only_pgvector_is_installed(connection):
    """The project's whole pitch is that no other extension is needed.

    Asserted rather than stated, so a dependency on pg_search, pg_trgm or anything
    else would fail the build here instead of failing for someone on RDS.
    """
    installed = {row["extname"] for row in connection.execute("SELECT extname FROM pg_extension")}
    assert installed <= {"plpgsql", "vector"}, (
        f"the test database has extensions beyond pgvector: {installed}. "
        "Every claim this project makes depends on not needing them."
    )


# ------------------------------------------------------------------ the golden property


def test_fusion_beats_either_signal_alone(search):
    """Neither signal ranks the answer first. Fusion does. That is the product."""
    embedding = query_vector()

    vector_only = titles(search.search(None, embedding=embedding, limit=5))
    text_only = titles(search.search(DEMO_QUERY, limit=5))
    hybrid = titles(search.search(DEMO_QUERY, embedding=embedding, limit=5))

    assert vector_only[0] != PLANTED_TITLE, "vector-only was not supposed to find it"
    assert text_only[0] != PLANTED_TITLE, "keyword-only was not supposed to find it"
    assert hybrid[0] == PLANTED_TITLE, (
        f"fusion should surface {PLANTED_TITLE!r} first, got {hybrid[0]!r}"
    )
    # It has to be findable by both, or the win would be luck rather than fusion.
    assert PLANTED_TITLE in vector_only and PLANTED_TITLE in text_only


def test_or_semantics_keep_the_keyword_signal_alive(connection, config):
    """AND semantics make multi-word queries match almost nothing.

    This is the failure that silently reduces hybrid search to vector-only: the
    keyword side returns too few rows to rank anything.
    """
    run = lambda cfg: HybridSearch(  # noqa: E731
        cfg, execute=lambda sql, params: connection.execute(sql, params).fetchall()
    ).search(DEMO_QUERY, limit=10)

    config.text_match = "any"
    any_rows = run(config)
    config.text_match = "all"
    all_rows = run(config)

    assert len(all_rows) == 1, (
        f"the demo corpus is built so AND semantics collapse to a single row; got {len(all_rows)}"
    )
    assert len(any_rows) > len(all_rows)


def test_negation_is_not_a_naive_operator_rewrite(search):
    """Rewriting & into | would turn 'a -b' into 'match anything lacking b'."""
    excluded = titles(search.search("renewal -pricing", limit=10))
    assert "Renewal pricing" not in excluded
    assert excluded, "excluding one term should not empty the result set"


# ---------------------------------------------------------------------- correctness


def test_filters_apply_to_both_signals(connection, config, search):
    connection.execute("UPDATE chunks SET tenant_id = 2 WHERE title = %s", (PLANTED_TITLE,))
    try:
        rows = search.search(
            DEMO_QUERY, embedding=query_vector(), limit=10, filters={"tenant_id": 1}
        )
        assert PLANTED_TITLE not in titles(rows), (
            "a row excluded by the filter came back, so the filter is not being "
            "applied inside both candidate CTEs"
        )
        rows = search.search(
            DEMO_QUERY, embedding=query_vector(), limit=10, filters={"tenant_id": 2}
        )
        assert titles(rows) == [PLANTED_TITLE]
    finally:
        connection.execute("UPDATE chunks SET tenant_id = 1")


def test_both_paramstyles_return_identical_rows(connection, config):
    """$1 and %s are different wire formats for the same query, not different queries."""
    config.paramstyle = "pyformat"
    pyformat_rows = HybridSearch(
        config, execute=lambda sql, p: connection.execute(sql, p).fetchall()
    ).search(DEMO_QUERY, embedding=query_vector(), limit=5)

    # psycopg speaks %s, so numeric placeholders are checked by rendering them and
    # confirming the statement is one Postgres accepts via PREPARE.
    config.paramstyle = "numeric"
    from pghybrid.sql import build_search_sql

    sql, params = build_search_sql(config, embedding=query_vector(), text=DEMO_QUERY, limit=5)
    assert "$1" in sql and "%s" not in sql
    connection.execute("DEALLOCATE ALL")
    connection.execute(f"PREPARE parity AS {sql}")
    connection.execute("DEALLOCATE parity")
    assert len(pyformat_rows) == 5


def test_recency_decay_leaves_null_and_future_rows_alone(connection, config):
    config.recency = Recency(column="created_at", half_life_days=180)
    connection.execute("UPDATE chunks SET created_at = now() - interval '180 days'")
    try:
        rows = HybridSearch(
            config, execute=lambda sql, p: connection.execute(sql, p).fetchall()
        ).search(DEMO_QUERY, embedding=query_vector(), limit=3)
        for row in rows:
            assert row.recency_factor == pytest.approx(0.5, abs=0.01)

        connection.execute("UPDATE chunks SET created_at = now() + interval '30 days'")
        rows = HybridSearch(
            config, execute=lambda sql, p: connection.execute(sql, p).fetchall()
        ).search(DEMO_QUERY, embedding=query_vector(), limit=3)
        for row in rows:
            assert row.recency_factor == pytest.approx(1.0), (
                "a future timestamp must not inflate a score above its fused value"
            )
    finally:
        connection.execute("UPDATE chunks SET created_at = now()")


def test_highlight_marks_the_matched_terms(search):
    rows = search.search(DEMO_QUERY, embedding=query_vector(), limit=3, highlight=True)
    assert any("<mark>" in (r.highlight or "") for r in rows)


def test_a_stopword_only_query_leaves_the_vector_signal_working(search):
    rows = search.search("the and of", embedding=query_vector(), limit=3)
    assert len(rows) == 3
    assert all(r.text_rank is None for r in rows)


# --------------------------------------------------------------------------- explain


def test_explain_reports_the_near_miss_band(search):
    report = explain(search, DEMO_QUERY, query_vector(), limit=2, near_miss=3, label_column="title")
    assert len(report.rows) == 5
    assert sum(1 for row in report.rows if row.near_miss) == 3


def test_find_separates_never_indexed_from_outranked(search):
    present = explain(
        search,
        DEMO_QUERY,
        query_vector(),
        limit=2,
        near_miss=0,
        label_column="title",
        find="sixty days written notice",
    )
    assert present.find is not None and present.find.found

    absent = explain(
        search,
        DEMO_QUERY,
        query_vector(),
        limit=2,
        near_miss=0,
        label_column="title",
        find="force majeure pandemic clause",
    )
    assert absent.find is not None and not absent.find.found, (
        "text that no row contains is an ingestion bug, and saying so is the whole point of find"
    )


def test_explain_renders_without_crashing_on_a_single_signal(search):
    for text, embedding in ((DEMO_QUERY, None), (None, query_vector())):
        report = explain(search, text, embedding, limit=3, near_miss=2, label_column="title")
        assert report.to_text()


# ---------------------------------------------------------------- schema and doctor


def test_introspect_finds_what_is_actually_there(connection):
    info = introspect(dbapi_executor(connection), "chunks")
    assert info.row_count == len(DOCUMENTS)
    assert [c.name for c in info.columns if c.is_vector] == ["embedding"]
    assert any(c.name == "fts" for c in info.columns)
    assert {i.method for i in info.indexes} >= {"hnsw", "gin"}
    assert info.pgvector_version


def test_introspect_works_with_a_mapping_row_factory(connection):
    """A dict_row connection unpacks into column names unless the executor prevents it.

    Two catalog columns legitimately share a name, so a mapping row also arrives one
    column short. Both failures were real.
    """
    with psycopg.connect(DSN, row_factory=psycopg.rows.dict_row) as mapping_conn:
        info = introspect(dbapi_executor(mapping_conn), "chunks")
    assert info.kind in ("r", "p", "m", "table", "partitioned table", "materialized view")
    assert info.row_count == len(DOCUMENTS)


def test_migration_is_idempotent(connection):
    executor = dbapi_executor(connection)
    connection.execute("DROP TABLE IF EXISTS migration_target CASCADE")
    connection.execute(
        """CREATE TABLE migration_target (
               id bigserial PRIMARY KEY, body text NOT NULL, embedding vector(8))"""
    )
    try:
        info = introspect(executor, "migration_target")
        config = suggest_config(info)
        assert config.text_column == "body"

        required = [s for s in build_migration(config, info) if not s.optional]
        assert required, "a table with no tsvector column and no indexes needs work"
        for statement in required:
            connection.execute(statement.sql)

        again = introspect(executor, "migration_target")
        left = [
            s
            for s in build_migration(suggest_config(again), again)
            if not s.optional and s.kind != "maintenance"
        ]
        assert left == [], f"re-running the migration still wants to do {left}"
    finally:
        connection.execute("DROP TABLE IF EXISTS migration_target CASCADE")


def test_generated_tsvector_column_uses_the_immutable_form(connection):
    """to_tsvector(text) is STABLE and Postgres rejects it in a generated column.

    The error message does not name the cause, so this is worth pinning: if the
    migration ever emits the one-argument form, the failure should surface here.
    """
    connection.execute("DROP TABLE IF EXISTS immutability_check CASCADE")
    connection.execute("CREATE TABLE immutability_check (id bigserial PRIMARY KEY, body text)")
    try:
        with pytest.raises(psycopg.errors.InvalidObjectDefinition):
            connection.execute(
                "ALTER TABLE immutability_check ADD COLUMN bad tsvector "
                "GENERATED ALWAYS AS (to_tsvector(body)) STORED"
            )
        connection.execute(
            "ALTER TABLE immutability_check ADD COLUMN good tsvector "
            "GENERATED ALWAYS AS (to_tsvector('english', coalesce(body, ''))) STORED"
        )
    finally:
        connection.execute("DROP TABLE IF EXISTS immutability_check CASCADE")


def test_doctor_is_read_only_and_reports_recall(connection, config):
    before = connection.execute(
        "SELECT count(*) AS n FROM pg_indexes WHERE tablename = 'chunks'"
    ).fetchone()["n"]
    report = doctor(dbapi_executor(connection), config, sample=5, k=3)
    after = connection.execute(
        "SELECT count(*) AS n FROM pg_indexes WHERE tablename = 'chunks'"
    ).fetchone()["n"]

    assert before == after, "doctor created or dropped an index; it must be read-only"
    assert report.to_text()
    if report.recall is not None:
        assert 0.0 <= report.recall.recall <= 1.0
        assert report.recall.sample > 0
        assert report.recall.exact_ground_truth, (
            "recall is only meaningful when the comparison really was exact search"
        )


def test_weights_change_the_ordering(connection, config):
    def top(vector_weight: float, text_weight: float) -> str:
        config.weights = Weights(vector=vector_weight, text=text_weight)
        rows = HybridSearch(
            config, execute=lambda sql, p: connection.execute(sql, p).fetchall()
        ).search(DEMO_QUERY, embedding=query_vector(), limit=1)
        return str(rows[0].get("title"))

    assert top(1.0, 0.0) != top(0.0, 1.0), (
        "leaning entirely on one signal should reproduce that signal's own winner"
    )


def test_scores_survive_integer_typed_parameters(connection, config):
    """The RRF arithmetic must not depend on how a driver types its parameters.

    JavaScript has one number type, so a JS driver sends 1 where Python sends 1.0.
    Postgres then infers integer for the weight and for k, `1 / (60 + rank)` becomes
    integer division, and every contribution truncates to zero. Nothing errors: the
    query succeeds, returns the right rows, and scores them all 0, so the ranking
    silently degrades to whatever the tiebreaker happens to be.

    This reproduces that by coercing every numeric parameter to an int before binding,
    which is exactly what the TypeScript package does on the wire.
    """
    from pghybrid.sql import build_search_sql

    sql, params = build_search_sql(config, embedding=query_vector(), text=DEMO_QUERY, limit=5)
    as_integers = [int(p) if isinstance(p, float) else p for p in params]
    assert any(isinstance(p, float) for p in params), "the fixture should exercise floats"

    rows = connection.execute(sql, as_integers).fetchall()
    assert rows, "the query should still return rows"
    assert all(row["score"] > 0 for row in rows), (
        "every score came back as zero, so the ::float8 casts in the fusion "
        "expression have been dropped and the arithmetic is truncating"
    )
    assert rows[0]["title"] == PLANTED_TITLE


# --------------------------------------------------------------------------- adapters


def _adapter_titles(search) -> list[str]:
    return titles(search.search(DEMO_QUERY, embedding=query_vector(), limit=3))


def test_every_adapter_returns_the_same_results(connection):
    """A driver is a transport, not a dialect.

    Each adapter also pins the placeholder style its driver needs, which is the single
    thing users get wrong: $1 and %s are not interchangeable, and the failure message
    talks about parameter counts rather than about the cause.
    """
    from pghybrid.adapters import for_psycopg, for_sqlalchemy

    kwargs = dict(
        table="chunks",
        text_column="content",
        vector_column="embedding",
        tsvector_column="fts",
        extra_columns=["title"],
    )
    expected = [PLANTED_TITLE, "Renewal pricing", "Renewal terms"]

    assert _adapter_titles(for_psycopg(connection, **kwargs)) == expected

    sqlalchemy = pytest.importorskip("sqlalchemy")
    engine = sqlalchemy.create_engine(DSN.replace("postgresql://", "postgresql+psycopg://"))
    try:
        with engine.connect() as sa_connection:
            assert _adapter_titles(for_sqlalchemy(sa_connection, **kwargs)) == expected
        # An Engine and a Session reach the same place by different routes.
        assert _adapter_titles(for_sqlalchemy(engine, **kwargs)) == expected
        from sqlalchemy.orm import Session

        with Session(engine) as session:
            assert _adapter_titles(for_sqlalchemy(session, **kwargs)) == expected
    finally:
        engine.dispose()


def test_adapters_override_a_wrong_paramstyle(connection):
    """A psycopg connection cannot execute $1 however firmly the config asks for it."""
    from pghybrid.adapters import for_psycopg

    search = for_psycopg(
        connection,
        Config(
            table="chunks",
            text_column="content",
            vector_column="embedding",
            tsvector_column="fts",
            extra_columns=["title"],
            paramstyle="numeric",
        ),
    )
    assert search.config.paramstyle == "pyformat"
    assert _adapter_titles(search)[0] == PLANTED_TITLE


@pytest.mark.asyncio
async def test_asyncpg_adapter_matches_the_sync_result():
    asyncpg = pytest.importorskip("asyncpg")
    from pghybrid.adapters import for_asyncpg

    conn = await asyncpg.connect(DSN)
    try:
        search = for_asyncpg(
            conn,
            table="chunks",
            text_column="content",
            vector_column="embedding",
            tsvector_column="fts",
            extra_columns=["title"],
        )
        results = await search.search(DEMO_QUERY, embedding=query_vector(), limit=3)
    finally:
        await conn.close()
    assert titles(results) == [PLANTED_TITLE, "Renewal pricing", "Renewal terms"]


# ---------------------------------------------------------------- non-English corpora


@pytest.fixture
def french_table(connection):
    """A small French corpus with a French-configured tsvector column."""
    connection.execute("DROP TABLE IF EXISTS chunks_fr CASCADE")
    connection.execute(
        """
        CREATE TABLE chunks_fr (
            id bigserial PRIMARY KEY,
            title text NOT NULL,
            content text NOT NULL,
            embedding vector(8),
            fts tsvector GENERATED ALWAYS AS
                (to_tsvector('french', coalesce(content, ''))) STORED
        )
        """
    )
    rows = [
        (
            "Loyers impayés",
            "Le locataire doit régler les loyers impayés dans un délai de trente jours.",
        ),
        (
            "Préavis de résiliation",
            "Le préavis de résiliation est de trois mois avant la date anniversaire.",
        ),
        (
            "Charges locatives",
            "Les charges locatives sont régularisées annuellement par le bailleur.",
        ),
        (
            "Dépôt de garantie",
            "Le dépôt de garantie est restitué dans les deux mois suivant la remise des clés.",
        ),
    ]
    for index, (title, content) in enumerate(rows):
        connection.execute(
            "INSERT INTO chunks_fr (title, content, embedding) VALUES (%s, %s, %s)",
            (title, content, to_pgvector(unit_vector(0.2 * index))),
        )
    yield "chunks_fr"
    connection.execute("DROP TABLE IF EXISTS chunks_fr CASCADE")


def _french_config(table: str, language: str) -> Config:
    return Config(
        table=table,
        text_column="content",
        vector_column="embedding",
        tsvector_column="fts",
        extra_columns=["title"],
        language=language,
        paramstyle="pyformat",
    )


def test_french_stemming_matches_an_inflected_query(connection, french_table):
    """'loyers impayés' must find 'loyer impayé', which needs French stemming.

    English stemming leaves the French plural and accents alone, so the same query
    against an English-configured query would not match the stored lexemes. Getting the
    text search configuration wrong does not error — it silently returns nothing — which
    is why this is asserted rather than assumed.
    """
    search = HybridSearch(
        _french_config(french_table, "french"),
        execute=lambda sql, params: connection.execute(sql, params).fetchall(),
    )
    results = search.search("loyers impayés", limit=5)
    assert titles(results) and titles(results)[0] == "Loyers impayés"


def test_the_wrong_text_config_degrades_silently(connection, french_table):
    """The failure this project keeps warning about, pinned to what really happens.

    The column is built with 'french', which stores 'impai' for "impayés" and 'loyer'
    for "loyers". Querying it as 'english' yields 'impayé' and 'loyer': one of the two
    terms still matches by luck, because it stems identically in both languages, and the
    other is lost. So the wrong configuration does not fail loudly and does not
    necessarily return nothing — it quietly answers with less than you asked for, which
    is harder to notice.
    """
    french = HybridSearch(
        _french_config(french_table, "french"),
        execute=lambda sql, params: connection.execute(sql, params).fetchall(),
    )
    english = HybridSearch(
        _french_config(french_table, "english"),
        execute=lambda sql, params: connection.execute(sql, params).fetchall(),
    )

    # The term that only French stemming can match.
    assert titles(french.search("impayés", limit=5)) == ["Loyers impayés"]
    assert english.search("impayés", limit=5) == [], (
        "'impayés' stems to 'impai' in the stored column and to 'impayé' under English, "
        "so the English-configured query cannot match it"
    )

    # The term that happens to stem the same either way still matches, which is exactly
    # what makes the misconfiguration survive a casual test.
    assert titles(english.search("loyers", limit=5)) == ["Loyers impayés"]


def test_a_query_config_must_match_the_column_config(connection, french_table):
    """'simple' does no stemming, so it cannot match a stemmed column.

    This is the same lesson from the other direction, and it is why the migration names
    the text search configuration explicitly rather than relying on a database default
    that can be changed underneath the column.
    """
    simple = HybridSearch(
        _french_config(french_table, "simple"),
        execute=lambda sql, params: connection.execute(sql, params).fetchall(),
    )
    # The column stored 'locatair'; 'simple' asks for 'locataire' and finds nothing.
    assert simple.search("locataire", limit=5) == []

    # With a vector alongside, the search still returns rows — from one signal only.
    degraded = simple.search("locataire", embedding=query_vector(), limit=3)
    assert degraded and all(row.text_rank is None for row in degraded)
