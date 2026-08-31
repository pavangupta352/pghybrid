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
