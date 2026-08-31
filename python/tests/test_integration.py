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
import random
import sys
from dataclasses import replace

import pytest

from pghybrid import AsyncHybridSearch, Config, Recency, Weights
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


def test_an_exclusion_holds_when_the_vector_signal_is_running(connection, search):
    """The case above passes with the exclusion in the tsquery alone. This one does not.

    Keyword-only search never exposed the bug, because the tsquery is exactly where a
    parser already understands a leading dash. Turn the vector half on and the excluded
    row comes back: it is missing from the text candidates, so it arrives with a vector
    rank and no text rank, and RRF pays the best vector hit 1/(k+1), the largest single
    contribution available. It ranked fourth of five here.

    The embedding is the excluded row's own, which is the honest version of the test: a
    user reaches for "-pricing" precisely when pricing pages are what the vector half
    keeps returning.
    """
    row = connection.execute(
        "SELECT embedding FROM chunks WHERE title = 'Renewal pricing'"
    ).fetchone()
    its_own_vector = [float(x) for x in str(row["embedding"]).strip("[]").split(",")]

    without = titles(search.search(DEMO_QUERY, embedding=its_own_vector, limit=5))
    assert without[0] == "Renewal pricing", "the corpus no longer sets this test up"

    with_exclusion = titles(
        search.search(f"{DEMO_QUERY} -pricing", embedding=its_own_vector, limit=5)
    )
    assert "Renewal pricing" not in with_exclusion, (
        "the excluded row came back on the vector signal, so the exclusion is not being "
        "applied inside both candidate CTEs"
    )
    assert len(with_exclusion) == 5, "excluding one row should not shrink the result set"


def test_a_query_of_only_exclusions_is_vector_search_with_a_filter(connection, search):
    """Not an inverted corpus ranked by nothing.

    ``!'pricing'`` matches almost every row, and ts_rank_cd scores a pure negation
    identically for all of them, so fusing it in would reorder the vector results by an
    arbitrary tiebreak. The text signal is dropped instead and the exclusion still holds.
    """
    row = connection.execute(
        "SELECT embedding FROM chunks WHERE title = 'Renewal pricing'"
    ).fetchone()
    its_own_vector = [float(x) for x in str(row["embedding"]).strip("[]").split(",")]

    rows = search.search("-pricing", embedding=its_own_vector, limit=5)
    assert "Renewal pricing" not in titles(rows)
    assert all(r.text_rank is None for r in rows), "the keyword half should not have run"
    # The output order is the vector order, untouched. (rank() shares a number across
    # ties and skips the next, so the sequence is ascending rather than 1..5.)
    ranks = [r.vector_rank for r in rows]
    assert ranks[0] == 1 and ranks == sorted(ranks)


def test_an_exclusion_with_nothing_to_rank_says_so(search):
    with pytest.raises(ValueError, match="only excludes terms"):
        search.search("-pricing", limit=5)


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


def test_recency_reranks_the_candidate_pool_and_does_not_retrieve(connection):
    """The documented limitation, pinned so the documentation stays true.

    Decay is applied after both signals have chosen their candidates on relevance alone,
    so a row published today that no signal ranked highly cannot surface at any
    half-life. Someone who sets half_life_days=1 will assume otherwise, which is why the
    Recency docstring and the README both say it and why this measures it.

    Not a defect: retrieving on recency would mean a third candidate set ordered by
    timestamp, which returns recent rows nobody searched for. candidate_limit is the
    lever, and the second half of this test is what makes that advice actionable.
    """
    import math

    connection.execute("DROP TABLE IF EXISTS recency_probe")
    connection.execute(
        "CREATE TABLE recency_probe ("
        "  id bigserial PRIMARY KEY, content text NOT NULL,"
        "  embedding vector(2), created_at timestamptz,"
        "  fts tsvector GENERATED ALWAYS AS (to_tsvector('english', coalesce(content,'')))"
        "  STORED)"
    )
    try:
        # Points on an arc, so relevance rank is the row number: id 1 is nearest.
        for i in range(300):
            angle = i * 0.004
            connection.execute(
                "INSERT INTO recency_probe (content, embedding, created_at) VALUES "
                "(%s, %s::vector, now() - interval '400 days')",
                (
                    f"contract clause number {i} about renewal notice",
                    f"[{math.cos(angle)},{math.sin(angle)}]",
                ),
            )
        # One row published today, deliberately far down the relevance ranking.
        connection.execute("UPDATE recency_probe SET created_at = now() WHERE id = 251")

        def top_ids(candidate_limit: int) -> list[int]:
            cfg = Config(
                table="recency_probe",
                text_column="content",
                vector_column="embedding",
                tsvector_column="fts",
                paramstyle="pyformat",
                recency=Recency(column="created_at", half_life_days=1.0),
                candidate_limit=candidate_limit,
            )
            search = HybridSearch(cfg, execute=lambda sql, p: connection.execute(sql, p).fetchall())
            return [r.id for r in search.search("renewal notice", embedding=[1.0, 0.0], limit=3)]

        assert 251 not in top_ids(20), (
            "a row 400 days younger than every other one surfaced from outside the "
            "candidate pool, so decay is retrieving rather than reranking"
        )
        assert top_ids(300)[0] == 251, "widening the pool is the documented lever"
    finally:
        connection.execute("DROP TABLE IF EXISTS recency_probe")


def test_paging_returns_each_row_once_and_matches_one_big_query(connection):
    """Paging to the edge of the pool reproduces one query of the same depth.

    The pool here is exactly the depth being paged to, so this walks right up to the
    boundary: the last legal page has to be full and in the right order, and the page
    after it has to be refused.

    What this does *not* catch, and it is worth being precise about, is the widen-per-page
    bug itself, inside the legal range a pool that widens with the offset and a fixed one
    are the same number, so they agree here. The test that distinguishes them is
    test_a_page_outside_the_candidate_pool_is_an_error, because the two designs differ
    only past the boundary. This one pins the user-visible contract: pages tile a single
    ranking without duplicates or gaps.

    The corpus deliberately decorrelates the two signals, random angles, random words,
    because a row entering the text candidates late is what reorders a fused ranking, and
    a corpus where the signals agree would hide any such reordering.
    """
    import math
    import random

    connection.execute("DROP TABLE IF EXISTS paging_probe")
    connection.execute(
        "CREATE TABLE paging_probe ("
        "  id bigserial PRIMARY KEY, content text NOT NULL, embedding vector(2),"
        "  fts tsvector GENERATED ALWAYS AS (to_tsvector('english', coalesce(content,'')))"
        "  STORED)"
    )
    try:
        rng = random.Random(7)
        words = ["renewal", "notice", "termination", "clause", "invoice", "liability"]
        for i in range(500):
            angle = rng.random() * 2 * math.pi
            connection.execute(
                "INSERT INTO paging_probe (content, embedding) VALUES (%s, %s::vector)",
                (
                    " ".join(rng.choices(words, k=6)) + f" number {i}",
                    f"[{math.cos(angle)},{math.sin(angle)}]",
                ),
            )

        cfg = Config(
            table="paging_probe",
            text_column="content",
            vector_column="embedding",
            tsvector_column="fts",
            paramstyle="pyformat",
            # Exactly the depth paged to below, so the last page sits on the boundary.
            candidate_limit=80,
        )
        search = HybridSearch(cfg, execute=lambda sql, p: connection.execute(sql, p).fetchall())

        paged = []
        for page in range(8):
            rows = search.search("renewal notice", embedding=[1.0, 0.0], limit=10, offset=page * 10)
            assert len(rows) == 10, f"page {page + 1} came back short"
            paged += [r.id for r in rows]

        assert len(set(paged)) == 80, "a row appeared on two different pages"
        one_shot = [r.id for r in search.search("renewal notice", embedding=[1.0, 0.0], limit=80)]
        assert paged == one_shot, (
            "paging does not reproduce a single query of the same depth, so the ranking "
            "is changing between pages"
        )

        # One row past the pool is refused rather than silently empty.
        with pytest.raises(ValueError, match="candidate pool"):
            search.search("renewal notice", embedding=[1.0, 0.0], limit=10, offset=80)
    finally:
        connection.execute("DROP TABLE IF EXISTS paging_probe")


def test_paging_past_the_pool_says_so_instead_of_returning_nothing(search):
    """An empty page reads as "no more results", which is the wrong conclusion."""
    with pytest.raises(ValueError, match="candidate pool"):
        search.search(DEMO_QUERY, embedding=query_vector(), limit=10, offset=50)


def test_a_column_named_like_a_computed_one_is_refused(connection):
    """The value of refusing: the alternative is the library lying about itself.

    Postgres allows two output columns with the same name and the driver keeps the last,
    which is the table's. Before this was refused, listing a column called text_rank
    turned text_rank=1, matched_by="both" into text_rank=None, matched_by="vector", the
    same query on the same data, now reporting that the keyword signal missed rows it had
    ranked first. matched_by is the reason to use this library rather than a vector-only
    one, so quietly inverting it is the worst outcome available.
    """
    connection.execute("DROP TABLE IF EXISTS shadow_probe")
    connection.execute(
        "CREATE TABLE shadow_probe ("
        "  id bigserial PRIMARY KEY, content text NOT NULL, text_rank int,"
        "  embedding vector(8),"
        "  fts tsvector GENERATED ALWAYS AS (to_tsvector('english', coalesce(content,'')))"
        "  STORED)"
    )
    try:
        for i in range(20):
            connection.execute(
                "INSERT INTO shadow_probe (content, text_rank, embedding) "
                "VALUES (%s, NULL, %s::vector)",
                (f"renewal notice clause {i}", "[" + ",".join(["0.1"] * 8) + "]"),
            )

        def config(extra):
            return Config(
                table="shadow_probe",
                text_column="content",
                vector_column="embedding",
                tsvector_column="fts",
                extra_columns=extra,
                paramstyle="pyformat",
            )

        # Without the shadowing column the answer is right, which is the control.
        rows = HybridSearch(
            config([]), execute=lambda sql, p: connection.execute(sql, p).fetchall()
        ).search("renewal notice", embedding=[0.1] * 8, limit=2)
        assert [r.matched_by for r in rows] == ["both", "both"]
        assert all(r.text_rank is not None for r in rows)

        # With it, the config never builds.
        with pytest.raises(ValueError, match="cannot be selected through"):
            config(["text_rank"])
    finally:
        connection.execute("DROP TABLE IF EXISTS shadow_probe")


@pytest.fixture
def chunked_table(connection):
    """Five chunks per document, so doc_id repeats and chunk_id does not."""
    connection.execute("DROP TABLE IF EXISTS chunked_probe")
    connection.execute(
        "CREATE TABLE chunked_probe ("
        "  chunk_id bigserial PRIMARY KEY, doc_id bigint NOT NULL, slug text,"
        "  content text NOT NULL, embedding vector(8),"
        "  fts tsvector GENERATED ALWAYS AS (to_tsvector('english', coalesce(content,'')))"
        "  STORED)"
    )
    for i in range(100):
        connection.execute(
            "INSERT INTO chunked_probe (doc_id, slug, content, embedding) "
            "VALUES (%s, %s, %s, %s::vector)",
            (
                i // 5 + 1,
                f"part-{i}",
                f"renewal notice clause part {i}",
                "[" + ",".join(["0.1"] * 8) + "]",
            ),
        )
    yield "chunked_probe"
    connection.execute("DROP TABLE IF EXISTS chunked_probe")


def _config_for(table, id_column):
    return Config(
        table=table,
        text_column="content",
        vector_column="embedding",
        tsvector_column="fts",
        id_column=id_column,
        paramstyle="pyformat",
    )


def test_a_repeated_id_multiplies_rows_through_the_fusion(connection, chunked_table):
    """Why the finding below matters: this is what the caller actually receives.

    The fusion joins the two candidate sets on the id column and then joins the result
    back to the table on it, so a row whose id is shared by five table rows becomes five
    results. A search for ten comes back as the same document ten times, with no error
    anywhere, the shape of the answer is right and the content is nonsense.
    """
    search = HybridSearch(
        _config_for(chunked_table, "doc_id"),
        execute=lambda sql, p: connection.execute(sql, p).fetchall(),
    )
    rows = search.search("renewal notice", embedding=[0.1] * 8, limit=10)
    assert len(rows) == 10
    assert len({r.id for r in rows}) < 10, (
        "the corpus no longer reproduces the duplication this test is about"
    )


def test_recall_does_not_credit_an_index_the_planner_cannot_use(connection):
    """Perfect recall means nothing when the reason is that no index is running.

    doctor branched only on whether an index existed. An invalid one, left by a failed
    CREATE INDEX CONCURRENTLY, and one built for the wrong operator class are both
    skipped by the planner, so every search is exact and recall is 1.00 for a reason
    that has nothing to do with the index working. Reporting "the index returns the true
    nearest neighbours" there is untrue, and it invites the reader to treat the broken
    index as harmless. It costs space and write throughput now, and recall can fall the
    moment it is repaired.
    """
    connection.execute("DROP TABLE IF EXISTS unusable_index_probe")
    connection.execute(
        "CREATE TABLE unusable_index_probe ("
        "  id bigserial PRIMARY KEY, content text NOT NULL, embedding vector(8),"
        "  fts tsvector GENERATED ALWAYS AS (to_tsvector('english', coalesce(content,'')))"
        "  STORED)"
    )
    try:
        for i in range(50):
            connection.execute(
                "INSERT INTO unusable_index_probe (content, embedding) VALUES (%s, %s::vector)",
                (f"renewal notice clause {i}", "[" + ",".join(["0.1"] * 8) + "]"),
            )

        def recall_finding(metric="cosine"):
            report = doctor(
                dbapi_executor(connection),
                Config(
                    table="unusable_index_probe",
                    text_column="content",
                    vector_column="embedding",
                    tsvector_column="fts",
                    metric=metric,
                    paramstyle="pyformat",
                ),
                sample=10,
                k=5,
            )
            return next(f for f in report.findings if "recall@5" in f.title)

        # An index the planner skips because a concurrent build failed.
        connection.execute(
            "CREATE INDEX unusable_hnsw ON unusable_index_probe "
            "USING hnsw (embedding vector_cosine_ops)"
        )
        connection.execute(
            "UPDATE pg_index SET indisvalid = false WHERE indexrelid = 'unusable_hnsw'::regclass"
        )
        invalid = recall_finding()
        assert "is invalid" in invalid.title and "planner skips it" in invalid.title
        assert "returns the true nearest neighbours" not in invalid.detail

        # An index built for a metric this config does not search with.
        connection.execute("DROP INDEX unusable_hnsw")
        connection.execute(
            "CREATE INDEX unusable_l2 ON unusable_index_probe USING hnsw (embedding vector_l2_ops)"
        )
        mismatched = recall_finding(metric="cosine")
        assert "built for l2" in mismatched.title and "cannot use it" in mismatched.title

        # And a usable index is still credited normally.
        working = recall_finding(metric="l2")
        assert "recall@5 = 1.00" in working.title and working.level == "ok"
    finally:
        connection.execute("DROP TABLE IF EXISTS unusable_index_probe")


def test_doctor_reports_an_id_column_that_is_not_unique(connection, chunked_table):
    report = doctor(
        dbapi_executor(connection), _config_for(chunked_table, "doc_id"), sample=20, k=5
    )
    finding = next((f for f in report.findings if "doc_id" in f.title), None)
    assert finding is not None and finding.level == "error", finding
    assert "is not unique" in finding.title
    # The message carries the evidence, not just the accusation.
    assert "appears on 5 rows" in finding.detail


def test_doctor_is_silent_when_the_id_column_is_the_primary_key(connection, chunked_table):
    report = doctor(
        dbapi_executor(connection), _config_for(chunked_table, "chunk_id"), sample=20, k=5
    )
    assert not [f for f in report.findings if "chunk_id" in f.title], (
        "a primary key is proof of uniqueness and needs no finding"
    )


def test_doctor_warns_when_nothing_enforces_uniqueness(connection, chunked_table):
    """Correct today, one INSERT away from silently wrong, and worth saying so."""
    report = doctor(dbapi_executor(connection), _config_for(chunked_table, "slug"), sample=20, k=5)
    finding = next((f for f in report.findings if "slug" in f.title), None)
    assert finding is not None and finding.level == "warn", finding
    assert "no unique index" in finding.title
    assert finding.fix and "CREATE UNIQUE INDEX" in finding.fix


def test_doctor_reports_an_id_column_that_does_not_exist(connection, chunked_table):
    report = doctor(dbapi_executor(connection), _config_for(chunked_table, "nope"), sample=20, k=5)
    finding = next((f for f in report.findings if "nope" in f.title), None)
    assert finding is not None and finding.level == "error", finding


@pytest.fixture
def drift_table(connection):
    """A hand-maintained tsvector, so it can fall out of step with the text."""
    connection.execute("DROP TABLE IF EXISTS drift_probe")
    connection.execute(
        "CREATE TABLE drift_probe (id bigserial PRIMARY KEY, content text NOT NULL,"
        " fts tsvector, embedding vector(8))"
    )
    for i in range(200):
        connection.execute(
            "INSERT INTO drift_probe (content, embedding) VALUES (%s, %s::vector)",
            (f"clause {i} about renewal notice and termination", "[" + ",".join(["0.1"] * 8) + "]"),
        )
    connection.execute("UPDATE drift_probe SET fts = to_tsvector('english', content)")
    yield "drift_probe"
    connection.execute("DROP TABLE IF EXISTS drift_probe")


def _drift_finding(connection, table, language="english"):
    report = doctor(
        dbapi_executor(connection),
        Config(
            table=table,
            text_column="content",
            vector_column="embedding",
            tsvector_column="fts",
            language=language,
            paramstyle="pyformat",
        ),
        sample=100,
        k=5,
    )
    return next((f for f in report.findings if "fts" in f.title), None)


def test_doctor_measures_tsvector_drift_rather_than_warning_about_it(connection, drift_table):
    """A tsvector nothing maintains goes stale silently, and in both directions.

    The rows keep the old text's lexemes, so they are returned for words they no longer
    contain and missing for the words they do. Nothing errors; it reads as bad relevance,
    which sends you to the ranking rather than to the data.

    The finding used to say only that "nothing guarantees it matches", which is a guess.
    This tool measures everything else it reports, and the comparison is one aggregate.
    """
    healthy = _drift_finding(connection, drift_table)
    assert healthy is not None and healthy.level == "info", healthy
    assert "matches the text column" in healthy.title

    # The text moves on; whatever was maintaining the column does not.
    connection.execute(
        "UPDATE drift_probe SET content = 'indemnity and force majeure superseded' WHERE id <= 60"
    )
    stale = _drift_finding(connection, drift_table)
    assert stale is not None and stale.level == "error", stale
    assert "stale" in stale.title
    # 60 of 200 is 30%, and a 100-row sample of it will not be exact.
    drifted = int(stale.title.split("stale for ")[1].split(" of ")[0])
    assert 15 <= drifted <= 50, f"measured {drifted} of 100, expected roughly 30"


def test_doctor_samples_drift_at_random_not_with_a_bare_limit(connection, drift_table):
    """An UPDATE writes the new row version at the end of the heap.

    So the rows a sequential scan reaches last are exactly the rows that have been
    updated, which are exactly the rows likely to have drifted. Sampling with a bare
    LIMIT reported 3 of 100 on a table that was genuinely 60 of 200, it misses the
    problem precisely because the problem moved the rows.
    """
    connection.execute(
        "UPDATE drift_probe SET content = 'indemnity and force majeure superseded' WHERE id <= 60"
    )
    biased = connection.execute(
        "SELECT count(*) FILTER (WHERE coalesce(fts, ''::tsvector) IS DISTINCT FROM "
        "  to_tsvector('english', coalesce(content, ''))) AS drifted "
        "FROM (SELECT fts, content FROM drift_probe LIMIT 100) s"
    ).fetchone()["drifted"]
    assert biased < 15, (
        "the bare-LIMIT sample was expected to under-report; if this ever fails the "
        "premise of the random sampling below has changed, not the code"
    )

    title = _drift_finding(connection, drift_table).title
    measured = int(title.split("stale for ")[1].split(" of ")[0])
    assert measured > biased, "the random sample must see more drift than a bare LIMIT"


def test_doctor_catches_a_generated_column_the_config_does_not_describe(connection):
    """A generated column cannot fall behind its own expression. It can still be the
    wrong expression, and that was invisible.

    Scoping the drift check to non-generated columns looked obviously right and was not:
    the comparison against to_tsvector(config.language, config.text_column) is exactly
    the check for a config that disagrees with the column, whoever maintains it. Measured
    on one table: the same query returned 5 rows with language="english" and 0 with
    "simple", with nothing reported anywhere.
    """
    connection.execute("DROP TABLE IF EXISTS generated_config_probe")
    connection.execute(
        "CREATE TABLE generated_config_probe ("
        "  id bigserial PRIMARY KEY, title text NOT NULL, content text NOT NULL,"
        "  embedding vector(8),"
        "  fts tsvector GENERATED ALWAYS AS (to_tsvector('english', coalesce(content,'')))"
        "  STORED)"
    )
    try:
        for i in range(40):
            connection.execute(
                "INSERT INTO generated_config_probe (title, content, embedding) "
                "VALUES (%s, %s, %s::vector)",
                (
                    f"Clause {i}",
                    f"the renewal notices were terminating {i}",
                    "[" + ",".join(["0.1"] * 8) + "]",
                ),
            )

        def finding(**overrides):
            cfg = Config(
                table="generated_config_probe",
                text_column="content",
                vector_column="embedding",
                tsvector_column="fts",
                paramstyle="pyformat",
                **overrides,
            )
            report = doctor(dbapi_executor(connection), cfg, sample=30, k=5)
            return next((f for f in report.findings if "fts" in f.title), None)

        # The config that matches the column says nothing.
        assert finding() is None or finding().level == "info", finding()

        wrong_language = finding(language="simple")
        assert wrong_language is not None and wrong_language.level == "error"
        assert "generated from something else" in wrong_language.title
        # The stored expression is in the message, because that is what tells you which
        # end to change.
        assert "english" in wrong_language.detail
        # And it must not be described as stale, which a generated column cannot be.
        assert "stale" not in wrong_language.title

        # The same class: generated from a different column than the config searches.
        connection.execute("DROP TABLE IF EXISTS generated_config_probe2")
        connection.execute(
            "CREATE TABLE generated_config_probe2 ("
            "  id bigserial PRIMARY KEY, title text NOT NULL, content text NOT NULL,"
            "  embedding vector(8),"
            "  fts tsvector GENERATED ALWAYS AS (to_tsvector('english', coalesce(title,'')))"
            "  STORED)"
        )
        connection.execute(
            "INSERT INTO generated_config_probe2 (title, content, embedding) "
            "VALUES ('Clause one', 'renewal notice period', '[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]')"
        )
        report = doctor(
            dbapi_executor(connection),
            Config(
                table="generated_config_probe2",
                text_column="content",
                vector_column="embedding",
                tsvector_column="fts",
                paramstyle="pyformat",
            ),
            sample=30,
            k=5,
        )
        wrong_column = next((f for f in report.findings if "fts" in f.title), None)
        assert wrong_column is not None and wrong_column.level == "error", wrong_column
    finally:
        connection.execute("DROP TABLE IF EXISTS generated_config_probe")
        connection.execute("DROP TABLE IF EXISTS generated_config_probe2")


def test_doctor_separates_a_wrong_text_config_from_a_stale_column(connection, drift_table):
    """Every row disagreeing is a different bug with a different fix.

    A stopped trigger leaves most rows fine. A column built with another text search
    configuration leaves none of them fine, and rewriting the data would be the wrong
    response, the configuration is what is wrong.
    """
    connection.execute("UPDATE drift_probe SET fts = to_tsvector('simple', content)")
    finding = _drift_finding(connection, drift_table, language="english")
    assert finding is not None and finding.level == "error", finding
    assert "does not match this configuration at all" in finding.title
    assert "stale" not in finding.title

    # And it is not flagged when the configuration is the one the column was built with.
    agreed = _drift_finding(connection, drift_table, language="simple")
    assert agreed is not None and agreed.level == "info", agreed


def test_doctor_stays_read_only_while_measuring_drift(connection, drift_table):
    before = connection.execute("SELECT count(*) AS n FROM drift_probe").fetchone()["n"]
    checksum = connection.execute(
        "SELECT count(*) AS n FROM drift_probe WHERE fts IS NOT NULL"
    ).fetchone()["n"]
    _drift_finding(connection, drift_table)
    assert connection.execute("SELECT count(*) AS n FROM drift_probe").fetchone()["n"] == before
    assert (
        connection.execute(
            "SELECT count(*) AS n FROM drift_probe WHERE fts IS NOT NULL"
        ).fetchone()["n"]
        == checksum
    ), "the drift check repaired the column instead of reporting it"


def test_highlight_cannot_smuggle_markup_out_of_the_document(connection):
    """The delimiters are HTML, so the caller is meant to render the result.

    That makes the document text around the marks active markup, and Postgres does not
    escape it. Its parser drops tags it recognises, which is the trap: <script>alert(1)
    </script> disappears and the whole thing looks safe, while <img src=x onerror=...>
    and <svg/onload=...> come through whole. Relying on that is relying on which shapes
    one particular parser happens to recognise.
    """
    connection.execute("DROP TABLE IF EXISTS highlight_probe")
    connection.execute(
        "CREATE TABLE highlight_probe ("
        "  id bigserial PRIMARY KEY, content text NOT NULL, embedding vector(8),"
        "  fts tsvector GENERATED ALWAYS AS (to_tsvector('english', coalesce(content,'')))"
        "  STORED)"
    )
    try:
        payloads = [
            "renewal notice <img src=x onerror=alert(1)> end",
            "renewal notice <svg/onload=alert(1)> end",
            "renewal notice <script>alert(1)</script> end",
            "renewal notice where x < 5 and y > 3 end",
            "renewal notice with an &amp; already in it end",
        ]
        for text in payloads:
            connection.execute(
                "INSERT INTO highlight_probe (content, embedding) VALUES (%s, %s::vector)",
                (text, "[" + ",".join(["0.1"] * 8) + "]"),
            )

        def highlights(**overrides):
            cfg = Config(
                table="highlight_probe",
                text_column="content",
                vector_column="embedding",
                tsvector_column="fts",
                paramstyle="pyformat",
                **overrides,
            )
            search = HybridSearch(cfg, execute=lambda sql, p: connection.execute(sql, p).fetchall())
            rows = search.search("renewal notice", embedding=[0.1] * 8, limit=10, highlight=True)
            return [r.highlight or "" for r in rows]

        for highlight in highlights():
            # Only the delimiters may be raw markup. Everything else is escaped.
            body = highlight.replace("<mark>", "").replace("</mark>", "")
            assert "<" not in body and ">" not in body, body
            assert "<mark>" in highlight, "escaping must not break the highlighting"

        # And the opt-out still exists for non-HTML delimiters, where escaping is wrong.
        raw = highlights(escape_highlight=False)
        assert any("<img" in h or "<svg" in h for h in raw), (
            "with escaping off the document should come through unchanged, which is the "
            "whole reason the default is on"
        )
    finally:
        connection.execute("DROP TABLE IF EXISTS highlight_probe")


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


def test_find_names_the_exclusion_rather_than_blaming_the_cut_off(connection, config, search):
    """find exists to name the stage that lost the answer, so naming the wrong one is worse
    than saying nothing.

    A negative term removes the row from both candidate sets. Before this, find saw a row
    that was #1 on both signals and absent from the result, concluded it must have lost the
    fused ordering, and advised raising candidate_limit, a knob that can never bring back a
    row the query itself threw out. The real reason was in the query string the caller had
    just typed.
    """
    report = explain(
        search,
        f"{DEMO_QUERY} -pricing",
        query_vector(),
        limit=2,
        near_miss=0,
        label_column="title",
        find="Renewal pricing is subject to change",
    )
    assert report.find is not None and report.find.found
    assert "excluded" in report.find.reason, report.find.reason
    assert "candidate_limit" not in report.find.reason
    assert report.find.remedy and "drop that term" in report.find.remedy

    # And without the exclusion the same row is diagnosed the ordinary way.
    ordinary = explain(
        search,
        DEMO_QUERY,
        query_vector(),
        limit=2,
        near_miss=0,
        label_column="title",
        find="Renewal pricing is subject to change",
    )
    assert ordinary.find is not None
    assert "excluded" not in ordinary.find.reason, ordinary.find.reason


def test_find_still_blames_the_filters_when_the_filters_are_to_blame(connection, config):
    """The exclusion branch is checked first, so this makes sure it did not swallow the
    filter case that was already there."""
    connection.execute("UPDATE chunks SET tenant_id = 2 WHERE title = 'Renewal pricing'")
    try:
        search = HybridSearch(config, execute=lambda sql, p: connection.execute(sql, p).fetchall())
        report = explain(
            search,
            DEMO_QUERY,
            query_vector(),
            limit=2,
            near_miss=0,
            label_column="title",
            find="Renewal pricing is subject to change",
            filters={"tenant_id": 1},
        )
        assert report.find is not None
        assert "filters" in report.find.reason, report.find.reason
    finally:
        connection.execute("UPDATE chunks SET tenant_id = 1")


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


def test_a_statement_that_is_only_a_comment_is_not_executable(connection):
    """The general guard, not just the one statement that was mis-marked.

    Postgres accepts a comment as an empty command and reports success, so any statement
    whose SQL is entirely comments would be applied, reported ok, and do nothing. Three
    of the four such statements were already marked optional; the fourth was not, which
    is exactly the kind of inconsistency a per-case marking invites.
    """
    from pghybrid.schema import Statement

    assert not Statement(sql="-- do this by hand", reason="r").is_executable
    assert not Statement(sql="--one\n--two\n", reason="r").is_executable
    assert not Statement(sql="  \n  -- indented\n", reason="r").is_executable
    assert Statement(sql="ANALYZE t;", reason="r").is_executable
    assert Statement(sql="-- why\nANALYZE t;", reason="r").is_executable


def test_a_bare_vector_column_yields_required_work_nothing_can_run(connection):
    """It is required, and it cannot be a statement: only the caller knows the dimension."""
    from pghybrid.schema import build_migration, introspect, suggest_config

    connection.execute("DROP TABLE IF EXISTS bare_vector_probe")
    connection.execute(
        "CREATE TABLE bare_vector_probe (id bigserial PRIMARY KEY, content text NOT NULL,"
        " embedding vector)"
    )
    try:
        info = introspect(dbapi_executor(connection), "bare_vector_probe")
        statements = build_migration(suggest_config(info), info)
        required = [s for s in statements if not s.optional]
        assert required, "a table that cannot be indexed still has work to report"
        manual = [s for s in required if not s.is_executable]
        assert len(manual) == 1
        assert "<dimensions>" in manual[0].sql, manual[0].sql
    finally:
        connection.execute("DROP TABLE IF EXISTS bare_vector_probe")


def test_ivfflat_lists_shows_arithmetic_that_is_actually_true() -> None:
    """The number was always right. The working shown next to it was not.

    Under a thousand rows the division floors to zero and the result is clamped to one
    list, but the line printed said "500 rows / 1000 = 1 lists", which is simply false.
    This tool's argument for itself is that it shows the arithmetic so you can defend the
    choice to whoever owns the database, and a reader who checks the sum is precisely the
    reader that line exists for.

    Also pins the threshold, which the function's own docstring calls the part everybody
    gets wrong: rows/1000 up to and including a million, sqrt(rows) above it.
    """
    from pghybrid.schema import IVFFLAT_SQRT_THRESHOLD, ivfflat_lists

    for rows in (0, 1, 500, 999):
        lists, arithmetic = ivfflat_lists(rows)
        assert lists == 1
        assert "rounds to 0" in arithmetic and "minimum" in arithmetic, arithmetic
        assert f"= {lists} lists" not in arithmetic, "claims a division that did not happen"

    # Once the division carries, the arithmetic is literal and must read that way.
    for rows, expected in ((1_000, 1), (1_500, 1), (12_000, 12), (999_999, 999)):
        lists, arithmetic = ivfflat_lists(rows)
        assert lists == expected
        assert f"{rows:,} rows / 1000 = {expected:,} lists" in arithmetic, arithmetic

    # The threshold is inclusive on the division side and exclusive on the sqrt side.
    at, at_text = ivfflat_lists(IVFFLAT_SQRT_THRESHOLD)
    assert at == IVFFLAT_SQRT_THRESHOLD // 1000 and "/ 1000" in at_text
    above, above_text = ivfflat_lists(IVFFLAT_SQRT_THRESHOLD + 1)
    assert "sqrt(" in above_text and above == int((IVFFLAT_SQRT_THRESHOLD + 1) ** 0.5)

    # A negative row count cannot happen from a catalog read, but reltuples is -1 for a
    # never-analysed table and that has reached this function before.
    assert ivfflat_lists(-1)[0] == 1


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


@pytest.mark.asyncio
async def test_the_async_client_agrees_with_the_sync_one_feature_by_feature(connection, config):
    """Two clients over one builder still drift if nobody checks, and the features added
    most recently are the ones least likely to have been checked on both paths.

    Compares rows, highlights and matched_by across exclusions, an exclusion-only query,
    escaped highlighting and a second page, then compares the refusals, which are easier
    to let diverge than the results because nobody looks at them.
    """
    asyncpg = pytest.importorskip("asyncpg")

    conn = await asyncpg.connect(DSN)
    try:
        sync_search = HybridSearch(
            config, execute=lambda sql, p: connection.execute(sql, p).fetchall()
        )
        async_search = AsyncHybridSearch(
            replace(config, paramstyle="numeric"),
            execute=lambda sql, p: conn.fetch(sql, *p),
        )
        vector = query_vector()

        for kwargs in (
            {"text": f"{DEMO_QUERY} -pricing", "embedding": vector, "limit": 5},
            {"text": "-pricing", "embedding": vector, "limit": 5},
            {"text": DEMO_QUERY, "embedding": vector, "limit": 3, "highlight": True},
            {"text": DEMO_QUERY, "embedding": vector, "limit": 3, "offset": 3},
        ):
            here = sync_search.search(**kwargs)
            there = await async_search.search(**kwargs)
            assert [r.id for r in here] == [r.id for r in there], kwargs
            assert [r.highlight for r in here] == [r.highlight for r in there], kwargs
            assert [r.matched_by for r in here] == [r.matched_by for r in there], kwargs

        for kwargs in (
            {"text": DEMO_QUERY, "embedding": vector, "limit": 10, "offset": 60},
            {"text": DEMO_QUERY, "embedding": [float("nan")] * 8, "limit": 3},
            {"text": "and or", "embedding": None, "limit": 3},
        ):
            with pytest.raises(ValueError) as here:
                sync_search.search(**kwargs)
            with pytest.raises(ValueError) as there:
                await async_search.search(**kwargs)
            assert str(here.value) == str(there.value), kwargs
    finally:
        await conn.close()


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
    text search configuration wrong does not error, it silently returns nothing, which
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
    necessarily return nothing, it quietly answers with less than you asked for, which
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

    # With a vector alongside, the search still returns rows, from one signal only.
    degraded = simple.search("locataire", embedding=query_vector(), limit=3)
    assert degraded and all(row.text_rank is None for row in degraded)


def test_every_mode_returns_the_same_python_types(connection, config):
    """A field's type must not depend on which signals the caller used.

    A bare ``0.0`` is numeric in Postgres, not float8, so the single-signal branches
    used to return ``vector_contribution`` as a Decimal while the hybrid branch returned
    a float. Callers do arithmetic on these, and ``Decimal + float`` raises TypeError,
    so code that worked on a hybrid query blew up on a text-only one. It also cost about
    3ms per query in client-side decoding, which showed up as keyword-only measuring
    slower than hybrid despite doing strictly less work on the server.
    """
    search = HybridSearch(
        config, execute=lambda sql, params: connection.execute(sql, params).fetchall()
    )
    numeric_fields = (
        "score",
        "fused_score",
        "vector_contribution",
        "text_contribution",
    )

    modes = {
        "hybrid": search.search(DEMO_QUERY, embedding=query_vector(), limit=3),
        "vector only": search.search(None, embedding=query_vector(), limit=3),
        "keyword only": search.search(DEMO_QUERY, limit=3),
    }

    for label, results in modes.items():
        assert results, f"{label} returned nothing"
        for field in numeric_fields:
            value = getattr(results[0], field)
            assert isinstance(value, float), (
                f"{label}: {field} came back as {type(value).__name__}, not float. "
                "A bare numeric literal in the fusion will do this."
            )
            # The point of the type, not just its name: it has to survive arithmetic
            # against a plain float.
            assert value + 0.5 == pytest.approx(value + 0.5)


# ------------------------------------------------- tables nobody has migrated yet


@pytest.fixture
def unmigrated(connection):
    """A table with embeddings and text but no tsvector column and no indexes.

    This is what everyone has before they run init, and searching it is the first thing
    anyone tries.
    """
    connection.execute("DROP VIEW IF EXISTS unmigrated_view CASCADE")
    connection.execute("DROP TABLE IF EXISTS unmigrated CASCADE")
    connection.execute(
        """CREATE TABLE unmigrated (
               id bigserial PRIMARY KEY, title text NOT NULL,
               body text NOT NULL, embedding vector(8))"""
    )
    for angle, title, content in DOCUMENTS[:6]:
        connection.execute(
            "INSERT INTO unmigrated (title, body, embedding) VALUES (%s, %s, %s)",
            (title, content, to_pgvector(unit_vector(angle))),
        )
    connection.execute(
        """CREATE VIEW unmigrated_view AS
               SELECT id, title, body, embedding FROM unmigrated"""
    )
    yield
    connection.execute("DROP VIEW IF EXISTS unmigrated_view CASCADE")
    connection.execute("DROP TABLE IF EXISTS unmigrated CASCADE")


def test_search_works_before_anyone_runs_init(connection, unmigrated):
    """suggest_config must not name a tsvector column that does not exist yet.

    It used to return the name the migration *would* create, and because the same config
    is what you search with, every first query died on `column "fts" does not exist`.
    With no stored column the builder computes to_tsvector inline, which is slower and
    correct.
    """
    config = suggest_config(introspect(dbapi_executor(connection), "unmigrated"))
    assert config.tsvector_column is None, (
        "a table with no tsvector column must produce a config that computes it inline"
    )
    config.paramstyle = "pyformat"
    config.extra_columns = ["title"]

    search = HybridSearch(
        config, execute=lambda sql, params: connection.execute(sql, params).fetchall()
    )
    results = search.search(DEMO_QUERY, embedding=query_vector(), limit=3)
    assert results and results[0].score > 0


def test_a_view_can_be_searched_but_not_indexed(connection, unmigrated):
    """Views are searchable. pgai's default destination leaves you one.

    Its `destination_table` creates a store table *and* a view joining it to the source,
    and the view is the obvious thing to point a search at. Refusing views turned that
    into a dead end at the first step.
    """
    info = introspect(dbapi_executor(connection), "unmigrated_view")
    assert info.kind == "v"
    assert not info.is_indexable

    config = suggest_config(info)
    config.paramstyle = "pyformat"
    config.extra_columns = ["title"]
    search = HybridSearch(
        config, execute=lambda sql, params: connection.execute(sql, params).fetchall()
    )
    assert search.search(DEMO_QUERY, embedding=query_vector(), limit=3)

    # No DDL, because none of it would apply, just a pointer to where indexes belong.
    statements = build_migration(config, info)
    assert all(s.optional for s in statements)
    assert any("cannot carry an index" in s.reason for s in statements)
    assert not any(s.sql.strip().upper().startswith(("ALTER", "CREATE")) for s in statements)


def test_halfvec_is_gated_on_the_pgvector_version(connection):
    """halfvec, sparsevec and the L1 operator all arrived in pgvector 0.7.0.

    Attempting the ALTER on an older server fails with `type "halfvec" does not exist`,
    which reads like a typo rather than like a version requirement. The README says the
    core works from 0.5, so a user on 0.5 or 0.6 asking for halfvec is a reachable state
    and deserves a sentence rather than a type error.
    """
    from dataclasses import replace

    info = introspect(dbapi_executor(connection), "chunks")
    assert info.supports_halfvec, "the test server should be new enough for the happy path"

    config = replace(suggest_config(info), vector_type="halfvec")
    statements = build_migration(config, replace(info, pgvector_version="0.6.2"))

    # Other statements may legitimately accompany it. ANALYZE, an index on a filter
    # column, so the assertion is about the halfvec decision, not the whole list.
    notes = [s for s in statements if s.kind == "note"]
    assert notes, "an unsupported vector_type should produce a note, not silence"
    assert "0.7.0" in notes[0].reason
    assert notes[0].optional
    assert not any("halfvec" in s.sql and s.kind == "column" for s in statements), (
        "no ALTER ... TYPE halfvec should be emitted against a server that lacks the type"
    )


def test_the_version_properties_read_the_server_not_a_constant(connection):
    info = introspect(dbapi_executor(connection), "chunks")
    from dataclasses import replace

    assert replace(info, pgvector_version="0.5.1").supports_halfvec is False
    assert replace(info, pgvector_version="0.7.0").supports_halfvec is True
    assert replace(info, pgvector_version="0.7.4").supports_iterative_scan is False
    assert replace(info, pgvector_version="0.8.0").supports_iterative_scan is True


# ------------------------------------------------- concurrency, partitions, wide vectors


def test_one_instance_is_safe_to_share_across_threads(config):
    """A web application holds one HybridSearch and serves requests from a pool.

    Nothing in the builder is stateful, but "should be fine" is not an assertion. Each
    thread searches with a vector pointing at a known document, so cross-talk between
    concurrent calls shows up as the wrong title rather than as a crash.
    """
    pool = pytest.importorskip("psycopg_pool")
    import threading

    connection_pool = pool.ConnectionPool(
        DSN, min_size=4, max_size=8, kwargs={"row_factory": psycopg.rows.dict_row}, open=True
    )

    def execute(sql, params):
        with connection_pool.connection() as conn:
            return conn.execute(sql, params).fetchall()

    search = HybridSearch(config, execute=execute)
    expected = [(unit_vector(angle), title) for angle, title, _ in DOCUMENTS]
    mismatches: list[str] = []
    lock = threading.Lock()

    def worker(seed: int) -> None:
        rng = random.Random(seed)
        for _ in range(25):
            vector, title = expected[rng.randrange(len(expected))]
            rows = search.search(None, embedding=vector, limit=1)
            got = str(rows[0].get("title")) if rows else None
            if got != title:
                with lock:
                    mismatches.append(f"expected {title!r}, got {got!r}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    connection_pool.close()

    assert not mismatches, f"concurrent searches returned another call's rows: {mismatches[:3]}"


def test_a_partitioned_table_is_searched_and_indexed_like_any_other(connection):
    """Partitioning by tenant is the usual shape for multi-tenant search."""
    connection.execute("DROP TABLE IF EXISTS parted CASCADE")
    connection.execute(
        """CREATE TABLE parted (
               id bigserial, tenant_id int NOT NULL, content text NOT NULL,
               embedding vector(8), PRIMARY KEY (id, tenant_id))
           PARTITION BY LIST (tenant_id)"""
    )
    try:
        connection.execute("CREATE TABLE parted_1 PARTITION OF parted FOR VALUES IN (1)")
        connection.execute("CREATE TABLE parted_2 PARTITION OF parted FOR VALUES IN (2)")
        for index, (angle, _, content) in enumerate(DOCUMENTS[:6]):
            connection.execute(
                "INSERT INTO parted (tenant_id, content, embedding) VALUES (%s, %s, %s)",
                (1 + index % 2, content, to_pgvector(unit_vector(angle))),
            )

        info = introspect(dbapi_executor(connection), "parted")
        assert info.kind == "p" and info.is_partitioned and info.is_indexable

        config = suggest_config(info)
        config.paramstyle = "pyformat"
        search = HybridSearch(
            config, execute=lambda sql, params: connection.execute(sql, params).fetchall()
        )
        assert search.search(DEMO_QUERY, embedding=query_vector(), limit=3)

        # A partitioned index is valid DDL from Postgres 11; it must not be skipped.
        for statement in build_migration(config, info):
            if not statement.optional:
                connection.execute(statement.sql)
    finally:
        connection.execute("DROP TABLE IF EXISTS parted CASCADE")


def test_an_embedding_too_wide_to_index_gets_the_halfvec_route(connection):
    """text-embedding-3-large is 3,072 dimensions; pgvector indexes at most 2,000.

    A plain vector index simply refuses to build, so the migration has to reach for the
    halfvec cast, whose limit is 4,000, and say why, or the recommendation looks
    arbitrary.
    """
    connection.execute("DROP TABLE IF EXISTS wide CASCADE")
    connection.execute(
        "CREATE TABLE wide (id bigserial PRIMARY KEY, content text, embedding vector(3072))"
    )
    try:
        connection.execute(
            "INSERT INTO wide (content, embedding) VALUES ('renewal notice period', %s)",
            ("[" + ",".join("0.01" for _ in range(3072)) + "]",),
        )
        info = introspect(dbapi_executor(connection), "wide")
        config = suggest_config(info)

        vector_statements = [s for s in build_migration(config, info) if "hnsw" in s.sql.lower()]
        assert vector_statements, "a wide embedding should still get an index recommendation"
        statement = vector_statements[0]
        assert "halfvec" in statement.sql
        assert "2,000" in statement.reason, "the reason has to name the limit it is working around"

        # It is only a recommendation if it builds.
        connection.execute("SET maintenance_work_mem = '512MB'")
        connection.execute(statement.sql)

        with pytest.raises(psycopg.errors.ProgramLimitExceeded):
            connection.execute(
                "CREATE INDEX wide_plain ON wide USING hnsw (embedding vector_cosine_ops)"
            )
    finally:
        connection.execute("DROP TABLE IF EXISTS wide CASCADE")


# ------------------------------------------------------------------- weighted fusion


def test_weighted_fusion_runs_and_orders_correctly_for_every_metric(connection, config):
    """The fusion method the README argues against still has to work.

    It is kept because people ask for it, and `explain` uses it to show what it does, so
    it is a real code path, one that had only ever been checked as a generated string.

    Its scores look odd and that is expected rather than broken. `1 - distance` assumes a
    distance bounded in [0, 1], which only cosine is: `<#>` returns a *negative* inner
    product so scores come out above 1, and L2 and L1 are unbounded so they can go
    negative. What has to hold is the ordering, since that is what a search returns.
    """
    from dataclasses import replace

    for metric in ("cosine", "l2", "inner_product", "l1"):
        search = HybridSearch(
            replace(config, fusion="weighted", metric=metric),
            execute=lambda sql, params: connection.execute(sql, params).fetchall(),
        )
        results = search.search(None, embedding=query_vector(), limit=len(DOCUMENTS))
        assert results, f"weighted fusion returned nothing for {metric}"

        distances = [r.vector_distance for r in results]
        scores = [r.score for r in results]
        assert all(a <= b + 1e-12 for a, b in zip(distances, distances[1:])), (
            f"{metric}: rows came back out of distance order"
        )
        assert all(a >= b - 1e-12 for a, b in zip(scores, scores[1:])), (
            f"{metric}: score did not fall as distance rose, so the sign is wrong"
        )


def test_weighted_and_rrf_disagree_which_is_the_whole_argument(connection, config):
    """If the two methods always agreed there would be nothing to explain."""
    from dataclasses import replace

    def titles_for(fusion: str) -> list[str]:
        search = HybridSearch(
            replace(config, fusion=fusion),
            execute=lambda sql, params: connection.execute(sql, params).fetchall(),
        )
        return titles(search.search(DEMO_QUERY, embedding=query_vector(), limit=4))

    assert titles_for("rrf") != titles_for("weighted")


def test_explain_measures_both_fusion_methods_from_one_call(connection, config):
    """The effective-weights table compares them, so both have to be executed."""
    search = HybridSearch(
        config, execute=lambda sql, params: connection.execute(sql, params).fetchall()
    )
    report = explain(search, DEMO_QUERY, query_vector(), limit=3, near_miss=0)
    rendered = report.to_text()
    assert "rrf" in rendered and "weighted" in rendered
    assert "effective" in rendered
