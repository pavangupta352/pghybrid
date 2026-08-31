"""The command line, exercised as a person would use it.

The CLI is the surface most people touch first and the only one that takes strings from
outside the program, and it had never been tested. Both bugs found in it so far came from
that: a hostile ``--language`` reached the statement because overrides were applied with
``setattr`` and skipped validation, and a missing table produced a traceback instead of a
sentence.

``main`` is called directly rather than through a subprocess. It returns the exit code, so
the assertions can be about behaviour rather than about text scraped from a shell.
"""

from __future__ import annotations

import json
import os

import pytest

from pghybrid.cli import main

psycopg = pytest.importorskip("psycopg")

DSN = os.environ.get("PGHYBRID_TEST_DSN", "postgresql://postgres:pghybrid@localhost:55432/pghybrid")
TABLE = "chunks"
QUERY = "renewal notice period"
ANSWER = "Termination for convenience"


def reachable() -> bool:
    try:
        with psycopg.connect(DSN, connect_timeout=3):
            return True
    except Exception:
        return False


needs_database = pytest.mark.skipif(not reachable(), reason=f"no Postgres at {DSN}")


def with_database(func):
    """skipif plus a marker the seeding fixture can detect."""
    return pytest.mark.needs_database(needs_database(func))


@pytest.fixture(autouse=True)
def dsn_in_environment(monkeypatch):
    """The CLI reads PGHYBRID_DSN, so every test gets it unless it is testing its absence."""
    monkeypatch.setenv("PGHYBRID_DSN", DSN)


@pytest.fixture(autouse=True)
def _seeded(request):
    """Every test that needs a server also needs the corpus, and must not assume another
    module put it there."""
    if any(mark.name == "needs_database" for mark in request.node.iter_markers()):
        request.getfixturevalue("demo_table")


def run(*argv: str) -> int:
    return main(list(argv))


# ------------------------------------------------------------------ no database needed


def test_version(capsys):
    assert run("--version") == 0
    assert "pghybrid" in capsys.readouterr().out


def test_no_subcommand_prints_help_and_fails(capsys):
    assert run() == 1
    assert "usage" in capsys.readouterr().out.lower()


def test_sql_needs_no_database(capsys, monkeypatch):
    """`pghybrid sql` exists so the statement can be read without a server anywhere."""
    monkeypatch.delenv("PGHYBRID_DSN", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert run("sql", "--table", "chunks", "--tsvector-column", "fts", QUERY) == 0
    out = capsys.readouterr().out
    assert "WITH vector_candidates" in out and "FULL OUTER JOIN" in out


def test_sql_paramstyle_switches_the_placeholders(capsys):
    run("sql", "--table", "c", "--paramstyle", "pyformat", "hi")
    assert "%s" in capsys.readouterr().out
    run("sql", "--table", "c", "--paramstyle", "numeric", "hi")
    assert "$1" in capsys.readouterr().out


def test_sql_carries_the_language_through(capsys):
    """Someone printing the statement for a French config must get their statement,
    not the English default with the flag silently dropped."""
    assert run("sql", "--table", "c", "--language", "french", "hi") == 0
    assert "'french'" in capsys.readouterr().out


def test_a_hostile_table_name_is_refused_without_a_traceback(capsys):
    assert run("sql", "--table", 'bad"; DROP TABLE users; --') == 2
    assert "not a valid Postgres identifier" in capsys.readouterr().err


def test_missing_connection_string_says_which_variables_it_reads(capsys, monkeypatch):
    monkeypatch.delenv("PGHYBRID_DSN", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert run("search", "--table", TABLE, "hi") == 2
    error = capsys.readouterr().err
    assert "PGHYBRID_DSN" in error and "DATABASE_URL" in error


# --------------------------------------------------------------------- against a server


@with_database
def test_search_finds_the_planted_answer(capsys):
    assert (
        run(
            "search",
            QUERY,
            "--table",
            TABLE,
            "--embedding-from",
            "1",
            "--limit",
            "3",
            "--label",
            "title",
        )
        == 0
    )
    assert ANSWER in capsys.readouterr().out


@with_database
def test_search_without_an_embedding_says_it_is_running_one_signal(capsys):
    """Silently returning half a hybrid search would be the worst possible default."""
    assert run("search", QUERY, "--table", TABLE, "--limit", "2", "--label", "title") == 0
    out = capsys.readouterr().out
    assert "keyword signal alone" in out


@with_database
def test_search_json_is_machine_readable(capsys):
    assert (
        run("search", QUERY, "--table", TABLE, "--embedding-from", "1", "--limit", "2", "--json")
        == 0
    )
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == 2
    assert {"id", "score", "vector_rank", "text_rank", "matched_by"} <= set(rows[0])


@with_database
def test_explain_shows_the_near_miss_band_and_find(capsys):
    assert (
        run(
            "explain",
            QUERY,
            "--table",
            TABLE,
            "--embedding-from",
            "1",
            "--limit",
            "2",
            "--near-miss",
            "3",
            "--label",
            "title",
            "--find",
            "sixty days written notice",
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "near miss" in out
    assert "find" in out and "sixty days written notice" in out


@with_database
def test_explain_find_reports_text_that_is_not_there(capsys):
    assert (
        run(
            "explain",
            QUERY,
            "--table",
            TABLE,
            "--embedding-from",
            "1",
            "--limit",
            "2",
            "--label",
            "title",
            "--find",
            "force majeure pandemic clause",
        )
        == 0
    )
    assert "no row in" in capsys.readouterr().out


@with_database
def test_init_prints_statements_and_does_not_apply_them(capsys, connection_for_cli):
    before = connection_for_cli.execute(
        "SELECT count(*) AS n FROM pg_indexes WHERE tablename = %s", (TABLE,)
    ).fetchone()["n"]
    assert run("init", "--table", TABLE) == 0
    after = connection_for_cli.execute(
        "SELECT count(*) AS n FROM pg_indexes WHERE tablename = %s", (TABLE,)
    ).fetchone()["n"]
    assert before == after, "init without --apply must not change anything"
    assert "table" in capsys.readouterr().out


@with_database
def test_apply_does_not_claim_to_have_run_a_comment(capsys, connection_for_cli):
    """Required work that cannot be written as a statement must not be reported as done.

    A bare `vector` column has to be given the dimension the model produces, and nothing
    here knows what that is, so the migration carries it as a comment with a placeholder.
    --apply sent it to the server, which accepts a comment as an empty command, printed
    "ok" for it and finished with "done." The column was untouched and the exit code was
    0, so nothing a caller could check would have told them the table was not ready.
    """
    connection_for_cli.execute("DROP TABLE IF EXISTS cli_bare_vector")
    connection_for_cli.execute(
        "CREATE TABLE cli_bare_vector (id bigserial PRIMARY KEY, content text NOT NULL,"
        " embedding vector)"
    )
    connection_for_cli.execute(
        "INSERT INTO cli_bare_vector (content, embedding) VALUES ('renewal notice', '[1,0]')"
    )
    try:
        assert run("init", "--table", "cli_bare_vector", "--apply") == 1
        out = capsys.readouterr().out
        assert "still to do by hand" in out
        assert "not done" in out
        # The work it *could* do is still done, so re-running finishes the job.
        assert "ok  ALTER TABLE" in out and "ADD COLUMN" in out

        # And the comment was not reported as applied.
        applied = [line for line in out.splitlines() if line.strip().startswith("ok")]
        assert not any("--" in line for line in applied), applied

        # Nothing changed the column, which is the point.
        kind = connection_for_cli.execute(
            "SELECT format_type(atttypid, atttypmod) AS t FROM pg_attribute "
            "WHERE attrelid = 'cli_bare_vector'::regclass AND attname = 'embedding'"
        ).fetchone()["t"]
        assert kind == "vector", kind

        # Once the dimension is supplied the same command succeeds.
        connection_for_cli.execute(
            "ALTER TABLE cli_bare_vector ALTER COLUMN embedding TYPE vector(2)"
        )
        capsys.readouterr()
        assert run("init", "--table", "cli_bare_vector", "--apply") == 0
        assert "still to do by hand" not in capsys.readouterr().out
    finally:
        connection_for_cli.execute("DROP TABLE IF EXISTS cli_bare_vector")


@with_database
def test_doctor_is_read_only(capsys, connection_for_cli):
    before = connection_for_cli.execute(
        "SELECT count(*) AS n FROM pg_indexes WHERE tablename = %s", (TABLE,)
    ).fetchone()["n"]
    assert run("doctor", "--table", TABLE, "--sample", "3", "--k", "2") == 0
    after = connection_for_cli.execute(
        "SELECT count(*) AS n FROM pg_indexes WHERE tablename = %s", (TABLE,)
    ).fetchone()["n"]
    assert before == after
    assert "read-only" in capsys.readouterr().out


@with_database
def test_the_librarys_own_refusals_are_messages_not_tracebacks(capsys):
    """The library raises ValueError with a sentence written for a person, and the CLI
    was printing that sentence underneath a traceback, which is exactly where a person
    stops reading. A query of only exclusions with no embedding is the cleanest case:
    the message tells you the two ways to fix it."""
    assert run("search", "--table", TABLE, "--", "-pricing") == 2
    error = capsys.readouterr().err
    assert "only excludes terms" in error
    assert "Traceback" not in error


@with_database
def test_a_non_finite_embedding_is_refused_with_its_index(capsys):
    """json.loads accepts NaN and Infinity, float() keeps them, and pgvector rejects
    them server-side with an error naming neither the argument nor the position. NaN is
    the worst case because if it ever got through, every comparison with it is false and
    the ordering is garbage rather than an error."""
    assert run("search", "x", "--table", TABLE, "--embedding", "[0.1, NaN]") == 2
    error = capsys.readouterr().err
    assert "finite" in error and "index 1" in error
    assert "Traceback" not in error


@with_database
def test_a_server_side_refusal_is_a_message_not_a_traceback(capsys):
    """Wrong dimensions only fail once the statement reaches the server, so this is the
    path the ValueError catch cannot cover and the database-error catch must."""
    assert run("search", "x", "--table", TABLE, "--embedding", "[1, 2]") == 2
    error = capsys.readouterr().err
    assert "database error" in error and "dimensions" in error
    assert "Traceback" not in error


@with_database
def test_label_selects_the_column_it_names(capsys, connection_for_cli):
    """--label aux printed None for every row when introspection had not already chosen
    aux as an extra column, which reads as broken data rather than as an unselected
    column. The flag now puts the column into the selection."""
    connection_for_cli.execute("DROP TABLE IF EXISTS cli_label_probe")
    connection_for_cli.execute(
        "CREATE TABLE cli_label_probe (id bigserial PRIMARY KEY, content text NOT NULL,"
        " aux text, embedding vector(8))"
    )
    try:
        connection_for_cli.execute(
            "INSERT INTO cli_label_probe (content, aux, embedding) "
            "VALUES ('renewal notice clause', 'AUXVALUE', %s::vector)",
            ("[" + ",".join(["0.1"] * 8) + "]",),
        )
        assert run("search", "renewal", "--table", "cli_label_probe", "--label", "aux") == 0
        out = capsys.readouterr().out
        assert "AUXVALUE" in out
        assert "None" not in out
    finally:
        connection_for_cli.execute("DROP TABLE IF EXISTS cli_label_probe")


@with_database
def test_a_label_that_names_no_column_says_which_columns_exist(capsys):
    assert run("search", "x", "--table", TABLE, "--label", "nope") == 2
    error = capsys.readouterr().err
    assert "no such column" in error and "title" in error
    assert "Traceback" not in error


# ------------------------------------------------------------------------- error paths


@with_database
@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (("search", "x", "--table", "no_such_table"), "does not exist"),
        (("search", "x", "--table", TABLE, "--embedding", "not json"), "valid JSON"),
        (("search", "x", "--table", TABLE, "--embedding", "[]"), "non-empty"),
        (("search", "x", "--table", TABLE, "--embedding", '["a"]'), "only numbers"),
        (("search", "x", "--table", TABLE, "--embedding-from", "999999"), "no row with"),
        (("search", "x", "--table", TABLE, "--weights", "nonsense"), "two numbers"),
        (("search", "x", "--table", TABLE, "--recency", "nonsense"), "half-life"),
        (
            ("search", "x", "--table", TABLE, "--language", "english'); DROP TABLE chunks; --"),
            "text search configuration",
        ),
    ],
)
def test_bad_input_is_a_message_not_a_traceback(capsys, argv, expected):
    """Every one of these used to be, or could become, a stack trace.

    The language case is the one that matters most: it is the injection that reached the
    statement because the CLI applied overrides with setattr and so skipped validation.
    """
    assert run(*argv) == 2
    error = capsys.readouterr().err
    assert expected in error
    assert "Traceback" not in error


@with_database
def test_the_table_survives_a_hostile_language(connection_for_cli):
    run("search", "x", "--table", TABLE, "--language", "english'); DROP TABLE chunks; --")
    remaining = connection_for_cli.execute(f"SELECT count(*) AS n FROM {TABLE}").fetchone()["n"]
    assert remaining > 0, "the injection ran"
