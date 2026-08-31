"""Command line interface.

Every subcommand works from a table name alone: the config is discovered by
introspection rather than declared, because someone diagnosing a bad search result
should not first have to describe their own schema to a tool.

Nothing here calls an embedding model. A query embedding is supplied with
``--embedding``, or borrowed from a row already in the table with ``--embedding-from``,
which is usually what you want when the question is "why did this row not come back".
Without either, the command runs the keyword signal alone and says so, rather than
silently reporting half a hybrid search as if it were the whole thing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import replace
from typing import Any, Optional

from .config import Config, Recency, Weights
from .doctor import doctor
from .explain import explain
from .schema import (
    TableNotFound,
    build_migration,
    dbapi_executor,
    introspect,
    suggest_config,
)
from .search import HybridSearch
from .sql import IdentifierError, build_search_sql
from .textquery import parse_query

PROGRAM = "pghybrid"


class CliError(Exception):
    """A problem worth reporting without a traceback."""


def _connect(dsn: Optional[str]) -> Any:
    """Open a connection, with an error that names the fix rather than the exception."""
    dsn = dsn or os.environ.get("PGHYBRID_DSN") or os.environ.get("DATABASE_URL")
    if not dsn:
        raise CliError("no connection string. Pass --dsn, or set PGHYBRID_DSN or DATABASE_URL.")
    try:
        import psycopg
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on the install
        raise CliError(
            "the CLI needs a Postgres driver, which the library itself does not.\n"
            "  pip install 'pghybrid[cli]'"
        ) from exc

    try:
        return psycopg.connect(dsn, autocommit=True, row_factory=psycopg.rows.dict_row)
    except Exception as exc:
        raise CliError(f"could not connect: {exc}") from exc


def _load_embedding(spec: Optional[str]) -> Optional[list[float]]:
    """Read a vector from a JSON literal, a file with @path, or stdin with '-'."""
    if spec is None:
        return None
    if spec == "-":
        payload = sys.stdin.read()
    elif spec.startswith("@"):
        try:
            with open(spec[1:], encoding="utf-8") as handle:
                payload = handle.read()
        except OSError as exc:
            raise CliError(f"could not read {spec[1:]}: {exc}") from exc
    else:
        payload = spec

    try:
        values = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise CliError(
            f"--embedding is not valid JSON ({exc.msg}). Pass a JSON array, "
            "@file.json, or - to read stdin."
        ) from exc

    if not isinstance(values, list) or not values:
        raise CliError("--embedding must be a non-empty JSON array of numbers.")
    try:
        return [float(v) for v in values]
    except (TypeError, ValueError) as exc:
        raise CliError(f"--embedding must contain only numbers: {exc}") from exc


def _embedding_from_row(connection: Any, config: Config, row_id: str) -> list[float]:
    """Borrow an existing row's vector, so a diagnostic needs no embedding model."""
    from .sql import quote_ident

    statement = (
        f"SELECT {quote_ident(config.vector_column)}::text AS v "
        f"FROM {quote_ident(config.table)} "
        f"WHERE {quote_ident(config.id_column)}::text = %s"
    )
    rows = connection.execute(statement, (row_id,)).fetchall()
    if not rows:
        raise CliError(f"no row with {config.id_column} = {row_id!r}")
    raw = rows[0]["v"]
    if raw is None:
        raise CliError(
            f"row {row_id!r} has no {config.vector_column}; pick a row that is embedded."
        )
    return [float(part) for part in raw.strip("[]").split(",")]


def _resolve_config(connection: Any, args: argparse.Namespace) -> Config:
    """Introspect the table, then apply whichever overrides were passed."""
    info = introspect(dbapi_executor(connection), args.table)
    config = suggest_config(info)

    # Collected and applied through dataclasses.replace rather than set one at a time.
    # Assigning to a field of an existing Config skips __post_init__, and __post_init__
    # is where the fields that get interpolated into the statement — language above all
    # — are validated. Setting them directly let a hostile --language through.
    overrides: dict[str, Any] = {}
    for attribute in (
        "text_column",
        "vector_column",
        "id_column",
        "tsvector_column",
        "language",
        "fusion",
        "text_match",
    ):
        value = getattr(args, attribute, None)
        if value is not None:
            overrides[attribute] = value

    if getattr(args, "k", None) is not None:
        overrides["k"] = args.k
    if getattr(args, "candidates", None) is not None:
        overrides["candidate_limit"] = args.candidates
    if getattr(args, "weights", None):
        try:
            vector_weight, text_weight = (float(p) for p in args.weights.split(",", 1))
        except ValueError as exc:
            raise CliError("--weights takes two numbers, e.g. --weights 0.7,0.3") from exc
        overrides["weights"] = Weights(vector=vector_weight, text=text_weight)
    if getattr(args, "recency", None):
        try:
            column, half_life = args.recency.split(",", 1)
            overrides["recency"] = Recency(column=column.strip(), half_life_days=float(half_life))
        except ValueError as exc:
            raise CliError(
                "--recency takes a column and a half-life in days, e.g. --recency created_at,90"
            ) from exc

    # psycopg is the only driver the CLI opens for itself, and it wants %s.
    overrides["paramstyle"] = "pyformat"
    try:
        return replace(config, **overrides)
    except ValueError as exc:
        # Config's own validation messages are written for a person; a traceback on top
        # of one only buries it.
        raise CliError(str(exc)) from exc


def _print_missing_signal(embedding: Optional[list[float]], query: str = "") -> None:
    """Say when only one of the two signals actually ran.

    Both directions are worth reporting. A search that silently drops half of itself
    looks like a relevance problem, and the two causes — no embedding, or a query with
    nothing to search *for* — are fixed in completely different places.
    """
    if embedding is None:
        print(
            "  note: no embedding given, so this is the keyword signal alone.\n"
            "        pass --embedding '[...]' or --embedding-from <id> for hybrid.\n"
        )
    elif query and not parse_query(query).positive:
        print(
            "  note: that query only excludes terms, so there is no keyword signal.\n"
            "        the exclusion still applies; ranking is the vector signal alone.\n"
        )


def command_init(args: argparse.Namespace) -> int:
    connection = _connect(args.dsn)
    info = introspect(dbapi_executor(connection), args.table)
    config = _resolve_config(connection, args)

    print(info.to_text())
    print()

    statements = build_migration(config, info)
    required = [s for s in statements if not s.optional]
    optional = [s for s in statements if s.optional]

    if not required:
        print("Nothing to do: this table is already set up for hybrid search.")
    else:
        print("REQUIRED")
        print("-" * 78)
        for statement in required:
            print(statement.to_text(concurrent=args.concurrent))
            print()

    if optional:
        print("OPTIONAL — alternatives and tuning, not applied by --apply")
        print("-" * 78)
        for statement in optional:
            print(statement.to_text(concurrent=args.concurrent))
            print()

    if args.apply and required:
        if args.concurrent:
            raise CliError(
                "--apply cannot be combined with --concurrent: CREATE INDEX "
                "CONCURRENTLY cannot run inside a transaction. Run the printed "
                "statements yourself."
            )
        runnable = [s for s in required if s.is_executable]
        manual = [s for s in required if not s.is_executable]

        print("applying...")
        for statement in runnable:
            connection.execute(statement.sql)
            print(f"  ok  {statement.sql.splitlines()[0][:70]}")

        if manual:
            # Required work that cannot be written as a statement, because the value is
            # the caller's to supply. Sending it anyway is what this used to do: Postgres
            # accepts a comment as an empty command, so it printed ok and changed nothing.
            print("\nstill to do by hand — these could not be applied:")
            for statement in manual:
                print(f"  !!  {statement.sql.strip().lstrip('- ')}")
                print(f"      {statement.reason}")
            print(
                "\nnot done: the table is not ready for hybrid search until the above is finished."
            )
            return 1

        print("\ndone.")
    elif required:
        print("Re-run with --apply to execute the required statements.")
    return 0


def command_search(args: argparse.Namespace) -> int:
    connection = _connect(args.dsn)
    config = _resolve_config(connection, args)
    embedding = _load_embedding(args.embedding)
    if embedding is None and args.embedding_from is not None:
        embedding = _embedding_from_row(connection, config, args.embedding_from)

    search = HybridSearch(
        config, execute=lambda sql, params: connection.execute(sql, params).fetchall()
    )
    results = search.search(
        args.query, embedding=embedding, limit=args.limit, highlight=args.highlight
    )
    _print_missing_signal(embedding, args.query)

    if not results:
        print("  no results")
        return 0

    if args.json:
        print(
            json.dumps(
                [
                    {
                        "id": r.id,
                        "score": r.score,
                        "vector_rank": r.vector_rank,
                        "text_rank": r.text_rank,
                        "matched_by": r.matched_by,
                        **{k: v for k, v in r.row.items() if k != config.vector_column},
                    }
                    for r in results
                ],
                indent=2,
                default=str,
            )
        )
        return 0

    label = args.label or config.extra_columns[0] if config.extra_columns else None
    for position, result in enumerate(results, 1):
        text = result.get(label) if label else result.get(config.text_column)
        print(f"  {position:>3}  {result.score:.6f}  {str(text)[:88]}")
        if args.highlight and result.highlight:
            print(f"       {result.highlight[:100]}")
    return 0


def command_explain(args: argparse.Namespace) -> int:
    connection = _connect(args.dsn)
    config = _resolve_config(connection, args)
    embedding = _load_embedding(args.embedding)
    if embedding is None and args.embedding_from is not None:
        embedding = _embedding_from_row(connection, config, args.embedding_from)

    search = HybridSearch(
        config, execute=lambda sql, params: connection.execute(sql, params).fetchall()
    )
    _print_missing_signal(embedding, args.query)
    report = explain(
        search,
        args.query,
        embedding,
        limit=args.limit,
        near_miss=args.near_miss,
        label_column=args.label,
        find=args.find,
    )
    print(report.to_text())
    return 0


def command_doctor(args: argparse.Namespace) -> int:
    connection = _connect(args.dsn)
    config = _resolve_config(connection, args)
    report = doctor(dbapi_executor(connection), config, sample=args.sample, k=args.k)
    print(report.to_text())
    return 0


def command_sql(args: argparse.Namespace) -> int:
    """Print the generated statement without touching a database.

    Useful for reading it, for pasting it somewhere else, and for stopping using
    this package entirely — all of which are supported.
    """
    config = Config(
        table=args.table,
        text_column=args.text_column or "content",
        vector_column=args.vector_column or "embedding",
        id_column=args.id_column or "id",
        tsvector_column=args.tsvector_column,
        paramstyle=args.paramstyle,
    )
    sql, params = build_search_sql(
        config,
        embedding=[0.0] if not args.text_only else None,
        text=args.query,
        limit=args.limit,
        highlight=args.highlight,
    )
    print(sql)
    if args.show_params:
        print(f"\n-- {len(params)} parameters")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROGRAM,
        description="Hybrid search on the Postgres you already have.",
    )
    parser.add_argument("--version", action="store_true", help="print the version and exit")
    subparsers = parser.add_subparsers(dest="command")

    def add_connection(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--dsn", help="connection string (or PGHYBRID_DSN / DATABASE_URL)")
        sub.add_argument("--table", required=True, help="table to search")
        sub.add_argument("--text-column")
        sub.add_argument("--vector-column")
        sub.add_argument("--id-column")
        sub.add_argument("--tsvector-column")
        sub.add_argument("--language")

    def add_query(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("query", nargs="?", help="the search text")
        sub.add_argument(
            "--embedding",
            help="query vector as a JSON array, @file.json, or - for stdin",
        )
        sub.add_argument(
            "--embedding-from",
            metavar="ID",
            help="borrow the vector of an existing row, so no model is needed",
        )
        sub.add_argument("--fusion", choices=["rrf", "weighted"])
        sub.add_argument("--text-match", choices=["any", "all"], dest="text_match")
        sub.add_argument("--weights", help="vector,text — e.g. 0.7,0.3")
        sub.add_argument("--k", type=int, help="the RRF constant (default 60)")
        sub.add_argument("--candidates", type=int, help="candidates per signal")
        sub.add_argument("--recency", help="column,half_life_days — e.g. created_at,90")
        sub.add_argument("--label", help="column to show in the output")

    init = subparsers.add_parser("init", help="inspect a table and write the migration")
    add_connection(init)
    init.add_argument("--apply", action="store_true", help="execute the required statements")
    init.add_argument("--concurrent", action="store_true", help="print CONCURRENTLY forms")
    init.set_defaults(func=command_init)

    search = subparsers.add_parser("search", help="run a hybrid search")
    add_connection(search)
    add_query(search)
    search.add_argument("--limit", type=int, default=10)
    search.add_argument("--highlight", action="store_true")
    search.add_argument("--json", action="store_true")
    search.set_defaults(func=command_search)

    explain_parser = subparsers.add_parser(
        "explain", help="decompose one result set signal by signal"
    )
    add_connection(explain_parser)
    add_query(explain_parser)
    explain_parser.add_argument("--limit", type=int, default=10)
    explain_parser.add_argument("--near-miss", type=int, default=10, dest="near_miss")
    explain_parser.add_argument("--find", metavar="TEXT", help="text you expected to be retrieved")
    explain_parser.set_defaults(func=command_explain)

    doctor_parser = subparsers.add_parser("doctor", help="measure recall and grade the indexes")
    add_connection(doctor_parser)
    doctor_parser.add_argument("--sample", type=int, default=50)
    doctor_parser.add_argument("--k", type=int, default=10)
    doctor_parser.set_defaults(func=command_doctor)

    sql_parser = subparsers.add_parser("sql", help="print the generated SQL without a database")
    sql_parser.add_argument("--table", required=True)
    sql_parser.add_argument("query", nargs="?", default="example query")
    sql_parser.add_argument("--text-column")
    sql_parser.add_argument("--vector-column")
    sql_parser.add_argument("--id-column")
    sql_parser.add_argument("--tsvector-column")
    sql_parser.add_argument("--paramstyle", choices=["numeric", "pyformat"], default="numeric")
    sql_parser.add_argument("--limit", type=int, default=10)
    sql_parser.add_argument("--highlight", action="store_true")
    sql_parser.add_argument("--text-only", action="store_true")
    sql_parser.add_argument("--show-params", action="store_true")
    sql_parser.set_defaults(func=command_sql)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if getattr(args, "version", False):
        from . import __version__

        print(f"{PROGRAM} {__version__}")
        return 0
    if not getattr(args, "func", None):
        parser.print_help()
        return 1

    try:
        return int(args.func(args) or 0)
    except (CliError, TableNotFound, IdentifierError) as exc:
        # These already carry a message written for a person — a traceback on top of
        # "relation 'nope' does not exist" only buries it.
        print(f"{PROGRAM}: {exc}", file=sys.stderr)
        return 2
    except BrokenPipeError:  # pragma: no cover - piping into head and similar
        return 0
    except KeyboardInterrupt:  # pragma: no cover
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
