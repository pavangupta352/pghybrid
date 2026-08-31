"""Schema introspection and migration generation.

The purpose of this module is that ``pghybrid init`` needs no arguments. It reads the
table you already have, picks the columns that make sense, and prints DDL with the
reasoning attached, so each statement can be reviewed rather than trusted.

Every statement here is a read of the system catalogs, and none of them take bind
parameters: identifiers are validated by :func:`~pghybrid.sql.quote_ident` before they
are embedded, so the same code runs against psycopg, asyncpg or a raw psql pipe
without knowing the driver's placeholder style.

The ``execute`` argument is a callable the caller supplies::

    execute(sql: str, params: Sequence[Any] | None = None) -> Sequence[Sequence[Any]]

It returns rows for statements that produce them and an empty sequence for statements
that do not. :func:`dbapi_executor` builds one from any DB-API connection.
"""

from __future__ import annotations

import inspect
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Optional

from .config import METRICS, Config, Metric, VectorType

# Imported rather than re-derived: the stored tsvector column and the expression the
# query builder computes inline must be character-identical, or switching
# Config.tsvector_column on and off would silently change which rows match.
from .sql import _tsvector_expr as _inline_tsvector_expr
from .sql import quote_ident

Row = Sequence[Any]
Execute = Callable[..., Sequence[Row]]

#: Postgres truncates identifiers at 63 bytes, silently. Generated index names are cut
#: to fit so that the name we print is the name that ends up in the catalog.
MAX_IDENTIFIER_LENGTH = 63

#: pgvector's own limits. ``vector`` can be indexed up to 2000 dimensions and
#: ``halfvec`` up to 4000; a column wider than its limit can still be stored and
#: searched, just never with an index.
MAX_INDEXED_DIMENSIONS = {"vector": 2000, "halfvec": 4000, "sparsevec": 1000}

#: Below this heap size an exact ``count(*)`` is cheap enough to prefer over the
#: planner's estimate. 32 MB of heap counts in low single-digit milliseconds.
EXACT_COUNT_MAX_BYTES = 32 * 1024 * 1024

#: pgvector's defaults for HNSW. Named here so the migration can say "these are the
#: defaults" instead of presenting them as tuning that someone did on your behalf.
HNSW_DEFAULT_M = 16
HNSW_DEFAULT_EF_CONSTRUCTION = 64

#: The row count at which pgvector's own guidance for ``ivfflat`` changes shape.
IVFFLAT_SQRT_THRESHOLD = 1_000_000

#: Column names that are usually the body of a document, best first.
_TEXT_NAME_PREFERENCE = (
    "content",
    "chunk",
    "chunk_text",
    "body",
    "text",
    "document",
    "passage",
    "page_content",
    "description",
    "summary",
    "title",
    "name",
)
#: Column names that are usually an embedding, best first.
_VECTOR_NAME_PREFERENCE = ("embedding", "embeddings", "vector", "vec", "emb")
#: Columns a multi-tenant application almost always filters on.
_FILTER_NAME_HINTS = (
    "tenant_id",
    "org_id",
    "organization_id",
    "workspace_id",
    "account_id",
    "customer_id",
    "user_id",
    "project_id",
    "collection_id",
    "namespace",
    "source",
    "language",
    "lang",
    "status",
    "kind",
    "type",
    "category",
    "is_deleted",
    "deleted_at",
    "archived",
)

_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_$]*")
_IDENT_SAFE_RE = re.compile(r"[^A-Za-z0-9_]+")

#: Operator classes that carry no information because they are the type's default.
#: Printing them turns the index list into noise and hides the ones that matter.
_DEFAULT_OPCLASSES = frozenset(
    {
        "int2_ops",
        "int4_ops",
        "int8_ops",
        "text_ops",
        "varchar_ops",
        "bpchar_ops",
        "uuid_ops",
        "timestamptz_ops",
        "timestamp_ops",
        "date_ops",
        "bool_ops",
        "numeric_ops",
        "float4_ops",
        "float8_ops",
        "tsvector_ops",
        "jsonb_ops",
        "array_ops",
        "oid_ops",
        "name_ops",
        "citext_ops",
    }
)

# Opclass name -> metric, so an existing index can tell us which metric the table was
# built for. Recommending cosine for a table indexed with vector_l2_ops would produce a
# query that silently cannot use the index that is already there.
_OPS_TO_METRIC: dict[str, Metric] = {}
for _metric in dict.fromkeys(METRICS.values()):
    _OPS_TO_METRIC[_metric.ops_vector] = _metric
    _OPS_TO_METRIC[_metric.ops_halfvec] = _metric


class TableNotFound(ValueError):
    """Raised when the table named in a Config does not exist on this server."""


# --------------------------------------------------------------------------- values


@dataclass(frozen=True)
class ColumnInfo:
    """One column, as the catalog describes it."""

    name: str
    data_type: str
    type_name: str
    position: int
    not_null: bool
    generated: bool
    default: Optional[str] = None
    #: Declared dimensions for a vector-family column. None means the column was
    #: created as bare ``vector``, which stores fine but can never be indexed.
    dimensions: Optional[int] = None

    @property
    def is_vector(self) -> bool:
        return self.type_name in ("vector", "halfvec", "sparsevec")

    @property
    def is_text(self) -> bool:
        return self.type_name in ("text", "varchar", "bpchar", "citext")

    @property
    def is_tsvector(self) -> bool:
        return self.type_name == "tsvector"


@dataclass(frozen=True)
class IndexInfo:
    """One index, including the parts that decide whether a query can use it."""

    name: str
    method: str
    keys: list[str]
    opclasses: list[str]
    definition: str
    size_bytes: int
    valid: bool
    primary: bool
    unique: bool
    options: dict[str, str] = field(default_factory=dict)
    predicate: Optional[str] = None

    @property
    def is_vector(self) -> bool:
        return self.method in ("hnsw", "ivfflat")

    @property
    def metric(self) -> Optional[Metric]:
        """The metric this index can answer, read from its operator class."""
        for opclass in self.opclasses:
            metric = _OPS_TO_METRIC.get(opclass)
            if metric is not None:
                return metric
        return None

    @property
    def lists(self) -> Optional[int]:
        value = self.options.get("lists")
        return int(value) if value is not None and value.isdigit() else None

    def covers(self, column: str) -> bool:
        """Whether this index is built on ``column``, directly or in an expression."""
        return any(column in _WORD_RE.findall(key) for key in self.keys)

    @property
    def key_text(self) -> str:
        """The key columns with their operator classes, the way psql prints them.

        The operator class is shown for the vector methods because it is the field that
        decides whether a query can use the index at all, and it is invisible otherwise.
        """
        parts = []
        for position, key in enumerate(self.keys):
            opclass = self.opclasses[position] if position < len(self.opclasses) else ""
            if opclass and (self.is_vector or opclass not in _DEFAULT_OPCLASSES):
                parts.append(f"{key} {opclass}")
            else:
                parts.append(key)
        return ", ".join(parts)

    @property
    def is_expression(self) -> bool:
        return any(_WORD_RE.fullmatch(key) is None for key in self.keys)


@dataclass(frozen=True)
class TableInfo:
    """Everything the migration generator and the doctor need to know about a table."""

    table: str
    schema: str
    name: str
    oid: int
    kind: str
    columns: list[ColumnInfo]
    indexes: list[IndexInfo]
    row_count: int
    #: Where ``row_count`` came from: "count(*)", "pg_class.reltuples" or "planner".
    row_count_source: str
    table_size_bytes: int
    indexes_size_bytes: int
    primary_key: list[str]
    #: pg_class.reltuples verbatim. -1 means the table has never been analysed, which
    #: is a different and more actionable problem than the count being stale.
    reltuples: float = -1.0
    pgvector_version: Optional[str] = None
    pgvector_available_version: Optional[str] = None
    server_version_num: int = 0
    server_version: str = ""

    @property
    def row_count_is_estimate(self) -> bool:
        return self.row_count_source != "count(*)"

    @property
    def analyzed(self) -> bool:
        """Whether the planner has any statistics at all for this table."""
        return self.reltuples >= 0

    @property
    def qualified(self) -> str:
        return f"{self.schema}.{self.name}"

    @property
    def is_partitioned(self) -> bool:
        return self.kind == "p"

    def column(self, name: Optional[str]) -> Optional[ColumnInfo]:
        for column in self.columns:
            if column.name == name:
                return column
        return None

    @property
    def vector_columns(self) -> list[ColumnInfo]:
        return [c for c in self.columns if c.is_vector]

    @property
    def text_columns(self) -> list[ColumnInfo]:
        return [c for c in self.columns if c.is_text]

    @property
    def tsvector_columns(self) -> list[ColumnInfo]:
        return [c for c in self.columns if c.is_tsvector]

    def indexes_on(self, column: str) -> list[IndexInfo]:
        return [i for i in self.indexes if i.covers(column)]

    def vector_index_for(self, column: str) -> Optional[IndexInfo]:
        for index in self.indexes:
            if index.is_vector and index.covers(column):
                return index
        return None

    def text_index_for(self, column: Optional[str], expression: str) -> Optional[IndexInfo]:
        """A GIN index on the stored tsvector column, or on the equivalent expression."""
        normalized = _normalize_sql(expression)
        for index in self.indexes:
            if index.method not in ("gin", "rum"):
                continue
            if column and index.covers(column):
                return index
            if any(_normalize_sql(key) == normalized for key in index.keys):
                return index
        return None

    @property
    def pgvector_version_tuple(self) -> tuple[int, ...]:
        return parse_version(self.pgvector_version)

    @property
    def supports_iterative_scan(self) -> bool:
        """pgvector 0.8.0 added iterative index scans, the fix for filtered recall."""
        return self.pgvector_version_tuple >= (0, 8)

    def to_text(self) -> str:
        """Render the inventory, the same block the doctor report opens with."""
        lines = [f"table            {self.qualified}"]
        count = f"{self.row_count:,}"
        if self.row_count_is_estimate:
            count += f" (estimated from {self.row_count_source})"
        else:
            count += " (exact)"
        lines.append(f"rows             {count}")
        lines.append(
            f"size             {human_bytes(self.table_size_bytes)} heap, "
            f"{human_bytes(self.indexes_size_bytes)} indexes"
        )
        for column in self.vector_columns:
            lines.append(f"vector column    {column.name}  {column.data_type}")
        for column in self.tsvector_columns:
            suffix = " (generated)" if column.generated else " (maintained by you)"
            lines.append(f"tsvector column  {column.name}{suffix}")
        if self.indexes:
            lines.append("indexes")
            width = max(len(i.name) for i in self.indexes)
            for index in self.indexes:
                flag = "" if index.valid else "  INVALID"
                lines.append(
                    f"  {index.name.ljust(width)}  {index.method:<8}"
                    f"({index.key_text})  {human_bytes(index.size_bytes)}{flag}"
                )
        else:
            lines.append("indexes          none")
        lines.append(
            f"pgvector         {self.pgvector_version or 'not installed'}"
            f"    server {self.server_version or 'unknown'}"
        )
        return "\n".join(lines)


@dataclass(frozen=True)
class Statement:
    """One DDL statement plus the reason it exists.

    The reason is not decoration. Someone has to defend this statement to whoever owns
    the database, and "the tool said so" is not a defence.
    """

    sql: str
    reason: str
    kind: str = "index"
    #: Optional statements are alternatives and tuning, not part of making search work.
    optional: bool = False
    #: The same statement built with CONCURRENTLY, when that form exists.
    concurrent_sql: Optional[str] = None
    notes: list[str] = field(default_factory=list)

    def to_text(self, *, concurrent: bool = False) -> str:
        """The statement as it would appear in a migration file, comments included."""
        lines = [f"-- {self.reason}"]
        for note in self.notes:
            lines.append(f"--   {note}")
        sql = self.concurrent_sql if concurrent and self.concurrent_sql else self.sql
        if concurrent and self.concurrent_sql:
            lines.append(
                "--   CONCURRENTLY cannot run inside a transaction block: send this "
                "statement on its own, with autocommit on."
            )
        lines.append(sql if sql.rstrip().endswith(";") else sql + ";")
        return "\n".join(lines)


# ------------------------------------------------------------------------ execution


def _positional(row: Any) -> Any:
    """Return a row as a positional sequence, whatever shape the driver produced.

    Introspection unpacks its rows by position. A driver configured to return mappings
    — ``psycopg.rows.dict_row``, ``RealDictCursor``, SQLAlchemy's ``RowMapping`` — would
    otherwise unpack into column *names*, so the queried relkind would come back as the
    string "relkind" and the failure would be reported as an unsupported table type.
    Mapping order matches the SELECT list in every driver that offers this, which is
    what makes the conversion safe.
    """
    if isinstance(row, Mapping):
        return tuple(row.values())
    return row


class Executor:
    """Adapts a caller-supplied ``execute`` callable to the shape this module wants.

    A callable that takes only the SQL string is accepted as well as one that takes
    SQL and parameters, because half the people who wire this up write the one-argument
    version first and the resulting TypeError is a confusing way to learn the contract.
    """

    def __init__(self, execute: Execute) -> None:
        self._execute: Execute
        self._accepts_params: bool
        if isinstance(execute, Executor):
            self._execute = execute._execute
            self._accepts_params = execute._accepts_params
            return
        if not callable(execute):
            raise TypeError(
                "execute must be a callable of (sql, params) -> rows; got "
                f"{type(execute).__name__}. pghybrid.schema.dbapi_executor(conn) "
                "builds one from a DB-API connection."
            )
        self._execute = execute
        self._accepts_params = _accepts_parameters(execute)

    def __call__(self, sql: str, params: Optional[Sequence[Any]] = None) -> list[Row]:
        if params is None and not self._accepts_params:
            rows = self._execute(sql)
        elif params is None:
            rows = self._execute(sql, None)
        else:
            if not self._accepts_params:
                raise TypeError(
                    "execute must accept (sql, params); this query is parameterised "
                    "and its values cannot be inlined safely."
                )
            rows = self._execute(sql, params)
        return [_positional(row) for row in (rows or [])]

    def scalar(self, sql: str, params: Optional[Sequence[Any]] = None) -> Any:
        rows = self(sql, params)
        if not rows:
            return None
        row = rows[0]
        return row[0] if isinstance(row, Sequence) and not isinstance(row, str) else row

    def run(self, sql: str) -> None:
        """Execute a statement whose result is irrelevant (SET, BEGIN, ROLLBACK)."""
        self(sql)


def _accepts_parameters(execute: Execute) -> bool:
    """Whether ``execute`` takes a second positional argument for bind parameters."""
    try:
        signature = inspect.signature(execute)
    except (TypeError, ValueError):
        # Builtins and C callables have no introspectable signature. Assume the full
        # contract; a genuine mismatch surfaces as a TypeError from the driver.
        return True
    positional = 0
    for parameter in signature.parameters.values():
        if parameter.kind is parameter.VAR_POSITIONAL:
            return True
        if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD):
            positional += 1
    return positional >= 2


def dbapi_executor(connection: Any) -> Execute:
    """Build an ``execute`` callable from any DB-API connection.

    Deliberately duck-typed: nothing here imports psycopg, so the package keeps its
    promise of zero runtime dependencies while still being one line to use.
    """

    def open_cursor() -> Any:
        """A cursor that yields positional rows, whatever the connection's default is.

        Introspection unpacks its rows by position, and the catalog queries below
        legitimately select two columns with the same underlying name. A connection
        configured for mapping rows — ``psycopg.rows.dict_row`` and its equivalents are
        common — would collapse that pair into one key and hand back a row one column
        short. Asking for positional rows here is cheaper than aliasing every column in
        every catalog query and hoping nobody adds a colliding one later.
        """
        try:
            from psycopg.rows import tuple_row
        except ModuleNotFoundError:
            return connection.cursor()
        try:
            return connection.cursor(row_factory=tuple_row)
        except TypeError:
            # psycopg2 and other DB-API drivers take no row_factory; their default
            # cursor already returns tuples.
            return connection.cursor()

    def execute(sql: str, params: Optional[Sequence[Any]] = None) -> list[Row]:
        with open_cursor() as cursor:
            # An empty sequence is normalised to None because psycopg treats any
            # non-None params as a request to interpolate, and a statement containing
            # a literal percent sign then fails for no visible reason.
            cursor.execute(sql, list(params) if params else None)
            if cursor.description is None:
                return []
            return list(cursor.fetchall())

    return execute


# ------------------------------------------------------------------- introspection


def introspect(execute: Execute, table: str) -> TableInfo:
    """Read everything about ``table`` that a search configuration depends on."""
    run = Executor(execute)
    quoted = quote_ident(table)

    row = run(
        "SELECT c.oid::bigint, n.nspname, c.relname, c.relkind, c.reltuples::double precision, "
        "       pg_catalog.pg_table_size(c.oid)::bigint, "
        "       pg_catalog.pg_indexes_size(c.oid)::bigint "
        "FROM pg_catalog.pg_class c "
        "JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace "
        f"WHERE c.oid = to_regclass({_literal(quoted)})"
    )
    if not row:
        raise TableNotFound(_missing_table_message(run, table))
    oid, schema, name, kind, reltuples, table_size, indexes_size = row[0]

    if kind not in ("r", "p", "m", "f"):
        raise TableNotFound(
            f"{table!r} exists but is a {_RELKIND_NAMES.get(kind, kind)}, which cannot "
            "be indexed for search. Point pghybrid at the underlying table."
        )

    columns = _read_columns(run, oid)
    indexes = _read_indexes(run, oid)
    primary_key = next((i.keys for i in indexes if i.primary), [])
    reltuples = float(reltuples if reltuples is not None else -1.0)
    rows, source = _count_rows(run, quoted, reltuples, int(table_size))
    version_row = run(
        "SELECT e.extversion, a.default_version, "
        "       current_setting('server_version_num')::int, current_setting('server_version') "
        "FROM pg_catalog.pg_available_extensions a "
        "LEFT JOIN pg_catalog.pg_extension e ON e.extname = a.name "
        "WHERE a.name = 'vector'"
    )
    installed = available = None
    server_num, server_text = 0, ""
    if version_row:
        installed, available, server_num, server_text = version_row[0]
    else:
        settings = run(
            "SELECT current_setting('server_version_num')::int, current_setting('server_version')"
        )
        if settings:
            server_num, server_text = settings[0]

    return TableInfo(
        table=table,
        schema=schema,
        name=name,
        oid=int(oid),
        kind=kind,
        columns=columns,
        indexes=indexes,
        row_count=rows,
        row_count_source=source,
        table_size_bytes=int(table_size or 0),
        indexes_size_bytes=int(indexes_size or 0),
        primary_key=list(primary_key),
        reltuples=reltuples,
        pgvector_version=installed,
        pgvector_available_version=available,
        server_version_num=int(server_num or 0),
        server_version=str(server_text or ""),
    )


_RELKIND_NAMES = {
    "v": "view",
    "i": "index",
    "S": "sequence",
    "c": "composite type",
    "t": "TOAST table",
    "I": "partitioned index",
}


def _missing_table_message(run: Executor, table: str) -> str:
    """A not-found error that names the tables the user probably meant."""
    target = table.split(".")[-1].lower()
    candidates = [
        str(row[0])
        for row in run(
            "SELECT n.nspname || '.' || c.relname "
            "FROM pg_catalog.pg_class c "
            "JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace "
            "WHERE c.relkind IN ('r', 'p', 'm') "
            "  AND n.nspname NOT IN ('pg_catalog', 'information_schema') "
            "ORDER BY c.relname LIMIT 200"
        )
    ]
    near = [c for c in candidates if target in c.lower() or c.split(".")[-1][:4] == target[:4]]
    message = f"relation {table!r} does not exist, or is not visible to this role."
    listed = near or candidates
    if listed:
        message += " Tables here: " + ", ".join(listed[:8])
        if len(listed) > 8:
            message += f", and {len(listed) - 8} more"
    return message


def _read_columns(run: Executor, oid: int) -> list[ColumnInfo]:
    rows = run(
        "SELECT a.attname, pg_catalog.format_type(a.atttypid, a.atttypmod), t.typname, "
        "       a.attnum, a.attnotnull, a.attgenerated <> '', "
        "       pg_catalog.pg_get_expr(d.adbin, d.adrelid), a.atttypmod "
        "FROM pg_catalog.pg_attribute a "
        "JOIN pg_catalog.pg_type t ON t.oid = a.atttypid "
        "LEFT JOIN pg_catalog.pg_attrdef d ON d.adrelid = a.attrelid AND d.adnum = a.attnum "
        f"WHERE a.attrelid = {oid} AND a.attnum > 0 AND NOT a.attisdropped "
        "ORDER BY a.attnum"
    )
    columns = []
    for name, data_type, type_name, position, not_null, generated, default, typmod in rows:
        dimensions = None
        if type_name in MAX_INDEXED_DIMENSIONS and typmod is not None and int(typmod) > 0:
            # Unlike varchar, the vector types store the dimension in atttypmod
            # directly rather than offset by the four-byte header length.
            dimensions = int(typmod)
        columns.append(
            ColumnInfo(
                name=str(name),
                data_type=str(data_type),
                type_name=str(type_name),
                position=int(position),
                not_null=bool(not_null),
                generated=bool(generated),
                default=default,
                dimensions=dimensions,
            )
        )
    return columns


def _read_indexes(run: Executor, oid: int) -> list[IndexInfo]:
    rows = run(
        # Every column is aliased. Two of these are array_agg subqueries, and a driver
        # returning mapping rows would collapse the pair into one key, so the row would
        # arrive one column short of what this unpacks.
        "SELECT i.relname AS index_name, am.amname AS method, "
        "       pg_catalog.pg_get_indexdef(i.oid) AS definition, "
        "       pg_catalog.pg_relation_size(i.oid)::bigint AS size_bytes, "
        "       x.indisvalid AS is_valid, x.indisprimary AS is_primary, "
        "       x.indisunique AS is_unique, i.reloptions AS options, "
        "       pg_catalog.pg_get_expr(x.indpred, x.indrelid) AS predicate, "
        "       (SELECT array_agg(pg_catalog.pg_get_indexdef(i.oid, k.ord::int, true) "
        "               ORDER BY k.ord) "
        "          FROM generate_series(1, x.indnkeyatts) WITH ORDINALITY AS k(n, ord)"
        "       ) AS keys, "
        "       (SELECT array_agg(o.opcname ORDER BY k.ord) "
        "          FROM unnest(x.indclass::oid[]) WITH ORDINALITY AS k(cls, ord) "
        "          JOIN pg_catalog.pg_opclass o ON o.oid = k.cls"
        "       ) AS opclasses "
        "FROM pg_catalog.pg_index x "
        "JOIN pg_catalog.pg_class i ON i.oid = x.indexrelid "
        "JOIN pg_catalog.pg_am am ON am.oid = i.relam "
        f"WHERE x.indrelid = {oid} "
        "ORDER BY i.relname"
    )
    indexes = []
    for (
        name,
        method,
        definition,
        size,
        valid,
        primary,
        unique,
        reloptions,
        predicate,
        keys,
        opclasses,
    ) in rows:
        indexes.append(
            IndexInfo(
                name=str(name),
                method=str(method),
                keys=[str(k) for k in (keys or [])],
                opclasses=[str(o) for o in (opclasses or [])],
                definition=str(definition),
                size_bytes=int(size or 0),
                valid=bool(valid),
                primary=bool(primary),
                unique=bool(unique),
                options=_parse_reloptions(reloptions),
                predicate=predicate,
            )
        )
    return indexes


def _parse_reloptions(reloptions: Any) -> dict[str, str]:
    """``{m=16,ef_construction=64}`` as a dict, for reporting what an index was built with."""
    options: dict[str, str] = {}
    for entry in reloptions or []:
        text = str(entry)
        if "=" in text:
            key, _, value = text.partition("=")
            options[key.strip()] = value.strip()
    return options


def _count_rows(run: Executor, quoted: str, reltuples: float, table_size: int) -> tuple[int, str]:
    """The row count, exact when that is cheap and honest about it when it is not.

    ``reltuples`` is -1 on a table that has never been analysed, and 0 on a partitioned
    parent, so neither value can be trusted on its own. The physical size can, because
    it is measured rather than estimated.
    """
    if table_size <= EXACT_COUNT_MAX_BYTES:
        exact = run(f"SELECT count(*) FROM {quoted}")
        if exact:
            return int(exact[0][0]), "count(*)"
    if reltuples is not None and reltuples >= 0 and (reltuples > 0 or table_size == 0):
        return int(reltuples), "pg_class.reltuples"
    # Never analysed and too big to count: ask the planner, because its estimate is
    # also the number that decides whether the vector index gets used at all.
    plan = run(f"EXPLAIN (FORMAT JSON) SELECT * FROM {quoted}")
    if plan:
        node = _explain_root(plan[0][0])
        if node is not None:
            return int(node.get("Plan Rows", 0)), "planner"
    return max(int(reltuples or 0), 0), "pg_class.reltuples"


def _explain_root(payload: Any) -> Optional[dict[str, Any]]:
    """The top plan node from an ``EXPLAIN (FORMAT JSON)`` result.

    Drivers disagree about JSON: psycopg decodes it, asyncpg hands back a string.
    """
    import json

    if isinstance(payload, (str, bytes, bytearray)):
        payload = json.loads(payload)
    if isinstance(payload, list) and payload:
        payload = payload[0]
    if isinstance(payload, dict):
        plan = payload.get("Plan")
        if isinstance(plan, dict):
            return plan
    return None


def parse_version(version: Optional[str]) -> tuple[int, ...]:
    """``"0.8.6"`` as ``(0, 8, 6)``, ignoring any suffix, for ordered comparison."""
    if not version:
        return ()
    parts: list[int] = []
    for chunk in str(version).split("."):
        match = re.match(r"\d+", chunk)
        if not match:
            break
        parts.append(int(match.group()))
    return tuple(parts)


# ------------------------------------------------------------------ config guessing


def suggest_config(info: TableInfo) -> Config:
    """Pick a working configuration out of the columns that actually exist.

    Every choice here is a guess, but each one is a guess the report prints, so a wrong
    guess is visible rather than silently wrong.
    """
    vector_column = _pick_vector_column(info)
    if vector_column is None:
        raise ValueError(
            f"{info.qualified} has no vector column. Add one with "
            f"ALTER TABLE {quote_ident(info.table)} ADD COLUMN embedding vector(1536), "
            "then run init again."
        )

    tsvector_column = _pick_tsvector_column(info)
    text_column = _pick_text_column(info, tsvector_column)
    if text_column is None:
        raise ValueError(
            f"{info.qualified} has no text column to search. Hybrid search needs one "
            "text column alongside the embedding."
        )

    index = info.vector_index_for(vector_column.name)
    metric = index.metric if index is not None and index.metric is not None else METRICS["cosine"]
    vector_type: VectorType = "halfvec" if vector_column.type_name == "halfvec" else "vector"

    return Config(
        table=info.qualified,
        text_column=text_column.name,
        vector_column=vector_column.name,
        id_column=_pick_id_column(info),
        tsvector_column=(
            tsvector_column.name
            if tsvector_column
            else _propose_tsvector_name(info, text_column.name)
        ),
        language=_detect_language(tsvector_column),
        vector_type=vector_type,
        metric=metric,
        filter_columns=_pick_filter_columns(info, exclude={text_column.name, vector_column.name}),
        extra_columns=_pick_extra_columns(info, text_column.name),
    )


def _pick_vector_column(info: TableInfo) -> Optional[ColumnInfo]:
    candidates = info.vector_columns
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    # More than one embedding column is normal in tables that are mid-migration between
    # two models. Prefer the one that is already indexed, since that is the one the
    # application is querying today.
    indexed = [c for c in candidates if info.vector_index_for(c.name)]
    pool = indexed or candidates
    for preferred in _VECTOR_NAME_PREFERENCE:
        for column in pool:
            if column.name == preferred:
                return column
    return pool[0]


def _pick_tsvector_column(info: TableInfo) -> Optional[ColumnInfo]:
    candidates = info.tsvector_columns
    if not candidates:
        return None
    generated = [c for c in candidates if c.generated]
    return (generated or candidates)[0]


def _propose_tsvector_name(info: TableInfo, text_column: str) -> str:
    """A name for the tsvector column the migration is about to add.

    Naming it here rather than leaving it None means ``init`` produces the stored
    column, which is the arrangement the GIN index can actually use. Until the
    migration runs, a search against this config fails on a missing column, and the
    doctor reports exactly that with the statement that fixes it.
    """
    for candidate in ("fts", f"{text_column}_tsv", f"{text_column}_tsvector", "search_tsv"):
        if info.column(candidate) is None:
            return candidate
    return f"{text_column}_fts"


def _pick_text_column(
    info: TableInfo, tsvector_column: Optional[ColumnInfo]
) -> Optional[ColumnInfo]:
    candidates = info.text_columns
    if not candidates:
        return None
    # If a generated tsvector already exists, the column it reads from is the answer by
    # definition: matching it keeps ranking and highlighting on the same text.
    if tsvector_column is not None and tsvector_column.default:
        words = set(_WORD_RE.findall(tsvector_column.default))
        sourced = [c for c in candidates if c.name in words]
        if sourced:
            return sourced[0]
    for preferred in _TEXT_NAME_PREFERENCE:
        for column in candidates:
            if column.name == preferred:
                return column
    return candidates[0]


def _pick_id_column(info: TableInfo) -> str:
    if len(info.primary_key) == 1:
        return info.primary_key[0]
    if info.column("id") is not None:
        return "id"
    if info.primary_key:
        # A composite key cannot address a row with one value, and the fusion joins on
        # exactly one. Say which column was picked rather than failing at query time.
        return info.primary_key[0]
    return info.columns[0].name if info.columns else "id"


def _detect_language(tsvector_column: Optional[ColumnInfo]) -> str:
    """Reuse the text search configuration the existing generated column was built with."""
    if tsvector_column is not None and tsvector_column.default:
        match = re.search(r"'([a-z_]+)'::regconfig", tsvector_column.default)
        if match:
            return match.group(1)
    return "english"


def _pick_filter_columns(info: TableInfo, exclude: set[str]) -> list[str]:
    """Columns worth allowing as filters: the ones already indexed, plus tenant keys.

    ``filter_columns`` is an allow-list, so being generous here weakens it. The rule is
    that a column earns a place by already being indexed or by being named like the
    thing every multi-tenant application filters on.
    """
    chosen: list[str] = []
    for column in info.columns:
        if column.name in exclude or column.is_vector or column.is_tsvector:
            continue
        if column.name in info.primary_key:
            continue
        indexed = any(
            index.method == "btree" and index.covers(column.name) and not index.primary
            for index in info.indexes
        )
        if indexed or column.name in _FILTER_NAME_HINTS:
            chosen.append(column.name)
    return chosen[:8]


def _pick_extra_columns(info: TableInfo, text_column: str) -> list[str]:
    """One or two columns that make a result row readable in a terminal."""
    extras = []
    for name in ("title", "name", "url", "source", "path"):
        column = info.column(name)
        if column is not None and column.name != text_column and column.is_text:
            extras.append(column.name)
    return extras[:2]


# ------------------------------------------------------------------------ migration


def ivfflat_lists(rows: int) -> tuple[int, str]:
    """The ``lists`` value for an ivfflat index, with the arithmetic that produced it.

    pgvector's guidance changes shape at one million rows: ``rows / 1000`` up to and
    including a million, ``sqrt(rows)`` above it. The threshold is the part everybody
    gets wrong, so the arithmetic is returned alongside the number and printed.
    """
    rows = max(int(rows), 0)
    if rows <= IVFFLAT_SQRT_THRESHOLD:
        lists = max(rows // 1000, 1)
        arithmetic = (
            f"{rows:,} rows / 1000 = {lists:,} lists "
            f"(at or below {IVFFLAT_SQRT_THRESHOLD:,} rows pgvector's rule is rows/1000)"
        )
    else:
        lists = max(int(math.sqrt(rows)), 1)
        arithmetic = (
            f"sqrt({rows:,}) = {lists:,} lists "
            f"(above {IVFFLAT_SQRT_THRESHOLD:,} rows pgvector's rule changes to sqrt(rows))"
        )
    return lists, arithmetic


def ivfflat_probes(lists: int) -> int:
    """The starting ``ivfflat.probes``: sqrt(lists), which is pgvector's own advice."""
    return max(int(math.sqrt(max(lists, 1))), 1)


def build_migration(config: Config, info: TableInfo) -> list[Statement]:
    """The DDL that turns ``info`` into a table ``config`` can search efficiently.

    Only statements that would change something are returned, so re-running this after
    applying it yields an empty list. Alternatives and tuning are marked ``optional``.
    """
    statements: list[Statement] = []
    table = quote_ident(config.table)

    if not info.pgvector_version:
        statements.append(
            Statement(
                sql="CREATE EXTENSION IF NOT EXISTS vector;",
                reason="pgvector is not installed in this database; the vector type "
                "and both index methods come from it.",
                kind="extension",
                notes=[
                    "Requires a superuser or a role with CREATE on the database.",
                    "On managed Postgres this is usually the only privileged step.",
                ],
            )
        )

    statements.extend(_vector_column_statements(config, info, table))
    statements.extend(_text_statements(config, info, table))
    statements.extend(_vector_index_statements(config, info, table))
    statements.extend(_filter_index_statements(config, info, table))

    required = [s for s in statements if not s.optional]
    if required or not info.analyzed:
        statements.append(
            Statement(
                sql=f"ANALYZE {table};",
                reason="The planner decides whether to use the vector index from table "
                "statistics; without them a filtered query silently falls back to a "
                "sequential scan."
                if info.analyzed
                else "This table has never been analysed, so every plan the server "
                "produces for it is a guess from default constants.",
                kind="maintenance",
                notes=[
                    "ivfflat must also be built after the data is loaded: an index "
                    "built on an empty table has meaningless centroids."
                ],
            )
        )
    return statements


def _vector_column_statements(config: Config, info: TableInfo, table: str) -> list[Statement]:
    column = info.column(config.vector_column)
    if column is None:
        raise ValueError(
            f"{config.table} has no column named {config.vector_column!r}. "
            f"Columns: {', '.join(c.name for c in info.columns) or 'none'}."
        )
    if not column.is_vector:
        raise ValueError(
            f"{config.vector_column!r} is {column.data_type}, not a vector type. "
            "pghybrid needs a vector, halfvec or sparsevec column to search."
        )
    if column.dimensions is None:
        return [
            Statement(
                sql=f"-- ALTER TABLE {table} ALTER COLUMN {quote_ident(column.name)} "
                f"TYPE {column.type_name}(<dimensions>);",
                reason=f"{column.name} was declared as bare {column.type_name} with no "
                "dimension. pgvector can store such a column but cannot index it, so "
                "every search is a sequential scan.",
                kind="column",
                notes=[
                    "Fill in the dimension your model produces, then run init again.",
                    "The rewrite takes an ACCESS EXCLUSIVE lock for its duration.",
                ],
            )
        ]
    if config.vector_type == "halfvec" and column.type_name != "halfvec":
        dimensions = column.dimensions
        return [
            Statement(
                sql=f"ALTER TABLE {table} ALTER COLUMN {quote_ident(column.name)} "
                f"TYPE halfvec({dimensions}) "
                f"USING {quote_ident(column.name)}::halfvec({dimensions});",
                reason="The config asks for halfvec but the column is stored as "
                f"{column.data_type}; the distance operator needs both sides to be the "
                "same type for the index to be usable.",
                kind="column",
                notes=[
                    "Halves storage and index size. Recall loss is typically under a "
                    "tenth of a percent at 1536 dimensions, but measure it with "
                    "`pghybrid doctor` before and after rather than taking that on faith.",
                    "The rewrite takes an ACCESS EXCLUSIVE lock for its duration.",
                ],
            )
        ]
    return []


def _text_statements(config: Config, info: TableInfo, table: str) -> list[Statement]:
    statements: list[Statement] = []
    text_column = info.column(config.text_column)
    if text_column is None:
        raise ValueError(
            f"{config.table} has no column named {config.text_column!r}. "
            f"Columns: {', '.join(c.name for c in info.columns) or 'none'}."
        )

    # The expression the query builder computes inline when no stored column exists.
    # Both forms have to agree exactly, so it is taken from the query builder itself.
    inline = _inline_tsvector_expr(replace(config, tsvector_column=None))
    stored = config.tsvector_column
    stored_column = info.column(stored) if stored else None

    if stored and stored_column is None:
        statements.append(
            Statement(
                sql=f"ALTER TABLE {table} ADD COLUMN {quote_ident(stored)} tsvector "
                f"GENERATED ALWAYS AS ({inline}) STORED;",
                reason="A stored tsvector is what makes the keyword half of the search "
                "indexable; computing it per query costs a full parse of every row.",
                kind="column",
                notes=[
                    "The two-argument to_tsvector is mandatory here. to_tsvector(text) "
                    "is STABLE rather than IMMUTABLE because it reads "
                    "default_text_search_config, and Postgres rejects it in a generated "
                    "column with 'generation expression is not immutable'.",
                    "coalesce keeps NULL text from producing a NULL tsvector, which "
                    "would match nothing and quietly drop those rows from results.",
                    "ADD COLUMN with a generated expression rewrites the table under an "
                    "ACCESS EXCLUSIVE lock. On a large table the expression index below "
                    "is the alternative: no column, no rewrite, slightly slower ranking.",
                ],
            )
        )

    target = quote_ident(stored) if stored else inline
    existing = info.text_index_for(stored, inline)
    if existing is None:
        name = _index_name(info, stored or config.text_column, "gin")
        statements.append(
            Statement(
                sql=f"CREATE INDEX IF NOT EXISTS {quote_ident(name)} ON {table} "
                f"USING gin ({target});",
                reason="Without a GIN index the keyword half of every hybrid query "
                "scans and re-parses the whole table, which is usually the slower half.",
                kind="index",
                concurrent_sql=f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {quote_ident(name)} "
                f"ON {table} USING gin ({target});",
                notes=[
                    "GIN, not GiST: GIN is larger and slower to build but answers @@ "
                    "far faster, and search tables are read far more than written.",
                ]
                + (
                    []
                    if stored
                    else [
                        "This indexes the expression, so it is only used when the query "
                        "spells the expression identically. pghybrid does; hand-written "
                        "queries often do not."
                    ]
                ),
            )
        )
    return statements


def _vector_index_statements(config: Config, info: TableInfo, table: str) -> list[Statement]:
    statements: list[Statement] = []
    column = info.column(config.vector_column)
    if column is None or column.dimensions is None:
        return statements

    dimensions = column.dimensions
    rows = info.row_count
    ops = config.ops_class
    existing = info.vector_index_for(column.name)
    column_sql = quote_ident(column.name)

    stored_type = "halfvec" if column.type_name == "halfvec" else column.type_name
    limit = MAX_INDEXED_DIMENSIONS.get(stored_type, 2000)
    over_limit = dimensions > limit

    if (
        existing is not None
        and existing.metric is not None
        and existing.metric is not config.metric
    ):
        statements.append(
            Statement(
                sql=f"-- DROP INDEX {quote_ident(existing.name)};",
                reason=f"{existing.name} is built with {existing.opclasses[0]} but the "
                f"config searches with {config.metric.name} ({config.metric.operator}). "
                "An operator class mismatch does not error; the index is simply never "
                "used and every search silently becomes a sequential scan.",
                kind="index",
                optional=True,
                notes=[
                    f"Either change the config metric to {existing.metric.name}, or drop "
                    "this index and build the one below.",
                ],
            )
        )
    elif existing is not None and not over_limit:
        # An index that already matches is left alone, but ivfflat sizing goes stale as
        # the table grows and nothing warns you.
        if existing.method == "ivfflat" and existing.lists:
            wanted, arithmetic = ivfflat_lists(rows)
            if existing.lists < wanted // 2 or existing.lists > wanted * 2:
                statements.append(
                    Statement(
                        sql=f"DROP INDEX {quote_ident(existing.name)};\n"
                        f"CREATE INDEX {quote_ident(existing.name)} ON {table} "
                        f"USING ivfflat ({column_sql} {ops}) WITH (lists = {wanted});",
                        reason=f"{existing.name} was built with lists = "
                        f"{existing.lists:,}, but the table now holds {rows:,} rows: "
                        f"{arithmetic}.",
                        kind="index",
                        optional=True,
                        notes=[
                            "ivfflat lists are fixed at build time; the index has to be "
                            "rebuilt to resize them."
                        ],
                    )
                )
        return statements

    if over_limit and stored_type != "halfvec" and dimensions <= MAX_INDEXED_DIMENSIONS["halfvec"]:
        # The 3072-dimension case. A plain vector index is not merely slow here, it is
        # rejected outright, and the halfvec route is the documented way through.
        expression = f"(({column_sql}::halfvec({dimensions})) {config.metric.ops_halfvec})"
        name = _index_name(info, column.name, "hnsw_halfvec")
        statements.append(
            Statement(
                sql=f"CREATE INDEX IF NOT EXISTS {quote_ident(name)} ON {table} "
                f"USING hnsw {expression};",
                reason=f"{column.name} has {dimensions:,} dimensions and pgvector can "
                f"only index {limit:,} of a {stored_type} column. Indexing the halfvec "
                "cast is the supported way to index a wider embedding.",
                kind="index",
                concurrent_sql=f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {quote_ident(name)} "
                f"ON {table} USING hnsw {expression};",
                notes=[
                    "An expression index is only used when the query writes the same "
                    f"expression, so searches must ORDER BY {column.name}"
                    f"::halfvec({dimensions}) {config.metric.operator} "
                    f"$1::halfvec({dimensions}).",
                    "To keep pghybrid's generated SQL matching it, store the column as "
                    "halfvec and set vector_type='halfvec' instead of casting per query.",
                    "CONCURRENTLY cannot run inside a transaction block.",
                ],
            )
        )
        return statements

    name = _index_name(info, column.name, "hnsw")
    build_note = (
        f"m = {HNSW_DEFAULT_M} and ef_construction = {HNSW_DEFAULT_EF_CONSTRUCTION} are "
        "pgvector's defaults, so they are left out of the statement rather than written "
        "in to look like tuning. Raising ef_construction improves recall at build time "
        "and costs build time only."
    )
    statements.append(
        Statement(
            sql=f"CREATE INDEX IF NOT EXISTS {quote_ident(name)} ON {table} "
            f"USING hnsw ({column_sql} {ops});",
            reason="HNSW is the default recommendation: it needs no data to be present "
            "at build time, its recall is set per query with hnsw.ef_search, and it "
            "degrades gracefully under filters. ivfflat is smaller and faster to build.",
            kind="index",
            concurrent_sql=f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {quote_ident(name)} "
            f"ON {table} USING hnsw ({column_sql} {ops});",
            notes=[
                build_note,
                f"{ops} is required because the config searches with "
                f"{config.metric.name} ({config.metric.operator}); an index with a "
                "different operator class is never used and never warns.",
                "CONCURRENTLY cannot run inside a transaction block. Send it on its own "
                "connection with autocommit on, and expect roughly double the build time.",
                f"Build cost scales with the row count ({rows:,} today). Raise "
                "maintenance_work_mem until the graph fits in memory, or the build "
                "falls back to a much slower two-pass path and says so in the log.",
            ],
        )
    )

    lists, arithmetic = ivfflat_lists(rows)
    probes = ivfflat_probes(lists)
    ivf_name = _index_name(info, column.name, "ivfflat")
    statements.append(
        Statement(
            sql=f"CREATE INDEX IF NOT EXISTS {quote_ident(ivf_name)} ON {table} "
            f"USING ivfflat ({column_sql} {ops}) WITH (lists = {lists});",
            reason=f"Alternative to HNSW: builds in a fraction of the time and takes "
            f"far less space, at lower recall for the same latency. {arithmetic}.",
            kind="index",
            optional=True,
            concurrent_sql=f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {quote_ident(ivf_name)} "
            f"ON {table} USING ivfflat ({column_sql} {ops}) WITH (lists = {lists});",
            notes=[
                arithmetic,
                f"Start at ivfflat.probes = {probes} (sqrt({lists:,})), then use the "
                "sweep in `pghybrid doctor` to pick the point you actually want.",
                "Build this after the rows are loaded. An ivfflat index built on an "
                "empty or small table has centroids that describe nothing.",
                "lists cannot be changed without rebuilding, so a table that grows by an "
                "order of magnitude needs the index rebuilt.",
            ],
        )
    )

    if (
        stored_type == "vector"
        and rows >= 100_000
        and dimensions <= MAX_INDEXED_DIMENSIONS["halfvec"]
    ):
        expression = f"(({column_sql}::halfvec({dimensions})) {config.metric.ops_halfvec})"
        half_name = _index_name(info, column.name, "hnsw_halfvec")
        statements.append(
            Statement(
                sql=f"CREATE INDEX IF NOT EXISTS {quote_ident(half_name)} ON {table} "
                f"USING hnsw {expression};",
                reason=f"At {rows:,} rows the index is large enough that half precision "
                "is worth it: a halfvec HNSW index is about half the size and builds in "
                "about half the time, and recall is usually unchanged.",
                kind="index",
                optional=True,
                concurrent_sql=f"CREATE INDEX CONCURRENTLY IF NOT EXISTS "
                f"{quote_ident(half_name)} ON {table} USING hnsw {expression};",
                notes=[
                    "Usually free on recall, but 'usually' is not 'always': measure it "
                    "with `pghybrid doctor` on your own data before and after.",
                    "The query has to spell the cast the same way for the index to be "
                    f"used: ORDER BY {column.name}::halfvec({dimensions}) "
                    f"{config.metric.operator} $1::halfvec({dimensions}).",
                    "The column keeps full precision; only the index is quantised.",
                ],
            )
        )
    return statements


def _filter_index_statements(config: Config, info: TableInfo, table: str) -> list[Statement]:
    """B-tree indexes for the filters, which is what stops filtered search seq-scanning."""
    statements = []
    for column_name in config.filter_columns:
        column = info.column(column_name)
        if column is None:
            continue
        if any(i.method == "btree" and i.covers(column_name) for i in info.indexes):
            continue
        name = _index_name(info, column_name, "btree")
        statements.append(
            Statement(
                sql=f"CREATE INDEX IF NOT EXISTS {quote_ident(name)} ON {table} "
                f"({quote_ident(column_name)});",
                reason=f"{column_name} is a declared filter column with no b-tree index. "
                "When a filter is selective the planner abandons the vector index and "
                "scans, so the filter needs an index of its own to stay cheap.",
                kind="index",
                optional=True,
                concurrent_sql=f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {quote_ident(name)} "
                f"ON {table} ({quote_ident(column_name)});",
                notes=[
                    "For a filter that is always the same value, a partial vector index "
                    f"(WHERE {column_name} = ...) beats both: it keeps ANN search inside "
                    "the subset instead of filtering after it.",
                ],
            )
        )
    return statements


def render_migration(statements: Sequence[Statement], *, concurrent: bool = False) -> str:
    """The whole migration as a runnable .sql file, reasons kept as comments."""
    if not statements:
        return "-- nothing to do: the schema already matches this configuration.\n"
    blocks = [s.to_text(concurrent=concurrent) for s in statements if not s.optional]
    optional = [s.to_text(concurrent=concurrent) for s in statements if s.optional]
    text = "\n\n".join(blocks)
    if optional:
        text += "\n\n-- Options and alternatives below. None of these are required.\n\n"
        text += "\n\n".join(optional)
    return text + "\n"


def _index_name(info: TableInfo, column: str, suffix: str) -> str:
    """A predictable index name, truncated the way Postgres would truncate it anyway."""
    base = _IDENT_SAFE_RE.sub("_", f"{info.name}_{column}_{suffix}_idx").strip("_")
    if len(base.encode("utf-8")) <= MAX_IDENTIFIER_LENGTH:
        return base
    encoded = base.encode("utf-8")[:MAX_IDENTIFIER_LENGTH]
    return encoded.decode("utf-8", "ignore")


# --------------------------------------------------------------------------- shared


def _literal(value: str) -> str:
    """A single-quoted SQL string literal.

    Used only for catalog lookups of names that have already been validated, but the
    quote doubling is done properly anyway: a helper that is safe only sometimes is a
    helper that will eventually be used at the wrong moment.
    """
    if "\x00" in value:
        raise ValueError("SQL string literals cannot contain a null byte")
    escaped = value.replace("'", "''")
    return f"'{escaped}'"


def _normalize_sql(expression: str) -> str:
    """Collapse whitespace and casing so two spellings of one expression compare equal."""
    return re.sub(r"\s+", "", expression).lower().replace('"', "")


def human_bytes(value: Optional[int]) -> str:
    """Sizes in the same units psql prints, so they can be compared by eye."""
    if value is None:
        return "-"
    size = float(value)
    for unit in ("bytes", "kB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            if unit == "bytes":
                return f"{int(size)} bytes"
            return f"{size:.0f} {unit}" if size >= 10 else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.0f} TB"
