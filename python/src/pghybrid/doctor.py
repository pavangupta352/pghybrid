"""The diagnostic that produces a number worth acting on.

Every vector database ships a recall figure from a benchmark someone else ran. This
one measures recall on the table in front of you: it draws query vectors from your own
rows, computes exact ground truth by forcing a sequential scan, runs the same query
through the index, and reports the overlap. A measured ``recall@10: 0.68`` is the only
version of that number anyone should act on, so the sample size is printed next to it
and the ground truth is verified rather than assumed.

Everything here is read-only by default. Each probe runs inside its own transaction
with a statement timeout and, where the server allows it, an explicit read-only flag,
so a diagnostic that runs against production cannot leave anything behind. The one
statement that writes -- ``ANALYZE`` -- only runs when ``allow_write=True`` is passed.

The ``execute`` argument is the same callable :mod:`pghybrid.schema` takes; see
:func:`pghybrid.schema.dbapi_executor` for the one-line adapter.
"""

from __future__ import annotations

import contextlib
import json
import math
import re
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .config import Config
from .schema import (
    Executor,
    IndexInfo,
    Statement,
    TableInfo,
    build_migration,
    introspect,
    ivfflat_probes,
    parse_version,
)
from .sql import build_search_sql, quote_ident

Row = Sequence[Any]
Execute = Callable[..., Sequence[Row]]

#: Ceiling on every probe. A diagnostic that hangs on a production table is worse than
#: one that reports less, so the timeout is short and its expiry is reported as a
#: finding rather than raised.
DEFAULT_STATEMENT_TIMEOUT_MS = 15_000

#: pgvector's own defaults, used to say whether a setting has actually been changed.
HNSW_DEFAULT_EF_SEARCH = 40
IVFFLAT_DEFAULT_PROBES = 1

#: Recall below this is reported as a problem rather than a measurement.
POOR_RECALL = 0.90
BAD_RECALL = 0.75

_LEVEL_ORDER = {"error": 0, "warn": 1, "info": 2, "ok": 3}


# --------------------------------------------------------------------------- values


@dataclass(frozen=True)
class Timing:
    """Client-side latency for one batch of identical-shaped queries."""

    n: int = 0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    mean_ms: float = 0.0

    @classmethod
    def of(cls, samples: Sequence[float]) -> Timing:
        if not samples:
            return cls()
        ordered = sorted(samples)
        return cls(
            n=len(ordered),
            p50_ms=_percentile(ordered, 0.50),
            p95_ms=_percentile(ordered, 0.95),
            mean_ms=sum(ordered) / len(ordered),
        )

    def __str__(self) -> str:
        return f"{self.p50_ms:.2f} ms p50 / {self.p95_ms:.2f} ms p95"


@dataclass(frozen=True)
class RecallResult:
    """One measured recall@k, with everything needed to judge how much to trust it."""

    label: str
    k: int
    sample: int
    recall: float
    timing: Timing
    settings: list[str] = field(default_factory=list)
    used_vector_index: Optional[bool] = None
    exact_ground_truth: bool = True
    notes: list[str] = field(default_factory=list)

    @property
    def headline(self) -> str:
        return f"recall@{self.k}: {self.recall:.2f}  ({self.sample} sampled queries)"


@dataclass(frozen=True)
class SweepRow:
    """Recall and latency at one value of the index's search-effort knob."""

    setting: str
    value: int
    recall: float
    timing: Timing
    is_current: bool = False
    is_default: bool = False


@dataclass(frozen=True)
class PlanProbe:
    """What the planner does with one real query shape."""

    label: str
    sql: str
    scan: str
    used_vector_index: bool
    index_name: Optional[str] = None
    estimated_rows: float = 0.0
    filter: Optional[str] = None
    error: Optional[str] = None


@dataclass(frozen=True)
class FilterProbe:
    """A filter value drawn from the table, and what it does to the plan and to recall."""

    column: str
    value: Any
    literal: str
    selectivity: Optional[float]
    source: str

    @property
    def predicate(self) -> str:
        return f"{self.column} = {self.literal}"


@dataclass(frozen=True)
class IterativeScan:
    """Whether this server can rescue filtered recall, and what it did when asked to."""

    supported: bool
    version: Optional[str]
    settings: dict[str, str] = field(default_factory=dict)
    measured: Optional[RecallResult] = None
    baseline: Optional[RecallResult] = None


@dataclass(frozen=True)
class Finding:
    """One thing that is true about this table, with the statement that changes it."""

    level: str
    title: str
    detail: str
    fix: Optional[str] = None


@dataclass
class DoctorReport:
    """The whole diagnostic. ``to_text()`` renders what the CLI prints."""

    config: Config
    info: TableInfo
    findings: list[Finding] = field(default_factory=list)
    recall: Optional[RecallResult] = None
    filtered: list[RecallResult] = field(default_factory=list)
    sweep: list[SweepRow] = field(default_factory=list)
    plans: list[PlanProbe] = field(default_factory=list)
    filters: list[FilterProbe] = field(default_factory=list)
    iterative: Optional[IterativeScan] = None
    recommendations: list[Statement] = field(default_factory=list)
    null_fraction: Optional[float] = None
    null_fraction_source: str = ""
    sample_requested: int = 0
    sample_used: int = 0
    k: int = 10
    duration_ms: float = 0.0
    read_only: bool = True
    session_settings_shared: bool = True

    @property
    def headline(self) -> str:
        """The one line people paste into an issue."""
        if self.recall is None:
            return "recall: not measured"
        return self.recall.headline

    @property
    def worst_level(self) -> str:
        if not self.findings:
            return "ok"
        return min((f.level for f in self.findings), key=lambda lv: _LEVEL_ORDER.get(lv, 9))

    def to_text(self, *, width: int = 78) -> str:
        return _render_report(self, width=width)


# ---------------------------------------------------------------------- entry point


def doctor(
    execute: Execute,
    config: Config,
    *,
    sample: int = 50,
    k: int = 10,
    statement_timeout_ms: int = DEFAULT_STATEMENT_TIMEOUT_MS,
    filters: Optional[dict[str, Any]] = None,
    sweep: bool = True,
    allow_write: bool = False,
) -> DoctorReport:
    """Measure what this table's search actually does, and say what to change.

    ``sample`` query vectors are drawn from the table itself and every recall figure is
    an average over them. ``allow_write`` permits exactly one statement, ``ANALYZE``,
    and only when the table has never been analysed; nothing else in this module writes.
    """
    if sample < 1:
        raise ValueError("sample must be >= 1")
    if k < 1:
        raise ValueError("k must be >= 1")

    started = time.perf_counter()
    run = Executor(execute)
    info = introspect(run, config.table)
    report = DoctorReport(
        config=config, info=info, sample_requested=sample, k=k, read_only=not allow_write
    )

    probe = _Prober(run, config, info, timeout_ms=statement_timeout_ms)
    report.session_settings_shared = probe.session_settings_shared
    gucs = probe.vector_settings()

    _check_inventory(report, probe)

    if allow_write and not info.analyzed:
        probe.analyze()
        report.info = probe.info = info = introspect(run, config.table)
        report.findings.append(
            Finding("info", "ANALYZE ran", f"{info.qualified} had no statistics; they exist now.")
        )

    report.null_fraction, report.null_fraction_source = probe.null_fraction()
    report.filters = probe.filter_probes(filters)
    report.plans = probe.plan_probes(report.filters)

    vectors = probe.sample_vectors(sample)
    report.sample_used = len(vectors)
    if not vectors:
        report.findings.append(
            Finding(
                "warn",
                "recall not measured",
                f"No rows have a non-NULL {config.vector_column}, so there is nothing "
                "to sample query vectors from.",
            )
        )
    else:
        truth, exact_ok, truth_note = probe.ground_truth(vectors, k)
        if truth_note:
            report.findings.append(truth_note)
        report.recall = probe.measure(vectors, truth, k, label="unfiltered", exact=exact_ok)
        if sweep:
            report.sweep = probe.sweep(vectors, truth, k, gucs)
        # After the sweep, so the verdict can name the setting the number came from.
        _check_recall(report)
        _measure_filtered(report, probe, vectors, k, exact_ok, gucs)

    report.recommendations = _recommendations(report)
    for message in probe.errors:
        report.findings.append(Finding("warn", "probe did not complete", message))
    report.findings = _dedupe(report.findings)
    report.findings.sort(key=lambda f: _LEVEL_ORDER.get(f.level, 9))
    report.duration_ms = (time.perf_counter() - started) * 1000.0
    return report


def _dedupe(findings: Sequence[Finding]) -> list[Finding]:
    """One line per problem.

    The same failure is reached from several probes -- a dropped table breaks the
    unfiltered measurement and every filtered one -- and repeating it makes the report
    look like several problems instead of one.
    """
    seen: set[tuple[str, str, str]] = set()
    unique: list[Finding] = []
    for finding in findings:
        key = (finding.level, finding.title, finding.detail)
        if key not in seen:
            seen.add(key)
            unique.append(finding)
    return unique


# ------------------------------------------------------------------------- probing


class _Prober:
    """Every statement the doctor sends, with the safety rails attached.

    Probes are grouped into as few transactions as possible: opening one transaction
    per query would triple the round trips and make the latency numbers mostly network.
    """

    def __init__(self, run: Executor, config: Config, info: TableInfo, *, timeout_ms: int) -> None:
        self.run = run
        self.config = config
        self.info = info
        self.timeout_ms = int(timeout_ms)
        self.table = quote_ident(config.table)
        self.id_column = quote_ident(config.id_column)
        self.vector_column = quote_ident(config.vector_column)
        self.errors: list[str] = []
        self.read_only_supported = False
        self.session_settings_shared = True
        self._in_session = False
        self._cached_vector: Optional[str] = None
        self._check_session()

    # -- session plumbing ---------------------------------------------------

    def _check_session(self) -> None:
        """Confirm that SET LOCAL in one call is visible to the next.

        A connection pooler in statement mode, or an ``execute`` that opens a fresh
        connection per call, breaks that assumption -- and would silently turn the
        forced-sequential-scan ground truth into an index-assisted one, which is the
        one failure that would make every number in this report wrong.
        """
        try:
            self.run("BEGIN")
            self.run("SET LOCAL statement_timeout = '7919ms'")
            observed = self.run.scalar("SHOW statement_timeout")
            self.session_settings_shared = str(observed).startswith("7919")
            try:
                self.run("SET LOCAL transaction_read_only = on")
                self.read_only_supported = str(self.run.scalar("SHOW transaction_read_only")) in (
                    "on",
                    "true",
                    "True",
                )
            except Exception:
                # An aborted transaction cannot be probed further; the finally clause
                # rolls it back and the read-only rail is simply reported as absent.
                self.read_only_supported = False
        except Exception as exc:
            self.session_settings_shared = False
            self.errors.append(f"session probe failed: {_short(exc)}")
        finally:
            self._rollback()

    def _rollback(self) -> None:
        # A rollback that fails leaves nothing to do about it, and raising here would
        # replace the real error with a less useful one.
        with contextlib.suppress(Exception):
            self.run("ROLLBACK")

    @contextmanager
    def session(self, *, settings: Sequence[str] = (), read_only: bool = True) -> Iterator[None]:
        """One transaction, rolled back on the way out.

        ROLLBACK rather than COMMIT for two reasons: it undoes every SET LOCAL even on
        a server that ignored the read-only rail, and it means the diagnostic cannot
        commit anything at all, whatever went wrong inside.

        Nesting is refused rather than tolerated. A nested BEGIN is a no-op in Postgres
        but the inner ROLLBACK ends the outer transaction, which would silently discard
        the enable_indexscan = off that makes the ground truth exact -- and every recall
        number in the report would then be wrong by an unknown amount, with nothing to
        show for it.
        """
        if self._in_session:
            raise RuntimeError(
                "probe sessions cannot nest: the inner ROLLBACK would end the outer "
                "transaction and discard its SET LOCAL settings"
            )
        self.run("BEGIN")
        self._in_session = True
        try:
            self.run(f"SET LOCAL statement_timeout = '{self.timeout_ms}ms'")
            if read_only and self.read_only_supported:
                self.run("SET LOCAL transaction_read_only = on")
            for setting in settings:
                self.run(f"SET LOCAL {setting}")
            yield
        finally:
            self._in_session = False
            self._rollback()

    def analyze(self) -> None:
        """The only statement in this module that writes, and only on request."""
        self.run(f"ANALYZE {self.table}")

    # -- catalog-ish reads --------------------------------------------------

    def vector_settings(self) -> dict[str, str]:
        """pgvector's GUCs and their current values.

        The GUCs only exist once the extension's library has been loaded into the
        backend, which happens on first use of a vector value -- so a trivial cast comes
        first. Skipping it makes a perfectly healthy 0.8 server look like it has no
        iterative scan support.
        """
        settings: dict[str, str] = {}
        try:
            with self.session():
                self.run("SELECT '[1]'::vector")
                rows = self.run(
                    "SELECT name, setting FROM pg_catalog.pg_settings "
                    "WHERE name LIKE 'hnsw.%' OR name LIKE 'ivfflat.%'"
                )
            for name, value in rows:
                settings[str(name)] = str(value)
        except Exception as exc:
            self.errors.append(f"could not read pgvector settings: {_short(exc)}")
        return settings

    def null_fraction(self) -> tuple[Optional[float], str]:
        """How much of the vector column is missing, which caps recall for free."""
        column = self.config.vector_column
        try:
            with self.session():
                stats = self.run(
                    "SELECT null_frac FROM pg_catalog.pg_stats "
                    f"WHERE schemaname = {_literal(self.info.schema)} "
                    f"  AND tablename = {_literal(self.info.name)} "
                    f"  AND attname = {_literal(column)}"
                )
                if stats and stats[0][0] is not None:
                    return float(stats[0][0]), "pg_stats"
                if self.info.table_size_bytes <= 64 * 1024 * 1024:
                    rows = self.run(
                        f"SELECT count(*) FILTER (WHERE {self.vector_column} IS NULL)::float8 "
                        f"/ greatest(count(*), 1) FROM {self.table}"
                    )
                    if rows:
                        return float(rows[0][0]), "count(*)"
        except Exception as exc:
            self.errors.append(f"could not measure null fraction: {_short(exc)}")
        return None, ""

    def sample_vectors(self, sample: int) -> list[str]:
        """Query vectors drawn from the table's own rows.

        Real embeddings are the point: random vectors of the right dimension land in a
        part of the space where nothing lives, and every index looks perfect there.
        """
        vectors: list[str] = []
        try:
            with self.session():
                if self.info.row_count > 200_000:
                    # ORDER BY random() reads every row. TABLESAMPLE reads a few pages,
                    # at the cost of a sample that is clustered rather than uniform --
                    # acceptable for query vectors, and stated in the report.
                    fraction = min(100.0, max(0.01, 400.0 * sample / max(self.info.row_count, 1)))
                    rows = self.run(
                        f"SELECT {self.vector_column} FROM {self.table} "
                        f"TABLESAMPLE SYSTEM ({fraction:.4f}) "
                        f"WHERE {self.vector_column} IS NOT NULL LIMIT {int(sample)}"
                    )
                else:
                    rows = self.run(
                        f"SELECT {self.vector_column} FROM {self.table} "
                        f"WHERE {self.vector_column} IS NOT NULL "
                        f"ORDER BY random() LIMIT {int(sample)}"
                    )
                for row in rows:
                    literal = _vector_literal(row[0])
                    if literal:
                        vectors.append(literal)
        except Exception as exc:
            self.errors.append(f"could not sample query vectors: {_short(exc)}")
        if vectors and self._cached_vector is None:
            self._cached_vector = vectors[0]
        return vectors

    def one_vector(self) -> Optional[str]:
        """A single real embedding, reused by every plan probe that needs one."""
        if self._cached_vector is None:
            self.sample_vectors(1)
        return self._cached_vector

    # -- the measurement ----------------------------------------------------

    def vector_sql(self, vector: str, k: int, *, predicate: str = "", exact: bool = False) -> str:
        """The vector half of the hybrid query, which is the half an index changes."""
        cast = f"{vector}::{self.config.vector_type}"
        distance = f"{self.vector_column} {self.config.metric.operator} {cast}"
        if exact:
            # Adding zero preserves the ordering exactly while making the expression
            # something no pgvector index can answer. It is the belt to the braces of
            # enable_indexscan=off: on a server where session settings do not carry
            # between calls, this is what still guarantees exact ground truth.
            distance = f"({distance}) + 0"
        where = f"WHERE {self.vector_column} IS NOT NULL{predicate}"
        return (
            f"SELECT {self.id_column} FROM {self.table} {where} ORDER BY {distance} LIMIT {int(k)}"
        )

    def ground_truth(
        self, vectors: Sequence[str], k: int, *, predicate: str = ""
    ) -> tuple[list[list[Any]], bool, Optional[Finding]]:
        """Exact nearest neighbours, with the plan checked rather than assumed."""
        settings = [
            "enable_indexscan = off",
            "enable_bitmapscan = off",
            "enable_indexonlyscan = off",
            "enable_seqscan = on",
        ]
        if self.session_settings_shared:
            try:
                with self.session(settings=settings):
                    # The plan is checked from inside the same transaction, because that
                    # is the only place the SET LOCALs above are in effect.
                    probe_sql = self.vector_sql(vectors[0], k, predicate=predicate)
                    if self.uses_vector_index(self.explain_here(probe_sql)) is False:
                        truth = [
                            [row[0] for row in self.run(self.vector_sql(v, k, predicate=predicate))]
                            for v in vectors
                        ]
                        return truth, True, None
            except Exception as exc:
                self.errors.append(f"forced sequential scan failed: {_short(exc)}")

        # Either the settings did not stick, or the planner used the index anyway.
        try:
            with self.session():
                truth = [
                    [
                        row[0]
                        for row in self.run(self.vector_sql(v, k, predicate=predicate, exact=True))
                    ]
                    for v in vectors
                ]
        except Exception as exc:
            self.errors.append(f"exact ground truth failed: {_short(exc)}")
            return (
                [],
                False,
                Finding(
                    "error",
                    "recall could not be measured",
                    f"The exact-search probe did not complete: {_short(exc)}",
                ),
            )
        note = Finding(
            "info",
            "ground truth computed without the planner's help",
            "enable_indexscan = off either did not carry between statements or did not "
            "change the plan, so exact neighbours were computed with an ORDER BY no "
            "index can answer. The recall figure is still exact.",
        )
        return truth, True, note

    def measure(
        self,
        vectors: Sequence[str],
        truth: Sequence[Sequence[Any]],
        k: int,
        *,
        label: str,
        predicate: str = "",
        settings: Sequence[str] = (),
        exact: bool = True,
    ) -> Optional[RecallResult]:
        """Run the indexed query for every sampled vector and score it against truth."""
        if not truth:
            return None
        hits = 0.0
        counted = 0
        latencies: list[float] = []
        notes: list[str] = []
        short_results = 0
        try:
            with self.session(settings=settings):
                for vector, expected in zip(vectors, truth):
                    sql = self.vector_sql(vector, k, predicate=predicate)
                    started = time.perf_counter()
                    rows = self.run(sql)
                    latencies.append((time.perf_counter() - started) * 1000.0)
                    wanted = set(expected)
                    if not wanted:
                        continue
                    got = {row[0] for row in rows}
                    if len(rows) < len(wanted):
                        short_results += 1
                    # The denominator is the number of rows that exist, not k: a filter
                    # that leaves eight matching rows cannot produce ten of them, and
                    # scoring it out of ten would invent a recall problem.
                    hits += len(wanted & got) / len(wanted)
                    counted += 1
        except Exception as exc:
            self.errors.append(f"{label} recall probe failed: {_short(exc)}")
            return None
        if not counted:
            return None
        if short_results:
            notes.append(
                f"{short_results} of {counted} queries returned fewer than {k} rows; "
                "the index ran out of candidates before the limit."
            )
        used = self._plan_uses_vector_index(
            self.vector_sql(vectors[0], k, predicate=predicate), settings=settings
        )
        return RecallResult(
            label=label,
            k=k,
            sample=counted,
            recall=hits / counted,
            timing=Timing.of(latencies),
            settings=list(settings),
            used_vector_index=used,
            exact_ground_truth=exact,
            notes=notes,
        )

    def sweep(
        self,
        vectors: Sequence[str],
        truth: Sequence[Sequence[Any]],
        k: int,
        gucs: dict[str, str],
    ) -> list[SweepRow]:
        """Recall and latency across the index's search-effort setting.

        The point is not to pick a value on the user's behalf. Latency and recall trade
        against each other and only the person running the query knows which side of
        that trade their product is on.
        """
        index = self._active_vector_index()
        if index is None or not truth:
            return []
        if index.method == "hnsw":
            guc = "hnsw.ef_search"
            current = _as_int(gucs.get(guc), HNSW_DEFAULT_EF_SEARCH)
            default = HNSW_DEFAULT_EF_SEARCH
            values = _sweep_values(
                [k, k * 2, k * 4, 40, 100, 400, current], low=max(k, 1), high=1000
            )
        else:
            guc = "ivfflat.probes"
            current = _as_int(gucs.get(guc), IVFFLAT_DEFAULT_PROBES)
            default = IVFFLAT_DEFAULT_PROBES
            lists = index.lists or 100
            # The top of the range is probes = lists, which scans every list and is
            # therefore exact. Including it gives the table an anchor: it shows what
            # the index costs when it is not allowed to approximate at all.
            values = _sweep_values(
                [1, 2, ivfflat_probes(lists), lists // 4, lists // 2, lists, current],
                low=1,
                high=max(lists, 1),
            )

        rows = []
        for value in values:
            result = self.measure(
                vectors, truth, k, label=f"{guc}={value}", settings=[f"{guc} = {value}"]
            )
            if result is None:
                continue
            rows.append(
                SweepRow(
                    setting=guc,
                    value=value,
                    recall=result.recall,
                    timing=result.timing,
                    is_current=value == current,
                    is_default=value == default,
                )
            )
        return rows

    # -- plans --------------------------------------------------------------

    def explain(
        self,
        sql: str,
        params: Optional[Sequence[Any]] = None,
        settings: Sequence[str] = (),
    ) -> Optional[dict[str, Any]]:
        """EXPLAIN without ANALYZE: it costs nothing and executes nothing."""
        try:
            with self.session(settings=settings):
                return self.explain_here(sql, params)
        except Exception as exc:
            self.errors.append(f"EXPLAIN failed: {_short(exc)}")
            return None

    def explain_here(
        self, sql: str, params: Optional[Sequence[Any]] = None
    ) -> Optional[dict[str, Any]]:
        """EXPLAIN inside the session that is already open, settings and all."""
        rows = self.run("EXPLAIN (FORMAT JSON) " + sql, params)
        return _explain_plan(rows[0][0]) if rows else None

    def uses_vector_index(self, plan: Optional[dict[str, Any]]) -> Optional[bool]:
        if plan is None:
            return None
        names = {i.name for i in self.info.indexes if i.is_vector}
        return any(node.get("Index Name") in names for node in _walk(plan))

    def _plan_uses_vector_index(self, sql: str, settings: Sequence[str] = ()) -> Optional[bool]:
        return self.uses_vector_index(self.explain(sql, settings=settings))

    def _active_vector_index(self) -> Optional[IndexInfo]:
        """The vector index the planner actually chose, not the one we hoped for.

        A table with both an HNSW and an ivfflat index has two different knobs, and
        sweeping the one the planner is ignoring produces a flat, meaningless table.
        """
        candidates = [i for i in self.info.indexes if i.is_vector and i.valid]
        if not candidates:
            return None
        vector = self.one_vector()
        if vector:
            plan = self.explain(self.vector_sql(vector, 10))
            if plan is not None:
                used = {node.get("Index Name") for node in _walk(plan)}
                for index in candidates:
                    if index.name in used:
                        return index
        return candidates[0]

    def plan_probes(self, filters: Sequence[FilterProbe]) -> list[PlanProbe]:
        """EXPLAIN the query shapes the application really runs."""
        probes: list[PlanProbe] = []
        vector = self.one_vector()
        if not vector:
            return probes

        probes.append(self._plan_probe("vector search", self.vector_sql(vector, 10)))
        for probe in filters:
            probes.append(
                self._plan_probe(
                    f"vector search + {probe.predicate}",
                    self.vector_sql(vector, 10, predicate=f" AND {probe.predicate}"),
                    filter_text=probe.predicate,
                )
            )
        hybrid = self._hybrid_sql()
        if hybrid is not None:
            probes.append(self._plan_probe("hybrid search (the query pghybrid emits)", *hybrid))
        return probes

    def _plan_probe(
        self,
        label: str,
        sql: str,
        params: Optional[Sequence[Any]] = None,
        filter_text: Optional[str] = None,
    ) -> PlanProbe:
        plan = self.explain(sql, params)
        if plan is None:
            return PlanProbe(
                label=label,
                sql=sql,
                scan="unknown",
                used_vector_index=False,
                filter=filter_text,
                error=self.errors[-1] if self.errors else "EXPLAIN produced no plan",
            )
        names = {i.name for i in self.info.indexes if i.is_vector}
        used = False
        index_name = None
        scans: list[str] = []
        for node in _walk(plan):
            node_type = str(node.get("Node Type", ""))
            if node.get("Index Name") in names:
                used = True
                index_name = str(node.get("Index Name"))
            if not node_type.endswith("Scan"):
                continue
            if node.get("Relation Name") != self.info.name and not node.get("Index Name"):
                continue
            description = node_type
            if node.get("Index Name"):
                description += f" using {node['Index Name']}"
            # The hybrid statement plans each candidate CTE separately, so one scan node
            # is not the answer: the vector half can be perfect while the text half
            # reads the whole table.
            if description not in scans:
                scans.append(description)
        scan = " + ".join(scans) if scans else "unknown"
        return PlanProbe(
            label=label,
            sql=sql,
            scan=scan,
            used_vector_index=used,
            index_name=index_name,
            estimated_rows=float(plan.get("Plan Rows", 0) or 0),
            filter=filter_text,
        )

    def _hybrid_sql(self) -> Optional[tuple[str, list[Any]]]:
        """The statement the library itself generates, so the plan probe is honest.

        This is the shape that matters: the two candidate CTEs are planned separately,
        and the text half can be the one doing a sequential scan while the vector half
        looks perfect.
        """
        dimensions = self._dimensions()
        if not dimensions:
            return None
        embedding = [0.0] * dimensions
        embedding[0] = 1.0
        try:
            return build_search_sql(
                self.config, embedding=embedding, text="the quick brown fox", limit=10
            )
        except ValueError as exc:
            self.errors.append(f"could not build the hybrid statement: {_short(exc)}")
            return None

    def _dimensions(self) -> int:
        column = self.info.column(self.config.vector_column)
        return column.dimensions if column and column.dimensions else 0

    # -- filters ------------------------------------------------------------

    def filter_probes(self, filters: Optional[dict[str, Any]]) -> list[FilterProbe]:
        """Pick one realistic value per filter column, preferring the common case.

        The most common value is deliberate: the biggest tenant is where filtered
        search is slowest and where a filtered recall collapse hurts the most people.
        """
        probes: list[FilterProbe] = []
        if filters:
            for column, value in filters.items():
                literal = _literal_value(value)
                if literal is None:
                    continue
                probes.append(
                    FilterProbe(
                        column=column,
                        value=value,
                        literal=literal,
                        selectivity=self._selectivity(column, literal),
                        source="caller",
                    )
                )
            return probes

        for column in self.config.filter_columns[:2]:
            info_column = self.info.column(column)
            if info_column is None:
                continue
            value, source = self._common_value(column)
            if value is None:
                continue
            value = _coerce(value, info_column.type_name)
            literal = _literal_value(value)
            if literal is None:
                continue
            probes.append(
                FilterProbe(
                    column=column,
                    value=value,
                    literal=literal,
                    selectivity=self._selectivity(column, literal),
                    source=source,
                )
            )
        return probes

    def _common_value(self, column: str) -> tuple[Any, str]:
        quoted = quote_ident(column)
        try:
            with self.session():
                stats = self.run(
                    "SELECT (most_common_vals::text::text[])[1] FROM pg_catalog.pg_stats "
                    f"WHERE schemaname = {_literal(self.info.schema)} "
                    f"  AND tablename = {_literal(self.info.name)} "
                    f"  AND attname = {_literal(column)}"
                )
                if stats and stats[0][0] is not None:
                    return stats[0][0], "pg_stats most common value"
                rows = self.run(
                    f"SELECT {quoted} FROM {self.table} WHERE {quoted} IS NOT NULL LIMIT 1"
                )
                if rows:
                    return rows[0][0], "first non-null value"
        except Exception as exc:
            self.errors.append(f"could not pick a filter value for {column}: {_short(exc)}")
        return None, ""

    def _selectivity(self, column: str, literal: str) -> Optional[float]:
        """The planner's own belief about how much of the table the filter keeps.

        Its belief is what matters here, because it is what decides whether the vector
        index gets used -- not the true fraction.
        """
        if self.info.row_count <= 0:
            return None
        plan = self.explain(f"SELECT 1 FROM {self.table} WHERE {quote_ident(column)} = {literal}")
        if plan is None:
            return None
        return min(float(plan.get("Plan Rows", 0) or 0) / self.info.row_count, 1.0)


# ------------------------------------------------------------------------- findings


def _check_inventory(report: DoctorReport, probe: _Prober) -> None:
    """Everything that can be judged from the catalog alone, before any measurement."""
    config, info = report.config, report.info
    add = report.findings.append

    column = info.column(config.vector_column)
    if column is None:
        add(
            Finding(
                "error",
                f"{config.vector_column} does not exist",
                f"The config searches {config.vector_column}, which is not a column of "
                f"{info.qualified}. Columns: "
                f"{', '.join(c.name for c in info.columns) or 'none'}.",
            )
        )
    elif column.dimensions is None:
        add(
            Finding(
                "error",
                f"{column.name} has no declared dimension",
                f"{column.name} is bare {column.type_name}. Values store fine, but "
                "pgvector cannot build an index on a column of unknown width, so every "
                "search reads the whole table.",
                fix=f"ALTER TABLE {quote_ident(config.table)} ALTER COLUMN "
                f"{quote_ident(column.name)} TYPE {column.type_name}(<dimensions>);",
            )
        )

    if config.tsvector_column and info.column(config.tsvector_column) is None:
        add(
            Finding(
                "error",
                f"{config.tsvector_column} does not exist yet",
                "The config names a stored tsvector column that has not been created. "
                "Every search against this config fails until the migration runs.",
                fix="pghybrid migrate",
            )
        )

    ts_column = info.column(config.tsvector_column) if config.tsvector_column else None
    if ts_column is not None and not ts_column.generated:
        add(
            Finding(
                "warn",
                f"{ts_column.name} is not a generated column",
                "It is maintained by a trigger or by the application, so nothing "
                "guarantees it matches the text column. Rows whose tsvector was never "
                "written are invisible to the keyword half of every search.",
            )
        )

    vector_index = info.vector_index_for(config.vector_column)
    if vector_index is None:
        add(
            Finding(
                "warn" if info.row_count < 10_000 else "error",
                "no vector index",
                f"{info.qualified} has no HNSW or ivfflat index on "
                f"{config.vector_column}, so every vector search is exact and reads "
                f"all {info.row_count:,} rows. Recall is 1.0 by construction and "
                "latency grows linearly with the table.",
                fix=f"CREATE INDEX ON {quote_ident(config.table)} USING hnsw "
                f"({quote_ident(config.vector_column)} {config.ops_class});",
            )
        )
    else:
        if not vector_index.valid:
            add(
                Finding(
                    "error",
                    f"{vector_index.name} is INVALID",
                    "A CREATE INDEX CONCURRENTLY that failed leaves the index behind in "
                    "an invalid state. It takes up space, is maintained on write, and is "
                    "never used.",
                    fix=f"REINDEX INDEX CONCURRENTLY {quote_ident(vector_index.name)};",
                )
            )
        if vector_index.metric is not None and vector_index.metric is not config.metric:
            add(
                Finding(
                    "error",
                    "index operator class does not match the configured metric",
                    f"{vector_index.name} is built with "
                    f"{vector_index.opclasses[0] if vector_index.opclasses else '?'} but "
                    f"the config searches with {config.metric.name} "
                    f"({config.metric.operator}). Postgres does not report this: it just "
                    "never uses the index, and every search becomes a sequential scan.",
                    fix=f"Either set metric='{vector_index.metric.name}' in the config, "
                    f"or rebuild the index with {config.ops_class}.",
                )
            )

    inline = _inline_expression(config)
    if info.text_index_for(config.tsvector_column, inline) is None:
        add(
            Finding(
                "warn",
                "no GIN index for the text half",
                "The keyword half of each hybrid query scans the table and re-parses "
                "every row. On most tables this, not the vector search, is the slow half.",
                fix=f"CREATE INDEX ON {quote_ident(config.table)} USING gin "
                f"({quote_ident(config.tsvector_column) if config.tsvector_column else inline});",
            )
        )

    if not info.analyzed:
        add(
            Finding(
                "warn",
                "this table has never been analysed",
                "Without statistics the planner estimates row counts from built-in "
                "constants, which is the most common reason a filtered vector query "
                "abandons the index.",
                fix=f"ANALYZE {quote_ident(config.table)};",
            )
        )

    if (
        info.pgvector_version
        and info.pgvector_available_version
        and parse_version(info.pgvector_available_version) > parse_version(info.pgvector_version)
    ):
        add(
            Finding(
                "info",
                f"pgvector {info.pgvector_available_version} is available "
                f"(this database runs {info.pgvector_version})",
                "0.8 added iterative index scans, which is the fix for recall "
                "collapsing under selective filters.",
                fix="ALTER EXTENSION vector UPDATE;",
            )
        )

    if info.is_partitioned:
        add(
            Finding(
                "info",
                "this is a partitioned table",
                "Index DDL on the parent creates a partitioned index and builds one per "
                "partition; CREATE INDEX CONCURRENTLY is not supported on the parent, so "
                "each partition has to be indexed separately to avoid a long lock.",
            )
        )

    for message in probe.errors:
        add(Finding("warn", "probe did not complete", message))
    probe.errors.clear()


def _check_recall(report: DoctorReport) -> None:
    result = report.recall
    if result is None:
        return
    index = report.info.vector_index_for(report.config.vector_column)
    if index is None:
        report.findings.append(
            Finding(
                "info",
                f"recall@{result.k} is 1.00 because there is no index",
                "Every search is exact today. The number to watch after building an "
                "index is this same measurement, which is why it is worth recording now.",
            )
        )
        return
    knob = "hnsw.ef_search" if index.method == "hnsw" else "ivfflat.probes"
    missed = (1 - result.recall) * result.k
    if result.recall < BAD_RECALL:
        level = "error"
        detail = (
            f"About {missed:.1f} of every {result.k} results are not among the true "
            "nearest neighbours. At this level the wrong answer is often the one on "
            "screen, and no amount of prompt engineering downstream recovers it."
        )
    elif result.recall < POOR_RECALL:
        level = "warn"
        detail = (
            f"About {missed:.1f} of every {result.k} results are not among the true "
            "nearest neighbours. That is the gap users describe as 'search sometimes "
            "misses the obvious one'."
        )
    else:
        level = "ok"
        detail = (
            f"The index returns the true nearest neighbours for these queries, so "
            f"nothing is being lost at {knob} = "
            f"{next((r.value for r in report.sweep if r.is_current), 'the current setting')}."
        )
    report.findings.append(
        Finding(
            level,
            f"recall@{result.k} = {result.recall:.2f} over {result.sample} sampled queries",
            detail,
            fix=None
            if level == "ok"
            else f"Raise {knob}; the sweep shows what each value costs in latency.",
        )
    )
    current = next((r.value for r in report.sweep if r.is_current), None)
    if current is not None and current < result.k:
        # pgvector explores ef_search candidates and returns the best k of them, so an
        # ef_search below k cannot fill the result set however good the graph is.
        report.findings.append(
            Finding(
                "error",
                f"{knob} = {current} is below k = {result.k}",
                f"The index is asked for {result.k} rows but only allowed to consider "
                f"{current} candidates, so it can return fewer results than requested "
                "and the ones it returns are the shallowest part of the graph.",
                fix=f"SET {knob} = {max(result.k * 2, current)};",
            )
        )
    for note in result.notes:
        report.findings.append(
            Finding("warn", f"the index ran short of candidates at k = {result.k}", note)
        )
    if report.null_fraction and report.null_fraction > 0.01:
        report.findings.append(
            Finding(
                "warn",
                f"{report.null_fraction:.1%} of rows have no embedding",
                "Those rows can never be returned by the vector half of the search. If "
                "they are supposed to be searchable, the backfill is unfinished.",
            )
        )


def _measure_filtered(
    report: DoctorReport,
    probe: _Prober,
    vectors: Sequence[str],
    k: int,
    exact_ok: bool,
    gucs: dict[str, str],
) -> None:
    """Filtered recall, which is where most multi-tenant applications actually live.

    ANN recall under a selective filter is a different number from unfiltered recall,
    usually a much worse one, and no benchmark anywhere reports it.
    """
    if not report.filters:
        return
    index = report.info.vector_index_for(report.config.vector_column)
    supports_iterative = report.info.supports_iterative_scan and bool(gucs)
    iterative_guc = None
    if index is not None:
        candidate = f"{index.method}.iterative_scan"
        if candidate in gucs:
            iterative_guc = candidate

    worst: Optional[RecallResult] = None
    for filter_probe in report.filters:
        predicate = f" AND {filter_probe.predicate}"
        truth, _, note = probe.ground_truth(vectors, k, predicate=predicate)
        if not truth:
            if note is not None:
                report.findings.append(note)
            continue
        result = probe.measure(
            vectors,
            truth,
            k,
            label=f"filtered on {filter_probe.predicate}",
            predicate=predicate,
            exact=exact_ok,
        )
        if result is None:
            continue
        report.filtered.append(result)
        if worst is None or result.recall < worst.recall:
            worst = result
        if result.used_vector_index is False:
            selectivity = (
                f" (the planner expects it to keep {filter_probe.selectivity:.1%} of the table)"
                if filter_probe.selectivity is not None
                else ""
            )
            report.findings.append(
                Finding(
                    "warn",
                    f"the vector index is not used when filtering on {filter_probe.column}",
                    f"{filter_probe.predicate}{selectivity} makes a sequential scan look "
                    "cheaper than the index. Results stay exact; latency grows with the "
                    "table.",
                    fix=f"CREATE INDEX ON {quote_ident(report.config.table)} "
                    f"({quote_ident(filter_probe.column)});  -- or a partial vector index "
                    f"WHERE {filter_probe.predicate}",
                )
            )
        else:
            # The comparison that matters is against unfiltered recall on the same
            # index, not against 1.0. A filter that leaves recall where it already was
            # is not the problem; reporting it as one buries the filter that is.
            baseline_recall = report.recall.recall if report.recall else 1.0
            lost = baseline_recall - result.recall
            if lost > 0.05 and result.recall < POOR_RECALL:
                selectivity = (
                    f" keeps {filter_probe.selectivity:.1%} of the table and"
                    if filter_probe.selectivity is not None
                    else ""
                )
                report.findings.append(
                    Finding(
                        "error" if result.recall < BAD_RECALL else "warn",
                        f"filtering on {filter_probe.predicate} drops recall@{k} from "
                        f"{baseline_recall:.2f} to {result.recall:.2f}",
                        f"{filter_probe.predicate}{selectivity} costs {lost:.2f} of "
                        "recall. The index searches first and the filter is applied to "
                        "whatever it returned, so a selective filter throws most of the "
                        "candidates away before they can be results.",
                        fix=f"SET {index.method if index else 'hnsw'}.iterative_scan = "
                        "relaxed_order;"
                        if supports_iterative
                        else "Upgrade to pgvector 0.8 for iterative scans, or build a "
                        f"partial index WHERE {filter_probe.predicate}.",
                    )
                )

    measured = None
    baseline = worst
    unfiltered = report.recall.recall if report.recall else 1.0
    if (
        iterative_guc is not None
        and worst is not None
        and worst.used_vector_index
        and unfiltered - worst.recall > 0.05
    ):
        # Measure the fix rather than recommending it on faith: the before-and-after is
        # the single most useful number this report produces for a multi-tenant table.
        target = next((f for f in report.filters if f.predicate in worst.label), report.filters[0])
        predicate = f" AND {target.predicate}"
        truth, _, _ = probe.ground_truth(vectors, k, predicate=predicate)
        measured = probe.measure(
            vectors,
            truth,
            k,
            label=f"{target.predicate} with {iterative_guc} = relaxed_order",
            predicate=predicate,
            settings=[f"{iterative_guc} = relaxed_order"],
        )
        if measured is not None:
            report.findings.append(
                Finding(
                    "info" if measured.recall > worst.recall else "warn",
                    f"iterative scans move filtered recall@{k} from "
                    f"{worst.recall:.2f} to {measured.recall:.2f}",
                    "pgvector 0.8 keeps searching the index until it has enough rows "
                    "that pass the filter, instead of filtering whatever the first pass "
                    "returned. relaxed_order is faster and may return rows slightly out "
                    "of distance order; strict_order preserves the order and costs more.",
                    fix=f"SET {iterative_guc} = relaxed_order;  -- and cap the work with "
                    f"{iterative_guc.split('.')[0]}.max_scan_tuples",
                )
            )

    report.iterative = IterativeScan(
        supported=supports_iterative and iterative_guc is not None,
        version=report.info.pgvector_version,
        settings={n: v for n, v in gucs.items() if "iterative" in n or "max_scan" in n},
        measured=measured,
        baseline=baseline,
    )


def _recommendations(report: DoctorReport) -> list[Statement]:
    try:
        return build_migration(report.config, report.info)
    except ValueError:
        # A config that does not describe this table is already reported as an error
        # finding; failing the whole report on top of that helps nobody.
        return []


# ------------------------------------------------------------------------ rendering


def _render_report(report: DoctorReport, *, width: int = 78) -> str:
    rule = "=" * width
    thin = "-" * width
    out: list[str] = [f"pghybrid doctor   {report.info.qualified}", rule, ""]

    if report.recall is not None:
        out.append(f"  recall@{report.recall.k}  {report.recall.recall:.2f}")
        detail = f"{report.recall.sample} query vectors sampled from the table"
        if report.recall.settings:
            detail += ", " + ", ".join(report.recall.settings)
        out.append(f"            {detail}")
        if report.sweep:
            current = next((r for r in report.sweep if r.is_current), None)
            if current is not None:
                out.append(
                    f"            at {current.setting} = {current.value}"
                    + (" (pgvector's default)" if current.is_default else "")
                )
        # Said out loud because it is the one way to over-read this number: a query
        # vector taken from the table sits exactly on a data point, which is the
        # easiest possible query for any index. Real queries score at or below this.
        out.append(
            "            read as a ceiling: queries drawn from the table are easier than real ones"
        )
    else:
        out.append("  recall    not measured")
    out += ["", "INVENTORY", thin]
    out += ["  " + line for line in report.info.to_text().splitlines()]
    if report.null_fraction is not None:
        out.append(
            f"  {report.config.vector_column} NULL  {report.null_fraction:.2%} of rows"
            f"   (from {report.null_fraction_source})"
        )
    if report.iterative is not None and report.iterative.settings:
        settings = ", ".join(f"{n} = {v}" for n, v in sorted(report.iterative.settings.items()))
        out.append(f"  runtime          {settings}")

    if report.sweep:
        out += ["", f"SWEEP  recall@{report.k} and latency at each setting", thin]
        name = report.sweep[0].setting
        out.append(f"  {name:<22}{'recall':>8}   {'p50':>9}   {'p95':>9}")
        for row in report.sweep:
            tag = ""
            if row.is_current and row.is_default:
                tag = "  <- current, pgvector's default"
            elif row.is_current:
                tag = "  <- current"
            elif row.is_default:
                tag = "  <- pgvector's default"
            out.append(
                f"  {row.value:<22}{row.recall:>8.2f}   {row.timing.p50_ms:>6.2f} ms   "
                f"{row.timing.p95_ms:>6.2f} ms{tag}"
            )
        out.append("  Latency is measured client-side, so it includes one round trip per query.")

    if report.filtered:
        heading = "FILTERED RECALL"
        if report.recall is not None:
            heading += f"  (unfiltered recall@{report.k} was {report.recall.recall:.2f})"
        out += ["", heading, thin]
        rows: list[tuple[str, RecallResult, str]] = [
            (
                result.label,
                result,
                "index used"
                if result.used_vector_index
                else "index NOT used, results exact"
                if result.used_vector_index is False
                else "plan unknown",
            )
            for result in report.filtered
        ]
        if report.iterative is not None and report.iterative.measured is not None:
            measured = report.iterative.measured
            rows.append((measured.label, measured, "pgvector 0.8 iterative scan"))
        label_width = min(max(len(label) for label, _, _ in rows), width - 34)
        for label, result, used in rows:
            out.append(
                f"  {label.ljust(label_width)}  {result.recall:>5.2f}   "
                f"{result.timing.p50_ms:>6.2f} ms   {used}"
            )
        out.append("")
        for probe in report.filters:
            selectivity = (
                f"{probe.selectivity:.1%} of rows" if probe.selectivity is not None else "unknown"
            )
            out.append(f"  {probe.predicate}  keeps {selectivity}  ({probe.source})")

    if report.plans:
        out += ["", "PLANS", thin]
        for plan in report.plans:
            out.append(f"  {plan.label}")
            body = plan.error or plan.scan
            for line in _wrap(body, width - 6):
                out.append(f"      {line}")
            if not plan.used_vector_index and not plan.error:
                out.append("      ^ the vector index is not used in this shape")

    if report.findings:
        out += ["", "FINDINGS", thin]
        for finding in report.findings:
            out.append(f"  [{finding.level}] {finding.title}")
            for line in _wrap(finding.detail, width - 8):
                out.append(f"        {line}")
            if finding.fix:
                for position, line in enumerate(_wrap(finding.fix, width - 13)):
                    out.append(("        fix: " if position == 0 else "             ") + line)
            out.append("")

    required = [s for s in report.recommendations if not s.optional]
    optional = [s for s in report.recommendations if s.optional]
    if required or optional:
        out += ["RECOMMENDED STATEMENTS", thin]
        for statement in required:
            out += ["  " + line for line in statement.to_text().splitlines()]
            out.append("")
        if optional:
            out.append("  Options and alternatives. None of these are required.")
            out.append("")
            for statement in optional:
                out += ["  " + line for line in statement.to_text().splitlines()]
                out.append("")

    out.append(
        f"read-only: {'yes' if report.read_only else 'no (ANALYZE was allowed)'}   "
        f"probes finished in {report.duration_ms / 1000:.2f}s"
    )
    if not report.session_settings_shared:
        out.append(
            "note: SET LOCAL did not carry between statements on this connection, so "
            "ground truth was computed with a non-indexable ORDER BY instead."
        )
    return "\n".join(out)


def _wrap(text: str, width: int) -> list[str]:
    """Wrap without importing textwrap's whole machinery for two lines of output."""
    words = str(text).split()
    lines: list[str] = []
    current = ""
    for word in words:
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines or [""]


# --------------------------------------------------------------------------- helpers


def _percentile(ordered: Sequence[float], q: float) -> float:
    """Nearest-rank percentile.

    Nearest-rank rather than interpolation: at these sample sizes the difference is
    smaller than the run-to-run noise, and nearest-rank only ever reports a latency
    that was actually observed.
    """
    if not ordered:
        return 0.0
    index = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def _sweep_values(candidates: Sequence[int], *, low: int, high: int) -> list[int]:
    values = sorted({int(v) for v in candidates if low <= int(v) <= high})
    if not values:
        values = [low]
    if len(values) > 7:
        # Keep the ends and thin the middle: the shape of the curve is the point, and
        # eight rows of a table nobody reads is not.
        step = len(values) / 7.0
        values = [values[int(i * step)] for i in range(7)]
    return values


def _as_int(value: Optional[str], fallback: int) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return fallback


def _vector_literal(value: Any) -> Optional[str]:
    """A pgvector value from any driver, back into a SQL literal.

    Drivers disagree: psycopg hands back the text form, an adapter-registered
    connection hands back a list, asyncpg hands back whatever was registered.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return "'[" + ",".join(repr(float(v)) for v in value) + "]'"
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", "ignore")
    text = str(value).strip()
    if not text.startswith("["):
        return None
    if "'" in text or "\\" in text:
        # The text form of a vector is digits, commas, brackets and signs. Anything
        # else did not come from pgvector and is not going into a statement.
        return None
    return f"'{text}'"


#: Types whose values read wrong when they come back as quoted text.
_NUMERIC_TYPES = frozenset({"int2", "int4", "int8", "float4", "float8", "numeric", "oid"})


def _coerce(value: Any, type_name: str) -> Any:
    """Turn a filter value read as text back into a number where the column is numeric.

    pg_stats returns most-common values as text whatever the column type. Postgres will
    coerce ``tenant_id = '4'`` happily, but the report prints these predicates and
    ``tenant_id = '4'`` reads like a bug in the tool rather than a description of the
    query someone actually runs.
    """
    if not isinstance(value, str) or type_name not in _NUMERIC_TYPES:
        return value
    try:
        return int(value) if type_name in ("int2", "int4", "int8", "oid") else float(value)
    except ValueError:
        return value


def _literal_value(value: Any) -> Optional[str]:
    """A filter value as a SQL literal.

    Everything scalar is rendered as an unknown-typed quoted literal and left for
    Postgres to coerce to the column's type, which is what makes one code path work for
    integers, uuids, enums and timestamps alike.
    """
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (bytes, bytearray, memoryview, list, tuple, dict, set)):
        return None
    return _literal(str(value))


def _literal(value: str) -> str:
    if "\x00" in value:
        raise ValueError("SQL string literals cannot contain a null byte")
    return "'" + value.replace("'", "''") + "'"


def _inline_expression(config: Config) -> str:
    """The tsvector expression the query builder computes when there is no column."""
    return f"to_tsvector('{config.language}', coalesce({quote_ident(config.text_column)}, ''))"


def _explain_plan(payload: Any) -> Optional[dict[str, Any]]:
    if isinstance(payload, (str, bytes, bytearray)):
        payload = json.loads(payload)
    if isinstance(payload, list) and payload:
        payload = payload[0]
    if isinstance(payload, dict):
        plan = payload.get("Plan")
        if isinstance(plan, dict):
            return plan
    return None


def _walk(node: dict[str, Any]) -> Iterator[dict[str, Any]]:
    yield node
    for child in node.get("Plans", []) or []:
        if isinstance(child, dict):
            yield from _walk(child)


def _short(exc: Exception) -> str:
    """One line of an exception, because a driver traceback in a report is noise."""
    text = re.sub(r"\s+", " ", str(exc)).strip()
    return text[:200] if text else exc.__class__.__name__
