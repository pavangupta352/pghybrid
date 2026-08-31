/**
 * SQL generation for hybrid search on plain Postgres.
 *
 * Everything in this module is a pure function of a {@link Config} and the call
 * arguments. Nothing here touches a database, which is what makes the generated SQL
 * auditable, snapshot-testable, and copy-pasteable by people who never install the
 * package.
 *
 * The generated statement is one query with two candidate CTEs — one per signal —
 * fused by Reciprocal Rank Fusion. Filters are applied *inside* each CTE so that both
 * signals search the same subset of rows; applying them after the fusion silently
 * destroys recall, which is the single most common way a hand-rolled implementation
 * goes wrong.
 *
 * It has a twin: the Python package generates the same statement byte for byte, and
 * `scripts/check_parity.mjs` fails the build when the two disagree. Any change here
 * that alters the emitted text has to be made there too.
 */

import type { Config, FusionMethod, ParamStyle, Recency, ResolvedConfig } from "./config.js";
import { resolveConfig } from "./config.js";
import { parseQuery } from "./textquery.js";

/**
 * Postgres identifiers we are willing to interpolate. Anything outside this set is
 * rejected rather than escaped, because a column name that needs escaping is far more
 * likely to be a mistake (or an injection attempt) than a deliberate choice.
 */
const IDENT_RE = /^[A-Za-z_][A-Za-z0-9_$]*$/;

/** ln(2), for converting a half-life into an exponential decay rate. */
const LN2 = "0.6931471805599453";

/** Raised when a table or column name cannot be safely interpolated. */
export class IdentifierError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "IdentifierError";
  }
}

/**
 * Validate and double-quote a Postgres identifier.
 *
 * Qualified names (`schema.table`) are supported and each part is validated
 * separately, so `public.chunks` becomes `"public"."chunks"`.
 */
export function quoteIdent(name: string): string {
  if (typeof name !== "string" || !name) {
    throw new IdentifierError(
      `identifier must be a non-empty string, got ${JSON.stringify(name) ?? String(name)}`,
    );
  }

  const parts = name.split(".");
  if (parts.length > 2) {
    throw new IdentifierError(
      `${JSON.stringify(name)} has too many parts; expected 'name' or 'schema.name'`,
    );
  }

  return parts
    .map((part) => {
      if (!IDENT_RE.test(part)) {
        throw new IdentifierError(
          `${JSON.stringify(part)} is not a valid Postgres identifier. Use letters, ` +
            "digits and underscores, starting with a letter or underscore.",
        );
      }
      return `"${part}"`;
    })
    .join(".");
}

/**
 * Internal sentinel wrapping a parameter slot while the statement is assembled.
 * Placeholders cannot be written directly during assembly because the right text
 * depends on the driver, and because one logical value may appear in the statement
 * more than once.
 */
// A control character is the point: it cannot occur in a statement this builder
// assembles, so a token can never collide with SQL text or with a bound value.
// eslint-disable-next-line no-control-regex
const TOKEN_RE = /\x01p(\d+)\x01/g;

function token(index: number): string {
  return `\x01p${index}\x01`;
}

/** One rendered statement and the values to bind to it, in order. */
export interface BuiltQuery {
  sql: string;
  params: unknown[];
}

/**
 * Accumulates bind parameters and renders them in the driver's placeholder style.
 *
 * Every value that originates outside the config — the query text, the embedding,
 * limits, filter values — goes through here, so the generated SQL never contains an
 * interpolated literal.
 *
 * Placeholders are emitted as opaque tokens during assembly and resolved in
 * {@link Params.render}. That indirection exists because `$1` may be referenced twice
 * in one statement while `%s` may not: numbered styles deduplicate, positional styles
 * have to repeat the value. Writing the placeholder text at the point of use would
 * force the builder to know which of those it is emitting, in every branch.
 */
export class Params {
  private readonly slots: unknown[] = [];

  add(value: unknown): string {
    this.slots.push(value);
    return token(this.slots.length - 1);
  }

  addCast(value: unknown, cast: string): string {
    return `${this.add(value)}::${cast}`;
  }

  /** Substitute placeholders and return the statement with its final values. */
  render(sql: string, paramStyle: ParamStyle): BuiltQuery {
    if (paramStyle === "numeric") {
      const numbers = new Map<number, number>();
      const values: unknown[] = [];
      const rendered = sql.replace(TOKEN_RE, (_match, slot: string) => {
        const index = Number(slot);
        let number = numbers.get(index);
        if (number === undefined) {
          values.push(this.slots[index]);
          number = values.length;
          numbers.set(index, number);
        }
        return `$${number}`;
      });
      return { sql: rendered, params: values };
    }

    if (paramStyle === "pyformat") {
      const values: unknown[] = [];
      // A literal percent would otherwise be read as the start of a placeholder by
      // drivers that use pyformat.
      const escaped = sql.replace(/%/g, "%%");
      const rendered = escaped.replace(TOKEN_RE, (_match, slot: string) => {
        values.push(this.slots[Number(slot)]);
        return "%s";
      });
      return { sql: rendered, params: values };
    }

    throw new Error(
      `unknown paramStyle ${JSON.stringify(paramStyle)}; expected 'numeric' ` +
        "(node-postgres, asyncpg, raw SQL) or 'pyformat' (psycopg)",
    );
  }
}

/** The distance operator for the configured metric, applied to the vector column. */
function distanceExpr(cfg: ResolvedConfig, vecPlaceholder: string): string {
  const col = quoteIdent(cfg.vectorColumn);
  const op = cfg.metric.operator;
  // halfvec columns compare against a halfvec-cast query vector; this halves index
  // size and build time and is usually free in recall terms.
  if (cfg.vectorType === "halfvec") {
    return `${col} ${op} ${vecPlaceholder}::halfvec`;
  }
  return `${col} ${op} ${vecPlaceholder}`;
}

/**
 * The searchable tsvector: a stored column when configured, computed otherwise.
 *
 * Computing it inline means the package works against an untouched table with no
 * migration at all — slower, but it lets someone try the library before changing their
 * schema. The migration generator emits the stored form.
 */
function tsvectorExpr(cfg: ResolvedConfig): string {
  if (cfg.tsvectorColumn) {
    return quoteIdent(cfg.tsvectorColumn);
  }
  // NOTE: the two-argument form is required. to_tsvector(text) is STABLE, not
  // IMMUTABLE, because it reads default_text_search_config.
  return `to_tsvector('${cfg.language}', coalesce(${quoteIdent(cfg.textColumn)}, ''))`;
}

/**
 * Build the tsquery expression for the configured match mode.
 *
 * `all` hands the whole string to one parser, giving AND semantics. `any` combines one
 * parser call per term with `||`, so the keyword signal still produces a ranked
 * candidate list when no single document contains every word.
 */
function tsqueryExpr(cfg: ResolvedConfig, text: string, params: Params): string {
  const call = (value: string): string =>
    `${cfg.queryParser}('${cfg.language}', ${params.add(value)})`;

  if (cfg.textMatch === "all") {
    return call(text);
  }

  const parsed = parseQuery(text);
  if (parsed.positive.length === 0) {
    // Nothing to OR — a query of only exclusions, or only noise. Fall back to the
    // parser's own reading of the string rather than inventing a match-everything
    // query, which would flood the fusion with irrelevant candidates.
    return call(text);
  }

  let expression = parsed.positive.map(call).join(" || ");
  if (parsed.positive.length > 1) {
    expression = `(${expression})`;
  }
  for (const term of parsed.negative) {
    expression = `${expression} && !!${call(term)}`;
  }
  return expression;
}

/** Anything a caller may filter on. Arrays and Sets become `= ANY($n)`. */
export type Filters = Record<string, unknown>;

/** Render the caller's filters as an AND-chain scoped to one candidate CTE. */
function filterSql(cfg: ResolvedConfig, filters: Filters | null | undefined, params: Params): string {
  if (!filters) {
    return "";
  }
  const entries = Object.entries(filters);
  if (entries.length === 0) {
    return "";
  }
  if (cfg.filterColumns.length === 0) {
    throw new Error(
      "filters were passed but Config.filterColumns is empty. List the columns you " +
        "intend to filter on so they can be validated and indexed.",
    );
  }

  const clauses: string[] = [];
  for (const [column, value] of entries) {
    if (!cfg.filterColumns.includes(column)) {
      throw new Error(
        `${JSON.stringify(column)} is not in Config.filterColumns ` +
          `(${cfg.filterColumns.join(", ") || "none"}).`,
      );
    }
    const col = quoteIdent(column);
    if (value === null || value === undefined) {
      clauses.push(`${col} IS NULL`);
    } else if (Array.isArray(value) || value instanceof Set) {
      const values = [...(value as Iterable<unknown>)];
      if (values.length === 0) {
        // An empty IN () is a syntax error in Postgres and an empty result set
        // semantically, so say so directly.
        clauses.push("FALSE");
      } else {
        clauses.push(`${col} = ANY(${params.add(values)})`);
      }
    } else {
      clauses.push(`${col} = ${params.add(value)}`);
    }
  }
  return " AND " + clauses.join(" AND ");
}

/**
 * Exponential decay on the fused score, expressed as a half-life in days.
 *
 * A row with no timestamp decays by a factor of 1 (no penalty) rather than 0, so a
 * partially-populated column cannot silently erase results.
 */
function recencyExpr(recency: Recency | null, params: Params): string | null {
  if (recency === null) {
    return null;
  }
  const col = quoteIdent(recency.column);
  const halfLife = params.add(recency.halfLifeDays);
  return (
    `coalesce(exp(-${LN2} * ` +
    `greatest(extract(epoch from (now() - ${col})), 0) ` +
    `/ (${halfLife} * 86400.0)), 1.0)`
  );
}

/** Arguments for one call to {@link buildSearchSql}. */
export interface BuildOptions {
  embedding?: readonly number[] | null;
  text?: string | null;
  limit: number;
  offset?: number;
  filters?: Filters | null;
  candidateLimit?: number | null;
  nearMiss?: number;
  highlight?: boolean;
  fusion?: FusionMethod | null;
}

/**
 * Build the hybrid search statement and its bind parameters.
 *
 * Either signal may be omitted: passing only `embedding` produces a pure vector search
 * and only `text` a pure full-text search, both with the same output columns, which is
 * what makes a three-way comparison of the two signals honest.
 *
 * `nearMiss` extends the result set past `limit` so callers can show the rows that just
 * missed the cut — the ones that are usually the reason a search "failed".
 */
export function buildSearchSql(config: Config, options: BuildOptions): BuiltQuery {
  const cfg = resolveConfig(config);
  const embedding = options.embedding ?? null;
  const text = options.text ?? null;
  const limit = options.limit;
  const offset = options.offset ?? 0;
  const nearMiss = options.nearMiss ?? 0;
  const highlight = options.highlight ?? false;

  if (embedding === null && text === null) {
    throw new Error("at least one of embedding or text must be provided");
  }
  if (limit < 1) {
    throw new Error("limit must be >= 1");
  }
  if (offset < 0) {
    throw new Error("offset must be >= 0");
  }

  const fusion = options.fusion || cfg.fusion;
  // A zero candidate limit is treated as "not supplied" rather than as an out-of-range
  // value, which is what the Python package does; the two have to agree.
  let candidateLimit = options.candidateLimit || cfg.candidateLimit;
  if (candidateLimit < limit + nearMiss) {
    // Fusing fewer candidates than we intend to return would truncate the result set
    // before ranking ever happens.
    candidateLimit = limit + nearMiss;
  }

  const params = new Params();
  const table = quoteIdent(cfg.table);
  const idCol = quoteIdent(cfg.idColumn);

  const ctes: string[] = [];
  const haveVector = embedding !== null;
  const haveText = text !== null;

  if (haveVector) {
    const vec = params.addCast(formatVector(embedding), cfg.vectorType);
    const distance = distanceExpr(cfg, vec);
    const where =
      `WHERE ${quoteIdent(cfg.vectorColumn)} IS NOT NULL` +
      filterSql(cfg, options.filters, params);
    ctes.push(
      "vector_candidates AS (\n" +
        `    SELECT ${idCol} AS id,\n` +
        `           ${distance} AS distance,\n` +
        `           rank() OVER (ORDER BY ${distance}) AS rank\n` +
        `    FROM ${table}\n` +
        `    ${where}\n` +
        `    ORDER BY ${distance}\n` +
        `    LIMIT ${params.add(candidateLimit)}\n` +
        ")",
    );
  }

  if (haveText) {
    const tsquery = tsqueryExpr(cfg, text, params);
    const tsv = tsvectorExpr(cfg);
    const rankExpr = `${cfg.rankFunction}(${tsv}, tsq)`;
    const where = `WHERE ${tsv} @@ tsq` + filterSql(cfg, options.filters, params);
    ctes.push(
      "text_query AS (\n" +
        `    SELECT ${tsquery} AS tsq\n` +
        "),\n" +
        "text_candidates AS (\n" +
        `    SELECT ${idCol} AS id,\n` +
        `           ${rankExpr} AS score,\n` +
        `           rank() OVER (ORDER BY ${rankExpr} DESC) AS rank\n` +
        `    FROM ${table}, text_query\n` +
        `    ${where}\n` +
        `    ORDER BY ${rankExpr} DESC\n` +
        `    LIMIT ${params.add(candidateLimit)}\n` +
        ")",
    );
  }

  const [scoredSelect, scoredFrom] = fusionClause(cfg, params, haveVector, haveText, fusion);
  ctes.push(`scored AS (\n    SELECT ${scoredSelect}\n    FROM ${scoredFrom}\n)`);
  // Summing in a second step keeps each contribution expression written once, which
  // makes the generated SQL readable enough that people copy it out of the README.
  ctes.push(
    "fused AS (\n" +
      "    SELECT id, vector_rank, vector_distance, vector_contribution,\n" +
      "           text_rank, text_score, text_contribution,\n" +
      "           vector_contribution + text_contribution AS fused_score\n" +
      "    FROM scored\n" +
      ")",
  );

  const decay = recencyExpr(cfg.recency, params);
  const scoreExpr = decay === null ? "f.fused_score" : `(f.fused_score * ${decay})`;

  const outColumns = [
    "f.id",
    `${scoreExpr} AS score`,
    "f.fused_score AS fused_score",
    "f.vector_rank",
    "f.vector_distance",
    "f.vector_contribution",
    "f.text_rank",
    "f.text_score",
    "f.text_contribution",
  ];
  if (decay !== null) {
    outColumns.push(`${decay} AS recency_factor`);
  }

  for (const column of outputColumns(cfg)) {
    outColumns.push(`t.${quoteIdent(column)}`);
  }

  if (highlight && haveText) {
    const headlineOpts = params.add(cfg.headlineOptions);
    // ts_headline is expensive and is deliberately evaluated only for the rows that
    // survive ranking, never inside the candidate CTEs.
    outColumns.push(
      `ts_headline('${cfg.language}', t.${quoteIdent(cfg.textColumn)}, ` +
        `(SELECT tsq FROM text_query), ${headlineOpts}) AS highlight`,
    );
  }

  const sql =
    "WITH " +
    ctes.join(",\n") +
    "\n" +
    "SELECT " +
    outColumns.join(",\n       ") +
    "\n" +
    "FROM fused f\n" +
    `JOIN ${table} t ON t.${idCol} = f.id\n` +
    "ORDER BY score DESC, f.id\n" +
    `LIMIT ${params.add(limit + nearMiss)} OFFSET ${params.add(offset)}`;

  return params.render(sql, cfg.paramStyle);
}

/**
 * The SELECT list and FROM clause that combine the two candidate sets.
 *
 * A FULL OUTER JOIN is what lets a row found by only one signal still compete; an
 * INNER JOIN here would quietly reduce hybrid search to the intersection of the two
 * result sets, which is a different and much worse product.
 */
function fusionClause(
  cfg: ResolvedConfig,
  params: Params,
  haveVector: boolean,
  haveText: boolean,
  fusion: FusionMethod,
): [string, string] {
  // Every arithmetic parameter is cast explicitly rather than left to the server's
  // type inference. JavaScript has one number type, so a driver sends 1 where Python
  // sends 1.0, Postgres infers integer, and `1 / (60 + rank)` becomes integer division:
  // every contribution truncates to zero and the ranking collapses to the tiebreaker.
  // The failure is invisible — the query succeeds and returns rows, all scored 0.
  const vectorWeight = `${params.add(cfg.weights.vector)}::float8`;
  const textWeight = `${params.add(cfg.weights.text)}::float8`;

  let vectorContribution: string;
  let textContribution: string;
  if (fusion === "rrf") {
    const k = `${params.add(cfg.k)}::float8`;
    vectorContribution = `${vectorWeight} / (${k} + v.rank)`;
    textContribution = `${textWeight} / (${k} + t.rank)`;
  } else if (fusion === "weighted") {
    // Kept because people ask for it, and documented as the trap it is: cosine
    // distance is bounded and ts_rank is not, so the nominal weights do not describe
    // the actual influence of each signal.
    vectorContribution = `${vectorWeight} * (1.0 - v.distance)`;
    textContribution = `${textWeight} * t.score`;
  } else {
    throw new Error(
      `unknown fusion method ${JSON.stringify(fusion)}; expected 'rrf' or 'weighted'`,
    );
  }

  if (haveVector && haveText) {
    return [
      "coalesce(v.id, t.id) AS id,\n" +
        "           v.rank AS vector_rank,\n" +
        "           v.distance AS vector_distance,\n" +
        `           coalesce(${vectorContribution}, 0) AS vector_contribution,\n` +
        "           t.rank AS text_rank,\n" +
        "           t.score AS text_score,\n" +
        `           coalesce(${textContribution}, 0) AS text_contribution`,
      "vector_candidates v\n    FULL OUTER JOIN text_candidates t ON v.id = t.id",
    ];
  }
  if (haveVector) {
    return [
      "v.id AS id,\n" +
        "           v.rank AS vector_rank,\n" +
        "           v.distance AS vector_distance,\n" +
        `           ${vectorContribution} AS vector_contribution,\n` +
        "           NULL::bigint AS text_rank,\n" +
        "           NULL::double precision AS text_score,\n" +
        "           0.0 AS text_contribution",
      "vector_candidates v",
    ];
  }
  return [
    "t.id AS id,\n" +
      "           NULL::bigint AS vector_rank,\n" +
      "           NULL::double precision AS vector_distance,\n" +
      "           0.0 AS vector_contribution,\n" +
      "           t.rank AS text_rank,\n" +
      "           t.score AS text_score,\n" +
      `           ${textContribution} AS text_contribution`,
    "text_candidates t",
  ];
}

/** Columns copied through from the source table, de-duplicated and ordered. */
function outputColumns(cfg: ResolvedConfig): string[] {
  const columns: string[] = [];
  const seen = new Set<string>();
  for (const column of [cfg.textColumn, ...cfg.extraColumns]) {
    if (column && !seen.has(column)) {
      seen.add(column);
      columns.push(column);
    }
  }
  return columns;
}

/**
 * Render one number the way Python's `repr` renders a float.
 *
 * The vector literal has to be identical in both packages or the parity check is
 * comparing two different statements. JavaScript prints 1 for a whole number where
 * Python prints 1.0, and the two disagree again about when to switch to exponent
 * notation, so neither `String(x)` nor `toFixed` can be used here. The digits
 * themselves already agree: both languages print the shortest decimal that reads back
 * as the same double.
 */
function formatFloat(value: number): string {
  if (Number.isNaN(value)) {
    return "nan";
  }
  if (value === Infinity) {
    return "inf";
  }
  if (value === -Infinity) {
    return "-inf";
  }

  const negative = value < 0 || Object.is(value, -0);
  const magnitude = Math.abs(value);
  const [mantissa = "0", exponent = "0"] = magnitude.toExponential().split("e");
  const digits = mantissa.replace(".", "");
  // Position of the decimal point relative to the first digit, which is what CPython's
  // float_repr_style thresholds are expressed in.
  const decpt = Number(exponent) + 1;

  let body: string;
  if (decpt <= -4 || decpt > 16) {
    // CPython switches to exponent notation at 1e16 and writes at least two exponent
    // digits with an explicit sign: 1e+16, 1e-05.
    const exp = decpt - 1;
    const sign = exp < 0 ? "-" : "+";
    const width = String(Math.abs(exp)).padStart(2, "0");
    const lead = digits.slice(0, 1);
    const rest = digits.slice(1);
    body = `${lead}${rest ? `.${rest}` : ""}e${sign}${width}`;
  } else if (decpt <= 0) {
    body = `0.${"0".repeat(-decpt)}${digits}`;
  } else if (decpt >= digits.length) {
    body = `${digits}${"0".repeat(decpt - digits.length)}.0`;
  } else {
    body = `${digits.slice(0, decpt)}.${digits.slice(decpt)}`;
  }
  return negative ? `-${body}` : body;
}

/**
 * Render a vector in pgvector's text input format.
 *
 * Passing the vector as text and casting keeps the package driver-agnostic: it works
 * with node-postgres, postgres.js, Drizzle and Supabase without any of them
 * registering a pgvector type adapter.
 */
export function formatVector(embedding: readonly number[]): string {
  if (embedding === null || embedding === undefined) {
    throw new Error("embedding must not be null");
  }
  return `[${Array.from(embedding, (value) => formatFloat(toNumber(value))).join(",")}]`;
}

/**
 * Coerce one element of an embedding, rejecting what pgvector would reject anyway.
 *
 * `Number(null)` is 0 and `Number("")` is 0, so the obvious coercion turns a hole in
 * the embedding into a legitimate-looking coordinate. Catching it here beats a
 * driver-specific cast error from the server, or worse, a silently wrong vector.
 */
function toNumber(value: unknown): number {
  if (typeof value === "number") {
    return value;
  }
  if (typeof value === "bigint") {
    return Number(value);
  }
  if (typeof value === "boolean") {
    return value ? 1 : 0;
  }
  if (typeof value === "string" && value.trim() !== "") {
    const parsed = Number(value);
    if (!Number.isNaN(parsed)) {
      return parsed;
    }
  }
  throw new Error(
    `embedding must be a sequence of numbers: ${JSON.stringify(value) ?? String(value)}`,
  );
}
