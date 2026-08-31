/**
 * Configuration objects describing the table being searched.
 *
 * A {@link Config} is the only thing the SQL builder needs. It is deliberately
 * declarative and free of connection details so that the same object can generate a
 * migration, a search query and a diagnostic report.
 *
 * The object is a plain literal rather than a class: a config that has to be
 * constructed cannot be written in a JSON file, loaded from an environment, or spread
 * over a default, and all three are things people do with this.
 */

export type VectorType = "vector" | "halfvec";
export type FusionMethod = "rrf" | "weighted";
/** "numeric" produces $1, $2 (node-postgres, asyncpg, raw SQL, Supabase). */
/** "pyformat" produces %s (psycopg 2 and 3), for a Python service sharing this config. */
export type ParamStyle = "numeric" | "pyformat";
/**
 * "any" OR-combines the query terms so the keyword signal still ranks when no
 * document contains every word; "all" keeps Postgres' native AND semantics.
 */
export type TextMatch = "any" | "all";
export type QueryParser = "websearch_to_tsquery" | "plainto_tsquery" | "phraseto_tsquery";
export type RankFunction = "ts_rank_cd" | "ts_rank";

/**
 * The RRF constant from Cormack, Clarke & Buettcher (2009), "Reciprocal Rank Fusion
 * outperforms Condorcet and individual rank learning methods". 60 is their reported
 * value and remains the sane default: it flattens the difference between the top few
 * ranks so neither signal can dominate on its first result alone.
 */
export const DEFAULT_RRF_K = 60;

/** A pgvector distance metric and the index operator classes that match it. */
export interface Metric {
  readonly name: string;
  readonly operator: string;
  readonly opsVector: string;
  readonly opsHalfvec: string;
  /**
   * Whether a smaller value means a closer match. Every pgvector operator is a
   * distance, so this is always true; it exists to keep the ranking code honest.
   */
  readonly ascending: boolean;
}

export const COSINE: Metric = {
  name: "cosine",
  operator: "<=>",
  opsVector: "vector_cosine_ops",
  opsHalfvec: "halfvec_cosine_ops",
  ascending: true,
};

export const L2: Metric = {
  name: "l2",
  operator: "<->",
  opsVector: "vector_l2_ops",
  opsHalfvec: "halfvec_l2_ops",
  ascending: true,
};

export const INNER_PRODUCT: Metric = {
  name: "inner_product",
  operator: "<#>",
  opsVector: "vector_ip_ops",
  opsHalfvec: "halfvec_ip_ops",
  ascending: true,
};

export const L1: Metric = {
  name: "l1",
  operator: "<+>",
  opsVector: "vector_l1_ops",
  opsHalfvec: "halfvec_l1_ops",
  ascending: true,
};

export const METRICS = {
  cosine: COSINE,
  l2: L2,
  euclidean: L2,
  inner_product: INNER_PRODUCT,
  ip: INNER_PRODUCT,
  l1: L1,
  manhattan: L1,
} as const satisfies Record<string, Metric>;

export type MetricName = keyof typeof METRICS;

/** The operator class an index on the vector column must use for this metric. */
export function opsFor(metric: Metric, vectorType: VectorType): string {
  return vectorType === "vector" ? metric.opsVector : metric.opsHalfvec;
}

/**
 * Relative influence of each signal.
 *
 * Under RRF these behave the way they read, because both terms are computed from
 * ranks and therefore share a scale. Under `weighted` fusion they do not.
 */
export interface Weights {
  vector?: number;
  text?: number;
}

/**
 * Exponential decay applied to the fused score.
 *
 * `halfLifeDays` is the age at which a row's score is halved. Rows with a NULL
 * timestamp are left undecayed rather than dropped.
 */
export interface Recency {
  column: string;
  halfLifeDays: number;
}

/**
 * Describes one searchable table.
 *
 * Only `table`, `textColumn` and `vectorColumn` are required. Everything else has a
 * defensible default, and every default is stated in the README so the behaviour is
 * never a surprise.
 */
export interface Config {
  table: string;
  textColumn: string;
  vectorColumn: string;

  idColumn?: string;
  /**
   * A stored tsvector column. Leave it undefined to have the query compute the
   * tsvector inline, which needs no migration but cannot use a GIN index.
   */
  tsvectorColumn?: string | null;

  language?: string;
  vectorType?: VectorType;
  metric?: MetricName | Metric;

  fusion?: FusionMethod;
  k?: number;
  weights?: Weights;

  /**
   * How many rows each signal contributes to the fusion. Larger values find more
   * rows that one signal ranked poorly, at a proportional cost per query.
   */
  candidateLimit?: number;

  filterColumns?: readonly string[];
  extraColumns?: readonly string[];
  recency?: Recency | null;

  queryParser?: QueryParser;
  rankFunction?: RankFunction;
  /**
   * Placeholder style for the driver you use. Getting this wrong is the first thing
   * that breaks for a new user, so it is explicit rather than guessed.
   */
  paramStyle?: ParamStyle;
  /**
   * See {@link TextMatch}. "any" is the default because AND semantics make the
   * keyword half of a hybrid search return nothing for most multi-word queries, which
   * silently degrades the whole system to vector-only search.
   */
  textMatch?: TextMatch;
  headlineOptions?: string;
}

/** A {@link Config} with every default filled in and every value checked. */
export interface ResolvedConfig {
  readonly table: string;
  readonly textColumn: string;
  readonly vectorColumn: string;
  readonly idColumn: string;
  readonly tsvectorColumn: string | null;
  readonly language: string;
  readonly vectorType: VectorType;
  readonly metric: Metric;
  readonly fusion: FusionMethod;
  readonly k: number;
  readonly weights: { readonly vector: number; readonly text: number };
  readonly candidateLimit: number;
  readonly filterColumns: readonly string[];
  readonly extraColumns: readonly string[];
  readonly recency: Recency | null;
  readonly queryParser: QueryParser;
  readonly rankFunction: RankFunction;
  readonly paramStyle: ParamStyle;
  readonly textMatch: TextMatch;
  readonly headlineOptions: string;
}

export const DEFAULT_HEADLINE_OPTIONS =
  "StartSel=<mark>, StopSel=</mark>, MaxFragments=2, MinWords=8, MaxWords=30";

function resolveMetric(metric: MetricName | Metric | undefined): Metric {
  if (metric === undefined) {
    return COSINE;
  }
  if (typeof metric === "string") {
    const known: Metric | undefined = METRICS[metric];
    if (known === undefined) {
      throw new Error(
        `unknown metric ${JSON.stringify(metric)}; expected one of ` +
          `${Object.keys(METRICS).sort().join(", ")}`,
      );
    }
    return known;
  }
  return metric;
}

function resolveWeights(weights: Weights | undefined): { vector: number; text: number } {
  const vector = weights?.vector ?? 1.0;
  const text = weights?.text ?? 1.0;
  if (vector < 0 || text < 0) {
    throw new Error("weights must be non-negative");
  }
  if (vector === 0 && text === 0) {
    throw new Error("at least one weight must be greater than zero");
  }
  return { vector, text };
}

function resolveRecency(recency: Recency | null | undefined): Recency | null {
  if (recency === undefined || recency === null) {
    return null;
  }
  if (!(recency.halfLifeDays > 0)) {
    throw new Error("halfLifeDays must be greater than zero");
  }
  return { column: recency.column, halfLifeDays: recency.halfLifeDays };
}

/**
 * Apply the defaults and reject the values that would otherwise fail as a database
 * error hours later.
 *
 * Validation lives here rather than in the builder so that a bad config is caught by
 * the first query it is used for, with a message naming the field, instead of by the
 * server with a message naming a placeholder.
 */
export function resolveConfig(config: Config): ResolvedConfig {
  const textMatch = config.textMatch ?? "any";
  if (textMatch !== "any" && textMatch !== "all") {
    throw new Error(`textMatch must be 'any' or 'all', got ${JSON.stringify(textMatch)}`);
  }

  const paramStyle = config.paramStyle ?? "numeric";
  if (paramStyle !== "numeric" && paramStyle !== "pyformat") {
    throw new Error(
      `paramStyle must be 'numeric' or 'pyformat', got ${JSON.stringify(paramStyle)}`,
    );
  }

  const vectorType = config.vectorType ?? "vector";
  if (vectorType !== "vector" && vectorType !== "halfvec") {
    throw new Error(
      `vectorType must be 'vector' or 'halfvec', got ${JSON.stringify(vectorType)}`,
    );
  }

  const k = config.k ?? DEFAULT_RRF_K;
  if (k < 0) {
    throw new Error("k must be non-negative");
  }

  const candidateLimit = config.candidateLimit ?? 50;
  if (candidateLimit < 1) {
    throw new Error("candidateLimit must be >= 1");
  }

  return {
    table: config.table,
    textColumn: config.textColumn,
    vectorColumn: config.vectorColumn,
    idColumn: config.idColumn ?? "id",
    tsvectorColumn: config.tsvectorColumn ?? null,
    language: config.language ?? "english",
    vectorType,
    metric: resolveMetric(config.metric),
    fusion: config.fusion ?? "rrf",
    k,
    weights: resolveWeights(config.weights),
    candidateLimit,
    filterColumns: config.filterColumns ?? [],
    extraColumns: config.extraColumns ?? [],
    recency: resolveRecency(config.recency),
    queryParser: config.queryParser ?? "websearch_to_tsquery",
    rankFunction: config.rankFunction ?? "ts_rank_cd",
    paramStyle,
    textMatch,
    headlineOptions: config.headlineOptions ?? DEFAULT_HEADLINE_OPTIONS,
  };
}

/** The operator class an index on the config's vector column must use. */
export function opsClass(config: Config): string {
  const resolved = resolveConfig(config);
  return opsFor(resolved.metric, resolved.vectorType);
}
