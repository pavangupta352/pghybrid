/** Hybrid search on plain Postgres. */

export {
  COSINE,
  DEFAULT_HEADLINE_OPTIONS,
  DEFAULT_RRF_K,
  INNER_PRODUCT,
  L1,
  L2,
  METRICS,
  opsClass,
  opsFor,
  resolveConfig,
} from "./config.js";
export type {
  Config,
  FusionMethod,
  Metric,
  MetricName,
  ParamStyle,
  QueryParser,
  RankFunction,
  Recency,
  ResolvedConfig,
  TextMatch,
  VectorType,
  Weights,
} from "./config.js";

export { buildSearchSql, formatVector, IdentifierError, Params, quoteIdent } from "./sql.js";
export type { BuildOptions, BuiltQuery, Filters } from "./sql.js";

export { parseQuery } from "./textquery.js";
export type { ParsedQuery } from "./textquery.js";

export {
  asFloat,
  HybridSearch,
  resultFromRow,
  resultsFromRows,
  rowMapping,
} from "./search.js";
export type { Executor, MatchedBy, Row, SearchOptions, SearchResult } from "./search.js";

export const VERSION = "0.1.2";

export * from "./adapters.js";
