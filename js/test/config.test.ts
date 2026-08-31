/**
 * Unit tests for the configuration object.
 *
 * The config is the whole public surface of the library: everything else is a pure
 * function of it. So the validation here is the only place a mistake can be caught
 * before it turns into a confusing runtime error from a driver, or worse, into search
 * results that are quietly a little bit wrong.
 *
 * Each rejection test asserts that the message names the valid options. An error that
 * only says "invalid" costs the reader a trip to the source.
 */

import { describe, expect, it } from "vitest";

import {
  COSINE,
  DEFAULT_RRF_K,
  INNER_PRODUCT,
  L1,
  L2,
  METRICS,
  opsClass,
  opsFor,
  resolveConfig,
} from "../src/config.js";
import type { Config, Metric } from "../src/config.js";

const REQUIRED: Config = { table: "chunks", textColumn: "content", vectorColumn: "embedding" };

describe("metrics", () => {
  it.each([
    ["cosine", COSINE],
    ["l2", L2],
    ["euclidean", L2],
    ["ip", INNER_PRODUCT],
    ["inner_product", INNER_PRODUCT],
    ["l1", L1],
    ["manhattan", L1],
  ] as const)("resolves the alias %s to the shared singleton", (alias, expected) => {
    // People name these distances differently depending on where they learned them.
    expect(resolveConfig({ ...REQUIRED, metric: alias }).metric).toBe(expected);
  });

  it("resolves every registered alias", () => {
    // Guards against an alias being added to METRICS but not to the coercion path.
    for (const alias of Object.keys(METRICS)) {
      const metric = resolveConfig({ ...REQUIRED, metric: alias as keyof typeof METRICS }).metric;
      expect(metric.operator).toBeTypeOf("string");
    }
  });

  it("passes a metric object through untouched", () => {
    expect(resolveConfig({ ...REQUIRED, metric: L2 }).metric).toBe(L2);
  });

  it("names the valid options when the metric is unknown", () => {
    let message = "";
    try {
      resolveConfig({ ...REQUIRED, metric: "dot" as never });
    } catch (error) {
      message = (error as Error).message;
    }
    expect(message).toContain("dot");
    for (const alias of ["cosine", "euclidean", "inner_product", "manhattan"]) {
      expect(message).toContain(alias);
    }
  });

  it.each([
    [COSINE, "vector_cosine_ops", "halfvec_cosine_ops"],
    [L2, "vector_l2_ops", "halfvec_l2_ops"],
    [INNER_PRODUCT, "vector_ip_ops", "halfvec_ip_ops"],
    [L1, "vector_l1_ops", "halfvec_l1_ops"],
  ] as [Metric, string, string][])(
    "picks the operator class from the vector type",
    (metric, vectorOps, halfvecOps) => {
      // An index built with the wrong operator class is simply never used. Postgres
      // does not complain: it plans a sequential scan and the query gets slower by
      // orders of magnitude with no error to explain why.
      expect(opsClass({ ...REQUIRED, metric, vectorType: "vector" })).toBe(vectorOps);
      expect(opsClass({ ...REQUIRED, metric, vectorType: "halfvec" })).toBe(halfvecOps);
      expect(opsFor(metric, "halfvec")).toBe(halfvecOps);
    },
  );

  it("treats every metric as a distance", () => {
    // The ranking code assumes smaller is closer; pgvector has no similarity operators.
    for (const metric of Object.values(METRICS)) {
      expect(metric.ascending).toBe(true);
    }
  });
});

describe("enumerated fields", () => {
  it("names the valid options for an invalid vectorType", () => {
    expect(() => resolveConfig({ ...REQUIRED, vectorType: "float16" as never })).toThrow(
      /float16.*vector.*halfvec/s,
    );
  });

  it("names the valid options for an invalid paramStyle", () => {
    // Getting the placeholder style wrong is the first thing that breaks for a new
    // user: node-postgres raises a syntax error on %s and psycopg raises one on $1, and
    // neither message mentions this library.
    expect(() => resolveConfig({ ...REQUIRED, paramStyle: "qmark" as never })).toThrow(
      /qmark.*numeric.*pyformat/s,
    );
  });

  it("names the valid options for an invalid textMatch", () => {
    expect(() => resolveConfig({ ...REQUIRED, textMatch: "either" as never })).toThrow(
      /either.*'any'.*'all'/s,
    );
  });

  it.each(["vector", "halfvec"] as const)("accepts the vector type %s", (vectorType) => {
    expect(resolveConfig({ ...REQUIRED, vectorType }).vectorType).toBe(vectorType);
  });

  it.each(["numeric", "pyformat"] as const)("accepts the param style %s", (paramStyle) => {
    expect(resolveConfig({ ...REQUIRED, paramStyle }).paramStyle).toBe(paramStyle);
  });

  it.each(["any", "all"] as const)("accepts the text match mode %s", (textMatch) => {
    expect(resolveConfig({ ...REQUIRED, textMatch }).textMatch).toBe(textMatch);
  });
});

describe("weights", () => {
  it.each([
    [-0.1, 1.0],
    [1.0, -0.1],
    [-1.0, -1.0],
    [0.0, -1.0],
  ])("rejects the negative weights %f / %f", (vector, text) => {
    // A negative weight would rank a signal's best results last, which is never meant.
    expect(() => resolveConfig({ ...REQUIRED, weights: { vector, text } })).toThrow(
      /non-negative/,
    );
  });

  it("rejects both weights at zero", () => {
    // Every row would score zero and the ordering would collapse to the id tiebreak.
    expect(() => resolveConfig({ ...REQUIRED, weights: { vector: 0, text: 0 } })).toThrow(
      /greater than zero/,
    );
  });

  it.each([
    [1.0, 0.0],
    [0.0, 1.0],
  ])("allows zeroing one weight (%f / %f)", (vector, text) => {
    // Turning one signal off is a legitimate way to measure what the other contributes.
    expect(resolveConfig({ ...REQUIRED, weights: { vector, text } }).weights).toEqual({
      vector,
      text,
    });
  });

  it("defaults to parity", () => {
    expect(resolveConfig(REQUIRED).weights).toEqual({ vector: 1.0, text: 1.0 });
  });

  it("takes the defaults for a partially specified weight object", () => {
    // Configs routinely arrive from JSON, YAML or a settings file.
    expect(resolveConfig({ ...REQUIRED, weights: { vector: 3.0 } }).weights).toEqual({
      vector: 3.0,
      text: 1.0,
    });
  });
});

describe("recency", () => {
  it.each([0, -1, -30.5])("rejects the half-life %f", (halfLifeDays) => {
    // A zero or negative half-life makes the decay term explode or invert.
    expect(() =>
      resolveConfig({ ...REQUIRED, recency: { column: "published_at", halfLifeDays } }),
    ).toThrow(/halfLifeDays/);
  });

  it("accepts a valid recency", () => {
    expect(
      resolveConfig({ ...REQUIRED, recency: { column: "published_at", halfLifeDays: 30 } })
        .recency,
    ).toEqual({ column: "published_at", halfLifeDays: 30 });
  });

  it("defaults to off", () => {
    // Decay changes ranking, so it is never applied unless it was asked for.
    expect(resolveConfig(REQUIRED).recency).toBeNull();
  });
});

describe("numeric bounds and defaults", () => {
  it.each([-1, -60])("rejects a negative k (%i)", (k) => {
    expect(() => resolveConfig({ ...REQUIRED, k })).toThrow(/k must be non-negative/);
  });

  it("allows k of zero", () => {
    // k=0 makes RRF the plain reciprocal of the rank, which is a defensible choice.
    expect(resolveConfig({ ...REQUIRED, k: 0 }).k).toBe(0);
  });

  it.each([0, -1])("rejects a candidateLimit of %i", (candidateLimit) => {
    // A zero candidate budget would fuse two empty lists and return nothing.
    expect(() => resolveConfig({ ...REQUIRED, candidateLimit })).toThrow(/candidateLimit/);
  });

  it("uses the defaults the README documents", () => {
    // These defaults are the library's argument, so pin them. textMatch="any" in
    // particular is the whole thesis: Postgres' native AND semantics make the keyword
    // half of a hybrid search return nothing for most multi-word queries, which
    // silently degrades the system to vector-only search.
    const cfg = resolveConfig(REQUIRED);
    expect(cfg.idColumn).toBe("id");
    expect(cfg.tsvectorColumn).toBeNull();
    expect(cfg.language).toBe("english");
    expect(cfg.vectorType).toBe("vector");
    expect(cfg.metric).toBe(COSINE);
    expect(cfg.fusion).toBe("rrf");
    expect(cfg.k).toBe(60);
    expect(DEFAULT_RRF_K).toBe(60);
    expect(cfg.candidateLimit).toBe(50);
    expect(cfg.textMatch).toBe("any");
    expect(cfg.paramStyle).toBe("numeric");
    expect(cfg.queryParser).toBe("websearch_to_tsquery");
    expect(cfg.rankFunction).toBe("ts_rank_cd");
    expect(cfg.filterColumns).toEqual([]);
    expect(cfg.extraColumns).toEqual([]);
    expect(cfg.headlineOptions).toBe(
      "StartSel=<mark>, StopSel=</mark>, MaxFragments=2, MinWords=8, MaxWords=30",
    );
  });
});
