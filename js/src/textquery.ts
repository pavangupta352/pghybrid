/**
 * Turning a user's search box into a tsquery that is useful for ranking.
 *
 * Postgres' query parsers all combine terms with AND. `websearch_to_tsquery` turns
 * `renewal notice period` into `'renew' & 'notic' & 'period'`, which matches only
 * documents containing all three. For a filter that is correct; for the keyword half
 * of a hybrid search it is quietly destructive, because a query of four or five words
 * usually matches nothing at all and the fusion silently degrades to vector-only
 * search without reporting that anything went wrong.
 *
 * So the default here is ANY: terms are OR-ed, and documents matching more of them
 * rank higher because `ts_rank_cd` already accounts for that. Precision is recovered
 * by ranking rather than by exclusion, which is how a search engine is supposed to
 * behave.
 *
 * The OR is built by tokenising the query and combining one `websearch_to_tsquery`
 * call per term with the `||` operator, rather than by rewriting the operators inside
 * a parsed tsquery. Rewriting looks simpler and is wrong: `'a' & !'b'` becomes
 * `'a' | !'b'`, which matches every document that merely lacks `b`.
 */

/** A quoted phrase, a negated term (optionally quoted), or a bare run of non-space. */
const TOKEN_RE = /(-?)"([^"]*)"|(-?)(\S+)/g;

/**
 * Bare boolean words a person might type. They are dropped rather than searched for:
 * under ANY semantics the OR is already implied, and searching for the literal word
 * "or" pollutes the ranking.
 */
const NOISE = new Set(["or", "and"]);

/** A search box split into terms to include and terms to exclude. */
export interface ParsedQuery {
  readonly positive: string[];
  readonly negative: string[];
  /** True when nothing in the string survived tokenising. */
  readonly isEmpty: boolean;
}

/**
 * Split a raw search string into positive and negative terms.
 *
 * Supports the syntax people already expect from a search box: double-quoted phrases
 * are kept whole, and a leading `-` excludes a term.
 *
 * ```ts
 * parseQuery('renewal "notice period" -pricing');
 * // { positive: ["renewal", "notice period"], negative: ["pricing"], isEmpty: false }
 * ```
 */
export function parseQuery(text: string | null | undefined): ParsedQuery {
  const positive: string[] = [];
  const negative: string[] = [];
  // OR-ing a term with itself is the same term, so a repeat only makes the statement
  // bigger. It matters more than it sounds: text pasted into a search box repeats words
  // constantly, and every repeat is another parser call in the generated SQL.
  const seenPositive = new Set<string>();
  const seenNegative = new Set<string>();

  // A fresh regex per call: a module-level /g regex carries lastIndex between calls,
  // so a shared one would tokenise every second query from the wrong offset.
  const pattern = new RegExp(TOKEN_RE.source, "g");
  let match: RegExpExecArray | null;
  while ((match = pattern.exec(text ?? "")) !== null) {
    const [, quotedNegation, quoted, bareNegation, bare] = match;

    let term: string;
    let negated: boolean;
    if (quoted !== undefined) {
      term = quoted.trim();
      negated = quotedNegation === "-";
    } else {
      term = (bare ?? "").trim();
      negated = bareNegation === "-";
    }

    if (!term) {
      continue;
    }
    if (!negated && NOISE.has(term.toLowerCase())) {
      continue;
    }

    const bucket = negated ? negative : positive;
    const seen = negated ? seenNegative : seenPositive;
    const folded = term.toLowerCase();
    if (!seen.has(folded)) {
      seen.add(folded);
      bucket.push(term);
    }
  }

  return { positive, negative, isEmpty: positive.length === 0 && negative.length === 0 };
}
