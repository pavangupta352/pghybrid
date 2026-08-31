"""Turning a user's search box into a tsquery that is useful for ranking.

Postgres' query parsers all combine terms with AND. ``websearch_to_tsquery`` turns
``renewal notice period`` into ``'renew' & 'notic' & 'period'``, which matches only
documents containing all three. For a filter that is correct; for the keyword half of
a hybrid search it is quietly destructive, because a query of four or five words
usually matches nothing at all and the fusion silently degrades to vector-only search
without reporting that anything went wrong.

So the default here is ANY: terms are OR-ed, and documents matching more of them rank
higher because ``ts_rank_cd`` already accounts for that. Precision is recovered by
ranking rather than by exclusion, which is how a search engine is supposed to behave.

The OR is built by tokenising the query and combining one ``websearch_to_tsquery``
call per term with the ``||`` operator, rather than by rewriting the operators inside
a parsed tsquery. Rewriting looks simpler and is wrong: ``'a' & !'b'`` becomes
``'a' | !'b'``, which matches every document that merely lacks ``b``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# A quoted phrase, a negated term (optionally quoted), or a bare run of non-space.
_TOKEN_RE = re.compile(r'(-?)"([^"]*)"|(-?)(\S+)')

# Bare boolean words a person might type. They are dropped rather than searched for:
# under ANY semantics the OR is already implied, and searching for the literal word
# "or" pollutes the ranking.
_NOISE = {"or", "and"}


@dataclass(frozen=True)
class ParsedQuery:
    """A search box split into terms to include and terms to exclude."""

    positive: list[str]
    negative: list[str]

    @property
    def is_empty(self) -> bool:
        return not self.positive and not self.negative


def parse_query(text: str) -> ParsedQuery:
    """Split a raw search string into positive and negative terms.

    Supports the syntax people already expect from a search box: double-quoted
    phrases are kept whole, and a leading ``-`` excludes a term.

        >>> parse_query('renewal "notice period" -pricing')
        ParsedQuery(positive=['renewal', 'notice period'], negative=['pricing'])
    """
    positive: list[str] = []
    negative: list[str] = []

    for match in _TOKEN_RE.finditer(text or ""):
        quoted_neg, quoted, bare_neg, bare = match.groups()
        if quoted is not None:
            term, negated = quoted.strip(), quoted_neg == "-"
        else:
            term, negated = bare.strip(), bare_neg == "-"

        if not term:
            continue
        if not negated and term.lower() in _NOISE:
            continue

        (negative if negated else positive).append(term)

    return ParsedQuery(positive=positive, negative=negative)
