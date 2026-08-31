# Security

## Reporting a vulnerability

Please report privately through GitHub's
[private vulnerability reporting](https://github.com/pavangupta352/pghybrid/security/advisories/new)
rather than in a public issue. I will confirm receipt and, if the report holds, fix it and
credit you in the release notes unless you would rather not be named.

## What is in scope

This library builds SQL and hands it to your driver. The interesting question is therefore
always the same one: **can a value a caller supplies escape into the statement?**

The rule the code follows is that there are exactly three kinds of input:

| Kind | Handling | Examples |
|---|---|---|
| **Values** | bound as parameters, never interpolated | the query text, the embedding, filter values, limits, `headline_options` |
| **Identifiers** | validated by `quote_ident` and double-quoted; anything not identifier-shaped is rejected rather than escaped | table, schema and column names |
| **Query fragments** | cannot be bound, so validated against a strict shape or a closed set | `language`, `query_parser`, `rank_function` |

A report that a caller-supplied value reaches the statement outside those three paths is a
vulnerability, and so is a way to make one of the three accept something it should not.

Known and deliberate, so not vulnerabilities on their own:

- The generated SQL is meant to be read, copied and run by hand. Anything you paste into a
  psql session runs with your privileges.
- `pghybrid` never opens a connection. Whatever your `execute` callable can do, a search
  can do, so grant it the least privilege that works — read-only is enough for everything
  except `init --apply`.
- `explain` and `doctor` print row content and index definitions. Treat their output as you
  would treat the data itself.

## Previously fixed

- **Interpolated query fragments were unvalidated.** `language`, `query_parser` and
  `rank_function` are parts of the query rather than values and so cannot be bound. They
  were typed but never checked at run time, which meant a `Config` built from user input —
  an application letting someone pick a search language, for instance — could close the
  string literal it sat inside and append arbitrary SQL. Fixed before the first release,
  in both packages, along with a second path in the CLI that applied overrides with
  `setattr` and so skipped the validation entirely.
