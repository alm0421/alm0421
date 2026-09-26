# Backtester — notes for Claude

This repo backtests trading strategies written in plain English. See README.md for the user-facing guide.

## Running a backtest the user describes

1. `python -m backtester "<their sentence>" --dry-run` and check the printed interpretation
   against what they meant (ticker, entry timing, condition, exits, sizing). Pay attention to notes.
2. If the parser refuses a phrase or misreads it, don't rephrase their idea into something different;
   express the exact rule in the expression language, either inline in backticks inside the sentence
   or via `--tickers/--entry/--exit-when/--hold` (vocabulary: `python -m backtester --help-expr`,
   defined in `backtester/expr.py`). Ambiguities that change results (e.g. "down 5 days" = at least vs.
   exactly 5; entry at today's open vs. next open) should be stated to the user.
3. Run without `--dry-run`. Report the interpretation, the headline numbers, and the path to
   `reports/<slug>/report.html` (send that file to the user). Mention survivorship bias for
   multi-stock universes and the cost-sensitivity table for short holding periods.
4. If a phrase is common and missing from the parser, add a pattern in `backtester/parser.py`
   (`parse_condition`) plus a case in `tests/test_backtester.py`.

## Layout

- `backtester/parser.py` English → `Strategy` (regex patterns; refuses rather than guesses)
- `backtester/expr.py` safe rule language (AST whitelist) and indicators
- `backtester/engine.py` bar-by-bar portfolio simulator (open → intraday → close ordering)
- `backtester/metrics.py`, `backtester/report.py`, `backtester/report_template.html` statistics and report
- `scripts/fetch_data.py` + `.github/workflows/fetch-data.yml` data download (the sandbox can't reach
  Yahoo; the GitHub Action can — push or run the workflow, then `git pull`)
- `data/prices/*.csv` daily OHLCV (split-adjusted OHLC, `adj_close` incl. dividends)

## Invariants (tests enforce these)

- No lookahead: a rule filled "at the open" may only use open-time data (`parser._open_safe`); ranking
  for open entries uses the previous bar.
- Final equity == capital + sum of trade P&L.
- `python -m pytest -q` must pass before committing.
