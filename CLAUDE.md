# Backtester — notes for Claude

This repo backtests trading strategies and portfolios described in plain English. README.md is the
user guide. The site is `python -m backtester web`.

## Running a backtest the user describes

1. Run `python -m backtester "<their sentence>" --dry-run` and check the interpretation against what
   they meant: ticker(s), entry timing, conditions, exits, sizing and rebalancing. Read the notes.
2. If the parser refuses or misreads, don't substitute a different idea. Express the exact rule in
   backticks inside the sentence, or use `--tickers/--entry/--exit-when/--hold`, or write a JSON spec
   (`Strategy` in backtester/strategy.py, `Portfolio` in backtester/portfolio.py) and pass it with
   `--spec`. Rule vocabulary: `python -m backtester --help-expr`.
3. Tell the user about any ambiguity that changes results: "down 5 days" means at least 5, not
   exactly 5; entry at today's open vs the next open; the 4% rule vs 4% of the balance.
4. Run it without `--dry-run`. Report:
   - the interpretation and the headline numbers
   - the path to `reports/<slug>/report.html`
   - for Nasdaq-100 universes, the remaining survivorship caveat
   - for short holding periods, the cost-sensitivity table
5. If a common phrase is missing, add a pattern to `backtester/parser.py` (`parse_condition` for
   conditions, `_node` for portfolio phrases), plus a test.

## Layout

- `backtester/parser.py`: English → `Strategy` (signals) or `Portfolio` (allocations). Tracks every
  word it consumes and refuses anything left over.
- `backtester/expr.py`: the safe rule language (AST whitelist), indicators, and the `open_safe`
  lookahead guard.
- `backtester/engine.py`: the signal simulator (orders, stops, sizing, financing, dividends).
- `backtester/portfolio.py`: the allocation simulator (tree of weights / if / filter nodes,
  rebalancing, cash flows).
- `backtester/runner.py`: dispatches to the right engine and loads/saves JSON specs.
- `backtester/metrics.py` and `backtester/report.py` with `report_template.html`: statistics and the
  multi-run HTML report, plus Excel/CSV/PDF.
- `backtester/research.py` and `research_report.py` with `research_template.html`: sweep,
  walk-forward and optimiser.
- `backtester/signals.py`: today's signals and paper trading.
- `backtester/costs.py`: commissions (IBKR presets) and volume slippage for both engines.
- `backtester/broker.py`: Alpaca paper/live orders (`python -m backtester trade ... --dry-run`).
- `backtester/web.py` with `webapp.html`: the site (standard library HTTP server).
- `scripts/fetch_data.py` with `.github/workflows/fetch-data.yml`: data. The sandbox can't reach
  Yahoo or Wikipedia; the Action can. Push, or run the workflow, then `git pull`.
- `scripts/daily_signals.py` with `.github/workflows/signals.yml`: daily paper-trading scan and
  webhook.

## Invariants (tests enforce these)

- No lookahead:
  - Truncating the data at date D must not change any trade or equity value before D.
  - Open fills use only open-safe rules.
  - Negative offsets are errors.
- Final equity == capital + sum of trade P&L + interest (signals). A 60/40 portfolio matches the
  independent calculation to 1e-9.
- `python -m pytest -q` must pass before committing.
