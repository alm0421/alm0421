"""Offline historical backtesting of intraday strategies on 1-minute bars.

This package is research tooling. It never constructs a broker and never
touches ``app.execution``; the ``backtest`` execution mode in the router stays
refused. Everything here is pure computation over bars loaded from disk (or
fetched once and cached by ``app.backtest.data``).

Modules
-------
data        Load, cache and slice 1-minute bars into regular-session days.
hitchhiker  The HitchHiker scalp: setup detection and single-trade simulation.
portfolio   Position sizing, costs and a minute-resolution equity curve.
metrics     Performance statistics (returns, drawdown, Sharpe, trade stats).
report      Self-contained HTML report plus CSV/JSON artefacts.
"""
