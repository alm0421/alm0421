"""Python API for notebooks and scripts.

    from backtester import api
    r = api.backtest("buy MSFT at the close when it is down 5 days in a row, hold 1 day")
    r.stats["cagr"], r.trades.head(), r.equity.plot()
    api.report(r, "reports/msft")                                 # HTML/Excel/CSV like the CLI

    # rules as Python functions (receive the ticker's DataFrame and the rule namespace):
    from backtester.strategy import Strategy
    def cheap(df, ns):
        return (df.close < df.close.rolling(10).min().shift(1)) & (ns["rsi"](2) < 10)
    r = api.backtest(Strategy(universe=["QQQ"], entry=cheap, hold_bars=3))

    # custom portfolio logic: f(date, history) -> {ticker: weight}, history = data up to `date`
    from backtester.portfolio import Portfolio
    def risk_on(date, h):
        spy = h["SPY"].close
        return {"QQQ": 1.0} if spy.iloc[-1] > spy.tail(200).mean() else {"TLT": 0.5, "GLD": 0.5}
    r = api.backtest(Portfolio(tree={"custom": risk_on, "tickers": ["SPY", "QQQ", "TLT", "GLD"]}, rebalance="monthly"))

Rules written as functions must only use data up to each row (no .shift(-1), no centred windows): each run
probes them on data cut at many dates (every day the rule fires or changes its answer, the days before those,
the latest bars one by one, and a dense random grid) and refuses one whose output changes (expr.callable_lookahead_probe).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import parser, report as _report, runner
from .engine import Result


@dataclass
class Backtest:
    spec: object
    result: Result
    analysis: dict

    @property
    def equity(self) -> pd.Series:
        return self.result.equity

    @property
    def trades(self) -> pd.DataFrame:
        return self.result.trades

    @property
    def holdings(self) -> pd.DataFrame | None:
        return self.result.holdings

    @property
    def stats(self) -> dict:
        return self.analysis["stats"]

    @property
    def trade_stats(self) -> dict:
        return self.analysis["trade_stats"]

    def summary(self) -> str:
        return _report.console_summary(self.analysis)


def backtest(spec_or_text, rf="tbill", detail: bool = False, **overrides) -> Backtest:
    spec = parser.parse(spec_or_text, **overrides) if isinstance(spec_or_text, str) else spec_or_text
    for k, v in overrides.items():
        if not isinstance(spec_or_text, str) and v is not None and hasattr(spec, k):
            setattr(spec, k, v)
    spec.validate()
    res = runner.run(spec)
    A = _report.analyze(res, rf=rf, sensitivity=detail, mc=detail, detail=detail)
    return Backtest(spec, res, A)


def compare(*items, names: list[str] | None = None) -> pd.DataFrame:
    """Stats for several strategies over their common period, as a DataFrame."""
    bts = [i if isinstance(i, Backtest) else backtest(i) for i in items]
    for j, b in enumerate(bts):
        b.spec.name = (names[j] if names else b.spec.name) or f"Strategy {chr(65 + j)}"
    C = _report.common_window_stats([b.analysis for b in bts])
    return pd.DataFrame(C["columns"]).T


def composer_export(spec_or_text) -> dict:
    """A portfolio (spec, dict or sentence) as a Composer symphony dict; raises ComposerExportError for what Composer
    lacks (shorts, leverage, other indicators). The notes on what was left out are under "_notes"."""
    from . import composer_export as _ce
    spec = parser.parse(spec_or_text) if isinstance(spec_or_text, str) else spec_or_text
    if isinstance(spec, Backtest):
        spec = spec.spec
    sym, notes = _ce.export(spec)
    return {**sym, "_notes": notes} if notes else sym


def report(bt: Backtest | list[Backtest], out: str | Path) -> Path:
    items = bt if isinstance(bt, list) else [bt]
    analyses = [_report.analyze(b.result) for b in items]
    return _report.write_outputs(analyses, Path(out))
