"""Run any strategy spec (signal Strategy or allocation Portfolio) and load specs from JSON."""
from __future__ import annotations

import json
from pathlib import Path

from . import engine, portfolio
from .engine import Result
from .portfolio import Portfolio
from .strategy import Strategy

Spec = Strategy | Portfolio


def run(spec: Spec) -> Result:
    if isinstance(spec, Portfolio):
        res = portfolio.run(spec)
    else:
        res = engine.run(spec)
    _identity_notes(spec, res)
    return res


def _identity_notes(spec: Spec, res: Result) -> None:
    """Warn when a ticker the user named is probably not the company they mean in the period (a recycled
    symbol or junk series), or stopped trading within it. Index universes are skipped: their
    point-in-time membership already excludes such series."""
    from . import data
    try:
        if isinstance(spec, Portfolio):
            names = portfolio.tickers_in(spec.tree) if isinstance(spec.tree, dict) else []
        elif getattr(spec, "universe_name", None):
            names = []
        else:
            names = list(spec.universe)
        eq = res.equity
        start, end = (eq.index[0], eq.index[-1]) if len(eq) else (spec.start, spec.end)
        for t in names:
            for n in data.identity_notes(t, start, end):
                if n not in spec.notes:
                    spec.notes.append(n)
    except Exception:  # noqa: BLE001 - a warning must never break a backtest
        pass


def from_dict(d: dict) -> Spec:
    d = dict(d)
    kind = d.pop("kind", None)
    if kind == "allocation" or "tree" in d:
        return Portfolio.from_dict(d)
    return Strategy.from_dict(d)


def load(path: str | Path) -> Spec:
    return from_dict(json.loads(Path(path).read_text()))


def to_dict(spec: Spec) -> dict:
    return json.loads(spec.to_json())
