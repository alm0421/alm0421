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
            names = portfolio.tickers_in(spec.tree, index_universes=False) if isinstance(spec.tree, dict) else []
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
    _corporate_action_note(spec, res)


def _corporate_action_note(spec: Spec, res: Result) -> None:
    """List the reconciled corporate-action days (data.reconcile_actions) of the tickers actually held."""
    from . import data
    try:
        hw = res.holdings
        eq = res.equity
        if hw is None or hw.empty or not len(eq):
            return
        held = [t for t in hw.columns if (hw[t] != 0).any()]
        n = data.corporate_action_note(held, eq.index[0], eq.index[-1])
        if n and n not in spec.notes:
            spec.notes.append(n)
        n = data.distribution_note(held, eq.index[0], eq.index[-1])
        if n and n not in spec.notes:
            spec.notes.append(n)
        # SIM series held during their model period: that period is net of an estimated fee/cost drag
        n = data.sim_drag_note([t for t in held if data.is_sim(t)], eq.index[0], eq.index[-1])
        if n and n not in spec.notes:
            spec.notes.append(n)
    except Exception:  # noqa: BLE001 - a note must never break a backtest
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
