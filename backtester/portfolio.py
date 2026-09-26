"""Allocation ("portfolio") strategies: target weights, rebalancing, regime switches, rotations.

A portfolio is a tree of nodes (JSON-compatible dicts), in the spirit of Composer symphonies and
Portfolio Visualizer tactical models:

  {"asset": "SPY"}                                   hold one ticker
  {"cash": true}                                     hold cash (earns the T-bill rate)
  {"weights": "specified", "w": [0.6, 0.4], "children": [...]}
  {"weights": "equal" | "inverse_vol" | "market_cap", "lookback": 20, "children": [...]}
  {"if": "close > sma(close, 200)", "on": "SPY", "then": node, "else": node}
  {"filter": {"select": "top" | "bottom", "n": 5, "by": "ret(63)",
              "require": "ret(252) > 0", "weights": "equal" | "inverse_vol" | "market_cap"},
   "universe": "NDX" | ["QQQ", "SPY", ...] | "children", "children": [...], "fallback": node}

A filter's children and a weights node's children may be any nodes (groups, if-nodes, nested filters).
Ranking metrics and non-equal weightings of a child that is not a single asset use the child's
synthetic daily NAV: the sub-portfolio simulated on its own, evaluated every day, no costs (as
Composer does for sub-symphonies). See `_Evaluator.nav`.

Targets are evaluated on each rebalance date using that day's close and traded at the close (or
the next open). Cash flows, dividends and interest are applied daily.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, fields
from typing import Literal

import numpy as np
import pandas as pd

from . import calendar as _cal
from . import data, expr
from .engine import Result, _daily_rate

FREQS = ("daily", "weekly", "monthly", "quarterly", "semiannual", "yearly", "none")
FLOW_FREQS = ("monthly", "quarterly", "semiannual", "yearly")
NAV_WARMUP = 504   # trading days simulated before the start for the synthetic NAVs of groups


@dataclass
class Portfolio:
    tree: dict
    rebalance: Literal["daily", "weekly", "monthly", "quarterly", "semiannual", "yearly", "none"] = "monthly"
    drift_band: float | None = None              # also rebalance when any weight drifts this far
    drift_band_relative: float | None = None     # ... or drifts this fraction of its own target (0.25: 40% -> 30%/50%)
    min_trade: float = 0.0                       # skip trades smaller than this fraction of equity
    fill: Literal["close", "next_open"] = "close"
    capital: float = 10_000.0
    start: str | None = None
    end: str | None = None
    slippage_bps: float = 0.0
    commission: float = 0.0                      # $ per order
    commission_pct: float = 0.0
    cash_rate: str | float | None = "tbill"
    reinvest_dividends: bool = True
    fractional_shares: bool = True
    point_in_time: bool = True
    # cash flows
    contribution: float = 0.0                    # $ added each period
    contribution_freq: Literal["monthly", "quarterly", "semiannual", "yearly"] = "monthly"
    withdrawal: float = 0.0                      # $ withdrawn each period
    withdrawal_pct: float = 0.0                  # fraction of the balance withdrawn each period
    withdrawal_freq: Literal["monthly", "quarterly", "semiannual", "yearly"] = "yearly"
    inflation_adjust: bool = False               # grow $ contributions/withdrawals with CPI
    # flow windows: a year (2030 = 1 Jan 2030), "YYYY-MM-DD", or N < 1900 = year N of the backtest
    # (contribution_end=20: the first 20 years; withdrawal_start=21: from the 21st year on). Inclusive.
    contribution_start: int | str | None = None
    contribution_end: int | str | None = None
    withdrawal_start: int | str | None = None
    withdrawal_end: int | str | None = None
    contribution_growth: float = 0.0             # annual step-up of $ contributions, counted from their start
    withdrawal_growth: float = 0.0               # annual step-up of $ withdrawals (on top of any CPI indexing)
    leverage: float = 1.0                        # scale every target weight; the excess is borrowed
    margin_rate: float = 0.0                     # extra annual rate paid on borrowed cash (above T-bills)
    maintenance_margin: float = 0.25             # with leverage or shorts: equity / gross exposure below this at
                                                 # a close -> margin call, cut pro rata back to target leverage
    expense_ratio: float = 0.0                   # annual fee on invested assets, charged daily
    benchmark: str | None = None                 # comparison ticker for alpha/beta (default SPY)
    name: str = ""
    description: str = ""
    notes: list[str] = field(default_factory=list)
    kind: str = "allocation"

    def validate(self) -> None:
        if self.rebalance not in FREQS:
            raise ValueError(f"rebalance must be one of {FREQS}")
        validate_node(self.tree)
        if not (0 < self.leverage <= 10):
            raise ValueError("leverage must be between 0 and 10")
        if not 0 <= self.maintenance_margin < 1:
            raise ValueError("maintenance_margin must be at least 0 and below 1")
        if self.maintenance_margin > 1 / self.leverage + 1e-12:
            raise ValueError(f"maintenance_margin ({self.maintenance_margin:.0%}) is above the initial margin of "
                             f"{self.leverage:g}x leverage ({1 / self.leverage:.0%}): every close would be a margin call. "
                             f"Use at most {1 / self.maintenance_margin:g}x, or a lower maintenance_margin "
                             "(0 turns margin calls off).")
        for f in ("contribution_freq", "withdrawal_freq"):
            if getattr(self, f) not in FLOW_FREQS:
                raise ValueError(f"{f} must be one of {FLOW_FREQS}")
        for f in ("drift_band", "drift_band_relative"):
            v = getattr(self, f)
            if v is not None and not v > 0:
                raise ValueError(f"{f} must be above 0 (or None)")
        for f in ("contribution_start", "contribution_end", "withdrawal_start", "withdrawal_end"):
            _flow_bound(getattr(self, f), pd.Timestamp("2000-01-03"), f.endswith("end"), f)
        for f in ("contribution_growth", "withdrawal_growth"):
            if not -1 < float(getattr(self, f)) < 10:
                raise ValueError(f"{f} is an annual fraction (0.03 = 3% a year)")
        if self.capital <= 0 and self.contribution <= 0:
            raise ValueError("need starting capital or contributions")

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=lambda o: f"<python function {getattr(o, '__name__', 'custom')}>")

    @classmethod
    def from_dict(cls, d: dict) -> "Portfolio":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown portfolio field(s): {', '.join(sorted(unknown))}")
        return cls(**d)

    @property
    def universe(self) -> list[str]:
        return tickers_in(self.tree)

    def summary(self) -> str:
        lines = ["Portfolio:"] + ["  " + l for l in describe(self.tree)]
        rb = {"none": "never rebalanced (buy and hold)", "daily": "re-evaluated and rebalanced daily",
              "semiannual": "re-evaluated and rebalanced every six months (end of June and December)"}.get(
            self.rebalance, f"re-evaluated and rebalanced {self.rebalance}")
        bands = []
        if self.drift_band:
            bands.append(f"drifts {self.drift_band:.0%} from its target")
        if self.drift_band_relative:
            bands.append(f"drifts by {self.drift_band_relative:.0%} of its own target weight")
        if bands:
            b = " or ".join(bands)
            if self.rebalance == "none":
                rb = f"rebalanced only when a holding {b}"
            else:
                rb += f", plus whenever a holding {b} (scheduled trades are skipped while every holding is within the band)"
        lines.append(f"Rebalancing: {rb}; trades at the {'close' if self.fill == 'close' else 'next open'}")
        cf = []
        if self.contribution:
            cf.append(f"add ${self.contribution:,.0f} {self.contribution_freq}"
                      + _window_text(self.contribution_start, self.contribution_end)
                      + (f", growing {self.contribution_growth:.1%} a year" if self.contribution_growth else ""))
        if self.withdrawal:
            cf.append(f"withdraw ${self.withdrawal:,.0f} {self.withdrawal_freq}"
                      + _window_text(self.withdrawal_start, self.withdrawal_end)
                      + (f", growing {self.withdrawal_growth:.1%} a year" if self.withdrawal_growth else ""))
        if self.withdrawal_pct:
            cf.append(f"withdraw {self.withdrawal_pct:.1%} of the balance {self.withdrawal_freq}"
                      + ("" if self.withdrawal else _window_text(self.withdrawal_start, self.withdrawal_end)))
        if cf and self.inflation_adjust:
            grow = (self.contribution and self.contribution_growth) or (self.withdrawal and self.withdrawal_growth)
            cf.append("$ amounts also grow with inflation (CPI)" + (", the growth rates compounding on top" if grow else ""))
        lines.append(f"Money: ${self.capital:,.0f} start" + ("; " + ", ".join(cf) if cf else "")
                     + ("; dividends reinvested" if self.reinvest_dividends else "; dividends kept as cash"))
        costs = []
        if self.commission:
            costs.append(f"${self.commission:g}/order")
        if self.commission_pct:
            costs.append(f"{self.commission_pct:.3%} of value")
        if self.slippage_bps:
            costs.append(f"{self.slippage_bps:g} bps slippage/side")
        if self.expense_ratio:
            costs.append(f"{self.expense_ratio:.2%}/yr expense ratio")
        if self.leverage != 1:
            costs.append(f"{self.leverage:g}x leverage, borrowing at the T-bill rate"
                         + (f" + {self.margin_rate:.2%}" if self.margin_rate else ""))
        elif self.margin_rate:
            costs.append(f"margin rate T-bills + {self.margin_rate:.2%} on any borrowing")
        if self.leverage > 1 or _has_short(self.tree):
            costs.append(f"{self.maintenance_margin:.0%} maintenance margin (margin calls cut positions at the close)"
                         if self.maintenance_margin else "no margin calls")
        lines.append("Costs: " + (", ".join(costs) if costs else "none"))
        cr = self.cash_rate
        lines.append("Cash: " + ("earns the 3-month T-bill rate" if cr == "tbill" else
                                 f"earns {float(cr):.2%}/yr" if cr else "earns nothing"))
        if self.start or self.end:
            lines.append(f"Period: {self.start or 'start of data'} to {self.end or 'latest'}")
        return "\n".join(lines)


# ------------------------------------------------------------------ tree helpers

WEIGHTINGS = ("equal", "inverse_vol", "market_cap", "risk_parity", "min_variance", "max_sharpe", "max_diversification")


def _wlabel(kind: str, lookback: int | None = None) -> str:
    base = {"equal": "equal weight", "inverse_vol": "inverse volatility", "market_cap": "market-cap weight",
            "risk_parity": "risk parity (equal risk contribution)", "min_variance": "minimum variance",
            "max_sharpe": "maximum Sharpe", "max_diversification": "maximum diversification"}[kind]
    if kind in ("equal", "market_cap"):
        return base
    return f"{base} ({lookback or (20 if kind == 'inverse_vol' else 60)}-day)"


def optimal_weights(method: str, R: np.ndarray) -> np.ndarray | None:
    """Long-only weights from a (days x assets) matrix of daily returns. None if it cannot be computed."""
    from scipy.optimize import minimize
    R = R[np.isfinite(R).all(axis=1)]
    k = R.shape[1]
    if k == 1:
        return np.ones(1)
    if len(R) < max(10, k + 2):
        return None
    cov = np.cov(R, rowvar=False) * 252
    cov = cov + np.eye(k) * 1e-10
    sd = np.sqrt(np.diag(cov))
    mu = R.mean(axis=0) * 252
    x0 = np.ones(k) / k
    bounds = [(0.0, 1.0)] * k
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    if method == "min_variance":
        fun = lambda w: w @ cov @ w  # noqa: E731
    elif method == "max_sharpe":
        fun = lambda w: -(w @ mu) / np.sqrt(max(w @ cov @ w, 1e-16))  # noqa: E731
    elif method == "max_diversification":
        fun = lambda w: -(w @ sd) / np.sqrt(max(w @ cov @ w, 1e-16))  # noqa: E731
    elif method == "risk_parity":
        def fun(w):
            v = w @ cov @ w
            rc = w * (cov @ w) / max(v, 1e-16)
            return float(((rc - 1 / k) ** 2).sum()) * 1e4
        x0 = (1 / sd) / (1 / sd).sum()
    else:
        return None
    r = minimize(fun, x0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 200, "ftol": 1e-12})
    w = np.clip(r.x if r.success or np.isfinite(r.x).all() else x0, 0, None)
    return w / w.sum() if w.sum() > 0 else None

NODE_KEYS = {  # allowed keys per node type ("name" is an optional label on any node)
    "asset": {"asset"},
    "cash": {"cash"},
    "custom": {"custom", "tickers"},
    "weights": {"weights", "w", "children", "lookback", "allow_short"},
    "if": {"if", "on", "then", "else"},
    "filter": {"filter", "universe", "children", "fallback"},
}
FILTER_KEYS = {"select", "n", "by", "require", "weights", "lookback"}


def _node_type(n: dict) -> str | None:
    for k in ("asset", "custom", "weights", "if", "filter"):
        if k in n:
            return k
    if n.get("cash"):
        return "cash"
    return None


def _kids(n: dict) -> list[dict]:
    """Direct sub-nodes of a node (children, then/else, fallback)."""
    out = list(n.get("children") or []) if ("weights" in n or "filter" in n) else []
    if "if" in n:
        out += [n["then"], n["else"]]
    if "filter" in n and n.get("fallback"):
        out.append(n["fallback"])
    return out


def _is_asset(k: dict) -> bool:
    return isinstance(k, dict) and "asset" in k


def _has_short(n: dict) -> bool:
    if not isinstance(n, dict):
        return False
    if n.get("weights") == "specified" and any(w < 0 for w in n.get("w") or []):
        return True
    return any(_has_short(k) for k in _kids(n))


def _needs_nav(n: dict) -> bool:
    """Does any filter / weighting rank or weigh a child that is not a single asset?"""
    if "filter" in n and n.get("universe", "children") == "children" and not all(map(_is_asset, n.get("children") or [])):
        return True
    if "weights" in n and n["weights"] not in ("equal", "specified") and not all(map(_is_asset, n["children"])):
        return True
    return any(_needs_nav(k) for k in _kids(n))


def _flow_bound(v, cal0: pd.Timestamp, end: bool, name: str = "flow date") -> pd.Timestamp | None:
    """A flow-window bound as an inclusive date. A year means 1 Jan of that year; N < 1900 means year N of
    the backtest (a start on the first day of year N, an end on the last day of year N)."""
    if v is None or v == "":
        return None
    if isinstance(v, str) and re.fullmatch(r"\s*\d+\s*", v):
        v = int(v)
    if isinstance(v, bool):
        raise ValueError(f"{name}: expected a year, a date or a year number of the backtest")
    if isinstance(v, (int, float, np.integer)):
        k = int(v)
        if k < 1:
            raise ValueError(f"{name}: year numbers of the backtest start at 1")
        if k < 1900:
            return cal0 + pd.DateOffset(years=k) - pd.Timedelta(days=1) if end else cal0 + pd.DateOffset(years=k - 1)
        return pd.Timestamp(year=k, month=1, day=1)
    try:
        return pd.Timestamp(v)
    except (ValueError, TypeError) as e:
        raise ValueError(f"{name}: can't read {v!r} as a date") from e


def _window_text(start, end) -> str:
    def norm(v):
        if isinstance(v, str) and v.strip().isdigit():
            return int(v)
        return v

    start, end = norm(start), norm(end)
    parts = []
    if start not in (None, ""):
        if isinstance(start, (int, np.integer)):
            parts.append(f"from year {start} of the backtest" if start < 1900 else f"from 1 Jan {start}")
        else:
            parts.append(f"from {start}")
    if end not in (None, ""):
        if isinstance(end, (int, np.integer)):
            if end < 1900:
                parts.append(f"for the first {end} year{'s' if end != 1 else ''}" if start in (None, "")
                             else f"through year {end}")
            else:
                parts.append(f"until 1 Jan {end}")
        else:
            parts.append(f"until {end}")
    return (" " + " ".join(parts)) if parts else ""


def validate_node(n: dict, depth: int = 0) -> None:
    if depth > 20:
        raise ValueError("portfolio tree is nested too deeply")
    if not isinstance(n, dict):
        raise ValueError(f"portfolio node must be an object, got {n!r}")
    kind = _node_type(n)
    if kind is None:
        raise ValueError(f"unrecognised portfolio node {n!r}: a node needs one of 'asset', 'cash', 'weights', "
                         "'if', 'filter' (or 'custom' from Python)")
    extra = set(n) - NODE_KEYS[kind] - {"name"}
    if kind != "cash" and "cash" in extra and not n.get("cash"):
        extra.discard("cash")  # {"cash": false} alongside another type is harmless
    if extra:
        raise ValueError(f"unknown key(s) {', '.join(repr(k) for k in sorted(extra))} in {kind} node {_short(n)}; "
                         f"allowed: {', '.join(sorted(NODE_KEYS[kind] | {'name'}))}")
    if kind == "asset":
        data.load(n["asset"])  # raises DataError for unknown tickers
    elif kind == "custom":
        if not callable(n["custom"]) or not n.get("tickers"):
            raise ValueError("custom node needs a Python callable and a 'tickers' list")
        for t in n["tickers"]:
            data.load(t)
    elif kind == "weights":
        kids = n.get("children") or []
        if not isinstance(kids, list) or not kids:
            raise ValueError("weights node needs children")
        if n["weights"] == "specified":
            w = n.get("w")
            if not w or len(w) != len(kids):
                raise ValueError("specified weights need one weight per child")
            if abs(sum(w) - 1) > 1e-6:
                raise ValueError(f"weights must add up to 100% (got {sum(w):.1%})")
            if any(x < 0 for x in w) and not n.get("allow_short", True):
                raise ValueError("negative weights are not allowed here")
        elif n["weights"] not in WEIGHTINGS:
            raise ValueError(f"unknown weighting {n['weights']!r} (one of specified, {', '.join(WEIGHTINGS)})")
        if n.get("lookback") is not None and int(n["lookback"]) < 2:
            raise ValueError("lookback must be at least 2 days")
        for k in kids:
            validate_node(k, depth + 1)
    elif kind == "if":
        for k in ("then", "else"):
            if k not in n:
                raise ValueError(f"if node needs '{k}'")
        expr.compile_expr(n["if"])
        data.load(n.get("on", "SPY"))
        validate_node(n["then"], depth + 1)
        validate_node(n["else"], depth + 1)
    elif kind == "filter":
        f = n["filter"]
        if not isinstance(f, dict):
            raise ValueError("'filter' must be an object with select, n, by, ...")
        bad = set(f) - FILTER_KEYS
        if bad:
            raise ValueError(f"unknown key(s) {', '.join(repr(k) for k in sorted(bad))} in filter settings; "
                             f"allowed: {', '.join(sorted(FILTER_KEYS))}")
        if not f.get("by"):
            raise ValueError("filter needs 'by' (the ranking metric, e.g. tret(tr, 63))")
        expr.compile_expr(f["by"])
        if f.get("require"):
            expr.compile_expr(f["require"])
        if int(f.get("n", 1)) < 1:
            raise ValueError("filter n must be at least 1")
        if f.get("select", "top") not in ("top", "bottom"):
            raise ValueError("filter select must be 'top' or 'bottom'")
        if f.get("weights", "equal") not in WEIGHTINGS:
            raise ValueError(f"unknown filter weighting {f['weights']!r} (one of {', '.join(WEIGHTINGS)})")
        u = n.get("universe", "children")
        if u == "children":
            kids = n.get("children") or []
            if not kids:
                raise ValueError("filter over its children needs children")
            for k in kids:
                validate_node(k, depth + 1)
        elif not (u in ("NDX", "nasdaq100") or (isinstance(u, list) and u)):
            raise ValueError("filter universe must be 'children', 'NDX' or a list of tickers")
        elif n.get("children"):
            raise ValueError("a filter with a ticker/index universe takes no children (use universe 'children')")
        if n.get("fallback"):
            validate_node(n["fallback"], depth + 1)


def _short(n: dict) -> str:
    s = json.dumps({k: v for k, v in n.items() if k not in ("children", "then", "else", "fallback")}, default=str)
    return s if len(s) < 80 else s[:77] + "..."


def _universe(n: dict) -> list[str]:
    """Tickers a filter ranks directly (for universe 'children': its single-asset children)."""
    u = n.get("universe", "children")
    if u == "children":
        return [data.canonical(k["asset"]) for k in n.get("children") or [] if _is_asset(k)]
    if u in ("NDX", "nasdaq100"):
        return data.nasdaq100_ever()
    return [data.canonical(t) for t in u]


def tickers_in(n: dict) -> list[str]:
    out: list[str] = []

    def add(t):
        t = data.canonical(t)
        if t not in out:
            out.append(t)

    def walk(x):
        if "asset" in x:
            add(x["asset"])
        elif "custom" in x:
            for t in x["tickers"]:
                add(t)
        elif "weights" in x:
            for k in x["children"]:
                walk(k)
        elif "if" in x:
            add(x.get("on", "SPY"))
            walk(x["then"])
            walk(x["else"])
        elif "filter" in x:
            if x.get("universe", "children") == "children":
                for k in x.get("children") or []:
                    walk(k)
            else:
                for t in _universe(x):
                    add(t)
            if x.get("fallback"):
                walk(x["fallback"])
    walk(n)
    return out


def fixed_tickers(n: dict) -> list[str]:
    """Tickers that must have data for the portfolio to be defined (excludes index universes)."""
    out: list[str] = []

    def walk(x):
        if "asset" in x:
            out.append(data.canonical(x["asset"]))
        elif "custom" in x:
            out.extend(data.canonical(t) for t in x["tickers"])
        elif "weights" in x:
            for k in x["children"]:
                walk(k)
        elif "if" in x:
            out.append(data.canonical(x.get("on", "SPY")))
            walk(x["then"])
            walk(x["else"])
        elif "filter" in x:
            u = x.get("universe", "children")
            if u == "children":
                for k in x.get("children") or []:
                    walk(k)
            elif isinstance(u, list):
                out.extend(_universe(x))
            if x.get("fallback"):
                walk(x["fallback"])
    walk(n)
    return list(dict.fromkeys(out))


def describe(n: dict, indent: int = 0) -> list[str]:
    pad = "  " * indent
    if "asset" in n:
        return [f"{pad}{data.canonical(n['asset'])}"]
    if n.get("cash"):
        return [f"{pad}cash (T-bills)"]
    if "custom" in n:
        return [f"{pad}python function {getattr(n['custom'], '__name__', 'custom')}({', '.join(n['tickers'])})"]
    if "weights" in n:
        kind = n["weights"]
        lines = []
        if kind == "specified":
            for w, k in zip(n["w"], n["children"]):
                sub = describe(k, indent + 1)
                if len(sub) == 1:
                    lines.append(f"{pad}{w:.0%} {sub[0].strip()}")
                else:
                    lines.append(f"{pad}{w:.0%}:")
                    lines.extend(sub)
            return lines
        groups = not all(map(_is_asset, n["children"]))
        if kind == "market_cap" and groups:
            label = "equal weight (market-cap weighting needs single assets)"
        else:
            label = _wlabel(kind, n.get("lookback"))
            if groups and kind != "equal":
                label += ", groups measured on their simulated daily returns"
        lines.append(f"{pad}{label} of:")
        for k in n["children"]:
            lines.extend(describe(k, indent + 1))
        return lines
    if "if" in n:
        return ([f"{pad}if {n['if']} (on {data.canonical(n.get('on', 'SPY'))}):"] + describe(n["then"], indent + 1)
                + [f"{pad}otherwise:"] + describe(n["else"], indent + 1))
    if "filter" in n:
        f = n["filter"]
        u = n.get("universe", "children")
        kids = n.get("children") or []
        groups = u == "children" and not all(map(_is_asset, kids))
        wk = f.get("weights", "equal")
        wl = _wlabel(wk, f.get("lookback"))
        if groups and wk == "market_cap":
            wl = "equal weight (market-cap weighting needs single assets)"
        if groups:
            out = [f"{pad}{f.get('select', 'top')} {f.get('n', 1)} of these by {f['by']} "
                   f"(groups ranked on their simulated daily NAV), {wl}:"]
            for k in kids:
                out.extend(describe(k, indent + 2))
        else:
            uname = "Nasdaq-100 members (point-in-time)" if u in ("NDX", "nasdaq100") else ", ".join(_universe(n))
            out = [f"{pad}{f.get('select', 'top')} {f.get('n', 1)} of [{uname}] by {f['by']}, {wl}"]
        if f.get("require"):
            out.append(f"{pad}  only if {f['require']}, else:")
            out.extend(describe(n.get("fallback") or {"cash": True}, indent + 2))
        elif n.get("fallback"):
            out.append(f"{pad}  if nothing qualifies:")
            out.extend(describe(n["fallback"], indent + 2))
        return out
    return [f"{pad}{n}"]


# ------------------------------------------------------------------ evaluation

OPTIMISERS = ("risk_parity", "min_variance", "max_sharpe", "max_diversification")


class _Evaluator:
    """Evaluates the tree to target weights at the close of day i of `cal`.

    `cal` may start before the backtest (`off` = index of its first day), so that the synthetic NAVs
    of groups have history to warm up on. A "member" of a filter or weighting is either a ticker
    (a single-asset child, measured on its own price data) or a node (any other child, measured on
    its synthetic NAV, see `nav`)."""

    def __init__(self, p: Portfolio, cal: pd.DatetimeIndex, dfs: dict[str, pd.DataFrame], off: int = 0):
        self.p, self.cal, self.dfs, self.off = p, cal, dfs, off
        self.ns = {t: expr.Namespace(df, ticker=t) for t, df in dfs.items()}
        self.cache: dict = {}
        self.close = {t: df["close"].reindex(cal).to_numpy() for t, df in dfs.items()}
        self._rets: dict = {}
        self.members = None
        self._nav: dict = {}       # id(node) -> synthetic NAV over cal
        self._rows: dict = {}      # id(node) -> its target weights on every day (memo from the NAV simulation)
        self._nav_ns: dict = {}
        self._keep: list = []      # nodes whose id() is a cache key stay alive
        self._rate = None
        self._quiet = 0            # > 0 while simulating a NAV: no notes (the real evaluation adds its own)

    # ------------------------------------------------------------ data
    def series(self, rule: str, t: str, kind: str) -> np.ndarray:
        key = (rule, t, kind)
        if key not in self.cache:
            ns = self.ns[t]
            if kind == "bool":
                s = expr.evaluate(rule, ns).reindex(self.cal, fill_value=False).to_numpy()
            else:
                s = expr.evaluate_value(rule, ns).reindex(self.cal).to_numpy()
            self.cache[key] = s
        return self.cache[key]

    def mseries(self, rule: str, m, kind: str) -> np.ndarray:
        """A rule evaluated on a member: a ticker's own data, or a node's synthetic NAV."""
        if isinstance(m, str):
            return self.series(rule, m, kind)
        key = (rule, id(m), kind)
        if key not in self.cache:
            ns = self._nav_namespace(m)
            if kind == "bool":
                s = expr.evaluate(rule, ns).to_numpy()
            else:
                s = expr.evaluate_value(rule, ns).to_numpy()
            self.cache[key] = s
        return self.cache[key]

    def vol(self, t: str, n: int) -> np.ndarray:
        return self.series(f"volatility({int(n)})", t, "value")

    def mcap(self, t: str) -> np.ndarray:
        key = ("mcap", t)
        if key not in self.cache:
            sh = data.shares_outstanding(t)
            if sh.empty:
                self.cache[key] = np.full(len(self.cal), np.nan)
            else:
                s = sh.reindex(self.cal.union(sh.index)).ffill().reindex(self.cal)
                self.cache[key] = (s * self.dfs[t]["close"].reindex(self.cal)).to_numpy()
        return self.cache[key]

    def is_member(self, t: str, i: int) -> bool:
        if self.members is None:
            names = data.nasdaq100_ever()
            m, _ = data.member_mask(names, self.cal)
            self.members = {n: m[:, j] for j, n in enumerate(names)}
            cov = data.coverage_note(str(self.cal[min(self.off, len(self.cal) - 1)].date()), str(self.cal[-1].date()))
            if cov:
                self.note(cov)
        return bool(self.members.get(t, np.zeros(len(self.cal), bool))[i])

    def has(self, t: str, i: int) -> bool:
        """Has the ticker started trading by bar i? A missing price on a day it has already traded
        (another calendar's holiday) uses its last price rather than dropping it to cash."""
        c = self.close[t]
        if np.isfinite(c[i]):
            return True
        if not hasattr(self, "_first"):
            self._first = {}
        if t not in self._first:
            ok = np.flatnonzero(np.isfinite(c))
            self._first[t] = int(ok[0]) if len(ok) else len(c)
        return i > self._first[t] and i - self._first[t] > 0 and np.isfinite(c[max(0, i - 10): i]).any()

    def rets(self, t: str) -> np.ndarray:
        if t not in self._rets:
            df = self.dfs[t]
            x = df["adj_close"] if "adj_close" in df else df["close"]
            self._rets[t] = x.pct_change().reindex(self.cal).to_numpy()
        return self._rets[t]

    def rate(self) -> np.ndarray:
        if self._rate is None:
            self._rate = _daily_rate(self.cal, self.p.cash_rate)
        return self._rate

    # ------------------------------------------------------------ synthetic NAV of a sub-tree
    def nav(self, n: dict) -> np.ndarray:
        """The daily NAV of sub-tree `n` held on its own (NaN before all its fixed tickers trade).

        Every close it is evaluated like the real tree (data up to that close) and the resulting
        weights earn the next day's total returns (adj_close); unallocated weight earns the cash rate.
        No costs, no leverage, rebalanced to target daily. NAV[t] therefore depends only on data up to t.
        The per-day weights are memoised, so evaluating `n` again later costs nothing."""
        k = id(n)
        if k in self._nav:
            return self._nav[k]
        self._keep.append(n)
        T = len(self.cal)
        self._quiet += 1
        try:
            rows = [self.eval(n, i) for i in range(T)]
        finally:
            self._quiet -= 1
        names = sorted({t for r in rows for t in r if t != "cash"})
        col = {t: j for j, t in enumerate(names)}
        W = np.zeros((T, len(names)))
        for i, r in enumerate(rows):
            for t, x in r.items():
                if t != "cash":
                    W[i, col[t]] += x
        R = np.column_stack([np.nan_to_num(self.rets(t)) for t in names]) if names else np.zeros((T, 0))
        cashw = 1.0 - W.sum(axis=1)
        rate = self.rate()
        r = np.zeros(T)
        if T > 1:
            r[1:] = (W[:-1] * R[1:]).sum(axis=1) + cashw[:-1] * rate[:-1]
        v = np.cumprod(1.0 + r)
        first = self._first_valid(n)
        v[:first] = np.nan
        if first < T:
            v = v / v[first]
        self._rows[k] = rows
        self._nav[k] = v
        return v

    def _first_valid(self, n: dict) -> int:
        first = 0
        for t in fixed_tickers(n):
            ok = np.flatnonzero(np.isfinite(self.close[t]))
            first = max(first, int(ok[0]) if len(ok) else len(self.cal))
        return first

    def _nav_namespace(self, n: dict) -> expr.Namespace:
        k = id(n)
        if k not in self._nav_ns:
            v = pd.Series(self.nav(n), index=self.cal)
            df = pd.DataFrame({"open": v, "high": v, "low": v, "close": v, "volume": 0.0, "adj_close": v,
                               "dividend": 0.0}, index=self.cal)
            self._nav_ns[k] = expr.Namespace(df)
        return self._nav_ns[k]

    def mrets(self, m) -> np.ndarray:
        if isinstance(m, str):
            return self.rets(m)
        key = ("navret", id(m))
        if key not in self.cache:
            v = self.nav(m)
            out = np.full(len(v), np.nan)
            out[1:] = v[1:] / v[:-1] - 1
            self.cache[key] = out
        return self.cache[key]

    # ------------------------------------------------------------ members
    @staticmethod
    def _member(k: dict):
        return data.canonical(k["asset"]) if _is_asset(k) else k

    def _live(self, m, i: int) -> bool:
        return self.has(m, i) if isinstance(m, str) else bool(np.isfinite(self.nav(m)[i]))

    def _expand(self, m, i: int) -> dict[str, float]:
        return {m: 1.0} if isinstance(m, str) else self.eval(m, i)

    def _weigh(self, method: str, members: list, i: int, lookback: int | None = None) -> list[float]:
        """Weights (aligned with `members`, adding up to 1) for a weighting method on day i."""
        k = len(members)
        eq = [1 / k] * k
        if method == "equal" or k == 0:
            return eq
        if method in OPTIMISERS:
            n = int(lookback or 60)
            lo = max(0, i - n + 1)
            R = np.column_stack([self.mrets(m)[lo: i + 1] for m in members])
            w = optimal_weights(method, R)
            if w is not None:
                return [float(x) if x > 1e-6 else 0.0 for x in w]
            self.note(f"Not enough history for {method.replace('_', ' ')} weights on some dates: equal-weighted then.")
            return eq
        lookback = lookback or 20
        if method == "inverse_vol":
            v = np.array([self.mseries(f"volatility({int(lookback)})", m, "value")[i] for m in members])
            ok = np.isfinite(v) & (v > 0)
            if ok.any():
                inv = np.where(ok, 1 / np.where(ok, v, 1), 0)
                return [float(x) for x in inv / inv.sum()]
            return eq
        if method == "market_cap":
            if not all(isinstance(m, str) for m in members):
                self.note("Market-cap weighting needs single assets: groups were equal-weighted.")
                return eq
            mc = np.array([self.mcap(t)[i] for t in members])
            ok = np.isfinite(mc) & (mc > 0)
            if ok.all():
                return [float(x) for x in mc / mc.sum()]
            self.note("Market-cap weights need share counts; tickers without them were equal-weighted.")
            if ok.any():
                out = list(eq)
                capw = mc[ok] / mc[ok].sum() * (ok.sum() / k)
                for j, w in zip(np.flatnonzero(ok), capw):
                    out[j] = float(w)
                return out
        return eq

    def note(self, msg: str) -> None:
        if not self._quiet and msg not in self.p.notes:
            self.p.notes.append(msg)

    # ------------------------------------------------------------ the tree
    def eval(self, n: dict, i: int) -> dict[str, float]:
        rows = self._rows.get(id(n))
        if rows is not None:
            return dict(rows[i])
        if "asset" in n:
            t = data.canonical(n["asset"])
            if self.has(t, i):
                return {t: 1.0}
            if i >= self.off:
                self.note(f"{t} had no price yet on some rebalance dates (before its history starts); its slice was held in cash then.")
            return {"cash": 1.0}
        if n.get("cash"):
            return {"cash": 1.0}
        if "custom" in n:
            # Python API: fn(date, history) -> {ticker: weight}; history holds data up to and including date
            d = self.cal[i]
            hist = {t: self.dfs[data.canonical(t)].loc[:d] for t in n["tickers"]}
            w = n["custom"](d, hist) or {}
            tot = sum(max(v, 0) for v in w.values())
            if tot > 1 + 1e-9:
                raise ValueError(f"custom weights on {d.date()} add up to {tot:.2%}")
            out = {data.canonical(t): float(v) for t, v in w.items() if v > 0}
            if tot < 1 - 1e-6:
                out["cash"] = 1 - tot
            return out
        if "weights" in n:
            kids = n["children"]
            method = n["weights"]
            if method == "specified":
                ws = n["w"]
            elif method == "equal":
                ws = [1 / len(kids)] * len(kids)
            else:
                mem = [self._member(k) for k in kids]
                live = [j for j, m in enumerate(mem) if self._live(m, i)]
                if live:
                    ws = [0.0] * len(kids)
                    for j, w in zip(live, self._weigh(method, [mem[j] for j in live], i, n.get("lookback"))):
                        ws[j] = w
                else:
                    ws = [1 / len(kids)] * len(kids)
            out: dict[str, float] = {}
            for w, k in zip(ws, kids):
                if w == 0:
                    continue
                for t, x in self.eval(k, i).items():
                    out[t] = out.get(t, 0.0) + w * x
            return out
        if "if" in n:
            on = data.canonical(n.get("on", "SPY"))
            cond = self.series(n["if"], on, "bool")[i]
            return self.eval(n["then"] if cond else n["else"], i)
        if "filter" in n:
            f = n["filter"]
            u = n.get("universe", "children")
            mem = [self._member(k) for k in n.get("children") or []] if u == "children" else _universe(n)
            pit = u in ("NDX", "nasdaq100") and self.p.point_in_time
            cands = []
            for m in mem:
                if isinstance(m, str) and (not self.has(m, i) or (pit and not self.is_member(m, i))):
                    continue
                v = self.mseries(f["by"], m, "value")[i]
                if np.isfinite(v):
                    cands.append((v, m))
            cands.sort(key=lambda x: x[0], reverse=f.get("select", "top") == "top")
            chosen = [m for _, m in cands[: int(f.get("n", 1))]]
            if f.get("require"):
                passed = [m for m in chosen if self.mseries(f["require"], m, "bool")[i]]
            else:
                passed = chosen
            if not passed:
                return self.eval(n.get("fallback") or {"cash": True}, i)
            w: dict[str, float] = {}
            for m, x in zip(passed, self._weigh(f.get("weights", "equal"), passed, i, f.get("lookback"))):
                if x <= 0:
                    continue
                for t, y in self._expand(m, i).items():
                    w[t] = w.get(t, 0.0) + x * y
            if f.get("require") and len(passed) < len(chosen):
                # slots whose pick failed the requirement go to the fallback
                share = len(passed) / len(chosen)
                out = {t: x * share for t, x in w.items()}
                for t, x in self.eval(n.get("fallback") or {"cash": True}, i).items():
                    out[t] = out.get(t, 0.0) + x * (1 - share)
                return out
            return w
        raise ValueError(f"bad node {n}")


def _period_ids(idx: pd.DatetimeIndex, freq: str):
    if freq == "semiannual":
        return np.asarray(idx.year * 2 + (idx.month > 6).astype(int))
    code = {"weekly": "W-FRI", "monthly": "M", "quarterly": "Q", "yearly": "Y"}[freq]
    return idx.to_period(code)


def _schedule(cal: pd.DatetimeIndex, freq: str) -> np.ndarray:
    """True on the last trading day of each period (daily: every day; none: only the first day)."""
    T = len(cal)
    if freq == "daily":
        return np.ones(T, bool)
    out = np.zeros(T, bool)
    if freq == "none":
        out[0] = True
        return out
    per = _period_ids(cal, freq)
    last = pd.Series(np.arange(T), index=cal).groupby(np.asarray(per)).max().to_numpy()
    out[last] = True
    # the latest bar only ends its period if the next NYSE session starts a new one
    if _period_ids(_cal.next_sessions(cal[-1]), freq)[0] == per[-1]:
        out[-1] = False
    out[0] = True  # initial allocation
    return out


def _period_starts(cal: pd.DatetimeIndex, freq: str) -> np.ndarray:
    per = _period_ids(cal, freq)
    first = pd.Series(np.arange(len(cal)), index=cal).groupby(np.asarray(per)).min().to_numpy()
    out = np.zeros(len(cal), bool)
    out[first] = True
    out[0] = False  # the starting capital covers the first period
    return out


def _flow_schedule(cal: pd.DatetimeIndex, freq: str, start, end, growth: float, what: str) -> tuple[np.ndarray, np.ndarray]:
    """Flow days (period starts inside the window) and the growth multiplier (1+g)^(whole years since the
    window opened) of each day."""
    days = _period_starts(cal, freq)
    s = _flow_bound(start, cal[0], False, f"{what}_start")
    e = _flow_bound(end, cal[0], True, f"{what}_end")
    if s is not None:
        days &= np.asarray(cal >= s)
    if e is not None:
        days &= np.asarray(cal <= e)
    mult = np.ones(len(cal))
    if growth:
        s0 = s if s is not None and s > cal[0] else cal[0]
        yrs = np.asarray(cal.year - s0.year - ((cal.month < s0.month) | ((cal.month == s0.month) & (cal.day < s0.day))))
        mult = (1 + growth) ** np.maximum(yrs, 0)
    return days, mult


def run(p: Portfolio) -> Result:
    p.validate()
    names = tickers_in(p.tree)
    dfs = data.load_many(names)
    must = fixed_tickers(p.tree)
    cal = None
    for df in dfs.values():
        cal = df.index if cal is None else cal.union(df.index)
    cal_all = cal
    first_common = max(dfs[t].index[0] for t in must) if must else cal[0]
    start = pd.Timestamp(p.start) if p.start else first_common
    if p.start and pd.Timestamp(p.start) < first_common:
        late = [t for t in must if dfs[t].index[0] == first_common]
        p.notes.append(f"Start moved to {first_common.date()}, when {', '.join(late)} began trading.")
        start = first_common
    cal = cal[cal >= start]
    if p.end:
        cal = cal[cal <= pd.Timestamp(p.end)]
    # start on a day every fixed holding actually traded (not, say, a stock-market holiday on
    # which only a crypto or foreign series has a row)
    if must and len(cal):
        both = np.ones(len(cal), bool)
        for t in must:
            both &= np.isin(cal.values, dfs[t].index.values)
        if both.any():
            cal = cal[int(np.argmax(both)):]
    if len(cal) < 2:
        raise ValueError("no price data in the requested period")
    # rule indicators come from each ticker's full history; synthetic NAVs of groups (filters or weightings
    # over sub-trees) are simulated from NAV_WARMUP sessions before the start so they are warm on day one
    full = cal_all[cal_all <= cal[-1]]
    off = int(full.searchsorted(cal[0]))
    lo = max(0, off - NAV_WARMUP) if _needs_nav(p.tree) else off
    ev = _Evaluator(p, full[lo:], dfs, off=off - lo)
    base = off - lo
    T = len(cal)
    tick = list(dfs)
    idx = {t: j for j, t in enumerate(tick)}
    N = len(tick)
    O = np.column_stack([dfs[t]["open"].reindex(cal).to_numpy() for t in tick])
    C = np.column_stack([ev.close[t][base:] for t in tick])
    DIV = np.column_stack([dfs[t]["dividend"].reindex(cal).fillna(0.0).to_numpy() for t in tick])
    rate = _daily_rate(cal, p.cash_rate)
    borrow_extra = p.margin_rate / 252.0
    fee_daily = p.expense_ratio / 252.0
    fees = 0.0
    sched = _schedule(cal, p.rebalance)
    slip = p.slippage_bps / 1e4

    cpi = data.cpi()
    if p.inflation_adjust and not cpi.empty:
        c = cpi.reindex(cal.union(cpi.index)).ffill().reindex(cal)
        infl = (c / c.dropna().iloc[0]).fillna(1.0).to_numpy()
    else:
        if p.inflation_adjust:
            p.notes.append("CPI data unavailable: cash flows were not inflation-adjusted.")
        infl = np.ones(T)
    no_flows = (np.zeros(T, bool), np.ones(T))
    contrib_days, contrib_mult = (_flow_schedule(cal, p.contribution_freq, p.contribution_start, p.contribution_end,
                                                 p.contribution_growth, "contribution") if p.contribution else no_flows)
    wd_days, wd_mult = (_flow_schedule(cal, p.withdrawal_freq, p.withdrawal_start, p.withdrawal_end,
                                       p.withdrawal_growth, "withdrawal") if (p.withdrawal or p.withdrawal_pct) else no_flows)
    mm = p.maintenance_margin
    margin_days: list = []
    lev_peak = (0.0, 0.0, None)       # (gross/equity, target gross, date): worst drift above target between rebalances

    shares = np.zeros(N)
    cash = p.capital
    interest = 0.0
    last_px = np.full(N, np.nan)
    target: dict[str, float] = {}
    pending_target = None
    equity = np.zeros(T)
    flows = np.zeros(T)
    weights = np.zeros((T, N))
    cashw = np.zeros(T)
    orders: list[dict] = []
    # P&L attribution: every cash movement caused by a ticker (trades incl. costs, dividends,
    # reinvestment), and a ledger of those events in simulation order for the holding periods
    tcash = np.zeros(N)
    tdiv = np.zeros(N)
    tcom = np.zeros(N)
    ledger: list[tuple] = []          # (date, ticker, kind, shares, value, commission)
    turnover = 0.0
    n_rebal = 0

    def px_now(prices):
        return np.where(np.isfinite(prices), prices, last_px)

    def value(prices) -> float:
        pv = px_now(prices)
        return cash + float(np.nansum(shares * np.nan_to_num(pv)))

    def trade_to(tgt: dict[str, float], prices: np.ndarray, i: int, reason: str, min_trade: bool = True) -> None:
        nonlocal cash, turnover
        pv = px_now(prices)
        eq = value(prices)
        if eq <= 0:
            return
        # cash the target allows us to borrow (leverage or weights above 100%)
        borrow_ok = max(0.0, sum(w for t, w in tgt.items() if t != "cash") - 1.0) * eq
        want = np.zeros(N)
        for t, w in tgt.items():
            if t == "cash":
                continue
            j = idx[t]
            if np.isfinite(pv[j]) and pv[j] > 0:
                want[j] = eq * w / pv[j]
        if not p.fractional_shares:
            want = np.floor(want)
        delta = want - shares
        # sells first, then buys (scaled to the cash available)
        for sgn in (-1, 1):
            js = [j for j in range(N) if delta[j] * sgn > 1e-12 and np.isfinite(pv[j])]
            if sgn == 1 and js:
                need = sum(delta[j] * pv[j] * (1 + slip) * (1 + p.commission_pct) + p.commission for j in js)
                scale = min(1.0, max(cash + borrow_ok, 0) / need) if need > 0 else 1.0
            else:
                scale = 1.0
            for j in js:
                q = delta[j] * scale
                if min_trade and abs(q) * pv[j] < p.min_trade * eq:
                    continue
                if not p.fractional_shares:
                    q = np.floor(q) if q > 0 else -np.floor(-q)
                if q == 0:
                    continue
                fill = pv[j] * (1 + np.sign(q) * slip)
                com = p.commission + p.commission_pct * abs(q) * fill
                cash -= q * fill + com
                shares[j] += q
                tcash[j] -= q * fill + com
                tcom[j] += com
                ledger.append((cal[i], tick[j], "buy" if q > 0 else "sell", q, abs(q) * fill, com))
                turnover += abs(q) * fill / eq
                orders.append({"date": cal[i].date(), "ticker": tick[j], "side": "buy" if q > 0 else "sell",
                               "shares": abs(q), "price": fill, "value": abs(q) * fill, "commission": com,
                               "reason": reason})

    bands = bool(p.drift_band or p.drift_band_relative)

    def drifted(prices) -> bool:
        """Has any holding left its absolute (drift_band) or relative (drift_band_relative) band?"""
        eq = value(prices)
        if eq <= 0 or not target:
            return False
        pv = np.nan_to_num(px_now(prices))
        cur = {tick[j]: shares[j] * pv[j] / eq for j in range(N) if shares[j]}
        for t in set(cur) | {t for t in target if t != "cash"}:
            d = abs(cur.get(t, 0.0) - target.get(t, 0.0))
            if p.drift_band and d > p.drift_band:
                return True
            if p.drift_band_relative and d > max(p.drift_band_relative * abs(target.get(t, 0.0)), 1e-9):
                return True
        return False

    def gross_now(prices) -> float:
        return float(np.abs(shares * np.nan_to_num(px_now(prices))).sum())

    for i in range(T):
        o, c = O[i], C[i]
        # overnight interest and dividends
        if i > 0:
            r = rate[i - 1]
            earned = cash * r if cash >= 0 else cash * (r + borrow_extra)
            cash += earned
            interest += earned
            if fee_daily:
                held = float(np.nansum(np.abs(shares) * np.nan_to_num(last_px)))
                cash -= held * fee_daily
                fees += held * fee_daily
        div_cash = shares * DIV[i]
        got = float(np.nansum(div_cash))
        if got:
            cash += got
            for j in np.flatnonzero(div_cash != 0):
                tcash[j] += div_cash[j]
                tdiv[j] += div_cash[j]
                ledger.append((cal[i], tick[j], "dividend", 0.0, float(div_cash[j]), 0.0))
        # cash flows at the start of the day
        f = 0.0
        if contrib_days[i]:
            f += p.contribution * infl[i] * contrib_mult[i]
        if wd_days[i]:
            f -= p.withdrawal * infl[i] * wd_mult[i]
            if p.withdrawal_pct:
                f -= p.withdrawal_pct * max(value(np.where(np.isfinite(o), o, last_px)), 0)
        if f:
            cash += f
            flows[i] = f
        # next-open execution of yesterday's decision
        if pending_target is not None and p.fill == "next_open":
            trade_to(pending_target, o, i, "rebalance")
            pending_target = None
        # withdrawals that overdraw cash: sell proportionally at the close
        np.copyto(last_px, c, where=np.isfinite(c))
        eq_close = value(c)
        tgt_cash = 1.0 - sum(w for t, w in target.items() if t != "cash")
        if f < 0 and cash < min(0.0, tgt_cash) * eq_close - 1e-6 * max(eq_close, 1.0) and eq_close > 0 and target:
            trade_to(target, c, i, "raise cash")
        # invest new contributions at the close in the current target mix (no selling)
        if f > 0 and target and not sched[i]:
            pv = px_now(c)
            live = {t: w for t, w in target.items() if t != "cash" and np.isfinite(pv[idx[t]]) and pv[idx[t]] > 0}
            tot = sum(live.values())
            if tot > 0:
                budget = min(f, max(cash, 0.0))
                for t, w in live.items():
                    j = idx[t]
                    amt = budget * w / tot * (1 - p.commission_pct) / (1 + slip)
                    q = amt / pv[j] if p.fractional_shares else np.floor(amt / pv[j])
                    if q <= 0:
                        continue
                    fill = pv[j] * (1 + slip)
                    com = p.commission + p.commission_pct * q * fill
                    cash -= q * fill + com
                    shares[j] += q
                    tcash[j] -= q * fill + com
                    tcom[j] += com
                    ledger.append((cal[i], t, "buy", q, q * fill, com))
                    orders.append({"date": cal[i].date(), "ticker": t, "side": "buy", "shares": q, "price": fill,
                                   "value": q * fill, "commission": com, "reason": "contribution"})
        # reinvest dividends into the same holding at the close
        if got and p.reinvest_dividends:
            for j in np.flatnonzero(div_cash > 0):
                if np.isfinite(c[j]) and c[j] > 0 and cash > 0:
                    q = min(div_cash[j], cash) / c[j]
                    shares[j] += q
                    cash -= q * c[j]
                    tcash[j] -= q * c[j]
                    ledger.append((cal[i], tick[j], "reinvest", q, q * c[j], 0.0))
        # rebalance decision at the close
        decide = sched[i]
        rebalanced_at_close = False
        if decide:
            new = ev.eval(p.tree, base + i)
            if p.leverage != 1.0:
                new = {t: w * p.leverage for t, w in new.items() if t != "cash"}
            new = {t: w for t, w in new.items() if abs(w) > 1e-9 and t != "cash"}
            changed = set(new) != set(target) or any(abs(new.get(t, 0) - target.get(t, 0)) > 1e-9 for t in new)
            if bands and target and not changed and not drifted(c) and i > 0:
                decide = False
            target = new
            if decide:
                n_before = len(orders)
                if p.fill == "close":
                    trade_to(target, c, i, "rebalance" if i else "initial")
                    n_rebal += len(orders) > n_before
                    rebalanced_at_close = True
                else:
                    pending_target = dict(target)
                    n_rebal += 1
        elif bands and target and drifted(c):
            n_rebal += 1
            if p.fill == "close":
                trade_to(target, c, i, "drift rebalance")
                rebalanced_at_close = True
            else:
                pending_target = dict(target)
        # maintenance margin (leverage or shorts): equity must cover `mm` of gross exposure at the close,
        # otherwise every position is cut pro rata at the close back to the target gross exposure
        if (p.leverage > 1 or (shares < 0).any()) and not rebalanced_at_close:
            g = gross_now(c)
            eq_now = value(c)
            tgt_gross = sum(abs(w) for w in target.values()) or p.leverage
            if mm:
                tgt_gross = min(tgt_gross, 1 / mm)
            if g > 0 and eq_now > 0:
                if mm and eq_now / g < mm - 1e-12:
                    pv = np.nan_to_num(px_now(c))
                    k = tgt_gross * eq_now / g
                    trade_to({tick[j]: shares[j] * pv[j] * k / eq_now for j in range(N) if shares[j]}, c, i,
                             "margin call", min_trade=False)
                    margin_days.append(cal[i].date())
                    g, eq_now = gross_now(c), value(c)
                if eq_now > 0 and g / eq_now > max(lev_peak[0], 1.5 * tgt_gross):
                    lev_peak = (g / eq_now, tgt_gross, cal[i].date())
        equity[i] = value(c)
        if equity[i] > 0:
            pv = np.nan_to_num(px_now(c))
            weights[i] = shares * pv / equity[i]
            cashw[i] = cash / equity[i]
        if equity[i] <= 0 and (p.withdrawal or p.withdrawal_pct or p.leverage > 1 or cash < 0 or (shares != 0).any()):
            if p.withdrawal or p.withdrawal_pct:
                p.notes.append(f"Money ran out on {cal[i].date()}.")
            else:
                p.notes.append(f"The leveraged portfolio was wiped out on {cal[i].date()}: equity fell to "
                               f"${equity[i]:,.0f} at the close (a loss bigger than the margin could absorb); the "
                               "simulation stopped there and the balance is shown as $0 from then on.")
            equity[i:] = 0.0
            break
    if margin_days:
        more = f" and {len(margin_days) - 5} more" if len(margin_days) > 5 else ""
        p.notes.append(f"Margin call on {', '.join(str(d) for d in margin_days[:5])}{more}: equity fell below "
                       f"{mm:.0%} of gross exposure, so every position was cut pro rata at the close back to the "
                       "target leverage (orders marked 'margin call').")
    if lev_peak[2] is not None:
        p.notes.append(f"Between rebalances leverage drifted up to {lev_peak[0]:.1f}x gross exposure (on {lev_peak[2]}; "
                       f"target {lev_peak[1]:.1f}x)" + (f"; margin calls cap it at {1 / mm:g}x." if mm else
                                                        " with margin calls turned off."))

    start_day = cal[0] - pd.Timedelta(days=1)
    idx_all = pd.DatetimeIndex([start_day]).append(cal)
    eq = pd.Series(np.concatenate([[p.capital], equity]), index=idx_all, name="equity")
    fl = pd.Series(np.concatenate([[0.0], flows]), index=idx_all, name="flows")
    hw = pd.DataFrame(weights, index=cal, columns=tick)
    hw = hw.loc[:, (hw.abs() > 1e-9).any()]
    hw["cash"] = cashw
    gross = hw.drop(columns="cash").abs().sum(axis=1)
    ex = pd.Series(np.concatenate([[0.0], gross.to_numpy()]), index=idx_all, name="exposure")
    npos = pd.Series(np.concatenate([[0], (hw.drop(columns="cash").abs() > 1e-6).sum(axis=1).to_numpy()]), index=idx_all)
    od = pd.DataFrame(orders)
    end_px = np.nan_to_num(last_px)
    trades = _round_trips(ledger, tick, end_px, cal[-1])
    res = Result(strategy=p, equity=eq, trades=trades, exposure=ex, positions=npos, prices=dfs,
                 holdings=hw, interest=interest, in_market=pd.Series(gross.reindex(idx_all).fillna(0).to_numpy() > 1e-6, index=idx_all),
                 kind="allocation", orders=od)
    res.extras.update({"fees": fees, "flows": fl, "turnover_annual": turnover / max((cal[-1] - cal[0]).days / 365.25, 1e-9),
                       "rebalances": n_rebal,
                       "attribution": _attribution(tick, tcash, tdiv, tcom, shares, end_px, od)})
    return res


def _attribution(tick, tcash, tdiv, tcom, shares, end_px, od: pd.DataFrame) -> pd.DataFrame:
    """P&L per ticker: sales - purchases - costs + dividends (cash or reinvested) + value still held.

    sum(pnl) + interest - fees == final equity - starting capital - net cash flows (exactly, unless
    a leveraged portfolio was wiped out and the simulation stopped)."""
    mv = shares * end_px
    bought = od[od.side == "buy"].groupby("ticker")["value"].sum() if not od.empty else pd.Series(dtype=float)
    sold = od[od.side == "sell"].groupby("ticker")["value"].sum() if not od.empty else pd.Series(dtype=float)
    rows = []
    for j, t in enumerate(tick):
        if not (tcash[j] or mv[j] or tdiv[j]):
            continue
        rows.append({"ticker": t, "pnl": float(tcash[j] + mv[j]), "dividends": float(tdiv[j]),
                     "commissions": float(tcom[j]), "bought": float(bought.get(t, 0.0)), "sold": float(sold.get(t, 0.0)),
                     "end_value": float(mv[j]), "end_shares": float(shares[j])})
    df = pd.DataFrame(rows, columns=["ticker", "pnl", "dividends", "commissions", "bought", "sold", "end_value", "end_shares"])
    if not df.empty:
        tot = df["pnl"].abs().sum()
        df["share_of_pnl"] = df["pnl"] / df["pnl"].sum() if abs(df["pnl"].sum()) > 1e-9 else np.nan
        df = df.sort_values("pnl", key=lambda s: -s.abs() if tot else s).reset_index(drop=True)
    return df


def _round_trips(ledger: list[tuple], tick: list[str], end_px: np.ndarray, last_day) -> pd.DataFrame:
    """Holding periods per ticker, from the trade that opens a position to the one that closes it.

    The ledger holds every event in simulation order (dividends at the start of the day, trades,
    dividend reinvestment at the close), so the share count is exact, dividends are income of the
    holding period they were paid in and reinvested dividends add to its cost. The holding-period
    P&Ls of a ticker add up to its P&L in the attribution table."""
    if not ledger:
        return pd.DataFrame()
    rows = []
    px = dict(zip(tick, end_px))
    by: dict[str, list] = {}
    for e in ledger:
        by.setdefault(e[1], []).append(e)
    for t, evs in by.items():
        pos, peak, st = 0.0, 0.0, None
        for d, _, kind, q, v, com in evs:
            if kind == "dividend":
                if st is not None:
                    st["income"] += v
                continue
            if st is None and kind in ("buy", "sell"):
                st = {"start": d, "spent": 0.0, "got": 0.0, "income": 0.0, "com": 0.0, "side": "long" if q > 0 else "short"}
                peak = 0.0
            if st is None:
                continue
            if kind in ("buy", "reinvest"):
                st["spent"] += v + com
            else:
                st["got"] += v - com
            st["com"] += com
            pos += q
            peak = max(peak, abs(pos))
            if abs(pos) <= 1e-9 * max(peak, 1.0):
                rows.append(_trip(t, st, d.date()))
                st, pos = None, 0.0
        if st is not None:
            st["got"] += pos * px[t]
            rows.append(_trip(t, st, last_day.date(), open_=True))
    tr = pd.DataFrame(rows)
    if tr.empty:
        return tr
    tr = tr.sort_values(["exit_date", "entry_date", "ticker"], kind="stable").reset_index(drop=True)
    tr.index = tr.index + 1
    tr["cum_pnl"] = tr["pnl"].cumsum()
    return tr


def _trip(t, st: dict, end, open_=False) -> dict:
    start = st["start"].date() if hasattr(st["start"], "date") else st["start"]
    pnl = st["got"] + st["income"] - st["spent"]
    basis = st["spent"] if st["side"] == "long" else st["got"]
    return {"ticker": t, "side": st["side"], "entry_date": start, "exit_date": end,
            "entry_price": np.nan, "exit_price": np.nan, "shares": np.nan, "position_value": basis,
            "pnl": pnl, "return": pnl / basis if basis else 0.0,
            "bars_held": int(np.busday_count(pd.Timestamp(start).date(), pd.Timestamp(end).date())),
            "exit_reason": "still held" if open_ else "rebalanced out", "mae": np.nan, "mfe": np.nan,
            "commission": st["com"], "income": st["income"], "entry_fill": "close", "exit_fill": "close"}

