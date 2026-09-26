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

Targets are evaluated on each rebalance date using that day's close and traded at the close (or
the next open). Cash flows, dividends and interest are applied daily.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from typing import Literal

import numpy as np
import pandas as pd

from . import data, expr
from .engine import Result, _daily_rate

FREQS = ("daily", "weekly", "monthly", "quarterly", "yearly", "none")


@dataclass
class Portfolio:
    tree: dict
    rebalance: Literal["daily", "weekly", "monthly", "quarterly", "yearly", "none"] = "monthly"
    drift_band: float | None = None              # also rebalance when any weight drifts this far
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
    contribution_freq: Literal["monthly", "quarterly", "yearly"] = "monthly"
    withdrawal: float = 0.0                      # $ withdrawn each period
    withdrawal_pct: float = 0.0                  # fraction of the balance withdrawn each period
    withdrawal_freq: Literal["monthly", "quarterly", "yearly"] = "yearly"
    inflation_adjust: bool = False               # grow $ contributions/withdrawals with CPI
    benchmark: str | None = None                 # comparison ticker for alpha/beta (default SPY)
    name: str = ""
    description: str = ""
    notes: list[str] = field(default_factory=list)
    kind: str = "allocation"

    def validate(self) -> None:
        if self.rebalance not in FREQS:
            raise ValueError(f"rebalance must be one of {FREQS}")
        validate_node(self.tree)
        if self.capital <= 0 and self.contribution <= 0:
            raise ValueError("need starting capital or contributions")

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

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
        rb = {"none": "never rebalanced (buy and hold)", "daily": "re-evaluated and rebalanced daily"}.get(
            self.rebalance, f"re-evaluated and rebalanced {self.rebalance}")
        if self.drift_band:
            if self.rebalance == "none":
                rb = f"rebalanced only when a holding drifts {self.drift_band:.0%} from its target"
            else:
                rb += f", plus whenever a holding drifts {self.drift_band:.0%} from target"
        lines.append(f"Rebalancing: {rb}; trades at the {'close' if self.fill == 'close' else 'next open'}")
        cf = []
        if self.contribution:
            cf.append(f"add ${self.contribution:,.0f} {self.contribution_freq}")
        if self.withdrawal:
            cf.append(f"withdraw ${self.withdrawal:,.0f} {self.withdrawal_freq}")
        if self.withdrawal_pct:
            cf.append(f"withdraw {self.withdrawal_pct:.1%} of the balance {self.withdrawal_freq}")
        if cf and self.inflation_adjust:
            cf.append("amounts grow with inflation (CPI)")
        lines.append(f"Money: ${self.capital:,.0f} start" + ("; " + ", ".join(cf) if cf else "")
                     + ("; dividends reinvested" if self.reinvest_dividends else "; dividends kept as cash"))
        costs = []
        if self.commission:
            costs.append(f"${self.commission:g}/order")
        if self.commission_pct:
            costs.append(f"{self.commission_pct:.3%} of value")
        if self.slippage_bps:
            costs.append(f"{self.slippage_bps:g} bps slippage/side")
        lines.append("Costs: " + (", ".join(costs) if costs else "none"))
        cr = self.cash_rate
        lines.append("Cash: " + ("earns the 3-month T-bill rate" if cr == "tbill" else
                                 f"earns {float(cr):.2%}/yr" if cr else "earns nothing"))
        if self.start or self.end:
            lines.append(f"Period: {self.start or 'start of data'} to {self.end or 'latest'}")
        return "\n".join(lines)


# ------------------------------------------------------------------ tree helpers

def validate_node(n: dict, depth: int = 0) -> None:
    if depth > 20:
        raise ValueError("portfolio tree is nested too deeply")
    if not isinstance(n, dict):
        raise ValueError(f"portfolio node must be an object, got {n!r}")
    if "asset" in n:
        data.load(n["asset"])  # raises DataError for unknown tickers
    elif n.get("cash"):
        pass
    elif "weights" in n:
        kids = n.get("children") or []
        if not kids:
            raise ValueError("weights node needs children")
        if n["weights"] == "specified":
            w = n.get("w")
            if not w or len(w) != len(kids):
                raise ValueError("specified weights need one weight per child")
            if abs(sum(w) - 1) > 1e-6:
                raise ValueError(f"weights must add up to 100% (got {sum(w):.1%})")
        elif n["weights"] not in ("equal", "inverse_vol", "market_cap"):
            raise ValueError(f"unknown weighting {n['weights']!r}")
        for k in kids:
            validate_node(k, depth + 1)
    elif "if" in n:
        expr.compile_expr(n["if"])
        data.load(n.get("on", "SPY"))
        validate_node(n["then"], depth + 1)
        validate_node(n["else"], depth + 1)
    elif "filter" in n:
        f = n["filter"]
        expr.compile_expr(f["by"])
        if f.get("require"):
            expr.compile_expr(f["require"])
        if int(f.get("n", 1)) < 1:
            raise ValueError("filter n must be at least 1")
        u = n.get("universe", "children")
        if u == "children":
            for k in n.get("children") or []:
                if "asset" not in k:
                    raise ValueError("filter children must be single assets")
                validate_node(k, depth + 1)
        if n.get("fallback"):
            validate_node(n["fallback"], depth + 1)
    else:
        raise ValueError(f"unrecognised portfolio node {n!r}")


def _universe(n: dict) -> list[str]:
    u = n.get("universe", "children")
    if u == "children":
        return [data.canonical(k["asset"]) for k in n.get("children") or []]
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
        elif "weights" in x:
            for k in x["children"]:
                walk(k)
        elif "if" in x:
            add(x.get("on", "SPY"))
            walk(x["then"])
            walk(x["else"])
        elif "filter" in x:
            for t in _universe(x):
                add(t)
            if x.get("fallback"):
                walk(x["fallback"])
    walk(n)
    return out


def fixed_tickers(n: dict) -> list[str]:
    """Tickers that must have data for the portfolio to be defined (excludes filter universes)."""
    out: list[str] = []

    def walk(x):
        if "asset" in x:
            out.append(data.canonical(x["asset"]))
        elif "weights" in x:
            for k in x["children"]:
                walk(k)
        elif "if" in x:
            out.append(data.canonical(x.get("on", "SPY")))
            walk(x["then"])
            walk(x["else"])
        elif "filter" in x:
            if x.get("universe", "children") == "children" or isinstance(x.get("universe"), list):
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
        label = {"equal": "equal weight", "inverse_vol": f"inverse volatility ({n.get('lookback', 20)}-day)",
                 "market_cap": "market-cap weight"}[kind]
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
        uname = "Nasdaq-100 members (point-in-time)" if u in ("NDX", "nasdaq100") else ", ".join(_universe(n))
        s = (f"{pad}{f.get('select', 'top')} {f.get('n', 1)} of [{uname}] by {f['by']}, "
             f"{ {'equal': 'equal weight', 'inverse_vol': 'inverse-volatility weight', 'market_cap': 'market-cap weight'}[f.get('weights', 'equal')] }")
        out = [s]
        if f.get("require"):
            out.append(f"{pad}  only if {f['require']}, else:")
            out.extend(describe(n.get("fallback") or {"cash": True}, indent + 2))
        return out
    return [f"{pad}{n}"]


# ------------------------------------------------------------------ evaluation

class _Evaluator:
    def __init__(self, p: Portfolio, cal: pd.DatetimeIndex, dfs: dict[str, pd.DataFrame]):
        self.p, self.cal, self.dfs = p, cal, dfs
        self.ns = {t: expr.Namespace(df) for t, df in dfs.items()}
        self.cache: dict = {}
        self.close = {t: df["close"].reindex(cal).to_numpy() for t, df in dfs.items()}
        self.members = None

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
        return bool(self.members.get(t, np.zeros(len(self.cal), bool))[i])

    def has(self, t: str, i: int) -> bool:
        return np.isfinite(self.close[t][i])

    def weigh(self, method: str, tickers: list[str], i: int, lookback: int = 20) -> dict[str, float]:
        if not tickers:
            return {"cash": 1.0}
        if method == "inverse_vol":
            v = np.array([self.vol(t, lookback)[i] for t in tickers])
            ok = np.isfinite(v) & (v > 0)
            if ok.any():
                inv = np.where(ok, 1 / np.where(ok, v, 1), 0)
                return {t: w for t, w in zip(tickers, inv / inv.sum()) if w > 0}
        if method == "market_cap":
            m = np.array([self.mcap(t)[i] for t in tickers])
            ok = np.isfinite(m) & (m > 0)
            if ok.all():
                return {t: w for t, w in zip(tickers, m / m.sum())}
            self.note("Market-cap weights need share counts; tickers without them were equal-weighted.")
            if ok.any():
                eqw = 1 / len(tickers)
                base = {t: eqw for t in tickers}
                capw = m[ok] / m[ok].sum() * (ok.sum() / len(tickers))
                for t, w in zip([t for t, o in zip(tickers, ok) if o], capw):
                    base[t] = w
                return base
        return {t: 1 / len(tickers) for t in tickers}

    def note(self, msg: str) -> None:
        if msg not in self.p.notes:
            self.p.notes.append(msg)

    def eval(self, n: dict, i: int) -> dict[str, float]:
        if "asset" in n:
            t = data.canonical(n["asset"])
            if self.has(t, i):
                return {t: 1.0}
            self.note(f"{t} had no price on some rebalance dates (before it listed); its slice was held in cash then.")
            return {"cash": 1.0}
        if n.get("cash"):
            return {"cash": 1.0}
        if "weights" in n:
            kids = n["children"]
            method = n["weights"]
            if method == "specified":
                ws = n["w"]
            elif method in ("inverse_vol", "market_cap") and all("asset" in k for k in kids):
                live = [data.canonical(k["asset"]) for k in kids if self.has(data.canonical(k["asset"]), i)]
                base = self.weigh(method, live, i, n.get("lookback", 20))
                ws = [base.get(data.canonical(k["asset"]), 0.0) for k in kids]
                if not live:
                    ws = [1 / len(kids)] * len(kids)
            else:
                if method != "equal":
                    self.note(f"{method} weighting applies to single assets only; groups were equal-weighted.")
                ws = [1 / len(kids)] * len(kids)
            out: dict[str, float] = {}
            for w, k in zip(ws, kids):
                if w <= 0:
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
            names = _universe(n)
            pit = u in ("NDX", "nasdaq100") and self.p.point_in_time
            cands = []
            for t in names:
                if not self.has(t, i) or (pit and not self.is_member(t, i)):
                    continue
                v = self.series(f["by"], t, "value")[i]
                if np.isfinite(v):
                    cands.append((v, t))
            cands.sort(key=lambda x: x[0], reverse=f.get("select", "top") == "top")
            chosen = [t for _, t in cands[: int(f.get("n", 1))]]
            if f.get("require"):
                passed = [t for t in chosen if self.series(f["require"], t, "bool")[i]]
            else:
                passed = chosen
            if not passed:
                return self.eval(n.get("fallback") or {"cash": True}, i)
            w = self.weigh(f.get("weights", "equal"), passed, i, f.get("lookback", 20))
            if f.get("require") and len(passed) < len(chosen):
                # slots whose pick failed the requirement go to the fallback
                share = len(passed) / len(chosen)
                out = {t: x * share for t, x in w.items()}
                for t, x in self.eval(n.get("fallback") or {"cash": True}, i).items():
                    out[t] = out.get(t, 0.0) + x * (1 - share)
                return out
            return w
        raise ValueError(f"bad node {n}")


def _schedule(cal: pd.DatetimeIndex, freq: str) -> np.ndarray:
    """True on the last trading day of each period (daily: every day; none: only the first day)."""
    T = len(cal)
    if freq == "daily":
        return np.ones(T, bool)
    out = np.zeros(T, bool)
    if freq == "none":
        out[0] = True
        return out
    code = {"weekly": "W-FRI", "monthly": "M", "quarterly": "Q", "yearly": "Y"}[freq]
    per = cal.to_period(code)
    last = pd.Series(np.arange(T), index=cal).groupby(per).max().to_numpy()
    out[last] = True
    out[0] = True  # initial allocation
    return out


def _period_starts(cal: pd.DatetimeIndex, freq: str) -> np.ndarray:
    code = {"monthly": "M", "quarterly": "Q", "yearly": "Y"}[freq]
    per = cal.to_period(code)
    first = pd.Series(np.arange(len(cal)), index=cal).groupby(per).min().to_numpy()
    out = np.zeros(len(cal), bool)
    out[first] = True
    out[0] = False  # the starting capital covers the first period
    return out


def run(p: Portfolio) -> Result:
    p.validate()
    names = tickers_in(p.tree)
    dfs = data.load_many(names)
    must = fixed_tickers(p.tree)
    cal = None
    for df in dfs.values():
        cal = df.index if cal is None else cal.union(df.index)
    first_common = max(dfs[t].index[0] for t in must) if must else cal[0]
    start = pd.Timestamp(p.start) if p.start else first_common
    if p.start and pd.Timestamp(p.start) < first_common:
        late = [t for t in must if dfs[t].index[0] == first_common]
        p.notes.append(f"Start moved to {first_common.date()}, when {', '.join(late)} began trading.")
        start = first_common
    cal = cal[cal >= start]
    if p.end:
        cal = cal[cal <= pd.Timestamp(p.end)]
    if len(cal) < 2:
        raise ValueError("no price data in the requested period")
    # need a warm-up for indicators used by rules: signals come from full history (computed per ticker)
    ev = _Evaluator(p, cal, dfs)
    T = len(cal)
    tick = list(dfs)
    idx = {t: j for j, t in enumerate(tick)}
    N = len(tick)
    O = np.column_stack([dfs[t]["open"].reindex(cal).to_numpy() for t in tick])
    C = np.column_stack([ev.close[t] for t in tick])
    DIV = np.column_stack([dfs[t]["dividend"].reindex(cal).fillna(0.0).to_numpy() for t in tick])
    rate = _daily_rate(cal, p.cash_rate)
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
    contrib_days = _period_starts(cal, p.contribution_freq) if p.contribution else np.zeros(T, bool)
    wd_days = _period_starts(cal, p.withdrawal_freq) if (p.withdrawal or p.withdrawal_pct) else np.zeros(T, bool)

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
    turnover = 0.0
    n_rebal = 0

    def px_now(prices):
        return np.where(np.isfinite(prices), prices, last_px)

    def value(prices) -> float:
        pv = px_now(prices)
        return cash + float(np.nansum(shares * np.nan_to_num(pv)))

    def trade_to(tgt: dict[str, float], prices: np.ndarray, i: int, reason: str) -> None:
        nonlocal cash, turnover
        pv = px_now(prices)
        eq = value(prices)
        if eq <= 0:
            return
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
                scale = min(1.0, max(cash, 0) / need) if need > 0 else 1.0
            else:
                scale = 1.0
            for j in js:
                q = delta[j] * scale
                if abs(q) * pv[j] < p.min_trade * eq:
                    continue
                if not p.fractional_shares:
                    q = np.floor(q) if q > 0 else -np.floor(-q)
                if q == 0:
                    continue
                fill = pv[j] * (1 + np.sign(q) * slip)
                com = p.commission + p.commission_pct * abs(q) * fill
                cash -= q * fill + com
                shares[j] += q
                turnover += abs(q) * fill / eq
                orders.append({"date": cal[i].date(), "ticker": tick[j], "side": "buy" if q > 0 else "sell",
                               "shares": abs(q), "price": fill, "value": abs(q) * fill, "commission": com,
                               "reason": reason})

    def drift(prices) -> float:
        eq = value(prices)
        if eq <= 0 or not target:
            return 0.0
        pv = np.nan_to_num(px_now(prices))
        cur = {tick[j]: shares[j] * pv[j] / eq for j in range(N) if shares[j]}
        keys = set(cur) | {t for t in target if t != "cash"}
        return max((abs(cur.get(t, 0.0) - target.get(t, 0.0)) for t in keys), default=0.0)

    for i in range(T):
        o, c = O[i], C[i]
        # overnight interest and dividends
        if i > 0:
            r = rate[i - 1]
            earned = cash * r if cash >= 0 else cash * r
            cash += earned
            interest += earned
        div_cash = shares * DIV[i]
        got = float(np.nansum(div_cash))
        if got:
            cash += got
        # cash flows at the start of the day
        f = 0.0
        if contrib_days[i]:
            f += p.contribution * infl[i]
        if wd_days[i]:
            f -= p.withdrawal * infl[i]
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
        if cash < -1e-6 * max(eq_close, 1.0) and eq_close > 0 and target:
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
                    orders.append({"date": cal[i].date(), "ticker": t, "side": "buy", "shares": q, "price": fill,
                                   "value": q * fill, "commission": com, "reason": "contribution"})
        # reinvest dividends into the same holding at the close
        if got and p.reinvest_dividends:
            for j in np.flatnonzero(div_cash > 0):
                if np.isfinite(c[j]) and c[j] > 0 and cash > 0:
                    q = min(div_cash[j], cash) / c[j]
                    shares[j] += q
                    cash -= q * c[j]
        # rebalance decision at the close
        decide = sched[i]
        if decide:
            new = ev.eval(p.tree, i)
            new = {t: w for t, w in new.items() if w > 1e-9}
            changed = set(new) != set(target) or any(abs(new.get(t, 0) - target.get(t, 0)) > 1e-9 for t in new)
            if p.drift_band and target and not changed and drift(c) <= p.drift_band and i > 0:
                decide = False
            target = new
            if decide:
                n_rebal += 1
                if p.fill == "close":
                    trade_to(target, c, i, "rebalance" if i else "initial")
                else:
                    pending_target = dict(target)
        elif p.drift_band and target and drift(c) > p.drift_band:
            n_rebal += 1
            if p.fill == "close":
                trade_to(target, c, i, "drift rebalance")
            else:
                pending_target = dict(target)
        equity[i] = value(c)
        if equity[i] > 0:
            pv = np.nan_to_num(px_now(c))
            weights[i] = shares * pv / equity[i]
            cashw[i] = cash / equity[i]
        if equity[i] <= 0 and (p.withdrawal or p.withdrawal_pct):
            p.notes.append(f"Money ran out on {cal[i].date()}.")
            equity[i:] = 0.0
            break

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
    trades = _round_trips(od, hw, dfs, cal, equity, p.reinvest_dividends)
    res = Result(strategy=p, equity=eq, trades=trades, exposure=ex, positions=npos, prices=dfs,
                 holdings=hw, interest=interest, in_market=pd.Series(gross.reindex(idx_all).fillna(0).to_numpy() > 1e-6, index=idx_all),
                 kind="allocation", orders=od)
    res.extras.update({"flows": fl, "turnover_annual": turnover / max((cal[-1] - cal[0]).days / 365.25, 1e-9),
                       "rebalances": n_rebal})
    return res


def _round_trips(orders: pd.DataFrame, hw: pd.DataFrame, dfs, cal, equity, reinvest: bool = True) -> pd.DataFrame:
    """Holding periods per ticker (from the first buy to the sale that empties it) with P&L.

    With dividends reinvested, the extra shares show up in the sale proceeds; otherwise the cash
    dividends received during the holding period are added explicitly.
    """
    if orders is None or orders.empty:
        return pd.DataFrame()
    rows = []
    for t, g in orders.groupby("ticker", sort=False):
        div = dfs[t]["dividend"]
        pos, spent, got, com, start, held = 0.0, 0.0, 0.0, 0.0, None, []
        for _, o in g.sort_values("date", kind="stable").iterrows():
            q = o["shares"] if o["side"] == "buy" else -o["shares"]
            if pos <= 1e-12 and q > 0:
                start, spent, got, com, held = o["date"], 0.0, 0.0, 0.0, []
            if q > 0:
                spent += o["value"] + o["commission"]
            else:
                got += o["value"] - o["commission"]
            com += o["commission"]
            pos += q
            held.append((pd.Timestamp(o["date"]), pos))
            if pos <= 1e-9 and start is not None:
                d = 0.0 if reinvest else _divs(held, div)
                rows.append(_trip(t, start, o["date"], spent, got + d, com, d))
                start = None
        if start is not None and pos > 1e-9:
            last = dfs[t]["close"].reindex(cal).ffill().iloc[-1]
            d = 0.0 if reinvest else _divs(held + [(cal[-1] + pd.Timedelta(days=1), 0.0)], div)
            rows.append(_trip(t, start, cal[-1].date(), spent, got + pos * last + d, com, d, open_=True))
    tr = pd.DataFrame(rows)
    if tr.empty:
        return tr
    tr = tr.sort_values(["exit_date", "entry_date", "ticker"], kind="stable").reset_index(drop=True)
    tr.index = tr.index + 1
    tr["cum_pnl"] = tr["pnl"].cumsum()
    return tr


def _divs(held: list, div: pd.Series) -> float:
    """Cash dividends received while held: `held` is [(date, shares after that day's trade), ...]."""
    total = 0.0
    for (d0, q), (d1, _) in zip(held, held[1:]):
        seg = div[(div.index > d0) & (div.index <= d1)]
        total += q * float(seg.sum())
    return total


def _trip(t, start, end, spent, got, com, d, open_=False) -> dict:
    pnl = got - spent
    return {"ticker": t, "side": "long", "entry_date": start, "exit_date": end,
            "entry_price": np.nan, "exit_price": np.nan, "shares": np.nan, "position_value": spent,
            "pnl": pnl, "return": pnl / spent if spent else 0.0,
            "bars_held": int(np.busday_count(pd.Timestamp(start).date(), pd.Timestamp(end).date())),
            "exit_reason": "still held" if open_ else "rebalanced out", "mae": np.nan, "mfe": np.nan,
            "commission": com, "income": d, "entry_fill": "close", "exit_fill": "close"}
