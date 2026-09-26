"""Research tools: parameter sweeps, walk-forward optimisation and portfolio optimisation.

Parameters are written as placeholders anywhere in a plain-English sentence (or rule), e.g.

    buy QQQ when RSI({2..5}) is below {5..20 step 5}, hold {1,3,5} days

`{a..b}` is an integer range, `{a..b step s}` a stepped range, `{x,y,z}` a list.
"""
from __future__ import annotations

import dataclasses
import itertools
import os
import re
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from . import data, metrics, parser, runner

PLACEHOLDER = re.compile(r"\{([^{}]+)\}")
OBJECTIVES = {"sharpe": "Sharpe", "cagr": "CAGR", "calmar": "Calmar", "sortino": "Sortino",
              "total_return": "Total return", "profit_factor": "Profit factor"}
MAX_COMBOS = 2000


def _values(spec: str) -> list:
    s = spec.strip()
    m = re.fullmatch(r"(-?[\d.]+)\s*\.\.\s*(-?[\d.]+)(?:\s+step\s+([\d.]+))?", s)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        step = float(m.group(3)) if m.group(3) else 1.0
        if step <= 0:
            raise ValueError("step must be positive")
        n = int(round((b - a) / step)) + 1
        vals = [a + i * step for i in range(max(n, 0))]
        return [int(v) if float(v).is_integer() else round(v, 10) for v in vals]
    parts = [p.strip() for p in s.split(",")]
    out = []
    for p in parts:
        try:
            v = float(p)
            out.append(int(v) if v.is_integer() else v)
        except ValueError:
            out.append(p)
    return out


ARTICLES = ("the", "a", "an")


def _label(text: str, m: re.Match) -> str:
    """A readable column label for one placeholder, with the placeholder written out in full.

    A placeholder glued to the word before it is labelled by that whole word ("RSI({2..5})"); otherwise the
    previous word is prefixed ("below {5..25 step 5}", "hold {1,3,5}", "at {1..3}%").
    """
    a, b = m.start(), m.end()
    while a > 0 and not text[a - 1].isspace():
        a -= 1
    while b < len(text) and not text[b].isspace():
        b += 1
    while b > m.end() and (text[b - 1] in ",.;:!?" or (text[b - 1] == ")" and text[a:b].count(")") > text[a:b].count("("))):
        b -= 1  # trailing punctuation and unmatched closing brackets are not part of the word
    while a < m.start() and text[a] == "(" and text[a:b].count("(") > text[a:b].count(")"):
        a += 1  # likewise an unmatched opening bracket
    word = text[a:b]
    if a < m.start():  # glued on the left: "RSI({2..5})" names itself
        return word
    before = [w.rstrip(',.;:') for w in text[:a].split()]  # free-standing or only a suffix ("{1..3}%"): prefix the
    while before and before[-1].lower() in ARTICLES:     # previous word, skipping articles ("buy the {3,5}" -> "buy {3,5}")
        before.pop()
    return f"{before[-1]} {word}" if before and before[-1] else word


def expand(text: str) -> tuple[list[str], list[str], list[tuple]]:
    """-> (parameter labels, filled-in texts, value tuples)."""
    specs = PLACEHOLDER.findall(text)
    if not specs:
        raise ValueError("No {placeholders} found. Write e.g. 'hold {1..5} days' or 'RSI({2,3,4}) below {5..20 step 5}'.")
    grids = [_values(s) for s in specs]
    labels = []
    for m in PLACEHOLDER.finditer(text):
        lab = _label(text, m)
        k, base = 2, lab
        while lab in labels:  # labels key the CSV columns, so keep them unique
            lab, k = f"{base} #{k}", k + 1
        labels.append(lab)
    combos = list(itertools.product(*grids))
    if len(combos) > MAX_COMBOS:
        raise ValueError(f"{len(combos)} combinations; the limit is {MAX_COMBOS}. Use coarser steps.")
    texts = []
    for c in combos:
        it = iter(c)
        texts.append(PLACEHOLDER.sub(lambda m: str(next(it)), text))
    return labels, texts, combos


def _one(args) -> dict:
    text, overrides = args
    try:
        spec = parser.parse(text, **overrides)
        res = runner.run(spec)
        fl = res.extras.get("flows")
        st = metrics.equity_stats(res.equity, flows=fl)
        ts = metrics.trade_stats(res.trades, st["years"])
        return {"ok": True, "cagr": st["cagr"], "sharpe": st["sharpe"], "sortino": st["sortino"], "calmar": st["calmar"],
                "max_drawdown": st["max_drawdown"], "total_return": st["total_return"], "final": st["end_equity"],
                "trades": ts.get("trades", 0), "win_rate": ts.get("win_rate"), "profit_factor": ts.get("profit_factor"),
                "skew": st["skew"], "kurtosis": st["kurtosis"], "n_obs": max(len(res.equity) - 1, 0),
                "equity": res.equity}
    except Exception as e:  # noqa: BLE001 - one bad combination shouldn't sink the sweep
        return {"ok": False, "error": str(e).splitlines()[0]}


def _workers(n: int) -> int:
    return max(1, min(n, (os.cpu_count() or 2)))


def sweep(text: str, objective: str = "sharpe", overrides: dict | None = None, keep_equity: int = 5,
          parallel: bool = True) -> dict:
    labels, texts, combos = expand(text)
    args = [(t, overrides or {}) for t in texts]
    if parallel and len(args) > 3:
        with ProcessPoolExecutor(max_workers=_workers(len(args))) as ex:
            outs = list(ex.map(_one, args, chunksize=max(1, len(args) // 32)))
    else:
        outs = [_one(a) for a in args]
    rows = []
    for c, t, o in zip(combos, texts, outs):
        row = {"params": list(c), "text": t, **{k: v for k, v in o.items() if k != "equity"}}
        rows.append(row)
    ok = [r for r in rows if r["ok"]]
    key = lambda r: (r.get(objective) if r.get(objective) is not None and np.isfinite(r.get(objective) or np.nan) else -np.inf)  # noqa: E731
    ok.sort(key=key, reverse=True)
    best = ok[:keep_equity]
    curves = {}
    for r in best:
        i = rows.index(r)
        curves[" / ".join(str(p) for p in r["params"])] = outs[i]["equity"]
    return {"labels": labels, "objective": objective, "rows": rows, "ranked": ok, "curves": curves,
            "errors": [r for r in rows if not r["ok"]], "multiple_testing": multiple_testing(rows)}


def multiple_testing(rows: list[dict]) -> dict:
    """How impressive the best Sharpe is, given how many combinations were tried (Deflated Sharpe Ratio).

    Only combinations that traded count as trials: a combination that never trades has no Sharpe.
    """
    trials = [r for r in rows if r.get("ok") and r.get("trades") and metrics._finite(r.get("sharpe"))]
    if not trials:
        return {"n_trials": 0, "n_combinations": len(rows)}
    best = max(trials, key=lambda r: r["sharpe"])
    out = metrics.deflated_sharpe(best["sharpe"], [r["sharpe"] for r in trials], best.get("n_obs") or 0,
                                  best.get("skew") or 0.0, best.get("kurtosis") or 0.0)
    out.update({"n_combinations": len(rows), "best_params": list(best["params"])})
    return out


def walk_forward(text: str, objective: str = "sharpe", in_sample_years: float = 5, out_sample_years: float = 1,
                 anchored: bool = False, start: str | None = None, end: str | None = None,
                 overrides: dict | None = None) -> dict:
    """Optimise on each in-sample window, trade the winner on the following out-of-sample window."""
    labels, texts, combos = expand(text)
    first = parser.parse(texts[0], **(overrides or {}))
    probe = runner.run(first)
    t0 = pd.Timestamp(start) if start else probe.equity.index[1]
    t1 = pd.Timestamp(end) if end else probe.equity.index[-1]
    windows = []
    is_start = t0
    oos_start = t0 + pd.DateOffset(months=int(in_sample_years * 12))
    while oos_start < t1:
        oos_end = min(oos_start + pd.DateOffset(months=int(out_sample_years * 12)) - pd.Timedelta(days=1), t1)
        windows.append((is_start, oos_start - pd.Timedelta(days=1), oos_start, oos_end))
        oos_start = oos_end + pd.Timedelta(days=1)
        if not anchored:
            is_start = oos_start - pd.DateOffset(months=int(in_sample_years * 12))
    if not windows:
        raise ValueError("Period too short for the chosen in-sample window.")
    results = []
    oos_curve = []
    level = 10_000.0
    for (a, b, c, d) in windows:
        ov_is = dict(overrides or {}, start=str(a.date()), end=str(b.date()))
        args = [(t, ov_is) for t in texts]
        with ProcessPoolExecutor(max_workers=_workers(len(args))) as ex:
            outs = list(ex.map(_one, args, chunksize=max(1, len(args) // 32)))
        scored = [(o.get(objective) if o["ok"] and o.get(objective) is not None and np.isfinite(o.get(objective)) else -np.inf, i) for i, o in enumerate(outs)]
        score, bi = max(scored)
        if not np.isfinite(score):
            results.append({"in_sample": [a.date(), b.date()], "out_sample": [c.date(), d.date()], "params": None,
                            "is_score": None, "oos_return": None, "oos_sharpe": None})
            continue
        ov_oos = dict(overrides or {}, start=str(c.date()), end=str(d.date()), capital=level)
        o = _one((texts[bi], ov_oos))
        if not o["ok"]:
            continue
        eq = o["equity"]
        seg = eq.iloc[1:] if oos_curve else eq
        oos_curve.append(seg)
        level = float(eq.iloc[-1])
        results.append({"in_sample": [a.date(), b.date()], "out_sample": [c.date(), d.date()],
                        "params": list(combos[bi]), "is_score": score, "oos_return": float(eq.iloc[-1] / eq.iloc[0] - 1),
                        "oos_sharpe": o["sharpe"], "oos_max_drawdown": o["max_drawdown"]})
    stitched = pd.concat(oos_curve) if oos_curve else pd.Series(dtype=float)
    stitched = stitched[~stitched.index.duplicated(keep="last")]
    oos_stats = metrics.equity_stats(stitched) if len(stitched) > 30 else {}
    # full-period in-sample optimum for comparison (the overfitting gap)
    full = sweep(text, objective, dict(overrides or {}, start=str(windows[0][2].date()), end=str(t1.date())), keep_equity=1)
    best_full = full["ranked"][0] if full["ranked"] else None
    return {"labels": labels, "objective": objective, "anchored": anchored, "windows": results,
            "oos_equity": stitched, "oos_stats": oos_stats, "hindsight_best": best_full}




# ------------------------------------------------------------------ portfolio optimisation

OPT_METHODS = {
    "max_sharpe": "Max Sharpe", "min_variance": "Min variance", "max_sortino": "Max Sortino",
    "min_cvar": "Min CVaR (95%)", "risk_parity": "Risk parity", "max_diversification": "Max diversification",
    "target_return": "Target return", "target_vol": "Target volatility",
    "inverse_vol": "Inverse volatility", "equal": "Equal weight",
}
_CON = re.compile(r"^\s*(?:(?P<lo>-?[\d.]+%?)\s*(?:<=|<)\s*)?(?P<names>[A-Za-z0-9^.\-_]+(?:\s*\+\s*[A-Za-z0-9^.\-_]+)*)"
                  r"\s*(?P<op><=|>=|=|<|>)\s*(?P<val>-?[\d.]+%?)\s*$")


def _frac(s: str) -> float:
    s = s.strip()
    if s.endswith("%"):
        return float(s[:-1]) / 100
    v = float(s)
    return v / 100 if abs(v) > 1 else v


def parse_constraints(items, tickers: list[str]) -> tuple[dict[str, list[float]], list[dict]]:
    """Constraints like "SPY <= 50%", "TLT >= 10%", "SPY+QQQ <= 70%", "20% <= TLT+IEF <= 60%".

    A single ticker sets that ticker's bounds; several tickers joined by + make a group constraint.
    Returns (bounds {ticker: [lo, hi]}, groups [{tickers, lo, hi}])."""
    if isinstance(items, str):
        items = [x for x in re.split(r"[;\n,]", items) if x.strip()]
    bounds: dict[str, list[float]] = {}
    groups: list[dict] = []
    known = set(tickers)
    for raw in items or []:
        m = _CON.match(raw)
        if not m:
            raise ValueError(f"Could not read the constraint {raw!r}. Write e.g. 'SPY <= 50%', 'TLT >= 10%' or 'SPY+QQQ <= 70%'.")
        names = [data.canonical(x) for x in m.group("names").split("+")]
        bad = [x for x in names if x not in known]
        if bad:
            raise ValueError(f"Constraint {raw.strip()!r} names {', '.join(bad)}, which is not among the tickers.")
        v = _frac(m.group("val"))
        op = m.group("op")
        lo, hi = (-np.inf, np.inf)
        if op in ("<=", "<"):
            hi = v
        elif op in (">=", ">"):
            lo = v
        else:
            lo = hi = v
        if m.group("lo"):
            if op not in ("<=", "<"):
                raise ValueError(f"Constraint {raw.strip()!r}: write a range as 'lo <= A+B <= hi'.")
            lo = _frac(m.group("lo"))
        if lo > hi + 1e-12:
            raise ValueError(f"Constraint {raw.strip()!r} has its lower limit above its upper limit.")
        if len(names) == 1:
            b = bounds.setdefault(names[0], [-np.inf, np.inf])
            b[0], b[1] = max(b[0], lo), min(b[1], hi)
        else:
            groups.append({"tickers": names, "lo": None if not np.isfinite(lo) else lo,
                           "hi": None if not np.isfinite(hi) else hi, "text": raw.strip()})
    return bounds, groups


def round_weights(weights: dict[str, float], total: int = 100) -> dict[str, int]:
    """Integer percentages that add up to exactly `total` (largest-remainder rounding)."""
    items = [(t, max(float(w), 0.0)) for t, w in weights.items()]
    s = sum(w for _, w in items)
    if s <= 0:
        return {}
    raw = [(t, w / s * total) for t, w in items]
    fl = {t: int(np.floor(x)) for t, x in raw}
    left = total - sum(fl.values())
    order = sorted(raw, key=lambda kv: (-(kv[1] - np.floor(kv[1])), -kv[1]))
    for t, _ in order[:left]:
        fl[t] += 1
    return {t: v for t, v in sorted(fl.items(), key=lambda kv: -kv[1]) if v > 0}


def weights_sentence(weights: dict[str, float], rebalance: str = "quarterly") -> str:
    """A sentence the parser reads back as these weights (rounded to whole percents adding to 100)."""
    r = round_weights(weights)
    parts = [f"{v}% {t}" for t, v in r.items()]
    if len(parts) == 1:
        return f"buy and hold {next(iter(r))}"
    body = ", ".join(parts[:-1]) + " and " + parts[-1]
    return f"hold {body}, rebalance {rebalance}"


class _Opt:
    """Long-only optimiser on a (months x assets) matrix of monthly returns, with per-asset bounds
    and linear group constraints."""

    def __init__(self, R: np.ndarray, tickers: list[str], rf_annual: float, bounds: dict, groups: list[dict],
                 min_weight: float = 0.0, max_weight: float = 1.0):
        self.R = R
        self.t = tickers
        self.n = n = len(tickers)
        self.mu = R.mean(axis=0) * 12
        self.cov = np.cov(R, rowvar=False).reshape(n, n) * 12 + np.eye(n) * 1e-12
        self.sd = np.sqrt(np.diag(self.cov))
        self.rf = rf_annual
        lo = np.full(n, max(0.0, min_weight))
        hi = np.full(n, min(1.0, max_weight))
        for j, t in enumerate(tickers):
            if t in bounds:
                if np.isfinite(bounds[t][0]):
                    lo[j] = max(lo[j], bounds[t][0])
                if np.isfinite(bounds[t][1]):
                    hi[j] = min(hi[j], bounds[t][1])
        lo = np.maximum(lo, 0.0)
        if (lo > hi + 1e-12).any():
            j = int(np.argmax(lo - hi))
            raise ValueError(f"The limits for {tickers[j]} contradict each other ({lo[j]:.0%} minimum, {hi[j]:.0%} maximum).")
        if lo.sum() > 1 + 1e-9 or hi.sum() < 1 - 1e-9:
            raise ValueError("The per-asset limits cannot add up to 100% (the minimums add to more than 100% or the maximums to less).")
        self.lo, self.hi = lo, hi
        A, b = [], []
        for g in groups:
            row = np.array([1.0 if t in g["tickers"] else 0.0 for t in tickers])
            if g.get("hi") is not None:
                A.append(row)
                b.append(g["hi"])
            if g.get("lo") is not None:
                A.append(-row)
                b.append(-g["lo"])
        self.A = np.array(A) if A else np.zeros((0, n))
        self.b = np.array(b) if b else np.zeros(0)
        self.x0 = self._feasible()

    # -- helpers
    def _lp(self, c):
        from scipy.optimize import linprog
        r = linprog(c, A_ub=self.A if len(self.b) else None, b_ub=self.b if len(self.b) else None,
                    A_eq=np.ones((1, self.n)), b_eq=[1.0], bounds=list(zip(self.lo, self.hi)), method="highs")
        return r.x if r.status == 0 else None

    def _feasible(self):
        x = self._lp(np.zeros(self.n))
        if x is None:
            raise ValueError("No portfolio satisfies all the constraints together. Loosen the limits.")
        # the feasible point closest to equal weight keeps the solver's starts central
        w = self.project(np.full(self.n, 1 / self.n), start=x)
        return w if w is not None else x

    def cons(self, extra=()):
        c = [{"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones_like(w)}]
        if len(self.b):
            c.append({"type": "ineq", "fun": lambda w: self.b - self.A @ w, "jac": lambda w: -self.A})
        return c + list(extra)

    def solve(self, fun, extra=(), starts=None):
        from scipy.optimize import minimize
        best = None
        for x0 in (starts or [self.x0]):
            x0 = np.clip(x0, self.lo, self.hi)
            r = minimize(fun, x0, method="SLSQP", bounds=list(zip(self.lo, self.hi)), constraints=self.cons(extra),
                         options={"maxiter": 500, "ftol": 1e-12})
            if self.ok(r.x) and all(c["fun"](r.x) >= -1e-6 if c["type"] == "ineq" else abs(c["fun"](r.x)) < 1e-6 for c in extra) \
                    and (best is None or r.fun < best[0]):
                best = (r.fun, r.x)
        return None if best is None else self.clean(best[1])

    def ok(self, w, tol=1e-6):
        return bool(np.isfinite(w).all() and abs(w.sum() - 1) < tol and (w >= self.lo - tol).all() and (w <= self.hi + tol).all()
                    and (len(self.b) == 0 or (self.A @ w <= self.b + tol).all()))

    def clean(self, w):
        w = np.clip(w, self.lo, self.hi)
        w = np.where(w < 1e-7, 0.0, w)
        return w / w.sum()

    def project(self, target, start=None):
        """The feasible weights closest to `target` (target itself when it is feasible)."""
        from scipy.optimize import minimize
        if self.ok(target):
            return target
        x0 = start if start is not None else self.x0
        r = minimize(lambda w: ((w - target) ** 2).sum(), x0, jac=lambda w: 2 * (w - target), method="SLSQP",
                     bounds=list(zip(self.lo, self.hi)), constraints=self.cons(), options={"maxiter": 500, "ftol": 1e-14})
        return self.clean(r.x) if self.ok(r.x) else None

    def vol(self, w):
        return float(np.sqrt(max(w @ self.cov @ w, 0.0)))

    def downside(self, w):
        ex = self.R @ w - self.rf / 12
        return float(np.sqrt((np.minimum(ex, 0) ** 2).mean()) * np.sqrt(12))

    def cvar(self, w, beta=0.95):
        """Historical CVaR of monthly returns: the average loss in the worst (1 - beta) of months."""
        loss = np.sort(-(self.R @ w))[::-1]
        k = max(1, int(np.ceil((1 - beta) * len(loss))))
        return float(loss[:k].mean())

    def max_return(self):
        return self._lp(-self.mu)

    # -- objectives
    def weights(self, method: str, target: float | None = None):
        n = self.n
        if method == "equal":
            return self.project(np.full(n, 1 / n))
        if method == "inverse_vol":
            iv = 1 / self.sd
            return self.project(iv / iv.sum())
        if method == "min_variance":
            return self.solve(lambda w: w @ self.cov @ w * 100)
        if method == "max_sharpe":
            wr = self.max_return()
            return self.solve(lambda w: -(w @ self.mu - self.rf) / max(self.vol(w), 1e-9),
                              starts=[self.x0] + ([wr] if wr is not None else []))
        if method == "max_sortino":
            starts = [x for x in (self.x0, self.max_return(), self.weights("max_sharpe")) if x is not None]
            return self.solve(lambda w: -(w @ self.mu - self.rf) / max(self.downside(w), 1e-9), starts=starts)
        if method == "max_diversification":
            return self.solve(lambda w: -(w @ self.sd) / max(self.vol(w), 1e-9))
        if method == "risk_parity":
            iv = 1 / self.sd
            p = self.project(iv / iv.sum())

            def fun(w):
                v = w @ self.cov @ w
                rc = w * (self.cov @ w) / max(v, 1e-16)
                return float(((rc - 1 / n) ** 2).sum()) * 1e4
            return self.solve(fun, starts=[x for x in (p, self.x0) if x is not None])
        if method == "min_cvar":
            return self.min_cvar()
        if method == "target_return":
            if target is None:
                return None
            wr = self.max_return()
            if wr is None or wr @ self.mu < target - 1e-9:
                return None
            return self.solve(lambda w: w @ self.cov @ w * 100,
                              extra=[{"type": "ineq", "fun": lambda w: w @ self.mu - target}], starts=[wr, self.x0])
        if method == "target_vol":
            if target is None:
                return None
            mv = self.weights("min_variance")
            if mv is None or self.vol(mv) > target + 1e-9:
                return None
            return self.solve(lambda w: -(w @ self.mu),
                              extra=[{"type": "ineq", "fun": lambda w: target ** 2 - w @ self.cov @ w}], starts=[mv])
        raise ValueError(f"unknown method {method}")

    def min_cvar(self, beta: float = 0.95):
        """Rockafellar-Uryasev linear programme on the historical monthly scenarios."""
        from scipy.optimize import linprog
        T, n = self.R.shape
        c = np.concatenate([np.zeros(n), [1.0], np.full(T, 1 / ((1 - beta) * T))])
        A = [np.hstack([-self.R, -np.ones((T, 1)), -np.eye(T)])]        # loss - alpha - u <= 0
        b = [np.zeros(T)]
        if len(self.b):
            A.append(np.hstack([self.A, np.zeros((len(self.A), 1 + T))]))
            b.append(self.b)
        Aeq = np.concatenate([np.ones(n), [0.0], np.zeros(T)])[None, :]
        bnds = list(zip(self.lo, self.hi)) + [(None, None)] + [(0, None)] * T
        r = linprog(c, A_ub=np.vstack(A), b_ub=np.concatenate(b), A_eq=Aeq, b_eq=[1.0], bounds=bnds, method="highs")
        return self.clean(r.x[:n]) if r.status == 0 else None

    def describe(self, w) -> dict:
        v = self.vol(w)
        rc = w * (self.cov @ w) / max(v * v, 1e-16)
        ret = float(w @ self.mu)
        dd = self.downside(w)
        return {"weights": {t: float(x) for t, x in zip(self.t, w) if x > 1e-4},
                "exp_return": ret, "exp_vol": v, "exp_sharpe": (ret - self.rf) / v if v > 0 else np.nan,
                "exp_sortino": (ret - self.rf) / dd if dd > 0 else np.nan, "cvar_95_monthly": self.cvar(w),
                "diversification_ratio": float(w @ self.sd) / v if v > 0 else np.nan,
                "risk_contributions": {t: float(x) for t, x, wi in zip(self.t, rc, w) if wi > 1e-4}}


def _monthly_stats(r: pd.Series, rf_monthly: pd.Series | None = None) -> dict:
    r = r.dropna()
    if len(r) < 2:
        return {}
    g = (1 + r).cumprod()
    years = len(r) / 12
    ex = r - (rf_monthly.reindex(r.index).fillna(0) if rf_monthly is not None else 0)
    sd = r.std()
    dd = (g / np.maximum(g.cummax(), 1.0) - 1).min()
    down = np.sqrt((np.minimum(ex, 0) ** 2).mean())
    return {"cagr": g.iloc[-1] ** (1 / years) - 1 if g.iloc[-1] > 0 else np.nan, "volatility": sd * np.sqrt(12),
            "sharpe": ex.mean() / sd * np.sqrt(12) if sd > 0 else np.nan,
            "sortino": ex.mean() * 12 / (down * np.sqrt(12)) if down > 0 else np.nan,
            "max_drawdown": min(float(dd), 0.0), "growth": float(g.iloc[-1])}


def _rf_monthly(index: pd.DatetimeIndex) -> pd.Series:
    tb = data.tbill_rate()
    if tb.empty:
        return pd.Series(0.0, index=index)
    me = tb.resample("ME").mean()
    return me.reindex(me.index.union(index)).ffill().reindex(index).fillna(0.0) / 12


def _methods(target_return, target_vol) -> list[str]:
    return [m for m in OPT_METHODS if not (m == "target_return" and target_return is None)
            and not (m == "target_vol" and target_vol is None)]


def optimize(tickers: list[str], start: str | None = None, end: str | None = None, max_weight: float = 1.0,
             min_weight: float = 0.0, test_start: str | None = None, points: int = 30,
             constraints=None, target_return: float | None = None, target_vol: float | None = None,
             rolling_months: int | None = None, lookback_months: int = 60, rebalance: str = "quarterly") -> dict:
    """Long-only portfolio optimisation on monthly total returns.

    Objectives: max Sharpe, min variance, max Sortino, min CVaR (95%), risk parity (equal risk
    contribution), max diversification ratio, target return (min vol s.t. return >= x), target
    volatility (max return s.t. vol <= x), inverse volatility and equal weight - all subject to
    per-asset min/max weights and group constraints ("SPY+QQQ <= 70%").

    test_start: estimate before it and evaluate after it (daily, out of sample).
    rolling_months: walk-forward - every N months re-optimise on the trailing `lookback_months` and
    hold the weights for the next N months; the stitched out-of-sample curve is compared with the
    static (full-period, hindsight) weights over the same months.
    """
    tickers = list(dict.fromkeys(data.canonical(t) for t in tickers))
    if len(tickers) < 2:
        raise ValueError("Give at least two tickers.")
    px = pd.concat({t: data.load(t)["adj_close"] for t in tickers}, axis=1).dropna()
    if start:
        px = px[px.index >= pd.Timestamp(start)]
    if end:
        px = px[px.index <= pd.Timestamp(end)]
    fit = px if not test_start else px[px.index < pd.Timestamp(test_start)]
    mr = metrics.monthly_returns_frame(fit)
    if len(mr) < 24:
        raise ValueError("Need at least 24 months of overlapping history for these tickers.")
    bounds, groups = parse_constraints(constraints or [], tickers)
    rfm = _rf_monthly(mr.index)
    rf = float(rfm.mean() * 12)
    opt = _Opt(mr.to_numpy(), tickers, rf, bounds, groups, min_weight, max_weight)
    ports = {}
    notes = []
    for m in _methods(target_return, target_vol):
        tgt = target_return if m == "target_return" else target_vol if m == "target_vol" else None
        w = opt.weights(m, tgt)
        name = OPT_METHODS[m] + (f" {tgt:.1%}" if tgt is not None else "")
        if w is None:
            notes.append(f"{name}: no portfolio meets the constraints" + (" and the target." if tgt is not None else "."))
            continue
        d = opt.describe(w)
        d["method"] = m
        d["rounded"] = round_weights(d["weights"])
        d["sentence"] = weights_sentence(d["weights"], rebalance)
        ports[name] = d
    # efficient frontier: the minimum volatility for each attainable return
    frontier = []
    w_min = opt.weights("min_variance")
    w_max = opt.max_return()
    if w_min is not None and w_max is not None:
        lo_r, hi_r = float(w_min @ opt.mu), float(w_max @ opt.mu)
        prev = w_min
        for target in np.linspace(lo_r, hi_r, points):
            w = opt.solve(lambda w: w @ opt.cov @ w * 100,
                          extra=[{"type": "eq", "fun": lambda w, t=target: w @ opt.mu - t}], starts=[prev, w_max])
            if w is not None:
                prev = w
                frontier.append({"return": float(w @ opt.mu), "vol": opt.vol(w), "weights": w.round(4).tolist()})
    assets = [{"ticker": t, "return": float(opt.mu[i]), "vol": float(opt.sd[i])} for i, t in enumerate(tickers)]
    fin = lambda x: None if not np.isfinite(x) else float(x)  # noqa: E731
    out = {"tickers": tickers, "fit_start": mr.index[0].date(), "fit_end": mr.index[-1].date(), "rf": rf,
           "portfolios": ports, "frontier": frontier, "assets": assets, "notes": notes,
           "constraints": {"min_weight": min_weight, "max_weight": max_weight,
                           "bounds": {t: [fin(a), fin(b)] for t, (a, b) in bounds.items()},
                           "groups": groups, "target_return": target_return, "target_vol": target_vol},
           "correlation": mr.corr().round(3).to_numpy().tolist()}
    if test_start:
        test = px[px.index >= pd.Timestamp(test_start)]
        out["test_start"] = test.index[0].date() if len(test) else None
        out["test"] = {}
        for name, p in ports.items():
            w = pd.Series(p["weights"]).reindex(tickers).fillna(0)
            ret = (test.pct_change().fillna(0) @ w)
            eq = 10_000 * (1 + ret).cumprod()
            if len(eq) > 20:
                st = metrics.equity_stats(eq)
                out["test"][name] = {"cagr": st["cagr"], "volatility": st["volatility"], "sharpe": st["sharpe"], "max_drawdown": st["max_drawdown"]}
    if rolling_months:
        out["rolling"] = rolling_optimize(px, tickers, int(rolling_months), int(lookback_months), bounds, groups,
                                          min_weight, max_weight, target_return, target_vol, ports)
    return out


def rolling_optimize(px: pd.DataFrame, tickers: list[str], every: int, lookback: int, bounds, groups,
                     min_weight, max_weight, target_return, target_vol, static: dict) -> dict:
    """Walk-forward re-optimisation on monthly returns: every `every` months, fit on the trailing
    `lookback` months only and hold those weights (rebalanced monthly) until the next re-fit."""
    if every < 1 or lookback < 12:
        raise ValueError("Re-optimise at least every month, on at least 12 months of history.")
    mr = metrics.monthly_returns_frame(px)
    if len(mr) < lookback + 6:
        raise ValueError(f"Walk-forward with a {lookback}-month lookback needs more than {lookback + 6} months of history; there are {len(mr)}.")
    rfm = _rf_monthly(mr.index)
    methods = _methods(target_return, target_vol)
    R = mr.to_numpy()
    oos = {m: np.full(len(mr), np.nan) for m in methods}
    hist = {m: [] for m in methods}
    for k in range(lookback, len(mr), every):
        try:
            o = _Opt(R[k - lookback:k], tickers, float(rfm.iloc[k - lookback:k].mean() * 12), bounds, groups,
                     min_weight, max_weight)
        except ValueError:
            continue
        for m in methods:
            tgt = target_return if m == "target_return" else target_vol if m == "target_vol" else None
            w = o.weights(m, tgt)
            if w is None and m in ("target_return", "target_vol"):
                # target out of reach in this window: the closest attainable portfolio
                w = o.max_return() if m == "target_return" else o.weights("min_variance")
                w = o.clean(w) if w is not None else None
            if w is None:
                continue
            seg = slice(k, min(k + every, len(mr)))
            oos[m][seg] = R[seg] @ w
            hist[m].append({"date": mr.index[k].date(), "weights": {t: round(float(x), 4) for t, x in zip(tickers, w) if x > 1e-4}})
    idx = mr.index[lookback:]
    curves, stats = {}, {}
    for m in methods:
        name = next((k for k, p in static.items() if p.get("method") == m), OPT_METHODS[m])
        r = pd.Series(oos[m][lookback:], index=idx)
        if r.isna().all():
            continue
        r = r.fillna(0.0)
        curves[name] = (10_000 * (1 + r).cumprod()).round(2).tolist()
        st = {"rolling": _monthly_stats(r, rfm)}
        if name in static:
            w = pd.Series(static[name]["weights"]).reindex(tickers).fillna(0).to_numpy()
            rs = pd.Series(R[lookback:] @ w, index=idx)
            curves[name + " (static)"] = (10_000 * (1 + rs).cumprod()).round(2).tolist()
            st["static"] = _monthly_stats(rs, rfm)
        stats[name] = st
    return {"every_months": every, "lookback_months": lookback, "start": idx[0].date(), "end": idx[-1].date(),
            "dates": [d.strftime("%Y-%m-%d") for d in idx], "curves": curves, "stats": stats,
            "weights_history": {next((k for k, p in static.items() if p.get("method") == m), OPT_METHODS[m]): h
                                for m, h in hist.items() if h},
            "note": "Static weights are fitted on the whole period, so they know the future; rolling weights only use "
                    "the trailing window. The gap between them is the cost of estimating from the past."}
