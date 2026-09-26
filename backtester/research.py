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
CONNECTORS = {"than", "to", "of", "is", "are", "by", "at", "from", "over", "as", "with", "when", "if", "and", "or"}
FILLERS = {"if", "when", "and", "or", "is", "the", "of"}
UNITS = {"day", "days", "d", "week", "weeks", "month", "months", "year", "years", "bar", "bars"}
INDICATOR_WORDS = {"rsi", "sma", "ema", "wma", "moving", "average", "return", "returns", "momentum", "volatility",
                   "high", "low", "drawdown", "max", "stdev", "standard", "cumulative", "price", "change", "macd"}


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
    # a lookback followed by its unit and the indicator names what it measures: "{10,14} day RSI",
    # "{10,14}-day RSI", "{3,6} month momentum"
    after = re.findall(r"[^\s,;]+", text[b:])[:3]
    tail = []
    unit = re.sub(r"^.*?-", "", word[len(m.group(0)):]) if word != m.group(0) else ""
    if not unit and after and after[0].lower().rstrip(".") in UNITS:
        unit, after = after[0], after[1:]
        tail.append(unit)
    if unit.lower().rstrip(".") in UNITS and after and after[0].lower().strip("()") in INDICATOR_WORDS:
        tail.append(after[0])
        if len(after) > 1 and after[0].lower() == "moving" and after[1].lower().startswith("average"):
            tail.append(after[1])
    else:
        tail = []
    before = [w.rstrip(',.;:') for w in text[:a].split()]  # free-standing or only a suffix ("{1..3}%"): prefix the
    while before and before[-1].lower() in ARTICLES:     # previous word, skipping articles ("buy the {3,5}" -> "buy {3,5}")
        before.pop()
    lead = []
    while before and len(lead) < 3:
        w = before.pop()
        lead.insert(0, w)
        if w.lower() not in CONNECTORS:  # "than {75..85}" says nothing on its own: "greater than {75..85}"
            break
    if tail and lead and lead[-1].lower() in FILLERS | CONNECTORS | {"its", "their"}:
        lead = []
    return " ".join(lead + [word] + tail) if lead and lead[-1] else " ".join([word] + tail)


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
    "max_return_over_maxdd": "Max return / max drawdown", "omega": "Max Omega",
    "target_return": "Target return", "target_vol": "Target volatility",
    "inverse_vol": "Inverse volatility", "equal": "Equal weight",
    "min_tracking_error": "Min tracking error", "max_information_ratio": "Max information ratio",
}
BENCH_METHODS = ("min_tracking_error", "max_information_ratio")
# objectives that only need expected returns and covariances: the ones the resampled frontier re-solves
RESAMPLABLE = ("max_sharpe", "min_variance", "target_return", "target_vol", "max_diversification", "risk_parity",
               "min_tracking_error", "max_information_ratio")
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
                 min_weight: float = 0.0, max_weight: float = 1.0, omega_threshold: float = 0.0,
                 bench: np.ndarray | None = None, wb: np.ndarray | None = None, te_target: tuple | None = None):
        """bench: the benchmark's monthly returns over the same months (a ticker or blend outside the universe);
        wb: the benchmark as weights of these tickers (an investable benchmark: tracking error is then exact,
        sqrt((w - wb)' cov (w - wb))). te_target: ("return", x) or ("active", x), the return floor of the
        minimum tracking error portfolio."""
        self.R = R
        self.omega_l = (1 + float(omega_threshold)) ** (1 / 12) - 1   # monthly threshold of the Omega ratio
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
        self.wb = None if wb is None else np.asarray(wb, float)
        self.bench = None if bench is None or wb is not None else np.asarray(bench, float)
        self.te_target = te_target
        if self.wb is not None:
            self.mu_b = float(self.wb @ self.mu)
        elif self.bench is not None:
            self.mu_b = float(self.bench.mean() * 12)
            full = np.cov(np.column_stack([R, self.bench]), rowvar=False) * 12
            self.cb, self.vb = full[:n, n], float(full[n, n])

    @property
    def has_bench(self) -> bool:
        return self.wb is not None or self.bench is not None

    def te2(self, w):
        """Annualised tracking-error variance against the benchmark."""
        if self.wb is not None:
            d = w - self.wb
            return float(d @ self.cov @ d)
        return float(w @ self.cov @ w - 2 * w @ self.cb + self.vb)

    def te(self, w):
        return float(np.sqrt(max(self.te2(w), 0.0)))

    def active(self, w):
        return float(w @ self.mu - self.mu_b)

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

    def path(self, w):
        """(CAGR, max drawdown) of the historical monthly path of these weights, rebalanced monthly."""
        r = np.maximum(self.R @ w, -1.0)
        g = np.cumprod(1 + r)
        peak = np.maximum.accumulate(np.concatenate([[1.0], g]))[1:]
        mdd = float(min((g / peak - 1).min(), 0.0))
        cagr = float(g[-1] ** (12 / len(r)) - 1) if g[-1] > 0 else -1.0
        return cagr, mdd

    def omega(self, w, L=None):
        """Omega ratio of the monthly returns at the threshold L: E[(r - L)+] / E[(L - r)+]."""
        L = self.omega_l if L is None else L
        r = self.R @ w
        dn = np.maximum(L - r, 0).mean()
        return float(np.maximum(r - L, 0).mean() / dn) if dn > 0 else np.inf

    def _candidates(self, k: int = 3, samples: int = 400):
        """Feasible starting points for the path-dependent objectives: the classic portfolios plus the best of a
        random (Dirichlet) sample of feasible weights, ranked by `self._score`."""
        base = [self.x0, self.max_return(), self.weights("min_variance"), self.weights("max_sharpe")]
        base = [x for x in base if x is not None]
        rng = np.random.default_rng(3)
        pool = [x for x in rng.dirichlet(np.ones(self.n), samples) if self.ok(x)]
        pool.sort(key=self._score)
        return base + pool[:k]

    def max_return_over_maxdd(self):
        """Maximise CAGR / |max drawdown| on the historical monthly path. The objective is exact but only
        piecewise smooth (the drawdown's trough and peak switch between months), so it is searched from several
        feasible starts (the classic portfolios and the best random feasible weights) with SLSQP on
        finite-difference gradients, and the best result on the exact objective is kept."""
        def score(w):
            c, m = self.path(w)
            return -c / max(abs(m), 1e-4)
        self._score = score
        best = None
        for x0 in self._candidates():
            w = self.solve(score, starts=[x0])
            for cand in (w, self.clean(np.clip(x0, self.lo, self.hi))):
                if cand is not None and self.ok(cand) and (best is None or score(cand) < score(best)):
                    best = cand
        return best

    def max_omega(self):
        """Maximise the Omega ratio at the threshold. (Omega - 1) = (mean - L) / E[(L - r)+] is a linear-fractional
        programme, solved exactly as a linear programme with the Charnes-Cooper transform (y = t w,
        E[(L - r)+] scaled to 1); if that fails (e.g. a portfolio with no month below the threshold makes it
        unbounded) SLSQP maximises the ratio directly."""
        from scipy.optimize import linprog
        T, n = self.R.shape
        L = self.omega_l
        mu_m = self.R.mean(axis=0)
        # variables: y (n), t (1), d (T)
        c = np.concatenate([-mu_m, [L], np.zeros(T)])
        A = [np.hstack([-self.R, np.full((T, 1), L), -np.eye(T)])]           # L t - R y - d <= 0
        b = [np.zeros(T)]
        A.append(np.hstack([-np.eye(n), self.lo[:, None], np.zeros((n, T))]))   # lo t - y <= 0
        b.append(np.zeros(n))
        A.append(np.hstack([np.eye(n), -self.hi[:, None], np.zeros((n, T))]))   # y - hi t <= 0
        b.append(np.zeros(n))
        if len(self.b):
            A.append(np.hstack([self.A, -self.b[:, None], np.zeros((len(self.b), T))]))
            b.append(np.zeros(len(self.b)))
        Aeq = np.vstack([np.concatenate([np.zeros(n), [0.0], np.full(T, 1 / T)]),
                         np.concatenate([np.ones(n), [-1.0], np.zeros(T)])])
        r = linprog(c, A_ub=np.vstack(A), b_ub=np.concatenate(b), A_eq=Aeq, b_eq=[1.0, 0.0],
                    bounds=[(0, None)] * (n + 1 + T), method="highs")
        if r.status == 0 and r.x[n] > 1e-12:
            w = self.clean(r.x[:n] / r.x[n])
            if self.ok(w):
                return w
        self._score = lambda w: -min(self.omega(w), 1e6)
        return self.solve(self._score, starts=self._candidates())

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
        if method in BENCH_METHODS:
            if not self.has_bench:
                return None
            wr = self.max_return()
            wb = self.project(self.wb) if self.wb is not None else None
            starts = [x for x in (wb, self.x0, wr) if x is not None]
            if method == "min_tracking_error":
                extra = []
                if self.te_target:
                    kind, x = self.te_target
                    floor = x + (self.mu_b if kind == "active" else 0.0)
                    if wr is None or wr @ self.mu < floor - 1e-9:
                        return None
                    extra = [{"type": "ineq", "fun": lambda w: w @ self.mu - floor}]
                return self.solve(lambda w: self.te2(w) * 100, extra=extra, starts=starts)
            return self.solve(lambda w: -self.active(w) / max(self.te(w), 1e-6), starts=starts)
        if method == "max_return_over_maxdd":
            return self.max_return_over_maxdd()
        if method == "omega":
            return self.max_omega()
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
        cagr, mdd = self.path(w)
        return {"weights": {t: float(x) for t, x in zip(self.t, w) if x > 1e-4},
                "hist_cagr": cagr, "hist_max_drawdown": mdd,
                "return_over_maxdd": cagr / abs(mdd) if mdd < 0 else np.nan,
                "omega": self.omega(w), "omega_threshold": (1 + self.omega_l) ** 12 - 1,
                "exp_return": ret, "exp_vol": v, "exp_sharpe": (ret - self.rf) / v if v > 0 else np.nan,
                "exp_sortino": (ret - self.rf) / dd if dd > 0 else np.nan, "cvar_95_monthly": self.cvar(w),
                "diversification_ratio": float(w @ self.sd) / v if v > 0 else np.nan,
                "risk_contributions": {t: float(x) for t, x, wi in zip(self.t, rc, w) if wi > 1e-4},
                **({"tracking_error": self.te(w), "active_return": self.active(w),
                    "information_ratio": self.active(w) / self.te(w) if self.te(w) > 1e-9 else np.nan}
                   if self.has_bench else {})}


# ------------------------------------------------------------------ optimiser inputs: forecasts, Black-Litterman

_TK = r"[A-Za-z0-9^][A-Za-z0-9^.\-_]*"


def _items(x, seps=r"[;\n,]") -> list:
    if x is None:
        return []
    if isinstance(x, str):
        return [p.strip() for p in re.split(seps, x) if p.strip()]
    return list(x)


def parse_asset_values(items, known: list[str] | None = None, what: str = "value") -> dict[str, float]:
    """'SPY=7%, TLT=4%', 'SPY 7%; TLT 0.04', {'SPY': 0.07} or ['SPY:7%'] -> {ticker: fraction}."""
    if isinstance(items, dict):
        pairs = list(items.items())
    else:
        pairs = []
        for raw in _items(items):
            m = re.fullmatch(rf"\s*({_TK})\s*(?:=|:|\s)\s*(-?[\d.]+%?)\s*", str(raw))
            if not m:
                raise ValueError(f"Could not read the {what} {raw!r}. Write e.g. 'SPY=7%'.")
            pairs.append((m.group(1), m.group(2)))
    out = {}
    for t, v in pairs:
        t = data.canonical(str(t))
        if known is not None and t not in known:
            raise ValueError(f"The {what} for {t}: {t} is not among the tickers ({', '.join(known)}).")
        out[t] = _frac(v) if isinstance(v, str) else float(v)
    return out


def parse_correlations(items, known: list[str]) -> dict[tuple[str, str], float]:
    """'SPY/TLT=-0.2; SPY,GLD=0.1' (or a list, or {'SPY/TLT': -0.2}) -> {(a, b): rho}."""
    pairs = list(items.items()) if isinstance(items, dict) else [(None, x) for x in _items(items, r"[;\n]")]
    out = {}
    for k, raw in pairs:
        text = f"{k}={raw}" if k is not None else str(raw)
        m = re.fullmatch(rf"\s*({_TK})\s*[/,&\s]\s*({_TK})\s*[=:]?\s*(-?[\d.]+)\s*", text)
        if not m:
            raise ValueError(f"Could not read the correlation {text!r}. Write e.g. 'SPY/TLT=-0.2'.")
        a, b, v = data.canonical(m.group(1)), data.canonical(m.group(2)), float(m.group(3))
        for t in (a, b):
            if t not in known:
                raise ValueError(f"The correlation {text.strip()!r} names {t}, which is not among the tickers.")
        if a == b or not -1 <= v <= 1:
            raise ValueError(f"The correlation {text.strip()!r} must link two different tickers with a value in [-1, 1].")
        out[(a, b)] = v
    return out


_VIEW_CONF = (r"(?:\s*(?:@|,|with|at)?\s*(?:(?:a )?confidence(?: of)?|conf\.?)?\s*[:=]?\s*\(?(?P<c>[\d.]+%?)\)?"
              r"(?:\s*(?:confidence|conf\.?|confident|sure))?)?")


def parse_views(items, known: list[str]) -> list[dict]:
    """Black-Litterman views, one per item (a list, or text separated by ';' or new lines):

        absolute: 'SPY = 8%', 'SPY 8%', 'SPY will return 8%'
        relative: 'QQQ > SPY by 2%', 'QQQ - SPY = 2%', 'QQQ outperforms SPY by 2%', 'TLT underperforms SPY by 3%'

    each optionally followed by a confidence, '@ 60%' / 'confidence 60%' / '(60%)'; the default is 50%. Returns
    [{"p": {ticker: coefficient}, "q": annual value, "confidence": c, "kind": "absolute"|"relative", "text"}]."""
    out = []
    for raw in _items(items, r"[;\n]"):
        if isinstance(raw, dict):
            p = {data.canonical(k): float(v) for k, v in (raw.get("p") or {}).items()}
            view = {"p": p, "q": float(raw["q"]), "confidence": float(raw.get("confidence", 0.5)),
                    "kind": raw.get("kind") or ("absolute" if len(p) == 1 else "relative"), "text": raw.get("text") or str(raw)}
        else:
            text = str(raw).strip()
            m = re.fullmatch(rf"(?i)\s*(?P<a>{_TK}?)\s*(?P<op>>|<| - |outperforms?|beats?|underperforms?|lags?|trails?)\s*(?P<b>{_TK})"
                             rf"\s*(?:by|=|:)?\s*(?P<q>-?[\d.]+%?)(?:\s*(?:a|per) year)?{_VIEW_CONF}\s*", text)
            if m:
                a, b = data.canonical(m.group("a")), data.canonical(m.group("b"))
                op = m.group("op").strip().lower()
                sign = -1.0 if op == "<" or re.match(r"under|lag|trail", op) else 1.0
                view = {"p": {a: 1.0, b: -1.0}, "q": sign * _frac(m.group("q")), "kind": "relative", "text": text}
            else:
                m = re.fullmatch(rf"(?i)\s*(?P<a>{_TK}?)\s*(?:=|:|will return|returns?|expected(?: return)?(?: of)?)?\s*(?P<q>-?[\d.]+%?)"
                                 rf"(?:\s*(?:a|per) year)?{_VIEW_CONF}\s*", text)
                if not m:
                    raise ValueError(f"Could not read the view {text!r}. Write e.g. 'SPY = 8% @ 60%' (absolute) or "
                                     "'QQQ > SPY by 2% @ 50%' (relative).")
                view = {"p": {data.canonical(m.group("a")): 1.0}, "q": _frac(m.group("q")), "kind": "absolute", "text": text}
            c = m.group("c")
            view["confidence"] = _frac(c) if c else 0.5
        for t in view["p"]:
            if t not in known:
                raise ValueError(f"The view {view['text']!r} names {t}, which is not among the tickers.")
        if not 0 < view["confidence"] <= 1:
            raise ValueError(f"The view {view['text']!r}: the confidence must be above 0% and at most 100%.")
        if view["kind"] == "relative" and len(view["p"]) < 2:
            raise ValueError(f"The view {view['text']!r} compares a ticker with itself.")
        out.append(view)
    return out


def black_litterman(cov: np.ndarray, w_eq: np.ndarray, P: np.ndarray | None = None, Q: np.ndarray | None = None,
                    confidence: np.ndarray | None = None, tau: float = 0.05, delta: float = 2.5) -> dict:
    """Black-Litterman posterior (excess returns, annual).

    Prior: the implied equilibrium excess returns pi = delta * cov @ w_eq (reverse optimisation of the
    equilibrium weights). Views P @ mu = Q + e, e ~ N(0, Omega) with Omega_kk = (1 - c_k) / c_k * p_k (tau cov) p_k'
    (Idzorek's confidence in the closed form PyPortfolioOpt uses: 50% = the He-Litterman default diag(P tau cov P'),
    100% = the view holds exactly). Posterior mean and the covariance of that estimate:

        mu  = pi + tau cov P' (P tau cov P' + Omega)^-1 (Q - P pi)
        M   = tau cov - tau cov P' (P tau cov P' + Omega)^-1 P tau cov

    (the same as [(tau cov)^-1 + P' Omega^-1 P]^-1 [(tau cov)^-1 pi + P' Omega^-1 Q], written so that a 100% view,
    Omega = 0, is allowed). The covariance of returns used afterwards is cov + M."""
    cov = np.asarray(cov, float)
    w_eq = np.asarray(w_eq, float)
    pi = delta * cov @ w_eq
    tS = tau * cov
    if P is None or not len(P):
        return {"pi": pi, "mu": pi.copy(), "cov": cov + tS, "M": tS, "omega": np.zeros((0, 0))}
    P = np.atleast_2d(np.asarray(P, float))
    Q = np.asarray(Q, float)
    c = np.asarray(confidence if confidence is not None else np.full(len(Q), 0.5), float)
    base = np.einsum("ij,jk,ik->i", P, tS, P)
    omega = np.diag((1 - c) / c * base)
    K = P @ tS @ P.T + omega
    if np.linalg.matrix_rank(K) < len(K):
        raise ValueError("The views contradict or repeat each other at 100% confidence; lower a confidence or drop a view.")
    mu = pi + tS @ P.T @ np.linalg.solve(K, Q - P @ pi)
    M = tS - tS @ P.T @ np.linalg.solve(K, P @ tS)
    return {"pi": pi, "mu": mu, "cov": cov + M, "M": M, "omega": omega}


def _nearest_psd(S: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    v, V = np.linalg.eigh((S + S.T) / 2)
    return (V * np.maximum(v, eps)) @ V.T


def _chol(S: np.ndarray) -> np.ndarray:
    k = len(S)
    for j in (0.0, 1e-12, 1e-10, 1e-8):
        try:
            return np.linalg.cholesky(S + np.eye(k) * j * max(1e-12, float(np.trace(S)) / k))
        except np.linalg.LinAlgError:
            continue
    return np.linalg.cholesky(_nearest_psd(S, 1e-10))


def retarget(R: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Re-shape the historical scenarios R (T x k) so that their sample mean is `mean` and their sample covariance
    is `cov` exactly (both per period): Y = mean + (R - m) Lh^-T Ln', with Lh Lh' the sample covariance of R and
    Ln Ln' = cov. The months keep their joint ordering and shape (fat tails, co-crashes), so the scenario-based
    objectives (Sortino, CVaR, Omega, drawdown) stay consistent with the forecast means and covariances."""
    m = R.mean(axis=0)
    Lh = _chol(np.cov(R, rowvar=False).reshape(R.shape[1], R.shape[1]))
    Ln = _chol(cov)
    X = np.linalg.solve(Lh, (R - m).T).T          # (R - m) Lh^-T
    return mean + X @ Ln.T


def _market_caps(tickers: list[str], asof) -> dict[str, float]:
    out = {}
    for t in tickers:
        try:
            mc = data.market_cap(t)
        except Exception:  # noqa: BLE001 - no share data is the same as no market cap
            mc = pd.Series(dtype=float)
        mc = mc[mc.index <= pd.Timestamp(asof)].dropna() if len(mc) else mc
        if len(mc):
            out[t] = float(mc.iloc[-1])
    return out


def _bench_weights(benchmark) -> dict[str, float]:
    """'SPY', '60% SPY 40% AGG', 'SPY 60 AGG 40', 'SPY:60,AGG:40' or {'SPY': 0.6, 'AGG': 0.4} -> weights adding to 1."""
    from .montecarlo import parse_weights
    if isinstance(benchmark, dict):
        w = {data.canonical(k): float(v) for k, v in benchmark.items()}
    else:
        w = parse_weights(str(benchmark))
    if any(v < 0 for v in w.values()) or sum(w.values()) <= 0:
        raise ValueError("Benchmark weights must be positive.")
    tot = sum(w.values())
    return {k: v / tot for k, v in w.items()}


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


def _methods(target_return, target_vol, methods=None, bench: bool = False) -> list[str]:
    if methods:
        bad = [m for m in methods if m not in OPT_METHODS]
        if bad:
            raise ValueError(f"Unknown objective(s) {', '.join(bad)}; choose from {', '.join(OPT_METHODS)}.")
        if not bench and any(m in BENCH_METHODS for m in methods):
            raise ValueError("Min tracking error and max information ratio need a benchmark (a ticker or a blend).")
    return [m for m in OPT_METHODS if not (m == "target_return" and target_return is None)
            and not (m == "target_vol" and target_vol is None) and not (m in BENCH_METHODS and not bench)
            and (not methods or m in methods)]


def _inputs(R: np.ndarray, tickers: list[str], rf: float, fit_end, expected_returns=None, expected_vols=None,
            correlations=None, views=None, prior=None, tau: float = 0.05, risk_aversion: float = 2.5,
            bench_extra: np.ndarray | None = None) -> dict:
    """The expected returns and covariance the optimiser uses (annual): historical by default; forecasts
    (expected returns, volatilities, correlations) replace the historical figures they name; Black-Litterman
    (a prior and/or views) replaces the expected returns with its posterior and the covariance with cov + M.
    Returns the (re-shaped) monthly scenarios and a description for the report."""
    n = len(tickers)
    mu_h = R.mean(axis=0) * 12
    cov_h = np.cov(R, rowvar=False).reshape(n, n) * 12
    sd_h = np.sqrt(np.diag(cov_h))
    corr_h = cov_h / np.outer(sd_h, sd_h)
    er = parse_asset_values(expected_returns, tickers, "expected return") if expected_returns else {}
    ev = parse_asset_values(expected_vols, tickers, "volatility") if expected_vols else {}
    cr = parse_correlations(correlations, tickers) if correlations else {}
    vw = parse_views(views, tickers) if views else []
    use_bl = bool(vw) or prior not in (None, "", "none")
    if er and use_bl:
        raise ValueError("Give either expected returns or Black-Litterman views/prior, not both: the Black-Litterman "
                         "posterior replaces the expected returns.")
    if any(v <= 0 for v in ev.values()):
        raise ValueError("Volatilities must be positive.")
    notes: list[str] = []
    sd = np.array([ev.get(t, sd_h[j]) for j, t in enumerate(tickers)])
    corr = corr_h.copy()
    for (a, b), v in cr.items():
        i, j = tickers.index(a), tickers.index(b)
        corr[i, j] = corr[j, i] = v
    if cr and np.linalg.eigvalsh(corr).min() < -1e-10:
        raise ValueError("The correlations given are inconsistent with each other and the historical ones (the correlation "
                         "matrix is not positive semi-definite). Change them, or give fewer.")
    cov = corr * np.outer(sd, sd)
    mu = np.array([er.get(t, mu_h[j]) for j, t in enumerate(tickers)])
    out = {"source": "historical", "notes": notes}
    if er or ev or cr:
        out["source"] = "forecast"
        miss = [t for t in tickers if t not in er] if er else []
        if miss:
            notes.append(f"No expected return given for {', '.join(miss)}: the historical mean was used.")
    if use_bl:
        pri = prior if prior not in (None, "", "none") else "market_cap"
        if isinstance(pri, str) and pri.lower().replace("-", "_").replace(" ", "_") in ("market_cap", "market", "cap", "caps"):
            caps = _market_caps(tickers, fit_end)
            if len(caps) == n and sum(caps.values()) > 0:
                w_eq = np.array([caps[t] for t in tickers]) / sum(caps.values())
                pname = f"market capitalisation on {pd.Timestamp(fit_end).date()}"
            elif prior in (None, "", "none"):
                w_eq = np.full(n, 1 / n)
                pname = "equal weights"
                notes.append("Black-Litterman prior: no market capitalisation for " + ", ".join(t for t in tickers if t not in caps)
                             + " (funds have no share counts here), so the equilibrium weights are equal; give --prior "
                               "weights (e.g. 'SPY=60%, TLT=40%') for a market portfolio.")
            else:
                raise ValueError("No market capitalisation for " + ", ".join(t for t in tickers if t not in caps)
                                 + ": give the equilibrium (prior) weights instead, e.g. 'SPY=60%, TLT=40%', or 'equal'.")
        elif isinstance(pri, str) and pri.lower() == "equal":
            w_eq, pname = np.full(n, 1 / n), "equal weights"
        else:
            pw = parse_asset_values(pri, tickers, "prior weight")
            if any(v < 0 for v in pw.values()) or sum(pw.values()) <= 0:
                raise ValueError("Prior (equilibrium) weights must be positive.")
            w_eq = np.array([pw.get(t, 0.0) for t in tickers]) / sum(pw.values())
            pname = "given weights (market capitalisations or a policy portfolio)"
        P = np.array([[v["p"].get(t, 0.0) for t in tickers] for v in vw]) if vw else None
        # views state total returns; the model works in excess returns (a relative view is unchanged)
        Q = np.array([v["q"] - (rf if v["kind"] == "absolute" else 0.0) for v in vw]) if vw else None
        bl = black_litterman(cov, w_eq, P, Q, np.array([v["confidence"] for v in vw]) if vw else None, tau, risk_aversion)
        mu, cov = bl["mu"] + rf, bl["cov"]
        out["source"] = "black_litterman"
        out["black_litterman"] = {
            "prior": pname, "tau": tau, "risk_aversion": risk_aversion,
            "prior_weights": {t: float(x) for t, x in zip(tickers, w_eq)},
            "equilibrium_returns": {t: float(x + rf) for t, x in zip(tickers, bl["pi"])},
            "posterior_returns": {t: float(x) for t, x in zip(tickers, mu)},
            "historical_returns": {t: float(x) for t, x in zip(tickers, mu_h)},
            "views": [{"text": v["text"], "kind": v["kind"], "p": v["p"], "q": v["q"], "confidence": v["confidence"],
                       "omega": float(bl["omega"][k, k]),
                       "prior_value": float(sum(c * (bl["pi"][tickers.index(t)] + (rf if v["kind"] == "absolute" else 0))
                                                for t, c in v["p"].items())),
                       "posterior_value": float(sum(c * (mu[tickers.index(t)] - (0 if v["kind"] == "absolute" else rf))
                                                    for t, c in v["p"].items()))} for k, v in enumerate(vw)],
        }
    changed = out["source"] != "historical"
    k = n + (1 if bench_extra is not None else 0)
    if changed:
        J = R if bench_extra is None else np.column_stack([R, bench_extra])
        cov_j = np.cov(J, rowvar=False).reshape(k, k) * 12
        mean_j = J.mean(axis=0) * 12
        mean_j[:n] = mu
        if bench_extra is not None:
            # the benchmark keeps its own mean and volatility and its historical correlations with each asset
            sdj = np.sqrt(np.diag(cov_j))
            cj = cov_j / np.outer(sdj, sdj)
            new_sd = np.concatenate([np.sqrt(np.diag(cov)), [sdj[n]]])
            cov_j = cj * np.outer(new_sd, new_sd)
            cov_j[:n, :n] = cov
            if np.linalg.eigvalsh(cov_j).min() < -1e-12:
                cov_j = _nearest_psd(cov_j)
        cov_j[:n, :n] = cov
        Y = retarget(J, mean_j / 12, cov_j / 12)
        out["R"], out["bench"] = Y[:, :n], (Y[:, n] if bench_extra is not None else None)
    else:
        out["R"], out["bench"] = R, bench_extra
    out["mu"], out["cov"] = mu, cov
    out["table"] = [{"ticker": t, "hist_return": float(mu_h[j]), "hist_vol": float(sd_h[j]), "return": float(mu[j]),
                     "vol": float(np.sqrt(cov[j, j]))} for j, t in enumerate(tickers)]
    return out


def _resample(Rm: np.ndarray, bench: np.ndarray | None, draws: int, tickers, rf, bounds, groups, min_weight, max_weight,
              omega_threshold, wb, te_target, methods: list[str], targets: dict, base: "_Opt", points: int = 15,
              seed: int = 11) -> dict:
    """Michaud resampling: draw `draws` sets of T months from a multivariate normal with the inputs' means and
    covariances (the estimation error of a T-month sample), re-optimise each and average the weights. The
    resampled frontier averages the frontier portfolios rank by rank (the k-th of `points` equally spaced returns
    between each draw's minimum-variance and maximum-return portfolios) and is evaluated with the inputs."""
    T, n = Rm.shape
    J = Rm if bench is None else np.column_stack([Rm, bench])
    mean, cov = J.mean(axis=0), np.cov(J, rowvar=False).reshape(J.shape[1], J.shape[1])
    rng = np.random.default_rng(seed)
    L = _chol(cov)
    ws = {m: [] for m in methods}
    fr = []
    for _ in range(int(draws)):
        X = mean + rng.standard_normal((T, J.shape[1])) @ L.T
        try:
            o = _Opt(X[:, :n], tickers, rf, bounds, groups, min_weight, max_weight, omega_threshold,
                     bench=X[:, n] if bench is not None else None, wb=wb, te_target=te_target)
        except ValueError:
            continue
        for m in methods:
            w = o.weights(m, targets.get(m))
            if w is None and m in ("target_return", "target_vol", "min_tracking_error"):
                # the target is out of reach in this draw: its closest attainable portfolio
                w = o.max_return() if m != "target_vol" else o.weights("min_variance")
                w = o.clean(w) if w is not None else None
            if w is not None:
                ws[m].append(w)
        w0, w1 = o.weights("min_variance"), o.max_return()
        if w0 is None or w1 is None:
            continue
        row, prev = [], w0
        for tgt in np.linspace(float(w0 @ o.mu), float(w1 @ o.mu), points):
            w = o.solve(lambda w: w @ o.cov @ w * 100, extra=[{"type": "eq", "fun": lambda w, t=tgt: w @ o.mu - t}],
                        starts=[prev, w1])
            prev = w if w is not None else prev
            row.append(prev)
        fr.append(row)
    out = {"draws": int(draws), "portfolios": {}, "frontier": []}
    for m, lst in ws.items():
        if lst:
            W = np.array(lst)
            w = base.clean(W.mean(axis=0))
            out["portfolios"][m] = {"w": w, "sd": W.std(axis=0), "n": len(lst)}
    if fr:
        F = np.array(fr).mean(axis=0)
        out["frontier"] = [{"return": float(w @ base.mu), "vol": base.vol(w), "weights": w.round(4).tolist()} for w in F]
    return out


def optimize(tickers: list[str], start: str | None = None, end: str | None = None, max_weight: float = 1.0,
             min_weight: float = 0.0, test_start: str | None = None, points: int = 30,
             constraints=None, target_return: float | None = None, target_vol: float | None = None,
             rolling_months: int | None = None, lookback_months: int = 60, rebalance: str = "quarterly",
             methods: list[str] | None = None, omega_threshold: float = 0.0,
             expected_returns=None, expected_vols=None, correlations=None, views=None, prior=None,
             tau: float = 0.05, risk_aversion: float = 2.5, benchmark=None, target_active: float | None = None,
             resample: int = 0, resample_seed: int = 11) -> dict:
    """Long-only portfolio optimisation on monthly total returns.

    Objectives: max Sharpe, min variance, max Sortino, min CVaR (95%), risk parity (equal risk
    contribution), max diversification ratio, max return / max drawdown (CAGR over the worst drawdown of the
    historical monthly path), max Omega (at `omega_threshold`, an annual rate, default 0), target return
    (min vol s.t. return >= x), target volatility (max return s.t. vol <= x), inverse volatility and equal
    weight - all subject to per-asset min/max weights and group constraints ("SPY+QQQ <= 70%").
    `methods` limits the run to some of them (keys of OPT_METHODS).

    Inputs (default: the historical means and covariances of the fit period):
      expected_returns / expected_vols: {ticker: annual} (or "SPY=7%, TLT=4%") replace the historical figures;
      correlations: {"SPY/TLT": -0.2} or "SPY/TLT=-0.2; ..." replace single historical correlations.
      views / prior: Black-Litterman. prior = "market_cap" (default; equal weights with a note when a ticker has
      no market capitalisation), "equal", or weights {ticker: w}; views like "SPY = 8% @ 60%", "QQQ > SPY by 2%".
      tau and risk_aversion (delta) are the model's scalars. The posterior replaces the expected returns.
    Scenario-based objectives (Sortino, CVaR, Omega, return/drawdown) use the historical months re-shaped to the
    same means and covariances (see `retarget`).

    benchmark: a ticker or blend ("60% SPY 40% AGG"): adds min tracking error (subject to `target_active`, the
    excess return over the benchmark, or else `target_return`, when given) and max information ratio, and the
    tracking error / information ratio of every portfolio.
    resample: Michaud resampled efficiency with this many draws: "Resampled ..." versions of the mean-variance
    objectives and a resampled frontier.

    test_start: estimate before it and evaluate after it (daily, out of sample).
    rolling_months: walk-forward - every N months re-optimise on the trailing `lookback_months` and
    hold the weights for the next N months; the stitched out-of-sample curve is compared with the
    static (full-period, hindsight) weights over the same months.
    """
    tickers = list(dict.fromkeys(data.canonical(t) for t in tickers))
    if len(tickers) < 2:
        raise ValueError("Give at least two tickers.")
    bw = _bench_weights(benchmark) if benchmark not in (None, "", {}) else None
    load = tickers + [t for t in (bw or {}) if t not in tickers]
    px = pd.concat({t: data.load(t)["adj_close"] for t in load}, axis=1).dropna()
    if start:
        px = px[px.index >= pd.Timestamp(start)]
    if end:
        px = px[px.index <= pd.Timestamp(end)]
    fit = px if not test_start else px[px.index < pd.Timestamp(test_start)]
    mr_all = metrics.monthly_returns_frame(fit)
    mr = mr_all[tickers]
    if len(mr) < 24:
        raise ValueError("Need at least 24 months of overlapping history for these tickers"
                         + (" and the benchmark." if bw else "."))
    bounds, groups = parse_constraints(constraints or [], tickers)
    rfm = _rf_monthly(mr.index)
    rf = float(rfm.mean() * 12)
    wb, bench_r, bname = None, None, None
    if bw:
        bname = " ".join(f"{v:.0%} {k}" for k, v in bw.items()) if len(bw) > 1 else next(iter(bw))
        if set(bw) <= set(tickers):
            wb = np.array([bw.get(t, 0.0) for t in tickers])      # investable: a portfolio of these tickers
        else:
            bench_r = (mr_all[list(bw)] * pd.Series(bw)).sum(axis=1).to_numpy()   # rebalanced monthly
    te_target = ("active", float(target_active)) if target_active is not None else (
        ("return", float(target_return)) if target_return is not None and bw else None)
    inp = _inputs(mr.to_numpy(), tickers, rf, mr.index[-1], expected_returns, expected_vols, correlations, views, prior,
                  tau, risk_aversion, bench_r)
    opt = _Opt(inp["R"], tickers, rf, bounds, groups, min_weight, max_weight, omega_threshold,
               bench=inp["bench"], wb=wb, te_target=te_target)
    ports = {}
    notes = list(inp["notes"])
    infeasible = {}
    meths = _methods(target_return, target_vol, methods, bench=bw is not None)
    targets = {"target_return": target_return, "target_vol": target_vol}
    names = {}
    for m in meths:
        tgt = targets.get(m)
        w = opt.weights(m, tgt)
        name = OPT_METHODS[m] + (f" {tgt:.1%}" if tgt is not None else "") + (
            f" (threshold {omega_threshold:.1%}/yr)" if m == "omega" and omega_threshold else "")
        if m == "min_tracking_error" and te_target:
            name += (f" (≥ {te_target[1]:+.1%} over the benchmark)" if te_target[0] == "active" else f" (return ≥ {te_target[1]:.1%})")
        names[m] = name
        if w is None:
            msg = f"{name}: no portfolio meets the constraints" + (" and the target." if tgt is not None or (m == "min_tracking_error" and te_target) else ".")
            if m == "target_vol":
                mv = opt.weights("min_variance")
                if mv is not None:
                    infeasible["min_vol"] = opt.vol(mv)
                    msg = (f"{name}: out of reach - the minimum achievable volatility is {opt.vol(mv):.1%} "
                           f"(the minimum-variance portfolio{' under these constraints' if bounds or groups else ''}).")
            elif m in ("target_return", "min_tracking_error"):
                wr = opt.max_return()
                if wr is not None:
                    infeasible["max_return"] = float(wr @ opt.mu)
                    msg = (f"{name}: out of reach - the maximum achievable expected return is {float(wr @ opt.mu):.1%} "
                           f"({', '.join(f'{x:.0%} {t}' for t, x in zip(tickers, wr) if x > 1e-4)}).")
            notes.append(msg)
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
    resampled = None
    if resample:
        if not 5 <= int(resample) <= 2000:
            raise ValueError("Resample between 5 and 2,000 times.")
        rm = [m for m in meths if m in RESAMPLABLE]
        rs = _resample(opt.R, opt.bench, int(resample), tickers, rf, bounds, groups, min_weight, max_weight,
                       omega_threshold, wb, te_target, rm, targets, opt, seed=resample_seed)
        for m, r in rs["portfolios"].items():
            d = opt.describe(r["w"])
            d.update({"method": m, "resampled": r["n"], "weights_sd": {t: float(x) for t, x, wi in zip(tickers, r["sd"], r["w"]) if wi > 1e-4},
                      "rounded": round_weights(d["weights"]), "sentence": weights_sentence(d["weights"], rebalance)})
            ports["Resampled " + names.get(m, OPT_METHODS[m]).lower()[:1] + names.get(m, OPT_METHODS[m])[1:]] = d
        resampled = {"draws": rs["draws"], "frontier": rs["frontier"], "seed": resample_seed,
                     "note": f"Michaud resampling: {rs['draws']} simulated {len(mr)}-month histories drawn from the inputs "
                             "(multivariate normal), each optimised; the weights are averaged. The spread of the weights "
                             "across draws (±) shows how much of each allocation is estimation noise."}
    assets = [{"ticker": t, "return": float(opt.mu[i]), "vol": float(opt.sd[i])} for i, t in enumerate(tickers)]
    fin = lambda x: None if not np.isfinite(x) else float(x)  # noqa: E731
    out = {"tickers": tickers, "fit_start": mr.index[0].date(), "fit_end": mr.index[-1].date(), "rf": rf,
           "portfolios": ports, "frontier": frontier, "assets": assets, "notes": notes, "infeasible": infeasible,
           "omega_threshold": omega_threshold,
           "constraints": {"min_weight": min_weight, "max_weight": max_weight,
                           "bounds": {t: [fin(a), fin(b)] for t, (a, b) in bounds.items()},
                           "groups": groups, "target_return": target_return, "target_vol": target_vol,
                           "target_active": target_active},
           "correlation": mr.corr().round(3).to_numpy().tolist(),
           "inputs": {"source": inp["source"], "table": inp["table"]}}
    if inp["source"] != "historical":
        c = opt.cov / np.outer(opt.sd, opt.sd)
        out["inputs"]["correlation"] = c.round(3).tolist()
        if inp["source"] == "forecast":
            notes.append("Forecast inputs: the expected returns, volatilities and correlations given replace the historical "
                         "ones; the scenario-based objectives use the historical months re-shaped to match them.")
    if inp.get("black_litterman"):
        out["black_litterman"] = inp["black_litterman"]
    if bw:
        out["benchmark"] = {"name": bname, "weights": bw, "investable": wb is not None, "return": opt.mu_b,
                            "vol": float(np.sqrt(wb @ opt.cov @ wb)) if wb is not None else float(np.sqrt(opt.vb))}
    if resampled:
        out["resampled"] = resampled
    if test_start:
        test = px[px.index >= pd.Timestamp(test_start)]
        out["test_start"] = test.index[0].date() if len(test) else None
        out["test"] = {}
        for name, p in ports.items():
            w = pd.Series(p["weights"]).reindex(tickers).fillna(0)
            ret = (test[tickers].pct_change().fillna(0) @ w)
            eq = 10_000 * (1 + ret).cumprod()
            if len(eq) > 20:
                st = metrics.equity_stats(eq)
                out["test"][name] = {"cagr": st["cagr"], "volatility": st["volatility"], "sharpe": st["sharpe"], "max_drawdown": st["max_drawdown"]}
    if rolling_months:
        if inp["source"] != "historical":
            notes.append("Walk-forward: each window is fitted on its own trailing history; the forecasts and views apply "
                         "to the static portfolios only.")
        out["rolling"] = rolling_optimize(px[tickers], tickers, int(rolling_months), int(lookback_months), bounds, groups,
                                          min_weight, max_weight, target_return, target_vol, ports,
                                          [m for m in meths if m not in BENCH_METHODS], omega_threshold)
    return out


def rolling_optimize(px: pd.DataFrame, tickers: list[str], every: int, lookback: int, bounds, groups,
                     min_weight, max_weight, target_return, target_vol, static: dict, methods=None,
                     omega_threshold: float = 0.0) -> dict:
    """Walk-forward re-optimisation on monthly returns: every `every` months, fit on the trailing
    `lookback` months only and hold those weights (rebalanced monthly) until the next re-fit."""
    if every < 1 or lookback < 12:
        raise ValueError("Re-optimise at least every month, on at least 12 months of history.")
    mr = metrics.monthly_returns_frame(px)
    if len(mr) < lookback + 6:
        raise ValueError(f"Walk-forward with a {lookback}-month lookback needs more than {lookback + 6} months of history; there are {len(mr)}.")
    rfm = _rf_monthly(mr.index)
    methods = _methods(target_return, target_vol, methods)
    R = mr.to_numpy()
    oos = {m: np.full(len(mr), np.nan) for m in methods}
    hist = {m: [] for m in methods}
    for k in range(lookback, len(mr), every):
        try:
            o = _Opt(R[k - lookback:k], tickers, float(rfm.iloc[k - lookback:k].mean() * 12), bounds, groups,
                     min_weight, max_weight, omega_threshold)
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
