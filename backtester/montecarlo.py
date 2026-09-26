"""Monte Carlo simulation of a portfolio's future (Portfolio Visualizer style).

Everything runs on monthly steps. For each simulated path the tool draws monthly asset returns
(and monthly inflation), grows the holdings, rebalances on the chosen schedule and applies the
cash-flow schedule. Return models:

  historical  block bootstrap of historical monthly returns: whole months are drawn together for
              every asset (and CPI), in blocks of consecutive months, so cross-asset correlation,
              the link with inflation and short-term autocorrelation are all preserved
  normal      multivariate normal with the historical mean vector and covariance matrix
  t           multivariate Student-t with the same mean and covariance and fitted degrees of
              freedom (maximum likelihood), i.e. fat tails
  forecast    user-supplied expected return and volatility per asset (annual), historical
              correlations, multivariate normal

Cash flows are applied at the start of each period, pro rata to the current holdings (so they do
not change the mix); that makes the portfolio's monthly return independent of the flows, and the
balance follows B[m+1] = (B[m] + F[m]) * (1 + r[m]). A path fails when the balance reaches zero.
The first period's flow is made at the very start, like Portfolio Visualizer and like the backtest
(portfolio.py): "add $1,000 a month for 20 years" is 240 contributions, the first one on day one.

History: monthly returns run month end to month end; a month still in progress at the end of the data is
left out, and the history is labelled with the last trading day actually used.

Stress tests (Settings.stress):
  worst_sequence  every path starts with the worst historical run of `stress_years` years (the lowest
                  compounded return of the portfolio over any window of that length), then continues with
                  the chosen model: sequence-of-returns risk for someone retiring into a bad decade
  shock           every path's first year returns `stress_shock` (default -30%), spread evenly over its
                  twelve months, then continues with the model
Horizon: `years`, or `until_age - age` when both ages are given ("withdraw until age 95"), or
horizon="mortality": the paths run until the survival probability from the current age falls below 0.1%
(SSA period life table, see lifetable.py; sex "male", "female" or "joint" for a couple, where the money must
last until the second death) and the chance of success is weighted by survival: the sum over the years k of
P(death in year k) x P(money left at the end of year k), plus P(alive at the end) x P(money left then).

Withdrawals never take more than the balance: a path that cannot pay a withdrawal in full pays what is
left, ends at zero (no negative balances, no borrowing) and has failed from then on.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import data

MODELS = ("historical", "normal", "t", "forecast")
PERCENTILES = (10, 25, 50, 75, 90)
STEPS = {"monthly": 1, "quarterly": 3, "yearly": 12, "annual": 12, "none": 0}


@dataclass
class CashFlow:
    """One recurring cash flow. amount > 0 adds money, < 0 withdraws; pct withdraws (if < 0) or adds
    a fraction of the balance per year, split evenly over the periods."""
    amount: float = 0.0
    pct: float = 0.0
    freq: str = "yearly"
    inflation_adjusted: bool = True
    start_year: int = 1            # first year it applies (1 = from the start)
    end_year: int | None = None    # last year it applies (inclusive); None = to the end

    def describe(self) -> str:
        if not self.amount and not self.pct:
            return "no cash flows"
        what = []
        if self.amount:
            what.append(f"{'add' if self.amount > 0 else 'withdraw'} ${abs(self.amount):,.0f} {self.freq}"
                        + (" (grows with inflation)" if self.inflation_adjusted else " (fixed dollars)"))
        if self.pct:
            what.append(f"{'add' if self.pct > 0 else 'withdraw'} {abs(self.pct):.2%} of the balance per year, paid {self.freq}")
        yrs = ""
        if self.start_year > 1 or self.end_year:
            yrs = f", years {self.start_year}-{self.end_year or 'end'}"
        return "; ".join(what) + yrs


@dataclass
class Settings:
    weights: dict[str, float] = field(default_factory=dict)
    start_balance: float = 1_000_000.0
    years: int = 30
    flows: list[CashFlow] = field(default_factory=list)
    model: str = "historical"
    block_months: int = 12                        # historical bootstrap block length
    inflation: str | float = "historical"         # "historical" or a fixed annual rate
    rebalance: str = "yearly"
    sims: int = 5000
    seed: int | None = 7
    start: str | None = None                      # history window used to fit/bootstrap
    end: str | None = None
    forecast: dict[str, tuple[float, float]] = field(default_factory=dict)   # ticker -> (mean, vol), annual
    success_target: float = 0.95                  # for the safe withdrawal rate
    series: pd.Series | None = None               # monthly returns of a whole strategy (instead of weights)
    series_name: str = "Portfolio"
    stress: str | None = None                     # None, "worst_sequence" or "shock"
    stress_years: int = 10                        # length of the worst historical sequence placed first
    expense_ratio: float = 0.0                    # annual fee on the portfolio, taken monthly
    stress_shock: float = -0.30                   # first-year return of the "shock" stress test
    age: float | None = None                      # current age: with until_age, the horizon is until_age - age
    until_age: float | None = None
    horizon: str = "fixed"                        # "fixed" (years / until_age) or "mortality" (SSA life table)
    sex: str = "male"                             # mortality: "male", "female" or "joint" (a couple)
    age2: float | None = None                     # joint: the second person's age (default: the same age)


# ------------------------------------------------------------------ history

def monthly_asset_returns(tickers: list[str], start=None, end=None) -> pd.DataFrame:
    """Monthly total returns (from adj_close month-ends) over the common history of the tickers."""
    px = pd.concat({data.canonical(t): data.load(t)["adj_close"] for t in tickers}, axis=1).dropna()
    if start:
        px = px[px.index >= pd.Timestamp(start)]
    if end:
        px = px[px.index <= pd.Timestamp(end)]
    me = complete_months(px)
    # the first month-end is the base of the first return (a partial first month has no return)
    r = me.pct_change().iloc[1:]
    return r.dropna()


def complete_months(px):
    from .metrics import complete_months as cm
    return cm(px)


def monthly_inflation(index: pd.DatetimeIndex | None = None) -> pd.Series:
    """Monthly CPI change; aligned to `index` (month ends) if given."""
    c = data.cpi()
    if c.empty:
        return pd.Series(dtype=float)
    c = c.copy()
    c.index = c.index.to_period("M").to_timestamp("M")
    infl = c.pct_change().dropna()
    if index is not None:
        idx = pd.DatetimeIndex(index).to_period("M").to_timestamp("M")
        infl = infl.reindex(idx)
        infl.index = index
    return infl


def fit_t_df(X: np.ndarray) -> float:
    """Degrees of freedom of a multivariate Student-t fitted by maximum likelihood (mean and
    covariance held at their sample values; the scale matrix is cov * (df - 2) / df)."""
    from scipy.optimize import minimize_scalar
    from scipy.special import gammaln
    X = np.atleast_2d(X)
    n, k = X.shape
    mu = X.mean(axis=0)
    cov = np.cov(X, rowvar=False).reshape(k, k) + np.eye(k) * 1e-12
    inv = np.linalg.inv(cov)
    d = X - mu
    maha = np.einsum("ij,jk,ik->i", d, inv, d)
    _, logdet = np.linalg.slogdet(cov)

    def nll(nu):
        s = (nu - 2) / nu                    # scale = cov * s
        q = maha / s
        ll = (gammaln((nu + k) / 2) - gammaln(nu / 2) - k / 2 * np.log(nu * np.pi)
              - 0.5 * (logdet + k * np.log(s)) - (nu + k) / 2 * np.log1p(q / nu))
        return -ll.sum()
    r = minimize_scalar(nll, bounds=(2.05, 200.0), method="bounded")
    return float(r.x)


# ------------------------------------------------------------------ simulation core

def _draw_blocks(rng, n_hist: int, months: int, sims: int, block: int) -> np.ndarray:
    """(sims, months) indices into the history: blocks of consecutive months (circular)."""
    block = max(1, min(block, n_hist))
    nb = int(np.ceil(months / block))
    starts = rng.integers(0, n_hist, size=(sims, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n_hist
    return idx.reshape(sims, nb * block)[:, :months]


def _portfolio_returns(A: np.ndarray, w: np.ndarray, rebalance_every: int) -> np.ndarray:
    """Monthly portfolio returns (sims, months) for asset returns A (sims, months, n) with target
    weights w, rebalanced every `rebalance_every` months (0 = never)."""
    sims, months, n = A.shape
    hold = np.broadcast_to(w, (sims, n)).copy()
    out = np.empty((sims, months))
    for m in range(months):
        before = hold.sum(axis=1)
        hold = hold * (1 + A[:, m, :])
        after = hold.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[:, m] = np.where(before > 0, after / before - 1, 0.0)
        if rebalance_every and (m + 1) % rebalance_every == 0:
            hold = np.maximum(after, 0)[:, None] * w[None, :]
    return np.maximum(out, -1.0)


def simulate(P: np.ndarray, cum_infl: np.ndarray, start: float, flows: list[CashFlow]) -> dict:
    """Balances (sims, months + 1) given portfolio returns P (sims, months) and the cumulative inflation
    index at the start of each month cum_infl (sims, months + 1), plus what was actually withdrawn.
    A withdrawal is capped at the balance available (contributions of the same period count first), so a
    balance never goes below zero; the unpaid part is the shortfall."""
    sims, months = P.shape
    B = np.empty((sims, months + 1))
    B[:, 0] = start
    b = np.full(sims, float(start))
    paid = np.zeros(sims)
    paid_real = np.zeros(sims)
    asked = np.zeros(sims)
    for m in range(months):
        add = np.zeros(sims)
        take = np.zeros(sims)
        year = m // 12 + 1
        for cf in flows:
            step = STEPS.get(cf.freq, 12) or 12
            if m % step or year < cf.start_year or (cf.end_year and year > cf.end_year):
                continue
            v = np.zeros(sims)
            if cf.amount:
                v = v + cf.amount * (cum_infl[:, m] if cf.inflation_adjusted else 1.0)
            if cf.pct:
                v = v + cf.pct * step / 12 * np.maximum(b, 0)
            add += np.maximum(v, 0.0)
            take += np.maximum(-v, 0.0)
        avail = np.maximum(b, 0.0) + add
        got = np.minimum(take, avail)            # never withdraw more than there is
        paid += got
        paid_real += got / cum_infl[:, m]
        asked += take
        b = (avail - got) * (1 + P[:, m])
        b = np.where(b > 1e-9, b, 0.0)
        B[:, m + 1] = b
    return {"B": B, "withdrawn": paid, "withdrawn_real": paid_real, "requested": asked}


def simulate_balances(P: np.ndarray, cum_infl: np.ndarray, start: float, flows: list[CashFlow]) -> np.ndarray:
    """Balances (sims, months + 1); see simulate()."""
    return simulate(P, cum_infl, start, flows)["B"]


def mortality_curve(s: "Settings", years: int = 120) -> tuple[np.ndarray, str]:
    """Survival probabilities S[k] (k = 0..years) for the settings' age(s) and sex, and a description."""
    from . import lifetable
    if s.age is None:
        raise ValueError("The mortality horizon needs the current age.")
    sex = str(s.sex or "male").strip().lower()
    if sex == "joint":
        a2 = s.age if s.age2 is None else s.age2
        return (lifetable.joint_survival(s.age, "male", a2, "female", years),
                f"a couple (man aged {s.age:g}, woman aged {a2:g}), until the second death")
    return lifetable.survival(s.age, sex, years), f"a {sex} aged {s.age:g}"


def survival_weighted_success(alive: np.ndarray, S: np.ndarray) -> float:
    """sum_k P(death in year k) * alive[k] + P(alive after the last year) * alive[-1]; alive[k] is the share of
    paths with money left at the end of year k (k = 0..years), S the survival curve over the same years."""
    S = np.asarray(S, float)[: len(alive)]
    d = S[:-1] - S[1:]
    return float((d * alive[1:]).sum() + S[-1] * alive[-1])


def _alive_to_end(P: np.ndarray, cum_infl: np.ndarray, start: float, rate: float) -> tuple[np.ndarray, np.ndarray]:
    """Constant inflation-adjusted annual withdrawal of rate * start at the start of each year.
    Returns (survived the whole horizon, final real balance)."""
    sims, months = P.shape
    b = np.full(sims, float(start))
    alive = np.ones(sims, bool)
    for m in range(months):
        if m % 12 == 0:
            b = b - rate * start * cum_infl[:, m]
            alive &= b > 0
            b = np.maximum(b, 0.0)
        b = b * (1 + P[:, m])
    return alive, b / cum_infl[:, months]


def safe_withdrawal_rate(P, cum_infl, start, success=0.95) -> float:
    """Highest constant inflation-adjusted withdrawal (share of the starting balance, per year,
    taken at the start of each year) that leaves money at the end in at least `success` of paths."""
    lo, hi = 0.0, 1.0
    if _alive_to_end(P, cum_infl, start, lo)[0].mean() < success:
        return 0.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if _alive_to_end(P, cum_infl, start, mid)[0].mean() >= success:
            lo = mid
        else:
            hi = mid
    return lo


def perpetual_withdrawal_rate(P, cum_infl, start) -> float:
    """Highest constant inflation-adjusted withdrawal that keeps the median final real balance at or
    above the starting balance."""
    lo, hi = -1.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        _, real = _alive_to_end(P, cum_infl, start, mid)
        if np.median(real) >= start:
            lo = mid
        else:
            hi = mid
    return max(lo, 0.0)


def _max_drawdown(P: np.ndarray) -> np.ndarray:
    g = np.cumprod(1 + P, axis=1)
    g = np.concatenate([np.ones((len(P), 1)), g], axis=1)
    return (g / np.maximum.accumulate(g, axis=1) - 1).min(axis=1)


# ------------------------------------------------------------------ driver

def _history(s: Settings) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    if s.series is not None:
        hist = s.series.dropna().to_frame(s.series_name)
        return hist, np.ones(1), [s.series_name]
    if not s.weights:
        raise ValueError("Give tickers with weights (e.g. SPY 60, TLT 40) or a strategy.")
    tick = [data.canonical(t) for t in s.weights]
    w = np.array([float(v) for v in s.weights.values()])
    if (w < 0).any():
        raise ValueError("Weights must be positive.")
    if w.sum() <= 0:
        raise ValueError("Weights must add up to more than zero.")
    w = w / w.sum()
    hist = monthly_asset_returns(tick, s.start, s.end)
    return hist, w, tick


def run(s: Settings) -> dict:
    if s.model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}")
    notes = []
    S_full = None
    if s.horizon not in ("fixed", "mortality", None, ""):
        raise ValueError("horizon must be 'fixed' or 'mortality'")
    if s.horizon == "mortality":
        from . import lifetable
        S_full, who = mortality_curve(s)
        s.years = max(1, lifetable.horizon_years(S_full))
        notes.append(f"Horizon from the {lifetable.SOURCE}: {who}. The paths run {s.years} years (until the chance of "
                     f"being alive falls below 0.1%) and the chance of success is weighted by the chance of being alive.")
    elif s.age is not None and s.until_age is not None:
        yrs = int(round(float(s.until_age) - float(s.age)))
        if yrs < 1:
            raise ValueError(f"The final age ({s.until_age:g}) must be above the current age ({s.age:g}).")
        s.years = yrs
        notes.append(f"Horizon: {yrs} years, from age {s.age:g} until age {s.until_age:g}.")
    if not (1 <= s.years <= 100):
        raise ValueError("Horizon must be between 1 and 100 years.")
    if s.stress not in (None, "", "none", "worst_sequence", "shock"):
        raise ValueError("stress must be 'worst_sequence' or 'shock'")
    if not (100 <= s.sims <= 50_000):
        raise ValueError("Number of simulations must be between 100 and 50,000.")
    hist, w, tick = _history(s)
    if len(hist) < 36:
        raise ValueError(f"Need at least 36 months of common history; {', '.join(tick)} have {len(hist)}.")
    rng = np.random.default_rng(s.seed)
    months = int(s.years * 12)
    sims = int(s.sims)
    H = hist.to_numpy()
    n = H.shape[1]
    if sims * months * n > 60_000_000:
        raise ValueError(f"{sims:,} simulations x {months} months x {n} assets is too large; use fewer simulations.")
    hist_infl = monthly_inflation(hist.index)
    fixed_infl = None if s.inflation == "historical" else float(s.inflation)
    if fixed_infl is None and hist_infl.isna().all():
        fixed_infl = 0.025
        notes.append("CPI data unavailable: inflation assumed at 2.5% a year.")
    t_df = None
    if s.model == "historical":
        idx = _draw_blocks(rng, len(H), months, sims, s.block_months)
        A = H[idx]                                  # (sims, months, n): same months for every asset
        if fixed_infl is None:
            hi = hist_infl.to_numpy()
            if np.isnan(hi).any():
                # months before CPI exists: fill with the average monthly inflation
                hi = np.where(np.isnan(hi), np.nanmean(hi), hi)
                notes.append("Months of history without CPI data (usually the latest month or two) use the average historical inflation.")
            I = hi[idx]
    else:
        mu = H.mean(axis=0)
        cov = np.cov(H, rowvar=False).reshape(n, n)
        if s.model == "forecast":
            sd = np.sqrt(np.diag(cov))
            corr = cov / np.outer(sd, sd)
            fm, fs = mu.copy(), sd.copy()
            for j, t in enumerate(tick):
                if t in s.forecast:
                    ann_mu, ann_vol = s.forecast[t]
                    fm[j] = (1 + ann_mu) ** (1 / 12) - 1 if ann_mu > -1 else -1
                    fs[j] = ann_vol / np.sqrt(12)
                else:
                    notes.append(f"No forecast for {t}: its historical mean and volatility were used.")
            mu, cov = fm, corr * np.outer(fs, fs)
        L = np.linalg.cholesky(cov + np.eye(n) * 1e-14)
        Z = rng.standard_normal((sims, months, n)) @ L.T
        if s.model == "t":
            t_df = fit_t_df(H)
            chi = rng.chisquare(t_df, size=(sims, months, 1))
            Z = Z * np.sqrt((t_df - 2) / chi)   # covariance stays equal to cov
        A = np.maximum(mu + Z, -1.0)
        if fixed_infl is None:
            # inflation from the same window as the asset returns, so real results are consistent
            # across models (the full 1947+ CPI history would mix eras)
            hi = hist_infl.dropna()
            full = hi.to_numpy() if len(hi) >= 24 else monthly_inflation().dropna().to_numpy()
            I = full[_draw_blocks(rng, len(full), months, sims, max(s.block_months, 1))]
    if fixed_infl is not None:
        I = np.full((sims, months), (1 + fixed_infl) ** (1 / 12) - 1)
    stress = None
    if s.stress == "worst_sequence":
        k = int(min(max(1, s.stress_years) * 12, months, len(H)))
        hp = H @ w if n > 1 else H[:, 0]
        logg = np.log1p(np.maximum(hp, -0.999999))
        if fixed_infl is None and any(getattr(f, "inflation_adjusted", False) and (f.amount or f.pct) and (f.amount < 0 or f.pct < 0) for f in (s.flows or [])):
            # with inflation-indexed cash flows the damaging sequence is the worst in real terms (the 1970s)
            hi0 = hist_infl.reindex(hist.index).to_numpy() if hasattr(hist_infl, "reindex") else np.asarray(hist_infl)
            hi0 = np.where(np.isfinite(hi0), hi0, 0.0)
            logg = logg - np.log1p(hi0[: len(logg)])
        c = np.concatenate([[0.0], np.cumsum(logg)])
        tot = c[k:] - c[:-k]
        j = int(np.argmin(tot))
        A[:, :k, :] = H[j:j + k][None, :, :]
        if fixed_infl is None:
            hi_ = hist_infl.to_numpy()
            hi_ = np.where(np.isnan(hi_), np.nanmean(hi_), hi_) if np.isfinite(hi_).any() else np.zeros(len(hi_))
            I[:, :k] = hi_[j:j + k][None, :]
        stress = {"kind": "worst_sequence", "months": k, "from": hist.index[j].date(), "to": hist.index[j + k - 1].date(),
                  "return": float(np.expm1(tot[j]))}
        notes.append(f"Stress test: every path starts with the worst {k // 12 if k % 12 == 0 else round(k / 12, 1)}-year "
                     f"stretch of this history ({hist.index[j].strftime('%Y-%m')} to {hist.index[j + k - 1].strftime('%Y-%m')}, "
                     f"{np.expm1(tot[j]):.1%} in total), then continues with the {s.model} model.")
    cum_infl = np.concatenate([np.ones((sims, 1)), np.cumprod(1 + I, axis=1)], axis=1)

    rb = STEPS.get(s.rebalance, 12)
    P = _portfolio_returns(A, w, rb) if n > 1 else A[:, :, 0]
    if s.stress == "shock":
        k = min(12, months)
        shock = float(s.stress_shock)
        if not -1 < shock < 1:
            raise ValueError("The shock is a first-year return between -100% and +100% (e.g. -0.30).")
        P = P.copy()
        P[:, :k] = (1 + shock) ** (1 / k) - 1
        stress = {"kind": "shock", "months": k, "return": shock}
        notes.append(f"Stress test: every path loses {-shock:.0%} in its first year" if shock < 0 else
                     f"Stress test: every path returns {shock:.0%} in its first year")
        notes[-1] += f" (evenly over {k} months), then continues with the {s.model} model."
    if s.expense_ratio:
        P = P - s.expense_ratio / 12.0
    SIM = simulate(P, cum_infl, s.start_balance, s.flows)
    B = SIM["B"]
    R = B / cum_infl

    yrs = np.arange(0, s.years + 1)
    cols = yrs * 12
    pct = lambda X, p: np.percentile(X, p, axis=0)  # noqa: E731
    bands = {str(p): pct(B[:, cols], p).tolist() for p in PERCENTILES}
    bands_real = {str(p): pct(R[:, cols], p).tolist() for p in PERCENTILES}
    alive = (B[:, cols] > 0).mean(axis=0)
    twr = np.prod(1 + P, axis=1)
    ann = np.where(twr > 0, twr ** (12 / months) - 1, -1.0)
    real_twr = twr / cum_infl[:, -1]
    ann_real = np.where(real_twr > 0, real_twr ** (12 / months) - 1, -1.0)
    mdd = _max_drawdown(P)
    final = B[:, -1]
    q = lambda a: {str(p): float(np.percentile(a, p)) for p in PERCENTILES}  # noqa: E731
    has_wd = any(cf.amount < 0 or cf.pct < 0 for cf in s.flows)
    has_contrib = any(cf.amount > 0 or cf.pct > 0 for cf in s.flows)
    out = {
        "settings": {"mode": "strategy" if s.series is not None else "assets", "name": s.series_name if s.series is not None else None,
                     "weights": {t: float(x) for t, x in zip(tick, w)}, "start_balance": s.start_balance,
                     "years": s.years, "model": s.model, "block_months": s.block_months,
                     "inflation": "historical CPI" if fixed_infl is None else fixed_infl,
                     "rebalance": s.rebalance, "sims": sims,
                     "flows": [cf.describe() for cf in s.flows] or ["no cash flows"],
                     "history_start": hist.index[0].date(), "history_end": hist.index[-1].date(),
                     "history_months": len(hist), "t_df": t_df, "success_target": s.success_target,
                     "age": s.age, "until_age": s.until_age, "stress": s.stress or None,
                     "horizon": s.horizon or "fixed", "sex": s.sex if s.horizon == "mortality" else None,
                     "age2": (s.age if s.age2 is None else s.age2) if s.horizon == "mortality" and str(s.sex).lower() == "joint" else None},
        "years": yrs.tolist(),
        "bands": bands, "bands_real": bands_real,
        "success_by_year": alive.tolist(),
        "prob_success": float((final > 0).mean()),
        "final": q(final), "final_real": q(R[:, -1]),
        "annual_return": q(ann), "annual_return_real": q(ann_real),
        "max_drawdown": q(mdd),
        "mean_final": float(final.mean()),
        "safe_withdrawal_rate": safe_withdrawal_rate(P, cum_infl, s.start_balance, s.success_target),
        "perpetual_withdrawal_rate": perpetual_withdrawal_rate(P, cum_infl, s.start_balance),
        "has_withdrawals": has_wd, "has_contributions": has_contrib,
        # safe / perpetual withdrawal rates answer a retirement question; with contributions only they are noise
        "show_withdrawal_rates": has_wd or not has_contrib,
        "stress": stress,
        "notes": notes,
        "hist_stats": {"mean_annual": (hist.mean() * 12).to_dict(), "vol_annual": (hist.std() * np.sqrt(12)).to_dict(),
                       "correlation": hist.corr().round(3).to_numpy().tolist(), "tickers": tick},
    }
    if has_wd:
        short = SIM["requested"] - SIM["withdrawn"]
        out["withdrawals"] = {"total": q(SIM["withdrawn"]), "total_real": q(SIM["withdrawn_real"]),
                              "requested_mean": float(SIM["requested"].mean()),
                              "share_of_paths_short": float((short > 1e-6 * max(1.0, s.start_balance)).mean())}
    if S_full is not None:
        from . import lifetable
        S = S_full[: s.years + 1]
        w = survival_weighted_success(alive, S)
        out["prob_success_to_horizon"] = out["prob_success"]
        out["prob_success"] = w
        age0 = float(s.age)
        out["mortality"] = {"source": lifetable.SOURCE, "table_year": lifetable.TABLE_YEAR, "sex": s.sex, "age": s.age,
                            "age2": out["settings"]["age2"], "survival_by_year": S.tolist(),
                            "life_expectancy": lifetable.life_expectancy(S_full),
                            "median_years_left": int(np.argmax(S_full < 0.5)) if (S_full < 0.5).any() else None,
                            "prob_success_weighted": w, "prob_outlive_money": 1 - w,
                            "ages": [age0 + k for k in yrs.tolist()]}
    if has_wd:
        depleted = B[:, 1:] <= 0
        first = np.where(depleted.any(axis=1), depleted.argmax(axis=1) + 1, -1)
        fails = first[first > 0] / 12
        out["depletion_years"] = q(fails) if len(fails) else {}
    return out


# ------------------------------------------------------------------ helpers for callers

def weights_from_tree(tree: dict) -> dict[str, float] | None:
    """Static weights of a portfolio tree made only of fixed-weight groups of assets, else None."""
    out: dict[str, float] = {}

    def walk(n, scale):
        if "asset" in n:
            t = data.canonical(n["asset"])
            out[t] = out.get(t, 0.0) + scale
            return True
        if "weights" in n and n["weights"] in ("specified", "equal"):
            kids = n["children"]
            ws = n["w"] if n["weights"] == "specified" else [1 / len(kids)] * len(kids)
            return all(walk(k, scale * x) for k, x in zip(kids, ws))
        return False
    if not walk(tree, 1.0) or any(v < 0 for v in out.values()):
        return None
    return out


def _years(v, end: bool):
    """A Portfolio flow bound as a year number of the simulation (years of the backtest map one to one;
    calendar dates cannot be placed in a simulated future and are ignored)."""
    if v in (None, ""):
        return None
    try:
        k = int(v)
    except (TypeError, ValueError):
        return None
    return k if 1 <= k < 1900 else None


def flows_from_portfolio(p) -> list[CashFlow]:
    """The cash-flow schedule of a Portfolio spec, as Monte Carlo cash flows (frequency, inflation indexing and
    'for N years' / 'from year N' windows carry over; the first flow is at the start, as in the backtest)."""
    out = []
    cs, ce = _years(p.contribution_start, False) or 1, _years(p.contribution_end, True)
    ws, we = _years(p.withdrawal_start, False) or 1, _years(p.withdrawal_end, True)
    if p.contribution:
        out.append(CashFlow(amount=p.contribution, freq=p.contribution_freq, inflation_adjusted=p.inflation_adjust,
                            start_year=cs, end_year=ce))
    if p.withdrawal:
        out.append(CashFlow(amount=-p.withdrawal, freq=p.withdrawal_freq, inflation_adjusted=p.inflation_adjust,
                            start_year=ws, end_year=we))
    if p.withdrawal_pct:
        per_year = 12 / (STEPS.get(p.withdrawal_freq, 12) or 12)
        out.append(CashFlow(pct=-p.withdrawal_pct * per_year, freq=p.withdrawal_freq, start_year=ws, end_year=we))
    return out


def settings_from_spec(spec, s: Settings) -> tuple[dict[str, float], str]:
    """Point the settings at a strategy/portfolio spec. A fixed-weight portfolio is simulated from its
    assets (so rebalancing matters); anything else (rules, rotations, signals) resamples the
    strategy's own monthly returns. Returns (weights, name) and sets s.series when needed."""
    name = getattr(spec, "name", "") or getattr(spec, "description", "") or "Strategy"
    # the spec's own money: its starting balance (unless the caller set one) and fees
    if s.start_balance == Settings.start_balance and getattr(spec, "capital", None):
        s.start_balance = float(spec.capital)
    w = weights_from_tree(spec.tree) if hasattr(spec, "tree") else None
    if w:
        s.expense_ratio = float(getattr(spec, "expense_ratio", 0.0) or 0.0)
    if w:
        if getattr(spec, "leverage", 1.0) == 1.0:
            rb = getattr(spec, "rebalance", "yearly")
            s.rebalance = {"daily": "monthly", "weekly": "monthly"}.get(rb, rb)
            if s.start is None and spec.start:
                s.start = spec.start
            if s.end is None and spec.end:
                s.end = spec.end
            return w, name
    ser = strategy_monthly_returns(spec)
    if s.start:
        ser = ser[ser.index >= pd.Timestamp(s.start)]
    if s.end:
        ser = ser[ser.index <= pd.Timestamp(s.end)]
    s.series, s.series_name = ser, name[:60]
    return {}, name


def console(R: dict) -> str:
    st = R["settings"]
    money = lambda v: f"${v:,.0f}"  # noqa: E731
    pc = lambda v: f"{v * 100:.2f}%"  # noqa: E731
    L = [f"Monte Carlo: {st['sims']:,} paths x {st['years']} years, model {st['model']}"
         + (f" (Student-t, fitted df {st['t_df']:.1f})" if st.get("t_df") else "")
         + (f" (blocks of {st['block_months']} months)" if st["model"] == "historical" else ""),
         (f"Strategy: {st['name']} (its own monthly returns are resampled); start {money(st['start_balance'])}"
          if st.get("mode") == "strategy" else
          "Portfolio: " + ", ".join(f"{w:.1%} {t}" for t, w in st["weights"].items())
          + f"; rebalanced {st['rebalance']}; start {money(st['start_balance'])}"),
         f"History {st['history_start']} -> {st['history_end']} ({st['history_months']} months); inflation {st['inflation']}",
         "Cash flows: " + "; ".join(st["flows"])]
    for n in R["notes"]:
        L.append("Note: " + n)
    L.append(f"{'percentile':>28s} " + " ".join(f"{p + 'th':>13s}" for p in R["final"]))
    L.append(f"{'Final balance (nominal)':>28s} " + " ".join(f"{money(v):>13s}" for v in R["final"].values()))
    L.append(f"{'Final balance (real)':>28s} " + " ".join(f"{money(v):>13s}" for v in R["final_real"].values()))
    L.append(f"{'Annual return (nominal)':>28s} " + " ".join(f"{pc(v):>13s}" for v in R["annual_return"].values()))
    L.append(f"{'Annual return (real)':>28s} " + " ".join(f"{pc(v):>13s}" for v in R["annual_return_real"].values()))
    L.append(f"{'Max drawdown':>28s} " + " ".join(f"{pc(v):>13s}" for v in R["max_drawdown"].values()))
    if R.get("mortality"):
        M = R["mortality"]
        L.append(f"Chance the money lasts a lifetime (weighted by survival, SSA {M['table_year']} life table): {R['prob_success']:.1%}"
                 f"   (life expectancy {M['life_expectancy']:.1f} years; money left after all {st['years']} years: "
                 f"{R['prob_success_to_horizon']:.1%})")
    else:
        L.append(f"Chance of success (money left after {st['years']} years): {R['prob_success']:.1%}")
    if R.get("withdrawals"):
        W = R["withdrawals"]
        L.append(f"Total withdrawn (median; withdrawals never exceed the balance): ${W['total']['50']:,.0f} "
                 f"(${W['total_real']['50']:,.0f} in today's dollars); paths that could not pay in full: {W['share_of_paths_short']:.1%}")
    if R.get("show_withdrawal_rates", True):
        L.append(f"Safe withdrawal rate ({st['success_target']:.0%} success, inflation-adjusted, from the start balance): "
                 f"{pc(R['safe_withdrawal_rate'])}   perpetual withdrawal rate: {pc(R['perpetual_withdrawal_rate'])}")
    return "\n".join(L)


def strategy_monthly_returns(spec) -> pd.Series:
    """Monthly time-weighted returns of any strategy or portfolio spec (flows removed)."""
    from . import metrics, runner
    res = runner.run(spec)
    nv = metrics.nav(res.equity, res.extras.get("flows"))
    me = complete_months(nv.iloc[1:])          # without the synthetic starting point; complete months only
    base = pd.concat([nv.iloc[:1], me])
    r = (base / base.shift(1) - 1).iloc[1:]
    # a first month holding only a few days is still a return (from the starting capital); keep it
    return r.dropna()


def historical_withdrawal_rates(monthly: pd.Series, start: float = 1.0) -> dict:
    """SWR / PWR on the actual historical sequence of monthly returns (one path): the highest
    inflation-adjusted annual withdrawal (share of the start) that never runs out, and the highest
    that leaves the real balance at or above the start. Also a 95% bootstrap SWR over the same
    horizon."""
    r = monthly.dropna()
    if len(r) < 24:
        return {}
    infl = monthly_inflation(r.index)
    if infl.isna().all():
        infl = pd.Series((1.025) ** (1 / 12) - 1, index=r.index)
    infl = infl.fillna(infl.mean())
    P = r.to_numpy()[None, :]
    ci = np.concatenate([[1.0], np.cumprod(1 + infl.to_numpy())])[None, :]
    out = {"horizon_years": len(r) / 12, "from": r.index[0].date(), "to": r.index[-1].date(),
           "swr": safe_withdrawal_rate(P, ci, start, 1.0), "pwr": perpetual_withdrawal_rate(P, ci, start)}
    rng = np.random.default_rng(11)
    idx = _draw_blocks(rng, len(r), len(r), 2000, 12)
    Pb = r.to_numpy()[idx]
    Ib = infl.to_numpy()[idx]
    cib = np.concatenate([np.ones((len(Pb), 1)), np.cumprod(1 + Ib, axis=1)], axis=1)
    out["swr_mc95"] = safe_withdrawal_rate(Pb, cib, start, 0.95)
    return out


def parse_weights(text: str) -> dict[str, float]:
    """'SPY 60 TLT 40', 'SPY:60,TLT:40', '60% SPY, 40% TLT' or 'SPY TLT' (equal) -> {ticker: weight}."""
    toks = [t for t in text.replace(",", " ").replace(":", " ").replace("=", " ").split() if t]
    out: dict[str, float] = {}
    pending_w, last_t = None, None
    for t in toks:
        v = t.rstrip("%")
        try:
            w = float(v)
        except ValueError:
            tk = data.canonical(t)
            if pending_w is not None:
                out[tk] = out.get(tk, 0.0) + pending_w
                pending_w, last_t = None, None
            else:
                out.setdefault(tk, 0.0)
                last_t = tk
            continue
        if last_t is not None and out[last_t] == 0.0:
            out[last_t] = w
            last_t = None
        else:
            pending_w = w
    if not out:
        raise ValueError("Give tickers and weights, e.g. 'SPY 60 TLT 40'.")
    if all(v == 0 for v in out.values()):
        out = {k: 1.0 for k in out}
    if any(v == 0 for v in out.values()):
        raise ValueError(f"Missing weight for {', '.join(k for k, v in out.items() if v == 0)}.")
    return out

