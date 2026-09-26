"""Factor regressions (Portfolio Visualizer's "factor analysis"): CAPM, Fama-French 3, Carhart 4,
Fama-French 5, FF5 + momentum, developed-markets Fama-French 3 and bond factors (TERM, DEF), on daily or
monthly excess returns, with t-stats, R², annualised alpha and rolling 36-month loadings.

Factor data:
- US equity factors: data/factors/ff3_daily.csv when present (otherwise the first three factors of
  ff5_daily.csv - the market and SMB/HML from the 5-factor file, which French builds slightly
  differently), ff5_daily.csv and mom_daily.csv (Kenneth French's data library).
- Developed markets: data/factors/dev_ff3_daily.csv (French's "Developed" 3 factors, from 1990), model
  "dev_ff3" (alias "intl"). Its RF is the US one-month T-bill.
- Bond factors, built from the price data (total returns, adj_close):
    TERM = long Treasuries (TLTSIM, else IEFSIM) - T-bills (BILSIM): the term premium
    DEF  = investment-grade corporates (LQDSIM when the data job has built it, else LQD from July 2002) -
           Treasuries of similar duration (IEFSIM): the
           credit (default) premium. When the returns being explained start before LQD, DEF is left out
           with a note (the regression then covers the whole period without it).
- Factors that are not available offline (AQR's QMJ, BAB, ...) can be added: put a CSV of daily decimal
  returns named data/factors/extra_<name>.csv (a "date" column plus one column per factor) and every
  column becomes a factor, with a model "ff3+<name>" (FF3 plus those columns); or call
  register_factor() / add_model() from Python.

Monthly factor returns are compounded from the daily ones: the market as (1 + Mkt-RF + RF) compounded minus
compounded RF, spread factors built from two return series (TERM, DEF) as the difference of the two legs'
compounded returns, and the other long-short factors compounded directly.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Callable

import numpy as np
import pandas as pd

from . import data, metrics

# model key -> (label, factor columns). Extend with add_model().
MODELS = {
    "capm": ("CAPM", ["Mkt-RF"]),
    "ff3": ("Fama-French 3", ["Mkt-RF", "SMB", "HML"]),
    "carhart": ("Carhart 4", ["Mkt-RF", "SMB", "HML", "Mom"]),
    "ff5": ("Fama-French 5", ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]),
    "ff6": ("Fama-French 5 + momentum", ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]),
    "dev_ff3": ("Developed markets Fama-French 3", ["Mkt-RF", "SMB", "HML"]),
    "bonds": ("Market + bond factors", ["Mkt-RF", "TERM", "DEF"]),
    "ff3+bonds": ("Fama-French 3 + bond factors", ["Mkt-RF", "SMB", "HML", "TERM", "DEF"]),
}
ALIASES = {"intl": "dev_ff3", "developed": "dev_ff3", "ff3_bonds": "ff3+bonds"}
# where a model's market/size/value factors come from ("us" = French's US files)
REGION = {"dev_ff3": "dev"}
DESCRIBE = {"Mkt-RF": "market excess return", "SMB": "size (small minus big)", "HML": "value (high minus low book/price)",
            "RMW": "profitability (robust minus weak)", "CMA": "investment (conservative minus aggressive)",
            "Mom": "momentum (winners minus losers)",
            "TERM": "term premium (long Treasuries minus T-bills)",
            "DEF": "credit premium (investment-grade corporates minus Treasuries)"}
ABOUT = {
    "capm": "the market alone",
    "ff3": "market, size and value (US)",
    "carhart": "Fama-French 3 plus momentum (US)",
    "ff5": "adds profitability and investment (US, from 1963)",
    "ff6": "Fama-French 5 plus momentum (US)",
    "dev_ff3": "market, size and value of developed markets (from 1990); also 'intl'",
    "bonds": "the US market plus the term and credit premiums (TERM from 1962, DEF from 2002)",
    "ff3+bonds": "Fama-French 3 plus TERM and DEF, for stock/bond portfolios",
}

# extra factors: name -> (about, loader returning a daily DataFrame with that column, and optional legs
# "<name>|a" / "<name>|b" when the factor is the difference of two return series)
_EXTRA: dict[str, tuple[str, Callable[[], pd.DataFrame]]] = {}


def register_factor(name: str, about: str, loader: Callable[[], pd.DataFrame]) -> None:
    """Make a factor available to models (e.g. AQR's QMJ or BAB from a file you downloaded)."""
    _EXTRA[name] = (about, loader)
    DESCRIBE[name] = about
    factor_table.cache_clear()


def add_model(key: str, label: str, factors: list[str], about: str = "", region: str = "us") -> None:
    MODELS[key] = (label, list(factors))
    ABOUT[key] = about or label
    if region != "us":
        REGION[key] = region
    factor_table.cache_clear()


def resolve(model: str) -> str:
    m = str(model or "ff3").strip()
    m = ALIASES.get(m.lower(), m)
    if m not in MODELS:
        raise ValueError(f"model must be one of {', '.join(list(MODELS) + list(ALIASES))}")
    return m


def model_list() -> list[dict]:
    """The models for the CLI help and the Factors page."""
    return [{"key": k, "label": v[0], "factors": v[1], "about": ABOUT.get(k, ""),
             "aliases": [a for a, t in ALIASES.items() if t == k]} for k, v in MODELS.items()]


def _read(name: str) -> pd.DataFrame:
    p = data.DATA / "factors" / name
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, parse_dates=["date"], index_col="date").sort_index()
    df.columns = [c.strip() for c in df.columns]
    return df.apply(pd.to_numeric, errors="coerce")


def _tr(ticker: str) -> pd.Series | None:
    """Daily total returns of a series that is in the local data (never downloaded on demand)."""
    if not (data.PRICES / f"{ticker}.csv").exists():
        return None
    try:
        return data.load(ticker)["adj_close"].pct_change()
    except (FileNotFoundError, data.DataError, KeyError):
        return None


@lru_cache(maxsize=2)
def bond_factors() -> pd.DataFrame:
    """Daily TERM and DEF (decimal) with their legs; DEF is NaN before the corporate series starts."""
    long_ = _tr("TLTSIM")
    long_name = "TLTSIM"
    if long_ is None:
        long_, long_name = _tr("IEFSIM"), "IEFSIM"
    bills = _tr("BILSIM")
    parts = {}
    if long_ is not None and bills is not None:
        df = pd.concat({"a": long_, "b": bills}, axis=1, sort=True).dropna()
        parts["TERM"], parts["TERM|a"], parts["TERM|b"] = df["a"] - df["b"], df["a"], df["b"]
    corp, corp_name = _tr("LQDSIM"), "LQDSIM"
    if corp is None:
        corp, corp_name = _tr("LQD"), "LQD"
    tsy = _tr("IEFSIM")
    if corp is not None and tsy is not None:
        df = pd.concat({"a": corp, "b": tsy}, axis=1, sort=True).dropna()
        parts["DEF"], parts["DEF|a"], parts["DEF|b"] = df["a"] - df["b"], df["a"], df["b"]
    if not parts:
        return pd.DataFrame()
    out = pd.DataFrame(parts).sort_index()
    out.attrs["term_leg"] = long_name
    out.attrs["def_leg"] = corp_name
    return out


def _load_extra_files() -> None:
    base = data.DATA / "factors"
    if not base.is_dir():
        return
    for p in sorted(base.glob("extra_*.csv")):
        stem = p.stem[len("extra_"):]
        df = _read(p.name)
        cols = [c for c in df.columns if c.upper() != "RF"]
        for c in cols:
            if c not in _EXTRA:
                register_factor(c, f"{c} (from {p.name})", lambda df=df, c=c: df[[c]])
        if cols and f"ff3+{stem}" not in MODELS:
            add_model(f"ff3+{stem}", f"Fama-French 3 + {', '.join(cols)}", ["Mkt-RF", "SMB", "HML"] + cols,
                      f"FF3 plus the factors in {p.name}")



@lru_cache(maxsize=8)
def factor_table(model: str) -> pd.DataFrame:
    """Daily factor returns (decimal) for a model, plus RF (and the legs of spread factors). A factor that
    starts later than the others (DEF) is NaN before it starts; analyze() decides whether to use it."""
    model = resolve(model)
    cols = MODELS[model][1]
    region = REGION.get(model, "us")
    base_cols = [c for c in cols if c in ("Mkt-RF", "SMB", "HML", "RMW", "CMA")]
    if region == "dev":
        base = _read("dev_ff3_daily.csv")
        if base.empty:
            raise ValueError("Developed-markets factor data is missing (data/factors/dev_ff3_daily.csv); the data job downloads it.")
    elif any(c in ("RMW", "CMA") for c in cols):
        base = _read("ff5_daily.csv")
    else:
        ff3 = _read("ff3_daily.csv")
        base = ff3 if not ff3.empty else _read("ff5_daily.csv")
    if base.empty:
        raise ValueError("Factor data is missing (data/factors/ff5_daily.csv); the data job downloads it.")
    missing = [c for c in base_cols if c not in base]
    if missing:
        raise ValueError(f"The factor file has no {', '.join(missing)} column.")
    df = base[base_cols + ["RF"]].dropna()
    if "Mom" in cols:
        mom = _read("mom_daily.csv")
        if mom.empty:
            raise ValueError("Momentum factor data is missing (data/factors/mom_daily.csv).")
        df = df.join(mom[["Mom"]], how="inner").dropna()
    if "TERM" in cols or "DEF" in cols:
        bf = bond_factors()
        need = [c for c in ("TERM", "DEF") if c in cols]
        if bf.empty or "TERM" in need and "TERM" not in bf:
            raise ValueError("Bond factors need TLTSIM (or IEFSIM) and BILSIM price data.")
        keep = [c for c in bf.columns if c.split("|")[0] in need]
        df = df.join(bf[keep], how="inner")
        df = df[df["TERM"].notna()] if "TERM" in df else df
    for c in cols:
        if c in _EXTRA:
            ex = _EXTRA[c][1]()
            df = df.join(ex[[x for x in ex.columns if x.split("|")[0] == c]], how="inner").dropna(subset=[c])
    return df


def monthly_factors(daily: pd.DataFrame) -> pd.DataFrame:
    legs = [c for c in daily.columns if "|" in c]
    g = (1 + daily).resample("ME").prod(min_count=1) - 1
    out = g.drop(columns=legs)
    if "Mkt-RF" in daily:
        mkt = (1 + daily["Mkt-RF"] + daily["RF"]).resample("ME").prod() - 1
        out["Mkt-RF"] = mkt - g["RF"]
    for c in {x.split("|")[0] for x in legs}:
        if f"{c}|a" in g and f"{c}|b" in g:
            out[c] = g[f"{c}|a"] - g[f"{c}|b"]
    counts = daily["RF"].resample("ME").count()
    return out[counts > 0]


def ols(y: np.ndarray, X: np.ndarray) -> dict:
    """OLS with an intercept column already in X. Coefficients, standard errors, t-stats, p-values."""
    from scipy import stats
    n, k = X.shape
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    dof = n - k
    s2 = resid @ resid / dof if dof > 0 else np.nan
    cov = s2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    t = coef / se
    p = 2 * stats.t.sf(np.abs(t), dof) if dof > 0 else np.full(k, np.nan)
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - resid @ resid / ss_tot if ss_tot > 0 else np.nan
    adj = 1 - (1 - r2) * (n - 1) / dof if dof > 0 else np.nan
    return {"coef": coef, "se": se, "t": t, "p": p, "r2": r2, "adj_r2": adj, "n": n,
            "resid_sd": float(np.sqrt(s2)) if np.isfinite(s2) else np.nan}


def returns_for(target) -> tuple[pd.Series, str]:
    """Daily total returns for a ticker, a {ticker: weight} dict (rebalanced monthly), or a spec."""
    from . import parser, portfolio, runner
    if isinstance(target, str) and target.strip() and " " not in target.strip():
        t = data.canonical(target)
        return data.load(t)["adj_close"].pct_change().dropna(), t
    if isinstance(target, dict):
        w = {data.canonical(k): float(v) for k, v in target.items()}
        s = sum(w.values())
        if s <= 0:
            raise ValueError("Weights must add up to more than zero.")
        p = portfolio.Portfolio(tree={"weights": "specified", "w": [v / s for v in w.values()],
                                      "children": [{"asset": k} for k in w]}, rebalance="monthly")
        res = portfolio.run(p)
        return metrics.nav(res.equity, res.extras.get("flows")).pct_change().dropna().iloc[1:], \
            " / ".join(f"{v / s:.0%} {k}" for k, v in w.items())
    spec = parser.parse(target) if isinstance(target, str) else target
    res = runner.run(spec)
    nv = metrics.nav(res.equity, res.extras.get("flows"))
    return nv.pct_change().dropna().iloc[1:], (spec.name or spec.description or "strategy")


def analyze(returns: pd.Series, model: str = "ff3", freq: str = "monthly", start=None, end=None,
            rolling_months: int = 36, name: str = "") -> dict:
    if freq not in ("daily", "monthly"):
        raise ValueError("freq must be daily or monthly")
    model = resolve(model)
    label, cols = MODELS[model]
    cols = list(cols)
    notes: list[str] = []
    f = factor_table(model)
    r = returns.dropna()
    if start:
        r = r[r.index >= pd.Timestamp(start)]
    if end:
        r = r[r.index <= pd.Timestamp(end)]
    if "DEF" in cols and len(r):
        d0 = f["DEF"].first_valid_index() if "DEF" in f else None
        if d0 is None or r.index[0] < d0 - pd.Timedelta(days=31):
            cols.remove("DEF")
            f = f.drop(columns=[c for c in f.columns if c.split("|")[0] == "DEF"])
            leg = bond_factors().attrs.get("def_leg", "LQD")
            notes.append(f"DEF (credit) left out: it needs the corporate bond series {leg}, which starts "
                         f"{d0.date() if d0 is not None else 'later'}, and these returns start {r.index[0].date()}. "
                         "Set the start to after that date to include it.")
    if "TERM" in cols:
        leg = bond_factors().attrs.get("term_leg", "TLTSIM")
        dleg = bond_factors().attrs.get("def_leg", "LQD")
        notes.append(f"TERM = {leg} - BILSIM total returns" + (f"; DEF = {dleg} - IEFSIM." if "DEF" in cols else "."))
    if REGION.get(model) == "dev":
        notes.append("Developed-markets factors (Kenneth French's 'Developed' series, US dollars, from July 1990).")
    fd = f.reindex(r.index).dropna(subset=cols + ["RF"])
    r = r.reindex(fd.index)
    if freq == "monthly":
        rm = (1 + r).resample("ME").prod() - 1
        fm = monthly_factors(fd)
        df = pd.concat([rm.rename("r"), fm], axis=1, join="inner").dropna(subset=["r", "RF"] + cols)
        # drop the first/last month only when the data covers part of it (a short month in the
        # middle, like September 2001, is a full month of returns)
        days = r.resample("ME").count().reindex(df.index)
        keep = pd.Series(True, index=df.index)
        if len(df):
            first_m, last_m = r.index[0], r.index[-1]
            if first_m.day > 7 and days.iloc[0] < 0.8 * days.median():
                keep.iloc[0] = False
            if len(df) > 1 and days.iloc[-1] < 0.8 * days.median() and (last_m + pd.offsets.BMonthEnd(0)).date() != last_m.date():
                keep.iloc[-1] = False
        df = df[keep]
        per_year, min_obs = 12, 24
    else:
        df = pd.concat([r.rename("r"), fd], axis=1, join="inner").dropna(subset=["r", "RF"] + cols)
        per_year, min_obs = 252, 120
    if len(df) < min_obs:
        fe = f[cols].dropna().index
        raise ValueError(f"Need at least {min_obs} {freq} observations that overlap the factor data "
                         f"(factors {fe[0].date() if len(fe) else '?'} to {fe[-1].date() if len(fe) else '?'}); got {len(df)}.")
    y = (df["r"] - df["RF"]).to_numpy()
    X = np.column_stack([np.ones(len(df))] + [df[c].to_numpy() for c in cols])
    o = ols(y, X)
    names = ["Alpha"] + cols
    coefs = []
    for i, n in enumerate(names):
        coefs.append({"factor": n, "about": DESCRIBE.get(n, "intercept (return not explained by the factors)"),
                      "loading": float(o["coef"][i]), "std_error": float(o["se"][i]), "t_stat": float(o["t"][i]),
                      "p_value": float(o["p"][i]),
                      "significant": bool(abs(o["t"][i]) >= 1.96)})
    alpha = o["coef"][0]
    out = {"name": name, "model": model, "model_label": label, "freq": freq, "factors": cols,
           "start": df.index[0].date(), "end": df.index[-1].date(), "observations": int(o["n"]),
           "r_squared": o["r2"], "adj_r_squared": o["adj_r2"],
           "alpha_period": float(alpha), "alpha_annual": float((1 + alpha) ** per_year - 1),
           "alpha_annual_simple": float(alpha * per_year),
           "alpha_t": float(o["t"][0]), "alpha_p": float(o["p"][0]),
           "tracking_error_annual": float(o["resid_sd"] * np.sqrt(per_year)),
           "coefficients": coefs, "notes": notes,
           "explained": {"mean_excess_annual": float(y.mean() * per_year),
                         "contributions_annual": {c: float(o["coef"][i + 1] * df[c].mean() * per_year) for i, c in enumerate(cols)}},
           "rolling": rolling(df, cols, rolling_months, freq)}
    return out


def rolling(df: pd.DataFrame, cols: list[str], months: int, freq: str) -> dict:
    """Loadings re-estimated at each month end on the trailing `months` months of observations."""
    ends = df.index.to_series().groupby(df.index.to_period("M")).max()
    y_all = (df["r"] - df["RF"]).to_numpy()
    X_all = np.column_stack([np.ones(len(df))] + [df[c].to_numpy() for c in cols])
    need = int(months * (0.8 if freq == "monthly" else 15))
    dates, rows = [], {c: [] for c in ["Alpha"] + cols}
    r2 = []
    pos = pd.Series(np.arange(len(df)), index=df.index)
    for d in ends:
        lo = d - pd.DateOffset(months=months)
        sel = pos[(pos.index > lo) & (pos.index <= d)].to_numpy()
        if len(sel) < need or len(sel) <= len(cols) + 2:
            continue
        o = ols(y_all[sel], X_all[sel])
        dates.append(d.strftime("%Y-%m-%d"))
        per_year = 12 if freq == "monthly" else 252
        rows["Alpha"].append(round(float(o["coef"][0] * per_year), 5))
        for i, c in enumerate(cols):
            rows[c].append(round(float(o["coef"][i + 1]), 4))
        r2.append(round(float(o["r2"]), 4))
    return {"window_months": months, "dates": dates, "loadings": rows, "r_squared": r2}


def console(R: dict) -> str:
    L = [f"{R['model_label']} regression of {R['name']} ({R['freq']} excess returns, {R['start']} -> {R['end']}, "
         f"{R['observations']} observations)",
         f"  {'':8s} {'loading':>9s} {'t-stat':>7s} {'p-value':>8s}"]
    for c in R["coefficients"]:
        L.append(f"  {c['factor']:8s} {c['loading']:>9.4f} {c['t_stat']:>7.2f} {c['p_value']:>8.4f}{'  *' if c['significant'] else ''}")
    L.append(f"  Alpha {R['alpha_annual'] * 100:.2f}%/yr (t {R['alpha_t']:.2f})   R² {R['r_squared']:.3f}   adj. R² {R['adj_r_squared']:.3f}"
             f"   residual volatility {R['tracking_error_annual'] * 100:.1f}%/yr")
    L.append("  * |t| >= 1.96 (significant at about 5%)")
    for n in R.get("notes") or []:
        L.append("  Note: " + n)
    return "\n".join(L)


_load_extra_files()
