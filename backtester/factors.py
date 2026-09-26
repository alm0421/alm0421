"""Factor regressions (Portfolio Visualizer's "factor analysis"): CAPM, Fama-French 3, Carhart 4,
Fama-French 5, FF5 + momentum for the US and for Kenneth French's international regions, AQR's quality
(QMJ) and betting-against-beta (BAB) factors, and bond factors (TERM, DEF), on daily or monthly excess
returns, with t-stats, R², annualised alpha and rolling 36-month loadings.

Models are named "<region>_<kind>": kind is ff3, ff5, carhart (FF3 + momentum) or ff6 (FF5 + momentum);
the US models have no prefix (ff3, ff5, ...). Regions: developed (all developed markets incl. the US),
developed_ex_us (key dev_ff3 for its 3-factor model, alias intl), europe, japan, asia_pacific_ex_japan,
north_america, emerging (monthly only). Add-ons join with "+": "ff5+qmj+bab", "europe_ff5+qmj",
"ff3+bonds", "ff3+mom". "auto" picks the region from the ticker (EFA -> developed_ex_us, EEM -> emerging...).

Factor data (data/factors, downloaded by scripts/fetch_data.py; see backtester/sources.py for the formats):
- Monthly regressions use Kenneth French's official MONTHLY files: ff3_monthly.csv, ff5_monthly.csv,
  mom_monthly.csv (US) and <region>_ff3_monthly.csv, <region>_ff5_monthly.csv, <region>_mom_monthly.csv
  (em_ff5_monthly.csv and emerging_mom_monthly.csv for emerging markets). French builds his monthly factors
  from monthly portfolio returns, so they differ slightly from compounded daily factors. Until a monthly file
  has been downloaded, its daily counterpart is compounded instead (with a note).
- Daily regressions use the daily files: ff3_daily.csv (else the first three factors of ff5_daily.csv),
  ff5_daily.csv, mom_daily.csv and <region>_<ff3|ff5|mom>_daily.csv (dev_ff3_daily.csv for developed ex US).
  French publishes no daily emerging-markets factors.
- AQR: aqr_qmj_monthly.csv and aqr_bab_monthly.csv (one column per country/aggregate); QMJ and BAB use the
  column of the model's region (USA, Global, Global Ex USA, Europe, North America, JPN). Monthly only. AQR
  publishes no emerging or Asia Pacific ex Japan aggregate.
- Bond factors, built from the price data (total returns, adj_close):
    TERM = long Treasuries (TLTSIM, else IEFSIM) - T-bills (BILSIM): the term premium
    DEF  = investment-grade corporates (LQDSIM when the data job has built it, else LQD from July 2002) -
           Treasuries of similar duration (IEFSIM): the credit (default) premium. When the returns being
           explained start before LQD, DEF is left out with a note.
  For monthly regressions each leg is compounded over the month and the factor is the difference.
- Other factors: put a CSV of daily decimal returns named data/factors/extra_<name>.csv (a "date" column plus
  one column per factor) and every column becomes a factor, with a model "ff3+<name>"; or call
  register_factor() / add_model() from Python.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Callable

import numpy as np
import pandas as pd

from . import data, metrics

BASE = ("Mkt-RF", "SMB", "HML", "RMW", "CMA")
KINDS = {"ff3": ["Mkt-RF", "SMB", "HML"], "ff5": ["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
         "carhart": ["Mkt-RF", "SMB", "HML", "Mom"], "ff6": ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]}
KIND_LABEL = {"ff3": "Fama-French 3", "ff5": "Fama-French 5", "carhart": "Carhart 4 (FF3 + momentum)",
              "ff6": "Fama-French 5 + momentum"}
# region key -> (label, AQR column for QMJ/BAB or None, daily factors published)
REGIONS = {
    "us": ("US", "USA", True),
    "developed": ("Developed markets (incl. US)", "Global", True),
    "developed_ex_us": ("Developed markets ex US", "Global Ex USA", True),
    "europe": ("Europe", "Europe", True),
    "japan": ("Japan", "JPN", True),
    "asia_pacific_ex_japan": ("Asia Pacific ex Japan", None, True),
    "north_america": ("North America", "North America", True),
    "emerging": ("Emerging markets", None, False),
}
SHORT = {"dev": "developed_ex_us", "devexus": "developed_ex_us", "intl": "developed_ex_us", "global": "developed",
         "world": "developed", "eu": "europe", "jp": "japan", "apxj": "asia_pacific_ex_japan",
         "asia": "asia_pacific_ex_japan", "pacific": "asia_pacific_ex_japan", "na": "north_america", "em": "emerging",
         "emerging_markets": "emerging", "developed_markets": "developed"}

# model key -> (label, factor columns). Extend with add_model().
MODELS = {
    "capm": ("CAPM", ["Mkt-RF"]),
    "ff3": ("Fama-French 3", KINDS["ff3"]),
    "carhart": ("Carhart 4", KINDS["carhart"]),
    "ff5": ("Fama-French 5", KINDS["ff5"]),
    "ff6": ("Fama-French 5 + momentum", KINDS["ff6"]),
    "bonds": ("Market + bond factors", ["Mkt-RF", "TERM", "DEF"]),
    "ff3+bonds": ("Fama-French 3 + bond factors", ["Mkt-RF", "SMB", "HML", "TERM", "DEF"]),
}
ALIASES = {"ff3_bonds": "ff3+bonds", "ff4": "carhart", "developed": "developed_ff3", "international": "dev_ff3"}
# where a model's market/size/value factors come from ("us" = French's US files)
REGION: dict[str, str] = {}
DESCRIBE = {"Mkt-RF": "market excess return", "SMB": "size (small minus big)", "HML": "value (high minus low book/price)",
            "RMW": "profitability (robust minus weak)", "CMA": "investment (conservative minus aggressive)",
            "Mom": "momentum (winners minus losers)",
            "QMJ": "quality minus junk (AQR: profitable, growing, safe, high-payout stocks minus the opposite)",
            "BAB": "betting against beta (AQR: leveraged low-beta minus de-leveraged high-beta stocks)",
            "TERM": "term premium (long Treasuries minus T-bills)",
            "DEF": "credit premium (investment-grade corporates minus Treasuries)"}
ABOUT = {
    "capm": "the market alone",
    "ff3": "market, size and value (US)",
    "carhart": "Fama-French 3 plus momentum (US)",
    "ff5": "adds profitability and investment (US, from 1963)",
    "ff6": "Fama-French 5 plus momentum (US)",
    "bonds": "the US market plus the term and credit premiums (TERM from 1962, DEF from 2002)",
    "ff3+bonds": "Fama-French 3 plus TERM and DEF, for stock/bond portfolios",
}
_START = {"developed": 1990, "developed_ex_us": 1990, "europe": 1990, "japan": 1990, "asia_pacific_ex_japan": 1990,
          "north_america": 1990, "emerging": 1989}
for _r, (_lab, _aqr, _daily) in REGIONS.items():
    if _r == "us":
        continue
    for _k, _cols in KINDS.items():
        _key = "dev_ff3" if (_r, _k) == ("developed_ex_us", "ff3") else f"{_r}_{_k}"
        MODELS[_key] = (f"{_lab} {KIND_LABEL[_k]}", list(_cols))
        REGION[_key] = _r
        ABOUT[_key] = (f"Kenneth French's {_lab} factors, from {_START[_r]}"
                       + ("" if _daily else ", monthly only") + ("; also 'intl'" if _key == "dev_ff3" else ""))
        if _key == "dev_ff3":
            ALIASES["developed_ex_us_ff3"] = "dev_ff3"
            ALIASES["intl"] = "dev_ff3"
        for _s, _t in SHORT.items():
            if _t == _r and f"{_s}_{_k}" not in MODELS:
                ALIASES.setdefault(f"{_s}_{_k}", _key)
# add-ons for "<model>+<addon>+..."
ADDONS = {"mom": ["Mom"], "momentum": ["Mom"], "umd": ["Mom"], "qmj": ["QMJ"], "quality": ["QMJ"], "bab": ["BAB"],
          "bonds": ["TERM", "DEF"], "term": ["TERM"], "def": ["DEF"]}

# tickers whose region is known, for "auto" and for the hint when a model's region does not match
TICKER_REGION = {
    **{t: "us" for t in ("SPY", "SPYSIM", "VOO", "IVV", "VTI", "QQQ", "QQQM", "IWM", "IWB", "IWV", "DIA", "VTV", "VUG",
                          "VB", "VBR", "VBK", "IWD", "IWF", "IWN", "IWO", "MDY", "RSP", "SCHB", "SCHX", "SCHG", "VIG",
                          "VTVSIM", "VUGSIM", "VBSIM", "VBRSIM", "VBKSIM")},
    **{t: "developed_ex_us" for t in ("EFA", "EFASIM", "VEA", "IEFA", "SCZ", "SCHF", "VEU", "IXUS", "VXUS", "VSS", "EFV", "EFG")},
    **{t: "europe" for t in ("VGK", "IEUR", "IEV", "EZU", "FEZ", "HEDJ", "EWU", "EWG", "EWQ", "EWI", "EWP", "EWL", "EWN", "EWD")},
    **{t: "japan" for t in ("EWJ", "DXJ", "BBJP", "HEWJ", "SCJ")},
    **{t: "asia_pacific_ex_japan" for t in ("EPP", "EWA", "EWH", "EWS", "ENZL")},
    **{t: "north_america" for t in ("EWC",)},
    **{t: "emerging" for t in ("EEM", "EEMSIM", "VWO", "IEMG", "SCHE", "EEMV", "DEM", "SPEM", "FXI", "EWZ", "INDA", "EWT", "EWY", "MCHI")},
    **{t: "developed" for t in ("VT", "ACWI", "URTH", "IOO", "ACWX")},
}

# extra factors: name -> (about, loader returning a DataFrame with that column, and optional legs
# "<name>|a" / "<name>|b" when the factor is the difference of two return series; frequency of the loader)
_EXTRA: dict[str, tuple[str, Callable[[], pd.DataFrame]]] = {}
_EXTRA_FREQ: dict[str, str] = {}


def register_factor(name: str, about: str, loader: Callable[[], pd.DataFrame], freq: str = "daily") -> None:
    """Make a factor available to models. `freq` is the frequency of the loader's returns: daily factors are
    compounded for monthly regressions; monthly factors can only be used in monthly regressions."""
    _EXTRA[name] = (about, loader)
    _EXTRA_FREQ[name] = freq
    DESCRIBE[name] = about
    factor_table.cache_clear()
    factor_data.cache_clear()


def add_model(key: str, label: str, factors: list[str], about: str = "", region: str = "us") -> None:
    MODELS[key] = (label, list(factors))
    ABOUT[key] = about or label
    if region != "us":
        REGION[key] = region
    factor_table.cache_clear()
    factor_data.cache_clear()


def _base_key(m: str) -> str | None:
    low = m.strip().lower()
    if low in ALIASES:
        return ALIASES[low]
    if m.strip() in MODELS:
        return m.strip()
    if low in MODELS:
        return low
    if low in REGIONS and low != "us":                # "europe" -> europe_ff3
        return "dev_ff3" if low == "developed_ex_us" else f"{low}_ff3"
    if low in SHORT:
        r = SHORT[low]
        return "dev_ff3" if r == "developed_ex_us" else f"{r}_ff3"
    return None


def _addon_cols(part: str) -> list[str] | None:
    p = part.strip().lower()
    if p in ADDONS:
        return ADDONS[p]
    for name in _EXTRA:
        if name.lower() == p:
            return [name]
    return None


def resolve(model: str) -> str:
    """The canonical key of a model name: a listed model, an alias, or "<model>+<add-on>+..." (add-ons: mom,
    qmj, bab, bonds, term, def, or a registered factor)."""
    m = str(model or "ff3").strip()
    base = _base_key(m)
    if base is not None:
        return base
    if "+" in m:
        parts = [p.strip() for p in m.split("+") if p.strip()]
        base = _base_key(parts[0])
        if base is not None and "+" not in base and len(parts) > 1:
            bad = [p for p in parts[1:] if _addon_cols(p) is None]
            if not bad:
                return "+".join([base] + [p.lower() for p in parts[1:]])
            raise ValueError(f"Unknown factor add-on {', '.join(bad)}: use {', '.join(sorted(set(ADDONS)))}"
                             + (f" or {', '.join(_EXTRA)}" if _EXTRA else "") + ".")
    raise ValueError(f"model must be one of {', '.join(list(MODELS) + list(ALIASES))}, a region "
                     f"({', '.join(r for r in REGIONS if r != 'us')}), 'auto', or a model plus add-ons like 'ff5+qmj+bab'")


def model_spec(model: str) -> tuple[str, list[str], str]:
    """(label, factor columns, region) of a resolved model key."""
    key = resolve(model)
    if key in MODELS:
        label, cols = MODELS[key]
        return label, list(cols), REGION.get(key, "us")
    parts = key.split("+")
    label, cols = MODELS[parts[0]]
    cols = list(cols)
    extra = []
    for p in parts[1:]:
        for c in _addon_cols(p) or []:
            if c not in cols:
                cols.append(c)
                extra.append(c)
    return f"{label} + {', '.join(extra)}" if extra else label, cols, REGION.get(parts[0], "us")


def model_list() -> list[dict]:
    """The models for the CLI help and the Factors page."""
    return [{"key": k, "label": v[0], "factors": v[1], "about": ABOUT.get(k, ""),
             "region": REGION.get(k, "us"), "region_label": REGIONS.get(REGION.get(k, "us"), ("US",))[0],
             "aliases": [a for a, t in ALIASES.items() if t == k]} for k, v in MODELS.items()]


def suggest_model(name: str, kind: str = "ff3") -> tuple[str, str] | None:
    """(model key, region label) matching a ticker's region, or None when the ticker's region is not known."""
    t = str(name or "").strip().upper()
    r = TICKER_REGION.get(t)
    if r is None:
        return None
    if r == "us":
        return kind, REGIONS[r][0]
    key = "dev_ff3" if (r, kind) == ("developed_ex_us", "ff3") else f"{r}_{kind}"
    return key, REGIONS[r][0]


def auto_model(name: str, kind: str = "ff3") -> str:
    s = suggest_model(name, kind)
    return s[0] if s else kind


def model_hint(name: str, model: str) -> str | None:
    """A note when a ticker's region does not match the model's (EFA on the US model, SPY on Europe's...)."""
    s = suggest_model(name)
    if not s:
        return None
    try:
        _, _, region = model_spec(model)
    except ValueError:
        return None
    want = TICKER_REGION[str(name).strip().upper()]
    if want == region:
        return None
    return (f"{str(name).upper()} invests in {REGIONS[want][0]}, but this model uses {REGIONS[region][0]} factors: "
            f"try --model {s[0]} (or 'auto').")


def _read(name: str) -> pd.DataFrame:
    p = data.DATA / "factors" / name
    if not p.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(p, parse_dates=["date"], index_col="date").sort_index()
    except (ValueError, KeyError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return pd.DataFrame()
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={c: "Mom" for c in df.columns if c.upper() in ("WML", "MOM", "UMD")})
    df = df.apply(pd.to_numeric, errors="coerce")
    if name.endswith("_monthly.csv"):
        df.index = df.index.to_period("M").to_timestamp("M")
        df = df[~df.index.duplicated(keep="last")]
    return df


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


def _files(region: str, kind: str, freq: str) -> list[str]:
    """Candidate local files for a region's base factors (kind ff3/ff5) or momentum (kind mom), best first."""
    if region == "us":
        if kind == "mom":
            return [f"mom_{freq}.csv"]
        return [f"ff5_{freq}.csv"] if kind == "ff5" else [f"ff3_{freq}.csv", f"ff5_{freq}.csv"]
    names = [f"{region}_{kind}_{freq}.csv"]
    if kind == "ff3":
        if region == "developed_ex_us" and freq == "daily":
            names.append("dev_ff3_daily.csv")          # the name used before the regional downloads
        names.append(f"{region}_ff5_{freq}.csv")         # the first three factors of the 5-factor file
    if region == "emerging" and kind in ("ff3", "ff5") and freq == "monthly":
        names.append("em_ff5_monthly.csv")
    return names


def _first(names: list[str], need: list[str]) -> tuple[pd.DataFrame, str | None]:
    for n in names:
        df = _read(n)
        if not df.empty and all(c in df for c in need):
            return df, n
    return pd.DataFrame(), None


def compound_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    """Monthly factor returns (month-end index) compounded from daily ones: the market as (1 + Mkt-RF + RF)
    compounded minus compounded RF, spread factors with legs ("X|a", "X|b") as the difference of the legs'
    compounded returns, the other long-short factors compounded directly. NaN where a month has no data."""
    legs = [c for c in daily.columns if "|" in c]
    g = (1 + daily).resample("ME").prod(min_count=1) - 1
    out = g.drop(columns=legs)
    if "Mkt-RF" in daily and "RF" in daily:
        mkt = (1 + daily["Mkt-RF"] + daily["RF"]).resample("ME").prod(min_count=1) - 1
        out["Mkt-RF"] = mkt - g["RF"]
    for c in {x.split("|")[0] for x in legs}:
        if f"{c}|a" in g and f"{c}|b" in g:
            out[c] = g[f"{c}|a"] - g[f"{c}|b"]
    return out


def monthly_factors(daily: pd.DataFrame) -> pd.DataFrame:
    """compound_monthly() over the months that have risk-free data (the fallback before the official monthly
    files are downloaded, and the monthly TERM/DEF)."""
    out = compound_monthly(daily)
    if "RF" in daily:
        counts = daily["RF"].resample("ME").count()
        return out[counts > 0]
    return out.dropna(how="all")


@lru_cache(maxsize=32)
def factor_data(model: str, freq: str = "daily") -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Factor returns (decimal) of a model at a frequency, plus RF (and the legs of spread factors in daily
    tables), and notes about the sources. Monthly tables are indexed by calendar month end. A factor that
    starts later than the others (DEF) is NaN before it starts; analyze() decides whether to use it."""
    if freq not in ("daily", "monthly"):
        raise ValueError("freq must be daily or monthly")
    label, cols, region = model_spec(model)
    rlabel, aqr_col, has_daily = REGIONS.get(region, ("US", "USA", True))
    notes: list[str] = []
    if freq == "daily" and not has_daily:
        raise ValueError(f"Kenneth French publishes {rlabel} factors monthly only: use monthly returns.")
    monthly_only = [c for c in cols if c in ("QMJ", "BAB") or _EXTRA_FREQ.get(c) == "monthly"]
    if freq == "daily" and monthly_only:
        raise ValueError(f"{', '.join(monthly_only)} {'is a' if len(monthly_only) == 1 else 'are'} monthly factor"
                         f"{'' if len(monthly_only) == 1 else 's'} (AQR publishes QMJ and BAB monthly here): use monthly returns.")
    if aqr_col is None and any(c in ("QMJ", "BAB") for c in cols):
        raise ValueError(f"AQR publishes no {'/'.join(c for c in cols if c in ('QMJ', 'BAB'))} factor for {rlabel}.")
    base_cols = [c for c in cols if c in BASE]
    kind = "ff5" if any(c in ("RMW", "CMA") for c in cols) else "ff3"
    need = base_cols + ["RF"]
    base, src = _first(_files(region, kind, freq), need)
    if base.empty and freq == "monthly":
        daily, dsrc = _first(_files(region, kind, "daily"), need)
        if not daily.empty:
            base = monthly_factors(daily[need])
            notes.append(f"Monthly {rlabel} factors compounded from the daily file {dsrc}: the official monthly file "
                         f"({_files(region, kind, 'monthly')[0]}) has not been downloaded yet.")
    elif freq == "monthly" and src:
        notes.append(f"Monthly factors: Kenneth French's official monthly {rlabel} file ({src}).")
    if base.empty:
        where = "data/factors/" + _files(region, kind, freq)[0]
        raise ValueError(f"{rlabel} factor data is missing ({where}); the data job downloads it.")
    df = base[need].dropna()
    if "Mom" in cols:
        mom, msrc = _first(_files(region, "mom", freq), ["Mom"])
        if mom.empty and freq == "monthly":
            md, _ = _first(_files(region, "mom", "daily"), ["Mom"])
            if not md.empty:
                mom = monthly_factors(md[["Mom"]])
                notes.append(f"Monthly {rlabel} momentum compounded from the daily file (the monthly file is not downloaded yet).")
        if mom.empty:
            raise ValueError(f"{rlabel} momentum factor data is missing (data/factors/{_files(region, 'mom', freq)[0]}).")
        df = df.join(mom[["Mom"]], how="inner").dropna(subset=["Mom"])
    for c in ("QMJ", "BAB"):
        if c not in cols:
            continue
        if freq == "daily":
            raise ValueError(f"{c} comes from AQR's monthly file: use monthly returns.")
        if aqr_col is None:
            raise ValueError(f"AQR publishes no {c} factor for {rlabel}.")
        f = _read(f"aqr_{c.lower()}_monthly.csv")
        if f.empty or aqr_col not in f:
            raise ValueError(f"AQR {c} data is missing (data/factors/aqr_{c.lower()}_monthly.csv, column {aqr_col}); "
                             "the data job downloads it.")
        df = df.join(f[[aqr_col]].rename(columns={aqr_col: c}), how="inner").dropna(subset=[c])
        notes.append(f"{c}: AQR's {aqr_col} factor (monthly, AQR data library).")
    if "TERM" in cols or "DEF" in cols:
        bf = bond_factors()
        need_b = [c for c in ("TERM", "DEF") if c in cols]
        if bf.empty or "TERM" in need_b and "TERM" not in bf:
            raise ValueError("Bond factors need TLTSIM (or IEFSIM) and BILSIM price data.")
        keep = [c for c in bf.columns if c.split("|")[0] in need_b]
        b = bf[keep]
        if freq == "monthly":
            b = compound_monthly(b)[[c for c in need_b if c in bf]]
        df = df.join(b, how="inner")
        df = df[df["TERM"].notna()] if "TERM" in df else df
    for c in cols:
        if c in _EXTRA:
            ex = _EXTRA[c][1]()
            ex = ex[[x for x in ex.columns if x.split("|")[0] == c]]
            efreq = _EXTRA_FREQ.get(c, "daily")
            if freq == "daily" and efreq == "monthly":
                raise ValueError(f"{c} is a monthly factor: use monthly returns.")
            if freq == "monthly":
                if efreq == "daily":
                    ex = compound_monthly(ex)[[c]]
                else:
                    ex = ex[[c]].copy()
                    ex.index = pd.DatetimeIndex(ex.index).to_period("M").to_timestamp("M")
            df = df.join(ex, how="inner").dropna(subset=[c])
    return df, tuple(notes)


def factor_table(model: str, freq: str = "daily") -> pd.DataFrame:
    """The factor returns of factor_data() without the notes."""
    return factor_data(resolve(model), freq)[0]


factor_table.cache_clear = factor_data.cache_clear  # type: ignore[attr-defined]


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
    notes: list[str] = []
    m0 = str(model or "").strip()
    if m0.lower() == "auto" or m0.lower().startswith("auto+"):
        model = auto_model(name) + m0[4:]
        notes.append(f"Model chosen automatically for {name}: {model_spec(model)[0]}.")
    model = resolve(model)
    label, cols, region = model_spec(model)
    f, fnotes = factor_data(model, freq)
    notes += list(fnotes)
    hint = model_hint(name, model)
    if hint:
        notes.append(hint)
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
    if region != "us":
        notes.append(f"{REGIONS[region][0]} factors from Kenneth French's data library (US dollars; the risk-free rate is the US one-month T-bill).")
    if freq == "monthly":
        rm = (1 + r).resample("ME").prod(min_count=1) - 1
        df = pd.concat([rm.rename("r"), f], axis=1, join="inner").dropna(subset=["r", "RF"] + cols)
        # drop the first/last month only when the data covers part of it (a short month in the
        # middle, like September 2001, is a full month of returns)
        days = r.resample("ME").count().reindex(df.index)
        keep = pd.Series(True, index=df.index)
        if len(df):
            first_m, last_m = r.index[0], r.index[-1]
            if df.index[0] == first_m + pd.offsets.MonthEnd(0) and first_m.day > 7 and days.iloc[0] < 0.8 * days.median():
                keep.iloc[0] = False
            if (len(df) > 1 and df.index[-1] == last_m + pd.offsets.MonthEnd(0) and days.iloc[-1] < 0.8 * days.median()
                    and (last_m + pd.offsets.BMonthEnd(0)).date() != last_m.date()):
                keep.iloc[-1] = False
        df = df[keep]
        per_year, min_obs = 12, 24
    else:
        fd = f.reindex(r.index).dropna(subset=cols + ["RF"])
        r = r.reindex(fd.index)
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
    out = {"name": name, "model": model, "model_label": label, "region": region, "freq": freq, "factors": cols,
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
