"""Load daily price history and reference data from data/."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PRICES = DATA / "prices"
UNIVERSE_FILE = DATA / "universe.json"
MEMBERSHIP_FILE = DATA / "ndx_membership.csv"

BENCHMARKS = ["SPY", "QQQ"]
ALIASES = {"VIX": "^VIX", "NDX100": "^NDX", "SPX": "^GSPC", "GSPC": "^GSPC", "IRX": "^IRX", "TNX": "^TNX",
           "RUT": "^RUT", "DJI": "^DJI", "NASDAQ100": "^NDX"}


class DataError(ValueError):
    pass


def canonical(ticker: str) -> str:
    t = ticker.strip().upper().lstrip("$")
    return ALIASES.get(t, t)


def available_tickers() -> list[str]:
    return sorted(p.stem for p in PRICES.glob("*.csv"))


def suggest(ticker: str, n: int = 5) -> list[str]:
    """The `n` available tickers closest in spelling to `ticker` (for "did you mean" messages)."""
    from difflib import SequenceMatcher
    t = canonical(ticker).lstrip("^")

    def score(h: str) -> float:
        b = h.lstrip("^")
        return (SequenceMatcher(None, t, b).ratio() + 0.3 * (sorted(t) == sorted(b))
                + 0.1 * (b[:1] == t[:1]) - 0.05 * abs(len(b) - len(t)))
    ranked = sorted(available_tickers(), key=lambda h: -score(h))
    return [h for h in ranked[:n] if score(h) > 0.3]


def unknown_ticker_message(ticker: str) -> str:
    t = canonical(ticker)
    s = suggest(t)
    return (f"No price data for {t}." + (f" Did you mean {', '.join(s)}?" if s else "")
            + f" To add {t}, put it on its own line in data/extra_tickers.txt and run the 'Fetch price data' workflow "
            "(GitHub Actions; pushing the file also starts it), then pull the new data. (The Data page lists every "
            "ticker; on your own computer new tickers are downloaded automatically.)")


@lru_cache(maxsize=1)
def universe_meta() -> dict:
    return json.loads(UNIVERSE_FILE.read_text()) if UNIVERSE_FILE.exists() else {}


def nasdaq100() -> list[str]:
    """Today's Nasdaq-100 members."""
    m = universe_meta()
    if m.get("nasdaq100"):
        return list(m["nasdaq100"])
    return [t for t in available_tickers() if t not in BENCHMARKS and not t.startswith("^") and not is_sim(t)]


def etfs() -> list[str]:
    return list(universe_meta().get("etfs", []))


def funds() -> list[str]:
    """Mutual funds downloaded from the broad list (backtester/fund_lists.py)."""
    return list(universe_meta().get("funds", []))


# Long-history simulated series built by scripts/fetch_data.py (build_sims): a total-return index
# (close = adj_close, no dividends, volume 0, open = high = low = close) that uses a model before
# the fund existed and the real fund's total return after.
SIMS = {"SPYSIM": "US stock market (Fama-French market return) from 1926, spliced into SPY",
        "TLTSIM": "20-year Treasuries priced from FRED yields from 1962, spliced into TLT",
        "IEFSIM": "~9-year Treasuries from the 10-year yield from 1962, spliced into IEF",
        "SHYSIM": "2-year Treasuries from the 2-year yield from 1962, spliced into SHY",
        "BILSIM": "1-month T-bills (Fama-French RF) from 1926, spliced into BIL",
        "IEISIM": "5-year Treasuries from the 5-year yield from 1962, spliced into IEI",
        "BNDSIM": "US aggregate bonds: 70% 5-year Treasury / 30% IG corporate model from 1962, the Vanguard Total Bond "
                  "Market Index fund (VBMFX) from Dec 1986, then BND",
        "LQDSIM": "Investment-grade corporates priced off Moody's Aaa/Baa yields from 1953, spliced into LQD",
        "HYGSIM": "US high-yield bonds: the Vanguard High-Yield Corporate fund (VWEHX) from 1985, then HYG (no model before)",
        "TIPSIM": "US TIPS: the Vanguard Inflation-Protected Securities fund (VIPSX) from mid-2000, then TIP (no model before)",
        "BNDXSIM": "International government bonds hedged to USD: a par-bond model on OECD 10-year yields of up to 12 "
                   "developed markets (monthly steps) from 1970, PIMCO International Bond USD-hedged (PFORX) from 1993, then BNDX",
        "VBSIM": "US small caps (Fama-French small portfolios) from 1926, the Vanguard Small-Cap Index fund (NAESX) from "
                 "late 1989, then VB",
        "VBRSIM": "US small-cap value (the Fama-French model that tracks the fund best, see data/sims_log.txt) from 1926, "
                  "the Vanguard Small-Cap Value Index fund (VISVX) from 1998, then VBR",
        "VBKSIM": "US small-cap growth (the Fama-French model that tracks the fund best) from 1926, the Vanguard Small-Cap "
                  "Growth Index fund (VISGX) from 1998, then VBK",
        "MIDSIM": "US mid caps (Fama-French 30th-70th NYSE size percentiles) from 1926, spliced into MDY (S&P 400)",
        "VTVSIM": "US large-cap value (the Fama-French model that tracks the fund best, e.g. 1/3 big/high + 2/3 big/neutral "
                  "B/M) from 1926, the Vanguard Value Index fund (VIVAX) from 1992, then VTV",
        "VUGSIM": "US large-cap growth (Fama-French big/low B/M) from 1926, the Vanguard Growth Index fund (VIGRX) from "
                  "1992, then VUG",
        "EFASIM": "Developed ex-US stocks: Fama-French EAFE index (monthly steps) from 1975, daily from 1990, spliced into EFA",
        "EFVSIM": "Developed ex-US value: Fama-French EAFE high-B/M index (monthly steps) from 1975, daily big/high B/M "
                  "from 1990, spliced into EFV",
        "SCZSIM": "Developed ex-US small caps (Fama-French, daily) from 1990, spliced into SCZ",
        "AVDVSIM": "Developed ex-US small-cap value (Fama-French small/high B/M, daily) from 1990, spliced into AVDV",
        "VGKSIM": "European stocks: Fama-French Europe index (monthly steps) from 1975, daily from 1990, spliced into VGK",
        "EEMSIM": "Emerging markets (Fama-French, monthly steps) from 1989, spliced into EEM",
        "VNQSIM": "US REITs: FTSE Nareit All Equity REITs total return (monthly steps) from 1972, the Vanguard REIT Index "
                  "fund (VGSIX) from 1996, then VNQ",
        "GLDSIM": "Gold: LBMA PM fixing (daily) from April 1968 (World Bank monthly average price 1960-68), spliced into GLD",
        "DBCSIM": "Commodity futures: AQR equal-weight commodity index excess return + T-bills (monthly steps) from 1960, "
                  "spliced into DBC (through the PIMCO CommodityRealReturn fund PCRIX from 2002 when it tracks DBC better; "
                  "see data/sims_log.txt)",
        # "fund-exact": the named fund as soon as it or its mutual-fund twin exists
        "VTISIM": "US total stock market: Fama-French market return from 1926, the Vanguard Total Stock Market Index fund "
                  "(VTSMX) from April 1992, then VTI from 2001",
        "VXUSSIM": "Total international stocks: 80% developed ex-US (Fama-French EAFE) + 20% emerging markets (from 1989) "
                   "from 1975, the Vanguard Total International Stock Index fund (VGTSX) from 1996, then VXUS from 2011",
        "VWOSIM": "Emerging markets (Fama-French, monthly steps) from 1989, the Vanguard Emerging Markets Stock Index fund "
                  "(VEIEX) from 1994, then VWO",
        "VOESIM": "US mid-cap value (Fama-French 25 size x B/M portfolios, mid size / high B/M) from 1926, spliced into VOE",
        "VOTSIM": "US mid-cap growth (Fama-French 25 size x B/M portfolios, mid size / low B/M) from 1926, spliced into VOT",
        "VCLTSIM": "Long-term IG corporates: 20-year par bond at Moody's Aaa/Baa average yield from 1953, the Vanguard "
                   "Long-Term Investment-Grade fund (VWESX) from its Yahoo history (1980), then VCLT from 2009",
        "MUBSIM": "US municipal bonds: the Vanguard Intermediate-Term Tax-Exempt fund (VWITX) from its Yahoo history, then "
                  "MUB from 2007 (no model before)",
        "EMBSIM": "Emerging-market USD bonds: the Fidelity New Markets Income fund (FNMIX) from 1993, then EMB from 2007 "
                  "(no model before)",
        "EWJSIM": "Japanese stocks: Fama-French Japan index (monthly steps) from 1975, daily from 1990, spliced into EWJ",
        "EWUSIM": "UK stocks: Fama-French UK index (monthly steps) from 1975, spliced into EWU",
        "EWGSIM": "German stocks: Fama-French Germany index (monthly steps) from 1975, spliced into EWG",
        "EWCSIM": "Canadian stocks: Fama-French Canada index (monthly steps) from 1975, spliced into EWC",
        "EWASIM": "Australian stocks: Fama-French Australia index (monthly steps) from 1975, spliced into EWA",
        "EWQSIM": "French stocks: Fama-French France index (monthly steps) from 1975, spliced into EWQ",
        "EWLSIM": "Swiss stocks: Fama-French Switzerland index (monthly steps) from 1975, spliced into EWL",
        "EWHSIM": "Hong Kong stocks: Fama-French Hong Kong index (monthly steps) from 1975, spliced into EWH",
        }


def is_sim(ticker: str) -> bool:
    return canonical(ticker).endswith("SIM")


def sims() -> list[str]:
    """The simulated long-history series that exist in data/prices."""
    have = set(available_tickers())
    return [t for t in SIMS if t in have] + sorted(t for t in have if t.endswith("SIM") and t not in SIMS)


# symbols that show up in old revisions of the Wikipedia article but were never index members
LARGE_STOCKS = set("""JPM XOM BRK-B JNJ UNH V MA HD PG CVX LLY ABBV MRK KO BAC WFC DIS MCD NKE ORCL CRM IBM GE CAT
BA GS MS C T VZ PFE TMO DHR ABT NEE DUK SO LMT RTX UPS UNP MMM MSTR COIN""".split())
NOT_MEMBERS = {"NDX", "QQQ", "QQQQ", "TQQQ", "SQQQ", "QLD", "QID", "PSQ", "ONEQ", "NASDAQ", "ETF", "NQ", "ND",
               "NXP"}  # NXP is a Nuveen municipal fund (a typo for NXPI in one stretch of revisions)

# Symbols reused by a different company. The price file holds the current company, so it only stands
# for the index member from this date on (earlier member-months count as missing data).
IDENTITY_FROM = {
    "MNST": "2012-01-09",   # Monster Worldwide until 2011; the file is Monster Beverage (ex-HANS)
}
# the same company listed under two symbols in some revisions: keep the second
DUPLICATES = {"KLA": "KLAC", "WFMI": "WFM", "ERTS": "EA"}

# a member's series must look like a large Nasdaq stock: this catches recycled tickers (a small
# company that later took over a former member's symbol) and junk series with no trading
MIN_DOLLAR_VOLUME = 2_000_000.0


@lru_cache(maxsize=None)
def quality(ticker: str) -> pd.Series:
    """Per-bar bool: does the series look like a real, liquid listing on that day?

    Rules (20-bar window): median dollar volume >= $2M and at most half the bars with zero volume.
    (No price floor: prices are split-adjusted, so early AAPL or NVDA trade below $1.)"""
    try:
        df = load(ticker)
    except DataError:
        return pd.Series(dtype=bool)
    dv = (df["close"] * df["volume"]).rolling(20, min_periods=5).median().shift(1)
    zero = (df["volume"] <= 0).astype(float).rolling(20, min_periods=5).mean().shift(1)
    ok = (dv >= MIN_DOLLAR_VOLUME) & (zero <= 0.5)
    return ok.fillna(False)


def quality_report(tickers: list[str] | None = None) -> pd.DataFrame:
    """Member-months whose price series fails the quality rules, per ticker."""
    mem = membership()
    if mem is None:
        return pd.DataFrame(columns=["ticker", "member_months", "failed_months"])
    have = set(available_tickers())
    rows = []
    for t in tickers or [c for c in mem.columns if c in have]:
        months = mem.index[mem[t].to_numpy()] if t in mem else []
        if len(months) == 0:
            continue
        q = quality(t)
        if q.empty:
            continue
        per = q.groupby(q.index.to_period("M")).mean()
        failed = [m for m in months if per.get(m.to_period("M"), 0.0) < 0.5]
        if failed:
            rows.append({"ticker": t, "member_months": len(months), "failed_months": len(failed),
                         "first_failed": str(failed[0].date())[:7], "last_failed": str(failed[-1].date())[:7]})
    return pd.DataFrame(rows)


@lru_cache(maxsize=64)
def coverage(start=None, end=None) -> pd.DataFrame:
    """Per year: average number of index members, how many have usable price data, and the share."""
    mem = membership()
    if mem is None:
        return pd.DataFrame()
    have = set(available_tickers())
    m = mem
    if start is not None:
        m = m[m.index >= pd.Timestamp(start).to_period("M").to_timestamp()]
    if end is not None:
        m = m[m.index <= pd.Timestamp(end)]
    rows = []
    cols_all = [c for c in m.columns if c in have]
    # usable[d, c]: member c's price data passes the quality rules in snapshot month d (only asked of members)
    member = m[cols_all].to_numpy(dtype=bool)
    usable = np.zeros(member.shape, bool)
    month_starts = bool(len(m)) and bool(((m.index.day == 1) & (m.index == m.index.normalize())).all())
    months = m.index.year * 12 + m.index.month - 1
    for j, c in enumerate(cols_all):
        rows_j = np.flatnonzero(member[:, j])
        if not len(rows_j):
            continue
        if month_starts:     # calendar-month snapshots (the usual case): one table lookup per member-month
            share = _quality_by_month(c)
            usable[rows_j, j] = [share.get(int(months[i]), -1.0) >= 0.5 for i in rows_j]
        else:
            for i in rows_j:
                usable[i, j] = _quality_month(c, m.index[i])
    usable_by_date = pd.Series(usable.sum(axis=1), index=m.index)
    for y, g in m.groupby(m.index.year):
        tot = g.sum(axis=1).mean()
        ok = float(usable_by_date.loc[g.index].sum())
        rows.append({"year": int(y), "members": round(float(tot), 1), "with_data": round(ok / len(g), 1),
                     "coverage": ok / len(g) / tot if tot else 0.0})
    return pd.DataFrame(rows)


@lru_cache(maxsize=None)
def _quality_by_month(t: str) -> dict:
    """{calendar month as year * 12 + month - 1: share of the ticker's bars in it that pass `quality`}, computed once."""
    q = quality(t)
    if q.empty:
        return {}
    key = q.index.year * 12 + q.index.month - 1
    per = q.astype(float).groupby(np.asarray(key)).mean()
    return dict(zip(per.index.tolist(), per.to_numpy().tolist()))


_QUALITY_MONTHS: dict = {}   # ticker -> (its quality series, share of good bars per calendar month)


def _quality_month(t: str, month_start: pd.Timestamp) -> bool:
    if month_start.day == 1 and month_start == month_start.normalize():   # a calendar month: the cached table
        v = _quality_by_month(t).get(month_start.year * 12 + month_start.month - 1)
        return v is not None and v >= 0.5
    q = quality(t)
    if q.empty:
        return False
    if month_start != month_start.to_period("M").to_timestamp():   # not a month start: the window as written
        sl = q[(q.index >= month_start) & (q.index < month_start + pd.offsets.MonthBegin(1))]
        return bool(len(sl)) and sl.mean() >= 0.5
    v = _quality_months(t).get(month_start.to_period("M"))
    return v is not None and bool(v >= 0.5)


def _quality_months(t: str) -> dict:
    """{calendar month (Period): share of the ticker's bars that month that pass `quality`}."""
    q = quality(t)
    hit = _QUALITY_MONTHS.get(t)
    if hit is None or hit[0] is not q:   # computed once per quality series (a cache_clear makes a new one)
        per = q.groupby(q.index.to_period("M")).mean().to_dict() if len(q) else {}
        hit = _QUALITY_MONTHS[t] = (q, per)
    return hit[1]


def corporate_action_days(ticker: str, threshold: float = 0.15) -> pd.DatetimeIndex:
    """Days where the quoted price jumps by more than `threshold` and the total-return series does not
    agree with price + cash dividend - usually a spin-off, special dividend or split that the free data
    did not fully adjust (e.g. EXPE on 2011-12-21, the TripAdvisor spin-off)."""
    try:
        df = load(ticker)
    except Exception:  # noqa: BLE001 - unknown or synthetic ticker
        return pd.DatetimeIndex([])
    if not {"close", "dividend", "adj_close"} <= set(df.columns):
        return pd.DatetimeIndex([])
    c, d, a = df["close"], df["dividend"], df["adj_close"]
    price_tr = (c + d) / c.shift(1) - 1
    adj_tr = a / a.shift(1) - 1
    big = price_tr.abs() > threshold
    mismatch = (price_tr - adj_tr).abs() > 0.05
    return df.index[(big & mismatch).fillna(False).to_numpy()]


SPINOFF_SHARE = 0.15


_SPINOFF_MEMO: dict = {}   # (ticker, threshold) -> (the price frame it was computed from, the days)


def spinoff_days(ticker: str, threshold: float = SPINOFF_SHARE) -> pd.DatetimeIndex:
    """spinoff_days_uncached, computed once per loaded price frame (a reload or cache_clear recomputes it)."""
    try:
        df = load(ticker)
    except Exception:  # noqa: BLE001 - unknown or synthetic ticker
        return spinoff_days_uncached(ticker, threshold)
    hit = _SPINOFF_MEMO.get((ticker, threshold))
    if hit is None or hit[0] is not df:
        hit = _SPINOFF_MEMO[(ticker, threshold)] = (df, spinoff_days_uncached(ticker, threshold))
    return hit[1]


def spinoff_days_uncached(ticker: str, threshold: float = SPINOFF_SHARE) -> pd.DatetimeIndex:
    """Ex-dates whose 'dividend' is really a spin-off distribution: a cash-equivalent payout of more than
    `threshold` of the previous close (the data source books the spun-off shares' value as a dividend, e.g.
    MDLZ's $14.17 on 2012-10-02 for Kraft Foods Group), or a payout on a day corporate_action_days flags.
    The value is kept (paid in cash like a dividend); only its label differs."""
    try:
        df = load(ticker)
    except Exception:  # noqa: BLE001 - unknown or synthetic ticker
        return pd.DatetimeIndex([])
    if "dividend" not in df or not len(df):
        return pd.DatetimeIndex([])
    d = df["dividend"].fillna(0.0)
    big = (d / df["close"].shift(1)) > threshold
    flagged = d.index.isin(corporate_action_days(ticker)) & (d > 0).to_numpy()
    return df.index[(big.fillna(False).to_numpy() | flagged)]


def coverage_note(start, end) -> str | None:
    c = coverage(start, end)
    if c.empty:
        return None
    tot = (c["members"]).sum()
    got = (c["with_data"]).sum()
    worst = c.loc[c["coverage"].idxmin()]
    note = (f"Survivorship: {got / tot:.0%} of index member-months in this period have usable price data "
            f"(lowest {worst['coverage']:.0%} in {int(worst['year'])}). The rest are mostly companies that were acquired "
            f"or went bankrupt; free data sources no longer carry them, so the former members that are included "
            f"are almost all companies still trading today and results lean optimistic.")
    gaps = [g for g in _membership_gaps(membership())
            if pd.Timestamp(g.split(" .. ")[1]) >= pd.Timestamp(start) and pd.Timestamp(g.split(" .. ")[0]) <= pd.Timestamp(end)]
    if gaps:
        note += (" Membership snapshots are missing for " + ", ".join(gaps)
                 + "; the last known list is carried forward across those gaps.")
    miss = missing_members(start, end)
    if miss:
        dl = delisted()
        note += (" Biggest missing members (member-months without usable data): "
                 + ", ".join(f"{t} ({n}{'; ' + dl[t]['history'] if dl.get(t, {}).get('history') else ''})" for t, n in miss)
                 + ". " + TIINGO_HINT)
    return note


TIINGO_HINT = ("A free Tiingo API key (repository secret TIINGO_API_KEY, see README 'Delisted former members') lets the "
               "data job download most of them.")


@lru_cache(maxsize=64)
def missing_members(start=None, end=None, n: int = 6) -> tuple[tuple[str, int], ...]:
    """The index members with the most member-months in [start, end] without usable price data (no file, a junk
    or recycled-symbol series, or history lost - e.g. EA), largest first: ((ticker, months), ...)."""
    mem = membership()
    if mem is None:
        return ()
    m = mem
    if start is not None:
        m = m[m.index >= pd.Timestamp(start).to_period("M").to_timestamp()]
    if end is not None:
        m = m[m.index <= pd.Timestamp(end)]
    have = set(available_tickers())
    counts = {}
    for t in m.columns:
        months = m.index[m[t].to_numpy()]
        if not len(months):
            continue
        k = len(months) if t not in have else sum(1 for d in months if not _quality_month(t, d))
        if k:
            counts[t] = k
    # a symbol listed under two names (DUPLICATES) counts once, under the later one
    for old, new in DUPLICATES.items():
        if old in counts and new in counts:
            counts[new] += counts.pop(old)
    return tuple(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:n])


@lru_cache(maxsize=32)
def survivorship(start, end) -> dict | None:
    """Headline figures for a Nasdaq-100 (point-in-time) run over [start, end]: the share of member-months with
    data, the worst year, the biggest missing members, and a rough bias estimate (bias_estimate)."""
    c = coverage(start, end)
    if c.empty:
        return None
    tot, got = float(c["members"].sum()), float(c["with_data"].sum())
    worst = c.loc[c["coverage"].idxmin()]
    out = {"coverage": got / tot if tot else 0.0, "worst_coverage": float(worst["coverage"]),
           "worst_year": int(worst["year"]), "missing": list(missing_members(start, end))}
    out["headline"] = (f"Survivorship: {out['coverage']:.0%} of member-months have data ({out['worst_coverage']:.0%} in "
                       f"{out['worst_year']}) - results are biased upward")
    try:
        out["bias"] = bias_estimate(start, end)
    except Exception:  # noqa: BLE001 - context only
        out["bias"] = None
    return out


BIAS_REFERENCES = (("QQQE", "equal-weight Nasdaq-100 fund QQQE"), ("QQQ", "Nasdaq-100 fund QQQ (cap-weighted)"))


@lru_cache(maxsize=32)
def bias_estimate(start, end) -> dict | None:
    """Context for the survivorship bias: an equal-weight, monthly rebalanced portfolio of the point-in-time members
    WITH data (what an index strategy here can choose from) against a real fund holding the whole index over the
    same months - QQQE (equal weight, from 2012) where the period allows, else QQQ (cap weight, so the gap also
    includes equal- vs cap-weighting). Total returns from month-end adjusted closes."""
    mem = membership()
    if mem is None:
        return None
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    for ref, label in BIAS_REFERENCES:
        try:
            r_px = load(ref)["adj_close"]
        except DataError:
            continue
        s2 = max(s, r_px.index[0])
        if (e - s2).days < 365:
            continue
        months = pd.period_range(s2.to_period("M") + 1, e.to_period("M"), freq="M")
        if len(months) < 12:
            continue
        me_ref = r_px.groupby(r_px.index.to_period("M")).last()
        names = [t for t in nasdaq100_ever() if t in mem.columns]
        px = {t: load(t)["adj_close"] for t in names}
        me = pd.DataFrame({t: p.groupby(p.index.to_period("M")).last() for t, p in px.items()})
        rets = []
        ref_rets = []
        for mth in months:
            prev = mth - 1
            snap = mem[mem.index <= prev.to_timestamp()]
            if snap.empty or prev not in me.index or mth not in me.index:
                continue
            row = snap.iloc[-1]
            ok = [t for t in names if row.get(t, False) and _quality_month(t, prev.to_timestamp())
                  and np.isfinite(me.at[prev, t]) and np.isfinite(me.at[mth, t])]
            if not ok or prev not in me_ref.index or mth not in me_ref.index:
                continue
            rets.append(float(np.mean([me.at[mth, t] / me.at[prev, t] - 1 for t in ok])))
            ref_rets.append(float(me_ref[mth] / me_ref[prev] - 1))
        if len(rets) < 12:
            continue
        yrs = len(rets) / 12.0
        a = float(np.prod(1 + np.array(rets)) ** (1 / yrs) - 1)
        b = float(np.prod(1 + np.array(ref_rets)) ** (1 / yrs) - 1)
        first = months[0].to_timestamp()
        return {"reference": ref, "label": label, "from": str(first.date()), "to": str(e.date()), "months": len(rets),
                "covered_cagr": a, "reference_cagr": b, "gap": a - b,
                "text": (f"Context: an equal-weight portfolio of the members that have data returned {a:.1%}/yr from "
                         f"{first:%Y-%m} to {e:%Y-%m}, against {b:.1%}/yr for the {label} holding every member "
                         f"({a - b:+.1%}/yr; part of that gap is survivorship bias"
                         + (", part is equal- vs cap-weighting" if ref == "QQQ" else ", part fund costs and tracking")
                         + ").")}
    return None


@lru_cache(maxsize=1)
def membership() -> pd.DataFrame | None:
    """Monthly point-in-time Nasdaq-100 membership (rows: month start; columns: tickers; bool)."""
    if not MEMBERSHIP_FILE.exists():
        return None
    raw = pd.read_csv(MEMBERSHIP_FILE, dtype=str)
    if raw.empty:
        return None
    # strip funds, never stocks (older universe files listed some large stocks among the ETFs)
    stocks = set(universe_meta().get("stocks", [])) | LARGE_STOCKS
    from .fund_lists import ALL_FUNDS
    bad = NOT_MEMBERS | ((set(etfs()) | set(funds()) | ALL_FUNDS) - stocks)
    rows = {pd.Period(m, "M").to_timestamp(): set(t.split()) - bad for m, t in zip(raw["month"], raw["tickers"])}
    for d, syms in rows.items():
        for old, new in DUPLICATES.items():
            if old in syms and new in syms:
                syms.discard(old)
    names = sorted(set().union(*rows.values()))
    df = pd.DataFrame(False, index=sorted(rows), columns=names)
    for d, syms in rows.items():
        df.loc[d, list(syms)] = True
    # a name missing from one snapshot but present before and after is a transcription slip, not a
    # removal and re-addition
    v = df.to_numpy(copy=True)
    blip = ~v[1:-1] & v[:-2] & v[2:]
    v[1:-1] |= blip
    return pd.DataFrame(v, index=df.index, columns=df.columns)


def nasdaq100_ever() -> list[str]:
    """Every ticker that has been a Nasdaq-100 member (in the membership history) and has price data.

    A symbol from the membership history whose price file never stands for the member on any member date
    (a recycled ticker now used by a small company, a junk series with no trading) is left out: it could
    never be selected, and its data must not affect the run (e.g. an 'opens never quoted' check)."""
    return list(_nasdaq100_ever())


@lru_cache(maxsize=1)
def _nasdaq100_ever() -> tuple[str, ...]:
    have = set(available_tickers())
    mem = membership()
    cur = set(nasdaq100())
    names = set(cur)
    if mem is not None:
        names |= set(mem.columns)
    out = []
    for t in sorted(n for n in names if n in have):
        if t in cur or mem is None:
            out.append(t)
            continue
        try:
            idx = load(t).index
        except DataError:
            continue
        m, _ = member_mask([t], idx)
        if m.any():
            out.append(t)
    return tuple(out)


def member_mask(tickers: list[str], index: pd.DatetimeIndex) -> tuple[np.ndarray, pd.Timestamp | None]:
    """(T x N) bool: was ticker a Nasdaq-100 member on each date?

    Only dates covered by point-in-time snapshots can have members: before the first snapshot nobody
    is treated as a member (using a later list there would be pure survivorship bias). The quality
    and company-identity checks apply on every date."""
    mem = membership()
    cur = set(nasdaq100())
    out = np.zeros((len(index), len(tickers)), bool)
    tickers = list(tickers)
    if mem is None:
        out[:, [j for j, t in enumerate(tickers) if t in cur]] = True
        return out, None
    snap = mem.reindex(columns=tickers, fill_value=False)
    first = snap.index[0]
    # each day uses the latest snapshot at or before it; the current list covers the months after the last snapshot
    aligned = snap.reindex(index.union(snap.index)).ffill().reindex(index).fillna(False).astype(bool).to_numpy()
    out[:] = aligned
    out[index < first] = False
    last = snap.index[-1]
    after = index >= last + pd.offsets.MonthBegin(1)
    if after.any():
        out[after] = np.array([t in cur for t in tickers])
    # never treat a junk or recycled-ticker series as the index member
    for j, t in enumerate(tickers):
        if t in IDENTITY_FROM:
            out[:, j] &= np.asarray(index >= pd.Timestamp(IDENTITY_FROM[t]))
        q = quality(t)
        out[:, j] &= q.reindex(index).fillna(False).to_numpy(dtype=bool) if not q.empty else False
    return out, first


@lru_cache(maxsize=None)
def load(ticker: str) -> pd.DataFrame:
    """Daily bars for `ticker`.

    open/high/low/close/volume are split-adjusted but NOT dividend-adjusted (prices as quoted, like a
    chart). `dividend` is the cash dividend per share on its ex-date; `adj_close` is the
    total-return (dividend-reinvested) close; `tr` is the total-return index (adj_close / first close).
    """
    t = canonical(ticker)
    path = PRICES / f"{t}.csv"
    if not path.exists() and not fetch_on_demand(t):
        raise DataError(unknown_ticker_message(t))
    raw = pd.read_csv(path, parse_dates=["date"], index_col="date").sort_index()
    raw = raw[~raw.index.duplicated(keep="last")]
    raw = raw[(raw["close"] > 0) & raw["close"].notna()]
    if raw.empty:
        # a header-only (or all-invalid) file is treated as missing data, never as an empty series that later
        # code would index into
        raise DataError(f"No price data for {t}: its price file ({path.name}) has no valid rows (empty, or no positive "
                        "closes). Re-run the 'Fetch price data' workflow (or delete the file so it is downloaded again).")
    raw, _ = reconcile_actions(t, raw)
    raw, repaired = repair_bars(t, raw)
    df = pd.DataFrame(index=raw.index)
    opn = raw["open"].where(raw["open"] > 0, raw["close"]).fillna(raw["close"])
    low = raw["low"].where(raw["low"] > 0)
    df["open"] = opn
    df["high"] = pd.concat([raw["high"], opn, raw["close"]], axis=1).max(axis=1)
    df["low"] = pd.concat([low, opn, raw["close"]], axis=1).min(axis=1)
    df["close"] = raw["close"]
    df["volume"] = raw["volume"].fillna(0) if "volume" in raw else 0.0
    df["dividend"] = raw["dividend"].fillna(0.0) if "dividend" in raw else 0.0
    adj = raw["adj_close"] if "adj_close" in raw else raw["close"]
    df["adj_close"] = adj.where(adj > 0, raw["close"]).ffill()
    df["quote_close"] = df["close"]
    # split ratio on its ex-date (1.0 = none), after reconcile_actions: see split_factor_after / as-traded units
    sp = pd.to_numeric(raw["split"], errors="coerce").fillna(0.0) if "split" in raw else pd.Series(0.0, index=raw.index)
    df["split"] = sp.where((sp > 0) & ((sp - 1).abs() > 1e-9), 1.0)
    if t.endswith("SIM") and len(df):
        # simulated series are built from sources with other calendars (Fama-French, World Bank,
        # FRED): keep only NYSE sessions so month-ends line up with every real ticker. Levels are
        # total-return indexes, so a dropped day's return simply rolls into the next session.
        from . import calendar as _cal
        # (the modern holiday rules only from 1972: earlier NYSE calendars differed - Saturday sessions,
        # fixed-date holidays - so older rows are kept as they are)
        keep = np.array([d.year < 1972 or _cal.is_session(d) for d in df.index])
        df = df[keep]
    # opening prices that were never quoted: missing, or a flat bar (open = high = low = close), as
    # for mutual funds, simulated series and very old index data. Such an "open" is really the close.
    flat = (raw["open"] == raw["close"]) & (raw["high"] == raw["low"]) & (raw["high"] == raw["close"])
    df["open_ok"] = (raw["open"] > 0).fillna(False) & ~flat.fillna(False)
    # old records often carry the previous close as the "open": where that happens on most days of a
    # rolling quarter, the open was never really quoted
    stale = (raw["open"] - raw["close"].shift(1)).abs() <= 1e-9 * raw["close"].abs().clip(lower=1)
    # causal: judged on the quarter up to the day before (so it never depends on later bars)
    df["open_ok"] &= ~(stale.astype(float).rolling(63, min_periods=20).mean().shift(1).fillna(0) > 0.5)
    if repaired.any():
        # a repaired open is an estimate, not a quote: no fills at it
        df.loc[repaired.reindex(df.index, fill_value=False).to_numpy(), "open_ok"] = False
    return df


# ------------------------------------------------------------------ corporate actions
#
# Yahoo sometimes books one event twice or in inconsistent units on its ex-date, e.g.
#  - DHR 2016-07-05 (Fortive spin-off): a $24.56 "dividend" (the FTV shares' value per DHR share - Danaher's own
#    figure) AND a 1.319 "split" (the same spin-off as a price ratio), so a holder gained ~39% overnight;
#  - EXPE 2011-12-21 (TripAdvisor spin-off after a 1-for-2 reverse split) and TMUS 2013-05-01 (MetroPCS: 1-for-2
#    reverse split plus $4.06 cash per pre-split share): the payout is per PRE-split share while the prices are
#    on the post-split basis, so the payout is counted at half its value.
# Yahoo's adj_close is no independent check on these days: it is derived from the same two fields
# (adj ratio = close / (prev_close - dividend) on the split-adjusted basis, verified on all three), so it
# double-counts too (DHR +61%). reconcile_actions therefore re-reads the event: where the day's split and
# payout together disagree with the adjusted close by more than CA_TOLERANCE, it tries the other readings
#   "per_presplit"  the payout is per share before that day's split (divide it by the ratio)
#   "no_split"      the "split" is the spin-off booked as a price ratio (not a whole-number-ish ratio): drop it and
#                   keep the payout (earlier prices go back on their real, as-traded basis)
#   "no_payout"     the split alone stands for the spin-off (drop the payout)
# (each with the payout also scaled by later splits, which Yahoo applies inconsistently to such payouts) and
# keeps the one whose one-day total return is closest to the market's (SPY) that day. adj_close is then
# rescaled before the day so its total return agrees with the reconciled one. CA_FIXES records every change.

CA_TOLERANCE = 0.02
CA_MAX_GAP = 0.10          # a reconciled day must end within 10% (log) of the market's move that day
CA_FIXES: dict[str, pd.DataFrame] = {}
# days where the engine's (close + payout) / previous close legitimately differs from Yahoo's adjusted close by
# more than CA_TOLERANCE: Yahoo's adjustment close / (prev_close - payout) overstates the move for a payout this
# large, and the cash accounting is right (checked by tests/test_data_integrity.py)
CA_WHITELIST = {
    ("BKR", "2017-07-05"): "Baker Hughes / GE merger: $17.50 special dividend per share; cash accounting is right",
    ("KDP", "2018-07-10"): "Dr Pepper Snapple / Keurig merger: $103.75 special dividend per share; cash accounting is right",
    ("VIP", "2019-12-27"): "VEON ADR: a $48.31 payout on a thin (~4,000 shares/day) series; recorded as reported",
    ("HANS", "1990-11-08"): "Hansen Natural 1990: sub-cent prices and a $0.0026 payout; too coarse to reconcile",
    ("MNST", "1990-11-08"): "Monster Beverage (ex-Hansen) 1990: sub-cent prices and a $0.0026 payout; too coarse to reconcile",
}


def _whole_ratio(r: float) -> bool:
    """A split ratio like 2, 3/2, 1/2, 7 or 1/15: p/q (or its inverse) with q <= 10 within 0.1%."""
    for x in (r, 1.0 / r):
        for q in range(1, 11):
            p = round(x * q)
            if p >= 1 and abs(p / q - x) <= 1e-3 * x:
                return True
    return False


@lru_cache(maxsize=1)
def _market_day_returns() -> pd.Series:
    """SPY close-to-close returns (the reference market move for reconcile_actions), read from the file directly."""
    raw = _raw_file("SPY")
    if raw is None or raw.empty:
        return pd.Series(dtype=float)
    return raw["close"].pct_change()


def reconcile_actions(t: str, raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconcile days whose split and payout conflict (see the comment above). Returns (bars, log).

    Only the event day's payout and the basis of the bars BEFORE it change; a day's value never depends on
    later days except through a later event's rescaling of the price basis, which, like any split adjustment,
    leaves every return unchanged."""
    cols = ["date", "reading", "split_was", "split_now", "payout_was", "payout_now", "return_was", "return_now",
            "adj_close_return", "market_return"]
    empty = pd.DataFrame(columns=cols)
    if raw.empty or not {"close", "adj_close", "dividend", "split"} <= set(raw.columns):
        CA_FIXES[t] = empty
        return raw, empty
    c = raw["close"].astype(float)
    d = pd.to_numeric(raw["dividend"], errors="coerce").fillna(0.0)
    sp = pd.to_numeric(raw["split"], errors="coerce").fillna(0.0)
    a = pd.to_numeric(raw["adj_close"], errors="coerce")
    eng = (c + d) / c.shift(1)
    adj = a / a.shift(1)
    cand_days = raw.index[((sp > 0) & ((sp - 1).abs() > 1e-9) & (d > 0) & ((eng - adj).abs() > CA_TOLERANCE)).to_numpy()]
    if not len(cand_days):
        CA_FIXES[t] = empty
        return raw, empty
    raw = raw.copy()
    for k in ("open", "high", "low", "close", "adj_close", "volume", "dividend", "split"):
        if k in raw:
            raw[k] = pd.to_numeric(raw[k], errors="coerce").astype(float)
    raw["dividend"] = raw["dividend"].fillna(0.0)
    raw["split"] = raw["split"].fillna(0.0)
    mkt = _market_day_returns() if t != "SPY" else pd.Series(dtype=float)
    log = []
    for day in cand_days:
        i = raw.index.get_loc(day)
        if i == 0:
            continue
        cc, pc = float(raw["close"].iloc[i]), float(raw["close"].iloc[i - 1])
        dv, r = float(raw["dividend"].iloc[i]), float(raw["split"].iloc[i])
        later = sp[(sp.index > day) & (sp > 0) & ((sp - 1).abs() > 1e-9)]
        f_after = float(np.prod(later.to_numpy())) if len(later) else 1.0
        scales = [1.0] + ([1.0 / f_after] if abs(f_after - 1) > 1e-9 else [])
        m = float(mkt.get(day, 0.0)) if len(mkt) and np.isfinite(mkt.get(day, np.nan)) else 0.0
        opts = [("as_reported", 1.0, dv, (cc + dv) / pc)]
        for g in scales:
            opts.append(("per_presplit", 1.0, dv * g / r, (cc + dv * g / r) / pc))
        if not _whole_ratio(r):
            for g in scales:
                opts.append(("no_split", 0.0, dv * g, (cc + dv * g) / (pc * r)))
            opts.append(("no_payout", 1.0, 0.0, cc / pc))
        best = min(opts, key=lambda o: abs(np.log(o[3]) - np.log1p(m)))
        reading, keep_split, pay, tr = best
        if reading == "as_reported":
            continue
        if abs(np.log(tr) - np.log1p(m)) > CA_MAX_GAP:
            # no reading gives a plausible day: leave the data as it is and say so
            log.append((day, "unresolved", r, r, dv, dv, float(eng.iloc[i]) - 1, float(eng.iloc[i]) - 1,
                        float(adj.iloc[i]) - 1, m))
            continue
        before = raw.index < day
        if keep_split == 0.0:
            for k in ("open", "high", "low", "close"):
                if k in raw:
                    raw.loc[before, k] = raw.loc[before, k] * r
            raw.loc[before, "dividend"] = raw.loc[before, "dividend"].fillna(0.0) * r
            if "volume" in raw:
                raw.loc[before, "volume"] = raw.loc[before, "volume"] / r
            raw.iloc[i, raw.columns.get_loc("split")] = 0.0
        raw.iloc[i, raw.columns.get_loc("dividend")] = pay
        a_prev, a_now = float(raw["adj_close"].iloc[i - 1]), float(raw["adj_close"].iloc[i])
        if np.isfinite(a_prev) and a_prev > 0 and np.isfinite(a_now):
            raw.loc[before, "adj_close"] = raw.loc[before, "adj_close"] * (a_now / a_prev) / tr
        log.append((day, reading, r, r if keep_split else 1.0, dv, pay, float(eng.iloc[i]) - 1, tr - 1,
                    float(adj.iloc[i]) - 1, m))
    out = pd.DataFrame(log, columns=cols)
    CA_FIXES[t] = out
    return raw, out


@lru_cache(maxsize=None)
def corporate_action_fixes(ticker: str) -> pd.DataFrame:
    """The days reconcile_actions re-read for `ticker` (see the comment above reconcile_actions)."""
    t = canonical(ticker)
    raw = _raw_file(t)
    if raw is None:
        return pd.DataFrame(columns=["date", "reading"])
    return reconcile_actions(t, raw)[1]


_READING = {"per_presplit": "the payout was per pre-split share",
            "no_split": "the 'split' was the spin-off booked a second time as a price ratio (dropped)",
            "no_payout": "the split alone stands for the spin-off (payout dropped)",
            "unresolved": "no consistent reading found; left as reported - trades held over it may be misstated"}


@lru_cache(maxsize=None)
def distribution_mismatch_days(ticker: str) -> tuple:
    """Days where the file's own figures disagree: the total return from price + reported distribution differs
    from the adjusted close's by more than CA_TOLERANCE, and no reconciliation or whitelist entry explains it.
    Mostly old mutual-fund histories from Yahoo (capital-gains distributions missing or garbled): ((date, gap), ...)."""
    try:
        df = load(ticker)
    except Exception:  # noqa: BLE001
        return ()
    if len(df) < 2 or not {"close", "dividend", "adj_close"} <= set(df.columns):
        return ()
    c, d, a = df["close"], df["dividend"], df["adj_close"]
    diff = ((c + d) / c.shift(1) - a / a.shift(1)).abs()
    bad = diff[(diff > CA_TOLERANCE).fillna(False)]
    return tuple((day, float(v)) for day, v in bad.items() if (canonical(ticker), str(day.date())) not in CA_WHITELIST)


def distribution_note(tickers, start=None, end=None) -> str | None:
    """A warning naming held tickers whose distributions disagree with their adjusted close inside [start, end]."""
    parts = []
    for t in tickers:
        days = [(d, g) for d, g in distribution_mismatch_days(t)
                if (start is None or d >= pd.Timestamp(start)) and (end is None or d <= pd.Timestamp(end))]
        if days:
            d, g = max(days, key=lambda x: x[1])
            parts.append(f"{t} on {len(days)} day(s) (largest {g:.0%} on {d.date()})")
    if not parts:
        return None
    return ("Warning: data quality: the price + distribution figures disagree with the adjusted close for "
            + "; ".join(parts) + ". This is usually a mutual fund whose free (Yahoo) history misreports "
            "capital-gains distributions; returns across those days may be wrong. Prefer the fund's ETF share class "
            "or a long-history series for those dates.")


def corporate_action_note(tickers, start=None, end=None) -> str | None:
    """A note listing the reconciled corporate-action days of `tickers` inside [start, end]."""
    parts = []
    for t in tickers:
        try:
            fx = corporate_action_fixes(t)
        except Exception:  # noqa: BLE001 - synthetic ticker
            continue
        for _, r in fx.iterrows():
            day = pd.Timestamp(r["date"])
            if (start is not None and day < pd.Timestamp(start)) or (end is not None and day > pd.Timestamp(end)):
                continue
            parts.append(f"{t} {day.date()} ({_READING.get(r['reading'], r['reading'])}: one-day total return "
                         f"{r['return_was']:+.1%} as reported, {r['return_now']:+.1%} reconciled)")
    if not parts:
        return None
    return ("Corporate actions: the data books these events inconsistently (a payout and a split for one spin-off, "
            "or a payout per pre-split share), so they were reconciled: " + "; ".join(parts) + ".")


# Bars whose repair needs outside knowledge: {ticker: {date: {field: value}}}. Values are split-adjusted.
BAR_FIXES: dict[str, dict[str, dict[str, float]]] = {}
REPAIRS: dict[str, pd.DataFrame] = {}   # what repair_bars changed, per ticker (for notes and the Data page)


def _raw_file(t: str) -> pd.DataFrame | None:
    path = PRICES / f"{t}.csv"
    if not path.exists():
        return None
    raw = pd.read_csv(path, parse_dates=["date"], index_col="date").sort_index()
    raw = raw[~raw.index.duplicated(keep="last")]
    return raw[(raw["close"] > 0) & raw["close"].notna()]


def repair_bars(t: str, raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Data sanity pass on opening prices, using only each bar and the ones before it (never later bars,
    so truncating the data cannot change a repaired value). Returns (bars, repaired-open mask).

    - An open outside the bar's own [low, high] (when high/low/close are consistent) is a bad print: it is
      clipped into the range.
    - For a share class of a company with another listed class (SHARE_CLASSES), an open that disagrees with
      the other class's open by more than 4% - after scaling by the two classes' closes that day - where the two classes' price ratio has been
      steady (daily changes under 1% over the previous month), while the
      other class's own gap is ordinary, is replaced by the other class's open scaled the same way; a high
      or low more than 3% outside the other class's scaled range is pulled back to it (GOOG on 2014-04-02,
      the day before the Class C listing, has an open and high about 6% above anything that traded).
    - BAR_FIXES: explicit corrections.
    Every repaired open is flagged (open_ok False: no fills at it)."""
    if raw.empty or not {"open", "high", "low", "close"} <= set(raw.columns):
        return raw, pd.Series(False, index=raw.index)
    raw = raw.copy()
    o, h, lo, c = (raw[k].astype(float) for k in ("open", "high", "low", "close"))
    fixed = pd.Series(False, index=raw.index)
    log = []
    ok_range = (lo > 0) & (h >= lo) & (c <= h * 1.001) & (c >= lo * 0.999) & (o > 0)
    out = ok_range & ((o > h * 1.005) | (o < lo * 0.995))
    if out.any():
        new = o.clip(lower=lo, upper=h)
        for d in raw.index[out.to_numpy()]:
            log.append((d, "open", float(o[d]), float(new[d]), "outside the bar's low-high range"))
        raw.loc[out, "open"] = new[out]
        fixed |= out
    sib = next((g for g in SHARE_CLASSES if t in g), None)
    if sib:
        for other in (g for g in sib if g != t):
            b = _raw_file(other)
            if b is None or b.empty:
                continue
            common = raw.index.intersection(b.index)
            if len(common) < 2:
                continue
            a_ = raw.loc[common]
            b_ = b.loc[common].astype(float)
            ratio = a_["close"] / b_["close"]             # same-day class ratio
            est_o = b_["open"] * ratio
            dev = np.log(a_["open"] / est_o)
            gap_a = np.log(a_["open"] / a_["close"].shift(1)).abs()
            gap_b = np.log(b_["open"] / b_["close"].shift(1)).abs()
            # only where the two classes track each other tightly (the ratio held that day and its daily
            # changes over the previous month were tiny): thinly traded classes legitimately diverge
            dr = np.log(ratio / ratio.shift(1))
            steady = (dr.abs() < 0.01) & (dr.abs().rolling(20, min_periods=10).max().shift(1) < 0.01)
            bad = (dev.abs() > 0.04) & steady & (b_["open"] > 0) & (a_["open"] > 0) & (gap_b < gap_a - 0.02)
            bad = bad.fillna(False)
            for d in common[bad.to_numpy()]:
                log.append((d, "open", float(raw.at[d, "open"]), float(est_o[d]), f"inconsistent with {other}"))
                raw.at[d, "open"] = float(est_o[d])
                hi_est, lo_est = float(b_.at[d, "high"] * ratio[d]), float(b_.at[d, "low"] * ratio[d])
                cc = float(raw.at[d, "close"])
                if raw.at[d, "high"] > hi_est * 1.03:
                    new_h = max(hi_est, cc, float(est_o[d]))
                    log.append((d, "high", float(raw.at[d, "high"]), new_h, f"inconsistent with {other}"))
                    raw.at[d, "high"] = new_h
                if raw.at[d, "low"] < lo_est / 1.03:
                    new_l = min(lo_est, cc, float(est_o[d]))
                    log.append((d, "low", float(raw.at[d, "low"]), new_l, f"inconsistent with {other}"))
                    raw.at[d, "low"] = new_l
            fixed.loc[common[bad.to_numpy()]] = True
    for ds, fields in BAR_FIXES.get(t, {}).items():
        d = pd.Timestamp(ds)
        if d in raw.index:
            for k, v in fields.items():
                log.append((d, k, float(raw.at[d, k]), float(v), "BAR_FIXES"))
                raw.at[d, k] = v
            fixed.loc[d] = fixed.loc[d] or "open" in fields
    REPAIRS[t] = pd.DataFrame(log, columns=["date", "field", "was", "now", "why"])
    return raw, fixed


def open_anomalies(ticker: str, jump: float = 0.08) -> pd.DataFrame:
    """Report (not used by the simulators): opens that look like bad prints judged with hindsight - far
    from both the previous close and the next open while the close barely moved. Judging a bar by the
    next one would leak the future into a backtest, so these are only listed for review; the causal
    repairs are in repair_bars."""
    t = canonical(ticker)
    raw = _raw_file(t)
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["date", "open", "prev_close", "close", "next_open"])
    o, c = raw["open"].astype(float), raw["close"].astype(float)
    pc, no = c.shift(1), o.shift(-1)
    g = np.log(o / pc)
    typical = g.abs().rolling(60, min_periods=20).median().shift(1)
    wild = ((g.abs() > np.maximum(jump, 10 * typical)) & (np.log(o / no).abs() > np.maximum(jump, 10 * typical))
            & (np.log(c / pc).abs() < 0.25 * g.abs()) & (o > 0)).fillna(False)
    rows = raw.index[wild.to_numpy()]
    return pd.DataFrame({"date": rows, "open": o[rows].to_numpy(), "prev_close": pc[rows].to_numpy(),
                         "close": c[rows].to_numpy(), "next_open": no[rows].to_numpy()})


def fetch_on_demand(ticker: str) -> bool:
    """Download a missing ticker's full daily history with yfinance (needs internet; used on your own
    machine - the cloud sandbox relies on the GitHub Action instead). Returns True on success."""
    import os
    if os.environ.get("BACKTESTER_OFFLINE"):
        return False
    try:
        import yfinance as yf
    except ImportError:
        return False
    try:
        df = yf.Ticker(ticker).history(period="max", interval="1d", auto_adjust=False, actions=True)
    except Exception:  # noqa: BLE001 - no internet, unknown symbol, rate limit...
        return False
    if df is None or len(df) < 20:
        return False
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Adj Close": "adj_close",
                            "Volume": "volume", "Dividends": "dividend", "Stock Splits": "split"})
    cols = [c for c in ("open", "high", "low", "close", "adj_close", "volume", "dividend", "split") if c in df]
    df = df[cols][~df.index.duplicated(keep="last")].dropna(subset=["close"])
    df.index.name = "date"
    PRICES.mkdir(parents=True, exist_ok=True)
    df.round(6).to_csv(PRICES / f"{ticker}.csv", float_format="%.6g")
    return True


EXTRA_TICKERS_FILE = DATA / "extra_tickers.txt"
TICKER_SYMBOL_RE = r"\^?[A-Z0-9][A-Z0-9.\-]{0,14}"


def requested_tickers() -> list[str]:
    """The symbols listed in data/extra_tickers.txt ('#' starts a comment)."""
    if not EXTRA_TICKERS_FILE.exists():
        return []
    out = []
    for line in EXTRA_TICKERS_FILE.read_text().splitlines():
        out += [t.strip().upper().lstrip("$") for t in line.split("#", 1)[0].replace(",", " ").split()]
    return list(dict.fromkeys(out))


def request_ticker(ticker: str) -> str:
    """Queue a symbol for the data job: append it to data/extra_tickers.txt (which the 'Fetch price data'
    workflow reads). Returns 'added', 'already requested' or 'in the built-in list' (the broad fund list in
    backtester/fund_lists.py, downloaded in rotating batches: it arrives with one of the next data runs)."""
    import re
    t = canonical(ticker)
    if not re.fullmatch(TICKER_SYMBOL_RE, t):
        raise DataError(f"{ticker!r} is not a ticker symbol.")
    from .fund_lists import ALL_FUNDS
    if t in ALL_FUNDS:
        return "in the built-in list"
    if t in requested_tickers():
        return "already requested"
    EXTRA_TICKERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    text = EXTRA_TICKERS_FILE.read_text() if EXTRA_TICKERS_FILE.exists() else ""
    with EXTRA_TICKERS_FILE.open("a") as f:
        f.write(("" if not text or text.endswith("\n") else "\n") + t + "\n")
    return "added"


def load_many(tickers: list[str]) -> dict[str, pd.DataFrame]:
    return {canonical(t): load(t) for t in tickers}


def total_return_close(ticker: str) -> pd.Series:
    return load(ticker)["adj_close"]


@lru_cache(maxsize=1)
def tbill_rate() -> pd.Series:
    """Annualised 3-month T-bill yield (decimal), daily, forward-filled. FRED DTB3, else Yahoo ^IRX."""
    f = DATA / "macro" / "DTB3.csv"
    s = None
    if f.exists():
        d = pd.read_csv(f, parse_dates=["date"], index_col="date")["value"]
        s = pd.to_numeric(d, errors="coerce").dropna() / 100.0
    irx = PRICES / "^IRX.csv"
    if irx.exists():
        y = pd.read_csv(irx, parse_dates=["date"], index_col="date")["close"] / 100.0
        s = y if s is None else s.combine_first(y)
    if s is None:
        return pd.Series(dtype=float)
    return s.sort_index()


@lru_cache(maxsize=1)
def cpi() -> pd.Series:
    """US CPI, indexed by the date each month's figure was published (about the 15th of the following
    month), so inflation-indexed cash flows and real returns only use CPI that was known at the time.
    CPI-U not seasonally adjusted (CPIAUCNS, from 1913), as cpi_monthly()."""
    s = cpi_monthly()
    if s.empty:
        return s
    s = s.copy()
    s.index = s.index + pd.offsets.MonthBegin(1) + pd.Timedelta(days=14)
    return s


@lru_cache(maxsize=1)
def cpi_monthly() -> pd.Series:
    """US CPI by the month it measures (dated the 1st of that month, not lagged): for reporting inflation over
    calendar periods (December to December), not for decisions. CPI-U not seasonally adjusted (CPIAUCNS, from
    1913: the official inflation figure and what Portfolio Visualizer uses; 1967 is 3.0%, where the seasonally
    adjusted series gives 3.3%), with the seasonally adjusted CPIAUCSL only as a fallback when it is missing."""
    parts = []
    for sid in ("CPIAUCNS", "CPIAUCSL"):
        f = DATA / "macro" / f"{sid}.csv"
        if f.exists():
            d = pd.read_csv(f, parse_dates=["date"], index_col="date")["value"]
            parts.append(pd.to_numeric(d, errors="coerce").dropna())
    if not parts:
        return pd.Series(dtype=float)
    s = parts[0]
    if len(parts) > 1 and len(parts[1]) and parts[1].index[0] < s.index[0]:
        older = parts[1][parts[1].index < s.index[0]]
        if len(older):
            # splice: scale the older series to meet the newer one at the join
            join = parts[1].reindex([s.index[0]]).iloc[0] if s.index[0] in parts[1].index else older.iloc[-1]
            s = pd.concat([older * (s.iloc[0] / join), s])
    return s


@lru_cache(maxsize=1)
def factors() -> pd.DataFrame:
    """Daily Fama-French 5 factors + momentum (decimal returns), if downloaded."""
    base = DATA / "factors"
    parts = []
    for fn in ("ff5_daily.csv", "mom_daily.csv"):
        p = base / fn
        if p.exists():
            parts.append(pd.read_csv(p, parse_dates=["date"], index_col="date"))
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, axis=1)
    df.columns = [c.strip() for c in df.columns]
    return df


# Symbol changes (same company, new symbol): mirrors RENAMES in scripts/fetch_data.py. The old symbol's
# price file is a copy of the new one's, so its share counts can come from the new symbol too.
RENAMED = {"FB": "META", "PCLN": "BKNG", "DISCA": "WBD", "RIMM": "BB", "MYL": "VTRS", "NLOK": "GEN",
           "SYMC": "GEN", "JDSU": "VIAV", "JDSUD": "VIAV", "HANS": "MNST", "CTRP": "TCOM", "WLTW": "WTW",
           "UAUA": "UAL", "KFT": "MDLZ", "KLA": "KLAC", "ERICY": "ERIC", "WFMI": "WFM", "LINTA": "QRTEA"}

# Listed share classes of one company. Yahoo reports the whole company's share count for each class,
# so a class's market cap is approximated as the company's divided by the number of listed classes
# (the index itself weights each class by its own share count, which free data does not give).
SHARE_CLASSES = [("GOOG", "GOOGL"), ("FOX", "FOXA"), ("LBTYA", "LBTYK"), ("BATRA", "BATRK"), ("LILA", "LILAK"),
                 ("NWS", "NWSA"), ("DISCA", "DISCK"), ("LMCA", "LMCK")]


SHARES_SEC = DATA / "shares_sec"
SEC_GAP_DAYS = 120


def _read_shares(folder: Path, sym: str) -> pd.Series:
    p = folder / f"{sym}.csv"
    if not p.exists():
        return pd.Series(dtype=float)
    s = pd.read_csv(p, parse_dates=["date"], index_col="date")["shares"]
    s = pd.to_numeric(s, errors="coerce").dropna()
    return s[~s.index.duplicated(keep="last")].sort_index()


def _shares_from(folder: Path, t: str) -> pd.Series:
    s = _read_shares(folder, t)
    if t in RENAMED:
        alt = _read_shares(folder, RENAMED[t])
        if len(alt):
            s = s.combine_first(alt) if len(s) else alt
    return s


@lru_cache(maxsize=None)
def shares_outstanding(ticker: str) -> pd.Series:
    """Shares outstanding as reported (point in time: each count on the date it became known, in the share units
    of that date - NOT adjusted for later splits). See shares_adjusted for the split-consistent count.

    Two sources: Yahoo (data/shares, mostly from late 2015) and SEC EDGAR XBRL company facts (data/shares_sec,
    from ~2009, dated by the filing date; scripts/fetch_data.py fetch_sec_shares). Yahoo's counts are used where
    they exist; an SEC count fills in before Yahoo's first count and wherever Yahoo has none in the previous
    SEC_GAP_DAYS days."""
    t = canonical(ticker)
    y = _shares_from(DATA / "shares", t)
    sec = _shares_from(SHARES_SEC, t)
    if sec.empty:
        return y
    if y.empty:
        return sec
    yd = y.index.to_numpy()
    keep = []
    for d in sec.index:
        k = int(np.searchsorted(yd, d.to_datetime64(), side="right")) - 1   # latest Yahoo count on or before d
        keep.append(k < 0 or (d - y.index[k]).days > SEC_GAP_DAYS)
    return pd.concat([y, sec[np.array(keep, bool)]]).sort_index().pipe(lambda s: s[~s.index.duplicated(keep="first")])


@lru_cache(maxsize=None)
def splits(ticker: str) -> pd.Series:
    """Stock splits on their ex-dates (ratio, e.g. 4.0 for 4-for-1), from the price file's split column after
    reconcile_actions (a spin-off also booked as a 'split' is not a split)."""
    t = canonical(ticker)
    raw = _raw_file(t)
    if raw is None or "split" not in raw:
        return pd.Series(dtype=float)
    raw = reconcile_actions(t, raw)[0]
    s = pd.to_numeric(raw["split"], errors="coerce").fillna(0.0)
    s = s[(s > 0) & ((s - 1).abs() > 1e-9)]
    return s[~s.index.duplicated(keep="last")].sort_index()


def split_factor_after(ticker: str, dates) -> np.ndarray:
    """Product of the split ratios with an ex-date after each date: a quantity dated d in the share units of
    that day times this is in today's units (and a split-adjusted price times it is the price as quoted).

    As-traded units: shares as traded on d = split-adjusted shares / this factor, and the price as traded =
    split-adjusted price x this factor. It uses later splits, but only as a unit conversion: the split-adjusted
    series itself is defined by those later splits, while the as-traded price and share count are exactly what
    a trader saw on d (known then), and no return, signal or equity value changes with it. Per-share
    commissions, whole-share sizing and reported share counts use as-traded units."""
    dates = pd.DatetimeIndex(dates)
    sp = splits(ticker)
    out = np.ones(len(dates))
    for d, r in sp.items():
        out[dates < d] *= float(r)
    return out


def as_traded_factor(ticker: str, dates, df: pd.DataFrame | None = None) -> np.ndarray:
    """split_factor_after for the simulators: from the price file's (reconciled) splits when the ticker has a
    file, else from the frame's own split column (synthetic data), else 1. Taken from the file, not the frame,
    so a frame cut at some date (the lookahead tests) keeps the same units."""
    dates = pd.DatetimeIndex(dates)
    t = canonical(ticker)
    if (PRICES / f"{t}.csv").exists():
        return split_factor_after(t, dates)
    out = np.ones(len(dates))
    if df is not None and "split" in df:
        sp = pd.to_numeric(df["split"], errors="coerce").fillna(0.0)
        for d, r in sp[(sp > 0) & ((sp - 1).abs() > 1e-9)].items():
            out[dates < d] *= float(r)
    return out


def quoted_close(ticker: str) -> pd.Series:
    """The close as actually quoted on each day (not adjusted for later splits or for dividends)."""
    c = load(ticker)["close"]
    return c * split_factor_after(ticker, c.index)


@lru_cache(maxsize=None)
def shares_adjusted(ticker: str) -> pd.Series:
    """Shares outstanding in today's share units (reported count x later splits), cleaned causally:

    - a count reported after a split but still in pre-split units (Yahoo lags a few weeks, e.g. AAPL and
      TSLA after their 2020 splits) is put on the right basis when it is off from the previous count by
      about a split ratio that took effect within ~6 months;
    - a count that jumps more than 15% from the accepted one is only used once a later report confirms it
      (from that report's date on), so isolated junk values never are.
    Each value is used from the trading day after its date (it is known once reported)."""
    raw = shares_outstanding(ticker)
    raw = raw[raw > 0]
    if raw.empty:
        return pd.Series(dtype=float)
    adj = raw.to_numpy(dtype=float) * split_factor_after(ticker, raw.index)
    sp = splits(ticker)
    out_d, out_v = [], []
    last = None
    pending = None
    for d, v in zip(raw.index, adj):
        if last is not None and abs(np.log(v / last)) > 0.2:
            near = [float(r) for sd, r in sp.items() if abs((sd - d).days) <= 190]
            for r in near:
                for k in (r, 1.0 / r):
                    if abs(np.log(v * k / last)) < 0.1:
                        v *= k
                        break
                else:
                    continue
                break
        if last is None:
            last = v
        elif abs(np.log(v / last)) > np.log(1.15):
            if pending is not None and abs(np.log(v / pending)) <= np.log(1.15):
                last, pending = v, None      # confirmed by a second report
            else:
                pending = v
                continue
        else:
            last, pending = v, None
        out_d.append(d)
        out_v.append(last)
    s = pd.Series(out_v, index=pd.DatetimeIndex(out_d), dtype=float)
    return s[~s.index.duplicated(keep="last")]


def _class_divisor(ticker: str) -> int:
    t = canonical(ticker)
    for grp in SHARE_CLASSES:
        if t in grp:
            return max(1, sum(1 for g in grp if (PRICES / f"{g}.csv").exists()))
    return 1


@lru_cache(maxsize=None)
def market_cap(ticker: str) -> pd.Series:
    """Daily market capitalisation in dollars: the close as quoted that day x the shares outstanding last
    reported before that day (both on the same split basis). NaN before the first share count, and where
    the result is implausible against the traded dollar volume (see MCAP_TURNOVER). A total-return
    (dividend-adjusted) price is never used: its level is not a price anyone paid."""
    t = canonical(ticker)
    try:
        df = load(t)
    except DataError:
        return pd.Series(dtype=float)
    idx = df.index
    sh = shares_adjusted(t)
    if sh.empty or not len(idx):
        return pd.Series(np.nan, index=idx)
    # point in time: a count dated d is known from the next session
    known = sh.copy()
    known.index = known.index + pd.Timedelta(days=1)
    known = known[~known.index.duplicated(keep="last")]
    s = known.reindex(idx.union(known.index)).ffill().reindex(idx)
    mc = df["close"] * s / _class_divisor(t)
    if t in IDENTITY_FROM:
        mc[idx < pd.Timestamp(IDENTITY_FROM[t])] = np.nan
    # plausibility: daily turnover (dollar volume / market cap) of a listed stock is far inside
    # [0.001%, 100%]; outside it the share count is in the wrong units or belongs to another company
    dv = (df["close"] * df["volume"]).rolling(60, min_periods=20).median().shift(1)
    turn = dv / mc
    lo, hi = MCAP_TURNOVER
    bad = (turn > hi) | (turn < lo)
    return mc.where(~bad.fillna(False))


MCAP_TURNOVER = (1e-5, 1.0)
# ranking or weighting an index universe by market cap needs a count for most of its members: with only a few,
# the "top 10" is the top of whichever names have counts (before SEC/Yahoo coverage, 100% SYMC on 2015-06-30)
MCAP_MIN_COVERAGE = 0.8


def mcap_coverage(eligible: np.ndarray, tickers: list[str], index: pd.DatetimeIndex) -> np.ndarray:
    """Per day: the share of the eligible names (T x N bool, e.g. index members with a price) with a market cap."""
    known = np.column_stack([market_cap(t).reindex(index).notna().to_numpy() for t in tickers]) if tickers else \
        np.zeros((len(index), 0), bool)
    n = eligible.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (eligible & known).sum(axis=1) / np.maximum(n, 1), 0.0)


def mcap_start(eligible: np.ndarray, tickers: list[str], index: pd.DatetimeIndex,
               threshold: float = MCAP_MIN_COVERAGE) -> tuple[pd.Timestamp | None, str | None]:
    """(first day market caps cover `threshold` of the eligible names, note). The first day is None when that is
    never reached (the note then says so and the run keeps its start)."""
    if not len(index):
        return None, None
    cov = mcap_coverage(eligible, tickers, index)
    ok = np.flatnonzero(cov >= threshold)
    if not len(ok):
        return None, (f"Market cap: share counts cover at most {cov.max():.0%} of the universe on any day of this "
                      f"period (below {threshold:.0%}), so market-cap rankings pick from a subset throughout.")
    if ok[0] == 0:
        return index[0], None
    d = index[ok[0]]
    return d, (f"Market cap: market-cap data covers too little of the universe before {d.date()} (under {threshold:.0%} "
               f"of the members have a share count; {cov[0]:.0%} on {index[0].date()}), so the test starts on "
               f"{d.date()}.")
MCAP_NOTE = ("Market cap: the close as quoted that day x the shares outstanding last reported before it (Yahoo "
             "share counts from about 2015, SEC EDGAR filings from about 2009, each used from the day after it became "
             "public; dates and tickers without a count have no market cap). Share classes of one company "
             "(GOOG/GOOGL, FOX/FOXA, ...) each count as the company's value divided by the number of listed classes.")


def _membership_gaps(mem) -> list[str]:
    if mem is None or len(mem) < 2:
        return []
    d = pd.Series(mem.index)
    g = d.diff().dt.days
    return [f"{d[i - 1].date()} .. {d[i].date()}" for i in range(1, len(d)) if g[i] > 62]


DELISTED_FILE = DATA / "delisted.json"


@lru_cache(maxsize=1)
def delisted() -> dict:
    """{ticker: {"last_date", "reason", ...}} for tickers whose price history has ended (written by the
    data job, which keeps their last good file)."""
    try:
        return json.loads(DELISTED_FILE.read_text()) if DELISTED_FILE.exists() else {}
    except ValueError:
        return {}


def identity_notes(ticker: str, start=None, end=None) -> list[str]:
    """Warnings when the price file for `ticker` is probably not the company the user means in the
    period (a recycled symbol, a junk series) or when the listing ended within it."""
    t = canonical(ticker)
    if is_sim(t) or t.startswith("^"):
        return []
    try:
        df = load(t)
    except DataError:
        return []
    if df.empty:
        return []
    out = []
    s = pd.Timestamp(start) if start else df.index[0]
    e = pd.Timestamp(end) if end else df.index[-1]
    f0, f1 = df.index[0], df.index[-1]
    mem = membership()
    months = mem.index[mem[t].to_numpy()] if mem is not None and t in mem else pd.DatetimeIndex([])
    if len(months):
        m0, m1 = months[0], months[-1] + pd.offsets.MonthEnd(0)
        span = f"{m0:%Y-%m}..{m1:%Y-%m}"
        # only when the period overlaps the membership: then the user most likely means the member company
        overlap = s <= m1 and e >= m0
        if t in IDENTITY_FROM and s < pd.Timestamp(IDENTITY_FROM[t]):
            out.append(f"Identity: {t}'s price file is a different company before {IDENTITY_FROM[t]} (the symbol was "
                       f"reused); earlier dates do not show the {t} that was in the Nasdaq-100.")
        elif overlap and f0 > m0 + pd.DateOffset(months=2):
            out.append(f"Identity: {t} was a Nasdaq-100 member in {span}, but its price file only starts on "
                       f"{f0.date()}: most likely a later company that reused the symbol, not that member.")
        elif overlap:
            q = quality(t)
            covered = [m for m in months if m >= f0.to_period("M").to_timestamp()]
            if covered and not q.empty:
                per = q.groupby(q.index.to_period("M")).mean()
                bad = sum(1 for m in covered if per.get(m.to_period("M"), 0.0) < 0.5)
                if bad >= 0.5 * len(covered):
                    out.append(f"Identity: during {t}'s Nasdaq-100 membership ({span}) its price file trades like a tiny, "
                               f"illiquid stock - probably a different company that took over the symbol, or junk data. "
                               f"It is not the index member.")
    win = df[(df.index >= s) & (df.index <= e)]
    # only bars with real trading: stale rows (no volume, or the same close carried forward - e.g. the history a
    # data vendor pads in before a US listing, as for FER) would pull the median to zero
    live = (win["volume"] > 0) & (win["close"] != win["close"].shift(1))
    win = win[live.to_numpy()]
    if len(win) >= 20 and not any(n.startswith("Identity:") for n in out):
        dv = float((win["close"] * win["volume"]).median())
        if dv < MIN_DOLLAR_VOLUME:
            out.append(f"Liquidity: {t} trades about ${dv:,.0f} a day in this period (median dollar volume) - a very "
                       f"thin series; check it is the company you mean.")
    info = delisted().get(t)
    if info and info.get("last_date") and pd.Timestamp(info["last_date"]) < e:
        why = f" ({info['reason']})" if info.get("reason") else ""
        out.append(f"Delisted: {t} stopped trading on {info['last_date']}{why}; the data ends there.")
    return out


def data_status() -> dict:
    m = universe_meta()
    mem = membership()
    return {
        "updated_utc": m.get("updated_utc"),
        "tickers": len(available_tickers()),
        "nasdaq100_current": len(nasdaq100()),
        "former_members_with_data": len(m.get("former_members", [])),
        "former_members_missing": len(m.get("former_members_missing_data", [])),
        "delisted": len(delisted()),
        "membership_from": str(mem.index[0].date()) if mem is not None else None,
        "membership_to": str(mem.index[-1].date()) if mem is not None else None,
        "membership_gaps": _membership_gaps(mem),
        "has_tbill": not tbill_rate().empty,
        "has_cpi": not cpi().empty,
        "has_factors": not factors().empty,
    }


# ------------------------------------------------------------------ monthly-stepped segments and holes

STEP_MAX_MOVES = 3        # a month whose series moves on at most this many sessions ...
STEP_MIN_JUMP = 1e-4      # ... by at least this much (0.01%) on one of them is a monthly step
STEP_MIN_MONTHS = 3       # a stepped stretch has at least this many such months


def _stepped_months(px: pd.Series) -> pd.Series:
    """Per calendar month: "j" (moves on 1-3 sessions only: a monthly step), "f" (flat or a constant daily accrual)
    or "d" (moves on most days: a daily series). A day "moves" when its return per calendar day differs from the
    month's median rate, so a bond model that accrues its coupon daily and re-prices on one day a month (LQDSIM
    before 1986) counts as stepped, and a constant T-bill accrual counts as flat."""
    px = pd.to_numeric(px, errors="coerce").dropna()
    px = px[px > 0]
    if len(px) < 3:
        return pd.Series(dtype=object)
    r = px.pct_change().iloc[1:]
    gap = pd.Series(px.index, index=px.index).diff().dt.days.iloc[1:].clip(lower=1)
    rate = r / gap
    per = r.index.to_period("M")
    med = rate.groupby(per).transform("median")
    dev = (rate - med).abs() * gap
    moved = dev > 1e-9 + 1e-3 * (med.abs() * gap)
    g = pd.DataFrame({"moved": moved, "dev": dev.where(moved, 0.0)}).groupby(per)
    cnt, big = g["moved"].sum(), g["dev"].max()
    return pd.Series(np.where(cnt > STEP_MAX_MOVES, "d", np.where(big >= STEP_MIN_JUMP, "j", "f")), index=cnt.index)


def stepped_ranges_of(px: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Date ranges (first day of the first stepped month, last day of the last) where a price / total-return
    series changes only about once a month (monthly source data spread over daily sessions)."""
    st = _stepped_months(px)
    out, run = [], []

    def close(run):
        js = [m for m, s in run if s == "j"]
        if len(js) >= STEP_MIN_MONTHS:
            out.append((js[0].start_time.normalize(), js[-1].end_time.normalize()))
    for m, s in st.items():
        if s == "d":
            close(run)
            run = []
        else:
            run.append((m, s))
    close(run)
    return out


@lru_cache(maxsize=None)
def stepped_ranges(ticker: str) -> tuple[tuple[pd.Timestamp, pd.Timestamp], ...]:
    """Where `ticker`'s total-return series moves in monthly steps (EFASIM before 1990, EEMSIM before 2003,
    VNQSIM before 2004, DBCSIM before 2006, LQDSIM's monthly-yield years...): daily statistics, daily
    correlations and daily regressions are meaningless there. Detected from the data, so any series built
    from a monthly source is caught."""
    try:
        df = load(ticker)
    except (DataError, FileNotFoundError):
        return ()
    return tuple(stepped_ranges_of(df["adj_close"]))


def stepped_in(tickers, start=None, end=None) -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]:
    """{ticker: its stepped ranges clipped to [start, end]} for the tickers that have any there."""
    s = pd.Timestamp(start) if start is not None else pd.Timestamp.min
    e = pd.Timestamp(end) if end is not None else pd.Timestamp.max
    out = {}
    for t in dict.fromkeys(canonical(x) for x in tickers):
        rng = [(max(a, s), min(b, e)) for a, b in stepped_ranges(t) if a <= e and b >= s]
        if rng:
            out[t] = rng
    return out


def stepped_text(found: dict) -> str:
    return "; ".join(f"{t} " + ", ".join(f"{a:%Y-%m}..{b:%Y-%m}" for a, b in r) for t, r in found.items())


MAX_GAP_DAYS = 10   # business days: a longer hole inside a simulated series is a data error


@lru_cache(maxsize=None)
def data_gaps(ticker: str) -> tuple[tuple[pd.Timestamp, pd.Timestamp, int], ...]:
    """Holes of more than MAX_GAP_DAYS business days inside a SIM series (after its first date): (last date
    before, first date after, business days missing). The data build logs these as failures; the loader keeps
    the series and portfolios note the hole (the slice is held in cash through it)."""
    t = canonical(ticker)
    if not is_sim(t):
        return ()
    try:
        idx = load(t).index
    except (DataError, FileNotFoundError):
        return ()
    if len(idx) < 2:
        return ()
    a = idx[:-1].values.astype("datetime64[D]")
    b = idx[1:].values.astype("datetime64[D]")
    miss = np.busday_count(a, b) - 1
    return tuple((idx[k], idx[k + 1], int(miss[k])) for k in np.nonzero(miss > MAX_GAP_DAYS)[0])


def gap_notes(tickers, start=None, end=None) -> list[str]:
    s = pd.Timestamp(start) if start is not None else pd.Timestamp.min
    e = pd.Timestamp(end) if end is not None else pd.Timestamp.max
    out = []
    for t in dict.fromkeys(canonical(x) for x in tickers):
        for a, b, n in data_gaps(t):
            if a < e and b > s:
                out.append(f"Data gap: {t} has no data between {a.date()} and {b.date()} ({n} trading days missing, a "
                           "data-build error); holdings of it are carried at the last price, or held in cash if it is "
                           "bought then. Re-run the data workflow to rebuild it.")
    return out
