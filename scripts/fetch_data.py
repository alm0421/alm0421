"""Download daily OHLCV history for the Nasdaq-100 constituents plus QQQ and SPY.

Writes one CSV per ticker to data/prices/<TICKER>.csv with columns:
    date, open, high, low, close, adj_close, volume, dividend, split

`open/high/low/close` are split-adjusted (as Yahoo reports them) but NOT
dividend-adjusted; `adj_close` also includes dividends.  The backtester
derives total-return OHLC from the ratio adj_close / close.

Runs inside GitHub Actions (which has open internet access); see
.github/workflows/fetch-data.yml.
"""
from __future__ import annotations

import io
import re
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from backtester import sources  # noqa: E402  (factor-file parsers shared with the tests)
from backtester import fund_lists  # noqa: E402  (the broad ETF / mutual fund universe)

PRICES = ROOT / "data" / "prices"
UA = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/126.0 Safari/537.36 backtester-data-fetch (github.com/alm0421/alm0421)",
    "Accept-Language": "en-US,en;q=0.9",
}

# Used only if both live constituent sources fail.
FALLBACK_NDX = """
AAPL ABNB ADBE ADI ADP ADSK AEP AMAT AMD AMGN AMZN APP ARM ASML AVGO AXON AZN
BIIB BKNG BKR CCEP CDNS CDW CEG CHTR CMCSA COST CPRT CRWD CSCO CSGP CSX CTAS
CTSH DASH DDOG DXCM EA EXC FANG FAST FTNT GEHC GFS GILD GOOG GOOGL HON IDXX
INTC INTU ISRG KDP KHC KLAC LIN LRCX LULU MAR MCHP MDLZ MELI META MNST MRVL
MSFT MSTR MU NFLX NVDA NXPI ODFL ON ORLY PANW PAYX PCAR PDD PEP PLTR PYPL QCOM
REGN ROP ROST SBUX SHOP SNPS TEAM TMUS TRI TSLA TTD TTWO TXN VRSK VRTX WBD WDAY
XEL ZS
""".split()



TICKER_RE = r"^[A-Z]{1,5}([.-][A-Z])?$"


def tickers_from_html(url: str, headers: dict | None = None) -> list[str]:
    """Find a column of ~100 ticker-shaped strings (including MSFT) in any table on the page."""
    html = requests.get(url, headers=headers or UA, timeout=30).text
    for t in pd.read_html(io.StringIO(html)):
        for col in t.columns:
            vals = t[col].astype(str).str.strip()
            syms = vals[vals.str.match(TICKER_RE)].tolist()
            if 90 <= len(syms) <= 110 and "MSFT" in syms:
                return syms
    raise RuntimeError("constituents table not found")


def ndx_from_nasdaq() -> list[str]:
    r = requests.get("https://api.nasdaq.com/api/quote/list-type/nasdaq100", headers=UA, timeout=20)
    rows = r.json()["data"]["data"]["rows"]
    return [row["symbol"].strip() for row in rows]


SOURCES = (
    ("wikipedia", lambda: tickers_from_html("https://en.wikipedia.org/wiki/Nasdaq-100", WIKI_UA)),
    ("stockanalysis.com", lambda: tickers_from_html("https://stockanalysis.com/list/nasdaq-100-stocks/")),
    ("slickcharts.com", lambda: tickers_from_html("https://www.slickcharts.com/nasdaq100")),
    ("nasdaq.com", ndx_from_nasdaq),
)


def constituents() -> tuple[list[str], str]:
    for name, fn in SOURCES:
        try:
            syms = fn()
            if len(syms) >= 90:
                return sorted(set(s.replace(".", "-") for s in syms)), name
        except Exception as e:  # noqa: BLE001
            print(f"constituent source {name} failed: {e}", file=sys.stderr)
    return sorted(FALLBACK_NDX), "fallback"


def fetch(ticker: str) -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period="max", interval="1d", auto_adjust=False, actions=True)
    if df.empty:
        raise RuntimeError("no data")
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low", "Close": "close",
        "Adj Close": "adj_close", "Volume": "volume",
        "Dividends": "dividend", "Stock Splits": "split",
    })
    df = df[["open", "high", "low", "close", "adj_close", "volume", "dividend", "split"]]
    df = df[~df.index.duplicated(keep="last")].dropna(subset=["close"])
    df.index.name = "date"
    return df


# ETFs and indexes useful for allocation / regime strategies (Composer / Portfolio Visualizer style)
ETFS = """
QQQ SPY DIA IWM MDY VTI VOO QLD TQQQ PSQ SQQQ SSO UPRO SPXL SH SDS SPXU SOXL SOXS TECL
TLT IEF SHY BIL SGOV TMF TMV TBT AGG BND LQD HYG TIP GLD SLV DBC USO UUP
VIXY UVXY SVXY EFA EEM VEA VWO VNQ
XLK XLF XLE XLV XLY XLP XLI XLU XLB XLRE XLC SMH SOXX IBB ARKK
TECS SPXS QID FNGU FNGD TNA TZA LABU LABD UGL TYD TYO VIXM VXX BTAL KMLM DBMF UDOW SDOW UCO SCO NUGT DUST
ERX ERY FAS FAZ URTY SRTY DRN DRV CURE YINN YANG EDC EDZ UBT UST PST TTT UGE SPXU TMF
TLH IEI VGSH VGIT VGLT EDV GOVT SHV USFR TFLO SCHD VIG VYM DVY SPYG SPYV QUAL MTUM USMV VLUE SIZE RSP QQQE
IAU GLDM PDBC GSG CPER KWEB FXI EWJ EWZ EWG EWU VGK VPL VXUS BNDX EMB IJH IJR VB VO VUG VTV IWF IWD IWB IWN IWO
XBI XHB XRT XME KRE KBE ITB JETS TAN ICLN LIT URA BITO IBIT GBTC
VFINX VUSTX VBMFX VFITX VWESX VGSIX VWINX VWELX VTSMX VGTSX VIPSX FSUTX
BTC-USD ETH-USD
SVIX UVIX SVOL TSLL TSLQ NVDL NVDS BITX CONL MSTU USD HIBL HIBS TARK SARK BSV BIV BLV VCIT VCSH VGLT SPTL SPIB
SCHP STIP VTIP SPHQ SPLV XLG QQQM SCHG SCHB SCHX SCHA SCHF SCHE VEU IXUS IEMG ACWI VT VSS VBR VBK VOE VOT VNQI
REET RWR SCHH GDX GDXJ SIL PPLT PALL DBA DBB DBE DBO UNG CORN WEAT BNO COPX TAIL CTA DBMF PFIX RPAR NTSX
VTSAX VTIAX VBTLX VGSLX VIMAX VSMAX VBIAX VWIAX VFIAX FXAIX FSKAX FTIHX SWPPX VTMGX VEMAX VSIAX VGSTX
SCZ EFV AVDV DLS JNK VWEHX SCHP PFORX
BOIL KOLD UTSL WEBL BULZ DPST IGV FXE FXY FXF FXB UDN SPHB NAIL JNUG JDST GLL AGQ ZSL TBF ROM GUSH DRIP
CWEB RETL MIDU DFEN PILL DUSL UYG SPDN SPUU FNGO BNKU TPOR WANT
""".split()
# real funds the simulated series splice in before their ETF existed (build_sims): refreshed every run
SIM_FUNDS = """
VTSMX VGTSX VGSIX VIVAX VIGRX NAESX VISVX VISGX VWESX VWITX VWLTX FNMIX PCRIX VEIEX
EWJ EWU EWG EWC EWA EWQ EWL EWH VCLT MUB EMB VOE VOT BWX IGOV HYG VIPSX TIP
""".split()
ETFS = list(dict.fromkeys(ETFS + SIM_FUNDS))
# large US stocks outside the Nasdaq-100 (stocks, not ETFs: kept separate so they are never mistaken
# for funds, e.g. when ETFs are stripped from index membership)
STOCKS = """
JPM XOM BRK-B JNJ UNH V MA HD PG CVX LLY ABBV MRK KO BAC WFC DIS MCD NKE ORCL CRM IBM GE CAT BA GS MS C T VZ
PFE TMO DHR ABT NEE DUK SO LMT RTX UPS UNP MMM MSTR COIN
""".split()
INDEXES = ["^NDX", "^GSPC", "^VIX", "^IRX", "^TNX", "^DJI", "^RUT", "^SP500TR", "^VIX3M", "^TYX", "^FVX"]
# tickers users ask for: one or more per line in data/extra_tickers.txt ('#' starts a comment)
EXTRA_TICKERS_FILE = ROOT / "data" / "extra_tickers.txt"


def requested_tickers(path: Path = EXTRA_TICKERS_FILE) -> list[str]:
    """The tickers listed in data/extra_tickers.txt (upper-cased, de-duplicated; invalid symbols are skipped
    with a message)."""
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        for tok in line.split("#", 1)[0].replace(",", " ").split():
            t = tok.strip().upper().lstrip("$")
            if re.fullmatch(r"\^?[A-Z0-9][A-Z0-9.\-]{0,14}", t):
                out.append(t)
            else:
                print(f"extra_tickers.txt: skipping {tok!r} (not a ticker symbol)", file=sys.stderr)
    return list(dict.fromkeys(out))


STOCKS += [t for t in fund_lists.BROAD_STOCKS if t not in STOCKS]
REQUESTED = [t for t in requested_tickers() if t not in set(ETFS) | set(STOCKS) | set(INDEXES)]
EXTRA = ETFS + STOCKS + INDEXES + REQUESTED
# The broad universe (backtester/fund_lists.py: ~650 ETFs and ~200 mutual funds), refreshed in rotating
# batches: each run downloads at most BROAD_PER_RUN of them - missing files first, then the ones updated
# longest ago - so a run stays well inside the Action's time limit and polite to Yahoo. A symbol that
# failed is retried after BROAD_RETRY_DAYS (Yahoo doesn't know it, or it was delisted).
BROAD_ETFS = [t for t in fund_lists.BROAD_ETFS if t not in set(EXTRA)]
BROAD_FUNDS = [t for t in fund_lists.MUTUAL_FUNDS if t not in set(EXTRA) | set(BROAD_ETFS)]
BROAD = BROAD_ETFS + BROAD_FUNDS
BROAD_PER_RUN = 450
BROAD_RETRY_DAYS = 30


def broad_batch(broad: list[str], info: dict, failed: dict, today: str, budget: int = BROAD_PER_RUN,
                on_disk=None) -> list[str]:
    """The broad-universe symbols to download this run: at most `budget`, skipping symbols that failed
    within BROAD_RETRY_DAYS, missing files first, then the oldest last update (`info`: {ticker: {"last": date}},
    the previous run's universe.json). Ties keep the list order."""
    on_disk = on_disk if on_disk is not None else (lambda t: (PRICES / f"{t}.csv").exists())
    now = pd.Timestamp(today)
    cand = []
    for i, t in enumerate(broad):
        f = failed.get(t)
        if f and (now - pd.Timestamp(f)).days < BROAD_RETRY_DAYS:
            continue
        last = (info.get(t) or {}).get("last") if on_disk(t) else None
        cand.append((last or "0000-00-00", i, t))
    return [t for *_, t in sorted(cand)[:max(0, budget)]]

# Symbol changes: membership lists use the old symbol, Yahoo keeps history under the new one.
# Only renames where Yahoo's history for the new symbol genuinely continues the same company.
RENAMES = {"FB": "META", "PCLN": "BKNG", "DISCA": "WBD", "RIMM": "BB", "MYL": "VTRS", "NLOK": "GEN",
           "SYMC": "GEN", "JDSU": "VIAV", "JDSUD": "VIAV", "HANS": "MNST", "CTRP": "TCOM", "WLTW": "WTW",
           "UAUA": "UAL", "KFT": "MDLZ", "KLA": "KLAC", "ERICY": "ERIC", "WFMI": "WFM", "LINTA": "QRTEA"}


STOOQ_FAILS = [0]
# per run. Free tiers: Alpha Vantage 25 requests/day; Tiingo 50 requests/hour, 1,000/day and 500 distinct
# symbols/month (so a run stays under the hourly cap and the rest are picked up by the next runs).
KEYED_BUDGET = {"alphavantage": 20, "tiingo": 45}


KEYED_FILE = ROOT / "data" / "delisted_sources.json"
KEYED_OK: set[str] = set(json.loads(KEYED_FILE.read_text())) if KEYED_FILE.exists() else set()


def fetch_delisted_keyed(t: str) -> pd.DataFrame | None:
    """Delisted former members from a keyed source (optional repository secrets TIINGO_API_KEY or
    ALPHAVANTAGE_API_KEY). Both keep acquired/bankrupt companies' histories that Yahoo drops.
    A few names per run are fetched and kept, so coverage grows over successive runs."""
    import os
    key = os.environ.get("TIINGO_API_KEY")
    if key and KEYED_BUDGET["tiingo"] > 0:
        KEYED_BUDGET["tiingo"] -= 1
        try:
            r = requests.get(f"https://api.tiingo.com/tiingo/daily/{t.lower()}/prices",
                             params={"startDate": "1990-01-01", "token": key, "format": "json"}, timeout=60)
            rows = r.json() if r.status_code == 200 else []
            if isinstance(rows, list) and len(rows) > 20:
                df = pd.DataFrame(rows)
                df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
                df = df.set_index("date").sort_index()
                # Tiingo: raw OHLC, split factor and cash dividend; adjClose is split+dividend adjusted
                out = pd.DataFrame({"open": df["open"], "high": df["high"], "low": df["low"], "close": df["close"],
                                    "adj_close": df["adjClose"], "volume": df["volume"],
                                    "dividend": df.get("divCash", 0.0), "split": df.get("splitFactor", 1.0)})
                # split-adjust OHLC/volume like Yahoo's history (dividends stay as paid)
                f = out["split"].replace(0, 1.0)
                cum = f[::-1].cumprod()[::-1].shift(-1).fillna(1.0)
                for col in ("open", "high", "low", "close"):
                    out[col] = out[col] / cum
                out["dividend"] = out["dividend"] / cum
                out["volume"] = out["volume"] * cum
                out.attrs["source"] = "tiingo"
                try:   # the company name, logged so a recycled symbol can be spotted by eye
                    m = requests.get(f"https://api.tiingo.com/tiingo/daily/{t.lower()}", params={"token": key}, timeout=30)
                    out.attrs["name"] = (m.json() or {}).get("name", "") if m.status_code == 200 else ""
                except Exception:  # noqa: BLE001
                    out.attrs["name"] = ""
                return out
        except Exception as e:  # noqa: BLE001
            print(f"tiingo {t}: {e}", file=sys.stderr)
    key = os.environ.get("ALPHAVANTAGE_API_KEY")
    if key and KEYED_BUDGET["alphavantage"] > 0:
        KEYED_BUDGET["alphavantage"] -= 1
        try:
            r = requests.get("https://www.alphavantage.co/query", timeout=60, params={
                "function": "TIME_SERIES_DAILY_ADJUSTED", "symbol": t, "outputsize": "full",
                "datatype": "csv", "apikey": key})
            if r.status_code == 200 and r.text.lower().startswith("timestamp"):
                df = pd.read_csv(io.StringIO(r.text), parse_dates=["timestamp"], index_col="timestamp").sort_index()
                df.index.name = "date"
                coef = df["split_coefficient"].replace(0, 1.0)
                cum = coef[::-1].cumprod()[::-1].shift(-1).fillna(1.0)
                out = pd.DataFrame({"open": df["open"] / cum, "high": df["high"] / cum, "low": df["low"] / cum,
                                    "close": df["close"] / cum, "adj_close": df["adjusted_close"],
                                    "volume": df["volume"] * cum, "dividend": df["dividend_amount"] / cum,
                                    "split": df["split_coefficient"]})
                if len(out) > 20:
                    out.attrs["source"] = "alphavantage"
                    return out
        except Exception as e:  # noqa: BLE001
            print(f"alphavantage {t}: {e}", file=sys.stderr)
    return None


def fetch_stooq(t: str) -> pd.DataFrame | None:
    """Stooq daily history (split-adjusted OHLC, no dividends: adj_close = close, a price-return series).

    Since early 2026 Stooq's CSV endpoint answers "Access denied" without an API key; a free key comes from
    https://stooq.com/q/d/?s=aapl.us&get_apikey (a CAPTCHA). Set it as the STOOQ_API_KEY repository secret.
    Stooq carries few delisted US stocks, so it mostly helps with renamed or thinly covered symbols."""
    import os
    key = os.environ.get("STOOQ_API_KEY")
    if not key or STOOQ_FAILS[0] >= 5:  # no key, or the source is unreachable: stop paying for timeouts
        return None
    try:
        r = requests.get("https://stooq.com/q/d/l/", headers=UA, timeout=15,
                         params={"s": f"{t.lower().replace('-', '.')}.us", "i": "d", "apikey": key})
        if r.status_code != 200 or not r.text.lower().startswith("date"):
            STOOQ_FAILS[0] += 1
            return None
        df = pd.read_csv(io.StringIO(r.text), parse_dates=["Date"], index_col="Date")
        if len(df) < 20:
            return None
        df = df.rename(columns=str.lower)
        if "volume" not in df:
            df["volume"] = 0
        df["adj_close"] = df["close"]
        df["dividend"], df["split"] = 0.0, 0.0
        df.index.name = "date"
        df = df[["open", "high", "low", "close", "adj_close", "volume", "dividend", "split"]]
        df.attrs["source"] = "stooq (split-adjusted, no dividends)"
        return df
    except Exception as e:  # noqa: BLE001
        print(f"stooq {t}: {e}", file=sys.stderr)
        STOOQ_FAILS[0] += 1
        return None


# ------------------------------------------------------------------ never lose a good history

DELISTED_FILE = ROOT / "data" / "delisted.json"
# what we know about why a listing ended (shown to users when a backtest runs past the last date)
DELISTED_REASONS = {
    "EA": "taken private: acquired for $210 a share in cash by a PIF / Silver Lake / Affinity Partners group",
}


HISTORY_MIN_ROWS = 250
HISTORY_UNAVAILABLE = "history unavailable - needs TIINGO_API_KEY (see README: delisted former members)"


def load_delisted() -> dict:
    try:
        return json.loads(DELISTED_FILE.read_text()) if DELISTED_FILE.exists() else {}
    except ValueError:
        return {}


def read_prices(t: str) -> pd.DataFrame | None:
    p = PRICES / f"{t}.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p, parse_dates=["date"], index_col="date").sort_index()
    except Exception:  # noqa: BLE001 - unreadable file: treat as absent, but never delete it
        return None
    return df[~df.index.duplicated(keep="last")]


def merge_history(t: str, new: pd.DataFrame, old: pd.DataFrame | None) -> tuple[pd.DataFrame, str]:
    """The history to save for `t` given a fresh download and the file already on disk.

    A fresh download normally covers the whole history (and Yahoo re-bases adj_close after each new
    dividend), so it replaces the file. But sources sometimes return a truncated history - Yahoo reset EA to
    a single bar when it was taken private in August 2026, and the old job then overwrote years of data.
    Rules:
      - new starts at (or within 5 days of) the old start: use new;
      - new starts later but overlaps old: keep old rows before the overlap, rescaled onto new's split and
        dividend basis when the prices on the overlap agree up to a constant factor; if they don't agree
        (another company, bad data), keep old unchanged;
      - new starts after old ends: append it if it follows on (within 10 days and a 30% move), else keep old.
    Returns (frame, how) where how is "new", "spliced", "appended" or "kept-old: <why>"."""
    cols = ["open", "high", "low", "close", "adj_close", "volume", "dividend", "split"]
    if old is None or len(old) < 2 or not len(new):
        return new, "new"
    for c in cols:
        if c not in old:
            old[c] = 0.0 if c in ("dividend", "split", "volume") else old["close"]
    if new.index[0] <= old.index[0] + pd.Timedelta(days=5):
        return new, "new"
    ov = old.index.intersection(new.index)
    if len(ov) >= 3:
        k = (new.loc[ov, "close"] / old.loc[ov, "close"]).replace([np.inf, -np.inf], np.nan).dropna()
        ka = (new.loc[ov, "adj_close"] / old.loc[ov, "adj_close"]).replace([np.inf, -np.inf], np.nan).dropna()
        if len(k) < 3 or np.log(k).std() > 0.01 or np.log(ka).std() > 0.01:
            return old, "kept-old: the new download disagrees with the saved prices on the overlap"
        kc, kac = float(k.median()), float(ka.median())
        head = old[old.index < new.index[0]].copy()
        for c in ("open", "high", "low", "close", "dividend"):
            head[c] = head[c] * kc
        head["volume"] = head["volume"] / kc
        head["adj_close"] = head["adj_close"] * kac
        return pd.concat([head[cols], new[cols]]), "spliced"
    if new.index[0] > old.index[-1]:
        gap = (new.index[0] - old.index[-1]).days
        jump = abs(float(new["close"].iloc[0]) / float(old["close"].iloc[-1]) - 1)
        if gap <= 10 and jump < 0.3:
            tail = new[cols].copy()
            ratio = float(old["adj_close"].iloc[-1]) / float(old["close"].iloc[-1])
            tail["adj_close"] = tail["close"] * ratio * (tail["adj_close"] / tail["close"]) / (
                float(new["adj_close"].iloc[0]) / float(new["close"].iloc[0]))
            return pd.concat([old[cols], tail]), "appended"
    return old, "kept-old: the new download does not connect to the saved history"


def plausible_member_series(t: str, df: pd.DataFrame, months: list[str]) -> str | None:
    """Identity check for a history from a fallback source: None if it can be the Nasdaq-100 member `t`
    (it trades during the membership months like a large Nasdaq stock), else the reason it can't.
    Recycled symbols (a small company that later took the ticker) fail: their data starts after the
    membership, or trades a few thousand dollars a day during it."""
    if df is None or len(df) < 20:
        return "fewer than 20 bars"
    if (df["close"] <= 0).any():
        return "non-positive prices"
    if not months:
        return None
    m = pd.PeriodIndex(months, freq="M")
    per = df.index.to_period("M")
    cover = np.isin(m, per).mean()
    if cover < 0.5:
        return f"covers only {cover:.0%} of the membership months"
    inm = df[np.isin(per, m)]
    dv = float((inm["close"] * inm["volume"]).median())
    if inm["volume"].sum() > 0 and dv < 2_000_000:
        return f"median dollar volume ${dv:,.0f} during membership - not the index member"
    return None


WIKI_API = "https://en.wikipedia.org/w/api.php"
# Wikimedia asks automated clients for a descriptive user agent with a contact URL (browser-like
# agents from cloud IPs get blocked).
WIKI_UA = {"User-Agent": "BacktesterDataBot/1.0 (https://github.com/alm0421/alm0421) python-requests",
           "Accept": "application/json"}
MEMBERSHIP = ROOT / "data" / "ndx_membership.csv"
MACRO = ROOT / "data" / "macro"
FACTORS = ROOT / "data" / "factors"
SHARES = ROOT / "data" / "shares"


def save_prices(t: str, df: pd.DataFrame, precision: str = "%.6g") -> dict:
    df.round(6).to_csv(PRICES / f"{t}.csv", float_format=precision)
    return {"first": str(df.index[0].date()), "last": str(df.index[-1].date()), "rows": len(df)}


def fetch_with_retry(t: str, tries: int = 3) -> pd.DataFrame | None:
    for attempt in range(tries):
        try:
            return fetch(t)
        except Exception as e:  # noqa: BLE001
            print(f"{t}: attempt {attempt + 1} failed: {e}", file=sys.stderr)
            time.sleep(1.5 * (attempt + 1))
    return None


# ------------------------------------------------------------------ point-in-time membership

MEMBERSHIP_PARSER = "4"   # 4: piped-link tickers ([[Apple Inc.|AAPL]]); every month is fetched again
NOT_MEMBERS = {"NDX", "QQQ", "QQQQ", "TQQQ", "SQQQ", "QLD", "QID", "PSQ", "ONEQ", "NASDAQ", "ETF", "US", "USD", "CEO",
               "S", "P", "NQ", "ND", "RIC", "DJIA", "NYSE", "REIT", "II", "III", "IV", "A", "B", "C", "ADR", "ADS", "LLC", "INC"}


def components_section(wikitext: str) -> str:
    """The part of the article that lists current components.

    Many revisions also carry 'Changes in 20XX' lists (dropped companies) and mention ETFs such as
    TQQQ in prose; scraping the whole article mixed those in. Take the text from the components
    heading to the next level-2 heading, and drop history subsections inside it."""
    wikitext = re.sub(r"<!--.*?-->", "", wikitext, flags=re.S)  # e.g. "==Current components==<!-- ...component changes -->"
    m = re.search(r"(?im)^(={2,4})(?!=)(?![^\n]*(?:historical|former|changes|past))[^\n]*?\b(?:components?|constituents|companies)\b[^\n]*$", wikitext)
    if not m:
        return wikitext
    level = len(m.group(1))
    rest = wikitext[m.end():]
    nxt = re.search(rf"(?m)^={{2,{level}}}(?!=)", rest)
    sec = rest[: nxt.start()] if nxt else rest
    # cut "===Historical components===" / "===Changes...===" subsections
    cut = re.search(r"(?im)^===+[^=\n]*\b(?:historical|former|changes?|yearly|past|removed|additions|deletions|annual)\b", sec)
    return sec[: cut.start()] if cut else sec


def wiki_tickers(wikitext: str) -> set[str]:
    """Extract ticker symbols from a revision of the Nasdaq-100 article (formats changed over the years)."""
    wikitext = components_section(wikitext)
    found: set[str] = set()
    for m in re.finditer(r"\{\{\s*(?:NASDAQ|Nasdaq|nasdaq|NasdaqSymbol|NASDAQ link)\s*\|\s*([A-Za-z.]{1,6})\s*[|}]", wikitext):
        found.add(m.group(1).upper())
    for line in wikitext.splitlines():
        s = line.strip()
        if s.startswith(("#", "*")):
            # "([[AAPL]])", "(NASDAQ: AAPL)" and piped links "([[Apple Inc.|AAPL]])" (2006-2008 revisions wrote
            # Apple, Akamai and Flextronics that way; parser 3 missed them)
            s = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]|]*)\]\]", r"\1", s)
            for m in re.finditer(r"\(\s*(?:NASDAQ:\s*|Nasdaq:\s*)?([A-Z]{1,5}(?:\.[A-Z])?)\s*\)", s):
                found.add(m.group(1))
        if s.startswith("|") and not s.startswith(("|-", "|}", "|+")):
            # cells are separated by "||" (older revisions) or " | " (2025+); strip links first
            row = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", s.lstrip("|"))
            for cell in re.split(r"\|\||\s\|\s", row):
                cell = cell.strip()
                if re.fullmatch(r"[A-Z]{1,5}(?:\.[A-Z])?", cell):
                    found.add(cell)
    return found


WIKI_TITLES = ["Nasdaq-100", "List of NASDAQ-100 companies"]  # the list moved to its own article in late 2025


def wiki_revision_at(ts: str, title: str = "Nasdaq-100") -> tuple[int, str, str] | None:
    r = requests.get(WIKI_API, headers=WIKI_UA, timeout=30, params={
        "action": "query", "prop": "revisions", "titles": title, "rvlimit": 1, "rvstart": ts,
        "rvdir": "older", "rvprop": "ids|timestamp|content", "rvslots": "main", "format": "json",
        "formatversion": 2, "maxlag": 5})
    if r.status_code != 200 or "json" not in r.headers.get("content-type", ""):
        raise RuntimeError(f"HTTP {r.status_code} {r.headers.get('content-type')}: {r.text[:200]!r}")
    pages = r.json()["query"]["pages"]
    revs = pages[0].get("revisions") if pages else None
    if not revs:
        return None
    rev = revs[0]
    return rev["revid"], rev["timestamp"], rev["slots"]["main"]["content"]


MEMBERSHIP_LOG: list[str] = []

# Dated component changes ("Historical components of the Nasdaq-100": one wikitable, columns Date | Added
# Ticker | Added Security | Removed Ticker | Removed Security | Reason, newest first, from February 2007). They give
# the exact effective day of each change; the monthly snapshots above only say which month.
CHANGES = ROOT / "data" / "ndx_changes.csv"
CHANGE_TITLES = ["Historical components of the Nasdaq-100", "List of NASDAQ-100 companies", "Nasdaq-100"]


def _cell_text(cell: str) -> str:
    cell = re.sub(r"<ref[^>]*/>|<ref[^>]*>.*?</ref>", "", cell, flags=re.S)
    cell = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", cell)       # [[link|text]] -> text
    cell = re.sub(r"\{\{[^{}]*\}\}", "", cell)
    if "|" in cell:                                                     # 'style="..." | value'
        cell = cell.rsplit("|", 1)[-1]
    return re.sub(r"<[^>]+>", "", cell).strip()


def parse_changes(wikitext: str) -> list[dict]:
    """The dated component-change table(s) of a Nasdaq-100 article revision -> [{date, added, removed, reason}]
    (tickers with '.' written as '-'; an empty side is ''). A table counts when its header names Date, Added and
    Removed. Rows whose date does not parse are skipped."""
    out: list[dict] = []
    for tm in re.finditer(r"(?s)\{\|(.*?)\n\|\}", wikitext):
        body = tm.group(1)
        head = body.split("\n|-", 2)
        if not (re.search(r"(?i)\bdate\b", body[:600]) and re.search(r"(?i)\badded\b", body[:600])
                and re.search(r"(?i)\bremoved\b", body[:600])):
            continue
        for row in re.split(r"\n\|-[^\n]*", body)[1:]:
            cells: list[str] = []
            for line in row.split("\n"):
                line = line.strip()
                if not line.startswith("|") or line.startswith(("|}", "|+")):
                    continue
                # "||" separates cells on one line; links are protected from the split
                prot = re.sub(r"\[\[[^\]]*\]\]", lambda m: m.group(0).replace("|", "\x00"), line[1:])
                prot = re.sub(r"\{\{[^{}]*\}\}", lambda m: m.group(0).replace("|", "\x00"), prot)
                cells += [c.replace("\x00", "|") for c in prot.split("||")]
            if len(cells) < 5:
                continue
            txt = [_cell_text(c) for c in cells]
            try:
                d = pd.Timestamp(txt[0])
            except (ValueError, TypeError):
                continue
            if pd.isna(d):
                continue
            tick = lambda x: x.upper().replace(".", "-") if re.fullmatch(r"[A-Za-z.]{1,6}", x or "") else ""  # noqa: E731
            out.append({"date": str(d.date()), "added": tick(txt[1]), "removed": tick(txt[3]),
                        "reason": (txt[5] if len(txt) > 5 else "")[:200]})
    return [r for r in out if r["added"] or r["removed"]]


def update_changes() -> pd.DataFrame | None:
    """data/ndx_changes.csv from the current revision of the change table (kept as it is when that fails)."""
    for title in CHANGE_TITLES:
        try:
            got = wiki_revision_at(pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"), title)
        except Exception as e:  # noqa: BLE001
            print(f"component changes ({title}): {e}", file=sys.stderr)
            continue
        rows = parse_changes(got[2]) if got else []
        if len(rows) >= 50:
            df = pd.DataFrame(rows).drop_duplicates().sort_values(["date", "added", "removed"])
            df.to_csv(CHANGES, index=False)
            print(f"component changes: {len(df)} rows ({df['date'].min()} .. {df['date'].max()}) from {title}")
            return df
    print("component changes: no table found; data/ndx_changes.csv kept", file=sys.stderr)
    return None


def update_membership() -> pd.DataFrame:
    """Monthly snapshots of Nasdaq-100 membership reconstructed from Wikipedia's revision history."""
    have = pd.read_csv(MEMBERSHIP, dtype=str) if MEMBERSHIP.exists() else pd.DataFrame(columns=["month", "revid", "timestamp", "count", "tickers"])
    if "parser" not in have.columns:
        have["parser"] = ""
    # months parsed by an older version of wiki_tickers are fetched again
    done = set(have.loc[have["parser"].fillna("") == MEMBERSHIP_PARSER, "month"])
    months = pd.period_range("2003-01", pd.Timestamp.today().to_period("M"), freq="M")
    rows = []
    fails = 0
    for m in months:
        if fails >= 5:
            print("membership: 5 consecutive failures, giving up for this run", file=sys.stderr)
            break
        key = str(m)
        if key in done and key != str(months[-1]):
            continue
        ts = (m.to_timestamp(how="start")).strftime("%Y-%m-%dT00:00:00Z")
        found = None
        for title in WIKI_TITLES:
            try:
                got = wiki_revision_at(ts, title)
            except Exception as e:  # noqa: BLE001
                print(f"membership {key} ({title}): {e}", file=sys.stderr)
                fails += 1
                time.sleep(2)
                continue
            fails = 0
            time.sleep(0.2)
            if not got:
                continue
            revid, stamp, text = got
            syms = {s.replace(".", "-") for s in wiki_tickers(text)} - NOT_MEMBERS
            if 85 <= len(syms) <= 115:
                found = (revid, stamp, syms)
                break
            print(f"membership {key}: {title} revision {revid} gave {len(syms)} symbols, skipped")
            MEMBERSHIP_LOG.append(f"{key}\t{title}\t{revid}\t{len(syms)} symbols\tskipped\t"
                                  f"section={'yes' if components_section(text) is not text else 'no'}\t"
                                  f"sample={' '.join(sorted(syms)[:12])}")
        if not found:
            continue
        revid, stamp, syms = found
        rows.append({"month": key, "revid": str(revid), "timestamp": stamp, "count": str(len(syms)),
                     "tickers": " ".join(sorted(syms)), "parser": MEMBERSHIP_PARSER})
    if rows:
        new = pd.DataFrame(rows)
        have = pd.concat([have[~have["month"].isin(new["month"])], new]).sort_values("month")
        have.to_csv(MEMBERSHIP, index=False)
    print(f"membership: {len(have)} monthly snapshots ({have['month'].min()} .. {have['month'].max()})")
    # snapshots short of 100 companies are filled from their neighbours when the backtester loads them
    # (backtester.data.repair_membership); log what that does and what is still short
    try:
        from backtester import data as bt_data
        bt_data.membership.cache_clear()
        bt_data.ndx_changes.cache_clear()
        mem = bt_data.membership()
        raw = {pd.Period(m, "M").to_timestamp(): set(t.split()) for m, t in zip(have["month"], have["tickers"])}
        for d, row in mem.iterrows():
            names = set(row.index[row.to_numpy()])
            added = sorted(names - raw.get(d, set()))
            n = bt_data.company_count(names)
            if added or not 100 <= n <= 103:
                MEMBERSHIP_LOG.append(f"{d:%Y-%m}\trepair\t{n} companies\tfilled={' '.join(added)}")
    except Exception as e:  # noqa: BLE001 - the log is informational
        print(f"membership repair log failed: {e}", file=sys.stderr)
    (ROOT / "data" / "membership_log.tsv").write_text("\n".join(MEMBERSHIP_LOG) + "\n")
    return have


# ------------------------------------------------------------------ macro / factors / shares

# OECD long-term (10-year) government bond yields and short rates, monthly, for an international
# bond index before BNDX: (country, weight). Weights are roughly BNDX's country weights; they are
# renormalised each month over the countries that have data.
INTL_BONDS = [("JP", 0.20), ("FR", 0.13), ("DE", 0.11), ("IT", 0.10), ("GB", 0.08), ("CA", 0.06), ("ES", 0.06),
              ("AU", 0.03), ("NL", 0.03), ("BE", 0.03), ("CH", 0.02), ("SE", 0.01)]


def intl_bond_ids() -> list[str]:
    out = []
    for c, _ in INTL_BONDS:
        out += [f"IRLTLT01{c}M156N", f"IR3TIB01{c}M156N", f"IRSTCI01{c}M156N"]
    return out


# Exchange rates for the unhedged international bond model (BWXSIM), FRED H.10 / G.5 series. Per country:
# (daily series or None, monthly-average series, True when quoted in USD per foreign unit). The daily series
# give month-end rates; the monthly averages fill months without a daily quote (the legacy euro-area
# currencies have monthly series only, 1971-2001). The euro-area members convert to the euro from 1999 at the
# fixed conversion rates (EURO_RATES). Series ids verified on fred.stlouisfed.org (fredgraph.csv) in 2026-09.
FX_SERIES = {"JP": ("DEXJPUS", "EXJPUS", False), "GB": ("DEXUSUK", "EXUSUK", True), "CH": ("DEXSZUS", "EXSZUS", False),
             "CA": ("DEXCAUS", "EXCAUS", False), "AU": ("DEXUSAL", "EXUSAL", True), "SE": ("DEXSDUS", "EXSDUS", False),
             "DE": (None, "EXGEUS", False), "FR": (None, "EXFRUS", False), "IT": (None, "EXITUS", False),
             "ES": (None, "EXSPUS", False), "NL": (None, "EXNEUS", False), "BE": (None, "EXBEUS", False),
             "EU": ("DEXUSEU", "EXUSEU", True)}
# legacy currency units per euro, fixed on 1999-01-01 (Council Regulation (EC) No 2866/98)
EURO_RATES = {"DE": 1.95583, "FR": 6.55957, "IT": 1936.27, "ES": 166.386, "NL": 2.20371, "BE": 40.3399}


def fx_ids() -> list[str]:
    return [sid for d, m, _ in FX_SERIES.values() for sid in (d, m) if sid]


# the TIPS model: the Cleveland Fed's 10-year real interest rate and expected inflation (monthly, from 1982)
TIPS_IDS = ["REAINTRATREARAT10Y", "EXPINF10YR"]


def fetch_macro() -> None:
    MACRO.mkdir(parents=True, exist_ok=True)
    for sid in ["CPIAUCSL", "DTB3", "DGS10", "DGS20", "DGS30", "DGS5", "DGS2", "DGS1", "GS10", "TB3MS",
                "DAAA", "DBAA", "AAA", "BAA", "CPIAUCNS"] + intl_bond_ids() + TIPS_IDS + fx_ids():
        try:
            txt = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}", headers=UA, timeout=60).text
            df = pd.read_csv(io.StringIO(txt))
            df.columns = ["date", "value"]
            df = df[pd.to_numeric(df["value"], errors="coerce").notna()]
            if df.empty:
                raise RuntimeError("no observations")
            df.to_csv(MACRO / f"{sid}.csv", index=False)
            print(f"macro {sid}: {len(df)} rows {df['date'].iloc[0]} .. {df['date'].iloc[-1]}")
        except Exception as e:  # noqa: BLE001
            print(f"macro {sid} failed: {e}", file=sys.stderr)


# ---- Robert Shiller's monthly US stock market data (price, dividends, earnings, CPI, 10-year yield, CAPE)

SHILLER_PAGE = "https://shillerdata.com/"        # the data moved here from econ.yale.edu in 2024
SHILLER_FALLBACK = "http://www.econ.yale.edu/~shiller/data/ie_data.xls"
SHILLER_COLUMNS = {"P": "price", "D": "dividend", "E": "earnings", "CPI": "cpi", "CAPE": "cape", "TR CAPE": "tr_cape"}


def _shiller_month(v) -> str | None:
    """Shiller's date cell -> 'YYYY-MM'. The sheet writes October as 1871.1 (a number), January as 1871.01."""
    if v is None or (isinstance(v, float) and v != v):
        return None
    if isinstance(v, (int, float, np.floating)):
        y = int(v)
        mo = int(round((float(v) - y) * 100))
    else:
        m = re.fullmatch(r"\s*(\d{4})\.(\d{1,2})\s*", str(v))
        if not m:
            return None
        y, frac = int(m.group(1)), m.group(2)
        mo = int(frac.ljust(2, "0"))
    if not (1 <= mo <= 12 and 1800 < y < 2200):
        return None
    return f"{y:04d}-{mo:02d}"


def parse_shiller(raw: pd.DataFrame) -> pd.DataFrame:
    """The 'Data' sheet of Shiller's ie_data.xls (read with header=None) -> one row per month: month (YYYY-MM),
    price (S&P Composite, the month's average of daily closes), dividend and earnings (four-quarter totals,
    interpolated to months), cpi, gs10 (the long interest rate, percent), cape (P/E10) and tr_cape (the total
    return version). The header spans several rows; the last one ('Date', 'P', 'D', 'E', 'CPI', ...) names the
    columns. Checked against the file published on shillerdata.com in 2026-09."""
    txt = raw.map(lambda v: "" if v is None or (isinstance(v, float) and v != v) else str(v).strip())
    hdr = next((r for r in range(min(len(raw), 40)) if txt.iat[r, 0].lower() == "date"
                and r + 1 < len(raw) and _shiller_month(raw.iat[r + 1, 0]) is not None), None)
    if hdr is None:
        raise RuntimeError("Shiller data: no 'Date' header row followed by monthly rows")
    cols = {}
    for j in range(raw.shape[1]):
        lab = txt.iat[hdr, j]
        if lab in SHILLER_COLUMNS and SHILLER_COLUMNS[lab] not in cols.values():
            cols[j] = SHILLER_COLUMNS[lab]
        elif "GS10" in lab.upper().replace(" ", "") and "gs10" not in cols.values():
            cols[j] = "gs10"
    missing = {"price", "earnings", "cpi", "cape"} - set(cols.values())
    if missing:
        raise RuntimeError(f"Shiller data: columns {sorted(missing)} not found in header {list(txt.iloc[hdr])[:16]}")
    rows = []
    for r in range(hdr + 1, len(raw)):
        mo = _shiller_month(raw.iat[r, 0])
        if mo is None:
            if rows:
                break
            continue
        row = {"month": mo}
        for j, name in cols.items():
            v = raw.iat[r, j]
            if isinstance(v, str):
                v = v.replace(",", "").strip()
            row[name] = pd.to_numeric(v, errors="coerce")
        rows.append(row)
    df = pd.DataFrame(rows).drop_duplicates("month", keep="last")
    if len(df) < 120:
        raise RuntimeError(f"Shiller data: only {len(df)} monthly rows")
    return df[["month"] + [c for c in ("price", "dividend", "earnings", "cpi", "gs10", "cape", "tr_cape") if c in df]]


def fetch_shiller() -> None:
    """data/macro/shiller.csv from ie_data.xls (the link on shillerdata.com changes with every update, so it is
    read off the page; the old Yale address is the fallback). A failure keeps the previous file."""
    try:
        url = SHILLER_FALLBACK
        try:
            page = requests.get(SHILLER_PAGE, headers=UA, timeout=60).text
            m = re.search(r'https?://[^"\'\s>]+ie_data\.xls[^"\'\s>]*', page)
            if m:
                url = m.group(0).replace("&amp;", "&")
        except Exception as e:  # noqa: BLE001
            print(f"shiller page failed ({e}); trying {url}", file=sys.stderr)
        content = requests.get(url, headers=UA, timeout=120).content
        book = pd.ExcelFile(io.BytesIO(content))
        sheet = "Data" if "Data" in book.sheet_names else book.sheet_names[0]
        df = parse_shiller(book.parse(sheet, header=None))
        MACRO.mkdir(parents=True, exist_ok=True)
        df.to_csv(MACRO / "shiller.csv", index=False)
        print(f"shiller: {len(df)} months {df['month'].iloc[0]} .. {df['month'].iloc[-1]} from {url}")
    except Exception as e:  # noqa: BLE001
        print(f"shiller failed: {e}", file=sys.stderr)
        SIM_LOG.append(f"Shiller CAPE data failed: {e}")


def parse_french_csv(raw: list[str]) -> pd.DataFrame:
    """The first (value-weighted) table of a Ken French data-library CSV: a header line starting with
    a comma, then rows 'YYYYMMDD, r1, r2, ...' (or YYYYMM) in percent. Returns decimal returns with a
    `date` column (monthly rows dated at month end); -99.99 / -999 (missing) become NaN."""
    start = next(i for i, l in enumerate(raw) if re.match(r"^\s*,", l) or l.lower().startswith(",mkt") or "Mkt-RF" in l or "Mom" in l)
    rows = []
    header = [h.strip() for h in raw[start].split(",")]
    for l in raw[start + 1:]:
        parts = [x.strip() for x in l.split(",")]
        if len(parts) != len(header) or not re.fullmatch(r"\d{8}|\d{6}", parts[0]):
            if rows:
                break
            continue
        rows.append(parts)
    df = pd.DataFrame(rows, columns=["date"] + header[1:])
    fmt = "%Y%m%d" if len(df["date"].iloc[0]) == 8 else "%Y%m"
    dt_ = pd.to_datetime(df["date"], format=fmt)
    if fmt == "%Y%m":
        dt_ = dt_ + pd.offsets.MonthEnd(0)
    df["date"] = dt_.dt.strftime("%Y-%m-%d")
    for c in header[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce").where(lambda x: x > -99) / 100.0
    return df


FRENCH_FILES = (("ff3_daily", "F-F_Research_Data_Factors_daily_CSV.zip"),
                ("ff5_daily", "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"),
                ("mom_daily", "F-F_Momentum_Factor_daily_CSV.zip"),
                ("port6_daily", "6_Portfolios_2x3_daily_CSV.zip"),
                ("me_daily", "Portfolios_Formed_on_ME_Daily_CSV.zip"),               # size deciles (mid caps)
                ("dev_ff3_daily", "Developed_ex_US_3_Factors_Daily_CSV.zip"),
                ("dev_port6_daily", "Developed_ex_US_6_Portfolios_ME_BE-ME_daily_CSV.zip"),  # intl size/value
                ("eu_ff3_daily", "Europe_3_Factors_Daily_CSV.zip"),
                ("em_ff5_monthly", "Emerging_5_Factors_CSV.zip"),
                # 25 size x B/M portfolios (value-weighted, daily from 1926): mid-cap value/growth and the
                # large value / small value refinements in build_sims; rounded when saved (about 4 MB)
                ("port25_daily", "25_Portfolios_5x5_Daily_CSV.zip"))


def fetch_factors() -> None:
    """Kenneth French's factor files (US daily and official monthly; the international regions' 3-factor,
    5-factor and momentum files, monthly and daily; emerging markets monthly) and AQR's QMJ and BAB monthly
    factors. Each file is parsed by backtester/sources.py; a failure is logged (data/factors/fetch_log.txt)
    and the other files carry on."""
    import zipfile
    FACTORS.mkdir(parents=True, exist_ok=True)
    ok, bad = [], []
    for name, fn in list(sources.FRENCH_FILES) + [f for f in FRENCH_FILES if f[0] not in dict(sources.FRENCH_FILES)]:
        try:
            z = zipfile.ZipFile(io.BytesIO(requests.get(sources.FRENCH_BASE + fn, headers=UA, timeout=120).content))
            member = next(n for n in z.namelist() if n.lower().endswith(".csv"))
            df = sources.parse_french_csv(z.read(member).decode("latin-1"))
            df.to_csv(FACTORS / f"{name}.csv", index=False, float_format="%.6g" if name == "port25_daily" else None)
            ok.append(name)
            print(f"factors {name}: {len(df)} rows {df['date'].iloc[0]} .. {df['date'].iloc[-1]}, columns {list(df.columns[1:])}")
        except Exception as e:  # noqa: BLE001
            bad.append(f"{name} ({fn}): {e}")
            print(f"factors {name} failed: {e}", file=sys.stderr)
            SIM_LOG.append(f"factors {name} ({fn}) failed: {e}")
        time.sleep(0.3)
    for name, fn, sheet in sources.AQR_FILES:
        try:
            content = requests.get(sources.AQR_BASE + fn, headers=UA, timeout=180).content
            xl = pd.ExcelFile(io.BytesIO(content))
            use = sheet if sheet in xl.sheet_names else xl.sheet_names[0]
            df = sources.parse_aqr_sheet(xl.parse(use, header=None))
            df.to_csv(FACTORS / f"{name}.csv", index=False)
            ok.append(name)
            print(f"factors {name}: {len(df)} rows {df['date'].iloc[0]} .. {df['date'].iloc[-1]}, columns {list(df.columns[1:])}")
        except Exception as e:  # noqa: BLE001
            bad.append(f"{name} ({fn}): {e}")
            print(f"factors {name} failed: {e}", file=sys.stderr)
    (FACTORS / "fetch_log.txt").write_text(
        f"{len(ok)} factor files downloaded\n" + ("failed:\n" + "\n".join(bad) + "\n" if bad else "no failures\n"))


def fetch_shares(tickers: list[str]) -> None:
    """Shares outstanding history (Yahoo, point in time: each count in the share units of its date, not
    split-adjusted - backtester/data.py puts them on the price files' split basis). Fetched under the
    renamed symbol for old symbols (FB -> META). Merged into the saved file, never replacing it: Yahoo
    only serves the last several years, so the saved file is the only record of older counts."""
    SHARES.mkdir(parents=True, exist_ok=True)
    for t in tickers:
        src = RENAMES.get(t, t)
        try:
            s = yf.Ticker(src).get_shares_full(start="2000-01-01")
            if s is None or len(s) == 0:
                continue
            s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
            s = pd.to_numeric(s, errors="coerce").dropna()
            s = s[(s > 0) & ~s.index.duplicated(keep="last")].sort_index().rename("shares")
            path = SHARES / f"{t}.csv"
            if path.exists():
                old = pd.read_csv(path, parse_dates=["date"], index_col="date")["shares"]
                old = old[~old.index.duplicated(keep="last")]
                s = s.combine_first(old).sort_index().rename("shares")
            s.to_csv(path, index_label="date")
        except Exception as e:  # noqa: BLE001
            print(f"shares {t} failed: {e}", file=sys.stderr)


# ------------------------------------------------------------------ SEC EDGAR share counts

# SEC asks automated clients for a descriptive User-Agent with a contact; the repository is the contact
# SEC's fair-access policy refuses requests whose User-Agent has no contact e-mail (403 on every call). The repo's
# GitHub no-reply address is used by default; set SEC_CONTACT (a repository variable or secret) to override it.
import os as _os
SEC_UA = {"User-Agent": "backtester-data-job " + (_os.environ.get("SEC_CONTACT") or "alm0421@users.noreply.github.com"),
          "Accept-Encoding": "gzip, deflate"}
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
# delisted former members, absent from company_tickers.json (each checked against data.sec.gov/submissions:
# name and former names)
SEC_CIKS = {"EA": 712515,      # ELECTRONIC ARTS INC.
            "ATVI": 718877,    # Activision Blizzard, Inc. (formerly ACTIVISION INC /NY)
            "CELG": 816284,    # CELGENE CORP /DE/
            "XLNX": 743988,    # XILINX INC
            "YHOO": 1011006,   # ALTABA INC. (formerly YAHOO INC)
            "BRCM": 1054374}   # BROADCOM CORP
# cover-page count first (as of a date just before the filing), then the balance-sheet count, then the
# period's weighted average (the weakest: an average, not a count on a date)
SEC_CONCEPTS = [("dei", "EntityCommonStockSharesOutstanding"), ("us-gaap", "CommonStockSharesOutstanding"),
                ("us-gaap", "WeightedAverageNumberOfSharesOutstandingBasic")]


def sec_ticker_map(raw: dict) -> dict[str, int]:
    """company_tickers.json ({"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}) ->
    {ticker: cik}, with class suffixes as in the price files (BRK-B)."""
    out = {}
    for v in raw.values():
        t = str(v.get("ticker", "")).upper().replace(".", "-")
        if t and v.get("cik_str") is not None:
            out.setdefault(t, int(v["cik_str"]))
    return out


def sec_share_counts(facts: dict) -> pd.Series:
    """Point-in-time shares outstanding from a companyfacts document: one count per filing, dated by the
    filing date (when it became public), in the share units reported then (NOT adjusted for later splits;
    backtester/data.py applies the same split basis as for Yahoo counts). Per filing, the first concept of
    SEC_CONCEPTS it reports is used, at the latest period end in that filing (later filings restate earlier
    periods, often on a new split basis - those restatements are ignored)."""
    per: dict[str, tuple[int, str, float, str]] = {}   # accn -> (concept rank, period end, value, filed)
    f = facts.get("facts") or {}
    for rank, (tax, name) in enumerate(SEC_CONCEPTS):
        units = ((f.get(tax) or {}).get(name) or {}).get("units") or {}
        for x in units.get("shares") or []:
            try:
                accn, end, filed, val = x["accn"], x["end"], x["filed"], float(x["val"])
            except (KeyError, TypeError, ValueError):
                continue
            if val <= 0:
                continue
            if rank == 2 and x.get("start"):
                # a weighted average: only the filing's own quarter or year (not a restated older period)
                days = (pd.Timestamp(end) - pd.Timestamp(x["start"])).days
                if days > 370:
                    continue
            cur = per.get(accn)
            if cur is None or rank < cur[0] or (rank == cur[0] and end > cur[1]):
                per[accn] = (rank, end, val, filed)
    if not per:
        return pd.Series(dtype=float, name="shares")
    rows = sorted((v[3], v[2]) for v in per.values())
    s = pd.Series([v for _, v in rows], index=pd.DatetimeIndex([d for d, _ in rows]), name="shares", dtype=float)
    return s[~s.index.duplicated(keep="last")]


def fetch_sec_shares(tickers: list[str]) -> None:
    """Share counts back to ~2009 (XBRL) from SEC EDGAR company facts, saved per ticker in data/shares_sec/
    (date = filing date, shares, as reported). data.shares_outstanding merges them with the Yahoo counts in
    data/shares. Keyless; SEC's fair-access limit is 10 requests a second."""
    import time
    folder = ROOT / "data" / "shares_sec"      # (from ROOT at call time, like the other outputs)
    folder.mkdir(parents=True, exist_ok=True)
    try:
        r = requests.get(SEC_TICKERS_URL, headers=SEC_UA, timeout=60)
        r.raise_for_status()
        ciks = sec_ticker_map(r.json())
    except Exception as e:  # noqa: BLE001
        print(f"sec tickers failed: {e}", file=sys.stderr)
        ciks = {}
        errors = [f"company_tickers.json: {e}"]
    else:
        errors = []
    ciks.update(SEC_CIKS)
    got, missing = 0, []
    for t in tickers:
        cik = ciks.get(t) or ciks.get(RENAMES.get(t, ""))
        if not cik:
            missing.append(t)
            continue
        try:
            time.sleep(0.15)
            r = requests.get(SEC_FACTS_URL.format(cik=cik), headers=SEC_UA, timeout=60)
            if r.status_code == 404:
                missing.append(t)
                continue
            r.raise_for_status()
            s = sec_share_counts(r.json())
            if s.empty:
                missing.append(t)
                continue
            path = folder / f"{t}.csv"
            if path.exists():
                old = pd.read_csv(path, parse_dates=["date"], index_col="date")["shares"]
                s = s.combine_first(old[~old.index.duplicated(keep="last")]).sort_index().rename("shares")
            s.to_csv(path, index_label="date", float_format="%.0f")
            got += 1
        except Exception as e:  # noqa: BLE001
            print(f"sec shares {t} failed: {e}", file=sys.stderr)
            missing.append(t)
            if len(errors) < 5:
                errors.append(f"{t}: {e}")
    (folder / "fetch_log.txt").write_text(f"{got} tickers with SEC share counts\n"
                                              + ("errors (first 5): " + " | ".join(errors) + "\n" if errors else "")
                                              + ("no CIK or no counts: " + " ".join(missing) + "\n" if missing else ""))


# ------------------------------------------------------------------ main

# ------------------------------------------------------------------ simulated long histories

def _bond_returns(y: pd.Series, maturity: float, step_years: float | None = None) -> pd.Series:
    """Daily total return of a constant-maturity Treasury fund priced off a yield series (in %).

    Each day: yesterday's par bond earns its coupon for the days held and is re-priced at today's
    yield (semi-annual coupons). The standard way to extend bond funds back before they existed.
    (Works on any spacing of observations; `step_years` fixes the accrual period, e.g. 1/12 for a
    monthly series, instead of the calendar days between observations.)"""
    y = (y / 100.0).dropna()
    c = y.shift(1).to_numpy()
    r = y.to_numpy()
    d = (np.full(len(y), float(step_years)) if step_years is not None
         else pd.Series(y.index, index=y.index).diff().dt.days.to_numpy() / 365.25)
    k = np.arange(1, int(round(maturity * 2)) + 1)
    disc = np.power(1 + r[:, None] / 2, -k[None, :])
    price = (c[:, None] / 2 * disc).sum(axis=1) + disc[:, -1]
    return pd.Series(price - 1 + c * d, index=y.index)


def _series_file(t: str, level: pd.Series, note: str) -> None:
    level = level.dropna()
    df = pd.DataFrame({"open": level, "high": level, "low": level, "close": level, "volume": 0,
                       "dividend": 0.0, "adj_close": level})
    df.index.name = "date"
    save_prices(t, df, precision="%.10g")  # full precision: daily returns are rebuilt from these levels
    print(f"sim {t}: {len(df)} rows {df.index[0].date()} .. {df.index[-1].date()} ({note})")


def _real_returns(real: str) -> pd.Series:
    """Daily total returns (from adj_close) of a downloaded fund, or an empty series (REAL_OVERRIDES: a fund
    whose distributions were repaired for its use in the SIMs)."""
    if real in REAL_OVERRIDES:
        return REAL_OVERRIDES[real].copy()
    p = PRICES / f"{real}.csv"
    if not p.exists():
        return pd.Series(dtype=float)
    df = pd.read_csv(p, parse_dates=["date"], index_col="date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    adj = pd.to_numeric(df["adj_close"], errors="coerce")
    return adj.where(adj > 0).dropna().pct_change().dropna()


def _splice_returns(sim_ret: pd.Series, *reals: str) -> pd.Series:
    """Simulated returns until the first real fund starts, then each real fund in turn (the later
    fund takes over from its own start). Missing funds are skipped. "NAESX@1989-10-01" uses a fund only
    from that date (e.g. when it became an index fund)."""
    ret = sim_ret.dropna().sort_index()
    for real in reals:
        real, _, since = real.partition("@")
        r = _real_returns(real)
        if since:
            r = r[r.index >= pd.Timestamp(since)]
        if len(r):
            ret = pd.concat([ret[ret.index < r.index[0]], r])
    return ret.fillna(0.0)


def _splice(sim_ret: pd.Series, *reals: str) -> pd.Series:
    """Simulated returns before the real fund(s) existed, then their total return, as a level from 100."""
    return 100 * (1 + _splice_returns(sim_ret, *reals)).cumprod()


# ---- fee / cost haircut of the MODEL segment of each SIM
#
# The models (Fama-French portfolios, index returns, par bonds priced off yields) are gross: no expense ratio,
# no trading costs, no cash drag, no securities-lending or tax leakage. On their overlap with the real funds
# they beat them by about 1-2% a year (data/sims_log.txt). Before splicing, each model's returns are cut by a
# constant annual drag: the geometric CAGR gap between the model and the fund that takes over from it, on their
# overlap (full months), floored at that fund's expense ratio and capped at SIM_DRAG_CAP. The real fund segment is
# never touched (it is already net of every cost).
#
# Expense ratios (annual, as fractions) of the funds a model hands over to, and of the ETFs after them. Sources:
# each issuer's fund page / latest prospectus (Vanguard, iShares, SPDR, Invesco, PIMCO, Fidelity), as of 2025,
# rounded to 0.01%. These are CURRENT figures: most were higher decades ago (VTSMX charged 0.25% in the 1990s),
# which the overlap calibration picks up as part of the measured gap. Only used as the floor of the drag.
SIM_EXPENSE_RATIOS = {
    "SPY": 0.000945, "BIL": 0.001356, "VTSMX": 0.0014, "VTI": 0.0003, "TLT": 0.0015, "IEF": 0.0015, "SHY": 0.0015,
    "IEI": 0.0015, "VIVAX": 0.0017, "VTV": 0.0004, "VIGRX": 0.0017, "VUG": 0.0004, "VISVX": 0.0019, "VBR": 0.0007,
    "VISGX": 0.0019, "VBK": 0.0007, "NAESX": 0.0017, "VB": 0.0005, "VOE": 0.0007, "VOT": 0.0007, "MDY": 0.0023,
    "IJH": 0.0005, "EFA": 0.0035, "VGK": 0.0006, "SCZ": 0.0040, "AVDV": 0.0036, "EFV": 0.0033, "LQD": 0.0014,
    "VBMFX": 0.0015, "BND": 0.0003, "PFORX": 0.0050, "BNDX": 0.0007, "EEM": 0.0070, "VEIEX": 0.0029, "VWO": 0.0007,
    "VGTSX": 0.0017, "VXUS": 0.0005, "VGSIX": 0.0027, "VNQ": 0.0013, "GLD": 0.0040, "DBC": 0.0085, "PCRIX": 0.0074,
    "VWESX": 0.0021, "VCLT": 0.0004, "EWJ": 0.0050, "EWU": 0.0050, "EWG": 0.0050, "EWC": 0.0050, "EWA": 0.0050,
    "EWQ": 0.0050, "EWL": 0.0050, "EWH": 0.0050,
    # added with the TIPS / high-yield / unhedged-bond models (Vanguard, iShares, SPDR fund pages; BWX checked
    # on ssga.com and Morningstar 2026-09: 0.35%)
    "VIPSX": 0.0020, "TIP": 0.0018, "VWEHX": 0.0023, "HYG": 0.0049, "BWX": 0.0035, "IGOV": 0.0035,
}
SIM_DRAG_CAP = 0.03          # a larger gap is model error, not costs: never haircut more than 3% a year
SIM_DRAG: dict[str, dict] = {}   # ticker -> the drag applied this run (written to data/sims_drag.json)


def overlap_cagrs(model: pd.Series, fund: pd.Series) -> tuple[float, float, int, str, str] | None:
    """(model CAGR, fund CAGR, months, first month, last month) of two daily return series on their overlap,
    in full calendar months (the partial first and last months are dropped), or None under 12 months."""
    s, r = model.dropna(), fund.dropna()
    if s.empty or r.empty:
        return None
    lo, hi = max(s.index[0], r.index[0]), min(s.index[-1], r.index[-1])
    if lo >= hi:
        return None
    both = pd.concat({"m": s, "e": r}, axis=1).loc[lo:hi].fillna(0.0)
    mo = ((1 + both).groupby(both.index.to_period("M")).prod() - 1).iloc[1:-1]
    if len(mo) < 12:
        return None
    yrs = len(mo) / 12
    cm = float((1 + mo["m"]).prod() ** (1 / yrs) - 1)
    ce = float((1 + mo["e"]).prod() ** (1 / yrs) - 1)
    return cm, ce, len(mo), str(mo.index[0]), str(mo.index[-1])


def calibrate_drag(model: pd.Series, fund: pd.Series, expense_ratio: float, cap: float = SIM_DRAG_CAP) -> dict:
    """The annual drag for a model that hands over to `fund`: the geometric CAGR gap (1 + model) / (1 + fund) - 1
    on their overlap (0 when the fund did better), at least `expense_ratio`, at most `cap` (and never below the
    expense ratio even when that exceeds the cap). Without 12 overlapping months: the expense ratio alone."""
    out = {"expense_ratio": float(expense_ratio), "gap": None, "months": 0, "overlap": None,
           "model_cagr": None, "fund_cagr": None}
    ov = overlap_cagrs(model, fund)
    if ov is None:
        out["drag"] = float(expense_ratio)
        out["basis"] = "expense ratio (no overlap to calibrate on)"
        return out
    cm, ce, n, a, b = ov
    gap = max(0.0, (1 + cm) / (1 + ce) - 1)
    drag = max(float(expense_ratio), min(cap, gap))
    out.update(gap=gap, months=n, overlap=f"{a}..{b}", model_cagr=cm, fund_cagr=ce, drag=drag,
               basis=("expense ratio (the model did not beat the fund)" if gap <= expense_ratio else
                      f"overlap gap capped at {cap:.0%}" if gap > cap else "overlap gap"))
    return out


def apply_drag(ret: pd.Series, drag: float, before=None) -> pd.Series:
    """Daily returns with a constant annual `drag` taken out: each row's growth is divided by (1 + drag) ** years,
    years = the calendar days since the previous row / 365.25 (1/252 for the first row), so a monthly-stepped
    series pays the same drag per year as a daily one. Only rows before `before` (a date) are changed."""
    ret = ret.dropna().sort_index()
    if not drag or ret.empty:
        return ret
    idx = pd.DatetimeIndex(ret.index)
    days = np.array(pd.Series(idx, index=idx).diff().dt.days, dtype=float)
    days[0] = 365.25 / 252
    f = (1 + float(drag)) ** (-days / 365.25)
    if before is not None:
        f = np.where(idx < pd.Timestamp(before), f, 1.0)
    return pd.Series((1 + ret.to_numpy()) * f - 1, index=ret.index)


def _handover(reals: tuple[str, ...]) -> tuple[str, pd.Series]:
    """The first real fund of a splice that has data (the one that takes over from the model) and its returns
    from its '@since' date on; ("", empty) when none has data."""
    for real in reals:
        t, _, since = real.partition("@")
        r = _real_returns(t)
        if since:
            r = r[r.index >= pd.Timestamp(since)]
        if len(r):
            return t, r
    return "", pd.Series(dtype=float)


def haircut_model(t: str, sim_ret: pd.Series, reals: tuple[str, ...]) -> pd.Series:
    """The model's daily returns net of its fee/cost drag (calibrate_drag against the fund it hands over to),
    logged to SIM_NOTES and recorded in SIM_DRAG. Only the model rows before the hand-over date change; the
    splice then replaces everything from that date with the fund itself."""
    fund, r = _handover(reals)
    etf = reals[-1].partition("@")[0] if reals else ""
    er = SIM_EXPENSE_RATIOS.get(fund, SIM_EXPENSE_RATIOS.get(etf, 0.0))
    if fund and fund not in SIM_EXPENSE_RATIOS:
        _simnote(f"drag {t}: no expense ratio on file for {fund}; using {etf}'s ({er:.2%})")
    cal = calibrate_drag(sim_ret, r, er) if fund else {"expense_ratio": er, "drag": er, "gap": None, "months": 0,
                                                       "overlap": None, "basis": "expense ratio (no fund data)"}
    until = r.index[0] if len(r) else None
    net = apply_drag(sim_ret, cal["drag"], before=until)
    # the check: the whole model with the same drag, against the fund on their overlap (after the hand-over date,
    # where the spliced series is the fund itself)
    after = overlap_cagrs(apply_drag(sim_ret, cal["drag"]), r) if len(r) else None
    SIM_DRAG[t] = {"drag": cal["drag"], "expense_ratio": cal["expense_ratio"], "fund": fund or etf,
                   "gap": cal["gap"], "overlap": cal["overlap"],
                   "months": cal["months"], "basis": cal["basis"],
                   "model_until": None if until is None else str(pd.Timestamp(until).date())}
    _simnote(f"drag {t}: {cal['drag']:.2%}/yr taken off the model before {SIM_DRAG[t]['model_until'] or 'the end'} "
             f"(hands over to {fund or 'no fund'}; expense ratio {er:.2%}; "
             + (f"model beat the fund by {cal['gap']:.2%}/yr on {cal['overlap']} ({cal['months']} months)"
                if cal["gap"] is not None else "no overlap to calibrate on")
             + f"; basis: {cal['basis']})"
             + (f"; with the drag the model returns {after[0]:.2%}/yr vs the fund's {after[1]:.2%} on the overlap" if after else ""))
    return net


def write_sim_drag(path: Path | None = None) -> None:
    """Merge this run's drags into data/sims_drag.json (a series that failed this run keeps its old entry, as
    its price file is kept too)."""
    path = path or ROOT / "data" / "sims_drag.json"
    try:
        old = json.loads(path.read_text()).get("series", {}) if path.exists() else {}
    except Exception:  # noqa: BLE001
        old = {}
    old.update(SIM_DRAG)
    path.write_text(json.dumps({"cap": SIM_DRAG_CAP, "method": "max(expense ratio, min(cap, model/fund CAGR gap on "
                                "their overlap)), taken daily off the model segment before the fund's first day",
                                "series": dict(sorted(old.items()))}, indent=1))


SIM_LOG: list[str] = []     # failures (with a traceback)
SIM_NOTES: list[str] = []   # validation of each model against the real fund on their overlap


def _simlog(msg: str) -> None:
    import traceback
    SIM_LOG.append(msg + "\n" + traceback.format_exc(limit=3))
    print(msg, flush=True)


def _simnote(msg: str) -> None:
    SIM_NOTES.append(msg)
    print(msg, flush=True)


# ---- the NYSE calendar for monthly-stepped segments

def _nyse():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from backtester import calendar as nyse
    return nyse


def _sessions(start, end) -> pd.DatetimeIndex:
    """NYSE sessions from `start` to `end` (inclusive, never after today).

    Actual trading days from the Fama-French daily files where they exist (they include unscheduled
    closures such as 9/11), the regular NYSE schedule (backtester/calendar.py) outside them; from
    1972 on, only days the schedule counts as sessions, which is what the backtester keeps for SIMs."""
    start = pd.Timestamp(start).normalize()
    end = min(pd.Timestamp(end).normalize(), pd.Timestamp.today().normalize())
    days = pd.bdate_range(start, end)
    try:
        known = pd.DatetimeIndex(pd.read_csv(FACTORS / "ff3_daily.csv", usecols=["date"], parse_dates=["date"])["date"])
        known = known.sort_values().unique()
    except Exception:  # noqa: BLE001 - no factor file: the schedule alone
        known = pd.DatetimeIndex([])
    if len(known):
        days = days[days < known[0]].append(known[(known >= start) & (known <= end)]).append(days[days > known[-1]])
    nyse = _nyse()
    keep = np.array([d.year < 1972 or nyse.is_session(d) for d in days], dtype=bool)
    return pd.DatetimeIndex(days[keep]).unique().sort_values()


def _last_sessions(start=None, end=None, sess: pd.DatetimeIndex | None = None) -> pd.Series:
    """The last NYSE session of each month (index: monthly periods)."""
    if sess is None:
        sess = _sessions(start, end)
    return pd.Series(sess, index=sess).groupby(sess.to_period("M")).max()


def _monthly_steps(r: pd.Series) -> pd.Series:
    """Monthly returns (indexed by any date within their month) as a daily return series on NYSE
    sessions: zero on every session except the LAST session of each month, which carries that month's
    return. So a month that ends on a weekend or holiday (May 1975 ended on a Saturday) is booked on
    its last trading day (Friday 1975-05-30), not on the next month's first session, and month-end
    rebalancing sees the same month-end as every real ticker."""
    r = pd.to_numeric(r, errors="coerce").dropna().sort_index()
    if r.empty:
        return pd.Series(dtype=float)
    months = r.index.asfreq("M") if isinstance(r.index, pd.PeriodIndex) else pd.DatetimeIndex(r.index).to_period("M")
    r = pd.Series(r.to_numpy(), index=months)
    r = r[~r.index.duplicated(keep="last")]
    sess = _sessions(r.index[0].start_time, r.index[-1].end_time)
    out = pd.Series(0.0, index=sess)
    hit = _last_sessions(sess=sess).reindex(r.index).dropna()
    out.loc[pd.DatetimeIndex(hit.to_numpy())] = r.loc[hit.index].to_numpy()
    return out


def _monthly_level_steps(level: pd.Series) -> pd.Series:
    """A monthly level series (e.g. month-end prices) as monthly steps on the last session."""
    level = pd.to_numeric(level, errors="coerce").dropna().sort_index()
    level = level[level > 0]
    return _monthly_steps(level.pct_change().dropna())


def _on_sessions(level: pd.Series) -> pd.Series:
    """A daily level series from another market's calendar (e.g. the London gold fixing) as daily
    returns on NYSE sessions: each session takes the latest level known at or before it."""
    level = pd.to_numeric(level, errors="coerce").dropna().sort_index()
    level = level[level > 0]
    level = level[~level.index.duplicated(keep="last")]
    sess = _sessions(level.index[0], level.index[-1])
    return level.reindex(level.index.union(sess)).ffill().reindex(sess).pct_change().dropna()


def corporate_yield(daaa: pd.Series, dbaa: pd.Series, aaa: pd.Series, baa: pd.Series,
                    since: str = "1953-01-01") -> pd.Series:
    """The average of Moody's Aaa and Baa yields as one series: the monthly averages (AAA, BAA; each held from
    the last session of its month) until BOTH daily series (DAAA from 1983, DBAA from 1986) exist, then the
    daily average. (Averaging the daily series where only one of them exists gives NaN, which once dropped
    1983-85 entirely.)"""
    daily = pd.concat({"a": daaa, "b": dbaa}, axis=1).dropna().mean(axis=1)
    if daily.empty:
        raise RuntimeError("no day with both daily Aaa and Baa yields")
    monthly = pd.concat({"a": aaa, "b": baa}, axis=1).dropna().mean(axis=1)
    m = monthly[(monthly.index < daily.index[0]) & (monthly.index >= since)]
    if m.empty:
        return daily
    per = pd.DatetimeIndex(m.index).to_period("M")
    at = _last_sessions(per[0].start_time, per[-1].end_time).reindex(per)
    mm = pd.Series(m.to_numpy(), index=pd.DatetimeIndex(at.to_numpy()))
    mm = mm[mm.index.notna() & (mm.index < daily.index[0])]
    sess = _sessions(mm.index[0], daily.index[0] - pd.Timedelta(days=1))
    lvl = mm.reindex(sess.union(mm.index)).ffill()
    corp = pd.concat([lvl, daily]).sort_index()
    return corp[~corp.index.duplicated(keep="last")]


MAX_GAP_SESSIONS = 10


def internal_gaps(level: pd.Series, max_sessions: int = MAX_GAP_SESSIONS) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    """Holes inside a daily series after its first date: (last date before, first date after, business days
    missing) for every gap of more than `max_sessions` business days. A SIM must have none (a missing stretch
    is held in cash by every portfolio that owns it)."""
    idx = pd.DatetimeIndex(pd.Series(level).dropna().index).sort_values().unique()
    if len(idx) < 2:
        return []
    a = idx[:-1].values.astype("datetime64[D]")
    b = idx[1:].values.astype("datetime64[D]")
    missing = np.busday_count(a, b) - 1
    return [(idx[k], idx[k + 1], int(missing[k])) for k in np.nonzero(missing > max_sessions)[0]]


def _validate(name: str, sim_ret: pd.Series, real: str, since=None) -> None:
    """Log how the model (before splicing) compares with the real fund over their overlap (monthly); `since`: only
    from that date (a whole month), e.g. from when the fund became an index fund."""
    try:
        r = _real_returns(real)
        s = sim_ret.dropna()
        if since is not None:
            r, s = r[r.index >= pd.Timestamp(since)], s[s.index >= pd.Timestamp(since)]
        if r.empty or s.empty:
            _simnote(f"validate {name} vs {real}: no overlap (fund or model missing)")
            return
        both = pd.concat({"m": s, "e": r}, axis=1).loc[max(s.index[0], r.index[0]):min(s.index[-1], r.index[-1])].fillna(0.0)
        mo = (1 + both).groupby(both.index.to_period("M")).prod() - 1
        # partial first / last months (with `since` on a month start the first month is whole)
        whole_first = since is not None and pd.Timestamp(since).day == 1 and both.index[0] - pd.Timestamp(since) < pd.Timedelta(days=5)
        mo = mo.iloc[(0 if whole_first else 1):-1]
        if len(mo) < 12:
            _simnote(f"validate {name} vs {real}: only {len(mo)} overlapping months")
            return
        yrs = len(mo) / 12
        cm, ce = (1 + mo["m"]).prod() ** (1 / yrs) - 1, (1 + mo["e"]).prod() ** (1 / yrs) - 1
        _simnote(f"validate {name} vs {real}{' (from ' + str(since)[:10] + ')' if since is not None else ''} "
                 f"{mo.index[0]}..{mo.index[-1]}: monthly correlation "
                 f"{mo['m'].corr(mo['e']):.3f}, tracking error {(mo['m'] - mo['e']).std() * 12 ** 0.5:.2%}/yr, "
                 f"CAGR model {cm:.2%} vs fund {ce:.2%}, volatility {mo['m'].std() * 12 ** 0.5:.1%} vs {mo['e'].std() * 12 ** 0.5:.1%}")
    except Exception as e:  # noqa: BLE001
        _simlog(f"validate {name} vs {real} failed: {e}")


def _tracking_error(sim_ret: pd.Series, real: str | pd.Series, since=None) -> tuple[float, int] | None:
    """(annualised monthly tracking error, months) of a model against a fund's total return on their
    overlap (full months only), or None with less than 24 months. `real` is a ticker or a return series."""
    r = _real_returns(real) if isinstance(real, str) else real.dropna()
    s = sim_ret.dropna()
    if r.empty or s.empty:
        return None
    lo, hi = max(s.index[0], r.index[0]), min(s.index[-1], r.index[-1])
    if since is not None:
        lo = max(lo, pd.Timestamp(since))
    both = pd.concat({"m": s, "e": r}, axis=1).loc[lo:hi].fillna(0.0)
    mo = ((1 + both).groupby(both.index.to_period("M")).prod() - 1).iloc[1:-1]
    if len(mo) < 24:
        return None
    return float((mo["m"] - mo["e"]).std() * 12 ** 0.5), len(mo)


def _pick_model(name: str, cands: dict, targets: tuple[str, ...], since=None) -> tuple[str, pd.Series]:
    """The candidate model (label -> daily returns) that tracks the first target fund with data best
    (lowest tracking error on the overlap); every candidate's score goes to the log. With no fund data,
    the first candidate."""
    cands = {k: v.dropna() for k, v in cands.items() if v is not None and len(v.dropna())}
    if not cands:
        raise RuntimeError(f"{name}: no candidate model")
    for real in targets:
        scores = {k: _tracking_error(v, real, since=since) for k, v in cands.items()}
        scores = {k: v for k, v in scores.items() if v}
        if scores:
            best = min(scores, key=lambda k: scores[k][0])
            _simnote(f"{name} model choice vs {real}: " + "; ".join(
                f"{k}: tracking error {te:.2%}/yr ({n} months)" for k, (te, n) in scores.items()) + f" -> {best}")
            return best, cands[best]
    first = next(iter(cands))
    _simnote(f"{name} model choice: no fund data to compare, using {first}")
    return first, cands[first]


def _fred(sid: str) -> pd.Series:
    d = pd.read_csv(MACRO / f"{sid}.csv", parse_dates=["date"], index_col="date")["value"]
    return pd.to_numeric(d, errors="coerce").dropna().sort_index()


def _factor_file(name: str) -> pd.DataFrame:
    df = pd.read_csv(FACTORS / f"{name}.csv", parse_dates=["date"], index_col="date").sort_index()
    df.columns = [c.strip() for c in df.columns]
    return df


def _col(df: pd.DataFrame, *keys: str) -> str:
    """The column whose name, upper-cased without spaces, is one of `keys`."""
    norm = {c.upper().replace(" ", ""): c for c in df.columns}
    for k in keys:
        if k in norm:
            return norm[k]
    raise KeyError(f"none of {keys} in {list(df.columns)}")


def build_sims(only: set[str] | None = None) -> list[str]:
    """Long total-return histories for portfolio research: a model (index data or yields) before each
    fund existed, then the fund's own total return. Each series is built independently: one failing
    source is logged in data/sims_log.txt and never stops the others. `only`: build just these groups (the
    attempt labels, e.g. {"SPYSIM/BILSIM", "TIPSIM"}), to rebuild a few series by hand."""
    made: list[str] = []

    def build(t: str, sim_ret: pd.Series, reals: tuple[str, ...], note: str, validate: str | None = None,
              model: bool = True, validate_since=None) -> None:
        """`model=False`: sim_ret is itself a real fund's return (net of its costs), so no fee/cost drag."""
        sim_ret = sim_ret.dropna()
        if sim_ret.empty:
            raise RuntimeError("empty model series")
        if validate:
            _validate(t, sim_ret, validate, since=validate_since)   # the gross model: the gap it shows calibrates the drag
        if model:
            sim_ret = haircut_model(t, sim_ret, reals)
            note += f" (model period net of an estimated {SIM_DRAG[t]['drag']:.2%}/yr fee/cost drag)"
        level = _splice(sim_ret, *reals)
        for a, b, n in internal_gaps(level):
            # a data-quality failure, logged as one (the series is still written; the loader notes the hole)
            SIM_LOG.append(f"sim {t} data-quality failure: {n} business days missing between {a.date()} and {b.date()}")
            print(SIM_LOG[-1], flush=True)
        _series_file(t, level, note)
        made.append(t)

    def attempt(label: str, fn) -> None:
        if only is not None and label not in only:
            return
        try:
            fn()
        except Exception as e:  # noqa: BLE001 - one series never breaks the run
            _simlog(f"sim {label} failed: {e}")

    ff = {}

    def us_market():
        f = _factor_file("ff3_daily")
        ff["rf"] = f["RF"].dropna()
        build("SPYSIM", f["Mkt-RF"] + f["RF"], ("SPY",), "US market total return from Fama-French before SPY, then SPY", "SPY")
        build("BILSIM", f["RF"], ("BIL",), "1-month T-bill (Fama-French RF) before BIL, then BIL")
        ff["mkt"] = (f["Mkt-RF"] + f["RF"]).dropna()
    attempt("SPYSIM/BILSIM", us_market)

    # "fund-exact" series: the named fund as soon as it or its mutual-fund twin exists
    def total_market():
        note = ("US total market: Fama-French market return until April 1992, then the Vanguard Total Stock Market "
                "Index fund (VTSMX), then VTI from June 2001")
        try:
            # Yahoo's VTSMX misses part of some early distributions: repair those ex-dates against the market
            # (see repair_missed_distributions) for its use here only
            p = pd.read_csv(PRICES / "VTSMX.csv", parse_dates=["date"], index_col="date").sort_index()
            p = p[~p.index.duplicated(keep="last")]
            raw = _real_returns("VTSMX")
            vti = _real_returns("VTI")
            fixed, days = repair_missed_distributions(raw, ff["mkt"], p["dividend"],
                                                      until=vti.index[0] if len(vti) else None)
            if days:
                REAL_OVERRIDES["VTSMX"] = fixed
                yr = lambda s: (1 + s).groupby(s.index.year).prod() - 1   # noqa: E731
                a, b = yr(raw), yr(fixed)
                years = sorted({d.year for d, *_ in days})
                _simnote("VTSMX distribution repair (for VTISIM): " + "; ".join(
                    f"{d.date()} fund {f:+.2%} vs market {m:+.2%}" for d, f, m in days)
                    + " -> calendar years " + ", ".join(f"{y} {a[y]:.2%} -> {b[y]:.2%}" for y in years))
                note += " (early VTSMX ex-dates whose distribution Yahoo understates repaired, see data/sims_log.txt)"
        except Exception as e:  # noqa: BLE001 - the unrepaired fund
            _simlog(f"VTSMX distribution repair failed: {e}")
        build("VTISIM", ff["mkt"], ("VTSMX", "VTI"), note, "VTSMX")
    attempt("VTISIM", total_market)

    bonds = {}

    def treasuries():
        y = {sid: _fred(sid) for sid in ("DGS10", "DGS20", "DGS30", "DGS2")}
        long = y["DGS20"].combine_first((y["DGS10"] + y["DGS30"]) / 2).combine_first(y["DGS10"])
        build("TLTSIM", _bond_returns(long, 20), ("TLT",), "20-year Treasury priced off FRED yields, then TLT", "TLT")
        build("IEFSIM", _bond_returns(y["DGS10"], 9), ("IEF",), "9-year Treasury off the 10-year yield, then IEF", "IEF")
        short = y["DGS2"].combine_first(_fred("DGS1"))  # the 1-year yield stands in before the 2-year series (1976)
        build("SHYSIM", _bond_returns(short, 2), ("SHY",), "2-year Treasury off the 2-year yield (1-year before 1976), then SHY", "SHY")
    attempt("Treasuries", treasuries)

    def five_year():
        bonds["t5"] = _bond_returns(_fred("DGS5"), 5)
        build("IEISIM", bonds["t5"], ("IEI",), "5-year Treasury off the 5-year yield, then IEI", "IEI")
    attempt("IEISIM", five_year)

    # more US equity classes from Ken French's data library (value-weighted portfolios, daily)
    def size_value():
        p6 = _factor_file("port6_daily")
        g = {}
        try:
            g = port25_grid(_factor_file("port25_daily"))
        except Exception as e:  # noqa: BLE001 - the 6 portfolios alone
            _simlog(f"25 size x B/M portfolios unavailable: {e}")

        def mix(cells):   # equal mix of 25-portfolio cells (size quintile, B/M quintile), 1 = small / low B/M
            return pd.concat([g[c] for c in cells], axis=1).mean(axis=1) if g else None
        H, N, L = p6[_col(p6, "BIGHIBM")], p6[_col(p6, "ME2BM2")], p6[_col(p6, "BIGLOBM")]
        SH, SN, SL = p6[_col(p6, "SMALLHIBM")], p6[_col(p6, "ME1BM2")], p6[_col(p6, "SMALLLOBM")]
        # (ticker, candidates, real funds spliced in: the mutual-fund twin then the ETF, fund for the choice, label)
        specs = (
            ("VTVSIM", {"big/high B/M": H, "1/3 big/high + 2/3 big/neutral B/M": H / 3 + N * 2 / 3,
                        "25 portfolios: 2 largest size x 2 highest B/M quintiles": mix([(4, 4), (4, 5), (5, 4), (5, 5)]),
                        "25 portfolios: largest size x B/M quintiles 3-5": mix([(5, 3), (5, 4), (5, 5)])},
             ("VIVAX", "VTV"), "US large-cap value", "Vanguard Value Index fund (VIVAX)"),
            ("VUGSIM", {"big/low B/M": L, "2/3 big/low + 1/3 big/neutral B/M": L * 2 / 3 + N / 3,
                        "25 portfolios: 2 largest size x 2 lowest B/M quintiles": mix([(4, 1), (4, 2), (5, 1), (5, 2)])},
                       ("VIGRX", "VUG"), "US large-cap growth", "Vanguard Growth Index fund (VIGRX)"),
            ("VBRSIM", {"small/high B/M": SH, "1/2 small/high + 1/2 small/neutral B/M": (SH + SN) / 2,
                        "25 portfolios: size quintiles 2-3 x 2 highest B/M quintiles": mix([(2, 4), (2, 5), (3, 4), (3, 5)])},
             ("VISVX", "VBR"), "US small-cap value", "Vanguard Small-Cap Value Index fund (VISVX)"),
            ("VBKSIM", {"small/low B/M": SL, "1/2 small/low + 1/2 small/neutral B/M": (SL + SN) / 2,
                        "25 portfolios: size quintiles 2-3 x 2 lowest B/M quintiles": mix([(2, 1), (2, 2), (3, 1), (3, 2)])},
             ("VISGX", "VBK"), "US small-cap growth", "Vanguard Small-Cap Growth Index fund (VISGX)"),
            ("VBSIM", {"small portfolios of the 6 size x B/M": (SH + SN + SL) / 3,
                       "25 portfolios: size quintiles 2-3": mix([(q, b) for q in (2, 3) for b in range(1, 6)])},
             ("NAESX@1989-10-01", "VB"), "US small caps", "Vanguard Small-Cap Index fund (NAESX, an index fund from late 1989)"),
            ("VOESIM", {"25 portfolios: size quintiles 3-4 x 2 highest B/M quintiles": mix([(3, 4), (3, 5), (4, 4), (4, 5)]),
                        "25 portfolios: size quintile 3 x B/M quintiles 3-5": mix([(3, 3), (3, 4), (3, 5)])},
             ("VOE",), "US mid-cap value", None),
            ("VOTSIM", {"25 portfolios: size quintiles 3-4 x 2 lowest B/M quintiles": mix([(3, 1), (3, 2), (4, 1), (4, 2)]),
                        "25 portfolios: size quintile 3 x B/M quintiles 1-3": mix([(3, 1), (3, 2), (3, 3)])},
             ("VOT",), "US mid-cap growth", None),
        )
        for t, cands, reals, what, twin in specs:
            try:
                funds = tuple(r.partition("@")[0] for r in reals)
                label, model = _pick_model(t, cands, funds, since=reals[0].partition("@")[2] or None)
                etf = funds[-1]
                note = f"{what}: Fama-French {label} (daily from 1926)" + (f", then the {twin}" if twin else "") + f", then {etf}"
                # NAESX was an actively managed small-cap fund until late 1989: validate from 1990 only
                build(t, model, reals, note, funds[0], validate_since="1990-01-01" if t == "VBSIM" else None)
                if len(funds) > 1:
                    _validate(t, model, etf)
            except Exception as e:  # noqa: BLE001
                _simlog(f"sim {t} failed: {e}")
    attempt("size/value", size_value)

    def mid_caps():
        me = _factor_file("me_daily")
        # the middle 40% of NYSE market caps (30th-70th percentile breakpoints), value-weighted
        build("MIDSIM", me[_col(me, "MED40")], ("MDY",),
              "US mid caps (Fama-French portfolio of the 30th-70th NYSE size percentiles) from 1926, then MDY", "IJH")
    attempt("MIDSIM", mid_caps)

    def developed():
        dev = _factor_file("dev_ff3_daily")
        daily = (dev["Mkt-RF"] + dev["RF"]).dropna()
        note = "developed ex-US market (Fama-French, from 1990), then EFA"
        try:
            eafe = _monthly_steps(french_international_index("all"))   # 13+ developed markets ex-US, monthly USD, from 1975
            daily = pd.concat([eafe[eafe.index < daily.index[0]], daily])
            note = "developed ex-US: Fama-French EAFE index (monthly steps on the last session) from 1975, daily from 1990, then EFA"
        except Exception as e:  # noqa: BLE001
            _simlog(f"EAFE monthly failed: {e}")
        ff["efa"] = daily
        build("EFASIM", daily, ("EFA",), note, "EFA")
    attempt("EFASIM", developed)

    def europe():
        eu = _factor_file("eu_ff3_daily")
        daily = (eu["Mkt-RF"] + eu["RF"]).dropna()
        note = "European stocks (Fama-French Europe, daily from 1990), then VGK"
        try:
            m = _monthly_steps(french_international_index("eur_with_uk"))   # Europe incl. UK
            daily = pd.concat([m[m.index < daily.index[0]], daily])
            note = "European stocks: Fama-French Europe index (monthly steps) from 1975, daily from 1990, then VGK"
        except Exception as e:  # noqa: BLE001
            _simlog(f"Europe monthly failed: {e}")
        build("VGKSIM", daily, ("VGK",), note, "VGK")
    attempt("VGKSIM", europe)

    def intl_size_value():
        d6 = _factor_file("dev_port6_daily")
        small = [c for c in d6.columns if c.upper().replace(" ", "").startswith(("SMALL", "ME1"))]
        build("SCZSIM", d6[small].mean(axis=1), ("SCZ",),
              "developed ex-US small caps (Fama-French small portfolios, daily from 1990), then SCZ", "SCZ")
        build("AVDVSIM", d6[_col(d6, "SMALLHIBM")], ("AVDV",),
              "developed ex-US small-cap value (Fama-French small/high B/M, daily from 1990), then AVDV", "AVDV")
        daily = d6[_col(d6, "BIGHIBM")].dropna()
        note = "developed ex-US large-cap value (Fama-French big/high B/M, daily from 1990), then EFV"
        try:
            m = _monthly_steps(french_international_index("all", column=("HIGH",)))   # first "High" = high BE/ME
            daily = pd.concat([m[m.index < daily.index[0]], daily])
            note = ("developed ex-US value: Fama-French EAFE high book-to-market index (monthly steps) from 1975, "
                    "big/high B/M daily from 1990, then EFV")
        except Exception as e:  # noqa: BLE001
            _simlog(f"EAFE value monthly failed: {e}")
        build("EFVSIM", daily, ("EFV",), note, "EFV")
    attempt("international size/value", intl_size_value)

    def corporates():
        # investment-grade corporates: a 10-year par bond at the average of Moody's Aaa and Baa yields
        # (monthly averages until both daily series exist, daily after: see corporate_yield)
        corp = corporate_yield(_fred("DAAA"), _fred("DBAA"), _fred("AAA"), _fred("BAA"))
        bonds["corp"] = _bond_returns(corp, 10)
        build("LQDSIM", bonds["corp"], ("LQD",), "investment-grade corporates priced off Moody's Aaa/Baa yields, then LQD", "LQD")
    attempt("LQDSIM", corporates)

    def total_bond():
        # the US aggregate bond market: before the Vanguard Total Bond Market Index fund (VBMFX, Dec 1986)
        # a 70/30 blend of 5-year Treasuries and IG corporates (duration about 5.5, like the Agg; no
        # mortgages: no free long MBS total-return series), then VBMFX, then BND
        blend = (0.7 * bonds["t5"] + 0.3 * bonds["corp"]).dropna()
        build("BNDSIM", blend, ("VBMFX", "BND"),
              "US aggregate bonds: 70% 5-year Treasury + 30% IG corporate model before Dec 1986, "
              "then the Vanguard Total Bond Market Index fund (VBMFX), then BND", "VBMFX")
    attempt("BNDSIM", total_bond)

    def tips():
        # TIPS were first issued in 1997: a MODEL before the Vanguard Inflation-Protected Securities fund (VIPSX,
        # June 2000) - an 8-year real par bond off an estimated real yield plus lagged CPI accrual (tips_model)
        m, info = tips_model(_fred("GS10"), _fred("CPIAUCNS"), _fred("REAINTRATREARAT10Y"))
        _simnote(f"TIPSIM model: {len(m)} months {m.index[0]}..{m.index[-1]}; real yield = Cleveland Fed 10-year real "
                 f"rate from {info['first_real_month']}, before that GS10 minus trailing {info['proxy_window']}-month CPI "
                 "inflation (monthly-change RMSE vs the Cleveland rate 1982-99: "
                 + ", ".join(f"{k} months {v:.3f}" for k, v in info["change_rmse"].items())
                 + f"); {info['maturity']:g}-year real par bond + CPI-U accrual lagged 3 months")
        steps = _monthly_steps(pd.Series(m.to_numpy(), index=m.index.to_timestamp(how="end").normalize()))
        _validate("TIPSIM model", steps, "TIP")
        build("TIPSIM", steps, ("VIPSX", "TIP"),
              "US TIPS: MODEL (monthly steps) from 1972 - an 8-year real par bond priced off the Cleveland Fed 10-year "
              "real rate (from 1982; before, the 10-year Treasury yield minus trailing CPI inflation) plus CPI accrual "
              "lagged 3 months - then the Vanguard Inflation-Protected Securities fund (VIPSX) from mid-2000, then TIP",
              "VIPSX")
    attempt("TIPSIM", tips)

    def high_yield():
        # FRED's ICE BofA high-yield total-return index only covers the last three years; the Vanguard
        # High-Yield Corporate fund (VWEHX, Yahoo history from 1980) is the longest free record. Before it, a
        # factor-mimicking MODEL: a 7-year par bond at Moody's Baa yield (the lowest investment grade; credit
        # carry and rate risk) mixed with the US stock market (the equity-like default risk of junk bonds: HY fell
        # 20-30% in 2008 when Baa bonds did not), the stock share picked by tracking error against VWEHX; the
        # CAGR gap on the overlap (defaults and fees) is taken off as the model's drag.
        baa = corporate_yield(_fred("DBAA"), _fred("DBAA"), _fred("BAA"), _fred("BAA"))
        bond = _bond_returns(baa, 7)
        mkt = ff.get("mkt")
        if mkt is None:
            f = _factor_file("ff3_daily")
            mkt = (f["Mkt-RF"] + f["RF"]).dropna()
        days = bond.index.intersection(mkt.index)
        cands = {f"{1 - w:.0%} Baa 7-year par bond + {w:.0%} US stock market": (1 - w) * bond.reindex(days) + w * mkt.reindex(days)
                 for w in (0.0, 0.2, 0.25, 0.3, 0.35)}
        label, model = _pick_model("HYGSIM", cands, ("VWEHX",))
        build("HYGSIM", model, ("VWEHX", "HYG"),
              f"US high-yield bonds: MODEL from 1953 ({label}, a factor mimic net of a default-loss/fee haircut "
              "calibrated on the overlap), then the Vanguard High-Yield Corporate fund (VWEHX) from 1980, then HYG", "VWEHX")
        _validate("HYGSIM model", model, "HYG")
    attempt("HYGSIM", high_yield)

    def unhedged_intl_bonds():
        m = unhedged_bond_monthly()
        steps = _monthly_steps(m)
        _validate("BWXSIM model", steps, "IGOV")
        build("BWXSIM", steps, ("BWX",),
              "international government bonds, NOT hedged: OECD 10-year yields of up to 12 developed markets (9-year "
              "par bonds) converted to USD at month-end exchange rates (monthly steps) from 1971, then BWX", "BWX")
    attempt("BWXSIM", unhedged_intl_bonds)

    def intl_bonds():
        m = intl_bond_monthly()
        steps = _monthly_steps(m)
        _validate("BNDXSIM model", steps, "BNDX")
        build("BNDXSIM", steps, ("PFORX", "BNDX"),
              "international government bonds hedged to USD: OECD 10-year yields of up to 12 developed markets "
              "(par-bond model, monthly steps) until 1993, then the PIMCO International Bond (USD-hedged) fund "
              "PFORX, then BNDX", "PFORX")
    attempt("BNDXSIM", intl_bonds)

    def emerging():
        em = _factor_file("em_ff5_monthly")
        r = (em["Mkt-RF"] + em["RF"]).dropna()
        ff["eem"] = _monthly_steps(r)
        build("EEMSIM", ff["eem"], ("EEM",), "emerging markets (Fama-French, monthly steps, from 1989), then EEM", "EEM")
        build("VWOSIM", ff["eem"], ("VEIEX", "VWO"),
              "emerging markets: Fama-French (monthly steps) from 1989, then the Vanguard Emerging Markets Stock "
              "Index fund (VEIEX) from 1994, then VWO", "VEIEX")
    attempt("EEMSIM", emerging)

    def total_international():
        # all-world ex-US: developed ex-US (EAFE) 80% and emerging 20% (about VXUS's split), rebalanced daily;
        # before the emerging-markets series (1989) developed alone. Then the Vanguard Total International
        # Stock Index fund (VGTSX, April 1996), then VXUS (January 2011)
        efa = ff["efa"]
        eem = ff.get("eem", pd.Series(dtype=float))
        days = efa.index.union(eem.index)
        e, m = efa.reindex(days).fillna(0.0), eem.reindex(days).fillna(0.0)
        blend = e.copy()
        if len(eem):
            after = days >= eem.index[0]
            blend[after] = 0.8 * e[after] + 0.2 * m[after]
        build("VXUSSIM", blend, ("VGTSX", "VXUS"),
              "international stocks: 80% developed ex-US (Fama-French EAFE) + 20% emerging (from 1989) until 1996, "
              "then the Vanguard Total International Stock Index fund (VGTSX), then VXUS from 2011", "VGTSX")
    attempt("VXUSSIM", total_international)

    def reits():
        m = _monthly_steps(nareit_monthly())
        _validate("VNQSIM model", m, "VNQ")
        build("VNQSIM", m, ("VGSIX", "VNQ"),
              "US REITs: FTSE Nareit All Equity REITs total return (monthly steps) from 1972, then the Vanguard REIT "
              "Index fund (VGSIX) from 1996, then VNQ", "VGSIX")
    attempt("VNQSIM", reits)

    def gold():
        wb = None
        try:
            wb = gold_monthly()
        except Exception as e:  # noqa: BLE001
            _simlog(f"World Bank gold failed: {e}")
        try:
            lbma = lbma_gold_daily()
            daily = _on_sessions(lbma)
            note = "gold: LBMA PM fixing (daily, USD) from 1968"
            if wb is not None:
                early = _monthly_level_steps(wb)
                daily = pd.concat([early[early.index < lbma.index[0]], daily])
                note = "gold: World Bank monthly average price (monthly steps) 1960-1968, LBMA PM fixing daily from April 1968"
        except Exception as e:  # noqa: BLE001
            _simlog(f"LBMA gold failed, using World Bank monthly averages: {e}")
            if wb is None:
                raise
            daily = _monthly_level_steps(wb)
            note = "gold: World Bank monthly average price (monthly steps) from 1960"
        build("GLDSIM", daily, ("GLD",), note + ", then GLD", "GLD")
    attempt("GLDSIM", gold)

    def commodities():
        # AQR "Commodities for the Long Run": an equal-weight portfolio of commodity futures (excess
        # return over T-bills, monthly from 1877) plus the T-bill return = a fully collateralised index
        ex = aqr_commodities_monthly()
        rf = ff.get("rf")
        if rf is None:
            rf = _factor_file("ff3_daily")["RF"].dropna()
        rfm = (1 + rf).groupby(rf.index.to_period("M")).prod() - 1
        ex.index = pd.DatetimeIndex(ex.index).to_period("M")
        tr = (ex + rfm.reindex(ex.index)).dropna()
        tr = tr[tr.index >= pd.Period("1960-01", "M")]
        model = _monthly_steps(tr)
        reals, note = ("DBC",), ("commodity futures: AQR equal-weight commodity index excess return + T-bills (monthly "
                                 "steps) from 1960, then DBC")
        # a real commodity fund before DBC (Feb 2006): the PIMCO CommodityRealReturn Strategy fund (PCRIX, June
        # 2002; Bloomberg Commodity Index futures collateralised with TIPS), used only if it tracks DBC better than
        # the model does on their common overlap
        te_model = _tracking_error(model, "DBC", since="2006-03-01")
        te_fund = _tracking_error(_real_returns("PCRIX"), "DBC", since="2006-03-01")
        _simnote(f"DBCSIM: tracking error vs DBC from 2006: AQR model {te_model[0] if te_model else float('nan'):.2%}/yr, "
                 f"PCRIX {te_fund[0] if te_fund else float('nan'):.2%}/yr")
        if te_fund and (not te_model or te_fund[0] < te_model[0]):
            reals = ("PCRIX", "DBC")
            note = note.replace("then DBC", "then the PIMCO CommodityRealReturn Strategy fund (PCRIX) from mid-2002, then DBC")
        build("DBCSIM", model, reals, note, "DBC")
    attempt("DBCSIM", commodities)

    def long_corporates():
        # long-term investment-grade corporates (VCLT / IGLB): a 20-year par bond at the average of Moody's Aaa
        # and Baa yields (Moody's seasoned yields are themselves 20-30 year bonds), then the Vanguard Long-Term
        # Investment-Grade fund (VWESX, 1973), then VCLT (2009)
        model = _bond_returns(moody_yield(), 20)
        _validate("VCLTSIM model", model, "VCLT")
        build("VCLTSIM", model, ("VWESX", "VCLT"),
              "long-term IG corporates: 20-year par bond at Moody's Aaa/Baa average yield (monthly averages before "
              "1986) until the Vanguard Long-Term Investment-Grade fund (VWESX), then VCLT from 2009", "VWESX")
    attempt("VCLTSIM", long_corporates)

    def munis():
        # municipal bonds: no free long muni total-return index, so real funds only - the Vanguard
        # Intermediate-Term Tax-Exempt fund (VWITX, 1977), then MUB (2007)
        r = _real_returns("VWITX")
        _validate("MUBSIM (VWITX)", r, "MUB")
        build("MUBSIM", r, ("MUB",), "US municipal bonds: Vanguard Intermediate-Term Tax-Exempt fund (VWITX) from "
              "its Yahoo history, then MUB from 2007 (no model before)", model=False)
    attempt("MUBSIM", munis)

    def em_bonds():
        # emerging-market USD bonds: the Fidelity New Markets Income fund (FNMIX, 1993), then EMB (Dec 2007)
        r = _real_returns("FNMIX")
        _validate("EMBSIM (FNMIX)", r, "EMB")
        build("EMBSIM", r, ("EMB",), "emerging-market USD bonds: Fidelity New Markets Income fund (FNMIX) from 1993, "
              "then EMB (no model before)", model=False)
    attempt("EMBSIM", em_bonds)

    # single countries: Fama-French country indexes (USD, value-weighted, dividends; monthly from 1975),
    # Japan daily from 1990, then the iShares country ETF
    for t, member, country in COUNTRY_SIMS:
        def one(t=t, member=member, country=country):
            m = _monthly_steps(french_country_index(member))
            note = f"{country} stocks: Fama-French {country} index (monthly steps) from 1975"
            if t == "EWJSIM":
                try:
                    j = _factor_file("japan_ff3_daily")
                    d = (j["Mkt-RF"] + j["RF"]).dropna()
                    m = pd.concat([m[m.index < d.index[0]], d])
                    note += ", Fama-French Japan daily from 1990"
                except Exception as e:  # noqa: BLE001
                    _simlog(f"Japan daily failed: {e}")
            etf = t[:-3]
            build(t, m, (etf,), note + f", then {etf}", etf)
        attempt(t, one)
    return made


# (series, member of F-F_International_Countries.zip, country); member names as in the zip
COUNTRY_SIMS = (("EWJSIM", "Japan.Dat", "Japan"), ("EWUSIM", "UK.Dat", "UK"), ("EWGSIM", "Germany.Dat", "Germany"),
                ("EWCSIM", "Canada.Dat", "Canada"), ("EWASIM", "Austrlia.Dat", "Australia"),
                ("EWQSIM", "France.Dat", "France"), ("EWLSIM", "Swtzrlnd.Dat", "Switzerland"),
                ("EWHSIM", "HongKong.Dat", "Hong Kong"))
_ZIPS: dict[str, object] = {}


def french_country_index(member: str) -> pd.Series:
    """Monthly USD total return (decimal) of a country's market in Fama-French's international country
    portfolios (F-F_International_Countries.zip: Japan.Dat, UK.Dat, Germany.Dat, ..., 1975 onward)."""
    import zipfile
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_International_Countries.zip"
    if url not in _ZIPS:
        _ZIPS[url] = zipfile.ZipFile(io.BytesIO(requests.get(url, headers=UA, timeout=120).content))
    z = _ZIPS[url]
    names = z.namelist()
    pick = next((n for n in names if n.lower() == member.lower() or n.lower().endswith("/" + member.lower())), None)
    if pick is None:
        raise RuntimeError(f"{member} not in {names}")
    s = parse_french_dat(z.read(pick).decode("latin-1"), ("MKT",), label=pick)
    _simnote(f"French country index {pick}: {len(s)} months {s.index[0]:%Y-%m}..{s.index[-1]:%Y-%m}")
    return s


def moody_yield() -> pd.Series:
    """The average of Moody's seasoned Aaa and Baa corporate yields (percent) on every NYSE session from
    1953: the daily series (DAAA/DBAA, from 1986) where they exist, before that the monthly averages
    (AAA/BAA) held from the last session of their month."""
    daily = ((_fred("DAAA") + _fred("DBAA")) / 2).dropna()
    monthly = ((_fred("AAA") + _fred("BAA")) / 2).dropna()
    monthly = monthly[(monthly.index >= "1953-01-01") & (monthly.index < daily.index[0])]
    per = pd.DatetimeIndex(monthly.index).to_period("M")
    at = _last_sessions(per[0].start_time, per[-1].end_time).reindex(per)
    mm = pd.Series(monthly.to_numpy(), index=pd.DatetimeIndex(at.to_numpy()))
    mm = mm[mm.index.notna()]
    sess = _sessions(mm.index[0], daily.index[-1])
    lvl = pd.concat([mm, daily]).sort_index()
    lvl = lvl[~lvl.index.duplicated(keep="last")]
    return lvl.reindex(lvl.index.union(sess)).ffill().reindex(sess).dropna()


def port25_grid(df: pd.DataFrame) -> dict:
    """{(size quintile, B/M quintile): daily returns} from French's 25 portfolios formed on size and
    book-to-market (25_Portfolios_5x5_Daily_CSV.zip). Columns are named 'SMALL LoBM', 'ME1 BM2', ...,
    'ME2 BM1', ..., 'BIG HiBM' (size outer, B/M inner); names that don't parse fall back to that order."""
    cols = [c for c in df.columns if c != "date"]
    if len(cols) != 25:
        raise RuntimeError(f"expected 25 portfolio columns, got {len(cols)}: {cols[:6]}")
    out = {}
    for i, c in enumerate(cols):
        k = c.upper().replace(" ", "")
        m = re.fullmatch(r"(SMALL|BIG|ME(\d))(LOBM|HIBM|BM(\d))", k)
        if m:
            size = 1 if m.group(1) == "SMALL" else 5 if m.group(1) == "BIG" else int(m.group(2))
            bm = 1 if m.group(3) == "LOBM" else 5 if m.group(3) == "HIBM" else int(m.group(4))
        else:
            size, bm = i // 5 + 1, i % 5 + 1
        out[(size, bm)] = pd.to_numeric(df[c], errors="coerce")
    if len(out) != 25:
        raise RuntimeError(f"25 portfolios: columns {cols} don't form a 5 x 5 grid")
    return out


def french_international_index(name: str, column=("MKT",)) -> pd.Series:
    """Monthly value-weighted return (USD, decimal) of a Fama-French international index (e.g. EAFE
    'all', Europe) from F-F_International_Indices.zip, 1975 onward. `column`: accepted names of the
    column, upper-cased without spaces or punctuation (the market, or e.g. HIBM for value)."""
    import zipfile
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_International_Indices.zip"
    z = zipfile.ZipFile(io.BytesIO(requests.get(url, headers=UA, timeout=120).content))
    names = z.namelist()
    cands = [n for n in names if f"_{name.lower()}." in n.lower() or n.lower().startswith(name.lower())]
    if not cands:
        raise RuntimeError(f"{name} not in {names[:20]}")
    pick = min(cands, key=len)   # 'Europe...' rather than 'Europe x UK...'
    s = parse_french_dat(z.read(pick).decode("latin-1"), column, label=pick)
    _simnote(f"French international index {name!r} column {column[0]}: {pick}, {len(s)} months {s.index[0]:%Y-%m}..{s.index[-1]:%Y-%m}")
    return s


def parse_french_dat(text: str, column=("MKT",), label: str = "") -> pd.Series:
    """The first table of a French .Dat text file (whitespace or comma separated, YYYYMM rows, percent)."""
    raw = text.splitlines()

    def cells(line):
        return [x for x in re.split(r"[\s,]+", line.strip()) if x]

    def norm(h):
        return re.sub(r"[^A-Z0-9]", "", h.upper())
    want = {norm(c) for c in column}
    hdr_i = next((i for i, l in enumerate(raw) if any(norm(c) in want for c in cells(l))), None)
    if hdr_i is None:
        heads = [l.strip()[:120] for l in raw if l.strip() and not re.match(r"^\s*\d{6}", l)][:12]
        raise RuntimeError(f"{label}: no column named {sorted(want)}; header lines {heads}")
    header = cells(raw[hdr_i])
    out = {}
    for l in raw[hdr_i + 1:]:
        parts = cells(l)
        if not parts or not re.fullmatch(r"\d{6}", parts[0]):
            if out:
                break
            continue
        vals = parts[1:]
        # the header may or may not name the date column
        hdr = header[1:] if len(header) == len(parts) else header
        col = next(i for i, h in enumerate(hdr) if norm(h) in want)
        try:
            v = float(vals[col])
        except (IndexError, ValueError):
            continue
        if v <= -99:
            continue
        out[pd.Timestamp(parts[0][:4] + "-" + parts[0][4:] + "-01") + pd.offsets.MonthEnd(0)] = v / 100
    if not out:
        raise RuntimeError(f"{label}: no monthly rows")
    print(f"french {label}: {len(out)} months")
    return pd.Series(out).sort_index()


def nareit_monthly() -> pd.Series:
    """FTSE Nareit All Equity REITs monthly total return (decimal) from 1972 (reit.com)."""
    url = "https://www.reit.com/sites/default/files/returns/MonthlyHistoricalReturns.xls"
    content = requests.get(url, headers=UA, timeout=120).content
    book = pd.read_excel(io.BytesIO(content), sheet_name=None, header=None)
    for name, raw in book.items():
        txt = raw.map(lambda v: "" if v is None or (isinstance(v, float) and v != v) else str(v))
        hits = [(r, c) for r in range(min(len(raw), 15)) for c in range(raw.shape[1])
                if "all equity" in txt.iat[r, c].lower()]
        if not hits:
            continue
        r0, c0 = hits[0]
        # the total-return column under "All Equity REITs": first header in the next rows mentioning return
        cands = []
        for c in range(c0, min(c0 + 8, raw.shape[1])):
            head = " ".join(txt.iat[r, c].lower() for r in range(r0, min(r0 + 4, len(raw))))
            if "total" in head and "return" in head:
                cands.append(c)
        if not cands:
            print(f"nareit {name}: headers near row {r0}: {[txt.iat[r0 + 1, c] for c in range(c0, min(c0 + 8, raw.shape[1]))]}")
            continue
        col = cands[0]
        dates = pd.to_datetime(raw.iloc[r0 + 1:, 0], errors="coerce")
        vals = pd.to_numeric(raw.iloc[r0 + 1:, col], errors="coerce")
        ser = pd.Series(vals.to_numpy(), index=dates.to_numpy()).dropna()
        ser = ser[~ser.index.isna()].sort_index()
        ser.index = pd.DatetimeIndex(ser.index) + pd.offsets.MonthEnd(0)
        if ser.min() > 0 and ser.iloc[-1] > 10 * ser.iloc[0]:
            ser = ser.pct_change().dropna()          # an index level (1971-12 = 100)
        elif ser.abs().median() > 0.5:
            ser = ser / 100                          # monthly returns in percent
        print(f"nareit {name}: column {col}, {len(ser)} months {ser.index[0].date()}..{ser.index[-1].date()}")
        return ser
    raise RuntimeError(f"All Equity REITs column not found in sheets {list(book)}")


def _pink_sheet(sheet: str) -> pd.DataFrame:
    page = requests.get("https://www.worldbank.org/en/research/commodity-markets", headers=UA, timeout=60).text
    m = re.search(r'https://thedocs\.worldbank\.org/[^"\']+CMO-Historical-Data-Monthly\.xlsx', page)
    if not m:
        raise RuntimeError("Pink Sheet link not found")
    return pd.read_excel(io.BytesIO(requests.get(m.group(0), headers=UA, timeout=120).content), sheet_name=sheet, header=None)


def commodity_monthly() -> pd.Series:
    """World Bank monthly commodity price index (energy and non-energy, 2010=100), from 1960."""
    raw = _pink_sheet("Monthly Indices")
    hdr = next(i for i in range(min(len(raw), 20)) if any("energy" in str(v).lower() for v in raw.iloc[i]))
    cols = {str(v).strip().lower(): j for j, v in enumerate(raw.iloc[hdr])}
    pick = [j for k, j in cols.items() if k in ("energy", "non-energy", "non energy")]
    if not pick:
        raise RuntimeError(f"energy/non-energy columns not found: {list(cols)[:12]}")
    rows = raw.iloc[hdr + 1:]
    rows = rows[rows[0].astype(str).str.fullmatch(r"\d{4}M\d{2}")]
    idx = pd.to_datetime(rows[0].str.replace("M", "-") + "-01") + pd.offsets.MonthEnd(0)
    vals = rows[pick].apply(pd.to_numeric, errors="coerce").mean(axis=1).to_numpy()
    return pd.Series(vals, index=idx).dropna()


def gold_monthly() -> pd.Series:
    """Monthly AVERAGE gold price (USD/oz) from the World Bank 'Pink Sheet' historical data (1960 on).
    Used only before the daily LBMA fixing starts (April 1968), or if that source fails."""
    raw = _pink_sheet("Monthly Prices")
    hdr = next(i for i in range(min(len(raw), 20)) if any(str(v).strip() == "Gold" for v in raw.iloc[i]))
    gcol = next(j for j, v in enumerate(raw.iloc[hdr]) if str(v).strip() == "Gold")
    rows = raw.iloc[hdr + 1:]
    rows = rows[rows[0].astype(str).str.fullmatch(r"\d{4}M\d{2}")]
    idx = pd.to_datetime(rows[0].str.replace("M", "-") + "-01") + pd.offsets.MonthEnd(0)
    return pd.Series(pd.to_numeric(rows[gcol], errors="coerce").to_numpy(), index=idx).dropna()


LBMA_GOLD_URL = "https://prices.lbma.org.uk/json/gold_pm.json"


def parse_lbma_json(text: str) -> pd.Series:
    """LBMA price JSON ([{"d": "1968-04-01", "v": [USD, GBP, EUR]}, ...]) -> daily USD price series."""
    rows = json.loads(text)
    out = {}
    for row in rows:
        try:
            v = row["v"][0]
            if v is not None and float(v) > 0:
                out[pd.Timestamp(row["d"])] = float(v)
        except (KeyError, IndexError, TypeError, ValueError):
            continue
    if len(out) < 1000:
        raise RuntimeError(f"LBMA JSON: only {len(out)} usable rows")
    s = pd.Series(out).sort_index()
    return s[~s.index.duplicated(keep="last")]


def lbma_gold_daily() -> pd.Series:
    """The LBMA Gold Price PM (the London afternoon fixing), USD per troy ounce, daily from 1968-04-01."""
    headers = {**UA, "Referer": "https://www.lbma.org.uk/prices-and-data/precious-metal-prices",
               "Accept": "application/json,text/plain,*/*"}
    r = requests.get(LBMA_GOLD_URL, headers=headers, timeout=60)
    r.raise_for_status()
    s = parse_lbma_json(r.text)
    print(f"LBMA gold: {len(s)} days {s.index[0].date()} .. {s.index[-1].date()}")
    return s


AQR_COMMODITIES_URL = ("https://www.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/"
                       "Commodities-for-the-Long-Run-Index-Level-Data-Monthly.xlsx")


def parse_aqr_commodities(raw: pd.DataFrame) -> pd.Series:
    """The first sheet of AQR's 'Commodities for the Long Run: Index Level Data, Monthly' (read with
    header=None): a header row naming 'Excess return of equal-weight commodities portfolio', then
    rows of month-end date and returns (fractions, or strings like '-8.26%'). Returns the equal-weight
    excess return by month end."""
    txt = raw.map(lambda v: "" if v is None or (isinstance(v, float) and v != v) else str(v).lower())
    hit = next(((r, c) for r in range(min(len(raw), 40)) for c in range(raw.shape[1])
                if "excess return" in txt.iat[r, c] and "equal" in txt.iat[r, c]), None)
    if hit is None:
        raise RuntimeError("AQR: 'Excess return of equal-weight' header not found")
    r0, col = hit

    def num(v):
        if isinstance(v, str):
            v = v.strip().replace("−", "-")
            pct = v.endswith("%")
            try:
                x = float(v.rstrip("%"))
            except ValueError:
                return np.nan
            return x / 100 if pct else x
        try:
            return float(v)
        except (TypeError, ValueError):
            return np.nan
    dates = pd.to_datetime(raw.iloc[r0 + 1:, 0], errors="coerce")
    vals = raw.iloc[r0 + 1:, col].map(num)
    s = pd.Series(vals.to_numpy(dtype=float), index=dates.to_numpy()).dropna()
    s = s[~s.index.isna()].sort_index()
    if len(s) < 120:
        raise RuntimeError(f"AQR: only {len(s)} monthly rows")
    if s.abs().median() > 0.5:          # percent numbers without a % sign
        s = s / 100
    s.index = pd.DatetimeIndex(s.index) + pd.offsets.MonthEnd(0)
    return s[~s.index.duplicated(keep="last")]


def aqr_commodities_monthly() -> pd.Series:
    r = requests.get(AQR_COMMODITIES_URL, headers=UA, timeout=120)
    r.raise_for_status()
    raw = pd.read_excel(io.BytesIO(r.content), sheet_name=0, header=None)
    s = parse_aqr_commodities(raw)
    print(f"AQR commodities: {len(s)} months {s.index[0].date()} .. {s.index[-1].date()}")
    return s


def intl_bond_model(long: dict, short: dict, us_short: pd.Series,
                    weights: dict | None = None, maturity: float = 9) -> pd.Series:
    """Monthly USD-hedged return of a developed-market government bond index.

    Per country: a `maturity`-year par bond priced off the 10-year yield (percent, one value per
    month), return = coupon carry + price change; hedged to USD by covered interest parity, i.e. plus
    the US short rate minus the local short rate (last month's rates, known when the hedge is set).
    Countries are weighted by `weights`, renormalised over those with data that month."""
    weights = weights or dict(INTL_BONDS)
    rets = {}
    for c, y in long.items():
        y = pd.to_numeric(y, errors="coerce").dropna()
        s = short.get(c)
        if s is None or len(y) < 13:
            continue
        y.index = pd.DatetimeIndex(y.index).to_period("M")
        y = y[~y.index.duplicated(keep="last")]
        # _bond_returns needs dates: month ends
        yd = pd.Series(y.to_numpy(), index=y.index.to_timestamp(how="end").normalize())
        local = _bond_returns(yd, maturity, step_years=1 / 12)
        local.index = local.index.to_period("M")
        s = pd.to_numeric(s, errors="coerce").dropna()
        s.index = pd.DatetimeIndex(s.index).to_period("M")
        s = s[~s.index.duplicated(keep="last")]
        s = s.reindex(pd.period_range(s.index[0], y.index[-1], freq="M")).ffill(limit=3)
        rets[c] = (local - s.shift(1).reindex(local.index) / 1200).dropna()
    if not rets:
        raise RuntimeError("no country has both long and short rates")
    df = pd.DataFrame(rets).sort_index()
    w = pd.DataFrame({c: np.where(df[c].notna(), weights.get(c, 0.0), 0.0) for c in df.columns}, index=df.index)
    tot = w.sum(axis=1)
    local_mix = (df.fillna(0.0) * w).sum(axis=1) / tot.where(tot > 0)
    us = pd.to_numeric(us_short, errors="coerce").dropna()
    us.index = pd.DatetimeIndex(us.index).to_period("M")
    us = us[~us.index.duplicated(keep="last")]
    us = us.reindex(pd.period_range(us.index[0], df.index[-1], freq="M")).ffill(limit=3)
    out = (local_mix + us.shift(1).reindex(df.index) / 1200)
    # at least three countries (a third of the weight) for a diversified index
    ok = ((df.notna().sum(axis=1) >= 3) & (tot >= 0.3))
    out = out[ok].dropna()
    if out.empty:
        raise RuntimeError("no month with enough countries")
    return out


def intl_bond_monthly() -> pd.Series:
    long, short = {}, {}
    for c, _ in INTL_BONDS:
        try:
            long[c] = _fred(f"IRLTLT01{c}M156N")
        except Exception as e:  # noqa: BLE001
            print(f"intl bonds: no long yield for {c}: {e}")
            continue
        s = None
        for sid in (f"IR3TIB01{c}M156N", f"IRSTCI01{c}M156N"):   # 3-month interbank, else call money
            try:
                x = _fred(sid)
                s = x if s is None else s.combine_first(x)
            except Exception:  # noqa: BLE001
                pass
        if s is not None:
            short[c] = s
    m = intl_bond_model(long, short, _fred("TB3MS"))
    used = sorted(set(long) & set(short))
    _simnote(f"BNDXSIM model: {len(m)} months {m.index[0]}..{m.index[-1]}, countries {used}")
    return pd.Series(m.to_numpy(), index=m.index.to_timestamp(how="end").normalize())


# ---- TIPS before 1997: a real-yield model (TIPSIM)
#
# A TIPS fund earns (1) the return of a real par bond priced off the real yield and (2) the inflation accrual of
# its principal. The model:
#   real yield: the Cleveland Fed's 10-year real interest rate (FRED REAINTRATREARAT10Y, monthly, from 1982; a
#     model estimate from Treasury yields, inflation, swaps and surveys). Before 1982: the 10-year Treasury yield
#     (GS10) minus trailing CPI inflation, the trailing window (1, 3, 5 or 10 years) picked by how closely the
#     monthly changes of GS10 - trailing inflation match those of the Cleveland real rate over 1982-1999
#     (logged; the longer windows win: a 1-year window swings far more than any real yield).
#     The Cleveland value dated the 1st of month m is read as the real yield at the end of month m-1.
#   price: an 8-year real par bond (duration about 7, like VIPSX / TIP) re-priced monthly off that yield
#     (_bond_returns), plus its real coupon.
#   inflation accrual: TIPS principal follows CPI-U (not seasonally adjusted) with a 3-month lag, so the
#     accrual over month m is CPI(m-2) / CPI(m-3) - 1.
# Returns are spliced, not yields: the months before the first Cleveland month are priced off the proxy yield,
# the rest off the Cleveland yield, so the switch adds no jump. The model is a model: before 1997 no US
# inflation-linked bond existed, and the pre-1982 real yield is a rough estimate (the 1970s' trailing inflation
# moved far faster than any expectation). Validated against VIPSX and TIP on their overlap (data/sims_log.txt).

TIPS_MATURITY = 8.0
TIPS_START = "1972-01"
TIPS_PROXY_WINDOWS = (12, 36, 60, 120)


def _monthly(s: pd.Series) -> pd.Series:
    """A monthly series indexed by month (Period 'M'), last value per month."""
    s = pd.to_numeric(s, errors="coerce").dropna().sort_index()
    idx = s.index if isinstance(s.index, pd.PeriodIndex) else pd.DatetimeIndex(s.index).to_period("M")
    out = pd.Series(s.to_numpy(), index=idx)
    return out[~out.index.duplicated(keep="last")]


def trailing_inflation(cpi: pd.Series, months: int) -> pd.Series:
    """Annualised CPI inflation (percent) over the `months` months to the month before (known at month m)."""
    c = _monthly(cpi)
    c = c.reindex(pd.period_range(c.index[0], c.index[-1], freq="M")).interpolate()
    return (((c / c.shift(months)) ** (12 / months) - 1) * 100).shift(1)


def _real_bond_monthly(real_yield: pd.Series, maturity: float) -> pd.Series:
    """Monthly return of a `maturity`-year par bond re-priced monthly off `real_yield` (percent, monthly)."""
    y = _monthly(real_yield)
    yd = pd.Series(y.to_numpy(), index=y.index.to_timestamp(how="end").normalize())
    r = _bond_returns(yd, maturity, step_years=1 / 12)
    r.index = r.index.to_period("M")
    return r.dropna()


def tips_model(gs10: pd.Series, cpi: pd.Series, real10: pd.Series, maturity: float = TIPS_MATURITY,
               start: str = TIPS_START) -> tuple[pd.Series, dict]:
    """Monthly model return of a TIPS index (PeriodIndex) and a description of the choices made.
    gs10: 10-year Treasury yield (percent, monthly); cpi: CPI-U NSA level (monthly); real10: the Cleveland Fed
    10-year real rate (percent, monthly, from 1982)."""
    g, c, rr = _monthly(gs10), _monthly(cpi), _monthly(real10)
    if rr.empty:
        raise RuntimeError("no real-rate series")
    # the Cleveland estimate dated the 1st of month m is built from data through the end of month m-1 (it is
    # published at the start of the month): it is that month-end's real yield (checked: its bond returns
    # correlate 0.63 with VIPSX's in the same month when shifted, 0.03 when not)
    rr.index = rr.index - 1
    lo, hi = rr.index[0], min(rr.index[-1], pd.Period("1999-12", "M"))
    scores = {}
    for k in TIPS_PROXY_WINDOWS:
        # scored on monthly CHANGES (what drives the bond's return), not levels: the 1-year window matches the
        # level best but swings far more than any real yield (1972-81 real-bond returns of -40% and +40% a year)
        proxy = (g - trailing_inflation(c, k)).dropna()
        both = pd.concat({"p": proxy, "r": rr}, axis=1).loc[lo:hi].dropna()
        if len(both) >= 60:
            scores[k] = float(((both["p"].diff() - both["r"].diff()) ** 2).mean() ** 0.5)
    if not scores:
        raise RuntimeError("no overlap between GS10 - trailing inflation and the real rate")
    k = min(scores, key=scores.get)
    proxy = (g - trailing_inflation(c, k)).dropna()
    early = _real_bond_monthly(proxy, maturity)
    late = _real_bond_monthly(rr, maturity)
    real = pd.concat([early[early.index <= late.index[0] - 1], late]) if len(late) else early
    real = real[~real.index.duplicated(keep="last")]
    cm = c.reindex(pd.period_range(c.index[0], c.index[-1] + 3, freq="M"))
    accrual = (cm.shift(2) / cm.shift(3) - 1).reindex(real.index)
    out = ((1 + real) * (1 + accrual) - 1).dropna()
    out = out[out.index >= pd.Period(start, "M")]
    info = {"proxy_window": k, "change_rmse": scores, "first_real_month": str(rr.index[0]), "maturity": maturity}
    return out, info


# ---- unhedged international government bonds (BWXSIM)
#
# The same par-bond model as BNDXSIM (OECD 10-year yields, 9-year par bonds, BNDX-like country weights), but in
# USD without a currency hedge: each country's local bond return is converted at the change in its exchange
# rate. BWX (2007) and IGOV (2009) hold developed ex-US government bonds unhedged.

def fx_usd_per_unit(daily: pd.Series | None, monthly: pd.Series | None, usd_per_unit: bool) -> pd.Series:
    """Month-end USD value of one unit of a currency (PeriodIndex 'M'): the last daily quote of each month
    where a daily series exists, else the monthly average."""
    parts = []
    if daily is not None and len(daily):
        parts.append(_monthly(daily))
    if monthly is not None and len(monthly):
        parts.append(_monthly(monthly))
    if not parts:
        return pd.Series(dtype=float)
    s = parts[0]
    for p in parts[1:]:
        s = s.combine_first(p)
    s = s[s > 0]
    return s if usd_per_unit else 1.0 / s


def unhedged_bond_model(long: dict, fx: dict, weights: dict | None = None, maturity: float = 9) -> pd.Series:
    """Monthly USD return of an unhedged developed-market government bond index (PeriodIndex): per country the
    local par-bond return times the change in the currency's USD value; weights renormalised each month over
    the countries with data; at least three countries (a third of the weight) per month."""
    weights = weights or dict(INTL_BONDS)
    rets = {}
    for cty, y in long.items():
        f = fx.get(cty)
        y = _monthly(y)
        if f is None or len(f) < 13 or len(y) < 13:
            continue
        local = _real_bond_monthly(y, maturity)      # a nominal par bond: the same pricing
        fxr = f.sort_index().pct_change()
        rets[cty] = ((1 + local) * (1 + fxr.reindex(local.index)) - 1).dropna()
    if not rets:
        raise RuntimeError("no country has both a yield and an exchange rate")
    df = pd.DataFrame(rets).sort_index()
    w = pd.DataFrame({c: np.where(df[c].notna(), weights.get(c, 0.0), 0.0) for c in df.columns}, index=df.index)
    tot = w.sum(axis=1)
    out = (df.fillna(0.0) * w).sum(axis=1) / tot.where(tot > 0)
    ok = (df.notna().sum(axis=1) >= 3) & (tot >= 0.3)
    out = out[ok].dropna()
    if out.empty:
        raise RuntimeError("no month with enough countries")
    return out


def fx_monthly_all() -> dict:
    """{country: month-end USD per unit} from data/macro; the euro-area members follow their legacy currency
    until 1998 and the euro (at the fixed conversion rate) from 1999."""
    def opt(sid):
        try:
            return _fred(sid) if sid else None
        except Exception:  # noqa: BLE001 - a missing file: the other source, or no country
            return None
    eur = None
    d, m, usd = FX_SERIES["EU"]
    try:
        eur = fx_usd_per_unit(opt(d), opt(m), usd)
    except Exception:  # noqa: BLE001
        eur = None
    out = {}
    for cty, (d, m, usd) in FX_SERIES.items():
        if cty == "EU":
            continue
        s = fx_usd_per_unit(opt(d), opt(m), usd)
        if cty in EURO_RATES:
            s = s[s.index < pd.Period("1999-01", "M")]
            if eur is not None and len(eur):
                s = pd.concat([s, eur / EURO_RATES[cty]])
                s = s[~s.index.duplicated(keep="last")]
        if len(s):
            out[cty] = s
    return out


def unhedged_bond_monthly() -> pd.Series:
    long = {}
    for c, _ in INTL_BONDS:
        try:
            long[c] = _fred(f"IRLTLT01{c}M156N")
        except Exception as e:  # noqa: BLE001
            print(f"unhedged bonds: no long yield for {c}: {e}")
    fx = fx_monthly_all()
    m = unhedged_bond_model(long, fx)
    used = sorted(set(long) & set(fx))
    _simnote(f"BWXSIM model: {len(m)} months {m.index[0]}..{m.index[-1]}, countries {used} (unhedged, month-end "
             "exchange rates; monthly averages where FRED has no daily series)")
    return pd.Series(m.to_numpy(), index=m.index.to_timestamp(how="end").normalize())


# ---- missed mutual-fund distributions (VTSMX)
#
# Yahoo's mutual-fund histories understate some early distributions: on the ex-date the NAV drops by the full
# payout, but the dividend column (and Yahoo's adj_close, which is derived from it - the two agree to within
# 0.03%/yr for VTSMX) records less, so the day shows a loss the market never had. VTSMX 1993-1996 (the
# Vanguard-published calendar-year returns are 10.62%, -0.17%, 35.79%, 20.96%): from its price file 10.34%,
# -0.43%, 34.97%, 20.74%; 1995-12-22, for example, records a $0.10 dividend while the NAV fell 1.0% on a day the
# market rose 0.34%. The repair: on an ex-dividend day (dividend > 0) where the fund trails a daily reference of
# the same market (the Fama-French US market for VTSMX, tracking error 0.6%/yr) by more than max(3 robust daily
# tracking deviations, 0.10%), the day's return is set to the reference's. With it VTSMX's 1993-1996 read 10.55%,
# -0.21%, 35.88%, 21.04%. Used only where the fund is a SIM twin (VTISIM); VTSMX's own file is not changed.

def repair_missed_distributions(fund_ret: pd.Series, ref_ret: pd.Series, dividends: pd.Series,
                                k: float = 3.0, floor: float = 0.001, window: int = 250,
                                until=None) -> tuple[pd.Series, list]:
    """(repaired daily returns, [(date, fund return, reference return)]) - see the note above. The threshold
    uses the median absolute deviation of fund - reference over the previous `window` days (x 1.4826). Only
    days before `until` (the date the fund stops being used) are repaired."""
    both = pd.concat({"f": fund_ret, "r": ref_ret}, axis=1, sort=True).dropna()
    res = both["f"] - both["r"]
    mad = res.abs().rolling(window, min_periods=60).median().shift(1) * 1.4826
    div = pd.to_numeric(dividends, errors="coerce").reindex(both.index).fillna(0.0)
    hit = (div > 0) & (res < -np.maximum(k * mad, floor))
    if until is not None:
        hit &= both.index < pd.Timestamp(until)
    out = fund_ret.copy()
    fixed = []
    for d in both.index[hit.fillna(False).to_numpy()]:
        fixed.append((d, float(both.at[d, "f"]), float(both.at[d, "r"])))
        out.loc[d] = both.at[d, "r"]
    return out, fixed


REAL_OVERRIDES: dict[str, pd.Series] = {}   # fund -> repaired daily returns, used by _real_returns in the SIMs


# ------------------------------------------------------------------ fund research metadata

FUNDS_META_FILE = ROOT / "data" / "funds_meta.json"
FUNDS_META_PER_RUN = 300      # symbols looked up per run (two Yahoo requests each)
FUNDS_META_MAX_AGE = 30       # days before an entry is refreshed


def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def extract_fund_meta(info: dict | None, funds=None) -> dict:
    """Fund metadata from yfinance: `info` (Ticker.info, Yahoo's quoteSummary: longName, category, fundFamily,
    legalType, quoteType, fundInceptionDate (epoch seconds), totalAssets, yield (a fraction), netExpenseRatio (in
    PERCENT) / annualReportExpenseRatio (a fraction)) and `funds` (Ticker.funds_data, yfinance >= 0.2.40:
    fund_overview {categoryName, family, legalType}, fund_operations (rows 'Annual Report Expense Ratio' (a
    fraction), 'Annual Holdings Turnover', 'Total Net Assets'), top_holdings (Symbol index, Name, Holding
    Percent), asset_classes {stockPosition, bondPosition, cashPosition, ...}, sector_weightings, description).
    Fields that are missing stay out; every value is a plain JSON type. Checked against yfinance 1.7.0's
    scrapers/funds.py."""
    info = info or {}
    out: dict = {}

    def put(k, v):
        if v is not None and v != "" and not (isinstance(v, float) and not np.isfinite(v)):
            out[k] = v
    put("name", info.get("longName") or info.get("shortName"))
    put("quote_type", info.get("quoteType"))
    put("category", info.get("category"))
    put("family", info.get("fundFamily"))
    put("legal_type", info.get("legalType"))
    inc = _num(info.get("fundInceptionDate"))
    if inc:
        put("inception", str(pd.Timestamp(inc, unit="s").date()))
    put("net_assets", _num(info.get("totalAssets")) or _num(info.get("netAssets")))
    put("yield", _num(info.get("yield")))
    er = _num(info.get("netExpenseRatio"))
    if er is not None:
        put("expense_ratio", er / 100)                        # Yahoo gives this one in percent
    elif _num(info.get("annualReportExpenseRatio")) is not None:
        put("expense_ratio", _num(info.get("annualReportExpenseRatio")))
    if funds is not None:
        def safe(attr):
            try:
                return getattr(funds, attr)
            except Exception:  # noqa: BLE001 - yfinance raises on funds it has no profile for
                return None
        ov = safe("fund_overview") or {}
        if isinstance(ov, dict):
            for key, src in (("category", "categoryName"), ("family", "family"), ("legal_type", "legalType")):
                if ov.get(src) and key not in out:
                    out[key] = ov[src]
        ops = safe("fund_operations")
        if isinstance(ops, pd.DataFrame) and len(ops.columns):
            col = ops.columns[0]

            def row(name):
                return _num(ops[col].get(name)) if name in ops.index else None
            if row("Annual Report Expense Ratio") is not None:
                out["expense_ratio"] = row("Annual Report Expense Ratio")   # a fraction: preferred
            put("turnover", row("Annual Holdings Turnover"))
            if "net_assets" not in out:
                put("net_assets", row("Total Net Assets"))
        th = safe("top_holdings")
        if isinstance(th, pd.DataFrame) and len(th):
            hold = []
            for sym, r in th.head(10).iterrows():
                w = _num(r.get("Holding Percent"))
                hold.append({"symbol": str(sym), "name": str(r.get("Name") or ""), "weight": w})
            out["top_holdings"] = hold
        ac = safe("asset_classes")
        if isinstance(ac, dict):
            cls = {k.replace("Position", ""): _num(v) for k, v in ac.items() if _num(v) is not None}
            if cls:
                out["asset_classes"] = cls
        sw = safe("sector_weightings")
        if isinstance(sw, dict):
            sec = {k: _num(v) for k, v in sw.items() if _num(v)}
            if sec:
                out["sectors"] = sec
        desc = safe("description")
        if isinstance(desc, str) and desc.strip():
            out["description"] = desc.strip()[:600]
    return out


def funds_meta_batch(universe: list[str], entries: dict, today: str, budget: int = FUNDS_META_PER_RUN,
                     max_age: int = FUNDS_META_MAX_AGE, failed: dict | None = None) -> list[str]:
    """The symbols to look up this run: those with no entry first, then the stalest entries older than
    `max_age` days; a symbol that failed is retried after `max_age` days."""
    failed = failed or {}
    t0 = pd.Timestamp(today)

    def age(d):
        try:
            return (t0 - pd.Timestamp(d)).days
        except Exception:  # noqa: BLE001
            return 10 ** 6
    missing = [t for t in universe if t not in entries and age(failed.get(t, "1900-01-01")) >= max_age]
    stale = sorted((t for t in universe if t in entries and age(entries[t].get("fetched")) >= max_age),
                   key=lambda t: entries[t].get("fetched") or "")
    return (missing + stale)[:budget]


def fetch_funds_meta(universe: list[str], path: Path | None = None, today: str | None = None) -> dict:
    """data/funds_meta.json: fund metadata (extract_fund_meta) for the ETFs and mutual funds of the universe, a
    rotating batch per run; an entry that fails to refresh is kept."""
    path = path or FUNDS_META_FILE
    today = today or str(pd.Timestamp.today().date())
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError):
        doc = {}
    entries, failed = dict(doc.get("funds") or {}), dict(doc.get("failed") or {})
    batch = funds_meta_batch(universe, entries, today, failed=failed)

    def one(t):
        try:
            tk = yf.Ticker(t)
            try:
                info = tk.info or {}
            except Exception:  # noqa: BLE001
                info = {}
            try:
                funds = tk.funds_data
            except Exception:  # noqa: BLE001
                funds = None
            return t, extract_fund_meta(info, funds)
        except Exception as e:  # noqa: BLE001
            print(f"fund meta {t} failed: {e}", file=sys.stderr)
            return t, None
    from concurrent.futures import ThreadPoolExecutor
    got = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        for t, m in pool.map(one, batch):
            if m and len(m) >= 2:
                m["fetched"] = today
                entries[t] = m
                failed.pop(t, None)
                got += 1
            else:
                failed[t] = today
    doc = {"updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "source": "Yahoo Finance via yfinance (Ticker.info and Ticker.funds_data); expense_ratio and yield are "
                     "fractions, net_assets in USD, top_holdings weights are fractions",
           "funds": dict(sorted(entries.items())), "failed": dict(sorted(failed.items()))}
    path.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(f"fund metadata: {got} of {len(batch)} looked up; {len(entries)} funds on file")
    return doc


def main() -> None:
    PRICES.mkdir(parents=True, exist_ok=True)
    ndx, source = constituents()
    try:
        membership = update_membership()
        try:
            update_changes()
        except Exception as e:  # noqa: BLE001
            print(f"component changes failed: {e}", file=sys.stderr)
        # the live constituent list is this month's snapshot (Wikipedia's format may not parse)
        cur_month = str(pd.Timestamp.today().to_period("M"))
        if source != "fallback" and cur_month not in set(membership["month"]):
            row = pd.DataFrame([{"month": cur_month, "revid": "live:" + source, "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                 "count": str(len(ndx)), "tickers": " ".join(ndx)}])
            membership = pd.concat([membership, row]).sort_values("month")
            membership.to_csv(MEMBERSHIP, index=False)
        hist_members = sorted({t for row in membership["tickers"] for t in row.split()})
    except Exception as e:  # noqa: BLE001
        print(f"membership history failed: {e}", file=sys.stderr)
        membership, hist_members = None, []
    former = [t for t in hist_members if t not in ndx]
    tickers = ndx + [t for t in EXTRA if t not in ndx]
    print(f"{len(ndx)} current constituents from {source}; {len(former)} former members; {len(tickers)} other tickers")

    ok, failed = {}, []
    delisted = load_delisted()
    merge_log: list[str] = []

    def save_merged(t: str, df: pd.DataFrame) -> dict:
        """Save a download without ever losing the history already on disk (see merge_history)."""
        out, how = merge_history(t, df, read_prices(t))
        if how != "new":
            merge_log.append(f"{t}: {how} (download {df.index[0].date()}..{df.index[-1].date()}, {len(df)} rows)")
            print(f"{t}: {how}", flush=True)
        return save_prices(t, out)

    t0 = time.time()
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=6) as pool:
        for t, df in zip(tickers, pool.map(fetch_with_retry, tickers)):
            if df is None:
                failed.append(t)
                continue
            ok[t] = save_merged(t, df)
            print(f"{t:6s} {ok[t]}", flush=True)
    print(f"prices: {len(ok)} ok, {len(failed)} failed in {time.time() - t0:.0f}s", flush=True)

    # the broad ETF / mutual fund universe, a rotating batch per run (see broad_batch)
    try:
        prev = json.loads((ROOT / "data" / "universe.json").read_text())
    except Exception:  # noqa: BLE001
        prev = {}
    broad_failed = {t: d for t, d in (prev.get("broad_failed") or {}).items() if t in set(BROAD)}
    today = str(pd.Timestamp.today().date())
    batch = broad_batch([t for t in BROAD if t not in ok], prev.get("tickers") or {}, broad_failed, today)
    broad_ok = {}
    t1 = time.time()
    with ThreadPoolExecutor(max_workers=6) as pool:
        for t, df in zip(batch, pool.map(lambda x: fetch_with_retry(x, tries=2), batch)):
            if df is None:
                broad_failed.setdefault(t, today)
                continue
            broad_failed.pop(t, None)
            broad_ok[t] = save_merged(t, df)
    print(f"broad universe: {len(broad_ok)} of {len(batch)} downloaded in {time.time() - t1:.0f}s "
          f"({len(BROAD)} symbols in all; {len(broad_failed)} waiting to be retried)", flush=True)

    def file_info(t: str) -> dict | None:
        df = read_prices(t)
        if df is None or not len(df):
            return None
        return {"first": str(df.index[0].date()), "last": str(df.index[-1].date()), "rows": len(df)}

    # a failed refresh keeps the last good copy (the file is never deleted)
    kept = {}
    for t in failed:
        info = file_info(t)
        if info:
            kept[t] = info
            print(f"{t}: refresh failed, keeping the saved history {info}")

    # months each symbol was an index member (for the identity check of fallback sources)
    member_months: dict[str, list[str]] = {}
    if membership is not None:
        for m, row in zip(membership["month"], membership["tickers"]):
            for sym in str(row).split():
                member_months.setdefault(sym, []).append(m)

    # former members: fetch under the (possibly renamed) current symbol, store under the old symbol
    former_ok, former_missing = {}, []
    for t in former:
        src = RENAMES.get(t, t)
        if src in ok and src != t:
            df = pd.read_csv(PRICES / f"{src}.csv", parse_dates=["date"], index_col="date")
        else:
            df = fetch_with_retry(src, tries=2)
        saved = read_prices(t)
        if t in KEYED_OK and saved is not None and len(saved) >= 20:
            # a delisted history from a keyed source: Yahoo's symbol may now be another company
            former_ok[t] = file_info(t)
            continue
        if df is not None and len(df) >= 5:
            former_ok[t] = save_merged(t, df)
            print(f"former {t:6s} {former_ok[t]}{' (from ' + src + ')' if src != t else ''}")
            continue
        # Yahoo has nothing (usual for acquired / bankrupt companies). A saved history of reasonable length
        # is kept as it is (delisted histories don't change); a short or missing one is looked up in the
        # fallback sources, and a result is only used if it passes the identity check.
        if saved is not None and len(saved) >= 20:
            former_ok[t] = file_info(t)
            continue
        got, why_not = None, []
        for name, fn in (("keyed", fetch_delisted_keyed), ("stooq", fetch_stooq)):
            cand = fn(t)
            if cand is None:
                continue
            problem = plausible_member_series(t, cand, member_months.get(t, []))
            if problem:
                why_not.append(f"{name}: {problem}")
                print(f"former {t}: {name} history rejected ({problem})")
                continue
            got = cand
            if name == "keyed":
                KEYED_OK.add(t)
            print(f"former {t}: {len(cand)} rows from {name} {cand.attrs.get('name', '')}")
            break
        if got is not None:
            former_ok[t] = save_merged(t, got)
            delisted.setdefault(t, {}).update({k: v for k, v in (("source", got.attrs.get("source")),
                                                                  ("source_name", got.attrs.get("name"))) if v})
            continue
        if saved is not None and len(saved):
            former_ok[t] = file_info(t) if len(saved) >= 20 else None
            if former_ok[t] is None:
                former_ok.pop(t)
                former_missing.append(t)       # kept on disk, but too short to be useful
        else:
            former_missing.append(t)
        if why_not:
            merge_log.append(f"{t}: fallback histories rejected: {'; '.join(why_not)}")

    # current constituents whose history stopped are not really current (their files are kept)
    last_bench = max(pd.Timestamp((ok.get(b) or kept.get(b) or {"last": "1900-01-01"})["last"]) for b in ("SPY", "QQQ"))
    cur = {t: ok.get(t) or kept.get(t) for t in ndx if ok.get(t) or kept.get(t)}
    stale = [t for t, info in cur.items() if (last_bench - pd.Timestamp(info["last"])).days > 10 or info["rows"] < 5]
    for t in stale:
        print(f"{t}: stale or too little data {cur[t]}; kept on disk, not listed as a current member")
    ndx = [t for t in ndx if t in cur and t not in stale]

    # every symbol whose history has ended: its last date (and the reason, when known)
    for t in sorted(set(former) | set(stale) | set(kept)):
        info = former_ok.get(t) or kept.get(t) or file_info(t)
        if not info:
            continue
        if (last_bench - pd.Timestamp(info["last"])).days <= 10:
            delisted.pop(t, None)          # trading (again): not delisted
            continue
        entry = delisted.get(t, {})
        entry.update({"last_date": info["last"], "first_date": info["first"], "rows": info["rows"]})
        if info["rows"] < HISTORY_MIN_ROWS:
            # Yahoo drops a delisted symbol's history; only what the repository saved survives
            entry["history"] = HISTORY_UNAVAILABLE
        else:
            entry.pop("history", None)
        if t in DELISTED_REASONS:
            entry["reason"] = DELISTED_REASONS[t]
        elif t in RENAMES:
            entry.setdefault("reason", f"symbol changed to {RENAMES[t]}")
        if t in member_months:
            entry["member_months"] = f"{member_months[t][0]}..{member_months[t][-1]}"
        delisted[t] = entry
    DELISTED_FILE.write_text(json.dumps(dict(sorted(delisted.items())), indent=1) + "\n")
    (ROOT / "data" / "merge_log.txt").write_text("\n".join(merge_log) + "\n")

    fetch_macro()
    fetch_shiller()
    fetch_factors()
    fetch_shares(sorted(set(ndx) | set(former_ok)))
    try:
        fetch_sec_shares(sorted(set(ndx) | set(former_ok) | set(kept) | set(hist_members) & set(p.stem for p in PRICES.glob("*.csv"))))
    except Exception as e:  # noqa: BLE001 - share counts are optional
        print(f"sec shares failed: {e}", file=sys.stderr)
    sims = build_sims()

    # every broad symbol with a file: this run's download, else what the last run recorded, else the file
    broad_info = {}
    for t in BROAD:
        if t in broad_ok:
            broad_info[t] = broad_ok[t]
        elif (PRICES / f"{t}.csv").exists():
            info = (prev.get("tickers") or {}).get(t) or file_info(t)
            if info:
                broad_info[t] = info
    meta = {
        "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "constituent_source": source,
        "nasdaq100": ndx,
        "etfs": [t for t in ETFS if t in ok or t in kept] + [t for t in BROAD_ETFS if t in broad_info],
        "funds": [t for t in BROAD_FUNDS if t in broad_info],               # mutual funds (backtester/fund_lists.py)
        "broad_failed": dict(sorted(broad_failed.items())),
        "stocks": [t for t in STOCKS if t in ok or t in kept],
        "indexes": [t for t in INDEXES if t in ok or t in kept],
        "requested": [t for t in REQUESTED if t in ok or t in kept],            # from data/extra_tickers.txt
        "requested_failed": [t for t in REQUESTED if t not in ok and t not in kept],
        "benchmarks": ["SPY", "QQQ"],
        "sims": sims,
        "former_members": sorted(former_ok),
        "former_members_missing_data": sorted(former_missing),
        "tickers": {**broad_info, **kept, **ok, **former_ok},
        "failed": failed,
        "kept_after_failed_refresh": sorted(kept),
        "dropped_stale": stale,
    }
    (ROOT / "data" / "universe.json").write_text(json.dumps(meta, indent=1))
    KEYED_FILE.write_text(json.dumps(sorted(KEYED_OK), indent=1))
    try:
        # fund research: metadata for every ETF and mutual fund (a rotating batch per run)
        fund_syms = [t for t in dict.fromkeys(meta["etfs"] + meta["funds"]) if "-" not in t and not t.startswith("^")]
        fetch_funds_meta(fund_syms)
    except Exception as e:  # noqa: BLE001 - metadata is optional
        print(f"fund metadata failed: {e}", file=sys.stderr)
    write_sim_drag()
    try:
        write_integrity_log()
    except Exception as e:  # noqa: BLE001 - the log is informational
        print(f"price integrity log failed: {e}", file=sys.stderr)
    (ROOT / "data" / "sims_log.txt").write_text(
        ("\n".join(SIM_LOG) if SIM_LOG else "all simulated series built") + "\n\n"
        + "Model vs fund on their overlap:\n" + "\n".join(SIM_NOTES) + "\n")
    print(f"done: {len(ok)} current/ETF ok, {len(former_ok)} former members ok, "
          f"{len(former_missing)} former members without data, {len(failed)} failed {failed}")
    if len(ok) < len(tickers) * 0.85:
        sys.exit(1)


INTEGRITY_FILE = ROOT / "data" / "inferred_splits.json"


def write_integrity_log(tickers: list[str] | None = None, path: Path | None = None) -> dict:
    """Run the price-integrity gate (backtester/integrity.py: splits the source data missed or booked wrongly,
    isolated bad ticks) over every price file and record what it repaired, with the evidence, under "inferred" in
    data/inferred_splits.json (its "overrides" - manual decisions - are kept). The repair itself happens when the
    backtester loads a file, so a new download that still carries the error is repaired the same way; this log is
    what a reviewer checks (and turns into an override when the gate got one wrong)."""
    from backtester import data as bt_data
    # next to the price folder in use (a test pointing PRICES at a temporary folder writes there, not into data/)
    path = path or (PRICES.parent / INTEGRITY_FILE.name)
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError):
        doc = {}
    doc.setdefault("overrides", {})
    names = tickers if tickers is not None else sorted(p.stem for p in PRICES.glob("*.csv"))
    bt_data.load.cache_clear()
    bt_data.PRICE_REPAIRS.clear()
    found: dict = {}
    for t in names:
        try:
            ev = bt_data.price_repairs(t)
        except Exception as e:  # noqa: BLE001
            print(f"{t}: integrity check failed: {e}", file=sys.stderr)
            continue
        for _, r in ev.iterrows():
            found.setdefault(t, {})[str(pd.Timestamp(r["date"]).date())] = {
                "kind": r["kind"], "ratio": None if pd.isna(r["ratio"]) else round(float(r["ratio"]), 6),
                "applied": r["applied"], "day_return_in_file": round(float(r["day_return_raw"]), 6),
                "day_return_repaired": round(float(r["day_return_now"]), 6),
                "expected_from_reference": round(float(r["expected"]), 6), "residual_sd": round(float(r["sigma"]), 6),
                "reference": r["reference"] if isinstance(r["reference"], str) else None,
                "volume_ratio_after": None if r["volume_ratio"] is None or pd.isna(r["volume_ratio"]) else round(float(r["volume_ratio"]), 4),
                "why": r["why"]}
    doc["inferred"] = found
    path.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    n = sum(len(v) for v in found.values())
    print(f"price integrity: {n} repairs in {len(found)} files (data/inferred_splits.json)")
    return doc


if __name__ == "__main__":
    main()
