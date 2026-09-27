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
EWJ EWU EWG EWC EWA EWQ EWL EWH VCLT MUB EMB VOE VOT
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

MEMBERSHIP_PARSER = "3"
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
            for m in re.finditer(r"\(\s*(?:NASDAQ:\s*|Nasdaq:\s*)?\[?\[?([A-Z]{1,5}(?:\.[A-Z])?)\]?\]?\s*\)", s):
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


def fetch_macro() -> None:
    MACRO.mkdir(parents=True, exist_ok=True)
    for sid in ["CPIAUCSL", "DTB3", "DGS10", "DGS20", "DGS30", "DGS5", "DGS2", "DGS1", "GS10", "TB3MS",
                "DAAA", "DBAA", "AAA", "BAA", "CPIAUCNS"] + intl_bond_ids():
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
    """Daily total returns (from adj_close) of a downloaded fund, or an empty series."""
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
}
SIM_DRAG_CAP = 0.03          # a larger gap is model error, not costs: never haircut more than 3% a year
SIM_DRAG: dict[str, dict] = {}   # ticker -> the drag applied this run (written to data/sims_drag.json)
SIM_DRAG_FILE = ROOT / "data" / "sims_drag.json"


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


def write_sim_drag(path: Path = SIM_DRAG_FILE) -> None:
    """Merge this run's drags into data/sims_drag.json (a series that failed this run keeps its old entry, as
    its price file is kept too)."""
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


def _validate(name: str, sim_ret: pd.Series, real: str) -> None:
    """Log how the model (before splicing) compares with the real fund over their overlap (monthly)."""
    try:
        r = _real_returns(real)
        s = sim_ret.dropna()
        if r.empty or s.empty:
            _simnote(f"validate {name} vs {real}: no overlap (fund or model missing)")
            return
        both = pd.concat({"m": s, "e": r}, axis=1).loc[max(s.index[0], r.index[0]):min(s.index[-1], r.index[-1])].fillna(0.0)
        mo = (1 + both).groupby(both.index.to_period("M")).prod() - 1
        mo = mo.iloc[1:-1]   # partial first / last months
        if len(mo) < 12:
            _simnote(f"validate {name} vs {real}: only {len(mo)} overlapping months")
            return
        yrs = len(mo) / 12
        cm, ce = (1 + mo["m"]).prod() ** (1 / yrs) - 1, (1 + mo["e"]).prod() ** (1 / yrs) - 1
        _simnote(f"validate {name} vs {real} {mo.index[0]}..{mo.index[-1]}: monthly correlation "
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


def build_sims() -> list[str]:
    """Long total-return histories for portfolio research: a model (index data or yields) before each
    fund existed, then the fund's own total return. Each series is built independently: one failing
    source is logged in data/sims_log.txt and never stops the others."""
    made: list[str] = []

    def build(t: str, sim_ret: pd.Series, reals: tuple[str, ...], note: str, validate: str | None = None,
              model: bool = True) -> None:
        """`model=False`: sim_ret is itself a real fund's return (net of its costs), so no fee/cost drag."""
        sim_ret = sim_ret.dropna()
        if sim_ret.empty:
            raise RuntimeError("empty model series")
        if validate:
            _validate(t, sim_ret, validate)   # the gross model: the gap it shows calibrates the drag
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
        build("VTISIM", ff["mkt"], ("VTSMX", "VTI"),
              "US total market: Fama-French market return until April 1992, then the Vanguard Total Stock Market "
              "Index fund (VTSMX), then VTI from June 2001", "VTSMX")
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
                build(t, model, reals, note, funds[0])
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
        # TIPS were first issued in 1997: the Vanguard Inflation-Protected Securities fund (VIPSX, June
        # 2000) is the longest real history; no model before it
        r = _real_returns("VIPSX")
        build("TIPSIM", r, ("TIP",), "US TIPS: Vanguard Inflation-Protected Securities fund (VIPSX) from 2000, then TIP",
              model=False)
    attempt("TIPSIM", tips)

    def high_yield():
        # FRED's ICE BofA high-yield total-return index only covers the last three years; the Vanguard
        # High-Yield Corporate fund (VWEHX, Yahoo history from 1985) is the longest free record
        r = _real_returns("VWEHX")
        build("HYGSIM", r, ("HYG",), "US high-yield bonds: Vanguard High-Yield Corporate fund (VWEHX) from 1985, then HYG",
              model=False)
    attempt("HYGSIM", high_yield)

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


def main() -> None:
    PRICES.mkdir(parents=True, exist_ok=True)
    ndx, source = constituents()
    try:
        membership = update_membership()
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
    write_sim_drag()
    (ROOT / "data" / "sims_log.txt").write_text(
        ("\n".join(SIM_LOG) if SIM_LOG else "all simulated series built") + "\n\n"
        + "Model vs fund on their overlap:\n" + "\n".join(SIM_NOTES) + "\n")
    print(f"done: {len(ok)} current/ETF ok, {len(former_ok)} former members ok, "
          f"{len(former_missing)} former members without data, {len(failed)} failed {failed}")
    if len(ok) < len(tickers) * 0.85:
        sys.exit(1)


if __name__ == "__main__":
    main()
