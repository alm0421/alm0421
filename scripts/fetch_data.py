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
""".split()
ETFS = list(dict.fromkeys(ETFS))
# large US stocks outside the Nasdaq-100 (stocks, not ETFs: kept separate so they are never mistaken
# for funds, e.g. when ETFs are stripped from index membership)
STOCKS = """
JPM XOM BRK-B JNJ UNH V MA HD PG CVX LLY ABBV MRK KO BAC WFC DIS MCD NKE ORCL CRM IBM GE CAT BA GS MS C T VZ
PFE TMO DHR ABT NEE DUK SO LMT RTX UPS UNP MMM
""".split()
INDEXES = ["^NDX", "^GSPC", "^VIX", "^IRX", "^TNX", "^DJI", "^RUT", "^SP500TR", "^VIX3M", "^TYX", "^FVX"]
EXTRA = ETFS + STOCKS + INDEXES

# Symbol changes: membership lists use the old symbol, Yahoo keeps history under the new one.
# Only renames where Yahoo's history for the new symbol genuinely continues the same company.
RENAMES = {"FB": "META", "PCLN": "BKNG", "DISCA": "WBD", "RIMM": "BB", "MYL": "VTRS", "NLOK": "GEN",
           "SYMC": "GEN", "JDSU": "VIAV", "JDSUD": "VIAV", "HANS": "MNST", "CTRP": "TCOM", "WLTW": "WTW",
           "UAUA": "UAL", "KFT": "MDLZ", "KLA": "KLAC", "ERICY": "ERIC", "WFMI": "WFM", "LINTA": "QRTEA"}


STOOQ_FAILS = [0]
KEYED_BUDGET = {"alphavantage": 20, "tiingo": 400}   # per run; free tiers allow 25/day and 1,000/day


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
                    return out
        except Exception as e:  # noqa: BLE001
            print(f"alphavantage {t}: {e}", file=sys.stderr)
    return None


def fetch_stooq(t: str) -> pd.DataFrame | None:
    """Secondary source for delisted US stocks (no dividends; split-adjusted closes used as adj_close)."""
    if STOOQ_FAILS[0] >= 5:  # the source is unreachable or has nothing: stop paying for timeouts
        return None
    try:
        r = requests.get(f"https://stooq.com/q/d/l/?s={t.lower().replace('-', '.')}.us&i=d", headers=UA, timeout=10)
        if r.status_code != 200 or not r.text.lower().startswith("date"):
            STOOQ_FAILS[0] += 1
            return None
        df = pd.read_csv(io.StringIO(r.text), parse_dates=["Date"], index_col="Date")
        if len(df) < 20:
            return None
        df = df.rename(columns=str.lower)
        df["adj_close"] = df["close"]
        df["dividend"], df["split"] = 0.0, 0.0
        df.index.name = "date"
        return df[["open", "high", "low", "close", "adj_close", "volume", "dividend", "split"]]
    except Exception as e:  # noqa: BLE001
        print(f"stooq {t}: {e}", file=sys.stderr)
        STOOQ_FAILS[0] += 1
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

def fetch_macro() -> None:
    MACRO.mkdir(parents=True, exist_ok=True)
    for sid in ("CPIAUCSL", "DTB3", "DGS10", "DGS20", "DGS30", "DGS5", "DGS2", "DGS1", "GS10", "TB3MS",
                "DAAA", "DBAA", "AAA", "BAA"):
        try:
            txt = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}", headers=UA, timeout=60).text
            df = pd.read_csv(io.StringIO(txt))
            df.columns = ["date", "value"]
            df = df[pd.to_numeric(df["value"], errors="coerce").notna()]
            df.to_csv(MACRO / f"{sid}.csv", index=False)
            print(f"macro {sid}: {len(df)} rows {df['date'].iloc[0]} .. {df['date'].iloc[-1]}")
        except Exception as e:  # noqa: BLE001
            print(f"macro {sid} failed: {e}", file=sys.stderr)


def fetch_factors() -> None:
    import zipfile
    FACTORS.mkdir(parents=True, exist_ok=True)
    base = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    for name, fn in (("ff3_daily", "F-F_Research_Data_Factors_daily_CSV.zip"),
                     ("ff5_daily", "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"),
                     ("mom_daily", "F-F_Momentum_Factor_daily_CSV.zip"),
                     ("port6_daily", "6_Portfolios_2x3_daily_CSV.zip"),
                     ("dev_ff3_daily", "Developed_ex_US_3_Factors_Daily_CSV.zip"),
                     ("em_ff5_monthly", "Emerging_5_Factors_CSV.zip")):
        try:
            z = zipfile.ZipFile(io.BytesIO(requests.get(base + fn, headers=UA, timeout=120).content))
            raw = z.read(z.namelist()[0]).decode("latin-1").splitlines()
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
            df.to_csv(FACTORS / f"{name}.csv", index=False)
            print(f"factors {name}: {len(df)} rows, columns {header[1:]}")
        except Exception as e:  # noqa: BLE001
            print(f"factors {name} failed: {e}", file=sys.stderr)


def fetch_shares(tickers: list[str]) -> None:
    SHARES.mkdir(parents=True, exist_ok=True)
    for t in tickers:
        try:
            s = yf.Ticker(t).get_shares_full(start="2000-01-01")
            if s is None or len(s) == 0:
                continue
            s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
            s = s[~s.index.duplicated(keep="last")].sort_index()
            s.rename("shares").to_csv(SHARES / f"{t}.csv", index_label="date")
        except Exception as e:  # noqa: BLE001
            print(f"shares {t} failed: {e}", file=sys.stderr)


# ------------------------------------------------------------------ main

# ------------------------------------------------------------------ simulated long histories

def _bond_returns(y: pd.Series, maturity: float) -> pd.Series:
    """Daily total return of a constant-maturity Treasury fund priced off a yield series (in %).

    Each day: yesterday's par bond earns its coupon for the days held and is re-priced at today's
    yield (semi-annual coupons). The standard way to extend bond funds back before they existed."""
    y = (y / 100.0).dropna()
    c = y.shift(1).to_numpy()
    r = y.to_numpy()
    d = pd.Series(y.index, index=y.index).diff().dt.days.to_numpy() / 365.25
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


def _splice(sim_ret: pd.Series, real: str) -> pd.Series:
    """Simulated returns before `real` existed, then the real fund's total return (adj_close)."""
    p = PRICES / f"{real}.csv"
    r = pd.Series(dtype=float)
    if p.exists():
        df = pd.read_csv(p, parse_dates=["date"], index_col="date")
        r = df["adj_close"].pct_change().dropna()
    first = r.index[0] if len(r) else sim_ret.index[-1] + pd.Timedelta(days=1)
    ret = pd.concat([sim_ret[sim_ret.index < first], r]).fillna(0.0)
    return 100 * (1 + ret).cumprod()


def build_sims() -> list[str]:
    """SPYSIM / TLTSIM / IEFSIM / SHYSIM / BILSIM: long total-return histories for portfolio research."""
    made = []
    try:
        ff = pd.read_csv(FACTORS / "ff3_daily.csv", parse_dates=["date"], index_col="date")
        mkt = ff["Mkt-RF"] + ff["RF"]
        _series_file("SPYSIM", _splice(mkt, "SPY"), "US market total return from Fama-French before SPY, then SPY")
        _series_file("BILSIM", _splice(ff["RF"], "BIL"), "1-month T-bill (Fama-French RF) before BIL, then BIL")
        made += ["SPYSIM", "BILSIM"]
    except Exception as e:  # noqa: BLE001
        print(f"sim SPYSIM failed: {e}", file=sys.stderr)
    try:
        y = {sid: pd.read_csv(MACRO / f"{sid}.csv", parse_dates=["date"], index_col="date")["value"].astype(float)
             for sid in ("DGS10", "DGS20", "DGS30", "DGS2")}
        long = y["DGS20"].combine_first((y["DGS10"] + y["DGS30"]) / 2).combine_first(y["DGS10"])
        _series_file("TLTSIM", _splice(_bond_returns(long, 20), "TLT"), "20-year Treasury priced off FRED yields, then TLT")
        _series_file("IEFSIM", _splice(_bond_returns(y["DGS10"], 9), "IEF"), "9-year Treasury off the 10-year yield, then IEF")
        y1 = pd.read_csv(MACRO / "DGS1.csv", parse_dates=["date"], index_col="date")["value"].astype(float)
        short = y["DGS2"].combine_first(y1)  # the 1-year yield stands in before the 2-year series (1976)
        _series_file("SHYSIM", _splice(_bond_returns(short, 2), "SHY"), "2-year Treasury off the 2-year yield (1-year before 1976), then SHY")
        made += ["TLTSIM", "IEFSIM", "SHYSIM"]
    except Exception as e:  # noqa: BLE001
        print(f"sim bonds failed: {e}", file=sys.stderr)
    # more asset classes from Ken French's data library (value-weighted portfolios, daily)
    try:
        p6 = pd.read_csv(FACTORS / "port6_daily.csv", parse_dates=["date"], index_col="date")
        col = {c.upper().replace(" ", ""): c for c in p6.columns}
        for t, key, real, note in (("VBRSIM", "SMALLHIBM", "VBR", "US small-cap value (Fama-French small/high B/M)"),
                                   ("VTVSIM", "BIGHIBM", "VTV", "US large-cap value (Fama-French big/high B/M)"),
                                   ("VUGSIM", "BIGLOBM", "VUG", "US large-cap growth (Fama-French big/low B/M)")):
            if key in col:
                _series_file(t, _splice(p6[col[key]].dropna(), real), note + ", then " + real)
                made.append(t)
        small = [c for k, c in col.items() if k.startswith("SMALL") or k.startswith("ME1")]
        if small:
            _series_file("VBSIM", _splice(p6[small].mean(axis=1).dropna(), "VB"), "US small-cap (Fama-French small portfolios), then VB")
            made.append("VBSIM")
    except Exception as e:  # noqa: BLE001
        print(f"sim size/value failed: {e}", file=sys.stderr)
    try:
        dev = pd.read_csv(FACTORS / "dev_ff3_daily.csv", parse_dates=["date"], index_col="date")
        _series_file("EFASIM", _splice((dev["Mkt-RF"] + dev["RF"]).dropna(), "EFA"),
                     "developed ex-US market (Fama-French, from 1990), then EFA")
        made.append("EFASIM")
    except Exception as e:  # noqa: BLE001
        print(f"sim EFASIM failed: {e}", file=sys.stderr)
    # (the Fama-French real-estate industry is operating companies, not REITs: a poor VNQ proxy, so no VNQSIM)
    try:
        y5 = pd.read_csv(MACRO / "DGS5.csv", parse_dates=["date"], index_col="date")["value"].astype(float)
        _series_file("IEISIM", _splice(_bond_returns(y5, 5), "IEI"), "5-year Treasury off the 5-year yield, then IEI")
        made.append("IEISIM")
    except Exception as e:  # noqa: BLE001
        print(f"sim IEISIM failed: {e}", file=sys.stderr)
    try:
        # investment-grade corporates: a 10-year par bond at the average of Moody's Aaa and Baa yields
        # (daily from 1986, monthly before), then LQD
        def fred(sid):
            return pd.read_csv(MACRO / f"{sid}.csv", parse_dates=["date"], index_col="date")["value"].astype(float)
        daily = (fred("DAAA") + fred("DBAA")) / 2
        monthly = (fred("AAA") + fred("BAA")) / 2
        m = monthly[monthly.index < daily.index[0]]
        m.index = m.index + pd.offsets.MonthEnd(0)
        corp = pd.concat([m.resample("B").ffill(), daily]).sort_index()
        corp = corp[~corp.index.duplicated(keep="last")]
        _series_file("LQDSIM", _splice(_bond_returns(corp[corp.index >= "1953-01-01"], 10), "LQD"),
                     "investment-grade corporates priced off Moody's Aaa/Baa yields, then LQD")
        made.append("LQDSIM")
    except Exception as e:  # noqa: BLE001
        print(f"sim LQDSIM failed: {e}", file=sys.stderr)
    try:
        em = pd.read_csv(FACTORS / "em_ff5_monthly.csv", parse_dates=["date"], index_col="date")
        r = (em["Mkt-RF"] + em["RF"]).dropna()
        level = (1 + r).cumprod()
        daily = level.resample("B").ffill().pct_change().dropna()
        _series_file("EEMSIM", _splice(daily, "EEM"), "emerging markets (Fama-French, monthly, from 1989), then EEM")
        made.append("EEMSIM")
    except Exception as e:  # noqa: BLE001
        print(f"sim EEMSIM failed: {e}", file=sys.stderr)
    try:
        c = commodity_monthly()
        tb = pd.read_csv(MACRO / "TB3MS.csv", parse_dates=["date"], index_col="date")["value"].astype(float) / 100 / 12
        tb.index = tb.index + pd.offsets.MonthEnd(0)
        # spot price change only: futures indexes also earn T-bill collateral but lose the roll yield,
        # and over 1972-2025 those two roughly offset (the S&P GSCI total return is ~7%/yr)
        r = c.pct_change().dropna()
        level = (1 + r).cumprod()
        daily = level.resample("B").ffill().pct_change().dropna()
        _series_file("DBCSIM", _splice(daily, "DBC"),
                     "commodities: World Bank energy + non-energy spot price indexes (monthly steps), then DBC")
        made.append("DBCSIM")
    except Exception as e:  # noqa: BLE001
        print(f"sim DBCSIM failed: {e}", file=sys.stderr)
    try:
        g = gold_monthly()
        daily = g.resample("B").ffill()
        _series_file("GLDSIM", _splice(daily.pct_change().dropna(), "GLD"),
                     "gold (World Bank monthly average price, stepped daily) from 1960, then GLD")
        made.append("GLDSIM")
    except Exception as e:  # noqa: BLE001
        print(f"sim GLDSIM failed: {e}", file=sys.stderr)
    return made


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
    """Monthly gold price (USD/oz) from the World Bank 'Pink Sheet' historical data (1960 onward)."""
    page = requests.get("https://www.worldbank.org/en/research/commodity-markets", headers=UA, timeout=60).text
    m = re.search(r'https://thedocs\.worldbank\.org/[^"\']+CMO-Historical-Data-Monthly\.xlsx', page)
    if not m:
        raise RuntimeError("Pink Sheet link not found")
    raw = pd.read_excel(io.BytesIO(requests.get(m.group(0), headers=UA, timeout=120).content),
                        sheet_name="Monthly Prices", header=None)
    hdr = next(i for i in range(min(len(raw), 20)) if any(str(v).strip() == "Gold" for v in raw.iloc[i]))
    gcol = next(j for j, v in enumerate(raw.iloc[hdr]) if str(v).strip() == "Gold")
    rows = raw.iloc[hdr + 1:]
    rows = rows[rows[0].astype(str).str.fullmatch(r"\d{4}M\d{2}")]
    idx = pd.to_datetime(rows[0].str.replace("M", "-") + "-01") + pd.offsets.MonthEnd(0)
    return pd.Series(pd.to_numeric(rows[gcol], errors="coerce").to_numpy(), index=idx).dropna()


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
    t0 = time.time()
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=6) as pool:
        for t, df in zip(tickers, pool.map(fetch_with_retry, tickers)):
            if df is None:
                failed.append(t)
                continue
            ok[t] = save_prices(t, df)
            print(f"{t:6s} {ok[t]}", flush=True)
    print(f"prices: {len(ok)} ok, {len(failed)} failed in {time.time() - t0:.0f}s", flush=True)

    # former members: fetch under the (possibly renamed) current symbol, store under the old symbol
    former_ok, former_missing = {}, []
    for t in former:
        src = RENAMES.get(t, t)
        if src in ok and src != t:
            df = pd.read_csv(PRICES / f"{src}.csv", parse_dates=["date"], index_col="date")
        else:
            df = fetch_with_retry(src, tries=2)
        if (df is None or len(df) < 5) and (PRICES / f"{t}.csv").exists() and t in KEYED_OK:
            df = pd.read_csv(PRICES / f"{t}.csv", parse_dates=["date"], index_col="date")  # fetched on an earlier run
        if df is None or len(df) < 5:
            df = fetch_delisted_keyed(t)
            if df is not None:
                KEYED_OK.add(t)
                print(f"former {t}: {len(df)} rows from a keyed delisted-data source")
        if df is None or len(df) < 5:
            df = fetch_stooq(t)
            if df is not None:
                print(f"former {t}: {len(df)} rows from stooq")
        if df is None or len(df) < 5:
            former_missing.append(t)
            continue
        former_ok[t] = save_prices(t, df)
        print(f"former {t:6s} {former_ok[t]}{' (from ' + src + ')' if src != t else ''}")

    # current constituents whose history stopped are not really current
    last_bench = max(pd.Timestamp(ok[b]["last"]) for b in ("SPY", "QQQ") if b in ok)
    stale = [t for t in ndx if t in ok and ((last_bench - pd.Timestamp(ok[t]["last"])).days > 10 or ok[t]["rows"] < 5)]
    for t in stale:
        print(f"dropping {t}: stale or too little data {ok[t]}")
        (PRICES / f"{t}.csv").unlink(missing_ok=True)
        ok.pop(t)
    ndx = [t for t in ndx if t in ok]
    keep = set(ok) | set(former_ok)
    for f in PRICES.glob("*.csv"):
        if f.stem not in keep:
            f.unlink()

    fetch_macro()
    fetch_factors()
    fetch_shares(sorted(set(ndx) | set(former_ok)))
    sims = build_sims()

    meta = {
        "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "constituent_source": source,
        "nasdaq100": ndx,
        "etfs": [t for t in ETFS if t in ok],
        "stocks": [t for t in STOCKS if t in ok],
        "indexes": [t for t in INDEXES if t in ok],
        "benchmarks": ["SPY", "QQQ"],
        "sims": sims,
        "former_members": sorted(former_ok),
        "former_members_missing_data": sorted(former_missing),
        "tickers": {**ok, **former_ok},
        "failed": failed,
        "dropped_stale": stale,
    }
    (ROOT / "data" / "universe.json").write_text(json.dumps(meta, indent=1))
    KEYED_FILE.write_text(json.dumps(sorted(KEYED_OK), indent=1))
    print(f"done: {len(ok)} current/ETF ok, {len(former_ok)} former members ok, "
          f"{len(former_missing)} former members without data, {len(failed)} failed {failed}")
    if len(ok) < len(tickers) * 0.85:
        sys.exit(1)


if __name__ == "__main__":
    main()
