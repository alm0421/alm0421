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
""".split()
INDEXES = ["^NDX", "^GSPC", "^VIX", "^IRX", "^TNX", "^DJI", "^RUT"]
EXTRA = ETFS + INDEXES

# Symbol changes: membership lists use the old symbol, Yahoo keeps history under the new one.
RENAMES = {"FB": "META", "PCLN": "BKNG", "DISCA": "WBD", "GOOG": "GOOG", "BRCM": "AVGO"}
# BRCM (Broadcom Corp) was acquired by Avago, which took the AVGO name; history differs, so only
# use a rename when Yahoo's history for the new symbol genuinely continues the old company.
RENAMES.pop("BRCM")

WIKI_API = "https://en.wikipedia.org/w/api.php"
# Wikimedia asks automated clients for a descriptive user agent with a contact URL (browser-like
# agents from cloud IPs get blocked).
WIKI_UA = {"User-Agent": "BacktesterDataBot/1.0 (https://github.com/alm0421/alm0421) python-requests",
           "Accept": "application/json"}
MEMBERSHIP = ROOT / "data" / "ndx_membership.csv"
MACRO = ROOT / "data" / "macro"
FACTORS = ROOT / "data" / "factors"
SHARES = ROOT / "data" / "shares"


def save_prices(t: str, df: pd.DataFrame) -> dict:
    df.round(6).to_csv(PRICES / f"{t}.csv", float_format="%.6g")
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

def wiki_tickers(wikitext: str) -> set[str]:
    """Extract ticker symbols from a revision of the Nasdaq-100 article (formats changed over the years)."""
    found: set[str] = set()
    for m in re.finditer(r"\{\{\s*(?:NASDAQ|Nasdaq|nasdaq|NasdaqSymbol|NASDAQ link)\s*\|\s*([A-Za-z.]{1,6})\s*[|}]", wikitext):
        found.add(m.group(1).upper())
    for line in wikitext.splitlines():
        s = line.strip()
        if s.startswith(("#", "*")):
            for m in re.finditer(r"\(\s*(?:NASDAQ:\s*|Nasdaq:\s*)?\[?\[?([A-Z]{1,5}(?:\.[A-Z])?)\]?\]?\s*\)", s):
                found.add(m.group(1))
        if s.startswith("|") and not s.startswith(("|-", "|}", "|+")):
            for cell in re.split(r"\|\|", s.lstrip("|")):
                cell = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", cell).strip()
                cell = cell.split("|")[-1].strip()
                if re.fullmatch(r"[A-Z]{1,5}(?:\.[A-Z])?", cell):
                    found.add(cell)
    return found


def wiki_revision_at(ts: str) -> tuple[int, str, str] | None:
    r = requests.get(WIKI_API, headers=WIKI_UA, timeout=30, params={
        "action": "query", "prop": "revisions", "titles": "Nasdaq-100", "rvlimit": 1, "rvstart": ts,
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


def update_membership() -> pd.DataFrame:
    """Monthly snapshots of Nasdaq-100 membership reconstructed from Wikipedia's revision history."""
    have = pd.read_csv(MEMBERSHIP, dtype=str) if MEMBERSHIP.exists() else pd.DataFrame(columns=["month", "revid", "timestamp", "count", "tickers"])
    done = set(have["month"])
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
        try:
            got = wiki_revision_at(ts)
        except Exception as e:  # noqa: BLE001
            print(f"membership {key}: {e}", file=sys.stderr)
            fails += 1
            time.sleep(2)
            continue
        fails = 0
        time.sleep(0.2)
        if not got:
            continue
        revid, stamp, text = got
        syms = {s.replace(".", "-") for s in wiki_tickers(text)} - {"NDX", "QQQ", "NASDAQ", "ETF", "US", "USD", "CEO", "S", "P"}
        if not (85 <= len(syms) <= 115):
            print(f"membership {key}: revision {revid} gave {len(syms)} symbols, skipped")
            continue
        rows.append({"month": key, "revid": str(revid), "timestamp": stamp, "count": str(len(syms)), "tickers": " ".join(sorted(syms))})
    if rows:
        new = pd.DataFrame(rows)
        have = pd.concat([have[~have["month"].isin(new["month"])], new]).sort_values("month")
        have.to_csv(MEMBERSHIP, index=False)
    print(f"membership: {len(have)} monthly snapshots ({have['month'].min()} .. {have['month'].max()})")
    return have


# ------------------------------------------------------------------ macro / factors / shares

def fetch_macro() -> None:
    MACRO.mkdir(parents=True, exist_ok=True)
    for sid in ("CPIAUCSL", "DTB3"):
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
    for name, fn in (("ff5_daily", "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"),
                     ("mom_daily", "F-F_Momentum_Factor_daily_CSV.zip")):
        try:
            z = zipfile.ZipFile(io.BytesIO(requests.get(base + fn, headers=UA, timeout=120).content))
            raw = z.read(z.namelist()[0]).decode("latin-1").splitlines()
            start = next(i for i, l in enumerate(raw) if re.match(r"^\s*,", l) or l.lower().startswith(",mkt") or "Mkt-RF" in l or "Mom" in l)
            rows = []
            header = [h.strip() for h in raw[start].split(",")]
            for l in raw[start + 1:]:
                parts = [x.strip() for x in l.split(",")]
                if len(parts) != len(header) or not re.fullmatch(r"\d{8}", parts[0]):
                    if rows:
                        break
                    continue
                rows.append(parts)
            df = pd.DataFrame(rows, columns=["date"] + header[1:])
            df["date"] = pd.to_datetime(df["date"], format="%Y%m%d").dt.strftime("%Y-%m-%d")
            for c in header[1:]:
                df[c] = pd.to_numeric(df[c]) / 100.0
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

def main() -> None:
    PRICES.mkdir(parents=True, exist_ok=True)
    ndx, source = constituents()
    try:
        membership = update_membership()
        hist_members = sorted({t for row in membership["tickers"] for t in row.split()})
    except Exception as e:  # noqa: BLE001
        print(f"membership history failed: {e}", file=sys.stderr)
        membership, hist_members = None, []
    former = [t for t in hist_members if t not in ndx]
    tickers = ndx + [t for t in EXTRA if t not in ndx]
    print(f"{len(ndx)} current constituents from {source}; {len(former)} former members; {len(tickers)} other tickers")

    ok, failed = {}, []
    for t in tickers:
        df = fetch_with_retry(t)
        if df is None:
            failed.append(t)
            continue
        ok[t] = save_prices(t, df)
        print(f"{t:6s} {ok[t]}")

    # former members: fetch under the (possibly renamed) current symbol, store under the old symbol
    former_ok, former_missing = {}, []
    for t in former:
        src = RENAMES.get(t, t)
        if src in ok and src != t:
            df = pd.read_csv(PRICES / f"{src}.csv", parse_dates=["date"], index_col="date")
        else:
            df = fetch_with_retry(src, tries=2)
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

    meta = {
        "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "constituent_source": source,
        "nasdaq100": ndx,
        "etfs": [t for t in ETFS if t in ok],
        "indexes": [t for t in INDEXES if t in ok],
        "benchmarks": ["SPY", "QQQ"],
        "former_members": sorted(former_ok),
        "former_members_missing_data": sorted(former_missing),
        "tickers": {**ok, **former_ok},
        "failed": failed,
        "dropped_stale": stale,
    }
    (ROOT / "data" / "universe.json").write_text(json.dumps(meta, indent=1))
    print(f"done: {len(ok)} current/ETF ok, {len(former_ok)} former members ok, "
          f"{len(former_missing)} former members without data, {len(failed)} failed {failed}")
    if len(ok) < len(tickers) * 0.85:
        sys.exit(1)


if __name__ == "__main__":
    main()
