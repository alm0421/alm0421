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

EXTRA = ["QQQ", "SPY"]


TICKER_RE = r"^[A-Z]{1,5}([.-][A-Z])?$"


def tickers_from_html(url: str) -> list[str]:
    """Find a column of ~100 ticker-shaped strings (including MSFT) in any table on the page."""
    html = requests.get(url, headers=UA, timeout=30).text
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
    ("wikipedia", lambda: tickers_from_html("https://en.wikipedia.org/wiki/Nasdaq-100")),
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


def main() -> None:
    PRICES.mkdir(parents=True, exist_ok=True)
    ndx, source = constituents()
    tickers = ndx + [t for t in EXTRA if t not in ndx]
    print(f"{len(ndx)} Nasdaq-100 constituents from {source}; fetching {len(tickers)} tickers")
    ok, failed = {}, []
    for t in tickers:
        for attempt in range(3):
            try:
                df = fetch(t)
                df.round(6).to_csv(PRICES / f"{t}.csv", float_format="%.6g")
                ok[t] = {"first": str(df.index[0].date()), "last": str(df.index[-1].date()), "rows": len(df)}
                print(f"{t:6s} {ok[t]}")
                break
            except Exception as e:  # noqa: BLE001
                print(f"{t}: attempt {attempt + 1} failed: {e}", file=sys.stderr)
                time.sleep(2 * (attempt + 1))
        else:
            failed.append(t)
    # Drop constituents whose history stopped (delisted / acquired) or is too short to be real.
    last_bench = max(pd.Timestamp(ok[b]["last"]) for b in EXTRA if b in ok)
    stale = [
        t for t in ndx
        if t in ok and ((last_bench - pd.Timestamp(ok[t]["last"])).days > 10 or ok[t]["rows"] < 5)
    ]
    for t in stale:
        print(f"dropping {t}: stale or too little data {ok[t]}")
        (PRICES / f"{t}.csv").unlink(missing_ok=True)
        ok.pop(t)
    ndx = [t for t in ndx if t in ok]
    # remove files for tickers no longer in the universe
    for f in PRICES.glob("*.csv"):
        if f.stem not in ok:
            f.unlink()
    meta = {
        "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "constituent_source": source,
        "nasdaq100": ndx,
        "benchmarks": EXTRA,
        "tickers": ok,
        "failed": failed,
        "dropped_stale": stale,
    }
    (ROOT / "data" / "universe.json").write_text(json.dumps(meta, indent=1))
    print(f"done: {len(ok)} ok, {len(failed)} failed {failed}")
    if len(ok) < len(tickers) * 0.85:
        sys.exit(1)


if __name__ == "__main__":
    main()
