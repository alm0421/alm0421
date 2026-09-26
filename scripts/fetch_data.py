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
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) backtester-data-fetch"}

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


def ndx_from_wikipedia() -> list[str]:
    html = requests.get("https://en.wikipedia.org/wiki/Nasdaq-100", headers=UA, timeout=30).text
    for t in pd.read_html(io.StringIO(html)):
        cols = [str(c).lower() for c in t.columns]
        for key in ("ticker", "symbol"):
            if key in cols:
                syms = t.iloc[:, cols.index(key)].astype(str).str.strip().tolist()
                if 90 <= len(syms) <= 110:
                    return syms
    raise RuntimeError("constituents table not found")


def ndx_from_nasdaq() -> list[str]:
    r = requests.get("https://api.nasdaq.com/api/quote/list-type/nasdaq100", headers=UA, timeout=30)
    rows = r.json()["data"]["data"]["rows"]
    return [row["symbol"].strip() for row in rows]


def constituents() -> tuple[list[str], str]:
    for name, fn in (("wikipedia", ndx_from_wikipedia), ("nasdaq.com", ndx_from_nasdaq)):
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
    meta = {
        "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "constituent_source": source,
        "nasdaq100": ndx,
        "benchmarks": EXTRA,
        "tickers": ok,
        "failed": failed,
    }
    (ROOT / "data" / "universe.json").write_text(json.dumps(meta, indent=1))
    print(f"done: {len(ok)} ok, {len(failed)} failed {failed}")
    if len(ok) < len(tickers) * 0.9:
        sys.exit(1)


if __name__ == "__main__":
    main()
