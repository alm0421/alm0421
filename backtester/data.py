"""Load daily price history from data/prices/*.csv."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PRICES = ROOT / "data" / "prices"
UNIVERSE_FILE = ROOT / "data" / "universe.json"

BENCHMARKS = ["SPY", "QQQ"]


def available_tickers() -> list[str]:
    return sorted(p.stem for p in PRICES.glob("*.csv"))


def nasdaq100() -> list[str]:
    if UNIVERSE_FILE.exists():
        return list(json.loads(UNIVERSE_FILE.read_text())["nasdaq100"])
    return [t for t in available_tickers() if t not in BENCHMARKS]


@lru_cache(maxsize=None)
def load(ticker: str) -> pd.DataFrame:
    """Return a DataFrame indexed by date with total-return adjusted OHLC.

    Columns: open, high, low, close (split+dividend adjusted), volume,
    quote_close (split-adjusted close as quoted, no dividend adjustment),
    dividend, split.
    """
    path = PRICES / f"{ticker.upper()}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No price data for {ticker!r}. Run scripts/fetch_data.py or the "
            f"'Fetch price data' GitHub Action. Available: {', '.join(available_tickers())[:300]}"
        )
    raw = pd.read_csv(path, parse_dates=["date"], index_col="date").sort_index()
    raw = raw[(raw["close"] > 0) & raw[["open", "high", "low", "close"]].notna().all(axis=1)]
    factor = (raw["adj_close"] / raw["close"]).fillna(1.0)
    df = pd.DataFrame(index=raw.index)
    # Some very old Yahoo bars have open == 0; fall back to the close.
    opn = raw["open"].where(raw["open"] > 0, raw["close"])
    df["open"] = opn * factor
    df["high"] = pd.concat([raw["high"], opn, raw["close"]], axis=1).max(axis=1) * factor
    low = raw["low"].where(raw["low"] > 0)
    df["low"] = pd.concat([low, opn, raw["close"]], axis=1).min(axis=1) * factor
    df["close"] = raw["adj_close"]
    df["volume"] = raw["volume"].fillna(0)
    df["quote_close"] = raw["close"]
    df["dividend"] = raw.get("dividend", 0.0)
    df["split"] = raw.get("split", 0.0)
    return df


def load_many(tickers: list[str]) -> dict[str, pd.DataFrame]:
    out = {}
    for t in tickers:
        out[t] = load(t)
    return out
