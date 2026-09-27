"""pct_positive_days counts the days in the market, not the cash days earning interest."""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtester import metrics


def test_mostly_cash_strategy_is_not_99_percent_positive():
    idx = pd.bdate_range("2019-01-01", periods=500)
    r = metrics.rf_daily(idx, "tbill").to_numpy().copy()   # cash interest (T-bills) every day ...
    rng = np.random.default_rng(0)
    inv = rng.choice(len(idx), 50, replace=False)
    r[inv] = rng.normal(0, 0.01, 50)                       # ... and 50 days in the market
    eq = pd.Series(10_000 * np.cumprod(1 + r), index=idx)
    st = metrics.equity_stats(eq, rf=0.02)
    ups = (r[inv] > 0).mean()
    assert abs(st["pct_positive_days"] - ups) < 0.03
    assert st["pct_positive_days"] < 0.8
    assert 0.08 < st["active_days"] < 0.12
