"""Round 8 (Portfolio Visualizer expert review): historical withdrawal rates independent of the amount entered,
weight-first / N/M / asset-class phrases and month-name dates, more lazy portfolios, one frequency in the
correlation tool's asset table, calendar-month lookbacks, and report cosmetics (monthly max drawdown, labels)."""
import numpy as np
import pandas as pd
import pytest

from backtester import data, metrics, parser, report, runner
from backtester.portfolio import Portfolio

AVAIL = set(data.available_tickers())
needs = lambda *ts: pytest.mark.skipif(not set(ts) <= AVAIL, reason="price data not downloaded")  # noqa: E731


def port(text) -> Portfolio:
    p = parser.parse(text)
    assert isinstance(p, Portfolio), text
    return p


# ------------------------------------------------------------------ 1. historical SWR / PWR

def _independent_swr_6040_1966() -> float:
    """60/40 SPYSIM/IEFSIM rebalanced on the first day of each year, from 1966; month-end returns; the highest
    CPI-indexed yearly withdrawal (share of the start, taken at the start of each year) that never runs out."""
    df = pd.concat([data.load("SPYSIM")["close"], data.load("IEFSIM")["close"]], axis=1, keys=["s", "b"],
                   sort=True).dropna()
    r = df.pct_change().dropna()
    r = r[r.index >= "1966-01-01"]
    hs, hb, yr, nav = 0.6, 0.4, None, []
    for d, s_, b_ in zip(r.index, r["s"].to_numpy(), r["b"].to_numpy()):
        if yr is not None and d.year != yr:
            t = hs + hb
            hs, hb = 0.6 * t, 0.4 * t
        yr = d.year
        hs *= 1 + s_
        hb *= 1 + b_
        nav.append(hs + hb)
    me = pd.Series(nav, r.index).resample("ME").last()
    mr = me.pct_change()
    mr.iloc[0] = me.iloc[0] - 1
    c = data.cpi()
    c.index = c.index.to_period("M").to_timestamp("M")
    inf = c.pct_change().reindex(mr.index.to_period("M").to_timestamp("M")).fillna(0).to_numpy()
    ci = np.concatenate([[1.0], np.cumprod(1 + inf)])
    R = mr.to_numpy()

    def lasts(rate):
        bal = 1.0
        for m in range(len(R)):
            if m % 12 == 0:
                bal -= rate * ci[m]
                if bal <= 0:
                    return False
            bal *= 1 + R[m]
        return True
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if lasts(mid) else (lo, mid)
    return lo


@needs("SPYSIM", "IEFSIM")
def test_historical_swr_does_not_depend_on_the_amount_withdrawn():
    want = _independent_swr_6040_1966()
    assert 0.03 < want < 0.04
    got = {}
    for w in ("$60,000 a year adjusted for inflation", "4% per year adjusted for inflation",
              "$30,000 a year adjusted for inflation"):
        p = port(f"hold 60% SPYSIM and 40% IEFSIM since 1966, rebalance yearly, starting with $1,000,000, withdraw {w}")
        A = report.analyze(runner.run(p), sensitivity=False, mc=False, detail=False)
        wr = A["withdrawal_rates"]
        got[w] = wr["swr"]
        # the full requested period, even when $60,000 a year empties the account around 1981
        assert str(wr["from"]).startswith("1966") and wr["to"].year >= 2020, wr
        assert wr["basis"] == "flow_free_returns"
        assert "over the full period 1966" in report.console_summary(A)
    vals = list(got.values())
    assert max(vals) - min(vals) < 1e-9, got
    assert vals[0] == pytest.approx(want, abs=5e-4)


@needs("VTISIM", "BNDSIM")
def test_contribute_then_withdraw_rate_uses_the_whole_withdrawal_phase():
    rates = []
    for amt in ("$50,000", "$20,000"):
        p = port(f"add $1,000 a month for 20 years, then withdraw {amt} a year adjusted for inflation, "
                 "hold 60% VTISIM and 40% BNDSIM, since 1980, starting with $10,000")
        A = report.analyze(runner.run(p), sensitivity=False, mc=False, detail=False)
        wr = A["withdrawal_rates"]
        assert wr["base"] == "withdrawal_start" and wr["base_date"].year == 2000 and wr["to"].year >= 2020
        rates.append((wr["swr"], wr["pwr"]))
    assert rates[0] == pytest.approx(rates[1], abs=1e-12)
    assert rates[0][0] < 0.07          # not the 7.72% measured only up to the 2012 depletion


def test_monthly_withdrawals_pay_the_same_yearly_amount_in_instalments():
    from backtester import montecarlo as mc
    P = np.zeros((1, 120))
    ci = np.ones((1, 121))
    # no growth, no inflation: 10 years of withdrawals last exactly at 10% a year, monthly or yearly
    assert mc.safe_withdrawal_rate(P, ci, 1.0, 1.0, every=12) == pytest.approx(0.1, abs=1e-6)
    assert mc.safe_withdrawal_rate(P, ci, 1.0, 1.0, every=1) == pytest.approx(0.1, abs=1e-6)
