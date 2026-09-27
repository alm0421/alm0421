"""Portfolio Visualizer review, round 10: time-weighted returns with cash flows, asset-class names in the
analysis tools, custom series as benchmarks and ending early, partial benchmark years, grid labels and
proxies before a sleeve's inception."""
import numpy as np
import pandas as pd
import pytest

from backtester import data, metrics, report
from backtester import portfolio as pf

HAVE = {"SPY", "QQQ"} <= set(data.available_tickers())
needs_data = pytest.mark.skipif(not HAVE, reason="price data not downloaded")


def frame(closes, start="2019-01-01"):
    closes = np.asarray(closes, dtype=float)
    idx = pd.bdate_range(start, periods=len(closes))
    df = pd.DataFrame({"open": closes, "high": closes, "low": closes, "close": closes}, index=idx)
    df["volume"] = 1e6
    df["quote_close"] = df["close"]
    df["adj_close"] = df["close"]
    df["dividend"] = 0.0
    df["split"] = 0.0
    return df


def walk(seed, n=700, drift=0.0003, vol=0.015):
    r = np.random.default_rng(seed).normal(drift, vol, n)
    r[0] = 0.0
    return 100 * np.cumprod(1 + r)


@pytest.fixture
def fake(monkeypatch):
    frames = {}

    def load(t):
        return frames[data.canonical(t)]

    monkeypatch.setattr(data, "load", load)
    monkeypatch.setattr(data, "load_many", lambda ts: {data.canonical(t): load(t) for t in ts})
    return frames


# ------------------------------------------------------------ 1. TWR convention = flows at the close

FLOWS = [dict(contribution=500.0, contribution_freq="monthly"),
         dict(withdrawal=300.0, withdrawal_freq="monthly"),
         dict(withdrawal_pct=0.10, withdrawal_freq="yearly"),
         dict(contribution=1000.0, contribution_freq="quarterly", withdrawal=200.0, withdrawal_freq="monthly")]


@pytest.mark.parametrize("fill", ["close", "next_open"])
@pytest.mark.parametrize("fl", FLOWS)
def test_single_asset_twr_is_the_assets_own_return_whatever_the_flows(fake, fl, fill):
    """Invariant: a portfolio 100% in one asset has a time-weighted index identical (1e-9) to the asset's own
    total-return index, and to its buy-and-hold benchmark receiving the same flows."""
    fake["A"] = frame(walk(3))
    p = pf.Portfolio(tree={"asset": "A"}, rebalance="none", cash_rate=None, capital=10_000, fill=fill, **fl)
    res = pf.run(p)
    flows = res.extras["flows"]
    assert flows.abs().sum() > 0
    nv = metrics.nav(res.equity, flows)
    px = fake["A"]["close"].reindex(nv.index)
    # from the first full day in the market (a next-open start buys on the first day's open)
    k = 2 if fill == "next_open" else 1
    got = nv.iloc[k:] / nv.iloc[k]
    want = px.iloc[k:] / px.iloc[k]
    np.testing.assert_allclose(got.to_numpy(), want.to_numpy(), rtol=1e-9)
    # the benchmark with the same flows (bought at the portfolio's first value) has the same index
    growth = fake["A"]["close"].reindex(res.equity.index).bfill()
    b = report.benchmarks_with_flows({"A": growth}, flows, res.equity)["A"]
    bn = metrics.nav(b, flows)
    np.testing.assert_allclose((bn.iloc[k:] / bn.iloc[k]).to_numpy(), want.to_numpy(), rtol=1e-9)


def test_withdrawal_that_empties_the_account_keeps_the_market_move(fake):
    fake["A"] = frame(walk(5, drift=-0.001))
    p = pf.Portfolio(tree={"asset": "A"}, rebalance="none", cash_rate=None, capital=10_000,
                     withdrawal=2_000, withdrawal_freq="quarterly")
    res = pf.run(p)
    dep = res.extras["depleted"]
    assert dep is not None
    nv = metrics.nav(res.equity, res.extras["flows"])
    px = fake["A"]["close"].reindex(nv.index)
    cut = nv.index <= dep
    np.testing.assert_allclose((nv[cut] / nv.iloc[1]).iloc[1:].to_numpy(), (px[cut] / px.iloc[1]).iloc[1:].to_numpy(),
                               rtol=1e-9)
    assert (nv[~cut] == nv[cut].iloc[-1]).all()      # nothing invested afterwards: no return


def test_twr_formula():
    idx = pd.bdate_range("2020-01-01", periods=4)
    eq = pd.Series([100.0, 160.0, 150.0, 0.0], index=idx)
    fl = pd.Series([0.0, 50.0, -20.0, -140.0], index=idx)
    r = metrics.twr_returns(eq, fl)
    assert r.tolist() == pytest.approx([110 / 100 - 1, 170 / 160 - 1, 140 / 150 - 1])


@needs_data
def test_qqq_with_contributions_has_the_benchmarks_cagr():
    from backtester import parser, runner
    res = runner.run(parser.parse("buy and hold QQQ, add $5,000 every month, starting with $1,000, since 2000"))
    A = report.analyze(res, sensitivity=False, mc=False)
    nv = A["nav"]
    c = data.load("QQQ")["adj_close"].reindex(nv.index).ffill()
    years = (nv.index[-1] - nv.index[0]).days / 365.25
    want = (c.iloc[-1] / c.iloc[0]) ** (1 / years) - 1
    assert A["stats"]["cagr"] == pytest.approx(want, abs=2e-4)


@needs_data
def test_qqq_withdrawals_cagr_until_depleted():
    from backtester import parser, runner
    res = runner.run(parser.parse("hold 100% QQQ, withdraw 10% per year adjusted for inflation, starting with "
                                  "$1,000,000, since 2000"))
    A = report.analyze(res, sensitivity=False, mc=False)
    assert A["stats"]["cagr"] == pytest.approx(-0.2054, abs=0.002)
