"""Allocation engine, round 3: filters and weightings over groups (synthetic NAVs), strict node
validation, maintenance margin / liquidation, semiannual rebalancing, relative drift bands and
cash-flow windows."""
import time

import numpy as np
import pandas as pd
import pytest

from backtester import data
from backtester import portfolio as pf

HAVE = {"SPY", "QQQ", "TLT", "GLD", "IEF", "EFA"} <= set(data.available_tickers())
needs_data = pytest.mark.skipif(not HAVE, reason="price data not downloaded")


# ------------------------------------------------------------ synthetic data

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


def walk(seed, n=500, drift=0.0003, vol=0.012):
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


def four_assets(fake, n=500, cut=None):
    for k, (t, s) in enumerate(zip("ABCD", (1, 2, 3, 4))):
        df = frame(walk(s, n, drift=0.0002 * (k + 1), vol=0.006 * (k + 1)))
        fake[t] = df if cut is None else df[df.index <= cut]


G1 = {"weights": "equal", "children": [{"asset": "A"}, {"asset": "B"}]}
G2 = {"weights": "equal", "children": [{"asset": "C"}, {"asset": "D"}]}


def hand_nav(fake, names):
    r = pd.concat([fake[t]["close"].pct_change() for t in names], axis=1).fillna(0).mean(axis=1)
    return (1 + r).cumprod()


# ------------------------------------------------------------ filters over groups

def test_filter_over_groups_ranks_on_the_groups_synthetic_nav(fake):
    four_assets(fake)
    tree = {"filter": {"select": "top", "n": 1, "by": "tret(tr, 10)"}, "universe": "children",
            "children": [G1, G2], "fallback": {"cash": True}}
    p = pf.Portfolio(tree=tree, rebalance="daily", cash_rate=None)
    r = pf.run(p)
    n1, n2 = hand_nav(fake, "AB"), hand_nav(fake, "CD")
    m1, m2 = n1 / n1.shift(10) - 1, n2 / n2.shift(10) - 1
    h = r.holdings
    checked = 0
    for d in h.index[11:]:
        want = ("A", "B") if m1[d] > m2[d] else ("C", "D")
        other = ("C", "D") if want == ("A", "B") else ("A", "B")
        assert h.loc[d, list(want)].to_numpy() == pytest.approx([0.5, 0.5], abs=1e-9), d
        assert h.reindex(columns=list(other), fill_value=0.0).loc[d].abs().sum() < 1e-9
        checked += 1
    assert checked > 400
    # the evaluator's NAV is exactly the independent daily-rebalanced equal-weight NAV
    ev = pf._Evaluator(p, fake["A"].index, {t: fake[t] for t in "ABCD"})   # (the run itself starts after the warm-up)
    np.testing.assert_allclose(ev.nav(G1), n1.to_numpy() / n1.iloc[0], rtol=1e-12)
    assert "groups ranked on their simulated daily NAV" in p.summary()


def test_nested_filters_and_if_nodes_as_filter_children(fake):
    four_assets(fake)
    fake["SPY"] = frame(walk(9, 500))
    inner = {"filter": {"select": "bottom", "n": 1, "by": "stdev_return(tr, 20)"}, "universe": "children",
             "children": [{"asset": "C"}, {"asset": "D"}]}
    cond = {"if": "close > sma(close, 50)", "on": "SPY", "then": {"asset": "A"}, "else": {"asset": "B"}}
    tree = {"filter": {"select": "top", "n": 2, "by": "ma_return(tr, 5)", "weights": "inverse_vol", "lookback": 10},
            "universe": "children", "children": [inner, cond, G1, {"asset": "D"}]}
    pf.validate_node(tree)
    for by in ("tret(tr, 10)", "rsi(close, 10)", "stdev_return(tr, 20)", "max_drawdown(tr, 20)", "ma_return(tr, 5)"):
        tree["filter"]["by"] = by
        r = pf.run(pf.Portfolio(tree=tree, rebalance="daily", cash_rate=None))
        w = r.holdings.drop(columns="cash").sum(axis=1).iloc[60:]
        assert w.to_numpy() == pytest.approx(1.0, abs=1e-9), by   # always fully invested in the picks


def test_inverse_vol_and_optimisers_over_groups_use_synthetic_returns(fake):
    four_assets(fake)
    tree = {"weights": "inverse_vol", "lookback": 10, "children": [G1, G2, {"asset": "D"}]}
    p = pf.Portfolio(tree=tree, rebalance="monthly", cash_rate=None)
    cal = fake["A"].index
    ev = pf._Evaluator(p, cal, {t: fake[t] for t in "ABCD"})
    i = 300
    vol = lambda s: s.pct_change().rolling(10).std().iloc[i] * np.sqrt(252)  # noqa: E731
    v = np.array([vol(hand_nav(fake, "AB")), vol(hand_nav(fake, "CD")), vol(fake["D"]["close"])])
    wg = (1 / v) / (1 / v).sum()
    w = ev.eval(tree, i)
    assert w["A"] == pytest.approx(wg[0] / 2, rel=1e-9)
    assert w["C"] == pytest.approx(wg[1] / 2, rel=1e-9)
    assert w["D"] == pytest.approx(wg[1] / 2 + wg[2], rel=1e-9)
    assert "equal-weighted" not in " ".join(p.notes)
    assert "groups measured on their simulated daily returns" in p.summary()
    for m in ("risk_parity", "min_variance", "max_sharpe", "max_diversification"):
        t2 = {"weights": m, "lookback": 60, "children": [G1, G2]}
        w = ev.eval(t2, i)
        assert sum(w.values()) == pytest.approx(1.0)
        assert w.get("A", 0) == pytest.approx(w.get("B", 0)) and w.get("C", 0) == pytest.approx(w.get("D", 0))
    r = pf.run(pf.Portfolio(tree=tree, rebalance="monthly", cash_rate=None))
    assert r.equity.iloc[-1] > 0


def test_group_filter_does_not_depend_on_future_data(fake):
    tree = {"filter": {"select": "top", "n": 1, "by": "tret(tr, 20)", "weights": "inverse_vol"},
            "universe": "children", "children": [G1, G2, {"filter": {"select": "bottom", "n": 1, "by": "rsi(close, 5)"},
                                                          "universe": "children", "children": [{"asset": "A"}, {"asset": "D"}]}]}
    four_assets(fake)
    start = str(fake["A"].index[250].date())
    full = pf.run(pf.Portfolio(tree=tree, rebalance="daily", cash_rate=None, start=start))
    cut = fake["A"].index[400]
    four_assets(fake, cut=cut)
    past = pf.run(pf.Portfolio(tree=tree, rebalance="daily", cash_rate=None, start=start))
    a = full.equity[full.equity.index < cut]
    np.testing.assert_allclose(a.to_numpy(), past.equity.reindex(a.index).to_numpy(), rtol=1e-12)
    # the warm-up: the group NAVs were simulated before the start, so the first day is already invested
    assert full.holdings.drop(columns="cash").iloc[0].sum() == pytest.approx(1.0)


# ------------------------------------------------------------ validation

def test_unknown_keys_are_rejected_with_the_allowed_list(fake):
    four_assets(fake)
    with pytest.raises(ValueError, match=r"'bogus'.*allowed: asset, name"):
        pf.validate_node({"asset": "A", "bogus": 1})
    with pytest.raises(ValueError, match="allowed: allow_short, children, lookback, name, w, weights"):
        pf.validate_node({"weights": "equal", "children": [{"asset": "A"}], "weight": [1]})
    with pytest.raises(ValueError, match="filter settings"):
        pf.validate_node({"filter": {"by": "tret(10)", "top": 3}, "children": [{"asset": "A"}]})
    with pytest.raises(ValueError, match="'fallback'"):
        pf.validate_node({"if": "close > 1", "on": "A", "then": {"asset": "A"}, "else": {"cash": True}, "fallback": {}})
    with pytest.raises(ValueError, match="by"):
        pf.validate_node({"filter": {"select": "top"}, "children": [{"asset": "A"}]})
    pf.validate_node({"filter": {"by": "tret(10)"}, "children": [G1, {"cash": True}, {"asset": "C"}], "name": "x"})


# ------------------------------------------------------------ maintenance margin and liquidation

def _gross_ok(r, cap):
    live = r.equity.iloc[1:] > 0
    return (r.exposure.iloc[1:][live] <= cap + 1e-9).all()


def _identity(r):
    a = r.extras["attribution"]
    lhs = a["pnl"].sum() + r.interest - r.extras["fees"]
    rhs = r.equity.iloc[-1] - r.equity.iloc[0] - r.extras["flows"].sum()
    return lhs == pytest.approx(rhs, abs=1e-6)


def test_four_x_through_a_30pct_slide_gets_margin_calls(fake):
    px = [100.0] * 5 + [100 * 0.95 ** k for k in range(1, 8)] + [100 * 0.95 ** 7] * 10   # -30% over 7 days
    fake["X"] = frame(px)
    # 4x needs portfolio margin and a maintenance margin below 25% (the same rules as signal strategies)
    p = pf.Portfolio(tree={"asset": "X"}, rebalance="none", leverage=4, cash_rate=None, margin_account="portfolio",
                     maintenance_margin=0.2)
    r = pf.run(p)
    mc = r.orders[r.orders.reason == "margin call"]
    assert len(mc) >= 3 and (mc.side == "sell").all()
    assert _gross_ok(r, 1 / p.maintenance_margin)
    assert (r.equity > 0).all()
    assert any(n.startswith("Margin call on") for n in p.notes)
    assert _identity(r)


def test_three_x_through_a_30pct_day_is_cut_back_and_four_x_is_wiped_out(fake):
    fake["X"] = frame([100, 100, 100, 70, 70, 75, 80])
    p = pf.Portfolio(tree={"asset": "X"}, rebalance="none", leverage=3, cash_rate=None, margin_account="portfolio")
    r = pf.run(p)
    d = fake["X"].index[3]
    assert r.equity[d] == pytest.approx(10_000 * (1 - 3 * 0.3))
    mc = r.orders[r.orders.reason == "margin call"]
    assert list(pd.to_datetime(mc.date)) == [d]
    assert r.exposure[d] == pytest.approx(3.0)
    assert _gross_ok(r, 4.0) and _identity(r)

    p4 = pf.Portfolio(tree={"asset": "X"}, rebalance="none", leverage=4, cash_rate=None, margin_account="portfolio",
                      maintenance_margin=0.2)
    r4 = pf.run(p4)
    assert (r4.equity >= 0).all()
    assert r4.equity[d:].eq(0).all()
    assert any("wiped out" in n for n in p4.notes)


def test_short_squeeze_margin_call_covers_the_short(fake):
    fake["L"] = frame([100] * 8)
    fake["S"] = frame([100, 100, 100, 200, 200, 200, 200, 200])
    tree = {"weights": "specified", "w": [1.5, -0.5], "children": [{"asset": "L"}, {"asset": "S"}]}
    p = pf.Portfolio(tree=tree, rebalance="none", cash_rate=None)
    r = pf.run(p)
    mc = r.orders[r.orders.reason == "margin call"]
    assert set(mc.side) == {"sell", "buy"}     # sell some of the long, buy back some of the short
    d = fake["S"].index[3]
    assert r.equity[d] == pytest.approx(5_000)
    assert r.exposure[d] == pytest.approx(2.0)
    assert _gross_ok(r, 4.0) and _identity(r)
    assert "maintenance margin" in p.summary()


def test_leverage_drift_between_rebalances_is_capped_and_reported(fake):
    fake["X"] = frame(100 * np.cumprod([1] + [0.985] * 60))
    p = pf.Portfolio(tree={"asset": "X"}, rebalance="none", leverage=2, cash_rate=None)
    r = pf.run(p)
    assert _gross_ok(r, 4.0)
    assert r.exposure.max() > 3.0
    assert any(n.startswith("Between rebalances leverage drifted") for n in p.notes)
    p0 = pf.Portfolio(tree={"asset": "X"}, rebalance="none", leverage=2, cash_rate=None, maintenance_margin=0)
    r0 = pf.run(p0)
    assert (r0.orders.reason != "margin call").all() and r0.exposure.max() > 4


def test_leverage_above_the_maintenance_limit_is_refused(fake):
    fake["X"] = frame([100] * 5)
    with pytest.raises(ValueError, match="Regulation T"):
        pf.Portfolio(tree={"asset": "X"}, leverage=3).validate()
    with pytest.raises(ValueError, match="maintenance_margin"):
        pf.Portfolio(tree={"asset": "X"}, leverage=4, margin_account="portfolio").validate()   # 25% x 4 = 100%
    pf.Portfolio(tree={"asset": "X"}, leverage=4, margin_account="portfolio", maintenance_margin=0.15).validate()
    with pytest.raises(ValueError, match="portfolio-margin account allows"):
        pf.Portfolio(tree={"asset": "X"}, leverage=5, margin_account="portfolio", maintenance_margin=0.15).validate()


# ------------------------------------------------------------ schedules, bands, flows

def test_semiannual_rebalances_at_the_end_of_june_and_december(fake):
    fake["A"] = frame(walk(1, 700), start="2019-01-01")
    fake["B"] = frame(walk(2, 700), start="2019-01-01")
    p = pf.Portfolio(tree={"weights": "specified", "w": [0.5, 0.5], "children": [{"asset": "A"}, {"asset": "B"}]},
                     rebalance="semiannual", cash_rate=None)
    r = pf.run(p)
    days = sorted(set(pd.to_datetime(r.orders[r.orders.reason == "rebalance"].date)))
    assert days and all(d.month in (6, 12) for d in days)
    cal = fake["A"].index
    for d in days:
        nxt = cal[cal > d]
        assert len(nxt) == 0 or nxt[0].month != d.month
    assert "every six months" in p.summary()


def test_relative_drift_band(fake):
    fake["A"] = frame(100 * np.cumprod([1] + [1.01] * 200))
    fake["B"] = frame([100] * 201)
    p = pf.Portfolio(tree={"weights": "specified", "w": [0.5, 0.5], "children": [{"asset": "A"}, {"asset": "B"}]},
                     rebalance="none", drift_band_relative=0.2, cash_rate=None)
    r = pf.run(p)
    assert (r.orders.reason == "drift rebalance").sum() >= 5
    assert r.holdings["A"].max() <= 0.6 + 1e-9
    assert "20% of its own target" in p.summary()


def test_flow_windows_and_growth(fake):
    fake["A"] = frame([100.0] * 800, start="2019-01-01")
    p = pf.Portfolio(tree={"asset": "A"}, rebalance="none", cash_rate=None, contribution=100, contribution_freq="monthly",
                     contribution_end=2, contribution_growth=0.10, withdrawal=50, withdrawal_freq="monthly",
                     withdrawal_start=3)
    r = pf.run(p)
    fl = r.extras["flows"]
    c0 = fake["A"].index[0]
    pos, neg = fl[fl > 0], fl[fl < 0]
    assert pos.index.max() < c0 + pd.DateOffset(years=2) and neg.index.min() >= c0 + pd.DateOffset(years=2)
    assert set(np.round(pos[pos.index < c0 + pd.DateOffset(years=1)], 6)) == {100.0}
    assert set(np.round(pos[pos.index >= c0 + pd.DateOffset(years=1)], 6)) == {110.0}
    assert set(np.round(neg, 6)) == {-50.0}
    s = p.summary()
    assert "for the first 2 years, growing 10.0% a year" in s and "from year 3 of the backtest" in s
    # calendar years and dates
    p2 = pf.Portfolio(tree={"asset": "A"}, rebalance="none", cash_rate=None, contribution=100, contribution_start=2020,
                      contribution_end="2020-06-30")
    fl2 = pf.run(p2).extras["flows"]
    got = fl2[fl2 > 0].index
    assert len(got) == 6 and got.min() >= pd.Timestamp("2020-01-01") and got.max() <= pd.Timestamp("2020-06-30")
    with pytest.raises(ValueError):
        pf.Portfolio(tree={"asset": "A"}, contribution=100, contribution_start="soon").validate()


def test_summary_lists_all_costs(fake):
    fake["A"] = frame([100] * 5)
    s = pf.Portfolio(tree={"asset": "A"}, leverage=2, margin_rate=0.015, expense_ratio=0.002).summary()
    assert "0.20%/yr expense ratio" in s and "2x leverage" in s and "+ 1.50%" in s and "25% maintenance margin" in s


# ------------------------------------------------------------ real data

@needs_data
def test_group_filter_real_data_speed_accounting_and_no_lookahead(monkeypatch):
    g1 = {"weights": "equal", "children": [{"asset": "SPY"}, {"asset": "TLT"}]}
    g2 = {"filter": {"select": "top", "n": 1, "by": "tret(tr, 21)"}, "universe": "children",
          "children": [{"asset": "QQQ"}, {"asset": "GLD"}]}
    g3 = {"if": "close > sma(close, 200)", "on": "SPY", "then": {"asset": "QQQ"}, "else": {"asset": "IEF"}}
    tree = {"weights": "equal", "children": [
        {"filter": {"select": "top", "n": 1, "by": "tret(tr, 10)", "weights": "inverse_vol"}, "universe": "children",
         "children": [g1, g2, g3]},
        {"filter": {"select": "bottom", "n": 2, "by": "stdev_return(tr, 20)"}, "universe": "children",
         "children": [g1, {"asset": "GLD"}, g3, {"filter": {"select": "top", "n": 2, "by": "rsi(close, 10)"},
                                                 "universe": "children", "children": [{"asset": "SPY"}, {"asset": "EFA"}, {"asset": "TLT"}]}]}]}
    mk = lambda: pf.Portfolio(tree=tree, rebalance="daily", start="2015-01-01", end="2019-12-31",  # noqa: E731
                              commission=1.0, slippage_bps=2)
    t0 = time.time()
    full = pf.run(mk())
    assert time.time() - t0 < 10
    assert _identity(full)
    cut = pd.Timestamp("2017-06-30")
    real = data.load
    cache = {}

    def load(t):
        t = data.canonical(t)
        if t not in cache:
            df = real(t)
            cache[t] = df[df.index <= cut]
        return cache[t]
    monkeypatch.setattr(data, "load", load)
    monkeypatch.setattr(data, "load_many", lambda ts: {data.canonical(t): load(t) for t in ts})
    past = pf.run(mk())
    a = full.equity[full.equity.index < cut - pd.Timedelta(days=3)]
    np.testing.assert_allclose(a.to_numpy(), past.equity.reindex(a.index).to_numpy(), rtol=1e-12)


@needs_data
def test_four_x_qqq_since_2000_never_exceeds_the_margin_limit():
    p = pf.Portfolio(tree={"asset": "QQQ"}, rebalance="none", leverage=4, start="2000-01-01", end="2010-12-31",
                     margin_account="portfolio", maintenance_margin=0.2)
    r = pf.run(p)
    assert _gross_ok(r, 5.0)
    assert (r.equity >= 0).all()
    assert (r.orders.reason == "margin call").any()
