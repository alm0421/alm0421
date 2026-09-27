"""Round 9 (QuantConnect review): a causal total-return index `tr`, one margin model for both engines (Reg T /
portfolio margin, leveraged-ETF requirements, margin calls with a cushion), borrow-fee phrases, share classes of one
company as one name, survivorship coverage from the first real session, synthetic opens, no interest during the
signal warm-up, broker catch-up orders and faster Python-function rules."""
import numpy as np
import pandas as pd
import pytest

from backtester import broker, data, engine, expr, margin, parser, report
from backtester import portfolio as pf
from backtester.strategy import Strategy

HAVE = {"MSFT", "SPY", "AAPL", "QQQ", "TQQQ"} <= set(data.available_tickers())
needs_data = pytest.mark.skipif(not HAVE, reason="price data not downloaded")


def frame(closes, start="2005-01-03", dividends=None):
    closes = np.asarray(closes, dtype=float)
    idx = pd.bdate_range(start, periods=len(closes))
    opens = np.r_[closes[0], closes[:-1]] * 1.001
    df = pd.DataFrame({"open": opens, "high": np.maximum(opens, closes) * 1.004,
                       "low": np.minimum(opens, closes) * 0.996, "close": closes}, index=idx)
    df["volume"] = 1e6
    df["quote_close"] = df["close"]
    df["dividend"] = 0.0 if dividends is None else np.asarray(dividends, dtype=float)
    df["adj_close"] = yahoo_adjusted(df["close"], df["dividend"])
    df["split"] = 1.0
    return df


def walk(seed, n=600, drift=0.0003, vol=0.012):
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
    expr._STREAMED.clear()
    return frames


def yahoo_adjusted(close: pd.Series, dividend: pd.Series) -> pd.Series:
    """Yahoo's back-adjusted close as it would be published with data up to the last row: each price is scaled by
    (1 - dividend / previous close) for every dividend paid AFTER it (so its level depends on later dividends)."""
    c = close.to_numpy(dtype=float)
    d = dividend.fillna(0.0).to_numpy(dtype=float)
    f = np.ones(len(c))
    f[1:] = np.where(d[1:] > 0, 1 - d[1:] / c[:-1], 1.0)
    after = np.r_[np.cumprod(f[::-1])[::-1][1:], 1.0]      # product of the factors of later ex-dates
    return pd.Series(c * after, index=close.index)


# ------------------------------------------------------------ 1. tr is causal

def test_tr_recomputed_on_truncated_raw_data_matches_up_to_the_cut():
    n = 400
    divs = np.zeros(n)
    divs[[60, 120, 180, 240, 300, 360]] = 0.8
    df = frame(walk(1, n), dividends=divs)
    full = expr.Namespace(df, ticker="X")["tr"]
    for D in (50, 61, 130, 250, 399):
        cut = df.iloc[: D + 1].copy()
        cut["adj_close"] = yahoo_adjusted(cut["close"], cut["dividend"])     # published with data up to D only
        tr = expr.Namespace(cut, ticker="X")["tr"]
        np.testing.assert_allclose(tr.to_numpy(), full.iloc[: D + 1].to_numpy(), rtol=1e-12)
        # the stored back-adjusted level itself is not causal: this is what the old `tr` read
        if D < 300:
            assert not np.allclose(cut["adj_close"].to_numpy(), df["adj_close"].iloc[: D + 1].to_numpy(), rtol=1e-6)
    assert full.iloc[0] == pytest.approx(df["close"].iloc[0])            # starts at the first quoted close
    # tret (a ratio) is numerically what adj_close's ratio was
    ns = expr.Namespace(df, ticker="X")
    t5 = expr.evaluate_value("tret(5)", ns)
    np.testing.assert_allclose(t5.to_numpy()[5:], (df["adj_close"] / df["adj_close"].shift(5) - 1).to_numpy()[5:],
                               rtol=1e-12, atol=1e-15)
    # sym().tr and the adjusted basis read the same causal index
    assert np.allclose(expr.Bars(df, df.index).tr.to_numpy(), full.to_numpy())
    adj = expr.Namespace(df, ticker="X", price_basis="adjusted")
    np.testing.assert_allclose(adj["tr"].to_numpy(), full.to_numpy(), rtol=1e-12)
    # (adjusted_frame reinvests at (close + dividend) / previous close, Yahoo at close / (previous close - dividend))
    np.testing.assert_allclose(adj["close"].to_numpy(), full.to_numpy(), rtol=5e-3)


@needs_data
def test_tr_of_real_data_is_causal_and_the_repro_trades_from_the_start():
    raw = data.load("MSFT")[["open", "high", "low", "close", "volume", "dividend", "adj_close", "open_ok"]]
    full = expr.Namespace(raw, ticker="MSFT")["tr"]
    assert full.iloc[0] == pytest.approx(raw["close"].iloc[0])
    for D in ("2003-01-15", "2012-06-29", "2020-01-02"):
        cut = raw.loc[:D].copy()
        cut["adj_close"] = yahoo_adjusted(cut["close"], cut["dividend"])
        tr = expr.Namespace(cut, ticker="MSFT")["tr"]
        # Yahoo's own rounding of its factors is the only difference left
        np.testing.assert_allclose(tr.to_numpy(), full.loc[:D].to_numpy(), rtol=2e-4)
    r = engine.run(Strategy(universe=["MSFT"], entry="tr / close > 0.99", hold_bars=1, start="2020-01-01"))
    assert str(pd.Timestamp(r.trades["entry_date"].min()).date()) == "2020-01-02"


# ------------------------------------------------------------ 2. one margin model

def test_leveraged_etf_requirements_in_both_engines():
    assert margin.maintenance("TQQQ", 0.25) == 0.75 and margin.maintenance("SSO", 0.25) == 0.5
    assert margin.maintenance("SPY", 0.25) == 0.25 and margin.initial("SPY", 0.25) == 0.5
    with pytest.raises(ValueError, match="3x leveraged ETF"):
        pf.Portfolio(tree={"asset": "TQQQ"}, leverage=3).validate()
    with pytest.raises(ValueError, match="3x leveraged ETF"):
        pf.Portfolio(tree={"asset": "TQQQ"}, leverage=3, margin_account="portfolio", maintenance_margin=0.15).validate()
    with pytest.raises((ValueError, parser.ParseError), match="leveraged ETF"):
        parser.parse("hold TQQQ with 3x leverage, rebalance monthly, from February 2020 to June 2020").validate()
    pf.Portfolio(tree={"asset": "TQQQ"}).validate()                       # no borrowing: fine
    p = pf.Portfolio(tree={"asset": "TQQQ"}, leverage=1.2)
    p.validate()
    assert any(n.startswith("Leveraged ETFs: TQQQ (3x)") for n in p.notes)
    with pytest.raises(ValueError, match="3x leveraged ETF"):
        Strategy(universe=["TQQQ"], entry="True", hold_bars=1, leverage=2).validate()
    Strategy(universe=["TQQQ"], entry="True", hold_bars=1, leverage=1.2).validate()
    # the same account rules: Reg T 2x, portfolio margin 4x with maintenance below 1/leverage
    with pytest.raises(ValueError, match="Regulation T"):
        pf.Portfolio(tree={"asset": "SPY"}, leverage=2.5).validate()
    with pytest.raises(ValueError, match="maintenance_margin"):
        pf.Portfolio(tree={"asset": "SPY"}, leverage=4, margin_account="portfolio").validate()
    pf.Portfolio(tree={"asset": "SPY"}, leverage=4, margin_account="portfolio", maintenance_margin=0.2).validate()


def test_margin_calls_de_risk_with_a_cushion_in_both_engines(fake):
    px = [100.0] * 30 + [90.0] * 30          # a 10% drop at 4x
    fake["X"] = frame(px)
    d = fake["X"].index[30]
    want = 1 / (margin.MARGIN_CALL_CUSHION * 0.24)          # 3.33x, not straight back to 4x
    p = pf.Portfolio(tree={"asset": "X"}, rebalance="none", leverage=4, cash_rate=None, margin_account="portfolio",
                     maintenance_margin=0.24)
    r = pf.run(p)
    assert (r.orders.reason == "margin call").sum() == 1
    assert r.exposure[d] == pytest.approx(want, rel=1e-6)
    s = Strategy(universe=["X"], entry="True", hold_bars=200, leverage=4, position_size=4.0, cash_rate=None,
                 margin_account="portfolio", maintenance_margin=0.24)
    rs = engine.run(s)
    assert (rs.trades["exit_reason"] == "margin call").any()
    assert rs.exposure[d] == pytest.approx(want, rel=1e-6)
    assert any("125% of the requirement" in n for n in s.notes)


# ------------------------------------------------------------ 3. borrow fee phrases

@pytest.mark.parametrize("tail", ["borrow fee 2%", "borrow fee of 2% per year", "2% annual borrow fee", "borrow cost of 2%",
                                  "a borrow cost 2% a year", "2% borrow fee", "borrow fee: 2% annually",
                                  "with a 2% borrowing cost on shorts"])
def test_borrow_fee_word_orders(tail):
    s = parser.parse(f"short QQQ at the open when it gaps up 1%, cover at the close, {tail}")
    assert s.borrow_fee == pytest.approx(0.02)
    p = parser.parse(f"hold 100% SPY and -30% QQQ, rebalance monthly, {tail}")
    assert p.borrow_fee == pytest.approx(0.02)


# ------------------------------------------------------------ 4. share classes

def test_share_classes_take_one_slot(fake):
    for t, seed in (("GOOG", 1), ("GOOGL", 1), ("A", 2), ("B", 3)):
        fake[t] = frame(walk(seed, 300))
    fake["GOOGL"]["volume"] = 2e6                     # the more liquid class
    fake["GOOG"]["close"] *= 1.0001                   # (nearly) the same stock
    tree = {"filter": {"select": "top", "n": 2, "by": "ret(20)"}, "children": [{"asset": t} for t in ("GOOG", "GOOGL", "A", "B")]}
    p = pf.Portfolio(tree=tree, rebalance="monthly", cash_rate=None)
    r = pf.run(p)
    h = r.holdings
    both = (h.get("GOOG", 0) > 1e-9) & (h.get("GOOGL", 0) > 1e-9)
    assert not both.any()
    assert "GOOG" not in h or (h["GOOG"] < 1e-9).all()        # held through the more liquid class
    assert any(n.startswith("Share classes: GOOG/GOOGL") for n in p.notes)
    sep = pf.Portfolio(tree=tree, rebalance="monthly", cash_rate=None, share_classes="separate")
    hs = pf.run(sep).holdings
    assert ((hs.get("GOOG", 0) > 1e-9) & (hs.get("GOOGL", 0) > 1e-9)).any()
    # signal engine: no second position in the other class while one is held
    s = Strategy(universe=["GOOG", "GOOGL"], entry="True", hold_bars=5, max_positions=2, cash_rate=None)
    rs = engine.run(s)
    assert set(rs.trades["ticker"]) == {"GOOGL"} and rs.positions.max() == 1
    assert any(n.startswith("Share classes: GOOG/GOOGL") for n in s.notes)
    s2 = Strategy(universe=["GOOG", "GOOGL"], entry="True", hold_bars=5, max_positions=2, cash_rate=None,
                  share_classes="separate")
    assert engine.run(s2).positions.max() == 2


# ------------------------------------------------------------ 5. survivorship coverage from the first real session

@needs_data
def test_survivorship_lowest_year_ignores_the_day0_bar():
    p = parser.parse("hold the top 2 Nasdaq 100 stocks by 1 month return, rebalance monthly, from 2006 to 2006-03-31")
    pf.run(p)
    note = next(n for n in p.notes if n.startswith("Survivorship:"))
    assert "in 2005" not in note and "in 2006" in note


# ------------------------------------------------------------ 6. synthetic opens

def _raw(o, h, lo, c):
    idx = pd.bdate_range("1980-12-12", periods=len(c))
    return pd.DataFrame({"open": o, "high": h, "low": lo, "close": c}, index=idx)


def test_synthetic_open_stretches_are_flagged():
    rng = np.random.default_rng(3)
    c = 10 * np.cumprod(1 + rng.normal(0, 0.03, 200))
    tick = c * 0.004
    same = rng.random(200) < 0.6
    o = np.where(same, c, c + tick)                          # the "open" is the close, or one tick from it
    f = data.synthetic_open_days(_raw(o, np.maximum(o, c), np.minimum(o, c), c))
    assert f.iloc[5:].mean() > 0.9
    # a real stock: opens away from the close, ranges wide enough for the day's move
    o2 = c * (1 + rng.normal(0, 0.01, 200))
    f2 = data.synthetic_open_days(_raw(o2, np.maximum(o2, c) * 1.02, np.minimum(o2, c) * 0.98, c))
    assert not f2.any()
    # a T-bill ETF: opens at the close most days, but the price barely moves
    cb = 91.5 + np.cumsum(np.where(rng.random(200) < 0.3, 0.01, 0.0))
    ob = np.where(rng.random(200) < 0.7, cb, cb - 0.01)
    assert not data.synthetic_open_days(_raw(ob, np.maximum(ob, cb), np.minimum(ob, cb), cb)).any()
    # causal: a truncated history gives the same flags up to the cut
    raw = _raw(o, np.maximum(o, c), np.minimum(o, c), c)
    np.testing.assert_array_equal(data.synthetic_open_days(raw.iloc[:120]).to_numpy(), f.iloc[:120].to_numpy())


@needs_data
def test_early_aapl_opens_are_not_quoted():
    ok = data.load("AAPL")["open_ok"]
    assert not ok.loc["1981-01-01":"1981-12-31"].any()
    assert ok.loc["2000-01-01":"2000-12-31"].mean() > 0.95
    with pytest.raises(ValueError, match="filled in from the close"):
        engine.run(Strategy(universe=["AAPL"], entry="open > ref(close, 1)", hold_bars=0, entry_fill="open",
                            hold_exit_fill="close", start="1980-12-16", end="1981-12-31"))


# ------------------------------------------------------------ 7. no interest during the signal warm-up

def test_signal_warmup_earns_nothing_and_stats_start_at_the_capital(fake):
    fake["X"] = frame(walk(5, 300))
    s = Strategy(universe=["X"], entry="close > sma(close, 50)", hold_bars=2, cash_rate=0.05)
    r = engine.run(s)
    warm = fake["X"].index[49]
    assert (r.equity.loc[:warm] == 10_000).all()
    wd, notes = report.warmup(r)           # where the statistics start: the starting capital there
    assert wd == fake["X"].index[49] and r.equity[wd] == 10_000
    assert any("No interest is earned during the warm-up" in n for n in notes)
    # accounting identity: capital + trade P&L + interest
    assert r.interest > 0 and r.equity.iloc[-1] == pytest.approx(10_000 + r.trades.pnl.sum() + r.interest)
    # truncation inside and after the warm-up changes nothing before the cut
    for k in (30, 120):
        fake["X"] = frame(walk(5, 300))[: k + 1]
        rk = engine.run(Strategy(universe=["X"], entry="close > sma(close, 50)", hold_bars=2, cash_rate=0.05))
        common = rk.equity.index[:-1]
        np.testing.assert_allclose(rk.equity.loc[common].to_numpy(), r.equity.loc[common].to_numpy(), rtol=1e-12)


# ------------------------------------------------------------ 8. broker catch-up

def test_catch_up_is_skipped_unless_asked():
    rows = [{"ticker": "SPY", "side": "BUY", "shares": 12, "current_shares": 0, "target_shares": 12,
             "catch_up": "2026-09-16"}]
    notes: list = []
    o = broker.build_orders(rows, kind="signal", as_of="2026-09-25", tag="t", fill="next_open", entry_order="limit",
                            notes=notes, fractional=False)
    assert o == [] and any("CATCH-UP" in n and "no order sent" in n for n in notes)
    notes = []
    o = broker.build_orders(rows, kind="signal", as_of="2026-09-25", tag="t", fill="next_open", entry_order="limit",
                            notes=notes, fractional=False, catch_up="market")
    assert len(o) == 1 and (o[0]["type"], o[0]["time_in_force"]) == ("market", "day")
    assert o[0]["client_order_id"].endswith("catchup") and any("DAY MARKET" in n for n in notes)
    # a new tested entry due today keeps its order type and level
    lv = {"SPY": {"price": 700.0, "valid_sessions": 1}}
    o = broker.build_orders([{**rows[0], "catch_up": None}], kind="signal", as_of="d", tag="t", fill="next_open",
                            entry_order="limit", entry_levels=lv, fractional=False)
    assert (o[0]["type"], o[0]["limit_price"]) == ("limit", "700.00")


@needs_data
def test_plan_marks_backtest_positions_as_catch_up():
    s = parser.parse("buy SPY with a limit order 1% below the close when `change > -1`, hold 40 days")
    pl = broker.plan(s, 10_000, {}, fractional=False)
    held = [r for r in pl.rows if r["ticker"] == "SPY" and r.get("catch_up")]
    if not held:
        pytest.skip("the backtest holds no SPY position from an earlier entry at the end of the data")
    assert not [o for o in pl.orders if o["symbol"] == "SPY" and o["type"] == "market"]
    pl2 = broker.plan(s, 10_000, {}, fractional=False, catch_up="market")
    assert [o for o in pl2.orders if o["symbol"] == "SPY" and o["type"] == "market" and o["time_in_force"] == "day"]


# ------------------------------------------------------------ 9. Python-function rules: faster, same answers

def _cheap(df, ns):
    return (df.close < df.close.rolling(10).min().shift(1)) & (ns["rsi"](2) < 30)


def test_prefix_namespace_answers_equal_a_fresh_namespace_on_the_prefix(fake):
    df = frame(walk(7, 400))
    ns = expr.Namespace(df, ticker="X")
    for k in (5, 60, 250, 400):
        sub = expr._sub_namespace(ns, k)
        ref = expr.Namespace(df.iloc[:k].copy(), ticker="X")
        for name, args in (("rsi", (2,)), ("sma", ("close", 20)), ("atr", (14,)), ("tret", (5,)), ("ema", ("high", 9))):
            a = [sub[x] if isinstance(x, str) else x for x in args]
            b = [ref[x] if isinstance(x, str) else x for x in args]
            pd.testing.assert_series_equal(sub[name](*a), ref[name](*b), check_names=False)
        assert sub.df.index[-1] == df.index[k - 1] and len(sub["close"]) == k


def test_stream_from_parallel_and_stateful_fallback(fake, monkeypatch):
    df = frame(walk(8, 1500))
    ns = expr.Namespace(df, ticker="X")
    monkeypatch.setattr(expr, "STREAM_WORKERS", 1)
    seq = expr.evaluate(_cheap, ns)
    expr._STREAMED.clear()
    monkeypatch.setattr(expr, "STREAM_WORKERS", 3)
    monkeypatch.setattr(expr, "STREAM_PARALLEL_MIN", 300)
    par = expr.evaluate(_cheap, expr.Namespace(df, ticker="X"))
    pd.testing.assert_series_equal(seq, par)
    ref = pd.Series([bool(expr._last(_cheap(df.iloc[:i + 1], expr.Namespace(df.iloc[:i + 1], ticker="X")), "bool"))
                     for i in range(0, 1500, 37)], index=df.index[::37])
    assert (par.iloc[::37] == ref).all()
    # only from the start asked for: earlier bars are not evaluated
    expr._STREAMED.clear()
    with expr.stream_from(df.index[1000]):
        part = expr.evaluate(_cheap, expr.Namespace(df, ticker="X"))
    assert not part.iloc[:1000].any() and (part.iloc[1000:] == seq.iloc[1000:]).all()
    # a function that keeps state between calls answers differently in chunks: streamed in one pass instead
    calls = {"n": 0}

    def counter(d, ns):
        calls["n"] += 1
        return pd.Series(calls["n"] % 3 == 0, index=d.index)
    expr._STREAMED.clear()
    out = expr.evaluate(counter, expr.Namespace(df, ticker="X"))
    calls["n"] = 0
    monkeypatch.setattr(expr, "STREAM_WORKERS", 1)
    expr._STREAMED.clear()
    want = expr.evaluate(counter, expr.Namespace(df, ticker="X"))
    pd.testing.assert_series_equal(out, want)


def test_run_streams_only_the_backtest_window(fake):
    fake["X"] = frame(walk(9, 1200))
    from backtester import sandbox

    def spy_rule(d, ns):          # runs in a sealed process: its calls are counted through sandbox.STATS
        return d.close > d.close.rolling(5).mean()
    s0 = sandbox.STATS["steps"]
    engine.run(Strategy(universe=["X"], entry=spy_rule, hold_bars=1, cash_rate=None,
                        start=str(fake["X"].index[1000].date())))
    assert sandbox.STATS["steps"] - s0 == 200        # the 200 bars from the start on, streamed once
