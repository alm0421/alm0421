"""Round 13 (QuantConnect reviewer).

1. An open_safe Python rule acted on at the open could read today's close of ANOTHER ticker (ns['sym']('SPY').close):
   the sampled probe let it through depending on the start date (1182 trades, 162% CAGR on QQQ). Now a rule acted on
   at the open is evaluated as known at each open - the day's bar holds only its open, for its own ticker and for
   every ticker / data series it loads (sandbox OPEN_KNOWN) - and it is refused when that changes any answer. A
   vectorized_causal function is checked on every day (streamed), so a leak on one date a year cannot pass; only an
   explicit trust_vectorized opts out. ha_open, xrank(gap) and sym("X").gap are open-safe.
2. "S&P 500 stocks" / "Russell 2000 stocks" / "Dow 30 stocks" were silently read as SPY / IWM; now the parser asks
   (the fund, or today's members with a survivorship warning).
3. The few-trades warning counts closed trades and says how many are still open; weights above 100% (in an
   if-branch too: '200% QQQ', '2x QQQ') are borrowed; a whole index can be held at once ('all Nasdaq 100 stocks
   equally weighted').
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from backtester import data, engine, expr, metrics, parser, sandbox
from backtester import portfolio as pf
from backtester.strategy import Strategy

HAVE = {"QQQ", "SPY"} <= set(data.available_tickers())
needs_data = pytest.mark.skipif(not HAVE, reason="price data not downloaded")


def walk(seed, n=500, start="2012-01-02"):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    c = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n)))
    o = c * np.exp(rng.normal(0, 0.006, n))
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.004, "low": np.minimum(o, c) * 0.996, "close": c,
                         "adj_close": c, "volume": 1e6, "dividend": 0.0, "split": 1.0, "open_ok": True,
                         "quote_close": c}, index=idx)


@pytest.fixture
def fake(monkeypatch):
    frames = {}
    monkeypatch.setattr(data, "load", lambda t: frames[data.canonical(t)])
    monkeypatch.setattr(data, "load_many", lambda ts: {data.canonical(t): frames[data.canonical(t)] for t in ts})
    expr._STREAMED.clear()
    expr._CALLABLE_PROBES.clear()
    return frames


# ------------------------------------------------------------ 1. structural open-time evaluation

def seen_at_open(d, ns):
    """1.0 where the function saw exactly what is known at the day's open: its own bar and Y's bar (loaded directly
    and through sym) with the open only, and the complete bars before."""
    y = data.load("Y")
    s = ns["sym"]("Y")
    ok = (np.isfinite(d.open.iloc[-1]) and all(np.isnan(d[c].iloc[-1]) for c in ("close", "high", "low", "volume",
                                                                                   "adj_close", "dividend"))
          and len(d) < 2 or np.isfinite(d.close.iloc[-2]))
    ok = ok and y.index[-1] == d.index[-1] and np.isfinite(y.open.iloc[-1]) and np.isnan(y.close.iloc[-1]) \
        and np.isnan(y.high.iloc[-1]) and (len(y) < 2 or np.isfinite(y.close.iloc[-2]))
    ok = ok and s.close.iloc[-1] == y.close.iloc[-2] if len(y) > 1 else ok      # (sym fills the day with the last close)
    return pd.Series(1.0 if ok else 0.0, index=d.index)


def test_open_view_holds_only_the_open_of_the_day_for_every_ticker(fake):
    fake["X"], fake["Y"] = walk(1), walk(2)
    ns = expr.Namespace(fake["X"], ticker="X")
    frm = fake["X"].index[400]
    with expr.stream_from(frm), expr.open_view():
        v = expr.evaluate_value(seen_at_open, ns)
    assert (v[frm:] == 1.0).all()
    with expr.stream_from(frm):       # at the close the day is complete: the function sees it
        assert (expr.evaluate_value(seen_at_open, expr.Namespace(fake["X"], ticker="X"))[frm:] == 0.0).all()
    assert any("acted on at the open" in n for n in ns.notes)


def test_open_cut_of_data_requests():
    idx = pd.bdate_range("2020-01-01", periods=5)
    fr = pd.DataFrame({"open": [1.0, 2, 3, 4, 5], "close": [1.5, 2.5, 3.5, 4.5, 5.5], "volume": [10, 20, 30, 40, 50],
                       "open_ok": True}, index=idx)
    got = sandbox._open_cut(fr, idx[2])
    assert list(got.index) == list(idx[:3]) and got.open.iloc[-1] == 3 and np.isnan(got.close.iloc[-1])
    assert np.isnan(got.volume.iloc[-1]) and got.close.iloc[-2] == 2.5 and got.open_ok.iloc[-1]
    ser = fr.close
    assert list(sandbox._open_cut(ser, idx[2]).index) == list(idx[:2])            # a close series: the day before
    no_open = fr[["close"]]
    assert list(sandbox._open_cut(no_open, idx[2]).index) == list(idx[:2])
    # a data function other than a price frame is answered as of the day before
    msg = ("rpc", "tbill_rate", (), {})
    s = sandbox._answer_rpc(msg, pd.Timestamp("2024-03-15"), open_view=True)[1]
    if isinstance(s, pd.Series) and len(s):
        assert s.index[-1] < pd.Timestamp("2024-03-15")


def _peek_y(d, ns):
    s = ns["sym"]("Y")
    return s.close > s.open           # Y's close today, "known" at X's open


_peek_y.open_safe = True


def test_other_tickers_close_at_the_open_is_refused_and_cannot_be_used(fake, monkeypatch):
    fake["X"], fake["Y"] = walk(3), walk(4)
    base = dict(cash_rate=None, universe=["X"], entry=_peek_y, entry_fill="open", hold_bars=0, hold_exit_fill="close")
    with pytest.raises(ValueError, match="Lookahead.*open_safe.*later in the day"):
        engine.run(Strategy(**base))
    # even with the check switched off, the run cannot use Y's close: at the open its bar has only the open, so the
    # function answers "yesterday's close > today's open" - Y gapped down - on every day
    monkeypatch.setattr(expr, "callable_open_check", lambda *a, **k: None)
    expr._STREAMED.clear()
    r = engine.run(Strategy(**base))
    ref = engine.run(Strategy(**{**base, "entry": 'sym("Y").gap < 0'}))
    assert len(r.trades) and list(r.trades.entry_date) == list(ref.trades.entry_date)


def test_open_safe_function_known_at_the_open_matches_the_text_rule(fake):
    fake["X"], fake["Y"] = walk(5), walk(6)

    def y_gap(d, ns):
        s = ns["sym"]("Y")
        return s.open / s.close.shift(1) - 1 < -0.005
    y_gap.open_safe = True
    base = dict(cash_rate=None, universe=["X"], entry_fill="open", hold_bars=0, hold_exit_fill="close")
    r = engine.run(Strategy(**base, entry=y_gap))
    ref = engine.run(Strategy(**base, entry='sym("Y").gap < -0.005'))
    assert len(r.trades) > 5
    assert list(r.trades.entry_date) == list(ref.trades.entry_date)
    assert np.allclose(r.equity.to_numpy(), ref.equity.to_numpy())


def test_open_safe_exit_function_at_the_open(fake):
    fake["X"] = walk(7)

    def peek_exit(d, ns):             # claims open-safety but reads today's close
        return d.close < d.open
    peek_exit.open_safe = True

    def gap_exit(d, ns):
        return d.open > d.close.shift(1)
    gap_exit.open_safe = True
    base = dict(cash_rate=None, universe=["X"], entry="gap < 0", entry_fill="open", exit_when_fill="open")
    with pytest.raises(ValueError, match="Lookahead: the exit rule is filled at the open"):
        engine.run(Strategy(**base, exit_when=peek_exit))
    r = engine.run(Strategy(**base, exit_when=gap_exit))
    ref = engine.run(Strategy(**base, exit_when="gap > 0"))
    assert len(r.trades) > 5 and list(r.trades.exit_date) == list(ref.trades.exit_date)


def _march13(d, ns):
    """Causal except on March 13 each year, where it knows whether a next bar exists (a one-day-a-year leak)."""
    leak = pd.Series(d.close.shift(-1).notna().to_numpy(), index=d.index)
    on = (d.index.month == 3) & (d.index.day == 13)
    return pd.Series(np.where(on, leak, d.close > d.close.shift(1)), index=d.index)


def test_one_day_a_year_leak_in_a_vectorized_function_is_refused(fake):
    fake["X"] = walk(8, n=900)
    fn = _march13
    fn.vectorized_causal = True
    with pytest.raises(ValueError, match="uses future data: its result on 20..-03-13"):
        engine.run(Strategy(cash_rate=None, universe=["X"], entry=fn, hold_bars=1))


def test_vectorized_causal_is_streamed_and_trust_vectorized_runs_once(fake):
    fake["X"] = walk(9)

    def fast(d, ns):
        return d.close > d.close.shift(1)
    fast.vectorized_causal = True
    s0 = sandbox.STATS["steps"]
    s = expr.evaluate(fast, expr.Namespace(fake["X"], ticker="X"))
    assert sandbox.STATS["steps"] - s0 >= len(fake["X"]) - 1 and s.sum() > 0

    def trusted(d, ns):
        return d.close > d.close.shift(1)
    trusted.trust_vectorized = True
    ns = expr.Namespace(fake["X"], ticker="X")
    j0, s0 = sandbox.STATS["jobs"], sandbox.STATS["steps"]
    t = expr.evaluate(trusted, ns)
    assert sandbox.STATS["jobs"] == j0 + 1 and sandbox.STATS["steps"] == s0 and (t == s).all()
    assert any("trust_vectorized" in n and "Warning" in n for n in ns.notes)
    # at the open it is streamed anyway (a whole-history call cannot hide each day's close)
    trusted.open_safe = True
    with expr.open_view():
        expr.evaluate(trusted, expr.Namespace(fake["X"], ticker="X"))
    assert sandbox.STATS["steps"] > s0


@needs_data
def test_reviewer_repro_spy_close_at_qqq_open_is_refused():
    from backtester import api
    t = time.time()
    with pytest.raises(ValueError, match="Lookahead"):
        api.backtest(Strategy(universe=["QQQ"], start="2024-01-01", entry=_peek_y_spy, entry_fill="open",
                              hold_bars=0, hold_exit_fill="close"))
    assert time.time() - t < 120


def _peek_y_spy(df, ns):
    s = ns["sym"]("SPY")
    return s.close > s.open


_peek_y_spy.open_safe = True


# ------------------------------------------------------------ 1b. rules the static check now allows at the open

def test_open_safe_static_additions():
    for rule in ("ha_open > ref(close, 1)", 'sym("QQQ").gap < -0.01', "xrank(gap) < 0.2",
                 'sym("QQQ").gap < 0 and ref(ha_close, 1) > ha_open'):
        assert expr.open_safe(rule), rule
    for rule in ("ha_close > ha_open", 'sym("QQQ").close > sym("QQQ").open', "xrank(ret(1)) < 0.2"):
        assert not expr.open_safe(rule), rule
    assert expr.reads_todays_open('sym("QQQ").gap < 0')


def test_ha_open_never_reads_its_own_bar():
    df = walk(10, n=120)
    df.iloc[50, df.columns.get_loc("close")] = np.nan          # a gap in the data: the recursion restarts
    base = expr.heikin_ashi(df)["ha_open"]
    assert np.isnan(base.iloc[0]) and np.isnan(base.iloc[51])
    for i in (5, 30, 52, 100):
        d2 = df.copy()
        for c in ("close", "high", "low"):
            d2.iloc[i, d2.columns.get_loc(c)] *= 1.07
        got = expr.heikin_ashi(d2)["ha_open"]
        assert np.allclose(got.iloc[: i + 1], base.iloc[: i + 1], equal_nan=True), i


def test_sym_gap_and_xrank_warm_up(fake):
    fake["X"], fake["Y"] = walk(11), walk(12)
    ns = expr.Namespace(fake["X"], ticker="X")
    y = fake["Y"]
    g = expr.evaluate_value('sym("Y").gap', ns)
    assert np.allclose(g.iloc[1:], (y.open / y.close.shift(1) - 1).iloc[1:])
    assert expr.never_defined("xrank(gap) < 0.2", ns) == []


def test_xrank_gap_at_the_open_runs(fake):
    for i, t in enumerate("ABCD"):
        fake[t] = walk(20 + i)
    r = engine.run(Strategy(cash_rate=None, universe=list("ABCD"), entry="xrank(gap) <= 0.25", entry_fill="open",
                            hold_bars=0, hold_exit_fill="close", max_positions=1))
    assert len(r.trades) > 50


# ------------------------------------------------------------ 2. index universes are never silently the ETF

@pytest.mark.parametrize("text, etf", [
    ("buy S&P 500 stocks when RSI(2) is below 5, hold 3 days", "SPY"),
    ("buy all stocks in the S&P 500 when RSI(2) is below 5, hold 3 days", "SPY"),
    ("buy Russell 2000 stocks when RSI(2) is below 5, hold 3 days", "IWM"),
    ("buy Dow 30 stocks when RSI(2) is below 5, hold 3 days", "DIA"),
    ("buy S&P 600 companies when RSI(2) is below 5, hold 3 days", "IJR"),
    ("hold S&P 500 stocks", "SPY"),
    ("hold the top 10 S&P 500 stocks by 12 month return, rebalance monthly", "SPY"),
])
def test_index_stocks_ask_instead_of_the_etf(text, etf):
    with pytest.raises(parser.ParseError) as e:
        parser.parse(text)
    msg = str(e.value)
    assert f"say '{etf}'" in msg and "point-in-time" in msg


def test_the_index_itself_is_still_its_fund():
    s = parser.parse("buy the S&P 500 when RSI(2) is below 5, hold 3 days")
    assert s.universe == ["SPY"]


def _sp_members():
    from backtester import index_universes
    return index_universes.members(("sp500",))


@pytest.mark.skipif(not _sp_members(), reason="no S&P 500 constituent data")
def test_todays_members_universe_with_a_survivorship_warning():
    s = parser.parse("buy today's S&P 500 members when RSI(2) is below 5, hold 3 days")
    assert set(s.universe) == set(_sp_members()) and len(s.universe) > 50
    assert s.universe_name == "S&P 500" and not s.point_in_time
    assert any(n.startswith("Warning: survivorship bias") and "TODAY'S S&P 500" in n for n in s.notes)
    assert "today's members (survivorship-biased" in s.summary()
    p = parser.parse("hold the top 10 of today's S&P 500 members by 12 month return, rebalance monthly")
    assert set(p.tree["universe"]) == set(_sp_members())
    assert any(n.startswith("Warning: survivorship bias") for n in p.notes)


# ------------------------------------------------------------ 3. minor

def test_few_trades_warning_counts_closed_and_open():
    w = metrics.result_warnings("signal", {}, {"trades": 2, "open_trades": 1})
    d = next(x for x in w if x["code"] == "few_trades")["detail"]
    assert d.startswith("Only 2 closed trades (and 1 still open at the end")


def test_weights_above_100_percent_are_borrowed_in_branches():
    for text in ("if SPY is above its 200 day moving average hold 200% QQQ, otherwise hold TLT",
                 "if SPY is above its 200 day moving average hold 2x QQQ, otherwise hold TLT"):
        p = parser.parse(text)
        then = p.tree["then"]
        assert then == {"weights": "specified", "w": [2.0, -1.0], "children": [{"asset": "QQQ"}, {"cash": True}]}
        assert any("borrowed" in n for n in p.notes)
        assert "short" not in p.summary().split("Costs:")[1].split("\n")[0]
        assert "maintenance margin" in p.summary()
    p = parser.parse("hold 150% SPY and 50% TLT, rebalance monthly")
    assert p.tree["w"] == [1.5, 0.5, -1.0]
    with pytest.raises(parser.ParseError, match="not 100%"):
        parser.parse("hold 300% SPY and 200% TLT")          # beyond 4x: refused (a typo more likely than a plan)
    # 3x of a named index is left alone (it may mean a leveraged fund)
    assert parser._times_leverage("3x S&P 500", []) == "3x S&P 500"


def test_borrowed_branch_simulates_like_portfolio_leverage(fake):
    fake["AAA"] = walk(30)
    tree = {"weights": "specified", "w": [2.0, -1.0], "children": [{"asset": "AAA"}, {"cash": True}]}
    a = pf.run(pf.Portfolio(tree=tree, rebalance="monthly", cash_rate=0.0, benchmark=None))
    b = pf.run(pf.Portfolio(tree={"asset": "AAA"}, leverage=2.0, rebalance="monthly", cash_rate=0.0, benchmark=None))
    assert np.allclose(a.equity.to_numpy(), b.equity.to_numpy(), rtol=1e-9)


@needs_data
def test_whole_index_equal_weight_phrases():
    p = parser.parse("hold all Nasdaq 100 stocks equally weighted, rebalance monthly")
    assert p.tree["filter"]["select"] == "all" and p.tree["universe"] == "NDX"
    assert p.tree["filter"]["weights"] == "equal"
    p2 = parser.parse("equal weight all Nasdaq 100 stocks, rebalance monthly")
    assert p2.tree == p.tree
    assert "all of [Nasdaq-100 members]" in p.summary()


def test_whole_index_equal_weight(fake):
    # the simulation holds every member trading that day, equally
    fake["A"], fake["B"], fake["C"] = walk(31), walk(32), walk(33)
    tree = {"filter": {"select": "all", "weights": "equal"}, "universe": ["A", "B", "C"]}
    p = pf.Portfolio(tree=tree, rebalance="monthly", cash_rate=0.0, benchmark=None)
    r = pf.run(p)
    h = r.holdings
    month_ends = h.groupby(h.index.to_period("M")).tail(1)
    assert np.allclose(month_ends[list("ABC")].to_numpy(), 1 / 3, atol=1e-9)
    ref = pf.run(pf.Portfolio(tree={"weights": "equal", "children": [{"asset": t} for t in "ABC"]},
                              rebalance="monthly", cash_rate=0.0, benchmark=None))
    assert np.allclose(r.equity.to_numpy(), ref.equity.to_numpy(), rtol=1e-12)
    assert "All of A, B, C" in pf.short_name(tree)
    assert not any("never leaves anything out" in n or "thin" in n.lower() for n in p.notes)
