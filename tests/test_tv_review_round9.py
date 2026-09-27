"""Round 9 (a TradingView expert's review): scale-outs are resting limit orders filled in price-path order (and at
the open on a gap), charts draw each series with the offset the rule reads it at, TradingView-compatible sizing
(quantity from the signal bar's close, whole shares) and holding periods, new signal phrases, date validation."""
import numpy as np
import pandas as pd
import pytest

from backtester import data, engine, expr, parser, report
from backtester.parser import ParseError
from backtester.strategy import Strategy

AVAIL = set(data.available_tickers())
needs_data = pytest.mark.skipif(not {"SPY", "NVDA"} <= AVAIL, reason="price data not downloaded")


def bars(rows, start="2019-12-30", volume=1e6):
    idx = pd.bdate_range(start, periods=len(rows))
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx, dtype=float)
    df["volume"] = volume
    df["quote_close"] = df["close"]
    df["adj_close"] = df["close"]
    df["dividend"] = 0.0
    return df


def rand_bars(n=500, seed=0):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, n)))
    o = np.r_[100.0, c[:-1]] * np.exp(rng.normal(0, 0.006, n))
    hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.006, n)))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.006, n)))
    df = bars(np.column_stack([o, hi, lo, c]))
    df["volume"] = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return df


@pytest.fixture
def fake(monkeypatch):
    frames = {}
    real = data.load

    def load(t):
        if t in frames:
            return frames[t]
        if t in ("SPY", "QQQ", "SPYSIM"):
            raise FileNotFoundError(t)
        return real(t)
    monkeypatch.setattr(data, "load", load)
    monkeypatch.setattr(data, "load_many", lambda ts: {t: frames[t] for t in ts})
    return frames


def books(r):
    s = r.strategy
    assert r.equity.iloc[-1] == pytest.approx(s.capital + r.trades.pnl.sum() + r.interest, abs=1e-6)


# ------------------------------------------------------------ 1-2. scale-outs in path order, gaps at the open

def one_bar_case(bar1, side):
    """Enter at bar 0's close (100); bar 1 is the bar under test; then flat bars until a 5-bar time exit.
    A short mirrors the prices around 100 (high <-> low)."""
    rows = [[100, 100, 100, 100], bar1] + [[100, 100, 100, 100]] * 6
    if side == "short":
        rows = [[200 - o, 200 - lo, 200 - h, 200 - c] for o, h, lo, c in rows]
    df = bars(rows)
    df.loc[df.index[0], "volume"] = 2e6         # the entry signal: bar 0 only
    return df


def run_case(fake, bar1, side, tv, scale_at=0.03, target=0.06, stop=0.05):
    fake["SYN"] = one_bar_case(bar1, side)
    s = Strategy(universe=["SYN"], entry="volume > 1.5e6", side=side, entry_fill="close", hold_bars=5,
                 scale_out=[{"at": scale_at, "fraction": 0.5}], take_profit=target, stop_loss=stop,
                 tv_compat=tv, fractional_shares=True, cash_rate=None, dividends=False, start="2019-12-30")
    r = engine.run(s)
    books(r)
    t = r.trades
    total = t.shares.sum()
    m = 1 if side == "long" else -1
    # (reason, price as a long's price, fraction of the position, exit fill)
    return [(x.exit_reason, round(100 + m * (x.exit_price - 100), 6), round(x.shares / total, 6), x.exit_fill)
            for _, x in t.iterrows()]


SO, TP, SL = ("scale out", "take profit", "stop loss")
# bar 1 (as a long sees it: the scale-out at 103, the target at 106, the stop at 95) -> exits, in both modes
CASES = [
    # scale-out and target on one bar, no stop: the scale-out (nearer) fills first, then the target
    ("both profit levels", [101, 107, 100.5, 104], [(SO, 103, .5, "intraday"), (TP, 106, .5, "intraday")],
     [(SO, 103, .5, "intraday"), (TP, 106, .5, "intraday")]),
    # stop and scale-out touched, open nearer the high: default = stop first (all of it); TradingView's path goes
    # up first (scale-out) and then down to the stop for the rest
    ("stop and scale-out, high first", [101, 104, 94, 100], [(SL, 95, 1.0, "intraday")],
     [(SO, 103, .5, "intraday"), (SL, 95, .5, "intraday")]),
    # the same with the open nearer the low: the path reaches the stop first in both modes
    ("stop and scale-out, low first", [97, 104, 94, 100], [(SL, 95, 1.0, "intraday")], [(SL, 95, 1.0, "intraday")]),
    # stop, scale-out and target, high first: TradingView fills both profit levels before the low
    ("all three, high first", [101, 107, 94, 100], [(SL, 95, 1.0, "intraday")],
     [(SO, 103, .5, "intraday"), (TP, 106, .5, "intraday")]),
    # a gap above the scale-out: it fills at the open, the target later that bar
    ("gap over the scale-out", [104, 107, 103.5, 106], [(SO, 104, .5, "open"), (TP, 106, .5, "intraday")],
     [(SO, 104, .5, "open"), (TP, 106, .5, "intraday")]),
    # a gap over the scale-out only
    ("gap over the scale-out only", [104, 105, 103.5, 104], [(SO, 104, .5, "open"), ("time exit", 100, .5, "close")],
     [(SO, 104, .5, "open"), ("time exit", 100, .5, "close")]),
    # a gap over both: both at the open, the scale-out first
    ("gap over both", [107, 108, 106.5, 107], [(SO, 107, .5, "open"), (TP, 107, .5, "open")],
     [(SO, 107, .5, "open"), (TP, 107, .5, "open")]),
    # a gap under the stop: all at the open
    ("gap under the stop", [93, 104, 92, 100], [(SL, 93, 1.0, "open")], [(SL, 93, 1.0, "open")]),
]


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("name,bar1,default,tv", CASES, ids=[c[0] for c in CASES])
def test_scale_out_path_order(fake, name, bar1, default, tv, side):
    assert run_case(fake, bar1, side, tv=False) == default
    assert run_case(fake, bar1, side, tv=True) == tv


@pytest.mark.parametrize("tv", [False, True])
def test_scale_out_above_the_target_never_fills(fake, tv):
    got = run_case(fake, [101, 109, 100.5, 104], "long", tv, scale_at=0.08)
    assert got == [(TP, 106, 1.0, "intraday")]


def _independent_nvda_trades(df, entry_sig):
    """A plain re-implementation of 'buy at the close when the signal is true, sell half at +3%, take profit at 6%,
    stop loss 5%' on as-quoted (split-adjusted) bars: resting orders, gaps fill at the open, a stop touched on a bar
    goes first, profit levels nearest first. Returns [(entry date, exit date, reason, fraction, exit price)]."""
    o, h, lo, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    out, i, n = [], 0, len(df)
    while i < n:
        if not entry_sig[i]:
            i += 1
            continue
        e, left, scaled, j = c[i], 1.0, False, i + 1
        stop, so, tp = e * 0.95, e * 1.03, e * 1.06
        while j < n and left > 0:
            if o[j] <= stop:
                out.append((df.index[i], df.index[j], "stop loss", left, o[j]))
                left = 0
                break
            if not scaled and o[j] >= so:
                out.append((df.index[i], df.index[j], "scale out", left / 2, o[j]))
                left, scaled = left / 2, True
            if o[j] >= tp:
                out.append((df.index[i], df.index[j], "take profit", left, o[j]))
                left = 0
                break
            if lo[j] <= stop:
                out.append((df.index[i], df.index[j], "stop loss", left, stop))
                left = 0
                break
            if not scaled and h[j] >= so:
                out.append((df.index[i], df.index[j], "scale out", left / 2, so))
                left, scaled = left / 2, True
            if h[j] >= tp:
                out.append((df.index[i], df.index[j], "take profit", left, tp))
                left = 0
                break
            j += 1
        # a new entry can happen on the exit bar's close (the engine allows re-entry at the close after an exit)
        i = j if j < n else n
    return out


@needs_data
def test_nvda_scale_out_repro_matches_an_independent_calculation():
    s = parser.parse("buy NVDA when RSI(2) is below 10, sell half at +3%, take profit at 6%, stop loss 5%, since 2012")
    r = engine.run(s)
    books(r)
    t = r.trades
    # the reviewer's trade: entered 2012-09-04, half out at +3% and the rest at +6% on 2012-09-06
    tr = t[t.entry_date.astype(str) == "2012-09-04"]
    assert list(tr.exit_reason) == ["scale out", "take profit"] and set(tr.exit_date.astype(str)) == {"2012-09-06"}
    assert list(tr["return"].round(6)) == [0.03, 0.06]
    # the gap: 2013-09-10 opened above the scale-out level and fills at that open, not at the level
    g = t[(t.exit_date.astype(str) == "2013-09-10") & (t.exit_reason == "scale out")]
    assert len(g) == 1 and g.exit_fill.iloc[0] == "open"
    df = data.load("NVDA")
    assert g.exit_price.iloc[0] / g.exit_split_factor.iloc[0] == pytest.approx(df.loc["2013-09-10", "open"])
    # the whole trade list against a plain re-implementation of the rules
    ns = expr.Namespace(df, ticker="NVDA")
    sig = (ns["rsi"](2) < 10).to_numpy() & (df.index >= "2012-01-01")
    want = _independent_nvda_trades(df, sig)
    end = pd.Timestamp(t.exit_date.max())
    want = [w for w in want if w[1] < end]
    got = t[pd.to_datetime(t.exit_date) < end]
    assert len(got) == len(want)
    size = t.groupby("entry_date")["shares"].sum()
    for (_, x), w in zip(got.iterrows(), want):
        assert (str(x.entry_date), str(x.exit_date), x.exit_reason) == (str(w[0].date()), str(w[1].date()), w[2])
        assert x.shares / size[x.entry_date] == pytest.approx(w[3], rel=1e-9)
        assert x.exit_price / x.exit_split_factor == pytest.approx(w[4], rel=1e-9)


@pytest.mark.parametrize("tv", [False, True])
def test_scale_outs_keep_the_books_and_truncation_invariance(fake, tv):
    full = rand_bars(500, seed=9)
    fake["RND"] = full

    def strat():
        return Strategy(universe=["RND"], entry="rsi(2) < 15", entry_fill="next_open" if tv else "close",
                        scale_out=[{"at": 0.02, "fraction": 0.5}, {"at": 0.035, "fraction": 0.5}], take_profit=0.05,
                        stop_loss=0.03, tv_compat=tv, cash_rate=None, start="2019-12-30")
    a = engine.run(strat())
    books(a)
    assert (a.trades.exit_reason == "scale out").sum() > 10
    cut = full.index[300]
    fake["RND"] = full.loc[:cut]
    b = engine.run(strat())
    ta = a.trades[pd.to_datetime(a.trades.exit_date) < cut].reset_index(drop=True)
    tb = b.trades[pd.to_datetime(b.trades.exit_date) < cut].reset_index(drop=True)
    pd.testing.assert_frame_equal(ta.drop(columns=["cum_pnl"]), tb.drop(columns=["cum_pnl"]))
    pd.testing.assert_series_equal(a.equity.loc[: cut - pd.Timedelta(days=1)], b.equity.loc[: cut - pd.Timedelta(days=1)])


# ------------------------------------------------------------ 3. charts draw the series the rule reads

def test_chart_keeps_ref_offsets():
    L = report.chart_layout(["(close > ref(highest(high, 20), 1))"], "NVDA")
    assert L["overlays"] == ["highest(high, 20)[1]"]
    L = report.chart_layout(["(close < ref(lowest(low, 5), 1))"], "SPY")
    assert L["overlays"] == ["lowest(low, 5)[1]"]
    L = report.chart_layout(["(ref(rsi(close, 2), 1) < 10)"], "SPY")
    assert L["panes"] == [{"name": "RSI", "calls": ["rsi(close, 2)[1]"], "levels": [10.0], "range": [0, 100]}]
    # both the shifted and the unshifted series when the rule reads both
    L = report.chart_layout(["close > sma(close, 50) and ref(sma(close, 50), 1) < sma(close, 50)"], "SPY")
    assert set(L["overlays"]) == {"sma(close, 50)", "sma(close, 50)[1]"}
    L = report.chart_layout(["ref(ibs, 1) < 0.2"], "SPY")
    assert L["panes"][0]["calls"] == ["ibs[1]"] and L["panes"][0]["levels"] == [0.2]


def test_chart_values_are_the_shifted_series(fake):
    df = rand_bars(120, seed=2)
    fake["RND"] = df
    s = Strategy(universe=["RND"], entry="close > ref(highest(high, 20), 1)", hold_bars=2, cash_rate=None,
                 start="2019-12-30")
    r = engine.run(s)
    P = report.ticker_chart(r, "RND")
    got = pd.Series(P["overlays"]["highest(high, 20)[1]"], index=pd.to_datetime(P["dates"]), dtype=float)
    want = df["high"].rolling(20).max().shift(1).reindex(got.index)
    assert np.allclose(got.dropna(), want.loc[got.dropna().index].round(4), atol=1e-3)
    assert got.isna().sum() == want.isna().sum()


@needs_data
def test_chart_offsets_in_the_reviewers_examples():
    s = parser.parse("buy NVDA when it closes above its 20 day high, with a 2 ATR stop and a 3 ATR trailing stop, "
                     "target 3R, since 2015")
    assert report.chart_layout([s.entry], "NVDA")["overlays"] == ["highest(high, 20)[1]"]
    s = parser.parse("buy SPY when `close < ta.lowest(low, 5)[1]`, hold 3 days")
    assert report.chart_layout([s.entry], "SPY")["overlays"] == ["lowest(low, 5)[1]"]
    s = parser.parse("buy SPY when `ta.rsi(close, 2)[1] < 10`, hold 3 days")
    assert report.chart_layout([s.entry], "SPY")["panes"][0]["calls"] == ["rsi(close, 2)[1]"]


# ------------------------------------------------------------ 4. TradingView-compatible sizing and holding periods

def test_tv_percent_sizing_uses_the_signal_bars_close_and_whole_shares(fake):
    rows = [[100, 101, 99, 100], [100, 101, 99, 100], [95, 96, 94, 95],       # signal on bar 1 (close 100)
            [96, 97, 95, 96], [97, 98, 96, 97], [97, 98, 96, 97], [97, 98, 96, 97]]
    df = bars(rows)
    df.loc[df.index[1], "volume"] = 2e6
    fake["TVS"] = df
    s = Strategy(universe=["TVS"], entry="volume > 1.5e6", entry_fill="next_open", hold_bars=2, tv_compat=True,
                 capital=10_050, cash_rate=None, start="2019-12-30")
    r = engine.run(s)
    x = r.trades.iloc[0]
    # 100% of 10,050 at the signal bar's close of 100 -> 100.5 -> 100 whole shares, filled at the next open (95)
    assert x.shares == 100 and x.entry_price == 95
    assert s.fractional_shares is False and any(n.startswith("TradingView sizing:") for n in s.notes)
    books(r)
    # without TradingView mode: 100% of equity at the fill price, fractional
    s2 = Strategy(universe=["TVS"], entry="volume > 1.5e6", entry_fill="next_open", hold_bars=2, capital=10_050,
                  cash_rate=None, start="2019-12-30")
    assert engine.run(s2).trades.iloc[0].shares == pytest.approx(10_050 / 95)


def test_tv_order_that_cash_cannot_pay_for_is_skipped(fake):
    rows = [[100, 101, 99, 100], [100, 101, 99, 100], [105, 106, 104, 105],   # a gap up after the signal
            [105, 106, 104, 105], [105, 106, 104, 105], [105, 106, 104, 105]]
    df = bars(rows)
    df.loc[df.index[1], "volume"] = 2e6
    fake["TVG"] = df
    s = Strategy(universe=["TVG"], entry="volume > 1.5e6", entry_fill="next_open", hold_bars=2, tv_compat=True,
                 capital=10_000, cash_rate=None, start="2019-12-30")
    r = engine.run(s)
    assert r.trades.empty
    assert any(n.startswith("TradingView orders skipped: 1 entry order") for n in s.notes)
    # 95% of equity leaves room: 95 shares at 105
    s = Strategy(universe=["TVG"], entry="volume > 1.5e6", entry_fill="next_open", hold_bars=2, tv_compat=True,
                 capital=10_000, position_size=0.95, cash_rate=None, start="2019-12-30")
    assert engine.run(s).trades.iloc[0].shares == 95


def test_tv_fractional_shares_on_request():
    s = parser.parse("buy SPY when RSI(2) is below 10, sell when RSI(2) is above 70, fractional shares", tv_compat=True)
    assert s.fractional_shares is True and not any(n.startswith("TradingView sizing:") for n in s.notes)
    s = parser.parse("buy SPY when RSI(2) is below 10, sell when RSI(2) is above 70", tv_compat=True)
    assert s.fractional_shares is False
    assert parser.parse("buy SPY when RSI(2) is below 10, sell when RSI(2) is above 70").fractional_shares is True


def test_tv_hold_n_days_is_a_strategy_close_filled_at_the_next_open():
    s = parser.parse("buy SPY when RSI(2) is below 10, hold 3 days", tv_compat=True)
    assert (s.entry_fill, s.hold_bars, s.hold_exit_fill) == ("next_open", 4, "open")
    assert any(n.startswith("Holding period in TradingView-compatible mode") for n in s.notes)
    # a stated exit timing is kept
    s = parser.parse("buy SPY when RSI(2) is below 10, hold 3 days and sell at the close", tv_compat=True)
    assert (s.hold_bars, s.hold_exit_fill) == (3, "close")
    # an entry at the close (process_orders_on_close) exits at the close N bars later
    s = parser.parse("buy SPY at the close when RSI(2) is below 10, hold 3 days", tv_compat=True)
    assert (s.hold_bars, s.hold_exit_fill) == (3, "close")
    s = parser.parse("buy SPY when RSI(2) is below 10, hold 3 days")
    assert (s.hold_bars, s.hold_exit_fill) == (3, "close")
