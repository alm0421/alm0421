"""Round 10 (a TradingView expert's review): whole-share scale-outs, TradingView's split-adjusted units, stops that
cover only part of the position, exit levels re-evaluated on every bar, trailing stops that ratchet inside the bar,
the current ATR, Pine var state / ternaries / functions / limit entries / ticks, unbackticked Pine calls and new phrases."""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtester import data, engine, metrics, parser, pine_import
from backtester.parser import ParseError
from backtester.strategy import Strategy

AVAIL = set(data.available_tickers())
needs_data = pytest.mark.skipif(not {"SPY", "QQQ", "NVDA"} <= AVAIL, reason="price data not downloaded")
PINE = Path(__file__).parent / "fixtures" / "pine"


def bars(rows, start="2019-12-30", volume=1e6):
    idx = pd.bdate_range(start, periods=len(rows))
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx, dtype=float)
    df["volume"] = volume
    df["quote_close"] = df["close"]
    df["adj_close"] = df["close"]
    df["dividend"] = 0.0
    return df


def rand_bars(n=600, seed=0, start="2015-01-02"):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, n)))
    o = np.r_[100.0, c[:-1]] * np.exp(rng.normal(0, 0.006, n))
    hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.008, n)))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.008, n)))
    return bars(np.column_stack([o, hi, lo, c]), start=start)


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


def base(**kw):
    d = dict(universe=["SYN"], cash_rate=None, dividends=False, start="2019-12-30")
    d.update(kw)
    return Strategy(**d)


# ------------------------------------------------------------ 1. whole-share scale-outs

@needs_data
def test_scale_outs_are_whole_shares_in_tradingview_mode():
    s = parser.parse("buy QQQ when it closes above its 20 day high, sell half at +5%, take profit at 10%, stop loss 5%, "
                     "since 2018", tv_compat=True)
    r = engine.run(s)
    books(r)
    t = r.trades
    assert (t.shares == np.round(t.shares)).all() and (t.exit_shares == np.round(t.exit_shares)).all()
    so = t[t.exit_reason == "scale out"]
    assert len(so) > 5
    for _, row in so.iterrows():       # half, rounded down; the rest stays and leaves by the stop / target
        whole = t[(t.entry_date == row.entry_date)].shares.sum()
        assert row.shares == math.floor(whole / 2)


@needs_data
def test_a_third_scale_out_with_whole_shares():
    s = parser.parse("buy QQQ when it closes above its 20 day high, sell a third at +4%, take profit at 10%, stop loss 5%, "
                     "whole shares, since 2018")
    r = engine.run(s)
    books(r)
    t = r.trades
    assert (np.abs(t.shares - np.round(t.shares)) < 1e-9).all()
    so = t[t.exit_reason == "scale out"]
    for _, row in so.iterrows():
        whole = t[(t.entry_date == row.entry_date)].shares.sum()
        assert row.shares == pytest.approx(math.floor(whole / 3 + 1e-9), abs=1e-9)


def test_a_scale_out_of_less_than_one_share_is_skipped(fake):
    # $150 buys one $100 share: half of it rounds down to 0, so the scale-out is skipped and the share exits at the target
    rows = [[100, 100, 100, 100], [100, 104, 99, 103], [103, 107, 102, 106], [106, 106, 106, 106]]
    fake["SYN"] = bars(rows)
    fake["SYN"].loc[fake["SYN"].index[0], "volume"] = 2e6
    s = base(entry="volume > 1.5e6", scale_out=[{"at": 0.03, "fraction": 0.5}], take_profit=0.06, capital=150,
             fractional_shares=False)
    r = engine.run(s)
    books(r)
    assert list(r.trades.exit_reason) == ["take profit"] and r.trades.shares.iloc[0] == 1
    assert any(n.startswith("Scale-outs skipped: 1 scale-out") for n in s.notes)


# ------------------------------------------------------------ 2. TradingView's split-adjusted units

def split_bars():
    """Adjusted prices around 100; a 2:1 split on the 6th bar (as traded, the price before it was 200)."""
    rows = [[100, 100, 100, 100], [100, 101, 99, 100], [100, 101, 99, 100], [100, 101, 99, 100], [100, 101, 99, 100],
            [100, 111, 99, 110], [110, 111, 109, 110], [110, 111, 109, 110]]
    df = bars(rows)
    df["split"] = 0.0
    df.loc[df.index[5], "split"] = 2.0
    df.loc[df.index[0], "volume"] = 2e6
    return df


def test_tradingview_mode_sizes_and_lists_in_split_adjusted_units(fake):
    fake["SYN"] = split_bars()
    kw = dict(entry="volume > 1.5e6", hold_bars=6, capital=1150, fractional_shares=False)
    tv = engine.run(base(tv_compat=True, entry_fill="close", **kw))
    books(tv)
    # TradingView: 1150 / 100 on the chart = 11 shares, listed as 11 at $100
    assert tv.trades.shares.iloc[0] == 11 and tv.trades.entry_price.iloc[0] == pytest.approx(100)
    assert tv.trades.exit_shares.iloc[0] == 11
    assert any(n.startswith("TradingView units: SYN split") for n in tv.strategy.notes)
    # default mode: whole shares as traded (5 shares at $200 before the split = 10 after it)
    d = engine.run(base(**kw))
    books(d)
    assert d.trades.shares.iloc[0] == 5 and d.trades.entry_price.iloc[0] == pytest.approx(200)
    assert d.trades.exit_shares.iloc[0] == 10


def tv_reference(df, start):
    """TradingView's broker emulator for 'buy when it closes above its 20 day high, stop at the low of the entry
    (signal) bar, target 2R': quantity = floor(equity / signal close) on the chart's split-adjusted prices, filled at the
    next open (skipped if the cash can't pay for it), the stop / target at the open if gapped, else in the order of the
    OHLC path (open nearer the high: high first)."""
    o, h, lo, c = df.open, df.high, df.low, df.close
    sig = c > h.rolling(20).max().shift(1)
    idx = df.index
    cash, pos, pending, trades = 10000.0, None, None, []
    for i in range(idx.searchsorted(pd.Timestamp(start)), len(idx)):
        if pending is not None and pos is None:
            qty, stop = pending
            pending = None
            px = o.iloc[i]
            if qty >= 1 and qty * px <= cash + 1e-9:
                risk = px - stop
                pos = dict(q=qty, e=px, stop=stop, tgt=px + 2 * risk if risk > 0 else None, ed=idx[i])
                cash -= qty * px
        if pos is not None:
            oo, hh, ll = o.iloc[i], h.iloc[i], lo.iloc[i]
            st, tg, fill = pos["stop"], pos["tgt"], None
            if oo <= st:
                fill = oo
            elif tg is not None and oo >= tg:
                fill = oo
            else:
                for leg in (("H", "L") if abs(hh - oo) < abs(oo - ll) else ("L", "H")):
                    if leg == "H" and tg is not None and hh >= tg:
                        fill = tg
                        break
                    if leg == "L" and ll <= st:
                        fill = st
                        break
            if fill is not None:
                cash += pos["q"] * fill
                trades.append((pos["ed"].date(), idx[i].date(), pos["q"], pos["e"], fill))
                pos = None
        if pos is None and pending is None and sig.iloc[i]:
            pending = (math.floor(cash / c.iloc[i]), lo.iloc[i])
    final = cash + (pos["q"] * c.iloc[-1] if pos else 0.0)
    return trades, final, pos


@needs_data
def test_nvda_matches_tradingview_reference_trade_for_trade():
    s = parser.parse("buy NVDA when it closes above its 20 day high, stop at the low of the entry bar, target 2R, since 2015",
                     tv_compat=True)
    r = engine.run(s)
    books(r)
    ref, final, still = tv_reference(data.load("NVDA"), "2015-01-01")
    t = r.trades[r.trades.exit_reason != "open at end"]
    assert len(t) == len(ref)
    for (ed, xd, q, e, x), (_, row) in zip(ref, t.iterrows()):
        assert (row.entry_date, row.exit_date, row.shares) == (ed, xd, q)
        assert row.entry_price == pytest.approx(e, rel=1e-9) and row.exit_price == pytest.approx(x, rel=1e-9)
    assert r.equity.iloc[-1] == pytest.approx(final, rel=1e-9)
    assert any(n.startswith("TradingView units: NVDA split") for n in s.notes)


# ------------------------------------------------------------ 3. a stop on part of the position

def tranche_case(fake, covers):
    # entry 100 at bar 0's close; bar 1 falls to 94 (the 5% stop at 95); bar 3 rallies to 106 (the +5% scale-out)
    rows = [[100, 100, 100, 100], [99, 99.5, 94, 96], [96, 99, 95.5, 98], [98, 106, 97.5, 105], [105, 105, 105, 105]]
    fake["SYN"] = bars(rows)
    fake["SYN"].loc[fake["SYN"].index[0], "volume"] = 2e6
    s = base(entry="volume > 1.5e6", scale_out=[{"at": 0.05, "fraction": 0.5}], take_profit=0.10, stop_loss=0.05,
             capital=10000, fractional_shares=False, stop_covers_scale_outs=covers)
    r = engine.run(s)
    books(r)
    return r.trades


def test_a_stop_can_leave_the_scale_out_tranche_uncovered(fake):
    t = tranche_case(fake, covers=False)
    assert list(t.exit_reason) == ["stop loss", "scale out"]
    assert list(t.shares) == [50, 50]
    assert t.exit_price.iloc[0] == pytest.approx(95) and t.exit_price.iloc[1] == pytest.approx(105)
    t2 = tranche_case(fake, covers=True)     # the default: the stop covers everything
    assert list(t2.exit_reason) == ["stop loss"] and t2.shares.iloc[0] == 100


@needs_data
def test_pine_partial_exit_without_stop_leaves_its_half_uncovered():
    s = parser.parse((PINE / "scale_out.pine").read_text())
    assert s.stop_covers_scale_outs is False and s.stop_loss == pytest.approx(0.05)
    assert any("cover only the rest" in n for n in s.notes)
    r = engine.run(s)
    books(r)
    assert "stop loss" in set(r.trades.exit_reason)
    # the same stop on both exits covers everything
    txt = (PINE / "scale_out.pine").read_text().replace(
        'qty_percent=50, limit=strategy.position_avg_price * 1.03)',
        'qty_percent=50, limit=strategy.position_avg_price * 1.03, stop=strategy.position_avg_price * 0.95)')
    assert parser.parse(txt).stop_covers_scale_outs is True
    bad = txt.replace("0.95)\nstrategy.exit(\"TP2\"", "0.90)\nstrategy.exit(\"TP2\"")
    with pytest.raises(ParseError, match="line 12:.*differs"):
        parser.parse(bad)


# ------------------------------------------------------------ 4. exit levels re-evaluated on every bar

def atr_series(df, n=14):
    pc = df.close.shift()
    tr = pd.concat([df.high - df.low, (df.high - pc).abs(), (df.low - pc).abs()], axis=1).max(axis=1)
    tr.iloc[0] = df.high.iloc[0] - df.low.iloc[0]
    out, s, acc, cnt = np.full(len(tr), np.nan), None, 0.0, 0
    for i, v in enumerate(tr.to_numpy()):
        if s is None:
            acc += v
            cnt += 1
            if cnt == n:
                s = acc / n
                out[i] = s
        else:
            s = (s * (n - 1) + v) / n
            out[i] = s
    return out


def dyn_reference(df, entry_sig, mult=2.0, hold=15):
    """Entry at the signal bar's close; the stop is the fill - mult x ATR as of the previous close, re-evaluated
    every bar (the entry bar's own ATR for the first check); exit at the stop (the open if gapped) or after `hold`."""
    atr = atr_series(df)
    o, lo, c = df.open.to_numpy(), df.low.to_numpy(), df.close.to_numpy()
    trades, i, n = [], 0, len(df)
    while i < n:
        if entry_sig[i] and np.isfinite(atr[i]):
            e, stop = c[i], c[i] - mult * atr[i]
            j = i + 1
            while j < n:
                if j > i + 1 and np.isfinite(atr[j - 1]):
                    stop = e - mult * atr[j - 1]
                if o[j] <= stop:
                    trades.append((i, j, o[j]))
                    break
                if lo[j] <= stop:
                    trades.append((i, j, stop))
                    break
                if j - i >= hold:
                    trades.append((i, j, c[j]))
                    break
                j += 1
            else:
                break
            i = j
            continue
        i += 1
    return trades


def test_dynamic_stop_level_follows_the_current_atr(fake):
    df = rand_bars(700, seed=3)
    sig = (df.close < df.close.shift(1)) & (df.close.shift(1) < df.close.shift(2))
    df["volume"] = np.where(sig, 2e6, 1e6)
    fake["SYN"] = df
    s = base(entry="volume > 1.5e6", stop_level="entry_price - 2 * atr(14)", dynamic_levels=True, hold_bars=15,
             start="2015-01-02", fractional_shares=True)
    r = engine.run(s)
    books(r)
    ref = dyn_reference(df, sig.to_numpy())
    got = [(df.index.get_loc(pd.Timestamp(a)), df.index.get_loc(pd.Timestamp(b)), x)
           for a, b, x in zip(r.trades.entry_date, r.trades.exit_date, r.trades.exit_price)]
    got = [g for g in got if r.trades.exit_reason.iloc[got.index(g)] != "open at end"]
    assert len(got) == len(ref) > 10
    for (a, b, x), (a2, b2, x2) in zip(ref, got):
        assert (a, b) == (a2, b2) and x == pytest.approx(x2, rel=1e-9)
    # static (the default): the level is fixed at entry, so the results differ
    st = engine.run(base(entry="volume > 1.5e6", stop_level="entry_price - 2 * atr(14)", hold_bars=15,
                         start="2015-01-02", fractional_shares=True))
    assert list(st.trades.exit_date) != list(r.trades.exit_date)


def test_dynamic_levels_have_no_lookahead(fake):
    df = rand_bars(500, seed=5)
    df["volume"] = np.where(df.close < df.close.shift(1), 2e6, 1e6)
    kw = dict(entry="volume > 1.5e6", stop_level="lowest(low, 5)", dynamic_levels=True, stop_ratchet=True,
              take_profit=0.04, start="2015-01-02", tv_compat=True, fractional_shares=True)
    fake["SYN"] = df
    full = engine.run(base(**kw))
    cut = df.index[300]
    fake["SYN"] = df.loc[:cut]
    part = engine.run(base(**kw))
    a = full.equity.loc[: df.index[299]]
    b = part.equity.loc[: df.index[299]]
    assert np.allclose(a.to_numpy(), b.to_numpy())
    done = part.trades[pd.to_datetime(part.trades.exit_date) < cut]
    assert len(done) > 5
    pd.testing.assert_frame_equal(full.trades.iloc[: len(done)][["entry_date", "exit_date", "exit_price"]].reset_index(drop=True),
                                  done[["entry_date", "exit_date", "exit_price"]].reset_index(drop=True))


# ------------------------------------------------------------ 5. trailing stops ratchet inside the bar (TradingView)

def trail_case(fake, bar1, tv):
    rows = [[100, 100, 100, 100], bar1] + [[100, 100, 100, 100]] * 4
    fake["SYN"] = bars(rows)
    fake["SYN"].loc[fake["SYN"].index[0], "volume"] = 2e6
    s = base(entry="volume > 1.5e6", trailing_stop=0.05, hold_bars=4, tv_compat=tv, fractional_shares=True)
    r = engine.run(s)
    books(r)
    return r.trades.iloc[0]


def test_trailing_stop_ratchets_along_tradingviews_path(fake):
    # the open is nearer the high: open -> high (110, the trail rises to 104.5) -> low (103): stopped at 104.5
    t = trail_case(fake, [108, 110, 103, 106], tv=True)
    assert t.exit_reason == "trailing stop" and t.exit_price == pytest.approx(104.5) and str(t.exit_date) == "2019-12-31"
    # default mode: the level set by today's high applies from the next bar (95 was not reached today)
    d = trail_case(fake, [108, 110, 103, 106], tv=False)
    assert str(d.exit_date) != "2019-12-31" or d.exit_reason != "trailing stop"
    # open nearer the low: open -> low (103 > 95, no stop) -> high (110) -> close 104 < 104.5: stopped on the way down
    t = trail_case(fake, [104, 110, 103, 104], tv=True)
    assert t.exit_reason == "trailing stop" and t.exit_price == pytest.approx(104.5)


def trail_reference(df, sig, pct):
    """Entry at the signal close; TradingView's path per bar; the trail = best price x (1 - pct), ratcheting with the
    price along the path; a gap through it fills at the open."""
    o, h, lo, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    trades, i, n = [], 0, len(df)
    while i < n - 1:
        if not sig[i]:
            i += 1
            continue
        peak, j, done = c[i], i + 1, False
        while j < n and not done:
            stop = peak * (1 - pct)
            if o[j] <= stop:
                trades.append((i, j, o[j]))
                done = True
                break
            peak = max(peak, o[j])
            path = (o[j], h[j], lo[j], c[j]) if abs(h[j] - o[j]) < abs(o[j] - lo[j]) else (o[j], lo[j], h[j], c[j])
            for a, b in zip(path[:-1], path[1:]):
                if b > a:
                    peak = max(peak, b)
                elif b < a and b <= peak * (1 - pct):
                    stop = peak * (1 - pct)
                    trades.append((i, j, stop if a > stop else a))
                    done = True
                    break
            j += 1
        if not done:
            break
        i = trades[-1][1]
    return trades


def test_tv_trailing_stop_matches_a_path_reference(fake):
    df = rand_bars(800, seed=11)
    sig = (df.close > df.close.rolling(10).max().shift(1)).to_numpy()
    df["volume"] = np.where(sig, 2e6, 1e6)
    fake["SYN"] = df
    r = engine.run(base(entry="volume > 1.5e6", trailing_stop=0.04, tv_compat=True, entry_fill="close",
                        fractional_shares=True, start="2015-01-02"))
    books(r)
    ref = trail_reference(df, sig, 0.04)
    t = r.trades[r.trades.exit_reason != "open at end"]
    assert len(t) == len(ref) > 10
    for (a, b, x), (_, row) in zip(ref, t.iterrows()):
        assert (df.index[a].date(), df.index[b].date()) == (row.entry_date, row.exit_date)
        assert row.exit_price == pytest.approx(x, rel=1e-9)


# ------------------------------------------------------------ 9. the chandelier stop's ATR

def test_chandelier_uses_the_current_atr_in_tradingview_mode(fake):
    df = rand_bars(500, seed=2)
    df["volume"] = np.where(df.close > df.close.rolling(20).max().shift(1), 2e6, 1e6)
    fake["SYN"] = df
    kw = dict(entry="volume > 1.5e6", trailing_atr=3, start="2015-01-02", fractional_shares=True, entry_fill="close")
    tv = base(tv_compat=True, **kw)
    r_tv = engine.run(tv)
    assert "the current ATR" in tv.summary()
    d = base(**kw)
    r_d = engine.run(d)
    assert "the ATR at entry" in d.summary()
    r_tv_entry = engine.run(base(tv_compat=True, current_atr=False, **kw))
    books(r_tv)
    books(r_d)
    assert list(r_tv.trades.exit_price) != list(r_tv_entry.trades.exit_price)


# ------------------------------------------------------------ 7. R multiples, trailing bar lows

@needs_data
def test_r_multiple_scale_out_and_breakeven(fake):
    # entry 100 at the close, stop at the entry bar's low 96 (R = 4): a third at 1R = 104, breakeven after 1R
    rows = [[99, 100, 96, 100], [100, 105, 99.5, 104], [104, 104.5, 99, 100], [100, 100, 100, 100]]
    fake["SYN"] = bars(rows)
    fake["SYN"].loc[fake["SYN"].index[0], "volume"] = 2e6
    s = parser.parse("buy SPY when `volume > 1.5e6`, stop at the low of the entry bar, sell a third at 1R, "
                     "breakeven after 1R, target 3R", start="2019-12-30")
    s.universe = ["SYN"]
    assert s.scale_out == [{"fraction": pytest.approx(1 / 3), "r": 1.0}] and s.breakeven_r == 1.0
    s.cash_rate, s.fractional_shares = None, True
    r = engine.run(s)
    books(r)
    t = r.trades
    assert list(t.exit_reason) == ["scale out", "breakeven stop"]
    assert t.exit_price.iloc[0] == pytest.approx(104) and t.exit_price.iloc[1] == pytest.approx(100)
    assert t.shares.iloc[0] == pytest.approx(t.shares.sum() / 3)


@needs_data
def test_trailing_stop_at_the_three_bar_low(fake):
    rows = [[100, 101, 99, 100]] * 4 + [[100, 103, 99.5, 102], [102, 104, 101, 103], [103, 103.5, 100, 101],
                                        [101, 101.5, 100.5, 101], [101, 102, 100.8, 101.5], [101.5, 101.5, 99, 99.5],
                                        [99.5, 99.5, 99.5, 99.5]]
    fake["SYN"] = bars(rows)
    fake["SYN"].loc[fake["SYN"].index[3], "volume"] = 2e6
    s = parser.parse("buy SPY when `volume > 1.5e6`, trailing stop at the 3 bar low", start="2019-12-30")
    s.universe = ["SYN"]
    assert (s.stop_level, s.dynamic_levels, s.stop_ratchet) == ("lowest(low, 3)", True, True)
    s.cash_rate = None
    r = engine.run(s)
    books(r)
    t = r.trades.iloc[0]
    # the level on each bar is the lowest low of the 3 bars before it: 99, 99, 99, 99.5, 100 (bars 4-8), 100 at bar 9,
    # whose low 99 reaches it
    assert t.exit_reason == "trailing stop" and str(t.exit_date) == str(fake["SYN"].index[9].date())
    assert t.exit_price == pytest.approx(100)


PHRASES = [
    ("buy SPY on a bullish engulfing, sell after 5 days",
     "ref(close, 1) < ref(open, 1) and close > open and open <= ref(close, 1) and close >= ref(open, 1)"),
    ("short SPY on a bearish engulfing candle, cover after 5 days",
     "ref(close, 1) > ref(open, 1) and close < open and open >= ref(close, 1) and close <= ref(open, 1)"),
    ("buy SPY on a Donchian 55-day breakout, sell after 20 days", "close > ref(highest(high, 55), 1)"),
    ("buy SPY when the MACD line is below zero, sell after 5 days", "macd() < 0"),
    ("buy SPY when tenkan crosses above kijun, sell when tenkan crosses below kijun", "crossover(tenkan(), kijun())"),
    ("buy SPY when the 20 EMA slope is positive, sell after 10 days", "ema(close, 20) > ref(ema(close, 20), 1)"),
    ("buy SPY when the slope of the 50 day SMA is negative, sell after 10 days", "sma(close, 50) < ref(sma(close, 50), 1)"),
]


@needs_data
@pytest.mark.parametrize("text,rule", PHRASES)
def test_new_signal_phrases(text, rule):
    s = parser.parse(text)
    assert rule in s.entry


@needs_data
def test_new_exit_phrases():
    s = parser.parse("buy SPY when RSI(2) < 10, sell when weekly RSI falls below 40")
    assert "crossunder(weekly_rsi(14), 40)" in s.exit_when
    s = parser.parse("buy SPY when RSI(2) < 10, sell on the close")
    assert (s.hold_bars, s.hold_exit_fill) == (1, "close")
    s = parser.parse("buy SPY when tenkan crosses above kijun, sell when tenkan crosses below kijun")
    assert "crossunder(tenkan(), kijun())" in s.exit_when
    s = parser.parse("buy SPY when RSI(2) < 10, stop at yesterday's low, sell after 5 days")
    assert s.stop_level == "ref(low, 1)"
    s = parser.parse("buy SPY at the next open when RSI(2) < 10, stop at yesterday's low, sell after 5 days")
    assert s.stop_level == "low"          # read on the signal bar (yesterday) when the order fills at the open
    s = parser.parse("go long SPY when supertrend flips to up, go short when it flips to down")
    assert s.side == "both" and "crossunder(close, supertrend(10, 3))" in s.short_entry
    # (this used to be refused as "Could not interpret '33.3333 % 1 r'")
    s = parser.parse("buy SPY when RSI(2) < 10, stop at the low of the entry bar, sell a third at 1R, target 3R")
    assert s.scale_out[0]["r"] == 1 and s.target_r == 3


@needs_data
@pytest.mark.parametrize("text,entry,exit_", [
    ("buy SPY when close crosses above ta.ema(close, 20), sell when close crosses below ta.ema(close, 20)",
     "crossover(close, ema(close, 20))", "crossunder(close, ema(close, 20))"),
    ("buy SPY when close > ta.highest(high, 55)[1], sell when close < ta.lowest(low, 20)[1]",
     "close > ref(highest(high, 55), 1)", "close < ref(lowest(low, 20), 1)"),
])
def test_unbackticked_pine_calls(text, entry, exit_):
    s = parser.parse(text)
    assert entry in s.entry and exit_ in s.exit_when


# ------------------------------------------------------------ 8. Pine features

@needs_data
def test_pine_var_state_matches_a_pandas_count():
    s = parser.parse((PINE / "var_streak.pine").read_text())
    assert s.entry == "pv_upDays >= 3" and s.exit_when == "pv_upDays == 0"
    df = data.load("SPY")
    from backtester import expr
    st = pine_import.state_series(s.state_vars, expr.Namespace(df, ticker="SPY"))
    up = (df.close > df.close.shift(1)).astype(int)
    count = up.groupby((up == 0).cumsum()).cumsum()
    assert (st["pv_upDays"].to_numpy() == count.to_numpy()).all()
    # the peak: the close on a reset bar, else the running maximum
    pk = np.empty(len(df))
    for i, (u, c) in enumerate(zip(count.to_numpy(), df.close.to_numpy())):
        pk[i] = c if u == 0 or i == 0 else max(pk[i - 1], c)
    assert np.allclose(st["pv_peak"].to_numpy()[1:], pk[1:])
    r = engine.run(s)
    books(r)
    assert len(r.trades) > 50


@needs_data
def test_pine_var_state_is_causal():
    s = parser.parse((PINE / "var_streak.pine").read_text())
    from backtester import expr
    df = data.load("SPY")
    full = pine_import.state_series(s.state_vars, expr.Namespace(df, ticker="SPY"))
    cut = df.iloc[:3000]
    part = pine_import.state_series(s.state_vars, expr.Namespace(cut, ticker="SPY"))
    for k in full:
        np.testing.assert_allclose(full[k].iloc[:3000].to_numpy(dtype=float), part[k].to_numpy(dtype=float))


@needs_data
def test_pine_ticks_trailing_ternary_function_and_position_checks():
    s = parser.parse((PINE / "ticks_trail.pine").read_text())
    assert s.stop_level == "entry_price - 10" and s.trailing_points == pytest.approx(3)
    assert s.trail_activation_points == pytest.approx(5)
    assert s.entry == "crossover(ema(close, 12), ema(close, 26)) and close > ref(close, 1)"   # isUp() inlined, not inLong
    assert any("$0.01" in n for n in s.notes)
    r = engine.run(s)
    books(r)
    assert "trailing stop" in set(r.trades.exit_reason)


@needs_data
def test_pine_atr_brackets_follow_the_current_atr():
    s = parser.parse((PINE / "atr_brackets.pine").read_text())
    assert (s.stop_atr, s.take_profit_atr, s.current_atr, s.side) == (2, 3, True, "both")
    r = engine.run(s)
    books(r)
    s2 = parser.parse((PINE / "atr_brackets.pine").read_text())
    s2.current_atr = False
    assert list(engine.run(s2).trades.exit_price) != list(r.trades.exit_price)


@needs_data
def test_pine_limit_entry_and_a_stop_that_moves():
    s = parser.parse((PINE / "limit_channel_stop.pine").read_text())
    assert (s.entry_order, s.entry_level, s.order_valid_bars) == ("limit", "close * 0.99", pine_import.GTC_BARS)
    assert s.stop_level == "lowest(low, 10)" and s.dynamic_levels and s.take_profit == pytest.approx(0.05)
    r = engine.run(s)
    books(r)
    assert set(r.trades.entry_fill) <= {"limit", "open"}


H = '//@version=5\nstrategy("x", overlay=true)\n'


def test_pine_ternary_and_functions():
    t = pine_import._ternary("a ? b : c ? d : e", 1)
    assert t == "where(a, b, where(c, d, e))"
    s = parser.parse(H + "f(x, n = 10) => ta.sma(x, n)\nlen = 20\nc = close > f(close) ? 1 : 0\n"
                     "if c > 0 and close > f(open, len)\n    strategy.entry(\"L\", strategy.long)\n"
                     "if close < f(close)\n    strategy.close(\"L\")\n", ticker="SPY")
    assert s.entry == "where(close > sma(close, 10), 1, 0) > 0 and close > sma(open, 20)"
    assert s.exit_when == "close < sma(close, 10)"


def test_pine_trail_price_and_not_in_position():
    s = parser.parse(H + "inLong = strategy.position_size > 0\nif close > open and not inLong\n"
                     "    strategy.entry(\"L\", strategy.long)\n"
                     "strategy.exit(\"T\", \"L\", trail_price = strategy.position_avg_price * 1.05, "
                     "trail_offset = close * 0.03 / syminfo.mintick, stop = strategy.position_avg_price * 0.93)\n",
                     ticker="SPY")
    assert s.entry == "close > open"
    assert (s.trail_activation, s.trailing_stop, s.stop_loss) == (pytest.approx(0.05), pytest.approx(0.03), pytest.approx(0.07))


def test_pine_var_used_before_its_update_is_refused():
    with pytest.raises(ParseError, match="line 5:.*before its update on line 7"):
        parser.parse(H + "var float x = 0\nif close > x\n    strategy.entry(\"L\", strategy.long)\nif close > open\n"
                     "    x := close\n", ticker="SPY")


def test_trail_activation_in_the_engine(fake):
    # $3 trail activated once $5 in favour: bar 1 reaches 104 (not active), bar 2 reaches 106 (active: 103), bar 3 falls
    rows = [[100, 100, 100, 100], [100, 104, 99, 103], [103, 106, 102.5, 105], [105, 105, 101, 102], [102, 102, 102, 102]]
    fake["SYN"] = bars(rows)
    fake["SYN"].loc[fake["SYN"].index[0], "volume"] = 2e6
    r = engine.run(base(entry="volume > 1.5e6", trailing_points=3, trail_activation_points=5, hold_bars=4,
                        fractional_shares=True))
    books(r)
    t = r.trades.iloc[0]
    assert t.exit_reason == "trailing stop" and t.exit_price == pytest.approx(103)
    assert str(t.exit_date) == str(fake["SYN"].index[3].date())


# ------------------------------------------------------------ 9. CAGR from the first bar

def test_signal_cagr_counts_years_from_the_first_bar():
    idx = pd.DatetimeIndex(["2019-12-31", "2020-01-02", "2021-01-04"])
    eq = pd.Series([100.0, 100.0, 121.0], index=idx)
    a = metrics.equity_stats(eq, rf=None, first_bar=idx[1], years_from_first_bar=True)
    b = metrics.equity_stats(eq, rf=None, first_bar=idx[1])
    assert a["years"] == pytest.approx((idx[2] - idx[1]).days / 365.25)
    assert b["years"] == pytest.approx((idx[2] - idx[0]).days / 365.25)
    assert a["cagr"] > b["cagr"]



# ------------------------------------------------------------ the report's chart draws levels that move

def _chromium():
    import glob
    return (glob.glob("/opt/pw-browsers/chromium*/chrome-linux*/chrome") or [None])[0]


@pytest.mark.skipif(_chromium() is None, reason="headless Chromium not available")
def test_report_chart_draws_the_levels_the_engine_moved(fake, tmp_path):
    from playwright.sync_api import sync_playwright
    from backtester import report
    df = rand_bars(300, seed=4)
    df["volume"] = np.where(df.close > df.close.rolling(10).max().shift(1), 2e6, 1e6)
    fake["X"] = df
    s = Strategy(universe=["X"], entry="volume > 1.5e6", trailing_stop=0.04, stop_level="lowest(low, 5)",
                 dynamic_levels=True, tv_compat=True, cash_rate=None, fractional_shares=True)
    res = engine.run(s)
    paths = res.extras["level_paths"]
    assert len(paths) == len(res.trades.entry_date.unique())
    one = next(iter(paths.values()))
    assert len(one["d"]) == len(one["s"]) and any(v is not None for v in one["s"])
    report.write_outputs(report.analyze(res, rf=0.0, sensitivity=False, mc=False), tmp_path, excel=False)
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=_chromium())
        pg = b.new_page(viewport={"width": 1280, "height": 900})
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.goto((tmp_path / "report.html").as_uri())
        pg.wait_for_function("document.querySelector('#pxChart canvas') && document.querySelector('#pxChart canvas').dataset.panes")
        pg.click("#pxRange button[data-v='0']")
        pg.wait_for_function("+document.querySelector('#pxChart canvas').dataset.lvPaths > 0")
        b.close()
    assert not errs
