import numpy as np
import pandas as pd
import pytest

from backtester import data, engine, expr, metrics, parser
from backtester.strategy import Strategy

HAVE = {"MSFT", "QQQ", "SPY"} <= set(data.available_tickers())
needs_data = pytest.mark.skipif(not HAVE, reason="price data not downloaded")


# ------------------------------------------------------------ synthetic data

def synthetic(rows):
    """rows: list of (open, high, low, close)."""
    idx = pd.bdate_range("2020-01-01", periods=len(rows))
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx, dtype=float)
    df["volume"] = 1e6
    df["quote_close"] = df["close"]
    df["dividend"] = 0.0
    df["split"] = 0.0
    return df


@pytest.fixture
def fake(monkeypatch):
    frames = {}

    def load(t):
        return frames[t]

    monkeypatch.setattr(data, "load", load)
    monkeypatch.setattr(data, "load_many", lambda ts: {t: frames[t] for t in ts})
    return frames


def test_hold_one_day_close_to_close(fake):
    fake["X"] = synthetic([(10, 10, 10, 10), (10, 10, 9, 9), (9, 12, 9, 11), (11, 11, 11, 11)])
    s = Strategy(cash_rate=None, universe=["X"], entry="change < 0", hold_bars=1)
    r = engine.run(s)
    assert len(r.trades) == 1
    t = r.trades.iloc[0]
    assert t.entry_price == 9 and t.exit_price == 11
    assert r.equity.iloc[-1] == pytest.approx(10_000 * 11 / 9)


def test_stop_loss_intraday_and_gap(fake):
    # entry at 100 close; next bar trades down to 94 -> 5% stop fills at 95
    fake["X"] = synthetic([(100, 100, 100, 100), (100, 100, 100, 100), (99, 99, 94, 96), (96, 96, 96, 96)])
    s = Strategy(cash_rate=None, universe=["X"], entry="dow == 1", stop_loss=0.05, hold_bars=5)  # 2020-01-02 is a Thursday
    fake["X"].index = pd.bdate_range("2019-12-30", periods=4)  # Mon..Thu; entry on Tue close
    r = engine.run(s)
    t = r.trades.iloc[0]
    assert t.exit_reason == "stop loss" and t.exit_price == pytest.approx(95)

    # gap below the stop fills at the open, not at the stop price
    fake["X"] = synthetic([(100, 100, 100, 100), (100, 100, 100, 100), (90, 91, 88, 90), (90, 90, 90, 90)])
    fake["X"].index = pd.bdate_range("2019-12-30", periods=4)
    r = engine.run(s)
    t = r.trades.iloc[0]
    assert t.exit_reason == "stop loss" and t.exit_price == pytest.approx(90)


def test_take_profit_and_short(fake):
    fake["X"] = synthetic([(100, 100, 100, 100), (100, 100, 100, 100), (100, 101, 89, 95), (95, 95, 95, 95)])
    fake["X"].index = pd.bdate_range("2019-12-30", periods=4)
    s = Strategy(cash_rate=None, universe=["X"], entry="dow == 1", side="short", take_profit=0.10, hold_bars=5)
    r = engine.run(s)
    t = r.trades.iloc[0]
    assert t.exit_reason == "take profit" and t.exit_price == pytest.approx(90)
    assert t.pnl == pytest.approx(10_000 * 0.10)


def test_slippage_and_commission(fake):
    fake["X"] = synthetic([(10, 10, 10, 10), (10, 10, 9, 10), (10, 10, 10, 10), (10, 10, 10, 10)])
    s = Strategy(cash_rate=None, universe=["X"], entry="dow == 1", hold_bars=1, slippage_bps=10, commission=1)
    fake["X"].index = pd.bdate_range("2019-12-30", periods=4)
    r = engine.run(s)
    t = r.trades.iloc[0]
    assert t.entry_price == pytest.approx(10.01) and t.exit_price == pytest.approx(9.99)
    assert r.equity.iloc[-1] == pytest.approx(10_000 + t.pnl)
    assert t.commission == pytest.approx(2)


def test_max_positions_and_no_leverage(fake):
    for k in "ABC":
        fake[k] = synthetic([(10, 10, 10, 10)] * 3 + [(10, 10, 9, 9), (9, 9, 9, 10), (10, 10, 10, 10)])
    s = Strategy(cash_rate=None, universe=list("ABC"), entry="change < 0", hold_bars=1, max_positions=2)
    r = engine.run(s)
    assert len(r.trades) == 2
    assert r.exposure.max() <= 1.0 + 1e-9


# ------------------------------------------------------------ expression safety

def test_expression_sandbox():
    ns = expr.Namespace(synthetic([(1, 1, 1, 1)] * 5))
    for bad in ["__import__('os')", "close.to_csv('x')", "().__class__", "open.__class__", "lambda: 1"]:
        with pytest.raises((ValueError, SyntaxError)):
            expr.evaluate(bad, ns)
    assert expr.evaluate("0.1 < ibs < 0.9 or not (close > 0)", ns).dtype == bool


def test_streak():
    s = pd.Series([5, 4, 3, 3, 2, 1, 2, 1.0])
    assert expr.streak(s, -1).tolist() == [0, 1, 2, 0, 1, 2, 0, 1]


# ------------------------------------------------------------ parser

@pytest.mark.parametrize("text, entry, fill, hold, hold_fill", [
    ("buy at the close Microsoft when it trades down 5 days in a row, hold for 1 day, and sell at the close",
     "(down_days >= 5)", "close", 1, "close"),
    ("Buy MSFT on the close after 3 consecutive down days. Sell at the next day's open.",
     "(down_days >= 3)", "close", 1, "open"),
    ("buy MSFT at the close when it is down exactly 3 days in a row, hold 2 days",
     "(down_days == 3)", "close", 2, "close"),
    ("short QQQ at the open when it gaps up 1%, cover at the close", "(gap >= 0.01)", "open", 1, "close"),
    ("buy SPY at the close on the last trading day of the month, sell at the close 5 days later",
     "(trading_days_left_in_month == 1)", "close", 5, "close"),
    ("buy AAPL at the next open when RSI(2) < 10, hold 3 days", "(rsi(close, 2) < 10)", "next_open", 3, "close"),
])
def test_parser_cases(text, entry, fill, hold, hold_fill):
    s = parser.parse(text)
    assert s.entry == entry and s.entry_fill == fill and s.hold_bars == hold and s.hold_exit_fill == hold_fill


def test_parser_open_entry_never_peeks_at_close():
    s = parser.parse("buy QQQ at the open when it gaps down 1% and SPY is above its 200 day moving average, sell at the close")
    assert s.entry_fill == "open"
    assert "gap <= -0.01" in s.entry
    assert 'ref((sym("SPY").close > sma(sym("SPY").close, 200)), 1)' in s.entry.replace("'", '"')


def test_parser_refuses_unknown():
    with pytest.raises(parser.ParseError):
        parser.parse("buy MSFT when the moon is full, hold 1 day")
    with pytest.raises(parser.ParseError):
        parser.parse("buy MSFT when it is down 3 days in a row")  # no exit


def test_parser_exits_and_sizing():
    s = parser.parse("buy Nasdaq 100 stocks at the close when RSI(2) is below 5, exit when RSI(2) > 70 or after 10 days, "
                     "max 5 positions, 5% stop loss, 10 bps slippage, since 2005")
    assert s.exit_when == "(rsi(close, 2) > 70)" and s.hold_bars == 10 and s.max_positions == 5
    assert s.stop_loss == 0.05 and s.slippage_bps == 10 and s.start == "2005-01-01"
    assert len(s.universe) >= 90


# ------------------------------------------------------------ real data

@needs_data
def test_msft_example_matches_vectorised():
    s = parser.parse("buy at the close Microsoft when it trades down 5 days in a row, hold for 1 day, and sell at the close")
    s.cash_rate = None
    r = engine.run(s)
    df = data.load("MSFT")
    c, div = df["close"], df["dividend"]
    down = (c.diff() < 0).astype(int)
    streak = down.groupby((down == 0).cumsum()).cumsum()
    # next-day return including a dividend paid on the exit day (ex-date while held overnight)
    nxt = ((c.shift(-1) + div.shift(-1)) / c - 1)[streak >= 5].dropna()
    assert len(r.trades) == len(nxt)
    assert r.equity.iloc[-1] == pytest.approx(10_000 * (1 + nxt).prod())


@needs_data
def test_gap_short_matches_vectorised():
    s = parser.parse("short QQQ at the open when it gaps up 1%, cover at the close")
    s.cash_rate = None
    r = engine.run(s)
    q = data.load("QQQ")
    sig = q.open / q.close.shift() - 1 >= 0.01
    assert r.equity.iloc[-1] == pytest.approx(10_000 * (2 - q.close / q.open)[sig].prod())


@needs_data
def test_metrics_basics():
    r = engine.run(parser.parse("buy SPY at the close when it drops 2% in a day, hold 5 days"))
    st = metrics.equity_stats(r.equity)
    assert -1 < st["max_drawdown"] <= 0
    assert r.interest > 0  # idle cash earns T-bill interest
    assert st["end_equity"] == pytest.approx(10_000 + r.trades.pnl.sum() + r.interest)
    y = metrics.yearly_detail(r.equity, r.trades, r.exposure)
    assert np.prod(1 + y["return"]) == pytest.approx(r.equity.iloc[-1] / 10_000)
