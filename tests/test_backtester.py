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


# ------------------------------------------------------------ lookahead: the future must not change the past

def _truncated(monkeypatch, cut):
    real = data.load.__wrapped__ if hasattr(data.load, "__wrapped__") else data.load
    cache = {}

    def load(t):
        t = data.canonical(t)
        if t not in cache:
            df = real(t)
            cache[t] = df[df.index <= cut]
        return cache[t]
    monkeypatch.setattr(data, "load", load)
    monkeypatch.setattr(data, "load_many", lambda ts: {data.canonical(t): load(t) for t in ts})


@needs_data
@pytest.mark.parametrize("text", [
    "buy QQQ at the close when RSI(2) is below 10 and SPY is above its 200-day moving average, sell when it closes above its 5 day moving average",
    "buy QQQ at the open when it gaps down 1% and SPY is above its 200 day moving average, sell at the close",
    "buy MSFT at a limit 2% below the close when it is down 3 days in a row, hold 3 days, with a 2 ATR stop",
    "go long SPY when it closes above its 50 day moving average, go short SPY when it closes below its 50 day moving average",
])
def test_signal_trades_do_not_depend_on_future_data(monkeypatch, text):
    cut = pd.Timestamp("2015-06-30")
    full = engine.run(parser.parse(text + ", until 2020"))
    _truncated(monkeypatch, cut)
    past = engine.run(parser.parse(text + ", until 2020"))
    done = lambda tr: tr[pd.to_datetime(tr["exit_date"]) < cut - pd.Timedelta(days=5)][["ticker", "entry_date", "exit_date", "shares", "pnl"]].reset_index(drop=True)  # noqa: E731
    pd.testing.assert_frame_equal(done(full.trades), done(past.trades))


@needs_data
def test_allocation_does_not_depend_on_future_data(monkeypatch):
    from backtester import portfolio
    text = "if SPY is above its 200-day moving average hold QQQ, otherwise hold TLT, rebalance daily"
    cut = pd.Timestamp("2016-01-29")
    full = portfolio.run(parser.parse(text))
    _truncated(monkeypatch, cut)
    past = portfolio.run(parser.parse(text))
    a = full.equity[full.equity.index < cut - pd.Timedelta(days=3)]
    b = past.equity.reindex(a.index)
    assert np.allclose(a.to_numpy(), b.to_numpy(), rtol=1e-12)


def test_negative_offsets_rejected():
    ns = expr.Namespace(synthetic([(1, 1, 1, 1)] * 5))
    for bad in ["ref(close, -1) > close", "ret(close, -1) > 0", "sma(close, 0) > 1"]:
        with pytest.raises(ValueError):
            expr.evaluate(bad, ns)


def test_open_fill_requires_open_safe_rule():
    for rule in ["ret(1) > 0.01", "close > open", "ref(close, 0) > open", "rsi(2) < 10"]:
        with pytest.raises(ValueError):
            Strategy(universe=["X"], entry=rule, entry_fill="open", hold_bars=1).validate()
    Strategy(universe=["X"], entry="gap < -0.01 and ref(rsi(2), 1) < 10", entry_fill="open", hold_bars=1).validate()


# ------------------------------------------------------------ allocation engine

@needs_data
def test_sixty_forty_matches_independent_calculation():
    from backtester import portfolio as pf
    p = pf.Portfolio(tree={"weights": "specified", "w": [0.6, 0.4], "children": [{"asset": "SPY"}, {"asset": "QQQ"}]},
                     rebalance="monthly", cash_rate=None)
    r = pf.run(p)

    def tr(t):
        d = data.load(t)
        return (d.close + d.dividend) / d.close.shift() - 1
    ret = pd.concat([tr("SPY"), tr("QQQ")], axis=1).loc[r.equity.index[1]:].fillna(0)
    w = np.array([0.6, 0.4])
    hold = 10_000 * w
    ends = set(pd.Series(ret.index, index=ret.index).groupby(ret.index.to_period("M")).max())
    for n, (d, row) in enumerate(ret.iterrows()):
        if n:
            hold = hold * (1 + row.values)
        if d in ends:
            hold = hold.sum() * w
    assert r.equity.iloc[-1] == pytest.approx(hold.sum(), rel=1e-9)


@needs_data
def test_contributions_buy_at_that_days_price():
    from backtester import portfolio as pf
    p = pf.Portfolio(tree={"asset": "SPY"}, rebalance="none", cash_rate=None, contribution=500,
                     contribution_freq="monthly", start="2010-01-01", end="2012-12-31")
    r = pf.run(p)
    d = data.load("SPY").loc["2010-01-01":"2012-12-31"]
    g = ((d.close + d.dividend) / d.close.shift()).fillna(1.0)
    growth = g.cumprod()
    flows = r.extras["flows"]
    expected = 10_000 * growth.iloc[-1] / growth.iloc[0]
    for day, amt in flows[flows > 0].items():
        expected += amt * growth.iloc[-1] / growth.loc[day]
    assert r.equity.iloc[-1] == pytest.approx(expected, rel=1e-6)


@needs_data
def test_filter_respects_point_in_time_membership():
    from backtester import portfolio as pf
    if data.membership() is None:
        pytest.skip("no membership history")
    p = parser.parse("hold the top 10 Nasdaq 100 stocks by 3 month momentum, rebalance monthly, from 2010 to 2011")
    r = pf.run(p)
    mask, _ = data.member_mask(list(r.holdings.columns.drop("cash")), r.holdings.index)
    held = r.holdings.drop(columns="cash").to_numpy() > 1e-9
    # anything held on a rebalance day must have been a member that day
    ends = r.holdings.index.to_series().groupby(r.holdings.index.to_period("M")).max()
    rows = [r.holdings.index.get_loc(d) for d in ends]
    assert not (held[rows] & ~mask[rows]).any()


# ------------------------------------------------------------ signal engine features

def test_dividends_paid_to_longs_and_charged_to_shorts(fake):
    df = synthetic([(10, 10, 10, 10)] * 4)
    df["dividend"] = [0, 0, 0.5, 0]
    df.index = pd.bdate_range("2019-12-30", periods=4)
    fake["X"] = df
    for side, sign in (("long", 1), ("short", -1)):
        r = engine.run(Strategy(cash_rate=None, universe=["X"], entry="dow == 0", side=side, hold_bars=3))
        t = r.trades.iloc[0]
        assert t.income == pytest.approx(sign * 0.5 * t.shares)
        assert r.equity.iloc[-1] == pytest.approx(10_000 + t.pnl)


def test_limit_entry_fills_at_limit_or_better(fake):
    df = synthetic([(100, 100, 100, 100), (100, 100, 100, 100), (99, 100, 97, 99), (99, 99, 99, 99)])
    df.index = pd.bdate_range("2019-12-30", periods=4)
    fake["X"] = df
    r = engine.run(Strategy(cash_rate=None, universe=["X"], entry="dow == 1", entry_order="limit",
                            entry_level="close * 0.98", hold_bars=1))
    assert r.trades.iloc[0].entry_price == pytest.approx(98.0)
    # a gap below the limit fills at the open
    df2 = df.copy()
    df2.iloc[2] = [96, 97, 95, 96, 1e6, 96, 0, 0]
    fake["X"] = df2
    r = engine.run(Strategy(cash_rate=None, universe=["X"], entry="dow == 1", entry_order="limit",
                            entry_level="close * 0.98", hold_bars=1))
    assert r.trades.iloc[0].entry_price == pytest.approx(96.0)


def test_pyramiding_and_scale_out(fake):
    df = synthetic([(10, 10, 10, 10), (10, 10, 10, 10), (10, 10, 10, 10), (10, 12, 10, 11), (11, 11, 11, 11)])
    df.index = pd.bdate_range("2019-12-30", periods=5)
    fake["X"] = df
    r = engine.run(Strategy(cash_rate=None, universe=["X"], entry="dow <= 1", pyramiding=2, position_size=0.4,
                            scale_out=[{"at": 0.1, "fraction": 0.5}], hold_bars=4))
    assert (r.trades.exit_reason == "scale out").sum() == 2          # half of each of the two lots
    assert r.trades.shares.sum() == pytest.approx(0.4 * 10_000 / 10 + 0.4 * 10_000 / 10)


def test_leverage_allows_more_than_equity(fake):
    fake["X"] = synthetic([(10, 10, 10, 10)] * 3 + [(10, 10, 10, 11)])
    r = engine.run(Strategy(cash_rate=None, universe=["X"], entry="dow == 0", hold_bars=1, leverage=2.0))
    assert r.trades.iloc[0].position_value == pytest.approx(20_000)


# ------------------------------------------------------------ parser: allocations, strictness

@needs_data
@pytest.mark.parametrize("text, check", [
    ("hold 60% SPY and 40% QQQ, rebalance monthly", lambda p: p.tree["w"] == [0.6, 0.4] and p.rebalance == "monthly"),
    ("60/40 SPY/TLT rebalanced quarterly", lambda p: p.tree["w"] == [0.6, 0.4] and p.rebalance == "quarterly"),
    ("buy and hold QQQ", lambda p: p.tree == {"asset": "QQQ"} and p.rebalance == "none"),
    ("if SPY is above its 200-day moving average hold QQQ, otherwise hold TLT", lambda p: p.tree["on"] == "SPY" and p.tree["then"] == {"asset": "QQQ"}),
    ("hold QQQ when it is above its 10-month moving average, otherwise cash, rebalance monthly", lambda p: "monthly_sma(10)" in p.tree["if"]),
    ("hold the top 5 Nasdaq 100 stocks by 3 month momentum, rebalance monthly", lambda p: p.tree["filter"]["n"] == 5 and p.tree["universe"] == "NDX"),
    ("rotate monthly between QQQ, SPY and TLT by 3 month return", lambda p: p.tree["filter"]["n"] == 1 and p.rebalance == "monthly"),
    ("dual momentum between SPY and EFA with AGG as the safe asset", lambda p: p.tree["fallback"] == {"asset": "AGG"}),
    ("buy and hold SPY, add $500 every month", lambda p: p.contribution == 500),
    ("hold 60% SPY and 40% AGG, withdraw 4% per year adjusted for inflation, starting with $1,000,000, rebalance annually",
     lambda p: p.withdrawal == 40_000 and p.inflation_adjust and p.capital == 1_000_000),
    ("hold 70% QQQ and 30% TLT, rebalance when any weight drifts 5% from target", lambda p: p.drift_band == 0.05 and p.rebalance == "none"),
])
def test_allocation_parser(text, check):
    from backtester.portfolio import Portfolio
    p = parser.parse(text)
    assert isinstance(p, Portfolio) and check(p)


@needs_data
@pytest.mark.parametrize("text", [
    "buy TQQQX when RSI(2) is below 10, hold 3 days",                       # unknown ticker
    "buy QQQ when it is down 3 days in a row, hold 3 days, some slippage",    # unparsed cost phrase
    "rotate monthly between QQQ, SPY and TLT by the phase of the moon",
    "buy MSFT when it is down 3 days in a row, hold 2 days, and pray",
])
def test_parser_refuses_instead_of_guessing(text):
    with pytest.raises(parser.ParseError):
        parser.parse(text)


@needs_data
def test_parser_signal_extras():
    s = parser.parse("buy AAPL when RSI(2) is below 10 but only if SPY is above its 200-day moving average, hold 3 days, 0.5% slippage, 2x leverage")
    assert "rsi(close, 2) < 10" in s.entry and 'sym("SPY")' in s.entry
    assert s.slippage_bps == 50 and s.leverage == 2 and s.position_size == 2
    s = parser.parse("buy QQQ when it crosses above its 50 day moving average, sell when it crosses back below")
    assert s.exit_when == "close < sma(close, 50)"
    s = parser.parse("buy NVDA when MACD crosses above its signal line, with a 2 ATR stop and a 3 ATR trailing stop, risk 1% per trade")
    assert s.stop_atr == 2 and s.trailing_atr == 3 and s.sizing == "risk"


def test_sweep_expansion():
    from backtester import research
    labels, texts, combos = research.expand("buy QQQ when RSI({2..3}) is below {5..15 step 5}, hold {1,3} days")
    assert len(texts) == 12 and texts[0] == "buy QQQ when RSI(2) is below 5, hold 1 days"
