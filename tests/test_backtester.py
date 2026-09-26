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
    # no holding period + an entry at the open: the close of the entry bar (hold 0)
    ("short QQQ at the open when it gaps up 1%, cover at the close", "(gap >= 0.01)", "open", 0, "close"),
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


@needs_data
@pytest.mark.parametrize("rule", ["open < ref(x=close, n=0)", "open < sma(1+0)", "open < sma(-(-1))",
                                  "open < highest(0+1)", "open < ref(close, 2-2)"])
def test_open_guard_cannot_be_bypassed(rule):
    assert not expr.open_safe(rule)
    s = Strategy(universe=["QQQ"], entry=rule, entry_fill="open", hold_bars=1)
    with pytest.raises(ValueError):
        engine.run(s)


def test_open_time_probe_catches_close_use():
    idx = pd.bdate_range("2020-01-01", periods=300)
    rng = np.random.default_rng(1)
    c = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))), index=idx)
    df = pd.DataFrame({"open": c.shift(1).fillna(100), "high": c * 1.01, "low": c * 0.99, "close": c,
                       "volume": 1e6, "dividend": 0.0, "adj_close": c})
    assert expr.open_time_probe("open < close", df) is not None
    assert expr.open_time_probe("open < ref(close, 1)", df) is None


def test_nyse_calendar_month_end_at_data_edge():
    from backtester import calendar
    assert not calendar.is_session("2026-11-26")  # Thanksgiving
    assert not calendar.is_session("2025-04-18")  # Good Friday
    assert calendar.next_sessions("2026-09-25")[0] == pd.Timestamp("2026-09-28")
    idx = pd.bdate_range("2026-08-03", "2026-09-25")
    c = pd.Series(np.linspace(100, 110, len(idx)), index=idx)
    df = pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1e6, "dividend": 0.0, "adj_close": c})
    ns = expr.Namespace(df)
    assert not bool(ns["is_month_end"]().iloc[-1])            # Sept 25 is not the last session of September
    assert int(ns["trading_days_left_in_month"].iloc[-1]) == 4  # 25, 28, 29, 30
    assert bool(ns["is_month_end"]().loc["2026-08-31"])


@needs_data
def test_membership_has_no_etfs_and_quality_gate():
    mem = data.membership()
    for etf in ("TQQQ", "SQQQ", "QQQ", "QLD"):
        assert etf not in mem.columns
    months = mem.sum(axis=1)
    assert months.min() >= 85 and months.max() <= 115
    m, _ = data.member_mask(["GENZ", "SSCC", "AAPL"], data.load("AAPL").loc["2005-01":"2005-06"].index)
    assert not m[:, 0].any() and not m[:, 1].any() and m[:, 2].all()


def test_wiki_components_section_ignores_change_lists():
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location("fetch_data", pathlib.Path(__file__).parents[1] / "scripts" / "fetch_data.py")
    fd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fd)
    text = ("==Investing==\n* ProShares UltraPro QQQ (TQQQ)\n==Components==\n#[[Apple Inc.]] (AAPL)\n#[[Microsoft]] (MSFT)\n"
            "===Historical Components===\n* [[Apollo Group]] (APOL)\n==Yearly Changes==\n* [[Dell]] (DELL)\n")
    assert fd.wiki_tickers(text) == {"AAPL", "MSFT"}


@needs_data
@pytest.mark.parametrize("text, check", [
    ("if the 10 day RSI of QQQ is greater than 79 then buy UVXY else buy TQQQ",
     lambda p: p.tree["if"] == "(rsi(close, 10) > 79)" and p.tree["on"] == "QQQ" and p.rebalance == "daily"),
    ("if QQQ 10 day RSI is greater than SPY 10 day RSI then hold QQQ else hold SPY",
     lambda p: p.tree["if"] == 'rsi(close, 10) > rsi(sym("SPY").close, 10)'),
    ("hold the top 2 of QQQ, SPY, TLT and GLD by 3 month return, only if QQQ is above its 200 day moving average, otherwise SHY",
     lambda p: p.tree["on"] == "QQQ" and "filter" in p.tree["then"] and p.tree["else"] == {"asset": "SHY"}),
    ("minimum variance weighted SPY, TLT and GLD using a 60 day lookback",
     lambda p: p.tree["weights"] == "min_variance" and p.tree["lookback"] == 60),
    ("hold 60% SPY and 40% TLT with 2x leverage", lambda p: p.leverage == 2),
    ("hold 60% SPY and 40% TLT with a 0.5% expense ratio", lambda p: abs(p.expense_ratio - 0.005) < 1e-12),
    ("hold 120% SPY and -20% TLT", lambda p: p.tree["w"] == [1.2, -0.2]),
    ("hold 50% QQQ and 50% (if SPY is above its 200 day moving average then TLT else GLD)",
     lambda p: "if" in p.tree["children"][1]),
])
def test_allocation_parser_round2(text, check):
    p = parser.parse(text)
    assert check(p)


@needs_data
@pytest.mark.parametrize("text", ["hold 60% SPY and 40% TLT, purple elephant",
                                  "hold 60% SPY and 40% TLT, minimum purple weighted"])
def test_allocation_refuses_unknown_words(text):
    with pytest.raises(parser.ParseError):
        parser.parse(text)


@needs_data
def test_leverage_and_optimiser_weights_run():
    from backtester import runner
    r1 = runner.run(parser.parse("hold 60% SPY and 40% TLT, rebalance monthly, since 2010"))
    r2 = runner.run(parser.parse("hold 60% SPY and 40% TLT with 2x leverage, rebalance monthly, since 2010"))
    assert r2.holdings.drop(columns="cash").sum(axis=1).iloc[5] > 1.8
    assert r2.equity.iloc[-1] != r1.equity.iloc[-1]
    r3 = runner.run(parser.parse("risk parity SPY, TLT and GLD over 90 days, rebalance monthly, since 2010"))
    w = r3.holdings.drop(columns="cash").iloc[-1]
    assert abs(w.sum() - 1) < 0.05 and w["TLT"] > w["SPY"]


# ------------------------------------------------------------ round 3: phrasings from TradingView / QuantConnect users

@needs_data
@pytest.mark.parametrize("text, check", [
    # 1. "the 9 EMA", "9 SMA", "the 50 MA", "20 period EMA", "EMA(9)"
    ("buy SPY when the 9 EMA crosses above the 21 EMA, sell when the 9 EMA crosses below the 21 EMA",
     lambda s: s.entry == "(crossover(ema(close, 9), ema(close, 21)))" and s.exit_when == "(crossunder(ema(close, 9), ema(close, 21)))"),
    ("buy SPY when the 9 SMA crosses above the 21 SMA, hold 5 days", lambda s: s.entry == "(crossover(sma(close, 9), sma(close, 21)))"),
    ("buy SPY when it closes above the 50 MA, sell when it closes below the 50 MA",
     lambda s: s.entry == "(close > sma(close, 50))" and s.exit_when == "(close < sma(close, 50))"),
    ("buy SPY when the 20 period EMA crosses above the 50 period SMA, hold 5 days",
     lambda s: s.entry == "(crossover(ema(close, 20), sma(close, 50)))"),
    ("buy QQQ when EMA(9) crosses above EMA(21), hold 5 days", lambda s: s.entry == "(crossover(ema(close, 9), ema(close, 21)))"),
    # 2. "it" in the exit is the entry's indicator
    ("buy AAPL when RSI(2) is below 10, sell when it's over 70", lambda s: s.exit_when.strip("()") == "rsi(close, 2) > 70"),
    ("buy AAPL when RSI(2) is below 10, sell when it is under 30", lambda s: s.exit_when.strip("()") == "rsi(close, 2) < 30"),
    ("buy QQQ when it is down 3 days in a row, sell when it is above 100", lambda s: s.exit_when == "(close > 100)"),
    # 3. rule exits at the open
    ("buy AAPL when RSI(2) is below 10, sell at the open when RSI(2) is above 70",
     lambda s: s.exit_when == "(rsi(close, 2) > 70)" and s.exit_when_fill == "next_open" and s.hold_bars is None),
    ("short AAPL when RSI(2) is above 90, cover at the next open when RSI(2) is below 30",
     lambda s: s.side == "short" and s.exit_when == "(rsi(close, 2) < 30)" and s.exit_when_fill == "next_open" and s.hold_bars is None),
    ("buy QQQ when it gaps down 1%, sell at the open when it gaps up 1%",
     lambda s: s.exit_when == "(gap >= 0.01)" and s.exit_when_fill == "open"),
    # 4. VWAP
    ("buy QQQ when price is above VWAP, hold 5 days", lambda s: s.entry == "(close > vwap(20))" and any("VWAP" in n for n in s.notes)),
    ("buy QQQ when it is above its 10 day VWAP, hold 5 days", lambda s: s.entry == "(close > vwap(10))"),
    # 5. in the market while a condition holds
    ("buy TSLA while it is above its 50 day moving average",
     lambda s: s.entry == "(close > sma(close, 50))" and s.exit_when == "not ((close > sma(close, 50)))"),
    ("hold TSLA when it is above its 50 day moving average",
     lambda s: isinstance(s, Strategy) and s.exit_when == "not ((close > sma(close, 50)))"),
    # 6. "after 3 down days"
    ("buy QQQ at the open after 3 down days, sell at the close",
     lambda s: s.entry == "ref((down_days >= 3), 1)" and s.entry_fill == "open" and s.hold_bars == 0),
    # 7. time exit OR rule exit; a bare RSI takes the entry's period
    ("buy QQQ when RSI(2) is below 10, sell after 10 days or when RSI above 70",
     lambda s: s.hold_bars == 10 and s.exit_when == "(rsi(close, 2) > 70)"),
    ("buy QQQ when RSI(2) is below 10, sell when RSI(2) rises above 70", lambda s: s.exit_when == "(rsi(close, 2) > 70)"),
    # 8. Bollinger bands without the word Bollinger
    ("buy QQQ when it closes below the lower band, sell when it closes above the middle band",
     lambda s: s.entry == "(close < bb_lower(20, 2))" and s.exit_when == "(close > sma(close, 20))"),
    ("buy QQQ when RSI(2) is below 10, sell when it closes above the upper band", lambda s: s.exit_when == "(close > bb_upper(20, 2))"),
    # 9. exits that name a ticker
    ("buy QQQ when it closes above its 50 day moving average, sell when QQQ closes below it",
     lambda s: s.exit_when == "close < sma(close, 50)"),
    ("buy QQQ when SPY closes above its 200 day moving average, sell when SPY closes below it",
     lambda s: s.exit_when.replace("'", '"') == 'sym("SPY").close < sma(sym("SPY").close, 200)'),
    ("buy QQQ when RSI(2) is below 10, sell when SPY closes below its 50 day moving average",
     lambda s: s.exit_when == '(sym("SPY").close < sma(sym("SPY").close, 50))'),
    # 10. rank a universe and fill N slots
    ("buy the 5 Nasdaq 100 stocks with the lowest RSI(2) each day, hold 3 days",
     lambda s: s.universe_name == "NDX" and s.max_positions == 5 and s.rank_by == "rsi(close, 2)" and s.rank_ascending
     and s.entry == "(True)" and s.hold_bars == 3),
    # 11. crossing back
    ("buy QQQ when RSI(2) falls back below 30, hold 3 days", lambda s: s.entry == "(crossunder(rsi(close, 2), 30))"),
    ("buy QQQ when RSI(2) rises back above 30, hold 3 days", lambda s: s.entry == "(crossover(rsi(close, 2), 30))"),
    # 12. trading days of the month
    ("buy SPY on the third trading day of the month, hold 5 days", lambda s: s.entry == "(trading_day_of_month == 3)"),
    ("buy SPY on the last trading day of the month, sell on the first trading day of the next month",
     lambda s: s.entry == "(trading_days_left_in_month == 1)" and s.exit_when == "(trading_day_of_month == 1)"),
    ("buy SPY on the second to last trading day of the month, hold 3 days", lambda s: s.entry == "(trading_days_left_in_month == 2)"),
    # 13. assorted
    ("buy SPY when it is up 3 days in a row, hold 3 days", lambda s: s.entry == "(up_days >= 3)"),
    ("buy SPY when it gaps down more than 2%, hold 3 days", lambda s: s.entry == "(gap < -0.02)"),
    ("buy SPY when it closes in the top 10% of its daily range, hold 3 days", lambda s: s.entry == "(ibs > 0.9)"),
    ("buy SPY when it makes a new 52-week high, hold 3 days", lambda s: s.entry == "(close >= highest(close, 252))"),
    ("buy SPY when it is within 2% of its 52 week high, hold 3 days", lambda s: s.entry == "(drawdown(close, 252) >= -0.02)"),
    ("buy SPY when volume is twice its 20 day average, hold 3 days", lambda s: s.entry == "(volume >= 2 * sma(volume, 20))"),
    ("buy SPY when the 5 day RSI is below 30, hold 3 days", lambda s: s.entry == "(rsi(close, 5) < 30)"),
    ("buy SPY when price crosses above the upper Bollinger band, hold 3 days", lambda s: s.entry == "(crossover(close, bb_upper(20, 2)))"),
    ("buy SPY when ATR(14) is above 2% of price, hold 3 days", lambda s: s.entry == "(natr(14) > 0.02)"),
    ("buy SPY when it closes 2% above its 20 day VWAP, hold 2 days", lambda s: s.entry == "(close >= vwap(20) * 1.02)"),
    # costs
    ("buy SPY when RSI(2) is below 10, hold 3 days, IBKR commissions", lambda s: s.commission_model == "ibkr_fixed"),
    ("buy SPY when RSI(2) is below 10, hold 3 days, Interactive Brokers fixed pricing", lambda s: s.commission_model == "ibkr_fixed"),
    ("buy SPY when RSI(2) is below 10, hold 3 days, IBKR tiered commissions", lambda s: s.commission_model == "ibkr_tiered"),
    ("buy SPY when RSI(2) is below 10, hold 3 days, volume-based slippage", lambda s: s.slippage_model == "volume"),
    ("buy SPY when RSI(2) is below 10, hold 3 days, with market impact", lambda s: s.slippage_model == "volume"),
    ("short QQQ when RSI(2) is above 90, hold 3 days, 30% maintenance margin", lambda s: s.maintenance_margin == 0.3),
])
def test_parser_round3_phrases(text, check):
    s = parser.parse(text)
    assert check(s), s.to_json()


@needs_data
@pytest.mark.parametrize("text, words", [
    ("buy QQQ when RSI(2) is below 10, sell when QQQ closes below it", "nothing to refer back to"),  # no price comparison
    ("buy QQQ when RSI(2) is below 10 and CCI is below -100, sell when it is above 50", "ambiguous"),
    ("buy QQQ after 3 down dayz, hold 2 days", "'3 down dayz'"),                 # the user's words, not '3 s'
    ("buy MSFT when it is down 3 days in a row", "No exit rule"),              # 'when' alone does not imply an exit
    ("buy QQQ when RSI(2) is below 10 and RSI(5) is below 20, sell when RSI is above 70", "which RSI"),
])
def test_parser_round3_refusals(text, words):
    with pytest.raises(parser.ParseError) as e:
        parser.parse(text)
    assert words in str(e.value)


def _week(rows, volume=1e6):
    df = synthetic(rows)
    df.index = pd.bdate_range("2019-12-30", periods=len(rows))  # starts on a Monday
    df["volume"] = volume
    return df


def test_rule_exit_at_same_open_and_next_open(fake):
    fake["X"] = _week([(100, 100, 100, 100), (100, 102, 100, 102), (105, 106, 104, 105), (105, 105, 105, 105)])
    r = engine.run(Strategy(cash_rate=None, universe=["X"], entry="dow == 0", exit_when="gap >= 0.02", exit_when_fill="open"))
    t = r.trades.iloc[0]
    assert t.exit_price == 105 and t.exit_fill == "open" and str(t.exit_date) == "2020-01-01"
    r = engine.run(Strategy(cash_rate=None, universe=["X"], entry="dow == 0", exit_when="close > 101", exit_when_fill="next_open"))
    t = r.trades.iloc[0]
    assert t.exit_price == 105 and t.exit_fill == "open" and str(t.exit_date) == "2020-01-01"
    with pytest.raises(ValueError):
        Strategy(universe=["X"], entry="dow == 0", exit_when="close > 101", exit_when_fill="open").validate()


def test_short_rebate_haircut(fake):
    fake["X"] = _week([(100, 100, 100, 100)] * 6)
    r = 0.0504
    base = dict(cash_rate=r, universe=["X"], entry="dow == 0", side="short", hold_bars=4, maintenance_margin=0.0)
    full = engine.run(Strategy(short_rebate_spread=0.0, **base))
    cut = engine.run(Strategy(short_rebate_spread=0.0025, **base))
    none = engine.run(Strategy(short_rebate_spread=1.0, **base))     # floored at zero: proceeds earn nothing
    nights = 4
    assert full.interest - cut.interest == pytest.approx(nights * 10_000 * 0.0025 / 252, rel=1e-3)
    assert full.interest - none.interest == pytest.approx(nights * 10_000 * r / 252, rel=1e-3)
    for res in (full, cut, none):
        assert res.equity.iloc[-1] == pytest.approx(10_000 + res.trades.pnl.sum() + res.interest)


def test_margin_call_cuts_positions_pro_rata(fake):
    fake["X"] = _week([(100, 100, 100, 100), (120, 120, 120, 120), (150, 150, 150, 150), (170, 170, 170, 170), (170, 170, 170, 170)])
    s = Strategy(cash_rate=None, universe=["X"], entry="dow == 0", side="short", hold_bars=10)
    r = engine.run(s)
    mc = r.trades[r.trades.exit_reason == "margin call"]
    assert len(mc) == 1 and str(mc.iloc[0].exit_date) == "2020-01-02" and mc.iloc[0].exit_price == 170
    # equity 3,000 against a 17,000 short: cut back to 1x, i.e. 3,000 of exposure (17.65 shares) remains
    assert mc.iloc[0].shares == pytest.approx(100 * (1 - 3000 / 17000))
    assert r.trades.shares.sum() == pytest.approx(100)
    assert any(n.startswith("Margin call on 2020-01-02") for n in s.notes)
    assert r.equity.iloc[-1] == pytest.approx(10_000 + r.trades.pnl.sum())
    # disabled: no margin call
    r2 = engine.run(Strategy(cash_rate=None, universe=["X"], entry="dow == 0", side="short", hold_bars=10, maintenance_margin=0))
    assert (r2.trades.exit_reason != "margin call").all()
    with pytest.raises(ValueError):
        Strategy(universe=["X"], entry="True", hold_bars=1, leverage=5).validate()   # 25% maintenance > 20% initial


def test_ibkr_commission_presets(fake):
    from backtester.strategy import broker_commission
    assert broker_commission("ibkr_fixed", 100, 1_000) == pytest.approx(1.0)       # minimum $1
    assert broker_commission("ibkr_fixed", 1_000, 10_000) == pytest.approx(5.0)    # $0.005/share
    assert broker_commission("ibkr_fixed", 1_000, 50) == pytest.approx(0.5)        # capped at 1% of value
    assert broker_commission("ibkr_tiered", 10, 1_000) == pytest.approx(0.35 + 0.002)
    assert broker_commission("ibkr_tiered", 1_000, 10_000) == pytest.approx(3.5 + 0.2)
    fake["X"] = _week([(10, 10, 10, 10)] * 4)
    r = engine.run(Strategy(cash_rate=None, universe=["X"], entry="dow == 0", hold_bars=1, commission_model="ibkr_fixed"))
    t = r.trades.iloc[0]
    assert t.shares * 10 + 0.005 * t.shares <= 10_000 + 1e-9                     # the fee is affordable
    assert t.commission == pytest.approx(2 * 0.005 * t.shares)
    assert r.equity.iloc[-1] == pytest.approx(10_000 + t.pnl)


def test_volume_slippage_square_root_impact(fake):
    fake["X"] = _week([(10, 10, 10, 10)] * 4, volume=1e6)
    s = Strategy(cash_rate=None, universe=["X"], entry="dow == 1", hold_bars=1, slippage_model="volume",
                 spread_bps=2, impact_bps=100)
    r = engine.run(s)
    t = r.trades.iloc[0]
    first_guess = 10_000 / 10                                                     # shares before the impact is known
    bps = 1 + 100 * np.sqrt(first_guess / 1e6)
    assert t.entry_price == pytest.approx(10 * (1 + bps / 1e4))
    assert t.exit_price == pytest.approx(10 * (1 - (1 + 100 * np.sqrt(t.shares / 1e6)) / 1e4))
    assert r.equity.iloc[-1] == pytest.approx(10_000 + t.pnl)


@needs_data
@pytest.mark.parametrize("text", [
    "short QQQ when RSI(2) is above 90, cover at the next open when RSI(2) is below 30, 2x leverage, IBKR commissions, volume-based slippage",
    "buy QQQ when it gaps down 1%, sell at the open when it gaps up 1%",
])
def test_round3_features_do_not_depend_on_future_data(monkeypatch, text):
    test_signal_trades_do_not_depend_on_future_data(monkeypatch, text)
