"""Round 7 (a TradingView expert's review): fixed sizes filled in full or skipped, negative costs refused, diff of
the open is open-safe (and partial lagging is a warning), empty price files, Pine translations (ta.cci, ta.stoch,
ta.vwma, ta.hma, ta.linreg, ta.mfi, ta.wpr, ta.obv, ta.vwap, ta.sar, ta.cross, request.security), new indicators
checked against reference implementations, TradingView-compatible mode, and new sentence phrases."""
import numpy as np
import pandas as pd
import pytest

from backtester import data, engine, expr, parser, report
from backtester.parser import ParseError
from backtester.portfolio import Portfolio
from backtester.strategy import Strategy

HAVE = {"SPY", "AAPL", "MSFT", "NVDA", "QQQ"} <= set(data.available_tickers())
needs_data = pytest.mark.skipif(not HAVE, reason="price data not downloaded")


def bars(rows, start="2019-12-30", volume=1e6):
    idx = pd.bdate_range(start, periods=len(rows))
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx, dtype=float)
    df["volume"] = volume
    df["quote_close"] = df["close"]
    df["adj_close"] = df["close"]
    df["dividend"] = 0.0
    return df


def rand_bars(n=400, seed=0):
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


def check_books(r):
    s = r.strategy
    assert r.equity.iloc[-1] == pytest.approx(s.capital + r.trades.pnl.sum() + r.interest)


# ------------------------------------------------------------ 1. fixed sizes: in full or skipped

def test_fixed_shares_are_whole_and_skipped_without_funds(fake):
    # $100 stock, 100 shares = $10,000 + $1 commission: more than the $10,000 account -> skipped, counted
    fake["X"] = bars([(100, 101, 99, 100)] * 30)
    s = Strategy(universe=["X"], entry="True", hold_bars=2, sizing="fixed_shares", fixed_amount=100, commission=1.0,
                 cash_rate=None)
    r = engine.run(s)
    assert r.trades.empty
    w = [n for n in s.notes if n.startswith("Warning: insufficient funds")]
    assert w and "skipped" in w[0] and "100 shares" in w[0]
    # 50 shares fit: every trade holds exactly 50 whole shares
    s = Strategy(universe=["X"], entry="True", hold_bars=2, sizing="fixed_shares", fixed_amount=50, commission=1.0,
                 cash_rate=None)
    r = engine.run(s)
    assert len(r.trades) and (r.trades["shares"] == 50).all()
    assert not any(n.startswith("Warning: insufficient funds") for n in s.notes)
    check_books(r)


def test_fixed_shares_count_every_skipped_order(fake):
    # 60 shares fit at $100 but not at $200: each order at $200 is skipped (never cut to 50 shares) and counted
    rows = [(100, 101, 99, 100)] * 10 + [(200, 201, 199, 200)] * 10
    fake["X"] = bars(rows)
    s = Strategy(universe=["X"], entry="close > 150 or dow < 5", hold_bars=1, sizing="fixed_shares", fixed_amount=60,
                 cash_rate=None, start="2020-01-13")
    r = engine.run(s)
    assert r.trades.empty
    w = [n for n in s.notes if n.startswith("Warning: insufficient funds")]
    assert w and int(w[0].split(": ")[2].split()[0]) >= 8


def test_fixed_dollars_above_cash_is_skipped(fake):
    fake["X"] = bars([(100, 101, 99, 100)] * 20)
    s = Strategy(universe=["X"], entry="True", hold_bars=1, sizing="fixed_dollars", fixed_amount=20_000, cash_rate=None)
    r = engine.run(s)
    assert r.trades.empty and any(n.startswith("Warning: insufficient funds") for n in s.notes)
    s = Strategy(universe=["X"], entry="True", hold_bars=1, sizing="fixed_dollars", fixed_amount=5_000, cash_rate=None)
    r = engine.run(s)
    assert np.allclose(r.trades["position_value"], 5_000)


@needs_data
def test_reviewer_fixed_share_repro():
    s = parser.parse("buy 100 shares of AAPL when RSI(2) < 5, sell when RSI(2) > 60, $1 commission per trade, since 2024")
    r = engine.run(s)
    assert r.trades.empty or (r.trades["shares"] == 100).all()
    assert any(n.startswith("Warning: insufficient funds") for n in s.notes)


# ------------------------------------------------------------ 2. negative costs

@pytest.mark.parametrize("field", ["slippage_bps", "commission", "commission_per_share", "commission_pct", "borrow_fee",
                                   "margin_rate", "spread_bps", "impact_bps"])
def test_strategy_refuses_negative_costs(field):
    s = Strategy(universe=["SPY"], entry="True", hold_bars=1, **{field: -5.0})
    with pytest.raises(ValueError, match="cannot be negative"):
        s.validate()


@pytest.mark.parametrize("field", ["slippage_bps", "commission", "commission_pct", "expense_ratio", "borrow_fee"])
def test_portfolio_refuses_negative_costs(field):
    p = Portfolio(tree={"asset": "SPY"}, **{field: -0.01})
    with pytest.raises(ValueError, match="cannot be negative"):
        p.validate()


@pytest.mark.parametrize("text", [
    "buy SPY when rsi(2) < 10, hold 3 days, -50 bps slippage",
    "buy SPY when rsi(2) < 10, hold 3 days, -$5 commission",
    "buy SPY when rsi(2) < 10, hold 3 days, commission of -$5",
    "buy SPY when rsi(2) < 10, hold 3 days, slippage of -0.1%",
    "hold 60% SPY and 40% TLT with a -0.5% expense ratio",
])
def test_sentences_refuse_negative_costs(text):
    with pytest.raises(ParseError, match="cannot be negative"):
        parser.parse(text)


def test_cli_and_web_options_refuse_negative_costs():
    from backtester import web
    with pytest.raises(ValueError, match="cannot be negative"):
        parser.parse("buy SPY when rsi(2) < 10, hold 3 days", slippage_bps=-50)
    with pytest.raises(ValueError, match="cannot be negative"):
        web._spec({"text": "buy SPY when rsi(2) < 10, hold 3 days", "options": {"slippage_bps": -50}})
    with pytest.raises(ValueError, match="cannot be negative"):
        web._spec({"spec": {"universe": ["SPY"], "entry": "True", "hold_bars": 1}, "options": {"commission": -5}})


def test_negative_numbers_elsewhere_still_parse():
    p = parser.parse("if TQQQ 6 day cumulative return is less than -12% then hold TECL else hold TQQQ, 5 bps slippage")
    assert p.slippage_bps == 5


# ------------------------------------------------------------ 3. diff of the open; partial lagging is a warning

@pytest.mark.parametrize("rule", ["diff(open) > 0", "diff(open, 3) < 0", "ta.change(open) > 0", "ta.mom(open, 2) > 0",
                                  "diff(ref(close, 1), 5) > 0", "diff(gap, 1) > 0", "diff(2, open) > 0"])
def test_diff_of_open_safe_series_is_open_safe(rule):
    assert expr.open_safe(rule)
    from tests.test_lookahead_adversarial import decisions_invariant, frame
    decisions_invariant(expr.pine_to_rule(rule), frame(500, seed=5))


@pytest.mark.parametrize("rule", ["diff(close) > 0", "diff(5) > 0", "diff() > 0", "diff(open, abs(1)) > 0",
                                  "diff(high, 1) > 0", "ta.change(close) > 0", "diff(open * 0 + close) > 0"])
def test_diff_of_the_close_is_not_open_safe(rule):
    assert not expr.open_safe(rule)


@needs_data
def test_partial_lag_is_a_warning_naming_the_lagged_part():
    s = parser.parse("buy SPY at the open when `ta.change(open) > 0 and ibs < 0.5`, sell at the close")
    assert s.entry == "diff(open) > 0 and ref((ibs < 0.5), 1)"
    w = [n for n in s.notes if n.startswith("Warning: entry at the open")]
    assert w and "`ibs < 0.5`" in w[0] and "`diff(open) > 0` uses today's values" in w[0]
    assert "diff(open) > 0`" not in w[0].split("uses today's close")[0]


# ------------------------------------------------------------ 4. empty price files

def test_empty_price_file_is_missing_data(tmp_path, monkeypatch):
    from backtester import __main__ as cli
    (tmp_path / "ZZEMPTY.csv").write_text("date,open,high,low,close,volume\n")
    monkeypatch.setattr(data, "PRICES", tmp_path)
    with pytest.raises(data.DataError, match="no valid rows"):
        data.load("ZZEMPTY")
    s = Strategy(universe=["ZZEMPTY"], entry="True", hold_bars=1)
    with pytest.raises(data.DataError, match="ZZEMPTY"):
        cli.preflight(s)


# ------------------------------------------------------------ 5. Pine translations

@pytest.mark.parametrize("pine,rule", [
    ("ta.cci(close, 20) < -100", "cci(close, 20) < -100"),
    ("ta.cci(hlc3, 20) < -100", "cci(20) < -100"),
    ("ta.stoch(close, high, low, 14) < 20", "stoch_k(14, 1) < 20"),
    ("close > ta.vwma(close, 20)", "close > vwma(close, 20)"),
    ("close > ta.hma(close, 9)", "close > hma(close, 9)"),
    ("close > ta.linreg(close, 20, 0)", "close > linreg(close, 20, 0)"),
    ("ta.mfi(hlc3, 14) < 20", "mfi(14) < 20"),
    ("ta.wpr(14) < -80", "willr(14) < -80"),
    ("ta.obv > ta.sma(ta.obv, 20)", "obv() > sma(obv(), 20)"),
    ("close > ta.vwap", "close > hlc3"),
    ("close > ta.sar(0.02, 0.02, 0.2)", "close > sar(0.02, 0.2, 0.02)"),
    ("ta.cross(close, ta.sma(close, 20))", "cross(close, sma(close, 20))"),
    ("ta.alma(close, 9, 0.85, 6) > close", "alma(close, 9, 0.85, 6) > close"),
    ("not na(ta.pivothigh(5, 5))", "not na(pivothigh(5, 5))"),
    ("ta.tr > ta.atr(14)", "true_range > atr(14)"),
    ('request.security(syminfo.tickerid, "W", ta.rsi(close, 14)) > 50', "weekly(rsi(close, 14)) > 50"),
    ('request.security(syminfo.tickerid, "M", close) > 0', "monthly(close) > 0"),
    ('request.security(syminfo.tickerid, "D", close) > 0', "close > 0"),
    ('request.security("SPY", "D", close) > 0', "sym('SPY').close > 0"),
    ('close > request.security("AMEX:SPY", "D", ta.sma(close, 200))', "close > sma(sym('SPY').close, 200)"),
    ('request.security(syminfo.tickerid, "W", close, lookahead = barmerge.lookahead_off) > 0', "weekly(close) > 0"),
])
def test_pine_translations(pine, rule):
    assert expr.pine_to_rule(pine) == rule
    expr.compile_expr(pine)


@pytest.mark.parametrize("pine,msg", [
    ('request.security(syminfo.tickerid, "W", close, lookahead = barmerge.lookahead_on) > 0', "lookahead"),
    ('request.security(syminfo.tickerid, "W", close, barmerge.gaps_off, barmerge.lookahead_on) > 0', "lookahead"),
    ('request.security(syminfo.tickerid, "60", close) > 0', "timeframe"),
    ('request.security("SPY", "D", ta.rsi(14)) > 50', "own bars"),
    ("ta.bb(close, 20, 2) > 0", "bb_upper"),
    ("ta.supertrend(3, 10) > 0", "supertrend_dir"),
    ("ta.dmi(14, 14) > 0", "adx"),
    ("ta.stoch(open, high, low, 14) < 20", "ta.stoch"),
])
def test_pine_refusals(pine, msg):
    with pytest.raises(ValueError, match=msg):
        expr.pine_to_rule(pine)


def test_security_higher_timeframe_has_no_lookahead():
    df = rand_bars(300, seed=9)
    rule = 'request.security(syminfo.tickerid, "W", ta.sma(close, 3))'
    full = expr.evaluate_value(rule, expr.Namespace(df))
    for i in (120, 177, 250, 299):
        cut = expr.evaluate_value(rule, expr.Namespace(df.iloc[: i + 1]))
        pd.testing.assert_series_equal(cut, full.iloc[: i + 1], check_names=False)


# ------------------------------------------------------------ 6. new indicators vs reference implementations

def _wma_ref(v, n):
    out = np.full(len(v), np.nan)
    w = np.arange(1, n + 1)
    for i in range(n - 1, len(v)):
        out[i] = np.dot(v[i - n + 1: i + 1], w) / w.sum()
    return out


def _ns(df):
    return expr.Namespace(df)


def ev(rule, df):
    return expr.evaluate_value(rule, _ns(df)).to_numpy()


def test_hma_vwma_linreg_alma_kama_match_references():
    df = rand_bars(300, seed=1)
    c, v = df["close"].to_numpy(), df["volume"].to_numpy()
    # HMA(16): wma(2 wma(8) - wma(16), 4)
    raw = 2 * _wma_ref(c, 8) - _wma_ref(c, 16)
    np.testing.assert_allclose(ev("hma(close, 16)", df), _wma_ref(raw, 4), equal_nan=True)
    np.testing.assert_allclose(ev("hma(close, 10)", df), _wma_ref(2 * _wma_ref(c, 5) - _wma_ref(c, 10), 3), equal_nan=True)
    # VWMA
    ref = np.full(len(c), np.nan)
    for i in range(19, len(c)):
        ref[i] = (c[i - 19: i + 1] * v[i - 19: i + 1]).sum() / v[i - 19: i + 1].sum()
    np.testing.assert_allclose(ev("vwma(close, 20)", df), ref, equal_nan=True)
    # linreg: least squares through the window, evaluated at (n - 1 - offset)
    for off in (0, 2):
        ref = np.full(len(c), np.nan)
        for i in range(19, len(c)):
            b, a = np.polyfit(np.arange(20), c[i - 19: i + 1], 1)
            ref[i] = a + b * (19 - off)
        np.testing.assert_allclose(ev(f"linreg(close, 20, {off})", df), ref, rtol=1e-9, equal_nan=True)
    # ALMA, per Pine's reference loop
    n, off, sig = 9, 0.85, 6
    ref = np.full(len(c), np.nan)
    m, s = off * (n - 1), n / sig
    for t in range(n - 1, len(c)):
        norm = tot = 0.0
        for i in range(n):
            w = np.exp(-((i - m) ** 2) / (2 * s * s))
            norm += w
            tot += c[t - (n - i - 1)] * w
        ref[t] = tot / norm
    np.testing.assert_allclose(ev("alma(close, 9, 0.85, 6)", df), ref, rtol=1e-12, equal_nan=True)
    # KAMA, per TradingView's built-in script
    n = 10
    ref = np.full(len(c), np.nan)
    prev = None
    for t in range(len(c)):
        if t < n:
            continue
        mom = abs(c[t] - c[t - n])
        vol = sum(abs(c[j] - c[j - 1]) for j in range(t - n + 1, t + 1))
        er = mom / vol if vol else 0.0
        a = (er * (2 / 3 - 2 / 31) + 2 / 31) ** 2
        base = c[t] if prev is None else prev
        prev = a * c[t] + (1 - a) * base
        ref[t] = prev
    np.testing.assert_allclose(ev("kama(close, 10)", df), ref, rtol=1e-12, equal_nan=True)


def test_ichimoku_aroon_cmf_match_references():
    df = rand_bars(300, seed=2)
    h, lo, c, v = (df[k].to_numpy() for k in ("high", "low", "close", "volume"))

    def mid(n):
        out = np.full(len(c), np.nan)
        for i in range(n - 1, len(c)):
            out[i] = (h[i - n + 1: i + 1].max() + lo[i - n + 1: i + 1].min()) / 2
        return out
    np.testing.assert_allclose(ev("tenkan(9)", df), mid(9), equal_nan=True)
    np.testing.assert_allclose(ev("kijun(26)", df), mid(26), equal_nan=True)
    # the cloud on bar t is the value computed 25 bars before (plotted with offset displacement - 1)
    a = (mid(9) + mid(26)) / 2
    np.testing.assert_allclose(ev("senkou_a(9, 26, 26)", df), np.r_[np.full(25, np.nan), a[:-25]], equal_nan=True)
    np.testing.assert_allclose(ev("senkou_b(52, 26)", df), np.r_[np.full(25, np.nan), mid(52)[:-25]], equal_nan=True)
    # Aroon: 100 * (n - bars since the extreme of the last n + 1 bars) / n (the most recent extreme)
    n = 14
    up, dn = np.full(len(c), np.nan), np.full(len(c), np.nan)
    for i in range(n, len(c)):
        wh, wl = h[i - n: i + 1], lo[i - n: i + 1]
        up[i] = 100 * (n - (n - max(j for j in range(n + 1) if wh[j] == wh.max()))) / n
        dn[i] = 100 * (n - (n - max(j for j in range(n + 1) if wl[j] == wl.min()))) / n
    np.testing.assert_allclose(ev("aroon_up(14)", df), up, equal_nan=True)
    np.testing.assert_allclose(ev("aroon_down(14)", df), dn, equal_nan=True)
    np.testing.assert_allclose(ev("aroon_osc(14)", df), up - dn, equal_nan=True)
    # CMF
    mfv = np.where(h != lo, (2 * c - lo - h) / (h - lo) * v, 0.0)
    ref = np.full(len(c), np.nan)
    for i in range(19, len(c)):
        ref[i] = mfv[i - 19: i + 1].sum() / v[i - 19: i + 1].sum()
    np.testing.assert_allclose(ev("cmf(20)", df), ref, rtol=1e-12, equal_nan=True)


def test_heikin_ashi_pivots_avwap_match_references():
    df = rand_bars(200, seed=4)
    o, h, lo, c, v = (df[k].to_numpy() for k in ("open", "high", "low", "close", "volume"))
    hc = (o + h + lo + c) / 4
    ho = np.empty(len(c))
    ho[0] = (o[0] + c[0]) / 2
    for i in range(1, len(c)):
        ho[i] = (ho[i - 1] + hc[i - 1]) / 2
    np.testing.assert_allclose(ev("ha_close", df), hc)
    np.testing.assert_allclose(ev("ha_open", df), ho)
    np.testing.assert_allclose(ev("ha_high", df), np.maximum.reduce([h, ho, hc]))
    np.testing.assert_allclose(ev("ha_low", df), np.minimum.reduce([lo, ho, hc]))
    # pivots: reported `right` bars after the pivot bar, strictly above/below both sides
    L, R = 3, 2
    ph, pl = np.full(len(c), np.nan), np.full(len(c), np.nan)
    for t in range(L + R, len(c)):
        k = t - R
        if all(h[k] > h[j] for j in range(k - L, k + R + 1) if j != k):
            ph[t] = h[k]
        if all(lo[k] < lo[j] for j in range(k - L, k + R + 1) if j != k):
            pl[t] = lo[k]
    np.testing.assert_allclose(ev(f"pivothigh({L}, {R})", df), ph, equal_nan=True)
    np.testing.assert_allclose(ev(f"pivotlow(low, {L}, {R})", df), pl, equal_nan=True)
    assert np.isfinite(ph).sum() > 3
    # anchored VWAP of the typical price from the anchor date
    d = df.index[50]
    tp = (h + lo + c) / 3
    ref = np.full(len(c), np.nan)
    ref[50:] = np.cumsum(tp[50:] * v[50:]) / np.cumsum(v[50:])
    np.testing.assert_allclose(ev(f'avwap("{d.date()}")', df), ref, equal_nan=True)


NEW = ["hma(close, 12)", "vwma(close, 10)", "linreg(close, 15)", "linreg(close, 15, 3)", "alma(close, 9, 0.85, 6)",
       "kama(close, 10)", "tenkan()", "kijun()", "senkou_a()", "senkou_b()", "aroon_up(14)", "aroon_down(14)",
       "aroon_osc(14)", "cmf(20)", "ha_open", "ha_close", "ha_high", "ha_low", "pivothigh(4, 3)", "pivotlow(4, 3)",
       "nz(pivothigh(4, 3))", 'avwap("2020-03-02")', "supertrend_dir(10, 3)", "cci(close, 20)", "mfi(14)", "hlc3",
       "true_range", "sar(0.02, 0.2, 0.01)"]


@pytest.mark.parametrize("rule", NEW)
def test_new_indicators_are_causal(rule):
    """Truncating the data at D never changes a value on or before D."""
    df = rand_bars(260, seed=7)
    full = expr.evaluate_value(rule, _ns(df))
    for i in (60, 101, 150, 259):
        cut = expr.evaluate_value(rule, _ns(df.iloc[: i + 1]))
        np.testing.assert_allclose(cut.to_numpy(), full.iloc[: i + 1].to_numpy(), equal_nan=True, err_msg=rule)


@pytest.mark.parametrize("rule", ["hma(20) > open", "open > vwma(open, 5)", "open > tenkan()", "senkou_a() < open",
                                  "aroon_up(14) > 50", "cmf(20) > 0", "ha_close > open", "pivothigh(3, 3) > 0",
                                  'open > avwap("2020-01-01")', "alma(open, 9, 0.85, 6) > open"])
def test_new_indicators_of_the_close_are_refused_at_the_open(rule):
    assert not expr.open_safe(rule)


def test_new_indicators_of_the_open_are_open_safe():
    from tests.test_lookahead_adversarial import decisions_invariant, frame
    for rule in ("open > hma(open, 9)", "open > linreg(open, 10)", "open > kama(open, 10)", "cross(open, ref(high, 1))"):
        assert expr.open_safe(rule), rule
        decisions_invariant(rule, frame(400, seed=8), days=6, reps=2)


def test_supertrend_starts_in_a_downtrend_like_tradingview():
    df = rand_bars(120, seed=11)
    ns = _ns(df)
    d = expr.evaluate_value("supertrend_dir(10, 3)", ns)
    line = expr.evaluate_value("supertrend(10, 3)", ns)
    first = d.first_valid_index()
    assert d[first] == 1.0                                    # direction := 1 while atr[1] is na
    ub = ((df["high"] + df["low"]) / 2 + 3 * expr.evaluate_value("atr(10)", ns))[first]
    assert line[first] == pytest.approx(ub)                    # the upper band
    assert set(d.dropna().unique()) <= {1.0, -1.0}
    ok = d.notna()
    assert ((d[ok] == -1) == (line[ok] < df["close"][ok])).all()


def test_help_lists_new_indicators():
    for name in ("hma(", "vwma(", "linreg(", "alma(", "kama(", "tenkan(", "senkou_a(", "aroon_up(", "cmf(", "ha_open",
                 "pivothigh(", "avwap(", "supertrend_dir(", "request.security"):
        assert name in expr.HELP, name


def test_chart_layout_maps_new_indicators_and_cross_levels():
    L = report.chart_layout(["crossover(rsi(close, 14), 50)", "hma(close, 20) > close and aroon_up(14) > 70",
                             "ha_close > ha_open and cmf(20) > 0.1 and close > senkou_a()"])
    assert {"hma(close, 20)", "ha_close", "ha_open", "senkou_a()"} <= set(L["overlays"])
    panes = {p["name"]: p for p in L["panes"]}
    assert panes["RSI"]["levels"] == [50.0]
    assert panes["Aroon"]["levels"] == [70.0] and panes["CMF"]["levels"] == [0.1]


# ------------------------------------------------------------ 8. TradingView-compatible mode

def test_tv_compat_resolves_stop_and_target_on_one_bar_by_the_ohlc_path(fake):
    # entry at 100 (close); stop 5% (95), target 5% (105); bar 2 touches both
    near_high = [(100, 100, 100, 100)] * 3 + [(104, 106, 94, 100)] + [(100, 100, 100, 100)] * 3
    near_low = [(100, 100, 100, 100)] * 3 + [(96, 106, 94, 100)] + [(100, 100, 100, 100)] * 3
    for rows, tv, want in ((near_high, False, "stop loss"), (near_high, True, "take profit"),
                           (near_low, False, "stop loss"), (near_low, True, "stop loss")):
        fake["X"] = bars(rows)
        s = Strategy(universe=["X"], entry="count(close > 0, 10) == 3", stop_loss=0.05, take_profit=0.05, cash_rate=None,
                     tv_compat=tv)
        r = engine.run(s)
        assert r.trades.iloc[0]["exit_reason"] == want, (tv, rows[3])
        check_books(r)


def test_tv_compat_short_positions_use_the_mirror_path(fake):
    # short at 100: stop above (105), target below (95); open nearer the high -> high first -> the stop
    rows = [(100, 100, 100, 100)] * 3 + [(104, 106, 94, 100)] + [(100, 100, 100, 100)] * 3
    fake["X"] = bars(rows)
    s = Strategy(universe=["X"], entry="count(close > 0, 10) == 3", side="short", stop_loss=0.05, take_profit=0.05,
                 cash_rate=None, tv_compat=True)
    assert engine.run(s).trades.iloc[0]["exit_reason"] == "stop loss"
    rows[3] = (96, 106, 94, 100)
    fake["X"] = bars(rows)
    s = Strategy(universe=["X"], entry="count(close > 0, 10) == 3", side="short", stop_loss=0.05, take_profit=0.05,
                 cash_rate=None, tv_compat=True)
    assert engine.run(s).trades.iloc[0]["exit_reason"] == "take profit"


@needs_data
def test_tv_compat_defaults_unstated_timing_to_the_next_open():
    s = parser.parse("buy SPY when rsi(2) < 10, sell when rsi(2) > 70", tv_compat=True)
    assert s.tv_compat and s.entry_fill == "next_open" and s.exit_when_fill == "next_open"
    assert any(n.startswith("TradingView-compatible mode") for n in s.notes)
    s = parser.parse("buy SPY at the close when rsi(2) < 10, sell at the close when rsi(2) > 70", tv_compat=True)
    assert s.entry_fill == "close" and s.exit_when_fill == "close"
    s = parser.parse("buy SPY when rsi(2) < 10, sell when rsi(2) > 70")
    assert not s.tv_compat and s.entry_fill == "close" and s.exit_when_fill == "close"


def test_tv_compat_is_a_cli_flag_and_a_web_option():
    from backtester import __main__ as cli
    from backtester import web
    a = cli.build_run_parser().parse_args(["buy SPY when rsi(2) < 10, hold 2 days", "--tv-compat"])
    assert cli._overrides(a)["tv_compat"] is True
    assert web._options({"options": {"tv_compat": "true"}}) == {"tv_compat": True}
    assert web._options({"options": {"tv_compat": False}}) == {"tv_compat": False}
    with pytest.raises(web.ClientError):
        web._options({"options": {"tv_compat": "maybe"}})


# ------------------------------------------------------------ 7. phrases

@needs_data
@pytest.mark.parametrize("text,entry,extra", [
    ("buy AAPL, MSFT and NVDA when they cross above their 50 day moving average, sell when they cross below it",
     "(crossover(close, sma(close, 50)))", {"exit_when": "close < sma(close, 50)", "max_positions": 3}),
    ("buy SPY at the open on Monday if it closed down on Friday, sell at the close",
     "(dow == 0) and (ref(dow, 1) == 4 and ref(change, 1) < 0)", {"entry_fill": "open", "hold_bars": 0}),
    ("buy SPY when it pulls back to the 50 day moving average, hold 10 days",
     "(low <= sma(close, 50) and ref(close, 1) > ref(sma(close, 50), 1))", {}),
    ("buy SPY when the close is higher than the high of the previous 3 days, hold 5 days",
     "(close > ref(highest(high, 3), 1))", {}),
    ("buy SPY when it is 3 standard deviations below its 20 day mean, hold 5 days", "(zscore(close, 20) <= -3)", {}),
    ("buy SPY when the 14 day momentum is above 0, sell when the 14 day momentum is below 0",
     "(diff(close, 14) > 0)", {"exit_when": "(diff(close, 14) < 0)"}),
    ("buy SPY when it is above its 200 day moving average, sell when it falls 10% from its peak",
     "(close > sma(close, 200))", {"trailing_stop": 0.10, "exit_when": None}),
    ("buy NVDA when it closes above its 20 day high, with a 10% trailing stop, risk 1% per trade",
     "(close > ref(highest(high, 20), 1))", {"sizing": "risk", "risk_per_trade": 0.01, "trailing_stop": 0.10}),
])
def test_signal_phrases(text, entry, extra):
    s = parser.parse(text)
    assert s.entry == entry
    for k, v in extra.items():
        assert getattr(s, k) == v, k


@needs_data
def test_falls_from_peak_notes():
    s = parser.parse("buy SPY when it is above its 200 day moving average, sell when it falls 10% from its peak")
    assert any("trailing stop" in n and "highest high since entry" in n for n in s.notes)
    s = parser.parse("buy SPY when it falls 10% from its peak, hold 20 days")
    assert s.entry == "(drawdown(close) <= -0.1)" and any("no period" in n for n in s.notes)
    with pytest.raises(ParseError, match="trailing stop"):
        parser.parse("buy SPY when rsi(2) < 10, sell when it falls 10% from its peak or rsi(2) > 70")


@needs_data
def test_weekly_rotation_phrase():
    p = parser.parse("buy the 3 Nasdaq 100 stocks with the highest 20 day rate of change each week")
    assert isinstance(p, Portfolio) and p.rebalance == "weekly"
    f = p.tree["filter"]
    assert f["select"] == "top" and f["n"] == 3 and "20" in f["by"]
    assert any("rotation" in n for n in p.notes)


@needs_data
def test_risk_sizing_with_a_trailing_stop_uses_its_distance():
    s = parser.parse("buy NVDA when it closes above its 20 day high, with a 10% trailing stop, risk 1% per trade, since 2020")
    r = engine.run(s)
    assert len(r.trades)
    first = r.trades.iloc[0]
    # 1% of $10,000 at risk over a 10% distance -> a $1,000 position
    assert first["position_value"] == pytest.approx(1000, rel=0.02)
    assert any(n.startswith("Risk sizing:") for n in s.notes)
