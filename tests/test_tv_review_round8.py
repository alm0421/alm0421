"""Round 8 (a TradingView expert's review): the Build form round-trips every field (Playwright), share links restore
every setting, TradingView's stochastic defaults, one survivorship figure per run, price-only returns in
TradingView-compatible mode, clearer errors, limit/stop entry labels, new phrases, and stops / targets at a price
expression fixed at entry (stop_level, target_level, target_r) plus "trail the rest"."""
import glob
import shutil
import subprocess
import threading
import time

import numpy as np
import pandas as pd
import pytest

from backtester import data, engine, parser, web
from backtester.parser import ParseError
from backtester.strategy import Strategy

AVAIL = set(data.available_tickers())
needs_data = pytest.mark.skipif(not {"SPY", "NVDA", "AAPL"} <= AVAIL, reason="price data not downloaded")


def bars(rows, start="2019-12-30", volume=1e6, dividend=None):
    idx = pd.bdate_range(start, periods=len(rows))
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx, dtype=float)
    df["volume"] = volume
    df["quote_close"] = df["close"]
    df["adj_close"] = df["close"]
    df["dividend"] = 0.0 if dividend is None else dividend
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


def sig(text, **kw):
    return parser.parse(text, **kw)


# ------------------------------------------------------------ 3. stochastic: TradingView's defaults (14, 1, 3)

def test_stochastic_defaults_follow_tradingview():
    s = sig("buy SPY when %K crosses above %D, hold 3 days")
    assert s.entry == "(crossover(stoch_k(14, 1), stoch_d(14, 1, 3)))"
    assert any("%K smoothing 1" in n for n in s.notes)
    assert sig("buy SPY when stochastic below 20, hold 3 days").entry == "(stoch_k(14, 1) < 20)"
    s = sig("buy SPY when the slow stochastic is below 20, hold 3 days")
    assert s.entry == "(stoch_k(14, 3) < 20)" and any(n.startswith("Slow stochastic") for n in s.notes)
    assert "stoch_k(14, 3), stoch_d(14, 3, 3)" in sig("buy SPY when the slow stochastic %K crosses above %D, hold 3 days").entry


def test_stoch_d_is_the_3_bar_sma_of_k():
    from backtester import expr
    df = rand_bars(80)
    ns = expr.Namespace(df)
    k = ns["stoch_k"](14, 1)
    ll, hh = df["low"].rolling(14).min(), df["high"].rolling(14).max()
    pd.testing.assert_series_equal(k, 100 * (df["close"] - ll) / (hh - ll), check_names=False)
    pd.testing.assert_series_equal(ns["stoch_d"](14, 1, 3), k.rolling(3).mean(), check_names=False)


# ------------------------------------------------------------ 5. TradingView-compatible mode: price-only returns

def test_tv_mode_defaults_to_no_interest_and_no_dividends():
    s = sig("buy SPY when rsi(2) < 10, sell when rsi(2) > 70", tv_compat=True)
    assert s.cash_rate is None and s.dividends is False
    note = next(n for n in s.notes if n.startswith("TradingView-compatible returns:"))
    assert "cash earns nothing" in note and "dividends are not credited" in note and "price-only" in note
    s = sig("buy SPY when rsi(2) < 10, sell when rsi(2) > 70, with interest on cash, with dividends", tv_compat=True)
    assert s.cash_rate == "tbill" and s.dividends is True
    s = sig("buy SPY when rsi(2) < 10, sell when rsi(2) > 70")
    assert s.cash_rate == "tbill" and s.dividends is True


@pytest.mark.parametrize("phrase,cash,div", [("no dividends", "tbill", False), ("no interest on cash", None, True),
                                             ("cash earns nothing", None, True), ("price-only returns", None, False)])
def test_interest_and_dividend_phrases_in_every_mode(phrase, cash, div):
    for tv in (False, True):
        s = sig(f"buy SPY when rsi(2) < 10, hold 3 days, {phrase}", tv_compat=tv)
        assert s.dividends is (div and not tv)      # TradingView-compatible mode credits none unless asked
        assert s.cash_rate == (cash if not tv or cash is None else None)


def test_portfolio_refuses_no_dividends_clearly():
    with pytest.raises(ParseError, match="always receives its holdings' dividends"):
        sig("hold 60% SPY and 40% TLT, rebalance monthly, no dividends")


def test_json_spec_in_tv_mode_defaults_to_no_interest():
    s = Strategy.from_dict({"universe": ["SPY"], "entry": "rsi(2) < 10", "hold_bars": 3, "tv_compat": True})
    s.validate()
    assert s.cash_rate is None and s.dividends is False
    s = Strategy.from_dict({"universe": ["SPY"], "entry": "rsi(2) < 10", "hold_bars": 3, "tv_compat": True, "cash_rate": "tbill"})
    s.validate()
    assert s.cash_rate == "tbill"


def test_dividends_off_are_not_credited(fake):
    rows = [[100, 101, 99, 100]] * 30
    div = np.zeros(30)
    div[10] = 2.0
    fake["DIVX"] = bars(rows, dividend=div)
    kw = dict(universe=["DIVX"], entry="True", hold_bars=25, cash_rate=None, start="2019-12-30")
    on = engine.run(Strategy(**kw))
    off = engine.run(Strategy(**kw, dividends=False))
    assert on.trades.income.sum() > 0 and off.trades.income.sum() == 0
    assert on.equity.iloc[-1] > off.equity.iloc[-1]
    books(on)
    books(off)


# ------------------------------------------------------------ 6. clearer errors

def test_zero_holding_period_and_zero_stop_messages():
    with pytest.raises(ParseError, match="holding period must be at least 1 day"):
        sig("buy SPY when RSI(2) is below 10, hold 0 days")
    assert sig("buy SPY at the open when RSI(2) is below 10, hold 0 days").hold_bars == 0
    with pytest.raises(ParseError, match="stop of 0%"):
        sig("buy SPY when RSI(2) is below 10, stop loss 0%")
    with pytest.raises(ParseError, match="take profit of 0%"):
        sig("buy SPY when RSI(2) is below 10, take profit 0%")


def test_two_trailing_stops_are_both_active():
    s = sig("buy SPY when RSI(2) is below 10, exit on a 10% trailing stop or a chandelier stop of 3 ATR")
    assert s.trailing_stop == pytest.approx(0.10) and s.trailing_atr == 3
    assert any(n.startswith("Two trailing stops") for n in s.notes)


# ------------------------------------------------------------ 7. limit / stop entry labels

def test_limit_fills_are_labelled_limit_and_gaps_open(fake):
    rows = [[100, 101, 99, 100], [100, 100.5, 97, 99],     # signal day 0; day 1: touches 98 intraday -> limit
            [99, 100, 98.5, 99.5], [99, 99.5, 98, 99], [99, 99, 98, 98.5],
            [95, 96, 94, 95], [95, 96, 94, 95], [95, 96, 94, 95], [95, 96, 94, 95]]   # day 5: gaps below -> open
    fake["LIM"] = bars(rows)
    s = Strategy(universe=["LIM"], entry="dow >= 0", entry_order="limit", entry_level="close * 0.98", hold_bars=1,
                 hold_exit_fill="close", cash_rate=None, start="2019-12-30")
    r = engine.run(s)
    fills = dict(zip(r.trades.entry_date.astype(str), r.trades.entry_fill))
    assert fills[str(bars(rows).index[1].date())] == "limit"
    assert "open" in fills.values() and set(fills.values()) <= {"limit", "open"}
    books(r)


@needs_data
def test_limit_entry_label_on_real_data():
    s = sig("buy AAPL at a limit 2% below the close when RSI(2) is below 10, hold 3 days, since 2016")
    r = engine.run(s)
    t = r.trades[r.trades.entry_date.astype(str) == "2016-11-01"]
    assert len(t) and (t.entry_fill == "limit").all()


# ------------------------------------------------------------ 8. phrases

@pytest.mark.parametrize("text,want", [
    ("buy SPY when the close is higher than 5 days ago, hold 3 days", "(close > ref(close, 5))"),
    ("buy SPY when it closes lower than 10 days ago, hold 3 days", "(close < ref(close, 10))"),
    ("buy SPY when yesterday's RSI(2) was below 10 and today RSI(2) is above 10, hold 3 days",
     "ref(((rsi(close, 2) < 10)), 1) and (rsi(close, 2) > 10)"),
    ("buy SPY when supertrend direction flips to up, hold 3 days",
     "(supertrend_dir(10, 3) < 0 and ref(supertrend_dir(10, 3), 1) > 0)"),
    ("buy SPY when price is above the Ichimoku cloud, hold 3 days", "(close > maximum(senkou_a(), senkou_b()))"),
    ("buy SPY when close > ta.sma(close, 200), hold 3 days", "(close > sma(close, 200))"),
    ("buy SPY when close > ta.sma(close, 200) and RSI(2) is below 10, hold 3 days", "(close > sma(close, 200)) and (rsi(close, 2) < 10)"),
    ("buy SPY 2 days after RSI(2) is below 10, hold 3 days", "ref(((rsi(close, 2) < 10)), 2)"),
])
def test_new_phrases(text, want):
    assert sig(text).entry == want


@pytest.mark.parametrize("text", ["buy SPY when RSI(2) crosses 70, hold 3 days", "buy SPY when it crosses the 50 SMA, hold 3 days"])
def test_crosses_without_a_direction_asks_which_way(text):
    with pytest.raises(ParseError, match="crosses which way.*crosses above.*crosses below"):
        sig(text)


def test_stop_and_target_phrases():
    s = sig("buy SPY when RSI(2) is below 10, stop at the low of the entry bar, take profit at 2 times the risk")
    assert s.stop_level == "low" and s.target_r == 2 and s.entry_fill == "close"
    assert any("entry bar's values" in n for n in s.notes)
    s = sig("buy SPY at the next open when RSI(2) is below 10, stop at the 5 day low, target 2R")
    assert s.stop_level == "lowest(low, 5)" and s.target_r == 2
    assert any("previous bar's values" in n for n in s.notes)
    s = sig("buy SPY when RSI(2) is below 10, sell half at +5% and trail the rest with an 8% trailing stop")
    assert s.trail_after_scale_out and s.trailing_stop == pytest.approx(0.08) and s.scale_out == [{"fraction": 0.5, "at": 0.05}]
    with pytest.raises(ParseError, match="by how much"):
        sig("buy SPY when RSI(2) is below 10, sell half at +5% and trail the rest")
    with pytest.raises(ParseError, match="use the high"):
        sig("short SPY when RSI(2) is above 90, stop at the low of the entry bar, hold 5 days")
    # an entry stop order is still an entry order
    s = sig("buy QQQ with a buy stop at yesterday's high when RSI(2) is below 10, hold 3 days")
    assert s.entry_order == "stop" and s.stop_level is None


# ------------------------------------------------------------ 9. stop / target levels in the engine

def level_strats(**kw):
    base = dict(universe=["RND"], entry="rsi(2) < 15", cash_rate=None, start="2019-12-30")
    return [Strategy(**base, stop_level="low", target_r=2, **kw),
            Strategy(**base, stop_level="lowest(low, 5)", target_level="entry_price + 3 * (entry_price - stop_price)", **kw),
            Strategy(**base, stop_level="entry_price - 2 * atr(14)", target_r=1.5, entry_fill="next_open", **kw),
            Strategy(**base, trailing_stop=0.04, scale_out=[{"at": 0.03, "fraction": 0.5}], trail_after_scale_out=True,
                     hold_bars=15, **kw),
            Strategy(**{**base, "sizing": "risk", "risk_per_trade": 0.01}, stop_level="low", hold_bars=10, **kw)]


def test_levels_keep_the_books_and_exit_at_the_levels(fake):
    fake["RND"] = rand_bars(500, seed=3)
    for s in level_strats():
        r = engine.run(s)
        assert len(r.trades) > 5
        books(r)
    r = engine.run(level_strats()[0])
    t = r.trades
    stopped = t[t.exit_reason == "stop level"]
    assert len(stopped) and len(t[t.exit_reason == "take profit"])
    df = fake["RND"]
    for _, x in stopped.iterrows():   # a close entry: the stop is that bar's low, filled there or at a gap open
        low = df.loc[pd.Timestamp(x.entry_date), "low"]
        assert x.stop_level == pytest.approx(low)
        assert x.exit_price <= low + 1e-9
    for _, x in t[t.exit_reason == "take profit"].iterrows():
        assert x.target_level == pytest.approx(x.entry_price + 2 * (x.entry_price - x.stop_level))


def test_levels_use_only_data_known_at_the_fill(fake):
    """Truncating the data at D changes no trade or equity value before D; a next-open entry's stop level is the
    signal bar's value (the entry bar's own low is not known at its open)."""
    full = rand_bars(500, seed=4)
    fake["RND"] = full
    cut = full.index[300]
    for mk in range(5):
        a = engine.run(level_strats()[mk])
        fake["RND"] = full.loc[:cut]
        b = engine.run(level_strats()[mk])
        fake["RND"] = full
        ta = a.trades[pd.to_datetime(a.trades.exit_date) < cut].reset_index(drop=True)
        tb = b.trades[pd.to_datetime(b.trades.exit_date) < cut].reset_index(drop=True)
        pd.testing.assert_frame_equal(ta.drop(columns=["cum_pnl"]), tb.drop(columns=["cum_pnl"]))
        ea, eb = a.equity.loc[: cut - pd.Timedelta(days=1)], b.equity.loc[: cut - pd.Timedelta(days=1)]
        pd.testing.assert_series_equal(ea, eb)
    r = engine.run(level_strats()[2])
    for _, x in r.trades.head(10).iterrows():
        i = full.index.get_loc(pd.Timestamp(x.entry_date))
        atr_prev = engine.expr.Namespace(full)["atr"](14).iloc[i - 1]
        assert x.stop_level == pytest.approx(x.entry_price - 2 * atr_prev)


def test_level_validation():
    with pytest.raises(ValueError, match="needs a stop"):
        Strategy(universe=["SPY"], entry="True", target_r=2).validate()
    with pytest.raises(ValueError, match="not known when the position opens"):
        Strategy(universe=["SPY"], entry="True", stop_level="low - bars_held").validate()
    with pytest.raises(ValueError, match="not known when the position opens"):
        Strategy(universe=["SPY"], entry="True", stop_level="stop_price").validate()


# ------------------------------------------------------------ 4. one survivorship figure per run

@pytest.mark.skipif(data.membership() is None, reason="no membership data")
def test_survivorship_note_and_headline_agree():
    from backtester import report
    s = sig("buy Nasdaq 100 stocks when RSI(2) is below 5, sell when RSI(2) is above 50, max 5 positions, since 2005")
    r = engine.run(s)
    note = next(n for n in s.notes if n.startswith("Survivorship:"))
    A = report.analyze(r, sensitivity=False)
    w = next(x for x in A["warnings"] if x["code"] == "survivorship")
    pct = f"{w['coverage']:.0%}"
    assert note.startswith(f"Survivorship: {pct} of index member-months")
    worst = w["message"].split("(")[1].split(")")[0]            # "49% in 2005"
    assert f"lowest {worst}" in note
    for t, k in data.coverage_figures(*data.coverage_window(note))["missing"][:2]:
        assert f"{t} ({k}" in note and f"{t} ({k} member-months)" in w["detail"]


# ------------------------------------------------------------ 2. options and share links (server side)

def test_site_options_cover_cash_dividends_and_commissions():
    o = web._options({"options": {"cash_rate": "none", "dividends": "false", "commission_per_share": "0.005",
                                  "commission_model": "ibkr_tiered", "tv_compat": "true"}})
    assert o == {"cash_rate": 0.0, "dividends": False, "commission_per_share": 0.005, "commission_model": "ibkr_tiered",
                 "tv_compat": True}
    assert web._options({"options": {"cash_rate": "2"}})["cash_rate"] == pytest.approx(0.02)
    with pytest.raises(web.ClientError):
        web._options({"options": {"cash_rate": "lots"}})


@needs_data
def test_share_token_keeps_no_interest_and_tv_mode():
    s = web._spec({"text": "buy SPY when rsi(2) < 10, sell when rsi(2) > 70, since 2018", "options": {"tv_compat": "true"}})
    tok = web.share_token(s)
    back = web._spec({"share": tok, "options": {"capital": "25000"}})
    assert back.tv_compat and back.cash_rate is None and back.dividends is False and back.capital == 25000
    assert back.entry_fill == "next_open"
    off = web._spec({"share": tok, "options": {"tv_compat": "false"}})
    assert not off.tv_compat and off.cash_rate == "tbill" and off.dividends is True


@needs_data
def test_build_form_spec_equals_the_parsed_spec():
    """The repro sentence: every field survives a trip through a JSON spec (what the Build page sends)."""
    text = ("buy NVDA at the next open when it closes above its 20 day high, trailing stop 8%, move the stop to "
            "breakeven after +5%, sell half at +10%, take profit 3 ATR, since 2018")
    a = web.api_parse({"text": text})["spec"]
    b = web.api_parse({"spec": {k: v for k, v in a.items() if k not in ("notes", "description")}})["spec"]
    assert {k: v for k, v in a.items() if k not in ("notes", "description")} == \
           {k: v for k, v in b.items() if k not in ("notes", "description")}
    assert "breakeven" in web.api_parse({"spec": a})["interpretation"]


# ------------------------------------------------------------ Playwright: Build round trip and share links

PORT = 8840


def _chromium():
    try:
        import playwright  # noqa: F401
    except ImportError:
        return None
    return (glob.glob("/opt/pw-browsers/chromium*/chrome-linux*/chrome") or [None])[0]


browser = pytest.mark.skipif(_chromium() is None or not {"SPY", "NVDA"} <= AVAIL, reason="headless Chromium or data missing")


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    old = web.RUNS
    web.RUNS = tmp_path_factory.mktemp("runs8")
    try:
        srv = web.ThreadingHTTPServer(("127.0.0.1", PORT), web.Handler)
    except OSError:
        if shutil.which("fuser"):
            subprocess.run(["fuser", "-k", f"{PORT}/tcp"], capture_output=True)
            time.sleep(1.0)
        srv = web.ThreadingHTTPServer(("127.0.0.1", PORT), web.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{PORT}"
    srv.shutdown()
    srv.server_close()
    web.RUNS = old


def _page(p, errs):
    b = p.chromium.launch(executable_path=_chromium())
    pg = b.new_page(viewport={"width": 1280, "height": 900})
    pg.on("pageerror", lambda e: errs.append(f"pageerror: {e}"))
    pg.set_default_timeout(180_000)
    return b, pg


@browser
def test_build_round_trips_every_field(site):
    from playwright.sync_api import sync_playwright
    text = ("buy NVDA at the next open when it closes above its 20 day high, trailing stop 8%, move the stop to "
            "breakeven after +5%, sell half at +10%, take profit 3 ATR, since 2018")
    errs: list[str] = []
    with sync_playwright() as p:
        b, pg = _page(p, errs)
        pg.goto(site + "/#backtest")
        pg.fill("#text", text)
        pg.wait_for_function("document.querySelector('#interp').textContent.includes('breakeven')")
        with pg.expect_response(lambda r: "/api/run" in r.url) as rr:
            pg.click("#runBtn")
        bt = rr.value.json()
        pg.click("#toBuild")
        pg.wait_for_selector("#bSignal:not(.hide)")
        pg.wait_for_function("document.querySelector('#buildInterp').textContent.includes('breakeven')")
        assert pg.input_value("#s_breakeven_after") == "5" and pg.input_value("#s_scale_out") == "50@10"
        assert pg.input_value("#s_take_profit_atr") == "3"
        time.sleep(1.0)   # the guard's normalised copy of the loaded spec
        with pg.expect_response(lambda r: "/api/run" in r.url) as rr:
            pg.click("#buildRun")
        bd = rr.value.json()
        pg.wait_for_function("document.querySelector('#buildStatus').textContent.includes('CAGR')")
        skip = {"notes", "description"}
        assert {k: v for k, v in bd["spec"].items() if k not in skip} == {k: v for k, v in bt["spec"].items() if k not in skip}
        assert bd["summary"]["cagr"] == pytest.approx(bt["summary"]["cagr"], abs=1e-12)
        assert "breakeven" in pg.text_content("#buildInterp") and "sell 50% at +10.0%" in pg.text_content("#buildInterp")
        # an advanced field (no control) is carried and editable
        adv = pg.input_value("#s_advanced")
        assert '"min_order"' in adv and '"reverse"' in adv
        # switching modes refreshes the interpretation
        pg.click("#buildMode button[data-v='portfolio']")
        pg.wait_for_function("!document.querySelector('#buildInterp').textContent.includes('breakeven')")
        assert not errs, errs
        b.close()


@browser
def test_build_guard_refuses_a_lossy_run(site):
    from playwright.sync_api import sync_playwright
    errs: list[str] = []
    with sync_playwright() as p:
        b, pg = _page(p, errs)
        pg.goto(site + "/#build")
        pg.wait_for_selector("#p-build:not(.hide)")
        pg.wait_for_load_state("networkidle")
        pg.evaluate("""() => { document.querySelector('#specJson').value = JSON.stringify({universe: ['SPY'], entry: 'rsi(2) < 10',
            hold_bars: 3, exit_when_fill: 'close', start: '2020-01-01'}); document.querySelector('#fromJson').click(); }""")
        pg.wait_for_selector("#bSignal:not(.hide)", timeout=20_000)
        time.sleep(1.5)
        # simulate a control that can't hold the value: the select loses it without an edit event
        pg.evaluate("() => { const e = document.querySelector('#s_hold_exit_fill'); e.value = 'open'; }")
        pg.click("#buildRun")
        pg.wait_for_selector("#buildLost")
        assert "hold_exit_fill" in pg.text_content("#buildLost")
        assert not errs, errs
        b.close()


@browser
def test_portfolio_editor_round_trips_every_field(site):
    from playwright.sync_api import sync_playwright
    spec = {"kind": "allocation", "name": "sixty forty", "tree": {"weights": "specified", "w": [0.6, 0.4],
            "children": [{"asset": "SPY"}, {"asset": "TLT"}]}, "rebalance": "quarterly", "start": "2012-01-01",
            "cash_rate": None, "reinvest_dividends": False, "contribution": 100, "contribution_start": 2014,
            "contribution_end": 2018, "commission": 1.0, "warmup": "first", "maintenance_margin": 0.3,
            "drift_band_relative": 0.25, "min_trade": 0.001}
    errs: list[str] = []
    with sync_playwright() as p:
        b, pg = _page(p, errs)
        pg.goto(site + "/#build")
        pg.wait_for_selector("#p-build:not(.hide)")
        pg.wait_for_load_state("networkidle")
        pg.evaluate("s => { document.querySelector('#specJson').value = JSON.stringify(s); document.querySelector('#fromJson').click(); }", spec)
        time.sleep(1.5)
        adv = pg.input_value("#p_advanced")
        for k in ("cash_rate", "reinvest_dividends", "contribution_start", "warmup", "drift_band_relative", "min_trade"):
            assert f'"{k}"' in adv
        with pg.expect_response(lambda r: "/api/run" in r.url) as rr:
            pg.click("#buildRun")
        run = rr.value.json()["spec"]
        for k, v in spec.items():
            if k != "tree":
                assert run[k] == v, k
        assert not errs, errs
        b.close()


@browser
def test_share_link_restores_tv_mode_and_every_setting(site):
    from playwright.sync_api import sync_playwright
    errs: list[str] = []
    with sync_playwright() as p:
        b, pg = _page(p, errs)
        pg.goto(site + "/#backtest")
        pg.wait_for_load_state("networkidle")
        pg.evaluate("() => document.querySelectorAll('#p-backtest details').forEach(d => d.open = true)")
        pg.select_option("#o_tv_compat", "true")
        pg.fill("#text", "buy SPY when rsi(2) < 10, sell when rsi(2) > 70, since 2015")
        pg.wait_for_function("document.querySelector('#interp').textContent.includes('next day')")
        with pg.expect_response(lambda r: "/api/run" in r.url) as rr:
            pg.click("#runBtn")
        first = rr.value.json()
        assert first["spec"]["tv_compat"] and first["spec"]["cash_rate"] is None
        link = f"{site}/#backtest?s={first['share']}"
        pg2 = b.new_page()
        pg2.on("pageerror", lambda e: errs.append(f"pageerror: {e}"))
        pg2.set_default_timeout(180_000)
        with pg2.expect_response(lambda r: "/api/run" in r.url):
            pg2.goto(link)
        pg2.evaluate("() => document.querySelectorAll('#p-backtest details').forEach(d => d.open = true)")
        assert pg2.input_value("#o_tv_compat") == "true"
        assert pg2.input_value("#o_cash_rate") == "none"
        pg2.fill("#o_capital", "25000")
        pg2.dispatch_event("#o_capital", "change")
        with pg2.expect_response(lambda r: "/api/run" in r.url) as rr:
            pg2.click("#runBtn")
        again = rr.value.json()
        assert again["spec"]["tv_compat"] and again["spec"]["capital"] == 25000
        assert again["spec"]["entry_fill"] == "next_open" and again["spec"]["cash_rate"] in (None, 0)   # no interest
        assert again["interpretation"].split("Sizing:")[0] == first["interpretation"].split("Sizing:")[0]
        assert not errs, errs
        b.close()
