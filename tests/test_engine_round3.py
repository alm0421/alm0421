"""Round 3 engine and report fixes: the interpretation equals what runs, holding periods, TradingView pyramiding,
closed-trade statistics, wiped-out accounts, volume caps at the open, lazily loaded charts, long-history
benchmarks, allocation tables and indicator warm-up."""
import glob
import json
import warnings

import numpy as np
import pandas as pd
import pytest

from backtester import data, engine, metrics, parser, report, research, research_report
from backtester.strategy import Strategy

HAVE = {"MSFT", "QQQ", "SPY", "SPYSIM", "TLTSIM"} <= set(data.available_tickers())
needs_data = pytest.mark.skipif(not HAVE, reason="price data not downloaded")


def bars(rows, start="2019-12-30", volume=1e6):
    """rows: (open, high, low, close); the default start is a Monday."""
    idx = pd.bdate_range(start, periods=len(rows))
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx, dtype=float)
    df["volume"] = volume
    df["quote_close"] = df["close"]
    df["adj_close"] = df["close"]
    df["dividend"] = 0.0
    df["split"] = 0.0
    return df


@pytest.fixture
def fake(monkeypatch):
    frames = {}
    real = data.load

    def load(t):
        if t in frames:
            return frames[t]
        if t in ("SPY", "QQQ", "SPYSIM"):  # benchmarks: not part of these synthetic tests
            raise FileNotFoundError(t)
        return real(t)

    monkeypatch.setattr(data, "load", load)
    monkeypatch.setattr(data, "load_many", lambda ts: {t: frames[t] for t in ts})
    return frames


def flat(n, px=100.0):
    return [(px, px, px, px)] * n


def check_books(r, capital=10_000):
    assert r.equity.iloc[-1] == pytest.approx(capital + r.trades.pnl.sum() + r.interest)


# ------------------------------------------------------------ 1. the interpretation equals what runs

def test_position_size_above_leverage_raises_leverage_and_says_so(fake):
    fake["X"] = bars(flat(3) + [(100, 100, 100, 110)] * 3)
    s = Strategy(cash_rate=None, universe=["X"], entry="dow == 0", hold_bars=1, position_size=2.0)
    r = engine.run(s)
    assert s.leverage == 2.0
    assert any(n.startswith("Leverage: 200% per position needs 2x leverage: using 2x") for n in s.notes)
    assert "raised automatically" in s.summary()
    t = r.trades.iloc[0]
    assert t.position_value == pytest.approx(20_000)  # not silently capped at 100%
    check_books(r)
    s.validate()  # idempotent: one note
    assert sum(n.startswith("Leverage:") for n in s.notes) == 1


def test_impossible_stops_and_capital_are_refused():
    with pytest.raises(ValueError, match="can never trigger"):
        Strategy(universe=["X"], entry="True", stop_loss=1.5).validate()
    with pytest.raises(ValueError, match="can never trigger"):
        Strategy(universe=["X"], entry="True", trailing_stop=1.0).validate()
    Strategy(universe=["X"], entry="True", side="short", stop_loss=1.5).validate()  # a short can lose 150%
    for cap in (0, -5_000):
        with pytest.raises(ValueError, match="starting capital must be positive"):
            Strategy(universe=["X"], entry="True", hold_bars=1, capital=cap).validate()


# ------------------------------------------------------------ 2. hold N = N bars after the entry bar

@pytest.mark.parametrize("fill, rule", [("close", "dow == 0"), ("next_open", "dow == 0"), ("open", "dow == 0"),
                                        ("next_close", "dow == 0")])
@pytest.mark.parametrize("hold", [1, 3])
def test_hold_exits_n_bars_after_the_entry_bar(fake, fill, rule, hold):
    fake["X"] = bars([(100 + i, 101 + i, 99 + i, 100.5 + i) for i in range(10)])
    s = Strategy(cash_rate=None, universe=["X"], entry=rule, entry_fill=fill, hold_bars=hold)
    r = engine.run(s)
    t = r.trades.iloc[0]
    idx = fake["X"].index
    entry_bar = idx.get_loc(pd.Timestamp(t.entry_date))
    assert t.bars_held == hold
    assert pd.Timestamp(t.exit_date) == idx[entry_bar + hold] and t.exit_price == pytest.approx(100.5 + entry_bar + hold)
    assert f"at the close {hold} bar{'s' if hold > 1 else ''} after the entry bar" in s.summary()
    check_books(r)


def test_hold_zero_is_the_close_of_an_entry_bar_entered_at_the_open(fake):
    fake["X"] = bars([(100 + i, 101 + i, 99 + i, 100.5 + i) for i in range(6)])
    r = engine.run(Strategy(cash_rate=None, universe=["X"], entry="dow == 0", entry_fill="open", hold_bars=0))
    t = r.trades.iloc[0]
    assert t.bars_held == 0 and t.entry_price == 100 and t.exit_price == 100.5 and t.entry_date == t.exit_date
    with pytest.raises(ValueError, match="hold_bars 0"):
        Strategy(universe=["X"], entry="dow == 0", hold_bars=0).validate()  # a close entry can't exit at that close
    # "cover at the close" with no holding period after an open entry is that same close
    assert parser.parse("short QQQ at the open when it gaps up 1%, cover at the close").hold_bars == 0
    assert parser.parse("buy AAPL at the next open when RSI(2) < 10, hold 3 days").hold_bars == 3


@needs_data
def test_msft_reference_still_sells_at_the_next_close():
    s = parser.parse("buy at the close Microsoft when it trades down 5 days in a row, hold for 1 day, and sell at the close")
    r = engine.run(s)
    c = data.quoted_close("MSFT")        # trade prices are as traded (MSFT has split nine times since 1987)
    t = r.trades.iloc[3]
    nxt = c.index[c.index.get_loc(pd.Timestamp(t.entry_date)) + 1]
    assert (r.trades.bars_held == 1).all() and pd.Timestamp(t.exit_date) == nxt and t.exit_price == pytest.approx(c[nxt])


# ------------------------------------------------------------ 3. TradingView pyramiding: no stale entries

def test_signal_while_in_position_is_not_filled_after_a_stop_at_the_open(fake):
    # the entry fires every day; the position gaps through its 5% stop at the open of bar 3
    rows = flat(3) + [(90, 91, 89, 90)] + flat(4, 90)
    fake["X"] = bars(rows)
    s = Strategy(cash_rate=None, universe=["X"], entry="close > 0", entry_fill="next_open", stop_loss=0.05, hold_bars=50)
    r = engine.run(s)
    idx = fake["X"].index
    first, second = r.trades.iloc[0], r.trades.iloc[1]
    assert pd.Timestamp(first.entry_date) == idx[1] and first.exit_reason == "stop loss" and pd.Timestamp(first.exit_date) == idx[3]
    # the bar-2 signal (placed while holding) is dropped; the bar-3 signal (flat after the stop) fills at bar 4's open
    assert pd.Timestamp(second.entry_date) == idx[4]
    check_books(r)


def test_signal_on_a_close_exit_bar_is_valid(fake):
    fake["X"] = bars(flat(8))
    s = Strategy(cash_rate=None, universe=["X"], entry="close > 0", entry_fill="next_open", hold_bars=1)
    r = engine.run(s)
    idx = fake["X"].index
    # in at bar 1's open, out at bar 2's close; bar 2's signal (flat after that close) enters at bar 3's open
    assert [idx.get_loc(pd.Timestamp(d)) for d in r.trades.entry_date] == [1, 3, 5, 7]
    assert [idx.get_loc(pd.Timestamp(d)) for d in r.trades.exit_date][:2] == [2, 4]


def test_pyramiding_still_adds_while_below_the_limit(fake):
    fake["X"] = bars(flat(6))
    r = engine.run(Strategy(cash_rate=None, universe=["X"], entry="close > 0", entry_fill="next_open", hold_bars=10,
                            pyramiding=2, position_size=0.4))
    assert len(r.trades) == 2 and r.trades.entry_date.nunique() == 2


# ------------------------------------------------------------ 4. closed-trade statistics and open P&L

def test_trade_stats_use_closed_trades_and_report_open_pnl(fake, tmp_path):
    rows = flat(2) + [(100, 100, 100, 110)] * 3 + [(110, 110, 90, 90), (90, 90, 80, 80)]
    fake["X"] = bars(rows)
    s = Strategy(cash_rate=None, universe=["X"], entry="dow == 0", hold_bars=2)
    r = engine.run(s)
    closed, opened = metrics.split_open(r.trades)
    assert len(closed) == 1 and len(opened) == 1 and opened.iloc[0].exit_reason == "open at end"
    ts = metrics.trade_stats(r.trades, 1.0)
    assert ts["trades"] == 1 and ts["win_rate"] == 1.0 and ts["total_pnl"] == pytest.approx(closed.pnl.sum())
    assert ts["open_trades"] == 1 and ts["open_pnl"] == pytest.approx(opened.pnl.sum()) and ts["open_pnl"] < 0
    A = report.analyze(r, rf=0.0, sensitivity=False, mc=False)
    assert "Open P&L" in report.console_summary(A)
    report.write_outputs(A, tmp_path, excel=False)
    summ = json.loads((tmp_path / "summary.json").read_text())
    assert summ["open_pnl"] == pytest.approx(opened.pnl.sum()) and summ["trade_stats"]["trades"] == 1
    check_books(r)


# ------------------------------------------------------------ 5. wiped out: CAGR -100%, no warnings

def test_negative_final_equity_is_minus_100_percent_without_warnings(fake):
    fake["X"] = bars(flat(40) + flat(41, 400))
    s = Strategy(cash_rate=None, universe=["X"], entry="dow == 0", side="short", hold_bars=60, leverage=3, margin_account="portfolio",
                 maintenance_margin=0)
    r = engine.run(s)
    assert r.equity.iloc[-1] < 0
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        A = report.analyze(r, rf=0.0, sensitivity=False)
        report.console_summary(A)
    assert A["stats"]["cagr"] == -1.0 and A["stats"]["wiped_out"]
    assert A["stats"]["max_drawdown"] == pytest.approx(-1.0)
    assert "wiped_out" in [w["code"] for w in A["warnings"]]


# ------------------------------------------------------------ 6. volume caps at the open use yesterday's volume

def test_volume_cap_at_the_open_uses_the_previous_bars_volume(fake):
    df = bars(flat(5, 10.0))
    df["volume"] = [1e6, 1_000, 1e6, 1e6, 1e6]     # Mon, Tue (thin), Wed ...
    fake["X"] = df
    # Tuesday's signal fills at Wednesday's open: only Tuesday's 1,000 shares are known then
    r = engine.run(Strategy(cash_rate=None, universe=["X"], entry="dow == 1", entry_fill="next_open", hold_bars=1,
                            max_volume_pct=0.10))
    assert r.trades.iloc[0].shares == pytest.approx(100)
    # at the close the day's own volume is known
    r = engine.run(Strategy(cash_rate=None, universe=["X"], entry="dow == 1", hold_bars=1, max_volume_pct=0.10))
    assert r.trades.iloc[0].shares == pytest.approx(100)
    r = engine.run(Strategy(cash_rate=None, universe=["X"], entry="dow == 2", hold_bars=1, max_volume_pct=0.10))
    assert r.trades.iloc[0].shares == pytest.approx(1_000)   # 10,000 / $10: the 100,000-share cap doesn't bind


# ------------------------------------------------------------ 7. every traded ticker can be charted

def _multi(fake, n=4):
    rng = np.random.default_rng(3)
    for k in range(n):
        c = 50 * np.exp(np.cumsum(rng.normal(0, 0.02, 300)))
        fake[f"T{k}"] = bars([(x * 0.999, x * 1.01, x * 0.99, x) for x in c])
    return engine.run(Strategy(cash_rate=None, universe=[f"T{k}" for k in range(n)], entry="rsi(close, 2) < 15",
                               exit_when="rsi(close, 2) > 70", max_positions=4))


def test_charts_beyond_the_embedded_ones_are_written_next_to_the_report(fake, tmp_path, monkeypatch):
    r = _multi(fake)
    monkeypatch.setattr(report, "MAX_EMBED_TICKERS", 1)
    A = report.analyze(r, rf=0.0, sensitivity=False, mc=False)
    report.write_outputs(A, tmp_path, excel=False)
    html = (tmp_path / "report.html").read_text()
    D = json.loads(html.split('<script id="data" type="application/json">')[1].split("</script>")[0].replace("<\\/", "</"))
    run = D["runs"][0]
    assert len(run["chart_tickers"]) == 4 and len(run["prices"]) == 1
    lazy = run["chart_files"]
    assert sorted(lazy) == sorted(t for t in run["chart_tickers"] if t not in run["prices"])
    for t, f in lazy.items():
        text = (tmp_path / f).read_text()
        pre = f'(window.__charts=window.__charts||{{}})["{f}"]='
        assert text.startswith(pre)
        P = json.loads(text[len(pre):].rstrip().rstrip(";"))
        full = report.ticker_chart(r, t)
        dates = pd.Timestamp(P["d0"]) + pd.to_timedelta(np.concatenate([[0], np.cumsum(P["dd"])]), unit="D")
        assert [d.strftime("%Y-%m-%d") for d in dates] == full["dates"] and P["c"] == full["c"]
    # a small run stays self-contained: no chart files
    A1 = report.analyze(engine.run(Strategy(cash_rate=None, universe=["T0"], entry="rsi(close, 2) < 15", hold_bars=2)),
                        rf=0.0, sensitivity=False, mc=False)
    out1 = tmp_path / "one"
    report.write_outputs(A1, out1, excel=False)
    assert not (out1 / "charts").exists() or not list((out1 / "charts").iterdir())


def _chromium():
    try:
        import playwright  # noqa: F401
    except ImportError:
        return None
    return (glob.glob("/opt/pw-browsers/chromium*/chrome-linux*/chrome") or [None])[0]


@pytest.mark.skipif(_chromium() is None, reason="headless Chromium not available")
def test_lazy_charts_load_over_http_and_from_file(fake, tmp_path, monkeypatch):
    import http.server
    import threading
    from functools import partial

    from playwright.sync_api import sync_playwright
    r = _multi(fake)
    monkeypatch.setattr(report, "MAX_EMBED_TICKERS", 1)
    report.write_outputs(report.analyze(r, rf=0.0, sensitivity=False, mc=False), tmp_path, excel=False)
    h = partial(http.server.SimpleHTTPRequestHandler, directory=str(tmp_path))
    h.log_message = lambda *a, **k: None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), h)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    lazy = sorted(t for t in report.chart_tickers(r) if t != report.chart_tickers(r)[0])
    try:
        with sync_playwright() as p:
            b = p.chromium.launch(executable_path=_chromium())
            for url in (f"http://127.0.0.1:{srv.server_address[1]}/report.html", (tmp_path / "report.html").as_uri()):
                pg = b.new_page()
                errs = []
                pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
                pg.on("pageerror", lambda e: errs.append(str(e)))
                pg.goto(url)
                pg.select_option("#pxTicker", lazy[0])
                pg.wait_for_function("t => window.__charts && Object.keys(window.__charts).some(k => k.endsWith(t + '.js'))"
                                     " && !/Loading/.test(document.getElementById('pxLegend').textContent)", arg=lazy[0], timeout=10_000)
                assert "rsi(close, 2)" in pg.inner_text("#pxLegend")
                # clicking a trade of another lazily loaded ticker loads and shows it
                pg.fill("#tradeFilter", lazy[1])
                pg.click("#trades tbody tr.click >> nth=0")
                pg.wait_for_function("t => document.getElementById('pxTicker').value === t && "
                                     "!/Loading/.test(document.getElementById('pxLegend').textContent)", arg=lazy[1], timeout=10_000)
                assert pg.evaluate("t => Object.keys(window.__charts).some(k => k.endsWith(t + '.js'))", lazy[1])
                assert errs == []
                pg.close()
            b.close()
    finally:
        srv.shutdown()


# ------------------------------------------------------------ 8. benchmarks and allocation reports

def test_late_benchmark_starts_with_the_portfolios_balance():
    idx = pd.bdate_range("2000-01-03", periods=300)
    equity = pd.Series(np.linspace(1_000_000, 2_000_000, 300), index=idx)
    flows = pd.Series(0.0, index=idx)
    flows.iloc[[20, 120, 220]] = -50_000
    bench = pd.Series(np.linspace(10_000, 12_000, 200), index=idx[100:])
    out = report.benchmarks_with_flows({"B": bench}, flows, equity)["B"]
    assert out.index[0] == idx[100] and out.iloc[0] == pytest.approx(equity.iloc[100])
    # only the flows after its first day, unchanged
    cs = metrics.cashflow_stats(out, flows[flows.index > idx[100]])
    assert cs["total_withdrawals"] == pytest.approx(100_000) and not cs["ran_out"]


def test_yearly_balances_follow_the_account():
    idx = pd.bdate_range("2019-12-31", "2021-12-31")
    eq = pd.Series(1_000.0, index=idx)
    fl = pd.Series(0.0, index=idx)
    fl[pd.Timestamp("2020-06-01")] = 500.0
    eq[eq.index >= "2020-06-01"] = 1_500.0
    fl[pd.Timestamp("2021-03-01")] = -200.0
    eq[eq.index >= "2021-03-01"] = 1_300.0
    y = metrics.yearly_balances(eq, metrics.nav(eq, fl), fl)
    assert y.loc[2020, "contributions"] == 500 and y.loc[2020, "end_balance"] == 1_500
    assert y.loc[2021, "start_balance"] == 1_500 and y.loc[2021, "withdrawals"] == 200 and y.loc[2021, "end_balance"] == 1_300


@needs_data
def test_long_history_portfolio_benchmarks():
    from backtester import runner
    spec = parser.parse("hold 60% SPYSIM and 40% TLTSIM, rebalance yearly, withdraw 4% per year adjusted for inflation, "
                        "starting with $1,000,000, since 1972")
    res = runner.run(spec)
    A = report.analyze(res, sensitivity=False, mc=False)
    assert A["primary_benchmark"] == "SPYSIM buy & hold" and A["relative"]["period_start"].year == 1972
    assert A["benchmark_from"]["SPY buy & hold"] == "1993-01-29"
    spy = A["benchmark_cash"]["SPY buy & hold"]
    assert str(spy["from"]) == "1993-01-29"
    # the strategy runs out of money in the 2010s (2013 with the fee-drag-adjusted series; the exact year moves with
    # the data); the benchmarks keep receiving the scheduled withdrawals (each capped at its own balance) instead of
    # stopping when the strategy's money ran out
    assert A["cash"]["depleted_on"] is not None and 2008 <= A["cash"]["depleted_on"].year <= 2020
    assert (spy["depleted_on"] is not None) == spy["ran_out"]
    assert A["benchmark_cash"]["SPYSIM buy & hold"]["total_withdrawals"] > A["cash"]["total_withdrawals"]
    assert spy["starting_balance"] == pytest.approx(res.equity[pd.Timestamp("1993-01-29")])
    C = report.common_window_stats([A], "tbill")
    assert str(C["full_start"]).startswith("1972") and C["full_from"]["SPY buy & hold"] == "1993-01-29"
    # the common period is the strategies' (a young default benchmark such as QQQ doesn't cut it short); a
    # benchmark that starts later is labelled with its own first date
    assert "SPY buy & hold" in C["full"] and str(C["start"]).startswith("1972")
    assert C["benchmarks_own_from"]["QQQ buy & hold"].startswith("1999") and "SPYSIM buy & hold" in C["columns"]
    assert "QQQ buy & hold" not in C["columns"] and "SPY buy & hold" not in C["columns"]
    # allocation tables: balances, not trade statistics
    y = A["yearly"]
    assert {"start_balance", "withdrawals", "end_balance", "inflation", "real_return"} <= set(y.columns)
    # everything is sold when the money runs out: both holding periods are closed
    assert A["trade_stats"]["trades"] == 2 and not A["trade_stats"].get("open_trades")
    text = report.console_summary(A)
    assert "Win rate" not in text and "win rate" not in text
    # crises before a benchmark's inception are blank only for that benchmark
    row = next(c for c in A["crises"] if c["event"] == "1987 crash")
    assert row["SPY buy & hold"] is None and row["SPYSIM buy & hold"] is not None


@needs_data
def test_indicator_warm_up_starts_the_stats_later():
    from backtester import runner
    res = runner.run(parser.parse("hold QQQ when it is above its 10-month moving average, otherwise cash, rebalance monthly"))
    A = report.analyze(res, sensitivity=False, mc=False, detail=False)
    # the portfolio itself starts trading after the warm-up, with the starting capital (nothing to trim)
    assert A["warmup_start"] is None and res.equity.index[1] > data.load("QQQ").index[200]
    assert A["stats"]["start_equity"] == pytest.approx(10_000)
    assert any(n.startswith("Warm-up: the portfolio starts on") for n in res.strategy.notes)


def test_signal_warm_up_skips_idle_bars_before_the_rule_is_defined(fake):
    rng = np.random.default_rng(5)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 260)))
    fake["X"] = bars([(x, x * 1.01, x * 0.99, x) for x in c])
    r = engine.run(Strategy(cash_rate=None, universe=["X"], entry="close > sma(close, 100)", hold_bars=2))
    A = report.analyze(r, rf=0.0, sensitivity=False, mc=False, detail=False)
    assert A["warmup_start"] == fake["X"].index[99]
    assert A["result"].equity.index[1] == fake["X"].index[99] and A["result_full"] is r
    check_books(r)


def test_expense_ratio_is_listed_as_a_cost():
    from backtester.portfolio import Portfolio
    p = Portfolio(tree={"asset": "SPY"}, expense_ratio=0.0059)
    assert "0.59%/yr expense ratio" in report.interpretation(p)


# ------------------------------------------------------------ 9. minor

def test_short_rebate_line_only_when_cash_earns_interest():
    s = Strategy(universe=["X"], entry="True", side="short", hold_bars=1, cash_rate=None)
    assert "short proceeds" not in s.summary()
    assert "short proceeds" in Strategy(universe=["X"], entry="True", side="short", hold_bars=1).summary()


def test_sweep_labels_drop_articles_and_one_luck_threshold():
    labels, _, _ = research.expand("buy the {5,10,20} Nasdaq 100 stocks with the lowest RSI(2) each day, hold 3 days")
    assert labels == ["buy {5,10,20}"]
    note = research_report.multiple_testing_note({"n_trials": 20, "best_sharpe": 1.2, "best_params": [5], "sd_sharpe": 0.3,
                                                  "expected_max_sharpe": 0.55, "expected_max_sharpe_simple": 0.73, "dsr": 0.9})
    assert "0.55" in note and "0.73" not in note


@needs_data
@pytest.mark.parametrize("text", [
    "buy QQQ at the next open when RSI(2) is below 10, hold 3 days, with a 3% stop loss",
    "buy QQQ at the open when it gaps down 1%, sell at the close, cap at 1% of volume",
])
def test_round3_engine_changes_do_not_depend_on_future_data(monkeypatch, text):
    from tests.test_backtester import test_signal_trades_do_not_depend_on_future_data
    test_signal_trades_do_not_depend_on_future_data(monkeypatch, text)
