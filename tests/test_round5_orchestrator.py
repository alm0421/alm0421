import pandas as pd
import pytest

from backtester import data, engine, parser, runner

HAVE = bool(data.available_tickers())
needs_data = pytest.mark.skipif(not HAVE, reason="price data not downloaded")


@needs_data
def test_no_index_members_before_the_first_snapshot():
    idx = data.load("AAPL").loc["1999-01":"2004-02"].index
    m, first = data.member_mask(["SSCC", "CECO", "AAPL", "MSFT"], idx)
    assert not m.any()
    assert first == data.membership().index[0]


@needs_data
def test_ndx_runs_start_at_the_membership_start():
    s = parser.parse("buy Nasdaq 100 stocks when RSI(2) is below 5, hold 3 days, max 5 positions, since 1995")
    s.end = "2004-12-31"
    r = runner.run(s)
    assert r.equity.index[1] >= data.membership().index[0]
    assert r.trades.empty or not r.trades["ticker"].isin(["SSCC", "CECO"]).any()
    with pytest.raises(ValueError, match="only known from"):
        s2 = parser.parse("buy Nasdaq 100 stocks when RSI(2) is below 5, hold 3 days, max 5 positions")
        s2.start, s2.end = "1997-01-01", "2003-12-31"
        runner.run(s2)


@needs_data
def test_signal_scan_reports_next_open_entry_on_the_right_day(monkeypatch):
    from backtester import signals
    real = data.load

    def cut(t):
        return real(t).loc[:"2026-09-10"]
    monkeypatch.setattr(data, "load", cut)
    spec = parser.parse("buy QQQ at the open when RSI(2) is below 10, sell at the close")
    rsi_true = bool((__import__("backtester.expr", fromlist=["x"]).evaluate(
        "rsi(close, 2) < 10", __import__("backtester.expr", fromlist=["x"]).Namespace(cut("QQQ")))).iloc[-1])
    out = signals.scan(spec)
    assert (len(out["entry_signals"]) == 1) == rsi_true


@needs_data
def test_monte_carlo_uses_the_specs_balance_and_fees():
    from backtester import montecarlo as mc
    spec = parser.parse("hold 60% SPY and 40% AGG, withdraw 4% per year adjusted for inflation, starting with $250,000")
    s = mc.Settings(years=5, sims=200)
    s.weights, _ = mc.settings_from_spec(spec, s)
    assert s.start_balance == 250_000
    fee = parser.parse("hold 60% SPY and 40% AGG with a 2% expense ratio")
    s2 = mc.Settings(years=5, sims=200)
    s2.weights, _ = mc.settings_from_spec(fee, s2)
    assert abs(s2.expense_ratio - 0.02) < 1e-12
