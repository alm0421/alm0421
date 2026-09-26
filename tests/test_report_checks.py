"""Result sanity warnings, oscillator panes in the price chart, sweep labels and the deflated Sharpe."""
import json
import math

import numpy as np
import pandas as pd
import pytest

from backtester import data, engine, metrics, report, research
from backtester.strategy import Strategy

HAVE = {"SPY", "QQQ"} <= set(data.available_tickers())
needs_data = pytest.mark.skipif(not HAVE, reason="price data not downloaded")


def stats(**kw):
    base = {"sharpe": 0.8, "cagr": 0.08, "max_drawdown": -0.2, "start_equity": 10_000.0, "end_equity": 20_000.0,
            "years": 10.0}
    base.update(kw)
    return base


def trades(n, **kw):
    base = {"trades": n, "win_rate": 0.55, "profit_factor": 1.4}
    base.update(kw)
    return base


def codes(ws):
    return [w["code"] for w in ws]


# ------------------------------------------------------------ warnings

def test_no_trades_warning_and_suppression():
    ws = metrics.result_warnings("signal", stats(sharpe=5.0, cagr=0.02), {"trades": 0}, interest=2000.0)
    assert codes(ws) == ["no_trades"]  # no "too good" or interest note on an idle-cash curve
    assert ws[0]["message"] == "No trades" and ws[0]["level"] == "error"
    s = metrics.suppress_degenerate({"sharpe": 1.0, "calmar": 2.0, "cagr": 0.03, "sortino": 1.0})
    assert math.isnan(s["sharpe"]) and math.isnan(s["calmar"]) and math.isnan(s["sortino"]) and s["cagr"] == 0.03


@pytest.mark.parametrize("n,expect", [(1, True), (9, True), (10, False), (200, False)])
def test_few_trades_warning(n, expect):
    ws = metrics.result_warnings("signal", stats(), trades(n))
    assert ("few_trades" in codes(ws)) is expect
    if expect:
        assert "Too few trades" in ws[0]["message"]


def test_allocations_are_not_judged_by_trade_count():
    assert metrics.result_warnings("allocation", stats(), {"trades": 1}) == []
    assert metrics.result_warnings("allocation", stats(), {"trades": 0}) == []


@pytest.mark.parametrize("st,ts,fragment", [
    (stats(sharpe=3.4), trades(40), "Sharpe ratio 3.40"),
    (stats(cagr=1.5), trades(40), "CAGR 150%"),
    (stats(), trades(150, win_rate=0.93), "win rate 93.0%"),
    (stats(max_drawdown=0.0), trades(12), "never had a drawdown"),
    (stats(), trades(60, profit_factor=6.0), "profit factor 6.0"),
    (stats(), trades(60, profit_factor=float("inf")), "infinite"),
])
def test_too_good_alarms(st, ts, fragment):
    ws = metrics.result_warnings("signal", st, ts)
    tg = [w for w in ws if w["code"] == "too_good"]
    assert len(tg) == 1 and "too good" in tg[0]["message"]
    assert any(fragment in r for r in tg[0]["rules"]) and fragment in tg[0]["detail"]


@pytest.mark.parametrize("st,ts", [
    (stats(sharpe=2.9, cagr=0.99), trades(40)),
    (stats(), trades(100, win_rate=0.95)),        # needs more than 100 trades
    (stats(), trades(50, profit_factor=9.0)),     # needs more than 50 trades
    (stats(max_drawdown=-0.001), trades(40)),
])
def test_too_good_thresholds_not_triggered(st, ts):
    assert "too_good" not in codes(metrics.result_warnings("signal", st, ts))


def test_too_good_lists_every_rule_that_fired():
    ws = metrics.result_warnings("signal", stats(sharpe=4, cagr=2.0, max_drawdown=0.0), trades(120, win_rate=1.0,
                                                                                               profit_factor=float("inf")))
    tg = next(w for w in ws if w["code"] == "too_good")
    assert len(tg["rules"]) == 5


def test_interest_share_note():
    # $10k -> $20k over 10 years, $4k of it interest: 40% of profit
    ws = metrics.result_warnings("signal", stats(cagr=0.0718), trades(40), interest=4000.0)
    w = next(w for w in ws if w["code"] == "interest_share")
    assert w["level"] == "info" and "40%" in w["message"]
    assert w["cagr_ex_interest"] == pytest.approx((16_000 / 10_000) ** 0.1 - 1)
    assert "approximation" in w["detail"]
    # below the 25% threshold, with cash flows, or for allocations: no note
    assert "interest_share" not in codes(metrics.result_warnings("signal", stats(), trades(40), interest=2000.0))
    assert "interest_share" not in codes(metrics.result_warnings("signal", stats(), trades(40), interest=4000.0, has_flows=True))
    assert "interest_share" not in codes(metrics.result_warnings("allocation", stats(), trades(40), interest=4000.0))
    # interest larger than the whole profit: the trading lost money
    w = next(w for w in metrics.result_warnings("signal", stats(), trades(40), interest=12_000.0) if w["code"] == "interest_share")
    assert w["interest_share"] > 1 and w["cagr_ex_interest"] < 0


@needs_data
def test_zero_trade_run_end_to_end(tmp_path):
    from backtester import runner
    s = Strategy(universe=["SPY"], entry="rsi(close, 14) < 0.01", hold_bars=3, start="2015-01-01", end="2018-12-31")
    res = runner.run(s)
    assert res.trades.empty
    A = report.analyze(res, sensitivity=False, mc=False, detail=False)
    assert codes(A["warnings"]) == ["no_trades"]
    assert math.isnan(A["stats"]["sharpe"]) and math.isnan(A["stats"]["calmar"]) and A["relative"] == {}
    assert "No trades" in report.console_summary(A)
    report.write_outputs(A, tmp_path, excel=False)
    summ = json.loads((tmp_path / "summary.json").read_text())
    assert summ["warnings"][0]["code"] == "no_trades" and summ["stats"]["sharpe"] is None
    assert '"code":"no_trades"' in (tmp_path / "report.html").read_text()


# ------------------------------------------------------------ price chart payload

def test_oscillators_are_routed_to_panes():
    panes = report.oscillator_panes("(rsi(close, 2) < 10) and (close > sma(close, 200)) and 20 < stoch_k(14, 3)")
    assert [p["name"] for p in panes] == ["RSI", "Stochastic"]
    assert panes[0]["calls"] == ["rsi(close, 2)"] and panes[0]["levels"] == [10] and panes[0]["range"] == [0, 100]
    assert panes[1]["calls"] == ["stoch_k(14, 3)", "stoch_d(14, 3)"] and panes[1]["levels"] == [20]
    macd = report.oscillator_panes("crossover(macd(), macd_signal())")
    assert macd == [{"name": "MACD", "calls": ["macd()", "macd_signal()", "macd_hist()"], "levels": [], "range": None}]
    assert report.oscillator_panes('sym("SPY").rsi(2) < 5') == []  # another ticker's indicator isn't plotted


def test_price_payload_splits_overlays_and_panes(monkeypatch):
    rng = np.random.default_rng(1)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 400)))
    idx = pd.bdate_range("2020-01-01", periods=len(c))
    df = pd.DataFrame({"open": c * 0.999, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": 1e6,
                       "quote_close": c, "dividend": 0.0, "split": 0.0}, index=idx)
    monkeypatch.setattr(data, "load", lambda t: df)
    monkeypatch.setattr(data, "load_many", lambda ts: {t: df for t in ts})
    s = Strategy(cash_rate=None, universe=["X"], entry="rsi(close, 2) < 20 and close > sma(close, 50)",
                 exit_when="rsi(close, 2) > 70")
    res = engine.run(s)
    assert not res.trades.empty
    P = report.price_payload(res)["X"]
    assert list(P["overlays"]) == ["sma(close, 50)"]
    assert P["panes"][0]["name"] == "RSI" and list(P["panes"][0]["series"]) == ["rsi(close, 2)"]
    assert sorted(P["panes"][0]["levels"]) == [20, 70]
    assert len(P["panes"][0]["series"]["rsi(close, 2)"]) == len(P["dates"])


# ------------------------------------------------------------ sweeps

def test_sweep_labels_are_complete():
    labels, _, _ = research.expand("buy QQQ when RSI({2..5}) is below {5..25 step 5}, hold {1,3} days")
    assert labels == ["RSI({2..5})", "below {5..25 step 5}", "hold {1,3}"]
    labels, _, _ = research.expand("buy SPY when rsi({2,3}) < {10,20} and rsi({2,3}) > 1, hold 2 days")
    assert labels == ["rsi({2,3})", "< {10,20}", "rsi({2,3}) #2"]


def test_expected_max_sharpe_matches_extreme_value_theory():
    # E[max of 1000 standard normals] is about 3.24
    assert metrics.expected_max_sharpe(1000, 1.0) == pytest.approx(3.24, abs=0.05)
    assert metrics.expected_max_sharpe(1, 1.0) == 0.0


def test_deflated_sharpe():
    rng = np.random.default_rng(0)
    trials = list(rng.normal(0, 0.3, 50))
    # a best Sharpe no better than luck: low DSR; a much better one on a long sample: high DSR
    lucky = metrics.deflated_sharpe(max(trials), trials, n_obs=2520)
    assert lucky["n_trials"] == 50 and lucky["dsr"] < 0.8
    assert lucky["expected_max_sharpe"] == pytest.approx(
        metrics.expected_max_sharpe(50, np.std(trials, ddof=1) / np.sqrt(252)) * np.sqrt(252))
    strong = metrics.deflated_sharpe(2.5, trials, n_obs=2520)
    assert strong["dsr"] > 0.99
    # fat tails and negative skew make the same Sharpe less convincing
    fat = metrics.deflated_sharpe(1.2, trials, n_obs=1000, skew=-2, excess_kurt=20)
    thin = metrics.deflated_sharpe(1.2, trials, n_obs=1000)
    assert fat["dsr"] < thin["dsr"]
    # the probabilistic Sharpe ratio formula, by hand
    sr, sr0 = 1.2 / np.sqrt(252), thin["expected_max_sharpe"] / np.sqrt(252)
    from statistics import NormalDist
    assert thin["dsr"] == pytest.approx(NormalDist().cdf((sr - sr0) * np.sqrt(999) / np.sqrt(1 + 0.5 * sr ** 2)))
    assert math.isnan(metrics.deflated_sharpe(1.0, [1.0], n_obs=500)["dsr"])


def test_multiple_testing_summary_ignores_idle_combinations():
    rows = [{"ok": True, "trades": 10, "sharpe": s, "params": [i], "n_obs": 1000, "skew": 0.0, "kurtosis": 0.0}
            for i, s in enumerate([0.1, 0.5, 0.9, 0.3])]
    rows += [{"ok": True, "trades": 0, "sharpe": 9.0, "params": [9]}, {"ok": False, "error": "x", "params": [10]}]
    mt = research.multiple_testing(rows)
    assert mt["n_trials"] == 4 and mt["n_combinations"] == 6 and mt["best_params"] == [2]
    assert mt["best_sharpe"] == 0.9 and 0 <= mt["dsr"] <= 1
