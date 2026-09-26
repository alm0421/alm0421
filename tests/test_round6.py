"""Round 6 fixes: relative hurdles ("only if their 12 month return is above BIL's") and the guard against
conditions that compare a value with itself, blended benchmark phrases, more lazy portfolios, "20% each ...
each only when above its 10 month moving average" (Faber GTAA), per-flow inflation indexing and the dollars a
real flow is expressed in, withdrawals capped at the balance (depletion), a $0 start, December-to-December
inflation in the yearly table, and console table alignment."""
import re

import numpy as np
import pandas as pd
import pytest

from backtester import data, metrics, parser, report, runner
from backtester.parser import ParseError
from backtester.portfolio import Portfolio
from backtester.strategy import Strategy

HAVE = {"SPY", "EFA", "IEF", "VNQ", "DBC", "BIL", "AGG", "TLT", "SPYSIM", "EFASIM", "BILSIM", "IEFSIM"} <= set(
    data.available_tickers())
pytestmark = pytest.mark.skipif(not HAVE, reason="price data not downloaded")

WD = ("add $1,000 a month for 20 years, then withdraw $50,000 a year adjusted for inflation, hold 60% SPYSIM and "
      "40% IEFSIM, since 1980, starting with $10,000, rebalance yearly")


def port(text) -> Portfolio:
    p = parser.parse(text)
    assert isinstance(p, Portfolio), text
    return p


# ------------------------------------------------------------------ relative hurdles

HURDLE = 'tret(tr, 252) > tret(sym("BILSIM").tr, 252)'


@pytest.mark.parametrize("phrase", [
    "is above BILSIM 12 month return", "is above BILSIM's 12 month return", "beats BILSIM's", "beats BILSIM",
    "exceeds that of BILSIM", "is greater than the 12 month return of BILSIM", "is higher than BILSIM's 12 month return",
    "exceeds BILSIM",
])
def test_relative_hurdle_each_candidate_vs_the_other_ticker(phrase):
    p = port(f"hold the top 1 of SPYSIM and EFASIM by 12 month return, only if their 12 month return {phrase}, "
             "otherwise hold IEFSIM, since 1976, rebalance monthly")
    f = p.tree["filter"]
    assert f["require"] == HURDLE
    assert p.tree["fallback"] == {"asset": "IEFSIM"}
    assert "if" not in p.tree                      # not a gate evaluated on BILSIM
    assert any("relative hurdle" in n for n in p.notes)


def test_relative_hurdle_on_an_if_node_is_about_the_holding():
    p = port("hold SPY if its 12 month return is above BIL's 12 month return, otherwise IEF")
    assert p.tree["on"] == "SPY"
    assert p.tree["if"] == 'tret(tr, 252) > tret(sym("BIL").tr, 252)'
    p = port("hold SPY if its 6 month return is below BIL's, otherwise IEF")
    assert p.tree["if"] == 'tret(tr, 126) < tret(sym("BIL").tr, 126)'


def test_relative_hurdle_in_a_signal_rule():
    s = parser.parse("buy QQQ when its 20 day return is above SPY's 20 day return, sell when its 20 day return is below SPY's")
    assert isinstance(s, Strategy)
    assert s.entry == 'ret(close, 20) > ret(sym("SPY").close, 20)'
    assert s.exit_when == 'ret(close, 20) < ret(sym("SPY").close, 20)'


def test_that_beat_bil_means_the_fund_and_otherwise_is_taken():
    p = port("hold the top 1 of SPY and EFA by 12 month return, that beat BIL, otherwise IEF")
    assert p.tree["filter"]["require"] == 'tret(tr, 252) > tret(sym("BIL").tr, 252)'
    assert p.tree["fallback"] == {"asset": "IEF"}
    p = port("hold the top 1 of SPY and EFA by 12 month return, that beat cash, otherwise IEF")
    assert p.tree["filter"]["require"] == "tret(tr, 252) > tbill_ret(252)"


@pytest.mark.parametrize("text", [
    "hold SPY if its 12 month return is above 12 month return, otherwise IEF",
    "hold SPY if `close > close` otherwise IEF",
    "buy SPY when `rsi(close, 2) < rsi(close, 2)`, hold 3 days",
])
def test_conditions_comparing_a_value_with_itself_are_refused(text):
    with pytest.raises(ParseError, match="compares a value with itself"):
        parser.parse(text)


def test_pronoun_without_a_subject_is_refused():
    with pytest.raises(ParseError, match="whose"):
        parser.parse("if its 12 month return is above BIL then SPY otherwise IEF")


def test_unreadable_requirement_is_refused_not_dropped():
    with pytest.raises(ParseError):
        parser.parse("hold the top 1 of SPY and EFA by 12 month return, only if their 12 month return wobbles, otherwise IEF")


def test_relative_hurdle_matches_pandas():
    p = port("hold the top 1 of SPYSIM and EFASIM by 12 month return, only if their 12 month return is above BILSIM "
             "12 month return, otherwise hold IEFSIM, since 2000, rebalance monthly")
    res = runner.run(p)
    h = res.holdings.drop(columns="cash")
    held = h.idxmax(axis=1).where(h.max(axis=1) > 0.5)
    cal = h.index
    r12 = pd.DataFrame({t: data.load(t)["adj_close"].reindex(cal) / data.load(t)["adj_close"].shift(252).reindex(cal) - 1
                        for t in ("SPYSIM", "EFASIM", "BILSIM")})
    month_ends = pd.Series(cal, index=cal).groupby(cal.to_period("M")).max().values[:-1]
    n_safe = 0
    for d in month_ends:
        x = r12.loc[d]
        best = x[["SPYSIM", "EFASIM"]].idxmax()
        exp = best if x[best] > x["BILSIM"] else "IEFSIM"
        n_safe += exp == "IEFSIM"
        assert held.loc[d] == exp, d
    assert 0 < n_safe < len(month_ends)            # the hurdle bites sometimes, not always


# ------------------------------------------------------------------ blended benchmarks

@pytest.mark.parametrize("phrase", ["vs 60/40 SPY/AGG", "vs 60% SPY and 40% AGG", "benchmark 60/40 SPY/AGG",
                                    "compared with 60% SPY / 40% AGG", "compared with a 60/40 SPY/AGG blend",
                                    "versus 60% SPY, 40% AGG"])
def test_blended_benchmark_phrases(phrase):
    p = port(f"hold 60% SPY and 40% TLT, {phrase}")
    assert p.benchmark == "60% SPY / 40% AGG"
    assert p.tree["w"] == [0.6, 0.4]


def test_blended_benchmark_on_a_signal_strategy_and_bad_weights():
    s = parser.parse("buy SPY when RSI(2) is below 10, hold 3 days, vs 60/40 SPY/AGG")
    assert isinstance(s, Strategy) and metrics.benchmark_label(s.benchmark) == "60% SPY / 40% AGG"
    with pytest.raises(ParseError, match="100%"):
        parser.parse("hold 60% SPY and 40% TLT, vs 60/30 SPY/AGG")
    assert port("hold 60% SPY and 40% TLT, vs QQQ").benchmark == "QQQ"


# ------------------------------------------------------------------ lazy portfolios and per-asset timing

@pytest.mark.parametrize("name,first", [
    ("buffett 90/10", "VOO"), ("global market portfolio", "SPY"), ("sandwich portfolio", "SPY"),
    ("desert portfolio", "IEF"), ("ultimate buy and hold portfolio", "SPY"), ("paul merriman's ultimate portfolio", "SPY"),
    ("weird portfolio", "VBR"), ("core four", "VTI"), ("talmud portfolio", "VTI"), ("pinwheel portfolio", "SPY"),
    ("coffeehouse portfolio", "BND"), ("no-brainer portfolio", "SPY"),
])
def test_lazy_portfolios(name, first):
    p = port(name)
    assert p.tree["weights"] == "specified"
    assert abs(sum(p.tree["w"]) - 1) < 1e-9
    kids = [k["asset"] for k in p.tree["children"]]
    assert kids[0] == first and len(set(kids)) == len(kids)
    assert set(kids) <= set(data.available_tickers())


def test_desert_and_buffett_weights_and_proxy_notes():
    p = port("desert portfolio")
    assert dict(zip([k["asset"] for k in p.tree["children"]], p.tree["w"])) == {"IEF": 0.6, "VTI": 0.3, "GLD": 0.1}
    p = port("buffett 90/10 since 1976")
    assert [k["asset"] for k in p.tree["children"]] == ["SPYSIM", "SHYSIM"] and p.tree["w"] == [0.9, 0.1]
    assert any("Proxies" in n for n in port("global market portfolio").notes)
    assert any("VSS" in n for n in port("weird portfolio").notes)


def test_each_asset_timed_on_its_own_with_stated_weights():
    p = port("hold 20% each of SPY, EFA, IEF, VNQ, DBC, each only when above its 10 month moving average, "
             "otherwise BIL, rebalance monthly")
    assert p.tree["weights"] == "equal" and len(p.tree["children"]) == 5
    for k, t in zip(p.tree["children"], ["SPY", "EFA", "IEF", "VNQ", "DBC"]):
        assert (k["on"], k["then"], k["else"]) == (t, {"asset": t}, {"asset": "BIL"})
        assert "monthly_sma(10)" in k["if"]
    p = port("hold 15% each of SPY, EFA, IEF, VNQ, DBC, each only when above its 10 month moving average, otherwise BIL")
    assert p.tree["w"] == [0.15] * 5 + [0.25] and p.tree["children"][-1] == {"cash": True}
    with pytest.raises(ParseError, match="more than 100%"):
        parser.parse("hold 25% each of SPY, EFA, IEF, VNQ, DBC, each only when above its 10 month moving average, otherwise BIL")


def test_gtaa_timing_matches_pandas():
    p = port("hold 20% each of SPY, EFA, IEF, VNQ, DBC, each only when above its 10 month moving average, otherwise "
             "BIL, rebalance monthly, since 2010")
    h = runner.run(p).holdings
    cal = h.index
    tk = ["SPY", "EFA", "IEF", "VNQ", "DBC"]
    ac = {t: data.load(t)["adj_close"] for t in tk}
    month_ends = pd.Series(cal, index=cal).groupby(cal.to_period("M")).max().values[:-1]
    for d in month_ends[::3]:
        exp = {"BIL": 0.0}
        for t in tk:
            s = ac[t][ac[t].index <= d]
            sma = s.groupby(s.index.to_period("M")).last().iloc[-10:].mean()
            if s.iloc[-1] > sma:
                exp[t] = 0.2
            else:
                exp["BIL"] += 0.2
        got = {c: h.loc[d, c] for c in h.columns if c != "cash" and h.loc[d, c] > 1e-6}
        for k in set(exp) | set(got):
            assert abs(exp.get(k, 0) - got.get(k, 0)) < 1e-6, (d, k)


# ------------------------------------------------------------------ cash flows

def test_inflation_flag_attaches_to_its_own_flow():
    p = port(WD)
    assert (p.contribution_inflation, p.withdrawal_inflation) == (False, True)
    assert "withdraw $50,000 yearly in 1980 dollars" in p.summary()
    p = port(WD.replace("add $1,000 a month for 20 years", "add $1,000 a month adjusted for inflation for 20 years")
             .replace("withdraw $50,000 a year adjusted for inflation", "withdraw $50,000 a year"))
    assert (p.contribution_inflation, p.withdrawal_inflation) == (True, False)
    p = port("add $500 a month for 10 years, then withdraw $20,000 a year, hold 60% SPY and 40% TLT, adjusted for inflation")
    assert (p.contribution_inflation, p.withdrawal_inflation) == (True, True)
    assert any("applies to both" in n for n in p.notes)


def test_dollar_base_of_a_real_flow():
    p = port("withdraw $40,000 a year in 2000 dollars, hold 60% SPY and 40% TLT, starting with $1,000,000")
    assert (p.withdrawal_dollars, p.withdrawal_inflation, p.start) == ("2000", True, None)
    p = port("withdraw $50,000 a year in today's dollars starting in 2000, hold 60% SPYSIM and 40% IEFSIM, since 1980, "
             "starting with $500,000, rebalance yearly")
    assert p.withdrawal_dollars == "flow"
    res = runner.run(p)
    f = res.extras["flows"]
    assert f[f < 0].iloc[0] == pytest.approx(-50_000)
    assert any("first paid 2000-01-03 as $50,000" in n for n in p.notes)


def _depleted():
    p = port(WD)
    res = runner.run(p)
    return p, res


def test_withdrawals_are_capped_at_the_balance():
    p, res = _depleted()
    eq, f = res.equity, res.extras["flows"]
    prev = eq.shift(1)
    # no withdrawal is bigger than what the account held before it (plus that day's move)
    assert (f[f < 0].abs() <= (prev[f < 0] * 1.2)).all()
    dep = res.extras["depleted"]
    assert eq[dep] == 0 and (eq[eq.index > dep] == 0).all() and (f[f.index > dep] == 0).all()
    assert f[dep] < 0 and -f[dep] < -res.extras["flows_requested"][dep]
    note = next(n for n in p.notes if n.startswith("Portfolio depleted on"))
    assert str(dep.date()) in note
    # contributions are not indexed: $12,000 a year
    y = f[f > 0].groupby(f[f > 0].index.year).sum()
    assert (y.loc[1980:1999] == 12_000).all()
    first_wd = f[f < 0].iloc[0]
    assert first_wd == pytest.approx(-110_789, abs=1)
    # nothing is left held: every holding period closed
    assert not res.trades["exit_reason"].isin(metrics.OPEN_REASONS).any()
    # P&L by holding + interest - fees == final equity - capital - net flows actually made
    pnl = res.extras["attribution"]["pnl"].sum() + res.interest - res.extras["fees"]
    assert pnl == pytest.approx(eq.iloc[-1] - p.capital - f.sum(), abs=1e-6)


def test_depleted_report_numbers_add_up():
    p, res = _depleted()
    A = report.analyze(res, sensitivity=False, mc=True)
    dep = res.extras["depleted"]
    c = A["cash"]
    assert c["depleted_on"] == dep.date() and c["ran_out"] and c["ending_balance"] == 0
    y = A["yearly"]
    assert y["withdrawals"].sum() == pytest.approx(c["total_withdrawals"])
    assert y["contributions"].sum() == pytest.approx(c["total_contributions"])
    assert c["starting_balance"] + c["total_contributions"] - c["total_withdrawals"] + c["net_gain"] == pytest.approx(0, abs=1e-6)
    after = y.index.astype(int) > dep.year
    assert after.any() and y.loc[after, "return"].isna().all() and y.loc[after, "real_return"].isna().all()
    assert y.loc[~after, "return"].notna().all()
    assert A["stats"]["end"] == dep.date() and np.isfinite(A["stats"]["cagr"]) and A["stats"]["cagr"] > 0
    assert not A["stats"]["wiped_out"]
    assert np.isfinite(c["money_weighted_return"])
    # benchmarks get the scheduled flows, each capped at its own balance
    for k, bc in A["benchmark_cash"].items():
        v = A["benchmarks_with_flows"][k]
        assert (v >= 0).all()
        assert bc["starting_balance"] + bc["total_contributions"] - bc["total_withdrawals"] + bc["net_gain"] \
            == pytest.approx(bc["ending_balance"], abs=1e-6)
        if bc["ran_out"]:
            assert bc["ending_balance"] == 0 and bc["depleted_on"] is not None
    assert A["monte_carlo"] and 0 <= A["monte_carlo"]["success_rate"] <= 1
    out = report.console_summary(A)
    assert f"portfolio depleted on {dep.date()}" in out
    assert re.search(rf"^{dep.year + 1}\s+n/a\s+n/a", out, re.M)


def test_apply_flows_caps_and_stops():
    g = pd.Series([100.0, 110.0, 99.0, 120.0, 130.0], index=pd.bdate_range("2020-01-01", periods=5))
    f = pd.Series([0.0, -50.0, -100.0, -10.0, 20.0], index=g.index)
    v, paid = metrics.apply_flows(g, f)
    assert v.tolist() == pytest.approx([100.0, 60.0, 0.0, 0.0, 0.0])
    assert paid.tolist() == pytest.approx([0.0, -50.0, -54.0, 0.0, 0.0])


def test_twr_on_the_day_the_money_runs_out_is_the_market_move():
    idx = pd.bdate_range("2020-01-01", periods=3)
    eq = pd.Series([100.0, 105.0, 0.0], index=idx)
    fl = pd.Series([0.0, 0.0, -102.0], index=idx)   # the day's close value (102) is paid out
    r = metrics.twr_returns(eq, fl)
    assert r.iloc[-1] == pytest.approx(102 / 105 - 1)


def test_starting_with_zero():
    p = port(WD.replace("$10,000", "$0"))
    assert p.capital == 0
    res = runner.run(p)
    A = report.analyze(res, sensitivity=False, mc=False)
    assert A["nav"].iloc[0] > 0 and np.isfinite(A["stats"]["cagr"]) and np.isfinite(A["stats"]["sharpe"])
    assert A["cash"]["starting_balance"] == 0 and A["cash"]["total_contributions"] > 0
    assert all(np.isfinite(b.iloc[-1]) for b in A["benchmarks"].values())
    with pytest.raises(ParseError, match=r"\$0"):
        parser.parse("hold 60% SPY and 40% TLT, starting with $0")
    with pytest.raises(ParseError, match=r"\$0"):
        parser.parse("add $500 a month starting in 2010, hold 60% SPY and 40% TLT, since 2000, starting with $0")


def test_yearly_inflation_is_december_to_december():
    m = data.cpi_monthly()
    if m.empty:
        pytest.skip("no CPI data")
    exp = float(m[m.index == "2010-12-01"].iloc[0] / m[m.index == "2009-12-01"].iloc[0] - 1)
    assert metrics.calendar_inflation(pd.Timestamp("2010-01-04"), pd.Timestamp("2010-12-31")) == pytest.approx(exp)
    p = port("hold 60% SPY and 40% TLT, since 2005, rebalance yearly, add $100 a month")
    A = report.analyze(runner.run(p), sensitivity=False, mc=False)
    assert A["yearly"].loc[2010, "inflation"] == pytest.approx(exp)
    # flow indexing keeps the published (lagged) CPI
    assert data.cpi().index[0] > m.index[0]


def test_console_tables_stay_aligned_with_long_names():
    assert report._fit("60% SPY / 40% AGG blend", 10) == "60% SPY /~"
    assert report._fit("SPY", 6, right=True) == "   SPY"
    p = port("hold 60% SPY and 40% TLT, since 2010, vs 60/40 SPY/AGG, rebalance quarterly")
    out = report.console_summary(report.analyze(runner.run(p), sensitivity=False, mc=False))
    lines = out.splitlines()
    head = next(i for i, l in enumerate(lines) if "Final $" in l)
    rows = lines[head + 1: head + 4]
    ends = {len(r.split("  (from")[0].rstrip()) for r in rows}
    assert len(ends) == 1, rows
    ylines = [l for l in lines if re.match(r"^20\d\d", l)]
    assert len({len(l) for l in ylines}) == 1
