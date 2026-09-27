"""Round 12 (Composer expert review): 'or' between holdings is refused (never read as both), the drawdown ranking
notes say which drawdown is used, 'top ... by lowest ...' says it flips, 'X, except when C hold Y', rebalance timing
(first trading day of the period, weekdays; Composer imports trade on the first trading day, as Composer documents),
filters always show their fallback, and new condition phrases and Composer's editor shorthand."""
import numpy as np
import pandas as pd
import pytest

from backtester import composer_export as ce
from backtester import composer_import as ci
from backtester import data, parser
from backtester import portfolio as pf

AVAIL = set(data.available_tickers())


def needs(*t):
    return pytest.mark.skipif(not set(t) <= AVAIL, reason="price data not downloaded")


# ------------------------------------------------------------ 1. 'or' between holdings

@needs("TQQQ", "TMF", "SOXL", "SPY", "BIL", "TLT", "GLD")
@pytest.mark.parametrize("text", [
    "hold TQQQ or TMF",
    "hold either TQQQ or TMF",
    "hold TQQQ and/or TMF",
    "if SPY is above its 200 day moving average then TQQQ or SOXL else BIL",
    "if SPY is above its 200 day moving average then TQQQ else TMF or BIL",
    "equal weight SPY or TLT",
    "SPY, TLT or GLD, inverse volatility weighted",
    "hold SPY, TLT or GLD",
])
def test_or_between_holdings_is_refused_with_both_readings(text):
    with pytest.raises(parser.ParseError, match="'or' between holdings is a choice") as e:
        parser.parse(text)
    msg = str(e.value)
    assert " and " in msg and "the top 1 of" in msg       # both suggestions: hold all of them, or pick one


@needs("TQQQ", "SOXL", "SPY", "TLT")
@pytest.mark.parametrize("text", ["hold the top 1 of TQQQ or SOXL by 10 day return",
                                  "whichever of SPY or TLT has the higher 12 month return",
                                  "rotate between SPY or TLT by 3 month return"])
def test_or_where_something_chooses_is_fine(text):
    p = parser.parse(text)
    assert p.tree["filter"]["n"] == 1 and len(p.tree["children"]) == 2


@needs("TQQQ", "SPY", "BIL")
def test_or_else_is_still_otherwise():
    p = parser.parse("if SPY is above its 200 day moving average then TQQQ or else BIL")
    assert p.tree["then"] == {"asset": "TQQQ"} and p.tree["else"] == {"asset": "BIL"}


@needs("TQQQ", "TMF")
def test_vs_ticker_is_named_as_the_benchmark():
    p = parser.parse("hold TQQQ vs TMF")
    assert p.tree == {"asset": "TQQQ"} and p.benchmark == "TMF"
    assert any(n.startswith("Benchmark: TMF") and "not held" in n for n in p.notes)


# ------------------------------------------------------------ 2. drawdown vs max drawdown

@needs("TQQQ", "SOXL")
def test_drawdown_ranking_note_names_the_measure():
    p = parser.parse("hold the top 1 of TQQQ and SOXL by 20 day drawdown")
    assert p.tree["filter"]["by"] == "-drawdown(close, 20)" and p.tree["filter"]["select"] == "top"
    note = next(n for n in p.notes if n.startswith("Ranking by '20 day drawdown'"))
    assert "current drawdown" in note and "not Composer's Max Drawdown" in note and "'20 day max drawdown'" in note
    assert "as Composer's max drawdown filter" not in note
    p = parser.parse("hold the top 1 of TQQQ and SOXL by 20 day max drawdown")
    assert p.tree["filter"]["by"] == "max_drawdown(tr, 20)"
    note = next(n for n in p.notes if n.startswith("Ranking by '20 day max drawdown'"))
    assert "deepest peak-to-trough fall" in note and "Composer's Max Drawdown" in note


def test_drawdown_measures_differ_on_a_recovered_path():
    # a fall of 20% then a full recovery: the max drawdown over the window is 20%, the current drawdown 0
    from backtester import expr
    close = pd.Series([100, 100, 80, 90, 100, 100.0], index=pd.bdate_range("2024-01-01", periods=6))
    df = pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 1e6})
    df["quote_close"] = df["close"]
    df["dividend"] = 0.0
    df["split"] = 0.0
    ns = expr.Namespace(df)
    mdd = expr.evaluate_value("max_drawdown(close, 5)", ns).iloc[-1]
    dd = expr.evaluate_value("drawdown(close, 5)", ns).iloc[-1]
    assert mdd == pytest.approx(0.2) and dd == pytest.approx(0.0)


# ------------------------------------------------------------ 3. top ... by lowest ...

@needs("TQQQ", "SOXL", "TECL")
def test_top_by_lowest_says_it_is_the_bottom():
    p = parser.parse("hold the top 1 of TQQQ and SOXL by lowest 10 day RSI")
    assert p.tree["filter"]["select"] == "bottom" and p.tree["filter"]["by"] == "rsi(close, 10)"
    assert any("read as the bottom 1 by 10 day RSI" in n for n in p.notes)
    p = parser.parse("hold the bottom 2 of TQQQ, TECL and SOXL by lowest 10 day RSI")
    assert p.tree["filter"]["select"] == "top"
    assert any("cancel out" in n and "top 2" in n for n in p.notes)
    p = parser.parse("hold the top 1 of TQQQ and SOXL by 10 day RSI")
    assert not any("read as the" in n for n in p.notes)


# ------------------------------------------------------------ 4. except when

@needs("TQQQ", "SPY", "BIL", "TLT")
@pytest.mark.parametrize("text", [
    "hold TQQQ, except when SPY is below its 200 day moving average hold BIL",
    "hold TQQQ except when SPY is below its 200 day moving average, in which case BIL",
    "TQQQ, except while SPY is below its 200 day moving average hold BIL",
])
def test_except_when(text):
    p = parser.parse(text)
    assert p.tree == {"if": "(close < sma(close, 200))", "on": "SPY", "then": {"asset": "BIL"}, "else": {"asset": "TQQQ"}}
    assert any("except when" in n for n in p.notes)


@needs("TQQQ", "SPY", "BIL", "TLT")
def test_except_when_with_a_group_and_options():
    p = parser.parse("hold 60% SPY and 40% TLT, except when SPY is below its 200 day moving average hold BIL, rebalance monthly")
    assert p.rebalance == "monthly" and p.tree["then"] == {"asset": "BIL"} and p.tree["else"]["w"] == [0.6, 0.4]


@needs("TQQQ", "SPY")
def test_except_when_without_the_other_holding_asks_for_it():
    with pytest.raises(parser.ParseError, match="what should it hold while that condition is true") as e:
        parser.parse("hold TQQQ, except when SPY is below its 200 day moving average")
    assert "otherwise cash" not in str(e.value)


@needs("TQQQ", "SPY")
def test_missing_otherwise_suggestion_does_not_propose_cash():
    # the suggestion names the missing piece; it does not offer an edit that silently means something else
    with pytest.raises(parser.ParseError) as e:
        parser.parse("hold TQQQ when SPY is above its 200 day moving average, rebalance monthly")
    msg = str(e.value)
    assert "otherwise cash'" not in msg


# ------------------------------------------------------------ 5. rebalance timing

def _cal():
    cal = pd.bdate_range("2023-12-20", "2024-07-10")
    hol = pd.to_datetime(["2023-12-25", "2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29", "2024-05-27",
                          "2024-06-19", "2024-07-04"])
    return cal[~cal.isin(hol)]


def _days(cal, mask):
    return [str(d.date()) for d in cal[mask]]


def test_schedule_first_trading_day_of_each_period():
    cal = _cal()
    assert _days(cal, pf._schedule(cal, "monthly", "start")) == [
        "2023-12-20", "2024-01-02", "2024-02-01", "2024-03-01", "2024-04-01", "2024-05-01", "2024-06-03", "2024-07-01"]
    assert _days(cal, pf._schedule(cal, "quarterly", "start")) == ["2023-12-20", "2024-01-02", "2024-04-01", "2024-07-01"]
    assert _days(cal, pf._schedule(cal, "yearly", "start")) == ["2023-12-20", "2024-01-02"]
    wk = _days(cal, pf._schedule(cal, "weekly", "start"))
    assert wk[:4] == ["2023-12-20", "2023-12-26", "2024-01-02", "2024-01-08"] and "2024-01-16" in wk   # MLK Monday
    # the default is the period's last trading day
    assert _days(cal, pf._schedule(cal, "monthly"))[:4] == ["2023-12-20", "2023-12-29", "2024-01-31", "2024-02-29"]
    assert _days(cal, pf._schedule(cal, "monthly", "end")) == _days(cal, pf._schedule(cal, "monthly"))


def test_schedule_weekdays():
    cal = _cal()
    mon = _days(cal, pf._schedule(cal, "weekly", "monday"))
    assert "2024-01-16" in mon and "2024-01-15" not in mon          # a Monday holiday: the Tuesday
    assert all(pd.Timestamp(d).dayofweek in (0, 1) for d in mon[1:])
    fri = _days(cal, pf._schedule(cal, "weekly", "friday"))
    assert "2024-03-28" in fri                                        # Good Friday: that week's last session (Thursday)
    assert all(pd.Timestamp(d).dayofweek in (3, 4) for d in fri[1:])
    wed = _days(cal, pf._schedule(cal, "weekly", "wednesday"))
    assert wed[1:4] == ["2023-12-27", "2024-01-03", "2024-01-10"]
    assert "2024-06-20" in wed      # Juneteenth (a Wednesday): the Thursday


def test_schedule_start_has_no_lookahead():
    cal = _cal()
    full = pf._schedule(cal, "monthly", "start")
    for cut in (40, 41, 42, 90):
        assert (pf._schedule(cal[:cut], "monthly", "start") == full[:cut]).all()


@pytest.fixture
def two_assets(monkeypatch):
    cal = _cal()
    rng = np.random.default_rng(3)
    frames = {}
    for t, s in (("AAA", 0.02), ("BBB", 0.01)):
        c = 100 * np.cumprod(1 + rng.normal(0.0005, s, len(cal)))
        df = pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1e6}, index=cal)
        df["quote_close"] = df["close"]
        df["adj_close"] = df["close"]
        df["dividend"] = 0.0
        df["split"] = 0.0
        frames[t] = df
    real = data.load

    def load(t, *a, **k):
        t = data.canonical(t)
        return frames[t] if t in frames else real(t, *a, **k)
    monkeypatch.setattr(data, "load", load)
    monkeypatch.setattr(data, "load_many", lambda ts, *a, **k: {data.canonical(t): load(t) for t in ts})
    return cal


@pytest.mark.parametrize("rb, day, want", [
    ("monthly", "start", ["2024-01-02", "2024-02-01", "2024-03-01", "2024-04-01", "2024-05-01", "2024-06-03", "2024-07-01"]),
    ("monthly", None, ["2023-12-29", "2024-01-31", "2024-02-29", "2024-03-28", "2024-04-30", "2024-05-31", "2024-06-28"]),
    ("quarterly", "start", ["2024-01-02", "2024-04-01", "2024-07-01"]),
])
def test_rebalance_dates_in_a_run(two_assets, rb, day, want):
    p = pf.Portfolio(tree={"weights": "specified", "w": [0.5, 0.5], "children": [{"asset": "AAA"}, {"asset": "BBB"}]},
                     rebalance=rb, rebalance_day=day, cash_rate=0.0, benchmark="AAA")
    res = pf.run(p)
    od = res.orders
    days = sorted({str(d) for d in od.loc[od["reason"] == "rebalance", "date"]})
    assert days == want
    assert str(od["date"].min()) == "2023-12-20"      # the initial purchase on the first day


def test_rebalance_day_validation():
    tree = {"asset": "SPY"}
    with pytest.raises(ValueError, match="needs rebalance 'weekly'"):
        pf.Portfolio(tree=tree, rebalance="monthly", rebalance_day="monday").validate()
    with pytest.raises(ValueError, match="needs a weekly, monthly"):
        pf.Portfolio(tree=tree, rebalance="daily", rebalance_day="start").validate()
    with pytest.raises(ValueError, match="rebalance_day must be"):
        pf.Portfolio(tree=tree, rebalance="monthly", rebalance_day="first").validate()


@needs("SPY", "TLT")
@pytest.mark.parametrize("text, rb, day", [
    ("hold 60% SPY and 40% TLT, rebalance on the first trading day of each month", "monthly", "start"),
    ("hold 60% SPY and 40% TLT, rebalance monthly at the start", "monthly", "start"),
    ("hold 60% SPY and 40% TLT, rebalance monthly at the beginning of the month", "monthly", "start"),
    ("hold 60% SPY and 40% TLT, rebalance at the start of every quarter", "quarterly", "start"),
    ("hold 60% SPY and 40% TLT, rebalance on the first trading day of each year", "yearly", "start"),
    ("hold 60% SPY and 40% TLT, rebalance on the first trading day of each week", "weekly", "start"),
    ("hold 60% SPY and 40% TLT, rebalance every Monday", "weekly", "monday"),
    ("hold 60% SPY and 40% TLT, rebalance weekly on Fridays", "weekly", "friday"),
    ("hold 60% SPY and 40% TLT, rebalance on Tuesdays", "weekly", "tuesday"),
    ("hold 60% SPY and 40% TLT, rebalance on the last trading day of each month", "monthly", None),
])
def test_rebalance_day_phrases(text, rb, day):
    p = parser.parse(text)
    assert p.rebalance == rb and p.rebalance_day == day
    line = next(x for x in p.summary().splitlines() if x.startswith("Rebalancing:"))
    if day == "start":
        assert "first trading day of each" in line
    elif day:
        assert f"every {day.capitalize()}" in line
    else:
        assert "last trading day of each month" in line


@needs("SPY", "TLT")
def test_rebalance_day_phrase_that_contradicts_the_schedule_is_refused():
    with pytest.raises(parser.ParseError, match="monthly schedule cannot trade at the start of each quarter"):
        parser.parse("hold 60% SPY and 40% TLT, rebalance monthly at the start of the quarter")


def _sym(rb):
    return {"step": "root", "name": "x", "rebalance": rb, "children": [
        {"step": "wt-cash-equal", "children": [{"step": "asset", "ticker": "SPY"}, {"step": "asset", "ticker": "TLT"}]}]}


@pytest.mark.parametrize("rb", ["weekly", "monthly", "quarterly", "yearly"])
def test_composer_import_trades_on_the_first_trading_day(rb):
    spec = ci.convert(_sym(rb))
    assert spec["rebalance"] == rb and spec["rebalance_day"] == "start"
    assert any("first trading day of each" in n and "as Composer documents it" in n for n in spec["notes"])
    p = pf.Portfolio.from_dict(spec)
    sym, notes = ce.export(p)
    assert sym["rebalance"] == rb and not any("Rebalance timing" in n for n in notes)
    # a period-end spec exported to Composer says the timing will differ
    p.rebalance_day = None
    assert any("first trading day" in n and "different days" in n for n in ce.export(p)[1])


@pytest.mark.parametrize("rb", ["daily", "threshold"])
def test_composer_import_daily_and_threshold_have_no_day(rb):
    sym = _sym(rb)
    if rb == "threshold":
        sym["rebalance-corridor-width"] = 0.05
    spec = ci.convert(sym)
    assert "rebalance_day" not in spec


# ------------------------------------------------------------ 6. the fallback is always shown

@needs("TQQQ", "SOXL")
def test_filter_interpretation_shows_the_fallback_however_it_was_made():
    typed = parser.parse("hold the top 1 of TQQQ and SOXL by 10 day return")
    assert "fallback" not in typed.tree
    built = pf.Portfolio(tree={**typed.tree, "fallback": {"cash": True}}, rebalance="daily")
    a = [x for x in typed.summary().splitlines() if x.startswith("  ")]
    b = [x for x in built.summary().splitlines() if x.startswith("  ")]
    assert a == b and any("if nothing qualifies:" in x for x in a) and any("cash (T-bills)" in x for x in a)


# ------------------------------------------------------------ 7. phrases

@needs("TQQQ", "SOXL", "BIL", "SPY", "TLT")
@pytest.mark.parametrize("cond, rule, on", [
    ("TQQQ has a 6 day cumulative return of less than -12%", "tret(tr, 6) < -0.12", "TQQQ"),
    ("TQQQ 6 day cumulative return is less than negative 12%", "tret(tr, 6) < -0.12", "TQQQ"),
    ("SPY gained less than 5% over 10 days", "(tret(tr, 10) < 0.05)", "SPY"),
    ("SPY gained more than 5% over 10 days", "(tret(tr, 10) > 0.05)", "SPY"),
    ("SPY lost more than 5% over the last 10 days", "(tret(tr, 10) < -0.05)", "SPY"),
    ("SPY lost less than 5% over 10 days", "(tret(tr, 10) > -0.05)", "SPY"),
    ("SPY 10 day return is down more than 5%", "(tret(tr, 10) < -0.05)", "SPY"),
    ("SPY 10 day return is up more than 5%", "(tret(tr, 10) > 0.05)", "SPY"),
    ("SPY 10 day return is down less than 5%", "(tret(tr, 10) > -0.05)", "SPY"),
    ("SPY is below its 200 day MA by less than 2%", "(close < sma(close, 200) and close > sma(close, 200) * 0.98)", "SPY"),
    ("SPY is less than 2% above its 200 day moving average", "(close > sma(close, 200) and close < sma(close, 200) * 1.02)", "SPY"),
    ("SPY's 20 day return is more than 2% higher than TLT's", 'tret(tr, 20) - tret(sym("TLT").tr, 20) > 0.02', "SPY"),
    ("SPY's 20 day return is more than 2 points above TLT's", 'tret(tr, 20) - tret(sym("TLT").tr, 20) > 0.02', "SPY"),
    ("SPY's 20 day return is more than 2% lower than TLT's 20 day return", 'tret(sym("TLT").tr, 20) - tret(tr, 20) > 0.02', "SPY"),
    ("TQQQ's 10 day RSI is below SOXL's 10 day RSI minus 5", 'rsi(sym("SOXL").close, 10) - rsi(close, 10) > 5', "TQQQ"),
    ("TQQQ's 10 day RSI is above SOXL's 10 day RSI plus 5", 'rsi(close, 10) - rsi(sym("SOXL").close, 10) > 5', "TQQQ"),
    ("TQQQ's 10 day RSI is above SOXL's 10 day RSI minus 5", 'rsi(sym("SOXL").close, 10) - rsi(close, 10) < 5', "TQQQ"),
    ("TQQQ's 10 day RSI is below SOXL's 10 day RSI plus 5", 'rsi(close, 10) - rsi(sym("SOXL").close, 10) < 5', "TQQQ"),
    ("SPY is under 10% from its 52 week low", "(close / lowest(close, 252) - 1 < 0.1)", "SPY"),
    ("SPY is oversold", "(rsi(close, 14) < 30)", "SPY"),
    ("SPY is overbought", "(rsi(close, 14) > 70)", "SPY"),
])
def test_condition_phrases(cond, rule, on):
    p = parser.parse(f"if {cond} then SOXL else BIL")
    assert p.tree["if"] == rule and p.tree["on"] == on


@needs("SPY", "SOXL", "BIL")
def test_oversold_is_a_documented_default_with_a_warning():
    p = parser.parse("if SPY is oversold then SOXL else BIL")
    assert any(n.startswith("Warning: 'is oversold' has no single definition") and "14 day RSI below 30" in n
               for n in p.notes)


@needs("SPY", "TLT", "SOXL", "BIL")
def test_percent_higher_than_another_return_is_percentage_points_with_a_note():
    p = parser.parse("if SPY's 20 day return is more than 2% higher than TLT's then SOXL else BIL")
    assert any("percentage points" in n for n in p.notes)


@needs("SPY", "SOXL", "BIL")
def test_fails_to_stay_above_is_refused_as_ambiguous():
    with pytest.raises(parser.ParseError, match="ambiguous") as e:
        parser.parse("if SPY fails to stay above its 200 day moving average then SOXL else BIL")
    assert "is below its 200 day moving average" in str(e.value) and "crosses below" in str(e.value)


@needs("SPY", "TLT", "TQQQ", "SOXL", "TECL")
@pytest.mark.parametrize("text, tree", [
    ("Weight equal: SPY, TLT", {"weights": "equal", "children": [{"asset": "SPY"}, {"asset": "TLT"}]}),
    ("Weight specified 60% SPY 40% TLT", {"weights": "specified", "w": [0.6, 0.4], "children": [{"asset": "SPY"}, {"asset": "TLT"}]}),
    ("Weight inverse volatility 20d: SPY, TLT", {"weights": "inverse_vol", "children": [{"asset": "SPY"}, {"asset": "TLT"}], "lookback": 20}),
    ("Filter top 2 by 10d cumulative return: TQQQ, SOXL, TECL",
     {"filter": {"select": "top", "n": 2, "by": "tret(tr, 10)", "weights": "equal"}, "universe": "children",
      "children": [{"asset": "TQQQ"}, {"asset": "SOXL"}, {"asset": "TECL"}]}),
    ("Filter: Bottom 1 by Cumulative Return 10d: TQQQ, SOXL",
     {"filter": {"select": "bottom", "n": 1, "by": "tret(tr, 10)", "weights": "equal"}, "universe": "children",
      "children": [{"asset": "TQQQ"}, {"asset": "SOXL"}]}),
])
def test_composer_editor_shorthand(text, tree):
    p = parser.parse(text)
    assert p.tree == tree
    assert any(n.startswith("Composer editor shorthand") for n in p.notes)


@needs("SPY", "TLT")
def test_composer_editor_shorthand_keeps_options():
    p = parser.parse("Weight equal: SPY, TLT, rebalance quarterly")
    assert p.rebalance == "quarterly" and p.tree["weights"] == "equal"
