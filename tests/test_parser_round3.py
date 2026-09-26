"""Round 3 parser fixes: move signs, backticks/brackets in if-chains, Composer thresholds, buy-stop
entries, new signal phrases, higher-timeframe indicators and portfolio cash-flow/rebalance phrases."""
import numpy as np
import pandas as pd
import pytest

from backtester import data, expr, parser
from backtester.portfolio import Portfolio
from backtester.strategy import Strategy

HAVE = {"SPY", "QQQ", "TQQQ", "BIL", "UVXY", "TECL", "SOXL", "SQQQ", "TLT", "AAPL", "^VIX", "EFA", "AGG", "GLD"} <= set(data.available_tickers())
needs_data = pytest.mark.skipif(not HAVE, reason="price data not downloaded")
pytestmark = needs_data


def sig(text) -> Strategy:
    s = parser.parse(text)
    assert isinstance(s, Strategy), text
    return s


def port(text) -> Portfolio:
    p = parser.parse(text)
    assert isinstance(p, Portfolio), text
    return p


# ------------------------------------------------------------------ 1. the sign of a move

DOWN = ["has fallen", "has dropped", "has declined", "has lost", "is down", "was down", "fell", "dropped", "declined",
        "lost", "slid", "plunged", "tumbled", "falls", "drops", "declines", "has sunk", "sank", "has been down", "went down",
        "dipped", "has decreased"]
UP = ["has risen", "has gained", "has climbed", "has jumped", "is up", "was up", "rose", "gained", "jumped", "surged",
      "climbed", "rallied", "rises", "gains", "has surged", "soared", "has been up", "went up", "advanced", "has increased"]
MODS = ["", "by ", "more than ", "at least ", "by more than ", "by at least "]
WINDOWS = [(" over the last 10 days", 10), (" in 10 days", 10), (" over 10 days", 10), (" in the past 10 trading days", 10),
           (" over the last 2 weeks", 10), ("", 1)]


def _moves():
    for verb in DOWN + UP:
        for mod in MODS:
            for win, n in WINDOWS:
                yield verb, mod, win, n, verb in DOWN


@pytest.mark.parametrize("verb,mod,win,n,down", list(_moves()))
def test_move_sign_signal(verb, mod, win, n, down):
    s = sig(f"buy SPY when it {verb} {mod}5%{win}, hold 5 days")
    r = "change" if n == 1 else f"ret(close, {n})"
    assert s.entry == (f"({r} <= -0.05)" if down else f"({r} >= 0.05)"), (verb, mod, win, s.entry)


@pytest.mark.parametrize("verb,mod,win,n,down", [x for x in _moves() if x[1] in ("", "more than ")])
def test_move_sign_portfolio_uses_total_return(verb, mod, win, n, down):
    p = port(f"if TQQQ {verb} {mod}10%{win} hold TQQQ else hold BIL")
    assert p.tree["if"] == (f"(tret(tr, {n}) <= -0.1)" if down else f"(tret(tr, {n}) >= 0.1)"), (verb, mod, win, p.tree["if"])


def test_the_reported_sign_flips():
    assert port("if TQQQ has fallen more than 10% over the last 10 days hold TQQQ else hold BIL").tree["if"] == "(tret(tr, 10) <= -0.1)"
    assert sig("buy SPY when it has fallen 5% over the last 10 days, hold 5 days").entry == "(ret(close, 10) <= -0.05)"


# ------------------------------------------------------------------ 2. backticks and brackets

def test_backticks_with_commas_in_portfolio_condition():
    p = port("if TQQQ `ma_return(tr, 5) < -0.02` then hold TQQQ else hold BIL")
    assert p.tree["if"] == "(ma_return(tr, 5) < -0.02)" and p.tree["on"] == "TQQQ"


def test_comma_before_a_backtick_block_still_splits():
    s = sig("buy SPY when RSI(2) is below 10, and `close > sma(close, 200)`, hold 2 days")
    assert s.entry == "(rsi(close, 2) < 10) and (close > sma(close, 200))"


def test_backtick_or_is_not_split():
    s = sig("buy SPY when `rsi(close, 2) < 5 or ret(close, 3) < -0.05`, hold 2 days")
    assert s.entry == "(rsi(close, 2) < 5 or ret(close, 3) < -0.05)"


def test_mask_helper():
    assert parser._msplit("a, `b, c`, (d, e), f", r",") == ["a", " `b, c`", " (d, e)", " f"]
    assert parser._msplit("a, (unbalanced, b", r",") == ["a", " (unbalanced", " b"]


# ------------------------------------------------------------------ 3/4. if-chains

def test_if_then_filter_without_then():
    p = port("if TQQQ RSI(10) is above 79 hold UVXY, otherwise hold the top 2 of TQQQ, SOXL, TECL by 10 day cumulative return")
    assert p.tree["if"] == "(rsi(close, 10) > 79)" and p.tree["then"] == {"asset": "UVXY"}
    f = p.tree["else"]
    assert f["filter"]["n"] == 2 and f["filter"]["by"] == "tret(tr, 10)"
    assert [c["asset"] for c in f["children"]] == ["TQQQ", "SOXL", "TECL"]


def test_else_if_chain_ending_in_a_filter():
    p = port("if TQQQ RSI(10) is above 79 hold UVXY, else if SPY is above its 200 day moving average hold TQQQ, "
             "otherwise hold the bottom 1 of SQQQ and TLT by 10 day RSI")
    e = p.tree["else"]
    assert e["on"] == "SPY" and e["if"] == "(close > sma(close, 200))" and e["then"] == {"asset": "TQQQ"}
    assert e["else"]["filter"] == {"select": "bottom", "n": 1, "by": "rsi(close, 10)", "weights": "equal"}


@pytest.mark.parametrize("o,c", [("(", ")"), ("[", "]")])
def test_nested_if_in_both_branches(o, c):
    p = port(f"if SPY is above its 200 day moving average then {o}if TQQQ RSI(10) is above 79 then hold UVXY else hold TQQQ{c} "
             f"else {o}if SPY RSI(10) is below 30 then hold TECL else hold BIL{c}")
    assert p.tree == {
        "if": "(close > sma(close, 200))", "on": "SPY",
        "then": {"if": "(rsi(close, 10) > 79)", "on": "TQQQ", "then": {"asset": "UVXY"}, "else": {"asset": "TQQQ"}},
        "else": {"if": "(rsi(close, 10) < 30)", "on": "SPY", "then": {"asset": "TECL"}, "else": {"asset": "BIL"}}}


def test_weighted_and_grouped_branches():
    p = port("if SPY is above its 200 day moving average hold 60% TQQQ and 40% TLT, otherwise hold (equal weight BIL and GLD)")
    assert p.tree["then"]["w"] == [0.6, 0.4]
    assert p.tree["else"] == {"weights": "equal", "children": [{"asset": "BIL"}, {"asset": "GLD"}]}


def test_filter_only_if_inside_a_branch_claims_its_own_otherwise():
    p = port("if SPY is above its 200 day moving average hold the top 1 of QQQ and TLT by 3 month return, "
             "only if their 3 month return is positive, otherwise BIL, otherwise hold GLD")
    assert p.tree["then"]["fallback"] == {"asset": "BIL"} and p.tree["else"] == {"asset": "GLD"}


def test_if_without_otherwise_is_refused():
    with pytest.raises(parser.ParseError, match="otherwise"):
        parser.parse("if SPY is above its 200 day moving average then hold TQQQ, rebalance daily")


def test_unparenthesised_nested_then_is_refused():
    with pytest.raises(parser.ParseError):
        parser.parse("if SPY is above its 200 day moving average then if QQQ is above its 50 day moving average then TQQQ else QQQ else BIL")


# ------------------------------------------------------------------ 5. Composer thresholds in conditions

@pytest.mark.parametrize("cond,rule", [
    ("TQQQ 6 day cumulative return is less than -12%", "tret(tr, 6) < -0.12"),
    ("the 10 day max drawdown of TQQQ is above 20%", "max_drawdown(tr, 10) > 0.2"),
    ("SPY 10 day standard deviation of return is above 2%", "stdev_return(tr, 10) > 0.02"),
    ("TQQQ 10 day moving average of return is below 0", "ma_return(tr, 10) < 0"),
    ("the 10 day return of TQQQ is below -10%", "tret(tr, 10) < -0.1"),
    ("QQQ 20 day return is greater than 5%", "(tret(tr, 20) > 0.05)"),
    ("TQQQ current price is above its 20 day moving average", "close > sma(close, 20)"),
    ("SPY 10 day RSI is above 70", "(rsi(close, 10) > 70)"),
    ("TQQQ 10 day RSI is at least 79", "rsi(close, 10) >= 79"),
    ("TQQQ return over the last 10 days is above 5%", "tret(tr, 10) > 0.05"),
    ("TQQQ 5 day return over the last 10 days is above 1%", None),
])
def test_threshold_conditions(cond, rule):
    if rule is None:  # two different lookbacks
        with pytest.raises(parser.ParseError):
            parser.parse(f"if {cond} then hold TQQQ else hold BIL")
        return
    p = port(f"if {cond} then hold BIL else hold TQQQ")
    assert p.tree["if"] == rule


def test_threshold_return_over_lookback_and_other_ticker():
    p = port("if SPY is above its 200 day moving average and TQQQ 3 day return over the last 3 days is above 1% then TQQQ else BIL")
    assert "tret(tr, 3) > 0.01" in p.tree["if"] or 'tret(sym("TQQQ").tr, 3) > 0.01' in p.tree["if"]


def test_threshold_units_are_checked():
    with pytest.raises(parser.ParseError, match="%"):
        parser.parse("if TQQQ 6 day cumulative return is less than -12 then hold TECL else hold BIL")
    with pytest.raises(parser.ParseError, match="percentage"):
        parser.parse("if TQQQ 10 day RSI is above 79% then hold UVXY else hold TQQQ")


def test_signal_thresholds_keep_price_returns_and_say_so():
    s = sig("buy SPY when the 10 day return is greater than 5%, hold 3 days")
    assert s.entry == "(ret(close, 10) > 0.05)"
    assert any("price returns" in n for n in s.notes)
    p = port("if QQQ 20 day return is greater than 5% then TQQQ else BIL")
    assert any("total returns" in n for n in p.notes)


# ------------------------------------------------------------------ 6/7. routing and default rebalance

def test_verbless_if_is_an_allocation():
    assert parser.looks_like_allocation("if SPY is above its 200 day SMA then TQQQ else BIL")
    p = port("if SPY is above its 200 day SMA then TQQQ else BIL")
    assert p.tree == {"if": "(close > sma(close, 200))", "on": "SPY", "then": {"asset": "TQQQ"}, "else": {"asset": "BIL"}}


def test_if_trees_default_to_daily_even_with_filters():
    p = port("if TQQQ RSI(10) is above 79 hold UVXY, otherwise hold the top 2 of TQQQ, SOXL, TECL by 10 day cumulative return")
    assert p.rebalance == "daily" and any("every day" in n and "Composer" in n for n in p.notes)
    assert port("hold the top 2 of QQQ, SPY, TLT and GLD by 3 month return").rebalance == "daily"   # round 4: filters default to daily (Composer)
    assert port("equal weight SPY, TLT and GLD").rebalance == "monthly"
    assert port("if SPY is above its 200 day moving average hold QQQ, otherwise TLT, rebalance monthly").rebalance == "monthly"


# ------------------------------------------------------------------ 8. negations

def test_not_above_and_not_below():
    assert port("if TQQQ RSI(10) is not above 79 hold TQQQ else hold BIL").tree["if"] == "(rsi(close, 10) <= 79)"
    s = sig("buy SPY when RSI(2) is not below 30 and it is not above its 200 day moving average, hold 5 days")
    assert s.entry == "(rsi(close, 2) >= 30) and (close <= sma(close, 200))"
    assert sig("buy SPY when RSI(2) <= 10, hold 2 days").entry == "(rsi(close, 2) <= 10)"


# ------------------------------------------------------------------ 9. buy-stop entries

@pytest.mark.parametrize("phrase,level", [
    ("at a stop 1% above the close", "close * 1.01"),
    ("at a stop above the high", "high"),
    ("with a buy stop at yesterday's high", "high"),
    ("on a stop order at the high", "high"),
    ("at a limit 2% below the close", "close * 0.98"),
    ("at a limit at the low", "low"),
    ("on a stop at the 20 day high", "highest(high, 20)"),
])
def test_entry_orders(phrase, level):
    s = sig(f"buy AAPL {phrase} when it is down 3 days in a row, hold 2 days")
    assert s.entry_level == level and s.entry_order == ("limit" if "limit" in phrase else "stop")
    assert s.stop_loss is None


def test_stop_entry_and_stop_loss_together():
    s = sig("buy AAPL at a stop 1% above the close when it is down 3 days in a row, hold 2 days, 5% stop loss")
    assert (s.entry_order, s.entry_level, s.stop_loss) == ("stop", "close * 1.01", 0.05)


# ------------------------------------------------------------------ 10. signal phrases

@pytest.mark.parametrize("text,field,want", [
    ("buy TQQQ when QQQ closes above its 200 day moving average, sell TQQQ when QQQ closes below its 200 day moving average",
     "exit_when", '(sym("QQQ").close < sma(sym("QQQ").close, 200))'),
    ("buy SPY when it is down 3 days in a row, not on Fridays, hold 2 days", "entry", "(down_days >= 3) and (dow != 4)"),
    ("buy SPY when it is up 2 days in a row except in October, hold 2 days", "entry", "(month != 10) and (up_days >= 2)"),
    ("short SPY when RSI(2) is above 90, buy to cover after 2 days", "hold_bars", 2),
    ("buy SPY when it is below the lower band, sell at the middle band", "exit_when", "(close > sma(close, 20))"),
    ("buy SPY when it is below the lower band, sell when it closes above the middle band", "exit_when", "(close > sma(close, 20))"),
    ("buy SPY when VIX is above its 10 day average, hold 5 days", "entry", '(sym("^VIX").close > sma(sym("^VIX").close, 10))'),
    ("buy SPY when volume is above its 20 day average, hold 2 days", "entry", "(volume > sma(volume, 20))"),
    ("buy SPY when ROC(10) is above 5, hold 5 days", "entry", "(ret(close, 10) > 0.05)"),
    ("buy SPY when the 10 day rate of change is above 5%, hold 5 days", "entry", "(ret(close, 10) > 0.05)"),
    ("buy SPY when the rate of change crosses above 0, hold 5 days", "entry", "(crossover(ret(close, 9), 0))"),
    ("buy SPY when MACD histogram turns negative, hold 5 days", "entry", "(crossunder(macd_hist(), 0))"),
    ("buy SPY when %K crosses above %D, hold 5 days", "entry", "(crossover(stoch_k(14, 3), stoch_d(14, 3, 3)))"),
    ("buy SPY when stochastic %K crosses below %D, hold 5 days", "entry", "(crossunder(stoch_k(14, 3), stoch_d(14, 3, 3)))"),
    ("buy SPY when it closes above the previous day's high, hold 5 days", "entry", "(close > ref(high, 1))"),
    ("buy SPY when it is above yesterday's high, hold 5 days", "entry", "(close > ref(high, 1))"),
    ("buy SPY when it closes below the prior day's low, hold 5 days", "entry", "(close < ref(low, 1))"),
    ("buy SPY when the weekly RSI is above 50, hold 5 days", "entry", "(weekly_rsi(14) > 50)"),
    ("buy SPY when weekly RSI(14) crosses above 50, hold 5 days", "entry", "(crossover(weekly_rsi(14), 50))"),
    ("buy SPY when the monthly RSI(10) is below 30, hold 5 days", "entry", "(monthly_rsi(10) < 30)"),
    ("buy SPY when it is above the monthly 10 SMA, hold 5 days", "entry", "(close > monthly_sma(10))"),
    ("buy SPY when it is above its 20 week EMA, hold 5 days", "entry", "(close > weekly_ema(20))"),
    ("buy SPY when it is above its weekly 20 EMA, hold 5 days", "entry", "(close > weekly_ema(20))"),
    ("buy SPY when QQQ is above its 10 month moving average, hold 5 days", "entry", '(sym("QQQ").close > monthly_sma(10, sym("QQQ").close))'),
])
def test_signal_phrases(text, field, want):
    assert getattr(sig(text), field) == want


def test_exit_naming_another_ticker_is_refused():
    with pytest.raises(parser.ParseError, match="not what the strategy trades"):
        parser.parse("buy TQQQ when QQQ closes above its 200 day moving average, sell SPY when QQQ closes below its 200 day moving average")


def test_bare_k_crosses_d_is_refused():
    with pytest.raises(parser.ParseError, match="which way"):
        parser.parse("buy SPY when stochastic %K crosses %D, hold 5 days")


def test_n_week_rsi_is_refused_as_ambiguous():
    with pytest.raises(parser.ParseError, match="weekly RSI"):
        parser.parse("buy SPY when the 14 week RSI is above 50, hold 5 days")


@pytest.mark.parametrize("text", ["buy UVXY and hold", "buy and hold UVXY", "buy UVXY and hold it, starting with $5000"])
def test_buy_and_hold_is_an_allocation(text):
    p = port(text)
    assert p.tree == {"asset": "UVXY"} and p.rebalance == "none"
    assert p.description == text


def test_buy_and_hold_does_not_swallow_signal_sentences():
    assert isinstance(parser.parse("buy SPY when it is down 3 days in a row and hold 2 days"), Strategy)


# ------------------------------------------------------------------ 11. higher-timeframe indicators

def _ns(df):
    return expr.Namespace(df)


HTF = ["weekly_rsi(14)", "monthly_rsi(3)", "weekly_ema(10)", "monthly_ema(3)", "weekly_sma(10)", "weekly_ret(2)", "monthly_ret(1)"]


@pytest.mark.parametrize("e", HTF)
def test_htf_indicators_do_not_look_ahead(e):
    df = data.load("SPY").loc["2015":"2020"]
    full = expr.evaluate_value(e, _ns(df))
    for cut in ["2017-03-15", "2018-06-29", "2019-11-27", "2020-04-09", "2020-07-02"]:
        part = df.loc[:cut]
        got = expr.evaluate_value(e, _ns(part))
        np.testing.assert_allclose(got.to_numpy(), full.loc[:cut].to_numpy(), equal_nan=True, err_msg=f"{e} cut at {cut}")


def test_weekly_value_changes_only_at_week_close():
    idx = pd.bdate_range("2021-01-04", periods=120)
    rng = np.random.default_rng(1)
    c = 100 * np.cumprod(1 + rng.normal(0, 0.01, len(idx)))
    df = pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1e6}, index=idx)
    w = expr.evaluate_value("weekly_rsi(5)", _ns(df))
    changes = w.diff().fillna(0).to_numpy() != 0
    assert all(idx[i].dayofweek == 4 for i in np.flatnonzero(changes))
    # the value at a Friday equals Wilder's RSI of the Friday closes up to that Friday
    fri = df["close"][df.index.dayofweek == 4]
    np.testing.assert_allclose(w[fri.index].to_numpy(), expr.rsi_wilder(fri, 5).to_numpy(), equal_nan=True)


def test_daily_indicator_of_weekly_close_is_refused():
    df = data.load("SPY").loc["2020"]
    with pytest.raises(ValueError, match="weekly_rsi"):
        expr.evaluate("rsi(weekly_close(), 14) > 50", _ns(df))
    expr.evaluate("close > weekly_close()", _ns(df))           # comparing with it is fine
    expr.evaluate("ref(weekly_close(), 1) > 0", _ns(df))


def test_htf_help_and_open_safety():
    assert "weekly_rsi" in expr.HELP and "monthly_ema" in expr.HELP
    assert not expr.open_safe("weekly_rsi(14) > 50")


# ------------------------------------------------------------------ portfolio phrases (Portfolio Visualizer)

def test_versus_t_bills_is_the_hurdle_not_ticker_t():
    p = port("dual momentum between SPY and EFA with AGG as the safe asset, using 12 month return versus T-bills")
    assert p.benchmark is None
    assert p.tree["filter"]["require"] == "tret(252) > tbill_ret(252)"
    with pytest.raises(parser.ParseError):
        parser.parse("hold 60% SPY and 40% TLT, against cash")


def test_sim_benchmark():
    assert port("hold 60% SPY and 40% TLT vs SPYSIM").benchmark == "SPYSIM"


@pytest.mark.parametrize("phrase", ["rebalance semi-annually", "rebalance twice a year", "rebalance every 6 months",
                                    "semiannual rebalancing"])
def test_semiannual(phrase):
    assert port(f"hold 60% SPY and 40% TLT, {phrase}").rebalance == "semiannual"


def test_bands():
    p = port("hold 60% SPY and 40% TLT, rebalance quarterly or when any weight drifts more than 5%")
    assert (p.rebalance, p.drift_band, p.drift_band_relative) == ("quarterly", 0.05, None)
    p = port("hold 60% SPY and 40% TLT, rebalance when a weight drifts 25% relative to its target")
    assert (p.rebalance, p.drift_band, p.drift_band_relative) == ("none", None, 0.25)
    with pytest.raises(parser.ParseError):
        parser.parse("hold 60% SPY and 40% TLT, rebalance every 5 months")


def test_cash_flow_schedules():
    p = port("hold 60% SPY and 40% TLT, add $1,000 a month for 20 years, then withdraw $50,000 a year")
    assert (p.contribution, p.contribution_freq, p.contribution_end) == (1000, "monthly", 20)
    assert (p.withdrawal, p.withdrawal_freq, p.withdrawal_start) == (50000, "yearly", 21)
    p = port("hold 60% SPY and 40% AGG, withdraw $40,000 a year starting in 2000, starting with $1,000,000")
    assert (p.withdrawal_start, p.start, p.capital) == (2000, None, 1_000_000)
    p = port("hold 60% SPY and 40% AGG, add $500 a month, contributions growing 3% a year")
    assert (p.contribution, p.contribution_growth) == (500, 0.03)
    p = port("hold 60% SPY and 40% AGG, withdraw 4% a year from year 10, starting with $1,000,000")
    assert (p.withdrawal_pct, p.withdrawal_start) == (0.04, 10)
    p = port("hold 60% SPY and 40% AGG, add $500 a month until 2010")
    assert p.contribution_end == "2010-12-31"
    p = port("hold 60% SPY and 40% AGG, withdraw $30,000 a year after 5 years, growing 2% a year, starting with $500,000")
    assert (p.withdrawal_start, p.withdrawal_growth) == (6, 0.02)
    with pytest.raises(parser.ParseError):
        parser.parse("hold 60% SPY and 40% AGG, then withdraw $50,000 a year")


def test_high_portfolio_leverage_needs_a_margin_setting():
    with pytest.raises(parser.ParseError, match="maintenance margin"):
        parser.parse("hold 60% SPY and 40% TLT with 5x leverage")
    p = port("hold 60% SPY and 40% TLT with 5x leverage, with a 15% maintenance margin")
    assert (p.leverage, p.maintenance_margin) == (5, 0.15)


# ------------------------------------------------------------------ strictness regressions found along the way

def test_move_window_in_exit_is_not_dropped():
    s = sig("buy SPY when it has fallen more than 5% over the last 10 days, sell when it has risen 3% in 2 days")
    assert s.exit_when == "(ret(close, 2) >= 0.03)" and s.hold_bars is None


def test_conflicting_holding_periods_are_refused():
    with pytest.raises(parser.ParseError, match="Two holding periods"):
        parser.parse("buy SPY when RSI(2) < 10, hold 5 days, sell when RSI(2) > 70 or after 10 days")
    s = sig("buy SPY when RSI(2) < 10, sell when RSI(2) > 70 or after 10 days")
    assert (s.exit_when, s.hold_bars) == ("(rsi(close, 2) > 70)", 10)


def test_if_hold_without_otherwise_says_so():
    with pytest.raises(parser.ParseError, match="otherwise"):
        parser.parse("if SPY is above its 200 day moving average hold QQQ")


def test_two_lookbacks_are_refused():
    with pytest.raises(parser.ParseError, match="two different lookbacks"):
        parser.parse("if TQQQ 5 day return over the last 10 days is above 1% then BIL else TQQQ")
