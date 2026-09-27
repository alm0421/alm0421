"""Round 4 parser fixes (Composer, Portfolio Visualizer, TradingView and QuantConnect reviews): distance
from a high/low, volatility thresholds, crossing notes, error messages, Composer phrasing, default
rebalancing, named model portfolios, TradingView phrases, backticks and parse-time validation."""
import itertools

import pytest

from backtester import data, library, parser
from backtester.parser import ParseError
from backtester.portfolio import Portfolio
from backtester.strategy import Strategy

HAVE = {"SPY", "QQQ", "TQQQ", "BIL", "UVXY", "TECL", "SOXL", "SVIX", "TLT", "AAPL", "AMZN", "VTI", "VBR", "SHY", "GLD",
        "SPYSIM", "TLTSIM", "SHYSIM", "VBRSIM", "GLDSIM"} <= set(data.available_tickers())
pytestmark = pytest.mark.skipif(not HAVE, reason="price data not downloaded")


def sig(text) -> Strategy:
    s = parser.parse(text)
    assert isinstance(s, Strategy), text
    return s


def port(text) -> Portfolio:
    p = parser.parse(text)
    assert isinstance(p, Portfolio), text
    return p


# ------------------------------------------------------------------ 1. distance from a high / low

DOWN_FORMS = ["is down", "has fallen", "fell", "dropped", "has dropped", "declined", "is off", "falls", "trades", "is",
              "has declined", "is trading", "sits"]
UP_FORMS = ["rallied", "bounced", "is up", "has risen", "rose", "has rallied", "is off", "is", "trades", "has bounced"]
SPANS = [("its 200 day", 200), ("its 10 week", 50), ("its 52 week", 252), ("its all-time", None), ("its all time", None),
         ("its", None), ("the 20 day", 20), ("its 3 month", 63)]


def _dd(n):
    return f"drawdown(close{'' if n is None else f', {n}'})"


def _up(n):
    return f"close / {'cummin(close)' if n is None else f'lowest(close, {n})'} - 1"


HIGH_CASES = [(v, p, sp, n) for v, p, (sp, n) in itertools.product(DOWN_FORMS, ["from", "off", "below"], SPANS)
              if not (v in ("is", "trades", "sits", "is trading") and p == "from")]
LOW_CASES = [(v, p, sp, n) for v, p, (sp, n) in itertools.product(UP_FORMS, ["from", "off", "above"], SPANS)]


@pytest.mark.parametrize("verb,prep,span,n", HIGH_CASES)
def test_distance_below_a_high_signal(verb, prep, span, n):
    s = sig(f"buy SPY when it {verb} 10% {prep} {span} high, hold 5 days")
    assert s.entry == f"({_dd(n)} <= -0.1)", (verb, prep, span, s.entry)


@pytest.mark.parametrize("verb,prep,span,n", HIGH_CASES[::3])
def test_distance_below_a_high_portfolio(verb, prep, span, n):
    p = port(f"if QQQ {verb} 10% {prep} {span} high then TQQQ else QQQ")
    assert p.tree["if"] == f"({_dd(n)} <= -0.1)" and p.tree["on"] == "QQQ", (verb, prep, span, p.tree["if"])


@pytest.mark.parametrize("verb,prep,span,n", LOW_CASES)
def test_distance_above_a_low_signal(verb, prep, span, n):
    s = sig(f"buy SPY when it {verb} 10% {prep} {span} low, hold 5 days")
    assert s.entry == f"({_up(n)} >= 0.1)", (verb, prep, span, s.entry)


@pytest.mark.parametrize("verb,prep,span,n", LOW_CASES[::3])
def test_distance_above_a_low_portfolio(verb, prep, span, n):
    p = port(f"if QQQ {verb} 10% {prep} {span} low then TQQQ else BIL")
    assert p.tree["if"] == f"({_up(n)} >= 0.1)", (verb, prep, span, p.tree["if"])


def test_the_reported_misparse():
    p = port("if QQQ is down 10% from its 200 day high then TQQQ else QQQ")
    assert p.tree["if"] == "(drawdown(close, 200) <= -0.1)"
    # no spurious 'and (change < 0)' from the verb
    assert sig("buy SPY when it falls 20% below its 52 week high, hold 10 days").entry == "(drawdown(close, 252) <= -0.2)"
    assert sig("buy SPY when it has fallen more than 20% from its 52 week high, hold 10 days").entry == "(drawdown(close, 252) < -0.2)"


def test_distance_from_another_tickers_high():
    s = sig("buy TQQQ when QQQ is down 10% from its 200 day high, hold 5 days")
    assert s.entry == '(drawdown(sym("QQQ").close, 200) <= -0.1)'


def test_within_a_low_and_a_high():
    assert sig("buy SPY when it is within 2% of its 52 week high, hold 5 days").entry == "(drawdown(close, 252) >= -0.02)"
    assert sig("buy SPY when it is within 3% of its 20 day low, hold 5 days").entry == "(close / lowest(close, 20) - 1 <= 0.03)"


@pytest.mark.parametrize("text", [
    "buy SPY when it rose 10% from its 200 day high, hold 5 days",
    "buy SPY when it is 10% above its 52 week high, hold 5 days",
    "buy SPY when it fell 10% from its 20 day low, hold 5 days",
    "buy SPY when it is 10% below its 20 day low, hold 5 days",
    "if QQQ has risen 10% from its all-time high then TQQQ else QQQ",
])
def test_impossible_directions_are_refused(text):
    with pytest.raises(ParseError, match="cannot be"):
        parser.parse(text)


# ------------------------------------------------------------------ 2. volatility thresholds

def test_small_bare_volatility_threshold_is_refused():
    for text in ("if SPY 10 day volatility is above 2% then BIL else SPY",
                 "buy SPY when its 20 day volatility is below 1%, hold 5 days"):
        with pytest.raises(ParseError, match="standard deviation of return.*annualized volatility"):
            parser.parse(text)


def test_volatility_kinds():
    assert port("if SPY 10 day daily volatility is above 2% then BIL else SPY").tree["if"] == "(stdev_return(tr, 10) > 0.02)"
    assert port("if SPY 10 day standard deviation of return is above 2% then BIL else SPY").tree["if"] == "stdev_return(tr, 10) > 0.02"
    assert port("if SPY 10 day annualized volatility is above 20% then BIL else SPY").tree["if"] == "(volatility(10) > 0.2)"
    assert port("if SPY 10 day annualised volatility is above 5% then BIL else SPY").tree["if"] == "(volatility(10) > 0.05)"
    p = port("if SPY 10 day volatility is above 20% then BIL else SPY")
    assert p.tree["if"] == "(volatility(10) > 0.2)" and any("annualised" in n for n in p.notes)
    # through the threshold grammar too ("the 10 day volatility of SPY")
    with pytest.raises(ParseError, match="daily or an annualised"):
        parser.parse("if the 10 day volatility of SPY is greater than 3% then BIL else SPY")


# ------------------------------------------------------------------ 3. crossings as allocation conditions

def test_cross_condition_in_an_if_node_has_a_note():
    p = port("if SPY crosses above its 200 day moving average then QQQ else BIL")
    assert p.tree["if"] == "(crossover(close, sma(close, 200)))"
    assert any("only on the day of the cross" in n and "is above" in n for n in p.notes)
    assert not any("day of the cross" in n for n in port("if SPY is above its 200 day moving average then QQQ else BIL").notes)


# ------------------------------------------------------------------ 4. error messages

def test_unknown_ticker_is_named_with_suggestions():
    if "ARKW" not in data.available_tickers():     # the data job may have downloaded it since
        with pytest.raises(ParseError, match=r"(?i)unknown ticker ARKW \(closest: ARKK"):
            parser.parse("hold ARKW and TQQQ")
    with pytest.raises(ParseError, match=r"(?i)unknown ticker TQQQX \(closest: TQQQ"):
        parser.parse("buy TQQQX when RSI(2) is below 10, hold 3 days")


def test_lists_and_weightings_route_to_portfolios():
    p = port("hold TQQQ, SOXL and TECL, weighted by inverse 10 day volatility")
    assert p.tree == {"weights": "inverse_vol", "lookback": 10,
                      "children": [{"asset": "TQQQ"}, {"asset": "SOXL"}, {"asset": "TECL"}]}
    assert port("hold ARKK and TQQQ").tree == {"weights": "equal", "children": [{"asset": "ARKK"}, {"asset": "TQQQ"}]}
    assert port("hold QQQ").tree == {"asset": "QQQ"}


def test_if_without_otherwise_says_so():
    with pytest.raises(ParseError, match="needs an 'otherwise'"):
        parser.parse("if SPY is above its 200 day moving average then TQQQ")


def test_nested_unknown_word_is_named():
    with pytest.raises(ParseError, match="glorp"):
        parser.parse("if SPY RSI(10) is above 70 then (if QQQ is above its 20 day moving average then TQQQ else BIL) else SPY with glorp")
    with pytest.raises(ParseError, match="glorp"):
        parser.parse("if SPY RSI(10) is above 70 then (if QQQ is above its 20 day moving average then TQQQ else BIL) else SPY, with glorp")


def test_otherwise_without_if():
    with pytest.raises(ParseError, match="'otherwise' with no 'if'"):
        parser.parse("hold 50% QQQ and 50% (TQQQ else BIL)")


# ------------------------------------------------------------------ 5. Composer phrasing

def test_lowercase_tickers():
    p = port("if tqqq 10 day rsi is above 79 then uvxy else tqqq")
    assert p.tree == {"if": "(rsi(close, 10) > 79)", "on": "TQQQ", "then": {"asset": "UVXY"}, "else": {"asset": "TQQQ"}}
    assert any("'tqqq' = TQQQ" in n for n in p.notes)
    assert port("hold 60% spy and 40% tlt").tree["children"] == [{"asset": "SPY"}, {"asset": "TLT"}]
    # English words that are also tickers are never read as tickers
    for w in ("all", "on", "it", "cat", "fast", "gold", "team", "tip", "has", "be"):
        assert w not in [x.lower() for x in parser.find_tickers(parser._lowercase_tickers(f"hold {w} spy"))]


def test_lowercase_never_touches_backticks():
    s = sig('buy QQQ when `sym("spy").close > 0`, hold 1 day')
    assert 'sym("spy")' in s.entry


def test_select_top_and_bottom():
    p = port("select the bottom 1 of TQQQ, SOXL and TECL by 10 day RSI")
    assert p.tree["filter"] == {"select": "bottom", "n": 1, "by": "rsi(close, 10)", "weights": "equal"}
    assert port("select top 2 of TQQQ, SOXL and TECL by 10 day cumulative return").tree["filter"]["n"] == 2


@pytest.mark.parametrize("text", [
    "hold TQQQ, SOXL and TECL weighted by inverse volatility over 10 days",
    "hold TQQQ, SOXL and TECL, weighted by inverse 10 day volatility",
    "hold TQQQ, SOXL and TECL, inverse volatility weighted over 10 days",
])
def test_inverse_volatility_phrasings(text):
    t = port(text).tree
    assert t["weights"] == "inverse_vol" and t["lookback"] == 10 and len(t["children"]) == 3


def test_inverse_n_day_volatility_on_a_filter():
    f = port("hold the top 2 of TQQQ, SOXL and TECL by 10 day return, weighted by inverse 20 day volatility").tree["filter"]
    assert f["weights"] == "inverse_vol" and f["lookback"] == 20


@pytest.mark.parametrize("phrase", ["greater than or equal to", "at least", "≥", ">=", "equal to or greater than"])
def test_greater_or_equal(phrase):
    assert port(f"if SPY 10 day RSI is {phrase} 70 then BIL else SPY").tree["if"].strip("()") == "rsi(close, 10) >= 70"


@pytest.mark.parametrize("phrase", ["less than or equal to", "at most", "≤"])
def test_less_or_equal(phrase):
    assert port(f"if SPY 10 day RSI is {phrase} 30 then SPY else BIL").tree["if"].strip("()") == "rsi(close, 10) <= 30"


def test_bare_n_day_means_moving_average():
    p = port("if SPY is above its 200-day then QQQ else BIL")
    assert p.tree["if"] == "(close > sma(close, 200))" and any("simple moving average" in n for n in p.notes)
    assert sig("buy SPY when it crosses above its 50 day, hold 5 days").entry == "(crossover(close, sma(close, 50)))"


@pytest.mark.parametrize("verb,op", [("outperformed", ">"), ("has outperformed", ">"), ("outperforms", ">"),
                                     ("beat", ">"), ("has underperformed", "<")])
def test_outperformed(verb, op):
    p = port(f"if QQQ {verb} TLT over 20 days then QQQ else TLT")
    assert p.tree["if"] == f'tret(tr, 20) {op} tret(sym("TLT").tr, 20)'


def test_both_tickers_above_their_averages():
    p = port("if both SPY and QQQ are above their 200 day moving averages then TQQQ else BIL")
    assert p.tree["if"] == '(close > sma(close, 200)) and (sym("QQQ").close > sma(sym("QQQ").close, 200))'


def test_drawdown_phrases():
    for text in ("if SPY drawdown over 10 days exceeds 20% then BIL else SPY",
                 "if SPY 10 day max drawdown exceeds 20% then BIL else SPY"):
        assert port(text).tree["if"] == "max_drawdown(tr, 10) > 0.2", text
    # a positive threshold on the (negative) current drawdown is read as its size
    p = port("if SPY 10 day drawdown is above 20% then BIL else SPY")
    assert p.tree["if"] == "drawdown(close, 10) < -0.2" and any("fall of more than 20%" in n for n in p.notes)


def test_fell_yesterday():
    s = sig("buy SPY when it fell more than 2% yesterday, hold 3 days")
    assert s.entry == "(ref(change, 1) < -0.02)" and any("previous bar" in n for n in s.notes)   # "more than": strict
    assert port("if SPY fell 2% yesterday then BIL else SPY").tree["if"] == "(ref(tret(tr, 1), 1) <= -0.02)"


def test_top_n_over_bracketed_groups():
    p = port("hold the top 1 of (60% TECL and 40% BIL), (SVIX) and (TQQQ) by 10 day cumulative return")
    assert p.tree == {"filter": {"select": "top", "n": 1, "by": "tret(tr, 10)", "weights": "equal"}, "universe": "children",
                      "children": [{"weights": "specified", "w": [0.6, 0.4], "children": [{"asset": "TECL"}, {"asset": "BIL"}]},
                                   {"asset": "SVIX"}, {"asset": "TQQQ"}]}
    p.validate()


def test_twelve_minus_one_momentum():
    p = port("hold the top 3 of SPY, QQQ, TLT and GLD by 12-1 momentum")
    assert p.tree["filter"]["by"] == "ref(tret(tr, 231), 21)" and any("12-1 momentum" in n for n in p.notes)


# ------------------------------------------------------------------ 6. default rebalancing

def test_default_rebalance():
    assert port("hold the top 2 of QQQ, SPY, TLT and GLD by 3 month return").rebalance == "daily"
    p = port("inverse volatility weighted SPY, TLT and GLD using a 60 day lookback")
    assert p.rebalance == "daily" and any("dynamic weights" in n for n in p.notes)
    assert port("risk parity SPY, TLT, GLD and DBC over 90 days").rebalance == "daily"
    assert port("if SPY is above its 200 day moving average then QQQ else BIL").rebalance == "daily"
    p = port("hold 60% SPY and 40% TLT")
    assert p.rebalance == "monthly" and any("fixed weights are rebalanced monthly" in n for n in p.notes)
    assert port("equal weight SPY, TLT and GLD").rebalance == "monthly"
    assert port("hold the top 2 of QQQ, SPY, TLT and GLD by 3 month return, rebalance monthly").rebalance == "monthly"


# ------------------------------------------------------------------ 7. named model portfolios

MODELS = [
    ("three fund portfolio", {"VTI": 0.64, "VXUS": 0.16, "BND": 0.2}),
    ("all weather portfolio", {"VTI": 0.3, "TLT": 0.4, "IEF": 0.15, "GLD": 0.075, "DBC": 0.075}),
    ("golden butterfly", {"VTI": 0.2, "VBR": 0.2, "TLT": 0.2, "SHY": 0.2, "GLD": 0.2}),
    ("permanent portfolio", {"VTI": 0.25, "TLT": 0.25, "BIL": 0.25, "GLD": 0.25}),
    ("coffeehouse portfolio", {"BND": 0.4, "SPY": 0.1, "VTV": 0.1, "VB": 0.1, "VBR": 0.1, "EFA": 0.1, "VNQ": 0.1}),
    ("ivy portfolio", {"VTI": 0.2, "VEU": 0.2, "BND": 0.2, "VNQ": 0.2, "DBC": 0.2}),
    ("Bernstein no-brainer", {"SPY": 0.25, "VB": 0.25, "VGK": 0.25, "SHY": 0.25}),
    ("60/40 portfolio", {"VTI": 0.6, "BND": 0.4}),
    ("Hedgefundie adventure", {"UPRO": 0.55, "TMF": 0.45}),
    ("Hedgefundie's excellent adventure", {"UPRO": 0.55, "TMF": 0.45}),
    ("Swensen portfolio", {"VTI": 0.3, "EFA": 0.15, "EEM": 0.05, "VNQ": 0.2, "TIP": 0.15, "IEF": 0.15}),
    ("Yale portfolio", {"VTI": 0.3, "EFA": 0.15, "EEM": 0.05, "VNQ": 0.2, "TIP": 0.15, "IEF": 0.15}),
    ("larry portfolio", {"VBR": 0.15, "VSS": 0.075, "VWO": 0.075, "IEF": 0.7}),
]


@pytest.mark.parametrize("name,weights", MODELS)
def test_model_portfolios(name, weights):
    if not set(weights) <= set(data.available_tickers()):
        pytest.skip("fund data missing")
    for text in (name, f"hold the {name}"):
        p = port(text)
        t = p.tree
        assert t["weights"] == "specified" and dict(zip([k["asset"] for k in t["children"]], t["w"])) == pytest.approx(weights)
        assert any(all(e in n for e in weights) for n in p.notes), p.notes   # the note lists the holdings
        p.validate()


def test_model_portfolio_uses_sims_before_the_funds():
    p = port("golden butterfly since 1972, rebalance yearly")
    assert [k["asset"] for k in p.tree["children"]] == ["SPYSIM", "VBRSIM", "TLTSIM", "SHYSIM", "GLDSIM"]
    assert p.rebalance == "yearly" and p.start == "1972-01-01"
    assert any("SPYSIM for VTI" in n and "GLDSIM for GLD" in n for n in p.notes)


def test_model_portfolio_without_a_sim_keeps_the_etf():
    p = port("all weather portfolio since 1980")
    kids = [k["asset"] for k in p.tree["children"]]
    if "DBCSIM" not in data.available_tickers():
        assert kids[-1] == "DBC" and any("No long-history series for DBC" in n for n in p.notes)
    assert kids[:4] == ["SPYSIM", "TLTSIM", "IEFSIM", "GLDSIM"]


def test_model_portfolio_nested():
    p = port("hold 50% golden butterfly and 50% QQQ")
    assert p.tree["w"] == [0.5, 0.5] and p.tree["children"][0]["weights"] == "specified"


def test_sim_table_is_extensible(monkeypatch):
    monkeypatch.setitem(parser.SIM_FOR, "DBC", [("NOPESIM", None)])
    kids = [k["asset"] for k in port("all weather portfolio since 1980").tree["children"]]
    assert kids[-1] == "DBC"   # a missing SIM falls back to the ETF


def test_gtaa_library_entry_is_timed():
    x = library.find("gtaa-5-faber")
    p = library.entry_spec(x)
    assert p.rebalance == "monthly" and p.tree["weights"] == "specified" and p.tree["w"] == [0.2] * 5
    for kid in p.tree["children"]:
        assert kid["if"] == "(close > monthly_sma(10))" and kid["then"] == {"asset": kid["on"]} and kid["else"] == {"cash": True}
    assert [k["on"] for k in p.tree["children"]] == ["SPY", "EFA", "IEF", "VNQ", "DBC"]


# ------------------------------------------------------------------ TradingView phrases

@pytest.mark.parametrize("text,entry,exit_", [
    ("buy QQQ when it's up 3 days in a row, hold 2 days", "(up_days >= 3)", None),
    ("buy QQQ when it isn't above its 200 day moving average, hold 2 days", "(close <= sma(close, 200))", None),
    ("buy QQQ when it doesn't close above its 200 day moving average, hold 2 days", "(close <= sma(close, 200))", None),
    ("buy QQQ when +DI crosses above -DI, sell when -DI crosses above +DI", "(crossover(plus_di(), minus_di()))",
     "(crossover(minus_di(), plus_di()))"),
    ("buy QQQ when -DI is below +DI, hold 2 days", "(minus_di() < plus_di())", None),
    ("buy AMZN when MACD histogram turns positive, sell when it turns negative", "(crossover(macd_hist(), 0))",
     "(crossunder(macd_hist(), 0))"),
    ("buy QQQ when the 14 day ROC crosses above 0, sell when it crosses below 0", "(crossover(ret(close, 14), 0))",
     "(crossunder(ret(close, 14), 0))"),
    ("buy QQQ when the close crosses above the upper Keltner channel, hold 5 days", "(crossover(close, keltner_upper(20, 2)))", None),
    ("buy QQQ when the close crosses above the upper Bollinger band, hold 5 days", "(crossover(close, bb_upper(20, 2)))", None),
    ("buy QQQ when Parabolic SAR flips below price, sell when Parabolic SAR flips above price", "(crossover(close, sar()))",
     "(crossunder(close, sar()))"),
    ("buy QQQ when it crosses above the parabolic SAR, hold 5 days", "(crossover(close, sar()))", None),
    ("buy QQQ when OBV crosses above its 20 day SMA, hold 5 days", "(crossover(obv(), sma(obv(), 20)))", None),
    ("buy QQQ when daily RSI(2) is below 10, hold 5 days", "(rsi(close, 2) < 10)", None),
    ("buy QQQ when the weekly close crosses above the weekly 20 EMA, hold 5 days", "(crossover(weekly_close(), weekly_ema(20)))", None),
    ("buy QQQ when the weekly MACD histogram turns positive, hold 5 days", "crossover(weekly(macd_hist()), 0)", None),
    ("buy QQQ when the weekly ATR(14) is above 10, hold 5 days", "weekly(atr(14)) > 10", None),
    ("buy QQQ when it closes above the weekly supertrend, hold 5 days", "close > weekly(supertrend(10, 3))", None),
    ("buy QQQ when bollinger %b is below 0, hold 5 days",
     "(((close - bb_lower(20, 2)) / (bb_upper(20, 2) - bb_lower(20, 2))) < 0)", None),
    ("buy QQQ when the close is 2 ATR below the 20 day EMA, hold 5 days", "(close < ema(close, 20) - 2 * atr(14))", None),
    ("buy QQQ when the close breaks above the highest high of the last 20 days, hold 5 days", "(close > ref(highest(high, 20), 1))", None),
    ("buy QQQ at the open when it opens above yesterday's high, sell at the close", "(open > ref(high, 1))", None),
    ("buy QQQ when %K crosses above %D below 20, hold 5 days",
     "(crossover(stoch_k(14, 3), stoch_d(14, 3, 3)) and (stoch_k(14, 3) < 20))", None),
    ("buy QQQ when they gap down, hold 1 day", "(gap < 0)", None),
    ("buy QQQ when it fell on Friday, hold 1 day", "(dow == 4) and (change < 0)", None),
])
def test_tradingview_phrases(text, entry, exit_):
    s = sig(text)
    assert s.entry == entry, s.entry
    if exit_:
        assert s.exit_when == exit_, s.exit_when
    s.validate()


def test_position_size_each():
    assert sig("buy Nasdaq 100 stocks when RSI(2) is below 5, hold 3 days, 50% each").position_size == 0.5
    assert sig("buy Nasdaq 100 stocks when RSI(2) is below 5, hold 3 days, 50% per position").position_size == 0.5


def test_ticks_slippage_is_refused_and_cents_are_dollars():
    with pytest.raises(ParseError, match="ticks"):
        parser.parse("buy QQQ when RSI(2) is below 10, hold 3 days, slippage 2 ticks")
    assert sig("buy QQQ when it gaps down 1%, hold 1 day, 1 cent per share commission").commission_per_share == 0.01


def test_intraday_times_are_refused():
    for text in ("buy SPY at 10am when it is down 1%, hold 1 day", "buy SPY at 15:30 when it is down 1%, hold 1 day",
                 "buy SPY on the 5 minute chart when RSI(2) is below 10, hold 1 day"):
        with pytest.raises(ParseError, match="intraday"):
            parser.parse(text)


def test_turns_needs_one_indicator():
    with pytest.raises(ParseError, match="what turns negative"):
        parser.parse("buy QQQ when it is above its 50 day moving average, sell when it turns negative")


# ------------------------------------------------------------------ backticks are never rewritten

def test_traded_ticker_inside_backticks_is_kept():
    s = sig('buy QQQ at the close when `sym("QQQ").close > sma(sym("QQQ").close, 50)`, hold 5 days')
    assert s.entry == '(sym("QQQ").close > sma(sym("QQQ").close, 50))'
    s = sig('buy AAPL when `sym("AAPL").close > 100` and Apple is above its 50 day moving average, hold 5 days')
    assert s.entry == '(sym("AAPL").close > 100) and (close > sma(close, 50))'
    s = sig('buy QQQ when RSI(2) is below 10, sell when `rsi(close, 5) > 70`')
    assert s.exit_when == "(rsi(close, 5) > 70)"


# ------------------------------------------------------------------ the dry run validates like the run

@pytest.mark.parametrize("text,msg", [
    ("buy QQQ when RSI(2) is below 10, hold 3 days, since 2020 until 2019", "reversed"),
    ("hold 60% SPY and 40% TLT, from 2021 to 2020", "reversed"),
    ("buy QQQ when RSI(2) is below 10, hold 3 days, since 2090", "after the last date"),
    ("buy QQQ when `ref(close, -1) > close`, hold 3 days", "negative offsets"),
    ("buy QQQ when `sma(close, 0) > close`, hold 3 days", "at least 1"),
    ("if SPY `rsi(close, 0) > 50` then QQQ else BIL", "at least 1"),
])
def test_parse_time_validation(text, msg):
    with pytest.raises(ParseError, match=msg):
        parser.parse(text)
