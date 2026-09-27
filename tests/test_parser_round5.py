"""Round 5 parser fixes (Composer, Portfolio Visualizer and TradingView reviews): indicator lookbacks written
after the indicator ("the RSI over 10 days"), one comparison per condition, drawdown signs, indicator ranges,
ranges and between, distance from a moving average, least / most volatile, rising / falling, politeness,
subject words of a signal, 4% rule and cash-flow phrases, tactical model rebalancing."""
import ast
import itertools

import pytest

from backtester import data, parser
from backtester.parser import ParseError
from backtester.portfolio import Portfolio
from backtester.strategy import Strategy

HAVE = {"SPY", "QQQ", "TQQQ", "BIL", "UVXY", "TLT", "IEF", "EFA", "VNQ", "DBC", "GLD", "AGG"} <= set(data.available_tickers())
pytestmark = pytest.mark.skipif(not HAVE, reason="price data not downloaded")


def sig(text) -> Strategy:
    s = parser.parse(text)
    assert isinstance(s, Strategy), text
    return s


def port(text) -> Portfolio:
    p = parser.parse(text)
    assert isinstance(p, Portfolio), text
    return p


def comparisons(rule: str) -> list[tuple[str, str, str]]:
    """Every comparison / crossing in a rule as (left, op, right)."""
    out = []
    for n in ast.walk(ast.parse(rule, mode="eval")):
        if isinstance(n, ast.Compare):
            left = n.left
            for op, right in zip(n.ops, n.comparators):
                out.append((ast.unparse(left), type(op).__name__, ast.unparse(right)))
                left = right
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ("crossover", "crossunder"):
            out.append((ast.unparse(n.args[0]), n.func.id, ast.unparse(n.args[1])))
    return out


# ------------------------------------------------------------------ 1. lookback after the indicator

# name as written, expected call for period 7, a sensible threshold
OSCS = [
    ("RSI", "rsi(close, 7)", "79"),
    ("relative strength index", "rsi(close, 7)", "30"),
    ("stochastic", "stoch_k(7, 1)", "80"),
    ("CCI", "cci(7)", "100"),
    ("Williams %R", "willr(7)", "-80"),
    ("MFI", "mfi(7)", "20"),
    ("money flow index", "mfi(7)", "80"),
    ("ADX", "adx(7)", "25"),
]
PHRASINGS = [
    "the 7 day {x}", "{x}(7)", "the {x} over 7 days", "the {x} for the last 7 days", "the 7-day {x}",
    "the {x} of {t} over 7 days", "{t}'s 7 day {x}", "the {x} over the past 7 sessions", "{t} {x} over 7 days",
    "the {x} across the last 7 trading days", "its {x} during the past 7 days", "the {x} over the last 7 bars",
]
CMPS = [("is above", "Gt"), ("is greater than", "Gt"), ("is over", "Gt"), ("is higher than", "Gt"), ("is below", "Lt"),
        ("is less than", "Lt"), ("is under", "Lt"), ("is lower than", "Lt"), ("is at least", "GtE"), ("is at most", "LtE"),
        (">", "Gt"), ("<", "Lt"), (">=", "GtE"), ("<=", "LtE")]
FUZZ = list(itertools.product(OSCS, PHRASINGS, CMPS))


def _one_comparison(rule, call, op, value, text):
    cs = comparisons(rule)
    assert cs == [(call, op, value)], (text, rule)


def _port_subject(phr, name):
    subject = phr.format(x=name, t="QQQ")
    if "QQQ" not in subject:   # name the ticker: "QQQ's 7 day RSI", "QQQ's RSI over 7 days"
        subject = "QQQ's " + subject.removeprefix("the ").removeprefix("its ")
    return subject


@pytest.mark.parametrize("osc,phr", list(itertools.product(OSCS, PHRASINGS)))
def test_indicator_lookback_fuzz_conditions(osc, phr):
    """Every oscillator x period phrasing x comparison, in both engines' condition contexts (fast: the
    condition parser only); the full sentences are sampled below."""
    name, call, v = osc
    for words, op in CMPS:
        cond = f"{_port_subject(phr, name)} {words} {v}"
        _one_comparison(parser.parse_conditions(parser._normalize(cond), ["QQQ"], total=True), call, op, v, cond)
        cond = f"{phr.format(x=name, t='SPY')} {words} {v}"
        _one_comparison(parser.parse_conditions(parser._normalize(cond), ["SPY"]), call, op, v, cond)


@pytest.mark.parametrize("osc,phr,cmp", FUZZ[::5])
def test_indicator_lookback_fuzz_portfolio(osc, phr, cmp):
    name, call, v = osc
    words, op = cmp
    text = f"if {_port_subject(phr, name)} {words} {v} then hold UVXY else hold TQQQ"
    p = port(text)
    assert p.tree["on"] == "QQQ"
    _one_comparison(p.tree["if"], call, op, v, text)


@pytest.mark.parametrize("osc,phr,cmp", FUZZ[2::7])
def test_indicator_lookback_fuzz_signal(osc, phr, cmp):
    name, call, v = osc
    words, op = cmp
    subject = phr.format(x=name, t="SPY")
    text = f"buy SPY when {subject} {words} {v}, hold 3 days"
    _one_comparison(sig(text).entry, call, op, v, text)


@pytest.mark.parametrize("text,rule", [
    ("if the RSI of QQQ over 10 days is greater than 79 then hold UVXY else hold TQQQ", "rsi(close, 10) > 79"),
    ("if QQQ RSI over 10 days is above 79 then hold UVXY else hold TQQQ", "rsi(close, 10) > 79"),
    ("if the stochastic of QQQ over 14 days is above 80 then hold UVXY else hold TQQQ", "stoch_k(14, 1) > 80"),
    ("if the ADX of QQQ over 20 days is above 25 then hold UVXY else hold TQQQ", "adx(20) > 25"),
    ("if the Williams %R of QQQ over 10 days is below -80 then hold UVXY else hold TQQQ", "willr(10) < -80"),
    ("if the MFI of QQQ over 10 days is below 20 then hold UVXY else hold TQQQ", "mfi(10) < 20"),
    ("if the CCI of QQQ over 20 days is above 100 then hold UVXY else hold TQQQ", "cci(20) > 100"),
])
def test_reported_lookback_misreads_portfolio(text, rule):
    assert port(text).tree["if"].strip("()") == rule


@pytest.mark.parametrize("text,rule", [
    ("buy QQQ when its RSI over 2 days is below 10, hold 3 days", "rsi(close, 2) < 10"),
    ("buy SPY when the CCI over 20 days is above 100, hold 3 days", "cci(20) > 100"),
    ("buy SPY when the ADX over the past 14 sessions is above 25, hold 3 days", "adx(14) > 25"),
])
def test_reported_lookback_misreads_signal(text, rule):
    assert sig(text).entry.strip("()") == rule


def test_lookback_after_indicator_in_pair_comparison_and_ranking():
    p = port("if QQQ RSI over 10 days is above SPY RSI over 10 days then hold QQQ else hold SPY")
    assert p.tree["if"] == 'rsi(close, 10) > rsi(sym("SPY").close, 10)'
    p = port("the top 1 of QQQ, SPY and TLT by RSI over 10 days")
    assert p.tree["filter"]["by"] == "rsi(close, 10)"


@pytest.mark.parametrize("text", [
    "buy SPY when rsi over 10 is below 20, hold 3 days",                 # 'over 10' then 'below 20': two comparisons
    "if QQQ rsi over 10 is above 79 then hold UVXY else hold TQQQ",
    "buy SPY when the 10 day RSI over 20 days is below 20, hold 3 days",  # two lookbacks
    "buy SPY when RSI(10) over 20 days is below 20, hold 3 days",
    "buy SPY when the RSI over 10 days is above 79 days, hold 3 days",    # a unit on a threshold
    "buy SPY when the RSI is above 10 days, hold 3 days",
    "if QQQ RSI is over 10 days then hold UVXY else hold TQQQ",
    "buy SPY when the cci is above 100 over 20 days, hold 3 days",
])
def test_stray_numbers_and_units_are_refused(text):
    with pytest.raises(ParseError):
        parser.parse(text)


# ------------------------------------------------------------------ 2. drawdown signs and indicator ranges

@pytest.mark.parametrize("cmp,rule", [
    ("is below -20%", "max_drawdown(tr, 10) > 0.2"),
    ("is less than -20%", "max_drawdown(tr, 10) > 0.2"),
    ("is at most -20%", "max_drawdown(tr, 10) >= 0.2"),
    ("is above -20%", "max_drawdown(tr, 10) < 0.2"),
    ("is greater than -20%", "max_drawdown(tr, 10) < 0.2"),
    ("is at least -20%", "max_drawdown(tr, 10) <= 0.2"),
    ("is above 20%", "max_drawdown(tr, 10) > 0.2"),       # the positive size, as written
    ("is below 20%", "max_drawdown(tr, 10) < 0.2"),
])
def test_max_drawdown_negative_threshold_is_a_size(cmp, rule):
    p = port(f"if the 10 day max drawdown of TQQQ {cmp} then hold BIL else hold TQQQ")
    assert p.tree["if"] == rule
    if "-" in cmp:
        assert any("positive size" in n for n in p.notes)
    else:
        assert not any("positive size" in n for n in p.notes)


@pytest.mark.parametrize("cmp,rule,noted", [
    ("is below -20%", "drawdown(close, 10) < -0.2", False),   # drawdown() is <= 0: as written
    ("is above -20%", "drawdown(close, 10) > -0.2", False),
    ("is above 20%", "drawdown(close, 10) < -0.2", True),     # a positive size: flipped, with a note
    ("is below 20%", "drawdown(close, 10) > -0.2", True),
])
def test_current_drawdown_signs(cmp, rule, noted):
    p = port(f"if TQQQ's 10 day drawdown {cmp} then hold BIL else hold TQQQ")
    assert p.tree["if"] == rule
    assert any("drawdown() is zero or negative" in n for n in p.notes) == noted


@pytest.mark.parametrize("text,hint", [
    ("buy SPY when RSI(2) is below 0.1, hold 3 days", "did you mean 10?"),
    ("if QQQ's 10 day RSI is above 0.7 then hold UVXY else hold TQQQ", "did you mean 70?"),
    ("buy SPY when RSI(2) is above 120, hold 3 days", "0 to 100"),
    ("buy SPY when RSI(2) is below -5, hold 3 days", "0 to 100"),
    ("buy SPY when the stochastic crosses above 120, hold 3 days", "0 to 100"),
    ("buy SPY when the 14 day MFI is above 101, hold 3 days", "0 to 100"),
    ("buy SPY when the ADX is above 150, hold 3 days", "0 to 100"),
    ("buy SPY when williams %r is below 80, hold 3 days", "did you mean -80?"),
    ("buy SPY when the 10 day williams %r is below -0.8, hold 3 days", "did you mean -80?"),
    ("buy SPY when williams %r is below -120, hold 3 days", "-100 to 0"),
    ("if the weekly RSI of QQQ is above 170 then hold UVXY else hold TQQQ", "0 to 100"),
    ("buy SPY when RSI(2) is below 10, sell when RSI(2) is above 0.7", "did you mean 70?"),
])
def test_oscillator_thresholds_out_of_range_are_refused(text, hint):
    with pytest.raises(ParseError, match=hint.replace("?", r"\?")):
        parser.parse(text)


@pytest.mark.parametrize("text,rule", [
    ("buy SPY when RSI(2) is below 1, hold 3 days", "(rsi(close, 2) < 1)"),
    ("buy SPY when RSI(2) is above 0, hold 3 days", "(rsi(close, 2) > 0)"),
    ("buy SPY when RSI(2) is below 100, hold 3 days", "(rsi(close, 2) < 100)"),
    ("buy SPY when williams %r is below -80, hold 3 days", "(willr(14) < -80)"),
    ("buy SPY when the CCI is below -250, hold 3 days", "(cci(20) < -250)"),     # CCI has no bounds
    ("buy SPY when `rsi(2) < 120`, hold 3 days", "(rsi(2) < 120)"),               # backticks are taken as written
])
def test_oscillator_thresholds_in_range(text, rule):
    assert sig(text).entry == rule


# ------------------------------------------------------------------ 3. ranges, distance from an average, rankings, trends

@pytest.mark.parametrize("text,rule", [
    ("buy SPY when the 10 day RSI is between 30 and 50, hold 3 days", "(rsi(close, 10) >= 30) and (rsi(close, 10) <= 50)"),
    ("buy SPY when the 10 day RSI is between 50 and 30, hold 3 days", "(rsi(close, 10) >= 30) and (rsi(close, 10) <= 50)"),
    ("buy SPY when it closes between 400 and 420, hold 3 days", "(close >= 400) and (close <= 420)"),
    ("buy SPY when RSI(2) is above 5 and below 10, hold 3 days", "(rsi(close, 2) > 5) and (rsi(close, 2) < 10)"),
    ("buy SPY when RSI(2) is below 10 and above 5 and the 50 day sma is above the 200 day sma, hold 3 days",
     "(rsi(close, 2) < 10) and (rsi(close, 2) > 5) and (sma(close, 50) > sma(close, 200))"),
])
def test_ranges_signal(text, rule):
    assert sig(text).entry == rule


@pytest.mark.parametrize("cond,rule", [
    ("QQQ's 10 day RSI is between 70 and 79", "(rsi(close, 10) >= 70) and (rsi(close, 10) <= 79)"),
    ("QQQ's 10 day RSI is above 79 and below 90", "(rsi(close, 10) > 79) and (rsi(close, 10) < 90)"),
    ("the 10 day RSI of QQQ is above 79 and is below 90", "(rsi(close, 10) > 79) and (rsi(close, 10) < 90)"),
    ("QQQ's 10 day return is between 30% and 50%", "(tret(tr, 10) >= 0.3) and (tret(tr, 10) <= 0.5)"),
    ("the RSI of QQQ over 10 days is between 70 and 79", "(rsi(close, 10) >= 70) and (rsi(close, 10) <= 79)"),
])
def test_ranges_portfolio(cond, rule):
    p = port(f"if {cond} then hold UVXY else hold TQQQ")
    assert p.tree["if"] == rule


@pytest.mark.parametrize("text", [
    "if QQQ's 10 day return is between 30% and 50 then hold UVXY else hold TQQQ",     # mixed units
    "buy SPY when the 10 day RSI is between 30 and 30, hold 3 days",
    "buy SPY when the 10 day RSI is between 30 and 150, hold 3 days",                 # out of range
    "buy SPY when below 90, hold 3 days",                                              # no subject to inherit
])
def test_bad_ranges_are_refused(text):
    with pytest.raises(ParseError):
        parser.parse(text)


@pytest.mark.parametrize("phrase,rule", [
    ("is 5% above its 200 day moving average", "close >= sma(close, 200) * 1.05"),
    ("is at least 5% above its 200-day SMA", "close >= sma(close, 200) * 1.05"),
    ("is more than 5% below its 200-day SMA", "close < sma(close, 200) * 0.95"),     # "more than": strict
    ("is trading 10% below its 50 day ma", "close <= sma(close, 50) * 0.9"),
    ("closes 3% above its 20 day EMA", "close >= ema(close, 20) * 1.03"),
])
def test_percent_from_a_moving_average(phrase, rule):
    assert sig(f"buy SPY when it {phrase}, hold 3 days").entry == f"({rule})"
    assert port(f"if SPY {phrase} then hold QQQ else hold BIL").tree["if"] == f"({rule})"


@pytest.mark.parametrize("text,select,n,by", [
    ("the least volatile 2 of SPY, QQQ, TLT and GLD", "bottom", 2, "volatility(20)"),
    ("the 2 least volatile of SPY, QQQ, TLT and GLD over the last 60 days", "bottom", 2, "volatility(60)"),
    ("the most volatile 1 of SPY, QQQ, TLT and GLD, rebalance monthly", "top", 1, "volatility(20)"),
    ("the 3 most volatile of SPY, QQQ, TLT and GLD over the past 3 months", "top", 3, "volatility(63)"),
    ("the worst performing 2 of SPY, QQQ, TLT and GLD over the last 60 days", "bottom", 2, "tret(tr, 60)"),
    ("the best performing 2 of SPY, QQQ, TLT and GLD over the last 3 months, rebalance monthly", "top", 2, "tret(tr, 63)"),
    ("the 2 best performing of SPY, QQQ, TLT and GLD over the last 3 months", "top", 2, "tret(tr, 63)"),
    ("the best performing 2 of SPY, QQQ, TLT and GLD over the last 60 days, inverse volatility weighted", "top", 2, "tret(tr, 60)"),
])
def test_least_most_volatile_and_best_worst_performing(text, select, n, by):
    p = port(text)
    f = p.tree["filter"]
    assert (f["select"], f["n"], f["by"]) == (select, n, by)
    assert [c["asset"] for c in p.tree["children"]] == ["SPY", "QQQ", "TLT", "GLD"]
    if "volatile" in text:
        assert any("annualised standard deviation" in x for x in p.notes)


@pytest.mark.parametrize("text", [
    "the least performing 2 of SPY, QQQ and TLT over the last 60 days",
    "the best volatile 2 of SPY, QQQ and TLT",
    "the 2 least volatile 3 of SPY, QQQ and TLT",
])
def test_bad_rankings_are_refused(text):
    with pytest.raises(ParseError):
        parser.parse(text)


@pytest.mark.parametrize("text,rule", [
    ("if the 200 day SMA of SPY is rising then hold QQQ else hold BIL", "sma(close, 200) > ref(sma(close, 200), 1)"),
    ("if SPY's 50 day moving average is falling then hold BIL else hold QQQ", "sma(close, 50) < ref(sma(close, 50), 1)"),
    ("if SPY is rising then hold QQQ else hold BIL", "close > ref(close, 1)"),
    ("if SPY is falling then hold BIL else hold QQQ", "close < ref(close, 1)"),
    ("if the 10 day RSI of QQQ is increasing then hold QQQ else hold BIL", "rsi(close, 10) > ref(rsi(close, 10), 1)"),
])
def test_rising_falling_portfolio(text, rule):
    p = port(text)
    assert p.tree["if"] == rule
    assert any("than on the previous day" in n for n in p.notes)


@pytest.mark.parametrize("text,rule", [
    ("buy SPY when the 200 day moving average is rising and RSI(2) is below 10, hold 3 days",
     "sma(close, 200) > ref(sma(close, 200), 1) and (rsi(close, 2) < 10)"),
    ("buy SPY when the 10 day RSI is falling, hold 3 days", "rsi(close, 10) < ref(rsi(close, 10), 1)"),
    ("buy SPY when its 50 day sma is sloping up, hold 3 days", "sma(close, 50) > ref(sma(close, 50), 1)"),
    ("buy SPY when the 20 day EMA is trending down, hold 3 days", "ema(close, 20) < ref(ema(close, 20), 1)"),
])
def test_rising_falling_signal(text, rule):
    assert sig(text).entry == rule


@pytest.mark.parametrize("text", [
    "buy SPY when the weather is rising, hold 3 days",
    "buy SPY when the volume is rising, hold 3 days",
])
def test_unknown_rising_is_refused(text):
    with pytest.raises(ParseError):
        parser.parse(text)


@pytest.mark.parametrize("text,rule", [
    ("buy SPY when it has a 10 day RSI above 70, hold 3 days", "(rsi(close, 10) > 70)"),
    ("buy SPY when it closes up with a 10 day RSI above 70, hold 3 days", "(change > 0) and (rsi(close, 10) > 70)"),
    ("buy SPY whenever it has an RSI(2) below 10, hold 3 days", "(rsi(close, 2) < 10)"),
])
def test_has_a_and_whose_signal(text, rule):
    assert sig(text).entry == rule


@pytest.mark.parametrize("cond", ["QQQ has a 10 day RSI above 70", "QQQ, whose 10 day RSI is above 70,",
                                  "QQQ whose 10 day RSI is above 70"])
def test_has_a_and_whose_portfolio(cond):
    assert port(f"if {cond} then hold UVXY else hold TQQQ").tree["if"] == "(rsi(close, 10) > 70)"


# ------------------------------------------------------------------ politeness is accepted; other filler is not

def test_please_is_accepted_explicitly():
    assert "please" in parser.STOP
    a = sig("please buy SPY when RSI(2) is below 10, hold 3 days")
    b = sig("buy SPY when RSI(2) is below 10, hold 3 days")
    assert (a.entry, a.hold_bars) == (b.entry, b.hold_bars)
    assert port("please hold 60% SPY and 40% TLT").tree == port("hold 60% SPY and 40% TLT").tree


# ("kindly" left this list in round 13: it is one of the documented politeness fillers, accepted everywhere)
@pytest.mark.parametrize("word", ["basically", "maybe", "definitely", "roughly", "never", "sometimes"])
@pytest.mark.parametrize("template", [
    "{w} buy SPY when RSI(2) is below 10, hold 3 days",
    "buy SPY {w} when RSI(2) is below 10, hold 3 days",
    "buy SPY when RSI(2) is {w} below 10, hold 3 days",
    "buy SPY when RSI(2) is below 10, hold 3 days {w}",
    "if QQQ's 10 day RSI is {w} above 79 then hold UVXY else hold TQQQ",
    "{w} hold 60% SPY and 40% TLT, rebalance monthly",
])
def test_unknown_filler_words_are_refused(word, template):
    assert word not in parser.STOP
    with pytest.raises(ParseError):
        parser.parse(template.format(w=word))


# ------------------------------------------------------------------ 4. words between the entry verb and 'when'

@pytest.mark.parametrize("text,fields", [
    ("buy SPY with 100 shares when RSI(2) is below 10, hold 3 days", {"sizing": "fixed_shares", "fixed_amount": 100.0}),
    ("buy 100 shares of SPY when RSI(2) is below 10, hold 3 days", {"sizing": "fixed_shares", "fixed_amount": 100.0}),
    ("buy $5000 of SPY when RSI(2) is below 10, hold 3 days", {"sizing": "fixed_dollars", "fixed_amount": 5000.0}),
    ("buy SPY with $2500 when RSI(2) is below 10, hold 3 days", {"capital": 2500.0}),   # "with $X" is the capital
    ("buy SPY with half my account when RSI(2) is below 10, hold 3 days", {"sizing": "percent", "position_size": 0.5}),
    ("buy SPY with 25% of the equity when RSI(2) is below 10, hold 3 days", {"sizing": "percent", "position_size": 0.25}),
    ("buy SPY for 3 days when RSI(2) is below 10", {"hold_bars": 3}),
    ("buy SPY for 2 weeks when RSI(2) is below 10", {"hold_bars": 10}),
    ("buy SPY and QQQ with 100 shares each when RSI(2) is below 10, hold 3 days",
     {"sizing": "fixed_shares", "fixed_amount": 100.0, "universe": ["SPY", "QQQ"]}),
    ("buy SPY at the next open when RSI(2) is below 10, hold 3 days", {"entry_fill": "next_open"}),
    ("please buy SPY when RSI(2) is below 10, hold 3 days", {"hold_bars": 3}),
])
def test_entry_subject_sizing_and_hold(text, fields):
    s = sig(text)
    assert s.entry == "(rsi(close, 2) < 10)"
    for k, v in fields.items():
        assert getattr(s, k) == v, (k, getattr(s, k))


@pytest.mark.parametrize("text,msg", [
    ("buy SPY but not QQQ when RSI(2) is below 10, sell when RSI(2) is above 70", "excluding tickers"),
    ("buy SPY except QQQ when RSI(2) is below 10, hold 3 days", "excluding tickers"),
    ("buy SPY on margin when RSI(2) is below 10, hold 3 days", "leverage"),
    ("buy SPY except during recessions when RSI(2) is below 10, hold 3 days", "except during recessions"),
    ("buy SPY aggressively when RSI(2) is below 10, hold 3 days", "aggressively"),
    # (round 13: "kindly" is a documented politeness filler now; "hastily" stands in as an unknown adverb)
    ("hastily buy SPY when RSI(2) is below 10, hold 3 days", "hastily"),
    ("buy SPY for 3 days when RSI(2) is below 10, hold 5 days", "Two holding periods"),
    ("buy 100 shares of SPY with half my account when RSI(2) is below 10, hold 3 days", "position size is given twice"),
    ("buy SPY with 100 shares and $500 of it when RSI(2) is below 10, hold 3 days", "two position sizes"),
])
def test_entry_subject_leftovers_are_refused(text, msg):
    with pytest.raises(ParseError, match=msg):
        parser.parse(text)


# ------------------------------------------------------------------ 5. Portfolio Visualizer phrasing

@pytest.mark.parametrize("text,wd", [
    ("hold 60% SPY and 40% AGG using the 4% rule", 400.0),
    ("hold 60% SPY and 40% AGG, 4% rule", 400.0),
    ("hold 60% SPY and 40% AGG with $100000, following the 4 percent rule", 4000.0),
    ("hold 60% SPY and 40% AGG, withdraw 4% annually adjusted for inflation", 400.0),
    ("hold 60% SPY and 40% AGG, withdraw 4% a year adjusted for inflation", 400.0),
    ("hold 60% SPY and 40% AGG with $100000, using the 3.5% rule", 3500.0),
    ("the 60/40 portfolio with $100000, withdraw 4% annually adjusted for inflation", 4000.0),
])
def test_four_percent_rule(text, wd):
    p = port(text)
    assert (p.withdrawal_pct, p.inflation_adjust, p.withdrawal_freq) == (0.0, True, "yearly")
    assert p.withdrawal == pytest.approx(wd)
    assert any("4% rule" in n for n in p.notes)


def test_percent_of_balance_is_not_the_rule():
    p = port("hold 60% SPY and 40% AGG, withdraw 4% of the balance annually")
    assert (p.withdrawal, p.withdrawal_pct, p.inflation_adjust) == (0.0, 0.04, False)


def test_sixty_forty_model_with_options():
    p = port("60/40 portfolio, rebalance yearly, since 2005")
    assert p.tree["w"] == [0.6, 0.4] and p.rebalance == "yearly" and p.start.startswith("2005")


@pytest.mark.parametrize("text,amount,freq", [
    ("hold 60% SPY and 40% AGG, add $500 monthly", 500.0, "monthly"),
    ("hold 60% SPY and 40% AGG, contribute $500 quarterly", 500.0, "quarterly"),
    ("hold 60% SPY and 40% AGG, invest $6000 annually", 6000.0, "yearly"),
    ("hold 60% SPY and 40% AGG, add $500 a month", 500.0, "monthly"),
])
def test_contribution_adverbs(text, amount, freq):
    p = port(text)
    assert (p.contribution, p.contribution_freq) == (amount, freq)


def test_contribution_growth():
    p = port("hold 60% SPY and 40% AGG, contribute $500 a month increasing 3% per year")
    assert (p.contribution, p.contribution_growth) == (500.0, 0.03)


@pytest.mark.parametrize("text,band,rel", [
    ("hold 60% SPY and 40% AGG, rebalance when a weight is off by 5 percentage points", 0.05, None),
    ("hold 60% SPY and 40% AGG, rebalance when any weight drifts by more than 5 pp", 0.05, None),
    ("hold 60% SPY and 40% AGG, rebalance with 25% relative bands", None, 0.25),
    ("hold 60% SPY and 40% AGG, with a 25% relative band", None, 0.25),
    ("hold 60% SPY and 40% AGG, rebalance with 5% bands", 0.05, None),
])
def test_drift_bands(text, band, rel):
    p = port(text)
    assert (p.drift_band, p.drift_band_relative, p.rebalance) == (band, rel, "none")


@pytest.mark.parametrize("text,n,safe", [
    ("dual momentum between SPYSIM and EFASIM with IEFSIM as the safe asset, 12 month lookback", 252, "IEFSIM"),
    ("dual momentum between SPYSIM and EFASIM with IEFSIM as the safe asset, using a 6 month lookback", 126, "IEFSIM"),
    ("dual momentum between SPY and EFA with AGG as the safe asset", 252, "AGG"),
])
def test_dual_momentum_lookbacks_and_monthly_default(text, n, safe):
    p = port(text)
    assert p.tree["filter"]["by"] == f"tret({n})" and p.tree["fallback"] == {"asset": safe}
    assert p.rebalance == "monthly"
    assert any("Dual momentum" in x and "month-end" in x for x in p.notes)


def test_tactical_models_respect_an_explicit_schedule():
    assert port("dual momentum between SPY and EFA with AGG as the safe asset, rebalance daily").rebalance == "daily"
    assert port("hold SPY when it is above its 10 month moving average, otherwise cash, rebalance daily").rebalance == "daily"


def test_ten_month_timing_defaults_to_monthly():
    p = port("hold SPY when it is above its 10-month moving average, otherwise cash")
    assert p.rebalance == "monthly" and any("Faber" in n for n in p.notes)
    # a daily-bar condition keeps the Composer default (checked every day)
    assert port("hold SPY when it is above its 200 day moving average, otherwise cash").rebalance == "daily"


@pytest.mark.parametrize("text", [
    "hold SPY, EFA, IEF, VNQ and DBC equally, each only when above its 10 month moving average, otherwise cash",
    "SPY, EFA, IEF, VNQ and DBC in equal weights, each only when it is above its 10 month moving average, otherwise cash",
    "hold SPY, EFA, IEF, VNQ and DBC, each only if above its 10 month moving average, else cash",
])
def test_per_asset_timing(text):
    p = port(text)
    kids = p.tree["children"]
    assert p.tree["weights"] == "equal" and [k["on"] for k in kids] == ["SPY", "EFA", "IEF", "VNQ", "DBC"]
    for k in kids:
        assert k["if"] == "(close > monthly_sma(10))" and k["then"] == {"asset": k["on"]} and k["else"] == {"cash": True}
    assert p.rebalance == "monthly"


def test_per_asset_timing_with_another_fallback():
    p = port("hold SPY, EFA and IEF equally, each only when its 12 month return is positive, otherwise BIL")
    assert [k["else"] for k in p.tree["children"]] == [{"asset": "BIL"}] * 3
    assert p.tree["children"][1]["if"] == "(tret(tr, 252) > 0)"


@pytest.mark.parametrize("text,tv,look", [
    ("hold 60% SPY and 40% AGG, target 10% volatility", 0.10, None),
    ("hold 60% SPY and 40% AGG, target 10% volatility using 60 day volatility", 0.10, 60),
    ("hold 60% SPY and 40% AGG with a 12% volatility target", 0.12, None),
])
def test_portfolio_target_volatility(text, tv, look):
    import dataclasses
    if "target_vol" not in {f.name for f in dataclasses.fields(Portfolio)}:
        with pytest.raises(ParseError, match="target_vol"):
            parser.parse(text)
        return
    p = port(text)
    assert p.target_vol == tv
    if look is not None:
        assert p.target_vol_lookback == look


# ------------------------------------------------------------------ 6. TradingView phrasing

@pytest.mark.parametrize("phrase", ["commission $1 per trade", "commission of $1 per trade", "commissions: $1 per trade",
                                    "$1 per trade commission", "with a commission of $1"])
def test_commission_word_order(phrase):
    assert sig(f"buy SPY when RSI(2) is below 10, hold 3 days, {phrase}").commission == 1.0


def test_commission_per_share_word_order():
    s = sig("buy SPY when RSI(2) is below 10, hold 3 days, commission $0.005 per share")
    assert s.commission_per_share == 0.005 and s.commission == 0.0


@pytest.mark.parametrize("text", [
    "buy QQQ every Monday at the open and sell Friday at the close",
    "buy QQQ on Mondays at the open and sell on Friday at the close",
    "buy QQQ every Monday at the open, sell every Friday at the close",
])
def test_weekday_entry_and_exit(text):
    s = sig(text)
    assert (s.entry, s.entry_fill, s.exit_when, s.exit_when_fill) == ("(dow == 0)", "open", "(dow == 4)", "close")


@pytest.mark.parametrize("text", [
    "buy SPY when RSI(2) is below 10, hold -3 days",
    "buy SPY when RSI(2) is below 10, sell after -2 days",
    "buy SPY when it is up 5% in the last -3 days, hold 3 days",
])
def test_negative_periods_are_refused(text):
    with pytest.raises(ParseError, match="positive number"):
        parser.parse(text)


@pytest.mark.parametrize("text,rule", [
    ("buy SPY when RSI(2) was below 10 within the last 5 bars, hold 3 days", "count(((rsi(close, 2) < 10)), 5) >= 1"),
    ("buy SPY when RSI(2) was below 10 at least once in the last 5 days, hold 3 days", "count(((rsi(close, 2) < 10)), 5) >= 1"),
    ("buy SPY when RSI(2) is below 10 for 2 consecutive days, hold 3 days", "count(((rsi(close, 2) < 10)), 2) == 2"),
    ("buy SPY when RSI(2) has been below 10 for 3 days in a row, hold 3 days", "count(((rsi(close, 2) < 10)), 3) == 3"),
    ("buy SPY when it has been above its 20 day moving average for the last 10 days, hold 3 days",
     "count(((close > sma(close, 20))), 10) == 10"),
    ("buy SPY when it is down 3 days in a row, hold 3 days", "(down_days >= 3)"),            # the native streak is unchanged
    ("buy SPY when it closed down for 3 consecutive days, hold 3 days", "count(((change < 0)), 3) == 3"),   # = down_days >= 3
])
def test_conditions_over_several_bars(text, rule):
    assert sig(text).entry == rule


def test_over_bars_in_a_portfolio_condition():
    p = port("if QQQ's 10 day RSI was above 79 within the last 5 days then hold UVXY else hold TQQQ")
    assert p.tree["if"] == "count(((rsi(close, 10) > 79)), 5) >= 1"


@pytest.mark.parametrize("text", [
    "buy SPY when RSI(2) was below 10 within the last 5 bars of fun, hold 3 days",
    "buy SPY when the frobnicator is below 10 for 2 consecutive days, hold 3 days",
])
def test_bad_over_bars_are_refused(text):
    with pytest.raises(ParseError):
        parser.parse(text)


@pytest.mark.parametrize("phrase,rule", [
    ("it closes 2 standard deviations below the 20 day average", "close < sma(close, 20) - 2 * stdev(close, 20)"),
    ("it is 2.5 std devs above its 50 day moving average", "close > sma(close, 50) + 2.5 * stdev(close, 50)"),
    ("it closes 1 sigma below the 10 day EMA", "close < ema(close, 10) - 1 * stdev(close, 10)"),
])
def test_standard_deviation_bands(phrase, rule):
    assert sig(f"buy SPY when {phrase}, hold 3 days").entry == f"({rule})"


@pytest.mark.parametrize("phrase,rule", [
    ("the supertrend flips bullish", "crossover(close, supertrend(10, 3))"),
    ("the supertrend flips to green", "crossover(close, supertrend(10, 3))"),
    ("the supertrend turns bearish", "crossunder(close, supertrend(10, 3))"),
    ("supertrend(7,2) flips up", "crossover(close, supertrend(7, 2))"),
    ("close crosses above supertrend(10,3)", "crossover(close, supertrend(10, 3))"),
    ("the close crosses below the supertrend", "crossunder(close, supertrend(10, 3))"),
    ("it closes above the supertrend", "close > supertrend(10, 3)"),
    ("the price is below supertrend(14, 2.5)", "close < supertrend(14, 2.5)"),
])
def test_supertrend(phrase, rule):
    assert sig(f"buy SPY when {phrase}, hold 3 days").entry == f"({rule})"


def test_supertrend_flip_needs_a_direction():
    with pytest.raises(ParseError, match="which way"):
        parser.parse("buy SPY when the supertrend flips, hold 3 days")


@pytest.mark.parametrize("phrase,rule", [
    ("the high is above yesterday's high", "high > ref(high, 1)"),
    ("today's low is below the previous day's low", "low < ref(low, 1)"),
    ("the close is above the prior close", "close > ref(close, 1)"),
    ("the open is below yesterday's close", "open < ref(close, 1)"),
    ("the high crosses above yesterday's high", "crossover(high, ref(high, 1))"),
])
def test_bar_fields_against_the_previous_bar(phrase, rule):
    assert sig(f"buy SPY when {phrase}, hold 3 days").entry == f"({rule})"


def test_share_class_with_a_dot():
    if "BRK-B" not in data.available_tickers():
        pytest.skip("no BRK-B data")
    s = sig("buy BRK.B when RSI(2) is below 10, hold 3 days")
    assert s.universe == ["BRK-B"] and any("BRK-B" in n for n in s.notes)
    assert sig("buy BRK-B when RSI(2) is below 10, hold 3 days").universe == ["BRK-B"]
    p = port("hold 50% BRK.B and 50% SPY")
    assert [c["asset"] for c in p.tree["children"]] == ["BRK-B", "SPY"]


@pytest.mark.parametrize("text,rule", [
    ("buy SPY when the ma over 50 days crosses above the ma over 200 days, hold 3 days", "(crossover(sma(close, 50), sma(close, 200)))"),
    ("buy SPY when the sma over 50 days is above the sma over 200 days, hold 3 days", "(sma(close, 50) > sma(close, 200))"),
    ("buy SPY when the ATR over 10 days is above 5, hold 3 days", "(atr(10) > 5)"),
    ("buy SPY when RSI is above 70 for the last 10 days, hold 3 days", "count(((rsi(close, 14) > 70)), 10) == 10"),
])
def test_lookback_after_other_indicators(text, rule):
    assert sig(text).entry == rule


@pytest.mark.parametrize("text", [
    "buy SPY when RSI is below 10 over 5 days, hold 3 days",
    "buy SPY when RSI(2) < 10 days, hold 3 days",
    "if QQQ 10 day RSI is above 79 days then hold UVXY else hold TQQQ",
    "buy SPY when the 10 day ATR is above 5 days, hold 3 days",
    "buy SPY when the vix is above 30 days, hold 3 days",
    "buy SPY when the 20 day high is above 5 days, hold 3 days",
    "if the 10 day return of QQQ over 20 days is above 5% then hold QQQ else hold BIL",
])
def test_units_on_thresholds_are_refused(text):
    with pytest.raises(ParseError):
        parser.parse(text)
