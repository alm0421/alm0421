"""Round 11 (Composer expert review): market cap over funds is refused everywhere (sentences, JSON trees, Composer
wt-marketcap), "falls below" / "rises above" are read one documented way per context (a crossing in signal rules,
the level in portfolio conditions, each with a note), contradictory or tautological bounds on one value are caught,
the RSI period note is not left behind by a discarded reading, the drift-band warning says what still trades, and
new phrasings (bounds joined by or, 'not below 30 and not above 70', 'not' scoping, number words, 'rebalance at 5%
corridor', a weighting's lookback after the ranking)."""
import pytest

from backtester import composer_import as ci
from backtester import data, expr, parser
from backtester import portfolio as pf
from backtester.strategy import Strategy

AVAIL = set(data.available_tickers())


def needs(*t):
    return pytest.mark.skipif(not set(t) <= AVAIL, reason="price data not downloaded")


# ------------------------------------------------------------ 3. market cap over funds

@needs("SPY", "QQQ", "TLT")
@pytest.mark.parametrize("text", ["market cap weighted SPY, QQQ and TLT", "SPY, QQQ and TLT, market cap weighted"])
def test_market_cap_weighting_of_etfs_is_refused_with_a_suggestion(text):
    with pytest.raises(parser.ParseError, match="funds have no market cap") as e:
        parser.parse(text)
    assert "inverse-volatility" in str(e.value) and "AUM" in str(e.value)


@needs("AAPL", "MSFT", "NVDA")
def test_market_cap_weighting_of_stocks_still_parses():
    p = parser.parse("market cap weighted AAPL, MSFT and NVDA")
    assert p.tree["weights"] == "market_cap"


@needs("SPY", "QQQ", "TLT", "AAPL")
@pytest.mark.parametrize("tree", [
    {"weights": "market_cap", "children": [{"asset": "SPY"}, {"asset": "QQQ"}]},
    {"filter": {"select": "top", "n": 1, "by": "market_cap", "weights": "equal"}, "universe": "children",
     "children": [{"asset": "SPY"}, {"asset": "AAPL"}, {"asset": "TLT"}]},
    {"filter": {"select": "top", "n": 2, "by": "tret(tr, 21)", "weights": "market_cap"}, "universe": "children",
     "children": [{"asset": "SPY"}, {"asset": "QQQ"}, {"asset": "TLT"}]},
    {"if": "market_cap > 1e12", "on": "SPY", "then": {"asset": "QQQ"}, "else": {"asset": "TLT"}},
])
def test_market_cap_over_funds_in_trees_is_refused(tree):
    with pytest.raises(ValueError, match="funds have no market cap"):
        pf.Portfolio(tree=tree).validate()


@needs("SPY", "QQQ")
def test_composer_wt_marketcap_over_etfs_is_refused_with_its_path():
    sym = {"step": "root", "name": "T", "rebalance": "monthly", "children": [
        {"step": "wt-marketcap", "children": [{"step": "asset", "ticker": "SPY"}, {"step": "asset", "ticker": "QQQ"}]}]}
    with pytest.raises(ci.ComposerImportError, match=r"symphony.*> 1.*market-cap weighting of SPY, QQQ") as e:
        ci.convert(sym)
    assert "inverse-volatility" in str(e.value)


# ------------------------------------------------------------ 4. the RSI period note

@needs("SPY", "UVXY", "TQQQ")
@pytest.mark.parametrize("text", ["if SPY 10 days RSI > 79 then UVXY else TQQQ", "if SPY 10 day RSI > 79 then UVXY else TQQQ",
                                  "if SPY RSI(10) is above 79 then UVXY else TQQQ"])
def test_no_spurious_rsi_period_note(text):
    p = parser.parse(text)
    assert "rsi(close, 10) > 79" in p.tree["if"]
    assert not any("No RSI period given" in n for n in p.notes), p.notes


@needs("SPY", "UVXY", "TQQQ")
def test_rsi_period_note_when_none_is_given():
    p = parser.parse("if SPY RSI > 79 then UVXY else TQQQ")
    assert "rsi(close, 14)" in p.tree["if"]
    assert any("No RSI period given" in n for n in p.notes)


# ------------------------------------------------------------ 5. falls below / rises above

@needs("SPY")
@pytest.mark.parametrize("text, rule", [
    ("buy SPY when RSI(2) falls below 10, sell after 5 days", "crossunder(rsi(close, 2), 10)"),
    ("buy SPY when RSI(2) drops below 10, sell after 5 days", "crossunder(rsi(close, 2), 10)"),
    ("buy SPY when the 10 day RSI rises above 70, sell after 5 days", "crossover(rsi(close, 10), 70)"),
    ("buy SPY when the price falls below its 200 day moving average, sell after 5 days", "crossunder(close, sma(close, 200))"),
    ("buy SPY when the price falls below 400, sell after 5 days", "crossunder(close, 400)"),
    ("buy SPY when the 20 day return falls below -5%, sell after 5 days", "crossunder(ret(close, 20), -0.05)"),
])
def test_signal_falls_below_is_a_crossing_with_a_note(text, rule):
    s = parser.parse(text)
    assert rule in s.entry
    assert any("read as a crossing" in n for n in s.notes), s.notes


@needs("SPY")
def test_signal_exit_rises_above_is_a_crossing():
    s = parser.parse("buy SPY when RSI(2) is below 10, sell when RSI(2) rises above 70")
    assert s.entry == "(rsi(close, 2) < 10)"
    assert "crossover(rsi(close, 2), 70)" in s.exit_when
    s = parser.parse("buy SPY when RSI(2) is below 10, sell when it rises above 70")
    assert "crossover(rsi(close, 2), 70)" in s.exit_when


@needs("SPY")
def test_crosses_below_a_return_level():
    s = parser.parse("buy SPY when the 20 day return crosses below -5%, sell after 5 days")
    assert "crossunder(ret(close, 20), -0.05)" in s.entry


@needs("SPY", "TQQQ", "BIL")
@pytest.mark.parametrize("text, rule, word", [
    ("if SPY 10 day RSI falls below 30 then TQQQ else BIL", "rsi(close, 10) < 30", "below"),
    ("if SPY 10 day RSI rises above 70 then BIL else TQQQ", "rsi(close, 10) > 70", "above"),
    ("if SPY 63 day return drops below -5% then BIL else TQQQ", "tret(tr, 63) < -0.05", "below"),
])
def test_portfolio_falls_below_is_the_level_with_a_note(text, rule, word):
    p = parser.parse(text)
    assert rule in p.tree["if"] and "cross" not in p.tree["if"]
    assert any(f"read as 'is {word}" in n and f"crosses {word}" in n for n in p.notes), p.notes


@needs("SPY", "TQQQ", "BIL")
def test_portfolio_crosses_below_is_still_a_crossing():
    p = parser.parse("if SPY 10 day RSI crosses below 30 then TQQQ else BIL")
    assert "crossunder(rsi(close, 10), 30)" in p.tree["if"]


# ------------------------------------------------------------ 6. contradictions and tautologies

@pytest.mark.parametrize("rule, const", [
    ("(rsi(close, 10) > 79) and (rsi(close, 10) < 30)", False),
    ("rsi(close, 10) > 30 and rsi(close, 10) < 70", None),
    ("tret(tr, 63) < 0.05 or tret(tr, 63) > -0.05", True),
    ("tret(tr, 63) < -0.05 or tret(tr, 63) > 0.05", None),
    ("70 < rsi(close, 2) < 30", False),
    ("x > 5 and x <= 5", False),
    ("x >= 5 and x <= 5", None),
    ("rsi(close, 10) > 79 and rsi(close, 14) < 30", None),       # different values: not compared
    ("not (x > 5 and x < 3)", True),
    ("close > sma(close, 200)", None),
])
def test_bound_conflicts(rule, const):
    c, msgs = expr.bound_conflicts(rule)
    assert c is const
    assert bool(msgs) == (const is not None)


@needs("SPY", "UVXY", "TQQQ")
def test_contradictory_portfolio_condition_warns():
    p = parser.parse("if SPY 10 day RSI is above 79 and below 30 then UVXY else TQQQ")
    assert any(n.startswith("Warning:") and "can never be true" in n and "otherwise branch is always held" in n for n in p.notes)
    p = parser.parse("if SPY 10 day RSI is below 30 or above 20 then UVXY else TQQQ")
    assert any(n.startswith("Warning:") and "always true" in n for n in p.notes)


@needs("SPY")
def test_always_false_entry_is_refused():
    with pytest.raises(parser.ParseError, match="can never be true"):
        parser.parse("buy SPY when RSI(2) is above 90 and below 10, sell after 5 days")
    with pytest.raises(ValueError, match="never trade"):
        Strategy(universe=["SPY"], entry="rsi(close, 2) > 90 and rsi(close, 2) < 10", hold_bars=5).validate()


@needs("SPY")
def test_always_true_entry_warns():
    s = Strategy(universe=["SPY"], entry="rsi(close, 2) < 60 or rsi(close, 2) > 40", hold_bars=5)
    s.validate()
    assert any(n.startswith("Warning:") and "always true" in n for n in s.notes)


# ------------------------------------------------------------ 7. the drift-band warning

@needs("SPY", "TQQQ", "BIL", "TLT")
def test_unreachable_band_on_a_dynamic_tree_says_what_still_trades():
    p = parser.parse("if SPY is above its 200 day moving average then TQQQ else BIL, rebalance when drift exceeds 150%")
    w = [n for n in p.notes if "can never trigger" in n]
    assert w and "only when the if/else switches" in w[0] and "never rebalanced after the first day" not in w[0]
    p = parser.parse("SPY and TLT, rebalance when drift exceeds 150%")
    assert any("never rebalanced after the first day" in n for n in p.notes)
    p = parser.parse("SPY and TLT, rebalance monthly and when drift exceeds 150%")
    assert any("only on its monthly schedule" in n for n in p.notes)


# ------------------------------------------------------------ 8. phrasings

@needs("SPY", "TQQQ", "BIL")
def test_or_of_two_bounds_on_one_indicator():
    p = parser.parse("if SPY 63 day return is less than -5% or greater than 5% then TQQQ else BIL")
    assert p.tree["if"] == "((tret(tr, 63) < -0.05) or (tret(tr, 63) > 0.05))"


@needs("SPY", "TQQQ", "BIL")
def test_not_below_and_not_above():
    p = parser.parse("if SPY 10 day RSI is not below 30 and not above 70 then TQQQ else BIL")
    assert p.tree["if"] == "(rsi(close, 10) >= 30) and (rsi(close, 10) <= 70)"


@needs("SPY", "QQQ", "UVXY", "TQQQ")
def test_not_scopes_to_the_next_condition_only():
    p = parser.parse("if not SPY 10 day RSI is above 79 or QQQ 10 day RSI is above 79 then UVXY else TQQQ")
    assert p.tree["if"] == '(not (rsi(close, 10) > 79) or (rsi(sym("QQQ").close, 10) > 79))'
    assert any("'not' applies to the condition right after it only" in n for n in p.notes)


@needs("SPY")
def test_not_before_a_calendar_phrase_keeps_its_reading():
    s = parser.parse("buy SPY when it is down 3 days in a row, not on Fridays, hold 2 days")
    assert s.entry == "(down_days >= 3) and (dow != 4)"


@needs("SPY")
@pytest.mark.parametrize("text", ["buy SPY when RSI(2) is above 10 and closes above 5, hold 2 days",
                                  "buy SPY when RSI(2) is above 10 and close > 5, hold 2 days"])
def test_closes_above_after_an_indicator_is_the_price(text):
    # "closes above 5" names the price: it does not continue the RSI comparison (it used to read as RSI > 5)
    assert parser.parse(text).entry == "(rsi(close, 2) > 10) and (close > 5)"


@needs("SPY", "UVXY", "TQQQ")
@pytest.mark.parametrize("text, rule", [
    ("if SPY 10 day RSI is above seventy nine then UVXY else TQQQ", "rsi(close, 10) > 79"),
    ("if SPY 10 day RSI is above seventy-nine then UVXY else TQQQ", "rsi(close, 10) > 79"),
    ("if SPY twenty day RSI is above 79 then UVXY else TQQQ", "rsi(close, 20) > 79"),
    ("if SPY is above its one hundred day moving average then UVXY else TQQQ", "close > sma(close, 100)"),
    ("if SPY is above its two hundred day moving average then UVXY else TQQQ", "close > sma(close, 200)"),
    ("if SPY ten day RSI is below forty then UVXY else TQQQ", "rsi(close, 10) < 40"),
])
def test_number_words(text, rule):
    assert rule in parser.parse(text).tree["if"]


def test_number_words_function():
    f = parser._number_words
    assert f("seventy nine") == "79" and f("a hundred and twenty") == "120" and f("ninety-five") == "95"
    assert f("`x > one`") == "`x > one`"          # the rule language in backticks is left alone


@needs("SPY", "TLT")
@pytest.mark.parametrize("text", ["SPY and TLT, rebalance at 5% corridor", "SPY and TLT, rebalance at a 5% threshold",
                                  "SPY and TLT, rebalance with a 5% corridor"])
def test_rebalance_at_corridor(text):
    p = parser.parse(text)
    assert p.rebalance == "none" and p.drift_band == pytest.approx(0.05)


@needs("SPY", "QQQ", "TLT", "GLD", "IEF")
@pytest.mark.parametrize("text", [
    "inverse volatility weighted top 3 of SPY, QQQ, TLT, GLD, IEF by 63 day return using a 10 day lookback",
    "top 3 of SPY, QQQ, TLT, GLD, IEF by 63 day return, inverse volatility weighted using a 10 day lookback",
])
def test_weighting_lookback_after_the_ranking(text):
    f = parser.parse(text).tree["filter"]
    assert f["by"] == "tret(tr, 63)" and f["weights"] == "inverse_vol" and f["lookback"] == 10


@needs("SPY", "QQQ", "TLT", "GLD", "IEF")
def test_lookback_without_a_weighting_that_uses_one_is_refused():
    with pytest.raises(parser.ParseError, match="has no lookback"):
        parser.parse("top 3 of SPY, QQQ, TLT, GLD, IEF by 63 day return using a 10 day lookback")
