"""Round 10 (Composer expert review): windows are validated, never rewritten (tree conditions, filters, JSON); the
English parser's unit and range checks apply to every tree (signed max drawdown, oscillator ranges, impossible
returns); 'and' binds tighter than 'or' with a note when a sentence mixes them, and brackets group; bracketed groups
naming tickers and 'neither ... nor ...'; threshold / corridor / band phrasings and bands that can never trigger;
filter requirements on any indicator; 'is exactly' / 'equals'; the real Composer export format (a compound condition
with its mirrored single-comparison fields)."""
import ast
import json
from pathlib import Path

import pytest

from backtester import composer_export as ce, composer_import as ci, data, expr, parser
from backtester.parser import ParseError
from backtester.portfolio import Portfolio

AVAIL = set(data.available_tickers())
needs = lambda *ts: pytest.mark.skipif(not set(ts) <= AVAIL, reason="price data not downloaded")  # noqa: E731
FIX = Path(__file__).parent / "fixtures"


def norm(rule: str) -> str:
    return ast.unparse(ast.parse(rule.strip(), mode="eval"))


def port(text):
    p = parser.parse(text)
    assert isinstance(p, Portfolio), text
    return p


def if_tree(rule, on="SPY"):
    return Portfolio(tree={"if": rule, "on": on, "then": {"asset": "QQQ"}, "else": {"asset": "TLT"}})


# ------------------------------------------------------------ 1. windows: refused, never rewritten

@needs("SPY", "QQQ", "TLT")
@pytest.mark.parametrize("rule", ["tret(tr, -5) > 0", "rsi(close, -10) > 50", "rsi(close, 0) > 50", "rsi(close, 2.5) > 50",
                                  "sma(close, 0.5) < close", "macd(12, 0) > 0", "bb_upper(0, 2) < close",
                                  'rsi(sym("QQQ").close, -3) > 70'])
def test_bad_windows_in_tree_conditions_are_refused(rule):
    with pytest.raises(ValueError, match="positive whole number of days"):
        if_tree(rule).validate()


@needs("SPY", "QQQ", "TLT")
def test_bad_windows_in_filters_and_requirements_are_refused():
    base = {"universe": "children", "children": [{"asset": "QQQ"}, {"asset": "SPY"}, {"asset": "TLT"}]}
    for f in ({"select": "top", "n": 1, "by": "tret(tr, -5)"},
              {"select": "top", "n": 1, "by": "tret(tr, 2.5)"},
              {"select": "top", "n": 1, "by": "tret(tr, 21)", "require": "rsi(close, 0) > 50"}):
        with pytest.raises(ValueError, match="positive whole number of days"):
            Portfolio(tree={"filter": f, **base}).validate()
    for bad in (2.5, 0, -1):
        with pytest.raises(ValueError, match="whole number"):
            Portfolio(tree={"filter": {"select": "top", "n": bad, "by": "tret(tr, 21)"}, **base}).validate()
    with pytest.raises(ValueError, match="whole number"):
        Portfolio(tree={"weights": "inverse_vol", "lookback": 20.5, "children": base["children"]}).validate()


@needs("SPY", "QQQ", "TLT")
def test_good_windows_and_multipliers_pass():
    for rule in ("bb_upper(20, 2.5) < close", "macd(12, 26) > macd_signal(12, 26, 9)", "ppo(12, 26) > 0",
                 "stdev(close, 20) > 1", 'bb_lower(20, 2, sym("QQQ").close) > sym("QQQ").close'):
        p = if_tree(rule)
        p.validate()


def test_check_windows_directly():
    expr.check_windows("tret(tr, 10) > 0 and rsi(close, 14) < 30")
    with pytest.raises(ValueError, match="-5"):
        expr.check_windows("tret(tr, 10) > 0 and ret(close, -5) < 0")
    expr.check_windows("ref(close, 0) > 1")     # an offset of 0 is fine (it is not a window)


# ------------------------------------------------------------ 2. unit and range checks on any tree

@needs("SPY", "QQQ", "TLT")
@pytest.mark.parametrize("rule, want, deeper", [
    ("max_drawdown(tr, 10) > -0.1", "max_drawdown(tr, 10) < 0.1", False),
    ("max_drawdown(tr, 10) < -0.1", "max_drawdown(tr, 10) > 0.1", True),
    ("-0.2 >= max_drawdown(tr, 63)", "max_drawdown(tr, 63) >= 0.2", True),
    ("close > sma(close, 200) and max_drawdown(tr, 10) > -0.1", "close > sma(close, 200) and max_drawdown(tr, 10) < 0.1", False),
])
def test_signed_max_drawdown_is_normalised_with_a_note(rule, want, deeper):
    p = if_tree(rule)
    p.validate()
    assert norm(p.tree["if"]) == norm(want)
    assert any("positive size" in n and ("deeper" if deeper else "shallower") in n for n in p.notes), p.notes


@needs("SPY", "QQQ", "TLT")
@pytest.mark.parametrize("rule, msg", [
    ("rsi(close, 10) > 120", "0 to 100"),
    ("rsi(close, 10) < -5", "0 to 100"),
    ("rsi(close, 10) > 0.7", "fraction"),
    ("max_drawdown(tr, 10) > 1.5", "between 0 and 100%"),
    ("tret(tr, 10) < -1.2", "below -100%"),
])
def test_impossible_thresholds_are_refused_on_any_tree(rule, msg):
    with pytest.raises(ValueError, match=msg):
        if_tree(rule).validate()
    with pytest.raises(ValueError, match=msg):
        Portfolio(tree={"filter": {"select": "top", "n": 1, "by": "tret(tr, 21)", "require": rule}, "universe": "children",
                        "children": [{"asset": "QQQ"}, {"asset": "TLT"}]}).validate()


def test_the_parser_and_the_tree_share_one_range_check():
    assert parser.OSC_RANGES is expr.OSC_RANGES
    with pytest.raises(ParseError, match="0 to 100"):
        parser._check_ranges("rsi(close, 10) > 120", "RSI above 120")


# ------------------------------------------------------------ 4. and / or precedence, brackets

A, B, C = "SPY 10 day RSI is above 70", "QQQ 10 day RSI is above 70", "TLT 10 day RSI is below 30"
RA, RB, RC = "rsi(close, 10) > 70", 'rsi(sym("QQQ").close, 10) > 70', 'rsi(sym("TLT").close, 10) < 30'


@needs("SPY", "QQQ", "TLT", "UVXY", "TQQQ")
@pytest.mark.parametrize("cond, want, noted", [
    (f"{A} and {B} or {C}", f"({RA} and {RB}) or {RC}", True),
    (f"{A} or {B} and {C}", f"{RA} or ({RB} and {RC})", True),
    (f"({A}) and ({B}) or ({C})", f"({RA} and {RB}) or {RC}", True),
    (f"({A} and {B}) or {C}", f"({RA} and {RB}) or {RC}", False),
    (f"{A} and ({B} or {C})", f"{RA} and ({RB} or {RC})", False),
    (f"({A} or {B}) and {C}", f"({RA} or {RB}) and {RC}", False),
    (f"{A} and {B} and {C}", f"{RA} and {RB} and {RC}", False),
    (f"{A} or {B} or {C}", f"{RA} or {RB} or {RC}", False),
])
def test_and_binds_tighter_than_or_and_brackets_group(cond, want, noted):
    p = port(f"if {cond} then UVXY else TQQQ")
    assert p.tree["on"] == "SPY"
    assert norm(p.tree["if"]) == norm(want)
    has = any("binds tighter" in n for n in p.notes)
    assert has == noted, p.notes
    if noted:
        n = next(n for n in p.notes if "binds tighter" in n)
        assert "use brackets" in n and "read as " in n and "(" in n


@needs("SPY", "QQQ", "TQQQ")
def test_precedence_in_signal_rules():
    s = parser.parse("buy SPY when RSI(2) is below 10 and close is above sma(200) or RSI(2) is below 5, sell after 5 days")
    assert norm(s.entry) == norm("(rsi(close, 2) < 10 and close > sma(close, 200)) or rsi(close, 2) < 5")
    assert any("binds tighter" in n for n in s.notes)


@needs("SPY", "TQQQ", "BIL")
def test_a_bare_comparison_after_or_takes_the_subject():
    p = port("if SPY 10 day RSI is below 30 or above 70 then TQQQ else BIL")
    assert norm(p.tree["if"]) == norm("rsi(close, 10) < 30 or rsi(close, 10) > 70")


# ------------------------------------------------------------ 5. brackets naming tickers, neither / nor

@needs("SPY", "QQQ", "TLT", "UVXY", "TQQQ")
def test_bracketed_group_naming_tickers():
    p = port(f"if ({A} and {B}) or {C} then UVXY else TQQQ")
    assert norm(p.tree["if"]) == norm(f"({RA} and {RB}) or {RC}")
    p = port("if (both SPY and QQQ 10 day RSI are above 70) or TLT 10 day RSI is below 30 then UVXY else TQQQ")
    assert norm(p.tree["if"]) == norm(f"({RA} and {RB}) or {RC}")


@needs("SPY", "QQQ", "SMH", "TQQQ", "BIL")
def test_neither_nor():
    p = port("if neither SPY nor QQQ 10 day RSI is above 70 then TQQQ else BIL")
    assert norm(p.tree["if"]) == norm(f"not ({RA}) and not ({RB})")
    assert any("none of SPY, QQQ" in n for n in p.notes)
    p = port("if none of SPY, QQQ and SMH 10 day RSI is above 70 then TQQQ else BIL")
    assert norm(p.tree["if"]) == norm(f'not ({RA}) and not ({RB}) and not (rsi(sym("SMH").close, 10) > 70)')
    p = port("if neither SPY 10 day RSI is above 70 nor QQQ is below its 200 day moving average then TQQQ else BIL")
    assert norm(p.tree["if"]) == norm(f'not ({RA}) and not (sym("QQQ").close < sma(sym("QQQ").close, 200))')
    with pytest.raises(ParseError, match="neither SPY nor QQQ"):
        parser.parse("if neither SPY or QQQ 10 day RSI is above 70 then TQQQ else BIL")


# ------------------------------------------------------------ 6. threshold phrasing

@needs("TQQQ", "SQQQ")
@pytest.mark.parametrize("phrase", ["threshold rebalance 5%", "5% threshold rebalancing", "with a 5% rebalance corridor",
                                    "rebalance band 5%", "5% corridor", "threshold 5%", "rebalance when weights drift 5%"])
def test_threshold_phrasings(phrase):
    p = port(f"60% TQQQ 40% SQQQ, {phrase}")
    assert p.drift_band == pytest.approx(0.05)
    assert p.rebalance == "none"


@needs("SPY", "TLT")
def test_bands_that_can_never_trigger_are_warned():
    for pct in (150, 50):
        p = port(f"50% SPY 50% TLT, rebalance when weights drift {pct}%")
        assert any(n.startswith("Warning:") and "never trigger" in n for n in p.notes), p.notes
    p = port("50% SPY 50% TLT, rebalance when weights drift 49%")
    assert not any("never trigger" in n for n in p.notes)
    p = port("60% SPY 40% TLT, rebalance when weights drift 200% relative")
    assert any("never trigger" in n for n in p.notes)
    p = port("60% SPY 40% TLT, rebalance when weights drift 25% relative")
    assert not any("never trigger" in n for n in p.notes)


# ------------------------------------------------------------ 7. smaller

@needs("TQQQ", "SOXL", "TECL", "BIL")
def test_filter_requirement_on_any_indicator():
    p = port("hold the top 1 of TQQQ, SOXL and TECL by 10 day cumulative return only if their 20 day RSI is above 50")
    assert norm(p.tree["filter"]["require"]) == norm("rsi(close, 20) > 50")
    assert p.tree["filter"]["by"] == "tret(tr, 10)"
    p = port("hold the top 1 of TQQQ, SOXL and TECL by 10 day cumulative return, only if their 20 day RSI is above 50 "
             "and their 10 day return is positive, otherwise BIL")
    assert norm(p.tree["filter"]["require"]) == norm("rsi(close, 20) > 50 and tret(tr, 10) > 0")
    assert p.tree["fallback"] == {"asset": "BIL"}
    with pytest.raises(ParseError, match="0 to 100"):
        parser.parse("hold the top 1 of TQQQ, SOXL and TECL by 10 day cumulative return, only if their 20 day RSI is above 150")


@needs("SPY", "TQQQ", "BIL")
@pytest.mark.parametrize("word", ["is exactly", "equals", "is equal to"])
def test_equality(word):
    p = port(f"if SPY 10 day RSI {word} 50 then TQQQ else BIL")
    assert norm(p.tree["if"]) == norm("rsi(close, 10) == 50")
    assert any("exact equality rarely holds" in n for n in p.notes)
    p = port("if SPY 10 day RSI is greater than or equal to 50 then TQQQ else BIL")
    assert norm(p.tree["if"]) == norm("rsi(close, 10) >= 50")


def test_old_format_negative_window_message():
    sym = {"step": "root", "name": "x", "rebalance": "daily", "children": [{"step": "filter", "sort-by-fn": "cumulative-return",
           "sort-by-window-days": "-5", "select-fn": "top", "select-n": "1",
           "children": [{"step": "asset", "ticker": "SPY"}, {"step": "asset", "ticker": "QQQ"}]}]}
    with pytest.raises(ci.ComposerImportError, match="must be a positive whole number of days"):
        ci.convert(sym)


@needs("SPY", "QQQ", "SMH", "UVXY", "TQQQ")
def test_no_double_space_in_the_any_of_note():
    p = port("if any of SPY, QQQ and SMH has a 10 day RSI above 79 then UVXY else TQQQ")
    n = next(n for n in p.notes if "at least one of" in n)
    assert "  " not in n and "SMH 10 day RSI" in n


# ------------------------------------------------------------ 9. the real Composer export format

REAL = FIX / "composer_frontrunner_2026.json"


def _walk(b):
    yield b
    for k in b.get("children") or []:
        yield from _walk(k)


def test_a_real_export_imports():
    """tests/fixtures/composer_frontrunner_2026.json: https://backtest-api.composer.trade/api/v1/public/symphonies/
    4aI4kVT5cEc0XJpTLei3/score (as shown by https://composeratlas.com/converter), fetched Sept 2026. It carries
    "suppress_incomplete_warnings" (underscores, no '?') and a compound condition beside mirrored single fields."""
    d = ci.convert(json.loads(REAL.read_text()))
    rules = []

    def walk(n):
        if isinstance(n, dict):
            if "if" in n:
                rules.append((n["on"], n["if"]))
            for k in ("then", "else", "fallback"):
                walk(n.get(k))
            for k in n.get("children") or []:
                walk(k)
    walk(d["tree"])
    anyof = next(r for on, r in rules if " or " in r)
    assert anyof.count(" or ") == 11 and 'sym("XLY")' in anyof
    assert ("SMH", "rsi(close, 10) < 23") in rules


def test_export_writes_condition_blocks_as_composer_does():
    """The compound if-child in the real export: {"condition": {"condition-type": "compound", "operator": "any",
    "conditions": [{"condition-type": "binary", "lhs": {"fn", "ticker", "params": {"window": 10}}, "comparator": "gt",
    "rhs": {"constant": 79.0}}, ...]}} plus lhs-fn / lhs-window-days / lhs-val / comparator / rhs-val / rhs-fixed-value?
    repeating the last comparison. The export writes the same keys."""
    real = json.loads(REAL.read_text())
    real_child = next(b for b in _walk(real) if b.get("condition"))
    spec = {"kind": "allocation", "rebalance": "daily", "tree": {
        "if": 'rsi(close, 10) > 79 or rsi(sym("QQQ").close, 21) < 30', "on": "SMH",
        "then": {"asset": "VXX"}, "else": {"asset": "TQQQ"}}}
    sym, _ = ce.export(spec)
    child = next(b for b in _walk(sym) if b.get("condition"))
    legacy = {"lhs-fn", "lhs-window-days", "lhs-val", "comparator", "rhs-val", "rhs-fixed-value?"}
    assert legacy <= set(real_child) and legacy <= set(child)
    c = child["condition"]
    assert c["condition-type"] == "compound" and c["operator"] == "any"
    shapes = {json.dumps(sorted(x)) for x in real_child["condition"]["conditions"] if x["condition-type"] == "binary"}
    assert {json.dumps(sorted(x)) for x in c["conditions"]} <= shapes
    assert set(c["conditions"][0]["lhs"]) == set(real_child["condition"]["conditions"][0]["lhs"])
    assert child["lhs-val"] == "QQQ" and child["lhs-window-days"] == "21" and child["rhs-val"] == "30"
    assert child["comparator"] == "lt" and child["rhs-fixed-value?"] is True
    # and it imports back to the same rule
    back = ci.convert(sym)
    assert norm(back["tree"]["if"]) == norm(spec["tree"]["if"])


def test_export_binary_compound_mirror_uses_the_last_ticker():
    spec = {"kind": "allocation", "rebalance": "daily", "tree": {
        "if": 'rsi(close, 10) > 79 or rsi(sym("QQQ").close, 10) > 79', "on": "SPY",
        "then": {"asset": "VXX"}, "else": {"asset": "TQQQ"}}}
    sym, _ = ce.export(spec)
    child = next(b for b in _walk(sym) if b.get("condition"))
    assert child["condition"]["condition-type"] == "binary-compound"
    assert child["lhs-val"] == "QQQ" and child["rhs-val"] == "79"
