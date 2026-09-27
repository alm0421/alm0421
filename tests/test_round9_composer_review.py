"""Round 9 (Composer expert review): conditions on several tickers keep every ticker (both / all / any / either, a
bare list refused), negative lookbacks are refused, Composer's corridor width is a fraction both ways, the importer
reads any/all condition blocks, MACD / PPO / Bollinger functions, eq, wt-marketcap and empty blocks (and the exporter
writes all / any blocks and empty), threshold rebalancing with if/else exports, report folders are unique, group
names show in the interpretation, and phrase gaps (52 week low, unless ... in which case, EMA(8) > SMA(21), N points
above, below its average by more than 5%, weighted 70/30 by rank)."""
import ast
import json

import numpy as np
import pandas as pd
import pytest

from backtester import composer_export as ce, composer_import as ci, data, expr, parser, portfolio as pf, report
from backtester.parser import ParseError
from backtester.portfolio import Portfolio

AVAIL = set(data.available_tickers())
needs = lambda *ts: pytest.mark.skipif(not set(ts) <= AVAIL, reason="price data not downloaded")  # noqa: E731


def port(text):
    p = parser.parse(text)
    assert isinstance(p, Portfolio), text
    return p


def norm(rule: str) -> str:
    """A rule without redundant brackets, for comparing two spellings of it."""
    return ast.unparse(ast.parse(rule.strip(), mode="eval"))


def tickers_in_rule(rule: str) -> set:
    import re
    return set(re.findall(r'sym\("([^"]+)"\)', rule))


# ------------------------------------------------------------ 1. several tickers in one condition

MULTI = [
    ("if both SPY and QQQ 10 day RSI are above 79 then UVXY else TQQQ", "and", ["QQQ"]),
    ("if any of SPY, QQQ and SMH has a 10 day RSI above 79 then UVXY else TQQQ", "or", ["QQQ", "SMH"]),
    ("if all of SPY, QQQ and SMH 10 day RSI are above 79 then UVXY else TQQQ", "and", ["QQQ", "SMH"]),
    ("if SPY and QQQ 10 day cumulative return is below -5% then UVXY else TQQQ", "and", ["QQQ"]),
    ("if either SPY or QQQ 10 day RSI is above 79 then UVXY else TQQQ", "or", ["QQQ"]),
    ("if SPY or QQQ 10 day RSI is above 79 then UVXY else TQQQ", "or", ["QQQ"]),
    ("if the 10 day RSI of both SPY and QQQ is above 79 then UVXY else TQQQ", "and", ["QQQ"]),
    ("if all of SPY, QQQ and SMH have a 10 day RSI above 79 then UVXY else TQQQ", "and", ["QQQ", "SMH"]),
    ("if any of SPY, QQQ and SMH 10 day RSI are above 79 then UVXY else TQQQ", "or", ["QQQ", "SMH"]),
]


@needs("SPY", "QQQ", "SMH", "UVXY", "TQQQ")
@pytest.mark.parametrize("text, op, others", MULTI)
def test_a_condition_on_several_tickers_keeps_every_ticker(text, op, others):
    p = port(text)
    rule, on = p.tree["if"], p.tree["on"]
    assert on == "SPY"
    assert tickers_in_rule(rule) == set(others)             # the other tickers via sym(), SPY as the rule's own
    assert expr.reads_own_series(rule)
    body = ast.parse(rule, mode="eval").body
    assert isinstance(body, ast.BoolOp) and isinstance(body.op, ast.And if op == "and" else ast.Or)
    assert len(body.values) == 1 + len(others)
    assert "(on SPY)" in p.summary()


@needs("SPY", "QQQ", "UVXY", "TQQQ")
def test_each_ticker_gets_the_same_comparison():
    p = port("if both SPY and QQQ 10 day RSI are above 79 then UVXY else TQQQ")
    assert norm(p.tree["if"]) == norm('rsi(close, 10) > 79 and rsi(sym("QQQ").close, 10) > 79')
    p = port("if SPY and QQQ 10 day cumulative return is below -5% then UVXY else TQQQ")
    assert norm(p.tree["if"]) == norm('tret(tr, 10) < -0.05 and tret(sym("QQQ").tr, 10) < -0.05')


@needs("SPY", "QQQ", "SMH", "UVXY", "TQQQ")
@pytest.mark.parametrize("text", [
    "if SPY, QQQ 10 day RSI is above 79 then UVXY else TQQQ",
    "if SPY, QQQ, SMH 10 day RSI is above 79 then UVXY else TQQQ",
    "if SPY and QQQ or SMH 10 day RSI is above 79 then UVXY else TQQQ",
])
def test_a_bare_list_asks_all_or_any(text):
    with pytest.raises(ParseError, match="all of .* or any of them"):
        parser.parse(text)


@needs("SPY", "QQQ", "TLT", "TQQQ", "BIL")
def test_groups_combine_with_other_clauses():
    p = port("if SPY is above its 200 day moving average and both SPY and QQQ 10 day RSI are below 79 then TQQQ else BIL")
    assert norm(p.tree["if"]) == norm('close > sma(close, 200) and (rsi(close, 10) < 79 and rsi(sym("QQQ").close, 10) < 79)')
    # an 'and' group inside an 'or' keeps its own brackets
    p = port("if TLT is above its 200 day moving average or both SPY and QQQ 10 day RSI are below 30 then TQQQ else BIL")
    assert norm(p.tree["if"]) == norm('close > sma(close, 200) or (rsi(sym("SPY").close, 10) < 30 and '
                                      'rsi(sym("QQQ").close, 10) < 30)')
    # a range for each ticker
    p = port("if any of SPY, QQQ and TLT 10 day RSI is between 30 and 50 then TQQQ else BIL")
    assert tickers_in_rule(p.tree["if"]) == {"QQQ", "TLT"} and p.tree["if"].count(">= 30") == 3


@needs("SPY", "QQQ", "TQQQ")
def test_groups_in_signal_rules():
    s = parser.parse("buy TQQQ when both SPY and QQQ 2 day RSI are below 10, sell after 5 days")
    assert norm(s.entry) == norm('rsi(sym("SPY").close, 2) < 10 and rsi(sym("QQQ").close, 2) < 10')
    s = parser.parse("buy TQQQ when either SPY or QQQ 2 day RSI is below 10, sell after 5 days")
    assert norm(s.entry) == norm('rsi(sym("SPY").close, 2) < 10 or rsi(sym("QQQ").close, 2) < 10')
    with pytest.raises(ParseError, match="or any of them"):
        parser.parse("buy TQQQ when SPY, QQQ 2 day RSI is below 10, sell after 5 days")


@needs("SPY", "QQQ")
def test_every_ticker_named_must_be_in_the_rule():
    # the general guard: a rule that lost a ticker is refused, whatever phrase produced it
    with pytest.raises(ParseError, match="QQQ"):
        parser._every_ticker_used("QQQ 10 day RSI is above 79", "rsi(close, 10) > 79", ["SPY"])
    with pytest.raises(ParseError, match="SPY"):
        parser._every_ticker_used("SPY is above 79", 'rsi(sym("QQQ").close, 10) > 79', ["SPY"])
    assert parser._every_ticker_used("SPY RSI is above 79", "rsi(close, 10) > 79", ["SPY"]) == "rsi(close, 10) > 79"
    assert parser._every_ticker_used("QQQ RSI is above 79", 'rsi(sym("QQQ").close, 10) > 79', ["SPY"])
    # a bare ticker with no comparison is never dropped
    with pytest.raises(ParseError, match="no condition was found for SPY"):
        parser.parse_conditions("SPY", ["TLT"], total=True)


def test_reads_own_series():
    assert expr.reads_own_series("rsi(10) > 79")
    assert expr.reads_own_series("close > sma(close, 200)")
    assert expr.reads_own_series("bb_lower(20, 2) > close")
    assert not expr.reads_own_series('rsi(sym("QQQ").close, 10) > 79')
    assert not expr.reads_own_series('sym("QQQ").close > sma(sym("QQQ").close, 50)')
    assert not expr.reads_own_series("dow == 4")


def test_the_on_label_names_the_rule_subject():
    lines = pf.describe({"if": 'rsi(sym("QQQ").close, 10) > 79', "on": "SPY", "then": {"asset": "UVXY"},
                         "else": {"asset": "TQQQ"}})
    assert lines[0] == 'if rsi(sym("QQQ").close, 10) > 79:'        # SPY is not read, so it is not named
    lines = pf.describe({"if": "rsi(close, 10) > 79", "on": "SPY", "then": {"asset": "UVXY"}, "else": {"asset": "TQQQ"}})
    assert lines[0] == "if rsi(close, 10) > 79 (on SPY):"


# ------------------------------------------------------------ 2. negative lookbacks

@needs("SPY", "BIL", "TQQQ", "QQQ", "TLT")
@pytest.mark.parametrize("text", [
    "if SPY -10 day RSI is above 70 then BIL else TQQQ",
    "if SPY -10 day return is above 5% then BIL else TQQQ",
    "if SPY is above its -200 day moving average then BIL else TQQQ",
    "if SPY -10-day RSI is above 70 then BIL else TQQQ",
    "hold the top 1 of SPY, QQQ, TLT by -12 month return",
    "buy SPY when its -10 day RSI is below 30, sell after 5 days",
    "buy SPY when SPY -10 day RSI is below 30, sell after 5 days",
    "buy SPY when RSI(-2) is below 10, sell after 5 days",
])
def test_negative_lookbacks_are_refused(text):
    with pytest.raises(ParseError, match="positive number of days"):
        parser.parse(text)


@needs("SPY", "BIL", "TQQQ")
def test_negative_thresholds_still_parse():
    p = port("if SPY 10 day return is below -5% then BIL else TQQQ")
    assert p.tree["if"].strip("()") == "tret(tr, 10) < -0.05"


# ------------------------------------------------------------ 3. the corridor width is a fraction, both ways

@needs("TQQQ", "TMF")
@pytest.mark.parametrize("pct, frac", [("0.5", 0.005), ("5", 0.05), ("10", 0.1)])
def test_corridor_width_round_trip(pct, frac):
    p = port(f"hold 50% TQQQ and 50% TMF, never rebalance, rebalance when any weight drifts {pct}% from target")
    assert p.drift_band == pytest.approx(frac)
    sym, _ = ce.export(p)
    assert sym["rebalance"] == "none" and sym["rebalance-corridor-width"] == pytest.approx(frac)
    back = Portfolio.from_dict(ci.convert(json.dumps(sym)))
    assert back.drift_band == pytest.approx(frac) and back.rebalance == "none"
    assert pf.without_ids(back.tree) == pf.without_ids(p.tree)
    assert f"drifts {pct}% from its target" in p.summary()          # 0.5% is not rounded to 0%


def test_corridor_width_import_is_a_fraction_with_no_guessing():
    root = lambda cw: {"step": "root", "rebalance": "none", "rebalance-corridor-width": cw,  # noqa: E731
                       "children": [{"step": "asset", "ticker": "SPY"}]}
    assert ci.convert(root(0.05))["drift_band"] == 0.05
    assert ci.convert(root(0.005))["drift_band"] == 0.005
    assert ci.convert(root("0.13"))["drift_band"] == 0.13
    for bad in (5, 1, 0, -0.05):
        with pytest.raises(ci.ComposerImportError, match="fraction"):
            ci.convert(root(bad))


# ------------------------------------------------------------ 4. Composer's current condition blocks and functions

def rsi_bin(t, v=79, cmp="gt", w=10):
    return {"condition-type": "binary", "lhs": {"fn": "relative-strength-index", "ticker": t, "params": {"window": w}},
            "comparator": cmp, "rhs": {"constant": v}}


def sym_if(cond_child: dict, then="UVXY", other=None) -> dict:
    other = other or {"step": "asset", "ticker": "TQQQ"}
    return {"step": "root", "name": "t", "rebalance": "daily", "children": [{"step": "if", "children": [
        {"step": "if-child", "is-else-condition?": False, **cond_child, "children": [{"step": "asset", "ticker": then}]},
        {"step": "if-child", "is-else-condition?": True, "children": [other]}]}]}


def test_import_compound_and_binary_compound_conditions():
    # as Composer's editor writes an "Any of" block (the stale single-condition fields stay beside it)
    cond = {"condition-type": "compound", "operator": "any", "conditions": [
        rsi_bin("SMH"), rsi_bin("QQQ"),
        {"condition-type": "binary-compound", "operator": "any", "tickers": ["VTV", "XLP"], "comparator": "gt",
         "lhs": {"fn": "relative-strength-index", "ticker": "%", "params": {"window": 10}}, "rhs": {"constant": 79}},
        {"condition-type": "compound", "operator": "all", "conditions": [rsi_bin("SPY", 30, "lt"), rsi_bin("TLT", 50)]}]}
    d = ci.convert(sym_if({"lhs-fn": "relative-strength-index", "lhs-window-days": "10", "lhs-val": "XLY",
                           "comparator": "gt", "rhs-val": "79", "rhs-fixed-value?": True, "condition": cond}))
    t = d["tree"]
    assert t["on"] == "SMH"
    assert norm(t["if"]) == norm('rsi(close, 10) > 79 or rsi(sym("QQQ").close, 10) > 79 or (rsi(sym("VTV").close, 10) > 79 '
                                 'or rsi(sym("XLP").close, 10) > 79) or (rsi(sym("SPY").close, 10) < 30 and '
                                 'rsi(sym("TLT").close, 10) > 50)')
    assert any("condition' block is used" in n for n in d["notes"])
    Portfolio.from_dict(d)
    # binary-compound "all" on its own
    d = ci.convert(sym_if({"condition": {"condition-type": "binary-compound", "operator": "all",
                                         "tickers": ["SPY", "QQQ"], "comparator": "lt",
                                         "lhs": {"fn": "cumulative-return", "ticker": "%", "params": {"window": 10}},
                                         "rhs": {"constant": -5}}}))
    assert norm(d["tree"]["if"]) == norm('tret(tr, 10) < -0.05 and tret(sym("QQQ").tr, 10) < -0.05')


def test_import_refuses_unknown_condition_shapes():
    with pytest.raises(ci.ComposerImportError, match="condition-type"):
        ci.convert(sym_if({"condition": {"condition-type": "ternary"}}))
    with pytest.raises(ci.ComposerImportError, match="operator"):
        ci.convert(sym_if({"condition": {"condition-type": "compound", "operator": "xor", "conditions": [rsi_bin("SPY")]}}))
    with pytest.raises(ci.ComposerImportError, match="unknown field"):
        ci.convert(sym_if({"condition": {**rsi_bin("SPY"), "weight": 2}}))


@pytest.mark.parametrize("fn, params, rule", [
    ("moving-average-convergence-divergence", {"fast-window": 12, "slow-window": 26}, "macd(12, 26)"),
    ("moving-average-convergence-divergence-signal", {"fast-window": 12, "slow-window": 26, "signal-window": 9},
     "macd_signal(12, 26, 9)"),
    ("percentage-price-oscillator", {"fast-window": 10, "slow-window": 30}, "ppo(10, 30)"),
    ("percentage-price-oscillator-signal", {"fast-window": 10, "slow-window": 30, "signal-window": 5}, "ppo_signal(10, 30, 5)"),
])
def test_import_macd_and_ppo(fn, params, rule):
    c = {"condition-type": "binary", "comparator": "gt", "lhs": {"fn": fn, "ticker": "QQQ", "params": params},
         "rhs": {"constant": 0}}
    d = ci.convert(sym_if({"condition": c}))
    assert d["tree"]["if"] == f"{rule} > 0" and d["tree"]["on"] == "QQQ"
    # another ticker's: the series is the last argument
    c2 = {**c, "lhs": {"fn": "relative-strength-index", "ticker": "QQQ", "params": {"window": 10}}, "comparator": "lt",
          "rhs": {"fn": fn, "ticker": "SPY", "params": params}}
    d = ci.convert(sym_if({"condition": c2}))
    assert d["tree"]["if"] == f'rsi(close, 10) < {rule[:-1]}, sym("SPY").close)'


def test_import_bollinger_bands_and_the_legacy_fields():
    d = ci.convert(sym_if({"lhs-fn": "current-price", "lhs-val": "SPY", "comparator": "lt", "rhs-fixed-value?": False,
                           "rhs-fn": "lower-bollinger", "rhs-val": "SPY", "rhs-fn-params": {"window": 20, "std-dev": 2}}))
    assert d["tree"]["if"] == "close < bb_lower(20, 2)"
    d = ci.convert(sym_if({"lhs-fn": "upper-bollinger", "lhs-val": "SPY", "lhs-fn-params": {"window": 20, "std-dev": 2.5},
                           "comparator": "gt", "rhs-fixed-value?": False, "rhs-fn": "current-price", "rhs-val": "SPY"}))
    assert d["tree"]["if"] == "bb_upper(20, 2.5) > close"
    # against a fixed number, a band is a price level: read on quoted prices, as the other price levels
    d = ci.convert(sym_if({"lhs-fn": "upper-bollinger", "lhs-val": "SPY", "lhs-fn-params": {"window": 20, "std-dev": 2},
                           "comparator": "gt", "rhs-val": "400"}))
    assert d["tree"]["if"] == "quoted(bb_upper(20, 2)) > 400"


def test_macd_params_errors_say_what_is_expected():
    c = {"condition-type": "binary", "comparator": "gt", "rhs": {"constant": 0},
         "lhs": {"fn": "moving-average-convergence-divergence", "ticker": "QQQ", "params": {"window": 12}}}
    with pytest.raises(ci.ComposerImportError, match="unknown parameter.*window.*fast window") as e:
        ci.convert(sym_if({"condition": c}))
    assert "must be a number, got None" not in str(e.value)
    # a single number for a function of several parameters
    legacy = {"lhs-fn": "moving-average-convergence-divergence", "lhs-val": "QQQ", "lhs-window-days": "12",
              "comparator": "gt", "rhs-val": "0"}
    with pytest.raises(ci.ComposerImportError, match="several parameters"):
        ci.convert(sym_if(legacy))
    # none given: the standard 12 / 26 (/ 9), with a note
    d = ci.convert(sym_if({**legacy, "lhs-window-days": None}))
    assert d["tree"]["if"] == "macd(12, 26) > 0" and any("standard" in n for n in d["notes"])
    # and a window-only function given a dict of other things
    bad = {"lhs-fn": "relative-strength-index", "lhs-val": "QQQ", "lhs-fn-params": {"fast-window": 3},
           "comparator": "gt", "rhs-val": "50"}
    with pytest.raises(ci.ComposerImportError, match='"window"'):
        ci.convert(sym_if(bad))


def test_import_eq_marketcap_and_empty():
    d = ci.convert(sym_if({"lhs-fn": "relative-strength-index", "lhs-val": "SPY", "lhs-window-days": "10",
                           "comparator": "eq", "rhs-val": "50"}, other={"step": "empty"}))
    assert d["tree"]["if"] == "rsi(close, 10) == 50" and d["tree"]["else"] == {"cash": True}
    assert any("exact equality" in n for n in d["notes"])
    mc = {"step": "root", "rebalance": "monthly", "children": [{"step": "wt-marketcap", "children": [
        {"step": "asset", "ticker": "AAPL", "has_marketcap": True}, {"step": "asset", "ticker": "MSFT"}]}]}
    d = ci.convert(mc)
    assert pf.without_ids(d["tree"]) == {"weights": "market_cap", "children": [{"asset": "AAPL"}, {"asset": "MSFT"}]}
    with pytest.raises(ci.ComposerImportError, match="empty block has no children"):
        ci.convert({"step": "root", "rebalance": "daily", "children": [{"step": "empty", "children": [
            {"step": "asset", "ticker": "SPY"}]}]})


def test_ppo_and_macd_formulas_match_tradingview():
    idx = pd.bdate_range("2020-01-01", periods=300)
    rng = np.random.default_rng(3)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))), index=idx)
    df = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1e6})
    ns = expr.Namespace(df, ticker="X")
    fast, slow = expr.ema_tv(close, 12), expr.ema_tv(close, 26)
    ppo = (fast - slow) / slow * 100
    got = expr.evaluate_value("ppo(12, 26)", ns)
    assert np.allclose(got.dropna(), ppo.dropna())
    sig = expr.ema_tv(ppo, 9)
    assert np.allclose(expr.evaluate_value("ppo_signal(12, 26, 9)", ns).dropna(), sig.dropna())
    macd = fast - slow
    assert np.allclose(expr.evaluate_value("macd_signal(12, 26, 9)", ns).dropna(), expr.ema_tv(macd, 9).dropna())
    mid, sd = close.rolling(20).mean(), close.rolling(20).std(ddof=0)
    assert np.allclose(expr.evaluate_value("bb_lower(20, 2)", ns).dropna(), (mid - 2 * sd).dropna())
    # the series argument (another ticker's close in a rule) is used when given
    assert np.allclose(expr.evaluate_value("bb_upper(20, 2, close * 2)", ns).dropna(), (2 * (mid + 2 * sd)).dropna())
    assert np.allclose(expr.evaluate_value("ppo(12, 26, close * 2)", ns).dropna(), ppo.dropna())


# ------------------------------------------------------------ 4b. the exporter writes all / any blocks and empty

def _top(sym):
    """The block under the root's wt-cash-equal wrapper (round 13: as Composer's own exports)."""
    assert sym["children"][0]["step"] == "wt-cash-equal"
    return sym["children"][0]["children"][0]


def _ic(sym):
    return _top(sym)["children"][0]


@needs("SPY", "QQQ", "SMH", "UVXY", "TQQQ", "TLT", "BIL")
@pytest.mark.parametrize("text", [
    "if SPY is above its 200 day moving average and QQQ 10 day RSI is below 79 then hold TQQQ else hold BIL",
    "if any of SPY, QQQ and SMH has a 10 day RSI above 79 then UVXY else TQQQ",
    "if both SPY and QQQ 10 day RSI are above 79 then UVXY else TQQQ",
    "if TLT is above its 200 day moving average or both SPY and QQQ 10 day RSI are below 30 then TQQQ else BIL",
    "if not (SPY is above its 200 day moving average or QQQ 10 day RSI is above 80) then hold QQQ else hold BIL",
])
def test_and_or_conditions_round_trip_as_condition_blocks(text):
    p = port(text)
    sym, _ = ce.export(p)
    ic = _ic(sym)
    assert ic["step"] == "if-child" and "condition" in ic
    # round 10: beside the block, the single-comparison fields repeat its last comparison, as in Composer's own exports
    # (tests/fixtures/composer_frontrunner_2026.json; see test_round10_composer_review)
    assert ic["lhs-fn"] and ic["comparator"] and "rhs-val" in ic
    assert len(_top(sym)["children"]) == 2              # one if-child and the else: no nested ifs
    back = Portfolio.from_dict(ci.convert(json.dumps(sym)))
    a, b = back.tree, p.tree
    assert norm(a["if"]) == norm(b["if"]) or _same_truth(a["if"], b["if"])
    assert pf.without_ids(a["then"]) == pf.without_ids(b["then"]) and pf.without_ids(a["else"]) == pf.without_ids(b["else"])


def _same_truth(r1: str, r2: str) -> bool:
    """not (A or B) is exported as all(not A, not B) with flipped comparators: compare them on data."""
    from backtester import portfolio as P
    p1 = Portfolio(tree={"if": r1, "on": "SPY", "then": {"asset": "QQQ"}, "else": {"asset": "BIL"}}, rebalance="daily",
                   start="2015-01-01", end="2016-12-31")
    p2 = Portfolio(tree={"if": r2, "on": "SPY", "then": {"asset": "QQQ"}, "else": {"asset": "BIL"}}, rebalance="daily",
                   start="2015-01-01", end="2016-12-31")
    return np.allclose(P.run(p1).equity.to_numpy(), P.run(p2).equity.to_numpy())


@needs("SPY", "QQQ", "SMH", "UVXY", "TQQQ")
def test_the_same_comparison_on_several_tickers_exports_as_binary_compound():
    sym, _ = ce.export(port("if any of SPY, QQQ and SMH has a 10 day RSI above 79 then UVXY else TQQQ"))
    c = _ic(sym)["condition"]
    assert c == {"condition-type": "binary-compound", "operator": "any", "tickers": ["SPY", "QQQ", "SMH"],
                 "lhs": {"fn": "relative-strength-index", "ticker": "%", "params": {"window": 10}}, "comparator": "gt",
                 "rhs": {"constant": 79}}


def test_export_eq_market_cap_and_empty():
    p = Portfolio(tree={"if": "rsi(close, 10) == 50", "on": "SPY", "then": {"cash": True},
                        "else": {"weights": "market_cap", "children": [{"asset": "AAPL"}, {"asset": "MSFT"}]}},
                  rebalance="daily")
    sym, _ = ce.export(p)
    kids = _top(sym)["children"]
    assert kids[0]["comparator"] == "eq" and kids[0]["children"][0]["step"] == "empty"
    assert kids[1]["children"][0]["step"] == "wt-marketcap"
    back = ci.convert(json.dumps(sym))["tree"]
    assert back["if"] == "rsi(close, 10) == 50" and back["then"] == {"cash": True}
    assert pf.without_ids(back["else"]) == p.tree["else"]
    with pytest.raises(ce.ComposerExportError, match="not equal"):
        ce.export(Portfolio(tree={"if": "not (rsi(close, 10) == 50)", "on": "SPY", "then": {"asset": "SPY"},
                                  "else": {"asset": "BIL"}}, rebalance="daily"))
    with pytest.raises(ce.ComposerExportError, match="do(es)? not name the keys"):
        ce.export(Portfolio(tree={"if": "macd(12, 26) > 0", "on": "SPY", "then": {"asset": "SPY"},
                                  "else": {"asset": "BIL"}}, rebalance="daily"))


# ------------------------------------------------------------ 5. threshold rebalancing, slugs, group names

@needs("SPY", "TLT")
def test_if_else_with_a_drift_band_is_threshold_rebalancing():
    p = port("if SPY is above its 200 day moving average then hold 60% SPY and 40% TLT else hold TLT, rebalance when any "
             "weight drifts 5% from target")
    assert not any("Rebalance frequency not stated" in n for n in p.notes)
    assert any("Threshold rebalancing" in n for n in p.notes)
    sym, _ = ce.export(p)                                    # no 'never rebalance' needed: it is Composer's threshold mode
    assert sym["rebalance"] == "none" and sym["rebalance-corridor-width"] == 0.05
    back = Portfolio.from_dict(ci.convert(json.dumps(sym)))
    assert back.drift_band == 0.05
    with pytest.raises(ce.ComposerExportError, match="not both"):
        ce.export(port("hold 60% SPY and 40% TLT, rebalance quarterly or when any weight drifts more than 5%"))


def test_report_folder_names_are_unique_and_readable():
    a = report.run_slug("if SPY is above its 200 day moving average then hold QQQ else hold BIL, rebalance monthly",
                        '{"a": 1}')
    b = report.run_slug("if SPY is above its 200 day moving average then hold QQQ else hold BIL, rebalance weekly",
                        '{"a": 2}')
    assert a != b and a.startswith("if-spy-is-above-its-200-day-moving-average-then-ho")
    assert len(a) <= 59 and a == report.run_slug("if SPY is above its 200 day moving average then hold QQQ else hold BIL, "
                                                 "rebalance monthly", '{"a": 1}')
    assert report.run_slug("x", {"b": 1, "a": 2}) == report.run_slug("x", {"a": 2, "b": 1})


def test_group_names_show_in_the_interpretation():
    sym = {"step": "root", "name": "S", "rebalance": "daily", "children": [{"step": "group", "name": "Risk On",
           "children": [{"step": "wt-cash-equal", "children": [{"step": "asset", "ticker": "SPY"},
                                                                {"step": "asset", "ticker": "QQQ"}]}]}]}
    p = Portfolio.from_dict(ci.convert(sym))
    s = p.summary()
    assert "group Risk On:" in s
    sym["children"][0]["children"] = [{"step": "asset", "ticker": "SPY"}]
    assert "Risk On: SPY" in Portfolio.from_dict(ci.convert(sym)).summary()


# ------------------------------------------------------------ 6. phrase gaps

@needs("SPY", "BIL", "QQQ", "UVXY", "TQQQ")
@pytest.mark.parametrize("text, rule", [
    ("if SPY closes more than 20% above its 52 week low then SPY else BIL", "close / lowest(close, 252) - 1 > 0.2"),
    ("if SPY EMA(8) > SPY SMA(21) then SPY else BIL", "ema(close, 8) > sma(close, 21)"),
    ("if the 8 day EMA of SPY is above the 21 day SMA of SPY then SPY else BIL", "ema(close, 8) > sma(close, 21)"),
    ("if SPY is below its 50 day moving average by more than 5% then BIL else SPY", "close < sma(close, 50) * 0.95"),
    ("if the RSI of SPY is 10 points above the RSI of QQQ then SPY else QQQ",
     'rsi(close, 14) - rsi(sym("QQQ").close, 14) >= 10'),
    ("if SPY 10 day RSI is more than 10 points below QQQ 10 day RSI then SPY else QQQ",
     'rsi(sym("QQQ").close, 10) - rsi(close, 10) > 10'),
    ("if SPY 12 month return is 5 points above QQQ 12 month return then SPY else QQQ",
     'tret(tr, 252) - tret(sym("QQQ").tr, 252) >= 0.05'),
])
def test_phrase_gaps(text, rule):
    assert norm(port(text).tree["if"]) == norm(rule)


@needs("SPY", "BIL")
def test_closes_above_its_52_week_low_in_a_signal():
    s = parser.parse("buy SPY when it closes more than 20% above its 52 week low, sell after 10 days")
    assert norm(s.entry) == norm("close / lowest(close, 252) - 1 > 0.2")


@needs("SPY", "BIL", "QQQ", "UVXY", "TQQQ")
def test_unless_in_which_case():
    p = port("hold SPY unless SPY is below its 200 day moving average, in which case hold BIL")
    assert norm(p.tree["if"]) == norm("close < sma(close, 200)")
    assert p.tree["then"] == {"asset": "BIL"} and p.tree["else"] == {"asset": "SPY"}
    p = port("hold TQQQ unless both SPY and QQQ 10 day RSI are above 79, in which case hold UVXY")
    assert p.tree["then"] == {"asset": "UVXY"} and p.tree["else"] == {"asset": "TQQQ"}
    assert tickers_in_rule(p.tree["if"]) == {"QQQ"}


@needs("SPY", "QQQ", "TLT", "GLD")
def test_filter_weighted_by_rank():
    p = port("hold the top 2 of SPY, QQQ, TLT, GLD by 6 month return, weighted 70/30, rebalance monthly")
    f = p.tree["filter"]
    assert f["weights"] == "specified" and f["w"] == [0.7, 0.3]
    assert "weighted 70%/30% by rank" in p.summary()
    with pytest.raises(ParseError, match="add up to 100"):
        parser.parse("hold the top 2 of SPY, QQQ, TLT, GLD by 6 month return, weighted 70/20")
    with pytest.raises(ParseError, match="one weight per pick"):
        parser.parse("hold the top 3 of SPY, QQQ, TLT, GLD by 6 month return, weighted 70/30")
    # the best pick gets about 70% (70% on each month-end, drifting in between)
    p.start, p.end = "2015-01-01", "2016-12-31"
    h = pf.run(p).holdings
    top = h.drop(columns=[c for c in h.columns if c == "cash"]).max(axis=1)
    top = top[top > 0]
    assert len(top) and ((top > 0.55) & (top < 0.85)).mean() > 0.9


def test_filter_rank_weights_are_validated():
    base = {"filter": {"select": "top", "n": 2, "by": "tret(tr, 20)", "weights": "specified", "w": [0.7, 0.3]},
            "universe": ["SPY", "TLT", "GLD"]}
    Portfolio.from_dict({"tree": base}).validate()
    for w in ([0.7], [0.7, 0.2], [1.2, -0.2]):
        with pytest.raises(ValueError):
            Portfolio.from_dict({"tree": {**base, "filter": {**base["filter"], "w": w}}}).validate()
