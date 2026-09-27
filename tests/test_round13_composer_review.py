"""Round 13 (a Composer user's review): Composer timing for typed schedules, the export's wt-cash-equal wrapper,
import of unknown metadata with a warning, clearer refusals, new phrases, politeness, and the original FNGU ETN's
history as a former listing."""
import json

import numpy as np
import pandas as pd
import pytest

from backtester import composer_export as ce
from backtester import composer_import as ci
from backtester import data, expr, parser
from backtester.parser import ParseError
from backtester.portfolio import Portfolio, without_ids

AVAIL = set(data.available_tickers())


def needs(*t):
    return pytest.mark.skipif(not set(t) <= AVAIL, reason="price data not downloaded")


def port(s):
    return parser.parse(s)


IF = "if SPY is above its 200 day moving average then hold QQQ else hold TLT"


# ------------------------------------------------------------ 2. rebalance timing of typed schedules

@needs("SPY", "QQQ", "TLT")
@pytest.mark.parametrize("freq, per", [("weekly", "week"), ("monthly", "month"), ("quarterly", "quarter"),
                                       ("yearly", "year")])
def test_typed_schedule_notes_its_day_and_how_to_get_composer_timing(freq, per):
    p = port(f"{IF}, rebalance {freq}")
    assert p.rebalance == freq and getattr(p, "rebalance_day", None) in (None, "end")   # the default is kept
    note = next(n for n in p.notes if n.startswith("Rebalance timing:"))
    assert f"last trading day of each {per}" in note and "with Composer timing" in note
    assert f"first trading day of each {per}" in note and "checks the rules" in note
    # fixed weights: the same note (without the rules)
    q = port(f"hold 60% SPY and 40% TLT, rebalance {freq}")
    assert any(n.startswith("Rebalance timing:") and "checks the rules" not in n for n in q.notes)


@needs("SPY", "QQQ", "TLT")
@pytest.mark.parametrize("phrase", ["with Composer timing", "using Composer's schedule", "Composer-style",
                                    "like Composer", "as Composer does"])
def test_composer_timing_phrase_trades_on_the_first_trading_day(phrase):
    p = port(f"{IF}, rebalance monthly {phrase}")
    assert p.rebalance == "monthly" and p.rebalance_day == "start"
    assert any(n.startswith("Composer timing:") for n in p.notes)
    assert not any(n.startswith("Rebalance timing:") for n in p.notes)
    q = port(f"{IF}, quarterly rebalancing, with Composer timing")
    assert q.rebalance == "quarterly" and q.rebalance_day == "start"
    # no calendar schedule: daily, which is Composer's default too (a note says so)
    d = port(f"{IF}, with Composer timing")
    assert d.rebalance == "daily" and any(n.startswith("Composer timing:") for n in d.notes)
    # a stated day that is not Composer's: refused
    with pytest.raises(ParseError, match="already names a rebalance day"):
        port(f"{IF}, rebalance on the last trading day of each month, with Composer timing")


@needs("SPY", "QQQ", "TLT")
def test_explicit_days_get_no_timing_note():
    for s in (f"{IF}, rebalance on the last trading day of each month", f"{IF}, rebalance every Monday",
              f"{IF}, rebalance on the first trading day of each month", f"{IF}, rebalance daily", IF):
        assert not any(n.startswith("Rebalance timing:") for n in port(s).notes), s


# ------------------------------------------------------------ 3. export: the wt-cash-equal wrapper

def _plain(n):
    """A tree without Composer ids, the brackets around a whole rule and a filter's default cash fallback."""
    if isinstance(n, dict):
        return {k: (v[1:-1] if k == "if" and isinstance(v, str) and v.startswith("(") and v.endswith(")") else _plain(v))
                for k, v in n.items() if k != "composer" and not (k == "fallback" and v == {"cash": True})}   # (cash is the default)
    if isinstance(n, list):
        return [_plain(x) for x in n]
    return n


@needs("SPY", "QQQ", "TLT")
def test_export_wraps_the_root_in_wt_cash_equal_and_reimports_identically():
    for s in (IF, "hold 60% SPY and 40% TLT, rebalance monthly", "hold SPY, rebalance monthly",
              "hold the top 1 of SPY, QQQ and TLT by 20 day return"):
        p = port(s)
        sym, _ = ce.export(p)
        top = sym["children"]
        assert len(top) == 1 and top[0]["step"] == "wt-cash-equal" and top[0]["id"]
        assert len(top[0]["children"]) == 1
        back = ci.convert(json.dumps(sym))
        assert _plain(back["tree"]) == _plain(p.tree)
        assert "wrap_id" not in (back["tree"].get("composer") or {})      # the export's own wrapper is not kept as an id
        again, _ = ce.export(Portfolio.from_dict(back))
        assert again["children"] == sym["children"] and again["id"] == sym["id"]   # the same blocks and ids again
    # an equal-weight portfolio already is a wt-cash-equal block: not wrapped twice
    sym, _ = ce.export(port("hold SPY, QQQ and TLT equally, rebalance monthly"))
    assert sym["children"][0]["step"] == "wt-cash-equal" and len(sym["children"][0]["children"]) == 3


@needs("SPY", "QQQ", "TLT")
def test_a_composer_wrapper_keeps_its_id_through_a_round_trip():
    sym = {"id": "R1", "step": "root", "name": "w", "rebalance": "daily", "children": [
        {"id": "W1", "step": "wt-cash-equal", "children": [
            {"id": "I1", "step": "if", "children": [
                {"id": "C1", "step": "if-child", "is-else-condition?": False, "lhs-fn": "relative-strength-index",
                 "lhs-val": "SPY", "lhs-fn-params": {"window": 10}, "comparator": "gt", "rhs-val": "79",
                 "rhs-fixed-value?": True, "children": [{"id": "A1", "step": "asset", "ticker": "TLT"}]},
                {"id": "C2", "step": "if-child", "is-else-condition?": True,
                 "children": [{"id": "A2", "step": "asset", "ticker": "QQQ"}]}]}]}]}
    spec = ci.convert(sym)
    assert spec["tree"]["composer"]["wrap_id"] == "W1"
    out, _ = ce.export(Portfolio.from_dict(spec))
    assert out["children"][0]["id"] == "W1" and out["children"][0]["step"] == "wt-cash-equal"
    assert out["children"][0]["children"][0]["id"] == "I1"


# ------------------------------------------------------------ 4. import: unknown metadata warns, structure refuses

def _root(child, **extra):
    return {"step": "root", "name": "t", "rebalance": "daily", "children": [child], **extra}


@needs("SPY")
def test_unknown_metadata_fields_are_ignored_with_a_warning():
    sym = _root({"step": "asset", "ticker": "SPY", "sparkline-url": "x", "ui_state": {"open": True}},
                **{"thumbnail": "t.png", "last-edited-by": "me"})
    spec = ci.convert(sym)
    assert without_ids(spec["tree"]) == {"asset": "SPY"}
    warns = [n for n in spec["notes"] if n.startswith("Warning:")]
    assert any("sparkline-url" in n and "ui_state" in n and "ignored" in n for n in warns)
    assert any("thumbnail" in n and "last-edited-by" in n for n in warns)
    # inside a condition block too
    cond = {"condition-type": "binary", "lhs": {"fn": "relative-strength-index", "ticker": "SPY", "params": {"window": 10}},
            "comparator": "gt", "rhs": {"constant": 70}, "display-label": "RSI"}
    sym = _root({"step": "if", "children": [
        {"step": "if-child", "is-else-condition?": False, "condition": cond, "children": [{"step": "asset", "ticker": "SPY"}]},
        {"step": "if-child", "is-else-condition?": True, "children": [{"step": "asset", "ticker": "SPY"}]}]})
    spec = ci.convert(sym)
    assert any("display-label" in n for n in spec["notes"] if n.startswith("Warning:"))


@pytest.mark.parametrize("child, needle", [
    ({"step": "asset", "ticker": "SPY", "weight-override": 0.5}, "weight-override"),
    ({"step": "asset", "ticker": "SPY", "leverage": 2}, "leverage"),
    ({"step": "asset", "ticker": "SPY", "hedge": {"step": "asset", "ticker": "SH"}}, "hedge"),
    ({"step": "asset", "ticker": "SPY", "extra": [{"step": "asset", "ticker": "TLT"}]}, "extra"),   # holds blocks
    ({"step": "filter", "sort-by-fn": "cumulative-return", "sort-by-fn-params": {"window": 10}, "select-fn": "top",
      "select-n": 1, "select-threshold": 3, "children": [{"step": "asset", "ticker": "SPY"}]}, "select-threshold"),
    ({"step": "wt-magic", "children": [{"step": "asset", "ticker": "SPY"}]}, "wt-magic"),
    ({"step": "filter", "sort-by-fn": "hurst-exponent", "sort-by-fn-params": {"window": 10}, "select-fn": "top",
      "select-n": 1, "children": [{"step": "asset", "ticker": "SPY"}]}, "hurst-exponent"),
])
def test_unknown_structure_still_refuses_with_the_path(child, needle):
    with pytest.raises(ci.ComposerImportError) as e:
        ci.convert(_root(child))
    assert needle in str(e.value) and "symphony" in str(e.value)


def test_unknown_comparator_and_condition_fields_refuse():
    base = {"condition-type": "binary", "lhs": {"fn": "relative-strength-index", "ticker": "SPY", "params": {"window": 10}},
            "comparator": "ne", "rhs": {"constant": 70}}

    def sym(c):
        return _root({"step": "if", "children": [
            {"step": "if-child", "is-else-condition?": False, "condition": c, "children": [{"step": "asset", "ticker": "SPY"}]},
            {"step": "if-child", "is-else-condition?": True, "children": [{"step": "asset", "ticker": "SPY"}]}]})
    with pytest.raises(ci.ComposerImportError, match="comparator"):
        ci.convert(sym(base))
    with pytest.raises(ci.ComposerImportError, match="rhs-offset"):
        ci.convert(sym({**base, "comparator": "gt", "rhs-offset": 1}))


# ------------------------------------------------------------ 5. clearer refusals

@needs("SPY", "QQQ", "TLT")
@pytest.mark.parametrize("text, needle", [
    (f"{IF}, rebalance daily, rebalance monthly", "Two rebalancing schedules conflict: 'rebalance daily' and 'rebalance monthly'"),
    (f"{IF}, monthly rebalancing, never rebalance", "conflict"),
    ("if SPY is above its 200 day moving average then hold QQQ else SPY, but only on tuesdays", "rebalance every Tuesday"),
    ("if SPY is above its 200 day moving average then hold QQQ else SPY and also", "ends with 'and also'"),
    ("hold SPY and", "ends with 'and'"),
    ("hold 3x leveraged QQQ", "hold TQQQ"),
    ("hold 3x QQQ", "hold QQQ with 3x leverage"),
    ("hold 60% SPY and 40% also, rebalance monthly", "'also' is not understood here"),
])
def test_refusals_say_what_is_wrong(text, needle):
    with pytest.raises(ParseError) as e:
        port(text)
    msg = str(e.value)
    assert needle in msg, msg
    assert "Could not interpret 'd'" not in msg and "Unknown ticker ALSO" not in msg and "No ticker found in 'but only'" not in msg


@needs("SPY", "QQQ", "TLT")
def test_the_same_schedule_twice_is_accepted():
    p = port(f"{IF}, rebalance monthly, rebalance monthly")
    assert p.rebalance == "monthly"


@needs("SPY", "QQQ", "TLT", "AGG")
def test_a_benchmark_schedule_is_not_a_second_portfolio_schedule():
    p = port(f"{IF}, rebalance quarterly, vs 60/40 SPY/AGG rebalanced monthly")
    assert p.rebalance == "quarterly"


# ------------------------------------------------------------ 6. phrases

@needs("SPY", "TLT", "^VIX")
@pytest.mark.parametrize("cond", ["VIX > 30", "VIX is above 30"])
def test_vix_level(cond):
    p = port(f"if {cond} then hold TLT else hold SPY")
    assert p.tree["on"] == "^VIX" and p.tree["if"] in ("(quoted(close) > 30)", "(close > 30)")
    s = port(f"buy SPY when {cond}, sell after 5 days")
    assert s.entry == '(sym("^VIX").close > 30)'
    assert port("if VIX <= 20 then hold SPY else hold TLT").tree["if"].replace("quoted(close)", "close") == "(close <= 20)"
    # ^VIX closes at 4:15pm: a rule acted on at the 4:00pm close reads the day before's value (engine.late_close)
    assert expr.is_late_close("^VIX")


@needs("SPY", "QQQ", "TLT")
def test_previous_close_and_streaks():
    assert port("if SPY is higher than yesterday then hold QQQ else hold TLT").tree["if"] == "(close > ref(close, 1))"
    assert port("if SPY closed lower than the previous close then hold TLT else hold QQQ").tree["if"] == "(close < ref(close, 1))"
    assert port("buy SPY when it is higher than yesterday, sell after 5 days").entry == "(close > ref(close, 1))"
    assert port("if SPY has risen for 3 days in a row then hold QQQ else hold TLT").tree["if"] == "(up_days >= 3)"
    assert port("if SPY has fallen 4 days in a row then hold QQQ else hold TLT").tree["if"] == "(down_days >= 4)"


@needs("SPY", "QQQ", "TLT")
def test_return_spread_in_percentage_points():
    p = port("if SPY's 20 day return exceeds TLT's by more than 2% then hold SPY else hold TLT")
    assert p.tree["if"] == '(tret(tr, 20) - tret(sym("TLT").tr, 20) > 0.02)' and p.tree["on"] == "SPY"
    assert any("percentage points" in n for n in p.notes)
    q = port("if SPY's 20 day return is more than 2% above TLT's 20 day return then hold SPY else hold TLT")
    assert q.tree["if"] == p.tree["if"]
    s = port("buy QQQ when SPY's 20 day return trails TLT's by at least 3%, sell after 5 days")
    assert s.entry == '(tret(sym("SPY").tr, 20) - tret(sym("TLT").tr, 20) <= -0.03)'


@needs("SPY", "QQQ", "TLT")
def test_month_to_date_return_is_causal_and_correct():
    p = port("if SPY is down more than 10% this month then hold TLT else hold SPY")
    rule = p.tree["if"]
    assert rule == "(tr / valuewhen(trading_day_of_month == 1, ref(tr, 1), 0) - 1 < -0.1)"
    assert any(n.startswith("Month-to-date return") for n in p.notes)
    s = port("buy SPY when its month to date return is below -10%, sell after 5 days")
    assert s.entry == rule
    up = port("if SPY is up at least 5% this month then hold SPY else hold TLT").tree["if"]
    assert up.endswith(">= 0.05)")
    # the value: the total return since the previous month's last close
    df = data.load("SPY")
    ns = expr.Namespace(df, ticker="SPY")
    got = expr.evaluate_value("tr / valuewhen(trading_day_of_month == 1, ref(tr, 1), 0) - 1", ns)
    tr = expr.evaluate_value("tr", ns)
    month_end = tr.groupby(tr.index.to_period("M")).last().shift(1)
    want = tr / month_end.reindex(tr.index.to_period("M")).to_numpy() - 1
    ok = want.notna() & got.notna()
    assert ok.sum() > 1000 and np.allclose(got[ok], want[ok])
    # no lookahead: the data truncated at D gives the same values before D
    cut = df.index[len(df) // 2]
    short = expr.evaluate_value("tr / valuewhen(trading_day_of_month == 1, ref(tr, 1), 0) - 1",
                                expr.Namespace(df[df.index <= cut], ticker="SPY"))
    both = short.dropna().index
    assert np.allclose(short[both], got[both])


@needs("SPY", "QQQ", "TLT", "GLD")
def test_the_n_with_the_smallest_among():
    p = port("hold the 2 with the smallest drawdown among SPY, QQQ, TLT and GLD")
    assert p.tree["filter"]["select"] == "bottom" and p.tree["filter"]["n"] == 2
    assert p.tree["filter"]["by"] == "-drawdown(close)"        # the fall below the high, as a positive size
    assert [c["asset"] for c in p.tree["children"]] == ["SPY", "QQQ", "TLT", "GLD"]
    q = port("hold the 1 with the highest 60 day return among SPY, QQQ and TLT")
    assert q.tree["filter"]["select"] == "top" and q.tree["filter"]["by"] == "tret(tr, 60)"
    m = port("hold the 2 with the smallest 60 day max drawdown among SPY, QQQ, TLT and GLD")
    assert m.tree["filter"] == {**m.tree["filter"], "select": "bottom", "by": "max_drawdown(tr, 60)"}


@needs("SPY", "TLT")
def test_composer_label_order_and_less_negative():
    assert port("if SPY RSI 10 day greater than 80 then hold TLT else hold SPY").tree["if"] == "(rsi(close, 10) > 80)"
    assert port("if SPY 10 day return is less negative than -5% then hold SPY else hold TLT").tree["if"] == "(tret(tr, 10) > -0.05)"
    assert port("if SPY 10 day return is more negative than -5% then hold TLT else hold SPY").tree["if"] == "(tret(tr, 10) < -0.05)"


@needs("QQQ", "TQQQ", "SQQQ")
def test_leveraged_etf_phrases():
    p = port("hold 3x leveraged QQQ ETF")
    assert p.tree == {"asset": "TQQQ"} and any("read as TQQQ" in n for n in p.notes)
    assert port("hold -3x QQQ ETF").tree == {"asset": "SQQQ"}
    assert port("hold 3x inverse QQQ ETF").tree == {"asset": "SQQQ"}
    # not a ticker after the "x": left alone ("volume is 2x its average", "2x ATR")
    s = port("buy QQQ when volume is 2x its 20 day average, sell after 5 days")
    assert "volume" in s.entry


# ------------------------------------------------------------ 7. politeness

@needs("SPY", "TLT")
@pytest.mark.parametrize("text", ["please hold SPY", "hold SPY, thanks", "kindly hold SPY", "hold SPY please",
                                  "hold SPY. Thank you!", "can you please hold SPY", "could you hold SPY, cheers",
                                  "pls hold SPY thx"])
def test_politeness_is_ignored_everywhere(text):
    assert without_ids(port(text).tree) == {"asset": "SPY"}


@needs("SPY")
def test_politeness_in_signal_sentences():
    base = port("buy SPY when RSI(2) is below 10, sell after 5 days")
    for t in ("please buy SPY when RSI(2) is below 10, sell after 5 days, thanks",
              "kindly buy SPY when RSI(2) is below 10, sell after 5 days"):
        s = port(t)
        assert s.entry == base.entry and s.hold_bars == base.hold_bars


# ------------------------------------------------------------ 8. the original FNGU ETN as a former listing

def test_fngu_former_listing():
    assert data.FORMER_LISTINGS["FNGU"] == ("FNGU-2025", "2025-02-20")
    old = data.load("FNGU-2025")
    assert str(old.index[0].date()) == "2018-01-23" and str(old.index[-1].date()) == "2024-09-27"
    # splits are in the adjusted series (10:1 in 2021, 1:10 in 2022): no jump on those days
    r = old["close"].pct_change()
    assert abs(r.loc["2021-02-12"]) < 0.05 and abs(r.loc["2022-10-31"]) < 0.1 and r.abs().max() < 0.35
    assert (old["dividend"] == 0).all()
    src = json.loads((data.DATA / "delisted_sources.json").read_text())["FNGU-2025"]
    assert src["segments"][0]["source"] == "intrader" and src["splits"] == {"2021-02-12": 10.0, "2022-10-31": 0.1}
    # a backtest of FNGU before the new ETN's listing says where the old history is
    notes = data.identity_notes("FNGU", "2020-01-01", "2026-01-01")
    assert any(n.startswith("Former listing:") and "FNGU-2025" in n for n in notes)
    assert not any(n.startswith("Former listing:") for n in data.identity_notes("FNGU", "2025-03-01", "2026-01-01"))
    assert any(n.startswith("Delisted:") for n in data.identity_notes("FNGU-2025", "2020-01-01", "2026-01-01"))
    # BULZ's file starts at its own inception (2021-08): nothing is missing
    if "BULZ" in AVAIL:
        assert data.load("BULZ").index[0] <= pd.Timestamp("2021-08-20")
