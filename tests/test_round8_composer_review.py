"""Round 8 (Composer expert review): threshold rebalancing follows the rules every day, bare numbers on percent-valued
indicators are refused, the short-history warning names the requested start, group lists parse before and after the
weighting phrase, fortnightly rebalancing, portfolio stops and bare shorts, notes on defaults, and a Composer export
that keeps block ids, group names and both window formats."""
import json
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtester import composer_export as ce, composer_import as ci, data, expr, parser, portfolio as pf, runner
from backtester.parser import ParseError
from backtester.portfolio import Portfolio

AVAIL = set(data.available_tickers())
needs = lambda *ts: pytest.mark.skipif(not set(ts) <= AVAIL, reason="price data not downloaded")  # noqa: E731
SYM2 = Path(__file__).with_name("fixtures") / "sym2.json"


def port(text):
    p = parser.parse(text)
    assert isinstance(p, Portfolio), text
    return p


def trade_days(res) -> list:
    """Days with orders after the initial purchase."""
    d = sorted(set(pd.to_datetime(res.orders["date"])))
    return d[1:]


# ------------------------------------------------------------ 1. threshold rebalancing re-evaluates the logic daily

def test_sym2_imports_as_threshold_rebalancing_with_a_note():
    d = ci.load(SYM2)
    assert d["rebalance"] == "none" and d["drift_band"] == pytest.approx(0.05)
    tree = pf.without_ids(d["tree"])
    assert tree["weights"] == "specified" and tree["w"] == [0.5, 0.5]
    assert tree["children"][0] == {"if": "ema(close, 20) <= sma(close, 50)", "on": "QQQ", "then": {"asset": "PSQ"},
                                   "else": {"asset": "QQQ"}, "name": "Trend switch"}
    assert tree["children"][1]["filter"] == {"select": "bottom", "n": 1, "by": "stdev_return(tr, 20)", "weights": "equal"}
    assert "evaluated at every close" in d["notes"][0] and "5 percentage points" in d["notes"][0]
    s = Portfolio.from_dict(dict(d)).summary()
    assert "threshold rebalancing" in s and "re-evaluated at every close" in s


@needs("QQQ", "PSQ", "TLT", "GLD", "UUP")
def test_sym2_trades_on_branch_switches_and_drift_verified_with_pandas():
    p = Portfolio.from_dict(dict(ci.load(SYM2)))
    res = runner.run(p)
    got = trade_days(res)
    cal = res.equity.index
    # independent: the target each close (pandas), then a drift simulation on total-return prices
    adj = {t: data.load(t)["adj_close"] for t in ("QQQ", "PSQ", "TLT", "GLD", "UUP")}
    q = adj["QQQ"]
    down = (expr.ema_tv(q, 20) <= q.rolling(50).mean()).reindex(cal)
    vol = pd.DataFrame({t: adj[t].pct_change(fill_method=None).rolling(20).std() for t in ("TLT", "GLD", "UUP")}).reindex(cal)
    pick = vol.idxmin(axis=1)
    px = pd.DataFrame(adj).reindex(cal)
    first = pd.Timestamp(res.orders["date"].iloc[0])
    want, hold, prev = [], None, None
    for d in cal[cal >= first]:
        tgt = {("PSQ" if down[d] else "QQQ"): 0.5, pick[d]: 0.5}
        if hold is None:
            hold, prev = {k: w / px.at[d, k] for k, w in tgt.items()}, tgt
            continue
        val = {k: s * px.at[d, k] for k, s in hold.items()}
        eq = sum(val.values())
        drift = any(abs(val.get(k, 0) / eq - tgt.get(k, 0)) > 0.05 for k in set(val) | set(tgt))
        if set(tgt) != set(prev) or drift:
            want.append(d)
            hold = {k: w * eq / px.at[d, k] for k, w in tgt.items()}
        prev = tgt
    assert len(got) > 100            # the logic is followed (the bug held PSQ and UUP from 2007 on)
    assert got == want
    reasons = set(res.orders["reason"])
    assert {"rebalance", "drift rebalance"} <= reasons
    # and it holds all three defensive funds and both QQQ legs at some point
    assert {"PSQ", "QQQ", "TLT", "GLD", "UUP"} <= set(res.orders["ticker"])


@needs("SPY", "TQQQ", "BIL")
@pytest.mark.parametrize("tail", [", never rebalance, rebalance when any weight drifts 5% from target",
                                  ", never rebalance"])
def test_sentence_threshold_and_never_rebalance_follow_the_switch(tail):
    p = port("if SPY is above its 200 day moving average then hold TQQQ else hold BIL" + tail)
    assert p.rebalance == "none"
    s = p.summary()
    assert "re-evaluated at every close" in s and "buy and hold" not in s
    res = runner.run(p)
    cal = res.equity.index
    a = data.load("SPY")["adj_close"]
    up = (a > a.rolling(200).mean()).reindex(cal)
    first = pd.Timestamp(res.orders["date"].iloc[0])
    u = up[cal >= first]
    flips = list(u.index[1:][(u.to_numpy()[1:] != u.to_numpy()[:-1])])
    got = trade_days(res)
    assert got == flips and len(got) > 20
    # each switch sells one fund and buys the other; nothing else trades (a single holding cannot drift)
    assert (res.orders.groupby("date").size().iloc[1:] == 2).all()


@needs("SPY", "TLT")
def test_rebalance_combinations_static_tree():
    base = "hold 60% SPY and 40% TLT, since 2010"
    # never: buy and hold (one purchase)
    p = port(base + ", never rebalance")
    assert "buy and hold" in p.summary()
    res = runner.run(p)
    assert len(trade_days(res)) == 0
    # band only: trades only when a weight leaves its band, measured with pandas
    p = port(base + ", rebalance when any weight drifts 5% from target")
    res = runner.run(p)
    got = trade_days(res)
    assert got and set(res.orders["reason"].iloc[2:]) == {"drift rebalance"}
    w = res.weights["SPY"] if hasattr(res, "weights") and res.weights is not None and "SPY" in res.weights else None
    cal = res.equity.index
    # the engine's accounting: quoted shares, cash dividends reinvested at that day's close
    fr = {t: data.load(t).reindex(cal) for t in ("SPY", "TLT")}
    px = pd.DataFrame({t: f["close"] for t, f in fr.items()})
    grow = pd.DataFrame({t: 1 + f["dividend"].fillna(0) / f["close"] for t, f in fr.items()})
    hold = {"SPY": 6000 / px["SPY"].iloc[0], "TLT": 4000 / px["TLT"].iloc[0]}
    want = []
    for d in cal[1:]:
        hold = {k: s * grow.at[d, k] for k, s in hold.items()}
        v = {k: s * px.at[d, k] for k, s in hold.items()}
        eq = sum(v.values())
        if abs(v["SPY"] / eq - 0.6) > 0.05:
            want.append(d)
            hold = {"SPY": 0.6 * eq / px.at[d, "SPY"], "TLT": 0.4 * eq / px.at[d, "TLT"]}
    # (a band crossing that sits right on the edge can land a day or two apart between the two accountings early on)
    assert abs(len(got) - len(want)) <= 3 and got[-10:] == want[-10:], w
    # schedule or band: every quarter-end trades, and the band in between
    p = port(base + ", rebalance quarterly or when any weight drifts 5% from target")
    assert p.rebalance == "quarterly" and p.drift_band == 0.05
    assert "in between" in p.summary()
    res = runner.run(p)
    assert {"rebalance"} <= set(res.orders["reason"])


@needs("SPY", "QQQ", "BIL")
def test_never_rebalance_inverse_vol_dynamic_tree_trades_only_on_switches():
    # a dynamic tree with weights that move every day: without a band only a change of holdings trades
    p = Portfolio(tree={"if": "close > sma(close, 200)", "on": "SPY",
                        "then": {"weights": "inverse_vol", "lookback": 20, "children": [{"asset": "SPY"}, {"asset": "QQQ"}]},
                        "else": {"asset": "BIL"}}, rebalance="none", start="2012-01-01")
    assert "pick different holdings" in p.summary()
    res = runner.run(p)
    for d, g in res.orders.groupby("date"):
        if d == res.orders["date"].iloc[0]:
            continue
        assert ("BIL" in set(g["ticker"])), d     # every trade after the first is a switch in or out of BIL


# ------------------------------------------------------------ 2. percent-valued indicators need the % sign

@pytest.mark.parametrize("text, opts", [
    ("if SPY 20 day moving average of return is above 0.1 then SPY else BIL", "0.1% or 10%"),
    ("if SPY 20 day standard deviation of return is above 0.5 then BIL else SPY", "0.5% or 50%"),
    ("if SPY 60 day cumulative return is below -0.5 then BIL else SPY", "-0.5% or -50%"),
    ("if SPY 63 day max drawdown is above 0.1 then BIL else SPY", "0.1% or 10%"),
    ("if SPY 20 day volatility is above 0.2 then BIL else SPY", "0.2% or 20%"),
    ("if SPY distance from its 200 day moving average is above 0.05 then SPY else BIL", "0.05% or 5%"),
    ("if SPY 20 day return is above 2 then SPY else BIL", "2%"),
    ("buy SPY when its 20 day return is above 0.05, sell after 5 days", "0.05% or 5%"),
    ("buy SPY when its 20 day moving average of return is above 0.1, sell after 5 days", "0.1% or 10%"),
])
def test_bare_numbers_on_percent_indicators_are_refused(text, opts):
    with pytest.raises(ParseError, match="did you mean " + opts.replace(".", r"\.") + r"\?"):
        parser.parse(text)


def test_percent_written_or_zero_is_fine():
    assert port("if SPY 20 day moving average of return is above 0.1% then SPY else BIL").tree["if"] == "ma_return(tr, 20) > 0.001"
    assert port("if SPY 20 day return is above 0 then SPY else BIL").tree["if"].endswith("> 0")
    # Composer imports have typed units: a moving average of return of 0.1 (percent) is 0.001
    c = {"step": "if-child", "is-else-condition?": False, "lhs-fn": "moving-average-return", "lhs-val": "SPY",
         "lhs-fn-params": {"window": 20}, "comparator": "gt", "rhs-val": "0.1", "rhs-fixed-value?": True,
         "children": [{"step": "asset", "ticker": "SPY"}]}
    sym = {"step": "root", "rebalance": "daily", "children": [{"step": "if", "children": [
        c, {"step": "if-child", "is-else-condition?": True, "children": [{"step": "asset", "ticker": "BIL"}]}]}]}
    assert ci.convert(sym)["tree"]["if"] == "ma_return(tr, 20) > 0.001"


# ------------------------------------------------------------ 3. a requested start cut short by a young ticker

@needs("FNGU", "QQQ", "SPY")
def test_short_history_warning_names_the_requested_start_and_ticker():
    p = port("hold the top 1 of FNGU, QQQ, SPY by 20 day return, rebalance monthly, since 2019")
    res = runner.run(p)
    notes = res.notes if getattr(res, "notes", None) else p.notes
    w = [n for n in notes if n.startswith("Warning: You asked to start on 2019-01-01")]
    assert w, notes
    assert "FNGU (from 20" in w[0] and "years of the requested period are missing" in w[0]
    warm = next(n for n in notes if n.startswith("Warm-up:"))
    assert "instead of the requested 2019-01-01" in warm


# ------------------------------------------------------------ 4-5. parser phrases

def test_group_lists_parse_before_and_after_the_weighting_phrase():
    a = port("inverse volatility weighted (50% SPY and 50% QQQ), (TLT and GLD equally) using a 30 day lookback").tree
    b = port("hold (50% SPY and 50% QQQ) and (TLT and GLD equally), inverse volatility weighted over 30 days").tree
    c = port("inverse volatility weighted (50% SPY and 50% QQQ) and (TLT and GLD equally) using a 30 day lookback").tree
    assert a == b == c
    assert a["weights"] == "inverse_vol" and a["lookback"] == 30 and len(a["children"]) == 2


def test_fortnightly_and_unknown_schedules():
    for w in ("fortnightly", "biweekly", "every other week"):
        p = port(f"hold 60% SPY and 40% TLT, rebalance {w}")
        assert p.rebalance == "every_2_weeks", w
    with pytest.raises(ParseError, match="unknown rebalancing schedule 'hourly'"):
        parser.parse("hold 60% SPY and 40% TLT, rebalance hourly")


def test_portfolio_stops_are_refused_with_the_equivalent():
    with pytest.raises(ParseError, match="stops and profit targets are for signal strategies.*if SPY is down 10%"):
        parser.parse("hold 60% SPY and 40% TLT, stop loss 10%")
    with pytest.raises(ParseError, match="portfolio's own value"):
        parser.parse("hold 60% SPY and 40% TLT, go to cash when the portfolio falls 10% from its peak")


def test_bare_short_is_a_static_short_allocation():
    for text in ("short SQQQ", "go short SQQQ"):
        p = port(text)
        assert p.tree == port("hold 100% short SQQQ").tree
        assert "no entry condition" in p.notes[0]
    p = port("go short SQQQ, since 2015")
    assert p.start == "2015-01-01"
    # with a condition it is still a trading rule
    assert not isinstance(parser.parse("short SQQQ when its 2 day RSI is above 90, cover after 3 days"), Portfolio)


# ------------------------------------------------------------ 6. defaults get notes

def test_default_rsi_window_and_top_n_notes():
    p = port("if SPY RSI is above 70 then BIL else SPY")
    assert "rsi(close, 14)" in p.tree["if"] and any("No RSI period given" in n for n in p.notes)
    p = port("if SPY 10 day RSI is above 70 then BIL else SPY")
    assert not any("No RSI period" in n for n in p.notes)
    p = port("hold the top 3 of SPY and QQQ by 20 day return")
    assert any("top 3 of 2 candidates" in n for n in p.notes)


# ------------------------------------------------------------ 7. Composer export fidelity

def _blocks(b):
    yield b
    for k in b.get("children") or []:
        yield from _blocks(k)


def test_export_keeps_ids_group_names_and_both_window_formats():
    src = json.loads(SYM2.read_text())
    p = Portfolio.from_dict(dict(ci.convert(src)))
    sym, _ = ce.export(p)
    ids_in = {b["id"] for b in _blocks(src)}
    # (round 13: the content hangs from a wt-cash-equal block under the root, as in Composer's own exports; this
    # symphony had none, so the export adds one with a generated id)
    assert sym["children"][0]["step"] == "wt-cash-equal" and len(sym["children"][0]["children"]) == 1
    ids_out = {b["id"] for b in _blocks(sym)} - {sym["children"][0]["id"]}
    # every block that survives the import keeps its id (the root too); only the collapsed wrappers are gone
    assert sym["id"] == src["id"] and ids_out <= ids_in and len(ids_out) >= len(ids_in) - 1
    group = next(b for b in _blocks(sym) if b["step"] == "group")
    assert group["name"] == "Trend switch" and group["id"] == "b2c0e6a1-0d3e-4c54-9d1a-5a0b6a1f0002"
    assert group["weight"] == {"num": 50, "den": 100}
    ic = next(b for b in _blocks(sym) if b["step"] == "if-child" and not b["is-else-condition?"])
    assert ic["lhs-fn-params"] == {"window": 20} and ic["lhs-window-days"] == "20"
    assert ic["rhs-fn-params"] == {"window": 50} and ic["rhs-window-days"] == "50"
    flt = next(b for b in _blocks(sym) if b["step"] == "filter")
    assert flt["sort-by-fn-params"] == {"window": 20} and flt["sort-by-window-days"] == "20"
    assert flt["select-fn"] == "bottom" and flt["select-n"] == 1
    assert sym["rebalance"] == "none" and sym["rebalance-corridor-width"] == 0.05
    # round trip: the same tree, ids included
    back = Portfolio.from_dict(dict(ci.convert(json.dumps(sym))))
    assert back.tree == p.tree and back.drift_band == p.drift_band


def test_export_generates_stable_uuids_when_there_are_none():
    p = port("if SPY is above its 200 day moving average and TQQQ 10 day RSI is below 79 then hold TQQQ else hold 60% "
             "SPY and 40% TLT, rebalance daily")
    a, _ = ce.export(p)
    b, _ = ce.export(port("if SPY is above its 200 day moving average and TQQQ 10 day RSI is below 79 then hold TQQQ else "
                          "hold 60% SPY and 40% TLT, rebalance daily"))
    assert a == b
    ids = [x["id"] for x in _blocks(a)]
    assert len(ids) == len(set(ids)) and all(uuid.UUID(i) for i in ids)
    # "and" is one if-child with Composer's "all" condition block
    ic = next(x for x in _blocks(a) if x["step"] == "if-child" and "condition" in x)
    assert ic["condition"]["condition-type"] == "compound" and ic["condition"]["operator"] == "all"
    rsi = next(c for c in ic["condition"]["conditions"] if c["lhs"]["fn"] == "relative-strength-index")
    assert rsi["lhs"] == {"fn": "relative-strength-index", "ticker": "TQQQ", "params": {"window": 10}}
    assert rsi["comparator"] == "lt" and rsi["rhs"] == {"constant": 79}
    # the import keeps the ids, and exporting that gives the same symphony again
    again, _ = ce.export(Portfolio.from_dict(dict(ci.convert(json.dumps(a)))))
    assert again["children"] == a["children"]


def test_import_accepts_leftover_weights_and_editor_flags():
    sym = {"step": "root", "rebalance": "daily", "children": [{"step": "wt-cash-equal", "children": [
        {"step": "asset", "ticker": "SPY", "weight": {"num": 100, "den": 100}, "collapsed-specified-weight?": True,
         "suppress-incomplete-warnings?": True},
        {"step": "asset", "ticker": "TLT"}]}]}
    d = ci.convert(sym)
    assert pf.without_ids(d["tree"]) == {"weights": "equal", "children": [{"asset": "SPY"}, {"asset": "TLT"}]}
    assert any("ignores it there" in n for n in d["notes"])
