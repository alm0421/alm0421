"""Round 7 (Composer expert review): price levels of different tickers compared or ranked on quoted prices
(quoted()), strict "more than" / inclusive "at least", equal thirds from 33/33/33, headline tiles that never wrap
inside a number, new phrasings (worse than, the N day SMA of X crosses ..., the top N by ... of ..., rebalance every
N days, not (...)), the Composer export, ticker autocomplete on the Build page and the speed of daily index filters."""
import glob
import json
import re
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pandas as pd
import pytest

from backtester import composer_export as ce, composer_import as ci, data, expr, parser, portfolio as pf, report, runner, web
from backtester.parser import ParseError
from backtester.portfolio import Portfolio

AVAIL = set(data.available_tickers())
needs = lambda *ts: pytest.mark.skipif(not set(ts) <= AVAIL, reason="price data not downloaded")  # noqa: E731


def port(text):
    p = parser.parse(text)
    assert isinstance(p, Portfolio), text
    return p


def sig(text):
    s = parser.parse(text)
    assert not isinstance(s, Portfolio), text
    return s


# ------------------------------------------------------------ 1. price levels across tickers are read as quoted

@pytest.mark.parametrize("rule, on, want", [
    ('close > sym("HYG").close', "GLD", 'quoted(close) > quoted(sym("HYG").close)'),
    ('sma(close, 50) > sma(sym("HYG").close, 50)', "GLD", 'quoted(sma(close, 50)) > quoted(sma(sym("HYG").close, 50))'),
    ('ema(sym("A").close, 20) < ema(sym("B").close, 20)', "SPY", 'quoted(ema(sym("A").close, 20)) < quoted(ema(sym("B").close, 20))'),
    ('close > 1.05 * sym("HYG").close', "GLD", 'quoted(close) > 1.05 * quoted(sym("HYG").close)'),
    ('crossover(sma(close, 20), sma(sym("HYG").close, 20))', "GLD",
     'crossover(quoted(sma(close, 20)), quoted(sma(sym("HYG").close, 20)))'),
    ('stdev(close, 20) > stdev(sym("HYG").close, 20)', "GLD", 'quoted(stdev(close, 20)) > quoted(stdev(sym("HYG").close, 20))'),
    # within one series the total-return basis is consistent: left alone
    ("close > sma(close, 200)", "GLD", "close > sma(close, 200)"),
    ('close > sym("GLD").close * 1.0', "GLD", 'close > sym("GLD").close * 1.0'),
    # ratios, returns and oscillators are comparable across tickers as they are
    ('rsi(close, 10) > rsi(sym("HYG").close, 10)', "GLD", 'rsi(close, 10) > rsi(sym("HYG").close, 10)'),
    ('tret(tr, 20) > tret(sym("HYG").tr, 20)', "GLD", 'tret(tr, 20) > tret(sym("HYG").tr, 20)'),
    ('close / sma(close, 20) > sym("HYG").close / sma(sym("HYG").close, 20)', "GLD",
     'close / sma(close, 20) > sym("HYG").close / sma(sym("HYG").close, 20)'),
    # a fixed price level (the earlier rule) still works
    ("close > 400", "SPY", "quoted(close) > 400"),
    ('quoted(close) > quoted(sym("HYG").close)', "GLD", 'quoted(close) > quoted(sym("HYG").close)'),   # idempotent
])
def test_quote_levels_across_tickers(rule, on, want):
    assert pf.quote_levels(rule, on) == want


def test_quoted_cross_rules_stay_open_safe_correct():
    assert expr.open_safe('quoted(open) > quoted(sym("HYG").open)')
    assert expr.open_safe('quoted(ref(close, 1)) > quoted(ref(sym("HYG").close, 1))')
    assert not expr.open_safe('quoted(close) > quoted(sym("HYG").close)')


def test_ranking_by_a_price_level_uses_quoted_prices():
    assert pf.quote_metric("close") == "quoted(close)"
    assert pf.quote_metric("sma(close, 20)") == "quoted(sma(close, 20))"
    assert pf.quote_metric("stdev(close, 20)") == "quoted(stdev(close, 20))"
    for m in ("tret(tr, 20)", "rsi(close, 10)", "close / sma(close, 20)", "volatility(20)", "quoted(close)"):
        assert pf.quote_metric(m) == m


def test_validate_quotes_json_specs_and_notes_why():
    p = Portfolio(tree={"if": 'sma(close, 50) > sma(sym("HYG").close, 50)', "on": "GLD", "then": {"asset": "GLD"},
                        "else": {"asset": "HYG"}})
    p.validate()
    assert p.tree["if"] == 'quoted(sma(close, 50)) > quoted(sma(sym("HYG").close, 50))'
    assert any("price levels of different tickers" in n for n in p.notes)
    f = Portfolio(tree={"filter": {"select": "top", "n": 1, "by": "close"}, "universe": ["GLD", "HYG"]})
    f.validate()
    assert f.tree["filter"]["by"] == "quoted(close)" and any("Ranking by `close`" in n for n in f.notes)
    q = Portfolio(tree={"if": 'close > sym("HYG").close', "on": "GLD", "then": {"asset": "GLD"}, "else": {"asset": "HYG"}},
                  price_basis="quoted")
    q.validate()
    assert q.tree["if"] == 'close > sym("HYG").close'     # prices as quoted already


def test_composer_import_quotes_cross_ticker_prices_and_rankings():
    sym = {"step": "root", "rebalance": "daily", "children": [
        {"step": "if", "children": [
            {"step": "if-child", "is-else-condition?": False, "lhs-fn": "moving-average-price", "lhs-val": "GLD",
             "lhs-window-days": 50, "comparator": "gt", "rhs-fixed-value?": False, "rhs-fn": "moving-average-price",
             "rhs-val": "HYG", "rhs-window-days": 50, "children": [{"step": "asset", "ticker": "GLD"}]},
            {"step": "if-child", "is-else-condition?": True, "children": [
                {"step": "filter", "sort-by-fn": "current-price", "select-fn": "top", "select-n": 1,
                 "children": [{"step": "asset", "ticker": "TLT"}, {"step": "asset", "ticker": "IEF"}]}]}]}]}
    d = ci.convert(sym)
    assert d["tree"]["if"] == 'quoted(sma(close, 50)) > quoted(sma(sym("HYG").close, 50))'
    assert d["tree"]["else"]["filter"]["by"] == "quoted(close)"
    assert any("another ticker" in n for n in d["notes"]) and any("ranking by current-price" in n for n in d["notes"])


@needs("GLD", "HYG")
def test_parser_cross_ticker_levels_and_decisions_match_quoted_closes():
    p = port("if GLD price is above HYG price then GLD else HYG, rebalance daily, since 2015")
    assert p.tree["if"] == 'quoted(close) > quoted(sym("HYG").close)'
    assert any("price levels of different tickers" in n for n in p.notes)
    r = runner.run(p)
    g, h = data.load("GLD"), data.load("HYG")
    q = pd.concat([g["close"], h["close"]], axis=1, keys=["g", "h"]).dropna()
    want = (q.g > q.h).reindex(r.holdings.index)
    held = r.holdings["GLD"] > 0.5
    assert (held == want).all()
    # the decision differs from the total-return levels on dates where those cross (HYG's level carries 20+ years
    # of reinvested coupons): 2020-01-02 quoted 143.95 vs 88.31, total-return 143.95 vs ~206
    ag, ah = expr.adjusted_frame(g)["close"], expr.adjusted_frame(h)["close"]
    d = pd.Timestamp("2020-01-02")
    assert g.at[d, "close"] > h.at[d, "close"] and ag.at[d] < ah.at[d] and held.at[d]
    # ranking by the current price and a moving average of it: the same quoted comparison
    k = port("hold the top 1 of GLD and HYG by current price, rebalance daily, since 2015")
    assert k.tree["filter"]["by"] == "quoted(close)"
    m = port("if the 50 day SMA of GLD is above the 50 day SMA of HYG then GLD else HYG, since 2015")
    assert m.tree["if"] == 'quoted(sma(close, 50)) > quoted(sma(sym("HYG").close, 50))'


# ------------------------------------------------------------ 2. strict and inclusive bounds

@needs("SPY", "BIL", "TQQQ")
@pytest.mark.parametrize("cond, rule", [
    ("SPY is more than 5% above its 200 day moving average", "(close > sma(close, 200) * 1.05)"),
    ("SPY is at least 5% above its 200 day moving average", "(close >= sma(close, 200) * 1.05)"),
    ("SPY is 5% or more above its 200 day moving average", "(close >= sma(close, 200) * 1.05)"),
    ("SPY is more than 5% below its 200 day moving average", "(close < sma(close, 200) * 0.95)"),
    ("SPY has fallen more than 5%", "(tret(tr, 1) < -0.05)"),
    ("SPY has fallen at least 5%", "(tret(tr, 1) <= -0.05)"),
    ("SPY has fallen 5% or more", "(tret(tr, 1) <= -0.05)"),
    ("SPY rose over 5% in the last 10 days", "(tret(tr, 10) > 0.05)"),
    ("SPY is more than 10% below its 52 week high", "(drawdown(close, 252) < -0.1)"),
    ("SPY is at most 10% below its 52 week high", "(drawdown(close, 252) >= -0.1)"),
    ("SPY gaps down more than 2%", "(gap < -0.02)"),
    ("SPY 10 day RSI is 70 or more", "(rsi(close, 10) >= 70)"),
    ("SPY 10 day RSI is 30 or less", "(rsi(close, 10) <= 30)"),
    ("SPY 10 day RSI is more than 70", "(rsi(close, 10) > 70)"),
    ("SPY 10 day RSI exceeds 70", "rsi(close, 10) > 70"),
    ("SPY 10 day RSI is under 30", "(rsi(close, 10) < 30)"),
    ("SPY 10 day RSI is no more than 30", "(rsi(close, 10) <= 30)"),
    ("SPY volume is more than 2 times its 20 day average", "(volume > 2 * sma(volume, 20))"),
])
def test_strict_and_inclusive_bounds_in_portfolio_conditions(cond, rule):
    assert port(f"if {cond} then hold TQQQ else hold BIL").tree["if"] == rule


@needs("SPY")
@pytest.mark.parametrize("cond, rule", [
    ("it is more than 5% above its 200 day moving average", "(close > sma(close, 200) * 1.05)"),
    ("it has fallen more than 5%", "(change < -0.05)"),
    ("it has fallen 5%", "(change <= -0.05)"),
    ("it closes at least 2 ATRs below its 20 day moving average", "(close <= sma(close, 20) - 2 * atr(14))"),
    ("it closes 2 ATRs below its 20 day moving average", "(close < sma(close, 20) - 2 * atr(14))"),
])
def test_strict_and_inclusive_bounds_in_signal_rules(cond, rule):
    assert sig(f"buy SPY when {cond}, sell after 5 days").entry == rule


# ------------------------------------------------------------ 3. equal weights

@needs("SPY", "TLT", "GLD", "TQQQ", "TMF", "SVIX")
@pytest.mark.parametrize("text, n", [
    ("33% SPY, 33% TLT, 33% GLD", 3), ("SPY, TLT, GLD 33% / 33% / 33%", 3), ("SPY, TLT and GLD weighted 33/33/33", 3),
    ("33.3% SPY, 33.3% TLT, 33.3% GLD", 3), ("33.33% SPY, 33.33% TLT and 33.33% GLD", 3),
    ("SPY, TLT and GLD in equal thirds", 3), ("SPY, TLT and GLD in equal parts", 3), ("SPY, TLT and GLD equally weighted", 3),
    ("equal thirds of SPY, TLT and GLD", 3), ("TQQQ and TMF equally", 2), ("hold SPY and TLT equally, rebalance monthly", 2),
])
def test_equal_weights(text, n):
    p = port(text)
    assert p.tree["weights"] == "equal" and len(p.tree["children"]) == n
    assert not any(k.get("cash") for k in p.tree["children"])


@needs("SPY", "TLT", "GLD", "TQQQ", "TMF", "SVIX")
def test_equal_weight_notes_and_specified_slashes():
    p = port("33% SPY, 33% TLT, 33% GLD")
    assert any("read as equal weights" in n for n in p.notes)
    q = port("TQQQ, TMF and SVIX weighted 50/30/20")
    assert q.tree["weights"] == "specified" and q.tree["w"] == [0.5, 0.3, 0.2]
    assert port("30% SPY, 30% TLT, 30% GLD").tree["children"][-1] == {"cash": True}   # 90%: the rest really is cash
    with pytest.raises(ParseError, match="thirds of 2"):
        parser.parse("SPY and TLT in equal thirds")


# ------------------------------------------------------------ 4. headline tiles

def test_report_tiles_never_wrap_inside_a_number():
    tpl = (report.Path(report.__file__).parent / "report_template.html").read_text()
    css = re.search(r"\.tile \.v \{[^}]*\}", tpl).group(0)
    assert "nowrap" in css and "overflow-wrap: anywhere" not in css
    assert "function fitTiles" in tpl and "const pctH" in tpl


def _chromium():
    try:
        import playwright  # noqa: F401
    except ImportError:
        return None
    return (glob.glob("/opt/pw-browsers/chromium*/chrome-linux*/chrome") or [None])[0]


@pytest.mark.skipif(_chromium() is None or not {"SPY", "TLT"} <= AVAIL, reason="headless Chromium or data not available")
def test_report_tiles_fit_at_desktop_and_phone_width(tmp_path):
    from playwright.sync_api import sync_playwright
    res = runner.run(Portfolio(tree={"weights": "specified", "w": [0.6, 0.4], "children": [{"asset": "SPY"}, {"asset": "TLT"}]},
                               start="2015-01-01", end="2019-12-31"))
    A = report.analyze(res, sensitivity=False)
    # huge numbers, as a daily-rebalanced or leveraged strategy produces
    A["turnover"] = 49.34
    A["stats"].update(best_year=11.46, worst_year=0.02, cagr=24.67, total_return=123456.7, end_equity=1.23e9)
    path = report.write_outputs(A, tmp_path / "r", excel=False)
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=_chromium())
        for w in (1280, 390):
            pg = b.new_page(viewport={"width": w, "height": 900})
            pg.goto(path.resolve().as_uri())
            pg.wait_for_function("document.querySelectorAll('#tiles .tile').length > 5")
            pg.wait_for_timeout(300)
            rows = pg.evaluate("""() => [...document.querySelectorAll('#tiles .tile .v')].map(v => {
                const r = document.createRange(); r.selectNodeContents(v);
                return [v.textContent, r.getBoundingClientRect().width <= v.clientWidth + 0.5, v.getClientRects().length,
                        Math.round(v.getBoundingClientRect().height)]; })""")
            vals = [r[0] for r in rows]
            assert "2,467%/yr" in vals and "1,146% / 2%" in vals and "2,467%" in vals, vals
            for text, fits, _, height in rows:
                assert fits, (w, text)
                assert height < 34, (w, text, height)        # one line
            assert pg.evaluate("document.documentElement.scrollWidth") <= w
            pg.close()
        b.close()


# ------------------------------------------------------------ 5. phrasings

@needs("SPY", "BIL", "QQQ", "TLT")
@pytest.mark.parametrize("text, check", [
    ("if SPY 60 day max drawdown is worse than 10% then hold BIL else hold SPY",
     lambda t: t["if"] == "max_drawdown(tr, 60) > 0.1"),
    ("if SPY 60 day max drawdown is worse than -10% then hold BIL else hold SPY",
     lambda t: t["if"] == "max_drawdown(tr, 60) > 0.1"),
    ("if SPY 60 day max drawdown is better than 10% then hold SPY else hold BIL",
     lambda t: t["if"] == "max_drawdown(tr, 60) < 0.1"),
    ("if SPY drawdown is worse than 10% then hold BIL else hold SPY", lambda t: t["if"] == "drawdown(close) < -0.1"),
    ("if the 20 day SMA of SPY crosses below its 50 day SMA then hold BIL else hold SPY",
     lambda t: t["if"] == "(crossunder(sma(close, 20), sma(close, 50)))" and t["on"] == "SPY"),
    ("hold the top 2 by 10 day RSI of SPY, QQQ, TLT",
     lambda t: t["filter"]["select"] == "top" and t["filter"]["n"] == 2 and t["filter"]["by"] == "rsi(close, 10)"
     and [k["asset"] for k in t["children"]] == ["SPY", "QQQ", "TLT"]),
    ("hold the highest 10 day RSI of SPY, QQQ, TLT",
     lambda t: t["filter"]["select"] == "top" and t["filter"]["n"] == 1 and t["filter"]["by"] == "rsi(close, 10)"),
    ("hold the lowest 10 day RSI of SPY, QQQ and TLT",
     lambda t: t["filter"]["select"] == "bottom" and t["filter"]["n"] == 1 and t["filter"]["by"] == "rsi(close, 10)"),
    ("if it is not the case that SPY is above its 200 day moving average then hold BIL else hold SPY",
     lambda t: t["if"] == "not ((close > sma(close, 200)))"),
    ("if not (SPY is above its 200 day moving average or QQQ 10 day RSI is above 80) then hold QQQ else hold BIL",
     lambda t: t["if"] == 'not (((close > sma(close, 200)) or (rsi(sym("QQQ").close, 10) > 80)))'),
])
def test_new_phrasings(text, check):
    assert check(port(text).tree)


@needs("SPY")
def test_signal_negation_and_crossover_of_named_averages():
    assert sig("buy SPY when not (RSI(2) is above 10), sell after 5 days").entry == "not ((rsi(close, 2) > 10))"
    assert (sig("buy SPY when the 20 day SMA of SPY crosses below its 50 day SMA, sell after 5 days").entry
            == "(crossunder(sma(close, 20), sma(close, 50)))")


def test_every_n_rebalance_schedule():
    cal = pd.bdate_range("2021-01-01", "2021-06-30")
    d = pf._schedule(cal, "every_3_days")
    assert d[0] and list(np.flatnonzero(d)[:4]) == [0, 3, 6, 9]
    w1 = np.flatnonzero(pf._schedule(cal, "weekly")[1:]) + 1
    w3 = pf._schedule(cal, "every_3_weeks")
    assert w3[0] and list(np.flatnonzero(w3[1:]) + 1) == list(w1[2::3])
    m1 = np.flatnonzero(pf._schedule(cal, "monthly")[1:]) + 1
    m2 = pf._schedule(cal, "every_2_months")
    assert m2[0] and list(np.flatnonzero(m2[1:]) + 1) == list(m1[1::2])
    with pytest.raises(ValueError, match="every_N_days"):
        Portfolio(tree={"asset": "SPY"}, rebalance="every_0_days").validate()


@needs("SPY", "TLT")
def test_every_n_rebalance_phrases_run():
    p = port("hold SPY and TLT equally, rebalance every 2 days")
    assert p.rebalance == "every_2_days" and "every 2 trading days" in p.summary()
    q = port("hold 60% SPY and 40% TLT, rebalance every 3 weeks, since 2020")
    assert q.rebalance == "every_3_weeks"
    r = runner.run(q)
    years = (r.equity.index[-1] - r.equity.index[1]).days / 365.25
    assert abs(r.extras["rebalances"] - years * 52 / 3) < 6      # every third week-end, plus the first day
    assert port("hold 60% SPY and 40% TLT, rebalance every 2 months").rebalance == "every_2_months"
    assert port("hold 60% SPY and 40% TLT, rebalance every 3 months").rebalance == "quarterly"


# ------------------------------------------------------------ 6. Composer export

FIXTURE = json.loads((report.Path(__file__).parent / "fixtures" / "composer_tqqq_ftlt.json").read_text())


def test_composer_round_trip_import_export_import():
    d = ci.convert(FIXTURE)
    p = Portfolio.from_dict(dict(d))
    sym, notes = ce.export(p)
    assert sym["step"] == "root" and sym["rebalance"] == p.rebalance
    d2 = ci.convert(json.dumps(sym))
    p2 = Portfolio.from_dict(dict(d2))
    assert p2.tree == p.tree and p2.rebalance == p.rebalance and p2.drift_band == p.drift_band
    # and once more: the exported symphony is stable
    assert ce.export(p2)[0]["children"] == sym["children"]


@needs("SPY", "QQQ", "BIL", "TQQQ", "TLT", "GLD")
def test_composer_export_of_parsed_portfolios_gives_the_same_results():
    for text in ("if SPY is above its 200 day moving average then hold QQQ else hold BIL, rebalance daily, since 2015",
                 "if SPY is above its 200 day moving average and TQQQ 10 day RSI is below 79 then hold TQQQ else hold BIL, "
                 "rebalance daily, since 2015",
                 "if SPY is not above its 200 day moving average then BIL else SPY, rebalance daily, since 2015",
                 "hold the top 2 of SPY, QQQ, TLT, GLD by 60 day return, rebalance monthly, since 2015",
                 "hold 60% SPY and 40% TLT, rebalance quarterly, since 2015"):
        p = port(text)
        sym, notes = ce.export(p)
        back = Portfolio.from_dict(dict(ci.convert(json.dumps(sym))))
        back.start = p.start
        # round 12: Composer trades monthly / quarterly symphonies on the first trading day of each period, so the
        # import says rebalance_day "start" and the export of a period-end spec says the timing differs; the tree and
        # everything else round-trip exactly (compared here with the spec's own rebalance day)
        if p.rebalance in ("monthly", "quarterly"):
            assert back.rebalance_day == "start" and any("Rebalance timing" in n for n in notes), text
            back.rebalance_day = p.rebalance_day
        a, b = runner.run(p).equity, runner.run(back).equity
        assert np.allclose(a.to_numpy(), b.to_numpy(), rtol=0, atol=1e-9), text


@pytest.mark.parametrize("tree, msg", [
    ({"weights": "specified", "w": [-1.0, 2.0], "children": [{"asset": "SPY"}, {"cash": True}]}, "short"),
    ({"weights": "specified", "w": [0.6, 0.3], "children": [{"asset": "SPY"}, {"asset": "TLT"}]}, "cash"),
    ({"weights": "risk_parity", "children": [{"asset": "SPY"}, {"asset": "TLT"}]}, "risk parity"),
    ({"filter": {"select": "top", "n": 5, "by": "tret(tr, 20)"}, "universe": "NDX"}, "Nasdaq-100"),
    ({"filter": {"select": "top", "n": 1, "by": "tret(tr, 20)", "require": "tret(tr, 20) > 0"},
      "universe": ["SPY", "TLT"]}, "requirement"),
    ({"if": "adx(14) > 25", "on": "SPY", "then": {"asset": "SPY"}, "else": {"asset": "BIL"}}, "no Composer equivalent"),
    ({"if": "close != 3", "on": "SPY", "then": {"asset": "SPY"}, "else": {"asset": "BIL"}}, "single >"),
    ({"weights": "equal", "children": [{"asset": "SPY"}, {"cash": True}]}, "cash"),
])
def test_composer_export_refuses_what_composer_lacks(tree, msg):
    with pytest.raises(ce.ComposerExportError, match=msg):
        ce.export(Portfolio(tree=tree, rebalance="daily"))


def test_composer_export_settings():
    with pytest.raises(ce.ComposerExportError, match="leverage"):
        ce.export(Portfolio(tree={"asset": "SPY"}, leverage=2.0, maintenance_margin=0.25))
    with pytest.raises(ce.ComposerExportError, match="Volatility targeting"):
        ce.export(Portfolio(tree={"asset": "SPY"}, target_vol=0.1))
    with pytest.raises(ce.ComposerExportError, match="semiannual"):
        ce.export(Portfolio(tree={"asset": "SPY"}, rebalance="semiannual"))
    sym, notes = ce.export(Portfolio(tree={"asset": "SPY"}, rebalance="none", drift_band=0.05, slippage_bps=5,
                                     contribution=100))
    assert sym["rebalance"] == "none" and sym["rebalance-corridor-width"] == 0.05   # a fraction, as Composer stores it
    assert any("costs" in n for n in notes) and any("Cash flows" in n for n in notes)
    # an if with cash in the else branch: Composer's empty block, and the importer reads it back as cash
    sym, _ = ce.export(Portfolio(tree={"if": "rsi(close, 10) > 70", "on": "SPY", "then": {"asset": "SPY"},
                                       "else": {"cash": True}}, rebalance="daily"))
    assert sym["children"][0]["step"] == "wt-cash-equal"   # under the root, as Composer's own exports
    assert sym["children"][0]["children"][0]["children"][1]["children"][0]["step"] == "empty"
    assert ci.convert(sym)["tree"]["else"] == {"cash": True}
    # percent-valued Composer functions are written in percent
    sym, _ = ce.export(Portfolio(tree={"if": "tret(tr, 5) <= -0.06", "on": "QQQ", "then": {"asset": "QQQ"},
                                       "else": {"asset": "BIL"}}, rebalance="daily"))
    c = sym["children"][0]["children"][0]["children"][0]
    assert c["lhs-fn"] == "cumulative-return" and c["rhs-val"] == "-6" and c["comparator"] == "lte"
    with pytest.raises(ce.ComposerExportError, match="trading rule"):
        ce.export(parser.parse("buy SPY when RSI(2) is below 10, sell after 5 days"))


@needs("SPY", "QQQ", "BIL")
def test_composer_export_cli_and_api(tmp_path, capsys):
    from backtester import api
    from backtester.__main__ import main
    out = tmp_path / "s.json"
    assert main(["composer-export", "if SPY is above its 200 day moving average then hold QQQ else hold BIL",
                 "--out", str(out)]) == 0
    sym = json.loads(out.read_text())
    assert sym["step"] == "root" and sym["children"][0]["children"][0]["step"] == "if"
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"kind": "allocation", "tree": {"asset": "SPY"}, "rebalance": "monthly"}))
    assert main(["composer-export", str(spec)]) == 0
    assert json.loads(capsys.readouterr().out)["children"][0]["children"][0]["ticker"] == "SPY"
    assert main(["composer-export", "hold the top 5 Nasdaq 100 stocks by 20 day return"]) == 2
    assert "Nasdaq-100" in capsys.readouterr().err
    assert api.composer_export("hold 60% SPY and 40% QQQ")["children"][0]["children"][0]["step"] == "wt-cash-specified"


# ------------------------------------------------------------ 6/7. site: export endpoint, tickers, autocomplete

@pytest.fixture(scope="module")
def site(tmp_path_factory):
    old = web.RUNS
    web.RUNS = tmp_path_factory.mktemp("runs")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), web.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    web.RUNS = old


def _post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


@needs("SPY", "QQQ", "BIL")
def test_export_endpoint_and_ticker_list(site):
    code, j = _post(site + "/api/export/composer", {"spec": {"kind": "allocation", "rebalance": "daily", "tree": {
        "if": "close > sma(close, 200)", "on": "SPY", "then": {"asset": "QQQ"}, "else": {"asset": "BIL"}}}})
    assert code == 200 and j["symphony"]["children"][0]["children"][0]["step"] == "if"
    code, j = _post(site + "/api/export/composer", {"spec": {"kind": "allocation", "tree": {"asset": "SPY"}, "leverage": 2}})
    assert code == 400 and "leverage" in j["error"]
    with urllib.request.urlopen(site + "/api/tickers") as r:
        t = json.loads(r.read())
    assert "SPY" in t["tickers"] and len(t["tickers"]) > 100


@pytest.mark.skipif(_chromium() is None or not {"SPY", "QQQ", "TLT", "NVDA"} <= AVAIL, reason="headless Chromium or data not available")
def test_build_page_ticker_autocomplete_and_composer_button(site):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=_chromium())
        pg = b.new_page(viewport={"width": 1280, "height": 900}, accept_downloads=True)
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.goto(site + "/#build")
        pg.wait_for_function("document.querySelector('#tree .node') && window.fetch")
        pg.wait_for_timeout(300)
        # an asset box in the block editor: type, pick with the keyboard
        box = pg.locator("#tree input[placeholder='ticker']").first
        box.click()
        box.fill("")
        box.type("QQ")
        pg.wait_for_selector("#tickerAc:not(.hide) .acitem")
        items = pg.eval_on_selector_all("#tickerAc .acitem", "es => es.map(e => e.dataset.t)")
        assert items[0] == "QQQ" and all("QQ" in t for t in items)
        box.press("ArrowDown")
        box.press("Enter")
        assert box.input_value() == "QQQ" and pg.is_hidden("#tickerAc")
        assert pg.evaluate("JSON.stringify(document.querySelector('#specJson') && true)")
        # the benchmark box, with the mouse
        pg.fill("#p_benchmark", "")
        pg.click("#p_benchmark")
        pg.type("#p_benchmark", "TL")
        pg.wait_for_selector("#tickerAc:not(.hide) .acitem[data-t='TLT']")
        pg.click("#tickerAc .acitem[data-t='TLT']")
        assert pg.input_value("#p_benchmark") == "TLT"
        # the Composer export of the current blocks downloads a symphony
        with pg.expect_download() as dl:
            pg.click("#buildComposer")
        sym = json.loads(open(dl.value.path()).read())
        assert sym["step"] == "root"
        # several tickers in one box (signal mode): only the last one is completed
        pg.click("#buildMode button[data-v='signal']")
        pg.fill("#s_universe", "")
        pg.click("#s_universe")
        pg.type("#s_universe", "MSFT, NVD")
        pg.wait_for_selector("#tickerAc:not(.hide) .acitem[data-t='NVDA']")
        pg.click("#tickerAc .acitem[data-t='NVDA']")
        assert pg.input_value("#s_universe") == "MSFT, NVDA"
        assert errs == []
        b.close()


# ------------------------------------------------------------ 8. speed without changing results

def _loop_rank(self, n, mem, by, pit, i, top, k):
    """The per-day loop the vectorised ranking replaced (the reference)."""
    cands = []
    for m in mem:
        if not self.has(m, i) or (pit and not self.is_member(m, i)):
            continue
        v = self.mseries(by, m, "value")[i]
        if np.isfinite(v):
            cands.append((v, m))
    cands.sort(key=lambda x: x[0], reverse=top)
    return [m for _, m in cands[:k]]


@needs("SPY", "QQQ", "TLT", "GLD", "IWM", "EFA")
def test_vectorised_ranking_matches_the_loop(monkeypatch):
    specs = [Portfolio(tree={"filter": {"select": sel, "n": 2, "by": "tret(tr, 5)"},
                             "universe": ["SPY", "QQQ", "TLT", "GLD", "IWM", "EFA"]}, rebalance="daily", start="2012-01-01")
             for sel in ("top", "bottom")]
    fast = [runner.run(Portfolio.from_dict(json.loads(s.to_json()))).equity for s in specs]
    monkeypatch.setattr(pf._Evaluator, "_rank_assets", _loop_rank)
    slow = [runner.run(Portfolio.from_dict(json.loads(s.to_json()))).equity for s in specs]
    for a, b in zip(fast, slow):
        assert (a - b).abs().max() <= 1e-9


@needs("SPY", "QQQ", "TLT", "GLD")
def test_cost_sensitivity_reuses_evaluations_with_the_same_results():
    p = Portfolio(tree={"filter": {"select": "top", "n": 1, "by": "tret(tr, 20)"}, "universe": ["SPY", "QQQ", "TLT", "GLD"]},
                  rebalance="daily", start="2015-01-01")
    res = runner.run(p)
    rows = report.cost_sensitivity(res, levels=(0, 10))
    alone = runner.run(Portfolio.from_dict({**json.loads(p.to_json()), "slippage_bps": 10.0}))
    assert abs(rows[1]["final_equity"] - alone.equity.iloc[-1]) < 1e-6


@needs("SPY", "TLT")
def test_fast_excel_writer_round_trips_values(tmp_path):
    res = runner.run(Portfolio(tree={"weights": "specified", "w": [0.6, 0.4], "children": [{"asset": "SPY"}, {"asset": "TLT"}]},
                               start="2018-01-01", end="2019-12-31"))
    A = report.analyze(res, sensitivity=False)
    report.write_outputs(A, tmp_path / "r")
    x = pd.read_excel(tmp_path / "r" / "report.xlsx", sheet_name=None)
    assert {"Summary", "Equity", "Orders", "Holdings (month-end)"} <= set(x)
    eq = x["Equity"].set_index("date")["equity"]
    ref = res.equity.reindex(pd.DatetimeIndex(eq.index))
    assert len(eq) >= len(res.equity) - 1 and ref.notna().all() and np.allclose(eq.to_numpy(), ref.to_numpy())
    o = x["Orders"]
    assert list(o.columns[:3]) == ["date", "ticker", "side"] and len(o) == len(res.orders)
