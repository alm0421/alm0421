"""Portfolio Visualizer review, round 12: asset-class words read by one rule in sentences and the grid (never a
lowercase word as an unrelated stock), the 4% rule paid monthly, Monte Carlo stress honesty (historical floor and
warnings), month-end lookbacks for the named tactical models, correlation period labels, pre-1987 crises,
trailing returns for every portfolio of a report, and Portfolio Visualizer's yearly rebalancing for named mixes."""
import glob

import numpy as np
import pandas as pd
import pytest

from backtester import data, metrics, parser, report, runner
from backtester import montecarlo as mc

HAVE = set(data.available_tickers())


def needs(*t):
    return pytest.mark.skipif(not set(t) <= HAVE, reason="price data not downloaded")


def port(text):
    p = parser.parse(text)
    assert not hasattr(p, "entry"), text
    return p


def assets(p):
    t = p.tree
    return [k.get("asset") for k in t["children"]] if "children" in t else [t.get("asset")]


# ------------------------------------------------------------ 1. asset-class words: one rule, never a stray stock

@needs("SPY", "GLD", "GLDSIM", "GOLD")
@pytest.mark.parametrize("text,want", [
    ("hold 60% SPY and 40% gold since 1972", ["SPY", "GLDSIM"]),      # the start is before GLD: its long series
    ("hold 60% SPY and 40% gold", ["SPY", "GLD"]),                    # next to a ticker, no start: the fund
    ("hold 60% SPY and 40% Gold", ["SPY", "GLD"]),
    ("hold 60% SPY and 40% gold since 2010", ["SPY", "GLD"]),         # the fund existed then
    ("hold 60% SPY and 40% GOLD", ["SPY", "GOLD"]),                   # capitals: the ticker (Barrick Gold)
    ("hold SPY and gold", ["SPY", "GLD"]),
    ("SPY 60% gold 40%", ["SPY", "GLD"]),
    ("hold 60% SPY, 40% GOLD since 2015", ["SPY", "GOLD"]),
])
def test_gold_is_gold_not_barrick(text, want):
    p = port(text)
    assert assets(p) == want, (text, p.tree)
    if want[-1] != "GOLD":
        assert any(n.startswith("Warning:") and "GOLD" in n and "capitals" in n for n in p.notes), p.notes


@needs("SPY", "GLD", "GLDSIM")
def test_since_1972_note_names_the_stand_in_and_the_fund():
    p = port("hold 60% SPY and 40% gold since 1972")
    n = next(x for x in p.notes if x.startswith("'gold' (gold) is read as GLDSIM"))
    assert "before GLD existed" in n and "GLD itself" in n and "Say GLD" in n


@needs("VTISIM", "GLDSIM", "VTI", "GLD")
def test_asset_classes_alone_use_the_long_series_unless_the_start_is_late():
    assert assets(port("60% stocks and 40% gold since 1972")) == ["VTISIM", "GLDSIM"]
    assert assets(port("60% stocks and 40% gold")) == ["VTISIM", "GLDSIM"]     # no start: as early as possible
    assert assets(port("60% stocks and 40% gold since 2010")) == ["VTI", "GLD"]


@needs("SPY", "BND", "VNQ", "IEF", "TIP", "DBC")
@pytest.mark.parametrize("text,want", [
    ("hold 60% SPY and 40% bonds", "BND"), ("hold 60% SPY and 40% reits", "VNQ"), ("hold 60% SPY and 40% REITs", "VNQ"),
    ("hold 60% SPY and 40% real estate", "VNQ"), ("hold 60% SPY and 40% tips", "TIP"),
    ("hold 60% SPY and 40% commodities", "DBC"), ("hold 60% SPY and 40% treasuries", "IEF"),
])
def test_common_asset_class_words_next_to_tickers(text, want):
    p = port(text)
    assert assets(p) == ["SPY", want], (text, p.tree)


@needs("SPY", "BNDSIM", "VNQSIM")
def test_bonds_and_reits_with_an_early_start():
    assert assets(port("hold 60% SPY and 40% bonds since 1990")) == ["SPY", "BNDSIM"]
    assert assets(port("hold 60% SPY and 40% reits since 1990")) == ["SPY", "VNQSIM"]


@needs("SPY", "IEF")
def test_treasuries_without_a_maturity_warns():
    p = port("hold 60% SPY and 40% treasuries")
    assert any(n.startswith("Warning:") and "long-term treasuries" in n and "TLT" in n for n in p.notes)


@needs("SPY", "GLD", "GLDSIM", "GOLD")
def test_grid_reads_asset_classes_by_the_same_rule():
    from backtester import web
    g = {"rows": [{"asset": "SPY", "w": [60]}, {"asset": "gold", "w": [40]}], "rebalance": "yearly"}
    s, _, pr = web.grid_specs({**g, "start": "1972"})
    assert not pr and [k["asset"] for k in s[0].tree["children"]] == ["SPY", "GLDSIM"]
    s, _, pr = web.grid_specs({**g, "start": "2010"})
    assert not pr and [k["asset"] for k in s[0].tree["children"]] == ["SPY", "GLD"]
    s, _, pr = web.grid_specs(g)
    assert not pr and [k["asset"] for k in s[0].tree["children"]] == ["SPY", "GLD"]
    s, _, pr = web.grid_specs({**g, "rows": [{"asset": "SPY", "w": [60]}, {"asset": "GOLD", "w": [40]}]})
    assert not pr and [k["asset"] for k in s[0].tree["children"]] == ["SPY", "GOLD"]
    # asset classes alone, no start: the long-history series (Portfolio Visualizer's asset-class mode)
    s, _, pr = web.grid_specs({"rows": [{"asset": "US Stock Market", "w": [60]}, {"asset": "Gold", "w": [40]}]})
    assert not pr and [k["asset"] for k in s[0].tree["children"]][1] == "GLDSIM"


def test_class_series_rule_directly():
    if not {"GLD", "GLDSIM"} <= HAVE:
        pytest.skip("price data not downloaded")
    assert parser.class_series("gold", "1972-01-01", True)[0] == "GLDSIM"
    assert parser.class_series("gold", None, True)[0] == "GLD"
    assert parser.class_series("gold", None, False)[0] == "GLDSIM"
    assert parser.class_series("gold", "2012-01-01", False)[0] == "GLD"
    assert parser.class_series("not an asset class", None, True) is None


# ------------------------------------------------------------ 2. the 4% rule, paid monthly

@needs("SPY", "TLT")
@pytest.mark.parametrize("text", [
    "hold 60% SPY and 40% TLT, start with $1,000,000, withdraw 4% a year taken monthly adjusted for inflation",
    "hold 60% SPY and 40% TLT, start with $1,000,000, withdraw 4% per year adjusted for inflation, taken monthly",
])
def test_four_percent_rule_taken_monthly(text):
    p = port(text)
    assert p.withdrawal_freq == "monthly" and p.withdrawal == pytest.approx(1_000_000 * 0.04 / 12)
    assert p.withdrawal_pct == 0 and p.withdrawal_inflation
    n = next(x for x in p.notes if "rule'" in x)
    assert "$40,000 a year" in n and "$3,333 a month" in n
    assert not any("of the balance a year taken every month" in x for x in p.notes)


@needs("SPY", "TLT")
def test_four_percent_rule_yearly_note_unchanged():
    p = port("hold 60% SPY and 40% TLT, start with $1,000,000, withdraw 4% a year adjusted for inflation")
    assert p.withdrawal == pytest.approx(40_000) and any("'4% rule'" in n and "$40,000 a year" in n for n in p.notes)


# ------------------------------------------------------------ 3. Monte Carlo stress honesty

@needs("SPY", "TLT", "SPYSIM", "TLTSIM")
def test_worst_sequence_has_a_long_history_floor_and_warnings():
    base = dict(weights={"SPY": 60, "TLT": 40}, model="t", years=30, start_balance=1_000_000, sims=400, seed=1,
                flows=[mc.CashFlow(amount=-40_000)])
    N = mc.run(mc.Settings(**base))
    assert any(n.startswith("Warning: the history window is only") and "SPYSIM for SPY" in n for n in N["notes"])
    S = mc.run(mc.Settings(**base, stress="worst_sequence"))
    assert S["stress"]["source"] == "long_history" and S["stress"]["from"].year < 1990
    assert any(n.startswith("Stress floor:") for n in S["notes"])
    assert S["prob_success"] <= S["prob_success_unstressed"]
    W = mc.run(mc.Settings(**base, stress="worst_sequence", stress_history="window"))
    assert W["stress"]["source"] == "window" and W["stress"]["from"].year >= 2002
    # the 2002+ window's worst decade (2015-25) is mild: no stress at all, and the result says so
    assert W["prob_success"] >= W["prob_success_unstressed"]
    assert any(n.startswith("Warning: the stressed paths do better") for n in W["notes"])


def test_stress_history_value_is_checked():
    with pytest.raises(ValueError):
        mc.run(mc.Settings(weights={"SPY": 1}, sims=100, years=5, stress="worst_sequence", stress_history="bogus"))


# ------------------------------------------------------------ 4. GEM / GTAA month-end lookbacks by default

@needs("SPY", "VEU", "AGG")
def test_gem_defaults_to_calendar_months():
    p = port("Antonacci GEM")
    assert p.month_lookbacks == "calendar"
    gem = next(n for n in p.notes if n.startswith("Global Equities Momentum (Gary"))
    assert "calendar months" in gem and "month-end to month-end" in gem
    q = port("dual momentum GEM using trading days")
    assert q.month_lookbacks == "trading"
    assert "252 trading days" in next(n for n in q.notes if n.startswith("Global Equities Momentum (Gary"))


@needs("SPY", "EFA", "IEF", "VNQ", "DBC")
def test_gtaa_defaults_to_month_end():
    p = port("Faber GTAA")
    assert p.month_lookbacks == "calendar" and p.rebalance == "monthly"
    assert any("month-end prices" in n and "Faber" in n for n in p.notes)


@needs("SPY", "TLT")
def test_other_rotations_keep_trading_day_months():
    assert port("hold the top 1 of SPY and TLT by 12 month return, rebalance monthly").month_lookbacks == "trading"


# ------------------------------------------------------------ 5. correlations: which ticker moved the start

@needs("SPY", "TLT", "GLD", "EFASIM", "VNQSIM")
def test_correlation_names_the_late_ticker_and_labels_periods():
    from backtester import correlation
    R = correlation.analyze(["SPY", "TLT", "GLD", "EFASIM", "VNQSIM"], freq="monthly", start="2000-01-01")
    w = next(n for n in R["notes"] if n.startswith("Warning: the requested start"))
    assert "GLD" in w and "2004" in w
    ro = R["rolling"]
    i, j = R["tickers"].index(ro["pair"][0]), R["tickers"].index(ro["pair"][1])
    assert ro["matrix_period"] == pytest.approx(R["matrix"][i][j])
    assert str(ro["start"]) < str(R["start"])        # the pair's own window is longer
    txt = correlation.console(R)
    assert "over the pair's own history" in txt and "over the matrix period" in txt


# ------------------------------------------------------------ 6. crises before 1987

def test_crisis_list_has_the_long_history_events():
    names = [c[0] for c in metrics.CRISES]
    for k in ("1929", "1937-38", "1946-49", "1962", "1966", "1968-70", "1973-74", "1980", "1981-82"):
        assert any(k in n for n in names), k
    assert [pd.Timestamp(a) for _, a, _ in metrics.CRISES] == sorted(pd.Timestamp(a) for _, a, _ in metrics.CRISES)


def test_crisis_table_shows_old_events_only_with_data():
    idx = pd.bdate_range("1970-01-01", "1990-12-31")
    long = pd.Series(np.linspace(100, 300, len(idx)), index=idx)
    short = long[long.index >= "1985-01-01"]
    rows = metrics.crisis_table({"Long": long, "Short": short})
    ev = {r["event"]: r for r in rows}
    assert "1973-74 bear market" in ev and ev["1973-74 bear market"]["Short"] is None
    assert ev["1973-74 bear market"]["Long"] > 0
    assert "1929 crash and Great Depression" not in ev
    only_short = metrics.crisis_table({"Short": short})
    assert all(pd.Timestamp(r["start"]) >= pd.Timestamp("1985-01-01") for r in only_short)


# ------------------------------------------------------------ 7. trailing returns for every portfolio

def _chromium():
    return (glob.glob("/opt/pw-browsers/chromium*/chrome-linux*/chrome") or [None])[0]


@pytest.mark.skipif(_chromium() is None, reason="headless Chromium not available")
@needs("SPY", "TLT", "AGG")
def test_report_lists_trailing_returns_of_every_portfolio(tmp_path):
    from playwright.sync_api import sync_playwright
    A = []
    for text in ("hold 60% SPY and 40% TLT, rebalance yearly, since 2012",
                 "hold 30% SPY and 70% AGG, rebalance yearly, since 2012"):
        res = runner.run(port(text))
        A.append(report.analyze(res, sensitivity=False, mc=False))
    report.write_outputs(A, tmp_path, excel=False)
    errs: list[str] = []
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=_chromium())
        pg = b.new_page(viewport={"width": 1280, "height": 900})
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.goto((tmp_path / "report.html").as_uri())
        pg.wait_for_selector("#trailTbl tr")
        rows = pg.eval_on_selector_all("#trailTbl tbody tr", "rs => rs.map(r => r.cells[0].textContent)")
        assert rows[:2] == ["Strategy A ◀", "Strategy B"], rows       # both portfolios, the selected one marked
        assert "SPY" in rows and len(rows) == len(set(rows)), rows      # the benchmarks once each
        assert "all portfolios" in pg.inner_text("#trailCard")
        b.close()
    assert not errs, errs


# ------------------------------------------------------------ 8. Portfolio Visualizer's yearly rebalancing

@needs("VTI", "VXUS", "BND")
def test_named_lazy_portfolios_rebalance_yearly_by_default():
    p = port("three fund portfolio")
    assert p.rebalance == "yearly" and any("Portfolio Visualizer's default" in n for n in p.notes)
    assert port("three fund portfolio, rebalance monthly").rebalance == "monthly"


@needs("SPY", "TLT")
def test_pv_defaults_phrase():
    assert port("hold 60% SPY and 40% TLT").rebalance == "monthly"          # unchanged, with its note
    p = port("hold 60% SPY and 40% TLT with Portfolio Visualizer defaults")
    assert p.rebalance == "yearly" and p.month_lookbacks == "calendar"
    q = port("hold the top 1 of SPY and TLT by 12 month return with PV defaults")
    assert q.rebalance == "monthly" and q.month_lookbacks == "calendar"
