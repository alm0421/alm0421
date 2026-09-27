"""Price-integrity gate (backtester/integrity.py): unrecorded / phantom splits and bad ticks are repaired on load,
real crashes are not, and every price file is free of unexplained one-day moves beyond 60%."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtester import data, integrity, parser, price_flags, runner

HAVE = all((data.PRICES / f"{t}.csv").exists() for t in ("SOXS", "SOXL", "PGOVX", "NVDS", "CPER", "SVXY", "SPY"))
needs = pytest.mark.skipif(not HAVE, reason="price data not downloaded")


def _close_ratio(t, day):
    c = data.load(t)["close"]
    i = c.index.get_loc(pd.Timestamp(day))
    return float(c.iloc[i] / c.iloc[i - 1])


@needs
def test_reported_phantom_returns_are_repaired():
    # SOXS 2026-05-26: 1159.5 -> 62.9 in the file (the earlier prices 15x too high); now the -3x SOXX move
    assert 0.75 < _close_ratio("SOXS", "2026-05-26") < 0.9
    # PIMCO reverse splits missing from the data (PGOVX 1-for-4, PCRAX / PSLDX 1-for-3)
    assert abs(_close_ratio("PGOVX", "2023-03-27") - 1) < 0.03
    assert abs(_close_ratio("PCRAX", "2023-03-27") - 1) < 0.03
    sp = data.splits("PGOVX")
    assert abs(float(sp.loc["2023-03-27"]) - 0.25) < 1e-9
    # quoted (as-traded) prices are what traded: 3.97 before the reverse split, 15.57 after
    q = data.quoted_close("PGOVX")
    assert abs(q.loc["2023-03-24"] - 3.97) < 1e-6 and abs(q.loc["2023-03-27"] - 15.57) < 1e-6
    # PGOVX's 2015-12-16 payout was already on the post-split basis: the day's total return is now ordinary
    df = data.load("PGOVX")
    i = df.index.get_loc(pd.Timestamp("2015-12-16"))
    tr = (df["close"].iloc[i] + df["dividend"].iloc[i]) / df["close"].iloc[i - 1]
    assert abs(tr - 1) < 0.05
    # NVDS 2023-08-09: a 1-for-5 booked twice; the phantom one is dropped, the 2023-08-15 one kept
    assert abs(_close_ratio("NVDS", "2023-08-09") - 1.0586) < 0.01
    assert pd.Timestamp("2023-08-09") not in data.splits("NVDS").index
    assert pd.Timestamp("2023-08-15") in data.splits("NVDS").index
    # CPER bad tick
    assert abs(data.load("CPER")["close"].loc["2014-12-04"] - 19.54) < 0.05
    assert not data.load("CPER")["open_ok"].loc["2014-12-04"]


@needs
def test_real_crashes_are_not_repaired():
    assert _close_ratio("SVXY", "2018-02-06") < 0.2                  # volmageddon
    if (data.PRICES / "UVXY.csv").exists():
        assert _close_ratio("UVXY", "2018-02-05") > 1.6
    if (data.PRICES / "SVIX.csv").exists():
        assert _close_ratio("SVIX", "2024-08-05") < 0.65
    if (data.PRICES / "AAPL.csv").exists():
        assert _close_ratio("AAPL", "2000-09-29") < 0.55            # a real -52% day, not a 2-for-1 split
    for t in ("SVXY", "UVXY", "AAPL", "SMCI", "TTD"):
        if (data.PRICES / f"{t}.csv").exists():
            assert not (data.price_repairs(t)["kind"] != "bad_tick").any(), t


@needs
def test_inferred_splits_match_known_history():
    # McDonald's own split history: 2-for-1 distributed 1968-05-20 and 1969-06-13, missing from the data
    if not (data.PRICES / "MCD.csv").exists():
        pytest.skip("no MCD")
    ev = data.price_repairs("MCD")
    got = {str(pd.Timestamp(d).date()): r for d, r in zip(ev["date"], ev["ratio"])}
    assert got.get("1968-05-21") == 2.0 and got.get("1969-06-13") == 2.0


@needs
def test_backtests_say_when_they_cross_a_repaired_day():
    spec = parser.parse("hold 50% SOXL and 50% SOXS, rebalance daily, since 2026")
    res = runner.run(spec)
    assert any(n.startswith("Data repaired: SOXS 2026-05-26") for n in spec.notes)
    eq = res.equity
    assert (eq.pct_change().abs().dropna() < 0.25).all()
    spec = parser.parse("hold 100% PGOVX, since 2022, until 2023-12")
    res = runner.run(spec)
    assert any("PGOVX 2023-03-27" in n for n in spec.notes)
    assert res.equity.iloc[-1] < res.equity.iloc[0] * 1.2
    spec = parser.parse("buy NVDS at the close when RSI(2) is below 90, hold 3 days, since 2023-06")
    res = runner.run(spec)
    tr = res.trades
    ratio = (tr["exit_price"] * tr["exit_shares"] + tr["income"]) / (tr["entry_price"] * tr["shares"]) - 1
    assert (abs(ratio - tr["return"]) < 0.02).all()          # the as-traded prices and the return agree


def _synthetic(n=300, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    c = 50 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "adj_close": c,
                         "volume": 1e6 + rng.uniform(0, 1e5, n), "dividend": 0.0, "split": 0.0}, index=idx)


def test_synthetic_unrecorded_reverse_split_and_bad_tick():
    raw = _synthetic()
    k = 200
    for col in ("open", "high", "low", "close", "adj_close"):
        raw.iloc[k:, raw.columns.get_loc(col)] *= 10          # a 1-for-10 reverse split the data never booked
    raw.iloc[k:, raw.columns.get_loc("volume")] /= 10
    raw.iloc[100, raw.columns.get_loc("close")] *= 0.6        # an isolated bad tick
    out, ev = integrity.check("ZZZ", raw, lambda t: None, fund=True)
    kinds = dict(zip(ev["kind"], ev["date"]))
    assert kinds.get("inferred_split") == raw.index[k] and kinds.get("bad_tick") == raw.index[100]
    r = np.log(out["close"]).diff().abs()
    assert r.max() < 0.06
    assert abs(out["split"].iloc[k] - 0.1) < 1e-12


def test_synthetic_crash_with_volume_burst_is_left_alone():
    raw = _synthetic()
    k = 200
    for col in ("open", "high", "low", "close", "adj_close"):
        raw.iloc[k:, raw.columns.get_loc(col)] *= 0.5          # -50% on news ...
    raw.iloc[k, raw.columns.get_loc("volume")] *= 15           # ... on heavy volume that then fades
    raw.iloc[k + 1:k + 6, raw.columns.get_loc("volume")] *= 3
    out, ev = integrity.check("ZZZ", raw, lambda t: None, fund=False)
    assert ev.empty


# ---------------------------------------------------------------- every price file

# The curated explanations (the whitelists of real moves, with their evidence) live in backtester/known_moves.py; the
# price-flag scan (backtester/price_flags.py) counts them as explained. What nothing explains is flagged instead.
from backtester.known_moves import WHITELIST, WHITELIST_BEFORE, WHITELIST_TICKERS  # noqa: E402,F401


@needs
def test_every_huge_one_day_move_is_explained_or_flagged():
    # the property that scales with the data job (thousands of new price files): every one-day total return beyond
    # +60% / -60% in any price file is repaired on load (then it is no longer one), explained (a security break or a
    # curated entry with its evidence), or recorded in data/price_flags.json - where every backtest holding across it
    # is warned (tests/test_price_flags.py). In the core universe (Nasdaq-100 members past and present, core ETFs,
    # SPY / QQQ) nothing may be left unexplained.
    missing, core_bad = price_flags.unrecorded("move")
    assert not missing, ("unexplained one-day moves not in data/price_flags.json (regenerate it: python -m "
                         f"backtester.price_flags; or explain them in backtester/known_moves.py): {missing}")
    assert not core_bad, f"unexplained one-day moves in the core universe (explain or repair them): {core_bad}"


@needs
def test_whitelist_entries_are_real_days_with_reasons():
    for (t, day), why in WHITELIST.items():
        assert len(why) > 8, (t, day)
    for t, why in WHITELIST_TICKERS.items():
        assert len(why) > 20, t


@needs
def test_whitelisted_leveraged_pairs_agree():
    # evidence for the whitelisted fund days: the bull and bear funds moved opposite ways, as their index did
    for bull, bear, day in (("GUSH", "DRIP", "2020-03-09"), ("ERX", "ERY", "2020-03-09"), ("YINN", "YANG", "2022-03-16"),
                            ("JNUG", "JDST", "2020-03-16")):
        if all((data.PRICES / f"{t}.csv").exists() for t in (bull, bear)):
            assert (_close_ratio(bull, day) - 1) * (_close_ratio(bear, day) - 1) < 0, (bull, bear)
