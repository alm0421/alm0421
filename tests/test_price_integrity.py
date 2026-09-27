"""Price-integrity gate (backtester/integrity.py): unrecorded / phantom splits and bad ticks are repaired on load,
real crashes are not, and every price file is free of unexplained one-day moves beyond 60%."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtester import data, integrity, runner, parser

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

# one-day total-return moves beyond +60% / -60% that are real (or are known junk series), with the evidence
WHITELIST_TICKERS = {
    "CPWR": "an OTC quote series of a delisted symbol (mostly zero volume, prices in cents): the quality filter "
            "never treats it as a member, and its jumps are single trades",
    "SSCC": "an OTC quote series after the company left Nasdaq (zero volume on most days, prices of $100s-$1000s)",
}
WHITELIST_BEFORE = {"HANS": "1993", "MNST": "1993", "CECO": "1990", "SIRI": "2004"}   # sub-penny / 1/8-tick era quotes
WHITELIST = {
    ("CONL", "2024-11-06"): "2x COIN on election day (COIN +31%)",
    ("CWEB", "2022-03-16"): "2x KWEB; China internet +40% on the State Council support pledge",
    ("DRIP", "2020-03-09"): "2x inverse XOP on the 2020-03-09 oil crash (GUSH -81% the same day)",
    ("GUSH", "2020-03-09"): "2x XOP on the 2020-03-09 oil crash",
    ("ERX", "2020-03-09"): "then 3x XLE on the oil crash (ERY +60% the same day)",
    ("ERY", "2020-03-09"): "then 3x inverse XLE on the oil crash",
    ("ETHE", "2019-06-20"): "first week of OTC quotation (a stale placeholder price before)",
    ("ETHE", "2019-06-21"): "first week of OTC quotation", ("ETHE", "2019-06-24"): "first week of OTC quotation",
    ("ETHE", "2019-06-25"): "first week of OTC quotation",
    ("FOSL", "2018-02-14"): "earnings, volume x6", ("FOSL", "2021-01-27"): "meme squeeze, volume x14",
    ("FOSL", "2024-12-02"): "turnaround news, volume x100",
    ("IDXX", "1997-03-24"): "earnings warning, volume x139", ("INSM", "2002-09-10"): "trial failure, volume x144",
    ("INSM", "2017-09-05"): "phase 3 success", ("INSM", "2024-05-28"): "ASPEN trial success",
    ("JDST", "2020-03-12"): "3x inverse junior gold miners, March 2020 (JNUG moves the other way)",
    ("JDST", "2020-03-17"): "March 2020", ("JNUG", "2020-03-12"): "March 2020", ("JNUG", "2020-03-13"): "March 2020",
    ("JNUG", "2020-03-16"): "March 2020 (JDST -59% the same day)", ("JNUG", "2020-03-18"): "March 2020",
    ("LILAK", "2015-07-01"): "first days of the LiLAC tracking stock (when-issued)",
    ("MRNA", "2026-08-19"): "news day: volume x46, a 54% intraday range, not a split (volume would fall)",
    ("MS", "2008-10-13"): "Mitsubishi UFJ investment closed", ("MSTR", "2000-03-20"): "accounting restatement",
    ("MSTR", "2001-04-19"): "dot-com rebound", ("REGN", "2000-02-23"): "genomics rally",
    ("SIRI", "2002-08-15"): "heavy trading both days (Sirius's 2002 recapitalisation)",
    ("SVXY", "2018-02-06"): "volmageddon (-83%; volume did not follow a split)",
    ("TSCO", "1994-02-18"): "IPO day: the first row is a pre-listing placeholder",
    ("UAL", "2008-07-22"): "earnings / oil drop (+68%)", ("UAUA", "2008-07-22"): "same company as UAL",
    ("UVIX", "2024-08-05"): "VIX spike of 2024-08-05", ("UVXY", "2018-02-05"): "volmageddon",
    ("VIP", "2021-03-22"): "the symbol changed hands (see CA_WHITELIST): another listing's first day",
    ("VIP", "2023-01-12"): "thin, recycled symbol", ("VRTX", "2013-04-19"): "cystic fibrosis data",
    ("WEBS", "2020-03-16"): "3x inverse dot-com on 2020-03-16", ("YANG", "2022-03-16"): "China rally (YINN +65%)",
    ("YINN", "2022-03-16"): "China rally",
}


@needs
def test_no_unexplained_huge_one_day_moves_in_any_price_file():
    bad = []
    for t in data.available_tickers():
        if t.startswith("^") or t in WHITELIST_TICKERS:
            continue
        try:
            df = data.load(t)
        except Exception:  # noqa: BLE001
            continue
        tr = (df["close"] + df["dividend"]) / df["close"].shift(1)
        for d in tr.index[((tr > 1.6) | (tr < 0.4)).fillna(False).to_numpy()]:
            if (t, str(d.date())) in WHITELIST or (t in WHITELIST_BEFORE and str(d.date()) < WHITELIST_BEFORE[t]):
                continue
            bad.append((t, str(d.date()), round(float(tr[d]), 3)))
    assert not bad, bad


@needs
def test_whitelisted_leveraged_pairs_agree():
    # evidence for the whitelisted fund days: the bull and bear funds moved opposite ways, as their index did
    for bull, bear, day in (("GUSH", "DRIP", "2020-03-09"), ("ERX", "ERY", "2020-03-09"), ("YINN", "YANG", "2022-03-16"),
                            ("JNUG", "JDST", "2020-03-16")):
        if all((data.PRICES / f"{t}.csv").exists() for t in (bull, bear)):
            assert (_close_ratio(bull, day) - 1) * (_close_ratio(bear, day) - 1) < 0, (bull, bear)
