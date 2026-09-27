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
WHITELIST_BEFORE = {"HANS": "1993", "MNST": "1993", "CECO": "1990", "SIRI": "2004",   # sub-penny / 1/8-tick era quotes
                    # InterActive Group, the OTC shell Arrowhead reverse-merged into in January 2004 (1-for-65, booked
                    # 2004-01-15): 1/16-tick quotes x 650 after the splits, volumes of 0-10 (split-adjusted) shares
                    # a day, the price hopping between a few tick levels; still a handful of trades a day to March 2004
                    "ARWR": "2004-04"}
# the S&P 1500 additions of 2026-09 (volume multiples: the day's volume / the median of the 10 sessions before)
WHITELIST_SP1500 = {
    ("AN", "1995-05-22"): "Republic Industries (now AutoNation), Huizenga's investment: volume x64, the level held",
    ("ARWR", "2010-03-22"): "Arrowhead news day: volume x250 (2.6M shares against ~10,000), then fading",
    ("ARWR", "2016-11-30"): "ARC-520 program discontinued after the FDA clinical hold: -67%, volume x6",
    ("CAR", "2009-03-20"): "Avis Budget at $0.55 in the March 2009 low: +98%, volume x7.5",
    ("CAR", "2021-11-02"): "the Avis short squeeze (EV fleet news; +108%, a 545 intraday high), volume x19",
    ("CELH", "2008-10-20"): "Celsius as a thin OTC penny stock ($0.20 -> $0.47 in 1/15-cent ticks), volume x4",
    ("CHRD", "2020-03-09"): "Oasis Petroleum (now Chord) in the 2020-03-09 oil crash (GUSH -81%), volume x5",
    ("CHRD", "2020-03-13"): "Oasis at $0.37: March 2020 oil-crash rebound (SPY +8.5% that day)",
    ("CHRD", "2020-04-23"): "Oasis penny stock before its Chapter 11: volume x14",
    ("CHRD", "2020-06-05"): "Oasis in the June 2020 bankrupt-stock rally: volume x14",
    ("CHRD", "2020-06-08"): "same rally, volume x16",
    ("CHRD", "2020-11-11"): "Oasis old shares at $0.09-0.17 ahead of the reorganisation, volume x10",
    ("CHRD", "2020-11-20"): "not a return: Oasis's old shares were cancelled in its Chapter 11 and the new shares "
                            "listed on 2020-11-20 under the same symbol; the source splices the two securities",
    ("CLH", "2001-10-22"): "Clean Harbors news day: volume x500 (7.4M shares against ~15,000), the level mostly held",
    ("CNO", "2008-09-18"): "Conseco in the Lehman week: -42% then +81%, heavy volume both days",
    ("CORT", "2025-03-31"): "relacorilant ROSELLA ovarian-cancer trial success: volume x19",
    ("CYTK", "2014-04-25"): "tirasemtiv BENEFIT-ALS trial missed its primary endpoint: -65%, volume x13",
    ("CYTK", "2023-12-27"): "aficamten SEQUOIA-HCM trial success: volume x9",
    ("DAR", "1999-11-08"): "Darling as a thin penny stock ($1.25 -> $2.13 on 22,900 shares, 1/16 ticks)",
    ("DAR", "2000-09-21"): "thin penny stock in 1/16 ticks: a $0.25 print the day before, back to $0.69",
    ("EEFT", "1998-09-08"): "Euronet a year after its IPO at $2-3 in 1/16 ticks; SPY +5.4% that day",
    ("EEFT", "1998-11-25"): "Euronet at $2-4 in 1/16 ticks: volume x7, the level held",
    ("EHC", "2003-03-26"): "HealthSouth's first trade after the SEC fraud charges and trading halt: -97%, volume x120",
    ("EHC", "2003-07-07"): "HealthSouth on the pink sheets during its restructuring: +110%, volume x7",
    ("FLR", "2020-03-19"): "Fluor at $3.40 in the March 2020 crash: +76% rebound",
    ("GME", "2021-01-26"): "GameStop short squeeze", ("GME", "2021-01-27"): "GameStop short squeeze",
    ("GME", "2021-01-29"): "GameStop short squeeze", ("GME", "2021-02-24"): "GameStop second squeeze, volume x6",
    ("GME", "2024-05-13"): "Roaring Kitty's return, volume x8", ("GME", "2024-05-14"): "Roaring Kitty, volume x8",
    ("GPK", "2008-10-10"): "Graphic Packaging at $1.25 in the October 2008 crash: +65% rebound",
    ("HXL", "1993-12-08"): "Hexcel's Chapter 11 filing (December 1993): -60%, volume x80",
    ("IDCC", "1999-12-10"): "InterDigital in the 1999 wireless-stock mania: volume x6",
    ("IDCC", "1999-12-29"): "InterDigital, same mania: +112% to a 55 intraday high, volume x2 then higher",
    ("PCG", "2019-01-24"): "Cal Fire found PG&E not responsible for the 2017 Tubbs fire: +75%",
    ("PNW", "1989-12-07"): "Pinnacle West rebounding from its MeraBank-crisis low: volume x15, the level held",
    ("PWR", "2002-07-02"): "Quanta Services cut its earnings forecast: -68%, volume x11",
    ("WMB", "2002-07-22"): "Williams' 2002 liquidity crisis (dividend cut, credit downgrade): -61%, volume x10",
    ("WMB", "2002-07-29"): "Williams' rebound from its $0.78 low of 2002-07-25 (+101%)",
}

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
    ("VIP", "2021-03-22"): "the symbol changed hands: another listing's first day",
    ("VIP", "2023-01-12"): "thin, recycled symbol", ("VRTX", "2013-04-19"): "cystic fibrosis data",
    ("WEBS", "2020-03-16"): "3x inverse dot-com on 2020-03-16", ("YANG", "2022-03-16"): "China rally (YINN +65%)",
    ("YINN", "2022-03-16"): "China rally",
    **WHITELIST_SP1500,
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
