"""Nasdaq-100 membership snapshots: ~100 companies every month, and changes explained by the dated change table."""
from __future__ import annotations

import importlib.util
import pathlib

import pandas as pd
import pytest

from backtester import data

HAVE = data.MEMBERSHIP_FILE.exists() and data.CHANGES_FILE.exists()
needs = pytest.mark.skipif(not HAVE, reason="membership data not downloaded")

# months the sources cannot fill: SPOT (PanAmSat) left in late 2004 and its replacement only appears in the
# 2005-04 snapshot (no dated change table before 2007)
SHORT_MONTHS = {"2004-12", "2005-01", "2005-02", "2005-03"}

# reviewed snapshot-to-snapshot changes that the dated change table does not explain: symbol changes it records
# under another spelling (SUNW->JAVA, ERTS->EA, HANS->MNST, NWSA->FOXA, FB, WLTW, FISV->FI), extra share classes
# (GOOGL, CMCSK, LMCK, DISCK, LILA/LILAK, BATRA/BATRK, TFCF/TFCFA, QVCA/LVNTA/QRTEA tracking stocks) and a few
# Wikipedia snapshots that lagged the index by a month or more
KNOWN_UNEXPLAINED = {
    ("2007-09", "+JAVA"), ("2007-09", "-NTLI"), ("2007-09", "-SUNW"), ("2007-10", "+NELV"), ("2007-11", "+HSIC"),
    ("2007-11", "-MXIM"), ("2007-11", "-NELV"), ("2009-10", "+CERN"), ("2009-10", "-JAVA"), ("2011-01", "+CTRP"),
    ("2011-01", "+WFMI"), ("2012-01", "+EA"), ("2012-01", "+HANS"), ("2012-01", "-ERTS"), ("2012-08", "-CTRP"),
    ("2013-01", "+FB"), ("2013-08", "+FOXA"), ("2013-08", "-NWSA"), ("2014-05", "+GOOGL"), ("2014-09", "+LMCK"),
    ("2014-12", "+DISCK"), ("2014-12", "+LVNTA"), ("2014-12", "+QVCA"), ("2014-12", "-LINTA"), ("2015-01", "+CMCSK"),
    ("2016-01", "+CTRP"), ("2016-01", "-BRCM"), ("2016-01", "-CMCSK"), ("2016-12", "+MCHP"), ("2016-12", "+XRAY"),
    ("2016-12", "-BATRA"), ("2016-12", "-BATRK"), ("2016-12", "-ENDP"), ("2016-12", "-LMCA"), ("2016-12", "-LMCK"),
    ("2017-05", "+JBHT"), ("2017-05", "-NXPI"), ("2017-07", "+LILA"), ("2017-07", "+LILAK"), ("2018-02", "-LILA"),
    ("2018-02", "-LILAK"), ("2018-05", "+QRTEA"), ("2018-05", "-LVNTA"), ("2018-05", "-QVCA"), ("2019-01", "+WLTW"),
    ("2019-04", "+TFCF"), ("2019-04", "+TFCFA"), ("2019-05", "-TFCF"), ("2019-05", "-TFCFA"), ("2019-12", "-SYMC"),
    ("2020-06", "-WLTW"), ("2023-07", "-FISV"),
}


@needs
def test_every_month_has_about_100_companies():
    mem = data.membership()
    bad = []
    for d, row in mem.iterrows():
        names = set(row.index[row.to_numpy()])
        n, k = data.company_count(names), len(names)
        if f"{d:%Y-%m}" in SHORT_MONTHS:
            assert 97 <= n < 100
            continue
        if not (100 <= n <= 103 and k <= 112):          # 100 companies; up to ~9 extra share classes
            bad.append((f"{d:%Y-%m}", n, k))
    assert not bad, bad


@needs
def test_2007_members_that_the_old_parser_dropped_are_back():
    mem = data.membership()
    for t in ("AAPL", "AKAM", "FLEX"):
        assert mem.loc["2007-01-01":"2008-02-01", t].all(), t
    if (data.PRICES / "AAPL.csv").exists():
        m, _ = data.member_mask(["AAPL", "AKAM"], data.load("AAPL").loc["2007-03-01":"2007-12-31"].index)
        assert m.all()


@needs
def test_membership_changes_are_explained_or_flagged():
    new = set(data.membership_unexplained()) - KNOWN_UNEXPLAINED
    assert not new, sorted(new)


def test_wiki_parser_reads_piped_link_tickers():
    spec = importlib.util.spec_from_file_location("fetch_data", pathlib.Path(__file__).parents[1] / "scripts" / "fetch_data.py")
    fd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fd)
    # the format of the 2007 revisions (e.g. revid 104675414)
    text = ("==Components==\n*[[Akamai|Akamai Technologies, Inc.]]  ([[Akamai|AKAM]])\n*[[Apple Inc.]] ([[Apple Inc.|AAPL]])\n"
            "*[[Flextronics|Flextronics International Ltd.]] ([[Flextronics|FLEX]])\n*[[NVIDIA|NVIDIA Corporation]] (NVDA)\n"
            "*[[Oracle Corporation]] ([[ORCL]])\n==External links==\n")
    assert fd.wiki_tickers(text) == {"AKAM", "AAPL", "FLEX", "NVDA", "ORCL"}
    assert fd.MEMBERSHIP_PARSER == "4"


def test_repair_fills_a_gap_only_without_a_dated_change():
    idx = pd.date_range("2007-01-01", periods=5, freq="MS")
    base = {f"T{i}": [True] * 5 for i in range(99)}
    df = pd.DataFrame({**base, "AAPL": [True, False, False, True, True], "GONE": [True, False, False, True, True]},
                      index=idx)
    ch = pd.DataFrame({"date": [pd.Timestamp("2007-02-10")], "added": [""], "removed": ["GONE"]})
    out, log = data.repair_membership(df, ch)
    assert out["AAPL"].all() and not out.loc["2007-02-01", "GONE"]
    assert any("AAPL" in x for x in log)
