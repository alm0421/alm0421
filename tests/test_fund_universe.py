"""The broad ETF / mutual fund universe, fund-exact SIMs, new long-history builders and the Portfolio
Visualizer asset-class vocabulary."""
import importlib.util
import io
import pathlib
import zipfile

import numpy as np
import pandas as pd
import pytest

from backtester import data, fund_lists, parser

ROOT = pathlib.Path(__file__).parents[1]


@pytest.fixture(scope="module")
def fd():
    spec = importlib.util.spec_from_file_location("fetch_data_universe", ROOT / "scripts" / "fetch_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ the universe

def test_fund_lists_are_funds_not_index_members():
    assert len(fund_lists.BROAD_ETFS) >= 400 and len(fund_lists.MUTUAL_FUNDS) >= 150
    for t in ("AVUV", "AVDV", "VCLT", "IGLB", "MUB", "VTEB", "EWJ", "MTUM", "IJS", "VOOG", "USDU", "EUO", "YCS",
              "TBX", "BITI", "ETHU", "GDXU", "RXL", "NRGU", "OILU"):
        assert t in fund_lists.BROAD_ETFS, t
    for t in ("VTSAX", "VWELX", "FXAIX", "DFSVX", "PRPFX", "AGTHX", "VWITX", "FNMIX", "PCRIX"):
        assert t in fund_lists.MUTUAL_FUNDS, t
    assert "SQM" in fund_lists.BROAD_STOCKS and "SQM" not in fund_lists.ALL_FUNDS
    raw = pd.read_csv(ROOT / "data" / "ndx_membership.csv", dtype=str)
    members = {t for row in raw["tickers"] for t in str(row).split()}
    assert not (members & fund_lists.ALL_FUNDS)
    mem = data.membership()
    if mem is not None:
        assert not (set(mem.columns) & fund_lists.ALL_FUNDS)


def test_broad_batch_missing_first_then_stalest(fd):
    broad = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    info = {"AAA": {"last": "2026-09-20"}, "BBB": {"last": "2026-09-10"}, "DDD": {"last": "2026-09-25"}}
    disk = {"AAA", "BBB", "DDD", "EEE"}          # EEE's file exists but last run didn't record it
    failed = {"CCC": "2026-09-20"}               # failed six days ago: waits
    got = fd.broad_batch(broad, info, failed, "2026-09-26", budget=3, on_disk=lambda t: t in disk)
    assert got == ["EEE", "BBB", "AAA"]
    # a failure older than the retry window is tried again (as a missing file, first)
    got = fd.broad_batch(broad, info, {"CCC": "2026-07-01"}, "2026-09-26", budget=2, on_disk=lambda t: t in disk)
    assert got == ["CCC", "EEE"]
    assert fd.broad_batch(broad, info, {}, "2026-09-26", budget=0, on_disk=lambda t: True) == []


def test_broad_lists_exclude_the_core_lists(fd):
    core = set(fd.EXTRA)
    assert not (set(fd.BROAD) & core)
    assert set(fd.SIM_FUNDS) <= set(fd.ETFS)      # SIM twins refresh every run
    assert "SQM" in fd.STOCKS


def test_request_ticker_appends_to_the_queue(tmp_path, monkeypatch):
    f = tmp_path / "requested_tickers.txt"
    f.write_text("# comment\nABCD")
    monkeypatch.setenv("BACKTESTER_QUEUE_FILE", str(f))
    monkeypatch.setattr(data, "EXTRA_TICKERS_FILE", tmp_path / "extra_tickers.txt")
    (tmp_path / "extra_tickers.txt").write_text("EFGH\n")
    assert data.request_ticker("wxyz") == "added"
    assert data.request_ticker("WXYZ") == "already requested"
    assert data.request_ticker("ABCD") == "already requested"
    assert data.request_ticker("EFGH") == "already requested"          # refreshed every run already
    assert data.request_ticker("VWELX") == "added"                     # a listed fund jumps the rotation
    assert f.read_text().splitlines() == ["# comment", "ABCD", "WXYZ  # requested from the site / CLI",
                                          "VWELX  # requested from the site / CLI"]
    with pytest.raises(data.DataError):
        data.request_ticker("not a ticker!")


def test_site_add_ticker_queues_when_offline(tmp_path, monkeypatch):
    from backtester import web
    f = tmp_path / "requested_tickers.txt"
    monkeypatch.setenv("BACKTESTER_QUEUE_FILE", str(f))
    monkeypatch.setattr(data, "fetch_on_demand", lambda t: False)
    out = web.api_fetch({"ticker": "QZQZ"})
    assert out["queued"] == "added" and "push" in out["status"] and "not in the US listing" in out["status"]
    assert "QZQZ" in f.read_text()
    out = web.api_fetch({"ticker": "LLPFX"} if not (data.PRICES / "LLPFX.csv").exists() else {"ticker": "QZQZ"})
    assert "requested_tickers.txt" in out["status"]


# ------------------------------------------------------------------ SIM building blocks

def test_port25_grid_named_and_positional(fd):
    names = ["SMALL LoBM", "ME1 BM2", "ME1 BM3", "ME1 BM4", "SMALL HiBM"]
    for q in (2, 3, 4):
        names += [f"ME{q} BM1"] + [f"ME{q} BM{b}" for b in (2, 3, 4)] + [f"ME{q} BM5"]
    names += ["BIG LoBM", "ME5 BM2", "ME5 BM3", "ME5 BM4", "BIG HiBM"]
    df = pd.DataFrame({n: [float(i)] for i, n in enumerate(names)})
    g = fd.port25_grid(df)
    assert g[(1, 1)].iloc[0] == 0 and g[(1, 5)].iloc[0] == 4 and g[(3, 2)].iloc[0] == 11 and g[(5, 5)].iloc[0] == 24
    pos = fd.port25_grid(pd.DataFrame({f"c{i}": [float(i)] for i in range(25)}))
    assert pos[(2, 1)].iloc[0] == 5 and pos[(5, 5)].iloc[0] == 24
    with pytest.raises(RuntimeError):
        fd.port25_grid(pd.DataFrame({f"c{i}": [0.0] for i in range(6)}))


def _write_fund(path, idx, rets):
    lvl = 100 * np.cumprod(1 + np.asarray(rets))
    pd.DataFrame({"date": idx, "open": lvl, "high": lvl, "low": lvl, "close": lvl, "adj_close": lvl, "volume": 0,
                  "dividend": 0.0, "split": 0.0}).to_csv(path, index=False)


def test_splice_from_date_and_pick_model(fd, tmp_path, monkeypatch):
    monkeypatch.setattr(fd, "PRICES", tmp_path)
    idx = pd.bdate_range("2000-01-03", "2006-12-29")
    rng = np.random.default_rng(0)
    fund = rng.normal(0.0003, 0.01, len(idx))
    _write_fund(tmp_path / "FUNDX.csv", idx, fund)
    sim = pd.Series(0.0, index=pd.bdate_range("1995-01-02", "2006-12-29"))
    # "@date": the fund only from that date
    out = fd._splice_returns(sim, "FUNDX@2003-01-02")
    assert (out[out.index < "2003-01-02"] == 0).all() and out[out.index >= "2003-01-03"].abs().sum() > 0
    good = pd.Series(fund, index=idx) + rng.normal(0, 0.001, len(idx))
    bad = pd.Series(fund, index=idx) * 2
    label, model = fd._pick_model("TESTSIM", {"bad": bad, "good": good, "none": None}, ("MISSING", "FUNDX"))
    assert label == "good" and model is not None
    assert any("TESTSIM model choice vs FUNDX" in n for n in fd.SIM_NOTES)


def test_moody_yield_covers_every_session(fd):
    y = fd.moody_yield()
    assert y.index[0].year == 1953 and y.notna().all()
    gaps = pd.Series(y.index).diff().dt.days.max()
    assert gaps <= 7, gaps          # no hole where the monthly and daily series meet (9/11 closed NYSE for 4 days)
    r = fd._bond_returns(y, 20)
    assert r.abs().max() < 0.2


def test_french_country_index_reads_the_zip_members(fd, monkeypatch):
    body = ("This file was created using the 202512 Bloomberg database.\n\n"
            " Japan -- Value Weighted Returns\n        Mkt    HiBM    LoBM\n"
            "197501   1.00    2.00   -1.00\n197502  -0.50    0.10    0.20\n\n Annual\n")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for n in ("UK.Dat", "Japan.Dat", "Austrlia.Dat"):
            z.writestr(n, body)

    class R:
        content = buf.getvalue()
    monkeypatch.setattr(fd.requests, "get", lambda *a, **k: R())
    fd._ZIPS.clear()
    s = fd.french_country_index("Japan.Dat")
    assert list(s.round(4)) == [0.01, -0.005] and s.index[0] == pd.Timestamp("1975-01-31")
    with pytest.raises(RuntimeError):
        fd.french_country_index("Brazil.Dat")
    fd._ZIPS.clear()
    assert {m for _, m, _ in fd.COUNTRY_SIMS} <= {"Japan.Dat", "UK.Dat", "Germany.Dat", "Canada.Dat", "Austrlia.Dat",
                                                  "France.Dat", "Swtzrlnd.Dat", "HongKong.Dat"}


def test_new_sims_are_described_and_mapped():
    for t in ("VTISIM", "VXUSSIM", "VWOSIM", "VOESIM", "VOTSIM", "VCLTSIM", "MUBSIM", "EMBSIM", "EWJSIM", "EWUSIM"):
        assert t in data.SIMS, t
    # fund-exact first, the older series as fallback while the data job hasn't built the new one
    assert [s for s, _ in parser.SIM_FOR["VTI"]] == ["VTISIM", "SPYSIM"]
    assert [s for s, _ in parser.SIM_FOR["VXUS"]] == ["VXUSSIM", "EFASIM"]
    assert parser.SIM_FOR["VCLT"][0][0] == "VCLTSIM" and parser.SIM_FOR["MUB"][0][0] == "MUBSIM"


def test_three_fund_uses_fund_exact_series_when_present(monkeypatch):
    real = parser._known

    def known():
        return real() | {"VTISIM", "VXUSSIM"}
    monkeypatch.setattr(parser, "_known", known)
    notes: list[str] = []
    parser._TL.start = "1985-01-01"
    try:
        node = parser._model_portfolio("3 fund portfolio", notes)
    finally:
        parser._TL.start = None
    assert [k["asset"] for k in node["children"]] == ["VTISIM", "VXUSSIM", "BNDSIM"]
    assert any("VTSMX" in n and "VGTSX" in n for n in notes)


# ------------------------------------------------------------------ asset-class vocabulary

def _port(text):
    return parser.parse(text)


def test_pv_asset_class_names_after_a_weight():
    p = _port("40% US stock market, 20% international stocks, 40% total bond since 1972")
    kids = [k["asset"] for k in p.tree["children"]]
    have = set(data.available_tickers())
    assert kids[0] == ("VTISIM" if "VTISIM" in have else "SPYSIM")
    assert kids[1] == ("VXUSSIM" if "VXUSSIM" in have else "EFASIM")
    assert kids[2] == "BNDSIM"
    assert any("'US stock market'" in n for n in p.notes)


@pytest.mark.parametrize("phrase,want", [
    ("US large cap value", "VTVSIM"), ("US large cap growth", "VUGSIM"), ("US large cap", "SPYSIM"),
    ("US mid cap", "MIDSIM"), ("US small cap value", "VBRSIM"), ("small cap growth", "VBKSIM"),
    ("US small cap", "VBSIM"), ("international developed", "EFASIM"), ("REITs", "VNQSIM"), ("gold", "GLDSIM"),
    ("commodities", "DBCSIM"), ("short term treasury", "SHYSIM"), ("intermediate term treasuries", "IEFSIM"),
    ("long-term treasuries", "TLTSIM"), ("TIPS", "TIPSIM"), ("corporate bonds", "LQDSIM"), ("high yield", "HYGSIM"),
    ("international bonds", "BNDXSIM"), ("T-bills", "BILSIM"), ("international small cap value", "AVDVSIM"),
])
def test_asset_class_table(phrase, want):
    got = parser._asset_class_ticker(phrase)
    assert got and got[0] == want, (phrase, got)


def test_asset_class_names_only_after_a_weight():
    # a signal rule's words are untouched; a weight-first list of SIMs is an allocation
    assert parser._asset_class_names("buy GLD when gold is up 3 days in a row") == "buy GLD when gold is up 3 days in a row"
    assert parser._asset_class_names("hold 60% US stock market and 40% bonds") == "hold 60% " + (
        "VTISIM" if "VTISIM" in data.available_tickers() else "SPYSIM") + " and 40% BNDSIM"
    assert parser.looks_like_allocation("40% SPYSIM, 20% EFASIM, 40% BNDSIM since 1972")
    p = _port("40% SPYSIM, 20% EFASIM, 40% BNDSIM since 1972")
    assert [k["asset"] for k in p.tree["children"]] == ["SPYSIM", "EFASIM", "BNDSIM"]


def test_fund_phrases_next_to_real_tickers_keep_their_fund():
    # mixed with real tickers, a phrase that already named a fund stays that fund; alone, the long series
    # (round 12: the substitution is the fund itself now, settled against the start by the holdings parser)
    assert parser._asset_class_names("hold 60% SPY and 40% gold") == "hold 60% SPY and 40% GLD"
    assert parser._asset_class_names("hold 60% US stock market and 40% gold").endswith("40% GLDSIM")
    assert parser._asset_class_names("60% SPY, 40% municipal bonds") != "60% SPY, 40% municipal bonds" or \
        parser._asset_class_ticker("municipal bonds") is None
