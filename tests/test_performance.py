"""Speed work that must not change a number: the processed-price cache (data.load results kept on disk, invalidated
when a price file, the reference data or the code changes), the cost-sensitivity reruns that reuse the prepared
signals, the site's download files written after the answer, and a loose timing guard."""
import dataclasses
import shutil
import threading
import time

import pandas as pd
import pytest

from backtester import data, engine, parser, report, runner, web

AVAIL = set(data.available_tickers())
needs = lambda *t: pytest.mark.skipif(not set(t) <= AVAIL, reason="price data not available")  # noqa: E731


@pytest.fixture
def cache_on(tmp_path, monkeypatch):
    """The cache switched on, in a folder of its own; every in-memory cache forgotten before and after."""
    monkeypatch.setenv("BACKTESTER_CACHE", "1")
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path / "cache")
    data.clear_caches()
    data.PRICE_REPAIRS.clear()
    data.splits.cache_clear()
    data.corporate_action_fixes.cache_clear()
    yield tmp_path / "cache"
    data.clear_caches()
    data.PRICE_REPAIRS.clear()
    data.splits.cache_clear()
    data.corporate_action_fixes.cache_clear()


def _forget():
    data.clear_caches()
    data.PRICE_REPAIRS.clear()
    data.splits.cache_clear()
    data.corporate_action_fixes.cache_clear()


def _fresh(t):
    _forget()
    return data.load(t)


@needs("MSFT", "VTISIM", "GOOG", "GOOGL", "SPY", "EBAY")
def test_cached_load_is_identical(cache_on, monkeypatch):
    for t in ("MSFT", "VTISIM", "GOOG", "EBAY"):
        monkeypatch.setenv("BACKTESTER_CACHE", "0")
        _forget()   # computed on their own first (not from what data.load leaves behind)
        want_splits, want_rep, want_ca = data.splits(t), data.price_repairs(t), data.corporate_action_fixes(t)
        want = _fresh(t)
        monkeypatch.setenv("BACKTESTER_CACHE", "1")
        cold = _fresh(t)                        # computed and written
        assert list(cache_on.rglob(f"{t}-*.pkl"))
        warm = _fresh(t)                        # read back
        for got in (cold, warm):
            pd.testing.assert_frame_equal(got, want, check_exact=True)
        pd.testing.assert_series_equal(data.splits(t), want_splits, check_exact=True)
        pd.testing.assert_frame_equal(data.price_repairs(t), want_rep, check_exact=True)
        pd.testing.assert_frame_equal(data.corporate_action_fixes(t), want_ca, check_exact=True)


@needs("SPY")
def test_cache_is_invalidated_when_price_files_change(cache_on, tmp_path, monkeypatch):
    prices = tmp_path / "prices"
    prices.mkdir()
    shutil.copy(data.PRICES / "SPY.csv", prices / "AAA.csv")
    shutil.copy(data.PRICES / "SPY.csv", prices / "BBB.csv")
    monkeypatch.setattr(data, "PRICES", prices)
    monkeypatch.setattr(data, "CUSTOM", tmp_path / "custom")
    first = _fresh("AAA")
    dirs = [p for p in cache_on.iterdir() if p.is_dir()]
    assert len(dirs) == 1 and len(list(dirs[0].glob("AAA-*.pkl"))) == 1

    # served from the cache: the processing is not run again
    real = data._load_file
    monkeypatch.setattr(data, "_load_file", lambda *a: pytest.fail("processed again although cached"))
    pd.testing.assert_frame_equal(_fresh("AAA"), first, check_exact=True)
    monkeypatch.setattr(data, "_load_file", real)

    # the file itself changes (as the data job rewrites data/prices): the new prices, not the cached ones
    raw = pd.read_csv(prices / "AAA.csv")
    raw.loc[raw.index[-1], "close"] = raw["close"].iloc[-1] * 1.5
    raw.to_csv(prices / "AAA.csv", index=False)
    second = _fresh("AAA")
    assert second["close"].iloc[-1] == pytest.approx(first["close"].iloc[-1] * 1.5)
    new_dirs = [p for p in cache_on.iterdir() if p.is_dir()]
    assert len(new_dirs) == 1 and new_dirs[0] != dirs[0]          # a new fingerprint; the stale cache is removed

    # another price file changes (the integrity gate compares tickers with each other): a fresh cache again
    (prices / "BBB.csv").write_text((prices / "BBB.csv").read_text() + "\n")
    _fresh("AAA")
    assert [p for p in cache_on.iterdir() if p.is_dir()] != new_dirs


def test_cache_off(cache_on, tmp_path, monkeypatch):
    monkeypatch.setenv("BACKTESTER_CACHE", "0")
    assert not data.cache_enabled()
    if "SPY" in AVAIL:
        _fresh("SPY")
        assert not cache_on.exists() or not list(cache_on.rglob("*.pkl"))


@needs("MSFT", "SPY", "QQQ")
def test_cost_sensitivity_reuses_signals_with_the_same_numbers():
    s = parser.parse("buy MSFT at the close when it is down 4 days in a row, sell at the next close since 2015")
    res = runner.run(s)
    assert getattr(res, "_prepared", None) is not None
    rows = report.cost_sensitivity(res)
    plain = []
    for r in rows:
        s2 = dataclasses.replace(res.strategy, slippage_bps=float(r["slippage_bps"]), notes=list(res.strategy.notes))
        eq = runner.run(s2).equity
        plain.append(float(eq.iloc[-1]))
    assert [r["final_equity"] for r in rows] == plain
    # outside the context nothing is reused
    assert engine._PREP_REUSE["on"] == 0 and engine._PREP_REUSE["P"] is None


@needs("VTI", "BND", "SPY", "QQQ")
def test_site_writes_downloads_after_answering(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "RUNS", tmp_path)
    j = web.api_run({"text": "60% VTI 40% BND since 2018", "force": True})
    d = tmp_path / j["id"]
    assert (d / "report.html").is_file() and (d / "strategy.json").is_file() and (d / "summary.json").is_file()
    assert {"report.xlsx", "trades.csv", "equity.csv", "yearly.csv", "monthly.csv"} <= set(j["files"])
    web.wait_exports(d)
    assert set(j["files"]) == {p.name for p in d.iterdir()}
    assert pd.read_excel(d / "report.xlsx", sheet_name="Summary").shape[0] > 5


@needs("VTI", "BND", "SPY", "QQQ")
def test_site_waits_only_for_files_still_being_written(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "RUNS", tmp_path)
    real = report.export_jobs
    gate = threading.Event()
    monkeypatch.setattr(report, "export_jobs", lambda *a, **k: [("slow", gate.wait)] + real(*a, **k))
    j = web.api_run({"text": "60% VTI 40% BND since 2019", "force": True})
    d = tmp_path / j["id"]
    t = time.time()
    web.wait_exports(d, "report.html")          # written before the answer: no wait
    assert time.time() - t < 0.5 and not (d / "trades.csv").exists()
    threading.Timer(0.3, gate.set).start()
    web.wait_exports(d, "trades.csv")           # still to be written: waits for it
    assert (d / "trades.csv").is_file() and (d / "report.xlsx").is_file()


@needs("VTI", "BND", "SPY", "QQQ")
def test_typical_run_stays_fast(tmp_path):
    """A loose guard against a large slowdown (about 1-2 s of CPU on a laptop when this was written, including the
    cost-sensitivity reruns and every export)."""
    text = "60% VTI 40% BND since 2010"
    report.write_outputs(report.analyze(runner.run(parser.parse(text))), tmp_path / "warm")   # imports, data
    t = time.process_time()
    report.write_outputs(report.analyze(runner.run(parser.parse(text))), tmp_path / "run")
    assert time.process_time() - t < 15
