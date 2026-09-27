"""Monte Carlo glide path: the target mix moves once a year from the start mix to the end mix (linear, target-date
shaped or custom points) and the portfolio is rebalanced to it at every year boundary."""
import json

import numpy as np
import pandas as pd
import pytest

from backtester import data, montecarlo as mc


def test_linear_fraction_hits_both_ends():
    f = [mc.glide_fraction(k, 30) for k in range(35)]
    assert f[0] == 0.0 and f[29] == 1.0 and f[34] == 1.0            # year 1: start mix; year 30 on: end mix
    assert f[15] == pytest.approx(15 / 29) and np.all(np.diff(f[:30]) > 0)
    assert [mc.glide_fraction(k, 1) for k in range(3)] == [0.0, 1.0, 1.0]
    w = mc.glide_weights(np.array([0.9, 0.1]), np.array([0.4, 0.6]), 30 * 12, 30)
    assert w.shape == (360, 2) and np.allclose(w[:12], [0.9, 0.1]) and np.allclose(w[-12:], [0.4, 0.6])
    assert np.allclose(w.sum(axis=1), 1.0)
    assert all((w[12 * y: 12 * y + 12] == w[12 * y]).all() for y in range(30))   # constant within each year
    assert w[12, 0] == pytest.approx(0.9 - 0.5 / 29)                  # 1.72 points a year


def test_target_date_and_custom_schedules():
    td = [mc.glide_fraction(k, 30, "target_date") for k in range(30)]
    assert td[0] == 0.0 and td[5] == 0.0 and td[29] == 1.0            # holds the start mix for the first fifth
    assert np.all(np.diff(td) >= 0)
    lin = [mc.glide_fraction(k, 30) for k in range(30)]
    assert all(t <= l_ + 1e-12 for t, l_ in zip(td, lin))             # slower at first, faster near the end
    steps = np.diff(td)
    assert steps[-1] > steps[10] > 0
    pts = mc.parse_glide_points("0:0, 10:0, 20:50, 30:100")
    assert pts == [(0.0, 0.0), (10.0, 0.0), (20.0, 0.5), (30.0, 1.0)]
    assert [mc.glide_fraction(k, 30, points=pts) for k in (0, 10, 15, 20, 25, 40)] == [0, 0, 0.25, 0.5, 0.75, 1.0]
    for bad in ("0:0", "a:b", "0:0, 10:150"):
        with pytest.raises(ValueError):
            mc.parse_glide_points(bad)
    with pytest.raises(ValueError):
        mc.glide_fraction(3, 10, "bogus")


def test_portfolio_returns_rebalance_to_each_year_target():
    months = 36
    A = np.zeros((2, months, 2))
    A[:, :, 0] = 0.01                                                  # asset 0 earns 1% a month, asset 1 nothing
    W = mc.glide_weights(np.array([1.0, 0.0]), np.array([0.0, 1.0]), months, 3)   # 100/0, 50/50, 0/100
    P = mc._portfolio_returns(A, W, 0)                                 # never rebalanced except at the year ends
    assert np.allclose(P[:, :12], 0.01) and np.allclose(P[:, 24:], 0.0)
    # year 2 starts at 50/50 again (rebalanced on the boundary) and drifts within the year
    h0, h1 = 0.5, 0.5
    exp = []
    for _ in range(12):
        b = h0 + h1
        h0 *= 1.01
        exp.append((h0 + h1) / b - 1)
    assert np.allclose(P[0, 12:24], exp)
    # a constant target is the plain fixed-mix simulation
    rng = np.random.default_rng(1)
    B = rng.normal(0.005, 0.03, (50, 48, 3))
    w = np.array([0.5, 0.3, 0.2])
    assert np.allclose(mc._portfolio_returns(B, np.broadcast_to(w, (48, 3)), 12), mc._portfolio_returns(B, w, 12))


def _const_history(monkeypatch, rets: dict):
    idx = pd.date_range("1990-01-31", periods=240, freq="ME")
    hist = pd.DataFrame({t: r for t, r in rets.items()}, index=idx)
    monkeypatch.setattr(mc, "monthly_asset_returns", lambda tick, start=None, end=None: hist[tick])
    return hist


def test_run_with_a_glide_path_is_exact_on_constant_returns(monkeypatch):
    _const_history(monkeypatch, {"AAA": 0.01, "BBB": 0.002, "CCC": 0.0})
    s = mc.Settings(weights={"AAA": 90, "BBB": 10}, glide_to={"AAA": 40, "BBB": 50, "CCC": 10}, years=10, sims=100,
                    rebalance="monthly", inflation=0.02, start_balance=1000.0)
    R = mc.run(s)
    g = R["settings"]["glide"]
    assert g["years"] == 10 and g["schedule"] == "linear" and len(g["path"]) == 10
    assert g["path"][0]["weights"] == pytest.approx({"AAA": 0.9, "BBB": 0.1, "CCC": 0.0})
    assert g["path"][-1]["weights"] == pytest.approx({"AAA": 0.4, "BBB": 0.5, "CCC": 0.1})
    assert R["settings"]["weights"] == pytest.approx({"AAA": 0.9, "BBB": 0.1, "CCC": 0.0})
    # monthly rebalancing and constant returns: each year's return is its target mix times the returns
    growth = np.prod([(1 + (1 - f) * (0.9 * 0.01 + 0.1 * 0.002) + f * (0.4 * 0.01 + 0.5 * 0.002)) ** 12
                      for f in (k / 9 for k in range(10))])
    assert R["final"]["50"] == pytest.approx(1000 * growth, rel=1e-9) and R["final"]["10"] == pytest.approx(R["final"]["90"])
    assert any(n.startswith("Glide path (linear): from 90% AAA, 10% BBB to 40% AAA, 50% BBB, 10% CCC over 10 years")
               for n in R["notes"])
    # a glide over fewer years than the horizon: the end mix from year N on
    R2 = mc.run(mc.Settings(weights={"AAA": 90, "BBB": 10}, glide_to={"AAA": 40, "BBB": 60}, glide_years=5, years=10,
                            sims=100, inflation=0.02))
    p2 = R2["settings"]["glide"]["path"]
    assert p2[4]["weights"]["AAA"] == pytest.approx(0.4) and p2[9]["weights"]["AAA"] == pytest.approx(0.4)
    assert "the end mix is held from year 5 on" in " ".join(R2["notes"])
    # the same start and end mix is the fixed mix
    a = mc.run(mc.Settings(weights={"AAA": 60, "BBB": 40}, years=5, sims=200, inflation=0.02, model="normal"))
    b = mc.run(mc.Settings(weights={"AAA": 60, "BBB": 40}, glide_to={"AAA": 60, "BBB": 40}, years=5, sims=200,
                           inflation=0.02, model="normal"))
    assert a["final"] == pytest.approx(b["final"])


def test_glide_errors(monkeypatch):
    _const_history(monkeypatch, {"AAA": 0.01, "BBB": 0.0})
    ser = pd.Series(0.01, index=pd.date_range("2000-01-31", periods=60, freq="ME"))
    with pytest.raises(ValueError, match="glide path needs tickers"):
        mc.run(mc.Settings(series=ser, glide_to={"AAA": 1}, years=5, sims=100, inflation=0.02))
    with pytest.raises(ValueError, match="End-mix weights must add up"):
        mc.run(mc.Settings(weights={"AAA": 1}, glide_to={"BBB": 0}, years=5, sims=100, inflation=0.02))
    with pytest.raises(ValueError, match="1 to 100 years"):
        mc.run(mc.Settings(weights={"AAA": 1}, glide_to={"BBB": 1}, glide_years=0, years=5, sims=100, inflation=0.02))


needs = pytest.mark.skipif(not {"SPY", "TLT"} <= set(data.available_tickers()), reason="price data not available")


@needs
def test_site_api_and_cli(capsys):
    from backtester import web
    from backtester.__main__ import cmd_montecarlo
    R = web.api_montecarlo({"weights": "SPY 90 TLT 10", "glide_to": "SPY 40, TLT 60", "glide_years": 20,
                            "glide": "target_date", "years": 25, "sims": 200, "flows": [{"type": "none"}]})
    g = R["settings"]["glide"]
    assert g["schedule"] == "target_date" and g["years"] == 20 and g["to"] == pytest.approx({"SPY": 0.4, "TLT": 0.6})
    assert g["path"][0]["weights"]["SPY"] == pytest.approx(0.9) and g["path"][19]["weights"]["SPY"] == pytest.approx(0.4)
    R = web.api_montecarlo({"weights": "SPY 90 TLT 10", "glide_to": "SPY 40 TLT 60", "glide_points": "0:0,5:100",
                            "years": 10, "sims": 200})
    assert R["settings"]["glide"]["schedule"] == "custom" and R["settings"]["glide"]["path"][5]["weights"]["TLT"] == pytest.approx(0.6)
    with pytest.raises(web.ClientError):
        web.api_montecarlo({"text": "hold 60% SPY and 40% TLT", "glide_to": "SPY 40 TLT 60", "sims": 200})
    with pytest.raises(web.ClientError):
        web.api_montecarlo({"weights": "SPY 90 TLT 10", "glide_to": "SPY 40 TLT 60", "glide": "zigzag", "sims": 200})
    assert cmd_montecarlo(["--weights", "SPY 90 TLT 10", "--glide-to", "SPY 40 TLT 60", "--years", "20", "--sims", "200",
                           "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["settings"]["glide"]["path"][-1]["weights"]["TLT"] == pytest.approx(0.6)
    assert cmd_montecarlo(["--weights", "SPY 90 TLT 10", "--glide-to", "SPY 40 TLT 60", "--glide", "target_date",
                           "--years", "20", "--sims", "200"]) == 0
    txt = capsys.readouterr().out
    assert "gliding to 40.0% SPY, 60.0% TLT over 20 years (target-date)" in txt and "Glide path (target-date)" in txt
    with pytest.raises(ValueError, match="need --glide-to"):
        cmd_montecarlo(["--weights", "SPY 90 TLT 10", "--glide-years", "10", "--sims", "200"])
