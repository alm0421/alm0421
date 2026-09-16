"""Scanner filtering and ranking."""

from __future__ import annotations

import pytest

from app.scanners import MomentumScanner
from app.scanners.base import ScannerError, build_scanners
from tests.conftest import flat_closes, make_bars, uptrend_pullback_closes


@pytest.fixture()
def scanner() -> MomentumScanner:
    return MomentumScanner({})


def surging_volumes(n: int = 80) -> list[float]:
    """Flat baseline volume with a surge on the final bar."""
    return [100_000.0] * (n - 1) + [400_000.0]


def test_trending_symbol_with_volume_surge_passes(scanner):
    bars = make_bars("AAPL", closes=uptrend_pullback_closes(), volumes=surging_volumes())
    assert scanner.evaluate_symbol(bars) is not None


def test_flat_symbol_filtered_out(scanner):
    """No movement means no momentum, whatever the volume."""
    bars = make_bars("AAPL", closes=flat_closes(), volumes=surging_volumes(), spread=0.0)
    assert scanner.evaluate_symbol(bars) is None


def test_low_relative_volume_filtered_out(scanner):
    """Price movement on ordinary volume is noise, not participation."""
    bars = make_bars("AAPL", closes=uptrend_pullback_closes(), volumes=[100_000.0] * 80)
    assert scanner.evaluate_symbol(bars) is None


def test_illiquid_symbol_filtered_out(scanner):
    bars = make_bars("AAPL", closes=uptrend_pullback_closes(), volumes=[10.0] * 79 + [40.0])
    assert scanner.evaluate_symbol(bars) is None


def test_sub_dollar_symbol_filtered_out(scanner):
    closes = [c / 200.0 for c in uptrend_pullback_closes()]
    bars = make_bars("PENNY", closes=closes, volumes=surging_volumes())
    assert scanner.evaluate_symbol(bars) is None


def test_results_ranked_by_score_descending(scanner):
    bar_sets = {
        "STRONG": make_bars("STRONG", closes=uptrend_pullback_closes(), volumes=surging_volumes()),
        "WEAK": make_bars(
            "WEAK",
            closes=[100.0 + i * 0.08 for i in range(80)],
            volumes=[100_000.0] * 79 + [150_000.0],
        ),
    }
    results = scanner.scan(bar_sets)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_scan_is_deterministic(scanner):
    bar_sets = {
        "AAA": make_bars("AAA", closes=uptrend_pullback_closes(), volumes=surging_volumes()),
        "BBB": make_bars("BBB", closes=uptrend_pullback_closes(), volumes=surging_volumes()),
    }
    first = [r.symbol for r in scanner.scan(bar_sets)]
    second = [r.symbol for r in scanner.scan(bar_sets)]
    assert first == second


def test_limit_is_respected(scanner):
    bar_sets = {
        f"S{i}": make_bars(f"S{i}", closes=uptrend_pullback_closes(), volumes=surging_volumes())
        for i in range(10)
    }
    assert len(scanner.scan(bar_sets, limit=3)) == 3


def test_short_history_skipped(scanner):
    bar_sets = {"AAPL": make_bars("AAPL", closes=[100.0] * 5)}
    assert scanner.scan(bar_sets) == []


def test_one_bad_symbol_does_not_abort_the_scan(scanner, monkeypatch):
    """A single failure must not blind the operator to the rest of the universe."""
    good = make_bars("GOOD", closes=uptrend_pullback_closes(), volumes=surging_volumes())
    bad = make_bars("BAD", closes=uptrend_pullback_closes(), volumes=surging_volumes())

    original = scanner.evaluate_symbol

    def explode(bars):
        if bars.symbol == "BAD":
            raise RuntimeError("boom")
        return original(bars)

    monkeypatch.setattr(scanner, "evaluate_symbol", explode)
    results = scanner.scan({"GOOD": good, "BAD": bad})
    assert [r.symbol for r in results] == ["GOOD"]


def test_result_carries_data_provenance(scanner):
    bars = make_bars("AAPL", closes=uptrend_pullback_closes(), volumes=surging_volumes())
    result = scanner.evaluate_symbol(bars)
    assert result.data_freshness is bars.freshness
    assert result.data_timestamp == bars.last_timestamp
    assert "relative_volume" in result.metrics


@pytest.mark.parametrize(
    "params,match",
    [
        ({"min_price": 100, "max_price": 10}, "min_price"),
        ({"min_atr_pct": 10, "max_atr_pct": 1}, "min_atr_pct"),
        ({"momentum_lookback": 0}, "positive"),
    ],
)
def test_invalid_params_rejected(params, match):
    with pytest.raises(ScannerError, match=match):
        MomentumScanner(params)


def test_unknown_scanner_is_a_hard_error():
    with pytest.raises(ScannerError, match="Unknown scanner"):
        build_scanners(("nope",))
