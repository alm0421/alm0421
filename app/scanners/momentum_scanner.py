"""Momentum scanner.

Shortlists symbols that are moving with unusual participation, while excluding
names that are untradeable in practice.

Filters (all must pass)
-----------------------
``min_price`` / ``max_price``
    Excludes sub-dollar names, where spreads and borrow make the strategy's
    ATR-based stops meaningless, and very high-priced names where the position
    sizer cannot buy a sensible share count.
``min_avg_volume``
    Average share volume per bar over the lookback. A liquidity floor.
``min_relative_volume``
    Current bar volume against the trailing average. This is the participation
    filter - price movement on ordinary volume is noise.
``min_abs_momentum_pct``
    Absolute percent change over ``momentum_lookback`` bars. Absolute, so the
    scanner surfaces both long and short candidates and lets the strategy decide
    direction.
``min_atr_pct`` / ``max_atr_pct``
    ATR as a percentage of price. Floors out names too quiet to reach a target,
    and caps names so volatile that an ATR stop would exceed risk limits.

Score
-----
A weighted blend of normalised momentum, relative volume and ATR percentage.
It ranks *within this scanner only* and is not comparable to a strategy's
confidence score.
"""

from __future__ import annotations

from app.data.base import BarSet
from app.scanners.base import ScanResult, Scanner, ScannerError, register
from app.signals import indicators as ind


@register
class MomentumScanner(Scanner):
    name = "momentum"

    DEFAULTS = {
        "min_price": 5.0,
        "max_price": 2000.0,
        "min_avg_volume": 10_000.0,
        "min_relative_volume": 1.2,
        "min_abs_momentum_pct": 0.5,
        "min_atr_pct": 0.3,
        "max_atr_pct": 12.0,
        "momentum_lookback": 12,
        "volume_lookback": 20,
        "atr_period": 14,
    }

    def _validate_params(self) -> None:
        merged = {**self.DEFAULTS, **self.params}
        self.params = merged
        if merged["min_price"] >= merged["max_price"]:
            raise ScannerError(
                f"{self.name}: min_price ({merged['min_price']}) must be below "
                f"max_price ({merged['max_price']})"
            )
        if merged["min_atr_pct"] >= merged["max_atr_pct"]:
            raise ScannerError(
                f"{self.name}: min_atr_pct must be below max_atr_pct"
            )
        for key in ("momentum_lookback", "volume_lookback", "atr_period"):
            if int(merged[key]) <= 0:
                raise ScannerError(f"{self.name}: {key} must be positive")
        if merged["min_relative_volume"] <= 0:
            raise ScannerError(f"{self.name}: min_relative_volume must be positive")

    @property
    def min_bars(self) -> int:
        return int(
            max(
                self.params["momentum_lookback"],
                self.params["volume_lookback"],
                self.params["atr_period"],
            )
        ) + 2

    def evaluate_symbol(self, bars: BarSet) -> ScanResult | None:
        closes = bars.closes
        last_price = closes[-1]

        if not (float(self.params["min_price"]) <= last_price <= float(self.params["max_price"])):
            return None

        volumes = bars.volumes
        volume_lookback = int(self.params["volume_lookback"])
        avg_volume = sum(volumes[-volume_lookback:]) / volume_lookback
        if avg_volume < float(self.params["min_avg_volume"]):
            return None

        rel_volume = ind.last_value(ind.relative_volume(volumes, volume_lookback))
        if rel_volume is None or rel_volume < float(self.params["min_relative_volume"]):
            return None

        momentum = ind.last_value(
            ind.percent_change(closes, int(self.params["momentum_lookback"]))
        )
        if momentum is None or abs(momentum) < float(self.params["min_abs_momentum_pct"]):
            return None

        atr_value = ind.last_value(
            ind.atr(bars.highs, bars.lows, closes, int(self.params["atr_period"]))
        )
        if atr_value is None or last_price <= 0:
            return None
        atr_pct = (atr_value / last_price) * 100.0
        if not (float(self.params["min_atr_pct"]) <= atr_pct <= float(self.params["max_atr_pct"])):
            return None

        score = self._score(momentum=momentum, rel_volume=rel_volume, atr_pct=atr_pct)

        return ScanResult(
            symbol=bars.symbol,
            asset_class=bars.asset_class,
            scanner=self.name,
            score=score,
            last_price=last_price,
            metrics={
                "momentum_pct": round(momentum, 4),
                "relative_volume": round(rel_volume, 4),
                "avg_volume": round(avg_volume, 2),
                "atr": round(atr_value, 6),
                "atr_pct": round(atr_pct, 4),
                "bars_used": len(bars),
            },
            data_freshness=bars.freshness,
            data_timestamp=bars.last_timestamp,
        )

    def _score(self, *, momentum: float, rel_volume: float, atr_pct: float) -> float:
        """Blend the three signals into a bounded 0-1 rank score.

        Each component is capped before weighting so a single extreme value
        (a 400% relative volume print, say) cannot dominate the ranking.
        """
        momentum_score = min(abs(momentum) / 5.0, 1.0)        # 5% move saturates
        volume_score = min(rel_volume / 4.0, 1.0)             # 4x volume saturates
        atr_score = min(atr_pct / float(self.params["max_atr_pct"]), 1.0)
        return round(0.50 * momentum_score + 0.35 * volume_score + 0.15 * atr_score, 6)
