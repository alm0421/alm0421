"""EMA pullback continuation strategy.

Intent
------
Trade a pullback *within* an established trend, not a reversal against one.

Long setup (short is the exact mirror)
--------------------------------------
1. **Trend filter** - the last close is above the slow trend EMA, so the
   prevailing direction is up.
2. **Momentum alignment** - the fast EMA is above the slow EMA.
3. **Pullback then reclaim** - the *previous* bar closed below the fast EMA
   (the pullback) and the *current* bar closed back above it (the reclaim).
   Requiring both bars is what makes this a pullback entry rather than a
   breakout chase.
4. **Not already extended** - RSI sits inside ``[rsi_floor, rsi_ceiling]``.
   Below the floor the trend is failing; above the ceiling the move is already
   stretched and the stop distance becomes unattractive.

Exits
-----
Stop and target are both ATR-derived, so they adapt to the symbol's own
volatility instead of using a fixed percentage that is too tight for a volatile
name and too loose for a quiet one.

* stop   = close -/+ ``stop_atr_multiple``   * ATR
* target = close +/- ``target_atr_multiple`` * ATR

Assumptions and limitations
---------------------------
* Evaluated on the **last completed bar**. The caller is responsible for having
  dropped any still-forming bar (see :func:`app.data.base.drop_forming_bar`).
* Uses bar closes only; it does not model intrabar path, so a stop and target
  that would both be touched inside one bar is not resolved here. A backtest
  engine must decide that ordering explicitly.
* Produces at most one signal per symbol per evaluation. It holds no position
  state and does not know whether a position is already open - that is the
  portfolio and risk layer's responsibility.
* No slippage, commission or borrow availability is considered at this layer.
"""

from __future__ import annotations

from app.data.base import BarSet
from app.enums import SignalDirection
from app.signals import indicators as ind
from app.signals.models import Signal
from app.strategies.base import Strategy, StrategyError, register


@register
class EmaPullbackStrategy(Strategy):
    name = "ema_pullback"

    DEFAULTS = {
        "fast_period": 9,
        "slow_period": 21,
        "trend_period": 50,
        "rsi_period": 14,
        "rsi_floor": 40.0,
        "rsi_ceiling": 70.0,
        "atr_period": 14,
        "stop_atr_multiple": 1.5,
        "target_atr_multiple": 3.0,
        "min_confidence": 0.55,
        "allow_short": True,
    }

    def _validate_params(self) -> None:
        merged = {**self.DEFAULTS, **self.params}
        self.params = merged

        fast, slow, trend = merged["fast_period"], merged["slow_period"], merged["trend_period"]
        if not (fast < slow < trend):
            raise StrategyError(
                f"{self.name}: periods must satisfy fast < slow < trend, got "
                f"fast={fast}, slow={slow}, trend={trend}"
            )
        for key in ("fast_period", "slow_period", "trend_period", "rsi_period", "atr_period"):
            if int(merged[key]) <= 0:
                raise StrategyError(f"{self.name}: {key} must be positive, got {merged[key]}")
        if not 0.0 <= merged["rsi_floor"] < merged["rsi_ceiling"] <= 100.0:
            raise StrategyError(
                f"{self.name}: require 0 <= rsi_floor < rsi_ceiling <= 100, got "
                f"{merged['rsi_floor']} / {merged['rsi_ceiling']}"
            )
        if merged["stop_atr_multiple"] <= 0 or merged["target_atr_multiple"] <= 0:
            raise StrategyError(f"{self.name}: ATR multiples must be positive")
        if merged["target_atr_multiple"] <= merged["stop_atr_multiple"]:
            raise StrategyError(
                f"{self.name}: target_atr_multiple ({merged['target_atr_multiple']}) must exceed "
                f"stop_atr_multiple ({merged['stop_atr_multiple']}), otherwise every trade has a "
                "reward:risk below 1.0"
            )
        if not 0.0 <= merged["min_confidence"] <= 1.0:
            raise StrategyError(f"{self.name}: min_confidence must be within 0.0-1.0")

    @property
    def min_bars(self) -> int:
        # The trend EMA is the slowest series; ATR/RSI need their own warm-up.
        # +2 covers the previous-bar comparison and one settled value.
        return int(
            max(
                self.params["trend_period"],
                self.params["atr_period"],
                self.params["rsi_period"],
            )
        ) + 2

    def evaluate(self, bars: BarSet) -> Signal | None:
        if len(bars) < self.min_bars:
            return None

        closes = bars.closes
        highs = bars.highs
        lows = bars.lows

        fast = ind.ema(closes, int(self.params["fast_period"]))
        slow = ind.ema(closes, int(self.params["slow_period"]))
        trend = ind.ema(closes, int(self.params["trend_period"]))
        rsi_series = ind.rsi(closes, int(self.params["rsi_period"]))
        atr_series = ind.atr(highs, lows, closes, int(self.params["atr_period"]))

        i = len(closes) - 1
        fast_now, fast_prev = fast[i], fast[i - 1]
        slow_now, trend_now = slow[i], trend[i]
        rsi_now, atr_now = rsi_series[i], atr_series[i]

        # Any unwarmed indicator means we cannot evaluate honestly.
        if None in (fast_now, fast_prev, slow_now, trend_now, rsi_now, atr_now):
            return None
        if atr_now <= 0:
            return None

        close_now, close_prev = closes[i], closes[i - 1]
        rsi_floor = float(self.params["rsi_floor"])
        rsi_ceiling = float(self.params["rsi_ceiling"])

        direction: SignalDirection | None = None
        reasons: list[str] = []

        uptrend = close_now > trend_now and fast_now > slow_now
        downtrend = close_now < trend_now and fast_now < slow_now

        if uptrend and close_prev < fast_prev and close_now > fast_now:
            if rsi_floor <= rsi_now <= rsi_ceiling:
                direction = SignalDirection.LONG
                reasons = [
                    f"close {close_now:.4f} above trend EMA{int(self.params['trend_period'])} {trend_now:.4f}",
                    f"fast EMA {fast_now:.4f} above slow EMA {slow_now:.4f}",
                    f"pullback below fast EMA on prior bar ({close_prev:.4f} < {fast_prev:.4f}) then reclaim",
                    f"RSI {rsi_now:.1f} within [{rsi_floor:.0f}, {rsi_ceiling:.0f}]",
                ]
        elif downtrend and bool(self.params["allow_short"]) and close_prev > fast_prev and close_now < fast_now:
            # Mirror the RSI band around 50 for shorts.
            if (100.0 - rsi_ceiling) <= rsi_now <= (100.0 - rsi_floor):
                direction = SignalDirection.SHORT
                reasons = [
                    f"close {close_now:.4f} below trend EMA{int(self.params['trend_period'])} {trend_now:.4f}",
                    f"fast EMA {fast_now:.4f} below slow EMA {slow_now:.4f}",
                    f"rally above fast EMA on prior bar ({close_prev:.4f} > {fast_prev:.4f}) then rejection",
                    f"RSI {rsi_now:.1f} within mirrored band "
                    f"[{100.0 - rsi_ceiling:.0f}, {100.0 - rsi_floor:.0f}]",
                ]

        if direction is None:
            return None

        stop_distance = atr_now * float(self.params["stop_atr_multiple"])
        target_distance = atr_now * float(self.params["target_atr_multiple"])

        if direction is SignalDirection.LONG:
            stop_price = close_now - stop_distance
            target_price = close_now + target_distance
        else:
            stop_price = close_now + stop_distance
            target_price = close_now - target_distance

        # A stop at or below zero cannot be represented as a real order.
        if stop_price <= 0 or target_price <= 0:
            return None

        confidence = self._confidence(
            close_now=close_now, trend_now=trend_now, fast_now=fast_now,
            slow_now=slow_now, rsi_now=rsi_now, atr_now=atr_now, direction=direction,
        )
        if confidence < float(self.params["min_confidence"]):
            return None

        return Signal(
            symbol=bars.symbol,
            asset_class=bars.asset_class,
            direction=direction,
            strategy=self.name,
            confidence=confidence,
            reference_price=close_now,
            stop_price=stop_price,
            target_price=target_price,
            timeframe=bars.timeframe,
            data_freshness=bars.freshness,
            data_timestamp=bars.bars[i].timestamp,
            rationale={
                "reasons": reasons,
                "indicators": {
                    "ema_fast": round(fast_now, 6),
                    "ema_slow": round(slow_now, 6),
                    "ema_trend": round(trend_now, 6),
                    "rsi": round(rsi_now, 4),
                    "atr": round(atr_now, 6),
                },
                "bar_count": len(bars),
                "source": bars.source,
            },
        )

    def _confidence(
        self, *, close_now: float, trend_now: float, fast_now: float, slow_now: float,
        rsi_now: float, atr_now: float, direction: SignalDirection,
    ) -> float:
        """Blend three bounded components into a 0-1 conviction score.

        This is a strategy-relative ranking aid, not a probability. It is only
        meaningful when comparing signals from this same strategy.
        """
        # 1. Trend separation, normalised by ATR and capped at 3 ATR.
        separation = abs(close_now - trend_now) / atr_now
        trend_score = min(separation / 3.0, 1.0)

        # 2. EMA spread, also ATR-normalised and capped.
        spread_score = min((abs(fast_now - slow_now) / atr_now) / 1.5, 1.0)

        # 3. RSI position: best at the midpoint of the permitted band, falling
        #    off towards either edge.
        floor = float(self.params["rsi_floor"])
        ceiling = float(self.params["rsi_ceiling"])
        if direction is SignalDirection.SHORT:
            floor, ceiling = 100.0 - ceiling, 100.0 - floor
        midpoint = (floor + ceiling) / 2.0
        half_width = max((ceiling - floor) / 2.0, 1e-9)
        rsi_score = max(0.0, 1.0 - abs(rsi_now - midpoint) / half_width)

        score = 0.45 * trend_score + 0.30 * spread_score + 0.25 * rsi_score
        return round(min(max(score, 0.0), 1.0), 4)
