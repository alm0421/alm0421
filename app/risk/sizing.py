"""Position sizing.

Sizing is risk-based, not notional-based: the share count is derived from how
much money the account is willing to lose if the stop is hit, divided by the
per-share distance to that stop. A signal with no stop cannot be sized this way
and is refused rather than falling back to an arbitrary fixed size.

The result is then clamped by every applicable notional cap, so no single
input can produce an oversized position.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from app.config import RiskConfig
from app.enums import AssetClass
from app.signals.models import Signal


@dataclass(frozen=True)
class PositionSize:
    """A computed position size and the reasoning behind it."""

    quantity: float
    #: Notional value at the reference price.
    notional: float
    #: Money at risk if the stop fills exactly.
    risk_amount: float
    #: Which constraint ended up binding, for operator visibility.
    binding_constraint: str
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def is_tradable(self) -> bool:
        return self.quantity > 0


def size_position(
    signal: Signal,
    *,
    equity: float,
    risk_config: RiskConfig,
    risk_per_trade_pct: float = 0.01,
    allow_fractional: bool | None = None,
) -> PositionSize:
    """Compute the share/contract count for ``signal``.

    Parameters
    ----------
    equity
        Account equity. Must be known and positive; an unknown account cannot
        be sized against.
    risk_per_trade_pct
        Fraction of equity to risk if the stop is hit. 0.01 = 1%.
    allow_fractional
        Whether fractional quantities are permitted. Defaults to True for
        crypto and False otherwise, matching what the venues actually accept.
    """
    if allow_fractional is None:
        allow_fractional = signal.asset_class is AssetClass.CRYPTO

    detail: dict[str, Any] = {
        "equity": equity,
        "risk_per_trade_pct": risk_per_trade_pct,
        "reference_price": signal.reference_price,
        "stop_price": signal.stop_price,
    }

    if equity <= 0:
        return PositionSize(0.0, 0.0, 0.0, "unknown_or_zero_equity", detail)

    risk_per_share = signal.risk_per_share
    if risk_per_share is None or risk_per_share <= 0:
        # No stop means no defined risk. Refuse rather than inventing a size.
        return PositionSize(
            0.0, 0.0, 0.0, "no_stop_loss",
            {**detail, "note": "Signal carries no usable stop; risk-based sizing is impossible."},
        )

    price = signal.reference_price
    if price <= 0:
        return PositionSize(0.0, 0.0, 0.0, "invalid_price", detail)

    # 1. Risk-based size.
    risk_budget = equity * risk_per_trade_pct
    qty_by_risk = risk_budget / risk_per_share

    # 2. Notional cap.
    qty_by_notional = risk_config.max_position_notional / price

    # 3. Percent-of-equity cap.
    qty_by_equity_pct = (equity * risk_config.max_position_pct_equity) / price

    candidates = {
        "risk_budget": qty_by_risk,
        "max_position_notional": qty_by_notional,
        "max_position_pct_equity": qty_by_equity_pct,
    }
    binding_constraint = min(candidates, key=lambda k: candidates[k])
    quantity = candidates[binding_constraint]

    if not allow_fractional:
        # Floor, never round: rounding up could breach the cap that was binding.
        quantity = float(math.floor(quantity))
    else:
        # Crypto venues accept fractions but not unlimited precision.
        quantity = math.floor(quantity * 1e8) / 1e8

    if quantity <= 0:
        return PositionSize(
            0.0, 0.0, 0.0, f"{binding_constraint}_below_one_unit",
            {
                **detail,
                "candidates": {k: round(v, 8) for k, v in candidates.items()},
                "note": (
                    "Every cap resolved to less than one tradable unit. The account "
                    "is too small for this price and stop distance."
                ),
            },
        )

    return PositionSize(
        quantity=quantity,
        notional=round(quantity * price, 6),
        risk_amount=round(quantity * risk_per_share, 6),
        binding_constraint=binding_constraint,
        detail={
            **detail,
            "candidates": {k: round(v, 8) for k, v in candidates.items()},
            "risk_budget": round(risk_budget, 6),
            "risk_per_share": round(risk_per_share, 6),
            "allow_fractional": allow_fractional,
        },
    )
