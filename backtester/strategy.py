"""Strategy specification."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Literal

Fill = Literal["close", "open", "next_open", "next_close"]


@dataclass
class Strategy:
    universe: list[str]
    entry: str                                   # expression, see expr.HELP
    side: Literal["long", "short"] = "long"
    entry_fill: Fill = "close"                   # signal-bar close; "open" = same bar's open (rule must
                                                 # only use open-time data, e.g. gap); or next bar open/close

    # exits (first one to trigger wins)
    hold_bars: int | None = None                 # exit N bars after entry
    hold_exit_fill: Literal["close", "open"] = "close"
    exit_when: str | None = None                 # expression; may use bars_held / entry_price / pnl
    exit_when_fill: Literal["close", "next_open"] = "close"
    stop_loss: float | None = None               # 0.05 = exit if price moves 5% against you
    take_profit: float | None = None             # 0.10 = exit at +10%
    trailing_stop: float | None = None           # 0.08 = exit 8% below highest high since entry

    # portfolio
    capital: float = 10_000.0
    max_positions: int = 1
    position_size: float | None = None           # fraction of equity per position; default 1/max_positions
    leverage: float = 1.0                        # max gross exposure / equity for longs
    rank_by: str | None = None                   # when more signals than free slots, prefer highest value
    rank_ascending: bool = False
    fractional_shares: bool = True

    # costs
    commission: float = 0.0                      # $ per order
    commission_per_share: float = 0.0            # $ per share
    slippage_bps: float = 0.0                    # per side, in basis points

    # period
    start: str | None = None
    end: str | None = None

    name: str = ""
    description: str = ""
    notes: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if not self.universe:
            raise ValueError("universe is empty")
        if not any([self.hold_bars, self.exit_when, self.stop_loss, self.take_profit, self.trailing_stop]):
            raise ValueError("strategy needs at least one exit rule (hold_bars, exit_when, stop_loss, take_profit, trailing_stop)")
        if self.hold_bars is not None and self.hold_bars < 0:
            raise ValueError("hold_bars must be >= 0")
        if self.max_positions < 1:
            raise ValueError("max_positions must be >= 1")
        if self.entry_fill == "open":
            from .expr import open_safe
            if not open_safe(self.entry):
                raise ValueError(
                    "entry_fill 'open' needs a rule known at the open, but this rule uses today's "
                    "close/high/low (e.g. ret(1) or rsi(2) default to today's close). Wrap those parts "
                    "in ref(..., 1) to use yesterday's value, or use entry_fill 'next_open'.")
        if self.position_size is None:
            self.position_size = 1.0 / self.max_positions

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "Strategy":
        return cls(**d)

    def summary(self) -> str:
        side = "Buy" if self.side == "long" else "Short"
        fill = {"close": "at the close of the signal day", "open": "at the open of the signal day",
                "next_open": "at the next day's open",
                "next_close": "at the next day's close"}[self.entry_fill]
        u = self.universe
        uni = ", ".join(u) if len(u) <= 6 else f"{len(u)} tickers ({', '.join(u[:4])}, ...)"
        lines = [f"{side} {uni} {fill} when: {self.entry}"]
        ex = []
        if self.hold_bars is not None:
            ex.append(f"after {self.hold_bars} bar(s) at the {self.hold_exit_fill}")
        if self.exit_when:
            ex.append(f"when {self.exit_when} ({'same close' if self.exit_when_fill == 'close' else 'next open'})")
        if self.stop_loss:
            ex.append(f"stop loss {self.stop_loss:.1%}")
        if self.take_profit:
            ex.append(f"take profit {self.take_profit:.1%}")
        if self.trailing_stop:
            ex.append(f"trailing stop {self.trailing_stop:.1%}")
        lines.append("Exit: " + "; ".join(ex))
        lines.append(
            f"Sizing: ${self.capital:,.0f} start, up to {self.max_positions} position(s) at "
            f"{self.position_size:.0%} of equity each"
            + (f", ranked by {'lowest' if self.rank_ascending else 'highest'} {self.rank_by}" if self.rank_by else "")
        )
        costs = []
        if self.commission:
            costs.append(f"${self.commission:g}/order")
        if self.commission_per_share:
            costs.append(f"${self.commission_per_share:g}/share")
        if self.slippage_bps:
            costs.append(f"{self.slippage_bps:g} bps slippage/side")
        lines.append("Costs: " + (", ".join(costs) if costs else "none"))
        if self.start or self.end:
            lines.append(f"Period: {self.start or 'start of data'} to {self.end or 'latest'}")
        return "\n".join(lines)
