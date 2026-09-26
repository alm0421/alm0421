"""Strategy specification for rule-based (signal) strategies."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from typing import Literal

Fill = Literal["close", "open", "next_open", "next_close"]


@dataclass
class Strategy:
    universe: list[str]
    entry: str                                   # expression, see expr.HELP
    side: Literal["long", "short", "both"] = "long"
    short_entry: str | None = None               # side="both": rule for going short
    reverse: bool = True                         # side="both": an opposite signal closes and reverses
    entry_fill: Fill = "close"                   # signal-bar close; "open" = same bar's open (rule must
                                                 # only use open-time data, e.g. gap); or next bar open/close
    entry_order: Literal["market", "limit", "stop"] = "market"
    entry_level: str | None = None               # limit/stop price expression, evaluated on the signal bar
    order_valid_bars: int = 1                    # limit/stop orders are live for this many bars
    pyramiding: int = 1                          # max entries per ticker while the signal repeats

    # exits (first one to trigger wins)
    hold_bars: int | None = None                 # exit N bars after entry
    hold_exit_fill: Literal["close", "open"] = "close"
    exit_when: str | None = None                 # expression; may use bars_held / entry_price / pnl /
                                                 # highest_since_entry / lowest_since_entry
    exit_when_fill: Literal["close", "next_open"] = "close"
    stop_loss: float | None = None               # 0.05 = exit if price moves 5% against you
    stop_atr: float | None = None                # stop N x ATR(atr_period) from entry
    take_profit: float | None = None             # 0.10 = exit at +10%
    take_profit_atr: float | None = None
    trailing_stop: float | None = None           # 0.08 = exit 8% below highest high since entry
    trailing_atr: float | None = None            # chandelier: N x ATR below highest high since entry
    atr_period: int = 14
    scale_out: list[dict] = field(default_factory=list)   # [{"at": 0.05, "fraction": 0.5}, ...]

    # portfolio and sizing
    capital: float = 10_000.0
    max_positions: int = 1
    sizing: Literal["percent", "fixed_dollars", "fixed_shares", "risk", "volatility"] = "percent"
    position_size: float | None = None           # percent: fraction of equity per position (default 1/max_positions)
    fixed_amount: float | None = None            # fixed_dollars ($) or fixed_shares (shares)
    risk_per_trade: float | None = None          # risk: fraction of equity lost if the stop is hit
    target_vol: float | None = None              # volatility: annualised vol per position (e.g. 0.15)
    leverage: float = 1.0                        # max gross exposure / equity
    rank_by: str | None = None                   # when more signals than free slots, prefer highest value
    rank_ascending: bool = False
    fractional_shares: bool = True
    point_in_time: bool = True                   # only enter index stocks while they were members
    universe_name: str | None = None             # e.g. "NDX" when the universe is an index
    benchmark: str | None = None                 # comparison ticker for alpha/beta (default SPY)

    # costs and financing
    commission: float = 0.0                      # $ per order
    commission_per_share: float = 0.0            # $ per share
    commission_pct: float = 0.0                  # fraction of traded value
    slippage_bps: float = 0.0                    # per side, in basis points
    max_volume_pct: float | None = None          # cap each order at this fraction of the bar's volume
    cash_rate: str | float | None = "tbill"      # interest on idle cash: "tbill", annual rate, or None
    margin_rate: float = 0.0                     # annual rate charged on borrowed cash (added to T-bill)
    borrow_fee: float = 0.0                      # annual fee on short market value

    # period
    start: str | None = None
    end: str | None = None

    name: str = ""
    description: str = ""
    notes: list[str] = field(default_factory=list)

    def validate(self) -> None:
        from .expr import compile_expr, open_safe
        if not self.universe:
            raise ValueError("universe is empty")
        if not any([self.hold_bars, self.exit_when, self.stop_loss, self.take_profit, self.trailing_stop,
                    self.stop_atr, self.take_profit_atr, self.trailing_atr, self.side == "both"]):
            raise ValueError("strategy needs at least one exit rule (hold_bars, exit_when, stop_loss, "
                             "take_profit, trailing_stop, stop_atr, trailing_atr)")
        if self.hold_bars is not None and self.hold_bars < 1:
            raise ValueError("hold_bars must be at least 1")
        if self.max_positions < 1:
            raise ValueError("max_positions must be at least 1")
        if self.pyramiding < 1:
            raise ValueError("pyramiding must be at least 1")
        if self.side == "both" and not self.short_entry:
            raise ValueError("side 'both' needs a short_entry rule")
        for rule in (self.entry, self.short_entry, self.exit_when, self.entry_level, self.rank_by):
            if rule:
                compile_expr(rule)  # raises on syntax errors / disallowed constructs
        if self.entry_fill == "open":
            for rule in (self.entry, self.short_entry):
                if rule and not open_safe(rule):
                    raise ValueError(
                        "entry_fill 'open' needs a rule known at the open, but this rule uses today's "
                        "close/high/low (e.g. ret(1) or rsi(2) default to today's close). Wrap those parts "
                        "in ref(..., 1) to use yesterday's value, or use entry_fill 'next_open'.")
        if self.entry_order != "market":
            if not self.entry_level:
                raise ValueError("limit/stop entries need entry_level (e.g. 'close * 0.98')")
            if self.entry_fill not in ("next_open", "close"):
                self.entry_fill = "next_open"
        if self.sizing == "risk" and not (self.risk_per_trade and (self.stop_loss or self.stop_atr)):
            raise ValueError("risk sizing needs risk_per_trade and a stop_loss or stop_atr")
        if self.sizing == "volatility" and not self.target_vol:
            raise ValueError("volatility sizing needs target_vol")
        if self.sizing in ("fixed_dollars", "fixed_shares") and not self.fixed_amount:
            raise ValueError(f"{self.sizing} sizing needs fixed_amount")
        if self.leverage <= 0:
            raise ValueError("leverage must be positive")
        if self.position_size is None:
            self.position_size = self.leverage / self.max_positions

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "Strategy":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown strategy field(s): {', '.join(sorted(unknown))}")
        return cls(**d)

    def summary(self) -> str:
        side = {"long": "Buy", "short": "Short", "both": "Long/short"}[self.side]
        fill = {"close": "at the close of the signal day", "open": "at the open of the signal day",
                "next_open": "at the next day's open",
                "next_close": "at the next day's close"}[self.entry_fill]
        if self.entry_order != "market":
            fill = f"with a {self.entry_order} order at {self.entry_level} (valid {self.order_valid_bars} bar(s))"
        u = self.universe
        uni = self.universe_name or (", ".join(u) if len(u) <= 6 else f"{len(u)} tickers ({', '.join(u[:4])}, ...)")
        if self.universe_name:
            uni += f" ({len(u)} tickers{', point-in-time membership' if self.point_in_time else ', current members only'})"
        lines = [f"{side} {uni} {fill} when: {self.entry}"]
        if self.side == "both":
            lines.append(f"Short when: {self.short_entry}" + (" (signals reverse the position)" if self.reverse else ""))
        if self.pyramiding > 1:
            lines.append(f"Pyramiding: up to {self.pyramiding} entries per ticker")
        ex = []
        if self.hold_bars is not None:
            ex.append(f"after {self.hold_bars} bar(s) at the {self.hold_exit_fill}")
        if self.exit_when:
            ex.append(f"when {self.exit_when} ({'same close' if self.exit_when_fill == 'close' else 'next open'})")
        if self.stop_loss:
            ex.append(f"stop loss {self.stop_loss:.1%}")
        if self.stop_atr:
            ex.append(f"stop {self.stop_atr:g}x ATR({self.atr_period})")
        if self.take_profit:
            ex.append(f"take profit {self.take_profit:.1%}")
        if self.take_profit_atr:
            ex.append(f"take profit {self.take_profit_atr:g}x ATR({self.atr_period})")
        if self.trailing_stop:
            ex.append(f"trailing stop {self.trailing_stop:.1%}")
        if self.trailing_atr:
            ex.append(f"chandelier stop {self.trailing_atr:g}x ATR({self.atr_period}) from the high")
        for so in self.scale_out:
            ex.append(f"sell {so['fraction']:.0%} at +{so['at']:.1%}")
        if self.side == "both" and not ex:
            ex.append("opposite signal")
        lines.append("Exit: " + "; ".join(ex))
        sz = {
            "percent": lambda: f"{self.position_size:.0%} of equity each",
            "fixed_dollars": lambda: f"${self.fixed_amount:,.0f} each",
            "fixed_shares": lambda: f"{self.fixed_amount:g} shares each",
            "risk": lambda: f"risking {self.risk_per_trade:.2%} of equity to the stop",
            "volatility": lambda: f"sized to {self.target_vol:.0%} annualised volatility each",
        }[self.sizing]()
        lines.append(
            f"Sizing: ${self.capital:,.0f} start, up to {self.max_positions} position(s), {sz}"
            + (f", max {self.leverage:g}x gross exposure" if self.leverage != 1 else "")
            + (f", ranked by {'lowest' if self.rank_ascending else 'highest'} {self.rank_by}" if self.rank_by else "")
        )
        costs = []
        if self.commission:
            costs.append(f"${self.commission:g}/order")
        if self.commission_per_share:
            costs.append(f"${self.commission_per_share:g}/share")
        if self.commission_pct:
            costs.append(f"{self.commission_pct:.3%} of value")
        if self.slippage_bps:
            costs.append(f"{self.slippage_bps:g} bps slippage/side")
        if self.max_volume_pct:
            costs.append(f"orders capped at {self.max_volume_pct:.1%} of volume")
        if self.borrow_fee:
            costs.append(f"{self.borrow_fee:.2%}/yr borrow fee")
        lines.append("Costs: " + (", ".join(costs) if costs else "none"))
        cr = self.cash_rate
        lines.append("Cash: " + ("earns the 3-month T-bill rate" if cr == "tbill" else
                                 f"earns {float(cr):.2%}/yr" if cr else "earns nothing"))
        if self.start or self.end:
            lines.append(f"Period: {self.start or 'start of data'} to {self.end or 'latest'}")
        return "\n".join(lines)
