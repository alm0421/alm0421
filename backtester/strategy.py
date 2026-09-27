"""Strategy specification for rule-based (signal) strategies."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, fields
from typing import Literal

from .costs import COMMISSION_MODELS, broker_commission  # noqa: F401 - re-exported (older imports)

Fill = Literal["close", "open", "next_open", "next_close"]

TV_SIZING_NOTE = ("TradingView sizing: a percentage of equity is converted to a quantity on the bar the order is placed, "
                  "from that bar's equity and close, and the order fills at the next open (or at that close with "
                  "process_orders_on_close; TradingView's "
                  "strategy.percent_of_equity: \"position sizes will be calculated as a percentage of the available "
                  "equity\", \"subject to constraints due to the minimum tradable quantities for the symbol\" - TradingView "
                  "Help Center, Strategy properties). The quantity is rounded down to whole shares (the minimum quantity "
                  "of a stock or ETF is 1 share; say 'fractional shares' to allow fractions). An order the cash can't pay "
                  "for at the fill (a gap up) is skipped, not cut.")


def check_date(value, what: str = "start"):
    """A start / end date as given (None or '' for none), refused with a clear message when it is not a date:
    '2015', '2015-03' and '2015-03-02' (or anything else pandas reads as one date) are accepted."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    import pandas as pd
    text = str(value).strip()
    ok = bool(re.fullmatch(r"\d{4}(?:-\d{1,2}(?:-\d{1,2})?)?(?:[T ].*)?|[\w ,./-]*\d[\w ,./-]*", text))
    if ok:
        try:
            ts = pd.Timestamp(text)
            ok = ts is not pd.NaT and 1800 <= ts.year <= 2200
        except (ValueError, TypeError, OverflowError):
            ok = False
    if not ok:
        raise ValueError(f"The {what} date {text!r} is not a date. Write it as YYYY-MM-DD, e.g. 2015-01-02 (or a year, "
                         "e.g. 2015).")
    return value


def _fractional_market(t) -> bool:
    """Markets TradingView trades in fractions (crypto and FX pairs); stocks and ETFs trade in whole shares."""
    return isinstance(t, str) and bool(re.search(r"-(?:USD|USDT|EUR|BTC)$|=X$", t.upper()))


TV_NOTE = "TradingView-compatible mode: entries and rule exits with no timing stated fill at the next bar's open (TradingView's default, process_orders_on_close = false), and when a stop and a target are both touched on one bar, the one TradingView's broker emulator reaches first is filled (open -> high -> low -> close if the open is nearer the high, else open -> low -> high -> close)."



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
    hold_bars: int | None = None                 # exit exactly N bars after the entry bar, at hold_exit_fill, whatever
                                                 # the entry fill (0 = the close of an entry made at the open)
    hold_exit_fill: Literal["close", "open"] = "close"
    exit_when: str | None = None                 # expression; may use bars_held / entry_price / pnl /
                                                 # highest_since_entry / lowest_since_entry
    exit_when_fill: Literal["close", "open", "next_open"] = "close"   # "open": same bar's open (the rule must be
                                                 # known at the open); "next_open": checked at the close, sold next open
    stop_loss: float | None = None               # 0.05 = exit if price moves 5% against you
    stop_atr: float | None = None                # stop N x ATR(atr_period) from entry
    take_profit: float | None = None             # 0.10 = exit at +10%
    take_profit_atr: float | None = None
    trailing_stop: float | None = None           # 0.08 = exit 8% below highest high since entry
    trailing_atr: float | None = None            # chandelier: N x ATR below highest high since entry
    breakeven_after: float | None = None         # 0.02 = once the best price since entry is 2% in favour, a stop
                                                 # at the entry price (breakeven) is added; armed from the next bar
    atr_period: int = 14
    scale_out: list[dict] = field(default_factory=list)   # [{"at": 0.05, "fraction": 0.5}, ...]
    trail_after_scale_out: bool = False          # "sell half at +10% and trail the rest": the trailing / chandelier
                                                 # stop is armed only once the first scale-out has filled
    # stop / target at a price expression evaluated once, when the position opens, with the data known at the fill:
    # a fill at the close reads that bar; a fill at the open (same-day, next-day, limit/stop) reads the previous bar
    # unless the expression is open-safe. May use entry_price (the fill); target_level may also use stop_price.
    stop_level: str | None = None                # e.g. "low" (the entry bar's low), "lowest(low, 5)", "entry_price - 2 * atr(14)"
    target_level: str | None = None              # e.g. "highest(high, 20)", "entry_price + 2 * (entry_price - stop_price)"
    target_r: float | None = None                # R-multiple target: entry +/- target_r x (entry - initial stop)

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
    fractional_shares: bool | None = None        # None: fractional, except whole shares in TradingView-compatible mode
                                                 # for stocks and ETFs (TradingView's minimum quantity, 1 share)
    min_order: float = 1.0                       # orders worth less than this ($) are skipped (no dust trades)
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
    short_rebate_spread: float = 0.0025          # short sale proceeds earn the cash rate minus this (floored at 0)
    maintenance_margin: float = 0.25             # with leverage or shorts: if equity / gross exposure is below this
                                                 # at a close, positions are cut pro rata back to 1/leverage
    margin_account: Literal["reg_t", "portfolio"] = "reg_t"   # "reg_t": at most 2x overnight (US stocks, Regulation T);
                                                 # "portfolio": portfolio margin, up to 4x (maintenance below 1/leverage)
    commission_model: str | None = None          # None, "ibkr_fixed" or "ibkr_tiered" (added to the fields above)
    slippage_model: Literal["fixed", "volume"] = "fixed"   # "volume": adds spread_bps/2 + impact_bps*sqrt(shares/ADV20)
    spread_bps: float = 2.0                      # volume model: quoted bid-ask spread (half of it is paid per fill)
    impact_bps: float = 100.0                    # volume model: impact coefficient, in bps at 100% of ADV

    # TradingView-compatible mode: unstated entry timing = the next open (process_orders_on_close = false; applied
    # by the parser), and a stop and a target touched on the same bar are resolved with TradingView's OHLC path
    # (open -> high -> low -> close when the open is nearer the high, else open -> low -> high -> close) instead of
    # assuming the stop hit first
    tv_compat: bool = False
    dividends: bool | None = None                # credit cash dividends on the ex-date (charge them to shorts);
                                                 # None: yes, except in TradingView-compatible mode (price-only, as TV)

    # period
    start: str | None = None
    end: str | None = None

    name: str = ""
    description: str = ""
    notes: list[str] = field(default_factory=list)

    def validate(self) -> None:
        from .expr import compile_expr, open_safe, pine_to_rule
        if not self.universe:
            raise ValueError("universe is empty")
        for name in ("entry", "short_entry", "exit_when", "entry_level", "rank_by", "stop_level", "target_level"):
            v = getattr(self, name)
            if isinstance(v, str):   # TradingView spellings (close[1], ta.sma) -> the rule language
                setattr(self, name, pine_to_rule(v))
        for name in ("stop_level", "target_level"):
            v = getattr(self, name)
            if v is not None and not callable(v) and not (isinstance(v, str) and v.strip()):
                setattr(self, name, None)
        if not any([self.hold_bars is not None, self.exit_when, self.stop_loss, self.take_profit, self.trailing_stop,
                    self.stop_atr, self.take_profit_atr, self.trailing_atr, self.breakeven_after, self.side == "both",
                    self.stop_level, self.target_level, self.target_r]):
            raise ValueError("strategy needs at least one exit rule (hold_bars, exit_when, stop_loss, "
                             "take_profit, trailing_stop, stop_atr, trailing_atr, breakeven_after, stop_level, "
                             "target_level, target_r)")
        self._check_levels()
        if self.breakeven_after is not None and not self.breakeven_after > 0:
            raise ValueError("breakeven_after must be positive (0.02 = move the stop to the entry price after +2%)")
        if self.min_order is None or self.min_order < 0:
            raise ValueError("min_order cannot be negative")
        try:
            cap_ok = float(self.capital) > 0
        except (TypeError, ValueError):
            cap_ok = False
        if not cap_ok:
            raise ValueError(f"starting capital must be positive (got {self.capital!r})")
        self._apply_hold_compat()
        if self.hold_bars is not None:
            if self.hold_bars < 0:
                raise ValueError("hold_bars cannot be negative")
            if self.hold_bars == 0 and not (self.hold_exit_fill == "close" and self.enters_before_close()):
                raise ValueError("hold_bars 0 (exit at the close of the entry bar) needs an entry before the close: at the "
                                 "open, the next open, or a limit/stop order. Otherwise hold at least 1 bar.")
        for name in ("stop_loss", "trailing_stop", "take_profit"):
            v = getattr(self, name)
            if v is not None and v < 0:
                raise ValueError(f"{name} cannot be negative (write 0.05 for 5%)")
        for name, label in (("stop_loss", "stop loss"), ("trailing_stop", "trailing stop"), ("take_profit", "take profit"),
                            ("stop_atr", "ATR stop"), ("trailing_atr", "chandelier stop"),
                            ("take_profit_atr", "ATR take profit"), ("breakeven_after", "breakeven trigger")):
            v = getattr(self, name)
            if v is not None and not isinstance(v, bool) and isinstance(v, (int, float)) and v == 0:
                raise ValueError(f"A {label} of 0 would exit at the entry price at once (every trade would close on its "
                                 f"first check). Use a positive distance, e.g. 5% or 2 ATR, or leave it out.")
        if self.side in ("long", "both"):
            for name, label in (("stop_loss", "stop loss"), ("trailing_stop", "trailing stop")):
                v = getattr(self, name)
                if v is not None and v >= 1:
                    raise ValueError(f"A {v:.0%} {label} can never trigger on a long position (the price cannot fall "
                                     f"{v:.0%} or more). Use a value below 100%, e.g. 0.05 for 5%.")
        if self.max_positions < 1:
            raise ValueError("max_positions must be at least 1")
        if self.pyramiding < 1:
            raise ValueError("pyramiding must be at least 1")
        if self.side == "both" and not self.short_entry:
            raise ValueError("side 'both' needs a short_entry rule")
        for rule in (self.entry, self.short_entry, self.exit_when, self.entry_level, self.rank_by):
            if rule and not callable(rule):
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
        if self.sizing == "risk" and not (self.risk_per_trade and (self.stop_loss or self.stop_atr or self.trailing_stop
                                                                    or self.trailing_atr or self.stop_level)):
            raise ValueError("risk sizing needs risk_per_trade and a stop (stop_loss, stop_atr, stop_level, trailing_stop "
                             "or trailing_atr)")
        if self.sizing == "risk" and not (self.stop_loss or self.stop_atr or self.stop_level):
            what = (f"{self.trailing_stop:.0%} below the entry" if self.trailing_stop
                    else f"{self.trailing_atr:g} x ATR({self.atr_period}) from the entry")
            if not any(n.startswith("Risk sizing:") for n in self.notes):
                self.notes.append(f"Risk sizing: there is no fixed stop, so the trailing stop's starting distance ({what}) is "
                                  "the risk per share: a position loses about the stated risk if the trailing stop is hit "
                                  "before the price rises.")
        if self.sizing == "volatility" and not self.target_vol:
            raise ValueError("volatility sizing needs target_vol")
        if self.sizing in ("fixed_dollars", "fixed_shares") and not self.fixed_amount:
            raise ValueError(f"{self.sizing} sizing needs fixed_amount")
        if self.leverage <= 0:
            raise ValueError("leverage must be positive")
        if self.exit_when_fill not in ("close", "open", "next_open"):
            raise ValueError("exit_when_fill must be 'close', 'open' or 'next_open'")
        if self.exit_when_fill == "open" and self.exit_when and not open_safe(self.exit_when):
            raise ValueError("exit_when_fill 'open' needs an exit rule known at the open (e.g. gap, dow); this one uses "
                             "today's close/high/low. Use exit_when_fill 'next_open' to check it at the close and sell "
                             "at the next open.")
        for name, label in (("commission", "commission per order"), ("commission_per_share", "commission per share"),
                            ("commission_pct", "commission (% of value)"), ("slippage_bps", "slippage"),
                            ("borrow_fee", "borrow fee"), ("margin_rate", "margin rate"),
                            ("spread_bps", "bid-ask spread"), ("impact_bps", "market impact")):
            v = getattr(self, name)
            try:
                bad = v is not None and not float(v) >= 0
            except (TypeError, ValueError):
                bad = True
            if bad:
                raise ValueError(f"{label} cannot be negative (got {name}={v!r}): a negative cost would pay you for "
                                 "trading. Use 0 for none.")
        if not isinstance(self.tv_compat, bool):
            raise ValueError("tv_compat must be true or false")
        if self.dividends is None:
            self.dividends = not self.tv_compat          # TradingView's strategy tester credits no dividends
        if not isinstance(self.dividends, bool):
            raise ValueError("dividends must be true or false")
        if not isinstance(self.trail_after_scale_out, bool):
            raise ValueError("trail_after_scale_out must be true or false")
        if self.trail_after_scale_out:
            if not self.scale_out:
                raise ValueError("'trail the rest' (trail_after_scale_out) needs a scale-out first, e.g. 'sell half at +10% "
                                 "and trail the rest with an 8% trailing stop'.")
            if not (self.trailing_stop or self.trailing_atr):
                raise ValueError("'trail the rest' needs a trailing distance: e.g. 'trail the rest with an 8% trailing stop' "
                                 "(trailing_stop) or 'with a 3 ATR chandelier stop' (trailing_atr).")
        if self.fractional_shares is None:
            self.fractional_shares = not (self.tv_compat and not all(_fractional_market(t) for t in self.universe or []))
            if self.tv_compat and not self.fractional_shares and self.sizing == "percent" \
                    and not any(n.startswith("TradingView sizing:") for n in self.notes):
                self.notes.append(TV_SIZING_NOTE)
        if not isinstance(self.fractional_shares, bool):
            raise ValueError("fractional_shares must be true or false")
        if self.tv_compat and not any(n.startswith("TradingView-compatible mode") for n in self.notes):
            self.notes.append(TV_NOTE)
        self.notes = [n for n in self.notes if not n.startswith("TradingView-compatible returns:")]
        if self.tv_compat:
            no_cash = self.cash_rate in (None, 0, 0.0, False, "")
            cash = ("cash earns nothing (TradingView pays no interest; say 'with interest on cash' to add it)" if no_cash else
                    "cash earns the 3-month T-bill rate (TradingView pays none; say 'cash earns nothing' to match it)" if self.cash_rate == "tbill" else
                    f"cash earns {self.cash_rate}/yr (TradingView pays none; say 'cash earns nothing' to match it)")
            div = ("dividends are not credited (TradingView's strategies receive none, and its default charts are "
                   "split-adjusted, not dividend-adjusted; say 'with dividends' to credit them)" if not self.dividends else
                   "dividends are credited on the ex-date (TradingView credits none; say 'no dividends' to match it)")
            self.notes.append(f"TradingView-compatible returns: {cash}; {div}."
                              + (" These are price-only returns, as TradingView's strategy tester reports them."
                                 if no_cash and not self.dividends else ""))
        if self.commission_model not in COMMISSION_MODELS:
            raise ValueError(f"commission_model must be one of {sorted(k for k in COMMISSION_MODELS if k)} or null")
        if self.slippage_model not in ("fixed", "volume"):
            raise ValueError("slippage_model must be 'fixed' or 'volume'")
        if self.short_rebate_spread < 0:
            raise ValueError("short_rebate_spread cannot be negative")
        if not 0 <= self.maintenance_margin < 1:
            raise ValueError("maintenance_margin must be at least 0 and below 1")
        self._check_margin()
        if self.position_size is None:
            per = self.leverage / self.max_positions
            if (self.pyramiding or 1) > 1 and self.sizing == "percent":
                # the default size is the position's full share: each of the N allowed entries buys 1/N of it, so
                # every add-on has room (a first entry of the whole share would leave nothing to pyramid with)
                self.position_size = per / self.pyramiding
                self.notes = [n for n in self.notes if not n.startswith("Pyramiding:")]
                self.notes.append(f"Pyramiding: no size per entry given, so each position's {per:.0%} is split across its "
                                  f"{self.pyramiding} allowed entries: {self.position_size:.2%} of equity per entry.")
            else:
                self.position_size = per
        if self.sizing == "percent" and self.position_size > self.leverage + 1e-12:
            # "200% per position" at 1x would be silently capped to 100%: run what was asked instead
            need = float(self.position_size)
            self.notes = [n for n in self.notes if not n.startswith("Leverage:")]
            self.notes.append(f"Leverage: {need:.0%} per position needs {need:g}x leverage: using {need:g}x "
                              f"(raised from {self.leverage:g}x).")
            self.leverage = need
            self._check_margin(f"{need:.0%} per position needs {need:g}x leverage: ")

    LEVEL_VARS = {"stop_level": {"entry_price"}, "target_level": {"entry_price", "stop_price"}}

    def _check_levels(self) -> None:
        """stop_level / target_level: rule-language price expressions evaluated once, when a position opens;
        target_r: a multiple of the initial risk (entry - stop)."""
        from .expr import compile_expr, names_in
        for name, allowed in self.LEVEL_VARS.items():
            rule = getattr(self, name)
            if rule is None:
                continue
            if callable(rule) or not isinstance(rule, str):
                raise ValueError(f"{name} must be a rule-language expression (text), e.g. 'lowest(low, 5)'")
            compile_expr(rule)
            bad = names_in(rule) & ({"bars_held", "pnl", "highest_since_entry", "lowest_since_entry", "stop_price"} - allowed)
            if bad:
                raise ValueError(f"{name} {rule!r} uses {', '.join(sorted(bad))}, which is not known when the position opens "
                                 f"(it may use {' and '.join(sorted(allowed))}).")
        if self.target_r is not None:
            try:
                ok = float(self.target_r) > 0 and not isinstance(self.target_r, bool)
            except (TypeError, ValueError):
                ok = False
            if not ok:
                raise ValueError(f"target_r must be a positive multiple of the risk (got {self.target_r!r})")
            if not (self.stop_loss or self.stop_atr or self.stop_level or self.trailing_stop or self.trailing_atr):
                raise ValueError("An R-multiple target ('target 2R', 'take profit at 2 times the risk') needs a stop to "
                                 "measure the risk from: add e.g. 'stop at the low of the entry bar' or 'a 5% stop loss'.")

    MAX_LEVERAGE = {"reg_t": 2.0, "portfolio": 4.0}

    def _check_margin(self, why: str = "") -> None:
        """Leverage a broker would allow: Regulation T lends at most 2x overnight on US stocks (50% initial margin);
        a portfolio-margin account up to 4x. The maintenance margin must be below the initial margin (1/leverage):
        at or above it, the first close at or below the entry price is a margin call (4x with the default 25%
        maintenance cut positions on nearly every down day)."""
        if self.margin_account not in self.MAX_LEVERAGE:
            raise ValueError("margin_account must be 'reg_t' (at most 2x overnight) or 'portfolio' (portfolio margin, up to 4x)")
        cap = self.MAX_LEVERAGE[self.margin_account]
        if self.leverage > cap + 1e-12:
            if self.margin_account == "reg_t":
                raise ValueError(f"{why}{self.leverage:g}x leverage is more than Regulation T allows overnight on stocks "
                                 "(2x: 50% initial margin). With a portfolio-margin account up to 4x is possible: say "
                                 "'with portfolio margin' (margin_account 'portfolio') and a maintenance margin below "
                                 f"{1 / self.leverage:.0%}, e.g. 'a 15% maintenance margin'.")
            raise ValueError(f"{why}{self.leverage:g}x leverage is more than a portfolio-margin account allows (4x).")
        if self.leverage > 1 + 1e-12 and self.maintenance_margin >= 1 / self.leverage - 1e-12:
            raise ValueError(f"{why or 'T'}{'t' if why else ''}he {self.maintenance_margin:.0%} maintenance margin is not below the initial margin of "
                             f"{self.leverage:g}x leverage ({1 / self.leverage:.0%}): every close below the entry price would "
                             f"be a margin call. Use a maintenance margin below {1 / self.leverage:.0%} (e.g. 'a "
                             f"{max(1, int(100 / self.leverage * 0.6))}% maintenance margin'), less leverage, or 'no margin calls'.")

    def enters_before_close(self) -> bool:
        """True when entries fill before the close of their bar (at the open or intraday)."""
        return self.entry_order != "market" or self.entry_fill in ("open", "next_open")

    def _apply_hold_compat(self) -> None:
        """"cover at the close" with no holding period, after an entry at the open, means the close of the entry
        bar (hold 0). Older parsers wrote hold_bars 1 plus a note for it; read that as hold 0."""
        if (self.hold_bars == 1 and self.hold_exit_fill == "close" and self.enters_before_close()
                and any(n.startswith("No holding period stated: exiting at the first close after entry") for n in self.notes)):
            self.hold_bars = 0

    def hold_text(self) -> str:
        """The time exit in words: exactly N bars after the entry bar, at the configured fill."""
        n = self.hold_bars
        if n is None:
            return ""
        if n == 0:
            return f"at the {self.hold_exit_fill} of the entry bar"
        return f"at the {self.hold_exit_fill} {n} bar{'s' if n != 1 else ''} after the entry bar"

    def to_json(self) -> str:
        d = asdict(self)
        for k, v in d.items():
            if callable(v):
                d[k] = f"<python function {getattr(v, '__name__', 'rule')}>"
        return json.dumps(d, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "Strategy":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown strategy field(s): {', '.join(sorted(unknown))}")
        if d.get("tv_compat") is True and "cash_rate" not in d:
            d = {**d, "cash_rate": None}   # TradingView-compatible mode: cash earns nothing unless the spec says so
        return cls(**d)

    def summary(self) -> str:
        def f(v, spec: str, default: str = "?") -> str:
            """Format a number, tolerating None / strings from hand-written specs."""
            try:
                return format(float(v), spec)
            except (TypeError, ValueError):
                return default if v in (None, "") else str(v)

        side = {"long": "Buy", "short": "Short", "both": "Long/short"}.get(self.side, str(self.side))
        fill = {"close": "at the close of the signal day", "open": "at the open of the signal day",
                "next_open": "at the next day's open",
                "next_close": "at the next day's close"}.get(self.entry_fill, f"at {self.entry_fill}")
        if self.entry_order != "market":
            fill = f"with a {self.entry_order} order at {self.entry_level} (valid {self.order_valid_bars} bar(s))"
        u = list(self.universe or [])
        uni = self.universe_name or (", ".join(u) if len(u) <= 6 else f"{len(u)} tickers ({', '.join(u[:4])}, ...)")
        if self.universe_name:
            pit_txt = ", point-in-time membership" if self.point_in_time else ", today's members (survivorship-biased by construction)"
            uni += f" ({len(u)} tickers{pit_txt})"
        lines = [f"{side} {uni} {fill} when: {self.entry}"]
        if self.side == "both":
            lines.append(f"Short when: {self.short_entry}" + (" (signals reverse the position)" if self.reverse else ""))
        if (self.pyramiding or 1) > 1:
            lines.append(f"Pyramiding: up to {self.pyramiding} entries per ticker")
        ex = []
        if self.hold_bars is not None:
            ex.append(self.hold_text())
        if self.exit_when:
            when = {"close": "same close", "open": "same open", "next_open": "next open"}[self.exit_when_fill]
            ex.append(f"when {self.exit_when} ({when})")
        if self.stop_loss:
            ex.append(f"stop loss {f(self.stop_loss, '.1%')}")
        if self.stop_atr:
            ex.append(f"stop {f(self.stop_atr, 'g')}x ATR({self.atr_period})")
        if self.take_profit:
            ex.append(f"take profit {f(self.take_profit, '.1%')}")
        if self.take_profit_atr:
            ex.append(f"take profit {f(self.take_profit_atr, 'g')}x ATR({self.atr_period})")
        if self.trailing_stop:
            ex.append(f"trailing stop {f(self.trailing_stop, '.1%')}")
        if self.trailing_atr:
            extreme = {"long": "the high", "short": "the low"}.get(self.side, "the high (longs) / low (shorts)")
            ex.append(f"chandelier stop {f(self.trailing_atr, 'g')}x ATR({self.atr_period}) from {extreme}")
        if self.breakeven_after:
            ex.append(f"stop moves to breakeven (the entry price) after +{f(self.breakeven_after, '.1%')}")
        if self.stop_level:
            ex.append(f"stop at {self.stop_level} (fixed when the position opens)")
        if self.target_level:
            ex.append(f"target at {self.target_level} (fixed when the position opens)")
        if self.target_r:
            ex.append(f"target {f(self.target_r, 'g')}R ({f(self.target_r, 'g')} x the initial risk, entry - stop)")
        for so in self.scale_out or []:
            ex.append(f"sell {f(so.get('fraction'), '.0%')} at +{f(so.get('at'), '.1%')}")
        if self.trail_after_scale_out:
            ex.append("the trailing stop applies to the rest, from the first scale-out on")
        if self.side == "both" and not ex:
            ex.append("opposite signal")
        lines.append("Exit: " + ("; ".join(ex) if ex else "none"))
        lev = self.leverage if self.leverage not in (None, "") else 1.0
        size = self.position_size
        if size in (None, ""):
            try:
                size = float(lev) / max(int(self.max_positions or 1), 1)
            except (TypeError, ValueError):
                size = None
        sz = {
            "percent": lambda: f"{f(size, '.0%')} of equity each",
            "fixed_dollars": lambda: f"${f(self.fixed_amount, ',.0f')} each",
            "fixed_shares": lambda: f"{f(self.fixed_amount, 'g')} shares each",
            "risk": lambda: f"risking {f(self.risk_per_trade, '.2%')} of equity to the stop",
            "volatility": lambda: f"sized to {f(self.target_vol, '.0%')} annualised volatility each",
        }.get(self.sizing, lambda: str(self.sizing))()
        lines.append(
            f"Sizing: ${f(self.capital, ',.0f')} start, up to {self.max_positions} position(s), {sz}"
            + (f", max {f(lev, 'g')}x gross exposure" if lev != 1 else "")
            + (" (raised automatically: the position size needs it)" if any(n.startswith("Leverage:") for n in self.notes or []) else "")
            + (f", ranked by {'lowest' if self.rank_ascending else 'highest'} {self.rank_by}" if self.rank_by else "")
        )
        costs = []
        if self.commission:
            costs.append(f"${f(self.commission, 'g')}/order")
        if self.commission_per_share:
            costs.append(f"${f(self.commission_per_share, 'g')}/share")
        if self.commission_pct:
            costs.append(f"{f(self.commission_pct, '.3%')} of value")
        if self.commission_model:
            costs.append({"ibkr_fixed": "IBKR fixed ($0.005/share, min $1, max 1%)",
                          "ibkr_tiered": "IBKR tiered (~$0.0037/share incl. fees, min $0.35, max 1%)"}.get(self.commission_model, str(self.commission_model)))
        if self.slippage_bps:
            costs.append(f"{f(self.slippage_bps, 'g')} bps slippage/side")
        if self.slippage_model == "volume":
            costs.append(f"volume slippage ({f((self.spread_bps or 0) / 2, 'g')} bps + {f(self.impact_bps, 'g')} bps x sqrt(shares/ADV20))")
        if self.max_volume_pct:
            costs.append(f"orders capped at {f(self.max_volume_pct, '.1%')} of volume")
        if self.margin_rate:
            costs.append(f"borrowing at the T-bill rate + {f(self.margin_rate, '.2%')}")
        if self.borrow_fee:
            costs.append(f"{f(self.borrow_fee, '.2%')}/yr borrow fee")
        if self.side != "long" and self.short_rebate_spread and self.cash_rate not in (None, 0, 0.0, False, ""):
            costs.append(f"short proceeds earn the cash rate less {f(self.short_rebate_spread, '.2%')}")
        if (self.side != "long" or (self.leverage or 1) > 1) and self.maintenance_margin:
            costs.append(f"{f(self.maintenance_margin, '.0%')} maintenance margin"
                         + (" (portfolio margin account)" if self.margin_account == "portfolio" else ""))
        lines.append("Costs: " + (", ".join(costs) if costs else "none"))
        cr = self.cash_rate
        lines.append("Cash: " + ("earns the 3-month T-bill rate" if cr == "tbill" else
                                 f"earns {f(cr, '.2%')}/yr" if cr else "earns nothing")
                     + ("; dividends are not credited (price-only)" if self.dividends is False else ""))
        if self.start or self.end:
            lines.append(f"Period: {self.start or 'start of data'} to {self.end or 'latest'}")
        return "\n".join(lines)
