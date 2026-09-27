"""Allocation ("portfolio") strategies: target weights, rebalancing, regime switches, rotations.

A portfolio is a tree of nodes (JSON-compatible dicts), in the spirit of Composer symphonies and
Portfolio Visualizer tactical models:

  {"asset": "SPY"}                                   hold one ticker
  {"cash": true}                                     hold cash (earns the T-bill rate)
  {"weights": "specified", "w": [0.6, 0.4], "children": [...]}
  {"weights": "equal" | "inverse_vol" | "market_cap", "lookback": 20, "children": [...]}
  {"if": "close > sma(close, 200)", "on": "SPY", "then": node, "else": node}
  {"filter": {"select": "top" | "bottom", "n": 5, "by": "ret(63)",
              "require": "ret(252) > 0", "weights": "equal" | "inverse_vol" | "market_cap"},
   "universe": "NDX" | ["QQQ", "SPY", ...] | "children", "children": [...], "fallback": node}

A filter's children and a weights node's children may be any nodes (groups, if-nodes, nested filters).
Ranking metrics and non-equal weightings of a child that is not a single asset use the child's
synthetic daily NAV: the sub-portfolio simulated on its own, evaluated every day, no costs (as
Composer does for sub-symphonies). See `_Evaluator.nav`.

Targets are evaluated on each rebalance date using that day's close and traded at the close (or
the next open). Cash flows, dividends and interest are applied daily.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, fields
from typing import Literal

import numpy as np
import pandas as pd

from . import calendar as _cal
from . import costs as _costs
from . import data, expr
from . import margin as _margin
from .engine import Result, _daily_rate

FREQS = ("daily", "weekly", "monthly", "quarterly", "semiannual", "yearly", "none")
# plus "every_N_days" (every N trading days), "every_N_weeks" / "every_N_months" (every Nth week / month end)
_EVERY = re.compile(r"every_(\d+)_(days|weeks|months)")


def every_n(freq) -> tuple[int, str] | None:
    """'every_2_days' -> (2, 'days'); None for the named frequencies."""
    m = _EVERY.fullmatch(str(freq))
    return (int(m.group(1)), m.group(2)) if m else None


# plus "yearly_M": once a year, at the last trading day of month M (1-12): "rebalance every year in June" = yearly_6
_ANNUAL_IN = re.compile(r"yearly_(\d{1,2})")
MONTH_NAMES = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
               "November", "December")


def annual_month(freq) -> int | None:
    """'yearly_6' -> 6 (rebalanced at the end of June each year); None otherwise."""
    m = _ANNUAL_IN.fullmatch(str(freq))
    return int(m.group(1)) if m and 1 <= int(m.group(1)) <= 12 else None
FLOW_FREQS = ("monthly", "quarterly", "semiannual", "yearly")
NAV_WARMUP = 504   # trading days simulated before the start for the synthetic NAVs of groups
_MONTH_RET = re.compile(r"\b(?:t?ret|tbill_ret)\((?:[^()]*?,\s*)?(\d+)\s*\)")


_SKIP_RET = re.compile(r"\bref\(\s*(?:t?ret|tbill_ret)\((?:[^()]*?,\s*)?(\d+)\s*\)\s*,\s*(\d+)\s*\)")


def month_lookback_line(p) -> str | None:
    """How the rules' month lookbacks are measured, for the interpretation and the report's notes; None when no
    rule has a return over whole months (a multiple of 21 sessions)."""
    tree = json.dumps(getattr(p, "tree", {}), default=str)
    M = expr.MONTH_BARS
    skips = sorted({(int(a), int(b)) for a, b in _SKIP_RET.findall(tree) if int(a) % M == 0 and int(b) % M == 0})
    plain = _SKIP_RET.sub(" ", tree)
    ns = {int(n) for n in _MONTH_RET.findall(plain)}
    ns = sorted(n for n in ns if n % M == 0)
    if not ns and not skips:
        return None
    ms = ", ".join(f"{n // M}" for n in ns)
    cal = getattr(p, "month_lookbacks", "trading") == "calendar"
    sk = ", ".join(f"{(a + b) // M}-{b // M}" for a, b in skips)
    if cal:
        parts = []
        if ns:
            parts.append(f"Month lookbacks ({ms} months): calendar months, month-end to month-end (from the last "
                         "completed month-end), as Portfolio Visualizer and Antonacci measure them")
        if skips:
            parts.append(f"{sk} month momentum on month-end prices: " + "; ".join(
                f"the month-end price {b // M} month{'s' if b // M > 1 else ''} ago / the month-end price "
                f"{(a + b) // M} months ago - 1" for a, b in skips) + " (completed month-ends, counted from the last one)")
        return "; ".join(parts)
    if skips and not ns:
        return (f"{sk} month momentum: {M} trading days a month (the return over {', '.join(str(a) for a, _ in skips)} "
                f"sessions ending {', '.join(str(b) for _, b in skips)} sessions ago), not month-end to month-end; say "
                "'using calendar months' to measure it on month-end prices as Portfolio Visualizer does")
    return (f"Month lookbacks ({', '.join(filter(None, [ms, sk]))} months): {M} trading days a month (12 months = 252 "
            "sessions), not month-end to month-end; say 'using calendar months' (or month_lookbacks: \"calendar\" in a "
            "spec) to measure them month-end to month-end as Portfolio Visualizer does")


@dataclass
class Portfolio:
    tree: dict
    rebalance: Literal["daily", "weekly", "monthly", "quarterly", "semiannual", "yearly", "none"] = "monthly"
    drift_band: float | None = None              # also rebalance when any weight drifts this far
    drift_band_relative: float | None = None     # ... or drifts this fraction of its own target (0.25: 40% -> 30%/50%)
    min_trade: float = 0.0                       # skip trades smaller than this fraction of equity
    fill: Literal["close", "next_open"] = "close"
    capital: float = 10_000.0
    start: str | None = None
    end: str | None = None
    slippage_bps: float = 0.0
    commission: float = 0.0                      # $ per order
    commission_pct: float = 0.0
    commission_model: str | None = None          # None, "ibkr_fixed" or "ibkr_tiered" (added to the fields above)
    slippage_model: Literal["fixed", "volume"] = "fixed"   # "volume": adds spread_bps/2 + impact_bps*sqrt(shares/ADV20)
    spread_bps: float = 2.0                      # volume model: quoted bid-ask spread (half of it is paid per fill)
    impact_bps: float = 100.0                    # volume model: impact coefficient, in bps at 100% of ADV
    cash_rate: str | float | None = "tbill"
    reinvest_dividends: bool = True
    fractional_shares: bool = True
    point_in_time: bool = True
    # cash flows
    contribution: float = 0.0                    # $ added each period
    contribution_freq: Literal["monthly", "quarterly", "semiannual", "yearly"] = "monthly"
    withdrawal: float = 0.0                      # $ withdrawn each period
    withdrawal_pct: float = 0.0                  # fraction of the balance withdrawn each period
    withdrawal_freq: Literal["monthly", "quarterly", "semiannual", "yearly"] = "yearly"
    inflation_adjust: bool = False               # grow $ contributions/withdrawals with CPI
    # flow windows: a year (2030 = 1 Jan 2030), "YYYY-MM-DD", or N < 1900 = year N of the backtest
    # (contribution_end=20: the first 20 years; withdrawal_start=21: from the 21st year on). Inclusive.
    contribution_start: int | str | None = None
    contribution_end: int | str | None = None
    withdrawal_start: int | str | None = None
    withdrawal_end: int | str | None = None
    contribution_growth: float = 0.0             # annual step-up of $ contributions, counted from their start
    withdrawal_growth: float = 0.0               # annual step-up of $ withdrawals (on top of any CPI indexing)
    # per-flow CPI indexing (None: follow inflation_adjust), and the dollars a real amount is expressed in: None =
    # dollars of the backtest's first day, "flow" = of the flow's first payment, "YYYY" = that year's average CPI
    contribution_inflation: bool | None = None
    withdrawal_inflation: bool | None = None
    contribution_dollars: str | None = None
    withdrawal_dollars: str | None = None
    leverage: float = 1.0                        # scale every target weight; the excess is borrowed
    margin_rate: float = 0.0                     # extra annual rate paid on borrowed cash (above T-bills)
    maintenance_margin: float = 0.25             # with borrowing or shorts: equity below this x gross exposure (x the
                                                 # leverage factor for leveraged ETFs) at a close -> margin call
    # listed share classes of one company (GOOG/GOOGL, FOX/FOXA, ...): "company" = a ranking counts the company once
    # and holds its more liquid class; "separate" = each class is ranked as its own name
    share_classes: Literal["company", "separate"] = "company"
    margin_account: Literal["reg_t", "portfolio"] = "reg_t"   # the same margin model as signal strategies
                                                 # (backtester/margin.py): Reg T at most 2x, portfolio margin 4x
    expense_ratio: float = 0.0                   # annual fee on invested assets, charged daily
    short_rebate_spread: float = 0.0025          # short sale proceeds earn the cash rate minus this (floored at 0)
    borrow_fee: float = 0.0                      # annual fee on the market value of short positions, charged daily
    # statistics start once every asset a filter / ranking / weighting measures has its full lookback ("all"),
    # or once enough of them have one to fill the filter's slots ("first")
    warmup: Literal["all", "first"] = "all"
    # (the portfolio starts trading on that day - nothing is bought during the warm-up - so the statistics start
    # with the starting capital)
    # indicator prices: "adjusted" = open/high/low/close on a total-return basis, dividends reinvested (Composer and
    # Portfolio Visualizer); "quoted" = prices as quoted (TradingView). Trading and valuation always use quoted
    # prices plus cash dividends; only the rules' indicators change.
    price_basis: Literal["adjusted", "quoted"] = "adjusted"
    # volatility targeting: at each rebalance, scale every risky weight by target_vol / the realised volatility of
    # the target mix over target_vol_lookback days, capped at `leverage` (1 = never borrow); the rest is cash
    target_vol: float | None = None
    target_vol_lookback: int = 60
    # "N month" lookbacks (ret / tret / tbill_ret over a multiple of 21 sessions): "trading" = that many sessions
    # (12 months = 252), "calendar" = month-end to month-end over N calendar months (Portfolio Visualizer, Antonacci)
    month_lookbacks: Literal["trading", "calendar"] = "trading"
    # opt-in ("with proxies before inception"): {ticker: proxy}; before the ticker's first date the proxy's total return
    # stands in for it (e.g. {"TIPSIM": "IEFSIM"}: intermediate Treasuries before TIPS exist), so a mix can start
    # before one sleeve's history does. Default None: the backtest starts when every holding has data.
    proxies: dict | None = None
    benchmark: str | dict | None = None          # comparison ticker for alpha/beta (default SPY), or a blend:
                                                 # "60 SPY 40 AGG" / {"SPY": 0.6, "AGG": 0.4} (rebalanced monthly)
    name: str = ""
    description: str = ""
    notes: list[str] = field(default_factory=list)
    kind: str = "allocation"

    def _check_margin(self) -> None:
        """The signal engine's margin model (backtester/margin.py): Regulation T lends at most 2x gross overnight,
        portfolio margin 4x; a leveraged ETF needs the maintenance margin times its leverage factor (FINRA 4210:
        75% for a 3x fund), to open as well as to hold; with borrowing or shorts, the maintenance requirement at the
        largest target exposure must be below the equity. Leveraged ETFs are checked through the tree's weights."""
        from . import margin as _m
        _m.check_account(self.margin_account)
        mm, acct, lev = float(self.maintenance_margin), self.margin_account, float(self.leverage)
        init = lev * max_requirement(self.tree, lambda t: _m.initial(t, mm, acct) if t else 1 / _m.MAX_LEVERAGE[acct])
        maint = lev * max_requirement(self.tree, lambda t: _m.maintenance(t, mm) if t else mm)
        gross = max_gross(self.tree) * lev
        worst = max(tickers_in(self.tree, index_universes=False), key=_m.factor, default=None)
        if init > 1 + 1e-9:
            if not (worst and _m.factor(worst) > 1):
                raise ValueError(f"The portfolio's gross exposure can reach {gross:.3g}x its equity (the sum of the absolute "
                                 f"weights, longs plus shorts{', times the leverage' if lev != 1 else ''}), more than "
                                 + ("Regulation T allows overnight (2x: 50% initial margin). With a portfolio-margin "
                                    "account up to 4x is possible: say 'with portfolio margin' (margin_account "
                                    "'portfolio') and a maintenance margin below 1/leverage, e.g. 'a 15% maintenance margin'."
                                    if acct == "reg_t" else "a portfolio-margin account allows (4x)."))
            _m.refuse("", init, maint, lev, mm, acct, worst)
        borrowing = gross > 1 + 1e-9 or _has_short(self.tree)
        if borrowing and mm and maint >= 1 - 1e-9:
            if worst and _m.factor(worst) > 1:
                _m.refuse("", init, maint, lev, mm, acct, worst)
            raise ValueError(f"The portfolio's gross exposure can reach {gross:.3g}x its equity, and the {mm:.0%} "
                             f"maintenance_margin allows less than {1 / mm:.3g}x: every close below the entry would be a "
                             "margin call. Use smaller weights or less leverage, or a lower maintenance_margin (0 turns "
                             "margin calls off).")
        note = _m.leveraged_note(tickers_in(self.tree, index_universes=False)) if borrowing else None
        if note and note not in self.notes:
            self.notes.append(note)

    def validate(self) -> None:
        ev = every_n(self.rebalance)
        if self.rebalance not in FREQS and not (ev and ev[0] >= 1) and not annual_month(self.rebalance):
            raise ValueError(f"rebalance must be one of {FREQS}, or every_N_days / every_N_weeks / every_N_months "
                             "(e.g. every_2_days), or yearly_M (once a year at the end of month M, e.g. yearly_6 = June)")
        validate_node(self.tree)
        check_tree(self)
        if not (0 < self.leverage <= 10):
            raise ValueError("leverage must be between 0 and 10")
        if self.commission_model not in _costs.COMMISSION_MODELS:
            raise ValueError(f"commission_model must be one of {sorted(k for k in _costs.COMMISSION_MODELS if k)} or null")
        if self.slippage_model not in _costs.SLIPPAGE_MODELS:
            raise ValueError("slippage_model must be 'fixed' or 'volume'")
        if not 0 <= self.maintenance_margin < 1:
            raise ValueError("maintenance_margin must be at least 0 and below 1")
        if self.month_lookbacks not in ("trading", "calendar"):
            raise ValueError("month_lookbacks must be 'trading' or 'calendar'")
        if self.price_basis not in ("adjusted", "quoted"):
            raise ValueError("price_basis must be 'adjusted' (total-return prices, as Composer and Portfolio Visualizer) "
                             "or 'quoted' (prices as quoted, as TradingView)")
        if self.target_vol is not None:
            if not 0 < float(self.target_vol) < 5:
                raise ValueError("target_vol is an annual fraction above 0 (0.10 = 10% a year)")
            if int(self.target_vol_lookback) < 5:
                raise ValueError("target_vol_lookback must be at least 5 days")
        self._check_margin()
        if self.share_classes not in ("company", "separate"):
            raise ValueError("share_classes must be 'company' (one slot per company) or 'separate'")
        if self.benchmark not in (None, ""):
            from . import metrics
            self.benchmark = metrics.benchmark_label(self.benchmark)
        if self.proxies not in (None, {}):
            if not isinstance(self.proxies, dict) or not all(isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip()
                                                             for k, v in self.proxies.items()):
                raise ValueError("proxies is {ticker: proxy ticker}, e.g. {\"TIPSIM\": \"IEFSIM\"}")
            self.proxies = {data.canonical(k): data.canonical(v) for k, v in self.proxies.items()}
            for k, v in self.proxies.items():
                if k == v:
                    raise ValueError(f"proxies: {k} cannot stand in for itself")
        for f in ("contribution_freq", "withdrawal_freq"):
            if getattr(self, f) not in FLOW_FREQS:
                raise ValueError(f"{f} must be one of {FLOW_FREQS}")
        for f in ("drift_band", "drift_band_relative"):
            v = getattr(self, f)
            if v is not None and not v > 0:
                raise ValueError(f"{f} must be above 0 (or None)")
        for f in ("contribution_start", "contribution_end", "withdrawal_start", "withdrawal_end"):
            _flow_bound(getattr(self, f), pd.Timestamp("2000-01-03"), f.endswith("end"), f)
        for f in ("contribution_growth", "withdrawal_growth"):
            if not -1 < float(getattr(self, f)) < 10:
                raise ValueError(f"{f} is an annual fraction (0.03 = 3% a year)")
        if self.capital < 0:
            raise ValueError("the starting capital cannot be negative")
        if self.capital <= 0 and self.contribution <= 0:
            raise ValueError("Starting with $0 needs contributions to invest, e.g. 'add $500 a month'; with no "
                             "starting capital and no contributions nothing is ever invested.")
        if self.capital <= 0 and self.contribution_start not in (None, 1, "1"):
            raise ValueError("Starting with $0 needs the contributions to begin on the first day (the account would be "
                             "empty until then): drop the contribution start, or start the backtest when they begin.")
        for f in ("contribution_dollars", "withdrawal_dollars"):
            v = getattr(self, f)
            if v not in (None, "start", "flow") and not re.fullmatch(r"(18|19|20)\d\d", str(v)):
                raise ValueError(f"{f} is None (dollars of the first day), 'flow' (of the flow's first payment) or a year")
        if self.warmup not in ("all", "first"):
            raise ValueError("warmup must be 'all' (stats start when every ranked asset has its lookback) or 'first'")
        for f, label in (("slippage_bps", "slippage"), ("commission", "commission per order"),
                         ("commission_pct", "commission (% of value)"), ("expense_ratio", "expense ratio"),
                         ("borrow_fee", "borrow fee"), ("margin_rate", "margin rate")):
            v = getattr(self, f)
            try:
                bad = v is not None and not float(v) >= 0
            except (TypeError, ValueError):
                bad = True
            if bad:
                raise ValueError(f"{label} cannot be negative (got {f}={v!r}): a negative cost would pay you for "
                                 "trading or holding. Use 0 for none.")
        for f in ("short_rebate_spread", "borrow_fee"):
            if not 0 <= float(getattr(self, f)) < 1:
                raise ValueError(f"{f} is an annual fraction between 0 and 1 (0.01 = 1% a year)")
        if self.start and self.end and pd.Timestamp(self.start) >= pd.Timestamp(self.end):
            raise ValueError(f"The period is reversed or empty: it starts on {self.start} but ends on {self.end}.")

    def flow_inflation(self, kind: str) -> bool:
        """Is this flow ('contribution' / 'withdrawal') indexed to CPI?"""
        v = getattr(self, f"{kind}_inflation", None)
        return bool(self.inflation_adjust if v is None else v)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=lambda o: f"<python function {getattr(o, '__name__', 'custom')}>")

    @classmethod
    def from_dict(cls, d: dict) -> "Portfolio":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown portfolio field(s): {', '.join(sorted(unknown))}")
        return cls(**d)

    @property
    def universe(self) -> list[str]:
        return tickers_in(self.tree)

    def summary(self) -> str:
        lines = ["Portfolio:"] + ["  " + l for l in describe(self.tree)]
        if _has_ndx(self.tree):
            lines.append("  Nasdaq-100 universe: " + ("point-in-time membership" if self.point_in_time else
                                                      "today's members (survivorship-biased by construction)"))
        rb = {"none": "never rebalanced (buy and hold)", "daily": "re-evaluated and rebalanced daily",
              "semiannual": "re-evaluated and rebalanced every six months (end of June and December)"}.get(
            self.rebalance, f"re-evaluated and rebalanced {self.rebalance}")
        if annual_month(self.rebalance):
            rb = (f"re-evaluated and rebalanced yearly, at the end of {MONTH_NAMES[annual_month(self.rebalance) - 1]} "
                  "(its last trading day)")
        if every_n(self.rebalance):
            n_, u_ = every_n(self.rebalance)
            rb = (f"re-evaluated and rebalanced every {n_} trading days" if u_ == "days" else
                  f"re-evaluated and rebalanced at every "
                  f"{n_}{'nd' if n_ % 10 == 2 and n_ % 100 != 12 else 'rd' if n_ % 10 == 3 and n_ % 100 != 13 else 'th'} "
                  f"{u_[:-1]}-end" if n_ > 1 else f"re-evaluated and rebalanced {u_[:-1]}ly")
        bands = []
        if self.drift_band:
            bands.append(f"drifts {fmt_weight(self.drift_band)} from its target")
        if self.drift_band_relative:
            bands.append(f"drifts by {fmt_weight(self.drift_band_relative)} of its own target weight")
        dyn = _is_dynamic(self.tree)
        what = ("the rules pick different holdings" if _continuous_weights(self.tree) or self.target_vol
                else "the rules change the target allocation (a different branch, selection or weights)")
        if bands:
            b = " or ".join(bands)
            if self.rebalance == "none":
                rb = (f"no calendar schedule (threshold rebalancing): the rules are re-evaluated at every close and it "
                      f"trades only when {what} or a holding {b}" if dyn else
                      f"no calendar schedule (threshold rebalancing): rebalanced only when a holding {b}")
            else:
                rb += (f", trading only when the target changes or a holding {b}" if self.rebalance == "daily"
                       else f", and also whenever a holding {b} in between (back to the target set at the last "
                            "scheduled rebalance)")
        elif self.rebalance == "none" and dyn:
            rb = (f"never rebalanced back to its weights, but the rules are re-evaluated at every close and it trades "
                  f"when {what} (as Composer follows its logic)")
        lines.append(f"Rebalancing: {rb}; trades at the {'close' if self.fill == 'close' else 'next open'}")
        cf = []

        def real(kind: str) -> str:
            if not self.flow_inflation(kind):
                return ""
            d = getattr(self, f"{kind}_dollars", None)
            base = ("dollars of its first payment" if d == "flow" else f"{d} dollars" if d not in (None, "start") else
                    f"{str(self.start)[:4]} dollars" if self.start else "dollars of the first day")
            return f" in {base}, rising with inflation (CPI)"
        if self.contribution:
            cf.append(f"add ${self.contribution:,.0f} {self.contribution_freq}" + real("contribution")
                      + _window_text(self.contribution_start, self.contribution_end)
                      + (f", growing {self.contribution_growth:.1%} a year" if self.contribution_growth else ""))
        if self.withdrawal:
            cf.append(f"withdraw ${self.withdrawal:,.0f} {self.withdrawal_freq}" + real("withdrawal")
                      + _window_text(self.withdrawal_start, self.withdrawal_end)
                      + (f", growing {self.withdrawal_growth:.1%} a year" if self.withdrawal_growth else ""))
        if self.withdrawal_pct:
            per_year = {"monthly": 12, "quarterly": 4, "semiannual": 2}.get(self.withdrawal_freq, 1)
            pc = lambda x: f"{round(x * 100, 4):g}%"  # noqa: E731
            cf.append((f"withdraw {pc(self.withdrawal_pct * per_year)}/yr of the balance ({pc(self.withdrawal_pct)} "
                       f"{self.withdrawal_freq})" if per_year > 1 else
                       f"withdraw {pc(self.withdrawal_pct)} of the balance yearly")
                      + ("" if self.withdrawal else _window_text(self.withdrawal_start, self.withdrawal_end)))
        if self.withdrawal:
            cf.append("a withdrawal is capped at the balance: the account stops at $0 if it runs out")
        lines.append(f"Money: ${self.capital:,.0f} start" + ("; " + ", ".join(cf) if cf else "")
                     + ("; dividends reinvested" if self.reinvest_dividends else "; dividends kept as cash"))
        costs = []
        if self.commission:
            costs.append(f"${self.commission:g}/order")
        if self.commission_pct:
            costs.append(f"{self.commission_pct:.3%} of value")
        if self.commission_model:
            costs.append({"ibkr_fixed": "IBKR fixed ($0.005/share, min $1, max 1%)",
                          "ibkr_tiered": "IBKR tiered (~$0.0037/share incl. fees, min $0.35, max 1%)"}.get(
                              self.commission_model, str(self.commission_model)))
        if self.slippage_bps:
            costs.append(f"{self.slippage_bps:g} bps slippage/side")
        if self.slippage_model == "volume":
            costs.append(f"volume slippage ({self.spread_bps / 2:g} bps + {self.impact_bps:g} bps x sqrt(shares/ADV20))")
        if self.expense_ratio:
            costs.append(f"{self.expense_ratio:.2%}/yr expense ratio"
                         + (" (charged daily on the gross invested assets, including the borrowed or shorted part, not "
                            "only on the equity)" if self.leverage > 1 or _has_short(self.tree) else ""))
        if self.leverage != 1:
            costs.append(f"{self.leverage:g}x leverage, borrowing at the T-bill rate"
                         + (f" + {self.margin_rate:.2%}" if self.margin_rate else ""))
        elif self.margin_rate:
            costs.append(f"margin rate T-bills + {self.margin_rate:.2%} on any borrowing")
        if _has_short(self.tree):
            costs.append(f"short proceeds earn the cash rate less {self.short_rebate_spread:.2%}/yr"
                         + (f", {self.borrow_fee:.2%}/yr borrow fee on shorts" if self.borrow_fee else ""))
        if self.leverage > 1 or _has_short(self.tree):
            costs.append(f"{self.maintenance_margin:.0%} maintenance margin (margin calls cut positions at the close)"
                         if self.maintenance_margin else "no margin calls")
        lines.append("Costs: " + (", ".join(costs) if costs else "none"))
        if self.target_vol:
            cap = self.leverage
            lines.append(f"Volatility target: {float(self.target_vol):.1%} a year: at each rebalance every holding is scaled "
                         f"by the target over the {int(self.target_vol_lookback)}-day realised volatility of the target mix, "
                         + (f"up to {cap:g}x (borrowing above 1x)" if cap > 1 else "never above 100% invested")
                         + "; the rest is held in cash")
        if _has_rules(self.tree):
            lines.append("Indicators: " + ("computed on total-return prices (dividends reinvested), as Composer and "
                                           "Portfolio Visualizer do" if self.price_basis == "adjusted" else
                                           "computed on prices as quoted (not adjusted for dividends), as TradingView does")
                         + "; trades and valuation use quoted prices plus cash dividends")
        ml = month_lookback_line(self)
        if ml:
            lines.append(ml)
        cr = self.cash_rate
        lines.append("Cash: " + ("earns the 3-month T-bill rate" if cr == "tbill" else
                                 f"earns {float(cr):.2%}/yr" if cr else "earns nothing"))
        if self.proxies:
            lines.append("Before inception (opt-in proxies): " + "; ".join(
                f"{v} stands in for {k} until {k}'s data begins" + (f" ({PROXY_LABELS.get(v)})" if PROXY_LABELS.get(v) else "")
                for k, v in self.proxies.items()))
        if self.start or self.end:
            lines.append(f"Period: {self.start or 'start of data'} to {self.end or 'latest'}")
        return "\n".join(lines)


# ------------------------------------------------------------------ tree helpers

WEIGHTINGS = ("equal", "inverse_vol", "market_cap", "risk_parity", "min_variance", "max_sharpe", "max_diversification")


def _wlabel(kind: str, lookback: int | None = None, w: list | None = None) -> str:
    if kind == "specified":      # a filter's fixed weights by rank
        return "weighted " + "/".join(fmt_weight(x) for x in (w or [])) + " by rank"
    base = {"equal": "equal weight", "inverse_vol": "inverse volatility", "market_cap": "market-cap weight",
            "risk_parity": "risk parity (equal risk contribution)", "min_variance": "minimum variance",
            "max_sharpe": "maximum Sharpe", "max_diversification": "maximum diversification"}[kind]
    if kind in ("equal", "market_cap"):
        return base
    return f"{base} ({lookback or (20 if kind == 'inverse_vol' else 60)}-day)"


def optimal_weights(method: str, R: np.ndarray) -> np.ndarray | None:
    """Long-only weights from a (days x assets) matrix of daily returns. None if it cannot be computed."""
    from scipy.optimize import minimize
    R = R[np.isfinite(R).all(axis=1)]
    k = R.shape[1]
    if k == 1:
        return np.ones(1)
    if len(R) < max(10, k + 2):
        return None
    cov = np.cov(R, rowvar=False) * 252
    cov = cov + np.eye(k) * 1e-10
    sd = np.sqrt(np.diag(cov))
    mu = R.mean(axis=0) * 252
    x0 = np.ones(k) / k
    bounds = [(0.0, 1.0)] * k
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    if method == "min_variance":
        fun = lambda w: w @ cov @ w  # noqa: E731
    elif method == "max_sharpe":
        fun = lambda w: -(w @ mu) / np.sqrt(max(w @ cov @ w, 1e-16))  # noqa: E731
    elif method == "max_diversification":
        fun = lambda w: -(w @ sd) / np.sqrt(max(w @ cov @ w, 1e-16))  # noqa: E731
    elif method == "risk_parity":
        def fun(w):
            v = w @ cov @ w
            rc = w * (cov @ w) / max(v, 1e-16)
            return float(((rc - 1 / k) ** 2).sum()) * 1e4
        x0 = (1 / sd) / (1 / sd).sum()
    else:
        return None
    r = minimize(fun, x0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 200, "ftol": 1e-12})
    w = np.clip(r.x if r.success or np.isfinite(r.x).all() else x0, 0, None)
    return w / w.sum() if w.sum() > 0 else None

NODE_KEYS = {  # allowed keys per node type ("name" is an optional label on any node)
    "asset": {"asset"},
    "cash": {"cash"},
    "custom": {"custom", "tickers"},
    "weights": {"weights", "w", "children", "lookback", "allow_short"},
    "if": {"if", "on", "then", "else"},
    "filter": {"filter", "universe", "children", "fallback"},
}
FILTER_KEYS = {"select", "n", "by", "require", "weights", "lookback", "w"}
# labels any node may carry, which never change the allocation: "name" (a group name) and "composer" (the ids of
# the Composer blocks it came from: "id"; "group_id" of a named group around it; "then_id" / "else_id" of the
# if-children of an if block), kept so that an exported symphony keeps its ids
META_KEYS = {"name", "composer"}


def _node_type(n: dict) -> str | None:
    for k in ("asset", "custom", "weights", "if", "filter"):
        if k in n:
            return k
    if n.get("cash"):
        return "cash"
    return None


def _kids(n: dict) -> list[dict]:
    """Direct sub-nodes of a node (children, then/else, fallback)."""
    out = list(n.get("children") or []) if ("weights" in n or "filter" in n) else []
    if "if" in n:
        out += [n["then"], n["else"]]
    if "filter" in n and n.get("fallback"):
        out.append(n["fallback"])
    return out


def _is_asset(k: dict) -> bool:
    return isinstance(k, dict) and "asset" in k


def _has_short(n: dict) -> bool:
    if not isinstance(n, dict):
        return False
    if n.get("weights") == "specified" and any(w < 0 for w in n.get("w") or []):
        return True
    return any(_has_short(k) for k in _kids(n))


def _has_rules(n) -> bool:
    """Does the tree evaluate indicators (if-nodes, filters, look-back weightings)?"""
    if not isinstance(n, dict):
        return False
    if "if" in n or "filter" in n or ("weights" in n and n["weights"] not in ("equal", "specified", "market_cap")):
        return True
    return any(_has_rules(k) for k in _kids(n))


def without_ids(n):
    """A copy of a tree without the Composer block ids ("composer" labels) its nodes carry."""
    if isinstance(n, dict):
        return {k: without_ids(v) for k, v in n.items() if k != "composer"}
    if isinstance(n, list):
        return [without_ids(x) for x in n]
    return n


def _is_dynamic(n) -> bool:
    """Does the tree make decisions that can change from day to day (if-nodes, filters, Python functions)?"""
    if not isinstance(n, dict):
        return False
    if "if" in n or "filter" in n or "custom" in n:
        return True
    return any(_is_dynamic(k) for k in _kids(n))


def _continuous_weights(n) -> bool:
    """Do some target weights move a little every day (inverse volatility, market cap, optimisers), rather than
    only when a decision changes?"""
    if not isinstance(n, dict):
        return False
    if "weights" in n and n["weights"] not in ("equal", "specified"):
        return True
    if "filter" in n and isinstance(n["filter"], dict) and (n["filter"].get("weights") or "equal") not in ("equal", "specified"):
        return True
    return any(_continuous_weights(k) for k in _kids(n))


def follows_rules(p: "Portfolio") -> bool:
    """A portfolio that is never rebalanced on a schedule but still follows its rules every day: a drift band
    (Composer's threshold rebalancing) or a tree whose decisions change (if-nodes, filters). Its tree is
    re-evaluated at every close and it trades only when the target allocation changes or a holding leaves its
    band."""
    return p.rebalance == "none" and bool(p.drift_band or p.drift_band_relative or _is_dynamic(p.tree))


def _has_ndx(n) -> bool:
    if not isinstance(n, dict):
        return False
    if "filter" in n and n.get("universe") in ("NDX", "nasdaq100"):
        return True
    return any(_has_ndx(k) for k in _kids(n))


def _mcap_used(n) -> bool:
    """Does any rule, ranking or weighting in the tree read market caps?"""
    if not isinstance(n, dict):
        return False
    if "filter" in n and (any("market_cap" in str(n["filter"].get(k) or "") for k in ("by", "require"))
                          or n["filter"].get("weights") == "market_cap"):
        return True
    if n.get("weights") == "market_cap" or ("if" in n and "market_cap" in str(n.get("if"))):
        return True
    return any(_mcap_used(k) for k in _kids(n))


def _ndx_by_mcap(n) -> bool:
    """Does a Nasdaq-100 filter rank, require or weight by market cap?"""
    if not isinstance(n, dict):
        return False
    if "filter" in n and n.get("universe") in ("NDX", "nasdaq100"):
        f = n["filter"] if isinstance(n["filter"], dict) else {}
        if any("market_cap" in str(f.get(k) or "") for k in ("by", "require")) or f.get("weights") == "market_cap":
            return True
    return any(_ndx_by_mcap(k) for k in _kids(n))


def max_requirement(n, req) -> float:
    """Like max_gross, with each asset's weight multiplied by req(ticker) (its margin requirement per $ held): the
    largest requirement, as a fraction of equity before leverage, that any branch of the tree can ask for."""
    if not isinstance(n, dict):
        return req(None)
    kind = _node_type(n)
    if kind == "cash":
        return 0.0
    if kind == "asset":
        return req(n.get("asset"))
    if kind == "weights":
        kids = n.get("children") or []
        if n["weights"] == "specified":
            return float(sum(abs(float(w)) * max_requirement(k, req) for w, k in zip(n.get("w") or [], kids)))
        return max([max_requirement(k, req) for k in kids] or [req(None)])
    if kind == "if":
        return max(max_requirement(n.get("then"), req), max_requirement(n.get("else"), req))
    if kind == "filter":
        u = n.get("universe", "children")
        if u == "children":
            out = max([max_requirement(k, req) for k in (n.get("children") or [])] or [req(None)])
        elif isinstance(u, list):
            out = max([req(t) for t in u] or [req(None)])
        else:
            out = req(None)
        return max(out, max_requirement(n["fallback"], req)) if n.get("fallback") else out
    return req(None)


def max_gross(n) -> float:
    """The largest gross exposure (sum of absolute weights, longs plus shorts) the tree can ask for, over every
    branch an if-node or filter may take."""
    if not isinstance(n, dict):
        return 1.0
    kind = _node_type(n)
    if kind == "cash":
        return 0.0
    if kind == "weights":
        kids = n.get("children") or []
        if n["weights"] == "specified":
            return float(sum(abs(float(w)) * max_gross(k) for w, k in zip(n.get("w") or [], kids)))
        return max([max_gross(k) for k in kids] or [1.0])
    if kind == "if":
        return max(max_gross(n.get("then")), max_gross(n.get("else")))
    if kind == "filter":
        kids = (n.get("children") or []) if n.get("universe", "children") == "children" else []
        out = max([max_gross(k) for k in kids] or [1.0])
        return max(out, max_gross(n["fallback"])) if n.get("fallback") else out
    return 1.0


def _needs_nav(n: dict) -> bool:
    """Does any filter / weighting rank or weigh a child that is not a single asset?"""
    if "filter" in n and n.get("universe", "children") == "children" and not all(map(_is_asset, n.get("children") or [])):
        return True
    if "weights" in n and n["weights"] not in ("equal", "specified") and not all(map(_is_asset, n["children"])):
        return True
    return any(_needs_nav(k) for k in _kids(n))


def _flow_bound(v, cal0: pd.Timestamp, end: bool, name: str = "flow date") -> pd.Timestamp | None:
    """A flow-window bound as an inclusive date. A year means 1 Jan of that year; N < 1900 means year N of
    the backtest (a start on the first day of year N, an end on the last day of year N)."""
    if v is None or v == "":
        return None
    if isinstance(v, str) and re.fullmatch(r"\s*\d+\s*", v):
        v = int(v)
    if isinstance(v, bool):
        raise ValueError(f"{name}: expected a year, a date or a year number of the backtest")
    if isinstance(v, (int, float, np.integer)):
        k = int(v)
        if k < 1:
            raise ValueError(f"{name}: year numbers of the backtest start at 1")
        if k < 1900:
            return cal0 + pd.DateOffset(years=k) - pd.Timedelta(days=1) if end else cal0 + pd.DateOffset(years=k - 1)
        return pd.Timestamp(year=k, month=1, day=1)
    try:
        return pd.Timestamp(v)
    except (ValueError, TypeError) as e:
        raise ValueError(f"{name}: can't read {v!r} as a date") from e


def _window_text(start, end) -> str:
    def norm(v):
        if isinstance(v, str) and v.strip().isdigit():
            return int(v)
        return v

    start, end = norm(start), norm(end)
    parts = []
    if start not in (None, ""):
        if isinstance(start, (int, np.integer)):
            parts.append(f"from year {start} of the backtest" if start < 1900 else f"from 1 Jan {start}")
        else:
            parts.append(f"from {start}")
    if end not in (None, ""):
        if isinstance(end, (int, np.integer)):
            if end < 1900:
                parts.append(f"for the first {end} year{'s' if end != 1 else ''}" if start in (None, "")
                             else f"through year {end}")
            else:
                parts.append(f"until 1 Jan {end}")
        else:
            parts.append(f"until {end}")
    return (" " + " ".join(parts)) if parts else ""


def _whole(v, what: str, least: int) -> None:
    """A count or a window from a spec must be a whole number (never rounded): 2.5 or -5 is refused."""
    ok = isinstance(v, (int, float)) and not isinstance(v, bool) and float(v).is_integer() and v >= least
    if not ok:
        raise ValueError(f"{what} must be a whole number, at least {least} (got {v!r})")


def validate_node(n: dict, depth: int = 0) -> None:
    if depth > 20:
        raise ValueError("portfolio tree is nested too deeply")
    if not isinstance(n, dict):
        raise ValueError(f"portfolio node must be an object, got {n!r}")
    kind = _node_type(n)
    if kind is None:
        raise ValueError(f"unrecognised portfolio node {n!r}: a node needs one of 'asset', 'cash', 'weights', "
                         "'if', 'filter' (or 'custom' from Python)")
    extra = set(n) - NODE_KEYS[kind] - META_KEYS
    if kind != "cash" and "cash" in extra and not n.get("cash"):
        extra.discard("cash")  # {"cash": false} alongside another type is harmless
    if extra:
        raise ValueError(f"unknown key(s) {', '.join(repr(k) for k in sorted(extra))} in {kind} node {_short(n)}; "
                         f"allowed: {', '.join(sorted(NODE_KEYS[kind] | {'name'}))}")
    if "composer" in n:
        cm = n["composer"]
        if not isinstance(cm, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in cm.items()):
            raise ValueError(f"'composer' on {kind} node {_short(n)} must be an object of Composer block ids (text), "
                             'e.g. {"id": "..."}')
    if kind == "asset":
        data.load(n["asset"])  # raises DataError for unknown tickers
    elif kind == "custom":
        if not callable(n["custom"]) or not n.get("tickers"):
            raise ValueError("custom node needs a Python callable and a 'tickers' list")
        for t in n["tickers"]:
            data.load(t)
    elif kind == "weights":
        kids = n.get("children") or []
        if not isinstance(kids, list) or not kids:
            raise ValueError("weights node needs children")
        if n["weights"] == "specified":
            w = n.get("w")
            if not w or len(w) != len(kids):
                raise ValueError("specified weights need one weight per child")
            if abs(sum(w) - 1) > 1e-6:
                raise ValueError(f"weights must add up to 100% (got {sum(w):.1%})")
            if any(x < 0 for x in w) and not n.get("allow_short", True):
                raise ValueError("negative weights are not allowed here")
        elif n["weights"] not in WEIGHTINGS:
            raise ValueError(f"unknown weighting {n['weights']!r} (one of specified, {', '.join(WEIGHTINGS)})")
        if n.get("lookback") is not None:
            _whole(n["lookback"], "lookback (days)", 2)
        for k in kids:
            validate_node(k, depth + 1)
    elif kind == "if":
        for k in ("then", "else"):
            if k not in n:
                raise ValueError(f"if node needs '{k}'")
        expr.compile_expr(n["if"])
        data.load(n.get("on", "SPY"))
        validate_node(n["then"], depth + 1)
        validate_node(n["else"], depth + 1)
    elif kind == "filter":
        f = n["filter"]
        if not isinstance(f, dict):
            raise ValueError("'filter' must be an object with select, n, by, ...")
        bad = set(f) - FILTER_KEYS
        if bad:
            raise ValueError(f"unknown key(s) {', '.join(repr(k) for k in sorted(bad))} in filter settings; "
                             f"allowed: {', '.join(sorted(FILTER_KEYS))}")
        if not f.get("by"):
            raise ValueError("filter needs 'by' (the ranking metric, e.g. tret(tr, 63))")
        expr.compile_expr(f["by"])
        if f.get("require"):
            expr.compile_expr(f["require"])
        _whole(f.get("n", 1), "filter n (how many to pick)", 1)
        if f.get("lookback") is not None:
            _whole(f["lookback"], "filter lookback (days)", 2)
        if f.get("select", "top") not in ("top", "bottom"):
            raise ValueError("filter select must be 'top' or 'bottom'")
        if f.get("weights") == "specified":
            # fixed weights by rank: w[0] for the best pick, w[1] for the next, ...
            w = f.get("w")
            if not isinstance(w, list) or len(w) != int(f.get("n", 1)) or not all(
                    isinstance(x, (int, float)) and not isinstance(x, bool) and x > 0 for x in w):
                raise ValueError("a filter with 'specified' weights needs 'w': one positive weight per pick (n), best first")
            if abs(sum(w) - 1) > 1e-6:
                raise ValueError(f"a filter's weights by rank must add up to 1 (they add up to {sum(w):g})")
        elif f.get("weights", "equal") not in WEIGHTINGS:
            raise ValueError(f"unknown filter weighting {f['weights']!r} (one of {', '.join(WEIGHTINGS)}, or specified "
                             "with 'w')")
        elif "w" in f:
            raise ValueError("'w' (weights by rank) goes with weights 'specified'")
        u = n.get("universe", "children")
        if u == "children":
            kids = n.get("children") or []
            if not kids:
                raise ValueError("filter over its children needs children")
            for k in kids:
                validate_node(k, depth + 1)
        elif not (u in ("NDX", "nasdaq100") or (isinstance(u, list) and u)):
            raise ValueError("filter universe must be 'children', 'NDX' or a list of tickers")
        elif n.get("children"):
            raise ValueError("a filter with a ticker/index universe takes no children (use universe 'children')")
        if n.get("fallback"):
            validate_node(n["fallback"], depth + 1)


def _short(n: dict) -> str:
    s = json.dumps({k: v for k, v in n.items() if k not in ("children", "then", "else", "fallback")}, default=str)
    return s if len(s) < 80 else s[:77] + "..."


def _universe(n: dict, point_in_time: bool = True) -> list[str]:
    """Tickers a filter ranks directly (for universe 'children': its single-asset children). A Nasdaq-100 universe
    is every member there has been (filtered by point-in-time membership each day), or - point_in_time False,
    "today's members only" - the current members (data.current_members), with no membership filter."""
    u = n.get("universe", "children")
    if u == "children":
        return [data.canonical(k["asset"]) for k in n.get("children") or [] if _is_asset(k)]
    if u in ("NDX", "nasdaq100"):
        if not point_in_time:
            have = set(data.available_tickers())
            return [t for t in data.current_members() if t in have]
        return data.nasdaq100_ever()
    return [data.canonical(t) for t in u]


def tickers_in(n: dict, index_universes: bool = True) -> list[str]:
    """Every ticker the tree reads (index_universes=False: without the members of a Nasdaq-100 universe)."""
    out: list[str] = []

    def add(t):
        t = data.canonical(t)
        if t not in out:
            out.append(t)

    def walk(x):
        if "asset" in x:
            add(x["asset"])
        elif "custom" in x:
            for t in x["tickers"]:
                add(t)
        elif "weights" in x:
            for k in x["children"]:
                walk(k)
        elif "if" in x:
            add(x.get("on", "SPY"))
            walk(x["then"])
            walk(x["else"])
        elif "filter" in x:
            if x.get("universe", "children") == "children":
                for k in x.get("children") or []:
                    walk(k)
            elif index_universes or x.get("universe") not in ("NDX", "nasdaq100"):
                for t in _universe(x):
                    add(t)
            if x.get("fallback"):
                walk(x["fallback"])
    walk(n)
    return out


def fixed_tickers(n: dict) -> list[str]:
    """Tickers that must have data for the portfolio to be defined (excludes index universes)."""
    out: list[str] = []

    def walk(x):
        if "asset" in x:
            out.append(data.canonical(x["asset"]))
        elif "custom" in x:
            out.extend(data.canonical(t) for t in x["tickers"])
        elif "weights" in x:
            for k in x["children"]:
                walk(k)
        elif "if" in x:
            out.append(data.canonical(x.get("on", "SPY")))
            walk(x["then"])
            walk(x["else"])
        elif "filter" in x:
            u = x.get("universe", "children")
            if u == "children":
                for k in x.get("children") or []:
                    walk(k)
            elif isinstance(u, list):
                out.extend(_universe(x))
            if x.get("fallback"):
                walk(x["fallback"])
    walk(n)
    return list(dict.fromkeys(out))


_PRETTY = [
    (r'sym\("([^"]+)"\)\.close', r"\1 price"),
    (r'sym\("([^"]+)"\)\.', r"\1 "),
    (r"\brsi\((?:close\s*,\s*)?(\d+)\)", r"RSI(\1)"),
    (r"\b(sma|ema|wma)\((?:close\s*,\s*)?(\d+)\)", lambda m: f"{m.group(1).upper()}({m.group(2)})"),
    (r"\b(sma|ema|wma)\(([^(),]+),\s*(\d+)\)", lambda m: f"{m.group(1).upper()}({m.group(3)}) of {m.group(2).strip()}"),
    (r"\btret\((?:tr\s*,\s*)?(\d+)\)", r"\1-day return"),
    (r"\bret\((?:close\s*,\s*)?(\d+)\)", r"\1-day price return"),
    (r"\bmax_drawdown\((?:tr\s*,\s*)?(\d+)\)", r"\1-day max drawdown"),
    (r"\bstdev_return\((?:tr\s*,\s*)?(\d+)\)", r"\1-day stdev of return"),
    (r"\bvolatility\((\d+)\)", r"\1-day volatility"),
    (r"\bma_return\((?:tr\s*,\s*)?(\d+)\)", r"\1-day average return"),
    (r"\btbill_ret\((\d+)\)", r"T-bills' \1-day return"),
    (r"\bclose\b", "price"),
]


def pretty_rule(rule: str) -> str:
    """A rule written for people: rsi(close, 10) > 79 -> RSI(10) > 79, tret(tr, 63) -> 63-day return."""
    s = str(rule or "").strip()
    for pat, rep in _PRETTY:
        s = re.sub(pat, rep, s)
    return re.sub(r"\s+", " ", s)


def short_name(n: dict, limit: int = 80) -> str:
    """A readable default name for an unnamed portfolio tree, e.g. "If TQQQ RSI(10) > 79: UVXY, else TQQQ"
    or "60% SPY / 40% TLT", cut to `limit` characters."""
    def nm(x, depth=0) -> str:
        if not isinstance(x, dict):
            return "?"
        if x.get("name") and depth:
            return str(x["name"])
        if "asset" in x:
            return data.canonical(x["asset"])
        if x.get("cash"):
            return "cash"
        if "custom" in x:
            return "custom weights"
        if "weights" in x:
            kids = x.get("children") or []
            if x["weights"] == "specified":
                return " / ".join(f"{fmt_weight(w)} {nm(k, depth + 1)}" for w, k in zip(x.get("w") or [], kids))
            lab = {"equal": "Equal weight", "inverse_vol": "Inverse vol", "market_cap": "Market cap",
                   "risk_parity": "Risk parity", "min_variance": "Min variance", "max_sharpe": "Max Sharpe",
                   "max_diversification": "Max diversification"}.get(x["weights"], x["weights"])
            return f"{lab} of " + ", ".join(nm(k, depth + 1) for k in kids)
        if "if" in x:
            on = data.canonical(x.get("on", "SPY"))
            cond = pretty_rule(x["if"])
            if "sym(" not in x["if"] and not cond.startswith(on):
                cond = f"{on} {cond}"
            return f"If {cond}: {nm(x['then'], depth + 1)}, else {nm(x['else'], depth + 1)}"
        if "filter" in x:
            f = x["filter"]
            u = x.get("universe", "children")
            what = "Nasdaq-100" if u in ("NDX", "nasdaq100") else ", ".join(
                nm(k, depth + 1) for k in (x.get("children") or [])) if u == "children" else ", ".join(u)
            return f"{f.get('select', 'top').title()} {f.get('n', 1)} of {what} by {pretty_rule(f.get('by'))}"
        return "portfolio"
    s = nm(n)
    return s if len(s) <= limit else s[: limit - 1].rstrip(" ,/:") + "…"


def fmt_weight(w: float) -> str:
    """A weight as a percentage without rounding it away: 0.6 -> "60%", 0.075 -> "7.5%", 1/3 -> "33.33%"."""
    return f"{round(float(w) * 100, 2):g}%"


def _member_lines(k: dict, indent: int) -> list[str]:
    """One member of a list (a weighting's or a filter's children) as its own item. A specified-weight group
    is written on one line ("60% TECL / 40% BIL") when all its parts are single lines, otherwise under a
    "group:" header with its parts one level deeper, so its parts never read as siblings of the list's
    other members."""
    pad = "  " * indent
    if "weights" in k:
        name = f" {k['name']}" if k.get("name") else ""
        if k["weights"] == "specified":
            parts = [describe(c, 0) for c in k["children"]]
            if all(len(x) == 1 for x in parts):
                return [pad + (f"group{name}: " if name else "") + " / ".join(f"{fmt_weight(w)} {x[0].strip()}" for w, x in zip(k["w"], parts))]
        return [f"{pad}group{name}:"] + describe(k, indent + 1, named=False)
    return describe(k, indent)


def describe(n: dict, indent: int = 0, named: bool = True) -> list[str]:
    pad = "  " * indent
    if named and isinstance(n, dict) and n.get("name"):
        # a named group (a Composer group block, a named sub-portfolio): its name, then what it holds
        inner = describe(n, 0, named=False)
        if len(inner) == 1:
            return [f"{pad}{n['name']}: {inner[0].strip()}"]
        return [f"{pad}group {n['name']}:"] + describe(n, indent + 1, named=False)
    if "asset" in n:
        return [f"{pad}{data.canonical(n['asset'])}"]
    if n.get("cash"):
        return [f"{pad}cash (T-bills)"]
    if "custom" in n:
        return [f"{pad}python function {getattr(n['custom'], '__name__', 'custom')}({', '.join(n['tickers'])})"]
    if "weights" in n:
        kind = n["weights"]
        lines = []
        if kind == "specified":
            for w, k in zip(n["w"], n["children"]):
                sub = describe(k, indent + 1)
                if len(sub) == 1:
                    lines.append(f"{pad}{fmt_weight(w)} {sub[0].strip()}")
                else:
                    lines.append(f"{pad}{fmt_weight(w)}:")
                    lines.extend(sub)
            return lines
        groups = not all(map(_is_asset, n["children"]))
        if kind == "market_cap" and groups:
            label = "equal weight (market-cap weighting needs single assets)"
        else:
            label = _wlabel(kind, n.get("lookback"))
            if groups and kind != "equal":
                label += ", groups measured on their simulated daily returns"
        lines.append(f"{pad}{label} of:")
        for k in n["children"]:
            lines.extend(_member_lines(k, indent + 1))
        return lines
    if "if" in n:
        from .expr import reads_own_series
        # "(on X)": the ticker whose prices the rule's own series (close, rsi(10), ...) read; a rule that reads only
        # other tickers (sym("QQQ").close) names them itself
        on = f" (on {data.canonical(n.get('on', 'SPY'))})" if reads_own_series(n["if"]) else ""
        return ([f"{pad}if {n['if']}{on}:"] + describe(n["then"], indent + 1)
                + [f"{pad}otherwise:"] + describe(n["else"], indent + 1))
    if "filter" in n:
        f = n["filter"]
        u = n.get("universe", "children")
        kids = n.get("children") or []
        groups = u == "children" and not all(map(_is_asset, kids))
        wk = f.get("weights", "equal")
        wl = _wlabel(wk, f.get("lookback"), f.get("w"))
        if groups and wk == "market_cap":
            wl = "equal weight (market-cap weighting needs single assets)"
        if groups:
            out = [f"{pad}{f.get('select', 'top')} {f.get('n', 1)} of these by {f['by']} "
                   f"(groups ranked on their simulated daily NAV), {wl}:"]
            for k in kids:
                out.extend(_member_lines(k, indent + 2))
        else:
            uname = "Nasdaq-100 members" if u in ("NDX", "nasdaq100") else ", ".join(_universe(n))
            out = [f"{pad}{f.get('select', 'top')} {f.get('n', 1)} of [{uname}] by {f['by']}, {wl}"]
        if f.get("require"):
            out.append(f"{pad}  only if {f['require']}, else:")
            out.extend(describe(n.get("fallback") or {"cash": True}, indent + 2))
        elif n.get("fallback"):
            out.append(f"{pad}  if nothing qualifies:")
            out.extend(describe(n["fallback"], indent + 2))
        return out
    return [f"{pad}{n}"]


# ------------------------------------------------------------------ evaluation

OPTIMISERS = ("risk_parity", "min_variance", "max_sharpe", "max_diversification")


_LEVEL_FUNCS = {"sma", "ema", "wma", "rma", "ma", "stdev", "atr", "highest", "lowest", "vwap", "bb_upper", "bb_lower", "donchian_upper",
                "donchian_lower", "keltner_upper", "keltner_lower", "supertrend", "sar", "weekly_sma", "monthly_sma",
                "weekly_ema", "monthly_ema", "weekly_close", "monthly_close"}
_LEVEL_NAMES = {"close", "open", "high", "low", "price"}
LEVEL_NOTE = ("The rule `{rule}` compares a price level with a fixed number. Indicators use total-return prices "
              "(price_basis \"adjusted\"), whose level starts at the first quoted close and then grows with the "
              "dividends, so it is above today's quote for a dividend payer; set price_basis to \"quoted\" for a "
              "fixed price level.")


def price_level_threshold(rule) -> bool:
    """Does the rule compare a price level (close, a moving average of it, another ticker's price) with a fixed
    number? Such a comparison depends on the price basis; ratios, returns and oscillators do not."""
    import ast
    try:
        tree = ast.parse(str(rule).strip(), mode="eval")
    except SyntaxError:
        return False

    def is_num(x):
        if isinstance(x, ast.UnaryOp) and isinstance(x.op, (ast.USub, ast.UAdd)):
            x = x.operand
        return isinstance(x, ast.Constant) and isinstance(x.value, (int, float)) and not isinstance(x.value, bool)

    def is_level(x):
        if isinstance(x, ast.Name):
            return x.id in _LEVEL_NAMES
        if isinstance(x, ast.Attribute):
            return x.attr in _LEVEL_NAMES
        if isinstance(x, ast.Call) and isinstance(x.func, ast.Name) and x.func.id in _LEVEL_FUNCS:
            a = [y for y in x.args if not is_num(y)]
            return not a or is_level(a[0])
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            ops = [node.left] + list(node.comparators)
            if any(is_num(o) for o in ops) and any(is_level(o) for o in ops):
                return True
    return False


def _is_num(x) -> bool:
    import ast
    if isinstance(x, ast.UnaryOp) and isinstance(x.op, (ast.USub, ast.UAdd)):
        x = x.operand
    return isinstance(x, ast.Constant) and isinstance(x.value, (int, float)) and not isinstance(x.value, bool)


def _is_level(x) -> bool:
    import ast
    if isinstance(x, ast.Name):
        return x.id in _LEVEL_NAMES
    if isinstance(x, ast.Attribute):
        return x.attr in _LEVEL_NAMES
    if isinstance(x, ast.Call) and isinstance(x.func, ast.Name) and x.func.id in _LEVEL_FUNCS:
        a = [y for y in x.args if not _is_num(y)]
        return not a or _is_level(a[0])
    return False


_LEVEL_PASS = {"abs", "maximum", "minimum", "ref"}   # functions whose value is in the units of their argument


def _level_valued(x) -> bool:
    """Is the expression in price units (a level: close, sma(close, 50), 1.05 * sym("X").close, close - low)?
    A ratio of two levels (close / sma(close, 200)) and returns, oscillators and counts are not."""
    import ast
    if _is_level(x):
        return True
    if isinstance(x, ast.UnaryOp) and isinstance(x.op, (ast.USub, ast.UAdd)):
        return _level_valued(x.operand)
    if isinstance(x, ast.BinOp):
        if isinstance(x.op, (ast.Mult, ast.Div)):
            if _is_num(x.right):
                return _level_valued(x.left)
            return isinstance(x.op, ast.Mult) and _is_num(x.left) and _level_valued(x.right)
        if isinstance(x.op, (ast.Add, ast.Sub)):
            sides = [x.left, x.right]
            return any(_level_valued(y) for y in sides) and all(_level_valued(y) or _is_num(y) for y in sides)
    if isinstance(x, ast.Call) and isinstance(x.func, ast.Name) and x.func.id in _LEVEL_PASS and x.args:
        a = [y for y in x.args if not (x.func.id == "ref" and _is_num(y))]
        return bool(a) and all(_level_valued(y) for y in a)
    return False


def _level_terms(x) -> list:
    """The price levels an expression is built from (each close / moving average / sym("X").close in it), not
    looking inside ratios' functions such as ret(), rsi() or quoted()."""
    import ast
    if _is_level(x):
        return [x]
    if isinstance(x, ast.UnaryOp):
        return _level_terms(x.operand)
    if isinstance(x, ast.BinOp):
        return _level_terms(x.left) + _level_terms(x.right)
    if isinstance(x, ast.Call) and isinstance(x.func, ast.Name) and x.func.id in _LEVEL_PASS:
        return [t for y in x.args for t in _level_terms(y)]
    return []


def _term_ticker(x, own: str) -> str:
    """The ticker whose price a level term reads: sym("X") inside it, else the rule's own ticker."""
    import ast
    for n in ast.walk(x):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "sym" and n.args
                and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str)):
            return data.canonical(n.args[0].value)
    return own


def quote_levels_why(rule: str, on: str | None = None) -> tuple[str, set]:
    """quote_levels, also saying why: {"fixed"} (a price level against a fixed number) and/or {"cross"} (price
    levels of different tickers against each other)."""
    import ast
    if not isinstance(rule, str):
        return rule, set()
    text = " ".join(rule.split())   # one line, so AST column offsets index the text directly
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        return rule, set()
    own = data.canonical(on) if on else "\0self"
    spans, why = set(), set()

    def pair(a, b):
        if _level_valued(a) and _level_valued(b):
            terms = _level_terms(a) + _level_terms(b)
            if len({_term_ticker(t, own) for t in terms}) > 1:
                why.add("cross")
                spans.update((t.col_offset, t.end_col_offset) for t in terms)
        for x, y in ((a, b), (b, a)):
            if _is_num(y) and _level_valued(x):
                terms = _level_terms(x)
                why.add("cross" if len({_term_ticker(t, own) for t in terms}) > 1 else "fixed")
                spans.update((t.col_offset, t.end_col_offset) for t in terms)

    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            ops = [node.left] + list(node.comparators)
            for a, b in zip(ops, ops[1:]):
                pair(a, b)
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("crossover", "crossunder")
              and len(node.args) == 2):
            pair(*node.args)
    # an inner term inside another wrapped term (sma(close) never nests in a level, but be safe): outermost only
    spans = {s for s in spans if not any(o != s and o[0] <= s[0] and s[1] <= o[1] for o in spans)}
    if not spans:
        return rule, set()
    for a, b in sorted(spans, reverse=True):   # the rest of the text (quotes, brackets) is kept as written
        text = f"{text[:a]}quoted({text[a:b]}){text[b:]}"
    return text, why


def quote_levels(rule: str, on: str | None = None) -> str:
    """Wrap price levels whose comparison depends on the price basis in quoted(...):

    - a level compared with a fixed number: 'close > 400' -> 'quoted(close) > 400', 'sma(close, 200) < 350' ->
      'quoted(sma(close, 200)) < 350' (a fixed price level means the quoted price);
    - levels of different tickers compared with each other (`on` is the rule's own ticker): 'close >
      sym("HYG").close' -> 'quoted(close) > quoted(sym("HYG").close)'. Each ticker's total-return level starts at
      its own first quoted close and grows with its own dividends, so levels of two tickers are not comparable.

    Comparisons within one ticker (close > sma(close, 200)), ratios, returns and oscillators are left alone."""
    return quote_levels_why(rule, on)[0]


def quote_metric(by: str) -> str:
    """A ranking metric that is a price level (close, sma(close, 20), stdev(close, 20)) -> quoted(...): ranking
    tickers by their total-return levels would compare numbers that are not comparable across tickers."""
    import ast
    if not isinstance(by, str):
        return by
    try:
        tree = ast.parse(" ".join(by.split()), mode="eval")
    except SyntaxError:
        return by
    return f"quoted({' '.join(by.split())})" if _level_valued(tree.body) else by


CROSS_NOTE = ("The rule `{r}` compares price levels of different tickers, so they are read as quoted (`{q}`): on "
              "total-return prices (price_basis \"adjusted\") each ticker's level starts at its own first quoted "
              "close and grows with its own reinvested dividends, so levels of two tickers are not comparable.")
FIXED_NOTE = ("The rule `{r}` compares a price level with a fixed number, so that price is read as quoted (`{q}`): "
              "the other indicators use total-return prices (price_basis \"adjusted\"), whose level grows with the "
              "reinvested dividends and sits above the quote for a dividend payer.")
RANK_NOTE = ("Ranking by `{r}` compares price levels across tickers, so it uses quoted prices (`{q}`): total-return "
             "levels (price_basis \"adjusted\") start at each ticker's own first quoted close and are not comparable "
             "between tickers.")


def level_notes(r: str, q: str, why: set) -> list[str]:
    out = []
    if "cross" in why:
        out.append(CROSS_NOTE.format(r=r, q=q))
    if "fixed" in why:
        out.append(FIXED_NOTE.format(r=r, q=q))
    return out


def _self_comparison(rule: str, on: str | None) -> str | None:
    """The first comparison of an expression with itself ('close > close', 'rsi(close, 10) > rsi(close, 10)',
    'close > sym("SPY").close' on SPY), which is always true or always false, as text; else None."""
    import ast
    try:
        tree = ast.parse(str(rule).strip(), mode="eval")
    except SyntaxError:
        return None
    on_c = data.canonical(on) if on else None

    class _Own(ast.NodeTransformer):   # sym("<the if's own ticker>").close -> close
        def visit_Attribute(self, node):
            self.generic_visit(node)
            v = node.value
            if (on_c and isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id == "sym" and v.args
                    and isinstance(v.args[0], ast.Constant) and isinstance(v.args[0].value, str)
                    and data.canonical(v.args[0].value) == on_c):
                return ast.Name(id=node.attr, ctx=ast.Load())
            return node

    tree = _Own().visit(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            ops = [node.left] + list(node.comparators)
            for a, b in zip(ops, ops[1:]):
                if ast.dump(a) == ast.dump(b) and not _is_num(a):
                    return ast.unparse(node)
    return None


def check_tree(p: "Portfolio") -> None:
    """Checks and rewrites that apply however the tree was written (parser, JSON, Build page, Composer): a
    comparison of a value with itself is refused; an if whose two branches are the same earns a note; and on an
    adjusted price basis a price level compared with a fixed number is read on quoted prices (quoted(...))."""
    def note(msg):
        if msg not in p.notes:
            p.notes.append(msg)

    def walk(n):
        if not isinstance(n, dict):
            return
        # windows, oscillator ranges, a signed max drawdown, impossible returns: the checks the English parser makes,
        # for trees from the Build page, JSON and the API too (expr.check_rule)
        f0 = n.get("filter")
        for holder, key, what in ((n, "if", "the condition"), (f0 if isinstance(f0, dict) else {}, "require", "the requirement")):
            r = holder.get(key)
            if isinstance(r, str) and r.strip():
                q, msgs = expr.check_rule(r, f"{what} `{r}`")
                if q != r:
                    holder[key] = q
                for m in msgs:
                    note(m)
        if isinstance(f0, dict) and isinstance(f0.get("by"), str):
            expr.check_windows(f0["by"], f"the ranking `{f0['by']}`")
        if isinstance(n.get("if"), str):
            bad = _self_comparison(n["if"], n.get("on"))
            if bad:
                raise ValueError(f"The condition `{n['if']}` compares a value with itself ({bad}), so it is always true or "
                                 "always false. Compare it with a number, another indicator or another ticker.")
            if without_ids(n.get("then")) == without_ids(n.get("else")):
                note(f"Both branches of the condition `{n['if']}` on {n.get('on', 'SPY')} hold the same thing "
                     f"({short_name(n['then'], 60)}), so the condition changes nothing.")
        f = n.get("filter")
        if isinstance(f, dict) and isinstance(f.get("require"), str):
            bad = _self_comparison(f["require"], None)
            if bad:
                raise ValueError(f"The requirement `{f['require']}` compares a value with itself ({bad}).")
        if isinstance(f, dict):
            u = n.get("universe", "children")
            cands = (n.get("children") or []) if u == "children" else u if isinstance(u, list) else None
            k = int(f.get("n", 1) or 1)
            if cands is not None and len(cands) and k >= len(cands):
                what = ", ".join(short_name(c, 30) if isinstance(c, dict) else str(c) for c in cands[:6])
                note(f"The {f.get('select', 'top')} {k} of {len(cands)} candidate{'s' if len(cands) != 1 else ''} ({what}): "
                     f"the filter keeps every candidate that has a value, so its ranking by {pretty_rule(f.get('by'))} "
                     "never leaves anything out" + (" (only its requirement can)" if f.get("require") else "")
                     + f". Name more than {k} candidates, or pick fewer.")
        if getattr(p, "price_basis", "adjusted") == "adjusted":
            for holder, key, on in ((n, "if", n.get("on", "SPY")), (f if isinstance(f, dict) else {}, "require", None)):
                r = holder.get(key)
                if isinstance(r, str):
                    q, why = quote_levels_why(r, on)
                    if q != r:
                        holder[key] = q
                        for m in level_notes(r, q, why):
                            note(m)
            if isinstance(f, dict) and isinstance(f.get("by"), str):
                q = quote_metric(f["by"])
                if q != f["by"]:
                    note(RANK_NOTE.format(r=f["by"], q=q))
                    f["by"] = q
        for k in _kids(n):
            walk(k)

    walk(p.tree)


class _Namespaces(dict):
    """ticker -> expr.Namespace, each built on first use (a universe of hundreds of tickers mostly needs few)."""

    def __init__(self, dfs: dict, basis: str, months: str = "trading", close_fill: bool = False):
        super().__init__()
        self.dfs, self.basis, self.months, self.close_fill = dfs, basis, months, close_fill

    def __missing__(self, t: str):
        ns = self[t] = expr.Namespace(self.dfs[t], ticker=t, price_basis=self.basis, month_lookbacks=self.months,
                                      close_fill=self.close_fill)
        return ns


_REUSE: dict = {"on": 0, "light": False, "key": None, "ev": None}   # the evaluator runs inside the context may reuse


class reuse_evaluations:
    """Within this context, a run of the same tree on the same data and calendar as the previous run reuses its
    evaluated target weights (indicators, rankings, synthetic NAVs), which do not depend on costs: for
    re-running one portfolio at several slippage levels. Notes are not repeated (the copies carry them)."""

    def __init__(self, result=None, light: bool = False):
        self.result = result    # the run whose evaluations to start from (its Result)
        self.light = light      # also skip the trade list and P&L attribution (only the equity curve is used)

    def __enter__(self):
        self.prev = dict(_REUSE)
        _REUSE["on"] += 1
        _REUSE["light"] = self.light
        src = getattr(self.result, "_evaluations", None)
        if src is not None:
            _REUSE["key"], _REUSE["ev"] = src
        return self

    def __exit__(self, *exc):
        _REUSE.update(self.prev)   # nothing outlives the context (an evaluator holds every ticker's indicators)
        _REUSE["on"] = self.prev["on"]
        return False


def _reuse_key(p, cal, dfs, off) -> tuple:
    return (id(p.tree), json.dumps(p.tree, sort_keys=True, default=str), tuple((t, id(df)) for t, df in dfs.items()),
            len(cal), cal[0], cal[-1], off, p.price_basis, p.point_in_time, p.cash_rate, p.target_vol,
            p.target_vol_lookback, p.leverage, getattr(p, "month_lookbacks", "trading"))


class _Evaluator:
    """Evaluates the tree to target weights at the close of day i of `cal`.

    `cal` may start before the backtest (`off` = index of its first day), so that the synthetic NAVs
    of groups have history to warm up on. A "member" of a filter or weighting is either a ticker
    (a single-asset child, measured on its own price data) or a node (any other child, measured on
    its synthetic NAV, see `nav`)."""

    def __init__(self, p: Portfolio, cal: pd.DatetimeIndex, dfs: dict[str, pd.DataFrame], off: int = 0):
        self.p, self.cal, self.dfs, self.off = p, cal, dfs, off
        self.basis = getattr(p, "price_basis", "quoted") or "quoted"
        self.months = getattr(p, "month_lookbacks", "trading") or "trading"
        self.ns = _Namespaces(dfs, self.basis, self.months,       # built when a rule first reads the ticker
                              close_fill=getattr(p, "fill", "close") == "close")
        self.cache: dict = {}
        self.close = {t: df["close"].reindex(cal).to_numpy() for t, df in dfs.items()}
        self._rets: dict = {}
        self.members = None
        self._nav: dict = {}       # id(node) -> synthetic NAV over cal
        self._rows: dict = {}      # id(node) -> its target weights on every day (memo from the NAV simulation)
        self._nav_ns: dict = {}
        self._keep: list = []      # nodes whose id() is a cache key stay alive
        self._rate = None
        self._quiet = 0            # > 0 while simulating a NAV: no notes (the real evaluation adds its own)
        self.thin: list = []       # (date, candidates, eligible, n, by): filters that picked from too few names
        self.share_class_days: dict = {}   # share-class group -> days two of its classes competed in a ranking

    # ------------------------------------------------------------ data
    def series(self, rule: str, t: str, kind: str) -> np.ndarray:
        key = (rule, t, kind)
        if key not in self.cache:
            ns = self.ns[t]
            ev = expr.evaluate(rule, ns) if kind == "bool" else expr.evaluate_value(rule, ns)
            lag = self._lag_note(t)
            if lag:
                # a rule on a series that closes after the stock market (a crypto day ends at 8pm New York, VIX at
                # 4:15pm) is known at a US close only as of its previous bar
                ev = ev.shift(1)
                if lag not in self.p.notes:
                    self.p.notes.append(lag)
            if kind == "bool":
                s = ev.fillna(False).astype(bool).reindex(self.cal, fill_value=False).to_numpy()
            else:
                s = ev.reindex(self.cal).to_numpy()
            self.cache[key] = s
            self._rule_notes(rule, ns)
        return self.cache[key]

    def _lag_note(self, t: str) -> str | None:
        """Why rules evaluated on ticker t's own bars must lag a bar here (None: they need not)."""
        if expr.is_crypto(t) and any(not expr.is_crypto(x) and not x.startswith("^") for x in self.dfs):
            return (f"Rules on {t} (a crypto day closes at 00:00 UTC, 8pm New York) use its previous day's bar: the "
                    "latest one complete at the US close the portfolio trades at.")
        if expr.is_late_close(t) and getattr(self.p, "fill", "close") == "close":
            return expr.LATE_CLOSE_NOTE.format(t=t)
        return None

    def mseries(self, rule: str, m, kind: str) -> np.ndarray:
        """A rule evaluated on a member: a ticker's own data, or a node's synthetic NAV."""
        if isinstance(m, str):
            return self.series(rule, m, kind)
        key = (rule, id(m), kind)
        if key not in self.cache:
            ns = self._nav_namespace(m)
            if kind == "bool":
                s = expr.evaluate(rule, ns).to_numpy()
            else:
                s = expr.evaluate_value(rule, ns).to_numpy()
            self.cache[key] = s
            self._rule_notes(rule, ns)
        return self.cache[key]

    def _rule_notes(self, rule: str, ns) -> None:
        """Notes a rule earns whenever it is evaluated (also inside a synthetic NAV, where other notes are quiet)."""
        msgs = list(getattr(ns, "notes", []))
        if self.basis == "adjusted" and price_level_threshold(rule):
            msgs.append(LEVEL_NOTE.format(rule=rule))
        for m in msgs:
            if m not in self.p.notes:
                self.p.notes.append(m)

    def vol(self, t: str, n: int) -> np.ndarray:
        return self.series(f"volatility({int(n)})", t, "value")

    def mcap(self, t: str) -> np.ndarray:
        key = ("mcap", t)
        if key not in self.cache:
            # quoted close x point-in-time shares (data.market_cap), not the total-return price level
            mc = data.market_cap(t)
            self.note(data.MCAP_NOTE)
            if mc.empty:
                self.cache[key] = np.full(len(self.cal), np.nan)
            else:
                self.cache[key] = mc.reindex(self.cal).to_numpy(dtype=float)
        return self.cache[key]

    def is_member(self, t: str, i: int) -> bool:
        return bool(self.member_row(t)[i])

    def member_row(self, t: str) -> np.ndarray:
        """Point-in-time Nasdaq-100 membership of `t` on every day of the calendar."""
        if self.members is None:
            names = data.nasdaq100_ever()
            m, _ = data.member_mask(names, self.cal)
            self.members = {n: m[:, j] for j, n in enumerate(names)}
            self._no_member = np.zeros(len(self.cal), bool)
            # from the run's first real session: the day-0 bar (the purchase at the close before it) is not a period
            # of the run, so it does not count towards coverage (a start in 2006 has no "lowest year 2005")
            first = getattr(self, "first_real", self.off)
            cov = data.coverage_note(str(self.cal[min(first, len(self.cal) - 1)].date()), str(self.cal[-1].date()))
            if cov:
                self.note(cov)
        return self.members.get(t, self._no_member)

    def has(self, t: str, i: int) -> bool:
        """Has the ticker started trading by bar i (and not stopped: delisted or acquired)? A missing price on a
        day it has already traded (another calendar's holiday) uses its last price rather than dropping it to cash."""
        return bool(self.has_row(t)[i])

    def has_row(self, t: str) -> np.ndarray:
        """`has` on every day of the calendar, computed once per ticker."""
        if not hasattr(self, "_has"):
            self._has = {}
        h = self._has.get(t)
        if h is None:
            c = self.close[t]
            T = len(c)
            fin = np.isfinite(c)
            ok = np.flatnonzero(fin)
            first = int(ok[0]) if len(ok) else T
            idx = np.arange(T)
            cs = np.r_[0, np.cumsum(fin)]                         # cs[k] = finite bars before k
            recent = cs[idx] - cs[np.maximum(0, idx - 10)] > 0    # a finite bar in [i - 10, i)
            g = gone_bar(self.dfs[t], self.cal)
            alive = idx <= g if g is not None else np.ones(T, bool)
            h = self._has[t] = fin | (alive & (idx > first) & recent)
        return h

    def ended(self, t: str, i: int) -> bool:
        """Has the ticker's data ended by bar i, well before the end of the run (delisted or acquired)?"""
        if not hasattr(self, "_gone"):
            self._gone = {}
        if t not in self._gone:
            self._gone[t] = gone_bar(self.dfs[t], self.cal)
        g = self._gone[t]
        return g is not None and i > g

    def rets(self, t: str) -> np.ndarray:
        if t not in self._rets:
            df = self.dfs[t]
            x = df["adj_close"] if "adj_close" in df else df["close"]
            self._rets[t] = x.pct_change().reindex(self.cal).to_numpy()
        return self._rets[t]

    def rate(self) -> np.ndarray:
        if self._rate is None:
            self._rate = _daily_rate(self.cal, self.p.cash_rate)
        return self._rate

    # ------------------------------------------------------------ synthetic NAV of a sub-tree
    def nav(self, n: dict) -> np.ndarray:
        """The daily NAV of sub-tree `n` held on its own (NaN before all its fixed tickers trade).

        Every close it is evaluated like the real tree (data up to that close) and the resulting
        weights earn the next day's total returns (adj_close); unallocated weight earns the cash rate.
        No costs, no leverage, rebalanced to target daily. NAV[t] therefore depends only on data up to t.
        The per-day weights are memoised, so evaluating `n` again later costs nothing."""
        k = id(n)
        if k in self._nav:
            return self._nav[k]
        self._keep.append(n)
        T = len(self.cal)
        self._quiet += 1
        try:
            rows = [self.eval(n, i) for i in range(T)]
        finally:
            self._quiet -= 1
        names = sorted({t for r in rows for t in r if t != "cash"})
        col = {t: j for j, t in enumerate(names)}
        W = np.zeros((T, len(names)))
        for i, r in enumerate(rows):
            for t, x in r.items():
                if t != "cash":
                    W[i, col[t]] += x
        R = np.column_stack([np.nan_to_num(self.rets(t)) for t in names]) if names else np.zeros((T, 0))
        cashw = 1.0 - W.sum(axis=1)
        rate = self.rate()
        r = np.zeros(T)
        if T > 1:
            r[1:] = (W[:-1] * R[1:]).sum(axis=1) + cashw[:-1] * rate[:-1]
        v = np.cumprod(1.0 + r)
        first = self._first_valid(n)
        v[:first] = np.nan
        if first < T:
            v = v / v[first]
        self._rows[k] = rows
        self._nav[k] = v
        return v

    def _first_valid(self, n: dict) -> int:
        first = 0
        for t in fixed_tickers(n):
            ok = np.flatnonzero(np.isfinite(self.close[t]))
            first = max(first, int(ok[0]) if len(ok) else len(self.cal))
        return first

    def _nav_namespace(self, n: dict) -> expr.Namespace:
        k = id(n)
        if k not in self._nav_ns:
            v = pd.Series(self.nav(n), index=self.cal)
            df = pd.DataFrame({"open": v, "high": v, "low": v, "close": v, "volume": 0.0, "adj_close": v,
                               "dividend": 0.0}, index=self.cal)
            self._nav_ns[k] = expr.Namespace(df, price_basis=self.basis, month_lookbacks=self.months)
        return self._nav_ns[k]

    def mrets(self, m) -> np.ndarray:
        if isinstance(m, str):
            return self.rets(m)
        key = ("navret", id(m))
        if key not in self.cache:
            v = self.nav(m)
            out = np.full(len(v), np.nan)
            out[1:] = v[1:] / v[:-1] - 1
            self.cache[key] = out
        return self.cache[key]

    # ------------------------------------------------------------ members
    @staticmethod
    def _member(k: dict):
        return data.canonical(k["asset"]) if _is_asset(k) else k

    def _live(self, m, i: int) -> bool:
        return self.has(m, i) if isinstance(m, str) else bool(np.isfinite(self.nav(m)[i]))

    def _expand(self, m, i: int) -> dict[str, float]:
        return {m: 1.0} if isinstance(m, str) else self.eval(m, i)

    def _weigh(self, method: str, members: list, i: int, lookback: int | None = None) -> list[float]:
        """Weights (aligned with `members`, adding up to 1) for a weighting method on day i."""
        k = len(members)
        eq = [1 / k] * k
        if method == "equal" or k == 0:
            return eq
        if method in OPTIMISERS:
            n = int(lookback or 60)
            lo = max(0, i - n + 1)
            R = np.column_stack([self.mrets(m)[lo: i + 1] for m in members])
            w = optimal_weights(method, R)
            if w is not None:
                return [float(x) if x > 1e-6 else 0.0 for x in w]
            self.note(f"Not enough history for {method.replace('_', ' ')} weights on some dates: equal-weighted then.")
            return eq
        lookback = lookback or 20
        if method == "inverse_vol":
            v = np.array([self.mseries(f"volatility({int(lookback)})", m, "value")[i] for m in members])
            ok = np.isfinite(v) & (v > 0)
            if ok.any():
                inv = np.where(ok, 1 / np.where(ok, v, 1), 0)
                return [float(x) for x in inv / inv.sum()]
            return eq
        if method == "market_cap":
            if not all(isinstance(m, str) for m in members):
                self.note("Market-cap weighting needs single assets: groups were equal-weighted.")
                return eq
            mc = np.array([self.mcap(t)[i] for t in members])
            ok = np.isfinite(mc) & (mc > 0)
            if ok.all():
                return [float(x) for x in mc / mc.sum()]
            self.note("Market-cap weights need share counts; tickers without them were equal-weighted.")
            if ok.any():
                out = list(eq)
                capw = mc[ok] / mc[ok].sum() * (ok.sum() / k)
                for j, w in zip(np.flatnonzero(ok), capw):
                    out[j] = float(w)
                return out
        return eq

    def note(self, msg: str) -> None:
        if not self._quiet and msg not in self.p.notes:
            self.p.notes.append(msg)

    # ------------------------------------------------------------ the tree
    def eval(self, n: dict, i: int) -> dict[str, float]:
        rows = self._rows.get(id(n))
        if rows is not None:
            return dict(rows[i])
        if "asset" in n:
            t = data.canonical(n["asset"])
            if self.has(t, i):
                return {t: 1.0}
            if i >= self.off:
                if self.ended(t, i):
                    self.note(f"{t}'s data ends on {self.dfs[t].index[-1].date()} (delisted or acquired): its slice was "
                              "held in cash after that.")
                else:
                    self.note(f"{t} had no price yet on some rebalance dates (before its history starts); its slice was "
                              "held in cash then.")
            return {"cash": 1.0}
        if n.get("cash"):
            return {"cash": 1.0}
        if "custom" in n:
            # Python API: fn(date, history) -> {ticker: weight}; history holds data up to and including date
            d = self.cal[i]
            # copies (a slice could reach the whole frame through its buffer), with files, the network and
            # data.load beyond `d` out of reach while the function runs (see expr.stream_callable)
            hist = {t: self.dfs[data.canonical(t)].loc[:d].copy() for t in n["tickers"]}
            tok = data.LOAD_CUTOFF.set(d)
            try:
                with expr.io_blocked():
                    w = n["custom"](d, hist) or {}
            finally:
                data.LOAD_CUTOFF.reset(tok)
            tot = sum(max(v, 0) for v in w.values())
            if tot > 1 + 1e-9:
                raise ValueError(f"custom weights on {d.date()} add up to {tot:.2%}")
            out = {data.canonical(t): float(v) for t, v in w.items() if v > 0}
            if tot < 1 - 1e-6:
                out["cash"] = 1 - tot
            return out
        if "weights" in n:
            kids = n["children"]
            method = n["weights"]
            if method == "specified":
                ws = n["w"]
            elif method == "equal":
                ws = [1 / len(kids)] * len(kids)
            else:
                mem = [self._member(k) for k in kids]
                live = [j for j, m in enumerate(mem) if self._live(m, i)]
                if live:
                    ws = [0.0] * len(kids)
                    for j, w in zip(live, self._weigh(method, [mem[j] for j in live], i, n.get("lookback"))):
                        ws[j] = w
                else:
                    ws = [1 / len(kids)] * len(kids)
            out: dict[str, float] = {}
            for w, k in zip(ws, kids):
                if w == 0:
                    continue
                for t, x in self.eval(k, i).items():
                    out[t] = out.get(t, 0.0) + w * x
            return out
        if "if" in n:
            on = data.canonical(n.get("on", "SPY"))
            cond = self.series(n["if"], on, "bool")[i]
            return self.eval(n["then"] if cond else n["else"], i)
        if "filter" in n:
            f = n["filter"]
            u = n.get("universe", "children")
            mk = ("members", id(n))
            if mk not in self.cache:   # the same members every day
                self._keep.append(n)
                mem = [self._member(k) for k in n.get("children") or []] if u == "children" else _universe(n, self.p.point_in_time)
                if u in ("NDX", "nasdaq100") and not self.p.point_in_time:
                    self.note(data.TODAY_MEMBERS_WARNING.format(n=len(mem)))
                self.cache[mk] = (mem, bool(mem) and all(isinstance(m, str) for m in mem))
            mem, assets = self.cache[mk]
            pit = u in ("NDX", "nasdaq100") and self.p.point_in_time
            top = f.get("select", "top") == "top"
            if assets:
                self._rank_counts = None
                chosen = self._rank_assets(n, mem, f["by"], pit, i, top, int(f.get("n", 1)))
                n_cands, elig = self._rank_counts or (len(chosen), len(chosen))
            else:
                cands = []
                for m in mem:
                    if isinstance(m, str) and (not self.has(m, i) or (pit and not self.is_member(m, i))):
                        continue
                    v = self.mseries(f["by"], m, "value")[i]
                    if np.isfinite(v):
                        cands.append((v, m))
                cands.sort(key=lambda x: x[0], reverse=top)
                chosen = [m for _, m in cands[: int(f.get("n", 1))]]
                n_cands = len(cands)
                elig = sum(1 for m in mem if not isinstance(m, str) or (self.has(m, i) and (not pit or self.is_member(m, i))))
            if i >= self.off and not self._quiet:
                if n_cands < int(f.get("n", 1)) or (pit and elig and n_cands < data.MCAP_MIN_COVERAGE * elig):
                    self.thin.append((self.cal[i], n_cands, elig, int(f.get("n", 1)), f["by"]))
            if f.get("require"):
                passed = [m for m in chosen if self.mseries(f["require"], m, "bool")[i]]
            else:
                passed = chosen
            if not passed:
                return self.eval(n.get("fallback") or {"cash": True}, i)
            if f.get("weights") == "specified":
                # fixed weights by rank (scaled to 100% over the picks there are); a pick that fails the requirement
                # passes its weight to the fallback
                rw = [float(x) for x in f["w"]][: len(chosen)]
                tot = sum(rw) or 1.0
                out: dict[str, float] = {}
                fb = 0.0
                for m, x in zip(chosen, rw):
                    if not any(m is q for q in passed):
                        fb += x / tot
                        continue
                    for t, y in self._expand(m, i).items():
                        out[t] = out.get(t, 0.0) + x / tot * y
                if fb > 1e-12:
                    for t, x in self.eval(n.get("fallback") or {"cash": True}, i).items():
                        out[t] = out.get(t, 0.0) + x * fb
                return out
            w: dict[str, float] = {}
            for m, x in zip(passed, self._weigh(f.get("weights", "equal"), passed, i, f.get("lookback"))):
                if x <= 0:
                    continue
                for t, y in self._expand(m, i).items():
                    w[t] = w.get(t, 0.0) + x * y
            if f.get("require") and len(passed) < len(chosen):
                # slots whose pick failed the requirement go to the fallback
                share = len(passed) / len(chosen)
                out = {t: x * share for t, x in w.items()}
                for t, x in self.eval(n.get("fallback") or {"cash": True}, i).items():
                    out[t] = out.get(t, 0.0) + x * (1 - share)
                return out
            return w
        raise ValueError(f"bad node {n}")

    def _rank_assets(self, n: dict, mem: list, by: str, pit: bool, i: int, top: bool, k: int) -> list:
        """A filter over single assets on day i: the k members with the highest (top) or lowest `by` among those
        trading (and in the index, `pit`) with a value that day; ties keep the members' order. The same choice as
        sorting (value, member) pairs, with the per-ticker rows kept as matrices so a day costs a few array ops."""
        key = ("rank", id(n), by, pit)
        st = self.cache.get(key)
        if st is None:
            self._keep.append(n)
            T, M = len(self.cal), len(mem)
            elig = np.column_stack([self.has_row(m) for m in mem])
            if pit:
                elig &= np.column_stack([self.member_row(m) for m in mem])
            st = self.cache[key] = {"elig": elig, "V": np.full((T, M), np.nan), "done": np.zeros(M, bool)}
        elig, V, done = st["elig"], st["V"], st["done"]
        live = np.flatnonzero(elig[i])
        todo = live[~done[live]]
        for j in todo:   # a ticker's metric is computed the first time it is eligible, as the loop did
            V[:, j] = self.mseries(by, mem[j], "value")
            done[j] = True
        v = V[i, live]
        fin = np.isfinite(v)
        n_elig = len(live)
        live, v = live[fin], v[fin]
        if getattr(self.p, "share_classes", "company") == "company" and len(live) > 1:
            live, v = self._one_class_per_company(mem, by, live, v, i)
        self._rank_counts = (len(live), n_elig)      # candidates with a value, eligible members (thin-ranking note)
        order = np.argsort(-v if top else v, kind="stable")[:k]
        return [mem[j] for j in live[order]]

    def _one_class_per_company(self, mem: list, by: str, live: np.ndarray, v: np.ndarray, i: int):
        """Listed share classes of one company (data.SHARE_CLASSES: GOOG/GOOGL, FOX/FOXA, ...) compete for one slot:
        the company is ranked once and held through its most liquid class that day (highest 20-day average dollar
        volume over 63 sessions, known at that close). Ranked by market cap, the company counts at its full value (each class's
        market_cap is the company's divided by its number of listed classes)."""
        grp = {t: g for g in data.SHARE_CLASSES for t in g}
        by_s = re.sub(r"\s+", "", str(by))
        best: dict = {}
        keep = np.ones(len(live), bool)
        v = v.astype(float).copy()
        for pos, j in enumerate(live):
            t = mem[j]
            g = grp.get(t)
            if g is None:
                continue
            if by_s == "market_cap":
                v[pos] = v[pos] * data._class_divisor(t)
            liq = self._dollar_volume(t)[i]
            liq = liq if np.isfinite(liq) else -1.0
            if g in best:
                q, lq = best[g]
                if liq > lq:
                    keep[q] = False
                    best[g] = (pos, liq)
                else:
                    keep[pos] = False
                self.share_class_days.setdefault(g, set()).add(self.cal[i])
            else:
                best[g] = (pos, liq)
        return live[keep], v[keep]

    def _dollar_volume(self, t: str) -> np.ndarray:
        dv = self.cache.get(("dv20", t))
        if dv is None:
            df = self.dfs[t]
            dv = (df["close"] * df["volume"]).rolling(63, min_periods=1).mean().reindex(self.cal).ffill().to_numpy()
            self.cache[("dv20", t)] = dv
        return dv


DELIST_GAP_DAYS = 7


def gone_bar(df: pd.DataFrame, cal: pd.DatetimeIndex) -> int | None:
    """The index in `cal` of a ticker's last bar when its data ends more than DELIST_GAP_DAYS before the
    end of `cal` (delisted or acquired), else None. (A few days short is a late data refresh, not a delisting.)"""
    if not len(df) or not len(cal) or df.index[-1] >= cal[-1] - pd.Timedelta(days=DELIST_GAP_DAYS):
        return None
    k = int(cal.searchsorted(df.index[-1], side="right")) - 1
    return k if 0 <= k < len(cal) - 1 else None


def _period_ids(idx: pd.DatetimeIndex, freq: str):
    if freq == "semiannual":
        return np.asarray(idx.year * 2 + (idx.month > 6).astype(int))
    am = annual_month(freq)
    if am:     # years that end with month am: the period changes after its last session
        return np.asarray(idx.year + (idx.month > am).astype(int))
    code = {"weekly": "W-FRI", "monthly": "M", "quarterly": "Q", "yearly": "Y"}[freq]
    return idx.to_period(code)


def _schedule(cal: pd.DatetimeIndex, freq: str) -> np.ndarray:
    """True on the last trading day of each period (daily: every day; none: only the first day)."""
    T = len(cal)
    ev = every_n(freq)
    if ev:
        n, unit = ev
        if unit == "days":   # the first day, then every n-th trading day after it
            out = np.zeros(T, bool)
            out[::n] = True
            return out
        base = _schedule(cal, {"weeks": "weekly", "months": "monthly"}[unit])
        ends = np.flatnonzero(base[1:]) + 1   # the period ends after the first day
        out = np.zeros(T, bool)
        out[ends[n - 1::n]] = True            # every n-th of them
        out[0] = True                         # initial allocation
        return out
    if freq == "daily":
        return np.ones(T, bool)
    out = np.zeros(T, bool)
    if freq == "none":
        out[0] = True
        return out
    per = np.asarray(_period_ids(cal, freq))
    # a bar ends its period when the next *scheduled* NYSE session (weekends and holidays known in advance)
    # is in another period: what was known that day. After an unscheduled closure (2001-09-11..14) the
    # bar before it did not know its week was over, so it is not a rebalance day.
    out[:] = np.asarray(_period_ids(_cal.next_scheduled(cal), freq)) != per
    # a period whose bars never reached its scheduled last session (unscheduled closure, data gap): rebalance
    # on the first bar after it instead
    starts = np.flatnonzero(np.r_[True, per[1:] != per[:-1]])
    ends = np.r_[starts[1:] - 1, T - 1]
    for a, b in zip(starts[:-1], ends[:-1]):
        if not out[a:b + 1].any():
            out[b + 1] = True
    # the latest bar only ends its period if the next NYSE session starts a new one
    out[-1] = np.asarray(_period_ids(_cal.next_sessions(cal[-1]), freq))[0] != per[-1]
    out[0] = True  # initial allocation
    return out


def _period_starts(cal: pd.DatetimeIndex, freq: str) -> np.ndarray:
    """Cash-flow days: the first trading day of each period, the first day of the backtest included (as
    Portfolio Visualizer does, and as the Monte Carlo simulation does): "add $1,000 a month for 20 years"
    makes 240 contributions, the first on day one alongside the starting capital, and "withdraw 4% a year"
    takes the first withdrawal on day one. A backtest that starts mid-period makes that period's flow on its
    first day and the next one at the start of the next period."""
    per = _period_ids(cal, freq)
    first = pd.Series(np.arange(len(cal)), index=cal).groupby(np.asarray(per)).min().to_numpy()
    out = np.zeros(len(cal), bool)
    out[first] = True
    return out


def _flow_schedule(cal: pd.DatetimeIndex, freq: str, start, end, growth: float, what: str) -> tuple[np.ndarray, np.ndarray]:
    """Flow days (period starts inside the window) and the growth multiplier (1+g)^(whole years since the
    window opened) of each day."""
    days = _period_starts(cal, freq)
    s = _flow_bound(start, cal[0], False, f"{what}_start")
    e = _flow_bound(end, cal[0], True, f"{what}_end")
    # year numbers of the backtest count whole periods: "for 20 years" of monthly flows is 240 flows even when
    # the backtest starts after the 1st of its first month (the boundary is the start of the period that
    # contains the anniversary), and "from year 21" starts with the next one
    def yr(v):
        try:
            k = int(v)
        except (TypeError, ValueError):
            return None
        return k if 1 <= k < 1900 and not isinstance(v, bool) else None

    def period_start(d: pd.Timestamp) -> pd.Timestamp:
        if freq == "semiannual":
            return pd.Timestamp(year=d.year, month=1 if d.month <= 6 else 7, day=1)
        return pd.Period(d, {"monthly": "M", "quarterly": "Q", "yearly": "Y"}[freq]).start_time
    if yr(end) is not None:
        e = period_start(cal[0] + pd.DateOffset(years=yr(end))) - pd.Timedelta(days=1)
    if yr(start) is not None and yr(start) > 1:
        s = period_start(cal[0] + pd.DateOffset(years=yr(start) - 1))
    if s is not None:
        days &= np.asarray(cal >= s)
    if e is not None:
        days &= np.asarray(cal <= e)
    mult = np.ones(len(cal))
    if growth:
        s0 = s if s is not None and s > cal[0] else cal[0]
        yrs = np.asarray(cal.year - s0.year - ((cal.month < s0.month) | ((cal.month == s0.month) & (cal.day < s0.day))))
        mult = (1 + growth) ** np.maximum(yrs, 0)
    return days, mult


def _flow_cpi_index(p: "Portfolio", cal: pd.DatetimeIndex, kind: str, days: np.ndarray, mult: np.ndarray,
                    amount: float) -> np.ndarray:
    """The CPI multiplier of a $ flow on each day (ones when it is not indexed). CPI is the figure published by
    that day (data.cpi(): each month's figure about two weeks after the month), so the amounts only use what was
    known. The base is the CPI of the dollars the amount is expressed in: the first day's (default), the flow's
    first payment ("flow"), or a calendar year's average ("2000"). Adds a note saying which, and the first payment."""
    T = len(cal)
    if not p.flow_inflation(kind) or not amount:
        return np.ones(T)
    cpi = data.cpi()
    if cpi.empty:
        p.notes.append(f"CPI data unavailable: the {kind}s were not inflation-adjusted.")
        return np.ones(T)
    c = cpi.reindex(cal.union(cpi.index)).ffill().reindex(cal)
    first = np.flatnonzero(days)
    dollars = getattr(p, f"{kind}_dollars", None)
    if dollars == "flow" and len(first):
        base, label = c.iloc[first[0]], f"dollars of its first payment ({cal[first[0]].date()})"
    elif dollars not in (None, "start", "flow"):
        y = int(dollars)
        m = data.cpi_monthly()
        yr = m[m.index.year == y]
        if len(yr) < 12:
            raise ValueError(f"No full year of CPI data for {y}: express the {kind} in the dollars of another year.")
        base, label = float(yr.mean()), f"{y} dollars (that year's average CPI)"
    else:
        base, label = c.dropna().iloc[0] if c.notna().any() else np.nan, f"{cal[0].year} dollars (CPI as of {cal[0].date()})"
    idx = (c / base).fillna(1.0).to_numpy() if np.isfinite(base) and base > 0 else np.ones(T)
    if len(first):
        j = first[0]
        freq = {"monthly": "a month", "quarterly": "a quarter", "semiannual": "every six months", "yearly": "a year"}[
            getattr(p, f"{kind}_freq")]
        p.notes.append(f"{kind.capitalize()}s: ${amount:,.0f} {freq} in {label}, indexed to CPI as published (each month's "
                       f"figure from about two weeks after the month); first paid {cal[j].date()} as "
                       f"${amount * idx[j] * mult[j]:,.0f}.")
    return idx


def warmup_dates(p, frames=None) -> tuple[pd.Timestamp | None, dict, dict]:
    """(warm, late, waited): the first day every rule of the tree can be evaluated, and per ranked member the day
    its lookbacks complete.

    warm is the latest of: each if-condition's indicators on its ticker; for each filter, the day its members have
    both the ranking (and requirement) values and the filter weighting's lookback (inverse volatility, optimisers) -
    every member with mode "all", enough to fill the N slots with mode "first"; each look-back weighting's
    lookback on its members; and the volatility target's lookback on every fixed ticker. A group member is
    measured on the data of its latest-starting ticker (a proxy for its simulated NAV). Only data up to each day
    decides whether that day is warm, so the date does not depend on later data. Index universes (Nasdaq-100) are
    left out: their members come and go."""
    tree = getattr(p, "tree", None)
    if not isinstance(tree, dict):
        return None, {}, {}
    basis = getattr(p, "price_basis", "quoted") or "quoted"
    mode = getattr(p, "warmup", "all") or "all"
    nss: dict = {}

    def ns_of(t):
        if t not in nss:
            df = (frames or {}).get(t)
            if df is None:
                try:
                    df = data.load(t)
                except (FileNotFoundError, data.DataError, KeyError):
                    df = None
            nss[t] = (expr.Namespace(df, ticker=t, price_basis=basis,
                                     month_lookbacks=getattr(p, "month_lookbacks", "trading") or "trading")
                      if df is not None else None)
        return nss[t]

    dates: list = []
    late: dict = {}
    waited: dict = {}

    def member_date(k, rules):
        if "asset" in k:
            t = data.canonical(k["asset"])
            cands = [t]
        else:
            cands = fixed_tickers(k)
            t = k.get("name") or short_name(k, 40)
        best = None
        for c in cands:
            ns = ns_of(c)
            if ns is None:
                continue
            ds = [x for x in (expr.first_defined(r, ns) for r in rules) if x is not None]
            d = max(ds) if ds else ns.df.index[0]
            best = d if best is None else max(best, d)
        return t, best

    def wrule(kind, lb):
        if kind in ("inverse_vol",) + OPTIMISERS:
            return f"volatility({int(lb or (20 if kind == 'inverse_vol' else 60))})"
        return None

    def walk(n):
        if not isinstance(n, dict):
            return
        if "if" in n:
            d = expr.first_defined(n["if"], ns_of(data.canonical(n.get("on", "SPY"))))
            if d is not None:
                dates.append(d)
            walk(n.get("then"))
            walk(n.get("else"))
        elif "filter" in n:
            f = n["filter"]
            u = n.get("universe", "children")
            rules = [r for r in (f.get("by"), f.get("require"), wrule(f.get("weights", "equal"), f.get("lookback"))) if r]
            if not (isinstance(u, str) and u in ("NDX", "nasdaq100")):
                per = {}
                members = (n.get("children") or []) if u == "children" else [{"asset": t} for t in _universe(n)]
                for k in members:
                    lab, d = member_date(k, rules)
                    if d is not None:
                        per[lab] = d
                if per:
                    need = min(int(f.get("n", 1)), len(per))
                    dates.append(max(per.values()) if mode == "all" else sorted(per.values())[need - 1])
                    late.update(per)
                    if mode == "all":
                        waited.update(per)
            for k in (n.get("children") or []) if u == "children" else []:
                walk(k)
            if n.get("fallback"):
                walk(n["fallback"])
        elif "weights" in n:
            kids = n.get("children") or []
            r = wrule(n["weights"], n.get("lookback"))
            if r and kids:
                for k in kids:
                    lab, d = member_date(k, [r])
                    if d is not None:
                        dates.append(d)
                        waited[lab] = max(d, waited.get(lab, d))
            for k in kids:
                walk(k)
    walk(tree)
    if getattr(p, "target_vol", None):
        r = f"volatility({int(getattr(p, 'target_vol_lookback', 60) or 60)})"
        for t in fixed_tickers(tree):
            ns = ns_of(t)
            d = expr.first_defined(r, ns) if ns is not None else None
            if d is not None:
                dates.append(d)
    return (max(dates) if dates else None), late, waited


def _union_index(idxs: list) -> pd.DatetimeIndex | None:
    """The sorted union of many date indexes in one step (pairwise unions re-infer a frequency each time)."""
    if not idxs:
        return None
    if len(idxs) == 1:
        return idxs[0]
    out = pd.DatetimeIndex(np.unique(np.concatenate([np.asarray(ix.values) for ix in idxs])))
    names = {ix.name for ix in idxs}
    return out.rename(names.pop()) if len(names) == 1 else out


def _short_history_warning(p: Portfolio, dfs: dict, first: pd.Timestamp) -> None:
    """A requested start the data could not honour: name the requested date, the actual start, the tickers whose
    short history cut the period (fixed holdings that began trading later, or assets a rule, ranking or weighting
    had to wait for) and how much of the period was lost, as a Warning."""
    if not p.start:
        return
    asked = pd.Timestamp(p.start)
    lost = (first - asked).days / 365.25
    if lost < 30 / 365.25:
        return
    late = sorted(((t, df.index[0]) for t, df in dfs.items() if len(df) and df.index[0] > asked + pd.Timedelta(days=7)),
                  key=lambda kv: kv[1], reverse=True)
    late = [(t, d) for t, d in late if d <= first]
    if late:
        who = ", ".join(f"{t} (from {d.date()})" for t, d in late[:6]) + (f" and {len(late) - 6} more" if len(late) > 6 else "")
        why = (f"the price history of {who} starts later" + (" (plus the indicators' warm-up after it)"
                                                                if first > late[0][1] + pd.Timedelta(days=3) else ""))
    else:
        why = "the rules' indicators or the data only allowed it then (see the notes below)"
    span = f"{lost:.1f} years" if lost >= 1 else f"{lost * 12:.0f} months"
    # a sleeve with a stated proxy can start earlier on request (opt-in, never by default)
    prox = {t: proxy_before(t, asked) for t, _ in late[:6]} if late and not p.proxies else {}
    prox = {t: x for t, x in prox.items() if x}
    msg = (f"Warning: You asked to start on {asked.date()}, but the backtest starts on {first.date()}: {why}. "
           f"{span} of the requested period are missing"
           + (f"; drop or replace {', '.join(t for t, _ in late[:3])} to test the whole period" if late else "")
           + (", or add 'with proxies before inception' to let a stated proxy stand in until it exists ("
              + ", ".join(f"{x} for {t}" + (f", {PROXY_LABELS[x]}" if x in PROXY_LABELS else "") for t, x in prox.items()) + ")"
              if prox else "") + ".")
    if msg not in p.notes:
        p.notes.insert(0, msg)


# Stated proxies for a sleeve before its own history begins, used only on request ("with proxies before
# inception", Portfolio.proxies): {series: [candidate proxies, first with data wins]}. Keys are the series a
# portfolio holds (the long-history series and the funds they stand for); the proxy's total return is spliced in
# before the series' first date. The first candidate that has data from the requested start (else the one with the
# longest history) is used, so longer simulated histories added later are picked up without changes here.
PROXIES_BEFORE_INCEPTION = {
    # TIPS -> intermediate Treasuries (similar duration, no inflation indexing)
    **{t: ["IEFSIM", "IEF"] for t in ("TIPSIM", "TIP", "SCHP", "VTIP", "STIP")},
    # international small caps / small value / value -> developed ex-US value, then developed ex-US
    **{t: ["EFASIM", "EFA"] for t in ("SCZSIM", "SCZ", "VSS")},
    **{t: ["EFVSIM", "EFASIM", "EFV", "EFA"] for t in ("AVDVSIM", "AVDV", "DLS", "DISV")},
    **{t: ["EFASIM", "EFA"] for t in ("EFVSIM", "EFV")},
    # emerging markets -> developed ex-US
    **{t: ["EFASIM", "EFA"] for t in ("EEMSIM", "VWOSIM", "EEM", "VWO", "IEMG")},
    # developed ex-US / Europe / Japan / all-world ex-US before 1975 -> US stocks
    **{t: ["SPYSIM", "SPY"] for t in ("EFASIM", "VXUSSIM", "VGKSIM", "EWJSIM", "EFA", "VEA", "VXUS", "VEU", "VGK", "EWJ")},
    # credit -> investment-grade corporates, then intermediate Treasuries
    **{t: ["LQDSIM", "IEFSIM"] for t in ("HYGSIM", "HYG", "JNK", "EMBSIM", "EMB")},
    **{t: ["IEFSIM", "IEF"] for t in ("MUBSIM", "MUB", "BNDXSIM", "BNDX", "BNDSIM", "BND", "AGG")},
    # REITs before 1972 -> US stocks
    **{t: ["SPYSIM", "SPY"] for t in ("VNQSIM", "VNQ")},
}
PROXY_LABELS = {"IEFSIM": "intermediate Treasuries", "IEF": "intermediate Treasuries",
                "EFASIM": "developed ex-US stocks", "EFA": "developed ex-US stocks",
                "EFVSIM": "developed ex-US value stocks", "EFV": "developed ex-US value stocks",
                "SPYSIM": "US stocks", "SPY": "US stocks", "LQDSIM": "investment-grade corporate bonds"}


def proxy_before(ticker: str, start) -> str | None:
    """The stated proxy for `ticker` before its history begins (PROXIES_BEFORE_INCEPTION): the first candidate
    with data from `start`, else the one whose history starts earliest; None if there is none with data."""
    have = set(data.available_tickers())
    best = None
    for c in PROXIES_BEFORE_INCEPTION.get(data.canonical(ticker), []):
        if c not in have:
            continue
        try:
            f = data.load(c).index[0]
        except (FileNotFoundError, data.DataError, KeyError, IndexError):
            continue
        if start is not None and f <= pd.Timestamp(start) + pd.Timedelta(days=7):
            return c
        if best is None or f < best[1]:
            best = (c, f)
    return best[0] if best else None


def splice_proxy(df: pd.DataFrame, proxy: pd.DataFrame) -> pd.DataFrame:
    """`df` with the proxy's total return before df's first date: the proxy's adjusted close, scaled to join df
    on its first day (so that day's return is the proxy's), as the price (no separate dividends before it)."""
    if not len(df) or not len(proxy):
        return df
    t0 = df.index[0]
    pa = (proxy["adj_close"] if "adj_close" in proxy else proxy["close"]).dropna()
    at = pa[pa.index <= t0]
    if not len(at) or at.index[0] >= t0:
        return df
    base = float(at.iloc[-1])
    pre_idx = pa.index[pa.index < t0]
    pre = pd.DataFrame(index=pre_idx)
    for c in df.columns:
        if c in ("open", "high", "low", "close", "quote_close"):
            pre[c] = pa[pre_idx] / base * float(df[c].iloc[0]) if pd.notna(df[c].iloc[0]) else pa[pre_idx] / base * float(df["close"].iloc[0])
        elif c == "adj_close":
            pre[c] = pa[pre_idx] / base * float(df["adj_close"].iloc[0])
        elif c == "volume":
            pre[c] = proxy["volume"].reindex(pre_idx).fillna(0.0) if "volume" in proxy else 0.0
        elif c == "open_ok":
            pre[c] = False
        elif df[c].dtype == bool:
            pre[c] = False
        else:
            pre[c] = 0.0 if pd.api.types.is_numeric_dtype(df[c]) else None
    out = pd.concat([pre, df])
    out.attrs = dict(getattr(df, "attrs", {}) or {})
    return out


def _apply_proxies(p: "Portfolio", dfs: dict) -> None:
    """Splice each opted-in proxy (Portfolio.proxies) in before its ticker's history, with a note per sleeve."""
    for t, px in (p.proxies or {}).items():
        t, px = data.canonical(t), data.canonical(px)
        if t not in dfs or not len(dfs[t]):
            continue
        try:
            pdf = data.load(px)
        except (FileNotFoundError, data.DataError, KeyError):
            raise ValueError(f"The proxy {px} for {t} has no price data.")
        if not len(pdf) or pdf.index[0] >= dfs[t].index[0]:
            continue
        first = dfs[t].index[0]
        dfs[t] = splice_proxy(dfs[t], pdf)
        lab = PROXY_LABELS.get(px)
        msg = (f"Proxy before inception (opt-in): {px}{f' ({lab})' if lab else ''} stands in for {t} from "
               f"{dfs[t].index[0].date()} until {t}'s data begins on {first.date()}; the returns before that are the "
               f"proxy's, not {t}'s.")
        if msg not in p.notes:
            p.notes.append(msg)


def _ends_early(t: str, last: pd.Timestamp, cal: pd.DatetimeIndex, delisted: dict, ndx) -> bool:
    """Does the series stop well before the run ends without being a delisting? A stock that was delisted or
    acquired has a record (data/delisted.json) or is a former Nasdaq-100 member; an imported (custom) series, or a
    fund or simulated series whose file simply ends, is not a delisting. `ndx`: a callable giving the members."""
    if last >= cal[-1] - pd.Timedelta(days=DELIST_GAP_DAYS) or last < cal[0]:
        return False
    if data.is_custom(t):
        return True
    return t not in delisted and t not in ndx()


def _end_at_series_end(p: "Portfolio", dfs: dict, cal: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """End the backtest on the last date every holding (and the benchmark) has data, with a Warning, when one of
    them is a series that ends early without being delisted (a custom series that stops in 2019, a fund file
    that ends), as Portfolio Visualizer does. A delisted or acquired stock is sold at its last price instead and
    the run continues (see gone_bar)."""
    if len(cal) < 2:
        return cal
    delisted = data.delisted()
    members: list = []

    def ndx() -> set:
        if not members:
            try:
                members.append(set(data.nasdaq100_ever()))
            except Exception:  # noqa: BLE001 - no membership data: nothing counts as a former member
                members.append(set())
        return members[0]
    ends = {t: dfs[t].index[-1] for t in tickers_in(p.tree, index_universes=False) if t in dfs and len(dfs[t])}
    if p.benchmark:
        from .metrics import parse_blend
        try:
            parts = parse_blend(p.benchmark) or [(str(p.benchmark), 1.0)]
        except ValueError:
            parts = []
        for t, _ in parts:
            t = data.canonical(t)
            try:
                df = data.load(t)
            except (FileNotFoundError, data.DataError, KeyError):
                continue
            if len(df):
                ends.setdefault(t, df.index[-1])
    early = {t: e for t, e in ends.items() if _ends_early(t, e, cal, delisted, ndx)}
    if not early:
        return cal
    last = min(early.values())
    out = cal[cal <= last]
    if len(out) < 2:
        return cal
    who = ", ".join(f"{t} ({e.date()}{', custom series' if data.is_custom(t) else ''})"
                    for t, e in sorted(early.items(), key=lambda kv: kv[1]))
    msg = (f"Warning: the data of {who} ends before the end of the period ({cal[-1].date()}), so the backtest ends on "
           f"{out[-1].date()}, the last date every holding{' and the benchmark' if p.benchmark else ''} has data (as "
           "Portfolio Visualizer does). A stock that was delisted or acquired (data/delisted.json, former Nasdaq-100 "
           "members) is instead sold at its last price and the run continues with that slice in cash.")
    if msg not in p.notes:
        p.notes.insert(0, msg)
    return out


def run(p: Portfolio) -> Result:
    p.validate()
    names = tickers_in(p.tree)
    dfs = data.load_many(names)
    if p.proxies:
        dfs = dict(dfs)
        _apply_proxies(p, dfs)
    must = fixed_tickers(p.tree)
    cal = _union_index([df.index for df in dfs.values()])
    cal_all = cal
    first_common = max(dfs[t].index[0] for t in must) if must else cal[0]
    start = pd.Timestamp(p.start) if p.start else first_common
    if p.start and pd.Timestamp(p.start) < first_common:
        late = [t for t in must if dfs[t].index[0] == first_common]
        p.notes.append(f"Start moved to {first_common.date()}, when {', '.join(late)} began trading.")
        start = first_common
    floor = None     # the earliest day the portfolio may be bought (membership data, market caps, warm-up)
    if _has_ndx(p.tree) and p.point_in_time:
        mem = data.membership()
        if mem is not None and len(mem) and start < mem.index[0]:
            m0 = mem.index[0]
            p.notes.append(f"Start moved to {m0.date()}, when the point-in-time Nasdaq-100 membership data begins"
                           + (f" (you asked for {pd.Timestamp(p.start).date()})" if p.start else "")
                           + ": before it the filter would have no members to choose from.")
            start = m0
        if mem is not None and len(mem):
            floor = mem.index[0]
    cal = cal[cal >= start]
    if p.end:
        cal = cal[cal <= pd.Timestamp(p.end)]
    cal = _end_at_series_end(p, dfs, cal)
    if len(cal) and _mcap_used(p.tree):
        named = [t for t in tickers_in(p.tree, index_universes=False) if t in dfs]
        msgs, fund_list = data.mcap_notes(named, cal[0], cal[-1])
        if fund_list:
            msgs.append(f"Market cap: {', '.join(fund_list[:5])} {'is a fund' if len(fund_list) == 1 else 'are funds'}: "
                        "ETFs have no market cap, so market-cap rules and weights never use "
                        f"{'it' if len(fund_list) == 1 else 'them'} (market-cap weights fall back to equal weight).")
        for msg_ in msgs:
            if msg_ not in p.notes:
                p.notes.append(msg_)
    if _ndx_by_mcap(p.tree) and p.point_in_time and len(cal):
        # market-cap rankings need share counts for most members: start where they cover MCAP_MIN_COVERAGE
        names = [t for t in data.nasdaq100_ever() if t in dfs]
        elig, _ = data.member_mask(names, cal)
        elig &= np.column_stack([dfs[t]["close"].reindex(cal).notna().to_numpy() for t in names]) if names else elig
        d, msg = data.mcap_start(elig, names, cal)
        if msg:
            p.notes.append(msg)
        if d is not None and d > cal[0]:
            cal = cal[cal >= d]
        if d is not None:
            floor = d if floor is None else max(floor, d)
    if _has_rules(p.tree):
        basis_note = ("Indicator prices: total return (dividends reinvested; price_basis \"adjusted\", as Composer and "
                      "Portfolio Visualizer). Trades and valuation use quoted prices plus cash dividends."
                      if p.price_basis == "adjusted" else
                      "Indicator prices: as quoted, not adjusted for dividends (price_basis \"quoted\", as TradingView). "
                      "Composer and Portfolio Visualizer use total-return prices, so RSI and moving averages of dividend "
                      "payers (bond funds, BIL) can differ from theirs.")
        if basis_note not in p.notes:
            p.notes.append(basis_note)
    # the warm-up: start trading on the first day every rule, ranking and weighting lookback has its values, so
    # nothing is bought on incomplete indicators and the statistics start with the starting capital
    waited: dict = {}
    if len(cal) > 1:
        warm, _, waited = warmup_dates(p, dfs)
        if warm is not None:
            floor = warm if floor is None else max(floor, warm)
        if warm is not None and warm > cal[0]:
            slow = sorted(((t, d) for t, d in waited.items() if d > cal[0]), key=lambda kv: kv[1], reverse=True)
            who = ", ".join(f"{t} ({d.date()})" for t, d in slow[:6]) + (f" and {len(slow) - 6} more" if len(slow) > 6 else "")
            if warm <= cal[-2]:
                asked = pd.Timestamp(p.start) if p.start and pd.Timestamp(p.start) < cal[0] else None
                p.notes.append(f"Warm-up: the portfolio starts on {cal[cal >= warm][0].date()} instead of "
                               + (f"the requested {asked.date()} (the data allowed {cal[0].date()})" if asked else
                                  f"{cal[0].date()}") + ", the "
                               "first day every rule's indicators" + (" and every ranked or weighted asset's lookback" if waited else "")
                               + " have values" + (f" (waited for {who})" if who else "")
                               + ". Nothing is traded during the warm-up, so the statistics start with the starting capital."
                               + (" Set warmup to \"first\" to start once enough assets can fill the slots." if slow and p.warmup == "all" else ""))
                cal = cal[cal >= warm]
            else:
                p.notes.append(f"Warm-up: the rules' lookbacks are only complete on {warm.date()}, after the end of the "
                               "period; the portfolio traded on incomplete indicators throughout.")
    # start on a day every fixed holding actually traded (not, say, a stock-market holiday on
    # which only a crypto or foreign series has a row)
    if must and len(cal):
        both = np.ones(len(cal), bool)
        for t in must:
            both &= np.isin(cal.values, dfs[t].index.values)
        if both.any():
            cal = cal[int(np.argmax(both)):]
    if len(cal) < 2:
        raise ValueError("no price data in the requested period")
    # Day 0, as Portfolio Visualizer starts from the prior period-end: the starting balance is invested at the
    # close of the session before the first day, so the first day's return counts. Only when every fixed holding
    # has a price that day and the rules could decide then (warm-up, membership and market-cap data); the initial
    # allocation is decided on data up to that close. Otherwise the portfolio is bought at the first day's close
    # and the session before only carries the starting capital (as before).
    day0 = False
    if p.capital > 0:
        k = int(cal_all.searchsorted(cal[0])) - 1
        no_prior = list(must)
        if k >= 0:
            a = cal_all[k]
            no_prior = [t for t in must if not (a in dfs[t].index and np.isfinite(dfs[t].at[a, "close"]))]
            if not no_prior and (floor is None or a >= floor):
                cal = cal_all[k:k + 1].append(cal)
                day0 = True
        if not day0 and (no_prior or k < 0):
            who = (", ".join(no_prior[:4]) + (f" and {len(no_prior) - 4} more" if len(no_prior) > 4 else "")) or "The data"
            p.notes.append(f"First day: {who} {'has' if len(no_prior) <= 1 else 'have'} no price before {cal[0].date()}, "
                           "so the starting balance is invested at that day's close and that day's return is not counted "
                           f"(the balance is shown on the session before, {_cal.anchor_day(cal[0], cal_all).date()}). "
                           "When every holding has a price the session before the start, the portfolio is bought at "
                           "that close instead, so the first day's return counts, as Portfolio Visualizer starts from "
                           "the prior period-end.")
    _short_history_warning(p, {t: dfs[t] for t in set(must) | set(waited) if t in dfs}, cal[int(day0)])
    for n in data.gap_notes(names, cal[0], cal[-1]) + data.custom_notes(names):
        if n not in p.notes:
            p.notes.append(n)
    # rule indicators come from each ticker's full history; synthetic NAVs of groups (filters or weightings
    # over sub-trees) are simulated from NAV_WARMUP sessions before the start so they are warm on day one
    full = cal_all[cal_all <= cal[-1]]
    off = int(full.searchsorted(cal[0]))
    lo = max(0, off - NAV_WARMUP) if _needs_nav(p.tree) else off
    ev = None
    key = _reuse_key(p, full[lo:], dfs, off - lo)
    if _REUSE["on"] and _REUSE["key"] == key:
        ev = _REUSE["ev"]
        ev.p = p
    if ev is None:
        ev = _Evaluator(p, full[lo:], dfs, off=off - lo)
        if _REUSE["on"]:
            _REUSE["key"], _REUSE["ev"] = key, ev
    base = off - lo
    # the first real session (after the day-0 purchase bar, when there is one): survivorship coverage counts from it
    ev.first_real = min(base + int(day0), len(ev.cal) - 1)
    T = len(cal)
    tick = list(dfs)
    idx = {t: j for j, t in enumerate(tick)}
    N = len(tick)
    O = np.column_stack([dfs[t]["open"].reindex(cal).to_numpy() for t in tick])
    C = np.column_stack([ev.close[t][base:] for t in tick])
    DIV = np.column_stack([dfs[t]["dividend"].reindex(cal).fillna(0.0).to_numpy() for t in tick])
    # split-adjusted -> as-traded units (data.as_traded_factor): whole-share rounding and reported order sizes use
    # the shares as traded that day
    SF = (np.column_stack([data.as_traded_factor(t, cal, dfs[t]) for t in tick]) if N else np.ones((T, 0)))
    # a "dividend" that is really a spin-off (or special) distribution: same cash, labelled as such
    SPIN = np.column_stack([cal.isin(data.spinoff_days(t)) for t in tick]) if N else np.zeros((T, 0), bool)
    # delisted / acquired: data that ends well before the run does; sold at the last close, proceeds held in cash
    gone = np.full(N, T)
    for j, t in enumerate(tick):
        g = gone_bar(dfs[t], cal)
        if g is not None:
            gone[j] = g
    delisted: list[str] = []
    spun: list[str] = []
    rate = _daily_rate(cal, p.cash_rate)
    borrow_extra = p.margin_rate / 252.0
    fee_daily = p.expense_ratio / 252.0
    fees = 0.0
    sched = _schedule(cal, p.rebalance)
    bands = bool(p.drift_band or p.drift_band_relative)
    # never rebalanced on a schedule but following the rules (threshold rebalancing, or a dynamic tree held without
    # a schedule): re-evaluate every close, trade only when the target changes or a holding leaves its band
    follow = follows_rules(p)
    gated = follow or (bands and p.rebalance == "daily")
    if follow:
        sched = np.ones(T, bool)
    # weights that move a little every day (inverse volatility, ...): only a change of holdings is a new target;
    # smaller moves trade through the drift band
    cont_w = _continuous_weights(p.tree) or bool(p.target_vol)
    slip = p.slippage_bps / 1e4
    # volume-based slippage: average daily volume of the 20 bars before each bar (known when the order is placed)
    vol_slip = p.slippage_model == "volume"
    ADV = (np.column_stack([dfs[t]["volume"].rolling(20, min_periods=1).mean().shift(1).reindex(cal).to_numpy(dtype=float)
                            for t in tick]) if vol_slip and N else None)

    def slip_of(j: int, i: int, q: float) -> float:
        """Slippage per side of an order of q shares of ticker j on bar i (see costs.volume_slippage)."""
        if not vol_slip:
            return slip
        return _costs.volume_slippage(slip, q, float(ADV[i, j]), p.spread_bps, p.impact_bps)

    def order_fee(j: int, i: int, q: float, fill: float) -> float:
        """Commission of one order: $ per order, % of value and any broker preset (per-share fees go through
        costs.traded_shares)."""
        com = p.commission + p.commission_pct * abs(q) * fill
        if p.commission_model:
            com += _costs.broker_commission(p.commission_model, _costs.traded_shares(abs(q) / SF[i, j], tick[j], cal[i]), abs(q) * fill)
        return com

    # opens that were never quoted (simulated series, mutual funds, old index data): a next-open fill there is
    # not possible. A ticker with no real open at all in the period is refused (as the signal engine does);
    # a day without one fills that ticker at the day's close instead, with a note.
    OPEN_OK = np.column_stack([(dfs[t]["open_ok"] if "open_ok" in dfs[t] else pd.Series(True, index=dfs[t].index))
                               .reindex(cal).fillna(False).to_numpy(dtype=bool) for t in tick]) if N else np.zeros((T, 0), bool)
    has_px = ~np.isnan(np.column_stack([dfs[t]["close"].reindex(cal).to_numpy(dtype=float) for t in tick])) if N else OPEN_OK
    no_opens = {j for j in range(N) if has_px[:, j].any() and not OPEN_OK[has_px[:, j], j].any()}
    open_fallback: set = set()

    # cash flows are scheduled on the backtest's days from its first day (day 0 only holds the starting balance)
    fcal = cal[int(day0):]
    no_flows = (np.zeros(len(fcal), bool), np.ones(len(fcal)))
    contrib_days, contrib_mult = (_flow_schedule(fcal, p.contribution_freq, p.contribution_start, p.contribution_end,
                                                 p.contribution_growth, "contribution") if p.contribution else no_flows)
    wd_days, wd_mult = (_flow_schedule(fcal, p.withdrawal_freq, p.withdrawal_start, p.withdrawal_end,
                                       p.withdrawal_growth, "withdrawal") if (p.withdrawal or p.withdrawal_pct) else no_flows)
    if p.capital <= 0 and not contrib_days[0]:
        raise ValueError("Starting with $0 needs a contribution on the first day; the first one here is later.")
    cinfl = _flow_cpi_index(p, fcal, "contribution", contrib_days, contrib_mult, p.contribution)
    winfl = _flow_cpi_index(p, fcal, "withdrawal", wd_days, wd_mult, p.withdrawal)
    if day0:
        contrib_days, wd_days = np.r_[False, contrib_days], np.r_[False, wd_days]
        contrib_mult, wd_mult, cinfl, winfl = (np.r_[1.0, a_] for a_ in (contrib_mult, wd_mult, cinfl, winfl))
    # the flows as scheduled (before any cap at the balance): benchmarks and the Monte Carlo replay these
    req_flows = (np.where(contrib_days, p.contribution * cinfl * contrib_mult, 0.0)
                 - np.where(wd_days, p.withdrawal * winfl * wd_mult, 0.0))
    depleted = None
    mm = p.maintenance_margin
    mreq = np.array([_margin.maintenance(t, mm) for t in tick]) if N else np.zeros(0)   # per $ held (margin.py)
    margin_days: list = []
    lev_peak = (0.0, 0.0, None)       # (gross/equity, target gross, date): worst drift above target between rebalances

    shares = np.zeros(N)
    cash = p.capital
    interest = 0.0
    last_px = np.full(N, np.nan)
    target: dict[str, float] = {}
    pending_target = None
    equity = np.zeros(T)
    flows = np.zeros(T)
    weights = np.zeros((T, N))
    mvals = np.zeros((T, N))          # market value of each holding at the close (signed), for holding periods
    cashw = np.zeros(T)
    orders: list[dict] = []
    # P&L attribution: every cash movement caused by a ticker (trades incl. costs, dividends,
    # reinvestment), and a ledger of those events in simulation order for the holding periods
    tcash = np.zeros(N)
    tdiv = np.zeros(N)
    tcom = np.zeros(N)
    ledger: list[tuple] = []          # (date, ticker, kind, shares, value, commission)
    turnover = 0.0
    n_rebal = 0
    vol_scale: list = []              # (date, exposure multiplier) of each volatility-target decision
    inc_div = np.zeros(T)             # income received each day: dividends (and distributions), cash interest
    inc_int = np.zeros(T)

    def px_now(prices):
        return np.where(np.isfinite(prices), prices, last_px)

    def value(prices) -> float:
        pv = px_now(prices)
        return cash + float(np.nansum(shares * np.nan_to_num(pv)))

    def trade_to(tgt: dict[str, float], prices: np.ndarray, i: int, reason: str, min_trade: bool = True) -> None:
        nonlocal cash, turnover
        pv = px_now(prices)
        eq = value(prices)
        if eq <= 0:
            return
        # cash the target allows us to borrow (leverage or weights above 100%)
        borrow_ok = max(0.0, sum(w for t, w in tgt.items() if t != "cash") - 1.0) * eq
        want = np.zeros(N)
        for t, w in tgt.items():
            if t == "cash":
                continue
            j = idx[t]
            if i >= gone[j]:
                continue  # delisted / acquired: nothing left to buy (its weight stays in cash)
            if np.isfinite(pv[j]) and pv[j] > 0:
                want[j] = eq * w / pv[j]
        if not p.fractional_shares:
            want = np.floor(want / SF[i] + 1e-9) * SF[i]
        delta = want - shares
        # sells first, then buys (scaled to the cash available)
        tradable = np.isfinite(pv) & (i < gone)
        day = cal[i]
        for sgn in (-1, 1):
            js = np.flatnonzero((delta * sgn > 1e-12) & tradable).tolist()
            if sgn == 1 and js:
                if vol_slip or p.commission_model:
                    need = 0.0
                    for j in js:
                        f_ = pv[j] * (1 + slip_of(j, i, delta[j]))
                        need += delta[j] * f_ + order_fee(j, i, delta[j], f_)
                else:
                    need = sum(delta[j] * pv[j] * (1 + slip) * (1 + p.commission_pct) + p.commission for j in js)
                scale = min(1.0, max(cash + borrow_ok, 0) / need) if need > 0 else 1.0
            else:
                scale = 1.0
            for j in js:
                q = delta[j] * scale
                if min_trade and abs(q) * pv[j] < p.min_trade * eq:
                    continue
                if not p.fractional_shares:
                    q = (np.floor(q / SF[i, j] + 1e-9) if q > 0 else -np.floor(-q / SF[i, j] + 1e-9)) * SF[i, j]
                if q == 0:
                    continue
                fill = pv[j] * (1 + np.sign(q) * slip_of(j, i, q))
                com = order_fee(j, i, q, fill)
                cash -= q * fill + com
                shares[j] += q
                tcash[j] -= q * fill + com
                tcom[j] += com
                ledger.append((day, tick[j], "buy" if q > 0 else "sell", q, abs(q) * fill, com))
                turnover += abs(q) * fill / eq
                orders.append({"date": day.date(), "ticker": tick[j], "side": "buy" if q > 0 else "sell",
                               "shares": abs(q) / SF[i, j], "price": fill * SF[i, j], "value": abs(q) * fill, "commission": com,
                               "reason": reason})

    def drifted(prices) -> bool:
        """Has any holding left its absolute (drift_band) or relative (drift_band_relative) band?"""
        eq = value(prices)
        if eq <= 0 or not target:
            return False
        pv = np.nan_to_num(px_now(prices))
        cur = {tick[j]: shares[j] * pv[j] / eq for j in range(N) if shares[j]}
        for t in set(cur) | {t for t in target if t != "cash"}:
            d = abs(cur.get(t, 0.0) - target.get(t, 0.0))
            if p.drift_band and d > p.drift_band:
                return True
            if p.drift_band_relative and d > max(p.drift_band_relative * abs(target.get(t, 0.0)), 1e-9):
                return True
        return False

    def gross_now(prices) -> float:
        return float(np.abs(shares * np.nan_to_num(px_now(prices))).sum())

    for i in range(T):
        o, c = O[i], C[i]
        # overnight interest and dividends
        if i > 0:
            r = rate[i - 1]
            earned = cash * r if cash >= 0 else cash * (r + borrow_extra)
            short_mv = -np.nan_to_num(shares * last_px)
            short_mv = np.where(shares < 0, short_mv, 0.0)
            smv = float(short_mv.sum())
            if smv > 0 and cash > 0 and r > 0 and p.short_rebate_spread:
                # short sale proceeds (part of cash) earn the rate less the rebate spread, floored at zero
                earned -= min(smv, cash) * min(r, p.short_rebate_spread / 252.0)
            cash += earned
            interest += earned
            inc_int[i] = earned
            if smv > 0 and p.borrow_fee:
                for j in np.flatnonzero(short_mv > 0):
                    fee = float(short_mv[j]) * p.borrow_fee / 252.0
                    cash -= fee
                    tcash[j] -= fee
                    tcom[j] += fee
                    ledger.append((cal[i], tick[j], "fee", 0.0, -fee, 0.0))
            if fee_daily:
                held = float(np.nansum(np.abs(shares) * np.nan_to_num(last_px)))
                cash -= held * fee_daily
                fees += held * fee_daily
        div_cash = shares * DIV[i]
        got = float(np.nansum(div_cash))
        if got:
            cash += got
            inc_div[i] = got
            for j in np.flatnonzero(div_cash != 0):
                tcash[j] += div_cash[j]
                tdiv[j] += div_cash[j]
                kind = "spin-off distribution" if SPIN[i, j] else "dividend"
                if SPIN[i, j] and f"{tick[j]} {cal[i].date()}" not in spun:
                    spun.append(f"{tick[j]} {cal[i].date()}")
                ledger.append((cal[i], tick[j], kind, 0.0, float(div_cash[j]), 0.0))
        # next-open execution of yesterday's decision
        if pending_target is not None and p.fill == "next_open":
            po = o
            need_px = [j for j in range(N) if (shares[j] != 0 or abs(pending_target.get(tick[j], 0.0)) > 1e-12)
                       and not OPEN_OK[i, j] and np.isfinite(c[j])]
            if need_px:
                dead = [tick[j] for j in need_px if j in no_opens]
                if dead and not _has_ndx(p.tree):
                    raise ValueError(f"{', '.join(dead)} {'has' if len(dead) == 1 else 'have'} no real opening prices (only a daily close, e.g. a simulated "
                                     "series or a mutual fund), so it can't be traded at the next open. Trade at the close "
                                     "instead (drop 'at the next open').")
                po = o.copy()
                for j in need_px:
                    po[j] = c[j]
                    open_fallback.add(tick[j])
            trade_to(pending_target, po, i, "rebalance")
            pending_target = None
        # cash flows: made at the close, after the day's return and any next-open trades (contributions are
        # invested and withdrawals sold at the close below), so the time-weighted return is (E_t - cf_t) / E_(t-1) - 1
        f = 0.0
        w_req = 0.0
        if contrib_days[i]:
            f += p.contribution * cinfl[i] * contrib_mult[i]
        if wd_days[i]:
            w_req = p.withdrawal * winfl[i] * wd_mult[i]
            if p.withdrawal_pct:
                pct_amt = p.withdrawal_pct * max(value(np.where(np.isfinite(o), o, last_px)), 0)
                w_req += pct_amt
                req_flows[i] -= pct_amt
            f -= w_req
        if f:
            cash += f
            flows[i] = f
        # withdrawals that overdraw cash: sell proportionally at the close
        np.copyto(last_px, c, where=np.isfinite(c))
        # delisted / acquired today (its last bar of data): sell at this last close
        for j in np.flatnonzero(gone == i):
            if shares[j] and np.isfinite(c[j]):
                q = -shares[j]
                fill = c[j] * (1 + np.sign(q) * slip_of(j, i, q))
                com = order_fee(j, i, q, fill)
                eq_d = value(c)
                cash -= q * fill + com
                shares[j] = 0.0
                tcash[j] -= q * fill + com
                tcom[j] += com
                ledger.append((cal[i], tick[j], "buy" if q > 0 else "sell", q, abs(q) * fill, com))
                turnover += abs(q) * fill / eq_d if eq_d > 0 else 0.0
                orders.append({"date": cal[i].date(), "ticker": tick[j], "side": "buy" if q > 0 else "sell",
                               "shares": abs(q) / SF[i, j], "price": fill * SF[i, j], "value": abs(q) * fill, "commission": com,
                               "reason": "delisted"})
                delisted.append(f"{tick[j]} delisted/acquired on {cal[i].date()}")
            target.pop(tick[j], None)
            if pending_target:
                pending_target.pop(tick[j], None)
        eq_close = value(c)
        if w_req > 0 and eq_close < 0:
            # the withdrawal is more than the account holds: sell everything at the close and pay out what is left
            # (the withdrawal is capped at the balance); the account is empty from here on
            pv = px_now(c)
            for j in np.flatnonzero(shares != 0):
                if not np.isfinite(pv[j]):
                    continue
                q = -shares[j]
                fill = pv[j] * (1 + np.sign(q) * slip_of(j, i, q))
                com = order_fee(j, i, q, fill)
                cash -= q * fill + com
                tcash[j] -= q * fill + com
                tcom[j] += com
                ledger.append((cal[i], tick[j], "buy" if q > 0 else "sell", q, abs(q) * fill, com))
                orders.append({"date": cal[i].date(), "ticker": tick[j], "side": "buy" if q > 0 else "sell",
                               "shares": abs(q) / SF[i, j], "price": fill * SF[i, j], "value": abs(q) * fill, "commission": com,
                               "reason": "withdrawal (money ran out)"})
                shares[j] = 0.0
            paid = max(w_req + cash, 0.0)     # cash is negative: the part of the withdrawal the account could not pay
            flows[i] += w_req - paid
            cash = 0.0 if paid > 0 else cash + w_req
            depleted = {"date": cal[i], "requested": w_req, "paid": paid}
            eq_close = value(c)
        tgt_cash = 1.0 - sum(w for t, w in target.items() if t != "cash")
        if depleted is None and f < 0 and cash < min(0.0, tgt_cash) * eq_close - 1e-6 * max(eq_close, 1.0) and eq_close > 0 and target:
            trade_to(target, c, i, "raise cash")
        # invest new contributions at the close in the current target mix (no selling)
        if f > 0 and target and (not sched[i] or gated):
            pv = px_now(c)
            live = {t: w for t, w in target.items() if t != "cash" and np.isfinite(pv[idx[t]]) and pv[idx[t]] > 0
                    and i < gone[idx[t]]}
            tot = sum(live.values())
            if tot > 0:
                budget = min(f, max(cash, 0.0))
                for t, w in live.items():
                    j = idx[t]
                    amt = budget * w / tot * (1 - p.commission_pct) / (1 + slip)
                    if vol_slip or p.commission_model:
                        # the impact and broker fee of this order, estimated at its size, come out of its budget
                        q0 = amt / pv[j]
                        f0 = pv[j] * (1 + slip_of(j, i, q0))
                        amt = max(0.0, (budget * w / tot - (order_fee(j, i, q0, f0) - p.commission_pct * q0 * f0))
                                  * (1 - p.commission_pct) / (f0 / pv[j]))
                    q = amt / pv[j] if p.fractional_shares else np.floor(amt / pv[j] / SF[i, j] + 1e-9) * SF[i, j]
                    if q <= 0:
                        continue
                    fill = pv[j] * (1 + slip_of(j, i, q))
                    com = order_fee(j, i, q, fill)
                    cash -= q * fill + com
                    shares[j] += q
                    tcash[j] -= q * fill + com
                    tcom[j] += com
                    ledger.append((cal[i], t, "buy", q, q * fill, com))
                    orders.append({"date": cal[i].date(), "ticker": t, "side": "buy", "shares": q / SF[i, j], "price": fill * SF[i, j],
                                   "value": q * fill, "commission": com, "reason": "contribution"})
        # reinvest dividends into the same holding at the close
        if got and p.reinvest_dividends:
            for j in np.flatnonzero(div_cash > 0):
                if np.isfinite(c[j]) and c[j] > 0 and cash > 0:
                    q = min(div_cash[j], cash) / c[j]
                    shares[j] += q
                    cash -= q * c[j]
                    tcash[j] -= q * c[j]
                    ledger.append((cal[i], tick[j], "reinvest", q, q * c[j], 0.0))
        # rebalance decision at the close
        decide = sched[i]
        rebalanced_at_close = False
        if decide:
            new = ev.eval(p.tree, base + i)
            if p.target_vol:
                new, k = _vol_target(p, ev, new, base + i)
                vol_scale.append((cal[i], k))
            elif p.leverage != 1.0:
                new = {t: w * p.leverage for t, w in new.items() if t != "cash"}
            new = {t: w for t, w in new.items() if abs(w) > 1e-9 and t != "cash"}
            changed = set(new) != set(target) or (not cont_w and any(
                abs(new.get(t, 0) - target.get(t, 0)) > 1e-9 for t in set(new) | set(target)))
            # "rebalance quarterly or when a weight drifts 5%": the schedule always trades and the band trades in
            # between. A daily re-evaluation with a band (Composer-style threshold rebalancing) and a dynamic tree that
            # is never rebalanced are gated: they trade when the rules change the target or a holding leaves its band
            # (measured against the new target), not back to the exact weights every day.
            target = new
            if gated and i > 0 and not changed and not (bands and drifted(c)):
                decide = False
            if decide:
                n_before = len(orders)
                if p.fill == "close":
                    trade_to(target, c, i, "initial" if not i else "drift rebalance" if gated and not changed
                             else "rebalance")
                    n_rebal += len(orders) > n_before
                    rebalanced_at_close = True
                else:
                    pending_target = dict(target)
                    n_rebal += 1
        elif bands and target and drifted(c):
            n_rebal += 1
            if p.fill == "close":
                trade_to(target, c, i, "drift rebalance")
                rebalanced_at_close = True
            else:
                pending_target = dict(target)
        # maintenance margin (leverage or shorts): equity must cover `mm` of gross exposure at the close,
        # otherwise every position is cut pro rata at the close back to the target gross exposure
        if (p.leverage > 1 or (shares < 0).any() or cash < -1e-9) and not rebalanced_at_close:
            g = gross_now(c)
            eq_now = value(c)
            tgt_gross = sum(abs(w) for t_, w in target.items() if t_ != "cash" and w == w) or p.leverage
            if g > 0 and eq_now > 0:
                pv = np.nan_to_num(px_now(c))
                need = float(np.abs(shares * pv) @ mreq) if mm else 0.0
                if mm and eq_now < need * (1 - 1e-12):
                    # de-risk with a cushion (margin.call_scale): to the lower of the target exposure and the one at
                    # which equity is 125% of the maintenance requirement, so the next down close is not a call again
                    k = _margin.call_scale(eq_now, g, need, tgt_gross)
                    trade_to({tick[j]: shares[j] * pv[j] * k / eq_now for j in range(N) if shares[j]}, c, i,
                             "margin call", min_trade=False)
                    margin_days.append(cal[i].date())
                    g, eq_now = gross_now(c), value(c)
                if eq_now > 0 and g / eq_now > max(lev_peak[0], 1.5 * tgt_gross):
                    lev_peak = (g / eq_now, tgt_gross, cal[i].date())
        equity[i] = value(c)
        mvals[i] = shares * np.nan_to_num(px_now(c))
        if equity[i] > 0:
            pv = np.nan_to_num(px_now(c))
            weights[i] = shares * pv / equity[i]
            cashw[i] = cash / equity[i]
        if equity[i] <= 0 and (p.withdrawal or p.withdrawal_pct or p.leverage > 1 or cash < 0 or (shares != 0).any()):
            if depleted is not None and depleted["paid"] > 0:
                p.notes.append(f"Portfolio depleted on {cal[i].date()}: the withdrawal due that day was "
                               f"${depleted['requested']:,.0f} but only ${depleted['paid']:,.0f} was left, so everything was "
                               "sold at the close and that was withdrawn (withdrawals are capped at the balance). The "
                               "balance is $0 from then on; returns are only measured while the account was funded.")
            elif p.withdrawal or p.withdrawal_pct:
                p.notes.append(f"Money ran out on {cal[i].date()}.")
            else:
                p.notes.append(f"The leveraged portfolio was wiped out on {cal[i].date()}: equity fell to "
                               f"${equity[i]:,.0f} at the close (a loss bigger than the margin could absorb); the "
                               "simulation stopped there and the balance is shown as $0 from then on.")
            equity[i:] = 0.0
            break
    if delisted:
        more = f" and {len(delisted) - 5} more" if len(delisted) > 5 else ""
        p.notes.append(f"Delisted: {', '.join(delisted[:5])}{more}; the position was closed at its last price (the final "
                       "close in the data; orders marked 'delisted') and the proceeds were held in cash. From the next "
                       "rebalance on, the portfolio's rules treat it as no longer trading: a fixed slice of it stays in "
                       "cash, a filter or weighting chooses among the remaining assets.")
    if spun:
        p.notes.append(f"Distributions: {', '.join(spun[:5])}{' and more' if len(spun) > 5 else ''} paid a spin-off or "
                       "special distribution (more than 15% of the price; e.g. shares of a spun-off company booked at their "
                       "value). It is paid in cash like a dividend but labelled 'spin-off distribution', not a dividend.")
    if open_fallback:
        p.notes.append(f"Opens: {', '.join(sorted(open_fallback)[:8])}{' and more' if len(open_fallback) > 8 else ''} had no "
                       "quoted opening price on some next-open rebalance days (old or simulated data), so those orders "
                       "filled at that day's close instead.")
    if margin_days:
        more = f" and {len(margin_days) - 5} more" if len(margin_days) > 5 else ""
        p.notes.append(f"Margin call on {', '.join(str(d) for d in margin_days[:5])}{more}: equity fell below "
                       f"its maintenance requirement ({mm:.0%} of gross exposure; the leverage factor times that for a "
                       "leveraged ETF), so every position was cut pro rata at the close to the lower of the target "
                       f"leverage and the exposure at which equity is {_margin.MARGIN_CALL_CUSHION:.0%} of the "
                       "requirement (orders marked 'margin call').")
    if vol_scale:
        ks = np.array([k for _, k in vol_scale if np.isfinite(k)])
        if len(ks):
            p.notes.append(f"Volatility target {float(p.target_vol):.0%}: the holdings were scaled by {ks.min():.2f}x to "
                           f"{ks.max():.2f}x (median {np.median(ks):.2f}x) of the tree's weights; the rest was held in cash"
                           + (" or borrowed" if ks.max() > 1 + 1e-9 else "") + ".")
    if lev_peak[2] is not None:
        p.notes.append(f"Between rebalances leverage drifted up to {lev_peak[0]:.1f}x gross exposure (on {lev_peak[2]}; "
                       f"target {lev_peak[1]:.1f}x)" + (f"; margin calls cap it at {1 / (mreq.max() if len(mreq) else mm):.3g}x." if mm else
                                                        " with margin calls turned off."))

    hw = pd.DataFrame(weights, index=cal, columns=tick)
    hw = hw.loc[:, (hw.abs() > 1e-9).any()]
    hw["cash"] = cashw
    gross = hw.drop(columns="cash").abs().sum(axis=1)
    if day0:
        # day 0: the starting balance at that close (bought then; any costs of the purchase show in the first
        # day's return), the holdings as bought
        idx_all = cal
        equity[0] = p.capital
        eq = pd.Series(equity, index=idx_all, name="equity")
        fl = pd.Series(flows, index=idx_all, name="flows")
        ex = pd.Series(gross.to_numpy(), index=idx_all, name="exposure")
        npos = pd.Series((hw.drop(columns="cash").abs() > 1e-6).sum(axis=1).to_numpy(), index=idx_all)
    else:
        start_day = _cal.anchor_day(cal[0], cal_all)   # the previous session, never a weekend or holiday
        idx_all = pd.DatetimeIndex([start_day]).append(cal)
        eq = pd.Series(np.concatenate([[p.capital], equity]), index=idx_all, name="equity")
        fl = pd.Series(np.concatenate([[0.0], flows]), index=idx_all, name="flows")
        ex = pd.Series(np.concatenate([[0.0], gross.to_numpy()]), index=idx_all, name="exposure")
        npos = pd.Series(np.concatenate([[0], (hw.drop(columns="cash").abs() > 1e-6).sum(axis=1).to_numpy()]), index=idx_all)
    if ev.share_class_days:
        live = {g: [d for d in ds if d >= cal[max(ev.off, 0)]] for g, ds in ev.share_class_days.items()}
        live = {g: ds for g, ds in live.items() if ds}
        if live:
            p.notes.append("Share classes: " + ", ".join("/".join(g) for g in live) + " are classes of one company, so a "
                           "ranking counted each company once (at its full market cap when ranking by market cap) and "
                           "held it through its more liquid class that day (higher 3-month average dollar volume), not "
                           "two slots of the same company. Set share_classes 'separate' in the JSON spec to rank each "
                           "class on its own.")
    if ev.thin:
        d0, c0, e0, n0, by0 = ev.thin[0]
        p.notes.append(f"Thin ranking: on {len(ev.thin)} rebalance date(s) a top/bottom-{n0} filter by {by0} ranked fewer "
                       f"than {n0} candidates or under {data.MCAP_MIN_COVERAGE:.0%} of its eligible universe (first "
                       f"{d0.date()}: {c0} of {e0} names had a value), so its picks there come from a subset.")
    od = pd.DataFrame(orders)
    end_px = np.nan_to_num(last_px)
    tri = {} if (_REUSE["on"] and _REUSE["light"]) else {
        t: (dfs[t]["adj_close"] if "adj_close" in dfs[t] else dfs[t]["close"]).reindex(cal).ffill().to_numpy() for t in tick}
    light = bool(_REUSE["on"] and _REUSE["light"])   # a cost-sensitivity rerun: the equity curve is all it needs
    trades = pd.DataFrame() if light else _round_trips(ledger, tick, end_px, cal[-1], mv=mvals, cal=cal, tri=tri)
    res = Result(strategy=p, equity=eq, trades=trades, exposure=ex, positions=npos, prices=dfs,
                 holdings=hw, interest=interest, in_market=pd.Series(gross.reindex(idx_all).fillna(0).to_numpy() > 1e-6, index=idx_all),
                 kind="allocation", orders=od)
    if depleted is not None:
        res.extras["depleted"] = depleted["date"]
    res.extras["flows_requested"] = pd.Series(req_flows if day0 else np.concatenate([[0.0], req_flows]), index=idx_all,
                                              name="flows")
    res.extras["day0"] = day0          # bought at day 0's close (the first day's return counts); benchmarks follow
    pad = (lambda a: a) if day0 else (lambda a: np.concatenate([[0.0], a]))
    res.extras["income"] = pd.DataFrame({"dividends": pad(inc_div), "interest": pad(inc_int)}, index=idx_all)
    res.extras.update({"fees": fees, "flows": fl, "turnover_annual": turnover / max((cal[-1] - cal[0]).days / 365.25, 1e-9),
                       "rebalances": n_rebal,
                       "vol_scale": pd.Series([k for _, k in vol_scale], index=pd.DatetimeIndex([d for d, _ in vol_scale]), dtype=float),
                       "attribution": None if light else _attribution(tick, tcash, tdiv, tcom, shares, end_px, od)})
    res._evaluations = (key, ev)   # for reuse_evaluations(res): lives as long as the result, never in a file
    return res


def _vol_target(p: Portfolio, ev: "_Evaluator", new: dict[str, float], i: int) -> tuple[dict[str, float], float]:
    """Scale the target weights so the mix's realised volatility over the lookback (daily total returns, ending
    at the decision's close i) matches p.target_vol, never above p.leverage times the weights."""
    cap = float(p.leverage)
    risky = {t: w for t, w in new.items() if t != "cash" and abs(w) > 1e-12}
    if not risky:
        return {}, 1.0
    n = int(p.target_vol_lookback or 60)
    lo = max(0, i - n + 1)
    R = np.column_stack([ev.rets(t)[lo: i + 1] for t in risky])
    w = np.array(list(risky.values()))
    R = R[np.isfinite(R).all(axis=1)]
    if len(R) < max(5, min(n, 20)):
        ev.note(f"Volatility target: fewer than {max(5, min(n, 20))} days of returns on some rebalance dates; "
                "the tree's weights were used unscaled then.")
        k = min(1.0, cap)
    else:
        vol = float(np.std(R @ w, ddof=1) * np.sqrt(252))
        k = cap if vol <= 1e-12 else min(cap, float(p.target_vol) / vol)
    return {t: x * k for t, x in risky.items()}, k


def _attribution(tick, tcash, tdiv, tcom, shares, end_px, od: pd.DataFrame) -> pd.DataFrame:
    """P&L per ticker: sales - purchases - costs + dividends (cash or reinvested) + value still held.

    sum(pnl) + interest - fees == final equity - starting capital - net cash flows (exactly, unless
    a leveraged portfolio was wiped out and the simulation stopped)."""
    mv = shares * end_px
    bought = od[od.side == "buy"].groupby("ticker")["value"].sum() if not od.empty else pd.Series(dtype=float)
    sold = od[od.side == "sell"].groupby("ticker")["value"].sum() if not od.empty else pd.Series(dtype=float)
    rows = []
    for j, t in enumerate(tick):
        if not (tcash[j] or mv[j] or tdiv[j]):
            continue
        rows.append({"ticker": t, "pnl": float(tcash[j] + mv[j]), "dividends": float(tdiv[j]),
                     "commissions": float(tcom[j]), "bought": float(bought.get(t, 0.0)), "sold": float(sold.get(t, 0.0)),
                     "end_value": float(mv[j]), "end_shares": float(shares[j])})
    df = pd.DataFrame(rows, columns=["ticker", "pnl", "dividends", "commissions", "bought", "sold", "end_value", "end_shares"])
    if not df.empty:
        tot = df["pnl"].abs().sum()
        df["share_of_pnl"] = df["pnl"] / df["pnl"].sum() if abs(df["pnl"].sum()) > 1e-9 else np.nan
        df = df.sort_values("pnl", key=lambda s: -s.abs() if tot else s).reset_index(drop=True)
    return df


def _round_trips(ledger: list[tuple], tick: list[str], end_px: np.ndarray, last_day, mv: np.ndarray | None = None,
                 cal: pd.DatetimeIndex | None = None, tri: dict | None = None) -> pd.DataFrame:
    """Holding periods per ticker, from the trade that opens a position to the one that closes it.

    The ledger holds every event in simulation order (dividends at the start of the day, trades,
    dividend reinvestment at the close), so the share count is exact, dividends are income of the
    holding period they were paid in and reinvested dividends add to its cost. The holding-period
    P&Ls of a ticker add up to its P&L in the attribution table.

    Columns (trades.csv of an allocation run):
      position_value  the average market value of the holding over the period (its closes while held), so a
                      daily-rebalanced position does not show the sum of every day's purchases
      entry_value     the value bought (or sold short) on the first day of the period
      bought / sold   every purchase / sale in the period (rebalancing trims and adds included)
      pnl             sales - purchases - costs + dividends (+ the value still held, for an open period)
      return          the ticker's own total return (adjusted close, dividends reinvested) from the close of
                      the first purchase to the close of the final sale; negated for a short position
    """
    if not ledger:
        return pd.DataFrame()
    rows = []
    px = dict(zip(tick, end_px))
    col = {t: j for j, t in enumerate(tick)}
    by: dict[str, list] = {}
    for e in ledger:
        by.setdefault(e[1], []).append(e)

    def extra(t, st, end, open_):
        out = {}
        if mv is None or cal is None:
            return out
        a = int(cal.searchsorted(pd.Timestamp(st["start"])))
        b = min(int(cal.searchsorted(pd.Timestamp(end))), len(cal) - 1)
        hi = b + 1 if (open_ or b <= a) else b     # on the exit day the holding is sold at the close
        seg = np.abs(mv[a:hi, col[t]])
        out["position_value"] = float(seg.mean()) if len(seg) else np.nan
        x = (tri or {}).get(t)
        if x is not None and np.isfinite(x[a]) and np.isfinite(x[b]) and x[a] > 0:
            r = x[b] / x[a] - 1
            out["return"] = float(r if st["side"] == "long" else -r)
        return out

    for t, evs in by.items():
        pos, peak, st = 0.0, 0.0, None
        for d, _, kind, q, v, com in evs:
            if kind in ("dividend", "fee", "spin-off distribution"):
                if st is not None:
                    st["income"] += v
                continue
            if st is None and kind in ("buy", "sell"):
                st = {"start": d, "spent": 0.0, "got": 0.0, "income": 0.0, "com": 0.0, "side": "long" if q > 0 else "short",
                      "entry_value": v, "bought": 0.0, "sold": 0.0}
                peak = 0.0
            if st is None:
                continue
            if kind in ("buy", "reinvest"):
                st["spent"] += v + com
            else:
                st["got"] += v - com
            if kind in ("buy", "sell"):
                st["bought" if kind == "buy" else "sold"] += v
            st["com"] += com
            pos += q
            peak = max(peak, abs(pos))
            if abs(pos) <= 1e-9 * max(peak, 1.0):
                rows.append(_trip(t, st, d.date(), extra=extra(t, st, d, False)))
                st, pos = None, 0.0
        if st is not None:
            st["got"] += pos * px[t]
            rows.append(_trip(t, st, last_day.date(), open_=True, extra=extra(t, st, last_day, True)))
    tr = pd.DataFrame(rows)
    if tr.empty:
        return tr
    tr = tr.sort_values(["exit_date", "entry_date", "ticker"], kind="stable").reset_index(drop=True)
    tr.index = tr.index + 1
    tr["cum_pnl"] = tr["pnl"].cumsum()
    return tr


def _trip(t, st: dict, end, open_=False, extra: dict | None = None) -> dict:
    start = st["start"].date() if hasattr(st["start"], "date") else st["start"]
    pnl = st["got"] + st["income"] - st["spent"]
    basis = st["spent"] if st["side"] == "long" else st["got"]
    out = {"ticker": t, "side": st["side"], "entry_date": start, "exit_date": end,
           "entry_price": np.nan, "exit_price": np.nan, "shares": np.nan, "position_value": basis,
           "entry_value": st.get("entry_value", np.nan), "bought": st.get("bought", np.nan), "sold": st.get("sold", np.nan),
           "pnl": pnl, "return": pnl / basis if basis else 0.0,
           "bars_held": int(np.busday_count(pd.Timestamp(start).date(), pd.Timestamp(end).date())),
           "exit_reason": "still held" if open_ else "rebalanced out", "mae": np.nan, "mfe": np.nan,
           "commission": st["com"], "income": st["income"], "entry_fill": "close", "exit_fill": "close"}
    out.update(extra or {})
    return out

