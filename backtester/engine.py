"""Daily bar-by-bar portfolio simulator for rule-based strategies.

Order of events on each bar i:
  0. OVERNIGHT - interest on cash (T-bill rate) or margin interest on borrowed cash, short borrow
                 fees, and cash dividends for positions held into the ex-date.
  1. OPEN      - scheduled exits at the open, stops/targets gapped through at the open, then market
                 entries at the open (yesterday's "next_open" signals or open-safe same-day rules),
                 then limit/stop entry orders whose level the open already crossed.
  2. INTRADAY  - limit/stop entries touched during the bar, then stop loss / ATR stop / trailing
                 stop / breakeven stop / take profit / scale-outs from the bar's high and low. If a stop
                 and a target are both touched on one bar the stop is assumed to have hit first
                 (conservative); with tv_compat, TradingView's path decides (open -> high -> low -> close
                 when the open is nearer the high, else open -> low -> high -> close). The stop and the target are one-cancels-other (OCA): whichever fills
                 closes the position and cancels the other. Trailing and breakeven stops move with the
                 best price reached, using each bar's high (low for shorts) after that bar's own stop
                 check, so a level set by today's high applies from the next bar (daily bars can't tell
                 whether the high came before the low).
  3. CLOSE     - time exits, rule exits and reversals at the close, then entries at the close.
  4. MARK      - portfolio marked to market at the close; liquidation if equity is exhausted.

Holding periods: hold_bars N exits exactly N bars after the entry bar (the bar the entry filled on), at
hold_exit_fill, for every entry fill. Buy at the close + hold 1 + sell at the close = the next day's close;
buy at the (next) open + hold 3 + sell at the close = the close 3 bars after the entry bar; hold 0 = the close
of an entry bar entered at the open. Trades report bars_held = exit bar - entry bar.

Entry orders follow TradingView's pyramiding rule: an order generated while the ticker already holds its
maximum number of entries (pyramiding, default 1) is ignored rather than kept. Next-open / next-close market
orders and limit/stop orders are generated at the signal bar's close, after that close's exits: a position
closed at that close is flat (the order is valid); one only scheduled to exit at the next open is not.
Same-bar entries at the open are generated at the open, after the open's exits.

Volume caps (max_volume_pct) use the fill bar's volume for fills at the close, and the previous bar's volume
for fills at the open or intraday (the day's volume is not known yet).

Prices are split-adjusted as quoted; dividends are paid in cash on the ex-date (and charged to
shorts), so share counts, per-share commissions and whole-share sizing are realistic.
Signals use data up to the close of the signal bar ("at the close" = market-on-close order).

Warm-up: until the entry rules can first fire (every indicator has a value; never past the first entry signal)
idle cash earns no interest, so the statistics, which start there, start at the starting capital (as a portfolio
starts trading on its warm-up day). Margin (backtester/margin.py, shared with portfolios): Reg T 2x / portfolio
margin 4x, leveraged ETFs at the maintenance margin times their leverage factor; a margin call cuts pro rata to the
lower of the leverage cap and the exposure at which equity is 125% of the maintenance requirement.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import re

import numpy as np
import pandas as pd

from . import calendar as _cal
from . import costs, data, expr
from . import margin as _margin
from .strategy import Strategy

INTRADAY_REASONS = ("stop loss", "ATR stop", "trailing stop", "chandelier stop", "breakeven stop", "stop level", "take profit",
                    "scale out")


@dataclass
class Lot:
    shares: float
    price: float            # fill price incl. slippage
    bar: int
    at_open: bool
    commission: float
    income: float = 0.0     # dividends received (paid, for shorts) and borrow fees, allocated to the lot
    fill_kind: str = ""     # how the entry filled when not a plain close/open fill: "limit" / "stop" (at the order's
                            # level during the bar) or "open" (the open had already crossed the level)


@dataclass
class Position:
    k: int
    sign: int
    lots: list[Lot]
    entry_bar: int
    peak: float                          # best price since entry (high for longs, low for shorts)
    atr_at_entry: float
    exit_due_bar: int | None
    exit_due_open: bool
    exit_sig: np.ndarray | None = None
    pending_open_exit: bool = False
    pending_reason: str = ""
    scaled: set = field(default_factory=set)
    init_stop: float = np.nan            # fixed stop (stop loss / ATR stop) when the position opened
    init_trail: float = np.nan           # trailing / chandelier stop when the position opened
    init_tgt: float = np.nan             # take-profit level when the position opened
    lvl_stop: float = np.nan             # stop_level, evaluated when the position opened (fixed)
    lvl_tgt: float = np.nan              # target_level / target_r, evaluated when the position opened (fixed)
    atr_now: float = np.nan              # the ATR as of the previous close (current_atr: exits that follow the ATR)
    arm_peak: float = np.nan             # the best price as of the previous bar (arms the breakeven stop)
    tranche: dict = field(default_factory=dict)   # stop_covers_scale_outs False: scale-out index -> shares reserved
    rest_done: bool = False              # ... and the stop / target already closed the shares not reserved
    dyn_stop: np.ndarray | None = None   # dynamic_levels: stop_level / target_level on every bar (use bar i-1 on bar i)
    dyn_tgt: np.ndarray | None = None
    risk: float = np.nan                 # the initial risk per share (entry - initial stop, in favour): R multiples
    lv_path: list = field(default_factory=list)   # (bar, stop, target) at each bar's start, for the chart

    @property
    def shares(self) -> float:
        lots = self.lots
        if len(lots) == 1:      # the usual case, without the generator (the same value as sum)
            return 0 + lots[0].shares
        return sum(l.shares for l in lots)

    @property
    def avg_price(self) -> float:
        s = self.shares
        return sum(l.shares * l.price for l in self.lots) / s if s else np.nan


@dataclass
class Result:
    strategy: Strategy
    equity: pd.Series
    trades: pd.DataFrame
    exposure: pd.Series
    positions: pd.Series
    prices: dict[str, pd.DataFrame] = field(repr=False, default_factory=dict)
    holdings: pd.DataFrame | None = None     # daily weight per ticker (end of day)
    interest: float = 0.0                    # net interest earned on cash
    in_market: pd.Series | None = None       # held a position at any time during the day
    kind: str = "signal"
    orders: pd.DataFrame | None = None
    extras: dict = field(default_factory=dict)


# ------------------------------------------------------------------ preparation

def _daily_rate(index: pd.DatetimeIndex, spec) -> np.ndarray:
    """Per-trading-day interest rate for idle cash."""
    if spec is None or spec is False or spec == 0:
        return np.zeros(len(index))
    if spec == "tbill":
        s = data.tbill_rate()
        if s.empty:
            return np.zeros(len(index))
        s = s.reindex(index.union(s.index)).ffill().reindex(index).fillna(0.0)
        return s.to_numpy() / 252.0
    return np.full(len(index), float(spec) / 252.0)


def _warmup_bar(strat, cal, tick, C, namespaces, long_sig, short_sig) -> int | None:
    """The bar on which the entry rules can first fire (report.warmup's date: every indicator of the entry rules
    has a value on the ticker with the longest history), when that is after the first bar and no entry signal comes
    before it; else None. No cash interest accrues up to it (see run)."""
    if not len(cal) or not tick or strat.cash_rate in (None, 0, 0.0, False, ""):
        return None
    try:
        firsts = [int(np.flatnonzero(~np.isnan(C[:, j]))[0]) if (~np.isnan(C[:, j])).any() else len(cal)
                  for j in range(len(tick))]
        t0 = tick[int(np.argmin(firsts))]
        ns = namespaces.get(t0)
        # an indicator with no value anywhere yet (data ending inside the warm-up): the whole run is warm-up, so a run
        # truncated inside the warm-up earns what the full run earns up to that day (nothing)
        end = pd.Timestamp.max
        ds = [expr.first_defined(r, ns, never=end) for r in (strat.entry, getattr(strat, "short_entry", None)) if r]
        ds = [d for d in ds if d is not None]
    except Exception:  # noqa: BLE001 - a Python-function rule or an unusual namespace: no warm-up trim
        return None
    if not ds:
        return None
    wi = int(cal.searchsorted(max(ds))) if max(ds) != end else len(cal) - 1
    # never past the first entry signal (a rule such as "a or b" can fire before all its indicators are warm): both
    # bounds depend only on data up to them, so a run truncated at any date earns the same interest before it
    sig = np.flatnonzero((long_sig | short_sig).any(axis=1)) if long_sig.ndim == 2 else np.flatnonzero(long_sig | short_sig)
    if len(sig):
        wi = min(wi, int(sig[0]))
    wi = min(wi, len(cal) - 1)
    return wi if wi > 0 else None


_NS_CACHE: dict = {}      # (ticker, id(bars)) -> (bars, Namespace): indicators are reused by the next run on the same data


def _namespace(df: pd.DataFrame, t: str, close_fill: bool = False) -> expr.Namespace:
    """The rule namespace of a ticker's bars, cached across runs (sweeps, the optimiser and the site re-run the same
    tickers). Keyed on the DataFrame object itself (data.load caches it), so fresh or changed data gets a new one."""
    hit = _NS_CACHE.get((t, id(df), close_fill))
    if hit is not None and hit[0] is df:
        return hit[1]
    if len(_NS_CACHE) > 1500:
        _NS_CACHE.clear()
    ns = expr.Namespace(df, ticker=t, close_fill=close_fill)
    _NS_CACHE[(t, id(df), close_fill)] = (df, ns)
    return ns


def strategy_namespace(strat, df: pd.DataFrame, t: str, close_fill: bool = False) -> expr.Namespace:
    """The namespace a strategy's rules are read in: the ticker's bars, plus its Pine `var` state series if any (the
    report's chart and today's signals read the rules the way the engine does)."""
    states = getattr(strat, "state_vars", None)
    if not states:
        return expr.Namespace(df, ticker=t, close_fill=close_fill)
    from .pine_import import state_series
    base = expr.Namespace(df, ticker=t)
    return expr.Namespace(df, state_series(states, base), ticker=t, close_fill=close_fill)


_CORP_CACHE: dict = {}


def _corp_days(kind: str, t: str, df) -> pd.DatetimeIndex:
    """data.spinoff_days / corporate_action_days of a ticker, cached per data object (re-runs, e.g. the report's cost
    sensitivity, ask again for the same tickers)."""
    fn = data.spinoff_days if kind == "spin" else data.corporate_action_days
    if df is None:
        return fn(t)
    key = (kind, t, id(df))
    hit = _CORP_CACHE.get(key)
    if hit is not None and hit[0] is df:
        return hit[1]
    if len(_CORP_CACHE) > 3000:
        _CORP_CACHE.clear()
    days = fn(t)
    _CORP_CACHE[key] = (df, days)
    return days


def _union_index(indexes: list[pd.DatetimeIndex]) -> pd.DatetimeIndex | None:
    """The sorted union of many date indexes (as repeated Index.union, without its per-step frequency inference)."""
    if not indexes:
        return None
    if len(indexes) == 1:
        return indexes[0]
    first = indexes[0]
    if all(isinstance(ix, pd.DatetimeIndex) and ix.tz is None and ix.dtype == first.dtype for ix in indexes):
        out = pd.DatetimeIndex(np.unique(np.concatenate([ix.values for ix in indexes])))
        names = {ix.name for ix in indexes}
        out.name = names.pop() if len(names) == 1 else None
        return out
    cal = first
    for ix in indexes[1:]:
        cal = cal.union(ix)
    return cal


def _xrank_tables(strat, dfs, tick, cal, member=None, per_trade_exit=False) -> dict:
    """{ticker: {xrank argument: Series}} for the xrank(...) calls in the rules: each ticker's percentile (0..1],
    average rank / count, of the argument among the universe's members that day (tickers with a bar and a value that
    day; on a point-in-time index universe only the members). Each day's value depends only on that day's values."""
    srcs: list[str] = []
    for r in (strat.entry, strat.short_entry, strat.exit_when, strat.rank_by):
        for x in expr.xrank_calls(r):
            if x not in srcs:
                srcs.append(x)
    if not srcs:
        return {}
    if per_trade_exit and expr.xrank_calls(strat.exit_when):
        raise ValueError("xrank() cannot be combined with position variables (bars_held, pnl, entry_price) in the exit "
                         "rule; use it in the entry or ranking rule, or in an exit rule without them.")
    if len(tick) < 2:
        raise ValueError(expr.XRANK_NEEDS_UNIVERSE)
    elig = np.column_stack([np.asarray(cal.isin(dfs[t].index)) for t in tick])
    if strat.universe_name == "NDX" and strat.point_in_time:
        elig &= member if member is not None else data.member_mask(tick, cal)[0]
    out: dict = {t: {} for t in tick}
    for src in srcs:
        M = np.full((len(cal), len(tick)), np.nan)
        for j, t in enumerate(tick):
            M[:, j] = expr.evaluate_value(src, _namespace(dfs[t], t)).reindex(cal).to_numpy(dtype=float)
        M[~elig] = np.nan
        R = pd.DataFrame(M).rank(axis=1, pct=True).to_numpy()
        for j, t in enumerate(tick):
            out[t][src] = pd.Series(R[:, j], index=cal).reindex(dfs[t].index)
    note = (f"xrank(): each ticker's percentile among the {len(tick)} tickers of the universe with a value that day"
            + (" (index members only)" if strat.universe_name == "NDX" and strat.point_in_time else "")
            + "; 1 = the highest, ties share their average rank.")
    if note not in strat.notes:
        strat.notes.append(note)
    return out


def _prepare(strat: Strategy):
    """_prepare_bars with Python-function rules streamed only from the run's first day (expr.STREAM_FROM)."""
    tok = expr.STREAM_FROM.set(None)
    try:
        return _prepare_bars(strat, tok)
    finally:
        expr.STREAM_FROM.reset(tok)


def _prepare_bars(strat: Strategy, _stream_tok=None):
    tickers = [data.canonical(t) for t in strat.universe]
    for t in tickers:
        why = data.not_investable(t)
        if why:
            raise ValueError(why)
    if strat.universe_name == "NDX" and not strat.point_in_time:
        # "today's members only": the current member list (not every former member without a membership filter)
        cur = set(data.current_members())
        tickers = [t for t in tickers if t in cur]
        if not tickers:
            raise ValueError("No price data for today's Nasdaq-100 members.")
        strat.universe = tickers
        strat.notes = [n for n in strat.notes if not n.startswith("Warning: survivorship bias")]
        strat.notes.append(data.TODAY_MEMBERS_WARNING.format(n=len(tickers)))
    dfs = data.load_many(tickers)
    start = pd.Timestamp(strat.start) if strat.start else None
    if strat.universe_name == "NDX" and strat.point_in_time:
        mem = data.membership()
        if mem is not None and len(mem):
            first_snap = mem.index[0]
            if strat.end and pd.Timestamp(strat.end) < first_snap:
                raise ValueError(f"Point-in-time Nasdaq-100 membership is only known from {first_snap.date()}, and this "
                                 f"test ends on {pd.Timestamp(strat.end).date()}. Use a later period, or name the tickers "
                                 "(or say 'using today's members only', which carries survivorship bias).")
            if start is None or start < first_snap:
                strat.notes.append(
                    f"Membership: point-in-time Nasdaq-100 membership is known from {first_snap.date()}, so the test "
                    f"starts there{'' if start is None else f' (not {start.date()})'}; an earlier start would need a "
                    "member list from hindsight.")
                start = first_snap
    end = pd.Timestamp(strat.end) if strat.end else None
    cal = _union_index([df.index for df in dfs.values()])
    traded = cal
    if start is not None:
        cal = cal[cal >= start]
    if end is not None:
        cal = cal[cal <= end]
    if len(cal) < 2:
        raise ValueError("no price data in the requested period")
    rules = " ".join(str(r) for r in (strat.rank_by, strat.entry, strat.short_entry, strat.exit_when) if r)
    if "market_cap" in rules and not (strat.universe_name == "NDX"):
        msgs, fund_list = data.mcap_notes(list(dfs), cal[0], cal[-1])
        if fund_list:
            raise ValueError(f"market_cap: {', '.join(fund_list[:5])} {'is a fund' if len(fund_list) == 1 else 'are funds'} "
                             "(ETF, mutual fund, index or simulated series): ETFs have no market cap (no shares "
                             "outstanding of a company). Use a stock, or a rule on price or volume.")
        for msg_ in msgs:
            if msg_ not in strat.notes:
                strat.notes.append(msg_)
    if strat.universe_name == "NDX" and strat.point_in_time and "market_cap" in rules:
        # ranking / selecting members by market cap needs share counts for most of them: start where they do
        names = list(dfs)
        elig, _ = data.member_mask(names, cal)
        elig &= np.column_stack([dfs[t]["close"].reindex(cal).notna().to_numpy() for t in names])
        d, msg = data.mcap_start(elig, names, cal)
        if msg:
            strat.notes.append(msg)
        if d is not None and d > cal[0]:
            cal = cal[cal >= d]
            if len(cal) < 2:
                raise ValueError(f"Market-cap data only covers enough of the Nasdaq-100 from {d.date()}, at the end of "
                                 "this period.")

    tick = list(dfs)
    T, N = len(cal), len(tick)
    arr = lambda: np.full((T, N), np.nan)  # noqa: E731
    O, H, L, C, V, DIV, ATR, VOL, LEVEL, ADV = (arr() for _ in range(10))
    SF = np.ones((T, N))   # split-adjusted -> as-traded units (data.as_traded_factor)
    long_sig = np.zeros((T, N), bool)
    short_sig = np.zeros((T, N), bool)
    exit_ = np.zeros((T, N), bool)
    rank = np.zeros((T, N))
    per_trade_exit = bool(strat.exit_when) and bool(expr.names_in(strat.exit_when) & expr.POSITION_VARS)
    namespaces = {}
    _rules = [r for r in (strat.entry, strat.short_entry, strat.exit_when, strat.rank_by) if r]
    late_close = any(callable(r) for r in _rules) or any(
        expr.is_late_close(data.canonical(x)) for r in _rules if isinstance(r, str)
        for x in re.findall(r"""sym\(\s*["']([^"']+)["']\s*\)""", r)) or any(
        isinstance(r, str) and bool(expr.DAILY_MACRO_FNS & expr.names_in(r)) for r in _rules)
    # the static open-time check (Strategy.validate) is backed by an empirical one on the longest history
    t0 = max(tick, key=lambda t: len(dfs[t]))
    # Python-function rules: only the run's days are read, so they are streamed from its first day on
    expr.STREAM_FROM.set(cal[0])
    # rules written as Python functions have no static check: cut the data at sampled dates and compare
    ec = late_close and strat.entry_fill == "close"
    for what, rule, kind, cf in (("entry", strat.entry, "bool", ec), ("short entry", strat.short_entry, "bool", ec),
                                 ("exit", strat.exit_when, "bool", late_close and strat.exit_when_fill == "close"),
                                 ("order level", strat.entry_level, "value", False),
                                 ("ranking", strat.rank_by, "value", ec)):
        if callable(rule):
            # (with the namespace the run reads, so the run reuses this stream)
            bad = expr.callable_check(rule, dfs[t0], t0, kind, window=(cal[0], cal[-1]), close_fill=cf)
            if bad:
                nm = getattr(rule, "__name__", "")
                label = f"{what} function {nm}()" if nm and not nm.startswith("<") else f"{what} function"
                raise ValueError(f"Lookahead: the {label} uses future data: {bad} "
                                 f"({t0}). A rule may only use each row and the rows before it: no .shift(-n), "
                                 "centred rolling windows (center=True) or whole-series statistics such as "
                                 "df.close.mean().")
    if strat.entry_fill == "open":
        for rule in (strat.entry, strat.short_entry):
            if rule:
                bad = expr.open_time_probe(rule, dfs[t0], t0)
                if bad:
                    raise ValueError(f"Lookahead: the entry is filled at the open but {bad} ({t0}). "
                                     "Use ref(..., 1) for yesterday's values or fill at the next open.")
    if strat.exit_when_fill == "open" and strat.exit_when:
        bad = expr.open_time_probe(strat.exit_when, dfs[t0], t0)
        if bad:
            raise ValueError(f"Lookahead: the exit rule is filled at the open but {bad} ({t0}). "
                             "Use exit_when_fill 'next_open' instead.")
    # series only some settings read are skipped otherwise (a Nasdaq-100 run computes hundreds of them)
    need_atr = bool(strat.stop_atr or strat.take_profit_atr or strat.trailing_atr or strat.sizing == "risk"
                    or strat.stop_loss or strat.take_profit or strat.trailing_stop or strat.scale_out
                    or strat.breakeven_after or strat.stop_level or strat.target_level
                    or strat.target_r or strat.trailing_points or strat.breakeven_r)   # any stop: trades report the ATR
    # stop_level / target_level: a price expression evaluated when the position opens. Without entry_price /
    # stop_price it is the same series for every trade: computed once, as known at the close (NOW) and at the open
    # (PREV: the previous bar's value, or today's when the expression is open-safe)
    LVL = {}
    for name in ("stop_level", "target_level"):
        rule = getattr(strat, name)
        if rule and not (expr.names_in(rule) & {"entry_price", "stop_price", "side"}):
            LVL[name] = (np.full((T, N), np.nan), np.full((T, N), np.nan))
    need_vol = strat.sizing == "volatility"
    need_adv = strat.slippage_model == "volume"
    state = {}
    # a Python-function entry / ranking rule on a point-in-time index universe is only read while the ticker is a
    # member: it is streamed on those days only (expr.STREAM_NEED)
    pre_member = None
    if strat.universe_name == "NDX" and strat.point_in_time and any(
            callable(r) for r in (strat.entry, strat.short_entry, strat.rank_by)):
        pre_member, _ = data.member_mask(tick, cal)
    xtables = _xrank_tables(strat, dfs, tick, cal, pre_member, per_trade_exit)
    for j, t in enumerate(tick):
        df = dfs[t]
        ns = _namespace(df, t)
        if strat.state_vars:
            # Pine `var` state: computed bar by bar from the bars up to each bar (causal), then read like any series
            from .pine_import import state_series
            state[t] = state_series(strat.state_vars, ns)
            ns = expr.Namespace(df, dict(state[t]), ticker=t)
        namespaces[t] = ns
        pos = df.index.get_indexer(cal)            # row of each calendar day in the ticker's data (-1: none)
        have = pos >= 0

        def on_cal(values, _pos=pos, _have=have):
            v = np.asarray(values, dtype=float)
            out = np.full(len(_pos), np.nan)
            out[_have] = v[_pos[_have]]
            return out

        O[:, j], H[:, j], L[:, j], C[:, j] = (on_cal(df[k].to_numpy()) for k in ("open", "high", "low", "close"))
        V[:, j] = on_cal(df["volume"].to_numpy())
        # average daily volume of the 20 bars before the order's bar (known when it is placed)
        if need_adv:
            ADV[:, j] = on_cal(df["volume"].rolling(20, min_periods=1).mean().shift(1).to_numpy())
        DIV[:, j] = np.nan_to_num(on_cal(df["dividend"].to_numpy()), nan=0.0)
        SF[:, j] = data.as_traded_factor(t, cal, df)
        if need_atr:
            ATR[:, j] = on_cal(ns["atr"](strat.atr_period).to_numpy())
        if need_vol:
            VOL[:, j] = on_cal(ns["volatility"](20).to_numpy())

        # rules acted on at this bar's own close read late-closing series (VIX, 4:15pm) as of the day before
        ns_close = ((expr.Namespace(df, dict(state[t]), ticker=t, close_fill=True) if t in state
                     else _namespace(df, t, close_fill=True)) if late_close else ns)
        if late_close:
            namespaces[t + " (close)"] = ns_close
        if xtables:
            ns.xrank_table = ns_close.xrank_table = xtables[t]

        need_j = cal[pre_member[:, j]] if pre_member is not None else None

        def on_cal_bool(rule, _pos=pos, _have=have, at_close=False, need=None):
            tok_n = expr.STREAM_NEED.set(need)
            try:
                v = expr.evaluate(rule, ns_close if at_close else ns).to_numpy(dtype=bool)   # indexed like the ticker's data
            finally:
                expr.STREAM_NEED.reset(tok_n)
            out = np.zeros(len(_pos), bool)
            out[_have] = v[_pos[_have]]
            return out

        entry_close = strat.entry_fill == "close"
        if strat.side in ("long", "both"):
            long_sig[:, j] = on_cal_bool(strat.entry, at_close=entry_close, need=need_j)
        if strat.side == "short":
            short_sig[:, j] = on_cal_bool(strat.entry, at_close=entry_close, need=need_j)
        if strat.side == "both":
            short_sig[:, j] = on_cal_bool(strat.short_entry, at_close=entry_close, need=need_j)
        if strat.exit_when and not per_trade_exit:
            exit_[:, j] = on_cal_bool(strat.exit_when, at_close=strat.exit_when_fill == "close")
        if strat.entry_level:
            LEVEL[:, j] = expr.evaluate_value(strat.entry_level, ns).reindex(cal)
        for name, (now, prev) in LVL.items():
            rule = getattr(strat, name)
            lv = expr.evaluate_value(rule, ns)
            now[:, j] = on_cal(lv.to_numpy())
            prev[:, j] = on_cal((lv if expr.open_safe(rule) else lv.shift(1)).to_numpy())
        if strat.rank_by:
            tok_n = expr.STREAM_NEED.set(need_j)
            try:
                r = expr.evaluate_value(strat.rank_by, ns_close if strat.entry_fill == "close" else ns).reindex(cal)
            finally:
                expr.STREAM_NEED.reset(tok_n)
        else:  # default preference: most liquid (20-day average dollar volume)
            r = (df["close"] * df["volume"]).rolling(20, min_periods=1).mean().reindex(cal)
        rank[:, j] = r.fillna(-np.inf if not strat.rank_ascending else np.inf).to_numpy()
    # notes the rule evaluation earned (Python-function rules: bar-by-bar evaluation and its timing; crypto lag)
    for ns_ in namespaces.values():
        for n_ in ns_.notes:
            if (n_.startswith(("Python rule", "Warning: Python rule", "Stale data:")) or n_ == expr.FRED_CLOSE_NOTE or n_ == expr.CRYPTO_LAG_NOTE or "closes at 4:15pm" in n_) \
                    and not any(x.split(":")[0] == n_.split(":")[0] for x in strat.notes):
                strat.notes.append(n_)
    valid = ~np.isnan(C)
    long_sig &= valid
    short_sig &= valid
    # opens that were never quoted (mutual funds, simulated series, old index data) can't be traded at
    OK = np.column_stack([(dfs[t]["open_ok"] if "open_ok" in dfs[t] else pd.Series(True, index=dfs[t].index))
                          .reindex(cal).fillna(False).to_numpy(dtype=bool) for t in tick])
    import re as _re
    for rule in (strat.entry, strat.short_entry, strat.exit_when):
        if isinstance(rule, str):
            for other in set(_re.findall(r"""sym\(\s*["']([^"']+)["']\s*\)\.open""", rule)):
                od = data.load(other)
                w = od.loc[(od.index >= cal[0]) & (od.index <= cal[-1])]
                if "open_ok" in w and len(w) and w["open_ok"].mean() < 0.5:
                    raise ValueError(f"The rule uses {other}'s open, but {other} has no real opening prices (only daily "
                                     "closes, e.g. a mutual fund or simulated series). Use its close instead.")
    # delisted / acquired: a ticker whose data ends well before the run's last day is sold at its last close
    # (see run); it takes no new entries from that day on, and an index universe drops it from membership
    delist = np.full(N, -1)
    for j, t in enumerate(tick):
        has_j = np.flatnonzero(valid[:, j])
        if len(has_j) and has_j[-1] < T - 1 and dfs[t].index[-1] < cal[-1] - pd.Timedelta(days=7):
            delist[j] = has_j[-1]
            long_sig[delist[j]:, j] = False
            short_sig[delist[j]:, j] = False

    ndx = strat.universe_name == "NDX" and strat.point_in_time
    member = np.ones((T, N), bool)
    if ndx:
        member, first = data.member_mask(tick, cal)
        member &= valid
        long_sig &= member
        short_sig &= member
        if strat.rank_by and "market_cap" in strat.rank_by:
            ranked_ok = np.isfinite(rank) & member
            n_el = member.sum(axis=1)
            thin = (n_el > 0) & (ranked_ok.sum(axis=1) < data.MCAP_MIN_COVERAGE * n_el)
            if thin.any():
                k = int(np.argmax(thin))
                strat.notes.append(f"Thin ranking: on {int(thin.sum())} day(s) fewer than {data.MCAP_MIN_COVERAGE:.0%} of the "
                                   f"index members had a {strat.rank_by} value (first {cal[k].date()}: "
                                   f"{int(ranked_ok[k].sum())} of {int(n_el[k])}), so the ranking there covers a subset.")
    if strat.entry_fill in ("open", "next_open") and strat.entry_order == "market":
        fill_ok = OK if strat.entry_fill == "open" else np.vstack([OK[1:], np.ones((1, N), bool)])
        # judged only on the bars where the ticker can be traded (for an index universe: while a member), so a
        # never-member or recycled-ticker series in the universe file cannot fail the whole run
        live = valid & member
        dead = [t for j, t in enumerate(tick) if live[:, j].any() and not OK[live[:, j], j].any()]
        if dead and not ndx:
            later = []
            for t in dead:
                try:
                    ok_t = data.load(t)["open_ok"]
                except Exception:  # noqa: BLE001 - a synthetic or test frame
                    continue
                ok_t = ok_t[ok_t.index > cal[-1]]
                if ok_t.any():
                    later.append(f"{t} from {ok_t.index[int(np.argmax(ok_t.to_numpy()))].date()}")
            if later:
                raise ValueError(f"{', '.join(dead)} has no quoted opening prices in this period: its early records' opens "
                                 "were filled in from the close (open = close on most days while the price moved "
                                 "several percent a day), so an open fill would really be the close. Real opens start "
                                 f"later ({', '.join(later)}). Trade it at the close, or start after that.")
            raise ValueError(f"{', '.join(dead)} has no real opening prices (only a daily close, e.g. a mutual fund or a "
                             "simulated series), so it can't be bought at the open. Trade it at the close instead.")
        if dead:
            strat.notes.append(f"Opens: {', '.join(dead[:8])}{' and more' if len(dead) > 8 else ''} had no quoted opening "
                               "prices while in the index, so they were never bought at the open.")
        if (~fill_ok & (long_sig | short_sig)).any() and not any(n.startswith("Opens:") for n in strat.notes):
            strat.notes.append("Opens: some bars have no quoted opening price (old data); entries at the open are skipped on those days.")
        long_sig &= fill_ok
        short_sig &= fill_ok

    if ndx:
        cov = data.coverage_note(str(cal[0].date()), str(cal[-1].date()))
        if cov and not any(n.startswith("Survivorship:") for n in strat.notes):
            strat.notes.append(cov)

    elif N > 1:
        live = (np.cumsum(valid, axis=0) > 0).sum(axis=1)   # started trading (a later delisting still counts)
        need = max(2, int(0.5 * N))
        thin = np.flatnonzero(live < need)
        if len(thin) and thin[-1] > 20 and not any(n.startswith("Coverage:") for n in strat.notes):
            strat.notes.append(
                f"Coverage: until {cal[thin[-1]].date()} fewer than half of the {N} tickers had price history, "
                f"so early results come from a small subset. Add 'since <year>' to focus on the period with full coverage.")
    return dict(dfs=dfs, cal=cal, tick=tick, O=O, H=H, L=L, C=C, V=V, DIV=DIV, ATR=ATR, VOL=VOL, ADV=ADV, SF=SF,
                LEVEL=LEVEL, LVL=LVL, long_sig=long_sig, short_sig=short_sig, exit_sig=exit_, rank=rank,
                per_trade_exit=per_trade_exit, namespaces=namespaces, delist=delist, traded=traded, state=state)


# ------------------------------------------------------------------ simulation

def run(strat: Strategy) -> Result:
    strat.validate()
    P = _prepare(strat)
    cal, tick = P["cal"], P["tick"]
    # annual borrow fee per ticker for shorts: the one given, else the assumed default (margin.default_borrow_fee)
    BF = np.array([_margin.borrow_fee_of(t, strat.borrow_fee) for t in tick])
    if strat.borrow_fee is None and strat.side != "long":
        bn = _margin.borrow_note(tick)
        if bn not in strat.notes:
            strat.notes.append(bn)
    O, H, L, C, V, DIV, ATR, VOL = P["O"], P["H"], P["L"], P["C"], P["V"], P["DIV"], P["ATR"], P["VOL"]
    LEVEL, long_sig, short_sig, exit_sig, rank = P["LEVEL"], P["long_sig"], P["short_sig"], P["exit_sig"], P["rank"]
    namespaces = P["namespaces"]
    ADV = P["ADV"]
    SF = P["SF"]
    T, N = C.shape
    if strat.tv_compat:
        # TradingView sizes, rounds to whole shares, charges per-contract commission and lists trades in its chart's
        # units: prices adjusted for splits (not for dividends), so a share before a later split is a fraction of a
        # share as traded then. The default mode keeps share counts as traded (for commissions and whole shares).
        split = [tick[j] for j in range(N) if np.nanmax(np.abs(SF[:, j] - 1)) > 1e-9]
        if split:
            strat.notes = [n for n in strat.notes if not n.startswith("TradingView units:")]
            strat.notes.append(
                f"TradingView units: {', '.join(split[:5])}{' and more' if len(split) > 5 else ''} split during the test. "
                "As TradingView does, quantities, whole-share rounding, per-share commissions and the trade list use its "
                "chart's split-adjusted prices and share counts (TradingView's default chart is adjusted for splits, "
                "not for dividends), not the prices and share counts as traded on the day; without TradingView mode "
                "the trades are sized and listed in shares as traded.")
        SF = np.ones_like(SF)
    slip = strat.slippage_bps / 1e4
    # a same-bar open fill whose rule reads that open: filled open_reaction_bps worse than the print (strategy.py)
    react_rules = strat.open_reaction_rules()
    react = float(strat.open_reaction_bps) / 1e4

    def react_for(sgn: int) -> float:
        key = "short entry" if strat.side == "both" and sgn < 0 else "entry"
        return react if key in react_rules else 0.0
    react_exit = react if "exit" in react_rules else 0.0
    rate = _daily_rate(cal, strat.cash_rate)
    warm_i = _warmup_bar(strat, cal, tick, C, namespaces, long_sig, short_sig)
    if warm_i:
        # as a portfolio starts trading on its warm-up day with the starting capital, the account earns nothing while
        # the rules cannot fire yet: the statistics (which start on the warm-up day) start at the starting capital
        rate = rate.copy()
        rate[: warm_i + 1] = 0.0
    has = ~np.isnan(C)
    last_bar = np.array([np.flatnonzero(has[:, j])[-1] if has[:, j].any() else -1 for j in range(N)])
    delist = P["delist"]
    delisted: list[str] = []
    last_close = np.full(N, np.nan)

    S = dict(cash=strat.capital, interest=0.0, halted=False, small=0, addon_skipped=0, nofunds=0, nofunds_days=[],
             entry_stop=np.nan, no_risk=0, wrong_side=0, lvl_nan=0,
             tv_sb=None, tv_nofunds=0, tv_nofunds_days=[], so_zero=0, so_zero_days=[])
    slot_days: set = set()                          # days an entry signal found no free position slot
    positions: dict[int, Position] = {}
    pending_mkt_open: list[tuple[int, int]] = []    # (k, sign) market orders for the next open
    pending_mkt_close: list[tuple[int, int]] = []   # (k, sign) for the next close
    pending_lvl: list[dict] = []                    # limit / stop entry orders
    trades: list[dict] = []
    equity = np.zeros(T)
    exposure = np.zeros(T)
    npos = np.zeros(T, int)
    in_mkt = np.zeros(T, bool)
    hold_w = np.zeros((T, N))
    bar_gross = [0.0]
    margin_calls: list = []

    def price_or_last(k: int, prices: np.ndarray) -> float:
        px = prices[k]
        return px if not np.isnan(px) else last_close[k]

    def mark(prices: np.ndarray) -> float:
        v = S["cash"]
        for p in positions.values():
            px = prices[p.k]
            if px != px:          # NaN: no bar today, the last close
                px = last_close[p.k]
            v += p.sign * p.shares * px
        return v

    def gross(prices: np.ndarray) -> float:
        g = 0
        for p in positions.values():
            px = prices[p.k]
            if px != px:
                px = last_close[p.k]
            g += p.shares * px
        return g

    def note_gross(prices: np.ndarray) -> None:
        bar_gross[0] = max(bar_gross[0], gross(prices))

    def commission(shares: float, value: float, f: float = 1.0) -> float:
        """Costs of an order of `shares` split-adjusted shares worth `value`; per-share fees, minimums and caps
        apply to the shares as traded that day (shares / f, f = SF[i, k]: data.split_factor_after)."""
        return costs.order_commission(shares / f, value, per_order=strat.commission, per_share=strat.commission_per_share,
                                      pct=strat.commission_pct, model=strat.commission_model)

    def whole(shares: float, f: float) -> float:
        """Round down to whole shares as traded that day (f split-adjusted shares make one as-traded share)."""
        return np.floor(shares / f + 1e-9) * f

    def slip_for(i: int, k: int, shares: float) -> float:
        """Slippage per side as a fraction: fixed bps, plus (volume model) half the spread and a square-root
        market impact, impact_bps * sqrt(shares / ADV20)."""
        if strat.slippage_model != "volume":
            return slip
        return costs.volume_slippage(slip, shares, float(ADV[i, k]), strat.spread_bps, strat.impact_bps)

    def tv_percent_shares(i: int, k: int, fill: float, eq: float, prices: np.ndarray, sgn: int, at_open: bool) -> float:
        """TradingView's strategy.percent_of_equity (and strategy.cash): the quantity is computed on the bar the order is
        placed (the signal bar), from the equity (or the cash amount) and the close of that bar, rounded down to whole shares for stocks and ETFs; the order
        then fills at the next open. An order the cash can't pay for at the fill is skipped (-1), not cut."""
        sb = S["tv_sb"]
        eq_sig = eq if sb >= i else equity[sb]
        px = C[sb, k]
        if not (np.isfinite(px) and px > 0 and np.isfinite(fill) and fill > 0) or eq_sig <= 0:
            return 0.0
        shares = (eq_sig * strat.position_size if strat.sizing == "percent" else strat.fixed_amount) / px
        f = SF[i, k]
        vol_known = V[i - 1, k] if at_open and i > 0 else (np.nan if at_open else V[i, k])
        if strat.max_volume_pct and vol_known > 0:
            shares = min(shares, strat.max_volume_pct * vol_known)
        if not strat.fractional_shares:
            shares = whole(shares, f)
        if shares <= 0:
            return 0.0
        value = shares * fill
        room = strat.leverage * eq - gross(prices)
        cost = value + (commission(shares, value, f) if sgn == 1 else 0.0)
        avail = S["cash"] + (strat.leverage - 1) * max(eq, 0) if sgn == 1 else np.inf
        if value > room + 1e-9 or cost > avail + 1e-9:
            S["tv_short"] = True
            return -1.0
        return shares

    def size_shares(i: int, k: int, fill: float, eq: float, prices: np.ndarray, sgn: int, at_open: bool) -> float:
        # indicators used for sizing must be known when the order is placed
        ib = i - 1 if (at_open and i > 0) else i
        if strat.tv_compat and strat.sizing in ("percent", "fixed_dollars") and S["tv_sb"] is not None:
            return tv_percent_shares(i, k, fill, eq, prices, sgn, at_open)
        if strat.sizing == "percent":
            value = eq * strat.position_size
        elif strat.sizing == "fixed_dollars":
            value = strat.fixed_amount
        elif strat.sizing == "fixed_shares":
            value = strat.fixed_amount * SF[i, k] * fill    # a number of shares as traded that day
        elif strat.sizing == "risk":
            if strat.stop_loss or strat.stop_atr:
                dist = fill * strat.stop_loss if strat.stop_loss else strat.stop_atr * ATR[ib, k]
            elif strat.stop_level:   # the distance to the stop level evaluated for this entry
                dist = sgn * (fill - S["entry_stop"])
            else:   # only a trailing stop: its starting distance is the risk per share
                dist = fill * strat.trailing_stop if strat.trailing_stop else strat.trailing_atr * ATR[ib, k]
            if not np.isfinite(dist) or dist <= 0:
                return 0.0
            value = eq * strat.risk_per_trade / dist * fill
        else:  # volatility
            v = VOL[ib, k]
            if not np.isfinite(v) or v <= 0:
                return 0.0
            value = eq * strat.target_vol / v
        if strat.sizing in ("fixed_dollars", "fixed_shares"):
            # a stated size is honoured exactly or not at all: like TradingView, an order the account can't pay for
            # (cash, or the leverage allowed, including the commission) is skipped - never silently cut (counted and
            # reported as a warning). Fixed shares are whole shares.
            if not np.isfinite(fill) or fill <= 0:
                return 0.0
            f = SF[i, k]                   # split-adjusted shares per share as traded that day
            if strat.sizing == "fixed_shares":
                shares = float(strat.fixed_amount)
                if not strat.fractional_shares:
                    shares = float(np.floor(shares + 1e-9))
                shares *= f                    # the stated count is in shares as traded
            else:
                shares = strat.fixed_amount / fill
                if not strat.fractional_shares:
                    shares = whole(shares, f)
            vol_known = V[i - 1, k] if at_open else V[i, k]
            if at_open and i == 0:
                vol_known = np.nan
            if strat.max_volume_pct and vol_known > 0 and shares > strat.max_volume_pct * vol_known:
                return -1.0
            value = shares * fill
            room = strat.leverage * eq - gross(prices)
            cost = value + (commission(shares, value, f) if sgn == 1 else 0.0)
            avail = S["cash"] + (strat.leverage - 1) * max(eq, 0) if sgn == 1 else np.inf
            if shares <= 0:
                return 0.0
            if value > room + 1e-9 or cost > avail + 1e-9:
                return -1.0
            return shares
        # gross exposure cap and (for longs) cash / margin availability
        room = strat.leverage * eq - gross(prices)
        value = min(value, room)
        if sgn == 1:
            avail = S["cash"] + (strat.leverage - 1) * max(eq, 0)
            value = min(value, avail)
        if value <= 0 or not np.isfinite(fill) or fill <= 0:
            return 0.0
        shares = value / fill
        # volume cap: an order at the open can only know the previous bar's volume
        vol_known = V[i - 1, k] if at_open else V[i, k]
        if at_open and i == 0:
            vol_known = np.nan
        if strat.max_volume_pct and vol_known > 0:
            shares = min(shares, strat.max_volume_pct * vol_known)
        f = SF[i, k]
        if not strat.fractional_shares:
            shares = whole(shares, f)
        if sgn == 1 and shares * fill + commission(shares, shares * fill, f) > S["cash"] + (strat.leverage - 1) * max(eq, 0) + 1e-9:
            avail = S["cash"] + (strat.leverage - 1) * max(eq, 0)
            shares = max(0.0, (avail - strat.commission) / (fill * (1 + strat.commission_pct) + strat.commission_per_share / f))
            if strat.commission_model:  # one fixed-point step: the fee of a larger order is an upper bound
                shares = max(0.0, (avail - commission(shares, shares * fill, f)) / fill)
            if not strat.fractional_shares:
                shares = whole(shares, f)
        return max(shares, 0.0)

    def open_pos(i: int, k: int, px: float, at_open: bool, sgn: int, prices: np.ndarray) -> bool:
        if S["halted"]:
            return False
        fill = px * (1 + sgn * slip)
        eq = mark(prices)
        if eq <= 0:
            return False
        new = k not in positions
        lv = entry_levels(i, k, fill, at_open, sgn) if new and levels_used else (np.nan, np.nan)
        if new and ((strat.stop_level and not np.isfinite(lv[0])) or (strat.target_level and not np.isfinite(lv[1]))):
            S["lvl_nan"] += 1   # the level is not defined yet (an indicator still warming up): no entry without it
            return False
        S["entry_stop"] = lv[0] if new else (positions[k].lvl_stop if k in positions else np.nan)
        shares = size_shares(i, k, fill, eq, prices, sgn, at_open)
        if shares > 0 and strat.slippage_model != "fixed":
            fill = px * (1 + sgn * slip_for(i, k, shares))  # impact of the order size, then re-size at that price
            if new and levels_used and level_dyn:
                lv = entry_levels(i, k, fill, at_open, sgn)
                S["entry_stop"] = lv[0]
            shares = size_shares(i, k, fill, eq, prices, sgn, at_open)
        if shares < 0:   # a fixed size the account could not pay for: skipped, as TradingView does
            if S.pop("tv_short", False):
                S["tv_nofunds"] += 1
                S["tv_nofunds_days"].append(cal[i].date())
            else:
                S["nofunds"] += 1
                S["nofunds_days"].append(cal[i].date())
            return False
        if shares <= 0 or shares * fill < max(strat.min_order or 0.0, 1e-9):
            # no dust: an order worth less than min_order (e.g. the leftover cash of a full position) is skipped
            if k in positions:
                S["addon_skipped"] += 1
            elif shares > 0:
                S["small"] += 1
            return False
        com = commission(shares, shares * fill, SF[i, k])
        S["cash"] -= sgn * shares * fill + com
        lot = Lot(shares, fill, i, at_open, com)
        if k in positions:  # pyramiding
            positions[k].lots.append(lot)
            note_gross(prices)
            return True
        # hold N: exit exactly N bars after the entry bar at the configured exit fill, whatever the entry fill
        hb = strat.hold_bars
        if hb is None:
            due, due_open = None, False
        else:
            due, due_open = i + hb, strat.hold_exit_fill == "open"
        atr_ref = ATR[i - 1, k] if at_open and i > 0 else ATR[i, k]
        p = Position(k, sgn, [lot], i, px, atr_ref, due, due_open)
        if levels_used:
            p.lvl_stop = lv[0]
            p.lvl_tgt = lv[1]
            if strat.target_r:    # R-multiple: the initial risk is the distance from the entry to the initial stop
                stop0 = initial_stop(p, fill)
                risk = sgn * (fill - stop0) if np.isfinite(stop0) else np.nan
                if np.isfinite(risk) and risk > 0:
                    r_tgt = fill + sgn * float(strat.target_r) * risk
                    p.lvl_tgt = r_tgt if not np.isfinite(p.lvl_tgt) else (min(p.lvl_tgt, r_tgt) if sgn == 1 else max(p.lvl_tgt, r_tgt))
                else:
                    S["no_risk"] += 1
            if np.isfinite(p.lvl_stop) and (fill - p.lvl_stop) * sgn <= 0:
                S["wrong_side"] += 1
        if P["per_trade_exit"]:
            ns = namespaces[tick[k]]
            dfk = ns.df
            idx = dfk.index
            pos_in_df = idx.get_indexer([cal[i]])[0]
            after = idx > cal[i] if not at_open else idx >= cal[i]
            hs = dfk["high"].where(after).cummax()
            ls = dfk["low"].where(after).cummin()
            extra = {**P["state"].get(tick[k], {}),
                     "bars_held": pd.Series(np.arange(len(idx)) - pos_in_df, index=idx, dtype=float),
                     "entry_price": fill,
                     "pnl": sgn * (dfk["close"] / fill - 1),
                     "highest_since_entry": np.maximum(hs.fillna(fill), fill),
                     "lowest_since_entry": np.minimum(ls.fillna(fill), fill)}
            p.exit_sig = expr.evaluate(strat.exit_when, expr.Namespace(dfk, extra, ticker=tick[k], close_fill=strat.exit_when_fill == "close")).reindex(cal, fill_value=False).to_numpy()
        if stops_used:
            stop0 = initial_stop(p, fill)
            p.risk = sgn * (fill - stop0) if np.isfinite(stop0) and sgn * (fill - stop0) > 0 else np.nan
        if strat.scale_out and not strat.stop_covers_scale_outs:
            # each scale-out's tranche is set aside at the entry (its fraction of what is left, in level order)
            left = lot.shares
            for j, so in sorted(enumerate(strat.scale_out), key=lambda x: so_level(p, x[1]) * sgn if np.isfinite(so_level(p, x[1])) else np.inf):
                q = left * float(so["fraction"])
                if not strat.fractional_shares:
                    q = whole(q, SF[i, k])
                p.tranche[j] = q
                left -= q
        if dyn_levels:
            p.dyn_stop, p.dyn_tgt = dynamic_levels(k, fill, sgn, p)
        p.atr_now = atr_ref
        p.arm_peak = px
        if stops_used:
            p.init_stop, p.init_trail, p.init_tgt = split_levels(p)
        positions[k] = p
        note_gross(prices)
        return True

    def close_part(i: int, p: Position, px: float, reason: str, at_open: bool, fraction: float = 1.0,
                   qty: float | None = None) -> None:
        """Close `fraction` of every lot, or exactly `qty` shares (split-adjusted) taken from the lots first in, first
        out (a whole-share scale-out)."""
        if qty is not None:
            tot = p.shares
            if qty >= tot - 1e-9:
                qty, fraction = None, 1.0
            else:
                left_q, per = qty, []
                for lot in p.lots:
                    q_ = min(lot.shares, left_q)
                    per.append(q_ / lot.shares if lot.shares else 0.0)
                    left_q -= q_
                fraction = qty / tot
        fill = px * (1 - p.sign * slip_for(i, p.k, p.shares * fraction))
        a = p.entry_bar + (0 if p.lots[0].at_open else 1)
        b = i - (1 if at_open else 0)
        hi = np.nanmax(H[a:b + 1, p.k]) if b >= a else np.nan
        lo = np.nanmin(L[a:b + 1, p.k]) if b >= a else np.nan
        hi = fill if np.isnan(hi) else max(hi, fill)
        lo = fill if np.isnan(lo) else min(lo, fill)
        remaining = []
        for n_lot, lot in enumerate(p.lots):
            f_lot = fraction if qty is None else per[n_lot]
            if f_lot <= 0:
                remaining.append(lot)
                continue
            q = lot.shares * f_lot
            com = commission(q, q * fill, SF[i, p.k])
            S["cash"] += p.sign * q * fill - com
            share_in = lot.commission * f_lot
            inc = lot.income * f_lot
            pnl = p.sign * q * (fill - lot.price) - share_in - com + inc
            cost = q * lot.price
            if p.sign == 1:
                mfe, mae = hi / lot.price - 1, lo / lot.price - 1
            else:
                mfe, mae = 1 - lo / lot.price, 1 - hi / lot.price
            trades.append({
                "ticker": tick[p.k], "side": "long" if p.sign == 1 else "short",
                "entry_date": cal[lot.bar].date(),
                "entry_fill": lot.fill_kind or ("open" if lot.at_open else "close"),
                # shares and prices as traded (not adjusted for later splits); a split while held changes the
                # count: exit_shares x exit_price = shares x entry_price x (1 + price return)
                "entry_price": lot.price * SF[lot.bar, p.k],
                "exit_date": cal[i].date(),
                "exit_fill": "open" if at_open else ("intraday" if reason in INTRADAY_REASONS else "close"),
                "exit_price": fill * SF[i, p.k], "shares": q / SF[lot.bar, p.k], "exit_shares": q / SF[i, p.k],
                "entry_split_factor": SF[lot.bar, p.k], "exit_split_factor": SF[i, p.k],
                "position_value": cost, "pnl": pnl,
                "return": pnl / cost if cost else 0.0, "bars_held": i - lot.bar, "exit_reason": reason,
                "mae": mae, "mfe": mfe, "commission": share_in + com, "income": inc,
                **({k: (v * SF[lot.bar, p.k] if v is not None else v)          # as traded, like entry_price
                    for k, v in (("stop_level", p.init_stop), ("trail_level", p.init_trail),
                                 ("target_level", p.init_tgt), ("atr_at_entry", p.atr_at_entry))}
                   if stops_used else {}),
            })
            if f_lot < 1.0 - 1e-12:
                lot.shares -= q
                if qty is not None and not strat.fractional_shares:
                    lot.shares = whole(lot.shares, SF[i, p.k])     # whole shares stay whole (no float residue)
                lot.commission -= share_in
                lot.income -= inc
                remaining.append(lot)
        if track_levels and p.lv_path:
            level_paths[(tick[p.k], str(cal[p.entry_bar].date()))] = list(p.lv_path)
        if fraction >= 1.0 or not remaining:
            del positions[p.k]
        else:
            p.lots = remaining

    def fill_profit(i: int, p: Position, px: float, x, at_open: bool) -> None:
        """A profit-side resting order fills at px: a scale-out (whole shares as traded unless fractional shares are
        allowed; rounded down, the remainder stays in the position, and a scale-out of 0 shares is skipped) or the
        take-profit target (the rest of the position)."""
        lvl, what, j, frac = x
        if j is None:
            exit_rest(i, p, px, what, at_open)
            return
        p.scaled.add(j)
        if j in p.tranche:
            q = p.tranche.pop(j)
        elif strat.fractional_shares:
            close_part(i, p, px, what, at_open, fraction=frac)
            return
        else:
            q = whole(p.shares * frac, SF[i, p.k])
        if q <= 1e-12:
            S["so_zero"] += 1
            S["so_zero_days"].append(cal[i].date())
            return
        close_part(i, p, px, what, at_open, qty=q)

    def exit_rest(i: int, p: Position, px: float, why: str, at_open: bool) -> None:
        """The stop / trailing stop / target fills: the whole position, or - when the stop does not cover the scale-out
        tranches (TradingView: a strategy.exit with qty_percent and no stop) - the shares not reserved for them."""
        if p.tranche and not strat.stop_covers_scale_outs:
            rest = p.shares - sum(p.tranche.values())
            p.rest_done = True
            if rest > 1e-9:
                close_part(i, p, px, why, at_open, qty=rest)
            return
        close_part(i, p, px, why, at_open)

    def tv_path(i: int, p: Position, o_: float, h_: float, l_: float, c_: float) -> None:
        """TradingView's broker emulator inside a bar: the price moves open -> high -> low -> close when the open is
        nearer the high, else open -> low -> high -> close. On each leg towards profit the resting limit orders
        (scale-outs, then the target) fill in the order the price reaches them and the trailing stop ratchets with the
        price; on each leg against the position the stop - at its level updated so far - fills when the leg reaches it,
        so a trailing stop raised by the high can exit later in the same bar."""
        s, k = p.sign, p.k
        high_first = abs(h_ - o_) < abs(o_ - l_)
        pts = (o_, h_, l_, c_) if high_first else (o_, l_, h_, c_)
        # an entry filled at this open with its stop (or a level) already beyond the open: fills at the open
        stop, why, tgt = levels(p)
        if stop is not None and (o_ - stop) * s <= 0:
            exit_rest(i, p, o_, why, False)
            if k not in positions:
                return
        _, _, tgt = levels(p)
        for x in profit_levels(p, tgt):
            if k not in positions or (o_ - x[0]) * s < 0:
                break
            fill_profit(i, p, o_, x, False)
        p.peak = max(p.peak, o_) if s == 1 else min(p.peak, o_)
        for a, b in zip(pts[:-1], pts[1:]):
            if k not in positions:
                return
            if (b - a) * s > 0:            # towards profit
                _, _, tgt = levels(p)
                for x in profit_levels(p, tgt):
                    if k not in positions or (b - x[0]) * s < 0:
                        break
                    fill_profit(i, p, x[0] if (a - x[0]) * s < 0 else a, x, False)
                if k in positions:
                    p.peak = max(p.peak, b) if s == 1 else min(p.peak, b)
            elif (b - a) * s < 0:          # against the position
                stop, why, _ = levels(p)
                if stop is not None and (b - stop) * s <= 0:
                    exit_rest(i, p, stop if (a - stop) * s > 0 else a, why, False)

    def ranked(cands: list[int], i: int) -> list[int]:
        return sorted(cands, key=lambda k: rank[i, k], reverse=not strat.rank_ascending)

    def want(i: int, k: int, sgn: int, prices: np.ndarray, at_open: bool) -> None:
        """Handle an entry signal for ticker k in direction sgn."""
        if np.isnan(prices[k]):
            return
        if 0 <= delist[k] <= i and not at_open:
            return  # its last day of data: nothing is bought at (or after) its final close
        p = positions.get(k)
        if p is not None and p.sign != sgn:
            if strat.side == "both" and strat.reverse:
                close_part(i, p, prices[k], "reversal", at_open)
            else:
                return
        p = positions.get(k)
        if p is not None:
            if len(p.lots) < strat.pyramiding and p.lots[-1].bar != i:
                open_pos(i, k, prices[k], at_open, sgn, prices)
            return
        if sibling_held(k, i):
            return
        if len(positions) >= strat.max_positions:
            slot_days.add(i)
            return
        open_pos(i, k, prices[k], at_open, sgn, prices)

    sib = {}
    if getattr(strat, "share_classes", "company") == "company" and N > 1:
        grp = {t: g for g in data.SHARE_CLASSES for t in g}
        for k, t in enumerate(tick):
            g = grp.get(t)
            if g:
                sib[k] = [j for j, u in enumerate(tick) if j != k and grp.get(u) == g]
    sibling_days: dict = {}

    def sibling_held(k: int, i: int) -> bool:
        """Share classes of one company (GOOG/GOOGL, ...) take one position slot: no second class while one is held."""
        for j in sib.get(k, ()):
            if j in positions:
                sibling_days.setdefault("/".join(sorted((tick[k], tick[j]))), set()).add(i)
                return True
        return False

    def signals(i: int) -> list[tuple[int, int]]:
        out = [(k, 1) for k in np.flatnonzero(long_sig[i])] + [(k, -1) for k in np.flatnonzero(short_sig[i])]
        return out

    def ranked_pairs(pairs: list[tuple[int, int]], i: int) -> list[tuple[int, int]]:
        return sorted(pairs, key=lambda kp: rank[i, kp[0]], reverse=not strat.rank_ascending)

    def placeable(k: int, sgn: int) -> bool:
        """TradingView pyramiding: an entry order generated while the ticker's position already has its maximum
        number of entries is ignored, not kept for later. So a next-open order from a bar on which the position
        was still held never fills after a stop closes the position at that open. A position closed at this
        bar's close no longer counts (a signal on the exit bar is valid); one only scheduled to exit at the next
        open still does. An opposite signal counts only if it reverses the position."""
        p = positions.get(k)
        if p is None:
            return True
        if p.sign != sgn:
            return strat.side == "both" and strat.reverse
        return len(p.lots) < strat.pyramiding

    stops_used = any([strat.stop_loss, strat.stop_atr, strat.trailing_stop, strat.trailing_atr,
                      strat.take_profit, strat.take_profit_atr, strat.scale_out, strat.breakeven_after,
                      strat.stop_level, strat.target_level, strat.target_r, strat.trailing_points, strat.breakeven_r])
    levels_used = bool(strat.stop_level or strat.target_level or strat.target_r)
    LVL = P["LVL"]
    level_dyn = any(getattr(strat, n) and n not in LVL for n in ("stop_level", "target_level"))
    level_open_safe = {n: bool(getattr(strat, n)) and expr.open_safe(getattr(strat, n)) for n in ("stop_level", "target_level")}

    def level_value(name: str, i: int, k: int, at_open: bool, extra: dict) -> float:
        """stop_level / target_level for an entry filled on bar i: with the data known at the fill - the bar itself for
        a fill at the close, the previous bar for a fill at (or after) the open unless the expression is open-safe."""
        rule = getattr(strat, name)
        if not rule:
            return np.nan
        if name in LVL:
            v = LVL[name][1 if at_open else 0][i, k]
            return float(v) if np.isfinite(v) else np.nan
        ns = namespaces[tick[k]]
        dfk = ns.df
        pos_i = dfk.index.get_indexer([cal[i]])[0]
        if pos_i < 0:
            return np.nan
        rb = pos_i - 1 if at_open and not level_open_safe[name] else pos_i
        if rb < 0:
            return np.nan
        st = {n_: v_.iloc[: pos_i + 1] for n_, v_ in P["state"].get(tick[k], {}).items()}
        v = expr.evaluate_value(rule, expr.Namespace(dfk.iloc[: pos_i + 1], {**st, **extra}, ticker=tick[k])).iloc[rb]
        return float(v) if np.isfinite(v) else np.nan

    dyn_levels = bool(strat.dynamic_levels and (strat.stop_level or strat.target_level))
    # levels that move in ways the report's chart can't rebuild from the entry: recorded bar by bar
    track_levels = stops_used and bool(strat.tv_compat or dyn_levels or strat.trailing_points or strat.trail_activation
                                       or strat.trail_activation_points or strat.trail_activation_r or strat.breakeven_r or strat.current_atr
                                       or not strat.stop_covers_scale_outs)
    level_paths: dict = {}

    def dynamic_levels(k: int, fill: float, sgn: int, p: Position):
        """dynamic_levels: the stop / target level expression on every bar of the calendar (entry_price = the fill,
        stop_price = the initial stop). Bar i uses the value at bar i-1's close, as TradingView re-evaluates a
        strategy.exit() on every bar's close for the next bar: causal."""
        out = []
        for name in ("stop_level", "target_level"):
            rule = getattr(strat, name)
            if not rule:
                out.append(None)
                continue
            if name in LVL:
                out.append(LVL[name][0][:, k].copy())
                continue
            ns = namespaces[tick[k]]
            extra = {**P["state"].get(tick[k], {}), "entry_price": fill, "side": float(sgn)}
            if name == "target_level":
                extra["stop_price"] = initial_stop(p, fill)
            v = expr.evaluate_value(rule, expr.Namespace(ns.df, extra, ticker=tick[k])).reindex(cal)
            out.append(v.to_numpy(dtype=float))
        return out[0], out[1]

    def atr_of(p: Position, kind: str) -> float:
        """The ATR an ATR-based exit uses: at the entry, or (current_atr) as of the previous close, which TradingView's
        strategy.exit(..., stop = ... ta.atr(14)) re-evaluates on every bar. current_atr None: only the chandelier /
        ATR trailing stop follows the current ATR, and only in TradingView mode."""
        cur = strat.current_atr
        if cur is None:
            cur = strat.tv_compat and kind == "trail"
        return p.atr_now if cur and np.isfinite(p.atr_now) else p.atr_at_entry

    def trail_armed(p: Position) -> bool:
        """The trailing stop is live: after the first scale-out ('trail the rest'), and once the best price reached
        the activation distance (TradingView's trail_points / trail_price)."""
        if strat.trail_after_scale_out and not p.scaled:
            return False
        s, e = p.sign, p.avg_price
        if strat.trail_activation and (p.peak - e * (1 + s * strat.trail_activation)) * s < 0:
            return False
        if strat.trail_activation_points and (p.peak - (e + s * strat.trail_activation_points)) * s < 0:
            return False
        if strat.trail_activation_r and not (np.isfinite(p.risk) and (p.peak - (e + s * strat.trail_activation_r * p.risk)) * s >= 0):
            return False
        return True

    def trail_levels(p: Position) -> list:
        """(level, reason) of the trailing stops that are live."""
        if not trail_armed(p):
            return []
        s, out = p.sign, []
        if strat.trailing_stop:
            out.append((p.peak * (1 - s * strat.trailing_stop), "trailing stop"))
        a = atr_of(p, "trail")
        if strat.trailing_atr and np.isfinite(a):
            out.append((p.peak - s * strat.trailing_atr * a, "chandelier stop"))
        if strat.trailing_points:
            out.append((p.peak - s * strat.trailing_points, "trailing stop"))
        return out

    def initial_stop(p: Position, e: float) -> float:
        """The stop a new position starts with: the tightest fixed stop (stop loss, ATR stop, stop level), else the
        trailing stop's starting distance. NaN when there is none."""
        s = p.sign
        fixed = [x for x in ((e * (1 - s * strat.stop_loss)) if strat.stop_loss else None,
                             (e - s * strat.stop_atr * p.atr_at_entry) if strat.stop_atr and np.isfinite(p.atr_at_entry) else None,
                             p.lvl_stop if np.isfinite(p.lvl_stop) else None) if x is not None]
        if not fixed:
            fixed = [x for x in ((e * (1 - s * strat.trailing_stop)) if strat.trailing_stop else None,
                                 (e - s * strat.trailing_atr * p.atr_at_entry) if strat.trailing_atr and np.isfinite(p.atr_at_entry) else None,
                                 (e - s * strat.trailing_points) if strat.trailing_points else None)
                     if x is not None]
        if not fixed:
            return np.nan
        return max(fixed) if s == 1 else min(fixed)

    def entry_levels(i: int, k: int, fill: float, at_open: bool, sgn: int) -> tuple[float, float]:
        """(stop level, target level) of a position opening on bar i at `fill`; the target may use stop_price (the
        initial stop: the tightest of the stop level and the percentage / ATR stops)."""
        stop = level_value("stop_level", i, k, at_open, {"entry_price": fill, "side": float(sgn)})
        tgt = np.nan
        if strat.target_level:
            atr = ATR[i - 1, k] if at_open and i > 0 else ATR[i, k]
            probe = Position(k, sgn, [], i, fill, atr, None, False, lvl_stop=stop)
            tgt = level_value("target_level", i, k, at_open, {"entry_price": fill, "side": float(sgn),
                                                              "stop_price": initial_stop(probe, fill)})
        return stop, tgt

    def split_levels(p: Position):
        """(fixed stop, trailing stop, target) at this moment, NaN where not used: the chart draws them."""
        s, e = p.sign, p.avg_price
        a = atr_of(p, "stop")
        fixed = [x for x in ((e * (1 - s * strat.stop_loss)) if strat.stop_loss else None,
                             (e - s * strat.stop_atr * a) if strat.stop_atr and np.isfinite(a) else None,
                             p.lvl_stop if np.isfinite(p.lvl_stop) else None)
                 if x is not None]
        trail = [lv for lv, _ in trail_levels(p)]
        pick = (lambda xs: max(xs) if s == 1 else min(xs))
        ungated[0] = True
        try:
            _, _, tgt = levels(p)
        finally:
            ungated[0] = False
        return (pick(fixed) if fixed else np.nan, pick(trail) if trail else np.nan, tgt if tgt is not None else np.nan)

    def so_level(p: Position, so: dict) -> float:
        """A scale-out's price: a gain from the average entry price, or an R multiple of the initial risk (NaN when the
        trade has no risk to measure)."""
        if so.get("r") is not None:
            return p.avg_price + p.sign * float(so["r"]) * p.risk if np.isfinite(p.risk) else np.nan
        if so.get("points") is not None:     # a price distance (TradingView: profit = ticks, limit = avg price + x)
            return p.avg_price + p.sign * float(so["points"])
        return p.avg_price * (1 + p.sign * float(so["at"]))

    def profit_levels(p: Position, tgt) -> list:
        """The profit-side resting orders of a position, in the order the price reaches them (ascending for a long,
        descending for a short): (level, reason, scale-out index or None, fraction of what is left). Scale-outs are
        limit orders like the target; at the same level the scale-out comes first."""
        s = p.sign
        out = [(so_level(p, so), "scale out", j, float(so["fraction"]))
               for j, so in enumerate(strat.scale_out or []) if j not in p.scaled and np.isfinite(so_level(p, so))
               and placed(p, "scale_out" if so.get("after_fill") else "")]
        if tgt is not None:
            out.append((tgt, "take profit", None, 1.0))
        out.sort(key=lambda x: (x[0] * s, x[2] is None))
        return out

    AFTER_FILL = set(strat.exits_after_fill or []) | {"scale_out"}   # a scale-out says so itself (after_fill)
    ungated = [False]        # the levels a position will have (the trade list's stop / target), placed or not yet

    def placed(p: Position, kind: str) -> bool:
        """exits_after_fill: an exit order the script places from strategy.position_avg_price exists only from the bar
        after the first close in the position (an open fill's own bar, a close fill's next bar, are without it)."""
        if kind not in AFTER_FILL or ungated[0]:
            return True
        return i >= p.entry_bar + (1 if p.lots[0].at_open else 2)

    def levels(p: Position):
        """(stop level or None, stop reason, target level or None)"""
        s, e = p.sign, p.avg_price
        stop, why = None, ""
        if p.rest_done:          # only scale-out tranches left, which the stop and target do not cover
            return None, "", None
        cands = []
        a = atr_of(p, "stop")
        fixed_ok = placed(p, "stop")
        if strat.stop_loss and fixed_ok:
            cands.append((e * (1 - s * strat.stop_loss), "stop loss"))
        if strat.stop_atr and np.isfinite(a) and fixed_ok:
            cands.append((e - s * strat.stop_atr * a, "ATR stop"))
        if np.isfinite(p.lvl_stop) and fixed_ok:
            cands.append((p.lvl_stop, "trailing stop" if strat.stop_ratchet else "stop level"))
        cands += trail_levels(p)      # "trail the rest": after the first scale-out; after the activation distance
        arm = p.arm_peak if np.isfinite(p.arm_peak) else p.peak
        if strat.breakeven_after and (arm - e * (1 + s * strat.breakeven_after)) * s >= 0:
            cands.append((e, "breakeven stop"))    # armed once the best price reached +breakeven_after
        elif strat.breakeven_r and np.isfinite(p.risk) and (arm - (e + s * strat.breakeven_r * p.risk)) * s >= 0:
            cands.append((e, "breakeven stop"))    # ... or breakeven_r x the initial risk
        if cands:  # the tightest stop (closest to price) triggers first
            stop, why = max(cands, key=lambda c: c[0] * s)
        tgt = None
        if not placed(p, "target"):
            return stop, why, None
        if strat.take_profit:
            tgt = e * (1 + s * strat.take_profit)
        if strat.take_profit_atr and np.isfinite(a):
            t2 = e + s * strat.take_profit_atr * a
            tgt = t2 if tgt is None else (min(tgt, t2) if s == 1 else max(tgt, t2))
        if np.isfinite(p.lvl_tgt):
            t2 = p.lvl_tgt
            tgt = t2 if tgt is None else (min(tgt, t2) if s == 1 else max(tgt, t2))
        return stop, why, tgt

    for i in range(T):
        o, h, lo_, c = O[i], H[i], L[i], C[i]
        bar_gross[0] = 0.0
        had_position = bool(positions)

        # ---- 0. OVERNIGHT: interest, borrow fees, dividends
        if i > 0:
            r = rate[i - 1]
            if S["cash"] >= 0:
                earned = S["cash"] * r
                if r > 0 and strat.short_rebate_spread:
                    # short sale proceeds (part of cash) earn the rate less the rebate spread, floored at zero
                    short_mv = sum(p.shares * price_or_last(p.k, C[i - 1]) for p in positions.values() if p.sign == -1)
                    earned -= min(short_mv, S["cash"]) * min(r, strat.short_rebate_spread / 252.0)
            else:
                earned = S["cash"] * (r + strat.margin_rate / 252.0)
            S["cash"] += earned
            S["interest"] += earned
        for p in positions.values():
            k = p.k
            if i > 0 and p.sign == -1 and BF[k]:
                fee = p.shares * price_or_last(k, C[i - 1]) * BF[k] / 252.0
                S["cash"] -= fee
                for lot in p.lots:
                    lot.income -= fee * lot.shares / p.shares
            if DIV[i, k] > 0 and strat.dividends is not False:
                for lot in p.lots:
                    if lot.bar < i:  # held into the ex-date
                        amt = p.sign * lot.shares * DIV[i, k]
                        S["cash"] += amt
                        lot.income += amt

        # ---- exit levels that follow the market: known at the previous close
        if i > 0 and positions and (dyn_levels or strat.current_atr is not False):
            for p in positions.values():
                if p.entry_bar >= i:
                    continue
                a_ = ATR[i - 1, p.k]
                if np.isfinite(a_):
                    p.atr_now = a_
                if p.dyn_stop is not None and np.isfinite(p.dyn_stop[i - 1]):
                    v_ = float(p.dyn_stop[i - 1])
                    if strat.stop_ratchet and np.isfinite(p.lvl_stop):
                        v_ = max(v_, p.lvl_stop) if p.sign == 1 else min(v_, p.lvl_stop)   # only in the position's favour
                    p.lvl_stop = v_
                if p.dyn_tgt is not None and np.isfinite(p.dyn_tgt[i - 1]):
                    p.lvl_tgt = float(p.dyn_tgt[i - 1])

        # ---- 1. OPEN
        for p in list(positions.values()):
            k = p.k
            if np.isnan(o[k]) or (p.entry_bar == i):
                continue
            if p.pending_open_exit or (p.exit_due_bar is not None and p.exit_due_open and i >= p.exit_due_bar):
                close_part(i, p, o[k], p.pending_reason or "time exit", at_open=True)
                continue
            if strat.exit_when and strat.exit_when_fill == "open":  # open-safe rule, acted on at today's open
                if p.exit_sig[i] if p.exit_sig is not None else exit_sig[i, k]:
                    close_part(i, p, o[k] * (1 - p.sign * react_exit), "exit rule", at_open=True)
                    continue
            if stops_used:
                stop, why, tgt = levels(p)
                if stop is not None and (o[k] - stop) * p.sign <= 0:
                    exit_rest(i, p, o[k], why, True)
                    if p.rest_done and k in positions:
                        tgt = None      # the tranches left keep their own limit orders
                    else:
                        continue
                # profit-side resting orders the open already passed (a gap): each fills at the open, nearest first
                # (a scale-out below the target before the target, which then closes what is left)
                for x in profit_levels(p, tgt):
                    if (o[k] - x[0]) * p.sign < 0 or k not in positions:
                        break
                    fill_profit(i, p, o[k], x, True)
        # market entries at the open
        todays_open = []
        if strat.entry_fill == "open" and strat.entry_order == "market":
            todays_open = ranked_pairs(signals(i), max(i - 1, 0))  # rank with yesterday's values
        for n_, (k, sgn) in enumerate(pending_mkt_open + todays_open):
            S["tv_sb"] = i - 1 if n_ < len(pending_mkt_open) and i > 0 else None   # the order's signal bar
            r_ = react_for(sgn) if n_ >= len(pending_mkt_open) else 0.0
            if r_ and not np.isnan(o[k]):
                px_ = o.copy()
                px_[k] = o[k] * (1 + sgn * r_)       # reacting to the printed open: a little worse than the print
                want(i, k, sgn, px_, at_open=True)
                continue
            want(i, k, sgn, o, at_open=True)
        S["tv_sb"] = None
        pending_mkt_open = []

        # ---- limit / stop entry orders (open, then intraday)
        still = []
        for od in pending_lvl:
            k, sgn, lvl = od["k"], od["sign"], od["level"]
            if i > od["expires"] or S["halted"]:
                continue
            if np.isnan(o[k]) or (k in positions and positions[k].sign == sgn and len(positions[k].lots) >= strat.pyramiding):
                still.append(od)
                continue
            fill = None
            buyish = sgn == 1
            if strat.entry_order == "limit":
                if (o[k] <= lvl) if buyish else (o[k] >= lvl):
                    fill, at_o = o[k], True
                elif (lo_[k] <= lvl) if buyish else (h[k] >= lvl):
                    fill, at_o = lvl, False
            else:  # stop entry
                if (o[k] >= lvl) if buyish else (o[k] <= lvl):
                    fill, at_o = o[k], True
                elif (h[k] >= lvl) if buyish else (lo_[k] <= lvl):
                    fill, at_o = lvl, False
            if fill is None:
                still.append(od)
                continue
            if k not in positions and sibling_held(k, i):
                still.append(od)
                continue
            if k not in positions and len(positions) >= strat.max_positions:
                slot_days.add(i)
                still.append(od)
                continue
            S["tv_sb"] = od["placed"]
            opened = open_pos(i, k, fill, True, sgn, o)
            S["tv_sb"] = None
            if opened:
                positions[k].lots[-1].at_open = True  # exposed from the fill onward
                positions[k].lots[-1].fill_kind = "open" if at_o else strat.entry_order   # gapped through, or at the level
        pending_lvl = still

        # ---- 2. INTRADAY stops / targets / scale-outs
        if stops_used:
            for p in list(positions.values()):
                k = p.k
                if np.isnan(h[k]) or (p.lots[-1].bar == i and not p.lots[-1].at_open):
                    continue
                s = p.sign
                if track_levels:
                    st_, _, tg_ = levels(p)
                    p.lv_path.append((i, np.nan if st_ is None else float(st_), np.nan if tg_ is None else float(tg_)))
                if strat.tv_compat:
                    tv_path(i, p, o[k], h[k], lo_[k], c[k])
                else:
                    adverse = lo_[k] if s == 1 else h[k]
                    best = h[k] if s == 1 else lo_[k]
                    stop, why, tgt = levels(p)
                    stop_hit = stop is not None and (adverse - stop) * s <= 0
                    if stop_hit:
                        # the stop is assumed to come first when it and a profit level are both touched (conservative)
                        exit_rest(i, p, stop if (o[k] - stop) * s > 0 else o[k], why, at_open=False)
                    else:
                        # profit-side levels in the order the price reaches them: nearest first, so a scale-out below
                        # the target fills before the target closes the rest
                        for x in profit_levels(p, tgt):
                            if k not in positions or (best - x[0]) * s < 0:
                                break
                            fill_profit(i, p, x[0] if (o[k] - x[0]) * s < 0 else o[k], x, False)
                if k in positions:
                    p.peak = max(p.peak, h[k]) if s == 1 else min(p.peak, lo_[k])
                    p.arm_peak = p.peak

        # ---- 3. CLOSE exits
        for p in list(positions.values()):
            k = p.k
            if np.isnan(c[k]):
                if i > last_bar[k]:
                    close_part(i, p, last_close[k], "data ended", at_open=False)
                continue
            if p.entry_bar == i and not p.lots[0].at_open:
                continue
            if p.exit_due_bar is not None and not p.exit_due_open and i >= p.exit_due_bar:
                close_part(i, p, c[k], "time exit", at_open=False)
                continue
            sig = p.exit_sig[i] if p.exit_sig is not None else exit_sig[i, k]
            if strat.exit_when and sig and strat.exit_when_fill != "open":
                if strat.exit_when_fill == "close":
                    close_part(i, p, c[k], "exit rule", at_open=False)
                    continue
                p.pending_open_exit, p.pending_reason = True, "exit rule"
            if delist[k] == i:
                close_part(i, p, c[k], "delisted", at_open=False)
                delisted.append(f"{tick[k]} delisted/acquired on {cal[i].date()}")
        np.copyto(last_close, c, where=~np.isnan(c))

        # ---- 3b. entries at the close / orders for tomorrow
        if not S["halted"]:
            todays = ranked_pairs(signals(i), i)
            if strat.entry_fill != "close" or strat.entry_order != "market":
                todays = [(k, sgn) for k, sgn in todays if placeable(k, sgn)]
            if strat.entry_order != "market":
                for k, sgn in todays:
                    lvl = LEVEL[i, k]
                    if np.isfinite(lvl):
                        pending_lvl = [od for od in pending_lvl if od["k"] != k]
                        pending_lvl.append({"k": k, "sign": sgn, "level": lvl, "expires": i + strat.order_valid_bars,
                                            "placed": i})
            elif strat.entry_fill == "close":
                S["tv_sb"] = i
                for k, sgn in todays:
                    want(i, k, sgn, c, at_open=False)
                S["tv_sb"] = None
            elif strat.entry_fill == "next_close":
                for k, sgn in pending_mkt_close:
                    want(i, k, sgn, c, at_open=False)
                pending_mkt_close = todays
            elif strat.entry_fill == "next_open":
                pending_mkt_open = todays

        # ---- 4. MARK
        equity[i] = mark(c)
        if equity[i] <= 0 and positions and not S["halted"]:
            for p in list(positions.values()):
                close_part(i, p, price_or_last(p.k, c), "liquidated (equity exhausted)", at_open=False)
            S["halted"] = True
            equity[i] = S["cash"]
            if not any(n.startswith("Liquidated") for n in strat.notes):
                strat.notes.append(f"Liquidated: equity fell to zero on {cal[i].date()}; trading stopped.")
        elif positions and strat.maintenance_margin and (strat.leverage > 1 or any(p.sign == -1 for p in positions.values())):
            # maintenance margin: equity must cover this fraction of gross exposure at the close; otherwise
            # every position is cut pro rata at the close until gross exposure is back to leverage x equity
            g = gross(c)
            need = sum(_margin.maintenance(tick[p.k], strat.maintenance_margin) * p.shares * c[p.k]
                       for p in positions.values() if not np.isnan(c[p.k]))
            if g > 0 and equity[i] < need * (1 - 1e-12):
                # de-risk with a cushion (margin.py): to the lower of the leverage cap and the exposure at which
                # equity is 125% of the maintenance requirement
                cut = 1.0 - _margin.call_scale(equity[i], g, need, strat.leverage)
                for p in list(positions.values()):
                    if not np.isnan(c[p.k]):
                        close_part(i, p, c[p.k], "margin call", at_open=False, fraction=min(max(cut, 0.0), 1.0))
                margin_calls.append(cal[i].date())
                equity[i] = mark(c)
        g = gross(c)
        eq_pos = equity[i] if equity[i] > 0 else np.nan
        exposure[i] = max(g, bar_gross[0]) / eq_pos if eq_pos == eq_pos else 0.0
        npos[i] = len(positions)
        in_mkt[i] = bool(positions) or had_position or bar_gross[0] > 0
        if positions and equity[i] > 0:
            for p in positions.values():
                hold_w[i, p.k] = p.sign * p.shares * price_or_last(p.k, c) / equity[i]

    if delisted:
        more = f" and {len(delisted) - 5} more" if len(delisted) > 5 else ""
        strat.notes.append(f"Delisted: {', '.join(delisted[:5])}{more}; the position was closed at its last price (the "
                           "final close in the data; trades marked 'delisted') and the proceeds were held in cash.")
    if margin_calls:
        more = f" and {len(margin_calls) - 5} more" if len(margin_calls) > 5 else ""
        strat.notes = [n for n in strat.notes if not n.startswith("Margin call")]
        strat.notes.append(f"Margin call on {', '.join(str(d) for d in margin_calls[:5])}{more}: equity fell below "
                           f"its maintenance requirement ({strat.maintenance_margin:.0%} of gross exposure; the leverage "
                           "factor times that for a leveraged ETF), so positions were cut pro rata at the close to the "
                           f"lower of {strat.leverage:g}x and the exposure at which equity is "
                           f"{_margin.MARGIN_CALL_CUSHION:.0%} of the requirement (trades marked 'margin call').")

    if sibling_days:
        strat.notes = [n for n in strat.notes if not n.startswith("Share classes:")]
        strat.notes.append("Share classes: " + ", ".join(sorted(sibling_days)) + " are classes of one company, so an entry "
                           "signal in one class was skipped while the other was held (one position per company; the "
                           "first to signal, or the higher ranked / more liquid on the same day, is taken). Set "
                           "share_classes 'separate' in the JSON spec to trade them as separate names.")
    if slot_days and N > 1 and not strat.rank_by:
        strat.notes = [n for n in strat.notes if not n.startswith("Slots:")]
        strat.notes.append(f"Slots: on {len(slot_days)} day(s) more tickers signalled than there were free position slots. "
                           "No ranking was given, so the most liquid were taken first: highest 20-day average dollar "
                           "volume (close x volume) as known when the order was placed. Say e.g. 'prefer the lowest "
                           "RSI' to choose differently.")
    if S["addon_skipped"]:
        strat.notes = [n for n in strat.notes if not n.startswith("Warning: pyramiding")]
        strat.notes.append(f"Warning: pyramiding add-ons were skipped {S['addon_skipped']} time(s): the position had no "
                           f"headroom left (its size already used the capital or leverage available), or the order would "
                           f"have been worth less than the ${strat.min_order:g} minimum. Lower the size per entry to leave room.")
    if S["nofunds"]:
        what = (f"{strat.fixed_amount:g} shares" if strat.sizing == "fixed_shares" else f"${strat.fixed_amount:,.0f}")
        days = ", ".join(str(d) for d in S["nofunds_days"][:5]) + (" and more" if len(S["nofunds_days"]) > 5 else "")
        strat.notes = [n for n in strat.notes if not n.startswith("Warning: insufficient funds")]
        strat.notes.append(f"Warning: insufficient funds: {S['nofunds']} entry order(s) of {what} were skipped ({days}): the "
                           f"account could not pay for the stated size plus commission"
                           f"{' within the ' + format(strat.leverage, 'g') + 'x leverage allowed' if strat.leverage != 1 else ''}"
                           f"{' or it exceeded the volume cap' if strat.max_volume_pct else ''}. As in TradingView, a fixed size "
                           "is filled in full or not at all (never cut to what the cash allows). Lower the size, add "
                           "capital, or size as a percentage of equity.")
    if S["tv_nofunds"]:
        days = ", ".join(str(d) for d in S["tv_nofunds_days"][:5]) + (" and more" if len(S["tv_nofunds_days"]) > 5 else "")
        strat.notes = [n for n in strat.notes if not n.startswith("TradingView orders skipped:")]
        size = f"{strat.position_size:.0%} of equity" if strat.sizing == "percent" else f"${strat.fixed_amount:,.0f}"
        strat.notes.append(f"TradingView orders skipped: {S['tv_nofunds']} entry order(s) were skipped ({days}): the quantity, "
                           f"{size} at the signal bar's close, cost more than the cash "
                           "available when the order filled (the price gapped up by the next open). TradingView's broker "
                           "emulator skips such an order rather than cutting it; use a smaller size (e.g. 95% of equity) "
                           "to leave room.")
    if S["lvl_nan"]:
        strat.notes = [n for n in strat.notes if not n.startswith("Levels:")]
        strat.notes.append(f"Levels: {S['lvl_nan']} entry signal(s) were skipped because the stop / target level "
                           f"({strat.stop_level or strat.target_level}) was not defined yet (its indicator was still warming up).")
    if S["no_risk"] and strat.target_r:
        strat.notes = [n for n in strat.notes if not n.startswith("R target:")]
        strat.notes.append(f"R target: on {S['no_risk']} entr{'y' if S['no_risk'] == 1 else 'ies'} the stop was at or beyond the "
                           "entry price (no risk to measure), so no R-multiple target was set for those trades.")
    if S["wrong_side"] and strat.stop_level:
        strat.notes = [n for n in strat.notes if not n.startswith("Stop level:")]
        strat.notes.append(f"Stop level: on {S['wrong_side']} entr{'y' if S['wrong_side'] == 1 else 'ies'} {strat.stop_level} was "
                           "at or beyond the entry price, so the stop was already crossed: it filled at the next check "
                           "(the next open), as a broker's stop order would.")
    if S["so_zero"]:
        days = ", ".join(str(d) for d in S["so_zero_days"][:5]) + (" and more" if len(S["so_zero_days"]) > 5 else "")
        strat.notes = [n for n in strat.notes if not n.startswith("Scale-outs skipped:")]
        strat.notes.append(f"Scale-outs skipped: {S['so_zero']} scale-out(s) came to less than one whole share ({days}), so "
                           "no order was placed and the shares stayed in the position (scale-outs are rounded down to whole "
                           "shares; say 'fractional shares' to allow fractions).")
    if S["small"]:
        strat.notes = [n for n in strat.notes if not n.startswith("Min order:")]
        strat.notes.append(f"Min order: {S['small']} entry order(s) worth less than ${strat.min_order:g} were skipped.")

    # the state after the last bar: what is open, and which exits / stop levels apply at the next session
    # (signals.scan turns this into tomorrow's orders)
    open_state = []
    for p in positions.values():
        stop, why, tgt = levels(p) if stops_used else (None, "", None)
        so_levels = [{**({k_: float(so[k_]) for k_ in ("r", "points", "at") if so.get(k_) is not None}),
                      "fraction": float(so["fraction"]), "level": float(so_level(p, so))}
                     for j, so in enumerate(strat.scale_out or []) if j not in p.scaled and np.isfinite(so_level(p, so))]
        open_state.append({
            "ticker": tick[p.k], "side": "long" if p.sign == 1 else "short", "shares": float(p.shares),
            "avg_price": float(p.avg_price), "entry_date": str(cal[p.entry_bar].date()), "entries": len(p.lots),
            "has_last_bar": bool(last_bar[p.k] == T - 1),
            "pending_open_exit": bool(p.pending_open_exit), "pending_reason": p.pending_reason,
            "hold_bars_left": None if p.exit_due_bar is None else int(p.exit_due_bar - (T - 1)),
            "hold_exit_open": bool(p.exit_due_open),
            "stop": None if stop is None else float(stop), "stop_reason": why,
            "target": None if tgt is None else float(tgt), "scale_out": so_levels,
        })

    # close anything still open at the last bar
    for p in list(positions.values()):
        close_part(T - 1, p, last_close[p.k], "open at end", at_open=False)
    equity[-1] = S["cash"]

    tr = pd.DataFrame(trades)
    if not tr.empty:
        tr = tr.sort_values(["exit_date", "entry_date", "ticker"], kind="stable").reset_index(drop=True)
        tr.index = tr.index + 1
        tr["cum_pnl"] = tr["pnl"].cumsum()

    # prepend the starting capital on the previous session so returns include the first bar
    start = _cal.anchor_day(cal[0], P["traded"])
    idx = pd.DatetimeIndex([start]).append(cal)
    eq = pd.Series(np.concatenate([[strat.capital], equity]), index=idx, name="equity")
    ex = pd.Series(np.concatenate([[0.0], exposure]), index=idx, name="exposure")
    npo = pd.Series(np.concatenate([[0], npos]), index=idx, name="positions")
    inm = pd.Series(np.concatenate([[False], in_mkt]), index=idx, name="in_market")
    hw = pd.DataFrame(hold_w, index=cal, columns=tick)
    hw = hw.loc[:, (hw != 0).any()]
    if tr is not None and len(tr):
        spun = []
        for t in tr["ticker"].unique():
            g = tr[tr["ticker"] == t]
            for d in _corp_days("spin", t, P["dfs"].get(t)):
                if ((pd.to_datetime(g["entry_date"]) < d) & (pd.to_datetime(g["exit_date"]) >= d)).any():
                    spun.append(f"{t} {d.date()}")
        if spun:
            strat.notes.append(f"Distributions: {', '.join(spun[:5])}{' and more' if len(spun) > 5 else ''} paid a spin-off "
                               "or special distribution (more than 15% of the price; e.g. shares of a spun-off company "
                               "booked at their value). It is paid in cash like a dividend (in the trades' income) but it "
                               "is a spin-off distribution, not a dividend.")
        for t in tr["ticker"].unique():
            days = _corp_days("action", t, P["dfs"].get(t))
            if not len(days):
                continue
            g = tr[tr["ticker"] == t]
            hit = [d for d in days if ((pd.to_datetime(g["entry_date"]) < d) & (pd.to_datetime(g["exit_date"]) >= d)).any()]
            if hit:
                strat.notes.append(f"Data: {t}'s prices on {', '.join(str(d.date()) for d in hit)} don't reconcile with its "
                                   "total return (an unadjusted spin-off, split or special dividend); trades held over that "
                                   "day may be misstated.")
    res = Result(strategy=strat, equity=eq, trades=tr, exposure=ex, positions=npo, prices=P["dfs"],
                 holdings=hw, interest=S["interest"], in_market=inm)
    # the entry / exit rules' value on every bar (what the simulation acted on), for the report's rule-state strip;
    # an exit rule that uses the position (bars_held, entry_price...) has no per-bar value outside a trade
    res.extras["open_state"] = open_state
    if level_paths:
        # each trade's stop and target as they stood at the start of every bar it was held (split-adjusted prices, as the
        # chart draws them), keyed "ticker|entry date"
        res.extras["level_paths"] = {
            f"{t_}|{d_}": {"d": [str(cal[j].date()) for j, _, _ in pth],
                           "s": [None if not np.isfinite(a) else round(a, 6) for _, a, _ in pth],
                           "t": [None if not np.isfinite(b) else round(b, 6) for _, _, b in pth]}
            for (t_, d_), pth in level_paths.items()}
    # limit / stop entry orders still working after the last bar: the engine tries to fill them from the next
    # session on (signals.scan / broker.plan send exactly these, at these levels)
    res.extras["pending_entries"] = [
        {"ticker": tick[od["k"]], "side": "long" if od["sign"] == 1 else "short", "order": strat.entry_order,
         "level": float(od["level"]), "placed": str(cal[od["placed"]].date()),
         "sessions_left": int(od["expires"] - (T - 1))}
        for od in pending_lvl if od["expires"] >= T and not S["halted"]]
    res.extras["rule_state"] = {"cal": cal, "tick": tick, "entry": long_sig | short_sig,
                                "exit": None if P["per_trade_exit"] or not strat.exit_when else exit_sig}
    return res
