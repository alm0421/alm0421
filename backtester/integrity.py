"""Price-integrity gate: splits missing from the source data, splits booked but not applied (or applied to a split
that never happened), and isolated bad ticks. Called by data.load on every price file; scripts/fetch_data.py logs
what it finds to data/inferred_splits.json.

Free price data (Yahoo) sometimes gets a split wrong, and its adjusted close is no check on that: it is computed from
the same close, so it jumps with it. What is left to go on is the prices themselves:

- A split changes the price level by a whole ratio (2, 3, 1/10, 1/20, ...) overnight and for good, while the day's
  real move - the part the ratio leaves over - matches what the market did that day. The day's move is predicted from
  the reference ticker (SPY, TLT, SMH, ^VIX, NVDA, ... - the one most correlated with the ticker over the previous
  120 sessions) scaled by the ticker's beta to it, with the residual volatility as the yardstick. A forward split
  also multiplies the share volume (a reverse split divides it), so where volume is reported it must shift the same
  way. Real crashes (SVXY on 2018-02-06, -83%) fail these tests: the level change is no whole ratio of a plausible
  move, and the volume did not follow.
- A bad tick is one bar far (25% or more) from both neighbours, which agree with each other, that the reference
  move does not explain (CPER 2014-12-04 at 13.05 between 19.52 and 19.56; the NYSE's 2024-06-03 bad prints).

Repairs: an inferred split is recorded in the split column on its day and the earlier prices back-adjusted by it (so
returns, the as-traded prices and share counts are all consistent); a booked split that the prices contradict is
dropped or applied; a bad tick is replaced by the geometric mean of its neighbours (its open is flagged as not
quoted, so no fill uses it). Every change is kept with its evidence (data.price_repairs) and any backtest over a
repaired day says so. Only the event day's return and the price basis before it change; no later bar is used to
decide a signal (the repair is about the data, like a split adjustment).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

# split ratios considered (new shares per old share): the common forward splits, and the common reverse splits
# (1-for-2 ... 1-for-200). A sparse list on purpose: with every whole number some ratio would fit any large move.
FORWARD_RATIOS = (1.5, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20)
REVERSE_RATIOS = (2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 75, 100, 200)
SPLIT_CANDIDATES = tuple(sorted({float(r) for r in FORWARD_RATIOS} | {1.0 / r for r in REVERSE_RATIOS}))
# tickers a day's expected move is read from (the best-correlated over the previous WINDOW sessions is used)
REFERENCES = ("SPY", "QQQ", "IWM", "DIA", "TLT", "IEF", "SHY", "LQD", "HYG", "GLD", "SLV", "USO", "UNG", "DBC", "XLE",
              "XOP", "XLF", "XLK", "XLV", "XLU", "XLI", "XLB", "XLP", "XLY", "IYR", "SMH", "SOXX", "GDX", "GDXJ", "EEM",
              "EFA", "FXI", "KWEB", "EWZ", "EWJ", "INDA", "XBI", "IBB", "^VIX", "VIXY", "UUP", "BTC-USD", "ETH-USD",
              "NVDA", "TSLA", "COIN", "MSTR", "AAPL", "AMZN", "META", "MSFT", "GOOGL")
WINDOW = 120            # sessions of history for the reference fit
MIN_OBS = 40
MIN_CORR = 0.3
SPLIT_JUMP = np.log(1.45)     # smallest level change looked at as a split (a 3-for-2 split is log 1.5)
TICK_JUMP = np.log(1.25)      # a bad tick is at least 25% from both neighbours
TICK_NEIGHBOURS = 0.10        # ... which agree within 10% (and within a quarter of the spike)
PERSIST = 5                   # bars after (and before) the event that must stay on their level
STALE_RUN_MAX = 3             # longest run of untraded (zero-volume) bars replaced as bad ticks

EVENT_COLUMNS = ["date", "kind", "ratio", "applied", "day_return_raw", "day_return_now", "expected", "sigma",
                 "reference", "volume_ratio", "why"]


def _ok(x) -> bool:
    return x is not None and np.isfinite(x)


class _Ctx:
    """Per-ticker working data: log closes, log returns, volumes and the reference fits."""

    def __init__(self, t: str, raw: pd.DataFrame, ref_returns: Callable[[str], pd.Series | None]):
        self.t = t
        self.raw = raw
        self.ref_returns = ref_returns
        c = raw["close"].astype(float)
        self.lc = np.log(c.where(c > 0))
        self.r = self.lc.diff()
        v = raw["volume"].astype(float) if "volume" in raw else pd.Series(0.0, index=raw.index)
        self.v = v.fillna(0.0)
        self.dates = raw.index

    def predict(self, i: int) -> tuple[float, float, str | None]:
        """(expected log return of bar i, residual sd, reference) from the best-correlated reference's move that day,
        fitted on the WINDOW bars before i; (0, own sd, None) when nothing correlates."""
        y = self.r.iloc[max(1, i - WINDOW):i]
        y = y[np.isfinite(y.to_numpy())]
        own = float(y.std()) if len(y) > 5 else 0.05
        best = None
        for ref in REFERENCES:
            if ref == self.t:
                continue
            rr = self.ref_returns(ref)
            if rr is None or not len(rr):
                continue
            x = rr.reindex(y.index)
            ok = np.isfinite(x.to_numpy()) & np.isfinite(y.to_numpy())
            if ok.sum() < MIN_OBS:
                continue
            xx, yy = x.to_numpy()[ok], y.to_numpy()[ok]
            if xx.std() <= 0 or yy.std() <= 0:
                continue
            cor = float(np.corrcoef(xx, yy)[0, 1])
            if np.isfinite(cor) and (best is None or abs(cor) > abs(best[1])):
                best = (ref, cor, xx, yy)
        if best is None or abs(best[1]) < MIN_CORR:
            return 0.0, max(own, 1e-3), None
        ref, cor, xx, yy = best
        # fitted on simple returns (a leveraged or inverse fund is linear in them, not in log returns) and
        # returned as a log return
        xs, ys = np.expm1(xx), np.expm1(yy)
        beta = float(np.cov(xs, ys)[0, 1] / np.var(xs, ddof=1))
        sd = float(np.std(ys - beta * xs, ddof=1))
        xi = self.ref_returns(ref).get(self.dates[i], np.nan)
        if not _ok(xi):
            return 0.0, max(own, 1e-3), None
        p = max(beta * float(np.expm1(xi)), -0.95)
        return float(np.log1p(p)), max(sd / (1 + p), 1e-3), ref

    def volume_ratio(self, i: int) -> float | None:
        """Median volume of the 10 bars after bar i / of the 10 before (None when either side has none)."""
        before = self.v.iloc[max(0, i - 10):i]
        after = self.v.iloc[i + 1:i + 11]
        vb, va = float(before.median()) if len(before) else 0.0, float(after.median()) if len(after) else 0.0
        if vb <= 0 or va <= 0:
            return None
        return va / vb

    def day_volume_ratio(self, i: int) -> float | None:
        """Volume on bar i / median of the 10 bars before (None without volume)."""
        before = self.v.iloc[max(0, i - 10):i]
        vb, vd = (float(before.median()) if len(before) else 0.0), float(self.v.iloc[i])
        return vd / vb if vb > 0 and vd > 0 else None

    def level_holds(self, i: int, jump: float) -> bool:
        """The new level holds for the next bars and the old one held for the bars before (a level shift, not a
        spike): the median move from bar i over the next PERSIST bars, and into bar i-1 over the PERSIST before, is
        under half the jump."""
        after = (self.lc.iloc[i + 1:i + 1 + PERSIST] - self.lc.iloc[i]).to_numpy()
        before = (self.lc.iloc[max(0, i - 1 - PERSIST):i - 1] - self.lc.iloc[i - 1]).to_numpy()
        after, before = after[np.isfinite(after)], before[np.isfinite(before)]
        if len(after) < 2:
            return False
        lim = 0.5 * abs(jump)
        return abs(float(np.median(after))) < lim and (not len(before) or abs(float(np.median(before))) < lim)


def _dividend_basis(raw: pd.DataFrame, i: int, f: float) -> bool:
    """Scale the payouts before bar i with the prices (factor f)? Yahoo often has the dividends already on the new
    basis when it missed the split in the prices (PGOVX, 2023). Decided by which reading puts the payout yields of the
    two years before in line with those of the two years after (with the prices when neither side has payouts)."""
    if "dividend" not in raw:
        return True
    d = pd.to_numeric(raw["dividend"], errors="coerce").fillna(0.0)
    c = raw["close"].astype(float)
    y = d / c.shift(1)
    day = raw.index[i]
    pre = y[(raw.index < day) & (raw.index >= day - pd.Timedelta(days=730)) & (d > 0)]
    post = y[(raw.index >= day) & (raw.index < day + pd.Timedelta(days=730)) & (d > 0)]
    pre, post = pre[np.isfinite(pre)], post[np.isfinite(post)]
    if len(pre) < 2 or len(post) < 2:
        return True
    yp, yq = float(pre.median()), float(post.median())
    if yp <= 0 or yq <= 0:
        return True
    scaled = abs(np.log(yp / yq))            # payouts scaled with the prices: yields unchanged
    unscaled = abs(np.log(yp / f / yq))      # payouts kept: yields divided by f
    return scaled <= unscaled


def _rescale_before(raw: pd.DataFrame, i: int, f: float, adj: bool, div: bool) -> None:
    """Multiply the prices before bar i by f (volumes divided by it), in place; adj_close and payouts optionally."""
    before = raw.index < raw.index[i]
    for k in ("open", "high", "low", "close"):
        if k in raw:
            raw.loc[before, k] = raw.loc[before, k] * f
    if adj and "adj_close" in raw:
        raw.loc[before, "adj_close"] = raw.loc[before, "adj_close"] * f
    if div and "dividend" in raw:
        raw.loc[before, "dividend"] = raw.loc[before, "dividend"] * f
    if "volume" in raw:
        raw.loc[before, "volume"] = raw.loc[before, "volume"] / f


def _adj_moves_with_close(raw: pd.DataFrame, i: int, raw_jump: float, true_jump: float) -> bool:
    """Did adj_close jump with the close on bar i (so it needs the same back-adjustment)?"""
    if "adj_close" not in raw:
        return False
    a = pd.to_numeric(raw["adj_close"], errors="coerce")
    a0, a1 = float(a.iloc[i - 1]), float(a.iloc[i])
    if not (a0 > 0 and a1 > 0):
        return True
    ja = np.log(a1 / a0)
    return abs(ja - raw_jump) <= abs(ja - true_jump)


def check(t: str, raw: pd.DataFrame, ref_returns: Callable[[str], pd.Series | None],
          overrides: dict | None = None, fund: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The price-integrity pass on one ticker's raw bars (after data.reconcile_actions). Returns (bars, events).
    `fund`: an ETF or mutual fund (reverse splits missing from the data are common there; for a stock, whose
    split records are reliable and whose crashes are not rare, an inferred split must leave a smaller move)."""
    empty = pd.DataFrame(columns=EVENT_COLUMNS)
    overrides = overrides or {}
    if (raw.empty or len(raw) < 30 or t.startswith("^") or t.endswith("SIM")
            or not {"open", "high", "low", "close"} <= set(raw.columns)):
        return raw, empty
    raw = raw.copy()
    for k in ("open", "high", "low", "close", "adj_close", "volume", "dividend", "split"):
        if k in raw:
            raw[k] = pd.to_numeric(raw[k], errors="coerce").astype(float)
    if "split" not in raw:
        raw["split"] = 0.0
    raw["split"] = raw["split"].fillna(0.0)
    events: list[dict] = []
    ctx = _Ctx(t, raw, ref_returns)
    ignore = {pd.Timestamp(d) for d, o in overrides.items() if o.get("action") == "ignore"}

    # ---------------------------------------------------------------- isolated bad ticks
    r = ctx.r
    nxt = -r.shift(-1)                                   # log(c_i / c_{i+1})
    span = (ctx.lc.shift(-1) - ctx.lc.shift(1)).abs()    # neighbours' disagreement
    spike = ((r.abs() >= TICK_JUMP) & (nxt.abs() >= TICK_JUMP) & (np.sign(r) == np.sign(nxt))
             & (span < np.minimum(TICK_NEIGHBOURS, 0.25 * r.abs()))).fillna(False)
    last = -2
    for i in np.flatnonzero(spike.to_numpy()):
        if i < 2 or i >= len(raw) - 1 or raw.index[i] in ignore or i == last + 1:
            continue
        if i + 2 < len(raw) and abs(float(r.iloc[i + 2])) >= TICK_JUMP:
            continue                                     # the series keeps jumping (a stale, untraded quote)
        pred, sd, ref = ctx.predict(i)
        ri = float(r.iloc[i])
        if abs(ri - pred) < max(6 * sd, TICK_JUMP) or abs(pred) > 0.3 * abs(ri):
            continue                                     # the reference moved a lot too: a real (wild) day
        vd = ctx.day_volume_ratio(i)
        if vd is not None and vd > 3.0:
            continue                                     # heavy trading: a real (if short-lived) move
        last = i
        old = float(raw["close"].iloc[i])
        new = float(np.exp(0.5 * (ctx.lc.iloc[i - 1] + ctx.lc.iloc[i + 1])))
        for k in ("open", "high", "low"):
            v = raw[k].iloc[i]
            if not (v > 0) or abs(np.log(v / new)) > TICK_JUMP:
                raw.iloc[i, raw.columns.get_loc(k)] = new
        raw.iloc[i, raw.columns.get_loc("close")] = new
        raw.iloc[i, raw.columns.get_loc("high")] = max(raw["high"].iloc[i], raw["open"].iloc[i], new)
        raw.iloc[i, raw.columns.get_loc("low")] = min(raw["low"].iloc[i], raw["open"].iloc[i], new)
        if "adj_close" in raw and raw["adj_close"].iloc[i] > 0:
            raw.iloc[i, raw.columns.get_loc("adj_close")] = raw["adj_close"].iloc[i] * new / old
        events.append({"date": raw.index[i], "kind": "bad_tick", "ratio": np.nan, "applied": "replaced",
                       "day_return_raw": np.exp(ri) - 1, "day_return_now": new / float(raw["close"].iloc[i - 1]) - 1,
                       "expected": np.exp(pred) - 1, "sigma": sd, "reference": ref, "volume_ratio": np.nan,
                       "why": (f"close {old:.6g} is {abs(np.exp(ri) - 1):.0%} from the day before and "
                               f"{abs(np.exp(-float(nxt.iloc[i])) - 1):.0%} from the day after, which agree within "
                               f"{float(np.exp(span.iloc[i]) - 1):.1%}; replaced by their geometric mean {new:.6g}")})
    # ---------------------------------------------------------------- stale runs
    # two or three bars in a row with no volume (nothing traded), all far from the bar before them and the bar after
    # them, which agree with each other: a quote that was never a trade (INDV 2022-11-23..25 at 3.13 between 19.64 and
    # 21.03). Replaced by the geometric path between the neighbours; their opens are not traded at.
    if "volume" in raw:
        lc = ctx.lc.to_numpy()
        vol = ctx.v.to_numpy()
        n = len(raw)
        with np.errstate(invalid="ignore"):
            starts = np.flatnonzero((vol <= 0) & (np.abs(np.diff(lc, prepend=np.nan)) >= TICK_JUMP))
        taken = -1
        for i in starts.tolist():
            if i < 1 or i >= n - 2 or i <= taken:
                continue
            for k in range(STALE_RUN_MAX, 1, -1):
                j = i + k                                  # the bar after the run
                if j >= n or raw.index[i] in ignore:
                    continue
                run = lc[i:j]
                a, b = lc[i - 1], lc[j]
                if not (np.isfinite(run).all() and np.isfinite(a) and np.isfinite(b)):
                    continue
                if (vol[i:j] > 0).any() or abs(b - a) >= TICK_NEIGHBOURS:
                    continue
                far = np.minimum(np.abs(run - a), np.abs(run - b))
                if far.min() < TICK_JUMP:
                    continue
                if (vol[j] <= 0 and vol[i - 1] <= 0):
                    continue                               # a series with no trading around it at all: nothing to anchor
                pred = [ctx.predict(q)[0] for q in range(i, j)]
                if abs(sum(pred)) > 0.3 * far.min():
                    continue                               # the reference moved as much: a real (wild) stretch
                was = raw["close"].iloc[i - 1:j].to_numpy(dtype=float).copy()
                s0 = float(raw["split"].iloc[i])
                phantom = ""
                if s0 > 0 and abs(s0 - 1) > 1e-9 and abs(run[0] - a - np.log(s0)) < 0.05:
                    # the run starts with a split booked at exactly its ratio, and the prices went back after it:
                    # a split that never happened (IPAR 1990-08-03, a 0.4 "split" for three untraded days)
                    raw.iloc[i, raw.columns.get_loc("split")] = 0.0
                    phantom = f"; the {_ratio_text(s0)} split booked on {raw.index[i].date()} with it is dropped"
                for q in range(i, j):
                    new = float(np.exp(a + (b - a) * (q - i + 1) / (k + 1)))
                    old = float(was[q - i + 1])
                    prev = float(raw["close"].iloc[q - 1])
                    for col in ("open", "high", "low", "close"):
                        raw.iloc[q, raw.columns.get_loc(col)] = new
                    if "adj_close" in raw and raw["adj_close"].iloc[q] > 0:
                        raw.iloc[q, raw.columns.get_loc("adj_close")] = raw["adj_close"].iloc[q] * new / old
                    events.append({"date": raw.index[q], "kind": "bad_tick", "ratio": np.nan, "applied": "replaced",
                                   "day_return_raw": old / float(was[q - i]) - 1,
                                   "day_return_now": new / prev - 1, "expected": np.nan, "sigma": np.nan,
                                   "reference": None, "volume_ratio": np.nan,
                                   "why": (f"{k} bars in a row with no volume at {np.exp(run.min()):.6g}.."
                                           f"{np.exp(run.max()):.6g}, x{np.exp(far.min()):.3g} or more away from the "
                                           f"closes around them ({np.exp(a):.6g} and {np.exp(b):.6g}, which agree "
                                           f"within {abs(np.expm1(b - a)):.1%}): quotes that never traded, replaced "
                                           "by the path between those closes" + phantom)})
                taken = j
                break
    if events:
        ctx = _Ctx(t, raw, ref_returns)

    # ---------------------------------------------------------------- splits
    r = ctx.r
    sp = raw["split"]
    booked = (sp > 0) & ((sp - 1).abs() > 1e-9)
    vol = r.rolling(60, min_periods=20).std().shift(1)
    cand = ((r.abs() >= SPLIT_JUMP) & (r.abs() > 8 * vol)).fillna(False).to_numpy().copy()
    forced = {pd.Timestamp(d): o for d, o in overrides.items() if o.get("action") in ("split", "rescale")}
    for d in forced:
        if d in raw.index:
            cand[raw.index.get_loc(d)] = True
    for i in np.flatnonzero(cand):
        day = raw.index[i]
        if i < 1 or day in ignore:
            continue
        rj = float(r.iloc[i])
        if not np.isfinite(rj):
            continue
        # a split booked within two bars of the jump that the jump does not reflect: leave to the split column
        near = booked.iloc[max(0, i - 2):i + 3]
        pred, sd, ref = ctx.predict(i)
        vr = ctx.volume_ratio(i)
        ov = forced.get(day)
        if booked.iloc[i]:
            if "dividend" in raw and raw["dividend"].iloc[i] > 0:
                continue                                     # a spin-off (split + payout): data.reconcile_actions
            s = float(sp.iloc[i])
            # (a) the prices jump BY the booked ratio: they were adjusted for a split that did not happen that day
            #     (NVDS 2023-08-09: a 1-for-5 booked twice) -> drop it and undo the adjustment
            # (b) they jump by its inverse: a real split the prices were never adjusted for -> apply it
            true_a, true_b = rj - np.log(s), rj + np.log(s)
            tol = max(4 * sd, 0.03)
            if abs(true_a - pred) <= tol and abs(true_a - pred) < abs(true_b - pred):
                kind, f, keep, true = "phantom_split", s, False, true_a
            elif abs(true_b - pred) <= tol:
                kind, f, keep, true = "unapplied_split", 1.0 / s, True, true_b
            else:
                continue
            if abs(rj - pred) < max(10 * sd, SPLIT_JUMP) or not ctx.level_holds(i, rj):
                continue
            _rescale_before(raw, i, f, adj=_adj_moves_with_close(raw, i, rj, true), div=_dividend_basis(raw, i, f))
            if not keep:
                raw.iloc[i, raw.columns.get_loc("split")] = 0.0
            events.append({"date": day, "kind": kind, "ratio": s, "applied": "dropped" if not keep else "applied",
                           "day_return_raw": np.exp(rj) - 1, "day_return_now": np.exp(true) - 1,
                           "expected": np.exp(pred) - 1, "sigma": sd, "reference": ref, "volume_ratio": vr,
                           "why": (f"a {_ratio_text(s)} split is booked on {day.date()} but the prices "
                                   + ("already moved by it (the earlier prices were adjusted for a split that did not "
                                      "happen then): split dropped, earlier prices restored"
                                      if not keep else "were never adjusted for it: earlier prices back-adjusted"))})
            ctx = _Ctx(t, raw, ref_returns)
            r = ctx.r
            continue
        if near.any():
            continue
        if ov is None and "dividend" in raw and raw["dividend"].iloc[i] > 0:
            # a payout that day (a spin-off paid in cash, or re-read as one by data.reconcile_actions): judged on the
            # total return, which a split would move but a distribution does not (HTLD 2002-02-20)
            tr_day = np.log((float(raw["close"].iloc[i]) + float(raw["dividend"].iloc[i]))
                            / float(raw["close"].iloc[i - 1]))
            if abs(tr_day) < SPLIT_JUMP:
                continue
        if ov is not None and ov.get("ratio"):
            s = float(ov["ratio"])
        else:
            s = min(SPLIT_CANDIDATES, key=lambda x: abs(rj + np.log(x) - pred))
        true = rj + np.log(s)
        if ov is None:
            # the move left over must match the reference closely (tracking error grows with the day's move); a
            # stock's split data is rarely wrong while crashes are common, so a stock must also be left with a
            # small move in absolute terms
            gap = abs(true - pred)
            if gap > max(2.5 * sd, 0.1 * abs(pred), 0.005) or gap > 0.35 * abs(np.log(s)) or (not fund and gap > 0.03):
                continue                                     # the ratio does not leave a plausible day
            if abs(rj - pred) < max((10 if ref else 15) * sd, SPLIT_JUMP) or not ctx.level_holds(i, rj):
                continue                                     # not extreme, or not a lasting level change
            vd = ctx.day_volume_ratio(i)
            if vr is not None and vd is not None:
                # a forward split (s > 1) multiplies the share volume by about s from its day on, a reverse one
                # divides it; a crash instead brings a burst of volume on the day itself
                q, qd = np.log(vr) / np.log(s), np.log(vd) / np.log(s)
                if not (0.5 <= q <= 2.0 and -0.5 <= qd <= 2.0):
                    continue
            elif gap > max(2 * sd, 0.02) or abs(rj - pred) < 15 * sd:
                continue                                     # no volume to confirm it: demand a clearer case
        record = ov is None or ov.get("action") == "split"
        f = 1.0 / s
        _rescale_before(raw, i, f, adj=_adj_moves_with_close(raw, i, rj, true), div=_dividend_basis(raw, i, f))
        if record:
            raw.iloc[i, raw.columns.get_loc("split")] = s
        why = (ov.get("why") if ov is not None and ov.get("why") else
               f"the close moved {np.exp(rj) - 1:+.1%} on {day.date()} with no split in the source data; a "
               f"{_ratio_text(s)} split leaves {np.exp(true) - 1:+.1%}"
               + (f", in line with {ref} ({np.exp(pred) - 1:+.1%} expected, residual sd {sd:.1%})" if ref else
                  f" (no reference moves with it; its daily sd is {sd:.1%})")
               + (f"; volume x{vr:.3g} after" if vr is not None else ""))
        events.append({"date": day, "kind": "inferred_split" if record else "basis_change", "ratio": s,
                       "applied": "recorded" if record else "rescaled", "day_return_raw": np.exp(rj) - 1,
                       "day_return_now": np.exp(true) - 1, "expected": np.exp(pred) - 1, "sigma": sd,
                       "reference": ref, "volume_ratio": vr, "why": why})
        ctx = _Ctx(t, raw, ref_returns)
        r = ctx.r
    out = pd.DataFrame(events, columns=EVENT_COLUMNS)
    if len(out):
        out = out.sort_values("date").reset_index(drop=True)
    return raw, out


def _ratio_text(s: float) -> str:
    """4.0 -> '4-for-1', 0.05 -> '1-for-20', 1.5 -> '3-for-2'."""
    if s >= 1:
        return f"{s:g}-for-1" if float(s).is_integer() else f"{2 * s:g}-for-2"
    inv = 1.0 / s
    return f"1-for-{round(inv):g}" if abs(inv - round(inv)) < 1e-6 else f"2-for-{2 * inv:g}"


# ------------------------------------------------------------------ security breaks (bankruptcy / re-listing splices)
#
# A company that goes through Chapter 11 usually has its old shares cancelled; the reorganised company lists NEW
# shares, often under the same symbol, and free price sources splice the two series: CHRD (Oasis Petroleum) closed at
# $0.12 on 2020-11-19 - the old shares, days before their cancellation - and at $31.00 on 2020-11-20, the first day
# of the new shares (x258). That is not a return anybody earned: the old holders were (all but) wiped out, and the new
# shares are another security. A "security break" marks the first bar of the new series:
#   - a position held into it is closed at the old security's last price (the close before the break) - the market's
#     own value of what the old holders were left with (warrants, a small recovery, or nothing) - booked on the break
#     day at no cost; the proceeds are cash (an allocation buys the new security at its next rebalance);
#   - indicators do not span it: the engines evaluate rules on the bars before and after it separately;
#   - it is decided from the bars up to and including the break day only, never a later bar, so truncating the data
#     at any date gives the same breaks before it (the no-lookahead invariant). On the day before the break nothing
#     is known and nothing happens: the position is marked at that close, which is then also its exit price, so the
#     equity up to that day is the same whether or not the data goes on.
# Detected where the evidence is decisive (find_breaks); a known case can be forced or suppressed with an override in
# data/inferred_splits.json ("action": "break" / "no_break"). A large move that is not decisive is flagged instead
# (backtester/price_flags.py): backtests holding across it get a warning.

BREAK_JUMP = np.log(10.0)     # the close rises 10x or more overnight ...
BREAK_COLLAPSE = 0.10         # ... from a price at most 10% of its high over the BREAK_LOOKBACK sessions before
BREAK_LOOKBACK = 252
BREAK_MIN_HISTORY = 20
BREAK_REVERSE_SPLIT = 0.5     # a reverse split by R divides the share volume by about R; a splice does not (the day's
                              # volume falls by less than R ** 0.5 against the 10 sessions before) ...
BREAK_MAX_DAY_VOLUME = 3.0    # ... nor is it a burst of trading in the same shares (news on a penny stock)
BREAK_GAP_DAYS = 10           # with no volume to go on: calendar days without a bar before it (the old line stopped)

BREAK_COLUMNS = ["date", "kind", "prev_close", "close", "jump", "prior_high", "volume_ratio", "gap_days", "source",
                 "why"]


def find_breaks(t: str, df: pd.DataFrame, overrides: dict | None = None) -> pd.DataFrame:
    """Security breaks in a ticker's bars (data.load's output, or any bars with a close and optionally volume / split
    columns): one row per break with its evidence (BREAK_COLUMNS). Causal: bar i is judged on bars 0..i only."""
    overrides = overrides or {}
    if df is None or len(df) < 2 or "close" not in df or t.startswith("^") or t.endswith("SIM"):
        return pd.DataFrame(columns=BREAK_COLUMNS)
    c = pd.to_numeric(df["close"], errors="coerce").astype(float)
    lc = np.log(c.where(c > 0))
    jump = lc.diff()
    v = (pd.to_numeric(df["volume"], errors="coerce").fillna(0.0).astype(float) if "volume" in df
         else pd.Series(0.0, index=df.index))
    sp = (pd.to_numeric(df["split"], errors="coerce").fillna(0.0) if "split" in df
          else pd.Series(0.0, index=df.index))
    split_day = ((sp > 0) & ((sp - 1).abs() > 1e-9)).to_numpy()
    high = c.rolling(BREAK_LOOKBACK, min_periods=BREAK_MIN_HISTORY).max().shift(1)
    forced = {pd.Timestamp(d): o for d, o in overrides.items() if isinstance(o, dict) and o.get("action") == "break"}
    vetoed = {pd.Timestamp(d) for d, o in overrides.items()
              if isinstance(o, dict) and o.get("action") in ("no_break", "ignore")}
    cand = ((jump >= BREAK_JUMP) & (c.shift(1) <= BREAK_COLLAPSE * high)).fillna(False).to_numpy()
    idx = df.index
    rows = set(np.flatnonzero(cand).tolist()) | {int(idx.get_loc(d)) for d in forced if d in idx}
    out = []
    for i in sorted(rows):
        if i < 1:
            continue
        day = idx[i]
        if day in vetoed:
            continue
        pc, cc = float(c.iloc[i - 1]), float(c.iloc[i])
        before = v.iloc[max(0, i - 10):i]
        before = before[before > 0]
        vb = float(before.median()) if len(before) else 0.0
        vd = float(v.iloc[i]) / vb if vb > 0 and v.iloc[i] > 0 else None
        gap = int((day - idx[i - 1]).days)
        j = float(jump.iloc[i]) if np.isfinite(jump.iloc[i]) else 0.0
        ov = forced.get(day)
        if ov is None:
            if split_day[max(0, i - 2):i + 1].any():
                continue                  # a booked split within two bars: a reverse split, not a new security
            if vd is not None:
                if np.log(vd) <= -BREAK_REVERSE_SPLIT * j or vd >= BREAK_MAX_DAY_VOLUME:
                    continue              # volume fell like a reverse split's, or a burst of trading: flagged instead
            elif gap < BREAK_GAP_DAYS:
                continue                  # nothing but the price to go on: flagged instead
        hi = float(high.iloc[i]) if np.isfinite(high.iloc[i]) else np.nan
        why = (ov.get("why") if ov is not None and ov.get("why") else
               f"the close went from {pc:.4g} to {cc:.4g} (x{np.exp(j):.0f}) overnight, after the price had collapsed "
               f"to {pc / hi:.1%} of its {hi:.4g} high of the year before"
               + (f"; share volume x{vd:.2g} of the 10 sessions before (a reverse split would divide it by "
                  f"~{np.exp(j):.0f})" if vd is not None else f"; no volume, {gap} days without a bar before it")
               + ": the old shares' series spliced with a new security's (a bankruptcy re-listing)")
        out.append({"date": day, "kind": "security_break", "prev_close": pc, "close": cc,
                    "jump": float(np.exp(j)) - 1, "prior_high": hi, "volume_ratio": vd, "gap_days": gap,
                    "source": "override" if ov is not None else "detected", "why": why})
    return pd.DataFrame(out, columns=BREAK_COLUMNS)
