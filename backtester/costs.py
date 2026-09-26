"""Trading costs shared by the signal engine (engine.py) and the allocation engine (portfolio.py).

One place for every per-order fee and slippage formula, so both engines charge the same thing:

- `order_commission`: $ per order + $ per share + % of value + a broker preset (IBKR Pro fixed / tiered).
- `volume_slippage`: the square-root market-impact model (half the quoted spread plus
  impact_bps * sqrt(order shares / 20-day average volume)), on top of fixed slippage.

Share counts: prices in this project are split-adjusted, so a share count is in today's (adjusted) shares.
Per-share fees are only exact in as-traded shares (before later splits). `traded_shares` is the hook for that:
it returns the adjusted count unchanged for now; a data helper that knows the split history can convert here,
and both engines pick it up.
"""
from __future__ import annotations

import math

# broker presets: (per share, minimum per order, maximum as a fraction of trade value, pass-through fees per share)
COMMISSION_MODELS = {
    None: None,
    "ibkr_fixed": (0.005, 1.00, 0.01, 0.0),       # IBKR Pro Fixed: $0.005/share, min $1, max 1% of trade value
    "ibkr_tiered": (0.0035, 0.35, 0.01, 0.0002),  # IBKR Pro Tiered (first tier) plus ~$0.0002/share exchange,
                                                  # clearing and regulatory fees (an approximation)
}

SLIPPAGE_MODELS = ("fixed", "volume")


def traded_shares(shares: float, ticker: str | None = None, date=None) -> float:
    """The number of shares a broker would see for an order of `shares` adjusted shares of `ticker` on `date`.

    HOOK: prices are split-adjusted, so today this is the adjusted count. When the data layer can give the
    split factor between `date` and today (as-traded share counts), convert here; every per-share fee in both
    engines goes through this function."""
    return shares


def broker_commission(model: str | None, shares: float, value: float) -> float:
    """Commission of one order under a broker preset (0 for None)."""
    spec = COMMISSION_MODELS.get(model)
    if not spec or shares <= 0:
        return 0.0
    per_share, lo, cap, fees = spec
    return min(max(lo, per_share * shares), cap * abs(value)) + fees * shares


def order_commission(shares: float, value: float, *, per_order: float = 0.0, per_share: float = 0.0,
                     pct: float = 0.0, model: str | None = None, ticker: str | None = None, date=None) -> float:
    """Total commission of one order of `shares` (adjusted) shares worth `value` dollars."""
    q = traded_shares(abs(shares), ticker, date)
    return per_order + per_share * q + pct * abs(value) + broker_commission(model, q, value)


def volume_slippage(base: float, shares: float, adv: float, spread_bps: float, impact_bps: float) -> float:
    """Slippage per side as a fraction of the price: `base` (fixed slippage) plus half the spread and a square-root
    market impact, impact_bps * sqrt(shares / ADV20). No volume history (adv NaN or 0): no impact term."""
    impact = impact_bps * math.sqrt(abs(shares) / adv) if (adv == adv and adv > 0 and math.isfinite(adv)) else 0.0
    return base + (spread_bps / 2 + impact) / 1e4
