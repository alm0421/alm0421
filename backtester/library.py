"""A library of classic, published strategies and Composer-style symphonies.

Each entry has a name, a category, tags (classic, tactical, leveraged, signal, Nasdaq-100), a short
"about", and either "text" (a plain-English sentence for the parser) or "spec" (a JSON Portfolio, for
trees the English can't express yet). `entry_spec(x)` turns any entry into a runnable spec.
"""
from __future__ import annotations

import re

TAGS = ("classic", "tactical", "leveraged", "signal", "Nasdaq-100")


def _if(rule: str, on: str, then: dict, other: dict) -> dict:
    return {"if": rule, "on": on, "then": then, "else": other}


def _a(t: str) -> dict:
    return {"asset": t}


def _alloc(tree: dict, rebalance: str = "daily", **kw) -> dict:
    return {"kind": "allocation", "tree": tree, "rebalance": rebalance, **kw}


LIBRARY = [
    # --- asset allocation (Portfolio Visualizer / lazy portfolios)
    {"name": "60/40 stocks/bonds", "category": "Asset allocation", "tags": ["classic"],
     "text": "hold 60% SPY and 40% AGG, rebalance quarterly",
     "about": "The classic balanced portfolio."},
    {"name": "Permanent Portfolio", "category": "Asset allocation", "tags": ["classic"],
     "text": "permanent portfolio, rebalance yearly",
     "about": "Harry Browne: US stocks (VTI), long Treasuries (TLT), cash (BIL) and gold (GLD), 25% each. Add 'since 1972' for the long-history series."},
    {"name": "All Weather (simplified)", "category": "Asset allocation", "tags": ["classic"],
     "text": "all weather portfolio, rebalance quarterly",
     "about": "Ray Dalio's risk-balanced mix: 30% VTI, 40% TLT, 15% IEF, 7.5% GLD, 7.5% DBC."},
    {"name": "Golden Butterfly", "category": "Asset allocation", "tags": ["classic"],
     "text": "golden butterfly portfolio, rebalance yearly",
     "about": "Tyler's five-way split: 20% each VTI, VBR (small-cap value), TLT, SHY and GLD. Add 'since 1972' for the long-history series."},
    {"name": "Three-fund portfolio", "category": "Asset allocation", "tags": ["classic"],
     "text": "three fund portfolio, rebalance yearly",
     "about": "Bogleheads: 64% US stocks (VTI), 16% international stocks (VXUS), 20% bonds (BND)."},
    {"name": "Risk parity (inverse vol)", "category": "Asset allocation", "tags": ["classic"],
     "text": "inverse volatility weighted SPY, TLT, GLD and DBC using a 60 day lookback, rebalance monthly",
     "about": "Each asset contributes roughly equal volatility."},
    {"name": "Diversifier core (KMLM/BTAL)", "category": "Asset allocation", "tags": ["classic"],
     "text": "hold 40% SPY, 20% KMLM, 20% BTAL and 20% GLD, rebalance monthly",
     "about": "Stocks plus managed futures (KMLM), anti-beta (BTAL) and gold: sleeves that tend to zig when stocks zag. KMLM starts in 2020."},
    {"name": "Crisis alpha sleeve", "category": "Asset allocation", "tags": ["classic"],
     "text": "equal weight KMLM, DBMF and BTAL, rebalance monthly",
     "about": "Two managed-futures funds and an anti-beta fund on their own, to see what they add."},
    # --- tactical / momentum (Composer, PV tactical models)
    {"name": "Faber 10-month timing", "category": "Tactical", "tags": ["tactical", "classic"],
     "text": "hold SPY when it is above its 10-month moving average, otherwise cash, rebalance monthly",
     "about": "Meb Faber (2007): be in stocks only above the 10-month average."},
    {"name": "GTAA 5 (Faber)", "category": "Tactical", "tags": ["classic", "tactical"],
     "text": "hold 20% (SPY when it is above its 10 month moving average, otherwise cash), "
             "20% (EFA when it is above its 10 month moving average, otherwise cash), "
             "20% (IEF when it is above its 10 month moving average, otherwise cash), "
             "20% (VNQ when it is above its 10 month moving average, otherwise cash) and "
             "20% (DBC when it is above its 10 month moving average, otherwise cash), rebalance monthly",
     "about": "Meb Faber's Global Tactical Asset Allocation (2007): US stocks, foreign stocks, 10-year Treasuries, REITs "
              "and commodities, 20% each, each held only while its month-end close is above its 10-month average "
              "(otherwise that 20% sits in T-bills)."},
    {"name": "Dual momentum (Antonacci)", "category": "Tactical", "tags": ["tactical"],
     "text": "dual momentum between SPY and EFA with AGG as the safe asset, rebalance monthly",
     "about": "Relative momentum between US and international stocks, absolute momentum vs T-bills."},
    {"name": "Global Equities Momentum (GEM)", "category": "Tactical", "tags": ["tactical", "classic"],
     "spec": _alloc(_if("tret(tr, 252) > tbill_ret(252)", "SPY",
                        {"filter": {"select": "top", "n": 1, "by": "tret(tr, 252)", "weights": "equal"},
                         "universe": "children", "children": [_a("SPY"), _a("EFA")], "fallback": {"cash": True}},
                        _a("AGG")), "monthly",
                    name="Global Equities Momentum (GEM)",
                    description="GEM: if SPY's 12-month return beats T-bills hold the stronger of SPY and EFA, else AGG"),
     "about": "Antonacci's rules as published: absolute momentum of SPY against T-bills, then the stronger of US and international stocks by 12-month return; bonds otherwise."},
    {"name": "Accelerating dual momentum", "category": "Tactical", "tags": ["tactical"],
     "spec": _alloc({"filter": {"select": "top", "n": 1, "by": "tret(tr, 21) + tret(tr, 63) + tret(tr, 126)",
                                "require": "tret(tr, 21) + tret(tr, 63) + tret(tr, 126) > 0", "weights": "equal"},
                     "universe": "children", "children": [_a("SPY"), _a("EFA")], "fallback": _a("TLT")}, "monthly",
                    name="Accelerating dual momentum",
                    description="Accelerating dual momentum: SPY or EFA by 1+3+6 month return if positive, else TLT"),
     "about": "EngineeredPortfolio's faster variant: rank by the sum of 1, 3 and 6-month returns, hold long Treasuries when the winner's score is negative."},
    {"name": "Sector rotation", "category": "Tactical", "tags": ["tactical"],
     "text": "hold the top 3 sector ETFs by 6 month momentum, rebalance monthly",
     "about": "Own the three strongest S&P sectors."},
    {"name": "Sector momentum top 2 (3 months)", "category": "Tactical", "tags": ["tactical"],
     "text": "hold the top 2 of XLK, XLE, XLF, XLV, XLY, XLP, XLI, XLU and XLB by 3 month return, rebalance monthly",
     "about": "A faster sector rotation among the nine original SPDR sectors."},
    {"name": "Treasury duration rotation", "category": "Tactical", "tags": ["tactical"],
     "text": "hold the top 1 of TLT, IEF, SHY and TIP by 3 month return, rebalance monthly",
     "about": "Own the bond fund with the best 3-month return: long duration when rates fall, short when they rise."},
    {"name": "Nasdaq-100 momentum (point-in-time)", "category": "Tactical", "tags": ["tactical", "Nasdaq-100"],
     "text": "hold the top 10 Nasdaq 100 stocks by 12 month return, rebalance monthly, since 2005",
     "about": "Cross-sectional momentum inside the index, using historical membership."},
    {"name": "Nasdaq-100 top-5 momentum, inverse vol", "category": "Tactical", "tags": ["tactical", "Nasdaq-100"],
     "text": "hold the top 5 Nasdaq 100 stocks by 6 month momentum, inverse volatility weighted, rebalance monthly, since 2005",
     "about": "Five strongest members by 6-month return, sized so each carries similar volatility."},
    {"name": "Nasdaq-100 weekly oversold rotation", "category": "Tactical", "tags": ["tactical", "Nasdaq-100"],
     "text": "hold the bottom 3 Nasdaq 100 stocks by RSI(10), rebalance weekly, since 2005",
     "about": "Short-term reversal: each week own the three most oversold members (lowest RSI(10))."},
    {"name": "Low-volatility Nasdaq-100", "category": "Tactical", "tags": ["tactical", "Nasdaq-100"],
     "text": "hold the bottom 10 Nasdaq 100 stocks by 60 day volatility, rebalance monthly, since 2005",
     "about": "The low-volatility anomaly within the index."},
    {"name": "QQQ/TLT regime switch", "category": "Tactical", "tags": ["tactical"],
     "text": "if SPY is above its 200-day moving average hold QQQ, otherwise hold TLT, rebalance daily",
     "about": "Growth when the trend is up, long bonds when it isn't."},
    # --- leveraged, Composer-style symphonies
    {"name": "Leveraged trend (TQQQ)", "category": "Leveraged", "tags": ["tactical", "leveraged"],
     "text": "if QQQ is above its 200 day moving average hold TQQQ, otherwise hold BIL, rebalance daily",
     "about": "Only hold 3x Nasdaq in an uptrend."},
    {"name": "TQQQ For The Long Term (FTLT)", "category": "Leveraged", "tags": ["tactical", "leveraged"],
     "spec": _alloc(_if("close > sma(close, 200)", "SPY",
                        _if("rsi(close, 10) > 79", "TQQQ", _a("UVXY"), _a("TQQQ")),
                        _if("rsi(close, 10) < 31", "TQQQ", _a("TECL"),
                            _if("rsi(close, 10) < 30", "SPY", _a("UPRO"),
                                _if("close < sma(close, 20)", "TQQQ",
                                    {"filter": {"select": "top", "n": 1, "by": "rsi(close, 10)", "weights": "equal"},
                                     "universe": "children", "children": [_a("SQQQ"), _a("TLT")],
                                     "fallback": {"cash": True}},
                                    _a("TQQQ"))))), "daily",
                    name="TQQQ For The Long Term (FTLT)",
                    description="TQQQ FTLT: 200-day trend on SPY with RSI(10) overbought/oversold switches"),
     "about": "The popular Composer symphony: TQQQ in an uptrend but UVXY when TQQQ's RSI(10) is overbought; in a downtrend buy oversold dips with TECL/UPRO, otherwise the stronger of SQQQ and TLT by RSI below the 20-day average."},
    {"name": "Overbought hedge (SPY/UVXY)", "category": "Leveraged", "tags": ["tactical", "leveraged"],
     "spec": _alloc(_if("rsi(close, 10) > 80", "SPY", _a("UVXY"),
                        _if("rsi(close, 10) < 30", "SPY", _a("UPRO"), _a("SPY"))), "daily",
                    name="Overbought hedge (SPY/UVXY)",
                    description="SPY, but UVXY when SPY's RSI(10) is above 80 and UPRO when it is below 30"),
     "about": "Hold SPY; switch to volatility (UVXY) when it is very overbought and to 3x S&P when it is very oversold."},
    {"name": "VIX regime: UPRO or TLT", "category": "Leveraged", "tags": ["tactical", "leveraged"],
     "text": "if VIX is above 25 hold TLT, otherwise hold UPRO, rebalance daily",
     "about": "3x S&P while the VIX is calm, long Treasuries when fear is high."},
    {"name": "Risk-on TQQQ / risk-off TMF", "category": "Leveraged", "tags": ["tactical", "leveraged"],
     "text": "if SPY is above its 200-day moving average hold TQQQ, else if TLT is above its 50 day moving average hold TMF, otherwise hold BIL, rebalance daily",
     "about": "3x Nasdaq in an uptrend; 3x Treasuries when stocks trend down but bonds trend up; T-bills when both fall."},
    {"name": "Sideways-market deleverage", "category": "Leveraged", "tags": ["tactical", "leveraged"],
     "spec": _alloc(_if("close > sma(close, 200)", "QQQ",
                        _if("close > sma(close, 20)", "QQQ", _a("TQQQ"), _a("QQQ")),
                        _a("BIL")), "daily",
                    name="Sideways-market deleverage",
                    description="TQQQ above both the 20 and 200-day averages, QQQ above only the 200-day, else BIL"),
     "about": "Full 3x only when QQQ is above both its 20 and 200-day averages; 1x in a choppy uptrend; T-bills in a downtrend."},
    {"name": "SOXL/TECL/TQQQ momentum rotation", "category": "Leveraged", "tags": ["tactical", "leveraged"],
     "text": "hold the top 1 of SOXL, TECL and TQQQ by 1 month return, only if their 1 month return is positive, otherwise hold BIL, rebalance weekly",
     "about": "Own the strongest 3x tech fund each week while its 1-month return is positive."},
    {"name": "Hedgefundie (55/45 UPRO/TMF)", "category": "Leveraged", "tags": ["leveraged", "classic"],
     "text": "hold 55% UPRO and 45% TMF, rebalance quarterly",
     "about": "The Bogleheads 'Hedgefundie's excellent adventure': 3x S&P balanced by 3x long Treasuries."},
    {"name": "HFEA on the Nasdaq (TQQQ/TMF)", "category": "Leveraged", "tags": ["leveraged"],
     "text": "hold 55% TQQQ and 45% TMF, rebalance quarterly",
     "about": "The same adventure with 3x Nasdaq-100 instead of the S&P 500."},
    {"name": "Leveraged risk parity (UPRO/TMF/UGL)", "category": "Leveraged", "tags": ["leveraged"],
     "text": "inverse volatility weighted UPRO, TMF and UGL using a 20 day lookback, rebalance monthly",
     "about": "3x stocks, 3x bonds and 2x gold, each sized by inverse 20-day volatility."},
    # --- short-term signals (TradingView / QuantConnect style)
    {"name": "RSI(2) mean reversion (Connors)", "category": "Mean reversion", "tags": ["signal"],
     "text": "buy SPY at the close when RSI(2) is below 10 and it is above its 200-day moving average, sell when it closes above its 5-day moving average",
     "about": "Larry Connors' short-term pullback in an uptrend."},
    {"name": "RSI(2) across the Nasdaq-100", "category": "Mean reversion", "tags": ["signal", "Nasdaq-100"],
     "text": "buy Nasdaq 100 stocks at the close when RSI(2) is below 5 and it is above its 200-day moving average, sell when it closes above its 5-day moving average, max 10 positions, since 2005",
     "about": "The same idea on every index member, 10 slots."},
    {"name": "TQQQ RSI(10) swing", "category": "Mean reversion", "tags": ["signal", "leveraged"],
     "text": "buy TQQQ at the close when RSI(10) is below 31, sell when RSI(10) is above 79",
     "about": "The FTLT thresholds as a standalone trade: buy oversold 3x Nasdaq, sell when overbought."},
    {"name": "TQQQ dip in an uptrend", "category": "Mean reversion", "tags": ["signal", "leveraged"],
     "text": "buy TQQQ at the close when RSI(2) is below 10 and QQQ is above its 200-day moving average, sell when RSI(2) is above 70",
     "about": "Connors-style pullback on 3x Nasdaq, only while QQQ trends up."},
    {"name": "IBS reversion", "category": "Mean reversion", "tags": ["signal"],
     "text": "buy QQQ at the close when it closes in the bottom 20% of its range, hold 1 day",
     "about": "Internal bar strength: weak closes tend to bounce."},
    {"name": "Down 3 days in a row", "category": "Mean reversion", "tags": ["signal"],
     "text": "buy QQQ at the close when it is down 3 days in a row, hold 3 days",
     "about": "Buy short-term exhaustion."},
    {"name": "Turn of the month", "category": "Calendar", "tags": ["signal"],
     "text": "buy SPY at the close on the last trading day of the month, sell at the close 4 days later",
     "about": "Returns cluster around month-end flows."},
    {"name": "Donchian breakout with ATR stop", "category": "Trend following", "tags": ["signal"],
     "text": "buy QQQ when it closes above its 55 day high, with a 2 ATR stop and a 3 ATR trailing stop",
     "about": "Turtle-style breakout with volatility stops."},
    {"name": "Golden cross", "category": "Trend following", "tags": ["signal"],
     "text": "buy SPY when the 50 day moving average crosses above the 200 day moving average, sell when the 50 day moving average crosses below the 200 day moving average",
     "about": "The classic crossover."},
    {"name": "MACD crossover", "category": "Trend following", "tags": ["signal"],
     "text": "buy QQQ when MACD crosses above its signal line, sell when MACD crosses below its signal line",
     "about": "12/26/9 MACD signals."},
    {"name": "SOXL 20-day trend", "category": "Trend following", "tags": ["signal", "leveraged"],
     "text": "buy SOXL when it crosses above its 20 day moving average, sell when it crosses back below",
     "about": "Ride 3x semiconductors only above their 20-day average."},
    {"name": "Buy the VIX spike", "category": "Volatility", "tags": ["signal"],
     "text": "buy SPY when VIX is above 30, hold 20 days",
     "about": "Fear spikes have historically been buying opportunities."},
    {"name": "Monthly dollar-cost averaging", "category": "Savings plan", "tags": ["classic"],
     "text": "buy and hold SPY, add $500 every month",
     "about": "Regular contributions; see money-weighted vs time-weighted return."},
    {"name": "4% rule retirement", "category": "Retirement", "tags": ["classic"],
     "text": "hold 60% SPY and 40% AGG, withdraw 4% per year adjusted for inflation, starting with $1,000,000, rebalance annually",
     "about": "Bengen's safe-withdrawal test, with a Monte Carlo success rate."},
]

for _x in LIBRARY:
    _x["id"] = re.sub(r"[^a-z0-9]+", "-", _x["name"].lower()).strip("-")


def find(key: str) -> dict | None:
    """A library entry by id, name or sentence."""
    return next((x for x in LIBRARY if key in (x["id"], x["name"], x.get("text"))), None)


def entry_spec(x: dict):
    """The runnable Strategy/Portfolio for a library entry."""
    if x.get("spec"):
        import copy

        from . import runner
        return runner.from_dict(copy.deepcopy(x["spec"]))
    from . import parser
    return parser.parse(x["text"])
