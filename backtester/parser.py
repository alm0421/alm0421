"""Translate a plain-English strategy description into a Strategy (rule-based signals) or a
Portfolio (allocations: weights, rebalancing, regime switches, rotations).

The parser is deterministic and pattern based. It recognises a vocabulary of common trading
phrases (see README), echoes back exactly how it interpreted them, and refuses - rather than
guesses - whenever any part of the sentence is not understood. Anything it cannot express can be
written in the rule language inside backticks, e.g.

    buy AAPL at the close when `rsi(3) < 15 and close > sma(close, 100)`, hold 3 days
"""
from __future__ import annotations

import ast
import dataclasses
import re
import threading
from dataclasses import dataclass

import pandas as pd

from . import data
from .portfolio import Portfolio
from .portfolio import short_name as _pf_short_name
from .strategy import Strategy

NUM = r"(\d+(?:\.\d+)?)"
# a short-borrow fee in any common word order: "2% borrow fee", "2% annual borrow fee", "borrow fee 2%",
# "borrow fee of 2% per year", "borrow cost: 2% a year", "2% borrowing cost on shorts" (group 1 or 2 is the rate)
_BORROW_TAIL = (r"(?: (?:per|a|an) (?:year|annum)| annual(?:ly)?| yearly| p\.?a\.?)?"
                r"(?: on (?:the |any |all )?(?:shorts?|short (?:positions?|sales?)))?")
BORROW_FEE = (rf"(?:an? )?{NUM}% (?:(?:annual|annualized|yearly) )?(?:stock )?borrow(?:ing)? (?:fee|cost|rate)s?{_BORROW_TAIL}"
              rf"|(?:an? )?(?:(?:annual|annualized|yearly) )?(?:stock )?(?:short )?borrow(?:ing)? (?:fee|cost|rate)s?"
              rf"(?: of| at|:| =)? {NUM}%{_BORROW_TAIL}")
# "no borrow fee": shorts cost nothing to borrow (the default assumes 5%/yr for leveraged / inverse ETFs, 0.3% others)
NO_BORROW_FEE = r"(?:(?:with|and) )?(?:no |zero |without (?:a |any )?)(?:stock )?borrow(?:ing)? (?:fee|cost)s?(?: on (?:the )?shorts?)?"

WORD_NUMS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
}
_WN_UNIT = r"one|two|three|four|five|six|seven|eight|nine"
_WN_TEEN = r"ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen"
_WN_TENS = r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety"
_WN_99 = rf"(?:(?:{_WN_TENS})(?:[ -](?:{_WN_UNIT}))?|{_WN_TEEN}|{_WN_UNIT})"
# numbers written in words, up to 999: "seventy nine", "seventy-nine", "one hundred", "a hundred and twenty", "two
# hundred fifty"
_WN_RX = re.compile(rf"(?i)(?<!\w)(?:(?:(?P<h>{_WN_UNIT}|a) )?hundred(?:(?: and)? (?P<r>{_WN_99}))?|{_WN_99})(?!\w)")


def _word_value(w: str) -> int:
    parts = re.split(r"[ -]", w.lower())
    return sum(WORD_NUMS[p] for p in parts)


def _number_words(t: str) -> str:
    """Numbers written in words -> digits ('seventy nine' -> 79, 'one hundred' -> 100); outside backticks."""
    def fix(m):
        s = m.group(0)
        if re.search(r"(?i)hundred", s):
            h = m.group("h")
            n = 100 * (1 if not h or h.lower() == "a" else WORD_NUMS[h.lower()])
            return str(n + (_word_value(m.group("r")) if m.group("r") else 0))
        return str(_word_value(s))
    return _sub_outside(_WN_RX.pattern, fix, t, flags=re.I)

COMPANIES = {
    "microsoft": "MSFT", "apple": "AAPL", "amazon": "AMZN", "alphabet": "GOOGL", "google": "GOOGL",
    "meta platforms": "META", "facebook": "META", "nvidia": "NVDA", "tesla": "TSLA", "netflix": "NFLX",
    "broadcom": "AVGO", "costco": "COST", "adobe": "ADBE", "intel": "INTC", "cisco": "CSCO",
    "pepsico": "PEP", "pepsi": "PEP", "qualcomm": "QCOM", "texas instruments": "TXN",
    "starbucks": "SBUX", "paypal": "PYPL", "booking holdings": "BKNG", "intuit": "INTU", "amgen": "AMGN",
    "gilead": "GILD", "comcast": "CMCSA", "t-mobile": "TMUS", "honeywell": "HON", "micron": "MU",
    "applied materials": "AMAT", "lam research": "LRCX", "kla": "KLAC", "marvell": "MRVL",
    "palantir": "PLTR", "crowdstrike": "CRWD", "palo alto networks": "PANW", "fortinet": "FTNT",
    "mercadolibre": "MELI", "airbnb": "ABNB", "doordash": "DASH", "datadog": "DDOG",
    "zscaler": "ZS", "workday": "WDAY", "autodesk": "ADSK", "synopsys": "SNPS", "cadence": "CDNS",
    "regeneron": "REGN", "vertex": "VRTX", "biogen": "BIIB", "intuitive surgical": "ISRG",
    "lululemon": "LULU", "marriott": "MAR", "monster beverage": "MNST", "mondelez": "MDLZ",
    "kraft heinz": "KHC", "ross stores": "ROST", "o'reilly": "ORLY", "paccar": "PCAR",
    "paychex": "PAYX", "cintas": "CTAS", "fastenal": "FAST", "copart": "CPRT",
    "electronic arts": "EA", "take-two": "TTWO", "warner bros": "WBD", "microstrategy": "MSTR",
    "applovin": "APP", "shopify": "SHOP", "the trade desk": "TTD", "analog devices": "ADI",
    "astrazeneca": "AZN", "advanced micro devices": "AMD", "walmart": "WMT",
    "idexx": "IDXX", "dexcom": "DXCM", "moderna": "MRNA", "linde": "LIN",
    "constellation energy": "CEG", "exelon": "EXC", "xcel": "XEL", "charter": "CHTR",
    "coreweave": "CRWV", "sandisk": "SNDK", "seagate": "STX", "western digital": "WDC",
    # funds, indexes and asset classes
    "s&p 500": "SPY", "s&p500": "SPY", "the s&p": "SPY", "nasdaq 100 etf": "QQQ",
    "long-term treasuries": "TLT", "long term treasuries": "TLT", "long treasuries": "TLT", "long bonds": "TLT",
    "intermediate treasuries": "IEF", "short-term treasuries": "SHY", "short term treasuries": "SHY",
    "t-bills": "BIL", "treasury bills": "BIL", "aggregate bonds": "AGG", "bonds": "BND", "gold": "GLD",
    "silver": "SLV", "commodities": "DBC", "emerging markets": "EEM", "international stocks": "EFA",
    "developed markets": "EFA", "real estate": "VNQ", "reits": "VNQ", "small caps": "IWM", "small-caps": "IWM",
    "russell 2000": "IWM", "the vix": "^VIX", "vix": "^VIX", "the dow": "DIA", "dow jones": "DIA",
    "semiconductors": "SMH", "biotech": "IBB",
}

NOT_TICKERS = {
    "I", "A", "RSI", "SMA", "EMA", "ATR", "IBS", "MA", "AND", "OR", "THE", "US", "USD", "MOC",
    "MOO", "EOD", "PM", "AM", "NDX", "IF", "AT", "SL", "TP", "ETF", "ETFS", "IT", "BUY", "SELL", "ON",
    "ALL", "MY", "DO", "BE", "GO", "SO", "UP", "NOW", "OUT", "NEW", "HIGH", "LOW", "OPEN", "CLOSE",
    "MACD", "ADX", "CCI", "MFI", "OBV", "VWAP", "SAR", "DI", "ROC", "TR", "OK", "AI", "CAGR", "YTD",
    "BPS", "ADV", "PE", "EPS", "TO", "OF", "IN", "BY", "NO", "AS", "OFF", "WHEN", "HOLD", "FOR", "X",
    "IRA", "CPI", "FED", "N", "K", "M", "B", "T", "FOMC", "GDP", "EV", "USA", "NYSE", "NASDAQ", "SPX",
    "CAPE",   # Shiller's cyclically adjusted P/E (cape() in the rule language), not a ticker
}

DOW = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4}
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september",
          "october", "november", "december"]

SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB", "XLRE", "XLC"]

# Named model ("lazy") portfolios, as Portfolio Visualizer / Portfolio Charts list them. Each entry:
# (phrase regex, display name, [(weight %, ETF), ...], source). Numbers are written as digits because
# sentences are normalised first ("three" -> "3"). To add one, append a row.
MODEL_PORTFOLIOS = [
    (r"(?:bogleheads?'? )?3[- ]fund", "Three-fund portfolio", [(64, "VTI"), (16, "VXUS"), (20, "BND")],
     "Bogleheads three-fund portfolio: 80% stocks (a fifth of them international) and 20% bonds"),
    (r"(?:ray )?(?:dalio'?s? )?all[- ]weather", "All Weather portfolio",
     [(30, "VTI"), (40, "TLT"), (15, "IEF"), (7.5, "GLD"), (7.5, "DBC")], "Ray Dalio's All Weather (Tony Robbins' version)"),
    (r"golden butterfly", "Golden Butterfly", [(20, "VTI"), (20, "VBR"), (20, "TLT"), (20, "SHY"), (20, "GLD")],
     "Tyler (Portfolio Charts): total market, small-cap value, long and short Treasuries, gold"),
    (r"(?:harry )?(?:browne'?s? )?permanent", "Permanent Portfolio", [(25, "VTI"), (25, "TLT"), (25, "BIL"), (25, "GLD")],
     "Harry Browne: stocks, long Treasuries, cash, gold"),
    (r"(?:bill )?(?:schultheis'?s? )?coffee ?house", "Coffeehouse portfolio",
     [(40, "BND"), (10, "SPY"), (10, "VTV"), (10, "VB"), (10, "VBR"), (10, "EFA"), (10, "VNQ")],
     "Bill Schultheis: 40% bonds, 10% each in large, large value, small, small value, international and REITs"),
    (r"(?:meb )?(?:faber'?s? )?ivy", "Ivy portfolio", [(20, "VTI"), (20, "VEU"), (20, "BND"), (20, "VNQ"), (20, "DBC")],
     "Faber's Ivy / GTAA 5 asset classes held in equal weights (no timing)"),
    (r"(?:william )?(?:bernstein'?s? )?no[- ]brainer", "Bernstein No-Brainer", [(25, "SPY"), (25, "VB"), (25, "VGK"), (25, "SHY")],
     "William Bernstein: S&P 500, small caps, European stocks, short-term bonds"),
    (r"(?:classic )?60[/-]40", "60/40 portfolio", [(60, "VTI"), (40, "BND")], "60% US total market, 40% total bond market"),
    (r"hedgefundie'?s?(?: excellent)? adventure|hfea", "Hedgefundie's Excellent Adventure", [(55, "UPRO"), (45, "TMF")],
     "55% 3x S&P 500 (UPRO), 45% 3x long Treasuries (TMF)"),
    (r"(?:david )?swensen'?s?(?: yale)?|yale(?: endowment)?", "Swensen (Yale) portfolio",
     [(30, "VTI"), (15, "EFA"), (5, "EEM"), (20, "VNQ"), (15, "TIP"), (15, "IEF")],
     "David Swensen's individual-investor portfolio: US, developed, emerging, REITs, TIPS, Treasuries"),
    (r"larry(?: swedroe'?s?)?|swedroe'?s?", "Larry portfolio", [(15, "VBR"), (7.5, "VSS"), (7.5, "VWO"), (70, "IEF")],
     "Larry Swedroe: 30% small value / emerging stocks, 70% intermediate Treasuries (VSS stands in for international small value)"),
    (r"(?:warren )?buffett'?s?(?: 90[/-]10)?", "Buffett 90/10 portfolio", [(90, "VOO"), (10, "SHY")],
     "Warren Buffett's instructions for his estate: 90% an S&P 500 index fund, 10% short-term government bonds"),
    (r"global market", "Global Market Portfolio",
     [(24, "SPY"), (14, "EFA"), (7, "VB"), (5, "EEM"), (44, "IEF"), (4, "VNQ"), (2, "GLD")],
     "Doeswijk, Lam and Swinkels' market-cap mix of all assets (Portfolio Charts' version: 38% developed-world large caps, "
     "7% small caps, 5% emerging, 44% developed-world government bonds, 4% REITs, 2% gold). Proxies: the developed-world "
     "large caps are split about 60/40 US / ex-US (SPY / EFA), US small caps (VB) stand in for developed-world small caps, "
     "and US intermediate Treasuries (IEF) for developed-world government bonds"),
    (r"(?:bob )?(?:clyatt'?s? )?sandwich", "Sandwich portfolio",
     [(20, "SPY"), (8, "VB"), (10, "VSS"), (6, "EFA"), (6, "EEM"), (41, "IEF"), (4, "BIL"), (5, "VNQ")],
     "Bob Clyatt (Portfolio Charts' version): 20% US large, 8% US small, 10% ex-US small, 6% ex-US large, 6% emerging, "
     "30% US intermediate Treasuries, 11% ex-US intermediate government bonds, 4% T-bills, 5% REITs. Proxies: VSS (all-world "
     "ex-US small caps, emerging included) for developed ex-US small caps, and IEF also holds the 11% ex-US bonds (41% in all)"),
    (r"desert", "Desert portfolio", [(60, "IEF"), (30, "VTI"), (10, "GLD")],
     "60% intermediate Treasuries, 30% US total market, 10% gold"),
    (r"(?:paul )?merriman'?s?(?: ultimate)?(?: buy[- ]and[- ]hold)?|(?:the )?ultimate buy[- ]and[- ]hold", "Ultimate Buy and Hold portfolio",
     [(6, "SPY"), (6, "VTV"), (6, "VB"), (6, "VBR"), (12, "EFA"), (12, "VSS"), (6, "EEM"), (20, "IEF"), (20, "SHY"), (6, "VNQ")],
     "Paul Merriman: 6% each in US large, large value, small, small value, ex-US large, ex-US large value, ex-US small, "
     "ex-US small value and emerging, 6% REITs, 20% intermediate and 20% short-term Treasuries. Proxies: EFA holds both ex-US "
     "large sleeves (12%) and VSS both ex-US small sleeves (12%), as no ex-US value funds are in the data"),
    (r"weird", "Weird portfolio", [(20, "VBR"), (20, "VSS"), (20, "TLT"), (20, "VNQ"), (20, "GLD")],
     "Value Stock Geek: US small value, ex-US small caps, long Treasuries, REITs and gold, 20% each (VSS, all-world ex-US "
     "small caps, stands in for developed ex-US small caps)"),
    (r"(?:rick )?(?:ferri'?s? )?core[- ]?(?:4|four)", "Core Four portfolio", [(48, "VTI"), (24, "VXUS"), (20, "BND"), (8, "VNQ")],
     "Rick Ferri: 48% US total market, 24% international, 20% total bond market, 8% REITs"),
    (r"talmud(?:ic)?", "Talmud portfolio", [(100 / 3, "VTI"), (100 / 3, "VNQ"), (100 / 3, "IEF")],
     "a third each in stocks (US total market), real estate (REITs) and bonds (intermediate Treasuries)"),
    (r"pinwheel", "Pinwheel portfolio",
     [(15, "SPY"), (10, "VBR"), (15, "EFA"), (10, "EEM"), (15, "IEF"), (10, "BIL"), (15, "VNQ"), (10, "GLD")],
     "Portfolio Charts: 15% US large, 10% US small value, 15% ex-US large, 10% emerging, 15% intermediate Treasuries, "
     "10% T-bills, 15% REITs, 10% gold"),
    (r"(?:craig )?(?:israelsen'?s? )?7[- ]?(?:twelve|12)", "7Twelve portfolio",
     [(100 / 12, "VV"), (100 / 12, "IJH"), (100 / 12, "IJR"), (100 / 12, "VEA"), (100 / 12, "VWO"), (100 / 12, "VNQ"),
      (100 / 12, "GLD"), (100 / 12, "DBC"), (100 / 12, "BND"), (100 / 12, "TIP"), (100 / 12, "BNDX"), (100 / 12, "BIL")],
     "Craig Israelsen, '7Twelve: A Diversified Investment Portfolio with a Plan' (2010): 12 equal sleeves across 7 asset "
     "classes - US large, mid and small caps, developed ex-US, emerging, REITs, natural resources, commodities, US bonds, "
     "TIPS, international bonds and cash. Proxy: gold (GLD) for the natural-resources sleeve (no natural-resources fund in "
     "the data), as PortfoliosLab does"),
    (r"(?:scott )?(?:burns'?s? )?couch[- ]?potato|(?:scott )?burns'?s? couch", "Couch Potato portfolio", [(50, "VTI"), (50, "TIP")],
     "Scott Burns (Dallas Morning News), the current version: 50% US total stock market, 50% TIPS (his original 1991 "
     "Couch Potato was 50% S&P 500 / 50% total bond market: write '50% SPY and 50% BND' for it)"),
    (r"(?:paul )?merriman'?s? (?:4|four)[- ]fund(?: combo| combination)?", "Merriman 4-fund combo",
     [(25, "SPY"), (25, "VTV"), (25, "VB"), (25, "VBR")],
     "Paul Merriman's 4-Fund Combo (all equity): 25% each S&P 500, US large-cap value, US small-cap blend and US small-cap value"),
    (r"(?:frank )?(?:armstrong'?s? )?ideal index", "Ideal Index portfolio",
     [(6.25, "VV"), (9.25, "VTV"), (6.25, "IJT"), (9.25, "IJS"), (8, "VNQ"), (31, "VEU"), (30, "SHY")],
     "Frank Armstrong's Ideal Index (as Portfolio Visualizer / PortfoliosLab list it): 6.25% US large, 9.25% US large value, "
     "6.25% US small growth, 9.25% US small value, 8% REITs, 31% international, 30% short-term Treasuries"),
    (r"(?:bogleheads?'? )?(?:4|four)[- ]fund", "Four-fund portfolio", [(48, "VTI"), (24, "VXUS"), (20, "BND"), (8, "BNDX")],
     "Bogleheads four-fund portfolio: the three-fund portfolio plus international bonds - 72% stocks (a third of them "
     "international) and 28% bonds (8% international, USD-hedged). PortfoliosLab uses 50% VTI / 30% VEA / 15% BND / 5% BNDX; "
     "write those weights to use theirs"),
    (r"(?:the )?second[- ]grader'?s?(?: starter)?", "Second Grader's Starter portfolio", [(60, "VTI"), (30, "VEU"), (10, "BND")],
     "Allan Roth ('How a Second Grader Beats Wall Street'): 60% US total market, 30% all-world ex-US, 10% total bond market"),
    (r"(?:dave )?ramsey'?s?", "Dave Ramsey portfolio", [(25, "VUG"), (25, "SPY"), (25, "VOT"), (25, "VXUS")],
     "Dave Ramsey's four equity fund types, 25% each: growth (VUG), growth and income (S&P 500, SPY), aggressive growth "
     "(US mid-cap growth, VOT) and international (VXUS). Index-fund proxies: he recommends actively managed funds"),
    (r"(?:ted )?aronson(?: family)?(?: taxable)?", "Aronson Family Taxable portfolio",
     [(15, "TIP"), (15, "VPL"), (15, "VV"), (10, "TLT"), (10, "IJR"), (10, "EEM"), (5, "HYG"), (5, "IJT"), (5, "VTI"),
      (5, "VGK"), (5, "IJS")],
     "Ted Aronson's family taxable account (Jason Zweig, Money, 2004; as PortfoliosLab lists it): 15% TIPS, 15% Pacific, "
     "15% US large, 10% long Treasuries, 10% US small, 10% emerging, 5% high yield, 5% US small growth, 5% US total market, "
     "5% Europe, 5% US small value"),
]
# any "N/M" stock/bond split: "80/20 portfolio", "Stocks/Bonds 60/40", "US stocks and bonds 70/30", "70/30 stocks/bonds"
_SB = r"(?:u\.?s\.? )?(?:stocks?|equit(?:y|ies))(?: and | ?/ ?| & )(?:bonds?|fixed income)"
GENERIC_SB_RX = (rf"(?:the |a |an )?(?:{_SB} (?P<a1>\d+(?:\.\d+)?) ?/ ?(?P<b1>\d+(?:\.\d+)?)(?: portfolio| mix| allocation| split)?"
                 rf"|(?P<a2>\d+(?:\.\d+)?) ?/ ?(?P<b2>\d+(?:\.\d+)?)(?: (?:{_SB}))?(?: portfolio| mix| allocation| split)"
                 rf"|(?P<a3>\d+(?:\.\d+)?) ?/ ?(?P<b3>\d+(?:\.\d+)?) {_SB})")
MODEL_RX = (r"(?:(?:the |a |an )?(?:" + "|".join(p for p, *_ in MODEL_PORTFOLIOS)
            + r")(?:'s)?(?: lazy)?(?: portfolio| model| allocation| strategy)?|" + GENERIC_SB_RX + ")")

# long-history series that can stand in for an ETF before it existed: (series, None) when it IS that
# ETF extended back, (series, proxy description) when it approximates it. The first available wins,
# so e.g. BNDSIM is used for BND when the data has it and IEFSIM otherwise. Missing series are skipped.
SIM_FOR = {
    "SPY": [("SPYSIM", None)],
    # "fund-exact" series first: the named fund as soon as it or its mutual-fund twin exists
    "VTI": [("VTISIM", "the US market (Fama-French) until April 1992, then the Vanguard Total Stock Market Index fund "
                       "VTSMX, and VTI itself from 2001"), ("SPYSIM", "the US market (Fama-French), then SPY")],
    "VOO": [("SPYSIM", "the US market, then SPY")], "IVV": [("SPYSIM", "the US market, then SPY")],
    "VV": [("SPYSIM", "US large caps: the US market (Fama-French), then SPY")],
    "IJT": [("VBKSIM", "US small-cap growth (Fama-French), then VBK")],
    "VPL": [("EWJSIM", "Japanese stocks (about 60% of the Pacific index), then EWJ")],
    "TLT": [("TLTSIM", None)], "VGLT": [("TLTSIM", "long Treasuries, then TLT")],
    "IEF": [("IEFSIM", None)], "SHY": [("SHYSIM", None)], "BIL": [("BILSIM", None)],
    "IEI": [("IEISIM", None)], "VB": [("VBSIM", None)], "VBR": [("VBRSIM", None)], "VTV": [("VTVSIM", None)],
    "IWN": [("VBRSIM", "US small-cap value (Fama-French), then VBR")],
    "VBK": [("VBKSIM", None)], "IWO": [("VBKSIM", "US small-cap growth (Fama-French), then VBK")],
    "MDY": [("MIDSIM", None)], "IJH": [("MIDSIM", "US mid caps (Fama-French), then MDY (S&P 400)")],
    "VO": [("MIDSIM", "US mid caps (Fama-French), then MDY")],
    "VUG": [("VUGSIM", None)], "EFA": [("EFASIM", None)], "GLD": [("GLDSIM", None)],
    "IAU": [("GLDSIM", "gold, then GLD")], "GLDM": [("GLDSIM", "gold, then GLD")],
    "DBC": [("DBCSIM", None)], "GSG": [("DBCSIM", "commodity futures (AQR equal-weight index), then DBC")],
    "PDBC": [("DBCSIM", "commodity futures (AQR equal-weight index), then DBC")],
    "VNQ": [("VNQSIM", None)], "EEM": [("EEMSIM", None)],
    "VWO": [("VWOSIM", "emerging markets (Fama-French) until 1994, then the Vanguard Emerging Markets Stock Index fund "
                       "VEIEX, and VWO itself from 2005"), ("EEMSIM", "emerging markets, then EEM")],
    "IEMG": [("VWOSIM", "emerging markets, then VEIEX and VWO"), ("EEMSIM", "emerging markets, then EEM")],
    "LQD": [("LQDSIM", None)], "TIP": [("TIPSIM", None)], "SCHP": [("TIPSIM", "TIPS (VIPSX), then TIP")],
    "HYG": [("HYGSIM", None)], "JNK": [("HYGSIM", "high-yield bonds (VWEHX), then HYG")],
    "BND": [("BNDSIM", None), ("AGGSIM", "the aggregate bond market, then AGG"),
            ("IEFSIM", "intermediate Treasuries, then IEF")],
    "AGG": [("AGGSIM", None), ("BNDSIM", "the US aggregate bond market (VBMFX from 1986), then BND"),
            ("IEFSIM", "intermediate Treasuries, then IEF")],
    "BNDX": [("BNDXSIM", None)],
    "BWX": [("BWXSIM", None)],
    "IGOV": [("BWXSIM", "developed ex-US government bonds, unhedged (OECD yields and exchange rates), then BWX")],
    "VXUS": [("VXUSSIM", "80% developed ex-US + 20% emerging markets (Fama-French) until 1996, then the Vanguard Total "
                         "International Stock Index fund VGTSX, and VXUS itself from 2011"),
             ("EFASIM", "developed markets ex-US (no emerging markets), then EFA")],
    "VEU": [("VXUSSIM", "all-world ex-US: developed + emerging (Fama-French) until 1996, then the Vanguard Total "
                        "International Stock Index fund VGTSX, then VXUS"),
            ("EFASIM", "developed markets ex-US (no emerging markets), then EFA")],
    "IXUS": [("VXUSSIM", "all-world ex-US, then VGTSX and VXUS")], "ACWX": [("VXUSSIM", "all-world ex-US, then VGTSX and VXUS")],
    "VTSAX": [("VTISIM", "the US market (Fama-French), then VTSMX and VTI")],
    "VTIAX": [("VXUSSIM", "all-world ex-US, then VGTSX and VXUS")],
    "ITOT": [("VTISIM", "the US market, then VTSMX and VTI")], "SCHB": [("VTISIM", "the US market, then VTSMX and VTI")],
    "VOE": [("VOESIM", None)], "VOT": [("VOTSIM", None)],
    "IWS": [("VOESIM", "US mid-cap value (Fama-French), then VOE")], "IWP": [("VOTSIM", "US mid-cap growth (Fama-French), then VOT")],
    "IWD": [("VTVSIM", "US large-cap value (Fama-French, VIVAX), then VTV")],
    "IWF": [("VUGSIM", "US large-cap growth (Fama-French, VIGRX), then VUG")],
    "IWM": [("VBSIM", "US small caps (Fama-French, NAESX), then VB")],
    "IJR": [("VBSIM", "US small caps (Fama-French, NAESX), then VB")],
    "IJS": [("VBRSIM", "US small-cap value (Fama-French, VISVX), then VBR")],
    "AVUV": [("VBRSIM", "US small-cap value (Fama-French, VISVX), then VBR")],
    "VCLT": [("VCLTSIM", None)], "IGLB": [("VCLTSIM", "long-term IG corporates (Moody's yields, VWESX), then VCLT")],
    "SPLB": [("VCLTSIM", "long-term IG corporates (Moody's yields, VWESX), then VCLT")],
    "MUB": [("MUBSIM", None)], "VTEB": [("MUBSIM", "municipal bonds (VWITX), then MUB")],
    "TFI": [("MUBSIM", "municipal bonds (VWITX), then MUB")],
    "EMB": [("EMBSIM", None)], "VWOB": [("EMBSIM", "emerging-market USD bonds (FNMIX), then EMB")],
    "PCY": [("EMBSIM", "emerging-market USD bonds (FNMIX), then EMB")],
    "EWJ": [("EWJSIM", None)], "EWU": [("EWUSIM", None)], "EWG": [("EWGSIM", None)], "EWC": [("EWCSIM", None)],
    "EWA": [("EWASIM", None)], "EWQ": [("EWQSIM", None)], "EWL": [("EWLSIM", None)], "EWH": [("EWHSIM", None)],
    "VEA": [("EFASIM", "developed markets ex-US, then EFA")],
    "EFV": [("EFVSIM", None)], "SCZ": [("SCZSIM", None)], "AVDV": [("AVDVSIM", None)],
    "DLS": [("AVDVSIM", "developed ex-US small-cap value (Fama-French), then AVDV")],
    "DISV": [("AVDVSIM", "developed ex-US small-cap value (Fama-French), then AVDV")],
    "VSS": [("SCZSIM", "developed ex-US small caps (Fama-French; no emerging markets), then SCZ")],
    "VGK": [("VGKSIM", None)],
}


def _proxies_before_inception(tree: dict, start, notes: list[str]) -> dict:
    """'with proxies before inception': {holding: proxy} for each fixed holding whose history begins after the
    requested start and that has a stated proxy (portfolio.PROXIES_BEFORE_INCEPTION), with a note per sleeve.
    Opt-in only: by default a backtest starts when every holding has data."""
    from .portfolio import fixed_tickers, proxy_before
    if not start:
        notes.append("'With proxies before inception' needs a start date before a holding's history begins (e.g. "
                     "'since 1972'); without one the backtest starts when every holding has data, so no proxy is used.")
        return {}
    s0 = pd.Timestamp(start)
    out, missing = {}, []
    for t in fixed_tickers(tree):
        try:
            first = data.load(t).index[0]
        except Exception:  # noqa: BLE001 - reported when the portfolio runs
            continue
        if first <= s0 + pd.Timedelta(days=7):
            continue
        px = proxy_before(t, s0)
        if px is None:
            missing.append(f"{t} (from {first.date()})")
            continue
        out[t] = px              # the run notes each one, with its dates
    if missing:
        notes.append("No stated proxy for " + ", ".join(missing) + ": the backtest starts once it has data.")
    if not out and not missing:
        notes.append("'With proxies before inception': every holding has data from the start, so no proxy is used.")
    return out


def _model_portfolio(text: str, notes: list[str]) -> dict | None:
    """'golden butterfly' -> its weights node; with a start date before an ETF existed, the long-history
    series (SPYSIM, TLTSIM, ...) are used for the funds that have one, with a note."""
    s = text.strip().lower()
    if not re.fullmatch(MODEL_RX, s):
        return None
    for pat, name, holdings, about in MODEL_PORTFOLIOS:
        if re.fullmatch(rf"(?:the |a |an )?(?:{pat})(?:'s)?(?: lazy)?(?: portfolio| model| allocation| strategy)?", s):
            break
    else:
        g = re.fullmatch(GENERIC_SB_RX, s)
        a, b = (float(x) for x in next((g.group(f"a{i}"), g.group(f"b{i}")) for i in (1, 2, 3) if g.group(f"a{i}")))
        if abs(a + b - 100) > 1e-9:
            raise ParseError(f"'{text.strip()}': a stock/bond split must add up to 100 ({a:g} + {b:g} = {a + b:g}).")
        name = f"{a:g}/{b:g} stocks/bonds"
        holdings = [(a, "VTI"), (b, "BND")]
        about = (f"{a:g}% US total stock market, {b:g}% US total bond market, Portfolio Visualizer's 'Stocks/Bonds "
                 f"{a:g}/{b:g}'; with an early start the longest series stand in: the US market back to 1926 and the "
                 "total bond market back to 1962")
    node = _model_node(name, holdings, about, notes)
    _TL.model = (name, node)        # a named model portfolio: rebalanced yearly by default (Portfolio Visualizer's)
    return node


def _model_node(name: str, holdings: list, about: str, notes: list[str]) -> dict:
    """The weights node of a named mix, swapping in long-history series for funds that did not exist yet."""
    start = getattr(_TL, "start", None)
    known = _known()
    kids, ws, swaps, late = [], [], [], []
    for w, etf in holdings:
        use = etf
        first = None
        try:
            first = data.load(etf).index[0] if etf in known else None
        except Exception:
            first = None
        if start and (first is None or str(first.date()) > str(start)[:10]):
            for sim, proxy in SIM_FOR.get(etf, []):
                if sim in known:
                    use = sim
                    swaps.append(f"{sim} for {etf}" + (f" ({proxy})" if proxy else ""))
                    break
            else:
                if first is not None:
                    late.append(f"{etf} (from {first.date()})")
        if use not in known:
            raise ParseError(f"{name} needs {etf}, which has no price data here.")
        kids.append({"asset": use})
        ws.append(w / 100)
    notes.append(f"{name}: " + ", ".join(f"{round(w, 2):g}% {e}" for w, e in holdings) + f" ({about}).")
    if swaps:
        notes.append(f"Start {str(start)[:10]} is before some of the funds existed: using the long-history series "
                     + "; ".join(swaps) + ".")
    if late:
        notes.append("No long-history series for " + ", ".join(late) + ": the backtest can only start once it has data.")
    if not start and any(any(sim in known for sim, _ in SIM_FOR.get(e, [])) for _, e in holdings):
        notes.append(f"{name} uses the ETFs; add e.g. 'since 1972' to extend it back with the simulated long-history series.")
    return {"weights": "specified", "w": [round(w, 10) for w in ws], "children": kids}


def _long_history(etf: str, swaps: list, late: list) -> str:
    """The fund, or its long-history series when the backtest starts before the fund existed (as _model_node)."""
    start = getattr(_TL, "start", None)
    known = _known()
    try:
        first = data.load(etf).index[0] if etf in known else None
    except Exception:  # noqa: BLE001
        first = None
    use = etf
    if start and (first is None or str(first.date()) > str(start)[:10]):
        for sim, proxy in SIM_FOR.get(etf, []):
            if sim in known:
                swaps.append(f"{sim} for {etf}" + (f" ({proxy})" if proxy else ""))
                return sim
        if first is not None:
            late.append(f"{etf} (from {first.date()})")
    if use not in known:
        raise ParseError(f"This model needs {etf}, which has no price data here.")
    return use


def _history_notes(name: str, swaps: list, late: list, etfs: list, notes: list[str]) -> None:
    start = getattr(_TL, "start", None)
    known = _known()
    if swaps:
        notes.append(f"Start {str(start)[:10]} is before some of the funds existed: using the long-history series "
                     + "; ".join(swaps) + ".")
    if late:
        notes.append("No long-history series for " + ", ".join(late) + ": the backtest can only start once it has data.")
    if not start and any(any(sim in known for sim, _ in SIM_FOR.get(e, [])) for e in etfs):
        notes.append(f"{name} uses the ETFs; add e.g. 'since 1973' to extend it back with the simulated long-history series.")


# Named tactical models (Portfolio Visualizer's "Tactical Asset Allocation" models):
# Faber's GTAA / Ivy timing: 5 asset classes in equal shares, each held only while its month-end price is above its
# 10-month simple moving average, otherwise that fifth in cash (T-bills).
GTAA_SLEEVES = [("SPY", "US stocks (S&P 500)"), ("EFA", "developed ex-US stocks (MSCI EAFE)"),
                ("IEF", "intermediate Treasuries (7-10 year)"), ("VNQ", "US REITs"), ("DBC", "commodities")]
GTAA_RX = (r"(?:the )?(?:(?:meb )?faber'?s?(?: gtaa| ivy(?: portfolio)?(?: timing| with timing)?| tactical asset allocation"
           r"| global tactical asset allocation| timing model| 10[- ]month (?:sma|moving average) (?:timing|model))"
           r"|gtaa(?:[- ]?5)?|global tactical asset allocation(?: 5)?|ivy(?: portfolio)? (?:timing|with timing)|timed ivy(?: portfolio)?)"
           r"(?: model| portfolio| strategy| timing model)?")
# Antonacci's Global Equities Momentum: when US stocks' 12-month total return beats T-bills', hold the better of US and
# ex-US stocks over those 12 months; otherwise aggregate bonds.
GEM_RX = (r"(?:the )?(?:(?:gary )?antonacci'?s?(?: gem| global equit(?:y|ies) momentum| dual momentum(?: gem)?)"
          r"|dual momentum gem|gem dual momentum|global equit(?:y|ies) momentum(?: \(?gem\)?)?|gem)"
          r"(?: model| portfolio| strategy)?")


# the named tactical models whose N-month lookbacks default to calendar months (month-end to month-end)
TACTICAL_MONTH_END = ("GTAA (Faber)", "Global Equities Momentum (Antonacci)")


def _tactical_model(text: str, notes: list[str]) -> dict | None:
    """'GTAA', 'Faber GTAA', 'Ivy timing' -> the 5-sleeve 10-month moving-average timing model; 'Antonacci GEM',
    'global equities momentum', 'dual momentum GEM' -> Global Equities Momentum. The funds are swapped for their
    long-history series when the backtest starts before they existed."""
    s = " ".join(text.strip().lower().split())
    if re.fullmatch(GTAA_RX, s):
        swaps, late = [], []
        kids = []
        for etf, _ in GTAA_SLEEVES:
            tk = _long_history(etf, swaps, late)
            on, rule = _condition_on(f"{tk} is above its 10 month moving average", tk)
            kids.append({"if": rule, "on": on, "then": {"asset": tk}, "else": {"cash": True}})
        _TL.tactical = "GTAA (Faber)"
        notes.append("GTAA / Ivy timing (Mebane Faber, 'A Quantitative Approach to Tactical Asset Allocation', 2007): "
                     + ", ".join(f"{etf} ({about})" for etf, about in GTAA_SLEEVES)
                     + f", a fifth each; each sleeve is held only while its price is above its 10-month simple moving "
                       f"average of month-end prices (`{kids[0]['if']}` on {kids[0]['on']}), otherwise that fifth is in "
                       "cash (T-bills). Checked monthly unless you say otherwise.")
        _history_notes("GTAA", swaps, late, [e for e, _ in GTAA_SLEEVES], notes)
        return {"weights": "equal", "children": kids}
    if re.fullmatch(GEM_RX, s):
        swaps, late = [], []
        us, intl, bonds = (_long_history(e, swaps, late) for e in ("SPY", "VEU", "AGG"))
        n = _period(12, "month")
        _TL.tactical = "Global Equities Momentum (Antonacci)"
        trading = getattr(_TL, "months", None) == "trading"
        how = (f"{n} trading days, 21 a month" if trading else
               f"{n} = 12 months, measured on calendar months: month-end to month-end")
        notes.append(f"Global Equities Momentum (Gary Antonacci, 'Dual Momentum Investing', 2014): when US stocks ({us}) "
                     f"returned more than T-bills over the last 12 months (`tret({n}) > tbill_ret({n})` on {us}; {how}), "
                     f"hold the better of US ({us}) and ex-US stocks ({intl}) by 12-month total return; otherwise aggregate "
                     f"bonds ({bonds}). Checked monthly unless you say otherwise.")
        _history_notes("GEM", swaps, late, ["SPY", "VEU", "AGG"], notes)
        return {"if": f"tret({n}) > tbill_ret({n})", "on": us,
                "then": {"filter": {"select": "top", "n": 1, "by": f"tret({n})"}, "universe": "children",
                         "children": [{"asset": us}, {"asset": intl}]},
                "else": {"asset": bonds}}
    return None


# Portfolio Visualizer's asset-class names, read after a weight ("40% US stock market, 20% international
# stocks, 40% total bond"): (phrase regex, series to use - the first with data wins, fund last -, label).
# The long-history series are the named fund once it (or its mutual-fund twin) exists, so an asset class
# covers the longest possible period. Longer phrases first ("US small cap value" before "US small cap").
_US = r"(?:(?:the )?u\.?s\.? |american |domestic )"
_CAP = r"[- ]?caps?(?: stocks?| equit(?:y|ies))?"
_TSY = r"(?:u\.?s\.? )?(?:government |gov't |govt )?treasur(?:y|ies)(?: bonds?| notes?)?"
ASSET_CLASSES = [
    (rf"(?:total )?{_US}?(?:total )?stock market|{_US}?(?:stocks|equities)|total market", ["VTISIM", "SPYSIM", "VTI"],
     "US total stock market"),
    (rf"{_US}?large{_CAP} value", ["VTVSIM", "VTV"], "US large-cap value"),
    (rf"{_US}?large{_CAP} growth", ["VUGSIM", "VUG"], "US large-cap growth"),
    (rf"{_US}?large{_CAP}(?: blend)?", ["SPYSIM", "SPY"], "US large caps (S&P 500)"),
    (rf"{_US}?mid{_CAP} value", ["VOESIM", "VOE"], "US mid-cap value"),
    (rf"{_US}?mid{_CAP} growth", ["VOTSIM", "VOT"], "US mid-cap growth"),
    (rf"{_US}?mid{_CAP}(?: blend)?", ["MIDSIM", "MDY"], "US mid caps"),
    (rf"{_US}?small{_CAP} value", ["VBRSIM", "VBR"], "US small-cap value"),
    (rf"{_US}?small{_CAP} growth", ["VBKSIM", "VBK"], "US small-cap growth"),
    (rf"{_US}?small{_CAP}(?: blend)?", ["VBSIM", "VB"], "US small caps"),
    (r"(?:unhedged international (?:government |treasury )?bonds?|international (?:government |treasury )?bonds? \(?unhedged\)?|"
     r"global bonds?(?: ex[- ]u\.?s\.?)? \(?unhedged\)?)", ["BWXSIM", "BWX"], "international government bonds, unhedged"),
    (r"international (?:government )?bonds?|global bonds?(?: ex[- ]u\.?s\.?)?", ["BNDXSIM", "BNDX"],
     "international bonds (USD-hedged)"),
    (r"international small[- ]?caps? value(?: stocks)?|international small value", ["AVDVSIM", "AVDV"],
     "international (developed ex-US) small-cap value"),
    (r"international small[- ]?caps?(?: stocks)?", ["SCZSIM", "SCZ"], "international (developed ex-US) small caps"),
    (r"international (?:large[- ]?cap )?value(?: stocks)?", ["EFVSIM", "EFV"], "international (developed ex-US) value"),
    (r"international developed(?: markets?)?(?: stocks| equities)?|developed markets?(?: ex[- ]u\.?s\.?)?(?: stocks| equities)?|"
     r"(?:msci )?eafe", ["EFASIM", "EFA"], "international developed stocks"),
    (r"(?:total )?international(?: stock market| stocks| equities)?|(?:global |world )?ex[- ]u\.?s\.? stocks",
     ["VXUSSIM", "EFASIM", "VXUS"], "international stocks (developed + emerging)"),
    (r"emerging markets? (?:bonds|debt)", ["EMBSIM", "EMB"], "emerging-market bonds"),
    (r"emerging markets?(?: stocks| equities)?", ["VWOSIM", "EEMSIM", "VWO"], "emerging-market stocks"),
    (r"european stocks|europe(?:an)? equities|europe", ["VGKSIM", "VGK"], "European stocks"),
    (r"japan(?:ese stocks)?", ["EWJSIM", "EWJ"], "Japanese stocks"),
    (r"(?:us |u\.s\. )?reits?|real estate(?: investment trusts)?", ["VNQSIM", "VNQ"], "US REITs"),
    (r"gold", ["GLDSIM", "GLD"], "gold"),
    (r"commodit(?:y|ies)(?: futures)?", ["DBCSIM", "DBC"], "commodity futures"),
    (rf"(?:{_US})?(?:total bond(?: market)?|aggregate bonds?|(?:investment[- ]grade )?bonds? market|bonds)",
     ["BNDSIM", "BND"], "US total bond market"),
    (rf"short[- ]term {_TSY}", ["SHYSIM", "SHY"], "short-term Treasuries"),
    (rf"intermediate(?:[- ]term)? {_TSY}", ["IEFSIM", "IEF"], "intermediate-term Treasuries"),
    (rf"long[- ]term {_TSY}", ["TLTSIM", "TLT"], "long-term Treasuries"),
    (r"tips|treasury inflation[- ]protected securities|inflation[- ]protected (?:bonds|securities)", ["TIPSIM", "TIP"], "TIPS"),
    (r"long[- ]term (?:investment[- ]grade )?corporate bonds?|long[- ]term corporates", ["VCLTSIM", "VCLT"],
     "long-term corporate bonds"),
    (r"(?:investment[- ]grade )?corporate bonds?|corporates", ["LQDSIM", "LQD"], "investment-grade corporate bonds"),
    (r"high[- ]yield(?: corporate)?(?: bonds?)?|junk bonds", ["HYGSIM", "HYG"], "high-yield bonds"),
    (r"municipal bonds?|munis?|muni bonds?", ["MUBSIM", "MUB"], "municipal bonds"),
    (r"t-?bills|treasury bills", ["BILSIM", "BIL"], "T-bills"),
    (r"silver", ["SLVSIM", "SLV"], "silver"),
    # no maturity given: intermediate-term (7-10 years), with a Warning naming the other readings (last, so that
    # "treasury bills" and "long-term treasuries" are matched by their own rows first)
    (rf"{_TSY}|government bonds|gov't bonds|govt bonds", ["IEFSIM", "IEF"], "intermediate-term Treasuries"),
]
# asset-class words that need a Warning: the reading chosen and how to get the other ones
_CLASS_AMBIGUOUS = {
    "intermediate-term Treasuries": "no maturity was given, so Treasuries are read as intermediate-term (7-10 years, IEF); "
                                    "say 'long-term treasuries' (TLT) or 'short-term treasuries' (SHY) for the others",
}
ASSET_CLASS_RX = "|".join(f"(?:{p})" for p, *_ in ASSET_CLASSES)


def _asset_class_ticker(phrase: str) -> tuple[str, str] | None:
    """'US small cap value' -> ('VBRSIM', note) (the first series of the class with data), or None."""
    known = _known()
    for pat, opts, label in ASSET_CLASSES:
        if re.fullmatch(pat, phrase.strip(), re.I):
            for t in opts:
                if t in known:
                    about = data.sim_about(t) if t in data.SIMS else None
                    return t, (f"'{phrase.strip()}' ({label}) is read as {t}" + (f": {about}" if about else "")
                               + (". A long-history series that is the fund itself once the fund exists; name a fund "
                                  f"(e.g. {opts[-1]}) to use the fund alone" if t.endswith("SIM") else "")
                               + (f" ({t} stands in until the data job builds {opts[0]})" if t != opts[0] else "") + ".")
            return None
    return None


def _custom_names_rx() -> str:
    """A regex alternative of the imported (custom) series' names that a plain ticker pattern (1-5 letters) does
    not cover ('myfundx', 'fund2'), longest first; '' when there are none."""
    try:
        names = [p.stem for p in data.CUSTOM.glob("*.csv")] if data.CUSTOM.exists() else []
    except OSError:
        names = []
    names = sorted((n.lower() for n in names if len(n) > 5 or not n.isalpha()), key=len, reverse=True)
    return "|".join(re.escape(n) for n in names)


def resolve_asset_names(text: str) -> tuple[str, list[str]]:
    """Asset-class names in a list of tickers or tickers with weights ('US Stock Market 60, Total Bond Market 40',
    'VTI, Gold, TLT') -> their series, as the parser and the Backtest grid read them ('VTISIM 60, BNDSIM 40');
    plus one note per name. A single word in capitals that is a ticker with data stays that ticker ('GOLD' is
    Barrick Gold, 'Gold' the asset class); tickers and numbers are left alone."""
    text = str(text or "")
    if not re.search(r"[A-Za-z]{2}", text):
        return text, []
    known = _known()
    notes: list[str] = []

    def fix(m):
        name = m.group(0)
        if " " not in name and name == name.upper() and data.canonical(name) in known:
            return name
        hit = _asset_class_ticker(name)
        if not hit:
            return name
        if hit[1] not in notes:
            notes.append(hit[1])
        return hit[0]
    out = re.sub(rf"(?i)(?<![\w.$^-])(?:{ASSET_CLASS_RX})(?![\w-])", fix, text)
    return out, notes


def resolve_asset_list(items) -> tuple[list[str], list[str]]:
    """A list of tickers and asset-class names (['US Stock Market', 'TLT'], or one string 'US Stock Market, TLT')
    -> (canonical tickers, notes). Items that hold spaces are separate names; otherwise the words are read
    together, so an unquoted 'US Stock Market TLT' on the command line works too."""
    if isinstance(items, str):
        items = [items]
    items = [str(x) for x in (items or []) if str(x).strip()]
    joined = ", ".join(items) if any(" " in x.strip() for x in items) else " ".join(items)
    out, notes = resolve_asset_names(joined)
    ticks: list[str] = []
    for t in out.replace(",", " ").split():
        c = data.canonical(t)
        if c not in ticks:
            ticks.append(c)
    return ticks, notes


def _class_row(phrase: str):
    """The ASSET_CLASSES row a phrase names in full ('gold', 'US small cap value'), or None."""
    for row in ASSET_CLASSES:
        if re.fullmatch(row[0], phrase.strip(), re.I):
            return row
    return None


def _first_date(t: str) -> str | None:
    try:
        return str(data.load(t).index[0].date())
    except Exception:  # noqa: BLE001 - no data: reported when the portfolio runs
        return None


# tickers that are also asset-class words, for the Warning ("gold" the asset class, GOLD the stock)
_CLASS_WORD_TICKERS = {"GOLD": "Barrick Gold, a stock"}


def class_series(phrase: str, start=None, mixed: bool = False) -> tuple[str, list[str]] | None:
    """An asset-class word or name typed as a holding ('gold', 'Gold', 'bonds', 'REITs', 'US small cap value') ->
    (the series to hold, notes), or None when the phrase is no asset class. The one rule, for sentences and the
    Backtest grid alike:

    - the class's fund (gold = GLD, REITs = VNQ, bonds = AGG next to tickers / BND otherwise ...);
    - its long-history series (GLDSIM: gold before GLD existed, then GLD itself, so over the fund's life the two
      are the same) when the backtest starts before the fund's first price: the start given, or - with no start -
      a portfolio of asset classes alone, which starts as early as the data allows (Portfolio Visualizer's
      asset-class mode). A portfolio that also names tickers ('60% SPY and 40% gold') with no start holds the fund.

    A word in capitals that is a ticker with data is never an asset class (the caller checks: 'GOLD' is Barrick
    Gold). When the lowercase word is also such a ticker, a Warning names both readings."""
    raw = " ".join(str(phrase or "").split())
    row = _class_row(raw)
    if not row:
        return None
    pat, opts, label = row
    known = _known()
    low = raw.lower()
    fund = next((t for t in reversed(opts) if not t.endswith("SIM") and t in known), None)
    sims = [(t, None) for t in opts if t.endswith("SIM") and t in known]
    if fund:
        sims = [(x, pr) for x, pr in SIM_FOR.get(fund, []) if x in known] or sims
    sim, proxy = sims[0] if sims else (None, None)
    # what the long-history series is: the fund itself from its first day, or (a proxy) another fund's history
    then = f"the long-history series that is {fund} itself once {fund} exists" if not proxy else proxy
    first = _first_date(fund) if fund else None
    start = str(start)[:10] if start else None
    use_sim = sim is not None and (fund is None or (start is not None and (first is None or start < first))
                                   or (start is None and not mixed))
    notes: list[str] = []
    if use_sim:
        about = data.sim_about(sim) if sim in data.SIMS else None
        if start and fund:
            notes.append(f"'{raw}' ({label}) is read as {sim}: the start {start} is before {fund} existed ({first}), so "
                         f"{sim} stands in" + (f" ({about})" if about and not proxy else f" ({proxy})" if proxy else "")
                         + (f"; from {first} on it is {fund} itself" if not proxy else "") + f". Say {fund} to hold the "
                         f"fund alone (the backtest then starts in {first[:4] if first else 'its first year'}).")
        else:
            notes.append(f"'{raw}' ({label}) is read as {sim}" + (f": {about}" if about else "")
                         + (f". A long-history series that is the fund itself once the fund exists; name a fund (e.g. "
                            f"{fund or opts[-1]}) to use the fund alone" if sim.endswith("SIM") else "") + ".")
        use = sim
    elif fund:
        use = fund
        msg = f"'{raw}' ({label}) is read as {fund}"
        if start is None and sim:
            msg += (f", the fund (from {first}); give an earlier start (e.g. 'since 1972') to extend it back with "
                    f"{sim}, {then}")
        elif start is not None and first and start < first:
            msg += f"; its history starts on {first}, so the backtest can only start then"
        notes.append(msg + ".")
    else:
        return None
    if label in _CLASS_AMBIGUOUS:
        notes.append(f"Warning: '{raw}': {_CLASS_AMBIGUOUS[label]}.")
    tk = data.canonical(raw.upper())
    if " " not in raw and tk in known and tk != use:
        what = _CLASS_WORD_TICKERS.get(tk, "the stock or fund with that symbol")
        notes.append(f"Warning: '{raw}' is read as the asset class {label} ({use}), not the ticker {tk} ({what}); "
                     f"write {tk} in capitals to hold that ticker.")
    return use, notes


def _class_mixed(text: str) -> bool:
    """True when the text names real tickers (not SIM series) in capitals: asset-class words next to them are the
    funds (see class_series)."""
    known = _known()
    return any(t in known and not t.endswith("SIM") and t not in NOT_TICKERS
               for t in re.findall(r"(?<![\w^$])[$^]?([A-Z]{1,5}(?:-[A-Z])?)\b", _sub_outside(r"`[^`]*`", " ", text)))


def _class_word(phrase: str) -> str | None:
    """A holding typed as an asset-class word ('gold', 'Gold', 'real estate'), not an all-capitals ticker with data ->
    the series (class_series, with the start of the portfolio being parsed), notes added; else None."""
    w = " ".join(str(phrase or "").split())
    if not w or (" " not in w and w == w.upper() and data.canonical(w) in _known()):
        return None
    hit = class_series(w, getattr(_TL, "start", None), getattr(_TL, "mixed", False))
    if not hit:
        return None
    for n in hit[1]:
        _note(n)
    return hit[0]


def _asset_class_names(text: str) -> str:
    """Portfolio Visualizer asset-class names after a weight ('40% US stock market', '40% gold') -> the series
    class_series reads with no start (the fund next to real tickers, the long-history series otherwise). Only right
    after 'NN%' (optionally 'in' / 'of'), so words elsewhere in a sentence are untouched. Each substitution is
    recorded (_TL.class_subs): the holdings parser settles it once the start is known (a start before the fund
    existed swaps in the long-history series) and adds the notes then."""
    mixed = _class_mixed(text)
    _TL.mixed = mixed
    subs = getattr(_TL, "class_subs", None)
    if subs is None:
        subs = _TL.class_subs = {}

    def fix(m):
        name = m.group("name")
        if " " not in name and name == name.upper() and data.canonical(name) in _known():
            return m.group(0)       # 'GOLD' in capitals is the ticker
        # a blended benchmark ("benchmark 60% US Stock Market and 40% Total Bond Market") takes the long-history
        # series, as benchmarks do everywhere (metrics.benchmark_label, the grid's benchmark box)
        bench = re.search(r"(?i)\b(?:benchmark(?:ed)?|vs\.?|versus|compared (?:to|with))\b[^;]*$", m.string[:m.start()])
        hit = class_series(name, None, False if bench else mixed)
        if not hit:
            return m.group(0)
        if bench:
            for n_ in hit[1]:
                _note(n_)
        else:
            subs.setdefault(hit[0], name)
        return m.group("pre") + hit[0]
    return _sub_outside(rf"(?i)(?P<pre>\d+(?:\.\d+)?% (?:in |of |to )?(?:the )?)(?P<name>{ASSET_CLASS_RX})(?![\w-])", fix, text)


def _settle_class(ticker: str) -> str:
    """A holding substituted for an asset-class name by _asset_class_names -> the series for the actual start, with
    its notes (once per name)."""
    subs = getattr(_TL, "class_subs", None) or {}
    name = subs.get(ticker)
    if not name:
        return ticker
    done = getattr(_TL, "class_done", None)
    if done is None:
        done = _TL.class_done = set()
    done.add(ticker)
    return _class_word(name) or ticker


def _unsettled_class_notes() -> None:
    """Notes for asset-class names substituted in a sentence that never reached the holdings parser (a signal
    strategy): read with no start."""
    subs = getattr(_TL, "class_subs", None) or {}
    done = getattr(_TL, "class_done", None) or set()
    for t, name in subs.items():
        if t not in done:
            hit = class_series(name, None, getattr(_TL, "mixed", False))
            for n in (hit[1] if hit else []):
                _note(n)


# words that may be left over after everything meaningful was recognised
STOP = set("""
a an the it its it's is are was were be been being this that these those then than also just only
when whenever if once and or but so of on at in into for to with by from as per each every any all
stock stocks share shares price prices ticker tickers symbol symbols etf etfs fund funds day days
today same trade trades trading traded position positions order orders strategy backtest test please
i me my we our us you want would like let lets let's run show see what happens using use rule rules
signal signals market markets just simply both plus ones one's there here now time times they them their
buy buying bought sell selling sold hold holding keep own go goes going get enter exit cover long
short purchase invest investing put allocate allocation portfolio money account close
""".split())


class ParseError(ValueError):
    pass


class AmbiguousTickers(ParseError):
    """A condition written once for a bare list of tickers ("SPY, QQQ 10 day RSI ..."): every one, or any one?"""


# notes raised deep inside condition / value parsing (which has no notes list of its own); parse()
# resets them and adds them to the result's notes. Thread-local: the site parses in several threads.
_TL = threading.local()


def _note(msg: str) -> None:
    lst = getattr(_TL, "notes", None)
    if lst is not None and msg not in lst:
        lst.append(msg)


def _pending_notes() -> list[str]:
    return list(getattr(_TL, "notes", None) or [])


@dataclass
class Ctx:
    """Placeholders for the series a condition refers to."""
    c: str = "close"
    o: str = "open"
    h: str = "high"
    l: str = "low"
    v: str = "volume"
    total: bool = False   # returns are total returns (dividends reinvested): portfolio conditions

    @classmethod
    def for_ticker(cls, t: str | None, total: bool = False) -> "Ctx":
        if not t:
            return cls(total=total)
        s = f'sym("{t}")'
        return cls(f"{s}.close", f"{s}.open", f"{s}.high", f"{s}.low", f"{s}.volume", total)

    @property
    def base(self) -> bool:
        return self.c == "close"

    @property
    def tr(self) -> str:
        """The dividend-reinvested price series."""
        return "tr" if self.base else f"{self.c[:-len('.close')]}.tr"

    def ret(self, n: int) -> str:
        """n-bar return: total return for portfolio conditions, price return for signal rules."""
        if self.total:
            return f"tret({self.tr}, {n})"
        if n == 1:
            return "change" if self.base else f"ret({self.c}, 1)"
        return f"ret({self.c}, {n})"


# ----------------------------------------------------------------- helpers

def _normalize(text: str) -> str:
    t = text.replace("’", "'").replace("–", "-").replace("—", "-").replace("“", '"').replace("”", '"')
    t = re.sub(r"\s+", " ", t).strip()
    t = t.replace("≥", " >= ").replace("≤", " <= ").replace("＞", ">").replace("＜", "<")
    t = _sub_outside(r"(?i)\b(?:greater than or equal to|equal to or greater than|higher than or equal to|"
                     r"above or equal to|at or above|greater or equal to)(?= )", ">=", t)
    t = _sub_outside(r"(?i)\b(?:less than or equal to|equal to or less than|lower than or equal to|"
                     r"below or equal to|at or below|less or equal to)(?= )", "<=", t)
    # "RSI is 70 or more" = at least 70, "is 30 or less" = at most 30 (a percentage before a direction, "5% or more
    # above its average", is left to the patterns that read it)
    t = _sub_outside(r"(?i)\b(is|are|was|were|stays?|remains?|reads?|closes?|of) (\$?-?\d+(?:\.\d+)?%?) or "
                     r"(more|higher|greater|above|less|lower|fewer|below)\b(?! (?:above|below|over|under|from|off|than|in|of)\b)",
                     lambda m: f"{m.group(1)} {'at least' if m.group(3).lower() in ('more', 'higher', 'greater', 'above') else 'at most'} "
                               f"{m.group(2)}", t)
    # contractions ("don't rebalance" and "let's" stay as they are)
    t = re.sub(r"(?i)\b(it|that|this|there|he|she)'s\b", r"\1 is", t)
    t = re.sub(r"(?i)\b(is|does|has|was|are|have|did|were)n't\b", r"\1 not", t)
    t = re.sub(r"(?i)\b(they|we|you)'re\b", r"\1 are", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"(\d)\s*percent\b", r"\1%", t, flags=re.I)
    t = _sub_outside(r"(?i)(?<![\w.$])(\d+(?:\.\d+)?) ?(?:cents?\b|¢)", lambda m: f"${float(m.group(1)) / 100:g}", t)
    t = re.sub(r"\bper cent\b", "%", t, flags=re.I)
    t = _number_words(t)
    t = re.sub(r"\b(\d+)-(day|week|month|year|period|bar|session)s?\b", r"\1 \2", t, flags=re.I)
    t = re.sub(r"\b(\d+)(d|w|m)\b(?!\s*%)", lambda m: m.group(1) + {"d": " day", "w": " week", "m": " month"}[m.group(2).lower()], t)
    t = re.sub(r"\bovernight\b", "1 day", t, flags=re.I)
    t = re.sub(r"\ba (day|week|month)\b(?! in a row)", r"1 \1", t, flags=re.I)
    t = re.sub(r"\bhalf\b", "50%", t, flags=re.I)
    t = re.sub(r"\ba third\b", "33.3333%", t, flags=re.I)
    t = re.sub(r"\ba quarter\b(?! of)", "25%", t, flags=re.I)
    while re.search(r"(\d),(\d{3})\b", t):
        t = re.sub(r"(\d),(\d{3})\b", r"\1\2", t)  # 1,000,000 -> 1000000
    # share classes: "BRK.B" / "BF.B" are stored as BRK-B / BF-B (Yahoo's spelling)
    def share_class(m):
        dashed = f"{m.group(1)}-{m.group(2)}"
        if dashed in _known():
            _note(f"'{m.group(0)}' read as {dashed} (share classes are written with a dash here).")
            return dashed
        return m.group(0)
    t = _sub_outside(r"(?<![\w.$^-])([A-Z]{1,5})\.([A-Z])(?![\w.-])", share_class, t)
    t = re.sub(r"\$\s*(\d+(?:\.\d+)?)\s*k\b", lambda m: "$" + str(int(float(m.group(1)) * 1000)), t, flags=re.I)
    t = re.sub(r"\$\s*(\d+(?:\.\d+)?)\s*m(?:illion)?\b", lambda m: "$" + str(int(float(m.group(1)) * 1_000_000)), t, flags=re.I)
    return t


def _mask(s: str, parens: bool = True) -> str:
    """A same-length copy of `s` with the inside of `backtick` blocks (and, with parens=True, of balanced
    (...) and [...] groups) replaced by NUL characters, so separators inside them are invisible to
    splitting and matching. Unbalanced brackets are left alone."""
    out = list(s)
    tick = None
    stack: list[tuple[str, int]] = []
    pairs = {")": "(", "]": "["}
    for i, ch in enumerate(s):
        if ch == "`":
            if tick is None:
                tick = i
            else:
                for j in range(tick + 1, i):
                    out[j] = "\0"
                tick = None
            continue
        if tick is not None or not parens:
            continue
        if ch in "([":
            stack.append((ch, i))
        elif ch in ")]":
            if stack and stack[-1][0] == pairs[ch]:
                _, j0 = stack.pop()
                if not stack:
                    for j in range(j0 + 1, i):
                        out[j] = "\0"
            else:
                stack.clear()   # unbalanced: stop treating what came before as a group
    return "".join(out)


class _MM:
    """A match found on a masked string, reporting groups from the original string."""

    def __init__(self, m: re.Match, s: str):
        self.m, self.s = m, s

    def group(self, i=0):
        a, b = self.m.span(i)
        return None if a < 0 else self.s[a:b]

    def groups(self):
        return tuple(self.group(i) for i in range(1, (self.m.re.groups or 0) + 1))

    def groupdict(self):
        return {k: self.group(k) for k in self.m.re.groupindex}

    def start(self, i=0):
        return self.m.start(i)

    def end(self, i=0):
        return self.m.end(i)

    def span(self, i=0):
        return self.m.span(i)


def _msearch(pattern: str, s: str, flags=re.I, parens: bool = True, full: bool = False, match: bool = False):
    """re.search / re.fullmatch / re.match on `s` that cannot see inside backticks or brackets."""
    fn = re.fullmatch if full else (re.match if match else re.search)
    m = fn(pattern, _mask(s, parens), flags)
    return _MM(m, s) if m else None


def _msplit(s: str, pattern: str, flags=re.I, parens: bool = True) -> list[str]:
    """re.split that ignores separators inside backticks (and brackets)."""
    mk = _mask(s, parens)
    out, last = [], 0
    for m in re.finditer(pattern, mk, flags):
        if m.end() == m.start():
            continue
        out.append(s[last:m.start()])
        last = m.end()
    out.append(s[last:])
    return out


def _known() -> set[str]:
    return set(data.available_tickers())


def _custom_long_names() -> list[str]:
    """Imported custom series whose names the ticker pattern (1-5 letters) does not read: longer or with digits."""
    if not data.CUSTOM.exists():
        return []
    return [p.stem for p in data.CUSTOM.glob("*.csv") if not re.fullmatch(r"[A-Z]{1,5}", p.stem)]


def find_tickers(text: str, strict: bool = False) -> list[str]:
    """Tickers mentioned in `text` (original casing), in order of appearance.

    With strict=True, ticker-looking words without price data raise ParseError instead of being ignored.
    """
    known = _known()
    found: list[tuple[int, str]] = []
    unknown: list[str] = []
    stripped = re.sub(r"`[^`]*`", " ", text)
    # a letter joined by '&' is part of a name, never a ticker: 'S&P 500' (not P), 'AT&T', 'P&L', 'R&D'; nor is
    # one after '%' ('%K crosses above %D': the stochastic lines, not Dominion Energy)
    for m in re.finditer(r"(?<![\w^%])(?<![A-Za-z]&)(\$|\^)?([A-Z]{1,5}(?:SIM|-USD|[.-][A-Z](?![\w-]))?)\b(?!&[A-Za-z])", stripped):
        sym = m.group(2)
        pre = m.group(1) or ""
        if pre == "^":
            sym = "^" + sym
        cand = data.canonical(sym)
        if sym in NOT_TICKERS and not pre:
            continue
        if cand in known:
            found.append((m.start(), cand))
        elif pre or (len(sym) >= 2 and sym.isupper() and sym not in NOT_TICKERS):
            if strict and data.fetch_on_demand(cand):
                found.append((m.start(), cand))
            else:
                unknown.append(sym)
    # imported custom series may have longer names (MYFUND, up to 10 letters or digits): matched as whole words
    for sym in _custom_long_names():
        for m in re.finditer(rf"(?<![\w^$]){re.escape(sym)}(?![\w-])", stripped):
            found.append((m.start(), sym))
    low = stripped.lower()
    for name, sym in sorted(COMPANIES.items(), key=lambda kv: -len(kv[0])):
        for m in re.finditer(rf"(?<![\w$^]){re.escape(name)}\b", low):
            if any(p <= m.start() < p + 6 for p, _ in found):
                continue
            if sym in known:
                found.append((m.start(), sym))
            elif strict:
                unknown.append(f"{name} ({sym})")
    if strict and unknown:
        from . import coverage

        def one(u):
            s = u.split(" (")[-1].rstrip(")") if " (" in u else u
            if coverage.lookup(s):
                return f"{u} is not downloaded yet"
            sug = data.suggest(s)
            return f"unknown ticker {u}" + (f" (closest: {', '.join(sug)})" if sug else "")
        msg = "; ".join(one(u) for u in dict.fromkeys(unknown))
        syms = ", ".join(dict.fromkeys(u.split(" (")[-1].rstrip(")") if " (" in u else u for u in unknown))
        hint = coverage.missing_hint(syms.split(", "), data.offline())
        raise ParseError(msg[0].upper() + msg[1:] + f" - no price data for {syms}. {hint} "
                         "(`python -m backtester tickers` lists all.)")
    out: list[str] = []
    for _, s in sorted(found):
        if s not in out:
            out.append(s)
    return out


# tickers that are also English words (or parser vocabulary): never read from lowercase text
ENGLISH_TICKERS = set("""
all app arm be cat cost cure dell dis dust fang fast flex fox gold corn has jets java life lit lite mar mat on
qual shop size spot tail tan team tip trip ups usd vip yang are low high open close day hold buy sell top cash bond
bonds rate rates ring now well real key sun eat run win net plus ever tell love fun main dash pep coin hood
bar bars rest risk trail cloud candle
""".split())


# lowercase words the parser's own phrases read as that ticker ("... that beat bil"), so still a ticker
_VOCAB_TICKERS = {"bil"}
_VOCAB_CACHE: list[frozenset] = []


def _vocabulary() -> frozenset:
    """Every lowercase word of the parser's vocabulary: the hand lists (STOP, NOT_TICKERS, ENGLISH_TICKERS,
    numbers, days, months, company names) plus every 3-5 letter lowercase word written in a phrase pattern
    or table of this module and index_universes (docstrings and message f-strings aside), plus the names of
    the rule language (expr). A word the parser itself understands ('band', 'mean', 'cross', 'rsi') is never
    a lowercase ticker, however many ticker files the data refresh adds (BAND, MEAN, ...)."""
    if _VOCAB_CACHE:
        return _VOCAB_CACHE[0]
    import pathlib
    words = set(STOP) | {w.lower() for w in NOT_TICKERS} | ENGLISH_TICKERS | set(WORD_NUMS) | set(DOW) | set(MONTHS)
    for name in COMPANIES:
        words.update(name.split())
    here = pathlib.Path(__file__).parent
    for fn, idents in (("parser.py", False), ("index_universes.py", False), ("expr.py", True)):
        try:
            tree = ast.parse((here / fn).read_text())
        except (OSError, SyntaxError):
            continue
        skip = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant):
                skip.add(id(n.value))                      # docstrings
            elif isinstance(n, ast.JoinedStr):
                skip.update(id(v) for v in n.values)       # f-strings: messages naming example tickers
        for n in ast.walk(tree):
            if not (isinstance(n, ast.Constant) and isinstance(n.value, str)) or id(n) in skip:
                continue
            if idents:   # the rule language: identifier strings only ('bb_lower' -> bb, lower)
                if re.fullmatch(r"[a-z_][a-z0-9_]*", n.value):
                    words.update(w for w in n.value.split("_") if w)
            else:
                words.update(re.findall(r"(?<![A-Za-z])[a-z]{2,6}(?![A-Za-z])", re.sub(r"\\[A-Za-z]", " ", n.value)))
    out = frozenset(w for w in words if w not in _VOCAB_TICKERS)
    _VOCAB_CACHE.append(out)
    return out


def _lowercase_tickers(text: str) -> str:
    """'if tqqq 10 day rsi is above 79 then uvxy else tqqq' -> the same with TQQQ / UVXY: a lowercase word
    is read as a ticker only when it is 3+ letters, has price data and is not an English or vocabulary
    word ('spy' is accepted; 'band', 'mean', 'cross' never are, see _vocabulary). Text inside backticks is
    left alone. Adds a note listing what was read."""
    known = _known()
    words = _vocabulary()
    done: list[str] = []

    def fix(m):
        w = m.group(0)
        if len(w) < 3 or w in words or w.upper() not in known:
            return w
        done.append(w)
        return w.upper()
    parts = re.split(r"(`[^`]*`)", text)
    out = "".join(p if p.startswith("`") else re.sub(r"(?<![\w$^'.-])[a-z]{3,5}(?![\w'-])", fix, p) for p in parts)
    if done:
        _note("Lowercase ticker" + ("s" if len(set(done)) > 1 else "") + " read as "
              + ", ".join(f"'{w}' = {w.upper()}" for w in dict.fromkeys(done)) + ".")
    return out


def _intraday_check(text: str) -> None:
    m = re.search(r"(?i)\b(?:at|by|after|before|around|until) \d{1,2}(?::\d{2})? ?(?:am|pm|a\.m\.|p\.m\.)(?![\w])"
                  r"|\b(?:at|by|after|before|around|until) \d{1,2}:\d{2}\b"
                  r"|\b(?:\d+[- ]?(?:min(?:ute)?s?|hours?|h)|hourly|minute)(?: bars?| charts?| candles?| timeframe)\b", _mask(text, parens=False))
    if m:
        raise ParseError(f"'{text[m.start():m.end()].strip()}': intraday times and bars are not supported - the data is daily "
                         "(one open, high, low and close per day). Use 'at the open', 'at the close' or 'at the next open'.")


def _cmp(word: str) -> str:
    w = word.strip().lower()
    if w in ("<=", "at most", "no more than"):
        return "<="
    if w in (">=", "at least", "no less than"):
        return ">="
    if w in ("below", "under", "less than", "<", "lower than", "beneath", "under its", "falls below"):
        return "<"
    if w in ("above", "over", "greater than", ">", "higher than", "more than"):
        return ">"
    raise ParseError(f"unknown comparison {word!r}")


# the word before a quantity decides whether the bound is included: "more than 5%" / "over 5%" / "exceeds 5%" are
# strict, "at least 5%" / "5% or more" include it (and "less than" / "under" are strict, "at most" / "or less" not)
_STRICT_Q = re.compile(r"(?<![a-z])(?:more than|greater than|higher than|over|above|exceeds?|exceeding|exceeded|in excess of|"
                       r"less than|lower than|fewer than|under|below)\s+(?:by\s+)?\$?-?\d"
                       # "over 10 days" is a period, not a bound
                       r"(?![\d.]*\s*(?:trading\s+)?(?:days?|weeks?|months?|years?|sessions?|bars?|periods?)\b)")
_INCL_Q = re.compile(r"(?<![a-z])(?:at least|no less than|not less than|at most|no more than|not more than)\s+\$?-?\d|"
                     r"\d%?\s+or\s+(?:more|less|greater|higher|lower|fewer)(?![a-z])")


def _strict(phrase: str) -> bool | None:
    """True: the phrase bounds its quantity strictly ("more than 5%"); False: inclusively ("at least 5%",
    "5% or more"); None: no qualifier ("fell 5%")."""
    if _INCL_Q.search(phrase):
        return False
    if _STRICT_Q.search(phrase):
        return True
    return None


def _bound(op: str, phrase: str, default_strict: bool = False) -> str:
    """'<' / '>' or '<=' / '>=' for a bound written with `op`'s direction, strict as the phrase says."""
    st = _strict(phrase)
    st = default_strict if st is None else st
    return op[0] if st else op[0] + "="


CMPW = r"(at least|at most|no less than|no more than|below|under|less than|lower than|beneath|<=?|above|over|greater than|higher than|more than|>=?)"
MA = r"(simple |exponential |weighted )?(?:moving average|moving avg|ma|sma|ema|wma)\b"


MAK = r"(?:(?:simple|exponential|weighted) )?(?:moving average|moving avg|ma|sma|ema|wma)"


def _mat(p: str) -> str:
    """A moving-average term: '50 day moving average', '9 EMA', '20 period EMA', 'EMA(9)', '10 week SMA'."""
    return (rf"(?:(?P<{p}n>\d+) (?:(?P<{p}u>day|week|month|period|bar|session)s? )?(?P<{p}k>{MAK}|average(?! (?:true|volume|of|daily|range|return|gain|loss))\b"
            rf"|mean(?! (?:of|return|reversion|reverting|daily|absolute|deviation))\b)"
            rf"|(?P<{p}k2>{MAK}) ?\( ?(?P<{p}n2>\d+) ?\))")


def _mat_expr(m, p: str, x: str) -> str:
    kind = m.group(p + "k") or m.group(p + "k2")
    return _ma(kind, x, m.group(p + "n") or m.group(p + "n2"), m.group(p + "u"), kind)


def _period(n: str, unit: str | None) -> int:
    n = int(float(n))
    u = (unit or "day").lower()
    if u.startswith("week"):
        return n * 5
    if u.startswith("month"):
        return n * 21
    if u.startswith("year"):
        return n * 252
    return n


ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
            "ninth": 9, "tenth": 10}
ORD = r"first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|\d+(?:st|nd|rd|th)"


def _ord(w: str) -> int:
    return ORDINALS.get(w) or int(re.match(r"\d+", w).group(0))


def _ma(kind: str | None, x: str, n: str, unit: str | None, raw: str) -> str:
    """Moving average text; N-week / N-month averages use weekly / monthly closes (as charts do)."""
    u = (unit or "day").lower()
    xs = "" if x == "close" else f", {x}"
    if u.startswith("month") or u.startswith("week"):
        per = "monthly" if u.startswith("month") else "weekly"
        if (kind and "weight" in kind) or re.search(r"\bwma\b", raw or ""):
            raise ParseError(f"A {per} weighted moving average is not supported; use a {per} SMA or EMA, or backticks.")
        k = "ema" if (kind and "exp" in kind) or re.search(r"\bema\b", raw or "") else "sma"
        return f"{per}_{k}({int(float(n))}{xs})"
    k = "sma"
    if (kind and "exp" in kind) or re.search(r"\bema\b", raw):
        k = "ema"
    elif (kind and "weight" in kind) or re.search(r"\bwma\b", raw):
        k = "wma"
    return f"{k}({x}, {_period(n, unit)})"


_RSI_DEFAULT_NOTE = ("No RSI period given: using the standard 14 days (Wilder's default). Composer always states one "
                     "(its symphonies often use 10); say e.g. '10 day RSI' to choose.")


def _stoch_note(slow: bool, n: int = 14) -> None:
    if slow:
        _note(f"Slow stochastic: %K = the {n}-bar stochastic smoothed over 3 bars (stoch_k({n}, 3)), %D = the 3-bar SMA "
              f"of %K (stoch_d({n}, 3, 3)).")
    else:
        _note(f"Stochastic: TradingView's defaults - %K length {n}, %K smoothing 1 (the raw {n}-bar stochastic, "
              f"stoch_k({n}, 1)), %D = the 3-bar SMA of %K (stoch_d({n}, 1, 3)). Say 'slow stochastic' for %K "
              "smoothing 3.")


def _osc(name: str, n: str | None, ctx: Ctx) -> str:
    """Oscillator expression for a phrase name."""
    name = name.lower().replace("%", "").strip()
    if name.startswith("rsi"):
        if not n:
            _note(_RSI_DEFAULT_NOTE)
        return f"rsi({ctx.c}, {int(float(n or 14))})"
    if not ctx.base:
        raise ParseError(f"{name} of another ticker is not supported in English; use backticks")
    nn = int(float(n)) if n else None
    if name.startswith(("stoch", "slow", "fast", "full")):
        slow = name.startswith(("slow", "full"))
        _stoch_note(slow, nn or 14)
        return f"stoch_k({nn or 14}, {3 if slow else 1})"
    if name.startswith("cci"):
        return f"cci({nn or 20})"
    if name.startswith("williams") or name.startswith("willr") or name == "r":
        return f"willr({nn or 14})"
    if name.startswith("mfi") or name.startswith("money flow"):
        return f"mfi({nn or 14})"
    if name.startswith("adx"):
        return f"adx({nn or 14})"
    raise ParseError(f"unknown indicator {name!r}")


# price-move verbs, split by direction (the sign of a move comes from which list matched)
DOWN_VERBS = (r"(?:down|lower|falls?|fell|fallen|drops?|dropped|declines?|declined|loses?|lost|sinks?|sank|sunk|"
              r"plunges?|plunged|tumbles?|tumbled|slides?|slid|crash(?:es|ed)?|dips?|dipped|decreases?|decreased|slumps?|slumped)")
UP_VERBS = (r"(?:up|higher|rises?|rose|risen|gains?|gained|jumps?|jumped|rall(?:ies|ied|y)|climbs?|climbed|surges?|surged|"
            r"soars?|soared|advances?|advanced|increases?|increased|spikes?|spiked)")
MOVE_AUX = (r"(?:(?:has|have|had) (?:been |gone )?|(?:is|was|are|were) |closes? |closed |trades? |traded |goes |went |"
            r"moves? |moved |gets? |got )?")


OSC = r"(rsi|(?:slow |fast |full )?stochastic(?: %?k)?|(?:slow |fast |full )?stoch|cci|williams %?r|willr|mfi|money flow index|adx)"

# indicators whose lookback can follow them: "the RSI over 10 days", "CCI for the last 20 sessions"
PERIOD_IND = (r"(?:relative strength index|rsi|stochastics?(?: oscillator)?(?: %?k)?|stoch|commodity channel index|cci|williams %?r|"
              r"willr|money flow index|mfi|average directional index|adx|average true range|atr|rate of change|roc|"
              r"(?:simple |exponential |weighted )?(?:moving average|sma|ema|wma|ma))")
UNIT_WORDS = r"(?:trading )?(?:day|week|month|year|bar|session|period)s?"


def _indicator_periods(s: str, cond: bool = True) -> str:
    """'rsi over 10 days' / 'cci for the last 20 sessions' / 'stochastic over the past 14 days' ->
    '10 day rsi' / '20 day cci' / '14 day stochastic', so the lookback can never be read as a
    threshold ('rsi over 10' = RSI above 10). A second lookback is refused."""
    rx = (rf"(?<![a-z0-9%])(?P<ind>{PERIOD_IND})(?P<own>\s*\(\s*\d+\s*\)|\s+\d+(?! ?{UNIT_WORDS}(?![a-z])))?"
          rf"(?P<q> value| reading| indicator| line)?"
          rf"(?P<of> (?:of|for|on) (?!the |last |past |prior |previous |trailing )(?:it|[\^$]?[a-z][a-z0-9.-]{{0,7}}))? "
          rf"(?:(?P<p1>over|for|across) (?:the )?(?:last |past |prior |previous |trailing )?|(?P<p2>in|during|of) (?:the )?(?:last|past|prior|previous|trailing) )"
          rf"(?P<n>\d+) (?:trading )?(?P<u>day|week|month|year|bar|session|period)s?(?![a-z])"
          rf"(?! (?:simple |exponential |weighted )?(?:moving|average|avg|ma|sma|ema|wma|high|low|close|closing|return|rsi|vol|"
          rf"volatility|range|mean|line|band|lows|highs)\b)")

    def fix(m):
        before = s[: m.start()]
        own = m.group("own") or re.search(rf"\d+ {UNIT_WORDS} $", before)
        # at the end of a condition, 'for the last 10 days' says how long it held ('above its 20 day average for the
        # last 10 days'), which is read elsewhere; 'over 200 days' after a bare average is still its lookback
        if cond and not s[m.end():].strip(" ,.;") and (own or (m.group("p1") or m.group("p2")).lower() != "over"):
            return m.group(0)
        if own:
            lead = re.search(rf"(?:\d+ {UNIT_WORDS} )?$", before).group(0)
            raise ParseError(f"'{(lead + m.group(0)).strip()}' gives two lookbacks; keep one, "
                             f"e.g. '{m.group('n')} day {m.group('ind')}'.")
        u = m.group("u")
        u = "day" if u in ("session", "bar", "period") else u
        return f"{m.group('n')} {u.lower()} {m.group('ind')}{m.group('q') or ''}{m.group('of') or ''}"
    return re.sub(rx, fix, s, flags=re.I)


# ----------------------------------------------------------------- conditions

TF_INDICATORS = (r"(?:\d+ (?:day |period |bar )?)?(?:macd|atr|average true range|supertrend|parabolic sar|sar\b|cci|adx|stoch|"
                 r"mfi|money flow|williams|willr|obv|on[- ]balance|bollinger|%b|keltner|roc\b|rate of change|%k|%d|\+di|-di|"
                 r"plus di|minus di|upper|lower|middle)")
TF_CALLS = {"macd", "macd_signal", "macd_hist", "atr", "natr", "supertrend", "sar", "cci", "adx", "stoch_k", "stoch_d", "mfi",
            "willr", "obv", "bb_upper", "bb_lower", "keltner_upper", "keltner_lower", "plus_di", "minus_di", "ret"}


def _wrap_timeframe(e: str, per: str) -> str:
    """Wrap each side of every comparison / crossing that uses a daily-bar indicator in weekly(...) /
    monthly(...): close > supertrend(10, 3) -> close > weekly(supertrend(10, 3))."""
    def has_tf(node):
        return any(isinstance(x, ast.Call) and isinstance(x.func, ast.Name) and x.func.id in TF_CALLS for x in ast.walk(node))

    def wrap(node):
        if has_tf(node):
            return ast.Call(func=ast.Name(id=per, ctx=ast.Load()), args=[node], keywords=[])
        return node

    class T(ast.NodeTransformer):
        def visit_Compare(self, n):
            n.left = wrap(n.left)
            n.comparators = [wrap(x) for x in n.comparators]
            return n

        def visit_Call(self, n):
            if isinstance(n.func, ast.Name) and n.func.id in ("crossover", "crossunder"):
                n.args = [wrap(x) for x in n.args]
                return n
            return self.generic_visit(n)
    tree = T().visit(ast.parse(e, mode="eval"))
    out = ast.unparse(tree)
    if f"{per}(" not in out:
        raise ParseError(f"'{per}' could not be applied here; write the rule in backticks, e.g. `{per}(macd_hist()) > 0`.")
    return out


def _snap_fraction(f: float) -> float:
    """'a third' is written 33.3333%: the exact fraction (1/3, 2/3), so whole-share rounding is not a share short."""
    for k in (3, 6, 7, 9):
        for j in range(1, k):
            if abs(f - j / k) < 5e-6:
                return j / k
    return f


class _SlopeMatch:
    """'the slope of the 20 EMA' and 'the 20 EMA slope' as one match (n, k, s)."""

    def __init__(self, m):
        self.m = m

    def group(self, g):
        if g in ("n", "k"):
            return self.m.group(g) or self.m.group(g + "0")
        return self.m.group(g)


def _review_takes(take, c: str) -> None:
    """Condition patterns from a Composer user's review, registered with parse_condition's `take`:
    'below its 200 day MA by less than 2%' (= below it, and within 2% of it: close < sma and close > sma * 0.98),
    'less than 2% above its 200 day moving average' (above it, by under 2%)."""
    def near(m):
        below = m.group("rel") in ("below", "under", "beneath")
        f = float(m.group("pct")) / 100
        strict = m.group("q") in ("less than", "under")
        ma = _mat_expr(m, "a", c)
        k = f"{1 - f if below else 1 + f:.6g}"
        _note(f"'{m.group(0).strip()}' = {'below' if below else 'above'} {ma} and by {'less than' if strict else 'at most'} "
              f"{m.group('pct')}% ({'more than' if below else 'less than'} {ma} * {k}).")
        if below:
            return f"{c} < {ma} and {c} {'>' if strict else '>='} {ma} * {k}"
        return f"{c} > {ma} and {c} {'<' if strict else '<='} {ma} * {k}"
    take(rf"(?:(?:closes?|trades?|is|stays?|remains?) )?(?P<rel>below|under|beneath|above|over) (?:its |the )?{_mat('a')} by "
         rf"(?P<q>less than|under|at most|no more than) (?P<pct>\d+(?:\.\d+)?)%", near)
    take(rf"(?:(?:closes?|trades?|is|stays?|remains?) )?(?P<q>less than|under|at most|no more than) (?P<pct>\d+(?:\.\d+)?)% "
         rf"(?P<rel>below|under|beneath|above|over) (?:its |the )?{_mat('a')}", near)


def parse_condition(text: str, ctx: Ctx) -> tuple[str | None, str]:
    """Return (expression, leftover words) for one sub-clause."""
    # "daily RSI(2)" is the normal daily-bar indicator
    text = re.sub(rf"(?i)\bdaily (?={TF_INDICATORS}|rsi|close|sma|ema|\d+ (?:day |period )?(?:sma|ema|moving average|ma)\b)", "", text)
    # "the weekly MACD histogram", "weekly ATR(14)", "the weekly supertrend": the indicator on weekly bars
    tfs = re.findall(rf"(?i)\b(weekly|monthly) (?={TF_INDICATORS})", text)
    if tfs and "`" not in text:
        pers = {x.lower() for x in tfs}
        if len(pers) > 1:
            raise ParseError(f"'{text.strip()}' mixes weekly and monthly indicators; split it into two conditions.")
        per = pers.pop()
        if not ctx.base:
            _unsupported(f"a {per} indicator of another ticker")
        e, left = parse_condition(re.sub(rf"(?i)\b(?:weekly|monthly) (?={TF_INDICATORS})", "", text), ctx)
        return (_wrap_timeframe(e, per) if e else e), left
    s = " " + text.strip().lower() + " "
    s1 = re.sub(r"\s+", " ", s)
    if _indicator_periods(s1) != s1:
        s = _indicator_periods(s1)
    parts: list[str] = []
    level_parts: list[str] = []    # "is above 79" with no subject: the traded ticker's price

    def take(pattern: str, fn, level: bool = False) -> None:
        nonlocal s
        # a phrase must start and end on word boundaries ("down day" must not eat part of "down days")
        rx = re.compile(rf"(?<![a-z0-9])(?:{pattern})(?![a-z0-9])")
        pos = 0
        while True:
            m = rx.search(s, pos)
            if not m:
                return
            # a number followed by a time unit is a lookback, never a threshold: "rsi over 10 days" is not RSI > 10
            if re.search(r"\d$", m.group(0)) and re.match(rf" {UNIT_WORDS}(?![a-z])", s[m.end():]):
                pos = m.start() + 1
                continue
            parts.append(fn(m))
            if level:
                level_parts.append(parts[-1])
            s = re.sub(r" +", " ", s[: m.start()] + " " + s[m.end():])
            pos = 0

    c, o, h, l, v = ctx.c, ctx.o, ctx.h, ctx.l, ctx.v
    xs = "" if ctx.base else f", {c}"
    dn = r"(?:down|lower|declin\w*|fall\w*|fell|drop\w*|red|los\w*|negative|closes? down|closes? lower)"
    up = r"(?:up|higher|ris\w*|rose|gain\w*|green|advanc\w*|positive|closes? up|closes? higher)"
    down_x = "down_days" if ctx.base else f"down_streak({c})"
    up_x = "up_days" if ctx.base else f"up_streak({c})"
    chg = ctx.ret(1)

    # raw rule language in backticks (kept in its original case: True, sym("SPY"))
    raw_ticks = re.findall(r"`([^`]+)`", text.strip())

    def raw_rule(m):
        r = raw_ticks.pop(0).strip() if raw_ticks else m.group(1)
        from .expr import pine_to_rule   # TradingView spellings: close[1], ta.sma(...)
        try:
            out = pine_to_rule(r)
        except ValueError as e:
            raise ParseError(str(e)) from None
        if out != r:
            _note(f"TradingView syntax `{r}` was translated to {out}.")
        if m.group("tv") is not None:
            # "`ta.highest(high, 20)` crosses above 0", "`x` is below 10": the backticked value is the subject
            if re.search(r"[<>]|==|!=|\b(?:and|or|not|crossover|crossunder|cross)\b", out):
                raise ParseError(f"'`{r}` {m.group('tail')}': `{r}` is already a condition; put the whole comparison in the "
                                 f"backticks, e.g. `{r} and ...`.")
            v = "0" if m.group("tv") == "zero" else m.group("tv")
            if m.group("xd"):
                return f"{'crossover' if m.group('xd') in ('above', 'over') else 'crossunder'}({out}, {v})"
            return f"{out} {_cmp(m.group('cw'))} {v}"
        return out
    # (a value in backticks followed by a comparison with a number: the value is what is compared)
    take(rf"`([^`]+)`(?: (?P<tail>(?:(?:is |closes? |trades? )?cross(?:es|ed)?(?: back)? (?P<xd>above|over|below|under)"
         rf"|(?:is |stays? |trades? |closes? )?(?P<cw>at least|at most|no less than|no more than|below|under|less than|lower than|beneath|"
         rf"above|over|greater than|higher than|more than)) \$?(?P<tv>-?\d+(?:\.\d+)?|zero))(?![\d%])(?! (?:day|week|month|bar)))?",
         raw_rule)  # wrapped in () below

    # negations of relations: "not above 79" is "at most 79"
    s = _negations(s)
    # "below its 50 day moving average by more than 5%" = "more than 5% below its 50 day moving average"
    s = re.sub(r"(?<![a-z])(above|below|over|under) ((?:its |the )?(?:\d+ (?:day|week|month|bar|period|session)s? )?"
               r"(?:(?:simple |exponential |weighted )?(?:moving average|moving avg|ma|sma|ema|wma)(?: ?\( ?\d+ ?\))?|vwap|"
               r"(?:\d+ (?:day|week|month|year)s? )?(?:high|low|peak)s?)) by (at least |more than |over |greater than )?"
               r"(\d+(?:\.\d+)?)%", r"\3\4% \1 \2", s)
    _review_takes(take, c)
    s = re.sub(r"(?:(?<= )the |(?<= )its )?(?:close|closing price|price) (?:is |was )?(higher|greater|lower|less) than (?=(?:the )?"
               r"(?:highest |lowest )?(?:high|low|close)s? (?:of|in|over|during) (?:the )?(?:last|past|prior|previous) \d)",
               lambda m: "closes above " if m.group(1) in ("higher", "greater") else "closes below ", s)
    # "the close is higher than 5 days ago" / "closes above its close 5 days ago": close against ref(close, N)
    def vs_ago(m):
        op = ">" if m.group("rel") in ("higher", "greater", "above", "over") else "<"
        return f"{c} {op} ref({c}, {_period(m.group('n'), m.group('u'))})"
    take(r"(?:(?:the |its |today's )?(?:close|closing price|price) (?:is |was )?|(?:it )?closes? |(?:it )?closed |(?:it )?(?:is|trades?) )"
         r"(?P<rel>higher|greater|above|over|lower|less|below|under)(?: than)? (?:(?:it was|it did|the close|its close|the price|"
         r"its price|the closing price|its closing price) )?(?P<n>\d+) (?:trading )?(?P<u>day|bar|session|week|month)s? ago", vs_ago)
    # the Ichimoku cloud (as drawn on the current bar: senkou A and B computed 25 bars earlier)
    def cloud(m):
        if not ctx.base:
            _unsupported("the Ichimoku cloud of another ticker")
        rel, top, bot = m.group("rel"), "maximum(senkou_a(), senkou_b())", "minimum(senkou_a(), senkou_b())"
        _note("Ichimoku cloud: TradingView's defaults (9, 26, 52, displacement 26); the cloud on today's bar is senkou "
              "span A = (tenkan + kijun) / 2 and span B = the 52-bar high/low midpoint, both computed 25 bars earlier.")
        if rel in ("inside", "in", "within"):
            return f"{c} <= {top} and {c} >= {bot}"
        if m.group("cross"):
            return f"crossover({c}, {top})" if rel in ("above", "over") else f"crossunder({c}, {bot})"
        return f"{c} > {top}" if rel in ("above", "over") else f"{c} < {bot}"
    take(r"(?:(?:the )?(?:close|price|closing price) )?(?:closes? |is |trades? )?(?P<cross>cross(?:es|ed)? (?:back )?)?"
         r"(?P<rel>above|over|below|under|inside|within|in) (?:the )?(?:ichimoku )?(?:cloud|kumo)", cloud)
    # "above its 200-day" (no noun) = its 200-day simple moving average
    def bare_ma(m):
        _note(f"'{m.group(2).strip()}' with no indicator named was read as the {m.group(3)} {m.group(4)} simple moving average.")
        return f"{m.group(1)}{m.group(2)} moving average"
    s = re.sub(r"((?:above|below|over|under|>=?|<=?) )((?:its|the) (\d+) (day|week|month)s?)(?= *$)", bare_ma, s)
    # volume against its average (before the price-vs-average phrases)
    take(r"volume (?:is )?(above|below|over|under) (?:its |the )?(?:(\d+) day )?(?:average|avg|moving average)(?: volume)?",
         lambda m: f"{v} {_cmp(m.group(1))} sma({v}, {m.group(2) or 20})")
    # valuation: Shiller's CAPE (point in time, lagged data.CAPE_LAG_MONTHS months), its percentile, the earnings yield
    cape_w = r"(?:the )?(?:shiller(?:'s)? )?(?:cape|cyclically adjusted (?:p/?e|price[- ]to[- ]earnings)|p/?e ?10|shiller p/?e)(?: ratio)?"

    def cape_note():
        _note(f"CAPE = Shiller's cyclically adjusted P/E of the US market (monthly), used {data.CAPE_LAG_MONTHS} months "
              "after the month it describes so that its earnings were already reported: cape() in the rule language.")

    def cape_cmp(m):
        cape_note()
        return f"cape() {_cmp(m.group(1))} {float(m.group(2)):g}"
    take(rf"{cape_w} (?:is |was |stays? |remains? )?{CMPW} (\d+(?:\.\d+)?)(?!%)", cape_cmp)

    def cape_pct_cmp(m):
        cape_note()
        yrs = int(m.group(3)) if m.group(3) else 0
        _note("CAPE percentile: where today's CAPE ranks among " + (f"the last {yrs} years' values" if yrs else
                                                                     "all values since 1881") + " known at the time.")
        return f"cape_pct({yrs}) {_cmp(m.group(1))} {float(m.group(2)) / 100:g}"
    take(rf"{cape_w} (?:percentile|percentile rank|rank) (?:is |was )?{CMPW} (\d+(?:\.\d+)?)(?:%| percent|th percentile)?"
         r"(?: (?:over|of|in) (?:the )?(?:last |past )?(\d+) years?)?", cape_pct_cmp)

    def ey_cmp(m):
        cape_note()
        return f"earnings_yield() {_cmp(m.group(1))} treasury_10y()"
    take(rf"(?:the )?(?:cape |cyclically adjusted |shiller )?earnings yield (?:is )?{CMPW} (?:the )?(?:10[- ]year|ten[- ]year) "
         r"(?:treasury |bond )?yield", ey_cmp)

    # the yield curve (10-year minus 2-year Treasury yield, FRED): "the yield curve is inverted"
    curve_w = (r"(?:the )?(?:treasury )?(?:yield curve|10[- ]?year ?(?:-|minus) ?2[- ]?year(?: treasury)?(?: yield)? spread|"
               r"2s ?10s(?: spread)?|10s ?2s(?: spread)?|(?:10[- ]?2|2[- ]?10) spread)")

    def curve_note():
        _note("Yield curve = the 10-year minus the 2-year Treasury yield (FRED T10Y2Y, or DGS10 - DGS2): yield_curve() "
              "in the rule language; a day's value is published after that day's close.")

    def curve(m):
        curve_note()
        return "yield_curve() >= 0" if m.group(1) else "yield_curve() < 0"
    take(rf"{curve_w} (?:is |was |stays? |remains? )?(not |un)?inverted", curve)

    def curve_cmp(m):
        curve_note()
        return f"yield_curve() {_cmp(m.group(1))} {float(m.group(2)) / 100:g}"
    take(rf"{curve_w} (?:is |was )?{CMPW} (-?\d+(?:\.\d+)?)(?:%| percent| percentage points?)", curve_cmp)
    take(rf"{curve_w} (?:is )?(?:positive|upward sloping|normal)", lambda m: (curve_note(), "yield_curve() > 0")[1])

    # holidays, from the published NYSE schedule: "the day before Thanksgiving", "the first trading day after a holiday"
    hol_w = (r"(thanksgiving(?: day)?|christmas(?: day)?|new year'?s?(?: day)?|good friday|memorial day|labor day|"
             r"independence day|july 4(?:th)?|the fourth of july|juneteenth|presidents'? day|president's day|mlk day|"
             r"martin luther king(?: jr\.?)? day|a (?:market |stock market |trading )?holiday|(?:the )?next holiday)")

    def hol_before(m):
        name = m.group(1)
        if name.startswith(("a ", "the next", "next")):
            return "days_to_holiday() == 0"
        from .calendar import holiday_key
        return f'(days_to_holiday() == 0 and next_holiday_is("{holiday_key(name)}"))'

    def hol_after(m):
        name = m.group(1)
        if name.startswith(("a ", "the next", "next")):
            return "days_since_holiday() == 0"
        from .calendar import holiday_key
        return f'(days_since_holiday() == 0 and last_holiday_is("{holiday_key(name)}"))'
    take(rf"(?:it is |on |its )?(?:the )?(?:last )?(?:trading )?(?:day|session) (?:before|ahead of|preceding) {hol_w}", hol_before)
    take(rf"(?:it is |on |its )?(?:the )?(?:first )?(?:trading )?(?:day|session) (?:after|following) {hol_w}", hol_after)

    # cross-sectional rank within the universe: "in the bottom decile of 20 day return" -> xrank(ret(20)) <= 0.1
    frac = {"decile": 0.1, "quintile": 0.2, "quartile": 0.25, "tercile": 1 / 3, "third": 1 / 3, "half": 0.5}

    def xsec(m):
        if not ctx.base:
            _unsupported("a cross-sectional rank of another ticker")
        q = frac[m.group("q")] if m.group("q") else float(m.group("pct")) / 100
        n = _period(m.group("n") or "1", m.group("u") or "day")
        x = ctx.ret(n)
        _note(f"'{m.group(0).strip()}': this ticker's percentile of its {n}-bar return among the universe's members that "
              "day (xrank in the rule language; 1 = the highest).")
        return f"xrank({x}) <= {q:.6g}" if m.group("side") in ("bottom", "lowest") else f"xrank({x}) > {1 - q:.6g}"
    take(r"(?:it is |its |is )?(?:in |among )?the (?P<side>bottom|top|lowest|highest) (?:(?P<q>decile|quintile|quartile|tercile|third|half)|"
         r"(?P<pct>\d+(?:\.\d+)?)%) (?:of (?:the |its )?(?:universe|index|stocks|members|tickers) )?(?:of|by|for|on|in) "
         r"(?:its |the |their )?(?:(?P<n>\d+)[- ](?P<u>day|week|month|year|bar|session)s? )?(?:price )?(?:return|performance|change|momentum)",
         xsec)

    # "not on Fridays" / "except in October"
    take(r"(?:but )?(?:not|except|excluding)(?: on)? (monday|tuesday|wednesday|thursday|friday)s?",
         lambda m: f"dow != {DOW[m.group(1)]}")
    take(r"(?:but )?(?:not|except|excluding) (?:in |during )?(?:the month of )?(january|february|march|april|may|june|july|august|september|october|november|december)",
         lambda m: f"month != {MONTHS.index(m.group(1)) + 1}")

    # "it closed down on Friday" / "closed up yesterday": the previous bar's close-to-close change (on that weekday)
    def closed_on(m):
        op = "<" if m.group(1) in ("down", "lower") else ">"
        d = m.group(2)
        prev = f"ref({chg}, 1) {op} 0"
        if d in DOW:
            _note(f"'{m.group(0).strip()}' = the previous trading day was a {d.capitalize()} and closed "
                  f"{'below' if op == '<' else 'above'} the close before it.")
            return f"ref(dow, 1) == {DOW[d]} and {prev}"
        return prev
    take(r"(?:it |the (?:stock|price|market) )?(?:closed|finished|ended)(?: the day)? (down|lower|up|higher) (?:on |last )?"
         r"(monday|tuesday|wednesday|thursday|friday|yesterday|the (?:previous|prior) (?:day|session))", closed_on)
    # "pulls back to its 50 day moving average": today's low reaches the average from above
    def pullback(m):
        ma = _mat_expr(m, "pb", c)
        _note(f"'{m.group(0).strip()}' = today's low reaches the average ({l} <= {ma}) after closing above it the day "
              f"before (ref({c}, 1) > ref({ma}, 1)).")
        return f"{l} <= {ma} and ref({c}, 1) > ref({ma}, 1)"
    take(r"(?:it |the price )?(?:pulls?|pulled|pulling|dips?|dipped) back (?:down )?to (?:its |the )?" + _mat("pb"), pullback)
    take(r"(?:it |the price )?(?:retraces?|retraced|dips?|dipped) to (?:its |the )?" + _mat("pb"), pullback)
    # "3 standard deviations below its 20 day mean": the z-score of the close
    def sigmas(m):
        k, rel = float(m.group(1)), m.group(2)
        n = _period(m.group(3), m.group(4)) if m.group(3) else 20
        if not m.group(3):
            _note(f"'{m.group(0).strip()}': no lookback given, using a 20-day mean and standard deviation.")
        low_side = rel in ("below", "under")
        return f"zscore({c}, {n}) {'<=' if low_side else '>='} {-k if low_side else k:g}"
    take(rf"(?:is |closes? |trades? |falls? |drops? |rises? )?(?:at least |more than |over )?{NUM} (?:standard deviations?|std devs?|stdevs?|sigmas?|sd) "
         r"(below|under|above|over) (?:its |the )?(?:(\d+)[- ](day|week|month|bar|session)s? )?mean\b", sigmas)
    # "14 day momentum is above 0": TradingView's momentum, the price change over N bars (close - close N bars ago)
    if not ctx.total:
        def mom(m):
            n = _period(m.group(1), m.group(2))
            _note(f"'{m.group(0).strip()}': momentum = the price change over {n} days ({c} - its value {n} days ago, "
                  "TradingView's ta.mom); say 'the {n} day return' for a percentage.".replace("{n}", str(n)))
            return f"diff({c}, {n}) {_cmp(m.group(3))} {m.group(4)}"
        take(rf"(?:its |the )?(\d+)[- ](day|bar|period|session)s? momentum (?:is )?{CMPW} (-?\d+(?:\.\d+)?)(?![\d.%])(?! (?:day|week|month|bar))", mom)
    _r13_candle_takes(take, c, o)

    # consecutive down / up closes ("exactly N" fires only on the Nth day)
    take(rf"{dn} (?:for )?exactly {NUM} (?:straight |consecutive )?(?:days|closes|sessions|bars)(?: in a row| straight)?|exactly {NUM} (?:consecutive|straight) {dn} (?:days|closes|sessions|bars)",
         lambda m: f"{down_x} == {int(float(m.group(1) or m.group(2)))}")
    take(rf"{up} (?:for )?exactly {NUM} (?:straight |consecutive )?(?:days|closes|sessions|bars)(?: in a row| straight)?|exactly {NUM} (?:consecutive|straight) {up} (?:days|closes|sessions|bars)",
         lambda m: f"{up_x} == {int(float(m.group(1) or m.group(2)))}")
    take(rf"(?:at least )?{NUM} (?:or more )?(?:consecutive|straight|successive) {dn} (?:days|closes|sessions|bars)",
         lambda m: f"{down_x} >= {int(float(m.group(1)))}")
    take(rf"(?:at least )?{NUM} (?:or more )?(?:consecutive|straight|successive) {up} (?:days|closes|sessions|bars)",
         lambda m: f"{up_x} >= {int(float(m.group(1)))}")
    take(rf"(?:at least )?{NUM} (?:or more )?(?:down|red|losing|negative) (?:days|closes|sessions|bars)(?: in a row| straight| consecutively)?",
         lambda m: f"{down_x} >= {int(float(m.group(1)))}")
    take(rf"(?:at least )?{NUM} (?:or more )?(?:up|green|winning|positive) (?:days|closes|sessions|bars)(?: in a row| straight| consecutively)?",
         lambda m: f"{up_x} >= {int(float(m.group(1)))}")
    take(rf"{dn}(?: for)?(?: at least)? {NUM} (?:or more )?(?:straight |consecutive |trading )?(?:days|closes|sessions|bars|times)(?: in a row| straight| consecutively)",
         lambda m: f"{down_x} >= {int(float(m.group(1)))}")
    take(rf"{up}(?: for)?(?: at least)? {NUM} (?:or more )?(?:straight |consecutive |trading )?(?:days|closes|sessions|bars|times)(?: in a row| straight| consecutively)",
         lambda m: f"{up_x} >= {int(float(m.group(1)))}")

    # the ATR against its own average: "the 14 day ATR is above its 50 day average", "ATR(14) crosses above its 50 day SMA"
    def atr_ma(m):
        if not ctx.base:
            _unsupported("the ATR of another ticker")
        n = int(m.group("n") or m.group("n2") or 14)
        k = m.group("k") or ""
        a = f"atr({n})"
        ma = f"{'ema' if ('ema' in k or 'exp' in k) else 'sma'}({a}, {m.group('m')})"
        if m.group("cross"):
            return f"{'crossover' if _cmp(m.group('rel')) == '>' else 'crossunder'}({a}, {ma})"
        return f"{a} {_cmp(m.group('rel'))} {ma}"
    take(rf"(?:the |its )?(?:(?P<n>\d+)[- ](?:day|period|bar) |(?<![\d.] )(?<![\d.] x )(?<![\d.]x ))(?:atr|average true range)(?:\s*\(\s*(?P<n2>\d+)\s*\))? (?:is |stays |has )?"
         rf"(?P<cross>cross(?:es|ed)? )?(?P<rel>above|over|below|under) (?:its |the )?(?P<m>\d+)[- ](?:day |period |bar )?"
         rf"(?P<k>{MAK}|average|avg)", atr_ma)

    # MACD; "MACD(12,26,9) crosses above signal" gives its own periods
    def macd_sig(m):
        if not ctx.base:
            _unsupported("MACD of another ticker")
        rel = m.group("rel")
        f_, s_, g_ = m.group("mf"), m.group("ms"), m.group("mg")
        line = f"macd({f_}, {s_})" if f_ else "macd()"
        sig = f"macd_signal({f_}, {s_}, {g_ or 9})" if f_ else "macd_signal()"
        if "cross" in rel:
            return f"{'crossover' if ('above' in rel or 'over' in rel) else 'crossunder'}({line}, {sig})"
        return f"{line} {'>' if 'above' in rel else '<'} {sig}"
    take(r"macd(?:\s*\(\s*(?P<mf>\d+)\s*,\s*(?P<ms>\d+)\s*(?:,\s*(?P<mg>\d+)\s*)?\))?(?: line)? "
         r"(?P<rel>cross(?:es|ed)? (?:above|over)|cross(?:es|ed)? (?:below|under)|is above|is below|above|below) (?:its |the )?(?:macd )?signal(?: line)?",
         macd_sig)
    take(r"macd(?: histogram| line)? (turns positive|turns negative|crosses above zero|crosses below zero|crosses above 0|crosses below 0|is positive|is negative|is above zero|is below zero|is above 0|is below 0)",
         lambda m: ((f"crossover(macd_hist(), 0)" if "histogram" in m.group(0) else "crossover(macd(), 0)") if ("turns positive" in m.group(1) or "crosses above" in m.group(1))
                    else (f"crossunder(macd_hist(), 0)" if "histogram" in m.group(0) else "crossunder(macd(), 0)") if ("turns negative" in m.group(1) or "crosses below" in m.group(1))
                    else f"{'macd_hist()' if 'histogram' in m.group(0) else 'macd()'} {'>' if ('positive' in m.group(1) or 'above' in m.group(1)) else '<'} 0"))
    # directional movement
    def di(m):
        a, b = ("plus_di()", "minus_di()") if m.group("a").lstrip("the ").startswith(("+", "plus")) else ("minus_di()", "plus_di()")
        rel = m.group("rel")
        if m.group("cross"):
            return f"{'crossover' if rel in ('above', 'over') else 'crossunder'}({a}, {b})"
        return f"{a} {_cmp(rel)} {b}"
    take(r"(?P<a>(?:the )?(?:\+di|plus di|-di|minus di)) (?:line )?(?:is )?(?P<cross>cross(?:es|ed)? )?(?P<rel>above|over|below|under) "
         r"(?:the )?(?P<b>\+di|plus di|-di|minus di)(?: line)?", di)
    # stochastic %K vs %D
    def kd(m):
        if not ctx.base:
            _unsupported("the stochastic of another ticker")
        slow = bool(re.search(r"\b(?:slow|full)\b", m.group(0)))
        sm = 3 if slow else 1
        k, d = f"stoch_k(14, {sm})", f"stoch_d(14, {sm}, 3)"
        _stoch_note(slow)
        rel = m.group("rel")
        zone = f" and ({k} {_cmp(m.group('zrel'))} {m.group('zv')})" if m.group("zrel") else ""
        if m.group("cross"):
            return f"{'crossover' if rel in ('above', 'over') else 'crossunder'}({k}, {d}){zone}"
        return f"{k} {_cmp(rel)} {d}{zone}"
    # Stochastic RSI (TradingView's built-in: %K smoothing 3, %D 3, RSI length 14, stochastic length 14)
    SRSI = r"(?:the )?(?:stoch(?:astic)?[ -]?rsi|stochrsi)(?:\s*\(\s*(?P<sk>\d+)\s*,\s*(?P<sd>\d+)\s*,\s*(?P<rn>\d+)\s*,\s*(?P<sn>\d+)\s*\))?"

    def srsi_calls(m):
        if not ctx.base:
            _unsupported("the stochastic RSI of another ticker")
        sk, sd, rn, sn = (m.group("sk") or "3", m.group("sd") or "3", m.group("rn") or "14", m.group("sn") or "14")
        _note(f"Stochastic RSI: TradingView's built-in - %K = the {sk}-bar SMA of ta.stoch(rsi, rsi, rsi, {sn}) with "
              f"rsi = RSI({rn}) (stoch_rsi_k({sk}, {rn}, {sn})), %D = the {sd}-bar SMA of %K "
              f"(stoch_rsi_d({sk}, {sd}, {rn}, {sn})).")
        return f"stoch_rsi_k({sk}, {rn}, {sn})", f"stoch_rsi_d({sk}, {sd}, {rn}, {sn})"

    def srsi_kd(m):
        k, d = srsi_calls(m)
        rel = m.group("rel")
        if m.group("cross"):
            return f"{'crossover' if rel in ('above', 'over') else 'crossunder'}({k}, {d})"
        return f"{k} {_cmp(rel)} {d}"
    take(rf"{SRSI}(?:'s)? %?k(?: line)? (?:is )?(?:(?P<cross>cross(?:es|ed)?(?: back)?) )?(?P<rel>above|over|below|under) "
         rf"(?:the |its )?(?:stoch(?:astic)?[ -]?rsi(?:'s)? |stochrsi(?:'s)? )?%?d(?: line)?", srsi_kd)

    def srsi_lvl(m):
        k, d = srsi_calls(m)
        e = d if m.group("which") and m.group("which").strip().lstrip("%") == "d" else k
        op = _cmp(m.group("rel"))
        if m.group("cross"):
            return f"{'crossover' if op[0] == '>' else 'crossunder'}({e}, {m.group('v')})"
        return f"{e} {op} {m.group('v')}"
    take(rf"{SRSI}(?:'s)?(?P<which> %?k| %?d)?(?: line| value| reading)? (?:is |closes |drops |falls |rises |goes |reads )?"
         rf"(?:(?P<cross>cross(?:es|ed)?(?: back)?) )?(?P<rel>{CMPW[1:-1]}) (?P<v>-?{NUM[1:-1]})(?!%)", srsi_lvl)

    KP = r"(?:the )?(?:(?:slow |full )?stoch(?:astic)?(?:'s)? )?%k(?: line)?"
    DP = r"(?:the |its )?(?:(?:slow |full )?stoch(?:astic)?(?:'s)? )?%d(?: line)?"
    take(rf"{KP} (?:is )?(?:(?P<cross>cross(?:es|ed)?(?: back)?) )?(?P<rel>above|over|below|under|<=|>=) {DP}"
         rf"(?: (?:while |when |with )?(?:(?:%k|it|both) (?:is |are )?)?(?:in the oversold zone |in the overbought zone )?(?P<zrel>below|under|above|over) (?P<zv>{NUM[1:-1]}))?", kd)
    if re.search(rf"{KP} cross(?:es|ed)? {DP}", s):
        raise ParseError("'%K crosses %D': which way? Say '%K crosses above %D' or '%K crosses below %D'.")

    # "%K is below 20", "%D crosses above 80": the stochastic lines against a level (TradingView's 14, 1, 3)
    def kd_lvl(m):
        if not ctx.base:
            _unsupported("the stochastic of another ticker")
        slow = bool(re.search(r"\b(?:slow|full)\b", m.group(0)))
        sm = 3 if slow else 1
        _stoch_note(slow)
        e = f"stoch_d(14, {sm}, 3)" if m.group("which") == "d" else f"stoch_k(14, {sm})"
        op = _cmp(m.group("rel"))
        if m.group("cross"):
            return f"{'crossover' if op[0] == '>' else 'crossunder'}({e}, {m.group('v')})"
        return f"{e} {op} {m.group('v')}"
    take(rf"(?:the )?(?:(?:slow |full )?stoch(?:astic)?(?:'s)? )?%(?P<which>k|d)(?: line| value| reading)? (?:is )?"
         rf"(?:(?P<cross>cross(?:es|ed)?(?: back)?) )?(?P<rel>{CMPW[1:-1]}) (?P<v>-?{NUM[1:-1]})(?!%)", kd_lvl)

    # Heikin Ashi candles (TradingView's): green = ha_close > ha_open; "turns green" = green today, not yesterday
    def ha_colour(m):
        if not ctx.base:
            _unsupported("Heikin Ashi candles of another ticker")
        green = m.group("col") in ("green", "bullish", "up", "white")
        op, nop = (">", "<=") if green else ("<", ">=")
        if m.group("turn"):
            return f"ha_close {op} ha_open and ref(ha_close, 1) {nop} ref(ha_open, 1)"
        return f"ha_close {op} ha_open"
    HA = r"(?:the )?(?:heikin[- ]?ashi|ha)(?: candles?| candlesticks?| bars?)?"
    take(rf"{HA} (?:(?P<turn>turns?|turned|flips?|flipped|changes?(?: to)?|changed(?: to)?|switches(?: to)?)|is|are|closes?|was) "
         rf"(?P<col>green|red|bullish|bearish|white|black)", ha_colour)

    # higher-timeframe RSI, computed on completed weekly / monthly bars
    def htf_rsi(m):
        n = int(m.group("n") or m.group("n2") or m.group("n3") or 14)
        e = f"{m.group('per')}_rsi({n}{xs})"
        if m.group("cross"):
            return f"{'crossover' if _cmp(m.group('rel'))[0] == '>' else 'crossunder'}({e}, {m.group('v')})"
        return f"{e} {_cmp(m.group('rel'))} {m.group('v')}"
    take(rf"(?:the |its )?(?P<per>weekly|monthly) (?:(?P<n>\d+) (?:period |bar |week |month )?)?rsi\s*(?:\(\s*(?P<n2>\d+)\s*\)|(?P<n3>\d+))?(?: value| reading)? (?:is )?"
         rf"(?:(?P<cross>cross(?:es|ed)?(?: back)?|falls?(?: back)?|fell(?: back)?|drops?(?: back)?|dropped(?: back)?|"
         rf"rises?(?: back)?|rose(?: back)?|climbs?(?: back)?|climbed(?: back)?|moves?(?: back)?|moved(?: back)?) )?"
         rf"(?P<rel>{CMPW[1:-1]}) (?P<v>-?{NUM[1:-1]})", htf_rsi)
    mw = re.search(r"(?<![a-z0-9])(\d+) (week|month)s? rsi\b", s)
    if mw:
        raise ParseError(f"'{mw.group(0)}' is ambiguous: say 'the {mw.group(2)}ly RSI({mw.group(1)})' (RSI of {mw.group(2)}ly closes) "
                         f"or 'RSI({_period(mw.group(1), mw.group(2))})' (on daily bars).")
    # "the monthly 10 SMA" / "weekly 20 EMA" -> "10 month SMA" / "20 week EMA"
    s = re.sub(r"\b(weekly|monthly) (\d+)(?: (?:period|bar))? ((?:simple |exponential )?(?:moving average|sma|ema|ma)\b)",
               lambda m: f"{m.group(2)} {'week' if m.group(1) == 'weekly' else 'month'} {m.group(3)}", s)

    def per_close(m):
        pc = f"{m.group('per')}_close({'' if ctx.base else c})"
        ma = _mat_expr(m, "a", c)
        if m.group("cross"):
            return f"{'crossover' if _cmp(m.group('rel'))[0] == '>' else 'crossunder'}({pc}, {ma})"
        return f"{pc} {_cmp(m.group('rel'))} {ma}"
    take(rf"(?:the |its )?(?P<per>weekly|monthly) clos(?:e|ing price) (?:is |closes? |stays? )?(?P<cross>cross(?:es|ed)? (?:back )?)?"
         rf"(?P<rel>above|over|below|under|>=?|<=?) (?:the |its )?{_mat('a')}", per_close)

    # oscillator crossings and levels
    take(rf"(?:the )?(?:(\d+) (?:day|period|bar)s? )?{OSC}\s*(?:\(\s*(\d+)\s*\)|(\d+))? (?:line )?(?:cross(?:es|ed)?(?: back)?|(?:falls?|fell|drops?|dropped|dips?|dipped|moves?|moved|goes|went|gets?|rises?|rose|climbs?|climbed|comes?|came) back) (above|over|below|under) (-?{NUM})",
         lambda m: f"{'crossover' if m.group(5) in ('above', 'over') else 'crossunder'}({_osc(m.group(2), m.group(1) or m.group(3) or m.group(4), ctx)}, {m.group(6)})")
    take(rf"(?:the )?(?:(\d+) (?:day|period|bar)s? )?{OSC}\s*(?:\(\s*(\d+)\s*\)|(\d+))?(?: value| reading)? (?:is |closes |drops |falls |rises |goes |reads )?{CMPW} (-?{NUM})",
         lambda m: f"{_osc(m.group(2), m.group(1) or m.group(3) or m.group(4), ctx)} {_cmp(m.group(5))} {m.group(6)}")

    # moving-average relationships between two averages
    def ma_vs_ma(m):
        a, b = _mat_expr(m, "a", c), _mat_expr(m, "b", c)
        rel = (m.group("cross") or "") + m.group("rel")
        if "cross" in rel:
            return f"{'crossover' if 'above' in rel or 'over' in rel else 'crossunder'}({a}, {b})"
        return f"{a} {_cmp(m.group('rel'))} {b}"
    take(rf"(?:the |its )?{_mat('a')} (?:line )?(?:is |has |stays |remains )?(?P<cross>cross(?:es|ed)? )?(?P<rel>above|over|below|under|<=|>=|>|<) (?:the |its )?{_mat('b')}", ma_vs_ma)
    take(r"golden cross", lambda m: f"crossover(sma({c}, 50), sma({c}, 200))")
    take(r"death cross", lambda m: f"crossunder(sma({c}, 50), sma({c}, 200))")

    # candlestick engulfing patterns on daily candles: the body engulfs the previous candle's opposite-colour body
    def engulfing(m):
        if not ctx.base:
            _unsupported("an engulfing pattern of another ticker")
        _note("Engulfing: the standard two-candle pattern on daily bars - a bullish engulfing is a down candle (close < "
              "open) followed by an up candle whose body covers it (open <= the previous close, close >= the previous "
              "open, a larger body); bearish is the mirror image.")
        if m.group("dir") == "bullish":
            return (f"ref({c}, 1) < ref({o}, 1) and {c} > {o} and {o} <= ref({c}, 1) and {c} >= ref({o}, 1) "
                    f"and {c} - {o} > ref({o}, 1) - ref({c}, 1)")
        return (f"ref({c}, 1) > ref({o}, 1) and {c} < {o} and {o} >= ref({c}, 1) and {c} <= ref({o}, 1) "
                f"and {o} - {c} > ref({c}, 1) - ref({o}, 1)")
    take(r"(?:(?:on|after|with|there is|there's) )?(?:an? |the )?(?P<dir>bullish|bearish) engulfing(?: (?:candle|candlestick|pattern|bar|day))?",
         engulfing)

    # Donchian breakout (Turtle rules): the close above the highest high of the previous N bars
    def donchian_bo(m):
        n = int(m.group("n"))
        if not ctx.base:
            _unsupported("a Donchian breakout of another ticker")
        up = m.group("k") in ("breakout", "break out", "high", "upside breakout")
        return f"{c} > ref(highest({h}, {n}), 1)" if up else f"{c} < ref(lowest({l}, {n}), 1)"
    take(r"(?:(?:on|after) )?(?:an? |the )?donchian(?: channel)? (?P<n>\d+)[- ]?(?:day|bar|period|session)?s? (?:channel )?"
         r"(?P<k>breakout|break out|breakdown|break down|upside breakout|downside breakout|high|low)", donchian_bo)

    # the Ichimoku conversion line against the base line
    def tk_cross(m):
        if not ctx.base:
            _unsupported("the Ichimoku lines of another ticker")
        up = m.group("rel") in ("above", "over")
        if m.group("cross"):
            return f"{'crossover' if up else 'crossunder'}(tenkan(), kijun())"
        return f"tenkan() {'>' if up else '<'} kijun()"
    take(r"(?:the )?(?:tenkan(?:[- ]sen)?|conversion line) (?:is |line )?(?P<cross>cross(?:es|ed)? )?(?P<rel>above|over|below|under) "
         r"(?:the )?(?:kijun(?:[- ]sen)?|base line)", tk_cross)

    # the slope of a moving average: today's value against yesterday's
    def ma_slope(m):
        n = int(m.group("n"))
        k = m.group("k")
        f = "ema" if k in ("ema", "exponential moving average") else "sma"
        e = f"{f}({c}, {n})"
        pos = m.group("s") in ("positive", "rising", "up", "upward", "above zero")
        return f"{e} > ref({e}, 1)" if pos else f"{e} < ref({e}, 1)"
    take(r"(?:the |its )?(?:slope of (?:the |its )?(?P<n0>\d+)[- ]?(?:day |period |bar )?(?P<k0>ema|sma|ma|moving average|exponential moving average)"
         r"|(?P<n>\d+)[- ]?(?:day |period |bar )?(?P<k>ema|sma|ma|moving average|exponential moving average)(?:'s)? slope) "
         r"(?:is |turns |stays )?(?P<s>positive|negative|rising|falling|up|down|upward|downward|above zero|below zero)",
         lambda m: ma_slope(_SlopeMatch(m)))

    # on-balance volume against its own average
    def obv_x(m):
        if not ctx.base:
            _unsupported("OBV of another ticker")
        ma = f"{'ema' if 'ema' in m.group('k') or 'exp' in m.group('k') else 'sma'}(obv(), {m.group('n')})"
        if m.group("cross"):
            return f"{'crossover' if _cmp(m.group('rel')) == '>' else 'crossunder'}(obv(), {ma})"
        return f"obv() {_cmp(m.group('rel'))} {ma}"
    take(rf"(?:the )?(?:obv|on[- ]balance volume)(?: line)? (?:is )?(?P<cross>cross(?:es|ed)?(?: back)? )?(?P<rel>above|over|below|under) "
         rf"(?:its |the )?(?P<n>\d+) (?:day |period |bar )?(?P<k>{MAK})", obv_x)

    # Bollinger %B: (close - lower band) / (upper band - lower band)
    def pctb(m):
        if not ctx.base:
            _unsupported("Bollinger %B of another ticker")
        b = "((close - bb_lower(20, 2)) / (bb_upper(20, 2) - bb_lower(20, 2)))"
        if m.group("cross"):
            return f"{'crossover' if _cmp(m.group('rel')) == '>' else 'crossunder'}({b}, {m.group('v')})"
        return f"{b} {_cmp(m.group('rel'))} {m.group('v')}"
    take(rf"(?:the )?(?:bollinger |bb )?%b(?: value| reading)? (?:is )?(?P<cross>cross(?:es|ed)?(?: back)? )?(?P<rel>{CMPW[1:-1]}) (?P<v>-?{NUM[1:-1]})(?!%)", pctb)

    # "the close is 2 ATR below the 20 day EMA"
    def atr_off(m):
        if not ctx.base:
            _unsupported("ATR of another ticker")
        k, ma, below = float(m.group("k")), _mat_expr(m, "a", c), _cmp(m.group("rel")) == "<"
        return f"{c} {_bound('<' if below else '>', m.group(0), True)} {ma} {'-' if below else '+'} {k:g} * atr(14)"
    take(rf"(?:closes? |is |trades? |falls |drops |rises |moves )?(?:at least |more than )?(?P<k>{NUM[1:-1]}) ?(?:x )?atrs? (?P<rel>below|under|above|over) (?:the |its )?{_mat('a')}", atr_off)

    # "closes 2 standard deviations below the 20 day average": a Bollinger-style band on the same lookback
    def sd_off(m):
        unit = (m.group("au") or "day").lower()
        if unit.startswith(("week", "month")):
            raise ParseError(f"'{m.group(0).strip()}': standard-deviation bands on weekly or monthly averages are not supported; "
                             "use a daily lookback or backticks.")
        n = _period(m.group("an") or m.group("an2"), unit)
        k, ma, below = float(m.group("k")), _mat_expr(m, "a", c), _cmp(m.group("rel")) == "<"
        return f"{c} {_bound('<' if below else '>', m.group(0), True)} {ma} {'-' if below else '+'} {k:g} * stdev({c}, {n})"
    take(rf"(?:closes? |is |trades? |falls |drops |rises |moves )?(?:at least |more than )?(?P<k>{NUM[1:-1]}) (?:standard deviations?|std devs?|stdevs?|"
         rf"sigmas?|sds?) (?P<rel>below|under|above|over) (?:the |its )?{_mat('a')}", sd_off)

    # price vs moving average (or VWAP), optionally by X%
    def px_vs(m, ma):
        pct, cross, rel = m.group("pct"), m.group("cross"), m.group("rel")
        op = _cmp(rel)
        if cross:
            return f"{'crossover' if op[0] == '>' else 'crossunder'}({c}, {ma})"
        if pct:
            f = float(pct) / 100
            return f"{c} {_bound(op, m.group(0))} {ma} * {1 - f if op[0] == '<' else 1 + f:.6g}"
        return f"{c} {op} {ma}"
    PX = (rf"(?:(?:closes?|trades?|is|stays?|remains?) )?(?:(?:(?:by )?(?:at least|more than|over) )?(?P<pct>\d+(?:\.\d+)?)%(?: or more)? )?(?:(?:closes?|trades?|is|price is|price|stays?|remains?|falls|drops|moves|goes|rises|climbs) )?"
          rf"(?:back )?(?P<cross>cross(?:es|ed)? (?:back )?)?(?P<rel>{CMPW[1:-1]}) (?:its |the )?")
    take(PX + _mat("a"), lambda m: px_vs(m, _mat_expr(m, "a", c)))

    # VWAP: on daily bars, the rolling N-day volume-weighted average of the typical price (default 20)
    def vwap_x(m):
        n = int(m.group("vn") or 20)
        return px_vs(m, f"vwap({n})" if ctx.base else f"(sma((({h} + {l} + {c}) / 3) * {v}, {n}) / sma({v}, {n}))")
    take(PX + r"(?:(?P<vn>\d+) (?:day|bar|period|session)s? )?(?:rolling )?vwap", vwap_x)

    # distance from an N-day / 52-week / all-time high or low: "is down 10% from its 200 day high",
    # "has fallen 20% below its 52 week high", "rallied 10% off its 20 day low", "within 2% of its high"
    def extreme(m):
        g = m.groupdict()
        n, unit, hl = g["n"], g["unit"], g["hl"]
        high = hl in ("high", "peak", "top")
        look = (252 if (n == "52" and unit.startswith("week")) else _period(n, unit)) if n else None   # None: all history
        pct = float(g["pct"]) / 100
        verb = "down" if g.get("dn") else "up" if g.get("up") else None
        prep = g.get("prep")
        what = g["span"].strip()
        low_base = f"lowest({c}, {look})" if look else f"cummin({c})"
        if g.get("within"):
            # "within 10%" / "at most 10% below" include the bound; "less than 10% below" does not
            st = g["within"] in ("less than",)
            if high:
                return f"drawdown({c}{', ' + str(look) if look else ''}) {'>' if st else '>='} {-pct:g}"
            return f"{c} / {low_base} - 1 {'<' if st else '<='} {pct:g}"
        if high and not look and not g.get("within"):
            _note(f"'{m.group(0).strip()}' has no period: measured from the highest close so far (the running peak of all "
                  f"the data, drawdown({c})); say e.g. 'from its 52 week high' for a rolling window.")
        if high:
            if verb == "up" or prep in ("above", "over"):
                raise ParseError(f"'{m.group(0).strip()}': the price cannot be above {what} (the high includes today). "
                                 f"Did you mean below it, e.g. 'is down {g['pct']}% from {what}'?")
            return f"drawdown({c}{', ' + str(look) if look else ''}) {_bound('<', m.group(0))} {-pct:g}"
        if verb == "down" or prep in ("below", "under"):
            raise ParseError(f"'{m.group(0).strip()}': the price cannot be below {what} (the low includes today). "
                             f"Did you mean above it, e.g. 'is up {g['pct']}% from {what}'?")
        return f"{c} / {low_base} - 1 {_bound('>', m.group(0))} {pct:g}"
    EXT_SPAN = (r"(?P<span>(?:its |the |their )?(?:(?P<n>\d+) (?P<unit>day|week|month|year|bar|session)s? |(?:all[- ]time |record |lifetime ))?"
                r"(?P<hl>high|peak|top|low|bottom|trough)s?)(?! (?:of|in) the day)")
    take(rf"(?:is |trades? |closes? |sits? |stays? |remains? )?(?P<within>within) (?P<pct>{NUM[1:-1]})% (?:of|from) {EXT_SPAN}", extreme)
    # "less than 10% below its 52 week high" = within 10% of it
    take(rf"(?:is |trades? |closes? |sits? |stays? |remains? )?(?P<within>less than|no more than|not more than|at most) (?P<pct>{NUM[1:-1]})% "
         rf"(?:below|under|from|off|above|over) {EXT_SPAN}", extreme)
    take(rf"{MOVE_AUX}(?:(?P<dn>{DOWN_VERBS})|(?P<up>{UP_VERBS}|bounced|rebounded|recovered)|off|sits?|stays?|remains?)?(?: by)?"
         rf"(?: at least| more than| over| greater than)? (?P<pct>{NUM[1:-1]})%(?: or more)? (?P<prep>from|off|below|under|above|over) {EXT_SPAN}", extreme)
    take(rf"(?:(?:closes?|closed|trades?|traded|is|was|sits?|stays?|remains?) )?(?:at least |more than |over |greater than )?"
         rf"(?P<pct>{NUM[1:-1]})% (?:or more )?(?P<prep>from|off|below|under|above|over) {EXT_SPAN}", extreme)

    # breakouts vs prior N-day range
    take(r"(?:closes? |breaks? |trades? |moves? )?(above|over|below|under) (?:its |the )?(?:previous |prior |last )?(\d+) (day|week|month|bar)s? (high|low)",
         lambda m: (f"{c} > ref(highest({h}, {_period(m.group(2), m.group(3))}), 1)" if m.group(4) == "high" and _cmp(m.group(1)) == ">"
                    else f"{c} < ref(lowest({l}, {_period(m.group(2), m.group(3))}), 1)" if m.group(4) == "low" and _cmp(m.group(1)) == "<"
                    else f"{c} {_cmp(m.group(1))} ref({'highest(' + h if m.group(4) == 'high' else 'lowest(' + l}, {_period(m.group(2), m.group(3))}), 1)"))
    take(r"(?:closes? |breaks? |trades? |moves? |crosses |is |goes )?(above|over|below|under) (?:the )?(highest high|lowest low|highest close|lowest close|"
         r"high(?:est price)?|low(?:est price)?) (?:of|in|over|during) (?:the )?(?:last|past|prior|previous) (\d+) (day|week|month|bar|session)s?",
         lambda m: f"{c} {_cmp(m.group(1))} ref({'highest' if m.group(2).startswith('high') else 'lowest'}("
                   f"{c if m.group(2).endswith('close') else (h if m.group(2).startswith('high') else l)}, {_period(m.group(3), m.group(4))}), 1)")
    take(r"breaks? out(?: to a new)? (\d+) (day|week|month)s? high", lambda m: f"{c} > ref(highest({h}, {_period(m.group(1), m.group(2))}), 1)")
    take(r"breaks? down(?: to a new)? (\d+) (day|week|month)s? low", lambda m: f"{c} < ref(lowest({l}, {_period(m.group(1), m.group(2))}), 1)")

    # N-day closing highs / lows
    def hilo(m):
        n = 252 if m.group(1) == "52" and m.group(2).startswith("week") else _period(m.group(1), m.group(2))
        return f"{c} >= highest({c}, {n})" if m.group(3) == "high" else f"{c} <= lowest({c}, {n})"
    take(rf"(?:makes? |hits? |sets? |closes? at |at |reaches )?(?:a )?(?:new |fresh )?(\d+) (day|week|month|year|bar|session)s? (high|low)(?:est close)?(?: close)?", hilo)
    take(r"(?:lowest|highest) close (?:in|of) (?:the (?:last|past) )?(\d+) (day|week|month|year|bar|session)s?",
         lambda m: (f"{c} <= lowest({c}, {_period(m.group(1), m.group(2))})" if "lowest" in m.group(0)
                    else f"{c} >= highest({c}, {_period(m.group(1), m.group(2))})"))
    take(r"(?:makes? |hits? |closes? at |at )?(?:a )?(?:new )?all[- ]time high", lambda m: f"{c} >= cummax({c})")

    # supertrend / SAR / keltner / donchian
    def st_args(m):
        if not ctx.base:
            _unsupported("the supertrend of another ticker")
        return f"supertrend({m.group('n') or 10}, {m.group('k') or 3})"
    ST = r"(?:the )?supertrend(?:\s*\(\s*(?P<n>\d+)\s*,\s*(?P<k>\d+(?:\.\d+)?)\s*\))?"
    take(rf"(?:(?:the )?(?:close|price|closing price) )?(?:closes? |is |trades? )?(?P<cross>cross(?:es|ed)? (?:back )?)?(?P<rel>above|below|over|under) {ST}",
         lambda m: (f"{'crossover' if _cmp(m.group('rel')) == '>' else 'crossunder'}({c}, {st_args(m)})" if m.group("cross")
                    else f"{c} {_cmp(m.group('rel'))} {st_args(m)}"))
    def st_dir(m):
        if m.group("dir"):   # the direction itself (TradingView: -1 = up, +1 = down) changing on this bar
            d = st_args(m).replace("supertrend(", "supertrend_dir(")
            return f"{d} < 0 and ref({d}, 1) > 0" if m.group("up") else f"{d} > 0 and ref({d}, 1) < 0"
        return f"{'crossover' if m.group('up') else 'crossunder'}({c}, {st_args(m)})"
    take(rf"{ST}(?:'s)?(?P<dir> direction| trend)? (?:turns |flips |flipped |switches |changes |changed |goes )?(?:to )?"
         rf"(?:(?P<up>bullish|up|long|green|positive|buy)|(?P<dn>bearish|down|short|red|negative|sell))", st_dir)
    if re.search(r"supertrend(?:\s*\([^)]*\))?(?:'s)?(?: direction| trend)? (?:turns|flips|flipped|switches|changes|changed)\b", s):
        raise ParseError("'the supertrend flips': which way? Say 'the supertrend flips bullish' (the close crosses above it) or "
                         "'the supertrend flips bearish' (the close crosses below it).")
    take(r"(?:(?:the )?(?:parabolic )?sar|the dots?) (?:flips?|flipped|crosses|crossed|moves?|moved|goes|went|turns?|turned|switches) "
         r"(below|under|above|over) (?:the )?(?:price|close|closing price|candles?|bars?)",
         lambda m: f"{'crossover' if _cmp(m.group(1)) == '<' else 'crossunder'}({c}, sar())" if ctx.base else _unsupported("SAR of another ticker"))
    take(r"(?:closes? |is |trades? |price )?(?:(cross(?:es|ed)?(?: back)?) )?(above|below|over|under) (?:the )?(?:parabolic )?sar",
         lambda m: (f"{'crossover' if _cmp(m.group(2)) == '>' else 'crossunder'}({c}, sar())" if m.group(1)
                    else f"{c} {_cmp(m.group(2))} sar()") if ctx.base else _unsupported("SAR of another ticker"))
    take(r"(?:closes? |is |trades? |price )?(?:(cross(?:es|ed)?(?: back)?) )?(above|below|over|under) (?:the |its )?(upper|lower) keltner(?: channel)?(?: band| line)?",
         lambda m: ((f"{'crossover' if _cmp(m.group(2)) == '>' else 'crossunder'}({c}, keltner_{m.group(3)}(20, 2))" if m.group(1)
                     else f"{c} {_cmp(m.group(2))} keltner_{m.group(3)}(20, 2)") if ctx.base else _unsupported("Keltner channels of another ticker")))

    # gaps
    def gap(m):
        pct = m.group(2)
        g = "gap" if ctx.base else f"({o} / ref({c}, 1) - 1)"
        if pct is None:
            return f"{g} {'<' if m.group(1) == 'down' else '>'} 0"
        f = float(pct) / 100
        return f"{g} {_bound('<', m.group(0))} {-f:g}" if m.group(1) == "down" else f"{g} {_bound('>', m.group(0))} {f:g}"
    take(rf"(?:opens? with a |has a )?gaps? (down|up)(?: by)?(?: more than| at least| over)?(?: {NUM}%)?(?: or more)?", gap)

    # ATR as a fraction of price, or in price units
    ATRP = r"(?:(?:the )?(?:(\d+) (?:day|period|bar) )?(?:atr|average true range)\s*(?:\(\s*(\d+)\s*\))?)"
    take(rf"{ATRP} (?:is )?{CMPW} {NUM}% of (?:the )?(?:price|close|closing price)",
         lambda m: f"natr({m.group(1) or m.group(2) or 14}) {_cmp(m.group(3))} {float(m.group(4)) / 100:g}" if ctx.base
         else _unsupported("ATR of another ticker"))
    take(rf"{ATRP} (?:is )?{CMPW} \$?{NUM}(?![\d%])",
         lambda m: f"atr({m.group(1) or m.group(2) or 14}) {_cmp(m.group(3))} {m.group(4)}" if ctx.base
         else _unsupported("ATR of another ticker"))

    # volatility
    def vol(m):
        n, kind, rel, v = int(m.group(1) or 20), (m.group("kind") or "").strip(), m.group("rel"), float(m.group("v"))
        return _vol_threshold(m.group(0).strip(), n, kind, _cmp(rel), v / 100, ctx)
    take(rf"(?:(\d+) day )?(?P<kind>historical |realized |realised |annuali[sz]ed |daily )?(?:volatility|vol) (?:is )?(?P<rel>{CMPW[1:-1]}) (?P<v>{NUM[1:-1]})%", vol)

    # percentage moves over N days. The direction comes from which verb list matched (never from
    # string tests on the verb), so every auxiliary form ("has fallen", "was down") keeps its sign.
    def move(m):
        pct = float(m.group("pct")) / 100
        n, unit = m.group("n"), m.group("unit")
        n = _period(n, unit) if n else (_period(1, unit) if unit else 1)
        r = ctx.ret(n) if ctx.total else (chg if n == 1 else f"ret({c}, {n})")
        if m.group("yday"):
            if m.group("pre"):
                raise ParseError(f"'{m.group(0).strip()}': 'yesterday' and a period together are ambiguous; write it in backticks, "
                                 f"e.g. `ref(ret(close, 3), 1) <= -0.05`.")
            r = f"ref({r}, 1)"
            _note(f"'{m.group(0).strip()}' = the previous bar's 1-day move ({r}); the rule is checked at today's close. "
                  "Drop 'yesterday' for today's move.")
        return f"{r} {_bound('<', m.group(0))} {-pct:g}" if m.group("dn") else f"{r} {_bound('>', m.group(0))} {pct:g}"
    take(rf"{MOVE_AUX}(?:(?P<dn>{DOWN_VERBS})|(?P<up>{UP_VERBS}))(?: by)?(?: more than| at least| over| greater than)? (?P<pct>{NUM[1:-1]})%(?: or more)?"
         rf"(?P<pre>(?: in| over| during| within)(?: the)?(?: last| past| prior| previous)? (?:(?P<n>\d+) )?(?:trading )?(?P<unit>day|week|month|session|bar|year)s?| today| on the day| in a single day| in one day| intraday)?"
         rf"(?P<yday> yesterday(?!')| on the (?:previous|prior) (?:day|session)| the (?:previous|prior) (?:day|session)| the day before)?", move)

    def ret_cmp(m):
        n = _period(m.group(1) or 1, m.group(2))
        return f"{ctx.ret(n) if ctx.total else f'ret({c}, {n})'} {_cmp(m.group(3))} {float(m.group(4)) / 100:g}"
    take(rf"(?:its |the )?(?:(\d+) (day|week|month|year) )?(?:return|performance|momentum) (?:is )?{CMPW} (-?{NUM})%", ret_cmp)
    take(r"(?:its |the )?(?:(\d+) (day|week|month|year) )?(?:return|performance|momentum) (?:is )?(positive|negative)",
         lambda m: f"{(lambda n: ctx.ret(n) if ctx.total else f'ret({c}, {n})')(_period(m.group(1) or 1, m.group(2)))} {'>' if m.group(3) == 'positive' else '<'} 0")

    # rate of change (TradingView's ROC is in percent: ROC(10) > 5 means up more than 5% over 10 bars)
    def roc(m):
        n = int(m.group(1) or m.group(2) or m.group(3) or 9)
        r = ctx.ret(n) if ctx.total else f"ret({c}, {n})"
        if m.group("cross"):
            return f"{'crossover' if m.group('rel') in ('above', 'over') else 'crossunder'}({r}, {float(m.group('v')) / 100:g})"
        return f"{r} {_cmp(m.group('rel'))} {float(m.group('v')) / 100:g}"
    take(rf"(?:the )?(?:(\d+) (?:day|period|bar) )?(?:roc|rate of change)\s*(?:\(\s*(\d+)\s*\)|(\d+))?(?: value| reading)? (?:is )?"
         rf"(?P<cross>cross(?:es|ed)? )?(?P<rel>{CMPW[1:-1]}) (?P<v>-?{NUM[1:-1]})%?", roc)

    # internal bar strength / position in range
    take(rf"ibs (?:is )?{CMPW} {NUM}", lambda m: f"{'ibs' if ctx.base else f'(({c} - {l}) / ({h} - {l}))'} {_cmp(m.group(1))} {m.group(2)}")
    take(rf"closes? in the (bottom|lower|top|upper) {NUM}% of (?:its|the)(?: day's| daily)? range",
         lambda m: f"ibs < {float(m.group(2)) / 100:g}" if m.group(1) in ("bottom", "lower") else f"ibs > {1 - float(m.group(2)) / 100:g}")
    take(r"closes? (?:near|at) (?:its |the )?(?:daily |day's )?low", lambda m: "ibs < 0.2")
    take(r"closes? (?:near|at) (?:its |the )?(?:daily |day's )?high", lambda m: "ibs > 0.8")

    # previous-day references
    fld = {"high": h, "low": l, "close": c, "open": o}
    take(r"(?:the |today's |its |today )?(?P<a>high|low|open|close) (?:is |was |closes |trades )?(?P<cross>cross(?:es|ed)? )?(?P<rel>below|under|above|over|<=|>=|<|>) "
         r"(?:the )?(?:previous|prior|yesterday's|yesterdays|last) (?:day's |days |session's |bar's )?(?P<b>high|low|close|open)",
         lambda m: (f"{'crossover' if _cmp(m.group('rel'))[0] == '>' else 'crossunder'}({fld[m.group('a')]}, ref({fld[m.group('b')]}, 1))"
                    if m.group("cross") else f"{fld[m.group('a')]} {_cmp(m.group('rel'))} ref({fld[m.group('b')]}, 1)"))
    take(r"(?:closes?|is|trades?|price is|moves?|goes|rises|falls|drops) (below|under|above|over|<=|>=) (?:the )?(?:previous|prior|yesterday's|yesterdays|last) (?:day's |days |session's |bar's )?(high|low|close|open)",
         lambda m: f"{c} {_cmp(m.group(1))} ref({ {'high': h, 'low': l, 'close': c, 'open': o}[m.group(2)] }, 1)")
    take(r"opens? (below|under|above|over|<=|>=) (?:the )?(?:previous|prior|yesterday's|yesterdays|last) (?:day's |days |session's |bar's )?(high|low|close|open)",
         lambda m: f"{o} {_cmp(m.group(1))} ref({ {'high': h, 'low': l, 'close': c, 'open': o}[m.group(2)] }, 1)")
    take(r"inside day", lambda m: f"({h} < ref({h}, 1)) and ({l} > ref({l}, 1))")
    take(r"outside day", lambda m: f"({h} > ref({h}, 1)) and ({l} < ref({l}, 1))")

    # bollinger bands (20-day, 2 standard deviations; "the middle band" is the 20-day SMA)
    def band(m):
        which = m.group(3)
        if which in ("middle", "mid", "center", "centre", "basis"):
            b = f"sma({c}, 20)"
        elif ctx.base:
            b = f"bb_{which}(20, 2)"
        else:
            b = f"(sma({c}, 20) {'-' if which == 'lower' else '+'} 2 * stdev({c}, 20))"
        if m.group(1):
            return f"{'crossover' if _cmp(m.group(2)) == '>' else 'crossunder'}({c}, {b})"
        return f"{c} {_cmp(m.group(2))} {b}"
    take(r"(?:(?:closes?|is|trades?|falls|drops|rises|moves|goes|price|price is) )?(?:back )?(cross(?:es|ed)? (?:back )?)?(below|under|above|over) (?:the |its )?"
         r"(lower|upper|middle|mid|center|centre|basis) (?:bollinger )?(?:band|line)", band)

    # volume
    s = re.sub(r"\b(?:is )?(twice|double|triple)\b(?= (?:its |the )?(?:\d+ day )?(?:average|avg))",
               lambda m: {"twice": "2x", "double": "2x", "triple": "3x"}[m.group(1)], s)
    take(rf"volume (?:is )?(?:at least |more than |above |over )?{NUM} ?(?:x|times) (?:its |the )?(?:(\d+) day )?(?:average|avg)(?: volume)?(?: of the (?:last|past) (\d+) days)?",
         lambda m: f"{v} {_bound('>', m.group(0))} {m.group(1)} * sma({v}, {m.group(2) or m.group(3) or 20})")

    # a price crossing a level: "VIX crosses above 30" (the close crossing the number: TradingView's ta.crossover)
    take(rf"(?:closes? |is |trades? |price )?cross(?:es|ed)?(?: back)? {CMPW} \$?(-?{NUM})(?![\d%])(?! (?:day|week|month|bar))",
         lambda m: f"{'crossover' if _cmp(m.group(1))[0] == '>' else 'crossunder'}({c}, {m.group(2)})", level=True)
    # plain level comparisons, e.g. "VIX is above 30", "closes above 100"
    take(rf"(?:closes?|is|trades?|stays?) {CMPW} \$?(-?{NUM})(?![\d%])(?! (?:day|week|month|bar))",
         lambda m: f"{c} {_cmp(m.group(1))} {m.group(2)}", level=True)

    # calendar
    take(r"(?:on )?(monday|tuesday|wednesday|thursday|friday)s?", lambda m: f"dow == {DOW[m.group(1)]}")
    take(r"(?:on |at )?(?:the )?month[- ]end", lambda m: "trading_days_left_in_month == 0")
    take(rf"(?:on )?(?:the )?(?:({ORD}) (?:to |from )?last|last) (?:trading )?day (?:of|in) (?:the |each |every )?month",
         lambda m: f"trading_days_left_in_month == {_ord(m.group(1)) - 1 if m.group(1) else 0}")
    take(rf"(?:on )?(?:the )?({ORD}) trading day (?:of|in) (?:the |each |every )?(?:next |following |new )?month",
         lambda m: f"trading_day_of_month == {_ord(m.group(1))}")
    take(r"(?:on |in |during )?(?:the )?(first|last) (\d+) trading days (?:of|in) (?:the |each |every )?month",
         lambda m: f"trading_day_of_month <= {m.group(2)}" if m.group(1) == "first" else f"trading_days_left_in_month <= {int(m.group(2)) - 1}")
    take(r"(?:during |in )?(?:the )?(january|february|march|april|may|june|july|august|september|october|november|december)",
         lambda m: f"month == {MONTHS.index(m.group(1)) + 1}")

    # single down / up day (after the % and streak patterns)
    take(r"(?:closes? (?:down|lower|red)|closed (?:down|lower|red)|down (?:day|close)|red day|negative day|first down (?:day|close)|trades? down|"
         r"is down|was down|goes down|went down|moves down|moved down|falls|fell|drops|dropped|declines|declined)",
         lambda m: f"{chg} < 0")
    take(r"(?:closes? (?:up|higher|green)|closed (?:up|higher|green)|up (?:day|close)|green day|positive day|first up (?:day|close)|higher close|"
         r"trades? up|is up|was up|goes up|went up|moves up|moved up|rises|rose|gains|gained)",
         lambda m: f"{chg} > 0")
    take(r"(?:is |the (?:trade|position) is )?(?:profitable|in profit|shows a profit)", lambda m: "pnl > 0")

    raw_words = re.findall(r"[a-z%]+|\d+(?:\.\d+)?", s)
    words = [w for w in raw_words if w not in STOP]
    # One comparison, one subject: a bare "is above 79" (the price) next to another comparison in the same
    # clause means a phrase was split in two ("<indicator> over 10 days is above 79"), so refuse it.
    other = [p for p in parts if p not in level_parts and not re.match(r"(?:dow|month|trading_day)", p)]
    if level_parts and other:
        raise ParseError(f"'{text.strip()}' reads as two comparisons ({other[0]} and {level_parts[0]}): one condition must "
                         "compare one indicator with one threshold. Rephrase, e.g. '10 day RSI is above 79', or use backticks.")
    # a time unit left over means a lookback was not attached to anything
    if parts and not words and any(w in ("day", "days") for w in raw_words):
        words = [w for w in raw_words if w in ("day", "days")]
    expr_ = " and ".join(f"({p})" for p in parts) if parts else None
    return expr_, " ".join(words)


def _vol_threshold(phrase: str, n: int, kind: str, op: str, v: float, ctx: Ctx) -> str:
    """'N day volatility' compared with a threshold. 'annualized' -> volatility() (annualised stdev of daily
    returns); 'daily' -> stdev_return() (the plain daily stdev, as Composer's 'standard deviation of
    return'); bare 'volatility' with a threshold below 10% is refused as ambiguous."""
    kind, phrase = kind.lower(), re.sub(r"\s+", " ", phrase)
    if kind == "daily":
        return f"stdev_return({ctx.tr}, {n}) {op} {v:g}"
    if not kind.startswith("annuali") and abs(v) < 0.10:
        raise ParseError(f"'{phrase}': is {v * 100:g}% a daily or an annualised volatility? Say '{n} day standard deviation "
                         f"of return' (daily, e.g. above 2%) or '{n} day annualized volatility' (e.g. above 20%).")
    if not kind.startswith("annuali"):
        _note(f"'{phrase}': volatility is annualised (standard deviation of daily returns x sqrt(252)); say "
              f"'{n} day standard deviation of return' for the daily figure.")
    return f"volatility({n}{'' if ctx.base else ', ' + ctx.c}) {op} {v:g}"


def _negations(s: str) -> str:
    """'is not above 79' -> 'is <= 79'; 'is not below' -> 'is >='; 'does not close above' -> 'is <='."""
    s = re.sub(r"(?i)\b(?:does|did|do|has|have) not (?:close|closed|trade|traded|stay|stayed|remain|remained|go|gone|move|moved|get|got) "
               r"(above|over|greater than|higher than|more than|below|under|less than|lower than)(?![a-z])",
               lambda m: "is <=" if m.group(1).lower() in ("above", "over", "greater than", "higher than", "more than") else "is >=", s)
    s = re.sub(r"(?i)\b(?:is|closes?|trades?|stays?|remains?) not (?:above|over|greater than|higher than|more than)(?![a-z])", "is <=", s)
    s = re.sub(r"(?i)\b(?:is|closes?|trades?|stays?|remains?) not (?:below|under|less than|lower than)(?![a-z])", "is >=", s)
    # a bare second bound of the same subject: "is not below 30 and not above 70" -> "is >= 30 and <= 70"
    s = re.sub(r"(?i)(?:(?<=\band )|(?<=\bbut )|(?<=\bor ))not (?:above|over|greater than|higher than|more than)(?= -?\$?\d)", "<=", s)
    return re.sub(r"(?i)(?:(?<=\band )|(?<=\bbut )|(?<=\bor ))not (?:below|under|less than|lower than)(?= -?\$?\d)", ">=", s)


_MOTION_RX = (r"(?i)(?<!\bnot )(?<!\bnever )\b(?P<v>(?P<dn>falls?|fell|drops?|dropped|dips?|dipped|sinks?|sank)|(?P<up>rises?|rose|climbs?|climbed))"
              r"(?: to)? (?P<d>below|under|above|over)\b"
              # "rises over 10 days" is a period, not a level
              r"(?! (?:the )?(?:last |past )?\d+(?:\.\d+)? (?:trading )?(?:day|week|month|year|bar|session|period)s?\b)"
              r"(?P<rest>(?: (?!(?:and|or|then|else|otherwise|but)\b)[^\s,;]+){0,6})")


def _motion_verbs(text: str, total: bool) -> str:
    """'falls below 30' / 'drops below' / 'rises above 70': a move through a level.

    In a signal rule (entries, exits: evaluated once per bar as an event) it is the crossing, as TradingView users
    mean it: 'RSI(2) falls below 10' = crossunder(rsi(close, 2), 10), true only on the bar RSI moves from 10 or more
    to below 10. In a portfolio condition (a state checked every day: which branch to hold) it is the level: 'SPY 10
    day RSI falls below 30' = rsi(close, 10) < 30 on every day it is below. Either way a note says which, and how to
    say the other."""
    def fix(m):
        below = m.group("d").lower() in ("below", "under")
        if bool(m.group("dn")) != below:     # "falls above", "rises below": not a move through a level
            return m.group(0)
        # "SPY rose over 5% in the last 10 days" / "fell under 3%": the size of the move (a return), not a level
        if m.group("d").lower() in ("over", "under") and re.match(r"\s*-?\d+(?:\.\d+)?\s*%", m.group("rest")):
            return m.group(0)
        d = "below" if below else "above"
        phrase = re.sub(r"\s+", " ", f"{m.group('v')} {m.group('d')}{m.group('rest')}").strip()
        if total:
            _note(f"'{phrase}' was read as 'is {d}{m.group('rest')}' (a state checked each day: the branch is held "
                  f"every day it is {d}); say 'crosses {d}{m.group('rest')}' for the crossing day only.")
            return f"is {d}{m.group('rest')}"
        _note(f"'{phrase}' was read as a crossing ({'crossunder' if below else 'crossover'}: true only on the bar it "
              f"moves from {'at or above' if below else 'at or below'} the level to {d} it, as in TradingView); say 'is "
              f"{d}{m.group('rest')}' for every bar it is {d}.")
        return f"crosses {d}{m.group('rest')}"
    return _sub_outside(_MOTION_RX, fix, text)


def _as_crossing(rule: str) -> str | None:
    """A single comparison 'a < b' / 'a > b' -> crossunder(a, b) / crossover(a, b); None for anything else."""
    try:
        node = ast.parse(rule.strip(), mode="eval").body
    except SyntaxError:
        return None
    if not (isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], (ast.Lt, ast.LtE, ast.Gt, ast.GtE))):
        return None
    a, b = ast.unparse(node.left), ast.unparse(node.comparators[0])
    return f"{'crossunder' if isinstance(node.ops[0], (ast.Lt, ast.LtE)) else 'crossover'}({a}, {b})"


def _unsupported(what: str) -> str:
    raise ParseError(f"{what} is not supported in English; write it in backticks (see --help-expr)")


def _split_top(text: str, word: str) -> list[str]:
    # "2% or more", "or less", "or higher" are part of a phrase, not an alternative; never split
    # inside backticks or brackets
    return _msplit(text, rf"\b{word}\b(?! (?:(?:more|less|fewer|higher|lower|greater)\b(?! than\b)|(?:better|worse|so|after|in \d)\b))", flags=0)


def _sub_outside(pattern: str, repl, s: str, flags=0, count: int = 0) -> str:
    """re.sub applied only to the text outside `backtick` blocks (at most `count` replacements, 0 = all)."""
    parts = re.split(r"(`[^`]*`)", s)
    out, left = [], count
    for p in parts:
        if (p.startswith("`") and p.endswith("`") and len(p) > 1) or (count and left <= 0):
            out.append(p)
            continue
        if count:
            p, k = re.subn(pattern, repl, p, count=left, flags=flags)
            left -= k
        else:
            p = re.sub(pattern, repl, p, flags=flags)
        out.append(p)
    return "".join(out)


# comparison words of a threshold condition ("<indicator> <comparison> <number or indicator>")
TH_CMP = [
    (r"(?:is |are )?(?:greater than or equal to|at least|>=|no less than)", ">="),
    (r"(?:is |are )?(?:less than or equal to|at most|<=|no more than)", "<="),
    (r"(?:is |are )?(?:greater than|higher than|above|more than|over|exceeds?|>)", ">"),
    (r"(?:is |are )?(?:less than|lower than|below|under|<)", "<"),
    (r"(?:is |are )?(?:exactly equal to|exactly|equal to|equals?|==)", "=="),
]
_EQ_NOTE = ("exact equality rarely holds for a value computed from prices (an RSI of 50.0001 is not 50), so the "
            "condition may almost never be true; 'is at least' / 'is between' are usually what is meant.")
_FRACTION_FNS = ("tret(", "ret(", "ma_return(", "stdev_return(", "max_drawdown(", "drawdown(", "volatility(", "change",
                 "weekly_ret(", "monthly_ret(")
_DIST_RE = re.compile(r".*\s/\s(?:sma|ema|lowest|highest|cummin|cummax)\(.*\)\s-\s1\s*$")


def _fraction_valued(le: str) -> bool:
    """Is the indicator a fraction written in percent (returns, drawdowns, volatility, distance from a moving
    average or from a high or low)?"""
    return le.startswith(_FRACTION_FNS) or bool(_DIST_RE.fullmatch(le))


def _pct_missing(phrase: str, le: str, val: float) -> str:
    """The refusal for a bare number compared with a percent-valued indicator."""
    opts = f"{val:g}%" if abs(val) >= 1 else f"{val:g}% or {val * 100:g}%"
    return (f"'{phrase}': {le} is a percentage, so a bare {val:g} is ambiguous - did you mean {opts}? Write the % "
            "sign for returns, drawdowns, volatility and distances from an average, high or low.")


_OSC_FNS = ("rsi(", "weekly_rsi(","monthly_rsi(", "stoch_k(", "stoch_d(", "cci(", "willr(", "mfi(", "adx(")


FLIP_OP = {">": "<", ">=": "<=", "<": ">", "<=": ">="}


def _neg_max_drawdown(phrase: str, le: str, op: str, val: float) -> str:
    """max_drawdown() is a positive size (0.2 = a 20% fall). A negative threshold is the same size written as a
    loss, as some tools show it: 'max drawdown below -20%' = a fall deeper than 20% (max_drawdown > 0.2), 'above
    -20%' = shallower than 20%. Compared as written it would never (or always) be true."""
    flip = FLIP_OP[op]
    deeper = op[0] == "<"
    phrase = re.sub(r"\s+", " ", phrase)
    _note(f"'{phrase}': the max drawdown is measured as a positive size here (0.2 = a 20% fall), so {val * 100:g}% was read "
          f"as a fall {'deeper' if deeper else 'shallower'} than {-val * 100:g}%: {le} {flip} {-val:g}.")
    return f"{le} {flip} {-val:g}"


def _threshold(text: str, ctx: Ctx) -> str | None:
    """'<indicator phrase> <comparison> <number, % or indicator phrase>' as one rule, e.g.
    '6 day cumulative return is less than -12%' -> tret(tr, 6) < -0.12 (value_phrase vocabulary).
    None when the text is not of that shape."""
    t = " " + re.sub(r"\s+", " ", text.strip().lower()) + " "
    if "`" in t:
        return None
    mp = re.fullmatch(r" (.+?) (?:is|are|turns?|stays?) (positive|negative) ", t)
    if mp:
        try:
            le, _ = value_phrase(mp.group(1), ctx, default_n=None, total=ctx.total)
        except ParseError:
            return None
        return f"{le} {'>' if mp.group(2) == 'positive' else '<'} 0"
    # "max drawdown is worse than 10%" = a fall deeper than 10%; "better than" = shallower. The sign of the number
    # does not matter (a drawdown is a fall whichever way it is written); for returns worse = lower, better = higher
    mw = re.fullmatch(r" (.+?) (?:is |are |was |has been |gets? )?(worse|deeper|bigger|larger|better|shallower|smaller) than "
                      r"(-?\d+(?:\.\d+)?)% ", t)
    if mw:
        try:
            le, lnotes = value_phrase(mw.group(1), ctx, default_n=None, total=ctx.total)
        except ParseError:
            le = None
        if le is not None:
            worse = mw.group(2) in ("worse", "deeper", "bigger", "larger")
            val = abs(float(mw.group(3))) / 100
            for n_ in lnotes:
                _note(n_)
            if le.startswith("max_drawdown("):
                rule = f"{le} {'>' if worse else '<'} {val:g}"
                _note(f"'{text.strip()}': a fall {'deeper' if worse else 'shallower'} than {val * 100:g}% ({rule}; the max "
                      "drawdown is measured as a positive size here, 0.1 = a 10% fall).")
                return rule
            if le.startswith("drawdown("):
                rule = f"{le} {'<' if worse else '>'} {-val:g}"
                _note(f"'{text.strip()}': {'more' if worse else 'less'} than {val * 100:g}% below the high ({rule}; "
                      "drawdown() is zero or negative).")
                return rule
            if le.startswith(_FRACTION_FNS) and mw.group(2) in ("worse", "better"):
                v = float(mw.group(3)) / 100
                return f"{le} {'<' if worse else '>'} {v:g}"
    for pat, op in TH_CMP:
        for m in re.finditer(rf"\s{pat}\s", t):
            lhs, rhs = t[: m.start()], t[m.end():]
            if not lhs.strip() or not rhs.strip():
                continue
            try:
                le, lnotes = value_phrase(lhs, ctx, default_n=None, total=ctx.total)
            except ParseError as e:
                if "two different lookbacks" in str(e):
                    raise
                continue
            mn = re.fullmatch(r"\s*(?:\$\s*)?(-?\d+(?:\.\d+)?)\s*(%)?\s*", rhs)
            if mn:
                val = float(mn.group(1))
                frac = _fraction_valued(le)
                if mn.group(2):
                    if not frac:
                        raise ParseError(f"'{text.strip()}': {le} is not a percentage; drop the % sign.")
                    val /= 100
                elif frac and val != 0:
                    # a bare number on a percent-valued indicator is ambiguous ("0.1": 0.1% or 10%?): never guess
                    raise ParseError(_pct_missing(text.strip(), le, val))
                for n_ in lnotes:
                    _note(n_)
                if le.startswith("volatility("):
                    return _vol_threshold(text.strip(), int(re.match(r"volatility\((\d+)", le).group(1)),
                                          "annualized" if re.search(r"annuali[sz]ed", lhs) else "", op, val, ctx)
                if le.startswith("drawdown(") and val > 0:
                    # "the drawdown is above 20%": a fall of more than 20% from the high (drawdown() is <= 0)
                    flip = {">": "<", ">=": "<=", "<": ">", "<=": ">="}[op]
                    _note(f"'{text.strip()}' was read as a fall of {'more' if op[0] == '>' else 'less'} than "
                          f"{val * 100:g}% from the high: {le} {flip} {-val:g} (drawdown() is zero or negative).")
                    return f"{le} {flip} {-val:g}"
                if le.startswith("max_drawdown(") and val < 0 and op != "==":
                    return _neg_max_drawdown(text.strip(), le, op, val)
                if op == "==":
                    _note(f"'{' '.join(text.split())}' = {le} == {val:g}: {_EQ_NOTE}")
                return f"{le} {op} {val:g}"
            try:
                re_, rnotes = value_phrase(rhs, ctx, default_n=None, total=ctx.total)
            except ParseError:
                continue
            for n_ in lnotes + rnotes:
                _note(n_)
            if op == "==":
                _note(f"'{' '.join(text.split())}' = {le} == {re_}: {_EQ_NOTE}")
            return f"{le} {op} {re_}"
    return None


from .expr import OSC_RANGES  # noqa: E402  (one table for the parser, the Build page and JSON specs)


def _check_ranges(rule: str, text: str) -> None:
    """Refuse a threshold an oscillator can never reach ('RSI above 120', 'Williams %R below 80') or one
    written as a fraction ('RSI above 0.7' - did you mean 70?): expr.check_osc_ranges, the check every front end
    shares."""
    from .expr import check_osc_ranges
    try:
        check_osc_ranges(rule, text)
    except ParseError:
        raise
    except ValueError as e:
        raise ParseError(str(e)) from None


def _over_bars(text: str, traded: list[str], total: bool) -> str | None:
    """A condition held over several bars: 'RSI(2) was below 10 within the last 5 bars' -> count(cond, 5) >= 1
    (at least once, today included); 'RSI(2) is below 10 for 2 consecutive days' -> count(cond, 2) == 2 (every
    one of the last 2 bars). None when the text is not of that shape or its condition does not parse."""
    t = re.sub(r"\s+", " ", text.strip())
    if "`" in t:
        return None
    mw = re.fullmatch(r"(?i)(?P<body>.+?),? (?:(?:at least once|at any (?:time|point)|at some point|once) (?:with)?in|within|"
                      r"(?:at least once|at any (?:time|point)|at some point) (?:during|over)) the (?:last|past|previous|prior) "
                      r"(?P<n>\d+) (?:trading )?(?P<u>day|bar|session|week|candle)s?", t)
    mc = re.fullmatch(r"(?i)(?P<body>.+?),? for (?:the (?:last|past) )?(?:at least )?(?P<n>\d+) (?:(?:consecutive|straight|successive|trading) )*"
                      r"(?P<u>day|bar|session|close|week|candle)s?(?: in a row| straight| consecutively| running)?", t)
    m = mw or mc
    if not m:
        return None
    unit = {"candle": "bar", "close": "day"}.get(m.group("u").lower(), m.group("u").lower())
    n = _period(m.group("n"), unit)
    body = re.sub(r"(?i)\b(?:was|were|has been|have been|had been|has stayed|stayed|has remained|remained)\b", "is", m.group("body"))
    try:
        e = parse_conditions(body, traded, strict=True, total=total)
    except ParseError:
        return None
    if n < 1:
        raise ParseError(f"'{t}': the number of bars must be at least 1.")
    if mw:
        _note(f"'{t}' = true on at least one of the last {n} bars, today included: count(..., {n}) >= 1.")
        return f"count(({e}), {n}) >= 1"
    _note(f"'{t}' = true on each of the last {n} bars, today included: count(..., {n}) == {n}.")
    return f"count(({e}), {n}) == {n}"


def _trend(text: str, ctx: Ctx) -> str | None:
    """'the 200 day SMA is rising' -> sma(close, 200) > ref(sma(close, 200), 1): higher than on the previous
    bar (a note says so). 'is falling' / 'declining' is the reverse. None when the text is not of that shape."""
    t = re.sub(r"\s+", " ", text.strip().lower())
    m = re.fullmatch(r"(?:(?P<x>.+?) )?(?:is |are |has been |have been )?(?:(?P<up>rising|increasing|going up|trending up|sloping up|"
                     r"pointing up|moving up)|(?P<dn>falling|decreasing|declining|going down|trending down|sloping down|pointing down|"
                     r"moving down))(?: today)?", t)
    if not m or "`" in t:
        return None
    x = re.sub(r"(?:^|\s)(?:is|are|has been|have been)$", "", (m.group("x") or "").strip())
    x = re.sub(r"^(?:it|its|the)\b\s*", "", x.strip()).strip() or "price"
    try:
        e, notes = value_phrase(x, ctx, default_n=None, total=ctx.total)
    except ParseError:
        return None
    if re.fullmatch(r"(?:its |the )?(?:price|close|closing price)", x) is None and not re.search(r"\d|rsi|average|sma|ema|ma\b", x):
        return None     # a lookback is needed to say what is rising (except for the price itself)
    for n_ in notes:
        _note(n_)
    op = ">" if m.group("up") else "<"
    phrase = re.sub(r"\s+", " ", text.strip())
    _note(f"'{phrase}' = {'higher' if op == '>' else 'lower'} than on the previous day: {e} {op} ref({e}, 1).")
    return f"{e} {op} ref({e}, 1)"


def _both(m) -> str:
    """'both SPY and QQQ are above their 200 day moving averages' -> 'SPY is above its 200 day moving average
    and QQQ is above its 200 day moving average'."""
    a, b = m.group("a"), m.group("b")
    if len(find_tickers(a.upper(), strict=False)) != 1 or len(find_tickers(b.upper(), strict=False)) != 1:
        return m.group(0)
    rest = re.sub(r"(?i)\btheir\b", "its", m.group("rest"))
    rest = re.sub(r"(?i)\b(averages|smas|emas|mas|highs|lows|bands|lines)\b", lambda x: x.group(1)[:-1], rest)
    return f"{a} is {rest} and {b} is {rest}"


RANGE_VERB = r"(?:is|are|stays?|remains?|trades?|closes?|reads?)"
# a bare second bound that continues the previous subject ("RSI is above 79 and below 90"); "closes above 5" / "trades
# below 5" name the price, so they are a comparison of their own ("RSI(2) is above 10 and closes above 5" is not RSI > 5)
RANGE_VERB_SAME = r"(?:is|are|stays?|remains?|reads?)"
RANGE_CMP = r"(?:at least|at most|no less than|no more than|below|under|less than|lower than|above|over|greater than|higher than|more than|<=?|>=?)"
RANGE_NUM = r"-?\$?\d+(?:\.\d+)?%?"


def _expand_ranges(parts: list[str]) -> list[str]:
    """'X is between 30 and 50' -> 'X is at least 30', 'X is at most 50'; 'X is above 79 and below 90' ->
    'X is above 79', 'X is below 90' (a bare comparison after 'and' takes the previous clause's subject)."""
    out: list[str] = []
    for part in parts:
        if "\x02" in part:
            m = re.fullmatch(rf"(?is)\s*(?P<subj>.*?)\s*\b(?P<verb>{RANGE_VERB} )?(?:(?:somewhere |anywhere )?between) (?P<a>{RANGE_NUM}) \x02 (?P<b>{RANGE_NUM})\s*", part)
            if not m or "`" in part:
                raise ParseError(f"'{part.replace(chr(2), 'and').strip()}': write a range as '<indicator> is between 30 and 50'.")
            a, b = m.group("a"), m.group("b")
            if a.endswith("%") != b.endswith("%"):
                raise ParseError(f"'{part.replace(chr(2), 'and').strip()}': give both ends of the range in the same units (e.g. 30% and 50%).")
            if float(a.strip("$%")) > float(b.strip("$%")):
                a, b = b, a
            if float(a.strip("$%")) == float(b.strip("$%")):
                raise ParseError(f"'{part.replace(chr(2), 'and').strip()}': the two ends of the range are the same.")
            head = f"{m.group('subj')} {m.group('verb') or 'is '}".lstrip()
            _note(f"'{part.replace(chr(2), 'and').strip()}' was read as {a} or more and {b} or less (both ends included).")
            out += [f"{head}at least {a}", f"{head}at most {b}"]
            continue
        mb = re.fullmatch(rf"(?is)\s*(?:(?:and|but) )?(?:{RANGE_VERB_SAME} )?(?P<rel>{RANGE_CMP}) (?P<v>{RANGE_NUM})\s*", part)
        prev = re.fullmatch(rf"(?is)\s*(?P<subj>.*?)\s*\b(?P<verb>{RANGE_VERB} )?{RANGE_CMP} {RANGE_NUM}\s*", out[-1]) if out and mb else None
        if mb and prev and (prev.group("subj").strip() or prev.group("verb")) and "`" not in out[-1]:
            head = f"{prev.group('subj')} {prev.group('verb') or 'is '}".lstrip()
            out.append(f"{head}{mb.group('rel')} {mb.group('v')}")
            continue
        out.append(part)
    return out


def _or_subjects(parts: list[str]) -> list[str]:
    """'RSI is below 30 or above 70' -> 'RSI is below 30', 'RSI is above 70': a bare comparison after 'or' takes the
    subject of the comparison just before it (as _expand_ranges does after 'and')."""
    out: list[str] = []
    for part in parts:
        mb = re.fullmatch(rf"(?is)\s*(?:{RANGE_VERB_SAME} )?(?P<rel>{RANGE_CMP}) (?P<v>{RANGE_NUM})(?P<rest>(?:\s+(?:and|,).*)?)\s*", part)
        if mb and out and "`" not in out[-1]:
            last = _msplit(out[-1], r"\band\b|,|;", flags=0)[-1]
            prev = re.fullmatch(rf"(?is)\s*(?P<subj>.*?)\s*\b(?P<verb>{RANGE_VERB} )?{RANGE_CMP} {RANGE_NUM}\s*", last)
            if prev and prev.group("subj").strip():
                out.append(f" {prev.group('subj')} {prev.group('verb') or 'is '}{mb.group('rel')} {mb.group('v')}{mb.group('rest')}")
                continue
        out.append(part)
    return out


def _balanced(t: str) -> bool:
    """Are the brackets in t balanced (so 'not (A) and (B)' is not read as 'not (A) and (B)' = not(...))?"""
    d = 0
    for ch in t:
        d += (ch == "(") - (ch == ")")
        if d < 0:
            return False
    return d == 0


_GT_TK = r"[\^$]?[A-Z][A-Z0-9]{0,5}(?:[.\-][A-Z]{1,3})?"
_GT_SEP = r"(?:\s*,\s*(?:and\s+|or\s+|nor\s+|&\s*)?|\s+(?:and|or|nor|&)\s+)"
_GT_ALL = ("both", "all", "all of", "each of", "every one of", "all three of", "all four of")
_GT_ANY = ("any", "any of", "any one of", "either", "one of", "at least one of")
_GT_NONE = ("neither", "none of")
_GT_RX = re.compile(rf"(?<![\w^$.\-])(?:(?P<q>(?i:{'|'.join(sorted(_GT_ALL + _GT_ANY + _GT_NONE, key=len, reverse=True))}))\s+)?"
                    rf"(?P<list>{_GT_TK}(?:{_GT_SEP}{_GT_TK})+)(?P<poss>'s?)?(?=\s)")


def _ticker_groups(text: str, traded: list[str], strict: bool, total: bool) -> tuple[str, list]:
    """A comparison written once for several tickers -> one placeholder per group (\\x03i\\x03) and its compiled rule:

      both SPY and QQQ 10 day RSI are above 79           -> each of them (and)
      all of SPY, QQQ and SMH have a 10 day RSI above 79 -> and;  "SPY and QQQ ..." without a quantifier -> and
      any of SPY, QQQ and SMH has a 10 day RSI above 79  -> at least one (or); "either SPY or QQQ ..." -> or
      the 10 day RSI of both SPY and QQQ is above 79     -> the same, with the indicator written first

    A bare list ("SPY, QQQ 10 day RSI is above 79") is refused: every one, or at least one? Never keep one ticker."""
    groups: list = []
    mk = _mask(text, parens=False)
    pos = 0
    while True:
        m = _GT_RX.search(mk, pos)
        if not m:
            break
        toks = re.findall(_GT_TK, m.group("list"))
        tks = [find_tickers(t.lstrip("$").upper() if t.startswith("$") else t, strict=False) for t in toks]
        if not all(len(x) == 1 for x in tks) or len({x[0] for x in tks}) != len(tks):
            pos = m.start() + 1
            continue
        tickers = [x[0] for x in tks]
        # the rest of the clause: up to the next top-level ',', ';', ' and ' or ' or ' (not the 'and' of "above 79 and
        # below 90", which stays with this comparison)
        end_rx = re.compile(rf"(?i)[,;]|\s(?:and|or)\s(?!(?:(?:is|are|stays?|remains?) )?{RANGE_CMP}\s)")
        me = end_rx.search(mk, m.end())
        end = me.start() if me else len(text)
        rest = text[m.end():end].strip()
        # what comes before the list in the same clause: nothing ("SPY and QQQ 10 day RSI ...") or an indicator ending
        # in 'of' / 'for' ("the 10 day RSI of SPY and QQQ ...")
        starts = [x.end() for x in re.finditer(r"(?i)[,;]|\s(?:and|or)\s", mk[: m.start()])]
        cs = starts[-1] if starts else 0
        prefix = text[cs: m.start()].strip()
        if prefix and not re.search(r"(?i)\b(?:of|for|in|on)$", prefix):
            pos = m.start() + 1
            continue
        if not re.search(r"[a-z]", rest):
            pos = m.start() + 1
            continue
        q = (m.group("q") or "").lower()
        words = re.findall(r"(?i)\b(and|or|nor)\b|&", m.group("list"))
        has_and = any(w.lower() in ("and", "") for w in words)
        has_or = any(w.lower() == "or" for w in words)
        has_nor = any(w.lower() == "nor" for w in words)
        src = re.sub(r"\s+", " ", text[cs: end]).strip()
        if (has_nor and q != "neither") or (q == "neither" and (not has_nor or has_and or has_or)):
            raise ParseError(f"'{src}': write 'neither SPY nor QQQ ...' (or 'none of SPY, QQQ and SMH ...') when no "
                             "ticker may meet the condition.")
        if q in _GT_NONE:
            op = "none"
        elif q in _GT_ALL:
            op = "and"
        elif q in _GT_ANY:
            op = "or"
        elif has_and and not has_or:
            op = "and"
        elif has_or and not has_and:
            op = "or"
        else:
            lst = ", ".join(tickers)
            raise AmbiguousTickers(f"'{src}': all of {lst}, or any of them? Write 'all of {lst} ...' (or 'both ... and ...') "
                             f"when every one must meet the condition, 'any of {lst} ...' (or 'either ... or ...') when "
                             "one is enough.")
        r = re.sub(r"(?i)^(?:both|all|each|none)\s+", "", rest)
        r = re.sub(r"(?i)\b(are|were|have|has|is|was)\s+(?:both|all|each)\b", r"\1", r)
        r = re.sub(r"(?i)\bare\b", "is", re.sub(r"(?i)\bwere\b", "was", re.sub(r"(?i)\bhave\b", "has", r)))
        r = re.sub(r"(?i)\btheir\b", "its", r)
        r = re.sub(r"(?i)\b(averages|smas|emas|mas|highs|lows|bands|lines|returns|prices|rsis)\b", lambda x: x.group(1)[:-1], r)
        pre = re.sub(r"(?i)\b(?:both|all|each)\s*$", "", prefix).strip()
        poss = m.group("poss") or ""
        clauses = [f"{pre} {tok}{poss} {r}".strip() for tok in toks]
        es = [parse_conditions(c, traded, strict=strict, total=total) for c in clauses]
        es = [f"({e})" if re.search(r"\s(?:and|or)\s", e) and not (e.startswith("(") and _balanced(e[1:-1])) else e
              for e in es]
        if op == "none":
            rule = "(" + " and ".join(f"not ({e})" for e in es) + ")"
            _note(f"'{src}' = none of {', '.join(tickers)}: {rule}.")
        else:
            rule = "(" + f" {op} ".join(es) + ")"
            _note(f"'{src}' = {'every one' if op == 'and' else 'at least one'} of {', '.join(tickers)}: {rule}.")
        ph = f"\x03{len(groups)}\x03"
        groups.append((rule, src))
        text = text[:cs] + (" " if cs else "") + ph + " " + text[end:]
        mk = _mask(text, parens=False)
        pos = cs + len(ph) + 1
    return text, groups


def _unplace(s: str, groups: list) -> str:
    return re.sub(r"\x03(\d+)\x03", lambda m: groups[int(m.group(1))][1], s)


def _every_ticker_used(o: str, e: str, traded: list[str]) -> str:
    """Every ticker a condition names must be in its rule: the rule reads sym("X") for another ticker, or its own
    series for the ticker it is evaluated on. Otherwise a ticker was dropped: refuse rather than misread."""
    from .expr import reads_own_series
    missing = []
    for t in find_tickers(o):
        if f'sym("{t}")' in e or f"sym('{t}')" in e:
            continue
        if t in traded and (len(traded) > 1 or reads_own_series(e)):
            continue
        missing.append(t)
    if missing:
        raise ParseError(f"'{o.strip()}' names {', '.join(missing)}, but it was read as `{e}`, which does not use "
                         f"{'it' if len(missing) == 1 else 'them'}. Write one comparison per ticker (e.g. 'SPY 10 day RSI "
                         "is above 79 and QQQ 10 day RSI is above 79'), or put the rule in backticks.")
    return e


_RV_TK = r"[\^$]?[A-Za-z][A-Za-z0-9.\-]{0,9}"
_RV_END = r"(?=\s*(?:$|,|;|\bthen\b|\band\b|\bor\b|\bhold\b|\botherwise\b|\belse\b))"


def _review_condition_phrases(text: str) -> str:
    """Condition phrases from a Composer user's review, rewritten into forms the condition parser reads (outside
    backticks). Every rewrite keeps the direction and sign of the original:
      'has a 6 day cumulative return of less than -12%'  -> '6 day cumulative return is less than -12%'
      'less than negative 12%'                          -> 'less than -12%'
      'gained less than 5% over 10 days'                -> '10 day return is less than 5%'  (lost more -> < -5%)
      '10 day return is down more than 5%'              -> '10 day return is less than -5%'
      'under 10% from its 52 week low'                  -> 'less than 10% above its 52 week low'
      "SPY's 20 day return is more than 2% higher than TLT's" -> '... more than 2 points above TLT's 20 day return'
      "TQQQ's 10 day RSI is below SOXL's 10 day RSI minus 5"   -> '... is more than 5 points below SOXL's 10 day RSI'
      'is oversold' / 'is overbought'                   -> '14 day RSI is below 30' / 'above 70' (with a warning)
    'fails to stay above ...' is refused as ambiguous."""
    if not text or not re.search(r"[A-Za-z]", text):
        return text
    mf = _msearch(r"(?i)\b(?:fails?|failed|failing) to (?:stay|hold|remain|close|keep)(?: (?:itself|it))? (above|below|over|under)"
                  r" (.+?)" + _RV_END, text)
    if mf:
        rel, what = mf.group(1).lower(), mf.group(2).strip()
        opp = "below" if rel in ("above", "over") else "above"
        raise ParseError(f"'{mf.group(0).strip()}' is ambiguous: it could mean today's close is {opp} {what}, that it closed "
                         f"{opp} it on any day of a period, or that it crossed {opp} it today. Say which, e.g. 'is {opp} "
                         f"{what}', 'has closed {opp} {what} for 3 days', or 'crosses {opp} {what}'.")
    # "less than negative 12%" / "below minus 5%": a negative threshold
    text = _sub_outside(r"(?i)\b(?:negative|minus) (?=\d+(?:\.\d+)?\s*%)", "-", text)
    # "has a 6 day cumulative return of less than -12%" -> "6 day cumulative return is less than -12%"
    text = _sub_outside(r"(?i)\b(?:has|have|had) (?:a|an) ((?:\d+[- ](?:trading )?(?:day|week|month|year|bar|session)s? )?"
                        r"[a-z][a-z' ]*?) of (?=(?:less|more|greater|lower|higher|at least|at most|above|below|over|under|no more|"
                        r"no less)\b)", r"\1 is ", text)

    # "gained less than 5% over 10 days" -> "10 day return is less than 5%"; "lost more than 5% ..." -> "... less than -5%"
    def moved(m):
        loss = m.group("v").lower() in ("lost", "fell", "fallen", "dropped", "declined")
        q = m.group("q").lower()
        more = q in ("more than", "over", "at least")
        cmp_ = {"more than": "more than", "over": "more than", "at least": "at least", "less than": "less than",
                "under": "less than", "at most": "at most"}[q]
        if loss:     # a loss of more than 5% is a return below -5%
            cmp_ = {"more than": "less than", "at least": "at most", "less than": "more than", "at most": "at least"}[cmp_]
        _note(f"'{m.group(0).strip()}' = the {m.group('k')} {m.group('u')} return is {cmp_} "
              f"{'-' if loss else ''}{m.group('n')}%" + (" (a gain below the threshold includes any loss)."
                                                          if not loss and not more else "."))
        return f"{m.group('k')} {m.group('u')} return is {cmp_} {'-' if loss else ''}{m.group('n')}%"
    text = _sub_outside(r"(?i)\b(?:has |have )?(?P<v>gained|rose|risen|climbed|advanced|lost|fell|fallen|dropped|declined)(?: by)? "
                        r"(?P<q>more than|less than|over|under|at least|at most) (?P<n>\d+(?:\.\d+)?)% (?:over|in|during|within) (?:the )?"
                        r"(?:last |past |prior |previous )?(?P<k>\d+) (?:trading )?(?P<u>day|week|month|year)s?\b", moved, text)

    # "10 day return is down more than 5%" -> "10 day return is less than -5%"
    def updown(m):
        q = m.group("q").lower()
        q = {"over": "more than", "under": "less than"}.get(q, q)
        if m.group("d").lower() == "down":
            q = {"more than": "less than", "less than": "more than", "at least": "at most", "at most": "at least"}[q]
            return f"{m.group('m')} is {q} -{m.group('n')}%"
        return f"{m.group('m')} is {q} {m.group('n')}%"
    text = _sub_outside(r"(?i)\b(?P<m>returns?|performance|change|gain)(?: is| was| has been)? (?P<d>up|down)(?: by)? "
                        r"(?P<q>more than|less than|over|under|at least|at most) (?P<n>\d+(?:\.\d+)?)%", updown, text)
    # "under 10% from its 52 week low" -> "less than 10% above its 52 week low" (and "... below its high")
    text = _sub_outside(r"(?i)\b(?:is )?(?P<q>under|less than|no more than|at most|within) (?P<n>\d+(?:\.\d+)?)% (?:away )?(?:from|off) "
                        r"(?P<w>(?:its |the )?(?:\d+[- ](?:day|week|month|year)s? )?(?P<hl>high|low|peak))\b",
                        lambda m: (f"is {'less than' if m.group('q').lower() in ('under', 'less than') else 'at most'} "
                                   f"{m.group('n')}% {'above' if m.group('hl').lower() == 'low' else 'below'} {m.group('w')}"), text)

    # "SPY's 20 day return is more than 2% higher than TLT's": 2 percentage points (a difference of returns)
    def pp(m):
        rm = (m.group("rm") or "").strip() or m.group("m").strip()
        side = "above" if m.group("d").lower() in ("higher", "greater", "more", "above") else "below"
        q = (m.group("q") or "").strip()
        _note(f"'{m.group(0).strip()}' compares two returns, so {m.group('n')}% is read as {m.group('n')} percentage points "
              f"(a difference of returns, not {m.group('n')}% of {m.group('rt')}'s return).")
        return f"{m.group('lt')}'s {m.group('m').strip()} is {q + ' ' if q else ''}{m.group('n')} points {side} {m.group('rt')}'s {rm}"
    text = _sub_outside(rf"(?i)\b(?P<lt>{_RV_TK})'s (?P<m>(?:[\w-]+ ){{0,4}}?(?:returns?|performance|momentum|gain|change)) is "
                        rf"(?P<q>more than |at least |over |less than |under |at most )?(?P<n>\d+(?:\.\d+)?)% "
                        rf"(?P<d>higher|greater|more|above|lower|less|below) than (?P<rt>{_RV_TK})'s(?P<rm>(?: (?!then\b|and\b|or\b|hold\b)[\w-]+)*)"
                        + _RV_END, pp, text)
    # "... N points above TLT's" with the measure left out: the same measure as the left side
    text = _sub_outside(rf"(?i)\b(?P<lt>{_RV_TK})'s (?P<m>(?:[\w-]+ ){{0,4}}?[\w-]+) is (?P<rel>(?:more than |at least |over |less than |under |at most )?"
                        rf"\d+(?:\.\d+)? points? (?:above|below|higher than|lower than)) (?P<rt>{_RV_TK})'s" + _RV_END,
                        lambda m: f"{m.group('lt')}'s {m.group('m')} is {m.group('rel')} {m.group('rt')}'s {m.group('m')}", text)

    # "TQQQ's 10 day RSI is below SOXL's 10 day RSI minus 5" -> "... is more than 5 points below SOXL's 10 day RSI"
    def offset(m):
        below = m.group("rel").lower() in ("below", "under", "less than", "lower than")
        plus = m.group("op").lower() in ("plus", "+")
        n = m.group("n")
        # a < b - n: b - a > n (more than n below); a > b + n: more than n above; a < b + n: a - b < n (less than n
        # above); a > b - n: b - a < n (less than n below)
        if below and not plus:
            rel = f"more than {n} points below"
        elif not below and plus:
            rel = f"more than {n} points above"
        elif below and plus:
            rel = f"less than {n} points above"
        else:
            rel = f"less than {n} points below"
        _note(f"'{m.group(0).strip()}' = {m.group('lhs').strip()} is {rel} {m.group('rhs').strip()} "
              f"({'<' if below else '>'} {m.group('rhs').strip()} {'+' if plus else '-'} {n}).")
        return f"{m.group('lhs')} is {rel} {m.group('rhs')}"
    text = _sub_outside(rf"(?i)(?P<lhs>(?:\b{_RV_TK}'s |\bthe )(?:[\w-]+ ){{0,4}}?[\w-]+) is (?P<rel>below|under|less than|lower than|above|over|"
                        rf"greater than|higher than) (?P<rhs>(?:\b{_RV_TK}'s |\bthe )(?:[\w-]+ ){{0,6}}?[\w-]+) (?P<op>minus|plus|-|\+) "
                        rf"(?P<n>\d+(?:\.\d+)?)(?![\d.%])" + _RV_END, offset, text)

    # "SPY is oversold" / "overbought": no standard threshold; read as the common 14 day RSI below 30 / above 70
    def osc(m):
        over = m.group(1).lower() == "overbought"
        _note(f"Warning: 'is {m.group(1).lower()}' has no single definition; it was read as the 14 day RSI "
              f"{'above 70' if over else 'below 30'} (the usual RSI zones). Say e.g. '10 day RSI is {'above 80' if over else 'below 20'}' "
              "for another indicator or level.")
        return f" 14 day RSI is {'above 70' if over else 'below 30'}"
    text = _r13_osc_zone(text)
    text = _sub_outside(r"(?i)(?<!rsi)(?<!rsi\))\s+(?:is|are|gets|becomes|turns) (oversold|overbought)\b", osc, text)
    return text


def parse_conditions(text: str, traded: list[str], strict: bool = True, as_list: bool = False, total: bool = False):
    """Parse `x and y or z` style condition text into one expression.

    Each and/or part is bound to the ticker named inside it (e.g. "SPY is above its 200-day moving
    average" -> SPY's series); parts that name no other ticker refer to the traded ticker. With
    total=True (portfolio conditions) returns are total returns (tret).
    """
    _negative_lookbacks(text)
    text = _review_condition_phrases(text)
    # "it is not the case that A and B" / "not (A or B)": the negation of the whole condition
    mneg = re.fullmatch(r"(?is)\s*(?:(?:it is|it's) not (?:the case|true) that|not the case that|it is false that)\s+(.+?)\s*|"
                        r"\s*not\s*\((.+)\)\s*", text)
    if mneg and "`" not in text and (mneg.group(1) or _balanced(mneg.group(2))):
        inner = parse_conditions(mneg.group(1) or mneg.group(2), traded, strict=strict, total=total)
        _note(f"'{text.strip()}' = the opposite of '{(mneg.group(1) or mneg.group(2)).strip()}': not ({inner}).")
        return [f"not ({inner})"] if as_list else f"not ({inner})"
    # "neither A nor B" (two whole conditions) = not A and not B; "neither SPY nor QQQ 10 day RSI ..." (a ticker
    # list) is a ticker group, below
    mnn = _msearch(r"(?is)\s*neither\s+(?P<a>.+?)\s+nor\s+(?P<b>.+?)\s*", text, full=True)
    if mnn and "`" not in text and re.search(r"[a-z]{2}", mnn.group("a")):
        a = parse_conditions(mnn.group("a"), traded, strict=strict, total=total)
        b = parse_conditions(mnn.group("b"), traded, strict=strict, total=total)
        rule = f"not ({a}) and not ({b})"
        _note(f"'{re.sub(chr(92) + 's+', ' ', text.strip())}' = neither holds: {rule}.")
        return [f"not ({a})", f"not ({b})"] if as_list else rule
    text = _sub_outside(r"\b(?:but only if|but only when|only if|only when|provided that|provided|as long as|so long as|while|but)\b", " and ", text)
    text = _sub_outside(r".+", lambda m: _negations(m.group(0)), text)
    text = _motion_verbs(text, total)
    text = _sub_outside(r"(?i)\b(?:both )?(?P<a>[\^$]?[a-z]{1,5}(?:sim)?) and (?P<b>[\^$]?[a-z]{1,5}(?:sim)?) (?:are both|both are|are) "
                        r"(?P<rest>[^,;]+?)(?=$|,|;| and | or )", _both, text)
    # "has a 10 day RSI above 70" / "whose 10 day RSI is above 70" -> "10 day RSI is above 70"
    ind_ahead = rf"(?=(?:\d+ {UNIT_WORDS} )?(?:{PERIOD_IND}|return|volatility|drawdown)\b)"
    text = _sub_outside(rf"(?i)\b(?:(?:it|that|which) )?(?:has|have) (?:a|an) {ind_ahead}", " ", text)
    text = _sub_outside(rf"(?i)\bwith (?:a|an) {ind_ahead}", " and ", text)
    text = _sub_outside(r"(?i),?\s*\bwhose\b", " ", text)   # "QQQ, whose 10 day RSI ...": one clause
    # "is between 30 and 50": keep the 'and' from splitting the clause (expanded below)
    text = _sub_outside(r"(?i)\bbetween (-?\$?\d+(?:\.\d+)?%?) and (-?\$?\d+(?:\.\d+)?%?)(?![\w.])", "between \\1 \x02 \\2", text)
    # "both SPY and QQQ 10 day RSI are above 79", "any of SPY, QQQ and SMH has a 10 day RSI above 79": the comparison
    # for each ticker, joined with and / or (compiled here, so the ticker list is not split up below)
    text, groups = _ticker_groups(text, traded, strict, total)
    bad: list[str] = []

    def clause(o: str) -> str | None:
        """clause1, plus crossings of anything a level comparison can say: "the 20 day return crosses below -5%" is
        read as the comparison "... is below -5%" turned into crossunder(<value>, <level>)."""
        mx = re.search(r"(?i)\bcross(?:es|ed)? (below|under|above|over)\b", o) if "`" not in o else None
        if not mx:
            return clause1(o)
        nb, err = len(bad), None
        try:
            r = clause1(o)
        except ParseError as e:
            r, err = None, e
        if r is not None or (len(bad) == nb and err is None):
            return r
        failed = bad[nb:]
        del bad[nb:]
        level = o[: mx.start()] + "is " + mx.group(1) + o[mx.end():]
        try:
            r2 = clause1(level)
        except ParseError:
            r2 = None
        cr = _as_crossing(r2) if r2 is not None and len(bad) == nb else None
        if cr is None:
            del bad[nb:]
            if err is not None:
                raise err
            bad.extend(failed)
            return None
        return cr

    def clause1(o: str) -> str | None:
        """One comparison (no top-level and / or) -> its rule; None when it is not understood (added to `bad`)."""
        result = None
        ob = o.strip()
        # a bracketed group: "(SPY 10 day RSI is above 70 and QQQ 10 day RSI is above 70) or ...": parsed on its own
        # (every ticker inside keeps its own series), and kept together
        if ob.startswith("(") and ob.endswith(")") and _balanced(ob[1:-1]) and "`" not in ob:
            inner = _unplace(ob[1:-1], groups)
            if inner.strip():
                return f"({parse_conditions(inner, traded, strict=strict, total=total)})"
        mg = re.fullmatch(r"\s*\x03(\d+)\x03\s*", o)
        if mg:
            return groups[int(mg.group(1))][0]
        # "not SPY 10 day RSI is above 79 or ...": 'not' negates the condition right after it only (it binds tighter
        # than 'and' / 'or', as in the rule language); brackets negate several: "not (A or B)"
        mnot = re.fullmatch(r"(?is)\s*(?:(?:and|but|or) )?not\s+(?!\()(?P<c>(?!(?:only|just|when|if)\b).+)", o) if "`" not in o else None
        # (only a comparison on a ticker or an indicator: "not on Fridays" / "not in October" keep their own readings)
        if mnot and re.search(r"[a-z]{2}", mnot.group("c")) and (
                find_tickers(mnot.group("c").split()[0], strict=True)
                or re.match(r"(?i)(?:the |its )?(?:\d+[- ]?(?:day|week|month|bar|period)s? |rsi\b|price\b|close\b|return\b)", mnot.group("c"))):
            inner_t = _unplace(mnot.group("c"), groups)
            inner = parse_conditions(inner_t, traded, strict=strict, total=total)
            inner = inner if inner.startswith("(") and inner.endswith(")") and _balanced(inner[1:-1]) else f"({inner})"
            _note(f"'not {inner_t.strip()}': 'not' applies to the condition right after it only (up to the next 'and' / "
                  f"'or'): not {inner}. Put brackets after 'not' to negate several conditions, e.g. 'not (A or B)'.")
            return f"not {inner}"
        if "\x03" in o:
            raise ParseError(f"'{_unplace(o, groups).strip()}': write the condition on several tickers as its own "
                             "clause, e.g. 'if both SPY and QQQ 10 day RSI are above 79 then ...'.")
        o = _sub_outside(r".+", lambda m: _indicator_periods(m.group(0)), o)
        o = re.sub(r"(?i)^\s*(?:today(?:'s)?|now),?\s+(?=\S)", " ", o)       # "today RSI(2) is above 10"
        myd = re.match(r"(?is)\s*(?:yesterday'?s?|the (?:previous|prior) (?:day|session|bar)'?s?|(?:on )?the (?:day|bar|session) before,?)"
                       r"\s+(?P<rest>(?!(?:high|low|close|open)\b).+)$", o) if "`" not in o else None
        if myd:   # "yesterday's RSI(2) was below 10": the whole comparison on the previous bar
            inner = re.sub(r"(?i)\bwas\b", "is", re.sub(r"(?i)\bwere\b", "are", myd.group("rest")))
            e = parse_conditions(inner, traded, strict=strict, total=total)
            _note(f"'{o.strip()}' = the previous bar's value: ref(({e}), 1).")
            return f"ref(({e}), 1)"
        mentioned = [t for t in find_tickers(o, strict=True) if t not in traded]
        rel = _relative_compare(o, total) if mentioned else None
        if rel:
            return _every_ticker_used(o, rel, traded)
        pair = _pair_compare(o, traded)
        if pair:
            _check_ranges(pair, o)
            return _every_ticker_used(o, pair, traded)
        if len(mentioned) > 1:
            raise ParseError(f"'{o.strip()}' mentions several tickers ({', '.join(mentioned)}); split it into separate conditions")
        ctx = Ctx.for_ticker(mentioned[0] if mentioned else None, total=total)
        o_clean = o
        for t in find_tickers(o):
            o_clean = _sub_outside(rf"(?:\b(?:of|for|on) )?(?<![\w])[\$^]?{re.escape(t.lstrip('^'))}(?:'s)?\b", " ", o_clean, flags=re.I)
        for name, sym in COMPANIES.items():
            o_clean = _sub_outside(rf"(?:\b(?:of|for|on) )?\b{re.escape(name)}(?:'s)?\b", " ", o_clean, flags=re.I)
        o_clean = _sub_outside(r"(?i)\b(?:of|for)\s+it\b", " ", o_clean)   # "the RSI of it" (the traded ticker)
        o_clean = re.sub(r"^\s*the\b", " ", o_clean)
        o_clean = _sub_outside(r"\s{2,}", " ", o_clean)   # the gaps a removed ticker leaves ("20 day SMA  crosses")
        mno = re.fullmatch(r"(?is)\s*not\s*\((.+)\)\s*", o)
        if mno and "`" not in o and _balanced(mno.group(1)):
            return f"not ({parse_conditions(mno.group(1), traded, strict=strict, total=total)})"
        tl = getattr(_TL, "notes", None)
        n0 = len(tl) if tl is not None else 0
        e, left = parse_condition(o_clean, ctx)
        if left or not e:
            # the first reading is dropped when another one understands the whole clause: so are its notes (a
            # partial "10 days RSI" read as RSI(14) must not leave "No RSI period given" behind)
            first = tl[n0:] if tl is not None else []
            if tl is not None:
                del tl[n0:]
            th = _threshold(o_clean, ctx) or _trend(o_clean, ctx) or _over_bars(o, traded, total)
            if th:
                e, left = th, ""
            elif tl is not None:
                tl[n0:n0] = [x for x in first if x not in tl]
        if e:
            if "`" not in o:
                _check_ranges(e, o)
            result = _every_ticker_used(o, e, traded)
        elif not left and find_tickers(o):
            # a bare ticker ("SPY" of "if SPY, QQQ ..."): never drop it silently
            raise ParseError(f"'{o.strip()}': no condition was found for {', '.join(find_tickers(o))} here. For a "
                             "condition on several tickers write 'both SPY and QQQ ...' / 'all of ...' (every one) or "
                             "'either SPY or QQQ ...' / 'any of ...' (at least one).")
        if left and (strict or not e):
            words = set(re.findall(r"[a-z%']+|\d+(?:\.\d+)?", o.lower()))
            unknown = [w for w in left.split() if w in words]
            bad.append(o.strip() + (f"' (not understood: '{' '.join(unknown)}')" if e and unknown else "'"))
        return result

    # 'and' binds tighter than 'or' (as in the rule language and most tools): split at the top-level 'or' first,
    # then each alternative at 'and' / ',' / ';'. Brackets group ("(A or B) and C").
    or_parts = _or_subjects(_split_top(text, "or"))
    alts: list[list[str]] = []
    alt_texts: list[list[str]] = []
    for alt in or_parts:
        if not alt.strip():
            continue
        exprs_a, texts_a = [], []
        for part in _expand_ranges(_msplit(alt, r"\band\b|,|;|\bwith\b(?! a)", flags=0)):
            if not part.strip():
                continue
            e = clause(part)
            if e:
                exprs_a.append(e)
                texts_a.append(re.sub(r"\s+", " ", _unplace(part, groups).replace("\x02", "and")).strip())
        if exprs_a:
            alts.append(exprs_a)
            alt_texts.append(texts_a)
    if len(alts) <= 1:
        exprs = alts[0] if alts else []
    else:
        joined = [a[0] if len(a) == 1 else "(" + " and ".join(a) + ")" for a in alts]
        exprs = ["(" + " or ".join(joined) + ")"]
        if any(len(a) > 1 for a in alts) and not bad:
            read = " or ".join(t[0] if len(t) == 1 else "(" + " and ".join(t) + ")" for t in alt_texts)
            _note(f"'and' and 'or' without brackets: read as {read} ('and' binds tighter than 'or'); use brackets to group "
                  "the conditions differently.")
    for b in bad:
        mx = re.search(r"(?i)\b(cross(?:es|ed|ing)?)\b(?! (?:above|below|over|under|back|up|down)\b)\s*(?P<what>[^']*)", b)
        if mx and "`" not in b:
            what = mx.group("what").strip() or "it"
            raise ParseError(f"'{b.split(chr(39))[0].strip()}': crosses which way? Say 'crosses above {what}' (from below to "
                             f"above) or 'crosses below {what}' (from above to below). For a cross in either direction write "
                             "the rule in backticks, e.g. `cross(rsi(2), 70)`.")
    if bad:
        raise ParseError(
            "Could not interpret: " + "; ".join("'" + b for b in bad)
            + ".\nRephrase, or write that part in the rule language inside backticks, e.g. "
              "`rsi(2) < 10 and close > sma(close, 200)` (see --help-expr)."
        )
    if not exprs:
        raise ParseError(f"No conditions found in {text.strip()!r}")
    return exprs if as_list else " and ".join(exprs)


CMP_WORDS = [
    (r"(?:is |are )?(?:greater than or equal to|at least|>=)", ">="),
    (r"(?:is |are )?(?:less than or equal to|at most|<=)", "<="),
    (r"(?:is |are )?(?:greater than|higher than|above|more than|over|exceeds?|beats?|outperforms?|>)", ">"),
    (r"(?:is |are )?(?:less than|lower than|below|under|underperforms?|<)", "<"),
]


def _side(text: str, traded: list[str]) -> str | None:
    """"QQQ's 10 day RSI" / "the 10 day RSI of QQQ" / "QQQ 10 day RSI" -> expression on that ticker."""
    t = text.strip()
    tk = find_tickers(t, strict=True)
    if len(tk) != 1:
        return None
    rest = re.sub(rf"(?<![\w])[\$^]?{re.escape(tk[0].lstrip('^'))}(?:'s)?\b", " ", t, flags=re.I)
    for name, sym in COMPANIES.items():
        if sym == tk[0]:
            rest = re.sub(rf"\b{re.escape(name)}(?:'s)?\b", " ", rest, flags=re.I)
    rest = re.sub(r"(?i)\b(?:of|for|on)\s*$", " ", rest.strip())
    rest = re.sub(r"(?i)^\s*(?:the|its)\b", " ", rest).strip()
    if not rest:
        rest = "price"
    try:
        e, _ = value_phrase(rest, Ctx() if tk[0] in traded else Ctx.for_ticker(tk[0]), default_n=None)
    except ParseError:
        return None
    return e


def _pair_compare(text: str, traded: list[str]) -> str | None:
    """Two-ticker comparison, e.g. 'QQQ 10 day RSI is greater than SPY 10 day RSI' or
    'the 3 month return of QQQ beats the 3 month return of TLT'."""
    if len(find_tickers(text, strict=True)) != 2 or "`" in text:
        return None
    # "QQQ (has) outperformed TLT over 20 days" -> total return of QQQ over 20 days > TLT's
    mo = re.fullmatch(r"(?i)\s*(?P<a>\S+?)\s+(?:has |have )?(?:been )?(?P<v>outperform(?:ed|s|ing)?|beat(?:en|s)?|"
                      r"underperform(?:ed|s|ing)?|lagged|lags?|trailed|trails?)\s+(?P<b>\S+?)\s+(?:over|in|during|for|across) "
                      r"(?:the )?(?:last |past |prior |previous )?(?P<n>\d+) (?P<u>day|week|month|year|session|bar)s?\s*", text)
    if mo and len(find_tickers(mo.group("a"), strict=True)) == 1 and len(find_tickers(mo.group("b"), strict=True)) == 1:
        per = f"{mo.group('n')} {mo.group('u')} return"
        ea, eb = _side(f"{mo.group('a')} {per}", traded), _side(f"{mo.group('b')} {per}", traded)
        if ea and eb:
            op = "<" if re.match(r"(?i)under|lag|trail", mo.group("v")) else ">"
            return f"{ea} {op} {eb}"
    # "the RSI of SPY is 10 points above the RSI of QQQ": an absolute difference (for oscillators; for returns,
    # percentage points)
    mp = re.search(r"(?i)\s(?:is |are )?(?P<q>at least |more than |over |no less than |less than |under |at most |no more than )?(?P<n>\d+(?:\.\d+)?) "
                   r"(?:points?|pts?)(?: or more)? (?P<rel>above|below|higher than|lower than|over|under)\s", f" {text} ")
    if mp:
        a, b = f" {text} "[: mp.start()], f" {text} "[mp.end():]
        if len(find_tickers(a, strict=True)) == 1 and len(find_tickers(b, strict=True)) == 1:
            ea, eb = _side(a, traded), _side(b, traded)
            if ea and eb:
                n = float(mp.group("n"))
                if _fraction_valued(ea) and _fraction_valued(eb):
                    n /= 100
                    _note(f"'{mp.group(0).strip()}': percentage points of return ({n:g}).")
                up = mp.group("rel").lower() in ("above", "higher than", "over")
                # "less than 5 points below": the difference is under 5 (a < b + 5 for 'above', b - a < 5 for 'below')
                less = (mp.group("q") or "").strip().lower() in ("less than", "under", "at most", "no more than")
                bound = _bound("<" if less else ">", mp.group(0))
                diff = f"{ea} - {eb}" if up else f"{eb} - {ea}"
                return f"{diff} {bound} {n:g}"
    for pat, op in CMP_WORDS:
        m = re.search(rf"(?i)\s{pat}\s", f" {text} ")
        if not m:
            continue
        a, b = f" {text} "[: m.start()], f" {text} "[m.end():]
        if len(find_tickers(a, strict=True)) != 1 or len(find_tickers(b, strict=True)) != 1:
            return None
        ea = _side(a, traded)
        if ea is None:
            return None
        # "QQQ's RSI is above SPY's" -> same indicator on the other side
        tb = find_tickers(b, strict=True)[0]
        b_only = re.sub(rf"(?i)(?<![\w])[\$^]?{re.escape(tb.lstrip('^'))}(?:'s)?\b|\bthat of\b|\bthe\b", " ", b).strip()
        if not b_only:
            a_rest = re.sub(rf"(?i)(?<![\w])[\$^]?{re.escape(find_tickers(a, strict=True)[0].lstrip('^'))}(?:'s)?\b", " ", a)
            b = f"{tb} {a_rest}"
        eb = _side(b, traded)
        if eb is None:
            return None
        return f"{ea} {op} {eb}"
    return None


REL_CMP = [
    (r"(?:is |are )?(?:greater than or equal to|at least|no less than|>=)", ">="),
    (r"(?:is |are )?(?:less than or equal to|at most|no more than|<=)", "<="),
    (r"(?:is |are |has |have )?(?:greater than|higher than|above|more than|over|exceeds?|exceeded|beats?|beaten|"
     r"outperforms?|outperformed|better than|>)", ">"),
    (r"(?:is |are |has |have )?(?:less than|lower than|below|under|underperforms?|underperformed|worse than|<)", "<"),
]
PRONOUN_SUBJECT = (r"(?:its|their|the (?:selected|chosen) (?:assets?|stocks?|ones?|funds?)'?s?|"
                   r"each (?:one|asset|candidate|fund)'?s?)")


def _relative_compare(text: str, total: bool = True) -> str | None:
    """A relative hurdle: '<its / their> <indicator> <comparison> <X>'s [<indicator>]', e.g. 'their 12 month return
    is above BIL's 12 month return' -> tret(tr, 252) > tret(sym("BIL").tr, 252). The left side is the asset the
    condition belongs to (the traded ticker, an if-node's holding, or each candidate of a filter); the right side
    is ticker X (via sym()). 'above BIL' alone compares the same indicator. None when the text is not of that shape
    (no pronoun subject, or no other ticker on the right); a ParseError when it is but a side is not understood."""
    t = re.sub(r"\s+", " ", text.strip())
    m0 = re.fullmatch(rf"(?i){PRONOUN_SUBJECT}\s+(.+)", t)
    if not m0 or "`" in t:
        return None
    body = f" {m0.group(1)} "
    for pat, op in REL_CMP:
        m = re.search(rf"(?i)\s{pat}\s", body)
        if not m:
            continue
        lhs, rhs = body[: m.start()].strip(), body[m.end():].strip()
        tk = find_tickers(rhs, strict=True)
        if not lhs or find_tickers(lhs, strict=True) or not tk:
            return None
        if len(tk) > 1:
            raise ParseError(f"'{t}': compare with one ticker at a time (found {', '.join(tk)}).")
        x = tk[0]
        rest = re.sub(rf"(?i)(?<![\w])[\$^]?{re.escape(x.lstrip('^'))}(?:'s|')?(?![\w])", " ", rhs)
        for name, sym in COMPANIES.items():
            if sym == x:
                rest = re.sub(rf"(?i)\b{re.escape(name)}(?:'s)?\b", " ", rest)
        rest = re.sub(r"(?i)\b(?:that|those) of\b|\bthe\b|\bof\b|\bfor\b|\bits\b", " ", rest)
        rest = re.sub(r"\s+", " ", rest).strip()
        try:
            le, ln = value_phrase(lhs, Ctx(total=total), default_n=None, total=total)
            re_, rn = value_phrase(rest or lhs, Ctx.for_ticker(x, total=total), default_n=None, total=total)
        except ParseError as e:
            raise ParseError(f"'{t}': a comparison with {x} was recognised, but: {e}") from None
        for n_ in ln + rn:
            _note(n_)
        rule = f"{le} {op} {re_}"
        _note(f"'{t}' is a relative hurdle: the asset's own {lhs.strip()} compared with {x}'s ({rule}).")
        return rule
    return None


def _tautologies(rule: str) -> list[str]:
    """Comparisons in a rule whose two sides are textually identical (always true or always false)."""
    try:
        tree = ast.parse(rule.strip(), mode="eval")
    except SyntaxError:
        return []
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Compare):
            sides = [n.left] + list(n.comparators)
            for a, b in zip(sides, sides[1:]):
                if ast.unparse(a) == ast.unparse(b):
                    out.append(f"{ast.unparse(a)} vs {ast.unparse(b)}")
    return out


def split_and(rule: str) -> list[str]:
    """Top-level AND terms of a rule expression."""
    tree = ast.parse(rule.strip(), mode="eval").body
    out = []

    def walk(n):
        if isinstance(n, ast.BoolOp) and isinstance(n.op, ast.And):
            for v in n.values:
                walk(v)
        elif isinstance(n, ast.BinOp) and isinstance(n.op, ast.BitAnd):
            walk(n.left)
            walk(n.right)
        else:
            out.append(ast.unparse(n))
    walk(tree)
    return out


def _is_price(a: str) -> bool:
    return a == "close" or re.fullmatch(r"sym\((['\"])[^'\"]+\1\)\.close", a) is not None


def _flip(entry: str, below: bool, series: str | None = None, price_only: bool = False) -> str | None:
    """'sell when it crosses back below' -> the entry's comparison, reversed.

    `series` ('close' or 'sym("SPY").close') restricts it to comparisons of that ticker's price.
    Otherwise a comparison of the price is preferred; failing that (unless `price_only`), the entry's
    only comparison (e.g. 'the 9 EMA crosses above the 21 EMA' or 'RSI(2) crosses below 10').
    Returns None when nothing, or more than one thing, could be meant."""
    cands = []
    for term in split_and(entry):
        n = ast.parse(term, mode="eval").body
        if isinstance(n, ast.Compare) and len(n.ops) == 1 and isinstance(n.ops[0], (ast.Gt, ast.GtE, ast.Lt, ast.LtE)):
            a, b = ast.unparse(n.left), ast.unparse(n.comparators[0])
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ("crossover", "crossunder") and len(n.args) == 2:
            a, b = ast.unparse(n.args[0]), ast.unparse(n.args[1])
        else:
            continue
        cands.append((a, b))
    if series is not None:
        want = ast.unparse(ast.parse(series, mode="eval").body)
        cands = [(a, b) for a, b in cands if a == want]
    else:
        price = [(a, b) for a, b in cands if _is_price(a)]
        cands = price if (price or price_only) else cands
    if len(cands) != 1:
        return None
    a, b = cands[0]
    return f"{a} {'<' if below else '>'} {b}"


_CALL_NAMES = ("rsi", "stoch_k", "cci", "willr", "mfi", "adx")


def _entry_indicators(entry: str) -> list[str]:
    """The indicators (not prices) the entry compares with a number, e.g. ['macd_hist()'] for
    crossover(macd_hist(), 0). Empty when the entry also compares a price (then 'it' is the price)."""
    out, price = [], False
    for term in split_and(entry):
        n = ast.parse(term, mode="eval").body
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ("crossover", "crossunder") and len(n.args) == 2:
            a, b = n.args
        elif isinstance(n, ast.Compare) and len(n.ops) == 1:
            a, b = n.left, n.comparators[0]
        else:
            continue
        sa = ast.unparse(a)
        if _is_price(sa) or _is_price(ast.unparse(b)):
            price = True
            continue
        if isinstance(a, ast.Call) and isinstance(b, (ast.Constant, ast.UnaryOp)) and sa not in out:
            out.append(sa)
    return [] if price else out


_PRICE_NAMES = {"close", "open", "high", "low", "price", "down_days", "up_days", "gap", "change", "ibs", "range"}


def _entry_subjects(entry: str) -> list[str]:
    """What each top-level term of the entry is about, in order: 'close' for the price (a price comparison,
    down days, a gap...), otherwise the series compared or crossed, e.g. 'sma(close, 50)' for
    sma(close, 50) > ref(sma(close, 50), 1). This is what a pronoun in the exit ('sell when it is falling')
    refers back to."""
    out: list[str] = []
    try:
        terms = split_and(entry)
    except SyntaxError:
        return out
    for term in terms:
        n = ast.parse(term, mode="eval").body
        if isinstance(n, ast.Compare) and len(n.ops) == 1:
            a, b = n.left, n.comparators[0]
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ("crossover", "crossunder") and len(n.args) == 2:
            a, b = n.args
        else:
            continue
        sa, sb = ast.unparse(a), ast.unparse(b)
        if _is_price(sa) or (isinstance(a, ast.Name) and a.id in _PRICE_NAMES):
            subj = sa if _is_price(sa) else "close"
        elif sb == f"ref({sa}, 1)" or isinstance(a, ast.Call):
            subj = sa
        elif _is_price(sb):
            subj = sb
        else:
            continue
        if subj not in out:
            out.append(subj)
    return out


def _calls(rule: str, names=_CALL_NAMES) -> list[str]:
    """Distinct calls of the given functions in a rule, e.g. ['rsi(close, 2)']."""
    out = []
    for m in re.finditer(rf"\b(?:{'|'.join(names)})\(", rule):
        depth, j = 0, m.end() - 1
        while j < len(rule):
            depth += {"(": 1, ")": -1}.get(rule[j], 0)
            if depth == 0:
                break
            j += 1
        call = rule[m.start(): j + 1]
        if call not in out:
            out.append(call)
    return out


# ----------------------------------------------------------------- trailing-stop activation

# "once (it is) up 3%", "after the trade is up 1R", "when in profit by 2 ATR": how far the best price must move in
# favour before a trailing stop applies. Group "an" is the number, "u" its unit (%, R, ATR or points).
_ACT_WHEN = (r"(?:once|after|when|as soon as|if|only once|only after)(?: (?:it(?:'s| is| has| gets)?|the (?:trade|position|price|stock)"
             r"(?:'s| is| has| gets)?|price is|we are|we're|i'm|i am|the gain is|the profit is|profit is|gains? (?:reach|reaches|hit|hits)))?"
             r"(?: (?:up|gained|risen|rallied|moved up|moved|in profit|ahead|in the money|showing a (?:gain|profit) of))?(?: (?:by|at least|of))?")
_ACT_AMT = (r" \+?(?:\$(?P<an2>\d+(?:\.\d+)?)|(?P<an>\d+(?:\.\d+)?) ?(?P<u>%|r\b|x (?:the )?(?:initial )?risk\b|times (?:the )?(?:initial )?risk\b|atrs?\b|"
            r"points?\b|pts?\b|dollars?\b))(?: (?:in profit|gain|profit|up|higher|in (?:my |our |its |the trade's )?favou?r))?")
_ACT_VERB = r"(?:(?:which |that |and )?(?:only )?(?:activat(?:es|ed|ing)|starts?(?: trailing)?|kicks? in|arms?|armed|begins?|turns? on)(?: only)?)"


def _trail_activation(T: "Text", ex: dict, notes: list[str]) -> None:
    """A trailing stop that only applies once the trade is up by some amount: "trailing stop 1% once up 3%", "10%
    trailing stop once it is up 5%", "3 ATR trailing stop after 1R", "activate the trail after 1R". The activation
    clause is consumed here, so it can never be read as part of the entry condition."""
    trail = (r"(?:(?:with|use|and|add|place|plus) )?(?:a |an |the )?(?:(?:trailing|chandelier) stop(?:[- ]loss)?(?: of| at)? "
             r"\d+(?:\.\d+)?(?:%| ?(?:x )?atrs?\b)|\d+(?:\.\d+)?(?:%| ?(?:x )?atr) (?:trailing|chandelier) stop(?:[- ]loss)?)")
    pats = [rf"(?P<trail>{trail}),? (?:{_ACT_VERB} )?(?:{_ACT_WHEN}|at){_ACT_AMT}",
            # "activate the trail after 1R", "start trailing once up 3%", "the trailing stop activates at +3%"
            rf"(?:(?:and|then|with|plus) )?(?:(?:only )?(?:activate|arm|start|begin|turn on) (?:the |my |a )?(?:trail(?:ing stop(?:[- ]loss)?)?|trailing)"
            rf"|(?:the |my )?(?:trail(?:ing stop(?:[- ]loss)?)?) {_ACT_VERB}) (?:{_ACT_WHEN}|at){_ACT_AMT}"]
    for k, pat in enumerate(pats):
        m = T.find(pat, consume=False)
        if not m:
            continue
        if ex.get("trail_activation") or ex.get("trail_activation_points") or ex.get("trail_activation_r"):
            raise ParseError(f"The trailing stop's activation is given twice ('{m.group(0).strip()}'); keep one.")
        keep = m.group("trail") if k == 0 else ""
        T.rest = T.rest[: m.start()] + " ; " + keep + " ; " + T.rest[m.end():]
        v, u = (float(m.group("an2")), "points") if m.group("an2") else (float(m.group("an")), m.group("u").lower())
        if v <= 0:
            raise ParseError(f"'{m.group(0).strip()}': the activation distance must be above 0.")
        if u == "%":
            ex["trail_activation"] = v / 100
            what = f"the best price since entry is {v:g}% in favour"
        elif u.startswith(("r", "x", "times")):
            ex["trail_activation_r"] = v
            what = f"the best price since entry is {v:g}R in favour (R = the initial risk, entry price - initial stop)"
        elif u.startswith("atr"):
            raise ParseError(f"'{m.group(0).strip()}': a trail activation in ATRs is not supported; give it as a percentage "
                             "('once up 3%') or an R multiple ('after 1R').")
        else:
            ex["trail_activation_points"] = v
            what = f"the best price since entry is {v:g} points (dollars) in favour"
        notes.append(f"Trail activation: the trailing stop applies only once {what} (from a bar's high, or low for a short); "
                     "until then only the other stops protect the trade. It then trails the best price since entry.")
        return


_EXIT_WORDS = r"\b(?:stops?|stop[- ]loss|trail\w*|targets?|take[- ]profits?|profit target|break[- ]?even|chandelier)\b"


def _exit_tail_guard(before: str, after: str) -> None:
    """Exit clauses are never absorbed into the entry condition: when an exit phrase (stop / trail / target /
    breakeven) was read but the words right after it in the same clause talk about the trade's progress ("... once up
    3%", "... when it is 5% in profit"), refuse instead of letting them become an entry rule."""
    import difflib
    sm = difflib.SequenceMatcher(None, before, after, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag not in ("replace", "delete") or not re.search(_EXIT_WORDS, before[i1:i2], re.I):
            continue
        tail = re.split(r"[,;.]", after[j2:].lstrip(" ;"), maxsplit=1)[0].strip()
        if not tail or re.search(r"(?i)\bin a row\b|\b(?:days?|bars?|sessions?|weeks?|months?)\b", tail):
            continue
        if re.match(r"(?i)(?:(?:which|that|and) )?(?:only )?(?:activat\w*|kicks? in|arm(?:s|ed)?|starts?|begins?|once|after|"
                    r"as soon as|when|if|until|unless)\b", tail) and re.search(
                r"(?i)(?:\bup|\bgained|\brisen|\brallied|\bahead|\bin profit|\bprofit of|\bgain of)(?: by| at least)? \+?\d+(?:\.\d+)? ?(?:%|r\b)"
                r"|\d+(?:\.\d+)? ?(?:%|r\b) (?:in profit|gain|profit|up|in (?:my |our |its )?favou?r)", tail):
            a = max(before.rfind(",", 0, i1), before.rfind(";", 0, i1)) + 1
            phrase = (before[a:i2].strip(" ;,") + " " + tail).strip()
            raise ParseError(f"'{phrase}': the part '{tail}' belongs to the exit, not to the entry condition, and it was not "
                             "understood. Write e.g. 'trailing stop 1% once up 3%', 'move the stop to breakeven after +2%', "
                             "or put the exit rule in backticks with 'sell when `...`'.")


# ----------------------------------------------------------------- shared options

class Text:
    """Normalised sentence with consumption tracking: everything recognised is blanked out, and
    whatever is left at the end must be filler words, otherwise the parse is refused."""

    def __init__(self, t: str):
        self.t = t
        self.low = t.lower()
        self.rest = t          # original casing is kept so tickers stay recognisable

    def find(self, pattern: str, consume: bool = True):
        m = re.search(pattern, _mask(self.rest, parens=False), flags=re.I)   # never inside `backticks`
        m = _MM(m, self.rest) if m else None
        if m and consume:
            self.rest = self.rest[: m.start()] + " ; " + self.rest[m.end():]
        return m

    def findall(self, pattern: str):
        out = []
        while True:
            m = self.find(pattern)
            if not m:
                return out
            out.append(m)

    def blank(self, fragment: str) -> None:
        i = self.rest.lower().find(fragment.lower())
        if i >= 0:
            self.rest = self.rest[:i] + " ; " + self.rest[i + len(fragment):]

    def leftovers(self) -> list[str]:
        words = re.findall(r"[a-z%$]+[a-z'%]*|\d+(?:\.\d+)?", re.sub(r"`[^`]*`", " ", self.rest.lower()))
        return [w for w in words if w not in STOP]


def _broker_costs(T: Text, notes: list[str], sep: str = "") -> dict:
    """Broker fee presets ("IBKR commissions", "IBKR tiered") and volume-based slippage, read before the generic
    cost phrases (signal strategies and allocation portfolios alike). `sep` lets an allocation sentence's comma go
    with the phrase."""
    broker: dict = {}
    m = T.find(sep + r"(?:(?:with|using|and|at) )?(?:ibkr|interactive brokers?)(?: pro)?(?: \(?(fixed|tiered)\)?)?"
               r"(?: (?:pricing|commissions?|fees|rates?|commission (?:schedule|model|plan)|costs?))*(?: \((fixed|tiered)\))?")
    if m:
        tier = (m.group(1) or m.group(2) or "fixed").lower()
        broker["commission_model"] = f"ibkr_{tier}"
        if not (m.group(1) or m.group(2)):
            notes.append("IBKR commissions: using IBKR Pro Fixed pricing ($0.005/share, min $1, max 1% of the trade). "
                         "Say 'IBKR tiered' for the tiered schedule.")
    m = T.find(sep + r"(?:(?:(?:with|using|and) )?(?:(?:volume|liquidity)[- ](?:based|dependent|adjusted|aware)|square[- ]root)(?: (?:slippage|market impact|impact))+(?: model)?"
               r"|(?:(?:with|using|and) )?(?:market )?impact(?: model)? slippage|(?:(?:with|using|and) )?(?:a )?market impact(?: model| costs?)?)")
    if m:
        broker["slippage_model"] = "volume"
        notes.append("Volume-based slippage: each fill pays half the spread (2 bps default) plus 100 bps x sqrt(order shares / "
                     "20-day average volume), on top of any fixed slippage.")
    return broker


MONTH_RX = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|"
            r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?")


def _date_bound(m, end: bool) -> str:
    """A date match of common_options -> 'YYYY-MM-DD': a year is its first (last) day, a month its first (last) day."""
    import calendar as _calmod
    g = m.groupdict()
    if g.get("mon"):
        y, mo, d = int(g["my"]), MONTHS.index(next(x for x in MONTHS if x.startswith(g["mon"].lower().rstrip(".")[:3]))) + 1, None
    elif g.get("mn"):
        y, mo, d = int(g["my2"]), int(g["mn"]), None
    else:
        y, mo, d = int(g["y"]), int(g["m"]) if g.get("m") else None, int(g["d"]) if g.get("d") else None
    if mo is not None and not 1 <= mo <= 12:
        raise ParseError(f"'{m.group(0).strip()}': there is no month {mo}.")
    if mo is None:
        return f"{y}-12-31" if end else f"{y}-01-01"
    if d is None:
        d = _calmod.monthrange(y, mo)[1] if end else 1
    return f"{y}-{mo:02d}-{d:02d}"


def common_options(T: Text, notes: list[str]) -> dict:
    kw: dict = {}
    m = T.find(r"(?:start(?:ing)? with|capital of|initial capital of|account of|begin(?:ning)? with|with an? (?:initial |starting )?(?:balance|investment) of|with) \$(\d+(?:\.\d+)?)(?! (?:per|a|each|every|monthly|quarterly|yearly|annually))(?:(?: of)? (?:capital|in capital))?")
    if m:
        kw["capital"] = float(m.group(1))
    m = T.find(rf"(?:(?:with |and )?{NUM} ?(?:bps|basis points?)(?: of)? slippage|slippage(?: of)? {NUM} ?(?:bps|basis points?)|(?:with |and )?{NUM}% slippage|slippage(?: of|:|=)? {NUM}%)(?: per side| each way| per trade)?")
    if m:
        g = m.groups()
        kw["slippage_bps"] = float(g[0] or g[1]) if (g[0] or g[1]) else float(g[2] or g[3]) * 100
    m = T.find(rf"(?:(?:with |and )?\$\s?{NUM} (?:per trade|commissions?|per order|a trade|an order)(?: commissions?)?|(?:(?:with |and )?(?:a )?)?commissions?(?: of|:)? \$\s?{NUM}(?![\d.])(?: (?:per|a|an|each) (?:trade|order))?(?! per share| a share))")
    if m:
        kw["commission"] = float(m.group(1) or m.group(2))
    m = T.find(rf"(?:with |and )?\$\s?{NUM} (?:per share|a share)(?: commissions?)?|commissions?(?: of|:)? \$\s?{NUM} (?:per|a) share")
    if m:
        kw["commission_per_share"] = float(m.group(1) or m.group(2))
    m = T.find(rf"(?:with |and )?{NUM}% commissions?(?: per (?:trade|order|side))?"
               rf"|(?:(?:with |and )?(?:a )?)?commissions?(?: of|:|=)? {NUM}%(?: per (?:trade|order|side))?")
    if m:
        kw["commission_pct"] = float(m.group(1) or m.group(2)) / 100
    m = T.find(r"slippage(?: of)? \d+(?:\.\d+)? ticks?|\d+(?:\.\d+)? ticks?(?: of)? slippage", consume=False)
    if m:
        raise ParseError(f"'{m.group(0)}': slippage in ticks ($0.01 a share) is not supported; slippage is a percentage of the "
                         "price, in basis points. At $100 a share one tick is 1 bps, at $500 it is 0.2 bps: write e.g. '1 bps slippage'.")
    for word in ("slippage", "commission"):
        if re.search(rf"\b{word}", T.rest, re.I):
            raise ParseError(f"Could not understand the {word} amount. Write e.g. '5 bps slippage', '0.1% slippage', "
                             f"'$1 per trade commission', '$0.005 per share commission' or '0.1% commission'.")
    # dates
    # "2010", "2010-03-15", "2010-03" (a month) or "March 2010" / "Mar 2010" / "Mar. 2010" / "3/2010"
    when = rf"(?:(?P<mon>{MONTH_RX}) (?P<my>\d{{4}})|(?P<mn>\d{{1,2}})/(?P<my2>\d{{4}})|(?P<y>\d{{4}})(?:-(?P<m>\d{{2}})(?:-(?P<d>\d{{2}}))?)?)(?![\d/-])"
    mfrom = T.find(r"(?:from|since|starting(?: in)?|beginning(?: in)?|between|after) " + when)
    if mfrom:
        kw["start"] = _date_bound(mfrom, end=False)
    mto = T.find(r"(?:to|until|through|thru|before|and|ending(?: in)?) " + when + r"\b")
    if mto:
        kw["end"] = _date_bound(mto, end=True)
    m = T.find(r"(?:in|during|for) (\d{4})(?! \w*%)\b")
    if m and "start" not in kw:
        kw["start"], kw["end"] = f"{m.group(1)}-01-01", f"{m.group(1)}-12-31"
    # cash interest
    if T.find(r"(?:idle )?cash (?:earns|pays|yields) (?:nothing|no interest|0%)|no interest on cash|without (?:cash )?interest"):
        kw["cash_rate"] = None
    m = T.find(rf"(?:idle )?cash (?:earns|pays|yields) {NUM}%(?: (?:a|per) year| annually)?")
    if m:
        kw["cash_rate"] = float(m.group(1)) / 100
    if "cash_rate" not in kw and T.find(r"(?:(?:with|and|plus) )?(?:t-?bill )?interest on (?:idle |uninvested )?cash|(?:with|plus) (?:t-?bill |cash )?interest\b"
                                        r"|(?:idle )?cash earns (?:the )?(?:3[- ]month )?(?:t-?bills?|treasury bills?|interest)(?: rate)?"):
        kw["cash_rate"] = "tbill"
    # benchmark: a blend ("vs 60/40 SPY/AGG", "benchmark 60% SPY and 40% AGG"), or one ticker; imported (custom)
    # series by name too ("vs MYFUNDX", "vs 60% MYFUNDX 40% VBMFX"), whatever their length
    CT = _custom_names_rx()
    BT = (rf"(?:{CT}|" if CT else "(?:") + r"[\^$]?[a-z]{1,5}(?:sim)?(?:-usd)?)"
    m = T.find(r"\b(?:compared? (?:it )?(?:to|with|against)|benchmark(?:ed)?(?: it)?(?: (?:to|against|with))?|versus|vs\.?|against) "
               r"(?:an? |the )?(?:(?P<ws>\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)+) (?P<ts>" + BT + r"(?:/" + BT + r")+)"
               r"|(?P<list>\d+(?:\.\d+)?% " + BT + r"(?:(?:,? and |, ?| ?/ ?| plus | )\d+(?:\.\d+)?% " + BT + r")+))"
               r"(?: blend| mix| portfolio)?"
               # its own schedule: "vs 60/40 SPY/AGG rebalanced monthly" (else the portfolio's, or monthly)
               r"(?:,? (?:(?:re-?balanced|re-?balancing) (?P<brb>daily|weekly|monthly|quarterly|semi-?annually|annually|yearly)"
               r"|(?P<bnone>never re-?balanced|not re-?balanced|buy and hold|buy-and-hold)))?(?![\w-])")
    if m:
        if m.group("brb"):
            w_ = m.group("brb").lower().replace("-", "")
            kw["benchmark_rebalance"] = {"annually": "yearly", "semiannually": "semiannual"}.get(w_, w_)
        elif m.group("bnone"):
            kw["benchmark_rebalance"] = "none"
        if m.group("ws"):
            ws, ts = m.group("ws").split("/"), m.group("ts").split("/")
            if len(ws) != len(ts):
                raise ParseError(f"'{m.group(0).strip()}': {len(ws)} weights but {len(ts)} tickers.")
            pairs = list(zip(ts, ws))
        else:
            pairs = [(t, w) for w, t in re.findall(r"(\d+(?:\.\d+)?)% (" + BT + r")(?![\w-])", m.group("list"), re.I)]
        pairs = [(data.canonical(t), float(w)) for t, w in pairs]
        for t, _ in pairs:
            if t not in _known():
                raise ParseError(f"No price data for benchmark {t}")
        if abs(sum(w for _, w in pairs) - 100) > 1e-6:
            raise ParseError(f"'{m.group(0).strip()}': the benchmark weights add up to {sum(w for _, w in pairs):g}%, not 100%.")
        kw["benchmark"] = " ".join(f"{w:g} {t}" for t, w in pairs)
        notes.append(f"Benchmark: a blend of {' / '.join(f'{w:g}% {t}' for t, w in pairs)}, total returns"
                     + {None: "", "none": ", never rebalanced (buy and hold)"}.get(
                         kw.get("benchmark_rebalance"), f", rebalanced {kw.get('benchmark_rebalance')}") + ".")
    m = None if "benchmark" in kw else T.find(r"\b(?:compared? (?:it )?(?:to|with|against)|benchmark(?:ed)?(?: it)?(?: (?:to|against))?|versus|vs\.?|against) "
               r"(?!(?:t-?bills?|cash|treasury bills|the risk[- ]free rate)\b)(" + BT + r")(?![\w-])")
    if m:
        b = data.canonical(m.group(1))
        if b not in _known():
            raise ParseError(f"No price data for benchmark {b}")
        kw["benchmark"] = b
    # survivorship
    if T.find(r"(?:using |with )?(?:only )?(?:today's|current) (?:index )?members(?: only)?|ignore (?:index )?membership(?: history)?|without point[- ]in[- ]time(?: membership)?"):
        kw["point_in_time"] = False
    T.find(r"(?:using |with )?point[- ]in[- ]time(?: index)? membership")
    return kw


# ----------------------------------------------------------------- dispatch

ALLOC_HINT = re.compile(
    r"\b(?:rebalanc\w*|buy and hold|buy-and-hold|equal[- ]weight\w*|inverse[- ]volatility|market[- ]cap weight\w*|"
    r"risk parity|(?:top|bottom|best|worst) \d+(?![\d.%]|\s*%)|rotat\w+|dual momentum|otherwise hold|else hold|"
    r"(?:else|otherwise),? (?:buy|hold|be in|own|switch to|go to) (?:[a-z^]{1,5}|cash)\b(?! (?:at|when|if|on)\b)|"
    r"min(?:imum)?[- ]variance|max(?:imum)?[- ](?:sharpe|diversification)[- ]weight|whichever of|\d+ (?:best|worst)[- ]perform|"
    r"(?:\d+ )?(?:least|most) volatile(?: \d+)? (?:of|among|from)|(?:best|worst)[- ]performing \d+ (?:of|among|from)|"
    r"allocat\w+|portfolio of|contribut\w+|withdraw\w*|\d+/\d+ (?:[a-z]+/[a-z]+)|"
    r"equally(?:[- ]weighted)?\b|evenly\b|in equal (?:parts|shares|weights?|thirds|quarters|halves|fifths|amounts|proportions)|"
    r"equal (?:thirds|quarters|halves|fifths|parts|shares|amounts) (?:of|in|across|between|among)|"
    r"weighted \d+(?:\.\d+)?%?(?:\s*/\s*\d+(?:\.\d+)?%?)+|\d+(?:\.\d+)?%\s*/\s*\d+(?:\.\d+)?%|"
    r"(?:hold|invest|put)\s+\d+(?:\.\d+)?%|\d+(?:\.\d+)?% (?:in |of )?(?:[a-z^$]{1,5}|cash)\b(?:,| and| plus)|"
    r"otherwise (?:hold |be in |in |go to |switch to )?(?:cash|[a-z^]{1,5}\b(?! days))|"
    r"hold (?:[a-z^]{1,5}|cash) (?:when|while|if|as long as)\b|weighted by|weighting by|"
    r"^\s*(?:hold|own)\s+(?!(?:it|them|for|on|onto|until|till|the (?:position|stock|shares?)|positions?|\d)\b))", re.I)
SIGNAL_HINT = re.compile(
    r"\b(?:hold(?: it)?(?: for)? \d+ (?:trading )?(?:day|bar|week|session)s?|sell (?:when|if|at|on|after|half)|stop[- ]loss|"
    r"take[- ]profit|trailing stop|profit target|cover (?:at|when|on)|exit (?:when|at|after|on)|"
    r"\bshort\b|days? in a row|\bbuy\b[^,]*\b(?:when|if|after|once)\b)", re.I)


def _blend_bench(b) -> bool:
    from . import metrics as _metrics
    try:
        return _metrics.parse_blend(b) is not None
    except ValueError:
        return False


BLEND_BENCH_RX = (r"(?i),?\s*\b(?:compared? (?:it )?(?:to|with|against)|benchmark(?:ed)?(?: it)?(?: (?:to|against|with))?|"
                  r"versus|vs\.?|against) (?:an? |the )?(?:\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)+ \S+|"
                  r"\d+(?:\.\d+)?% \S+(?:(?:,? and |, ?| ?/ ?| plus | )\d+(?:\.\d+)?% [\w^$-]+)+)(?: blend| mix| portfolio)?")


def looks_like_allocation(text: str) -> bool:
    t = _normalize(text)
    t = re.sub(BLEND_BENCH_RX, " ", t)   # a blended benchmark ("vs 60/40 SPY/AGG") says nothing about the strategy
    # "SPY 60%, TLT 30%, GLD 10%": ticker-first weights (uppercase, known tickers only)
    tw = re.findall(r"(?<![\w^])(\^?[A-Z]{1,5}(?:SIM|-USD)?) \d+(?:\.\d+)?%", t)
    if len(tw) >= 2 and all(data.canonical(x) in _known() for x in tw):
        return True
    # "60% VTI 40% BND" / "VTI 60, BND 40": a list of known tickers with weights and no separators
    lead = re.match(r"(?i)\s*(?:(?:hold|own|buy and hold|invest in|allocate)\s+)?(.+?)(?=,\s*(?:re-?balanc|rebalance|with|from|since|starting|between)\b"
                    r"|\s+(?:re-?balanc\w*|from|since|starting|beginning|between|until|through|thru|to|ending|before)\b(?! the next| the close)"
                    r"|$)", t)
    if lead and _bare_weights(lead.group(1).strip(" ,.")) != lead.group(1).strip(" ,."):
        return True
    # a bare asset-class name ("long-term corporate bonds since 1955"): 100% of that class
    if re.match(rf"(?i)\s*(?:(?:hold|own|buy and hold|invest in|allocate to)\s+)?(?:100% (?:in |of )?)?(?:{ASSET_CLASS_RX})"
                r"(?:(?:,\s*|\s+)(?:re-?balanc|with|from|since|starting|beginning|between|until|through|thru|to|ending|before|in)\b.*)?\s*$", t):
        return True
    # "40% VTISIM, 60% BNDSIM": weight-first lists naming long-history series (asset classes)
    sw = re.findall(r"\d+(?:\.\d+)?% (?:in |of |to )?(\^?[A-Z]{1,5}SIM)\b", t)
    if len(sw) >= 2 and all(x in _known() for x in sw):
        return True
    # "if C [then] X else Y" is a regime switch between holdings (Composer), whatever the branches say
    if re.match(r"(?is)\s*[(\[]?\s*if\b", t) and (
            re.search(r"(?i)\b(?:else|otherwise)\b", _mask(t))
            or (re.search(r"(?i)\bthen\b", _mask(t)) and not SIGNAL_HINT.search(t)
                and not re.search(r"(?i)\bthen,? (?:buy|short|go long|go short|sell)\b", t))
            or (re.search(r"(?i)\b(?:hold|own|be in)\b", t) and not SIGNAL_HINT.search(t))):
        return True
    if re.match(rf"(?i)\s*(?:(?:hold|own|buy and hold|backtest|test|run|invest in) )?{MODEL_RX}(?![\w/])", t):
        return True
    # named tactical models: GTAA / Ivy timing, Antonacci's GEM
    if re.match(rf"(?i)\s*(?:(?:hold|own|backtest|test|run|invest in|use|follow) )?(?:{GTAA_RX}|{GEM_RX})(?![\w/])", t):
        return True
    if not ALLOC_HINT.search(t):
        return False
    strong = re.search(r"^\s*(?:hold|own)\s+\d+(?:\.\d+)?%|\b(?:rebalanc\w*|buy and hold|equal[- ]weight|inverse[- ]volatility|(?:top|bottom) \d+(?![\d.%]|\s*%)|rotat|dual momentum|otherwise hold|allocat|contribut|withdraw|\d+/\d+|"
                       r"(?:else|otherwise),? (?:buy|hold|be in|own|switch to|go to) (?:[a-z^]{1,5}|cash)\b(?! (?:at|when|if|on)\b)|min(?:imum)?[- ]variance|whichever of|\d+ (?:best|worst)[- ]perform|"
                       r"(?:\d+ )?(?:least|most) volatile(?: \d+)? (?:of|among|from)|(?:best|worst)[- ]performing \d+ (?:of|among|from))", t, re.I)
    if SIGNAL_HINT.search(t) and not strong:
        return False
    return True


def parse(text: str, **overrides):
    """Parse a sentence into a Strategy (signals) or a Portfolio (allocations)."""
    saved = getattr(_TL, "notes", None)
    _TL.notes = []
    try:
        obj = _parse(text, **overrides)
        extra = [n for n in _pending_notes() if n not in obj.notes]
        if extra:
            obj.notes = list(obj.notes) + extra
        if getattr(obj, "month_lookbacks", None) == "calendar":
            obj.notes = [_calendar_skip_note(n) for n in obj.notes]
        _check_runnable(obj)
        return obj
    finally:
        _TL.notes = saved


def _check_runnable(obj) -> None:
    """The checks a real run would fail on, made at parse time so a dry run refuses the same specs:
    an empty or reversed period, a start after the last data date, and lookbacks / offsets the rule
    language rejects when it evaluates (a lookback below 1, a negative offset)."""
    start, end = getattr(obj, "start", None), getattr(obj, "end", None)
    from .strategy import check_date
    for v_, what in ((start, "start"), (end, "end")):
        try:
            check_date(v_, what)
        except ValueError as e:
            raise ParseError(str(e)) from None
    if start and end and str(start)[:10] > str(end)[:10]:
        raise ParseError(f"The period is reversed: it starts on {str(start)[:10]} but ends on {str(end)[:10]}.")
    known = _known()
    if start:
        tickers = ([t for t in obj.universe if t in known][:5] if isinstance(obj, Strategy) else
                   [t for t in _tree_tickers(obj.tree) if t in known][:5])
        last = None
        for t in tickers:
            try:
                d = data.load(t).index[-1]
            except Exception:
                continue
            last = d if last is None or d > last else last
        if last is not None and str(start)[:10] > str(last.date()):
            raise ParseError(f"The start date {str(start)[:10]} is after the last date with data ({last.date()}).")
    rules = []
    if isinstance(obj, Strategy):
        rules = [r for r in (obj.entry, obj.short_entry, obj.exit_when, obj.entry_level, obj.rank_by) if isinstance(r, str)]
        on = next((t for t in obj.universe if t in known), None)
    else:
        rules = [r for r in _rules(obj.tree) if isinstance(r, str)]
        on = next((t for t in _tree_tickers(obj.tree) if t in known), None)
    for r in rules:
        same = _tautologies(r)
        if same:
            raise ParseError(f"The condition {r!r} compares a value with itself ({same[0]}), so it is always true or always "
                             "false: part of the sentence was misread. Name each side's ticker, e.g. \"if SPY's 12 month "
                             "return is above BIL's 12 month return\", or write the rule in backticks.")
        _probe_rule(r, on)


def _tree_tickers(n: dict) -> list[str]:
    out = []
    if n.get("asset"):
        out.append(data.canonical(n["asset"]))
    if n.get("on"):
        out.append(data.canonical(n["on"]))
    if isinstance(n.get("universe"), list):
        out += [data.canonical(t) for t in n["universe"]]
    for c in n.get("children") or []:
        out += _tree_tickers(c)
    for k in ("then", "else", "fallback"):
        if isinstance(n.get(k), dict):
            out += _tree_tickers(n[k])
    return list(dict.fromkeys(out))


def _probe_rule(rule: str, ticker: str | None) -> None:
    """Evaluate a rule on a short synthetic series so argument errors (lookback < 1, negative offsets,
    fractional periods) surface now. Data of other tickers is not loaded: sym() is left to the run."""
    if "sym(" in rule or "tbill_ret" in rule or "market_cap" in rule:
        # these need real data; check only the plain lookback / offset literals
        for m in re.finditer(r"\b(?:ref)\([^()]*,\s*(-\d+)\s*\)", rule):
            raise ParseError(f"'{rule}': negative offsets would look into the future and are not allowed.")
        return
    import numpy as np
    import pandas as pd
    from . import expr as _expr
    idx = pd.bdate_range("2000-01-03", periods=60)
    px = pd.Series(100 + np.cumsum(np.sin(np.arange(60))), index=idx)
    df = pd.DataFrame({"open": px, "high": px + 1, "low": px - 1, "close": px, "adj_close": px, "volume": 1e6,
                       "dividend": 0.0})
    try:
        _expr.evaluate_value(rule, _expr.Namespace(df, ticker=ticker))
    except ParseError:
        raise
    except ValueError as e:
        msg = str(e)
        if re.search(r"lookback|offset|whole number|at least 1|negative", msg, re.I):
            raise ParseError(f"'{rule}': {msg}") from None
    except Exception:
        pass   # anything else (missing data, shapes) is for the run to report


_COST_WORDS = (r"(?:slippage|commissions?|expense ratio|borrow(?:ing)? (?:fee|cost|rate)s?|management fee|annual fee|fees?"
               r"|margin rate|spread|(?:per|a|each) (?:trade|order|share))")


def _negative_costs(text: str) -> None:
    """Refuse a negative cost ("-50 bps slippage", "commission of -$5"): the cost phrases read the number without
    its sign, and a negative cost would pay the strategy for trading."""
    t = re.sub(r"`[^`]*`", " ", text)
    num = r"[-\u2212]\s?\$?\s?\d+(?:\.\d+)?\s?(?:%|bps|basis points?)?"
    m = (re.search(rf"(?i)(?<![\w.)]){num}(?:\s+(?:of|in|a|an|per|each|as|for))?\s+{_COST_WORDS}\b", t)
         or re.search(rf"(?i)\b{_COST_WORDS}(?:\s+(?:of|is|at|=|:))?\s*:?\s*{num}", t))
    if m:
        raise ParseError(f"'{m.group(0).strip()}': costs cannot be negative (a negative cost would pay the strategy for "
                         "trading). Write the cost as a positive amount, e.g. '5 bps slippage' or '$1 per trade', or 0 for none.")


_R12_EVERY = (r"(?:(?:each|every)\s+(?P<{g}p>day|week|month|quarter|year)|"
          r"(?P<{g}a>daily|weekly|monthly|quarterly|annually|yearly))")
_R12_EVERY_ADV = {"day": "daily", "week": "weekly", "month": "monthly", "quarter": "quarterly", "year": "annually",
              "yearly": "annually"}
_R12_TICKERS = r"[A-Za-z0-9^.=\-]+(?:\s*,\s*[A-Za-z0-9^.=\-]+)*,?\s+(?:and|or)\s+[A-Za-z0-9^.=\-]+"
_R12_ROTATE = re.compile(
    rf"(?is)\s*rotate\s+(?:{_R12_EVERY.format(g='f')}\s+)?(?:between|among|across)\s+(?P<list>{_R12_TICKERS})\s*,?\s+"
    rf"(?:{_R12_EVERY.format(g='g')}\s+)?(?:into|to|holding|picking|buying|choosing)\s+"
    r"(?:whichever(?:\s+one)?|the\s+one|the\s+(?:asset|fund|etf|ticker|winner))(?:\s+(?:that|which))?\s+"
    r"(?:had|has|have|with|shows|showed)\s+the\s+(?:best|highest|strongest|top)\s+(?P<metric>[^,;.]+?)\s*"
    r"(?P<rest>(?:[,;].*)?)")
_R12_TOP_BODY = re.compile(r"(?is)the\s+(?:top\s+(?P<n1>\d+)\s+(?P<u1>.+?)\s+by\s+(?P<m1>[^,;]+?)|"
                       r"(?P<n2>\d+)\s+(?P<u2>.+?)\s+with\s+the\s+(?:highest|best|strongest|largest|top)\s+"
                       r"(?P<m2>[^,;]+?))\s*(?P<rest>(?:[,;].*)?)")


def _r12_every(m, g: str) -> str | None:
    w = m.group(f"{g}p") or m.group(f"{g}a")
    return _R12_EVERY_ADV.get(w.lower(), w.lower()) if w else None


def _rotation_phrasings(text: str) -> str:
    """Common ways of saying a momentum rotation / top-N selection / long-short allocation, rewritten to the
    canonical sentence the portfolio parser reads (with a note saying so):
      "each month buy the top 10 Nasdaq 100 stocks by 12 month return, equal weight",
      "buy the top 10 Nasdaq 100 stocks by 12 month return every month",
      "each month hold the 10 Nasdaq 100 stocks with the highest 12 month return"
          -> "hold the top 10 Nasdaq 100 stocks by 12 month return, rebalance monthly"
      "rotate monthly between SPY, EFA and TLT into whichever had the best 3 month return"
          -> "hold the top 1 of SPY, EFA and TLT by 3 month return, rebalance monthly"
      "hold 100% SPY and short 50% SQQQ, rebalance monthly" -> "hold 100% SPY and -50% SQQQ, rebalance monthly"
      "buy the dip in NVDA: when it drops 10% from its 52 week high, ..." -> "buy NVDA when it drops ...".
    Anything else is returned unchanged."""
    before = text
    m = _R12_ROTATE.fullmatch(text)
    if m:
        freq = _r12_every(m, "f") or _r12_every(m, "g")
        rest = m.group("rest") or ""
        tail = f", rebalance {freq}" if freq and not re.search(r"(?i)\brebalanc", rest) else ""
        text = f"hold the top 1 of {m.group('list')} by {m.group('metric').strip()}{tail}{rest}"
    elif not re.search(r"(?i)\brebalanc", text) and not _ROTATE.fullmatch(text):    # _ROTATE: read as it is
        lead = re.fullmatch(rf"(?is)\s*{_R12_EVERY.format(g='f')}\s*,?\s+(?:buy|hold|own|pick)\s+(?P<body>the\s+.+)", text)
        trail = None if lead else re.fullmatch(
            rf"(?is)\s*(?:buy|hold|own)\s+(?P<body>the\s+.+?)\s*,?\s+{_R12_EVERY.format(g='f')}\s*(?P<after>(?:[,;].*)?)",
            text)
        m2 = lead or trail
        b = _R12_TOP_BODY.fullmatch(m2.group("body")) if m2 else None
        if b:
            n, uni, metric = ((b.group("n1"), b.group("u1"), b.group("m1")) if b.group("n1")
                              else (b.group("n2"), b.group("u2"), b.group("m2")))
            rest = (b.group("rest") or "") + ((m2.group("after") or "") if trail else "")
            tail = f", rebalance {_r12_every(m2, 'f')}"
            text = f"hold the top {n} {uni.strip()} by {metric.strip()}{rest}{tail}"
    # "short 50% SQQQ" among the weights of an allocation = a -50% weight
    if len(re.findall(r"\d\s*%\s+(?:of\s+)?[A-Z^][A-Z0-9.=^\-]*\b", text)) >= 2 and \
            re.match(r"(?i)\s*(?:hold|own|invest|allocate|portfolio|buy and hold)\b", text):
        text = re.sub(r"(?i)\bshort\s+(\d+(?:\.\d+)?)\s*%\s+(?:of\s+)?(?=[A-Z^][A-Z0-9.=^\-]*\b)", r"-\1% ", text)
    # "buy the dip in NVDA: when ..." = "buy NVDA when ..."
    text = re.sub(r"(?i)\bbuy\s+the\s+dips?\s+(?:in|on|of)\s+([A-Za-z^][\w.=^\-]*)\s*[:,\-\u2013\u2014]?\s*(?=(?:when|if|once)\b)",
                  r"buy \1 ", text)
    if text != before:
        _note(f"Read as: \"{text.strip()}\".")
    return text


def _parse(text: str, **overrides):
    if not text or not text.strip():
        raise ParseError("Describe a strategy, e.g. 'buy MSFT at the close when it is down 5 days in a row, hold 1 day'.")
    from . import pine_import
    if pine_import.looks_like_pine(text):
        # a pasted TradingView strategy script: translated (or refused with its line number), not read as English
        obj = pine_import.translate(text, ticker=overrides.get("ticker"))
        return _finish(obj, overrides)
    original = text
    _TL.class_subs, _TL.class_done, _TL.mixed = {}, set(), False
    text = _lowercase_tickers(text)
    text = _asset_class_names(text)
    _intraday_check(text)
    _negative_costs(text)
    text = _rotation_phrasings(text)
    text = _review_phrases(text)   # round 13: politeness, schedules, leverage on a ticker, more phrases
    # "buy UVXY and hold" = "buy and hold UVXY" (an allocation that never rebalances)
    mbh = re.fullmatch(r"(?is)\s*buy (?P<who>[^,;`]+?) and hold(?: (?:it|them|forever|onto it|on to it))?(?P<rest>\s*(?:[,;].*)?)", text)
    if mbh and not re.search(r"(?i)\b(?:when|if|while|once|after|at|on)\b", mbh.group("who")) and find_tickers(mbh.group("who")):
        text = f"buy and hold {mbh.group('who')}{mbh.group('rest')}"
    _TL.tv = bool(overrides.get("tv_compat"))
    _TL.opp = None
    try:
        obj = _parse_text(text)
        _unsettled_class_notes()
    finally:
        _TL.tv = False
        _TL.class_subs, _TL.class_done, _TL.mixed = {}, set(), False
        _TL.opp = None
    obj.description = original
    return _finish(obj, overrides)



# ------------------------------------------------------------------ round 13 (a Composer user's review): phrases
# and clearer refusals, applied to the whole sentence before it is parsed (_parse). Kept in one place so the shared
# parsing code stays as it is.

# politeness that carries no meaning: dropped anywhere in a sentence (signals and portfolios alike)
POLITE = (r"please|pls|plz|kindly|(?:many )?thanks(?: a lot| so much| very much)?|thank you(?: very much| so much)?|thx"
          r"|cheers")
# "can you ..." / "could you please ..." at the start of the sentence
POLITE_LEAD = r"(?:(?:can|could|would|will) you|i'?d like you to|i would like you to)"
# leveraged and inverse ETFs of an underlying: (underlying, factor) -> fund, for "3x leveraged QQQ ETF"
LEVERAGED_ETFS = {
    ("QQQ", 3): "TQQQ", ("QQQ", 2): "QLD", ("QQQ", -3): "SQQQ", ("QQQ", -2): "QID", ("QQQ", -1): "PSQ",
    ("SPY", 3): "UPRO", ("SPY", 2): "SSO", ("SPY", -3): "SPXU", ("SPY", -2): "SDS", ("SPY", -1): "SH",
    ("IWM", 3): "TNA", ("IWM", 2): "UWM", ("IWM", -3): "TZA", ("DIA", 3): "UDOW", ("DIA", 2): "DDM",
    ("TLT", 3): "TMF", ("TLT", 2): "UBT", ("TLT", -3): "TMV", ("SOXX", 3): "SOXL", ("SMH", 3): "SOXL",
    ("SOXX", -3): "SOXS", ("SMH", -3): "SOXS", ("XLK", 3): "TECL", ("XLF", 3): "FAS", ("XLV", 3): "CURE",
    ("GLD", 2): "UGL", ("EEM", 3): "EDC", ("FXI", 3): "YINN", ("GDX", 2): "NUGT",
}
_FREQ_RX = r"(daily|weekly|monthly|quarterly|annually|yearly|semi-?annually)"
_REB = r"re-?balanc\w*(?: (?:it|the weights|the portfolio|them))?"
_WD_RX = r"(mondays?|tuesdays?|wednesdays?|thursdays?|fridays?)"
_CMP_WORDS = {"below": "<", "under": "<", "less than": "<", "lower than": "<", "above": ">", "over": ">",
              "greater than": ">", "more than": ">", "higher than": ">", "at least": ">=", "at most": "<="}
_CMP_RX = r"(below|under|less than|lower than|above|over|greater than|more than|higher than|at least|at most|[<>]=?)"


def _freq_word(w: str) -> str:
    w = w.lower().replace("-", "")
    return {"annually": "yearly", "annual": "yearly", "semiannually": "semiannual", "semiannual": "semiannual"}.get(w, w)


def _pct(x: str) -> str:
    return f"{float(x) / 100:g}"


def _review_phrases(text: str) -> str:
    """Round 13 phrases (a Composer user's review) -> forms the parser reads, or a clear refusal:
      politeness ("please", "kindly", "thanks", "can you ...")          dropped everywhere
      "rebalance daily, rebalance monthly"                              refused: the two schedules conflict
      "rebalance monthly with Composer timing" / "Composer-style"       rebalance on the first trading day of each month
      "..., but only on Tuesdays" (an allocation)                       refused: say "rebalance every Tuesday"
      a sentence ending in "and also" / "and"                           refused: something is missing
      "3x leveraged QQQ ETF"                                            TQQQ (with a note); "3x QQQ" is refused with
                                                                        the two ways to say it (the ETF or margin)
      "VIX > 30"                                                        VIX is above 30
      "SPY is higher than yesterday"                                    SPY closes up on the day (close > previous close)
      "SPY has risen for 3 days in a row"                               SPY is up 3 days in a row
      "SPY's 20 day return exceeds TLT's by more than 2%"               a difference of returns (percentage points)
      "SPY is down more than 10% this month"                            the month-to-date total return < -10%
      "the 2 with the smallest drawdown among A, B, C"                  the bottom 2 of A, B, C by drawdown
      "SPY RSI 10 day greater than 80" (Composer's label order)         SPY 10 day RSI greater than 80
      "10 day return is less negative than -5%"                         ... is above -5%"""
    t = text
    # -- politeness
    t = _sub_outside(rf"(?i)^\s*{POLITE_LEAD}(?: (?:{POLITE}))?\s+", "", t)
    t = _sub_outside(rf"(?i)(?:(?<=^)|(?<=[\s,;.!(]))(?:{POLITE})(?![\w'-])[,.!]*", " ", t)
    t = re.sub(r"\s+([,;.!])", r"\1", t)
    t = re.sub(r"([,;])\s*(?:[,;]\s*)+", r"\1 ", t)
    t = re.sub(r"^\s*[,;.!]+\s*|\s*[,;]+\s*$", "", re.sub(r" {2,}", " ", t)).strip()
    if not t:
        raise ParseError("Describe a strategy or a portfolio, e.g. 'hold 60% SPY and 40% TLT, rebalance quarterly'.")
    # -- a dangling conjunction at the end ("... else SPY and also")
    m = re.search(r"(?i)[,;]?\s*\b((?:and|or|but|plus)(?: also| then| too)?|also|then)\s*[.!]*\s*$", _mask(t, parens=False))
    if m and not re.search(r"(?i)\bbuy and hold\s*$", t):
        raise ParseError(f"The sentence ends with '{t[m.start(1):m.end(1)]}': something seems to be missing after it. "
                         "Finish the phrase, or remove it.")
    # -- rebalancing schedules: two different ones conflict; Composer's timing
    sched = rf"(?i)(?:\b{_REB} {_FREQ_RX}\b|\b(daily|weekly|monthly|quarterly|annual|yearly|semi-?annual) re-?balanc\w*|\b(?:never re-?balanc\w*|no re-?balancing|without re-?balancing))"
    masked = _mask(t, parens=False)
    found = [(mm.group(0), _freq_word(mm.group(1) or mm.group(2) or "none")) for mm in re.finditer(sched, masked)
             if not re.search(r"(?i)\b(?:vs\.?|versus|benchmark|compared (?:to|with)|against)\b",
                              masked[max(masked.rfind(",", 0, mm.start()), masked.rfind(";", 0, mm.start()), 0):mm.start()])]
    if len({f for _, f in found}) > 1:
        a, b = found[0][0].strip(), next(x for x, f in found if f != found[0][1]).strip()
        raise ParseError(f"Two rebalancing schedules conflict: '{a}' and '{b}'. A portfolio has one schedule: keep one of "
                         "them (a drift band can be added to it: 'rebalance monthly or when any weight drifts 5%').")
    if len(found) > 1:     # the same schedule twice: once is enough
        dup = found[1][0]
        i = t.rfind(dup)
        t = (t[:i].rstrip(" ,;") + t[i + len(dup):]).strip()
    ct = re.search(r"(?i),? ?(?:and )?(?:re-?balanc\w* )?(?:(?:with|using|on|in|at) composer(?:'s)? (?:timing|schedule|trading days?|"
                   r"rebalanc\w*(?: timing| days?| schedule)?)|composer[- ]style(?: timing| rebalanc\w*| schedule)?|"
                   r"(?:like|as) (?:in )?composer(?: does)?)(?![\w-])", t)
    if ct:
        t = (t[:ct.start()] + t[ct.end():]).strip(" ,")
        fm = re.search(rf"(?i)\b(?:{_REB}|rotat\w*) {_FREQ_RX}\b|\b(weekly|monthly|quarterly|annual|yearly) re-?balanc\w*", t)
        per = {"weekly": "week", "monthly": "month", "quarterly": "quarter", "yearly": "year"}
        f = _freq_word(fm.group(1) or fm.group(2)) if fm else None
        if f in per:
            t = t[:fm.start()] + f"rebalance on the first trading day of each {per[f]}" + t[fm.end():]
            _note(f"Composer timing: a {f} symphony trades on the first trading day of each {per[f]}, near the close, as "
                  "Composer runs it (this parser's own 'rebalance " + f + f"' trades on the last trading day of each {per[f]}).")
        elif re.search(r"(?i)\b(?:first|last|final|start|beginning|end) (?:trading )?day\b|\bevery (?:mon|tues|wednes|thurs|fri)day", t):
            raise ParseError("'Composer timing' is the first trading day of each week / month / quarter / year: the sentence "
                             "already names a rebalance day. Keep one of them.")
        else:
            _note("Composer timing: Composer evaluates a symphony's rules at every close unless it is given a weekly, "
                  "monthly, quarterly or yearly schedule (then on the first trading day of each period); here the rules "
                  "are checked daily at the close, as Composer's default.")
    # -- "but only on Tuesdays" on an allocation: the rebalancing day
    allocation = (re.search(r"(?i)\b(?:hold|holding|allocate|invest|then|else|otherwise)\b", t)
                  and not re.search(r"(?i)\b(?:buy|sell|short|go long|enter|exit|cover)\b", _mask(t, parens=False)))
    m = re.search(rf"(?i),? ?(?:but |and )?only (?:on |every |each )?{_WD_RX}(?![\w-])", _mask(t, parens=False))
    if m and allocation:
        day = m.group(1).lower().rstrip("s").capitalize()
        raise ParseError(f"'{t[m.start():m.end()].strip(' ,')}': an allocation is limited to a weekday by its rebalancing "
                         f"schedule. Say 'rebalance every {day}': the rules are then checked and the portfolio trades each "
                         f"{day} at the close, and holds what it has on the other days.")
    # -- leverage written on a ticker: "3x QQQ", "3x leveraged QQQ", "-3x QQQ", "3x inverse QQQ ETF"
    def lev(mm):
        k = float(mm.group("k"))
        inv = bool(mm.group("neg")) or bool(mm.group("inv"))
        u = mm.group("u").upper()
        if u in NOT_TICKERS or data.canonical(u) not in _known():
            return mm.group(0)
        f = -k if inv else k
        fund = LEVERAGED_ETFS.get((u, int(f))) if float(f).is_integer() else None
        said = mm.group(0).strip()
        if mm.group("etf"):
            if fund and fund in _known():
                _note(f"'{said}' was read as {fund}, the {'inverse ' if f < 0 else ''}{abs(f):g}x leveraged {u} ETF: it resets "
                      f"its leverage daily and carries its fund fees and financing costs"
                      + (f" (a {f:g}x position in {u} itself is 'hold {u} with {f:g}x leverage')." if f > 0 else "."))
                return fund
            raise ParseError(f"'{said}': no {'inverse ' if f < 0 else ''}{abs(f):g}x {u} fund is on file. Name the fund's "
                             f"ticker, or hold {u} itself with leverage ('hold {u} with {abs(f):g}x leverage').")
        alt = f"hold {fund} (the {abs(f):g}x {'inverse ' if f < 0 else ''}{u} ETF, which resets daily), or write '{said} ETF'" if fund else \
            f"name the {abs(f):g}x {u} fund's ticker"
        raise ParseError(f"'{said}': leverage written on a ticker is ambiguous. For a leveraged fund, {alt}; for a margin "
                         f"position in {u} itself, say 'hold {u} with {abs(f):g}x leverage' (borrowed at the margin rate "
                         "and reset to that leverage at each rebalance; 'rebalance daily' resets it daily like the ETF"
                         "; beyond 2x say 'with portfolio margin').")
    t = _sub_outside(r"(?<![\w.$-])(?P<neg>-)?(?P<k>\d(?:\.\d+)?) ?(?:x|times) (?:(?:daily )?(?:leveraged|levered|lev) )?"
                     r"(?P<inv>(?:inverse|short) )?(?P<u>\^?[A-Za-z]{2,5})(?P<etf> (?:etf|fund|etn)s?)?(?![\w.-])"
                     r"(?! (?:leverage|margin|volume|average|avg|the|its|atr)\b)", lev, t, flags=re.I)
    # -- "VIX > 30": a ticker compared with a number
    def level(mm):
        u = mm.group(2)
        if u.upper() in NOT_TICKERS or data.canonical(u.upper()) not in _known():
            return mm.group(0)
        word = {">": "is above", "<": "is below", ">=": "is at least", "<=": "is at most"}[mm.group(3)]
        return f"{mm.group(1)}{u} {word} {mm.group(4)}"
    t = _sub_outside(r"((?:^|\b(?:if|when|whenever|while|unless|and|or|but)\s+|[,(]\s*))(\^?[A-Z][A-Z0-9]{0,4})\s*(>=|<=|>|<)\s*"
                     r"(-?\d+(?:\.\d+)?)(?![\w%.])", level, t)
    # (a rule on another ticker than the one traded names it with sym(); an allocation's condition is on its own ticker)
    def series(who: str, ser: str) -> str:
        w = re.sub(r"'s?$", "", who or "")
        if allocation or not w or w.lower() in ("it", "its", "its price", "the"):
            return ser
        return f'sym("{data.canonical(w.upper())}").{ser}'

    def subject(who: str) -> str:
        w = re.sub(r"'s?$", "", who or "")
        return "" if not w or w.lower() in ("its", "its price", "the") else w + " "
    # -- against the previous close
    def prev_close(mm):
        up = mm.group("d").lower() in ("higher", "up")
        c = series(mm.group("who"), "close")
        return f"{subject(mm.group('who'))}`{c} {'>' if up else '<'} ref({c}, 1)`"
    t = _sub_outside(r"(?i)\b(?:(?P<who>\^?[A-Z][A-Z0-9]{0,4}|it) )?(?:is|was|closed|closes|finished|finishes|ended|ends|trades|"
                     r"is trading) (?P<d>higher|up|lower|down) (?:than|from|vs\.?|versus|compared (?:to|with)) (?:yesterday(?:'s close)?|"
                     r"the (?:previous|prior|last) (?:day(?:'s close)?|close|session|trading day)|the day before|its (?:previous|prior|last) "
                     r"close)(?![\w-])", prev_close, t)
    # -- streaks: "has risen for 3 days in a row"
    t = _sub_outside(r"(?i)\b(?:has |have )?(?:risen|rose|gone up|went up|climbed|advanced|gained)"
                     r" (?:for )?(\d+) (?:straight |consecutive )?(?:days?|sessions?)(?: in a row| straight| running)?(?![\w-])",
                     r"is up \1 days in a row", t)
    t = _sub_outside(r"(?i)\b(?:has |have )?(?:fallen|fell|dropped|gone down|went down|declined|lost)"
                     r" (?:for )?(\d+) (?:straight |consecutive )?(?:days?|sessions?)(?: in a row| straight| running)?(?![\w-])",
                     r"is down \1 days in a row", t)
    # -- one return against another's by a margin (percentage points)
    def spread(mm):
        g = mm.groupdict()
        a, b, n = g["a"], g["b"].upper(), int(g["n"])
        n2 = int(g.get("n2") or n)
        if data.canonical(b) not in _known() or (a.upper() not in ("ITS",) and data.canonical(a.upper()) not in _known()):
            return mm.group(0)
        neg = bool(re.match(r"(?i)trails|lags|underperforms|is below|is lower than|below|lower than|less than", g["dir"]))
        op = ">=" if (g.get("bound") or "").strip().lower() == "at least" else ">"
        diff = f'tret({series(a, "tr")}, {n}) - tret(sym("{data.canonical(b)}").tr, {n2})'
        rule = f"{diff} {op} {_pct(g['k'])}" if not neg else f"{diff} {'<=' if op == '>=' else '<'} -{_pct(g['k'])}"
        who = "" if a.upper() == "ITS" else a.upper() + " "
        _note(f"'{mm.group(0).strip()}': read as a difference of returns in percentage points ({who or 'its '}{n} day return "
              f"minus {b}'s{'' if n2 == n else f' {n2} day'} return {'above' if not neg else 'below'} "
              f"{'-' if neg else ''}{g['k']}%, `{rule}`), not {g['k']}% of {b}'s return.")
        return f"{who}`{rule}`"
    ret = r"(?:total )?(?:return|performance|gain)"
    t = _sub_outside(rf"(?i)\b(?P<a>\^?[A-Z][A-Z0-9]{{0,4}}|its)(?:'s)? (?P<n>\d+) day {ret} (?P<dir>exceeds|beats|outperforms|tops|"
                     rf"is above|is higher than|is greater than|trails|lags|underperforms|is below|is lower than) (?P<b>\^?[A-Z][A-Z0-9]{{0,4}})'s?"
                     rf"(?: (?P<n2>\d+) day {ret})? by (?P<bound>more than |at least |over )?(?P<k>\d+(?:\.\d+)?) ?%", spread, t)
    t = _sub_outside(rf"(?i)\b(?P<a>\^?[A-Z][A-Z0-9]{{0,4}}|its)(?:'s)? (?P<n>\d+) day {ret} is (?P<bound>more than |at least |over )?"
                     rf"(?P<k>\d+(?:\.\d+)?) ?% (?P<dir>above|greater than|more than|below|less than) "
                     rf"(?P<b>\^?[A-Z][A-Z0-9]{{0,4}})'s?(?: (?P<n2>\d+) day {ret})?(?![\w-])", spread, t)
    # -- month to date
    def mtd(who: str) -> str:
        x = series(who, "tr")
        return f"{x} / valuewhen(trading_day_of_month == 1, ref({x}, 1), 0) - 1"
    mtd_note = ("Month-to-date return: the total return since the previous month's last close (`tr / valuewhen("
                "trading_day_of_month == 1, ref(tr, 1), 0) - 1`), known at each close, so no later price is used.")

    def mtd_move(mm):
        who = mm.group("who") or ""
        up = mm.group("dir").lower() in ("up", "higher")
        strict = (mm.group("bound") or "").strip().lower() in ("more than", "over")
        k = _pct(mm.group("k"))
        rule = f"{mtd(who)} {'>' if up else '<'}{'' if strict else '='} {'' if up else '-'}{k}"
        _note(mtd_note)
        return f"{subject(who)}`{rule}`"
    t = _sub_outside(r"(?i)\b(?:(?P<who>\^?[A-Z][A-Z0-9]{0,4}|it|its price) )?(?:is |was |has been |has |have )?(?:(?:fallen|dropped|"
                     r"risen|gained|lost|fell|rose) )?(?P<dir>down|up|lower|higher) (?P<bound>more than |at least |over )?(?P<k>\d+(?:\.\d+)?) ?%"
                     r" (?:this month|so far this month|month[- ]to[- ]date|mtd)(?![\w-])", mtd_move, t)

    def mtd_cmp(mm):
        who = mm.group("who") or ""
        op = _CMP_WORDS.get(mm.group("cmp").lower(), mm.group("cmp"))
        rule = f"{mtd(who)} {op} {_pct(mm.group('k'))}"
        _note(mtd_note)
        return f"{subject(who)}`{rule}`"
    t = _sub_outside(rf"(?i)\b(?:(?P<who>\^?[A-Z][A-Z0-9]{{0,4}}'s|its|the) )?(?:month[- ]to[- ]date|mtd|this month's) (?:total )?"
                     rf"(?:return|performance|change) (?:is )?{_CMP_RX.replace('(', '(?P<cmp>', 1)} (?P<k>-?\d+(?:\.\d+)?) ?%", mtd_cmp, t)
    # -- "the 2 with the smallest drawdown among A, B and C"
    items = rf"{_TICK_ITEM}(?:(?:,\s*(?:and\s+|&\s*)?|\s+and\s+|\s*&\s*){_TICK_ITEM})*"

    def among(mm):
        top = mm.group("dir").lower() in ("largest", "highest", "biggest", "greatest", "most")
        return f"the {'top' if top else 'bottom'} {mm.group('n')} of {mm.group('items')} by {mm.group('metric')}"
    t = _sub_outside(rf"(?i)\bthe (?P<n>\d+) (?:ones? |assets? |tickers? |funds? )?(?:with|having) the (?P<dir>smallest|lowest|least|"
                     rf"largest|highest|biggest|greatest|most) (?P<metric>[a-z0-9 ]+?) (?:among|of|from|out of) (?:the )?(?P<items>{items})(?![\w-])",
                     among, t)
    # -- Composer's label order: "SPY RSI 10 day", "QQQ cumulative return 20 day"
    t = _sub_outside(r"\b(\^?[A-Z][A-Z0-9]{0,4}(?:'s)?) ((?i:RSI|relative strength index|cumulative return|return|moving average(?: of price)?|"
                     r"max(?:imum)? drawdown|standard deviation(?: of (?:price|return))?|volatility|EMA|SMA|"
                     r"exponential moving average(?: of price)?)) (\d+)[- ]days?\b", r"\1 \3 day \2", t)
    # -- "less negative than -5%" (above it), "more negative than -5%" (below it)
    t = _sub_outside(r"(?i)\b(?:is )?less negative than (?=-\d)", "is above ", t)
    t = _sub_outside(r"(?i)\b(?:is )?more negative than (?=-\d)", "is below ", t)
    return t


def _not_a_word_ticker(it: str) -> None:
    """A lowercase word in a list of holdings that is no ticker with data ('also'): say it is not understood, instead
    of an unknown-ticker error for its capitals."""
    w = it.strip()
    if re.fullmatch(r"[a-z]{2,6}", w) and data.canonical(w.upper()) not in _known():
        raise ParseError(f"'{w}' is not understood here: it is not a phrase this parser reads, and as a ticker "
                         f"({w.upper()}) it has no price data. Remove it, or name a ticker (`python -m backtester tickers` "
                         "lists all) or 'cash'.")


def _calendar_timing_note(rb: str, day_phrase, stated, tree, notes: list[str]) -> None:
    """A typed weekly / monthly / quarterly / yearly schedule with no rebalance day: say which day it trades (the
    period's last trading day), and how to get Composer's (the first)."""
    if not stated or day_phrase or rb not in ("weekly", "monthly", "quarterly", "yearly"):
        return
    per = {"weekly": "week", "monthly": "month", "quarterly": "quarter", "yearly": "year"}[rb]
    rules = _has(tree, "if") or _has(tree, "filter") or _dynamic(tree)
    notes.append(f"Rebalance timing: '{rb}' {'checks the rules and ' if rules else ''}trades on the last trading day of each "
                 f"{per}, at its close. Composer runs a {rb} symphony on the first trading day of each {per}: say "
                 f"'rebalance on the first trading day of each {per}', or add 'with Composer timing', to match it.")


def _finish(obj, overrides: dict):
    """The option overrides (command line / site settings) on top of the parsed spec, then its own checks."""
    for k, v in overrides.items():
        if v is None or k == "ticker":
            continue
        if not hasattr(obj, k):
            if k in ("max_positions", "position_size", "stop_loss", "take_profit", "trailing_stop", "hold_bars",
                     "exit_when", "entry_fill", "hold_exit_fill", "rank_by") and isinstance(obj, Portfolio):
                raise ParseError(f"--{k.replace('_', '-')} applies to signal strategies, not allocation portfolios")
            continue
        setattr(obj, k, v)
    _check_runnable(obj)
    try:
        obj.validate()
    except ParseError:
        raise
    except ValueError as e:  # the spec's own checks, reported like any other parse problem
        raise ParseError(str(e)) from None
    return obj


_ROTATE = re.compile(
    r"(?is)\s*(?:buy|hold|own)\s+(?:the\s+)?(?:top\s+)?(?P<n>\d+)\s+(?P<uni>[^,;`]+?)\s+(?:with|having|by)\s+(?:the\s+)?"
    r"(?P<dir>highest|lowest|best|worst|strongest|weakest|largest|smallest|biggest)\s+(?P<metric>[^,;`]+?)\s+"
    r"(?:each|every)\s+(?P<per>day|week|month|quarter|year)(?P<rest>\s*(?:[,;].*)?)")


def _rotation(text: str) -> str | None:
    """'buy the 3 Nasdaq 100 stocks with the highest 20 day rate of change each week' (no rule, no exit): a rotation,
    i.e. hold the top N by the metric, re-chosen each period (an equal-weight allocation portfolio)."""
    m = _ROTATE.fullmatch(text)
    if not m or re.search(COND_START, m.group("uni") + " " + m.group("metric"), re.I) \
            or re.search(r"(?i)\b(?:sell|exit|cover|hold|keep|stop|target|trailing|profit|after|days?|bars?|sessions?)\b",
                         m.group("rest")):
        return None
    top = m.group("dir").lower() not in ("lowest", "worst", "weakest", "smallest")
    freq = {"day": "daily", "week": "weekly", "month": "monthly", "quarter": "quarterly", "year": "yearly"}[m.group("per").lower()]
    _note(f"'{m.group(0).strip()}' was read as a rotation: hold the {m.group('n')} {m.group('uni')} with the "
          f"{m.group('dir').lower()} {m.group('metric')}, equal weight, re-chosen and rebalanced {freq} (an allocation portfolio).")
    return (f"hold the {'top' if top else 'bottom'} {m.group('n')} {m.group('uni')} by {m.group('metric')}, "
            f"rebalance {freq}{m.group('rest')}")


_BARE_SHORT = re.compile(r"(?is)\s*(?:go short|short[- ]sell|sell short|short|be short)\s+(?:the\s+)?(\^?[A-Za-z][A-Za-z0-9.\-]{0,9})"
                         r"\s*(?:(,|;)\s*(?P<rest>.*))?")
_SIGNAL_WORDS = re.compile(r"(?i)\b(?:when|whenever|if|while|once|until|after|above|below|cross\w*|rsi|stop|exit|cover|sell|"
                           r"target|profit|days? in a row|breaks?|falls?|drops?|rises?|gaps?)\b")


def _bare_short(text: str) -> str | None:
    """'short SQQQ' / 'go short SQQQ, since 2015' with no entry condition: a static 100% short allocation, the same
    as 'hold 100% short SQQQ'. None when the sentence is anything else (a rule to short on a condition)."""
    m = _BARE_SHORT.fullmatch(text.strip().rstrip("."))
    if not m or (m.group("rest") and _SIGNAL_WORDS.search(m.group("rest"))):
        return None
    try:
        tk = find_tickers(m.group(1).upper() if m.group(1).islower() else m.group(1), strict=True)
    except ParseError:
        return None
    if len(tk) != 1:
        return None
    return f"hold 100% short {tk[0]}" + (f", {m.group('rest')}" if m.group("rest") else "")


_UI_TK = r"(?:[\^$]?[A-Za-z][A-Za-z0-9.\-]{0,9}|cash)"


def _composer_ui_shorthand(text: str) -> str | None:
    """Composer's editor labels typed as text -> the English the portfolio parser reads (None if not that shape):
      'Weight equal: SPY, TLT'                              -> 'equal weight SPY and TLT'
      'Weight specified 60% SPY 40% TLT'                    -> '60% SPY 40% TLT'
      'Weight inverse volatility 20d: SPY, TLT'             -> 'SPY and TLT, inverse volatility weighted over 20 days'
      'Filter top 2 by 10d cumulative return: TQQQ, SOXL'   -> 'the top 2 of TQQQ and SOXL by 10 day cumulative return'
    Options after the holdings (', rebalance monthly') are kept after the rewritten phrase."""
    t = text.strip()
    m = re.match(r"(?is)\s*(?:weight(?:ing)?|weights)\s*:?\s*(?P<kind>equal(?:ly)?|specified|inverse[- ]vol(?:atility)?)"
                 r"(?:\s+(?P<n>\d+)\s*(?:d|days?|-day)\b)?\s*[:\-]?\s*(?P<rest>.+)$", t)
    f = None if m else re.match(r"(?is)\s*filter\s*:?\s*(?P<sel>top|bottom)\s+(?P<n>\d+)\s+(?:by\s+)?(?P<met>[^:]+?)\s*:\s*(?P<rest>.+)$", t)
    if not m and not f:
        return None
    rest = (m or f).group("rest")
    if m and m.group("kind").lower().startswith("specified"):
        return rest
    items = [x.strip() for x in re.split(r"\s*,\s*|\s+and\s+", rest)]
    k = 0
    while k < len(items) and re.fullmatch(_UI_TK, items[k]) and not re.fullmatch(
            r"(?i)re-?balanc\w*|since|from|until|starting|with|vs|versus|benchmark", items[k]):
        k += 1
    hold, opts = items[:k], items[k:]
    if not hold:
        return None
    lst = hold[0] if len(hold) == 1 else ", ".join(hold[:-1]) + " and " + hold[-1]
    tail = (", " + ", ".join(opts)) if opts else ""
    if f:
        met = re.sub(r"(?i)\b(\d+)\s*d\b", r"\1 day", f.group("met").strip())
        # "cumulative return 10 day" (Composer's label order) -> "10 day cumulative return"
        met = re.sub(r"(?i)^(?P<w>[a-z][a-z ]*?)\s+(?P<n>\d+ day)$", r"\g<n> \g<w>", met)
        return f"the {f.group('sel').lower()} {f.group('n')} of {lst} by {met}{tail}"
    if m.group("kind").lower().startswith("equal"):
        return f"equal weight {lst}{tail}"
    look = f" over {m.group('n')} days" if m.group("n") else ""
    return f"{lst}, inverse volatility weighted{look}{tail}"


def _parse_text(text: str):
    ui = _composer_ui_shorthand(text)
    if ui:
        obj = parse_allocation(ui)
        obj.notes.insert(0, f"Composer editor shorthand '{text.strip()}' was read as '{ui}'.")
        return obj
    # "TQQQ, except when <condition> hold BIL": a switch between holdings (read by _except_when)
    if re.search(r"(?i)^\s*(?:hold |own |be in |invest in )?[^,;]+?,?\s+except (?:when(?:ever)?|while|if|as long as)\b.+?"
                 r"(?:\bhold\b|\bbe in\b|\bown\b|\bin which case\b|\bswitch to\b|\bgo to\b|\bmove to\b)", _mask(text)):
        return parse_allocation(text)
    rot = _rotation(text)
    if rot:
        return parse_allocation(rot)
    short = _bare_short(text)
    if short:
        obj = parse_allocation(short)
        head = short.split(",")[0]
        obj.notes.insert(0, f"'{text.strip()}' has no entry condition, so it was read as a static short allocation held the "
                            f"whole time, the same as '{head}': -100% reset at each rebalance, the sale proceeds kept in cash. "
                            f"For a trading rule, say e.g. '{text.strip().split(',')[0]} when ..., cover when ...'.")
        return obj
    held = _holding_signal(text)
    if held:
        obj = parse_signal(held, holding=True)
        obj.description = text
    elif looks_like_allocation(text):
        obj = parse_allocation(text)
    else:
        obj = parse_signal(text)
    return obj


# ----------------------------------------------------------------- signal strategies

ENTRY_VERB = r"^(?:buy|go long|long|purchase|enter(?: long)?|get in|short|sell short|go short|short[- ]sell|enter short)\b"
EXIT_VERB = r"^(?:hold|keep|sell|exit|cover|close (?:the|out|it|positions?)|get out|take profit|stop|use|with|place|then sell|and sell|after|or after|max(?:imum)? hold)\b"
COND_START = r"\b(?:when(?:ever)?|while|as long as|if|once|after|on days when|any day|every time|each time)\b"
TIMING = r"(?:(?:at|on) (?:the )?(?:next |following |tomorrow's )?(?:day's |trading day's )?(?:close|open)|market on (?:close|open)|\bmoc\b|\bmoo\b|at the next open|next day's open|next open)"


EXIT_WHEN = (r"(?:(?:sell|exit|cover|close (?:the position|out|it)|get out)\w*(?: it| the position| out| them| everything)?"
             r"(?: (?-i:(?P<xt>[\^$]?[A-Z]{1,5}(?:SIM)?))(?:'s)?(?: (?:shares|stock|position))?)?"
             r"(?: (?:at|on) (?:the )?(?:next |following |tomorrow's )?(?:day's |trading day's )?(?:open|close))?|\b(?:or|and))"
             r" (?:when(?:ever)?|if|once|on|as soon as|at the first|at the close of the first|at the close when|at the open after|the day after)\b(?P<body>.*)$")


_THEY_VERBS = {"cross": "crosses", "close": "closes", "trade": "trades", "fall": "falls", "rise": "rises", "drop": "drops",
               "break": "breaks", "move": "moves", "go": "goes", "gap": "gaps", "pull": "pulls", "make": "makes",
               "hit": "hits", "have": "has", "were": "was", "are": "is", "dip": "dips", "stay": "stays", "touch": "touches",
               "reach": "reaches", "open": "opens", "decline": "declines", "gain": "gains", "lose": "loses"}


def _they_to_it(s: str) -> str:
    """'they cross above their 50 day moving average' (several tickers) -> 'it crosses above its ...': each ticker is
    tested on its own, exactly as with 'it'."""
    s = re.sub(r"(?i)\bthey(?:'re| are)\b", "it is", s)
    s = re.sub(r"(?i)\bthey(?:'ve| have)\b", "it has", s)
    s = re.sub(r"(?i)\bthey (\w+)\b", lambda m: "it " + _THEY_VERBS.get(m.group(1).lower(), m.group(1)), s)
    s = re.sub(r"(?i)\bthey\b", "it", s)
    return re.sub(r"(?i)\btheir\b", "its", s)


def _entry_crosses(entry: str) -> list[tuple[str, str, str]]:
    """The distinct crossover(a, b) / crossunder(a, b) calls of an entry rule, as (function, a, b)."""
    import ast as _ast
    try:
        tree = _ast.parse(entry, mode="eval")
    except SyntaxError:
        return []
    out = []
    for n in _ast.walk(tree):
        if (isinstance(n, _ast.Call) and isinstance(n.func, _ast.Name) and n.func.id in ("crossover", "crossunder")
                and len(n.args) == 2 and not n.keywords):
            x = (n.func.id, _ast.unparse(n.args[0]), _ast.unparse(n.args[1]))
            if x not in out:
                out.append(x)
    return out


def _exit_rule(wl: str, entry: str, universe: list[str], notes: list[str]) -> str:
    """The rule of a 'sell when ...' clause. Resolves references back to the entry: 'it crosses back
    below' / 'QQQ closes below it' (the entry's price comparison reversed), 'it is over 70' (the
    entry's indicator), a bare 'RSI' (the entry's RSI period) and 'the signal ends'."""
    one = universe[0] if len(universe) == 1 else None
    wl = _r13_exit_refs(wl, entry, notes)
    # "they are falling" (several tickers) is the same pronoun as "it is falling"
    wl = re.sub(r"(?i)\bthey were\b", "it is", wl)
    wl = _they_to_it(wl)
    wl = re.sub(r"(?i)\btheir\b", "its", wl)
    # "it crosses below zero" / "the zero line": the number 0 (the entry's indicator crossing it, below)
    wl = _sub_outside(r"(?i)\b(above|below|over|under|than|through) (?:the )?zero(?: line)?\b", r"\1 0", wl)
    tick = r"(?P<tk>[\^$]?[a-z][a-z0-9.&'-]{0,24}(?: [a-z][a-z0-9.&'-]{0,24}){0,2}?)(?:'s(?: price)?)?"
    pron = re.fullmatch(rf"(?i)(?:it |the price |price |{tick} )?(?P<verb>crosses|falls|drops|closes|goes|moves|is|trades|gets)?(?: back)? ?"
                        r"(?P<rel>below|under|above|over)(?: (?:it|them|that|the average|the line|again))?", wl)
    tk = None
    if pron and pron.group("tk"):
        found = find_tickers(pron.group("tk"), strict=False)
        if len(found) != 1 or pron.group("tk").lower() in ("it", "the price", "price"):
            pron = None
        else:
            tk = found[0]
    opp = getattr(_TL, "opp", None)
    if pron and opp and tk is None and re.fullmatch(rf"(?i)it crosses {opp[1]} it", wl.strip()):
        # "exit on the opposite cross": the entry's own crossing, the other way (whatever else the entry requires)
        crosses = _entry_crosses(entry)
        alone = len(crosses) == 1 and re.sub(r"[()\s]", "", entry) == re.sub(r"[()\s]", "", "{}({}, {})".format(*crosses[0]))
        if len(crosses) == 1 and not alone:     # (the cross alone: the first close on the other side, below)
            fn, a, b = crosses[0]
            flip = f"{'crossunder' if fn == 'crossover' else 'crossover'}({a}, {b})"
            notes.append(f"'{opp[0]}' was read as the entry's cross the other way: {flip}.")
            return flip
        if len(crosses) > 1:
            raise ParseError(f"'{opp[0]}': the entry has {len(crosses)} crosses ({', '.join(f'{f}({a}, {b})' for f, a, b in crosses)}); "
                             "the opposite of which one? Say it, e.g. 'sell when the 10 day SMA crosses below the 50 day SMA'.")
    if pron:
        below = pron.group("rel").lower() in ("below", "under")
        series = None if tk is None else ("close" if tk == one else f'sym("{tk}").close')
        price_only = tk is not None or (pron.group("verb") or "").lower() in ("closes", "trades")
        flip = _flip(entry, below=below, series=series, price_only=price_only)
        if not flip:
            what = f"{tk}'s price" if tk else ("the price" if price_only else "anything")
            if opp and tk is None:
                raise ParseError(f"'{opp[0]}': the entry ({entry.strip('()')}) has no cross to reverse. Say when to sell, "
                                 "e.g. 'sell when the 10 day SMA crosses below the 50 day SMA'.")
            raise ParseError(
                f"'{wl}': 'it' has nothing to refer back to - the entry ({entry.strip('()')}) does not compare {what} "
                f"with exactly one level or average. Say what it crosses, e.g. "
                f"'sell when {tk or 'it'} closes below its 5 day moving average'.")
        notes.append(f"Warning: '{wl}' was read as the reverse of the entry comparison: {flip}.")
        return flip
    if re.fullmatch(r"(?i)(?:the |its )?(?:signal|condition|setup|entry (?:rule|condition))s? (?:reverses?|ends?|is no longer (?:true|met)|no longer holds?|turns? off)|it(?:'s| is)? no longer (?:true|met)|not ?", wl):
        return f"not ({entry})"

    # a bare "RSI" means the entry's RSI period when the entry uses exactly one
    periods = list(dict.fromkeys(re.findall(r"\brsi\(close, (\d+)\)", entry)))

    def bare_rsi(m):
        if m.group(1) or m.group(2):
            return m.group(0)
        if len(periods) == 1:
            notes.append(f"Warning: 'RSI' in the exit was read as the entry's RSI({periods[0]}).")
            return f"RSI({periods[0]})"
        if len(periods) > 1:
            raise ParseError(f"'{wl}': which RSI? The entry uses RSI({') and RSI('.join(periods)}). Give the period, e.g. 'RSI({periods[0]}) is above 70'.")
        notes.append("'RSI' with no period: using the standard 14-day RSI.")
        return m.group(0)
    wl = _sub_outside(r"(?i)(?<!weekly )(?<!monthly )\b(?:(\d+) (?:day|period|bar|session) )?rsi\b(\s*\(\s*\d+\s*\)|\s+\d+(?![\d.]|\s*%))?", bare_rsi, wl)

    # "it flips to down" after a Supertrend entry, "it turns red" after a Heikin Ashi entry: the same signal, reversed
    def it_flips(m):
        d = m.group("d").lower()
        up = d in ("up", "bullish", "green", "long", "buy", "positive")
        std = re.search(r"supertrend_dir\((\d+), (\d+(?:\.\d+)?)\)", entry)
        st = re.search(r"(?<!_)supertrend\((\d+), (\d+(?:\.\d+)?)\)", entry)
        if std:
            e = std.group(0)
            rule = f"{e} < 0 and ref({e}, 1) > 0" if up else f"{e} > 0 and ref({e}, 1) < 0"
            what = "the Supertrend direction (-1 up, +1 down) flipping"
        elif st:
            rule = f"{'crossover' if up else 'crossunder'}(close, {st.group(0)})"
            what = "the close crossing the Supertrend line"
        elif "ha_close" in entry and d in ("green", "red", "bullish", "bearish", "up", "down"):
            op, nop = (">", "<=") if up else ("<", ">=")
            rule = f"ha_close {op} ha_open and ref(ha_close, 1) {nop} ref(ha_open, 1)"
            what = "the Heikin Ashi candle changing colour"
        elif re.match(r"(?i)turn", m.group("verb")) and d in ("up", "down"):
            return m.group(0)       # "it turns down": the entry's subject falling (below)
        else:
            raise ParseError(f"'{m.group(0).strip()}': what flips? The entry has no Supertrend or Heikin Ashi signal to "
                             "reverse. Name it, e.g. 'sell when the supertrend flips bearish'.")
        notes.append(f"Warning: '{m.group(0).strip()}' was read as {what} the other way: {rule}.")
        return f" `{rule}` "
    wl = _sub_outside(r"(?i)\bit\s+(?P<verb>flips?|flipped|turns?|turned|switches|changes|changed)\s+(?:to\s+)?"
                      r"(?P<d>up|down|bullish|bearish|green|red|long|short)\b", it_flips, wl)

    # "it is over 70" / "it's under 30" / "it rises back above 50" -> the entry's indicator
    oscs = _calls(entry)

    inds = _entry_indicators(entry)

    def it_turns(m):
        if len(inds) != 1:
            raise ParseError(f"'{m.group(0).strip()}': what turns {m.group('sign')}? "
                             + (f"The entry uses {' and '.join(inds)}. " if inds else "")
                             + "Name it, e.g. 'sell when the MACD histogram turns negative'.")
        pos = m.group("sign").lower() == "positive"
        e = (f"{'crossover' if pos else 'crossunder'}({inds[0]}, 0)" if not re.match(r"(?i)is|stays", m.group("verb"))
             else f"{inds[0]} {'>' if pos else '<'} 0")
        notes.append(f"Warning: '{m.group(0).strip()}' was read as referring to the entry's indicator: {e}.")
        return f" `{e}` "
    wl = _sub_outside(r"(?i)\bit\s+(?P<verb>turns?|goes|flips?|becomes|gets|is|stays)\s+(?P<sign>positive|negative)\b", it_turns, wl)

    # what 'it' is: the entry's subject(s) - the price, or the indicator the entry is about
    subjects = _entry_subjects(entry)
    ind_subjects = [x for x in subjects if not _is_price(x)]

    def it_subject(phrase: str) -> str | None:
        """The one indicator 'it' refers to (None: the price). Refuses when the entry is about several things."""
        if not ind_subjects:
            return None
        if len(subjects) > 1:
            raise ParseError(f"'{phrase}': 'it' is ambiguous - the entry is about {' and '.join(subjects)}. Name the one you "
                             f"mean, e.g. 'sell when the 50 day moving average is falling' or 'sell when the price is falling'.")
        return ind_subjects[0]

    # "it is falling" / "it turns down" after "buy when the 50 day moving average is rising": the entry's indicator
    def it_trend(m):
        e = it_subject(m.group(0).strip())
        if e is None:
            return m.group(0)
        op = ">" if re.match(r"(?i)ris|increas|going up|trending up|sloping up|pointing up|moving up|turns? up|up", m.group("dir")) else "<"
        rule = f"{e} {op} ref({e}, 1)"
        notes.append(f"Warning: 'it' in '{m.group(0).strip()}' was read as the entry's {e} (not the price): {rule}, "
                     f"i.e. {'higher' if op == '>' else 'lower'} than on the previous day.")
        return f" `{rule}` "
    wl = _sub_outside(r"(?i)\bit(?:'s|\s+is|\s+has\s+been|\s+starts|\s+begins)?\s+(?P<dir>rising|falling|increasing|decreasing|declining|"
                      r"going up|going down|trending up|trending down|sloping up|sloping down|pointing up|pointing down|"
                      r"moving up|moving down|turns? up|turns? down|to rise|to fall)(?:\s+today)?\b", it_trend, wl)

    def it_osc(m):
        verb = (m.group("verb") or "").strip().lower()
        below_ = m.group("rel").lower() in ("below", "under", "less than", "lower than")
        # "it rises above 70" / "it falls below 30": a move through the level, the crossing (as _motion_verbs reads it)
        motion = (verb in ("rises", "climbs") and not below_) or (verb in ("falls", "drops", "dips") and below_)
        crossing = m.group("back") or verb.startswith("cross") or motion
        if motion and not m.group("back"):
            notes.append(f"'{verb} {m.group('rel')} {m.group('n')}' was read as a crossing ({'crossunder' if below_ else 'crossover'}: "
                         f"true only on the bar it moves {'below' if below_ else 'above'} the level, as in TradingView); say "
                         f"'is {m.group('rel')} {m.group('n')}' for every bar it is {'below' if below_ else 'above'}.")
        if not oscs:
            e_ = it_subject(m.group(0).strip())
            if e_ is None:
                return m.group(0)   # the entry is about the price (or no indicator): 'it' is the price
            op = _cmp(m.group("rel"))
            e = f"{'crossover' if op == '>' else 'crossunder'}({e_}, {m.group('n')})" if crossing else f"{e_} {op} {m.group('n')}"
            notes.append(f"Warning: '{m.group(0).strip()}' was read as referring to the entry's indicator: {e}.")
            return f" `{e}` "
        if len(oscs) > 1:
            raise ParseError(f"'{m.group(0).strip()}' is ambiguous: the entry uses {' and '.join(oscs)}. "
                             f"Name the indicator, e.g. 'RSI(2) is above 70'.")
        op = _cmp(m.group("rel"))
        if crossing:
            e = f"{'crossover' if op == '>' else 'crossunder'}({oscs[0]}, {m.group('n')})"
        else:
            e = f"{oscs[0]} {op} {m.group('n')}"
        notes.append(f"Warning: '{m.group(0).strip()}' was read as referring to the entry's indicator: {e}.")
        return f" `{e}` "
    wl = _sub_outside(r"(?i)\bit(?:'s|\s+is)?\s+(?:(?:now|still)\s+)?(?P<verb>(?:rises|climbs|falls|drops|goes|gets|moves|crosses|comes|dips|reads)\s+)?"
                r"(?P<back>back\s+)?(?P<rel>above|over|below|under|greater than|less than|higher than|lower than)\s+(?P<n>-?\d+(?:\.\d+)?)"
                r"(?![\d.]|\s*%|\s*(?:day|week|month|bar|period|session)s?\b)", it_osc, wl)
    return parse_conditions(wl, universe)


def _split_clauses(t: str) -> list[str]:
    t = _sub_outside(r"\band\s+(?=(?:then\s+)?(?:hold|keep|sell|exit|cover|close (?:the|out|it)|take profit|use a|with a|place a|go short|short|go long|buy)\b)", ", ", t, flags=re.I)
    t = _sub_outside(r"\bthen\b", ",", t, flags=re.I)
    mk = _mask(t)   # no splitting inside `backticks` or brackets
    chunks, buf = [], ""
    for i, ch in enumerate(t):
        nxt = t[i + 1] if i + 1 < len(t) else " "
        if mk[i] != "\0" and (ch in ",;" or (ch == "." and not (buf and buf[-1].isdigit() and nxt.isdigit()))):
            chunks.append(buf)
            buf = ""
        else:
            buf += ch
    chunks.append(buf)
    return [c.strip() for c in chunks if c.strip()]


def _universe_phrase(text: str) -> tuple[list[str] | None, str | None]:
    from . import index_universes
    iu = index_universes.universe(text)    # "today's S&P 500 members"; asks about "S&P 500 / Russell 2000 stocks"
    if iu is not None:
        return iu
    low = text.lower()
    if re.search(r"nasdaq[- ]?100|\bndx\b|(?:all|each|any|every) (?:the )?(?:index )?(?:stocks?|members?|components?|constituents?)", low):
        return data.nasdaq100_ever(), "NDX"
    if re.search(r"sector (?:etfs|spdrs|funds)", low):
        return [t for t in SECTOR_ETFS if t in _known()], None
    if re.search(r"all (?:available )?tickers|entire universe|whole universe", low):
        return data.available_tickers(), None
    return None, None


from .index_universes import PHRASE as _INDEX_MEMBERS  # noqa: E402

UNIVERSE_WORDS = (_INDEX_MEMBERS + r"|nasdaq[- ]?100|\bndx\b|(?:all|each|any|every) (?:the )?(?:index )?(?:stocks?|members?|components?|constituents?)|"
                  r"sector (?:etfs|spdrs|funds)|all (?:available )?tickers|entire universe|whole universe")


def _entry_subject(subj: str, clause: str, notes: list[str]) -> dict:
    """The words between the entry verb and 'when' ('buy SPY with 100 shares when ...'): sizing and holding
    phrases are read, and anything else is refused - never silently dropped. Returns option overrides."""
    out: dict = {}
    s = " " + subj + " "
    s = re.sub(ENTRY_VERB.replace("^", r"(?<![\w])"), " ", s.strip(), count=1, flags=re.I)
    s = " " + re.sub(TIMING, " ", s, flags=re.I) + " "
    m = re.search(r"(?i)\b(?:but|and|except|excluding) not\b[^,;]*|\b(?:but )?(?:not|except|excluding) (?-i:[\^$]?[A-Z]{1,5}(?:SIM)?)\b", s)
    if m:
        raise ParseError(f"'{m.group(0).strip()}' in '{clause.strip()}': excluding tickers is not supported. Name only the ticker(s) "
                         "to trade, e.g. 'buy SPY when ...'.")
    m = re.search(r"(?i)\b(?:on|with|using) margin\b|\bmargined\b", s)
    if m:
        raise ParseError(f"'{m.group(0).strip()}' in '{clause.strip()}': say how much leverage, e.g. 'buy SPY with 2x leverage when ...' "
                         "(the borrowed part pays the margin rate).")
    s = re.sub(rf"(?i){_INDEX_MEMBERS}", " ", s)      # "today's S&P 500 members" (before 'S&P 500' is read as SPY)
    for tk in find_tickers(s):
        s = _sub_outside(rf"(?<![\w])[\$^]?{re.escape(tk.lstrip('^'))}(?:'s)?\b", " ", s, flags=re.I)
    for name in COMPANIES:
        s = _sub_outside(rf"\b{re.escape(name)}\b", " ", s, flags=re.I)
    s = re.sub(rf"(?i){UNIVERSE_WORDS}", " ", s)
    # "with 100 shares (each)", "100 shares of"
    m = re.search(r"(?i)(?:\b(?:with|using|in) )?\b(\d+) shares?(?: (?:each|apiece|per (?:trade|position|ticker|stock)))?(?: of)?\b", s)
    if m:
        out["sizing"], out["fixed_amount"] = "fixed_shares", float(m.group(1))
        s = s.replace(m.group(0), " ")
    # "$5000 of", "with $5000 each"
    m = re.search(r"(?i)(?:\b(?:with|using|in) )?\$(\d+(?:\.\d+)?)(?: (?:each|apiece|per (?:trade|position|ticker|stock)))?(?: (?:worth )?of)?(?!\S)", s)
    if m:
        if out:
            raise ParseError(f"'{clause.strip()}' gives two position sizes; keep one.")
        out["sizing"], out["fixed_amount"] = "fixed_dollars", float(m.group(1))
        s = s.replace(m.group(0), " ")
    # "with half my account" (normalised to 50%), "with 25% of the equity"
    m = re.search(r"(?i)(?:\b(?:with|using|and put) )?(?<![\w.])(\d+(?:\.\d+)?)% (?:of )?(?:my |the |our )?(?:account|equity|capital|portfolio|money|cash|balance)"
                  r"(?: (?:each|per (?:trade|position)))?\b", s)
    if m:
        if out:
            raise ParseError(f"'{clause.strip()}' gives two position sizes; keep one.")
        out["position_size"] = float(m.group(1)) / 100
        s = s.replace(m.group(0), " ")
    # "for 3 days": the holding period
    m = re.search(r"(?i)\bfor (\d+) (?:trading )?(day|bar|session|week|month)s?\b", s)
    if m:
        out["_hold"] = _period(m.group(1), m.group(2))
        s = s.replace(m.group(0), " ")
    left = [w for w in re.findall(r"[a-z%$']+|\d+(?:\.\d+)?", s.lower()) if w not in STOP and w not in ("shares", "share", "of")]
    if left:
        raise ParseError(f"Could not interpret {' '.join(left)!r} in the entry '{clause.strip()}'. Put conditions after "
                         "'when', e.g. 'buy SPY when RSI(2) is below 10', and sizing as e.g. 'with 100 shares' or 'with 50% of the account'.")
    return out


def _holding_signal(text: str) -> str | None:
    """'hold TSLA when it is above its 50 day moving average' (one ticker, no otherwise / rebalancing)
    is a daily signal strategy: in the market while the condition holds. Returns it as 'buy TSLA while ...'."""
    t = _normalize(text)
    m = re.match(r"(?i)\s*(?:hold|own|stay long|be long|stay in|be in)\s+(?P<who>.+?)\s+(?:when(?:ever)?|while|if|as long as|only when|only while|only if)\s+(?P<rest>.+)$", t)
    if not m or re.search(r"(?i)\b(?:otherwise|else|rebalanc\w*|withdraw\w*|contribut\w*|rotat\w*|allocat\w*)\b|\d%\s+(?:in\s+|of\s+)?[A-Za-z^]{1,5}\b(?!\s+(?:stop|trailing|target|take|profit|slippage|commission|per|of|below|above|drawdown|gain|loss))", t):
        return None
    who = m.group("who")
    tk = find_tickers(who, strict=True)
    rest_who = re.sub(r"[\^$]?\b(?:" + "|".join(re.escape(x.lstrip("^")) for x in tk) + r")\b", " ", who, flags=re.I) if tk else who
    for name in COMPANIES:
        rest_who = re.sub(rf"\b{re.escape(name)}\b", " ", rest_who, flags=re.I)
    if len(tk) != 1 or re.sub(r"(?i)\b(?:the|shares?|stock|of)\b", "", rest_who).strip():
        return None
    return f"buy {who} while {m.group('rest')}"


_PINE_OPERAND = (r"(?:(?:ta|math)\.\w+|request\.security|[a-z_]\w*|\d+(?:\.\d+)?)(?:\[\d+\])?"
                 r"(?:\((?:[^()`]|\((?:[^()`]|\([^()`]*\))*\))*\)(?:\[\d+\])?)?")
_PINE_EXPR = rf"{_PINE_OPERAND}(?:\s*[-+*/]\s*{_PINE_OPERAND})*"
_PINE_CROSS = r"\s+cross(?:es|ed|ing)?\s+(?:above|over|below|under)\s+"
_PINE_RX = re.compile(rf"(?i)(?<![\w.`])(?:{_PINE_EXPR}\s*(?:>=|<=|==|!=|>|<)\s*{_PINE_EXPR}"
                      rf"|{_PINE_EXPR}{_PINE_CROSS}{_PINE_EXPR}"
                      rf"|ta\.(?:crossover|crossunder|cross|rising|falling)\((?:[^()`]|\((?:[^()`]|\([^()`]*\))*\))*\))(?![\w.(\[])")
_PINE_WORDS = {"close", "open", "high", "low", "volume", "hl2", "hlc3", "ohlc4", "hlcc4", "true", "false"}


def _pine_backticks(t: str) -> str:
    """TradingView / rule-language expressions written in a sentence without backticks ("buy SPY when close >
    ta.sma(close, 200)") are wrapped in backticks, so the dot and the commas inside them are not read as English.
    Only a comparison or ta.cross...() that contains a Pine marker (ta.xxx(, math.xxx(, close[1]) and nothing but
    price names, functions and numbers."""
    if not re.search(r"(?i)\b(?:ta|math)\.\w+\s*\(|\brequest\.security\s*\(|\b(?:close|open|high|low|volume|hl2|hlc3|ohlc4)\[\d+\]", t):
        return t
    from .expr import pine_to_rule

    def wrap(m):
        e = m.group(0)
        if not re.search(r"(?i)\b(?:ta|math)\.\w+|\brequest\.security\b|\[\d+\]", e):
            return e
        mc = re.fullmatch(rf"(?is)(.+?){_PINE_CROSS}(.+)", e)
        if mc:      # "close crosses above ta.ema(close, 20)" -> crossover(close, ta.ema(close, 20))
            up = bool(re.search(r"(?i)\b(?:above|over)\b", e[mc.end(1):mc.start(2)]))
            if re.search(r"(?i)\b(?:and|or|not|when|if|then)\b", mc.group(1) + " " + mc.group(2)):
                return e
            e = f"{'crossover' if up else 'crossunder'}({mc.group(1).strip()}, {mc.group(2).strip()})"
        # every bare word must be a price name or a function call (no English words inside; quoted strings such as
        # request.security's "W" are not words)
        for w in re.finditer(r"(?i)(?<![\w.\"'])([a-z_]\w*)(?!\w*\s*[.(])", re.sub(r'"[^"]*"|\'[^\']*\'', '""', e)):
            if w.group(1).lower() not in _PINE_WORDS:
                return e
        try:
            pine_to_rule(e)
        except ValueError:
            return e
        _note(f"'{e}' was read as a TradingView / rule-language expression (put such expressions in `backticks` to be sure).")
        return f"`{e}`"
    mk = _mask(t, parens=False)
    out, last = [], 0
    for m in _PINE_RX.finditer(mk):
        out.append(t[last:m.start()])
        out.append(wrap(_MM(m, t)))
        last = m.end()
    out.append(t[last:])
    return "".join(out)


_NEG_LOOKBACK = re.compile(r"(?<![\w.%$])-\s*\d+(?:\.\d+)?[- ]?(?:trading )?(?:day|bar|session|week|month|year|period)s?\b"
                           r"|(?i:\b(?:rsi|sma|ema|wma|hma|ma|roc|atr|natr|cci|mfi|adx|willr|stoch\w*|volatility|momentum|"
                           r"return)\s*\(\s*-\s*\d+)")


def _negative_lookbacks(t: str) -> None:
    """Refuse a negative lookback ("SPY -10 day RSI", "RSI(-2)", "by -12 month return"): the phrases read the number
    without its sign, so it would silently become a 10-day RSI."""
    m = _msearch(_NEG_LOOKBACK.pattern, t, flags=re.I, parens=False)
    if m:
        raise ParseError(f"'{m.group(0).strip()}': a holding period or lookback must be a positive number of days "
                         "(a negative one would look into the future, which is not allowed).")


_SUBJ = r"(?:it|the price|price|the stock|the etf|the close|the candle|the bar|today|the day)"


def _signal_phrases(t: str) -> str:
    """Trader phrases of signal strategies rewritten into the vocabulary the parser reads (each with a note):
    'exit on the opposite cross', 'breaks out above the Donchian channel (20)', 'crosses above 20 from below',
    '1:3 risk reward', 'closes above the open', 'closes up on the day', 'an inside bar breaks to the upside',
    'the 50 SMA is rising'; 'closes back inside' (inside what?) is refused with a question."""
    # "crosses above 20 from below" / "crosses below 80 from above": the direction is already in the verb
    t = _sub_outside(r"(?i)\b(cross(?:es|ed|ing)?(?: back)? (?:above|over|up through) [^,;`]+?) from below\b", r"\1", t)
    t = _sub_outside(r"(?i)\b(cross(?:es|ed|ing)?(?: back)? (?:below|under|down through) [^,;`]+?) from above\b", r"\1", t)
    m = _msearch(r"\bcross(?:es|ed|ing)?(?: back)? (?:above|over|up through) [^,;`]+? from above\b|"
                 r"\bcross(?:es|ed|ing)?(?: back)? (?:below|under|down through) [^,;`]+? from below\b", t)
    if m:
        raise ParseError(f"'{m.group(0)}': a cross above comes from below (and a cross below from above); which one is it?")
    # "exit on the opposite cross": the entry's crossing, the other way
    mo = _msearch(r"\b(?P<v>sell|exit|close(?: out)?(?: the position)?|get out|cover)(?: it| the position)? (?:on|at|with|after|when "
                  r"(?:there is|we get|it makes)) (?:the |an |a )?(?:opposite|reverse|opposing|inverse) cross(?:over|ing)?\b", t)
    if mo:
        before = t[: mo.start()]
        dirs = re.findall(r"(?i)\bcross(?:es|ed|ing)?(?: back)? (above|over|up through|below|under|down through)\b", _mask(before, False))
        dirs += [("above" if x.lower().startswith("crossover") or x.lower() == "golden" else "below")
                 for x in re.findall(r"(?i)\b(crossover|crossunder|golden|death) cross\b|`[^`]*\b(crossover|crossunder)\(", before)
                 if x]
        ups = {("above" if d.lower() in ("above", "over", "up through") else "below") for d in dirs}
        if len(ups) != 1:
            raise ParseError(f"'{mo.group(0)}': the opposite of which cross? Say e.g. 'buy when the 10 day SMA crosses above "
                             "the 30 day SMA, sell when it crosses below it'.")
        way = "below" if ups == {"above"} else "above"
        _TL.opp = (mo.group(0), way)      # the exit reverses the entry's own cross (_exit_rule)
        t = t[: mo.start()] + f"{mo.group('v').split()[0]} when it crosses {way} it" + t[mo.end():]
        _note(f"'{mo.group(0)}' was read as the entry's crossing the other way: sell when it crosses {way} it (the first "
              "close on the other side of the line).")
    # "breaks out above the Donchian channel (20)", "closes above the upper Donchian channel (20)",
    # "breaks below the 20 day Donchian channel"
    def dc(m):
        n = m.group("n1") or m.group("n2") or m.group("n3")
        if not n:
            raise ParseError(f"'{m.group(0).strip()}': a Donchian channel of how many days? e.g. 'breaks out above the "
                             "Donchian channel (20)'.")
        up = m.group("d").lower() in ("above", "over", "out above", "out of", "to the upside", "through")
        _note(f"'{m.group(0).strip()}' = the close {'above the highest high' if up else 'below the lowest low'} of the "
              f"{n} bars before (the Donchian channel as of the previous bar, the Turtle breakout).")
        return f" donchian {n} day {'breakout' if up else 'breakdown'}"
    t = _sub_outside(r"(?i)(?:\b(?:when |if )?)?(?<=\b)(?:(?:the )?(?:price|close|it|stock) )?(?:breaks?(?: out)?|closes|moves|pushes|trades)"
                     r" (?P<d>out above|above|over|through|below|under|out of|down through) (?:the |its )?(?:(?P<n1>\d+)[- ]?(?:day|bar|period)? )?"
                     r"(?:upper |lower )?donchian(?: channel)?(?: (?:high|low|band|upper|lower))?"
                     r"(?: ?\((?P<n2>\d+)\)| (?P<n3>\d+)(?:[- ]?(?:day|bar|period)s?)?)?(?![\w(])", dc, t)
    # "1:3 risk reward", "risk-reward of 1:3", "1:3 RR": a target at 3R
    def rr(m):
        a, b = float(m.group("a") or m.group("c")), float(m.group("b") or m.group("d"))
        if a <= 0 or b <= 0:
            raise ParseError(f"'{m.group(0).strip()}': both sides of the ratio must be positive.")
        r_ = b / a
        _note(f"'{m.group(0).strip()}' = a target at {r_:g}R: {r_:g} x the initial risk (entry price - stop) above the entry.")
        return f" target {r_:g}R "
    t = _sub_outside(r"(?i)(?:\b(?:with |and |for |using )?(?:an? )?)?(?<![\d.:])(?P<a>\d+(?:\.\d+)?) ?: ?(?P<b>\d+(?:\.\d+)?) "
                     r"(?:risk[- /]?(?:to[- ])?reward|risk[- /]reward|r ?: ?r|rr|r/r)(?: ratio)?(?: target)?"
                     r"|\b(?:a |with a |and a )?(?:risk[- /]?(?:to[- ])?reward|r ?: ?r|rr)(?: ratio)?(?: of)? (?P<c>\d+(?:\.\d+)?) ?: ?(?P<d>\d+(?:\.\d+)?)(?![\d.])",
                     rr, t)
    # a green candle: "closes above the open", "closes higher than it opened"; "closes up on the day": vs the prior close
    def green(m):
        up = m.group("d").lower() in ("above", "higher than", "over")
        _note(f"'{m.group(0).strip()}' = {'close > open' if up else 'close < open'} (a {'green' if up else 'red'} candle: "
              "the close against the same day's open).")
        return f"`close {'>' if up else '<'} open`"
    t = _sub_outside(rf"(?i)\b(?:{_SUBJ} )?closes? (?P<d>above|higher than|over|below|lower than|under) (?:(?:the|its|today's|the day's) open"
                     r"(?:ing price)?|(?:where )?it opened|the opening price)\b", green, t)

    def upday(m):
        up = m.group("d").lower() in ("up", "higher", "green", "positive")
        _note(f"'{m.group(0).strip()}' = the close {'above' if up else 'below'} the previous close "
              f"(close {'>' if up else '<'} ref(close, 1)); for a candle that closes {'above' if up else 'below'} its own "
              f"open say 'closes {'above' if up else 'below'} the open'.")
        return f"`close {'>' if up else '<'} ref(close, 1)`"
    t = _sub_outside(rf"(?i)\b(?:{_SUBJ} )?(?:closes?|finishes|ends) (?P<d>up|higher|down|lower) (?:on|for) the (?:day|session)\b", upday, t)
    # "an inside bar breaks to the upside": today is inside yesterday's range; a buy stop above today's high works
    # in the next session (the breakout), as other stop entries do
    def inside(m):
        up = m.group("d").lower() in ("upside", "up", "high")
        _note(f"'{m.group(0).strip()}': the signal day is an inside bar (its high below the previous high and its low above "
              f"the previous low), and the entry is a {'buy' if up else 'sell'} stop order at the inside bar's "
              f"{'high' if up else 'low'}, working in the next session: it fills when the price breaks "
              f"{'above' if up else 'below'} it (at the level, or at the open if it gaps through).")
        return (f"`high < ref(high, 1) and low > ref(low, 1)` at a stop {'above' if up else 'below'} the "
                f"{'high' if up else 'low'}")
    t = _sub_outside(r"(?i)\b(?:an? |the )?inside (?:bar|day|candle) (?:breaks?(?: out)?|breakout) (?:to|on) the (?P<d>upside|downside)\b"
                     r"|\b(?:an? |the )?inside (?:bar|day|candle) breaks? (?P<d2>up|down)\b", lambda m: inside(_Dir(m)), t)
    # "closes back inside": inside what?
    m = _msearch(r"\b(?:closes?|moves?|comes?|gets?|is) back inside\b(?!\s+(?:its |the )?(?:\d+[- ]day )?(?:bollinger|keltner|donchian|band|channel|range))", t)
    if m:
        raise ParseError(f"'{m.group(0)}' - back inside what? Say which band, e.g. 'closes back below its upper Bollinger "
                         "band' (it was above the 20-day, 2-standard-deviation band and closes back under it), 'closes back "
                         "above its lower Bollinger band', or write the rule in backticks, e.g. "
                         "`ref(close, 1) > ref(bb_upper(20, 2), 1) and close < bb_upper(20, 2)`.")
    # "the 50 SMA is rising": a moving average with its length and no unit
    t = _sub_outside(r"(?i)\bthe (\d+) (sma|ema|wma|hma|ma|moving average)\b(?= (?:is |are )?(?:rising|falling|increasing|decreasing|"
                     r"declining|trending|sloping|pointing|turning|going|moving|above|below|over|under|crosses|>|<))",
                     r"the \1 day \2", t)
    return _signal_phrases_r12(t)


_MA_KIND = {"sma": "sma", "ema": "ema", "ma": "sma", "moving average": "sma", "simple moving average": "sma",
            "exponential moving average": "ema"}


def _signal_phrases_r12(t: str) -> str:
    """More trader phrases (TradingView review, round 12), each with a note:
    'at today's open' (the signal day's open), 'RSI(2) is below 10 hold 3 days' (no comma), '3 positions',
    'the 20 EMA of the weekly chart' / 'the monthly SMA(10)' (= the weekly 20 EMA / monthly 10 SMA), 'the weekly 20 EMA
    is rising' / 'its slope is positive' (week over week), 'the monthly 10 SMA is below the close' (the close above it),
    'heikin ashi turns green two days in a row'."""
    # "at today's open" / "at the open today": the signal day's open (entry_fill "open"), open-safe rules only
    def today_open(m):
        _note(f"'{m.group(0).strip()}' = at the open of the signal day: the order fills at that open, so the rule uses only "
              "what is known at the open - today's open and gap, and the previous close for everything else (a condition "
              "on the close, such as RSI, is the previous day's). Say 'at the next open' to decide at the close and buy "
              "the next morning.")
        return f"{m.group('pre') or ''}at the open"
    t = _sub_outside(r"(?i)(?P<pre>\b)(?:at|on) today'?s open(?:ing)?(?: price)?\b|(?P<pre2>\b)at the open today\b|\btoday at the open\b",
                     today_open, t)
    t = _sub_outside(r"(?i)\b(?:at|on) today'?s close(?: price)?\b|\bat the close today\b", "at the close", t)
    # "RSI(2) is below 10 hold 3 days": a holding period with no comma before it
    t = _sub_outside(r"(?i)(?<=[\w)%])(?<!\band)(?<!\bthen)(?<!\bor)(?<!\bto)(?<!\bbuy)(?<!\bmax)(?<!\bmaximum)(?<!\bmin)(?<!\bminimum)(?<!\bat least) "
                     r"(?=hold(?: it| them)?(?: for)? (?:\d+|one|two|three|four|five|a) (?:trading )?(?:days?|bars?|weeks?|sessions?)\b)",
                     ", ", t)
    # "3 positions" (a clause of its own): at most 3 positions at a time
    def npos(m):
        _note(f"'{m.group(0).strip(' ,')}' = at most {m.group('n')} positions at a time.")
        return f"{m.group('lead')}max {m.group('n')} positions"
    t = _sub_outside(r"(?i)(?P<lead>(?:^|[,;] ?)(?:with |using |in |across |and )?)(?P<n>\d+) (?:open |simultaneous |concurrent )?positions"
                     r"(?= *(?:[,;.]|$))", npos, t)
    # "the 20 EMA of the weekly chart", "SMA(10) on the monthly chart" -> "the weekly 20 EMA", "the monthly 10 SMA"
    t = _sub_outside(r"(?i)\b(?:the )?(?:(?P<n>\d+)[- ]?(?:period |bar )?(?P<k>sma|ema|ma|moving average)|(?P<k2>sma|ema)\s*\(\s*(?P<n2>\d+)\s*\))"
                     r" (?:of|on|from) the (?P<per>weekly|monthly|week|month) (?:chart|timeframe|bars?|candles?)\b",
                     lambda m: f"the {({'week': 'weekly', 'month': 'monthly'}).get(m.group('per').lower(), m.group('per').lower())} "
                               f"{m.group('n') or m.group('n2')} {(m.group('k') or m.group('k2'))}", t)
    t = _sub_outside(r"(?i)\b(weekly|monthly) (sma|ema)\s*\(\s*(\d+)\s*\)", r"\1 \3 \2", t)
    # "the weekly 20 EMA is rising" / "... slope is positive" / "the slope of the weekly 20 EMA is positive": week over week
    tf_ma = r"(?:the )?(?P<per>weekly|monthly) (?P<n>\d+)[- ]?(?:period |bar |week |month )?(?P<k>simple moving average|exponential moving average|moving average|sma|ema|ma)"

    def tf_trend(m):
        per, n, k = m.group("per").lower(), int(m.group("n")), _MA_KIND[m.group("k").lower()]
        up = m.group("d").lower() in ("rising", "up", "positive", "increasing", "sloping up", "pointing up", "turning up", "going up")
        f_ = "weekly" if per == "weekly" else "monthly"
        rule = f"{f_}({k}(close, {n}) {'>' if up else '<'} ref({k}(close, {n}), 1))"
        unit = "week" if per == "weekly" else "month"
        _note(f"'{m.group(0).strip()}' = the {per} {n} {k.upper()} (on completed {unit}ly bars) {'higher' if up else 'lower'} than "
              f"a {unit} before: {rule}, known from each {unit}'s last close and held through the next {unit}.")
        return f"`{rule}`"
    dirs = (r"(?P<d>rising|falling|increasing|decreasing|sloping up|sloping down|pointing up|pointing down|turning up|turning down|"
            r"going up|going down|positive|negative|up|down)")
    for pat in (rf"(?i)\bthe slope of {tf_ma} (?:is |turns? |stays? )?{dirs}\b",
                rf"(?i)\b{tf_ma}(?:'s)?(?: slope)? (?:is |are |has been |turns? |stays? )?{dirs}(?! (?:than|by)\b)\b"):
        t = _sub_outside(pat, tf_trend, t)
    # "the monthly 10 SMA is below the close" -> "the close is above the monthly 10 SMA" (the price is the subject)
    flip_w = {"below": "above", "under": "above", "above": "below", "over": "below"}
    t = _sub_outside(rf"(?i)\b(?P<ma>{tf_ma}) is (?P<rel>below|under|above|over) (?:the |its )?(?P<px>close|closing price|price)\b",
                     lambda m: f"the {m.group('px')} is {flip_w[m.group('rel').lower()]} {m.group('ma')}", t)
    # "heikin ashi turns green two days in a row": the N-th green candle in a row after a red one
    def ha_run(m):
        n = int(m.group("n"))
        green = m.group("col").lower() in ("green", "bullish", "white")
        op, nop = (">", "<=") if green else ("<", ">=")
        rule = f"count(ha_close {op} ha_open, {n}) == {n} and ref(ha_close, {n}) {nop} ref(ha_open, {n})"
        _note(f"'{m.group(0).strip()}' = {n} {'green' if green else 'red'} Heikin Ashi candles in a row after a "
              f"{'red' if green else 'green'} one (true on the {n}th): {rule}. Say 'heikin ashi is {m.group('col')} {n} days in a "
              "row' for any run of that length.")
        return f"`{rule}`"
    t = _sub_outside(r"(?i)\b(?:the )?(?:heikin[- ]?ashi|ha)(?: candles?| candlesticks?| bars?)? (?:turns?|turned|flips?|flipped|changes? to|changed to)"
                     r" (?P<col>green|red|bullish|bearish|white|black)(?: for)? (?P<n>\d+) (?:consecutive )?(?:days|bars|candles|sessions)"
                     r"(?: in a row| straight| consecutively)", ha_run, t)
    # "heikin ashi candles are green 2 days in a row" = "heikin ashi is green for 2 days in a row"
    t = _sub_outside(r"(?i)\b((?:the )?(?:heikin[- ]?ashi|ha)(?: candles?| candlesticks?| bars?)?) (?:is|are) (green|red|bullish|bearish|white|black)"
                     r" (\d+ (?:consecutive )?(?:days|bars|candles|sessions) (?:in a row|straight|consecutively))",
                     r"\1 is \2 for \3", t)
    return _signal_phrases_r13(t)


# ---- round 13 (a TradingView expert's review): clause boundaries and more trader phrases
_R13_EXIT_CLAUSE = (r"(?:(?:sell|exit|cover|close (?:the position|it|out)|get out|take profits?)(?: it| the position| everything| them| out| all)?"
                    r"|short|sell short|go short|go long)"
                    r"(?: (?:at|on) the (?:next )?(?:open|close))? (?:when(?:ever)?|if|once|as soon as)\b")
_R13_ACTION = (r"(?:buy|sell|short|cover|exit|close|get|go|enter|hold|keep|take|use|with|place|move|raise|trail|stop|"
               r"wait|at|on|in|after|max(?:imum)?|risk|put|invest|allocate|trade|scale|add|reverse|flip|\d)\b")
_R13_COND = r"\b(?:when(?:ever)?|if|once|while|as long as)\b"


def _signal_phrases_r13(t: str) -> str:
    """Round 13 (TradingView review): clause boundaries that a missing comma used to merge into the entry as AND, and
    more trader phrases, each with a note:
      'buy when A sell when B' / 'exit when' / 'cover when' / 'take profit when' / 'short when'  -> a new clause
      'when A then B'  (a sequence, or both on the same day?)                                  -> refused, with a question
      'sell X when A, buy back when B'  (a short, or a holder's exit and re-entry?)            -> refused, with a question
      'otherwise sell' (out whenever the entry rule is false)  -> 'sell when the signal ends'
      'on a 20 day breakout' -> the Donchian breakout; 'the 3 day average of the RSI(2)' -> sma(rsi(close, 2), 3);
      'the 20 day linear regression slope is positive'; 'ichimoku tenkan crosses above kijun';
      'the price touches the lower keltner channel' (low <= keltner_lower)."""
    mk = _mask(t, parens=False)
    # 'sell AAPL when ..., buy back when ...': a short, or a holder selling and buying back?
    mb = re.match(r"(?is)\s*sell\s+(?!short\b|to\b|it\b|when\b|if\b)(?P<who>[^,;]*?)\s+(?P<a>(?:when(?:ever)?|if|once)\b[^;]*?)"
                  r"\s*[,;]?\s+(?:and\s+)?(?:then\s+)?(?:buy|purchase)\s+(?:it\s+|them\s+|\S+\s+)?back\s+(?P<b>(?:when(?:ever)?|if|once)\b.*)$", mk)
    if mb:
        who, a_, b_ = (t[mb.start(g):mb.end(g)].strip(" ,.;") for g in ("who", "a", "b"))
        raise ParseError(f"'sell {who} {a_}, buy back {b_}' can be read two ways: a short sale (say 'short {who} {a_}, "
                         f"cover {b_}') or a position you hold, sold and bought back later (say 'buy {who} {b_}, sell "
                         f"{a_}'). Which one?")
    # an exit (or short) clause with no comma before it: "buy AAPL when RSI(2) is below 50 sell when RSI(2) is above 80"
    # is two rules, never "RSI < 50 and RSI > 80"
    pos = [m.start() for m in re.finditer(rf"(?i)(?<=[\w)%`\]])(?<!\band)(?<!\bthen)(?<!\bto)(?<!\bor)(?<!\bnot)(?<!\bbuy)(?<!\bgo)(?<!\bsell)"
                                          rf"(?= +{_R13_EXIT_CLAUSE})", mk)]
    for p in reversed(pos):
        if re.search(_R13_COND, mk[:p], re.I):
            t = t[:p] + "," + t[p:]
    # "take profit when RSI(2) is above 70": a rule exit ("take profit at 10%" stays a target)
    t = _sub_outside(r"(?i)\btake profits? (?=(?:when(?:ever)?|if|once|as soon as)\b)", "sell ", t)
    # "when A then B": a sequence (A, and B on a later day) or both at once? Only "then" + an action is a new clause.
    mk = _mask(t, parens=False)
    for m in re.finditer(r"(?i)\bthen\b", mk):
        seg = re.split(r"[,;]", mk[: m.start()])[-1]
        after = mk[m.end():].lstrip(" ,:")
        if re.search(_R13_COND, seg, re.I) and after and not re.match(rf"(?i){_R13_ACTION}", after):
            a_ = re.split(_R13_COND, t[m.start() - len(seg): m.start()], flags=re.I)[-1].strip()
            b_ = re.split(r"[,;]", t[m.end():])[0].strip(" ,:")
            raise ParseError(f"'{a_} then {b_}': is that a sequence (first {a_}, then {b_} on a later day) or both on the "
                             f"same day? For both, say '{a_} and {b_}'. For a sequence, write it in backticks, e.g. "
                             f"`ref(A, 1) and B` (A the day before, B today) or `bars_since(A) <= 5 and B` (A within the "
                             f"last 5 days, then B), with A and B in the rule language (see --help-expr).")

    # "otherwise sell" / "else exit": out on the first close the entry rule is false
    def otherwise(m):
        _note(f"'{m.group(0).strip(' ,;')}' was read as: sell at the close of the first day the entry rule is no longer "
              "true (in the market while it holds, out otherwise). For a different exit, say it, e.g. 'sell when RSI(2) "
              "is above 70'.")
        return f"{m.group('lead')}sell when the signal ends"
    t = _sub_outside(r"(?i)(?P<lead>[,;]\s*|\s+)(?:and\s+)?(?:otherwise|else|or else)\s*,?\s*(?:sell|exit|get out|close the position|"
                     r"go to cash|move to cash|be in cash|stay out|stay in cash)(?: it| the position| everything)?(?=\s*(?:[,;.]|$))",
                     otherwise, t)
    m = _msearch(r"\b(?:otherwise|or else)\b", t)
    if m:
        raise ParseError(f"'{t[m.start():].split(',')[0].strip()}': otherwise what? For an exit whenever the entry rule is "
                         "false, say 'otherwise sell'; for a long/short reversal, say 'buy when ..., short when ...'.")

    # "on a 20 day breakout" / "a 55-day breakdown": the Donchian breakout of the close (Turtle rules)
    def nday_bo(m):
        n, up = m.group("n"), m.group("k").lower().replace(" ", "") in ("breakout", "highbreakout")
        _note(f"'{m.group(0).strip()}' = the close {'above the highest high' if up else 'below the lowest low'} of the {n} "
              f"bars before ({'close > ref(highest(high, ' if up else 'close < ref(lowest(low, '}{n}), 1)), a Donchian "
              f"{'breakout' if up else 'breakdown'}.")
        return f"{m.group('pre') or ''}donchian {n} day {'breakout' if up else 'breakdown'}"
    t = _sub_outside(r"(?i)(?<!donchian )(?<!channel )(?P<pre>\b(?:on|after|with) (?:an? |the )?|\b(?:an? |the ))?"
                     r"(?<![\w-])(?P<n>\d+)[- ](?:day|bar|session)s? (?P<k>breakout|break out|breakdown|break down|high breakout|low breakdown)\b",
                     nday_bo, t)

    # "the 3 day average of the RSI(2)" / "the 3 day moving average of the 2 day RSI": sma(rsi(close, 2), 3)
    def avg_rsi(m):
        k, given = m.group("k"), m.group("n1") or m.group("n2") or m.group("n3")
        n = given or "14"
        kind = "ema" if (m.group("kind") or "").lower() in ("exponential", "ema") or m.group("ma").lower() == "ema" else "sma"
        rule = f"{kind}(rsi(close, {n}), {k})"
        _note(f"'{m.group(0).strip()}' = {rule}: the {k}-day {'exponential ' if kind == 'ema' else ''}average of the "
              f"{n}-day RSI" + ("" if given else " (no RSI period given: 14)") + ".")
        return f"`{rule}`"
    t = _sub_outside(r"(?i)\b(?:the |its )?(?P<k>\d+)[- ](?:day|bar|period|session)s? (?:(?P<kind>simple|exponential) )?"
                     r"(?P<ma>(?:moving )?average|mean|sma|ema|ma) of (?:the |its )?(?:(?P<n1>\d+)[- ](?:day|bar|period) )?"
                     r"rsi(?:\s*\(\s*(?:close\s*,\s*)?(?P<n2>\d+)\s*\)| (?P<n3>\d+)(?![\d.%]| ?(?:day|bar|week|month)))?", avg_rsi, t)

    # "the 20 day linear regression slope is positive" (TradingView's ta.linreg: the least-squares line through the last
    # 20 closes; its slope is the line's value today minus its value one bar earlier)
    def lr_slope(m):
        n = m.group("n1") or m.group("n2")
        if not n:
            raise ParseError(f"'{m.group(0).strip()}': the linear regression over how many days? e.g. 'the 20 day linear "
                             "regression slope is positive'.")
        up = m.group("d").lower() in ("positive", "rising", "up", "above 0", "above zero", "greater than 0", "sloping up", "pointing up")
        rule = f"linreg(close, {n}, 0) {'>' if up else '<'} linreg(close, {n}, 1)"
        _note(f"'{m.group(0).strip()}' = the slope of the least-squares line through the last {n} closes is "
              f"{'positive' if up else 'negative'}: {rule} (TradingView's ta.linreg(close, {n}, 0) - ta.linreg(close, {n}, 1)).")
        return f"`{rule}`"
    t = _sub_outside(r"(?i)\b(?:the |its )?(?:(?P<n1>\d+)[- ](?:day|bar|period|session)s? )?(?:linear regression|linreg|lin reg)"
                     r"(?: line)?(?:\s*\(\s*(?P<n2>\d+)\s*\))?(?:'s)? slope (?:is |turns |stays |has turned )?"
                     r"(?P<d>positive|negative|rising|falling|up|down|above (?:0|zero)|below (?:0|zero)|greater than 0|less than 0|"
                     r"sloping up|sloping down|pointing up|pointing down)\b", lr_slope, t)
    # "ichimoku tenkan crosses above kijun": the word 'ichimoku' in front of the line names
    t = _sub_outside(r"(?i)\b(?:the )?ichimoku(?:'s)? (?=(?:tenkan|kijun|conversion line|base line)\b)", "", t)
    t = _sub_outside(r"(?i)\b(?:the )?ichimoku (?=(?:kijun|base line)\b)", "", t)

    # "the price touches the lower keltner channel" (the day's low at or below it) / "the upper" (the high at or above)
    def touch(m):
        up = m.group("side").lower() == "upper"
        n, k = m.group("n1") or m.group("n2") or "20", m.group("k") or "2"
        kel = m.group("band").lower().startswith("keltner")
        fn = ("keltner" if kel else "bb") + ("_upper" if up else "_lower")
        name = "Keltner channel" if kel else "Bollinger band"
        rule = f"{'high >=' if up else 'low <='} {fn}({n}, {k})"
        _note(f"'{m.group(0).strip()}' = the day's {'high at or above' if up else 'low at or below'} the "
              f"{'upper' if up else 'lower'} {name} ({n}, {k}): {rule}. Say 'closes {'above' if up else 'below'} the "
              f"{'upper' if up else 'lower'} {name}' to use the close.")
        return f"`{rule}`"
    t = _sub_outside(r"(?i)\b(?:it |the price |price |the stock )?(?:touches|touched|tags|tagged|hits|hit|reaches|reached)"
                     r" (?:the |its )?(?:(?P<n1>\d+)[- ](?:day|bar|period) )?(?P<side>upper|lower) (?P<band>keltner(?: channel)?(?: band| line)?|"
                     r"bollinger(?: band)?)(?:\s*\(\s*(?P<n2>\d+)\s*(?:,\s*(?P<k>\d+(?:\.\d+)?)\s*)?\))?", touch, t)
    return t


def _r13_candle_takes(take, c: str, o: str) -> None:
    """Candle colour as TradingView users mean it (round 13): 'closes red' / 'a red candle' / 'a bearish candle' =
    close < open (the close against the same day's open); green / bullish = close > open; 'closes red 3 days in a row'
    / '3 red candles in a row' = the last 3 candles all red. 'closes down' / 'a down day' stay against the previous
    close."""
    col = r"(?P<col>red|green|bearish|bullish)"
    run = r"(?: (?:for|on))?(?: at least)? (?P<n>\d+) (?:straight |consecutive |trading )?(?:days|bars|sessions|times|candles)(?: in a row| straight| consecutively)"

    def cond(m):
        return f"{c} {'<' if m.group('col') in ('red', 'bearish') else '>'} {o}"

    def note(m, rule):
        red = m.group("col") in ("red", "bearish")
        _note(f"'{m.group(0).strip()}' = {rule} (a {'red' if red else 'green'} candle, as TradingView colours it: the close "
              f"{'below' if red else 'above'} the same day's open). For a close {'below' if red else 'above'} the previous "
              f"close, say 'closes {'down' if red else 'up'}'.")
        return rule

    def streak(m):
        n = int(m.group("n"))
        return note(m, f"count({cond(m)}, {n}) == {n}")
    take(rf"(?:closes?|closed|finish(?:es|ed)?|ends?|ended|prints?|printed) {col}{run}", streak)
    take(rf"(?:at least )?(?P<n>\d+) (?:straight |consecutive )?{col} (?:candles?|candlesticks?|bars?)(?: in a row| straight| consecutively)?", streak)
    take(rf"(?:closes?|closed|finish(?:es|ed)?|ends?|ended) {col}(?: on the day| for the day)?", lambda m: note(m, cond(m)))
    take(rf"(?:(?:it |the day |today )?(?:is|was|prints?|printed|forms?|formed|makes?|made) )?(?:an? |the )?{col} (?:candles?|candlesticks?|bars?)",
         lambda m: note(m, cond(m)))


def _r13_osc_zone(text: str) -> str:
    """'the RSI(14) is oversold' / 'RSI 2 is overbought' / 'the 10 day RSI is oversold': the RSI named, with the usual
    zones (below 30 / above 70); without a period, the 14 day RSI."""
    def zone(m):
        n = m.group("n1") or m.group("n2") or m.group("n3") or "14"
        over = m.group("z").lower() == "overbought"
        _note(f"Warning: '{m.group(0).strip()}' has no single definition; it was read as the {n} day RSI "
              f"{'above 70' if over else 'below 30'} (the usual RSI zones). Say e.g. 'RSI({n}) is "
              f"{'above 80' if over else 'below 20'}' for another level.")
        return f"{m.group('lead')}{n} day RSI is {'above 70' if over else 'below 30'}"
    return _sub_outside(r"(?i)(?P<lead>^|\s+)(?:the |its )?(?:(?P<n1>\d+)[- ](?:day|period|bar|session) )?rsi(?:\s*\(\s*(?:close\s*,\s*)?(?P<n2>\d+)\s*\)"
                        r"| (?P<n3>\d+)(?![\d.%]))? (?:is|are|gets|becomes|turns) (?P<z>oversold|overbought)\b", zone, text)


def _r13_exit_refs(wl: str, entry: str, notes: list[str]) -> str:
    """References an exit makes back to the entry (round 13): 'the histogram turns negative' after a MACD-histogram
    entry (that histogram), 'it crosses above the mean' (the entry's moving average / Bollinger or Keltner middle /
    z-score mean; refused when the entry has none or several)."""
    if re.search(r"(?i)\bhistogram\b", wl) and not re.search(r"(?i)\bmacd\b", wl):
        calls = list(dict.fromkeys(re.findall(r"macd_hist\([^()]*\)", entry)))
        if len(calls) != 1:
            raise ParseError(f"'{wl.strip()}': which histogram? " + (f"The entry uses {' and '.join(calls)}. " if calls else
                             "The entry has no MACD histogram. ") + "Name it, e.g. 'sell when the MACD histogram turns negative'.")
        h = calls[0]

        def hist(m):
            pos = m.group("d").lower().startswith(("positive", "above", "over"))
            moving = re.match(r"(?i)turn|go|flip|become|cross|rise|fall|drop|move", m.group("v"))
            e = f"{'crossover' if pos else 'crossunder'}({h}, 0)" if moving else f"{h} {'>' if pos else '<'} 0"
            notes.append(f"'{m.group(0).strip()}' = the entry's MACD histogram ({h}): {e}.")
            return f" `{e}` "
        wl = _sub_outside(r"(?i)\b(?:the |its )?histogram (?P<v>turns?|goes|flips?|becomes|crosses|is|stays|rises|falls|drops|moves)"
                          r"(?: back)?(?: to)? (?P<d>positive|negative|above 0|below 0|under 0|over 0)\b", hist, wl)
    mm = _msearch(r"(?:\b(?:it|the price|price|the close)(?:'s| is)? )?\b(?P<v>crosses|crossed|closes|closed|is|rises|moves|goes|gets|falls|drops|trades)"
                  r"(?: back)? (?P<rel>above|over|below|under) (?:the |its )?(?P<w>mean|midline|mean line)\b", wl)
    if mm:
        cands = []
        for fn, n in re.findall(r"\b(bb_lower|bb_upper|zscore|keltner_lower|keltner_upper|sma|ema)\((?:close, )?(\d+)", entry):
            c = f"ema(close, {n})" if fn.startswith("keltner") or fn == "ema" else f"sma(close, {n})"
            if c not in cands:
                cands.append(c)
        if len(cands) != 1:
            raise ParseError(f"'{mm.group(0).strip()}': which mean? " + (f"The entry uses {' and '.join(cands)}. " if cands else
                             "The entry has no moving average or band to take it from. ") + "Name it, e.g. 'sell when it "
                             "crosses above its 20 day moving average'.")
        up = mm.group("rel").lower() in ("above", "over")
        e = (f"{'crossover' if up else 'crossunder'}(close, {cands[0]})" if mm.group("v").lower().startswith("cross")
             else f"close {'>' if up else '<'} {cands[0]}")
        notes.append(f"Warning: 'the {mm.group('w')}' was read as the entry's mean, {cands[0]}: {e}.")
        wl = wl[: mm.start()] + f" `{e}` " + wl[mm.end():]
    return wl


class _Dir:
    """A match whose 'd' group falls back to 'd2' (two spellings of one phrase)."""

    def __init__(self, m):
        self.m = m

    def group(self, k=0):
        if k == "d":
            return self.m.group("d") or self.m.group("d2")
        return self.m.group(k)


def parse_signal(text: str, holding: bool = False) -> Strategy:
    raw = text
    t = _normalize(text)
    t = _sub_outside(r"(?i)\bbuy(?:ing)? to (?:cover|close)\b", "cover", t)   # closing a short is an exit, not an entry
    _negative_lookbacks(t)
    # "sell Friday at the close" -> "sell on Friday at the close"
    t = _sub_outside(r"(?i)\b((?:sell|exit|cover|close out|get out)\w*(?: it| the position| everything)?) (?=(?:every |each )?(?:monday|tuesday|wednesday|thursday|friday)s?\b)",
                     r"\1 on ", t)
    t = _sub_outside(r"(?i)\bon (?:every|each) (monday|tuesday|wednesday|thursday|friday)\b", r"on \1", t)
    # "sell on the close" / "exit on the next open": the exit's timing, as "at the close"
    t = _sub_outside(r"(?i)\b((?:sell|exit|cover|close out|get out)\w*(?: it| the position| everything)?) on the (next |following )?(open|close)\b",
                     r"\1 at the \2\3", t)
    # "go long when the supertrend flips to up, go short when it flips to down": 'it' is the long entry's signal
    mst = re.search(r"(?i)\b(?:the )?(supertrend(?:\s*\(\s*\d+\s*,\s*\d+(?:\.\d+)?\s*\))?|heikin[- ]ashi(?: candles?)?)(?:'s)?(?: direction| trend)? "
                    r"(?:turns?|flips?|flipped|switches|changes|changed|goes)\b", t)
    if mst:
        t = _sub_outside(r"(?i)\b((?:go|going|went) short|short(?: it)?|sell short|go long|buy(?: it)?)( [^,;]*?)? (when|if|once) it "
                         r"(?=(?:flips?|flipped|turns?|turned|switches|changes|changed)\s+(?:to\s+)?(?:up|down|bullish|bearish|green|red)\b)",
                         lambda m: f"{m.group(1)}{m.group(2) or ''} {m.group(3)} the {mst.group(1)} ", t)
    # "buy SPY at the open on Monday if it closed down on Friday": the weekday is a condition of the entry
    t = _sub_outside(r"(?i)\b((?:buy|short|go long|go short|sell short)\b[^,;]*?) on (monday|tuesday|wednesday|thursday|friday)s?"
                     r"((?: at the (?:next )?(?:open|close))?) (if|when|whenever|provided)\b", r"\1\3 \4 on \2 and", t)
    t = _signal_phrases(t)
    # "sell when it falls 10% from its peak": a trailing stop from the highest price since entry
    mpk = _msearch(r"\b(?:and |then )?(?:sell|exit|get out|close (?:it|the position|out))(?: it| the position| them)?"
                   r"(?: at the close| at the next open)? (?:when|if|once|as soon as) (?:it|the price|the stock|the position|price|they)"
                   r"(?: has| have| is| are)? (?:falls?|fallen|drops?|dropped|declines?|declined|down|pulls? back|pulled back|"
                   r"retraces?|retraced|comes? off|came off)(?: by)?(?: more than| at least)? (\d+(?:\.\d+)?)% "
                   r"(?:from|off|below|under) (?:its|the|their) (?:peak|high|highest (?:price|high|close|point)|top)"
                   r"(?: since (?:the )?(?:entry|purchase|we bought|buying|it was bought|it was purchased|entering))?"
                   r"(?P<end>\s*(?:[,;.]|$))?", t, parens=False)
    if mpk:
        if mpk.group("end") is None:
            raise ParseError(f"'{mpk.group(0).strip()}' is a {mpk.group(1)}% trailing stop; give it its own clause, e.g. "
                             f"'sell when RSI(2) > 70, with a {mpk.group(1)}% trailing stop'.")
        t = t[: mpk.start()] + f" with a {mpk.group(1)}% trailing stop" + mpk.group("end") + t[mpk.end():]
        _note(f"'{mpk.group(0).strip(' ,;.')}' was read as a {mpk.group(1)}% trailing stop: sold during the day as soon as "
              f"the price falls {mpk.group(1)}% below the highest high since entry (at the open if it gaps below). For a "
              f"check at the close only, write `close <= {1 - float(mpk.group(1)) / 100:g} * highest_since_entry`.")
    # "on QQQ, go long when ..." / "for SPY: buy when ..." -> the tickers belong to the entry
    mon = re.match(r"(?is)\s*(?:on|for|with|trading|trade|using)\s+(?P<who>[^,;:`]+?)\s*[,;:]\s*(?P<rest>.+)$", t)
    if mon and find_tickers(mon.group("who"), strict=True):
        who = mon.group("who")
        left = who
        for tk in find_tickers(who, strict=True):
            left = re.sub(rf"(?i)(?<![\w])[\$^]?{re.escape(tk.lstrip('^'))}\b", " ", left)
        for name in COMPANIES:
            left = re.sub(rf"(?i)\b{re.escape(name)}\b", " ", left)
        mv = re.match(ENTRY_VERB, mon.group("rest").strip(), flags=re.I)
        if mv and not re.sub(r"(?i)\b(?:and|the|shares?|stock|of)\b|[,&]", "", left).strip():
            rest = mon.group("rest").strip()
            t = f"{rest[: mv.end()]} {who}{rest[mv.end():]}"
    t = _pine_backticks(t)
    T = Text(t)
    notes: list[str] = []
    broker = _broker_costs(T, notes)
    kw = common_options(T, notes)
    kw.update(broker)
    m = T.find(r"(?:(?:with|and) )?(?:no|without|ignor(?:e|ing)|exclud(?:e|ing)) (?:cash )?dividends|dividends (?:ignored|excluded|not credited)"
               r"|(?:(?:with|using) )?price[- ]only(?: returns)?")
    if m:
        kw["dividends"] = False
        if "price" in m.group(0).lower() and "cash_rate" not in kw:
            kw["cash_rate"] = None
        if not getattr(_TL, "tv", False):
            notes.append("No dividends: cash dividends are not credited to long positions (nor charged to shorts), so the "
                         "positions earn their price change only.")
    elif T.find(r"(?:with|and|including|plus) (?:cash )?dividends(?: (?:credited|included|paid))?|(?:credit|includ)(?:e|es|ing) (?:cash )?dividends"):
        kw["dividends"] = True
    if getattr(_TL, "tv", False) and "cash_rate" not in kw:
        kw["cash_rate"] = None     # TradingView-compatible mode: no interest unless asked for ("with interest on cash")

    # ---- sizing and portfolio options
    m = T.find(r"(?:max(?:imum)?(?: of)?|up to|at most|no more than|hold at most|limit(?:ed)? to) (\d+) (?:open |simultaneous |concurrent )?(?:positions?|stocks?|names?|holdings?|trades?)(?: at (?:a|any|one) time| at once)?|(\d+) (?:positions|stocks|names) max(?:imum)?")
    if m:
        kw["max_positions"] = int(m.group(1) or m.group(2))
    m = T.find(rf"risk(?:ing)? {NUM}% (?:of (?:the )?(?:equity|capital|account) )?(?:per|on each|each) trade")
    if m:
        kw["sizing"], kw["risk_per_trade"] = "risk", float(m.group(1)) / 100
    m = T.find(rf"(?:put |invest |allocate |using |with |use )?{NUM}% (?:of (?:the )?(?:equity|capital|account|portfolio) )?(?:per|in each|for each|each|on each|to each) (?:trade|position|stock|name|entry)")
    if not m:
        m = T.find(rf"(?:put |invest |allocate |using |with |use )?{NUM}% (?:each|apiece)\b")
    if m:
        kw["position_size"] = float(m.group(1)) / 100
    m = T.find(rf"(?:with |using |at )?{NUM} ?x (?:leverage|leveraged|margin)|leverage of {NUM}x?|{NUM}:1 (?:leverage|margin)")
    if m:
        kw["leverage"] = float(m.group(1) or m.group(2) or m.group(3))
    m = T.find(rf"(?:size (?:each )?positions? (?:for|to|at)|target(?:ing)?|volatility target(?: of)?) {NUM}% (?:annual(?:ized)? )?volatility|volatility target(?:ing)?(?: of)? {NUM}%")
    if m:
        kw["sizing"], kw["target_vol"] = "volatility", float(m.group(1) or m.group(2)) / 100
    m = T.find(r"\$(\d+(?:\.\d+)?) (?:per|in each|for each|each|on each) (?:trade|position|stock|entry)")
    if m:
        kw["sizing"], kw["fixed_amount"] = "fixed_dollars", float(m.group(1))
    m = T.find(r"(\d+) shares (?:per|each|in each|for each) (?:trade|position|entry)|(?:buy|trade) (\d+) shares")
    if m:
        kw["sizing"], kw["fixed_amount"] = "fixed_shares", float(m.group(1) or m.group(2))
    if T.find(r"(?:(?:with|using|in) )?whole shares(?: only)?|no fractional shares"):
        kw["fractional_shares"] = False
    elif T.find(r"(?:(?:with|using|allow(?:ing)?) )?fractional shares(?: allowed)?"):
        kw["fractional_shares"] = True
    m = T.find(rf"(?:no more than|at most|max(?:imum)?|cap(?:ped)? at|limit(?:ed)? to) {NUM}% of (?:the )?(?:day's |daily |average )?(?:volume|adv)")
    if m:
        kw["max_volume_pct"] = float(m.group(1)) / 100
    m = T.find(rf"(?:(?:with|and|paying) )?(?:{BORROW_FEE})")
    if m:
        kw["borrow_fee"] = float(m.group(1) or m.group(2)) / 100
    elif T.find(NO_BORROW_FEE):
        kw["borrow_fee"] = 0.0
    m = T.find(rf"(?:(?:with|and|paying) )?(?:an? )?{NUM}% margin (?:interest|rate)|(?:(?:with|and|paying) )?(?:a )?margin (?:interest|rate)(?: of|:)? {NUM}%")
    if m:
        kw["margin_rate"] = float(m.group(1) or m.group(2)) / 100
    if T.find(r"(?:(?:with|using|on|in|and) )?(?:an? )?portfolio[- ]margin(?:ing)?(?: account)?"):
        kw["margin_account"] = "portfolio"
    m = T.find(rf"(?:(?:with|and) )?(?:a )?{NUM}% maintenance(?: margin)?(?: requirement)?|maintenance margin(?: requirement)?(?: of)? {NUM}%")
    if m:
        kw["maintenance_margin"] = float(m.group(1) or m.group(2)) / 100
    if T.find(r"(?:(?:with|and) )?(?:no|without|ignore|ignoring) margin calls?"):
        kw["maintenance_margin"] = 0.0
    if getattr(_TL, "tv", False):
        # TradingView-compatible mode: TradingView's strategy tester charges no borrow fee on shorts and (at its
        # default 100% margin) makes no maintenance-margin calls; either applies here only when the sentence states it
        kw.setdefault("borrow_fee", 0.0)
        kw.setdefault("maintenance_margin", 0.0)
    m = T.find(rf"(?:(?:with|and) )?(?:a )?short rebate(?: spread)?(?: of)? {NUM}% (?:below|under|less than) (?:the )?(?:t-?bill|cash)(?: rate)?"
               rf"|(?:(?:with|and) )?(?:no|full) short rebate(?: haircut)?")
    if m:
        # "no short rebate": proceeds earn nothing (a spread larger than any rate); "full": the whole cash rate
        kw["short_rebate_spread"] = float(m.group(1)) / 100 if m.group(1) else (1.0 if "no" in m.group(0).lower().split() else 0.0)
    m = T.find(r"pyramid(?:ing)?(?: up to)? (\d+)(?: times| entries)?|(?:add to (?:the )?(?:position|winners)|scale in)(?: up to)? (\d+) times|up to (\d+) entries per (?:ticker|stock|position)")
    if m:
        kw["pyramiding"] = int(m.group(1) or m.group(2) or m.group(3))
    # ranking: by market cap ("rank by market cap", "prefer the largest"), or by a named indicator
    mcap = T.find(r"(?:prefer(?:ring)?|rank(?:ed|ing)?(?: them)? by|pick(?:ing)?|choose|choosing|favou?r(?:ing)?) (?:the )?"
                  r"(?:(largest|biggest|smallest|highest|lowest) )?market[- ]cap(?:itali[sz]ation)?s?(?: first)?"
                  r"|(?:prefer(?:ring)?|pick(?:ing)?|choose|choosing|favou?r(?:ing)?) (?:the )?(largest|biggest|smallest)"
                  r"(?: (?:companies|stocks|names|ones))?(?: (?:by|in) market[- ]cap(?:itali[sz]ation)?)?(?: first)?(?=\s*(?:[,;.]|$))")
    # "prefer the lowest RSI", "rank by lowest RSI(2)", "prefer the highest 20-day volatility", "rank by the lowest
    # 5 day return": group 1 the direction, 3 the indicator, 2/4/5 its period (before it, in brackets, after it)
    mr = T.find(r"(?:prefer(?:ring)?|rank(?:ed|ing)?(?: them)? by|pick(?:ing)?|choose|choosing|favou?r(?:ing)?) (?:the )?"
                r"(lowest|highest|weakest|strongest|biggest losers?|biggest gainers?|most oversold|most overbought|biggest declines?|largest declines?)"
                r"(?: (?:(\d+)[- ]?(?:day|bar|period|session)s? )?(rsi|returns?|declines?|change|volatility)"
                r"(?:\s*\(\s*(?:close\s*,\s*)?(\d+)\s*\)|\s+(\d+)(?![\d.%]| ?(?:day|bar|week|month|year|%)))?)?(?: first)?")

    # ---- exits and stops (anywhere)
    ex: dict = {}
    before_exits = T.rest
    for m in T.findall(rf"(?:sell|exit|close|take profits? on|take) (?:\d+% |{NUM}% )?(?:of (?:the )?(?:position|shares) )?(?:at|when (?:it(?:'s| is)? )?up) \+?{NUM}%(?: (?:gain|profit|up))?"):
        pass  # handled below via scale-out regex
    so = []
    for m in re.finditer(rf"(?:sell|exit|close|take profits? on|take) {NUM}% (?:of (?:the )?(?:position|shares) )?(?:at|when (?:it(?:'s| is)? )?up) "
                         rf"\+?{NUM} ?(?:r|x (?:the )?(?:initial )?risk|times (?:the )?(?:initial )?risk)(?![a-z0-9])", T.low):
        so.append({"fraction": _snap_fraction(float(m.group(1)) / 100), "r": float(m.group(2))})
        T.blank(m.group(0))
    if so:
        notes.append("R-multiple scale-out: " + "; ".join(f"sell {x['fraction']:.4g} of what is left at {x['r']:g}R" for x in so)
                     + " - R is the initial risk (entry price - initial stop), measured when the position opens.")
    for m in re.finditer(rf"(?:sell|exit|close|take profits? on|take) {NUM}% (?:of (?:the )?(?:position|shares) )?(?:at|when (?:it(?:'s| is)? )?up) \+?{NUM}%", T.low):
        so.append({"fraction": _snap_fraction(float(m.group(1)) / 100), "at": float(m.group(2)) / 100})
        T.blank(m.group(0))
    # breakeven stop after an R multiple: "breakeven after 1R", "move the stop to breakeven at 1R"
    m = T.find(rf"(?:(?:and|then|with a|use a|plus) )?(?:(?:(?:move|moving|raise|raising|set|put)s? (?:the |my |a )?(?:stop|stop[- ]loss) (?:up )?to )?"
               rf"(?:break[- ]?even|the entry(?: price)?|entry(?: price)?|cost)|break[- ]?even stop(?:[- ]loss)?) "
               rf"(?:after|once|when|if|at)(?: (?:it(?:'s| is| has)?|the (?:trade|position|price) is|price is|we are|we're))?"
               rf"(?: (?:up|gained|risen|in profit|ahead))?(?: by)? \+?{NUM} ?(?:r|x (?:the )?(?:initial )?risk)(?![a-z0-9])(?: (?:in profit|gain|up))?")
    if m:
        ex["breakeven_r"] = float(m.group(1))
        notes.append(f"Breakeven stop: once the best price since entry is {m.group(1)}R in favour (R = the initial risk, entry "
                     "price - initial stop; from a bar's high, or low for a short), a stop at the entry price applies from the "
                     "next bar; it fills at the entry price, or at the open if the price gaps through it.")
    # breakeven stop: "move the stop to breakeven after +2%", "breakeven stop once up 3%"
    m = T.find(rf"(?:(?:and|then|with a|use a|plus) )?(?:(?:move|moving|raise|raising|set|put|trail)s? (?:the |my |a )?(?:stop|stop[- ]loss) (?:up )?to "
               rf"(?:break[- ]?even|the entry(?: price)?|entry(?: price)?|cost)|break[- ]?even stop(?:[- ]loss)?) "
               rf"(?:after|once|when|if|at)(?: (?:it(?:'s| is| has)?|the (?:trade|position|price) is|price is|we are|we're))?"
               rf"(?: (?:up|gained|risen|in profit|in the money|ahead))?(?: by)? \+?{NUM}%(?: (?:gain|profit|up|in profit|higher))?")
    if m:
        ex["breakeven_after"] = float(m.group(1)) / 100
        notes.append(f"Breakeven stop: once the best price since entry is {m.group(1)}% in favour (from a bar's high, or low for a "
                     "short), a stop at the entry price applies from the next bar; it fills at the entry price, or at the open "
                     "if the price gaps through it.")
    elif re.search(r"\bbreak[- ]?even\b", T.rest, re.I) and not ex.get("breakeven_r"):
        raise ParseError("Breakeven stop: after how much gain? Say e.g. 'move the stop to breakeven after +2%'.")
    if so:
        ex["scale_out"] = so
    # "trail the rest (with an 8% trailing stop)": after the first scale-out, a trailing stop on the remainder
    m = T.find(rf"(?:(?:and|then) )?trail(?:ing)? (?:the )?(?:rest|remainder|remaining (?:shares|position|stake|half|\d+%))"
               rf"(?: of (?:the |it|the position|the shares))?(?: (?:with|using|on|by|at) (?:a |an )?(?:(?P<pct>\d+(?:\.\d+)?)%"
               rf"(?: trailing stop(?:[- ]loss)?)?|(?P<atr>\d+(?:\.\d+)?) ?(?:x )?atrs?(?: (?:trailing|chandelier) stop)?))?")
    if m:
        ex["trail_after_scale_out"] = True
        if m.group("pct"):
            ex["trailing_stop"] = float(m.group("pct")) / 100
        if m.group("atr"):
            ex["trailing_atr"] = float(m.group("atr"))
        notes.append("Trail the rest: the trailing stop is armed only once the first scale-out has filled (from the next "
                     "bar), for the shares that are left; it trails the best price since entry.")
    _trail_activation(T, ex, notes)
    # a trailing stop at the N-bar low: lowest(low, N) of the bars before each bar, moved up only
    m = T.find(r"(?:(?:with|use|and|place|set|put) )?(?:a |an |the )?trailing stop(?:[- ]loss)? (?:at|on|below|under|above|over) "
               r"(?:the |its )?(?:lowest low|low|highest high|high)? ?(?:of (?:the )?(?:last |past |prior |previous )?)?"
               r"(?P<n>\d+)[- ]?(?:day|bar|session|candle)s?(?:'s?)? ?(?P<f>low|high|lows|highs)?")
    if m and (m.group("f") or re.search(r"\b(?:low|high)\b", m.group(0))):
        n_ = int(m.group("n"))
        hi = "high" in m.group(0).split("stop", 1)[1]
        ex["stop_level"] = f"highest(high, {n_})" if hi else f"lowest(low, {n_})"
        ex["dynamic_levels"] = True
        ex["stop_ratchet"] = True
        ex["_trail_phrase"] = m.group(0).strip()
        notes.append(f"Trailing stop at the {n_}-bar {'high' if hi else 'low'}: on every bar the stop is the "
                     f"{'highest high' if hi else 'lowest low'} of the {n_} bars before it (known at the previous close), "
                     f"and it only moves {'down' if hi else 'up'} (never loosens); it fills at that level, or at the open "
                     "if the price gaps through it.")
    elif m:
        T.rest  # noqa: B018 - not a bar-low trailing stop: left for the checks below
    # a stop at a price level, fixed when the position opens: "stop at the low of the entry bar", "stop at the 5 day low"
    CS = r"(?:^|(?<=[,;] )|(?<=[,;])|(?<=\band )|(?<=\bthen )|(?<=\bwith ))"
    m = T.find(CS + r"(?:(?:with|use|place|set|put) )?(?:a |an |the |my |your )?(?:initial |protective |hard )?stop(?:[- ]loss)?(?: order)?"
               r" (?:at|below|under|just below|above|over|just above|on) (?:the |its )?(?:(?:(?P<who>entry|signal) (?:bar|day|candle)'?s? (?P<f1>low|high))"
               r"|(?:(?P<f2>low|high) of (?:the )?(?P<who2>entry|signal|previous|prior|last) (?:bar|day|candle|session))"
               r"|(?:(?P<n>\d+) (?:day|bar|session)s? (?P<f3>low|high))|(?:(?P<yd>yesterday|the prior day|the previous day)'?s? (?P<f4>low|high))"
               r"|`(?P<raw>[^`]+)`)")
    if m and ex.get("stop_ratchet"):
        raise ParseError(f"Two stop levels ('{ex['_trail_phrase']}' and '{m.group(0).strip()}'); keep one.")
    if m and m.group("yd"):
        ex["stop_level"] = f"__yday__{m.group('f4').lower()}"      # resolved once the entry's fill is known
        ex["_stop_phrase"] = m.group(0).strip()
        m = None
    if m:
        if m.group("raw"):
            ex["stop_level"] = m.group("raw").strip()
        elif m.group("n"):
            n_, f_ = int(m.group("n")), m.group("f3").lower()
            ex["stop_level"] = f"lowest(low, {n_})" if f_ == "low" else f"highest(high, {n_})"
        else:
            ex["stop_level"] = (m.group("f1") or m.group("f2")).lower()
            if (m.group("who2") or "").lower() in ("previous", "prior", "last"):
                ex["stop_level"] = f"ref({ex['stop_level']}, 1)"
        ex["_stop_phrase"] = m.group(0).strip()
    # a target at a price level, or an R multiple of the risk
    m = T.find(rf"(?:(?:with|and) (?:a |an )?)?(?:take[- ]profits?|profit target|target|exit)(?: of| at)? {NUM} ?(?:r\b|x (?:the )?(?:initial )?risk|times (?:the )?(?:initial )?(?:risk|stop distance)|r[- ]multiples?)"
               rf"|(?:(?:with|and) (?:a |an )?)?{NUM} ?r (?:target|take[- ]profit|profit target)"
               rf"|(?:(?:with|and) (?:a |an )?)?{NUM}(?::1| to 1) reward[- ]to[- ]risk(?: (?:target|ratio))?")
    if m:
        ex["target_r"] = float(m.group(1) or m.group(2) or m.group(3))
        notes.append(f"R-multiple target: {ex['target_r']:g} x the initial risk (entry price - initial stop) from the entry, "
                     "fixed when the position opens.")
    m = T.find(r"(?:(?:with|and) (?:a |an )?)?(?:take[- ]profits?|profit target|target) (?:at|near) (?:the |its )?(?:(?P<n>\d+) (?:day|bar|session)s? (?P<f>high|low)|`(?P<raw>[^`]+)`)")
    if m:
        ex["target_level"] = (m.group("raw").strip() if m.group("raw") else
                              f"highest(high, {m.group('n')})" if m.group("f").lower() == "high" else f"lowest(low, {m.group('n')})")
    m = T.find(rf"(?:with a |use a |place a |and a )?{NUM} ?(?:x )?atr (?:trailing|chandelier) stop|(?:trailing|chandelier) stop(?: loss)?(?: of| at)? {NUM} ?(?:x )?atrs?(?: (?:from|below|above) the (?:high|low|highest high|lowest low))?")
    if m:
        ex["trailing_atr"] = float(m.group(1) or m.group(2))
    m = T.find(rf"(?:with a |use a |place a |and a )?{NUM} ?(?:x )?atr stop(?:[- ]loss)?|stop(?:[- ]loss)?(?: of| at)? {NUM} ?(?:x )?atrs?(?: (?:below|under|above|over|from) (?:the )?entry(?: price)?)?")
    if m:
        ex["stop_atr"] = float(m.group(1) or m.group(2))
    m = T.find(rf"(?:take[- ]profits?|profit target|target)(?: of| at)? {NUM} ?(?:x )?atrs?|{NUM} ?(?:x )?atr (?:profit target|take[- ]profit|target)")
    if m:
        ex["take_profit_atr"] = float(m.group(1) or m.group(2))
    m = T.find(rf"(?:with a |use a |and a )?trailing stop(?:[- ]loss)?(?: of| at)? {NUM}%|(?:with a |use a |and a )?{NUM}% trailing stop(?:[- ]loss)?")
    if m:
        ex["trailing_stop"] = float(m.group(1) or m.group(2)) / 100
    m = T.find(rf"(?<!buy )(?<!sell )(?:with a |use a |place a |and a )?(?:hard )?stop[- ]?(?:loss)?(?: of| at)? {NUM}%(?: below (?:the )?entry)?(?! (?:above|below|over|under) {LEVEL_REF})"
               rf"|(?:with a |use a |place a |and a )?{NUM}% (?:hard )?stop(?:[- ]loss)?(?! (?:order|entry|above|below|at)\b)")
    if m:
        ex["stop_loss"] = float(m.group(1) or m.group(2)) / 100
    m = T.find(rf"(?:with a |and a |and )?(?:take[- ]profits?|profit target|target)(?: of| at)? \+?{NUM}%|(?:with a |and a )?{NUM}% (?:profit target|take[- ]profit|target|gain target)")
    if m:
        ex["take_profit"] = float(m.group(1) or m.group(2)) / 100
    for k_, label in (("stop_loss", "stop"), ("trailing_stop", "trailing stop"), ("take_profit", "take profit"),
                      ("stop_atr", "ATR stop"), ("trailing_atr", "chandelier stop"), ("take_profit_atr", "ATR take profit"),
                      ("target_r", "R-multiple target")):
        if ex.get(k_) == 0:
            raise ParseError(f"A {label} of 0{'%' if k_ in ('stop_loss', 'trailing_stop', 'take_profit') else ''} would exit at "
                             "the entry price at once (every trade would close on its first check). Use a positive distance, "
                             f"e.g. '5% {label}' or '2 ATR', or leave it out.")
    if ex.get("trail_after_scale_out") and not ex.get("scale_out"):
        raise ParseError("'Trail the rest' of what? It follows a partial exit: e.g. 'sell half at +10% and trail the rest with "
                         "an 8% trailing stop'.")
    if ex.get("trail_after_scale_out") and not (ex.get("trailing_stop") or ex.get("trailing_atr")):
        raise ParseError("'Trail the rest' by how much? Say e.g. 'trail the rest with an 8% trailing stop' or 'trail the rest "
                         "with a 3 ATR chandelier stop'.")
    if ex.get("trailing_stop") and ex.get("trailing_atr") and not ex.get("trail_after_scale_out"):
        notes.append(f"Two trailing stops (a {ex['trailing_stop'] * 100:g}% trailing stop and a {ex['trailing_atr']:g} ATR "
                     "chandelier stop): both are active, one-cancels-other; at any moment the one closer to the price is the "
                     "one that can trigger.")
    _exit_tail_guard(before_exits, T.rest)
    rest_wo_orders = _entry_order(T.rest)[3]
    if re.search(r"\b(?:stop|target|trailing)\b", rest_wo_orders, re.I) and re.search(r"\d", rest_wo_orders):
        mm = re.search(r"[^;]*\b(?:stop|target|trailing)\b[^;]*", rest_wo_orders, re.I)
        raise ParseError(f"Could not understand the stop/target in {mm.group(0).strip()!r}. Write e.g. '5% stop loss', "
                         f"'2 ATR stop', '10% trailing stop', '3 ATR trailing stop', 'take profit at 8%'.")

    # ---- "the 5 Nasdaq 100 stocks with the lowest RSI(2) each day": fill up to N slots, best-ranked first
    ranked = False
    mrk = re.search(r"\b(?:the )?(?:top |bottom )?(\d+) ((?:nasdaq[- ]?100|ndx) (?:stocks|names|members|components)|stocks|names|tickers|etfs|sector etfs)"
                    r" (?:with|having|that have|showing) (?:the )?(lowest|highest|smallest|largest|biggest|weakest|strongest) (.+?)"
                    r"(?: (?:each|every) (?:trading )?(?:day|session)| daily)?(?=\s*(?:[,;.]|$|\bat\b|\bwhen\b|\bif\b|\bwhile\b|\bon\b|\band\b))",
                    T.rest, flags=re.I)
    if mrk:
        if mr:
            raise ParseError("The ranking is given twice; keep one of them.")
        rank_expr, rank_notes = value_phrase(mrk.group(4), default_n=None)
        low_first = mrk.group(3).lower() in ("lowest", "smallest", "weakest")
        kw["max_positions"] = int(mrk.group(1))
        kw["rank_by"], kw["rank_ascending"] = rank_expr, low_first
        notes.extend(rank_notes)
        notes.append(f"Up to {mrk.group(1)} positions at once: each day, free slots are filled with the "
                     f"{mrk.group(3).lower()} {rank_expr} among the tickers that qualify; open positions are held until their exit.")
        T.rest = T.rest[: mrk.start()] + " " + mrk.group(2) + " " + T.rest[mrk.end():]
        ranked = True

    # ---- clause split on what's left
    clauses = _split_clauses(T.rest)
    entries: list[tuple[str, str]] = []   # (side, clause)
    exits: list[str] = []
    kind = None
    for cl in clauses:
        low = cl.strip().lower()
        if re.match(r"^(?:short|sell short|go short|short[- ]sell|enter short)\b", low):
            entries.append(("short", cl))
            kind = "entry"
        elif re.match(r"^(?:buy|go long|long|purchase|enter(?: long)?|get in)\b", low):
            entries.append(("long", cl))
            kind = "entry"
        elif re.match(EXIT_VERB, low) or re.match(r"^(?:\d+ (?:trading )?(?:day|bar|week|session)s? later)", low):
            exits.append(cl)
            kind = "exit"
        elif not low.strip(" ;"):
            continue
        elif kind == "entry" and entries:
            entries[-1] = (entries[-1][0], entries[-1][1] + " , " + cl)
        elif kind == "exit":
            exits[-1] = exits[-1] + " , " + cl
        else:
            # a clause that neither starts with buy/short nor sell/hold: part of the first entry
            if entries:
                entries[-1] = (entries[-1][0], entries[-1][1] + " , " + cl)
            else:
                entries.append(("long", "buy " + cl))
                kind = "entry"
    if not entries:
        raise ParseError("No entry found. Start with 'buy ...' or 'short ...', e.g. 'buy MSFT at the close when it is down 5 days in a row, hold 1 day'.")
    sides = {s for s, _ in entries}
    if len(entries) > 2 or (len(entries) == 2 and len(sides) == 1):
        raise ParseError("Found more than one entry rule for the same side; combine them with 'and' / 'or'.")

    # ---- entries
    universe, uni_name = None, None
    parsed: dict[str, dict] = {}
    stateful = holding
    subject_hold = None
    timing_unstated: set = set()
    delay: dict = {}
    for side, cl in entries:
        md = re.search(r"(?i)\b(\d+) (?:trading )?(day|bar|session|week)s? (?=after\b)", cl)
        if md:   # "buy SPY 2 days after RSI(2) is below 10": the condition, 2 bars before the entry signal
            delay[side] = _period(md.group(1), md.group(2))
            cl = cl[: md.start()] + cl[md.end():]
        cl_rest = cl
        mcond = re.search(COND_START, cl_rest, flags=re.I)
        subject = cl_rest[: mcond.start()] if mcond else cl_rest
        cond_text = cl_rest[mcond.end():] if mcond else ""
        u, uname = _universe_phrase(subject)
        tick = find_tickers(re.sub(_INDEX_MEMBERS, " ", subject, flags=re.I), strict=True)
        if u is None:
            if tick:
                u = tick
            elif universe is not None:
                u = universe
            else:
                ct = find_tickers(cond_text, strict=True)
                if not ct:
                    raise ParseError("Which ticker(s)? Name one (e.g. MSFT or Microsoft) or say 'Nasdaq 100 stocks'.")
                u = ct[:1]
                cond_text = _sub_outside(rf"(?<![\w])[\$^]?{re.escape(u[0].lstrip('^'))}\b", " it ", cond_text, count=1, flags=re.I)
        else:
            u = list(u) + [x for x in tick if x not in u]
        if universe is not None and set(u) != set(universe):
            raise ParseError("Long and short rules must trade the same ticker(s).")
        universe, uni_name = u, uname or uni_name
        low = cl_rest  # original casing; all matching below is case-insensitive
        # order type
        valid = 1
        order, level, onote, low = _entry_order(low)
        if onote:
            notes.append(onote)
        mv = re.search(r"(?:good|valid) for (\d+) (?:trading )?(?:days|bars|sessions)", low, flags=re.I)
        if mv:
            valid = int(mv.group(1))
            low = low.replace(mv.group(0), " ")
        # timing
        if re.search(r"(?:next|following|tomorrow's)(?: day's| trading day's)? open|market on open|\bmoo\b", low, flags=re.I):
            fill = "next_open"
        elif re.search(r"(?:at|on) the open\b", low, flags=re.I):
            fill = "open"
        elif re.search(r"(?:next|following|tomorrow's)(?: day's)? close", low, flags=re.I):
            fill = "next_close"
        else:
            fill = "close"
            if not re.search(r"at the close|on the close|at close|market on close|\bmoc\b", low, flags=re.I) and order == "market":
                if getattr(_TL, "tv", False):
                    fill = "next_open"
                    notes.append("Entry timing not stated: TradingView-compatible mode fills at the next bar's open "
                                 "(TradingView's default, process_orders_on_close = false). Say 'at the close' to fill "
                                 "at the signal bar's close.")
                else:
                    notes.append("Entry timing not stated: assuming a fill at the close of the signal day.")
                    timing_unstated.add(side)
        # conditions
        mcond = re.search(COND_START, low, flags=re.I)
        if mcond:
            for k, v in _entry_subject(low[: mcond.start()], cl, notes).items():
                if k == "_hold":
                    subject_hold = v
                elif k in kw and kw[k] != v or (k == "sizing" and "position_size" in kw) or (k == "position_size" and "sizing" in kw):
                    raise ParseError(f"'{cl.strip()}': the position size is given twice; keep one.")
                else:
                    kw[k] = v
        cond = low[mcond.end():] if mcond else ""
        cond = re.sub(TIMING, " ", cond, flags=re.I)
        if re.search(r"(?i)\b(?:they|their)\b", cond):
            cond = _they_to_it(cond)
        if mcond and mcond.group(0).lower() in ("while", "as long as"):
            stateful = True
        if not mcond:
            rest = re.sub(ENTRY_VERB, " ", low.strip(), flags=re.I)
            rest = re.sub(TIMING, " ", rest, flags=re.I)
            for tk in find_tickers(cl):
                rest = _sub_outside(rf"(?<![\w])[\$^]?{re.escape(tk.lstrip('^'))}\b", " ", rest, flags=re.I)
            for name in COMPANIES:
                rest = _sub_outside(rf"\b{re.escape(name)}\b", " ", rest, flags=re.I)
            rest = re.sub(r"nasdaq[- ]?100|\bndx\b|(?:all|each|any|every) (?:the )?(?:index )?(?:stocks?|members?|components?|constituents?)|stocks?|sector (?:etfs|spdrs|funds)", " ", rest, flags=re.I)
            cond = rest
        for tk in universe:
            cond = _sub_outside(rf"(?<![\w])[\$^]?{re.escape(tk.lstrip('^'))}\b(?!\s*(?:is|closes|trades|'s)\b)", " it ", cond, flags=re.I) if len(universe) == 1 else cond
        if not re.sub(r"[^a-z0-9`]", "", re.sub(r"\b(?:it|the|a|an|and|stock|shares?)\b", "", cond.lower())):
            if order != "market" or ranked:
                cond = "`True`"
            else:
                raise ParseError("No entry condition found. Say e.g. 'buy MSFT at the close when it is down 5 days in a row'.")
        pron = re.fullmatch(r"(?i)\s*(?:it |the price )?(?:crosses|falls|drops|closes|goes|moves|is|trades)?(?: back)? ?(below|under|above|over)(?: (?:it|them|that|the average|the line))?\s*", cond)
        other = parsed.get("long" if side == "short" else "short")
        if pron and other:
            flip = _flip(other["entry"], below=pron.group(1).lower() in ("below", "under"))
            if not flip:
                raise ParseError(f"'{cond.strip()}' refers back to the other rule, which has no comparison to reverse.")
            parts = [flip]
            notes.append(f"Warning: '{cond.strip()}' was read as the reverse of the {'long' if side == 'short' else 'short'} rule: {flip}.")
        else:
            parts = parse_conditions(cond, universe, as_list=True)
        if fill == "open":
            late = [p for p in parts if not _open_safe(p)]
            if late:
                # split backtick blocks too, so open-time terms (e.g. gap) keep today's value
                fine, lagged, kept, periodic = [], [], [], []
                from .expr import open_time_periodic
                for p in parts:
                    for term in split_and(p):
                        if _open_safe(term):
                            fine.append(term)
                            kept.append(term)
                        elif open_time_periodic(term):
                            # weekly_close() & co at the open: the last period completed before today
                            fine.append(open_time_periodic(term))
                            periodic.append(term)
                        else:
                            fine.append(f"ref(({term}), 1)")
                            lagged.append(term)
                parts = fine
                if periodic:
                    notes.append(
                        f"Entry at the open: {' and '.join(f'`{t}`' for t in periodic)} uses a weekly / monthly value, "
                        "read as the last period completed before today "
                        f"({' and '.join(f'`{open_time_periodic(t)}`' for t in periodic)}): on the last trading day "
                        "of a week or month that period only completes at the close, so at the open the completed "
                        "period is the one before (on a Monday, last week).")
            if late and lagged:
                one = len(lagged) == 1
                lag_txt = " and ".join(f"`{t}`" for t in lagged)
                msg = (f"Warning: entry at the open: {lag_txt} {'uses' if one else 'use'} today's close/high/low, "
                       f"which {'is' if one else 'are'} not known at the open, so {'it is' if one else 'they are'} "
                       f"checked on the previous day's bar (lagged one day: {' and '.join(f'ref({t}, 1)' for t in lagged)})")
                if kept:
                    msg += f"; {' and '.join(f'`{t}`' for t in kept)} {'uses' if len(kept) == 1 else 'use'} today's values"
                msg += (". To use today's values for everything, enter at the close; to check the whole rule on the "
                        "previous day, enter at the next open.")
                notes.append(msg)
        if side in delay:
            n_ = delay[side]
            inner = " and ".join(parts)
            parts = [f"ref(({inner}), {n_})"]
            notes.append(f"'{n_} day{'s' if n_ != 1 else ''} after': the condition is checked {n_} bar{'s' if n_ != 1 else ''} "
                         f"before the signal bar (ref(({inner}), {n_})); the entry then fills as stated for the signal bar.")
        parsed[side] = {"entry": " and ".join(parts), "fill": fill, "order": order, "level": level, "valid": valid}

    # a rule knowable at the open (a gap: today's open against yesterday's close) with no timing stated is acted
    # on at that open, the moment it becomes true, rather than at the gap day's close
    if parsed and timing_unstated == set(parsed) and all(p["order"] == "market" for p in parsed.values()):
        from .expr import names_in
        if all(_open_safe(p["entry"]) and ({"gap", "open"} & names_in(p["entry"])) for p in parsed.values()):
            for p in parsed.values():
                p["fill"] = "open"
            notes[:] = [n for n in notes if not n.startswith("Entry timing not stated")]
            rules_ = " / ".join(p["entry"] for p in parsed.values())
            notes.append(f"Entry timing not stated: {rules_} is known at the open (it uses today's open), so the entry "
                         "fills at the open of that day. Say 'at the close' to enter at that day's close instead.")
    if len({p["fill"] for p in parsed.values()}) > 1:
        raise ParseError("Long and short entries must use the same fill timing.")
    first = parsed.get("long") or parsed.get("short")

    # ---- exits (clauses)
    # a clause left with only connecting words once its stops were read ("exit on a 10% trailing stop or a 3 ATR
    # chandelier stop" -> "exit on a ; or a") says nothing more
    filler = (r"\b(?:sell|exit|get out|cover|close (?:it|the position|out)|it|the|position|trade|on|at|with|using|via|a|an|or|"
              r"and|whichever|comes?|first|hits?|is|triggered|either|of|both|use|place)\b|[,;.\s]")
    if any(v for k_, v in ex.items() if not k_.startswith("_")):
        exits = [cl for cl in exits if re.sub(filler, "", cl, flags=re.I)]
    hold_bars, hold_fill = None, "close"
    hold_fill_stated = False
    exit_when, exit_when_fill = None, "close"
    for k, cl in enumerate(exits):
        mb = re.search(r"(?i)\b(sell|exit|cover|close out|take profits?)(\w*)(?: it| the position)? (?:at|on|near) (?:the )?(middle|mid|center|centre|basis|upper|lower) (bollinger )?(band|line)\b", cl)
        if mb:
            if len(parsed) != 1:
                raise ParseError(f"'{mb.group(0)}': say 'sell when it closes above the {mb.group(3)} band' (long) or "
                                 f"'cover when it closes below the {mb.group(3)} band' (short).")
            rel = "above" if "long" in parsed else "below"
            exits[k] = cl = cl[: mb.start()] + f"{mb.group(1)}{mb.group(2)} when it closes {rel} the {mb.group(3)} {mb.group(4) or ''}{mb.group(5)}" + cl[mb.end():]
            notes.append(f"'{mb.group(0)}' was read as: exit at the close once the price closes {rel} the {mb.group(3)} band.")
    for cl in exits:
        low = cl  # original casing; matching is case-insensitive
        mh = (re.search(r"(?:max(?:imum)? hold(?:ing)?(?: period)?(?: of)?|time stop(?: of)?) (\d+) (?:trading )?(day|bar|session|week|month)s?", low, flags=re.I)
              or re.search(r"(?:hold|keep)\w*(?: it| the position| the stock| positions?| the trade| them)?(?: for)?(?: up to| at most| a maximum of| no more than)? (\d+) (?:trading )?(day|bar|session|week|month)s?", low, flags=re.I)
              or re.search(r"(?:sell|exit|cover|close)\w*(?: it| the position| out| them)? (?:after|in) (\d+) (?:trading )?(day|bar|session|week|month)s?", low, flags=re.I)
              or re.search(r"(?:after|or after|or in|max(?:imum)? hold(?:ing)?(?: period)?(?: of)?|time stop(?: of)?|or) (\d+) (?:trading )?(day|bar|session|week|month)s?", low, flags=re.I)
              or re.search(r"(\d+) (?:trading )?(day|bar|session|week|month)s? later", low, flags=re.I))
        if mh:
            hb = _period(mh.group(1), mh.group(2))
            if hold_bars is not None and hb != hold_bars:
                raise ParseError(f"Two holding periods ({hold_bars} and {hb} bars); keep one.")
            hold_bars = hb
            low = low.replace(mh.group(0), " ")  # noqa
        elif not re.search(EXIT_WHEN, low, flags=re.I) and (
                re.search(r"(?:sell|exit|cover)\w* (?:at |on )?(?:the )?(?:next|following|tomorrow)", low, flags=re.I)
                or re.search(r"\bnext day\b", low, flags=re.I)):
            hold_bars = 1 if hold_bars is None else hold_bars
        mw = re.search(EXIT_WHEN, low, flags=re.I)
        if mw and mw.group("xt"):
            xt = data.canonical(mw.group("xt"))
            if xt not in universe:
                raise ParseError(f"'{cl.strip()}': {mw.group('xt')} is not what the strategy trades ({', '.join(universe[:5])}). "
                                 f"Name the traded ticker, or say 'sell when {mw.group('xt')} ...'.")
        if mw and mw.group("body").strip():
            wtxt = mw.group("body")
            timing = wtxt + " " + mw.group(0)
            if re.search(r"(?:next|following|tomorrow's)(?: day's| trading day's)? open|the day after|open after", timing, re.I):
                fill_word = "next_open"
            elif re.search(r"(?:at|on) (?:the )?open\b|market on open|\bmoo\b", timing, re.I):
                fill_word = "open"
            elif getattr(_TL, "tv", False) and not re.search(r"(?:at|on) (?:the )?close\b|market on close|\bmoc\b",
                                                              timing, re.I):
                fill_word = "next_open"     # TradingView-compatible mode: no timing stated -> the next open
            else:
                fill_word = "close"
            wtxt = re.sub(TIMING, " ", wtxt, flags=re.I)
            # a time exit repeated inside the rule text ("... or after 10 days") is the holding period already
            # read above; anything else ("risen 3% in 2 days") is part of the rule and must stay
            wtxt = re.sub(r"\b(?:(?:or )?after|or in) (\d+) (?:trading )?(day|bar|session|week)s?\b",
                          lambda mm: " " if hold_bars is not None and _period(mm.group(1), mm.group(2)) == hold_bars else mm.group(0),
                          wtxt, flags=re.I)
            wtxt = re.sub(r"\b(?:the )?first\b(?! trading day)", " ", wtxt, flags=re.I)
            wl = wtxt.strip(" ,;")
            rule = _exit_rule(wl, first["entry"], universe, notes)
            if fill_word == "open":
                if _open_safe(rule):
                    notes.append(f"Exit rule {rule} is known at the open, so positions are sold at the open of the day it is true.")
                else:
                    fill_word = "next_open"
                    notes.append(f"Exit at the open: {rule} is only known at the close, so it is checked at the close "
                                 "and the position is sold at the next day's open.")
            elif fill_word == "next_open":
                notes.append(f"Exit rule {rule} is checked at the close and filled at the next day's open.")
            if exit_when is not None and fill_word != exit_when_fill:
                raise ParseError("All rule exits must fill at the same time (the close, or the next open).")
            exit_when = rule if exit_when is None else f"({exit_when}) or ({rule})"
            exit_when_fill = fill_word
        for mx in re.finditer(rf"(?:sell|exit|cover|close)\w*[^,;]*?(?:at|on) (?:the )?(?:next |following |tomorrow's )?(?:day's |trading day's )?(open|close)\b", low, flags=re.I):
            if not mw or mx.start() < (mw.start("body") if mw else 0):
                hold_fill = mx.group(1).lower() if hold_bars is not None or not mw else hold_fill
                hold_fill_stated = True
        if hold_bars is None and not mw and not exit_when:
            m1 = re.search(r"(?:sell|exit|cover)\w* (?:it |them )?(?:at|on) (?:the )?(next |following )?(close|open)", low, flags=re.I)
            if m1:
                hold_bars = 1
                hold_fill = m1.group(2).lower()
                hold_fill_stated = True
                notes.append(f"No holding period stated: exiting at the first {hold_fill} after entry.")
    if subject_hold is not None:
        if hold_bars is not None and hold_bars != subject_hold:
            raise ParseError(f"Two holding periods ({subject_hold} and {hold_bars} bars); keep one.")
        hold_bars = subject_hold
    if (getattr(_TL, "tv", False) and hold_bars and not hold_fill_stated and first["fill"] == "next_open"):
        # TradingView: a time exit is a strategy.close() once N bars have passed since the entry bar - placed at that
        # bar's close and, with process_orders_on_close = false, filled at the next open
        n_ = hold_bars
        hold_bars, hold_fill = n_ + 1, "open"
        notes.append(f"Holding period in TradingView-compatible mode: 'hold {n_} day{'s' if n_ != 1 else ''}' is "
                     f"TradingView's strategy.close() once {n_} bar{'s' if n_ != 1 else ''} have passed since the entry "
                     f"bar: the close order is placed at the close {n_} bar{'s' if n_ != 1 else ''} after the entry bar and "
                     f"fills at the next open ({n_ + 1} bars after the entry bar). Say 'sell at the close' to exit at "
                     f"that close instead.")
    stops_given = any(ex.get(k) for k in ("stop_loss", "trailing_stop", "take_profit", "stop_atr", "trailing_atr", "take_profit_atr",
                                          "scale_out", "stop_level", "target_level", "target_r"))
    if hold_bars == 0 and first["fill"] not in ("open", "next_open") and first["order"] == "market":
        raise ParseError("A holding period must be at least 1 day: an entry at the close can't be sold at that same close "
                         "('hold 0 days'). Say 'hold 1 day' to sell at the next close, or buy at the open and 'sell at the "
                         "close' for a same-day round trip.")
    if stateful and len(parsed) == 1 and not (hold_bars or exit_when or stops_given):
        exit_when = f"not ({first['entry']})"
        notes.append(f"No exit given: in the market while {first['entry']} is true, and out at the close of the "
                     "first day it is false (it is re-entered when it turns true again).")
    if not any([hold_bars is not None, exit_when, stops_given, len(parsed) == 2]):
        raise ParseError("No exit rule found. Say e.g. 'hold 1 day and sell at the close', 'sell when it closes above its 5-day moving average', or 'with a 5% stop loss'.")

    # leftover check over the whole sentence (entry conditions were parsed strictly already)
    for cl in exits:
        chk = re.sub(EXIT_WHEN, " ", cl, flags=re.I).lower()
        chk = re.sub(r"(?:max(?:imum)? hold(?:ing)?(?: period)?(?: of)?|time stop(?: of)?) \d+ (?:trading )?(?:day|bar|session|week|month)s?", " ", chk)
        chk = re.sub(r"\d+ (?:trading )?(?:day|bar|session|week|month)s? later", " ", chk)
        chk = re.sub(r"(?:hold|keep)\w*(?: it| the position| the stock| positions?| the trade| them)?(?: for)?(?: up to| at most| a maximum of| no more than)? \d+ (?:trading )?(?:day|bar|session|week|month)s?", " ", chk)
        chk = re.sub(r"(?:after|or after|or in|in|or|max(?:imum)? hold(?:ing)?(?: period)?(?: of)?|time stop(?: of)?) \d+ (?:trading )?(?:day|bar|session|week|month)s?(?: later)?", " ", chk)
        chk = re.sub(TIMING + r"|\bnext day\b|\bnext\b|\bfollowing\b", " ", chk)
        left = [w for w in re.findall(r"[a-z%']+|\d+(?:\.\d+)?", chk) if w not in STOP]
        if left:
            raise ParseError(f"Could not interpret {' '.join(left)!r} in the exit {cl.strip()!r}.")

    side = "both" if len(parsed) == 2 else ("short" if "short" in parsed else "long")
    if mcap:
        if mr or ranked:
            raise ParseError("The ranking is given twice; keep one of them.")
        w = (mcap.group(1) or mcap.group(2) or "largest").lower()
        kw["rank_by"], kw["rank_ascending"] = "market_cap", w in ("smallest", "lowest")
        notes.append(f"Ranking: when more tickers signal than there are free slots, the {'smallest' if kw['rank_ascending'] else 'largest'} "
                     "market caps (as of the signal day) are bought first.")
    if mr:
        w = mr.group(1).lower()                           # T.find matches case-insensitively: "Lowest RSI"
        ind = (mr.group(3) or "").lower()
        per = mr.group(2) or mr.group(4) or mr.group(5)   # "RSI(2)", "RSI 2", "2-day RSI"
        if ind == "rsi" or "oversold" in w or "overbought" in w:
            mn = re.search(r"rsi\([^,]+,\s*(\d+)\)", first["entry"])
            kw["rank_by"] = f"rsi(close, {per})" if per else f"rsi({mn.group(1) if mn else 2})"
            kw["rank_ascending"] = w in ("lowest", "most oversold")
        elif ind == "volatility":
            kw["rank_by"], kw["rank_ascending"] = f"volatility({per or 20})", w in ("lowest", "weakest")
        else:
            kw["rank_by"] = f"ret({per or 5})" if ind.startswith("return") else (f"ret({per})" if per else "change")
            kw["rank_ascending"] = w in ("lowest", "weakest", "biggest loser", "biggest losers", "biggest decline", "biggest declines", "largest decline", "largest declines")
        notes.append(f"Ranking: when more tickers signal than there are free slots, the {'lowest' if kw['rank_ascending'] else 'highest'} "
                     f"{kw['rank_by']} (as of the signal day) is bought first.")
    if 1 < len(universe) < 10 and uni_name is None and "max_positions" not in kw:
        # an explicit short list ("buy Tesla and Nvidia when ..."): one slot per ticker, equal shares
        k_ = len(universe)
        kw["max_positions"] = k_
        if "position_size" not in kw and kw.get("sizing", "percent") == "percent":
            notes.append(f"{k_} tickers and no position limit given: up to {k_} positions (one per ticker) at "
                         f"{100 / k_:.4g}% of equity each.")
        else:
            notes.append(f"{k_} tickers and no position limit given: up to {k_} positions (one per ticker).")
    elif len(universe) > 1 and "max_positions" not in kw:
        kw["max_positions"] = 10
        if "position_size" not in kw and kw.get("sizing", "percent") == "percent":
            notes.append("Multiple tickers and no position limit given: using up to 10 positions at 10% of equity each.")
        else:
            notes.append("Multiple tickers and no position limit given: using up to 10 positions.")
    if uni_name == "NDX":
        pit = kw.get("point_in_time", True)
        notes.append("Universe: Nasdaq-100 " + ("with point-in-time membership (stocks are only bought while in the index; "
                     "former members are included where price history exists)." if pit else
                     "using TODAY'S members only (the latest membership list, no membership filter) - survivorship-biased "
                     "by construction."))
        if not pit:
            cur = set(data.current_members())
            universe = [t for t in universe if t in cur]
    elif _index_today_note(uni_name, universe, notes):
        kw["point_in_time"] = False
    benchmark = kw.pop("benchmark", None)
    brb = kw.pop("benchmark_rebalance", None)
    if brb not in (None, "monthly"):
        raise ParseError("A blended benchmark of a trading strategy is rebalanced monthly; its schedule can be chosen for "
                         "portfolios (e.g. 'hold 60% SPY and 40% AGG, rebalance yearly, vs 60/40 SPY/AGG rebalanced "
                         "quarterly').")
    for i_, n_ in enumerate(notes):
        if n_.startswith("Benchmark: a blend of ") and n_.endswith("total returns."):
            notes[i_] = n_[:-1] + ", rebalanced monthly."

    stop_phrase = ex.pop("_stop_phrase", None)
    trail_phrase = ex.pop("_trail_phrase", None)
    if ex.get("stop_level") and (stop_phrase or trail_phrase) and side == "both":
        raise ParseError(f"'{stop_phrase or trail_phrase}': a long's stop is below the price and a short's above it; with "
                         "long and short entries, give the stop as a percentage or ATR distance.")
    if trail_phrase and ((side == "long" and "high" in ex["stop_level"]) or (side == "short" and "low" in ex["stop_level"])):
        raise ParseError(f"'{trail_phrase}': a {side} position's stop is {'below' if side == 'long' else 'above'} the price: "
                         f"use the {'low' if side == 'long' else 'high'}.")
    if str(ex.get("stop_level", "")).startswith("__yday__"):
        f_ = ex["stop_level"][8:]
        at_close = first["fill"] in ("close", "next_close") and first["order"] == "market"
        # a fill at the close reads the entry bar (yesterday = one bar back); a fill at the open reads the bar before it
        ex["stop_level"] = f"ref({f_}, 1)" if at_close else f_
        notes.append(f"Stop at yesterday's {f_}: the {f_} of the day before the entry day "
                     f"({'ref(' + f_ + ', 1) on the entry bar' if at_close else 'the signal bar, read when the order fills at the open'}).")
    if ex.get("stop_level") and stop_phrase:
        lv = ex["stop_level"]
        if side == "both":
            raise ParseError(f"'{stop_phrase}': a long's stop is below the price and a short's above it; with long and short "
                             "entries, give the stop as a percentage or ATR distance, or write it in backticks per side.")
        if (side == "long" and re.search(r"\bhigh", lv)) or (side == "short" and re.search(r"\blow", lv)):
            raise ParseError(f"'{stop_phrase}': a {side} position's stop is {'below' if side == 'long' else 'above'} the price: "
                             f"use the {'low' if side == 'long' else 'high'}.")
        at_close = first["fill"] in ("close", "next_close") and first["order"] == "market"
        when = ("the entry bar's values (the fill is at its close, when they are known)" if at_close else
                "the previous bar's values (the fill is at or after the open, before the entry bar's low/high is known: "
                "for a next-open entry that is the signal bar)")
        notes.append(f"Stop level: {lv}, evaluated once when the position opens with {when}, and fixed for the trade. It fills "
                     "at that level, or at the open if the price gaps through it.")
    long_rule = parsed.get("long", {}).get("entry")
    short_rule = parsed.get("short", {}).get("entry")
    strat = Strategy(
        universe=universe, universe_name=uni_name,
        entry=long_rule if side in ("long", "both") else short_rule,
        short_entry=short_rule if side == "both" else None,
        side=side, entry_fill=first["fill"], entry_order=first["order"], entry_level=first["level"],
        order_valid_bars=first["valid"],
        hold_bars=hold_bars, hold_exit_fill=hold_fill, exit_when=exit_when, exit_when_fill=exit_when_fill,
        description=raw, notes=notes, **ex, **kw,
    )
    strat.benchmark = benchmark
    rules = " ".join(r for r in (strat.entry, strat.short_entry, strat.exit_when) if isinstance(r, str))
    if "vwap(" in rules or re.search(r"sma\(\(\(.*volume, \d+\) / sma\(", rules):
        strat.notes.append("VWAP on daily bars: the rolling N-day (default 20) volume-weighted average of the typical "
                           "price (high + low + close) / 3, not an intraday VWAP.")
    if re.search(r"\b(?:upper|lower|middle|mid|center|centre|basis) (?:bollinger )?(?:band|line)\b", raw, re.I):
        strat.notes.append("Bollinger bands: 20-day SMA +/- 2 standard deviations; the middle band is the 20-day SMA.")
    if re.search(r"\bret\(", rules):
        strat.notes.append("Returns in the rules are price returns (close to close, dividends not included); "
                           "portfolio conditions use total returns.")
    if re.search(r"\b(?:tret|ma_return|stdev_return|max_drawdown)\(", rules + " " + str(strat.rank_by or "")):
        strat.notes.append("Return statistics (tret, moving average / standard deviation of return, max drawdown) use "
                           "total-return (dividend-adjusted) prices.")
    strat.notes = list(dict.fromkeys(strat.notes))
    if re.search(r"\b(?:down|up)_(?:days|streak)\b\s*>=|_streak\([^)]*\) >=", strat.entry):
        notes.append("'N days in a row' also fires on later days of a longer streak (6th, 7th...); say 'exactly N days' to fire only on the Nth.")
    return strat


# the bar an entry order's price refers to: all of these mean the signal bar (the order is placed
# after its close for the next session)
LEVEL_REF = r"(?:the |its |today's |yesterday's |yesterdays |the previous day's |the previous |the prior day's |the prior |the signal day's |the day's |the last )?(?:close|high|low|open)\b"


def _entry_order(text: str) -> tuple[str, str | None, str | None, str]:
    """Limit / stop ENTRY orders in an entry clause: 'at a limit 2% below the close', 'at a stop 1% above
    the close', 'at a stop above the high', 'with a buy stop at yesterday's high', 'on a stop at the
    20 day high'. Returns (order, level expression, note, text with the phrase removed)."""
    pre = r"(?:(?:with|using|on|at|via|place|placing) )?(?:a |an )?(?:buy |sell )?"
    m = re.search(rf"{pre}(limit|stop)(?: order| entry)?(?: at| of)? {NUM}% (below|above|under|over) {LEVEL_REF}", text, re.I)
    if m:
        f = float(m.group(2)) / 100
        field = re.search(r"(close|high|low|open)\s*$", m.group(0), re.I).group(1).lower()
        lvl = f"{field} * {1 - f if m.group(3).lower() in ('below', 'under') else 1 + f:.6g}"
        return m.group(1).lower(), lvl, _level_note(m.group(0), field), text.replace(m.group(0), " ")
    m = re.search(rf"{pre}(limit|stop)(?: order| entry)?(?: (?:at|above|below|of))? (?:the |its )?(\d+) day (high|low)", text, re.I)
    if m:
        lvl = f"{'highest(high' if m.group(3).lower() == 'high' else 'lowest(low'}, {m.group(2)})"
        return m.group(1).lower(), lvl, None, text.replace(m.group(0), " ")
    m = re.search(rf"{pre}(limit|stop)(?: order| entry)? (?:at|above|below|just above|just below|over|under) {LEVEL_REF}", text, re.I)
    if m:
        field = re.search(r"(close|high|low|open)\s*$", m.group(0), re.I).group(1).lower()
        return m.group(1).lower(), field, _level_note(m.group(0), field), text.replace(m.group(0), " ")
    return "market", None, None, text


def _level_note(phrase: str, field: str) -> str | None:
    if re.search(r"(?i)yesterday|previous|prior|last", phrase):
        return (f"'{phrase.strip()}' was read as the signal day's {field}: the order is placed after that day's close "
                f"and works in the next session(s).")
    return None


def _open_safe(rule: str) -> bool:
    from .expr import open_safe
    return open_safe(rule)


# ----------------------------------------------------------------- allocation portfolios

FREQ_WORDS = {"daily": "daily", "day": "daily", "weekly": "weekly", "week": "weekly", "monthly": "monthly",
              "month": "monthly", "quarterly": "quarterly", "quarter": "quarterly", "annually": "yearly",
              "yearly": "yearly", "year": "yearly", "annual": "yearly"}


DIRECTION_WORDS = {"top", "bottom", "best", "worst", "strongest", "weakest", "highest", "lowest", "most", "least",
                   "largest", "smallest", "greatest", "biggest"}
LOW_WORDS = r"\b(lowest|least|weakest|worst|smallest|bottom)\b"


def _unit_n(n, unit, default=None):
    if n is None:
        return default
    return _period(n, unit)


def value_phrase(text: str, ctx: Ctx | None = None, default_n: int | None = None,
                 total: bool = True) -> tuple[str, list[str]]:
    """An indicator phrase -> (numeric expression, notes), e.g. '10 day RSI' -> rsi(close, 10).

    Used for ranking metrics, threshold conditions and two-ticker comparisons. Every word must be
    understood. Returns are total returns (tret) unless total=False (price returns, ret)."""
    ctx = ctx or Ctx()
    s = " " + re.sub(r"\s+", " ", text.strip().lower()) + " "
    s = re.sub(r"'s\b", " ", s)
    s = _indicator_periods(s, cond=False)
    notes: list[str] = []
    c = ctx.c
    tr = "tr" if ctx.base else f"{ctx.c[:-len('.close')]}.tr"
    U = r"(day|week|month|year|bar|session)s?"
    if "`" not in s:
        s = _lookback_lists(s)
        # "risk-adjusted momentum", "12 month volatility-adjusted return" = the return divided by its volatility
        mr = re.search(rf"(?:risk|volatility|vol)[- ]adjusted (?:(\d+) {U} )?(?:total )?(?:returns?|momentum|performance)"
                       rf"(?: over (?:the )?(?:last |past )?(\d+) {U})?|(\d+) {U} (?:risk|volatility|vol)[- ]adjusted "
                       rf"(?:total )?(?:returns?|momentum|performance)", s)
        if mr:
            g = mr.groups()
            n, u = next(((g[i], g[i + 1]) for i in (0, 2, 4) if g[i]), (None, None))
            if n is None:
                n, u = "12", "month"
                notes.append(f"No lookback given for '{mr.group(0).strip()}': using 12 months.")
            s = f"{s[: mr.start()]} {n} {u} return divided by {n} {u} volatility {s[mr.end():]}"
        md = re.fullmatch(r"\s*(.+?)\s+(?:divided by|/|per unit of)\s+(.+?)\s*", s)
        if md:
            left, right = md.group(1), md.group(2)
            rn = re.search(rf"(\d+) {U}", left)
            if re.fullmatch(r"(?:its |their |the )?(?:annuali[sz]ed |realized |realised |historical |daily )?(?:volatility|vol|risk)",
                            right.strip()):
                # "... divided by volatility": over the same lookback as the return
                right = f"{rn.group(1)} {rn.group(2)} volatility" if rn else f"{default_n or 252} day volatility"
            a, na = value_phrase(left, ctx, default_n, total)
            b, nb = value_phrase(right, ctx, default_n, total)
            a_ = f"({a})" if re.search(r" [-+*/] ", a) else a
            b_ = f"({b})" if re.search(r" [-+*/] ", b) else b
            notes.extend(na + nb)
            notes.append(f"'{text.strip()}' = {a_} / {b_}: the return per unit of volatility (annualised standard deviation "
                         "of daily price returns), so a steadier gain ranks above a jumpier one of the same size.")
            return f"{a_} / {b_}", notes
    pats = [
        (rf"`([^`]+)`", lambda m: m.group(1)),
        # "average of 1, 3, 6 and 12 month return" (after _lookback_lists: "average of 1/3/6/12 month return")
        (rf"(?:the )?(?:average|avg\.?|mean|blend(?:ed)?)(?: of)?(?: the)? (\d+(?:/\d+)+) {U} (?:total )?(?:returns?|momentum|performance)"
         rf"|(\d+(?:/\d+)+) {U} (?:average|avg\.?|mean|blend(?:ed)?) (?:total )?(?:returns?|momentum|performance)",
         lambda m: _avg_returns(m.group(1) or m.group(3), m.group(2) or m.group(4), tr, c, total)),
        # "12 month return skipping the last month", "12 month momentum excluding the most recent month"
        (rf"(?:(\d+) {U} )?(?:cumulative |total |trailing )?(?:returns?|momentum|performance)(?: over (?:the )?(?:last |past |prior )?(\d+) {U})?"
         rf",? (?:skipping|skip|excluding|exclude|ex|except(?: for)?|without|ignoring|leaving out|lagged by|lagged|ending) (?:the )?"
         rf"(?:last |latest |most recent |recent |past |final )?(?:(\d+|one|a) )?{U}(?: ago)?",
         lambda m: _skip_return(m, tr, c, total)),
        # before the moving-average patterns, whose bare "ma" would otherwise match inside "market"
        (r"\bmarket[- ]cap(?:itali[sz]ation)?\b", lambda m: "market_cap" if ctx.base else _unsupported("market cap of another ticker")),
        (rf"(?:(\d+) {U} )?(?:moving average|average|mean|ma) of (?:the )?(?:daily )?returns?(?: over (?:the )?(?:last |past )?(\d+) {U})?",
         lambda m: f"ma_return({tr}, {_unit_n(m.group(1) or m.group(3), m.group(2) or m.group(4), 20)})"),
        (rf"(?:(\d+) {U} )?(?:max(?:imum)?|largest|biggest) drawdowns?(?: over (?:the )?(?:last |past )?(\d+) {U})?",
         lambda m: f"max_drawdown({tr}, {_unit_n(m.group(1) or m.group(3), m.group(2) or m.group(4), 63)})"),
        (rf"(?:(\d+) {U} )?(?:standard deviation|stdev|std dev|std) of (?:the )?(?:daily )?returns?(?: over (?:the )?(?:last |past )?(\d+) {U})?",
         lambda m: f"stdev_return({tr}, {_unit_n(m.group(1) or m.group(3), m.group(2) or m.group(4), 20)})"),
        (rf"(?:(\d+) {U} )?(?:standard deviation|stdev|std dev|std) of (?:the )?(?:price|prices|close)(?: over (?:the )?(?:last |past )?(\d+) {U})?",
         lambda m: f"stdev({c}, {_unit_n(m.group(1) or m.group(3), m.group(2) or m.group(4), 20)})"),
        (rf"(?:(\d+) {U} )?daily (?:volatility|vol)(?: over (?:the )?(?:last |past )?(\d+) {U})?",
         lambda m: f"stdev_return({tr}, {_unit_n(m.group(1) or m.group(3), m.group(2) or m.group(4), 20)})"),
        (rf"(?:(\d+) {U} )?(?:annuali[sz]ed |historical |realized )?(?:volatility|vol)(?: over (?:the )?(?:last |past )?(\d+) {U})?",
         lambda m: f"volatility({_unit_n(m.group(1) or m.group(3), m.group(2) or m.group(4), 20)}{'' if ctx.base else ', ' + c})"),
        (rf"(?:current )?drawdown (?:over|in|during) (?:the )?(?:last |past )?(\d+) {U}",
         lambda m: (_note(f"'{m.group(0).strip()}' was read as the maximum drawdown within the last {_period(m.group(1), m.group(2))} "
                          f"days (peak to trough, a positive number); say '{m.group(1)} {m.group(2)} drawdown' for the current "
                          f"distance below the {m.group(1)} {m.group(2)} high."),
                    f"max_drawdown({tr}, {_period(m.group(1), m.group(2))})")[1]),
        (rf"(\d+)[- ](\d+) (?:month )?momentum",
         lambda m: _skip_momentum(int(m.group(1)), int(m.group(2)), tr, c, total)),
        (rf"(?:(\d+) {U} )?(?:current )?drawdown(?: from (?:its |the )?(?:high|peak))?",
         lambda m: f"drawdown({c}{', ' + str(_period(m.group(1), m.group(2))) if m.group(1) else ''})"),
        (rf"distance (?:from|above|to) (?:its |the )?(\d+) {U} (?:moving average|ma|sma)",
         lambda m: f"{c} / sma({c}, {_period(m.group(1), m.group(2))}) - 1"),
        (rf"(?:(\d+) {U} )?(exponential moving average|ema)(?: of (?:the )?price)?",
         lambda m: f"ema({c}, {_unit_n(m.group(1), m.group(2), 20)})"),
        (rf"(?:(\d+) {U} )?(?:simple )?(?:moving average|sma|ma)(?: of (?:the )?price)?",
         lambda m: f"sma({c}, {_unit_n(m.group(1), m.group(2), 20)})"),
        (r"(?:the )?(weekly|monthly) (?:relative strength index|rsi)(?:\s*\(\s*(\d+)\s*\)|\s+(\d+)(?! (?:day|week|month|year|bar|session)))?",
         lambda m: f"{m.group(1)}_rsi({m.group(2) or m.group(3) or 14}{'' if ctx.base else ', ' + c})"),
        (rf"(?:(\d+) {U} )?(?:relative strength index|rsi)(?:\s*\(\s*(\d+)\s*\)|\s+(\d+)(?! {U}))?",
         lambda m: (notes.append(_RSI_DEFAULT_NOTE) if not (m.group(1) or m.group(3) or m.group(4)) else None,
                    f"rsi({c}, {m.group(3) or m.group(4) or _unit_n(m.group(1), m.group(2), 14)})")[1]),
        (rf"(?:(\d+) {U} )?(?:cumulative |total |trailing )?(?:returns?|momentum|performance|gains?|rate of change|roc|change|price change)(?: over (?:the )?(?:last |past |prior )?(\d+) {U})?",
         None),
        (r"(?:yesterday|the previous day|previous day|the prior day|prior day|the previous|previous|the prior|prior) (high|low|close|open)",
         lambda m: f"ref({ {'high': ctx.h, 'low': ctx.l, 'close': c, 'open': ctx.o}[m.group(1)] }, 1)"),
        (r"(?:current |latest |last )?(?:price|close|closing price)", lambda m: c),
    ]
    for pat, fn in pats:
        m = re.search(pat, s)
        if not m:
            continue
        g = m.groups()
        if "over (?:the )?" in pat and len(g) >= 4 and g[0] and g[2] and _period(g[0], g[1]) != _period(g[2], g[3]):
            raise ParseError(f"'{m.group(0).strip()}' gives two different lookbacks; keep one.")
        if fn is None:  # return / momentum
            n = _unit_n(m.group(1) or m.group(3), m.group(2) or m.group(4), None)
            if n is None:
                n = default_n or 252
                notes.append(f"No lookback given for '{m.group(0).strip()}': using {n} trading days ({n // 21} months).")
            expr_ = f"tret({tr}, {n})" if total else f"ret({c}, {n})"
        else:
            expr_ = fn(m)
        rest = (s[: m.start()] + " " + s[m.end():]).split()
        left = [w for w in rest if w not in STOP and w not in DIRECTION_WORDS and w not in ("its", "their", "by", "the", "of", "value")]
        if left:
            raise ParseError(f"Could not interpret {' '.join(left)!r} in {text.strip()!r}")
        return expr_, notes
    raise ParseError(f"Unknown indicator {text.strip()!r}. Use e.g. '3 month return', '10 day RSI', '60 day volatility', "
                     f"'20 day max drawdown', '20 day moving average of return', '20 day standard deviation of price', "
                     f"'market cap', or a rule in backticks.")


def _rank_words_note(word: str, n: int, metric_text: str, direction: str, metric: str) -> str | None:
    """'the top 1 ... by lowest 10 day RSI': the ranking word and the metric's own direction word disagree, so the
    selection flips (the bottom 1 by RSI). Say so: it is easy to misread."""
    low = re.search(LOW_WORDS, metric_text.lower())
    if not low:
        return None
    what = re.sub(LOW_WORDS, "", metric_text, flags=re.I).strip()
    what = re.sub(r"\s+", " ", what)
    picks = "the one" if n == 1 else f"the {n}"
    if direction == "bottom":
        return (f"'{word} {n} ... by {metric_text.strip()}' was read as the bottom {n} by {what} ({metric}): {picks} with "
                f"the lowest value{'s are' if n > 1 else ' is'} held.")
    return (f"'{word} {n} ... by {metric_text.strip()}': '{word}' and '{low.group(1)}' cancel out, so it was read as the "
            f"top {n} by {what} ({metric}): {picks} with the highest value{'s are' if n > 1 else ' is'} held.")


def _dd_rank(metric: str, direction: str, text: str, n: int = 1) -> tuple[str, str | None]:
    """Rankings by drawdown use its size, a positive number: 'top' = the largest drawdown. Two measures:
    'N day drawdown' is the current drawdown, how far the price is below its highest close of the last N days
    (drawdown(), <= 0, negated here); 'N day max drawdown' is Composer's Max Drawdown indicator, the deepest
    peak-to-trough fall within the last N days (max_drawdown(), a positive size)."""
    if metric.startswith("drawdown("):
        metric = f"-{metric}"
        current = True
    elif metric.startswith("max_drawdown("):
        current = False
    else:
        return metric, None
    which = "largest" if direction == "top" else "smallest"
    other = "smallest" if direction == "top" else "largest"
    what = "the asset" if n == 1 else f"the {n} assets"
    look = re.search(r",\s*(\d+)\)", metric)
    nd = look.group(1) if look else None
    if current:
        measure = (f"the current drawdown: how far each asset is below its highest close of the last {nd} days today, as a "
                   f"positive size. That is not Composer's Max Drawdown filter (the deepest peak-to-trough fall within "
                   f"the {nd} days, even if the price has recovered since); say '{nd} day max drawdown' for that"
                   if nd else
                   "the current drawdown: how far each asset is below its highest close so far, as a positive size (not "
                   "Composer's Max Drawdown, the deepest fall within a window; say e.g. '20 day max drawdown' for that)")
    else:
        measure = (f"the max drawdown, the deepest peak-to-trough fall within the last {nd or 'N'} days (as Composer's Max "
                   f"Drawdown filter), a positive size; say '{nd or 'N'} day drawdown' for the current fall from the "
                   f"{nd or 'N'} day high")
    return metric, (f"Ranking by '{text.strip()}' uses {measure}. This selects {what} with the {which} drawdown"
                    + (" (the most drawn down)" if which == "largest" else " (the least drawn down)")
                    + f"; say '{other} drawdown' for the opposite.")


def _skip_momentum(n: int, skip: int, tr: str, c: str, total: bool) -> str:
    """'12-1 momentum': the return over the last n months excluding the most recent `skip` months."""
    if not 0 < skip < n:
        raise ParseError(f"'{n}-{skip} momentum': the skipped months must be fewer than the lookback, e.g. '12-1 momentum'.")
    look, lag = (n - skip) * 21, skip * 21
    _note(f"'{n}-{skip} momentum' = the {n}-month return skipping the latest {skip} month{'s' if skip > 1 else ''}: the "
          f"{look}-day return ending {lag} trading days ago (ref(..., {lag})).")
    return f"ref(tret({tr}, {look}), {lag})" if total else f"ref(ret({c}, {look}), {lag})"


_NUM_LIST = re.compile(r"(?<![\d.%$/])(\d+)-?((?:\s*(?:,\s*(?:and\s+|&\s*)?|\band\b\s*|&\s*|/)\s*\d+-?)+)\s*[- ]?(day|week|month|year)(s?)\b",
                       re.I)


def _lookback_lists(s: str) -> str:
    """'1, 3, 6 and 12 month' / '1-, 3-, 6- and 12-month' / '1/3/6/12 month' -> '1/3/6/12 month' (one token, so the
    commas are not read as separators between options)."""
    def fix(m):
        nums = [m.group(1)] + re.findall(r"\d+", m.group(2))
        return "/".join(nums) + f" {m.group(3)}{m.group(4)}"
    return _NUM_LIST.sub(fix, s)


def _avg_returns(nums: str, unit: str, tr: str, c: str, total: bool) -> str:
    """'average of 1/3/6/12 month return' -> the plain average of those total returns."""
    ns = [_period(n, unit) for n in nums.split("/")]
    if len(set(ns)) < len(ns) or min(ns) < 1:
        raise ParseError(f"'average of {nums.replace('/', ', ')} {unit} return': give different lookbacks.")
    parts = [f"tret({tr}, {n})" if total else f"ret({c}, {n})" for n in ns]
    _note(f"'average of {', '.join(nums.split('/'))} {unit} returns' = the plain average of the {len(ns)} "
          f"{'total ' if total else ''}returns over {', '.join(str(n) for n in ns)} trading days "
          f"(({' + '.join(parts)}) / {len(ns)}).")
    return f"({' + '.join(parts)}) / {len(ns)}"


_SKIP_NOTE = re.compile(r"^('[^']+') = .*?the (\d+)-day return ending (\d+) (?:trading )?days ago \(ref\(\.\.\., \d+\)\).*$")


def _calendar_skip_note(note: str) -> str:
    """With calendar-month lookbacks a skip-month return is measured on month-end prices: say so instead of the
    trading-day reading (see expr.calendar_month_return's skip)."""
    m = _SKIP_NOTE.match(note)
    if not m or int(m.group(2)) % 21 or int(m.group(3)) % 21:
        return note
    k, n = int(m.group(3)) // 21, (int(m.group(2)) + int(m.group(3))) // 21
    return (f"{m.group(1)} = {n}-{k} month momentum on month-end prices (calendar months): the month-end price {k} "
            f"month{'s' if k > 1 else ''} ago / the month-end price {n} months ago - 1, both completed month-ends counted "
            "from the last one (on a month's last session, that session).")


def _skip_return(m, tr: str, c: str, total: bool) -> str:
    """'12 month return skipping the last month' -> the return from 12 months ago to 1 month ago."""
    g = m.groups()
    if not (g[0] or g[2]):
        raise ParseError(f"'{m.group(0).strip()}': give the lookback, e.g. '12 month return skipping the last month'.")
    n = _period(g[0] or g[2], g[1] or g[3])
    k = g[4] if g[4] and g[4].isdigit() else "1"
    lag = _period(k, g[5])
    if not 0 < lag < n:
        raise ParseError(f"'{m.group(0).strip()}': the skipped period must be shorter than the lookback.")
    look = n - lag
    _note(f"'{m.group(0).strip()}' = the return over the {n} trading days up to {lag} trading days ago: the {look}-day "
          f"return ending {lag} days ago (ref(..., {lag})), so the most recent {lag} days are left out.")
    return f"ref(tret({tr}, {look}), {lag})" if total else f"ref(ret({c}, {look}), {lag})"


WEIGHT_METHODS = [
    (r"(?:inverse[- ]vol(?:atility)?)", "inverse_vol"),
    (r"(?:equal[- ]risk(?: contribution)?|risk[- ]parity)", "risk_parity"),
    (r"(?:min(?:imum)?[- ]variance|min[- ]vol(?:atility)?|minimum volatility)", "min_variance"),
    (r"(?:max(?:imum)?[- ]sharpe(?: ratio)?)", "max_sharpe"),
    (r"(?:max(?:imum)?[- ]diversification)", "max_diversification"),
    (r"(?:market[- ]cap(?:italization)?)", "market_cap"),
    (r"(?:equal(?:ly)?)", "equal"),
]


def _weighting_phrase(s: str) -> tuple[str, int | None] | None:
    """'inverse volatility weighted over 60 days' -> ('inverse_vol', 60). None if not a weighting phrase."""
    t = s.strip().lower()
    # "inverse 10 day volatility" -> "inverse volatility over 10 days"
    t = re.sub(r"\binverse (?:of (?:the )?)?(\d+ (?:day|week|month)s?) (?:volatility|vol|standard deviation(?: of returns?)?)\b",
               r"inverse volatility over \1", t)
    for pat, name in WEIGHT_METHODS:
        m = re.fullmatch(rf"(?:and |then |with |using )?(?:weight(?:ed|ing)? (?:by |using )?|use )?{pat}(?:[- ]weight(?:ed|ing|s)?)?"
                         rf"(?:(?: weighted)?(?: over| using| with)? (?:an? |the )?(?:last |past )?(\d+) (day|week|month)s?(?: lookback| window| volatility)?)?", t)
        if m:
            return name, (_period(m.group(1), m.group(2)) if m.group(1) else None)
    if re.search(r"\bweight(?:ed|ing|s)?\b", t):
        raise ParseError(f"Unknown weighting {s.strip()!r}. Use equal, inverse volatility, risk parity, minimum variance, "
                         f"max Sharpe, max diversification or market cap (optionally 'over N days').")
    return None


def _split_top_level(s: str, seps: str = r",|;| and | plus |&") -> list[str]:
    """Split on separators that are not inside parentheses or backticks."""
    out, depth, tick, buf, i = [], 0, False, "", 0
    rx = re.compile(seps)
    while i < len(s):
        ch = s[i]
        if ch == "`":
            tick = not tick
        elif not tick and ch == "(":
            depth += 1
        elif not tick and ch == ")":
            depth -= 1
        if depth == 0 and not tick:
            m = rx.match(s, i)
            if m and m.end() > i:
                out.append(buf)
                buf = ""
                i = m.end()
                continue
        buf += ch
        i += 1
    out.append(buf)
    return [x.strip() for x in out if x.strip()]


def _strip_parens(s: str) -> str:
    s = s.strip()
    while s.startswith("(") and s.endswith(")"):
        depth = 0
        for i, ch in enumerate(s):
            depth += ch == "("
            depth -= ch == ")"
            if depth == 0 and i < len(s) - 1:
                return s
        s = s[1:-1].strip()
    return s


SIDE_WORDS = {"short", "shorting", "shorted", "sell", "selling", "sold", "exit", "cover", "long"}
EXIT_WORDS = r"(?:sell|exit|cover|close(?: out)?|get out(?: of)?|liquidate|dump)"
_NOT_SIDE = r"(?i)[- ]?term\b|treasur|bonds?\b"     # "short-term treasuries", "long bonds" are names, not directions


def _holding_side(it: str) -> tuple[str, str | None]:
    """Direction words on one holding: 'short X', 'go short X', 'sell short X', 'X short' -> (X, 'short');
    'buy X', 'hold X', 'go long X', 'X long' -> (X, 'long'). 'sell/exit/cover X' inside a list of holdings is
    refused: it could mean cash, a short or leaving X out (in an if branch it means cash, see _branch)."""
    t = it.strip()
    m = re.fullmatch(r"(?is)(?:(?:and|then) )?(?:(?:hold|be|stay|go|goes|going|sell) )?short(?:[- ]sell)?\s+(?:in |on )?(.+)", t)
    if m and not re.match(_NOT_SIDE, m.group(1)):
        return m.group(1), "short"
    m = re.fullmatch(r"(?is)(?:(?:and|then) )?(?:(?:hold|be|stay|buy|keep|own) )?(.+?)\s+short", t)
    if m and not re.search(r"(?i)\b(?:long|short)\b", m.group(1)):
        return m.group(1), "short"
    m = re.fullmatch(r"(?is)(?:(?:and|then) )?(?:(?:(?:hold|be|stay|go|goes|going|buy) )?long|buy|hold|own|keep)\s+(.+)", t)
    if m and not re.match(_NOT_SIDE, m.group(1)):
        return m.group(1), "long"
    m = re.fullmatch(r"(?is)(.+?)\s+long", t)
    if m and not re.search(r"(?i)\b(?:long|short)\b", m.group(1)):
        return m.group(1), "long"
    m = re.fullmatch(rf"(?is)(?:(?:and|then) )?{EXIT_WORDS}\b(.*)", t)
    if m:
        raise ParseError(f"{it.strip()!r} inside a list of holdings is ambiguous: it could mean holding cash instead, a short "
                         "position, or leaving it out. Write 'short X' for a short position, 'cash' for cash, or leave it out. "
                         "(In an if/otherwise branch, 'sell X' means going to cash.)")
    return t, None


def _borrow_default(t: str) -> str:
    from .margin import default_borrow_fee
    return f"{default_borrow_fee(t):.1%}"


def _short_node(kids: list[dict], what: str) -> dict:
    """A short position in a holding: -100% of it plus the sale proceeds (200% in all) in cash."""
    if len(kids) != 1 or "asset" not in kids[0]:
        raise ParseError(f"'short {what.strip()}': name one ticker to short, e.g. 'short SQQQ'.")
    tk = kids[0]["asset"]
    _note(f"'short {tk}' is a short position the size of its slice: -100% {tk} with the sale proceeds held as cash "
          f"(so the slice shows 200% cash). The proceeds earn the T-bill rate less the short rebate spread (0.25%/yr by "
          f"default) and the short pays an assumed borrow fee ({_borrow_default(tk)}/yr for {tk}: 5% for leveraged, "
          f"inverse and volatility ETPs, 0.3% for other tickers; say 'borrow fee 2%' to set it, 'no borrow fee' for "
          f"none). Leveraged and inverse ETFs are often costly or impossible to borrow, and a short loses when {tk} "
          f"rises, with no upper limit.")
    return {"weights": "specified", "w": [-1.0, 2.0], "children": [{"asset": tk}, {"cash": True}]}


def _branch(text: str, notes: list[str]) -> dict:
    """An if/otherwise branch: any node, or 'sell X' / 'exit' / 'cover X' / 'sell everything' = go to cash."""
    t = _strip_parens(text.strip().strip(",;. "))
    m = re.fullmatch(rf"(?is)(?:(?:and|then) )?{EXIT_WORDS}(?:\s+(?:all|everything|it|them|out|the position|positions?|of))*"
                     rf"(?:\s+(?P<what>.+?))?", t)
    if m and not re.match(r"(?i)short\b", m.group("what") or ""):
        what = m.group("what")
        if what:
            kids = _asset_list(what, choice=True)
            if not all("asset" in k for k in kids):
                raise ParseError(f"{t!r}: name the tickers to sell, e.g. 'otherwise sell TQQQ' (= hold cash).")
            names = ", ".join(k["asset"] for k in kids)
        else:
            names = "everything"
        notes.append(f"'{t}' in a branch is read as selling {names} and holding cash (earning the T-bill rate) while that "
                     f"branch applies; say 'short {names if what else 'X'}' for a short position.")
        return {"cash": True}
    return _node(text, notes)


def _or_holdings(s: str) -> None:
    """'TQQQ or TMF' in a list of holdings is a choice with no rule for choosing: refused (never read as both).
    'or' is fine where something does choose (a filter's candidates: 'the top 1 of TQQQ or TMF by ...')."""
    if not re.search(r"(?i)\s(?:or|and/or)\s", _mask(s)) or re.search(r"(?i)\bor else\b", _mask(s)):
        return
    parts = [p for p in _split_top_level(s, r",| and | or | and/or |/| plus |&") if p]
    if len(parts) < 2:
        return
    either = re.sub(r"(?i)^\s*either\s+", "", s.strip())
    both = re.sub(r"(?i)\s+(?:and/)?or\s+", " and ", either)
    raise ParseError(f"'{s.strip()}': 'or' between holdings is a choice, and nothing says how to choose, so it is not "
                     f"read as holding both. To hold all of them write '{both}'{'' if '%' in s else ' (equal weight)'}; to hold one of them, say "
                     f"how to pick it, e.g. 'the top 1 of {both} by 20 day return' (or by RSI, volatility, ...), or use "
                     "'if <condition> then ... otherwise ...'.")


def _asset_list(s: str, choice: bool = False) -> list[dict]:
    """'QQQ, SPY and cash' -> [{'asset': 'QQQ'}, {'asset': 'SPY'}, {'cash': True}]. 'or' between holdings is refused
    unless `choice` (something picks among them: a filter, a rotation)."""
    if not choice:
        _or_holdings(s)
    out = []
    for it in _split_top_level(s, r",| and | or |/| plus |&"):
        it = _strip_parens(it)
        it, side = _holding_side(it)
        if side == "short":
            out.append(_short_node(_asset_list(it, choice=choice), it))
            continue
        if re.fullmatch(r"(?i)(?:hold |in )?(?:cash|t-?bills? as cash|money market)", it):
            out.append({"cash": True})
            continue
        # an asset-class word ('gold', 'Gold', 'bonds', 'REITs'), not an all-capitals ticker: class_series' rule
        cw = _class_word(re.sub(r"(?i)^(?:the )", "", it.strip()))
        if cw:
            out.append({"asset": cw})
            continue
        _not_a_word_ticker(it)
        probe = it.upper() if re.fullmatch(r"[a-z^$]{1,6}", it) else it
        tk = find_tickers(probe, strict=True)
        if len(tk) > 1 and re.search(r"(?i)\b(?:else|otherwise)\b", it):
            raise ParseError(f"{it!r}: an 'otherwise' with no 'if' before it. Write 'if <condition> then <holding> "
                             "otherwise <holding>'.")
        if len(tk) > 1:
            raise ParseError(f"{it!r} names several tickers ({', '.join(tk)}): separate them with commas or 'and', "
                             "or put a group in brackets with its weights, e.g. '(60% TECL and 40% BIL)'.")
        if not tk:
            words = [w for w in re.findall(r"[A-Za-z%'^$]+|\d+", it) if w.lower() not in STOP]
            raise ParseError(f"No ticker found in {it!r}" + (f" (not understood: {' '.join(words)!r})" if words else
                                                             " (it is not a phrase this parser reads)")
                             + ". Name a ticker (`python -m backtester tickers` lists all) or 'cash'.")
        rest = re.sub(rf"(?<![\w])[\$^]?{re.escape(tk[0].lstrip('^'))}\b", " ", probe, flags=re.I)
        for name, sym in COMPANIES.items():
            if sym == tk[0]:
                rest = re.sub(rf"\b{re.escape(name)}\b", " ", rest, flags=re.I)
        left = [w for w in re.findall(r"[a-z%']+|\d+", rest.lower()) if w not in STOP or w in SIDE_WORDS]
        if left:
            raise ParseError(f"Could not interpret {' '.join(left)!r} in {it!r}"
                             + (": write 'short X' for a short position, or 'cash'" if set(left) & SIDE_WORDS else ""))
        out.append({"asset": _settle_class(tk[0])})
    if not out:
        raise ParseError(f"No holdings found in {s!r}")
    return out


def _children(s: str, notes: list[str], choice: bool = False) -> list[dict]:
    """A list of holdings where each item is a ticker, cash, or a bracketed group of any kind:
    'TQQQ, (60% TECL and 40% BIL) and SVIX'."""
    if not choice:
        _or_holdings(s)
    out = []
    for it in _split_top_level(s, r",| and | or |/| plus |&"):
        it = it.strip()
        if it.startswith("(") and _strip_parens(it) != it:
            out.append(_node(_strip_parens(it), notes))
        else:
            out.extend(_asset_list(it, choice=True))
    if not out:
        raise ParseError(f"No holdings found in {s!r}")
    return out


VERB = r"(?:hold |buy |own |be in |invest in |go (?:to|into) |switch (?:to|into) |rotate (?:to|into) |stay in |be |allocate to |put (?:it |everything )?in(?:to)? )"


IF_BOUNDARY = (r"(?:\s*,)?\s+then\b[\s,:]*"                                   # "if C then X"
               r"|(?:\s*,)?\s+(?=(?:hold|buy|own|be in|invest in|go (?:to|into)|switch (?:to|into)|rotate (?:to|into)|stay in|be|"
               r"allocate to|put (?:it |everything )?in(?:to)?)\b)"                  # "if C hold X"
               r"|\s*,\s*")                                                         # "if C, X"
ELSE_RX = r"(?:\s*[,;]\s*|\s+)(?:and\s+)?(?:otherwise|else|or else)\b[\s,:]*"


def _if_chain(s: str, notes: list[str]) -> dict:
    """'if C [then] X, else if C2 [then] Y, otherwise Z' -> nested if nodes.

    A small recursive-descent reader: the condition ends at the first 'then', holding verb or comma
    (outside backticks and brackets) after which the condition parses completely; the THEN branch runs
    to the first top-level otherwise/else (an 'only if' inside the branch claims the next one); the
    ELSE branch is any node, including another if. A branch that is itself an if/else goes in
    parentheses or brackets: 'if A then (if B then X else Y) else Z'."""
    mk = _mask(s)
    head = re.match(r"(?i)\s*if\s+", mk)
    body = head.end()
    first_err = None
    for b in re.finditer(IF_BOUNDARY, mk[body:], re.I):
        if b.start() == 0:
            continue
        cond = s[body: body + b.start()]
        try:
            on, rule = _condition_on(cond, None)
        except ParseError as e:
            # the question "all of them or any?" beats a failed shorter reading ("if SPY" of "if SPY, QQQ RSI ...")
            first_err = e if isinstance(e, AmbiguousTickers) and not isinstance(first_err, AmbiguousTickers) else first_err or e
            continue
        rest = s[body + b.end():]
        mr = _mask(rest)
        claimed, chosen = 0, None
        for e in re.finditer(ELSE_RX, mr, re.I):
            only = len(re.findall(r"(?i)\bonly (?:if|when)\b", mr[: e.start()]))
            if only > claimed:
                claimed += 1        # this otherwise belongs to a filter's "only if ..."
                continue
            chosen = e
            break
        if chosen is None:
            raise ParseError(f"'{s.strip()}': an 'if' needs an 'otherwise' branch, e.g. '..., otherwise hold BIL' "
                             "(or 'otherwise cash'). Put a nested if/else in parentheses.")
        then_txt, else_txt = rest[: chosen.start()], rest[chosen.end():]
        return {"if": rule, "on": on, "then": _branch(then_txt, notes), "else": _branch(else_txt, notes)}
    raise first_err or ParseError(f"Could not find the condition in {s.strip()!r}: write 'if <condition> then <holding> "
                                  "otherwise <holding>'.")


# "CAPE-based allocation": a valuation-driven stock/bond mix. Stocks get 80% while the (point-in-time, lagged) Shiller
# CAPE is in the cheapest third of its history since 1881, 60% in the middle third, 40% in the most expensive third;
# bonds the rest. Checked monthly (the CAPE changes once a month).
CAPE_TIERS = ((1 / 3, 80), (2 / 3, 60), (None, 40))
CAPE_MODEL_RX = (r"(?:the |a )?(?:shiller(?:'s)? )?(?:cape|valuation)[- ](?:based |driven |tilted )?(?:tactical )?(?:asset )?"
                 r"allocation(?: (?:model|strategy|portfolio))?(?: (?:between|of|with|using|on|across) (?P<a>.+?) and (?P<b>.+?))?")


def _cape_model(s: str, notes: list[str]) -> dict | None:
    """'CAPE-based allocation between VTI and BND' (or without the funds: VTI and BND, with long-history series
    for an early start) -> nested if nodes on cape_pct(0) with the CAPE_TIERS stock weights."""
    m = re.fullmatch(rf"(?i){CAPE_MODEL_RX}", s.strip())
    if not m:
        return None
    if m.group("a"):
        kids = _asset_list(f"{m.group('a')} and {m.group('b')}")
        if len(kids) != 2:
            raise ParseError(f"'{s.strip()}': a CAPE-based allocation takes a stock fund and a bond fund, e.g. "
                             "'CAPE-based allocation between VTI and BND'.")
    else:
        base = _model_node("CAPE-based allocation", [(60, "VTI"), (40, "BND")],
                           "a valuation-driven stock/bond mix; this 60/40 is its middle tier", notes)
        kids = base["children"]
        notes.pop(next(i for i in range(len(notes) - 1, -1, -1) if notes[i].startswith("CAPE-based allocation:")))
    stock, bond = kids
    on = stock.get("asset") or bond.get("asset")
    if not on:
        raise ParseError(f"'{s.strip()}': name at least one fund (the stock side), e.g. 'CAPE-based allocation between "
                         "VTI and cash'.")

    def mix(w):
        return {"weights": "specified", "w": [w / 100, round(1 - w / 100, 10)], "children": [stock, bond]}
    node = mix(CAPE_TIERS[-1][1])
    for cut, w in reversed(CAPE_TIERS[:-1]):
        node = {"if": f"cape_pct(0) <= {cut:.4f}", "on": on, "then": mix(w), "else": node}
    _TL.tactical = "CAPE-based allocation"
    sname = _pf_short_name(stock)
    bname = _pf_short_name(bond)
    notes.append(f"CAPE-based allocation: {sname} {CAPE_TIERS[0][1]}% / {bname} {100 - CAPE_TIERS[0][1]}% while Shiller's "
                 f"CAPE is in the cheapest third of its history since 1881 (its percentile among the values known at "
                 f"the time), {CAPE_TIERS[1][1]}/{100 - CAPE_TIERS[1][1]} in the middle third, {CAPE_TIERS[2][1]}/"
                 f"{100 - CAPE_TIERS[2][1]} in the most expensive third. The CAPE of month M is used from the start of "
                 f"month M+{data.CAPE_LAG_MONTHS + 1} (its earnings are reported late), so the model never sees data "
                 "it could not have had. Write your own tiers with cape(), cape_pct(years) or earnings_yield() - "
                 "treasury_10y() in backticks.")
    return node


def _except_when(s: str, notes: list[str]) -> dict | None:
    """'TQQQ, except when SPY is below its 200 day moving average hold BIL' = if <condition> then BIL else TQQQ
    (also 'except while / if ...', '..., in which case BIL'). None when the phrase is not of that shape."""
    m = _msearch(r"^(?P<d>.+?),?\s+except (?:when(?:ever)?|while|if|as long as|on days when|during periods when)\s+(?P<rest>.+)$",
                 s, flags=re.I | re.S, match=True)
    if not m or re.search(r"(?i)\b(?:otherwise|else)\b", _mask(m.group("rest"))):
        return None
    rest = _sub_outside(r"(?i),?\s+in which case,?\s+(?:(?:hold|buy|own|be in|switch to|go to|move to) )?", ", hold ", m.group("rest"))
    try:
        node = _if_chain(f"if {rest}, otherwise {m.group('d')}", notes)
    except ParseError as e:
        if "needs an 'otherwise'" not in str(e) and "Could not find the condition" not in str(e):
            raise
        raise ParseError(f"'{s.strip()}': what should it hold while that condition is true? Name it after the condition: "
                         f"'{m.group('d').strip()}, except when <condition> hold <holding>' (a ticker, a group or cash).") from None
    notes.append(f"'{m.group('d').strip()}, except when ...' holds {_pf_short_name(node['then'])} while `{node['if']}` is true "
                 f"and {_pf_short_name(node['else'])} the rest of the time.")
    return node


def _index_today_note(uname, tickers, notes: list[str]) -> bool:
    """The survivorship warning of a today's-members index universe (index_universes); True when it is one."""
    from . import index_universes
    if not index_universes.is_today_members(uname):
        return False
    w = index_universes.warning(uname, len(tickers))
    if w not in notes:
        notes.append(w)
    return True


_WHOLE_W = (r"(?:(?:weighted\s+)?(?:by\s+)?(?:equal(?:ly)?|market[- ]cap(?:italization)?|inverse[- ]vol(?:atility)?)"
            r"(?:[- ]weight(?:ed|s|ing)?)?|weighted\s+equally|(?:in|with|at)\s+equal\s+weights?)")


def _whole_universe_node(s: str, notes: list[str]) -> dict | None:
    """A whole index held at once: "all Nasdaq 100 stocks equally weighted", "equal weight all Nasdaq 100 stocks",
    "today's S&P 500 members, market cap weighted" -> a filter selecting every member (select "all"), point-in-time
    for the Nasdaq 100. None when `s` is not only an index universe (and its weighting)."""
    s = s.strip()
    m = (re.fullmatch(rf"(?is)(?P<w>{_WHOLE_W})\s+(?:(?:of|in|across)\s+)?(?P<u>.+)", s)
         or re.fullmatch(rf"(?is)(?P<u>.+?),?\s+(?:(?:and|with|using)\s+)?(?P<w>{_WHOLE_W})", s)
         or re.fullmatch(r"(?is)(?P<u>.+)", s))
    u = m.group("u")
    rest = re.sub(rf"(?i){UNIVERSE_WORDS}", " ", u)
    rest = re.sub(r"(?i)\b(?:all|each|every|of|the|stocks?|members?|components?|constituents?|companies|in|index)\b", " ",
                  rest)
    if rest.strip() or not re.search(rf"(?i){UNIVERSE_WORDS}", u) or re.search(r"(?i)sector|tickers|universe", u):
        return None
    uu, uname = _universe_phrase(u)
    if uu is None:
        return None
    wtxt = m.groupdict().get("w")
    wp = _weighting_phrase(wtxt) if wtxt else ("equal", None)
    if wp is None:
        return None
    f: dict = {"select": "all", "by": "close", "weights": wp[0]}
    if wp[1] and wp[0] != "equal":
        f["lookback"] = wp[1]
    node: dict = {"filter": f, "universe": "NDX" if uname == "NDX" else uu}
    if uname != "NDX":
        _index_today_note(uname, uu, notes)
    if not wtxt:
        notes.append(f"Every stock of {'the Nasdaq-100' if uname == 'NDX' else uname} held at once, in equal weights "
                     "(no weighting stated).")
    return node


def _times_leverage(s: str, notes: list[str]) -> str:
    """'QQQ with 2x leverage' / 'QQQ at 2x' (one ticker, stated leverage; in a branch, see _branch_leverage) ->
    '200% QQQ' (the extra 100% borrowed, see _levered_weights). '2x QQQ' is not read here: leverage written before a
    ticker may mean the leveraged fund (QLD), so _review_phrases refuses it and asks. A ticker only: '3x S&P 500' is
    left alone."""
    m = re.fullmatch(r"(?i)(?P<t>[\^$]?[A-Z][A-Z0-9.\-]{0,9})\s+(?:with|at|using)\s+(?P<k>\d+(?:\.\d+)?)\s*x(?:\s+leverage)?", s)
    if not m or not m.group("t").isupper():
        return s
    k = float(m.group("k"))
    tk = _one_ticker(m.group("t"))
    if tk is None or tk == "cash" or not 1 < k <= 4:
        return s
    note = (f"'{s}' read as {k * 100:g}% {tk}: the extra {k * 100 - 100:g}% is borrowed (a margin loan at the T-bill "
            f"rate plus the margin spread), rebalanced back to {k:g}x on each rebalance. For a leveraged ETF, name it.")
    if note not in notes:
        notes.append(note)
    return f"{k * 100:g}% {tk}"


def _branch_leverage(rest: str, m) -> bool:
    """'hold QQQ with 2x leverage' written inside a branch of a conditional allocation ('if ... hold QQQ with 2x
    leverage, otherwise hold TLT'), attached to one ticker with no comma between: that branch's position is levered
    (read by _times_leverage as 200% QQQ, the excess borrowed), not every weight of the portfolio. Written after a
    comma, or on a sentence with no if/otherwise, it is the portfolio's leverage as before."""
    said = m.group(0)
    if said[:1] in ",;" or not re.match(r"\s?(?:with|using|at)\b", said, flags=re.I):
        return False
    if not re.search(r"(?i)\b(?:if|when|whenever|while|unless)\b", rest[: m.start()]) \
            or not re.search(r"(?i)\b(?:otherwise|else|then)\b", rest):
        return False
    before = re.search(r"(?:^|\b(?:hold|buy|own|in|into|to|then|else|otherwise)\s+)([\^$]?[A-Z][A-Z0-9.\-]{0,9})\s*$", rest[: m.start()])
    return bool(before) and before.group(1).lstrip("$") not in NOT_TICKERS and data.canonical(before.group(1).lstrip("$")) in _known()


def _levered_weights(ws: list[float], kids: list[dict], s: str, notes: list[str]) -> dict:
    """Stated weights adding up to more than 100% (e.g. '150% SPY and 50% TLT', or '200% QQQ' in an if-branch): the
    excess is borrowed, as with portfolio leverage - a cash leg of minus the excess (a margin loan at the T-bill
    rate plus the margin spread; the margin rules of Portfolio.validate apply to the gross exposure)."""
    tot = sum(ws)
    if tot <= 1 + 1e-6 or any(w < 0 for w in ws) or any(k.get("cash") for k in kids) or tot > 4 + 1e-9:
        raise ParseError(f"Weights add up to {tot:.0%}, not 100%.")
    note = (f"Weights in '{s.strip()}' add up to {tot:.0%}: the extra {tot - 1:.0%} is borrowed ({tot:.2g}x gross "
            "exposure, a margin loan at the T-bill rate plus the margin spread). If that was a typo, make them add up "
            "to 100%.")
    if note not in notes:
        notes.append(note)
    return _weights_node(ws + [1 - tot], kids + [{"cash": True}], s)


def _node(text: str, notes: list[str] | None = None) -> dict:
    """Parse an allocation phrase into a portfolio tree node (strict: every word must be understood)."""
    notes = notes if notes is not None else []
    text = _sub_outside(r"\[", "(", _sub_outside(r"\]", ")", text))
    s = _strip_parens(text.strip().strip(",;. "))
    s = re.sub(r"(?i)^(?:and |then )?(?:hold|buy and hold|buy|own|be in|select|pick|choose|invest(?: in)?|allocate(?: to)?|put (?:everything |it all |all )?in(?:to)?|go (?:to|into)|switch (?:to|into)|rotate (?:to|into)|stay in|move (?:to|into)|in)\s+", "", s)
    s = _strip_parens(s)
    s = _times_leverage(s, notes)
    low = s.lower()
    if re.fullmatch(r"(?:cash|t-?bills|treasury bills|money market|nothing|flat)", low):
        return {"cash": True}
    wu = _whole_universe_node(s, notes)
    if wu is not None:
        return wu

    cm = _cape_model(s, notes)
    if cm:
        return cm
    tm = _tactical_model(s, notes)
    if tm:
        return tm
    mp = _model_portfolio(s, notes)
    if mp:
        return mp
    # a bare asset-class name ("long-term corporate bonds", "US small cap value"): 100% of its long-history series,
    # unless the phrase already names a fund ("long-term treasuries" = TLT)
    ma = re.fullmatch(rf"(?i)(?:100% (?:in |of )?)?(?:the )?({ASSET_CLASS_RX})", s)
    if ma:
        hit = _class_word(ma.group(1))
        if hit:
            return {"asset": hit}
    # if COND [then] X, (else if COND [then] Y,)* otherwise Z   -- any node in any branch
    if re.match(r"(?i)if\b", s):
        return _if_chain(s, notes)
    # "SPY unless SPY is below its 200 day moving average, in which case BIL" = if <condition> then BIL else SPY
    mu = _msearch(r"^(?P<d>.+?),? unless (?P<c>.+?),? in which case,? (?P<a>.+)$", s, flags=re.I | re.S, match=True)
    if mu:
        on, rule = _condition_on(mu.group("c"), None)
        notes.append(f"'... unless <condition>, in which case ...' holds the 'in which case' branch while `{rule}` is true, "
                     "and the first holding otherwise.")
        return {"if": rule, "on": on, "then": _branch(mu.group("a"), notes), "else": _branch(mu.group("d"), notes)}
    # "when/whenever/while C hold X, otherwise Y" = if; "unless C hold X, otherwise Y" = if C then Y else X
    mw = re.match(r"(?i)(when(?:ever)?|while|unless)\s+", s)
    if mw and re.search(ELSE_RX, _mask(s), re.I):
        node = _if_chain("if " + s[mw.end():], notes)
        if mw.group(1).lower() == "unless":
            node = {**node, "then": node["else"], "else": node["then"]}
            notes.append(f"'unless <condition> ..., otherwise ...' holds the otherwise branch while `{node['if']}` is true "
                         "and the first holding while it is false.")
        return node
    # per-asset timing: "SPY, EFA, IEF, VNQ and DBC equally, each only when above its 10 month moving average,
    # otherwise cash" -> equal weights of (asset if its own condition else cash)
    m = _msearch(r"^(?P<lst>.+?)(?:,? (?:equally|in equal weights?|(?:with )?equal weights?|equal[- ]weight(?:ed)?))?,? "
                 r"(?:each|every one|each one|each of them|each asset) (?:(?:is )?(?:held )?)?(?:only )?(?:when|if|while|as long as|whenever) "
                 r"(?:it is |it's |it )?(?P<cond>.+?),? (?:and )?(?:otherwise|else|or else)[, ]+(?:(?:hold|in|be in) )?(?P<other>.+)$", s,
                 flags=re.I | re.S, match=True)
    if m:
        lst = m.group("lst")
        # "20% each of SPY, EFA, ...": a stated weight per sleeve (the default is equal shares)
        each = re.match(r"(?i)\s*(?:(\d+(?:\.\d+)?)% (?:each|apiece)|(?:in )?equal(?:ly)?(?:[- ]weight(?:ed|s)?)?|an equal (?:share|weight))"
                        r"(?: (?:of|in|to|into))?\s+", lst)
        if each:
            lst = lst[each.end():]
        assets = _asset_list(lst)
        if len(assets) < 2 or not all("asset" in a for a in assets):
            raise ParseError(f"'{s.strip()}': 'each only when ...' needs a list of tickers, e.g. 'SPY, EFA and IEF equally, each "
                             "only when above its 10 month moving average, otherwise cash'.")
        cond = m.group("cond").strip()
        if re.match(r"(?i)(?:above|below|over|under|at least|at most|greater than|less than|higher than|lower than|up|down|within|between)\b", cond):
            cond = "is " + cond
        other = _node(m.group("other"), notes)
        kids = []
        for a in assets:
            tk = a["asset"]
            on, rule = _condition_on(f"{tk} {cond}", tk)
            kids.append({"if": rule, "on": on, "then": {"asset": tk}, "else": other})
        _TL.tactical = "Per-asset timing"
        pct = float(each.group(1)) / 100 if each and each.group(1) else None
        notes.append(f"Each of {', '.join(a['asset'] for a in assets)} gets "
                     + (f"{pct * 100:g}%" if pct else "an equal share")
                     + f", held only while its own condition is true ({kids[0]['if']} for {assets[0]['asset']}); otherwise "
                     f"that sleeve goes to {'cash' if other.get('cash') else _pf_short_name(other)}, each sleeve switching on its own.")
        if pct is None or abs(pct * len(kids) - 1) < 1e-9:
            return {"weights": "equal", "children": kids}
        if pct * len(kids) > 1 + 1e-9:
            raise ParseError(f"{pct:.0%} each of {len(kids)} holdings is {pct * len(kids):.0%}, more than 100%.")
        notes.append(f"{pct:.0%} each of {len(kids)} holdings is {pct * len(kids):.0%}: the remaining "
                     f"{1 - pct * len(kids):.0%} is held in cash.")
        return {"weights": "specified", "w": [round(pct, 10)] * len(kids) + [round(1 - pct * len(kids), 10)],
                "children": kids + [{"cash": True}]}
    ex = _except_when(s, notes)
    if ex:
        return ex
    # X if COND else Y
    m = _msearch(rf"(.+?) (?:if|when|while|as long as|whenever) (.+?),? (?:and )?(?:otherwise|else|or else)[, ]+{VERB}?(.+)$", s,
                 flags=re.I | re.S, match=True)
    if m and not re.search(r"(?i)\b(?:top|bottom|best|worst)\s+\d", m.group(1)) and not re.search(r"(?i)\bonly\s*$", m.group(1)):
        then_node = _branch(m.group(1), notes)
        on, rule = _condition_on(m.group(2), then_node.get("asset"))
        return {"if": rule, "on": on, "then": then_node, "else": _branch(m.group(3), notes)}

    # dual momentum
    m = re.match(r"(?i)dual momentum (?:between|among|of|on|with) (.+?)(?:,? (?:with|using) (.+?) as (?:the )?(?:safe|defensive|risk[- ]off) (?:asset|haven)| (?:otherwise|else|or) (.+?))?(?:,? (?:(?:using|with|on|by) )?(?:a |the )?(\d+) (month|day|week) (?:lookback|momentum|(?:total )?returns?)"
                 r"(?: (?:versus|vs\.?|against|relative to|over|above) (?:t-?bills?|treasury bills|cash|the risk[- ]free rate))?)?$", s)
    if m:
        kids = _asset_list(m.group(1), choice=True)
        safe = m.group(2) or m.group(3) or "AGG"
        n = _period(m.group(4) or 12, m.group(5) or "month")
        _TL.tactical = "Dual momentum (Antonacci)"
        return {"filter": {"select": "top", "n": 1, "by": f"tret({n})", "require": f"tret({n}) > tbill_ret({n})"},
                "universe": "children", "children": kids, "fallback": _node(safe, notes)}

    # "the top 2 by 10 day RSI of A, B and C" -> "the top 2 of A, B and C by 10 day RSI"; "the highest 10 day RSI of A,
    # B and C" -> the top 1 of them by it ("the lowest ..." the bottom 1)
    mrk = (re.fullmatch(r"(?is)(?:the )?(?P<sel>top|bottom|best|worst) (?P<n>\d+) (?:by|on|ranked by|sorted by) (?P<met>.+?) "
                        r"(?:of|among|from|in) (?P<lst>.+)", s)
           or re.fullmatch(r"(?is)(?:the )?(?P<hl>highest|lowest|greatest|smallest)(?: (?P<n>\d+)(?! (?:day|week|month|year|period|bar|session)s?\b))? (?P<met>(?!perform|volatil).+?) "
                           r"(?:of|among|from|in) (?P<lst>.+)", s))
    if mrk:
        lst, rest = mrk.group("lst"), ""
        mo = re.search(r"(?i),\s*(?:and |then |with |using |but )?(?:weight|equal|inverse|only|risk|min|max|market|that|which)\b.*$", lst)
        if mo:
            lst, rest = lst[: mo.start()], lst[mo.start():]
        many = len(find_tickers(lst, strict=True)) >= 2 or _universe_phrase(lst)[0] is not None
        if many and not re.search(r"(?i)\b(?:by|ranked by|sorted by)\b", lst):
            sel = mrk.groupdict().get("sel") or ("top" if mrk.group("hl").lower() in ("highest", "greatest") else "bottom")
            s = f"the {sel} {mrk.group('n') or 1} of {lst} by {mrk.group('met')}{rest}"
            low = s.lower()
    # [equal weight] the top/bottom N (of) UNIVERSE by METRIC[, options]
    lead_w = None
    mw = re.match(r"(?i)((?:equal(?:ly)?|inverse[- ]vol(?:atility)?|risk[- ]parity|min(?:imum)?[- ]variance|market[- ]cap|max(?:imum)?[- ](?:sharpe|diversification))[- ]weight(?:ed)?)\s+(?:the )?(?=(?:top|bottom|best|worst|\d))", s)
    if mw:
        lead_w = _weighting_phrase(mw.group(1))
        s = s[mw.end():]
        low = s.lower()
    # "the 2 least volatile of ...", "the most volatile 2 of ...", "the best performing 2 of ... over the last 60 days"
    m = re.match(r"(?is)(?:the )?(?:(?P<n1>\d+) )?(?P<w>least|most|best|worst|top|bottom)[- ](?P<k>volatile|perform(?:ing|ers))"
                 r"(?: (?P<n2>\d+))? (?:of|among|from|in) (?:the )?(?P<uni>.+?)"
                 r"(?:,? (?:over|in|during|across|based on|using) (?:the )?(?:last |past |prior |trailing )?(?P<n>\d+) (?P<u>day|week|month|year)s?(?: volatility| returns?| performance)?)?"
                 r"(?P<rest>,\s*(?:and |then |with |using |but )?(?:weight|equal|inverse|only|risk|min|max|market|that|which)\b.*)?$", s)
    if m and not (m.group("n1") and m.group("n2")) and not re.search(r"(?i)\b(?:by|ranked by|sorted by)\b", m.group("uni")):
        n_sel = m.group("n1") or m.group("n2") or "1"     # "the best performing of X, Y and Z": the single best
        vol = m.group("k").lower() == "volatile"
        w = m.group("w").lower()
        if vol and w not in ("least", "most") or not vol and w not in ("best", "worst", "top", "bottom"):
            raise ParseError(f"'{m.group(0).strip()}': say 'least/most volatile' or 'best/worst performing'.")
        top = w in ("most", "best", "top")
        look = f"{m.group('n')} {m.group('u')} " if m.group("n") else ""
        if vol:
            if not look:
                notes.append(f"No lookback given for '{w} volatile': using 20 trading days.")
                look = "20 day "
            notes.append(f"'{w} volatile' ranks by {look.strip()} volatility: the annualised standard deviation of daily price "
                         "returns, as in '60 day volatility' elsewhere (say 'by 60 day standard deviation of return' to rank by "
                         "the daily figure of total returns, as Composer does).")
        s = f"{'top' if top else 'bottom'} {n_sel} of {m.group('uni')} by {look}{'volatility' if vol else 'return'}{m.group('rest') or ''}"
        low = s.lower()
    # "the 2 of X, Y and Z with the highest 10 day return" -> "top 2 of X, Y and Z by 10 day return"
    m = re.match(r"(?is)(?:the )?(\d+) (?:of|among|from) (?:the )?(.+?) with the (highest|lowest|best|worst|strongest|weakest|largest|"
                 r"smallest|biggest|greatest|most|least) (.+)$", s)
    if m:
        low_adj = m.group(3).lower() in ("lowest", "worst", "weakest", "smallest", "least")
        s = f"{'bottom' if low_adj else 'top'} {m.group(1)} of {m.group(2)} by {m.group(4)}"
        low = s.lower()
    m = re.match(r"(?is)(?:the )?(\d+) (best|worst|top|bottom)[- ]perform(?:ing|ers)(?: (?:of|among|from|in))? (?:the )?(.+?) over (?:the )?(?:last |past )?(\d+) (day|week|month|year)s?(,.*)?$", s)
    if m:
        s = f"{'top' if m.group(2).lower() in ('best', 'top') else 'bottom'} {m.group(1)} of {m.group(3)} by {m.group(4)} {m.group(5)} return{m.group(6) or ''}"
        low = s.lower()
    # "the 5 largest Nasdaq 100 stocks", "the largest 5 ... by market cap": size means market cap
    m = re.match(r"(?is)(?:the )?(?:(\d+) (largest|biggest|smallest)|(largest|biggest|smallest) (\d+)) (?:of |among |from |in )?(?:the )?"
                 r"((?:(?! by ).)+?)(?: by (?:market[- ]cap(?:itali[sz]ation)?|size))?(,.*)?$", s)
    if m:
        n_sz, big = m.group(1) or m.group(4), (m.group(2) or m.group(3)).lower() != "smallest"
        s = f"{'top' if big else 'bottom'} {n_sz} of {m.group(5)} by market cap{m.group(6) or ''}"
        low = s.lower()
    m = _msearch(r"(?is)(?:the )?(top|bottom|best|worst|strongest|weakest|highest|lowest) (\d+) (?:of |among |from |in )?(?:the )?(.+?) (?:by|ranked by|based on|sorted by|according to|with the (?:highest|lowest|best|strongest|weakest)) (.+)$", s, flags=re.I | re.S, match=True)
    if m:
        word, n, uni, rest = m.groups()
        parts = _split_top_level(_lookback_lists(rest), r",|;")
        metric_text, opts = parts[0], parts[1:]
        # a requirement glued to the metric: "... by 10 day cumulative return only if their 20 day RSI is above 50"
        mq = re.search(r"(?i)\s+((?:but )?only (?:if|when)\s.*)$", metric_text)
        if mq:
            opts.insert(0, mq.group(1))
            metric_text = metric_text[: mq.start()]
        # weighting may be glued to the metric: "... by 6 month return inverse volatility weighted"
        # the weighting's lookback glued to the metric: "... by 63 day return using a 10 day lookback"
        mlk = re.search(r"(?i),?\s+(?:using|with|over) (?:an? |the )?(\d+)[- ](day|week|month)s? (?:lookback|window)(?: for (?:the )?(?:weights?|weighting|volatility))?\s*$", metric_text)
        if mlk:
            opts.insert(0, mlk.group(0).strip(" ,"))
            metric_text = metric_text[: mlk.start()]
        mm = re.search(r"(?i)\s+((?:weighted |weight )?(?:by |using )?(?:equal(?:ly)?|inverse[- ]vol(?:atility)?|risk[- ]parity|min(?:imum)?[- ]variance|max(?:imum)?[- ](?:sharpe|diversification)|market[- ]cap)\b.*)$", metric_text)
        if mm:
            opts.insert(0, mm.group(1))
            metric_text = metric_text[: mm.start()]
        metric, mnotes = value_phrase(metric_text)
        notes.extend(mnotes)
        direction = "bottom" if re.search(LOW_WORDS, metric_text.lower()) else "top"
        if word.lower() in ("bottom", "worst", "weakest", "lowest"):
            direction = "bottom" if direction == "top" else "top"
        rw = _rank_words_note(word, int(n), metric_text, direction, metric)
        if rw:
            notes.append(rw)
        metric, dd_note = _dd_rank(metric, direction, metric_text, int(n))
        if dd_note:
            notes.append(dd_note)
        uu, uname = _universe_phrase(uni)
        wt, look = lead_w or ("equal", None)
        node: dict = {"filter": {"select": direction, "n": int(n), "by": metric}}
        if uname == "NDX":
            node["universe"] = "NDX"
        elif uu is not None:
            node["universe"] = uu
            _index_today_note(uname, uu, notes)
        else:
            node["universe"] = "children"
            node["children"] = _children(uni, notes, choice=True)
        gate, fallback = None, None
        look_set = None
        i = 0
        while i < len(opts):
            o = opts[i].strip()
            nxt = opts[i + 1].strip() if i + 1 < len(opts) else ""
            # "weighted 70/30": fixed weights by rank (the best pick 70%, the next 30%)
            mr = re.fullmatch(r"(?i)(?:and |then |with |using )?(?:(?P<a>weight(?:ed|ing)?(?: by rank)?|rank[- ]weighted) )?"
                              r"(?P<ws>\d+(?:\.\d+)?%?(?:\s*/\s*\d+(?:\.\d+)?%?)+)(?P<b> (?:weights?|split|weighting))?"
                              r"(?: by rank)?", o)
            if mr and (mr.group("a") or mr.group("b")):
                ws = [float(x.strip().rstrip("%")) for x in mr.group("ws").split("/")]
                if len(ws) != int(n):
                    raise ParseError(f"'{o}': {len(ws)} weights for {n} picks; give one weight per pick, best first "
                                     f"(e.g. 'top 2 ..., weighted 70/30').")
                if any(w <= 0 for w in ws) or abs(sum(ws) - 100) > 1e-6:
                    raise ParseError(f"'{o}': the weights by rank must be positive and add up to 100 (they add up to "
                                     f"{sum(ws):g}).")
                wt = "specified"
                node["filter"]["w"] = [round(w / 100, 10) for w in ws]
                notes.append(f"'{o}': fixed weights by rank (#1 is the {direction} pick): " + ", ".join(
                    f"#{j + 1} {w:g}%" for j, w in enumerate(ws)) + ". If fewer are picked on a day, their weights are "
                    "scaled to add up to 100%.")
                i += 1
                continue
            wp = _weighting_phrase(o)
            if wp:
                wt, look = wp[0], wp[1] or look
                i += 1
                continue
            mlk = re.fullmatch(r"(?i)(?:and )?(?:using|with|over) (?:an? |the )?(\d+)[- ](day|week|month)s? (?:lookback|window)(?: for (?:the )?(?:weights?|weighting|volatility))?", o)
            if mlk:
                look_set = _period(mlk.group(1), mlk.group(2))
                i += 1
                continue
            mo = re.fullmatch(r"(?is)(?:but )?only (?:if|when) (.+)", o)
            if mo:
                cond = mo.group(1)
                other = None
                mo2 = re.fullmatch(r"(?is)(.+?)\s+(?:otherwise|else)\s+(.+)", cond)
                if mo2:
                    cond, other = mo2.group(1), mo2.group(2)
                elif re.match(r"(?i)(?:otherwise|else)\b", nxt):
                    other = re.sub(r"(?i)^(?:otherwise|else)[, ]+", "", nxt)
                    i += 1
                own = re.match(r"(?i)(?:their|its|the (?:selected|chosen) (?:assets?|stocks?|ones?)'?s?)\s+(.+?)\s+(?:is |are )?(positive|negative|above (-?[\d.]+)%|below (-?[\d.]+)%|(?:beats?|exceeds?|is above|are above|is greater than|is higher than) (?:cash|t-?bills|the risk[- ]free rate))\s*$", cond.strip())
                if own and re.search(r"(?i)\s(?:and|or)\s", own.group(1)):
                    own = None      # "their RSI is above 50 and their return is positive": several comparisons
                pron = re.match(rf"(?i)\s*{PRONOUN_SUBJECT}\s", cond)
                if pron and not own:
                    # a relative hurdle against another ticker: each candidate's value vs that ticker's
                    rel = _relative_compare(cond)
                    if rel is None and not find_tickers(cond, strict=True):
                        # any indicator of each candidate: "only if their 20 day RSI is above 50"
                        own_text = re.sub(rf"(?i)^\s*{PRONOUN_SUBJECT}\s+", "", cond)
                        own_text = _sub_outside(rf"(?i)\s+(?:and|or)\s+{PRONOUN_SUBJECT}\s+", lambda m: m.group(0).split()[0].join("  "), own_text)
                        try:
                            rel = parse_conditions(own_text, [], total=True)
                        except ParseError as e:
                            if re.search(r"runs from|looks like a fraction|is a percentage|drop the % sign", str(e)):
                                raise     # understood, but the number cannot be meant
                            rel = None
                        if rel is not None and "sym(" in rel:
                            rel = None
                    if rel is None:
                        raise ParseError(f"Could not interpret the requirement {cond.strip()!r}: write e.g. 'only if their 12 "
                                         "month return is positive', '... is above 2%', 'their 20 day RSI is above 50', '... beats cash' or '... is above "
                                         "BIL's 12 month return'.")
                    node["filter"]["require"] = rel
                    node["fallback"] = _node(other, notes) if other else {"cash": True}
                    if not other:
                        notes.append("No 'otherwise' given for the requirement: a slot whose pick fails it is held in cash.")
                    i += 1
                    continue
                if own:
                    mexpr, mn = value_phrase(own.group(1))
                    notes.extend(mn)
                    look_n = re.search(r"(\d+)\)", mexpr)
                    rel = own.group(2).lower()
                    if rel == "positive":
                        req = f"{mexpr} > 0"
                    elif rel == "negative":
                        req = f"{mexpr} < 0"
                    elif rel.startswith("above"):
                        req = f"{mexpr} > {float(own.group(3)) / 100:g}"
                    elif rel.startswith("below"):
                        req = f"{mexpr} < {float(own.group(4)) / 100:g}"
                    else:
                        req = f"{mexpr} > tbill_ret({look_n.group(1) if look_n else 252})"
                    node["filter"]["require"] = req
                    node["fallback"] = _node(other, notes) if other else {"cash": True}
                else:
                    gate = cond
                    fallback = _node(other, notes) if other else {"cash": True}
                    if not other:
                        notes.append("No 'otherwise' given for the condition: holding cash when it is false.")
                i += 1
                continue
            mb = re.fullmatch(r"(?i)(?:that |which )?(?:beat|beats|outperform|outperforms) (cash|t-?bills|bil|the risk[- ]free rate)(?:,? (?:otherwise|else) (.+))?", o)
            if mb:
                look_n = re.search(r"(\d+)\)", metric)
                if mb.group(1).lower() == "bil":
                    # the BIL fund itself (its total return), not the T-bill rate
                    other_m, _ = value_phrase(metric_text, Ctx.for_ticker("BIL", total=True))
                    node["filter"]["require"] = f"{metric} > {other_m}"
                else:
                    node["filter"]["require"] = f"{metric} > tbill_ret({look_n.group(1) if look_n else 252})"
                other = mb.group(2)
                if not other and re.match(r"(?i)(?:otherwise|else)\b", nxt):
                    other = re.sub(r"(?i)^(?:otherwise|else)[, ]+", "", nxt)
                    i += 1
                node["fallback"] = _node(other, notes) if other else {"cash": True}
                i += 1
                continue
            raise ParseError(f"Could not interpret {o!r} in {text.strip()!r}")
        node["filter"]["weights"] = wt
        if look_set is not None:
            if wt in ("equal", "specified", "market_cap"):
                raise ParseError(f"'{text.strip()}': a {look_set} day lookback, but the picks are "
                                 f"{'equally' if wt == 'equal' else 'market-cap' if wt == 'market_cap' else 'fixed'} "
                                 "weighted, which has no lookback. Put the lookback in the ranking ('by 10 day return') or "
                                 "choose a weighting that uses one ('inverse volatility weighted').")
            look = look_set
        if look:
            node["filter"]["lookback"] = look
        if gate:
            on, rule = _condition_on(gate, None)
            return {"if": rule, "on": on, "then": node, "else": fallback}
        return node

    # rotate between A, B and C by METRIC  (top 1)
    m = re.match(r"(?is)(?:rotate|rotation|switch)(?: \w+)? (?:between|among|across) (.+?) (?:by|based on|using|according to) (.+)$", s)
    if m:
        metric, mn = value_phrase(m.group(2))
        notes.extend(mn)
        sel = "bottom" if re.search(LOW_WORDS, m.group(2).lower()) else "top"
        metric, dd_note = _dd_rank(metric, sel, m.group(2))
        notes.extend([dd_note] if dd_note else [])
        return {"filter": {"select": sel, "n": 1, "by": metric, "weights": "equal"},
                "universe": "children", "children": _asset_list(m.group(1), choice=True)}
    m = re.match(r"(?is)whichever of (.+?) has (?:the )?(higher|highest|lower|lowest|best|stronger|strongest|weaker|weakest) (.+)$", s)
    if m:
        metric, mn = value_phrase(m.group(3))
        notes.extend(mn)
        direction = "top" if m.group(2).lower() in ("higher", "highest", "best", "stronger", "strongest") else "bottom"
        metric, dd_note = _dd_rank(metric, direction, m.group(3))
        notes.extend([dd_note] if dd_note else [])
        return {"filter": {"select": direction, "n": 1, "by": metric, "weights": "equal"},
                "universe": "children", "children": _asset_list(m.group(1), choice=True)}

    # "SPY, TLT and GLD equally / in equal parts / equally weighted", "equal thirds of SPY, TLT and GLD"
    eq_words = (r"equally(?:[- ]weighted)?|evenly(?: split)?|split evenly|(?:in |with )?equal (?:parts|shares|weights?|amounts|"
                r"proportions|thirds|quarters|halves|fifths)|equal[- ]weighted|(?:in |with )?an equal (?:share|weight|amount) each")
    me = (_msearch(rf"^(?P<lst>.+?),?\s+(?P<w>{eq_words})$", s, flags=re.I | re.S)
          or _msearch(r"^(?:in )?(?P<w>equal (?:parts|shares|weights?|amounts|thirds|quarters|halves|fifths)|an equal (?:share|weight|amount)(?: each)?)"
                      r" (?:of|in|across|between|among) (?P<lst>.+)$", s, flags=re.I | re.S))
    if me and not re.search(r"(?i)\b(?:top|bottom|if|when|by)\b", _mask(me.group("lst"))):
        kids = _children(me.group("lst"), notes)
        parts = {"halves": 2, "thirds": 3, "quarters": 4, "fifths": 5}
        mp = re.search(r"(?i)\b(halves|thirds|quarters|fifths)\b", me.group("w"))
        if mp and parts[mp.group(1).lower()] != len(kids):
            raise ParseError(f"'{s.strip()}': {mp.group(1).lower()} of {len(kids)} holdings? Name "
                             f"{parts[mp.group(1).lower()]} holdings, or say 'equally'.")
        if len(kids) < 2:
            raise ParseError(f"'{s.strip()}': equal weights of what? Name at least two holdings.")
        return {"weights": "equal", "children": kids}
    # "TQQQ, TMF and SVIX weighted 50/30/20", "SPY, TLT, GLD 33% / 33% / 33%"
    mws = _msearch(r"^(?P<lst>.+?),?\s+(?:(?:weighted|weights?|in|at|split)\s+(?:of\s+)?)?"
                   r"(?P<w>-?\d+(?:\.\d+)?%?(?:\s*/\s*-?\d+(?:\.\d+)?%?)+)$", s, flags=re.I | re.S)
    if mws and not re.search(r"/", _mask(mws.group("lst"))):
        ws = [float(x.strip().rstrip("%")) for x in mws.group("w").split("/")]
        kids = _children(mws.group("lst"), notes)
        if len(ws) != len(kids):
            raise ParseError(f"{len(ws)} weights but {len(kids)} holdings in {s.strip()!r}")
        return _listed_weights([w / 100 for w in ws], kids, s, notes)

    # 60/40 SPY/TLT  or  SPY/TLT 60/40
    m = re.fullmatch(r"(?i)(-?\d+(?:\.\d+)?(?:/-?\d+(?:\.\d+)?)+) ([a-z^$ -]+(?:/[a-z^$ -]+)+)|([a-z^$ -]+(?:/[a-z^$ -]+)+) (-?\d+(?:\.\d+)?(?:/-?\d+(?:\.\d+)?)+)", s)
    if m:
        ws = [float(x) for x in (m.group(1) or m.group(4)).split("/")]
        names = (m.group(2) or m.group(3)).split("/")
        if len(ws) != len(names):
            raise ParseError(f"{len(ws)} weights but {len(names)} holdings in {s!r}")
        kids = [_node(x, notes) for x in names]
        return _weights_node([w / sum(ws) for w in ws], kids, s)

    # "60% VTI 40% BND", "VTI 60% BND 40%", "VTI 60, BND 40": weights with no separator (or bare numbers adding to 100)
    s = _bare_weights(s)
    # weighted list: 60% SPY, 30% TLT and 10% (if ... else ...)   /   SPY 60%, TLT 40%
    items = _split_top_level(s, r",| and | plus |;")
    pct_re = re.compile(r"(?is)^(-?\d+(?:\.\d+)?)% (?:in |of |into )?(?:the )?(.+)$|^(.+?) (-?\d+(?:\.\d+)?)%$")
    if items and any(pct_re.match(x) for x in items):
        ws, kids = [], []
        for x in items:
            mm = pct_re.match(x)
            if not mm:
                raise ParseError(f"Could not interpret {x!r}: every holding needs a weight, e.g. '60% SPY and 40% TLT'.")
            w = float(mm.group(1) or mm.group(4)) / 100
            ws.append(w)
            kids.append(_node(mm.group(2) or mm.group(3), notes))
        ne = _near_equal(ws, kids, notes)
        if ne:
            return ne
        if abs(sum(ws) - 1) > 1e-6:
            if sum(ws) < 1 - 1e-6 and not any(k.get("cash") for k in kids):
                kids.append({"cash": True})
                ws.append(1 - sum(ws))
                notes.append(f"Weights add up to {sum(ws[:-1]):.0%}: the remaining {ws[-1]:.0%} is held in cash.")
            else:
                return _levered_weights(ws, kids, s, notes)
        return _weights_node(ws, kids, s)

    # equal weight / inverse volatility / ... of a list
    mlw = re.match(r"(?is)^((?:weight(?:ed|ing)? )?(?:by |using )?(?:equal(?:ly)?|inverse[- ]vol(?:atility)?|risk[- ]parity|equal[- ]risk(?: contribution)?|min(?:imum)?[- ](?:variance|vol(?:atility)?)|max(?:imum)?[- ](?:sharpe(?: ratio)?|diversification)|market[- ]cap(?:italization)?)(?:[- ]weight(?:ed|ing|s)?)?)(?: in| of| across| between| among)?\s+(.+)$", s)
    if mlw:
        wp = _weighting_phrase(mlw.group(1))
        lst = mlw.group(2)
        look = wp[1]
        mlb = re.search(r"(?i),?\s*(?:using |with |over )(?:an? |the )?(?:last |past )?(\d+) (day|week|month)s?(?: lookback| window| volatility)?\s*$", lst)
        if mlb:
            look = _period(mlb.group(1), mlb.group(2))
            lst = lst[: mlb.start()]
        uu, uname = _universe_phrase(lst)
        if uu is not None:
            raise ParseError("Weighting a whole index needs a selection: say e.g. 'top 20 Nasdaq 100 stocks by 12 month momentum, equal weight'.")
        # each item: a bracketed group of any kind "(TLT and GLD equally)", a rule or weighted group, or a ticker
        # (the same groups as the list-then-weighting order "(...) and (...), inverse volatility weighted")
        node = {"weights": wp[0], "children": [
            _node(_strip_parens(x.strip()), notes) if x.strip().startswith("(") and _strip_parens(x.strip()) != x.strip()
            else _node(x, notes) if re.search(r"\b(?:if|top|bottom)\b|%", x, re.I) else _asset_list(x)[0]
            for x in _split_top_level(lst, r",| and |&| plus ")]}
        if look and wp[0] != "equal":
            node["lookback"] = look
        return node

    mw_ = _msearch(r"^(.+?),? (?:when(?:ever)?|while|if|as long as) (.+)$", s, flags=re.I | re.S)
    if mw_ and not re.search(r"(?i)\b(?:otherwise|else)\b", _mask(s)):
        x_ = mw_.group(1).strip()
        raise ParseError(f"'{s.strip()}': {x_} is held only while a condition holds, and nothing says what to hold the rest "
                         f"of the time. Add it: '{x_} when <condition>, otherwise <holding>' (a ticker, a group or cash); "
                         f"to hold {x_} except while a condition holds, write '{x_}, except when <condition> hold <holding>'. "
                         "For a trading rule on one ticker write 'buy <ticker> when ..., sell when ...'.")
    # a list followed by its weighting: "TQQQ, SOXL and TECL, weighted by inverse 10 day volatility"
    mtw = _msearch(r"^(?P<lst>.+?),?\s+(?P<w>(?:and |then |with |using )?(?:weight(?:ed|ing)? (?:by|using) .+|"
                   r"(?:inverse[- ]vol(?:atility)?|risk[- ]parity|equal[- ]risk(?: contribution)?|min(?:imum)?[- ](?:variance|vol(?:atility)?)|"
                   r"max(?:imum)?[- ](?:sharpe(?: ratio)?|diversification)|market[- ]cap(?:italization)?|equal(?:ly)?)[- ]weight(?:ed|ing|s)?\b.*))$", s,
                   flags=re.I | re.S)
    if mtw:
        wp = _weighting_phrase(mtw.group("w"))
        if wp:
            node = {"weights": wp[0], "children": _children(mtw.group("lst"), notes)}
            if wp[1] and wp[0] != "equal":
                node["lookback"] = wp[1]
            return node
    # a single asset, or a plain list (equal weight)
    items = _asset_list(s)
    if len(items) == 1:
        return items[0]
    return {"weights": "equal", "children": items}


_BW_TK = r"[\^$]?[A-Za-z][A-Za-z0-9.\-]{0,9}"
_BW_SEP = r"(?:\s*,\s*(?:and\s+)?|\s+and\s+|\s*;\s*|\s+|\s*\+\s*)"


def _one_ticker(w: str) -> str | None:
    if w.lower() == "cash":
        return "cash"
    if not (w == w.upper() and data.canonical(w) in _known()):
        hit = class_series(w, getattr(_TL, "start", None), getattr(_TL, "mixed", False))
        if hit:     # the holdings parser settles it (and adds the notes)
            subs = getattr(_TL, "class_subs", None)
            if subs is None:
                subs = _TL.class_subs = {}
            subs.setdefault(hit[0], w)
            return hit[0]
    probe = w.upper() if re.fullmatch(r"[a-z^$]{1,6}", w) else w
    try:
        tk = find_tickers(probe, strict=True)
    except ParseError:
        return None
    return tk[0] if len(tk) == 1 and data.canonical(probe.lstrip("$")) == tk[0] else None


def _bare_weights(s: str) -> str:
    """'60% VTI 40% BND' / 'VTI 60% BND 40%' / 'VTI 60, BND 40' (numbers adding up to 100) -> '60% VTI, 40% BND'.
    Only when every token pairs one known ticker with one weight; anything else is returned unchanged."""
    t = s.strip()
    num = r"-?\d+(?:\.\d+)?"
    for pat, pct_needed in ((rf"(?:{num}%\s*{_BW_TK})(?:{_BW_SEP}{num}%\s*{_BW_TK})+", True),
                            (rf"(?:{_BW_TK}\s*:?\s*{num}%?)(?:{_BW_SEP}{_BW_TK}\s*:?\s*{num}%?)+", False)):
        if not re.fullmatch(pat, t):
            continue
        pairs = (re.findall(rf"({num})%\s*({_BW_TK})", t) if pct_needed else
                 [(w, k) for k, w in re.findall(rf"({_BW_TK})\s*:?\s*({num})%?", t)])
        tks = [_one_ticker(k) for _, k in pairs]
        if len(pairs) < 2 or any(x is None for x in tks):
            return s
        ws = [float(w) for w, _ in pairs]
        if not pct_needed and "%" not in t and abs(sum(ws) - 100) > 1e-6:
            return s     # bare numbers are weights only when they add up to 100
        return ", ".join(f"{w}% {k}" for (w, _), k in zip(pairs, tks))
    return s


def _near_equal(ws: list[float], kids: list[dict], notes: list[str]) -> dict | None:
    """'33% / 33% / 33%' (or 33.3% each) means equal thirds, not 99% invested and 1% in cash: weights that are all
    equal and add up to 99% .. 99.99% become equal weights, with a note."""
    tot = sum(ws)
    if len(ws) > 1 and max(ws) - min(ws) < 1e-9 and ws[0] > 0 and 0.99 - 1e-9 <= tot < 1 - 1e-6:
        notes.append(f"{len(ws)} weights of {ws[0] * 100:g}% add up to {tot * 100:g}%: read as equal weights "
                     f"(1/{len(ws)} = {100 / len(ws):.4g}% each), not {100 - tot * 100:.2g}% held in cash.")
        return {"weights": "equal", "children": kids}
    return None


def _listed_weights(ws: list[float], kids: list[dict], s: str, notes: list[str]) -> dict:
    """Stated weights of a list: equal thirds for 33/33/33, the rest in cash when they add up to less than 100%."""
    ne = _near_equal(ws, kids, notes)
    if ne:
        return ne
    if abs(sum(ws) - 1) > 1e-6:
        if sum(ws) < 1 - 1e-6 and not any(k.get("cash") for k in kids):
            kids = kids + [{"cash": True}]
            ws = ws + [1 - sum(ws)]
            notes.append(f"Weights add up to {sum(ws[:-1]):.0%}: the remaining {ws[-1]:.0%} is held in cash.")
        else:
            return _levered_weights(ws, kids, s, notes)
    return _weights_node(ws, kids, s)


def _weights_node(ws: list[float], kids: list[dict], s: str) -> dict:
    if any(w == 0 for w in ws):
        raise ParseError(f"A 0% weight in {s!r}")
    if any(w < 0 for w in ws) and not all(k.get("cash") or w > 0 for w, k in zip(ws, kids)):
        pass  # negative weights (short positions) are allowed; the engine borrows / shorts accordingly
    return {"weights": "specified", "w": [round(w, 10) for w in ws], "children": kids}


def _condition_on(cond: str, default: str | None) -> tuple[str, str]:
    """Condition text -> (ticker the rule is evaluated on, rule)."""
    tk = find_tickers(cond, strict=True)
    if re.match(rf"(?i)\s*{PRONOUN_SUBJECT}\s", cond):
        # "SPY if its 12 month return is above BIL's": the condition is about the holding, not about BIL
        if default is None:
            raise ParseError(f"{cond.strip()!r}: whose? Name the ticker, e.g. 'if SPY's 12 month return is above BIL's'.")
        on = default
    else:
        on = tk[0] if tk else default
    if on is None:
        raise ParseError(f"Which ticker does {cond.strip()!r} refer to? e.g. 'if SPY is above its 200-day moving average'.")
    rule = parse_conditions(cond, [on], total=True)
    return on, rule


_PF_STOP = re.compile(r"(?i)\b(?:(?:trailing )?stop[- ]?loss(?:es)?|trailing stop|stop(?:ped)? out|take[- ]profit|profit target|"
                      r"(?:go|move|switch|get out) (?:to|into) cash (?:when|if|once) (?:the |my )?(?:portfolio|account|balance|equity)"
                      r"|(?:the |my )?(?:portfolio|account|balance|equity) (?:falls?|drops?|is down|loses?|declines?) \d)")


def _refuse_portfolio_stops(t: str) -> None:
    """Stops and targets belong to signal strategies (they exit a position bought on a signal); an allocation portfolio
    has no entry price. A stop on the portfolio's own value is refused with the equivalents that do work."""
    m = _PF_STOP.search(_mask(t, parens=False))
    if not m:
        return
    own = re.search(r"(?i)portfolio|account|balance|equity", m.group(0))
    raise ParseError(
        f"'{m.group(0).strip()}': " + (
            "a stop on the portfolio's own value is not supported: after the stop there is no rule for when to buy back, "
            "so the portfolio would sit in cash for good. " if own else
            "stops and profit targets are for signal strategies (they exit a position bought on a signal, e.g. 'buy SPY "
            "when its RSI(2) is below 10, sell when it closes above its 5 day moving average, stop loss 10%'); an "
            "allocation portfolio has no entry price to measure them from. ")
        + "For a portfolio, use a drawdown rule on a market ticker, which also says when to get back in: e.g. 'if SPY is "
          "down 10% or more from its 52 week high then hold BIL else hold 60% SPY and 40% TLT, rebalance daily'.")


_MONTH_NUM = {m: i for i, m in enumerate(("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov",
                                          "dec"), 1)}
_MONTH_FULL = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
               "November", "December")
_TICK_ITEM = r"\^?[A-Z][A-Z0-9]{0,5}(?:[.-][A-Z]{1,3})?"
_LB_UNIT = r"(day|week|month|year)s?"


def _portfolio_phrases(t: str, notes: list[str]) -> str:
    """Portfolio phrases rewritten into the forms the rest of parse_allocation reads:
    'start 2008' -> 'since 2008'; 'starting balance $1,000,000' -> 'starting with $1000000'; a year range
    '1972-2020' -> 'from 1972 to 2020'; '25% each of VTI, TLT, GLD and SHY' -> '25% VTI, 25% TLT, 25% GLD and 25% SHY';
    'by 3 month return weighted 50%, 6 month weighted 30% and 12 month weighted 20%' -> the weighted average of those
    total returns (a rule in backticks: each lookback follows the portfolio's month convention)."""
    t = _sub_outside(r"(?i)\b(?:start|begin)(?: date)?(?: in| on| from|:)? (?=(?:" + MONTH_RX + r" )?\d{4}\b)", "since ", t)
    t = _sub_outside(r"(?i)\b(?:(?:starting|initial|opening|start|beginning) (?:balance|investment|amount|capital|value|portfolio "
                     r"(?:balance|value))|initial deposit)(?: of| is|:|=)? (?=\$)", "starting with ", t)
    t = _sub_outside(r"(?i)(?:\b(?:from|between|during|over|in|for) )?(?<![\w$./-])((?:18|19|20)\d\d) ?(?:-|–|to|through|thru) ?"
                     r"((?:18|19|20)\d\d)(?![\w%/.-])(?! dollars)", r"from \1 to \2", t)

    def each(m):
        items = [x for x in re.split(r",\s*(?:and\s+|&\s*)?|\s+and\s+|\s*&\s*", m.group("items")) if x]
        w = float(m.group("w"))
        if abs(w * len(items) - 100) > 0.01 and m.group("rest") is None:
            raise ParseError(f"'{m.group(0).strip()}': {w:g}% each of {len(items)} holdings is {w * len(items):g}%, not 100%.")
        return " and ".join(f"{m.group('w')}% {x}" for x in items) if len(items) > 1 else f"{m.group('w')}% {items[0]}"
    # (not "20% each of SPY, EFA, ..., each only when ...": per-asset timing, read by _node)
    t = t if re.search(r"(?i),? each (?:only|when|if|while|as long as)\b", t) else _sub_outside(r"(?P<w>\d+(?:\.\d+)?)% each (?:of |in |to |into )?(?:the )?(?P<items>" + _TICK_ITEM
                     + r"(?:(?:,\s*(?:and\s+|&\s*)?|\s+and\s+|\s*&\s*)" + _TICK_ITEM + r")+)(?![\w-])(?!,? each\b)(?P<rest>(?= plus| with \d))?", each, t)
    t = _sub_outside(r"(?P<items>" + _TICK_ITEM + r"(?:(?:,\s*(?:and\s+|&\s*)?|\s+and\s+|\s*&\s*)" + _TICK_ITEM
                     + r")+),? (?:at |with )?(?P<w>\d+(?:\.\d+)?)% each(?P<rest>(?!))?", each, t)

    def weighted(m):
        parts = re.findall(r"(\d+) " + _LB_UNIT + r"(?: (?:total )?(?:returns?|momentum|performance))? (?:weighted|at|with a weight of) "
                           r"(\d+(?:\.\d+)?)%", m.group("items"))
        if len(parts) < 2:
            return m.group(0)
        ws = [float(x) for _, _, x in parts]
        if abs(sum(ws) - 100) > 0.01:
            raise ParseError(f"'{m.group(0).strip()}': the lookback weights add up to {sum(ws):g}%, not 100%.")
        ns = [_period(n, u) for n, u, _ in parts]
        if len(set(ns)) < len(ns):
            raise ParseError(f"'{m.group(0).strip()}': give different lookbacks.")
        lag = 0
        if m.group("skip"):
            # "..., skipping the last month": every lookback ends that long ago (as '12 month return skipping the last month')
            k = m.group("sk_n")
            lag = _period(k if k and k.isdigit() else "1", m.group("sk_u"))
            if not 0 < lag < min(ns):
                raise ParseError(f"'{m.group(0).strip()}': the skipped period must be shorter than every lookback.")
        rule = " + ".join(f"{w / 100:g} * tret(tr, {n - lag})" for w, n in zip(ws, ns))
        if lag:
            rule = f"ref({rule}, {lag})"
        notes.append(f"'{m.group(0).strip(' ,')}' = the weighted average of the total returns: "
                     + ", ".join(f"{w:g}% x {n_} {u}" for w, (n_, u, _) in zip(ws, parts))
                     + (f", each ending {lag} trading days ago (the most recent {lag} days left out)" if lag else "")
                     + f" (`{rule}`; month lookbacks follow the portfolio's month convention: 21 trading days a month, "
                     "or month-end to month-end with 'using calendar months').")
        return f"`{rule}`"
    item = r"\d+ " + _LB_UNIT + r"(?: (?:total )?(?:returns?|momentum|performance))? (?:weighted|at|with a weight of) \d+(?:\.\d+)?%"
    skip = (r"(?P<skip>,? (?:skipping|skip|excluding|exclude|except(?: for)?|ignoring|leaving out|lagged by) (?:the )?"
            r"(?:last |latest |most recent |recent |past |final )?(?:(?P<sk_n>\d+|one|a) )?(?P<sk_u>day|week|month)s?)?")
    t = _sub_outside(r"(?i)(?P<items>" + item + r"(?:(?:,\s*(?:and\s+)?|\s+and\s+|\s+plus\s+)" + item + r")+)" + skip, weighted, t)
    return t


def _static_weights(n: dict, w: float = 1.0, out: dict | None = None) -> dict | None:
    """Target weight of each holding of a tree of fixed weights (equal / specified groups of assets); None for a tree
    whose weights change (if/else, filters, weightings on data) or that shorts."""
    out = {} if out is None else out
    if not isinstance(n, dict):
        return None
    if "asset" in n:
        out[n["asset"]] = out.get(n["asset"], 0.0) + w
        return out
    if n.get("cash"):
        out["cash"] = out.get("cash", 0.0) + w
        return out
    kids = n.get("children") or []
    if n.get("weights") == "equal" and kids:
        ws = [1 / len(kids)] * len(kids)
    elif n.get("weights") == "specified" and kids and len(n.get("w") or []) == len(kids):
        ws = [float(x) for x in n["w"]]
        if any(x < 0 for x in ws):
            return None
    else:
        return None
    for k, x in zip(kids, ws):
        if _static_weights(k, w * x, out) is None:
            return None
    return out


def _band_unreachable(tree: dict, band: float | None, band_rel: float | None, rb: str = "none") -> str | None:
    """A warning when a drift band can never trigger: a holding's weight stays between 0 and 100%, so a target w can
    drift by at most max(w, 1 - w) (absolute), or by max(100%, (1 - w) / w) of itself (relative). What still trades
    is said as it is: a tree with rules (if/else, filters) still trades whenever they switch, and a calendar schedule
    still rebalances on its dates."""
    ws = _static_weights(tree)
    ws = [x for x in (ws or {}).values() if x > 0] if ws else None
    sched = rb not in (None, "none", "daily")
    def has(n, key) -> bool:
        return isinstance(n, dict) and (key in n or any(has(c, key) for c in n.get("children") or [])
                                        or any(has(n.get(k), key) for k in ("then", "else", "fallback")))
    if has(tree, "if") or has(tree, "filter"):
        kinds = " or ".join(x for x, k in (("the if/else switches", "if"), ("a filter's picks change", "filter")) if has(tree, k))
        still = (f"so trades happen only when {kinds}"
                 + (f", and on the {rb.replace('_', ' ')} rebalance" if sched else "") + ".")
    elif sched:
        still = f"so the portfolio is rebalanced only on its {rb.replace('_', ' ')} schedule."
    else:
        still = "so the portfolio is never rebalanced after the first day."
    if band:
        reach = max(max(x, 1 - x) for x in ws) if ws else 1.0
        if band >= reach - 1e-12:
            return (f"Warning: a {band * 100:g}% drift band can never trigger"
                    + f": no weight can move more than {reach * 100:g} percentage points from its target"
                    + (" here" if ws else " (weights stay between 0 and 100%)")
                    + f", {still} Use a smaller band (e.g. 5%).")
    if band_rel:
        reach = max(max(1.0, (1 - x) / x) for x in ws) if ws else None
        if reach is not None and band_rel >= reach - 1e-12:
            return (f"Warning: a {band_rel * 100:g}% relative drift band can never trigger: no weight can move more than "
                    f"{reach * 100:g}% of its target here, {still} Use a smaller band (e.g. 25% relative).")
    return None


_RB_VERB = r"(?:re-?balanc\w*|trade|trading|rotat\w*|re-?evaluat\w*|check\w*)(?: (?:it|the weights|the portfolio|them))?"
_RB_WD = r"(monday|tuesday|wednesday|thursday|friday)s?"


def _rebalance_day_phrases(T: "Text", notes: list[str]) -> tuple[str, str | None] | None:
    """Which day of its period a calendar schedule trades on -> (rebalance, rebalance_day), or None:
      'rebalance on the first trading day of each month' / 'rebalance monthly at the start (of the month)' /
      'rebalance at the beginning of every quarter'                      -> ('monthly' / 'quarterly', 'start')
      'rebalance on the last trading day of each month' / '... at the end' -> ('monthly', None) (the default)
      'rebalance every Monday' / 'rebalance weekly on Mondays' / 'every Friday' -> ('weekly', 'monday' / 'friday')"""
    per = r"(?P<per>week|month|quarter|year)"
    m = T.find(rf",? ?(?:and )?{_RB_VERB} (?:(?P<f1>weekly|monthly|quarterly|annually|yearly),? )?(?:on|at) (?:the )?"
               rf"(?P<fl>first|last|final|start|beginning|end)(?: (?:trading |market )?day)? (?:of|in) (?:the |each |every |a )?"
               rf"(?:new )?{per}(?![\w-])")
    if m:
        freq = FREQ_WORDS[m.group("per").lower()]
        f1 = m.group("f1")
        if f1 and FREQ_WORDS[f1.lower()] != freq:
            raise ParseError(f"'{m.group(0).strip(' ,')}': a {f1.lower()} schedule cannot trade at the "
                             f"{m.group('fl').lower()} of each {m.group('per').lower()}.")
        start = m.group("fl").lower() in ("first", "start", "beginning")
        return freq, ("start" if start else None)
    m = T.find(rf",? ?(?:and )?{_RB_VERB} (?P<f>weekly|monthly|quarterly|annually|yearly),? (?:at|on) (?:the )?"
               rf"(?P<fl>start|beginning|end|first (?:trading )?day|last (?:trading )?day)(?: of (?:the |each |every )?{per})?(?![\w-])")
    if m:
        freq = FREQ_WORDS[m.group("f").lower()]
        if m.group("per") and FREQ_WORDS[m.group("per").lower()] != freq:
            raise ParseError(f"'{m.group(0).strip(' ,')}': a {m.group('f').lower()} schedule cannot trade at the "
                             f"{m.group('fl').lower()} of each {m.group('per').lower()}.")
        return freq, ("start" if re.match(r"(?i)start|beginning|first", m.group("fl")) else None)
    m = T.find(rf",? ?(?:and )?(?:{_RB_VERB} )?(?:weekly,? )?(?:(?:every|each|on) {_RB_WD}|on {_RB_WD})(?: (?:at|near) the close)?(?![\w-])")
    if m:
        day = (m.group(1) or m.group(2)).lower()
        return "weekly", day
    return None


def parse_allocation(text: str) -> Portfolio:
    raw = text
    t = _normalize(text)
    t = _sub_outside(r"\[", "(", _sub_outside(r"\]", ")", t))
    # "off by 5 percentage points": an absolute drift of 5% of the portfolio
    t = _sub_outside(r"(?i)(\d+(?:\.\d+)?) ?(?:percentage points?|pct points?|ppts?|pp)\b", r"\1%", t)
    notes: list[str] = []
    t = _portfolio_phrases(t, notes)
    _refuse_portfolio_stops(t)
    _negative_lookbacks(t)
    T = Text(t)
    flows = _cash_flows(T, notes)
    # "target 10% volatility (using 60 day volatility)": scale the whole portfolio towards that volatility
    tv: dict = {}
    m = T.find(rf",? ?(?:and |with |using )?(?:a )?(?:target(?:ing|ed)?(?: an?)?(?: annual(?:ized)?)? {NUM}% (?:annual(?:ized)? )?(?:volatility|vol)"
               rf"|{NUM}% (?:annual(?:ized)? )?(?:volatility|vol) target|volatility target(?:ing)?(?: of)? {NUM}%"
               rf"|{NUM}% (?:annual(?:ized)? )?target(?:ed)? (?:annual(?:ized)? )?(?:volatility|vol))"
               rf"(?:,? (?:using|over|with|measured over|based on) (?:an? |the )?(?:last |past |trailing )?(\d+) (day|week|month)s?(?: (?:realized |realised |trailing |historical )?(?:volatility|vol|lookback|window))?)?")
    if m:
        tv["target_vol"] = float(m.group(1) or m.group(2) or m.group(3) or m.group(4)) / 100
        if m.group(5):
            tv["target_vol_lookback"] = _period(m.group(5), m.group(6))
        if not {"target_vol"} <= {f.name for f in dataclasses.fields(Portfolio)}:
            raise ParseError("Volatility targeting for allocation portfolios needs the portfolio engine's target_vol setting, "
                             "which this version does not have.")
        notes.append(f"Volatility target {tv['target_vol']:.0%} a year: the portfolio's exposure is scaled towards it, using its "
                     + (f"{tv['target_vol_lookback']} day" if "target_vol_lookback" in tv else "default") + " realised volatility.")
    # "using calendar months" / "using month-end prices": N-month returns month-end to month-end (PV, Antonacci)
    if T.find(r",? ?(?:and )?(?:(?:using|with|on|measured (?:on|over|with)) )?(?:(?:the )?calendar months?(?: returns?| lookbacks?)?"
              r"|(?:the )?month[- ]end (?:prices|closes|values)|month[- ]end to month[- ]end(?: returns?)?)"):
        tv["month_lookbacks"] = "calendar"
    # "with Portfolio Visualizer defaults" / "PV defaults": yearly rebalancing for fixed weights and calendar-month
    # lookbacks unless stated otherwise (the other defaults - $10,000, dividends reinvested - are the same already)
    pv_defaults = bool(T.find(r",? ?(?:and )?(?:with|using|on|in) (?:the )?(?:portfolio ?visualizer|pv)(?:'s)? (?:defaults?|default settings|settings)"
                              r"|,? ?(?:as|like) (?:in )?(?:portfolio ?visualizer|pv)\b"))
    if pv_defaults and "month_lookbacks" not in tv:
        tv["month_lookbacks"] = "calendar"
    # "using trading days" / "with 21-day months": N-month returns as N x 21 sessions (the default outside the named
    # tactical models, which use month-ends as Portfolio Visualizer does)
    if "month_lookbacks" not in tv and T.find(
            r",? ?(?:and )?(?:(?:using|with|on|measured (?:on|over|with|in)) (?:(?:the )?trading[- ]days?(?: months?| lookbacks?)?"
            r"|21[- ](?:trading[- ])?days? (?:a |per )?months?)|(?:(?:using|with|on) )?trading[- ]day (?:months|lookbacks))(?![\w-])"):
        tv["month_lookbacks"] = "trading"
    # opt-in: a stated proxy for a holding before its history begins ("with proxies before inception")
    want_proxies = bool(T.find(r",? ?(?:and )?(?:with|using|use) (?:stated )?proxies(?: for (?:the )?(?:missing |late )?(?:sleeves?|holdings?|funds?))?"
                               r" (?:before|until|prior to) (?:(?:the|their|its|each (?:fund|sleeve|holding)'s) )?(?:inception|launch)s?"))
    broker = _broker_costs(T, notes, sep=",? ?")
    kw = common_options(T, notes)
    kw.update(broker)
    benchmark = kw.pop("benchmark", None)
    bench_rb = kw.pop("benchmark_rebalance", None)
    if "commission_per_share" in kw:
        raise ParseError("Per-share commissions are not supported for allocation portfolios; use '$1 per trade' or '0.1% commission'.")
    pk: dict = {k: v for k, v in kw.items() if k in ("capital", "slippage_bps", "commission", "commission_pct", "start", "end", "cash_rate", "point_in_time",
                                                   "commission_model", "slippage_model")}

    # rebalancing
    rb = None
    rb_day_hit = _rebalance_day_phrases(T, notes)
    rb_day = None
    if rb_day_hit:
        rb, rb_day = rb_day_hit
    # once a year in a chosen month: "rebalance every year in June", "rebalance annually in June", "rebalance each June",
    # "annual rebalancing in June": at that month's last trading day
    m = T.find(r",? ?(?:and )?(?:(?:re-?balanc\w*|reset|re-?evaluat\w*)(?: (?:it|the weights|the portfolio|them))?(?: back)?"
               r"(?: to (?:target|the target weights))?(?: (?:(?:once )?(?:every|each|a|per) year|annually|yearly|once a year))?"
               r"|(?:annual|yearly) re-?balanc\w*),? (?:in|every|each|at the end of(?: every| each)?|at (?:the )?(?:end|close) of) "
               r"(?P<mon>" + MONTH_RX + r")(?: (?:each|every) year)?(?![\w-])")
    if m:
        mon = _MONTH_NUM.get(m.group("mon").lower().rstrip(".")[:3])
        rb = f"yearly_{mon}"
        notes.append(f"Rebalanced once a year at the end of {_MONTH_FULL[mon - 1]} (its last trading day), and on the first day.")
    m = None if rb else T.find(r",? ?(?:and )?(?:re-?balanc\w*|reset|rotat\w*|re-?evaluat\w*|check\w*)(?: (?:it|the weights|the portfolio|them))?(?: back)?(?: to (?:target|the target weights))? "
               r"(?:semi-?annually|semi-?annual|twice (?:a|per|each) year|twice yearly|half-?yearly|every 6 months|once every 6 months)"
               r"|,? ?(?:semi-?annual|half-?yearly) re-?balanc\w*")
    if m:
        rb = "semiannual"
    m = T.find(r",? ?(?:and )?(?:re-?balanc\w*|reset|rotat\w*|re-?evaluat\w*|check\w*)(?: (?:it|the weights|the portfolio|them))?(?: back)?(?: to (?:target|the target weights))? "
               r"(?:once )?every (\d+) (?:trading )?(month|week|year|day|session|bar)s?")
    if m and rb is None:
        n_, u_ = int(m.group(1)), m.group(2).lower()
        u_ = "day" if u_ in ("session", "bar") else u_
        rb = {("month", 1): "monthly", ("month", 3): "quarterly", ("month", 6): "semiannual", ("month", 12): "yearly",
              ("week", 1): "weekly", ("year", 1): "yearly", ("day", 1): "daily"}.get((u_, n_))
        if rb is None and n_ >= 1 and u_ in ("day", "week", "month"):
            # every N trading days (counted from the first day), or every N-th week / month end
            rb = f"every_{n_}_{u_}s"
            nth = f"{n_}{'nd' if n_ % 10 == 2 and n_ % 100 != 12 else 'rd' if n_ % 10 == 3 and n_ % 100 != 13 else 'th'}"
            notes.append(f"Rebalancing every {n_} {u_}s: " + (f"on the first day, then every {nth} trading day after it."
                                                               if u_ == "day" else
                                                               f"at every {nth} {u_}-end (the last trading day of the {u_})."))
        if rb is None:
            raise ParseError(f"'{m.group(0).strip(' ,')}': rebalancing can be daily, weekly, monthly, quarterly, every 6 months, "
                             "yearly, or every N days / weeks / months.")
    m = None if rb else T.find(r",? ?(?:and )?(?:re-?balanc\w*|reset|rotat\w*|re-?evaluat\w*|check\w*)(?: (?:it|the weights|the portfolio|them))?(?: back)?(?: to (?:target|the target weights))? (?:every|each|once (?:a|per)) (day|week|month|quarter|year)|,? ?(?:and )?re-?balanc\w*(?: (?:it|the weights|the portfolio))? (daily|weekly|monthly|quarterly|annually|yearly)|,? ?(daily|weekly|monthly|quarterly|annual|yearly) re-?balanc\w*")
    if m:
        f_ = FREQ_WORDS[(m.group(1) or m.group(2) or m.group(3)).lower()]
        if rb_day_hit and f_ != rb:
            raise ParseError(f"'{m.group(0).strip(' ,')}' and the rebalance day given elsewhere name different schedules "
                             f"({f_} and {rb}).")
        rb = f_
    # "fortnightly" / "biweekly" / "every other week": every 2nd week-end
    m = T.find(r",? ?(?:and )?(?:(?:re-?balanc\w*|reset|rotat\w*|re-?evaluat\w*|check\w*)(?: (?:it|the weights|the portfolio|them))?"
               r"(?: back)?(?: to (?:target|the target weights))? (?:fortnightly|bi-?weekly|every (?:other|second|2nd) week|"
               r"every fortnight|once a fortnight)|,? ?(?:fortnightly|bi-?weekly) re-?balanc\w*)")
    if m:
        rb = "every_2_weeks"
        notes.append("Rebalancing fortnightly: at every 2nd week-end (the last trading day of every other week).")
    if T.find(r",? ?(?:and )?(?:never re-?balanc\w*|no re-?balancing|without re-?balancing|don't re-?balance|do not re-?balance"
              r"|re-?balanc\w*(?: (?:it|the weights|the portfolio|them))? (?:never|none))"):
        rb = "none"
    # threshold (corridor) rebalancing written the way Composer and other tools name it: "threshold rebalance 5%",
    # "5% threshold rebalancing", "a 5% rebalance corridor", "rebalance band 5%", "threshold 5%"
    band_pre = None
    m = T.find(r",? ?(?:and |or )?(?:with |using )?(?:a |an )?(?:"
               r"(?:threshold|corridor|band) re-?balanc\w*(?: (?:at|of))? (?P<a>\d+(?:\.\d+)?)%"
               r"|(?P<b>\d+(?:\.\d+)?)%(?P<relb> relative)? (?:re-?balanc\w* )?(?:threshold|corridor)(?: re-?balanc\w*)?"
               r"|re-?balanc\w* (?:threshold|corridor|band|bands)(?: of| at)? (?P<c>\d+(?:\.\d+)?)%"
               r"|threshold(?: of)? (?P<d>\d+(?:\.\d+)?)%"
               r"|re-?balanc\w* (?:at|with|using|on) (?:a |an )?(?P<e>\d+(?:\.\d+)?)%(?P<rele> relative)? (?:corridor|threshold)(?: re-?balanc\w*)?)(?P<rel> relative)?(?![\w%])")
    if m:
        band_pre = (float(m.group("a") or m.group("b") or m.group("c") or m.group("d") or m.group("e")) / 100,
                    bool(m.group("rel") or m.group("relb") or m.group("rele")))
    # a schedule word the phrases above do not know ("rebalance hourly", "rebalance on Tuesdays"): say so, instead of
    # letting the words fall through to the holdings as unknown tickers
    m = T.find(r",? ?(?:and )?re-?balanc\w*(?: (?:it|the weights|the portfolio|them))? (?!(?:only |also |and |or )?(?:when|if|whenever|at|on the|with|using|by|to|back)\b)"
               r"((?:every |each |once |on |at )?[a-z]+(?:[- ][a-z]+)?)(?=\s*(?:,|;|$))", consume=False)
    if m and not re.search(r"\d", m.group(1)):
        raise ParseError(f"'{m.group(0).strip(' ,')}': unknown rebalancing schedule '{m.group(1)}'. Rebalancing can be daily, "
                         "weekly, fortnightly, monthly, quarterly, every 6 months, yearly, every N days / weeks / months, "
                         "never, or when a weight drifts N% from its target.")
    band, band_rel = None, None
    m = T.find(r",? ?(?:and |or )?(?:re-?balanc\w* )?(?:only )?(?:(?:and|or) )?(?:also )?(?:when(?:ever)?|if) (?:any |a |the )?(?:weight|holding|position|allocation|asset)s? "
               r"(?:drifts?|moves?|deviates?|is off|gets? off|strays?) (?:by )?(?:more than |over )?(?P<n1>\d+(?:\.\d+)?)%"
               r"(?P<rel1> relative(?: to (?:its |their |the )?targets?(?: weights?)?)?| of (?:its |their |the )?targets?(?: weights?)?)?"
               r"(?: (?:from|away from|off) (?:its |their |the )?targets?(?: weights?)?)?"
               r"|,? ?(?:(?:and|or) )?(?:re-?balanc\w* )?(?:with |using )?(?:a )?(?P<n2>\d+(?:\.\d+)?)%(?P<rel2> relative)? (?:re-?balancing |drift |tolerance )?bands?"
               # "rebalance when drift exceeds 5%", "rebalance at 5% drift", "rebalance if the drift is more than 5%"
               r"|,? ?(?:(?:and|or) )?(?:re-?balanc\w* )?(?:only )?(?:(?:when(?:ever)?|if) (?:the )?(?:portfolio |weight |allocation )?drift "
               r"(?:exceeds|is (?:more than|over|above|greater than|bigger than)|goes (?:over|above)|gets (?:over|above)|>=?|reaches|hits) "
               r"|at (?:a )?(?=\d+(?:\.\d+)?% (?:relative )?(?:portfolio |weight )?drift))(?P<n3>\d+(?:\.\d+)?)%"
               r"(?P<rel3> relative(?: to (?:its |their |the )?targets?(?: weights?)?)?| of (?:its |their |the )?targets?(?: weights?)?)?"
               r"(?: (?:portfolio |weight )?drift)?")
    if m or band_pre:
        amount = float(m.group("n1") or m.group("n2") or m.group("n3")) / 100 if m else band_pre[0]
        if (m and (m.group("rel1") or m.group("rel2") or m.group("rel3"))) or (not m and band_pre[1]):
            band_rel = amount
            notes.append(f"Relative drift band: rebalance when a holding's weight is off its target by more than {amount:.0%} "
                         f"of that target (e.g. a 40% target outside {0.4 * (1 - amount):.0%}-{0.4 * (1 + amount):.0%}).")
        else:
            band = amount
    fill = "close"
    if T.find(r",? ?(?:trade|trading|rebalanc\w*|execute\w*)? ?(?:at|on) the next (?:day's )?open"):
        fill = "next_open"
    T.find(r",? ?(?:trade|trading|rebalanc\w*|execute\w*)? ?(?:at|on) the close")
    # cash flows (parsed before the general options, so "starting in 2000" dates the withdrawals)
    contrib, cfreq = flows["contribution"], flows["contribution_freq"]
    wd, wd_pct, wfreq = flows["withdrawal"], flows["withdrawal_pct"], flows["withdrawal_freq"]
    # "adjusted for inflation" written away from any flow ("..., hold 60/40, adjusted for inflation"): every $ flow
    loose = T.find(r",? ?(?:and )?(?:with )?(?:all |the )?(?:(?:cash )?flows? |amounts? )?(?:" + CF_INFL + r")")
    if loose:
        kinds = [k for k in ("contribution", "withdrawal") if flows[k] or (k == "withdrawal" and flows["withdrawal_pct"])]
        md = re.search(r"in (today's|todays|current|\d{4}) dollars", loose.group(0))
        for k in kinds:
            flows[f"{k}_inflation"] = True
            if md:
                flows[f"{k}_dollars"] = md.group(1) if md.group(1).isdigit() else "flow"
        if len(kinds) > 1:
            notes.append(f"'{loose.group(0).strip(' ,')}' is not attached to one cash flow, so it applies to both the "
                         "contributions and the withdrawals; write it right after the one you mean (e.g. 'withdraw $50,000 a "
                         "year adjusted for inflation') to index only that one.")
    infl = bool(flows.get("contribution_inflation") or flows.get("withdrawal_inflation"))
    wd_infl = bool(flows.get("withdrawal_inflation"))
    if T.find(r",? ?(?:do not|don't|without) reinvest(?:ing)? dividends|dividends (?:paid out|kept) (?:as|in) cash"):
        kw["reinvest_dividends"] = False
    T.find(r",? ?(?:with )?dividends reinvested|reinvest(?:ing)? dividends")
    m = T.find(r",? ?(?:with )?(?:no|without|ignor(?:e|ing)|exclud(?:e|ing)) (?:cash )?dividends|,? ?price[- ]only(?: returns)?", consume=False)
    if m:
        raise ParseError(f"'{m.group(0).strip(' ,')}': a portfolio always receives its holdings' dividends (they are part of "
                         "its total return). Say \"don't reinvest dividends\" to keep them as cash; price-only returns are "
                         "available for signal strategies ('buy ... when ..., no dividends').")

    extra: dict = {}
    lev_rx = (rf",? ?(?:(?:with|using|at|and) )?{NUM}(?:x| ?times) (?:leverage|leveraged)|,? ?(?:with |using )?(?:a )?leverage "
              rf"(?:of )?{NUM}(?:x| ?times)?|,? ?(?:levered|leveraged) {NUM}(?:x| ?times)")
    m = T.find(lev_rx, consume=False)
    if m and _branch_leverage(T.rest, m):
        m = None      # "if ... hold QQQ with 2x leverage, otherwise TLT": that branch's weight (200% QQQ), not the portfolio's
    elif m:
        m = T.find(lev_rx)
    if m:
        extra["leverage"] = float(m.group(1) or m.group(2) or m.group(3))
        notes.append(f"Leverage {extra['leverage']:g}x: every weight is scaled up and the difference is borrowed at the T-bill rate"
                     " (plus any margin rate).")
    m = T.find(rf",? ?(?:(?:with|and) )?(?:a )?{NUM}% maintenance(?: margin)?(?: requirement)?|,? ?(?:(?:with|and) )?maintenance margin(?: requirement)?(?: of)? {NUM}%")
    if m:
        extra["maintenance_margin"] = float(m.group(1) or m.group(2)) / 100
    if T.find(r",? ?(?:(?:with|and) )?(?:no|without|ignore|ignoring) margin calls?"):
        extra["maintenance_margin"] = 0.0
    if T.find(r",? ?(?:(?:with|using|on|in|and) )?(?:an? )?portfolio[- ]margin(?:ing)?(?: account)?"):
        extra["margin_account"] = "portfolio"
    lev = extra.get("leverage", 1.0)
    if lev > 4:
        raise ParseError(f"{lev:g}x leverage is more than a broker lends: Regulation T allows 2x overnight and a "
                         "portfolio-margin account 4x (say 'with portfolio margin' and e.g. 'a 15% maintenance margin').")
    # short-selling costs: "1% borrow fee", "borrow fee of 1%", "2% borrowing cost"; the short rebate
    m = T.find(rf",? ?(?:(?:with|and|paying) )?(?:{BORROW_FEE})")
    if m:
        extra["borrow_fee"] = float(m.group(1) or m.group(2)) / 100
    elif T.find(rf",? ?{NO_BORROW_FEE}"):
        extra["borrow_fee"] = 0.0
    m = T.find(rf",? ?(?:(?:with|and) )?(?:a )?short rebate(?: spread)?(?: of)? {NUM}% (?:below|under|less than) (?:the )?(?:t-?bill|cash)(?: rate)?"
               rf"|,? ?(?:(?:with|and) )?(?:no|full) short rebate(?: haircut)?")
    if m:
        extra["short_rebate_spread"] = (float(m.group(1)) / 100 if m.group(1)
                                        else (0.99 if "no" in m.group(0).lower().split() else 0.0))   # 0.99: above any rate
    m = T.find(rf",? ?(?:(?:with|and) )?(?:an? )?(?:expense ratio|annual fee|management fee|fee)(?: of| is| at|:|=)? {NUM}%(?: (?:a|per) year)?|,? ?(?:with |and )?(?:an? )?{NUM}% (?:expense ratio|annual fee|management fee|fee)(?: (?:a|per) year)?")
    if m:
        extra["expense_ratio"] = float(m.group(1) or m.group(2)) / 100
    FF = r"(?:the )?(?:effective )?(?:fed(?:eral)?[- ]funds?|ffr|fed)(?: (?:effective )?rate)?"
    TB = r"(?:the )?(?:3[- ]month )?(?:t-?bills?|treasury bills?)(?: rate)?"
    m = T.find(rf",? ?(?:(?:with|and|paying) )?(?:a )?margin (?:rate|interest)(?: of| at| is|:|=)? (?P<base>{FF}|{TB}) ?(?:plus|\+|over|above) ?{NUM}%"
               rf"|,? ?(?:(?:with|and|paying) )?(?:a )?margin (?:rate|interest|spread)(?: of| at| is|:|=)? {NUM}% (?:above|over|plus) (?P<base2>{FF}|{TB})")
    if m:
        extra["margin_rate"] = float(m.group(2) or m.group(4)) / 100
        if re.search(r"(?i)fed|ffr", m.group("base") or m.group("base2") or ""):
            notes.append(f"Margin rate: fed funds + {extra['margin_rate']:.2%}. The data has no fed funds series, so the "
                         "3-month T-bill rate stands in for it (the effective fed funds rate has run about 0.1% above "
                         "T-bills since 2009 and about 0.3-1% above them in earlier decades): borrowed cash pays T-bills + "
                         f"{extra['margin_rate']:.2%}; add the gap to the spread for a closer match.")
    m = None if "margin_rate" in extra else T.find(
        rf",? ?(?:(?:with|and|paying) )?(?:a )?margin (?:rate|interest|spread)(?: of|:)? {NUM}%(?: above (?:t-?bills|the t-?bill rate))?"
        rf"|,? ?(?:(?:with|and|paying) )?(?:an? )?{NUM}% margin (?:rate|interest|spread)")
    if m:
        extra["margin_rate"] = float(m.group(1) or m.group(2)) / 100
    mrot = re.search(r"\b(?:rotate|switch)\w* (daily|weekly|monthly|quarterly|annually|yearly)\b", T.rest, re.I)
    if mrot and rb is None:
        rb = FREQ_WORDS[mrot.group(1).lower()]
        T.rest = T.rest[: mrot.start(1)] + T.rest[mrot.end(1):]
    body = _sub_outside(r"(?i)\bplease\b,?", " ", T.rest)   # politeness: in STOP for signals, removed here for the tree
    body = re.sub(r"(?i)^\s*(?:backtest|test|simulate|run)?\s*(?:a |the )?(?:portfolio|strategy)?(?: (?:of|that|which))?\s*:?\s*", "", body)
    body = re.sub(r"\s*;\s*", " ", body).strip(" ,.;")
    body = re.sub(r" {2,}", " ", body)
    body = re.sub(r"\s+,", ",", body)
    buy_hold = bool(re.match(r"^buy[- ]and[- ]hold\b", body))
    body = re.sub(r"^buy[- ]and[- ]hold\s+", "", body)
    if not body:
        raise ParseError("What should the portfolio hold? e.g. 'hold 60% SPY and 40% TLT, rebalance quarterly'.")
    _TL.start = pk.get("start")
    _TL.tactical = None
    _TL.months = tv.get("month_lookbacks")
    _TL.model = None
    try:
        tree = _node(body, notes)
        tactical = _TL.tactical
        model = _TL.model[0] if _TL.model and _TL.model[1] is tree else None
    finally:
        _TL.start = None
        _TL.tactical = None
        _TL.months = None
        _TL.model = None
    if tactical in TACTICAL_MONTH_END and "month_lookbacks" not in tv:
        # Portfolio Visualizer, Antonacci and Faber measure the models on calendar months (month-end prices)
        tv["month_lookbacks"] = "calendar"
        if tactical.startswith("GTAA"):
            notes.append(f"{tactical}: the 10-month moving averages are of month-end prices and the model is checked at "
                         "each month-end close, as Portfolio Visualizer and Faber do (N-month lookbacks are calendar "
                         "months). For a daily version write e.g. 'SPY, EFA, IEF, VNQ and DBC equally, each only when "
                         "above its 210 day moving average, otherwise cash'.")
        else:
            notes.append(f"{tactical}: N-month returns are measured on calendar months, month-end to month-end (from the "
                         "last completed month-end), and the model is checked at each month-end close, as Portfolio "
                         "Visualizer and Antonacci do. Say 'using trading days' for 21-trading-day months instead.")
    if tactical is None and _has(tree, "if") and _if_rules(tree) and all(
            re.search(r"\bmonthly_\w+\(", r) and not re.search(r"\b(?!monthly_)(?:sma|ema|rsi|ret|tret|volatility|drawdown|max_drawdown|"
                                                                r"stdev_return|ma_return|change|ref)\(", r) for r in _if_rules(tree)):
        tactical = "Monthly moving-average timing (Faber)"
    rb_stated = rb
    if rb is None and tv.get("target_vol") and not (band or band_rel):
        # the exposure is rescaled at each rebalance: without one the target would be applied once, on the first day
        rb = "monthly"
        notes.append("Rebalance frequency not stated: with a volatility target the exposure is rescaled at each month-end "
                     "close; say 'rebalance weekly' or 'rebalance daily' to rescale more often.")
    if rb is None:
        if buy_hold:
            rb = "none"
        elif tactical:
            rb = "monthly"
            notes.append(f"Rebalance frequency not stated: {tactical} is checked and traded at each month-end close, its "
                         "published schedule. Say 'rebalance daily' to check every day.")
        elif pv_defaults and (_has(tree, "if") or _has(tree, "filter") or _dynamic(tree)) and not (band or band_rel):
            rb = "monthly"
            notes.append("Rebalance frequency not stated: with Portfolio Visualizer's defaults the rules, rankings and "
                         "weights are re-evaluated and traded at each month-end close, as its tactical models are; say "
                         "'rebalance daily' to check every day.")
        elif _has(tree, "if") or _has(tree, "filter") or _dynamic(tree):
            # Composer semantics: rules, rankings and dynamic weights are re-evaluated every day
            rb = "daily"
            what = [w for w, ok in (("the conditions are checked", _has(tree, "if")),
                                    ("the filters are re-ranked", _has(tree, "filter")),
                                    ("the dynamic weights are recomputed", _dynamic(tree))) if ok]
            if band or band_rel:
                # "rebalance when any weight drifts 5%" IS the schedule: Composer-style threshold rebalancing
                notes.append("Threshold rebalancing: " + ", ".join(what[:-1]) + (" and " if len(what) > 1 else "")
                             + what[-1] + " every day at the close; the portfolio trades when they change the target or "
                             "a holding drifts past the band (as Composer's threshold rebalancing).")
            else:
                notes.append("Rebalance frequency not stated: " + ", ".join(what[:-1]) + (" and " if len(what) > 1 else "") + what[-1]
                             + " every day at the close, and the portfolio trades as soon as they change (Composer-style; say "
                             "'rebalance monthly' to check less often).")
        elif "weights" in tree:
            if band or band_rel:
                rb = "none"  # threshold-only rebalancing
            elif model or pv_defaults:
                rb = "yearly"
                notes.append(f"Rebalance frequency not stated: {model or 'the portfolio'} is rebalanced yearly (at the "
                             "last close of each year), Portfolio Visualizer's default; say 'rebalance monthly' or "
                             "'rebalance quarterly' for more often, or 'never rebalance'.")
            else:
                rb = "monthly"
                notes.append("Rebalance frequency not stated: fixed weights are rebalanced monthly (month-end close); say "
                             "'rebalance yearly' for Portfolio Visualizer's default, or 'rebalance quarterly', 'never rebalance'.")
        else:
            rb = "none"
    if buy_hold and rb != "none":
        notes.append("'Buy and hold' with a rebalance schedule: the schedule wins.")
    if wd_pct and wd_infl:
        # "withdraw 4% a year adjusted for inflation" is the classic 4% rule: 4% of the starting
        # balance, then that dollar amount rising with CPI
        per = {"monthly": 12, "quarterly": 4}.get(wfreq, 1)
        wd, wd_pct = wd_pct * pk.get("capital", 10_000.0), 0.0
        rate = f"{round(wd * per / pk.get('capital', 10_000.0) * 100, 4):g}%"
        each = f" (${wd:,.0f} a {'month' if per == 12 else 'quarter'})" if per > 1 else ""
        notes.append(f"Read as the '4% rule'" + (f" (here {rate})" if rate != "4%" else "") + f": withdraw ${wd * per:,.0f} a year{each} in the first year ({rate} of the "
                     "starting balance), then the same amount grown with inflation. Say "
                     f"'withdraw {rate} of the balance each year' for a percentage of the current balance.")
    newer = {k: flows[k] for k in ("contribution_start", "contribution_end", "withdrawal_start", "withdrawal_end",
                                   "contribution_growth", "withdrawal_growth", "contribution_dollars", "withdrawal_dollars")
             if flows.get(k) is not None}
    if infl:
        # each flow carries its own flag: "add $1,000 a month, then withdraw $50,000 a year adjusted for inflation"
        # indexes only the withdrawals
        newer["contribution_inflation"] = bool(flows.get("contribution_inflation"))
        newer["withdrawal_inflation"] = bool(flows.get("withdrawal_inflation"))
        if contrib and not flows.get("contribution_inflation"):
            notes.append("Only the withdrawals are indexed to inflation; the contributions stay fixed in dollars "
                         "(say 'add $1,000 a month adjusted for inflation' to index them too).")
        if wd and not flows.get("withdrawal_inflation"):
            notes.append("Only the contributions are indexed to inflation; the withdrawals stay fixed in dollars.")
    if band_rel is not None:
        newer["drift_band_relative"] = band_rel
    if band or band_rel:
        w = _band_unreachable(tree, band, band_rel, rb)
        if w:
            notes.append(w)
    _calendar_timing_note(rb, rb_day_hit, rb_stated, tree, notes)
    if rb_day:
        from .portfolio import rebalance_day_text
        notes.append(f"Rebalanced {rebalance_day_text(rb, rb_day)}, and on the first day"
                     + ("; Composer trades weekly, monthly, quarterly and yearly symphonies on the first trading day of each "
                        "period, near the close" if rb_day == "start" else "") + ".")
        newer["rebalance_day"] = rb_day
    p = Portfolio(tree=tree, rebalance=rb, drift_band=band, fill=fill, contribution=contrib, contribution_freq=cfreq,
                  withdrawal=wd, withdrawal_pct=wd_pct, withdrawal_freq=wfreq, inflation_adjust=infl,
                  description=raw, notes=notes, **pk, **extra, **newer, **tv,
                  **({"reinvest_dividends": kw["reinvest_dividends"]} if "reinvest_dividends" in kw else {}))
    if any(re.search(r"\bcross(?:over|under)\(", r) for r in _if_rules(tree)):
        notes.append("A 'crosses above/below' condition is true only on the day of the cross, so its branch is held for "
                     "that one day (until the next check); say 'is above' / 'is below' to hold it while the price stays there.")
    if any(re.search(r"\btret\(", r) for r in _rules(tree)):
        notes.append("Returns in the conditions are total returns (dividends reinvested, from adjusted prices).")
    p.benchmark = benchmark
    if isinstance(benchmark, str) and benchmark and not _blend_bench(benchmark):
        # "hold TQQQ vs TMF": the ticker after 'vs' is the comparison, not a holding; say so
        notes.append(f"Benchmark: {benchmark} (the ticker after 'vs' / 'compared with'): the report compares the portfolio "
                     "with it; it is not held.")
    if bench_rb:
        p.benchmark_rebalance = bench_rb
    elif benchmark and _blend_bench(benchmark):
        # a blend without its own schedule follows the portfolio's (report.blend_rebalance): say which
        from .report import blend_rebalance, _rebalance_words
        rb_, why_ = blend_rebalance(p)
        for i_, n_ in enumerate(notes):
            if n_.startswith("Benchmark: a blend of ") and n_.endswith("total returns."):
                notes[i_] = n_[:-1] + f", {_rebalance_words(rb_)}" + (
                    " like the portfolio (say e.g. 'vs 60/40 SPY/AGG rebalanced monthly' for another schedule)."
                    if why_ == "the portfolio's schedule" else ".")
    if want_proxies:
        p.proxies = _proxies_before_inception(tree, pk.get("start"), notes) or None
    if _has(tree, "filter", universe="NDX") and p.point_in_time:
        notes.append("Universe: Nasdaq-100 with point-in-time membership (stocks only selected while in the index; "
                     "former members included where price history exists).")
    return p


CF_FREQ = r"(?:every|each|per|a|an|once a|1) (month|quarter|year)"   # "a month" is normalised to "1 month"
CF_SCHED = (r"(?:,? (?:for (?:the first |the next )?\d+ years?|(?:until|through|to) (?:year \d+|\d{4})|"
            r"(?:from|starting(?: in| from)?|beginning(?: in)?|after|in) (?:year \d+|\d{4})(?! dollars)|"
            r"(?:starting|beginning) (?:in|after) \d+ years?|after \d+ years?|"
            r"(?:growing|increasing|rising|indexed|increased) (?:by |at )?\d+(?:\.\d+)?% (?:a|per|each|every) year|"
            r"\(?(?:adjusted|indexed|rising|growing|increased) (?:for|with|by|to) (?:inflation|cpi)\)?|inflation[- ](?:adjusted|indexed)|"
            r"in real terms|in (?:today's|todays|current|\d{4}) dollars))*")
CF_INFL = (r"\(?(?:adjusted|indexed|rising|growing|increased) (?:for|with|by|to) (?:inflation|cpi)\)?|inflation[- ](?:adjusted|indexed)|"
           r"in real terms|in (?:today's|todays|current|\d{4}) dollars")


def _cf_schedule(text: str, kind: str, out: dict, notes: list[str]) -> None:
    """'for 20 years', 'starting in 2000', 'from year 10', 'until 2030', 'growing 3% a year' ->
    <kind>_start / _end / _growth, in Portfolio's conventions: N < 1900 is year N of the backtest (both
    ends inclusive: end=20 is the first 20 years, start=21 the 21st year on); a year >= 1900 is 1 Jan of
    that year; an end date is written 'YYYY-12-31'."""
    t = text.lower()
    for m in re.finditer(r"(?:from|starting(?: in| from)?|beginning(?: in)?|after|in) (?:year (\d+)|(\d{4}))(?! dollars)", t):
        if m.group(1):
            out[f"{kind}_start"] = int(m.group(1)) + (1 if m.group(0).startswith("after") else 0)
        else:
            out[f"{kind}_start"] = int(m.group(2)) + (1 if m.group(0).startswith("after") else 0)
    for m in re.finditer(r"(?:(?:starting|beginning) (?:in|after)|after) (\d+) years?", t):
        out[f"{kind}_start"] = int(m.group(1)) + 1
    for m in re.finditer(r"for (?:the first |the next )?(\d+) years?", t):
        n = int(m.group(1))
        st = out.get(f"{kind}_start")
        if isinstance(st, int) and st >= 1900:
            out[f"{kind}_end"] = f"{st + n - 1}-12-31"
        else:
            out[f"{kind}_end"] = (st or 1) + n - 1
    for m in re.finditer(r"(?:until|through|to) (?:year (\d+)|(\d{4}))", t):
        out[f"{kind}_end"] = int(m.group(1)) if m.group(1) else f"{m.group(2)}-12-31"
    for m in re.finditer(r"(?:growing|increasing|rising|indexed|increased) (?:by |at )?(\d+(?:\.\d+)?)% (?:a|per|each|every) year", t):
        out[f"{kind}_growth"] = float(m.group(1)) / 100
    # "adjusted for inflation" / "in 2000 dollars" written with this flow applies to this flow only
    for m in re.finditer(CF_INFL, t):
        out[f"{kind}_inflation"] = True
        md = re.search(r"in (today's|todays|current|\d{4}) dollars", m.group(0))
        if md:
            if md.group(1).isdigit():
                out[f"{kind}_dollars"] = md.group(1)
            else:
                out[f"{kind}_dollars"] = "flow"
                notes.append(f"'{m.group(0)}': read as dollars of the first {kind} (its amount is paid in full then, "
                             f"and rises with inflation after that). Say 'in {_last_cpi_year()} dollars' for the dollars of the latest full year of CPI.")


def _last_cpi_year() -> int:
    m = data.cpi_monthly()
    if m.empty:
        return 2025
    n = m.groupby(m.index.year).size()
    full = n[n >= 12]
    return int(full.index[-1]) if len(full) else int(m.index[-1].year) - 1


def _year_text(v) -> str:
    return f"year {v} of the backtest" if isinstance(v, int) and v < 1900 else f"{v}"


class _Unit:
    """A match whose 'u' group is 'month' / 'quarter' also when it was written 'monthly' / 'quarterly'."""
    def __init__(self, m):
        self.m = m

    def group(self, k=0):
        if k == "u":
            return self.m.group("u") or {"monthly": "month", "quarterly": "quarter"}[self.m.group("ly").lower()]
        return self.m.group(k)


def _cash_flows(T: "Text", notes: list[str]) -> dict:
    """Contributions and withdrawals, with optional schedules:
    'add $1,000 a month for 20 years, then withdraw $50,000 a year', 'withdraw $40,000 a year starting in
    2000', 'contributions growing 3% a year', 'withdraw 4% a year from year 10'."""
    out: dict = {"contribution": 0.0, "contribution_freq": "monthly", "withdrawal": 0.0, "withdrawal_pct": 0.0,
                 "withdrawal_freq": "yearly"}
    # "the 4% rule" / "using the 3.5% rule": withdraw that % of the starting balance, then the same amount grown with
    # inflation (read below as a % withdrawal adjusted for inflation)
    T.rest = _sub_outside(r"(?i),? ?(?:(?:and |then )?(?:using|with|following|apply(?:ing)?|by) )?(?:the )?(\d+(?:\.\d+)?)% "
                          r"(?:safe withdrawal |withdrawal |spending )?rule\b", r", withdraw \1% every year adjusted for inflation", T.rest)
    # "withdraw 5% a year taken quarterly" / "withdraw $40,000 a year paid monthly": the yearly amount split over the
    # periods (as Portfolio Visualizer's withdrawal frequency): 1.25% of the balance every quarter
    def split(m):
        n = {"month": 12, "quarter": 4}[m.group("u")]
        infl = m.group("infl") or ""
        tail = f" {infl.strip(' ,')}" if infl else ""
        # "withdraw 4% a year adjusted for inflation, taken monthly": the 4% rule paid monthly - the rule's note
        # (below) gives the yearly and the monthly dollars
        rule = m.group("pct") and (infl or re.match(rf"(?i),? ?(?:{CF_INFL})", m.m.string[m.m.end():]))
        if m.group("pct"):
            # the 4% rule turns this share into dollars of the start balance: keep it exact ($40,000 a year, not $39,999.96)
            amt = (f"{float(m.group('pct')) / n:.15g}" if rule else f"{round(float(m.group('pct')) / n, 6):g}") + "% of the balance"
            shown = f"{m.group('pct')}% of the balance"
        else:
            amt = f"${round(float(m.group('usd')) / n, 2):g}"
            shown = f"${float(m.group('usd')):,.0f}"
        each = amt if m.group("pct") else f"${float(m.group('usd')) / n:,.2f}"
        if not rule:
            notes.append(f"'{m.group(0).strip(' ,')}' = {shown} a year taken every {m.group('u')}: {each} each {m.group('u')}.")
        return f"{m.group('v')} {amt} every {m.group('u')}{tail}"
    T.rest = _sub_outside(r"(?i)\b(?P<v>withdraw\w*|take out|spend\w*) (?:(?P<pct>\d+(?:\.\d+)?)%(?: of the (?:balance|portfolio))?|"
                          r"\$(?P<usd>\d+(?:\.\d+)?)) (?:a|per|each|every|1) year"
                          rf"(?P<infl>,? ?(?:{CF_INFL}))?,? (?:(?:taken|paid|withdrawn|split|spread) (?:out )?)?"
                          r"(?:(?:in|as) (?:equal )?(?:monthly|quarterly) (?:instalments|installments|payments) ?)?"
                          r"(?:(?:every|each|per|1) (?P<u>month|quarter)|(?P<ly>monthly|quarterly))",
                          lambda m: split(_Unit(m)), T.rest)
    # "add $500 monthly", "withdraw 4% annually" -> "... every month / every year"
    T.rest = _sub_outside(r"(?i)\b((?:add(?:ing)?|invest(?:ing)?|contribut\w+|deposit\w*|put(?:ting)? in|withdraw\w*|take out|spend\w*|draw\w*(?: down)?)"
                          r"(?: an additional| another| a further)? (?:\$\d+(?:\.\d+)?|\d+(?:\.\d+)?%(?: of the (?:balance|portfolio))?)(?: more)?) "
                          r"(monthly|quarterly|annually|yearly)\b",
                          lambda m: f"{m.group(1)} every {({'monthly': 'month', 'quarterly': 'quarter'}).get(m.group(2).lower(), 'year')}", T.rest)
    m = T.find(rf",? ?(?:and )?(?:add(?:ing)?|invest(?:ing)?|contribut\w+|deposit\w*|put(?:ting)? in)(?: an additional| another| a further)? \$(\d+(?:\.\d+)?)(?: more)? {CF_FREQ}(?P<s>{CF_SCHED})"
               rf"|,? ?(?:with )?(?:\$(\d+(?:\.\d+)?) )?(monthly|quarterly|yearly|annual) contributions?(?: of \$(\d+(?:\.\d+)?))?(?P<s2>{CF_SCHED})")
    if m:
        amt = m.group(1) or m.group(4) or m.group(6)
        if amt is None:
            raise ParseError("How much is contributed? e.g. 'add $500 every month'.")
        out["contribution"] = float(amt)
        out["contribution_freq"] = FREQ_WORDS[(m.group(2) or m.group(5)).lower()]
        _cf_schedule(m.group("s") or m.group("s2") or "", "contribution", out, notes)
    m = T.find(rf",? ?(?:and )?(?P<then>then )?(?:withdraw\w*|take out|spend\w*|draw\w*(?: down)?)(?: of)? (?:\$(\d+(?:\.\d+)?)|{NUM}%(?: of the (?:balance|portfolio))?) {CF_FREQ}(?P<s>{CF_SCHED})"
               rf"|,? ?(?:with )?(?:a )?{NUM}% (?:annual |yearly )?(?:withdrawal|spending) rate(?P<s2>{CF_SCHED})")
    if m:
        if m.group(2):
            out["withdrawal"] = float(m.group(2))
        elif m.group(3):
            out["withdrawal_pct"] = float(m.group(3)) / 100
        else:
            out["withdrawal_pct"] = float(m.group(6)) / 100
        out["withdrawal_freq"] = FREQ_WORDS[(m.group(4) or "year").lower()]
        _cf_schedule(m.group("s") or m.group("s2") or "", "withdrawal", out, notes)
        if m.group("then") and out.get("withdrawal_start") is None:
            if out.get("contribution_end") is None:
                raise ParseError("'then withdraw ...': after what? Say e.g. 'add $1,000 a month for 20 years, then withdraw $50,000 a year'.")
            ce = out["contribution_end"]
            out["withdrawal_start"] = ce + 1 if isinstance(ce, int) else int(str(ce)[:4]) + 1
            notes.append(f"Withdrawals start when the contributions stop ({_year_text(out['withdrawal_start'])}).")
    for m in T.findall(r",? ?(?:with |and )?(?:the )?(contributions?|deposits?|withdrawals?|spending) (?:growing|increasing|rising|increased) (?:by |at )?(\d+(?:\.\d+)?)% (?:a|per|each|every) year"):
        out["contribution_growth" if m.group(1).lower().startswith(("contrib", "deposit")) else "withdrawal_growth"] = float(m.group(2)) / 100
    for kind in ("contribution", "withdrawal"):
        if (out.get(f"{kind}_start") is not None or out.get(f"{kind}_end") is not None or out.get(f"{kind}_growth")) and not (
                out[kind] or (kind == "withdrawal" and out["withdrawal_pct"])):
            raise ParseError(f"A {kind} schedule was given but no {kind} amount, e.g. 'add $500 a month' / 'withdraw $40,000 a year'.")
        if out.get(f"{kind}_growth") and kind == "withdrawal" and out["withdrawal_pct"]:
            raise ParseError("A growth rate applies to $ withdrawals, not to a % of the balance.")
    return out


def _if_rules(n: dict) -> list[str]:
    out = [n["if"]] if isinstance(n.get("if"), str) else []
    for c in n.get("children") or []:
        out += _if_rules(c)
    for k in ("then", "else", "fallback"):
        if isinstance(n.get(k), dict):
            out += _if_rules(n[k])
    return out


def _rules(n: dict) -> list[str]:
    """Every condition in a tree: if rules and filter requirements."""
    out = [n["if"]] if isinstance(n.get("if"), str) else []
    if isinstance((n.get("filter") or {}).get("require"), str):
        out.append(n["filter"]["require"])
    for c in n.get("children") or []:
        out += _rules(c)
    for k in ("then", "else", "fallback"):
        if isinstance(n.get(k), dict):
            out += _rules(n[k])
    return out


DYNAMIC_WEIGHTS = ("inverse_vol", "risk_parity", "min_variance", "max_sharpe", "max_diversification", "market_cap")


def _dynamic(n: dict) -> bool:
    """True when any node's weights move with the market (inverse volatility, risk parity, ...)."""
    if n.get("weights") in DYNAMIC_WEIGHTS or (n.get("filter") or {}).get("weights") in DYNAMIC_WEIGHTS:
        return True
    return any(_dynamic(c) for c in n.get("children") or []) or any(
        isinstance(n.get(k), dict) and _dynamic(n[k]) for k in ("then", "else", "fallback"))


def _has(n: dict, key: str, universe: str | None = None) -> bool:
    if key in n and (universe is None or n.get("universe") == universe):
        return True
    for k in ("children",):
        for c in n.get(k) or []:
            if _has(c, key, universe):
                return True
    for k in ("then", "else", "fallback"):
        if isinstance(n.get(k), dict) and _has(n[k], key, universe):
            return True
    return False
