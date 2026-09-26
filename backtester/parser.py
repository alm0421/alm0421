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
import re
from dataclasses import dataclass

from . import data
from .portfolio import Portfolio
from .strategy import Strategy

NUM = r"(\d+(?:\.\d+)?)"

WORD_NUMS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "twenty": 20, "thirty": 30, "fifty": 50, "hundred": 100,
}

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
    "t-bills": "BIL", "treasury bills": "BIL", "aggregate bonds": "AGG", "bonds": "AGG", "gold": "GLD",
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
}

DOW = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4}
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september",
          "october", "november", "december"]

SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB", "XLRE", "XLC"]

# words that may be left over after everything meaningful was recognised
STOP = set("""
a an the it its it's is are was were be been being this that these those then than also just only
when whenever if once and or but so of on at in into for to with by from as per each every any all
stock stocks share shares price prices ticker tickers symbol symbols etf etfs fund funds day days
today same trade trades trading traded position positions order orders strategy backtest test please
i me my we our us you want would like let lets let's run show see what happens using use rule rules
signal signals market markets just simply both plus ones one's there here now time times
buy buying bought sell selling sold hold holding keep own go goes going get enter exit cover long
short purchase invest investing put allocate allocation portfolio money account close
""".split())


class ParseError(ValueError):
    pass


@dataclass
class Ctx:
    """Placeholders for the series a condition refers to."""
    c: str = "close"
    o: str = "open"
    h: str = "high"
    l: str = "low"
    v: str = "volume"

    @classmethod
    def for_ticker(cls, t: str | None) -> "Ctx":
        if not t:
            return cls()
        s = f'sym("{t}")'
        return cls(f"{s}.close", f"{s}.open", f"{s}.high", f"{s}.low", f"{s}.volume")

    @property
    def base(self) -> bool:
        return self.c == "close"


# ----------------------------------------------------------------- helpers

def _normalize(text: str) -> str:
    t = text.replace("’", "'").replace("–", "-").replace("—", "-").replace("“", '"').replace("”", '"')
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"(\d)\s*percent\b", r"\1%", t, flags=re.I)
    t = re.sub(r"\bper cent\b", "%", t, flags=re.I)
    for w, n in WORD_NUMS.items():
        t = re.sub(rf"\b{w}\b", str(n), t, flags=re.I)
    t = re.sub(r"\b(\d+)-(day|week|month|year|period|bar|session)s?\b", r"\1 \2", t, flags=re.I)
    t = re.sub(r"\b(\d+)(d|w|m)\b(?!\s*%)", lambda m: m.group(1) + {"d": " day", "w": " week", "m": " month"}[m.group(2).lower()], t)
    t = re.sub(r"\bovernight\b", "1 day", t, flags=re.I)
    t = re.sub(r"\ba (day|week|month)\b(?! in a row)", r"1 \1", t, flags=re.I)
    t = re.sub(r"\bhalf\b", "50%", t, flags=re.I)
    t = re.sub(r"\ba third\b", "33.3333%", t, flags=re.I)
    t = re.sub(r"\ba quarter\b(?! of)", "25%", t, flags=re.I)
    while re.search(r"(\d),(\d{3})\b", t):
        t = re.sub(r"(\d),(\d{3})\b", r"\1\2", t)  # 1,000,000 -> 1000000
    t = re.sub(r"\$\s*(\d+(?:\.\d+)?)\s*k\b", lambda m: "$" + str(int(float(m.group(1)) * 1000)), t, flags=re.I)
    t = re.sub(r"\$\s*(\d+(?:\.\d+)?)\s*m(?:illion)?\b", lambda m: "$" + str(int(float(m.group(1)) * 1_000_000)), t, flags=re.I)
    return t


def _known() -> set[str]:
    return set(data.available_tickers())


def find_tickers(text: str, strict: bool = False) -> list[str]:
    """Tickers mentioned in `text` (original casing), in order of appearance.

    With strict=True, ticker-looking words without price data raise ParseError instead of being ignored.
    """
    known = _known()
    found: list[tuple[int, str]] = []
    unknown: list[str] = []
    stripped = re.sub(r"`[^`]*`", " ", text)
    for m in re.finditer(r"(?<![\w^])(\$|\^)?([A-Z]{1,5}(?:\.[A-Z])?)\b", stripped):
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
        raise ParseError(
            f"No price data for: {', '.join(dict.fromkeys(unknown))}. Available tickers: Nasdaq-100 members, "
            f"former members, SPY, QQQ and ETFs such as {', '.join(data.etfs()[:25])}... "
            f"(run `python -m backtester --tickers-list` for all).")
    out: list[str] = []
    for _, s in sorted(found):
        if s not in out:
            out.append(s)
    return out


def _cmp(word: str) -> str:
    w = word.strip().lower()
    if w in ("below", "under", "less than", "<", "lower than", "beneath", "at most", "<=", "under its", "falls below"):
        return "<"
    if w in ("above", "over", "greater than", ">", "higher than", "more than", "at least", ">="):
        return ">"
    raise ParseError(f"unknown comparison {word!r}")


CMPW = r"(below|under|less than|lower than|beneath|<=?|above|over|greater than|higher than|more than|>=?)"
MA = r"(simple |exponential |weighted )?(?:moving average|moving avg|ma|sma|ema|wma)\b"


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


def _ma(kind: str | None, x: str, n: str, unit: str | None, raw: str) -> str:
    """Moving average text; N-week / N-month averages use weekly / monthly closes (as charts do)."""
    u = (unit or "day").lower()
    xs = "" if x == "close" else f", {x}"
    if u.startswith("month"):
        return f"monthly_sma({int(float(n))}{xs})"
    if u.startswith("week"):
        return f"weekly_sma({int(float(n))}{xs})"
    k = "sma"
    if (kind and "exp" in kind) or re.search(r"\bema\b", raw):
        k = "ema"
    elif (kind and "weight" in kind) or re.search(r"\bwma\b", raw):
        k = "wma"
    return f"{k}({x}, {_period(n, unit)})"


def _osc(name: str, n: str | None, ctx: Ctx) -> str:
    """Oscillator expression for a phrase name."""
    name = name.lower().replace("%", "").strip()
    if name.startswith("rsi"):
        return f"rsi({ctx.c}, {int(float(n or 14))})"
    if not ctx.base:
        raise ParseError(f"{name} of another ticker is not supported in English; use backticks")
    nn = int(float(n)) if n else None
    if name.startswith("stoch"):
        return f"stoch_k({nn or 14}, 3)"
    if name.startswith("cci"):
        return f"cci({nn or 20})"
    if name.startswith("williams") or name.startswith("willr") or name == "r":
        return f"willr({nn or 14})"
    if name.startswith("mfi") or name.startswith("money flow"):
        return f"mfi({nn or 14})"
    if name.startswith("adx"):
        return f"adx({nn or 14})"
    raise ParseError(f"unknown indicator {name!r}")


OSC = r"(rsi|stochastic(?: %?k)?|stoch|cci|williams %?r|willr|mfi|money flow index|adx)"


# ----------------------------------------------------------------- conditions

def parse_condition(text: str, ctx: Ctx) -> tuple[str | None, str]:
    """Return (expression, leftover words) for one sub-clause."""
    s = " " + text.strip().lower() + " "
    parts: list[str] = []

    def take(pattern: str, fn) -> None:
        nonlocal s
        while True:
            m = re.search(pattern, s)
            if not m:
                return
            parts.append(fn(m))
            s = re.sub(r" +", " ", s[: m.start()] + " " + s[m.end():])

    c, o, h, l, v = ctx.c, ctx.o, ctx.h, ctx.l, ctx.v
    xs = "" if ctx.base else f", {c}"
    dn = r"(?:down|lower|declin\w*|fall\w*|fell|drop\w*|red|los\w*|negative|closes? down|closes? lower)"
    up = r"(?:up|higher|ris\w*|rose|gain\w*|green|advanc\w*|positive|closes? up|closes? higher)"
    down_x = "down_days" if ctx.base else f"down_streak({c})"
    up_x = "up_days" if ctx.base else f"up_streak({c})"
    chg = "change" if ctx.base else f"ret({c}, 1)"

    # raw rule language in backticks
    take(r"`([^`]+)`", lambda m: f"({m.group(1)})")

    # negations of relations
    s = re.sub(r"\b(?:is|closes?|trades?) not (above|over)\b", "is below", s)
    s = re.sub(r"\b(?:is|closes?|trades?) not (below|under)\b", "is above", s)

    # consecutive down / up closes ("exactly N" fires only on the Nth day)
    take(rf"{dn} (?:for )?exactly {NUM} (?:straight |consecutive )?(?:days|closes|sessions|bars)(?: in a row| straight)?|exactly {NUM} (?:consecutive|straight) {dn} (?:days|closes|sessions|bars)",
         lambda m: f"{down_x} == {int(float(m.group(1) or m.group(2)))}")
    take(rf"{up} (?:for )?exactly {NUM} (?:straight |consecutive )?(?:days|closes|sessions|bars)(?: in a row| straight)?|exactly {NUM} (?:consecutive|straight) {up} (?:days|closes|sessions|bars)",
         lambda m: f"{up_x} == {int(float(m.group(1) or m.group(2)))}")
    take(rf"(?:at least )?{NUM} (?:or more )?(?:consecutive|straight|successive) {dn} (?:days|closes|sessions|bars)",
         lambda m: f"{down_x} >= {int(float(m.group(1)))}")
    take(rf"(?:at least )?{NUM} (?:or more )?(?:consecutive|straight|successive) {up} (?:days|closes|sessions|bars)",
         lambda m: f"{up_x} >= {int(float(m.group(1)))}")
    take(rf"{dn}(?: for)?(?: at least)? {NUM} (?:or more )?(?:straight |consecutive |trading )?(?:days|closes|sessions|bars|times)(?: in a row| straight| consecutively)",
         lambda m: f"{down_x} >= {int(float(m.group(1)))}")
    take(rf"{up}(?: for)?(?: at least)? {NUM} (?:or more )?(?:straight |consecutive |trading )?(?:days|closes|sessions|bars|times)(?: in a row| straight| consecutively)",
         lambda m: f"{up_x} >= {int(float(m.group(1)))}")

    # MACD
    take(r"macd(?: line)? (cross(?:es|ed)? (?:above|over)|cross(?:es|ed)? (?:below|under)|is above|is below|above|below) (?:its |the )?signal(?: line)?",
         lambda m: (f"crossover(macd(), macd_signal())" if "cross" in m.group(1) and ("above" in m.group(1) or "over" in m.group(1))
                    else f"crossunder(macd(), macd_signal())" if "cross" in m.group(1)
                    else f"macd() {'>' if 'above' in m.group(1) else '<'} macd_signal()") if ctx.base else _unsupported("MACD of another ticker"))
    take(r"macd(?: histogram)? (turns positive|turns negative|crosses above zero|crosses below zero|crosses above 0|crosses below 0|is positive|is negative|is above zero|is below zero|is above 0|is below 0)",
         lambda m: ((f"crossover(macd_hist(), 0)" if "histogram" in m.group(0) else "crossover(macd(), 0)") if ("turns positive" in m.group(1) or "crosses above" in m.group(1))
                    else (f"crossunder(macd_hist(), 0)" if "histogram" in m.group(0) else "crossunder(macd(), 0)") if ("turns negative" in m.group(1) or "crosses below" in m.group(1))
                    else f"{'macd_hist()' if 'histogram' in m.group(0) else 'macd()'} {'>' if ('positive' in m.group(1) or 'above' in m.group(1)) else '<'} 0"))
    # directional movement
    take(r"(?:\+di|plus di) (?:is )?(above|below|crosses above|crosses below) (?:the )?(?:-di|minus di)",
         lambda m: (f"crossover(plus_di(), minus_di())" if m.group(1) == "crosses above" else
                    f"crossunder(plus_di(), minus_di())" if m.group(1) == "crosses below" else
                    f"plus_di() {_cmp(m.group(1))} minus_di()"))
    # oscillator crossings and levels
    take(rf"(?:the )?(?:(\d+) (?:day|period|bar) )?{OSC}\s*(?:\(\s*(\d+)\s*\)|(\d+))? (?:line )?cross(?:es|ed)? (above|over|below|under) (-?{NUM})",
         lambda m: f"{'crossover' if m.group(5) in ('above', 'over') else 'crossunder'}({_osc(m.group(2), m.group(1) or m.group(3) or m.group(4), ctx)}, {m.group(6)})")
    take(rf"(?:the )?(?:(\d+) (?:day|period|bar) )?{OSC}\s*(?:\(\s*(\d+)\s*\)|(\d+))?(?: value| reading)? (?:is |closes |drops |falls |rises |goes |reads )?{CMPW} (-?{NUM})",
         lambda m: f"{_osc(m.group(2), m.group(1) or m.group(3) or m.group(4), ctx)} {_cmp(m.group(5))} {m.group(6)}")

    # moving-average relationships between two averages
    def ma_vs_ma(m):
        a = _ma(m.group(3), c, m.group(1), m.group(2), m.group(0))
        b = _ma(m.group(8), c, m.group(6), m.group(7), m.group(0))
        rel = (m.group(4) or "") + m.group(5)
        if "cross" in rel:
            return f"{'crossover' if 'above' in rel or 'over' in rel else 'crossunder'}({a}, {b})"
        return f"{a} {_cmp(m.group(5))} {b}"
    take(rf"(?:the |its )?{NUM} (day|week|month) {MA} (?:is |has )?(cross(?:es|ed)? )?(above|over|below|under) (?:the |its )?{NUM} (day|week|month) {MA}", ma_vs_ma)
    take(r"golden cross", lambda m: f"crossover(sma({c}, 50), sma({c}, 200))")
    take(r"death cross", lambda m: f"crossunder(sma({c}, 50), sma({c}, 200))")

    # price vs moving average, optionally by X%
    def px_vs_ma(m):
        pct, cross, rel, n, unit, kind = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5), m.group(6)
        ma = _ma(kind, c, n, unit, m.group(0))
        op = _cmp(rel)
        if cross:
            return f"{'crossover' if op == '>' else 'crossunder'}({c}, {ma})"
        if pct:
            f = float(pct) / 100
            return f"{c} {op}= {ma} * {1 - f if op == '<' else 1 + f:.6g}"
        return f"{c} {op} {ma}"
    take(rf"(?:{NUM}%(?: or more)? )?(?:(?:closes?|trades?|is|price is|stays?|remains?|falls|drops|moves|goes|rises|climbs) )?(?:back )?(cross(?:es|ed)? (?:back )?)?{CMPW} (?:its |the )?{NUM} (day|week|month|period|bar)s? {MA}",
         px_vs_ma)

    # drawdown from a high
    take(rf"(?:is |trades? |closes? )?(?:at least |more than )?{NUM}% (?:or more )?(?:below|off|under) (?:its |the )?(?:(\d+) (day|week|month|year) high|52 week high|all[- ]time high|high)",
         lambda m: f"drawdown({c}{', ' + str(252 if '52 week' in m.group(0) else _period(m.group(2), m.group(3))) if (m.group(2) or '52 week' in m.group(0)) else ''}) <= {-float(m.group(1)) / 100:g}")

    # breakouts vs prior N-day range
    take(r"(?:closes? |breaks? |trades? |moves? )?(above|over|below|under) (?:its |the )?(?:previous |prior |last )?(\d+) (day|week|month|bar)s? (high|low)",
         lambda m: (f"{c} > ref(highest({h}, {_period(m.group(2), m.group(3))}), 1)" if m.group(4) == "high" and _cmp(m.group(1)) == ">"
                    else f"{c} < ref(lowest({l}, {_period(m.group(2), m.group(3))}), 1)" if m.group(4) == "low" and _cmp(m.group(1)) == "<"
                    else f"{c} {_cmp(m.group(1))} ref({'highest(' + h if m.group(4) == 'high' else 'lowest(' + l}, {_period(m.group(2), m.group(3))}), 1)"))
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
    take(r"(?:closes? |is |trades? )?(above|below) (?:the )?supertrend", lambda m: f"{c} {_cmp(m.group(1))} supertrend(10, 3)")
    take(r"supertrend (?:turns |flips )?(bullish|bearish)", lambda m: f"{'crossover' if m.group(1) == 'bullish' else 'crossunder'}({c}, supertrend(10, 3))")
    take(r"(?:closes? |is |trades? )?(above|below) (?:the )?(?:parabolic )?sar", lambda m: f"{c} {_cmp(m.group(1))} sar()")
    take(r"(?:closes? |is |trades? )?(above|below) (?:the )?(upper|lower) keltner(?: channel| band)?",
         lambda m: f"{c} {_cmp(m.group(1))} keltner_{m.group(2)}(20, 2)")

    # gaps
    def gap(m):
        pct = m.group(2)
        g = "gap" if ctx.base else f"({o} / ref({c}, 1) - 1)"
        if pct is None:
            return f"{g} {'<' if m.group(1) == 'down' else '>'} 0"
        f = float(pct) / 100
        return f"{g} <= {-f:g}" if m.group(1) == "down" else f"{g} >= {f:g}"
    take(rf"(?:opens? with a |has a )?gaps? (down|up)(?: by)?(?: more than| at least| over)?(?: {NUM}%)?(?: or more)?", gap)

    # volatility
    take(rf"(?:(\d+) day )?(?:historical |realized |annualized )?volatility (?:is )?{CMPW} {NUM}%",
         lambda m: f"volatility({int(m.group(1) or 20)}{xs}) {_cmp(m.group(2))} {float(m.group(3)) / 100:g}")

    # percentage moves over N days
    def move(m):
        verb, pct, n = m.group(1), float(m.group(2)) / 100, m.group(3)
        n = _period(n, m.group(4)) if n else (_period(1, m.group(4)) if m.group(4) else 1)
        down = re.match(r"(down|fall|fell|drop|declin|los|lower|sink|sank|plung|tumbl|slid|is down)", verb) is not None
        r = chg if n == 1 else f"ret({c}, {n})"
        return f"{r} <= {-pct:g}" if down else f"{r} >= {pct:g}"
    verbs = r"(down|up|falls?|fell|drops?|dropped|declines?|declined|loses?|lost|gains?|gained|rises?|rose|jumps?|jumped|rall(?:ies|ied|y)|climbs?|climbed|lower|higher|sinks?|sank|plunges?|plunged|tumbles?|tumbled|slides?|slid|is down|is up|has fallen|has risen|has dropped|has gained)"
    take(rf"(?:closes? |trades? |is |has )?{verbs}(?: by)?(?: more than| at least| over| greater than)? {NUM}%(?: or more)?(?:(?: in| over| during| within)(?: the)?(?: last| past| prior| previous)? (?:(\d+) )?(day|week|month|session|bar|year)s?| today| on the day| in a single day| in one day| intraday)?", move)
    take(rf"(?:its |the )?(?:(\d+) (day|week|month|year) )?(?:return|performance|momentum) (?:is )?{CMPW} (-?{NUM})%",
         lambda m: f"ret({c}, {_period(m.group(1) or 1, m.group(2))}) {_cmp(m.group(3))} {float(m.group(4)) / 100:g}")
    take(r"(?:its |the )?(?:(\d+) (day|week|month|year) )?(?:return|performance|momentum) (?:is )?(positive|negative)",
         lambda m: f"ret({c}, {_period(m.group(1) or 1, m.group(2))}) {'>' if m.group(3) == 'positive' else '<'} 0")

    # internal bar strength / position in range
    take(rf"ibs (?:is )?{CMPW} {NUM}", lambda m: f"{'ibs' if ctx.base else f'(({c} - {l}) / ({h} - {l}))'} {_cmp(m.group(1))} {m.group(2)}")
    take(rf"closes? in the (bottom|lower|top|upper) {NUM}% of (?:its|the)(?: day's| daily)? range",
         lambda m: f"ibs < {float(m.group(2)) / 100:g}" if m.group(1) in ("bottom", "lower") else f"ibs > {1 - float(m.group(2)) / 100:g}")
    take(r"closes? (?:near|at) (?:its |the )?(?:daily |day's )?low", lambda m: "ibs < 0.2")
    take(r"closes? (?:near|at) (?:its |the )?(?:daily |day's )?high", lambda m: "ibs > 0.8")

    # previous-day references
    take(r"closes? (below|under|above|over) (?:the )?(?:previous|prior|yesterday's|last) (?:day's )?(high|low|close)",
         lambda m: f"{c} {_cmp(m.group(1))} ref({ {'high': h, 'low': l, 'close': c}[m.group(2)] }, 1)")
    take(r"inside day", lambda m: f"({h} < ref({h}, 1)) and ({l} > ref({l}, 1))")
    take(r"outside day", lambda m: f"({h} > ref({h}, 1)) and ({l} < ref({l}, 1))")

    # bollinger bands
    take(r"(closes? |is |trades? |falls |drops |rises )?(below|under|above|over) (?:the )?(lower|upper) bollinger band",
         lambda m: f"{c} {_cmp(m.group(2))} bb_{m.group(3)}(20, 2)" if ctx.base
         else f"{c} {_cmp(m.group(2))} sma({c}, 20) {'-' if m.group(3) == 'lower' else '+'} 2 * stdev({c}, 20)")

    # volume
    take(rf"volume (?:is )?(?:at least |more than |above |over )?{NUM} ?(?:x|times) (?:its |the )?(?:(\d+) day )?(?:average|avg)(?: volume)?(?: of the (?:last|past) (\d+) days)?",
         lambda m: f"{v} >= {m.group(1)} * sma({v}, {m.group(2) or m.group(3) or 20})")

    # plain level comparisons, e.g. "VIX is above 30", "closes above 100"
    take(rf"(?:closes?|is|trades?|stays?) {CMPW} \$?(-?{NUM})(?![\d%])(?! (?:day|week|month|bar))",
         lambda m: f"{c} {_cmp(m.group(1))} {m.group(2)}")

    # calendar
    take(r"(?:on )?(monday|tuesday|wednesday|thursday|friday)s?", lambda m: f"dow == {DOW[m.group(1)]}")
    take(r"(?:on )?(?:the )?last trading day of the month|month[- ]end", lambda m: "trading_days_left_in_month == 1")
    take(r"(?:on )?(?:the )?first trading day of the month", lambda m: "trading_day_of_month == 1")
    take(r"(?:during |in )?(?:the )?(january|february|march|april|may|june|july|august|september|october|november|december)",
         lambda m: f"month == {MONTHS.index(m.group(1)) + 1}")

    # single down / up day (after the % and streak patterns)
    take(r"(?:closes? (?:down|lower|red)|down (?:day|close)|red day|negative day|first down (?:day|close)|trades? down|is down|goes down|moves down|falls|drops|declines)",
         lambda m: f"{chg} < 0")
    take(r"(?:closes? (?:up|higher|green)|up (?:day|close)|green day|positive day|first up (?:day|close)|higher close|trades? up|is up|goes up|moves up|rises|gains)",
         lambda m: f"{chg} > 0")
    take(r"(?:is |the (?:trade|position) is )?(?:profitable|in profit|shows a profit)", lambda m: "pnl > 0")

    words = [w for w in re.findall(r"[a-z%]+|\d+(?:\.\d+)?", s) if w not in STOP]
    expr_ = " and ".join(f"({p})" for p in parts) if parts else None
    return expr_, " ".join(words)


def _unsupported(what: str) -> str:
    raise ParseError(f"{what} is not supported in English; write it in backticks (see --help-expr)")


def _split_top(text: str, word: str) -> list[str]:
    # "2% or more", "or less", "or higher" are part of a phrase, not an alternative
    return re.split(rf"\b{word}\b(?! (?:more|less|fewer|higher|lower|better|worse|greater|so|after|in \d)\b)(?![^`]*`)", text)


def parse_conditions(text: str, traded: list[str], strict: bool = True, as_list: bool = False):
    """Parse `x and y or z` style condition text into one expression.

    Each and/or part is bound to the ticker named inside it (e.g. "SPY is above its 200-day moving
    average" -> SPY's series); parts that name no other ticker refer to the traded ticker.
    """
    text = re.sub(r"\b(?:but only if|but only when|only if|only when|provided that|provided|as long as|so long as|while|but)\b(?![^`]*`)", " and ", text)
    and_parts = re.split(r"(?:\band\b|,|;|\bwith\b(?! a))(?![^`]*`)", text)
    exprs, bad = [], []
    for part in and_parts:
        if not part.strip():
            continue
        or_exprs = []
        for o in _split_top(part, "or"):
            if not o.strip():
                continue
            mentioned = [t for t in find_tickers(o, strict=True) if t not in traded]
            if len(mentioned) > 1:
                raise ParseError(f"'{o.strip()}' mentions several tickers ({', '.join(mentioned)}); split it into separate conditions")
            ctx = Ctx.for_ticker(mentioned[0] if mentioned else None)
            o_clean = o
            for t in find_tickers(o):
                o_clean = re.sub(rf"(?<![\w])[\$^]?{re.escape(t.lstrip('^'))}\b", " ", o_clean, flags=re.I)
            for name, sym in COMPANIES.items():
                o_clean = re.sub(rf"\b{re.escape(name)}\b", " ", o_clean, flags=re.I)
            e, left = parse_condition(o_clean, ctx)
            if e:
                or_exprs.append(e)
            if left and (strict or not e):
                bad.append(left if e else o.strip())
        if or_exprs:
            exprs.append(or_exprs[0] if len(or_exprs) == 1 else "(" + " or ".join(or_exprs) + ")")
    if bad:
        raise ParseError(
            "Could not interpret: " + "; ".join(repr(b) for b in bad)
            + ".\nRephrase, or write that part in the rule language inside backticks, e.g. "
              "`rsi(2) < 10 and close > sma(close, 200)` (see --help-expr)."
        )
    if not exprs:
        raise ParseError(f"No conditions found in {text.strip()!r}")
    return exprs if as_list else " and ".join(exprs)


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


def _flip(entry: str, below: bool) -> str | None:
    """'sell when it crosses back below' -> the entry's first price/average comparison, reversed."""
    for term in split_and(entry):
        n = ast.parse(term, mode="eval").body
        if isinstance(n, ast.Compare) and len(n.ops) == 1 and isinstance(n.ops[0], (ast.Gt, ast.GtE, ast.Lt, ast.LtE)):
            a, b = ast.unparse(n.left), ast.unparse(n.comparators[0])
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ("crossover", "crossunder") and len(n.args) == 2:
            a, b = ast.unparse(n.args[0]), ast.unparse(n.args[1])
        else:
            continue
        return f"{a} {'<' if below else '>'} {b}"
    return None


# ----------------------------------------------------------------- shared options

class Text:
    """Normalised sentence with consumption tracking: everything recognised is blanked out, and
    whatever is left at the end must be filler words, otherwise the parse is refused."""

    def __init__(self, t: str):
        self.t = t
        self.low = t.lower()
        self.rest = t          # original casing is kept so tickers stay recognisable

    def find(self, pattern: str, consume: bool = True):
        m = re.search(pattern, self.rest, flags=re.I)
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


def common_options(T: Text, notes: list[str]) -> dict:
    kw: dict = {}
    m = T.find(r"(?:start(?:ing)? with|capital of|initial capital of|account of|begin(?:ning)? with|with an? (?:initial |starting )?(?:balance|investment) of|with) \$(\d+(?:\.\d+)?)(?! (?:per|a|each|every|monthly|quarterly|yearly|annually))(?:(?: of)? (?:capital|in capital))?")
    if m:
        kw["capital"] = float(m.group(1))
    m = T.find(rf"(?:(?:with |and )?{NUM} ?(?:bps|basis points?)(?: of)? slippage|slippage(?: of)? {NUM} ?(?:bps|basis points?)|(?:with |and )?{NUM}% slippage|slippage of {NUM}%)(?: per side| each way| per trade)?")
    if m:
        g = m.groups()
        kw["slippage_bps"] = float(g[0] or g[1]) if (g[0] or g[1]) else float(g[2] or g[3]) * 100
    m = T.find(rf"(?:(?:with |and )?\$\s?{NUM} (?:per trade|commissions?|per order|a trade|an order)(?: commissions?)?|commissions? of \$\s?{NUM}(?: per (?:trade|order))?)")
    if m:
        kw["commission"] = float(m.group(1) or m.group(2))
    m = T.find(rf"(?:with |and )?\$\s?{NUM} (?:per share|a share)(?: commissions?)?|commissions? of \$\s?{NUM} per share")
    if m:
        kw["commission_per_share"] = float(m.group(1) or m.group(2))
    m = T.find(rf"(?:with |and )?{NUM}% commissions?|commissions? of {NUM}%")
    if m:
        kw["commission_pct"] = float(m.group(1) or m.group(2)) / 100
    for word in ("slippage", "commission"):
        if re.search(rf"\b{word}", T.rest, re.I):
            raise ParseError(f"Could not understand the {word} amount. Write e.g. '5 bps slippage', '0.1% slippage', "
                             f"'$1 per trade commission', '$0.005 per share commission' or '0.1% commission'.")
    # dates
    mfrom = T.find(r"(?:from|since|starting(?: in)?|beginning(?: in)?|between|after) (\d{4})(?:-(\d{2})-(\d{2}))?")
    if mfrom:
        y = mfrom.group(1)
        kw["start"] = f"{y}-{mfrom.group(2)}-{mfrom.group(3)}" if mfrom.group(2) else f"{y}-01-01"
    mto = T.find(r"(?:to|until|through|thru|before|and|ending(?: in)?) (\d{4})(?:-(\d{2})-(\d{2}))?\b")
    if mto:
        y = mto.group(1)
        kw["end"] = f"{y}-{mto.group(2)}-{mto.group(3)}" if mto.group(2) else f"{y}-12-31"
    m = T.find(r"(?:in|during|for) (\d{4})(?! \w*%)\b")
    if m and "start" not in kw:
        kw["start"], kw["end"] = f"{m.group(1)}-01-01", f"{m.group(1)}-12-31"
    # cash interest
    if T.find(r"(?:idle )?cash (?:earns|pays|yields) (?:nothing|no interest|0%)|no interest on cash|without (?:cash )?interest"):
        kw["cash_rate"] = None
    m = T.find(rf"(?:idle )?cash (?:earns|pays|yields) {NUM}%(?: (?:a|per) year| annually)?")
    if m:
        kw["cash_rate"] = float(m.group(1)) / 100
    # benchmark
    m = T.find(r"(?:compared? (?:it )?(?:to|with|against)|benchmark(?:ed)?(?: it)?(?: (?:to|against))?|versus|vs\.?|against) ([\^$]?[a-z]{1,5})\b")
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
    r"allocat\w+|portfolio of|contribut\w+|withdraw\w*|\d+/\d+ (?:[a-z]+/[a-z]+)|"
    r"(?:hold|invest|put)\s+\d+(?:\.\d+)?%|\d+(?:\.\d+)?% (?:in |of )?(?:[a-z^$]{1,5}|cash)\b(?:,| and| plus)|"
    r"otherwise (?:hold |be in |in |go to |switch to )?(?:cash|[a-z^]{1,5}\b(?! days))|"
    r"hold (?:[a-z^]{1,5}|cash) (?:when|while|if|as long as)\b)", re.I)
SIGNAL_HINT = re.compile(
    r"\b(?:hold(?: it)?(?: for)? \d+ (?:trading )?(?:day|bar|week|session)s?|sell (?:when|if|at|on|after|half)|stop[- ]loss|"
    r"take[- ]profit|trailing stop|profit target|cover (?:at|when|on)|exit (?:when|at|after|on)|"
    r"\bshort\b|days? in a row|\bbuy\b[^,]*\b(?:when|if|after|once)\b)", re.I)


def looks_like_allocation(text: str) -> bool:
    t = _normalize(text)
    # "SPY 60%, TLT 30%, GLD 10%": ticker-first weights (uppercase, known tickers only)
    tw = re.findall(r"(?<![\w^])(\^?[A-Z]{1,5}) \d+(?:\.\d+)?%", t)
    if len(tw) >= 2 and all(data.canonical(x) in _known() for x in tw):
        return True
    if not ALLOC_HINT.search(t):
        return False
    strong = re.search(r"\b(?:rebalanc\w*|buy and hold|equal[- ]weight|inverse[- ]volatility|(?:top|bottom) \d+(?![\d.%]|\s*%)|rotat|dual momentum|otherwise hold|allocat|contribut|withdraw|\d+/\d+)", t, re.I)
    if SIGNAL_HINT.search(t) and not strong:
        return False
    return True


def parse(text: str, **overrides):
    """Parse a sentence into a Strategy (signals) or a Portfolio (allocations)."""
    if not text or not text.strip():
        raise ParseError("Describe a strategy, e.g. 'buy MSFT at the close when it is down 5 days in a row, hold 1 day'.")
    if looks_like_allocation(text):
        obj = parse_allocation(text)
    else:
        obj = parse_signal(text)
    for k, v in overrides.items():
        if v is None:
            continue
        if not hasattr(obj, k):
            if k in ("max_positions", "position_size", "stop_loss", "take_profit", "trailing_stop", "hold_bars",
                     "exit_when", "entry_fill", "hold_exit_fill", "rank_by") and isinstance(obj, Portfolio):
                raise ParseError(f"--{k.replace('_', '-')} applies to signal strategies, not allocation portfolios")
            continue
        setattr(obj, k, v)
    obj.validate()
    return obj


# ----------------------------------------------------------------- signal strategies

ENTRY_VERB = r"^(?:buy|go long|long|purchase|enter(?: long)?|get in|short|sell short|go short|short[- ]sell|enter short)\b"
EXIT_VERB = r"^(?:hold|keep|sell|exit|cover|close (?:the|out|it|positions?)|get out|take profit|stop|use|with|place|then sell|and sell|after|or after|max(?:imum)? hold)\b"
COND_START = r"\b(?:when(?:ever)?|if|once|after|on days when|any day|every time|each time)\b"
TIMING = r"(?:(?:at|on) (?:the )?(?:next |following |tomorrow's )?(?:day's |trading day's )?(?:close|open)|market on (?:close|open)|\bmoc\b|\bmoo\b|at the next open|next day's open|next open)"


def _split_clauses(t: str) -> list[str]:
    t = re.sub(r"\band\s+(?=(?:then\s+)?(?:hold|keep|sell|exit|cover|close (?:the|out|it)|take profit|use a|with a|place a|go short|short|go long|buy)\b)", ", ", t, flags=re.I)
    t = re.sub(r"\bthen\b", ",", t, flags=re.I)
    chunks, buf, in_tick = [], "", False
    for i, ch in enumerate(t):
        if ch == "`":
            in_tick = not in_tick
        nxt = t[i + 1] if i + 1 < len(t) else " "
        if not in_tick and (ch in ",;" or (ch == "." and not (buf and buf[-1].isdigit() and nxt.isdigit()))):
            chunks.append(buf)
            buf = ""
        else:
            buf += ch
    chunks.append(buf)
    return [c.strip() for c in chunks if c.strip()]


def _universe_phrase(text: str) -> tuple[list[str] | None, str | None]:
    low = text.lower()
    if re.search(r"nasdaq[- ]?100|\bndx\b|(?:all|each|any|every) (?:the )?(?:index )?(?:stocks?|members?|components?|constituents?)", low):
        return data.nasdaq100_ever(), "NDX"
    if re.search(r"sector (?:etfs|spdrs|funds)", low):
        return [t for t in SECTOR_ETFS if t in _known()], None
    if re.search(r"all (?:available )?tickers|entire universe|whole universe", low):
        return data.available_tickers(), None
    return None, None


def parse_signal(text: str) -> Strategy:
    raw = text
    t = _normalize(text)
    T = Text(t)
    notes: list[str] = []
    kw = common_options(T, notes)

    # ---- sizing and portfolio options
    m = T.find(r"(?:max(?:imum)?(?: of)?|up to|at most|no more than|hold at most|limit(?:ed)? to) (\d+) (?:open |simultaneous |concurrent )?(?:positions|stocks|names|holdings|trades)(?: at (?:a|any|one) time| at once)?|(\d+) (?:positions|stocks|names) max(?:imum)?")
    if m:
        kw["max_positions"] = int(m.group(1) or m.group(2))
    m = T.find(rf"risk(?:ing)? {NUM}% (?:of (?:the )?(?:equity|capital|account) )?(?:per|on each|each) trade")
    if m:
        kw["sizing"], kw["risk_per_trade"] = "risk", float(m.group(1)) / 100
    m = T.find(rf"(?:put |invest |allocate |using |with |use )?{NUM}% (?:of (?:the )?(?:equity|capital|account|portfolio) )?(?:per|in each|for each|each|on each|to each) (?:trade|position|stock|name|entry)")
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
    if T.find(r"whole shares(?: only)?|no fractional shares"):
        kw["fractional_shares"] = False
    m = T.find(rf"(?:no more than|at most|max(?:imum)?|cap(?:ped)? at|limit(?:ed)? to) {NUM}% of (?:the )?(?:day's |daily |average )?(?:volume|adv)")
    if m:
        kw["max_volume_pct"] = float(m.group(1)) / 100
    m = T.find(rf"{NUM}% (?:annual |yearly )?borrow(?:ing)? (?:fee|cost|rate)|borrow (?:fee|cost|rate) of {NUM}%")
    if m:
        kw["borrow_fee"] = float(m.group(1) or m.group(2)) / 100
    m = T.find(rf"{NUM}% margin (?:interest|rate)|margin (?:interest|rate) of {NUM}%")
    if m:
        kw["margin_rate"] = float(m.group(1) or m.group(2)) / 100
    m = T.find(r"pyramid(?:ing)?(?: up to)? (\d+)(?: times| entries)?|(?:add to (?:the )?(?:position|winners)|scale in)(?: up to)? (\d+) times|up to (\d+) entries per (?:ticker|stock|position)")
    if m:
        kw["pyramiding"] = int(m.group(1) or m.group(2) or m.group(3))
    # ranking
    mr = T.find(r"(?:prefer(?:ring)?|rank(?:ed)? by|pick(?:ing)?|choose|choosing|favou?r(?:ing)?) (?:the )?(lowest|highest|weakest|strongest|biggest losers?|biggest gainers?|most oversold|most overbought|biggest declines?|largest declines?)(?: (rsi|return|decline|change|volatility))?(?: first)?")

    # ---- exits and stops (anywhere)
    ex: dict = {}
    for m in T.findall(rf"(?:sell|exit|close|take profits? on|take) (?:\d+% |{NUM}% )?(?:of (?:the )?(?:position|shares) )?(?:at|when (?:it(?:'s| is)? )?up) \+?{NUM}%(?: (?:gain|profit|up))?"):
        pass  # handled below via scale-out regex
    so = []
    for m in re.finditer(rf"(?:sell|exit|close|take profits? on|take) {NUM}% (?:of (?:the )?(?:position|shares) )?(?:at|when (?:it(?:'s| is)? )?up) \+?{NUM}%", T.low):
        so.append({"fraction": float(m.group(1)) / 100, "at": float(m.group(2)) / 100})
        T.blank(m.group(0))
    if so:
        ex["scale_out"] = so
    m = T.find(rf"(?:with a |use a |place a |and a )?{NUM} ?(?:x )?atr (?:trailing|chandelier) stop|(?:trailing|chandelier) stop(?: loss)?(?: of| at)? {NUM} ?(?:x )?atrs?(?: (?:from|below) the high)?")
    if m:
        ex["trailing_atr"] = float(m.group(1) or m.group(2))
    m = T.find(rf"(?:with a |use a |place a |and a )?{NUM} ?(?:x )?atr stop(?:[- ]loss)?|stop(?:[- ]loss)?(?: of| at)? {NUM} ?(?:x )?atrs?(?: (?:below|from) (?:the )?entry)?")
    if m:
        ex["stop_atr"] = float(m.group(1) or m.group(2))
    m = T.find(rf"(?:take[- ]profits?|profit target|target)(?: of| at)? {NUM} ?(?:x )?atrs?|{NUM} ?(?:x )?atr (?:profit target|take[- ]profit|target)")
    if m:
        ex["take_profit_atr"] = float(m.group(1) or m.group(2))
    m = T.find(rf"(?:with a |use a |and a )?trailing stop(?:[- ]loss)?(?: of| at)? {NUM}%|(?:with a |use a |and a )?{NUM}% trailing stop(?:[- ]loss)?")
    if m:
        ex["trailing_stop"] = float(m.group(1) or m.group(2)) / 100
    m = T.find(rf"(?:with a |use a |place a |and a )?(?:hard )?stop[- ]?(?:loss)?(?: of| at)? {NUM}%(?: below (?:the )?entry)?|(?:with a |use a |place a |and a )?{NUM}% (?:hard )?stop(?:[- ]loss)?")
    if m:
        ex["stop_loss"] = float(m.group(1) or m.group(2)) / 100
    m = T.find(rf"(?:with a |and a |and )?(?:take[- ]profits?|profit target|target)(?: of| at)? \+?{NUM}%|(?:with a |and a )?{NUM}% (?:profit target|take[- ]profit|target|gain target)")
    if m:
        ex["take_profit"] = float(m.group(1) or m.group(2)) / 100
    if re.search(r"\b(?:stop|target|trailing)\b", T.rest, re.I) and re.search(r"\d", T.rest):
        mm = re.search(r"[^;]*\b(?:stop|target|trailing)\b[^;]*", T.rest, re.I)
        raise ParseError(f"Could not understand the stop/target in {mm.group(0).strip()!r}. Write e.g. '5% stop loss', "
                         f"'2 ATR stop', '10% trailing stop', '3 ATR trailing stop', 'take profit at 8%'.")

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
    for side, cl in entries:
        cl_rest = cl
        mcond = re.search(COND_START, cl_rest, flags=re.I)
        subject = cl_rest[: mcond.start()] if mcond else cl_rest
        cond_text = cl_rest[mcond.end():] if mcond else ""
        u, uname = _universe_phrase(subject)
        tick = find_tickers(subject, strict=True)
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
                cond_text = re.sub(rf"(?<![\w])[\$^]?{re.escape(u[0].lstrip('^'))}\b", " it ", cond_text, count=1, flags=re.I)
        else:
            u = list(u) + [x for x in tick if x not in u]
        if universe is not None and set(u) != set(universe):
            raise ParseError("Long and short rules must trade the same ticker(s).")
        universe, uni_name = u, uname or uni_name
        low = cl_rest  # original casing; all matching below is case-insensitive
        # order type
        order, level, valid = "market", None, 1
        mo = re.search(rf"(?:with |using |on |at )?(?:a )?(limit|stop) (?:order|entry)?\s*(?:at |of )?{NUM}% (below|above) (?:the |today's |yesterday's |the previous |the prior |the signal day's )?(close|high|low|open)", low, flags=re.I)
        if mo:
            order = mo.group(1)
            f = float(mo.group(2)) / 100
            level = f"{mo.group(4)} * {1 - f if mo.group(3) == 'below' else 1 + f:.6g}"
            low = low.replace(mo.group(0), " ")
        mo = re.search(r"(?:with |using |on |at )?(?:a )?(limit|stop) (?:order|entry)?\s*(?:at |of )?(?:the |its )?(\d+) day (high|low)", low, flags=re.I)
        if mo and not level:
            order = mo.group(1)
            level = f"{'highest(high' if mo.group(3) == 'high' else 'lowest(low'}, {mo.group(2)})"
            low = low.replace(mo.group(0), " ")
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
                notes.append("Entry timing not stated: assuming a fill at the close of the signal day.")
        # conditions
        mcond = re.search(COND_START, low, flags=re.I)
        cond = low[mcond.end():] if mcond else ""
        cond = re.sub(TIMING, " ", cond, flags=re.I)
        if not mcond:
            rest = re.sub(ENTRY_VERB, " ", low.strip(), flags=re.I)
            rest = re.sub(TIMING, " ", rest, flags=re.I)
            for tk in find_tickers(cl):
                rest = re.sub(rf"(?<![\w])[\$^]?{re.escape(tk.lstrip('^'))}\b", " ", rest, flags=re.I)
            for name in COMPANIES:
                rest = re.sub(rf"\b{re.escape(name)}\b", " ", rest, flags=re.I)
            rest = re.sub(r"nasdaq[- ]?100|\bndx\b|(?:all|each|any|every) (?:the )?(?:index )?(?:stocks?|members?|components?|constituents?)|stocks?|sector (?:etfs|spdrs|funds)", " ", rest, flags=re.I)
            cond = rest
        for tk in universe:
            cond = re.sub(rf"(?<![\w])[\$^]?{re.escape(tk.lstrip('^'))}\b(?!\s*(?:is|closes|trades|'s)\b)", " it ", cond, flags=re.I) if len(universe) == 1 else cond
        if not re.sub(r"[^a-z0-9`]", "", re.sub(r"\b(?:it|the|a|an|and|stock|shares?)\b", "", cond.lower())):
            if order != "market":
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
            notes.append(f"'{cond.strip()}' was read as the reverse of the {'long' if side == 'short' else 'short'} rule: {flip}.")
        else:
            parts = parse_conditions(cond, universe, as_list=True)
        if fill == "open":
            late = [p for p in parts if not _open_safe(p)]
            if late:
                # split backtick blocks too, so open-time terms (e.g. gap) keep today's value
                fine = []
                for p in parts:
                    for term in split_and(p):
                        fine.append(term if _open_safe(term) else f"ref(({term}), 1)")
                parts = fine
                notes.append("Entry at the open: " + " and ".join(late)
                             + " is not known until the close, so it is checked on the previous day's close.")
        parsed[side] = {"entry": " and ".join(parts), "fill": fill, "order": order, "level": level, "valid": valid}

    if len({p["fill"] for p in parsed.values()}) > 1:
        raise ParseError("Long and short entries must use the same fill timing.")
    first = parsed.get("long") or parsed.get("short")

    # ---- exits (clauses)
    hold_bars, hold_fill = None, "close"
    exit_when, exit_when_fill = None, "close"
    for cl in exits:
        low = cl  # original casing; matching is case-insensitive
        mh = (re.search(r"(?:max(?:imum)? hold(?:ing)?(?: period)?(?: of)?|time stop(?: of)?) (\d+) (?:trading )?(day|bar|session|week|month)s?", low, flags=re.I)
              or re.search(r"(?:hold|keep)\w*(?: it| the position| the stock| positions?| the trade| them)?(?: for)?(?: up to| at most| a maximum of| no more than)? (\d+) (?:trading )?(day|bar|session|week|month)s?", low, flags=re.I)
              or re.search(r"(?:sell|exit|cover|close)\w*(?: it| the position| out| them)? (?:after|in) (\d+) (?:trading )?(day|bar|session|week|month)s?", low, flags=re.I)
              or re.search(r"(?:after|or after|or in|max(?:imum)? hold(?:ing)?(?: period)?(?: of)?|time stop(?: of)?|or) (\d+) (?:trading )?(day|bar|session|week|month)s?", low, flags=re.I)
              or re.search(r"(\d+) (?:trading )?(day|bar|session|week|month)s? later", low, flags=re.I))
        if mh:
            hold_bars = _period(mh.group(1), mh.group(2))
            low = low.replace(mh.group(0), " ")  # noqa
        elif re.search(r"(?:sell|exit|cover)\w* (?:at |on )?(?:the )?(?:next|following|tomorrow)", low, flags=re.I) or re.search(r"\bnext day\b", low, flags=re.I):
            hold_bars = 1 if hold_bars is None else hold_bars
        mw = re.search(r"(?:sell|exit|cover|close (?:the position|out|it)|get out)\w*(?: it| the position| out| them)? (?:when|if|once|on|as soon as|at the first|at the close of the first|at the close when|at the open after|the day after)\b(.*)$", low, flags=re.I)
        if mw and mw.group(1).strip():
            wtxt = mw.group(1)
            if re.search(r"next (?:day's )?open|at the open|the day after", wtxt + " " + mw.group(0), re.I):
                exit_when_fill = "next_open"
            wtxt = re.sub(TIMING, " ", wtxt, flags=re.I)
            wtxt = re.sub(r"\b(?:or )?(?:after|in) \d+ (?:trading )?(?:days?|bars?|sessions?|weeks?)\b", " ", wtxt, flags=re.I)
            wtxt = re.sub(r"\b(?:the )?first\b", " ", wtxt, flags=re.I)
            wl = wtxt.strip(" ,;")
            pron = re.fullmatch(r"(?i)(?:it |the price )?(?:crosses|falls|drops|closes|goes|moves|is|trades|gets)?(?: back)? ?(below|under|above|over)(?: (?:it|them|that|the average|the line|again))?", wl)
            if pron:
                flip = _flip(first["entry"], below=pron.group(1) in ("below", "under"))
                if not flip:
                    raise ParseError(f"'{wl}' refers back to the entry rule, but the entry has no comparison to reverse; say what it crosses.")
                exit_when = flip
                notes.append(f"'{wl}' was read as the reverse of the entry comparison: {flip}.")
            elif re.fullmatch(r"(?i)(?:the |its )?(?:signal|condition|setup|entry (?:rule|condition))s? (?:reverses?|ends?|is no longer (?:true|met)|no longer holds?|turns? off)|it(?:'s| is)? no longer (?:true|met)|not ?", wl):
                exit_when = f"not ({first['entry']})"
            elif wl:
                exit_when = parse_conditions(wl, universe)
        for mx in re.finditer(rf"(?:sell|exit|cover|close)\w*[^,;]*?(?:at|on) (?:the )?(?:next |following |tomorrow's )?(?:day's |trading day's )?(open|close)\b", low, flags=re.I):
            if not mw or mx.start() < (mw.start(1) if mw else 0):
                hold_fill = mx.group(1).lower() if hold_bars is not None or not mw else hold_fill
        if hold_bars is None and not mw and not exit_when:
            m1 = re.search(r"(?:sell|exit|cover)\w* (?:it |them )?(?:at|on) (?:the )?(next |following )?(close|open)", low, flags=re.I)
            if m1:
                hold_bars = 1
                hold_fill = m1.group(2).lower()
                notes.append(f"No holding period stated: exiting at the first {hold_fill} after entry.")
    stops_given = any(ex.get(k) for k in ("stop_loss", "trailing_stop", "take_profit", "stop_atr", "trailing_atr", "take_profit_atr", "scale_out"))
    if not any([hold_bars, exit_when, stops_given, len(parsed) == 2]):
        raise ParseError("No exit rule found. Say e.g. 'hold 1 day and sell at the close', 'sell when it closes above its 5-day moving average', or 'with a 5% stop loss'.")

    # leftover check over the whole sentence (entry conditions were parsed strictly already)
    for cl in exits:
        chk = cl.lower()
        chk = re.sub(r"(?:sell|exit|cover|close (?:the position|out|it)|get out)\w*(?: it| the position| out| them)? (?:when|if|once|on|as soon as|at the first|at the close of the first|at the close when|at the open after|the day after)\b.*$", " ", chk)
        chk = re.sub(r"(?:max(?:imum)? hold(?:ing)?(?: period)?(?: of)?|time stop(?: of)?) \d+ (?:trading )?(?:day|bar|session|week|month)s?", " ", chk)
        chk = re.sub(r"\d+ (?:trading )?(?:day|bar|session|week|month)s? later", " ", chk)
        chk = re.sub(r"(?:hold|keep)\w*(?: it| the position| the stock| positions?| the trade| them)?(?: for)?(?: up to| at most| a maximum of| no more than)? \d+ (?:trading )?(?:day|bar|session|week|month)s?", " ", chk)
        chk = re.sub(r"(?:after|or after|or in|in|or|max(?:imum)? hold(?:ing)?(?: period)?(?: of)?|time stop(?: of)?) \d+ (?:trading )?(?:day|bar|session|week|month)s?(?: later)?", " ", chk)
        chk = re.sub(TIMING + r"|\bnext day\b|\bnext\b|\bfollowing\b", " ", chk)
        left = [w for w in re.findall(r"[a-z%']+|\d+(?:\.\d+)?", chk) if w not in STOP]
        if left:
            raise ParseError(f"Could not interpret {' '.join(left)!r} in the exit {cl.strip()!r}.")

    side = "both" if len(parsed) == 2 else ("short" if "short" in parsed else "long")
    if mr:
        w = mr.group(1)
        if mr.group(2) == "rsi" or "oversold" in w or "overbought" in w:
            mn = re.search(r"rsi\([^,]+,\s*(\d+)\)", first["entry"])
            kw["rank_by"] = f"rsi({mn.group(1) if mn else 2})"
            kw["rank_ascending"] = w in ("lowest", "most oversold")
        elif mr.group(2) == "volatility":
            kw["rank_by"], kw["rank_ascending"] = "volatility(20)", w in ("lowest", "weakest")
        else:
            kw["rank_by"] = "ret(5)" if "return" in (mr.group(2) or "") else "change"
            kw["rank_ascending"] = w in ("lowest", "weakest", "biggest loser", "biggest losers", "biggest decline", "biggest declines", "largest decline", "largest declines")
    if len(universe) > 1 and "max_positions" not in kw:
        kw["max_positions"] = 10
        if "position_size" not in kw and kw.get("sizing", "percent") == "percent":
            notes.append("Multiple tickers and no position limit given: using up to 10 positions at 10% of equity each.")
        else:
            notes.append("Multiple tickers and no position limit given: using up to 10 positions.")
    if uni_name == "NDX":
        pit = kw.get("point_in_time", True)
        notes.append("Universe: Nasdaq-100 " + ("with point-in-time membership (stocks are only bought while in the index; "
                     "former members are included where price history exists)." if pit else
                     "using TODAY'S members only - results carry survivorship bias."))
    benchmark = kw.pop("benchmark", None)

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
    strat.notes = list(dict.fromkeys(strat.notes))
    if re.search(r"\b(?:down|up)_(?:days|streak)\b\s*>=|_streak\([^)]*\) >=", strat.entry):
        notes.append("'N days in a row' also fires on later days of a longer streak (6th, 7th...); say 'exactly N days' to fire only on the Nth.")
    return strat


def _open_safe(rule: str) -> bool:
    from .expr import open_safe
    return open_safe(rule)


# ----------------------------------------------------------------- allocation portfolios

FREQ_WORDS = {"daily": "daily", "day": "daily", "weekly": "weekly", "week": "weekly", "monthly": "monthly",
              "month": "monthly", "quarterly": "quarterly", "quarter": "quarterly", "annually": "yearly",
              "yearly": "yearly", "year": "yearly", "annual": "yearly"}


def _metric(text: str) -> tuple[str, str]:
    """('ret(63)', 'top') from phrases like '3 month momentum', 'lowest volatility', 'RSI(10)'."""
    s = text.strip().lower()
    direction = "top"
    if re.search(r"\b(lowest|least|weakest|worst|smallest)\b", s):
        direction = "bottom"
    m = re.search(r"(\d+) (day|week|month|year) (?:total )?(?:momentum|return|returns|performance|change|gain)", s) or \
        re.search(r"(?:momentum|return|returns|performance) over (?:the (?:last|past) )?(\d+) (day|week|month|year)s?", s)
    if m:
        return f"ret({_period(m.group(1), m.group(2))})", direction
    if re.search(r"\b(?:momentum|return|performance)\b", s):
        return "ret(252)", direction
    m = re.search(r"(?:(\d+) (?:day|period) )?rsi(?:\s*\(?(\d+)\)?)?", s)
    if m:
        return f"rsi({m.group(1) or m.group(2) or 14})", direction
    m = re.search(r"(?:(\d+) (day|week|month) )?(?:volatility|vol|standard deviation)", s)
    if m:
        return f"volatility({_period(m.group(1) or 20, m.group(2))})", direction
    m = re.search(r"(?:(\d+) (day|week|month) )?(?:drawdown)", s)
    if m:
        return f"drawdown({_period(m.group(1) or 63, m.group(2))})", direction
    m = re.search(r"distance (?:from|above) (?:its |the )?(\d+) (day|week|month) (?:moving average|ma|sma)", s)
    if m:
        return f"close / sma({_period(m.group(1), m.group(2))}) - 1", direction
    m = re.search(r"`([^`]+)`", text)
    if m:
        return m.group(1), direction
    raise ParseError(f"Unknown ranking metric {text.strip()!r}. Use e.g. '3 month momentum', '10 day RSI', "
                     f"'60 day volatility', or a rule in backticks.")


def _weighting(s: str) -> str | None:
    if re.search(r"inverse[- ]vol(?:atility)?|risk[- ]parity", s):
        return "inverse_vol"
    if re.search(r"market[- ]cap", s):
        return "market_cap"
    if re.search(r"equal(?:ly)?[- ]?weight|equally", s):
        return "equal"
    return None


def _asset_list(s: str) -> list[dict]:
    """'QQQ, SPY and cash' -> [{'asset': 'QQQ'}, {'asset': 'SPY'}, {'cash': True}]"""
    out = []
    items = re.split(r",|\band\b|\bor\b|/|\bplus\b|&", s)
    for it in items:
        it = it.strip()
        if not it:
            continue
        if re.fullmatch(r"(?:hold |in )?(?:cash|t-?bills? as cash|money market)", it):
            out.append({"cash": True})
            continue
        probe = it.upper() if re.fullmatch(r"[a-z^$]{1,6}", it) else it
        tk = find_tickers(probe, strict=True)
        if len(tk) != 1:
            raise ParseError(f"Could not identify a single ticker in {it!r}")
        rest = re.sub(rf"(?<![\w])[\$^]?{re.escape(tk[0].lstrip('^'))}\b", " ", probe, flags=re.I)
        for name, sym in COMPANIES.items():
            if sym == tk[0]:
                rest = re.sub(rf"\b{re.escape(name)}\b", " ", rest, flags=re.I)
        left = [w for w in re.findall(r"[a-z%']+|\d+", rest.lower()) if w not in STOP]
        if left:
            raise ParseError(f"Could not interpret {' '.join(left)!r} in {it!r}")
        out.append({"asset": tk[0]})
    return out


def _node(text: str) -> dict:
    """Parse an allocation phrase into a portfolio tree node."""
    s = text.strip().strip(",;. ")
    s = re.sub(r"^(?:and |then )?(?:hold|buy and hold|buy|own|be in|invest(?: in)?|allocate(?: to)?|put (?:everything |it all |all )?in(?:to)?|go (?:to|into)|switch (?:to|into)|rotate (?:to|into)|stay in|move (?:to|into)|in)\s+", "", s)
    low = s.lower()
    if re.fullmatch(r"(?:cash|t-?bills|treasury bills|money market|nothing|flat)", low):
        return {"cash": True}

    # if COND[,] [then] hold X[,] (else if ...)* otherwise Y
    m = re.match(r"if (.+?),? (?:then )?(?:hold |buy |own |be in |invest in |go (?:to|into) |switch to |rotate (?:to|into) |stay in |be )(.+?)[,;]? (?:and )?(?:otherwise|else)[, ]+(?:hold |buy |own |be in |go to |switch to |in )?(.+)$", s, flags=re.I)
    if m:
        cond, then, other = m.group(1), m.group(2), m.group(3)
        on, rule = _condition_on(cond, None)
        return {"if": rule, "on": on, "then": _node(then), "else": _node(other)}
    # hold X when/while COND, otherwise Y
    m = re.match(r"(.+?) (?:when|while|if|as long as|whenever) (.+?),? (?:and )?(?:otherwise|else|or else)[, ]+(?:hold |buy |be in |go to |switch to |in )?(.+)$", s, flags=re.I)
    if m and not re.search(r"\b(?:top|bottom|best|worst)\b", m.group(1)):
        then_node = _node(m.group(1))
        ref = then_node.get("asset")
        on, rule = _condition_on(m.group(2), ref)
        return {"if": rule, "on": on, "then": then_node, "else": _node(m.group(3))}

    # dual momentum
    m = re.match(r"dual momentum (?:between|among|of|on|with) (.+?)(?:,? (?:with|using) (.+?) as (?:the )?(?:safe|defensive|risk[- ]off) (?:asset|haven)| (?:otherwise|else|or) (.+?))?(?:,? using (?:a )?(\d+) (month|day|week) (?:lookback|momentum))?$", low)
    if m:
        kids = _asset_list(m.group(1))
        safe = m.group(2) or m.group(3) or "AGG"
        n = _period(m.group(4) or 12, m.group(5) or "month")
        return {"filter": {"select": "top", "n": 1, "by": f"ret({n})",
                           "require": f"ret({n}) > ret(sym('BIL').close, {n})"},
                "universe": "children", "children": kids, "fallback": _node(safe)}

    # top/bottom N of UNIVERSE by METRIC
    m = re.match(r"(?:the )?(top|bottom|best|worst|strongest|weakest|highest|lowest) (\d+) (?:of |among |from |in )?(?:the )?(.+?) (?:by|ranked by|based on|sorted by|with the (?:highest|lowest|best|strongest|weakest)) (.+)$", low)
    if m:
        word, n, uni, rest = m.groups()
        wt = _weighting(rest) or "equal"
        rest = re.sub(r",?\s*(?:(?:and )?(?:weighted )?(?:equal(?:ly)?[- ]?weight(?:ed)?|inverse[- ]vol(?:atility)?(?:[- ]weight(?:ed)?)?|market[- ]cap(?:[- ]weight(?:ed)?)?|risk[- ]parity)(?: weighting)?)", "", rest)
        req = None
        fb = None
        mreq = re.search(r",? (?:but )?only (?:if|when) (?:their|its|the) (.+?)(?: is| are)? positive(?:,? (?:otherwise|else) (.+))?$", rest)
        if mreq:
            rest = rest[: mreq.start()]
            metric_req, _ = _metric(mreq.group(1))
            req = f"{metric_req} > 0"
            fb = _node(mreq.group(2)) if mreq.group(2) else {"cash": True}
        mfb = re.search(r",? (?:that )?(?:beat|beats|outperform|outperforms) (?:cash|t-?bills|bil)(?:,? (?:otherwise|else) (.+))?$", rest)
        if mfb:
            rest = rest[: mfb.start()]
            metric_req, _ = _metric(rest)
            nlook = re.search(r"\((\d+)\)", metric_req)
            req = f"{metric_req} > ret(sym('BIL').close, {nlook.group(1) if nlook else 252})"
            fb = _node(mfb.group(1)) if mfb.group(1) else {"cash": True}
        metric, direction = _metric(rest)
        if word in ("bottom", "worst", "weakest", "lowest"):
            direction = "bottom" if direction == "top" else "top"
        uu, uname = _universe_phrase(uni)
        node: dict = {"filter": {"select": direction, "n": int(n), "by": metric, "weights": wt}}
        if uname == "NDX":
            node["universe"] = "NDX"
        elif uu is not None:
            node["universe"] = uu
        else:
            kids = _asset_list(uni)
            node["universe"] = "children"
            node["children"] = kids
        if req:
            node["filter"]["require"] = req
            node["fallback"] = fb
        return node

    # rotate between A, B and C by METRIC  (top 1)
    m = re.match(r"(?:rotate|rotation|switch)(?: \w+)? (?:between|among|across) (.+?) (?:by|based on|using|according to) (.+)$", low)
    if m:
        metric, direction = _metric(m.group(2))
        return {"filter": {"select": direction, "n": 1, "by": metric, "weights": "equal"},
                "universe": "children", "children": _asset_list(m.group(1))}
    m = re.match(r"whichever of (.+?) has (?:the )?(higher|highest|lower|lowest|best|stronger|strongest|weaker|weakest) (.+)$", low)
    if m:
        metric, _ = _metric(m.group(3))
        direction = "top" if m.group(2) in ("higher", "highest", "best", "stronger", "strongest") else "bottom"
        return {"filter": {"select": direction, "n": 1, "by": metric, "weights": "equal"},
                "universe": "children", "children": _asset_list(m.group(1))}

    # 60/40 SPY/TLT  or  SPY/TLT 60/40
    m = re.fullmatch(r"(\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)+) ([a-z^$ -]+(?:/[a-z^$ -]+)+)|([a-z^$ -]+(?:/[a-z^$ -]+)+) (\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)+)", low)
    if m:
        ws = [float(x) for x in (m.group(1) or m.group(4)).split("/")]
        names = (m.group(2) or m.group(3)).split("/")
        if len(ws) != len(names):
            raise ParseError(f"{len(ws)} weights but {len(names)} holdings in {s!r}")
        kids = [_node(x) for x in names]
        return _weights_node([w / sum(ws) for w in ws], kids, s)

    # 60% SPY, 30% TLT and 10% GLD   /   SPY 60%, TLT 40%
    pairs = re.findall(r"(\d+(?:\.\d+)?)% (?:in |of |into )?(?:the )?([^,%]+?)(?=,|\band\b|\bplus\b|$| \d)", low)
    if not pairs:
        pairs = [(w, n) for n, w in re.findall(r"([a-z^$][^,%]*?) (\d+(?:\.\d+)?)%", low)]
    if pairs:
        ws = [float(w) / 100 for w, _ in pairs]
        kids = [_node(n) for _, n in pairs]
        if abs(sum(ws) - 1) > 1e-6:
            if sum(ws) < 1 - 1e-6 and not any(k.get("cash") for k in kids):
                kids.append({"cash": True})
                ws.append(1 - sum(ws))
            else:
                raise ParseError(f"Weights add up to {sum(ws):.0%}, not 100%.")
        return _weights_node(ws, kids, s)

    # equal weight / inverse volatility of a list
    wt = _weighting(low)
    if wt:
        lst = re.sub(r"(?:weighted |weight(?:ed)? |weighting )?(?:by )?(?:equal(?:ly)?[- ]?weight(?:ed)?|equally(?: in)?|inverse[- ]vol(?:atility)?(?:[- ]weight(?:ed)?)?|market[- ]cap(?:[- ]weight(?:ed)?)?|risk[- ]parity)(?: weighting)?(?: of| in| across| between| among)?", " ", low)
        mlb = re.search(r"(?:using |with |over )?(?:a )?(\d+) (day|week|month) (?:lookback|volatility|window)", lst)
        lookback = 20
        if mlb:
            lookback = _period(mlb.group(1), mlb.group(2))
            lst = lst.replace(mlb.group(0), " ")
        uu, uname = _universe_phrase(lst)
        if uu is not None:
            raise ParseError("Weighting a whole index needs a selection: say e.g. 'top 20 Nasdaq 100 stocks by 12 month momentum, equal weight'.")
        kids = _asset_list(lst.strip(" ,"))
        node = {"weights": wt, "children": kids}
        if wt == "inverse_vol":
            node["lookback"] = lookback
        return node

    # a single asset, or a plain list (equal weight)
    items = _asset_list(s)
    if len(items) == 1:
        return items[0]
    return {"weights": "equal", "children": items}


def _weights_node(ws: list[float], kids: list[dict], s: str) -> dict:
    if any(w <= 0 for w in ws):
        raise ParseError(f"Weights must be positive in {s!r}")
    return {"weights": "specified", "w": [round(w, 10) for w in ws], "children": kids}


def _condition_on(cond: str, default: str | None) -> tuple[str, str]:
    """Condition text -> (ticker the rule is evaluated on, rule)."""
    tk = find_tickers(cond, strict=True)
    on = tk[0] if tk else default
    if on is None:
        raise ParseError(f"Which ticker does {cond.strip()!r} refer to? e.g. 'if SPY is above its 200-day moving average'.")
    rule = parse_conditions(cond, [on])
    return on, rule


def parse_allocation(text: str) -> Portfolio:
    raw = text
    t = _normalize(text)
    T = Text(t)
    notes: list[str] = []
    kw = common_options(T, notes)
    benchmark = kw.pop("benchmark", None)
    for k in ("slippage_bps", "commission", "commission_pct"):
        pass
    if "commission_per_share" in kw:
        raise ParseError("Per-share commissions are not supported for allocation portfolios; use '$1 per trade' or '0.1% commission'.")
    pk: dict = {k: v for k, v in kw.items() if k in ("capital", "slippage_bps", "commission", "commission_pct", "start", "end", "cash_rate", "point_in_time")}

    # rebalancing
    rb = None
    m = T.find(r",? ?(?:and )?(?:re-?balanc\w*|reset|rotat\w*|re-?evaluat\w*|check\w*)(?: (?:it|the weights|the portfolio|them))?(?: back)?(?: to (?:target|the target weights))? (?:every|each|once (?:a|per)) (day|week|month|quarter|year)|,? ?(?:and )?re-?balanc\w*(?: (?:it|the weights|the portfolio))? (daily|weekly|monthly|quarterly|annually|yearly)|,? ?(daily|weekly|monthly|quarterly|annual|yearly) re-?balanc\w*")
    if m:
        rb = FREQ_WORDS[(m.group(1) or m.group(2) or m.group(3)).lower()]
    if T.find(r",? ?(?:and )?(?:never re-?balanc\w*|no re-?balancing|without re-?balancing|don't re-?balance|do not re-?balance)"):
        rb = "none"
    m = T.find(rf",? ?(?:and )?(?:re-?balanc\w* (?:only )?(?:when|if) (?:any |a )?(?:weight|holding|position|allocation)s? (?:drifts?|moves?|deviates?|is off) (?:by )?(?:more than )?{NUM}%(?: (?:from|away from) (?:its )?target)?|(?:with )?(?:a )?{NUM}% (?:re-?balancing |drift |tolerance )?band)")
    band = float(m.group(1) or m.group(2)) / 100 if m else None
    fill = "close"
    if T.find(r",? ?(?:trade|trading|rebalanc\w*|execute\w*)? ?(?:at|on) the next (?:day's )?open"):
        fill = "next_open"
    T.find(r",? ?(?:trade|trading|rebalanc\w*|execute\w*)? ?(?:at|on) the close")
    # cash flows
    contrib = 0.0
    cfreq = "monthly"
    m = T.find(r",? ?(?:and )?(?:add(?:ing)?|invest(?:ing)?|contribut\w+|deposit\w*|put(?:ting)? in)(?: an additional| another| a further)? \$(\d+(?:\.\d+)?)(?: more)? (?:every|each|per|a|an|once a) (month|quarter|year)|,? ?(?:with )?(?:\$(\d+(?:\.\d+)?) )?(monthly|quarterly|yearly|annual) contributions?(?: of \$(\d+(?:\.\d+)?))?")
    if m:
        amt = m.group(1) or m.group(3) or m.group(5)
        if amt is None:
            raise ParseError("How much is contributed? e.g. 'add $500 every month'.")
        contrib = float(amt)
        cfreq = FREQ_WORDS[(m.group(2) or m.group(4)).lower()]
    wd, wd_pct, wfreq = 0.0, 0.0, "yearly"
    m = T.find(rf",? ?(?:and )?(?:withdraw\w*|take out|spend\w*|draw\w*(?: down)?)(?: of)? (?:\$(\d+(?:\.\d+)?)|{NUM}%(?: of the (?:balance|portfolio))?) (?:every|each|per|a|an|once a) (month|quarter|year)|,? ?(?:with )?(?:a )?{NUM}% (?:annual |yearly )?(?:withdrawal|spending) rate")
    if m:
        if m.group(1):
            wd = float(m.group(1))
        elif m.group(2):
            wd_pct = float(m.group(2)) / 100
        else:
            wd_pct = float(m.group(4)) / 100
        wfreq = FREQ_WORDS[(m.group(3) or "year").lower()]
    infl = bool(T.find(r",? ?(?:\(?(?:adjusted|indexed|rising|growing|increased) (?:for|with|by) inflation\)?|inflation[- ](?:adjusted|indexed)|in real terms)"))
    if T.find(r",? ?(?:do not|don't|without) reinvest(?:ing)? dividends|dividends (?:paid out|kept) (?:as|in) cash"):
        kw["reinvest_dividends"] = False
    T.find(r",? ?(?:with )?dividends reinvested|reinvest(?:ing)? dividends")

    mrot = re.search(r"\b(?:rotate|switch)\w* (daily|weekly|monthly|quarterly|annually|yearly)\b", T.rest, re.I)
    if mrot and rb is None:
        rb = FREQ_WORDS[mrot.group(1).lower()]
        T.rest = T.rest[: mrot.start(1)] + T.rest[mrot.end(1):]
    body = T.rest
    body = re.sub(r"(?i)^\s*(?:backtest|test|simulate|run)?\s*(?:a |the )?(?:portfolio|strategy)?(?: (?:of|that|which))?\s*:?\s*", "", body)
    body = re.sub(r"\s*;\s*", " ", body).strip(" ,.;")
    body = re.sub(r" {2,}", " ", body)
    body = re.sub(r"\s+,", ",", body)
    buy_hold = bool(re.match(r"^buy[- ]and[- ]hold\b", body))
    body = re.sub(r"^buy[- ]and[- ]hold\s+", "", body)
    if not body:
        raise ParseError("What should the portfolio hold? e.g. 'hold 60% SPY and 40% TLT, rebalance quarterly'.")
    tree = _node(body)
    if rb is None:
        if buy_hold:
            rb = "none"
        elif _has(tree, "if") or _has(tree, "filter"):
            rb = "monthly"
            notes.append("Rebalance frequency not stated: re-evaluating the rules and rebalancing monthly (month-end close).")
        elif "weights" in tree:
            if band:
                rb = "none"  # threshold-only rebalancing
            else:
                rb = "monthly"
                notes.append("Rebalance frequency not stated: rebalancing monthly.")
        else:
            rb = "none"
    if buy_hold and rb != "none":
        notes.append("'Buy and hold' with a rebalance schedule: the schedule wins.")
    if wd_pct and infl:
        # "withdraw 4% a year adjusted for inflation" is the classic 4% rule: 4% of the starting
        # balance, then that dollar amount rising with CPI
        wd, wd_pct = wd_pct * pk.get("capital", 10_000.0), 0.0
        notes.append(f"Read as the '4% rule': withdraw ${wd:,.0f} in the first year (that % of the starting balance), "
                     "then the same amount grown with inflation. Say 'withdraw 4% of the balance each year' for a percentage of the current balance.")
    p = Portfolio(tree=tree, rebalance=rb, drift_band=band, fill=fill, contribution=contrib, contribution_freq=cfreq,
                  withdrawal=wd, withdrawal_pct=wd_pct, withdrawal_freq=wfreq, inflation_adjust=infl,
                  description=raw, notes=notes, **pk,
                  **({"reinvest_dividends": kw["reinvest_dividends"]} if "reinvest_dividends" in kw else {}))
    p.benchmark = benchmark
    if _has(tree, "filter", universe="NDX") and p.point_in_time:
        notes.append("Universe: Nasdaq-100 with point-in-time membership (stocks only selected while in the index; "
                     "former members included where price history exists).")
    return p


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
