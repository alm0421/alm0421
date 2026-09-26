"""Translate a plain-English strategy description into a Strategy.

The parser is deterministic and pattern based: it recognises a vocabulary of
common trading phrases (see README), echoes back exactly how it interpreted
them, and refuses - rather than guesses - when part of the entry condition is
not understood.  Anything it cannot express can be written directly in the
rule language inside backticks, e.g.

    buy AAPL at the close when `rsi(3) < 15 and close > sma(close, 100)`, hold 3 days
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from . import data
from .strategy import Strategy

NUM = r"(\d+(?:\.\d+)?)"

WORD_NUMS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "twenty": 20, "thirty": 30, "fifty": 50, "hundred": 100,
}

COMPANIES = {
    "microsoft": "MSFT", "apple": "AAPL", "amazon": "AMZN", "alphabet": "GOOGL", "google": "GOOGL",
    "meta": "META", "facebook": "META", "nvidia": "NVDA", "tesla": "TSLA", "netflix": "NFLX",
    "broadcom": "AVGO", "costco": "COST", "adobe": "ADBE", "intel": "INTC", "cisco": "CSCO",
    "pepsico": "PEP", "pepsi": "PEP", "qualcomm": "QCOM", "texas instruments": "TXN",
    "starbucks": "SBUX", "paypal": "PYPL", "booking": "BKNG", "intuit": "INTU", "amgen": "AMGN",
    "gilead": "GILD", "comcast": "CMCSA", "t-mobile": "TMUS", "honeywell": "HON", "micron": "MU",
    "applied materials": "AMAT", "lam research": "LRCX", "kla": "KLAC", "marvell": "MRVL",
    "palantir": "PLTR", "crowdstrike": "CRWD", "palo alto": "PANW", "fortinet": "FTNT",
    "mercadolibre": "MELI", "airbnb": "ABNB", "doordash": "DASH", "datadog": "DDOG",
    "zscaler": "ZS", "workday": "WDAY", "autodesk": "ADSK", "synopsys": "SNPS", "cadence": "CDNS",
    "regeneron": "REGN", "vertex": "VRTX", "biogen": "BIIB", "intuitive surgical": "ISRG",
    "lululemon": "LULU", "marriott": "MAR", "monster": "MNST", "mondelez": "MDLZ",
    "kraft heinz": "KHC", "ross stores": "ROST", "o'reilly": "ORLY", "paccar": "PCAR",
    "paychex": "PAYX", "csx": "CSX", "cintas": "CTAS", "fastenal": "FAST", "copart": "CPRT",
    "electronic arts": "EA", "take-two": "TTWO", "warner bros": "WBD", "microstrategy": "MSTR",
    "applovin": "APP", "shopify": "SHOP", "trade desk": "TTD", "analog devices": "ADI",
    "astrazeneca": "AZN", "asml": "ASML", "amd": "AMD", "advanced micro devices": "AMD",
    "adp": "ADP", "idexx": "IDXX", "dexcom": "DXCM", "moderna": "MRNA", "linde": "LIN",
    "constellation energy": "CEG", "exelon": "EXC", "xcel": "XEL", "charter": "CHTR",
    "s&p 500": "SPY", "s&p500": "SPY", "s&p": "SPY", "nasdaq 100 etf": "QQQ", "qqq": "QQQ", "spy": "SPY",
}

NOT_TICKERS = {
    "I", "A", "RSI", "SMA", "EMA", "ATR", "IBS", "MA", "AND", "OR", "THE", "US", "USD", "MOC",
    "MOO", "EOD", "PM", "AM", "NDX", "IF", "AT", "SL", "TP", "ETF", "IT", "BUY", "SELL", "ON",
    "ALL", "MY", "DO", "BE", "GO", "SO", "UP", "NOW", "OUT", "NEW", "HIGH", "LOW", "OPEN", "CLOSE",
}

DOW = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4}


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


# ----------------------------------------------------------------- helpers

def _normalize(text: str) -> str:
    t = text.replace("’", "'").replace("–", "-").replace("—", "-")
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"(\d)\s*percent\b", r"\1%", t, flags=re.I)
    t = re.sub(r"\bper cent\b", "%", t, flags=re.I)
    for w, n in WORD_NUMS.items():
        t = re.sub(rf"\b{w}\b", str(n), t, flags=re.I)
    t = re.sub(r"\b(\d+)-(day|week|month|year|period|bar)\b", r"\1 \2", t, flags=re.I)
    t = re.sub(r"\bovernight\b", "1 day", t, flags=re.I)
    return t


def _universe_all() -> set[str]:
    return set(data.available_tickers()) | set(data.nasdaq100()) | set(data.BENCHMARKS)


def find_tickers(text: str) -> list[str]:
    """Tickers mentioned in `text` (original casing), in order of appearance."""
    known = _universe_all()
    found: list[tuple[int, str]] = []
    for m in re.finditer(r"\$?\b([A-Z]{1,5})\b", text):
        sym = m.group(1)
        if sym in NOT_TICKERS and not m.group(0).startswith("$"):
            continue
        if sym in known or m.group(0).startswith("$"):
            found.append((m.start(), sym))
    low = text.lower()
    for name, sym in sorted(COMPANIES.items(), key=lambda kv: -len(kv[0])):
        for m in re.finditer(rf"\b{re.escape(name)}\b", low):
            if not any(abs(m.start() - p) < 2 for p, _ in found):
                found.append((m.start(), sym))
    out: list[str] = []
    for _, s in sorted(found):
        if s not in out:
            out.append(s)
    return out


def _universe_phrase(text: str) -> list[str] | None:
    low = text.lower()
    if re.search(r"nasdaq[- ]?100|\bndx\b|nasdaq 100 stocks|all (?:the )?stocks|each stock|any stock|every stock", low):
        return data.nasdaq100()
    if re.search(r"all (?:available )?tickers|entire universe|whole universe", low):
        return data.available_tickers()
    return None


# ----------------------------------------------------------------- conditions

def _cmp(word: str) -> str:
    w = word.strip().lower()
    if w in ("below", "under", "less than", "<", "lower than", "beneath", "at most", "<="):
        return "<"
    if w in ("above", "over", "greater than", ">", "higher than", "more than", "at least", ">="):
        return ">"
    raise ParseError(f"unknown comparison {word!r}")


CMPW = r"(below|under|less than|lower than|beneath|<=?|above|over|greater than|higher than|more than|>=?)"
MA = r"(simple |exponential )?(?:moving average|moving avg|ma|sma|ema)\b"


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


def _ma(kind: str | None, x: str, n: int, raw: str) -> str:
    ema = (kind and "exp" in kind) or re.search(r"\bema\b", raw)
    return f"{'ema' if ema else 'sma'}({x}, {n})"


def parse_condition(text: str, ctx: Ctx) -> tuple[str | None, str]:
    """Return (expression, leftover) for one sub-clause."""
    s = text.strip().lower()
    parts: list[str] = []

    def take(pattern: str, fn) -> None:
        nonlocal s
        while True:
            m = re.search(pattern, s)
            if not m:
                return
            parts.append(fn(m))
            s = (s[: m.start()] + " " + s[m.end():]).strip()

    c, o, h, l, v = ctx.c, ctx.o, ctx.h, ctx.l, ctx.v
    dn_ = r"(?:down|lower|declin\w*|fall\w*|fell|drop\w*|red|los\w*|negative)(?: for)?"
    up_ = r"(?:up|higher|ris\w*|rose|gain\w*|green|advanc\w*|positive)(?: for)?"
    down_x = "down_days" if c == "close" else f"down_streak({c})"
    up_x = "up_days" if c == "close" else f"up_streak({c})"

    # raw rule language in backticks
    take(r"`([^`]+)`", lambda m: f"({m.group(1)})")

    # consecutive down / up closes ("exactly N" fires only on the Nth day)
    take(rf"{dn_} exactly {NUM} (?:straight |consecutive )?(?:days|closes|sessions|bars)(?: in a row| straight)?|exactly {NUM} (?:consecutive|straight) {dn_} (?:days|closes|sessions|bars)",
         lambda m: f"{down_x} == {int(float(m.group(1) or m.group(2)))}")
    take(rf"{up_} exactly {NUM} (?:straight |consecutive )?(?:days|closes|sessions|bars)(?: in a row| straight)?|exactly {NUM} (?:consecutive|straight) {up_} (?:days|closes|sessions|bars)",
         lambda m: f"{up_x} == {int(float(m.group(1) or m.group(2)))}")
    dn = r"(?:down|lower|declin\w*|fall\w*|fell|drop\w*|red|los\w*|negative)"
    up = r"(?:up|higher|ris\w*|rose|gain\w*|green|advanc\w*|positive)"
    take(rf"(?:at least )?{NUM} (?:or more )?(?:consecutive|straight|successive) {dn} (?:days|closes|sessions|bars)",
         lambda m: f"{down_x} >= {int(float(m.group(1)))}")
    take(rf"(?:at least )?{NUM} (?:or more )?(?:consecutive|straight|successive) {up} (?:days|closes|sessions|bars)",
         lambda m: f"{up_x} >= {int(float(m.group(1)))}")
    take(rf"{dn}(?: for)?(?: at least)? {NUM} (?:or more )?(?:straight |consecutive |trading )?(?:days|closes|sessions|bars|times)(?: in a row| straight| consecutively)",
         lambda m: f"{down_x} >= {int(float(m.group(1)))}")
    take(rf"{up}(?: for)?(?: at least)? {NUM} (?:or more )?(?:straight |consecutive |trading )?(?:days|closes|sessions|bars|times)(?: in a row| straight| consecutively)",
         lambda m: f"{up_x} >= {int(float(m.group(1)))}")

    # moving-average relationships between two averages
    def ma_vs_ma(m):
        a = _ma(m.group(2), c, _period(m.group(1), None), m.group(0))
        b = _ma(m.group(6), c, _period(m.group(5), None), m.group(0))
        rel = (m.group(3) or "") + m.group(4)
        if "cross" in rel:
            return f"{'crossover' if 'above' in rel or 'over' in rel else 'crossunder'}({a}, {b})"
        return f"{a} {_cmp(m.group(4))} {b}"
    take(rf"(?:the )?{NUM} day {MA} (?:is )?(crosses |cross |crossed )?(above|over|below|under) (?:the |its )?{NUM} day {MA}", ma_vs_ma)
    take(r"golden cross", lambda m: f"crossover(sma({c}, 50), sma({c}, 200))")
    take(r"death cross", lambda m: f"crossunder(sma({c}, 50), sma({c}, 200))")

    # price vs moving average, optionally by X%
    def px_vs_ma(m):
        pct, rel, n, unit, kind = m.group(1), m.group(3), m.group(4), m.group(5), m.group(6)
        ma = _ma(kind, c, _period(n, unit), m.group(0))
        cross = m.group(2)
        op = _cmp(rel)
        if cross:
            return f"{'crossover' if op == '>' else 'crossunder'}({c}, {ma})"
        if pct:
            f = float(pct) / 100
            return f"{c} {op}= {ma} * {1 - f if op == '<' else 1 + f:.6g}"
        return f"{c} {op} {ma}"
    take(rf"(?:{NUM}%(?: or more)? )?(?:(?:closes?|trades?|is|price is|stays?|remains?) )?(cross(?:es|ed)? )?{CMPW} (?:its |the )?{NUM} (day|week|month|period|bar)s? {MA}",
         px_vs_ma)

    # RSI
    def rsi(m):
        n = m.group(1) or m.group(2) or m.group(3) or "14"
        return f"rsi({c}, {int(float(n))}) {_cmp(m.group(4))} {m.group(5)}"
    take(rf"(?:(\d+) (?:day|period|bar) )?rsi\s*(?:\(\s*(\d+)\s*\)|(\d+))?(?: value)? (?:is |closes |drops |falls |rises |goes )?{CMPW} {NUM}", rsi)

    # N-day highs / lows
    def hilo(m):
        n = 252 if m.group(1) == "52" and m.group(2).startswith("week") else _period(m.group(1), m.group(2))
        if m.group(3) == "high":
            return f"{c} >= highest({c}, {n})"
        return f"{c} <= lowest({c}, {n})"
    take(rf"(?:new |fresh )?(\d+) (day|week|month|year|bar|session)s? (high|low)(?:est close)?", hilo)
    take(r"(?:lowest|highest) close (?:in|of) (?:the (?:last|past) )?(\d+) (day|week|month|year|bar|session)s?",
         lambda m: (f"{c} <= lowest({c}, {_period(m.group(1), m.group(2))})" if "lowest" in m.group(0)
                    else f"{c} >= highest({c}, {_period(m.group(1), m.group(2))})"))
    take(r"(?:new )?all[- ]time high", lambda m: f"{c} >= cummax({c})")

    # gaps
    def gap(m):
        pct = m.group(2)
        g = "gap" if c == "close" else f"({o} / ref({c}, 1) - 1)"
        if pct is None:
            return f"{g} {'<' if m.group(1) == 'down' else '>'} 0"
        f = float(pct) / 100
        return f"{g} <= {-f:g}" if m.group(1) == "down" else f"{g} >= {f:g}"
    take(rf"gaps? (down|up)(?: by)?(?: more than| at least| over)?(?: {NUM}%)?(?: or more)?", gap)

    # percentage moves
    def move(m):
        verb, pct, n = m.group(1), float(m.group(2)) / 100, m.group(3)
        n = _period(n, m.group(4)) if n else (_period(1, m.group(4)) if m.group(4) else 1)
        down = re.match(r"(down|fall|fell|drop|declin|los|lower|sink|plung|tumbl|slid)", verb) is not None
        r = "change" if (n == 1 and c == "close") else f"ret({c}, {n})"
        return f"{r} <= {-pct:g}" if down else f"{r} >= {pct:g}"
    verbs = r"(down|up|falls?|fell|drops?|dropped|declines?|declined|loses?|lost|gains?|gained|rises?|rose|jumps?|jumped|rall(?:ies|ied|y)|climbs?|climbed|lower|higher|sinks?|sank|plunges?|plunged|tumbles?|tumbled|slides?|slid|is down|is up)"
    take(rf"(?:closes? |trades? |is )?{verbs}(?: by)?(?: more than| at least| over| greater than)? {NUM}%(?: or more)?(?:(?: in| over| during)(?: the)?(?: last| past| prior| previous)? (?:(\d+) )?(day|week|month|session|bar)s?| today| on the day| in a single day| in one day| intraday)?", move)

    # internal bar strength / position in range
    take(rf"ibs (?:is )?{CMPW} {NUM}", lambda m: f"{'ibs' if c == 'close' else f'(({c} - {l}) / ({h} - {l}))'} {_cmp(m.group(1))} {m.group(2)}")
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
    take(r"(closes? |is |trades? )?(below|under|above|over) (?:the )?(lower|upper) bollinger band",
         lambda m: f"{c} {_cmp(m.group(2))} bb_{m.group(3)}(20, 2)" if c == "close"
         else f"{c} {_cmp(m.group(2))} sma({c}, 20) {'-' if m.group(3) == 'lower' else '+'} 2 * stdev({c}, 20)")

    # volume
    take(rf"volume (?:is )?(?:at least |more than |above |over )?{NUM} ?(?:x|times) (?:its |the )?(?:(\d+) day )?(?:average|avg)(?: volume)?(?: of the (?:last|past) (\d+) days)?",
         lambda m: f"{v} >= {m.group(1)} * sma({v}, {m.group(2) or m.group(3) or 20})")

    # calendar
    take(r"(?:on )?(monday|tuesday|wednesday|thursday|friday)s?", lambda m: f"dow == {DOW[m.group(1)]}")
    take(r"(?:on )?(?:the )?last trading day of the month|month[- ]end", lambda m: "trading_days_left_in_month == 1")
    take(r"(?:on )?(?:the )?first trading day of the month", lambda m: "trading_day_of_month == 1")
    take(r"(?:in )?(january|february|march|april|may|june|july|august|september|october|november|december)",
         lambda m: f"month == {['january','february','march','april','may','june','july','august','september','october','november','december'].index(m.group(1)) + 1}")

    # single down / up day (after the % and streak patterns)
    take(r"(?:closes? (?:down|lower|red)|down (?:day|close)|red day|negative day|first down (?:day|close)|trades? down|is down|goes down|moves down|falls|drops|declines)",
         lambda m: f"{'change' if c == 'close' else f'ret({c}, 1)'} < 0")
    take(r"(?:closes? (?:up|higher|green)|up (?:day|close)|green day|positive day|first up (?:day|close)|higher close|trades? up|is up|goes up|moves up|rises|gains)",
         lambda m: f"{'change' if c == 'close' else f'ret({c}, 1)'} > 0")
    take(r"(?:is )?profitable|in profit|shows a profit", lambda m: "pnl > 0")

    leftover = re.sub(r"\b(it|its|the|stock|stocks|shares|share|price|market|when|if|whenever|once|after|and|has|have|had|is|are|was|trades?|closes?|closed|day|today|on|a|an|of|that|this|then|also|same|at|in|for|each|any|every|ticker|etf|index|symbol|[a-z]{1,5} )\b", " ", s + " ")
    leftover = re.sub(r"[^a-z0-9%]+", " ", leftover).strip()
    expr_ = " and ".join(f"({p})" for p in parts) if parts else None
    return expr_, leftover


def parse_conditions(text: str, traded: list[str], strict: bool = True, as_list: bool = False):
    """Parse `x and y or z` style condition text into one expression."""
    and_parts = re.split(r"(?:\band\b|,|\bwhile\b|\bwith\b(?! a))(?![^`]*`)", text)
    exprs, bad = [], []
    for part in and_parts:
        if not part.strip():
            continue
        ors = re.split(r"\bor\b(?![^`]*`)", part)
        or_exprs = []
        for o in ors:
            mentioned = [t for t in find_tickers(o) if t not in traded]
            ctx = Ctx.for_ticker(mentioned[0] if mentioned else None)
            o_clean = o
            for t in find_tickers(o):
                o_clean = re.sub(rf"\$?\b{re.escape(t)}\b", " ", o_clean, flags=re.I)
            for name, sym in COMPANIES.items():
                if sym in find_tickers(o):
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
        raise ParseError(f"No conditions found in {text!r}")
    return exprs if as_list else " and ".join(exprs)


# ----------------------------------------------------------------- top level

ENTRY_VERB = r"^(?:buy|go long|long|purchase|enter|get in|short|sell short|go short|short[- ]sell)\b"
EXIT_VERB = r"^(?:hold|keep|sell|exit|cover|close (?:the|out|it)|get out|take profit|stop|use|with|place|then sell|and sell|after)\b"
COND_START = r"\b(?:when(?:ever)?|if|once|after|on days when|any day|every time|each time)\b"


def _split_clauses(t: str) -> list[str]:
    t = re.sub(r"\band\s+(?=(?:then\s+)?(?:hold|keep|sell|exit|cover|close (?:the|out|it)|take profit|use a|with a|place a)\b)", ", ", t, flags=re.I)
    t = re.sub(r"\bthen\b", ",", t, flags=re.I)
    # protect backtick content from splitting
    chunks, buf, in_tick = [], "", False
    for ch in t:
        if ch == "`":
            in_tick = not in_tick
        if not in_tick and ch in ",;" or (not in_tick and ch == "." and (buf and not buf[-1].isdigit())):
            chunks.append(buf)
            buf = ""
        else:
            buf += ch
    chunks.append(buf)
    return [c.strip() for c in chunks if c.strip()]


OPEN_SAFE = {"gap", "dow", "month", "day", "year", "trading_day_of_month", "trading_days_left_in_month", "open"}


def _open_safe(rule: str) -> bool:
    from .expr import open_safe
    return open_safe(rule)


def parse(text: str, **overrides) -> Strategy:
    raw = text
    t = _normalize(text)
    notes: list[str] = []

    clauses = _split_clauses(t)
    entry_cl, exit_cl, kind = [], [], "entry"
    for cl in clauses:
        low = cl.lower()
        if re.match(ENTRY_VERB, low):
            kind = "entry"
        elif re.match(EXIT_VERB, low) or re.search(r"\b(stop[- ]?loss|trailing stop|profit target|take profit)\b", low):
            kind = "exit"
        (entry_cl if kind == "entry" else exit_cl).append(cl)
    entry_text = " , ".join(entry_cl)
    exit_text = " , ".join(exit_cl)
    low_all = t.lower()

    # --- side
    side = "short" if re.search(r"\b(short|sell short|go short|short[- ]sell)\b", entry_text.lower()) else "long"

    # --- universe and traded tickers: those named before the condition starts
    m = re.search(COND_START, entry_text, flags=re.I)
    subject = entry_text[: m.start()] if m else entry_text
    cond_text = entry_text[m.end():] if m else ""
    universe = _universe_phrase(subject) or _universe_phrase(entry_text if not m else subject)
    traded = find_tickers(subject)
    if universe is None:
        if traded:
            universe = traded
        else:
            ct = find_tickers(cond_text)
            if not ct:
                raise ParseError("Which ticker(s)? Name one (e.g. MSFT or Microsoft) or say 'Nasdaq 100 stocks'.")
            universe = ct[:1]
            traded = universe
    else:
        traded = list(universe)
    # extra tickers named alongside a universe phrase, e.g. "Nasdaq 100 stocks plus QQQ"
    for tk in find_tickers(subject):
        if tk not in universe:
            universe.append(tk)

    # --- entry timing
    el = entry_text.lower()
    if re.search(r"(next|following|tomorrow's)(?: day's| trading day's)? open|at the open|on the open|market on open|\bmoo\b", el):
        entry_fill = "next_open"
    elif re.search(r"(next|following|tomorrow's)(?: day's)? close", el):
        entry_fill = "next_close"
    else:
        entry_fill = "close"
        if not re.search(r"at the close|on the close|at close|market on close|\bmoc\b", el):
            notes.append("Entry timing not stated: assuming a fill at the close of the signal day.")

    # strip timing phrases before parsing conditions
    timing = r"(?:at|on) (?:the )?(?:next |following )?(?:day's |trading day's )?(?:close|open)|market on (?:close|open)|\bmoc\b|\bmoo\b"
    cond_clean = re.sub(timing, " ", cond_text, flags=re.I)
    if not m:
        rest = re.sub(ENTRY_VERB, " ", entry_text.strip(), flags=re.I)
        rest = re.sub(timing, " ", rest, flags=re.I)
        for tk in find_tickers(rest):
            rest = re.sub(rf"\$?\b{re.escape(tk)}\b", " ", rest)
        for name in COMPANIES:
            rest = re.sub(rf"\b{re.escape(name)}\b", " ", rest, flags=re.I)
        rest = re.sub(r"nasdaq[- ]?100|\bndx\b|(?:all|each|any|every) (?:the )?stocks?|stocks?", " ", rest, flags=re.I)
        cond_clean = rest
    if not re.sub(r"[^a-z0-9`]", "", cond_clean.lower()):
        raise ParseError("No entry condition found. Say e.g. 'buy MSFT at the close when it is down 5 days in a row'.")
    parts = parse_conditions(cond_clean, traded, as_list=True)
    if entry_fill == "next_open" and not re.search(r"(next|following|tomorrow's)(?: day's| trading day's)? open", el):
        # "buy at the open when it gaps down": fill at today's open. Any part of the rule that is
        # only known at today's close is checked as of yesterday's close instead.
        entry_fill = "open"
        late = [p for p in parts if not _open_safe(p)]
        if late:
            parts = [p if _open_safe(p) else f"ref({p}, 1)" for p in parts]
            notes.append("Entry at the open: " + " and ".join(late)
                         + " is not known until the close, so it is checked on the previous day's close.")
    entry = " and ".join(parts)
    if re.search(r"\b(?:down|up)_(?:days|streak)\b\s*>=|_streak\([^)]*\) >=", entry):
        notes.append("'N days in a row' also fires on later days of a longer streak (6th, 7th...); say 'exactly N days' to fire only on the Nth.")

    # --- exits
    xl = exit_text.lower()
    hold_bars = None
    hold_fill = "close"
    mh = re.search(r"(?:hold|keep)\w*(?: it| the position| the stock| positions?| the trade)?(?: for)?(?: up to| at most| a maximum of)? (\d+) (trading )?(day|bar|session|week|month)s?", xl) \
        or re.search(r"(?:sell|exit|cover|close)\w*(?: it| the position| out)? (?:after|in) (\d+) (trading )?(day|bar|session|week|month)s?", xl) \
        or re.search(r"(?:after|or after|max(?:imum)? hold(?:ing)?(?: period)?(?: of)?|time stop(?: of)?) (\d+) (trading )?(day|bar|session|week|month)s?", xl)
    mh = mh or re.search(r"(\d+) (trading )?(day|bar|session|week|month)s? later", xl)
    if mh:
        hold_bars = _period(mh.group(1), mh.group(3))
    elif re.search(r"(?:sell|exit|cover)\w* (?:at |on )?(?:the )?(?:next|following|tomorrow)", xl) or re.search(r"\bnext day\b", xl):
        hold_bars = 1
    if re.search(r"(?:sell|exit|cover|close)\w*[^,]*?(?:at|on) (?:the )?(?:next |following |day's )?open\b", xl) or re.search(r"\bnext (?:day's )?open\b", xl):
        hold_fill = "open"

    stop = re.search(rf"(?<!trailing )stop[- ]?(?:loss)?(?: of| at)? {NUM}%", xl) or re.search(rf"{NUM}% (?:hard )?stop(?![- ]?(?:out)?[- ]?trail)(?:[- ]loss)?", xl)
    trail = re.search(rf"trailing stop(?:[- ]loss)?(?: of| at)? {NUM}%", xl) or re.search(rf"{NUM}% trailing stop", xl)
    tp = re.search(rf"(?:take[- ]profit|profit target|target|take profits?)(?: of| at)? \+?{NUM}%", xl) or re.search(rf"{NUM}% (?:profit target|take[- ]profit|target|gain)", xl)
    stop_loss = float(stop.group(1)) / 100 if stop and not (trail and stop.start() == trail.start()) else None
    if trail and stop and re.search(rf"{re.escape(stop.group(0))}", trail.group(0)):
        stop_loss = None

    exit_when = None
    exit_when_fill = "close"
    mw = re.search(rf"(?:sell|exit|cover|close (?:the position|out|it))\w*(?: it| the position| out)? (?:when|if|once|on|as soon as|at the first|at the close of the first|at the close when)\b(.*?)(?=(?:,|$| or after \d| or in \d| or \d+ (?:days|bars)|;))", exit_text, flags=re.I)
    if mw and mw.group(1).strip():
        wtxt = mw.group(1)
        if re.search(r"next (?:day's )?open|at the open", wtxt, re.I):
            exit_when_fill = "next_open"
        wtxt = re.sub(timing, " ", wtxt, flags=re.I)
        wtxt = re.sub(rf"\b(?:or )?(?:after|in) \d+ (?:trading )?(?:days?|bars?|sessions?|weeks?).*$", " ", wtxt, flags=re.I)
        wtxt = re.sub(rf"\b(?:the )?first\b", " ", wtxt, flags=re.I)
        if wtxt.strip():
            exit_when = parse_conditions(wtxt, traded)

    if not any([hold_bars, exit_when, stop_loss, trail, tp]):
        m1 = re.search(r"(?:sell|exit|cover)\w* (?:at|on) (?:the )?(close|open)", xl)
        if m1:
            hold_bars = 1
            hold_fill = m1.group(1)
            notes.append(f"No holding period stated: exiting at the first {m1.group(1)} after entry.")
        else:
            raise ParseError("No exit rule found. Say e.g. 'hold 1 day and sell at the close', 'sell when it closes above its 5-day moving average', or 'with a 5% stop loss'.")

    # --- portfolio and costs (anywhere in text)
    kw: dict = {}
    mc = re.search(r"(?:start(?:ing)? with|capital of|initial capital|account of|with) \$?([\d,]+(?:\.\d+)?)\s*(k|m)?\b(?! ?(?:%|bps|basis|per|commission))", low_all)
    if mc and "$" in t[max(0, mc.start() - 2): mc.end() + 1] or (mc and mc.group(2)):
        val = float(mc.group(1).replace(",", "")) * {"k": 1e3, "m": 1e6, None: 1}[mc.group(2)]
        if val >= 100:
            kw["capital"] = val
    mp = re.search(r"(?:max(?:imum)?(?: of)?|up to|at most|no more than|hold at most) (\d+) (?:open |simultaneous |concurrent )?(?:positions|stocks|names|holdings|trades)", low_all)
    if mp:
        kw["max_positions"] = int(mp.group(1))
    ms = re.search(rf"{NUM}% (?:of (?:the )?(?:equity|capital|account|portfolio) )?(?:per|in each|for each|each) (?:trade|position|stock)", low_all)
    if ms:
        kw["position_size"] = float(ms.group(1)) / 100
    msl = re.search(rf"{NUM} ?(?:bps|basis points?)(?: of)? slippage|slippage(?: of)? {NUM} ?(?:bps|basis points?)", low_all)
    if msl:
        kw["slippage_bps"] = float(msl.group(1) or msl.group(2))
    mcm = re.search(rf"\$\s?{NUM} (?:per trade|commission|per order|a trade)|commissions? of \$\s?{NUM}", low_all)
    if mcm:
        kw["commission"] = float(mcm.group(1) or mcm.group(2))
    mcs = re.search(rf"\$\s?{NUM} (?:per share|a share)", low_all)
    if mcs:
        kw["commission_per_share"] = float(mcs.group(1))
    mlev = re.search(rf"{NUM}x leverage|leverage of {NUM}", low_all)
    if mlev:
        kw["leverage"] = float(mlev.group(1) or mlev.group(2))
    # dates
    mfrom = re.search(r"(?:from|since|starting(?: in)?|after|between) (\d{4})(?:-(\d{2})-(\d{2}))?", low_all)
    mto = re.search(r"(?:to|until|through|thru|before|and) (\d{4})(?:-(\d{2})-(\d{2}))?\b", low_all[mfrom.end():] if mfrom else low_all)
    if mfrom:
        y = mfrom.group(1)
        kw["start"] = f"{y}-{mfrom.group(2)}-{mfrom.group(3)}" if mfrom.group(2) else f"{y}-01-01"
    if mto and (mfrom or re.search(r"(?:until|through|before) \d{4}", low_all)):
        y = mto.group(1)
        kw["end"] = f"{y}-{mto.group(2)}-{mto.group(3)}" if mto.group(2) else f"{y}-12-31"
    # ranking
    mr = re.search(r"(?:prefer(?:ring)?|rank(?:ed)? by|pick(?:ing)?|choose|choosing|favor(?:ing)?) (?:the )?(lowest|highest|weakest|strongest|biggest losers?|biggest gainers?|most oversold|most overbought|biggest decline|largest decline)(?: (rsi|return|decline|change))?", low_all)
    if mr:
        w = mr.group(1)
        if mr.group(2) == "rsi" or "oversold" in w or "overbought" in w:
            mn = re.search(r"rsi\(\s*[^,]+,\s*(\d+)\)", entry)
            kw["rank_by"] = f"rsi({mn.group(1) if mn else 2})"
            kw["rank_ascending"] = w in ("lowest", "most oversold")
        else:
            kw["rank_by"] = "ret(5)" if "return" in (mr.group(2) or "") else "change"
            kw["rank_ascending"] = w in ("lowest", "weakest", "biggest loser", "biggest losers", "biggest decline", "largest decline")
    if len(universe) > 1 and "max_positions" not in kw:
        kw["max_positions"] = 10
        notes.append("Multiple tickers and no position limit given: using up to 10 positions at 10% of equity each.")

    strat = Strategy(
        universe=universe, entry=entry, side=side, entry_fill=entry_fill,
        hold_bars=hold_bars, hold_exit_fill=hold_fill,
        exit_when=exit_when, exit_when_fill=exit_when_fill,
        stop_loss=stop_loss,
        trailing_stop=float(trail.group(1)) / 100 if trail else None,
        take_profit=float(tp.group(1)) / 100 if tp else None,
        description=raw, notes=notes, **kw,
    )
    for k, v in overrides.items():
        if v is not None:
            setattr(strat, k, v)
    strat.validate()
    return strat
