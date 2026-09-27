"""Index universes named in a sentence ("S&P 500 stocks", "Russell 2000 stocks", "Dow 30 stocks").

The Nasdaq-100 has point-in-time membership (data/ndx_membership.csv) and is handled by the parser as the "NDX"
universe. For the other indexes the data has at most TODAY'S constituents (data/index_constituents.json: S&P 500,
400 and 600, refreshed by the data job) and no membership history, so:

  - "S&P 500 stocks" (members / companies / constituents / components) is ambiguous and is never silently read as
    the ETF (SPY): the parser asks whether the index fund or today's members are meant.
  - "today's S&P 500 members" (or "current S&P 500 stocks") is today's constituent list, traded over the whole
    period with a loud survivorship warning that also says how many of them have price data here.
  - Russell 1000 / 2000 / 3000 and Dow 30 stocks: no constituent list at all; the parser asks for the ETF or an
    S&P universe instead.

The index itself ("buy the S&P 500 when ...") stays its ETF (SPY), as before.
"""
from __future__ import annotations

import re

from . import data

# name pattern, constituent keys in index_constituents.json (None: no list), ETF, label
_INDEXES = [
    (r"s\s*(?:&|and)\s*p\s*1500|s\s*&\s*p\s*composite\s*1500", ("sp500", "sp400", "sp600"), "SPTM", "S&P 1500"),
    (r"s\s*(?:&|and)\s*p\s*(?:mid[- ]?cap\s*)?400", ("sp400",), "MDY", "S&P 400"),
    (r"s\s*(?:&|and)\s*p\s*(?:small[- ]?cap\s*)?600", ("sp600",), "IJR", "S&P 600"),
    (r"s\s*(?:&|and)\s*p\s*500|\bsp\s?500\b|\bspx\b", ("sp500",), "SPY", "S&P 500"),
    (r"russell\s*1000", None, "IWB", "Russell 1000"),
    (r"russell\s*2000", None, "IWM", "Russell 2000"),
    (r"russell\s*3000", None, "IWV", "Russell 3000"),
    (r"dow\s*jones\s*industrial(?:\s*average)?(?:\s*30)?|dow\s*jones(?:\s*30)?|dow\s*30|\bdjia\b|(?:the\s+)?dow", None,
     "DIA", "Dow Jones Industrial Average"),
]
_MEMBERS = r"(?:stocks?|compan(?:y|ies)|members?|constituents?|components?|names|equities)\b"
_TODAY = r"(?:today'?s|current(?:ly listed)?|present|latest|existing)"
_NAME = "|".join(f"(?:{p})" for p, *_ in _INDEXES)
# "S&P 500 stocks", "today's S&P 500 members", "the stocks in the S&P 500", "members of the current Russell 2000"
PHRASE = (rf"(?:(?:all|each|every|any)\s+(?:of\s+)?)?(?:the\s+)?(?:{_TODAY}\s+)?(?:the\s+)?\b(?:{_NAME})(?:\s+index)?\s+"
          rf"(?:{_TODAY}\s+)?{_MEMBERS}(?:\s+(?:today|now|currently))?"
          rf"|(?:(?:all|each|every|any)\s+(?:of\s+)?)?(?:the\s+)?(?:{_TODAY}\s+)?{_MEMBERS}\s+(?:in|of|from)\s+(?:the\s+)?"
          rf"(?:{_TODAY}\s+)?\b(?:{_NAME})(?:\s+index)?(?:\s+(?:today|now|currently))?")


def _which(text: str):
    for pat, keys, etf, label in _INDEXES:
        if re.search(pat, text, re.I):
            return keys, etf, label
    return None


def _question(label: str, etf: str, keys) -> str:
    alt = ""
    if keys is not None:
        n_all, n_have = coverage(keys)
        alt = (f", or today's {label} members as a stock universe: say 'today's {label} members' (the current list of "
               f"{n_all} stocks, {n_have} of them with price data here, traded over the whole period: survivorship-"
               "biased, since past members that were dropped, went bankrupt or were acquired are missing)")
    else:
        alt = (" (there is no constituent list for it here; for a stock universe use the Nasdaq 100, which has point-in-"
               "time membership, or today's S&P 500 / 400 / 600 members)")
    return (f"'{label}' stocks: there is no point-in-time membership history for the {label} here, so its stocks cannot "
            f"be traded the way they were in the index at the time. Did you mean the index fund - say '{etf}'{alt}?")


def coverage(keys) -> tuple[int, int]:
    """(constituents in today's list, constituents with price data)."""
    names = members(keys, with_data=False)
    have = set(data.available_tickers())
    return len(names), sum(1 for t in names if t in have)


def members(keys, with_data: bool = True) -> list[str]:
    doc = data.index_constituents()
    out: list[str] = []
    for k in keys:
        for t in (doc.get(k) or {}):
            t = data.canonical(t)
            if t not in out:
                out.append(t)
    if with_data:
        have = set(data.available_tickers())
        out = [t for t in out if t in have]
    return out


def universe(text: str):
    """(tickers, universe name) for an index-members phrase in `text`, None when there is none. Raises
    parser.ParseError with a question when the phrase is ambiguous (the index fund or its stocks?) or the index has
    no constituent list."""
    from .parser import ParseError
    m = re.search(PHRASE, text, re.I)
    if not m:
        return None
    hit = _which(m.group(0))
    if hit is None:
        return None
    keys, etf, label = hit
    today = re.search(rf"(?i)\b{_TODAY}\b|\b(?:today|now|currently)\b", m.group(0))
    if keys is None or not today:
        raise ParseError(_question(label, etf, keys))
    tick = members(keys)
    if not tick:
        raise ParseError(f"None of today's {label} members has price data here (the data job fetches them). Use {etf} "
                         "for the index.")
    return tick, universe_name(label)


def universe_name(label: str) -> str:
    """A Strategy's universe_name for today's members of an index (with point_in_time False, the description reads
    "S&P 500 (N tickers, today's members (survivorship-biased by construction))")."""
    return label


def is_today_members(name) -> bool:
    return any(name == lab for _p, keys, _e, lab in _INDEXES if keys is not None)


def warning(name: str, n: int) -> str:
    """The survivorship warning of a today's-members universe of n tickers."""
    label = name
    keys = next((k for p, k, e, lab in _INDEXES if lab == label), None)
    n_all = coverage(keys)[0] if keys else n
    return (f"Warning: survivorship bias - the universe is TODAY'S {label} members ({n} stocks with price data here, "
            f"of {n_all} in the current list), traded over the whole period with no membership filter: there is no "
            f"point-in-time {label} membership history here. They were chosen because they are in the index now (they "
            "survived and grew), which past-you could not know, and past members that were dropped, went bankrupt or "
            "were acquired are missing: results are biased upward, often strongly. For the index itself use its fund; "
            "for point-in-time membership use the Nasdaq 100.")
