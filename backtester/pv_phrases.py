"""Portfolio Visualizer phrasings read before the main parser (parser._parse calls `rewrite` first and `apply` on
the parsed spec):

- a single holding written as a weight: "100% SPY", "VTI 100", "VTI 100%" (and "SPY 60 AGG 40": weights after
  the tickers) -> "hold 100% SPY";
- "X% each of A, B, C and D" -> "X% A, X% B, X% C and X% D" (refused unless the weights add up to 100%);
- a glide path (dynamic allocation): "hold 90% VTI and 10% BND gliding to 40% VTI and 60% BND over 30 years",
  "... linearly by 2% a year", "... by 2050", "..., target date 2050 glide path" -> Portfolio.glide;
- dual momentum's absolute-momentum hurdle stated again: "..., only if their 12 month return is above BIL's 12
  month return" -> the hurdle applied as written (BIL's total return instead of the built-in T-bill return), and
  "... above T-bills / cash / the risk-free rate" -> the built-in hurdle, with a note.
"""
from __future__ import annotations

import re

from . import data

_TK = r"[\^$]?[A-Z][A-Z0-9.\-]{0,9}"          # a ticker as written (upper case)
_W = r"\d+(?:\.\d+)?"
_OPTS = r"(?P<rest>\s*(?:[,;].*)?)"


def _is_ticker(w: str) -> bool:
    return data.canonical(w) in set(data.available_tickers())


def _single_or_trailing_weights(text: str) -> str | None:
    """'100% SPY' / 'VTI 100' / 'VTI 100%' / 'SPY 60 AGG 40' (the whole leading clause) -> 'hold ...'."""
    m = re.fullmatch(rf"\s*(?P<w>{_W})\s*%\s*(?P<t>{_TK}){_OPTS}", text)
    if m and _is_ticker(m.group("t")):
        return f"hold {m.group('w')}% {m.group('t')}{m.group('rest')}"
    m = re.fullmatch(rf"\s*(?P<pairs>{_TK}\s+{_W}\s*%?(?:\s*,?\s*(?:and\s+)?{_TK}\s+{_W}\s*%?)*){_OPTS}", text)
    if m:
        pairs = re.findall(rf"({_TK})\s+({_W})\s*%?", m.group("pairs"))
        if pairs and all(_is_ticker(t) for t, _ in pairs):
            tot = sum(float(w) for _, w in pairs)
            if abs(tot - 100) > 0.01 and not (len(pairs) > 1 and abs(tot - 1) < 1e-6):
                from .parser import ParseError
                raise ParseError(f"'{m.group('pairs').strip()}': the weights add up to {tot:g}%, not 100%.")
            scale = 100 if abs(tot - 1) < 1e-6 else 1
            items = [f"{float(w) * scale:g}% {t}" for t, w in pairs]
            body = items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]
            return f"hold {body}{m.group('rest')}"
    return None


_EACH = re.compile(rf"(?is)(?P<pre>^|\b(?:hold|own|buy|invest|put)\s+)(?P<w>{_W})\s*%\s+(?:each\s+(?:of|in|into)|in\s+each\s+of|of\s+each\s+of)\s+"
                   r"(?P<lst>.+?)(?P<rest>(?:,\s*(?=(?:since|from|starting|between|rebalanc|with|and rebalanc|over|until|through|to \d|"
                   r"in \d|add|contribut|withdraw|start|end|using|benchmark)\b)|;|\.\s|$).*)$")


def _each_of(text: str) -> str:
    """'hold 25% each of A, B, C and D[, options]' -> 'hold 25% A, 25% B, 25% C and 25% D[, options]'."""
    m = _EACH.search(text)
    if not m:
        return text
    from .parser import ParseError
    lst = m.group("lst").strip().rstrip(",")
    items = [x.strip() for x in re.split(r"\s*,\s*(?:and\s+)?|\s+and\s+|\s*&\s*", lst) if x.strip()]
    if len(items) < 2:
        return text
    if re.search(r"(?i)\b(?:each|only|when|whenever|while|if|otherwise|else|above|below|unless|rebalanc\w*)\b", lst):
        return text              # a condition or option follows: the parser's own 'X% each of' reads it
    if all(re.fullmatch(_TK, x) for x in items):
        return text              # tickers only: the parser's own 'X% each of' reads them (with its own checks)
    w = float(m.group("w"))
    tot = w * len(items)
    if abs(tot - 100) > 0.01:
        raise ParseError(f"'{m.group('w')}% each of' {len(items)} holdings ({', '.join(items)}) adds up to {tot:g}%, "
                         "not 100%. Give weights that add up to 100% (e.g. "
                         f"{100 / len(items):g}% each), or name the rest (cash, another holding).")
    body = ", ".join(f"{m.group('w')}% {x}" for x in items[:-1]) + f" and {m.group('w')}% {items[-1]}"
    _note_later(f"'{m.group('w')}% each of {lst}' was read as {body} ({len(items)} x {m.group('w')}% = 100%).")
    return text[: m.start()] + m.group("pre") + body + m.group("rest")


_GLIDE_VERB = r"(?:gliding|glides?|moving|shifting|shifts?|transitioning|de-?risking)(?:\s+linearly)?\s+(?:to|towards?|into)"
_GLIDE = re.compile(
    rf"(?is)\s*,?\s*(?:and\s+|then\s+)?{_GLIDE_VERB}\s+(?P<to>.+?)"
    r"(?P<span>\s+(?:linearly\s+)?(?:over|across|in)\s+(?:the\s+next\s+)?(?P<yrs>\d+(?:\.\d+)?)\s+years?(?:\s+linearly)?"
    r"|\s+(?:linearly\s+)?(?:by|until|through)\s+(?:the\s+(?:end|start)\s+of\s+)?(?P<endy>(?:19|20)\d\d(?:-\d\d(?:-\d\d)?)?)(?:\s+linearly)?"
    r"|\s*,?\s+(?:linearly\s+)?(?:by|at|moving)\s+(?P<step>\d+(?:\.\d+)?)\s*%(?:\s*points?)?\s+(?:a|per|each)\s+year(?:\s+linearly)?)"
    r"(?=\s*(?:[,;.]|$))")
_TARGET_DATE = re.compile(r"(?is)\s*,?\s*(?:with\s+|using\s+|on\s+|and\s+)?(?:an?\s+|the\s+)?target[- ]date\s+(?P<y>(?:19|20)\d\d)"
                          r"(?:\s+(?:glide\s*path|fund|style))?(?:\s+glide\s*path)?(?=\s*(?:[,;.]|$))"
                          r"|\s*,?\s*(?:with\s+|using\s+|on\s+|and\s+)?(?:an?\s+|the\s+)?target[- ]date\s+glide\s*path\s+"
                          r"(?:to|for|ending(?:\s+in)?|until)\s+(?P<y2>(?:19|20)\d\d)(?=\s*(?:[,;.]|$))")
_START = re.compile(r"(?i)\b(?:since|from|starting(?:\s+in|\s+on|\s+from)?|beginning(?:\s+in)?)\s+((?:18|19|20)\d\d(?:-\d\d(?:-\d\d)?)?)\b")


def _glide(text: str, post: dict) -> str:
    m = _GLIDE.search(text)
    td = _TARGET_DATE.search(text)
    if not m and not td:
        return text
    from .parser import ParseError
    g: dict = {}
    if td:
        g["shape"] = "target_date"
        g["end"] = td.group("y") or td.group("y2")
        text = text[: td.start()] + text[td.end():]
        m = _GLIDE.search(text)
    if m:
        to_text = m.group("to").strip()
        if m.group("yrs"):
            g["years"] = float(m.group("yrs"))
            g.pop("end", None)
        elif m.group("endy"):
            g["end"] = m.group("endy")
        elif m.group("step"):
            g["per_year"] = float(m.group("step")) / 100
            g.pop("end", None)
        post["glide_to_text"] = to_text
        text = text[: m.start()] + text[m.end():]
    elif not g.get("end"):
        raise ParseError("A glide path needs where it goes and how fast, e.g. 'gliding to 40% VTI and 60% BND over 30 "
                         "years' or '... by 2% a year'.")
    ms = _START.search(text)
    post["glide_start_text"] = ms.group(1) if ms else None
    post["glide"] = g
    return text


def _hurdle(text: str, post: dict) -> str:
    """Dual momentum's absolute-momentum clause said again ('only if their 12 month return is above BIL's')."""
    if not re.search(r"(?i)\bdual momentum\b", text):
        return text
    m = re.search(r"(?is)\s*,?\s*(?:but\s+|and\s+)?only\s+(?:if|when|while)\s+(?:their|its|the\s+winner'?s?|the\s+chosen\s+(?:one|asset)'?s?|the\s+pick'?s?)\s+"
                  r"(?P<n>\d+)[- ](?P<u>month|week|day|year)s?\s+(?:total\s+)?(?:returns?|momentum|performance)\s+(?:is|are)\s+"
                  r"(?:above|greater\s+than|higher\s+than|over|more\s+than|better\s+than|beats?|exceeds?|positive\s+relative\s+to)\s+"
                  r"(?:that\s+of\s+|the\s+)?(?P<h>t-?bills?|treasury\s+bills|cash|the\s+risk[- ]free\s+rate|risk[- ]free|"
                  rf"(?P<t>{_TK}))(?:'s?)?(?:\s+(?:\d+[- ](?:month|week|day|year)s?\s+)?(?:total\s+)?(?:returns?|momentum|performance)"
                  r"(?:\s+over\s+the\s+same\s+period)?)?(?=\s*(?:[,;.]|$))", text)
    if not m:
        return text
    post["hurdle"] = {"n": int(m.group("n")), "unit": m.group("u").lower(), "ticker": m.group("t"),
                      "text": m.group(0).strip(" ,")}
    return text[: m.start()] + text[m.end():]


def _note_later(msg: str) -> None:
    from .parser import _note
    _note(msg)


_PV_DEFAULTS = re.compile(r"(?i)(?:with|using|on|in) (?:the )?(?:portfolio ?visualizer|pv)(?:'s)? (?:defaults?|default settings|settings)"
                         r"|(?:as|like) (?:in )?(?:portfolio ?visualizer|pv)\b")


def _conventions(text: str, post: dict) -> str:
    """'inflation adjusted annually' (PV's once-a-year CPI step-up), 'expense ratio net of leverage'."""
    if _PV_DEFAULTS.search(text):
        post["pv_defaults"] = True
    t = re.sub(r"(?i)\b(adjusted for|indexed to|rising with|growing with|grown with) inflation,? (?:once a year|annually|yearly|each year|every year)\b",
               lambda m: (post.__setitem__("indexing", "annual"), f"{m.group(1)} inflation")[1], text)
    t = re.sub(r"(?i)\b(?:inflation[- ]adjusted|inflation[- ]indexed) (?:once a year|annually|yearly)\b",
               lambda m: (post.__setitem__("indexing", "annual"), "adjusted for inflation")[1], t)
    t = re.sub(r"(?i),? ?(?:and |with )?(?:the )?inflation (?:adjustments?|indexing) (?:applied |made )?(?:once a year|annually|yearly)\b",
               lambda m: (post.__setitem__("indexing", "annual"), "")[1], t)
    t = re.sub(r"(?i),? ?(?:and |with )?(?:the )?inflation (?:adjustments?|indexing) (?:applied |made )?(?:monthly|as published|every month)\b",
               lambda m: (post.__setitem__("indexing", "published"), "")[1], t)
    er = r"(?P<er>expense ratio(?: of \d+(?:\.\d+)?%)?|expense ratio (?:is |at )?\d+(?:\.\d+)?%|annual fee|management fee)"
    t = re.sub(rf"(?i)\b{er},? (?:charged |applied |taken )?(?:net of leverage|on (?:the )?(?:equity|net assets|net asset value|"
               r"account value|balance)(?: \(net of leverage\))?)\b",
               lambda m: (post.__setitem__("expense_on", "equity"), m.group("er"))[1], t)
    t = re.sub(rf"(?i)\b{er},? (?:charged |applied |taken )?on (?:the )?(?:gross|leveraged|borrowed) (?:assets|exposure|holdings)\b",
               lambda m: (post.__setitem__("expense_on", "gross"), m.group("er"))[1], t)
    return t


def rewrite(text: str) -> tuple[str, dict]:
    """The sentence with these phrasings rewritten for the main parser, and what to apply to the parsed spec."""
    post: dict = {}
    if "`" in text:
        return text, post
    t = _conventions(text, post)
    t = _hurdle(t, post)
    t = _glide(t, post)
    t = _each_of(t)
    single = _single_or_trailing_weights(t)
    if single:
        t = single
    return t, post


def apply(obj, post: dict) -> None:
    """Set what `rewrite` took out of the sentence on the parsed spec (before its validation)."""
    if not post:
        return
    from .parser import ParseError, parse
    from .portfolio import Portfolio, _fixed_mix
    if isinstance(obj, Portfolio):
        ind = post.get("indexing") or ("annual" if post.get("pv_defaults") else None)
        if ind:
            obj.inflation_indexing = ind
            if post.get("pv_defaults") and not post.get("indexing") and (obj.withdrawal or obj.contribution) and \
                    (obj.flow_inflation("withdrawal") or obj.flow_inflation("contribution")):
                obj.notes.append("Portfolio Visualizer defaults: inflation-adjusted cash flows are stepped up once a "
                                 "year (PV's convention; the first year's payments total exactly the stated amount), "
                                 "not with each month's CPI.")
        if post.get("expense_on"):
            obj.expense_on = post["expense_on"]
    elif post.get("indexing") or post.get("expense_on"):
        raise ParseError("Inflation indexing and expense-ratio conventions apply to allocation portfolios.")
    if post.get("hurdle"):
        h = post["hurdle"]
        tree = getattr(obj, "tree", None)
        f = tree.get("filter") if isinstance(tree, dict) else None
        if not isinstance(obj, Portfolio) or not f or "tbill_ret" not in str(f.get("require", "")):
            raise ParseError(f"'{h['text']}': a hurdle is read with dual momentum ('dual momentum between SPY and EFA "
                             "with AGG as the safe asset').")
        n = {"month": 21, "week": 5, "day": 1, "year": 252}[h["unit"]] * h["n"]
        t = h["ticker"]
        if t is None:
            obj.notes.append(f"'{h['text']}' restates dual momentum's built-in absolute-momentum hurdle (the winner's "
                             "return must beat T-bills', else the safe asset): nothing changed.")
            if f"tbill_ret({n})" not in f["require"]:
                f["require"] = f"tret({n}) > tbill_ret({n})"
        else:
            tk = data.canonical(t)
            data.load(tk)
            f["require"] = f'tret({n}) > tret(sym("{tk}").tr, {n})'
            obj.notes.append(f"'{h['text']}': applied as written: the winner is held only when its {h['n']} {h['unit']} "
                             f"total return beats {tk}'s over the same {h['n']} {h['unit']}s (dividends reinvested), in "
                             "place of the built-in hurdle (the T-bill return, as Antonacci and Portfolio Visualizer "
                             f"use); otherwise the safe asset is held. {tk} is close to the T-bill return, less its fee.")
    if post.get("glide") is not None:
        if not isinstance(obj, Portfolio) or _fixed_mix(obj.tree) is None:
            raise ParseError("A glide path moves a fixed mix: write the start mix with weights, e.g. 'hold 90% VTI and "
                             "10% BND gliding to 40% VTI and 60% BND over 30 years'.")
        g = dict(post["glide"])
        mix0 = _fixed_mix(obj.tree)
        if post.get("glide_to_text"):
            to_text = post["glide_to_text"]
            start = post.get("glide_start_text")
            try:
                sub = parse(f"hold {to_text}" + (f", since {start}" if start else ""))
            except ParseError as e:
                raise ParseError(f"The glide path's end mix '{to_text}': {e}") from None
            to = _fixed_mix(getattr(sub, "tree", None))
            if to is None:
                raise ParseError(f"The glide path's end mix '{to_text}' must be tickers with weights, e.g. '40% VTI "
                                 "and 60% BND'.")
        else:
            if len(mix0) != 2:
                raise ParseError("A target-date glide path without an end mix needs two holdings (stocks first, then "
                                 "bonds); otherwise say where it goes, e.g. 'gliding to 40% VTI and 60% BND by 2050'.")
            a, b = list(mix0)
            to = {a: 0.4, b: 0.6}
            obj.notes.append(f"Target-date glide path to {g['end']}: no end mix was given, so it ends at 40% {a} / 60% "
                             f"{b} (the first holding read as the stock sleeve), about the stock share of target-date "
                             "funds at their target date. Say 'gliding to ...' to choose it.")
        g["to"] = to
        obj.glide = g
        if obj.rebalance == "none":
            obj.rebalance = "yearly"
            obj.notes.append("A glide path moves the target at rebalances: rebalanced yearly (say 'rebalance monthly' "
                             "or 'quarterly' to follow it more closely).")
