"""One margin model for both engines (signal strategies and portfolios).

Accounts (`margin_account`):
- "reg_t" (default): Regulation T. Initial margin 50%, so at most 2x gross exposure overnight.
- "portfolio": portfolio margin, up to 4x (the initial requirement is 25% of each position).

Leveraged and inverse ETFs carry higher requirements. FINRA Rule 4210(f)(8) sets the maintenance requirement of a
leveraged ETF at the standard requirement times the fund's leverage factor, capped at 100% (2x: 50%, 3x: 75% with
the usual 25% maintenance), and brokers (Schwab, Fidelity, IBKR, ...) apply the same multiple to the initial
requirement. So here, per position with leverage factor f (|f|, at least 1):

    maintenance requirement = min(100%, maintenance_margin x f)
    initial requirement     = max(1 / account cap, maintenance requirement)

A strategy or portfolio is refused when its largest target exposure needs more than the equity as initial margin
(the broker would not let you open it), or when its maintenance requirement at the target leaves less than
MARGIN_BUFFER (10%) of the equity above it (a margin call on the first ordinary move: TQQQ at 1.3x needs 97.5%).

Margin calls: at a close where equity is below the total maintenance requirement, every position is cut pro rata to
the LOWER of (a) the target exposure and (b) the exposure at which equity is MARGIN_CALL_CUSHION (125%) of the
maintenance requirement. A broker's liquidation restores a cushion rather than stopping exactly at the line; without
one, a target that sits just inside the maintenance limit would be cut straight back to the edge and called again on
the next down close.
"""
from __future__ import annotations

MAX_LEVERAGE = {"reg_t": 2.0, "portfolio": 4.0}
MARGIN_CALL_CUSHION = 1.25
# a target whose maintenance requirement leaves less than this share of the equity above it is refused: it would be
# margin-called on the first ordinary move (TQQQ at 1.3x: 97.5% of the equity, called on its second day in 2019)
MARGIN_BUFFER = 0.10

# |leverage factor| of leveraged and inverse ETFs (current prospectus factors; -1x funds are listed as 1)
_FACTORS = {
    3.0: """TQQQ SQQQ UPRO SPXU SPXL SPXS UDOW SDOW URTY SRTY TNA TZA UMDD SMDD MIDU TMF TMV TYD TYO TTT SOXL SOXS
            TECL TECS FAS FAZ LABU LABD DRN DRV CURE DPST NAIL RETL DFEN WEBL WEBS HIBL HIBS UTSL PILL TPOR WANT DUSL
            KORU MEXX EURL YINN YANG EDC EDZ FNGU FNGD BNKU BNKD NRGU NRGD OILU OILD GDXU GDXD""",
    2.0: """QLD QID SSO SDS SPUU DDM DXD UWM TWM MVV MZZ SAA SDD ROM REW USD SSG UYG SKF RXL RXD UGE SZK UXI UYM SMN
            DIG DUG URE SRS UPW SDP BIB BIS UBT TBT UST PST ULE EUO YCS YCL UCO SCO BOIL KOLD UGL GLL AGQ ZSL ERX ERY
            BRZU CHAU CWEB INDL JNUG JDST NUGT DUST GUSH DRIP FNGO BITX BITU SBIT ETHU ETHT ETHD CONL NVDL NVDX NVDQ
            TSLL TSLQ MSTU MSTZ AAPU AMZU GGLL MSFU METU UVIX TARK""",
    1.5: "UVXY NVDS",
}
LEVERAGE_FACTOR = {t: f for f, s in _FACTORS.items() for t in s.split()}


# ---- default borrow fees for shorts (when no borrow fee is given; borrow_fee = a number overrides it for every short,
# 0 turns it off). Assumptions, from typical Interactive Brokers indicative borrow rates (its "Short Stock
# Availability" / SLB data), which vary daily and by broker:
#   - leveraged, inverse and volatility ETPs (SQQQ, SOXS, UVXY, VXX, SH, ...): 5%/yr. They are usually "hard to
#     borrow": their rates mostly sit in the 3-10%/yr range and spike far higher when supply is short;
#   - other ETFs and stocks: 0.3%/yr, about the general-collateral rate of liquid large caps and major ETFs.
# A small, crowded or recently listed stock can cost 10-100%/yr to borrow; this default does not know that.
BORROW_HARD = 0.05
BORROW_GENERAL = 0.003
_HARD_TO_BORROW = set("""SH PSQ DOG RWM SEF EUM MYY SBB TBF TBX SJB BITI SVIX SVXY VXX VIXY VIXM VXZ UVIX UVXY
                         REK EFZ YXI""".split())


def default_borrow_fee(ticker) -> float:
    """The annual borrow fee assumed for a short in `ticker` when none is given (see above)."""
    t = str(ticker).upper()
    if factor(t) > 1 or t in _HARD_TO_BORROW:
        return BORROW_HARD
    return BORROW_GENERAL


def borrow_fee_of(ticker, given) -> float:
    """The borrow fee charged on a short in `ticker`: the one given (a number, 0 = none), else the default."""
    return float(given) if given is not None else default_borrow_fee(ticker)


def borrow_note(tickers) -> str:
    """What the default borrow fees assume, for the tickers that may be shorted."""
    hard = sorted({str(t).upper() for t in tickers if default_borrow_fee(t) == BORROW_HARD})
    rest = [t for t in tickers if default_borrow_fee(t) != BORROW_HARD]
    parts = []
    if hard:
        parts.append(f"{', '.join(hard[:6])}{' ...' if len(hard) > 6 else ''} {BORROW_HARD:.0%}/yr (leveraged, inverse "
                     "and volatility ETPs are usually hard to borrow: typically 3-10%/yr at Interactive Brokers, more "
                     "when supply is short)")
    if rest:
        parts.append(f"{'other tickers' if hard else 'the shorted tickers'} {BORROW_GENERAL:.1%}/yr (the general-"
                     "collateral rate of liquid stocks and ETFs; small or crowded stocks can cost far more)")
    return ("Borrow fee (assumed, none was given): shorts pay " + "; ".join(parts) + ", charged daily on the short's "
            "market value. Say e.g. 'borrow fee 2%' to set one rate for every short, or 'no borrow fee' for none.")


def factor(ticker) -> float:
    """|Leverage factor| of a fund (1 for an ordinary stock or fund)."""
    return LEVERAGE_FACTOR.get(str(ticker).upper(), 1.0)


def maintenance(ticker, base: float) -> float:
    return min(1.0, float(base) * factor(ticker))


def initial(ticker, base: float, account: str = "reg_t") -> float:
    return max(1.0 / MAX_LEVERAGE[account], maintenance(ticker, base))


def check_account(account: str) -> None:
    if account not in MAX_LEVERAGE:
        raise ValueError("margin_account must be 'reg_t' (Regulation T, at most 2x overnight) or 'portfolio' "
                         "(portfolio margin, up to 4x)")


def leveraged_note(tickers) -> str | None:
    lev = sorted({str(t).upper() for t in tickers if factor(t) > 1})
    if not lev:
        return None
    return ("Leveraged ETFs: " + ", ".join(f"{t} ({factor(t):g}x)" for t in lev[:8]) + (" ..." if len(lev) > 8 else "")
            + " carry a maintenance margin of the base requirement times the fund's leverage factor, capped at 100% "
              "(FINRA Rule 4210: 25% x 3 = 75% for a 3x fund), and brokers require at least as much to open them. "
              "Borrowing against them is limited accordingly.")


def call_drop(maint_req: float, gross: float, base: float) -> float:
    """The fall in a long book's prices (fraction) that brings equity down to the maintenance requirement: with gross
    exposure G x equity and requirement m x G, equity 1 - G x d meets m x G x (1 - d) at d = (1 - m G) / (G (1 - m))."""
    if gross <= 0:
        return float("inf")
    m = maint_req / gross
    if m >= 1:
        return 0.0
    return max(0.0, (1 - maint_req) / (gross * (1 - m)))


def refuse(why: str, gross_req: float, maint_req: float, leverage: float, base: float, account: str,
           worst: str | None, book: str | None = None, gross: float | None = None) -> None:
    """Raise when the target needs more initial margin than the equity, or its maintenance requirement leaves less
    than MARGIN_BUFFER of the equity above it (a margin call on the first small move).
    gross_req / maint_req: the target's initial / maintenance requirement as a multiple of equity (for the whole
    book). `book`: the positions it describes ("200% TQQQ, -100% SQQQ") when there is more than one; `gross`: the
    book's gross exposure (default: `leverage`)."""
    gross = leverage if gross is None else gross
    if gross_req > 1 + 1e-9:
        cap = MAX_LEVERAGE[account]
        lev_etf = worst and factor(worst) > 1
        if lev_etf and book:
            msg = (f"{why}the positions ({book}) need {gross_req:.0%} of the equity as initial margin: leveraged ETFs "
                   f"need their maintenance margin times their leverage factor per dollar held, long or short "
                   f"(FINRA Rule 4210: {worst} is {factor(worst):g}x, so {maintenance(worst, base):.0%} = the "
                   f"{base:.0%} maintenance margin x {factor(worst):g}). The whole book could be at most "
                   f"{1 / gross_req:.3g} times this size (every weight scaled by {1 / gross_req:.3g}), and less than "
                   "that to leave room above the maintenance requirement.")
        elif lev_etf:
            msg = (f"{why}{worst} is a {factor(worst):g}x leveraged ETF: brokers typically require {maintenance(worst, base):.0%} "
                   f"margin on it (FINRA Rule 4210: the {base:.0%} maintenance margin x {factor(worst):g}), so "
                   f"{leverage:g}x leverage on it needs {gross_req:.0%} of the equity as margin. At most "
                   f"{leverage / gross_req:.3g}x could be opened (and under that to leave room above the maintenance "
                   f"requirement); holding it with no borrowing (1x) already gives {factor(worst):g}x exposure to its index.")
        elif account == "reg_t":
            msg = (f"{why}the target exposure needs {gross_req * cap:.3g}x gross leverage, more than Regulation T allows "
                   "overnight (2x: 50% initial margin). With a portfolio-margin account up to 4x is possible: say "
                   "'with portfolio margin' (margin_account 'portfolio') and a maintenance margin below 1/leverage, "
                   "e.g. 'a 15% maintenance margin'.")
        else:
            msg = f"{why}the target exposure needs {gross_req * cap:.3g}x gross leverage, more than a portfolio-margin account allows (4x)."
        raise ValueError(msg[:1].upper() + msg[1:])
    if base and maint_req > 1 - MARGIN_BUFFER + 1e-12:
        d = call_drop(maint_req, gross, base)
        what = f"the positions ({book})" if book else f"{leverage:g}x" + (f" {worst}" if worst else "")
        safe = (1 - MARGIN_BUFFER) / (maint_req / leverage) if maint_req > 0 else leverage
        lead = f"{why}the" if why else "The"
        raise ValueError(f"{lead} maintenance requirement of {what} is {maint_req:.1%} of the equity "
                         f"(maintenance margin {base:.0%}"
                         + (f"; {worst} is a {factor(worst):g}x leveraged ETF, so {maintenance(worst, base):.0%}"
                            if worst and factor(worst) > 1 else "")
                         + f"): a fall of {d:.1%} would already bring a margin call, which a real account would get on "
                           f"the first day or two. The backtest needs at least {MARGIN_BUFFER:.0%} of the equity above the "
                           f"requirement: use at most about {safe:.3g}x"
                         + (" (scale the weights)" if book else "") + ", a lower maintenance margin, or 'no margin calls' "
                           "(maintenance margin 0).")


def call_scale(equity: float, gross: float, maint_req_value: float, target_gross: float) -> float:
    """Pro-rata scale (0..1) applied to every position in a margin call: down to the lower of the target gross
    exposure and the exposure at which equity is MARGIN_CALL_CUSHION x the maintenance requirement."""
    if gross <= 0 or equity <= 0:
        return 0.0
    k = target_gross * equity / gross
    if maint_req_value > 0:
        k = min(k, equity / (MARGIN_CALL_CUSHION * maint_req_value))
    return max(0.0, min(1.0, k))
