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
(the broker would not let you open it), or when its maintenance requirement at the target is not below the equity
(every down close would be a margin call).

Margin calls: at a close where equity is below the total maintenance requirement, every position is cut pro rata to
the LOWER of (a) the target exposure and (b) the exposure at which equity is MARGIN_CALL_CUSHION (125%) of the
maintenance requirement. A broker's liquidation restores a cushion rather than stopping exactly at the line; without
one, a target that sits just inside the maintenance limit would be cut straight back to the edge and called again on
the next down close.
"""
from __future__ import annotations

MAX_LEVERAGE = {"reg_t": 2.0, "portfolio": 4.0}
MARGIN_CALL_CUSHION = 1.25

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


def refuse(why: str, gross_req: float, maint_req: float, leverage: float, base: float, account: str,
           worst: str | None) -> None:
    """Raise when the target needs more initial margin than the equity, or its maintenance is not below it.
    gross_req / maint_req: the target's initial / maintenance requirement as a multiple of equity."""
    if gross_req > 1 + 1e-9:
        cap = MAX_LEVERAGE[account]
        lev_etf = worst and factor(worst) > 1
        if lev_etf:
            msg = (f"{why}{worst} is a {factor(worst):g}x leveraged ETF: brokers typically require {maintenance(worst, base):.0%} "
                   f"margin on it (FINRA Rule 4210: the {base:.0%} maintenance margin x {factor(worst):g}), so "
                   f"{leverage:g}x leverage on it needs {gross_req:.0%} of the equity as margin. At most "
                   f"{leverage / gross_req:.3g}x could be opened (and under that to leave room above the maintenance requirement); holding it with no borrowing (1x) already gives "
                   f"{factor(worst):g}x exposure to its index.")
        elif account == "reg_t":
            msg = (f"{why}the target exposure needs {gross_req * cap:.3g}x gross leverage, more than Regulation T allows "
                   "overnight (2x: 50% initial margin). With a portfolio-margin account up to 4x is possible: say "
                   "'with portfolio margin' (margin_account 'portfolio') and a maintenance margin below 1/leverage, "
                   "e.g. 'a 15% maintenance margin'.")
        else:
            msg = f"{why}the target exposure needs {gross_req * cap:.3g}x gross leverage, more than a portfolio-margin account allows (4x)."
        raise ValueError(msg)
    if base and maint_req >= 1 - 1e-9:
        raise ValueError(f"{why}the maintenance requirement at the target exposure is {maint_req:.0%} of the equity "
                         f"(maintenance margin {base:.0%}"
                         + (f"; {worst} is a {factor(worst):g}x leveraged ETF" if worst and factor(worst) > 1 else "")
                         + "), so every close below the entry would be a margin call. Use less leverage or a lower "
                           "maintenance margin (0 turns margin calls off).")


def call_scale(equity: float, gross: float, maint_req_value: float, target_gross: float) -> float:
    """Pro-rata scale (0..1) applied to every position in a margin call: down to the lower of the target gross
    exposure and the exposure at which equity is MARGIN_CALL_CUSHION x the maintenance requirement."""
    if gross <= 0 or equity <= 0:
        return 0.0
    k = target_gross * equity / gross
    if maint_req_value > 0:
        k = min(k, equity / (MARGIN_CALL_CUSHION * maint_req_value))
    return max(0.0, min(1.0, k))
