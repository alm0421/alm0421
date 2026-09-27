"""Financial goals planner (Portfolio Visualizer's "Financial Goals"): several goals on one portfolio, each a
contribution or a withdrawal on its own dates, and the probability of meeting each one, from the Monte Carlo
engine (backtester/montecarlo.py).

    python -m backtester goals --weights "VTISIM 60 BNDSIM 40" --balance 250000 --years 35 \\
        --goal "Savings: contribute 20000 a year for 15 years" \\
        --goal "College: withdraw 60000 a year from year 8 to year 11" \\
        --goal "House: withdraw 150000 in year 12" \\
        --goal "Retirement: withdraw 70000 a year from year 20"

Every goal becomes a Monte Carlo cash flow (dollars of today, grown with inflation unless "fixed dollars" /
inflation_adjusted false); all of them run together on the same simulated paths (montecarlo.run: the same return
model, rebalancing, inflation, stress tests and settings as the Monte Carlo page). A withdrawal goal is met on a
path when every one of its payments is paid in full; when a period's withdrawals are more than the balance, the
balance is shared among them in proportion to their amounts (montecarlo.simulate(by_flow=True)), so a later goal
is not starved by an earlier one on the same date. The probability of meeting a goal is the share of paths on
which it is met; the table also gives the median share of it that was funded and its median shortfall. Goals
are in the order of their first payment, which is also the order money is needed.

A goal's year N is year N of the plan (year 1 starts today); a calendar year (2035) is converted counting the
current year as year 1. "in year N" / "once" is a single payment at the start of that year.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

import numpy as np

from . import montecarlo as mc

FREQS = ("once", "yearly", "quarterly", "monthly")


@dataclass
class Goal:
    name: str
    amount: float                   # $ per payment, positive
    kind: str = "withdraw"          # "withdraw" or "contribute"
    start_year: int = 1
    end_year: int | None = None     # None: to the end (a "once" goal: its start year)
    freq: str = "yearly"            # once, yearly, quarterly, monthly
    inflation_adjusted: bool = True

    def check(self, years: int) -> None:
        if self.kind not in ("withdraw", "contribute"):
            raise ValueError(f"Goal '{self.name}': kind is withdraw or contribute.")
        if not self.amount > 0:
            raise ValueError(f"Goal '{self.name}': the amount must be above 0.")
        if self.freq not in FREQS:
            raise ValueError(f"Goal '{self.name}': frequency is one of {', '.join(FREQS)}.")
        if not 1 <= int(self.start_year) <= years:
            raise ValueError(f"Goal '{self.name}': year {self.start_year} is outside the plan's {years} years.")
        if self.end_year is not None and int(self.end_year) < int(self.start_year):
            raise ValueError(f"Goal '{self.name}': it ends (year {self.end_year}) before it starts (year {self.start_year}).")

    def cash_flow(self) -> mc.CashFlow:
        once = self.freq == "once"
        sign = -1.0 if self.kind == "withdraw" else 1.0
        return mc.CashFlow(amount=sign * float(self.amount), freq="yearly" if once else self.freq,
                           inflation_adjusted=bool(self.inflation_adjusted), start_year=int(self.start_year),
                           end_year=int(self.start_year) if once else (int(self.end_year) if self.end_year else None))

    def describe(self) -> str:
        per = {"once": "", "yearly": " a year", "quarterly": " a quarter", "monthly": " a month"}[self.freq]
        when = (f"in year {self.start_year}" if self.freq == "once" else
                f"years {self.start_year}-{self.end_year}" if self.end_year else f"from year {self.start_year}")
        return (f"{self.kind} ${self.amount:,.0f}{per} {when}"
                + ("" if self.inflation_adjusted else " (fixed dollars)"))


def _year(v, this_year: int | None = None) -> int:
    y = int(float(v))
    if y >= 1900:
        y = y - (this_year or date.today().year) + 1
    return y


_GOAL = re.compile(
    r"(?is)^\s*(?:(?P<name>[^:]+):\s*)?(?P<kind>withdraw|spend|take out|need|contribute|add|save|invest|deposit)\s+"
    r"\$?(?P<amt>[\d,]+(?:\.\d+)?)\s*(?P<k>k|m)?\s*"
    r"(?:(?:a|per|each|every)\s+(?P<per>year|quarter|month)|(?P<ly>yearly|annually|quarterly|monthly))?\s*"
    r"(?:(?:in|at)\s+(?:year\s+)?(?P<once>\d{1,4})"
    r"|for\s+(?P<for>\d+)\s+years?(?:\s+from\s+(?:year\s+)?(?P<for_from>\d{1,4}))?"
    r"|(?:from|starting(?:\s+in)?)\s+(?:year\s+)?(?P<from>\d{1,4})(?:\s+(?:to|until|through|-)\s+(?:year\s+)?(?P<to>\d{1,4}))?"
    r"|(?:years?\s+)?(?P<a>\d{1,4})\s*(?:-|to)\s*(?:year\s+)?(?P<b>\d{1,4}))?\s*"
    r"(?P<fixed>,?\s*(?:in\s+)?(?:fixed|nominal)\s+dollars|,?\s*not\s+(?:inflation[- ]adjusted|adjusted\s+for\s+inflation))?\s*$")


def parse_goal(text: str, this_year: int | None = None) -> Goal:
    """'College: withdraw 60000 a year from year 8 to year 11', 'House: withdraw $150k in year 12', 'Savings:
    contribute 20000 a year for 15 years', 'Retirement: withdraw 70000 a year from 2045' -> a Goal."""
    m = _GOAL.match(str(text).strip())
    if not m:
        raise ValueError(f"Could not read the goal '{text}': write e.g. 'College: withdraw 60000 a year from year 8 to "
                         "year 11', 'House: withdraw 150000 in year 12' or 'Savings: contribute 20000 a year for 15 years'.")
    amt = float(m.group("amt").replace(",", "")) * {"k": 1e3, "m": 1e6, None: 1.0}[(m.group("k") or "").lower() or None]
    kind = "withdraw" if m.group("kind").lower() in ("withdraw", "spend", "take out", "need") else "contribute"
    per = (m.group("per") or {"yearly": "year", "annually": "year", "quarterly": "quarter", "monthly": "month"}.get(
        (m.group("ly") or "").lower()) or None)
    freq = {"year": "yearly", "quarter": "quarterly", "month": "monthly", None: "once"}[per]
    s_y, e_y = 1, None
    if m.group("once"):
        s_y = _year(m.group("once"), this_year)
        if per:
            e_y = s_y
    elif m.group("for"):
        s_y = _year(m.group("for_from"), this_year) if m.group("for_from") else 1
        e_y = s_y + int(m.group("for")) - 1
    elif m.group("from"):
        s_y = _year(m.group("from"), this_year)
        e_y = _year(m.group("to"), this_year) if m.group("to") else None
    elif m.group("a"):
        s_y, e_y = _year(m.group("a"), this_year), _year(m.group("b"), this_year)
    if freq == "once" and not m.group("once"):
        freq = "yearly" if (e_y or m.group("for") or m.group("from")) else "once"
    name = (m.group("name") or "").strip() or f"{kind.capitalize()} ${amt:,.0f}"
    return Goal(name=name[:60], amount=amt, kind=kind, start_year=s_y, end_year=e_y, freq=freq,
                inflation_adjusted=not m.group("fixed"))


def goal_from_dict(d: dict, this_year: int | None = None) -> Goal:
    if isinstance(d, str):
        return parse_goal(d, this_year)
    try:
        amt = float(d.get("amount") or 0)
        s_y = _year(d.get("start_year") or d.get("year") or 1, this_year)
        e_raw = d.get("end_year")
        e_y = _year(e_raw, this_year) if e_raw not in (None, "") else None
    except (TypeError, ValueError):
        raise ValueError(f"Goal '{d.get('name') or '?'}': amounts and years are numbers.")
    return Goal(name=str(d.get("name") or "").strip()[:60] or f"Goal", amount=abs(amt),
                kind=str(d.get("kind") or ("contribute" if amt < 0 else "withdraw")),
                start_year=s_y, end_year=e_y, freq=str(d.get("freq") or ("once" if e_y is None and d.get("year") else "yearly")),
                inflation_adjusted=d.get("inflation_adjusted", True) not in (False, "false", 0))


def run(s: mc.Settings, goals: list[Goal]) -> dict:
    """The Monte Carlo run of the settings with the goals as its cash flows, plus each goal's probability of being
    met. Any cash flows already in the settings run too (as unnamed flows)."""
    if not goals:
        raise ValueError("Add at least one goal (a contribution or a withdrawal with its years).")
    if len(goals) > 20:
        raise ValueError("Up to 20 goals.")
    for g in goals:
        g.check(int(s.years))
    goals = sorted(goals, key=lambda g: (g.start_year, g.kind != "contribute"))
    base = list(s.flows or [])
    s.flows = base + [g.cash_flow() for g in goals]
    s.keep_paths = True
    R = mc.run(s)
    paths = R.pop("_paths")
    P, cum_infl = paths["P"], paths["cum_infl"]
    SIM = mc.simulate(P, cum_infl, s.start_balance, s.flows, by_flow=True)
    k0 = len(base)
    rows = []
    for j, g in enumerate(goals):
        req, paid, short = SIM["flow_requested"][k0 + j], SIM["flow_paid"][k0 + j], SIM["flow_short"][k0 + j]
        row = {"name": g.name, "kind": g.kind, "describe": g.describe(), "start_year": g.start_year,
               "end_year": g.end_year if g.freq != "once" else g.start_year, "freq": g.freq, "amount": g.amount,
               "inflation_adjusted": g.inflation_adjusted}
        if g.kind == "withdraw":
            with np.errstate(divide="ignore", invalid="ignore"):
                funded = np.where(req > 0, paid / req, 1.0)
            row.update(probability=float((~short).mean()), funded_median=float(np.median(funded)),
                       funded_p10=float(np.percentile(funded, 10)),
                       requested_median=float(np.median(req)), shortfall_median=float(np.median(req - paid)),
                       shortfall_mean=float((req - paid).mean()))
        else:
            row.update(probability=None, requested_median=float(np.median(req)))
        rows.append(row)
    # every withdrawal goal met on the same path
    wd = [k0 + j for j, g in enumerate(goals) if g.kind == "withdraw"]
    all_met = float((~SIM["flow_short"][wd].any(axis=0)).mean()) if wd else None
    R["goals"] = rows
    R["all_goals_met"] = all_met
    R["notes"] = list(R.get("notes") or []) + [
        "Goals: every goal is a cash flow on the same simulated paths (in today's dollars, grown with inflation "
        "unless marked fixed). A withdrawal goal is met on a path when all of its payments are paid in full; when "
        "the balance cannot cover a period's withdrawals it is shared among them in proportion to their amounts."]
    return R


def console(R: dict) -> str:
    L = [mc.console(R), "", "Financial goals (probability of meeting each one; median share funded and shortfall on the "
                            "paths, in nominal dollars):"]
    w = max([12] + [len(g["name"]) + 1 for g in R["goals"]])
    d = max([10] + [len(g["describe"]) + 1 for g in R["goals"]])
    L.append(f"{'Goal':<{w}s} {'what':<{d}s} {'chance met':>10s} {'funded (median)':>16s} {'shortfall (median)':>19s}")
    for g in R["goals"]:
        if g["kind"] == "withdraw":
            L.append(f"{g['name']:<{w}s} {g['describe']:<{d}s} {g['probability'] * 100:>9.1f}% "
                     f"{g['funded_median'] * 100:>15.1f}% {('$' + format(g['shortfall_median'], ',.0f')):>19s}")
        else:
            L.append(f"{g['name']:<{w}s} {g['describe']:<{d}s} {'(adds money)':>10s}")
    if R.get("all_goals_met") is not None:
        L.append(f"Every withdrawal goal met on the same path: {R['all_goals_met'] * 100:.1f}%")
    return "\n".join(L)
