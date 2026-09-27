"""Curated explanations of one-day price moves beyond +60% / -60% that are real (or known junk series), with the
evidence: the price-flag scan (backtester/price_flags.py) counts a day listed here as explained, so backtests holding
across it get no data-quality warning. Anything the automatic checks cannot explain and is not listed here is flagged
(data/price_flags.json) instead; for the core universe (Nasdaq-100 members past and present, the core ETFs) the test
suite requires every such day to be explained - by a repair, a corporate action, a security break or an entry here.

WHITELIST_TICKERS: whole series (junk quote series); WHITELIST_BEFORE: every day before a date (tick-era quotes);
WHITELIST: single days, (ticker, date) -> the evidence."""
from __future__ import annotations

# one-day total-return moves beyond +60% / -60% that are real (or are known junk series), with the evidence
WHITELIST_TICKERS = {
    "CPWR": "an OTC quote series of a delisted symbol (mostly zero volume, prices in cents): the quality filter "
            "never treats it as a member, and its jumps are single trades",
    "SSCC": "an OTC quote series after the company left Nasdaq (zero volume on most days, prices of $100s-$1000s)",
}
WHITELIST_BEFORE = {"HANS": "1993", "MNST": "1993", "CECO": "1990", "SIRI": "2004",   # sub-penny / 1/8-tick era quotes
                    # InterActive Group, the OTC shell Arrowhead reverse-merged into in January 2004 (1-for-65, booked
                    # 2004-01-15): 1/16-tick quotes x 650 after the splits, volumes of 0-10 (split-adjusted) shares
                    # a day, the price hopping between a few tick levels; still a handful of trades a day to March 2004
                    "ARWR": "2004-04"}
# the S&P 1500 additions of 2026-09 (volume multiples: the day's volume / the median of the 10 sessions before)
WHITELIST_SP1500 = {
    ("AN", "1995-05-22"): "Republic Industries (now AutoNation), Huizenga's investment: volume x64, the level held",
    ("ARWR", "2010-03-22"): "Arrowhead news day: volume x250 (2.6M shares against ~10,000), then fading",
    ("ARWR", "2016-11-30"): "ARC-520 program discontinued after the FDA clinical hold: -67%, volume x6",
    ("CAR", "2009-03-20"): "Avis Budget at $0.55 in the March 2009 low: +98%, volume x7.5",
    ("CAR", "2021-11-02"): "the Avis short squeeze (EV fleet news; +108%, a 545 intraday high), volume x19",
    ("CELH", "2008-10-20"): "Celsius as a thin OTC penny stock ($0.20 -> $0.47 in 1/15-cent ticks), volume x4",
    ("CHRD", "2020-03-09"): "Oasis Petroleum (now Chord) in the 2020-03-09 oil crash (GUSH -81%), volume x5",
    ("CHRD", "2020-03-13"): "Oasis at $0.37: March 2020 oil-crash rebound (SPY +8.5% that day)",
    ("CHRD", "2020-04-23"): "Oasis penny stock before its Chapter 11: volume x14",
    ("CHRD", "2020-06-05"): "Oasis in the June 2020 bankrupt-stock rally: volume x14",
    ("CHRD", "2020-06-08"): "same rally, volume x16",
    ("CHRD", "2020-11-11"): "Oasis old shares at $0.09-0.17 ahead of the reorganisation, volume x10",
    ("CLH", "2001-10-22"): "Clean Harbors news day: volume x500 (7.4M shares against ~15,000), the level mostly held",
    ("CNO", "2008-09-18"): "Conseco in the Lehman week: -42% then +81%, heavy volume both days",
    ("CORT", "2025-03-31"): "relacorilant ROSELLA ovarian-cancer trial success: volume x19",
    ("CYTK", "2014-04-25"): "tirasemtiv BENEFIT-ALS trial missed its primary endpoint: -65%, volume x13",
    ("CYTK", "2023-12-27"): "aficamten SEQUOIA-HCM trial success: volume x9",
    ("DAR", "1999-11-08"): "Darling as a thin penny stock ($1.25 -> $2.13 on 22,900 shares, 1/16 ticks)",
    ("DAR", "2000-09-21"): "thin penny stock in 1/16 ticks: a $0.25 print the day before, back to $0.69",
    ("EEFT", "1998-09-08"): "Euronet a year after its IPO at $2-3 in 1/16 ticks; SPY +5.4% that day",
    ("EEFT", "1998-11-25"): "Euronet at $2-4 in 1/16 ticks: volume x7, the level held",
    ("EHC", "2003-03-26"): "HealthSouth's first trade after the SEC fraud charges and trading halt: -97%, volume x120",
    ("EHC", "2003-07-07"): "HealthSouth on the pink sheets during its restructuring: +110%, volume x7",
    ("FLR", "2020-03-19"): "Fluor at $3.40 in the March 2020 crash: +76% rebound",
    ("GME", "2021-01-26"): "GameStop short squeeze", ("GME", "2021-01-27"): "GameStop short squeeze",
    ("GME", "2021-01-29"): "GameStop short squeeze", ("GME", "2021-02-24"): "GameStop second squeeze, volume x6",
    ("GME", "2024-05-13"): "Roaring Kitty's return, volume x8", ("GME", "2024-05-14"): "Roaring Kitty, volume x8",
    ("GPK", "2008-10-10"): "Graphic Packaging at $1.25 in the October 2008 crash: +65% rebound",
    ("HXL", "1993-12-08"): "Hexcel's Chapter 11 filing (December 1993): -60%, volume x80",
    ("IDCC", "1999-12-10"): "InterDigital in the 1999 wireless-stock mania: volume x6",
    ("IDCC", "1999-12-29"): "InterDigital, same mania: +112% to a 55 intraday high, volume x2 then higher",
    ("PCG", "2019-01-24"): "Cal Fire found PG&E not responsible for the 2017 Tubbs fire: +75%",
    ("PNW", "1989-12-07"): "Pinnacle West rebounding from its MeraBank-crisis low: volume x15, the level held",
    ("PWR", "2002-07-02"): "Quanta Services cut its earnings forecast: -68%, volume x11",
    ("WMB", "2002-07-22"): "Williams' 2002 liquidity crisis (dividend cut, credit downgrade): -61%, volume x10",
    ("WMB", "2002-07-29"): "Williams' rebound from its $0.78 low of 2002-07-25 (+101%)",
}

WHITELIST = {
    ("CONL", "2024-11-06"): "2x COIN on election day (COIN +31%)",
    ("CWEB", "2022-03-16"): "2x KWEB; China internet +40% on the State Council support pledge",
    ("DRIP", "2020-03-09"): "2x inverse XOP on the 2020-03-09 oil crash (GUSH -81% the same day)",
    ("GUSH", "2020-03-09"): "2x XOP on the 2020-03-09 oil crash",
    ("ERX", "2020-03-09"): "then 3x XLE on the oil crash (ERY +60% the same day)",
    ("ERY", "2020-03-09"): "then 3x inverse XLE on the oil crash",
    ("ETHE", "2019-06-20"): "first week of OTC quotation (a stale placeholder price before)",
    ("ETHE", "2019-06-21"): "first week of OTC quotation", ("ETHE", "2019-06-24"): "first week of OTC quotation",
    ("ETHE", "2019-06-25"): "first week of OTC quotation",
    ("FOSL", "2018-02-14"): "earnings, volume x6", ("FOSL", "2021-01-27"): "meme squeeze, volume x14",
    ("FOSL", "2024-12-02"): "turnaround news, volume x100",
    ("IDXX", "1997-03-24"): "earnings warning, volume x139", ("INSM", "2002-09-10"): "trial failure, volume x144",
    ("INSM", "2017-09-05"): "phase 3 success", ("INSM", "2024-05-28"): "ASPEN trial success",
    ("JDST", "2020-03-12"): "3x inverse junior gold miners, March 2020 (JNUG moves the other way)",
    ("JDST", "2020-03-17"): "March 2020", ("JNUG", "2020-03-12"): "March 2020", ("JNUG", "2020-03-13"): "March 2020",
    ("JNUG", "2020-03-16"): "March 2020 (JDST -59% the same day)", ("JNUG", "2020-03-18"): "March 2020",
    ("LILAK", "2015-07-01"): "first days of the LiLAC tracking stock (when-issued)",
    ("MRNA", "2026-08-19"): "news day: volume x46, a 54% intraday range, not a split (volume would fall)",
    ("MS", "2008-10-13"): "Mitsubishi UFJ investment closed", ("MSTR", "2000-03-20"): "accounting restatement",
    ("MSTR", "2001-04-19"): "dot-com rebound", ("REGN", "2000-02-23"): "genomics rally",
    ("SIRI", "2002-08-15"): "heavy trading both days (Sirius's 2002 recapitalisation)",
    ("SVXY", "2018-02-06"): "volmageddon (-83%; volume did not follow a split)",
    ("TSCO", "1994-02-18"): "IPO day: the first row is a pre-listing placeholder",
    ("UAL", "2008-07-22"): "earnings / oil drop (+68%)", ("UAUA", "2008-07-22"): "same company as UAL",
    ("UVIX", "2024-08-05"): "VIX spike of 2024-08-05", ("UVXY", "2018-02-05"): "volmageddon",
    ("VIP", "2021-03-22"): "the symbol changed hands: another listing's first day",
    ("VIP", "2023-01-12"): "thin, recycled symbol", ("VRTX", "2013-04-19"): "cystic fibrosis data",
    ("WEBS", "2020-03-16"): "3x inverse dot-com on 2020-03-16", ("YANG", "2022-03-16"): "China rally (YINN +65%)",
    ("YINN", "2022-03-16"): "China rally",
    **WHITELIST_SP1500,
    # former Nasdaq-100 members rebuilt from archives (data/delisted_sources.json)
    ("BBBY-2023", "2021-06-02"): "meme squeeze (Bed Bath & Beyond), volume x10",
    ("BBBY-2023", "2023-01-11"): "meme rally during the bankruptcy warning",
    ("BBBY-2023", "2023-02-06"): "meme rally on the Hudson Bay equity offering",
    ("CTXS", "1997-05-12"): "Microsoft licensing deal announced (Citrix +69%)",
    ("ENDP", "2022-06-28"): "penny stock ahead of the Chapter 11 filing ($0.38 to $0.71)",
    ("GMCR", "2015-12-07"): "JAB Holding buyout at $92 a share",
    ("JAVA-2010", "2009-03-18"): "reports of IBM takeover talks (Sun Microsystems +79%)",
    ("SUNW", "2009-03-18"): "same company as JAVA-2010: IBM takeover talks",
}
