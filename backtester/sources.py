"""Parsers for downloaded factor files (pure pandas, no network): Kenneth French's data-library CSVs and
AQR's data-set spreadsheets. scripts/fetch_data.py downloads the files in the GitHub Action and uses these
to write data/factors/*.csv; the tests feed them synthetic samples in the same layouts.

Kenneth French (https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html), one CSV per zip:
    a few lines of description, then a header line that starts with a comma (",Mkt-RF,SMB,HML,RF" or
    ",WML" / ",Mom"), then rows "YYYYMM,..." (monthly files) or "YYYYMMDD,..." (daily files) in percent,
    -99.99 or -999 for missing. Monthly files continue with an "Annual Factors: January-December" section
    (rows "YYYY,...") that is skipped.
AQR (https://www.aqr.com/Insights/Datasets), e.g. Quality-Minus-Junk-Factors-Monthly.xlsx and
    Betting-Against-Beta-Equity-Factors-Monthly.xlsx: the first sheet ("QMJ Factors" / "BAB Factors") has
    about 18 lines of notes, a row " | EQUITIES | ... | Aggregate Equity Portfolios", then the header row
    "DATE | AUS | AUT | ... | USA | Global | Global Ex USA | Europe | North America | Pacific" and one row per
    month ("07/31/1957" or an Excel date) of decimal returns shown as percentages; cells are blank before a
    country's series starts.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

# (local name, file on French's site). Monthly for the official monthly regressions, daily for daily ones.
FRENCH_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FRENCH_FILES = [
    # US, daily (the existing downloads) and the official monthly files
    ("ff3_daily", "F-F_Research_Data_Factors_daily_CSV.zip"),
    ("ff5_daily", "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"),
    ("mom_daily", "F-F_Momentum_Factor_daily_CSV.zip"),
    ("port6_daily", "6_Portfolios_2x3_daily_CSV.zip"),
    ("dev_ff3_daily", "Developed_ex_US_3_Factors_Daily_CSV.zip"),
    ("em_ff5_monthly", "Emerging_5_Factors_CSV.zip"),
    ("ff3_monthly", "F-F_Research_Data_Factors_CSV.zip"),
    ("ff5_monthly", "F-F_Research_Data_5_Factors_2x3_CSV.zip"),
    ("mom_monthly", "F-F_Momentum_Factor_CSV.zip"),
    ("emerging_mom_monthly", "Emerging_MOM_Factor_CSV.zip"),
]
# international regions: local prefix -> French's file prefix (3 factors, 5 factors, momentum; monthly and daily)
FRENCH_REGIONS = {
    "developed": "Developed",
    "developed_ex_us": "Developed_ex_US",
    "europe": "Europe",
    "japan": "Japan",
    "asia_pacific_ex_japan": "Asia_Pacific_ex_Japan",
    "north_america": "North_America",
}
for _loc, _fr in FRENCH_REGIONS.items():
    # French spells Asia Pacific's momentum file "MOM", the others "Mom"
    _mom = "MOM" if _loc == "asia_pacific_ex_japan" else "Mom"
    FRENCH_FILES += [(f"{_loc}_ff3_monthly", f"{_fr}_3_Factors_CSV.zip"),
                     (f"{_loc}_ff5_monthly", f"{_fr}_5_Factors_CSV.zip"),
                     (f"{_loc}_mom_monthly", f"{_fr}_{_mom}_Factor_CSV.zip"),
                     (f"{_loc}_ff3_daily", f"{_fr}_3_Factors_Daily_CSV.zip"),
                     (f"{_loc}_ff5_daily", f"{_fr}_5_Factors_Daily_CSV.zip"),
                     (f"{_loc}_mom_daily", f"{_fr}_{_mom}_Factor_Daily_CSV.zip")]

AQR_BASE = "https://www.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/"
AQR_FILES = [("aqr_qmj_monthly", "Quality-Minus-Junk-Factors-Monthly.xlsx", "QMJ Factors"),
             ("aqr_bab_monthly", "Betting-Against-Beta-Equity-Factors-Monthly.xlsx", "BAB Factors")]

MOM_NAMES = {"WML", "MOM", "UMD", "MOMENTUM"}


def parse_french_csv(text: str) -> pd.DataFrame:
    """The first table of a French data-library CSV (the monthly or daily section) as a DataFrame with a
    'date' column (YYYY-MM-DD; monthly rows dated at the month end) and decimal returns. The momentum
    column is called 'Mom' whatever the file calls it (WML in the international files)."""
    raw = text.replace("\r", "").split("\n")
    start = None
    for i, line in enumerate(raw):
        s = line.strip()
        if re.match(r"^,", s) or s.lower().startswith(",mkt") or ("Mkt-RF" in s and "," in s):
            start = i
            break
    if start is None:
        raise ValueError("no header line (a line starting with a comma) found")
    header = [h.strip() for h in raw[start].split(",")]
    cols = header[1:]
    rows = []
    for line in raw[start + 1:]:
        parts = [x.strip() for x in line.split(",")]
        if len(parts) != len(header) or not re.fullmatch(r"\d{8}|\d{6}", parts[0]):
            if rows:
                break
            continue
        rows.append(parts)
    if not rows:
        raise ValueError("no data rows after the header")
    df = pd.DataFrame(rows, columns=["date"] + cols)
    fmt = "%Y%m%d" if len(df["date"].iloc[0]) == 8 else "%Y%m"
    dt = pd.to_datetime(df["date"], format=fmt)
    if fmt == "%Y%m":
        dt = dt + pd.offsets.MonthEnd(0)
    df["date"] = dt.dt.strftime("%Y-%m-%d")
    for c in cols:
        v = pd.to_numeric(df[c], errors="coerce")
        df[c] = v.where(v > -99) / 100.0
    df = df.rename(columns={c: "Mom" for c in cols if c.upper() in MOM_NAMES})
    return df


def _pct(v):
    """A spreadsheet cell as a decimal return: numbers as they are, '1.12%' strings divided by 100."""
    if isinstance(v, str):
        s = v.strip().replace(",", "")
        if not s:
            return np.nan
        try:
            return float(s[:-1]) / 100 if s.endswith("%") else float(s)
        except ValueError:
            return np.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def parse_aqr_sheet(raw: pd.DataFrame) -> pd.DataFrame:
    """An AQR factor sheet read with header=None (all cells) -> DataFrame with 'date' (YYYY-MM-DD, month end)
    and one decimal-return column per country/aggregate that has data (USA, Global, Global Ex USA, Europe,
    North America, Pacific, JPN, ...)."""
    hdr = None
    for i in range(min(len(raw), 200)):
        if str(raw.iat[i, 0]).strip().upper() == "DATE":
            hdr = i
            break
    if hdr is None:
        raise ValueError("no DATE header row in the AQR sheet")
    names = [str(x).strip() for x in raw.iloc[hdr]]
    body = raw.iloc[hdr + 1:]
    dates = pd.to_datetime(body.iloc[:, 0].map(lambda x: x if not isinstance(x, str) else x.strip()),
                           format="mixed", errors="coerce") if len(body) else pd.Series(dtype="datetime64[ns]")
    keep = dates.notna().to_numpy()
    body, dates = body[keep], dates[keep]
    out = {"date": (pd.DatetimeIndex(dates) + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")}
    for j, n in enumerate(names[1:], start=1):
        if not n or n.lower() == "nan":
            continue
        v = body.iloc[:, j].map(_pct).astype(float).to_numpy()
        if np.isfinite(v).any():
            out[n] = v
    df = pd.DataFrame(out)
    if len(df.columns) < 2:
        raise ValueError("no return columns in the AQR sheet")
    # a sheet stored in percent units (whole numbers) instead of decimals
    num = df.drop(columns="date")
    if np.nanmedian(np.abs(num.to_numpy())) > 0.5:
        df[num.columns] = num / 100.0
    return df.drop_duplicates("date", keep="last").reset_index(drop=True)
