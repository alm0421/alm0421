"""Mortality for the Monte Carlo "until death" horizon: the US Social Security period life table.

Source: Social Security Administration, Office of the Chief Actuary, "Actuarial Life Table" - the 2023
period life table for the Social Security area population, as used in the 2026 Trustees Report
(https://www.ssa.gov/oact/STATS/table4c6.html, retrieved September 2026). QX_* is the probability of
dying within one year at each exact age 0-119; LE_* the period life expectancy (years) the SSA publishes,
kept to check the calculation. A period table uses one year's death rates for the whole remaining life (no
future mortality improvement), so it slightly understates how long people now alive will live.

Joint (a couple): the money has to last until the second death; the two lives are treated as independent,
so P(at least one alive after k years) = 1 - (1 - S1(k)) (1 - S2(k)).
"""
from __future__ import annotations

import numpy as np

SOURCE = ("SSA 2023 period life table (Social Security area population), as used in the 2026 Trustees Report; "
          "ssa.gov/oact/STATS/table4c6.html")
TABLE_YEAR = 2023
MAX_AGE = 119

QX_MALE = (
    0.006015, 0.000479, 0.000320, 0.000249, 0.000194, 0.000159, 0.000137, 0.000125, 0.000120, 0.000120,  # ages 0-9
    0.000125, 0.000140, 0.000173, 0.000233, 0.000327, 0.000463, 0.000634, 0.000819, 0.000999, 0.001138,  # ages 10-19
    0.001235, 0.001315, 0.001378, 0.001439, 0.001509, 0.001595, 0.001685, 0.001783, 0.001876, 0.001970,  # ages 20-29
    0.002085, 0.002202, 0.002308, 0.002407, 0.002490, 0.002577, 0.002665, 0.002764, 0.002864, 0.002987,  # ages 30-39
    0.003115, 0.003253, 0.003419, 0.003600, 0.003777, 0.003931, 0.004073, 0.004245, 0.004477, 0.004795,  # ages 40-49
    0.005126, 0.005496, 0.005917, 0.006404, 0.006923, 0.007491, 0.008173, 0.008938, 0.009714, 0.010494,  # ages 50-59
    0.011337, 0.012232, 0.013196, 0.014229, 0.015316, 0.016455, 0.017574, 0.018735, 0.019981, 0.021366,  # ages 60-69
    0.022903, 0.024615, 0.026504, 0.028648, 0.031071, 0.033802, 0.037010, 0.041158, 0.045461, 0.050346,  # ages 70-79
    0.055633, 0.061757, 0.068358, 0.075420, 0.083364, 0.092680, 0.103459, 0.115502, 0.129018, 0.143810,  # ages 80-89
    0.159458, 0.176551, 0.195360, 0.216286, 0.238799, 0.262268, 0.286291, 0.310944, 0.332325, 0.349036,  # ages 90-99
    0.366568, 0.384960, 0.404252, 0.424488, 0.445712, 0.467998, 0.491398, 0.515968, 0.541766, 0.568854,  # ages 100-109
    0.597297, 0.627162, 0.658520, 0.691446, 0.726018, 0.762319, 0.800435, 0.840457, 0.882480, 0.926604,  # ages 110-119
)

QX_FEMALE = (
    0.005125, 0.000392, 0.000229, 0.000188, 0.000155, 0.000133, 0.000115, 0.000105, 0.000100, 0.000098,  # ages 0-9
    0.000101, 0.000111, 0.000126, 0.000152, 0.000188, 0.000229, 0.000273, 0.000323, 0.000372, 0.000410,  # ages 10-19
    0.000441, 0.000476, 0.000513, 0.000546, 0.000582, 0.000609, 0.000641, 0.000683, 0.000740, 0.000808,  # ages 20-29
    0.000878, 0.000947, 0.001018, 0.001089, 0.001154, 0.001209, 0.001263, 0.001347, 0.001438, 0.001533,  # ages 30-39
    0.001643, 0.001742, 0.001845, 0.001954, 0.002075, 0.002187, 0.002306, 0.002438, 0.002595, 0.002791,  # ages 40-49
    0.003030, 0.003288, 0.003554, 0.003847, 0.004172, 0.004532, 0.004923, 0.005365, 0.005815, 0.006333,  # ages 50-59
    0.006923, 0.007555, 0.008220, 0.008881, 0.009514, 0.010188, 0.010880, 0.011659, 0.012543, 0.013581,  # ages 60-69
    0.014769, 0.016153, 0.017705, 0.019495, 0.021533, 0.023846, 0.026458, 0.029700, 0.033135, 0.036982,  # ages 70-79
    0.041183, 0.045959, 0.051282, 0.057262, 0.064107, 0.071752, 0.080490, 0.090566, 0.102204, 0.115178,  # ages 80-89
    0.129176, 0.144229, 0.160353, 0.177635, 0.196502, 0.216846, 0.238750, 0.261359, 0.283899, 0.306491,  # ages 90-99
    0.329680, 0.353333, 0.377300, 0.401416, 0.425501, 0.451031, 0.478092, 0.506778, 0.537185, 0.568854,  # ages 100-109
    0.597297, 0.627162, 0.658520, 0.691446, 0.726018, 0.762319, 0.800435, 0.840457, 0.882480, 0.926604,  # ages 110-119
)

LE_MALE = (
    75.79, 75.25, 74.28, 73.31, 72.33, 71.34, 70.35, 69.36, 68.37, 67.38,  # ages 0-9
    66.39, 65.39, 64.40, 63.41, 62.43, 61.45, 60.48, 59.51, 58.56, 57.62,  # ages 10-19
    56.69, 55.76, 54.83, 53.90, 52.98, 52.06, 51.14, 50.23, 49.32, 48.41,  # ages 20-29
    47.50, 46.60, 45.70, 44.81, 43.91, 43.02, 42.13, 41.24, 40.36, 39.47,  # ages 30-39
    38.59, 37.71, 36.83, 35.95, 35.08, 34.21, 33.34, 32.48, 31.62, 30.76,  # ages 40-49
    29.90, 29.05, 28.21, 27.38, 26.55, 25.73, 24.92, 24.12, 23.34, 22.56,  # ages 50-59
    21.79, 21.04, 20.29, 19.56, 18.83, 18.12, 17.41, 16.71, 16.02, 15.34,  # ages 60-69
    14.66, 14.00, 13.34, 12.69, 12.05, 11.42, 10.80, 10.19, 9.61, 9.04,  # ages 70-79
    8.50, 7.97, 7.46, 6.97, 6.50, 6.04, 5.61, 5.20, 4.81, 4.45,  # ages 80-89
    4.11, 3.80, 3.50, 3.23, 2.99, 2.77, 2.58, 2.41, 2.27, 2.15,  # ages 90-99
    2.04, 1.93, 1.83, 1.72, 1.63, 1.54, 1.45, 1.36, 1.28, 1.20,  # ages 100-109
    1.13, 1.05, 0.98, 0.92, 0.85, 0.79, 0.74, 0.68, 0.63, 0.58,  # ages 110-119
)

LE_FEMALE = (
    81.06, 80.48, 79.51, 78.53, 77.54, 76.55, 75.56, 74.57, 73.58, 72.59,  # ages 0-9
    71.59, 70.60, 69.61, 68.62, 67.63, 66.64, 65.66, 64.67, 63.69, 62.72,  # ages 10-19
    61.74, 60.77, 59.80, 58.83, 57.86, 56.90, 55.93, 54.97, 54.00, 53.04,  # ages 20-29
    52.08, 51.13, 50.18, 49.23, 48.28, 47.34, 46.39, 45.45, 44.51, 43.58,  # ages 30-39
    42.64, 41.71, 40.78, 39.86, 38.93, 38.01, 37.10, 36.18, 35.27, 34.36,  # ages 40-49
    33.45, 32.55, 31.66, 30.77, 29.89, 29.01, 28.14, 27.28, 26.42, 25.57,  # ages 50-59
    24.73, 23.90, 23.08, 22.27, 21.46, 20.66, 19.87, 19.08, 18.30, 17.53,  # ages 60-69
    16.76, 16.01, 15.26, 14.53, 13.81, 13.10, 12.41, 11.73, 11.08, 10.44,  # ages 70-79
    9.82, 9.22, 8.64, 8.08, 7.54, 7.02, 6.53, 6.05, 5.61, 5.19,  # ages 80-89
    4.80, 4.44, 4.10, 3.79, 3.50, 3.23, 2.99, 2.77, 2.57, 2.39,  # ages 90-99
    2.23, 2.08, 1.94, 1.82, 1.70, 1.59, 1.48, 1.38, 1.29, 1.20,  # ages 100-109
    1.13, 1.05, 0.98, 0.92, 0.85, 0.79, 0.74, 0.68, 0.63, 0.58,  # ages 110-119
)


def _qx(sex: str) -> np.ndarray:
    s = str(sex).strip().lower()
    if s in ("m", "male", "man"):
        return np.asarray(QX_MALE, float)
    if s in ("f", "female", "woman"):
        return np.asarray(QX_FEMALE, float)
    raise ValueError("sex must be male or female")


def survival(age: float, sex: str, years: int) -> np.ndarray:
    """S[k] = probability that a person of exact age `age` (whole years; a fraction is dropped) is alive k
    years later, k = 0..years. Past age 119 nobody survives."""
    a = int(np.floor(float(age)))
    if not 0 <= a <= MAX_AGE:
        raise ValueError(f"Age must be between 0 and {MAX_AGE}.")
    q = _qx(sex)
    qs = np.ones(years)
    n = max(0, min(years, MAX_AGE + 1 - a))
    qs[:n] = q[a:a + n]
    return np.concatenate([[1.0], np.cumprod(1 - qs)])


def joint_survival(age1: float, sex1: str, age2: float, sex2: str, years: int) -> np.ndarray:
    """Probability that at least one of two independent lives is still alive k years later."""
    s1, s2 = survival(age1, sex1, years), survival(age2, sex2, years)
    return 1 - (1 - s1) * (1 - s2)


def life_expectancy(S: np.ndarray) -> float:
    """Expected remaining years from a survival curve (deaths spread evenly within each year)."""
    S = np.asarray(S, float)
    return float(S[1:].sum() + 0.5 * (S[0] - S[-1])) if len(S) > 1 else 0.0


def horizon_years(S_full: np.ndarray, tail: float = 0.001, cap: int = 100) -> int:
    """Years until the survival probability falls below `tail` (at most `cap`)."""
    below = np.nonzero(S_full < tail)[0]
    return int(min(cap, below[0] if len(below) else len(S_full) - 1))
