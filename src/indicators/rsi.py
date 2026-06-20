"""
rsi.py
======
Relative Strength Index (RSI) implementation built from scratch using
pandas and NumPy — no third-party technical analysis libraries required.

Mathematical background
-----------------------
RSI was introduced by J. Welles Wilder in *New Concepts in Technical Trading
Systems* (1978).  The calculation uses a modified exponential average
(Wilder's smoothing) to dampen noise.

::

    Delta_t  = Close_t − Close_{t-1}

    Gain_t   = max(Delta_t, 0)
    Loss_t   = max(-Delta_t, 0)                 # always positive

    AvgGain_t = EMA(Gain, α = 1/period)         # Wilder's smoothing
    AvgLoss_t = EMA(Loss, α = 1/period)

    RS_t     = AvgGain_t / AvgLoss_t
    RSI_t    = 100 − (100 / (1 + RS_t))

Interpretation zones
---------------------
* RSI ≥ 70  → **Overbought** (potential reversal / sell signal)
* RSI ≤ 30  → **Oversold**   (potential bounce / buy signal)
* 30 < RSI < 70 → **Neutral**

Typical usage
-------------
::

    from src.indicators.rsi import add_rsi
    df = add_rsi(df, period=14)          # adds column 'rsi_14'
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)-8s]  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_RSI_PERIOD: int = 14
DEFAULT_PRICE_COL: str = "close"

#: Classic threshold levels for signal interpretation.
RSI_OVERBOUGHT: float = 70.0
RSI_OVERSOLD: float = 30.0


# ──────────────────────────────────────────────────────────────────────────────
# Core computation
# ──────────────────────────────────────────────────────────────────────────────
def compute_rsi(
    series: pd.Series,
    period: int = DEFAULT_RSI_PERIOD,
) -> pd.Series:
    """
    Compute RSI using Wilder's exponential smoothing (alpha = 1 / period).

    The first *period* values of the output are ``NaN`` because at least
    *period* price changes are required before the first average can be
    formed.

    Parameters
    ----------
    series:
        Closing price series (or any price series).  Must have at least
        ``period + 1`` non-null elements.
    period:
        Look-back period.  Wilder's original recommendation is **14**.

    Returns
    -------
    pd.Series
        RSI values in the range [0, 100], indexed identically to *series*.
        Returns a fully-NaN Series if there is insufficient data.

    Raises
    ------
    TypeError
        If *series* is not a pd.Series.
    ValueError
        If *period* ≤ 0.

    Notes
    -----
    Division by zero (AvgLoss = 0) is handled by returning RSI = 100, which
    correctly represents the case where every bar in the window was an up-bar.
    """
    if not isinstance(series, pd.Series):
        raise TypeError(f"Expected pd.Series, received {type(series).__name__}.")
    if period <= 0:
        raise ValueError(f"'period' must be positive, received: {period}.")

    min_required = period + 1
    if series.notna().sum() < min_required:
        logger.warning(
            "Insufficient data: %d non-null values, need ≥ %d for RSI(%d).",
            series.notna().sum(), min_required, period,
        )
        return pd.Series(np.nan, index=series.index, dtype=float)

    # Price changes
    delta = series.diff()

    # Separate gains and losses — losses are kept positive
    gains = delta.clip(lower=0.0)
    losses = (-delta).clip(lower=0.0)

    # Wilder's smoothing: α = 1/period  (equivalent to an EMA with span = 2*period − 1)
    alpha = 1.0 / period
    avg_gain = gains.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    # Relative Strength — replace zero-loss rows with NaN to avoid ZeroDivisionError,
    # then handle them separately (RSI = 100 when all moves are gains).
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Where avg_loss was exactly 0 → RSI = 100 by definition
    rsi[avg_loss == 0.0] = 100.0

    logger.debug("RSI(%d) computed over %d data points.", period, len(series))
    return rsi


# ──────────────────────────────────────────────────────────────────────────────
# DataFrame-level helper
# ──────────────────────────────────────────────────────────────────────────────
def add_rsi(
    df: pd.DataFrame,
    period: int = DEFAULT_RSI_PERIOD,
    price_col: str = DEFAULT_PRICE_COL,
    col_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Append an RSI column to a copy of *df*.

    Parameters
    ----------
    df:
        Stock price DataFrame.  Must contain *price_col*.
    period:
        RSI look-back period (default 14).
    price_col:
        Column used as input for the RSI calculation.
    col_name:
        Custom name for the output column.  Defaults to ``'rsi_{period}'``.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with the RSI column appended.

    Raises
    ------
    KeyError
        If *price_col* does not exist in *df*.

    Examples
    --------
    >>> df = add_rsi(df)               # adds rsi_14
    >>> df = add_rsi(df, period=9)     # adds rsi_9
    >>> df = add_rsi(df, col_name="rsi_custom")
    """
    if price_col not in df.columns:
        raise KeyError(
            f"Price column '{price_col}' not found. Available: {df.columns.tolist()}"
        )
    df = df.copy()
    output_col = col_name or f"rsi_{period}"
    df[output_col] = compute_rsi(df[price_col], period)
    logger.info("Added column '%s' (RSI period=%d).", output_col, period)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Interpretation helpers
# ──────────────────────────────────────────────────────────────────────────────
def classify_rsi_zone(
    rsi_value: float,
    overbought: float = RSI_OVERBOUGHT,
    oversold: float = RSI_OVERSOLD,
) -> str:
    """
    Classify a single RSI reading into a named zone.

    Parameters
    ----------
    rsi_value:
        A scalar RSI reading.
    overbought:
        Upper threshold (default 70).
    oversold:
        Lower threshold (default 30).

    Returns
    -------
    str
        ``'overbought'``, ``'oversold'``, ``'neutral'``, or ``'na'`` when
        the value is NaN.
    """
    if np.isnan(rsi_value):
        return "na"
    if rsi_value >= overbought:
        return "overbought"
    if rsi_value <= oversold:
        return "oversold"
    return "neutral"


def rsi_divergence_hint(price: pd.Series, rsi: pd.Series, window: int = 14) -> str:
    """
    Heuristic check for *bullish* or *bearish* divergence over a recent window.

    Divergence is a popular confirmation tool:

    * **Bullish divergence** — price makes a lower low but RSI makes a higher
      low (momentum not confirming the price weakness → potential reversal up).
    * **Bearish divergence** — price makes a higher high but RSI makes a lower
      high (momentum not confirming the price strength → potential reversal down).

    Parameters
    ----------
    price:
        Closing price series.
    rsi:
        Corresponding RSI series.
    window:
        Number of recent bars to inspect.

    Returns
    -------
    str
        ``'bullish_divergence'``, ``'bearish_divergence'``, or ``'none'``.

    Notes
    -----
    This is a simplified heuristic comparing the *window*-period extremes.
    A production system would use proper swing-high / swing-low detection.
    """
    p = price.iloc[-window:]
    r = rsi.iloc[-window:]

    price_lower_low = p.iloc[-1] < p.iloc[0]
    rsi_higher_low = r.iloc[-1] > r.iloc[0]
    if price_lower_low and rsi_higher_low:
        return "bullish_divergence"

    price_higher_high = p.iloc[-1] > p.iloc[0]
    rsi_lower_high = r.iloc[-1] < r.iloc[0]
    if price_higher_high and rsi_lower_high:
        return "bearish_divergence"

    return "none"


# ──────────────────────────────────────────────────────────────────────────────
# Script entry-point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  RSI Indicator — Demo")
    print("=" * 55)

    processed_path = os.path.join("data", "processed", "TCS_processed.csv")

    if os.path.exists(processed_path):
        df = pd.read_csv(processed_path, parse_dates=["date"])
        print(f"\nLoaded TCS: {len(df)} rows")
    else:
        # Synthetic data for offline testing
        np.random.seed(7)
        n = 120
        prices = 3_500 + np.cumsum(np.random.randn(n) * 40)
        df = pd.DataFrame({
            "date":  pd.date_range("2023-01-01", periods=n, freq="B"),
            "close": prices,
        })
        print(f"\nUsing synthetic data: {len(df)} rows")

    df = add_rsi(df, period=14)

    print("\n── Last 10 rows (close + rsi_14) ──────────────────")
    print(df[["date", "close", "rsi_14"]].tail(10).to_string(index=False))

    # Classify last bar
    last_rsi = df["rsi_14"].iloc[-1]
    zone = classify_rsi_zone(last_rsi)
    print(f"\nLatest RSI(14): {last_rsi:.2f}  →  Zone: {zone.upper()}")

    # Zone distribution
    df["rsi_zone"] = df["rsi_14"].apply(classify_rsi_zone)
    print("\n── RSI Zone Distribution ───────────────────────────")
    print(df["rsi_zone"].value_counts().to_string())
    print()
