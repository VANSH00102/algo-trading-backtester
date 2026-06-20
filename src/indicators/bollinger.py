"""
bollinger.py
============
Bollinger Bands implementation built from scratch using pandas and NumPy.

Mathematical background
-----------------------
Introduced by John Bollinger in the 1980s, Bollinger Bands place a volatility
envelope around a moving average:

::

    Middle Band  = SMA(close, n)
    Upper Band   = Middle + k × σ(close, n)
    Lower Band   = Middle − k × σ(close, n)

where:
* n = rolling window (default **20** periods)
* k = number of standard deviations (default **2**)
* σ  = population standard deviation (``ddof=0``) of the last *n* closes

Derived metrics (added automatically)
--------------------------------------
* **%B** — where the current price sits *within* the bands:
  ``%B = (Close − Lower) / (Upper − Lower)``
  Values: 0 → at lower band, 0.5 → at midline, 1 → at upper band, >1 or <0 → outside bands

* **Bandwidth** — measures band width relative to the midline:
  ``BW = (Upper − Lower) / Middle``
  High BW → high volatility; low BW (Squeeze) → volatility compression before a move

Typical usage
-------------
::

    from src.indicators.bollinger import add_bollinger_bands
    df = add_bollinger_bands(df)
    # New columns: bb_middle, bb_upper, bb_lower, bb_width, bb_pct_b
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

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
DEFAULT_BB_PERIOD: int = 20
DEFAULT_BB_STD: float = 2.0
DEFAULT_PRICE_COL: str = "close"
DEFAULT_PREFIX: str = "bb"


# ──────────────────────────────────────────────────────────────────────────────
# Core computation
# ──────────────────────────────────────────────────────────────────────────────
def compute_bollinger_bands(
    series: pd.Series,
    period: int = DEFAULT_BB_PERIOD,
    num_std: float = DEFAULT_BB_STD,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute the three Bollinger Band lines for a price series.

    Parameters
    ----------
    series:
        Closing price series (or any price series).
    period:
        Rolling window size in bars (default 20).
    num_std:
        Standard deviation multiplier (default 2.0).  A value of 2.0 means
        roughly 95% of recent closes should lie within the bands under a
        normal distribution assumption.

    Returns
    -------
    Tuple[pd.Series, pd.Series, pd.Series]
        ``(middle_band, upper_band, lower_band)`` — all aligned with
        *series*'s index.  The first ``period − 1`` values are ``NaN``.

    Raises
    ------
    TypeError
        If *series* is not a pd.Series.
    ValueError
        If *period* ≤ 0 or *num_std* < 0.

    Notes
    -----
    Population standard deviation (``ddof=0``) is used here, consistent with
    John Bollinger's original specification and most charting platforms.
    """
    if not isinstance(series, pd.Series):
        raise TypeError(f"Expected pd.Series, received {type(series).__name__}.")
    if period <= 0:
        raise ValueError(f"'period' must be positive, received: {period}.")
    if num_std < 0:
        raise ValueError(f"'num_std' must be ≥ 0, received: {num_std}.")

    middle = series.rolling(window=period, min_periods=period).mean()
    # ddof=0 → population std (matches Bollinger's spec and TradingView default)
    std = series.rolling(window=period, min_periods=period).std(ddof=0)

    upper = middle + (num_std * std)
    lower = middle - (num_std * std)

    logger.debug("Bollinger Bands(%d, %.1fσ) computed.", period, num_std)
    return middle, upper, lower


# ──────────────────────────────────────────────────────────────────────────────
# DataFrame-level helper
# ──────────────────────────────────────────────────────────────────────────────
def add_bollinger_bands(
    df: pd.DataFrame,
    period: int = DEFAULT_BB_PERIOD,
    num_std: float = DEFAULT_BB_STD,
    price_col: str = DEFAULT_PRICE_COL,
    prefix: str = DEFAULT_PREFIX,
) -> pd.DataFrame:
    """
    Append Bollinger Band columns to a copy of *df*.

    Columns added
    -------------
    * ``{prefix}_middle`` — 20-period SMA
    * ``{prefix}_upper``  — upper band  (middle + 2σ)
    * ``{prefix}_lower``  — lower band  (middle − 2σ)
    * ``{prefix}_width``  — absolute band width  (upper − lower)
    * ``{prefix}_pct_b``  — %B position within bands
    * ``{prefix}_bw``     — normalised bandwidth  ((upper − lower) / middle)

    Parameters
    ----------
    df:
        Stock price DataFrame.  Must contain *price_col*.
    period:
        SMA and std look-back window (default 20).
    num_std:
        Standard deviation multiplier (default 2.0).
    price_col:
        Column used as input.  Defaults to ``'close'``.
    prefix:
        Column name prefix (default ``'bb'``).

    Returns
    -------
    pd.DataFrame
        Copy of *df* with the six Bollinger Band columns appended.

    Raises
    ------
    KeyError
        If *price_col* is not in *df*.

    Examples
    --------
    >>> df = add_bollinger_bands(df)                       # defaults: 20-period, 2σ
    >>> df = add_bollinger_bands(df, period=10, num_std=1.5)
    >>> df = add_bollinger_bands(df, prefix="bb_weekly")   # custom prefix
    """
    if price_col not in df.columns:
        raise KeyError(
            f"Price column '{price_col}' not found. Available: {df.columns.tolist()}"
        )
    df = df.copy()

    middle, upper, lower = compute_bollinger_bands(df[price_col], period, num_std)

    df[f"{prefix}_middle"] = middle
    df[f"{prefix}_upper"]  = upper
    df[f"{prefix}_lower"]  = lower

    # Absolute band width — raw volatility measure in price units
    df[f"{prefix}_width"] = upper - lower

    # %B — normalised position of price within the bands
    # Guard against zero-width bands (constant price over window)
    band_range = (upper - lower).replace(0.0, np.nan)
    df[f"{prefix}_pct_b"] = (df[price_col] - lower) / band_range

    # Normalised bandwidth — useful for detecting Bollinger Squeezes
    middle_safe = middle.replace(0.0, np.nan)
    df[f"{prefix}_bw"] = (upper - lower) / middle_safe

    added = [
        f"{prefix}_middle", f"{prefix}_upper", f"{prefix}_lower",
        f"{prefix}_width", f"{prefix}_pct_b", f"{prefix}_bw",
    ]
    logger.info("Added Bollinger Band columns: %s", added)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Interpretation helpers
# ──────────────────────────────────────────────────────────────────────────────
def classify_bb_position(
    price: float,
    upper: float,
    lower: float,
    middle: float,
) -> str:
    """
    Classify where *price* sits relative to the Bollinger Bands.

    Parameters
    ----------
    price:
        Current closing price.
    upper:
        Upper Bollinger Band value.
    lower:
        Lower Bollinger Band value.
    middle:
        Middle band (SMA) value.

    Returns
    -------
    str
        One of: ``'above_upper'``, ``'upper_half'``, ``'lower_half'``,
        ``'below_lower'``, or ``'na'`` when any value is NaN.
    """
    for val in (price, upper, lower, middle):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "na"

    if price > upper:
        return "above_upper"
    if price >= middle:
        return "upper_half"
    if price >= lower:
        return "lower_half"
    return "below_lower"


def detect_squeeze(
    df: pd.DataFrame,
    prefix: str = DEFAULT_PREFIX,
    lookback: int = 20,
) -> pd.Series:
    """
    Detect a **Bollinger Squeeze** — a period of unusually low bandwidth.

    A squeeze occurs when the current bandwidth falls below the lowest
    bandwidth reading of the past *lookback* periods.  It signals
    volatility compression and often precedes a sharp directional move.

    Parameters
    ----------
    df:
        DataFrame already enriched with Bollinger Band columns.
    prefix:
        Column prefix used in :func:`add_bollinger_bands`.
    lookback:
        Number of bars to use as the bandwidth look-back (default 20).

    Returns
    -------
    pd.Series
        Boolean Series — ``True`` on bars where a squeeze is active.

    Raises
    ------
    KeyError
        If the normalised bandwidth column is absent (run
        :func:`add_bollinger_bands` first).
    """
    bw_col = f"{prefix}_bw"
    if bw_col not in df.columns:
        raise KeyError(
            f"Column '{bw_col}' not found. "
            f"Call add_bollinger_bands(prefix='{prefix}') first."
        )
    bw = df[bw_col]
    bw_min = bw.rolling(window=lookback, min_periods=lookback).min()
    squeeze = bw <= bw_min
    n_squeeze = squeeze.sum()
    logger.debug("Squeeze bars detected: %d.", n_squeeze)
    return squeeze


def bb_signal(
    df: pd.DataFrame,
    prefix: str = DEFAULT_PREFIX,
    price_col: str = DEFAULT_PRICE_COL,
) -> pd.Series:
    """
    Generate a simple mean-reversion signal from Bollinger Bands.

    Rules
    -----
    * Close **below lower band** → +1  (potential long / oversold bounce)
    * Close **above upper band** → −1  (potential short / overbought fade)
    * Otherwise                  →  0  (no signal)

    Parameters
    ----------
    df:
        DataFrame with Bollinger Band columns and *price_col*.
    prefix:
        Column prefix.
    price_col:
        Price column to compare against the bands.

    Returns
    -------
    pd.Series
        Integer signal series: +1, −1, or 0.
    """
    upper_col = f"{prefix}_upper"
    lower_col = f"{prefix}_lower"
    for col in (upper_col, lower_col, price_col):
        if col not in df.columns:
            raise KeyError(
                f"Column '{col}' not found. "
                f"Ensure add_bollinger_bands() and the price column are present."
            )
    signal = pd.Series(0, index=df.index, dtype=int)
    signal[df[price_col] < df[lower_col]] = 1
    signal[df[price_col] > df[upper_col]] = -1
    return signal


# ──────────────────────────────────────────────────────────────────────────────
# Script entry-point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Bollinger Bands — Demo")
    print("=" * 55)

    processed_path = os.path.join("data", "processed", "TCS_processed.csv")

    if os.path.exists(processed_path):
        df = pd.read_csv(processed_path, parse_dates=["date"])
        print(f"\nLoaded TCS: {len(df)} rows")
    else:
        # Synthetic data for offline testing
        np.random.seed(99)
        n = 150
        prices = 3_500 + np.cumsum(np.random.randn(n) * 35)
        df = pd.DataFrame({
            "date":  pd.date_range("2022-01-01", periods=n, freq="B"),
            "close": prices,
        })
        print(f"\nUsing synthetic data: {len(df)} rows")

    df = add_bollinger_bands(df, period=20, num_std=2.0)

    display_cols = [
        "date", "close",
        "bb_middle", "bb_upper", "bb_lower",
        "bb_width", "bb_pct_b", "bb_bw",
    ]
    print("\n── Last 10 rows ─────────────────────────────────────")
    print(df[display_cols].tail(10).round(2).to_string(index=False))

    # Signal summary
    df["bb_pos"] = df.apply(
        lambda r: classify_bb_position(r["close"], r["bb_upper"], r["bb_lower"], r["bb_middle"]),
        axis=1,
    )
    df["bb_signal"] = bb_signal(df)
    squeeze_mask = detect_squeeze(df)

    print("\n── BB Position Distribution ────────────────────────")
    print(df["bb_pos"].value_counts().to_string())

    print("\n── Signal Summary ──────────────────────────────────")
    print(df["bb_signal"].value_counts().rename({1: "Long (+1)", -1: "Short (-1)", 0: "Neutral (0)"}).to_string())

    print(f"\n── Squeeze bars: {squeeze_mask.sum()} ──────────────────────────────")
    print()
