"""
moving_average.py
=================
Simple Moving Average (SMA) and Exponential Moving Average (EMA) indicator
calculations for use in strategy development and backtesting.

Mathematical definitions
------------------------
* **SMA(n)** = arithmetic mean of the last *n* closing prices.
* **EMA(n)** = price × k + EMA_prev × (1 − k),  where k = 2 / (n + 1).

The default windows (20, 50, 200) correspond to the most widely used
timeframes in Indian equity technical analysis:

* 20-period  → short-term trend / Bollinger Band basis
* 50-period  → intermediate-term trend
* 200-period → long-term trend; golden/death cross signals

Typical usage
-------------
::

    from src.indicators.moving_average import add_moving_averages
    df = add_moving_averages(df, windows=[20, 50, 200])
    # New columns: sma_20, sma_50, sma_200, ema_20, ema_50, ema_200
"""

from __future__ import annotations

import logging
import os
from typing import List

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
#: Canonical window set used throughout the strategy codebase.
DEFAULT_WINDOWS: List[int] = [20, 50, 200]

#: Default OHLCV column used for MA calculations.
DEFAULT_PRICE_COL: str = "close"


# ──────────────────────────────────────────────────────────────────────────────
# Core computation functions (operate on a pd.Series)
# ──────────────────────────────────────────────────────────────────────────────
def compute_sma(series: pd.Series, window: int) -> pd.Series:
    """
    Calculate a Simple Moving Average over *window* periods.

    The first ``window - 1`` values are ``NaN`` because there is not yet
    enough history to form a complete window.

    Parameters
    ----------
    series:
        Price series (typically closing prices).
    window:
        Look-back period in bars.  Must be a positive integer.

    Returns
    -------
    pd.Series
        SMA values aligned with the index of *series*.

    Raises
    ------
    ValueError
        If *window* is not a positive integer.
    """
    if not isinstance(window, int) or window <= 0:
        raise ValueError(f"'window' must be a positive integer, received: {window!r}")
    return series.rolling(window=window, min_periods=window).mean()


def compute_ema(series: pd.Series, window: int) -> pd.Series:
    """
    Calculate an Exponential Moving Average over *window* periods.

    Uses the standard smoothing factor  α = 2 / (window + 1)  and
    ``adjust=False`` so the recursive formula is applied strictly:
    EMA_t = price_t × α + EMA_{t-1} × (1 − α).

    The first ``window - 1`` values are ``NaN`` (``min_periods=window``).

    Parameters
    ----------
    series:
        Price series (typically closing prices).
    window:
        EMA span — controls the decay speed of older observations.

    Returns
    -------
    pd.Series
        EMA values aligned with the index of *series*.

    Raises
    ------
    ValueError
        If *window* is not a positive integer.
    """
    if not isinstance(window, int) or window <= 0:
        raise ValueError(f"'window' must be a positive integer, received: {window!r}")
    return series.ewm(span=window, min_periods=window, adjust=False).mean()


# ──────────────────────────────────────────────────────────────────────────────
# DataFrame-level helpers  (add columns to a copy of the input)
# ──────────────────────────────────────────────────────────────────────────────
def add_sma(
    df: pd.DataFrame,
    windows: List[int] = DEFAULT_WINDOWS,
    price_col: str = DEFAULT_PRICE_COL,
) -> pd.DataFrame:
    """
    Add one SMA column per window to *df*.

    Column naming pattern: ``sma_{window}`` (e.g. ``sma_20``).

    Parameters
    ----------
    df:
        Stock price DataFrame.  Must contain *price_col*.
    windows:
        Ordered list of look-back periods (e.g. ``[20, 50, 200]``).
    price_col:
        Column used as input.  Defaults to ``'close'``.

    Returns
    -------
    pd.DataFrame
        A copy of *df* with the SMA columns appended.

    Raises
    ------
    KeyError
        If *price_col* is not in *df*.
    """
    if price_col not in df.columns:
        raise KeyError(
            f"Price column '{price_col}' not found. Available: {df.columns.tolist()}"
        )
    df = df.copy()
    for w in windows:
        col = f"sma_{w}"
        df[col] = compute_sma(df[price_col], w)
        logger.info("Added column '%s'.", col)
    return df


def add_ema(
    df: pd.DataFrame,
    windows: List[int] = DEFAULT_WINDOWS,
    price_col: str = DEFAULT_PRICE_COL,
) -> pd.DataFrame:
    """
    Add one EMA column per window to *df*.

    Column naming pattern: ``ema_{window}`` (e.g. ``ema_50``).

    Parameters
    ----------
    df:
        Stock price DataFrame.  Must contain *price_col*.
    windows:
        Ordered list of EMA spans.
    price_col:
        Column used as input.  Defaults to ``'close'``.

    Returns
    -------
    pd.DataFrame
        A copy of *df* with the EMA columns appended.

    Raises
    ------
    KeyError
        If *price_col* is not in *df*.
    """
    if price_col not in df.columns:
        raise KeyError(
            f"Price column '{price_col}' not found. Available: {df.columns.tolist()}"
        )
    df = df.copy()
    for w in windows:
        col = f"ema_{w}"
        df[col] = compute_ema(df[price_col], w)
        logger.info("Added column '%s'.", col)
    return df


def add_moving_averages(
    df: pd.DataFrame,
    windows: List[int] = DEFAULT_WINDOWS,
    price_col: str = DEFAULT_PRICE_COL,
) -> pd.DataFrame:
    """
    Convenience wrapper: add both SMA and EMA columns in one call.

    Equivalent to chaining :func:`add_sma` and :func:`add_ema`.

    Parameters
    ----------
    df:
        Stock price DataFrame.
    windows:
        Periods applied to **both** SMA and EMA.
    price_col:
        Input price column.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with ``sma_*`` and ``ema_*`` columns appended.

    Examples
    --------
    >>> df = add_moving_averages(df, windows=[20, 50, 200])
    >>> df.columns  # now includes sma_20, sma_50, sma_200, ema_20, ema_50, ema_200
    """
    df = add_sma(df, windows=windows, price_col=price_col)
    df = add_ema(df, windows=windows, price_col=price_col)
    return df


def detect_golden_cross(df: pd.DataFrame, fast: int = 50, slow: int = 200) -> pd.Series:
    """
    Identify Golden Cross and Death Cross events.

    A **Golden Cross** (+1) occurs when the fast SMA crosses *above* the slow
    SMA; a **Death Cross** (−1) occurs on the crossover downward.

    Parameters
    ----------
    df:
        DataFrame that already contains ``sma_{fast}`` and ``sma_{slow}``.
    fast:
        Fast SMA period.
    slow:
        Slow SMA period.

    Returns
    -------
    pd.Series
        Integer series: +1 (golden cross), −1 (death cross), 0 (no event).

    Raises
    ------
    KeyError
        If the required SMA columns are absent (run :func:`add_sma` first).
    """
    fast_col, slow_col = f"sma_{fast}", f"sma_{slow}"
    for col in (fast_col, slow_col):
        if col not in df.columns:
            raise KeyError(
                f"Column '{col}' not found. "
                f"Call add_sma(windows=[{fast}, {slow}]) first."
            )

    above = df[fast_col] > df[slow_col]
    cross = above.astype(int).diff()  # +1 = just crossed above, -1 = just crossed below
    return cross.fillna(0).astype(int)


# ──────────────────────────────────────────────────────────────────────────────
# Script entry-point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Moving Averages — Demo")
    print("=" * 55)

    processed_path = os.path.join("data", "processed", "TCS_processed.csv")

    if os.path.exists(processed_path):
        df = pd.read_csv(processed_path, parse_dates=["date"])
        print(f"\nLoaded TCS: {len(df)} rows")
    else:
        # Synthetic data for offline testing
        np.random.seed(42)
        n = 300
        prices = 3_000 + np.cumsum(np.random.randn(n) * 30)
        df = pd.DataFrame({
            "date":  pd.date_range("2018-01-01", periods=n, freq="B"),
            "close": prices,
        })
        print(f"\nUsing synthetic data: {len(df)} rows")

    df = add_moving_averages(df, windows=[20, 50, 200])
    cross_signal = detect_golden_cross(df)

    display_cols = [
        c for c in ["date", "close", "sma_20", "sma_50", "sma_200", "ema_20", "ema_50", "ema_200"]
        if c in df.columns
    ]
    print("\n── Last 5 rows ─────────────────────────────────────")
    print(df[display_cols].tail(5).to_string(index=False))

    n_golden = (cross_signal == 1).sum()
    n_death = (cross_signal == -1).sum()
    print(f"\nGolden crosses (50/200 SMA): {n_golden}")
    print(f"Death  crosses (50/200 SMA): {n_death}")
    print()
