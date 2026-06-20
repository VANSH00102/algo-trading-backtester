"""
atr.py
======
Average True Range (ATR) implementation built from scratch using pandas
and NumPy.

Mathematical background
-----------------------
ATR was introduced by J. Welles Wilder alongside RSI in *New Concepts in
Technical Trading Systems* (1978).

**True Range (TR)** captures the full price movement in a single bar,
including any overnight gap:

::

    TR_t = max(
        High_t − Low_t,               # intraday range
        |High_t − Close_{t-1}|,       # gap-up covered by high
        |Low_t  − Close_{t-1}|,       # gap-down covered by low
    )

**ATR(n)** is then the Wilder-smoothed average of TR:

::

    ATR_t = ((n − 1) × ATR_{t-1} + TR_t) / n

which is identical to an EMA with α = 1/n.

Use cases in Indian equity trading
------------------------------------
1. **Stop-loss placement** — ``Stop = Entry ± (multiplier × ATR)``
2. **Position sizing** — volatility-adjusted lot size
3. **Breakout confirmation** — unusually large ATR bar signals expansion
4. **Strategy filter** — avoid trading in low-ATR (choppy) environments

Typical usage
-------------
::

    from src.indicators.atr import add_atr, atr_stop_loss
    df = add_atr(df, period=14)          # adds 'true_range' and 'atr_14'
    sl = atr_stop_loss(entry=3400, atr=85.0, multiplier=2.0, direction="long")
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
DEFAULT_ATR_PERIOD: int = 14

#: Columns required by all ATR calculation functions.
REQUIRED_COLUMNS = ("high", "low", "close")


# ──────────────────────────────────────────────────────────────────────────────
# Core computation
# ──────────────────────────────────────────────────────────────────────────────
def compute_true_range(df: pd.DataFrame) -> pd.Series:
    """
    Compute the True Range (TR) for every bar in *df*.

    The first bar always has ``NaN`` for the previous-close components,
    so its TR equals ``High - Low`` (as defined by Wilder for bar 0).

    Parameters
    ----------
    df:
        DataFrame containing ``'high'``, ``'low'``, and ``'close'`` columns.

    Returns
    -------
    pd.Series
        True Range values aligned with *df*'s index.  The first value is
        always ``High_0 − Low_0``; subsequent values incorporate the
        previous close.

    Raises
    ------
    KeyError
        If any of the required columns are missing.
    """
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise KeyError(
                f"Required column '{col}' not found. "
                f"Available: {df.columns.tolist()}"
            )

    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)

    # Three candidate ranges — we broadcast them into a single DataFrame and
    # take the element-wise max.  Using abs() handles both gap-up and gap-down.
    hl_range = high - low
    hc_range = (high - prev_close).abs()
    lc_range = (low - prev_close).abs()

    true_range = pd.concat([hl_range, hc_range, lc_range], axis=1).max(axis=1)
    logger.debug("True Range computed (%d bars).", len(true_range))
    return true_range


def compute_atr(
    df: pd.DataFrame,
    period: int = DEFAULT_ATR_PERIOD,
) -> pd.Series:
    """
    Compute the Average True Range (ATR) using Wilder's smoothing.

    Wilder's smoothing uses α = 1/period, which is identical to an EMA
    with ``adjust=False``.  The first *period* values are ``NaN``
    (``min_periods=period``).

    Parameters
    ----------
    df:
        DataFrame with ``'high'``, ``'low'``, and ``'close'`` columns.
    period:
        Smoothing window.  Wilder's default (and industry standard) is **14**.

    Returns
    -------
    pd.Series
        ATR values indexed like *df*.

    Raises
    ------
    ValueError
        If *period* ≤ 0.
    KeyError
        If required columns are absent (propagated from
        :func:`compute_true_range`).
    """
    if period <= 0:
        raise ValueError(f"'period' must be positive, received: {period}.")

    tr = compute_true_range(df)
    alpha = 1.0 / period
    atr = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    logger.debug("ATR(%d) computed.", period)
    return atr


# ──────────────────────────────────────────────────────────────────────────────
# DataFrame-level helper
# ──────────────────────────────────────────────────────────────────────────────
def add_atr(
    df: pd.DataFrame,
    period: int = DEFAULT_ATR_PERIOD,
    col_name: Optional[str] = None,
    include_tr: bool = True,
) -> pd.DataFrame:
    """
    Append ATR (and optionally True Range) columns to a copy of *df*.

    Parameters
    ----------
    df:
        Stock price DataFrame with ``'high'``, ``'low'``, ``'close'``.
    period:
        ATR smoothing period (default 14).
    col_name:
        Custom name for the ATR column.  Defaults to ``'atr_{period}'``.
    include_tr:
        If ``True`` (default), also add a ``'true_range'`` column.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with ``'true_range'`` and/or ``'atr_{period}'``
        columns appended.

    Examples
    --------
    >>> df = add_atr(df)                       # adds true_range, atr_14
    >>> df = add_atr(df, period=7, include_tr=False)   # adds only atr_7
    """
    df = df.copy()
    tr = compute_true_range(df)

    if include_tr:
        df["true_range"] = tr
        logger.info("Added column 'true_range'.")

    alpha = 1.0 / period
    atr_col = col_name or f"atr_{period}"
    df[atr_col] = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    logger.info("Added column '%s' (ATR period=%d).", atr_col, period)

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Trading utilities
# ──────────────────────────────────────────────────────────────────────────────
def atr_stop_loss(
    entry: float,
    atr: float,
    multiplier: float = 2.0,
    direction: str = "long",
) -> float:
    """
    Calculate an ATR-based stop-loss price.

    ATR stop-loss places the stop a fixed *multiple* of ATR away from the
    entry price, adapting to current market volatility automatically.

    Parameters
    ----------
    entry:
        Trade entry price (in INR for Indian equities).
    atr:
        Current ATR value.
    multiplier:
        Number of ATR units to place the stop away from entry (default 2.0).
        Common values: 1.5 (tight), 2.0 (moderate), 3.0 (wide).
    direction:
        ``'long'`` → stop is *below* entry.
        ``'short'`` → stop is *above* entry.

    Returns
    -------
    float
        Stop-loss price.

    Raises
    ------
    ValueError
        If *direction* is not ``'long'`` or ``'short'``.

    Examples
    --------
    >>> atr_stop_loss(entry=3_400, atr=85.0, multiplier=2.0, direction="long")
    3230.0
    >>> atr_stop_loss(entry=3_400, atr=85.0, multiplier=2.0, direction="short")
    3570.0
    """
    if direction not in ("long", "short"):
        raise ValueError(
            f"'direction' must be 'long' or 'short', received: {direction!r}"
        )
    offset = multiplier * atr
    return entry - offset if direction == "long" else entry + offset


def atr_position_size(
    capital: float,
    risk_pct: float,
    entry: float,
    atr: float,
    atr_multiplier: float = 2.0,
) -> int:
    """
    Volatility-adjusted position size using the ATR-based stop distance.

    Limits the capital at risk to ``risk_pct``% of total capital, placing
    the stop ``atr_multiplier × ATR`` from entry.

    Parameters
    ----------
    capital:
        Total account capital in INR.
    risk_pct:
        Maximum allowable risk per trade as a percentage (e.g. 1.0 for 1%).
    entry:
        Entry price per share.
    atr:
        Current ATR value.
    atr_multiplier:
        ATR multiplier used for the stop distance.

    Returns
    -------
    int
        Number of shares to trade (rounded down to whole lots).

    Examples
    --------
    >>> atr_position_size(capital=500_000, risk_pct=1.0, entry=3_400, atr=85.0)
    # Risk budget = 5000 INR; stop distance = 170 INR → 29 shares
    29
    """
    risk_amount = capital * (risk_pct / 100.0)
    stop_distance = atr_multiplier * atr
    if stop_distance <= 0:
        raise ValueError("Stop distance must be positive (ATR may be zero or NaN).")
    return int(risk_amount // stop_distance)


# ──────────────────────────────────────────────────────────────────────────────
# Script entry-point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  ATR Indicator — Demo")
    print("=" * 55)

    processed_path = os.path.join("data", "processed", "TCS_processed.csv")

    if os.path.exists(processed_path):
        df = pd.read_csv(processed_path, parse_dates=["date"])
        print(f"\nLoaded TCS: {len(df)} rows")
    else:
        # Synthetic OHLCV data for offline testing
        np.random.seed(21)
        n = 100
        close = 3_000 + np.cumsum(np.random.randn(n) * 30)
        df = pd.DataFrame({
            "date":   pd.date_range("2023-01-01", periods=n, freq="B"),
            "open":   close * np.random.uniform(0.995, 1.005, n),
            "high":   close * np.random.uniform(1.005, 1.02, n),
            "low":    close * np.random.uniform(0.98, 0.995, n),
            "close":  close,
            "volume": np.random.randint(1_000_000, 5_000_000, n),
        })
        print(f"\nUsing synthetic data: {len(df)} rows")

    df = add_atr(df, period=14)

    display_cols = ["date", "high", "low", "close", "true_range", "atr_14"]
    print("\n── Last 10 rows ─────────────────────────────────────")
    print(df[display_cols].tail(10).to_string(index=False))

    # Practical stop-loss example
    last_close = df["close"].iloc[-1]
    last_atr = df["atr_14"].iloc[-1]
    sl_long = atr_stop_loss(last_close, last_atr, multiplier=2.0, direction="long")
    sl_short = atr_stop_loss(last_close, last_atr, multiplier=2.0, direction="short")
    lot_size = atr_position_size(
        capital=500_000, risk_pct=1.0, entry=last_close, atr=last_atr
    )

    print(f"\n── ATR-based Risk Management ───────────────────────")
    print(f"  Entry price      : ₹{last_close:,.2f}")
    print(f"  ATR(14)          : ₹{last_atr:,.2f}")
    print(f"  Long  stop (2×)  : ₹{sl_long:,.2f}")
    print(f"  Short stop (2×)  : ₹{sl_short:,.2f}")
    print(f"  Position size*   : {lot_size} shares")
    print("  (* 1% risk on ₹5,00,000 capital)")
    print()
