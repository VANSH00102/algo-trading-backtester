"""
fetch_data.py
=============
Downloads historical OHLCV data for Indian equities — TCS, Reliance, Infosys —
from Yahoo Finance via the yfinance library and saves each stock to its own
CSV file inside ``data/raw/``.

Key design decisions
--------------------
* yfinance ≥ 1.x returns a MultiIndex DataFrame; this module flattens it
  transparently so the rest of the pipeline always sees flat column names.
* Every stock is fetched independently with its own error handler so one
  failure does not abort the whole batch.
* Public constants (STOCKS, RAW_DATA_DIR, DEFAULT_START/END) can be
  overridden at import time or via function arguments — no magic globals.

Typical usage
-------------
Run as a script from the project root::

    python src/data/fetch_data.py

Or import into a pipeline::

    from src.data.fetch_data import fetch_all_stocks
    results = fetch_all_stocks(start_date="2020-01-01", end_date="2024-12-31")
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Dict, Optional

import pandas as pd
import yfinance as yf

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
#: Mapping from a human-readable display name to the NSE ticker on Yahoo Finance.
STOCKS: Dict[str, str] = {
    "TCS": "TCS.NS",
    "RELIANCE": "RELIANCE.NS",
    "INFOSYS": "INFY.NS",
}

#: Where raw CSVs are written relative to the project root.
RAW_DATA_DIR: str = os.path.join("data", "raw")

#: Default historical look-back start date.
DEFAULT_START: str = "2018-01-01"

#: Default end date — today at execution time.
DEFAULT_END: str = datetime.today().strftime("%Y-%m-%d")


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────
def _ensure_directory(path: str) -> None:
    """Create *path* (and any missing parents) if it does not already exist."""
    os.makedirs(path, exist_ok=True)
    logger.debug("Directory ready: %s", path)


def _flatten_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse a MultiIndex column structure produced by yfinance ≥ 1.x.

    yfinance returns columns like ``('Close', 'TCS.NS')`` so that
    single-ticker and multi-ticker downloads share the same shape.  For a
    single-ticker download we only need the *price level* (first level).

    Parameters
    ----------
    df:
        Raw DataFrame straight from ``yf.download()``.

    Returns
    -------
    DataFrame with a plain (non-hierarchical) column Index, e.g.
    ``['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']``.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        logger.debug("Flattened MultiIndex columns → %s", df.columns.tolist())
    return df


def _normalise_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the date column is always named exactly ``'Date'``.

    yfinance uses 'Date' for daily data and 'Datetime' for intraday; both
    land as the DataFrame index name before ``reset_index``.  After
    ``reset_index``, whichever name appears becomes a regular column.
    This helper renames it to the canonical ``'Date'``.

    Parameters
    ----------
    df:
        DataFrame after ``reset_index()`` has been called.

    Returns
    -------
    DataFrame with a guaranteed ``'Date'`` column.
    """
    for candidate in ("Datetime", "datetime", "date"):
        if candidate in df.columns:
            df.rename(columns={candidate: "Date"}, inplace=True)
            logger.debug("Renamed '%s' → 'Date'.", candidate)
            break
    if "Date" not in df.columns:
        logger.warning("Could not locate a date column after reset_index.")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────
def fetch_stock_data(
    ticker: str,
    start_date: str = DEFAULT_START,
    end_date: str = DEFAULT_END,
    interval: str = "1d",
) -> Optional[pd.DataFrame]:
    """
    Download historical OHLCV data for a single NSE ticker via yfinance.

    Parameters
    ----------
    ticker:
        Yahoo Finance ticker symbol (e.g. ``'TCS.NS'``).
    start_date:
        Inclusive start date in ``'YYYY-MM-DD'`` format.
    end_date:
        Exclusive end date in ``'YYYY-MM-DD'`` format.
    interval:
        Candle interval accepted by yfinance (``'1d'``, ``'1wk'``, etc.).
        Defaults to ``'1d'`` (daily).

    Returns
    -------
    pd.DataFrame
        Flat DataFrame with columns ``[Date, Open, High, Low, Close,
        Adj Close, Volume]`` and a RangeIndex.
    None
        If the download fails or returns an empty result.

    Raises
    ------
    Does **not** raise — all exceptions are caught and logged so the caller
    can decide how to handle individual failures.
    """
    logger.info(
        "Fetching  %-14s  %s → %s  (interval=%s)",
        ticker, start_date, end_date, interval,
    )
    try:
        raw: pd.DataFrame = yf.download(
            tickers=ticker,
            start=start_date,
            end=end_date,
            interval=interval,
            auto_adjust=True,   # Adjust OHLCV for splits & dividends
            progress=False,
        )
    except Exception as exc:                          # pragma: no cover
        logger.error("yfinance raised an exception for %s: %s", ticker, exc)
        return None

    if raw is None or raw.empty:
        logger.warning(
            "No data returned for %s — check ticker symbol or date range.", ticker
        )
        return None

    # Normalise structure ─────────────────────────────────────────────────────
    raw = _flatten_multiindex(raw)
    raw.reset_index(inplace=True)
    raw = _normalise_date_column(raw)

    logger.info("  ✓  %d rows fetched for %s.", len(raw), ticker)
    return raw


def save_to_csv(df: pd.DataFrame, filepath: str) -> bool:
    """
    Write *df* to a CSV file at *filepath*.

    Parameters
    ----------
    df:
        DataFrame to persist.
    filepath:
        Destination path.  The parent directory must already exist.

    Returns
    -------
    bool
        ``True`` on success, ``False`` if an :class:`OSError` occurs.
    """
    try:
        df.to_csv(filepath, index=False)
        logger.info("  ✓  Saved %d rows → %s", len(df), filepath)
        return True
    except OSError as exc:
        logger.error("Failed to write %s: %s", filepath, exc)
        return False


def fetch_all_stocks(
    stocks: Dict[str, str] = STOCKS,
    start_date: str = DEFAULT_START,
    end_date: str = DEFAULT_END,
    output_dir: str = RAW_DATA_DIR,
) -> Dict[str, bool]:
    """
    Fetch and persist historical data for every stock in *stocks*.

    Each stock is saved as ``{NAME}_raw.csv`` (e.g. ``TCS_raw.csv``) inside
    *output_dir*.  Stocks that fail to download are logged but do not
    interrupt processing of the remaining tickers.

    Parameters
    ----------
    stocks:
        Mapping ``{ display_name: yfinance_ticker }``.  Defaults to the
        module-level :data:`STOCKS` constant.
    start_date:
        Inclusive start date (``'YYYY-MM-DD'``).
    end_date:
        Exclusive end date (``'YYYY-MM-DD'``).
    output_dir:
        Directory where raw CSVs will be written.

    Returns
    -------
    dict
        ``{ display_name: True | False }`` indicating per-stock success.
    """
    _ensure_directory(output_dir)
    results: Dict[str, bool] = {}

    for name, ticker in stocks.items():
        df = fetch_stock_data(ticker, start_date, end_date)

        if df is None:
            logger.warning("Skipping save for '%s' — no data was returned.", name)
            results[name] = False
            continue

        filename = f"{name.upper()}_raw.csv"
        filepath = os.path.join(output_dir, filename)
        results[name] = save_to_csv(df, filepath)

    # ── Batch summary ─────────────────────────────────────────────────────────
    succeeded = [k for k, v in results.items() if v]
    failed = [k for k, v in results.items() if not v]
    logger.info(
        "Fetch batch complete.  Success: %s  |  Failed: %s", succeeded, failed
    )
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Script entry-point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Algorithmic Trading Backtester — Data Fetch")
    print("=" * 55)

    status = fetch_all_stocks(
        start_date="2018-01-01",
        end_date=datetime.today().strftime("%Y-%m-%d"),
    )

    print("\n── Fetch Results ──────────────────────────────────")
    for stock, ok in status.items():
        marker = "✓" if ok else "✗"
        print(f"  {marker}  {stock}")
    print()
