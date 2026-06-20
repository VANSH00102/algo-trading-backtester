"""
preprocess.py
=============
Cleans and standardises raw CSV files produced by ``fetch_data.py`` and
saves the results to ``data/processed/``.

Pipeline (applied per stock)
----------------------------
1. Load raw CSV from ``data/raw/``
2. Standardise column names → snake_case
3. Parse date column to ``datetime64``
4. Sort rows in ascending chronological order
5. Remove duplicate dates (keep first occurrence)
6. Handle missing values (forward-fill then drop remaining NaNs)
7. Drop rows with non-positive prices (data quality guard)
8. Keep only the required columns
9. Add a ``ticker`` identifier column
10. Save cleaned DataFrame to ``data/processed/``

Typical usage
-------------
Run as a script from the project root::

    python src/data/preprocess.py

Or import into a pipeline::

    from src.data.preprocess import preprocess_all
    dfs = preprocess_all()
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

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
RAW_DATA_DIR: str = os.path.join("data", "raw")
PROCESSED_DATA_DIR: str = os.path.join("data", "processed")

#: Display names used to locate raw files; must match ``fetch_data.STOCKS`` keys.
STOCK_NAMES: List[str] = ["TCS", "RELIANCE", "INFOSYS"]

#: Maps raw column names (yfinance / various casing) to our canonical names.
COLUMN_RENAME_MAP: Dict[str, str] = {
    "Date":      "date",
    "Datetime":  "date",
    "Open":      "open",
    "High":      "high",
    "Low":       "low",
    "Close":     "close",
    "Adj Close": "adj_close",
    "Volume":    "volume",
}

#: Columns the pipeline guarantees to be present in the output.
REQUIRED_COLUMNS: List[str] = ["date", "open", "high", "low", "close", "volume"]

#: Subset of REQUIRED_COLUMNS used for gap-fill and positivity checks.
PRICE_COLUMNS: List[str] = ["open", "high", "low", "close", "volume"]


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────
def _ensure_directory(path: str) -> None:
    """Create *path* (and any missing parents) if it does not exist."""
    os.makedirs(path, exist_ok=True)
    logger.debug("Directory ready: %s", path)


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline steps  (each accepts and returns a DataFrame)
# ──────────────────────────────────────────────────────────────────────────────
def load_raw_csv(filepath: str) -> Optional[pd.DataFrame]:
    """
    Load a raw CSV file produced by ``fetch_data.py``.

    Parameters
    ----------
    filepath:
        Absolute or relative path to the raw CSV.

    Returns
    -------
    pd.DataFrame
        The file's contents as a DataFrame.
    None
        If the file is missing or cannot be parsed.
    """
    if not os.path.exists(filepath):
        logger.error("Raw file not found: %s", filepath)
        return None
    try:
        df = pd.read_csv(filepath, low_memory=False)
        logger.info("Loaded %d rows from %s.", len(df), filepath)
        return df
    except Exception as exc:
        logger.error("Could not read %s: %s", filepath, exc)
        return None


def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns to a canonical snake_case format.

    Applies :data:`COLUMN_RENAME_MAP` first, then force-lowercases any
    remaining column names (replacing spaces with underscores).  This
    guarantees that downstream code can always reference ``df['close']``
    regardless of what yfinance happened to call it.

    Parameters
    ----------
    df:
        Raw DataFrame — columns may be in any casing.

    Returns
    -------
    DataFrame with normalised column names.
    """
    df = df.rename(columns=COLUMN_RENAME_MAP)
    df.columns = [col.lower().replace(" ", "_") for col in df.columns]
    logger.debug("Standardised columns: %s", df.columns.tolist())
    return df


def parse_dates(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Convert the date column to ``datetime64[ns]``.

    Parameters
    ----------
    df:
        DataFrame that must contain *date_col*.
    date_col:
        Name of the column holding date strings.

    Returns
    -------
    DataFrame with *date_col* as a proper datetime column.

    Raises
    ------
    KeyError
        If *date_col* is absent from *df*.
    ValueError
        If the column cannot be parsed as dates.
    """
    if date_col not in df.columns:
        raise KeyError(
            f"Date column '{date_col}' not found. Available: {df.columns.tolist()}"
        )
    df[date_col] = pd.to_datetime(df[date_col], utc=False, errors="coerce")
    unparseable = df[date_col].isna().sum()
    if unparseable:
        logger.warning("%d date value(s) could not be parsed and are NaT.", unparseable)
    else:
        logger.info("Date column parsed successfully (%d rows).", len(df))
    return df


def sort_by_date(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Sort *df* by *date_col* ascending and reset the integer index.

    Parameters
    ----------
    df:
        DataFrame with a datetime *date_col*.
    date_col:
        Column to sort on.

    Returns
    -------
    Sorted DataFrame with a clean 0-based RangeIndex.
    """
    df = df.sort_values(by=date_col, ascending=True).reset_index(drop=True)
    logger.info("Sorted %d rows by date (ascending).", len(df))
    return df


def remove_duplicates(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Drop rows where *date_col* is duplicated, keeping the first occurrence.

    Parameters
    ----------
    df:
        DataFrame (must contain *date_col*).
    date_col:
        Column used to identify duplicates.

    Returns
    -------
    De-duplicated DataFrame.
    """
    before = len(df)
    df = df.drop_duplicates(subset=[date_col], keep="first").reset_index(drop=True)
    removed = before - len(df)
    if removed:
        logger.warning("Removed %d duplicate date row(s).", removed)
    else:
        logger.info("No duplicate rows detected.")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute and report missing values in price columns.

    Strategy
    --------
    * **Forward-fill** — propagates the last known price across market holidays
      or data gaps (standard practice for OHLCV data).
    * **Drop** — any row still missing after forward-fill (e.g. NaNs at the
      very beginning of the series) is removed entirely.

    Parameters
    ----------
    df:
        DataFrame with canonical price column names.

    Returns
    -------
    DataFrame with no missing values in PRICE_COLUMNS (unless the entire
    column was empty, which triggers a warning).
    """
    target_cols = [c for c in PRICE_COLUMNS if c in df.columns]
    total_missing = df[target_cols].isna().sum().sum()

    if total_missing == 0:
        logger.info("No missing values in price columns.")
        return df

    # Per-column breakdown for transparency
    per_col = df[target_cols].isna().sum()
    logger.warning(
        "Missing values before fill:\n%s", per_col[per_col > 0].to_string()
    )

    df[target_cols] = df[target_cols].ffill()   # pandas 2.x+ API

    still_missing = df[target_cols].isna().sum().sum()
    if still_missing:
        logger.warning(
            "%d row(s) still contain NaN after forward-fill — dropping them.",
            still_missing,
        )
        df = df.dropna(subset=target_cols).reset_index(drop=True)

    logger.info(
        "Missing value handling complete. Remaining rows: %d.", len(df)
    )
    return df


def enforce_positive_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows where any OHLC column contains a zero or negative value.

    Zero/negative prices indicate corrupted or pre-IPO placeholder data
    that would distort every downstream calculation.

    Parameters
    ----------
    df:
        DataFrame with canonical column names.

    Returns
    -------
    Filtered DataFrame where all OHLC prices are strictly positive.
    """
    ohlc_cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]
    invalid_mask = (df[ohlc_cols] <= 0).any(axis=1)
    n_invalid = invalid_mask.sum()
    if n_invalid:
        logger.warning(
            "Dropping %d row(s) with non-positive OHLC values.", n_invalid
        )
        df = df[~invalid_mask].reset_index(drop=True)
    else:
        logger.info("All OHLC prices are positive — no rows dropped.")
    return df


def select_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retain only the columns listed in :data:`REQUIRED_COLUMNS`.

    Any extra columns (e.g. ``adj_close``) are silently dropped so that
    all processed files share an identical schema.  A warning is emitted
    for any required column that is missing.

    Parameters
    ----------
    df:
        DataFrame after all cleaning steps.

    Returns
    -------
    DataFrame containing exactly the columns in REQUIRED_COLUMNS that are
    present in *df*.
    """
    available = [col for col in REQUIRED_COLUMNS if col in df.columns]
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        logger.warning("Required columns missing from data: %s", missing)
    return df[available]


# ──────────────────────────────────────────────────────────────────────────────
# Orchestration
# ──────────────────────────────────────────────────────────────────────────────
def preprocess_stock(
    stock_name: str,
    raw_dir: str = RAW_DATA_DIR,
    processed_dir: str = PROCESSED_DATA_DIR,
) -> Optional[pd.DataFrame]:
    """
    Run the full preprocessing pipeline for a single stock.

    Reads ``{raw_dir}/{STOCK_NAME}_raw.csv``, applies every cleaning step,
    prepends a ``ticker`` column, and writes the result to
    ``{processed_dir}/{STOCK_NAME}_processed.csv``.

    Parameters
    ----------
    stock_name:
        Display name (e.g. ``'TCS'``, ``'RELIANCE'``, ``'INFOSYS'``).
    raw_dir:
        Directory containing raw CSVs.
    processed_dir:
        Directory where processed CSVs will be written.

    Returns
    -------
    pd.DataFrame
        The cleaned DataFrame (also persisted to disk).
    None
        If loading or any critical step fails.
    """
    logger.info("─── Preprocessing: %s ───", stock_name)

    raw_filepath = os.path.join(raw_dir, f"{stock_name.upper()}_raw.csv")
    df = load_raw_csv(raw_filepath)
    if df is None:
        return None

    try:
        df = standardise_columns(df)
        df = parse_dates(df)
        df = sort_by_date(df)
        df = remove_duplicates(df)
        df = handle_missing_values(df)
        df = enforce_positive_prices(df)
        df = select_required_columns(df)
    except (KeyError, ValueError) as exc:
        logger.error("Preprocessing failed for %s: %s", stock_name, exc)
        return None

    # Prepend a stock identifier so multi-stock DataFrames stay traceable
    df.insert(0, "ticker", stock_name.upper())

    # Persist
    _ensure_directory(processed_dir)
    out_path = os.path.join(processed_dir, f"{stock_name.upper()}_processed.csv")
    df.to_csv(out_path, index=False)
    logger.info(
        "✓  %s processed: %d rows × %d cols → %s",
        stock_name, len(df), len(df.columns), out_path,
    )
    return df


def preprocess_all(
    stock_names: List[str] = STOCK_NAMES,
    raw_dir: str = RAW_DATA_DIR,
    processed_dir: str = PROCESSED_DATA_DIR,
) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Preprocess every stock in *stock_names*.

    Parameters
    ----------
    stock_names:
        List of display names to process.
    raw_dir:
        Source directory for raw CSVs.
    processed_dir:
        Destination directory for processed CSVs.

    Returns
    -------
    dict
        ``{ stock_name: DataFrame | None }`` for each stock.
    """
    results: Dict[str, Optional[pd.DataFrame]] = {}
    for name in stock_names:
        results[name] = preprocess_stock(name, raw_dir, processed_dir)

    succeeded = [k for k, v in results.items() if v is not None]
    failed = [k for k, v in results.items() if v is None]
    logger.info(
        "Preprocessing batch complete. OK: %s  |  Failed: %s", succeeded, failed
    )
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Script entry-point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Algorithmic Trading Backtester — Preprocessing")
    print("=" * 55)

    dfs = preprocess_all()

    print("\n── Preprocessing Results ──────────────────────────")
    for stock, df in dfs.items():
        if df is not None:
            print(f"\n  {stock}: {df.shape[0]} rows × {df.shape[1]} cols")
            print(df.head(3).to_string(index=False))
        else:
            print(f"\n  {stock}: ✗  FAILED")
    print()
