"""
validator.py
============
Quality-gates the processed CSV files produced by ``preprocess.py`` before
any of the data reaches the indicator or strategy layers.

Design
------
Each check is an independent function that writes to a :class:`ValidationResult`
object.  This design means:

* Every check always runs (no short-circuit on first failure) so the full
  picture is visible at once.
* Checks are trivially unit-testable in isolation.
* The caller receives a structured object, not just a boolean, enabling
  downstream code to react to *specific* failure reasons.

Typical usage
-------------
Run as a script from the project root::

    python src/data/validator.py

Or gate the indicator layer in a pipeline::

    from src.data.validator import validate_all
    report = validate_all()
    if not all(r.passed for r in report.values()):
        raise RuntimeError("Data quality checks failed — aborting pipeline.")
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

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
PROCESSED_DATA_DIR: str = os.path.join("data", "processed")

#: Names must match ``preprocess.STOCK_NAMES``.
STOCK_NAMES: List[str] = ["TCS", "RELIANCE", "INFOSYS"]

#: Schema that every processed file must satisfy.
REQUIRED_COLUMNS: List[str] = ["date", "open", "high", "low", "close", "volume"]

#: Columns that must contain strictly positive values.
PRICE_COLUMNS: List[str] = ["open", "high", "low", "close"]

#: Minimum row count required for the 200-period moving average to be meaningful.
MIN_ROWS: int = 200


# ──────────────────────────────────────────────────────────────────────────────
# Validation result container
# ──────────────────────────────────────────────────────────────────────────────
class ValidationResult:
    """
    Collects pass/fail status and a human-readable issue list for one stock.

    Attributes
    ----------
    stock_name:
        Display name of the stock being validated.
    passed:
        ``True`` if no *fail* was recorded; ``False`` as soon as any *fail*
        is recorded.  Warnings do not change this flag.
    issues:
        Ordered list of strings, prefixed with ``'[FAIL]'`` or ``'[WARN]'``.
    """

    def __init__(self, stock_name: str) -> None:
        self.stock_name: str = stock_name
        self.passed: bool = True
        self.issues: List[str] = []

    # ── Mutators ─────────────────────────────────────────────────────────────
    def fail(self, reason: str) -> None:
        """Record a *critical* failure (sets :attr:`passed` to ``False``)."""
        self.passed = False
        self.issues.append(f"[FAIL]  {reason}")
        logger.error("[%s] FAIL — %s", self.stock_name, reason)

    def warn(self, reason: str) -> None:
        """Record a *non-blocking* warning (does **not** set :attr:`passed` to False)."""
        self.issues.append(f"[WARN]  {reason}")
        logger.warning("[%s] WARN — %s", self.stock_name, reason)

    def ok(self, check_name: str) -> None:
        """Log a passing check (does not add to :attr:`issues`)."""
        logger.info("[%s] OK   — %s", self.stock_name, check_name)

    # ── Representation ────────────────────────────────────────────────────────
    def __repr__(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        return (
            f"ValidationResult(stock='{self.stock_name}', "
            f"status={status}, issues={len(self.issues)})"
        )

    def summary(self) -> str:
        """Return a multi-line human-readable summary."""
        status = "✓ PASSED" if self.passed else "✗ FAILED"
        lines = [f"{self.stock_name}: {status}"]
        for issue in self.issues:
            lines.append(f"    → {issue}")
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Loader
# ──────────────────────────────────────────────────────────────────────────────
def load_processed_csv(
    stock_name: str,
    processed_dir: str = PROCESSED_DATA_DIR,
) -> Optional[pd.DataFrame]:
    """
    Load the processed CSV for *stock_name*.

    Parameters
    ----------
    stock_name:
        Display name (e.g. ``'TCS'``).
    processed_dir:
        Directory containing processed CSVs.

    Returns
    -------
    pd.DataFrame or None
    """
    filename = f"{stock_name.upper()}_processed.csv"
    filepath = os.path.join(processed_dir, filename)

    if not os.path.exists(filepath):
        logger.error("Processed file not found: %s", filepath)
        return None

    try:
        df = pd.read_csv(filepath, parse_dates=["date"])
        logger.debug("Loaded %d rows from %s.", len(df), filepath)
        return df
    except Exception as exc:
        logger.error("Error loading %s: %s", filepath, exc)
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Individual check functions
# ──────────────────────────────────────────────────────────────────────────────
def check_required_columns(df: pd.DataFrame, result: ValidationResult) -> None:
    """Verify that every column in :data:`REQUIRED_COLUMNS` is present."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        result.fail(f"Missing required columns: {missing}")
    else:
        result.ok("Required columns present")


def check_date_validity(df: pd.DataFrame, result: ValidationResult) -> None:
    """
    Confirm that the ``date`` column:

    * exists
    * is parsed as ``datetime64`` (not object/string)
    * contains no null values
    """
    if "date" not in df.columns:
        result.fail("Column 'date' is absent — cannot run date checks.")
        return

    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        result.fail(
            f"Column 'date' dtype is '{df['date'].dtype}' — expected datetime64."
        )
    else:
        result.ok("Date dtype is datetime64")

    null_dates = df["date"].isna().sum()
    if null_dates:
        result.fail(f"Column 'date' has {null_dates} null / NaT value(s).")
    else:
        result.ok("No null dates")


def check_chronological_order(df: pd.DataFrame, result: ValidationResult) -> None:
    """Ensure the data is sorted in strictly ascending date order."""
    if "date" not in df.columns:
        return
    if not df["date"].is_monotonic_increasing:
        result.fail("Data is NOT in ascending chronological order.")
    else:
        result.ok("Data is in chronological order")


def check_positive_prices(df: pd.DataFrame, result: ValidationResult) -> None:
    """Assert every OHLC column is strictly greater than zero."""
    all_positive = True
    for col in PRICE_COLUMNS:
        if col not in df.columns:
            result.warn(f"Column '{col}' absent — positivity check skipped.")
            continue
        n_bad = (df[col] <= 0).sum()
        if n_bad:
            result.fail(f"Column '{col}' contains {n_bad} non-positive value(s).")
            all_positive = False
    if all_positive:
        result.ok("All OHLC prices are strictly positive")


def check_no_duplicates(df: pd.DataFrame, result: ValidationResult) -> None:
    """Confirm there are no rows with a repeated date."""
    if "date" not in df.columns:
        return
    n_dupes = df.duplicated(subset=["date"]).sum()
    if n_dupes:
        result.fail(f"{n_dupes} duplicate date row(s) found.")
    else:
        result.ok("No duplicate dates")


def check_missing_values(df: pd.DataFrame, result: ValidationResult) -> None:
    """
    Report missing values per column.

    Missing values in **price columns** are *failures* (they should have been
    eliminated by preprocessing).  Missing values in other columns are
    *warnings*.
    """
    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]

    if cols_with_nulls.empty:
        result.ok("No missing values in any column")
        return

    for col, count in cols_with_nulls.items():
        msg = f"Column '{col}' has {count} missing value(s)."
        if col in PRICE_COLUMNS:
            result.fail(msg)
        else:
            result.warn(msg)


def check_high_low_consistency(df: pd.DataFrame, result: ValidationResult) -> None:
    """
    Validate that ``high >= low`` on every bar.

    Any row where ``high < low`` signals a data corruption issue.
    """
    if "high" not in df.columns or "low" not in df.columns:
        return
    violations = (df["high"] < df["low"]).sum()
    if violations:
        result.fail(
            f"{violations} bar(s) where 'high' < 'low' — likely data corruption."
        )
    else:
        result.ok("High ≥ Low on all bars")


def check_minimum_row_count(
    df: pd.DataFrame,
    result: ValidationResult,
    min_rows: int = MIN_ROWS,
) -> None:
    """
    Confirm the dataset has enough rows for all indicator calculations.

    The 200-period SMA (the longest default window) needs at least 200 data
    points to produce a single valid value.
    """
    n = len(df)
    if n < min_rows:
        result.fail(
            f"Only {n} rows available; minimum required is {min_rows} "
            f"(needed for 200-period indicators)."
        )
    else:
        result.ok(f"Row count {n} ≥ {min_rows} (sufficient for all indicators)")


def check_date_range(df: pd.DataFrame, result: ValidationResult) -> None:
    """
    Log the actual date range covered by the dataset.

    This is an *informational* check — it warns but never fails, since date
    ranges are driven by the fetch configuration.
    """
    if "date" not in df.columns or not pd.api.types.is_datetime64_any_dtype(df["date"]):
        return
    date_min = df["date"].min()
    date_max = df["date"].max()
    n_years = (date_max - date_min).days / 365.25
    msg = f"Date range: {date_min.date()} → {date_max.date()} ({n_years:.1f} years)"
    result.warn(msg) if n_years < 2 else result.ok(msg)


# ──────────────────────────────────────────────────────────────────────────────
# Orchestration
# ──────────────────────────────────────────────────────────────────────────────
def validate_stock(
    stock_name: str,
    processed_dir: str = PROCESSED_DATA_DIR,
) -> Tuple[bool, ValidationResult]:
    """
    Run the complete validation suite against one stock's processed CSV.

    Parameters
    ----------
    stock_name:
        Display name (e.g. ``'TCS'``).
    processed_dir:
        Directory containing processed CSVs.

    Returns
    -------
    Tuple[bool, ValidationResult]
        A ``(passed, result)`` pair where *passed* mirrors
        ``result.passed``.
    """
    logger.info("─── Validating: %s ───", stock_name)
    result = ValidationResult(stock_name)

    df = load_processed_csv(stock_name, processed_dir)
    if df is None:
        result.fail("Processed CSV could not be loaded.")
        return False, result

    # Run every check — order matters for readability of the report
    check_required_columns(df, result)
    check_date_validity(df, result)
    check_chronological_order(df, result)
    check_positive_prices(df, result)
    check_high_low_consistency(df, result)
    check_no_duplicates(df, result)
    check_missing_values(df, result)
    check_minimum_row_count(df, result)
    check_date_range(df, result)

    if result.passed:
        logger.info("[%s] ✓ All checks PASSED.", stock_name)
    else:
        fail_count = sum(1 for i in result.issues if i.startswith("[FAIL]"))
        logger.error("[%s] ✗ Validation FAILED — %d critical issue(s).", stock_name, fail_count)

    return result.passed, result


def validate_all(
    stock_names: List[str] = STOCK_NAMES,
    processed_dir: str = PROCESSED_DATA_DIR,
) -> Dict[str, ValidationResult]:
    """
    Run the full validation suite for every stock in *stock_names*.

    Parameters
    ----------
    stock_names:
        List of display names to validate.
    processed_dir:
        Directory containing processed CSVs.

    Returns
    -------
    dict
        ``{ stock_name: ValidationResult }`` for every stock.
    """
    report: Dict[str, ValidationResult] = {}
    for name in stock_names:
        _, vr = validate_stock(name, processed_dir)
        report[name] = vr

    passed = [k for k, v in report.items() if v.passed]
    failed = [k for k, v in report.items() if not v.passed]

    logger.info("=" * 55)
    logger.info("Validation complete.  PASSED: %s  |  FAILED: %s", passed, failed)
    logger.info("=" * 55)
    return report


def all_passed(report: Dict[str, ValidationResult]) -> bool:
    """
    Convenience function: returns ``True`` only if every stock passed.

    Parameters
    ----------
    report:
        Dict returned by :func:`validate_all`.
    """
    return all(vr.passed for vr in report.values())


# ──────────────────────────────────────────────────────────────────────────────
# Script entry-point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Algorithmic Trading Backtester — Validation")
    print("=" * 55)

    results = validate_all()

    print("\n── Validation Report ──────────────────────────────")
    for stock, vr in results.items():
        print()
        print(vr.summary())

    print()
    overall = "ALL CHECKS PASSED ✓" if all_passed(results) else "SOME CHECKS FAILED ✗"
    print(f"\n  Overall: {overall}")
    print()
