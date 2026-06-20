"""
take_profit.py
==============
Take-profit engine for the Algorithmic Trading Strategy Backtester.

Two take-profit modes
---------------------
1. **Fixed Percentage** — target = entry × (1 + pct).
   Calendar-agnostic; useful when you have a fixed return expectation.

2. **Risk-Reward (R-multiple)** — target = entry + (R × stop_distance).
   Derives the target from the stop-loss distance so risk and reward are
   always sized together.  A 2R target on a ₹100 stop means you aim for
   ₹200 profit before risking ₹100.  This is the professional standard.

Partial exits
-------------
:class:`PartialTakeProfitPlan` lets you book profits at multiple R levels
simultaneously, e.g.:

* Exit 50 % of position at 1R
* Exit 30 % of position at 2R
* Let 20 % ride with a trailing stop

This maximises per-trade expectancy by securing partial profits while
still participating in extended moves.

Architecture
------------
* :class:`TakeProfitResult` — immutable result object.
* :class:`FixedPercentageTarget` — stateless fixed-pct calculator.
* :class:`RiskRewardTarget` — stateless R-multiple calculator.
* :class:`PartialTakeProfitPlan` — multi-level exit plan.
* :class:`TakeProfitManager` — facade for strategy integration.

Integration with Phase 4
------------------------
The manager is designed to receive the ``StopLossResult`` from
``stop_loss.py`` and produce complementary ``TakeProfitResult`` objects:

::

    from src.risk.stop_loss    import StopLossManager, StopMode
    from src.risk.take_profit  import TakeProfitManager

    stop_mgr = StopLossManager(mode=StopMode.ATR, atr_multiplier=2.0)
    tp_mgr   = TakeProfitManager(r_multiple=2.0)

    stop_result = stop_mgr.calculate(entry_price=3_400, atr=85.0)
    tp_result   = tp_mgr.calculate_from_stop(stop_result)
    # tp_result.target_price → 3_740.0  (entry + 2 × stop_distance)

Usage
-----
::

    python src/risk/take_profit.py
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple

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
# Enums & value objects
# ──────────────────────────────────────────────────────────────────────────────
class TPMode(Enum):
    """Supported take-profit calculation modes."""
    FIXED_PCT   = auto()   # Fixed percentage above entry
    RISK_REWARD = auto()   # Derived from stop-loss distance × R-multiple


@dataclass(frozen=True)
class TakeProfitResult:
    """
    Immutable snapshot of a take-profit calculation.

    Attributes
    ----------
    target_price : float
        Absolute price level at which the take-profit triggers.
    distance : float
        ``target_price - entry_price`` in rupees.
    distance_pct : float
        ``distance / entry_price`` as a percentage.
    mode : TPMode
        Calculation mode used.
    entry_price : float
        Trade entry price.
    r_multiple : float or None
        Risk-reward ratio (``None`` for fixed-pct mode).
    stop_distance : float or None
        Stop-loss distance used in the R calculation (``None`` for fixed-pct).
    """
    target_price:  float
    distance:      float
    distance_pct:  float
    mode:          TPMode
    entry_price:   float
    r_multiple:    Optional[float] = field(default=None)
    stop_distance: Optional[float] = field(default=None)

    def __str__(self) -> str:
        r_str = (
            f"  R={self.r_multiple:.1f}  stop_dist=₹{self.stop_distance:.2f}"
            if self.r_multiple is not None else ""
        )
        return (
            f"TakeProfit [{self.mode.name}]  "
            f"entry=₹{self.entry_price:,.2f}  "
            f"target=₹{self.target_price:,.2f}  "
            f"dist=₹{self.distance:.2f} ({self.distance_pct:.2f}%)"
            f"{r_str}"
        )


@dataclass
class PartialExitLevel:
    """
    A single tier in a multi-level partial exit plan.

    Attributes
    ----------
    r_multiple : float
        R-level at which this tier triggers (e.g. 1.0, 2.0, 3.0).
    exit_fraction : float
        Fraction of the *remaining* position to close (0 < x ≤ 1).
    target_price : float
        Calculated target price for this tier.
    triggered : bool
        Whether this level has already been hit (mutable state).
    """
    r_multiple:     float
    exit_fraction:  float
    target_price:   float = 0.0
    triggered:      bool  = False

    def __post_init__(self) -> None:
        if not (0 < self.exit_fraction <= 1.0):
            raise ValueError(
                f"exit_fraction must be in (0, 1], got {self.exit_fraction}"
            )
        if self.r_multiple <= 0:
            raise ValueError(f"r_multiple must be > 0, got {self.r_multiple}")


# ──────────────────────────────────────────────────────────────────────────────
# Calculator 1 — Fixed Percentage Target
# ──────────────────────────────────────────────────────────────────────────────
class FixedPercentageTarget:
    """
    Take-profit at a fixed percentage above entry.

    Parameters
    ----------
    pct : float
        Target distance as a decimal fraction (e.g. ``0.05`` = 5 %).

    Examples
    --------
    >>> tp = FixedPercentageTarget(pct=0.05)
    >>> result = tp.calculate(entry_price=3_400)
    >>> result.target_price
    3570.0
    """

    def __init__(self, pct: float = 0.05) -> None:
        if not (0 < pct < 10):          # upper bound allows for > 100% targets
            raise ValueError(f"pct must be > 0, got {pct}")
        self.pct = pct

    def calculate(self, entry_price: float) -> TakeProfitResult:
        """
        Compute a fixed-percentage take-profit for a long entry.

        Parameters
        ----------
        entry_price : float
            Trade entry price in INR.

        Returns
        -------
        TakeProfitResult
        """
        if entry_price <= 0:
            raise ValueError(f"entry_price must be positive, got {entry_price}")

        target_price = entry_price * (1.0 + self.pct)
        distance     = target_price - entry_price

        result = TakeProfitResult(
            target_price  = round(target_price, 2),
            distance      = round(distance, 2),
            distance_pct  = self.pct * 100,
            mode          = TPMode.FIXED_PCT,
            entry_price   = entry_price,
        )
        logger.info("TPCalc  %s", result)
        return result


# ──────────────────────────────────────────────────────────────────────────────
# Calculator 2 — Risk-Reward Target
# ──────────────────────────────────────────────────────────────────────────────
class RiskRewardTarget:
    """
    Take-profit derived from the stop-loss distance × R-multiple.

    Formula::

        target_price = entry_price + (r_multiple × stop_distance)

    The stop distance should come from ``StopLossResult.distance`` so risk
    and reward are always calibrated against the same baseline.

    Parameters
    ----------
    r_multiple : float
        Risk-reward ratio.  Common values:
        * 1.5R — conservative  (high win-rate needed)
        * 2.0R — balanced (default)
        * 3.0R — aggressive (needs lower win-rate to be profitable)
        * 4.0R — trend-following (few winners, large payoffs)

    Why R-multiples matter
    ----------------------
    A strategy with 40 % win rate and 2R average win breaks even at:
    ``(0.4 × 2R) − (0.6 × 1R) = 0.8R − 0.6R = +0.2R per trade``
    That is, it is profitable without ever exceeding 40 % accuracy.

    Examples
    --------
    >>> tp = RiskRewardTarget(r_multiple=2.0)
    >>> result = tp.calculate(entry_price=3_400, stop_distance=170.0)
    >>> result.target_price
    3740.0
    """

    def __init__(self, r_multiple: float = 2.0) -> None:
        if r_multiple <= 0:
            raise ValueError(f"r_multiple must be > 0, got {r_multiple}")
        self.r_multiple = r_multiple

    def calculate(
        self,
        entry_price: float,
        stop_distance: float,
    ) -> TakeProfitResult:
        """
        Compute an R-multiple take-profit.

        Parameters
        ----------
        entry_price : float
            Trade entry price in INR.
        stop_distance : float
            Distance from entry to stop (positive, in INR).

        Returns
        -------
        TakeProfitResult

        Raises
        ------
        ValueError
            If *stop_distance* ≤ 0.
        """
        if entry_price <= 0:
            raise ValueError(f"entry_price must be positive, got {entry_price}")
        if stop_distance <= 0:
            raise ValueError(
                f"stop_distance must be positive, got {stop_distance}. "
                "Pass StopLossResult.distance."
            )

        distance     = self.r_multiple * stop_distance
        target_price = entry_price + distance

        result = TakeProfitResult(
            target_price  = round(target_price, 2),
            distance      = round(distance, 2),
            distance_pct  = (distance / entry_price) * 100,
            mode          = TPMode.RISK_REWARD,
            entry_price   = entry_price,
            r_multiple    = self.r_multiple,
            stop_distance = stop_distance,
        )
        logger.info("TPCalc  %s", result)
        return result

    def required_win_rate(self) -> float:
        """
        Minimum win rate (%) needed to break even at this R-multiple.

        Derived from the break-even equation::

            win_rate = 1 / (1 + r_multiple)

        Returns
        -------
        float
            Break-even win rate as a percentage.
        """
        return (1.0 / (1.0 + self.r_multiple)) * 100


# ──────────────────────────────────────────────────────────────────────────────
# Partial Take-Profit Plan
# ──────────────────────────────────────────────────────────────────────────────
class PartialTakeProfitPlan:
    """
    Multi-level exit plan that books profits at successive R-multiples.

    Common plan for a swing trade::

        1R → exit 50 %  (secured profit, remove risk)
        2R → exit 30 %  (partial capture of trend)
        3R+ → trail remainder with stop (let the winner ride)

    Parameters
    ----------
    levels : list of (r_multiple, exit_fraction) tuples
        Ordered by ascending R level.  Fractions should sum to ≤ 1.0.

    Examples
    --------
    >>> plan = PartialTakeProfitPlan(levels=[(1.0, 0.5), (2.0, 0.3)])
    >>> plan.initialise(entry_price=3_400, stop_distance=170)
    >>> plan.check(current_price=3_570)   # at 1R → (50% exit triggered)
    """

    def __init__(
        self,
        levels: List[Tuple[float, float]] = None,  # type: ignore[assignment]
    ) -> None:
        if levels is None:
            levels = [(1.0, 0.50), (2.0, 0.30), (3.0, 0.20)]

        total_fraction = sum(f for _, f in levels)
        if total_fraction > 1.0 + 1e-9:
            raise ValueError(
                f"Exit fractions sum to {total_fraction:.2f} — must be ≤ 1.0. "
                f"The remainder is held/trailed."
            )

        self._raw_levels = sorted(levels, key=lambda x: x[0])  # ascending R
        self.exit_levels: List[PartialExitLevel] = []
        self.entry_price:   Optional[float] = None
        self.stop_distance: Optional[float] = None

    def initialise(self, entry_price: float, stop_distance: float) -> None:
        """
        Compute target prices for each level given the entry and stop distance.

        Parameters
        ----------
        entry_price : float
            Trade entry price.
        stop_distance : float
            Stop-loss distance (positive, in INR).
        """
        if entry_price <= 0 or stop_distance <= 0:
            raise ValueError("entry_price and stop_distance must be positive.")

        self.entry_price   = entry_price
        self.stop_distance = stop_distance
        self.exit_levels   = []

        for r, frac in self._raw_levels:
            target = entry_price + r * stop_distance
            self.exit_levels.append(
                PartialExitLevel(
                    r_multiple    = r,
                    exit_fraction = frac,
                    target_price  = round(target, 2),
                )
            )
            logger.info(
                "PartialTP  Level %.1fR  target=₹%.2f  exit=%.0f%%",
                r, target, frac * 100,
            )

    def check(
        self, current_price: float
    ) -> List[PartialExitLevel]:
        """
        Check which (if any) levels have been newly triggered.

        Parameters
        ----------
        current_price : float
            Latest price.

        Returns
        -------
        list of PartialExitLevel
            Newly triggered levels (empty list if none).
        """
        if not self.exit_levels:
            raise RuntimeError("Call initialise() before check().")

        triggered = []
        for level in self.exit_levels:
            if not level.triggered and current_price >= level.target_price:
                level.triggered = True
                triggered.append(level)
                logger.info(
                    "PartialTP HIT  %.1fR  price=₹%.2f ≥ target=₹%.2f  "
                    "exit=%.0f%% of position",
                    level.r_multiple, current_price,
                    level.target_price, level.exit_fraction * 100,
                )
        return triggered

    def reset(self) -> None:
        """Clear all state — call before a new trade."""
        self.exit_levels   = []
        self.entry_price   = None
        self.stop_distance = None

    @property
    def remaining_fraction(self) -> float:
        """Fraction of the original position still open."""
        exited = sum(
            lvl.exit_fraction for lvl in self.exit_levels if lvl.triggered
        )
        return max(1.0 - exited, 0.0)

    @property
    def levels_hit(self) -> int:
        """Number of levels that have been triggered."""
        return sum(1 for lvl in self.exit_levels if lvl.triggered)


# ──────────────────────────────────────────────────────────────────────────────
# Facade — TakeProfitManager
# ──────────────────────────────────────────────────────────────────────────────
class TakeProfitManager:
    """
    High-level facade that strategies interact with for take-profit logic.

    Supports both single-target and partial-exit modes.

    Parameters
    ----------
    mode : TPMode
        Calculation mode.  Default: ``TPMode.RISK_REWARD``.
    pct : float
        Fixed-percentage target.  Default: ``0.05`` (5 %).
    r_multiple : float
        R-multiple for risk-reward mode.  Default: ``2.0``.
    partial_levels : list of (r, fraction) tuples or None
        If provided, enables partial exits instead of a single target.

    Examples
    --------
    Single target::

        mgr = TakeProfitManager(mode=TPMode.RISK_REWARD, r_multiple=2.0)
        result = mgr.calculate_from_stop(stop_result)

    Partial exits::

        mgr = TakeProfitManager(
            partial_levels=[(1.0, 0.5), (2.0, 0.3), (3.0, 0.2)]
        )
        mgr.initialise_partial(entry_price=3_400, stop_distance=170)
        hit = mgr.check_partial(current_price=3_570)
    """

    def __init__(
        self,
        mode: TPMode                           = TPMode.RISK_REWARD,
        pct: float                             = 0.05,
        r_multiple: float                      = 2.0,
        partial_levels: Optional[
            List[Tuple[float, float]]
        ]                                      = None,
    ) -> None:
        self.mode             = mode
        self._pct_calc        = FixedPercentageTarget(pct=pct)
        self._rr_calc         = RiskRewardTarget(r_multiple=r_multiple)
        self._partial: Optional[PartialTakeProfitPlan] = (
            PartialTakeProfitPlan(levels=partial_levels)
            if partial_levels is not None else None
        )
        self.last_result: Optional[TakeProfitResult] = None

    def calculate(self, entry_price: float) -> TakeProfitResult:
        """
        Compute a fixed-pct take-profit (does not require stop-loss info).

        Parameters
        ----------
        entry_price : float
            Trade entry price.
        """
        result = self._pct_calc.calculate(entry_price)
        self.last_result = result
        return result

    def calculate_from_stop(self, stop_result) -> TakeProfitResult:     # noqa: ANN001
        """
        Compute an R-multiple take-profit derived from a ``StopLossResult``.

        Parameters
        ----------
        stop_result : StopLossResult
            Result object from ``StopLossManager.calculate()``.

        Returns
        -------
        TakeProfitResult
        """
        if self.mode == TPMode.FIXED_PCT:
            result = self._pct_calc.calculate(stop_result.entry_price)
        else:
            result = self._rr_calc.calculate(
                entry_price   = stop_result.entry_price,
                stop_distance = stop_result.distance,
            )
        self.last_result = result
        return result

    def calculate_rr(
        self, entry_price: float, stop_distance: float
    ) -> TakeProfitResult:
        """
        Compute an R-multiple take-profit directly from numeric inputs.

        Parameters
        ----------
        entry_price : float
        stop_distance : float
            Distance from entry to stop in INR.
        """
        result = self._rr_calc.calculate(entry_price, stop_distance)
        self.last_result = result
        return result

    def is_hit(self, current_price: float) -> bool:
        """
        Return ``True`` if *current_price* has reached the last target.

        Parameters
        ----------
        current_price : float
            Latest price.
        """
        if self.last_result is None:
            return False
        hit = current_price >= self.last_result.target_price
        if hit:
            logger.info(
                "TakeProfit HIT  price=₹%.2f ≥ target=₹%.2f  "
                "| gain=₹%.2f (%.2f%%)",
                current_price,
                self.last_result.target_price,
                current_price - self.last_result.entry_price,
                (current_price - self.last_result.entry_price)
                / self.last_result.entry_price * 100,
            )
        return hit

    def initialise_partial(
        self, entry_price: float, stop_distance: float
    ) -> None:
        """Initialise the partial take-profit plan."""
        if self._partial is None:
            raise RuntimeError(
                "No partial_levels provided — pass partial_levels to __init__."
            )
        self._partial.initialise(entry_price, stop_distance)

    def check_partial(
        self, current_price: float
    ) -> List[PartialExitLevel]:
        """Check which partial exit levels have been newly triggered."""
        if self._partial is None:
            raise RuntimeError("Partial plan not configured.")
        return self._partial.check(current_price)

    @property
    def partial_remaining(self) -> Optional[float]:
        """Remaining position fraction for partial-exit mode."""
        return self._partial.remaining_fraction if self._partial else None

    def break_even_win_rate(self) -> Optional[float]:
        """Minimum win rate to be profitable at the current R-multiple."""
        if self.mode == TPMode.RISK_REWARD:
            return self._rr_calc.required_win_rate()
        return None

    def reset(self) -> None:
        """Clear per-trade state."""
        self.last_result = None
        if self._partial:
            self._partial.reset()


# ──────────────────────────────────────────────────────────────────────────────
# Vectorised helper
# ──────────────────────────────────────────────────────────────────────────────
def add_rr_target_column(
    df: pd.DataFrame,
    stop_col: str     = "stop_atr",
    price_col: str    = "close",
    r_multiple: float = 2.0,
    col_name: str     = "target_2r",
) -> pd.DataFrame:
    """
    Add an R-multiple take-profit column to a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain *price_col* and *stop_col*.
    stop_col : str
        Stop-loss price column (from ``stop_loss.add_atr_stop_column``).
    price_col : str
        Entry/current price column.
    r_multiple : float
        R-multiple for the target.
    col_name : str
        Output column name.

    Returns
    -------
    pd.DataFrame
        Copy with *col_name* added.
    """
    for col in (price_col, stop_col):
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")
    df = df.copy()
    stop_distance  = df[price_col] - df[stop_col]
    df[col_name]   = (df[price_col] + r_multiple * stop_distance).round(2)
    logger.info(
        "Added '%s' column: %.1fR take-profit above '%s'.",
        col_name, r_multiple, price_col,
    )
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Script entry-point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("  Take-Profit Engine — Demo")
    print("=" * 65)

    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)
    ))))
    from src.indicators.atr  import add_atr
    from src.risk.stop_loss  import StopLossManager, StopMode

    PROCESSED_DIR = os.path.join("data", "processed")
    STOCKS = ["TCS", "RELIANCE", "INFOSYS"]

    for stock in STOCKS:
        path = os.path.join(PROCESSED_DIR, f"{stock}_processed.csv")
        if not os.path.exists(path):
            print(f"  ✗  {stock}: not found")
            continue

        df  = pd.read_csv(path, parse_dates=["date"])
        df  = add_atr(df, period=14)
        row = df.iloc[-1]

        entry   = row["close"]
        atr_val = row["atr_14"]

        # Stop-loss first (needed for R-multiple targets)
        stop_mgr    = StopLossManager(mode=StopMode.ATR, atr_multiplier=2.0)
        stop_result = stop_mgr.calculate(entry_price=entry, atr=atr_val)

        print(f"\n{'─' * 60}")
        print(f"  {stock}  close=₹{entry:,.2f}  "
              f"stop=₹{stop_result.stop_price:,.2f}  "
              f"dist=₹{stop_result.distance:.2f}")
        print(f"{'─' * 60}")

        # Single targets at various R-multiples
        print("  Single-target R-multiples:")
        hdr = f"  {'R':>5} {'Target ₹':>12} {'Gain ₹':>10} {'Gain%':>8} {'BEW%':>8}"
        print(hdr)
        print("  " + "─" * (len(hdr) - 2))
        for r in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
            mgr = TakeProfitManager(mode=TPMode.RISK_REWARD, r_multiple=r)
            tp  = mgr.calculate_from_stop(stop_result)
            bew = mgr.break_even_win_rate()
            print(
                f"  {r:>4.1f}R  "
                f"₹{tp.target_price:>11,.2f}  "
                f"₹{tp.distance:>9.2f}  "
                f"{tp.distance_pct:>7.2f}%  "
                f"{bew:>7.1f}%"
            )

        # Partial exit plan simulation
        print(f"\n  Partial exit plan  (1R→50%  2R→30%  3R→20%):")
        partial_mgr = TakeProfitManager(
            partial_levels=[(1.0, 0.50), (2.0, 0.30), (3.0, 0.20)]
        )
        partial_mgr.initialise_partial(
            entry_price   = entry,
            stop_distance = stop_result.distance,
        )
        # Simulate price walking up through all three levels
        sim_prices = [
            entry + 0.5 * stop_result.distance,
            entry + 1.0 * stop_result.distance,
            entry + 1.8 * stop_result.distance,
            entry + 2.0 * stop_result.distance,
            entry + 2.6 * stop_result.distance,
            entry + 3.0 * stop_result.distance,
        ]
        for sim_price in sim_prices:
            hits = partial_mgr.check_partial(sim_price)
            for h in hits:
                print(
                    f"    ✓  price=₹{sim_price:>10,.2f}  "
                    f"→  {h.r_multiple:.1f}R HIT  "
                    f"exit {h.exit_fraction*100:.0f}%  "
                    f"remaining={partial_mgr.partial_remaining*100:.0f}%"
                )

    # Vectorised demo
    print("\n── Vectorised columns on TCS (last 5 rows) ──────────────────────")
    from src.risk.stop_loss import add_atr_stop_column
    df_tcs = pd.read_csv(
        os.path.join(PROCESSED_DIR, "TCS_processed.csv"), parse_dates=["date"]
    )
    df_tcs = add_atr(df_tcs, period=14)
    df_tcs = add_atr_stop_column(df_tcs, multiplier=2.0)
    df_tcs = add_rr_target_column(df_tcs, r_multiple=2.0)
    print(df_tcs[["date", "close", "stop_atr", "target_2r"]].tail(5).to_string(index=False))
    print()
