"""
stop_loss.py
============
Stop-loss engine for the Algorithmic Trading Strategy Backtester.

Three stop-loss modes
---------------------
1. **Fixed Percentage** — stop = entry × (1 − pct).
   Simple, calendar-agnostic.  Suitable when volatility is stable.

2. **ATR-based** — stop = entry − (k × ATR).
   Adapts to current market volatility.  Wider in choppy markets,
   tighter in calm ones.  Preferred for Indian large-caps where
   ATR is ~1.5–3% of price.

3. **Trailing Stop** — starts as either of the above, then *ratchets
   upward* bar-by-bar as price rises, locking in profits while never
   moving down.

Architecture
------------
* :class:`StopLossResult` — immutable value object returned by every
  calculator.  Carries the stop price, distance, distance-%, and mode.
* Three calculator classes  (``FixedPercentageStop``, ``ATRStop``,
  ``TrailingStop``) — stateless (fixed) or stateful (trailing).
* :class:`StopLossManager` — facade that picks the right calculator
  and manages state; this is what strategies import.
* :func:`backtrader_stop_mixin` — function that patches a Backtrader
  strategy class with stop-loss awareness; used by Phase 3 strategies.

Integration with Phase 3
------------------------
::

    from src.risk.stop_loss import StopLossManager, StopMode

    mgr = StopLossManager(mode=StopMode.ATR, atr_multiplier=2.0)
    result = mgr.calculate(entry_price=3_400, atr=85.0)
    # result.stop_price → 3230.0

Inside a Backtrader strategy ``next()``::

    if self.stop_mgr.is_hit(current_price):
        self.close()

Usage
-----
::

    python src/risk/stop_loss.py
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum, auto
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
# Enums & value objects
# ──────────────────────────────────────────────────────────────────────────────
class StopMode(Enum):
    """Supported stop-loss calculation modes."""
    FIXED_PCT = auto()   # Fixed percentage below entry
    ATR       = auto()   # ATR-based distance
    TRAILING  = auto()   # Ratcheting trailing stop (ATR or pct seed)


class TrailSeed(Enum):
    """What seeds the trailing stop's initial level."""
    FIXED_PCT = auto()
    ATR       = auto()


@dataclass(frozen=True)
class StopLossResult:
    """
    Immutable snapshot of a stop-loss calculation.

    Attributes
    ----------
    stop_price : float
        The absolute price level at which the stop triggers.
    distance : float
        ``entry_price - stop_price`` in rupees.
    distance_pct : float
        ``distance / entry_price`` as a percentage (0–100).
    mode : StopMode
        Which calculation mode produced this result.
    entry_price : float
        The entry price used in the calculation.
    atr_used : float or None
        The ATR value used (``None`` for fixed-pct stops).
    """
    stop_price:   float
    distance:     float
    distance_pct: float
    mode:         StopMode
    entry_price:  float
    atr_used:     Optional[float] = field(default=None)

    def __str__(self) -> str:
        atr_str = f"  ATR: ₹{self.atr_used:.2f}" if self.atr_used else ""
        return (
            f"StopLoss [{self.mode.name}]  "
            f"entry=₹{self.entry_price:,.2f}  "
            f"stop=₹{self.stop_price:,.2f}  "
            f"dist=₹{self.distance:.2f} ({self.distance_pct:.2f}%)"
            f"{atr_str}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Calculator 1 — Fixed Percentage Stop
# ──────────────────────────────────────────────────────────────────────────────
class FixedPercentageStop:
    """
    Stop-loss placed a fixed percentage below the entry price.

    Parameters
    ----------
    pct : float
        Stop distance as a decimal fraction (e.g. ``0.02`` = 2 %).
        Must be strictly between 0 and 1.

    Examples
    --------
    >>> stop = FixedPercentageStop(pct=0.02)
    >>> result = stop.calculate(entry_price=3_400)
    >>> result.stop_price
    3332.0
    """

    def __init__(self, pct: float = 0.02) -> None:
        if not (0 < pct < 1):
            raise ValueError(
                f"'pct' must be between 0 and 1 exclusive, received: {pct}"
            )
        self.pct = pct

    def calculate(self, entry_price: float) -> StopLossResult:
        """
        Compute stop price for a long entry.

        Parameters
        ----------
        entry_price : float
            Trade entry price in INR.

        Returns
        -------
        StopLossResult

        Raises
        ------
        ValueError
            If *entry_price* ≤ 0.
        """
        if entry_price <= 0:
            raise ValueError(f"entry_price must be positive, got {entry_price}")

        stop_price = entry_price * (1.0 - self.pct)
        distance   = entry_price - stop_price

        result = StopLossResult(
            stop_price   = round(stop_price, 2),
            distance     = round(distance, 2),
            distance_pct = self.pct * 100,
            mode         = StopMode.FIXED_PCT,
            entry_price  = entry_price,
        )
        logger.info("StopCalc  %s", result)
        return result


# ──────────────────────────────────────────────────────────────────────────────
# Calculator 2 — ATR Stop
# ──────────────────────────────────────────────────────────────────────────────
class ATRStop:
    """
    Stop-loss placed ``multiplier × ATR`` below the entry price.

    ATR-based stops automatically widen during volatile sessions and
    tighten during calm ones, avoiding premature exits from normal
    price noise.

    Parameters
    ----------
    multiplier : float
        ATR distance multiplier.  Common values:
        * 1.5 — tight (scalping / short-term)
        * 2.0 — moderate (swing trades, default)
        * 3.0 — wide  (position trades, earnings events)

    Examples
    --------
    >>> stop = ATRStop(multiplier=2.0)
    >>> result = stop.calculate(entry_price=3_400, atr=85.0)
    >>> result.stop_price
    3230.0
    """

    def __init__(self, multiplier: float = 2.0) -> None:
        if multiplier <= 0:
            raise ValueError(f"'multiplier' must be > 0, received: {multiplier}")
        self.multiplier = multiplier

    def calculate(self, entry_price: float, atr: float) -> StopLossResult:
        """
        Compute an ATR-based stop for a long entry.

        Parameters
        ----------
        entry_price : float
            Trade entry price in INR.
        atr : float
            Current ATR value (must be positive).

        Returns
        -------
        StopLossResult
        """
        if entry_price <= 0:
            raise ValueError(f"entry_price must be positive, got {entry_price}")
        if atr <= 0:
            raise ValueError(f"atr must be positive, got {atr}")

        distance   = self.multiplier * atr
        stop_price = entry_price - distance

        if stop_price <= 0:
            logger.warning(
                "ATR stop (₹%.2f) produced a non-positive stop price — "
                "clamping to ₹0.01.", stop_price
            )
            stop_price = 0.01

        result = StopLossResult(
            stop_price   = round(stop_price, 2),
            distance     = round(distance, 2),
            distance_pct = (distance / entry_price) * 100,
            mode         = StopMode.ATR,
            entry_price  = entry_price,
            atr_used     = atr,
        )
        logger.info("StopCalc  %s", result)
        return result


# ──────────────────────────────────────────────────────────────────────────────
# Calculator 3 — Trailing Stop
# ──────────────────────────────────────────────────────────────────────────────
class TrailingStop:
    """
    Ratcheting trailing stop — moves *up* with rising prices, never down.

    The stop is seeded at entry using either a fixed percentage or an
    ATR multiple.  After each price bar the stop is updated:

    ::

        new_stop = current_price − (seed_distance)
        stop     = max(stop, new_stop)          # ratchet: only ever moves up

    This locks in profits progressively while giving the trade room to
    breathe during normal pullbacks.

    Parameters
    ----------
    seed : TrailSeed
        Whether to use a fixed-pct or ATR-based seed.
    pct : float
        Seed percentage (used when ``seed=TrailSeed.FIXED_PCT``).
    multiplier : float
        ATR multiplier (used when ``seed=TrailSeed.ATR``).

    State
    -----
    After calling :meth:`initialise` the stop level is held in
    ``self.current_stop`` and updated via :meth:`update`.
    """

    def __init__(
        self,
        seed: TrailSeed  = TrailSeed.ATR,
        pct: float       = 0.02,
        multiplier: float = 2.0,
    ) -> None:
        if not (0 < pct < 1):
            raise ValueError(f"pct must be in (0,1), got {pct}")
        if multiplier <= 0:
            raise ValueError(f"multiplier must be > 0, got {multiplier}")

        self.seed       = seed
        self.pct        = pct
        self.multiplier = multiplier

        # Mutable state — reset on each new trade
        self.current_stop:  Optional[float] = None
        self.entry_price:   Optional[float] = None
        self.peak_price:    Optional[float] = None
        self._seed_distance: Optional[float] = None

    # ── Trade lifecycle ───────────────────────────────────────────────────────
    def initialise(self, entry_price: float, atr: Optional[float] = None) -> float:
        """
        Seed the trailing stop at trade entry.

        Parameters
        ----------
        entry_price : float
            Price at which the position was entered.
        atr : float, optional
            Current ATR — required when ``seed=TrailSeed.ATR``.

        Returns
        -------
        float
            Initial stop price.

        Raises
        ------
        ValueError
            If ``seed=ATR`` but *atr* is ``None`` or ≤ 0.
        """
        if entry_price <= 0:
            raise ValueError(f"entry_price must be positive, got {entry_price}")

        self.entry_price = entry_price
        self.peak_price  = entry_price

        if self.seed == TrailSeed.ATR:
            if atr is None or atr <= 0:
                raise ValueError(
                    "TrailingStop seeded with ATR requires a positive atr value."
                )
            self._seed_distance = self.multiplier * atr
        else:
            self._seed_distance = entry_price * self.pct

        self.current_stop = entry_price - self._seed_distance

        logger.info(
            "TrailingStop INIT  entry=₹%.2f  stop=₹%.2f  "
            "seed_dist=₹%.2f  mode=%s",
            entry_price, self.current_stop,
            self._seed_distance, self.seed.name,
        )
        return self.current_stop

    def update(self, current_price: float, atr: Optional[float] = None) -> float:
        """
        Ratchet the trailing stop upward given the latest price (and ATR).

        The stop *never moves down*.  When ``seed=ATR`` and *atr* is
        provided, the seed distance is refreshed each bar so the stop
        adapts to changing volatility.

        Parameters
        ----------
        current_price : float
            Latest closing price.
        atr : float, optional
            Latest ATR — refreshes the seed distance when using ATR mode.

        Returns
        -------
        float
            Updated stop price.

        Raises
        ------
        RuntimeError
            If :meth:`initialise` has not been called first.
        """
        if self.current_stop is None or self._seed_distance is None:
            raise RuntimeError(
                "TrailingStop must be initialised before calling update(). "
                "Call .initialise(entry_price) first."
            )

        # Optionally refresh seed distance using latest ATR
        if self.seed == TrailSeed.ATR and atr is not None and atr > 0:
            self._seed_distance = self.multiplier * atr

        new_candidate = current_price - self._seed_distance
        if new_candidate > self.current_stop:
            old_stop = self.current_stop
            self.current_stop = round(new_candidate, 2)
            logger.debug(
                "TrailingStop RATCHET  price=₹%.2f  stop ₹%.2f → ₹%.2f",
                current_price, old_stop, self.current_stop,
            )

        if current_price > (self.peak_price or 0):
            self.peak_price = current_price

        return self.current_stop

    def is_hit(self, current_price: float) -> bool:
        """
        Return ``True`` if *current_price* has breached the stop level.

        Parameters
        ----------
        current_price : float
            Latest price to check.
        """
        if self.current_stop is None:
            return False
        hit = current_price <= self.current_stop
        if hit:
            logger.info(
                "TrailingStop HIT  price=₹%.2f ≤ stop=₹%.2f  "
                "| peak=₹%.2f  | drawdown=%.2f%%",
                current_price, self.current_stop,
                self.peak_price or 0,
                ((self.peak_price or current_price) - current_price)
                / (self.peak_price or current_price) * 100,
            )
        return hit

    def reset(self) -> None:
        """Clear all state — call before the next trade."""
        self.current_stop   = None
        self.entry_price    = None
        self.peak_price     = None
        self._seed_distance = None
        logger.debug("TrailingStop RESET")

    @property
    def profit_locked_pct(self) -> Optional[float]:
        """
        Percentage of peak gain that has been *locked in* by the stop.

        Returns ``None`` if the stop is not initialised.
        """
        if self.entry_price is None or self.current_stop is None:
            return None
        if self.peak_price == self.entry_price:
            return 0.0
        peak_gain = self.peak_price - self.entry_price          # type: ignore[operator]
        locked    = self.current_stop - self.entry_price
        return max(locked / peak_gain * 100, 0.0)


# ──────────────────────────────────────────────────────────────────────────────
# Facade — StopLossManager
# ──────────────────────────────────────────────────────────────────────────────
class StopLossManager:
    """
    High-level facade that strategies interact with.

    Chooses the right calculator based on *mode*, holds per-trade state,
    and exposes a unified API regardless of which stop type is active.

    Parameters
    ----------
    mode : StopMode
        Which stop algorithm to use.
    pct : float
        Fixed-percentage stop distance.  Default: ``0.02`` (2 %).
    atr_multiplier : float
        ATR distance multiplier.  Default: ``2.0``.
    trail_seed : TrailSeed
        Seed method for trailing stops.  Default: ``TrailSeed.ATR``.

    Examples
    --------
    Fixed-pct stop::

        mgr = StopLossManager(mode=StopMode.FIXED_PCT, pct=0.03)
        result = mgr.calculate(entry_price=3_400)

    ATR stop::

        mgr = StopLossManager(mode=StopMode.ATR, atr_multiplier=2.0)
        result = mgr.calculate(entry_price=3_400, atr=85.0)

    Trailing stop (bar-by-bar)::

        mgr = StopLossManager(mode=StopMode.TRAILING)
        mgr.initialise_trailing(entry_price=3_400, atr=85.0)
        mgr.update_trailing(current_price=3_500, atr=82.0)
        if mgr.is_trailing_hit(current_price=3_200):
            # exit trade
    """

    def __init__(
        self,
        mode: StopMode       = StopMode.ATR,
        pct: float           = 0.02,
        atr_multiplier: float = 2.0,
        trail_seed: TrailSeed = TrailSeed.ATR,
    ) -> None:
        self.mode            = mode
        self._pct_calc       = FixedPercentageStop(pct=pct)
        self._atr_calc       = ATRStop(multiplier=atr_multiplier)
        self._trailing       = TrailingStop(
            seed=trail_seed, pct=pct, multiplier=atr_multiplier
        )
        # Last computed result — useful for Phase 4 take-profit integration
        self.last_result: Optional[StopLossResult] = None

    # ── Stateless calculation ─────────────────────────────────────────────────
    def calculate(
        self,
        entry_price: float,
        atr: Optional[float] = None,
    ) -> StopLossResult:
        """
        Compute a stop-loss level for a new trade entry.

        For ``FIXED_PCT`` mode *atr* is ignored.
        For ``ATR`` or ``TRAILING`` mode *atr* must be provided.

        Parameters
        ----------
        entry_price : float
            Trade entry price.
        atr : float, optional
            Current ATR value.

        Returns
        -------
        StopLossResult
        """
        if self.mode == StopMode.FIXED_PCT:
            result = self._pct_calc.calculate(entry_price)
        elif self.mode == StopMode.ATR:
            if atr is None:
                raise ValueError("ATR mode requires an 'atr' argument.")
            result = self._atr_calc.calculate(entry_price, atr)
        else:
            # TRAILING — seed with ATR or pct and return initial level
            if atr is None and self._trailing.seed == TrailSeed.ATR:
                raise ValueError("Trailing ATR mode requires an 'atr' argument.")
            stop_price = self._trailing.initialise(entry_price, atr)
            distance   = entry_price - stop_price
            result = StopLossResult(
                stop_price   = round(stop_price, 2),
                distance     = round(distance, 2),
                distance_pct = (distance / entry_price) * 100,
                mode         = StopMode.TRAILING,
                entry_price  = entry_price,
                atr_used     = atr,
            )

        self.last_result = result
        return result

    # ── Trailing stop state management ───────────────────────────────────────
    def initialise_trailing(
        self, entry_price: float, atr: Optional[float] = None
    ) -> float:
        """Delegate to :meth:`TrailingStop.initialise`."""
        return self._trailing.initialise(entry_price, atr)

    def update_trailing(
        self, current_price: float, atr: Optional[float] = None
    ) -> float:
        """Delegate to :meth:`TrailingStop.update`."""
        return self._trailing.update(current_price, atr)

    def is_trailing_hit(self, current_price: float) -> bool:
        """Delegate to :meth:`TrailingStop.is_hit`."""
        return self._trailing.is_hit(current_price)

    def reset(self) -> None:
        """Reset trailing stop state between trades."""
        self._trailing.reset()
        self.last_result = None

    @property
    def trailing_stop_price(self) -> Optional[float]:
        """Current trailing stop level (``None`` if not initialised)."""
        return self._trailing.current_stop

    @property
    def profit_locked_pct(self) -> Optional[float]:
        """Percentage of peak gain locked in by the trailing stop."""
        return self._trailing.profit_locked_pct


# ──────────────────────────────────────────────────────────────────────────────
# Vectorised helpers for pandas-based analysis
# ──────────────────────────────────────────────────────────────────────────────
def add_fixed_stop_column(
    df: pd.DataFrame,
    price_col: str = "close",
    pct: float     = 0.02,
    col_name: str  = "stop_fixed",
) -> pd.DataFrame:
    """
    Add a fixed-percentage stop-loss column to a DataFrame.

    Useful for offline signal analysis outside Backtrader.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain *price_col*.
    price_col : str
        Column to treat as entry price.
    pct : float
        Stop percentage below price.
    col_name : str
        Output column name.

    Returns
    -------
    pd.DataFrame
        Copy with *col_name* added.
    """
    if price_col not in df.columns:
        raise KeyError(f"Column '{price_col}' not found in DataFrame.")
    df = df.copy()
    df[col_name] = (df[price_col] * (1.0 - pct)).round(2)
    logger.info(
        "Added '%s' column: %.0f%% fixed stop below '%s'.",
        col_name, pct * 100, price_col,
    )
    return df


def add_atr_stop_column(
    df: pd.DataFrame,
    price_col: str  = "close",
    atr_col: str    = "atr_14",
    multiplier: float = 2.0,
    col_name: str   = "stop_atr",
) -> pd.DataFrame:
    """
    Add an ATR-based stop-loss column to a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain *price_col* and *atr_col*.
    price_col : str
        Entry price column.
    atr_col : str
        ATR column (produced by ``src/indicators/atr.py``).
    multiplier : float
        ATR multiplier.
    col_name : str
        Output column name.

    Returns
    -------
    pd.DataFrame
        Copy with *col_name* added.
    """
    for col in (price_col, atr_col):
        if col not in df.columns:
            raise KeyError(
                f"Column '{col}' not found. "
                f"Run add_atr() from src/indicators/atr.py first."
            )
    df = df.copy()
    df[col_name] = (df[price_col] - multiplier * df[atr_col]).round(2)
    logger.info(
        "Added '%s' column: %.1f × ATR stop below '%s'.",
        col_name, multiplier, price_col,
    )
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Script entry-point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("  Stop-Loss Engine — Demo")
    print("=" * 65)

    PROCESSED_DIR = os.path.join("data", "processed")
    STOCKS = ["TCS", "RELIANCE", "INFOSYS"]

    # ── 1. Build indicator DataFrame with ATR ─────────────────────────────────
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)
    ))))
    from src.indicators.atr import add_atr

    for stock in STOCKS:
        path = os.path.join(PROCESSED_DIR, f"{stock}_processed.csv")
        if not os.path.exists(path):
            print(f"  ✗  {stock}: processed CSV not found — run preprocess.py")
            continue

        df = pd.read_csv(path, parse_dates=["date"])
        df = add_atr(df, period=14, include_tr=True)

        last      = df.iloc[-1]
        entry     = last["close"]
        atr_val   = last["atr_14"]

        print(f"\n{'─' * 55}")
        print(f"  {stock}  |  close=₹{entry:,.2f}  |  ATR(14)=₹{atr_val:.2f}")
        print(f"{'─' * 55}")

        # ── Fixed pct stop ────────────────────────────────────────────────────
        for pct in (0.01, 0.02, 0.03):
            mgr = StopLossManager(mode=StopMode.FIXED_PCT, pct=pct)
            r   = mgr.calculate(entry_price=entry)
            print(
                f"  Fixed {pct*100:.0f}%  stop=₹{r.stop_price:>10,.2f}  "
                f"dist=₹{r.distance:>8.2f}  ({r.distance_pct:.2f}%)"
            )

        # ── ATR stop ──────────────────────────────────────────────────────────
        for mult in (1.5, 2.0, 3.0):
            mgr = StopLossManager(mode=StopMode.ATR, atr_multiplier=mult)
            r   = mgr.calculate(entry_price=entry, atr=atr_val)
            print(
                f"  ATR × {mult:.1f}    stop=₹{r.stop_price:>10,.2f}  "
                f"dist=₹{r.distance:>8.2f}  ({r.distance_pct:.2f}%)"
            )

        # ── Trailing stop simulation ───────────────────────────────────────────
        print(f"\n  Trailing Stop simulation (ATR × 2.0, 10-bar walk):")
        trail_mgr = StopLossManager(mode=StopMode.TRAILING, atr_multiplier=2.0)
        trail_mgr.initialise_trailing(entry_price=entry, atr=atr_val)
        sim_prices = df["close"].iloc[-11:-1].values
        for price in sim_prices:
            new_stop = trail_mgr.update_trailing(current_price=price, atr=atr_val)
            hit = "HIT ✗" if trail_mgr.is_trailing_hit(price) else "OK  ✓"
            print(
                f"    price=₹{price:>9,.2f}  "
                f"trail_stop=₹{new_stop:>9,.2f}  "
                f"locked={trail_mgr.profit_locked_pct or 0:.1f}%  {hit}"
            )

    # ── 2. Vectorised column demo ─────────────────────────────────────────────
    print("\n── Vectorised stop columns on TCS (last 5 rows) ─────────────────")
    df_tcs = pd.read_csv(
        os.path.join(PROCESSED_DIR, "TCS_processed.csv"), parse_dates=["date"]
    )
    df_tcs = add_atr(df_tcs, period=14)
    df_tcs = add_fixed_stop_column(df_tcs, pct=0.02)
    df_tcs = add_atr_stop_column(df_tcs, multiplier=2.0)
    print(df_tcs[["date", "close", "atr_14", "stop_fixed", "stop_atr"]].tail(5).to_string(index=False))
    print()
