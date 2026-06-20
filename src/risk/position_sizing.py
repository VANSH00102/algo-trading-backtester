"""
position_sizing.py
==================
Position sizing engine for the Algorithmic Trading Strategy Backtester.

Why position sizing is the most important risk control
------------------------------------------------------
Two traders can run the exact same strategy and have opposite outcomes
purely because of how many shares they buy per trade.  Over-sizing turns
a 5-loss streak into a portfolio wipeout; under-sizing leaves profitable
edges unexploited.

Four sizing models
------------------
1. **Fixed Capital** — invest a fixed rupee amount or fixed percentage
   of total capital regardless of stop distance.
   Simple and predictable but ignores how far the stop is.

2. **Risk-Based (Van Tharp method)** — the *industry standard*:
   ``size = (capital × risk_pct) / stop_distance``
   Ties every trade's rupee risk to the same fraction of capital so
   losing trades always hurt equally, regardless of price level.

3. **ATR Volatility** — risk-based but uses ATR × multiplier as the
   stop distance proxy, even when you have not computed an explicit stop.
   Adapts position size to current market volatility automatically.

4. **Fractional Kelly** — position size that maximises log-growth of
   capital given an edge and odds.  Half-Kelly is common in practice to
   reduce variance.  *Only use with measured edge from backtests.*

Architecture
------------
* :class:`SizingResult` — immutable value object.
* :class:`FixedCapitalSizer` — allocates a fixed amount/fraction.
* :class:`RiskBasedSizer` — risk-pct ÷ stop-distance (primary model).
* :class:`ATRVolatilitySizer` — risk-pct ÷ (ATR × mult) sizing.
* :class:`KellySizer` — fractional Kelly criterion.
* :class:`PositionSizingManager` — facade coordinating all sizers.
* :func:`build_trade_setup` — one-call function that produces the
  complete trade plan (entry, stop, target, size) from a price + ATR.

Integration with Phase 3 & Phase 4
------------------------------------
::

    from src.risk.position_sizing import PositionSizingManager, SizingModel
    from src.risk.stop_loss       import StopLossManager, StopMode
    from src.risk.take_profit     import TakeProfitManager

    # Inside a Backtrader strategy next():
    ps = PositionSizingManager(model=SizingModel.RISK_BASED, risk_pct=0.01)
    result = ps.calculate(
        capital=self.broker.get_cash(),
        entry_price=self.data.close[0],
        stop_distance=self.stop_mgr.last_result.distance,
    )
    self.buy(size=result.shares)

Usage
-----
::

    python src/risk/position_sizing.py
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Optional

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
# Constants (Indian equity market context)
# ──────────────────────────────────────────────────────────────────────────────
NSE_MIN_LOT:  int   = 1       # NSE cash equities trade in lots of 1
MAX_POSITION_PCT: float = 0.25   # Never put more than 25% of capital in one stock


# ──────────────────────────────────────────────────────────────────────────────
# Enums & value objects
# ──────────────────────────────────────────────────────────────────────────────
class SizingModel(Enum):
    """Supported position sizing models."""
    FIXED_CAPITAL     = auto()   # Fixed rupee or pct-of-capital allocation
    RISK_BASED        = auto()   # Risk X% of capital per trade
    ATR_VOLATILITY    = auto()   # ATR-derived stop distance
    KELLY             = auto()   # Fractional Kelly criterion


@dataclass(frozen=True)
class SizingResult:
    """
    Immutable result of a position sizing calculation.

    Attributes
    ----------
    shares : int
        Number of whole shares to buy.
    capital_deployed : float
        ``shares × entry_price`` in INR.
    capital_deployed_pct : float
        Capital deployed as % of total available capital.
    risk_amount : float
        Maximum rupee loss if stop is hit (``shares × stop_distance``).
    risk_pct_of_capital : float
        ``risk_amount / capital`` as a percentage.
    model : SizingModel
        Which model produced this result.
    entry_price : float
        Price used in the calculation.
    stop_distance : float or None
        Stop distance used (``None`` for FIXED_CAPITAL).
    capped : bool
        ``True`` if the size was reduced due to the capital cap.
    """
    shares:               int
    capital_deployed:     float
    capital_deployed_pct: float
    risk_amount:          float
    risk_pct_of_capital:  float
    model:                SizingModel
    entry_price:          float
    stop_distance:        Optional[float] = field(default=None)
    capped:               bool            = field(default=False)

    def __str__(self) -> str:
        cap_str = "  [CAPPED]" if self.capped else ""
        return (
            f"Sizing [{self.model.name}]  "
            f"shares={self.shares}  "
            f"entry=₹{self.entry_price:,.2f}  "
            f"deployed=₹{self.capital_deployed:,.2f} ({self.capital_deployed_pct:.1f}%)  "
            f"risk=₹{self.risk_amount:,.2f} ({self.risk_pct_of_capital:.2f}%)"
            f"{cap_str}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Internal helper
# ──────────────────────────────────────────────────────────────────────────────
def _apply_cap(
    raw_shares: float,
    capital: float,
    entry_price: float,
    max_pct: float = MAX_POSITION_PCT,
) -> tuple[int, bool]:
    """
    Clamp *raw_shares* so the deployed capital never exceeds *max_pct*
    of *capital*, and round down to whole shares.

    Parameters
    ----------
    raw_shares : float
        Uncapped fractional share count.
    capital : float
        Available capital.
    entry_price : float
        Share price.
    max_pct : float
        Maximum fraction of capital to deploy.

    Returns
    -------
    (int, bool)
        ``(clamped_shares, was_capped)``
    """
    max_shares = int((capital * max_pct) / entry_price)
    final      = max(int(raw_shares), 0)
    capped     = final > max_shares
    return min(final, max_shares), capped


# ──────────────────────────────────────────────────────────────────────────────
# Sizer 1 — Fixed Capital
# ──────────────────────────────────────────────────────────────────────────────
class FixedCapitalSizer:
    """
    Allocate a fixed rupee amount or percentage of capital per trade.

    Ignores stop distance — simplest possible model.  Useful as a baseline
    or when the stop cannot be computed in advance.

    Parameters
    ----------
    allocation : float
        Either a fixed INR amount (if ``use_pct=False``) or a fraction
        of capital (if ``use_pct=True``).  Default: ``0.10`` (10 %).
    use_pct : bool
        If ``True``, *allocation* is treated as a fraction of *capital*.
        If ``False``, *allocation* is a fixed INR amount.

    Examples
    --------
    >>> sizer = FixedCapitalSizer(allocation=0.10, use_pct=True)
    >>> result = sizer.calculate(capital=500_000, entry_price=3_400)
    >>> result.shares   # 50000 × 0.10 / 3400 = 14
    14
    """

    def __init__(self, allocation: float = 0.10, use_pct: bool = True) -> None:
        if allocation <= 0:
            raise ValueError(f"allocation must be > 0, got {allocation}")
        self.allocation = allocation
        self.use_pct    = use_pct

    def calculate(self, capital: float, entry_price: float) -> SizingResult:
        """
        Parameters
        ----------
        capital : float
            Available cash.
        entry_price : float
            Share price.

        Returns
        -------
        SizingResult
        """
        if capital <= 0 or entry_price <= 0:
            raise ValueError("capital and entry_price must be positive.")

        budget    = capital * self.allocation if self.use_pct else self.allocation
        budget    = min(budget, capital)           # can't spend more than we have
        shares, capped = _apply_cap(budget / entry_price, capital, entry_price)
        deployed  = shares * entry_price

        result = SizingResult(
            shares               = shares,
            capital_deployed     = round(deployed, 2),
            capital_deployed_pct = deployed / capital * 100,
            risk_amount          = 0.0,            # unknown without a stop
            risk_pct_of_capital  = 0.0,
            model                = SizingModel.FIXED_CAPITAL,
            entry_price          = entry_price,
            capped               = capped,
        )
        logger.info("Sizing  %s", result)
        return result


# ──────────────────────────────────────────────────────────────────────────────
# Sizer 2 — Risk-Based (primary model)
# ──────────────────────────────────────────────────────────────────────────────
class RiskBasedSizer:
    """
    Industry-standard position sizing: risk a fixed percentage of capital
    per trade and let the stop distance determine share count.

    Formula::

        risk_amount   = capital × risk_pct
        shares        = floor(risk_amount / stop_distance)

    This guarantees that if the stop is hit, you lose exactly ``risk_pct``
    of your capital — regardless of the stock's price or volatility.

    Parameters
    ----------
    risk_pct : float
        Fraction of capital to risk per trade.  Common values:
        * 0.005 — 0.5 % (very conservative)
        * 0.010 — 1.0 % (standard, default)
        * 0.020 — 2.0 % (aggressive)

    Examples
    --------
    >>> sizer = RiskBasedSizer(risk_pct=0.01)
    >>> result = sizer.calculate(
    ...     capital=500_000, entry_price=3_400, stop_distance=170
    ... )
    >>> result.shares   # (500000 × 0.01) / 170 = 29 shares
    29
    """

    def __init__(self, risk_pct: float = 0.01) -> None:
        if not (0 < risk_pct < 1):
            raise ValueError(f"risk_pct must be in (0, 1), got {risk_pct}")
        self.risk_pct = risk_pct

    def calculate(
        self,
        capital: float,
        entry_price: float,
        stop_distance: float,
    ) -> SizingResult:
        """
        Parameters
        ----------
        capital : float
            Total available cash.
        entry_price : float
            Trade entry price.
        stop_distance : float
            Distance from entry to stop in INR (must be positive).

        Returns
        -------
        SizingResult
        """
        if capital <= 0:
            raise ValueError(f"capital must be positive, got {capital}")
        if entry_price <= 0:
            raise ValueError(f"entry_price must be positive, got {entry_price}")
        if stop_distance <= 0:
            raise ValueError(
                f"stop_distance must be positive, got {stop_distance}. "
                "Ensure the stop is below the entry price."
            )

        risk_amount      = capital * self.risk_pct
        raw_shares       = risk_amount / stop_distance
        shares, capped   = _apply_cap(raw_shares, capital, entry_price)
        deployed         = shares * entry_price
        actual_risk      = shares * stop_distance

        result = SizingResult(
            shares               = shares,
            capital_deployed     = round(deployed, 2),
            capital_deployed_pct = deployed / capital * 100,
            risk_amount          = round(actual_risk, 2),
            risk_pct_of_capital  = actual_risk / capital * 100,
            model                = SizingModel.RISK_BASED,
            entry_price          = entry_price,
            stop_distance        = stop_distance,
            capped               = capped,
        )
        logger.info("Sizing  %s", result)
        return result


# ──────────────────────────────────────────────────────────────────────────────
# Sizer 3 — ATR Volatility
# ──────────────────────────────────────────────────────────────────────────────
class ATRVolatilitySizer:
    """
    Risk-based sizing where the stop distance is derived from ATR.

    Combines two ideas: (1) risk a fixed % of capital and (2) use ATR
    to define what "1 unit of risk" means for *this stock right now*.

    Formula::

        stop_distance = atr_multiplier × ATR
        shares        = floor((capital × risk_pct) / stop_distance)

    Parameters
    ----------
    risk_pct : float
        Capital fraction to risk (default ``0.01``).
    atr_multiplier : float
        ATR distance multiplier (default ``2.0``).
    """

    def __init__(
        self, risk_pct: float = 0.01, atr_multiplier: float = 2.0
    ) -> None:
        if not (0 < risk_pct < 1):
            raise ValueError(f"risk_pct must be in (0,1), got {risk_pct}")
        if atr_multiplier <= 0:
            raise ValueError(f"atr_multiplier must be > 0, got {atr_multiplier}")
        self._core     = RiskBasedSizer(risk_pct=risk_pct)
        self.risk_pct  = risk_pct
        self.atr_mult  = atr_multiplier

    def calculate(
        self,
        capital: float,
        entry_price: float,
        atr: float,
    ) -> SizingResult:
        """
        Parameters
        ----------
        capital : float
            Available cash.
        entry_price : float
            Trade entry price.
        atr : float
            Current ATR value (must be positive).

        Returns
        -------
        SizingResult
            With ``model=SizingModel.ATR_VOLATILITY``.
        """
        if atr <= 0:
            raise ValueError(f"atr must be positive, got {atr}")

        stop_distance = self.atr_mult * atr
        result        = self._core.calculate(capital, entry_price, stop_distance)

        # Override the model tag
        return SizingResult(
            shares               = result.shares,
            capital_deployed     = result.capital_deployed,
            capital_deployed_pct = result.capital_deployed_pct,
            risk_amount          = result.risk_amount,
            risk_pct_of_capital  = result.risk_pct_of_capital,
            model                = SizingModel.ATR_VOLATILITY,
            entry_price          = entry_price,
            stop_distance        = stop_distance,
            capped               = result.capped,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Sizer 4 — Fractional Kelly
# ──────────────────────────────────────────────────────────────────────────────
class KellySizer:
    """
    Fractional Kelly Criterion position sizing.

    The Kelly fraction maximises the geometric growth rate of capital:

    ::

        full_kelly = (win_rate × avg_rr - (1 - win_rate)) / avg_rr
        fractional_kelly = full_kelly × kelly_fraction

    **Use only with measured edge from live backtests.**  The Kelly
    formula amplifies errors in estimated win_rate and avg_rr, so
    ``kelly_fraction=0.5`` (Half Kelly) is the practical standard.

    Parameters
    ----------
    win_rate : float
        Estimated win rate from backtests (0 < x < 1).
    avg_rr : float
        Average winning trade's R-multiple (e.g. 2.0 = 2R winners).
    kelly_fraction : float
        Fraction of full Kelly to use.  Default: ``0.5`` (Half Kelly).
    """

    def __init__(
        self,
        win_rate: float       = 0.45,
        avg_rr: float         = 2.0,
        kelly_fraction: float = 0.5,
    ) -> None:
        for name, val in [("win_rate", win_rate), ("avg_rr", avg_rr)]:
            if val <= 0:
                raise ValueError(f"{name} must be > 0, got {val}")
        if not (0 < kelly_fraction <= 1):
            raise ValueError(f"kelly_fraction must be in (0,1], got {kelly_fraction}")

        self.win_rate       = win_rate
        self.avg_rr         = avg_rr
        self.kelly_fraction = kelly_fraction

        # Pre-compute the full Kelly pct
        self.full_kelly_pct = max(
            (win_rate * avg_rr - (1.0 - win_rate)) / avg_rr, 0.0
        )
        self.effective_pct  = self.full_kelly_pct * kelly_fraction

        logger.info(
            "KellySizer  win_rate=%.1f%%  avg_rr=%.1fR  "
            "full_kelly=%.2f%%  effective=%.2f%%",
            win_rate * 100, avg_rr,
            self.full_kelly_pct * 100, self.effective_pct * 100,
        )

    def calculate(
        self,
        capital: float,
        entry_price: float,
        stop_distance: Optional[float] = None,
    ) -> SizingResult:
        """
        Parameters
        ----------
        capital : float
            Available cash.
        entry_price : float
            Share price.
        stop_distance : float, optional
            Used for the ``risk_amount`` field only; does not affect
            share count in pure Kelly mode.

        Returns
        -------
        SizingResult
        """
        if capital <= 0 or entry_price <= 0:
            raise ValueError("capital and entry_price must be positive.")
        if self.effective_pct <= 0:
            logger.warning(
                "Kelly fraction is zero or negative (edge may be negative) "
                "— returning 0 shares."
            )
            return SizingResult(
                shares=0, capital_deployed=0, capital_deployed_pct=0,
                risk_amount=0, risk_pct_of_capital=0,
                model=SizingModel.KELLY, entry_price=entry_price,
            )

        budget         = capital * self.effective_pct
        shares, capped = _apply_cap(budget / entry_price, capital, entry_price)
        deployed       = shares * entry_price
        risk_amt       = (shares * stop_distance) if stop_distance else 0.0

        result = SizingResult(
            shares               = shares,
            capital_deployed     = round(deployed, 2),
            capital_deployed_pct = deployed / capital * 100,
            risk_amount          = round(risk_amt, 2),
            risk_pct_of_capital  = risk_amt / capital * 100 if stop_distance else 0.0,
            model                = SizingModel.KELLY,
            entry_price          = entry_price,
            stop_distance        = stop_distance,
            capped               = capped,
        )
        logger.info("Sizing  %s", result)
        return result


# ──────────────────────────────────────────────────────────────────────────────
# Complete trade setup builder
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class TradeSetup:
    """
    Complete trade plan for one entry signal.

    Bundles entry, stop-loss, take-profit, and position size into one
    object that can be passed directly to a Backtrader order or logged
    for a trade journal.

    Attributes
    ----------
    stock : str
        Ticker / display name.
    entry_price : float
        Planned entry price.
    stop_price : float
        Stop-loss price.
    target_price : float
        Take-profit price.
    shares : int
        Number of shares to buy.
    capital_deployed : float
        Total value of the position (shares × entry_price).
    max_risk_inr : float
        Maximum loss in INR if stop is hit.
    potential_gain_inr : float
        Potential gain in INR if target is hit.
    r_multiple : float
        Risk-reward ratio of the trade.
    stop_distance : float
        Distance from entry to stop.
    target_distance : float
        Distance from entry to target.
    """
    stock:              str
    entry_price:        float
    stop_price:         float
    target_price:       float
    shares:             int
    capital_deployed:   float
    max_risk_inr:       float
    potential_gain_inr: float
    r_multiple:         float
    stop_distance:      float
    target_distance:    float

    def summary(self) -> str:
        """Return a formatted multi-line trade plan summary."""
        lines = [
            f"  ┌─ Trade Setup: {self.stock} ───────────────────────────────",
            f"  │  Entry     : ₹{self.entry_price:>10,.2f}",
            f"  │  Stop      : ₹{self.stop_price:>10,.2f}  "
            f"(dist ₹{self.stop_distance:.2f})",
            f"  │  Target    : ₹{self.target_price:>10,.2f}  "
            f"(dist ₹{self.target_distance:.2f})",
            f"  │  R:R       : {self.r_multiple:.2f}R",
            f"  │  Shares    : {self.shares}",
            f"  │  Capital   : ₹{self.capital_deployed:>10,.2f}",
            f"  │  Max risk  : ₹{self.max_risk_inr:>10,.2f}",
            f"  │  Pot. gain : ₹{self.potential_gain_inr:>10,.2f}",
            f"  └─────────────────────────────────────────────────────────",
        ]
        return "\n".join(lines)


def build_trade_setup(
    stock: str,
    capital: float,
    entry_price: float,
    atr: float,
    risk_pct: float       = 0.01,
    atr_stop_mult: float  = 2.0,
    r_multiple: float     = 2.0,
) -> TradeSetup:
    """
    One-call function: compute stop, target, and size for a new signal.

    Uses ATR-based stop-loss, R-multiple take-profit, and risk-based
    position sizing.  This is the primary integration point for Phase 3
    strategies and the Phase 5 backtesting engine.

    Parameters
    ----------
    stock : str
        Display name for logging / journaling.
    capital : float
        Available cash (from ``broker.get_cash()`` in Backtrader).
    entry_price : float
        Entry price (``self.data.close[0]`` in Backtrader).
    atr : float
        Current ATR value (from ``bt.indicators.ATR``).
    risk_pct : float
        Capital fraction to risk.  Default: 1 %.
    atr_stop_mult : float
        ATR multiplier for stop distance.  Default: 2.0.
    r_multiple : float
        Take-profit R-multiple.  Default: 2.0.

    Returns
    -------
    TradeSetup

    Examples
    --------
    Inside a Backtrader ``next()``::

        setup = build_trade_setup(
            stock       = "TCS",
            capital     = self.broker.get_cash(),
            entry_price = self.data.close[0],
            atr         = self.atr[0],
        )
        self.buy(size=setup.shares)
    """
    # ── Stop-loss ─────────────────────────────────────────────────────────────
    from src.risk.stop_loss  import StopLossManager, StopMode  # lazy import
    stop_mgr    = StopLossManager(mode=StopMode.ATR, atr_multiplier=atr_stop_mult)
    stop_result = stop_mgr.calculate(entry_price=entry_price, atr=atr)

    # ── Take-profit ───────────────────────────────────────────────────────────
    from src.risk.take_profit import TakeProfitManager, TPMode  # lazy import
    tp_mgr    = TakeProfitManager(mode=TPMode.RISK_REWARD, r_multiple=r_multiple)
    tp_result = tp_mgr.calculate_from_stop(stop_result)

    # ── Position size ─────────────────────────────────────────────────────────
    sizer        = RiskBasedSizer(risk_pct=risk_pct)
    size_result  = sizer.calculate(
        capital       = capital,
        entry_price   = entry_price,
        stop_distance = stop_result.distance,
    )

    setup = TradeSetup(
        stock              = stock,
        entry_price        = entry_price,
        stop_price         = stop_result.stop_price,
        target_price       = tp_result.target_price,
        shares             = size_result.shares,
        capital_deployed   = size_result.capital_deployed,
        max_risk_inr       = size_result.risk_amount,
        potential_gain_inr = round(size_result.shares * tp_result.distance, 2),
        r_multiple         = r_multiple,
        stop_distance      = stop_result.distance,
        target_distance    = tp_result.distance,
    )

    logger.info(
        "TradeSetup  %s  entry=₹%.2f  stop=₹%.2f  target=₹%.2f  "
        "shares=%d  risk=₹%.2f",
        stock, entry_price,
        stop_result.stop_price, tp_result.target_price,
        size_result.shares, size_result.risk_amount,
    )
    return setup


# ──────────────────────────────────────────────────────────────────────────────
# Facade — PositionSizingManager
# ──────────────────────────────────────────────────────────────────────────────
class PositionSizingManager:
    """
    High-level facade that strategies use for position sizing decisions.

    Parameters
    ----------
    model : SizingModel
        Which sizer to use.  Default: ``RISK_BASED``.
    risk_pct : float
        Capital fraction to risk (RISK_BASED and ATR_VOLATILITY).
    allocation_pct : float
        Capital fraction to allocate (FIXED_CAPITAL).
    atr_multiplier : float
        ATR distance multiplier (ATR_VOLATILITY).
    kelly_win_rate : float
        Historical win rate for KELLY mode.
    kelly_avg_rr : float
        Average R-multiple for KELLY mode.
    kelly_fraction : float
        Fraction of full Kelly to use.

    Examples
    --------
    ::

        mgr = PositionSizingManager(model=SizingModel.RISK_BASED, risk_pct=0.01)
        result = mgr.calculate(
            capital=500_000, entry_price=3_400, stop_distance=170
        )
        # result.shares → 29
    """

    def __init__(
        self,
        model: SizingModel      = SizingModel.RISK_BASED,
        risk_pct: float         = 0.01,
        allocation_pct: float   = 0.10,
        atr_multiplier: float   = 2.0,
        kelly_win_rate: float   = 0.45,
        kelly_avg_rr: float     = 2.0,
        kelly_fraction: float   = 0.5,
    ) -> None:
        self.model = model
        self._fixed   = FixedCapitalSizer(allocation=allocation_pct, use_pct=True)
        self._risk    = RiskBasedSizer(risk_pct=risk_pct)
        self._atr_vol = ATRVolatilitySizer(
            risk_pct=risk_pct, atr_multiplier=atr_multiplier
        )
        self._kelly   = KellySizer(
            win_rate=kelly_win_rate,
            avg_rr=kelly_avg_rr,
            kelly_fraction=kelly_fraction,
        )
        self.last_result: Optional[SizingResult] = None

    def calculate(
        self,
        capital: float,
        entry_price: float,
        stop_distance: Optional[float] = None,
        atr: Optional[float]           = None,
    ) -> SizingResult:
        """
        Calculate position size using the configured model.

        Parameters
        ----------
        capital : float
            Available cash.
        entry_price : float
            Share price.
        stop_distance : float, optional
            Stop-loss distance (required for RISK_BASED and KELLY).
        atr : float, optional
            Current ATR (required for ATR_VOLATILITY).

        Returns
        -------
        SizingResult
        """
        if self.model == SizingModel.FIXED_CAPITAL:
            result = self._fixed.calculate(capital, entry_price)

        elif self.model == SizingModel.RISK_BASED:
            if stop_distance is None:
                raise ValueError("RISK_BASED model requires 'stop_distance'.")
            result = self._risk.calculate(capital, entry_price, stop_distance)

        elif self.model == SizingModel.ATR_VOLATILITY:
            if atr is None:
                raise ValueError("ATR_VOLATILITY model requires 'atr'.")
            result = self._atr_vol.calculate(capital, entry_price, atr)

        elif self.model == SizingModel.KELLY:
            result = self._kelly.calculate(capital, entry_price, stop_distance)

        else:
            raise ValueError(f"Unknown SizingModel: {self.model}")

        self.last_result = result
        return result

    def compare_models(
        self,
        capital: float,
        entry_price: float,
        stop_distance: float,
        atr: float,
    ) -> Dict[str, SizingResult]:
        """
        Run all four models and return results side-by-side for comparison.

        Parameters
        ----------
        capital : float
        entry_price : float
        stop_distance : float
        atr : float

        Returns
        -------
        dict mapping model name → SizingResult
        """
        return {
            "FIXED_CAPITAL":  self._fixed.calculate(capital, entry_price),
            "RISK_BASED":     self._risk.calculate(capital, entry_price, stop_distance),
            "ATR_VOLATILITY": self._atr_vol.calculate(capital, entry_price, atr),
            "KELLY":          self._kelly.calculate(capital, entry_price, stop_distance),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Script entry-point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("  Position Sizing Engine — Demo")
    print("=" * 65)

    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)
    ))))
    from src.indicators.atr import add_atr

    PROCESSED_DIR = os.path.join("data", "processed")
    STOCKS        = ["TCS", "RELIANCE", "INFOSYS"]
    CAPITAL       = 1_000_000.0   # ₹10,00,000

    print(f"\n  Capital: ₹{CAPITAL:,.0f}  |  Risk per trade: 1%")

    # ── Per-stock sizing demo ─────────────────────────────────────────────────
    for stock in STOCKS:
        path = os.path.join(PROCESSED_DIR, f"{stock}_processed.csv")
        if not os.path.exists(path):
            print(f"  ✗  {stock}: processed CSV not found")
            continue

        df  = pd.read_csv(path, parse_dates=["date"])
        df  = add_atr(df, period=14)
        row = df.iloc[-1]

        entry   = row["close"]
        atr_val = row["atr_14"]
        stop_d  = 2.0 * atr_val

        print(f"\n{'─' * 60}")
        print(
            f"  {stock:<10} close=₹{entry:,.2f}  "
            f"ATR=₹{atr_val:.2f}  2×ATR stop_dist=₹{stop_d:.2f}"
        )
        print(f"{'─' * 60}")

        mgr = PositionSizingManager(
            model=SizingModel.RISK_BASED, risk_pct=0.01,
            kelly_win_rate=0.45, kelly_avg_rr=2.0,
        )
        comparison = mgr.compare_models(CAPITAL, entry, stop_d, atr_val)

        hdr = (
            f"  {'Model':<18} {'Shares':>7} {'Deployed ₹':>13} "
            f"{'Deploy%':>8} {'Risk ₹':>10} {'Risk%':>7}"
        )
        print(hdr)
        print("  " + "─" * (len(hdr) - 2))
        for model_name, res in comparison.items():
            cap_str = " *" if res.capped else ""
            print(
                f"  {model_name:<18} {res.shares:>7}  "
                f"₹{res.capital_deployed:>11,.2f}  "
                f"{res.capital_deployed_pct:>7.1f}%  "
                f"₹{res.risk_amount:>9,.2f}  "
                f"{res.risk_pct_of_capital:>6.2f}%{cap_str}"
            )

    # ── Risk sensitivity: how share count changes with risk% ─────────────────
    print(f"\n{'─' * 60}")
    print("  TCS — Shares bought at various risk percentages")
    print(f"{'─' * 60}")

    df_tcs  = pd.read_csv(
        os.path.join(PROCESSED_DIR, "TCS_processed.csv"), parse_dates=["date"]
    )
    df_tcs  = add_atr(df_tcs, period=14)
    tcs_row = df_tcs.iloc[-1]
    tcs_e   = tcs_row["close"]
    tcs_sd  = 2.0 * tcs_row["atr_14"]

    hdr2 = f"  {'Risk%':>6} {'Shares':>8} {'Risk ₹':>10} {'Deploy ₹':>12}"
    print(hdr2)
    print("  " + "─" * (len(hdr2) - 2))
    for rpct in (0.005, 0.01, 0.015, 0.02, 0.025, 0.03):
        s = RiskBasedSizer(risk_pct=rpct)
        r = s.calculate(CAPITAL, tcs_e, tcs_sd)
        print(
            f"  {rpct*100:>5.1f}%  {r.shares:>8}  "
            f"₹{r.risk_amount:>9,.2f}  ₹{r.capital_deployed:>11,.2f}"
        )

    # ── Full trade setup example ───────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  Full Trade Setup — TCS  (all 3 risk modules in one call)")
    print(f"{'─' * 60}")
    setup = build_trade_setup(
        stock       = "TCS",
        capital     = CAPITAL,
        entry_price = tcs_e,
        atr         = tcs_row["atr_14"],
        risk_pct    = 0.01,
        r_multiple  = 2.0,
    )
    print(setup.summary())
