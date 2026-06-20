"""
allocation.py
=============
Portfolio capital allocation engine for the Algorithmic Trading
Strategy Backtester.

Four allocation schemes
-----------------------
1. **Equal Weight** — 1/N per stock.  Simple, diversified baseline.

2. **Custom Weight** — user-specified fractions that sum to 1.0.
   Used for conviction-based or thematic allocations.

3. **Risk Parity** — inverse-volatility weighting.
   Stocks with lower historical volatility receive larger allocations
   so each position contributes the same *risk* to the portfolio,
   rather than the same *capital*.
   ``w_i = (1/σ_i) / Σ(1/σ_j)``

4. **Market Cap Proxy** — proportional to approximate free-float
   market capitalisation.  Mimics passive index exposure.
   Defaults use 2024 NSE approximate figures for TCS, Reliance, Infosys.

Architecture
------------
* :class:`AllocationScheme` — enum of available schemes.
* :class:`AllocationResult` — immutable snapshot of one allocation run.
* ``EqualWeightAllocator``, ``CustomWeightAllocator``,
  ``RiskParityAllocator``, ``MarketCapAllocator`` — strategy objects.
* :class:`PortfolioAllocator` — facade that selects the right allocator
  and exposes a single ``allocate()`` call.

Integration with Phase 5
------------------------
``AllocationResult`` feeds directly into:
* ``rebalance.py`` — target weights and initial share counts
* ``position_sizing.py`` — per-stock capital budgets

Usage
-----
::

    python src/portfolio/allocation.py
    from src.portfolio.allocation import PortfolioAllocator, AllocationScheme
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

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
#: Default universe for this project.
DEFAULT_STOCKS: List[str] = ["TCS", "RELIANCE", "INFOSYS"]

#: Approximate NSE free-float market caps (₹ crore, end-2024).
#: Source: NSE market cap data.
DEFAULT_MARKET_CAPS: Dict[str, float] = {
    "TCS":      1_424_000,
    "RELIANCE": 1_980_000,
    "INFOSYS":    620_000,
}

#: Processed CSV directory relative to project root.
PROCESSED_DIR: str = os.path.join("data", "processed")

#: Tolerance for weight-sum validation (float precision buffer).
WEIGHT_TOLERANCE: float = 1e-6

#: Lookback window for risk-parity volatility calculation (trading days).
DEFAULT_LOOKBACK_DAYS: int = 60


# ──────────────────────────────────────────────────────────────────────────────
# Enums & value objects
# ──────────────────────────────────────────────────────────────────────────────
class AllocationScheme(Enum):
    """Available portfolio allocation strategies."""
    EQUAL_WEIGHT = auto()
    CUSTOM       = auto()
    RISK_PARITY  = auto()
    MARKET_CAP   = auto()


@dataclass(frozen=True)
class AllocationResult:
    """
    Immutable snapshot of one portfolio allocation decision.

    Attributes
    ----------
    scheme : AllocationScheme
        Which allocation method was used.
    total_capital : float
        Total capital available for deployment (INR).
    weights : Dict[str, float]
        Target weights per stock (values sum to 1.0).
    allocated_capital : Dict[str, float]
        INR amount allocated to each stock.
    allocated_shares : Dict[str, int]
        Whole-share count per stock (``{}`` if prices not provided).
    uninvested_cash : float
        Cash left over after rounding to whole shares.
    volatilities : Dict[str, float]
        Annualised volatility used for risk-parity (``{}`` otherwise).
    """
    scheme:            AllocationScheme
    total_capital:     float
    weights:           Dict[str, float]
    allocated_capital: Dict[str, float]
    allocated_shares:  Dict[str, int]    = field(default_factory=dict)
    uninvested_cash:   float             = 0.0
    volatilities:      Dict[str, float]  = field(default_factory=dict)

    # ── Derived properties ────────────────────────────────────────────────────
    @property
    def stocks(self) -> List[str]:
        """Ordered list of stock names."""
        return list(self.weights.keys())

    @property
    def deployment_pct(self) -> float:
        """Percentage of capital deployed into shares."""
        deployed = sum(self.allocated_shares.get(s, 0) * 0
                       for s in self.stocks)
        return (1 - self.uninvested_cash / self.total_capital) * 100

    def summary_df(self) -> pd.DataFrame:
        """Return a tidy DataFrame summarising the allocation."""
        rows = []
        for stock in self.stocks:
            rows.append({
                "stock":            stock,
                "weight_pct":       round(self.weights[stock] * 100, 2),
                "allocated_inr":    round(self.allocated_capital[stock], 2),
                "shares":           self.allocated_shares.get(stock, 0),
                "volatility_pct":   round(
                    self.volatilities.get(stock, 0) * 100, 2
                ),
            })
        return pd.DataFrame(rows)

    def __str__(self) -> str:
        lines = [
            f"AllocationResult  [{self.scheme.name}]  "
            f"capital=₹{self.total_capital:,.0f}",
        ]
        for stock in self.stocks:
            sh = self.allocated_shares.get(stock, "N/A")
            lines.append(
                f"  {stock:<12}  {self.weights[stock]*100:>6.2f}%  "
                f"₹{self.allocated_capital[stock]:>12,.2f}  "
                f"shares={sh}"
            )
        if self.uninvested_cash:
            lines.append(f"  Uninvested cash: ₹{self.uninvested_cash:,.2f}")
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Validation helpers
# ──────────────────────────────────────────────────────────────────────────────
def validate_weights(
    weights: Dict[str, float],
    stocks: List[str],
) -> Tuple[bool, str]:
    """
    Validate that *weights* covers all *stocks* and sums to 1.0.

    Parameters
    ----------
    weights:
        Proposed weight mapping ``{stock: fraction}``.
    stocks:
        Required stock names.

    Returns
    -------
    (bool, str)
        ``(True, "")`` on success; ``(False, reason)`` on failure.
    """
    # Coverage check
    missing = [s for s in stocks if s not in weights]
    if missing:
        return False, f"Weights missing for stocks: {missing}"

    extra = [s for s in weights if s not in stocks]
    if extra:
        return False, f"Weights provided for unknown stocks: {extra}"

    # Negativity check
    neg = {s: w for s, w in weights.items() if w < 0}
    if neg:
        return False, f"Negative weights not allowed: {neg}"

    # Sum check
    total = sum(weights.values())
    if abs(total - 1.0) > WEIGHT_TOLERANCE:
        return False, (
            f"Weights sum to {total:.6f}, must be 1.0 "
            f"(tolerance ±{WEIGHT_TOLERANCE})."
        )
    return True, ""


def _compute_shares(
    capital_per_stock: Dict[str, float],
    prices: Dict[str, float],
) -> Tuple[Dict[str, int], float]:
    """
    Convert capital budgets to whole-share counts.

    Parameters
    ----------
    capital_per_stock:
        ``{stock: inr_budget}``.
    prices:
        ``{stock: current_price}``.

    Returns
    -------
    (shares_dict, uninvested_cash)
    """
    shares: Dict[str, int] = {}
    spent = 0.0
    for stock, budget in capital_per_stock.items():
        price = prices.get(stock)
        if price is None or price <= 0:
            logger.warning("No valid price for %s — shares set to 0.", stock)
            shares[stock] = 0
        else:
            shares[stock] = int(budget // price)
            spent += shares[stock] * price
    uninvested = sum(capital_per_stock.values()) - spent
    return shares, max(uninvested, 0.0)


# ──────────────────────────────────────────────────────────────────────────────
# Allocator 1 — Equal Weight
# ──────────────────────────────────────────────────────────────────────────────
class EqualWeightAllocator:
    """
    Allocate capital equally across all stocks (1/N per stock).

    The simplest diversification strategy.  All positions start with
    identical INR value, so the portfolio is automatically rebalanced
    toward equal weight at inception.

    Examples
    --------
    >>> alloc = EqualWeightAllocator()
    >>> result = alloc.allocate(
    ...     capital=1_000_000,
    ...     stocks=["TCS", "RELIANCE", "INFOSYS"],
    ...     prices={"TCS": 3400, "RELIANCE": 2800, "INFOSYS": 1500},
    ... )
    """

    def allocate(
        self,
        capital: float,
        stocks: List[str],
        prices: Optional[Dict[str, float]] = None,
    ) -> AllocationResult:
        """
        Parameters
        ----------
        capital:
            Total capital to deploy (INR).
        stocks:
            List of stock names.
        prices:
            Current prices — required to compute share counts.

        Returns
        -------
        AllocationResult
        """
        if capital <= 0:
            raise ValueError(f"capital must be positive, got {capital}")
        if not stocks:
            raise ValueError("stocks list must not be empty.")

        w = 1.0 / len(stocks)
        weights = {s: w for s in stocks}
        allocated = {s: capital * w for s in stocks}
        shares, cash = _compute_shares(allocated, prices or {})

        result = AllocationResult(
            scheme=AllocationScheme.EQUAL_WEIGHT,
            total_capital=capital,
            weights=weights,
            allocated_capital=allocated,
            allocated_shares=shares,
            uninvested_cash=cash,
        )
        logger.info(
            "EqualWeight  N=%d  each=%.2f%%  capital=₹%s",
            len(stocks), w * 100, f"{capital:,.0f}",
        )
        return result


# ──────────────────────────────────────────────────────────────────────────────
# Allocator 2 — Custom Weight
# ──────────────────────────────────────────────────────────────────────────────
class CustomWeightAllocator:
    """
    Allocate capital using user-specified target weights.

    Parameters
    ----------
    weights : Dict[str, float]
        ``{stock: fraction}`` — must sum to 1.0.

    Examples
    --------
    >>> alloc = CustomWeightAllocator(
    ...     weights={"TCS": 0.40, "RELIANCE": 0.35, "INFOSYS": 0.25}
    ... )
    >>> result = alloc.allocate(capital=1_000_000, stocks=["TCS", "RELIANCE", "INFOSYS"])
    """

    def __init__(self, weights: Dict[str, float]) -> None:
        self._weights = weights

    def allocate(
        self,
        capital: float,
        stocks: List[str],
        prices: Optional[Dict[str, float]] = None,
    ) -> AllocationResult:
        """
        Parameters
        ----------
        capital:
            Total capital to deploy.
        stocks:
            Must match the keys of the weight dict.
        prices:
            Current prices for share-count calculation.

        Returns
        -------
        AllocationResult

        Raises
        ------
        ValueError
            If weights fail validation.
        """
        ok, reason = validate_weights(self._weights, stocks)
        if not ok:
            raise ValueError(f"Invalid custom weights: {reason}")

        allocated = {s: capital * self._weights[s] for s in stocks}
        shares, cash = _compute_shares(allocated, prices or {})

        result = AllocationResult(
            scheme=AllocationScheme.CUSTOM,
            total_capital=capital,
            weights=dict(self._weights),
            allocated_capital=allocated,
            allocated_shares=shares,
            uninvested_cash=cash,
        )
        logger.info(
            "CustomWeight  weights=%s  capital=₹%s",
            {s: f"{w*100:.1f}%" for s, w in self._weights.items()},
            f"{capital:,.0f}",
        )
        return result


# ──────────────────────────────────────────────────────────────────────────────
# Allocator 3 — Risk Parity (Inverse Volatility)
# ──────────────────────────────────────────────────────────────────────────────
class RiskParityAllocator:
    """
    Allocate capital such that every position contributes equal *risk*.

    Uses inverse-volatility weighting over a lookback window of recent
    closing prices:

    ::

        w_i = (1 / σ_i) / Σ_j (1 / σ_j)

    where σ_i is the annualised volatility of stock i.

    Parameters
    ----------
    lookback_days : int
        Number of recent trading days used to estimate volatility.
        Default: 60 (≈ 3 months on NSE).

    Why it suits Indian large-caps
    -------------------------------
    TCS historically has lower volatility than Reliance or Infosys
    (beta ~0.7 vs 1.0–1.2).  Risk parity naturally over-weights TCS,
    which on this dataset is the best-performing stock.
    """

    def __init__(self, lookback_days: int = DEFAULT_LOOKBACK_DAYS) -> None:
        if lookback_days < 10:
            raise ValueError(f"lookback_days must be ≥ 10, got {lookback_days}")
        self.lookback_days = lookback_days

    def compute_weights(
        self,
        price_series: Dict[str, pd.Series],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Compute inverse-volatility weights from price history.

        Parameters
        ----------
        price_series:
            ``{stock: pd.Series of closing prices}``.

        Returns
        -------
        (weights, annualised_vols)
        """
        vols: Dict[str, float] = {}
        for stock, prices in price_series.items():
            rets = prices.pct_change().dropna()
            recent = rets.iloc[-self.lookback_days:]
            if len(recent) < 10:
                raise ValueError(
                    f"Insufficient price history for {stock} — "
                    f"need ≥ 10 returns, got {len(recent)}."
                )
            ann_vol = recent.std() * np.sqrt(252)
            vols[stock] = ann_vol

        inv_vols   = {s: 1.0 / v for s, v in vols.items()}
        total_inv  = sum(inv_vols.values())
        weights    = {s: iv / total_inv for s, iv in inv_vols.items()}

        logger.info(
            "RiskParity  lookback=%d days  vols=%s  weights=%s",
            self.lookback_days,
            {s: f"{v*100:.1f}%" for s, v in vols.items()},
            {s: f"{w*100:.1f}%" for s, w in weights.items()},
        )
        return weights, vols

    def allocate(
        self,
        capital: float,
        stocks: List[str],
        price_series: Dict[str, pd.Series],
        prices: Optional[Dict[str, float]] = None,
    ) -> AllocationResult:
        """
        Parameters
        ----------
        capital:
            Total capital to deploy.
        stocks:
            Stock names — must match keys in *price_series*.
        price_series:
            Historical closing prices per stock for vol estimation.
        prices:
            Current prices for share-count calculation.

        Returns
        -------
        AllocationResult
        """
        if capital <= 0:
            raise ValueError(f"capital must be positive, got {capital}")

        weights, vols = self.compute_weights(
            {s: price_series[s] for s in stocks}
        )
        allocated = {s: capital * weights[s] for s in stocks}
        shares, cash = _compute_shares(allocated, prices or {})

        return AllocationResult(
            scheme=AllocationScheme.RISK_PARITY,
            total_capital=capital,
            weights=weights,
            allocated_capital=allocated,
            allocated_shares=shares,
            uninvested_cash=cash,
            volatilities=vols,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Allocator 4 — Market Cap Proxy
# ──────────────────────────────────────────────────────────────────────────────
class MarketCapAllocator:
    """
    Allocate capital proportional to approximate free-float market caps.

    Mimics a passive index fund's exposure.  For the three NSE stocks in
    this project, :data:`DEFAULT_MARKET_CAPS` provides 2024 reference figures.

    Parameters
    ----------
    market_caps : Dict[str, float]
        ``{stock: market_cap_in_crore}`` for the stocks in the universe.
    """

    def __init__(
        self,
        market_caps: Dict[str, float] = DEFAULT_MARKET_CAPS,
    ) -> None:
        if any(v <= 0 for v in market_caps.values()):
            raise ValueError("All market caps must be positive.")
        self._caps = market_caps

    def allocate(
        self,
        capital: float,
        stocks: List[str],
        prices: Optional[Dict[str, float]] = None,
    ) -> AllocationResult:
        """
        Parameters
        ----------
        capital:
            Total capital.
        stocks:
            Subset of stocks from the market-cap universe.
        prices:
            Current prices for share-count calculation.

        Returns
        -------
        AllocationResult
        """
        missing = [s for s in stocks if s not in self._caps]
        if missing:
            raise ValueError(
                f"Market cap not available for: {missing}. "
                f"Available: {list(self._caps.keys())}"
            )
        subset_caps = {s: self._caps[s] for s in stocks}
        total_cap   = sum(subset_caps.values())
        weights     = {s: c / total_cap for s, c in subset_caps.items()}
        allocated   = {s: capital * weights[s] for s in stocks}
        shares, cash = _compute_shares(allocated, prices or {})

        logger.info(
            "MarketCap  weights=%s  capital=₹%s",
            {s: f"{w*100:.1f}%" for s, w in weights.items()},
            f"{capital:,.0f}",
        )
        return AllocationResult(
            scheme=AllocationScheme.MARKET_CAP,
            total_capital=capital,
            weights=weights,
            allocated_capital=allocated,
            allocated_shares=shares,
            uninvested_cash=cash,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Facade — PortfolioAllocator
# ──────────────────────────────────────────────────────────────────────────────
class PortfolioAllocator:
    """
    High-level facade that selects and runs the appropriate allocator.

    Parameters
    ----------
    scheme : AllocationScheme
        Which allocation strategy to use.
    custom_weights : Dict[str, float], optional
        Required when ``scheme=CUSTOM``.
    market_caps : Dict[str, float], optional
        Overrides the default market caps for ``scheme=MARKET_CAP``.
    lookback_days : int
        Volatility lookback for risk-parity.  Default: 60.

    Examples
    --------
    Equal weight::

        pa = PortfolioAllocator(AllocationScheme.EQUAL_WEIGHT)
        result = pa.allocate(1_000_000, ["TCS", "RELIANCE", "INFOSYS"],
                             current_prices={"TCS": 3400, ...})

    Risk parity::

        pa = PortfolioAllocator(AllocationScheme.RISK_PARITY)
        result = pa.allocate(1_000_000, stocks,
                             current_prices=prices,
                             price_history=price_dict)
    """

    def __init__(
        self,
        scheme: AllocationScheme            = AllocationScheme.EQUAL_WEIGHT,
        custom_weights: Optional[Dict[str, float]] = None,
        market_caps: Optional[Dict[str, float]]    = None,
        lookback_days: int                         = DEFAULT_LOOKBACK_DAYS,
    ) -> None:
        self.scheme = scheme
        self._equal  = EqualWeightAllocator()
        self._custom = (
            CustomWeightAllocator(custom_weights) if custom_weights else None
        )
        self._rp     = RiskParityAllocator(lookback_days=lookback_days)
        self._mc     = MarketCapAllocator(
            market_caps=market_caps or DEFAULT_MARKET_CAPS
        )

    def allocate(
        self,
        capital: float,
        stocks: List[str],
        current_prices: Optional[Dict[str, float]]      = None,
        price_history: Optional[Dict[str, pd.Series]]   = None,
    ) -> AllocationResult:
        """
        Run the configured allocation scheme.

        Parameters
        ----------
        capital:
            Total capital to allocate.
        stocks:
            Stock names.
        current_prices:
            Prices at allocation date (for share count calculation).
        price_history:
            Historical close prices — required for ``RISK_PARITY``.

        Returns
        -------
        AllocationResult
        """
        if self.scheme == AllocationScheme.EQUAL_WEIGHT:
            return self._equal.allocate(capital, stocks, current_prices)

        elif self.scheme == AllocationScheme.CUSTOM:
            if self._custom is None:
                raise RuntimeError(
                    "CUSTOM scheme requires 'custom_weights' in __init__."
                )
            return self._custom.allocate(capital, stocks, current_prices)

        elif self.scheme == AllocationScheme.RISK_PARITY:
            if price_history is None:
                raise ValueError(
                    "RISK_PARITY scheme requires 'price_history' argument."
                )
            return self._rp.allocate(
                capital, stocks, price_history, current_prices
            )

        elif self.scheme == AllocationScheme.MARKET_CAP:
            return self._mc.allocate(capital, stocks, current_prices)

        else:
            raise ValueError(f"Unknown scheme: {self.scheme}")

    def load_price_history(
        self,
        stocks: List[str],
        processed_dir: str = PROCESSED_DIR,
    ) -> Dict[str, pd.Series]:
        """
        Convenience loader for risk-parity allocation.

        Reads processed CSVs and returns a ``{stock: close_series}`` dict.

        Parameters
        ----------
        stocks:
            Stock names matching processed CSV filenames.
        processed_dir:
            Directory containing processed CSVs.

        Returns
        -------
        Dict[str, pd.Series]
        """
        history: Dict[str, pd.Series] = {}
        for stock in stocks:
            path = os.path.join(processed_dir, f"{stock}_processed.csv")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Processed CSV not found: {path}\n"
                    "Run src/data/preprocess.py first."
                )
            df = pd.read_csv(path, parse_dates=["date"])
            df = df.set_index("date").sort_index()
            history[stock] = df["close"]
        return history


# ──────────────────────────────────────────────────────────────────────────────
# Script entry-point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _root not in sys.path:
        sys.path.insert(0, _root)

    print("=" * 68)
    print("  Portfolio Allocator — Demo")
    print("=" * 68)

    STOCKS  = ["TCS", "RELIANCE", "INFOSYS"]
    CAPITAL = 1_000_000.0   # ₹ 10,00,000

    # Load data
    pa_rp = PortfolioAllocator(AllocationScheme.RISK_PARITY)
    history = pa_rp.load_price_history(STOCKS)
    last_prices = {s: float(history[s].iloc[-1]) for s in STOCKS}

    print(f"\n  Capital: ₹{CAPITAL:,.0f}")
    print(f"  Stocks : {STOCKS}")
    print(f"  Prices : " + "  ".join(f"{s}=₹{p:,.2f}" for s, p in last_prices.items()))

    configs = [
        ("Equal Weight",  AllocationScheme.EQUAL_WEIGHT, {}),
        ("Custom (40/35/25)", AllocationScheme.CUSTOM,
         {"custom_weights": {"TCS": 0.40, "RELIANCE": 0.35, "INFOSYS": 0.25}}),
        ("Risk Parity",   AllocationScheme.RISK_PARITY, {}),
        ("Market Cap",    AllocationScheme.MARKET_CAP, {}),
    ]

    for label, scheme, kwargs in configs:
        print(f"\n{'─' * 60}")
        print(f"  {label}")
        print(f"{'─' * 60}")
        alloc = PortfolioAllocator(scheme, **kwargs)
        result = alloc.allocate(
            CAPITAL, STOCKS,
            current_prices=last_prices,
            price_history=history if scheme == AllocationScheme.RISK_PARITY else None,
        )
        df = result.summary_df()
        hdr = (f"  {'Stock':<12} {'Weight%':>8}  {'Allocated ₹':>13}  "
               f"{'Shares':>7}  {'Vol%':>7}")
        print(hdr)
        print("  " + "─" * (len(hdr) - 2))
        for _, row in df.iterrows():
            print(
                f"  {row['stock']:<12} {row['weight_pct']:>7.2f}%  "
                f"₹{row['allocated_inr']:>12,.2f}  "
                f"{row['shares']:>7}  "
                f"{row['volatility_pct']:>6.2f}%"
            )
        print(f"  {'Uninvested cash':<20} ₹{result.uninvested_cash:>10,.2f}")
    print()
