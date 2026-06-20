"""
rebalance.py
============
Portfolio rebalancing engine for the Algorithmic Trading Strategy
Backtester.

Why rebalancing matters
------------------------
Without rebalancing, a winning stock (TCS) will grow to dominate the
portfolio, transforming a diversified 3-stock portfolio into a
TCS-concentrated bet.  Rebalancing periodically sells the winner and
tops up the laggards, restoring the original risk profile.

Three rebalance trigger modes
------------------------------
1. **Scheduled** — rebalance at fixed calendar intervals regardless
   of drift (monthly ≈ every 21 trading days; quarterly ≈ every 63).

2. **Threshold (Drift)** — rebalance only when any stock's actual
   weight deviates more than *drift_threshold* from its target.
   ``drift_i = |actual_weight_i − target_weight_i|``
   Avoids unnecessary turnover in low-drift periods.

3. **Hybrid** — rebalance on schedule **or** if drift threshold is
   breached between scheduled dates, whichever comes first.
   The recommended production mode.

Transaction cost model
-----------------------
Each rebalance trade incurs a round-trip commission of *commission_pct*
per leg.  Default: 0.05 % (realistic NSE all-in cost).

Architecture
------------
* :class:`RebalanceFrequency` — enum of trigger modes.
* :class:`RebalanceTrade` — per-stock action at a rebalance event.
* :class:`RebalanceEvent` — full snapshot of one rebalance.
* :class:`RebalanceSimResult` — output of a historical simulation.
* :class:`RebalanceEngine` — orchestrates all logic.
* ``simulate_portfolio()`` — convenience function for end-to-end use.

Integration
-----------
Receives :class:`AllocationResult` from ``allocation.py`` for target
weights.  Produces :class:`RebalanceSimResult` whose
``portfolio_value_series`` feeds directly into ``portfolio_metrics.py``.

Usage
-----
::

    python src/portfolio/rebalance.py
    from src.portfolio.rebalance import RebalanceEngine, RebalanceFrequency
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
PROCESSED_DIR: str    = os.path.join("data", "processed")
DEFAULT_COMMISSION:float = 0.0005    # 0.05 % per leg
DEFAULT_DRIFT_THRESHOLD: float = 0.05  # 5 % absolute drift triggers rebalance
MONTHLY_TRADING_DAYS: int  = 21
QUARTERLY_TRADING_DAYS: int = 63


# ──────────────────────────────────────────────────────────────────────────────
# Enums & value objects
# ──────────────────────────────────────────────────────────────────────────────
class RebalanceFrequency(Enum):
    """Available rebalancing trigger modes."""
    MONTHLY   = auto()   # Every ~21 trading days
    QUARTERLY = auto()   # Every ~63 trading days
    THRESHOLD = auto()   # Only when drift > threshold
    HYBRID    = auto()   # Scheduled OR drift, whichever fires first


@dataclass
class RebalanceTrade:
    """
    A single buy or sell order generated during a rebalance.

    Attributes
    ----------
    stock : str
    action : str
        ``'BUY'``, ``'SELL'``, or ``'HOLD'``.
    current_shares : int
    target_shares : int
    shares_delta : int
        Positive = buy more; negative = sell.
    price : float
        Execution price (current market price).
    trade_value : float
        ``abs(shares_delta) × price``.
    commission : float
        Transaction cost for this leg.
    """
    stock:          str
    action:         str
    current_shares: int
    target_shares:  int
    shares_delta:   int
    price:          float
    trade_value:    float
    commission:     float

    @property
    def net_cash_flow(self) -> float:
        """
        Net cash impact on the portfolio.

        Negative when buying (cash outflow), positive when selling.
        Commission is always a cost.
        """
        return -(self.shares_delta * self.price) - self.commission


@dataclass
class RebalanceEvent:
    """
    Full snapshot of one portfolio rebalance.

    Attributes
    ----------
    date : pd.Timestamp
    trigger : str
        ``'SCHEDULED'`` or ``'DRIFT'``.
    portfolio_value_before : float
    portfolio_value_after : float
    trades : List[RebalanceTrade]
    pre_weights : Dict[str, float]
    post_weights : Dict[str, float]
    max_drift : float
        Largest single-stock weight deviation before rebalance.
    total_commission : float
    """
    date:                    pd.Timestamp
    trigger:                 str
    portfolio_value_before:  float
    portfolio_value_after:   float
    trades:                  List[RebalanceTrade]
    pre_weights:             Dict[str, float]
    post_weights:            Dict[str, float]
    max_drift:               float
    total_commission:        float

    @property
    def turnover_pct(self) -> float:
        """Total value traded as % of portfolio value before rebalance."""
        total_traded = sum(abs(t.trade_value) for t in self.trades)
        return total_traded / max(self.portfolio_value_before, 1) * 100

    def __str__(self) -> str:
        lines = [
            f"RebalanceEvent  {self.date.date()}  [{self.trigger}]",
            f"  Value: ₹{self.portfolio_value_before:>12,.2f} → "
            f"₹{self.portfolio_value_after:>12,.2f}",
            f"  Max drift: {self.max_drift*100:.2f}%  "
            f"Turnover: {self.turnover_pct:.1f}%  "
            f"Commission: ₹{self.total_commission:,.2f}",
        ]
        for t in self.trades:
            if t.action != "HOLD":
                lines.append(
                    f"    {t.action:<4}  {t.stock:<10}  "
                    f"{abs(t.shares_delta):>5} shares @ ₹{t.price:>8,.2f}  "
                    f"= ₹{t.trade_value:>10,.2f}"
                )
        return "\n".join(lines)


@dataclass
class RebalanceSimResult:
    """
    Output of a full historical rebalance simulation.

    Attributes
    ----------
    portfolio_value_series : pd.Series
        Daily portfolio value indexed by date.
    holdings_history : pd.DataFrame
        Shares held per stock at each date.
    rebalance_events : List[RebalanceEvent]
    trade_log : pd.DataFrame
        Flat log of every trade for integration with metrics.
    final_holdings : Dict[str, int]
    final_cash : float
    total_rebalances : int
    total_commission_paid : float
    """
    portfolio_value_series: pd.Series
    holdings_history:       pd.DataFrame
    rebalance_events:       List[RebalanceEvent]
    trade_log:              pd.DataFrame
    final_holdings:         Dict[str, int]
    final_cash:             float
    total_rebalances:       int
    total_commission_paid:  float


# ──────────────────────────────────────────────────────────────────────────────
# Core engine
# ──────────────────────────────────────────────────────────────────────────────
class RebalanceEngine:
    """
    Orchestrates portfolio rebalancing logic and historical simulation.

    Parameters
    ----------
    target_weights : Dict[str, float]
        ``{stock: target_fraction}`` — must sum to 1.0.
    frequency : RebalanceFrequency
        Rebalance trigger mode.  Default: ``HYBRID``.
    drift_threshold : float
        Max allowable weight deviation (absolute) before a drift
        rebalance fires.  Default: 0.05 (5 %).
    commission_pct : float
        Per-leg commission as a fraction.  Default: 0.0005 (0.05 %).
    """

    def __init__(
        self,
        target_weights:   Dict[str, float],
        frequency:        RebalanceFrequency = RebalanceFrequency.HYBRID,
        drift_threshold:  float              = DEFAULT_DRIFT_THRESHOLD,
        commission_pct:   float              = DEFAULT_COMMISSION,
    ) -> None:
        total = sum(target_weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"target_weights must sum to 1.0, got {total:.6f}."
            )
        if drift_threshold <= 0:
            raise ValueError(f"drift_threshold must be > 0, got {drift_threshold}")

        self.target_weights  = target_weights
        self.frequency       = frequency
        self.drift_threshold = drift_threshold
        self.commission_pct  = commission_pct
        self._stocks         = list(target_weights.keys())

        logger.info(
            "RebalanceEngine  freq=%s  drift_thr=%.1f%%  commission=%.3f%%",
            frequency.name, drift_threshold * 100, commission_pct * 100,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────
    def _compute_weights(
        self,
        holdings: Dict[str, int],
        prices:   Dict[str, float],
    ) -> Tuple[Dict[str, float], float]:
        """
        Compute current portfolio weights and total value.

        Returns
        -------
        (current_weights, portfolio_equity_value)
        """
        values = {s: holdings[s] * prices[s] for s in self._stocks}
        total  = sum(values.values())
        if total <= 0:
            return {s: 0.0 for s in self._stocks}, 0.0
        weights = {s: v / total for s, v in values.items()}
        return weights, total

    def _compute_max_drift(
        self,
        current_weights: Dict[str, float],
    ) -> float:
        """Maximum absolute deviation from target across all stocks."""
        return max(
            abs(current_weights[s] - self.target_weights[s])
            for s in self._stocks
        )

    def needs_rebalance(
        self,
        current_weights:     Dict[str, float],
        bars_since_last:     int,
    ) -> Tuple[bool, str]:
        """
        Decide whether a rebalance should fire on the current bar.

        Parameters
        ----------
        current_weights:
            Actual portfolio weights right now.
        bars_since_last:
            Number of trading bars since the last rebalance.

        Returns
        -------
        (bool, trigger_reason)
            ``(True, 'SCHEDULED')`` or ``(True, 'DRIFT')``
            or ``(False, '')``.
        """
        # Scheduled trigger
        if self.frequency == RebalanceFrequency.MONTHLY:
            scheduled_period = MONTHLY_TRADING_DAYS
        elif self.frequency == RebalanceFrequency.QUARTERLY:
            scheduled_period = QUARTERLY_TRADING_DAYS
        else:
            scheduled_period = None   # THRESHOLD has no schedule

        max_drift = self._compute_max_drift(current_weights)

        if self.frequency == RebalanceFrequency.THRESHOLD:
            if max_drift > self.drift_threshold:
                return True, "DRIFT"
            return False, ""

        if self.frequency in (RebalanceFrequency.MONTHLY,
                               RebalanceFrequency.QUARTERLY):
            if bars_since_last >= scheduled_period:  # type: ignore[operator]
                return True, "SCHEDULED"
            return False, ""

        # HYBRID: scheduled OR drift
        if bars_since_last >= MONTHLY_TRADING_DAYS:
            return True, "SCHEDULED"
        if max_drift > self.drift_threshold:
            return True, "DRIFT"
        return False, ""

    def compute_trades(
        self,
        holdings:        Dict[str, int],
        prices:          Dict[str, float],
        cash:            float,
    ) -> Tuple[List[RebalanceTrade], float]:
        """
        Compute share adjustments needed to restore target weights.

        The total portfolio value (equity + cash) is redistributed
        according to *target_weights*.  Share counts are rounded down
        to whole numbers; residual cash remains in the cash balance.

        Parameters
        ----------
        holdings:
            Current whole-share holdings per stock.
        prices:
            Current market prices per stock.
        cash:
            Current uninvested cash balance.

        Returns
        -------
        (trades, remaining_cash)
            Remaining cash after executing all trades.
        """
        equity  = sum(holdings[s] * prices[s] for s in self._stocks)
        total_v = equity + cash

        trades:    List[RebalanceTrade] = []
        new_cash   = cash

        # Compute target shares for each stock
        target_shares: Dict[str, int] = {}
        for stock in self._stocks:
            budget        = total_v * self.target_weights[stock]
            target_shares[stock] = int(budget // prices[stock])

        # Build trade list
        for stock in self._stocks:
            curr = holdings[stock]
            tgt  = target_shares[stock]
            delta = tgt - curr
            price = prices[stock]

            if delta > 0:
                action     = "BUY"
                trade_val  = delta * price
                commission = trade_val * self.commission_pct
                new_cash  -= (trade_val + commission)
            elif delta < 0:
                action     = "SELL"
                trade_val  = abs(delta) * price
                commission = trade_val * self.commission_pct
                new_cash  += (trade_val - commission)
            else:
                action    = "HOLD"
                trade_val = 0.0
                commission = 0.0

            trades.append(RebalanceTrade(
                stock          = stock,
                action         = action,
                current_shares = curr,
                target_shares  = tgt,
                shares_delta   = delta,
                price          = price,
                trade_value    = round(trade_val, 2),
                commission     = round(commission, 2),
            ))

        return trades, max(new_cash, 0.0)

    # ── Simulation ─────────────────────────────────────────────────────────────
    def simulate(
        self,
        price_data: pd.DataFrame,
        initial_capital: float,
    ) -> RebalanceSimResult:
        """
        Run a full historical portfolio simulation with rebalancing.

        Steps
        -----
        1. On the first bar, buy initial holdings per target weights.
        2. On each subsequent bar:
           a. Mark portfolio to market.
           b. Check if rebalance is needed.
           c. If yes, compute and apply rebalance trades.
        3. Track daily portfolio value, holdings, and all events.

        Parameters
        ----------
        price_data : pd.DataFrame
            Columns = stock names, index = DatetimeIndex of trading days.
            Produced by :func:`load_aligned_prices`.
        initial_capital : float
            Starting cash.

        Returns
        -------
        RebalanceSimResult
        """
        dates   = price_data.index
        stocks  = self._stocks

        # Validate that all stocks are in price_data
        missing = [s for s in stocks if s not in price_data.columns]
        if missing:
            raise KeyError(f"Stocks missing from price_data: {missing}")

        # ── Initialise ────────────────────────────────────────────────────────
        cash: float          = initial_capital
        holdings: Dict[str, int] = {s: 0 for s in stocks}

        portfolio_values: List[float]     = []
        holdings_log: List[Dict]          = []
        rebalance_events: List[RebalanceEvent] = []
        all_trades: List[Dict]            = []

        last_rebalance_bar: int = -999
        total_commission: float = 0.0

        for bar_idx, date in enumerate(dates):
            prices = {s: float(price_data.loc[date, s]) for s in stocks}

            # ── First bar — initial investment ────────────────────────────────
            if bar_idx == 0:
                for stock in stocks:
                    budget   = cash * self.target_weights[stock]
                    qty      = int(budget // prices[stock])
                    cost     = qty * prices[stock]
                    comm     = cost * self.commission_pct
                    holdings[stock] = qty
                    cash -= (cost + comm)
                    total_commission += comm
                    all_trades.append({
                        "date":   date, "stock":  stock,
                        "action": "BUY", "shares": qty,
                        "price":  prices[stock], "value": cost,
                        "commission": comm, "trigger": "INITIAL",
                    })
                last_rebalance_bar = 0
                logger.info(
                    "Simulation START  %s  capital=₹%s  "
                    "initial buys complete",
                    date.date(), f"{initial_capital:,.0f}",
                )

            # ── Subsequent bars — mark-to-market ──────────────────────────────
            else:
                equity = sum(holdings[s] * prices[s] for s in stocks)
                total_v = equity + cash

                current_weights, _ = self._compute_weights(holdings, prices)
                bars_since = bar_idx - last_rebalance_bar
                fire, trigger = self.needs_rebalance(current_weights, bars_since)

                if fire:
                    max_drift = self._compute_max_drift(current_weights)
                    pre_weights = dict(current_weights)
                    pre_value   = total_v

                    trades, cash = self.compute_trades(holdings, prices, cash)

                    # Apply share changes to holdings
                    for t in trades:
                        holdings[t.stock] = t.target_shares
                        if t.action != "HOLD":
                            total_commission += t.commission
                            all_trades.append({
                                "date":       date,
                                "stock":      t.stock,
                                "action":     t.action,
                                "shares":     abs(t.shares_delta),
                                "price":      t.price,
                                "value":      t.trade_value,
                                "commission": t.commission,
                                "trigger":    trigger,
                            })

                    post_weights, post_equity = self._compute_weights(holdings, prices)
                    post_value = post_equity + cash
                    event_commission = sum(t.commission for t in trades)

                    event = RebalanceEvent(
                        date=date,
                        trigger=trigger,
                        portfolio_value_before=pre_value,
                        portfolio_value_after=post_value,
                        trades=trades,
                        pre_weights=pre_weights,
                        post_weights=post_weights,
                        max_drift=max_drift,
                        total_commission=event_commission,
                    )
                    rebalance_events.append(event)
                    last_rebalance_bar = bar_idx

                    logger.info(
                        "REBALANCE  %s  [%s]  drift=%.2f%%  "
                        "value=₹%s → ₹%s  commission=₹%.2f",
                        date.date(), trigger, max_drift * 100,
                        f"{pre_value:,.0f}", f"{post_value:,.0f}",
                        event_commission,
                    )

            # ── Record daily state ─────────────────────────────────────────────
            equity    = sum(holdings[s] * float(price_data.loc[date, s])
                            for s in stocks)
            port_val  = equity + cash
            portfolio_values.append(port_val)
            snap = {"date": date, "cash": cash, "total": port_val}
            for s in stocks:
                snap[f"shares_{s}"]  = holdings[s]
                snap[f"value_{s}"]   = holdings[s] * float(price_data.loc[date, s])
            holdings_log.append(snap)

        # ── Assemble result ───────────────────────────────────────────────────
        value_series = pd.Series(portfolio_values, index=dates, name="portfolio_value")
        holdings_df  = pd.DataFrame(holdings_log).set_index("date")
        trade_log    = pd.DataFrame(all_trades) if all_trades else pd.DataFrame()

        logger.info(
            "Simulation END  %s → %s  rebalances=%d  "
            "start=₹%s  end=₹%s  commission_total=₹%.2f",
            dates[0].date(), dates[-1].date(),
            len(rebalance_events),
            f"{initial_capital:,.0f}",
            f"{portfolio_values[-1]:,.0f}",
            total_commission,
        )

        return RebalanceSimResult(
            portfolio_value_series = value_series,
            holdings_history       = holdings_df,
            rebalance_events       = rebalance_events,
            trade_log              = trade_log,
            final_holdings         = dict(holdings),
            final_cash             = cash,
            total_rebalances       = len(rebalance_events),
            total_commission_paid  = total_commission,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Data loader helper
# ──────────────────────────────────────────────────────────────────────────────
def load_aligned_prices(
    stocks: List[str],
    processed_dir: str = PROCESSED_DIR,
) -> pd.DataFrame:
    """
    Load and align closing prices for multiple stocks.

    Returns a DataFrame with one column per stock, indexed by the
    *intersection* of their trading dates (inner join).

    Parameters
    ----------
    stocks:
        Stock display names matching processed CSV filenames.
    processed_dir:
        Directory of processed CSV files.

    Returns
    -------
    pd.DataFrame
        Shape: (n_common_dates, n_stocks).  Columns = stock names.
    """
    frames: Dict[str, pd.Series] = {}
    for stock in stocks:
        path = os.path.join(processed_dir, f"{stock}_processed.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Processed CSV not found: {path}. Run preprocess.py first."
            )
        df = pd.read_csv(path, parse_dates=["date"])
        df = df.set_index("date").sort_index()
        frames[stock] = df["close"]

    aligned = pd.DataFrame(frames).dropna()
    logger.info(
        "Aligned %d stocks: %d common trading days  %s → %s",
        len(stocks), len(aligned),
        aligned.index[0].date(), aligned.index[-1].date(),
    )
    return aligned


# ──────────────────────────────────────────────────────────────────────────────
# Convenience function
# ──────────────────────────────────────────────────────────────────────────────
def simulate_portfolio(
    stocks:           List[str],
    target_weights:   Dict[str, float],
    initial_capital:  float                = 1_000_000.0,
    frequency:        RebalanceFrequency   = RebalanceFrequency.HYBRID,
    drift_threshold:  float                = 0.05,
    commission_pct:   float                = DEFAULT_COMMISSION,
    processed_dir:    str                  = PROCESSED_DIR,
) -> RebalanceSimResult:
    """
    End-to-end portfolio simulation in a single call.

    Parameters
    ----------
    stocks:
        Stock names.
    target_weights:
        ``{stock: fraction}`` — must sum to 1.0.
    initial_capital:
        Starting capital (INR).
    frequency:
        Rebalance trigger mode.
    drift_threshold:
        Maximum drift before a threshold rebalance fires.
    commission_pct:
        Per-leg commission.
    processed_dir:
        Path to processed CSVs.

    Returns
    -------
    RebalanceSimResult
    """
    prices = load_aligned_prices(stocks, processed_dir)
    engine = RebalanceEngine(
        target_weights  = target_weights,
        frequency       = frequency,
        drift_threshold = drift_threshold,
        commission_pct  = commission_pct,
    )
    return engine.simulate(prices, initial_capital)


# ──────────────────────────────────────────────────────────────────────────────
# Script entry-point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)
    )))
    if _root not in sys.path:
        sys.path.insert(0, _root)

    print("=" * 68)
    print("  Rebalance Engine — Demo")
    print("=" * 68)

    STOCKS          = ["TCS", "RELIANCE", "INFOSYS"]
    EQUAL_WEIGHTS   = {"TCS": 1/3, "RELIANCE": 1/3, "INFOSYS": 1/3}
    CAPITAL         = 1_000_000.0

    configs = [
        ("Equal Weight — Monthly",    EQUAL_WEIGHTS, RebalanceFrequency.MONTHLY,   0.05),
        ("Equal Weight — Quarterly",  EQUAL_WEIGHTS, RebalanceFrequency.QUARTERLY, 0.05),
        ("Equal Weight — Hybrid 5%",  EQUAL_WEIGHTS, RebalanceFrequency.HYBRID,    0.05),
        ("Equal Weight — Threshold 10%", EQUAL_WEIGHTS, RebalanceFrequency.THRESHOLD, 0.10),
    ]

    summary_rows = []
    for label, weights, freq, drift in configs:
        result = simulate_portfolio(
            stocks=STOCKS, target_weights=weights,
            initial_capital=CAPITAL, frequency=freq,
            drift_threshold=drift,
        )
        start_v = result.portfolio_value_series.iloc[0]
        end_v   = result.portfolio_value_series.iloc[-1]
        ret_pct = (end_v / CAPITAL - 1) * 100
        summary_rows.append({
            "Config":       label,
            "Rebalances":   result.total_rebalances,
            "Commission ₹": f"{result.total_commission_paid:,.0f}",
            "End Value ₹":  f"{end_v:,.0f}",
            "Return %":     f"{ret_pct:+.2f}%",
        })

    print("\n── Simulation Summary ───────────────────────────────────────────")
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    # Detailed rebalance event log for HYBRID strategy
    print("\n── Rebalance Events (Hybrid 5%) ─────────────────────────────────")
    hybrid_result = simulate_portfolio(
        stocks=STOCKS, target_weights=EQUAL_WEIGHTS,
        initial_capital=CAPITAL, frequency=RebalanceFrequency.HYBRID,
        drift_threshold=0.05,
    )
    for i, ev in enumerate(hybrid_result.rebalance_events[:6], 1):
        print(f"\n  Event #{i}")
        print(f"  {ev}")

    if len(hybrid_result.rebalance_events) > 6:
        print(f"\n  ... and {len(hybrid_result.rebalance_events) - 6} more events.")

    print(f"\n  Final holdings:")
    prices_last = load_aligned_prices(STOCKS).iloc[-1]
    for stock, shares in hybrid_result.final_holdings.items():
        val = shares * prices_last[stock]
        print(f"    {stock:<10}  {shares:>5} shares @ "
              f"₹{prices_last[stock]:>8,.2f} = ₹{val:>10,.2f}")
    print(f"    {'Cash':<10}  ₹{hybrid_result.final_cash:>10,.2f}")
    print(f"  Total commission paid: ₹{hybrid_result.total_commission_paid:,.2f}")
    print()
