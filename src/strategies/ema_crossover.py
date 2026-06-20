"""
ema_crossover.py
================
Backtrader strategy: EMA Golden Cross / Death Cross.

Trading logic
-------------
This is a classic **trend-following** strategy that exploits momentum in
Indian large-cap equities.  The two moving averages create a dynamic
support/resistance system:

* **GOLDEN CROSS** — fast EMA crosses *above* slow EMA → enter long.
  Historically, the 50×200 EMA golden cross on NSE stocks has marked
  the beginning of sustained multi-month up-trends.

* **DEATH CROSS** — fast EMA crosses *below* slow EMA → exit long.
  Protects capital by exiting before a bear trend deepens.

Defaults
--------
* ``fast_period``  = 50   (short-term trend)
* ``slow_period``  = 200  (long-term trend)
* ``stake_pct``    = 0.95 (invest 95 % of available cash per trade)
* Initial capital  = ₹5,00,000
* Commission       = 0.05 % per trade leg (NSE approximate all-in cost)

How it connects to Phase 4 (backtesting engine)
------------------------------------------------
``run_backtest()`` returns a plain ``dict`` with all key metrics so the
Phase 4 engine can call it for all three stocks and aggregate results into
a portfolio-level report without any modification to this file.

Usage
-----
::

    python src/strategies/ema_crossover.py          # demo run
    from src.strategies.ema_crossover import EMACrossoverStrategy, run_backtest
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Dict, Optional

import backtrader as bt
import pandas as pd

# ── Pandas 2.0 / 3.0 compatibility shim ──────────────────────────────────────
# backtrader 1.9.78 still calls DataFrame.iteritems() which was removed in
# pandas 2.0.  Patching it here is safe — items() is the identical replacement.
if not hasattr(pd.DataFrame, "iteritems"):
    pd.DataFrame.iteritems = pd.DataFrame.items

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
# Project-level constants
# ──────────────────────────────────────────────────────────────────────────────
PROCESSED_DATA_DIR: str = os.path.join("data", "processed")
INITIAL_CASH: float = 500_000.0   # ₹ 5,00,000 — realistic retail capital
COMMISSION: float = 0.0005        # 0.05 % per leg ≈ NSE all-in round-trip


# ──────────────────────────────────────────────────────────────────────────────
# Data feed loader
# ──────────────────────────────────────────────────────────────────────────────
def load_bt_feed(
    stock_name: str,
    processed_dir: str = PROCESSED_DATA_DIR,
) -> bt.feeds.PandasData:
    """
    Load a processed CSV file as a Backtrader ``PandasData`` feed.

    The processed CSV is expected to have columns:
    ``ticker, date, open, high, low, close, volume``.
    The ``ticker`` column is dropped before creating the feed; Backtrader
    auto-detects the remaining OHLCV columns by name.

    Parameters
    ----------
    stock_name:
        Display name (e.g. ``'TCS'``, ``'RELIANCE'``, ``'INFOSYS'``).
    processed_dir:
        Directory containing processed CSV files.

    Returns
    -------
    bt.feeds.PandasData
        Ready-to-use Backtrader data feed.

    Raises
    ------
    FileNotFoundError
        If the processed CSV does not exist.  Run ``preprocess.py`` first.
    """
    filepath = os.path.join(processed_dir, f"{stock_name.upper()}_processed.csv")
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Processed data not found: {filepath}\n"
            "Run src/data/preprocess.py to generate it."
        )
    df = pd.read_csv(filepath, parse_dates=["date"])
    df = df.drop(columns=["ticker"], errors="ignore")
    df = df.set_index("date")
    df.index = pd.DatetimeIndex(df.index)
    df = df.sort_index()
    return bt.feeds.PandasData(dataname=df)


# ──────────────────────────────────────────────────────────────────────────────
# Strategy
# ──────────────────────────────────────────────────────────────────────────────
class EMACrossoverStrategy(bt.Strategy):
    """
    EMA Golden Cross / Death Cross strategy (long-only, equity).

    Entry rule
    ----------
    When ``EMA(fast)`` crosses **above** ``EMA(slow)``, buy a position
    sized as ``stake_pct × available_cash / price`` (whole shares only).

    Exit rule
    ---------
    When ``EMA(fast)`` crosses **below** ``EMA(slow)``, close the full
    position in a single market order.

    Duplicate-entry guard
    ---------------------
    ``if not self.position`` ensures only one open trade at a time.
    A pending order flag (``self.order``) prevents issuing a second order
    on the same bar.

    Parameters
    ----------
    fast_period : int
        Period for the short EMA.  Default: ``50``.
    slow_period : int
        Period for the long EMA.  Default: ``200``.
    stake_pct : float
        Fraction of available cash to deploy per trade.  Default: ``0.95``.
    printlog : bool
        Emit per-event log messages.  Default: ``True``.
    """

    params = (
        ("fast_period", 50),
        ("slow_period", 200),
        ("stake_pct",   0.95),
        ("printlog",    True),
    )

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    def __init__(self) -> None:
        # Convenience reference to close prices
        self.dataclose = self.data.close

        # ── Indicators ────────────────────────────────────────────────────────
        self.fast_ema = bt.indicators.EMA(
            self.dataclose, period=self.params.fast_period
        )
        self.slow_ema = bt.indicators.EMA(
            self.dataclose, period=self.params.slow_period
        )
        # CrossOver: +1.0 on golden cross, -1.0 on death cross, 0.0 otherwise
        self.crossover = bt.indicators.CrossOver(self.fast_ema, self.slow_ema)

        # ── Internal state ────────────────────────────────────────────────────
        self.order:      Optional[bt.Order] = None
        self.buyprice:   Optional[float]    = None
        self.buycomm:    Optional[float]    = None
        self.trade_count: int               = 0

        self.log(
            f"Initialised  EMA({self.params.fast_period}) × "
            f"EMA({self.params.slow_period})  |  "
            f"stake: {self.params.stake_pct * 100:.0f}%"
        )

    # ── Logging helper ────────────────────────────────────────────────────────
    def log(self, text: str, dt=None) -> None:
        """Emit a bar-stamped log line (silenced when ``printlog=False``)."""
        if not self.params.printlog:
            return
        dt = dt or self.data.datetime.date(0)
        logger.info("[%s]  %s", dt.isoformat(), text)

    # ── Backtrader callbacks ──────────────────────────────────────────────────
    def notify_order(self, order: bt.Order) -> None:
        """
        Called every time an order changes status.

        Logs execution details on completion and clears the pending-order
        flag so the next signal can fire.
        """
        if order.status in (order.Submitted, order.Accepted):
            return  # In-flight — nothing to act on yet

        if order.status == order.Completed:
            ex = order.executed
            if order.isbuy():
                self.buyprice = ex.price
                self.buycomm  = ex.comm
                self.log(
                    f"  BUY  EXECUTED  "
                    f"₹{ex.price:>10,.2f}  ×  {ex.size:>6.0f} shares  "
                    f"| value ₹{ex.value:>12,.2f}  | comm ₹{ex.comm:,.2f}"
                )
            else:
                pnl_gross = (ex.price - (self.buyprice or ex.price)) * ex.size
                self.log(
                    f"  SELL EXECUTED  "
                    f"₹{ex.price:>10,.2f}  ×  {ex.size:>6.0f} shares  "
                    f"| value ₹{ex.value:>12,.2f}  | comm ₹{ex.comm:,.2f}  "
                    f"| gross P&L ₹{pnl_gross:>10,.2f}"
                )

        elif order.status in (order.Canceled, order.Margin, order.Rejected):
            self.log(f"  ORDER {order.getstatusname().upper()}")

        self.order = None  # Clear pending-order guard

    def notify_trade(self, trade: bt.Trade) -> None:
        """Log P&L when a round-trip trade is fully closed."""
        if not trade.isclosed:
            return
        self.trade_count += 1
        self.log(
            f"  ── TRADE #{self.trade_count:>3}  CLOSED  "
            f"| gross ₹{trade.pnl:>10,.2f}  "
            f"| net   ₹{trade.pnlcomm:>10,.2f}"
        )

    def next(self) -> None:
        """
        Core signal logic — called once per completed bar.

        Decision tree
        -------------
        1. If an order is already pending → skip (prevent double-orders).
        2. If flat AND golden cross just fired → buy.
        3. If long AND death cross just fired  → close.
        """
        if self.order:
            return

        # ── ENTRY: Golden Cross ────────────────────────────────────────────
        if not self.position and self.crossover > 0:
            cash  = self.broker.get_cash()
            price = self.dataclose[0]
            size  = int((cash * self.params.stake_pct) / price)
            if size > 0:
                self.order = self.buy(size=size)
                self.log(
                    f"▲ GOLDEN CROSS → BUY   ₹{price:,.2f}  ({size} shares)  "
                    f"| EMA{self.params.fast_period}: {self.fast_ema[0]:,.2f}  "
                    f"> EMA{self.params.slow_period}: {self.slow_ema[0]:,.2f}"
                )

        # ── EXIT: Death Cross ──────────────────────────────────────────────
        elif self.position and self.crossover < 0:
            self.order = self.close()
            self.log(
                f"▼ DEATH  CROSS → SELL  ₹{self.dataclose[0]:,.2f}  "
                f"({self.position.size} shares)  "
                f"| EMA{self.params.fast_period}: {self.fast_ema[0]:,.2f}  "
                f"< EMA{self.params.slow_period}: {self.slow_ema[0]:,.2f}"
            )

    def stop(self) -> None:
        """Print final portfolio summary at end of backtest."""
        final_value = self.broker.getvalue()
        self.log(
            f"STRATEGY END  |  Portfolio: ₹{final_value:>12,.2f}  "
            f"|  Trades completed: {self.trade_count}",
            dt=self.data.datetime.date(0),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Backtest runner
# ──────────────────────────────────────────────────────────────────────────────
def run_backtest(
    stock_name: str      = "TCS",
    fast_period: int     = 50,
    slow_period: int     = 200,
    initial_cash: float  = INITIAL_CASH,
    commission: float    = COMMISSION,
    processed_dir: str   = PROCESSED_DATA_DIR,
    printlog: bool       = False,
) -> Dict:
    """
    Set up and execute a single-stock EMA Crossover backtest.

    Creates a ``bt.Cerebro`` engine, loads data, adds the strategy and
    performance analyzers, runs the backtest, and returns a summary dict.

    Parameters
    ----------
    stock_name:
        One of ``'TCS'``, ``'RELIANCE'``, ``'INFOSYS'``.
    fast_period:
        Fast EMA period.
    slow_period:
        Slow EMA period.
    initial_cash:
        Starting capital in INR.
    commission:
        Per-leg commission fraction (e.g. ``0.0005`` = 0.05 %).
    processed_dir:
        Directory containing processed CSVs from Phase 1.
    printlog:
        Pass-through to strategy's ``printlog`` param.

    Returns
    -------
    dict
        Keys: ``stock``, ``strategy``, ``start_value``, ``end_value``,
        ``net_pnl``, ``return_pct``, ``total_trades``, ``won_trades``,
        ``lost_trades``, ``win_rate_pct``, ``sharpe_ratio``,
        ``max_drawdown_pct``.
    """
    cerebro = bt.Cerebro(stdstats=False)

    # Data
    cerebro.adddata(load_bt_feed(stock_name, processed_dir), name=stock_name)

    # Strategy
    cerebro.addstrategy(
        EMACrossoverStrategy,
        fast_period=fast_period,
        slow_period=slow_period,
        printlog=printlog,
    )

    # Broker
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=commission)

    # Analyzers
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(
        bt.analyzers.SharpeRatio,
        _name="sharpe",
        riskfreerate=0.06,     # 6 % — approximate Indian risk-free rate
        annualize=True,
        timeframe=bt.TimeFrame.Days,
    )
    cerebro.addanalyzer(bt.analyzers.DrawDown,  _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.Returns,   _name="returns")

    logger.info(
        "Running EMA Crossover — %s  |  EMA(%d)×EMA(%d)  |  capital: ₹%s",
        stock_name, fast_period, slow_period, f"{initial_cash:,.0f}",
    )

    start_value = cerebro.broker.getvalue()
    results     = cerebro.run()
    end_value   = cerebro.broker.getvalue()
    strat       = results[0]

    # ── Extract analyzer output safely ────────────────────────────────────────
    ta   = strat.analyzers.trades.get_analysis()
    sha  = strat.analyzers.sharpe.get_analysis()
    dda  = strat.analyzers.drawdown.get_analysis()

    total  = ta.get("total",  {}).get("total",  0) or 0
    won    = ta.get("won",    {}).get("total",  0) or 0
    lost   = ta.get("lost",   {}).get("total",  0) or 0

    return {
        "stock":            stock_name,
        "strategy":         f"EMA({fast_period})×EMA({slow_period})",
        "start_value":      start_value,
        "end_value":        end_value,
        "net_pnl":          end_value - start_value,
        "return_pct":       (end_value - start_value) / start_value * 100,
        "total_trades":     total,
        "won_trades":       won,
        "lost_trades":      lost,
        "win_rate_pct":     (won / total * 100) if total else 0.0,
        "sharpe_ratio":     sha.get("sharperatio"),
        "max_drawdown_pct": dda.get("max", {}).get("drawdown"),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Script entry-point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    STOCKS = ["TCS", "RELIANCE", "INFOSYS"]

    print("=" * 68)
    print("  EMA Crossover Strategy  —  Backtest  (EMA-50 × EMA-200)")
    print("=" * 68)

    all_results = []
    for stock in STOCKS:
        try:
            r = run_backtest(
                stock_name=stock,
                fast_period=50,
                slow_period=200,
                initial_cash=INITIAL_CASH,
                printlog=True,
            )
            all_results.append(r)
        except FileNotFoundError as exc:
            logger.error("Skipping %s — %s", stock, exc)

    if not all_results:
        print("No results — ensure processed CSVs exist (run preprocess.py).")
        sys.exit(1)

    # ── Performance summary table ─────────────────────────────────────────────
    print("\n── Performance Summary ─────────────────────────────────────────")
    hdr = (
        f"{'Stock':<12} {'Start ₹':>11} {'End ₹':>11} "
        f"{'Return':>8} {'Trades':>7} {'Win%':>7} "
        f"{'Sharpe':>8} {'MaxDD%':>8}"
    )
    print(hdr)
    print("─" * len(hdr))
    for r in all_results:
        sharpe = (
            f"{r['sharpe_ratio']:.3f}"
            if r["sharpe_ratio"] is not None else "   N/A"
        )
        maxdd = (
            f"{r['max_drawdown_pct']:.2f}%"
            if r["max_drawdown_pct"] is not None else "   N/A"
        )
        print(
            f"{r['stock']:<12} "
            f"₹{r['start_value']:>9,.0f} "
            f"₹{r['end_value']:>9,.0f} "
            f"{r['return_pct']:>+7.2f}% "
            f"{r['total_trades']:>7} "
            f"{r['win_rate_pct']:>6.1f}% "
            f"{sharpe:>8} "
            f"{maxdd:>8}"
        )
    print()
