"""
rsi_strategy.py
===============
Backtrader strategy: RSI mean-reversion with an optional EMA trend filter.

Trading logic
-------------
RSI (Relative Strength Index) oscillates between 0 and 100.  Extreme
readings reveal when a stock is statistically stretched and likely to
revert toward its mean:

* **OVERSOLD** — RSI drops below ``rsi_oversold`` (default 30) →
  the stock has been sold aggressively; a bounce is likely → enter long.

* **OVERBOUGHT** — RSI rises above ``rsi_overbought`` (default 70) →
  the stock has been bought aggressively; a pullback is likely → exit long.

Optional trend filter (``use_trend_filter=True``)
-------------------------------------------------
Buying blindly on every oversold reading can mean catching falling knives
in a bear market.  When the filter is active the strategy only buys if the
closing price is *above* the EMA(200), confirming the stock is still in a
long-term uptrend.  This dramatically reduces false entries.

Behaviour on Indian large-caps
-------------------------------
TCS, Reliance, and Infosys regularly produce RSI-30 entries during broader
NSE corrections while maintaining their long-term uptrends — making this
filter highly relevant for the portfolio.

Defaults
--------
* ``rsi_period``          = 14
* ``rsi_oversold``        = 30
* ``rsi_overbought``      = 70
* ``ema_trend_period``    = 200
* ``use_trend_filter``    = True
* ``stake_pct``           = 0.95

How it connects to Phase 4
--------------------------
The ``run_backtest()`` function returns a standardised result ``dict``
matching the schema of ``ema_crossover.run_backtest()``, so the Phase 4
engine can call both files uniformly.

Usage
-----
::

    python src/strategies/rsi_strategy.py
    from src.strategies.rsi_strategy import RSIStrategy, run_backtest
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Dict, Optional

import backtrader as bt
import pandas as pd

# ── Pandas 2.0 / 3.0 compatibility shim ──────────────────────────────────────
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
# Constants
# ──────────────────────────────────────────────────────────────────────────────
PROCESSED_DATA_DIR: str = os.path.join("data", "processed")
INITIAL_CASH: float     = 500_000.0
COMMISSION: float       = 0.0005


# ──────────────────────────────────────────────────────────────────────────────
# Data feed loader  (shared with other strategy files)
# ──────────────────────────────────────────────────────────────────────────────
def load_bt_feed(
    stock_name: str,
    processed_dir: str = PROCESSED_DATA_DIR,
) -> bt.feeds.PandasData:
    """
    Load a Phase-1 processed CSV as a Backtrader ``PandasData`` feed.

    Parameters
    ----------
    stock_name:
        Display name (e.g. ``'TCS'``).
    processed_dir:
        Directory containing processed CSVs.

    Returns
    -------
    bt.feeds.PandasData

    Raises
    ------
    FileNotFoundError
        If the processed CSV does not exist.
    """
    filepath = os.path.join(processed_dir, f"{stock_name.upper()}_processed.csv")
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Processed data not found: {filepath}\n"
            "Run src/data/preprocess.py first."
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
class RSIStrategy(bt.Strategy):
    """
    RSI mean-reversion strategy with an optional EMA-200 trend filter.

    Entry rule
    ----------
    ``RSI(period) < rsi_oversold``
    **AND** (if ``use_trend_filter``) ``close > EMA(ema_trend_period)``

    Exit rule
    ---------
    ``RSI(period) > rsi_overbought``

    Why the trend filter matters
    ----------------------------
    Without it, the strategy enters on every oversold dip — including
    during prolonged bear markets where "oversold gets more oversold".
    The EMA-200 acts as a regime filter: only buy dips in uptrending
    markets.

    Parameters
    ----------
    rsi_period : int
        RSI look-back.  Default: ``14``.
    rsi_oversold : float
        Buy threshold.  Default: ``30.0``.
    rsi_overbought : float
        Sell threshold.  Default: ``70.0``.
    ema_trend_period : int
        Trend-filter EMA period.  Default: ``200``.
    use_trend_filter : bool
        Activate EMA trend filter.  Default: ``True``.
    stake_pct : float
        Fraction of cash to deploy.  Default: ``0.95``.
    printlog : bool
        Emit per-event log lines.  Default: ``True``.
    """

    params = (
        ("rsi_period",       14),
        ("rsi_oversold",     30.0),
        ("rsi_overbought",   70.0),
        ("ema_trend_period", 200),
        ("use_trend_filter", True),
        ("stake_pct",        0.95),
        ("printlog",         True),
    )

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    def __init__(self) -> None:
        self.dataclose = self.data.close

        # ── Indicators ────────────────────────────────────────────────────────
        self.rsi = bt.indicators.RSI(
            self.dataclose,
            period=self.params.rsi_period,
            safediv=True,
        )
        # Trend filter EMA — always computed; only *used* when flag is set
        self.trend_ema = bt.indicators.EMA(
            self.dataclose, period=self.params.ema_trend_period
        )

        # ── State ─────────────────────────────────────────────────────────────
        self.order:       Optional[bt.Order] = None
        self.buyprice:    Optional[float]    = None
        self.buycomm:     Optional[float]    = None
        self.trade_count: int                = 0

        filter_label = (
            f"EMA({self.params.ema_trend_period}) trend filter ON"
            if self.params.use_trend_filter
            else "trend filter OFF"
        )
        self.log(
            f"Initialised  RSI({self.params.rsi_period})  "
            f"[oversold<{self.params.rsi_oversold} / "
            f"overbought>{self.params.rsi_overbought}]  |  {filter_label}"
        )

    # ── Logging helper ────────────────────────────────────────────────────────
    def log(self, text: str, dt=None) -> None:
        """Emit a bar-stamped log line."""
        if not self.params.printlog:
            return
        dt = dt or self.data.datetime.date(0)
        logger.info("[%s]  %s", dt.isoformat(), text)

    # ── Backtrader callbacks ──────────────────────────────────────────────────
    def notify_order(self, order: bt.Order) -> None:
        """Log order execution and clear the pending-order guard."""
        if order.status in (order.Submitted, order.Accepted):
            return

        if order.status == order.Completed:
            ex = order.executed
            if order.isbuy():
                self.buyprice = ex.price
                self.buycomm  = ex.comm
                self.log(
                    f"  BUY  EXECUTED  "
                    f"₹{ex.price:>10,.2f}  ×  {ex.size:>6.0f} shares  "
                    f"| value ₹{ex.value:>12,.2f}  | comm ₹{ex.comm:,.2f}  "
                    f"| RSI was: {self.rsi[0]:.2f}"
                )
            else:
                pnl_gross = (ex.price - (self.buyprice or ex.price)) * ex.size
                self.log(
                    f"  SELL EXECUTED  "
                    f"₹{ex.price:>10,.2f}  ×  {ex.size:>6.0f} shares  "
                    f"| value ₹{ex.value:>12,.2f}  | comm ₹{ex.comm:,.2f}  "
                    f"| gross P&L ₹{pnl_gross:>10,.2f}  "
                    f"| RSI was: {self.rsi[0]:.2f}"
                )

        elif order.status in (order.Canceled, order.Margin, order.Rejected):
            self.log(f"  ORDER {order.getstatusname().upper()}")

        self.order = None

    def notify_trade(self, trade: bt.Trade) -> None:
        """Log net P&L when a round-trip trade closes."""
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
        Per-bar signal logic.

        Decision tree
        -------------
        1. Pending order → skip.
        2. Flat:
           * RSI < oversold threshold AND (trend filter passes or is off) → BUY.
        3. Long:
           * RSI > overbought threshold → SELL.
        """
        if self.order:
            return

        current_rsi = self.rsi[0]

        # ── ENTRY ─────────────────────────────────────────────────────────────
        if not self.position:
            rsi_signal = current_rsi < self.params.rsi_oversold

            # Trend filter: price must be above the long-term EMA
            if self.params.use_trend_filter:
                trend_ok = self.dataclose[0] > self.trend_ema[0]
            else:
                trend_ok = True

            if rsi_signal and trend_ok:
                cash  = self.broker.get_cash()
                price = self.dataclose[0]
                size  = int((cash * self.params.stake_pct) / price)
                if size > 0:
                    self.order = self.buy(size=size)
                    trend_note = (
                        f"EMA{self.params.ema_trend_period}: "
                        f"{self.trend_ema[0]:,.2f} ✓"
                        if self.params.use_trend_filter
                        else "trend filter OFF"
                    )
                    self.log(
                        f"▲ RSI OVERSOLD  → BUY   ₹{price:,.2f}  ({size} shares)  "
                        f"| RSI: {current_rsi:.2f} < {self.params.rsi_oversold}  "
                        f"| {trend_note}"
                    )
            elif rsi_signal and not trend_ok:
                # Signal fired but trend filter blocked — useful to track
                self.log(
                    f"  RSI oversold ({current_rsi:.2f}) BUT price "
                    f"₹{self.dataclose[0]:,.2f} < "
                    f"EMA{self.params.ema_trend_period} "
                    f"{self.trend_ema[0]:,.2f} — FILTERED OUT"
                )

        # ── EXIT ──────────────────────────────────────────────────────────────
        elif self.position and current_rsi > self.params.rsi_overbought:
            self.order = self.close()
            self.log(
                f"▼ RSI OVERBOUGHT → SELL  ₹{self.dataclose[0]:,.2f}  "
                f"({self.position.size} shares)  "
                f"| RSI: {current_rsi:.2f} > {self.params.rsi_overbought}"
            )

    def stop(self) -> None:
        """Summarise final state at backtest end."""
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
    stock_name: str         = "TCS",
    rsi_period: int         = 14,
    rsi_oversold: float     = 30.0,
    rsi_overbought: float   = 70.0,
    ema_trend_period: int   = 200,
    use_trend_filter: bool  = True,
    initial_cash: float     = INITIAL_CASH,
    commission: float       = COMMISSION,
    processed_dir: str      = PROCESSED_DATA_DIR,
    printlog: bool          = False,
) -> Dict:
    """
    Configure and execute a single-stock RSI strategy backtest.

    Parameters
    ----------
    stock_name:
        Equity name (``'TCS'``, ``'RELIANCE'``, ``'INFOSYS'``).
    rsi_period:
        RSI look-back window.
    rsi_oversold:
        Buy trigger level.
    rsi_overbought:
        Sell trigger level.
    ema_trend_period:
        Trend-filter EMA period.
    use_trend_filter:
        Activate the EMA trend confirmation gate.
    initial_cash:
        Starting portfolio value in INR.
    commission:
        Per-leg commission rate.
    processed_dir:
        Path to processed CSV files.
    printlog:
        Enable per-event logging in the strategy.

    Returns
    -------
    dict
        Standardised result dict (same schema as ``ema_crossover.run_backtest``).
    """
    cerebro = bt.Cerebro(stdstats=False)

    cerebro.adddata(load_bt_feed(stock_name, processed_dir), name=stock_name)
    cerebro.addstrategy(
        RSIStrategy,
        rsi_period=rsi_period,
        rsi_oversold=rsi_oversold,
        rsi_overbought=rsi_overbought,
        ema_trend_period=ema_trend_period,
        use_trend_filter=use_trend_filter,
        printlog=printlog,
    )
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=commission)

    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(
        bt.analyzers.SharpeRatio,
        _name="sharpe",
        riskfreerate=0.06,
        annualize=True,
        timeframe=bt.TimeFrame.Days,
    )
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.Returns,  _name="returns")

    logger.info(
        "Running RSI Strategy — %s | RSI(%d) oversold=%.0f overbought=%.0f trend_filter=%s",
        stock_name, rsi_period, rsi_oversold, rsi_overbought, use_trend_filter,
    )

    start_value = cerebro.broker.getvalue()
    results     = cerebro.run()
    end_value   = cerebro.broker.getvalue()
    strat       = results[0]

    ta  = strat.analyzers.trades.get_analysis()
    sha = strat.analyzers.sharpe.get_analysis()
    dda = strat.analyzers.drawdown.get_analysis()

    total = ta.get("total", {}).get("total", 0) or 0
    won   = ta.get("won",   {}).get("total", 0) or 0
    lost  = ta.get("lost",  {}).get("total", 0) or 0

    return {
        "stock":            stock_name,
        "strategy":         f"RSI({rsi_period}) <{rsi_oversold}/>{rsi_overbought}",
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
    print("  RSI Mean-Reversion Strategy  —  Backtest  (RSI-14, filter ON)")
    print("=" * 68)

    # Run with trend filter ON vs OFF to show the difference
    configs = [
        {"label": "RSI + Trend Filter",    "use_trend_filter": True},
        {"label": "RSI (no trend filter)", "use_trend_filter": False},
    ]

    for cfg in configs:
        print(f"\n── {cfg['label']} ─────────────────────────────────────────")
        hdr = (
            f"{'Stock':<12} {'Start ₹':>11} {'End ₹':>11} "
            f"{'Return':>8} {'Trades':>7} {'Win%':>7} "
            f"{'Sharpe':>8} {'MaxDD%':>8}"
        )
        print(hdr)
        print("─" * len(hdr))

        for stock in STOCKS:
            try:
                r = run_backtest(
                    stock_name=stock,
                    use_trend_filter=cfg["use_trend_filter"],
                    printlog=False,
                )
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
            except FileNotFoundError as exc:
                logger.error("Skipping %s — %s", stock, exc)
    print()
