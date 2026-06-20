"""
combined_strategy.py
====================
Backtrader strategy: Multi-indicator confluence (EMA + RSI + ATR + Bollinger).

Philosophy
----------
Individual indicators produce false signals.  This strategy requires
*multiple indicators to agree* before acting — a technique called
**confluence** in professional trading:

    "A trade is only as strong as the number of independent signals
     confirming it."

Entry conditions  (ALL must be true simultaneously)
-----------------------------------------------------
1. **EMA trend** — ``EMA(fast) > EMA(slow)``
   Ensures we are buying in an uptrending market, not a falling one.

2. **RSI not overbought** — ``RSI(rsi_period) < rsi_entry_cap``
   Avoids chasing a move that is already exhausted.  Default cap: 60
   (more conservative than the classic 70 to give extra room to run).

3. **Bollinger Band proximity** — ``close ≤ bb_middle``
   Price is at or below the 20-SMA — a mild mean-reversion entry
   within the trend.  We buy dips, not breakouts.

4. **ATR volatility gate** — ``ATR(atr_period) > atr_min_threshold``
   Filters out dead, illiquid sessions where the spread cost erodes
   any potential profit.  Default: ``atr_min = 0`` (off by default;
   set to e.g. 20 for ₹-denominated NSE stocks).

Exit conditions  (first to trigger wins)
-----------------------------------------
A. **RSI overbought** — ``RSI > rsi_exit_level`` (default 70)
   Take profit as momentum peaks.

B. **ATR trailing stop** — ``close < entry_price − atr_stop_mult × ATR``
   Volatility-adjusted stop-loss.  Default multiplier: 2.0.
   Adapts to current market conditions — wider in volatile markets,
   tighter in calm ones.

C. **EMA trend breakdown** — ``EMA(fast) < EMA(slow)``
   The primary trend has reversed; exit regardless of RSI or stop.

Why this is "resume-worthy"
----------------------------
* Combines *trend-following* (EMA), *momentum oscillator* (RSI),
  *volatility sizing* (ATR), and *band context* (Bollinger) into one
  coherent thesis rather than stacking unrelated filters.
* Every parameter is named and configurable — easy to grid-search in
  Phase 4's optimisation engine.
* Risk is managed at two levels: the ATR trailing stop (micro) and the
  EMA breakdown exit (macro).

Defaults
--------
* ``fast_ema_period``    = 50
* ``slow_ema_period``    = 200
* ``rsi_period``         = 14
* ``rsi_entry_cap``      = 60   (don't buy when already overbought)
* ``rsi_exit_level``     = 70   (exit at overbought)
* ``bb_period``          = 20
* ``bb_dev``             = 2.0
* ``atr_period``         = 14
* ``atr_stop_mult``      = 2.0  (stop = entry − 2 × ATR)
* ``atr_min_threshold``  = 0    (disabled; set > 0 to filter low-vol days)
* ``stake_pct``          = 0.90

Usage
-----
::

    python src/strategies/combined_strategy.py
    from src.strategies.combined_strategy import CombinedStrategy, run_backtest
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
# Data feed loader
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
        Display name, e.g. ``'TCS'``, ``'RELIANCE'``, ``'INFOSYS'``.
    processed_dir:
        Directory containing processed CSV files.

    Returns
    -------
    bt.feeds.PandasData

    Raises
    ------
    FileNotFoundError
        If the processed CSV is missing (run ``preprocess.py`` first).
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
class CombinedStrategy(bt.Strategy):
    """
    Multi-indicator confluence strategy: EMA trend + RSI momentum
    + Bollinger proximity + ATR trailing stop.

    Parameters
    ----------
    fast_ema_period : int
        Fast EMA for trend direction.  Default: ``50``.
    slow_ema_period : int
        Slow EMA for trend direction.  Default: ``200``.
    rsi_period : int
        RSI look-back period.  Default: ``14``.
    rsi_entry_cap : float
        Maximum RSI allowed at entry (don't buy already-overbought).
        Default: ``60``.
    rsi_exit_level : float
        RSI overbought exit threshold.  Default: ``70``.
    bb_period : int
        Bollinger Band SMA period.  Default: ``20``.
    bb_dev : float
        Bollinger Band standard deviation multiplier.  Default: ``2.0``.
    atr_period : int
        ATR look-back for the trailing stop.  Default: ``14``.
    atr_stop_mult : float
        ATR multiplier for the stop-loss distance.  Default: ``2.0``.
    atr_min_threshold : float
        Minimum ATR required to enter a trade (volatility gate).
        Default: ``0`` (disabled).
    stake_pct : float
        Fraction of cash to deploy per trade.  Default: ``0.90``.
    printlog : bool
        Emit per-event log messages.  Default: ``True``.
    """

    params = (
        ("fast_ema_period",   50),
        ("slow_ema_period",   200),
        ("rsi_period",        14),
        ("rsi_entry_cap",     60.0),
        ("rsi_exit_level",    70.0),
        ("bb_period",         20),
        ("bb_dev",            2.0),
        ("atr_period",        14),
        ("atr_stop_mult",     2.0),
        ("atr_min_threshold", 0.0),
        ("stake_pct",         0.90),
        ("printlog",          True),
    )

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    def __init__(self) -> None:
        self.dataclose = self.data.close

        # ── Trend indicators ──────────────────────────────────────────────────
        self.fast_ema = bt.indicators.EMA(
            self.dataclose, period=self.params.fast_ema_period
        )
        self.slow_ema = bt.indicators.EMA(
            self.dataclose, period=self.params.slow_ema_period
        )

        # ── Momentum indicator ────────────────────────────────────────────────
        self.rsi = bt.indicators.RSI(
            self.dataclose,
            period=self.params.rsi_period,
            safediv=True,
        )

        # ── Bollinger Bands ───────────────────────────────────────────────────
        self.bb = bt.indicators.BollingerBands(
            self.dataclose,
            period=self.params.bb_period,
            devfactor=self.params.bb_dev,
        )
        # bt.indicators.BollingerBands exposes: .mid, .top, .bot

        # ── Volatility / stop-loss ────────────────────────────────────────────
        self.atr = bt.indicators.ATR(
            self.data,
            period=self.params.atr_period,
        )

        # ── Internal state ────────────────────────────────────────────────────
        self.order:        Optional[bt.Order] = None
        self.buyprice:     Optional[float]    = None
        self.buycomm:      Optional[float]    = None
        self.trailing_stop: Optional[float]   = None
        self.trade_count:  int                = 0

        self.log(
            f"Initialised  EMA({self.params.fast_ema_period})×"
            f"EMA({self.params.slow_ema_period})  |  "
            f"RSI({self.params.rsi_period}) cap={self.params.rsi_entry_cap}  |  "
            f"BB({self.params.bb_period},{self.params.bb_dev}σ)  |  "
            f"ATR({self.params.atr_period}) stop={self.params.atr_stop_mult}×"
        )

    # ── Logging helper ────────────────────────────────────────────────────────
    def log(self, text: str, dt=None) -> None:
        """Emit a bar-stamped log line (silenced when ``printlog=False``)."""
        if not self.params.printlog:
            return
        dt = dt or self.data.datetime.date(0)
        logger.info("[%s]  %s", dt.isoformat(), text)

    # ── Internal helpers ──────────────────────────────────────────────────────
    def _check_entry_conditions(self) -> tuple[bool, str]:
        """
        Evaluate all four entry conditions and return a pass/fail
        verdict with a human-readable explanation.

        Returns
        -------
        (bool, str)
            ``(True, reason)`` if all conditions pass;
            ``(False, reason)`` listing the first failing condition.
        """
        close = self.dataclose[0]

        # 1. EMA trend confirmation
        if self.fast_ema[0] <= self.slow_ema[0]:
            return False, (
                f"EMA trend BEARISH  "
                f"(fast {self.fast_ema[0]:,.2f} ≤ slow {self.slow_ema[0]:,.2f})"
            )

        # 2. RSI not overbought at entry
        if self.rsi[0] >= self.params.rsi_entry_cap:
            return False, (
                f"RSI too high  "
                f"({self.rsi[0]:.2f} ≥ cap {self.params.rsi_entry_cap})"
            )

        # 3. Price at or below Bollinger mid-band (buying a dip in the trend)
        if close > self.bb.mid[0]:
            return False, (
                f"Price above BB mid  "
                f"(₹{close:,.2f} > BB_mid ₹{self.bb.mid[0]:,.2f})"
            )

        # 4. ATR volatility gate (skip if threshold = 0, i.e. disabled)
        if self.params.atr_min_threshold > 0:
            if self.atr[0] < self.params.atr_min_threshold:
                return False, (
                    f"ATR too low  "
                    f"({self.atr[0]:.2f} < min {self.params.atr_min_threshold})"
                )

        return True, (
            f"ALL conditions met  "
            f"| EMA fast>{self.slow_ema[0]:,.2f}  "
            f"| RSI {self.rsi[0]:.2f}<{self.params.rsi_entry_cap}  "
            f"| price≤BB_mid {self.bb.mid[0]:,.2f}  "
            f"| ATR {self.atr[0]:.2f}"
        )

    def _check_exit_conditions(self) -> tuple[bool, str]:
        """
        Evaluate all three exit conditions.

        Returns
        -------
        (bool, str)
            ``(True, exit_reason)`` when any condition triggers an exit.
        """
        close = self.dataclose[0]

        # A. RSI overbought — take profit
        if self.rsi[0] > self.params.rsi_exit_level:
            return True, (
                f"RSI OVERBOUGHT EXIT  "
                f"RSI={self.rsi[0]:.2f} > {self.params.rsi_exit_level}"
            )

        # B. ATR trailing stop — capital protection
        if self.trailing_stop is not None and close < self.trailing_stop:
            return True, (
                f"ATR STOP HIT  "
                f"₹{close:,.2f} < stop ₹{self.trailing_stop:,.2f}  "
                f"(entry ₹{self.buyprice:,.2f}  "
                f"− {self.params.atr_stop_mult}×ATR {self.atr[0]:.2f})"
            )

        # C. EMA trend breakdown — macro exit
        if self.fast_ema[0] < self.slow_ema[0]:
            return True, (
                f"EMA BREAKDOWN EXIT  "
                f"fast {self.fast_ema[0]:,.2f} < slow {self.slow_ema[0]:,.2f}"
            )

        return False, ""

    def _update_trailing_stop(self) -> None:
        """
        Recalculate the ATR-based trailing stop after each bar.

        The stop is *only ever moved up* (ratchet mechanism) — it never
        moves down, so profits are progressively locked in as price rises.
        """
        if self.buyprice is None:
            return
        new_stop = self.dataclose[0] - (self.params.atr_stop_mult * self.atr[0])
        if self.trailing_stop is None or new_stop > self.trailing_stop:
            self.trailing_stop = new_stop

    # ── Backtrader callbacks ──────────────────────────────────────────────────
    def notify_order(self, order: bt.Order) -> None:
        """Log execution details and clear the pending-order guard."""
        if order.status in (order.Submitted, order.Accepted):
            return

        if order.status == order.Completed:
            ex = order.executed
            if order.isbuy():
                self.buyprice     = ex.price
                self.buycomm      = ex.comm
                # Set initial ATR stop immediately at entry
                self.trailing_stop = (
                    ex.price - self.params.atr_stop_mult * self.atr[0]
                )
                self.log(
                    f"  BUY  EXECUTED  "
                    f"₹{ex.price:>10,.2f}  ×  {ex.size:>6.0f} shares  "
                    f"| value ₹{ex.value:>12,.2f}  "
                    f"| comm ₹{ex.comm:,.2f}  "
                    f"| initial stop ₹{self.trailing_stop:,.2f}"
                )
            else:
                pnl_gross = (ex.price - (self.buyprice or ex.price)) * ex.size
                self.log(
                    f"  SELL EXECUTED  "
                    f"₹{ex.price:>10,.2f}  ×  {ex.size:>6.0f} shares  "
                    f"| value ₹{ex.value:>12,.2f}  "
                    f"| comm ₹{ex.comm:,.2f}  "
                    f"| gross P&L ₹{pnl_gross:>10,.2f}"
                )
                # Reset state after exit
                self.buyprice      = None
                self.trailing_stop = None

        elif order.status in (order.Canceled, order.Margin, order.Rejected):
            self.log(f"  ORDER {order.getstatusname().upper()}")

        self.order = None

    def notify_trade(self, trade: bt.Trade) -> None:
        """Log round-trip P&L when a trade fully closes."""
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
        Per-bar confluence decision engine.

        Workflow
        --------
        1. Skip if a pending order is in flight.
        2. If **flat**: evaluate all four entry conditions via
           ``_check_entry_conditions()``.  Buy only on full confluence.
        3. If **long**: update the ratcheting ATR trailing stop, then
           evaluate exit conditions via ``_check_exit_conditions()``.
        """
        if self.order:
            return

        # ── ENTRY ─────────────────────────────────────────────────────────────
        if not self.position:
            entry_ok, entry_reason = self._check_entry_conditions()

            if entry_ok:
                cash  = self.broker.get_cash()
                price = self.dataclose[0]
                size  = int((cash * self.params.stake_pct) / price)

                if size > 0:
                    self.order = self.buy(size=size)
                    self.log(
                        f"▲ CONFLUENCE BUY  ₹{price:,.2f}  ({size} shares)  "
                        f"| {entry_reason}"
                    )

        # ── HOLD: update ratchet stop ─────────────────────────────────────────
        else:
            self._update_trailing_stop()

            exit_triggered, exit_reason = self._check_exit_conditions()
            if exit_triggered:
                self.order = self.close()
                self.log(
                    f"▼ EXIT  ₹{self.dataclose[0]:,.2f}  "
                    f"({self.position.size} shares)  | {exit_reason}"
                )

    def stop(self) -> None:
        """Emit a final portfolio summary at the end of the backtest."""
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
    stock_name: str          = "TCS",
    fast_ema_period: int     = 50,
    slow_ema_period: int     = 200,
    rsi_period: int          = 14,
    rsi_entry_cap: float     = 60.0,
    rsi_exit_level: float    = 70.0,
    bb_period: int           = 20,
    bb_dev: float            = 2.0,
    atr_period: int          = 14,
    atr_stop_mult: float     = 2.0,
    atr_min_threshold: float = 0.0,
    initial_cash: float      = INITIAL_CASH,
    commission: float        = COMMISSION,
    processed_dir: str       = PROCESSED_DATA_DIR,
    printlog: bool           = False,
) -> Dict:
    """
    Configure and execute a single-stock Combined Strategy backtest.

    All parameters mirror the corresponding ``CombinedStrategy.params``
    entries, making this function suitable for parameter grid-searches
    in the Phase 4 optimisation engine.

    Parameters
    ----------
    stock_name:
        Equity name (``'TCS'``, ``'RELIANCE'``, ``'INFOSYS'``).
    fast_ema_period / slow_ema_period:
        EMA periods for trend direction.
    rsi_period:
        RSI look-back window.
    rsi_entry_cap:
        RSI must be *below* this to trigger an entry.
    rsi_exit_level:
        RSI must exceed this to trigger a profit-take exit.
    bb_period / bb_dev:
        Bollinger Band configuration.
    atr_period:
        ATR smoothing period.
    atr_stop_mult:
        ATR multiplier for the trailing stop distance.
    atr_min_threshold:
        Minimum ATR required for entry (``0`` = disabled).
    initial_cash:
        Starting capital in INR.
    commission:
        Per-leg commission fraction.
    processed_dir:
        Path to Phase-1 processed CSV files.
    printlog:
        Enable per-event logging.

    Returns
    -------
    dict
        Standardised result dict matching the schema of
        ``ema_crossover.run_backtest`` and ``rsi_strategy.run_backtest``.
    """
    cerebro = bt.Cerebro(stdstats=False)

    cerebro.adddata(load_bt_feed(stock_name, processed_dir), name=stock_name)
    cerebro.addstrategy(
        CombinedStrategy,
        fast_ema_period   = fast_ema_period,
        slow_ema_period   = slow_ema_period,
        rsi_period        = rsi_period,
        rsi_entry_cap     = rsi_entry_cap,
        rsi_exit_level    = rsi_exit_level,
        bb_period         = bb_period,
        bb_dev            = bb_dev,
        atr_period        = atr_period,
        atr_stop_mult     = atr_stop_mult,
        atr_min_threshold = atr_min_threshold,
        printlog          = printlog,
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
        "Running Combined Strategy — %s  |  EMA(%d×%d)  RSI(%d) cap=%.0f  "
        "BB(%d) ATR(%d) stop=%.1f×",
        stock_name,
        fast_ema_period, slow_ema_period,
        rsi_period, rsi_entry_cap,
        bb_period, atr_period, atr_stop_mult,
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
        "strategy":         (
            f"Combined EMA({fast_ema_period}×{slow_ema_period})"
            f"+RSI({rsi_period})+BB({bb_period})+ATR({atr_period})"
        ),
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
    print("  Combined Strategy  —  Backtest")
    print("  EMA(50×200) + RSI(14) + BB(20) + ATR(14) trailing stop")
    print("=" * 68)

    print("\n── Per-trade log: TCS (printlog=True) ─────────────────────────")
    try:
        run_backtest("TCS", printlog=True)
    except FileNotFoundError as exc:
        logger.error("%s", exc)

    print("\n── All-stock performance summary ───────────────────────────────")
    hdr = (
        f"{'Stock':<12} {'Start ₹':>11} {'End ₹':>11} "
        f"{'Return':>8} {'Trades':>7} {'Win%':>7} "
        f"{'Sharpe':>8} {'MaxDD%':>8}"
    )
    print(hdr)
    print("─" * len(hdr))

    for stock in STOCKS:
        try:
            r = run_backtest(stock_name=stock, printlog=False)
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

    # ── Strategy comparison across all three strategies ───────────────────────
    print("\n── Head-to-head: all 3 strategies × all 3 stocks ──────────────")
    # Ensure project root is on sys.path when running this file directly
    _project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    from src.strategies.ema_crossover import run_backtest as run_ema
    from src.strategies.rsi_strategy  import run_backtest as run_rsi

    col_w = 14
    strategies = [
        ("EMA Crossover", run_ema,    {"fast_period": 50, "slow_period": 200}),
        ("RSI (filter)",  run_rsi,    {"use_trend_filter": True}),
        ("Combined",      run_backtest, {}),
    ]

    print(f"\n{'':>20}", end="")
    for s_name, _, _ in strategies:
        print(f"  {s_name:^{col_w}}", end="")
    print()
    print(f"{'Stock':<12}{'Metric':<8}", end="")
    for _ in strategies:
        print(f"  {'Return':>6}  {'Trades':>6}", end="")
    print()
    print("─" * (20 + len(strategies) * (col_w + 4)))

    for stock in STOCKS:
        row_return = f"{stock:<12}{'Ret%':<8}"
        row_trades = f"{'':^12}{'#trd':<8}"
        for _, runner, kwargs in strategies:
            try:
                r = runner(stock_name=stock, printlog=False, **kwargs)
                row_return += f"  {r['return_pct']:>+6.1f}%  "
                row_trades += f"  {r['total_trades']:>6}    "
            except FileNotFoundError:
                row_return += f"  {'N/A':>7}  "
                row_trades += f"  {'N/A':>6}    "
        print(row_return)
        print(row_trades)
        print()
