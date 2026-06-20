"""
dashboard.py
============
Interactive Streamlit dashboard for the Algorithmic Trading Strategy
Backtester — Phase 9 (final layer).

What this file does
--------------------
Wires together every previous phase into one interactive application:

* **Phase 1/2** (data + indicators) — stock summary, live EMA/RSI/ATR/BB
  snapshot, and a rule-based BUY/HOLD/SELL signal for the selected strategy.
* **Phase 3/8** (strategies + optimisation engine) — runs a single-stock
  backtest via ``evaluate_params()`` (Phase 8's strategy-agnostic engine,
  reused here because — unlike Phase 3's plain ``run_backtest()`` — it
  reconstructs the full equity curve and trade log needed for charts).
* **Phase 4** (risk) — a "Sample Trade Setup" card showing the stop-loss,
  take-profit, and position size Phase 4's risk engine would recommend
  for the current signal.
* **Phase 5** (portfolio) — multi-stock allocation + rebalancing mode.
* **Phase 6** (analytics) — the full ``PerformanceReport`` / ``DrawdownResult``
  metric suite, computed identically to every other report in this project.
* **Phase 7** (visualization) — the dark-themed equity curve, drawdown
  tear-sheet, and monthly heatmap charts, rendered live via ``st.pyplot()``.
* **Phase 8** (optimisation) — on-demand grid search and walk-forward
  validation, with a Quick/Standard search-size toggle so the dashboard
  stays responsive.

Two design decisions worth calling out
----------------------------------------
1. **Why ``evaluate_params()`` instead of Phase 3's ``run_backtest()``?**
   The latter returns only flat summary stats; this dashboard needs the
   equity curve (for charts) and trade log (for the trade table), both of
   which only ``evaluate_params()`` exposes.
2. **One chart is rendered via a saved PNG, not a live figure.**
   Every Phase 7 plotting method returns a *live* (unclosed) Matplotlib
   figure suitable for ``st.pyplot()`` — except
   ``TradeAnalyzer.plot_dashboard()``, which always calls ``plt.close()``
   internally. Rather than patch a previously-delivered analytics file
   for one chart, this dashboard saves that one figure to PNG and displays
   it with ``st.image()`` — a deliberate, documented exception, not an
   oversight.

Streamlit rerun model
-----------------------
Streamlit reruns this entire script on every widget interaction. Expensive
results (backtests, grid search, walk-forward) are therefore stored in
``st.session_state`` and only recomputed when their "Run" button is
clicked — not on every sidebar tweak. Cheap, pure functions (price loading,
indicator computation) are wrapped in ``@st.cache_data`` instead.

Usage
-----
Not run directly — imported and invoked by ``main.py``::

    streamlit run main.py
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ── Project-root bootstrap ──────────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

if not hasattr(pd.DataFrame, "iteritems"):
    pd.DataFrame.iteritems = pd.DataFrame.items

# ── Cross-phase imports ──────────────────────────────────────────────────────
from src.indicators.moving_average import add_moving_averages
from src.indicators.rsi import add_rsi
from src.indicators.atr import add_atr
from src.indicators.bollinger import add_bollinger_bands

from src.risk.stop_loss import StopLossManager, StopMode
from src.risk.take_profit import TakeProfitManager, TPMode
from src.risk.position_sizing import build_trade_setup, TradeSetup

from src.portfolio.allocation import PortfolioAllocator, AllocationScheme
from src.portfolio.rebalance import simulate_portfolio, RebalanceFrequency, RebalanceSimResult

from src.analytics.performance_metrics import PerformanceAnalyzer, PerformanceReport
from src.analytics.drawdown import DrawdownAnalyzer, DrawdownResult

from src.visualization.equity_curve import EquityCurvePlotter
from src.visualization.drawdown_plot import DrawdownPlotter
from src.visualization.heatmap import ReturnsHeatmapPlotter

from src.optimization.grid_search import (
    DEFAULT_COMMISSION,
    DEFAULT_INITIAL_CASH,
    DEFAULT_PARAM_GRIDS,
    PROCESSED_DIR,
    GridSearchEngine,
    evaluate_params,
    get_strategy_class,
    load_price_data,
)
from src.optimization.walk_forward import WalkForwardEngine, WalkForwardResult

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
STOCKS: List[str] = ["TCS", "RELIANCE", "INFOSYS"]

STRATEGY_LABELS: Dict[str, str] = {
    "ema_crossover":     "EMA Crossover",
    "rsi_strategy":       "RSI Mean-Reversion",
    "combined_strategy":  "Combined (EMA + RSI + BB + ATR)",
}

#: Small search spaces for the dashboard's "Quick" optimisation mode —
#: full DEFAULT_PARAM_GRIDS (Phase 8) is used for "Standard" mode.
QUICK_PARAM_GRIDS: Dict[str, Dict[str, List[Any]]] = {
    "ema_crossover":     {"fast_period": [20, 50], "slow_period": [100, 200]},
    "rsi_strategy":       {"rsi_oversold": [25, 30], "rsi_overbought": [70, 75]},
    "combined_strategy":  {"fast_ema_period": [20, 50], "slow_ema_period": [150, 200]},
}

#: Approximate NSE Indian Securities Transaction Tax on equity delivery
#: trades — a simplified, single flat figure (real STT differs slightly
#: by buy/sell leg and segment, but 0.1% is the standard reference value
#: quoted for delivery-based equity trades).
DEFAULT_STT_PCT: float = 0.001

THEME = {
    "bg": "#0d1117", "panel": "#161b22", "text": "#e6edf3",
    "muted": "#8b949e", "green": "#3fb950", "red": "#f85149",
    "blue": "#58a6ff", "border": "#30363d",
}


# ──────────────────────────────────────────────────────────────────────────────
# Result bundles
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class SingleStockResult:
    """Everything the dashboard needs to render one single-stock backtest."""
    stock:            str
    strategy_name:     str
    params:            Dict[str, Any]
    equity_curve:      pd.Series
    buy_hold_curve:    pd.Series
    trade_log:         pd.DataFrame
    performance:       PerformanceReport
    benchmark_perf:    PerformanceReport
    drawdown_result:   DrawdownResult
    raw_metrics:        Dict[str, Any]
    cost_config:        "TransactionCostConfig"


@dataclass
class TransactionCostConfig:
    """Simplified Indian-equity transaction cost simulation inputs."""
    brokerage_pct: float = 0.0003   # 0.03% — typical discount-broker delivery rate
    stt_pct:       float = DEFAULT_STT_PCT
    slippage_pct:  float = 0.0005

    @property
    def total_commission_pct(self) -> float:
        """Brokerage + STT combined into Backtrader's single commission slot."""
        return self.brokerage_pct + self.stt_pct


# ──────────────────────────────────────────────────────────────────────────────
# Cached, pure data-loading helpers
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _load_price_data_cached(
    stock: str, start_date: Optional[str], end_date: Optional[str]
) -> pd.DataFrame:
    """Cached wrapper around Phase 8's ``load_price_data``."""
    return load_price_data(stock, PROCESSED_DIR, start_date, end_date)


@st.cache_data(show_spinner=False)
def get_available_date_range(stock: str) -> Tuple[date, date]:
    """
    Return the min/max available dates for *stock*'s processed CSV.

    Parameters
    ----------
    stock:
        Stock display name.

    Returns
    -------
    (date, date)
    """
    df = load_price_data(stock, PROCESSED_DIR)
    return df.index.min().date(), df.index.max().date()


@st.cache_data(show_spinner=False)
def _enrich_with_indicators(
    stock: str,
    start_date: str,
    end_date: str,
    ema_windows: Tuple[int, ...],
    rsi_period: int,
    atr_period: int,
    bb_period: int,
    bb_dev: float,
) -> pd.DataFrame:
    """
    Load price data and append every indicator the dashboard might display.

    Cached on every input that affects the output, so repeated reruns with
    unchanged sidebar values reuse the previous computation.

    Parameters
    ----------
    stock, start_date, end_date:
        Data selection.
    ema_windows:
        EMA spans to compute (deduplicated across whichever strategy is active).
    rsi_period, atr_period, bb_period, bb_dev:
        Indicator parameters.

    Returns
    -------
    pd.DataFrame
    """
    df = load_price_data(stock, PROCESSED_DIR, start_date, end_date)
    df = add_moving_averages(df, windows=list(ema_windows))
    df = add_rsi(df, period=rsi_period)
    df = add_atr(df, period=atr_period)
    df = add_bollinger_bands(df, period=bb_period, num_std=bb_dev)
    return df


def get_stock_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute a quick price/volume summary from the most recent bars.

    Parameters
    ----------
    df:
        Enriched or raw OHLCV DataFrame (DatetimeIndex, ascending).

    Returns
    -------
    dict
        Keys: ``last_close``, ``prev_close``, ``change_pct``, ``high_52w``,
        ``low_52w``, ``avg_volume_30d``, ``last_date``.
    """
    last_close = float(df["close"].iloc[-1])
    prev_close = float(df["close"].iloc[-2]) if len(df) > 1 else last_close
    window = df.tail(252)   # ~52 trading weeks
    return {
        "last_close":      last_close,
        "prev_close":      prev_close,
        "change_pct":      (last_close / prev_close - 1) * 100 if prev_close else 0.0,
        "high_52w":        float(window["high"].max()),
        "low_52w":         float(window["low"].min()),
        "avg_volume_30d":  float(df["volume"].tail(30).mean()),
        "last_date":       df.index[-1].date(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Signal logic
# ──────────────────────────────────────────────────────────────────────────────
def determine_signal(
    strategy_name: str, params: Dict[str, Any], df: pd.DataFrame
) -> Tuple[str, str, Dict[str, bool]]:
    """
    Apply simple, transparent entry/exit rules to the most recent bar.

    This mirrors each strategy's *entry condition* (not its full stateful
    position-tracking logic) purely to answer "what would this strategy's
    rules say about today?" — a live decision-support signal, independent
    of whatever the historical backtest's simulated position happens to be.

    Parameters
    ----------
    strategy_name:
        One of the registry keys.
    params:
        The active parameter set for that strategy.
    df:
        Indicator-enriched DataFrame with at least 2 rows.

    Returns
    -------
    (signal, reason, condition_checklist)
        ``signal`` is one of ``'BUY'``, ``'SELL'``, ``'HOLD'``.
        ``condition_checklist`` maps a human-readable condition to whether
        it currently holds — used to render a transparent checklist in the UI.
    """
    last, prev = df.iloc[-1], df.iloc[-2]

    if strategy_name == "ema_crossover":
        fast_col = f"ema_{params['fast_period']}"
        slow_col = f"ema_{params['slow_period']}"
        fast_now, slow_now = last[fast_col], last[slow_col]
        fast_prev, slow_prev = prev[fast_col], prev[slow_col]
        checklist = {f"EMA{params['fast_period']} > EMA{params['slow_period']}": bool(fast_now > slow_now)}
        if fast_now > slow_now and fast_prev <= slow_prev:
            return "BUY", "Fresh golden cross today.", checklist
        if fast_now < slow_now and fast_prev >= slow_prev:
            return "SELL", "Fresh death cross today.", checklist
        if fast_now > slow_now:
            return "HOLD", "Uptrend intact — stay long.", checklist
        return "HOLD", "Downtrend — stay flat.", checklist

    if strategy_name == "rsi_strategy":
        rsi_col = f"rsi_{params['rsi_period']}"
        rsi_now = float(last[rsi_col])
        checklist = {
            f"RSI < {params['rsi_oversold']} (oversold)":   rsi_now < params["rsi_oversold"],
            f"RSI > {params['rsi_overbought']} (overbought)": rsi_now > params["rsi_overbought"],
        }
        if rsi_now < params["rsi_oversold"]:
            return "BUY", f"RSI={rsi_now:.1f} is oversold.", checklist
        if rsi_now > params["rsi_overbought"]:
            return "SELL", f"RSI={rsi_now:.1f} is overbought.", checklist
        return "HOLD", f"RSI={rsi_now:.1f} is in the neutral zone.", checklist

    # combined_strategy
    fast_col = f"ema_{params['fast_ema_period']}"
    slow_col = f"ema_{params['slow_ema_period']}"
    rsi_col  = f"rsi_{params.get('rsi_period', 14)}"
    checklist = {
        "EMA trend bullish":                       bool(last[fast_col] > last[slow_col]),
        f"RSI < {params['rsi_entry_cap']} (not overbought)": bool(last[rsi_col] < params["rsi_entry_cap"]),
        "Price ≤ Bollinger mid-band (buying a dip)": bool(last["close"] <= last["bb_middle"]),
    }
    if all(checklist.values()):
        return "BUY", "All confluence conditions are met.", checklist
    return "HOLD", "Not all confluence conditions are met.", checklist


# ──────────────────────────────────────────────────────────────────────────────
# Backtest runners
# ──────────────────────────────────────────────────────────────────────────────
def run_single_stock_backtest(
    stock: str,
    strategy_name: str,
    params: Dict[str, Any],
    start_date: str,
    end_date: str,
    initial_cash: float,
    cost_config: TransactionCostConfig,
) -> SingleStockResult:
    """
    Execute one single-stock backtest and bundle every artefact the
    dashboard's tabs need to render.

    Parameters
    ----------
    stock:
        Stock to backtest.
    strategy_name:
        Registry key.
    params:
        Strategy parameter set.
    start_date, end_date:
        Backtest window (inclusive, ``'YYYY-MM-DD'``).
    initial_cash:
        Starting capital in INR.
    cost_config:
        Brokerage / STT / slippage configuration.

    Returns
    -------
    SingleStockResult
    """
    price_df = load_price_data(stock, PROCESSED_DIR, start_date, end_date)
    strategy_class = get_strategy_class(strategy_name)

    raw = evaluate_params(
        strategy_class, price_df, params,
        initial_cash=initial_cash,
        commission=cost_config.total_commission_pct,
        slippage_pct=cost_config.slippage_pct,
        label=f"{stock}_{strategy_name}",
        return_curve=True,
    )
    equity_curve = raw.pop("_equity_curve")
    trade_log_raw = raw.pop("_trade_log")
    trade_log = pd.DataFrame(trade_log_raw) if trade_log_raw else pd.DataFrame(
        columns=["exit_date", "pnl", "pnlcomm"]
    )

    # Buy-and-hold benchmark over the identical window/capital
    shares = initial_cash / float(price_df["close"].iloc[0])
    buy_hold_curve = (price_df["close"] * shares).rename("buy_hold")

    perf = PerformanceAnalyzer(label=f"{stock} {STRATEGY_LABELS[strategy_name]}").compute(
        equity_curve, benchmark_series=buy_hold_curve
    )
    bench_perf = PerformanceAnalyzer(label=f"{stock} Buy & Hold").compute(buy_hold_curve)
    dd_result = DrawdownAnalyzer(label=stock).analyse(equity_curve)

    return SingleStockResult(
        stock=stock, strategy_name=strategy_name, params=params,
        equity_curve=equity_curve, buy_hold_curve=buy_hold_curve,
        trade_log=trade_log, performance=perf, benchmark_perf=bench_perf,
        drawdown_result=dd_result, raw_metrics=raw, cost_config=cost_config,
    )


def run_portfolio_backtest(
    stocks: List[str],
    scheme: AllocationScheme,
    custom_weights: Optional[Dict[str, float]],
    frequency: RebalanceFrequency,
    drift_threshold: float,
    initial_cash: float,
    commission_pct: float,
) -> Tuple[RebalanceSimResult, Dict[str, float]]:
    """
    Run a multi-stock portfolio simulation (Phase 5).

    Parameters
    ----------
    stocks:
        Universe to allocate across.
    scheme:
        Allocation scheme.
    custom_weights:
        Required only when ``scheme == AllocationScheme.CUSTOM``.
    frequency:
        Rebalance trigger mode.
    drift_threshold:
        Drift tolerance before a threshold/hybrid rebalance fires.
    initial_cash:
        Starting capital.
    commission_pct:
        Per-leg commission for rebalance trades.

    Returns
    -------
    (RebalanceSimResult, target_weights)
    """
    allocator = PortfolioAllocator(scheme, custom_weights=custom_weights)
    prices = {s: float(load_price_data(s, PROCESSED_DIR)["close"].iloc[-1]) for s in stocks}

    if scheme == AllocationScheme.RISK_PARITY:
        history = allocator.load_price_history(stocks)
        alloc_result = allocator.allocate(initial_cash, stocks, current_prices=prices, price_history=history)
    else:
        alloc_result = allocator.allocate(initial_cash, stocks, current_prices=prices)

    sim_result = simulate_portfolio(
        stocks=stocks, target_weights=alloc_result.weights,
        initial_capital=initial_cash, frequency=frequency,
        drift_threshold=drift_threshold, commission_pct=commission_pct,
    )
    return sim_result, alloc_result.weights


@st.cache_data(show_spinner=False)
def run_grid_search_cached(
    strategy_name: str, stock: str, param_grid_key: str,
    start_date: str, end_date: str,
) -> pd.DataFrame:
    """
    Cached grid search — re-runs only when the strategy/stock/grid/date change.

    Parameters
    ----------
    strategy_name, stock:
        Search target.
    param_grid_key:
        ``'quick'`` or ``'standard'`` — selects which grid to use.
    start_date, end_date:
        Restricts the search to this window.

    Returns
    -------
    pd.DataFrame
    """
    grid = (QUICK_PARAM_GRIDS if param_grid_key == "quick" else DEFAULT_PARAM_GRIDS)[strategy_name]
    engine = GridSearchEngine(
        strategy_name=strategy_name, stock_name=stock, param_grid=grid,
        start_date=start_date, end_date=end_date,
    )
    return engine.run()


@st.cache_data(show_spinner=False)
def run_walk_forward_cached(
    strategy_name: str, stock: str, param_grid_key: str,
    train_years: float, test_years: float, window_type: str,
) -> WalkForwardResult:
    """
    Cached walk-forward validation run.

    Parameters
    ----------
    strategy_name, stock:
        Validation target.
    param_grid_key:
        ``'quick'`` or ``'standard'``.
    train_years, test_years, window_type:
        Fold configuration — see ``WalkForwardEngine``.

    Returns
    -------
    WalkForwardResult
    """
    grid = (QUICK_PARAM_GRIDS if param_grid_key == "quick" else DEFAULT_PARAM_GRIDS)[strategy_name]
    engine = WalkForwardEngine(
        strategy_name=strategy_name, stock_name=stock, param_grid=grid,
        train_years=train_years, test_years=test_years, window_type=window_type,
    )
    return engine.run()


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
def render_sidebar() -> Dict[str, Any]:
    """
    Render every sidebar control and return the resolved configuration.

    Returns
    -------
    dict
        All user selections, keyed by control name.
    """
    st.sidebar.title("📈 Strategy Controls")
    mode = st.sidebar.radio(
        "Analysis mode", ["Single Stock", "Portfolio (Multi-Stock)"],
        key="mode_radio",
    )

    cfg: Dict[str, Any] = {"mode": mode}

    if mode == "Single Stock":
        cfg["stock"] = st.sidebar.selectbox("Stock", STOCKS, key="stock_select")

        min_d, max_d = get_available_date_range(cfg["stock"])
        default_start = max(min_d, max_d - timedelta(days=365 * 4))
        date_range = st.sidebar.date_input(
            "Date range", value=(default_start, max_d),
            min_value=min_d, max_value=max_d, key="date_range",
        )
        if isinstance(date_range, tuple) and len(date_range) == 2:
            cfg["start_date"], cfg["end_date"] = str(date_range[0]), str(date_range[1])
        else:
            cfg["start_date"], cfg["end_date"] = str(min_d), str(max_d)

        cfg["initial_cash"] = st.sidebar.number_input(
            "Initial capital (₹)", min_value=10_000, max_value=100_000_000,
            value=1_000_000, step=50_000, key="capital_input",
        )

        strategy_key = st.sidebar.selectbox(
            "Strategy", list(STRATEGY_LABELS.keys()),
            format_func=lambda k: STRATEGY_LABELS[k], key="strategy_select",
        )
        cfg["strategy_name"] = strategy_key

        st.sidebar.markdown("**Strategy parameters**")
        params: Dict[str, Any] = {}
        if strategy_key == "ema_crossover":
            params["fast_period"] = st.sidebar.slider("Fast EMA period", 5, 100, 20, key="ema_fast")
            params["slow_period"] = st.sidebar.slider("Slow EMA period", 50, 300, 100, key="ema_slow")
        elif strategy_key == "rsi_strategy":
            params["rsi_period"]       = st.sidebar.slider("RSI period", 5, 30, 14, key="rsi_period")
            params["rsi_oversold"]     = float(st.sidebar.slider("Oversold threshold", 10, 40, 30, key="rsi_os"))
            params["rsi_overbought"]   = float(st.sidebar.slider("Overbought threshold", 60, 90, 70, key="rsi_ob"))
            params["ema_trend_period"] = st.sidebar.slider("Trend-filter EMA period", 50, 300, 200, key="rsi_trend_ema")
            params["use_trend_filter"] = st.sidebar.checkbox("Use trend filter", value=True, key="rsi_filter")
        else:  # combined_strategy
            params["fast_ema_period"] = st.sidebar.slider("Fast EMA period", 10, 100, 50, key="cs_fast")
            params["slow_ema_period"] = st.sidebar.slider("Slow EMA period", 100, 300, 200, key="cs_slow")
            params["rsi_period"]      = st.sidebar.slider("RSI period", 5, 30, 14, key="cs_rsi_period")
            params["rsi_entry_cap"]   = float(st.sidebar.slider("RSI entry cap", 40, 80, 60, key="cs_rsi_cap"))
            params["rsi_exit_level"]  = float(st.sidebar.slider("RSI exit level", 50, 90, 70, key="cs_rsi_exit"))
            params["bb_period"]       = st.sidebar.slider("Bollinger period", 10, 40, 20, key="cs_bb_period")
            params["bb_dev"]          = float(st.sidebar.slider("Bollinger σ multiplier", 1.0, 3.0, 2.0, 0.1, key="cs_bb_dev"))
            params["atr_period"]      = st.sidebar.slider("ATR period", 5, 30, 14, key="cs_atr_period")
            params["atr_stop_mult"]   = float(st.sidebar.slider("ATR stop multiplier", 1.0, 4.0, 2.0, 0.1, key="cs_atr_mult"))
        cfg["params"] = params

        with st.sidebar.expander("⚠️ Risk & position sizing (sample trade plan)"):
            cfg["risk_pct"]      = st.slider("Risk per trade (%)", 0.25, 5.0, 1.0, 0.25, key="risk_pct") / 100
            cfg["atr_stop_mult_risk"] = st.slider("Stop-loss: ATR multiplier", 1.0, 4.0, 2.0, 0.5, key="risk_atr_mult")
            cfg["r_multiple"]   = st.slider("Take-profit R-multiple", 1.0, 4.0, 2.0, 0.5, key="risk_r_mult")

        with st.sidebar.expander("💸 Transaction cost simulation"):
            brokerage = st.slider("Brokerage (%)", 0.0, 0.5, 0.03, 0.01, key="cost_brokerage") / 100
            stt       = st.slider("STT — securities transaction tax (%)", 0.0, 0.5, 0.10, 0.01, key="cost_stt") / 100
            slippage  = st.slider("Slippage (%)", 0.0, 0.5, 0.05, 0.01, key="cost_slippage") / 100
            cfg["cost_config"] = TransactionCostConfig(
                brokerage_pct=brokerage, stt_pct=stt, slippage_pct=slippage
            )

        cfg["run_backtest"] = st.sidebar.button("🚀 Run Backtest", type="primary", key="run_backtest_btn")

    else:  # Portfolio mode
        cfg["stocks"] = st.sidebar.multiselect("Stocks", STOCKS, default=STOCKS, key="portfolio_stocks")
        cfg["initial_cash"] = st.sidebar.number_input(
            "Initial capital (₹)", min_value=10_000, max_value=100_000_000,
            value=1_000_000, step=50_000, key="portfolio_capital",
        )

        scheme_labels = {
            "Equal Weight": AllocationScheme.EQUAL_WEIGHT,
            "Custom":        AllocationScheme.CUSTOM,
            "Risk Parity":   AllocationScheme.RISK_PARITY,
            "Market Cap":    AllocationScheme.MARKET_CAP,
        }
        scheme_name = st.sidebar.selectbox("Allocation scheme", list(scheme_labels.keys()), key="alloc_scheme")
        cfg["scheme"] = scheme_labels[scheme_name]

        cfg["custom_weights"] = None
        if cfg["scheme"] == AllocationScheme.CUSTOM and cfg["stocks"]:
            st.sidebar.markdown("**Custom weights (%)**")
            raw_weights = {}
            for s in cfg["stocks"]:
                raw_weights[s] = st.sidebar.slider(s, 0, 100, int(100 / len(cfg["stocks"])), key=f"weight_{s}")
            total = sum(raw_weights.values()) or 1
            cfg["custom_weights"] = {s: w / total for s, w in raw_weights.items()}
            st.sidebar.caption(f"Normalised to 100% (raw sum: {sum(raw_weights.values())}%)")

        freq_labels = {
            "Monthly":   RebalanceFrequency.MONTHLY,
            "Quarterly": RebalanceFrequency.QUARTERLY,
            "Threshold (drift-triggered)": RebalanceFrequency.THRESHOLD,
            "Hybrid (scheduled + drift)":  RebalanceFrequency.HYBRID,
        }
        freq_name = st.sidebar.selectbox("Rebalance frequency", list(freq_labels.keys()), index=3, key="rebal_freq")
        cfg["frequency"] = freq_labels[freq_name]
        cfg["drift_threshold"] = st.sidebar.slider("Drift threshold (%)", 1, 20, 5, key="drift_thr") / 100
        cfg["commission_pct"] = st.sidebar.slider("Commission per leg (%)", 0.0, 0.5, 0.05, 0.01, key="port_commission") / 100

        cfg["run_portfolio"] = st.sidebar.button("🚀 Run Portfolio Backtest", type="primary", key="run_portfolio_btn")

    return cfg


# ──────────────────────────────────────────────────────────────────────────────
# Tab renderers — Single Stock mode
# ──────────────────────────────────────────────────────────────────────────────
def render_overview_tab(cfg: Dict[str, Any]) -> None:
    """Render the live signal, indicator snapshot, and sample trade plan."""
    stock, strategy_name, params = cfg["stock"], cfg["strategy_name"], cfg["params"]

    #: Each strategy names its EMA-period params differently
    #: (fast_period / fast_ema_period / ema_trend_period), so an explicit
    #: per-strategy map is far more robust here than substring matching.
    _EMA_PARAM_KEYS = {
        "ema_crossover":     ["fast_period", "slow_period"],
        "rsi_strategy":       ["ema_trend_period"],
        "combined_strategy":  ["fast_ema_period", "slow_ema_period"],
    }
    ema_windows = sorted(set(
        params[k] for k in _EMA_PARAM_KEYS.get(strategy_name, []) if k in params
    ) | {20, 50, 200})
    rsi_period = params.get("rsi_period", 14)
    atr_period = params.get("atr_period", 14)
    bb_period  = params.get("bb_period", 20)
    bb_dev     = params.get("bb_dev", 2.0)

    df = _enrich_with_indicators(
        stock, cfg["start_date"], cfg["end_date"],
        tuple(ema_windows), rsi_period, atr_period, bb_period, bb_dev,
    )
    summary = get_stock_summary(df)

    min_required = _required_min_bars(strategy_name, params)
    if len(df) < min_required or df.iloc[-1][[f"ema_{w}" for w in ema_windows]].isna().any():
        st.warning(
            f"⚠️ The selected date range has only {len(df)} trading days, "
            f"but the current strategy parameters need roughly {min_required} "
            "days for every indicator to be fully warmed up. The signal and "
            "indicator values below may show as N/A — pick a longer date "
            "range for a meaningful reading."
        )

    st.subheader(f"{stock} — Live Snapshot ({summary['last_date']})")
    cols = st.columns(4)
    cols[0].metric("Last Close", f"₹{summary['last_close']:,.2f}", f"{summary['change_pct']:+.2f}%")
    cols[1].metric("52-Week High", f"₹{summary['high_52w']:,.2f}")
    cols[2].metric("52-Week Low", f"₹{summary['low_52w']:,.2f}")
    cols[3].metric("Avg Volume (30d)", f"{summary['avg_volume_30d']:,.0f}")

    st.divider()
    signal, reason, checklist = determine_signal(strategy_name, params, df)
    badge = {"BUY": "🟢 BUY", "SELL": "🔴 SELL", "HOLD": "🟡 HOLD"}[signal]
    st.markdown(f"## {badge}")
    st.caption(reason)
    for cond, met in checklist.items():
        st.markdown(f"{'✅' if met else '❌'} {cond}")

    st.divider()
    st.subheader("Indicator Snapshot")
    last = df.iloc[-1]
    fast_key = params.get("fast_period", params.get("fast_ema_period", 20))
    slow_key = params.get("slow_period", params.get("slow_ema_period", params.get("ema_trend_period", 200)))
    ind_cols = st.columns(4)
    ind_cols[0].metric(f"EMA-{fast_key}", f"₹{last.get(f'ema_{fast_key}', np.nan):,.2f}")
    ind_cols[1].metric(f"EMA-{slow_key}", f"₹{last.get(f'ema_{slow_key}', np.nan):,.2f}")
    ind_cols[2].metric(f"RSI-{rsi_period}", f"{last.get(f'rsi_{rsi_period}', np.nan):.1f}")
    ind_cols[3].metric(f"ATR-{atr_period}", f"₹{last.get(f'atr_{atr_period}', np.nan):,.2f}")

    bb_cols = st.columns(3)
    bb_cols[0].metric("BB Upper", f"₹{last['bb_upper']:,.2f}")
    bb_cols[1].metric("BB Middle", f"₹{last['bb_middle']:,.2f}")
    bb_cols[2].metric("BB Lower", f"₹{last['bb_lower']:,.2f}")

    st.divider()
    st.subheader("📋 Sample Trade Plan (Phase 4 risk engine)")
    st.caption(
        "If a BUY signal triggers at the current price, here is the "
        "risk-managed trade plan Phase 4's engine would recommend — "
        "independent of whether the strategy is currently in a position."
    )
    try:
        setup = build_trade_setup(
            stock=stock, capital=cfg["initial_cash"], entry_price=summary["last_close"],
            atr=float(last[f"atr_{atr_period}"]), risk_pct=cfg["risk_pct"],
            atr_stop_mult=cfg["atr_stop_mult_risk"], r_multiple=cfg["r_multiple"],
        )
        setup_cols = st.columns(5)
        setup_cols[0].metric("Entry", f"₹{setup.entry_price:,.2f}")
        setup_cols[1].metric("Stop-Loss", f"₹{setup.stop_price:,.2f}")
        setup_cols[2].metric("Take-Profit", f"₹{setup.target_price:,.2f}")
        setup_cols[3].metric("Shares", f"{setup.shares}")
        setup_cols[4].metric("Max Risk", f"₹{setup.max_risk_inr:,.0f}")
    except Exception as exc:                                    # noqa: BLE001
        st.warning(f"Could not compute a sample trade plan: {exc}")


def render_equity_drawdown_tab(result: SingleStockResult) -> None:
    """Render the equity-curve and drawdown charts (Phase 7)."""
    st.subheader("Equity Curve — Strategy vs Buy & Hold")
    plotter = EquityCurvePlotter(
        title=f"{result.stock} — {STRATEGY_LABELS[result.strategy_name]}"
    )
    fig, _ = plotter.plot_with_benchmark(
        result.equity_curve, result.buy_hold_curve,
        strategy_label="Strategy", benchmark_label="Buy & Hold",
        normalise=True,
    )
    st.pyplot(fig, clear_figure=False)

    st.divider()
    st.subheader("Drawdown — Underwater Curve")
    dd_plotter = DrawdownPlotter(title=f"{result.stock} Drawdown")
    fig2, _ = dd_plotter.plot_combined(result.equity_curve, result.drawdown_result, top_n_shade=3)
    st.pyplot(fig2, clear_figure=False)

    c1, c2, c3 = st.columns(3)
    c1.metric("Max Drawdown", f"{result.drawdown_result.max_drawdown_pct:.2f}%")
    c2.metric("Recovery Rate", f"{result.drawdown_result.recovery_rate_pct:.1f}%")
    c3.metric("Drawdown Periods", f"{result.drawdown_result.n_periods}")


def render_heatmap_tab(result: SingleStockResult) -> None:
    """Render the monthly returns heatmap (Phase 7)."""
    st.subheader("Monthly Returns Heatmap")
    if len(result.equity_curve) < 60:
        st.info("Select a longer date range (≥ 2 months) to render a meaningful heatmap.")
        return
    plotter = ReturnsHeatmapPlotter(title=f"{result.stock} — {STRATEGY_LABELS[result.strategy_name]}")
    fig, _ = plotter.plot_monthly_heatmap(result.equity_curve)
    st.pyplot(fig, clear_figure=False)


def render_trades_tab(result: SingleStockResult) -> None:
    """Render the trade log table and simple win/loss summary."""
    st.subheader("Trade Log")
    if result.trade_log.empty:
        st.info("No closed trades in this backtest window with the current parameters.")
        return

    tl = result.trade_log.copy()
    tl["exit_date"] = pd.to_datetime(tl["exit_date"]).dt.date
    tl = tl.rename(columns={"pnl": "Gross P&L (₹)", "pnlcomm": "Net P&L (₹)", "exit_date": "Exit Date"})
    tl.insert(0, "Trade #", range(1, len(tl) + 1))
    tl["Result"] = np.where(tl["Net P&L (₹)"] > 0, "✅ Win", "❌ Loss")

    wins = tl[tl["Net P&L (₹)"] > 0]
    losses = tl[tl["Net P&L (₹)"] < 0]
    win_rate = len(wins) / len(tl) * 100 if len(tl) else 0.0
    profit_factor = (
        wins["Net P&L (₹)"].sum() / abs(losses["Net P&L (₹)"].sum())
        if len(losses) and losses["Net P&L (₹)"].sum() != 0 else float("inf")
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Trades", len(tl))
    c2.metric("Win Rate", f"{win_rate:.1f}%")
    c3.metric("Profit Factor", f"{profit_factor:.2f}×" if np.isfinite(profit_factor) else "∞")
    c4.metric("Net P&L", f"₹{tl['Net P&L (₹)'].sum():+,.0f}")

    st.dataframe(
        tl[["Trade #", "Exit Date", "Gross P&L (₹)", "Net P&L (₹)", "Result"]],
        use_container_width=True, hide_index=True,
    )

    csv = tl.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Export trade log (CSV)", csv, f"{result.stock}_trades.csv", "text/csv")


def render_metrics_tab(result: SingleStockResult) -> None:
    """Render the full Phase 6 performance metrics table."""
    st.subheader("Performance Metrics — Strategy vs Buy & Hold")

    rows = [
        ("Total Return %",       "total_return_pct"),
        ("CAGR %",               "cagr_pct"),
        ("Annualised Vol %",     "vol_pct"),
        ("Sharpe Ratio",         "sharpe"),
        ("Sortino Ratio",        "sortino"),
        ("Calmar Ratio",         "calmar"),
        ("Omega Ratio",          "omega"),
        ("Ulcer Index",          "ulcer_index"),
        ("Recovery Factor",      "recovery_factor"),
        ("Max Drawdown %",       "max_drawdown_pct"),
        ("VaR 95% (daily)",      "var_95_pct"),
        ("CVaR 95% (daily)",     "cvar_95_pct"),
        ("Best Month %",         "best_month_pct"),
        ("Worst Month %",        "worst_month_pct"),
        ("% Positive Months",    "pct_positive_months"),
    ]
    table = pd.DataFrame({
        "Metric":     [label for label, _ in rows],
        "Strategy":   [getattr(result.performance, key) for _, key in rows],
        "Buy & Hold": [getattr(result.benchmark_perf, key) for _, key in rows],
    })
    st.dataframe(table, use_container_width=True, hide_index=True)

    if result.performance.alpha is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Alpha (ann.)", f"{result.performance.alpha:+.2f}%")
        c2.metric("Beta", f"{result.performance.beta:.3f}")
        c3.metric("Information Ratio", f"{result.performance.information_ratio:.3f}")

    csv = table.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Export metrics (CSV)", csv, f"{result.stock}_metrics.csv", "text/csv")


def render_optimization_tab(cfg: Dict[str, Any]) -> None:
    """Render the Phase 8 grid search + walk-forward validation controls."""
    st.subheader("🔧 Parameter Optimization (Grid Search)")
    st.caption(
        "Searches a range of parameter combinations and ranks them by Sharpe Ratio. "
        "'Quick' uses a 4-combo grid for a fast demo; 'Standard' uses the project's full grid."
    )
    grid_size = st.radio("Search size", ["Quick", "Standard"], horizontal=True, key="gs_size")

    if st.button("Run Grid Search", key="run_gs_btn"):
        with st.spinner("Running grid search…"):
            try:
                results = run_grid_search_cached(
                    cfg["strategy_name"], cfg["stock"], grid_size.lower(),
                    cfg["start_date"], cfg["end_date"],
                )
                st.session_state["gs_results"] = results
            except Exception as exc:                            # noqa: BLE001
                st.error(f"Grid search failed: {exc}")

    if st.session_state.get("gs_results") is not None:
        results = st.session_state["gs_results"]
        param_cols = [c for c in results.columns if c not in (
            "rank", "final_value", "return_pct", "cagr_pct", "vol_pct",
            "sharpe_ratio", "sortino_ratio", "calmar_ratio", "max_drawdown_pct",
            "total_trades", "win_rate_pct", "profit_factor", "avg_win_inr",
            "avg_loss_inr", "expectancy_inr",
        )]
        display_cols = ["rank", *param_cols, "return_pct", "sharpe_ratio", "max_drawdown_pct", "total_trades"]
        st.dataframe(results[display_cols].head(10), use_container_width=True, hide_index=True)
        best = results.iloc[0]
        st.success(
            f"Best parameters: {dict((c, best[c]) for c in param_cols)}  "
            f"→ Sharpe={best['sharpe_ratio']:.3f}, Return={best['return_pct']:+.2f}%"
        )

    st.divider()
    st.subheader("🧪 Walk-Forward Validation")
    st.caption(
        "Trains on a rolling window and tests on the immediately following, "
        "unseen period — the standard defence against overfitting."
    )
    wf_cols = st.columns(3)
    train_years = wf_cols[0].slider("Train window (years)", 1.0, 5.0, 3.0, 0.5, key="wf_train_years")
    test_years  = wf_cols[1].slider("Test window (years)", 0.5, 2.0, 1.0, 0.5, key="wf_test_years")
    window_type = wf_cols[2].selectbox("Window type", ["rolling", "expanding"], key="wf_window_type")

    if st.button("Run Walk-Forward Validation", key="run_wf_btn"):
        with st.spinner("Running walk-forward validation (training + testing each fold)…"):
            try:
                wf_result = run_walk_forward_cached(
                    cfg["strategy_name"], cfg["stock"], grid_size.lower(),
                    train_years, test_years, window_type,
                )
                st.session_state["wf_result"] = wf_result
            except Exception as exc:                            # noqa: BLE001
                st.error(f"Walk-forward validation failed: {exc}")

    if st.session_state.get("wf_result") is not None:
        wf_result: WalkForwardResult = st.session_state["wf_result"]
        st.dataframe(wf_result.fold_table, use_container_width=True, hide_index=True)

        st.markdown("**Out-of-sample summary (across all folds)**")
        s = wf_result.summary
        sc = st.columns(4)
        sc[0].metric("Mean OOS Return", f"{s['oos_return_pct_mean']:+.2f}%")
        sc[1].metric("Mean OOS Sharpe", f"{s['oos_sharpe_mean']:.3f}")
        sc[2].metric("IS→OOS Efficiency", f"{s['is_to_oos_efficiency']:.2f}")
        sc[3].metric("% Folds Profitable", f"{s['pct_folds_profitable']:.0f}%")
        if s["is_to_oos_efficiency"] < 0.5:
            st.warning(
                "⚠️ Low IS→OOS efficiency — the in-sample optimum may not "
                "generalise well to unseen data (possible overfitting)."
            )


# ──────────────────────────────────────────────────────────────────────────────
# Tab renderers — Portfolio mode
# ──────────────────────────────────────────────────────────────────────────────
def render_portfolio_tabs(sim_result: RebalanceSimResult, weights: Dict[str, float], stocks: List[str]) -> None:
    """Render allocation, equity/drawdown, rebalance log, and metrics for portfolio mode."""
    tabs = st.tabs(["Allocation", "Equity & Drawdown", "Rebalance Log", "Metrics"])

    with tabs[0]:
        st.subheader("Target Allocation Weights")
        weight_df = pd.DataFrame({
            "Stock": list(weights.keys()),
            "Weight %": [w * 100 for w in weights.values()],
        })
        st.dataframe(weight_df, use_container_width=True, hide_index=True)
        st.bar_chart(weight_df.set_index("Stock"))

    with tabs[1]:
        st.subheader("Portfolio Equity Curve")
        plotter = EquityCurvePlotter(title="Portfolio Equity Curve")
        fig, _ = plotter.plot_single(sim_result.portfolio_value_series, label="Portfolio")
        st.pyplot(fig, clear_figure=False)

        st.subheader("Portfolio Drawdown")
        dd_analyzer = DrawdownAnalyzer(label="Portfolio")
        dd_result = dd_analyzer.analyse(sim_result.portfolio_value_series)
        dd_plotter = DrawdownPlotter(title="Portfolio Drawdown")
        fig2, _ = dd_plotter.plot_combined(sim_result.portfolio_value_series, dd_result)
        st.pyplot(fig2, clear_figure=False)

    with tabs[2]:
        st.subheader("Rebalance Activity Log")
        st.caption(
            "Buy/sell orders generated at each rebalance event. This reflects "
            "portfolio weight restoration, not a strategy's own entry/exit P&L."
        )
        if sim_result.trade_log.empty:
            st.info("No rebalance trades occurred.")
        else:
            log = sim_result.trade_log.copy()
            log["date"] = pd.to_datetime(log["date"]).dt.date
            st.dataframe(log, use_container_width=True, hide_index=True)
            csv = log.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Export rebalance log (CSV)", csv, "portfolio_rebalance_log.csv", "text/csv")

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Rebalances", sim_result.total_rebalances)
        c2.metric("Total Commission Paid", f"₹{sim_result.total_commission_paid:,.0f}")
        c3.metric("Final Cash", f"₹{sim_result.final_cash:,.0f}")

    with tabs[3]:
        st.subheader("Portfolio Performance Metrics")
        perf = PerformanceAnalyzer(label="Portfolio").compute(sim_result.portfolio_value_series)
        rows = [
            ("Total Return %", "total_return_pct"), ("CAGR %", "cagr_pct"),
            ("Annualised Vol %", "vol_pct"), ("Sharpe Ratio", "sharpe"),
            ("Sortino Ratio", "sortino"), ("Calmar Ratio", "calmar"),
            ("Max Drawdown %", "max_drawdown_pct"), ("% Positive Months", "pct_positive_months"),
        ]
        table = pd.DataFrame({"Metric": [r[0] for r in rows], "Value": [getattr(perf, r[1]) for r in rows]})
        st.dataframe(table, use_container_width=True, hide_index=True)

        st.markdown("**Final Holdings**")
        last_prices = {s: float(load_price_data(s, PROCESSED_DIR)["close"].iloc[-1]) for s in stocks}
        holdings_rows = []
        for s, shares in sim_result.final_holdings.items():
            holdings_rows.append({"Stock": s, "Shares": shares, "Value (₹)": shares * last_prices[s]})
        st.dataframe(pd.DataFrame(holdings_rows), use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────────────────────────────────────
# Top-level orchestrator
# ──────────────────────────────────────────────────────────────────────────────
def _check_data_availability() -> Optional[str]:
    """Return an error message if any required processed CSV is missing."""
    missing = [s for s in STOCKS if not os.path.exists(
        os.path.join(PROCESSED_DIR, f"{s}_processed.csv")
    )]
    if missing:
        return (
            f"Missing processed data for: {', '.join(missing)}. "
            "Run `src/data/fetch_data.py` then `src/data/preprocess.py` first."
        )
    return None


def _required_min_bars(strategy_name: str, params: Dict[str, Any]) -> int:
    """
    Minimum trading-day history needed for every indicator to be valid.

    Selecting a date range shorter than this causes Backtrader's
    indicators to attempt to read buffer positions that don't exist yet,
    which surfaces as a cryptic low-level error rather than a clear
    message — this function lets the dashboard catch that case upfront.

    Parameters
    ----------
    strategy_name:
        Registry key.
    params:
        The active parameter set.

    Returns
    -------
    int
        Minimum number of bars required, with a small safety margin.
    """
    if strategy_name == "ema_crossover":
        lookback = params["slow_period"]
    elif strategy_name == "rsi_strategy":
        lookback = params["rsi_period"]
        if params.get("use_trend_filter"):
            lookback = max(lookback, params["ema_trend_period"])
    else:  # combined_strategy
        lookback = max(
            params["slow_ema_period"], params["rsi_period"],
            params["bb_period"], params["atr_period"],
        )
    return lookback + 20   # small safety margin for the indicator to stabilise


def render_dashboard() -> None:
    """
    Top-level entry point — renders the entire dashboard.

    Called by ``main.py``. Performs a startup data-availability check,
    renders the sidebar, and dispatches to the single-stock or portfolio
    tab layout based on the selected mode.
    """
    st.title("📊 Algorithmic Trading Strategy Backtester")
    st.caption("TCS · Reliance · Infosys — Indian equity strategy research dashboard")

    error = _check_data_availability()
    if error:
        st.error(error)
        st.stop()

    for key in ("backtest_result", "gs_results", "wf_result", "portfolio_result"):
        st.session_state.setdefault(key, None)

    cfg = render_sidebar()

    if cfg["mode"] == "Single Stock":
        if cfg.get("run_backtest"):
            available_days = (pd.Timestamp(cfg["end_date"]) - pd.Timestamp(cfg["start_date"])).days
            min_required = _required_min_bars(cfg["strategy_name"], cfg["params"])
            # Calendar days overestimate trading days; ×1.6 is a safe conversion margin.
            if available_days < min_required * 1.6:
                st.sidebar.error(
                    f"Selected date range is too short for these parameters. "
                    f"This strategy needs roughly {min_required} trading days of "
                    f"history (~{int(min_required * 1.6)} calendar days), but only "
                    f"{available_days} calendar days are selected. Pick a longer "
                    f"date range or reduce the longest lookback period."
                )
            else:
                with st.spinner(f"Running {STRATEGY_LABELS[cfg['strategy_name']]} backtest on {cfg['stock']}…"):
                    try:
                        st.session_state["backtest_result"] = run_single_stock_backtest(
                            cfg["stock"], cfg["strategy_name"], cfg["params"],
                            cfg["start_date"], cfg["end_date"], cfg["initial_cash"], cfg["cost_config"],
                        )
                        st.session_state["gs_results"] = None
                        st.session_state["wf_result"] = None
                    except Exception as exc:                        # noqa: BLE001
                        logger.exception("Backtest failed")
                        st.error(f"Backtest failed: {exc}")

        result: Optional[SingleStockResult] = st.session_state["backtest_result"]

        tabs = st.tabs([
            "Overview & Signal", "Equity & Drawdown", "Returns Heatmap",
            "Trades", "Performance Metrics", "Optimization",
        ])
        with tabs[0]:
            render_overview_tab(cfg)
        if result is None:
            for t in tabs[1:5]:
                with t:
                    st.info("Click **🚀 Run Backtest** in the sidebar to populate this tab.")
        else:
            with tabs[1]:
                render_equity_drawdown_tab(result)
            with tabs[2]:
                render_heatmap_tab(result)
            with tabs[3]:
                render_trades_tab(result)
            with tabs[4]:
                render_metrics_tab(result)
        with tabs[5]:
            render_optimization_tab(cfg)

    else:  # Portfolio mode
        if cfg.get("run_portfolio"):
            if len(cfg["stocks"]) < 2:
                st.sidebar.error("Select at least 2 stocks for portfolio mode.")
            else:
                with st.spinner("Running portfolio simulation…"):
                    try:
                        sim_result, weights = run_portfolio_backtest(
                            cfg["stocks"], cfg["scheme"], cfg["custom_weights"],
                            cfg["frequency"], cfg["drift_threshold"],
                            cfg["initial_cash"], cfg["commission_pct"],
                        )
                        st.session_state["portfolio_result"] = (sim_result, weights, cfg["stocks"])
                    except Exception as exc:                    # noqa: BLE001
                        logger.exception("Portfolio backtest failed")
                        st.error(f"Portfolio backtest failed: {exc}")

        bundle = st.session_state["portfolio_result"]
        if bundle is None:
            st.info("Configure your portfolio in the sidebar and click **🚀 Run Portfolio Backtest**.")
        else:
            render_portfolio_tabs(*bundle)
