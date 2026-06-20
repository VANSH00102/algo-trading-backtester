"""
grid_search.py
===============
Strategy-agnostic parameter grid search engine for the Algorithmic
Trading Strategy Backtester — Phase 8.

What this module does
----------------------
Given a Phase 3 strategy (``ema_crossover``, ``rsi_strategy``, or
``combined_strategy``) and a parameter search space, this module:

1. Builds the Cartesian product of every parameter combination
   (via ``itertools.product``), optionally filtered by a *constraint*
   function (e.g. ``fast_period < slow_period``) so nonsensical
   combinations are never backtested.
2. Runs an independent Backtrader backtest for every surviving
   combination.
3. Extracts a *complete* performance-metric row per combination by
   reconstructing the equity curve from Backtrader's ``TimeReturn``
   analyzer and feeding it through Phase 6's ``PerformanceAnalyzer`` —
   so grid search results use the exact same metric definitions as
   every other report in this project.
4. Ranks all combinations by a chosen metric (default: Sharpe Ratio)
   and returns/saves a tidy comparison table.

Design notes
------------
**Why a custom trade-log analyzer?**
Backtrader's built-in ``TradeAnalyzer`` only exposes aggregated
counts, not a per-trade list with exact exit dates. ``walk_forward.py``
needs per-trade exit dates to correctly scope trade statistics to an
out-of-sample test window, so this module defines :class:`TradeLogAnalyzer`
once here and both files share it.

**Why sequential execution by default?**
Each backtest in this project completes in roughly 100–300 ms. Empirical
benchmarking showed that joblib's process-based backend ("loky") spends
more time spawning workers and pickling data than it saves for grids of
this size — sequential execution was consistently faster for grids under
~50 combinations. ``n_jobs`` is fully supported for larger search spaces
or heavier strategies where the per-task cost amortises the overhead;
the default of ``n_jobs=1`` reflects what is fastest *for this project's
typical scale*, not a limitation of the engine.

**Why a strategy registry instead of one hardcoded strategy?**
``STRATEGY_REGISTRY`` maps a string name to the actual ``bt.Strategy``
subclass from Phase 3. This lets ``grid_search()`` and
``WalkForwardEngine`` (Phase 8b) work with *any* registered strategy
through one consistent API, satisfying the requirement to avoid
hardcoding a single strategy.

Integration
------------
* Reuses Phase 3 strategy classes directly (``EMACrossoverStrategy``,
  ``RSIStrategy``, ``CombinedStrategy``) — no duplication of signal logic.
* Reuses Phase 6's ``PerformanceAnalyzer`` for every metric calculation.
* ``walk_forward.py`` imports ``GridSearchEngine``, ``evaluate_params``,
  ``TradeLogAnalyzer``, and the registries directly from this module.

Usage
-----
::

    python src/optimization/grid_search.py
    from src.optimization.grid_search import grid_search, GridSearchEngine
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd
import backtrader as bt

# ── Pandas 2.0 / 3.0 compatibility shim (consistent with Phase 3) ───────────
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
PROCESSED_DIR: str          = os.path.join("data", "processed")
OPTIMIZATION_DIR: str       = os.path.join("reports", "optimization")
DEFAULT_INITIAL_CASH: float = 1_000_000.0
DEFAULT_COMMISSION: float   = 0.0005
DEFAULT_RISK_FREE: float    = 0.06

#: Metric values can be ±inf in degenerate cases (e.g. zero losing trades).
#: Sorting still needs a total order, so inf is mapped to this finite sentinel.
_INF_SENTINEL: float = 1e18


# ──────────────────────────────────────────────────────────────────────────────
# Strategy registry — maps a string name to the actual bt.Strategy subclass
# ──────────────────────────────────────────────────────────────────────────────
def _import_strategy_registry() -> Dict[str, Type[bt.Strategy]]:
    """
    Import Phase 3 strategy classes lazily and build the registry.

    Deferred (function-level) import keeps this module importable even
    in contexts where the strategies package has heavier dependencies
    not yet on the path, and gives a single clear error message if a
    Phase 3 file is missing.
    """
    try:
        from src.strategies.ema_crossover import EMACrossoverStrategy
        from src.strategies.rsi_strategy import RSIStrategy
        from src.strategies.combined_strategy import CombinedStrategy
    except ImportError as exc:                                  # pragma: no cover
        raise ImportError(
            "Could not import Phase 3 strategy classes. Ensure this is run "
            "from the project root and src/strategies/*.py exist. "
            f"Original error: {exc}"
        ) from exc

    return {
        "ema_crossover":     EMACrossoverStrategy,
        "rsi_strategy":      RSIStrategy,
        "combined_strategy": CombinedStrategy,
    }


#: Lazily populated on first access via :func:`get_strategy_class`.
_STRATEGY_REGISTRY: Optional[Dict[str, Type[bt.Strategy]]] = None


def get_strategy_class(strategy_name: str) -> Type[bt.Strategy]:
    """
    Resolve a strategy name string to its Backtrader strategy class.

    Parameters
    ----------
    strategy_name:
        One of ``'ema_crossover'``, ``'rsi_strategy'``, ``'combined_strategy'``.

    Returns
    -------
    Type[bt.Strategy]

    Raises
    ------
    ValueError
        If *strategy_name* is not registered.
    """
    global _STRATEGY_REGISTRY
    if _STRATEGY_REGISTRY is None:
        _STRATEGY_REGISTRY = _import_strategy_registry()
    if strategy_name not in _STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy '{strategy_name}'. "
            f"Available: {list(_STRATEGY_REGISTRY.keys())}"
        )
    return _STRATEGY_REGISTRY[strategy_name]


# ──────────────────────────────────────────────────────────────────────────────
# Default search spaces & constraints per strategy
# ──────────────────────────────────────────────────────────────────────────────
#: Example search spaces matching the project's stated parameter ranges.
DEFAULT_PARAM_GRIDS: Dict[str, Dict[str, List[Any]]] = {
    "ema_crossover": {
        "fast_period": [10, 20, 30, 50],
        "slow_period": [100, 150, 200, 250],
    },
    "rsi_strategy": {
        "rsi_oversold":     [25, 30, 35],
        "rsi_overbought":   [65, 70, 75],
        "use_trend_filter": [True, False],
    },
    "combined_strategy": {
        "fast_ema_period": [20, 50],
        "slow_ema_period": [150, 200],
        "rsi_entry_cap":   [55, 60, 65],
        "atr_stop_mult":   [1.5, 2.0, 2.5],
    },
}


def _ema_constraint(params: Dict[str, Any]) -> bool:
    """Reject combinations where the fast EMA is not faster than the slow EMA."""
    fast, slow = params.get("fast_period"), params.get("slow_period")
    return fast < slow if (fast is not None and slow is not None) else True


def _rsi_constraint(params: Dict[str, Any]) -> bool:
    """Reject combinations where the oversold threshold isn't below overbought."""
    lo, hi = params.get("rsi_oversold"), params.get("rsi_overbought")
    return lo < hi if (lo is not None and hi is not None) else True


def _combined_constraint(params: Dict[str, Any]) -> bool:
    """Reject combinations with an invalid EMA pair or RSI entry/exit ordering."""
    ok = True
    fast, slow = params.get("fast_ema_period"), params.get("slow_ema_period")
    if fast is not None and slow is not None:
        ok = ok and (fast < slow)
    cap, exit_ = params.get("rsi_entry_cap"), params.get("rsi_exit_level")
    if cap is not None and exit_ is not None:
        ok = ok and (cap < exit_)
    return ok


#: Sensible default constraints applied automatically unless overridden.
DEFAULT_CONSTRAINTS: Dict[str, Callable[[Dict[str, Any]], bool]] = {
    "ema_crossover":     _ema_constraint,
    "rsi_strategy":      _rsi_constraint,
    "combined_strategy": _combined_constraint,
}


# ──────────────────────────────────────────────────────────────────────────────
# Custom Backtrader analyzer — per-trade exit log
# ──────────────────────────────────────────────────────────────────────────────
class TradeLogAnalyzer(bt.Analyzer):
    """
    Captures a flat list of every closed trade with its exit date and P&L.

    Backtrader's built-in ``TradeAnalyzer`` only returns aggregated counts
    (total/won/lost), which is insufficient for walk-forward validation —
    that requires knowing exactly *when* each trade closed so out-of-sample
    statistics can be scoped to the official test window only (excluding
    any trade that closed during the pre-test warm-up buffer).

    This analyzer is attached to *any* strategy via
    ``cerebro.addanalyzer(TradeLogAnalyzer)`` — Backtrader automatically
    invokes ``notify_trade`` on every registered analyzer whenever a trade
    closes, independent of the strategy's own ``notify_trade`` logging.

    Attributes
    ----------
    trades : List[Dict[str, Any]]
        One entry per closed trade: ``{'exit_date', 'pnl', 'pnlcomm'}``.
    """

    def __init__(self) -> None:
        self.trades: List[Dict[str, Any]] = []

    def notify_trade(self, trade: bt.Trade) -> None:
        """Record the trade the moment it closes."""
        if trade.isclosed:
            self.trades.append({
                "exit_date": bt.num2date(trade.dtclose),
                "pnl":       trade.pnl,
                "pnlcomm":   trade.pnlcomm,
            })

    def get_analysis(self) -> List[Dict[str, Any]]:
        """Return the captured trade list."""
        return self.trades


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────
def load_price_data(
    stock_name: str,
    processed_dir: str = PROCESSED_DIR,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load a Phase-1 processed CSV as a Backtrader-ready DataFrame.

    Parameters
    ----------
    stock_name:
        Display name (e.g. ``'TCS'``).
    processed_dir:
        Directory containing processed CSVs.
    start_date, end_date:
        Optional inclusive date bounds (``'YYYY-MM-DD'``) to slice the
        series before backtesting.

    Returns
    -------
    pd.DataFrame
        DatetimeIndex, columns ``[open, high, low, close, volume]``.

    Raises
    ------
    FileNotFoundError
        If the processed CSV does not exist.
    """
    filepath = os.path.join(processed_dir, f"{stock_name.upper()}_processed.csv")
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Processed data not found: {filepath}\nRun src/data/preprocess.py first."
        )
    df = pd.read_csv(filepath, parse_dates=["date"])
    df = df.drop(columns=["ticker"], errors="ignore")
    df = df.set_index("date").sort_index()
    df.index = pd.DatetimeIndex(df.index)

    if start_date or end_date:
        df = df.loc[start_date:end_date]
    if df.empty:
        raise ValueError(
            f"No data for {stock_name} in range [{start_date}, {end_date}]."
        )
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Core single-combination evaluator
# ──────────────────────────────────────────────────────────────────────────────
def _summarise_trades(trade_log: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute win-rate / profit-factor / expectancy from a trade list.

    Parameters
    ----------
    trade_log:
        Output of :meth:`TradeLogAnalyzer.get_analysis`.

    Returns
    -------
    dict
        Keys: ``total_trades``, ``win_rate_pct``, ``profit_factor``,
        ``avg_win_inr``, ``avg_loss_inr``, ``expectancy_inr``.
    """
    if not trade_log:
        return {
            "total_trades": 0, "win_rate_pct": 0.0, "profit_factor": 0.0,
            "avg_win_inr": 0.0, "avg_loss_inr": 0.0, "expectancy_inr": 0.0,
        }
    pnl    = pd.Series([t["pnlcomm"] for t in trade_log])
    wins   = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_profit = float(wins.sum())
    gross_loss   = float(abs(losses.sum()))

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = float("inf") if gross_profit > 0 else 0.0

    return {
        "total_trades":   len(pnl),
        "win_rate_pct":   round(len(wins) / len(pnl) * 100, 2),
        "profit_factor":  round(profit_factor, 3) if np.isfinite(profit_factor) else profit_factor,
        "avg_win_inr":    round(float(wins.mean()), 2) if len(wins) else 0.0,
        "avg_loss_inr":   round(float(losses.mean()), 2) if len(losses) else 0.0,
        "expectancy_inr": round(float(pnl.mean()), 2),
    }


def _reconstruct_equity_curve(
    time_return_analysis: Dict,
    initial_cash: float,
    fallback_index: pd.DatetimeIndex,
    fallback_end_value: float,
) -> pd.Series:
    """
    Rebuild a daily portfolio-value series from Backtrader's TimeReturn analyzer.

    Parameters
    ----------
    time_return_analysis:
        ``OrderedDict[datetime, float]`` from ``TimeReturn.get_analysis()``.
    initial_cash:
        Starting capital — the series' first value.
    fallback_index:
        Used only if the analysis is empty (degenerate edge case).
    fallback_end_value:
        Used only if the analysis is empty.

    Returns
    -------
    pd.Series
        Cumulative portfolio value, indexed by date.
    """
    if not time_return_analysis:
        return pd.Series(
            [initial_cash, fallback_end_value],
            index=[fallback_index[0], fallback_index[-1]],
        )
    dates = pd.to_datetime(list(time_return_analysis.keys()))
    rets  = pd.Series(list(time_return_analysis.values()), index=dates).sort_index()
    return initial_cash * (1 + rets).cumprod()


def evaluate_params(
    strategy_class: Type[bt.Strategy],
    price_df: pd.DataFrame,
    params: Dict[str, Any],
    initial_cash: float = DEFAULT_INITIAL_CASH,
    commission: float   = DEFAULT_COMMISSION,
    label: str           = "backtest",
    return_curve: bool   = False,
    slippage_pct: float  = 0.0
) -> Dict[str, Any]:
    """
    Run a single Backtrader backtest for one parameter combination.

    This is the atomic unit of work for both grid search (called once
    per combination) and walk-forward validation (called once per fold
    for in-sample search, and once more for the out-of-sample test).

    Parameters
    ----------
    strategy_class:
        A ``bt.Strategy`` subclass (e.g. ``EMACrossoverStrategy``).
    price_df:
        OHLCV DataFrame with a DatetimeIndex (already date-sliced if needed).
    params:
        Keyword arguments passed to ``strategy_class`` as Backtrader params.
    initial_cash:
        Starting capital in INR.
    commission:
        Per-leg commission fraction.
    label:
        Used only for the internal ``PerformanceAnalyzer`` label (logging).
    return_curve:
        If ``True``, include the full equity curve and trade log in the
        returned dict under the private keys ``'_equity_curve'`` and
        ``'_trade_log'``. Default ``False`` keeps grid-search rows lean;
        ``walk_forward.py`` sets this ``True`` for its single OOS evaluation
        per fold, where the curve is needed to trim/rebase to the test window.

    Returns
    -------
    dict
        Flattened ``{**params, **metrics}`` row. Metric keys: ``final_value``,
        ``return_pct``, ``cagr_pct``, ``vol_pct``, ``sharpe_ratio``,
        ``sortino_ratio``, ``calmar_ratio``, ``max_drawdown_pct``,
        ``total_trades``, ``win_rate_pct``, ``profit_factor``,
        ``avg_win_inr``, ``avg_loss_inr``, ``expectancy_inr``.

    Raises
    ------
    Exception
        Propagates any Backtrader execution error to the caller, which
        is expected to catch and log it per-combination (see
        :meth:`GridSearchEngine.run`) so one bad combination cannot abort
        an entire search.
    """
    from src.analytics.performance_metrics import PerformanceAnalyzer  # lazy import

    cerebro = bt.Cerebro(stdstats=False)
    cerebro.adddata(bt.feeds.PandasData(dataname=price_df))
    cerebro.addstrategy(strategy_class, printlog=False, **params)
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=commission)
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="timereturn",
                        timeframe=bt.TimeFrame.Days)
    cerebro.addanalyzer(TradeLogAnalyzer, _name="tradelog")

    results   = cerebro.run()
    strat     = results[0]
    end_value = cerebro.broker.getvalue()

    tr_analysis = strat.analyzers.timereturn.get_analysis()
    equity = _reconstruct_equity_curve(
        tr_analysis, initial_cash, price_df.index, end_value
    )
    trade_log = strat.analyzers.tradelog.get_analysis()

    perf  = PerformanceAnalyzer(label=label, risk_free_rate=DEFAULT_RISK_FREE).compute(equity)
    trade_stats = _summarise_trades(trade_log)

    row: Dict[str, Any] = {
        **params,
        "final_value":       round(end_value, 2),
        "return_pct":        perf.total_return_pct,
        "cagr_pct":          perf.cagr_pct,
        "vol_pct":           perf.vol_pct,
        "sharpe_ratio":      perf.sharpe,
        "sortino_ratio":     perf.sortino,
        "calmar_ratio":      perf.calmar,
        "max_drawdown_pct":  perf.max_drawdown_pct,
        **trade_stats,
    }
    if return_curve:
        row["_equity_curve"] = equity
        row["_trade_log"]    = trade_log
    return row


# ──────────────────────────────────────────────────────────────────────────────
# Parameter grid builder
# ──────────────────────────────────────────────────────────────────────────────
def build_param_grid(
    param_space: Dict[str, List[Any]],
    constraint: Optional[Callable[[Dict[str, Any]], bool]] = None,
) -> List[Dict[str, Any]]:
    """
    Expand a parameter search space into a filtered list of combinations.

    Parameters
    ----------
    param_space:
        ``{param_name: [candidate_values]}``.
    constraint:
        Optional predicate; combinations for which it returns ``False``
        are discarded (e.g. requiring ``fast_period < slow_period``).

    Returns
    -------
    List[Dict[str, Any]]
        Every valid combination as a kwargs-style dict.

    Raises
    ------
    ValueError
        If *param_space* is empty.

    Examples
    --------
    >>> build_param_grid({"a": [1, 2], "b": [10, 20]})
    [{'a': 1, 'b': 10}, {'a': 1, 'b': 20}, {'a': 2, 'b': 10}, {'a': 2, 'b': 20}]
    """
    if not param_space:
        raise ValueError("param_space must not be empty.")

    keys   = list(param_space.keys())
    values = list(param_space.values())
    raw_combos = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    if constraint is not None:
        combos = [c for c in raw_combos if constraint(c)]
        logger.info(
            "Parameter grid: %d raw combinations → %d after constraint filtering.",
            len(raw_combos), len(combos),
        )
    else:
        combos = raw_combos
        logger.info("Parameter grid: %d combinations (no constraint applied).", len(combos))

    if not combos:
        raise ValueError(
            "No parameter combinations survived the constraint filter. "
            "Check that param_space values are compatible with the constraint."
        )
    return combos


# ──────────────────────────────────────────────────────────────────────────────
# Grid Search Engine
# ──────────────────────────────────────────────────────────────────────────────
class GridSearchEngine:
    """
    Runs a parameter grid search for one strategy over one price series.

    Accepts either a ``stock_name`` (loaded from ``data/processed/``) or a
    pre-loaded ``price_df`` directly — the latter is what
    :mod:`walk_forward` uses to grid-search on an in-memory training slice
    without touching disk.

    Parameters
    ----------
    strategy_name:
        Key into the strategy registry: ``'ema_crossover'``,
        ``'rsi_strategy'``, or ``'combined_strategy'``.
    param_grid:
        ``{param_name: [candidate_values]}`` search space.
    stock_name:
        Display name to load via :func:`load_price_data`. Mutually
        exclusive with *price_df* (one of the two is required).
    price_df:
        Pre-loaded OHLCV DataFrame (DatetimeIndex). Takes precedence over
        *stock_name* if both are given.
    start_date, end_date:
        Optional date bounds applied only when loading via *stock_name*.
    metric:
        Column name to rank by, descending. Default ``'sharpe_ratio'``.
        Any metric column returned by :func:`evaluate_params` is valid
        (``'cagr_pct'``, ``'return_pct'``, ``'calmar_ratio'``, etc.).
    constraint:
        Optional predicate filtering invalid combinations. If omitted,
        the strategy's default constraint from
        :data:`DEFAULT_CONSTRAINTS` is applied automatically (pass
        ``constraint=lambda p: True`` to disable filtering entirely).
    initial_cash, commission:
        Backtest configuration.
    n_jobs:
        ``1`` (default) runs sequentially. ``> 1`` or ``-1`` uses joblib's
        process-based backend — recommended only for grids with 50+
        combinations or the heavier ``combined_strategy`` (see module
        docstring for the benchmarking rationale).
    processed_dir:
        Used only with *stock_name*.

    Examples
    --------
    ::

        engine = GridSearchEngine(
            strategy_name="ema_crossover", stock_name="TCS",
            param_grid={"fast_period": [10, 20, 50], "slow_period": [100, 200]},
            metric="sharpe_ratio",
        )
        results_df = engine.run()
        print(engine.best_params)
    """

    def __init__(
        self,
        strategy_name: str,
        param_grid: Dict[str, List[Any]],
        stock_name: Optional[str]      = None,
        price_df: Optional[pd.DataFrame] = None,
        start_date: Optional[str]      = None,
        end_date: Optional[str]        = None,
        metric: str                    = "sharpe_ratio",
        constraint: Optional[Callable[[Dict[str, Any]], bool]] = None,
        initial_cash: float            = DEFAULT_INITIAL_CASH,
        commission: float              = DEFAULT_COMMISSION,
        n_jobs: int                    = 1,
        processed_dir: str             = PROCESSED_DIR,
    ) -> None:
        if price_df is None and stock_name is None:
            raise ValueError("Provide either 'stock_name' or 'price_df'.")

        self.strategy_name  = strategy_name
        self.strategy_class = get_strategy_class(strategy_name)
        self.param_grid     = param_grid
        self.stock_name     = stock_name
        self._price_df_arg  = price_df
        self.start_date     = start_date
        self.end_date       = end_date
        self.metric         = metric
        self.constraint     = (
            constraint if constraint is not None
            else DEFAULT_CONSTRAINTS.get(strategy_name)
        )
        self.initial_cash   = initial_cash
        self.commission     = commission
        self.n_jobs          = n_jobs
        self.processed_dir  = processed_dir

        self.results_: Optional[pd.DataFrame] = None
        self.n_failed_: int = 0

    # ── Data resolution ──────────────────────────────────────────────────────
    def _load_data(self) -> pd.DataFrame:
        """Return the price DataFrame to backtest against."""
        if self._price_df_arg is not None:
            return self._price_df_arg
        return load_price_data(
            self.stock_name, self.processed_dir, self.start_date, self.end_date
        )

    # ── Core run ─────────────────────────────────────────────────────────────
    def run(self) -> pd.DataFrame:
        """
        Execute the grid search.

        Returns
        -------
        pd.DataFrame
            One row per parameter combination, sorted by ``self.metric``
            descending, with a leading ``rank`` column (1 = best).

        Raises
        ------
        ValueError
            If ``self.metric`` is not among the computed metric columns,
            or if every combination fails.
        """
        df = self._load_data()
        combos = build_param_grid(self.param_grid, self.constraint)
        label_base = self.stock_name or "custom"

        logger.info(
            "GridSearch START  strategy=%s  data=%s  bars=%d  combos=%d  n_jobs=%d",
            self.strategy_name, label_base, len(df), len(combos), self.n_jobs,
        )

        if self.n_jobs == 1:
            rows = self._run_sequential(df, combos, label_base)
        else:
            rows = self._run_parallel(df, combos, label_base)

        self.n_failed_ = len(combos) - len(rows)
        if not rows:
            raise ValueError(
                "All parameter combinations failed during backtesting. "
                "Check the log output above for per-combination errors."
            )
        if self.n_failed_:
            logger.warning(
                "%d/%d combinations failed and were excluded from results.",
                self.n_failed_, len(combos),
            )

        results = pd.DataFrame(rows)
        if self.metric not in results.columns:
            raise ValueError(
                f"metric '{self.metric}' not found. Available: "
                f"{[c for c in results.columns if c not in self.param_grid]}"
            )

        sort_key = results[self.metric].replace(
            [np.inf, -np.inf], [_INF_SENTINEL, -_INF_SENTINEL]
        )
        results = (
            results.assign(_sort_key=sort_key)
            .sort_values("_sort_key", ascending=False)
            .drop(columns="_sort_key")
            .reset_index(drop=True)
        )
        results.insert(0, "rank", range(1, len(results) + 1))

        best = results.iloc[0]
        logger.info(
            "GridSearch DONE  best %s=%.4f  params=%s",
            self.metric, best[self.metric],
            {k: best[k] for k in self.param_grid.keys()},
        )

        self.results_ = results
        return results

    def _run_sequential(
        self, df: pd.DataFrame, combos: List[Dict[str, Any]], label_base: str
    ) -> List[Dict[str, Any]]:
        """Evaluate every combination one at a time, logging progress."""
        rows: List[Dict[str, Any]] = []
        checkpoint = max(1, len(combos) // 5)
        for i, params in enumerate(combos, start=1):
            try:
                row = evaluate_params(
                    self.strategy_class, df, params,
                    self.initial_cash, self.commission,
                    label=f"{label_base}_combo{i}",
                )
                rows.append(row)
            except Exception as exc:                            # noqa: BLE001
                logger.error("Combo %d/%d FAILED  params=%s  error=%s",
                            i, len(combos), params, exc)
            if i % checkpoint == 0 or i == len(combos):
                logger.info("Progress: %d/%d combinations evaluated.", i, len(combos))
        return rows

    def _run_parallel(
        self, df: pd.DataFrame, combos: List[Dict[str, Any]], label_base: str
    ) -> List[Dict[str, Any]]:
        """Evaluate combinations concurrently via joblib (loky backend)."""
        try:
            from joblib import Parallel, delayed
        except ImportError as exc:
            raise ImportError(
                "joblib is required for n_jobs != 1. "
                "Install with: pip install joblib --break-system-packages"
            ) from exc

        def _safe_eval(i: int, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            try:
                return evaluate_params(
                    self.strategy_class, df, params,
                    self.initial_cash, self.commission,
                    label=f"{label_base}_combo{i}",
                )
            except Exception as exc:                             # noqa: BLE001
                logger.error("Combo %d FAILED  params=%s  error=%s", i, params, exc)
                return None

        raw = Parallel(n_jobs=self.n_jobs, backend="loky")(
            delayed(_safe_eval)(i, p) for i, p in enumerate(combos, start=1)
        )
        return [r for r in raw if r is not None]

    # ── Accessors ────────────────────────────────────────────────────────────
    @property
    def best_params(self) -> Dict[str, Any]:
        """
        Best-performing parameter combination as a plain dict.

        Values are explicitly cast back to the native Python type used in
        the original search space (``int``, ``float``, or ``bool``).
        Building ``self.results_`` via ``pd.DataFrame(rows)`` upcasts plain
        Python ints/bools to ``numpy.int64``/``numpy.bool_`` scalars; several
        Backtrader internals (e.g. EMA period handling) reject those types
        outright, which matters because this dict is fed straight back into
        a *fresh* ``cerebro.addstrategy(strategy_class, **best_params)`` call
        by ``walk_forward.py``'s out-of-sample evaluation step.
        """
        if self.results_ is None:
            raise RuntimeError("Call .run() before accessing best_params.")
        top = self.results_.iloc[0]
        result: Dict[str, Any] = {}
        for k in self.param_grid.keys():
            native_type = type(self.param_grid[k][0])
            result[k] = native_type(top[k])
        return result

    @property
    def best_row(self) -> pd.Series:
        """Full result row (params + all metrics) for the top combination."""
        if self.results_ is None:
            raise RuntimeError("Call .run() before accessing best_row.")
        return self.results_.iloc[0]

    def save_results(self, path: Optional[str] = None) -> str:
        """
        Persist the full results table to CSV.

        Parameters
        ----------
        path:
            Destination path. Defaults to
            ``reports/optimization/grid_search_{strategy}_{label}.csv``.

        Returns
        -------
        str
            The path written to.

        Raises
        ------
        RuntimeError
            If called before :meth:`run`.
        """
        if self.results_ is None:
            raise RuntimeError("Call .run() before saving results.")
        if path is None:
            os.makedirs(OPTIMIZATION_DIR, exist_ok=True)
            label = self.stock_name or "custom"
            path = os.path.join(
                OPTIMIZATION_DIR, f"grid_search_{self.strategy_name}_{label}.csv"
            )
        self.results_.to_csv(path, index=False)
        logger.info("Grid search results saved → %s", path)
        return path


# ──────────────────────────────────────────────────────────────────────────────
# Convenience one-liner
# ──────────────────────────────────────────────────────────────────────────────
def grid_search(
    strategy_name: str,
    stock_name: str,
    param_grid: Optional[Dict[str, List[Any]]] = None,
    metric: str = "sharpe_ratio",
    **kwargs: Any,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run a grid search in one call.

    Parameters
    ----------
    strategy_name:
        Registry key (see :data:`DEFAULT_PARAM_GRIDS` for valid names).
    stock_name:
        Stock to backtest against.
    param_grid:
        Search space. If omitted, falls back to the project's
        :data:`DEFAULT_PARAM_GRIDS` for *strategy_name*.
    metric:
        Ranking metric (descending).
    **kwargs:
        Forwarded to :class:`GridSearchEngine` (e.g. ``n_jobs``,
        ``initial_cash``, ``start_date``).

    Returns
    -------
    (pd.DataFrame, dict)
        Full results table and the best parameter combination.

    Examples
    --------
    >>> results, best = grid_search("ema_crossover", "TCS")
    >>> best
    {'fast_period': 20, 'slow_period': 150}
    """
    if param_grid is None:
        param_grid = DEFAULT_PARAM_GRIDS.get(strategy_name)
        if param_grid is None:
            raise ValueError(
                f"No default param_grid for '{strategy_name}'; pass one explicitly."
            )
    engine = GridSearchEngine(
        strategy_name=strategy_name, stock_name=stock_name,
        param_grid=param_grid, metric=metric, **kwargs,
    )
    results = engine.run()
    return results, engine.best_params


# ──────────────────────────────────────────────────────────────────────────────
# Script entry-point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _root not in sys.path:
        sys.path.insert(0, _root)

    print("=" * 70)
    print("  Grid Search Engine — Demo")
    print("=" * 70)

    DISPLAY_COLS = [
        "rank", "return_pct", "cagr_pct", "sharpe_ratio", "sortino_ratio",
        "max_drawdown_pct", "total_trades", "win_rate_pct", "profit_factor",
    ]

    # ── 1. EMA Crossover grid search on TCS ─────────────────────────────────
    print("\n── EMA Crossover — TCS  (16 combos: fast∈{10,20,30,50} × "
          "slow∈{100,150,200,250}) ──")
    results_ema, best_ema = grid_search("ema_crossover", "TCS")
    param_cols = list(DEFAULT_PARAM_GRIDS["ema_crossover"].keys())
    print(results_ema[["rank", *param_cols, *DISPLAY_COLS[1:]]]
          .head(8).to_string(index=False))
    print(f"\n  Best params: {best_ema}")

    engine_ema = GridSearchEngine(
        strategy_name="ema_crossover", stock_name="TCS",
        param_grid=DEFAULT_PARAM_GRIDS["ema_crossover"],
    )
    engine_ema.run()
    saved_path = engine_ema.save_results()
    print(f"  Saved → {saved_path}")

    # ── 2. RSI Strategy grid search on TCS ──────────────────────────────────
    print("\n── RSI Strategy — TCS  (18 combos: oversold × overbought × "
          "trend_filter) ──")
    engine_rsi = GridSearchEngine(
        strategy_name="rsi_strategy", stock_name="TCS",
        param_grid=DEFAULT_PARAM_GRIDS["rsi_strategy"],
    )
    results_rsi = engine_rsi.run()
    param_cols_rsi = list(DEFAULT_PARAM_GRIDS["rsi_strategy"].keys())
    print(results_rsi[["rank", *param_cols_rsi, *DISPLAY_COLS[1:]]]
          .head(5).to_string(index=False))
    print(f"\n  Best params: {engine_rsi.best_params}")
    engine_rsi.save_results()

    # ── 3. Combined Strategy grid search on TCS ─────────────────────────────
    print("\n── Combined Strategy — TCS  (36 combos) ──")
    engine_combo = GridSearchEngine(
        strategy_name="combined_strategy", stock_name="TCS",
        param_grid=DEFAULT_PARAM_GRIDS["combined_strategy"],
    )
    results_combo = engine_combo.run()
    param_cols_combo = list(DEFAULT_PARAM_GRIDS["combined_strategy"].keys())
    print(results_combo[["rank", *param_cols_combo, *DISPLAY_COLS[1:]]]
          .head(5).to_string(index=False))
    print(f"\n  Best params: {engine_combo.best_params}")
    engine_combo.save_results()

    # ── 4. Cross-stock comparison using the same EMA grid ───────────────────
    print("\n── EMA Crossover — Best params across all 3 stocks ─────────────")
    cross_stock_rows = []
    for stock in ["TCS", "RELIANCE", "INFOSYS"]:
        eng = GridSearchEngine(
            strategy_name="ema_crossover", stock_name=stock,
            param_grid=DEFAULT_PARAM_GRIDS["ema_crossover"],
        )
        eng.run()
        top = eng.best_row
        cross_stock_rows.append({
            "stock": stock, **eng.best_params,
            "sharpe_ratio": top["sharpe_ratio"], "return_pct": top["return_pct"],
            "max_drawdown_pct": top["max_drawdown_pct"],
        })
    print(pd.DataFrame(cross_stock_rows).to_string(index=False))

    print(f"\n  All results saved to {OPTIMIZATION_DIR}/")
    print()
