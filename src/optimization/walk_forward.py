"""
walk_forward.py
================
Walk-forward validation engine for the Algorithmic Trading Strategy
Backtester — Phase 8.

Why walk-forward validation matters
-------------------------------------
``grid_search.py`` finds the parameters that performed *best on the
data it was given* — by construction, this is an in-sample (IS) result
and is optimistically biased: a wide-enough search will always find
*some* combination that looks great purely by chance, especially when
the strategy has 3-4 tunable parameters and the data covers a single
contiguous regime.

Walk-forward validation is the standard quant-research antidote: split
history into successive *train → optimise → test on unseen data* folds,
exactly mimicking how a real trader would have used the strategy —
parameters are *never* chosen with knowledge of the period they are
evaluated on.

::

    Fold 1:  Train [2018–2021]  →  optimise  →  Test [2022]  (unseen)
    Fold 2:  Train [2019–2022]  →  optimise  →  Test [2023]  (unseen)
    Fold 3:  Train [2020–2023]  →  optimise  →  Test [2024]  (unseen)

If out-of-sample (OOS) performance consistently tracks in-sample (IS)
performance, the strategy's edge is likely genuine. If OOS performance
collapses relative to IS, the grid search was overfitting noise.

Handling indicator warm-up without lookahead bias
----------------------------------------------------
A test fold may be as short as one calendar year. A 200-period EMA
needs 200 prior bars before it produces a single non-NaN value, which
would silently disable signal generation for the first ~10 months of
a 12-month test fold. The fix used here: prepend ``warmup_bars`` of
genuine pre-test history (data the strategy could legitimately have
seen by the test start date — no future information is used) so every
indicator is fully warmed up *before* the official test window begins.
The resulting equity curve is then **trimmed to the official test start
date and rebased to the initial capital**, so warm-up-period price
action never contaminates the reported out-of-sample metrics. Trade
statistics are scoped to the test window via :class:`TradeLogAnalyzer`'s
exact exit dates, applying the same trim.

Integration
------------
* Reuses :class:`GridSearchEngine` from ``grid_search.py`` for the
  in-sample optimisation step of every fold — no duplicated search logic.
* Reuses ``evaluate_params`` for the single out-of-sample backtest per fold.
* Reuses Phase 6's ``PerformanceAnalyzer`` for all OOS metric calculations.

Usage
-----
::

    python src/optimization/walk_forward.py
    from src.optimization.walk_forward import WalkForwardEngine
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import backtrader as bt

if not hasattr(pd.DataFrame, "iteritems"):
    pd.DataFrame.iteritems = pd.DataFrame.items

# ── Project-root bootstrap ──────────────────────────────────────────────────
# This module's class definitions (not just its __main__ demo) depend on
# src.optimization.grid_search, so the project root must be on sys.path
# before that import is attempted — bootstrapping only inside `__main__`
# (the pattern used in earlier phases' standalone demo scripts) would be
# too late here.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.optimization.grid_search import (
    DEFAULT_COMMISSION,
    DEFAULT_CONSTRAINTS,
    DEFAULT_INITIAL_CASH,
    DEFAULT_PARAM_GRIDS,
    DEFAULT_RISK_FREE,
    PROCESSED_DIR,
    GridSearchEngine,
    evaluate_params,
    get_strategy_class,
    load_price_data,
)

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
OPTIMIZATION_DIR: str   = os.path.join("reports", "optimization")
DEFAULT_TRAIN_YEARS: float = 4.0
DEFAULT_TEST_YEARS: float  = 1.0
DEFAULT_WARMUP_BARS: int   = 260   # ≈ 1 trading year — covers EMA-200, RSI-14, ATR-14

#: Threshold below which OOS/IS metric degradation is flagged as likely overfitting.
OVERFIT_WARNING_RATIO: float = 0.5


# ──────────────────────────────────────────────────────────────────────────────
# Value objects
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class WalkForwardFold:
    """
    Date boundaries for one walk-forward fold.

    Attributes
    ----------
    fold_id:      1-indexed fold number.
    train_start, train_end:   Inclusive training window.
    test_start, test_end:     Inclusive, strictly-out-of-sample test window.
    """
    fold_id:     int
    train_start: pd.Timestamp
    train_end:   pd.Timestamp
    test_start:  pd.Timestamp
    test_end:    pd.Timestamp

    def __str__(self) -> str:
        return (
            f"Fold {self.fold_id}: "
            f"train [{self.train_start.date()} → {self.train_end.date()}]  "
            f"test [{self.test_start.date()} → {self.test_end.date()}]"
        )


@dataclass
class FoldResult:
    """
    Complete in-sample + out-of-sample outcome for one fold.

    Attributes
    ----------
    fold:                The WalkForwardFold definition.
    best_params:         Parameters selected by in-sample grid search.
    train_metric_value:  The chosen metric's value, in-sample (train period).
    train_n_combos:      How many parameter combinations were evaluated.
    test_metrics:        Dict of OOS metrics (return_pct, cagr_pct, sharpe_ratio,
                          sortino_ratio, calmar_ratio, max_drawdown_pct, vol_pct,
                          total_trades, win_rate_pct, profit_factor).
    degradation_pct:     ``(1 - test_metric/train_metric) * 100`` — positive
                          means OOS underperformed IS by that percentage.
    """
    fold:               WalkForwardFold
    best_params:         Dict[str, Any]
    train_metric_value:  float
    train_n_combos:      int
    test_metrics:        Dict[str, Any] = field(default_factory=dict)
    degradation_pct:      Optional[float] = field(default=None)

    def to_row(self) -> Dict[str, Any]:
        """Flatten into one CSV-friendly row."""
        return {
            "fold_id":            self.fold.fold_id,
            "train_start":        self.fold.train_start.date(),
            "train_end":          self.fold.train_end.date(),
            "test_start":         self.fold.test_start.date(),
            "test_end":           self.fold.test_end.date(),
            "best_params":        json.dumps(self.best_params, default=str),
            "train_metric":       round(self.train_metric_value, 4),
            "train_n_combos":     self.train_n_combos,
            "test_return_pct":    self.test_metrics.get("return_pct"),
            "test_cagr_pct":      self.test_metrics.get("cagr_pct"),
            "test_vol_pct":       self.test_metrics.get("vol_pct"),
            "test_sharpe":        self.test_metrics.get("sharpe_ratio"),
            "test_sortino":       self.test_metrics.get("sortino_ratio"),
            "test_calmar":        self.test_metrics.get("calmar_ratio"),
            "test_max_dd_pct":    self.test_metrics.get("max_drawdown_pct"),
            "test_total_trades":  self.test_metrics.get("total_trades"),
            "test_win_rate_pct":  self.test_metrics.get("win_rate_pct"),
            "test_profit_factor": self.test_metrics.get("profit_factor"),
            "degradation_pct":    (
                round(self.degradation_pct, 1) if self.degradation_pct is not None else None
            ),
        }

    def __str__(self) -> str:
        m = self.test_metrics
        flag = ""
        if self.degradation_pct is not None and self.degradation_pct > OVERFIT_WARNING_RATIO * 100:
            flag = "  ⚠ possible overfit"
        return (
            f"{self.fold}\n"
            f"    best_params={self.best_params}  (IS metric={self.train_metric_value:.3f}, "
            f"{self.train_n_combos} combos tested)\n"
            f"    OOS: return={m.get('return_pct', 0):+.2f}%  "
            f"sharpe={m.get('sharpe_ratio', 0):.3f}  "
            f"max_dd={m.get('max_drawdown_pct', 0):.2f}%  "
            f"trades={m.get('total_trades', 0)}  "
            f"win_rate={m.get('win_rate_pct', 0):.1f}%{flag}"
        )


@dataclass
class WalkForwardResult:
    """
    Aggregated outcome of an entire walk-forward run.

    Attributes
    ----------
    strategy_name:  Registry key of the strategy tested.
    stock_name:     Stock tested.
    window_type:    ``'rolling'`` or ``'expanding'`` — included so that
                     runs with different window configurations for the
                     same strategy/stock don't collide on output filenames.
    folds:          One :class:`FoldResult` per fold, in chronological order.
    fold_table:     Flattened DataFrame (one row per fold) — CSV-ready.
    summary:        Dict of mean/std/min/max across all folds' OOS metrics.
    """
    strategy_name: str
    stock_name:    str
    window_type:   str
    folds:         List[FoldResult]
    fold_table:    pd.DataFrame
    summary:       Dict[str, float]

    def report(self) -> str:
        """Formatted console report covering every fold plus the summary."""
        lines = [
            "=" * 70,
            f"  Walk-Forward Validation — {self.strategy_name} on {self.stock_name} "
            f"({self.window_type} window)",
            "=" * 70,
        ]
        for fr in self.folds:
            lines.append(f"\n{fr}")
        lines.append("\n" + "-" * 70)
        lines.append("  OUT-OF-SAMPLE SUMMARY (across all folds)")
        lines.append("-" * 70)
        for key, val in self.summary.items():
            lines.append(f"    {key:<28}: {val:>+10.3f}" if isinstance(val, float) else
                         f"    {key:<28}: {val:>10}")
        return "\n".join(lines)

    def save(self, out_dir: str = OPTIMIZATION_DIR) -> Dict[str, str]:
        """
        Persist the fold table and summary to CSV.

        Parameters
        ----------
        out_dir:
            Destination directory.

        Returns
        -------
        dict
            ``{'folds': path, 'summary': path}``.
        """
        os.makedirs(out_dir, exist_ok=True)
        base = f"walk_forward_{self.strategy_name}_{self.stock_name}_{self.window_type}"
        fold_path    = os.path.join(out_dir, f"{base}_folds.csv")
        summary_path = os.path.join(out_dir, f"{base}_summary.csv")

        self.fold_table.to_csv(fold_path, index=False)
        pd.Series(self.summary, name="value").to_csv(summary_path)

        logger.info("Walk-forward fold table  saved → %s", fold_path)
        logger.info("Walk-forward summary     saved → %s", summary_path)
        return {"folds": fold_path, "summary": summary_path}


# ──────────────────────────────────────────────────────────────────────────────
# Fold generation
# ──────────────────────────────────────────────────────────────────────────────
def _years_to_offset(years: float) -> pd.DateOffset:
    """Convert a (possibly fractional) year count into a calendar-safe DateOffset."""
    months = round(years * 12)
    if months <= 0:
        raise ValueError(f"years must be > 0 (resolved to {months} months from {years}).")
    return pd.DateOffset(months=months)


def generate_folds(
    data_start: pd.Timestamp,
    data_end: pd.Timestamp,
    train_years: float = DEFAULT_TRAIN_YEARS,
    test_years: float  = DEFAULT_TEST_YEARS,
    step_years: Optional[float] = None,
    window_type: str    = "rolling",
) -> List[WalkForwardFold]:
    """
    Generate successive train/test fold boundaries spanning the available data.

    Parameters
    ----------
    data_start, data_end:
        Full available date range (typically the loaded price series' bounds).
    train_years:
        Training window length in years. Default ``4.0``.
    test_years:
        Test window length in years. Default ``1.0``.
    step_years:
        How far the window advances between folds. Defaults to ``test_years``
        (non-overlapping test windows — the standard setup). A smaller value
        produces overlapping, more numerous, slightly correlated test folds.
    window_type:
        ``'rolling'`` — both train_start and train_end advance each fold
        (fixed training-window length).
        ``'expanding'`` — train_start stays fixed at *data_start*; train_end
        (and hence training length) grows each fold.

    Returns
    -------
    List[WalkForwardFold]
        Chronologically ordered; empty if the data span is too short for
        even one fold.

    Raises
    ------
    ValueError
        If *window_type* is not ``'rolling'`` or ``'expanding'``.
    """
    if window_type not in ("rolling", "expanding"):
        raise ValueError(f"window_type must be 'rolling' or 'expanding', got {window_type!r}")

    step_years   = step_years if step_years is not None else test_years
    train_offset = _years_to_offset(train_years)
    test_offset  = _years_to_offset(test_years)
    step_offset  = _years_to_offset(step_years)

    folds: List[WalkForwardFold] = []
    i = 0
    while True:
        if window_type == "rolling":
            train_start    = data_start + i * step_offset
            train_end_excl = train_start + train_offset
        else:  # expanding
            train_start    = data_start
            train_end_excl = data_start + train_offset + i * step_offset

        test_start = train_end_excl
        if test_start > data_end or train_start >= train_end_excl:
            break

        test_end_incl  = min(test_start + test_offset - pd.Timedelta(days=1), data_end)
        train_end_incl = train_end_excl - pd.Timedelta(days=1)

        folds.append(WalkForwardFold(
            fold_id=i + 1, train_start=train_start, train_end=train_end_incl,
            test_start=test_start, test_end=test_end_incl,
        ))
        i += 1
        if test_end_incl >= data_end:
            break  # final fold reached the end of available data

    return folds


# ──────────────────────────────────────────────────────────────────────────────
# Engine
# ──────────────────────────────────────────────────────────────────────────────
class WalkForwardEngine:
    """
    Runs rolling or expanding walk-forward validation for one strategy/stock.

    Parameters
    ----------
    strategy_name:
        Registry key: ``'ema_crossover'``, ``'rsi_strategy'``, or
        ``'combined_strategy'``.
    stock_name:
        Equity to validate against (loads from ``data/processed/``).
    param_grid:
        Search space for the in-sample optimisation step of every fold.
        Defaults to the project's :data:`DEFAULT_PARAM_GRIDS` for
        *strategy_name* if omitted.
    train_years, test_years, step_years, window_type:
        Forwarded to :func:`generate_folds`.
    warmup_bars:
        Bars of genuine pre-test history fed to the strategy so indicators
        are warmed up by the official test start date (see module docstring).
    metric:
        Ranking metric used both for in-sample selection and for the
        cross-fold summary. Default ``'sharpe_ratio'``.
    constraint:
        Optional override of the strategy's default parameter constraint.
    initial_cash, commission:
        Backtest configuration, applied identically to every fold.
    n_jobs:
        Forwarded to the internal :class:`GridSearchEngine` for each fold's
        in-sample search.
    processed_dir:
        Source directory for processed price CSVs.

    Examples
    --------
    ::

        engine = WalkForwardEngine(
            strategy_name="ema_crossover", stock_name="TCS",
            train_years=4, test_years=1, window_type="rolling",
        )
        result = engine.run()
        print(result.report())
        result.save()
    """

    def __init__(
        self,
        strategy_name: str,
        stock_name: str,
        param_grid: Optional[Dict[str, List[Any]]] = None,
        train_years: float  = DEFAULT_TRAIN_YEARS,
        test_years: float   = DEFAULT_TEST_YEARS,
        step_years: Optional[float] = None,
        window_type: str     = "rolling",
        warmup_bars: int      = DEFAULT_WARMUP_BARS,
        metric: str           = "sharpe_ratio",
        constraint: Optional[Callable[[Dict[str, Any]], bool]] = None,
        initial_cash: float   = DEFAULT_INITIAL_CASH,
        commission: float     = DEFAULT_COMMISSION,
        n_jobs: int            = 1,
        processed_dir: str    = PROCESSED_DIR,
    ) -> None:
        self.strategy_name  = strategy_name
        self.strategy_class = get_strategy_class(strategy_name)
        self.stock_name     = stock_name
        self.param_grid      = param_grid or DEFAULT_PARAM_GRIDS.get(strategy_name)
        if self.param_grid is None:
            raise ValueError(
                f"No default param_grid for '{strategy_name}'; pass one explicitly."
            )
        self.train_years    = train_years
        self.test_years      = test_years
        self.step_years      = step_years
        self.window_type     = window_type
        self.warmup_bars      = warmup_bars
        self.metric           = metric
        self.constraint        = (
            constraint if constraint is not None
            else DEFAULT_CONSTRAINTS.get(strategy_name)
        )
        self.initial_cash    = initial_cash
        self.commission      = commission
        self.n_jobs            = n_jobs
        self.processed_dir   = processed_dir

    # ── OOS evaluation for one fold ──────────────────────────────────────────
    def _evaluate_oos(
        self,
        full_df: pd.DataFrame,
        fold: WalkForwardFold,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Run the single out-of-sample backtest for *fold* using *params*.

        Feeds ``warmup_bars`` of genuine pre-test history plus the test
        window itself to Backtrader (so indicators are valid from day one
        of the test period), then trims and rebases the resulting equity
        curve to the official test window before computing metrics — see
        the module docstring for the full rationale.

        Parameters
        ----------
        full_df:
            Complete price history (not date-limited) so warm-up bars are
            available even for the very first fold.
        fold:
            The fold whose test window is being evaluated.
        params:
            The in-sample-selected parameter combination.

        Returns
        -------
        dict
            OOS metrics: ``return_pct``, ``cagr_pct``, ``vol_pct``,
            ``sharpe_ratio``, ``sortino_ratio``, ``calmar_ratio``,
            ``max_drawdown_pct``, ``total_trades``, ``win_rate_pct``,
            ``profit_factor``.
        """
        from src.analytics.performance_metrics import PerformanceAnalyzer
        from src.optimization.grid_search import (
            _reconstruct_equity_curve, _summarise_trades,
        )

        pre_test = full_df.loc[:fold.test_start - pd.Timedelta(days=1)]
        actual_warmup = min(self.warmup_bars, len(pre_test))
        if actual_warmup < self.warmup_bars:
            logger.warning(
                "Fold %d: only %d warm-up bars available (wanted %d) — "
                "earliest indicator readings in this fold may be NaN.",
                fold.fold_id, actual_warmup, self.warmup_bars,
            )
        warmup_slice = pre_test.tail(actual_warmup) if actual_warmup else pre_test.iloc[0:0]
        test_slice   = full_df.loc[fold.test_start:fold.test_end]
        feed_df      = pd.concat([warmup_slice, test_slice]).sort_index()

        raw = evaluate_params(
            self.strategy_class, feed_df, params,
            self.initial_cash, self.commission,
            label=f"{self.stock_name}_fold{fold.fold_id}_OOS",
            return_curve=True,
        )
        full_equity   = raw.pop("_equity_curve")
        full_tradelog = raw.pop("_trade_log")

        # ── Trim & rebase the equity curve to the official test window ──────────
        oos_equity = full_equity.loc[fold.test_start:fold.test_end]
        if oos_equity.empty:
            raise ValueError(
                f"Fold {fold.fold_id}: no equity data in test window "
                f"[{fold.test_start.date()}, {fold.test_end.date()}]."
            )
        oos_equity = oos_equity / oos_equity.iloc[0] * self.initial_cash

        # ── Scope trade stats to trades that closed within the test window ──────
        oos_trades = [
            t for t in full_tradelog
            if pd.Timestamp(t["exit_date"]) >= fold.test_start
        ]
        trade_stats = _summarise_trades(oos_trades)

        perf = PerformanceAnalyzer(
            label=f"fold{fold.fold_id}_OOS", risk_free_rate=DEFAULT_RISK_FREE
        ).compute(oos_equity)

        return {
            "return_pct":       perf.total_return_pct,
            "cagr_pct":          perf.cagr_pct,
            "vol_pct":           perf.vol_pct,
            "sharpe_ratio":      perf.sharpe,
            "sortino_ratio":     perf.sortino,
            "calmar_ratio":      perf.calmar,
            "max_drawdown_pct":  perf.max_drawdown_pct,
            **trade_stats,
        }

    # ── Main run loop ────────────────────────────────────────────────────────
    def run(self) -> WalkForwardResult:
        """
        Execute walk-forward validation across every generated fold.

        Returns
        -------
        WalkForwardResult

        Raises
        ------
        ValueError
            If the available data span produces zero folds (e.g.
            ``train_years + test_years`` exceeds the dataset length).
        """
        full_df = load_price_data(self.stock_name, self.processed_dir)
        data_start, data_end = full_df.index[0], full_df.index[-1]

        folds = generate_folds(
            data_start, data_end,
            self.train_years, self.test_years, self.step_years, self.window_type,
        )
        if not folds:
            raise ValueError(
                f"No folds generated for {data_start.date()}→{data_end.date()} with "
                f"train={self.train_years}y, test={self.test_years}y. "
                "Reduce train_years/test_years or use a longer dataset."
            )

        logger.info(
            "WalkForward START  strategy=%s  stock=%s  window=%s  "
            "train=%.1fy  test=%.1fy  folds=%d",
            self.strategy_name, self.stock_name, self.window_type,
            self.train_years, self.test_years, len(folds),
        )

        fold_results: List[FoldResult] = []
        for fold in folds:
            logger.info("─── %s ───", fold)
            try:
                train_df = full_df.loc[fold.train_start:fold.train_end]

                gs = GridSearchEngine(
                    strategy_name=self.strategy_name,
                    price_df=train_df,
                    param_grid=self.param_grid,
                    metric=self.metric,
                    constraint=self.constraint,
                    initial_cash=self.initial_cash,
                    commission=self.commission,
                    n_jobs=self.n_jobs,
                )
                gs.run()
                best_params  = gs.best_params
                train_metric = float(gs.best_row[self.metric])

                oos_metrics = self._evaluate_oos(full_df, fold, best_params)
                test_metric = float(oos_metrics.get(self.metric, 0.0) or 0.0)

                degradation = None
                if abs(train_metric) > 1e-9:
                    degradation = (1 - test_metric / train_metric) * 100

                fr = FoldResult(
                    fold=fold, best_params=best_params,
                    train_metric_value=train_metric,
                    train_n_combos=len(gs.results_),
                    test_metrics=oos_metrics,
                    degradation_pct=degradation,
                )
                fold_results.append(fr)
                logger.info(str(fr).replace("\n", " | "))

                if degradation is not None and degradation > OVERFIT_WARNING_RATIO * 100:
                    logger.warning(
                        "Fold %d: OOS %s degraded %.0f%% vs in-sample — "
                        "possible overfitting in this fold.",
                        fold.fold_id, self.metric, degradation,
                    )
            except Exception as exc:                                # noqa: BLE001
                logger.error("Fold %d FAILED: %s", fold.fold_id, exc)

        if not fold_results:
            raise ValueError("Every fold failed — see log output above for details.")

        fold_table = pd.DataFrame([fr.to_row() for fr in fold_results])
        summary    = self._summarise(fold_results)

        logger.info(
            "WalkForward DONE  folds_ok=%d/%d  mean_OOS_sharpe=%.3f  "
            "mean_OOS_return=%.2f%%",
            len(fold_results), len(folds),
            summary.get("oos_sharpe_mean", 0.0), summary.get("oos_return_pct_mean", 0.0),
        )

        return WalkForwardResult(
            strategy_name=self.strategy_name, stock_name=self.stock_name,
            window_type=self.window_type,
            folds=fold_results, fold_table=fold_table, summary=summary,
        )

    # ── Cross-fold aggregation ───────────────────────────────────────────────
    @staticmethod
    def _summarise(fold_results: List[FoldResult]) -> Dict[str, float]:
        """
        Aggregate OOS metrics across folds and compute an overfitting signal.

        The **efficiency ratio** = mean(OOS metric) / mean(IS metric).
        A ratio near 1.0 indicates the strategy's edge persists out of
        sample; a ratio near 0 (or negative) indicates the in-sample
        optimum did not generalise — the classic overfitting signature.

        Parameters
        ----------
        fold_results:
            All successfully completed folds.

        Returns
        -------
        dict
        """
        returns = [fr.test_metrics.get("return_pct", 0.0) for fr in fold_results]
        sharpes = [fr.test_metrics.get("sharpe_ratio", 0.0) for fr in fold_results]
        max_dds = [fr.test_metrics.get("max_drawdown_pct", 0.0) for fr in fold_results]
        win_rts = [fr.test_metrics.get("win_rate_pct", 0.0) for fr in fold_results]
        is_vals = [fr.train_metric_value for fr in fold_results]

        mean_is = float(np.mean(is_vals)) if is_vals else 0.0
        mean_oos_sharpe = float(np.mean(sharpes)) if sharpes else 0.0
        efficiency = (mean_oos_sharpe / mean_is) if abs(mean_is) > 1e-9 else 0.0

        return {
            "n_folds":               len(fold_results),
            "oos_return_pct_mean":   round(float(np.mean(returns)), 2),
            "oos_return_pct_std":    round(float(np.std(returns)), 2),
            "oos_sharpe_mean":       round(mean_oos_sharpe, 3),
            "oos_sharpe_std":        round(float(np.std(sharpes)), 3),
            "oos_max_dd_mean":       round(float(np.mean(max_dds)), 2),
            "oos_max_dd_worst":      round(float(np.min(max_dds)), 2),
            "oos_win_rate_mean":     round(float(np.mean(win_rts)), 1),
            "in_sample_metric_mean": round(mean_is, 3),
            "is_to_oos_efficiency":  round(efficiency, 3),
            "pct_folds_profitable":  round(
                sum(1 for r in returns if r > 0) / len(returns) * 100, 1
            ),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Script entry-point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("  Walk-Forward Validation Engine — Demo")
    print("=" * 70)

    # ── 1. Rolling-window walk-forward: EMA Crossover on TCS ────────────────
    print("\n── EMA Crossover — TCS — Rolling 4yr-train / 1yr-test ───────────")
    engine = WalkForwardEngine(
        strategy_name="ema_crossover", stock_name="TCS",
        train_years=4.0, test_years=1.0, window_type="rolling",
    )
    result = engine.run()
    print(f"\n{result.report()}")
    paths = result.save()
    print(f"\n  Saved fold table   → {paths['folds']}")
    print(f"  Saved summary      → {paths['summary']}")

    # ── 2. Expanding-window walk-forward for comparison ──────────────────────
    print("\n\n── EMA Crossover — TCS — Expanding window (for comparison) ──────")
    engine_exp = WalkForwardEngine(
        strategy_name="ema_crossover", stock_name="TCS",
        train_years=4.0, test_years=1.0, window_type="expanding",
    )
    result_exp = engine_exp.run()
    print(f"\n{result_exp.report()}")
    result_exp.save()

    # ── 3. Walk-forward on RSI Strategy ──────────────────────────────────────
    print("\n\n── RSI Strategy — TCS — Rolling 4yr-train / 1yr-test ─────────────")
    engine_rsi = WalkForwardEngine(
        strategy_name="rsi_strategy", stock_name="TCS",
        train_years=4.0, test_years=1.0, window_type="rolling",
    )
    result_rsi = engine_rsi.run()
    print(f"\n{result_rsi.report()}")
    result_rsi.save()

    # ── Final cross-strategy comparison table ────────────────────────────────
    print("\n\n" + "=" * 70)
    print("  Cross-Strategy Walk-Forward Comparison (TCS)")
    print("=" * 70)
    comparison = pd.DataFrame({
        "EMA Crossover (rolling)":   result.summary,
        "EMA Crossover (expanding)": result_exp.summary,
        "RSI Strategy (rolling)":    result_rsi.summary,
    })
    print(comparison.to_string())
    print()
