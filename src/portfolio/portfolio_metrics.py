"""
portfolio_metrics.py
====================
Portfolio analytics engine for the Algorithmic Trading Strategy
Backtester.

Metrics computed
----------------
**Return metrics**
* Total return (%)
* CAGR — Compound Annual Growth Rate
* Best / worst month
* % of positive months

**Risk metrics**
* Annualised volatility (σ)
* Maximum drawdown (%)
* Max drawdown duration (days)
* Value at Risk 95 % (VaR)
* Conditional VaR / CVaR 95 %
* Skewness and kurtosis of daily returns

**Risk-adjusted metrics**
* Sharpe Ratio  (risk-free = 6 %, annualised)
* Sortino Ratio (downside deviation only)
* Calmar Ratio  (CAGR / |max drawdown|)

**Trade-based metrics**  (when a trade log is provided)
* Win rate (%)
* Profit factor (sum of wins / sum of losses)
* Average winning / losing trade (₹)
* Expectancy per trade (₹)

**Rolling analysis**
* Rolling 252-day annualised return
* Rolling 252-day Sharpe ratio
* Drawdown curve

Architecture
------------
* :class:`PortfolioMetrics` — frozen dataclass holding every metric.
* Pure calculation functions (``calculate_cagr``, ``calculate_sharpe``,
  etc.) — independently testable, importable.
* :class:`MetricsEngine` — facade: one ``compute()`` call returns
  a complete :class:`PortfolioMetrics`.
* :func:`compare_strategies` — runs ``compute()`` for multiple value
  series and returns a comparison ``pd.DataFrame``.

Integration
-----------
Accepts the ``portfolio_value_series`` from ``rebalance.py`` and
the ``trade_log`` from any Phase 3 strategy's Backtrader run.

Usage
-----
::

    python src/portfolio/portfolio_metrics.py
    from src.portfolio.portfolio_metrics import MetricsEngine
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
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
TRADING_DAYS_PER_YEAR: int   = 252
RISK_FREE_RATE_ANNUAL: float = 0.06   # 6 % — approximate Indian repo rate
VAR_CONFIDENCE: float        = 0.95   # 95th percentile for VaR
PROCESSED_DIR: str           = os.path.join("data", "processed")


# ──────────────────────────────────────────────────────────────────────────────
# Value object
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class PortfolioMetrics:
    """
    Comprehensive set of portfolio performance metrics.

    All percentage fields are expressed as plain percentages
    (e.g. ``12.5`` means 12.5 %, not 0.125).

    Attributes
    ----------
    label : str
        Strategy / portfolio name.
    start_date : str
    end_date : str
    trading_days : int
    initial_value : float
    final_value : float
    total_return_pct : float
    cagr_pct : float
    annualised_vol_pct : float
    sharpe_ratio : float
    sortino_ratio : float
    calmar_ratio : float
    max_drawdown_pct : float
    max_drawdown_duration_days : int
    var_95_pct : float
    cvar_95_pct : float
    skewness : float
    kurtosis : float
    best_month_pct : float
    worst_month_pct : float
    positive_months_pct : float
    win_rate_pct : float
    profit_factor : float
    total_trades : int
    avg_win_inr : float
    avg_loss_inr : float
    expectancy_inr : float
    """
    label:                     str
    start_date:                str
    end_date:                  str
    trading_days:              int
    initial_value:             float
    final_value:               float
    total_return_pct:          float
    cagr_pct:                  float
    annualised_vol_pct:        float
    sharpe_ratio:              float
    sortino_ratio:             float
    calmar_ratio:              float
    max_drawdown_pct:          float
    max_drawdown_duration_days: int
    var_95_pct:                float
    cvar_95_pct:               float
    skewness:                  float
    kurtosis:                  float
    best_month_pct:            float
    worst_month_pct:           float
    positive_months_pct:       float
    win_rate_pct:              float   = 0.0
    profit_factor:             float   = 0.0
    total_trades:              int     = 0
    avg_win_inr:               float   = 0.0
    avg_loss_inr:              float   = 0.0
    expectancy_inr:            float   = 0.0

    def to_dict(self) -> Dict:
        """Return all metrics as a plain dict."""
        return {f: getattr(self, f) for f in self.__dataclass_fields__}  # type: ignore[attr-defined]

    def to_series(self) -> pd.Series:
        """Return metrics as a named pd.Series."""
        return pd.Series(self.to_dict(), name=self.label)

    def report(self) -> str:
        """Return a formatted multi-line performance report."""
        SEP = "─" * 52
        lines = [
            f"  ╔{'═'*50}╗",
            f"  ║  Portfolio Report: {self.label:<31}║",
            f"  ║  Period: {self.start_date} → {self.end_date} ({self.trading_days} days){' '*4}║",
            f"  ╠{'═'*50}╣",
            f"  ║  RETURN METRICS{' '*35}║",
            f"  ║    Total return     : {self.total_return_pct:>+8.2f}%{' '*22}║",
            f"  ║    CAGR             : {self.cagr_pct:>+8.2f}%{' '*22}║",
            f"  ║    Best month       : {self.best_month_pct:>+8.2f}%{' '*22}║",
            f"  ║    Worst month      : {self.worst_month_pct:>+8.2f}%{' '*22}║",
            f"  ║    Positive months  : {self.positive_months_pct:>8.1f}%{' '*22}║",
            f"  ╠{'═'*50}╣",
            f"  ║  RISK METRICS{' '*37}║",
            f"  ║    Annualised vol   : {self.annualised_vol_pct:>8.2f}%{' '*22}║",
            f"  ║    Max drawdown     : {self.max_drawdown_pct:>+8.2f}%{' '*22}║",
            f"  ║    Max DD duration  : {self.max_drawdown_duration_days:>8} days{' '*19}║",
            f"  ║    VaR (95%)        : {self.var_95_pct:>+8.2f}%{' '*22}║",
            f"  ║    CVaR (95%)       : {self.cvar_95_pct:>+8.2f}%{' '*22}║",
            f"  ║    Skewness         : {self.skewness:>+8.3f}{' '*23}║",
            f"  ║    Kurtosis         : {self.kurtosis:>+8.3f}{' '*23}║",
            f"  ╠{'═'*50}╣",
            f"  ║  RISK-ADJUSTED{' '*37}║",
            f"  ║    Sharpe ratio     : {self.sharpe_ratio:>+8.3f}{' '*23}║",
            f"  ║    Sortino ratio    : {self.sortino_ratio:>+8.3f}{' '*23}║",
            f"  ║    Calmar ratio     : {self.calmar_ratio:>+8.3f}{' '*23}║",
        ]
        if self.total_trades:
            lines += [
                f"  ╠{'═'*50}╣",
                f"  ║  TRADE METRICS{' '*37}║",
                f"  ║    Total trades     : {self.total_trades:>8}{' '*23}║",
                f"  ║    Win rate         : {self.win_rate_pct:>8.1f}%{' '*22}║",
                f"  ║    Profit factor    : {self.profit_factor:>8.2f}×{' '*22}║",
                f"  ║    Avg win          : ₹{self.avg_win_inr:>8,.0f}{' '*22}║",
                f"  ║    Avg loss         : ₹{self.avg_loss_inr:>8,.0f}{' '*22}║",
                f"  ║    Expectancy       : ₹{self.expectancy_inr:>+8,.0f}{' '*22}║",
            ]
        lines.append(f"  ╚{'═'*50}╝")
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Pure calculation functions
# ──────────────────────────────────────────────────────────────────────────────
def calculate_returns(value_series: pd.Series) -> pd.Series:
    """
    Compute daily percentage returns from a portfolio value series.

    Parameters
    ----------
    value_series:
        Portfolio equity curve indexed by date.

    Returns
    -------
    pd.Series
        Daily returns (fractional, not percentages).
    """
    return value_series.pct_change().dropna()


def calculate_cagr(value_series: pd.Series) -> float:
    """
    Compound Annual Growth Rate.

    ::

        CAGR = (final / initial)^(252 / n_days) − 1

    Parameters
    ----------
    value_series:
        Portfolio equity curve.

    Returns
    -------
    float
        CAGR as a decimal (e.g. 0.12 = 12 %).
    """
    n = len(value_series)
    if n < 2:
        return 0.0
    start = value_series.iloc[0]
    end   = value_series.iloc[-1]
    if start <= 0:
        return 0.0
    years = n / TRADING_DAYS_PER_YEAR
    return (end / start) ** (1.0 / years) - 1.0


def calculate_annualised_volatility(returns: pd.Series) -> float:
    """
    Annualised standard deviation of daily returns.

    ``σ_annual = σ_daily × √252``

    Parameters
    ----------
    returns:
        Daily fractional returns.

    Returns
    -------
    float
        Annualised volatility as a decimal.
    """
    if len(returns) < 2:
        return 0.0
    return float(returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def calculate_sharpe(
    returns: pd.Series,
    risk_free_annual: float = RISK_FREE_RATE_ANNUAL,
) -> float:
    """
    Annualised Sharpe Ratio.

    ::

        Sharpe = (CAGR_returns − r_f) / σ_annual

    Parameters
    ----------
    returns:
        Daily fractional returns.
    risk_free_annual:
        Annual risk-free rate (default 6 % for Indian context).

    Returns
    -------
    float
        Sharpe ratio.  Positive values indicate excess return per unit risk.
    """
    if len(returns) < 2:
        return 0.0
    vol = calculate_annualised_volatility(returns)
    if vol == 0:
        return 0.0
    ann_return = (1 + returns.mean()) ** TRADING_DAYS_PER_YEAR - 1
    return (ann_return - risk_free_annual) / vol


def calculate_sortino(
    returns: pd.Series,
    risk_free_annual: float = RISK_FREE_RATE_ANNUAL,
) -> float:
    """
    Sortino Ratio — penalises *downside* volatility only.

    ::

        Sortino = (CAGR_returns − r_f) / σ_downside

    where ``σ_downside`` uses only negative daily returns.

    Parameters
    ----------
    returns:
        Daily fractional returns.
    risk_free_annual:
        Annual risk-free rate.

    Returns
    -------
    float
        Sortino ratio.  Higher is better.
    """
    if len(returns) < 2:
        return 0.0
    daily_rf       = risk_free_annual / TRADING_DAYS_PER_YEAR
    excess_returns = returns - daily_rf
    downside       = excess_returns[excess_returns < 0]
    if len(downside) == 0:
        return float("inf")
    downside_vol   = float(downside.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
    if downside_vol == 0:
        return 0.0
    ann_return     = (1 + returns.mean()) ** TRADING_DAYS_PER_YEAR - 1
    return (ann_return - risk_free_annual) / downside_vol


def calculate_max_drawdown(
    value_series: pd.Series,
) -> Tuple[float, int, pd.Series]:
    """
    Maximum drawdown, its duration, and the full drawdown curve.

    Max drawdown is the largest peak-to-trough decline:

    ::

        DD_t = (V_t − peak_t) / peak_t,   peak_t = max(V_s for s ≤ t)

    Parameters
    ----------
    value_series:
        Portfolio equity curve.

    Returns
    -------
    (max_drawdown, duration_days, drawdown_curve)
        ``max_drawdown`` is a negative decimal (e.g. -0.25 = -25 %).
        ``duration_days`` is the number of calendar bars in the longest
        underwater period.
        ``drawdown_curve`` is a pd.Series of daily drawdown values.
    """
    rolling_peak  = value_series.cummax()
    drawdown      = (value_series - rolling_peak) / rolling_peak
    max_dd        = float(drawdown.min())

    # Duration of the longest drawdown period
    underwater    = drawdown < 0
    max_duration  = 0
    current_dur   = 0
    for uw in underwater:
        if uw:
            current_dur += 1
            max_duration = max(max_duration, current_dur)
        else:
            current_dur = 0

    return max_dd, max_duration, drawdown


def calculate_var(returns: pd.Series, confidence: float = VAR_CONFIDENCE) -> float:
    """
    Historical Value at Risk at *confidence* level.

    Returns the daily return below which losses fall with probability
    ``1 − confidence``.  E.g. VaR(95%) = -2.1 % means there is a
    5 % chance of losing more than 2.1 % in a single day.

    Parameters
    ----------
    returns:
        Daily fractional returns.
    confidence:
        Confidence level (default 0.95).

    Returns
    -------
    float
        VaR as a decimal (negative = loss).
    """
    if len(returns) == 0:
        return 0.0
    return float(np.percentile(returns, (1 - confidence) * 100))


def calculate_cvar(returns: pd.Series, confidence: float = VAR_CONFIDENCE) -> float:
    """
    Conditional VaR (Expected Shortfall) at *confidence* level.

    Average of all daily returns worse than the VaR threshold.
    A more conservative risk measure than plain VaR.

    Parameters
    ----------
    returns:
        Daily fractional returns.
    confidence:
        Confidence level (default 0.95).

    Returns
    -------
    float
        CVaR as a decimal (negative).
    """
    var   = calculate_var(returns, confidence)
    tail  = returns[returns <= var]
    return float(tail.mean()) if len(tail) > 0 else var


def calculate_monthly_returns(value_series: pd.Series) -> pd.Series:
    """
    Resample portfolio value to month-end and compute monthly returns.

    Parameters
    ----------
    value_series:
        Daily equity curve with DatetimeIndex.

    Returns
    -------
    pd.Series
        Monthly fractional returns.
    """
    monthly = value_series.resample("ME").last()
    return monthly.pct_change().dropna()


def calculate_rolling_sharpe(
    returns: pd.Series,
    window: int = TRADING_DAYS_PER_YEAR,
    risk_free_annual: float = RISK_FREE_RATE_ANNUAL,
) -> pd.Series:
    """
    Rolling annualised Sharpe Ratio.

    Parameters
    ----------
    returns:
        Daily fractional returns.
    window:
        Rolling window in trading days (default 252 = 1 year).
    risk_free_annual:
        Annual risk-free rate.

    Returns
    -------
    pd.Series
        Rolling Sharpe values.
    """
    daily_rf    = risk_free_annual / TRADING_DAYS_PER_YEAR
    excess      = returns - daily_rf
    roll_mean   = excess.rolling(window).mean()
    roll_std    = returns.rolling(window).std()
    roll_sharpe = (roll_mean / roll_std) * np.sqrt(TRADING_DAYS_PER_YEAR)
    return roll_sharpe.rename("rolling_sharpe")


def calculate_rolling_returns(
    value_series: pd.Series,
    window: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """
    Rolling annualised return over *window* trading days.

    Parameters
    ----------
    value_series:
        Portfolio equity curve.
    window:
        Look-back in trading days.

    Returns
    -------
    pd.Series
        Rolling annualised return (decimal).
    """
    roll_return = (
        value_series / value_series.shift(window)
    ) ** (TRADING_DAYS_PER_YEAR / window) - 1
    return roll_return.rename(f"rolling_return_{window}d")


# ──────────────────────────────────────────────────────────────────────────────
# Trade-based metrics
# ──────────────────────────────────────────────────────────────────────────────
def calculate_trade_metrics(
    trade_log: pd.DataFrame,
) -> Dict[str, float]:
    """
    Compute win rate, profit factor, and expectancy from a trade log.

    The trade log must contain a ``'pnl'`` or ``'net_pnl'`` column
    representing the net INR P&L per closed trade.

    Parameters
    ----------
    trade_log:
        DataFrame with one row per closed trade.

    Returns
    -------
    dict
        Keys: ``win_rate_pct``, ``profit_factor``, ``total_trades``,
        ``avg_win_inr``, ``avg_loss_inr``, ``expectancy_inr``.
    """
    pnl_col = None
    for candidate in ("net_pnl", "pnl", "profit_loss", "pnl_net"):
        if candidate in trade_log.columns:
            pnl_col = candidate
            break

    if pnl_col is None or len(trade_log) == 0:
        return {
            "win_rate_pct":   0.0, "profit_factor":  0.0,
            "total_trades":   0,   "avg_win_inr":    0.0,
            "avg_loss_inr":   0.0, "expectancy_inr": 0.0,
        }

    pnl    = trade_log[pnl_col].dropna()
    wins   = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    total         = len(pnl)
    win_rate      = len(wins) / total * 100 if total else 0.0
    avg_win       = float(wins.mean())   if len(wins)   else 0.0
    avg_loss      = float(losses.mean()) if len(losses) else 0.0
    profit_factor = (
        float(wins.sum() / abs(losses.sum()))
        if len(losses) > 0 and abs(losses.sum()) > 0 else float("inf")
    )
    expectancy    = float(pnl.mean()) if total else 0.0

    return {
        "win_rate_pct":   win_rate,
        "profit_factor":  profit_factor,
        "total_trades":   total,
        "avg_win_inr":    avg_win,
        "avg_loss_inr":   avg_loss,
        "expectancy_inr": expectancy,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Facade — MetricsEngine
# ──────────────────────────────────────────────────────────────────────────────
class MetricsEngine:
    """
    Compute the full suite of portfolio metrics in a single call.

    Parameters
    ----------
    risk_free_rate : float
        Annual risk-free rate for Sharpe / Sortino.  Default: 6 %.
    trading_days : int
        Annualisation factor.  Default: 252.
    label : str
        Portfolio / strategy name for reporting.

    Examples
    --------
    ::

        engine  = MetricsEngine(label="EW Hybrid Portfolio")
        metrics = engine.compute(portfolio_value_series)
        print(metrics.report())
    """

    def __init__(
        self,
        risk_free_rate: float = RISK_FREE_RATE_ANNUAL,
        trading_days:   int   = TRADING_DAYS_PER_YEAR,
        label:          str   = "Portfolio",
    ) -> None:
        self.risk_free_rate = risk_free_rate
        self.trading_days   = trading_days
        self.label          = label

    def compute(
        self,
        value_series: pd.Series,
        trade_log: Optional[pd.DataFrame] = None,
    ) -> PortfolioMetrics:
        """
        Run all metrics against *value_series*.

        Parameters
        ----------
        value_series:
            Daily portfolio equity curve (DatetimeIndex).
        trade_log:
            Optional trade log for win-rate / profit-factor metrics.

        Returns
        -------
        PortfolioMetrics
        """
        if len(value_series) < 2:
            raise ValueError(
                f"value_series must have ≥ 2 data points, got {len(value_series)}."
            )

        value_series = value_series.dropna()
        returns      = calculate_returns(value_series)
        monthly_rets = calculate_monthly_returns(value_series)

        cagr        = calculate_cagr(value_series)
        vol         = calculate_annualised_volatility(returns)
        sharpe      = calculate_sharpe(returns, self.risk_free_rate)
        sortino     = calculate_sortino(returns, self.risk_free_rate)
        max_dd, dd_dur, _ = calculate_max_drawdown(value_series)
        calmar      = (cagr / abs(max_dd)) if max_dd != 0 else 0.0
        var_95      = calculate_var(returns)
        cvar_95     = calculate_cvar(returns)

        total_ret   = (value_series.iloc[-1] / value_series.iloc[0] - 1) * 100
        best_month  = float(monthly_rets.max()) * 100 if len(monthly_rets) else 0.0
        worst_month = float(monthly_rets.min()) * 100 if len(monthly_rets) else 0.0
        pos_months  = (float((monthly_rets > 0).mean()) * 100
                       if len(monthly_rets) else 0.0)

        trade_m = calculate_trade_metrics(trade_log) if trade_log is not None else {}

        metrics = PortfolioMetrics(
            label                     = self.label,
            start_date                = str(value_series.index[0].date()),
            end_date                  = str(value_series.index[-1].date()),
            trading_days              = len(value_series),
            initial_value             = round(float(value_series.iloc[0]), 2),
            final_value               = round(float(value_series.iloc[-1]), 2),
            total_return_pct          = round(total_ret, 2),
            cagr_pct                  = round(cagr * 100, 2),
            annualised_vol_pct        = round(vol * 100, 2),
            sharpe_ratio              = round(sharpe, 3),
            sortino_ratio             = round(sortino, 3),
            calmar_ratio              = round(calmar, 3),
            max_drawdown_pct          = round(max_dd * 100, 2),
            max_drawdown_duration_days= dd_dur,
            var_95_pct                = round(var_95 * 100, 2),
            cvar_95_pct               = round(cvar_95 * 100, 2),
            skewness                  = round(float(returns.skew()), 3),
            kurtosis                  = round(float(returns.kurtosis()), 3),
            best_month_pct            = round(best_month, 2),
            worst_month_pct           = round(worst_month, 2),
            positive_months_pct       = round(pos_months, 1),
            win_rate_pct              = trade_m.get("win_rate_pct", 0.0),
            profit_factor             = trade_m.get("profit_factor", 0.0),
            total_trades              = trade_m.get("total_trades", 0),
            avg_win_inr               = trade_m.get("avg_win_inr", 0.0),
            avg_loss_inr              = trade_m.get("avg_loss_inr", 0.0),
            expectancy_inr            = trade_m.get("expectancy_inr", 0.0),
        )

        logger.info(
            "Metrics computed  [%s]  return=%+.2f%%  CAGR=%+.2f%%  "
            "Sharpe=%.3f  MaxDD=%.2f%%",
            self.label, total_ret, cagr * 100, sharpe, max_dd * 100,
        )
        return metrics

    def rolling_metrics(
        self, value_series: pd.Series, window: int = TRADING_DAYS_PER_YEAR
    ) -> pd.DataFrame:
        """
        Compute rolling Sharpe and rolling annualised return.

        Parameters
        ----------
        value_series:
            Daily equity curve.
        window:
            Rolling window in trading days.

        Returns
        -------
        pd.DataFrame
            Columns: ``portfolio_value``, ``rolling_return``, ``rolling_sharpe``,
            ``drawdown``.
        """
        returns = calculate_returns(value_series)
        _, _, dd_curve = calculate_max_drawdown(value_series)
        roll_ret    = calculate_rolling_returns(value_series, window)
        roll_sharpe = calculate_rolling_sharpe(
            returns, window, self.risk_free_rate
        )
        df = pd.DataFrame({
            "portfolio_value": value_series,
            "daily_return":    returns,
            "rolling_return":  roll_ret,
            "rolling_sharpe":  roll_sharpe,
            "drawdown":        dd_curve,
        })
        return df

    def monthly_heatmap_data(self, value_series: pd.Series) -> pd.DataFrame:
        """
        Pivot monthly returns into a year × month heatmap table.

        Useful for visualising seasonal performance patterns.

        Parameters
        ----------
        value_series:
            Daily equity curve.

        Returns
        -------
        pd.DataFrame
            Rows = years, columns = months (Jan–Dec), values = monthly return %.
        """
        monthly = calculate_monthly_returns(value_series) * 100
        df      = monthly.to_frame("return")
        df["year"]  = df.index.year
        df["month"] = df.index.month
        pivot   = df.pivot_table(
            values="return", index="year", columns="month", aggfunc="sum"
        )
        month_names = {
            1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 5:"May", 6:"Jun",
            7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec",
        }
        pivot.columns = [month_names.get(c, c) for c in pivot.columns]
        pivot["Full Year"] = pivot.sum(axis=1)
        return pivot.round(2)


def compare_strategies(
    series_dict: Dict[str, pd.Series],
    trade_logs: Optional[Dict[str, pd.DataFrame]] = None,
    risk_free_rate: float = RISK_FREE_RATE_ANNUAL,
) -> pd.DataFrame:
    """
    Compute metrics for multiple strategies and return a comparison table.

    Parameters
    ----------
    series_dict:
        ``{label: portfolio_value_series}``
    trade_logs:
        Optional ``{label: trade_log_df}``
    risk_free_rate:
        Annual risk-free rate.

    Returns
    -------
    pd.DataFrame
        One column per strategy, rows = metric names.
    """
    results = {}
    for label, series in series_dict.items():
        tlog = (trade_logs or {}).get(label)
        engine   = MetricsEngine(risk_free_rate=risk_free_rate, label=label)
        metrics  = engine.compute(series, trade_log=tlog)
        results[label] = metrics.to_series()
    return pd.DataFrame(results)


# ──────────────────────────────────────────────────────────────────────────────
# Script entry-point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)
    )))
    if _root not in sys.path:
        sys.path.insert(0, _root)

    from src.portfolio.rebalance import (
        simulate_portfolio, RebalanceFrequency, load_aligned_prices
    )

    print("=" * 68)
    print("  Portfolio Metrics Engine — Demo")
    print("=" * 68)

    STOCKS  = ["TCS", "RELIANCE", "INFOSYS"]
    CAPITAL = 1_000_000.0

    # ── Build 4 portfolio simulations ─────────────────────────────────────────
    configs = {
        "EW Monthly":    (
            {s: 1/3 for s in STOCKS}, RebalanceFrequency.MONTHLY, 0.05
        ),
        "EW Quarterly":  (
            {s: 1/3 for s in STOCKS}, RebalanceFrequency.QUARTERLY, 0.05
        ),
        "EW Hybrid":     (
            {s: 1/3 for s in STOCKS}, RebalanceFrequency.HYBRID, 0.05
        ),
        "TCS-Heavy":     (
            {"TCS": 0.60, "RELIANCE": 0.25, "INFOSYS": 0.15},
            RebalanceFrequency.HYBRID, 0.05
        ),
    }

    series_dict: Dict[str, pd.Series] = {}
    for label, (weights, freq, drift) in configs.items():
        result = simulate_portfolio(
            stocks=STOCKS, target_weights=weights,
            initial_capital=CAPITAL, frequency=freq,
            drift_threshold=drift,
        )
        series_dict[label] = result.portfolio_value_series

    # ── Per-strategy detailed reports ─────────────────────────────────────────
    print("\n── Individual Reports ────────────────────────────────────────────")
    for label, series in series_dict.items():
        engine  = MetricsEngine(label=label)
        metrics = engine.compute(series)
        print(f"\n{metrics.report()}")

    # ── Comparison table ──────────────────────────────────────────────────────
    print("\n── Strategy Comparison ──────────────────────────────────────────")
    comp = compare_strategies(series_dict)

    display_rows = [
        "total_return_pct", "cagr_pct", "annualised_vol_pct",
        "sharpe_ratio", "sortino_ratio", "calmar_ratio",
        "max_drawdown_pct", "max_drawdown_duration_days",
        "best_month_pct", "worst_month_pct", "positive_months_pct",
        "var_95_pct", "cvar_95_pct",
    ]
    display_names = {
        "total_return_pct":            "Total Return %",
        "cagr_pct":                    "CAGR %",
        "annualised_vol_pct":          "Volatility % (ann.)",
        "sharpe_ratio":                "Sharpe Ratio",
        "sortino_ratio":               "Sortino Ratio",
        "calmar_ratio":                "Calmar Ratio",
        "max_drawdown_pct":            "Max Drawdown %",
        "max_drawdown_duration_days":  "Max DD Duration (days)",
        "best_month_pct":              "Best Month %",
        "worst_month_pct":             "Worst Month %",
        "positive_months_pct":         "% Positive Months",
        "var_95_pct":                  "VaR 95% (daily)",
        "cvar_95_pct":                 "CVaR 95% (daily)",
    }
    subset = comp.loc[display_rows].rename(index=display_names)
    print(subset.to_string(float_format=lambda x: f"{x:>+.2f}"))

    # ── Monthly heatmap for best strategy ─────────────────────────────────────
    best_label = max(
        series_dict,
        key=lambda lbl: MetricsEngine(label=lbl).compute(series_dict[lbl]).sharpe_ratio
    )
    best_engine = MetricsEngine(label=best_label)
    heatmap     = best_engine.monthly_heatmap_data(series_dict[best_label])
    print(f"\n── Monthly Return Heatmap: {best_label} ─────────────────────────")
    print(heatmap.to_string(float_format=lambda x: f"{x:>+.1f}"))

    # ── Rolling metrics sample ─────────────────────────────────────────────────
    print(f"\n── Rolling Metrics (last 5 rows): {best_label} ──────────────────")
    roll_df = best_engine.rolling_metrics(series_dict[best_label])
    print(
        roll_df[["portfolio_value", "rolling_return", "rolling_sharpe", "drawdown"]]
        .tail(5).round(4).to_string()
    )
    print()
