"""
performance_metrics.py
======================
Strategy and portfolio analytics engine for the Algorithmic Trading
Strategy Backtester — Phase 6.

Scope vs Phase 5
----------------
Phase 5's ``portfolio_metrics.py`` was the *internal* metrics layer for
the rebalancing engine.  Phase 6's ``performance_metrics.py`` is the
*analytical* layer used for strategy evaluation, research reporting,
and presentation.  It adds:

* **Information Ratio** — excess return per unit of tracking error vs
  a benchmark.
* **Omega Ratio** — probability-weighted ratio of gains to losses above
  a threshold.
* **Ulcer Index** — RMS of drawdown percentage series; measures the
  depth *and* duration of pain.
* **Recovery Factor** — net profit ÷ max drawdown.
* **Gain-to-Pain Ratio** — Jack Schwager's metric:
  sum of returns / sum of absolute negative returns.
* **Annual returns table** — year-by-year breakdown for visual inspection.
* **Benchmark comparison** — alpha, beta, and information ratio vs
  a configurable benchmark equity curve.

Architecture
------------
* :class:`PerformanceReport` — rich frozen dataclass.
* Pure calculation functions (importable individually).
* :class:`PerformanceAnalyzer` — facade with ``compute()``,
  ``annual_returns()``, ``rolling_metrics()``, and ``compare()``.

Usage
-----
::

    python src/analytics/performance_metrics.py
    from src.analytics.performance_metrics import PerformanceAnalyzer
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
TRADING_DAYS:  int   = 252
RISK_FREE_RATE: float = 0.06    # 6 % — Indian repo-rate proxy
CHARTS_DIR:    str   = os.path.join("reports", "charts")


# ──────────────────────────────────────────────────────────────────────────────
# Value object
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class PerformanceReport:
    """
    Full analytics report for one strategy / portfolio.

    All ratio fields are dimensionless; percentage fields are in
    plain-percent (12.5 means 12.5 %, not 0.125).

    Parameters
    ----------
    label:               Strategy identifier.
    start / end:         Backtest period.
    n_days:              Number of trading bars.
    initial / final:     Portfolio value (INR).
    total_return_pct:    (final/initial − 1) × 100.
    cagr_pct:            Compound annual growth rate %.
    vol_pct:             Annualised σ of daily returns.
    sharpe:              (CAGR − r_f) / vol.
    sortino:             (CAGR − r_f) / downside_vol.
    calmar:              CAGR / |max_drawdown|.
    omega:               Ω ratio above the daily risk-free threshold.
    ulcer_index:         RMS of daily drawdown depth.
    recovery_factor:     Net profit / |max_drawdown_inr|.
    gain_to_pain:        Sum(positive returns) / |sum(negative returns)|.
    max_drawdown_pct:    Deepest peak-to-trough decline %.
    avg_drawdown_pct:    Mean drawdown depth across all underwater periods.
    max_dd_duration:     Longest drawdown in trading bars.
    var_95_pct:          Historical daily VaR at 95 % confidence.
    cvar_95_pct:         Expected shortfall beyond VaR.
    skewness:            Daily return skewness.
    kurtosis:            Daily return excess kurtosis.
    best_day_pct:        Best single-day return.
    worst_day_pct:       Worst single-day return.
    best_month_pct:      Best calendar month return.
    worst_month_pct:     Worst calendar month return.
    pct_positive_months: % of months with positive return.
    alpha:               Jensen's alpha vs benchmark (annualised %).
    beta:                Market beta vs benchmark.
    information_ratio:   (return − benchmark_return) / tracking_error.
    """
    label:               str
    start:               str
    end:                 str
    n_days:              int
    initial:             float
    final:               float
    total_return_pct:    float
    cagr_pct:            float
    vol_pct:             float
    sharpe:              float
    sortino:             float
    calmar:              float
    omega:               float
    ulcer_index:         float
    recovery_factor:     float
    gain_to_pain:        float
    max_drawdown_pct:    float
    avg_drawdown_pct:    float
    max_dd_duration:     int
    var_95_pct:          float
    cvar_95_pct:         float
    skewness:            float
    kurtosis:            float
    best_day_pct:        float
    worst_day_pct:       float
    best_month_pct:      float
    worst_month_pct:     float
    pct_positive_months: float
    alpha:               Optional[float] = field(default=None)
    beta:                Optional[float] = field(default=None)
    information_ratio:   Optional[float] = field(default=None)

    # ── Derived ───────────────────────────────────────────────────────────────
    def to_series(self) -> pd.Series:
        """Flat pd.Series for comparison tables."""
        return pd.Series(
            {f: getattr(self, f) for f in self.__dataclass_fields__},  # type: ignore[attr-defined]
            name=self.label,
        )

    def report(self, width: int = 56) -> str:
        """
        Return a formatted console report.

        Parameters
        ----------
        width:
            Inner box width in characters.
        """
        W  = width
        hd = lambda t: f"  ║  {t:<{W-4}}║"
        hr = f"  ╠{'═'*W}╣"
        lines = [
            f"  ╔{'═'*W}╗",
            hd(f"Performance Report: {self.label}"),
            hd(f"Period : {self.start} → {self.end}  ({self.n_days} bars)"),
            hd(f"Capital: ₹{self.initial:>12,.0f} → ₹{self.final:>12,.0f}"),
            hr,
            hd("RETURN METRICS"),
            hd(f"  Total return      : {self.total_return_pct:>+10.2f} %"),
            hd(f"  CAGR              : {self.cagr_pct:>+10.2f} %"),
            hd(f"  Best single day   : {self.best_day_pct:>+10.2f} %"),
            hd(f"  Worst single day  : {self.worst_day_pct:>+10.2f} %"),
            hd(f"  Best month        : {self.best_month_pct:>+10.2f} %"),
            hd(f"  Worst month       : {self.worst_month_pct:>+10.2f} %"),
            hd(f"  % Positive months : {self.pct_positive_months:>10.1f} %"),
            hr,
            hd("RISK METRICS"),
            hd(f"  Annualised vol    : {self.vol_pct:>10.2f} %"),
            hd(f"  Max drawdown      : {self.max_drawdown_pct:>+10.2f} %"),
            hd(f"  Avg drawdown      : {self.avg_drawdown_pct:>+10.2f} %"),
            hd(f"  Max DD duration   : {self.max_dd_duration:>10} bars"),
            hd(f"  VaR 95% (daily)   : {self.var_95_pct:>+10.2f} %"),
            hd(f"  CVaR 95% (daily)  : {self.cvar_95_pct:>+10.2f} %"),
            hd(f"  Skewness          : {self.skewness:>+10.3f}"),
            hd(f"  Kurtosis          : {self.kurtosis:>+10.3f}"),
            hr,
            hd("RISK-ADJUSTED"),
            hd(f"  Sharpe Ratio      : {self.sharpe:>+10.3f}"),
            hd(f"  Sortino Ratio     : {self.sortino:>+10.3f}"),
            hd(f"  Calmar Ratio      : {self.calmar:>+10.3f}"),
            hd(f"  Omega Ratio       : {self.omega:>+10.3f}"),
            hd(f"  Ulcer Index       : {self.ulcer_index:>10.4f}"),
            hd(f"  Recovery Factor   : {self.recovery_factor:>+10.2f}"),
            hd(f"  Gain-to-Pain      : {self.gain_to_pain:>+10.3f}"),
        ]
        if self.alpha is not None:
            lines += [
                hr,
                hd("BENCHMARK ANALYTICS"),
                hd(f"  Alpha (ann.)      : {self.alpha:>+10.3f} %"),
                hd(f"  Beta              : {self.beta:>+10.3f}"),
                hd(f"  Information Ratio : {self.information_ratio:>+10.3f}"),
            ]
        lines.append(f"  ╚{'═'*W}╝")
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Pure metric functions
# ──────────────────────────────────────────────────────────────────────────────
def _returns(series: pd.Series) -> pd.Series:
    """Daily pct-change returns, NaN dropped."""
    return series.pct_change().dropna()


def _cagr(series: pd.Series) -> float:
    n = len(series)
    if n < 2 or series.iloc[0] <= 0:
        return 0.0
    return (series.iloc[-1] / series.iloc[0]) ** (TRADING_DAYS / n) - 1.0


def _vol(returns: pd.Series) -> float:
    return float(returns.std() * np.sqrt(TRADING_DAYS)) if len(returns) > 1 else 0.0


def _sharpe(returns: pd.Series, rf: float = RISK_FREE_RATE) -> float:
    v = _vol(returns)
    if v == 0:
        return 0.0
    ann = (1 + returns.mean()) ** TRADING_DAYS - 1
    return (ann - rf) / v


def _sortino(returns: pd.Series, rf: float = RISK_FREE_RATE) -> float:
    daily_rf = rf / TRADING_DAYS
    excess   = returns - daily_rf
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("inf")
    dv = float(downside.std() * np.sqrt(TRADING_DAYS))
    if dv == 0:
        return 0.0
    ann = (1 + returns.mean()) ** TRADING_DAYS - 1
    return (ann - rf) / dv


def _max_drawdown(series: pd.Series) -> Tuple[float, int, pd.Series]:
    """
    Returns (max_dd fraction, max_duration_bars, drawdown_series).
    """
    peak  = series.cummax()
    dd    = (series - peak) / peak
    max_d = float(dd.min())
    # Duration
    underwater  = dd < 0
    max_dur = cur = 0
    for uw in underwater:
        cur = cur + 1 if uw else 0
        max_dur = max(max_dur, cur)
    return max_d, max_dur, dd


def _avg_drawdown(dd_series: pd.Series) -> float:
    """Mean depth of underwater periods only."""
    uw = dd_series[dd_series < 0]
    return float(uw.mean()) if len(uw) > 0 else 0.0


def _var(returns: pd.Series, q: float = 0.05) -> float:
    return float(np.percentile(returns, q * 100))


def _cvar(returns: pd.Series, q: float = 0.05) -> float:
    v   = _var(returns, q)
    tail = returns[returns <= v]
    return float(tail.mean()) if len(tail) > 0 else v


def calculate_omega_ratio(
    returns: pd.Series,
    threshold: float = 0.0,
) -> float:
    """
    Omega Ratio — probability-weighted gains vs losses above *threshold*.

    ``Ω = sum(max(R − t, 0)) / sum(max(t − R, 0))``

    Parameters
    ----------
    returns:
        Daily fractional returns.
    threshold:
        Daily return threshold (default 0 — any positive day is a gain).

    Returns
    -------
    float
        Omega ratio.  Values > 1 indicate a net positive distribution.
    """
    gains  = (returns - threshold).clip(lower=0).sum()
    losses = (threshold - returns).clip(lower=0).sum()
    if losses == 0:
        return float("inf")
    return float(gains / losses)


def calculate_ulcer_index(series: pd.Series) -> float:
    """
    Ulcer Index — RMS of the drawdown percentage series.

    Captures both depth *and* duration of drawdowns; high values
    indicate prolonged painful periods, not just a single bad day.

    Parameters
    ----------
    series:
        Portfolio equity curve.

    Returns
    -------
    float
        Ulcer Index (non-negative).
    """
    peak = series.cummax()
    dd   = ((series - peak) / peak) * 100     # percent
    return float(np.sqrt((dd ** 2).mean()))


def calculate_recovery_factor(series: pd.Series) -> float:
    """
    Recovery Factor = Net Profit / |Max Drawdown in INR|.

    A high recovery factor means the strategy generates large profits
    relative to its worst loss, even if the max-drawdown percentage
    is moderate.

    Parameters
    ----------
    series:
        Portfolio equity curve.

    Returns
    -------
    float
    """
    net_profit = series.iloc[-1] - series.iloc[0]
    peak       = series.cummax()
    dd_inr     = series - peak           # always ≤ 0
    max_dd_inr = abs(float(dd_inr.min()))
    if max_dd_inr == 0:
        return float("inf")
    return net_profit / max_dd_inr


def calculate_gain_to_pain(returns: pd.Series) -> float:
    """
    Gain-to-Pain Ratio (Jack Schwager).

    ``G2P = sum(all returns) / |sum(negative returns)|``

    Intuitively: for every ₹1 of pain (negative return), how many
    rupees of gain does the strategy produce?

    Parameters
    ----------
    returns:
        Daily fractional returns.

    Returns
    -------
    float
    """
    total_gain = float(returns.sum())
    pain       = float(returns[returns < 0].sum())
    if pain == 0:
        return float("inf")
    return total_gain / abs(pain)


def calculate_beta_alpha(
    returns: pd.Series,
    bench_returns: pd.Series,
    rf: float = RISK_FREE_RATE,
) -> Tuple[float, float]:
    """
    OLS regression beta and Jensen's alpha.

    Parameters
    ----------
    returns:
        Strategy daily returns.
    bench_returns:
        Benchmark daily returns (aligned to same dates).
    rf:
        Annual risk-free rate.

    Returns
    -------
    (beta, alpha_annualised_pct)
    """
    aligned   = pd.concat([returns, bench_returns], axis=1).dropna()
    if len(aligned) < 30:
        return 0.0, 0.0
    strat_r = aligned.iloc[:, 0].values
    bench_r = aligned.iloc[:, 1].values
    cov     = np.cov(strat_r, bench_r)
    beta    = cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else 0.0
    daily_rf = rf / TRADING_DAYS
    alpha_daily = np.mean(strat_r) - (daily_rf + beta * (np.mean(bench_r) - daily_rf))
    alpha_ann = ((1 + alpha_daily) ** TRADING_DAYS - 1) * 100
    return float(beta), float(alpha_ann)


def calculate_information_ratio(
    returns: pd.Series,
    bench_returns: pd.Series,
) -> float:
    """
    Information Ratio = (strategy return − benchmark return) / tracking error.

    Parameters
    ----------
    returns:
        Strategy daily returns.
    bench_returns:
        Benchmark daily returns.

    Returns
    -------
    float
    """
    aligned     = pd.concat([returns, bench_returns], axis=1).dropna()
    if len(aligned) < 20:
        return 0.0
    active      = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    te          = float(active.std() * np.sqrt(TRADING_DAYS))
    active_ret  = float((1 + active.mean()) ** TRADING_DAYS - 1)
    if te == 0:
        return 0.0
    return active_ret / te


def annual_returns_table(series: pd.Series) -> pd.DataFrame:
    """
    Year-by-year return breakdown.

    Parameters
    ----------
    series:
        Daily portfolio value with DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        Columns: ``year``, ``start_value``, ``end_value``, ``return_pct``.
    """
    df = series.to_frame("value")
    df["year"] = df.index.year
    rows = []
    for year, grp in df.groupby("year"):
        start = grp["value"].iloc[0]
        end   = grp["value"].iloc[-1]
        rows.append({
            "year":       year,
            "start_value": round(start, 2),
            "end_value":   round(end, 2),
            "return_pct":  round((end / start - 1) * 100, 2),
        })
    return pd.DataFrame(rows).set_index("year")


# ──────────────────────────────────────────────────────────────────────────────
# Facade
# ──────────────────────────────────────────────────────────────────────────────
class PerformanceAnalyzer:
    """
    Compute the full Phase 6 analytics suite for a strategy equity curve.

    Parameters
    ----------
    label:
        Strategy / portfolio name.
    risk_free_rate:
        Annual risk-free rate.  Default: 6 % (Indian context).

    Examples
    --------
    ::

        pa = PerformanceAnalyzer(label="TCS-Heavy Hybrid")
        report = pa.compute(equity_curve, benchmark_series=bah_series)
        print(report.report())
    """

    def __init__(
        self,
        label:          str   = "Strategy",
        risk_free_rate: float = RISK_FREE_RATE,
    ) -> None:
        self.label = label
        self.rf    = risk_free_rate

    def compute(
        self,
        series:          pd.Series,
        benchmark_series: Optional[pd.Series] = None,
    ) -> PerformanceReport:
        """
        Compute the full :class:`PerformanceReport`.

        Parameters
        ----------
        series:
            Portfolio equity curve (DatetimeIndex, daily).
        benchmark_series:
            Optional buy-and-hold benchmark for alpha/beta/IR.

        Returns
        -------
        PerformanceReport
        """
        series  = series.dropna().sort_index()
        returns = _returns(series)
        cagr    = _cagr(series)
        vol     = _vol(returns)
        max_dd, max_dur, dd_curve = _max_drawdown(series)
        avg_dd  = _avg_drawdown(dd_curve)
        monthly = series.resample("ME").last().pct_change().dropna()

        alpha = beta = ir = None
        if benchmark_series is not None:
            bench_r = _returns(benchmark_series.dropna().sort_index())
            beta, alpha = calculate_beta_alpha(returns, bench_r, self.rf)
            ir = calculate_information_ratio(returns, bench_r)

        report = PerformanceReport(
            label               = self.label,
            start               = str(series.index[0].date()),
            end                 = str(series.index[-1].date()),
            n_days              = len(series),
            initial             = round(float(series.iloc[0]), 2),
            final               = round(float(series.iloc[-1]), 2),
            total_return_pct    = round((series.iloc[-1] / series.iloc[0] - 1) * 100, 2),
            cagr_pct            = round(cagr * 100, 2),
            vol_pct             = round(vol * 100, 2),
            sharpe              = round(_sharpe(returns, self.rf), 3),
            sortino             = round(_sortino(returns, self.rf), 3),
            calmar              = round((cagr / abs(max_dd)) if max_dd != 0 else 0.0, 3),
            omega               = round(calculate_omega_ratio(returns), 3),
            ulcer_index         = round(calculate_ulcer_index(series), 4),
            recovery_factor     = round(calculate_recovery_factor(series), 2),
            gain_to_pain        = round(calculate_gain_to_pain(returns), 3),
            max_drawdown_pct    = round(max_dd * 100, 2),
            avg_drawdown_pct    = round(avg_dd * 100, 2),
            max_dd_duration     = max_dur,
            var_95_pct          = round(_var(returns) * 100, 2),
            cvar_95_pct         = round(_cvar(returns) * 100, 2),
            skewness            = round(float(returns.skew()), 3),
            kurtosis            = round(float(returns.kurtosis()), 3),
            best_day_pct        = round(float(returns.max()) * 100, 2),
            worst_day_pct       = round(float(returns.min()) * 100, 2),
            best_month_pct      = round(float(monthly.max()) * 100, 2) if len(monthly) else 0.0,
            worst_month_pct     = round(float(monthly.min()) * 100, 2) if len(monthly) else 0.0,
            pct_positive_months = round(float((monthly > 0).mean()) * 100, 1) if len(monthly) else 0.0,
            alpha               = round(alpha, 3) if alpha is not None else None,
            beta                = round(beta, 3) if beta is not None else None,
            information_ratio   = round(ir, 3) if ir is not None else None,
        )
        logger.info(
            "PerformanceReport [%s]  return=%+.2f%%  CAGR=%+.2f%%  "
            "Sharpe=%.3f  MaxDD=%.2f%%  Omega=%.2f  UI=%.4f",
            self.label, report.total_return_pct, report.cagr_pct,
            report.sharpe, report.max_drawdown_pct,
            report.omega, report.ulcer_index,
        )
        return report

    def annual_returns(self, series: pd.Series) -> pd.DataFrame:
        """Year-by-year return table."""
        return annual_returns_table(series)

    def rolling_metrics(
        self,
        series: pd.Series,
        window: int = TRADING_DAYS,
    ) -> pd.DataFrame:
        """
        Rolling Sharpe, rolling CAGR, and drawdown curve.

        Parameters
        ----------
        series:
            Equity curve.
        window:
            Rolling window in bars.

        Returns
        -------
        pd.DataFrame
        """
        rets  = _returns(series)
        daily_rf = self.rf / TRADING_DAYS
        excess   = rets - daily_rf
        roll_s   = (excess.rolling(window).mean() /
                    rets.rolling(window).std()) * np.sqrt(TRADING_DAYS)
        roll_r   = (series / series.shift(window)) ** (TRADING_DAYS / window) - 1
        _, _, dd = _max_drawdown(series)
        return pd.DataFrame({
            "equity":           series,
            "daily_return":     rets,
            "rolling_cagr":     roll_r,
            "rolling_sharpe":   roll_s,
            "drawdown":         dd,
        })

    def compare(
        self,
        series_dict: Dict[str, pd.Series],
        benchmark_series: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Run ``compute()`` for multiple strategies and return a comparison
        DataFrame.

        Parameters
        ----------
        series_dict:
            ``{label: equity_curve}``.
        benchmark_series:
            Optional benchmark for alpha/beta/IR.

        Returns
        -------
        pd.DataFrame
            One column per strategy, rows = metric names.
        """
        results = {}
        for label, series in series_dict.items():
            pa = PerformanceAnalyzer(label=label, risk_free_rate=self.rf)
            rpt = pa.compute(series, benchmark_series)
            results[label] = rpt.to_series()
        return pd.DataFrame(results)


# ──────────────────────────────────────────────────────────────────────────────
# Script entry-point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _root not in sys.path:
        sys.path.insert(0, _root)

    import logging as _lg; _lg.disable(_lg.CRITICAL)
    from src.portfolio.rebalance import simulate_portfolio, RebalanceFrequency
    from src.portfolio.allocation import PortfolioAllocator, AllocationScheme

    STOCKS  = ["TCS", "RELIANCE", "INFOSYS"]
    CAPITAL = 1_000_000.0
    EW      = {s: 1/3 for s in STOCKS}
    TH      = {"TCS": 0.60, "RELIANCE": 0.25, "INFOSYS": 0.15}

    _lg.disable(_lg.NOTSET)
    logger.info("Building equity curves from Phase 5 simulations…")

    r_ew  = simulate_portfolio(STOCKS, EW,  CAPITAL, RebalanceFrequency.HYBRID, 0.05)
    r_th  = simulate_portfolio(STOCKS, TH,  CAPITAL, RebalanceFrequency.HYBRID, 0.05)

    # Build a buy-and-hold TCS benchmark
    import pandas as _pd
    tcs_df  = _pd.read_csv("data/processed/TCS_processed.csv", parse_dates=["date"])
    tcs_df  = tcs_df.set_index("date").sort_index()
    bah_tcs = (tcs_df["close"] / tcs_df["close"].iloc[0]) * CAPITAL
    bah_tcs.name = "benchmark_value"

    series_map = {
        "EW Hybrid":      r_ew.portfolio_value_series,
        "TCS-Heavy":      r_th.portfolio_value_series,
        "Buy-Hold TCS":   bah_tcs,
    }

    print("=" * 60)
    print("  Performance Analytics — Demo")
    print("=" * 60)

    # Individual reports
    for label, series in series_map.items():
        pa  = PerformanceAnalyzer(label=label)
        rpt = pa.compute(series, benchmark_series=bah_tcs)
        print(f"\n{rpt.report()}")

    # Annual returns table
    print("\n── Annual Returns: TCS-Heavy ─────────────────────────────")
    pa_ann = PerformanceAnalyzer(label="TCS-Heavy")
    ann    = pa_ann.annual_returns(r_th.portfolio_value_series)
    print(ann.to_string(float_format=lambda x: f"{x:>+.2f}"))

    # Comparison table
    print("\n── Strategy Comparison ───────────────────────────────────")
    pa_comp = PerformanceAnalyzer(label="comparison")
    comp    = pa_comp.compare(series_map, benchmark_series=bah_tcs)
    display = [
        "total_return_pct","cagr_pct","vol_pct","sharpe","sortino",
        "calmar","omega","ulcer_index","recovery_factor","gain_to_pain",
        "max_drawdown_pct","avg_drawdown_pct","pct_positive_months",
        "alpha","beta","information_ratio",
    ]
    print(comp.loc[display].to_string(float_format=lambda x: f"{x:>+.3f}"))
    print()
