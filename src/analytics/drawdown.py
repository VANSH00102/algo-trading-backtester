"""
drawdown.py
===========
Deep drawdown analysis engine for the Algorithmic Trading Strategy
Backtester — Phase 6.

What this module produces
-------------------------
**Underwater curve** — the daily drawdown series showing how far
the portfolio is below its previous peak at every point in time.

**Drawdown period table** — every distinct drawdown episode with:
* Start date (day after the previous peak)
* Peak date (high-water mark)
* Trough date (worst point of the episode)
* Recovery date (day the curve returns to previous peak, or ``None``)
* Depth (%, negative)
* Duration (start → trough, bars)
* Recovery time (trough → recovery, bars)
* Whether it has recovered

**Summary statistics** — max, average, median drawdown; total time
spent underwater; recovery rate.

**Matplotlib chart** — equity curve with shaded drawdowns and an
underwater chart below it, saved to ``reports/charts/``.

Architecture
------------
* :class:`DrawdownPeriod` — immutable dataclass for one episode.
* :class:`DrawdownResult` — complete analysis bundle.
* :class:`DrawdownAnalyzer` — facade that drives the full analysis.
* ``compute_drawdown_series()`` — pure function, importable standalone.

Integration
-----------
Called by ``performance_metrics.py`` for the drawdown fields, and by
the Phase 7 report generator for visualisation.

Usage
-----
::

    python src/analytics/drawdown.py
    from src.analytics.drawdown import DrawdownAnalyzer
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")          # non-interactive — safe for headless runs
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
CHARTS_DIR: str = os.path.join("reports", "charts")


# ──────────────────────────────────────────────────────────────────────────────
# Value objects
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class DrawdownPeriod:
    """
    One distinct drawdown episode.

    Attributes
    ----------
    rank : int
        Rank by depth (1 = deepest).
    start : pd.Timestamp
        First bar below the previous high-water mark.
    peak : pd.Timestamp
        Date of the high-water mark that preceded this drawdown.
    trough : pd.Timestamp
        Date of the deepest point.
    recovery : pd.Timestamp or None
        Date the equity curve returned to the prior peak.
        ``None`` if still underwater at end of series.
    depth_pct : float
        (trough − peak) / peak × 100  (negative).
    depth_inr : float
        Rupee loss from peak to trough.
    duration_bars : int
        Bars from *start* to *trough*.
    recovery_bars : Optional[int]
        Bars from *trough* to *recovery* (``None`` if not recovered).
    recovered : bool
        Whether the equity curve fully recovered.
    """
    rank:           int
    start:          pd.Timestamp
    peak:           pd.Timestamp
    trough:         pd.Timestamp
    recovery:       Optional[pd.Timestamp]
    depth_pct:      float
    depth_inr:      float
    duration_bars:  int
    recovery_bars:  Optional[int]
    recovered:      bool

    def __str__(self) -> str:
        rec = str(self.recovery.date()) if self.recovery else "Not recovered"
        return (
            f"DD#{self.rank:<2}  {str(self.peak.date()):<12}→"
            f"{str(self.trough.date()):<12}  "
            f"depth={self.depth_pct:>+7.2f}%  "
            f"dur={self.duration_bars:>4} bars  "
            f"recov={rec}"
        )


@dataclass
class DrawdownResult:
    """
    Complete drawdown analysis for one equity curve.

    Attributes
    ----------
    underwater_curve : pd.Series
        Daily drawdown percentage (0 at peaks, negative in troughs).
    periods : List[DrawdownPeriod]
        All distinct drawdown episodes, sorted by depth.
    max_drawdown_pct : float
    avg_drawdown_pct : float
    median_drawdown_pct : float
    max_duration_bars : int
    avg_duration_bars : float
    pct_time_underwater : float
        Fraction of bars spent below the previous peak (0–100).
    recovery_rate_pct : float
        Percentage of drawdown episodes that fully recovered.
    n_periods : int
    """
    underwater_curve:    pd.Series
    periods:             List[DrawdownPeriod]
    max_drawdown_pct:    float
    avg_drawdown_pct:    float
    median_drawdown_pct: float
    max_duration_bars:   int
    avg_duration_bars:   float
    pct_time_underwater: float
    recovery_rate_pct:   float
    n_periods:           int

    def top(self, n: int = 5) -> List[DrawdownPeriod]:
        """Return the *n* deepest drawdown periods."""
        return self.periods[:n]

    def summary_df(self) -> pd.DataFrame:
        """Tabulate all periods as a DataFrame."""
        rows = []
        for p in self.periods:
            rows.append({
                "rank":           p.rank,
                "peak_date":      p.peak.date(),
                "trough_date":    p.trough.date(),
                "recovery_date":  p.recovery.date() if p.recovery else "—",
                "depth_pct":      p.depth_pct,
                "depth_inr":      p.depth_inr,
                "duration_bars":  p.duration_bars,
                "recovery_bars":  p.recovery_bars if p.recovery_bars else "—",
                "recovered":      p.recovered,
            })
        return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Core computation
# ──────────────────────────────────────────────────────────────────────────────
def compute_drawdown_series(series: pd.Series) -> pd.Series:
    """
    Compute the daily drawdown percentage series.

    ``DD_t = (V_t − peak_t) / peak_t × 100``

    Values are ≤ 0; 0 means the portfolio is at or above its previous
    high-water mark.

    Parameters
    ----------
    series:
        Portfolio equity curve.

    Returns
    -------
    pd.Series
        Drawdown series (percent, ≤ 0).
    """
    peak = series.cummax()
    return ((series - peak) / peak * 100).rename("drawdown_pct")


def _identify_periods(series: pd.Series) -> List[DrawdownPeriod]:
    """
    Identify all distinct drawdown episodes from an equity curve.

    Algorithm
    ---------
    1. Compute running peak.
    2. Scan forward: when price falls below peak, a drawdown starts.
    3. Track the trough (minimum point) within the episode.
    4. The episode ends when price recovers to the previous peak.

    Parameters
    ----------
    series:
        Portfolio equity curve (sorted DatetimeIndex).

    Returns
    -------
    List[DrawdownPeriod]
        Sorted by ``depth_pct`` ascending (deepest first).
    """
    if len(series) < 2:
        return []

    peak_val   = series.iloc[0]
    peak_date  = series.index[0]
    start_date: Optional[pd.Timestamp] = None
    trough_val  = peak_val
    trough_date = peak_date
    in_dd       = False
    periods: List[DrawdownPeriod] = []

    for date, val in series.items():
        if val >= peak_val:
            # At or above high-water mark
            if in_dd:
                # Recovery achieved
                depth_pct  = (trough_val - peak_val) / peak_val * 100
                depth_inr  = trough_val - peak_val
                dur        = (series.loc[start_date:trough_date].index
                              .get_loc(trough_date) -
                              series.loc[start_date:trough_date].index
                              .get_loc(start_date)) + 1
                rec_bars   = (series.loc[trough_date:date].index
                              .get_loc(date) -
                              series.loc[trough_date:date].index
                              .get_loc(trough_date))
                periods.append(DrawdownPeriod(
                    rank=0,
                    start=start_date, peak=peak_date,
                    trough=trough_date, recovery=date,
                    depth_pct=round(depth_pct, 4),
                    depth_inr=round(depth_inr, 2),
                    duration_bars=dur,
                    recovery_bars=rec_bars,
                    recovered=True,
                ))
                in_dd = False
            peak_val  = val
            peak_date = date
        else:
            # Below high-water mark
            if not in_dd:
                in_dd       = True
                start_date  = date
                trough_val  = val
                trough_date = date
            elif val < trough_val:
                trough_val  = val
                trough_date = date

    # Close any open drawdown at the end of the series
    if in_dd and start_date is not None:
        depth_pct = (trough_val - peak_val) / peak_val * 100
        depth_inr = trough_val - peak_val
        try:
            dur = (series.loc[start_date:trough_date].index
                   .get_loc(trough_date) -
                   series.loc[start_date:trough_date].index
                   .get_loc(start_date)) + 1
        except Exception:
            dur = 0
        periods.append(DrawdownPeriod(
            rank=0,
            start=start_date, peak=peak_date,
            trough=trough_date, recovery=None,
            depth_pct=round(depth_pct, 4),
            depth_inr=round(depth_inr, 2),
            duration_bars=dur,
            recovery_bars=None,
            recovered=False,
        ))

    # Rank by depth (most negative = rank 1)
    periods_sorted = sorted(periods, key=lambda p: p.depth_pct)
    ranked = []
    for i, p in enumerate(periods_sorted, 1):
        ranked.append(DrawdownPeriod(
            rank=i, start=p.start, peak=p.peak,
            trough=p.trough, recovery=p.recovery,
            depth_pct=p.depth_pct, depth_inr=p.depth_inr,
            duration_bars=p.duration_bars,
            recovery_bars=p.recovery_bars,
            recovered=p.recovered,
        ))
    return ranked


# ──────────────────────────────────────────────────────────────────────────────
# Facade
# ──────────────────────────────────────────────────────────────────────────────
class DrawdownAnalyzer:
    """
    Full drawdown analysis for one equity curve.

    Parameters
    ----------
    label:
        Strategy / portfolio name used in charts and logs.

    Examples
    --------
    ::

        da = DrawdownAnalyzer(label="TCS-Heavy Hybrid")
        result = da.analyse(equity_curve)
        print(result.summary_df().to_string())
        da.plot(equity_curve, result, save_chart=True)
    """

    def __init__(self, label: str = "Portfolio") -> None:
        self.label = label

    def analyse(self, series: pd.Series) -> DrawdownResult:
        """
        Run the full drawdown analysis.

        Parameters
        ----------
        series:
            Portfolio equity curve (DatetimeIndex, daily).

        Returns
        -------
        DrawdownResult
        """
        series  = series.dropna().sort_index()
        dd_pct  = compute_drawdown_series(series)
        periods = _identify_periods(series)

        depths = [abs(p.depth_pct) for p in periods]
        durs   = [p.duration_bars  for p in periods]
        n_rec  = sum(1 for p in periods if p.recovered)
        n_tot  = len(periods)

        max_dd   = float(dd_pct.min()) if len(dd_pct) else 0.0
        avg_dd   = float(dd_pct[dd_pct < 0].mean()) if (dd_pct < 0).any() else 0.0
        med_dd   = float(np.median([p.depth_pct for p in periods])) if periods else 0.0
        pct_uw   = float((dd_pct < 0).mean()) * 100
        rec_rate = (n_rec / n_tot * 100) if n_tot > 0 else 100.0

        result = DrawdownResult(
            underwater_curve    = dd_pct,
            periods             = periods,
            max_drawdown_pct    = round(max_dd, 2),
            avg_drawdown_pct    = round(avg_dd, 2),
            median_drawdown_pct = round(med_dd, 2),
            max_duration_bars   = max(durs, default=0),
            avg_duration_bars   = round(float(np.mean(durs)), 1) if durs else 0.0,
            pct_time_underwater = round(pct_uw, 1),
            recovery_rate_pct   = round(rec_rate, 1),
            n_periods           = n_tot,
        )

        logger.info(
            "DrawdownAnalysis [%s]  max=%.2f%%  avg=%.2f%%  "
            "periods=%d  uw_time=%.1f%%  recovery_rate=%.1f%%",
            self.label, max_dd, avg_dd, n_tot, pct_uw, rec_rate,
        )
        return result

    # ── Statistics summary ────────────────────────────────────────────────────
    def statistics_table(self, result: DrawdownResult) -> pd.DataFrame:
        """
        Compact statistics table for the analytics report.

        Parameters
        ----------
        result:
            Output of :meth:`analyse`.

        Returns
        -------
        pd.DataFrame
            Single-column table of key drawdown statistics.
        """
        stats = {
            "Max Drawdown (%)":        result.max_drawdown_pct,
            "Avg Drawdown (%)":        result.avg_drawdown_pct,
            "Median Drawdown (%)":     result.median_drawdown_pct,
            "Max Duration (bars)":     result.max_duration_bars,
            "Avg Duration (bars)":     result.avg_duration_bars,
            "% Time Underwater":       result.pct_time_underwater,
            "No. of DD Periods":       result.n_periods,
            "Recovery Rate (%)":       result.recovery_rate_pct,
        }
        return pd.DataFrame.from_dict(
            stats, orient="index", columns=[self.label]
        )

    # ── Chart ─────────────────────────────────────────────────────────────────
    def plot(
        self,
        series: pd.Series,
        result: DrawdownResult,
        save_chart: bool = False,
        filename: Optional[str] = None,
    ) -> Optional[str]:
        """
        Two-panel chart: equity curve (top) + underwater plot (bottom).

        The top panel shades the top-5 drawdown periods in red.
        The bottom panel shows the full drawdown series filled in red.

        Parameters
        ----------
        series:
            Equity curve.
        result:
            Output of :meth:`analyse`.
        save_chart:
            If ``True``, save to ``reports/charts/``.
        filename:
            Override the default filename.

        Returns
        -------
        str or None
            Absolute path to the saved PNG, or ``None`` if not saved.
        """
        fig, axes = plt.subplots(
            2, 1, figsize=(14, 8),
            gridspec_kw={"height_ratios": [3, 1]},
            sharex=True,
        )
        fig.patch.set_facecolor("#0d1117")
        for ax in axes:
            ax.set_facecolor("#0d1117")
            ax.tick_params(colors="white")
            ax.spines["bottom"].set_color("#333")
            ax.spines["left"].set_color("#333")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        ax_eq, ax_dd = axes

        # ── Top: equity curve ────────────────────────────────────────────────
        dates = series.index
        ax_eq.plot(dates, series.values, color="#58a6ff", linewidth=1.5,
                   label=self.label)
        ax_eq.set_ylabel("Portfolio Value (₹)", color="white")
        ax_eq.yaxis.label.set_color("white")
        ax_eq.tick_params(colors="white")
        ax_eq.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"₹{x/1e5:.1f}L")
        )

        # Shade top-5 drawdowns on equity chart
        colors = ["#ff4444", "#ff7744", "#ffaa44", "#ffcc44", "#ffee44"]
        for i, period in enumerate(result.top(5)):
            end_d = period.recovery or series.index[-1]
            ax_eq.axvspan(period.start, end_d,
                          alpha=0.15, color=colors[i % len(colors)],
                          label=f"DD #{period.rank} ({period.depth_pct:.1f}%)")
        ax_eq.legend(facecolor="#161b22", edgecolor="#333",
                     labelcolor="white", fontsize=8, loc="upper left")
        ax_eq.set_title(f"Equity Curve & Drawdown Analysis — {self.label}",
                        color="white", fontsize=13, pad=12)

        # ── Bottom: underwater curve ─────────────────────────────────────────
        dd_vals = result.underwater_curve.reindex(dates).fillna(0)
        ax_dd.fill_between(dates, dd_vals.values, 0,
                           where=(dd_vals.values < 0),
                           color="#ff4444", alpha=0.6, linewidth=0)
        ax_dd.plot(dates, dd_vals.values, color="#ff6666", linewidth=0.8)
        ax_dd.axhline(0, color="#555", linewidth=0.8, linestyle="--")
        ax_dd.set_ylabel("Drawdown (%)", color="white")
        ax_dd.yaxis.label.set_color("white")
        ax_dd.tick_params(colors="white")
        ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax_dd.xaxis.set_major_locator(mdates.YearLocator())
        plt.setp(ax_dd.xaxis.get_majorticklabels(), color="white")

        plt.tight_layout()

        if save_chart:
            os.makedirs(CHARTS_DIR, exist_ok=True)
            fname = filename or f"drawdown_{self.label.replace(' ', '_')}.png"
            path  = os.path.join(CHARTS_DIR, fname)
            plt.savefig(path, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            plt.close()
            logger.info("Drawdown chart saved → %s", path)
            return path

        plt.close()
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Script entry-point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)
    )))
    if _root not in sys.path:
        sys.path.insert(0, _root)

    import logging as _lg; _lg.disable(_lg.CRITICAL)
    from src.portfolio.rebalance import simulate_portfolio, RebalanceFrequency

    STOCKS  = ["TCS", "RELIANCE", "INFOSYS"]
    CAPITAL = 1_000_000.0
    _lg.disable(_lg.NOTSET)

    configs = {
        "EW Hybrid":   ({"TCS": 1/3, "RELIANCE": 1/3, "INFOSYS": 1/3},
                        RebalanceFrequency.HYBRID, 0.05),
        "TCS-Heavy":   ({"TCS": 0.60, "RELIANCE": 0.25, "INFOSYS": 0.15},
                        RebalanceFrequency.HYBRID, 0.05),
    }

    print("=" * 60)
    print("  Drawdown Analysis — Demo")
    print("=" * 60)

    for label, (weights, freq, drift) in configs.items():
        result_sim = simulate_portfolio(
            STOCKS, weights, CAPITAL, freq, drift
        )
        series = result_sim.portfolio_value_series

        da     = DrawdownAnalyzer(label=label)
        result = da.analyse(series)

        print(f"\n{'─'*60}")
        print(f"  {label}")
        print(f"{'─'*60}")

        # Statistics table
        print(da.statistics_table(result).to_string())

        # Top-5 drawdowns
        print(f"\n  Top-{min(5, result.n_periods)} Drawdown Periods:")
        hdr = (f"  {'#':>3}  {'Peak Date':>12}  {'Trough Date':>12}  "
               f"{'Depth':>8}  {'₹ Loss':>12}  {'Dur(bars)':>10}  "
               f"{'Recov(bars)':>11}  {'Status':>10}")
        print(hdr)
        print("  " + "─" * (len(hdr) - 2))
        for p in result.top(5):
            recov_bars = str(p.recovery_bars) if p.recovery_bars else "  —"
            status     = "Recovered" if p.recovered else "OPEN"
            print(
                f"  {p.rank:>3}  {str(p.peak.date()):>12}  "
                f"{str(p.trough.date()):>12}  "
                f"{p.depth_pct:>7.2f}%  "
                f"₹{p.depth_inr:>10,.0f}  "
                f"{p.duration_bars:>10}  "
                f"{recov_bars:>11}  "
                f"{status:>10}"
            )

        # Full period table
        df_all = result.summary_df()
        print(f"\n  Total drawdown periods: {result.n_periods}")
        print(f"  Recovery rate: {result.recovery_rate_pct:.1f}%")
        print(f"  Time underwater: {result.pct_time_underwater:.1f}% of all bars")

        # Save chart
        chart_path = da.plot(series, result, save_chart=True)
        if chart_path:
            print(f"\n  Chart saved → {chart_path}")

    print()
