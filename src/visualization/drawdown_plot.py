"""
drawdown_plot.py
================
Drawdown visualization engine for the Algorithmic Trading Strategy
Backtester — Phase 7.

What this module produces
--------------------------
**Underwater plot** — a filled area chart showing the daily drawdown
percentage, with the single deepest point clearly annotated.

**Combined panel** — equity curve (top) + underwater curve (bottom)
sharing the same x-axis, the canonical "tear sheet" view used by
professional quant reports (QuantStats, pyfolio).

**Top-N drawdown periods chart** — horizontal bar chart ranking the
worst drawdown episodes by depth, with duration and recovery status
shown alongside.

**Recovery annotation** — marks the trough and (if recovered) the
recovery date directly on the underwater curve.

Design philosophy
------------------
This module is a *pure visualization layer* — it does not recompute
drawdown statistics itself.  It consumes the ``DrawdownResult`` object
produced by Phase 6's ``drawdown.py`` (or a raw equity-curve Series,
which it will analyse internally via the same compatible algorithm).
This avoids duplicating the drawdown-period-detection logic in two
places.

Integration
------------
::

    from src.analytics.drawdown import DrawdownAnalyzer
    from src.visualization.drawdown_plot import DrawdownPlotter

    analyzer = DrawdownAnalyzer(label="TCS-Heavy")
    result   = analyzer.analyse(equity_series)

    plotter  = DrawdownPlotter(title="TCS-Heavy Drawdown")
    plotter.plot_underwater(equity_series, result, save_chart=True)
    plotter.plot_combined(equity_series, result, save_chart=True)
    plotter.plot_top_periods(result, save_chart=True)

Usage
-----
::

    python src/visualization/drawdown_plot.py
"""

from __future__ import annotations

import logging
import os
import sys
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
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
# Theme — shared with equity_curve.py for visual consistency
# ──────────────────────────────────────────────────────────────────────────────
CHARTS_DIR: str = os.path.join("reports", "charts")

THEME = {
    "bg":      "#0d1117",
    "panel":   "#161b22",
    "grid":    "#21262d",
    "text":    "#e6edf3",
    "muted":   "#8b949e",
    "blue":    "#58a6ff",
    "green":   "#3fb950",
    "red":     "#f85149",
    "orange":  "#d29922",
    "border":  "#30363d",
}

_DD_PALETTE: List[str] = ["#f85149", "#ff7b72", "#ffa198", "#ffc7c3", "#ffe3e1"]


def _style_axes(ax: plt.Axes) -> None:
    """Apply the shared dark theme to a Matplotlib Axes object."""
    ax.set_facecolor(THEME["bg"])
    ax.tick_params(colors=THEME["text"], labelsize=9)
    ax.grid(True, color=THEME["grid"], linewidth=0.6, alpha=0.6)
    for spine in ax.spines.values():
        spine.set_color(THEME["border"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.label.set_color(THEME["muted"])
    ax.yaxis.label.set_color(THEME["muted"])


def _inr_formatter():
    """Matplotlib tick formatter: ₹1.2L / ₹3.4Cr style."""
    def _fmt(x: float, _pos: int) -> str:
        if abs(x) >= 1e7:
            return f"₹{x/1e7:.1f}Cr"
        if abs(x) >= 1e5:
            return f"₹{x/1e5:.1f}L"
        return f"₹{x:,.0f}"
    return mticker.FuncFormatter(_fmt)


# ──────────────────────────────────────────────────────────────────────────────
# Lightweight internal drawdown computation (kept in sync with Phase 6)
# ──────────────────────────────────────────────────────────────────────────────
def _compute_underwater(series: pd.Series) -> pd.Series:
    """
    Compute the daily drawdown percentage series from an equity curve.

    Duplicated (intentionally, as a small pure function) from
    ``src.analytics.drawdown.compute_drawdown_series`` so this module
    can render a chart even without importing the full Phase 6 module,
    e.g. for a quick exploratory plot.

    Parameters
    ----------
    series:
        Portfolio equity curve.

    Returns
    -------
    pd.Series
        Daily drawdown (%, ≤ 0).
    """
    peak = series.cummax()
    return (series - peak) / peak * 100


# ──────────────────────────────────────────────────────────────────────────────
# Facade
# ──────────────────────────────────────────────────────────────────────────────
class DrawdownPlotter:
    """
    Generates drawdown-focused visualisations for reports and dashboards.

    Parameters
    ----------
    title:
        Default chart title.
    figsize:
        Matplotlib figure size for single-panel charts.

    Examples
    --------
    ::

        plotter = DrawdownPlotter(title="TCS-Heavy Drawdown")
        plotter.plot_underwater(equity_series, save_chart=True)
        plotter.plot_combined(equity_series, save_chart=True)
    """

    def __init__(
        self,
        title:   str = "Drawdown Analysis",
        figsize: Tuple[float, float] = (14, 5),
    ) -> None:
        self.title   = title
        self.figsize = figsize

    # ── Underwater-only plot ─────────────────────────────────────────────────
    def plot_underwater(
        self,
        series: pd.Series,
        drawdown_result=None,        # Optional[DrawdownResult] from Phase 6
        save_chart: bool = False,
        filename: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot the underwater (drawdown) curve with the max drawdown annotated.

        Parameters
        ----------
        series:
            Portfolio equity curve.
        drawdown_result:
            Optional ``DrawdownResult`` from ``src.analytics.drawdown``.
            If provided, peak/trough/recovery dates are taken directly
            from it (more accurate for multi-episode series).  If
            omitted, the single global max drawdown is computed and
            annotated instead.
        save_chart:
            Save PNG to ``reports/charts/``.
        filename:
            Override default filename.

        Returns
        -------
        (Figure, Axes)
        """
        series = series.dropna().sort_index()
        if series.empty:
            raise ValueError("series is empty after dropping NaNs.")

        dd = _compute_underwater(series)

        fig, ax = plt.subplots(figsize=self.figsize, facecolor=THEME["bg"])
        _style_axes(ax)

        ax.fill_between(dd.index, dd.values, 0,
                        where=(dd.values < 0), color=THEME["red"],
                        alpha=0.55, linewidth=0)
        ax.plot(dd.index, dd.values, color="#ff8a80", linewidth=0.9)
        ax.axhline(0, color=THEME["muted"], linewidth=0.8, linestyle="--")

        # ── Annotate the deepest point ────────────────────────────────────────
        if drawdown_result is not None and getattr(drawdown_result, "periods", None):
            worst = drawdown_result.periods[0]   # rank 1 = deepest
            trough_date, depth = worst.trough, worst.depth_pct
            recovery = worst.recovery
        else:
            trough_date = dd.idxmin()
            depth       = float(dd.min())
            recovery    = None

        ax.scatter([trough_date], [depth], color=THEME["orange"], s=70,
                   zorder=5, edgecolor="white", linewidth=0.8)
        ax.annotate(
            f"Max DD: {depth:.2f}%\n{trough_date.date()}",
            xy=(trough_date, depth), xytext=(15, -25),
            textcoords="offset points", color=THEME["text"], fontsize=9,
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=THEME["orange"], lw=1.2),
            bbox=dict(boxstyle="round,pad=0.35", facecolor=THEME["panel"],
                      edgecolor=THEME["orange"], alpha=0.95),
        )

        if recovery is not None:
            ax.scatter([recovery], [0], color=THEME["green"], s=55,
                       zorder=5, marker="D", edgecolor="white", linewidth=0.6,
                       label=f"Recovered: {recovery.date()}")
            ax.legend(facecolor=THEME["panel"], edgecolor=THEME["border"],
                      labelcolor=THEME["text"], fontsize=8.5, loc="lower right")

        ax.set_ylabel("Drawdown (%)", color=THEME["muted"])
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.set_title(self.title, color=THEME["text"], fontsize=14, pad=14,
                     fontweight="bold")

        plt.tight_layout()
        if save_chart:
            self._save(fig, filename or "drawdown_underwater.png")
        return fig, ax

    # ── Combined equity + drawdown panel ─────────────────────────────────────
    def plot_combined(
        self,
        series: pd.Series,
        drawdown_result=None,
        top_n_shade: int = 3,
        save_chart: bool = False,
        filename: Optional[str] = None,
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        Two-panel "tear sheet" chart: equity curve on top, underwater below.

        The equity panel shades the *top_n_shade* deepest drawdown periods
        directly over the price action, making it easy to see exactly
        which market events caused the worst pain.

        Parameters
        ----------
        series:
            Portfolio equity curve.
        drawdown_result:
            Optional ``DrawdownResult`` — enables period shading on the
            equity panel.  Without it, only the underwater curve is shown.
        top_n_shade:
            Number of worst periods to shade on the equity panel.
        save_chart:
            Save PNG.
        filename:
            Override default filename.

        Returns
        -------
        (Figure, array of 2 Axes)
        """
        series = series.dropna().sort_index()
        dd     = _compute_underwater(series)

        fig, axes = plt.subplots(
            2, 1, figsize=(14, 8.5),
            gridspec_kw={"height_ratios": [2.2, 1]},
            sharex=True, facecolor=THEME["bg"],
        )
        ax_eq, ax_dd = axes
        for ax in axes:
            _style_axes(ax)

        # ── Top panel: equity curve ───────────────────────────────────────────
        ax_eq.plot(series.index, series.values, color=THEME["blue"],
                   linewidth=1.5)
        ax_eq.yaxis.set_major_formatter(_inr_formatter())
        ax_eq.set_ylabel("Portfolio Value", color=THEME["muted"])
        ax_eq.set_title(self.title, color=THEME["text"], fontsize=14,
                        pad=14, fontweight="bold")

        if drawdown_result is not None and getattr(drawdown_result, "periods", None):
            for i, period in enumerate(drawdown_result.top(top_n_shade)):
                end_d  = period.recovery or series.index[-1]
                colour = _DD_PALETTE[i % len(_DD_PALETTE)]
                ax_eq.axvspan(period.start, end_d, alpha=0.18, color=colour,
                              label=f"DD#{period.rank} ({period.depth_pct:.1f}%)")
            ax_eq.legend(facecolor=THEME["panel"], edgecolor=THEME["border"],
                        labelcolor=THEME["text"], fontsize=8, loc="upper left")

        # ── Bottom panel: underwater curve ────────────────────────────────────
        ax_dd.fill_between(dd.index, dd.values, 0, where=(dd.values < 0),
                           color=THEME["red"], alpha=0.55, linewidth=0)
        ax_dd.plot(dd.index, dd.values, color="#ff8a80", linewidth=0.8)
        ax_dd.axhline(0, color=THEME["muted"], linewidth=0.8, linestyle="--")
        ax_dd.set_ylabel("Drawdown (%)", color=THEME["muted"])
        ax_dd.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        plt.tight_layout()
        if save_chart:
            self._save(fig, filename or "drawdown_combined.png")
        return fig, axes

    # ── Top-N drawdown periods bar chart ─────────────────────────────────────
    def plot_top_periods(
        self,
        drawdown_result,
        top_n: int = 8,
        save_chart: bool = False,
        filename: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Horizontal bar chart ranking the worst drawdown episodes.

        Each bar shows the depth (%); annotations show duration in bars
        and whether the episode has recovered.

        Parameters
        ----------
        drawdown_result:
            ``DrawdownResult`` from ``src.analytics.drawdown``.
        top_n:
            Number of worst periods to display.
        save_chart:
            Save PNG.
        filename:
            Override default filename.

        Returns
        -------
        (Figure, Axes)

        Raises
        ------
        ValueError
            If *drawdown_result* has no periods.
        """
        periods = drawdown_result.top(top_n)
        if not periods:
            raise ValueError("drawdown_result contains no drawdown periods to plot.")

        labels  = [f"#{p.rank}  {p.peak.date()}" for p in periods]
        depths  = [p.depth_pct for p in periods]
        durs    = [p.duration_bars for p in periods]
        statuses = [p.recovered for p in periods]

        fig, ax = plt.subplots(
            figsize=(11, max(3.5, 0.6 * len(periods))), facecolor=THEME["bg"]
        )
        _style_axes(ax)

        y_pos  = np.arange(len(periods))[::-1]
        colors = [THEME["green"] if s else THEME["red"] for s in statuses]
        bars   = ax.barh(y_pos, depths, color=colors, alpha=0.85, height=0.6)

        for bar, dur, status in zip(bars, durs, statuses):
            tag = "✓ Recovered" if status else "✗ Open"
            ax.text(
                bar.get_width() - 0.3, bar.get_y() + bar.get_height() / 2,
                f"{dur}d  {tag}", ha="right", va="center",
                color="white", fontsize=8.5, fontweight="bold",
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, color=THEME["text"], fontsize=9)
        ax.set_xlabel("Drawdown Depth (%)", color=THEME["muted"])
        ax.set_title(f"Top {len(periods)} Drawdown Periods — {self.title}",
                     color=THEME["text"], fontsize=13, pad=12, fontweight="bold")
        ax.axvline(0, color=THEME["muted"], linewidth=0.8)

        plt.tight_layout()
        if save_chart:
            self._save(fig, filename or "drawdown_top_periods.png")
        return fig, ax

    # ── Comparative underwater (multiple strategies) ─────────────────────────
    def plot_underwater_comparison(
        self,
        series_dict: dict,
        save_chart: bool = False,
        filename: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Overlay multiple underwater curves for strategy comparison.

        Parameters
        ----------
        series_dict:
            ``{label: equity_series}``.
        save_chart:
            Save PNG.
        filename:
            Override default filename.

        Returns
        -------
        (Figure, Axes)
        """
        palette = ["#f85149", "#58a6ff", "#3fb950", "#d29922", "#bc8cff"]

        fig, ax = plt.subplots(figsize=self.figsize, facecolor=THEME["bg"])
        _style_axes(ax)

        for i, (label, series) in enumerate(series_dict.items()):
            dd = _compute_underwater(series.dropna().sort_index())
            colour = palette[i % len(palette)]
            ax.plot(dd.index, dd.values, color=colour, linewidth=1.3,
                    label=f"{label} (max {dd.min():.1f}%)")

        ax.axhline(0, color=THEME["muted"], linewidth=0.8, linestyle="--")
        ax.set_ylabel("Drawdown (%)", color=THEME["muted"])
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.set_title(f"Drawdown Comparison — {self.title}",
                     color=THEME["text"], fontsize=14, pad=14, fontweight="bold")
        ax.legend(facecolor=THEME["panel"], edgecolor=THEME["border"],
                  labelcolor=THEME["text"], fontsize=9, loc="lower left")

        plt.tight_layout()
        if save_chart:
            self._save(fig, filename or "drawdown_comparison.png")
        return fig, ax

    # ── Save helper ───────────────────────────────────────────────────────────
    def _save(self, fig: plt.Figure, filename: str) -> str:
        """Save *fig* as PNG to ``reports/charts/`` and return the path."""
        os.makedirs(CHARTS_DIR, exist_ok=True)
        path = os.path.join(CHARTS_DIR, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        logger.info("Drawdown chart saved → %s", path)
        return path


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
    from src.analytics.drawdown  import DrawdownAnalyzer

    STOCKS  = ["TCS", "RELIANCE", "INFOSYS"]
    CAPITAL = 1_000_000.0

    r_ew = simulate_portfolio(
        STOCKS, {s: 1/3 for s in STOCKS}, CAPITAL,
        RebalanceFrequency.HYBRID, 0.05,
    )
    r_th = simulate_portfolio(
        STOCKS, {"TCS": 0.60, "RELIANCE": 0.25, "INFOSYS": 0.15}, CAPITAL,
        RebalanceFrequency.HYBRID, 0.05,
    )

    _lg.disable(_lg.NOTSET)
    print("=" * 60)
    print("  Drawdown Visualization — Demo")
    print("=" * 60)

    analyzer  = DrawdownAnalyzer(label="TCS-Heavy")
    dd_result = analyzer.analyse(r_th.portfolio_value_series)

    plotter = DrawdownPlotter(title="TCS-Heavy Portfolio")

    # 1. Underwater-only plot
    plotter.plot_underwater(
        r_th.portfolio_value_series, dd_result,
        save_chart=True, filename="dd_underwater.png",
    )
    print("\n  ✓  Saved: dd_underwater.png")

    # 2. Combined equity + drawdown tear sheet
    plotter.plot_combined(
        r_th.portfolio_value_series, dd_result, top_n_shade=3,
        save_chart=True, filename="dd_combined_tearsheet.png",
    )
    print("  ✓  Saved: dd_combined_tearsheet.png")

    # 3. Top drawdown periods bar chart
    plotter.plot_top_periods(
        dd_result, top_n=6,
        save_chart=True, filename="dd_top_periods.png",
    )
    print("  ✓  Saved: dd_top_periods.png")

    # 4. Underwater comparison across strategies
    plotter.plot_underwater_comparison(
        {
            "Equal Weight": r_ew.portfolio_value_series,
            "TCS-Heavy":    r_th.portfolio_value_series,
        },
        save_chart=True, filename="dd_comparison.png",
    )
    print("  ✓  Saved: dd_comparison.png")

    print(f"\n  Max drawdown: {dd_result.max_drawdown_pct:.2f}%  "
          f"({dd_result.n_periods} total periods, "
          f"{dd_result.recovery_rate_pct:.1f}% recovery rate)")
    print(f"  All charts saved to {CHARTS_DIR}/")
    print()
