"""
heatmap.py
==========
Returns heatmap visualization engine for the Algorithmic Trading
Strategy Backtester — Phase 7.

What this module produces
--------------------------
**Monthly returns heatmap** — the canonical "calendar of returns" view
used in every professional quant tear sheet.  Rows = years, columns =
Jan–Dec, cell colour and annotation = that month's % return, plus a
"Full Year" summary column.

**Weekly returns heatmap** — same concept at ISO-week granularity,
useful for shorter backtests or detailed seasonality inspection.

**Strategy comparison heatmap** — annual returns for multiple
strategies side-by-side (rows = strategies, columns = years), making
it trivial to spot which year each strategy struggled or excelled in.

**Day-of-week / Month-of-year seasonality heatmap** — average return
by calendar day-of-week × month, revealing systematic seasonal patterns
(e.g. "Indian markets tend to rally in November–December").

Design philosophy
------------------
All heatmaps use a diverging red-white-green colour map centred at
zero, so the eye immediately distinguishes profitable months (green)
from losing months (red) regardless of the absolute scale.

Integration
------------
Consumes the daily portfolio value series from ``rebalance.py``
(Phase 5) or any equity curve, and can also accept a pre-computed
monthly-returns DataFrame from ``performance_metrics.py`` (Phase 6).

Usage
-----
::

    python src/visualization/heatmap.py
    from src.visualization.heatmap import ReturnsHeatmapPlotter
"""

from __future__ import annotations

import calendar
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns

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
# Theme — shared with equity_curve.py / drawdown_plot.py
# ──────────────────────────────────────────────────────────────────────────────
CHARTS_DIR: str = os.path.join("reports", "charts")

THEME = {
    "bg":     "#0d1117",
    "panel":  "#161b22",
    "text":   "#e6edf3",
    "muted":  "#8b949e",
    "border": "#30363d",
}

_MONTH_NAMES = {i: calendar.month_abbr[i] for i in range(1, 13)}
_DAY_NAMES   = ["Mon", "Tue", "Wed", "Thu", "Fri"]


# ──────────────────────────────────────────────────────────────────────────────
# Internal helper — diverging colormap centred at zero
# ──────────────────────────────────────────────────────────────────────────────
def _diverging_cmap() -> mcolors.LinearSegmentedColormap:
    """
    Return a red → dark → green diverging colormap for return heatmaps.

    Dark midpoint matches the dashboard's dark theme so zero-return
    cells blend naturally with the figure background.
    """
    return mcolors.LinearSegmentedColormap.from_list(
        "returns_diverging",
        ["#f85149", "#3d1f1f", "#161b22", "#1f3d2a", "#3fb950"],
        N=256,
    )


def _style_figure(fig: plt.Figure, ax: plt.Axes) -> None:
    """Apply the shared dark theme to a heatmap figure/axes pair."""
    fig.patch.set_facecolor(THEME["bg"])
    ax.set_facecolor(THEME["bg"])
    ax.tick_params(colors=THEME["text"])
    for spine in ax.spines.values():
        spine.set_visible(False)


# ──────────────────────────────────────────────────────────────────────────────
# Data preparation
# ──────────────────────────────────────────────────────────────────────────────
def compute_monthly_returns_table(value_series: pd.Series) -> pd.DataFrame:
    """
    Pivot a daily equity curve into a year × month % return table.

    Parameters
    ----------
    value_series:
        Daily portfolio value, DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        Rows = years, columns = Jan..Dec (+ 'Full Year'), values = % return.

    Raises
    ------
    ValueError
        If *value_series* has fewer than 2 data points.
    """
    if len(value_series) < 2:
        raise ValueError("value_series must have ≥ 2 points.")

    monthly = value_series.resample("ME").last().pct_change().dropna() * 100
    df = monthly.to_frame("ret")
    df["year"]  = df.index.year
    df["month"] = df.index.month

    pivot = df.pivot_table(values="ret", index="year", columns="month")
    pivot = pivot.reindex(columns=range(1, 13))
    pivot.columns = [_MONTH_NAMES[c] for c in pivot.columns]

    # Full-year compounded return per row
    full_year = []
    for year in pivot.index:
        year_rets = monthly[monthly.index.year == year] / 100
        compounded = (np.prod(1 + year_rets) - 1) * 100
        full_year.append(compounded)
    pivot["Full Year"] = full_year

    return pivot.round(2)


def compute_weekly_returns_table(value_series: pd.Series) -> pd.DataFrame:
    """
    Pivot a daily equity curve into a year × ISO-week % return table.

    Parameters
    ----------
    value_series:
        Daily portfolio value, DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        Rows = years, columns = week number (1–53), values = % return.
    """
    if len(value_series) < 2:
        raise ValueError("value_series must have ≥ 2 points.")

    weekly = value_series.resample("W-FRI").last().pct_change().dropna() * 100
    df = weekly.to_frame("ret")
    iso = df.index.isocalendar()
    df["year"] = iso["year"].values
    df["week"] = iso["week"].values

    pivot = df.pivot_table(values="ret", index="year", columns="week")
    return pivot.round(2)


def compute_day_month_seasonality(value_series: pd.Series) -> pd.DataFrame:
    """
    Average daily return by (day-of-week × month) — seasonality matrix.

    Parameters
    ----------
    value_series:
        Daily portfolio value, DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        Rows = Mon..Fri, columns = Jan..Dec, values = mean daily return (%).
    """
    rets = value_series.pct_change().dropna() * 100
    df   = rets.to_frame("ret")
    df["dow"]   = df.index.dayofweek
    df["month"] = df.index.month
    df = df[df["dow"] < 5]  # Indian markets: Mon–Fri only

    pivot = df.pivot_table(values="ret", index="dow", columns="month", aggfunc="mean")
    pivot = pivot.reindex(index=range(5), columns=range(1, 13))
    pivot.index   = _DAY_NAMES
    pivot.columns = [_MONTH_NAMES[c] for c in pivot.columns]
    return pivot.round(3)


# ──────────────────────────────────────────────────────────────────────────────
# Facade
# ──────────────────────────────────────────────────────────────────────────────
class ReturnsHeatmapPlotter:
    """
    Generates returns-heatmap visualisations for reports and dashboards.

    Parameters
    ----------
    title:
        Default chart title.

    Examples
    --------
    ::

        plotter = ReturnsHeatmapPlotter(title="TCS-Heavy Portfolio")
        plotter.plot_monthly_heatmap(equity_series, save_chart=True)
        plotter.plot_weekly_heatmap(equity_series, save_chart=True)
        plotter.plot_seasonality(equity_series, save_chart=True)
    """

    def __init__(self, title: str = "Returns Heatmap") -> None:
        self.title = title
        self.cmap  = _diverging_cmap()

    # ── Monthly heatmap ───────────────────────────────────────────────────────
    def plot_monthly_heatmap(
        self,
        value_series: pd.Series,
        annotate: bool   = True,
        save_chart: bool = False,
        filename: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot the classic year × month returns calendar heatmap.

        Parameters
        ----------
        value_series:
            Daily portfolio value series.
        annotate:
            Print the % value inside each cell.
        save_chart:
            Save PNG to ``reports/charts/``.
        filename:
            Override default filename.

        Returns
        -------
        (Figure, Axes)
        """
        table = compute_monthly_returns_table(value_series)

        # Separate the 'Full Year' column for distinct styling
        monthly_part = table.drop(columns=["Full Year"])
        year_part    = table[["Full Year"]]

        fig, (ax, ax_year) = plt.subplots(
            1, 2, figsize=(14, max(3.2, 0.62 * len(table) + 1.4)),
            gridspec_kw={"width_ratios": [12, 1.3], "wspace": 0.06},
            facecolor=THEME["bg"],
        )
        _style_figure(fig, ax)
        _style_figure(fig, ax_year)

        vmax = max(abs(monthly_part.min().min()), abs(monthly_part.max().max()), 1)

        sns.heatmap(
            monthly_part, ax=ax, cmap=self.cmap, center=0,
            vmin=-vmax, vmax=vmax,
            annot=annotate, fmt=".1f",
            annot_kws={"fontsize": 8.5, "color": "white"},
            linewidths=0.8, linecolor=THEME["bg"],
            cbar_kws={"label": "Monthly Return (%)"},
        )
        ax.set_title(self.title, color=THEME["text"], fontsize=14,
                     pad=14, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Year", color=THEME["muted"])
        ax.tick_params(colors=THEME["text"])
        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.set_tick_params(color=THEME["text"])
        cbar.ax.tick_params(labelcolor=THEME["text"])
        cbar.set_label("Monthly Return (%)", color=THEME["muted"])

        sns.heatmap(
            year_part, ax=ax_year, cmap=self.cmap, center=0,
            vmin=-vmax, vmax=vmax,
            annot=annotate, fmt=".1f",
            annot_kws={"fontsize": 9, "color": "white", "fontweight": "bold"},
            linewidths=0.8, linecolor=THEME["bg"],
            cbar=False, yticklabels=False,
        )
        ax_year.set_xlabel("")
        ax_year.set_ylabel("")
        ax_year.tick_params(colors=THEME["text"])

        fig.subplots_adjust(left=0.08, right=0.96, top=0.90, bottom=0.08)
        if save_chart:
            self._save(fig, filename or "heatmap_monthly.png")
        return fig, ax

    # ── Weekly heatmap ────────────────────────────────────────────────────────
    def plot_weekly_heatmap(
        self,
        value_series: pd.Series,
        save_chart: bool = False,
        filename: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot a year × ISO-week returns heatmap.

        Cell annotation is omitted by default (53 columns is too dense)
        but colour intensity still conveys magnitude clearly.

        Parameters
        ----------
        value_series:
            Daily portfolio value series.
        save_chart:
            Save PNG.
        filename:
            Override default filename.

        Returns
        -------
        (Figure, Axes)
        """
        table = compute_weekly_returns_table(value_series)
        vmax  = max(abs(np.nanmin(table.values)), abs(np.nanmax(table.values)), 1)

        fig, ax = plt.subplots(
            figsize=(16, max(2.6, 0.55 * len(table) + 1.2)), facecolor=THEME["bg"]
        )
        _style_figure(fig, ax)

        sns.heatmap(
            table, ax=ax, cmap=self.cmap, center=0,
            vmin=-vmax, vmax=vmax, annot=False,
            linewidths=0.3, linecolor=THEME["bg"],
            cbar_kws={"label": "Weekly Return (%)"},
        )
        ax.set_title(f"{self.title} — Weekly Returns",
                     color=THEME["text"], fontsize=14, pad=14, fontweight="bold")
        ax.set_xlabel("ISO Week", color=THEME["muted"])
        ax.set_ylabel("Year", color=THEME["muted"])
        ax.tick_params(colors=THEME["text"], labelsize=7)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelcolor=THEME["text"])
        cbar.set_label("Weekly Return (%)", color=THEME["muted"])

        plt.tight_layout()
        if save_chart:
            self._save(fig, filename or "heatmap_weekly.png")
        return fig, ax

    # ── Seasonality heatmap ──────────────────────────────────────────────────
    def plot_seasonality(
        self,
        value_series: pd.Series,
        save_chart: bool = False,
        filename: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot average daily return by day-of-week × month.

        Reveals systematic seasonal patterns — e.g. whether Mondays
        underperform, or whether November–December rallies are
        consistent across years (a commonly cited NSE seasonal pattern).

        Parameters
        ----------
        value_series:
            Daily portfolio value series.
        save_chart:
            Save PNG.
        filename:
            Override default filename.

        Returns
        -------
        (Figure, Axes)
        """
        table = compute_day_month_seasonality(value_series)
        vmax  = max(abs(np.nanmin(table.values)), abs(np.nanmax(table.values)), 0.01)

        fig, ax = plt.subplots(figsize=(12, 4.2), facecolor=THEME["bg"])
        _style_figure(fig, ax)

        sns.heatmap(
            table, ax=ax, cmap=self.cmap, center=0,
            vmin=-vmax, vmax=vmax, annot=True, fmt=".2f",
            annot_kws={"fontsize": 8, "color": "white"},
            linewidths=0.8, linecolor=THEME["bg"],
            cbar_kws={"label": "Avg Daily Return (%)"},
        )
        ax.set_title(f"{self.title} — Day-of-Week × Month Seasonality",
                     color=THEME["text"], fontsize=13, pad=14, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(colors=THEME["text"])
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelcolor=THEME["text"])
        cbar.set_label("Avg Daily Return (%)", color=THEME["muted"])

        plt.tight_layout()
        if save_chart:
            self._save(fig, filename or "heatmap_seasonality.png")
        return fig, ax

    # ── Strategy comparison heatmap ──────────────────────────────────────────
    def plot_strategy_comparison(
        self,
        series_dict: Dict[str, pd.Series],
        save_chart: bool = False,
        filename: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Annual returns heatmap across multiple strategies.

        Rows = strategy, columns = year — instantly shows which
        strategy handled which market regime best.

        Parameters
        ----------
        series_dict:
            ``{strategy_label: equity_series}``.
        save_chart:
            Save PNG.
        filename:
            Override default filename.

        Returns
        -------
        (Figure, Axes)
        """
        rows = {}
        for label, series in series_dict.items():
            yearly = series.resample("YE").last().pct_change()
            yearly.iloc[0] = series.resample("YE").last().iloc[0] / series.iloc[0] - 1
            yearly.index = yearly.index.year
            rows[label] = yearly * 100
        table = pd.DataFrame(rows).T.round(2)

        vmax = max(abs(table.min().min()), abs(table.max().max()), 1)

        fig, ax = plt.subplots(
            figsize=(max(8, 1.3 * len(table.columns)), max(3, 0.7 * len(table) + 1.2)),
            facecolor=THEME["bg"],
        )
        _style_figure(fig, ax)

        sns.heatmap(
            table, ax=ax, cmap=self.cmap, center=0,
            vmin=-vmax, vmax=vmax, annot=True, fmt=".1f",
            annot_kws={"fontsize": 9, "color": "white", "fontweight": "bold"},
            linewidths=1.0, linecolor=THEME["bg"],
            cbar_kws={"label": "Annual Return (%)"},
        )
        ax.set_title("Strategy Comparison — Annual Returns",
                     color=THEME["text"], fontsize=13, pad=14, fontweight="bold")
        ax.set_xlabel("Year", color=THEME["muted"])
        ax.set_ylabel("")
        ax.tick_params(colors=THEME["text"])
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelcolor=THEME["text"])
        cbar.set_label("Annual Return (%)", color=THEME["muted"])

        plt.tight_layout()
        if save_chart:
            self._save(fig, filename or "heatmap_strategy_comparison.png")
        return fig, ax

    # ── Save helper ───────────────────────────────────────────────────────────
    def _save(self, fig: plt.Figure, filename: str) -> str:
        """Save *fig* as PNG to ``reports/charts/`` and return the path."""
        os.makedirs(CHARTS_DIR, exist_ok=True)
        path = os.path.join(CHARTS_DIR, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        logger.info("Heatmap saved → %s", path)
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
    r_rp = simulate_portfolio(
        STOCKS, {"TCS": 0.4182, "RELIANCE": 0.3112, "INFOSYS": 0.2706},
        CAPITAL, RebalanceFrequency.HYBRID, 0.05,
    )

    _lg.disable(_lg.NOTSET)
    print("=" * 60)
    print("  Returns Heatmap Visualization — Demo")
    print("=" * 60)

    plotter = ReturnsHeatmapPlotter(title="TCS-Heavy Portfolio")

    # 1. Monthly returns calendar heatmap
    plotter.plot_monthly_heatmap(
        r_th.portfolio_value_series,
        save_chart=True, filename="heatmap_monthly_tcs_heavy.png",
    )
    print("\n  ✓  Saved: heatmap_monthly_tcs_heavy.png")

    # 2. Weekly returns heatmap
    plotter.plot_weekly_heatmap(
        r_th.portfolio_value_series,
        save_chart=True, filename="heatmap_weekly_tcs_heavy.png",
    )
    print("  ✓  Saved: heatmap_weekly_tcs_heavy.png")

    # 3. Seasonality heatmap
    plotter.plot_seasonality(
        r_th.portfolio_value_series,
        save_chart=True, filename="heatmap_seasonality_tcs_heavy.png",
    )
    print("  ✓  Saved: heatmap_seasonality_tcs_heavy.png")

    # 4. Strategy comparison heatmap
    plotter.plot_strategy_comparison(
        {
            "Equal Weight": r_ew.portfolio_value_series,
            "TCS-Heavy":    r_th.portfolio_value_series,
            "Risk Parity":  r_rp.portfolio_value_series,
        },
        save_chart=True, filename="heatmap_strategy_comparison.png",
    )
    print("  ✓  Saved: heatmap_strategy_comparison.png")

    # Console preview of the monthly table
    print("\n── Monthly Returns Table (TCS-Heavy) ────────────────────────")
    table = compute_monthly_returns_table(r_th.portfolio_value_series)
    print(table.to_string(float_format=lambda x: f"{x:>+.1f}" if pd.notna(x) else "  ·"))

    print(f"\n  All charts saved to {CHARTS_DIR}/")
    print()
