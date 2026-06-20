"""
equity_curve.py
================
Equity curve visualization engine for the Algorithmic Trading Strategy
Backtester — Phase 7.

What this module produces
--------------------------
**Static charts (Matplotlib)** — presentation-ready PNGs with:
* Single-strategy equity curve
* Strategy vs benchmark overlay (e.g. portfolio vs buy-and-hold TCS)
* Multi-strategy comparison (several curves on one axis)
* Optional entry/exit trade markers
* Optional normalised "growth of ₹1" view for fair comparison
* Optional log-scale y-axis for long backtests

**Interactive charts (Plotly)** — HTML output for dashboards, with
hover tooltips showing exact date/value, zoom/pan, and the same
overlay capabilities as the static version.

Design philosophy
------------------
Every public function returns the Matplotlib ``Figure``/Axes or the
Plotly ``Figure`` object *in addition to* optionally saving a PNG/HTML
file.  This means the same function can be:
* called directly in a Jupyter notebook for exploration,
* embedded in a Streamlit dashboard (``st.pyplot(fig)`` /
  ``st.plotly_chart(fig)``),
* or used headlessly to batch-generate report images.

Integration
------------
Consumes ``portfolio_value_series`` from ``rebalance.py`` (Phase 5) or
any Backtrader equity curve, and trade markers from
``trade_analysis.py`` (Phase 6).

Usage
-----
::

    python src/visualization/equity_curve.py
    from src.visualization.equity_curve import EquityCurvePlotter
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")          # headless-safe
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    _PLOTLY_AVAILABLE = True
except ImportError:                                    # pragma: no cover
    _PLOTLY_AVAILABLE = False

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
# Theme constants — shared visual identity across Phase 7 charts
# ──────────────────────────────────────────────────────────────────────────────
CHARTS_DIR: str = os.path.join("reports", "charts")

THEME = {
    "bg":        "#0d1117",
    "panel":     "#161b22",
    "grid":      "#21262d",
    "text":      "#e6edf3",
    "muted":     "#8b949e",
    "blue":      "#58a6ff",
    "green":     "#3fb950",
    "red":       "#f85149",
    "orange":    "#d29922",
    "purple":    "#bc8cff",
    "border":    "#30363d",
}

#: Cycle of distinct colours for multi-curve comparisons.
_PALETTE: List[str] = [
    THEME["blue"], THEME["green"], THEME["orange"],
    THEME["purple"], THEME["red"], "#39c5cf",
]


# ──────────────────────────────────────────────────────────────────────────────
# Internal styling helper
# ──────────────────────────────────────────────────────────────────────────────
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


def _inr_formatter(short: bool = True):
    """
    Return a Matplotlib tick formatter for INR values.

    Parameters
    ----------
    short:
        If ``True``, format large values as ₹1.2L / ₹3.4Cr.
    """
    def _fmt(x: float, _pos: int) -> str:
        if not short:
            return f"₹{x:,.0f}"
        if abs(x) >= 1e7:
            return f"₹{x/1e7:.1f}Cr"
        if abs(x) >= 1e5:
            return f"₹{x/1e5:.1f}L"
        return f"₹{x:,.0f}"
    return mticker.FuncFormatter(_fmt)


# ──────────────────────────────────────────────────────────────────────────────
# Data preparation helpers
# ──────────────────────────────────────────────────────────────────────────────
def normalise_to_growth(series: pd.Series, base: float = 1.0) -> pd.Series:
    """
    Rescale an equity curve to a "growth of ₹*base*" series.

    Useful for comparing strategies with different starting capital on
    the same chart — every curve starts at the same point.

    Parameters
    ----------
    series:
        Raw portfolio value series.
    base:
        Starting value to normalise to (default 1.0 → "growth of ₹1").

    Returns
    -------
    pd.Series
        Normalised series, same index, starting at *base*.
    """
    if series.empty or series.iloc[0] == 0:
        raise ValueError("Cannot normalise an empty series or one starting at 0.")
    return series / series.iloc[0] * base


def align_series(
    series_dict: Dict[str, pd.Series],
    method: str = "inner",
) -> pd.DataFrame:
    """
    Align multiple equity curves onto a common DatetimeIndex.

    Parameters
    ----------
    series_dict:
        ``{label: series}``.
    method:
        ``'inner'`` (intersection of dates) or ``'outer'`` (union, then
        forward-fill gaps).

    Returns
    -------
    pd.DataFrame
        One column per label.

    Raises
    ------
    ValueError
        If *method* is not recognised or series_dict is empty.
    """
    if not series_dict:
        raise ValueError("series_dict must not be empty.")
    if method not in ("inner", "outer"):
        raise ValueError(f"method must be 'inner' or 'outer', got {method!r}")

    df = pd.DataFrame(series_dict)
    if method == "inner":
        df = df.dropna()
    else:
        df = df.ffill().dropna()
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Facade
# ──────────────────────────────────────────────────────────────────────────────
class EquityCurvePlotter:
    """
    Generates static and interactive equity-curve visualisations.

    Parameters
    ----------
    title:
        Default chart title.
    figsize:
        Matplotlib figure size in inches.

    Examples
    --------
    Single curve::

        plotter = EquityCurvePlotter(title="TCS-Heavy Portfolio")
        fig, ax = plotter.plot_single(equity_series, save_chart=True)

    Strategy vs benchmark::

        fig, ax = plotter.plot_with_benchmark(
            equity_series, benchmark_series,
            strategy_label="TCS-Heavy", benchmark_label="Buy & Hold TCS",
        )

    Multi-strategy comparison::

        fig, ax = plotter.plot_comparison(
            {"EW Hybrid": s1, "TCS-Heavy": s2, "Risk Parity": s3}
        )
    """

    def __init__(
        self,
        title:   str = "Portfolio Equity Curve",
        figsize: Tuple[float, float] = (14, 6.5),
    ) -> None:
        self.title   = title
        self.figsize = figsize

    # ── Single curve ──────────────────────────────────────────────────────────
    def plot_single(
        self,
        series: pd.Series,
        label: str = "Portfolio",
        trade_markers: Optional[pd.DataFrame] = None,
        log_scale: bool = False,
        save_chart: bool = False,
        filename: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot a single equity curve, optionally with trade entry/exit markers.

        Parameters
        ----------
        series:
            Portfolio value series (DatetimeIndex).
        label:
            Legend label.
        trade_markers:
            Optional DataFrame with columns ``['date', 'action', 'price']``
            (or ``'entry_date'``/``'exit_date'``) to overlay buy/sell markers.
        log_scale:
            Use log y-axis — useful for long backtests with compounding growth.
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

        fig, ax = plt.subplots(figsize=self.figsize, facecolor=THEME["bg"])
        _style_axes(ax)

        ax.plot(series.index, series.values, color=THEME["blue"],
                linewidth=1.6, label=label)
        ax.fill_between(series.index, series.values, series.values.min(),
                        color=THEME["blue"], alpha=0.08)

        if trade_markers is not None:
            self._overlay_trade_markers(ax, series, trade_markers)

        if log_scale:
            ax.set_yscale("log")

        ax.yaxis.set_major_formatter(_inr_formatter())
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.set_title(self.title, color=THEME["text"], fontsize=14, pad=14,
                     fontweight="bold")
        ax.set_ylabel("Portfolio Value", color=THEME["muted"])
        ax.legend(facecolor=THEME["panel"], edgecolor=THEME["border"],
                  labelcolor=THEME["text"], fontsize=9, loc="upper left")

        self._annotate_summary(ax, series)
        plt.tight_layout()

        if save_chart:
            self._save(fig, filename or "equity_curve_single.png")
        return fig, ax

    # ── Strategy vs benchmark ────────────────────────────────────────────────
    def plot_with_benchmark(
        self,
        strategy_series: pd.Series,
        benchmark_series: pd.Series,
        strategy_label: str  = "Strategy",
        benchmark_label: str = "Benchmark (Buy & Hold)",
        normalise: bool      = True,
        save_chart: bool     = False,
        filename: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Overlay a strategy equity curve against a benchmark.

        Parameters
        ----------
        strategy_series:
            Strategy portfolio value series.
        benchmark_series:
            Benchmark value series (e.g. buy-and-hold equity).
        strategy_label / benchmark_label:
            Legend labels.
        normalise:
            If ``True`` (default), rescale both curves to "growth of ₹1"
            so they're visually comparable regardless of starting capital.
        save_chart:
            Save PNG.
        filename:
            Override default filename.

        Returns
        -------
        (Figure, Axes)
        """
        aligned = align_series(
            {strategy_label: strategy_series, benchmark_label: benchmark_series},
            method="inner",
        )
        if aligned.empty:
            raise ValueError(
                "No overlapping dates between strategy and benchmark series."
            )

        if normalise:
            aligned = aligned.apply(lambda c: normalise_to_growth(c, base=1.0))

        fig, ax = plt.subplots(figsize=self.figsize, facecolor=THEME["bg"])
        _style_axes(ax)

        ax.plot(aligned.index, aligned[strategy_label], color=THEME["blue"],
                linewidth=1.8, label=strategy_label, zorder=3)
        ax.plot(aligned.index, aligned[benchmark_label], color=THEME["muted"],
                linewidth=1.4, linestyle="--", label=benchmark_label, zorder=2)

        # Shade the area where strategy outperforms
        ax.fill_between(
            aligned.index, aligned[strategy_label], aligned[benchmark_label],
            where=(aligned[strategy_label] >= aligned[benchmark_label]),
            color=THEME["green"], alpha=0.12, interpolate=True,
        )
        ax.fill_between(
            aligned.index, aligned[strategy_label], aligned[benchmark_label],
            where=(aligned[strategy_label] < aligned[benchmark_label]),
            color=THEME["red"], alpha=0.12, interpolate=True,
        )

        if normalise:
            ax.set_ylabel("Growth of ₹1", color=THEME["muted"])
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:.2f}"))
        else:
            ax.yaxis.set_major_formatter(_inr_formatter())
            ax.set_ylabel("Portfolio Value", color=THEME["muted"])

        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.set_title(f"{strategy_label} vs {benchmark_label}",
                     color=THEME["text"], fontsize=14, pad=14, fontweight="bold")
        ax.legend(facecolor=THEME["panel"], edgecolor=THEME["border"],
                  labelcolor=THEME["text"], fontsize=9, loc="upper left")

        # Final outperformance annotation
        final_strat = aligned[strategy_label].iloc[-1]
        final_bench = aligned[benchmark_label].iloc[-1]
        outperf = (final_strat / final_bench - 1) * 100
        colour  = THEME["green"] if outperf >= 0 else THEME["red"]
        ax.annotate(
            f"{'+' if outperf >= 0 else ''}{outperf:.1f}% vs benchmark",
            xy=(aligned.index[-1], final_strat),
            xytext=(-10, 10), textcoords="offset points",
            color=colour, fontsize=10, fontweight="bold", ha="right",
        )

        plt.tight_layout()
        if save_chart:
            self._save(fig, filename or "equity_curve_vs_benchmark.png")
        return fig, ax

    # ── Multi-strategy comparison ────────────────────────────────────────────
    def plot_comparison(
        self,
        series_dict: Dict[str, pd.Series],
        normalise: bool   = True,
        save_chart: bool  = False,
        filename: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot multiple equity curves on one chart for side-by-side comparison.

        Parameters
        ----------
        series_dict:
            ``{label: equity_series}`` — any number of strategies.
        normalise:
            Rescale all curves to "growth of ₹1" (recommended when
            strategies started with different capital).
        save_chart:
            Save PNG.
        filename:
            Override default filename.

        Returns
        -------
        (Figure, Axes)
        """
        aligned = align_series(series_dict, method="inner")
        if aligned.empty:
            raise ValueError("No overlapping dates across all series.")

        if normalise:
            aligned = aligned.apply(lambda c: normalise_to_growth(c, base=1.0))

        fig, ax = plt.subplots(figsize=self.figsize, facecolor=THEME["bg"])
        _style_axes(ax)

        for i, label in enumerate(aligned.columns):
            colour = _PALETTE[i % len(_PALETTE)]
            ax.plot(aligned.index, aligned[label], color=colour,
                    linewidth=1.6, label=label)

        if normalise:
            ax.set_ylabel("Growth of ₹1", color=THEME["muted"])
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:.2f}"))
        else:
            ax.yaxis.set_major_formatter(_inr_formatter())
            ax.set_ylabel("Portfolio Value", color=THEME["muted"])

        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.set_title(self.title, color=THEME["text"], fontsize=14, pad=14,
                     fontweight="bold")
        ax.legend(facecolor=THEME["panel"], edgecolor=THEME["border"],
                  labelcolor=THEME["text"], fontsize=9, loc="upper left",
                  ncol=min(len(aligned.columns), 4))

        plt.tight_layout()
        if save_chart:
            self._save(fig, filename or "equity_curve_comparison.png")
        return fig, ax

    # ── Trade marker overlay ─────────────────────────────────────────────────
    def _overlay_trade_markers(
        self,
        ax: plt.Axes,
        series: pd.Series,
        trade_markers: pd.DataFrame,
    ) -> None:
        """
        Add ▲ (entry) / ▼ (exit) markers to an equity-curve axis.

        Accepts either a flat ``['date','action','price']`` log (e.g. from
        ``rebalance.trade_log``) or a round-trip ``['entry_date','exit_date']``
        log (e.g. from ``trade_analysis``).  Markers are plotted at the
        portfolio value on that date, not the trade price, so they sit on
        the equity curve itself.
        """
        df = trade_markers.copy()

        if {"entry_date", "exit_date"}.issubset(df.columns):
            entries = pd.to_datetime(df["entry_date"])
            exits   = pd.to_datetime(df["exit_date"])
        elif {"date", "action"}.issubset(df.columns):
            df["date"] = pd.to_datetime(df["date"])
            entries = df.loc[df["action"].str.upper() == "BUY", "date"]
            exits   = df.loc[df["action"].str.upper() == "SELL", "date"]
        else:
            logger.warning(
                "trade_markers must contain ['entry_date','exit_date'] or "
                "['date','action'] — skipping marker overlay."
            )
            return

        def _values_at(dates: pd.Series) -> Tuple[List, List]:
            valid_dates, valid_vals = [], []
            for d in dates:
                idx = series.index.searchsorted(d)
                if idx < len(series):
                    valid_dates.append(series.index[idx])
                    valid_vals.append(series.iloc[idx])
            return valid_dates, valid_vals

        ed, ev = _values_at(entries)
        xd, xv = _values_at(exits)

        if ed:
            ax.scatter(ed, ev, marker="^", color=THEME["green"], s=45,
                       zorder=5, label=f"Entry ({len(ed)})", alpha=0.85)
        if xd:
            ax.scatter(xd, xv, marker="v", color=THEME["red"], s=45,
                       zorder=5, label=f"Exit ({len(xd)})", alpha=0.85)

    # ── Summary annotation ────────────────────────────────────────────────────
    def _annotate_summary(self, ax: plt.Axes, series: pd.Series) -> None:
        """Add a small textbox with total return and CAGR."""
        start, end = series.iloc[0], series.iloc[-1]
        total_ret  = (end / start - 1) * 100
        years      = len(series) / 252
        cagr       = ((end / start) ** (1 / years) - 1) * 100 if years > 0 else 0.0
        colour     = THEME["green"] if total_ret >= 0 else THEME["red"]

        text = f"Return: {total_ret:+.1f}%   |   CAGR: {cagr:+.1f}%"
        ax.text(
            0.99, 0.04, text, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9.5, color=colour,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=THEME["panel"],
                      edgecolor=THEME["border"], alpha=0.9),
        )

    # ── Interactive (Plotly) ─────────────────────────────────────────────────
    def plot_interactive(
        self,
        series_dict: Dict[str, pd.Series],
        normalise: bool  = True,
        save_html: bool  = False,
        filename: Optional[str] = None,
    ):
        """
        Interactive Plotly equity-curve chart with hover tooltips.

        Parameters
        ----------
        series_dict:
            ``{label: equity_series}`` — one or more curves.
        normalise:
            Rescale to "growth of ₹1".
        save_html:
            Save a standalone HTML file to ``reports/charts/``.
        filename:
            Override default filename.

        Returns
        -------
        plotly.graph_objects.Figure

        Raises
        ------
        ImportError
            If plotly is not installed.
        """
        if not _PLOTLY_AVAILABLE:
            raise ImportError(
                "plotly is not installed. Run: pip install plotly --break-system-packages"
            )

        aligned = align_series(series_dict, method="inner")
        if normalise:
            aligned = aligned.apply(lambda c: normalise_to_growth(c, base=1.0))

        fig = go.Figure()
        for i, label in enumerate(aligned.columns):
            colour = _PALETTE[i % len(_PALETTE)]
            fig.add_trace(go.Scatter(
                x=aligned.index, y=aligned[label],
                mode="lines", name=label,
                line=dict(color=colour, width=2),
                hovertemplate=f"<b>{label}</b><br>%{{x|%Y-%m-%d}}<br>"
                              f"Value: ₹%{{y:,.2f}}<extra></extra>",
            ))

        fig.update_layout(
            title=dict(text=self.title, font=dict(color=THEME["text"], size=18)),
            plot_bgcolor=THEME["bg"], paper_bgcolor=THEME["bg"],
            font=dict(color=THEME["text"]),
            xaxis=dict(gridcolor=THEME["grid"], title="Date"),
            yaxis=dict(
                gridcolor=THEME["grid"],
                title="Growth of ₹1" if normalise else "Portfolio Value",
            ),
            legend=dict(bgcolor=THEME["panel"], bordercolor=THEME["border"]),
            hovermode="x unified",
            height=550,
        )

        if save_html:
            os.makedirs(CHARTS_DIR, exist_ok=True)
            fname = filename or "equity_curve_interactive.html"
            path  = os.path.join(CHARTS_DIR, fname)
            fig.write_html(path)
            logger.info("Interactive chart saved → %s", path)

        return fig

    # ── Save helper ───────────────────────────────────────────────────────────
    def _save(self, fig: plt.Figure, filename: str) -> str:
        """Save *fig* as PNG to ``reports/charts/`` and return the path."""
        os.makedirs(CHARTS_DIR, exist_ok=True)
        path = os.path.join(CHARTS_DIR, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        logger.info("Equity curve chart saved → %s", path)
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

    tcs_df  = pd.read_csv("data/processed/TCS_processed.csv", parse_dates=["date"])
    tcs_df  = tcs_df.set_index("date").sort_index()
    bah_tcs = (tcs_df["close"] / tcs_df["close"].iloc[0]) * CAPITAL
    bah_tcs.name = "Buy & Hold TCS"

    _lg.disable(_lg.NOTSET)
    print("=" * 60)
    print("  Equity Curve Visualization — Demo")
    print("=" * 60)

    # 1. Single curve with trade markers
    plotter = EquityCurvePlotter(title="TCS-Heavy Portfolio — Equity Curve")
    fig1, _ = plotter.plot_single(
        r_th.portfolio_value_series, label="TCS-Heavy",
        trade_markers=r_th.trade_log, save_chart=True,
        filename="equity_single_tcs_heavy.png",
    )
    print("\n  ✓  Saved: equity_single_tcs_heavy.png")

    # 2. Strategy vs benchmark
    plotter2 = EquityCurvePlotter(title="Strategy vs Buy-and-Hold")
    fig2, _ = plotter2.plot_with_benchmark(
        r_th.portfolio_value_series, bah_tcs,
        strategy_label="TCS-Heavy Portfolio",
        benchmark_label="Buy & Hold TCS",
        save_chart=True, filename="equity_vs_benchmark.png",
    )
    print("  ✓  Saved: equity_vs_benchmark.png")

    # 3. Multi-strategy comparison
    plotter3 = EquityCurvePlotter(title="Portfolio Strategy Comparison (Growth of ₹1)")
    fig3, _ = plotter3.plot_comparison(
        {
            "Equal Weight":  r_ew.portfolio_value_series,
            "TCS-Heavy":     r_th.portfolio_value_series,
            "Risk Parity":   r_rp.portfolio_value_series,
            "Buy&Hold TCS":  bah_tcs,
        },
        save_chart=True, filename="equity_comparison_all.png",
    )
    print("  ✓  Saved: equity_comparison_all.png")

    # 4. Interactive Plotly version
    plotter4 = EquityCurvePlotter(title="Interactive Portfolio Comparison")
    fig4 = plotter4.plot_interactive(
        {
            "Equal Weight": r_ew.portfolio_value_series,
            "TCS-Heavy":    r_th.portfolio_value_series,
            "Risk Parity":  r_rp.portfolio_value_series,
        },
        save_html=True, filename="equity_interactive.html",
    )
    print("  ✓  Saved: equity_interactive.html")

    print(f"\n  All charts saved to {CHARTS_DIR}/")
    print()
