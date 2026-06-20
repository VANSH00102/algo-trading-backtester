"""
trade_analysis.py
=================
Trade-level analytics engine for the Algorithmic Trading Strategy
Backtester — Phase 6.

What this module analyses
--------------------------
Given a trade log (one row per closed round-trip), it produces:

**Win/loss breakdown**
* Total trades, wins, losses, breakeven
* Win rate, loss rate
* Average win, average loss, best trade, worst trade
* Profit factor, expectancy per trade

**Streak analysis**
* Maximum consecutive wins / losses
* Current streak at end of series
* Streak histogram

**Holding period analysis**
* Average, median, min, max holding duration
* Distribution of holding periods (short/medium/long)
* Win rate by holding-period bucket

**Return distribution**
* Mean, std, skewness, kurtosis of trade returns (%)
* Percentile breakdown
* Distribution of positive vs negative return sizes

**Temporal breakdown**
* Monthly P&L aggregation (₹ and %)
* Quarterly P&L summary
* Year-by-year table
* Best and worst months

**By-stock breakdown**
* All key metrics computed per ticker

**Exit reason analysis**
* If a ``trigger`` column is present, group stats by exit reason

Architecture
------------
* :class:`TradeStats` — frozen dataclass, one per analysis scope.
* :class:`TradeAnalyzer` — facade driving all analyses.
* Pure helper functions (``_streak``, ``_profit_factor``, etc.).

Integration
-----------
Consumes trade logs from:
* Phase 3 Backtrader strategies (via ``notify_trade``)
* Phase 5 ``RebalanceSimResult.trade_log``
* CSV files produced by ``fetch_data.py`` or any external source

Expected column names (flexible — aliases accepted):

    entry_date / date_entry / open_date
    exit_date  / date_exit  / close_date
    pnl        / net_pnl    / profit_loss
    pnl_pct    / return_pct / pct_return
    stock      / ticker     / symbol
    duration_days / holding_days
    trigger    / exit_reason  (optional)

Usage
-----
::

    python src/analytics/trade_analysis.py
    from src.analytics.trade_analysis import TradeAnalyzer
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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

#: Column name aliases — first match wins.
_COL_ALIASES: Dict[str, List[str]] = {
    "entry_date":    ["entry_date", "date_entry", "open_date", "date"],
    "exit_date":     ["exit_date", "date_exit", "close_date"],
    "pnl":           ["pnl", "net_pnl", "profit_loss", "pnl_net"],
    "pnl_pct":       ["pnl_pct", "return_pct", "pct_return", "trade_return"],
    "stock":         ["stock", "ticker", "symbol"],
    "duration_days": ["duration_days", "holding_days", "bars_held"],
    "trigger":       ["trigger", "exit_reason", "signal"],
    "shares":        ["shares", "qty", "quantity", "size"],
    "entry_price":   ["entry_price", "buy_price", "open_price"],
    "exit_price":    ["exit_price", "sell_price", "close_price"],
}


# ──────────────────────────────────────────────────────────────────────────────
# Value object
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class TradeStats:
    """
    Complete trade statistics for one analysis scope (all trades or per stock).

    All monetary fields are in INR.

    Attributes
    ----------
    label:                 Scope label (e.g. ``'All'`` or ``'TCS'``).
    total_trades:          Total closed round-trip trades.
    winning_trades:        Trades with PnL > 0.
    losing_trades:         Trades with PnL < 0.
    breakeven_trades:      Trades with PnL == 0.
    win_rate_pct:          winning / total × 100.
    gross_profit_inr:      Sum of all positive PnL.
    gross_loss_inr:        Sum of all negative PnL (positive value).
    net_pnl_inr:           gross_profit − gross_loss.
    profit_factor:         gross_profit / gross_loss.
    avg_win_inr:           Mean PnL of winning trades.
    avg_loss_inr:          Mean PnL of losing trades (negative).
    best_trade_inr:        Largest single-trade profit.
    worst_trade_inr:       Largest single-trade loss.
    avg_trade_pct:         Mean trade return (%).
    expectancy_inr:        Expected PnL per trade = mean(all PnL).
    avg_holding_days:      Mean trade duration.
    median_holding_days:   Median trade duration.
    min_holding_days:      Shortest trade.
    max_holding_days:      Longest trade.
    max_consec_wins:       Longest winning streak.
    max_consec_losses:     Longest losing streak.
    pnl_skewness:          Skewness of trade PnL distribution.
    pnl_kurtosis:          Excess kurtosis of PnL distribution.
    """
    label:               str
    total_trades:        int
    winning_trades:      int
    losing_trades:       int
    breakeven_trades:    int
    win_rate_pct:        float
    gross_profit_inr:    float
    gross_loss_inr:      float
    net_pnl_inr:         float
    profit_factor:       float
    avg_win_inr:         float
    avg_loss_inr:        float
    best_trade_inr:      float
    worst_trade_inr:     float
    avg_trade_pct:       float
    expectancy_inr:      float
    avg_holding_days:    float
    median_holding_days: float
    min_holding_days:    int
    max_holding_days:    int
    max_consec_wins:     int
    max_consec_losses:   int
    pnl_skewness:        float
    pnl_kurtosis:        float

    def to_dict(self) -> Dict:
        return {f: getattr(self, f) for f in self.__dataclass_fields__}  # type: ignore[attr-defined]

    def to_series(self) -> pd.Series:
        return pd.Series(self.to_dict(), name=self.label)

    def report(self) -> str:
        """Multi-line formatted summary."""
        W = 52
        hd = lambda t: f"  ║  {t:<{W-4}}║"
        hr = f"  ╠{'═'*W}╣"
        lines = [
            f"  ╔{'═'*W}╗",
            hd(f"Trade Analysis: {self.label}"),
            hr,
            hd("VOLUME"),
            hd(f"  Total trades       : {self.total_trades:>8}"),
            hd(f"  Winning            : {self.winning_trades:>8}  ({self.win_rate_pct:.1f}%)"),
            hd(f"  Losing             : {self.losing_trades:>8}  ({100-self.win_rate_pct:.1f}%)"),
            hd(f"  Breakeven          : {self.breakeven_trades:>8}"),
            hr,
            hd("PROFITABILITY"),
            hd(f"  Net P&L            : ₹{self.net_pnl_inr:>+12,.0f}"),
            hd(f"  Gross profit       : ₹{self.gross_profit_inr:>12,.0f}"),
            hd(f"  Gross loss         : ₹{self.gross_loss_inr:>12,.0f}"),
            hd(f"  Profit factor      : {self.profit_factor:>12.2f}×"),
            hd(f"  Expectancy / trade : ₹{self.expectancy_inr:>+12,.0f}"),
            hd(f"  Avg trade return   : {self.avg_trade_pct:>+11.2f}%"),
            hr,
            hd("WIN / LOSS STATS"),
            hd(f"  Avg win            : ₹{self.avg_win_inr:>+12,.0f}"),
            hd(f"  Avg loss           : ₹{self.avg_loss_inr:>+12,.0f}"),
            hd(f"  Best trade         : ₹{self.best_trade_inr:>+12,.0f}"),
            hd(f"  Worst trade        : ₹{self.worst_trade_inr:>+12,.0f}"),
            hr,
            hd("HOLDING PERIODS"),
            hd(f"  Avg holding (days) : {self.avg_holding_days:>12.1f}"),
            hd(f"  Median holding     : {self.median_holding_days:>12.1f}"),
            hd(f"  Shortest trade     : {self.min_holding_days:>12} days"),
            hd(f"  Longest trade      : {self.max_holding_days:>12} days"),
            hr,
            hd("STREAKS & DISTRIBUTION"),
            hd(f"  Max consec. wins   : {self.max_consec_wins:>12}"),
            hd(f"  Max consec. losses : {self.max_consec_losses:>12}"),
            hd(f"  PnL skewness       : {self.pnl_skewness:>+12.3f}"),
            hd(f"  PnL kurtosis       : {self.pnl_kurtosis:>+12.3f}"),
            f"  ╚{'═'*W}╝",
        ]
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Column normalisation
# ──────────────────────────────────────────────────────────────────────────────
def _resolve_col(df: pd.DataFrame, canonical: str) -> Optional[str]:
    """Return the first matching column alias or ``None``."""
    for alias in _COL_ALIASES.get(canonical, [canonical]):
        if alias in df.columns:
            return alias
    return None


def normalise_trade_log(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise column names and data types in a trade log.

    Renames columns to canonical names, parses dates, converts
    numeric fields, and computes ``pnl_pct`` if missing.

    Parameters
    ----------
    df:
        Raw trade log (from CSV, Backtrader, or Phase 5).

    Returns
    -------
    pd.DataFrame
        Cleaned copy with canonical column names.

    Raises
    ------
    ValueError
        If a required column (``pnl``) cannot be found.
    """
    df = df.copy()
    rename_map: Dict[str, str] = {}
    for canonical, aliases in _COL_ALIASES.items():
        for alias in aliases:
            if alias in df.columns and alias != canonical:
                rename_map[alias] = canonical
                break
    df = df.rename(columns=rename_map)

    # Require PnL
    if "pnl" not in df.columns:
        raise ValueError(
            "Trade log must contain a PnL column. "
            f"Expected one of: {_COL_ALIASES['pnl']}. "
            f"Found: {df.columns.tolist()}"
        )

    # Parse dates
    for date_col in ("entry_date", "exit_date"):
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # Compute pnl_pct if missing but entry/exit prices are present
    if "pnl_pct" not in df.columns:
        if "entry_price" in df.columns and "exit_price" in df.columns:
            df["pnl_pct"] = (
                (df["exit_price"] - df["entry_price"]) / df["entry_price"] * 100
            ).round(4)
        else:
            df["pnl_pct"] = np.nan

    # Compute duration if missing
    if "duration_days" not in df.columns:
        if "entry_date" in df.columns and "exit_date" in df.columns:
            df["duration_days"] = (
                df["exit_date"] - df["entry_date"]
            ).dt.days
        else:
            df["duration_days"] = np.nan

    # Numeric coercion
    for col in ("pnl", "pnl_pct", "shares", "entry_price", "exit_price"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info(
        "Trade log normalised: %d trades  cols=%s",
        len(df), df.columns.tolist(),
    )
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Pure helpers
# ──────────────────────────────────────────────────────────────────────────────
def _profit_factor(pnl: pd.Series) -> float:
    wins   = pnl[pnl > 0].sum()
    losses = pnl[pnl < 0].sum()
    return float(wins / abs(losses)) if losses != 0 else float("inf")


def _streak(pnl: pd.Series) -> Tuple[int, int]:
    """Return (max_consecutive_wins, max_consecutive_losses)."""
    max_w = max_l = cur_w = cur_l = 0
    for v in pnl:
        if v > 0:
            cur_w += 1; cur_l = 0
        elif v < 0:
            cur_l += 1; cur_w = 0
        else:
            cur_w = cur_l = 0
        max_w = max(max_w, cur_w)
        max_l = max(max_l, cur_l)
    return max_w, max_l


def _compute_stats(df: pd.DataFrame, label: str = "All") -> TradeStats:
    """Core computation from a normalised trade-log slice."""
    pnl      = df["pnl"].dropna()
    pnl_pct  = df["pnl_pct"].dropna() if "pnl_pct" in df.columns else pd.Series(dtype=float)
    dur      = df["duration_days"].dropna() if "duration_days" in df.columns else pd.Series(dtype=float)

    wins = pnl[pnl > 0]
    loss = pnl[pnl < 0]
    be   = pnl[pnl == 0]
    n    = len(pnl)
    mw, ml = _streak(pnl)

    return TradeStats(
        label              = label,
        total_trades       = n,
        winning_trades     = len(wins),
        losing_trades      = len(loss),
        breakeven_trades   = len(be),
        win_rate_pct       = round(len(wins) / n * 100, 2) if n else 0.0,
        gross_profit_inr   = round(float(wins.sum()), 2),
        gross_loss_inr     = round(float(abs(loss.sum())), 2),
        net_pnl_inr        = round(float(pnl.sum()), 2),
        profit_factor      = round(_profit_factor(pnl), 3),
        avg_win_inr        = round(float(wins.mean()), 2) if len(wins) else 0.0,
        avg_loss_inr       = round(float(loss.mean()), 2) if len(loss) else 0.0,
        best_trade_inr     = round(float(pnl.max()), 2) if n else 0.0,
        worst_trade_inr    = round(float(pnl.min()), 2) if n else 0.0,
        avg_trade_pct      = round(float(pnl_pct.mean()), 4) if len(pnl_pct) else 0.0,
        expectancy_inr     = round(float(pnl.mean()), 2) if n else 0.0,
        avg_holding_days   = round(float(dur.mean()), 1) if len(dur) else 0.0,
        median_holding_days= round(float(dur.median()), 1) if len(dur) else 0.0,
        min_holding_days   = int(dur.min()) if len(dur) else 0,
        max_holding_days   = int(dur.max()) if len(dur) else 0,
        max_consec_wins    = mw,
        max_consec_losses  = ml,
        pnl_skewness       = round(float(pnl.skew()), 3) if n > 2 else 0.0,
        pnl_kurtosis       = round(float(pnl.kurtosis()), 3) if n > 3 else 0.0,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Facade
# ──────────────────────────────────────────────────────────────────────────────
class TradeAnalyzer:
    """
    Comprehensive trade-by-trade analytics from a trade log DataFrame.

    Parameters
    ----------
    trade_log:
        Raw or pre-normalised trade log.  Column names are resolved
        via :func:`normalise_trade_log`.

    Examples
    --------
    ::

        ta = TradeAnalyzer(trade_log=df)
        stats = ta.overall_stats()
        print(stats.report())
        ta.plot_dashboard(save_chart=True)
    """

    def __init__(self, trade_log: pd.DataFrame) -> None:
        self._raw   = trade_log
        self._df    = normalise_trade_log(trade_log)
        logger.info("TradeAnalyzer ready: %d trades", len(self._df))

    # ── Stats ─────────────────────────────────────────────────────────────────
    def overall_stats(self) -> TradeStats:
        """Overall stats across all stocks."""
        return _compute_stats(self._df, "All Trades")

    def by_stock(self) -> Dict[str, TradeStats]:
        """
        Per-stock stats.

        Returns
        -------
        dict
            ``{stock_name: TradeStats}``.
        """
        if "stock" not in self._df.columns:
            return {}
        return {
            stock: _compute_stats(grp, stock)
            for stock, grp in self._df.groupby("stock")
        }

    def by_trigger(self) -> Optional[Dict[str, TradeStats]]:
        """
        Stats broken down by exit trigger / signal.

        Returns ``None`` if no ``trigger`` column is present.
        """
        if "trigger" not in self._df.columns:
            return None
        return {
            trig: _compute_stats(grp, trig)
            for trig, grp in self._df.groupby("trigger")
        }

    # ── Temporal breakdowns ───────────────────────────────────────────────────
    def monthly_pnl(self) -> pd.DataFrame:
        """
        Monthly P&L aggregation.

        Returns
        -------
        pd.DataFrame
            Index = YearMonth, columns = ``[n_trades, total_pnl, avg_pnl, win_rate]``.
        """
        df = self._df.copy()
        date_col = "exit_date" if "exit_date" in df.columns else "entry_date"
        if date_col not in df.columns:
            logger.warning("No date column found — monthly P&L unavailable.")
            return pd.DataFrame()
        df["ym"] = df[date_col].dt.to_period("M")
        agg = df.groupby("ym").agg(
            n_trades  = ("pnl", "count"),
            total_pnl = ("pnl", "sum"),
            avg_pnl   = ("pnl", "mean"),
            win_rate  = ("pnl", lambda x: (x > 0).mean() * 100),
        ).round(2)
        return agg

    def annual_pnl(self) -> pd.DataFrame:
        """
        Year-by-year P&L summary.

        Returns
        -------
        pd.DataFrame
            Index = year, columns = ``[n_trades, total_pnl, win_rate, best, worst]``.
        """
        df = self._df.copy()
        date_col = "exit_date" if "exit_date" in df.columns else "entry_date"
        if date_col not in df.columns:
            return pd.DataFrame()
        df["year"] = df[date_col].dt.year
        agg = df.groupby("year").agg(
            n_trades  = ("pnl", "count"),
            total_pnl = ("pnl", "sum"),
            win_rate  = ("pnl", lambda x: (x > 0).mean() * 100),
            best      = ("pnl", "max"),
            worst     = ("pnl", "min"),
        ).round(2)
        return agg

    def holding_period_buckets(self) -> pd.DataFrame:
        """
        Win rate and average P&L by holding-period bucket.

        Buckets: Short (1–7 d), Medium (8–30 d), Long (31–90 d), Very Long (90+ d).

        Returns
        -------
        pd.DataFrame
        """
        df = self._df.dropna(subset=["duration_days"]).copy()
        bins   = [0, 7, 30, 90, float("inf")]
        labels = ["Short (1-7d)", "Medium (8-30d)", "Long (31-90d)", "VeryLong (90+d)"]
        df["bucket"] = pd.cut(df["duration_days"], bins=bins, labels=labels)
        agg = df.groupby("bucket", observed=True).agg(
            n_trades  = ("pnl", "count"),
            win_rate  = ("pnl", lambda x: (x > 0).mean() * 100),
            avg_pnl   = ("pnl", "mean"),
            total_pnl = ("pnl", "sum"),
        ).round(2)
        return agg

    def return_distribution(self) -> Dict[str, float]:
        """
        Descriptive statistics of trade return (% per trade).

        Returns
        -------
        dict
            Keys: mean, std, median, min, max, p25, p75, skew, kurt.
        """
        pnl_pct = self._df["pnl_pct"].dropna()
        if pnl_pct.empty:
            return {}
        return {
            "mean":   round(float(pnl_pct.mean()),   4),
            "std":    round(float(pnl_pct.std()),    4),
            "median": round(float(pnl_pct.median()), 4),
            "min":    round(float(pnl_pct.min()),    4),
            "max":    round(float(pnl_pct.max()),    4),
            "p25":    round(float(pnl_pct.quantile(0.25)), 4),
            "p75":    round(float(pnl_pct.quantile(0.75)), 4),
            "skew":   round(float(pnl_pct.skew()),   4),
            "kurt":   round(float(pnl_pct.kurtosis()), 4),
        }

    def streak_detail(self) -> pd.DataFrame:
        """
        Full streak table: every consecutive run of wins/losses.

        Returns
        -------
        pd.DataFrame
            Columns: ``streak_type``, ``length``, ``total_pnl``.
        """
        pnl = self._df["pnl"].dropna().reset_index(drop=True)
        rows   = []
        cur_type: Optional[str] = None
        cur_len  = 0
        cur_pnl  = 0.0
        for v in pnl:
            t = "WIN" if v > 0 else ("LOSS" if v < 0 else "BE")
            if t == cur_type:
                cur_len += 1
                cur_pnl += v
            else:
                if cur_type is not None:
                    rows.append({"type": cur_type, "length": cur_len,
                                 "total_pnl": round(cur_pnl, 2)})
                cur_type = t
                cur_len  = 1
                cur_pnl  = float(v)
        if cur_type is not None:
            rows.append({"type": cur_type, "length": cur_len,
                         "total_pnl": round(cur_pnl, 2)})
        df = pd.DataFrame(rows)
        return df.sort_values("length", ascending=False).reset_index(drop=True)

    def comparison_table(self) -> pd.DataFrame:
        """
        Side-by-side comparison of overall + per-stock stats.

        Returns
        -------
        pd.DataFrame
            Rows = metrics, columns = All + each stock.
        """
        cols: Dict[str, pd.Series] = {
            "All": self.overall_stats().to_series()
        }
        for stock, stats in self.by_stock().items():
            cols[stock] = stats.to_series()
        return pd.DataFrame(cols)

    # ── Chart ─────────────────────────────────────────────────────────────────
    def plot_dashboard(
        self,
        save_chart: bool = False,
        filename:   Optional[str] = None,
    ) -> Optional[str]:
        """
        4-panel trade analytics dashboard.

        Panels
        ------
        1. Cumulative P&L curve
        2. P&L per trade (bar chart, coloured by win/loss)
        3. Holding period histogram
        4. Trade return (%) distribution

        Parameters
        ----------
        save_chart:
            Save PNG to ``reports/charts/``.
        filename:
            Override default filename.

        Returns
        -------
        str or None
            Saved path or ``None``.
        """
        df = self._df.copy()
        pnl = df["pnl"].dropna().reset_index(drop=True)

        fig = plt.figure(figsize=(16, 10), facecolor="#0d1117")
        gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)
        axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
                fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]

        _DARK = "#0d1117"
        _BLUE = "#58a6ff"
        _GREEN = "#3fb950"
        _RED   = "#f85149"
        _GREY  = "#8b949e"

        for ax in axes:
            ax.set_facecolor(_DARK)
            ax.tick_params(colors="white", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#333")

        # ── Panel 1: Cumulative P&L ───────────────────────────────────────────
        ax = axes[0]
        cum_pnl = pnl.cumsum()
        color   = _GREEN if float(cum_pnl.iloc[-1]) >= 0 else _RED
        ax.plot(cum_pnl.values, color=color, linewidth=1.8)
        ax.fill_between(range(len(cum_pnl)), cum_pnl.values, 0,
                        where=(cum_pnl.values >= 0),
                        alpha=0.15, color=_GREEN)
        ax.fill_between(range(len(cum_pnl)), cum_pnl.values, 0,
                        where=(cum_pnl.values < 0),
                        alpha=0.15, color=_RED)
        ax.axhline(0, color=_GREY, linewidth=0.8, linestyle="--")
        ax.set_title("Cumulative P&L (₹)", color="white", fontsize=10)
        ax.set_xlabel("Trade #", color=_GREY, fontsize=8)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"₹{x/1000:.0f}K")
        )

        # ── Panel 2: Per-trade P&L bar ────────────────────────────────────────
        ax = axes[1]
        colors = [_GREEN if v > 0 else _RED for v in pnl.values]
        ax.bar(range(len(pnl)), pnl.values, color=colors, alpha=0.8, width=0.8)
        ax.axhline(0, color=_GREY, linewidth=0.8, linestyle="--")
        ax.set_title("P&L per Trade (₹)", color="white", fontsize=10)
        ax.set_xlabel("Trade #", color=_GREY, fontsize=8)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"₹{x/1000:.0f}K")
        )

        # ── Panel 3: Holding period histogram ────────────────────────────────
        ax = axes[2]
        if "duration_days" in df.columns:
            dur = df["duration_days"].dropna()
            ax.hist(dur.values, bins=20, color=_BLUE, alpha=0.75, edgecolor="#333")
            ax.axvline(float(dur.mean()), color=_RED, linewidth=1.5,
                       linestyle="--", label=f"Mean={dur.mean():.0f}d")
            ax.axvline(float(dur.median()), color=_GREEN, linewidth=1.5,
                       linestyle="--", label=f"Median={dur.median():.0f}d")
            ax.legend(facecolor="#161b22", edgecolor="#333",
                      labelcolor="white", fontsize=8)
        ax.set_title("Holding Period (days)", color="white", fontsize=10)
        ax.set_xlabel("Days", color=_GREY, fontsize=8)
        ax.set_ylabel("# Trades", color=_GREY, fontsize=8)

        # ── Panel 4: Return distribution ──────────────────────────────────────
        ax = axes[3]
        if "pnl_pct" in df.columns:
            rets = df["pnl_pct"].dropna()
            wins_r = rets[rets >= 0].values
            loss_r = rets[rets < 0].values
            if len(wins_r):
                ax.hist(wins_r, bins=15, color=_GREEN, alpha=0.65,
                        edgecolor="#333", label="Wins")
            if len(loss_r):
                ax.hist(loss_r, bins=15, color=_RED, alpha=0.65,
                        edgecolor="#333", label="Losses")
            ax.axvline(0, color=_GREY, linewidth=0.8, linestyle="--")
            ax.legend(facecolor="#161b22", edgecolor="#333",
                      labelcolor="white", fontsize=8)
        ax.set_title("Trade Return Distribution (%)", color="white", fontsize=10)
        ax.set_xlabel("Return %", color=_GREY, fontsize=8)
        ax.set_ylabel("# Trades", color=_GREY, fontsize=8)

        # Overall title
        total = self.overall_stats()
        fig.suptitle(
            f"Trade Analytics Dashboard  |  {total.total_trades} trades  |  "
            f"WR={total.win_rate_pct:.1f}%  |  PF={total.profit_factor:.2f}×  |  "
            f"Net ₹{total.net_pnl_inr:>+,.0f}",
            color="white", fontsize=11, y=1.01,
        )

        if save_chart:
            os.makedirs(CHARTS_DIR, exist_ok=True)
            fname = filename or "trade_dashboard.png"
            path  = os.path.join(CHARTS_DIR, fname)
            plt.savefig(path, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            plt.close()
            logger.info("Trade dashboard saved → %s", path)
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

    print("=" * 60)
    print("  Trade Analysis — Demo")
    print("=" * 60)

    # Load synthetic trade log generated by the integration test
    log_path = os.path.join("data", "processed", "synthetic_trade_log.csv")
    if not os.path.exists(log_path):
        print(f"  Trade log not found: {log_path}")
        print("  Run the Phase 6 integration script first.")
        sys.exit(1)

    raw = pd.read_csv(log_path, parse_dates=["entry_date", "exit_date"])
    ta  = TradeAnalyzer(trade_log=raw)

    # ── Overall report ────────────────────────────────────────────────────────
    stats = ta.overall_stats()
    print(f"\n{stats.report()}")

    # ── By stock ──────────────────────────────────────────────────────────────
    print("\n── Per-Stock Summary ───────────────────────────────────────")
    comp = ta.comparison_table()
    display = [
        "total_trades", "win_rate_pct", "net_pnl_inr",
        "profit_factor", "expectancy_inr",
        "avg_win_inr", "avg_loss_inr",
        "avg_holding_days", "max_consec_wins", "max_consec_losses",
    ]
    print(comp.loc[display].to_string(float_format=lambda x: f"{x:>+.2f}"))

    # ── By trigger ────────────────────────────────────────────────────────────
    trig = ta.by_trigger()
    if trig:
        print("\n── By Exit Trigger ─────────────────────────────────────────")
        trig_rows = []
        for t, s in trig.items():
            trig_rows.append({
                "Trigger": t, "Trades": s.total_trades,
                "Win%": f"{s.win_rate_pct:.1f}%",
                "Net P&L ₹": f"{s.net_pnl_inr:>+,.0f}",
                "Profit Factor": f"{s.profit_factor:.2f}×",
                "Expectancy ₹": f"{s.expectancy_inr:>+,.0f}",
            })
        print(pd.DataFrame(trig_rows).to_string(index=False))

    # ── Annual P&L ────────────────────────────────────────────────────────────
    print("\n── Annual P&L ──────────────────────────────────────────────")
    print(ta.annual_pnl().to_string(float_format=lambda x: f"{x:>+.2f}"))

    # ── Holding period buckets ────────────────────────────────────────────────
    print("\n── Holding Period Buckets ──────────────────────────────────")
    print(ta.holding_period_buckets().to_string(float_format=lambda x: f"{x:>+.2f}"))

    # ── Return distribution ───────────────────────────────────────────────────
    print("\n── Return Distribution (%) ─────────────────────────────────")
    dist = ta.return_distribution()
    for k, v in dist.items():
        print(f"  {k:<10}: {v:>+.4f}%")

    # ── Streak detail ─────────────────────────────────────────────────────────
    print("\n── Top 8 Streaks ───────────────────────────────────────────")
    print(ta.streak_detail().head(8).to_string(index=False))

    # ── Dashboard chart ───────────────────────────────────────────────────────
    path = ta.plot_dashboard(save_chart=True, filename="trade_dashboard.png")
    if path:
        print(f"\n  Dashboard chart → {path}")
    print()
