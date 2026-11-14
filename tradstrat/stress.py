"""Stress tests: the questions to ask before trusting any backtest.

A single backtest is one path through one stretch of history, with one set of costs. Each test here
changes one of those and asks whether the result survives:

* :func:`start_date_test` runs the rule over many windows of the same length, each starting in a
  different month, so a lucky or unlucky start date cannot decide the result.
* :func:`without_best_trades` removes the best trades and keeps the money in cash on those days instead,
  to see whether a few big winners carried everything.
* :func:`crash_test` finds the asset's deepest falls in the history and compares the rule with holding
  through each one.
* :func:`cost_test` doubles commission and slippage for both the rule and holding.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Final

import numpy as np
import pandas as pd

from tradstrat.engine import BacktestConfig, BacktestResult, _extract_signal, _simulate, run_backtest
from tradstrat.metrics import DAYS_PER_YEAR, cagr
from tradstrat.strategies import BuyAndHoldStrategy, Strategy

BEAR_MARKET: Final[float] = 0.20
MAX_WINDOWS: Final[int] = 120

# Falls whose bottom lands inside one of these spans get its name, so a reader recognizes them.
KNOWN_CRASHES: Final[tuple[tuple[str, str, str], ...]] = (
    ("1987 crash", "1987-08-01", "1987-12-31"),
    ("Dot-com crash", "2000-03-01", "2003-03-31"),
    ("2008 financial crisis", "2007-10-01", "2009-06-30"),
    ("2018 sell-off", "2018-10-01", "2018-12-31"),
    ("Covid crash", "2020-02-15", "2020-06-30"),
    ("2022 bear market", "2022-01-01", "2022-12-31"),
)


def _annual(final: float, initial: float, days: float) -> float:
    years = days / DAYS_PER_YEAR
    if years <= 0:
        return float("nan")
    growth = final / initial
    return -1.0 if growth <= 0 else growth ** (1 / years) - 1


@dataclass(frozen=True)
class StartDateResult:
    """The rule and holding over windows of equal length, one starting in each sampled month."""

    windows: pd.DataFrame
    years: int

    @property
    def share_beating(self) -> float:
        """Share of start dates where the rule grew faster than holding."""
        return float(self.windows["Beat"].mean())

    @property
    def worst(self) -> pd.Series:
        """The start date that turned out worst for the rule."""
        return self.windows.iloc[int(np.argmin(self.windows["Rule"].to_numpy()))]

    @property
    def best(self) -> pd.Series:
        return self.windows.iloc[int(np.argmax(self.windows["Rule"].to_numpy()))]


def start_date_test(
    data: pd.DataFrame, strategy: Strategy, config: BacktestConfig | None = None, *, years: int | None = None
) -> StartDateResult | None:
    """Run the rule over many windows of ``years`` years, each starting in a different month.

    Indicators are computed once on the full history, so every window starts with them warmed up, as a
    trader starting that month would have had. Both sides pay the same costs. Without ``years`` the window
    is half the history, at most five years. Returns ``None`` when the history is under two years long.
    """
    config = config or BacktestConfig()
    if not isinstance(data.index, pd.DatetimeIndex) or len(data) < 3:
        return None
    total_years = (data.index[-1] - data.index[0]).days / DAYS_PER_YEAR
    if years is None:
        years = min(5, int(total_years // 2))
    if years < 1 or total_years < years + 0.5:
        return None

    first = max(strategy.warmup + 1, 1)
    last_start = data.index[-1] - pd.DateOffset(years=years)
    months = pd.Series(data.index[first:], index=data.index[first:])
    starts = months[months <= last_start].groupby(months.dt.to_period("M")).first()
    if starts.empty:
        return None
    if len(starts) > MAX_WINDOWS:
        starts = starts.iloc[np.linspace(0, len(starts) - 1, MAX_WINDOWS).round().astype(int)]

    signal = _extract_signal(data, strategy)
    holding = _extract_signal(data, BuyAndHoldStrategy())
    benchmark = config.for_benchmark()
    rows = []
    for start_date in starts:
        start = int(data.index.searchsorted(start_date))
        end = int(data.index.searchsorted(start_date + pd.DateOffset(years=years), side="right"))
        window = data.iloc[:end]
        rule = _simulate(window, signal[:end], config, start).equity
        hold = _simulate(window, holding[:end], benchmark, start).equity
        days = (window.index[-1] - window.index[start - 1]).days
        rows.append(
            {
                "Start": start_date,
                "End": window.index[-1],
                "Rule": _annual(rule[-1], rule[0], days),
                "Hold": _annual(hold[-1], hold[0], days),
            }
        )
    windows = pd.DataFrame(rows)
    windows["Beat"] = windows["Rule"] > windows["Hold"]
    return StartDateResult(windows=windows, years=years)


@dataclass(frozen=True)
class BestTradesResult:
    """The rule's growth with and without its best trades."""

    removed: int
    trades: int
    cagr: float
    cagr_without: float
    hold_cagr: float
    share_of_profit: float
    equity_without: pd.Series


def without_best_trades(result: BacktestResult, count: int = 5) -> BestTradesResult | None:
    """Take out the ``count`` best trades by return, as if the money had stayed in cash on those days.

    Every day from a trade's entry to its exit earns the cash rate instead of the trade's return, so the
    rest of the history is left exactly as it was. ``share_of_profit`` is the part of the rule's total
    profit that came from those trades, which can exceed 100% when the other trades lost money overall.
    Returns ``None`` when the rule never traded.
    """
    trades = result.trade_log
    if trades.empty:
        return None
    removed = min(count, len(trades))
    best = trades.nlargest(removed, "Return")

    returns = result.returns.to_numpy(copy=True)
    cash = result.risk_free_returns.to_numpy()
    dates = result.returns.index
    for entry, exit_ in zip(best["Entry Date"], best["Exit Date"], strict=True):
        days = (dates >= entry) & (dates <= exit_)
        returns[days] = cash[days]
    equity = result.results["Strategy Equity"]
    growth = np.concatenate([[1.0], np.cumprod(1 + returns)])
    rebuilt = pd.Series(equity.iloc[0] * growth, index=equity.index, name="Without best trades")

    total = float(trades["P&L"].sum())
    return BestTradesResult(
        removed=removed,
        trades=len(trades),
        cagr=result.metrics.cagr,
        cagr_without=cagr(rebuilt),
        hold_cagr=result.benchmark_metrics.cagr,
        share_of_profit=float(best["P&L"].sum()) / total if total > 0 else float("nan"),
        equity_without=rebuilt,
    )


@dataclass(frozen=True)
class Crash:
    """One deep fall of the asset, from its peak to its bottom, and how the rule did over it."""

    name: str
    peak: pd.Timestamp
    bottom: pd.Timestamp
    hold: float
    rule: float

    @property
    def rule_lost_less(self) -> bool:
        return self.rule > self.hold


def _deepest_falls(equity: pd.Series, depth: float, limit: int) -> list[tuple[int, int]]:
    """Peak and bottom positions of the deepest non-overlapping falls of at least ``depth``."""
    values = equity.to_numpy(dtype=float)
    free = np.ones(len(values), dtype=bool)
    falls: list[tuple[int, int]] = []
    while len(falls) < limit:
        running_peak = np.maximum.accumulate(np.where(free, values, -np.inf))
        drawdown = np.where(free & (running_peak > 0), values / running_peak - 1, 0.0)
        bottom = int(np.argmin(drawdown))
        if drawdown[bottom] > -depth:
            break
        peak = int(np.flatnonzero(free[: bottom + 1] & (values[: bottom + 1] == running_peak[bottom]))[-1])
        # Block the whole episode, from the peak until the price regains it, so the next fall is a new one.
        recovered = np.flatnonzero(values[bottom:] >= values[peak])
        end = bottom + int(recovered[0]) if recovered.size else len(values)
        free[peak:end] = False
        falls.append((peak, bottom))
    return sorted(falls)


def _crash_name(peak: object, bottom: object) -> str:
    if not isinstance(peak, pd.Timestamp) or not isinstance(bottom, pd.Timestamp):
        return f"Bars {peak} to {bottom}"
    for name, start, end in KNOWN_CRASHES:
        if pd.Timestamp(start) <= bottom <= pd.Timestamp(end):
            return name
    return f"{peak:%b %Y} to {bottom:%b %Y}"


def crash_test(result: BacktestResult, depth: float = BEAR_MARKET, limit: int = 4) -> list[Crash]:
    """The asset's deepest falls of at least ``depth`` (a bear market by default), oldest first.

    Each compares the rule's return with holding's from the asset's peak to its bottom. An empty list means
    the history holds no fall that deep, so it cannot show how the rule copes with a crash.
    """
    hold = result.results["Buy and Hold Benchmark"]
    rule = result.results["Strategy Equity"]
    crashes = []
    for peak, bottom in _deepest_falls(hold, depth, limit):
        crashes.append(
            Crash(
                name=_crash_name(hold.index[peak], hold.index[bottom]),
                peak=hold.index[peak],
                bottom=hold.index[bottom],
                hold=float(hold.iloc[bottom] / hold.iloc[peak] - 1),
                rule=float(rule.iloc[bottom] / rule.iloc[peak] - 1),
            )
        )
    return crashes


@dataclass(frozen=True)
class CostResult:
    """Growth per year for the rule and holding, at the chosen costs and at double them."""

    commission_rate: float
    slippage_bps: float
    cagr: float
    hold_cagr: float
    doubled_cagr: float
    doubled_hold_cagr: float

    @property
    def free(self) -> bool:
        """No costs were set, so doubling them changes nothing."""
        return self.commission_rate == 0 and self.slippage_bps == 0


def cost_test(
    data: pd.DataFrame, strategy: Strategy, result: BacktestResult, factor: float = 2.0
) -> CostResult:
    """Rerun the rule and holding with commission and slippage multiplied by ``factor``."""
    config = result.config
    doubled = run_backtest(
        data,
        strategy,
        replace(
            config,
            commission_rate=config.commission_rate * factor,
            slippage_bps=config.slippage_bps * factor,
        ),
    )
    return CostResult(
        commission_rate=config.commission_rate,
        slippage_bps=config.slippage_bps,
        cagr=result.metrics.cagr,
        hold_cagr=result.benchmark_metrics.cagr,
        doubled_cagr=doubled.metrics.cagr,
        doubled_hold_cagr=doubled.benchmark_metrics.cagr,
    )
