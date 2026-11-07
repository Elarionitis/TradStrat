"""Performance metrics for backtest results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR: Final[int] = 252


@dataclass(frozen=True)
class Metrics:
    """Headline metrics for a backtest run."""

    total_return: float
    cagr: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    win_rate: float
    average_win: float
    average_loss: float
    number_of_trades: int


def _daily_returns(equity_curve: pd.Series) -> pd.Series:
    """Convert an equity curve into simple daily returns."""
    return equity_curve.pct_change().dropna()


def calculate_metrics(equity_curve: pd.Series, trade_log: pd.DataFrame) -> Metrics:
    """Calculate return, risk, and trade-quality statistics from a backtest."""
    if equity_curve.empty:
        raise ValueError("Metrics require a non-empty equity curve.")

    equity_curve = equity_curve.astype(float)
    daily_returns = _daily_returns(equity_curve)

    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    periods = max(len(equity_curve) - 1, 1)
    cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (TRADING_DAYS_PER_YEAR / periods) - 1

    return_std = daily_returns.std(ddof=0)
    downside_returns = daily_returns[daily_returns < 0]
    downside_std = downside_returns.std(ddof=0)

    sharpe_ratio = 0.0 if np.isclose(return_std, 0.0) else (daily_returns.mean() / return_std) * np.sqrt(TRADING_DAYS_PER_YEAR)
    sortino_ratio = 0.0 if np.isclose(downside_std, 0.0) else (daily_returns.mean() / downside_std) * np.sqrt(TRADING_DAYS_PER_YEAR)

    running_peak = equity_curve.cummax()
    drawdown = equity_curve / running_peak - 1
    max_drawdown = float(drawdown.min())

    duration = 0
    max_duration = 0
    for value in drawdown:
        if value < 0:
            duration += 1
            max_duration = max(max_duration, duration)
        else:
            duration = 0

    trade_count = 0 if trade_log.empty else len(trade_log)
    if trade_count == 0:
        return Metrics(
            total_return=float(total_return),
            cagr=float(cagr),
            sharpe_ratio=float(sharpe_ratio),
            sortino_ratio=float(sortino_ratio),
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_duration,
            win_rate=0.0,
            average_win=0.0,
            average_loss=0.0,
            number_of_trades=0,
        )

    profits = trade_log["P&L"].astype(float)
    wins = profits[profits > 0]
    losses = profits[profits < 0]

    win_rate = len(wins) / trade_count if trade_count else 0.0
    average_win = float(wins.mean()) if not wins.empty else 0.0
    average_loss = float(losses.mean()) if not losses.empty else 0.0

    return Metrics(
        total_return=float(total_return),
        cagr=float(cagr),
        sharpe_ratio=float(sharpe_ratio),
        sortino_ratio=float(sortino_ratio),
        max_drawdown=max_drawdown,
        max_drawdown_duration=max_duration,
        win_rate=float(win_rate),
        average_win=average_win,
        average_loss=average_loss,
        number_of_trades=trade_count,
    )


def performance_summary(equity_curve: pd.Series, benchmark_curve: pd.Series, trade_log: pd.DataFrame) -> dict[str, float]:
    """Return a flat dictionary for the Streamlit dashboard."""

    metrics = calculate_metrics(equity_curve, trade_log)
    benchmark_total_return = float(benchmark_curve.iloc[-1] / benchmark_curve.iloc[0] - 1) if not benchmark_curve.empty else 0.0
    return {
        "Total Return": metrics.total_return,
        "CAGR": metrics.cagr,
        "Sharpe Ratio": metrics.sharpe_ratio,
        "Sortino Ratio": metrics.sortino_ratio,
        "Max Drawdown": metrics.max_drawdown,
        "Max Drawdown Duration": float(metrics.max_drawdown_duration),
        "Win Rate": metrics.win_rate,
        "Average Win": metrics.average_win,
        "Average Loss": metrics.average_loss,
        "Number of Trades": float(metrics.number_of_trades),
        "Benchmark Total Return": benchmark_total_return,
    }
