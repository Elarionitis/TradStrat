"""Performance metrics for backtest results.

Ratios that are mathematically undefined (a flat equity curve has no Sharpe ratio, a strategy with no
losing trades has no finite profit factor denominator) are reported as ``NaN`` rather than ``0.0``.
A zero would rank a strategy that never traded above one that lost money during a parameter search.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Final

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR: Final[int] = 252
DAYS_PER_YEAR: Final[float] = 365.25


@dataclass(frozen=True)
class Metrics:
    """Headline statistics for one equity curve and its trades."""

    total_return: float
    cagr: float
    annual_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    number_of_trades: int
    win_rate: float
    average_win: float
    average_loss: float
    profit_factor: float
    expectancy: float
    exposure: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class RelativeMetrics:
    """How a strategy behaved relative to its benchmark."""

    alpha: float
    beta: float
    correlation: float
    information_ratio: float


def daily_returns(equity_curve: pd.Series) -> pd.Series:
    """Convert an equity curve into simple daily returns."""
    return equity_curve.astype(float).pct_change().dropna()


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualized Sharpe ratio. ``risk_free_rate`` is an annual rate."""
    excess = returns - risk_free_rate / TRADING_DAYS_PER_YEAR
    if len(excess) < 2:
        return float("nan")
    deviation = float(excess.std(ddof=1))
    if np.isclose(deviation, 0.0):
        return float("nan")
    return float(excess.mean() / deviation * np.sqrt(TRADING_DAYS_PER_YEAR))


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualized Sortino ratio.

    Downside deviation is the root mean square of below-target returns taken over *all* observations,
    with gains counted as zero. Taking the standard deviation of only the negative returns is a common
    mistake: it measures how much the losses differ from each other, so a strategy that loses the same
    amount every time would appear to have no downside risk.
    """
    excess = returns - risk_free_rate / TRADING_DAYS_PER_YEAR
    if excess.empty:
        return float("nan")
    downside_deviation = float(np.sqrt((excess.clip(upper=0.0) ** 2).mean()))
    if np.isclose(downside_deviation, 0.0):
        return float("nan")
    return float(excess.mean() / downside_deviation * np.sqrt(TRADING_DAYS_PER_YEAR))


def _years_elapsed(equity_curve: pd.Series) -> float:
    """Length of the curve in years, from calendar dates when the index has them."""
    if isinstance(equity_curve.index, pd.DatetimeIndex):
        return (equity_curve.index[-1] - equity_curve.index[0]).days / DAYS_PER_YEAR
    return (len(equity_curve) - 1) / TRADING_DAYS_PER_YEAR


def cagr(equity_curve: pd.Series) -> float:
    """Compound annual growth rate."""
    years = _years_elapsed(equity_curve)
    if years <= 0:
        return float("nan")
    growth = float(equity_curve.iloc[-1] / equity_curve.iloc[0])
    return -1.0 if growth <= 0 else growth ** (1 / years) - 1


def drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """Fractional decline from the running peak at each point in time."""
    return equity_curve / equity_curve.cummax() - 1


def max_drawdown_duration(equity_curve: pd.Series) -> int:
    """Longest run of consecutive bars spent below a previous peak."""
    underwater = drawdown_series(equity_curve) < 0
    if not underwater.any():
        return 0
    return int(underwater.groupby((~underwater).cumsum()).sum().max())


def calculate_metrics(
    equity_curve: pd.Series,
    trade_log: pd.DataFrame,
    positions: pd.Series | None = None,
    risk_free_rate: float = 0.0,
) -> Metrics:
    """Calculate return, risk, and trade-quality statistics from a backtest."""
    if equity_curve.empty:
        raise ValueError("Metrics require a non-empty equity curve.")

    equity_curve = equity_curve.astype(float)
    returns = daily_returns(equity_curve)
    annual_growth = cagr(equity_curve)
    max_drawdown = float(drawdown_series(equity_curve).min())
    volatility = float("nan")
    if len(returns) > 1:
        volatility = float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))

    trade_count = len(trade_log)
    profits = trade_log["P&L"].astype(float) if trade_count else pd.Series(dtype=float)
    wins = profits[profits > 0]
    losses = profits[profits < 0]
    gross_loss = float(-losses.sum())

    return Metrics(
        total_return=float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1),
        cagr=annual_growth,
        annual_volatility=volatility,
        sharpe_ratio=sharpe_ratio(returns, risk_free_rate),
        sortino_ratio=sortino_ratio(returns, risk_free_rate),
        calmar_ratio=annual_growth / abs(max_drawdown) if max_drawdown < 0 else float("nan"),
        max_drawdown=max_drawdown,
        max_drawdown_duration=max_drawdown_duration(equity_curve),
        number_of_trades=trade_count,
        win_rate=len(wins) / trade_count if trade_count else float("nan"),
        average_win=float(wins.mean()) if not wins.empty else float("nan"),
        average_loss=float(losses.mean()) if not losses.empty else float("nan"),
        profit_factor=float(wins.sum()) / gross_loss if gross_loss > 0 else float("nan"),
        expectancy=float(profits.mean()) if trade_count else float("nan"),
        exposure=float((positions != 0).mean()) if positions is not None else float("nan"),
    )


def calculate_relative_metrics(
    strategy_equity: pd.Series, benchmark_equity: pd.Series, risk_free_rate: float = 0.0
) -> RelativeMetrics:
    """Regress strategy returns on benchmark returns (CAPM-style alpha and beta)."""
    aligned = pd.concat(
        [daily_returns(strategy_equity), daily_returns(benchmark_equity)], axis=1, join="inner"
    ).dropna()
    undefined = RelativeMetrics(*[float("nan")] * 4)
    if len(aligned) < 2:
        return undefined

    strategy = aligned.iloc[:, 0].to_numpy() - risk_free_rate / TRADING_DAYS_PER_YEAR
    benchmark = aligned.iloc[:, 1].to_numpy() - risk_free_rate / TRADING_DAYS_PER_YEAR
    benchmark_variance = float(np.var(benchmark, ddof=1))
    if np.isclose(benchmark_variance, 0.0) or np.isclose(float(np.var(strategy, ddof=1)), 0.0):
        return undefined

    beta = float(np.cov(strategy, benchmark, ddof=1)[0, 1] / benchmark_variance)
    active = strategy - benchmark
    tracking_error = float(np.std(active, ddof=1))
    return RelativeMetrics(
        alpha=float((strategy.mean() - beta * benchmark.mean()) * TRADING_DAYS_PER_YEAR),
        beta=beta,
        correlation=float(np.corrcoef(strategy, benchmark)[0, 1]),
        information_ratio=(
            float(active.mean() / tracking_error * np.sqrt(TRADING_DAYS_PER_YEAR))
            if not np.isclose(tracking_error, 0.0)
            else float("nan")
        ),
    )
