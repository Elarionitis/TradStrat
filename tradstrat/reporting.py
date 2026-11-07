"""Turn metrics into display strings. Shared by the CLI and the dashboard."""

from __future__ import annotations

import math
from typing import Final

import pandas as pd

from tradstrat.metrics import Metrics

METRIC_LABELS: Final[dict[str, str]] = {
    "total_return": "Total Return",
    "cagr": "CAGR",
    "annual_volatility": "Annual Volatility",
    "sharpe_ratio": "Sharpe Ratio",
    "sortino_ratio": "Sortino Ratio",
    "calmar_ratio": "Calmar Ratio",
    "max_drawdown": "Max Drawdown",
    "max_drawdown_duration": "Max Drawdown Duration",
    "number_of_trades": "Number of Trades",
    "win_rate": "Win Rate",
    "average_win": "Average Win",
    "average_loss": "Average Loss",
    "profit_factor": "Profit Factor",
    "expectancy": "Expectancy",
    "exposure": "Exposure",
}
PERCENT_METRICS: Final[frozenset[str]] = frozenset(
    {"total_return", "cagr", "annual_volatility", "max_drawdown", "win_rate", "exposure"}
)
CURRENCY_METRICS: Final[frozenset[str]] = frozenset({"average_win", "average_loss", "expectancy"})
BENCHMARK_ONLY_HIDDEN: Final[frozenset[str]] = frozenset(
    {"number_of_trades", "win_rate", "average_win", "average_loss", "profit_factor", "expectancy", "exposure"}
)


def format_metric(name: str, value: float) -> str:
    """Format one metric for display. Undefined values show as ``n/a``."""
    if math.isnan(value):
        return "n/a"
    if name in PERCENT_METRICS:
        return f"{value:.2%}"
    if name in CURRENCY_METRICS:
        return f"{'-' if value < 0 else ''}${abs(value):,.2f}"
    if name == "max_drawdown_duration":
        return f"{int(value)} bars"
    if name == "number_of_trades":
        return str(int(value))
    return f"{value:.2f}"


def comparison_table(strategy: Metrics, benchmark: Metrics) -> pd.DataFrame:
    """Build a side-by-side table of a strategy against its benchmark."""
    strategy_values = strategy.as_dict()
    benchmark_values = benchmark.as_dict()
    rows = {
        label: {
            "Strategy": format_metric(name, strategy_values[name]),
            "Buy and Hold": (
                "-" if name in BENCHMARK_ONLY_HIDDEN else format_metric(name, benchmark_values[name])
            ),
        }
        for name, label in METRIC_LABELS.items()
    }
    return pd.DataFrame.from_dict(rows, orient="index")


def probability_verdict(probability: float) -> str:
    """Describe a deflated Sharpe probability in plain language."""
    if math.isnan(probability):
        return "Not enough information to judge."
    if probability >= 0.95:
        return "Significant at the 95% level, even after correcting for the number of trials."
    if probability >= 0.5:
        return "Not significant: this result is consistent with selecting the luckiest configuration."
    return "No evidence of skill: the winner did no better than the best of this many random trials would."
