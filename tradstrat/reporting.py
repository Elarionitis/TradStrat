"""Turn metrics into display strings. Shared by the CLI and the dashboard."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Literal

import pandas as pd

from tradstrat.metrics import Metrics

Tone = Literal["good", "warning", "bad", "neutral"]

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
    """Describe the deflated Sharpe ratio of active returns, measured against buy and hold."""
    if math.isnan(probability):
        return "Not enough information to judge."
    if probability >= 0.95:
        return "Beat buy and hold at the 95% level, even after correcting for the number of trials."
    if probability >= 0.5:
        return "Not significant: the edge over buy and hold is consistent with picking the luckiest trial."
    return "No evidence of skill: the best trial did not beat buy and hold by more than luck alone would."


def overfitting_verdict(probability: float) -> str:
    """Describe the probability of backtest overfitting from a parameter search."""
    if math.isnan(probability):
        return "Not enough history to estimate."
    if probability <= 0.1:
        return "Low: the in-sample winner usually stayed in the top half on unseen data."
    if probability < 0.5:
        return "Moderate: the in-sample winner often slipped to the bottom half on unseen data."
    return "High: picking the in-sample winner did no better than picking at random."


@dataclass(frozen=True)
class Finding:
    """One plain-English observation about a backtest, with a tone for display."""

    tone: Tone
    title: str
    detail: str


@dataclass(frozen=True)
class Verdict:
    """A headline judgment of a strategy against buy and hold, and the findings behind it."""

    tone: Tone
    headline: str
    summary: str
    findings: tuple[Finding, ...]


def _compare(strategy: float, benchmark: float) -> Tone:
    if math.isnan(strategy) or math.isnan(benchmark):
        return "neutral"
    return "good" if strategy > benchmark else "bad"


def assess(strategy: Metrics, benchmark: Metrics, probability_beats_benchmark: float) -> Verdict:
    """Judge a strategy against buy and hold in the terms a reader cares about.

    ``probability_beats_benchmark`` is the bootstrap share of resampled histories in which the strategy's
    Sharpe ratio was higher. The verdict only claims an edge when that share reaches 95%.
    """
    return_tone = _compare(strategy.cagr, benchmark.cagr)
    sharpe_tone = _compare(strategy.sharpe_ratio, benchmark.sharpe_ratio)
    drawdown_tone = _compare(strategy.max_drawdown, benchmark.max_drawdown)
    probability = probability_beats_benchmark

    findings = [
        Finding(
            return_tone,
            "Return",
            f"{format_metric('cagr', strategy.cagr)} a year, against "
            f"{format_metric('cagr', benchmark.cagr)} for buy and hold.",
        ),
        Finding(
            sharpe_tone,
            "Risk-adjusted return",
            f"Sharpe ratio {format_metric('sharpe_ratio', strategy.sharpe_ratio)}, against "
            f"{format_metric('sharpe_ratio', benchmark.sharpe_ratio)} for buy and hold.",
        ),
        Finding(
            drawdown_tone,
            "Worst decline",
            f"{format_metric('max_drawdown', strategy.max_drawdown)} from peak, against "
            f"{format_metric('max_drawdown', benchmark.max_drawdown)} for buy and hold.",
        ),
    ]
    if not math.isnan(probability):
        findings.append(
            Finding(
                "good" if probability >= 0.95 else "bad" if probability < 0.5 else "warning",
                "Confidence",
                f"Its Sharpe ratio was higher in {probability:.0%} of resampled histories. "
                "An edge needs 95% or more.",
            )
        )
    if not math.isnan(strategy.exposure):
        findings.append(
            Finding(
                "neutral",
                "Time in the market",
                f"Invested {strategy.exposure:.0%} of the time across {strategy.number_of_trades} trades.",
            )
        )

    if strategy.number_of_trades == 0:
        tone: Tone = "warning"
        headline = "The strategy never traded"
        summary = "No entry signal fired. Try a longer history or different parameters."
    elif sharpe_tone == "good" and probability >= 0.95:
        tone, headline = "good", "Beat buy and hold, and probably not by luck"
        summary = "The edge held up in at least 95% of resampled histories. Confirm it with walk-forward."
    elif sharpe_tone == "good":
        tone, headline = "warning", "Beat buy and hold, but it could be luck"
        summary = "The risk-adjusted return was higher, but not consistently enough to rule out chance."
    elif drawdown_tone == "good":
        tone, headline = "warning", "A smoother ride, but buy and hold did better"
        summary = "It cut the worst decline, but earned less per unit of risk than simply holding."
    else:
        tone, headline = "bad", "Buy and hold did better"
        summary = "Simply holding the asset earned more per unit of risk, with no deeper decline."
    return Verdict(tone=tone, headline=headline, summary=summary, findings=tuple(findings))
