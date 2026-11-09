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
        return "Beat buy and hold by more than luck would explain, allowing for how many settings were tried."
    if probability >= 0.5:
        return (
            "Not convincing: an edge this size is what picking the luckiest of many settings would produce."
        )
    return "No evidence of skill: the best settings did not beat buy and hold by more than luck alone would."


def overfitting_verdict(probability: float) -> str:
    """Describe the probability of backtest overfitting from a parameter search."""
    if math.isnan(probability):
        return "Not enough history to estimate."
    if probability <= 0.1:
        return "Low: the winning settings usually stayed in the top half on unseen data."
    if probability < 0.5:
        return "Moderate: the winning settings often slipped to the bottom half on unseen data."
    return "High: picking the winner in hindsight did no better than picking settings at random."


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
    takeaway: str = ""


@dataclass(frozen=True)
class Round:
    """One head-to-head comparison between the strategy and buy and hold.

    ``winner`` is ``"rule"`` only when the strategy did strictly better, so a tie goes to holding, and is
    ``None`` when either value is undefined.
    """

    title: str
    term: str
    metric: str
    rule: float
    hold: float
    winner: str | None


def rounds(strategy: Metrics, benchmark: Metrics) -> tuple[Round, ...]:
    """Growth, return for the risk, and the worst crash: the three things a newcomer asks about."""
    contests = (
        ("Growth per year", "CAGR", "cagr"),
        ("Return for the risk", "Sharpe Ratio", "sharpe_ratio"),
        ("Worst crash", "Max Drawdown", "max_drawdown"),
    )
    results = []
    for title, term, metric in contests:
        rule, hold = getattr(strategy, metric), getattr(benchmark, metric)
        tone = _compare(rule, hold)
        winner = None if tone == "neutral" else "rule" if tone == "good" else "hold"
        results.append(Round(title, term, metric, rule, hold, winner))
    return tuple(results)


def _compare(strategy: float, benchmark: float) -> Tone:
    if math.isnan(strategy) or math.isnan(benchmark):
        return "neutral"
    return "good" if strategy > benchmark else "bad"


def assess(
    strategy: Metrics, benchmark: Metrics, probability_beats_benchmark: float, asset: str = "the asset"
) -> Verdict:
    """Judge a strategy against buy and hold in the terms a reader cares about.

    ``asset`` names what was held, so the headline can say "just holding SPY" rather than an abstraction.

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
        takeaway = (
            "Its buy condition never came true in this stretch of history, so there is nothing to compare. "
            "Try a longer history, another market or different settings."
        )
    elif sharpe_tone == "good" and strategy.cagr < benchmark.cagr:
        # Winning on return for the risk while growing less is common for rules that sit in cash, and calling
        # it a win outright contradicts the money line a newcomer reads next.
        tone, headline = "warning", f"Better for the risk, but just holding {asset} made more money"
        luck = (
            "That lead held up in at least 95% of reshuffled histories."
            if probability >= 0.95
            else "That lead was not consistent across reshuffled histories, so it could also be luck."
        )
        summary = f"A higher risk-adjusted return, but lower growth than buying {asset} and holding it."
        takeaway = (
            "Each unit of risk earned more than holding did, but the rule sat in cash part of the time, "
            "so it grew less overall. It suits someone who values a calmer ride over the biggest total. "
            f"{luck}"
        )
    elif sharpe_tone == "good" and probability >= 0.95:
        tone, headline = "good", f"Beat just holding {asset}, and probably not by luck"
        summary = "The edge held up in at least 95% of resampled histories. Confirm it with walk-forward."
        takeaway = (
            "That is rare. Across thousands of reshuffled versions of this history it came out ahead at "
            "least 95% of the time. The next step is the walk-forward test in the Tune tab, which checks the "
            "idea on data it was never tuned on."
        )
    elif sharpe_tone == "good":
        tone, headline = "warning", f"Beat just holding {asset}, but it could be luck"
        summary = "The risk-adjusted return was higher, but not consistently enough to rule out chance."
        takeaway = (
            "It earned more for the risk it took, but the lead was not consistent across reshuffled "
            "histories, so luck can't be ruled out. A longer history, or the walk-forward test in the Tune "
            "tab, would tell you more."
        )
    elif drawdown_tone == "good":
        tone, headline = "warning", f"A smoother ride, but just holding {asset} did better"
        summary = (
            f"It cut the worst decline, but earned less per unit of risk than buying {asset} and holding it."
        )
        takeaway = (
            "The rule spent part of the time in cash, so it missed some of the gains but also some of the "
            "falls. That trade can still be worth it for someone who would sell in a panic during a crash, "
            "even though holding grew more."
        )
    else:
        tone, headline = "bad", f"Just holding {asset} did better"
        summary = f"Buying {asset} and holding it earned more per unit of risk, with no deeper decline."
        takeaway = (
            "Here, staying invested beat stepping in and out. That is the most common result for timing "
            "rules, and a useful one: it is why most investors are told to simply hold. Try another market, "
            "period or strategy to see where timing does help."
        )
    return Verdict(tone=tone, headline=headline, summary=summary, findings=tuple(findings), takeaway=takeaway)


def assess_search(deflated_sharpe: float, overfitting: float) -> Verdict:
    """Judge whether the best setting of a search is a real edge over buy and hold, or search luck.

    ``deflated_sharpe`` is the deflated Sharpe ratio of the winner against buy and hold, and ``overfitting``
    the probability of backtest overfitting, which is NaN when the history is too short to estimate it.
    """
    findings = (
        Finding(
            "good" if deflated_sharpe >= 0.95 else "bad",
            "Luck of the search",
            probability_verdict(deflated_sharpe),
        ),
        Finding(
            "neutral"
            if math.isnan(overfitting)
            else "good"
            if overfitting <= 0.1
            else "bad"
            if overfitting >= 0.5
            else "warning",
            "Overfitting",
            overfitting_verdict(overfitting),
        ),
    )
    stable = math.isnan(overfitting) or overfitting < 0.5
    if deflated_sharpe >= 0.95 and stable:
        return Verdict(
            "good",
            "The best settings look like a real edge",
            "They beat buy and hold by more than luck would explain. Confirm it on unseen data below.",
            findings,
        )
    if deflated_sharpe >= 0.95:
        return Verdict(
            "warning",
            "Promising, but fragile",
            "The winner beat buy and hold, but winners chosen this way often fail on new data.",
            findings,
        )
    if not stable:
        return Verdict(
            "bad",
            "The best settings look like luck",
            "Try enough settings and one will look good. Nothing here suggests it would keep working.",
            findings,
        )
    return Verdict(
        "warning",
        "No clear edge over buy and hold",
        "The best settings did not beat buy and hold by more than trying many settings would explain.",
        findings,
    )
