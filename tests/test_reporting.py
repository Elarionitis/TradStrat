"""Tests for display formatting."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from tradstrat.metrics import Metrics, calculate_metrics
from tradstrat.reporting import (
    METRIC_LABELS,
    assess,
    assess_search,
    comparison_table,
    format_metric,
    overfitting_verdict,
    probability_verdict,
)


@pytest.mark.parametrize(
    ("name", "value", "expected"),
    [
        ("total_return", 0.1234, "12.34%"),
        ("sharpe_ratio", 1.2345, "1.23"),
        ("average_win", 1234.5, "$1,234.50"),
        ("average_loss", -937.5, "-$937.50"),
        ("max_drawdown_duration", 42, "42 bars"),
        ("number_of_trades", 7, "7"),
        ("sharpe_ratio", float("nan"), "n/a"),
    ],
)
def test_format_metric(name: str, value: float, expected: str) -> None:
    assert format_metric(name, value) == expected


def test_every_metric_has_a_label() -> None:
    metrics = calculate_metrics(pd.Series([100.0, 101.0]), pd.DataFrame())

    assert set(METRIC_LABELS) == set(metrics.as_dict())


def test_comparison_hides_trade_statistics_for_the_benchmark() -> None:
    metrics = calculate_metrics(pd.Series([100.0, 101.0, 99.0]), pd.DataFrame({"P&L": [1.0]}))

    table = comparison_table(metrics, metrics)

    assert table.loc["Win Rate", "Buy and Hold"] == "-"
    assert table.loc["Total Return", "Buy and Hold"] == table.loc["Total Return", "Strategy"]


@pytest.mark.parametrize(
    ("probability", "phrase"),
    [
        (0.97, "Beat buy and hold"),
        (0.7, "Not convincing"),
        (0.2, "No evidence"),
        (float("nan"), "Not enough"),
    ],
)
def test_verdicts_match_their_probability(probability: float, phrase: str) -> None:
    assert probability_verdict(probability).startswith(phrase)


BASE = Metrics(
    total_return=1.0,
    cagr=0.10,
    annual_volatility=0.15,
    sharpe_ratio=0.8,
    sortino_ratio=1.0,
    calmar_ratio=0.3,
    max_drawdown=-0.30,
    max_drawdown_duration=100,
    number_of_trades=12,
    win_rate=0.5,
    average_win=100.0,
    average_loss=-50.0,
    profit_factor=2.0,
    expectancy=25.0,
    exposure=0.6,
)


@pytest.mark.parametrize(
    ("strategy", "probability", "tone", "headline"),
    [
        (replace(BASE, sharpe_ratio=1.2), 0.97, "good", "probably not by luck"),
        (replace(BASE, sharpe_ratio=1.2), 0.70, "warning", "could be luck"),
        (replace(BASE, sharpe_ratio=0.5, max_drawdown=-0.10), 0.20, "warning", "smoother ride"),
        (replace(BASE, sharpe_ratio=0.5, max_drawdown=-0.40), 0.20, "bad", "Buy and hold did better"),
        (replace(BASE, number_of_trades=0, sharpe_ratio=float("nan")), float("nan"), "warning", "never"),
    ],
)
def test_verdict_only_claims_an_edge_with_95_percent_confidence(
    strategy: Metrics, probability: float, tone: str, headline: str
) -> None:
    verdict = assess(strategy, BASE, probability)

    assert verdict.tone == tone
    assert headline in verdict.headline


def test_findings_compare_each_dimension_with_buy_and_hold() -> None:
    strategy = replace(BASE, cagr=0.05, sharpe_ratio=0.9, max_drawdown=-0.20)

    findings = {finding.title: finding for finding in assess(strategy, BASE, 0.6).findings}

    assert findings["Return"].tone == "bad"
    assert findings["Risk-adjusted return"].tone == "good"
    assert findings["Worst decline"].tone == "good"
    assert findings["Confidence"].tone == "warning"
    assert "60%" in findings["Confidence"].detail
    assert "12 trades" in findings["Time in the market"].detail


@pytest.mark.parametrize(
    ("probability", "phrase"), [(0.05, "Low"), (0.3, "Moderate"), (0.6, "High"), (float("nan"), "Not enough")]
)
def test_overfitting_verdict_bands(probability: float, phrase: str) -> None:
    assert overfitting_verdict(probability).startswith(phrase)


@pytest.mark.parametrize(
    ("deflated", "overfitting", "tone", "headline"),
    [
        (0.98, 0.05, "good", "real edge"),
        (0.98, float("nan"), "good", "real edge"),
        (0.98, 0.7, "warning", "fragile"),
        (0.30, 0.7, "bad", "luck"),
        (0.30, 0.2, "warning", "No clear edge"),
    ],
)
def test_search_verdict_needs_both_checks_to_pass(
    deflated: float, overfitting: float, tone: str, headline: str
) -> None:
    verdict = assess_search(deflated, overfitting)

    assert verdict.tone == tone
    assert headline in verdict.headline
    assert [finding.title for finding in verdict.findings] == ["Luck of the search", "Overfitting"]
