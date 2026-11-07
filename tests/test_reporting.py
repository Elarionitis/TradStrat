"""Tests for display formatting."""

from __future__ import annotations

import pandas as pd
import pytest

from tradstrat.metrics import calculate_metrics
from tradstrat.reporting import METRIC_LABELS, comparison_table, format_metric, probability_verdict


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
        (0.7, "Not significant"),
        (0.2, "No evidence"),
        (float("nan"), "Not enough"),
    ],
)
def test_verdicts_match_their_probability(probability: float, phrase: str) -> None:
    assert probability_verdict(probability).startswith(phrase)
