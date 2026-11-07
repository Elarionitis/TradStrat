"""Tests for chart construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import random_walk

from tradstrat.charts import (
    bootstrap_figure,
    drawdown_figure,
    equity_figure,
    fold_scores_figure,
    monthly_returns_figure,
    monthly_returns_table,
    price_figure,
    search_figure,
)
from tradstrat.engine import run_backtest
from tradstrat.glossary import GLOSSARY
from tradstrat.reporting import METRIC_LABELS
from tradstrat.strategies import MovingAverageCrossoverStrategy
from tradstrat.validation import build_grid, grid_search, walk_forward


@pytest.fixture
def result():
    return run_backtest(random_walk(500), MovingAverageCrossoverStrategy(10, 40))


def test_equity_and_drawdown_show_both_series(result) -> None:
    frame = result.results

    assert len(equity_figure(frame["Strategy Equity"], frame["Buy and Hold Benchmark"]).data) == 2
    assert len(drawdown_figure(frame["Strategy Drawdown"], frame["Benchmark Drawdown"]).data) == 2


def test_price_chart_marks_buys_and_sells(result) -> None:
    figure = price_figure(result.results["Close"], result.trade_log)

    assert [trace.name for trace in figure.data] == ["Close", "Buy", "Sell"]


def test_price_chart_handles_no_trades(result) -> None:
    assert len(price_figure(result.results["Close"], result.trade_log.iloc[0:0]).data) == 1


def test_monthly_returns_compound_correctly() -> None:
    index = pd.to_datetime(["2024-01-02", "2024-01-31", "2024-02-29", "2024-03-29"])

    table = monthly_returns_table(pd.Series([100.0, 110.0, 121.0, 108.9], index=index))

    assert table.loc[2024, 1] == pytest.approx(0.10)
    assert table.loc[2024, 2] == pytest.approx(0.10)
    assert table.loc[2024, 3] == pytest.approx(-0.10)
    assert np.isnan(table.loc[2024, 12])


def test_monthly_heatmap_is_centered_on_zero(result) -> None:
    heatmap = monthly_returns_figure(result.results["Strategy Equity"]).data[0]

    assert heatmap.zmin == -heatmap.zmax


def test_bootstrap_figure_ignores_undefined_samples() -> None:
    figure = bootstrap_figure(np.array([0.1, np.nan, 0.5]), 0.1, 0.5, 0.3)

    assert len(figure.data[0].x) == 2


@pytest.mark.parametrize(
    ("key", "kind"), [("momentum", "scatter"), ("ma_crossover", "heatmap"), ("rsi", "heatmap")]
)
def test_search_figure_fits_the_number_of_parameters(key: str, kind: str) -> None:
    from tradstrat.strategies import STRATEGIES

    search = grid_search(random_walk(400), key, build_grid(key, 3))

    figure = search_figure(search.table, [spec.name for spec in STRATEGIES[key].params], search.objective)

    assert figure.data[0].type == kind


def test_fold_chart_compares_in_and_out_of_sample() -> None:
    grid = [{"short_window": 5, "long_window": 20}, {"short_window": 10, "long_window": 40}]
    result = walk_forward(random_walk(600), "ma_crossover", grid, train_bars=200, test_bars=100)

    assert [trace.name for trace in fold_scores_figure(result.folds, "sharpe_ratio").data] == [
        "In-sample",
        "Out-of-sample",
    ]


def test_every_displayed_metric_has_a_glossary_entry() -> None:
    """Regression: missing terms made the help text repeat the label back, as 'Total Return: Total Return'."""
    assert set(METRIC_LABELS.values()) <= set(GLOSSARY)
