"""Tests for parameter search and walk-forward validation."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from conftest import make_bars, random_walk

from tradstrat.engine import BacktestConfig, run_backtest
from tradstrat.strategies import STRATEGIES, create_strategy
from tradstrat.validation import OBJECTIVES, build_grid, compare_strategies, grid_search, walk_forward

GRID = [
    {"short_window": 5, "long_window": 20},
    {"short_window": 10, "long_window": 40},
    {"short_window": 20, "long_window": 60},
]


def run_walk_forward(data: pd.DataFrame, **kwargs: object):
    options = {"train_bars": 200, "test_bars": 100, **kwargs}
    return walk_forward(data, "ma_crossover", GRID, BacktestConfig(), **options)


def test_grid_contains_only_valid_combinations() -> None:
    grid = build_grid("ma_crossover", points=6)

    assert grid
    assert all(params["short_window"] < params["long_window"] for params in grid)


def test_grid_overrides_replace_the_declared_range() -> None:
    grid = build_grid("ma_crossover", overrides={"short_window": [5], "long_window": [20, 30]})

    assert grid == [{"short_window": 5, "long_window": 20}, {"short_window": 5, "long_window": 30}]


def test_strategy_without_parameters_has_one_combination() -> None:
    assert build_grid("buy_and_hold") == [{}]


def test_unknown_strategy_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown strategy"):
        build_grid("astrology")


def test_search_ranks_best_first_and_reports_every_combination(walk: pd.DataFrame) -> None:
    result = grid_search(walk, "ma_crossover", GRID)

    assert len(result.table) == len(GRID)
    assert result.table["sharpe_ratio"].is_monotonic_decreasing


def test_winner_matches_a_direct_backtest_of_its_parameters(walk: pd.DataFrame) -> None:
    config = BacktestConfig(slippage_bps=10.0)

    result = grid_search(walk, "ma_crossover", GRID, config)

    direct = run_backtest(walk, create_strategy("ma_crossover", **result.best_params), config)
    assert result.table["sharpe_ratio"].iloc[0] == pytest.approx(direct.metrics.sharpe_ratio)
    assert all(isinstance(value, int) for value in result.best_params.values())


def test_configurations_that_never_trade_rank_last(walk: pd.DataFrame) -> None:
    """A flat curve has no Sharpe ratio, so it must not outrank a configuration that lost money."""
    grid = [{"short_window": 5, "long_window": 20}, {"short_window": 10, "long_window": 5000}]

    table = grid_search(walk, "ma_crossover", grid).table

    assert table["long_window"].iloc[-1] == 5000
    assert math.isnan(table["sharpe_ratio"].iloc[-1])


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_every_objective_is_usable(walk: pd.DataFrame, objective: str) -> None:
    assert grid_search(walk, "ma_crossover", GRID, objective=objective).objective == objective


def test_unknown_objective_is_rejected(walk: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Unknown objective"):
        grid_search(walk, "ma_crossover", GRID, objective="vibes")


def test_search_reports_deflated_sharpe_probabilities(walk: pd.DataFrame) -> None:
    result = grid_search(walk, "ma_crossover", GRID)

    assert 0.0 <= result.deflated_sharpe <= 1.0
    assert 0.0 <= result.deflated_active_sharpe <= 1.0


def test_owning_a_rising_market_is_not_mistaken_for_timing_skill() -> None:
    """Regression: a long-only strategy on a bull market reported 98% "significant" against zero.

    Prices here are pure drift plus noise, so no timing rule can have an edge over holding the asset.
    """
    rng = np.random.default_rng(3)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0012, 0.01, 2500)))
    data = make_bars(close)

    search = grid_search(data, "ma_crossover", build_grid("ma_crossover", 5))

    assert search.deflated_sharpe > 0.95
    assert search.deflated_active_sharpe < 0.5


def test_out_of_sample_curve_has_one_row_per_date() -> None:
    assert run_walk_forward(random_walk(800)).equity.index.is_unique


def test_out_of_sample_curve_starts_from_the_initial_capital() -> None:
    assert run_walk_forward(random_walk(800)).equity.iloc[0] == BacktestConfig().initial_capital


def test_test_windows_are_contiguous_and_follow_their_training_windows() -> None:
    data = random_walk(800)

    folds = run_walk_forward(data).folds

    assert (folds["Train End"] < folds["Test Start"]).all()
    positions = [data.index.get_loc(date) for date in folds["Test Start"]]
    assert positions == [200, 300, 400, 500, 600, 700]
    assert folds["Test End"].iloc[-1] == data.index[-1]


def test_sliding_windows_keep_their_length_and_anchored_windows_grow() -> None:
    data = random_walk(800)

    sliding = run_walk_forward(data).folds
    anchored = run_walk_forward(data, anchored=True).folds

    assert (sliding["Train Start"].iloc[1:] > sliding["Train Start"].iloc[0]).all()
    assert (anchored["Train Start"] == data.index[0]).all()


def test_all_trades_happen_inside_test_windows() -> None:
    data = random_walk(800)

    result = run_walk_forward(data)

    assert (result.trade_log["Entry Date"] >= result.folds["Test Start"].iloc[0]).all()
    for trade in result.trade_log.itertuples():
        matching = result.folds[
            (result.folds["Test Start"] <= trade[1]) & (trade[2] <= result.folds["Test End"])
        ]
        assert len(matching) == 1, "A trade must open and close within a single fold."


def test_no_future_data_reaches_a_completed_fold() -> None:
    """Rewriting everything after a fold must leave that fold's parameters and equity untouched."""
    data = random_walk(800)
    altered = data.copy()
    altered.iloc[500:, :4] = altered.iloc[500:, :4] * np.linspace(1.0, 3.0, 300)[:, None]

    original = run_walk_forward(data)
    rewritten = run_walk_forward(altered)

    completed = original.folds["Test End"] < data.index[500]
    assert completed.sum() == 3
    pd.testing.assert_frame_equal(original.folds[completed], rewritten.folds[completed])
    cutoff = data.index[499]
    pd.testing.assert_series_equal(original.equity.loc[:cutoff], rewritten.equity.loc[:cutoff])


def test_parameters_are_chosen_without_seeing_their_own_test_window() -> None:
    """Rewriting a fold's test window must not change which parameters that fold selected."""
    data = random_walk(800)
    altered = data.copy()
    altered.iloc[400:500, :4] = altered.iloc[400:500, :4] * np.linspace(1.0, 0.3, 100)[:, None]

    original = run_walk_forward(data).folds
    rewritten = run_walk_forward(altered).folds

    fold = original[original["Test Start"] == data.index[400]].index[0]
    assert original.loc[fold, "Parameters"] == rewritten.loc[fold, "Parameters"]
    assert original.loc[fold, "In-Sample Score"] == rewritten.loc[fold, "In-Sample Score"]
    assert original.loc[fold, "Test Return"] != rewritten.loc[fold, "Test Return"]


def test_capital_compounds_across_folds() -> None:
    result = run_walk_forward(random_walk(800))

    compounded = BacktestConfig().initial_capital * (1 + result.folds["Test Return"]).prod()

    assert result.equity.iloc[-1] == pytest.approx(compounded)


def test_trade_profits_reconcile_with_out_of_sample_equity() -> None:
    result = run_walk_forward(random_walk(800))

    change = result.equity.iloc[-1] - BacktestConfig().initial_capital
    assert result.trade_log["P&L"].sum() == pytest.approx(change)


def test_benchmark_covers_the_same_dates_as_the_strategy() -> None:
    result = run_walk_forward(random_walk(800))

    assert result.benchmark.index.equals(result.equity.index)


def test_in_sample_results_flatter_a_strategy_with_no_edge() -> None:
    """On driftless noise the best in-sample score should fall away out of sample."""
    gaps = []
    for seed in range(8):
        data = random_walk(800, seed=seed)
        data[["Open", "High", "Low", "Close"]] *= np.exp(-0.0003 * np.arange(800))[:, None]
        folds = run_walk_forward(data).folds
        gaps.append(folds["In-Sample Score"].mean() - folds["Out-of-Sample Score"].mean())

    assert np.mean(gaps) > 0.3


def test_efficiency_is_undefined_when_in_sample_results_are_negative() -> None:
    result = run_walk_forward(random_walk(800))
    losing = result.folds.assign(**{"In-Sample Score": -1.0})

    assert math.isnan(replace(result, folds=losing).efficiency)


def test_a_short_final_window_is_still_used() -> None:
    assert run_walk_forward(random_walk(650)).folds["Test End"].iloc[-1] == random_walk(650).index[-1]


@pytest.mark.parametrize(
    ("bars", "kwargs", "message"),
    [
        (150, {}, "more than 200"),
        (800, {"train_bars": 50}, "warm-up"),
        (800, {"test_bars": 1}, "at least two"),
    ],
)
def test_impossible_windows_are_rejected(bars: int, kwargs: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        run_walk_forward(random_walk(bars), **kwargs)


def test_search_estimates_the_probability_of_overfitting(walk: pd.DataFrame) -> None:
    result = grid_search(walk, "ma_crossover", GRID, partitions=8)

    assert result.overfitting is not None
    assert 0.0 <= result.overfitting.probability <= 1.0
    assert len(result.overfitting.logits) == 70  # 8 choose 4.


def test_overfitting_is_skipped_when_history_is_too_short_to_split() -> None:
    result = grid_search(random_walk(60), "ma_crossover", GRID, partitions=16)

    assert result.overfitting is None


def test_a_single_configuration_has_nothing_to_overfit(walk: pd.DataFrame) -> None:
    assert grid_search(walk, "ma_crossover", GRID[:1]).overfitting is None


def test_walk_forward_sharpe_is_measured_above_the_cash_rate() -> None:
    data = random_walk(800)

    plain = run_walk_forward(data)
    with_cash = walk_forward(
        data, "ma_crossover", GRID, BacktestConfig(cash_rate=0.04), train_bars=200, test_bars=100
    )

    assert with_cash.benchmark_metrics.sharpe_ratio < plain.benchmark_metrics.sharpe_ratio


def test_comparison_runs_every_strategy_and_buy_and_hold(walk: pd.DataFrame) -> None:
    comparison = compare_strategies(walk)

    assert set(comparison.table["Strategy"]) == {cls.label for cls in STRATEGIES.values()}
    assert list(comparison.equity.columns)[-1] == "Buy and Hold"
    assert comparison.table["sharpe_ratio"].is_monotonic_decreasing


def test_comparison_matches_a_direct_backtest(walk: pd.DataFrame) -> None:
    comparison = compare_strategies(walk, keys=["rsi"])

    direct = run_backtest(walk, create_strategy("rsi"))
    row = comparison.table.set_index("Strategy").loc["RSI Reversal"]
    assert row["total_return"] == pytest.approx(direct.metrics.total_return)


def test_comparison_skips_strategies_without_enough_history() -> None:
    comparison = compare_strategies(random_walk(100))

    assert "Moving Average Crossover" not in set(comparison.table["Strategy"])
    assert "RSI Reversal" in set(comparison.table["Strategy"])
