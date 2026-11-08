"""Tests for the statistical validation tools."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from tradstrat.metrics import sharpe_ratio
from tradstrat.stats import (
    _block_indices,
    bootstrap,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
)


def noise(observations: int = 1250, seed: int = 0, mean: float = 0.0) -> pd.Series:
    return pd.Series(np.random.default_rng(seed).normal(mean, 0.01, observations))


def paper_example(trials: int, skewness: float = -3.0, kurtosis: float = 10.0) -> float:
    """The numerical example from Bailey and Lopez de Prado (2014): annual SR 2.5, 5 daily years."""
    sharpe = 2.5 / math.sqrt(250)
    threshold = expected_max_sharpe(trials, 0.5 / 250)
    return probabilistic_sharpe(sharpe, 1250, skewness, kurtosis, threshold)


def test_deflated_sharpe_reproduces_the_published_example() -> None:
    """The paper states DSR = 0.9505 at N = 46, and "only a 90% chance" at its headline trial count."""
    assert paper_example(46) == pytest.approx(0.9505, abs=5e-5)
    assert paper_example(100) == pytest.approx(0.90, abs=0.005)


def test_probabilistic_sharpe_is_one_half_at_its_own_threshold() -> None:
    assert probabilistic_sharpe(0.1, 500, 0.0, 3.0, threshold=0.1) == pytest.approx(0.5)


def test_more_data_increases_confidence_in_the_same_sharpe() -> None:
    assert probabilistic_sharpe(0.05, 2000, 0.0, 3.0) > probabilistic_sharpe(0.05, 200, 0.0, 3.0)


def test_negative_skew_and_fat_tails_reduce_confidence() -> None:
    normal = probabilistic_sharpe(0.1, 500, 0.0, 3.0)

    assert probabilistic_sharpe(0.1, 500, -2.0, 3.0) < normal
    assert probabilistic_sharpe(0.1, 500, 0.0, 12.0) < normal


def test_expected_best_sharpe_grows_with_trials_and_dispersion() -> None:
    assert expected_max_sharpe(1000, 0.001) > expected_max_sharpe(10, 0.001)
    assert expected_max_sharpe(10, 0.01) > expected_max_sharpe(10, 0.001)
    assert expected_max_sharpe(1, 0.01) == 0.0


def test_expected_best_sharpe_matches_a_simulation() -> None:
    """The closed-form estimate should agree with actually taking the max of many random draws."""
    rng = np.random.default_rng(1)
    simulated = rng.normal(0, 0.1, size=(20_000, 50)).max(axis=1).mean()

    assert expected_max_sharpe(50, 0.01) == pytest.approx(simulated, rel=0.03)


def test_deflation_controls_false_discoveries_when_selecting_from_pure_noise() -> None:
    """The reason this module exists: the best of many skill-less strategies looks significant.

    Across many universes containing no edge at all, the naive test nearly always calls the winner
    significant. The deflated test stays at or below its 5% nominal rate. It is conservative here because
    the winner must clear the expected maximum by 1.645 standard errors of a single Sharpe estimate, which
    is much wider than the spread of the maximum itself.
    """
    rng = np.random.default_rng(0)
    naive_discoveries, deflated_discoveries, universes = 0, 0, 200
    for _ in range(universes):
        candidates = rng.normal(0, 0.01, size=(100, 750))
        sharpes = candidates.mean(axis=1) / candidates.std(axis=1, ddof=1) * math.sqrt(252)
        winner = pd.Series(candidates[int(np.argmax(sharpes))])
        naive_discoveries += probabilistic_sharpe_ratio(winner) > 0.95
        deflated_discoveries += deflated_sharpe_ratio(winner, sharpes.tolist()) > 0.95

    assert naive_discoveries / universes > 0.9
    assert deflated_discoveries / universes <= 0.05


def test_a_genuine_edge_survives_deflation() -> None:
    strong = noise(mean=0.002)
    sharpes = [sharpe_ratio(strong), *[sharpe_ratio(noise(seed=seed)) for seed in range(1, 30)]]

    assert deflated_sharpe_ratio(strong, sharpes) > 0.95


def test_deflated_sharpe_ignores_undefined_trials() -> None:
    returns = noise(mean=0.001)

    with_gaps = deflated_sharpe_ratio(returns, [1.0, float("nan"), 0.5, -0.2])

    assert with_gaps == deflated_sharpe_ratio(returns, [1.0, 0.5, -0.2])


def test_deflated_sharpe_needs_at_least_two_trials() -> None:
    assert math.isnan(deflated_sharpe_ratio(noise(), [1.0]))


def test_flat_returns_have_no_probabilistic_sharpe() -> None:
    assert math.isnan(probabilistic_sharpe_ratio(pd.Series([0.0] * 50)))


def test_block_indices_keep_blocks_contiguous_and_wrap_around() -> None:
    indices = _block_indices(100, 50, 10, np.random.default_rng(0))

    assert indices.shape == (50, 100)
    assert indices.min() >= 0
    assert indices.max() < 100
    steps = np.diff(indices[:, :10], axis=1) % 100
    assert (steps == 1).all()


def test_bootstrap_is_reproducible_with_a_seed() -> None:
    returns = noise(mean=0.0005)

    assert bootstrap(returns, seed=3).sharpe == bootstrap(returns, seed=3).sharpe
    assert bootstrap(returns, seed=3).sharpe != bootstrap(returns, seed=4).sharpe


def test_bootstrap_interval_brackets_the_observed_sharpe() -> None:
    returns = noise(mean=0.0005)

    result = bootstrap(returns)

    assert result.sharpe.low < result.sharpe.median < result.sharpe.high
    assert result.sharpe.contains(sharpe_ratio(returns))
    assert result.max_drawdown.high <= 0


def test_bootstrap_interval_covers_the_true_sharpe_at_about_the_stated_rate() -> None:
    """Calibration check: a 90% interval should contain the true value about 90% of the time."""
    true_sharpe = 0.0005 / 0.01 * math.sqrt(252)
    hits = [
        bootstrap(
            noise(750, seed=seed, mean=0.0005), samples=300, confidence=0.90, seed=seed
        ).sharpe.contains(true_sharpe)
        for seed in range(120)
    ]

    assert 0.80 <= np.mean(hits) <= 0.97


def test_bootstrap_detects_a_series_that_dominates_its_benchmark() -> None:
    benchmark = noise(seed=1, mean=0.0002)
    strategy = benchmark + 0.002

    assert bootstrap(strategy, benchmark).probability_beats_benchmark > 0.99
    assert bootstrap(benchmark, strategy).probability_beats_benchmark < 0.01


def test_bootstrap_without_a_benchmark_leaves_the_comparison_undefined() -> None:
    assert math.isnan(bootstrap(noise()).probability_beats_benchmark)


def test_bootstrap_rejects_short_series() -> None:
    with pytest.raises(ValueError, match="at least"):
        bootstrap(noise(10))


# Probability of backtest overfitting.


def _configurations(columns: int, rows: int = 1600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(0.0, 0.01, size=(rows, columns)))


def test_pure_noise_is_overfit_about_half_the_time() -> None:
    """With no skill anywhere, the in-sample winner's out-of-sample rank is a coin flip."""
    estimates = [
        probability_of_backtest_overfitting(_configurations(20, seed=seed), partitions=8).probability
        for seed in range(12)
    ]

    assert 0.35 < float(np.mean(estimates)) < 0.65


def test_a_genuinely_better_configuration_is_not_overfit() -> None:
    returns = _configurations(20)
    returns[0] += 0.002  # A strong, persistent edge for one configuration.

    result = probability_of_backtest_overfitting(returns, partitions=8)

    assert result.probability < 0.05
    assert result.probability_of_loss < 0.05


def test_every_split_of_the_partitions_is_evaluated() -> None:
    result = probability_of_backtest_overfitting(_configurations(5), partitions=10)

    assert len(result.logits) == 252  # 10 choose 5.
    assert result.partitions == 10


def test_ranking_is_shared_fairly_across_tied_configurations() -> None:
    """Identical columns tie, so the winner's rank is the middle and its logit is exactly zero."""
    column = _configurations(1)[0]
    returns = pd.concat([column] * 3, axis=1, ignore_index=True)

    result = probability_of_backtest_overfitting(returns, partitions=4)

    assert np.allclose(result.logits, 0.0)


def test_a_configuration_that_never_traded_ranks_last() -> None:
    returns = _configurations(3)
    returns[2] = 0.0  # Flat: its Sharpe ratio is undefined.

    result = probability_of_backtest_overfitting(returns, partitions=4)

    assert not np.isnan(result.logits).any()
    assert not np.isnan(result.in_sample_sharpe).any()


@pytest.mark.parametrize(
    ("columns", "rows", "partitions", "message"),
    [
        (1, 400, 8, "two configurations"),
        (3, 400, 7, "even number"),
        (3, 20, 8, "at least 32 returns"),
    ],
)
def test_unusable_inputs_are_rejected(columns: int, rows: int, partitions: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        probability_of_backtest_overfitting(_configurations(columns, rows), partitions=partitions)
