"""Statistical tests of whether a backtest result is believable.

A single Sharpe ratio says nothing about how much of it is luck. The tools here estimate that:

* ``bootstrap`` puts confidence intervals on a result by resampling its returns.
* ``probabilistic_sharpe_ratio`` asks whether a Sharpe ratio is significant given the sample length and
  the skew and fat tails of the returns.
* ``deflated_sharpe_ratio`` also corrects for how many configurations were tried. The best of many
  backtests looks good even when none of them has any edge, so it must clear a higher bar.
* ``probability_of_backtest_overfitting`` asks how often the configuration that looks best on one half
  of history lands in the bottom half of configurations on the other.

References:
    Bailey and Lopez de Prado (2012), "The Sharpe Ratio Efficient Frontier".
    Bailey and Lopez de Prado (2014), "The Deflated Sharpe Ratio".
    Bailey, Borwein, Lopez de Prado and Zhu (2017), "The Probability of Backtest Overfitting".
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from statistics import NormalDist
from typing import Final

import numpy as np
import pandas as pd

from tradstrat.metrics import TRADING_DAYS_PER_YEAR

EULER_MASCHERONI: Final[float] = 0.5772156649015329
NORMAL: Final[NormalDist] = NormalDist()


def _sharpe_standard_error_scale(sharpe: float, skewness: float, kurtosis: float) -> float:
    """Non-normality term of the Sharpe ratio's standard error. ``kurtosis`` is 3 for a normal."""
    return math.sqrt(max(1 - skewness * sharpe + (kurtosis - 1) / 4 * sharpe**2, 1e-12))


def probabilistic_sharpe(
    sharpe: float, observations: int, skewness: float, kurtosis: float, threshold: float = 0.0
) -> float:
    """Probability that the true Sharpe ratio exceeds ``threshold``. Sharpe ratios here are per-period."""
    if observations < 2:
        return float("nan")
    scale = _sharpe_standard_error_scale(sharpe, skewness, kurtosis)
    return NORMAL.cdf((sharpe - threshold) * math.sqrt(observations - 1) / scale)


def expected_max_sharpe(trials: int, variance: float) -> float:
    """Expected best per-period Sharpe among ``trials`` strategies that have no skill at all."""
    if trials < 2:
        return 0.0
    first = NORMAL.inv_cdf(1 - 1 / trials)
    second = NORMAL.inv_cdf(1 - 1 / (trials * math.e))
    return math.sqrt(variance) * ((1 - EULER_MASCHERONI) * first + EULER_MASCHERONI * second)


def _per_period_moments(returns: pd.Series) -> tuple[float, float, float]:
    """Per-period Sharpe, skewness and (non-excess) kurtosis of a return series."""
    deviation = float(returns.std(ddof=1))
    if len(returns) < 3 or np.isclose(deviation, 0.0):
        return float("nan"), float("nan"), float("nan")
    centered = (returns - returns.mean()) / returns.std(ddof=0)
    return float(returns.mean() / deviation), float((centered**3).mean()), float((centered**4).mean())


def probabilistic_sharpe_ratio(returns: pd.Series, annual_threshold: float = 0.0) -> float:
    """Probability that a return series has a true annual Sharpe ratio above ``annual_threshold``."""
    sharpe, skewness, kurtosis = _per_period_moments(returns)
    if math.isnan(sharpe):
        return float("nan")
    threshold = annual_threshold / math.sqrt(TRADING_DAYS_PER_YEAR)
    return probabilistic_sharpe(sharpe, len(returns), skewness, kurtosis, threshold)


def deflated_sharpe_ratio(returns: pd.Series, trial_sharpes: Sequence[float]) -> float:
    """Probability that the selected strategy has real skill, after correcting for selection.

    ``returns`` belong to the chosen configuration and ``trial_sharpes`` are the annualized Sharpe ratios
    of every configuration that was tried, including the chosen one. Parameter neighbours are correlated,
    so counting each as an independent trial makes this test somewhat conservative.
    """
    sharpes = np.asarray([value for value in trial_sharpes if not math.isnan(value)], dtype=float)
    sharpe, skewness, kurtosis = _per_period_moments(returns)
    if math.isnan(sharpe) or len(sharpes) < 2:
        return float("nan")
    variance = float(np.var(sharpes, ddof=1)) / TRADING_DAYS_PER_YEAR
    threshold = expected_max_sharpe(len(sharpes), variance)
    return probabilistic_sharpe(sharpe, len(returns), skewness, kurtosis, threshold)


@dataclass(frozen=True)
class Interval:
    """A bootstrap estimate with its confidence bounds."""

    low: float
    median: float
    high: float

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high


@dataclass(frozen=True)
class BootstrapResult:
    """Confidence intervals from resampling a return series."""

    sharpe: Interval
    annual_return: Interval
    max_drawdown: Interval
    probability_positive_sharpe: float
    probability_beats_benchmark: float
    sharpe_samples: np.ndarray


def _block_indices(
    observations: int, samples: int, block_length: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw circular block-bootstrap indices with shape ``(samples, observations)``."""
    blocks = math.ceil(observations / block_length)
    starts = rng.integers(0, observations, size=(samples, blocks))
    indices = (starts[:, :, None] + np.arange(block_length)) % observations
    return indices.reshape(samples, -1)[:, :observations]


def _sharpe_by_row(returns: np.ndarray) -> np.ndarray:
    deviation = returns.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe = returns.mean(axis=1) / deviation * math.sqrt(TRADING_DAYS_PER_YEAR)
    return np.where(deviation > 1e-12, sharpe, np.nan)


def bootstrap(
    returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    *,
    samples: int = 2000,
    block_length: int = 20,
    confidence: float = 0.95,
    seed: int = 42,
) -> BootstrapResult:
    """Estimate confidence intervals with a circular block bootstrap.

    Resampling whole blocks rather than single days preserves the volatility clustering and the serial
    dependence that a long/flat strategy's returns have. Strategy and benchmark share the same resampled
    dates, so their comparison reflects returns that actually occurred together.
    """
    values = returns.dropna().to_numpy(dtype=float)
    if len(values) < 2 * block_length:
        raise ValueError(f"Bootstrapping needs at least {2 * block_length} returns.")

    rng = np.random.default_rng(seed)
    indices = _block_indices(len(values), samples, block_length, rng)
    paths = values[indices]
    sharpes = _sharpe_by_row(paths)

    growth = np.cumprod(1 + paths, axis=1)
    annual_returns = growth[:, -1] ** (TRADING_DAYS_PER_YEAR / len(values)) - 1
    peaks = np.maximum(np.maximum.accumulate(growth, axis=1), 1.0)
    max_drawdowns = (growth / peaks - 1).min(axis=1)

    tail = (1 - confidence) / 2 * 100

    def interval(estimates: np.ndarray) -> Interval:
        low, median, high = np.nanpercentile(estimates, [tail, 50, 100 - tail])
        return Interval(float(low), float(median), float(high))

    beats_benchmark = float("nan")
    if benchmark_returns is not None:
        aligned = benchmark_returns.reindex(returns.dropna().index).fillna(0.0).to_numpy(dtype=float)
        difference = sharpes - _sharpe_by_row(aligned[indices])
        beats_benchmark = float(np.nanmean(difference > 0))

    return BootstrapResult(
        sharpe=interval(sharpes),
        annual_return=interval(annual_returns),
        max_drawdown=interval(max_drawdowns),
        probability_positive_sharpe=float(np.nanmean(sharpes > 0)),
        probability_beats_benchmark=beats_benchmark,
        sharpe_samples=sharpes,
    )


@dataclass(frozen=True)
class OverfittingResult:
    """Combinatorially symmetric cross-validation of a parameter search.

    Each split chooses the configuration with the best in-sample Sharpe ratio and records where it ranks
    out of sample. ``probability`` is the share of splits in which it ranked at or below the median, so
    0.5 is what choosing at random would give and values near 0 mean the selection carried over.
    """

    probability: float
    logits: np.ndarray
    in_sample_sharpe: np.ndarray
    out_of_sample_sharpe: np.ndarray
    partitions: int

    @property
    def probability_of_loss(self) -> float:
        """Share of splits in which the selected configuration lost money relative to cash."""
        return float(np.mean(self.out_of_sample_sharpe < 0))


def _sharpe_from_sums(total: np.ndarray, squares: np.ndarray, count: np.ndarray) -> np.ndarray:
    """Annualized Sharpe ratios from per-configuration sums of returns and of squared returns."""
    count = count[:, None]
    mean = total / count
    variance = (squares - count * mean**2) / (count - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe = mean / np.sqrt(variance) * math.sqrt(TRADING_DAYS_PER_YEAR)
    return np.where(variance > 1e-18, sharpe, np.nan)


def probability_of_backtest_overfitting(returns: pd.DataFrame, partitions: int = 16) -> OverfittingResult:
    """Estimate the probability of backtest overfitting by combinatorially symmetric cross-validation.

    ``returns`` has one column of per-period (excess) returns for each configuration tried, on shared
    dates. History is cut into ``partitions`` contiguous blocks, and every way of choosing half of them
    as the in-sample set is evaluated, with the other half as the out-of-sample set. Because every block
    is used both ways, the method needs no arbitrary train/test boundary.
    """
    if partitions < 2 or partitions % 2:
        raise ValueError("Partitions must be an even number of at least 2.")
    matrix = returns.to_numpy(dtype=float)
    observations, configurations = matrix.shape
    if configurations < 2:
        raise ValueError("Overfitting needs at least two configurations to compare.")
    if observations < 4 * partitions:
        raise ValueError(f"Overfitting needs at least {4 * partitions} returns for {partitions} partitions.")
    matrix = np.nan_to_num(matrix, nan=0.0)

    blocks = np.array_split(np.arange(observations), partitions)
    block_sums = np.stack([matrix[block].sum(axis=0) for block in blocks])
    block_squares = np.stack([(matrix[block] ** 2).sum(axis=0) for block in blocks])
    block_counts = np.array([len(block) for block in blocks], dtype=float)

    chosen = np.zeros((math.comb(partitions, partitions // 2), partitions))
    for row, combination in enumerate(combinations(range(partitions), partitions // 2)):
        chosen[row, list(combination)] = 1.0
    others = 1.0 - chosen

    in_sample = _sharpe_from_sums(chosen @ block_sums, chosen @ block_squares, chosen @ block_counts)
    out_of_sample = _sharpe_from_sums(others @ block_sums, others @ block_squares, others @ block_counts)

    # An undefined Sharpe ratio (a configuration that never traded) ranks last on either side.
    best = np.argmax(np.where(np.isnan(in_sample), -np.inf, in_sample), axis=1)
    rows = np.arange(len(best))
    scores = np.where(np.isnan(out_of_sample), -np.inf, out_of_sample)
    selected = scores[rows, best][:, None]
    below = (scores < selected).sum(axis=1)
    tied = (scores == selected).sum(axis=1)
    rank = below + (tied + 1) / 2  # Average rank from 1 (worst) to N (best), sharing ties.
    relative_rank = rank / (configurations + 1)
    logits = np.log(relative_rank / (1 - relative_rank))

    return OverfittingResult(
        probability=float(np.mean(logits <= 0)),
        logits=logits,
        in_sample_sharpe=in_sample[rows, best],
        out_of_sample_sharpe=out_of_sample[rows, best],
        partitions=partitions,
    )
