"""Parameter search and walk-forward validation.

Tuning parameters on a price history and then reporting performance on that same history measures how well
the tuning fitted the noise. Walk-forward validation removes that bias. It repeatedly picks parameters on
one window and then trades them on the following window, which played no part in the choice. Joining those
unseen windows gives an equity curve made only of decisions that could have been taken at the time.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from typing import Any, Final

import pandas as pd

from tradstrat.engine import BacktestConfig, BacktestResult, run_backtest
from tradstrat.metrics import Metrics, calculate_metrics, daily_returns, sharpe_ratio
from tradstrat.stats import OverfittingResult, deflated_sharpe_ratio, probability_of_backtest_overfitting
from tradstrat.strategies import STRATEGIES, BuyAndHoldStrategy, create_strategy

OBJECTIVES: Final[tuple[str, ...]] = (
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "cagr",
    "total_return",
)
Params = dict[str, Any]


def build_grid(key: str, points: int = 5, overrides: dict[str, list[float]] | None = None) -> list[Params]:
    """Build every valid parameter combination for a strategy.

    Each parameter spans its declared range unless ``overrides`` lists explicit values. Combinations the
    strategy rejects, such as a short window above the long window, are left out.
    """
    if key not in STRATEGIES:
        raise ValueError(f"Unknown strategy {key!r}.")
    overrides = overrides or {}
    axes = {spec.name: overrides.get(spec.name, spec.grid(points)) for spec in STRATEGIES[key].params}

    grid: list[Params] = []
    for values in product(*axes.values()):
        params = dict(zip(axes, values, strict=True))
        try:
            create_strategy(key, **params)
        except ValueError:
            continue
        grid.append(params)
    return grid


def _check_objective(objective: str) -> None:
    if objective not in OBJECTIVES:
        raise ValueError(f"Unknown objective {objective!r}. Choose from: {', '.join(OBJECTIVES)}.")


def _benchmark_returns(data: pd.DataFrame, config: BacktestConfig) -> pd.Series:
    """Daily returns of holding the asset over ``data``, paying the same costs as a strategy."""
    result = run_backtest(data, BuyAndHoldStrategy(), config.for_benchmark(), include_benchmark=False)
    return daily_returns(result.results["Strategy Equity"])


@dataclass(frozen=True)
class _Scores:
    """A ranked score table, plus each combination's excess returns in grid order."""

    table: pd.DataFrame
    returns: pd.DataFrame


def _score_grid(
    data: pd.DataFrame, key: str, grid: list[Params], config: BacktestConfig, objective: str
) -> _Scores:
    """Backtest each combination and rank them best first, with undefined scores last."""
    benchmark_returns = _benchmark_returns(data, config)
    rows = []
    returns = {}
    for number, params in enumerate(grid):
        result = run_backtest(data, create_strategy(key, **params), config, include_benchmark=False)
        metrics = result.metrics
        active_returns = result.returns - benchmark_returns
        returns[number] = result.excess_returns
        rows.append(
            {
                **params,
                "sharpe_ratio": metrics.sharpe_ratio,
                "sortino_ratio": metrics.sortino_ratio,
                "calmar_ratio": metrics.calmar_ratio,
                "cagr": metrics.cagr,
                "total_return": metrics.total_return,
                "max_drawdown": metrics.max_drawdown,
                "number_of_trades": metrics.number_of_trades,
                "active_sharpe": sharpe_ratio(active_returns),
            }
        )
    table = pd.DataFrame(rows).sort_values(objective, ascending=False, na_position="last", kind="stable")
    return _Scores(table.reset_index(drop=True), pd.DataFrame(returns))


def _row_params(row: pd.Series, grid: list[Params]) -> Params:
    """Read a table row's parameters back as their original types, since pandas may widen them."""
    names = list(grid[0])
    return {name: type(grid[0][name])(row[name]) for name in names}


@dataclass(frozen=True)
class SearchResult:
    """Every configuration tried, the winner, and how far the winner can be trusted.

    ``deflated_sharpe`` asks whether the winner made money for reasons beyond selection luck.
    ``deflated_active_sharpe`` asks the harder question of whether it beat simply holding the asset. A
    long-only strategy on a rising market passes the first test by owning the market, so the second is
    the one that speaks to timing skill. ``overfitting`` estimates how often the search's choice would
    have fallen to the bottom half on unseen data; it is ``None`` when history is too short to split.
    """

    table: pd.DataFrame
    objective: str
    best_params: Params
    best: BacktestResult
    deflated_sharpe: float
    deflated_active_sharpe: float
    overfitting: OverfittingResult | None = None


def grid_search(
    data: pd.DataFrame,
    key: str,
    grid: list[Params] | None = None,
    config: BacktestConfig | None = None,
    objective: str = "sharpe_ratio",
    *,
    partitions: int = 16,
) -> SearchResult:
    """Evaluate a parameter grid. The winner's score is in-sample, so ``deflated_sharpe`` discounts it.

    ``partitions`` sets how finely history is cut to estimate the probability of backtest overfitting.
    """
    _check_objective(objective)
    config = config or BacktestConfig()
    grid = grid if grid is not None else build_grid(key)
    if not grid:
        raise ValueError("The parameter grid is empty.")

    scores = _score_grid(data, key, grid, config, objective)
    table = scores.table
    best_params = _row_params(table.iloc[0], grid)
    best = run_backtest(data, create_strategy(key, **best_params), config)
    active_returns = best.returns - best.benchmark_returns

    overfitting = None
    if len(grid) >= 2:
        try:
            overfitting = probability_of_backtest_overfitting(scores.returns, partitions)
        except ValueError:
            overfitting = None  # Too little history for this many partitions.

    return SearchResult(
        table=table,
        objective=objective,
        best_params=best_params,
        best=best,
        deflated_sharpe=deflated_sharpe_ratio(best.excess_returns, table["sharpe_ratio"].tolist()),
        deflated_active_sharpe=deflated_sharpe_ratio(active_returns, table["active_sharpe"].tolist()),
        overfitting=overfitting,
    )


@dataclass(frozen=True)
class WalkForwardResult:
    """The joined out-of-sample results of a walk-forward run."""

    folds: pd.DataFrame
    equity: pd.Series
    benchmark: pd.Series
    trade_log: pd.DataFrame
    metrics: Metrics
    benchmark_metrics: Metrics
    objective: str

    @property
    def efficiency(self) -> float:
        """Mean out-of-sample score as a share of mean in-sample score. Near 1 means little overfitting."""
        in_sample = float(self.folds["In-Sample Score"].mean())
        if in_sample <= 0:
            return float("nan")
        return float(self.folds["Out-of-Sample Score"].mean()) / in_sample


def walk_forward(
    data: pd.DataFrame,
    key: str,
    grid: list[Params] | None = None,
    config: BacktestConfig | None = None,
    *,
    train_bars: int = 756,
    test_bars: int = 252,
    objective: str = "sharpe_ratio",
    anchored: bool = False,
) -> WalkForwardResult:
    """Optimize on a training window, trade the next window, then roll forward and repeat.

    With ``anchored`` the training window keeps its first bar and grows; otherwise it slides at a fixed
    length. Capital compounds from one fold into the next. Every fold starts flat and is liquidated at its
    end, which costs an extra round trip per fold and so errs slightly against the strategy.
    """
    _check_objective(objective)
    config = config or BacktestConfig()
    grid = grid if grid is not None else build_grid(key)
    if not grid:
        raise ValueError("The parameter grid is empty.")
    if train_bars < 2 or test_bars < 2:
        raise ValueError("Training and test windows must each have at least two bars.")
    if len(data) < train_bars + 2:
        raise ValueError(
            f"Walk-forward needs more than {train_bars} bars, but only {len(data)} are available."
        )
    longest_warmup = max(create_strategy(key, **params).warmup for params in grid)
    if train_bars <= longest_warmup:
        raise ValueError(
            f"The training window must exceed the longest indicator warm-up ({longest_warmup} bars)."
        )

    folds: list[dict[str, Any]] = []
    segments: list[BacktestResult] = []
    capital = config.initial_capital

    for test_start in range(train_bars, len(data) - 1, test_bars):
        train_start = 0 if anchored else test_start - train_bars
        test_end = min(test_start + test_bars, len(data))
        if test_end - test_start < 2:
            break

        scored = _score_grid(data.iloc[train_start:test_start], key, grid, config, objective).table
        params = _row_params(scored.iloc[0], grid)

        # Bars before test_start warm up the indicators but cannot be traded, and bars after test_end are
        # not passed in at all, so nothing from later in time can reach this fold.
        segment = run_backtest(
            data.iloc[train_start:test_end],
            create_strategy(key, **params),
            replace(config, initial_capital=capital),
            start=test_start - train_start,
            include_benchmark=False,
        )
        capital = float(segment.results["Strategy Equity"].iloc[-1])
        segments.append(segment)
        folds.append(
            {
                "Fold": len(folds) + 1,
                "Train Start": data.index[train_start],
                "Train End": data.index[test_start - 1],
                "Test Start": data.index[test_start],
                "Test End": data.index[test_end - 1],
                "Parameters": params,
                "In-Sample Score": float(scored.iloc[0][objective]),
                "Out-of-Sample Score": float(getattr(segment.metrics, objective)),
                "Test Return": segment.metrics.total_return,
                "Trades": segment.metrics.number_of_trades,
            }
        )

    # Each segment opens with a baseline row equal to the previous segment's final row, so drop the repeats.
    stitched = pd.concat([segments[0].results, *[segment.results.iloc[1:] for segment in segments[1:]]])
    equity = stitched["Strategy Equity"]
    trades = [segment.trade_log for segment in segments if not segment.trade_log.empty]
    trade_log = pd.concat(trades, ignore_index=True) if trades else segments[0].trade_log

    # One continuous holding over the same dates, so the benchmark is not charged a round trip per fold.
    benchmark = run_backtest(
        data, BuyAndHoldStrategy(), config.for_benchmark(), start=train_bars, include_benchmark=False
    ).results["Strategy Equity"]
    risk_free = stitched["Risk-Free Return"].iloc[1:]

    return WalkForwardResult(
        folds=pd.DataFrame(folds),
        equity=equity,
        benchmark=benchmark,
        trade_log=trade_log,
        metrics=calculate_metrics(equity, trade_log, stitched["Position"], risk_free=risk_free),
        benchmark_metrics=calculate_metrics(benchmark, pd.DataFrame(), risk_free=risk_free),
        objective=objective,
    )


@dataclass(frozen=True)
class Comparison:
    """Every strategy run with its default parameters on the same data, next to buy and hold."""

    table: pd.DataFrame
    equity: pd.DataFrame


def compare_strategies(
    data: pd.DataFrame, config: BacktestConfig | None = None, keys: list[str] | None = None
) -> Comparison:
    """Run each strategy with its declared defaults. No tuning happens, so nothing here is in-sample.

    Strategies whose indicators need more history than ``data`` holds are left out.
    """
    config = config or BacktestConfig()
    keys = keys if keys is not None else [key for key in STRATEGIES if key != BuyAndHoldStrategy.key]
    rows = []
    curves = {}
    for key in keys:
        strategy = create_strategy(key)
        if strategy.warmup >= len(data) - 1:
            continue
        result = run_backtest(data, strategy, config, include_benchmark=False)
        curves[STRATEGIES[key].label] = result.results["Strategy Equity"]
        rows.append({"Strategy": STRATEGIES[key].label, **result.metrics.as_dict()})

    if rows:
        # Run buy and hold as a strategy of its own, so its row has a trade count and exposure too.
        holding = run_backtest(data, BuyAndHoldStrategy(), config.for_benchmark(), include_benchmark=False)
        curves[BuyAndHoldStrategy.label] = holding.results["Strategy Equity"]
        rows.append({"Strategy": BuyAndHoldStrategy.label, **holding.metrics.as_dict()})
    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values("sharpe_ratio", ascending=False, na_position="last").reset_index(drop=True)
    return Comparison(table=table, equity=pd.DataFrame(curves))
