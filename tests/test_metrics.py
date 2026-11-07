"""Unit tests for the metrics module, using small hand-checkable examples."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from tradstrat.metrics import (
    cagr,
    calculate_metrics,
    calculate_relative_metrics,
    max_drawdown_duration,
    sharpe_ratio,
    sortino_ratio,
)


def test_headline_metrics_match_hand_calculation() -> None:
    equity = pd.Series([100.0, 102.0, 101.0, 99.0])
    trades = pd.DataFrame({"P&L": [10.0, -5.0]})

    metrics = calculate_metrics(equity, trades)

    returns = np.array([0.02, 101 / 102 - 1, 99 / 101 - 1])
    downside = math.sqrt((returns.clip(max=0) ** 2).mean())
    assert metrics.total_return == pytest.approx(-0.01)
    assert metrics.cagr == pytest.approx(0.99 ** (252 / 3) - 1)
    assert metrics.annual_volatility == pytest.approx(returns.std(ddof=1) * math.sqrt(252))
    assert metrics.sharpe_ratio == pytest.approx(returns.mean() / returns.std(ddof=1) * math.sqrt(252))
    assert metrics.sortino_ratio == pytest.approx(returns.mean() / downside * math.sqrt(252))
    assert metrics.max_drawdown == pytest.approx(99 / 102 - 1)
    assert metrics.max_drawdown_duration == 2
    assert metrics.calmar_ratio == pytest.approx(metrics.cagr / abs(metrics.max_drawdown))


def test_trade_statistics() -> None:
    equity = pd.Series([100.0, 110.0])
    trades = pd.DataFrame({"P&L": [10.0, 30.0, -5.0, -15.0]})

    metrics = calculate_metrics(equity, trades)

    assert metrics.number_of_trades == 4
    assert metrics.win_rate == pytest.approx(0.5)
    assert metrics.average_win == pytest.approx(20.0)
    assert metrics.average_loss == pytest.approx(-10.0)
    assert metrics.profit_factor == pytest.approx(2.0)
    assert metrics.expectancy == pytest.approx(5.0)


def test_sortino_sees_risk_when_every_loss_is_identical() -> None:
    """Regression: the std of negative returns alone is zero here, which hid all downside risk."""
    returns = pd.Series([0.02, -0.01, 0.02, -0.01, 0.02, -0.01])

    expected_downside = math.sqrt((3 * 0.01**2) / 6)

    assert sortino_ratio(returns) == pytest.approx(returns.mean() / expected_downside * math.sqrt(252))


def test_sharpe_subtracts_risk_free_rate() -> None:
    returns = pd.Series([0.01, 0.02, -0.005, 0.015])

    assert sharpe_ratio(returns, risk_free_rate=0.05) < sharpe_ratio(returns)


def test_cagr_uses_calendar_time_when_dates_are_available() -> None:
    index = pd.to_datetime(["2020-01-01", "2022-01-01"])

    assert cagr(pd.Series([100.0, 121.0], index=index)) == pytest.approx(0.1, abs=1e-3)


def test_drawdown_duration_tracks_the_longest_underwater_run() -> None:
    assert max_drawdown_duration(pd.Series([100.0, 90, 95, 101, 100, 99, 98, 97, 102])) == 4
    assert max_drawdown_duration(pd.Series([100.0, 101, 102])) == 0


def test_flat_equity_has_undefined_ratios_rather_than_zero() -> None:
    metrics = calculate_metrics(pd.Series([100.0] * 5), pd.DataFrame(columns=["P&L"]))

    assert metrics.total_return == 0.0
    assert metrics.number_of_trades == 0
    for value in (metrics.sharpe_ratio, metrics.sortino_ratio, metrics.win_rate, metrics.profit_factor):
        assert math.isnan(value)


def test_exposure_is_share_of_bars_in_market() -> None:
    equity = pd.Series([100.0, 101.0, 102.0, 103.0])
    positions = pd.Series([0, 5, 5, 0])

    assert calculate_metrics(equity, pd.DataFrame(), positions).exposure == pytest.approx(0.5)


def test_single_point_curve_does_not_crash() -> None:
    metrics = calculate_metrics(pd.Series([100.0]), pd.DataFrame())

    assert metrics.total_return == 0.0
    assert math.isnan(metrics.sharpe_ratio)


def test_empty_curve_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        calculate_metrics(pd.Series(dtype=float), pd.DataFrame())


def test_relative_metrics_recover_a_known_beta_and_alpha() -> None:
    rng = np.random.default_rng(0)
    benchmark_returns = rng.normal(0.0005, 0.01, 1000)
    daily_alpha = 0.0002
    strategy_returns = daily_alpha + 0.5 * benchmark_returns

    relative = calculate_relative_metrics(
        pd.Series(np.cumprod(1 + strategy_returns)), pd.Series(np.cumprod(1 + benchmark_returns))
    )

    assert relative.beta == pytest.approx(0.5)
    assert relative.alpha == pytest.approx(daily_alpha * 252)
    assert relative.correlation == pytest.approx(1.0)


def test_relative_metrics_are_undefined_for_a_flat_strategy() -> None:
    relative = calculate_relative_metrics(pd.Series([1.0, 1.0, 1.0]), pd.Series([1.0, 1.1, 1.05]))

    assert math.isnan(relative.beta)
