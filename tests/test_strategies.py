"""Tests for the strategy registry and each strategy's signal logic."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import make_bars, random_walk
from hypothesis import given, settings
from hypothesis import strategies as st

from tradstrat.rules import compile_rule
from tradstrat.strategies import (
    STRATEGIES,
    BuyAndHoldStrategy,
    DonchianBreakoutStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    MovingAverageCrossoverStrategy,
    RsiStrategy,
    create_strategy,
    latest_signal,
    relative_strength_index,
)

ALL_KEYS = sorted(STRATEGIES)
SHORT_WINDOW_OVERRIDES: dict[str, dict[str, float]] = {
    "ma_crossover": {"short_window": 5, "long_window": 20},
    "mean_reversion": {"lookback_window": 10, "entry_threshold": 1.0},
    "rsi": {"window": 7, "oversold": 40.0, "exit_level": 55.0},
    "donchian": {"entry_window": 15, "exit_window": 5},
    "momentum": {"lookback_window": 20},
}


def make(key: str):
    return create_strategy(key, **SHORT_WINDOW_OVERRIDES.get(key, {}))


@pytest.mark.parametrize("key", ALL_KEYS)
@settings(max_examples=25, deadline=None)
@given(cut=st.integers(min_value=30, max_value=399), seed=st.integers(min_value=0, max_value=5))
def test_no_strategy_can_see_the_future(key: str, cut: int, seed: int) -> None:
    """Signals up to any bar must be identical whether or not later bars exist.

    This is the definition of being free of look-ahead bias, applied to every registered strategy.
    """
    data = random_walk(400, seed=seed)
    strategy = make(key)

    full = strategy.generate_signals(data)["signal"].iloc[:cut]
    truncated = strategy.generate_signals(data.iloc[:cut])["signal"]

    pd.testing.assert_series_equal(full, truncated)


@pytest.mark.parametrize("key", ALL_KEYS)
def test_signals_are_long_or_flat_and_aligned_with_the_data(key: str) -> None:
    data = random_walk(400)

    signals = make(key).generate_signals(data)

    assert signals.index.equals(data.index)
    assert set(signals["signal"].unique()) <= {0, 1}


@pytest.mark.parametrize("key", [key for key in ALL_KEYS if key != "buy_and_hold"])
def test_no_position_during_warmup(key: str) -> None:
    strategy = make(key)

    signals = strategy.generate_signals(random_walk(400))["signal"]

    assert (signals.iloc[: strategy.warmup - 1] == 0).all()


@pytest.mark.parametrize("key", [key for key in ALL_KEYS if key != "buy_and_hold"])
def test_strategies_actually_trade_on_a_random_walk(key: str) -> None:
    assert make(key).generate_signals(random_walk(600))["signal"].sum() > 0


@pytest.mark.parametrize("key", ALL_KEYS)
def test_declared_defaults_are_valid_and_inside_their_range(key: str) -> None:
    create_strategy(key)
    for spec in STRATEGIES[key].params:
        assert spec.low <= spec.default <= spec.high


def test_create_strategy_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="Unknown strategy"):
        create_strategy("astrology")
    with pytest.raises(ValueError, match="Unknown parameters"):
        create_strategy("rsi", moon_phase=1.0)


def test_create_strategy_coerces_integer_parameters() -> None:
    strategy = create_strategy("ma_crossover", short_window=10.0, long_window=30.0)

    assert isinstance(strategy, MovingAverageCrossoverStrategy)
    assert strategy.short_window == 10


def test_param_grid_stays_inside_the_declared_range() -> None:
    spec = STRATEGIES["ma_crossover"].params[0]

    grid = spec.grid(4)

    assert grid[0] == spec.low
    assert grid[-1] == spec.high
    assert all(isinstance(value, int) for value in grid)


@pytest.mark.parametrize(
    ("cls", "kwargs"),
    [
        (MovingAverageCrossoverStrategy, {"short_window": 100, "long_window": 20}),
        (MovingAverageCrossoverStrategy, {"short_window": 0, "long_window": 20}),
        (MeanReversionStrategy, {"lookback_window": 1}),
        (MeanReversionStrategy, {"entry_threshold": -1.0}),
        (MeanReversionStrategy, {"entry_threshold": 1.0, "exit_threshold": -1.5}),
        (RsiStrategy, {"oversold": 60.0, "exit_level": 50.0}),
        (DonchianBreakoutStrategy, {"entry_window": 10, "exit_window": 20}),
        (MomentumStrategy, {"lookback_window": 0}),
    ],
)
def test_invalid_parameters_fail_on_construction(cls: type, kwargs: dict[str, float]) -> None:
    """Regression: the app built strategies from sliders and crashed on short_window >= long_window."""
    with pytest.raises(ValueError):
        cls(**kwargs)


def test_crossover_is_long_exactly_when_short_average_is_above_long_average() -> None:
    data = make_bars([1, 2, 3, 4, 5, 4, 3, 2, 1, 1])

    signals = MovingAverageCrossoverStrategy(2, 4).generate_signals(data)

    assert signals["signal"].tolist() == [0, 0, 0, 1, 1, 1, 0, 0, 0, 0]


def test_mean_reversion_holds_through_an_undefined_z_score() -> None:
    """Regression: a flat window gives a NaN z-score, which silently dropped the open position."""
    close = [*np.linspace(100, 80, 25), *[80.0] * 25, 79.0, 81.0, 82.0]
    signals = MeanReversionStrategy(5, 0.5, 0.5).generate_signals(make_bars(close))

    flat_window = signals.iloc[30:48]

    assert flat_window["z_score"].isna().all()
    assert (flat_window["signal"] == 1).all()


def test_mean_reversion_enters_on_a_drop_and_exits_on_recovery() -> None:
    close = [100.0, 101, 99, 100, 101, 99, 100, 90, 92, 95, 100, 105]

    signal = MeanReversionStrategy(5, 1.5, 0.0).generate_signals(make_bars(close))["signal"].tolist()

    assert signal[6] == 0
    assert signal[7] == 1
    assert signal[-1] == 0


def test_rsi_matches_wilder_smoothing_by_hand() -> None:
    close = pd.Series([10.0, 11.0, 10.0, 12.0])

    # Changes are +1, -1, +2. With alpha = 1/2, the average gain runs 1 -> 0.5 -> 1.25 and the average
    # loss runs 0 -> 0.5 -> 0.25, so RS = 5 and RSI = 100 - 100 / 6.
    assert relative_strength_index(close, 2).iloc[-1] == pytest.approx(100 - 100 / 6)


def test_rsi_is_100_when_there_are_no_losses() -> None:
    assert relative_strength_index(pd.Series(np.arange(1.0, 30.0)), 14).iloc[-1] == pytest.approx(100.0)


def test_donchian_channel_excludes_the_current_bar() -> None:
    close = [10.0, 11, 12, 13, 12, 11, 10, 9, 8, 7, 8, 9, 14]

    signals = DonchianBreakoutStrategy(entry_window=5, exit_window=2).generate_signals(make_bars(close))

    assert signals["upper"].iloc[-1] == 9.0  # The max of the prior five bars, not the current 14.
    assert signals["signal"].iloc[-1] == 1


def test_momentum_follows_the_sign_of_the_trailing_return() -> None:
    signal = MomentumStrategy(2).generate_signals(make_bars([10.0, 11, 12, 11, 10, 9]))["signal"]

    assert signal.tolist() == [0, 0, 1, 0, 0, 0]


def test_latest_signal_reports_the_last_change() -> None:
    from tradstrat.strategies import latest_signal

    data = make_bars([5, 15, 15, 5, 15, 15])

    signal = latest_signal(create_strategy(compile_rule("close > 10")), data)

    assert signal.long
    assert signal.as_of == data.index[-1]
    assert signal.since == data.index[4]


def test_latest_signal_has_no_start_when_it_never_changed() -> None:
    signal = latest_signal(BuyAndHoldStrategy(), random_walk(50))

    assert signal.long
    assert signal.since is None
