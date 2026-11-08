"""Tests for the backtesting engine: execution timing, costs, accounting and risk controls."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import pytest
from conftest import make_bars, random_walk
from hypothesis import given, settings
from hypothesis import strategies as st

from tradstrat.engine import BacktestConfig, run_backtest
from tradstrat.strategies import BuyAndHoldStrategy, MovingAverageCrossoverStrategy, Strategy

FRICTIONLESS = BacktestConfig(initial_capital=10_000.0, commission_rate=0.0, slippage_bps=0.0)


@dataclass(frozen=True)
class FixedSignal(Strategy):
    """A strategy that returns a predetermined signal, so tests can script exact trades."""

    pattern: tuple[int, ...]

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"signal": list(self.pattern)}, index=data.index)


def test_signal_fills_at_the_next_bars_open() -> None:
    data = make_bars(close=[10.0, 11, 12, 13, 14], open_=[10.0, 20, 25, 30, 35])

    trade = run_backtest(data, FixedSignal((1, 1, 0, 0, 0)), FRICTIONLESS).trade_log.iloc[0]

    assert trade["Entry Date"] == data.index[1]
    assert trade["Entry Price"] == 20.0
    assert trade["Exit Date"] == data.index[3]
    assert trade["Exit Price"] == 30.0
    assert trade["Exit Reason"] == "signal"


def test_equity_marks_to_the_close_while_holding() -> None:
    data = make_bars(close=[10.0, 12, 15, 15], open_=[10.0, 10, 12, 15])

    equity = run_backtest(data, FixedSignal((1, 1, 1, 1)), FRICTIONLESS).results["Strategy Equity"]

    assert equity.tolist() == [10_000.0, 12_000.0, 15_000.0, 15_000.0]


def test_commission_and_slippage_match_hand_calculation() -> None:
    data = make_bars(close=[100.0, 100, 110, 110], open_=[100.0, 100, 110, 110])
    config = BacktestConfig(initial_capital=10_000.0, commission_rate=0.01, slippage_bps=100.0)

    result = run_backtest(data, FixedSignal((1, 1, 0, 0)), config)
    trade = result.trade_log.iloc[0]

    buy_fill = 100 * 1.01
    size = int(10_000 / (buy_fill * 1.01))
    sell_fill = 110 * 0.99
    commission = size * buy_fill * 0.01 + size * sell_fill * 0.01
    expected_profit = (sell_fill - buy_fill) * size - commission

    assert trade["Size"] == size == 98
    assert trade["Entry Price"] == pytest.approx(buy_fill)
    assert trade["Exit Price"] == pytest.approx(sell_fill)
    assert trade["Commission"] == pytest.approx(commission)
    assert trade["P&L"] == pytest.approx(expected_profit)
    assert result.results["Strategy Equity"].iloc[-1] == pytest.approx(10_000 + expected_profit)


def test_slippage_always_hurts() -> None:
    data = random_walk(500)
    strategy = MovingAverageCrossoverStrategy(10, 40)

    cheap = run_backtest(data, strategy, BacktestConfig(commission_rate=0.0, slippage_bps=0.0))
    costly = run_backtest(data, strategy, BacktestConfig(commission_rate=0.0, slippage_bps=25.0))

    assert costly.results["Strategy Equity"].iloc[-1] < cheap.results["Strategy Equity"].iloc[-1]


def test_open_position_is_liquidated_on_the_last_bar() -> None:
    data = make_bars([10.0, 11, 12])

    result = run_backtest(data, FixedSignal((1, 1, 1)), FRICTIONLESS)

    assert result.trade_log.iloc[0]["Exit Reason"] == "end_of_data"
    assert result.results["Position"].iloc[-1] == 0


def test_position_fraction_limits_the_stake() -> None:
    data = make_bars([100.0, 100, 100, 100])

    result = run_backtest(data, FixedSignal((1, 1, 1, 1)), BacktestConfig(position_fraction=0.5))

    assert result.results["Position"].iloc[1] == pytest.approx(49, abs=1)


def test_no_signal_means_no_trades_and_flat_equity() -> None:
    data = make_bars([10.0, 11, 12, 13])

    result = run_backtest(data, FixedSignal((0, 0, 0, 0)), FRICTIONLESS)

    assert result.trade_log.empty
    assert list(result.trade_log.columns)[:2] == ["Entry Date", "Exit Date"]
    assert (result.results["Strategy Equity"] == 10_000.0).all()


def test_capital_too_small_for_one_share_never_trades() -> None:
    data = make_bars([500.0, 500, 500, 500])

    result = run_backtest(data, FixedSignal((1, 1, 1, 1)), BacktestConfig(initial_capital=100.0))

    assert result.trade_log.empty


def test_benchmark_equals_an_always_long_strategy() -> None:
    """The benchmark shares the simulator, so it must pay exactly what a strategy would pay."""
    data = random_walk(300)
    config = BacktestConfig(commission_rate=0.002, slippage_bps=10.0)

    result = run_backtest(data, BuyAndHoldStrategy(), config)

    np.testing.assert_allclose(result.results["Strategy Equity"], result.results["Buy and Hold Benchmark"])


def test_benchmark_is_unaffected_by_strategy_risk_settings() -> None:
    data = random_walk(300)
    plain = run_backtest(data, BuyAndHoldStrategy(), FRICTIONLESS)
    constrained = run_backtest(
        data,
        BuyAndHoldStrategy(),
        BacktestConfig(
            commission_rate=0.0,
            slippage_bps=0.0,
            position_fraction=0.3,
            stop_loss=0.01,
            target_volatility=0.05,
        ),
    )

    pd.testing.assert_series_equal(
        plain.results["Buy and Hold Benchmark"], constrained.results["Buy and Hold Benchmark"]
    )


def test_stop_fills_at_the_stop_price_on_an_intrabar_breach() -> None:
    data = make_bars(close=[100.0, 100, 100, 100], open_=[100.0, 100, 100, 100])
    data.loc[data.index[2], "Low"] = 80.0

    trade = run_backtest(data, FixedSignal((1, 1, 1, 1)), _with_stop(0.10)).trade_log.iloc[0]

    assert trade["Exit Reason"] == "stop_loss"
    assert trade["Exit Price"] == pytest.approx(90.0)
    assert trade["Exit Date"] == data.index[2]


def test_stop_fills_at_the_open_when_price_gaps_through_it() -> None:
    data = make_bars(close=[100.0, 100, 70, 70], open_=[100.0, 100, 70, 70])

    trade = run_backtest(data, FixedSignal((1, 1, 1, 1)), _with_stop(0.10)).trade_log.iloc[0]

    assert trade["Exit Price"] == pytest.approx(70.0)


def test_stop_can_trigger_on_the_entry_bar() -> None:
    """The fill happens at the open, so the bar's low comes after the position exists."""
    data = make_bars(close=[100.0, 100, 100], open_=[100.0, 100, 100])
    data.loc[data.index[1], "Low"] = 85.0

    trade = run_backtest(data, FixedSignal((1, 1, 1)), _with_stop(0.10)).trade_log.iloc[0]

    assert trade["Exit Date"] == data.index[1]


def test_no_reentry_after_a_stop_until_the_signal_resets() -> None:
    data = make_bars(close=[100.0] * 8, open_=[100.0] * 8)
    data.loc[data.index[2], "Low"] = 80.0

    result = run_backtest(data, FixedSignal((1, 1, 1, 1, 0, 1, 1, 1)), _with_stop(0.10))

    assert len(result.trade_log) == 2
    assert result.results["Position"].iloc[3:6].tolist() == [0, 0, 0]
    assert result.trade_log.iloc[1]["Entry Date"] == data.index[6]


def _with_stop(stop_loss: float) -> BacktestConfig:
    return BacktestConfig(commission_rate=0.0, slippage_bps=0.0, stop_loss=stop_loss)


def test_volatility_targeting_shrinks_positions_in_turbulent_markets() -> None:
    calm = random_walk(200, volatility=0.005)
    wild = random_walk(200, volatility=0.04)
    config = BacktestConfig(
        commission_rate=0.0, slippage_bps=0.0, target_volatility=0.10, volatility_lookback=20
    )

    def invested_share(data: pd.DataFrame) -> float:
        result = run_backtest(data, BuyAndHoldStrategy(), config)
        trade = result.trade_log.iloc[0]
        return trade["Size"] * trade["Entry Price"] / config.initial_capital

    assert invested_share(calm) > 0.9
    assert invested_share(wild) < 0.3


def test_volatility_targeting_waits_for_an_estimate() -> None:
    config = BacktestConfig(target_volatility=0.1, volatility_lookback=20)

    result = run_backtest(random_walk(100), BuyAndHoldStrategy(), config)

    assert (result.results["Position"].iloc[:20] == 0).all()


def test_start_skips_trading_but_keeps_indicators_warm() -> None:
    data = random_walk(300)
    strategy = MovingAverageCrossoverStrategy(10, 50)

    result = run_backtest(data, strategy, FRICTIONLESS, start=100)

    assert result.results["Signal"].iloc[1] == strategy.generate_signals(data)["signal"].iloc[100]
    assert result.trade_log.empty or result.trade_log["Entry Date"].min() >= data.index[100]


def test_start_opens_with_a_baseline_row_so_the_first_traded_bar_counts() -> None:
    """Regression: without a baseline, the return earned on the first traded bar was lost."""
    data = make_bars(close=[10.0, 10, 10, 20, 20], open_=[10.0, 10, 10, 10, 20])

    result = run_backtest(data, FixedSignal((1, 1, 1, 1, 1)), FRICTIONLESS, start=3)

    assert result.results.index[0] == data.index[2]
    assert result.results["Strategy Equity"].iloc[0] == FRICTIONLESS.initial_capital
    assert result.results["Position"].iloc[0] == 0
    assert result.metrics.total_return == pytest.approx(1.0)


def test_a_position_can_open_on_the_first_bar_after_warmup() -> None:
    """The signal from the last warm-up bar is known, so it is filled at the first traded open."""
    data = make_bars([10.0, 10, 10, 10, 10])

    result = run_backtest(data, FixedSignal((0, 0, 1, 1, 1)), FRICTIONLESS, start=3)

    assert result.trade_log.iloc[0]["Entry Date"] == data.index[3]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"initial_capital": 0.0},
        {"position_fraction": 0.0},
        {"position_fraction": 1.5},
        {"commission_rate": -0.01},
        {"slippage_bps": -1.0},
        {"stop_loss": 1.5},
        {"target_volatility": 0.0},
        {"volatility_lookback": 1},
    ],
)
def test_invalid_config_is_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        BacktestConfig(**kwargs)


def test_too_few_bars_are_rejected() -> None:
    with pytest.raises(ValueError, match="two bars"):
        run_backtest(make_bars([10.0]), BuyAndHoldStrategy())


def test_missing_columns_are_rejected() -> None:
    with pytest.raises(ValueError, match="Open"):
        run_backtest(make_bars([10.0, 11]).drop(columns="Open"), BuyAndHoldStrategy())


def test_a_misbehaving_strategy_is_caught() -> None:
    data = make_bars([10.0, 11, 12])

    with pytest.raises(ValueError, match="0 \\(flat\\) or 1"):
        run_backtest(data, FixedSignal((2, 0, 1)))


# Property-based tests: these invariants must hold for every price path and signal, not just examples.

scenarios = st.tuples(
    st.integers(min_value=0, max_value=10_000),  # price path seed
    st.integers(min_value=20, max_value=200),  # bars
    st.floats(min_value=0.0, max_value=0.02),  # commission
    st.floats(min_value=0.0, max_value=50.0),  # slippage
    st.one_of(st.none(), st.floats(min_value=0.01, max_value=0.5)),  # stop loss
    st.floats(min_value=0.1, max_value=1.0),  # position fraction
    st.booleans(),  # trailing stop, when there is a stop
    st.floats(min_value=0.0, max_value=0.1),  # annual cash rate
)


def _scenario(scenario: tuple) -> tuple[pd.DataFrame, FixedSignal, BacktestConfig]:
    seed, bars, commission, slippage, stop_loss, fraction, trailing, cash_rate = scenario
    data = random_walk(bars, seed=seed, volatility=0.03)
    signal = tuple(int(value) for value in np.random.default_rng(seed).integers(0, 2, bars))
    config = BacktestConfig(
        commission_rate=commission,
        slippage_bps=slippage,
        stop_loss=stop_loss,
        trailing_stop=trailing and stop_loss is not None,
        position_fraction=fraction,
        cash_rate=cash_rate,
    )
    return data, FixedSignal(signal), config


@settings(max_examples=200, deadline=None)
@given(scenarios)
def test_trade_profits_reconcile_exactly_with_the_change_in_equity(scenario: tuple) -> None:
    """Double-entry check: every dollar gained or lost must come from a closed trade or interest."""
    data, strategy, config = _scenario(scenario)

    result = run_backtest(data, strategy, config)

    change = result.results["Strategy Equity"].iloc[-1] - config.initial_capital
    explained = result.trade_log["P&L"].sum() + result.results["Interest"].sum()
    assert explained == pytest.approx(change, abs=1e-6)


@settings(max_examples=200, deadline=None)
@given(scenarios)
def test_cash_never_goes_negative_because_there_is_no_leverage(scenario: tuple) -> None:
    data, strategy, config = _scenario(scenario)

    result = run_backtest(data, strategy, config)

    assert (result.results["Cash"] >= -1e-9).all()
    assert (result.results["Position"] >= 0).all()


@settings(max_examples=100, deadline=None)
@given(scenarios, st.floats(min_value=0.2, max_value=0.9))
def test_future_bars_cannot_change_past_equity(scenario: tuple, cut_point: float) -> None:
    """Run on a prefix of the data: every bar except the liquidation bar must match exactly."""
    data, strategy, config = _scenario(scenario)
    cut = max(3, int(len(data) * cut_point))
    prefix_strategy = FixedSignal(strategy.pattern[:cut])

    full = run_backtest(data, strategy, config).results["Strategy Equity"].iloc[: cut - 1]
    prefix = run_backtest(data.iloc[:cut], prefix_strategy, config).results["Strategy Equity"].iloc[:-1]

    pd.testing.assert_series_equal(full, prefix)


@settings(max_examples=100, deadline=None)
@given(scenarios)
def test_every_stop_exit_respects_its_stop_level_unless_gapped(scenario: tuple) -> None:
    data, strategy, config = _scenario(scenario)

    trades = run_backtest(data, strategy, config).trade_log

    if config.stop_loss is not None:
        for trade in trades[trades["Exit Reason"] == "stop_loss"].itertuples():
            bar = data.loc[trade[2]]
            slippage = config.slippage_bps / 10_000
            stop_level = trade[3] * (1 - config.stop_loss)
            assert trade[4] <= stop_level * (1 - slippage) + 1e-9
            assert trade[4] >= min(bar["Open"], stop_level) * (1 - slippage) - 1e-9

        # A trailing stop only ever rises from the fixed level, so it can never fill worse than one.
        for trade in trades[trades["Exit Reason"] == "trailing_stop"].itertuples():
            bar = data.loc[trade[2]]
            slippage = config.slippage_bps / 10_000
            fixed_level = trade[3] * (1 - config.stop_loss)
            assert trade[4] >= min(bar["Open"], fixed_level) * (1 - slippage) - 1e-9


@settings(max_examples=100, deadline=None)
@given(scenarios)
def test_trades_never_overlap_and_run_forward_in_time(scenario: tuple) -> None:
    data, strategy, config = _scenario(scenario)

    trades = run_backtest(data, strategy, config).trade_log

    assert (trades["Exit Date"] >= trades["Entry Date"]).all()
    assert (trades["Entry Date"].iloc[1:].to_numpy() >= trades["Exit Date"].iloc[:-1].to_numpy()).all()


@settings(max_examples=100, deadline=None)
@given(scenarios, st.floats(min_value=0.05, max_value=0.5), st.floats(min_value=0.1, max_value=0.9))
def test_decisions_at_the_open_cannot_depend_on_that_bars_later_prices(
    scenario: tuple, target_volatility: float, location: float
) -> None:
    """Rewriting a bar's high, low and close must not change what was bought at its open.

    Truncating history cannot catch this bug, because a same-bar close is still inside the prefix.
    """
    data, strategy, config = _scenario(scenario)
    config = replace(
        config,
        target_volatility=target_volatility,
        stop_loss=None,
        trailing_stop=False,
        volatility_lookback=5,
    )
    bar = max(7, int(len(data) * location))
    altered = data.copy()
    altered.iloc[bar, altered.columns.get_loc("Close")] *= 1.5
    altered.iloc[bar, altered.columns.get_loc("High")] *= 1.5
    altered.iloc[bar, altered.columns.get_loc("Low")] *= 0.8

    original_position = run_backtest(data, strategy, config).results["Position"]
    altered_position = run_backtest(altered, strategy, config).results["Position"]

    if bar < len(data) - 1:  # The final bar is always liquidated.
        assert original_position.iloc[bar] == altered_position.iloc[bar]


# Cash interest and trailing stops.


def test_idle_cash_earns_the_cash_rate_over_calendar_days() -> None:
    data = make_bars([100.0] * 6)
    config = replace(FRICTIONLESS, cash_rate=0.0365)

    result = run_backtest(data, FixedSignal((0,) * 6), config)

    # Business days: Wednesday to the next Wednesday is 7 calendar days, including a weekend.
    expected = 10_000.0 * np.prod(1 + 0.0365 * (data.index[1:] - data.index[:-1]).days / 365.25)
    assert result.results["Strategy Equity"].iloc[-1] == pytest.approx(expected)
    assert result.results["Interest"].sum() == pytest.approx(expected - 10_000.0)


def test_a_cash_rate_column_overrides_the_fixed_rate_and_uses_the_prior_bars_rate() -> None:
    data = make_bars([100.0] * 3).assign(**{"Cash Rate": [0.10, 0.0, 0.50]})

    result = run_backtest(data, FixedSignal((0, 0, 0)), replace(FRICTIONLESS, cash_rate=0.3))

    days = (data.index[1:] - data.index[:-1]).days
    risk_free = result.results["Risk-Free Return"].to_numpy()
    assert risk_free[0] == 0.0
    assert risk_free[1] == pytest.approx(0.10 * days[0] / 365.25)
    assert risk_free[2] == 0.0  # The 50% rate on the last bar has not had time to accrue.


def test_sitting_in_cash_earns_the_risk_free_rate_but_no_sharpe() -> None:
    """Interest is compensation for time, not skill, so excess returns are zero and Sharpe is undefined."""
    data = random_walk(300)

    result = run_backtest(data, FixedSignal((0,) * 300), replace(FRICTIONLESS, cash_rate=0.05))

    assert result.metrics.total_return > 0.05
    assert result.excess_returns.abs().max() < 1e-12
    assert np.isnan(result.metrics.sharpe_ratio)


def test_cash_rate_lowers_the_benchmarks_sharpe_ratio() -> None:
    data = random_walk(400)

    plain = run_backtest(data, BuyAndHoldStrategy(), FRICTIONLESS)
    with_cash = run_backtest(data, BuyAndHoldStrategy(), replace(FRICTIONLESS, cash_rate=0.05))

    assert with_cash.benchmark_metrics.sharpe_ratio < plain.benchmark_metrics.sharpe_ratio


def test_trailing_stop_follows_the_highest_prior_high() -> None:
    # Enter at 100, rally to a high of 120, then fall. A 10% trail sits at 108; a fixed stop sits at 90.
    close = [100.0, 100, 110, 120, 115, 107, 100]
    data = make_bars(close)
    config = replace(FRICTIONLESS, stop_loss=0.10, trailing_stop=True)

    trade = run_backtest(data, FixedSignal((1,) * 7), config).trade_log.iloc[0]

    assert trade["Exit Reason"] == "trailing_stop"
    assert trade["Exit Date"] == data.index[5]
    assert trade["Exit Price"] == pytest.approx(min(data["Open"].iloc[5], 120 * 0.9))


def test_trailing_stop_ignores_the_current_bars_high() -> None:
    """A bar that sets a new high and then falls cannot be stopped by its own high."""
    data = make_bars([100.0, 100, 100])
    data.loc[data.index[2], ["High", "Low"]] = [130.0, 110.0]
    data.loc[data.index[2], ["Open", "Close"]] = [115.0, 115.0]

    trades = run_backtest(
        data, FixedSignal((1, 1, 1)), replace(FRICTIONLESS, stop_loss=0.10, trailing_stop=True)
    ).trade_log

    assert list(trades["Exit Reason"]) == ["end_of_data"]


@pytest.mark.parametrize("kwargs", [{"trailing_stop": True}, {"cash_rate": 0.9}, {"cash_rate": -0.2}])
def test_invalid_cash_and_trailing_settings_are_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        BacktestConfig(**kwargs)  # type: ignore[arg-type]
