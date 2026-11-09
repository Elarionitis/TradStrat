"""Tests for custom rules: parsing, plain-English descriptions, equivalence with the built-in strategies,
freedom from look-ahead, and tuning through the same search and walk-forward as everything else."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import make_bars, random_walk
from hypothesis import given, settings
from hypothesis import strategies as st

from tradstrat.engine import run_backtest
from tradstrat.guide import guide_for
from tradstrat.rules import EXAMPLES, RuleError, RuleStrategy, compile_rule, parse_rule
from tradstrat.strategies import (
    DonchianBreakoutStrategy,
    MomentumStrategy,
    MovingAverageCrossoverStrategy,
    RsiStrategy,
    create_strategy,
)
from tradstrat.validation import build_grid, compare_strategies, grid_search, walk_forward


def signals(buy: str, sell: str, data: pd.DataFrame) -> pd.Series:
    return create_strategy(compile_rule(buy, sell)).generate_signals(data)["signal"]


@pytest.mark.parametrize(
    ("buy", "sell", "builtin"),
    [
        ("sma(50) > sma(200)", "", MovingAverageCrossoverStrategy(50, 200)),
        ("return(60) > 0", "", MomentumStrategy(60)),
        ("close > high(55)", "close < low(20)", DonchianBreakoutStrategy(55, 20)),
        ("rsi(14) < 30", "rsi(14) > 55", RsiStrategy(14, 30.0, 55.0)),
    ],
)
def test_a_rule_matches_the_builtin_strategy_it_describes(buy: str, sell: str, builtin: object) -> None:
    """Writing a built-in strategy as a rule must give exactly the same positions."""
    data = random_walk(800, seed=3)

    expected = builtin.generate_signals(data)["signal"]  # type: ignore[attr-defined]

    np.testing.assert_array_equal(signals(buy, sell, data).to_numpy(), expected.to_numpy())


@pytest.mark.parametrize("name", list(EXAMPLES))
@settings(max_examples=15, deadline=None)
@given(cut=st.integers(min_value=30, max_value=599), seed=st.integers(min_value=0, max_value=4))
def test_no_rule_can_see_the_future(name: str, cut: int, seed: int) -> None:
    """Signals up to any bar must be identical whether or not later bars exist."""
    data = random_walk(600, seed=seed)
    strategy = create_strategy(compile_rule(*EXAMPLES[name]))

    full = strategy.generate_signals(data)["signal"].iloc[:cut]
    truncated = strategy.generate_signals(data.iloc[:cut])["signal"]

    pd.testing.assert_series_equal(full, truncated)


@pytest.mark.parametrize("name", list(EXAMPLES))
def test_every_example_trades_and_stays_flat_during_warmup(name: str) -> None:
    strategy = create_strategy(compile_rule(*EXAMPLES[name]))
    data = random_walk(900, seed=1)

    assert run_backtest(data, strategy).metrics.number_of_trades > 0
    assert (strategy.generate_signals(data)["signal"].iloc[: strategy.warmup - 1] == 0).all()


def test_crosses_above_fires_only_on_the_day_of_the_cross() -> None:
    data = make_bars([10, 10, 10, 12, 13, 9, 9, 12, 12])
    cross = create_strategy(compile_rule("close crosses above 11", "close < 11")).generate_signals(data)
    level = create_strategy(compile_rule("close > 11", "close < 11")).generate_signals(data)

    np.testing.assert_array_equal(cross["signal"], [0, 0, 0, 1, 1, 0, 0, 1, 1])
    np.testing.assert_array_equal(level["signal"], cross["signal"])


def test_a_cross_needs_a_change_not_just_a_level() -> None:
    """A price that starts above the line has not crossed it, so a cross rule waits for a real one."""
    data = make_bars([12, 12, 10, 12])

    position = signals("close crosses above 11", "close crosses below 11", data)

    np.testing.assert_array_equal(position, [0, 0, 0, 1])


def test_without_a_sell_rule_the_position_lasts_while_the_buy_rule_holds() -> None:
    data = make_bars([5, 15, 15, 5, 15])

    np.testing.assert_array_equal(signals("close > 10", "", data), [0, 1, 1, 0, 1])


def test_or_needs_any_condition_and_and_needs_all() -> None:
    data = make_bars([5, 15, 25, 15, 5])

    np.testing.assert_array_equal(signals("close > 10 and close < 20", "", data), [0, 1, 0, 1, 0])
    np.testing.assert_array_equal(signals("close < 10 or close > 20", "", data), [1, 0, 1, 0, 1])


def test_a_reversed_comparison_is_flipped_rather_than_rejected() -> None:
    data = random_walk(300)

    pd.testing.assert_series_equal(
        signals("30 > rsi(2)", "70 < rsi(2)", data), signals("rsi(2) < 30", "rsi(2) > 70", data)
    )


def test_rules_are_case_and_space_insensitive_and_accept_aliases() -> None:
    data = random_walk(300)

    pd.testing.assert_series_equal(
        signals("  PRICE   Crosses  Above  MA(20) ", "price crosses below ma(20)", data),
        signals("close crosses above sma(20)", "close crosses below sma(20)", data),
    )


def test_return_is_in_percent() -> None:
    data = make_bars([100, 100, 104, 110])

    np.testing.assert_array_equal(signals("return(1) > 5%", "", data), [0, 0, 0, 1])


def test_the_same_indicator_is_one_setting() -> None:
    rule = compile_rule(*EXAMPLES["Golden cross"])

    assert [spec.name for spec in rule.params] == ["n1", "n2"]
    assert [spec.default for spec in rule.params] == [50, 200]


def test_setting_names_are_unique_even_for_equal_numbers() -> None:
    rule = compile_rule("rsi(2) < 10 and rsi(5) < 10")

    labels = [spec.label for spec in rule.params]
    assert len(labels) == len(set(labels))


def test_defaults_sit_inside_their_search_range() -> None:
    for buy, sell in EXAMPLES.values():
        for spec in compile_rule(buy, sell).params:
            assert spec.low <= spec.default <= spec.high


def test_numbers_can_be_overridden_like_any_parameter() -> None:
    rule = compile_rule(*EXAMPLES["Golden cross"])

    strategy = create_strategy(rule, n1=20, n2=100)

    assert isinstance(strategy, RuleStrategy)
    assert (
        strategy.text == "buy when sma(20) crosses above sma(100); sell when sma(20) crosses below sma(100)"
    )
    assert strategy.warmup == 101


def test_an_impossible_window_is_rejected_on_construction() -> None:
    with pytest.raises(ValueError, match="whole number"):
        create_strategy(compile_rule("close > sma(20)"), n1=1)


def test_the_rule_is_described_in_plain_english() -> None:
    rule = compile_rule("rsi(2) < 10 and close > sma(200)", "rsi(2) > 70")
    params = {spec.name: spec.default for spec in rule.params}

    sentence = guide_for(rule).describe_rule("SPY", params)

    assert sentence == (
        "Buy SPY when its 2-day RSI is below 10 and its price is above its 200-day average price, "
        "and sell when its 2-day RSI is above 70."
    )


def test_a_hold_while_rule_is_described_as_owning_while_true() -> None:
    rule = compile_rule("return(126) > 0")

    sentence = guide_for(rule).describe_rule("QQQ", {"n1": 126, "n2": 0.0})

    assert sentence == "Own QQQ while its 126-day return is above 0%, and hold cash otherwise."


def test_price_scale_indicators_are_drawn_once_each() -> None:
    rule = compile_rule("close > sma(200) and rsi(2) < 10", "close < sma(200)")
    strategy = create_strategy(rule)

    assert strategy.overlays == (("line0", "200-day average"),)
    assert "line0" in strategy.generate_signals(random_walk(300)).columns


@pytest.mark.parametrize(
    ("buy", "sell", "message"),
    [
        ("", "", "buy rule is empty"),
        ("smaa(5) > 3", "", "not something a rule can use"),
        ("close above 3", "", "Could not read"),
        ("5 > 3", "", "two numbers"),
        ("sma(1) > close", "", "from 2 to"),
        ("sma(x) > close", "", "whole number of days"),
        ("close > 1 and close < 5 or close > 9", "", "either 'and' or 'or'"),
        ("close crosses above sma(20)", "", "needs a sell rule"),
        ("close > sma(20)", "foo", "sell rule"),
    ],
)
def test_unreadable_rules_say_how_to_fix_them(buy: str, sell: str, message: str) -> None:
    with pytest.raises(RuleError, match=message):
        parse_rule(buy, sell)


def test_a_custom_rule_can_be_searched_and_walked_forward() -> None:
    data = random_walk(900, seed=2)
    rule = compile_rule("close > sma(50)")

    search = grid_search(data, rule, build_grid(rule, 3))
    forward = walk_forward(data, rule, build_grid(rule, 3), train_bars=300, test_bars=150)

    assert len(search.table) == 3
    assert set(search.best_params) == {"n1"}
    assert len(forward.folds) >= 3


def test_a_custom_rule_appears_in_the_comparison_by_its_label() -> None:
    comparison = compare_strategies(random_walk(500), keys=[compile_rule("close > sma(50)"), "momentum"])

    assert set(comparison.table["Strategy"]) == {"Your own rule", "Time-Series Momentum", "Buy and Hold"}
