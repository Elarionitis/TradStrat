"""Tests for the stress tests and their plain-English answers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import make_bars, random_walk

from tradstrat.engine import BacktestConfig, run_backtest
from tradstrat.reporting import (
    best_trades_check,
    cost_check,
    crash_check,
    start_date_check,
    stress_checks,
    stress_summary,
)
from tradstrat.strategies import BuyAndHoldStrategy, create_strategy
from tradstrat.stress import (
    MAX_WINDOWS,
    BestTradesResult,
    CostResult,
    Crash,
    StartDateResult,
    cost_test,
    crash_test,
    start_date_test,
    without_best_trades,
)

CONFIG = BacktestConfig()


def crossover() -> object:
    return create_strategy("ma_crossover", short_window=10, long_window=40)


# ---------- start dates ----------


def test_start_dates_use_windows_of_equal_length() -> None:
    data = random_walk(bars=1300)
    test = start_date_test(data, crossover(), CONFIG, years=2)

    assert test is not None
    assert test.years == 2
    assert len(test.windows) > 10
    lengths = (test.windows["End"] - test.windows["Start"]).dt.days
    assert lengths.between(2 * 365 - 7, 2 * 366 + 7).all()
    assert test.windows["Start"].is_monotonic_increasing
    assert 0 <= test.share_beating <= 1


def test_each_window_matches_a_backtest_started_on_that_day() -> None:
    """Regression guard: a window must be a real backtest of those dates, not something computed apart."""
    data = random_walk(bars=1300)
    strategy = crossover()
    test = start_date_test(data, strategy, CONFIG, years=2)
    assert test is not None
    row = test.windows.iloc[5]
    start = int(data.index.searchsorted(row["Start"]))
    end = int(data.index.searchsorted(row["End"], side="right"))

    direct = run_backtest(data.iloc[:end], strategy, CONFIG, start=start)
    equity = direct.results["Strategy Equity"]
    years = (equity.index[-1] - equity.index[0]).days / 365.25

    assert row["Rule"] == pytest.approx((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


def test_holding_never_counts_as_beating_itself() -> None:
    test = start_date_test(random_walk(bars=1300), BuyAndHoldStrategy(), CONFIG, years=2)

    assert test is not None
    assert test.share_beating == 0
    assert np.allclose(test.windows["Rule"], test.windows["Hold"])


def test_the_worst_start_is_the_lowest_rule_return() -> None:
    test = start_date_test(random_walk(bars=1300), crossover(), CONFIG, years=2)

    assert test is not None
    assert test.worst["Rule"] == test.windows["Rule"].min()
    assert test.best["Rule"] == test.windows["Rule"].max()


def test_a_short_history_cannot_compare_start_dates() -> None:
    assert start_date_test(random_walk(bars=400), crossover(), CONFIG) is None


def test_window_length_defaults_to_half_the_history_up_to_five_years() -> None:
    medium = start_date_test(random_walk(bars=1100), crossover(), CONFIG)
    long = start_date_test(random_walk(bars=4000), crossover(), CONFIG)

    assert medium is not None and medium.years == 2
    assert long is not None and long.years == 5


def test_long_histories_are_sampled_down_to_a_fixed_number_of_windows() -> None:
    test = start_date_test(random_walk(bars=8000), crossover(), CONFIG, years=1)

    assert test is not None
    assert len(test.windows) == MAX_WINDOWS


# ---------- best trades ----------


def test_removing_no_trades_changes_nothing() -> None:
    result = run_backtest(random_walk(), crossover(), CONFIG)
    test = without_best_trades(result, count=0)

    assert test is not None
    assert np.allclose(test.equity_without, result.results["Strategy Equity"])


def test_removing_the_best_trades_lowers_growth_and_leaves_other_days_alone() -> None:
    result = run_backtest(random_walk(seed=3), crossover(), CONFIG)
    test = without_best_trades(result, count=2)
    assert test is not None
    assert test.removed == 2
    assert test.cagr_without < test.cagr

    best = result.trade_log.nlargest(2, "Return")
    inside = np.zeros(len(result.returns), dtype=bool)
    for entry, exit_ in zip(best["Entry Date"], best["Exit Date"], strict=True):
        inside |= (result.returns.index >= entry) & (result.returns.index <= exit_)
    rebuilt = test.equity_without.pct_change().dropna()

    assert np.allclose(rebuilt[~inside], result.returns[~inside])
    assert np.allclose(rebuilt[inside], 0.0)


def test_removed_trades_earn_the_cash_rate_instead() -> None:
    result = run_backtest(random_walk(seed=3), crossover(), BacktestConfig(cash_rate=0.04))
    test = without_best_trades(result, count=1)
    assert test is not None
    trade = result.trade_log.nlargest(1, "Return").iloc[0]
    days = (result.returns.index >= trade["Entry Date"]) & (result.returns.index <= trade["Exit Date"])

    rebuilt = test.equity_without.pct_change().dropna()

    assert np.allclose(rebuilt[days], result.risk_free_returns[days])


def test_a_rule_that_never_trades_has_nothing_to_remove() -> None:
    flat = make_bars(np.full(120, 100.0))
    result = run_backtest(flat, create_strategy("ma_crossover", short_window=5, long_window=20), CONFIG)

    assert without_best_trades(result) is None


# ---------- crashes ----------


def crash_prices() -> pd.DataFrame:
    """Up 50%, down a third into March 2020, back to a new high, a 10% dip, then a 25% fall."""
    path = np.concatenate(
        [
            np.linspace(100, 150, 30),
            np.linspace(150, 100, 20)[1:],
            np.linspace(100, 160, 40)[1:],
            np.linspace(160, 144, 10)[1:],
            np.linspace(144, 170, 20)[1:],
            np.linspace(170, 127.5, 20)[1:],
        ]
    )
    return make_bars(path)


def test_crashes_are_the_deep_falls_in_date_order() -> None:
    result = run_backtest(crash_prices(), BuyAndHoldStrategy(), CONFIG)
    crashes = crash_test(result)

    assert len(crashes) == 2
    assert crashes[0].peak < crashes[0].bottom < crashes[1].peak < crashes[1].bottom
    assert crashes[0].hold == pytest.approx(-1 / 3, abs=0.01)
    assert crashes[1].hold == pytest.approx(-0.25, abs=0.01)


def test_well_known_crashes_get_their_names() -> None:
    result = run_backtest(crash_prices(), BuyAndHoldStrategy(), CONFIG)

    assert crash_test(result)[0].name == "Covid crash"


def test_a_rule_in_cash_during_a_crash_lost_less() -> None:
    data = crash_prices()
    result = run_backtest(data, create_strategy("ma_crossover", short_window=3, long_window=10), CONFIG)
    crash = crash_test(result)[0]

    assert crash.rule > crash.hold
    assert crash.rule_lost_less


def test_a_calm_history_has_no_crashes() -> None:
    calm = make_bars(np.linspace(100, 130, 300))
    result = run_backtest(calm, BuyAndHoldStrategy(), CONFIG)

    assert crash_test(result) == []


# ---------- costs ----------


def test_doubling_costs_lowers_growth_for_a_rule_that_trades() -> None:
    data = random_walk()
    strategy = crossover()
    result = run_backtest(data, strategy, CONFIG)
    costs = cost_test(data, strategy, result)

    assert costs.cagr == result.metrics.cagr
    assert costs.doubled_cagr < costs.cagr
    assert costs.doubled_hold_cagr < costs.hold_cagr
    assert not costs.free


def test_zero_costs_are_flagged() -> None:
    data = random_walk()
    config = BacktestConfig(commission_rate=0, slippage_bps=0)
    costs = cost_test(data, crossover(), run_backtest(data, crossover(), config))

    assert costs.free
    assert cost_check(costs).tone == "bad"


# ---------- plain-English answers ----------


def windows(beats: list[bool]) -> StartDateResult:
    starts = pd.date_range("2010-01-01", periods=len(beats), freq="MS")
    return StartDateResult(
        windows=pd.DataFrame(
            {
                "Start": starts,
                "End": starts + pd.DateOffset(years=3),
                "Rule": [0.08 if beat else 0.02 for beat in beats],
                "Hold": 0.05,
                "Beat": beats,
            }
        ),
        years=3,
    )


@pytest.mark.parametrize(
    ("beats", "tone"),
    [
        ([True] * 8 + [False] * 2, "good"),
        ([True] * 5 + [False] * 5, "warning"),
        ([True] + [False] * 9, "bad"),
    ],
)
def test_start_date_answers_follow_the_share_of_wins(beats: list[bool], tone: str) -> None:
    check = start_date_check(windows(beats))

    assert check.tone == tone
    assert f"{sum(beats) / len(beats):.0%}" in check.answer
    assert "January 2010" in check.answer or "2010" in check.answer


def test_start_date_answer_without_enough_history() -> None:
    assert start_date_check(None).tone == "neutral"


def best(cagr_without: float, trades: int = 20) -> BestTradesResult:
    return BestTradesResult(
        removed=5,
        trades=trades,
        cagr=0.10,
        cagr_without=cagr_without,
        hold_cagr=0.06,
        share_of_profit=0.8,
        equity_without=pd.Series(dtype=float),
    )


@pytest.mark.parametrize(("without", "tone"), [(0.07, "good"), (0.03, "warning"), (-0.01, "bad")])
def test_best_trade_answers(without: float, tone: str) -> None:
    check = best_trades_check(best(without))

    assert check.tone == tone
    assert "80% of its profit" in check.answer
    assert check.question == "Does it still work without its 5 best trades?"


def test_a_rule_with_only_a_handful_of_trades_is_flagged() -> None:
    check = best_trades_check(best(0.0, trades=3))

    assert check.status == "Too few trades"


def test_crash_answers_count_the_crashes_the_rule_softened() -> None:
    peak, bottom = pd.Timestamp("2007-10-09"), pd.Timestamp("2009-03-09")
    softened = Crash("2008 financial crisis", peak, bottom, hold=-0.55, rule=-0.2)
    rode_down = Crash("2022 bear market", peak, bottom, hold=-0.25, rule=-0.25)

    assert crash_check([softened], "SPY").tone == "good"
    assert crash_check([softened, rode_down], "SPY").tone == "warning"
    assert crash_check([rode_down], "SPY").tone == "bad"
    assert "2008 financial crisis: rule -20%, holding -55%" in crash_check([softened], "SPY").answer
    assert crash_check([], "SPY").status == "No crash in view"


def costs(cagr: float, doubled: float, hold: float = 0.06, doubled_hold: float = 0.059) -> CostResult:
    return CostResult(0.001, 5.0, cagr, hold, doubled, doubled_hold)


def test_cost_answers() -> None:
    assert cost_check(costs(0.08, 0.075)).tone == "good"
    assert cost_check(costs(0.08, 0.05)).status == "Fragile"
    assert cost_check(costs(0.05, 0.03)).status == "Costs bite"
    assert "0.10% commission and 5 basis points" in cost_check(costs(0.08, 0.075)).answer


def test_the_six_questions_come_in_order_and_are_summarized() -> None:
    checks = stress_checks("SPY", costs(0.08, 0.075), windows([True] * 9 + [False]), best(0.07), [])

    assert [check.question.split()[0] for check in checks] == ["Was", "Are", "Are", "Does", "Does", "Did"]
    assert not checks[0].computed and not checks[1].computed
    assert stress_summary(checks) == "Passed 3 of 4 stress tests."


def test_an_asset_name_that_opens_a_sentence_is_capitalized() -> None:
    assert crash_check([], "the synthetic sample").answer.startswith("The synthetic sample never fell")
