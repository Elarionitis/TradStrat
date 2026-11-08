"""Headless tests of the dashboard, driving the real app through Streamlit's test runner."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")
TABS = ["Results", "Trades", "Tune", "Compare", "Guide"]


@pytest.fixture
def app() -> AppTest:
    """The dashboard running on the bundled dataset, so no test touches the network."""
    test = AppTest.from_file(APP, default_timeout=90)
    test.session_state["source"] = "Sample"
    test.run()
    assert not test.exception
    return test


def button(app: AppTest, label: str):
    return next(item for item in app.button if item.label == label)


def metric_labels(app: AppTest) -> set[str]:
    return {metric.label for metric in app.metric}


def test_first_load_renders_results_without_errors(app: AppTest) -> None:
    assert not app.error
    assert [tab.label for tab in app.tabs] == TABS
    assert metric_labels(app) >= {"Total return", "Annual return", "Sharpe ratio", "Max drawdown"}


def test_results_lead_with_a_verdict_against_buy_and_hold(app: AppTest) -> None:
    rendered = " ".join(str(element.proto) for element in app.get("html"))

    assert "Return." in rendered
    assert "Worst decline." in rendered
    assert "buy and hold" in rendered.lower()


@pytest.mark.parametrize("key", ["ma_crossover", "mean_reversion", "rsi", "donchian", "momentum"])
def test_each_strategy_renders(app: AppTest, key: str) -> None:
    app.sidebar.selectbox[0].set_value(key).run()

    assert not app.exception
    assert not app.error


def test_conflicting_windows_show_a_message_rather_than_crashing(app: AppTest) -> None:
    """Regression: dragging the short window past the long window raised an unhandled ValueError."""
    sliders = {slider.label: slider for slider in app.sidebar.slider}
    sliders["Long Window"].set_value(20)
    sliders["Short Window"].set_value(100).run()

    assert not app.exception
    assert "don't work together" in app.error[0].value


def test_results_survive_changing_an_unrelated_widget(app: AppTest) -> None:
    """Regression: results were gated on a button, so touching any widget wiped them."""
    {item.label: item for item in app.sidebar.number_input}["Slippage (bps)"].set_value(20.0).run()

    assert not app.exception
    assert "Sharpe ratio" in metric_labels(app)


def test_risk_controls_can_be_enabled(app: AppTest) -> None:
    {item.label: item for item in app.sidebar.toggle}["Stop loss"].set_value(True).run()
    toggles = {item.label: item for item in app.sidebar.toggle}
    toggles["Trail the stop"].set_value(True)
    toggles["Volatility targeting"].set_value(True).run()

    assert not app.exception
    assert not app.error


def test_a_fixed_cash_rate_is_credited_to_idle_cash(app: AppTest) -> None:
    app.sidebar.radio[0].set_value("Fixed rate").run()

    assert not app.exception
    interest = next(metric for metric in app.metric if metric.label == "Interest earned on idle cash")
    assert interest.value != "$0.00"


def test_search_results_persist_across_reruns(app: AppTest) -> None:
    button(app, "Run search").click().run()
    assert not app.exception
    assert {"Chance the edge is real", "Chance it is overfit"} <= metric_labels(app)

    app.sidebar.selectbox[0].set_value("ma_crossover").run()

    assert "Chance it is overfit" in metric_labels(app)


def test_search_results_are_cleared_when_the_settings_change(app: AppTest) -> None:
    """A search on one set of costs must not be shown as if it applied to another."""
    button(app, "Run search").click().run()

    {item.label: item for item in app.sidebar.number_input}["Slippage (bps)"].set_value(30.0).run()

    assert "Chance it is overfit" not in metric_labels(app)


def test_walk_forward_reports_out_of_sample_results(app: AppTest) -> None:
    button(app, "Run walk-forward").click().run()

    assert not app.exception
    assert "Sharpe on unseen data" in metric_labels(app)


def test_impossible_walk_forward_window_shows_an_error(app: AppTest) -> None:
    """A six-month training window cannot warm up a grid whose long average reaches 300 days."""
    {item.label: item for item in app.select_slider}["Choose settings using"].set_value("6 months").run()

    button(app, "Run walk-forward").click().run()

    assert not app.exception
    assert "warm-up" in app.error[0].value


def test_compare_tab_lists_every_strategy(app: AppTest) -> None:
    table = app.dataframe[-1].value

    assert set(table["Strategy"]) >= {"Buy and Hold", "RSI Reversal", "Donchian Breakout"}


def test_csv_source_asks_for_a_file_before_running(app: AppTest) -> None:
    app.sidebar.segmented_control[0].set_value("CSV").run()

    assert not app.exception
    assert "Upload a CSV file" in app.error[0].value


def test_the_rule_being_tested_is_stated_in_plain_english(app: AppTest) -> None:
    rendered = " ".join(str(element.proto) for element in app.get("html"))

    assert "50-day average price is above its 200-day average price" in rendered


def test_jargon_explains_itself_on_hover(app: AppTest) -> None:
    rendered = " ".join(element.proto.body for element in app.get("html"))

    assert 'class="term"' in rendered
    assert 'data-tip="Return earned for each unit of risk' in rendered


def test_newcomers_see_a_welcome_that_can_be_dismissed(app: AppTest) -> None:
    assert "New to backtesting?" in " ".join(str(element.proto) for element in app.get("html"))

    button(app, "Got it").click().run()

    assert "New to backtesting?" not in " ".join(str(element.proto) for element in app.get("html"))
    assert not app.exception


def test_every_strategy_is_explained_in_the_guide(app: AppTest) -> None:
    rendered = " ".join(str(element.proto) for element in app.get("html"))

    for phrase in ("Buys when", "Sells when", "Tends to struggle in"):
        assert phrase in rendered
    assert len([item for item in app.expander if "·" in item.label]) == 6
