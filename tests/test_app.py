"""Headless tests of the dashboard, driving the real app through Streamlit's test runner."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture
def app() -> AppTest:
    """The dashboard running on the bundled dataset, so no test touches the network."""
    test = AppTest.from_file(APP, default_timeout=60).run()
    test.sidebar.radio[0].set_value("Synthetic sample").run()
    assert not test.exception
    return test


def button(app: AppTest, label: str):
    return next(item for item in app.button if item.label == label)


def test_first_load_renders_results_without_errors(app: AppTest) -> None:
    assert not app.error
    assert [tab.label for tab in app.tabs] == [
        "Backtest",
        "Robustness",
        "Optimize",
        "Walk-Forward",
        "Glossary",
    ]
    assert {metric.label for metric in app.metric} >= {"Total Return", "CAGR", "Sharpe Ratio", "Max Drawdown"}


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
    app.sidebar.number_input[1].set_value(20.0).run()

    assert not app.exception
    assert any(metric.label == "Sharpe Ratio" for metric in app.metric)


def test_risk_controls_can_be_enabled(app: AppTest) -> None:
    checkboxes = {box.label: box for box in app.sidebar.checkbox}
    checkboxes["Stop loss"].check()
    checkboxes["Volatility targeting"].check().run()

    assert not app.exception
    assert not app.error


def test_search_results_persist_across_reruns(app: AppTest) -> None:
    button(app, "Run search").click().run()
    assert not app.exception
    assert any(metric.label == "Deflated Sharpe" for metric in app.metric)

    app.sidebar.number_input[1].set_value(20.0).run()

    assert any(metric.label == "Deflated Sharpe" for metric in app.metric)


def test_walk_forward_reports_out_of_sample_results(app: AppTest) -> None:
    button(app, "Run walk-forward").click().run()

    assert not app.exception
    assert any(metric.label == "Out-of-Sample Sharpe" for metric in app.metric)


def test_impossible_walk_forward_window_shows_an_error(app: AppTest) -> None:
    """A 100-bar training window cannot warm up a grid whose long average reaches 300 bars."""
    {item.label: item for item in app.number_input}["Training bars"].set_value(100).run()

    button(app, "Run walk-forward").click().run()

    assert not app.exception
    assert "warm-up" in app.error[0].value
