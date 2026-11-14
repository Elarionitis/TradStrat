"""Headless tests of the dashboard, driving the real app through Streamlit's test runner."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import pytest
from streamlit.proto.NewSession_pb2 import CustomThemeConfig
from streamlit.runtime.theme_util import parse_fonts_with_source
from streamlit.testing.v1 import AppTest

import tradstrat
import tradstrat.charts as charts_module
import tradstrat.data as data_module
from tradstrat.data import load_sample
from tradstrat.strategies import STRATEGIES

APP = str(Path(__file__).resolve().parents[1] / "app.py")
TABS = ["Results", "Stress test", "Trades", "Tune", "Compare", "Guide"]
QUESTION_KEYS = [
    "golden_cross",
    "bitcoin_trend",
    "nasdaq_dip",
    "momentum_crash",
    "gold_breakout",
    "nvidia_macd",
]


@pytest.fixture
def app() -> AppTest:
    """The dashboard running on the bundled dataset, so no test touches the network."""
    test = AppTest.from_file(APP, default_timeout=90)
    test.session_state["source"] = "Sample"
    test.session_state["started"] = True
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


@pytest.mark.parametrize("stamped", [True, False])
def test_a_package_changed_since_import_is_reloaded(monkeypatch: pytest.MonkeyPatch, stamped: bool) -> None:
    """Regression guard: Streamlit Cloud pulls new code without a restart, which once broke the live site.

    A package imported before the dashboard stamped its import time has no stamp at all.
    """
    loaded = {name: module for name, module in sys.modules.items() if name.partition(".")[0] == "tradstrat"}
    if stamped:
        monkeypatch.setattr(tradstrat, "LOADED_AT", 0.0)
    else:
        monkeypatch.delattr(tradstrat, "LOADED_AT")
    monkeypatch.delattr(charts_module, "crashes_figure")
    try:
        test = AppTest.from_file(APP, default_timeout=90)
        test.session_state["source"] = "Sample"
        test.session_state["started"] = True
        test.run()

        assert not test.exception
        assert sys.modules["tradstrat"] is not tradstrat
    finally:
        sys.modules.update(loaded)


def test_results_lead_with_a_verdict_against_buy_and_hold(app: AppTest) -> None:
    rendered = " ".join(str(element.proto) for element in app.get("html"))

    assert "Did the rule beat just holding" in rendered
    for contest in ("Growth per year", "Return for the risk", "Worst crash", "Could it be luck?"):
        assert contest in rendered
    assert "became" in rendered


def test_results_never_use_alarm_boxes(app: AppTest) -> None:
    """A result is information, not an error, so it never appears as a red or yellow alert."""
    for key in ("ma_crossover", "rsi", "macd"):
        app.sidebar.selectbox[0].set_value(key).run()

        assert not app.error
        assert not app.warning


@pytest.mark.parametrize(
    "key", ["ma_crossover", "mean_reversion", "rsi", "donchian", "momentum", "macd", "trend_filter"]
)
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
    assert not app.error
    assert "Upload a CSV file" in rendered(app)


def test_the_rule_being_tested_is_stated_in_plain_english(app: AppTest) -> None:
    rendered = " ".join(str(element.proto) for element in app.get("html"))

    assert "50-day average price is above its 200-day average price" in rendered


def test_jargon_explains_itself_on_hover(app: AppTest) -> None:
    rendered = " ".join(element.proto.body for element in app.get("html"))

    assert 'class="term"' in rendered
    assert 'data-tip="Return earned for each unit of risk' in rendered


def landing() -> AppTest:
    test = AppTest.from_file(APP, default_timeout=90)
    test.run()
    assert not test.exception
    return test


def test_newcomers_land_on_questions_not_a_dashboard() -> None:
    test = landing()

    assert "Start with a question" in rendered(test)
    assert not test.tabs
    assert not test.error
    assert not test.warning
    assert {item.key for item in test.button} >= {f"ask_{key}" for key in QUESTION_KEYS}


def test_a_question_runs_its_test_on_market_data(monkeypatch: pytest.MonkeyPatch) -> None:
    sample = load_sample()
    monkeypatch.setattr(data_module, "PROVIDERS", (("stub", lambda *_: sample),))
    test = landing()

    next(item for item in test.button if item.key == "ask_bitcoin_trend").click().run()

    assert not test.exception
    assert not test.error
    assert test.session_state["ticker"] == "BTC-USD"
    assert test.session_state["strategy"] == "trend_filter"
    assert [tab.label for tab in test.tabs] == TABS
    assert test.query_params["ticker"] == "BTC-USD"


def test_the_sidebar_route_and_ideas_button_switch_between_views() -> None:
    test = landing()

    button(test, "Or set up your own test in the sidebar").click().run()
    assert [tab.label for tab in test.tabs] == TABS

    button(test, "All questions").click().run()
    assert "Start with a question" in rendered(test)
    assert not test.tabs


def test_every_strategy_is_explained_in_the_guide(app: AppTest) -> None:
    rendered = " ".join(str(element.proto) for element in app.get("html"))

    for phrase in ("Buys when", "Sells when", "Tends to struggle in"):
        assert phrase in rendered
    assert len([item for item in app.expander if "·" in item.label]) == len(STRATEGIES)


def rendered(app: AppTest) -> str:
    return " ".join(str(element.proto) for element in app.get("html"))


def test_the_benchmark_and_latest_signal_are_explained(app: AppTest) -> None:
    text = rendered(app)

    assert "buying Demo data on the first day and keeping it" in text
    assert "Latest signal" in text
    assert "Hold cash" in text or "Own Demo data" in text


def test_a_custom_rule_runs_through_every_tab(app: AppTest) -> None:
    app.sidebar.selectbox[0].set_value("custom").run()

    assert not app.exception
    assert not app.error
    text = rendered(app)
    assert "50-day average price crosses above its 200-day average price" in text
    tables = [frame.value for frame in app.dataframe if "Style" in frame.value.columns]
    assert "Your own rule" in tables[0]["Strategy"].tolist()


def test_an_example_fills_in_the_rule(app: AppTest) -> None:
    app.sidebar.selectbox[0].set_value("custom").run()
    next(pills for pills in app.get("button_group") if "Dip in an uptrend" in str(pills.proto))
    app.session_state["rule_example"] = "Dip in an uptrend"
    app.session_state["rule_buy"] = "rsi(2) < 10 and close > sma(200)"
    app.session_state["rule_sell"] = "rsi(2) > 70 or close < sma(200)"
    app.run()

    assert not app.exception
    assert "is below 10 and its price is above its 200-day average price" in rendered(app)


def test_an_unreadable_rule_shows_how_to_fix_it(app: AppTest) -> None:
    app.sidebar.selectbox[0].set_value("custom").run()
    app.sidebar.text_input(key="rule_buy").set_value("close above sma(20)").run()

    assert not app.exception
    assert any("Could not read" in item.value for item in app.error)
    assert "What a rule can look at" in rendered(app)


def test_a_custom_rule_can_be_tuned(app: AppTest) -> None:
    app.sidebar.selectbox[0].set_value("custom").run()
    app.tabs[3].slider[0].set_value(3)
    button(app, "Run search").click().run()

    assert not app.exception
    assert "Chance it is overfit" in metric_labels(app)
    assert "buy when sma(" in rendered(app)


def test_a_shared_link_restores_the_test() -> None:
    test = AppTest.from_file(APP, default_timeout=90)
    test.query_params.update(
        {"source": "Sample", "strategy": "custom", "buy": "close > sma(100)", "sell": ""}
    )
    test.run()

    assert not test.exception
    assert test.session_state["strategy"] == "custom"
    assert "100-day average price" in rendered(test)


def test_a_shared_link_ignores_invalid_values() -> None:
    test = AppTest.from_file(APP, default_timeout=90)
    test.query_params.update({"source": "Sample", "strategy": "rsi", "window": "9999", "history": "nope"})
    test.run()

    assert not test.exception
    assert test.session_state["rsi.window"] == 14


def test_the_link_follows_the_settings(app: AppTest) -> None:
    app.sidebar.selectbox[0].set_value("rsi").run()

    assert app.query_params["strategy"] == "rsi"
    assert app.query_params["source"] == "Sample"


def test_rule_text_from_a_link_cannot_inject_markdown() -> None:
    test = AppTest.from_file(APP, default_timeout=90)
    test.query_params.update({"source": "Sample", "strategy": "custom", "buy": "[x](http://example.com) > 3"})
    test.run()

    assert not test.exception
    messages = [item.value for item in test.error]
    assert messages
    assert all("[x](http" not in message for message in messages)


def test_a_failed_download_is_explained_calmly_and_offers_the_demo(monkeypatch: pytest.MonkeyPatch) -> None:
    """A visitor who mistypes a ticker sees what to do next, not a red box of provider internals."""

    def offline(*_: object) -> None:
        raise ConnectionError("rate limited")

    monkeypatch.setattr(data_module, "PROVIDERS", (("stub", offline),))
    test = AppTest.from_file(APP, default_timeout=90)
    test.session_state["started"] = True
    test.session_state["ticker"] = "NOPE"
    test.run()

    assert not test.exception
    assert not test.error
    body = " ".join(element.proto.body for element in test.get("html"))
    assert "couldn't load prices for NOPE" in body
    assert "rate limited" not in body

    button(test, "Use demo data instead").click().run()

    assert [tab.label for tab in test.tabs] == TABS


def test_the_theme_config_is_one_streamlit_accepts() -> None:
    """Regression: a font URL naming two families made Streamlit silently drop the whole theme."""
    theme = tomllib.loads((Path(APP).parent / ".streamlit" / "config.toml").read_text())["theme"]

    parse_fonts_with_source(
        CustomThemeConfig(), theme.get("font"), theme.get("codeFont"), theme.get("headingFont"), "theme"
    )


def test_newcomers_can_watch_the_tour_from_the_landing_page() -> None:
    test = landing()

    next(item for item in test.button if item.key == "watch_tour").click().run()

    assert not test.exception
    assert test.get("video")


def test_the_stress_test_answers_six_questions(app: AppTest) -> None:
    rendered = " ".join(str(element.proto) for element in app.get("html"))

    for question in (
        "Was the market picked with hindsight?",
        "Are the fills realistic?",
        "Are all the costs included?",
        "Does it depend on the start date?",
        "best trades?",
        "Did it survive a real bear market?",
    ):
        assert question in rendered
    assert "stress tests." in rendered
    assert not app.exception


def test_results_summarize_the_stress_test(app: AppTest) -> None:
    rendered = " ".join(str(element.proto) for element in app.get("html"))

    assert "Stress test" in rendered
    assert "Costs, start dates, best trades and crashes" in rendered
