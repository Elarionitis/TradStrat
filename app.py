"""Streamlit dashboard for TradStrat."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import streamlit as st

from tradstrat.charts import (
    bootstrap_figure,
    drawdown_figure,
    equity_figure,
    fold_scores_figure,
    monthly_returns_figure,
    price_figure,
    search_figure,
)
from tradstrat.data import DataError, fetch_ohlcv, load_sample, validate_ohlcv
from tradstrat.engine import BacktestConfig, BacktestResult, run_backtest
from tradstrat.glossary import GLOSSARY
from tradstrat.metrics import daily_returns, sharpe_ratio
from tradstrat.reporting import comparison_table, format_metric, probability_verdict
from tradstrat.stats import bootstrap, probabilistic_sharpe_ratio
from tradstrat.strategies import STRATEGIES, create_strategy
from tradstrat.validation import OBJECTIVES, build_grid, grid_search, walk_forward

STRATEGY_KEYS = [key for key in STRATEGIES if key != "buy_and_hold"]
HEADLINE_METRICS = ("total_return", "cagr", "sharpe_ratio", "max_drawdown")


@dataclass(frozen=True)
class Inputs:
    """Everything the sidebar collects."""

    source: str
    ticker: str
    start: pd.Timestamp
    end: pd.Timestamp
    strategy_key: str
    params: dict[str, float]
    config: BacktestConfig | None
    config_error: str | None


@st.cache_data(show_spinner="Fetching market data...", ttl=3600)
def cached_fetch(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return fetch_ohlcv(ticker, start, end)


def without_times(table: pd.DataFrame) -> pd.DataFrame:
    """Show datetime columns as plain dates, since every bar is a whole day."""
    dates = table.select_dtypes("datetime").columns
    return table.assign(**{column: table[column].dt.date for column in dates})


def param_widget(strategy_key: str, spec: Any) -> float:
    """Render one strategy parameter, using the range the strategy declares."""
    kind = int if spec.is_integer else float
    return st.slider(
        spec.label,
        min_value=kind(spec.low),
        max_value=kind(spec.high),
        value=kind(spec.default),
        step=kind(spec.step),
        help=spec.help,
        key=f"{strategy_key}.{spec.name}",
    )


def read_sidebar() -> Inputs:
    with st.sidebar:
        st.header("Data")
        source = st.radio("Source", ["Yahoo Finance", "Synthetic sample"], horizontal=True)
        ticker, start, end = "", pd.Timestamp.min, pd.Timestamp.max
        if source == "Yahoo Finance":
            ticker = st.text_input(
                "Ticker", value="SPY", help="A Yahoo Finance symbol, such as AAPL or BTC-USD."
            )
            today = pd.Timestamp.today().normalize()
            start = pd.Timestamp(st.date_input("Start", value=today - pd.DateOffset(years=10)))
            end = pd.Timestamp(st.date_input("End", value=today))

        st.header("Strategy")
        strategy_key = st.selectbox(
            "Strategy",
            STRATEGY_KEYS,
            format_func=lambda key: STRATEGIES[key].label,
            help="Choose the trading rule to test.",
        )
        st.caption(STRATEGIES[strategy_key].description)
        params = {spec.name: param_widget(strategy_key, spec) for spec in STRATEGIES[strategy_key].params}

        st.header("Costs and Risk")
        capital = st.number_input("Starting Capital", min_value=1_000.0, value=10_000.0, step=500.0)
        commission_bps = st.number_input(
            "Commission (bps)", min_value=0.0, max_value=100.0, value=10.0, step=1.0
        )
        slippage_bps = st.number_input(
            "Slippage (bps)", min_value=0.0, max_value=100.0, value=5.0, step=1.0, help=GLOSSARY["Slippage"]
        )
        position_fraction = st.slider("Position Size", 0.1, 1.0, 1.0, 0.1, help=GLOSSARY["Position Sizing"])
        use_stop = st.checkbox("Stop loss", help=GLOSSARY["Stop Loss"])
        stop_loss = st.slider("Stop distance", 0.01, 0.50, 0.10, 0.01) if use_stop else None
        use_target = st.checkbox("Volatility targeting", help=GLOSSARY["Volatility Targeting"])
        target = st.slider("Annual volatility target", 0.05, 0.50, 0.15, 0.01) if use_target else None

    try:
        config = BacktestConfig(
            initial_capital=capital,
            commission_rate=commission_bps / 10_000,
            slippage_bps=slippage_bps,
            position_fraction=position_fraction,
            stop_loss=stop_loss,
            target_volatility=target,
        )
        error = None
    except ValueError as exc:
        config, error = None, str(exc)
    return Inputs(source, ticker, start, end, strategy_key, params, config, error)


def load_prices(inputs: Inputs) -> pd.DataFrame:
    if inputs.source == "Synthetic sample":
        return load_sample()
    return cached_fetch(inputs.ticker, inputs.start, inputs.end)


def render_headline(result: BacktestResult) -> None:
    strategy, benchmark = result.metrics.as_dict(), result.benchmark_metrics.as_dict()
    for column, name in zip(st.columns(len(HEADLINE_METRICS)), HEADLINE_METRICS, strict=True):
        label = name.replace("_", " ").title().replace("Cagr", "CAGR")
        column.metric(
            label,
            format_metric(name, strategy[name]),
            f"Buy and hold: {format_metric(name, benchmark[name])}",
            delta_color="off",
            delta_arrow="off",
        )


def render_backtest(result: BacktestResult) -> None:
    results = result.results
    render_headline(result)
    st.plotly_chart(equity_figure(results["Strategy Equity"], results["Buy and Hold Benchmark"]))
    left, right = st.columns(2)
    left.plotly_chart(drawdown_figure(results["Strategy Drawdown"], results["Benchmark Drawdown"]))
    right.plotly_chart(monthly_returns_figure(results["Strategy Equity"]))
    st.plotly_chart(price_figure(results["Close"], result.trade_log))

    left, right = st.columns([2, 3])
    with left:
        st.subheader("All Metrics")
        st.dataframe(comparison_table(result.metrics, result.benchmark_metrics))
        relative = result.relative_metrics
        st.caption(
            f"Alpha {format_metric('total_return', relative.alpha)} · Beta {relative.beta:.2f} · "
            f"Information ratio {relative.information_ratio:.2f}"
        )
    with right:
        st.subheader(f"Trades ({len(result.trade_log)})")
        st.dataframe(without_times(result.trade_log).round(4), hide_index=True)
        st.download_button(
            "Download trades as CSV",
            result.trade_log.to_csv(index=False),
            file_name="trades.csv",
            mime="text/csv",
        )


def render_robustness(result: BacktestResult) -> None:
    st.write(
        "A single backtest is one draw from what could have happened. Resampling its returns shows how far "
        "the result could plausibly move by chance."
    )
    returns = daily_returns(result.results["Strategy Equity"])
    benchmark = daily_returns(result.results["Buy and Hold Benchmark"])
    if len(returns) < 40:
        st.warning("At least 40 bars are needed to estimate confidence intervals.")
        return

    estimate = bootstrap(returns, benchmark)
    columns = st.columns(3)
    columns[0].metric(
        "Sharpe 95% Interval",
        f"{estimate.sharpe.low:.2f} to {estimate.sharpe.high:.2f}",
        help=GLOSSARY["Block Bootstrap"],
    )
    columns[1].metric(
        "Probabilistic Sharpe",
        f"{probabilistic_sharpe_ratio(returns):.1%}",
        help=GLOSSARY["Probabilistic Sharpe Ratio"],
    )
    columns[2].metric("Chance of Beating Buy and Hold", f"{estimate.probability_beats_benchmark:.1%}")
    st.plotly_chart(
        bootstrap_figure(
            estimate.sharpe_samples, estimate.sharpe.low, estimate.sharpe.high, sharpe_ratio(returns)
        )
    )


def render_optimize(prices: pd.DataFrame, inputs: Inputs) -> None:
    st.write(
        "Search every parameter combination. The winner's score is **in-sample**, so it is flattered by "
        "selection. The deflated Sharpe ratio corrects for how many combinations were tried."
    )
    left, right = st.columns(2)
    points = left.slider("Grid points per parameter", 3, 10, 6)
    objective = right.selectbox(
        "Objective", OBJECTIVES, format_func=lambda name: name.replace("_", " ").title()
    )
    if st.button("Run search"):
        grid = build_grid(inputs.strategy_key, points)
        with st.spinner(f"Testing {len(grid)} combinations..."):
            st.session_state["search"] = (
                inputs.strategy_key,
                grid_search(prices, inputs.strategy_key, grid, inputs.config, objective),
            )

    stored = st.session_state.get("search")
    if stored is None or stored[0] != inputs.strategy_key:
        return
    search = stored[1]
    parameters = [spec.name for spec in STRATEGIES[inputs.strategy_key].params]
    columns = st.columns(3)
    columns[0].metric("Combinations Tested", len(search.table))
    columns[1].metric("Best Sharpe (In-Sample)", f"{search.table['sharpe_ratio'].max():.2f}")
    columns[2].metric(
        "Deflated Sharpe", f"{search.deflated_sharpe:.1%}", help=GLOSSARY["Deflated Sharpe Ratio"]
    )
    st.info(probability_verdict(search.deflated_sharpe))
    st.caption(f"Best parameters: {search.best_params}")
    st.plotly_chart(search_figure(search.table, parameters, search.objective))
    st.dataframe(search.table.round(3), hide_index=True)


def render_walk_forward(prices: pd.DataFrame, inputs: Inputs) -> None:
    st.write(
        "The fair test. Parameters are chosen on one window and traded on the next, unseen window, then the "
        "process rolls forward. The equity curve below is made only of decisions that could have been "
        "taken at the time."
    )
    columns = st.columns(4)
    train_bars = columns[0].number_input("Training bars", 100, 2520, 756, 21)
    test_bars = columns[1].number_input("Test bars", 21, 1260, 252, 21)
    points = columns[2].slider("Grid points", 3, 8, 5)
    anchored = columns[3].checkbox("Anchored", help="Grow the training window instead of sliding it.")
    if st.button("Run walk-forward"):
        grid = build_grid(inputs.strategy_key, points)
        try:
            with st.spinner("Optimizing and testing each fold..."):
                st.session_state["walk"] = (
                    inputs.strategy_key,
                    walk_forward(
                        prices,
                        inputs.strategy_key,
                        grid,
                        inputs.config,
                        train_bars=int(train_bars),
                        test_bars=int(test_bars),
                        anchored=anchored,
                    ),
                )
        except ValueError as error:
            st.session_state.pop("walk", None)
            st.error(str(error))

    stored = st.session_state.get("walk")
    if stored is None or stored[0] != inputs.strategy_key:
        return
    result = stored[1]
    columns = st.columns(4)
    columns[0].metric("Out-of-Sample Sharpe", format_metric("sharpe_ratio", result.metrics.sharpe_ratio))
    columns[1].metric(
        "Buy and Hold Sharpe", format_metric("sharpe_ratio", result.benchmark_metrics.sharpe_ratio)
    )
    columns[2].metric("Out-of-Sample CAGR", format_metric("cagr", result.metrics.cagr))
    columns[3].metric(
        "Efficiency",
        format_metric("sharpe_ratio", result.efficiency),
        help=GLOSSARY["Walk-Forward Efficiency"],
    )
    st.plotly_chart(equity_figure(result.equity, result.benchmark, "Out-of-Sample Equity Curve"))
    st.plotly_chart(fold_scores_figure(result.folds, result.objective))
    folds = result.folds.assign(Parameters=result.folds["Parameters"].astype(str))
    st.dataframe(without_times(folds).round(3), hide_index=True)


def render_learn() -> None:
    query = st.text_input("Search terms", placeholder="Try: Sharpe").strip().lower()
    for term, definition in sorted(GLOSSARY.items()):
        if query in term.lower() or query in definition.lower():
            st.markdown(f"**{term}.** {definition}")


def main() -> None:
    st.set_page_config(page_title="TradStrat", layout="wide")
    st.title("TradStrat")
    st.caption(
        "Backtest a trading idea, then find out whether the result is real. "
        "For research and education only; this is not investment advice."
    )

    inputs = read_sidebar()
    if inputs.config is None:
        st.error(inputs.config_error)
        return

    try:
        strategy = create_strategy(inputs.strategy_key, **inputs.params)
    except ValueError as error:
        st.error(f"These parameters don't work together: {error}")
        return

    try:
        prices = validate_ohlcv(load_prices(inputs), inputs.ticker or "sample")
    except DataError as error:
        st.error(str(error))
        return

    if len(prices) <= strategy.warmup:
        st.warning(
            f"{STRATEGIES[inputs.strategy_key].label} needs more than {strategy.warmup} bars of history, "
            f"but only {len(prices)} are available. Use a longer date range or a shorter window."
        )
        return

    result = run_backtest(prices, strategy, inputs.config)
    tabs = st.tabs(["Backtest", "Robustness", "Optimize", "Walk-Forward", "Glossary"])
    with tabs[0]:
        render_backtest(result)
    with tabs[1]:
        render_robustness(result)
    with tabs[2]:
        render_optimize(prices, inputs)
    with tabs[3]:
        render_walk_forward(prices, inputs)
    with tabs[4]:
        render_learn()


if __name__ == "__main__":
    main()
