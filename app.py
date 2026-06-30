"""Streamlit user interface for TradStrat."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_loader import DataError, fetch_daily_ohlcv
from engine import BacktestConfig, run_backtest
from glossary import GLOSSARY
from metrics import performance_summary
from strategies import MeanReversionStrategy, MovingAverageCrossoverStrategy


APP_TITLE = "TradStrat"
CACHE_DIR = Path(".cache")


def tooltip(term: str) -> str:
    """Return a glossary explanation for a UI label."""

    return GLOSSARY.get(term, term)


def default_dates() -> tuple[pd.Timestamp, pd.Timestamp]:
    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=5)
    return start, end


def build_benchmark_curve(price_data: pd.DataFrame, initial_capital: float) -> pd.Series:
    """Buy-and-hold benchmark using the same period and starting capital."""

    start_price = float(price_data["Close"].iloc[0])
    shares = initial_capital / start_price
    return price_data["Close"] * shares


def plot_equity_curve(strategy_curve: pd.Series, benchmark_curve: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=strategy_curve.index, y=strategy_curve.values, name="Strategy", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=benchmark_curve.index, y=benchmark_curve.values, name="Buy and Hold Benchmark", line=dict(width=2, dash="dash")))
    fig.update_layout(title="Equity Curve", xaxis_title="Date", yaxis_title="Portfolio Value")
    return fig


def plot_price_signals(price_data: pd.DataFrame, trades: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=price_data.index, y=price_data["Close"], name="Close Price", line=dict(color="#1f77b4")))
    if not trades.empty:
        fig.add_trace(
            go.Scatter(
                x=trades["entry_date"],
                y=trades["entry_price"],
                mode="markers",
                name="Buy",
                marker=dict(symbol="triangle-up", size=10, color="green"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=trades["exit_date"],
                y=trades["exit_price"],
                mode="markers",
                name="Sell",
                marker=dict(symbol="triangle-down", size=10, color="red"),
            )
        )
    fig.update_layout(title="Price Chart with Trade Markers", xaxis_title="Date", yaxis_title="Price")
    return fig


def plot_drawdown(equity_curve: pd.Series) -> go.Figure:
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown.values, name="Drawdown", line=dict(color="#d62728")))
    fig.update_layout(title="Drawdown Over Time", xaxis_title="Date", yaxis_title="Drawdown")
    return fig


def make_strategy(strategy_name: str, params: dict[str, float]) -> object:
    if strategy_name == "Moving Average Crossover":
        return MovingAverageCrossoverStrategy(short_window=int(params["short_window"]), long_window=int(params["long_window"]))
    return MeanReversionStrategy(
        lookback_window=int(params["lookback_window"]),
        entry_threshold=float(params["entry_threshold"]),
        exit_threshold=float(params["exit_threshold"]),
    )


def parse_date_range(date_value: object) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Normalize Streamlit date input into a start/end timestamp pair."""

    if isinstance(date_value, tuple):
        if len(date_value) >= 2:
            return pd.Timestamp(date_value[0]), pd.Timestamp(date_value[1])
        if len(date_value) == 1:
            single_date = pd.Timestamp(date_value[0])
            return single_date, single_date
        return None

    if isinstance(date_value, list):
        if len(date_value) >= 2:
            return pd.Timestamp(date_value[0]), pd.Timestamp(date_value[1])
        if len(date_value) == 1:
            single_date = pd.Timestamp(date_value[0])
            return single_date, single_date
        return None

    if date_value is None:
        return None

    single_date = pd.Timestamp(date_value)
    return single_date, single_date


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.write("""This app lets you test a simple trading idea on past price data. You choose a ticker, a date range, and a strategy, then the app simulates how that strategy would have behaved day by day. It is for learning and exploration, not investment advice.""")

    start_default, end_default = default_dates()

    with st.sidebar:
        st.header("Configuration")
        ticker = st.text_input("Ticker", value="AAPL", help=tooltip("Ticker"))
        use_sample = st.checkbox("Use sample dataset (offline demo)", value=False, help="Use the included sample CSV instead of fetching live data from Yahoo.")
        date_range = st.date_input("Date Range", value=(start_default.date(), end_default.date()), help=tooltip("Date Range"))
        strategy_name = st.selectbox("Strategy", ["Moving Average Crossover", "Mean Reversion"], help="Choose the trading rule you want to test.")
        if strategy_name == "Moving Average Crossover":
            short_window = st.slider("Short Window", min_value=5, max_value=100, value=50, help=tooltip("Short Window"))
            long_window = st.slider("Long Window", min_value=20, max_value=300, value=200, help=tooltip("Long Window"))
            strategy_params = {"short_window": short_window, "long_window": long_window}
        else:
            lookback_window = st.slider("Lookback Window", min_value=5, max_value=100, value=20, help=tooltip("Lookback Window"))
            entry_threshold = st.slider("Entry Threshold", min_value=0.5, max_value=4.0, value=1.5, step=0.1, help=tooltip("Entry Threshold"))
            exit_threshold = st.slider("Exit Threshold", min_value=0.1, max_value=3.0, value=0.5, step=0.1, help=tooltip("Exit Threshold"))
            strategy_params = {"lookback_window": lookback_window, "entry_threshold": entry_threshold, "exit_threshold": exit_threshold}
        starting_capital = st.number_input("Starting Capital", min_value=1000.0, value=10000.0, step=500.0, help=tooltip("Starting Capital"))
        commission_rate = st.number_input("Commission Rate", min_value=0.0, max_value=0.01, value=0.001, step=0.0001, format="%.4f", help=tooltip("Commission Rate"))
        position_fraction = st.slider("Position Sizing", min_value=0.1, max_value=1.0, value=1.0, step=0.1, help=tooltip("Position Sizing"))
        run_clicked = st.button("Run Backtest", type="primary")

    if not run_clicked:
        st.info("Set your inputs in the sidebar, then click Run Backtest to see results.")
        return

    parsed_range = parse_date_range(date_range)
    if parsed_range is None:
        st.error("Please choose a valid start and end date.")
        return
    start_date, end_date = parsed_range

    strategy = make_strategy(strategy_name, strategy_params)

    price_data = None
    if use_sample:
        sample_path = Path(__file__).resolve().parent / "sample_data" / "aapl_sample.csv"
        if not sample_path.exists():
            st.error(f"Sample data not found at {sample_path}.")
            return
        try:
            price_data = pd.read_csv(sample_path, parse_dates=["Date"], index_col="Date")
            price_data.index = pd.to_datetime(price_data.index).tz_localize(None)
        except Exception as exc:
            st.error(f"Failed to load sample data: {exc}")
            return
        st.info("Using built-in sample dataset. Date range and ticker above are ignored for the demo.")
    else:
        try:
            price_data = fetch_daily_ohlcv(ticker, start_date, end_date, cache_dir=CACHE_DIR)
        except DataError as error:
            st.error(str(error))
            return
        except Exception:
            st.error("Could not load data for that ticker and date range.")
            return

    if strategy_name == "Moving Average Crossover" and len(price_data) < strategy_params["long_window"]:
        st.warning("The selected date range is shorter than the long moving-average window. Use a longer date range or a smaller long window.")
        return
    if strategy_name == "Mean Reversion" and len(price_data) < strategy_params["lookback_window"]:
        st.warning("The selected date range is shorter than the lookback window. Use a longer date range or a smaller lookback window.")
        return

    config = BacktestConfig(initial_capital=starting_capital, commission_rate=float(commission_rate), position_fraction=float(position_fraction))
    result = run_backtest(price_data, strategy, config)
    benchmark_curve = build_benchmark_curve(result.price_data, starting_capital)
    summary = performance_summary(result.equity_curve["Equity"], benchmark_curve, result.trades)

    st.subheader("Results")
    metric_cols = st.columns(5)
    metric_cols[0].metric("Total Return", f"{summary['Total Return']:.2%}")
    metric_cols[1].metric("CAGR", f"{summary['CAGR']:.2%}")
    metric_cols[2].metric("Sharpe Ratio", f"{summary['Sharpe Ratio']:.2f}")
    metric_cols[3].metric("Max Drawdown", f"{summary['Max Drawdown']:.2%}")
    metric_cols[4].metric("Win Rate", f"{summary['Win Rate']:.2%}")

    with st.expander("What do these metrics mean?"):
        for term in ["Total Return", "CAGR", "Sharpe Ratio", "Max Drawdown", "Win Rate", "Buy-and-Hold Benchmark"]:
            st.write(f"**{term}**: {tooltip(term)}")

    st.plotly_chart(plot_equity_curve(result.equity_curve["Equity"], benchmark_curve), use_container_width=True)
    st.plotly_chart(plot_price_signals(result.price_data, result.trades), use_container_width=True)
    st.plotly_chart(plot_drawdown(result.equity_curve["Equity"]), use_container_width=True)

    with st.expander("View all trades"):
        st.dataframe(result.trades, use_container_width=True)


if __name__ == "__main__":
    main()
