"""Plain-English definitions for terms used in the Streamlit UI."""

from __future__ import annotations

GLOSSARY: dict[str, str] = {
    "Backtesting": "Testing a trading idea on old market data to see how it would have behaved in the past.",
    "Buy-and-Hold Benchmark": "A simple comparison strategy that buys the asset once and keeps it for the whole period.",
    "CAGR": "Compound Annual Growth Rate. This turns the total return into an annualized growth rate so different time periods are easier to compare.",
    "Commission": "A fee charged by the broker or exchange every time you trade.",
    "Equity Curve": "A line chart showing how your portfolio value changes over time.",
    "Look-Ahead Bias": "A mistake where a strategy accidentally uses future information that would not have been available at the time.",
    "Max Drawdown": "The largest peak-to-trough drop in portfolio value during the test period.",
    "Mean Reversion": "The idea that prices often move back toward their recent average after stretching too far away from it.",
    "Moving Average Crossover": "A rule that buys when a shorter moving average rises above a longer one and exits when it falls back below it.",
    "Position Sizing": "How much of your portfolio you put into a trade.",
    "Sharpe Ratio": "A measure of return compared with total volatility. Higher is better if you want more return for each unit of risk.",
    "Slippage": "The difference between the price you expect and the price you actually get when a trade is filled.",
    "Sortino Ratio": "Like the Sharpe Ratio, but it only penalizes downside volatility, not upside swings.",
    "Win Rate": "The percentage of closed trades that ended with a profit.",
    "Z-Score": "A standardized distance from the average. A z-score of -2 means the price is two standard deviations below its recent mean.",
    "Starting Capital": "The amount of money the backtest begins with.",
    "Ticker": "The market symbol for the asset you want to test, such as AAPL for Apple.",
    "Short Window": "The number of recent days used for the faster moving average.",
    "Long Window": "The number of recent days used for the slower moving average.",
    "Lookback Window": "The number of recent days used to compute the rolling average and standard deviation.",
    "Entry Threshold": "How far the price must move away from its average before the strategy enters a trade.",
    "Exit Threshold": "How far back toward the average the price must move before the strategy exits.",
    "Commission Rate": "The percentage fee charged on each trade.",
    "Position Fraction": "The fraction of available portfolio value to commit when opening a trade.",
}
