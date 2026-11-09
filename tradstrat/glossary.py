"""Plain-English definitions for terms used in the dashboard, and a helper that explains them on hover."""

from __future__ import annotations

import html
import re

GLOSSARY: dict[str, str] = {
    "Alpha": "Annual return left over after accounting for the strategy's exposure to the benchmark.",
    "Annual Volatility": "How much returns vary, scaled to a year. It is the usual measure of risk.",
    "Average Loss": "The average dollar loss across losing trades, after costs.",
    "Average Win": "The average dollar profit across winning trades, after costs.",
    "Bar": "One row of price data. Here, one trading day, with its open, high, low and close prices.",
    "Basis Point": "One hundredth of a percent, written bps. A cost of 10 bps is 0.10% of the trade.",
    "Backtesting": "Testing a trading idea on old market data to see how it would have behaved in the past.",
    "Beta": "How strongly the strategy moves with the benchmark. A beta of 0.5 means about half the benchmark's moves.",
    "Block Bootstrap": "Reshuffling the history in chunks thousands of times, to see how much a result could change if the past had played out a little differently.",
    "Buy-and-Hold Benchmark": "A simple comparison that buys the asset once and keeps it for the whole period, paying the same costs as the strategy.",
    "CAGR": "Compound annual growth rate: the steady yearly return that would turn the starting amount into the final amount.",
    "Cash Yield": "Interest earned on money that is not invested. TradStrat can credit idle cash with real US Treasury bill yields, and it measures Sharpe ratios on returns above that rate, so sitting in cash is never mistaken for skill.",
    "Confidence Interval": "A range that the true value is likely to fall in. A 95% interval should contain it 95 times in 100.",
    "Calmar Ratio": "CAGR divided by the maximum drawdown: return earned per unit of the worst loss.",
    "Commission": "A fee charged by the broker every time you trade.",
    "Deflated Sharpe Ratio": "The chance that a strategy's edge is real, after allowing for how many settings were tried before picking the best. Try enough settings and one will look good by luck.",
    "Dividend Adjustment": "Scaling past prices so that returns include dividends, as if they had been reinvested.",
    "Donchian Channel": "The highest high and lowest low over a recent window. A close outside it is a breakout.",
    "Equity Curve": "A line showing how portfolio value changes over time.",
    "Expectancy": "The average profit or loss per trade, after costs.",
    "Exposure": "The share of days the strategy owned the asset rather than sitting in cash.",
    "In-Sample": "The data used to choose a strategy's settings. Results here look better than they will in future.",
    "Information Ratio": "Return above the benchmark, divided by how much the strategy deviates from it.",
    "Look-Ahead Bias": "A mistake where a backtest uses information that would not have been available at the time.",
    "Moving Average": "The average closing price over the last so many trading days. It smooths out daily noise.",
    "MACD": "Moving average convergence divergence: the gap between a fast and a slow exponential moving "
    "average. Rising means the trend is speeding up.",
    "Max Drawdown": "The biggest fall from a high point to a later low: the worst loss you would have had to sit through.",
    "Max Drawdown Duration": "The longest stretch spent below a previous peak.",
    "Mean Reversion": "The idea that prices often move back toward their recent average after stretching away from it.",
    "Momentum": "The tendency for assets that have been rising to keep rising for a while.",
    "Moving Average Crossover": "A rule that holds the asset while a short moving average is above a long one.",
    "Number of Trades": "The number of completed round trips, each one buy and one sell.",
    "Out-of-Sample": "Data that played no part in choosing the settings. Results here are a fair test.",
    "Overfitting": "Tuning a strategy so closely to past noise that it fails on new data.",
    "Position Sizing": "How much of your portfolio you put into a trade.",
    "Probabilistic Sharpe Ratio": "The chance that the strategy truly beats cash, given how long the test was and how unusual its returns were.",
    "Probability of Backtest Overfitting": "How often the configuration that looked best on one half of the history fell into the bottom half on the other half. Near 50% means the search was choosing by luck; near 0% means the choice carried over.",
    "Percentage Point": "The plain difference between two percentages. Going from 8% to 10% is a rise of 2 points.",
    "Profit Factor": "Gross profit divided by gross loss. Above 1 means the winners outweigh the losers.",
    "RSI": "Relative Strength Index, from 0 to 100. Low readings suggest a market has fallen too far, too fast.",
    "Sharpe Ratio": "Return earned for each unit of risk taken, measured on returns above cash. Higher is better, and it lets a calm strategy be compared fairly with a wild one.",
    "Slippage": "The gap between the price you expect and the price you actually get.",
    "Sortino Ratio": "Like the Sharpe ratio, but it only penalizes downside volatility.",
    "Stop Loss": "An order that exits a position if the price falls a set percentage below the entry.",
    "Ticker": "The short code an asset trades under, such as SPY for an S&P 500 fund or BTC-USD for Bitcoin.",
    "Total Return": "The overall percentage gain or loss across the test.",
    "Trailing Stop": "A stop that rises with the highest price reached since entry, locking in part of a gain. It never moves down.",
    "Volatility Targeting": "Holding less when markets are turbulent, so that risk stays closer to a chosen level.",
    "Walk-Forward Efficiency": "Out-of-sample score as a share of in-sample score. Near 1 suggests the strategy held up; near 0 suggests overfitting.",
    "Walk-Forward Validation": "Repeatedly choosing parameters on one window of data and testing them on the next, unseen window.",
    "Win Rate": "The percentage of closed trades that ended with a profit.",
    "Z-Score": "Distance from the average in standard deviations. A z-score of -2 is two standard deviations below the mean.",
}


# Everyday phrasings that refer to a glossary entry. Every entry's own name is matched as well.
ALIASES: dict[str, str] = {
    "bps": "Basis Point",
    "buy and hold": "Buy-and-Hold Benchmark",
    "deflated sharpe": "Deflated Sharpe Ratio",
    "drawdown": "Max Drawdown",
    "idle cash": "Cash Yield",
    "moving averages": "Moving Average",
    "overfit": "Overfitting",
    "probability of overfitting": "Probability of Backtest Overfitting",
    "resampled histories": "Block Bootstrap",
    "risk-adjusted return": "Sharpe Ratio",
    "sharpe": "Sharpe Ratio",
    "standard deviations": "Z-Score",
    "time in the market": "Exposure",
    "treasury bill yields": "Cash Yield",
    "unseen data": "Out-of-Sample",
    "walk-forward": "Walk-Forward Validation",
    "worst decline": "Max Drawdown",
}
_TERMS: dict[str, str] = {term.lower(): term for term in GLOSSARY} | ALIASES
# Longest phrases first, so "deflated sharpe ratio" wins over "sharpe ratio". The lookarounds stop a
# match inside a longer word or hyphenated phrase.
_PATTERN = re.compile(
    r"(?<![\w-])("
    + "|".join(re.escape(term) for term in sorted(_TERMS, key=len, reverse=True))
    + r")(?![\w-])",
    re.IGNORECASE,
)


def annotate(text: str, seen: set[str] | None = None) -> str:
    """Escape ``text`` for HTML and mark the first mention of each glossary term with its definition.

    A marked term becomes ``<span class="term" data-tip="...">``, which the dashboard styles as a dotted
    underline that shows the definition on hover or keyboard focus. Pass the same ``seen`` set to several
    calls to explain each term only once across a block of text.
    """
    seen = set() if seen is None else seen

    def mark(match: re.Match[str]) -> str:
        term = _TERMS[match.group(0).lower()]
        if term in seen:
            return match.group(0)
        seen.add(term)
        tip = html.escape(GLOSSARY[term], quote=True)
        return f'<span class="term" tabindex="0" data-tip="{tip}">{match.group(0)}</span>'

    return _PATTERN.sub(mark, html.escape(text, quote=False))
