"""Event-driven backtesting engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pandas as pd

from strategies import Strategy

TRADING_DAYS_PER_YEAR: Final[int] = 252


@dataclass(frozen=True)
class BacktestConfig:
    """Configuration for a single backtest run."""

    initial_capital: float
    commission_rate: float
    position_fraction: float


@dataclass
class BacktestResult:
    """Container for the time series and trade log produced by the engine."""

    results: pd.DataFrame
    trade_log: pd.DataFrame


def run_backtest(data: pd.DataFrame, strategy: Strategy, config: BacktestConfig) -> BacktestResult:
    """Run a day-by-day simulation using next-day-open execution."""
    if data.empty:
        raise ValueError("Backtests require at least one row of price data.")
    if not 0 < config.position_fraction <= 1:
        raise ValueError("Position fraction must be between 0 and 1.")
    if config.initial_capital <= 0:
        raise ValueError("Initial capital must be positive.")

    signals = strategy.generate_signals(data).copy()
    signals["signal"] = signals["signal"].fillna(0).astype(int)

    equity_curve = []
    benchmark_curve = []
    position_history = []
    signal_history = []
    action_markers = []
    trade_rows = []

    cash = config.initial_capital
    shares = 0
    entry_price = None
    entry_date = None

    first_close = float(data["Close"].iloc[0])

    for i, (date, row) in enumerate(data.iterrows()):
        current_signal = int(signals["signal"].iloc[i])
        signal_history.append(current_signal)

        if i > 0:
            execution_price = float(row["Open"])
            if shares == 0 and signal_history[-2] == 1:
                # The signal was formed on the prior day's close, but the order is filled here at
                # today's open. That one-bar delay is what keeps the event loop free of look-ahead bias.
                shares = int((cash * config.position_fraction) / (execution_price * (1 + config.commission_rate)))
                if shares > 0:
                    cost = shares * execution_price
                    commission = cost * config.commission_rate
                    cash -= cost + commission
                    entry_price = execution_price
                    entry_date = data.index[i]
                    action_markers.append({"Date": data.index[i], "Price": execution_price, "Action": "Buy"})
            elif shares > 0 and signal_history[-2] == 0:
                proceeds = shares * execution_price
                commission = proceeds * config.commission_rate
                cash += proceeds - commission
                pnl = cash - config.initial_capital if entry_price is not None else 0.0
                trade_rows.append(
                    {
                        "Entry Date": entry_date,
                        "Exit Date": data.index[i],
                        "Entry Price": entry_price,
                        "Exit Price": execution_price,
                        "Size": shares,
                        "P&L": (execution_price - entry_price) * shares - (shares * entry_price + proceeds) * config.commission_rate,
                    }
                )
                action_markers.append({"Date": data.index[i], "Price": execution_price, "Action": "Sell"})
                shares = 0
                entry_price = None
                entry_date = None

        close_price = float(row["Close"])
        equity = cash + shares * close_price
        benchmark_equity = config.initial_capital * (close_price / first_close)
        equity_curve.append(equity)
        benchmark_curve.append(benchmark_equity)
        position_history.append(shares)

    if shares > 0:
        final_date = data.index[-1]
        final_price = float(data["Close"].iloc[-1])
        proceeds = shares * final_price
        commission = proceeds * config.commission_rate
        cash += proceeds - commission
        trade_rows.append(
            {
                "Entry Date": entry_date,
                "Exit Date": final_date,
                "Entry Price": entry_price,
                "Exit Price": final_price,
                "Size": shares,
                "P&L": (final_price - entry_price) * shares - (shares * entry_price + proceeds) * config.commission_rate,
            }
        )
        action_markers.append({"Date": final_date, "Price": final_price, "Action": "Sell"})
        equity_curve[-1] = cash
        position_history[-1] = 0

    results = pd.DataFrame(
        {
            "Strategy Equity": equity_curve,
            "Buy and Hold Benchmark": benchmark_curve,
            "Signal": signal_history,
            "Position": position_history,
            "Close": data["Close"].astype(float).values,
        },
        index=data.index,
    )
    results["Strategy Drawdown"] = results["Strategy Equity"] / results["Strategy Equity"].cummax() - 1
    results["Benchmark Drawdown"] = results["Buy and Hold Benchmark"] / results["Buy and Hold Benchmark"].cummax() - 1

    return BacktestResult(results=results, trade_log=pd.DataFrame(trade_rows),)
