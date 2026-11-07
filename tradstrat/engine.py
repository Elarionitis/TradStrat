"""Event-driven backtesting engine.

The engine walks through history one bar at a time, as a live trading system would:

1. At the open of bar ``t`` it acts on the signal formed at the close of bar ``t - 1``.
2. During the bar it checks any resting stop order against the bar's low.
3. At the close it marks the position to market.

Because a signal is always filled on the following bar, a strategy cannot trade on a price it used to
make the decision. Fills pay slippage and commission, and only whole shares are traded.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import Any

import numpy as np
import pandas as pd

from tradstrat.metrics import (
    TRADING_DAYS_PER_YEAR,
    Metrics,
    RelativeMetrics,
    calculate_metrics,
    calculate_relative_metrics,
)
from tradstrat.strategies import BuyAndHoldStrategy, Strategy

TRADE_COLUMNS = [
    "Entry Date",
    "Exit Date",
    "Entry Price",
    "Exit Price",
    "Size",
    "Commission",
    "P&L",
    "Return",
    "Bars Held",
    "Exit Reason",
]
REQUIRED_COLUMNS = ["Open", "Low", "Close"]


@dataclass(frozen=True)
class BacktestConfig:
    """Settings for a single backtest.

    ``commission_rate`` is a fraction of traded value and ``slippage_bps`` is the adverse price move per
    fill in basis points. ``stop_loss`` is a fraction below the entry fill. ``target_volatility`` is an
    annual figure: when set, positions are scaled down so that recent volatility matches it.
    """

    initial_capital: float = 10_000.0
    commission_rate: float = 0.001
    slippage_bps: float = 5.0
    position_fraction: float = 1.0
    stop_loss: float | None = None
    target_volatility: float | None = None
    volatility_lookback: int = 20

    def __post_init__(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError("Initial capital must be positive.")
        if not 0 < self.position_fraction <= 1:
            raise ValueError("Position fraction must be between 0 and 1.")
        if self.commission_rate < 0 or self.slippage_bps < 0:
            raise ValueError("Commission and slippage cannot be negative.")
        if self.stop_loss is not None and not 0 < self.stop_loss < 1:
            raise ValueError("Stop loss must be between 0 and 1.")
        if self.target_volatility is not None and self.target_volatility <= 0:
            raise ValueError("Target volatility must be positive.")
        if self.volatility_lookback < 2:
            raise ValueError("Volatility lookback must be at least 2.")


@dataclass
class _Book:
    """The cash and open position during a simulation, plus the trades closed so far."""

    cash: float
    commission_rate: float
    slippage: float
    shares: int = 0
    entry_price: float = 0.0
    entry_commission: float = 0.0
    entry_bar: int = 0
    entry_date: pd.Timestamp | None = None
    trades: list[dict[str, Any]] = field(default_factory=list)

    def buy(self, bar: int, date: pd.Timestamp, price: float, budget: float) -> None:
        fill = price * (1 + self.slippage)
        shares = int(budget / (fill * (1 + self.commission_rate)))
        if shares <= 0:
            return
        commission = shares * fill * self.commission_rate
        self.cash -= shares * fill + commission
        self.shares = shares
        self.entry_price = fill
        self.entry_commission = commission
        self.entry_bar = bar
        self.entry_date = date

    def sell(self, bar: int, date: pd.Timestamp, price: float, reason: str) -> None:
        fill = price * (1 - self.slippage)
        commission = self.shares * fill * self.commission_rate
        self.cash += self.shares * fill - commission
        total_commission = self.entry_commission + commission
        profit = (fill - self.entry_price) * self.shares - total_commission
        self.trades.append(
            {
                "Entry Date": self.entry_date,
                "Exit Date": date,
                "Entry Price": self.entry_price,
                "Exit Price": fill,
                "Size": self.shares,
                "Commission": total_commission,
                "P&L": profit,
                "Return": profit / (self.shares * self.entry_price),
                "Bars Held": bar - self.entry_bar,
                "Exit Reason": reason,
            }
        )
        self.shares = 0
        self.entry_date = None


@dataclass(frozen=True)
class BacktestResult:
    """The time series and trade log produced by a run, with metrics computed on demand."""

    results: pd.DataFrame
    trade_log: pd.DataFrame
    config: BacktestConfig

    @cached_property
    def metrics(self) -> Metrics:
        return calculate_metrics(self.results["Strategy Equity"], self.trade_log, self.results["Position"])

    @cached_property
    def benchmark_metrics(self) -> Metrics:
        return calculate_metrics(self.results["Buy and Hold Benchmark"], pd.DataFrame())

    @cached_property
    def relative_metrics(self) -> RelativeMetrics:
        return calculate_relative_metrics(
            self.results["Strategy Equity"], self.results["Buy and Hold Benchmark"]
        )


def _extract_signal(data: pd.DataFrame, strategy: Strategy) -> np.ndarray:
    """Run the strategy and check that its output is a usable long/flat series."""
    signals = strategy.generate_signals(data)
    if not signals.index.equals(data.index):
        raise ValueError("A strategy must return signals indexed like its input data.")
    signal = signals["signal"].fillna(0).to_numpy(dtype=int)
    if not np.isin(signal, (0, 1)).all():
        raise ValueError("Signals must be 0 (flat) or 1 (long).")
    return signal


def _realized_volatility(close: pd.Series, lookback: int) -> np.ndarray:
    """Annualized volatility of close-to-close returns, using data up to and including each bar."""
    volatility = close.pct_change().rolling(lookback).std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
    return volatility.to_numpy()


def _simulate(
    data: pd.DataFrame, signal: np.ndarray, config: BacktestConfig, start: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Step through history from ``start``. Returns equity, cash, shares held and closed trades.

    The first row is a baseline holding the initial capital, with no trading on it. For ``start`` above
    zero that row is the bar before ``start``, so a return earned on the first traded bar is counted.
    """
    opens = data["Open"].to_numpy(dtype=float)
    lows = data["Low"].to_numpy(dtype=float)
    closes = data["Close"].to_numpy(dtype=float)
    dates = data.index
    baseline = max(start - 1, 0)
    bars = len(data) - baseline

    volatility = None
    if config.target_volatility is not None:
        volatility = _realized_volatility(data["Close"], config.volatility_lookback)

    book = _Book(
        cash=config.initial_capital,
        commission_rate=config.commission_rate,
        slippage=config.slippage_bps / 10_000,
    )
    equity = np.empty(bars)
    cash = np.empty(bars)
    position = np.empty(bars, dtype=int)
    stopped_out = False

    for step, bar in enumerate(range(baseline, len(data))):
        if bar > baseline:
            target = signal[bar - 1]
            if target == 0:
                stopped_out = False

            if book.shares > 0 and target == 0:
                book.sell(bar, dates[bar], opens[bar], "signal")
            elif book.shares == 0 and target == 1 and not stopped_out:
                fraction = config.position_fraction
                if volatility is not None and config.target_volatility is not None:
                    # Sizing reads the estimate from the prior bar. No estimate yet means no entry.
                    estimate = volatility[bar - 1]
                    fraction = min(fraction, config.target_volatility / estimate) if estimate > 0 else 0.0
                if fraction > 0:
                    book.buy(bar, dates[bar], opens[bar], book.cash * fraction)

        if book.shares > 0 and config.stop_loss is not None:
            stop_price = book.entry_price * (1 - config.stop_loss)
            if lows[bar] <= stop_price:
                # An open below the stop means the market gapped through it, so the fill is the open.
                # Filling at the stop price there would be an optimistic fiction.
                book.sell(bar, dates[bar], min(opens[bar], stop_price), "stop_loss")
                # Stay out until the signal resets, or the next bar would simply buy straight back in.
                stopped_out = True

        equity[step] = book.cash + book.shares * closes[bar]
        cash[step] = book.cash
        position[step] = book.shares

    if book.shares > 0:
        book.sell(len(data) - 1, dates[-1], closes[-1], "end_of_data")
        equity[-1] = cash[-1] = book.cash
        position[-1] = 0

    return equity, cash, position, book.trades


def run_backtest(
    data: pd.DataFrame,
    strategy: Strategy,
    config: BacktestConfig | None = None,
    *,
    start: int = 0,
    include_benchmark: bool = True,
) -> BacktestResult:
    """Run a strategy over ``data``.

    Bars before ``start`` warm up the indicators without being traded, which is how walk-forward
    validation tests a fresh window without waiting for a 200-bar average to fill. The results then begin
    at the last warm-up bar, as a baseline row holding the initial capital.

    The benchmark is a buy-and-hold position run through this same simulator, so it pays the same
    commission and slippage on its entry and exit. Comparing a strategy that pays costs against a
    frictionless benchmark would be unfair to the strategy.
    """
    config = config or BacktestConfig()
    if start < 0:
        raise ValueError("Start must not be negative.")
    baseline = max(start - 1, 0)
    if len(data) - baseline < 2:
        raise ValueError("A backtest requires at least two bars of price data.")
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Price data is missing required columns: {', '.join(missing)}.")

    signal = _extract_signal(data, strategy)
    equity, cash, position, trades = _simulate(data, signal, config, start)

    index = data.index[baseline:]
    results = pd.DataFrame(
        {
            "Strategy Equity": equity,
            "Cash": cash,
            "Signal": signal[baseline:],
            "Position": position,
            "Close": data["Close"].to_numpy(dtype=float)[baseline:],
        },
        index=index,
    )
    results["Strategy Drawdown"] = results["Strategy Equity"] / results["Strategy Equity"].cummax() - 1

    if include_benchmark:
        benchmark_config = replace(config, position_fraction=1.0, stop_loss=None, target_volatility=None)
        benchmark_signal = _extract_signal(data, BuyAndHoldStrategy())
        benchmark_equity = _simulate(data, benchmark_signal, benchmark_config, start)[0]
        results["Buy and Hold Benchmark"] = benchmark_equity
        results["Benchmark Drawdown"] = benchmark_equity / np.maximum.accumulate(benchmark_equity) - 1

    return BacktestResult(
        results=results, trade_log=pd.DataFrame(trades, columns=TRADE_COLUMNS), config=config
    )
