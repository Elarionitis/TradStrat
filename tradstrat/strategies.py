"""Trading strategies.

A strategy turns a price history into a target position for each bar: 1 for long, 0 for flat. Each one
declares its parameters as ``ParamSpec`` entries, and the registry exposes them so the CLI, the dashboard
and the optimizer can all be driven from that single declaration. Adding a strategy means writing one
class; nothing else needs to change.

Every signal must depend only on data up to and including its own bar. The test suite checks this for
each registered strategy by recomputing signals on truncated histories.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, TypeVar

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ParamSpec:
    """One tunable strategy parameter and the range it may take."""

    name: str
    label: str
    default: float
    low: float
    high: float
    step: float
    help: str = ""

    @property
    def is_integer(self) -> bool:
        return isinstance(self.default, int)

    def grid(self, points: int = 5) -> list[Any]:
        """Return evenly spaced values across the allowed range, for a parameter search."""
        values = np.linspace(self.low, self.high, points)
        if self.is_integer:
            return sorted({round(float(value)) for value in values})
        return sorted({round(float(value), 4) for value in values})


class Strategy(ABC):
    """Base class for all strategies."""

    key: ClassVar[str]
    label: ClassVar[str]
    description: ClassVar[str]
    params: ClassVar[tuple[ParamSpec, ...]] = ()
    # Indicator columns drawn on the price chart, as (column, display name) pairs.
    overlays: ClassVar[tuple[tuple[str, str], ...]] = ()

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a frame indexed like ``data`` with a ``signal`` column and any indicator columns."""

    @property
    def warmup(self) -> int:
        """Bars of history needed before the strategy can produce its first signal."""
        return 0


StrategyType = TypeVar("StrategyType", bound=type[Strategy])
STRATEGIES: dict[str, type[Strategy]] = {}


def register(cls: StrategyType) -> StrategyType:
    """Class decorator that adds a strategy to the registry."""
    STRATEGIES[cls.key] = cls
    return cls


def create_strategy(key: str, **overrides: float) -> Strategy:
    """Build a registered strategy, using declared defaults for any parameter not supplied."""
    if key not in STRATEGIES:
        raise ValueError(f"Unknown strategy {key!r}. Choose from: {', '.join(sorted(STRATEGIES))}.")
    cls = STRATEGIES[key]
    specs = {spec.name: spec for spec in cls.params}
    unknown = set(overrides) - set(specs)
    if unknown:
        raise ValueError(f"Unknown parameters for {key}: {', '.join(sorted(unknown))}.")

    values: dict[str, Any] = {}
    for name, spec in specs.items():
        value = overrides.get(name, spec.default)
        values[name] = int(value) if spec.is_integer else float(value)
    return cls(**values)


def _hold_until_exit(enter: np.ndarray, exit_: np.ndarray) -> np.ndarray:
    """Turn entry and exit conditions into a position that is held between them.

    An undefined indicator compares as False, so it leaves the position unchanged. That matters for a
    rolling z-score, which is undefined whenever the window's prices are all equal.
    """
    position = np.zeros(len(enter), dtype=int)
    held = 0
    for bar, (should_enter, should_exit) in enumerate(zip(enter, exit_, strict=True)):
        if held == 0 and should_enter:
            held = 1
        elif held == 1 and should_exit:
            held = 0
        position[bar] = held
    return position


@register
@dataclass(frozen=True)
class BuyAndHoldStrategy(Strategy):
    """Always long. Used as the benchmark."""

    key: ClassVar[str] = "buy_and_hold"
    label: ClassVar[str] = "Buy and Hold"
    description: ClassVar[str] = "Buy once and hold for the whole period."

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"signal": 1}, index=data.index)


@register
@dataclass(frozen=True)
class MovingAverageCrossoverStrategy(Strategy):
    """Trend following: long while the short moving average is above the long one."""

    short_window: int = 50
    long_window: int = 200

    key: ClassVar[str] = "ma_crossover"
    label: ClassVar[str] = "Moving Average Crossover"
    description: ClassVar[str] = "Long while a fast moving average is above a slow one."
    params: ClassVar[tuple[ParamSpec, ...]] = (
        ParamSpec("short_window", "Short Window", 50, 5, 100, 5, "Trading days in the fast moving average."),
        ParamSpec("long_window", "Long Window", 200, 20, 300, 10, "Trading days in the slow moving average."),
    )
    overlays: ClassVar[tuple[tuple[str, str], ...]] = (
        ("short_ma", "Fast average"),
        ("long_ma", "Slow average"),
    )

    def __post_init__(self) -> None:
        if self.short_window <= 0 or self.long_window <= 0:
            raise ValueError("Moving average windows must be positive integers.")
        if self.short_window >= self.long_window:
            raise ValueError("The short window must be smaller than the long window.")

    @property
    def warmup(self) -> int:
        return self.long_window

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        close = data["Close"]
        short_ma = close.rolling(self.short_window, min_periods=self.short_window).mean()
        long_ma = close.rolling(self.long_window, min_periods=self.long_window).mean()
        return pd.DataFrame(
            {"signal": (short_ma > long_ma).astype(int), "short_ma": short_ma, "long_ma": long_ma},
            index=data.index,
        )


@register
@dataclass(frozen=True)
class MeanReversionStrategy(Strategy):
    """Buy when price is stretched below its rolling mean and exit as it reverts."""

    lookback_window: int = 20
    entry_threshold: float = 1.5
    exit_threshold: float = 0.0

    key: ClassVar[str] = "mean_reversion"
    label: ClassVar[str] = "Mean Reversion (Z-Score)"
    description: ClassVar[str] = "Buy when the rolling z-score is deeply negative; exit as it recovers."
    params: ClassVar[tuple[ParamSpec, ...]] = (
        ParamSpec(
            "lookback_window",
            "Lookback Window",
            20,
            5,
            100,
            5,
            "Trading days used for the average and the usual swing.",
        ),
        ParamSpec(
            "entry_threshold",
            "Entry Threshold",
            1.5,
            0.5,
            3.0,
            0.1,
            "Buy when the price is this many standard deviations below average.",
        ),
        ParamSpec(
            "exit_threshold",
            "Exit Threshold",
            0.0,
            -0.5,
            2.0,
            0.1,
            "Sell when the z-score climbs back to this level. Zero is the average.",
        ),
    )

    def __post_init__(self) -> None:
        if self.lookback_window <= 1:
            raise ValueError("The lookback window must be greater than 1.")
        if self.entry_threshold <= 0:
            raise ValueError("The entry threshold must be positive.")
        if self.exit_threshold <= -self.entry_threshold:
            raise ValueError("The exit threshold must be above the entry level.")

    @property
    def warmup(self) -> int:
        return self.lookback_window

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        close = data["Close"]
        rolling = close.rolling(self.lookback_window, min_periods=self.lookback_window)
        z_score = (close - rolling.mean()) / rolling.std(ddof=0)
        signal = _hold_until_exit(
            (z_score <= -self.entry_threshold).to_numpy(), (z_score >= self.exit_threshold).to_numpy()
        )
        return pd.DataFrame({"signal": signal, "z_score": z_score}, index=data.index)


def relative_strength_index(close: pd.Series, window: int) -> pd.Series:
    """Wilder's RSI, which smooths gains and losses with an exponential average of alpha = 1 / window."""
    change = close.diff()
    average_gain = change.clip(lower=0).ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    average_loss = (-change.clip(upper=0)).ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    return 100 - 100 / (1 + average_gain / average_loss)


@register
@dataclass(frozen=True)
class RsiStrategy(Strategy):
    """Buy oversold conditions and exit once momentum has recovered."""

    window: int = 14
    oversold: float = 30.0
    exit_level: float = 55.0

    key: ClassVar[str] = "rsi"
    label: ClassVar[str] = "RSI Reversal"
    description: ClassVar[str] = "Buy when RSI is oversold and exit when it recovers."
    params: ClassVar[tuple[ParamSpec, ...]] = (
        ParamSpec(
            "window", "RSI Window", 14, 5, 30, 1, "Trading days of gains and losses that RSI averages."
        ),
        ParamSpec(
            "oversold", "Oversold Level", 30.0, 10.0, 45.0, 5.0, "Buy when RSI falls below this level."
        ),
        ParamSpec("exit_level", "Exit Level", 55.0, 50.0, 80.0, 5.0, "Sell when RSI rises above this level."),
    )

    def __post_init__(self) -> None:
        if self.window <= 1:
            raise ValueError("The RSI window must be greater than 1.")
        if not 0 < self.oversold < self.exit_level < 100:
            raise ValueError("RSI levels must satisfy 0 < oversold < exit level < 100.")

    @property
    def warmup(self) -> int:
        return self.window + 1

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        rsi = relative_strength_index(data["Close"], self.window)
        signal = _hold_until_exit((rsi < self.oversold).to_numpy(), (rsi > self.exit_level).to_numpy())
        return pd.DataFrame({"signal": signal, "rsi": rsi}, index=data.index)


@register
@dataclass(frozen=True)
class DonchianBreakoutStrategy(Strategy):
    """Trend following: buy a new high and exit on a shorter-term new low."""

    entry_window: int = 55
    exit_window: int = 20

    key: ClassVar[str] = "donchian"
    label: ClassVar[str] = "Donchian Breakout"
    description: ClassVar[str] = "Buy a new N-day high; exit on a new M-day low."
    params: ClassVar[tuple[ParamSpec, ...]] = (
        ParamSpec(
            "entry_window",
            "Entry Window",
            55,
            10,
            120,
            5,
            "Buy a close above the highest price of this many days.",
        ),
        ParamSpec(
            "exit_window",
            "Exit Window",
            20,
            5,
            60,
            5,
            "Sell a close below the lowest price of this many days.",
        ),
    )
    overlays: ClassVar[tuple[tuple[str, str], ...]] = (("upper", "Entry channel"), ("lower", "Exit channel"))

    def __post_init__(self) -> None:
        if self.entry_window <= 1 or self.exit_window <= 1:
            raise ValueError("Channel windows must be greater than 1.")
        if self.exit_window >= self.entry_window:
            raise ValueError("The exit window must be smaller than the entry window.")

    @property
    def warmup(self) -> int:
        return self.entry_window + 1

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        # shift(1) keeps the channel to prior bars. Including the current bar would make the close
        # compare against a high that already contains it, so a breakout could never register.
        upper = data["High"].rolling(self.entry_window, min_periods=self.entry_window).max().shift(1)
        lower = data["Low"].rolling(self.exit_window, min_periods=self.exit_window).min().shift(1)
        close = data["Close"]
        signal = _hold_until_exit((close > upper).to_numpy(), (close < lower).to_numpy())
        return pd.DataFrame({"signal": signal, "upper": upper, "lower": lower}, index=data.index)


@register
@dataclass(frozen=True)
class MomentumStrategy(Strategy):
    """Time-series momentum: long while the trailing return is positive."""

    lookback_window: int = 126

    key: ClassVar[str] = "momentum"
    label: ClassVar[str] = "Time-Series Momentum"
    description: ClassVar[str] = "Long while the return over the lookback window is positive."
    params: ClassVar[tuple[ParamSpec, ...]] = (
        ParamSpec(
            "lookback_window", "Lookback Window", 126, 20, 252, 2, "Trading days in the trailing return."
        ),
    )

    def __post_init__(self) -> None:
        if self.lookback_window <= 0:
            raise ValueError("The lookback window must be positive.")

    @property
    def warmup(self) -> int:
        return self.lookback_window + 1

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        trailing_return = data["Close"].pct_change(self.lookback_window, fill_method=None)
        return pd.DataFrame(
            {"signal": (trailing_return > 0).astype(int), "trailing_return": trailing_return},
            index=data.index,
        )


def indicator_overlays(strategy: Strategy, data: pd.DataFrame) -> pd.DataFrame | None:
    """The strategy's price-scale indicators, labeled for display, or None if it has none."""
    if not strategy.overlays:
        return None
    signals = strategy.generate_signals(data)
    return pd.DataFrame({label: signals[column] for column, label in strategy.overlays}, index=data.index)
