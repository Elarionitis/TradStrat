"""Strategy definitions for the trading strategy backtester."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


class Strategy(ABC):
    """Base class for all UI-agnostic strategy implementations."""

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame with a target-position signal for each date."""


@dataclass(frozen=True)
class MovingAverageCrossoverStrategy(Strategy):
    """Go long when the short moving average rises above the long moving average."""

    short_window: int
    long_window: int

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create long/flat target positions from rolling average crossovers."""
        if self.short_window <= 0 or self.long_window <= 0:
            raise ValueError("Moving average windows must be positive integers.")
        if self.short_window >= self.long_window:
            raise ValueError("The short window must be smaller than the long window.")

        frame = pd.DataFrame(index=data.index)
        frame["close"] = data["Close"]
        frame["short_ma"] = frame["close"].rolling(self.short_window, min_periods=self.short_window).mean()
        frame["long_ma"] = frame["close"].rolling(self.long_window, min_periods=self.long_window).mean()

        # No look-ahead bias: the moving averages only use closes available up to each row's date.
        # The engine will wait until the next day's open to fill any trade, so the signal never
        # uses future prices to decide today's position.
        frame["signal"] = 0
        frame.loc[frame["short_ma"] > frame["long_ma"], "signal"] = 1
        frame.loc[frame["long_ma"].isna(), "signal"] = 0
        return frame[["signal", "short_ma", "long_ma"]]


@dataclass(frozen=True)
class MeanReversionStrategy(Strategy):
    """Buy when price is stretched below its mean and exit when it snaps back."""

    lookback_window: int
    entry_threshold: float
    exit_threshold: float

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create long/flat target positions from a rolling z-score."""
        if self.lookback_window <= 1:
            raise ValueError("The lookback window must be greater than 1.")
        if self.entry_threshold <= 0 or self.exit_threshold <= 0:
            raise ValueError("Thresholds must be positive.")

        frame = pd.DataFrame(index=data.index)
        frame["close"] = data["Close"]
        rolling_mean = frame["close"].rolling(self.lookback_window, min_periods=self.lookback_window).mean()
        rolling_std = frame["close"].rolling(self.lookback_window, min_periods=self.lookback_window).std(ddof=0)
        frame["z_score"] = (frame["close"] - rolling_mean) / rolling_std

        target_position = []
        current_position = 0
        for z_score in frame["z_score"]:
            if pd.isna(z_score):
                target_position.append(0)
                continue
            if current_position == 0 and z_score <= -self.entry_threshold:
                current_position = 1
            elif current_position == 1 and z_score >= self.exit_threshold:
                current_position = 0
            target_position.append(current_position)

        # This state machine also avoids look-ahead bias: each decision uses only the rolling
        # statistics available on the current bar, and the engine still executes on the next open.
        frame["signal"] = target_position
        return frame[["signal", "z_score"]]
