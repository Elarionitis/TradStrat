"""Shared test fixtures and data builders."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_bars(close: list[float] | np.ndarray, open_: list[float] | np.ndarray | None = None) -> pd.DataFrame:
    """Build a valid OHLCV frame from closes, defaulting each open to the bar's close."""
    close_array = np.asarray(close, dtype=float)
    open_array = close_array.copy() if open_ is None else np.asarray(open_, dtype=float)
    return pd.DataFrame(
        {
            "Open": open_array,
            "High": np.maximum(open_array, close_array),
            "Low": np.minimum(open_array, close_array),
            "Close": close_array,
            "Volume": 1_000,
        },
        index=pd.date_range("2020-01-01", periods=len(close_array), freq="B", name="Date"),
    )


def random_walk(bars: int = 600, seed: int = 7, volatility: float = 0.015) -> pd.DataFrame:
    """A reproducible random-walk price series with realistic open, high and low values."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, volatility, bars)))
    open_ = np.concatenate([[100.0], close[:-1]]) * np.exp(rng.normal(0, 0.003, bars))
    frame = make_bars(close, open_)
    frame["High"] *= 1 + np.abs(rng.normal(0, 0.004, bars))
    frame["Low"] *= 1 - np.abs(rng.normal(0, 0.004, bars))
    return frame


@pytest.fixture
def walk() -> pd.DataFrame:
    return random_walk()
