"""Utilities for fetching and validating historical market data."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd
import yfinance as yf

CACHE_DIR: Final[Path] = Path(__file__).resolve().parent / ".cache" / "market_data"
REQUIRED_COLUMNS: Final[list[str]] = ["Open", "High", "Low", "Close", "Volume"]


class DataError(ValueError):
    """Raised when historical market data cannot be loaded or validated."""


def _cache_path(ticker: str, start_date: str, end_date: str) -> Path:
    """Return the CSV cache path for a ticker and date range."""
    safe_ticker = ticker.strip().upper().replace("/", "_")
    return CACHE_DIR / f"{safe_ticker}_{start_date}_{end_date}.csv"


def _validate_data_frame(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Validate the fetched data and raise a clear error if it is unusable."""
    if data.empty:
        raise DataError(f"No daily price data was found for {ticker.upper()}. Check the ticker symbol and date range.")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(-1)

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing_columns:
        raise DataError(f"The data for {ticker.upper()} is missing required columns: {', '.join(missing_columns)}.")

    data = data.dropna(subset=REQUIRED_COLUMNS).copy()
    if data.empty:
        raise DataError(f"The data for {ticker.upper()} contains no complete OHLCV rows in the requested range.")

    data.index = pd.to_datetime(data.index).tz_localize(None)
    data = data[~data.index.duplicated(keep="first")].sort_index()
    data.index.name = "Date"
    return data


def fetch_ohlcv_data(ticker: str, start_date: str, end_date: str, use_cache: bool = True) -> pd.DataFrame:
    """Fetch daily OHLCV data from Yahoo Finance and cache it as CSV."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_path(ticker, start_date, end_date)

    if use_cache and cache_path.exists():
        cached = pd.read_csv(cache_path, parse_dates=["Date"], index_col="Date")
        return _validate_data_frame(cached, ticker)

    download_end = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    ticker_symbol = ticker.strip().upper()
    data = yf.download(
        ticker_symbol,
        start=start_date,
        end=download_end.strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=False,
    )

    if data.empty:
        data = yf.Ticker(ticker_symbol).history(
            start=start_date,
            end=download_end.strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=False,
        )

    if data.empty:
        raise DataError(
            f"No daily price data was found for {ticker_symbol}. Yahoo Finance is not returning rows in this environment, so the app cannot run a live backtest right now."
        )

    data = _validate_data_frame(data, ticker_symbol)
    data.to_csv(cache_path)
    return data


def fetch_daily_ohlcv(ticker: str, start_date: pd.Timestamp, end_date: pd.Timestamp, cache_dir: str | Path | None = None) -> pd.DataFrame:
    """Fetch daily OHLCV data using timestamp inputs for the Streamlit app."""

    _ = cache_dir
    return fetch_ohlcv_data(ticker, str(pd.Timestamp(start_date).date()), str(pd.Timestamp(end_date).date()))
