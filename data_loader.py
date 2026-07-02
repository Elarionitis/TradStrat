"""Utilities for fetching and validating historical market data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
import yfinance as yf

CACHE_DIR: Final[Path] = Path(__file__).resolve().parent / ".cache" / "market_data"
REQUIRED_COLUMNS: Final[list[str]] = ["Open", "High", "Low", "Close", "Volume"]


class DataError(ValueError):
    """Raised when historical market data cannot be loaded or validated."""


def _flatten_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance MultiIndex columns while preserving OHLCV names."""

    if not isinstance(data.columns, pd.MultiIndex):
        return data

    for level in range(data.columns.nlevels):
        values = data.columns.get_level_values(level)
        if all(column in values for column in REQUIRED_COLUMNS):
            data = data.copy()
            data.columns = values
            return data

    data = data.copy()
    data.columns = [" ".join(str(part) for part in column if part) for column in data.columns.to_flat_index()]
    return data


def _cache_path(ticker: str, start_date: str, end_date: str) -> Path:
    """Return the CSV cache path for a ticker and date range."""
    safe_ticker = ticker.strip().upper().replace("/", "_")
    return CACHE_DIR / f"{safe_ticker}_{start_date}_{end_date}.csv"


def _validate_data_frame(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Validate the fetched data and raise a clear error if it is unusable."""
    data = _flatten_columns(data)

    if data.empty:
        raise DataError(f"No daily price data was found for {ticker.upper()}. Check the ticker symbol and date range.")

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


def _fetch_yahoo_download(ticker_symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch OHLCV rows through yfinance's download helper."""

    return yf.download(
        ticker_symbol,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=False,
    )


def _fetch_yahoo_history(ticker_symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch OHLCV rows through yfinance's Ticker history helper."""

    return yf.Ticker(ticker_symbol).history(
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=False,
    )


def _fetch_yahoo_chart(ticker_symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch OHLCV rows from Yahoo's chart endpoint without yfinance cookies."""

    period_start = int(pd.Timestamp(start_date).timestamp())
    period_end = int(pd.Timestamp(end_date).timestamp())
    params = urlencode(
        {
            "period1": period_start,
            "period2": period_end,
            "interval": "1d",
            "events": "history",
            "includeAdjustedClose": "true",
        }
    )
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker_symbol}?{params}"
    request = Request(url, headers={"User-Agent": "TradStrat/1.0"})

    with urlopen(request, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))

    chart = payload.get("chart", {})
    error = chart.get("error")
    if error:
        description = error.get("description") or error.get("code") or "unknown Yahoo chart error"
        raise DataError(f"Yahoo chart API rejected {ticker_symbol}: {description}.")

    results = chart.get("result") or []
    if not results:
        return pd.DataFrame()

    result = results[0]
    timestamps = result.get("timestamp") or []
    quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
    adjusted_close = ((result.get("indicators") or {}).get("adjclose") or [{}])[0].get("adjclose")

    if not timestamps or not quote:
        return pd.DataFrame()

    data = pd.DataFrame(
        {
            "Open": quote.get("open"),
            "High": quote.get("high"),
            "Low": quote.get("low"),
            "Close": quote.get("close"),
            "Volume": quote.get("volume"),
        },
        index=pd.to_datetime(timestamps, unit="s").tz_localize(None),
    )
    if adjusted_close:
        data["Adj Close"] = adjusted_close
    return data


def _format_provider_errors(errors: list[str]) -> str:
    if not errors:
        return "No provider returned rows."
    return " ".join(errors)


def fetch_ohlcv_data(ticker: str, start_date: str, end_date: str, use_cache: bool = True) -> pd.DataFrame:
    """Fetch daily OHLCV data from Yahoo Finance and cache it as CSV."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_path(ticker, start_date, end_date)
    start_timestamp = pd.Timestamp(start_date)
    end_timestamp = pd.Timestamp(end_date)

    if not ticker.strip():
        raise DataError("Please enter a ticker symbol.")
    if start_timestamp > end_timestamp:
        raise DataError("Start date must be before or equal to end date.")

    if use_cache and cache_path.exists():
        cached = pd.read_csv(cache_path, parse_dates=["Date"], index_col="Date")
        return _validate_data_frame(cached, ticker)

    download_end = end_timestamp + pd.Timedelta(days=1)
    ticker_symbol = ticker.strip().upper()
    yahoo_end = download_end.strftime("%Y-%m-%d")
    provider_errors: list[str] = []

    fetch_attempts = (
        ("Yahoo Finance download", _fetch_yahoo_download),
        ("Yahoo Finance history", _fetch_yahoo_history),
        ("Yahoo Finance chart API", _fetch_yahoo_chart),
    )
    for provider_name, fetcher in fetch_attempts:
        try:
            data = fetcher(ticker_symbol, str(start_timestamp.date()), yahoo_end)
        except (DataError, HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            provider_errors.append(f"{provider_name}: {exc}")
            continue

        if data.empty:
            provider_errors.append(f"{provider_name}: no rows returned.")
            continue

        data = _validate_data_frame(data, ticker_symbol)
        data.to_csv(cache_path)
        return data

    details = _format_provider_errors(provider_errors)
    raise DataError(
        f"No daily price data was found for {ticker_symbol} from {start_timestamp.date()} to {end_timestamp.date()}. "
        f"Check the ticker symbol and date range. Provider details: {details}"
    )


def fetch_daily_ohlcv(ticker: str, start_date: pd.Timestamp, end_date: pd.Timestamp, cache_dir: str | Path | None = None) -> pd.DataFrame:
    """Fetch daily OHLCV data using timestamp inputs for the Streamlit app."""

    _ = cache_dir
    return fetch_ohlcv_data(ticker, str(pd.Timestamp(start_date).date()), str(pd.Timestamp(end_date).date()))
