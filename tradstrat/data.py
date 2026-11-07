"""Load, validate and cache daily OHLCV market data."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Final
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import pandas as pd
import yfinance as yf

CACHE_DIR: Final[Path] = Path(os.environ.get("TRADSTRAT_CACHE_DIR", Path.home() / ".cache" / "tradstrat"))
SAMPLE_PATH: Final[Path] = Path(__file__).resolve().parent / "sample_data" / "synthetic_ohlcv.csv"
PRICE_COLUMNS: Final[list[str]] = ["Open", "High", "Low", "Close"]
REQUIRED_COLUMNS: Final[list[str]] = [*PRICE_COLUMNS, "Volume"]
TICKER_PATTERN: Final[re.Pattern[str]] = re.compile(r"[A-Z0-9.\-^=]{1,15}")


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

    raise DataError("Provider returned columns that do not include OHLCV data.")


def validate_ohlcv(data: pd.DataFrame, label: str) -> pd.DataFrame:
    """Return a clean frame: numeric OHLCV, unique ascending dates, coherent prices."""
    data = _flatten_columns(data)
    if data.empty:
        raise DataError(f"No daily price data was found for {label}. Check the symbol and date range.")

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing_columns:
        raise DataError(f"The data for {label} is missing required columns: {', '.join(missing_columns)}.")

    columns = [*REQUIRED_COLUMNS, *(["Adj Close"] if "Adj Close" in data.columns else [])]
    data = data[columns].apply(pd.to_numeric, errors="coerce").dropna(subset=REQUIRED_COLUMNS)
    data = data[(data[PRICE_COLUMNS] > 0).all(axis=1)]
    if data.empty:
        raise DataError(f"The data for {label} contains no complete OHLCV rows in the requested range.")

    # Providers disagree on timestamps (midnight, market open in UTC, exchange-local). A daily bar is
    # identified by its date alone, and two bars on one date would break the one-bar execution delay.
    index = pd.DatetimeIndex(pd.to_datetime(data.index))
    data.index = (index.tz_localize(None) if index.tz is not None else index).normalize()
    data = data[~data.index.duplicated(keep="first")].sort_index()
    data.index.name = "Date"

    # Stop orders are tested against High and Low, so they must bracket the open and close.
    prices = data[PRICE_COLUMNS]
    data["High"], data["Low"] = prices.max(axis=1), prices.min(axis=1)
    return data


def adjust_for_dividends(data: pd.DataFrame) -> pd.DataFrame:
    """Scale OHLC by the Adj Close ratio so that returns include dividends.

    Yahoo's raw Close already reflects splits but not dividends. Using it directly understates the return
    of holding the stock, and with it the buy-and-hold benchmark that every strategy is compared against.
    """
    if "Adj Close" not in data.columns:
        return data
    data = data.dropna(subset=["Adj Close"]).copy()
    factor = data["Adj Close"] / data["Close"]
    data[PRICE_COLUMNS] = data[PRICE_COLUMNS].mul(factor, axis=0)
    return data.drop(columns="Adj Close")


def load_csv(path: str | Path, adjust: bool = True) -> pd.DataFrame:
    """Load OHLCV rows from a CSV file with a ``Date`` column."""
    path = Path(path)
    try:
        raw = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise DataError(f"Could not read {path}: {exc}") from exc
    data = validate_ohlcv(raw, path.name)
    return adjust_for_dividends(data) if adjust else data


def load_sample() -> pd.DataFrame:
    """Load the bundled synthetic dataset, which needs no network access."""
    return load_csv(SAMPLE_PATH)


def _fetch_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch OHLCV rows through yfinance."""
    return yf.download(
        ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=False,
    )


def _fetch_yahoo_chart(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch OHLCV rows straight from Yahoo's chart endpoint, for when yfinance is failing."""
    params = urlencode(
        {
            "period1": int(pd.Timestamp(start).timestamp()),
            "period2": int(pd.Timestamp(end).timestamp()),
            "interval": "1d",
            "events": "history",
            "includeAdjustedClose": "true",
        }
    )
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(ticker, safe='')}?{params}"
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})

    with urlopen(request, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))

    chart = payload.get("chart", {})
    if chart.get("error"):
        raise DataError(str(chart["error"].get("description") or chart["error"]))

    results = chart.get("result") or []
    if not results:
        return pd.DataFrame()

    result = results[0]
    timestamps = result.get("timestamp") or []
    indicators = result.get("indicators") or {}
    quote_rows = (indicators.get("quote") or [{}])[0]
    if not timestamps or not quote_rows:
        return pd.DataFrame()

    columns = {name: quote_rows.get(name.lower()) for name in REQUIRED_COLUMNS}
    adjusted_close = (indicators.get("adjclose") or [{}])[0].get("adjclose")
    if adjusted_close:
        columns["Adj Close"] = adjusted_close
    return pd.DataFrame(columns, index=pd.to_datetime(timestamps, unit="s"))


Provider = Callable[[str, str, str], pd.DataFrame]
PROVIDERS: tuple[tuple[str, Provider], ...] = (
    ("yfinance", _fetch_yfinance),
    ("Yahoo chart API", _fetch_yahoo_chart),
)


def normalize_ticker(ticker: str) -> str:
    """Upper-case a ticker and reject anything that is not a plausible market symbol.

    The symbol ends up in a URL and a cache filename, so it is checked against an allowlist.
    """
    symbol = ticker.strip().upper()
    if not symbol:
        raise DataError("Please enter a ticker symbol.")
    if not TICKER_PATTERN.fullmatch(symbol):
        raise DataError(f"{ticker!r} is not a valid ticker symbol.")
    return symbol


def _read_cache(path: Path, label: str) -> pd.DataFrame | None:
    """Return cached rows, or None when the file is missing or unreadable."""
    if not path.exists():
        return None
    try:
        return validate_ohlcv(pd.read_csv(path, parse_dates=["Date"], index_col="Date"), label)
    except (ValueError, pd.errors.ParserError):
        return None  # A corrupt cache entry should cost one refetch, not a failed run.


def _write_cache(path: Path, data: pd.DataFrame) -> None:
    """Write through a temporary file so an interrupted run cannot leave a half-written cache entry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    data.to_csv(temporary)
    temporary.replace(path)


def fetch_ohlcv(
    ticker: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    *,
    use_cache: bool = True,
    adjust: bool = True,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Fetch daily OHLCV data for an inclusive date range, trying each provider in turn."""
    symbol = normalize_ticker(ticker)
    start_date = pd.Timestamp(start).normalize()
    end_date = pd.Timestamp(end).normalize()
    if start_date > end_date:
        raise DataError("Start date must be before or equal to end date.")

    cache_path = (cache_dir or CACHE_DIR) / f"{symbol}_{start_date.date()}_{end_date.date()}.csv"
    data = _read_cache(cache_path, symbol) if use_cache else None

    if data is None:
        errors: list[str] = []
        # Providers return exclusive end dates, so ask for one extra day.
        request_end = str((end_date + pd.Timedelta(days=1)).date())
        for name, provider in PROVIDERS:
            try:
                data = validate_ohlcv(provider(symbol, str(start_date.date()), request_end), symbol)
                break
            # Deliberately broad: yfinance surfaces rate limits, JSON errors and KeyErrors from its own
            # internals, and any of them should move on to the next provider rather than end the run.
            except Exception as exc:
                errors.append(f"{name}: {exc}")
        if data is None:
            raise DataError(
                f"No daily price data was found for {symbol} from {start_date.date()} to {end_date.date()}. "
                f"Check the symbol and date range. Provider details: {' | '.join(errors)}"
            )
        # A range that includes today can still gain bars, so caching it would freeze a partial result.
        if use_cache and end_date < pd.Timestamp.today().normalize():
            _write_cache(cache_path, data)

    data = data.loc[start_date:end_date]
    if data.empty:
        raise DataError(f"No daily price data was found for {symbol} in the requested range.")
    return adjust_for_dividends(data) if adjust else data
