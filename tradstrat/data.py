"""Load, validate and cache daily OHLCV market data."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import IO, Final, Literal
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import pandas as pd
import yfinance as yf

CACHE_DIR: Final[Path] = Path(os.environ.get("TRADSTRAT_CACHE_DIR", Path.home() / ".cache" / "tradstrat"))
SAMPLE_PATH: Final[Path] = Path(__file__).resolve().parent / "sample_data" / "synthetic_ohlcv.csv"
PRICE_COLUMNS: Final[list[str]] = ["Open", "High", "Low", "Close"]
REQUIRED_COLUMNS: Final[list[str]] = [*PRICE_COLUMNS, "Volume"]
TICKER_PATTERN: Final[re.Pattern[str]] = re.compile(r"[A-Z0-9.\-^=]{1,15}")
CASH_RATE_TICKER: Final[str] = "^IRX"
CASH_RATE_COLUMN: Final[str] = "Cash Rate"

# Exports from brokers, exchanges and charting tools name the same fields in different ways.
COLUMN_ALIASES: Final[dict[str, str]] = {
    "open": "Open",
    "o": "Open",
    "high": "High",
    "h": "High",
    "low": "Low",
    "l": "Low",
    "close": "Close",
    "c": "Close",
    "last": "Close",
    "price": "Close",
    "volume": "Volume",
    "vol": "Volume",
    "v": "Volume",
    "adj close": "Adj Close",
    "adj_close": "Adj Close",
    "adjclose": "Adj Close",
    "adjusted close": "Adj Close",
    "adjusted_close": "Adj Close",
}
DATE_ALIASES: Final[tuple[str, ...]] = ("date", "datetime", "timestamp", "time", "day")


class DataError(ValueError):
    """Raised when historical market data cannot be loaded or validated.

    ``details`` holds technical context, such as each provider's own error, that helps a developer but would
    only confuse someone reading the dashboard.
    """

    def __init__(self, message: str, details: str = "") -> None:
        super().__init__(f"{message} {details}".strip())
        self.message = message
        self.details = details


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


def _standardize_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Rename columns such as ``close`` or ``adj_close`` to the names used throughout TradStrat."""
    renames: dict[object, str] = {}
    for column in data.columns:
        name = COLUMN_ALIASES.get(str(column).strip().lower())
        if name is not None and name not in renames.values():
            renames[column] = name
    return data.rename(columns=renames)


def validate_ohlcv(data: pd.DataFrame, label: str) -> pd.DataFrame:
    """Return a clean frame: numeric OHLCV, unique ascending dates, coherent prices.

    Column names are matched case-insensitively. Volume is optional, because index and currency data
    often have none, and it is filled with zero when absent.
    """
    data = _standardize_columns(_flatten_columns(data))
    if data.empty:
        raise DataError(f"No daily price data was found for {label}. Check the symbol and date range.")

    if "Volume" not in data.columns:
        data = data.assign(Volume=0.0)
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


def _parse_dates(values: pd.Series) -> pd.DatetimeIndex:
    """Parse a date column, including Unix timestamps in seconds or milliseconds."""
    if pd.api.types.is_numeric_dtype(values):
        unit: Literal["s", "ms"] = "ms" if values.abs().max() > 1e11 else "s"
        return pd.DatetimeIndex(pd.to_datetime(values, unit=unit))
    return pd.DatetimeIndex(pd.to_datetime(values, format="mixed"))


def read_price_csv(source: str | Path | IO[str] | IO[bytes], label: str | None = None) -> pd.DataFrame:
    """Read an OHLCV CSV with any common column naming, and return it validated but unadjusted.

    ``source`` may be a path or an open file, such as a file uploaded to the dashboard. The date column
    may be called Date, Datetime, Timestamp or Time, and may hold Unix timestamps.
    """
    name = label or (Path(source).name if isinstance(source, str | Path) else "the uploaded file")
    try:
        raw = pd.read_csv(source)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise DataError(f"Could not read {name}: {exc}") from exc

    date_column = next(
        (column for column in raw.columns if str(column).strip().lower() in DATE_ALIASES), None
    )
    if date_column is None:
        raise DataError(f"{name} needs a date column named Date, Datetime, Timestamp or Time.")
    try:
        raw.index = _parse_dates(raw.pop(date_column))
    except (ValueError, TypeError, OverflowError) as exc:
        raise DataError(f"Could not parse the dates in {name}: {exc}") from exc
    return validate_ohlcv(raw, name)


def load_csv(source: str | Path | IO[str] | IO[bytes], adjust: bool = True) -> pd.DataFrame:
    """Load OHLCV rows from a CSV file, adjusting for dividends when it has an adjusted close."""
    data = read_price_csv(source)
    return adjust_for_dividends(data) if adjust else data


def load_sample() -> pd.DataFrame:
    """Load the bundled synthetic dataset, which needs no network access."""
    return load_csv(SAMPLE_PATH)


# yfinance reads a start of exactly the Unix epoch, timestamp 0, as "no start" and returns only its default
# month of history, so "all history" requests start a day later instead.
YFINANCE_EARLIEST: Final = pd.Timestamp("1970-01-02")


def _fetch_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch OHLCV rows through yfinance."""
    return yf.download(
        ticker,
        start=max(pd.Timestamp(start), YFINANCE_EARLIEST).strftime("%Y-%m-%d"),
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


def attach_cash_rate(data: pd.DataFrame, rate: float | pd.Series) -> pd.DataFrame:
    """Add the annual yield that uninvested cash earns, as a ``Cash Rate`` column.

    ``rate`` is either a fixed annual rate, such as 0.04, or a dated series of annual rates. A series is
    carried forward onto the price dates, so each bar uses the latest rate published on or before it.
    The engine accrues interest on cash and measures Sharpe ratios against this same rate.
    """
    if isinstance(rate, pd.Series):
        rates = rate.astype(float).sort_index()
        aligned = rates.reindex(rates.index.union(data.index)).ffill().reindex(data.index)
        column = aligned.fillna(0.0)
    else:
        column = pd.Series(float(rate), index=data.index)
    if (column < -0.05).any() or (column > 0.5).any():
        raise DataError("Cash rates must be annual decimals, such as 0.04 for 4%.")
    return data.assign(**{CASH_RATE_COLUMN: column.to_numpy()})


def fetch_cash_rate(
    start: str | pd.Timestamp, end: str | pd.Timestamp, *, cache_dir: Path | None = None
) -> pd.Series:
    """Fetch the 13-week US Treasury bill yield from Yahoo (``^IRX``) as annual decimals.

    The request starts a month early, so that the first price bar already has a published rate.
    """
    early_start = pd.Timestamp(start) - pd.DateOffset(months=1)
    bills = fetch_ohlcv(CASH_RATE_TICKER, early_start, end, adjust=False, cache_dir=cache_dir)
    return (bills["Close"] / 100).rename(CASH_RATE_COLUMN)


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
                "Check the symbol and date range.",
                details=f"Provider details: {' | '.join(errors)}",
            )
        # A range that includes today can still gain bars, so caching it would freeze a partial result.
        if use_cache and end_date < pd.Timestamp.today().normalize():
            _write_cache(cache_path, data)

    data = data.loc[start_date:end_date]
    if data.empty:
        raise DataError(f"No daily price data was found for {symbol} in the requested range.")
    return adjust_for_dividends(data) if adjust else data
