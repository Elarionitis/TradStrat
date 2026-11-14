"""Tests for market data loading. Provider calls are stubbed, so nothing here touches the network."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.request import Request

import numpy as np
import pandas as pd
import pytest

from tradstrat import data as data_module
from tradstrat.data import (
    DataError,
    _fetch_yahoo_chart,
    _fetch_yfinance,
    adjust_for_dividends,
    attach_cash_rate,
    fetch_cash_rate,
    fetch_ohlcv,
    load_csv,
    load_sample,
    normalize_ticker,
    read_price_csv,
    validate_ohlcv,
)


def bars(rows: int = 3, start: str = "2024-01-02") -> pd.DataFrame:
    close = 100.0 + np.arange(rows)
    return pd.DataFrame(
        {"Open": close, "High": close + 1, "Low": close - 1, "Close": close, "Volume": 1000},
        index=pd.date_range(start, periods=rows, freq="B"),
    )


def stub_provider(monkeypatch: pytest.MonkeyPatch, frame: pd.DataFrame) -> list[tuple[str, str, str]]:
    calls: list[tuple[str, str, str]] = []

    def provider(ticker: str, start: str, end: str) -> pd.DataFrame:
        calls.append((ticker, start, end))
        return frame

    monkeypatch.setattr(data_module, "PROVIDERS", (("stub", provider),))
    return calls


def test_flattens_yfinance_multiindex_columns() -> None:
    frame = bars()
    frame.columns = pd.MultiIndex.from_product([frame.columns, ["AAPL"]], names=["Price", "Ticker"])

    assert list(validate_ohlcv(frame, "AAPL").columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_timestamps_are_normalized_to_dates() -> None:
    """Regression: Yahoo's chart API stamps bars at 13:30 UTC, which leaked into the index."""
    frame = bars()
    frame.index = frame.index + pd.Timedelta(hours=13, minutes=30)

    assert (validate_ohlcv(frame, "X").index == frame.index.normalize()).all()


def test_timezone_aware_index_keeps_the_exchange_date() -> None:
    frame = bars()
    frame.index = frame.index.tz_localize("America/New_York")

    validated = validate_ohlcv(frame, "X")

    assert validated.index.tz is None
    assert validated.index[0] == pd.Timestamp("2024-01-02")


def test_duplicate_dates_are_removed_and_order_is_restored() -> None:
    frame = bars()
    shuffled = pd.concat([frame.iloc[[2, 0]], frame.iloc[[0]], frame.iloc[[1]]])

    validated = validate_ohlcv(shuffled, "X")

    assert validated.index.is_monotonic_increasing
    assert validated.index.is_unique


def test_rows_with_missing_or_non_positive_prices_are_dropped() -> None:
    frame = bars(4)
    frame.loc[frame.index[1], "Close"] = np.nan
    frame.loc[frame.index[2], "Open"] = 0.0

    assert len(validate_ohlcv(frame, "X")) == 2


def test_high_and_low_always_bracket_open_and_close() -> None:
    frame = bars(1)
    frame["High"] = 90.0
    frame["Low"] = 120.0

    validated = validate_ohlcv(frame, "X")

    assert validated["High"].iloc[0] == 120.0
    assert validated["Low"].iloc[0] == 90.0


@pytest.mark.parametrize("frame", [pd.DataFrame(), bars().drop(columns="Close")])
def test_unusable_frames_raise_a_clear_error(frame: pd.DataFrame) -> None:
    with pytest.raises(DataError):
        validate_ohlcv(frame, "X")


def test_dividend_adjustment_scales_prices_and_captures_total_return() -> None:
    frame = bars(2)
    frame["Close"] = [100.0, 100.0]
    frame["Open"] = [100.0, 100.0]
    frame["Adj Close"] = [98.0, 100.0]  # A $2 dividend paid between the two bars.

    adjusted = adjust_for_dividends(frame)

    assert "Adj Close" not in adjusted.columns
    assert adjusted["Close"].iloc[0] == pytest.approx(98.0)
    assert adjusted["Open"].iloc[0] == pytest.approx(98.0)
    assert adjusted["Close"].iloc[1] == pytest.approx(100.0)


def test_adjustment_is_a_no_op_without_adj_close() -> None:
    frame = bars()

    pd.testing.assert_frame_equal(adjust_for_dividends(frame), frame)


@pytest.mark.parametrize("ticker", ["aapl", " BRK-B ", "^GSPC", "EURUSD=X", "BTC-USD"])
def test_real_ticker_formats_are_accepted(ticker: str) -> None:
    assert normalize_ticker(ticker) == ticker.strip().upper()


@pytest.mark.parametrize("ticker", ["", "   ", "../../etc/passwd", "AAPL?x=1", "A/B", "A" * 16])
def test_malformed_tickers_are_rejected(ticker: str) -> None:
    with pytest.raises(DataError):
        normalize_ticker(ticker)


def test_reversed_dates_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="before"):
        fetch_ohlcv("AAPL", "2024-02-01", "2024-01-01", cache_dir=tmp_path)


def test_second_fetch_of_a_closed_range_comes_from_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = stub_provider(monkeypatch, bars())

    first = fetch_ohlcv("AAPL", "2024-01-02", "2024-01-04", cache_dir=tmp_path)
    second = fetch_ohlcv("AAPL", "2024-01-02", "2024-01-04", cache_dir=tmp_path)

    assert len(calls) == 1
    pd.testing.assert_frame_equal(first, second, check_freq=False, check_dtype=False)


def test_range_ending_today_is_not_cached(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A range that includes today can still gain bars, so caching would freeze a partial day."""
    calls = stub_provider(monkeypatch, bars(start=str(pd.Timestamp.today().date())))
    today = pd.Timestamp.today().normalize()

    fetch_ohlcv("AAPL", today - pd.Timedelta(days=5), today + pd.Timedelta(days=5), cache_dir=tmp_path)
    fetch_ohlcv("AAPL", today - pd.Timedelta(days=5), today + pd.Timedelta(days=5), cache_dir=tmp_path)

    assert len(calls) == 2


def test_corrupt_cache_triggers_a_refetch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = stub_provider(monkeypatch, bars())
    (tmp_path / "AAPL_2024-01-02_2024-01-04.csv").write_text("not,a,valid\ncache,file,here\n")

    data = fetch_ohlcv("AAPL", "2024-01-02", "2024-01-04", cache_dir=tmp_path)

    assert len(calls) == 1
    assert len(data) == 3


def test_provider_end_date_is_exclusive_so_one_day_is_added(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = stub_provider(monkeypatch, bars())

    fetch_ohlcv("AAPL", "2024-01-02", "2024-01-04", cache_dir=tmp_path, use_cache=False)

    assert calls == [("AAPL", "2024-01-02", "2024-01-05")]


def test_a_failing_provider_falls_through_to_the_next(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def broken(*_: str) -> pd.DataFrame:
        raise KeyError("rate limited")

    monkeypatch.setattr(data_module, "PROVIDERS", (("broken", broken), ("working", lambda *_: bars())))

    assert len(fetch_ohlcv("AAPL", "2024-01-02", "2024-01-04", cache_dir=tmp_path)) == 3


def test_a_provider_returning_bad_data_falls_through_to_the_next(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression: validation ran outside the fallback loop, so bad data ended the run."""
    monkeypatch.setattr(
        data_module,
        "PROVIDERS",
        (("bad", lambda *_: bars().drop(columns="Close")), ("working", lambda *_: bars())),
    )

    assert len(fetch_ohlcv("AAPL", "2024-01-02", "2024-01-04", cache_dir=tmp_path)) == 3


def test_all_providers_failing_reports_each_reason(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def broken(*_: str) -> pd.DataFrame:
        raise OSError("connection refused")

    monkeypatch.setattr(data_module, "PROVIDERS", (("alpha", broken), ("beta", broken)))

    with pytest.raises(DataError, match=r"alpha: connection refused \| beta: connection refused"):
        fetch_ohlcv("AAPL", "2024-01-02", "2024-01-04", cache_dir=tmp_path)


def test_load_csv_reports_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="Could not read"):
        load_csv(tmp_path / "missing.csv")


def test_sample_dataset_looks_like_a_market() -> None:
    """Regression: the old sample was a smooth curve with lag-1 autocorrelation of 0.997."""
    sample = load_sample()
    returns = sample["Close"].pct_change().dropna()

    assert len(sample) > 2000
    assert abs(returns.autocorr(1)) < 0.1
    assert 0.10 < returns.std() * np.sqrt(252) < 0.40


class FakeResponse:
    """Stands in for the object urlopen returns."""

    def __init__(self, payload: dict) -> None:
        self.body = json.dumps(payload).encode()

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


def stub_chart_api(monkeypatch: pytest.MonkeyPatch, payload: dict) -> list[str]:
    urls: list[str] = []

    def fake_urlopen(request: Request, timeout: int) -> FakeResponse:
        urls.append(request.full_url)
        return FakeResponse(payload)

    monkeypatch.setattr(data_module, "urlopen", fake_urlopen)
    return urls


CHART_PAYLOAD = {
    "chart": {
        "error": None,
        "result": [
            {
                "timestamp": [1704202200, 1704288600],  # 2024-01-02 and 2024-01-03 at 13:30 UTC
                "indicators": {
                    "quote": [
                        {
                            "open": [10.0, 11.0],
                            "high": [11.0, 12.0],
                            "low": [9.0, 10.0],
                            "close": [10.5, 11.5],
                            "volume": [100, 200],
                        }
                    ],
                    "adjclose": [{"adjclose": [10.0, 11.5]}],
                },
            }
        ],
    }
}


def test_chart_api_response_is_parsed_into_valid_bars(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_chart_api(monkeypatch, CHART_PAYLOAD)

    bars_ = validate_ohlcv(_fetch_yahoo_chart("AAPL", "2024-01-02", "2024-01-04"), "AAPL")

    assert list(bars_.index) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    assert bars_["Close"].tolist() == [10.5, 11.5]
    assert bars_["Adj Close"].tolist() == [10.0, 11.5]


def test_chart_api_escapes_the_symbol_in_its_url(monkeypatch: pytest.MonkeyPatch) -> None:
    urls = stub_chart_api(monkeypatch, CHART_PAYLOAD)

    _fetch_yahoo_chart("^GSPC", "2024-01-02", "2024-01-04")

    assert "/chart/%5EGSPC?" in urls[0]


def test_chart_api_error_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_chart_api(monkeypatch, {"chart": {"error": {"description": "No data found"}, "result": None}})

    with pytest.raises(DataError, match="No data found"):
        _fetch_yahoo_chart("NOPE", "2024-01-02", "2024-01-04")


@pytest.mark.parametrize(
    "payload",
    [
        {"chart": {"result": []}},
        {"chart": {"result": [{"timestamp": [], "indicators": {}}]}},
        {},
    ],
)
def test_empty_chart_api_responses_yield_no_rows(monkeypatch: pytest.MonkeyPatch, payload: dict) -> None:
    stub_chart_api(monkeypatch, payload)

    assert _fetch_yahoo_chart("AAPL", "2024-01-02", "2024-01-04").empty


def test_yfinance_is_asked_for_unadjusted_prices(monkeypatch: pytest.MonkeyPatch) -> None:
    """Adjustment happens in one place, so yfinance must return both Close and Adj Close."""
    captured: dict = {}

    def fake_download(ticker: str, **kwargs: object) -> pd.DataFrame:
        captured.update(kwargs, ticker=ticker)
        return bars()

    monkeypatch.setattr(data_module.yf, "download", fake_download)

    _fetch_yfinance("AAPL", "2024-01-02", "2024-01-04")

    assert captured["ticker"] == "AAPL"
    assert captured["auto_adjust"] is False


def test_all_history_does_not_start_at_the_epoch(monkeypatch: pytest.MonkeyPatch) -> None:
    """yfinance treats a start of timestamp 0 as missing and returns only a month, so it is moved a day on."""
    captured: dict = {}

    def fake_download(ticker: str, **kwargs: object) -> pd.DataFrame:
        captured.update(kwargs)
        return bars()

    monkeypatch.setattr(data_module.yf, "download", fake_download)

    _fetch_yfinance("BTC-USD", "1970-01-01", "2026-01-01")
    assert captured["start"] == "1970-01-02"

    _fetch_yfinance("BTC-USD", "2020-03-04", "2026-01-01")
    assert captured["start"] == "2020-03-04"


def test_provider_errors_are_kept_apart_from_the_message(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*_: object) -> None:
        raise ConnectionError("rate limited")

    monkeypatch.setattr(data_module, "PROVIDERS", (("alpha", broken),))

    with pytest.raises(DataError) as caught:
        fetch_ohlcv("SPY", "2024-01-02", "2024-01-31", use_cache=False)

    assert "rate limited" not in caught.value.message
    assert "alpha: rate limited" in caught.value.details
    assert "rate limited" in str(caught.value)


def test_unrecognized_multiindex_columns_are_rejected() -> None:
    frame = bars()
    frame.columns = pd.MultiIndex.from_product([["AAPL"], list("abcde")])

    with pytest.raises(DataError, match="OHLCV"):
        validate_ohlcv(frame, "AAPL")


def test_rows_that_are_all_invalid_raise_a_clear_error() -> None:
    frame = bars()
    frame["Close"] = np.nan

    with pytest.raises(DataError, match="no complete"):
        validate_ohlcv(frame, "AAPL")


def test_a_fetch_outside_the_returned_range_is_an_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stub_provider(monkeypatch, bars(start="2020-01-01"))

    with pytest.raises(DataError, match="requested range"):
        fetch_ohlcv("AAPL", "2024-01-02", "2024-01-04", cache_dir=tmp_path, use_cache=False)


# Real-world CSV exports and the cash rate.


def test_csv_column_names_are_matched_case_insensitively(tmp_path: Path) -> None:
    path = tmp_path / "export.csv"
    path.write_text(
        "timestamp,open,high,low,close,volume,adj_close\n"
        "2024-01-02,10,11,9,10,100,5\n"
        "2024-01-03,10,12,10,11,100,5.5\n"
    )

    data = load_csv(path)

    assert list(data.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert data["Close"].tolist() == [5.0, 5.5]  # Scaled by the adjusted close.


def test_csv_without_volume_is_accepted(tmp_path: Path) -> None:
    path = tmp_path / "index.csv"
    path.write_text("Date,Open,High,Low,Close\n2024-01-02,10,11,9,10\n2024-01-03,10,12,10,11\n")

    assert (load_csv(path)["Volume"] == 0).all()


@pytest.mark.parametrize("scale", [1, 1000])
def test_csv_unix_timestamps_in_seconds_or_milliseconds_are_parsed(tmp_path: Path, scale: int) -> None:
    seconds = int(pd.Timestamp("2024-01-02").timestamp())
    path = tmp_path / "crypto.csv"
    path.write_text(f"time,open,high,low,close\n{seconds * scale},10,11,9,10\n")

    assert load_csv(path).index[0] == pd.Timestamp("2024-01-02")


def test_uploaded_file_objects_can_be_read() -> None:
    import io

    upload = io.BytesIO(b"Date,Open,High,Low,Close\n2024-01-02,10,11,9,10\n")

    assert len(read_price_csv(upload)) == 1


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("when,open,high,low,close\n2024-01-02,1,1,1,1\n", "date column"),
        ("date,open,high,low,close\nnot a date,1,1,1,1\n", "parse the dates"),
    ],
)
def test_csv_problems_get_a_helpful_message(tmp_path: Path, contents: str, message: str) -> None:
    path = tmp_path / "bad.csv"
    path.write_text(contents)

    with pytest.raises(DataError, match=message):
        load_csv(path)


def test_a_fixed_cash_rate_fills_every_bar() -> None:
    assert (attach_cash_rate(bars(), 0.04)["Cash Rate"] == 0.04).all()


def test_a_cash_rate_series_carries_the_latest_published_rate_forward() -> None:
    prices = bars(4)  # 2024-01-02 to 2024-01-05
    rates = pd.Series([0.05, 0.03], index=pd.to_datetime(["2023-12-29", "2024-01-04"]))

    attached = attach_cash_rate(prices, rates)["Cash Rate"]

    assert attached.tolist() == [0.05, 0.05, 0.03, 0.03]


def test_cash_rates_in_percent_are_rejected() -> None:
    with pytest.raises(DataError, match="annual decimals"):
        attach_cash_rate(bars(), 5.0)


def test_cash_rate_comes_from_treasury_bill_yields_in_percent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    yields = bars(3).assign(Close=[5.0, 5.1, 5.2])
    calls = stub_provider(monkeypatch, yields)

    rates = fetch_cash_rate("2024-01-02", "2024-01-04", cache_dir=tmp_path)

    assert calls[0][0] == "^IRX"
    assert calls[0][1] == "2023-12-02"  # A month early, so the first bar already has a rate.
    assert rates.tolist() == pytest.approx([0.05, 0.051, 0.052])
