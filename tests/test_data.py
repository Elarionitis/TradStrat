"""Tests for market data loading. Provider calls are stubbed, so nothing here touches the network."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tradstrat import data as data_module
from tradstrat.data import (
    DataError,
    adjust_for_dividends,
    fetch_ohlcv,
    load_csv,
    load_sample,
    normalize_ticker,
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


@pytest.mark.parametrize("frame", [pd.DataFrame(), bars().drop(columns="Volume")])
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
