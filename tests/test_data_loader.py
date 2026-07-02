"""Unit tests for market data loading helpers."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data_loader import _validate_data_frame


class DataLoaderTests(unittest.TestCase):
    """Check validation behavior for common provider response shapes."""

    def test_validate_flattens_single_ticker_yfinance_columns(self) -> None:
        index = pd.date_range("2024-01-01", periods=2, name="Date")
        columns = pd.MultiIndex.from_product(
            [["Open", "High", "Low", "Close", "Volume"], ["AAPL"]],
            names=["Price", "Ticker"],
        )
        data = pd.DataFrame(
            [
                [10.0, 11.0, 9.5, 10.5, 1000],
                [10.5, 12.0, 10.0, 11.5, 1500],
            ],
            index=index,
            columns=columns,
        )

        validated = _validate_data_frame(data, "AAPL")

        self.assertEqual(["Open", "High", "Low", "Close", "Volume"], list(validated.columns))
        self.assertEqual("Date", validated.index.name)


if __name__ == "__main__":
    unittest.main()
