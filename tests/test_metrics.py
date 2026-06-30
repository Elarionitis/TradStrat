"""Unit tests for the metrics module."""

from __future__ import annotations

import math
import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from metrics import calculate_metrics


class MetricsTests(unittest.TestCase):
    """Check the metrics logic on small, hand-checkable examples."""

    def test_basic_metrics(self) -> None:
        equity_curve = pd.Series([100.0, 102.0, 101.0, 99.0])
        trade_log = pd.DataFrame(
            {
                "P&L": [10.0, -5.0],
            }
        )

        metrics = calculate_metrics(equity_curve, trade_log)

        expected_total_return = 99.0 / 100.0 - 1
        expected_daily_returns = np.array([0.02, -0.00980392, -0.01980198])
        expected_cagr = (99.0 / 100.0) ** (252 / 3) - 1
        expected_sharpe = expected_daily_returns.mean() / expected_daily_returns.std(ddof=0) * math.sqrt(252)
        expected_sortino = expected_daily_returns.mean() / expected_daily_returns[expected_daily_returns < 0].std(ddof=0) * math.sqrt(252)

        self.assertAlmostEqual(metrics.total_return, expected_total_return, places=10)
        self.assertAlmostEqual(metrics.cagr, expected_cagr, places=10)
        self.assertAlmostEqual(metrics.sharpe_ratio, expected_sharpe, places=5)
        self.assertAlmostEqual(metrics.sortino_ratio, expected_sortino, places=5)
        self.assertAlmostEqual(metrics.max_drawdown, -0.02941176470588236, places=10)
        self.assertEqual(metrics.max_drawdown_duration, 2)
        self.assertAlmostEqual(metrics.win_rate, 0.5, places=10)
        self.assertAlmostEqual(metrics.average_win, 10.0, places=10)
        self.assertAlmostEqual(metrics.average_loss, -5.0, places=10)
        self.assertEqual(metrics.number_of_trades, 2)

    def test_empty_trade_log(self) -> None:
        equity_curve = pd.Series([100.0, 101.0])
        trade_log = pd.DataFrame(columns=["P&L"])

        metrics = calculate_metrics(equity_curve, trade_log)

        self.assertEqual(metrics.number_of_trades, 0)
        self.assertEqual(metrics.win_rate, 0.0)
        self.assertEqual(metrics.average_win, 0.0)
        self.assertEqual(metrics.average_loss, 0.0)


if __name__ == "__main__":
    unittest.main()
