"""Check that the test suite can actually fail.

Passing tests prove little unless they would fail on broken code. This script injects realistic bugs one
at a time, including the classic backtesting mistakes, and confirms the suite catches each of them. Source
files are restored after every run.

Usage: python scripts/mutation_check.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEARCH_LINE = "scored = _score_grid(data.iloc[train_start:test_start], key, grid, config, objective)"

# Each entry is (file, original code, broken code, description).
MUTATIONS = [
    (
        "engine.py",
        "target = signal[bar - 1]",
        "target = signal[bar]",
        "Look-ahead: trade on the same bar's signal",
    ),
    (
        "engine.py",
        "estimate = volatility[bar - 1]",
        "estimate = volatility[bar]",
        "Look-ahead: size on the same bar's close",
    ),
    (
        "strategies.py",
        'close = data["Close"]\n        short_ma',
        'close = data["Close"].shift(-1)\n        short_ma',
        "Look-ahead: strategy reads tomorrow's close",
    ),
    (
        "strategies.py",
        "z_score = (close - rolling.mean()) / rolling.std(ddof=0)",
        "z_score = (close - close.mean()) / close.std(ddof=0)",
        "Look-ahead: z-score uses the full sample",
    ),
    (
        "validation.py",
        SEARCH_LINE,
        SEARCH_LINE.replace("train_start:test_start", "train_start:test_end"),
        "Leak: tune on the fold's own test window",
    ),
    (
        "validation.py",
        SEARCH_LINE,
        "scored = _score_grid(data, key, grid, config, objective)",
        "Leak: tune on all of the data",
    ),
    (
        "engine.py",
        "self.cash += self.shares * fill - commission",
        "self.cash += self.shares * fill",
        "Accounting: exit commission not charged",
    ),
    (
        "engine.py",
        "fill = price * (1 - self.slippage)",
        "fill = price * (1 + self.slippage)",
        "Costs: slippage helps on a sale",
    ),
    (
        "engine.py",
        'min(opens[bar], stop_price), "stop_loss"',
        'stop_price, "stop_loss"',
        "Execution: stop fills through a gap",
    ),
    (
        "engine.py",
        "                stopped_out = True\n",
        "                stopped_out = False\n",
        "Execution: re-entry straight after a stop",
    ),
    (
        "engine.py",
        "config.for_benchmark(), start)[0]",
        "config, start)[0]",
        "Benchmark inherits the strategy's risk settings",
    ),
    (
        "engine.py",
        "baseline = max(start - 1, 0)\n    bars =",
        "baseline = start\n    bars =",
        "First traded bar's return is lost",
    ),
    (
        "validation.py",
        "replace(config, initial_capital=capital),\n            start=",
        "config,\n            start=",
        "Capital does not compound across folds",
    ),
    (
        "validation.py",
        "segment.results.iloc[1:] for segment in segments[1:]",
        "segment.results for segment in segments[1:]",
        "Duplicate rows where folds join",
    ),
    (
        "strategies.py",
        ".rolling(self.entry_window, min_periods=self.entry_window).max().shift(1)",
        ".rolling(self.entry_window, min_periods=self.entry_window).max()",
        "Donchian channel includes the current bar",
    ),
    (
        "metrics.py",
        "(excess.clip(upper=0.0) ** 2).mean()",
        "(excess[excess < 0] ** 2).mean()",
        "Sortino averages over losing days only",
    ),
    (
        "data.py",
        "data[PRICE_COLUMNS] = data[PRICE_COLUMNS].mul(factor, axis=0)",
        'data[["Close"]] = data[["Close"]].mul(factor, axis=0)',
        "Dividend adjustment skips open, high and low",
    ),
    (
        "stats.py",
        "first = NORMAL.inv_cdf(1 - 1 / trials)",
        "first = NORMAL.inv_cdf(1 - 1 / (2 * trials))",
        "Wrong deflated Sharpe threshold",
    ),
    ("stats.py", "(kurtosis - 1) / 4", "(kurtosis - 3) / 4", "Sharpe standard error uses excess kurtosis"),
    ("stats.py", "% observations\n", "\n", "Bootstrap blocks do not wrap around"),
]


def main() -> int:
    survivors = 0
    for filename, original, broken, description in MUTATIONS:
        path = ROOT / "tradstrat" / filename
        source = path.read_text()
        if original not in source:
            print(f"STALE     {description}: the original code was not found")
            survivors += 1
            continue
        path.write_text(source.replace(original, broken, 1))
        try:
            outcome = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
        finally:
            path.write_text(source)
        caught = outcome.returncode != 0
        survivors += not caught
        print(f"{'caught  ' if caught else 'SURVIVED'}  {description}")

    print(f"\n{len(MUTATIONS) - survivors} of {len(MUTATIONS)} injected bugs were caught.")
    return 1 if survivors else 0


if __name__ == "__main__":
    sys.exit(main())
