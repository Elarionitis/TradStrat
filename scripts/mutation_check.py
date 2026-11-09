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
SEARCH_LINE = "scored = _score_grid(data.iloc[train_start:test_start], key, grid, config, objective).table"

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
        "scored = _score_grid(data, key, grid, config, objective).table",
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
        "min(opens[bar], stop_price), reason",
        "stop_price, reason",
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
        "config.for_benchmark(), start).equity",
        "config, start).equity",
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
    (
        "engine.py",
        "returns[1:] = rates[:-1] * days / DAYS_PER_YEAR",
        "returns[1:] = rates[1:] * days / DAYS_PER_YEAR",
        "Look-ahead: interest accrues at a rate published later",
    ),
    (
        "engine.py",
        "anchor = book.peak_price if config.trailing_stop",
        "anchor = max(book.peak_price, highs[bar]) if config.trailing_stop",
        "Look-ahead: trailing stop uses the bar's own high",
    ),
    (
        "engine.py",
        "                book.cash += interest[step]\n",
        "                pass\n",
        "Accounting: interest is reported but never credited",
    ),
    (
        "engine.py",
        'self.results["Position"],\n            risk_free=self.risk_free_returns,',
        'self.results["Position"],',
        "Sharpe ratio credits the cash rate as skill",
    ),
    (
        "stats.py",
        "best = np.argmax(np.where(np.isnan(in_sample), -np.inf, in_sample), axis=1)",
        "best = np.argmax(np.where(np.isnan(out_of_sample), -np.inf, out_of_sample), axis=1)",
        "Overfitting test picks its winner out of sample",
    ),
    (
        "stats.py",
        "relative_rank = rank / (configurations + 1)",
        "relative_rank = rank / configurations",
        "Overfitting test misplaces the median rank",
    ),
    (
        "rules.py",
        "return close.rolling(days, min_periods=days).mean()",
        "return close.rolling(days, min_periods=days, center=True).mean()",
        "Look-ahead: a rule's average is centered on today",
    ),
    (
        "rules.py",
        "np.concatenate([[False], (before_opposite & defined)[:-1]])",
        "np.concatenate([(before_opposite & defined)[1:], [False]])",
        "Look-ahead: a cross compares with tomorrow, not yesterday",
    ),
    (
        "rules.py",
        'data["High"].astype(float).rolling(days, min_periods=days).max().shift(1)',
        'data["High"].astype(float).rolling(days, min_periods=days).max()',
        "A rule's channel includes today, so a breakout can never fire",
    ),
    (
        "rules.py",
        "return close.pct_change(days, fill_method=None) * 100",
        "return close.pct_change(days, fill_method=None)",
        "A rule's return is a fraction, not the percent it is written in",
    ),
    (
        "rules.py",
        'combine = np.logical_and if condition.joiner == "and" else np.logical_or',
        'combine = np.logical_or if condition.joiner == "and" else np.logical_and',
        "A rule mixes up 'and' and 'or'",
    ),
    (
        "strategies.py",
        "signal.index[different[-1] + 1]",
        "signal.index[different[-1]]",
        "The latest signal's start date is a day early",
    ),
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
