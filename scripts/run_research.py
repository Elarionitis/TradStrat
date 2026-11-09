"""Run every strategy across a multi-asset universe and write docs/RESULTS.md.

Each strategy is tuned and tested by walk-forward validation, so the reported numbers are out-of-sample.
The universe is broad ETFs instead of individual stocks. Picking well-known stocks today would build
survivorship bias into the study, because those names are known only for having done well.

Usage: python scripts/run_research.py [--from-csv]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from tradstrat.data import attach_cash_rate, fetch_cash_rate, fetch_ohlcv
from tradstrat.engine import BacktestConfig
from tradstrat.strategies import STRATEGIES
from tradstrat.validation import build_grid, grid_search, walk_forward

UNIVERSE = {
    "SPY": "US large cap",
    "QQQ": "US technology",
    "IWM": "US small cap",
    "EFA": "Developed ex-US",
    "TLT": "Long Treasuries",
    "GLD": "Gold",
}
START, END = "2007-01-01", "2025-12-31"
NUMBER_WORDS = {5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}
TRAIN_BARS, TEST_BARS, GRID_POINTS = 756, 252, 5
CONFIG = BacktestConfig(commission_rate=0.001, slippage_bps=5.0)
DOCS = Path(__file__).resolve().parents[1] / "docs"
DETAIL_COLUMNS = [
    "Asset",
    "Strategy",
    "Mean In-Sample Sharpe",
    "OOS Sharpe",
    "B&H Sharpe",
    "OOS CAGR",
    "B&H CAGR",
    "OOS Max Drawdown",
    "B&H Max Drawdown",
    "Deflated Sharpe vs Zero",
    "Deflated Sharpe vs B&H",
    "Probability of Overfitting",
]


def evaluate(ticker: str, key: str, prices: pd.DataFrame) -> dict[str, object]:
    grid = build_grid(key, GRID_POINTS)
    search = grid_search(prices, key, grid, CONFIG)
    forward = walk_forward(prices, key, grid, CONFIG, train_bars=TRAIN_BARS, test_bars=TEST_BARS)
    return {
        "Asset": ticker,
        "Strategy": STRATEGIES[key].label,
        "Trials": len(grid),
        "Full-Sample Best Sharpe": search.table["sharpe_ratio"].max(),
        "Deflated Sharpe vs Zero": search.deflated_sharpe,
        "Deflated Sharpe vs B&H": search.deflated_active_sharpe,
        "Probability of Overfitting": (
            search.overfitting.probability if search.overfitting is not None else float("nan")
        ),
        "Mean In-Sample Sharpe": forward.folds["In-Sample Score"].mean(),
        "OOS Sharpe": forward.metrics.sharpe_ratio,
        "B&H Sharpe": forward.benchmark_metrics.sharpe_ratio,
        "OOS CAGR": forward.metrics.cagr,
        "B&H CAGR": forward.benchmark_metrics.cagr,
        "OOS Max Drawdown": forward.metrics.max_drawdown,
        "B&H Max Drawdown": forward.benchmark_metrics.max_drawdown,
        "OOS Trades": forward.metrics.number_of_trades,
        "Exposure": forward.metrics.exposure,
    }


def _summary(table: pd.DataFrame) -> dict[str, object]:
    return {
        "runs": len(table),
        "beat_sharpe": int((table["OOS Sharpe"] > table["B&H Sharpe"]).sum()),
        "beat_cagr": int((table["OOS CAGR"] > table["B&H CAGR"]).sum()),
        "shallower": int((table["OOS Max Drawdown"] > table["B&H Max Drawdown"]).sum()),
        "in_sample": table["Mean In-Sample Sharpe"].mean(),
        "out_of_sample": table["OOS Sharpe"].mean(),
        "benchmark": table["B&H Sharpe"].mean(),
        "decayed": int((table["Mean In-Sample Sharpe"] > table["OOS Sharpe"]).sum()),
        "significant_zero": int((table["Deflated Sharpe vs Zero"] >= 0.95).sum()),
        "significant_active": int((table["Deflated Sharpe vs B&H"] >= 0.95).sum()),
        "best_active": table["Deflated Sharpe vs B&H"].max(),
        "overfitting": table["Probability of Overfitting"].mean(),
        "overfit_runs": int((table["Probability of Overfitting"] > 0.5).sum()),
    }


def _markdown_table(frame: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(frame.columns) + " |", "|" + " --- |" * len(frame.columns)]
    lines += ["| " + " | ".join(str(value) for value in row) + " |" for row in frame.to_numpy()]
    return "\n".join(lines)


def write_report(table: pd.DataFrame) -> None:
    stats = _summary(table)
    by_strategy = (
        table.groupby("Strategy", sort=False)[
            ["Mean In-Sample Sharpe", "OOS Sharpe", "B&H Sharpe", "Exposure"]
        ]
        .mean()
        .round(2)
        .reset_index()
    )
    detail = table[DETAIL_COLUMNS].copy()
    for column in ("OOS CAGR", "B&H CAGR", "OOS Max Drawdown", "B&H Max Drawdown"):
        detail[column] = detail[column].map("{:.1%}".format)
    for column in ("Deflated Sharpe vs Zero", "Deflated Sharpe vs B&H", "Probability of Overfitting"):
        detail[column] = detail[column].map("{:.1%}".format)
    detail = detail.round(2)
    assets = ", ".join(f"{ticker} ({name})" for ticker, name in UNIVERSE.items())
    overfitting_headline = (
        "Choosing the best parameters did no better than chance."
        if stats["overfitting"] >= 0.5
        else "Choosing the best parameters helped, but only a little."
    )

    count = table["Strategy"].nunique()
    count_word = NUMBER_WORDS.get(count, str(count))
    report = f"""# Research Results

Generated by `scripts/run_research.py`. Do not edit by hand.

**Question:** do these {count_word} classic strategies beat buy and hold once they are tuned without hindsight?

**Setup:** {stats["runs"]} walk-forward runs: {count} strategies on {len(UNIVERSE)} ETFs, using daily data from {START}
to {END}. Parameters were chosen on {TRAIN_BARS} bars and traded on the next {TEST_BARS}, with a commission of
{CONFIG.commission_rate:.1%} and slippage of {CONFIG.slippage_bps:.0f} bps per fill. Prices include dividends, idle cash
earns the 13-week US Treasury bill yield, and Sharpe ratios are measured on returns above that yield.

**Assets:** {assets}.

## Findings

- **Backtests overstated these strategies.** The mean Sharpe ratio fell from {stats["in_sample"]:.2f} in-sample to
  {stats["out_of_sample"]:.2f} out-of-sample, and it fell in {stats["decayed"]} of {stats["runs"]} runs.
- **Buy and hold was hard to beat.** Its mean Sharpe over the same periods was {stats["benchmark"]:.2f}. Strategies
  had the higher Sharpe in {stats["beat_sharpe"]} of {stats["runs"]} runs and the higher CAGR in {stats["beat_cagr"]}.
- **The strategies did reduce risk.** {stats["shallower"]} of {stats["runs"]} runs had a shallower maximum drawdown
  than buy and hold, helped by spending much of the time out of the market.
- **The benchmark matters as much as the correction.** {stats["significant_zero"]} of {stats["runs"]} full-sample
  winners were significant against zero at the 95% level, and {stats["significant_active"]} were significant against
  buy and hold. The highest deflated Sharpe against buy and hold was {stats["best_active"]:.1%}.
- **{overfitting_headline}** The mean probability of backtest overfitting was {stats["overfitting"]:.0%},
  where picking parameters at random would score about 50%. In {stats["overfit_runs"]} of {stats["runs"]} searches, the
  in-sample winner landed in the bottom half on unseen data in most splits of the history.

## Limitations of This Study

- Treasury bill yields were close to zero from 2009 to 2015 and again from 2020 to early 2022, so cash earned
  little for much of this period.
- Six assets over one period is a small sample, and these assets mostly rose over this period.
- Every strategy is long or flat. None can profit from a falling market.
- A positive result here would not have established a tradable edge, and this negative result does not show
  that no variant of these ideas can work.

## By Strategy

Means across the six assets.

{_markdown_table(by_strategy)}

## All Runs

{_markdown_table(detail)}
"""
    (DOCS / "RESULTS.md").write_text(report)


def main() -> None:
    if "--from-csv" in sys.argv:
        # Rewrite the report from saved results, for wording changes that need no new backtests.
        write_report(pd.read_csv(DOCS / "research_results.csv"))
        print(f"Wrote {DOCS / 'RESULTS.md'}")
        return
    keys = [key for key in STRATEGIES if key != "buy_and_hold"]
    rows = []
    cash_rate = fetch_cash_rate(START, END)
    for ticker in UNIVERSE:
        prices = attach_cash_rate(fetch_ohlcv(ticker, START, END), cash_rate)
        print(f"{ticker}: {len(prices)} bars")
        rows.extend(evaluate(ticker, key, prices) for key in keys)
    table = pd.DataFrame(rows)
    table.to_csv(DOCS / "research_results.csv", index=False, float_format="%.4f")
    write_report(table)
    print(f"Wrote {DOCS / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
