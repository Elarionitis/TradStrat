"""Command-line interface.

Examples:
    tradstrat list
    tradstrat backtest --sample --strategy rsi --param window=10
    tradstrat optimize --ticker SPY --strategy ma_crossover --points 6
    tradstrat walk-forward --ticker SPY --strategy donchian --train-bars 756 --test-bars 252
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from tradstrat import __version__
from tradstrat.data import fetch_ohlcv, load_csv, load_sample
from tradstrat.engine import BacktestConfig, run_backtest
from tradstrat.metrics import daily_returns
from tradstrat.reporting import comparison_table, probability_verdict
from tradstrat.stats import bootstrap, probabilistic_sharpe_ratio
from tradstrat.strategies import STRATEGIES, create_strategy
from tradstrat.validation import OBJECTIVES, build_grid, grid_search, walk_forward


def _parse_param(text: str) -> tuple[str, float]:
    name, separator, value = text.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError(f"Expected NAME=VALUE, got {text!r}.")
    try:
        return name, float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not a number.") from exc


def _shared_options() -> argparse.ArgumentParser:
    """Options common to every command that runs a backtest."""
    parser = argparse.ArgumentParser(add_help=False)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--ticker", help="Yahoo Finance symbol, such as SPY.")
    source.add_argument("--csv", type=Path, help="CSV file with Date and OHLCV columns.")
    source.add_argument("--sample", action="store_true", help="Use the bundled synthetic dataset.")
    parser.add_argument("--start", help="First date, YYYY-MM-DD. Default: ten years before --end.")
    parser.add_argument("--end", help="Last date, YYYY-MM-DD. Default: today.")
    parser.add_argument("--strategy", default="ma_crossover", choices=sorted(STRATEGIES))
    parser.add_argument("--capital", type=float, default=10_000.0)
    parser.add_argument("--commission", type=float, default=0.001, help="Fraction of traded value.")
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--position-fraction", type=float, default=1.0)
    parser.add_argument("--stop-loss", type=float, help="Fraction below entry, such as 0.1.")
    parser.add_argument("--target-vol", type=float, help="Annual volatility target, such as 0.15.")
    return parser


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tradstrat", description="Backtest and validate trading strategies."
    )
    parser.add_argument("--version", action="version", version=f"tradstrat {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    shared = _shared_options()

    commands.add_parser("list", help="Show the available strategies and their parameters.")

    backtest = commands.add_parser("backtest", parents=[shared], help="Run one backtest.")
    backtest.add_argument(
        "--param", "-p", action="append", type=_parse_param, default=[], metavar="NAME=VALUE"
    )
    backtest.add_argument("--trades-out", type=Path, help="Write the trade log to this CSV file.")

    optimize = commands.add_parser("optimize", parents=[shared], help="Search a parameter grid.")
    optimize.add_argument("--points", type=int, default=5, help="Grid points per parameter.")
    optimize.add_argument("--objective", default="sharpe_ratio", choices=OBJECTIVES)
    optimize.add_argument("--top", type=int, default=10, help="Rows of results to show.")

    forward = commands.add_parser("walk-forward", parents=[shared], help="Run walk-forward validation.")
    forward.add_argument("--points", type=int, default=5, help="Grid points per parameter.")
    forward.add_argument("--objective", default="sharpe_ratio", choices=OBJECTIVES)
    forward.add_argument("--train-bars", type=int, default=756)
    forward.add_argument("--test-bars", type=int, default=252)
    forward.add_argument("--anchored", action="store_true", help="Grow the training window.")
    return parser


def _load_data(args: argparse.Namespace) -> pd.DataFrame:
    if args.sample:
        return load_sample()
    if args.csv:
        return load_csv(args.csv)
    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.today().normalize()
    start = pd.Timestamp(args.start) if args.start else end - pd.DateOffset(years=10)
    return fetch_ohlcv(args.ticker, start, end)


def _config(args: argparse.Namespace) -> BacktestConfig:
    return BacktestConfig(
        initial_capital=args.capital,
        commission_rate=args.commission,
        slippage_bps=args.slippage_bps,
        position_fraction=args.position_fraction,
        stop_loss=args.stop_loss,
        target_volatility=args.target_vol,
    )


def _period(data: pd.DataFrame) -> str:
    return f"{data.index[0].date()} to {data.index[-1].date()} ({len(data)} bars)"


def _list_strategies() -> None:
    for key, cls in sorted(STRATEGIES.items()):
        print(f"{key}: {cls.description}")
        for spec in cls.params:
            print(f"    {spec.name} (default {spec.default}, range {spec.low} to {spec.high}): {spec.help}")


def _run_backtest(args: argparse.Namespace) -> None:
    data = _load_data(args)
    strategy = create_strategy(args.strategy, **dict(args.param))
    result = run_backtest(data, strategy, _config(args))
    returns = daily_returns(result.results["Strategy Equity"])
    benchmark_returns = daily_returns(result.results["Buy and Hold Benchmark"])

    print(f"{strategy}\n{_period(data)}\n")
    print(comparison_table(result.metrics, result.benchmark_metrics).to_string())

    relative = result.relative_metrics
    print(
        f"\nAlpha {relative.alpha:.2%}  Beta {relative.beta:.2f}  Info ratio {relative.information_ratio:.2f}"
    )
    print(f"Probabilistic Sharpe (true Sharpe > 0): {probabilistic_sharpe_ratio(returns):.1%}")

    if len(returns) >= 40:
        estimate = bootstrap(returns, benchmark_returns)
        print(f"Sharpe 95% interval: {estimate.sharpe.low:.2f} to {estimate.sharpe.high:.2f}")
        print(f"Probability of beating buy and hold: {estimate.probability_beats_benchmark:.1%}")

    if args.trades_out:
        result.trade_log.to_csv(args.trades_out, index=False)
        print(f"\nWrote {len(result.trade_log)} trades to {args.trades_out}")


def _run_optimize(args: argparse.Namespace) -> None:
    data = _load_data(args)
    grid = build_grid(args.strategy, args.points)
    search = grid_search(data, args.strategy, grid, _config(args), args.objective)

    print(f"{args.strategy}: {len(grid)} combinations, {_period(data)}\n")
    print(search.table.head(args.top).round(3).to_string(index=False))
    print(f"\nBest: {search.best_params}")
    print("This score is in-sample. Use walk-forward to measure performance on unseen data.")
    print(f"Deflated Sharpe: {search.deflated_sharpe:.1%}. {probability_verdict(search.deflated_sharpe)}")


def _run_walk_forward(args: argparse.Namespace) -> None:
    data = _load_data(args)
    result = walk_forward(
        data,
        args.strategy,
        build_grid(args.strategy, args.points),
        _config(args),
        train_bars=args.train_bars,
        test_bars=args.test_bars,
        objective=args.objective,
        anchored=args.anchored,
    )

    columns = ["Fold", "Test Start", "Test End", "Parameters", "In-Sample Score", "Out-of-Sample Score"]
    print(f"{args.strategy}: {len(result.folds)} folds, {_period(data)}\n")
    print(result.folds[columns].to_string(index=False, float_format="{:.2f}".format))
    print("\nOut-of-sample performance:\n")
    print(comparison_table(result.metrics, result.benchmark_metrics).to_string())
    print(f"\nWalk-forward efficiency: {result.efficiency:.2f}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    commands = {
        "backtest": _run_backtest,
        "optimize": _run_optimize,
        "walk-forward": _run_walk_forward,
    }
    if args.command == "list":
        _list_strategies()
        return 0
    try:
        commands[args.command](args)
    except ValueError as error:  # DataError is a ValueError, so bad input of any kind lands here.
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
