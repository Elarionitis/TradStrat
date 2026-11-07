"""End-to-end tests of the command-line interface, run against the bundled dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tradstrat.cli import main


def test_list_shows_every_strategy(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["list"]) == 0

    output = capsys.readouterr().out
    for key in ("ma_crossover", "mean_reversion", "rsi", "donchian", "momentum"):
        assert key in output


def test_backtest_prints_a_comparison(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["backtest", "--sample", "--strategy", "rsi", "--param", "window=10"]) == 0

    output = capsys.readouterr().out
    assert "Sharpe Ratio" in output
    assert "Buy and Hold" in output
    assert "95% interval" in output


def test_backtest_writes_the_trade_log(tmp_path: Path) -> None:
    path = tmp_path / "trades.csv"

    assert main(["backtest", "--sample", "--trades-out", str(path)]) == 0

    assert "P&L" in pd.read_csv(path).columns


def test_backtest_reads_a_csv_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "prices.csv"
    pd.DataFrame(
        {"Date": pd.bdate_range("2024-01-01", periods=60), "Open": 10.0, "High": 11.0}
        | {"Low": 9.0, "Close": 10.5, "Volume": 100}
    ).to_csv(path, index=False)

    assert main(["backtest", "--csv", str(path), "--strategy", "buy_and_hold"]) == 0


def test_optimize_warns_that_its_score_is_in_sample(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["optimize", "--sample", "--strategy", "momentum", "--points", "3"]) == 0

    output = capsys.readouterr().out
    assert "in-sample" in output
    assert "Deflated Sharpe" in output


def test_walk_forward_reports_out_of_sample_results(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["walk-forward", "--sample", "--strategy", "momentum", "--points", "3"]) == 0

    output = capsys.readouterr().out
    assert "Out-of-sample" in output
    assert "efficiency" in output


@pytest.mark.parametrize(
    "arguments",
    [
        ["backtest", "--sample", "--param", "short_window=300", "--param", "long_window=20"],
        ["backtest", "--sample", "--param", "moon_phase=3"],
        ["backtest", "--ticker", "../etc/passwd"],
        ["backtest", "--csv", "does-not-exist.csv"],
        ["backtest", "--sample", "--stop-loss", "2"],
        ["walk-forward", "--sample", "--train-bars", "5000"],
    ],
)
def test_bad_input_gives_a_clean_error_and_nonzero_exit(
    arguments: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(arguments) == 1

    captured = capsys.readouterr()
    assert captured.err.startswith("error:")
    assert "Traceback" not in captured.err


def test_malformed_param_is_rejected_by_the_parser() -> None:
    with pytest.raises(SystemExit):
        main(["backtest", "--sample", "--param", "window"])


def test_a_data_source_is_required() -> None:
    with pytest.raises(SystemExit):
        main(["backtest"])
