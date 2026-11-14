"""End-to-end tests of the command-line interface, run against the bundled dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tradstrat.cli import main


def test_list_shows_every_strategy(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["list"]) == 0

    output = capsys.readouterr().out
    for key in ("ma_crossover", "mean_reversion", "rsi", "donchian", "momentum", "macd", "trend_filter"):
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


def test_backtest_ends_with_a_plain_english_verdict(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["backtest", "--sample", "--strategy", "momentum"]) == 0

    output = capsys.readouterr().out
    assert "buy and hold" in output.split("Probability of beating")[-1].lower()
    assert "Worst decline" in output


def test_report_writes_a_self_contained_html_file(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "report.html"

    assert main(["report", "--sample", "--strategy", "donchian", "--out", str(path)]) == 0

    text = path.read_text()
    assert text.startswith("<!doctype html>")
    assert "Donchian Breakout on the synthetic sample" in text
    assert "Entry channel" in text  # The strategy's indicator lines are drawn on the price chart.
    assert 'src="https://cdn.plot.ly' in text


def test_offline_report_embeds_the_chart_library(tmp_path: Path) -> None:
    path = tmp_path / "report.html"

    assert main(["report", "--sample", "--out", str(path), "--offline"]) == 0

    text = path.read_text()
    assert 'src="https://cdn.plot.ly' not in text
    assert len(text) > 1_000_000  # The library itself is inlined.


def test_compare_lists_every_strategy_and_buy_and_hold(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["compare", "--sample"]) == 0

    output = capsys.readouterr().out
    for label in ("Moving Average Crossover", "RSI Reversal", "Buy and Hold"):
        assert label in output
    assert "nothing here was tuned" in output


def test_optimize_reports_the_probability_of_overfitting(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["optimize", "--sample", "--strategy", "rsi", "--points", "3"]) == 0

    assert "Probability of backtest overfitting" in capsys.readouterr().out


def test_cash_rate_and_trailing_stop_options_are_accepted(capsys: pytest.CaptureFixture[str]) -> None:
    arguments = ["backtest", "--sample", "--cash-rate", "0.04", "--stop-loss", "0.1", "--trailing"]

    assert main(arguments) == 0


def test_treasury_bill_cash_rate_is_fetched_for_the_data_range(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from tradstrat import cli

    requested: list[pd.Timestamp] = []

    def fake_rates(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        requested.append(start)
        return pd.Series([0.05], index=[start])

    monkeypatch.setattr(cli, "fetch_cash_rate", fake_rates)

    assert main(["backtest", "--sample", "--cash-rate", "tbill"]) == 0
    assert requested


def test_a_malformed_cash_rate_is_rejected() -> None:
    with pytest.raises(SystemExit):
        main(["backtest", "--sample", "--cash-rate", "lots"])


def test_a_trailing_stop_without_a_distance_is_a_clean_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["backtest", "--sample", "--trailing"]) == 1

    assert "trailing stop needs" in capsys.readouterr().err


def test_report_escapes_untrusted_labels() -> None:
    """A file name is shown in the report, so markup in it must be escaped rather than executed."""
    from conftest import random_walk

    from tradstrat.engine import run_backtest
    from tradstrat.strategies import create_strategy
    from tradstrat.tearsheet import ReportContext, build_report

    result = run_backtest(random_walk(300), create_strategy("rsi"))

    text = build_report(result, ReportContext(strategy="RSI", asset="<script>alert(1)</script>.csv"))

    assert "<script>alert(1)</script>" not in text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;.csv" in text


def test_backtest_accepts_a_custom_rule(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        ["backtest", "--sample", "--buy", "rsi(2) < 10 and close > sma(200)", "--sell", "rsi(2) > 70"]
    )

    assert code == 0
    output = capsys.readouterr().out
    assert "buy when rsi(2) < 10 and close > sma(200); sell when rsi(2) > 70" in output
    assert "Sharpe Ratio" in output


def test_optimize_tunes_the_numbers_in_a_custom_rule(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["optimize", "--sample", "--buy", "close > sma(100)", "--points", "3"]) == 0

    output = capsys.readouterr().out
    assert "3 combinations" in output
    assert "Deflated Sharpe" in output


def test_an_unreadable_rule_is_a_clean_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["backtest", "--sample", "--buy", "close above sma(20)"]) == 1

    assert "Could not read" in capsys.readouterr().err


def test_sell_without_buy_is_a_clean_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["backtest", "--sample", "--sell", "close < sma(20)"]) == 1

    assert "--sell needs a --buy" in capsys.readouterr().err


def test_list_explains_how_to_write_a_rule(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["list"]) == 0

    assert "--buy" in capsys.readouterr().out


def test_stress_command_answers_the_six_questions(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["stress", "--sample"]) == 0

    output = capsys.readouterr().out
    assert "stress tests." in output
    for number in range(1, 7):
        assert f"{number}. " in output
    assert "Did it survive a real bear market?" in output


def test_reports_include_the_stress_test(tmp_path: Path) -> None:
    out = tmp_path / "report.html"
    assert main(["report", "--sample", "--out", str(out)]) == 0

    assert "Stress test: Passed" in out.read_text()
