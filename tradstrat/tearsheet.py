"""A self-contained HTML report of one backtest, for sharing or printing.

The report holds the verdict, the headline numbers, the charts, the robustness statistics and the full
trade log in a single file. Charts stay interactive. By default they load Plotly from its CDN, which keeps
the file small; ``offline=True`` embeds the library so the report also opens without a connection.
"""

from __future__ import annotations

import html
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import pandas as pd
import plotly.graph_objects as go

from tradstrat import __version__
from tradstrat.charts import (
    LIGHT,
    annual_returns_figure,
    drawdown_figure,
    equity_figure,
    monthly_returns_figure,
    price_figure,
    rolling_sharpe_figure,
)
from tradstrat.engine import BacktestResult
from tradstrat.reporting import Verdict, assess, comparison_table, format_metric
from tradstrat.stats import BootstrapResult, bootstrap, probabilistic_sharpe_ratio

HEADLINE: Final[tuple[str, ...]] = ("total_return", "cagr", "sharpe_ratio", "max_drawdown")
HEADLINE_LABELS: Final[dict[str, str]] = {
    "total_return": "Total return",
    "cagr": "Annual return",
    "sharpe_ratio": "Sharpe ratio",
    "max_drawdown": "Max drawdown",
}
TONE_ICONS: Final[dict[str, str]] = {"good": "✓", "warning": "!", "bad": "✕", "neutral": "•"}


@dataclass(frozen=True)
class ReportContext:
    """What was tested, in words, for the report's header."""

    strategy: str
    asset: str
    parameters: dict[str, float] = field(default_factory=dict)
    overlays: pd.DataFrame | None = None


def _figure_html(figure: go.Figure, include_plotlyjs: bool | str) -> str:
    return figure.to_html(
        full_html=False,
        include_plotlyjs=include_plotlyjs,
        config={"displaylogo": False, "responsive": True},
    )


def _kpis(result: BacktestResult) -> str:
    strategy, benchmark = result.metrics.as_dict(), result.benchmark_metrics.as_dict()
    tiles = []
    for name in HEADLINE:
        tiles.append(
            '<div class="kpi">'
            f'<div class="kpi-label">{HEADLINE_LABELS[name]}</div>'
            f'<div class="kpi-value">{format_metric(name, strategy[name])}</div>'
            f'<div class="kpi-compare">Buy and hold {format_metric(name, benchmark[name])}</div>'
            "</div>"
        )
    return f'<section class="kpis">{"".join(tiles)}</section>'


def _verdict(verdict: Verdict) -> str:
    findings = "".join(
        f'<li class="finding {finding.tone}"><span class="icon" aria-hidden="true">'
        f"{TONE_ICONS[finding.tone]}</span><span><strong>{html.escape(finding.title)}.</strong> "
        f"{html.escape(finding.detail)}</span></li>"
        for finding in verdict.findings
    )
    return (
        f'<section class="verdict {verdict.tone}">'
        f'<div class="verdict-icon" aria-hidden="true">{TONE_ICONS[verdict.tone]}</div>'
        f"<div><h2>{html.escape(verdict.headline)}</h2><p>{html.escape(verdict.summary)}</p></div>"
        f'</section><ul class="findings">{findings}</ul>'
    )


def _robustness(result: BacktestResult, estimate: BootstrapResult | None) -> str:
    psr = probabilistic_sharpe_ratio(result.excess_returns)
    rows = [
        ("Probabilistic Sharpe ratio (true Sharpe above zero)", "n/a" if math.isnan(psr) else f"{psr:.1%}")
    ]
    if estimate is not None:
        rows += [
            ("Sharpe ratio, 95% interval", f"{estimate.sharpe.low:.2f} to {estimate.sharpe.high:.2f}"),
            (
                "Annual return, 95% interval",
                f"{estimate.annual_return.low:.1%} to {estimate.annual_return.high:.1%}",
            ),
            (
                "Max drawdown, 95% interval",
                f"{estimate.max_drawdown.low:.1%} to {estimate.max_drawdown.high:.1%}",
            ),
            ("Chance of a higher Sharpe than buy and hold", f"{estimate.probability_beats_benchmark:.1%}"),
        ]
    body = "".join(f"<tr><th>{label}</th><td>{value}</td></tr>" for label, value in rows)
    return f'<table class="plain">{body}</table>'


def _settings(result: BacktestResult, context: ReportContext) -> str:
    config = result.config
    rows = [
        ("Strategy", context.strategy),
        ("Asset", context.asset),
        *[(name.replace("_", " ").capitalize(), f"{value:g}") for name, value in context.parameters.items()],
        ("Starting capital", f"${config.initial_capital:,.0f}"),
        ("Commission", f"{config.commission_rate * 10_000:g} bps per side"),
        ("Slippage", f"{config.slippage_bps:g} bps per fill"),
        ("Position size", f"{config.position_fraction:.0%} of equity"),
        (
            "Stop loss",
            "None"
            if config.stop_loss is None
            else f"{config.stop_loss:.0%}{' trailing' if config.trailing_stop else ''}",
        ),
        (
            "Volatility target",
            "None" if config.target_volatility is None else f"{config.target_volatility:.0%} a year",
        ),
        ("Interest earned on idle cash", f"${result.results['Interest'].sum():,.2f}"),
    ]
    body = "".join(
        f"<tr><th>{html.escape(label)}</th><td>{html.escape(value)}</td></tr>" for label, value in rows
    )
    return f'<table class="plain">{body}</table>'


def _trades(result: BacktestResult) -> str:
    trades = result.trade_log
    if trades.empty:
        return "<p>No trades.</p>"
    shown = trades.assign(
        **{
            "Entry Date": trades["Entry Date"].dt.date,
            "Exit Date": trades["Exit Date"].dt.date,
            "Entry Price": trades["Entry Price"].map("{:,.2f}".format),
            "Exit Price": trades["Exit Price"].map("{:,.2f}".format),
            "Commission": trades["Commission"].map("{:,.2f}".format),
            "P&L": trades["P&L"].map("{:,.2f}".format),
            "Return": trades["Return"].map("{:.2%}".format),
            "Exit Reason": trades["Exit Reason"].str.replace("_", " "),
        }
    )
    table = shown.to_html(index=False, classes="trades", border=0, escape=True)
    return f"<details><summary>Show all {len(trades)} trades</summary>{table}</details>"


def build_report(
    result: BacktestResult,
    context: ReportContext,
    *,
    estimate: BootstrapResult | None = None,
    offline: bool = False,
) -> str:
    """Render a backtest as a complete HTML document."""
    if estimate is None and len(result.excess_returns.dropna()) >= 40:
        estimate = bootstrap(result.excess_returns, result.benchmark_excess_returns)
    probability = estimate.probability_beats_benchmark if estimate is not None else float("nan")
    verdict = assess(result.metrics, result.benchmark_metrics, probability)

    results = result.results
    figures = [
        equity_figure(results["Strategy Equity"], results["Buy and Hold Benchmark"], LIGHT),
        drawdown_figure(results["Strategy Drawdown"], results["Benchmark Drawdown"], LIGHT),
        annual_returns_figure(results["Strategy Equity"], results["Buy and Hold Benchmark"], LIGHT),
        rolling_sharpe_figure(result.excess_returns, result.benchmark_excess_returns, LIGHT),
        monthly_returns_figure(results["Strategy Equity"], LIGHT),
        price_figure(
            results["Close"],
            result.trade_log,
            LIGHT,
            None if context.overlays is None else context.overlays.loc[results.index],
        ),
    ]
    charts = []
    for number, figure in enumerate(figures):
        include = (True if offline else "cdn") if number == 0 else False
        wide = number in (0, 5)
        charts.append(f'<div class="chart{" wide" if wide else ""}">{_figure_html(figure, include)}</div>')

    metrics = comparison_table(result.metrics, result.benchmark_metrics).to_html(classes="metrics", border=0)
    period = f"{results.index[0]:%d %b %Y} to {results.index[-1]:%d %b %Y}"
    generated = datetime.now(UTC).strftime("%d %b %Y")
    title = f"{context.strategy} on {context.asset}"

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>{STYLES}</style>
</head>
<body>
<main>
<header>
  <div class="eyebrow">TradStrat backtest report</div>
  <h1>{html.escape(title)}</h1>
  <p class="period">{period} · {len(results) - 1:,} trading days</p>
</header>
{_verdict(verdict)}
{_kpis(result)}
<section class="charts">{"".join(charts)}</section>
<section class="columns">
  <div><h3>All metrics</h3>{metrics}</div>
  <div><h3>How much to trust it</h3>{_robustness(result, estimate)}
  <h3>Settings</h3>{_settings(result, context)}</div>
</section>
<section><h3>Trades</h3>{_trades(result)}</section>
<footer>
  <p>Signals fill at the next bar's open, after commission and slippage. Prices include dividends. Sharpe
  ratios are measured on returns above the cash rate. Confidence intervals come from a 20-day circular block
  bootstrap with 2,000 resamples.</p>
  <p>Generated by TradStrat {__version__} on {generated}. For research and education only; this is not
  investment advice. Past performance does not predict future results.</p>
</footer>
</main>
</body>
</html>
"""


def write_report(path: str | Path, html_text: str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(html_text, encoding="utf-8")
    return destination


STYLES: Final[str] = """
:root {
  color-scheme: light;
  --page: #f9f9f7;
  --surface: #fcfcfb;
  --ink: #0b0b0b;
  --secondary: #52514e;
  --muted: #898781;
  --hairline: #e1e0d9;
  --good: #006300;
  --good-wash: #e7f4e7;
  --warning: #8a5a00;
  --warning-wash: #fdf3dc;
  --bad: #b42b2b;
  --bad-wash: #fbe9e9;
  --neutral-wash: #f0efec;
  --accent: #2a78d6;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--page);
  color: var(--ink);
  font: 15px/1.55 Inter, system-ui, -apple-system, "Segoe UI", sans-serif;
}
main { max-width: 1120px; margin: 0 auto; padding: 40px 16px 64px; }
header { margin-bottom: 24px; }
.eyebrow { color: var(--accent); font-weight: 600; font-size: 13px; letter-spacing: .04em;
  text-transform: uppercase; }
h1 { font-size: 32px; line-height: 1.2; margin: 6px 0 4px; letter-spacing: -.01em; }
h2 { font-size: 20px; margin: 0 0 2px; }
h3 { font-size: 16px; margin: 28px 0 10px; }
.period { color: var(--secondary); margin: 0; }
.verdict { display: flex; gap: 14px; align-items: flex-start; padding: 18px 20px; border-radius: 12px;
  border: 1px solid var(--hairline); background: var(--neutral-wash); }
.verdict p { margin: 0; color: var(--secondary); }
.verdict-icon { flex: none; width: 32px; height: 32px; border-radius: 50%; display: grid;
  place-items: center; font-weight: 700; color: #fff; background: var(--muted); }
.verdict.good { background: var(--good-wash); } .verdict.good .verdict-icon { background: var(--good); }
.verdict.warning { background: var(--warning-wash); }
.verdict.warning .verdict-icon { background: var(--warning); }
.verdict.bad { background: var(--bad-wash); } .verdict.bad .verdict-icon { background: var(--bad); }
.findings { list-style: none; padding: 0; margin: 14px 0 0; display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 8px 20px; }
.finding { display: flex; gap: 10px; color: var(--secondary); }
.finding .icon { flex: none; width: 20px; text-align: center; font-weight: 700; }
.finding.good .icon { color: var(--good); } .finding.bad .icon { color: var(--bad); }
.finding.warning .icon { color: var(--warning); } .finding.neutral .icon { color: var(--muted); }
.finding strong { color: var(--ink); }
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px;
  margin: 24px 0; }
.kpi { background: var(--surface); border: 1px solid var(--hairline); border-radius: 12px;
  padding: 14px 16px; }
.kpi-label { color: var(--secondary); font-size: 13px; }
.kpi-value { font-size: 28px; font-weight: 600; letter-spacing: -.01em; }
.kpi-compare { color: var(--muted); font-size: 13px; }
.charts { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
.chart { background: var(--surface); border: 1px solid var(--hairline); border-radius: 12px;
  padding: 8px; min-width: 0; }
.chart.wide { grid-column: 1 / -1; }
.columns { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 24px; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--hairline); }
.metrics td, .trades td { text-align: right; }
.plain th { font-weight: 500; color: var(--secondary); width: 60%; }
thead th { color: var(--secondary); font-weight: 600; font-size: 13px; }
details summary { cursor: pointer; color: var(--accent); font-weight: 500; margin-bottom: 8px; }
.trades { font-size: 13px; }
footer { margin-top: 40px; color: var(--muted); font-size: 13px; border-top: 1px solid var(--hairline);
  padding-top: 16px; }
@media (max-width: 720px) {
  .charts { grid-template-columns: minmax(0, 1fr); }
  h1 { font-size: 26px; }
  details { overflow-x: auto; }
}
@media print {
  body { background: #fff; }
  .chart, .kpi { break-inside: avoid; }
}
"""
