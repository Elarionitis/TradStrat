"""Streamlit dashboard for TradStrat.

Run it with ``streamlit run app.py``. Every number on the page comes from the ``tradstrat`` package, so
the dashboard, the command line and the HTML report always agree.
"""

from __future__ import annotations

import base64
import html
import io
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
import streamlit as st

from tradstrat import __version__
from tradstrat.charts import (
    ChartTheme,
    annual_returns_figure,
    bootstrap_figure,
    comparison_figure,
    drawdown_figure,
    equity_figure,
    fold_scores_figure,
    monthly_returns_figure,
    overfitting_figure,
    price_figure,
    rolling_sharpe_figure,
    search_figure,
    theme_for,
)
from tradstrat.data import (
    DataError,
    adjust_for_dividends,
    attach_cash_rate,
    fetch_cash_rate,
    fetch_ohlcv,
    load_sample,
    read_price_csv,
)
from tradstrat.engine import BacktestConfig, BacktestResult, run_backtest
from tradstrat.glossary import GLOSSARY
from tradstrat.metrics import calendar_year_returns, rolling_sharpe
from tradstrat.reporting import (
    METRIC_LABELS,
    Verdict,
    assess,
    comparison_table,
    format_metric,
    overfitting_verdict,
    probability_verdict,
)
from tradstrat.stats import BootstrapResult, bootstrap, probabilistic_sharpe_ratio
from tradstrat.strategies import STRATEGIES, create_strategy, indicator_overlays
from tradstrat.tearsheet import ReportContext, build_report
from tradstrat.validation import (
    OBJECTIVES,
    Comparison,
    build_grid,
    compare_strategies,
    grid_search,
    walk_forward,
)

STRATEGY_KEYS: Final[list[str]] = [key for key in STRATEGIES if key != "buy_and_hold"]
QUICK_TICKERS: Final[list[str]] = ["SPY", "QQQ", "AAPL", "NVDA", "BTC-USD", "GLD", "TLT"]
PERIODS: Final[dict[str, int | None]] = {"1Y": 1, "3Y": 3, "5Y": 5, "10Y": 10, "20Y": 20, "Max": None}
SOURCES: Final[list[str]] = ["Yahoo", "CSV", "Sample"]
EARLIEST: Final[pd.Timestamp] = pd.Timestamp("1970-01-01")
CASH_OPTIONS: Final[list[str]] = ["Treasury bills", "Fixed rate", "None"]
TONE_STYLES: Final[dict[str, dict[str, str]]] = {
    "light": {
        "good": "#006300|#e7f4e7",
        "warning": "#8a5a00|#fdf3dc",
        "bad": "#b42b2b|#fbe9e9",
        "neutral": "#52514e|#f0efec",
        "ink": "#0b0b0b",
        "secondary": "#52514e",
        "border": "#e1e0d9",
    },
    "dark": {
        "good": "#0ca30c|#122612",
        "warning": "#fab219|#2b2210",
        "bad": "#e66767|#2e1616",
        "neutral": "#c3c2b7|#1f1f1d",
        "ink": "#f0efec",
        "secondary": "#c3c2b7",
        "border": "#2c2c2a",
    },
}
TONE_ICONS: Final[dict[str, str]] = {"good": "✓", "warning": "!", "bad": "✕", "neutral": "•"}
LOGO: Final[str] = base64.b64encode((Path(__file__).parent / "assets" / "logo.svg").read_bytes()).decode()


@dataclass(frozen=True)
class Inputs:
    """Everything the sidebar collects."""

    source: str
    label: str
    ticker: str
    start: pd.Timestamp
    end: pd.Timestamp
    upload: Any
    strategy_key: str
    params: dict[str, float]
    config: BacktestConfig | None
    config_error: str | None
    cash: str
    fixed_rate: float

    def fingerprint(self) -> str:
        """Identifies the data and settings, so stored search results are never shown for other inputs."""
        upload = getattr(self.upload, "file_id", None)
        config = asdict(self.config) if self.config else None
        return repr(
            (self.source, self.ticker, self.start, self.end, upload, self.strategy_key, config, self.cash)
        )


# Data loading, cached so that moving a slider does not refetch anything.


@st.cache_data(show_spinner="Fetching market data...", ttl=3600)
def cached_fetch(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return fetch_ohlcv(ticker, start, end)


@st.cache_data(show_spinner="Fetching Treasury bill yields...", ttl=3600)
def cached_cash_rate(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    return fetch_cash_rate(start, end)


@st.cache_data(show_spinner=False)
def cached_upload(contents: bytes, name: str) -> pd.DataFrame:
    return adjust_for_dividends(read_price_csv(io.BytesIO(contents), name))


@st.cache_data(show_spinner="Resampling returns...")
def cached_bootstrap(returns: pd.Series, benchmark: pd.Series) -> BootstrapResult:
    return bootstrap(returns, benchmark)


@st.cache_data(show_spinner="Running every strategy...")
def cached_comparison(prices: pd.DataFrame, config: BacktestConfig) -> Comparison:
    return compare_strategies(prices, config)


def load_prices(inputs: Inputs) -> tuple[pd.DataFrame, str | None]:
    """Return prices with the cash rate attached, plus a warning if the cash rate could not be used."""
    if inputs.source == "Sample":
        prices = load_sample()
    elif inputs.source == "CSV":
        if inputs.upload is None:
            raise DataError("Upload a CSV file with a date column and open, high, low and close prices.")
        prices = cached_upload(inputs.upload.getvalue(), inputs.upload.name)
    else:
        prices = cached_fetch(inputs.ticker, inputs.start, inputs.end)

    if inputs.cash == "Fixed rate":
        return attach_cash_rate(prices, inputs.fixed_rate), None
    if inputs.cash == "Treasury bills":
        try:
            return attach_cash_rate(prices, cached_cash_rate(prices.index[0], prices.index[-1])), None
        except DataError:
            return (
                prices,
                "Treasury bill yields could not be fetched, so idle cash earns nothing in this run.",
            )
    return prices, None


# Sidebar.


def _pick_ticker() -> None:
    choice = st.session_state.get("ticker_pick")
    if choice:
        st.session_state["ticker"] = choice


def _typed_ticker() -> None:
    """Typing a symbol by hand deselects the shortcut, so the two never disagree."""
    if st.session_state.get("ticker_pick") != st.session_state.get("ticker", "").strip().upper():
        st.session_state["ticker_pick"] = None


def param_widget(strategy_key: str, spec: Any) -> float:
    """Render one strategy parameter, using the range the strategy declares."""
    kind = int if spec.is_integer else float
    return st.slider(
        spec.label,
        min_value=kind(spec.low),
        max_value=kind(spec.high),
        value=kind(spec.default),
        step=kind(spec.step),
        help=spec.help,
        key=f"{strategy_key}.{spec.name}",
    )


def read_sidebar() -> Inputs:
    today = pd.Timestamp.today().normalize()
    with st.sidebar:
        st.markdown("### Market data")
        source = st.segmented_control(
            "Source", SOURCES, default="Yahoo", key="source", required=True, width="stretch"
        )
        ticker, start, end, upload, label = "", today, today, None, "Sample data"
        if source == "Yahoo":
            st.session_state.setdefault("ticker", "SPY")
            ticker = st.text_input(
                "Ticker",
                key="ticker",
                on_change=_typed_ticker,
                help="Any Yahoo Finance symbol: stocks, ETFs, indices (^GSPC), BTC-USD.",
            )
            st.pills(
                "Popular",
                QUICK_TICKERS,
                key="ticker_pick",
                on_change=_pick_ticker,
                label_visibility="collapsed",
            )
            period = st.select_slider("Period", list(PERIODS), value="10Y")
            years = PERIODS[period]
            if st.toggle("Custom dates", key="custom_dates"):
                left, right = st.columns(2)
                start = pd.Timestamp(left.date_input("Start", value=today - pd.DateOffset(years=10)))
                end = pd.Timestamp(right.date_input("End", value=today))
            else:
                start, end = (today - pd.DateOffset(years=years) if years else EARLIEST), today
            label = ticker.strip().upper() or "?"
        elif source == "CSV":
            upload = st.file_uploader(
                "Price file",
                type=["csv"],
                help="Needs a date column (Date, Datetime, Timestamp or Time) and open, high, low and close "
                "columns in any capitalization. Volume and an adjusted close are optional.",
            )
            label = upload.name if upload else "your file"
        else:
            st.caption(
                "A simulated market that needs no internet. Its results say nothing about real markets."
            )

        st.markdown("### Strategy")
        strategy_key = st.selectbox(
            "Strategy",
            STRATEGY_KEYS,
            format_func=lambda key: STRATEGIES[key].label,
            label_visibility="collapsed",
        )
        st.caption(STRATEGIES[strategy_key].description)
        params = {spec.name: param_widget(strategy_key, spec) for spec in STRATEGIES[strategy_key].params}

        st.markdown("### Costs and risk")
        with st.expander("Capital and trading costs", expanded=False):
            capital = st.number_input("Starting capital ($)", min_value=1_000.0, value=10_000.0, step=1_000.0)
            commission_bps = st.number_input(
                "Commission (bps per side)",
                min_value=0.0,
                max_value=100.0,
                value=10.0,
                step=1.0,
                help=GLOSSARY["Commission"],
            )
            slippage_bps = st.number_input(
                "Slippage (bps per fill)",
                min_value=0.0,
                max_value=100.0,
                value=5.0,
                step=1.0,
                help=GLOSSARY["Slippage"],
            )
        with st.expander("Position sizing and stops", expanded=False):
            position_fraction = st.slider(
                "Share of equity per trade", 0.1, 1.0, 1.0, 0.1, help=GLOSSARY["Position Sizing"]
            )
            use_stop = st.toggle("Stop loss", help=GLOSSARY["Stop Loss"])
            stop_loss, trailing = None, False
            if use_stop:
                stop_loss = st.slider("Stop distance", 0.01, 0.50, 0.10, 0.01)
                trailing = st.toggle("Trail the stop", help=GLOSSARY["Trailing Stop"])
            use_target = st.toggle("Volatility targeting", help=GLOSSARY["Volatility Targeting"])
            target = st.slider("Annual volatility target", 0.05, 0.50, 0.15, 0.01) if use_target else None
        with st.expander("Idle cash", expanded=False):
            cash = st.radio(
                "Uninvested cash earns",
                CASH_OPTIONS,
                index=0 if source == "Yahoo" else 2,
                help=GLOSSARY["Cash Yield"],
            )
            fixed_rate = 0.0
            if cash == "Fixed rate":
                fixed_rate = st.number_input("Annual rate (%)", 0.0, 20.0, 4.0, 0.25) / 100
            if cash == "Treasury bills" and source != "Yahoo":
                st.caption("Uses real US Treasury bill yields for the dates in your data.")

        st.caption(f"TradStrat {__version__} · For research and education. Not investment advice.")

    try:
        config = BacktestConfig(
            initial_capital=capital,
            commission_rate=commission_bps / 10_000,
            slippage_bps=slippage_bps,
            position_fraction=position_fraction,
            stop_loss=stop_loss,
            trailing_stop=trailing,
            target_volatility=target,
        )
        error = None
    except ValueError as exc:
        config, error = None, str(exc)
    return Inputs(
        source or "Yahoo",
        label,
        ticker,
        start,
        end,
        upload,
        strategy_key,
        params,
        config,
        error,
        cash,
        fixed_rate,
    )


# Shared pieces.


def chart(figure: Any) -> None:
    st.plotly_chart(figure, theme=None, config={"displaylogo": False})


def callout(tone: str, text: str, icon: str) -> None:
    """A message box whose color matches its meaning, so bad news never looks like a neutral note."""
    box = {"good": st.success, "warning": st.warning, "bad": st.error}.get(tone, st.info)
    box(text, icon=icon)


def tone_colors(theme: ChartTheme, tone: str) -> tuple[str, str]:
    foreground, background = TONE_STYLES[theme.name][tone].split("|")
    return foreground, background


def render_verdict(verdict: Verdict, theme: ChartTheme) -> None:
    """A headline card in the verdict's tone, followed by the findings behind it."""
    styles = TONE_STYLES[theme.name]
    color, wash = tone_colors(theme, verdict.tone)
    findings = "".join(
        f'<div style="display:flex;gap:10px;align-items:flex-start;">'
        f'<span style="color:{tone_colors(theme, finding.tone)[0]};font-weight:700;width:16px;'
        f'text-align:center;flex:none;">{TONE_ICONS[finding.tone]}</span>'
        f'<span style="color:{styles["secondary"]};"><strong style="color:{styles["ink"]};">'
        f"{html.escape(finding.title)}.</strong> {html.escape(finding.detail)}</span></div>"
        for finding in verdict.findings
    )
    st.html(
        f"""
<div style="font-family:Inter,system-ui,sans-serif;">
  <div style="display:flex;gap:14px;align-items:center;padding:16px 18px;border-radius:12px;
      background:{wash};border:1px solid {styles["border"]};">
    <div style="flex:none;width:34px;height:34px;border-radius:50%;background:{color};color:#fff;
        display:grid;place-items:center;font-weight:700;font-size:16px;">{TONE_ICONS[verdict.tone]}</div>
    <div>
      <div style="font-size:18px;font-weight:600;color:{styles["ink"]};">{html.escape(verdict.headline)}</div>
      <div style="color:{styles["secondary"]};font-size:14px;">{html.escape(verdict.summary)}</div>
    </div>
  </div>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:8px 22px;
      margin-top:12px;font-size:14px;line-height:1.5;">{findings}</div>
</div>"""
    )


def _points(value: float, benchmark: float, percent: bool = True) -> str | None:
    """The gap to buy and hold, in percentage points for rates and plain units for ratios."""
    if math.isnan(value) or math.isnan(benchmark):
        return None
    difference = (value - benchmark) * (100 if percent else 1)
    digits = 1 if percent else 2
    if round(difference, digits) == 0:
        return f"{0:.{digits}f}{' pts' if percent else ''}"
    return f"{difference:+.{digits}f}{' pts' if percent else ''}"


def _weekly(series: pd.Series) -> list[float]:
    """Thin a daily series for a sparkline, which only needs its shape."""
    return series.dropna().iloc[:: max(1, len(series) // 120)].round(4).tolist()


def render_kpis(result: BacktestResult) -> None:
    strategy, benchmark = result.metrics, result.benchmark_metrics
    results = result.results
    years = calendar_year_returns(results["Strategy Equity"])
    tiles = [
        (
            "Total return",
            format_metric("total_return", strategy.total_return),
            _points(strategy.total_return, benchmark.total_return),
            _weekly(results["Strategy Equity"]),
            "area",
        ),
        (
            "Annual return",
            format_metric("cagr", strategy.cagr),
            _points(strategy.cagr, benchmark.cagr),
            years.round(4).tolist(),
            "bar",
        ),
        (
            "Sharpe ratio",
            format_metric("sharpe_ratio", strategy.sharpe_ratio),
            _points(strategy.sharpe_ratio, benchmark.sharpe_ratio, percent=False),
            _weekly(rolling_sharpe(result.excess_returns)),
            "line",
        ),
        (
            "Max drawdown",
            format_metric("max_drawdown", strategy.max_drawdown),
            _points(strategy.max_drawdown, benchmark.max_drawdown),
            _weekly(results["Strategy Drawdown"]),
            "area",
        ),
    ]
    for column, (label, value, delta, sparkline, kind) in zip(st.columns(4), tiles, strict=True):
        column.metric(
            label,
            value,
            delta,
            delta_description="vs B&H" if delta else None,
            delta_color="off" if delta and not delta.startswith(("+", "-")) else "normal",
            border=True,
            chart_data=sparkline or None,
            chart_type=kind,
            help=GLOSSARY.get({"Annual return": "CAGR"}.get(label, label.title()), None),
        )


def dataset_caption(prices: pd.DataFrame, inputs: Inputs, warning: str | None) -> None:
    cash = {
        "Treasury bills": "idle cash earns Treasury bill yields",
        "Fixed rate": f"idle cash earns {inputs.fixed_rate:.2%}",
        "None": "idle cash earns nothing",
    }[inputs.cash]
    if warning:
        cash = "idle cash earns nothing"
    st.caption(
        f"**{html.escape(inputs.label)}** · {prices.index[0]:%d %b %Y} to {prices.index[-1]:%d %b %Y} · "
        f"{len(prices):,} bars · {STRATEGIES[inputs.strategy_key].label} · {cash}"
    )
    if warning:
        st.warning(warning, icon=":material/cloud_off:")


# Tabs.


def render_overview(result: BacktestResult, estimate: BootstrapResult | None, theme: ChartTheme) -> None:
    probability = estimate.probability_beats_benchmark if estimate else float("nan")
    render_verdict(assess(result.metrics, result.benchmark_metrics, probability), theme)
    st.space("small")
    render_kpis(result)

    results = result.results
    log_scale = st.toggle("Log scale", key="log_equity", help="Show equal percentage moves as equal heights.")
    chart(
        equity_figure(
            results["Strategy Equity"], results["Buy and Hold Benchmark"], theme, log_scale=log_scale
        )
    )
    left, right = st.columns(2)
    with left:
        chart(drawdown_figure(results["Strategy Drawdown"], results["Benchmark Drawdown"], theme))
    with right:
        chart(annual_returns_figure(results["Strategy Equity"], results["Buy and Hold Benchmark"], theme))
    left, right = st.columns(2)
    with left:
        chart(rolling_sharpe_figure(result.excess_returns, result.benchmark_excess_returns, theme))
    with right:
        chart(monthly_returns_figure(results["Strategy Equity"], theme))

    left, right = st.columns([3, 2])
    with left:
        st.subheader("All metrics")
        st.dataframe(
            comparison_table(result.metrics, result.benchmark_metrics),
            height="content",
            column_config={"_index": st.column_config.Column(width="medium")},
        )
    with right:
        st.subheader("Against the benchmark")
        relative = result.relative_metrics
        grid = st.columns(2)
        grid[0].metric(
            "Alpha (annual)", format_metric("cagr", relative.alpha), help=GLOSSARY["Alpha"], border=True
        )
        grid[1].metric("Beta", f"{relative.beta:.2f}", help=GLOSSARY["Beta"], border=True)
        grid = st.columns(2)
        grid[0].metric("Correlation", f"{relative.correlation:.2f}", border=True)
        grid[1].metric(
            "Information ratio",
            format_metric("sharpe_ratio", relative.information_ratio),
            help=GLOSSARY["Information Ratio"],
            border=True,
        )
        interest = float(result.results["Interest"].sum())
        st.metric(
            "Interest earned on idle cash", f"${interest:,.2f}", help=GLOSSARY["Cash Yield"], border=True
        )


def render_trades(result: BacktestResult, prices: pd.DataFrame, inputs: Inputs, theme: ChartTheme) -> None:
    strategy = create_strategy(inputs.strategy_key, **inputs.params)
    overlays = indicator_overlays(strategy, prices)
    index = result.results.index
    chart(
        price_figure(
            result.results["Close"],
            result.trade_log,
            theme,
            None if overlays is None else overlays.loc[index],
        )
    )

    trades = result.trade_log
    metrics = result.metrics
    columns = st.columns(4)
    columns[0].metric("Trades", metrics.number_of_trades, border=True)
    columns[1].metric("Win rate", format_metric("win_rate", metrics.win_rate), border=True)
    columns[2].metric("Profit factor", format_metric("profit_factor", metrics.profit_factor), border=True)
    columns[3].metric("Average trade", format_metric("expectancy", metrics.expectancy), border=True)

    if trades.empty:
        st.info("No trades were made. Try a longer period or different parameters.", icon=":material/info:")
        return
    st.dataframe(
        trades.assign(**{"Exit Reason": trades["Exit Reason"].str.replace("_", " ")}),
        hide_index=True,
        column_config={
            "Entry Date": st.column_config.DateColumn(format="YYYY-MM-DD"),
            "Exit Date": st.column_config.DateColumn(format="YYYY-MM-DD"),
            "Entry Price": st.column_config.NumberColumn(format="%.2f"),
            "Exit Price": st.column_config.NumberColumn(format="%.2f"),
            "Commission": st.column_config.NumberColumn(format="dollar"),
            "P&L": st.column_config.NumberColumn(format="dollar"),
            "Return": st.column_config.NumberColumn(format="percent"),
        },
    )
    st.download_button(
        "Download trades (CSV)",
        trades.to_csv(index=False),
        file_name="trades.csv",
        mime="text/csv",
        icon=":material/download:",
        on_click="ignore",
    )


def render_robustness(result: BacktestResult, estimate: BootstrapResult | None, theme: ChartTheme) -> None:
    st.markdown(
        "A single backtest is one draw from what could have happened. Resampling its returns in 20-day "
        "blocks shows how far the result could plausibly move by chance."
    )
    if estimate is None:
        st.warning("At least 40 bars are needed to estimate confidence intervals.")
        return
    returns = result.excess_returns
    columns = st.columns(4)
    columns[0].metric(
        "Sharpe, 95% interval",
        f"{estimate.sharpe.low:.2f} to {estimate.sharpe.high:.2f}",
        help=GLOSSARY["Block Bootstrap"],
        border=True,
    )
    columns[1].metric(
        "Annual return, 95% interval",
        f"{estimate.annual_return.low:.1%} to {estimate.annual_return.high:.1%}",
        border=True,
    )
    columns[2].metric(
        "Probabilistic Sharpe",
        f"{probabilistic_sharpe_ratio(returns):.1%}",
        help=GLOSSARY["Probabilistic Sharpe Ratio"],
        border=True,
    )
    columns[3].metric(
        "Chance of beating buy and hold",
        f"{estimate.probability_beats_benchmark:.1%}",
        help="Share of resampled histories in which the strategy's Sharpe ratio beat buy and hold's.",
        border=True,
    )
    observed = float(returns.mean() / returns.std(ddof=1) * np.sqrt(252)) if returns.std(ddof=1) > 0 else 0.0
    chart(
        bootstrap_figure(estimate.sharpe_samples, estimate.sharpe.low, estimate.sharpe.high, observed, theme)
    )


def render_optimize(prices: pd.DataFrame, inputs: Inputs, theme: ChartTheme) -> None:
    st.markdown(
        "Search every parameter combination. The winner's score is **in-sample**, so it is flattered by "
        "selection. Two corrections follow: the deflated Sharpe ratio adjusts for how many combinations were "
        "tried, and the probability of backtest overfitting asks whether the winner would have stayed a "
        "winner on data it was not chosen on."
    )
    left, right, button = st.columns([2, 2, 1], vertical_alignment="bottom")
    points = left.slider("Grid points per parameter", 3, 10, 6)
    objective = right.selectbox(
        "Objective", OBJECTIVES, format_func=lambda name: name.replace("_", " ").title()
    )
    if button.button("Run search", type="primary", width="stretch", icon=":material/search:"):
        grid = build_grid(inputs.strategy_key, points)
        with st.spinner(f"Testing {len(grid)} combinations..."):
            st.session_state["search"] = (
                inputs.fingerprint(),
                grid_search(prices, inputs.strategy_key, grid, inputs.config, objective),
            )

    stored = st.session_state.get("search")
    if stored is None or stored[0] != inputs.fingerprint():
        st.info("Run a search to see how sensitive this strategy is to its settings.", icon=":material/tune:")
        return
    search = stored[1]
    parameters = [spec.name for spec in STRATEGIES[inputs.strategy_key].params]
    overfitting = search.overfitting.probability if search.overfitting else float("nan")
    columns = st.columns(4)
    columns[0].metric("Combinations tested", len(search.table), border=True)
    columns[1].metric("Best Sharpe (in-sample)", f"{search.table['sharpe_ratio'].max():.2f}", border=True)
    columns[2].metric(
        "Deflated Sharpe vs B&H",
        f"{search.deflated_active_sharpe:.1%}",
        help=GLOSSARY["Deflated Sharpe Ratio"],
        border=True,
    )
    columns[3].metric(
        "Probability of overfitting",
        "n/a" if math.isnan(overfitting) else f"{overfitting:.0%}",
        help=GLOSSARY["Probability of Backtest Overfitting"],
        border=True,
    )
    luck_tone = "good" if search.deflated_active_sharpe >= 0.95 else "bad"
    callout(
        luck_tone,
        f"**Selection luck.** {probability_verdict(search.deflated_active_sharpe)}",
        ":material/casino:",
    )
    if math.isnan(overfitting):
        overfit_tone = "neutral"
    else:
        overfit_tone = "good" if overfitting <= 0.1 else "bad" if overfitting >= 0.5 else "warning"
    callout(overfit_tone, f"**Overfitting.** {overfitting_verdict(overfitting)}", ":material/model_training:")
    st.caption(
        f"Best parameters: {search.best_params}. Deflated Sharpe vs zero: {search.deflated_sharpe:.1%}."
    )

    left, right = st.columns(2)
    with left:
        chart(search_figure(search.table, parameters, search.objective, theme))
    with right:
        if search.overfitting is not None:
            chart(overfitting_figure(search.overfitting.logits, theme))
    st.dataframe(search.table.round(3), hide_index=True)


def render_walk_forward(prices: pd.DataFrame, inputs: Inputs, theme: ChartTheme) -> None:
    st.markdown(
        "**The fair test.** Parameters are chosen on one window and traded on the next, unseen window, then "
        "the process rolls forward. The equity curve below is made only of decisions that could have been "
        "taken at the time."
    )
    columns = st.columns([1, 1, 1, 1, 1], vertical_alignment="bottom")
    train_bars = columns[0].number_input("Training bars", 100, 2520, 756, 21)
    test_bars = columns[1].number_input("Test bars", 21, 1260, 252, 21)
    points = columns[2].slider("Grid points", 3, 8, 5)
    anchored = columns[3].toggle("Anchored", help="Grow the training window instead of sliding it.")
    if columns[4].button("Run walk-forward", type="primary", width="stretch", icon=":material/fast_forward:"):
        grid = build_grid(inputs.strategy_key, points)
        try:
            with st.spinner("Optimizing and testing each fold..."):
                st.session_state["walk"] = (
                    inputs.fingerprint(),
                    walk_forward(
                        prices,
                        inputs.strategy_key,
                        grid,
                        inputs.config,
                        train_bars=int(train_bars),
                        test_bars=int(test_bars),
                        anchored=anchored,
                    ),
                )
        except ValueError as error:
            st.session_state.pop("walk", None)
            st.error(str(error))

    stored = st.session_state.get("walk")
    if stored is None or stored[0] != inputs.fingerprint():
        st.info(
            "Run walk-forward to see how the strategy does on data it was not tuned on.",
            icon=":material/timeline:",
        )
        return
    result = stored[1]
    columns = st.columns(4)
    columns[0].metric(
        "Out-of-sample Sharpe",
        format_metric("sharpe_ratio", result.metrics.sharpe_ratio),
        _points(result.metrics.sharpe_ratio, result.benchmark_metrics.sharpe_ratio, percent=False),
        delta_description="vs B&H",
        border=True,
    )
    columns[1].metric(
        "Out-of-sample CAGR",
        format_metric("cagr", result.metrics.cagr),
        _points(result.metrics.cagr, result.benchmark_metrics.cagr),
        delta_description="vs B&H",
        border=True,
    )
    columns[2].metric(
        "Max drawdown",
        format_metric("max_drawdown", result.metrics.max_drawdown),
        _points(result.metrics.max_drawdown, result.benchmark_metrics.max_drawdown),
        delta_description="vs B&H",
        border=True,
    )
    columns[3].metric(
        "Efficiency",
        format_metric("sharpe_ratio", result.efficiency),
        help=GLOSSARY["Walk-Forward Efficiency"],
        border=True,
    )
    chart(equity_figure(result.equity, result.benchmark, theme, "Out-of-sample growth"))
    chart(fold_scores_figure(result.folds, result.objective, theme))
    folds = result.folds.assign(Parameters=result.folds["Parameters"].astype(str))
    st.dataframe(
        folds.round({"In-Sample Score": 3, "Out-of-Sample Score": 3, "Test Return": 4}),
        hide_index=True,
        column_config={
            name: st.column_config.DateColumn(format="YYYY-MM-DD")
            for name in ("Train Start", "Train End", "Test Start", "Test End")
        }
        | {"Test Return": st.column_config.NumberColumn(format="percent")},
    )


def render_compare(prices: pd.DataFrame, inputs: Inputs, theme: ChartTheme) -> None:
    st.markdown(
        "Every strategy at its default settings, on the same data and with the same costs. Nothing is tuned "
        "here, so none of these numbers is flattered by a parameter search."
    )
    comparison = cached_comparison(prices, inputs.config)
    if comparison.table.empty:
        st.warning("This history is too short for any strategy's indicators.")
        return
    chart(comparison_figure(comparison.equity, theme))
    columns = ["Strategy", "cagr", "sharpe_ratio", "max_drawdown", "exposure", "number_of_trades"]
    table = comparison.table[columns].rename(columns=METRIC_LABELS)
    st.dataframe(
        table,
        hide_index=True,
        column_config={
            "CAGR": st.column_config.NumberColumn(format="percent"),
            "Sharpe Ratio": st.column_config.NumberColumn(format="%.2f"),
            "Max Drawdown": st.column_config.NumberColumn(format="percent"),
            "Exposure": st.column_config.ProgressColumn(format="percent", min_value=0.0, max_value=1.0),
        },
    )


def render_learn() -> None:
    left, right = st.columns([2, 3], gap="large")
    with left:
        st.subheader("How TradStrat tests an idea")
        st.markdown(
            """
1. **Simulate honestly.** A signal formed at today's close trades at tomorrow's open, after commission
   and slippage, in whole shares. Prices include dividends and idle cash earns Treasury bill yields.
2. **Compare with doing nothing clever.** Buy and hold runs through the same simulator with the same costs.
3. **Ask whether it was luck.** A block bootstrap puts intervals on every headline number.
4. **Correct for searching.** The deflated Sharpe ratio and the probability of backtest overfitting
   discount a winner picked from many trials.
5. **Test on unseen data.** Walk-forward validation only ever trades parameters chosen on the past.
"""
        )
    with right:
        st.subheader("Glossary")
        query = st.text_input("Search terms", placeholder="Try: Sharpe", label_visibility="collapsed")
        query = query.strip().lower()
        for term, definition in sorted(GLOSSARY.items()):
            if query in term.lower() or query in definition.lower():
                st.markdown(f"**{term}.** {definition}")


def report_builder(
    result: BacktestResult, prices: pd.DataFrame, inputs: Inputs, estimate: BootstrapResult | None
):
    """Build the HTML report only when someone clicks download."""

    def build() -> str:
        strategy = create_strategy(inputs.strategy_key, **inputs.params)
        context = ReportContext(
            strategy=STRATEGIES[inputs.strategy_key].label,
            asset=inputs.label,
            parameters={
                spec.name: getattr(strategy, spec.name) for spec in STRATEGIES[inputs.strategy_key].params
            },
            overlays=indicator_overlays(strategy, prices),
        )
        return build_report(result, context, estimate=estimate)

    return build


def render_header() -> None:
    st.html(
        f"""
<div style="display:flex;align-items:center;gap:12px;margin-bottom:-6px;">
  <img src="data:image/svg+xml;base64,{LOGO}" width="34" height="34" alt="">
  <div style="font-family:Inter,system-ui,sans-serif;font-size:28px;font-weight:700;letter-spacing:-.02em;">
    TradStrat</div>
</div>"""
    )
    st.markdown("Backtest a trading idea, then find out whether to believe it.")


def main() -> None:
    st.set_page_config(page_title="TradStrat", page_icon=":material/monitoring:", layout="wide")
    theme = theme_for(st.context.theme.type == "dark")
    inputs = read_sidebar()

    header, actions = st.columns([4, 1], vertical_alignment="bottom")
    with header:
        render_header()

    if inputs.config is None:
        st.error(inputs.config_error)
        return
    try:
        strategy = create_strategy(inputs.strategy_key, **inputs.params)
    except ValueError as error:
        st.error(f"These parameters don't work together: {error}", icon=":material/tune:")
        return
    try:
        prices, warning = load_prices(inputs)
    except DataError as error:
        st.error(str(error), icon=":material/error:")
        return
    if len(prices) <= strategy.warmup + 1:
        st.warning(
            f"{STRATEGIES[inputs.strategy_key].label} needs more than {strategy.warmup} bars of history, "
            f"but only {len(prices)} are available. Use a longer period or a shorter window."
        )
        return

    result = run_backtest(prices, strategy, inputs.config)
    estimate = (
        cached_bootstrap(result.excess_returns, result.benchmark_excess_returns)
        if len(result.excess_returns) >= 40
        else None
    )
    with actions:
        st.download_button(
            "Download report",
            report_builder(result, prices, inputs, estimate),
            file_name=f"tradstrat-{inputs.strategy_key}-{inputs.label.lower().replace(' ', '-')}.html",
            mime="text/html",
            icon=":material/description:",
            on_click="ignore",
            width="stretch",
        )
    dataset_caption(prices, inputs, warning)

    tabs = st.tabs(
        [
            ":material/insights: Overview",
            ":material/swap_vert: Trades",
            ":material/verified: Robustness",
            ":material/tune: Optimize",
            ":material/fast_forward: Walk-forward",
            ":material/leaderboard: Compare",
            ":material/school: Learn",
        ]
    )
    with tabs[0]:
        render_overview(result, estimate, theme)
    with tabs[1]:
        render_trades(result, prices, inputs, theme)
    with tabs[2]:
        render_robustness(result, estimate, theme)
    with tabs[3]:
        render_optimize(prices, inputs, theme)
    with tabs[4]:
        render_walk_forward(prices, inputs, theme)
    with tabs[5]:
        render_compare(prices, inputs, theme)
    with tabs[6]:
        render_learn()


if __name__ == "__main__":
    main()
