"""Streamlit dashboard for TradStrat.

Run it with ``streamlit run app.py``. Every number on the page comes from the ``tradstrat`` package, so
the dashboard, the command line and the HTML report always agree. This file only lays the results out.
"""

from __future__ import annotations

import base64
import html
import io
import math
import re
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
from tradstrat.glossary import GLOSSARY, annotate
from tradstrat.guide import guide_for
from tradstrat.metrics import calendar_year_returns, rolling_sharpe
from tradstrat.reporting import (
    METRIC_LABELS,
    Round,
    Verdict,
    assess,
    assess_search,
    comparison_table,
    format_metric,
    rounds,
)
from tradstrat.rules import EXAMPLES, RuleError, RuleStrategy, compile_rule
from tradstrat.stats import BootstrapResult, bootstrap, probabilistic_sharpe_ratio
from tradstrat.strategies import (
    STRATEGIES,
    StrategyRef,
    create_strategy,
    indicator_overlays,
    latest_signal,
    strategy_class,
)
from tradstrat.tearsheet import ReportContext, build_report
from tradstrat.validation import (
    OBJECTIVES,
    Comparison,
    build_grid,
    compare_strategies,
    grid_search,
    walk_forward,
)

REPOSITORY: Final[str] = "https://github.com/Elarionitis/TradStrat"
STRATEGY_KEYS: Final[list[str]] = [key for key in STRATEGIES if key != "buy_and_hold"]
CUSTOM: Final[str] = "custom"
STRATEGY_CHOICES: Final[list[str]] = [*STRATEGY_KEYS, CUSTOM]
DEFAULT_RULE: Final[tuple[str, str]] = EXAMPLES["Golden cross"]
# A search runs one backtest per combination, so a rule with many numbers can ask for far too many.
MAX_COMBINATIONS: Final[int] = 1_500
MAX_RULE_CHARS: Final[int] = 300
# The building blocks of a custom rule, with an example and what it means.
RULE_PARTS: Final[list[tuple[str, str]]] = [
    ("close", "The closing price."),
    ("sma(50)", "The average closing price of the last 50 days."),
    ("ema(20)", "Like sma, but recent days count for more, so it reacts faster."),
    ("rsi(14)", "The RSI, from 0 to 100. Under 30 is often called oversold, and over 70 overbought."),
    (
        "high(252)",
        "The highest price of the previous 252 days, not counting today. 252 days is about a year.",
    ),
    ("low(20)", "The lowest price of the previous 20 days, not counting today."),
    ("return(126)", "The percentage change over 126 days. return(126) > 0 means up over about six months."),
    ("zscore(20)", "How far the price is from its 20-day average, in standard deviations. Below -2 is rare."),
]
RULE_OPERATORS: Final[list[tuple[str, str]]] = [
    ("> and <", "Above or below, on each day."),
    ("crosses above", "Was at or below yesterday, and is above today. It happens on a single day."),
    ("crosses below", "Was at or above yesterday, and is below today."),
    ("and, or", "Combine conditions: all must hold, or any one is enough."),
]
QUICK_TICKERS: Final[list[str]] = ["SPY", "QQQ", "AAPL", "NVDA", "BTC-USD", "GLD"]
PERIODS: Final[dict[str, int | None]] = {"1Y": 1, "3Y": 3, "5Y": 5, "10Y": 10, "20Y": 20, "Max": None}
PERIOD_NAMES: Final[dict[str, str]] = {
    "1Y": "1 year",
    "3Y": "3 years",
    "5Y": "5 years",
    "10Y": "10 years",
    "20Y": "20 years",
    "Max": "all history",
}
SOURCES: Final[dict[str, str]] = {"Yahoo": "Ticker", "CSV": "CSV file", "Sample": "Demo"}
EARLIEST: Final[pd.Timestamp] = pd.Timestamp("1970-01-01")
CASH_OPTIONS: Final[list[str]] = ["Treasury bills", "Fixed rate", "None"]
TRAIN_WINDOWS: Final[dict[str, int]] = {
    "6 months": 126,
    "1 year": 252,
    "2 years": 504,
    "3 years": 756,
    "5 years": 1260,
}
TEST_WINDOWS: Final[dict[str, int]] = {"3 months": 63, "6 months": 126, "1 year": 252, "2 years": 504}
TABS: Final[list[str]] = ["Results", "Trades", "Tune", "Compare", "Guide"]
TONE_MARKS: Final[dict[str, str]] = {"good": "✓", "warning": "!", "bad": "✕", "neutral": "•"}
PALETTE: Final[dict[str, dict[str, str]]] = {
    "light": {
        "good": "#006300",
        "good-wash": "#e9f4e7",
        "warning": "#8a5a00",
        "warning-wash": "#fdf4de",
        "bad": "#b42b2b",
        "bad-wash": "#fbeaea",
        "neutral": "#52514e",
        "neutral-wash": "#ece8de",
        "ink": "#16140f",
        "muted": "#615d53",
        "border": "#dcd6c8",
        "surface": "#fbf9f4",
        "accent": "#2a78d6",
        "rule": "#2a78d6",
        "hold": "#eb6834",
        "surface-2": "#f1ede4",
        "tip": "#1f1d19",
        "tip-ink": "#f6f3ec",
    },
    "dark": {
        "good": "#3fbf3f",
        "good-wash": "#132613",
        "warning": "#fab219",
        "warning-wash": "#2b2210",
        "bad": "#ee7a7a",
        "bad-wash": "#2e1717",
        "neutral": "#c3c2b7",
        "neutral-wash": "#1f1d19",
        "ink": "#ece6d8",
        "muted": "#a39d8f",
        "border": "#2f2c26",
        "surface": "#191814",
        "accent": "#3987e5",
        "rule": "#3987e5",
        "hold": "#d95926",
        "surface-2": "#201e1a",
        "tip": "#ece6d8",
        "tip-ink": "#13120f",
    },
}
ASSETS: Final[Path] = Path(__file__).parent / "assets"
STYLESHEET: Final[str] = (ASSETS / "dashboard.css").read_text()


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
    strategy: StrategyRef | None
    params: dict[str, float]
    config: BacktestConfig | None
    config_error: str | None
    cash: str
    fixed_rate: float
    rule: str = ""
    rule_error: str | None = None

    @property
    def strategy_class(self) -> Any:
        assert self.strategy is not None
        return strategy_class(self.strategy)

    def fingerprint(self) -> str:
        """Identifies the data and settings, so stored search results are never shown for other inputs."""
        upload = getattr(self.upload, "file_id", None)
        config = asdict(self.config) if self.config else None
        return repr(
            (
                self.source,
                self.ticker,
                self.start,
                self.end,
                upload,
                self.strategy_key,
                self.rule,
                config,
                self.cash,
            )
        )


# Data loading, cached so that moving a slider does not refetch anything. Every cache expires and has a size
# limit, so a public server's memory stays bounded however many visitors try however many tickers.


@st.cache_data(show_spinner="Fetching market data...", ttl=3600, max_entries=200)
def cached_fetch(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return fetch_ohlcv(ticker, start, end)


@st.cache_data(show_spinner="Fetching Treasury bill yields...", ttl=3600, max_entries=50)
def cached_cash_rate(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    return fetch_cash_rate(start, end)


@st.cache_data(show_spinner=False, ttl=3600, max_entries=20)
def cached_upload(contents: bytes, name: str) -> pd.DataFrame:
    return adjust_for_dividends(read_price_csv(io.BytesIO(contents), name))


@st.cache_data(show_spinner="Checking how much of the result could be luck...", ttl=3600, max_entries=200)
def cached_bootstrap(returns: pd.Series, benchmark: pd.Series) -> BootstrapResult:
    return bootstrap(returns, benchmark)


@st.cache_data(show_spinner="Running every strategy...", ttl=3600, max_entries=50)
def cached_comparison(
    prices: pd.DataFrame, config: BacktestConfig, rule: tuple[str, str] | None = None
) -> Comparison:
    """Every built-in strategy at its defaults, plus the custom rule when there is one."""
    keys: list[StrategyRef] = list(STRATEGY_KEYS)
    if rule is not None:
        keys.insert(0, compile_rule(*rule))
    return compare_strategies(prices, config, keys)


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


# Styling. Streamlit has no component for a definition tooltip or a verdict card, so these few classes
# are injected once and reused by every block of text on the page.


def inject_styles(theme: ChartTheme) -> None:
    colors = PALETTE[theme.name]
    names = ("ink", "muted", "border", "surface", "surface-2", "accent", "rule", "hold", "tip", "tip-ink")
    variables = "".join(f"--ts-{name}:{colors[name]};" for name in names)
    tones = "".join(
        f".ts-tone-{tone}{{--tone:{colors[tone]};--wash:{colors[tone + '-wash']};}}"
        for tone in ("good", "warning", "bad", "neutral")
    )
    st.html(f"<style>:root{{{variables}}}{tones}{STYLESHEET}</style>")


def rich(text: str, seen: set[str] | None = None) -> str:
    """Explain glossary terms on hover, and turn **bold** markers into bold text."""
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", annotate(text, seen))


def prose(text: str, seen: set[str] | None = None, *, muted: bool = False) -> None:
    """A paragraph whose jargon explains itself on hover."""
    css = "ts ts-muted" if muted else "ts"
    st.html(f'<div class="{css}">{rich(text, seen)}</div>')


def section(title: str, intro: str | None = None) -> None:
    body = f'<div class="ts-muted">{rich(intro)}</div>' if intro else ""
    st.html(f'<div class="ts"><div class="ts-section-title">{html.escape(title)}</div>{body}</div>')


def chart(figure: Any) -> None:
    st.plotly_chart(figure, theme=None, config={"displaylogo": False})


EVIDENCE: Final[dict[str, tuple[str, str]]] = {
    "good": ("rule", "Strong evidence"),
    "warning": ("mixed", "Mixed evidence"),
    "bad": ("hold", "Weak evidence"),
    "neutral": ("mixed", "Note"),
}


def render_verdict(verdict: Verdict, seen: set[str] | None = None) -> None:
    """A calm card: how strong the evidence is, a headline, and the findings behind it.

    Results are information, not errors, so the card never takes an alarm color. The tone shows only in a
    small label and in each finding's mark.
    """
    seen = set() if seen is None else seen
    side, label = EVIDENCE[verdict.tone]
    findings = "".join(
        f'<div class="ts-finding ts-tone-{finding.tone}">'
        f'<span class="ts-mark">{TONE_MARKS[finding.tone]}</span>'
        f"<span><strong>{html.escape(finding.title)}.</strong> {rich(finding.detail, seen)}</span></div>"
        for finding in verdict.findings
    )
    st.html(
        f"""<div class="ts ts-answer ts-rise">
  <div class="ts-answer-head">
    <div class="ts-answer-title">
      <span class="ts-pill ts-pill-{side}">{label}</span>{html.escape(verdict.headline)}
    </div>
    <p class="ts-takeaway">{rich(verdict.summary, seen)}</p>
  </div>
  <div class="ts-findings">{findings}</div>
</div>"""
    )


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
    key = f"{strategy_key}.{spec.name}"
    st.session_state.setdefault(key, kind(spec.default))
    return st.slider(
        spec.label,
        min_value=kind(spec.low),
        max_value=kind(spec.high),
        step=kind(spec.step),
        help=spec.help,
        key=key,
    )


def literal(text: str) -> str:
    """Escape Markdown, so text quoted from a rule, which may come from a shared link, shows as typed."""
    return re.sub(r"([\\`*_{}\[\]()<>#+\-.!|~$])", r"\\\1", text)


def _use_example() -> None:
    choice = st.session_state.get("rule_example")
    if choice:
        st.session_state["rule_buy"], st.session_state["rule_sell"] = EXAMPLES[choice]


def _edited_rule() -> None:
    """Editing a rule by hand deselects the example it started from."""
    st.session_state["rule_example"] = None


def read_rule() -> tuple[StrategyRef | None, str, str | None]:
    """The custom-rule editor. Returns the compiled rule, its text, and an error if it cannot be read."""
    st.session_state.setdefault("rule_buy", DEFAULT_RULE[0])
    st.session_state.setdefault("rule_sell", DEFAULT_RULE[1])
    if "rule_example" not in st.session_state:
        current = (st.session_state["rule_buy"], st.session_state["rule_sell"])
        st.session_state["rule_example"] = next(
            (name for name, rule in EXAMPLES.items() if rule == current), None
        )
    st.pills("Start from an example", list(EXAMPLES), key="rule_example", on_change=_use_example)
    buy = st.text_input("Buy when", key="rule_buy", on_change=_edited_rule, max_chars=MAX_RULE_CHARS)
    sell = st.text_input(
        "Sell when",
        key="rule_sell",
        on_change=_edited_rule,
        max_chars=MAX_RULE_CHARS,
        placeholder="Empty: sell once the buy rule stops being true",
    )
    text = f"{buy.strip()} | {sell.strip()}"
    try:
        return compile_rule(buy, sell), text, None
    except RuleError as error:
        st.error(literal(str(error)), icon=":material/error:")
        return None, text, str(error)


# Shareable links. The address bar always holds the market, history and strategy, so copying it shares the
# exact test. Advanced settings are left out to keep links short.


def has_started(inputs: Inputs) -> bool:
    """Whether to show results rather than the landing page.

    Results appear once a visitor picks a question, opens a shared link, or changes any setting.
    """
    state = st.session_state
    fingerprint = inputs.fingerprint() + repr(sorted(inputs.params.items()))
    previous = state.get("last_inputs")
    state["last_inputs"] = fingerprint
    if previous is not None and previous != fingerprint:
        state["started"] = True
    return bool(state.get("started"))


def restore_from_link() -> None:
    """On the first run of a session, start from the settings in the link, ignoring anything invalid."""
    if st.session_state.get("link_read"):
        return
    st.session_state["link_read"] = True
    query = st.query_params
    if query.get("source") in SOURCES:
        st.session_state["source"] = query["source"]
    if query.get("ticker"):
        st.session_state["ticker"] = query["ticker"].strip().upper()[:20]
    if query.get("history") in PERIODS:
        st.session_state["history"] = query["history"]
    strategy = query.get("strategy")
    if strategy not in STRATEGY_CHOICES:
        return
    st.session_state["strategy"] = strategy
    st.session_state["started"] = True  # A shared link goes straight to its results.
    if strategy == CUSTOM:
        st.session_state["rule_buy"] = query.get("buy", "")[:MAX_RULE_CHARS]
        st.session_state["rule_sell"] = query.get("sell", "")[:MAX_RULE_CHARS]
        return
    for spec in STRATEGIES[strategy].params:
        try:
            value = float(query.get(spec.name, ""))
        except ValueError:
            continue
        if spec.low <= value <= spec.high:
            st.session_state[f"{strategy}.{spec.name}"] = int(value) if spec.is_integer else value


def write_link(inputs: Inputs, history: str | None) -> None:
    link: dict[str, str] = {"source": inputs.source}
    if inputs.source == "Yahoo":
        link["ticker"] = inputs.ticker.strip().upper()
        if history:
            link["history"] = history
    link["strategy"] = inputs.strategy_key
    if inputs.strategy_key == CUSTOM:
        link["buy"] = st.session_state.get("rule_buy", "")
        if st.session_state.get("rule_sell"):
            link["sell"] = st.session_state["rule_sell"]
    else:
        defaults = {spec.name: spec.default for spec in STRATEGIES[inputs.strategy_key].params}
        link |= {name: f"{value:g}" for name, value in inputs.params.items() if value != defaults[name]}
    if dict(st.query_params) != link:
        st.query_params.from_dict(link)


def read_sidebar() -> Inputs:
    today = pd.Timestamp.today().normalize()
    with st.sidebar:
        st.subheader("Market")
        st.session_state.setdefault("source", "Yahoo")
        source = st.segmented_control(
            "Data source",
            list(SOURCES),
            format_func=SOURCES.__getitem__,
            key="source",
            required=True,
            width="stretch",
            label_visibility="collapsed",
        )
        ticker, start, end, upload, label, period = "", today, today, None, "Demo data", None
        if source == "Yahoo":
            st.session_state.setdefault("ticker", "SPY")
            ticker = st.text_input(
                "Ticker",
                key="ticker",
                on_change=_typed_ticker,
                placeholder="e.g. SPY, AAPL, BTC-USD",
                help=GLOSSARY["Ticker"] + " Any symbol on Yahoo Finance works, including indices like ^GSPC.",
            )
            st.pills(
                "Popular",
                QUICK_TICKERS,
                key="ticker_pick",
                on_change=_pick_ticker,
                label_visibility="collapsed",
            )
            st.session_state.setdefault("history", "10Y")
            period = st.select_slider("History", list(PERIODS), key="history")
            years = PERIODS[period]
            if st.toggle("Pick exact dates", key="custom_dates"):
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
                "A simulated market for trying the app offline. Its results say nothing about real ones."
            )

        st.subheader("Strategy")
        strategy_key = st.selectbox(
            "Strategy",
            STRATEGY_CHOICES,
            format_func=lambda key: "Write your own rule" if key == CUSTOM else STRATEGIES[key].label,
            key="strategy",
            label_visibility="collapsed",
        )
        rule, rule_error = "", None
        if strategy_key == CUSTOM:
            strategy, rule, rule_error = read_rule()
            if st.button("How rules work", type="tertiary", icon=":material/menu_book:"):
                rules_dialog()
            params = {spec.name: spec.default for spec in strategy.params} if strategy else {}
        else:
            strategy = strategy_key
            st.caption(guide_for(STRATEGIES[strategy_key]).summary)
            if st.button("How this strategy works", type="tertiary", icon=":material/menu_book:"):
                strategy_dialog(strategy_key)
            params = {spec.name: param_widget(strategy_key, spec) for spec in STRATEGIES[strategy_key].params}

        st.space("small")
        with st.expander("Advanced settings"):
            st.caption("The defaults are realistic for a low-cost US broker.")
            capital = st.number_input("Starting capital ($)", min_value=1_000.0, value=10_000.0, step=1_000.0)
            commission_bps = st.number_input(
                "Commission (bps)",
                min_value=0.0,
                max_value=100.0,
                value=10.0,
                step=1.0,
                help=GLOSSARY["Commission"] + " " + GLOSSARY["Basis Point"],
            )
            slippage_bps = st.number_input(
                "Slippage (bps)",
                min_value=0.0,
                max_value=100.0,
                value=5.0,
                step=1.0,
                help=GLOSSARY["Slippage"] + " " + GLOSSARY["Basis Point"],
            )
            position_fraction = st.slider(
                "Share of portfolio per trade",
                10,
                100,
                100,
                10,
                format="%d%%",
                help=GLOSSARY["Position Sizing"],
            )
            use_stop = st.toggle("Stop loss", help=GLOSSARY["Stop Loss"])
            stop_loss, trailing = None, False
            if use_stop:
                stop_loss = st.slider("Sell after a fall of", 1, 50, 10, 1, format="%d%%") / 100
                trailing = st.toggle("Trail the stop", help=GLOSSARY["Trailing Stop"])
            use_target = st.toggle("Volatility targeting", help=GLOSSARY["Volatility Targeting"])
            target = (
                st.slider("Target yearly volatility", 5, 50, 15, 1, format="%d%%") / 100
                if use_target
                else None
            )
            cash = st.radio(
                "Uninvested cash earns",
                CASH_OPTIONS,
                index=0 if source == "Yahoo" else 2,
                help=GLOSSARY["Cash Yield"],
            )
            fixed_rate = 0.0
            if cash == "Fixed rate":
                fixed_rate = st.number_input("Annual rate (%)", 0.0, 20.0, 4.0, 0.25) / 100

    try:
        config = BacktestConfig(
            initial_capital=capital,
            commission_rate=commission_bps / 10_000,
            slippage_bps=slippage_bps,
            position_fraction=position_fraction / 100,
            stop_loss=stop_loss,
            trailing_stop=trailing,
            target_volatility=target,
        )
        error = None
    except ValueError as exc:
        config, error = None, str(exc)
    inputs = Inputs(
        source or "Yahoo",
        label,
        ticker,
        start,
        end,
        upload,
        strategy_key,
        strategy,
        params,
        config,
        error,
        cash,
        fixed_rate,
        rule,
        rule_error,
    )
    st.session_state["_history_for_link"] = None if st.session_state.get("custom_dates") else period
    return inputs


# Explanations.


def guide_html(key: str) -> str:
    """Everything a newcomer needs to understand one strategy, as a block of HTML."""
    guide = guide_for(STRATEGIES[key])
    seen: set[str] = set()
    settings = "".join(
        f"<li><strong>{html.escape(spec.label)}</strong> (default {spec.default}). "
        f"{rich(spec.help, seen)}</li>"
        for spec in STRATEGIES[key].params
    )
    settings_block = f'<div class="ts-label">Settings</div><ul>{settings}</ul>' if settings else ""
    cells = (
        ("Buys when", guide.buys),
        ("Sells when", guide.sells),
        ("Tends to work in", guide.works_when),
        ("Tends to struggle in", guide.struggles_when),
    )
    grid = "".join(
        f'<div><div class="ts-label">{label}</div>{rich(text, seen)}</div>' for label, text in cells if text
    )
    return f"""<div class="ts">
  <span class="ts-chip ts-tone-neutral">{html.escape(guide.style)}</span>
  <p><strong>{rich(guide.summary, seen)}</strong></p>
  {f'<p class="ts-muted">{rich(guide.idea, seen)}</p>' if guide.idea else ""}
  {f'<div class="ts-grid2">{grid}</div>' if grid else ""}
  {settings_block}
</div>"""


def rules_html() -> str:
    """The rule language on one card: what each building block means, and how to combine them."""
    seen: set[str] = set()
    parts = "".join(
        f"<tr><td><code>{html.escape(code)}</code></td><td>{rich(meaning, seen)}</td></tr>"
        for code, meaning in RULE_PARTS
    )
    operators = "".join(
        f"<tr><td><code>{html.escape(code)}</code></td><td>{rich(meaning, seen)}</td></tr>"
        for code, meaning in RULE_OPERATORS
    )
    examples = "".join(
        f"<li><strong>{html.escape(name)}.</strong> Buy when <code>{html.escape(buy)}</code>"
        + (f", sell when <code>{html.escape(sell)}</code>" if sell else ", and hold while it stays true")
        + ".</li>"
        for name, (buy, sell) in EXAMPLES.items()
    )
    return f"""<div class="ts">
  <p>A rule says when to buy and when to sell, in short phrases such as <code>close &gt; sma(200)</code>.
  Change any number, such as the 200, to test a variation. Leave <strong>Sell when</strong> empty to own the
  asset only while the buy rule is true.</p>
  <div class="ts-label">What a rule can look at</div>
  <table class="ts-table">{parts}</table>
  <div class="ts-label">How to compare and combine</div>
  <table class="ts-table">{operators}</table>
  <div class="ts-label">Examples</div>
  <ul>{examples}</ul>
  <p class="ts-muted ts-small">A rule only sees prices up to today's close, and trades at the next day's open,
  so it can never peek at the future. The Tune tab can search every number in your rule, and checks the winner
  for {rich("overfitting", seen)} just as it does for the built-in strategies.</p>
</div>"""


@st.dialog("How rules work", width="large")
def rules_dialog() -> None:
    st.html(rules_html())


@st.dialog("How this strategy works", width="large")
def strategy_dialog(key: str) -> None:
    st.subheader(STRATEGIES[key].label)
    st.html(guide_html(key))
    st.caption("The Trades tab draws every buy and sell this strategy made on the price chart.")


def logo(theme: ChartTheme) -> str:
    """The mark in the theme's colors: a solid line for the rule over a dotted one for holding."""
    colors = PALETTE[theme.name]
    svg = (ASSETS / "logo.svg").read_text().replace("#8a8478", colors["ink"])
    svg = svg.replace("#2a78d6", colors["rule"]).replace("#eb6834", colors["hold"])
    return base64.b64encode(svg.encode()).decode()


def render_header(theme: ChartTheme, *, tagline: bool) -> None:
    words = "Test a trading idea on past prices, then find out whether to believe it."
    line = f'<div class="ts-tagline">{words}</div>' if tagline else ""
    st.html(
        f"""<div class="ts"><div class="ts-header">
  <img src="data:image/svg+xml;base64,{logo(theme)}" width="34" height="34" alt="">
  <div><div class="ts-wordmark">TradStrat</div>{line}</div>
</div></div>"""
    )


@dataclass(frozen=True)
class Question:
    """A ready-made test, phrased as the question a newcomer would ask."""

    key: str
    tag: str
    question: str
    detail: str
    ticker: str
    history: str
    strategy: str
    rule: tuple[str, str] = ("", "")


QUESTIONS: Final[list[Question]] = [
    Question(
        "golden_cross",
        "Trend following",
        "Does the famous golden cross beat the S&P 500?",
        "Own the market only while its 50-day average is above its 200-day average.",
        "SPY",
        "20Y",
        "ma_crossover",
    ),
    Question(
        "bitcoin_trend",
        "Trend filter",
        "Could one simple rule have dodged Bitcoin's crashes?",
        "Hold Bitcoin only while it trades above its 200-day average price.",
        "BTC-USD",
        "Max",
        "trend_filter",
    ),
    Question(
        "nasdaq_dip",
        "Buy the dip",
        "Does buying sharp dips in the Nasdaq pay off?",
        "Buy after a few days of heavy selling, but only while the long-term trend is up.",
        "QQQ",
        "20Y",
        CUSTOM,
        EXAMPLES["Dip in an uptrend"],
    ),
    Question(
        "momentum_crash",
        "Momentum",
        "Would momentum have protected you in 2008?",
        "Own the S&P 500 only while it is up over the last six months.",
        "SPY",
        "20Y",
        "momentum",
    ),
    Question(
        "gold_breakout",
        "Breakout",
        "Does buying gold at new highs work?",
        "Buy when gold closes at a 55-day high, and sell at a 20-day low.",
        "GLD",
        "20Y",
        "donchian",
    ),
    Question(
        "nvidia_macd",
        "Chart signal",
        "Is the popular MACD signal worth following on Nvidia?",
        "Own Nvidia while its MACD line is above its signal line.",
        "NVDA",
        "10Y",
        "macd",
    ),
]


def ask(question: Question) -> None:
    """Load a ready-made test into the sidebar and show its results."""
    state = st.session_state
    state["source"] = "Yahoo"
    state["ticker"] = question.ticker
    state["ticker_pick"] = question.ticker if question.ticker in QUICK_TICKERS else None
    state["history"] = question.history
    state["custom_dates"] = False
    state["strategy"] = question.strategy
    if question.strategy == CUSTOM:
        state["rule_buy"], state["rule_sell"] = question.rule
        state["rule_example"] = next((name for name, rule in EXAMPLES.items() if rule == question.rule), None)
    else:
        for spec in STRATEGIES[question.strategy].params:
            state[f"{question.strategy}.{spec.name}"] = spec.default
    start()


def start() -> None:
    st.session_state["started"] = True


def use_demo_data() -> None:
    st.session_state["source"] = "Sample"


def render_problem(title: str, text: str) -> None:
    """A calm note for something the visitor can fix, so it never reads as the site breaking."""
    title, text = html.escape(title, quote=False), html.escape(text, quote=False)
    st.html(
        f"""<div class="ts ts-card ts-problem"><div class="ts-section-title">{title}</div>
<div class="ts-muted">{text}</div></div>"""
    )


def render_data_problem(inputs: Inputs, error: DataError) -> None:
    """Explain a failed price download in plain words, and offer a way to keep going."""
    if inputs.source == "CSV":
        title = "Upload a price file to begin" if inputs.upload is None else "This file could not be read"
        render_problem(title, error.message)
        return
    render_problem(
        f"We couldn't load prices for {inputs.ticker.strip().upper() or 'that ticker'}",
        "Check the spelling as Yahoo Finance writes it, such as BTC-USD for Bitcoin or ^GSPC for the S&P 500 "
        "index. If the symbol is right, Yahoo may be busy for a moment: try again shortly, or explore with "
        "the demo data meanwhile.",
    )
    st.button("Use demo data instead", on_click=use_demo_data, icon=":material/science:")
    if error.details:
        with st.expander("Technical details"):
            st.code(error.details, language=None, wrap_lines=True)


def show_ideas() -> None:
    st.session_state["started"] = False
    st.query_params.clear()


HERO_START_YEAR: Final[int] = 2006
HERO_YEARS: Final[int] = 19


def _hero_series() -> tuple[np.ndarray, np.ndarray, int, int]:
    """Simulated monthly prices with a 2008-style crash, and a 10-month trend rule run on them.

    The rule is real: it owns the asset only while the previous month closed above its 10-month average, so
    the drawing shows what such a rule does, rather than a line drawn to look good. The seed gives a typical
    outcome: a much smaller crash, but a slightly lower ending value than holding.
    """
    rng = np.random.default_rng(40)
    months = HERO_YEARS * 12
    returns = rng.normal(0.008, 0.035, months)
    crash_start, crash_end = 22, 38
    returns[crash_start:crash_end] += np.linspace(-0.03, -0.055, crash_end - crash_start)
    returns[crash_end : crash_end + 10] += 0.02
    returns[170:172] -= 0.09
    hold = 100 * np.cumprod(1 + returns)
    average = pd.Series(hold).rolling(10, min_periods=10).mean().to_numpy()
    owned = np.zeros(months, dtype=bool)
    owned[1:] = hold[:-1] > np.nan_to_num(average[:-1], nan=np.inf)
    owned[:10] = True
    rule = 100 * np.cumprod(1 + np.where(owned, returns, 0.0))
    # The stretch in cash that contains the bottom of the crash, for the shaded band.
    first = last = crash_start + int(np.argmin(hold[crash_start:crash_end]))
    while first > 0 and not owned[first - 1]:
        first -= 1
    while last < months - 1 and not owned[last + 1]:
        last += 1
    return hold, rule, first, last


def hero_chart(theme: ChartTheme) -> str:
    """The landing illustration: a trend rule and simply holding, through a crash, drawing themselves in.

    Streamlit strips inline SVG from HTML, so the drawing is an SVG image with the theme's colors and its
    animation written inside it.
    """
    colors = PALETTE[theme.name]
    rule_color, hold_color, grid, muted, ink = (
        colors["rule"],
        colors["hold"],
        colors["border"],
        colors["muted"],
        colors["ink"],
    )
    hold, rule, cash_from, cash_to = _hero_series()
    left, right, top, bottom = 34.0, 430.0, 34.0, 214.0
    low, high = np.log(min(hold.min(), rule.min())), np.log(max(hold.max(), rule.max()))
    step = (right - left) / (len(hold) - 1)

    def x(month: float) -> float:
        return left + month * step

    def y(value: float) -> float:
        return bottom - (np.log(value) - low) / (high - low) * (bottom - top)

    def points(series: np.ndarray) -> str:
        return " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(series))

    ticks = range(0, HERO_YEARS + 1, 4)
    years = "".join(
        f'<text x="{x(index * 12):.1f}" y="234" text-anchor="middle">{HERO_START_YEAR + index}</text>'
        for index in ticks
    )
    marks = "".join(
        f'<line x1="{x(index * 12):.1f}" y1="{bottom}" x2="{x(index * 12):.1f}" y2="{bottom + 4}"/>'
        for index in ticks
    )
    levels = "".join(
        f'<line x1="{left}" y1="{level:.1f}" x2="{right}" y2="{level:.1f}"/>'
        for level in np.linspace(top, bottom, 4)
    )
    band_left, band_right = x(cash_from), x(cash_to + 1)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 440 246">
<style>
.l{{fill:none;stroke-width:2.2;stroke-linejoin:round;stroke-linecap:round;stroke-dasharray:1;stroke-dashoffset:1;
  animation:draw 2.4s cubic-bezier(.5,.05,.25,1) forwards}}
.r{{stroke:{rule_color};stroke-width:2.6;animation-delay:.25s}} .h{{stroke:{hold_color}}}
.band,.note{{opacity:0;animation:fade .7s ease 1.3s forwards}}
.end{{opacity:0;animation:fade .4s ease 2.5s forwards}}
text{{fill:{muted};font:500 10.5px ui-monospace,SFMono-Regular,Menlo,monospace}}
.note text{{fill:{ink};font:500 11px system-ui,-apple-system,sans-serif}}
.end text{{font:500 11px system-ui,-apple-system,sans-serif}}
@keyframes draw{{to{{stroke-dashoffset:0}}}} @keyframes fade{{to{{opacity:1}}}}
@media (prefers-reduced-motion:reduce){{.l,.band,.note,.end{{animation:none;stroke-dashoffset:0;opacity:1}}}}
</style>
<g stroke="{grid}" stroke-width="1">{levels}</g>
<rect class="band" x="{band_left:.1f}" y="{top - 14}" width="{band_right - band_left:.1f}"
  height="{bottom - top + 14}" fill="{ink}" fill-opacity=".06"/>
<g class="note"><line x1="{band_left:.1f}" y1="{top - 14}" x2="{band_left:.1f}" y2="{bottom}" stroke="{ink}"
  stroke-opacity=".35" stroke-dasharray="2 3"/>
<text x="{band_left + 6:.1f}" y="{top - 3}">The rule sits in cash</text></g>
<polyline class="l h" pathLength="1" points="{points(hold)}"/>
<polyline class="l r" pathLength="1" points="{points(rule)}"/>
<g class="end"><circle cx="{right}" cy="{y(rule[-1]):.1f}" r="3.5" fill="{rule_color}"/>
<circle cx="{right}" cy="{y(hold[-1]):.1f}" r="3.5" fill="{hold_color}"/></g>
<g stroke="{grid}">{marks}</g>{years}
<g class="end"><rect x="{left}" y="6" width="14" height="2.6" fill="{rule_color}"/>
<text x="{left + 20}" y="10" fill="{ink}">A trend rule</text>
<rect x="{left + 110}" y="6" width="14" height="2.6" fill="{hold_color}"/>
<text x="{left + 130}" y="10" fill="{ink}">Just holding</text></g>
</svg>"""
    encoded = base64.b64encode(svg.encode()).decode()
    return (
        f'<img class="ts-hero-chart" src="data:image/svg+xml;base64,{encoded}" '
        'alt="A trend rule and simply holding over nineteen years: the rule sits in cash through a crash">'
    )


def render_landing(theme: ChartTheme) -> None:
    """The first screen: what TradStrat is for, in plain words, and questions to start from."""
    seen: set[str] = set()
    st.html(
        f"""<div class="ts ts-hero">
  <div class="ts-hero-text">
    <div class="ts-kicker ts-rise">A backtesting lab for everyone</div>
    <h2 class="ts-hero-title ts-rise" style="--delay:.06s">Would this trading idea have made money?
      <em>And should you believe it?</em></h2>
    <p class="ts-hero-lead ts-rise" style="--delay:.14s">Many people trade with simple rules, like "buy when
      the price goes above its average". TradStrat tests a rule on years of real prices and compares it with
      simply buying and holding. Then it tells you, in plain words, whether the rule really worked or just
      got lucky.</p>
    <div class="ts-hero-meta ts-rise" style="--delay:.22s">Seven classic strategies · Any Yahoo Finance
      ticker · Or write your own rule · Free and open source</div>
  </div>
  <figure class="ts-hero-figure ts-rise" style="--delay:.1s">{hero_chart(theme)}
    <figcaption>Simulated prices. The rule sidesteps the crash, yet holding still ends ahead. Which
      matters more is what TradStrat helps you judge.</figcaption>
  </figure>
</div>"""
    )
    st.html(
        '<div class="ts ts-landing-head ts-rise" style="--delay:.25s"><h3 class="ts-display">'
        'Start with a question</h3><div class="ts-muted">Six classic ideas, each tested on real prices '
        "in a few seconds.</div></div>"
    )
    # Questions rise one after another. Their containers are Streamlit's, so each gets its place by key.
    order = "".join(f".st-key-question_{q.key}{{--order:{n}}}" for n, q in enumerate(QUESTIONS))
    st.html(f"<style>{order}</style>")
    for number, question in enumerate(QUESTIONS, start=1):
        if number % 2:
            columns = iter(st.columns(2, gap="large"))  # A row per pair keeps the entries level.
        with next(columns), st.container(key=f"question_{question.key}"):
            st.html(
                f"""<div class="ts ts-question">
  <span class="ts-question-number">{number:02d}</span>
  <div class="ts-question-body">
    <div class="ts-question-tag">{html.escape(question.tag)} · {html.escape(question.ticker)} ·
      {PERIOD_NAMES[question.history]}</div>
    <div class="ts-question-title">{html.escape(question.question)}</div>
    <div class="ts-question-detail">{rich(question.detail, seen)}</div>
  </div>
</div>"""
            )
            st.button(
                "Run this test  →",
                key=f"ask_{question.key}",
                on_click=ask,
                args=(question,),
                type="tertiary",
            )

    steps = [
        (
            "Pick a market and a rule",
            "Any ticker on Yahoo Finance, one of seven classic strategies, or a rule you write yourself.",
        ),
        (
            "Replay the past",
            "Every trade fills at the next day's price, after commission and slippage, as it would have "
            "for real.",
        ),
        (
            "Get a straight answer",
            "Did the rule beat just holding? Could it be luck? Would it have held up on data it was never "
            "tuned on?",
        ),
    ]
    items = "".join(
        f'<div class="ts-step"><span class="ts-step-number">{number}</span>'
        f"<div><strong>{title}</strong><p>{rich(text, seen)}</p></div></div>"
        for number, (title, text) in enumerate(steps, start=1)
    )
    st.html(
        f"""<div class="ts ts-how">
  <div class="ts-why">
    <h3 class="ts-display">Why this exists</h3>
    <p>Most trading ideas look great on a chart and then lose real money. The usual reasons are hidden
    costs, hindsight and plain luck. TradStrat checks for all three, for free, before you risk anything.</p>
  </div>
  <div class="ts-steps">{items}</div>
  <div class="ts-muted ts-small">Words with a dotted underline, like {rich("Sharpe ratio")}, explain
  themselves when you hover over or tap them.</div>
</div>"""
    )
    st.button(
        "Or set up your own test in the sidebar",
        key="start_own",
        on_click=start,
        type="tertiary",
        icon=":material/tune:",
    )


def signal_html(prices: pd.DataFrame, inputs: Inputs) -> str:
    """What the rule says at the latest close, and since when."""
    signal = latest_signal(create_strategy(inputs.strategy_class, **inputs.params), prices)
    tone, mark = ("good", "●") if signal.long else ("neutral", "○")
    stance = f"Own {html.escape(inputs.label)}" if signal.long else "Hold cash"
    since = f"since {signal.since:%d %b %Y}" if signal.since is not None else "for the whole history"
    tip = (
        "What the rule said at the last close in the data. Following it means trading at the next open. "
        "This describes the rule, and is not a recommendation."
    )
    return f"""<div class="ts-signal ts-tone-{tone}">
  <div class="ts-eyebrow">
    <span class="term" tabindex="0" data-tip="{html.escape(tip)}">Latest signal</span>
  </div>
  <div class="ts-signal-stance"><span class="ts-signal-mark">{mark}</span>{stance}</div>
  <div class="ts-meta" style="margin-top:2px">{since}, as of {signal.as_of:%d %b %Y}</div>
</div>"""


def render_context(prices: pd.DataFrame, inputs: Inputs, warning: str | None) -> None:
    """The rule being tested, in one sentence, what it is compared with, and what it says today."""
    seen: set[str] = set()
    config = inputs.config
    assert config is not None
    rule = guide_for(inputs.strategy_class).describe_rule(inputs.label, inputs.params)
    benchmark = (
        f"Compared with **buy and hold**: buying {inputs.label} on the first day and keeping it, with the "
        "same costs. A rule is only worth following if it beats that."
    )
    cash = {
        "Treasury bills": "idle cash earns Treasury bill yields",
        "Fixed rate": f"idle cash earns {inputs.fixed_rate:.2%} a year",
        "None": "idle cash earns nothing",
    }[inputs.cash]
    if warning:
        cash = "idle cash earns nothing"
    details = [
        f"{prices.index[0]:%d %b %Y} to {prices.index[-1]:%d %b %Y}",
        f"${config.initial_capital:,.0f} to start",
        f"{config.commission_rate:.2%} commission and {config.slippage_bps / 10_000:.2%} slippage per trade",
        cash,
    ]
    if config.stop_loss:
        details.append(f"{'trailing' if config.trailing_stop else 'fixed'} stop {config.stop_loss:.0%} below")
    meta = " · ".join(rich(detail, seen) for detail in details)
    st.html(
        f"""<div class="ts ts-card ts-context">
  <div class="ts-context-main">
    <div class="ts-eyebrow">The rule being tested</div>
    <div class="ts-rule">{rich(rule, seen)}</div>
    <div class="ts-benchmark">{rich(benchmark, seen)}</div>
    <div class="ts-meta">{meta}</div>
  </div>
  {signal_html(prices, inputs)}
</div>"""
    )
    if warning:
        st.warning(warning)


# Tabs.


def _points(value: float, benchmark: float, percent: bool = True) -> str | None:
    """The gap to buy and hold, in percentage points for rates and plain units for ratios."""
    if math.isnan(value) or math.isnan(benchmark):
        return None
    difference = (value - benchmark) * (100 if percent else 1)
    digits = 1 if percent else 2
    if round(difference, digits) == 0:
        return f"{0:.{digits}f}{' pts' if percent else ''}"
    return f"{difference:+.{digits}f}{' pts' if percent else ''}"


def _thin(series: pd.Series) -> list[float]:
    """Thin a daily series for a sparkline, which only needs its shape."""
    return series.dropna().iloc[:: max(1, len(series) // 120)].round(4).tolist()


def _metric(column: Any, label: str, value: str, delta: str | None = None, **kwargs: Any) -> None:
    """A bordered metric whose gap to buy and hold is gray when there is no gap."""
    flat = delta is not None and not delta.startswith(("+", "-"))
    column.metric(
        label,
        value,
        delta,
        delta_description="vs buy & hold" if delta else None,
        delta_color="off" if flat else "normal",
        border=True,
        **kwargs,
    )


def render_kpis(result: BacktestResult) -> None:
    strategy, benchmark = result.metrics, result.benchmark_metrics
    results = result.results
    years = calendar_year_returns(results["Strategy Equity"])
    tiles = [
        (
            "Total return",
            format_metric("total_return", strategy.total_return),
            _points(strategy.total_return, benchmark.total_return),
            _thin(results["Strategy Equity"]),
            "area",
            GLOSSARY["Total Return"],
        ),
        (
            "Annual return",
            format_metric("cagr", strategy.cagr),
            _points(strategy.cagr, benchmark.cagr),
            years.round(4).tolist(),
            "bar",
            GLOSSARY["CAGR"],
        ),
        (
            "Sharpe ratio",
            format_metric("sharpe_ratio", strategy.sharpe_ratio),
            _points(strategy.sharpe_ratio, benchmark.sharpe_ratio, percent=False),
            _thin(rolling_sharpe(result.excess_returns)),
            "line",
            GLOSSARY["Sharpe Ratio"],
        ),
        (
            "Max drawdown",
            format_metric("max_drawdown", strategy.max_drawdown),
            _points(strategy.max_drawdown, benchmark.max_drawdown),
            _thin(results["Strategy Drawdown"]),
            "area",
            GLOSSARY["Max Drawdown"],
        ),
    ]
    for column, (label, value, delta, sparkline, kind, explanation) in zip(st.columns(4), tiles, strict=True):
        _metric(column, label, value, delta, chart_data=sparkline or None, chart_type=kind, help=explanation)


def _bar(value: float, scale: float) -> float:
    """Bar length as a percentage of the longer bar, for a value compared on its size."""
    if math.isnan(value) or scale <= 0:
        return 0.0
    return max(2.0, min(100.0, abs(value) / scale * 100))


def round_html(contest: Round, seen: set[str]) -> str:
    """One round of the head-to-head: two bars, the values, and who won."""
    values = (contest.rule, contest.hold)
    defined = [abs(value) for value in values if not math.isnan(value)]
    scale = max(defined) if defined else 0.0
    if contest.metric != "max_drawdown":
        # Growth and Sharpe are compared on the upside; a loss gets a stub rather than a long bar.
        scale = max([value for value in values if not math.isnan(value) and value > 0], default=0.0)
    rows = []
    for side, value, name in (("rule", contest.rule, "Your rule"), ("hold", contest.hold, "Just holding")):
        length = _bar(value, scale) if contest.metric == "max_drawdown" or value > 0 else 2.0
        won = contest.winner == side
        rows.append(
            f'<div class="ts-bar-row{" ts-won" if won else ""}">'
            f'<span class="ts-who">{name}</span>'
            f'<span class="ts-bar"><i class="ts-fill-{side}" style="--w:{length:.1f}%"></i></span>'
            f'<span class="ts-value">{html.escape(format_metric(contest.metric, value))}</span></div>'
        )
    hint = "Smaller is better." if contest.metric == "max_drawdown" else "Bigger is better."
    winner = {"rule": "Your rule wins", "hold": "Holding wins", None: "No winner"}[contest.winner]
    tip = html.escape(GLOSSARY[contest.term])
    title = f'<span class="term" tabindex="0" data-tip="{tip}">{contest.title}</span>'
    seen.add(contest.term)
    return f"""<div class="ts-round">
  <div class="ts-round-title">
    {title}<span class="ts-round-winner ts-win-{contest.winner}">{winner}</span>
  </div>
  {"".join(rows)}
  <div class="ts-hint">{hint}</div>
</div>"""


def luck_html(probability: float) -> str:
    """How often the rule beat holding across reshuffled histories, against the 95% bar."""
    if math.isnan(probability):
        return ""
    tip = html.escape(GLOSSARY["Block Bootstrap"])
    passed = probability >= 0.95
    status = "Passes" if passed else "Not yet"
    return f"""<div class="ts-round ts-luck">
  <div class="ts-round-title"><span class="term" tabindex="0" data-tip="{tip}">Could it be luck?</span>
    <span class="ts-round-winner ts-win-{"rule" if passed else "None"}">{status}</span></div>
  <div class="ts-meter"><i style="--w:{probability * 100:.1f}%"></i><b style="left:95%"></b></div>
  <div class="ts-luck-scale"><span>0%</span><span class="ts-luck-mark">95% needed</span></div>
  <div class="ts-hint">Beat holding in <strong>{probability:.0%}</strong> of 2,000 reshuffled histories.</div>
</div>"""


ANSWER_CHIPS: Final[dict[str, tuple[str, str]]] = {
    "good": ("rule", "Your rule won"),
    "warning": ("mixed", "Mixed result"),
    "bad": ("hold", "Holding won"),
    "neutral": ("mixed", "No result"),
}


def render_answer(result: BacktestResult, probability: float, asset: str) -> None:
    """The verdict as a head-to-head between the rule and holding, in calm colors, with what it means."""
    verdict = assess(result.metrics, result.benchmark_metrics, probability, asset)
    contests = rounds(result.metrics, result.benchmark_metrics)
    side, label = ANSWER_CHIPS[verdict.tone]
    if verdict.tone == "warning" and all(contest.winner == "rule" for contest in contests[:2]):
        side = "rule"
    seen: set[str] = set()
    metrics = result.metrics
    activity = (
        f"Invested {metrics.exposure:.0%} of the time, across {metrics.number_of_trades} "
        f"trade{'s' if metrics.number_of_trades != 1 else ''}."
    )
    cards = "".join(round_html(contest, seen) for contest in contests) + luck_html(probability)
    st.html(
        f"""<div class="ts ts-answer ts-rise">
  <div class="ts-answer-head">
    <div class="ts-eyebrow">Did the rule beat just holding {html.escape(asset)}?</div>
    <div class="ts-answer-title">
      <span class="ts-pill ts-pill-{side}">{label}</span>{html.escape(verdict.headline)}
    </div>
    <p class="ts-takeaway">{rich(verdict.takeaway, seen)} <span class="ts-muted">{activity}</span></p>
  </div>
  <div class="ts-rounds">{cards}</div>
</div>"""
    )


def render_results(
    result: BacktestResult, estimate: BootstrapResult | None, theme: ChartTheme, asset: str
) -> None:
    probability = estimate.probability_beats_benchmark if estimate else float("nan")
    render_answer(result, probability, asset)

    results = result.results
    start = result.config.initial_capital
    rule_end = float(results["Strategy Equity"].iloc[-1])
    hold_end = float(results["Buy and Hold Benchmark"].iloc[-1])
    st.html(
        f"""<div class="ts ts-money ts-rise" style="--delay:.12s">
  <span class="ts-money-start">${start:,.0f}</span> became
  <span class="ts-money-rule">${rule_end:,.0f}</span> following the rule, and
  <span class="ts-money-hold">${hold_end:,.0f}</span> just holding {html.escape(asset)}.
</div>"""
    )
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

    with st.expander("How sure can we be? The luck test in detail"):
        render_luck(result, estimate, theme)

    with st.expander("Every number, and more charts"):
        render_kpis(result)
        left, right = st.columns([3, 2], gap="large")
        with left:
            st.dataframe(
                comparison_table(result.metrics, result.benchmark_metrics),
                height="content",
                column_config={"_index": st.column_config.Column(width="medium")},
            )
        with right:
            relative = result.relative_metrics
            grid = st.columns(2)
            grid[0].metric(
                "Alpha (annual)", format_metric("cagr", relative.alpha), help=GLOSSARY["Alpha"], border=True
            )
            grid[1].metric("Beta", f"{relative.beta:.2f}", help=GLOSSARY["Beta"], border=True)
            grid = st.columns(2)
            grid[0].metric(
                "Correlation",
                f"{relative.correlation:.2f}",
                help="How closely daily returns move with buy and hold, from -1 to 1.",
                border=True,
            )
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
        left, right = st.columns(2)
        with left:
            chart(rolling_sharpe_figure(result.excess_returns, result.benchmark_excess_returns, theme))
        with right:
            chart(monthly_returns_figure(results["Strategy Equity"], theme))


def render_luck(result: BacktestResult, estimate: BootstrapResult | None, theme: ChartTheme) -> None:
    prose(
        "A backtest is one version of history. TradStrat reshuffles it in 20-day blocks 2,000 times, a "
        "block bootstrap, to see how much the result could change by chance.",
        muted=True,
    )
    if estimate is None:
        st.caption("At least 40 trading days are needed to estimate this.")
        return
    returns = result.excess_returns
    columns = st.columns(4)
    columns[0].metric(
        "Chance it beats buy & hold",
        f"{estimate.probability_beats_benchmark:.0%}",
        help="Share of reshuffled histories in which the strategy's Sharpe ratio was higher than buy and "
        "hold's. A real edge should be 95% or more.",
        border=True,
    )
    columns[1].metric(
        "Chance it beats cash",
        f"{probabilistic_sharpe_ratio(returns):.0%}",
        help=GLOSSARY["Probabilistic Sharpe Ratio"],
        border=True,
    )
    columns[2].metric(
        "Likely Sharpe ratio",
        f"{estimate.sharpe.low:.2f} to {estimate.sharpe.high:.2f}",
        help="The 95% confidence interval. " + GLOSSARY["Confidence Interval"],
        border=True,
    )
    columns[3].metric(
        "Likely annual return",
        f"{estimate.annual_return.low:.0%} to {estimate.annual_return.high:.0%}",
        help="The 95% confidence interval for the annual return above cash.",
        border=True,
    )
    std = returns.std(ddof=1)
    observed = float(returns.mean() / std * np.sqrt(252)) if std > 0 else 0.0
    chart(
        bootstrap_figure(estimate.sharpe_samples, estimate.sharpe.low, estimate.sharpe.high, observed, theme)
    )


def render_trades(result: BacktestResult, prices: pd.DataFrame, inputs: Inputs, theme: ChartTheme) -> None:
    prose(
        "Every buy and sell the strategy made, drawn on the price together with the indicator lines it "
        "watches. Trades fill at the next day's open, after commission and slippage.",
        muted=True,
    )
    strategy = create_strategy(inputs.strategy_class, **inputs.params)
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
    columns[0].metric("Trades", metrics.number_of_trades, help=GLOSSARY["Number of Trades"], border=True)
    columns[1].metric(
        "Win rate", format_metric("win_rate", metrics.win_rate), help=GLOSSARY["Win Rate"], border=True
    )
    columns[2].metric(
        "Profit factor",
        format_metric("profit_factor", metrics.profit_factor),
        help=GLOSSARY["Profit Factor"],
        border=True,
    )
    columns[3].metric(
        "Average trade",
        format_metric("expectancy", metrics.expectancy),
        help=GLOSSARY["Expectancy"],
        border=True,
    )

    if trades.empty:
        st.info("No trades were made. Try a longer history or different settings.")
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


def _settings_text(strategy: StrategyRef, params: dict[str, Any]) -> str:
    """Settings as a reader would write them: a custom rule with its numbers filled in, or label and value."""
    cls = strategy_class(strategy)
    if issubclass(cls, RuleStrategy):
        rule = create_strategy(cls, **params)
        assert isinstance(rule, RuleStrategy)
        return rule.text
    labels = {spec.name: spec.label for spec in cls.params}
    return ", ".join(f"{labels[name]} {value}" for name, value in params.items())


def render_search(prices: pd.DataFrame, inputs: Inputs, theme: ChartTheme) -> None:
    section(
        "1. Find the best settings",
        "Tries every combination of this strategy's settings on the whole history and picks the winner. "
        "A winner picked in hindsight always looks good, so TradStrat also checks whether it is real or "
        "just overfitting.",
    )
    left, right, button = st.columns([2, 2, 1], vertical_alignment="bottom")
    points = left.slider(
        "Values to try per setting",
        3,
        10,
        6,
        help="More values give a finer search, and more chances to find a winner by luck.",
    )
    objective = right.selectbox(
        "Pick the winner by", OBJECTIVES, format_func=lambda name: METRIC_LABELS.get(name, name)
    )
    combinations = math.prod(len(spec.grid(points)) for spec in inputs.strategy_class.params)
    if combinations > MAX_COMBINATIONS:
        st.caption(
            f"That would test {combinations:,} combinations. Try fewer values per setting, or fewer "
            "numbers in the rule."
        )
    elif button.button("Run search", type="primary", width="stretch"):
        grid = build_grid(inputs.strategy_class, points)
        with st.spinner(f"Testing {len(grid)} combinations..."):
            st.session_state["search"] = (
                inputs.fingerprint(),
                grid_search(prices, inputs.strategy_class, grid, inputs.config, objective),
            )

    stored = st.session_state.get("search")
    if stored is None or stored[0] != inputs.fingerprint():
        st.caption("Run a search to see how much the result depends on the settings.")
        return
    search = stored[1]
    parameters = [spec.name for spec in inputs.strategy_class.params]
    overfitting = search.overfitting.probability if search.overfitting else float("nan")
    columns = st.columns(4)
    columns[0].metric("Combinations tried", len(search.table), border=True)
    columns[1].metric(
        "Best Sharpe, in hindsight",
        f"{search.table['sharpe_ratio'].max():.2f}",
        help=GLOSSARY["In-Sample"],
        border=True,
    )
    columns[2].metric(
        "Chance the edge is real",
        f"{search.deflated_active_sharpe:.0%}",
        help="The deflated Sharpe ratio of the best settings against buy and hold. "
        + GLOSSARY["Deflated Sharpe Ratio"],
        border=True,
    )
    columns[3].metric(
        "Chance it is overfit",
        "n/a" if math.isnan(overfitting) else f"{overfitting:.0%}",
        help=GLOSSARY["Probability of Backtest Overfitting"],
        border=True,
    )
    render_verdict(assess_search(search.deflated_active_sharpe, overfitting))
    prose(
        f"**Best settings:** {_settings_text(inputs.strategy_class, search.best_params)}. Against cash "
        f"rather than buy and hold, the chance the edge is real would be {search.deflated_sharpe:.0%}.",
        muted=True,
    )

    left, right = st.columns(2)
    with left:
        labels = {spec.name: spec.label for spec in inputs.strategy_class.params}
        chart(search_figure(search.table, parameters, search.objective, theme, labels))
    with right:
        if search.overfitting is not None:
            chart(overfitting_figure(search.overfitting.logits, theme))
    with st.expander("Every combination tried"):
        st.dataframe(search.table.rename(columns=METRIC_LABELS).round(3), hide_index=True)


def render_walk_forward(prices: pd.DataFrame, inputs: Inputs, theme: ChartTheme) -> None:
    section(
        "2. Test on unseen data",
        "The fair test, called walk-forward validation. Settings are chosen on a training window, then "
        "traded on the next window, which they have never seen. Everything rolls forward and repeats, so "
        "the result only uses decisions that could have been made at the time.",
    )
    columns = st.columns([2, 2, 1, 1], vertical_alignment="bottom")
    train = columns[0].select_slider("Choose settings using", list(TRAIN_WINDOWS), value="3 years")
    test = columns[1].select_slider("Then trade them for", list(TEST_WINDOWS), value="1 year")
    with columns[2].popover("Options", width="stretch"):
        points = st.slider("Values to try per setting", 3, 8, 5, key="walk_points")
        anchored = st.toggle(
            "Anchored",
            help="Keep every past year in the training window instead of sliding it forward.",
        )
    if columns[3].button("Run walk-forward", type="primary", width="stretch"):
        grid = build_grid(inputs.strategy_class, points)
        try:
            with st.spinner("Choosing settings and testing each window..."):
                st.session_state["walk"] = (
                    inputs.fingerprint(),
                    walk_forward(
                        prices,
                        inputs.strategy_class,
                        grid,
                        inputs.config,
                        train_bars=TRAIN_WINDOWS[train],
                        test_bars=TEST_WINDOWS[test],
                        anchored=anchored,
                    ),
                )
        except ValueError as error:
            st.session_state.pop("walk", None)
            st.error(str(error))

    stored = st.session_state.get("walk")
    if stored is None or stored[0] != inputs.fingerprint():
        st.caption("Run walk-forward to see how the strategy does on data it was not tuned on.")
        return
    result = stored[1]
    metrics, benchmark = result.metrics, result.benchmark_metrics
    columns = st.columns(4)
    _metric(
        columns[0],
        "Sharpe on unseen data",
        format_metric("sharpe_ratio", metrics.sharpe_ratio),
        _points(metrics.sharpe_ratio, benchmark.sharpe_ratio, percent=False),
        help=GLOSSARY["Sharpe Ratio"],
    )
    _metric(
        columns[1],
        "Annual return",
        format_metric("cagr", metrics.cagr),
        _points(metrics.cagr, benchmark.cagr),
        help=GLOSSARY["CAGR"],
    )
    _metric(
        columns[2],
        "Max drawdown",
        format_metric("max_drawdown", metrics.max_drawdown),
        _points(metrics.max_drawdown, benchmark.max_drawdown),
        help=GLOSSARY["Max Drawdown"],
    )
    columns[3].metric(
        "Kept of the hindsight score",
        "n/a" if math.isnan(result.efficiency) else f"{result.efficiency:.0%}",
        help=GLOSSARY["Walk-Forward Efficiency"],
        border=True,
    )
    chart(equity_figure(result.equity, result.benchmark, theme, "Growth on unseen data"))
    chart(fold_scores_figure(result.folds, result.objective, theme))
    with st.expander("Window by window"):
        folds = result.folds.assign(
            Parameters=[
                _settings_text(inputs.strategy_class, params) for params in result.folds["Parameters"]
            ]
        )
        st.dataframe(
            folds.round({"In-Sample Score": 3, "Out-of-Sample Score": 3, "Test Return": 4}),
            hide_index=True,
            column_config={
                name: st.column_config.DateColumn(format="YYYY-MM-DD")
                for name in ("Train Start", "Train End", "Test Start", "Test End")
            }
            | {"Test Return": st.column_config.NumberColumn(format="percent")},
        )


def render_tune(prices: pd.DataFrame, inputs: Inputs, theme: ChartTheme) -> None:
    render_search(prices, inputs, theme)
    st.divider()
    render_walk_forward(prices, inputs, theme)


def render_compare(prices: pd.DataFrame, inputs: Inputs, theme: ChartTheme) -> None:
    prose(
        "Same data, same costs and no tuning, so nothing here is flattered by hindsight. Strategies that "
        "sit in cash part of the time often fall less in a crash, and rise less in a boom.",
        muted=True,
    )
    rule = None
    if inputs.strategy_key == CUSTOM:
        rule = (st.session_state.get("rule_buy", ""), st.session_state.get("rule_sell", ""))
    comparison = cached_comparison(prices, inputs.config, rule)
    if comparison.table.empty:
        st.warning("This history is too short for any strategy's indicators.")
        return
    chart(comparison_figure(comparison.equity, theme))
    styles = {cls.label: guide_for(cls).style for cls in STRATEGIES.values()}
    styles[RuleStrategy.label] = "Your rule"
    columns = ["Strategy", "cagr", "sharpe_ratio", "max_drawdown", "exposure", "number_of_trades"]
    table = comparison.table[columns].rename(columns=METRIC_LABELS)
    table.insert(1, "Style", table["Strategy"].map(styles))
    st.dataframe(
        table,
        hide_index=True,
        column_config={
            "CAGR": st.column_config.NumberColumn(format="percent", help=GLOSSARY["CAGR"]),
            "Sharpe Ratio": st.column_config.NumberColumn(format="%.2f", help=GLOSSARY["Sharpe Ratio"]),
            "Max Drawdown": st.column_config.NumberColumn(format="percent", help=GLOSSARY["Max Drawdown"]),
            "Exposure": st.column_config.ProgressColumn(
                format="percent", min_value=0.0, max_value=1.0, help=GLOSSARY["Exposure"]
            ),
        },
    )


def render_guide() -> None:
    left, right = st.columns([3, 2], gap="large")
    with left:
        section("How TradStrat tests an idea")
        steps = [
            "**It trades the way you could have.** A signal formed at one day's close trades at the next "
            "day's open, after commission and slippage, in whole shares. Prices include dividends and idle "
            "cash earns Treasury bill yields.",
            "**It compares with doing nothing clever.** Buy and hold runs through the same simulator with "
            "the same costs. A strategy only counts if it beats that.",
            "**It asks whether the result was luck.** A block bootstrap reshuffles history thousands of "
            "times to show how much each number could have varied.",
            "**It corrects for searching.** Try enough settings and one will look good. The deflated "
            "Sharpe ratio and the probability of backtest overfitting discount for that.",
            "**It tests on unseen data.** Walk-forward validation only ever trades settings chosen on the "
            "past.",
        ]
        seen: set[str] = set()
        items = "".join(f"<li>{rich(step, seen)}</li>" for step in steps)
        st.html(f'<div class="ts"><ol>{items}</ol></div>')

        section("The strategies", "Each one is a simple, well-known rule. Open one to see how it decides.")
        for key in [*STRATEGY_KEYS, "buy_and_hold"]:
            with st.expander(f"{STRATEGIES[key].label}  ·  {guide_for(STRATEGIES[key]).style}"):
                st.html(guide_html(key))

        section(
            "Write your own rule",
            "Choose **Write your own rule** in the sidebar to test an idea of your own, with the same "
            "checks.",
        )
        with st.expander("What a rule can use"):
            st.html(rules_html())
    with right:
        section("Glossary")
        query = st.text_input(
            "Search terms", placeholder="Search, e.g. drawdown", label_visibility="collapsed"
        )
        query = query.strip().lower()
        matches = [
            (term, definition)
            for term, definition in sorted(GLOSSARY.items())
            if query in term.lower() or query in definition.lower()
        ]
        if not matches:
            st.caption("No terms match.")
        entries = "".join(
            f"<p><strong>{html.escape(term)}.</strong> "
            f'<span class="ts-muted">{html.escape(definition)}</span></p>'
            for term, definition in matches
        )
        st.html(f'<div class="ts ts-small">{entries}</div>')


def render_footer() -> None:
    st.html(
        f"""<div class="ts ts-footer">TradStrat {__version__} · <a href="{REPOSITORY}" target="_blank"
rel="noopener">Source on GitHub</a> · MIT license · For research and education, not investment advice.
Past performance does not predict future results.</div>"""
    )


def report_builder(
    result: BacktestResult, prices: pd.DataFrame, inputs: Inputs, estimate: BootstrapResult | None
):
    """Build the HTML report only when someone clicks download."""

    def build() -> str:
        cls = inputs.strategy_class
        strategy = create_strategy(cls, **inputs.params)
        context = ReportContext(
            strategy=cls.label if inputs.strategy_key != CUSTOM else repr(strategy),
            asset=inputs.label,
            parameters={spec.name: getattr(strategy, spec.name) for spec in cls.params},
            overlays=indicator_overlays(strategy, prices),
        )
        return build_report(result, context, estimate=estimate)

    return build


def main() -> None:
    st.set_page_config(page_title="TradStrat", page_icon=":material/monitoring:", layout="wide")
    theme = theme_for(st.context.theme.type == "dark")
    inject_styles(theme)
    restore_from_link()
    inputs = read_sidebar()
    started = has_started(inputs)

    header, ideas, actions = st.columns([5, 1.1, 1.4], vertical_alignment="center")
    with header:
        render_header(theme, tagline=started)
    if not started:
        render_landing(theme)
        render_footer()
        return
    write_link(inputs, st.session_state.get("_history_for_link"))
    ideas.button(
        "All questions", on_click=show_ideas, type="tertiary", icon=":material/arrow_back:", width="stretch"
    )

    if inputs.config is None:
        st.error(inputs.config_error)
        return
    if inputs.strategy is None:
        st.error(f"Your rule could not be read. {literal(inputs.rule_error or '')}")
        st.html(f'<div class="ts-card">{rules_html()}</div>')
        return
    try:
        strategy = create_strategy(inputs.strategy_class, **inputs.params)
    except ValueError as error:
        st.error(f"These settings don't work together: {error}")
        return
    try:
        prices, warning = load_prices(inputs)
    except DataError as error:
        render_data_problem(inputs, error)
        return
    if len(prices) <= strategy.warmup + 1:
        render_problem(
            "Not enough history for this rule yet",
            f"{inputs.strategy_class.label} needs more than {strategy.warmup} trading days before it can "
            f"make its first decision, and only {len(prices)} are available. Choose a longer history in the "
            "sidebar, or shorter windows.",
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
            icon=":material/download:",
            on_click="ignore",
            width="stretch",
            help="A single HTML file with the verdict, charts and every trade, to keep or share.",
        )
    render_context(prices, inputs, warning)

    results, trades, tune, compare, guide = st.tabs(TABS)
    with results:
        render_results(result, estimate, theme, inputs.label)
    with trades:
        render_trades(result, prices, inputs, theme)
    with tune:
        render_tune(prices, inputs, theme)
    with compare:
        render_compare(prices, inputs, theme)
    with guide:
        render_guide()
    render_footer()


if __name__ == "__main__":
    main()
