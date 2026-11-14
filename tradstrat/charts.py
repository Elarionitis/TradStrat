"""Plotly figures shared by the dashboard and the HTML report.

Every figure takes a ``ChartTheme``, so the same chart renders correctly on a light or a dark page. Colors
are assigned by role, never by rank, and the palettes were checked with a color-vision-deficiency
validator in both modes. Every multi-series chart has a legend, the key lines are also labeled directly,
and buy and sell markers differ in shape, so identity never rests on color alone. Heatmaps of signed
values use a diverging blue and red scale with a neutral gray midpoint at zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from tradstrat.metrics import calendar_year_returns, rolling_sharpe
from tradstrat.rules import RuleStrategy
from tradstrat.strategies import STRATEGIES, BuyAndHoldStrategy
from tradstrat.stress import Crash

# The two lines on every comparison chart, named as the dashboard's answer names them.
RULE_NAME: Final[str] = "Your rule"
HOLD_NAME: Final[str] = "Just holding"
FONT_FAMILY: Final[str] = "'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif"
MONTHS: Final[list[str]] = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]


@dataclass(frozen=True)
class ChartTheme:
    """The colors a figure needs, chosen separately for light and dark surfaces."""

    name: str
    strategy: str
    benchmark: str
    buy: str
    sell: str
    positive: str
    negative: str
    neutral: str
    ink: str
    secondary: str
    muted: str
    grid: str
    axis: str
    hover_background: str
    # Strategy identity colors in registry order. The benchmark always keeps ``benchmark``.
    series: tuple[str, ...]


LIGHT: Final[ChartTheme] = ChartTheme(
    name="light",
    strategy="#2a78d6",
    benchmark="#eb6834",
    buy="#1baf7a",
    sell="#eb6834",
    positive="#2a78d6",
    negative="#e34948",
    neutral="#ece8de",
    ink="#16140f",
    secondary="#615d53",
    muted="#898781",
    grid="#e4dfd3",
    axis="#c9c2b2",
    hover_background="#fbf9f4",
    series=("#2a78d6", "#1baf7a", "#eda100", "#e87ba4", "#6250d6", "#008300", "#e34948"),
)
DARK: Final[ChartTheme] = ChartTheme(
    name="dark",
    strategy="#3987e5",
    benchmark="#d95926",
    buy="#199e70",
    sell="#d95926",
    positive="#3987e5",
    negative="#e66767",
    neutral="#3a362f",
    ink="#ece6d8",
    secondary="#c4bdae",
    muted="#898781",
    grid="#2a2722",
    axis="#3a362f",
    hover_background="#1f1d19",
    series=("#3987e5", "#199e70", "#c98500", "#d55181", "#9085e9", "#008300", "#e66767"),
)


def theme_for(dark: bool) -> ChartTheme:
    return DARK if dark else LIGHT


def strategy_color(label: str, theme: ChartTheme) -> str:
    """A fixed color per strategy, so a filter that hides one never repaints the others."""
    if label == BuyAndHoldStrategy.label:
        return theme.benchmark
    if label == RuleStrategy.label:
        return theme.ink  # The reader's own rule stands apart from every built-in one.
    labels = [cls.label for key, cls in STRATEGIES.items() if key != BuyAndHoldStrategy.key]
    position = labels.index(label) if label in labels else len(labels)
    return theme.series[position % len(theme.series)]


def _style(
    figure: go.Figure, theme: ChartTheme, title: str, y_title: str = "", x_title: str = "", height: int = 380
) -> go.Figure:
    """Apply the shared layout: recessive grid, one axis, legend above the plot."""
    figure.update_layout(
        template="none",
        height=height,
        title={"text": title, "x": 0, "xanchor": "left", "font": {"size": 15, "color": theme.ink}},
        margin={"l": 8, "r": 16, "t": 64, "b": 8},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.0,
            "x": 0,
            "font": {"color": theme.secondary},
            "bgcolor": "rgba(0,0,0,0)",
        },
        font={"family": FONT_FAMILY, "color": theme.secondary, "size": 12},
        hoverlabel={
            "bgcolor": theme.hover_background,
            "bordercolor": theme.axis,
            "font": {"family": FONT_FAMILY, "color": theme.ink},
        },
    )
    figure.update_xaxes(
        title=x_title,
        showgrid=False,
        linecolor=theme.axis,
        tickcolor=theme.axis,
        tickfont={"color": theme.muted},
        zeroline=False,
        automargin=True,
    )
    figure.update_yaxes(
        title=y_title,
        gridcolor=theme.grid,
        zerolinecolor=theme.axis,
        tickfont={"color": theme.muted},
        title_font={"color": theme.muted},
        automargin=True,
    )
    return figure


def _crosshair(figure: go.Figure, theme: ChartTheme) -> None:
    """A vertical guide and a single tooltip for every series at the hovered date."""
    figure.update_layout(hovermode="x unified")
    figure.update_xaxes(
        showspikes=True, spikemode="across", spikethickness=1, spikecolor=theme.muted, spikedash="solid"
    )


def _line(
    series: pd.Series, name: str, color: str, *, width: float = 2, dash: str = "solid", **kwargs: Any
) -> go.Scatter:
    return go.Scatter(
        x=series.index,
        y=series,
        name=name,
        mode="lines",
        line={"width": width, "color": color, "dash": dash},
        **kwargs,
    )


def _label_line_end(figure: go.Figure, series: pd.Series, text: str, theme: ChartTheme) -> None:
    """Name a line at its last point, so a reader need not match colors against the legend."""
    clean = series.dropna()
    if clean.empty:
        return
    figure.add_annotation(
        x=clean.index[-1],
        y=float(clean.iloc[-1]),
        text=text,
        showarrow=False,
        xanchor="left",
        xshift=6,
        font={"color": theme.secondary, "size": 12},
    )


def equity_figure(
    strategy: pd.Series,
    benchmark: pd.Series,
    theme: ChartTheme = LIGHT,
    title: str = "Growth of the portfolio",
    *,
    log_scale: bool = False,
) -> go.Figure:
    figure = go.Figure(
        [
            _line(benchmark, HOLD_NAME, theme.benchmark),
            _line(strategy, RULE_NAME, theme.strategy),
        ]
    )
    _label_line_end(figure, strategy, RULE_NAME, theme)
    _label_line_end(figure, benchmark, HOLD_NAME, theme)
    _crosshair(figure, theme)
    figure.update_traces(hovertemplate="$%{y:,.0f}")
    figure.update_yaxes(tickprefix="$", separatethousands=True, type="log" if log_scale else "linear")
    _style(figure, theme, title, "Portfolio value", height=420)
    figure.update_layout(margin={"r": 100})
    _fit_dates(figure, strategy.index)
    return figure


def _fit_dates(figure: go.Figure, index: pd.Index) -> None:
    """Pin the date axis to the data, so end labels and markers do not pad it with empty years."""
    if len(index):
        figure.update_xaxes(range=[index[0], index[-1]])


def drawdown_figure(strategy: pd.Series, benchmark: pd.Series, theme: ChartTheme = LIGHT) -> go.Figure:
    figure = go.Figure(
        [
            _line(benchmark, HOLD_NAME, theme.benchmark),
            _line(
                strategy, RULE_NAME, theme.strategy, fill="tozeroy", fillcolor=_alpha(theme.strategy, 0.14)
            ),
        ]
    )
    _crosshair(figure, theme)
    figure.update_traces(hovertemplate="%{y:.1%}")
    figure.update_yaxes(tickformat=".0%")
    return _style(figure, theme, "Drawdown from peak", "Decline")


def _alpha(color: str, opacity: float) -> str:
    red, green, blue = (int(color[index : index + 2], 16) for index in (1, 3, 5))
    return f"rgba({red},{green},{blue},{opacity})"


def price_figure(
    close: pd.Series,
    trades: pd.DataFrame,
    theme: ChartTheme = LIGHT,
    overlays: pd.DataFrame | None = None,
) -> go.Figure:
    """Price with the strategy's own indicator lines and every entry and exit."""
    figure = go.Figure([_line(close, "Close", theme.strategy, width=1.5)])
    if overlays is not None:
        # Indicator lines share one recessive color and differ by dash, so the trades stay prominent.
        for column, dash in zip(overlays.columns, ("dot", "dash", "dashdot"), strict=False):
            figure.add_trace(
                _line(
                    overlays[column],
                    str(column),
                    theme.secondary,
                    width=1.25,
                    dash=dash,
                    hovertemplate="%{y:.2f}",
                )
            )
    if not trades.empty:
        for name, date, price, symbol, color in (
            ("Buy", "Entry Date", "Entry Price", "triangle-up", theme.buy),
            ("Sell", "Exit Date", "Exit Price", "triangle-down", theme.sell),
        ):
            figure.add_trace(
                go.Scatter(
                    x=trades[date],
                    y=trades[price],
                    name=name,
                    mode="markers",
                    hovertemplate=f"{name} at %{{y:.2f}}<extra></extra>",
                    marker={
                        "symbol": symbol,
                        "size": 11,
                        "color": color,
                        "line": {"width": 1.5, "color": theme.hover_background},
                    },
                )
            )
    _crosshair(figure, theme)
    figure.update_layout(hovermode="closest")
    _fit_dates(figure, close.index)
    return _style(figure, theme, "Price, indicators and trades", "Price (dividend-adjusted)", height=440)


def monthly_returns_table(equity: pd.Series) -> pd.DataFrame:
    """Compound an equity curve into a table of returns with years as rows and months as columns."""
    month_ends = equity.resample("ME").last()
    returns = month_ends.pct_change()
    returns.iloc[0] = month_ends.iloc[0] / equity.iloc[0] - 1
    dates = pd.DatetimeIndex(returns.index)
    frame = pd.DataFrame({"year": dates.year, "month": dates.month, "value": returns.to_numpy()})
    return frame.pivot_table(index="year", columns="month", values="value").reindex(columns=range(1, 13))


def _diverging(theme: ChartTheme) -> list[list[Any]]:
    return [[0.0, theme.negative], [0.5, theme.neutral], [1.0, theme.positive]]


def monthly_returns_figure(equity: pd.Series, theme: ChartTheme = LIGHT) -> go.Figure:
    table = monthly_returns_table(equity)
    limit = float(np.nanmax(np.abs(table.to_numpy()))) or 1.0
    figure = go.Figure(
        go.Heatmap(
            z=table.to_numpy(),
            x=MONTHS,
            y=[str(year) for year in table.index],
            colorscale=_diverging(theme),
            zmin=-limit,
            zmax=limit,
            xgap=2,
            ygap=2,
            colorbar={
                "tickformat": ".0%",
                "title": "",
                "outlinewidth": 0,
                "tickfont": {"color": theme.muted},
            },
            hovertemplate="%{x} %{y}: %{z:.2%}<extra></extra>",
        )
    )
    figure.update_yaxes(autorange="reversed", type="category")
    _style(figure, theme, "Monthly returns")
    figure.update_yaxes(showgrid=False)
    return figure


def annual_returns_figure(strategy: pd.Series, benchmark: pd.Series, theme: ChartTheme = LIGHT) -> go.Figure:
    """Calendar-year returns side by side, which shows which kinds of year a strategy helps in."""
    strategy_years = calendar_year_returns(strategy)
    benchmark_years = calendar_year_returns(benchmark)
    years = [str(year) for year in strategy_years.index]
    figure = go.Figure(
        [
            go.Bar(x=years, y=strategy_years, name=RULE_NAME, marker={"color": theme.strategy}),
            go.Bar(x=years, y=benchmark_years, name=HOLD_NAME, marker={"color": theme.benchmark}),
        ]
    )
    figure.update_traces(hovertemplate="%{y:.1%}", marker_cornerradius=4)
    figure.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08, hovermode="x unified")
    figure.update_yaxes(tickformat=".0%")
    figure.update_xaxes(type="category")
    return _style(figure, theme, "Calendar-year returns", "Return")


def rolling_sharpe_figure(
    strategy_returns: pd.Series, benchmark_returns: pd.Series, theme: ChartTheme = LIGHT, window: int = 252
) -> go.Figure:
    figure = go.Figure(
        [
            _line(rolling_sharpe(benchmark_returns, window), HOLD_NAME, theme.benchmark),
            _line(rolling_sharpe(strategy_returns, window), RULE_NAME, theme.strategy),
        ]
    )
    figure.add_hline(y=0, line={"color": theme.axis, "width": 1})
    _crosshair(figure, theme)
    figure.update_traces(hovertemplate="%{y:.2f}")
    title = (
        "Sharpe ratio over the past year" if window == 252 else f"Sharpe ratio over the past {window} days"
    )
    return _style(figure, theme, title, "Sharpe ratio")


def bootstrap_figure(
    samples: np.ndarray, low: float, high: float, observed: float, theme: ChartTheme = LIGHT
) -> go.Figure:
    figure = go.Figure(
        go.Histogram(
            x=samples[~np.isnan(samples)],
            nbinsx=50,
            marker={"color": theme.strategy, "line": {"width": 1, "color": theme.hover_background}},
            name="Resamples",
            hovertemplate="Sharpe %{x}<br>%{y} histories<extra></extra>",
        )
    )
    figure.add_vline(x=0, line={"color": theme.muted, "width": 1})
    for value, label in ((low, "2.5%"), (high, "97.5%")):
        figure.add_vline(
            x=value,
            line={"color": theme.secondary, "width": 1.5, "dash": "dash"},
            annotation_text=label,
            annotation_font_color=theme.secondary,
        )
    figure.add_vline(
        x=observed,
        line={"color": theme.benchmark, "width": 2},
        annotation_text="Actual",
        annotation_position="top left",
        annotation_font_color=theme.secondary,
    )
    return _style(
        figure, theme, "Sharpe ratio across 2,000 reshuffled histories", "Histories", "Sharpe ratio"
    )


def search_figure(
    table: pd.DataFrame,
    parameters: list[str],
    objective: str,
    theme: ChartTheme = LIGHT,
    labels: dict[str, str] | None = None,
) -> go.Figure:
    """Show how the objective varies across parameters: a line for one, a heatmap for two or more.

    A broad plateau suggests a robust idea, and an isolated peak suggests a fitted one. With three or more
    parameters each cell shows the best score over the parameters not on the axes. ``labels`` gives display
    names for the parameters.
    """
    label = objective.replace("_", " ").title()
    names = {name: (labels or {}).get(name, name) for name in parameters}
    if len(parameters) == 1:
        ordered = table.sort_values(parameters[0])
        figure = go.Figure(
            go.Scatter(
                x=ordered[parameters[0]],
                y=ordered[objective],
                mode="lines+markers",
                line={"width": 2, "color": theme.strategy},
                marker={
                    "size": 8,
                    "color": theme.strategy,
                    "line": {"width": 2, "color": theme.hover_background},
                },
                hovertemplate=f"{names[parameters[0]]} %{{x}}<br>{label} %{{y:.2f}}<extra></extra>",
            )
        )
        return _style(figure, theme, f"{label} for each setting", label, names[parameters[0]])

    x_name, y_name = parameters[:2]
    grid = table.pivot_table(index=y_name, columns=x_name, values=objective, aggfunc="max")
    limit = float(np.nanmax(np.abs(grid.to_numpy()))) if grid.notna().any().any() else 1.0
    figure = go.Figure(
        go.Heatmap(
            z=grid.to_numpy(),
            x=[str(value) for value in grid.columns],
            y=[str(value) for value in grid.index],
            colorscale=_diverging(theme),
            zmin=-limit,
            zmax=limit,
            xgap=2,
            ygap=2,
            colorbar={"outlinewidth": 0, "tickfont": {"color": theme.muted}},
            hovertemplate=(
                f"{names[x_name]} %{{x}}<br>{names[y_name]} %{{y}}<br>{label} %{{z:.2f}}<extra></extra>"
            ),
        )
    )
    _style(figure, theme, f"{label} for every combination tried", names[y_name], names[x_name])
    figure.update_yaxes(showgrid=False, type="category")
    figure.update_xaxes(type="category")
    return figure


def overfitting_figure(logits: np.ndarray, theme: ChartTheme = LIGHT) -> go.Figure:
    """Where the in-sample winner ranked out of sample, across every split of the history.

    Mass left of zero means the winner fell to the bottom half on unseen data: the search was fitting noise.
    """
    finite = logits[np.isfinite(logits)]
    figure = go.Figure(
        go.Histogram(
            x=finite,
            nbinsx=40,
            marker={"color": theme.strategy, "line": {"width": 1, "color": theme.hover_background}},
            hovertemplate="Rank score %{x:.2f}<br>%{y} splits<extra></extra>",
            name="Splits",
        )
    )
    figure.add_vline(
        x=0,
        line={"color": theme.negative, "width": 2},
        annotation_text="Middle of the pack",
        annotation_position="top right",
        annotation_font_color=theme.secondary,
    )
    return _style(
        figure,
        theme,
        "How the hindsight winner ranked on unseen data",
        "Splits of the history",
        "Rank on unseen data (left of zero means the bottom half)",
    )


def fold_scores_figure(folds: pd.DataFrame, objective: str, theme: ChartTheme = LIGHT) -> go.Figure:
    """Compare what each fold's parameters promised in training with what they delivered afterwards."""
    label = objective.replace("_", " ").title()
    names = [f"Window {fold}" for fold in folds["Fold"]]
    figure = go.Figure(
        [
            go.Bar(
                x=names,
                y=folds["In-Sample Score"],
                name="Training (hindsight)",
                marker={"color": theme.benchmark},
            ),
            go.Bar(
                x=names,
                y=folds["Out-of-Sample Score"],
                name="Next window (unseen)",
                marker={"color": theme.strategy},
            ),
        ]
    )
    figure.update_traces(hovertemplate="%{y:.2f}", marker_cornerradius=4)
    figure.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08, hovermode="x unified")
    return _style(figure, theme, f"{label}: promised vs delivered", label)


def comparison_figure(
    equity: pd.DataFrame, theme: ChartTheme = LIGHT, *, log_scale: bool = True
) -> go.Figure:
    """Every strategy's equity on one log scale, so equal percentage moves look equal across time."""
    figure = go.Figure(
        [
            _line(
                equity[column],
                column,
                strategy_color(column, theme),
                width=2.5 if column in (BuyAndHoldStrategy.label, RuleStrategy.label) else 1.75,
                hovertemplate="$%{y:,.0f}",
            )
            for column in equity.columns
        ]
    )
    _crosshair(figure, theme)
    figure.update_yaxes(tickprefix="$", separatethousands=True, type="log" if log_scale else "linear")
    _style(figure, theme, "Every strategy at its default settings", "Portfolio value", height=480)
    # Six names do not fit on one line above the plot, so the legend moves below it.
    figure.update_layout(legend={"yanchor": "top", "y": -0.12}, margin={"b": 8})
    return figure


def start_dates_figure(windows: pd.DataFrame, years: int, theme: ChartTheme = LIGHT) -> go.Figure:
    """Growth per year of each equal-length window, plotted at the month it starts."""
    figure = go.Figure(
        [
            _line(windows.set_index("Start")["Hold"], HOLD_NAME, theme.benchmark),
            _line(windows.set_index("Start")["Rule"], RULE_NAME, theme.strategy),
        ]
    )
    figure.add_hline(y=0, line={"color": theme.axis, "width": 1})
    _crosshair(figure, theme)
    figure.update_traces(hovertemplate="%{y:.1%} a year")
    figure.update_yaxes(tickformat=".0%")
    title = f"Growth per year over the next {years} year{'s' if years != 1 else ''}, by start month"
    return _style(figure, theme, title, "Growth per year", "Start month")


def best_trades_figure(
    strategy: pd.Series, without: pd.Series, benchmark: pd.Series, removed: int, theme: ChartTheme = LIGHT
) -> go.Figure:
    """The rule with and without its best trades, against holding."""
    label = f"Without its best {removed}"
    figure = go.Figure(
        [
            _line(benchmark, HOLD_NAME, theme.benchmark),
            _line(strategy, RULE_NAME, theme.strategy),
            _line(without, label, theme.strategy, dash="dot"),
        ]
    )
    _label_line_end(figure, without, label, theme)
    _crosshair(figure, theme)
    figure.update_traces(hovertemplate="$%{y:,.0f}")
    figure.update_yaxes(tickprefix="$", separatethousands=True)
    _style(figure, theme, "Growth without the best trades", "Portfolio value")
    figure.update_layout(margin={"r": 130})
    _fit_dates(figure, strategy.index)
    return figure


def crashes_figure(crashes: list[Crash], theme: ChartTheme = LIGHT) -> go.Figure:
    """Each deep fall of the asset, from peak to bottom, for the rule and for holding."""
    names = [crash.name for crash in crashes]
    figure = go.Figure(
        [
            go.Bar(
                x=names, y=[crash.rule for crash in crashes], name=RULE_NAME, marker={"color": theme.strategy}
            ),
            go.Bar(
                x=names,
                y=[crash.hold for crash in crashes],
                name=HOLD_NAME,
                marker={"color": theme.benchmark},
            ),
        ]
    )
    figure.update_traces(hovertemplate="%{y:.1%}", marker_cornerradius=4)
    figure.update_layout(barmode="group", bargap=0.35, bargroupgap=0.08, hovermode="x unified")
    figure.update_yaxes(tickformat=".0%")
    figure.update_xaxes(type="category")
    return _style(figure, theme, "Loss from the asset's peak to its bottom", "Change")
