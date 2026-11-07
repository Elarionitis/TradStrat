"""Plotly figures for the dashboard.

Colors are assigned by role and were checked with a color-vision-deficiency validator. Every multi-series
chart has a legend, equity curves are also labeled directly, and buy and sell markers differ in shape, so
identity never rests on color alone. Heatmaps of signed values use a diverging scale centered on zero.
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np
import pandas as pd
import plotly.graph_objects as go

STRATEGY_COLOR: Final[str] = "#2a78d6"
BENCHMARK_COLOR: Final[str] = "#eb6834"
BUY_COLOR: Final[str] = "#1baf7a"
GRID_COLOR: Final[str] = "#e8e7e3"
MUTED_COLOR: Final[str] = "#52514e"
DIVERGING_SCALE: Final[list[list[Any]]] = [[0.0, "#eb6834"], [0.5, "#f1f0ec"], [1.0, "#2a78d6"]]
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


def _style(figure: go.Figure, title: str, y_title: str = "", x_title: str = "") -> go.Figure:
    """Apply the shared layout: recessive grid, one axis, legend above the plot."""
    figure.update_layout(
        title={"text": title, "x": 0, "xanchor": "left"},
        margin={"l": 10, "r": 10, "t": 70, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "x": 0},
        font={"color": MUTED_COLOR},
    )
    figure.update_xaxes(title=x_title, showgrid=False, linecolor=GRID_COLOR)
    figure.update_yaxes(title=y_title, gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
    return figure


def _line(series: pd.Series, name: str, color: str) -> go.Scatter:
    return go.Scatter(x=series.index, y=series, name=name, mode="lines", line={"width": 2, "color": color})


def _label_line_end(figure: go.Figure, series: pd.Series, text: str) -> None:
    """Name a line at its last point, so a reader need not match colors against the legend."""
    figure.add_annotation(
        x=series.index[-1], y=float(series.iloc[-1]), text=text, showarrow=False, xanchor="left", xshift=6
    )


def equity_figure(strategy: pd.Series, benchmark: pd.Series, title: str = "Equity Curve") -> go.Figure:
    figure = go.Figure(
        [
            _line(strategy, "Strategy", STRATEGY_COLOR),
            _line(benchmark, "Buy and Hold", BENCHMARK_COLOR),
        ]
    )
    _label_line_end(figure, strategy, "Strategy")
    _label_line_end(figure, benchmark, "Buy and Hold")
    figure.update_layout(hovermode="x unified")
    figure.update_yaxes(tickprefix="$", separatethousands=True)
    _style(figure, title, "Portfolio value")
    figure.update_layout(margin={"r": 95})
    return figure


def drawdown_figure(strategy: pd.Series, benchmark: pd.Series) -> go.Figure:
    figure = go.Figure(
        [
            _line(strategy, "Strategy", STRATEGY_COLOR),
            _line(benchmark, "Buy and Hold", BENCHMARK_COLOR),
        ]
    )
    figure.update_layout(hovermode="x unified")
    figure.update_yaxes(tickformat=".0%")
    return _style(figure, "Drawdown", "Decline from peak")


def price_figure(close: pd.Series, trades: pd.DataFrame) -> go.Figure:
    figure = go.Figure([_line(close, "Close", STRATEGY_COLOR)])
    if not trades.empty:
        for name, date, price, symbol, color in (
            ("Buy", "Entry Date", "Entry Price", "triangle-up", BUY_COLOR),
            ("Sell", "Exit Date", "Exit Price", "triangle-down", BENCHMARK_COLOR),
        ):
            figure.add_trace(
                go.Scatter(
                    x=trades[date],
                    y=trades[price],
                    name=name,
                    mode="markers",
                    marker={
                        "symbol": symbol,
                        "size": 10,
                        "color": color,
                        "line": {"width": 1, "color": "white"},
                    },
                )
            )
    figure.update_layout(hovermode="closest")
    return _style(figure, "Price and Trades", "Price (dividend-adjusted)")


def monthly_returns_table(equity: pd.Series) -> pd.DataFrame:
    """Compound an equity curve into a table of returns with years as rows and months as columns."""
    month_ends = equity.resample("ME").last()
    returns = month_ends.pct_change()
    returns.iloc[0] = month_ends.iloc[0] / equity.iloc[0] - 1
    dates = pd.DatetimeIndex(returns.index)
    frame = pd.DataFrame({"year": dates.year, "month": dates.month, "value": returns.to_numpy()})
    return frame.pivot_table(index="year", columns="month", values="value").reindex(columns=range(1, 13))


def monthly_returns_figure(equity: pd.Series) -> go.Figure:
    table = monthly_returns_table(equity)
    limit = float(np.nanmax(np.abs(table.to_numpy()))) or 1.0
    figure = go.Figure(
        go.Heatmap(
            z=table.to_numpy(),
            x=MONTHS,
            y=[str(year) for year in table.index],
            colorscale=DIVERGING_SCALE,
            zmin=-limit,
            zmax=limit,
            xgap=2,
            ygap=2,
            colorbar={"tickformat": ".0%", "title": ""},
            hovertemplate="%{x} %{y}: %{z:.2%}<extra></extra>",
        )
    )
    figure.update_yaxes(autorange="reversed", type="category")
    _style(figure, "Monthly Returns")
    figure.update_yaxes(showgrid=False)
    return figure


def bootstrap_figure(samples: np.ndarray, low: float, high: float, observed: float) -> go.Figure:
    figure = go.Figure(
        go.Histogram(
            x=samples[~np.isnan(samples)], nbinsx=50, marker={"color": STRATEGY_COLOR}, name="Resamples"
        )
    )
    figure.update_traces(marker_line={"width": 2, "color": "white"})
    figure.add_vline(x=0, line={"color": MUTED_COLOR, "width": 1})
    for value, label in ((low, "2.5%"), (high, "97.5%")):
        figure.add_vline(
            x=value, line={"color": MUTED_COLOR, "width": 2, "dash": "dash"}, annotation_text=label
        )
    figure.add_vline(
        x=observed,
        line={"color": BENCHMARK_COLOR, "width": 2},
        annotation_text="Observed",
        annotation_position="top left",
    )
    return _style(figure, "Bootstrap Distribution of the Sharpe Ratio", "Resamples", "Sharpe ratio")


def search_figure(table: pd.DataFrame, parameters: list[str], objective: str) -> go.Figure:
    """Show how the objective varies across parameters: a line for one, a heatmap for two or more.

    A broad plateau suggests a robust idea, and an isolated peak suggests a fitted one. With three or more
    parameters each cell shows the best score over the parameters not on the axes.
    """
    label = objective.replace("_", " ").title()
    if len(parameters) == 1:
        ordered = table.sort_values(parameters[0])
        figure = go.Figure(
            go.Scatter(
                x=ordered[parameters[0]],
                y=ordered[objective],
                mode="lines+markers",
                line={"width": 2, "color": STRATEGY_COLOR},
                marker={"size": 8, "color": STRATEGY_COLOR},
            )
        )
        return _style(figure, f"{label} by Parameter", label, parameters[0])

    x_name, y_name = parameters[:2]
    grid = table.pivot_table(index=y_name, columns=x_name, values=objective, aggfunc="max")
    limit = float(np.nanmax(np.abs(grid.to_numpy()))) if grid.notna().any().any() else 1.0
    figure = go.Figure(
        go.Heatmap(
            z=grid.to_numpy(),
            x=[str(value) for value in grid.columns],
            y=[str(value) for value in grid.index],
            colorscale=DIVERGING_SCALE,
            zmin=-limit,
            zmax=limit,
            xgap=2,
            ygap=2,
            hovertemplate=f"{x_name}=%{{x}}<br>{y_name}=%{{y}}<br>{label}=%{{z:.2f}}<extra></extra>",
        )
    )
    _style(figure, f"{label} Across the Parameter Grid", y_name, x_name)
    figure.update_yaxes(showgrid=False, type="category")
    figure.update_xaxes(type="category")
    return figure


def fold_scores_figure(folds: pd.DataFrame, objective: str) -> go.Figure:
    """Compare what each fold's parameters promised in training with what they delivered afterwards."""
    label = objective.replace("_", " ").title()
    names = [f"Fold {fold}" for fold in folds["Fold"]]
    figure = go.Figure(
        [
            go.Bar(x=names, y=folds["In-Sample Score"], name="In-sample", marker={"color": BENCHMARK_COLOR}),
            go.Bar(
                x=names,
                y=folds["Out-of-Sample Score"],
                name="Out-of-sample",
                marker={"color": STRATEGY_COLOR},
            ),
        ]
    )
    figure.update_layout(barmode="group", bargap=0.3, bargroupgap=0.05, hovermode="x unified")
    return _style(figure, f"{label}: Promised vs Delivered", label)
