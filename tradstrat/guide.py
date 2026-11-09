"""Plain-English explanations of the built-in strategies, for readers who have never met them before.

Each guide says what the strategy believes, exactly when it buys and sells, and the markets it suits and
struggles in. ``rule`` is a template filled with the strategy's current settings, so the dashboard can state
the rule being tested in one sentence, such as "Own SPY while its 50-day average price is above its 200-day
average price".

Guides are optional. A strategy without one is shown with its own one-line ``description`` instead.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from tradstrat.strategies import Strategy


@dataclass(frozen=True)
class StrategyGuide:
    """Everything a newcomer needs to understand one strategy."""

    style: str
    summary: str
    idea: str = ""
    buys: str = ""
    sells: str = ""
    works_when: str = ""
    struggles_when: str = ""
    rule: str = ""

    def describe_rule(self, asset: str, params: Mapping[str, float]) -> str:
        """The rule in one sentence, with this run's asset and settings filled in."""
        if not self.rule:
            return self.summary
        return self.rule.format(asset=asset, **params)


GUIDES: dict[str, StrategyGuide] = {
    "buy_and_hold": StrategyGuide(
        style="Benchmark",
        summary="Buys on the first day and never sells.",
        idea=(
            "Doing nothing clever is hard to beat. Every strategy is compared with simply holding the asset, "
            "paying the same costs, so a result only counts if it beats this."
        ),
        buys="On the first day of the test.",
        sells="Never.",
        works_when="Markets rise over time, as broad stock indices have historically done.",
        struggles_when="Long bear markets, which it rides all the way down.",
        rule="Buy {asset} on the first day and hold it to the end.",
    ),
    "ma_crossover": StrategyGuide(
        style="Trend following",
        summary="Owns the asset while its recent average price is above its longer-term average.",
        idea=(
            "Prices that have been rising tend to keep rising for a while. Comparing a fast moving average "
            "with a slow one is a simple way to tell whether the trend is up, while ignoring daily noise."
        ),
        buys="The fast average rises above the slow average.",
        sells="The fast average falls back below the slow average.",
        works_when="Long, steady trends, and prolonged crashes that it can step aside from.",
        struggles_when=(
            "Choppy, sideways markets, where the averages cross back and forth and every crossing costs a "
            "trade. It also reacts late, so it gives back part of each move."
        ),
        rule=(
            "Own {asset} while its {short_window}-day average price is above its {long_window}-day average "
            "price, and hold cash otherwise."
        ),
    ),
    "mean_reversion": StrategyGuide(
        style="Mean reversion",
        summary=(
            "Buys after the price falls unusually far below its recent average, then sells on the bounce."
        ),
        idea=(
            "Short-term moves often overshoot. A z-score measures how far today's price is from its recent "
            "average, in units of its usual swing, so it can tell a normal dip from an unusual one."
        ),
        buys="The z-score drops to the entry threshold below zero: the price is stretched unusually low.",
        sells="The z-score climbs back to the exit threshold. An exit of zero means back to average.",
        works_when="Range-bound markets that swing around a stable level.",
        struggles_when=(
            "Strong downtrends, where cheap keeps getting cheaper. It has no exit for losses of its own, so "
            "consider adding a stop loss."
        ),
        rule=(
            "Buy {asset} when its price falls {entry_threshold:.1f} standard deviations below its "
            "{lookback_window}-day average, and sell once its z-score recovers to {exit_threshold:.1f}."
        ),
    ),
    "rsi": StrategyGuide(
        style="Mean reversion",
        summary="Buys after a run of heavy selling, and sells once buyers return.",
        idea=(
            "The Relative Strength Index (RSI) scores recent gains against recent losses on a scale from "
            "0 to 100. Low readings mean sellers have been in control for a while, which is often "
            "followed by a rebound. Readings under 30 are traditionally called oversold."
        ),
        buys="RSI falls below the oversold level.",
        sells="RSI rises above the exit level.",
        works_when="Markets that dip and recover, such as broad stock indices in a rising market.",
        struggles_when="Long selloffs, where RSI can stay low for weeks while the price keeps falling.",
        rule=(
            "Buy {asset} when its {window}-day RSI drops below {oversold:.0f}, and sell when it rises above "
            "{exit_level:.0f}."
        ),
    ),
    "donchian": StrategyGuide(
        style="Trend following",
        summary="Buys a breakout to a new high, and sells on a breakdown to a shorter-term low.",
        idea=(
            "A close above every price of recent weeks can mark the start of a new trend. A Donchian channel "
            "is just the highest high and lowest low over a window. Trend followers popularized this rule in "
            "the 1980s."
        ),
        buys="The close is above the highest high of the entry window: the top of the channel.",
        sells="The close is below the lowest low of the shorter exit window.",
        works_when="Markets that break out and keep going, such as commodities in strong trends.",
        struggles_when=(
            "False breakouts in range-bound markets. Expect many small losses paid for by a few large wins."
        ),
        rule=(
            "Buy {asset} when it closes above its highest price of the last {entry_window} days, and sell "
            "when it closes below its lowest price of the last {exit_window} days."
        ),
    ),
    "momentum": StrategyGuide(
        style="Trend following",
        summary="Owns the asset while it is up over the lookback period, and holds cash while it is down.",
        idea=(
            "Time-series momentum: across many markets, an asset's own return over the past few months has "
            "tended to say something about its direction over the next few."
        ),
        buys="The return over the lookback window turns positive.",
        sells="The return over the lookback window turns negative.",
        works_when="Persistent rising or falling markets.",
        struggles_when="Sharp V-shaped reversals, where it sells near the bottom and buys back higher.",
        rule=(
            "Own {asset} while it is up over the last {lookback_window} trading days, and hold cash "
            "otherwise."
        ),
    ),
}


def guide_for(strategy: type[Strategy]) -> StrategyGuide:
    """The written guide for a strategy, or a minimal one built from its own description."""
    return GUIDES.get(strategy.key) or StrategyGuide(style="Custom", summary=strategy.description)
