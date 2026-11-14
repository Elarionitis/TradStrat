"""Trading rules written in short, readable phrases, so anyone can test an idea without writing code.

A rule is a buy condition and an optional sell condition, such as::

    buy:  sma(50) crosses above sma(200)
    sell: sma(50) crosses below sma(200)

Each condition compares two things with ``>``, ``<``, ``crosses above`` or ``crosses below``, and joins any
number of comparisons with ``and`` or with ``or``. The things it can compare are:

    close           the closing price (``price`` also works)
    sma(days)       simple moving average of the close
    ema(days)       exponential moving average of the close
    rsi(days)       Wilder's Relative Strength Index, from 0 to 100
    high(days)      highest high of the previous ``days`` bars, not counting today
    low(days)       lowest low of the previous ``days`` bars, not counting today
    return(days)    percentage change in the close over ``days`` bars, so 5 means 5%
    zscore(days)    distance of the close from its average, in standard deviations
    a number        such as 30, -1.5 or 10%

Without a sell condition the rule owns the asset only while the buy condition is true. A cross lasts a single
day, so a rule that buys on one needs a sell condition.

Every value on a bar uses data up to and including that bar's close, and the engine trades at the next open,
so a rule cannot see the future. Every number in a rule becomes a parameter, so a parameter search and
walk-forward validation can tune a custom rule exactly as they tune the built-in strategies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any, ClassVar, Final

import numpy as np
import pandas as pd

from tradstrat.guide import StrategyGuide
from tradstrat.strategies import ParamSpec, Strategy, _hold_until_exit, relative_strength_index

OPERATORS: Final[dict[str, str]] = {
    ">": "is above",
    "<": "is below",
    "crosses above": "crosses above",
    "crosses below": "crosses below",
}
FLIPPED: Final[dict[str, str]] = {
    ">": "<",
    "<": ">",
    "crosses above": "crosses below",
    "crosses below": "crosses above",
}


@dataclass(frozen=True)
class Indicator:
    phrase: str  # Plain English, with {days} where the window goes.
    minimum: int  # Fewest days the indicator accepts.
    extra: int  # Bars it needs beyond its window before its first value.
    line: str = ""  # Legend name when drawn on the price chart, for indicators in price units.


INDICATORS: Final[dict[str, Indicator]] = {
    "sma": Indicator("its {days}-day average price", 2, 0, "{days}-day average"),
    "ema": Indicator("its {days}-day exponential average price", 2, 0, "{days}-day EMA"),
    "rsi": Indicator("its {days}-day RSI", 2, 1),
    "high": Indicator("its highest price of the previous {days} days", 1, 1, "{days}-day high"),
    "low": Indicator("its lowest price of the previous {days} days", 1, 1, "{days}-day low"),
    "return": Indicator("its {days}-day return", 1, 1),
    "zscore": Indicator("its {days}-day z-score", 2, 0),
}
MAX_DAYS: Final[int] = 2_000
VOCABULARY: Final[str] = (
    "close, sma(days), ema(days), rsi(days), high(days), low(days), return(days), zscore(days)"
)

# Ready-made rules that show what the language can do. Each is a well-known idea.
EXAMPLES: Final[dict[str, tuple[str, str]]] = {
    "Golden cross": ("sma(50) crosses above sma(200)", "sma(50) crosses below sma(200)"),
    "Above the 200-day": ("close > sma(200)", ""),
    "Dip in an uptrend": ("rsi(2) < 10 and close > sma(200)", "rsi(2) > 70 or close < sma(200)"),
    "52-week high": ("close > high(252)", "close < low(50)"),
    "Fast EMA trend": ("ema(20) > ema(50)", ""),
}

_ALIASES: Final[dict[str, str]] = {"price": "close", "ma": "sma", "roc": "return", "z": "zscore"}
_COMPARISON = re.compile(r"^(?P<left>.+?)\s*(?P<op>crosses\s+above|crosses\s+below|>|<)\s*(?P<right>.+)$")
_INDICATOR = re.compile(r"^(?P<name>[a-z]+)\s*\(\s*(?P<days>[^)]*?)\s*\)$")
_NUMBER = re.compile(r"^[-+]?(\d+(\.\d*)?|\.\d+)%?$")


class RuleError(ValueError):
    """A rule that cannot be read, with a message that says how to fix it."""


@dataclass(frozen=True)
class Operand:
    """One side of a comparison. ``slot`` indexes the rule's numbers: an indicator's days or a constant."""

    kind: str
    slot: int | None = None


@dataclass(frozen=True)
class Comparison:
    left: Operand
    operator: str
    right: Operand


@dataclass(frozen=True)
class Condition:
    comparisons: tuple[Comparison, ...]
    joiner: str = "and"


@dataclass(frozen=True)
class ParsedRule:
    buy: Condition
    sell: Condition | None
    defaults: tuple[float, ...]
    kinds: tuple[str, ...]  # What each number is: an indicator's name for its days, or "number".


class _Parser:
    """Reads both conditions of a rule, numbering every value it meets in the order it meets them."""

    def __init__(self) -> None:
        self.values: list[float] = []
        self.kinds: list[str] = []
        self.windows: dict[tuple[str, int], int] = {}

    def condition(self, text: str, side: str) -> Condition:
        text = " ".join(text.lower().split())
        if not text:
            raise RuleError(f"The {side} rule is empty.")
        has_and = re.search(r"\band\b", text) is not None
        has_or = re.search(r"\bor\b", text) is not None
        if has_and and has_or:
            raise RuleError(f"Use either 'and' or 'or' in the {side} rule, not both.")
        joiner = "or" if has_or else "and"
        parts = re.split(rf"\s+{joiner}\s+", text)
        return Condition(tuple(self.comparison(part, side) for part in parts), joiner)

    def comparison(self, text: str, side: str) -> Comparison:
        match = _COMPARISON.match(text.strip())
        if not match:
            raise RuleError(
                f"Could not read '{text}' in the {side} rule. Compare two things with >, <, "
                "'crosses above' or 'crosses below', as in 'close > sma(200)'."
            )
        operator = " ".join(match["op"].split())
        left_text, right_text = match["left"].strip(), match["right"].strip()
        if _NUMBER.match(left_text) and _NUMBER.match(right_text):
            raise RuleError(
                f"'{text}' compares two numbers. Compare an indicator with a number or another one."
            )
        if _NUMBER.match(left_text):
            # "30 > rsi(14)" reads more naturally the other way round, and means the same.
            left_text, right_text, operator = right_text, left_text, FLIPPED[operator]
        return Comparison(self.operand(left_text), operator, self.operand(right_text))

    def operand(self, text: str) -> Operand:
        if _NUMBER.match(text):
            return Operand("number", self._slot(float(text.rstrip("%")), "number"))
        name = _ALIASES.get(text, text)
        if name == "close":
            return Operand("close")
        match = _INDICATOR.match(text)
        if match:
            name = _ALIASES.get(match["name"], match["name"])
            if name in INDICATORS:
                days = match["days"]
                minimum = INDICATORS[name].minimum
                if not days.isdigit() or not minimum <= int(days) <= MAX_DAYS:
                    raise RuleError(
                        f"{name}({days}) needs a whole number of days from {minimum} to {MAX_DAYS:,}."
                    )
                # The same indicator used twice, as in a buy and a sell rule, is one setting, not two.
                window = (name, int(days))
                if window not in self.windows:
                    self.windows[window] = self._slot(int(days), name)
                return Operand(name, self.windows[window])
        raise RuleError(f"'{text}' is not something a rule can use. Try one of: {VOCABULARY}, or a number.")

    def _slot(self, value: float, kind: str) -> int:
        self.values.append(value)
        self.kinds.append(kind)
        return len(self.values) - 1


def parse_rule(buy: str, sell: str = "") -> ParsedRule:
    """Read a rule, raising ``RuleError`` with a fix-it message if it cannot be read."""
    parser = _Parser()
    buy_condition = parser.condition(buy, "buy")
    sell_condition = parser.condition(sell, "sell") if sell.strip() else None
    if sell_condition is None and any(c.operator.startswith("crosses") for c in buy_condition.comparisons):
        raise RuleError(
            "A cross happens on a single day, so a rule that buys on one needs a sell rule too. Add one, or "
            "use > or < to own the asset for as long as the condition holds."
        )
    return ParsedRule(buy_condition, sell_condition, tuple(parser.values), tuple(parser.kinds))


def _partner(comparison: Comparison, operand: Operand) -> str:
    """The kind of thing a constant is compared with, which sets its units and sensible range."""
    other = comparison.left if operand is comparison.right else comparison.right
    return other.kind


def _number_spec(name: str, value: float, partner: str) -> ParamSpec:
    if partner == "rsi":
        low, high, step, label = max(1.0, value - 15), min(99.0, value + 15), 1.0, "RSI level"
    elif partner == "return":
        low, high, step, label = value - 10, value + 10, 0.5, "Return level"
    elif partner == "zscore":
        low, high, step, label = value - 1, value + 1, 0.1, "Z-score level"
    else:
        span = abs(value) / 2 or 1.0
        low, high, step, label = value - span, value + span, round(span / 25, 4) or 0.01, "Level"
    return ParamSpec(
        name,
        f"{label} {value:g}{'%' if partner == 'return' else ''}",
        value,
        round(low, 4),
        round(high, 4),
        step,
        f"The number {value:g} in the rule.",
    )


def _days_spec(name: str, kind: str, days: int) -> ParamSpec:
    indicator = INDICATORS[kind]
    low = max(indicator.minimum, round(days / 2))
    high = max(low + 1, round(days * 1.5))
    return ParamSpec(name, f"Days in {kind}({days})", days, low, high, 1, f"The {days} in {kind}({days}).")


def _param_specs(rule: ParsedRule) -> tuple[ParamSpec, ...]:
    specs: dict[int, ParamSpec] = {}
    conditions = [rule.buy, *([rule.sell] if rule.sell else [])]
    for condition in conditions:
        for comparison in condition.comparisons:
            for operand in (comparison.left, comparison.right):
                if operand.slot is None:
                    continue
                name = f"n{operand.slot + 1}"
                value = rule.defaults[operand.slot]
                if operand.kind == "number":
                    specs[operand.slot] = _number_spec(name, value, _partner(comparison, operand))
                else:
                    specs[operand.slot] = _days_spec(name, operand.kind, int(value))
    # Two identical constants, such as two RSI levels of 30, need different names in a results table.
    unique: list[ParamSpec] = []
    for slot in sorted(specs):
        spec = specs[slot]
        taken = sum(other.label == spec.label for other in unique)
        unique.append(replace(spec, label=f"{spec.label} ({taken + 1})") if taken else spec)
    return tuple(unique)


def _render(operand: Operand, names: list[str], *, english: bool, percent: bool = False) -> str:
    """Write an operand back out, with ``{nK}`` placeholders where its numbers go."""
    slot = "" if operand.slot is None else "{" + names[operand.slot] + "}"
    if operand.kind == "close":
        return "its price" if english else "close"
    if operand.kind == "number":
        number = "{" + names[operand.slot or 0] + ":g}"
        return number + ("%" if percent else "")
    if english:
        return INDICATORS[operand.kind].phrase.format(days=slot)
    return f"{operand.kind}({slot})"


def _render_condition(condition: Condition, names: list[str], *, english: bool) -> str:
    parts = []
    for comparison in condition.comparisons:
        percent = "return" in (comparison.left.kind, comparison.right.kind)
        left = _render(comparison.left, names, english=english, percent=percent)
        right = _render(comparison.right, names, english=english, percent=percent)
        operator = OPERATORS[comparison.operator] if english else comparison.operator
        parts.append(f"{left} {operator} {right}")
    return f" {condition.joiner} ".join(parts)


def _templates(rule: ParsedRule) -> tuple[str, str]:
    """The rule in plain English for ``{asset}``, and in its own syntax, both with number placeholders."""
    names = [f"n{slot + 1}" for slot in range(len(rule.defaults))]
    buy_english = _render_condition(rule.buy, names, english=True)
    buy_text = _render_condition(rule.buy, names, english=False)
    if rule.sell is None:
        return (
            f"Own {{asset}} while {buy_english}, and hold cash otherwise.",
            f"buy while {buy_text}",
        )
    sell_english = _render_condition(rule.sell, names, english=True)
    sell_text = _render_condition(rule.sell, names, english=False)
    return (
        f"Buy {{asset}} when {buy_english}, and sell when {sell_english}.",
        f"buy when {buy_text}; sell when {sell_text}",
    )


class RuleStrategy(Strategy):
    """A strategy defined by a rule. ``compile_rule`` makes one subclass per rule, with its numbers as
    parameters, so the optimizer can build variants of it by name like any other strategy."""

    key: ClassVar[str] = "custom"
    label: ClassVar[str] = "Your own rule"
    description: ClassVar[str] = "A rule you write."
    rule: ClassVar[ParsedRule]
    guide: ClassVar[StrategyGuide]
    syntax: ClassVar[str]

    def __init__(self, **values: float) -> None:
        for spec, kind in zip(self.params, self.rule.kinds, strict=True):
            value = values.get(spec.name, spec.default)
            if kind != "number":
                minimum = INDICATORS[kind].minimum
                if int(value) != value or not minimum <= int(value) <= MAX_DAYS:
                    raise ValueError(f"{spec.label} must be a whole number from {minimum} to {MAX_DAYS:,}.")
                value = int(value)
            setattr(self, spec.name, value)

    @property
    def values(self) -> tuple[float, ...]:
        return tuple(getattr(self, spec.name) for spec in self.params)

    @property
    def text(self) -> str:
        """The rule in its own syntax, with this instance's numbers filled in."""
        return self.syntax.format(**{spec.name: getattr(self, spec.name) for spec in self.params})

    def __repr__(self) -> str:
        return f"Rule({self.text!r})"

    @property
    def warmup(self) -> int:
        bars = 0
        conditions = [self.rule.buy, *([self.rule.sell] if self.rule.sell else [])]
        for condition in conditions:
            for comparison in condition.comparisons:
                crossing = 1 if comparison.operator.startswith("crosses") else 0
                for operand in (comparison.left, comparison.right):
                    if operand.kind in INDICATORS and operand.slot is not None:
                        days = int(self.values[operand.slot])
                        bars = max(bars, days + INDICATORS[operand.kind].extra + crossing)
        return bars

    def _series(self, operand: Operand, data: pd.DataFrame) -> pd.Series:
        close = data["Close"].astype(float)
        if operand.kind == "close":
            return close
        value = self.values[operand.slot or 0]
        if operand.kind == "number":
            return pd.Series(float(value), index=data.index)
        days = int(value)
        if operand.kind == "sma":
            return close.rolling(days, min_periods=days).mean()
        if operand.kind == "ema":
            return close.ewm(span=days, adjust=False, min_periods=days).mean()
        if operand.kind == "rsi":
            return relative_strength_index(close, days)
        if operand.kind == "high":
            return data["High"].astype(float).rolling(days, min_periods=days).max().shift(1)
        if operand.kind == "low":
            return data["Low"].astype(float).rolling(days, min_periods=days).min().shift(1)
        if operand.kind == "return":
            return close.pct_change(days, fill_method=None) * 100
        rolling = close.rolling(days, min_periods=days)
        return (close - rolling.mean()) / rolling.std(ddof=0)

    def _evaluate(self, condition: Condition, data: pd.DataFrame) -> np.ndarray:
        results = []
        for comparison in condition.comparisons:
            left = self._series(comparison.left, data).to_numpy()
            right = self._series(comparison.right, data).to_numpy()
            # NaN compares as False, so nothing fires until both sides are defined.
            above, below = left > right, left < right
            if comparison.operator == ">":
                results.append(above)
            elif comparison.operator == "<":
                results.append(below)
            else:
                now, before_opposite = (
                    (above, ~above) if comparison.operator == "crosses above" else (below, ~below)
                )
                defined = ~(np.isnan(left) | np.isnan(right))
                previous = np.concatenate([[False], (before_opposite & defined)[:-1]])
                results.append(now & previous)
        stacked = np.stack(results)
        return np.all(stacked, axis=0) if condition.joiner == "and" else np.any(stacked, axis=0)

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        buy = self._evaluate(self.rule.buy, data)
        if self.rule.sell is None:
            signal = buy.astype(int)
        else:
            signal = _hold_until_exit(buy, self._evaluate(self.rule.sell, data))
        columns: dict[str, Any] = {"signal": signal}
        for column, _ in self.overlays:
            slot = int(column.removeprefix("line"))
            operand = _price_operands(self.rule)[slot]
            columns[column] = self._series(operand, data)
        return pd.DataFrame(columns, index=data.index)


def _price_operands(rule: ParsedRule) -> list[Operand]:
    """Indicators drawn in price units, which can share the price chart. Each appears once."""
    seen: list[Operand] = []
    conditions = [rule.buy, *([rule.sell] if rule.sell else [])]
    for condition in conditions:
        for comparison in condition.comparisons:
            for operand in (comparison.left, comparison.right):
                if operand.kind in INDICATORS and INDICATORS[operand.kind].line and operand not in seen:
                    seen.append(operand)
    return seen


def compile_rule(buy: str, sell: str = "") -> type[RuleStrategy]:
    """Turn a rule into a strategy class whose parameters are the rule's numbers."""
    rule = parse_rule(buy, sell)
    params = _param_specs(rule)
    english, syntax = _templates(rule)
    names = {spec.name: spec.default for spec in params}
    overlays = tuple(
        (f"line{index}", INDICATORS[operand.kind].line.format(days=int(rule.defaults[operand.slot or 0])))
        for index, operand in enumerate(_price_operands(rule))
    )
    guide = StrategyGuide(style="Your rule", summary=syntax.format(**names).capitalize() + ".", rule=english)
    attributes = {
        "rule": rule,
        "params": params,
        "overlays": overlays,
        "guide": guide,
        "syntax": syntax,
        "description": guide.summary,
    }
    return type("CustomRule", (RuleStrategy,), attributes)
