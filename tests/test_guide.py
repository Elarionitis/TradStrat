"""Tests for the plain-English strategy guides and the glossary's hover definitions."""

from __future__ import annotations

import re

import pytest

from tradstrat.glossary import ALIASES, GLOSSARY, annotate
from tradstrat.guide import GUIDES
from tradstrat.strategies import STRATEGIES


def test_every_registered_strategy_has_a_guide() -> None:
    """A new strategy must explain itself before it appears in the dashboard."""
    assert set(GUIDES) == set(STRATEGIES)


@pytest.mark.parametrize("key", sorted(STRATEGIES))
def test_each_rule_names_the_asset_and_its_settings(key: str) -> None:
    params = {spec.name: spec.default for spec in STRATEGIES[key].params}

    rule = GUIDES[key].describe_rule("QQQ", params)

    assert "QQQ" in rule
    assert "{" not in rule
    for value in params.values():
        assert f"{value:g}" in rule or f"{value:.1f}" in rule


def test_rules_follow_the_settings_being_tested() -> None:
    rule = GUIDES["ma_crossover"].describe_rule("SPY", {"short_window": 20, "long_window": 100})

    assert "20-day average" in rule
    assert "100-day average" in rule


def test_every_alias_points_at_a_real_definition() -> None:
    assert set(ALIASES.values()) <= set(GLOSSARY)


def test_annotate_explains_a_term_with_its_definition() -> None:
    marked = annotate("A high Sharpe ratio")

    assert marked.startswith("A high <span")
    assert 'data-tip="Return earned for each unit of risk' in marked
    assert marked.endswith(">Sharpe ratio</span>")


def test_annotate_prefers_the_longest_matching_term() -> None:
    marked = annotate("The deflated Sharpe ratio")

    assert ">deflated Sharpe ratio</span>" in marked
    assert marked.count("<span") == 1


def test_annotate_explains_each_term_once_across_shared_blocks() -> None:
    seen: set[str] = set()

    first = annotate("Sharpe ratio and drawdown", seen)
    second = annotate("A worse Sharpe ratio and a deeper worst decline", seen)

    assert first.count("<span") == 2
    assert "<span" not in second


def test_annotate_escapes_text_and_definitions() -> None:
    marked = annotate("<script>alert(1)</script> & momentum")

    assert "<script>" not in marked
    assert "&lt;script&gt;" in marked
    assert "&amp;" in marked
    tip = re.search(r'data-tip="([^"]*)"', marked)
    assert tip is not None
    assert '"' not in tip.group(1)


def test_annotate_ignores_terms_inside_longer_words() -> None:
    assert "<span" not in annotate("Rsizing and walk-forwardness")
