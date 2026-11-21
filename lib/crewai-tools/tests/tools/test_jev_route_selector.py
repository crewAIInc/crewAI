"""Behavioral tests for the Jev Choice confidence-aware route selector.

All tests use synthetic decisions; none require provider credentials or network.
"""

from __future__ import annotations

import math
from typing import Any

from crewai.flow import Flow, listen, router, start
from crewai_tools.tools.jev_decision_tool.route_selector import select_route
import pytest


# A set, so an unhashable ``choice`` would raise if the function did not guard it.
ROUTES: frozenset[str] = frozenset({"a", "b", "fallback"})
FALLBACK = "fallback"
THRESHOLD = 0.7


def _select(choice: Any, confidence: Any, **overrides: Any) -> str:
    kwargs: dict[str, Any] = {
        "allowed_routes": ROUTES,
        "fallback": FALLBACK,
        "min_confidence": THRESHOLD,
    }
    kwargs.update(overrides)
    return select_route(choice, confidence, **kwargs)


# --- threshold -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("confidence", "expected"),
    [
        pytest.param(0.71, "a", id="above"),
        pytest.param(0.70, "a", id="equal"),
        pytest.param(0.69, FALLBACK, id="below"),
        # The supplied value is compared as-is; any rounding would select here.
        pytest.param(0.7 - 1e-9, FALLBACK, id="just-below"),
    ],
)
def test_threshold_comparison(confidence: float, expected: str) -> None:
    assert _select("a", confidence) == expected


@pytest.mark.parametrize(
    ("confidence", "min_confidence", "expected"),
    [
        pytest.param(0.0, 0.0, "a", id="zero-threshold-zero-confidence"),
        pytest.param(1.0, 1.0, "a", id="one-threshold-one-confidence"),
        pytest.param(1, 1, "a", id="ints-accepted"),
    ],
)
def test_threshold_edges(
    confidence: float, min_confidence: float, expected: str
) -> None:
    assert _select("a", confidence, min_confidence=min_confidence) == expected


# --- choice --------------------------------------------------------------------


@pytest.mark.parametrize(
    "choice",
    [
        pytest.param("zzz", id="unknown-label"),
        pytest.param(None, id="none"),
        pytest.param(["a"], id="non-string"),
    ],
)
def test_untrusted_choice_selects_fallback(choice: Any) -> None:
    """A malformed payload must never crash the router; it falls back."""
    assert _select(choice, 0.99) == FALLBACK


# --- confidence ----------------------------------------------------------------


@pytest.mark.parametrize(
    "confidence",
    [
        pytest.param(None, id="none"),
        pytest.param(math.nan, id="nan"),
        pytest.param(math.inf, id="inf"),
        pytest.param(-0.01, id="below-zero"),
        pytest.param(1.01, id="above-one"),
        pytest.param(True, id="bool"),
        pytest.param("0.9", id="numeric-string"),
    ],
)
def test_invalid_confidence_selects_fallback(confidence: Any) -> None:
    """Only int/float in [0, 1] is trusted; everything else falls back."""
    assert _select("a", confidence) == FALLBACK


# --- configuration -------------------------------------------------------------


@pytest.mark.parametrize(
    "min_confidence",
    [
        pytest.param(-0.1, id="below-zero"),
        pytest.param(1.1, id="above-one"),
        pytest.param(math.nan, id="nan"),
        pytest.param(True, id="bool"),
    ],
)
def test_invalid_threshold_raises(min_confidence: Any) -> None:
    with pytest.raises(ValueError, match="min_confidence"):
        _select("a", 0.9, min_confidence=min_confidence)


def test_fallback_not_in_routes_raises() -> None:
    with pytest.raises(ValueError, match="fallback 'nope' is not one of"):
        _select("a", 0.9, fallback="nope")


def test_single_string_routes_raises() -> None:
    """A str is a Collection[str] of characters; substring matching must not route."""
    with pytest.raises(ValueError, match="not a string"):
        _select("ria", 0.99, allowed_routes="triage", fallback="triage")


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        pytest.param({"min_confidence": 2.0}, "min_confidence", id="threshold"),
        pytest.param({"fallback": "nope"}, "fallback", id="fallback"),
        pytest.param(
            {"allowed_routes": "triage", "fallback": "triage"},
            "not a string",
            id="str-routes",
        ),
    ],
)
def test_invalid_config_raises_even_when_answer_would_fall_back(
    overrides: dict[str, Any], match: str
) -> None:
    """Misconfiguration must never be masked by the fallback path."""
    with pytest.raises(ValueError, match=match):
        _select("unknown", None, **overrides)


# --- Flow @router integration ----------------------------------------------------


def _build_flow(decision: dict[str, Any]) -> Flow[dict[str, Any]]:
    """Build a Flow whose start step yields a synthetic Jev Choice decision."""

    class TicketFlow(Flow[dict[str, Any]]):
        @start()
        def classify(self) -> dict[str, Any]:
            return decision

        @router(classify)
        def route(self, result: dict[str, Any]) -> str:
            # Same keys as JevDecisionTool's result dict (PR #8076).
            return select_route(
                result.get("choice"),
                result.get("confidence"),
                allowed_routes=("refund", "triage"),
                fallback="triage",
                min_confidence=0.8,
            )

        # Handler names must differ from the labels they listen to; Flow
        # rejects a method that listens to its own name.
        @listen("refund")
        def handle_refund(self) -> str:
            return "hit refund"

        @listen("triage")
        def handle_triage(self) -> str:
            return "hit triage"

    return TicketFlow()


@pytest.mark.parametrize(
    ("decision", "expected"),
    [
        pytest.param(
            {"success": True, "choice": "refund", "confidence": 0.88},
            "hit refund",
            id="confident-known-choice",
        ),
        pytest.param(
            {
                "success": False,
                "error": "Choice answer did not include a selected label.",
            },
            "hit triage",
            id="tool-error-payload",
        ),
    ],
)
def test_router_selects_expected_branch(
    decision: dict[str, Any], expected: str
) -> None:
    assert _build_flow(decision).kickoff() == expected
