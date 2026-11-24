"""Tests for matching human-feedback text to emit outcomes."""

from __future__ import annotations

import pytest

from crewai.flow.human_feedback import (
    HumanFeedbackCollapseError,
    _match_collapse_outcome,
    _require_collapse_outcome,
)


def test_exact_match_wins() -> None:
    assert _match_collapse_outcome("approved", ["approved", "rejected"]) == "approved"


def test_partial_match_inside_a_sentence() -> None:
    text = "The outcome is approved based on the feedback"
    assert _match_collapse_outcome(text, ["approved", "rejected"]) == "approved"


def test_outcome_must_be_a_whole_word() -> None:
    """A short outcome inside another word is not a match."""
    assert _match_collapse_outcome("I do not know", ["no", "yes"]) is None
    assert _match_collapse_outcome("The change was disapproved", ["approved"]) is None


def test_negated_outcome_does_not_match() -> None:
    """A negated label must not route to that label."""
    assert _match_collapse_outcome("not approved", ["approved", "rejected"]) is None
    assert (
        _match_collapse_outcome("This isn't approved", ["approved", "rejected"])
        is None
    )
    assert (
        _match_collapse_outcome("I cannot approve this", ["approve", "reject"])
        is None
    )


def test_negated_label_does_not_hide_a_later_positive_label() -> None:
    text = "The output was not approved; it was rejected."
    assert _match_collapse_outcome(text, ["approved", "rejected"]) == "rejected"


def test_negation_in_an_earlier_clause_does_not_affect_the_match() -> None:
    assert _match_collapse_outcome("Not good. Approved.", ["approved", "rejected"]) == (
        "approved"
    )


def test_require_collapse_outcome_fails_closed_for_negated_text() -> None:
    with pytest.raises(HumanFeedbackCollapseError, match="Could not match"):
        _require_collapse_outcome("not approved", ["approved", "rejected"])
