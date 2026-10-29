"""Regression tests for GuardrailResult result/error exclusivity validation.

The exclusivity rule was implemented as a per-field validator, but pydantic v2's
``info.data`` only contains *previously validated* fields (declaration order:
success, result, error). Validating ``result`` could never see ``error``, so
``GuardrailResult(success=True, result=..., error=...)`` validated silently.
The rule now runs as a model-level validator on the complete instance.
"""

import pytest
from pydantic import ValidationError

from crewai.utilities.guardrail import GuardrailResult


def test_success_true_with_both_result_and_error_rejected() -> None:
    with pytest.raises(ValidationError, match="success is True"):
        GuardrailResult(success=True, result="answer", error="boom")


def test_success_false_with_both_result_and_error_rejected() -> None:
    with pytest.raises(ValidationError, match="success is False"):
        GuardrailResult(success=False, result="answer", error="boom")


def test_success_true_with_result_only_accepted() -> None:
    result = GuardrailResult(success=True, result="answer")
    assert result.result == "answer" and result.error is None


def test_success_false_with_error_only_accepted() -> None:
    result = GuardrailResult(success=False, error="boom")
    assert result.error == "boom" and result.result is None


def test_from_tuple_still_works_for_both_directions() -> None:
    ok = GuardrailResult.from_tuple((True, "answer"))
    assert ok.success and ok.result == "answer" and ok.error is None

    bad = GuardrailResult.from_tuple((False, "boom"))
    assert not bad.success and bad.error == "boom" and bad.result is None


def test_empty_strings_do_not_trigger_exclusivity() -> None:
    """Falsy values never triggered the rule before; keep that behavior."""
    result = GuardrailResult(success=True, result="answer", error="")
    assert result.error == ""
