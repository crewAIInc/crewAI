"""Tests for LLMGuardrail error handling fix (issue #7150)."""

from unittest.mock import patch, MagicMock

import pytest

from crewai.tasks.llm_guardrail import LLMGuardrail, LLMGuardrailResult
from crewai.tasks.task_output import TaskOutput
from crewai.utilities.guardrail import GuardrailResult


class TestLLMGuardrailErrorDistinction:
    """LLM errors should be distinguishable from validation failures."""

    def test_llm_error_returns_errored_flag(self):
        """When _validate_output raises, the result should have errored=True."""
        g = LLMGuardrail(description="must be under 100 words", llm=None)
        out = TaskOutput(description="d", agent="a", raw="the agent's answer")

        with patch.object(
            LLMGuardrail, "_validate_output",
            side_effect=RuntimeError("litellm.APIConnectionError: provider unavailable"),
        ):
            result = g(out)

        # Should be a 3-tuple with errored=True
        assert len(result) == 3
        assert result[0] is False
        assert "Error while validating" in result[1]
        assert result[2] is True

    def test_validation_failure_returns_errored_false(self):
        """When guardrail rejects output, errored should be False."""
        g = LLMGuardrail(description="must be under 100 words", llm=None)
        out = TaskOutput(description="d", agent="a", raw="the agent's answer")

        mock_output = MagicMock()
        mock_output.pydantic = LLMGuardrailResult(valid=False, feedback="too long")
        with patch.object(LLMGuardrail, "_validate_output", return_value=mock_output):
            result = g(out)

        # Should be a 2-tuple (normal validation failure)
        assert len(result) == 2
        assert result[0] is False
        assert result[1] == "too long"

    def test_passing_validation_returns_true(self):
        """When guardrail passes, should return (True, raw)."""
        g = LLMGuardrail(description="must be under 100 words", llm=None)
        out = TaskOutput(description="d", agent="a", raw="short answer")

        mock_output = MagicMock()
        mock_output.pydantic = LLMGuardrailResult(valid=True, feedback=None)
        with patch.object(LLMGuardrail, "_validate_output", return_value=mock_output):
            result = g(out)

        assert result[0] is True
        assert result[1] == "short answer"


class TestGuardrailResultFromTuple:
    """GuardrailResult.from_tuple should handle 2-tuple and 3-tuple."""

    def test_from_2_tuple(self):
        result = GuardrailResult.from_tuple((False, "error msg"))
        assert result.success is False
        assert result.error == "error msg"
        assert result.errored is False

    def test_from_3_tuple_errored(self):
        result = GuardrailResult.from_tuple((False, "provider down", True))
        assert result.success is False
        assert result.error == "provider down"
        assert result.errored is True

    def test_from_2_tuple_success(self):
        result = GuardrailResult.from_tuple((True, "looks good"))
        assert result.success is True
        assert result.result == "looks good"
        assert result.errored is False
