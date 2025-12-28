"""Unit tests for the @human_feedback decorator.

This module tests the @human_feedback decorator's validation logic,
async support, and attribute preservation functionality.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from crewai.flow import Flow, human_feedback, listen, start
from crewai.flow.human_feedback import (
    HumanFeedbackConfig,
    HumanFeedbackResult,
)


class TestHumanFeedbackValidation:
    """Tests for decorator parameter validation."""

    def test_emit_requires_llm(self):
        """Test that specifying emit without llm raises ValueError."""
        with pytest.raises(ValueError) as exc_info:

            @human_feedback(
                message="Review this:",
                emit=["approve", "reject"],
                # llm not provided
            )
            def test_method(self):
                return "output"

        assert "llm is required" in str(exc_info.value)

    def test_default_outcome_requires_emit(self):
        """Test that specifying default_outcome without emit raises ValueError."""
        with pytest.raises(ValueError) as exc_info:

            @human_feedback(
                message="Review this:",
                default_outcome="approve",
                # emit not provided
            )
            def test_method(self):
                return "output"

        assert "requires emit" in str(exc_info.value)

    def test_default_outcome_must_be_in_emit(self):
        """Test that default_outcome must be one of the emit values."""
        with pytest.raises(ValueError) as exc_info:

            @human_feedback(
                message="Review this:",
                emit=["approve", "reject"],
                llm="gpt-4o-mini",
                default_outcome="invalid_outcome",
            )
            def test_method(self):
                return "output"

        assert "must be one of" in str(exc_info.value)

    def test_valid_configuration_with_routing(self):
        """Test that valid configuration with routing doesn't raise."""

        @human_feedback(
            message="Review this:",
            emit=["approve", "reject"],
            llm="gpt-4o-mini",
            default_outcome="reject",
        )
        def test_method(self):
            return "output"

        # Should not raise
        assert hasattr(test_method, "__human_feedback_config__")
        assert test_method.__is_router__ is True
        assert test_method.__router_paths__ == ["approve", "reject"]

    def test_valid_configuration_without_routing(self):
        """Test that valid configuration without routing doesn't raise."""

        @human_feedback(message="Review this:")
        def test_method(self):
            return "output"

        # Should not raise
        assert hasattr(test_method, "__human_feedback_config__")
        assert not hasattr(test_method, "__is_router__") or not test_method.__is_router__


class TestHumanFeedbackConfig:
    """Tests for HumanFeedbackConfig dataclass."""

    def test_config_creation(self):
        """Test HumanFeedbackConfig can be created with all parameters."""
        config = HumanFeedbackConfig(
            message="Test message",
            emit=["a", "b"],
            llm="gpt-4",
            default_outcome="a",
            metadata={"key": "value"},
        )

        assert config.message == "Test message"
        assert config.emit == ["a", "b"]
        assert config.llm == "gpt-4"
        assert config.default_outcome == "a"
        assert config.metadata == {"key": "value"}


class TestHumanFeedbackResult:
    """Tests for HumanFeedbackResult dataclass."""

    def test_result_creation(self):
        """Test HumanFeedbackResult can be created with all fields."""
        result = HumanFeedbackResult(
            output={"title": "Test"},
            feedback="Looks good",
            outcome="approved",
            method_name="test_method",
        )

        assert result.output == {"title": "Test"}
        assert result.feedback == "Looks good"
        assert result.outcome == "approved"
        assert result.method_name == "test_method"
        assert isinstance(result.timestamp, datetime)
        assert result.metadata == {}

    def test_result_with_metadata(self):
        """Test HumanFeedbackResult with custom metadata."""
        result = HumanFeedbackResult(
            output="test",
            feedback="feedback",
            metadata={"channel": "slack", "user": "test_user"},
        )

        assert result.metadata == {"channel": "slack", "user": "test_user"}


class TestDecoratorAttributePreservation:
    """Tests for preserving Flow decorator attributes."""

    def test_preserves_start_method_attributes(self):
        """Test that @human_feedback preserves @start decorator attributes."""

        class TestFlow(Flow):
            @start()
            @human_feedback(message="Review:")
            def my_start_method(self):
                return "output"

        # Check that start method attributes are preserved
        flow = TestFlow()
        method = flow._methods.get("my_start_method")
        assert method is not None
        assert hasattr(method, "__is_start_method__") or "my_start_method" in flow._start_methods

    def test_preserves_listen_method_attributes(self):
        """Test that @human_feedback preserves @listen decorator attributes."""

        class TestFlow(Flow):
            @start()
            def begin(self):
                return "start"

            @listen("begin")
            @human_feedback(message="Review:")
            def review(self):
                return "review output"

        flow = TestFlow()
        # The method should be registered as a listener
        assert "review" in flow._listeners or any(
            "review" in str(v) for v in flow._listeners.values()
        )

    def test_sets_router_attributes_when_emit_specified(self):
        """Test that router attributes are set when emit is specified."""

        # Test the decorator directly without @start wrapping
        @human_feedback(
            message="Review:",
            emit=["approved", "rejected"],
            llm="gpt-4o-mini",
        )
        def review_method(self):
            return "output"

        assert review_method.__is_router__ is True
        assert review_method.__router_paths__ == ["approved", "rejected"]


class TestAsyncSupport:
    """Tests for async method support."""

    def test_async_method_detection(self):
        """Test that async methods are properly detected and wrapped."""

        @human_feedback(message="Review:")
        async def async_method(self):
            return "async output"

        assert asyncio.iscoroutinefunction(async_method)

    def test_sync_method_remains_sync(self):
        """Test that sync methods remain synchronous."""

        @human_feedback(message="Review:")
        def sync_method(self):
            return "sync output"

        assert not asyncio.iscoroutinefunction(sync_method)


class TestHumanFeedbackExecution:
    """Tests for actual human feedback execution."""

    @patch("builtins.input", return_value="This looks great!")
    @patch("builtins.print")
    def test_basic_feedback_collection(self, mock_print, mock_input):
        """Test basic feedback collection without routing."""

        class TestFlow(Flow):
            @start()
            @human_feedback(message="Please review:")
            def generate(self):
                return "Generated content"

        flow = TestFlow()

        with patch.object(flow, "_request_human_feedback", return_value="Great job!"):
            result = flow.kickoff()

        assert flow.last_human_feedback is not None
        assert flow.last_human_feedback.output == "Generated content"
        assert flow.last_human_feedback.feedback == "Great job!"

    @patch("builtins.input", return_value="")
    @patch("builtins.print")
    def test_empty_feedback_with_default_outcome(self, mock_print, mock_input):
        """Test empty feedback uses default_outcome."""

        class TestFlow(Flow):
            @start()
            @human_feedback(
                message="Review:",
                emit=["approved", "needs_work"],
                llm="gpt-4o-mini",
                default_outcome="needs_work",
            )
            def review(self):
                return "Content"

        flow = TestFlow()

        with patch.object(flow, "_request_human_feedback", return_value=""):
            result = flow.kickoff()

        assert result == "needs_work"
        assert flow.last_human_feedback is not None
        assert flow.last_human_feedback.outcome == "needs_work"

    @patch("builtins.input", return_value="Approved!")
    @patch("builtins.print")
    def test_feedback_collapsing(self, mock_print, mock_input):
        """Test that feedback is collapsed to an outcome."""

        class TestFlow(Flow):
            @start()
            @human_feedback(
                message="Review:",
                emit=["approved", "rejected"],
                llm="gpt-4o-mini",
            )
            def review(self):
                return "Content"

        flow = TestFlow()

        with (
            patch.object(flow, "_request_human_feedback", return_value="Looks great, approved!"),
            patch.object(flow, "_collapse_to_outcome", return_value="approved"),
        ):
            result = flow.kickoff()

        assert result == "approved"
        assert flow.last_human_feedback is not None
        assert flow.last_human_feedback.outcome == "approved"


class TestHumanFeedbackHistory:
    """Tests for human feedback history tracking."""

    @patch("builtins.input", return_value="feedback")
    @patch("builtins.print")
    def test_history_accumulates(self, mock_print, mock_input):
        """Test that multiple feedbacks are stored in history."""

        class TestFlow(Flow):
            @start()
            @human_feedback(message="Review step 1:")
            def step1(self):
                return "Step 1 output"

            @listen(step1)
            @human_feedback(message="Review step 2:")
            def step2(self, prev):
                return "Step 2 output"

        flow = TestFlow()

        with patch.object(flow, "_request_human_feedback", return_value="feedback"):
            flow.kickoff()

        # Both feedbacks should be in history
        assert len(flow.human_feedback_history) == 2
        assert flow.human_feedback_history[0].method_name == "step1"
        assert flow.human_feedback_history[1].method_name == "step2"

    @patch("builtins.input", return_value="")
    @patch("builtins.print")
    def test_human_feedback_property_returns_last(self, mock_print, mock_input):
        """Test that human_feedback property returns the last result."""

        class TestFlow(Flow):
            @start()
            @human_feedback(message="Review:")
            def generate(self):
                return "output"

        flow = TestFlow()

        with patch.object(flow, "_request_human_feedback", return_value="last feedback"):
            flow.kickoff()

        assert flow.last_human_feedback is not None
        assert flow.last_human_feedback.feedback == "last feedback"
        assert flow.last_human_feedback is flow.last_human_feedback


class TestCollapseToOutcome:
    """Tests for the _collapse_to_outcome method."""

    def test_exact_match(self):
        """Test exact match returns the correct outcome."""
        flow = Flow()

        with patch("crewai.llm.LLM") as MockLLM:
            mock_llm = MagicMock()
            mock_llm.call.return_value = "approved"
            MockLLM.return_value = mock_llm

            result = flow._collapse_to_outcome(
                feedback="I approve this",
                outcomes=["approved", "rejected"],
                llm="gpt-4o-mini",
            )

        assert result == "approved"

    def test_partial_match(self):
        """Test partial match finds the outcome in the response."""
        flow = Flow()

        with patch("crewai.llm.LLM") as MockLLM:
            mock_llm = MagicMock()
            mock_llm.call.return_value = "The outcome is approved based on the feedback"
            MockLLM.return_value = mock_llm

            result = flow._collapse_to_outcome(
                feedback="Looks good",
                outcomes=["approved", "rejected"],
                llm="gpt-4o-mini",
            )

        assert result == "approved"

    def test_fallback_to_first(self):
        """Test that unmatched response falls back to first outcome."""
        flow = Flow()

        with patch("crewai.llm.LLM") as MockLLM:
            mock_llm = MagicMock()
            mock_llm.call.return_value = "something completely different"
            MockLLM.return_value = mock_llm

            result = flow._collapse_to_outcome(
                feedback="Unclear feedback",
                outcomes=["approved", "rejected"],
                llm="gpt-4o-mini",
            )

        assert result == "approved"  # First in list
