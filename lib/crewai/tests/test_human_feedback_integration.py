"""Integration tests for the @human_feedback decorator with Flow.

This module tests the integration of @human_feedback with @listen,
routing behavior, multi-step flows, and state management.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from crewai.flow import Flow, HumanFeedbackResult, human_feedback, listen, start
from crewai.flow.flow import FlowState


class TestRoutingIntegration:
    """Tests for routing integration with @listen decorators."""

    @patch("builtins.input", return_value="I approve")
    @patch("builtins.print")
    def test_routes_to_matching_listener(self, mock_print, mock_input):
        """Test that collapsed outcome routes to the matching @listen method."""
        execution_order = []

        class ReviewFlow(Flow):
            @start()
            @human_feedback(
                message="Review:",
                emit=["approved", "rejected"],
                llm="gpt-4o-mini",
            )
            def generate(self):
                execution_order.append("generate")
                return "content"

            @listen("approved")
            def on_approved(self):
                execution_order.append("on_approved")
                return "published"

            @listen("rejected")
            def on_rejected(self):
                execution_order.append("on_rejected")
                return "discarded"

        flow = ReviewFlow()

        with (
            patch.object(flow, "_request_human_feedback", return_value="Approved!"),
            patch.object(flow, "_collapse_to_outcome", return_value="approved"),
        ):
            result = flow.kickoff()

        assert "generate" in execution_order
        assert "on_approved" in execution_order
        assert "on_rejected" not in execution_order

    @patch("builtins.input", return_value="")
    @patch("builtins.print")
    def test_default_outcome_routes_correctly(self, mock_print, mock_input):
        """Test that default_outcome routes when no feedback provided."""
        executed_listener = []

        class ReviewFlow(Flow):
            @start()
            @human_feedback(
                message="Review:",
                emit=["approved", "needs_work"],
                llm="gpt-4o-mini",
                default_outcome="needs_work",
            )
            def generate(self):
                return "content"

            @listen("approved")
            def on_approved(self):
                executed_listener.append("approved")

            @listen("needs_work")
            def on_needs_work(self):
                executed_listener.append("needs_work")

        flow = ReviewFlow()

        with patch.object(flow, "_request_human_feedback", return_value=""):
            flow.kickoff()

        assert "needs_work" in executed_listener
        assert "approved" not in executed_listener


class TestMultiStepFlows:
    """Tests for multi-step flows with multiple @human_feedback decorators."""

    @patch("builtins.input", side_effect=["Good draft", "Final approved"])
    @patch("builtins.print")
    def test_multiple_feedback_steps(self, mock_print, mock_input):
        """Test a flow with multiple human feedback steps."""

        class MultiStepFlow(Flow):
            @start()
            @human_feedback(message="Review draft:")
            def draft(self):
                return "Draft content"

            @listen(draft)
            @human_feedback(message="Final review:")
            def final_review(self, prev_result: HumanFeedbackResult):
                return f"Final content based on: {prev_result.feedback}"

        flow = MultiStepFlow()

        with patch.object(
            flow, "_request_human_feedback", side_effect=["Good draft", "Approved"]
        ):
            flow.kickoff()

        # Both feedbacks should be recorded
        assert len(flow.human_feedback_history) == 2
        assert flow.human_feedback_history[0].method_name == "draft"
        assert flow.human_feedback_history[0].feedback == "Good draft"
        assert flow.human_feedback_history[1].method_name == "final_review"
        assert flow.human_feedback_history[1].feedback == "Approved"

    @patch("builtins.input", return_value="feedback")
    @patch("builtins.print")
    def test_mixed_feedback_and_regular_methods(self, mock_print, mock_input):
        """Test flow with both @human_feedback and regular methods."""
        execution_order = []

        class MixedFlow(Flow):
            @start()
            def generate(self):
                execution_order.append("generate")
                return "generated"

            @listen(generate)
            @human_feedback(message="Review:")
            def review(self):
                execution_order.append("review")
                return "reviewed"

            @listen(review)
            def finalize(self, result):
                execution_order.append("finalize")
                return "finalized"

        flow = MixedFlow()

        with patch.object(flow, "_request_human_feedback", return_value="feedback"):
            flow.kickoff()

        assert execution_order == ["generate", "review", "finalize"]


class TestStateManagement:
    """Tests for state management with human feedback."""

    @patch("builtins.input", return_value="approved")
    @patch("builtins.print")
    def test_feedback_available_in_listener(self, mock_print, mock_input):
        """Test that feedback is accessible in downstream listeners."""
        captured_feedback = []

        class StateFlow(Flow):
            @start()
            @human_feedback(
                message="Review:",
                emit=["approved", "rejected"],
                llm="gpt-4o-mini",
            )
            def review(self):
                return "Content to review"

            @listen("approved")
            def on_approved(self):
                # Access the feedback via property
                captured_feedback.append(self.last_human_feedback)
                return "done"

        flow = StateFlow()

        with (
            patch.object(flow, "_request_human_feedback", return_value="Great content!"),
            patch.object(flow, "_collapse_to_outcome", return_value="approved"),
        ):
            flow.kickoff()

        assert len(captured_feedback) == 1
        result = captured_feedback[0]
        assert isinstance(result, HumanFeedbackResult)
        assert result.output == "Content to review"
        assert result.feedback == "Great content!"
        assert result.outcome == "approved"

    @patch("builtins.input", return_value="")
    @patch("builtins.print")
    def test_history_preserved_across_steps(self, mock_print, mock_input):
        """Test that feedback history is preserved across flow execution."""

        class HistoryFlow(Flow):
            @start()
            @human_feedback(message="Step 1:")
            def step1(self):
                return "Step 1"

            @listen(step1)
            @human_feedback(message="Step 2:")
            def step2(self, result):
                return "Step 2"

            @listen(step2)
            def final(self, result):
                # Access history
                return len(self.human_feedback_history)

        flow = HistoryFlow()

        with patch.object(flow, "_request_human_feedback", return_value="feedback"):
            result = flow.kickoff()

        # Final method should see 2 feedback entries
        assert result == 2


class TestAsyncFlowIntegration:
    """Tests for async flow integration."""

    @pytest.mark.asyncio
    async def test_async_flow_with_human_feedback(self):
        """Test that @human_feedback works with async flows."""
        executed = []

        class AsyncFlow(Flow):
            @start()
            @human_feedback(message="Review:")
            async def async_review(self):
                executed.append("async_review")
                await asyncio.sleep(0.01)  # Simulate async work
                return "async content"

        flow = AsyncFlow()

        with patch.object(flow, "_request_human_feedback", return_value="feedback"):
            await flow.kickoff_async()

        assert "async_review" in executed
        assert flow.last_human_feedback is not None
        assert flow.last_human_feedback.output == "async content"


class TestWithStructuredState:
    """Tests for flows with structured (Pydantic) state."""

    @patch("builtins.input", return_value="approved")
    @patch("builtins.print")
    def test_with_pydantic_state(self, mock_print, mock_input):
        """Test human feedback with structured Pydantic state."""

        class ReviewState(FlowState):
            content: str = ""
            review_count: int = 0

        class StructuredFlow(Flow[ReviewState]):
            initial_state = ReviewState

            @start()
            @human_feedback(
                message="Review:",
                emit=["approved", "rejected"],
                llm="gpt-4o-mini",
            )
            def review(self):
                self.state.content = "Generated content"
                self.state.review_count += 1
                return self.state.content

            @listen("approved")
            def on_approved(self):
                return f"Approved: {self.state.content}"

        flow = StructuredFlow()

        with (
            patch.object(flow, "_request_human_feedback", return_value="LGTM"),
            patch.object(flow, "_collapse_to_outcome", return_value="approved"),
        ):
            result = flow.kickoff()

        assert flow.state.review_count == 1
        assert flow.last_human_feedback is not None
        assert flow.last_human_feedback.feedback == "LGTM"


class TestMetadataPassthrough:
    """Tests for metadata passthrough functionality."""

    @patch("builtins.input", return_value="")
    @patch("builtins.print")
    def test_metadata_included_in_result(self, mock_print, mock_input):
        """Test that metadata is passed through to HumanFeedbackResult."""

        class MetadataFlow(Flow):
            @start()
            @human_feedback(
                message="Review:",
                metadata={"channel": "slack", "priority": "high"},
            )
            def review(self):
                return "content"

        flow = MetadataFlow()

        with patch.object(flow, "_request_human_feedback", return_value="feedback"):
            flow.kickoff()

        result = flow.last_human_feedback
        assert result is not None
        assert result.metadata == {"channel": "slack", "priority": "high"}


class TestEventEmission:
    """Tests for event emission during human feedback."""

    @patch("builtins.input", return_value="test feedback")
    @patch("builtins.print")
    def test_events_emitted_on_feedback_request(self, mock_print, mock_input):
        """Test that events are emitted when feedback is requested."""
        from crewai.events.event_listener import event_listener

        class EventFlow(Flow):
            @start()
            @human_feedback(message="Review:")
            def review(self):
                return "content"

        flow = EventFlow()

        # We can't easily capture events in tests, but we can verify
        # the flow executes without errors
        with (
            patch.object(
                event_listener.formatter, "pause_live_updates", return_value=None
            ),
            patch.object(
                event_listener.formatter, "resume_live_updates", return_value=None
            ),
        ):
            flow.kickoff()

        assert flow.last_human_feedback is not None


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @patch("builtins.input", return_value="")
    @patch("builtins.print")
    def test_empty_feedback_first_outcome_fallback(self, mock_print, mock_input):
        """Test that empty feedback without default uses first outcome."""

        class FallbackFlow(Flow):
            @start()
            @human_feedback(
                message="Review:",
                emit=["first", "second", "third"],
                llm="gpt-4o-mini",
                # No default_outcome specified
            )
            def review(self):
                return "content"

        flow = FallbackFlow()

        with patch.object(flow, "_request_human_feedback", return_value=""):
            result = flow.kickoff()

        assert result == "first"  # Falls back to first outcome

    @patch("builtins.input", return_value="whitespace only   ")
    @patch("builtins.print")
    def test_whitespace_only_feedback_treated_as_empty(self, mock_print, mock_input):
        """Test that whitespace-only feedback is treated as empty."""

        class WhitespaceFlow(Flow):
            @start()
            @human_feedback(
                message="Review:",
                emit=["approve", "reject"],
                llm="gpt-4o-mini",
                default_outcome="reject",
            )
            def review(self):
                return "content"

        flow = WhitespaceFlow()

        with patch.object(flow, "_request_human_feedback", return_value="   "):
            result = flow.kickoff()

        assert result == "reject"  # Uses default because feedback is empty after strip

    @patch("builtins.input", return_value="feedback")
    @patch("builtins.print")
    def test_feedback_result_without_routing(self, mock_print, mock_input):
        """Test that HumanFeedbackResult is returned when not routing."""

        class NoRoutingFlow(Flow):
            @start()
            @human_feedback(message="Review:")
            def review(self):
                return "content"

        flow = NoRoutingFlow()

        with patch.object(flow, "_request_human_feedback", return_value="feedback"):
            result = flow.kickoff()

        # Result should be HumanFeedbackResult when not routing
        assert isinstance(result, HumanFeedbackResult)
        assert result.output == "content"
        assert result.feedback == "feedback"
        assert result.outcome is None  # No routing, no outcome
