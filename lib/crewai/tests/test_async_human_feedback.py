"""Tests for async human feedback functionality.

This module tests the async/non-blocking human feedback flow, including:
- PendingFeedbackContext creation and serialization
- HumanFeedbackPending exception handling
- HumanFeedbackProvider protocol
- ConsoleProvider
- Flow.from_pending() and Flow.resume()
- SQLite persistence with pending feedback
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from crewai.flow import Flow, start, listen, human_feedback
from crewai.flow.async_feedback import (
    ConsoleProvider,
    HumanFeedbackPending,
    HumanFeedbackProvider,
    PendingFeedbackContext,
)
from crewai.flow.persistence import SQLiteFlowPersistence


# =============================================================================
# PendingFeedbackContext Tests
# =============================================================================


class TestPendingFeedbackContext:
    """Tests for PendingFeedbackContext dataclass."""

    def test_create_basic_context(self) -> None:
        """Test creating a basic pending feedback context."""
        context = PendingFeedbackContext(
            flow_id="test-flow-123",
            flow_class="myapp.flows.ReviewFlow",
            method_name="review_content",
            method_output="Content to review",
            message="Please review this content:",
        )

        assert context.flow_id == "test-flow-123"
        assert context.flow_class == "myapp.flows.ReviewFlow"
        assert context.method_name == "review_content"
        assert context.method_output == "Content to review"
        assert context.message == "Please review this content:"
        assert context.emit is None
        assert context.default_outcome is None
        assert context.metadata == {}
        assert isinstance(context.requested_at, datetime)

    def test_create_context_with_emit(self) -> None:
        """Test creating context with routing outcomes."""
        context = PendingFeedbackContext(
            flow_id="test-flow-456",
            flow_class="myapp.flows.ApprovalFlow",
            method_name="submit_for_approval",
            method_output={"document": "content"},
            message="Approve or reject:",
            emit=["approved", "rejected", "needs_revision"],
            default_outcome="needs_revision",
            llm="gpt-4o-mini",
        )

        assert context.emit == ["approved", "rejected", "needs_revision"]
        assert context.default_outcome == "needs_revision"
        assert context.llm == "gpt-4o-mini"

    def test_to_dict_serialization(self) -> None:
        """Test serializing context to dictionary."""
        context = PendingFeedbackContext(
            flow_id="test-flow-789",
            flow_class="myapp.flows.TestFlow",
            method_name="test_method",
            method_output={"key": "value"},
            message="Test message",
            emit=["yes", "no"],
            metadata={"channel": "#reviews"},
        )

        result = context.to_dict()

        assert result["flow_id"] == "test-flow-789"
        assert result["flow_class"] == "myapp.flows.TestFlow"
        assert result["method_name"] == "test_method"
        assert result["method_output"] == {"key": "value"}
        assert result["message"] == "Test message"
        assert result["emit"] == ["yes", "no"]
        assert result["metadata"] == {"channel": "#reviews"}
        assert "requested_at" in result

    def test_from_dict_deserialization(self) -> None:
        """Test deserializing context from dictionary."""
        data = {
            "flow_id": "test-flow-abc",
            "flow_class": "myapp.flows.TestFlow",
            "method_name": "my_method",
            "method_output": "output value",
            "message": "Feedback message",
            "emit": ["option_a", "option_b"],
            "default_outcome": "option_a",
            "metadata": {"user_id": "123"},
            "llm": "gpt-4o-mini",
            "requested_at": "2024-01-15T10:30:00",
        }

        context = PendingFeedbackContext.from_dict(data)

        assert context.flow_id == "test-flow-abc"
        assert context.flow_class == "myapp.flows.TestFlow"
        assert context.method_name == "my_method"
        assert context.emit == ["option_a", "option_b"]
        assert context.default_outcome == "option_a"
        assert context.llm == "gpt-4o-mini"

    def test_roundtrip_serialization(self) -> None:
        """Test that to_dict/from_dict roundtrips correctly."""
        original = PendingFeedbackContext(
            flow_id="roundtrip-test",
            flow_class="test.TestFlow",
            method_name="test",
            method_output={"nested": {"data": [1, 2, 3]}},
            message="Test",
            emit=["a", "b"],
            metadata={"key": "value"},
        )

        serialized = original.to_dict()
        restored = PendingFeedbackContext.from_dict(serialized)

        assert restored.flow_id == original.flow_id
        assert restored.flow_class == original.flow_class
        assert restored.method_name == original.method_name
        assert restored.method_output == original.method_output
        assert restored.emit == original.emit
        assert restored.metadata == original.metadata


# =============================================================================
# HumanFeedbackPending Exception Tests
# =============================================================================


class TestHumanFeedbackPending:
    """Tests for HumanFeedbackPending exception."""

    def test_basic_exception(self) -> None:
        """Test creating basic pending exception."""
        context = PendingFeedbackContext(
            flow_id="exc-test",
            flow_class="test.Flow",
            method_name="method",
            method_output="output",
            message="message",
        )

        exc = HumanFeedbackPending(context=context)

        assert exc.context == context
        assert exc.callback_info == {}
        assert "exc-test" in str(exc)
        assert "method" in str(exc)

    def test_exception_with_callback_info(self) -> None:
        """Test pending exception with callback information."""
        context = PendingFeedbackContext(
            flow_id="callback-test",
            flow_class="test.Flow",
            method_name="method",
            method_output="output",
            message="message",
        )

        exc = HumanFeedbackPending(
            context=context,
            callback_info={
                "webhook_url": "https://example.com/webhook",
                "slack_thread": "123456",
            },
        )

        assert exc.callback_info["webhook_url"] == "https://example.com/webhook"
        assert exc.callback_info["slack_thread"] == "123456"

    def test_exception_with_custom_message(self) -> None:
        """Test pending exception with custom message."""
        context = PendingFeedbackContext(
            flow_id="msg-test",
            flow_class="test.Flow",
            method_name="method",
            method_output="output",
            message="message",
        )

        exc = HumanFeedbackPending(
            context=context,
            message="Custom pending message",
        )

        assert str(exc) == "Custom pending message"

    def test_exception_is_catchable(self) -> None:
        """Test that exception can be caught and handled."""
        context = PendingFeedbackContext(
            flow_id="catch-test",
            flow_class="test.Flow",
            method_name="method",
            method_output="output",
            message="message",
        )

        with pytest.raises(HumanFeedbackPending) as exc_info:
            raise HumanFeedbackPending(context=context)

        assert exc_info.value.context.flow_id == "catch-test"


# =============================================================================
# HumanFeedbackProvider Protocol Tests
# =============================================================================


class TestHumanFeedbackProvider:
    """Tests for HumanFeedbackProvider protocol."""

    def test_protocol_compliance_sync_provider(self) -> None:
        """Test that sync provider complies with protocol."""

        class SyncProvider:
            def request_feedback(
                self, context: PendingFeedbackContext, flow: Flow
            ) -> str:
                return "sync feedback"

        provider = SyncProvider()
        assert isinstance(provider, HumanFeedbackProvider)

    def test_protocol_compliance_async_provider(self) -> None:
        """Test that async provider complies with protocol."""

        class AsyncProvider:
            def request_feedback(
                self, context: PendingFeedbackContext, flow: Flow
            ) -> str:
                raise HumanFeedbackPending(context=context)

        provider = AsyncProvider()
        assert isinstance(provider, HumanFeedbackProvider)


# =============================================================================
# ConsoleProvider Tests
# =============================================================================


class TestConsoleProvider:
    """Tests for ConsoleProvider."""

    def test_provider_initialization(self) -> None:
        """Test console provider initialization."""
        provider = ConsoleProvider()
        assert provider.verbose is True

        quiet_provider = ConsoleProvider(verbose=False)
        assert quiet_provider.verbose is False



# =============================================================================
# SQLite Persistence Tests for Async Feedback
# =============================================================================


class TestSQLitePendingFeedback:
    """Tests for SQLite persistence with pending feedback."""

    def test_save_and_load_pending_feedback(self) -> None:
        """Test saving and loading pending feedback context."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_flows.db")
            persistence = SQLiteFlowPersistence(db_path)

            context = PendingFeedbackContext(
                flow_id="persist-test-123",
                flow_class="test.TestFlow",
                method_name="review",
                method_output={"data": "test"},
                message="Review this:",
                emit=["approved", "rejected"],
                llm="gpt-4o-mini",
            )

            state_data = {"counter": 10, "items": ["a", "b"]}

            # Save pending feedback
            persistence.save_pending_feedback(
                flow_uuid="persist-test-123",
                context=context,
                state_data=state_data,
            )

            # Load pending feedback
            result = persistence.load_pending_feedback("persist-test-123")

            assert result is not None
            loaded_state, loaded_context = result
            assert loaded_state["counter"] == 10
            assert loaded_state["items"] == ["a", "b"]
            assert loaded_context.flow_id == "persist-test-123"
            assert loaded_context.emit == ["approved", "rejected"]

    def test_load_nonexistent_pending_feedback(self) -> None:
        """Test loading pending feedback that doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_flows.db")
            persistence = SQLiteFlowPersistence(db_path)

            result = persistence.load_pending_feedback("nonexistent-id")
            assert result is None

    def test_clear_pending_feedback(self) -> None:
        """Test clearing pending feedback after resume."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_flows.db")
            persistence = SQLiteFlowPersistence(db_path)

            context = PendingFeedbackContext(
                flow_id="clear-test",
                flow_class="test.Flow",
                method_name="method",
                method_output="output",
                message="message",
            )

            persistence.save_pending_feedback(
                flow_uuid="clear-test",
                context=context,
                state_data={"key": "value"},
            )

            # Verify it exists
            assert persistence.load_pending_feedback("clear-test") is not None

            # Clear it
            persistence.clear_pending_feedback("clear-test")

            # Verify it's gone
            assert persistence.load_pending_feedback("clear-test") is None

    def test_replace_existing_pending_feedback(self) -> None:
        """Test that saving pending feedback replaces existing entry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_flows.db")
            persistence = SQLiteFlowPersistence(db_path)

            flow_id = "replace-test"

            # Save first version
            context1 = PendingFeedbackContext(
                flow_id=flow_id,
                flow_class="test.Flow",
                method_name="method1",
                method_output="output1",
                message="message1",
            )
            persistence.save_pending_feedback(
                flow_uuid=flow_id,
                context=context1,
                state_data={"version": 1},
            )

            # Save second version (should replace)
            context2 = PendingFeedbackContext(
                flow_id=flow_id,
                flow_class="test.Flow",
                method_name="method2",
                method_output="output2",
                message="message2",
            )
            persistence.save_pending_feedback(
                flow_uuid=flow_id,
                context=context2,
                state_data={"version": 2},
            )

            # Load and verify it's the second version
            result = persistence.load_pending_feedback(flow_id)
            assert result is not None
            state, context = result
            assert state["version"] == 2
            assert context.method_name == "method2"


# =============================================================================
# Custom Async Provider Tests
# =============================================================================


class TestCustomAsyncProvider:
    """Tests for custom async providers."""

    def test_provider_raises_pending_exception(self) -> None:
        """Test that async provider raises HumanFeedbackPending."""

        class WebhookProvider:
            def __init__(self, webhook_url: str):
                self.webhook_url = webhook_url

            def request_feedback(
                self, context: PendingFeedbackContext, flow: Flow
            ) -> str:
                raise HumanFeedbackPending(
                    context=context,
                    callback_info={"url": f"{self.webhook_url}/{context.flow_id}"},
                )

        provider = WebhookProvider("https://example.com/api")
        context = PendingFeedbackContext(
            flow_id="webhook-test",
            flow_class="test.Flow",
            method_name="method",
            method_output="output",
            message="message",
        )
        mock_flow = MagicMock()

        with pytest.raises(HumanFeedbackPending) as exc_info:
            provider.request_feedback(context, mock_flow)

        assert exc_info.value.callback_info["url"] == (
            "https://example.com/api/webhook-test"
        )


# =============================================================================
# Flow.from_pending and resume Tests
# =============================================================================


class TestFlowResumeWithFeedback:
    """Tests for Flow.from_pending and resume."""

    def test_from_pending_uses_default_persistence(self) -> None:
        """Test that from_pending uses SQLiteFlowPersistence by default."""

        class TestFlow(Flow):
            @start()
            def begin(self):
                return "started"

        # When no persistence is provided, it uses default SQLiteFlowPersistence
        # This will raise "No pending feedback found" (not a persistence error)
        with pytest.raises(ValueError, match="No pending feedback found"):
            TestFlow.from_pending("nonexistent-id")

    def test_from_pending_raises_for_missing_flow(self) -> None:
        """Test that from_pending raises error for nonexistent flow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_flows.db")
            persistence = SQLiteFlowPersistence(db_path)

            class TestFlow(Flow):
                @start()
                def begin(self):
                    return "started"

            with pytest.raises(ValueError, match="No pending feedback found"):
                TestFlow.from_pending("nonexistent-id", persistence)

    def test_from_pending_restores_state(self) -> None:
        """Test that from_pending correctly restores flow state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_flows.db")
            persistence = SQLiteFlowPersistence(db_path)

            class TestState(BaseModel):
                id: str = "test-restore-123"
                counter: int = 0

            class TestFlow(Flow[TestState]):
                @start()
                def begin(self):
                    return "started"

            # Manually save pending feedback
            context = PendingFeedbackContext(
                flow_id="test-restore-123",
                flow_class="test.TestFlow",
                method_name="review",
                method_output="content",
                message="Review:",
            )
            persistence.save_pending_feedback(
                flow_uuid="test-restore-123",
                context=context,
                state_data={"id": "test-restore-123", "counter": 42},
            )

            # Restore flow
            flow = TestFlow.from_pending("test-restore-123", persistence)

            assert flow._pending_feedback_context is not None
            assert flow._pending_feedback_context.flow_id == "test-restore-123"
            assert flow._is_execution_resuming is True
            assert flow.state.counter == 42

    def test_resume_without_pending_raises_error(self) -> None:
        """Test that resume raises error without pending context."""

        class TestFlow(Flow):
            @start()
            def begin(self):
                return "started"

        flow = TestFlow()

        with pytest.raises(ValueError, match="No pending feedback context"):
            flow.resume("some feedback")

    def test_resume_from_async_context_raises_error(self) -> None:
        """Test that resume() raises RuntimeError when called from async context."""
        import asyncio

        class TestFlow(Flow):
            @start()
            def begin(self):
                return "started"

        async def call_resume_from_async():
            with tempfile.TemporaryDirectory() as tmpdir:
                db_path = os.path.join(tmpdir, "test.db")
                persistence = SQLiteFlowPersistence(db_path)

                # Save pending feedback
                context = PendingFeedbackContext(
                    flow_id="async-context-test",
                    flow_class="TestFlow",
                    method_name="begin",
                    method_output="output",
                    message="Review:",
                )
                persistence.save_pending_feedback(
                    flow_uuid="async-context-test",
                    context=context,
                    state_data={"id": "async-context-test"},
                )

                flow = TestFlow.from_pending("async-context-test", persistence)

                # This should raise RuntimeError because we're in an async context
                with pytest.raises(RuntimeError, match="cannot be called from within an async context"):
                    flow.resume("feedback")

        asyncio.run(call_resume_from_async())

    @pytest.mark.asyncio
    async def test_resume_async_direct(self) -> None:
        """Test resume_async() can be called directly in async context."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            persistence = SQLiteFlowPersistence(db_path)

            class TestFlow(Flow):
                @start()
                @human_feedback(message="Review:")
                def generate(self):
                    return "content"

                @listen(generate)
                def process(self, result):
                    return f"processed: {result.feedback}"

            # Save pending feedback
            context = PendingFeedbackContext(
                flow_id="async-direct-test",
                flow_class="TestFlow",
                method_name="generate",
                method_output="content",
                message="Review:",
            )
            persistence.save_pending_feedback(
                flow_uuid="async-direct-test",
                context=context,
                state_data={"id": "async-direct-test"},
            )

            flow = TestFlow.from_pending("async-direct-test", persistence)

            with patch("crewai.flow.flow.crewai_event_bus.emit"):
                result = await flow.resume_async("async feedback")

            assert flow.last_human_feedback is not None
            assert flow.last_human_feedback.feedback == "async feedback"

    @patch("crewai.flow.flow.crewai_event_bus.emit")
    def test_resume_basic(self, mock_emit: MagicMock) -> None:
        """Test basic resume functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_flows.db")
            persistence = SQLiteFlowPersistence(db_path)

            class TestFlow(Flow):
                @start()
                @human_feedback(message="Review this:")
                def generate(self):
                    return "generated content"

                @listen(generate)
                def process(self, feedback_result):
                    return f"Processed: {feedback_result.feedback}"

            # Manually save pending feedback (simulating async pause)
            context = PendingFeedbackContext(
                flow_id="resume-test-123",
                flow_class="test.TestFlow",
                method_name="generate",
                method_output="generated content",
                message="Review this:",
            )
            persistence.save_pending_feedback(
                flow_uuid="resume-test-123",
                context=context,
                state_data={"id": "resume-test-123"},
            )

            # Restore and resume
            flow = TestFlow.from_pending("resume-test-123", persistence)
            result = flow.resume("looks good!")

            # Verify feedback was processed
            assert flow.last_human_feedback is not None
            assert flow.last_human_feedback.feedback == "looks good!"
            assert flow.last_human_feedback.output == "generated content"

            # Verify pending feedback was cleared
            assert persistence.load_pending_feedback("resume-test-123") is None

    @patch("crewai.flow.flow.crewai_event_bus.emit")
    def test_resume_routing(self, mock_emit: MagicMock) -> None:
        """Test resume with routing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_flows.db")
            persistence = SQLiteFlowPersistence(db_path)

            class TestFlow(Flow):
                result_path: str = ""

                @start()
                @human_feedback(
                    message="Approve?",
                    emit=["approved", "rejected"],
                    llm="gpt-4o-mini",
                )
                def review(self):
                    return "content"

                @listen("approved")
                def handle_approved(self):
                    self.result_path = "approved"
                    return "Approved!"

                @listen("rejected")
                def handle_rejected(self):
                    self.result_path = "rejected"
                    return "Rejected!"

            # Save pending feedback
            context = PendingFeedbackContext(
                flow_id="route-test-123",
                flow_class="test.TestFlow",
                method_name="review",
                method_output="content",
                message="Approve?",
                emit=["approved", "rejected"],
                llm="gpt-4o-mini",
            )
            persistence.save_pending_feedback(
                flow_uuid="route-test-123",
                context=context,
                state_data={"id": "route-test-123"},
            )

            # Restore and resume - mock _collapse_to_outcome directly
            flow = TestFlow.from_pending("route-test-123", persistence)

            with patch.object(flow, "_collapse_to_outcome", return_value="approved"):
                result = flow.resume("yes, this looks great")

            # Verify routing worked
            assert flow.last_human_feedback.outcome == "approved"
            assert flow.result_path == "approved"


# =============================================================================
# Integration Tests with @human_feedback decorator
# =============================================================================


class TestAsyncHumanFeedbackIntegration:
    """Integration tests for async human feedback with decorator."""

    def test_decorator_with_provider_parameter(self) -> None:
        """Test that decorator accepts provider parameter."""

        class MockProvider:
            def request_feedback(
                self, context: PendingFeedbackContext, flow: Flow
            ) -> str:
                raise HumanFeedbackPending(context=context)

        # This should not raise
        class TestFlow(Flow):
            @start()
            @human_feedback(
                message="Review:",
                provider=MockProvider(),
            )
            def review(self):
                return "content"

        flow = TestFlow()
        # Verify the method has the provider config
        method = getattr(flow, "review")
        assert hasattr(method, "__human_feedback_config__")
        assert method.__human_feedback_config__.provider is not None

    @patch("crewai.flow.flow.crewai_event_bus.emit")
    def test_async_provider_pauses_flow(self, mock_emit: MagicMock) -> None:
        """Test that async provider pauses flow execution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_flows.db")
            persistence = SQLiteFlowPersistence(db_path)

            class PausingProvider:
                def __init__(self, persistence: SQLiteFlowPersistence):
                    self.persistence = persistence

                def request_feedback(
                    self, context: PendingFeedbackContext, flow: Flow
                ) -> str:
                    # Save pending state
                    self.persistence.save_pending_feedback(
                        flow_uuid=context.flow_id,
                        context=context,
                        state_data=flow.state if isinstance(flow.state, dict) else flow.state.model_dump(),
                    )
                    raise HumanFeedbackPending(
                        context=context,
                        callback_info={"saved": True},
                    )

            class TestFlow(Flow):
                @start()
                @human_feedback(
                    message="Review:",
                    provider=PausingProvider(persistence),
                )
                def generate(self):
                    return "generated content"

            flow = TestFlow(persistence=persistence)

            # kickoff now returns HumanFeedbackPending instead of raising it
            result = flow.kickoff()

            assert isinstance(result, HumanFeedbackPending)
            assert result.callback_info["saved"] is True

            # Get flow ID from the returned pending context
            flow_id = result.context.flow_id

            # Verify state was persisted
            persisted = persistence.load_pending_feedback(flow_id)
            assert persisted is not None

    @patch("crewai.flow.flow.crewai_event_bus.emit")
    def test_full_async_flow_cycle(self, mock_emit: MagicMock) -> None:
        """Test complete async flow: start -> pause -> resume."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_flows.db")
            persistence = SQLiteFlowPersistence(db_path)

            flow_id_holder: list[str] = []

            class SaveAndPauseProvider:
                def __init__(self, persistence: SQLiteFlowPersistence):
                    self.persistence = persistence

                def request_feedback(
                    self, context: PendingFeedbackContext, flow: Flow
                ) -> str:
                    flow_id_holder.append(context.flow_id)
                    self.persistence.save_pending_feedback(
                        flow_uuid=context.flow_id,
                        context=context,
                        state_data=flow.state if isinstance(flow.state, dict) else flow.state.model_dump(),
                    )
                    raise HumanFeedbackPending(context=context)

            class ReviewFlow(Flow):
                processed_feedback: str = ""

                @start()
                @human_feedback(
                    message="Review this content:",
                    provider=SaveAndPauseProvider(persistence),
                )
                def generate(self):
                    return "AI generated content"

                @listen(generate)
                def process(self, feedback_result):
                    self.processed_feedback = feedback_result.feedback
                    return f"Final: {feedback_result.feedback}"

            # Phase 1: Start flow (should pause)
            flow1 = ReviewFlow(persistence=persistence)
            result = flow1.kickoff()

            # kickoff now returns HumanFeedbackPending instead of raising it
            assert isinstance(result, HumanFeedbackPending)
            assert len(flow_id_holder) == 1
            paused_flow_id = flow_id_holder[0]

            # Phase 2: Resume flow
            flow2 = ReviewFlow.from_pending(paused_flow_id, persistence)
            result = flow2.resume("This is my feedback")

            # Verify feedback was processed
            assert flow2.last_human_feedback.feedback == "This is my feedback"
            assert flow2.processed_feedback == "This is my feedback"


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestAutoPersistence:
    """Tests for automatic persistence when no persistence is provided."""

    @patch("crewai.flow.flow.crewai_event_bus.emit")
    def test_auto_persistence_when_none_provided(self, mock_emit: MagicMock) -> None:
        """Test that persistence is auto-created when HumanFeedbackPending is raised."""

        class PausingProvider:
            def request_feedback(
                self, context: PendingFeedbackContext, flow: Flow
            ) -> str:
                raise HumanFeedbackPending(
                    context=context,
                    callback_info={"paused": True},
                )

        class TestFlow(Flow):
            @start()
            @human_feedback(
                message="Review:",
                provider=PausingProvider(),
            )
            def generate(self):
                return "content"

        # Create flow WITHOUT persistence
        flow = TestFlow()
        assert flow.persistence is None  # No persistence initially

        # kickoff should auto-create persistence when HumanFeedbackPending is raised
        result = flow.kickoff()

        # Should return HumanFeedbackPending (not raise it)
        assert isinstance(result, HumanFeedbackPending)

        # Persistence should have been auto-created
        assert flow.persistence is not None

        # The pending feedback should be saved
        flow_id = result.context.flow_id
        loaded = flow.persistence.load_pending_feedback(flow_id)
        assert loaded is not None


class TestCollapseToOutcomeJsonParsing:
    """Tests for _collapse_to_outcome JSON parsing edge cases."""

    def test_json_string_response_is_parsed(self) -> None:
        """Test that JSON string response from LLM is correctly parsed."""
        flow = Flow()

        with patch("crewai.llm.LLM") as MockLLM:
            mock_llm = MagicMock()
            # Simulate LLM returning JSON string (the bug we fixed)
            mock_llm.call.return_value = '{"outcome": "approved"}'
            MockLLM.return_value = mock_llm

            result = flow._collapse_to_outcome(
                feedback="I approve this",
                outcomes=["approved", "rejected"],
                llm="gpt-4o-mini",
            )

        assert result == "approved"

    def test_plain_string_response_is_matched(self) -> None:
        """Test that plain string response is correctly matched."""
        flow = Flow()

        with patch("crewai.llm.LLM") as MockLLM:
            mock_llm = MagicMock()
            # Simulate LLM returning plain outcome string
            mock_llm.call.return_value = "rejected"
            MockLLM.return_value = mock_llm

            result = flow._collapse_to_outcome(
                feedback="This is not good",
                outcomes=["approved", "rejected"],
                llm="gpt-4o-mini",
            )

        assert result == "rejected"

    def test_invalid_json_falls_back_to_matching(self) -> None:
        """Test that invalid JSON falls back to string matching."""
        flow = Flow()

        with patch("crewai.llm.LLM") as MockLLM:
            mock_llm = MagicMock()
            # Invalid JSON that contains "approved"
            mock_llm.call.return_value = "{invalid json but says approved"
            MockLLM.return_value = mock_llm

            result = flow._collapse_to_outcome(
                feedback="looks good",
                outcomes=["approved", "rejected"],
                llm="gpt-4o-mini",
            )

        assert result == "approved"

    def test_llm_exception_falls_back_to_simple_prompting(self) -> None:
        """Test that LLM exception triggers fallback to simple prompting."""
        flow = Flow()

        with patch("crewai.llm.LLM") as MockLLM:
            mock_llm = MagicMock()
            # First call raises, second call succeeds (fallback)
            mock_llm.call.side_effect = [
                Exception("Structured output failed"),
                "approved",
            ]
            MockLLM.return_value = mock_llm

            result = flow._collapse_to_outcome(
                feedback="I approve",
                outcomes=["approved", "rejected"],
                llm="gpt-4o-mini",
            )

        assert result == "approved"
        # Verify it was called twice (initial + fallback)
        assert mock_llm.call.call_count == 2


class TestLLMObjectPreservedInContext:
    """Tests that BaseLLM objects have their model string preserved in PendingFeedbackContext."""

    @patch("crewai.flow.flow.crewai_event_bus.emit")
    def test_basellm_object_model_string_survives_roundtrip(self, mock_emit: MagicMock) -> None:
        """Test that when llm is a BaseLLM object, its model string is stored in context
        so that outcome collapsing works after async pause/resume.

        This is the exact bug: locally the sync path keeps the LLM object in memory,
        but in production the async path serializes the context and the LLM object was
        discarded (stored as None), causing resume to skip classification and always
        fall back to emit[0].
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_flows.db")
            persistence = SQLiteFlowPersistence(db_path)

            # Create a real LLM object (not a string)
            from crewai.llm import LLM
            mock_llm_obj = LLM(model="gemini-2.0-flash", provider="gemini")

            class PausingProvider:
                def __init__(self, persistence: SQLiteFlowPersistence):
                    self.persistence = persistence
                    self.captured_context: PendingFeedbackContext | None = None

                def request_feedback(
                    self, context: PendingFeedbackContext, flow: Flow
                ) -> str:
                    self.captured_context = context
                    self.persistence.save_pending_feedback(
                        flow_uuid=context.flow_id,
                        context=context,
                        state_data=flow.state if isinstance(flow.state, dict) else flow.state.model_dump(),
                    )
                    raise HumanFeedbackPending(context=context)

            provider = PausingProvider(persistence)

            class TestFlow(Flow):
                result_path: str = ""

                @start()
                @human_feedback(
                    message="Approve?",
                    emit=["needs_changes", "approved"],
                    llm=mock_llm_obj,
                    default_outcome="approved",
                    provider=provider,
                )
                def review(self):
                    return "content for review"

                @listen("approved")
                def handle_approved(self):
                    self.result_path = "approved"
                    return "Approved!"

                @listen("needs_changes")
                def handle_changes(self):
                    self.result_path = "needs_changes"
                    return "Changes needed"

            # Phase 1: Start flow (should pause)
            flow1 = TestFlow(persistence=persistence)
            result = flow1.kickoff()
            assert isinstance(result, HumanFeedbackPending)

            # Verify the context stored the model config dict, not None
            assert provider.captured_context is not None
            assert isinstance(provider.captured_context.llm, dict)
            assert provider.captured_context.llm["model"] == "gemini/gemini-2.0-flash"

            # Verify it survives persistence roundtrip
            flow_id = result.context.flow_id
            loaded = persistence.load_pending_feedback(flow_id)
            assert loaded is not None
            _, loaded_context = loaded
            assert isinstance(loaded_context.llm, dict)
            assert loaded_context.llm["model"] == "gemini/gemini-2.0-flash"

            # Phase 2: Resume with positive feedback - should use LLM to classify
            flow2 = TestFlow.from_pending(flow_id, persistence)
            assert flow2._pending_feedback_context is not None
            assert isinstance(flow2._pending_feedback_context.llm, dict)
            assert flow2._pending_feedback_context.llm["model"] == "gemini/gemini-2.0-flash"

            # Mock _collapse_to_outcome to verify it gets called (not skipped)
            with patch.object(flow2, "_collapse_to_outcome", return_value="approved") as mock_collapse:
                flow2.resume("this looks good, proceed!")

            # The key assertion: _collapse_to_outcome was called (not skipped due to llm=None)
            mock_collapse.assert_called_once()
            call_kwargs = mock_collapse.call_args
            assert call_kwargs.kwargs["feedback"] == "this looks good, proceed!"
            assert call_kwargs.kwargs["outcomes"] == ["needs_changes", "approved"]
            # LLM should be a live object (from _hf_llm) or reconstructed, not None
            assert call_kwargs.kwargs["llm"] is not None
            assert getattr(call_kwargs.kwargs["llm"], "model", None) == "gemini-2.0-flash"
            assert flow2.last_human_feedback.outcome == "approved"
            assert flow2.result_path == "approved"

    def test_string_llm_still_works(self) -> None:
        """Test that passing llm as a string still works correctly."""
        context = PendingFeedbackContext(
            flow_id="str-llm-test",
            flow_class="test.Flow",
            method_name="review",
            method_output="output",
            message="Review:",
            emit=["approved", "rejected"],
            llm="gpt-4o-mini",
        )

        serialized = context.to_dict()
        restored = PendingFeedbackContext.from_dict(serialized)
        assert restored.llm == "gpt-4o-mini"

    def test_none_llm_when_no_model_attr(self) -> None:
        """Test that llm is None when object has no model attribute."""
        from crewai.flow.human_feedback import _serialize_llm_for_context

        mock_obj = MagicMock(spec=[])  # No attributes
        assert _serialize_llm_for_context(mock_obj) is None

    def test_provider_prefix_added_to_bare_model(self) -> None:
        """Test that provider prefix is added when model has no slash."""
        from crewai.flow.human_feedback import _serialize_llm_for_context
        from crewai.llm import LLM

        llm = LLM(model="gemini-2.0-flash", provider="gemini")
        result = _serialize_llm_for_context(llm)
        assert isinstance(result, dict)
        assert result["model"] == "gemini/gemini-2.0-flash"

    def test_provider_prefix_not_doubled_when_already_present(self) -> None:
        """Test that provider prefix is not added when model already has a slash."""
        from crewai.flow.human_feedback import _serialize_llm_for_context
        from crewai.llm import LLM

        llm = LLM(model="gemini/gemini-2.0-flash")
        result = _serialize_llm_for_context(llm)
        assert isinstance(result, dict)
        assert result["model"] == "gemini/gemini-2.0-flash"

    def test_no_provider_attr_falls_back_to_bare_model(self) -> None:
        """Test that objects without to_config_dict fall back to model string."""
        from crewai.flow.human_feedback import _serialize_llm_for_context

        mock_obj = MagicMock(spec=[])
        mock_obj.model = "gpt-4o-mini"
        assert _serialize_llm_for_context(mock_obj) == "gpt-4o-mini"


class TestAsyncHumanFeedbackEdgeCases:
    """Edge case tests for async human feedback."""

    def test_pending_context_with_complex_output(self) -> None:
        """Test context with complex nested output."""
        complex_output = {
            "items": [{"id": 1, "name": "Item 1"}, {"id": 2, "name": "Item 2"}],
            "metadata": {"total": 2, "page": 1},
            "nested": {"deep": {"value": "test"}},
        }

        context = PendingFeedbackContext(
            flow_id="complex-test",
            flow_class="test.Flow",
            method_name="method",
            method_output=complex_output,
            message="Review:",
        )

        # Serialize and deserialize
        serialized = context.to_dict()
        json_str = json.dumps(serialized)  # Should be JSON serializable
        restored = PendingFeedbackContext.from_dict(json.loads(json_str))

        assert restored.method_output == complex_output

    def test_empty_feedback_uses_default_outcome(self) -> None:
        """Test that empty feedback uses default outcome during resume."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_flows.db")
            persistence = SQLiteFlowPersistence(db_path)

            class TestFlow(Flow):
                @start()
                def generate(self):
                    return "content"

            # Save pending feedback with default_outcome
            context = PendingFeedbackContext(
                flow_id="default-test",
                flow_class="test.Flow",
                method_name="generate",
                method_output="content",
                message="Review:",
                emit=["approved", "rejected"],
                default_outcome="approved",
                llm="gpt-4o-mini",
            )
            persistence.save_pending_feedback(
                flow_uuid="default-test",
                context=context,
                state_data={"id": "default-test"},
            )

            flow = TestFlow.from_pending("default-test", persistence)

            with patch("crewai.flow.flow.crewai_event_bus.emit"):
                result = flow.resume("")  # Empty feedback

            assert flow.last_human_feedback.outcome == "approved"

    def test_resume_without_feedback_uses_default(self) -> None:
        """Test that resume() can be called without feedback argument."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            persistence = SQLiteFlowPersistence(db_path)

            class TestFlow(Flow):
                @start()
                def step(self):
                    return "output"

            context = PendingFeedbackContext(
                flow_id="no-feedback-test",
                flow_class="TestFlow",
                method_name="step",
                method_output="test output",
                message="Review:",
                emit=["approved", "rejected"],
                default_outcome="approved",
                llm="gpt-4o-mini",
            )
            persistence.save_pending_feedback(
                flow_uuid="no-feedback-test",
                context=context,
                state_data={"id": "no-feedback-test"},
            )

            flow = TestFlow.from_pending("no-feedback-test", persistence)

            with patch("crewai.flow.flow.crewai_event_bus.emit"):
                # Call resume() with no arguments - should use default
                result = flow.resume()

            assert flow.last_human_feedback.outcome == "approved"
            assert flow.last_human_feedback.feedback == ""


# =============================================================================
# Tests for _hf_llm attribute and live LLM resolution on resume
# =============================================================================


class TestLiveLLMPreservationOnResume:
    """Tests for preserving the full LLM config across HITL resume."""

    def test_hf_llm_attribute_set_on_wrapper_with_basellm(self) -> None:
        """Test that _hf_llm is set on the wrapper when llm is a BaseLLM instance."""
        from crewai.llms.base_llm import BaseLLM

        # Create a mock BaseLLM object
        mock_llm = MagicMock(spec=BaseLLM)
        mock_llm.model = "gemini/gemini-3-flash"

        class TestFlow(Flow):
            @start()
            @human_feedback(
                message="Review:",
                emit=["approved", "rejected"],
                llm=mock_llm,
            )
            def review(self):
                return "content"

        flow = TestFlow()
        method = flow._methods.get("review")
        assert method is not None
        assert hasattr(method, "_hf_llm")
        assert method._hf_llm is mock_llm

    def test_hf_llm_attribute_set_on_wrapper_with_string(self) -> None:
        """Test that _hf_llm is set on the wrapper even when llm is a string."""

        class TestFlow(Flow):
            @start()
            @human_feedback(
                message="Review:",
                emit=["approved", "rejected"],
                llm="gpt-4o-mini",
            )
            def review(self):
                return "content"

        flow = TestFlow()
        method = flow._methods.get("review")
        assert method is not None
        assert hasattr(method, "_hf_llm")
        assert method._hf_llm == "gpt-4o-mini"

    @patch("crewai.flow.flow.crewai_event_bus.emit")
    def test_resume_async_uses_live_basellm_over_serialized_string(
        self, mock_emit: MagicMock
    ) -> None:
        """Test that resume_async uses the live BaseLLM from decorator instead of serialized string.

        This is the main bug fix: when a flow resumes, it should use the fully-configured
        LLM from the re-imported decorator (with credentials, project, etc.) instead of
        creating a new LLM from just the model string.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_flows.db")
            persistence = SQLiteFlowPersistence(db_path)

            from crewai.llms.base_llm import BaseLLM

            # Create a mock BaseLLM with full config (simulating Gemini with service account)
            live_llm = MagicMock(spec=BaseLLM)
            live_llm.model = "gemini/gemini-3-flash"

            class TestFlow(Flow):
                result_path: str = ""

                @start()
                @human_feedback(
                    message="Approve?",
                    emit=["approved", "rejected"],
                    llm=live_llm,  # Full LLM object with credentials
                )
                def review(self):
                    return "content"

                @listen("approved")
                def handle_approved(self):
                    self.result_path = "approved"
                    return "Approved!"

            # Save pending feedback with just a model STRING (simulating serialization)
            context = PendingFeedbackContext(
                flow_id="live-llm-test",
                flow_class="TestFlow",
                method_name="review",
                method_output="content",
                message="Approve?",
                emit=["approved", "rejected"],
                llm="gemini/gemini-3-flash",  # Serialized string, NOT the live object
            )
            persistence.save_pending_feedback(
                flow_uuid="live-llm-test",
                context=context,
                state_data={"id": "live-llm-test"},
            )

            # Restore flow - this re-imports the class with the live LLM
            flow = TestFlow.from_pending("live-llm-test", persistence)

            # Mock _collapse_to_outcome to capture what LLM it receives
            captured_llm = []

            def capture_llm(feedback, outcomes, llm):
                captured_llm.append(llm)
                return "approved"

            with patch.object(flow, "_collapse_to_outcome", side_effect=capture_llm):
                flow.resume("looks good!")

            # The key assertion: _collapse_to_outcome received the LIVE BaseLLM object,
            # NOT the serialized string. The live_llm was captured at class definition
            # time and stored on the method wrapper as _hf_llm.
            assert len(captured_llm) == 1
            # Verify it's the same object that was passed to the decorator
            # (which is stored on the method's _hf_llm attribute)
            method = flow._methods.get("review")
            assert method is not None
            assert captured_llm[0] is method._hf_llm
            # And verify it's a BaseLLM instance, not a string
            assert isinstance(captured_llm[0], BaseLLM)

    @patch("crewai.flow.flow.crewai_event_bus.emit")
    def test_resume_async_falls_back_to_serialized_string_when_no_hf_llm(
        self, mock_emit: MagicMock
    ) -> None:
        """Test that resume_async falls back to context.llm when _hf_llm is not available.

        This ensures backward compatibility with flows that were paused before this fix.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_flows.db")
            persistence = SQLiteFlowPersistence(db_path)

            class TestFlow(Flow):
                @start()
                @human_feedback(
                    message="Approve?",
                    emit=["approved", "rejected"],
                    llm="gpt-4o-mini",
                )
                def review(self):
                    return "content"

            # Save pending feedback
            context = PendingFeedbackContext(
                flow_id="fallback-test",
                flow_class="TestFlow",
                method_name="review",
                method_output="content",
                message="Approve?",
                emit=["approved", "rejected"],
                llm="gpt-4o-mini",
            )
            persistence.save_pending_feedback(
                flow_uuid="fallback-test",
                context=context,
                state_data={"id": "fallback-test"},
            )

            flow = TestFlow.from_pending("fallback-test", persistence)

            # Remove _hf_llm to simulate old decorator without this attribute
            method = flow._methods.get("review")
            if hasattr(method, "_hf_llm"):
                delattr(method, "_hf_llm")

            # Mock _collapse_to_outcome to capture what LLM it receives
            captured_llm = []

            def capture_llm(feedback, outcomes, llm):
                captured_llm.append(llm)
                return "approved"

            with patch.object(flow, "_collapse_to_outcome", side_effect=capture_llm):
                flow.resume("looks good!")

            # Should fall back to deserialized LLM from context string
            assert len(captured_llm) == 1
            from crewai.llms.base_llm import BaseLLM as BaseLLMClass
            assert isinstance(captured_llm[0], BaseLLMClass)
            assert captured_llm[0].model == "gpt-4o-mini"

    @patch("crewai.flow.flow.crewai_event_bus.emit")
    def test_resume_async_uses_string_from_context_when_hf_llm_is_string(
        self, mock_emit: MagicMock
    ) -> None:
        """Test that when _hf_llm is a string (not BaseLLM), we still use context.llm.

        String LLM values offer no benefit over the serialized context.llm,
        so we don't prefer them.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_flows.db")
            persistence = SQLiteFlowPersistence(db_path)

            class TestFlow(Flow):
                @start()
                @human_feedback(
                    message="Approve?",
                    emit=["approved", "rejected"],
                    llm="gpt-4o-mini",  # String LLM
                )
                def review(self):
                    return "content"

            # Save pending feedback
            context = PendingFeedbackContext(
                flow_id="string-llm-test",
                flow_class="TestFlow",
                method_name="review",
                method_output="content",
                message="Approve?",
                emit=["approved", "rejected"],
                llm="gpt-4o-mini",
            )
            persistence.save_pending_feedback(
                flow_uuid="string-llm-test",
                context=context,
                state_data={"id": "string-llm-test"},
            )

            flow = TestFlow.from_pending("string-llm-test", persistence)

            # Verify _hf_llm is a string
            method = flow._methods.get("review")
            assert method._hf_llm == "gpt-4o-mini"

            # Mock _collapse_to_outcome to capture what LLM it receives
            captured_llm = []

            def capture_llm(feedback, outcomes, llm):
                captured_llm.append(llm)
                return "approved"

            with patch.object(flow, "_collapse_to_outcome", side_effect=capture_llm):
                flow.resume("looks good!")

            # _hf_llm is a string, so resume deserializes context.llm into an LLM instance
            assert len(captured_llm) == 1
            from crewai.llms.base_llm import BaseLLM as BaseLLMClass
            assert isinstance(captured_llm[0], BaseLLMClass)
            assert captured_llm[0].model == "gpt-4o-mini"

    def test_hf_llm_set_for_async_wrapper(self) -> None:
        """Test that _hf_llm is set on async wrapper functions."""
        import asyncio
        from crewai.llms.base_llm import BaseLLM

        mock_llm = MagicMock(spec=BaseLLM)
        mock_llm.model = "gemini/gemini-3-flash"

        class TestFlow(Flow):
            @start()
            @human_feedback(
                message="Review:",
                emit=["approved", "rejected"],
                llm=mock_llm,
            )
            async def async_review(self):
                return "content"

        flow = TestFlow()
        method = flow._methods.get("async_review")
        assert method is not None
        assert hasattr(method, "_hf_llm")
        assert method._hf_llm is mock_llm
