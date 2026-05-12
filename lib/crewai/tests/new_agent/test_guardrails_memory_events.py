"""Tests for guardrails, memory integration, events, and advanced features."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
import pytest

from crewai.new_agent import AgentSettings, Message, NewAgent
from crewai.new_agent.events import (
    NewAgentConversationStartedEvent,
    NewAgentGuardrailPassedEvent,
    NewAgentGuardrailRejectedEvent,
    NewAgentMessageReceivedEvent,
    NewAgentMessageSentEvent,
    NewAgentDelegationStartedEvent,
    NewAgentDelegationCompletedEvent,
    NewAgentToolUsageStartedEvent,
    NewAgentToolUsageCompletedEvent,
    NewAgentDreamingStartedEvent,
    NewAgentDreamingCompletedEvent,
    NewAgentPlanningStartedEvent,
    NewAgentPlanningCompletedEvent,
    NewAgentSpawnStartedEvent,
    NewAgentSpawnCompletedEvent,
    NewAgentMemorySaveEvent,
    NewAgentMemoryRecallEvent,
    NewAgentKnowledgeQueryEvent,
    NewAgentExplainRequestedEvent,
)


# ── Guardrail tests ─────────────────────────────────────────

class TestGuardrails:
    @patch("crewai.new_agent.executor.aget_llm_response")
    @pytest.mark.asyncio
    async def test_code_guardrail_passes(self, mock_llm):
        mock_llm.return_value = "Safe response."

        def my_guardrail(response: str) -> tuple[bool, str]:
            return True, ""

        agent = NewAgent(
            role="R", goal="g",
            guardrail=my_guardrail,
            settings=AgentSettings(memory_enabled=False),
        )
        result = await agent.amessage("Hi")
        assert result.content == "Safe response."

    @patch("crewai.new_agent.executor.aget_llm_response")
    @pytest.mark.asyncio
    async def test_code_guardrail_rejects_and_retries(self, mock_llm):
        mock_llm.side_effect = ["Bad response with SECRET.", "Clean response."]

        call_count = 0

        def my_guardrail(response: str) -> tuple[bool, str]:
            nonlocal call_count
            call_count += 1
            if "SECRET" in response:
                return False, "Do not include secrets."
            return True, ""

        agent = NewAgent(
            role="R", goal="g",
            guardrail=my_guardrail,
            settings=AgentSettings(memory_enabled=False, max_retry_limit=2),
        )
        result = await agent.amessage("Tell me a secret")
        assert call_count >= 1

    @patch("crewai.new_agent.executor.aget_llm_response")
    @pytest.mark.asyncio
    async def test_bool_guardrail(self, mock_llm):
        mock_llm.return_value = "OK response."

        def simple_guard(response: str) -> bool:
            return len(response) > 0

        agent = NewAgent(
            role="R", goal="g",
            guardrail=simple_guard,
            settings=AgentSettings(memory_enabled=False),
        )
        result = await agent.amessage("Hi")
        assert result.content == "OK response."


# ── Memory integration tests ────────────────────────────────

class TestMemoryIntegration:
    def test_memory_enabled_by_default(self):
        agent = NewAgent(role="R", goal="g")
        assert agent.settings.memory_enabled is True

    def test_memory_disabled(self):
        agent = NewAgent(
            role="R", goal="g",
            memory=False,
            settings=AgentSettings(memory_enabled=False),
        )
        assert agent._memory_instance is None

    @patch("crewai.new_agent.executor.aget_llm_response")
    @pytest.mark.asyncio
    async def test_memory_recall_in_prompt(self, mock_llm):
        mock_llm.return_value = "Response with memory context."

        agent = NewAgent(
            role="Researcher",
            goal="Research",
            settings=AgentSettings(memory_enabled=False),
        )
        result = await agent.amessage("What do you know?")

        stack = agent.last_prompt_stack
        assert stack is not None
        layer_names = [l.name for l in stack.layers]
        assert "soul" in layer_names
        assert "temporal" in layer_names


# ── Event types tests ────────────────────────────────────────

class TestAllEventTypes:
    """Verify all event types can be instantiated with proper defaults."""

    def test_conversation_started(self):
        e = NewAgentConversationStartedEvent(new_agent_id="a1", new_agent_role="R", conversation_id="c1")
        assert e.type == "new_agent_conversation_started"

    def test_message_received(self):
        e = NewAgentMessageReceivedEvent(new_agent_id="a1", message_length=42, conversation_id="c1")
        assert e.message_length == 42

    def test_message_sent(self):
        e = NewAgentMessageSentEvent(new_agent_id="a1", model="gpt-4o", input_tokens=100, output_tokens=50, conversation_id="c1")
        assert e.input_tokens == 100

    def test_tool_usage_started(self):
        e = NewAgentToolUsageStartedEvent(new_agent_id="a1", tool_name="search")
        assert e.tool_name == "search"

    def test_tool_usage_completed(self):
        e = NewAgentToolUsageCompletedEvent(new_agent_id="a1", tool_name="search")
        assert e.type == "new_agent_tool_usage_completed"

    def test_delegation_started(self):
        e = NewAgentDelegationStartedEvent(
            new_agent_id="a1",
            coworker_role="Writer",
            delegation_mode="sync",
            coworker_source="local",
        )
        assert e.coworker_source == "local"

    def test_delegation_completed(self):
        e = NewAgentDelegationCompletedEvent(
            new_agent_id="a1",
            coworker_role="Writer",
            tokens_consumed=500,
            response_time_ms=2000,
        )
        assert e.tokens_consumed == 500

    def test_guardrail_passed(self):
        e = NewAgentGuardrailPassedEvent(new_agent_id="a1", guardrail_type="code")
        assert e.guardrail_type == "code"

    def test_guardrail_rejected(self):
        e = NewAgentGuardrailRejectedEvent(new_agent_id="a1", guardrail_type="llm", retries=2)
        assert e.retries == 2

    def test_dreaming(self):
        e = NewAgentDreamingStartedEvent(new_agent_id="a1")
        assert e.type == "new_agent_dreaming_started"
        e2 = NewAgentDreamingCompletedEvent(
            new_agent_id="a1",
            memories_processed=10,
            canonical_created=3,
            workflows_detected=1,
        )
        assert e2.canonical_created == 3

    def test_planning(self):
        e = NewAgentPlanningStartedEvent(new_agent_id="a1")
        assert e.type == "new_agent_planning_started"
        e2 = NewAgentPlanningCompletedEvent(new_agent_id="a1", plan_steps_count=5)
        assert e2.plan_steps_count == 5

    def test_spawn(self):
        e = NewAgentSpawnStartedEvent(
            new_agent_id="a1",
            spawn_id="s1",
            parent_id="p1",
            spawn_depth=1,
        )
        assert e.spawn_depth == 1
        e2 = NewAgentSpawnCompletedEvent(new_agent_id="a1", spawn_id="s1")
        assert e2.type == "new_agent_spawn_completed"

    def test_memory_events(self):
        e = NewAgentMemorySaveEvent(new_agent_id="a1", scope="/user")
        assert e.scope == "/user"
        e2 = NewAgentMemoryRecallEvent(new_agent_id="a1", scope="/user", results_count=3)
        assert e2.results_count == 3

    def test_explain_event(self):
        e = NewAgentExplainRequestedEvent(new_agent_id="a1")
        assert e.type == "new_agent_explain_requested"


# ── Event emission tests ─────────────────────────────────────

class TestEventEmission:
    @patch("crewai.new_agent.executor.aget_llm_response")
    @pytest.mark.asyncio
    async def test_events_emitted_on_message(self, mock_llm):
        mock_llm.return_value = "Response."

        emitted_events = []

        def capture_event(source, event):
            emitted_events.append(event)

        with patch("crewai.events.event_bus.crewai_event_bus.emit", side_effect=capture_event):
            agent = NewAgent(
                role="R", goal="g",
                settings=AgentSettings(memory_enabled=False),
            )
            await agent.amessage("Hello")

        event_types = [type(e).__name__ for e in emitted_events]
        # GAP-84: At construction, NewAgentCreatedEvent is emitted instead of ConversationStarted
        assert "NewAgentCreatedEvent" in event_types
        assert "NewAgentMessageReceivedEvent" in event_types
        assert "NewAgentMessageSentEvent" in event_types


# ── Structured output tests ──────────────────────────────────

class TestStructuredOutput:
    def test_response_model_attribute(self):
        from pydantic import BaseModel

        class Result(BaseModel):
            summary: str
            confidence: float

        agent = NewAgent(
            role="R", goal="g",
            response_model=Result,
            settings=AgentSettings(memory_enabled=False),
        )
        assert agent.response_model is Result


# ── Multi-agent delegation tests ─────────────────────────────

class TestMultiAgentDelegation:
    def test_multiple_coworkers(self):
        writer = NewAgent(role="Writer", goal="Write", settings=AgentSettings(memory_enabled=False))
        reviewer = NewAgent(role="Reviewer", goal="Review", settings=AgentSettings(memory_enabled=False))

        manager = NewAgent(
            role="Manager",
            goal="Manage",
            coworkers=[writer, reviewer],
            settings=AgentSettings(memory_enabled=False),
        )

        assert len(manager._resolved_coworkers) == 2
        # 2 individual delegation tools + 1 multi-delegate tool
        assert len(manager._coworker_tools) == 3

        tool_names = [t.name for t in manager._coworker_tools]
        assert any("writer" in n.lower() for n in tool_names)
        assert any("reviewer" in n.lower() for n in tool_names)
        assert any("multiple" in n.lower() for n in tool_names)

    @patch("crewai.new_agent.executor.aget_llm_response")
    @pytest.mark.asyncio
    async def test_delegation_via_tool(self, mock_llm):
        mock_llm.return_value = "Writer's output."

        writer = NewAgent(
            role="Writer", goal="Write articles",
            settings=AgentSettings(memory_enabled=False),
        )

        from crewai.new_agent.coworker_tools import DelegateToCoworkerTool
        tool = DelegateToCoworkerTool(coworker=writer, source="local")

        result = tool._run(message="Write about AI")
        assert "Writer's output." in result

    def test_coworker_tool_args_schema(self):
        writer = NewAgent(role="Writer", goal="Write", settings=AgentSettings(memory_enabled=False))

        from crewai.new_agent.coworker_tools import DelegateToCoworkerTool
        tool = DelegateToCoworkerTool(coworker=writer)

        schema = tool.args_schema.model_json_schema()
        assert "message" in schema["properties"]
        assert "fire_and_forget" in schema["properties"]


# ── LLM Guardrail tests ────────────────────────────────────

class TestLLMGuardrails:
    @patch("crewai.new_agent.executor.aget_llm_response")
    @pytest.mark.asyncio
    async def test_llm_guardrail_passes(self, mock_llm):
        """LLM guardrail that returns PASS should let the response through."""
        from crewai.tasks.llm_guardrail import LLMGuardrail

        # First call: the main agent response. Second call: guardrail evaluation.
        mock_llm.side_effect = ["A good response.", "PASS"]

        mock_guardrail_llm = MagicMock()
        guardrail = LLMGuardrail(
            description="Response must be polite.",
            llm=mock_guardrail_llm,
        )

        agent = NewAgent(
            role="R", goal="g",
            guardrail=guardrail,
            settings=AgentSettings(memory_enabled=False),
        )
        result = await agent.amessage("Hi")
        assert result.content == "A good response."

    @patch("crewai.new_agent.executor.aget_llm_response")
    @pytest.mark.asyncio
    async def test_llm_guardrail_rejects_and_retries(self, mock_llm):
        """LLM guardrail that returns FAIL should trigger regeneration."""
        from crewai.tasks.llm_guardrail import LLMGuardrail

        # Call sequence:
        # 1. Main response: "Bad response"
        # 2. Guardrail evaluation: "FAIL: contains rude language"
        # 3. Regeneration: "Fixed response"
        # 4. Guardrail re-evaluation: "PASS"
        mock_llm.side_effect = [
            "Bad response",
            "FAIL: contains rude language",
            "Fixed response",
            "PASS",
        ]

        mock_guardrail_llm = MagicMock()
        guardrail = LLMGuardrail(
            description="Response must be polite.",
            llm=mock_guardrail_llm,
        )

        agent = NewAgent(
            role="R", goal="g",
            guardrail=guardrail,
            settings=AgentSettings(memory_enabled=False, max_retry_limit=2),
        )
        result = await agent.amessage("Be rude")
        # After FAIL, it regenerates and the guardrail passes
        assert result.content == "Fixed response"

    @patch("crewai.new_agent.executor.aget_llm_response")
    @pytest.mark.asyncio
    async def test_llm_guardrail_falls_back_to_agent_llm(self, mock_llm):
        """When guardrail has no LLM, it should use the agent's LLM."""
        from crewai.tasks.llm_guardrail import LLMGuardrail

        mock_llm.side_effect = ["Some response.", "PASS"]

        guardrail = LLMGuardrail(
            description="Response must be safe.",
            llm=None,  # No guardrail LLM — should fall back to agent's
        )
        # Override llm to None so the isinstance(llm, str) path is not hit
        guardrail.llm = None

        agent = NewAgent(
            role="R", goal="g",
            guardrail=guardrail,
            settings=AgentSettings(memory_enabled=False),
        )
        result = await agent.amessage("Hello")
        assert result.content == "Some response."

    @patch("crewai.new_agent.executor.aget_llm_response")
    @pytest.mark.asyncio
    async def test_llm_guardrail_emits_correct_event_type(self, mock_llm):
        """LLM guardrail should emit events with guardrail_type='llm'."""
        from crewai.tasks.llm_guardrail import LLMGuardrail

        mock_llm.side_effect = ["Response.", "PASS"]

        emitted_events = []

        def capture_event(source, event):
            emitted_events.append(event)

        guardrail = LLMGuardrail(
            description="Must be safe.",
            llm=MagicMock(),
        )

        with patch("crewai.events.event_bus.crewai_event_bus.emit", side_effect=capture_event):
            agent = NewAgent(
                role="R", goal="g",
                guardrail=guardrail,
                settings=AgentSettings(memory_enabled=False),
            )
            await agent.amessage("Hi")

        guardrail_events = [
            e for e in emitted_events
            if type(e).__name__ == "NewAgentGuardrailPassedEvent"
        ]
        assert len(guardrail_events) >= 1
        assert guardrail_events[0].guardrail_type == "llm"


# ── Structured output tests (parsing) ──────────────────────

class TestStructuredOutputParsing:
    @patch("crewai.new_agent.executor.aget_llm_response")
    @pytest.mark.asyncio
    async def test_structured_output_from_json(self, mock_llm):
        """When LLM returns valid JSON, it should be parsed into response_model."""
        from pydantic import BaseModel

        class Result(BaseModel):
            summary: str
            confidence: float

        json_response = json.dumps({"summary": "Test summary", "confidence": 0.95})
        mock_llm.return_value = json_response

        agent = NewAgent(
            role="R", goal="g",
            response_model=Result,
            settings=AgentSettings(memory_enabled=False),
        )
        result = await agent.amessage("Analyze this")
        assert result.content == json_response
        assert result.metadata is not None
        assert "structured_output" in result.metadata
        assert result.metadata["structured_output"]["summary"] == "Test summary"
        assert result.metadata["structured_output"]["confidence"] == 0.95

    @patch("crewai.new_agent.executor.aget_llm_response")
    @pytest.mark.asyncio
    async def test_structured_output_from_markdown_json(self, mock_llm):
        """When LLM returns JSON wrapped in markdown fences, it should still parse."""
        from pydantic import BaseModel

        class Result(BaseModel):
            summary: str
            confidence: float

        json_str = json.dumps({"summary": "Parsed from markdown", "confidence": 0.8})
        markdown_response = f"```json\n{json_str}\n```"
        mock_llm.return_value = markdown_response

        agent = NewAgent(
            role="R", goal="g",
            response_model=Result,
            settings=AgentSettings(memory_enabled=False),
        )
        result = await agent.amessage("Analyze this")
        assert result.metadata is not None
        assert result.metadata["structured_output"]["summary"] == "Parsed from markdown"

    @patch("crewai.new_agent.executor.aget_llm_response")
    @pytest.mark.asyncio
    async def test_structured_output_llm_extraction_fallback(self, mock_llm):
        """When text is not JSON, it should ask the LLM to extract structured data."""
        from pydantic import BaseModel

        class Result(BaseModel):
            summary: str
            confidence: float

        # First call: main agent response (not JSON).
        # Second call: LLM extraction returns valid JSON.
        mock_llm.side_effect = [
            "The analysis shows high confidence in the results.",
            json.dumps({"summary": "High confidence analysis", "confidence": 0.92}),
        ]

        agent = NewAgent(
            role="R", goal="g",
            response_model=Result,
            settings=AgentSettings(memory_enabled=False),
        )
        result = await agent.amessage("Analyze this")
        assert result.content == "The analysis shows high confidence in the results."
        assert result.metadata is not None
        assert result.metadata["structured_output"]["summary"] == "High confidence analysis"

    @patch("crewai.new_agent.executor.aget_llm_response")
    @pytest.mark.asyncio
    async def test_structured_output_none_when_no_model(self, mock_llm):
        """When response_model is not set, metadata should not contain structured_output."""
        mock_llm.return_value = "Plain response."

        agent = NewAgent(
            role="R", goal="g",
            settings=AgentSettings(memory_enabled=False),
        )
        result = await agent.amessage("Hello")
        assert result.metadata is None

    @patch("crewai.new_agent.executor.aget_llm_response")
    @pytest.mark.asyncio
    async def test_structured_output_none_on_failure(self, mock_llm):
        """When both direct parse and LLM extraction fail, metadata should be None."""
        from pydantic import BaseModel

        class Result(BaseModel):
            summary: str
            confidence: float

        # First call: main response (not JSON).
        # Second call: LLM extraction also returns non-JSON.
        mock_llm.side_effect = [
            "Not JSON at all.",
            "I cannot extract structured data from this.",
        ]

        agent = NewAgent(
            role="R", goal="g",
            response_model=Result,
            settings=AgentSettings(memory_enabled=False),
        )
        result = await agent.amessage("Hello")
        assert result.content == "Not JSON at all."
        # metadata should be None since structured parsing failed
        assert result.metadata is None
