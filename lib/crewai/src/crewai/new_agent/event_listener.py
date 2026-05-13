"""Event listeners for the NewAgent system — bridges events to telemetry.

GAP-47: Uses a module-level registry to look up telemetry instances by agent ID.
GAP-61: Registers handlers for ALL event types defined in events.py.
"""

from __future__ import annotations

import logging
from typing import Any


logger = logging.getLogger(__name__)


def _get_tel(agent_id: str) -> Any:
    """Look up the telemetry instance for *agent_id* via the registry.

    Returns None (graceful degradation) if the agent is not registered.
    """
    try:
        from crewai.new_agent.telemetry import get_telemetry_for_agent

        return get_telemetry_for_agent(agent_id)
    except Exception:
        return None


def register_new_agent_listeners() -> None:
    """Register all NewAgent event listeners on the crewai event bus."""
    try:
        from crewai.events.event_bus import crewai_event_bus
        from crewai.new_agent.events import (
            NewAgentContextSummarizedEvent,
            NewAgentConversationResetEvent,
            NewAgentConversationStartedEvent,
            NewAgentDelegationCompletedEvent,
            NewAgentDelegationFailedEvent,
            NewAgentDelegationStartedEvent,
            NewAgentDreamingCompletedEvent,
            NewAgentDreamingStartedEvent,
            NewAgentExplainRequestedEvent,
            NewAgentFireAndForgetCompletedEvent,
            NewAgentFireAndForgetDispatchedEvent,
            NewAgentGuardrailPassedEvent,
            NewAgentGuardrailRejectedEvent,
            NewAgentKnowledgeConfirmedEvent,
            NewAgentKnowledgeQueryEvent,
            NewAgentKnowledgeRejectedEvent,
            NewAgentKnowledgeSuggestedEvent,
            NewAgentLLMCallCompletedEvent,
            NewAgentLLMCallFailedEvent,
            NewAgentLLMCallStartedEvent,
            NewAgentMemoryRecallEvent,
            NewAgentMemorySaveEvent,
            NewAgentMessageReceivedEvent,
            NewAgentMessageSentEvent,
            NewAgentNarrationGuardTriggeredEvent,
            NewAgentPlanningCompletedEvent,
            NewAgentPlanningStartedEvent,
            NewAgentSpawnCompletedEvent,
            NewAgentSpawnFailedEvent,
            NewAgentSpawnStartedEvent,
            NewAgentStatusUpdateEvent,
            NewAgentToolUsageCompletedEvent,
            NewAgentToolUsageFailedEvent,
            NewAgentToolUsageStartedEvent,
            NewAgentWorkflowConfirmedEvent,
            NewAgentWorkflowDetectedEvent,
            NewAgentWorkflowProposedEvent,
        )

        # ── Conversation ──────────────────────────────────────────

        @crewai_event_bus.on(NewAgentConversationStartedEvent)
        def _on_conversation_started(
            source: Any, event: NewAgentConversationStartedEvent
        ) -> None:
            logger.debug("NewAgent %s conversation started", event.new_agent_id)
            tel = _get_tel(event.new_agent_id)
            if tel:
                tel.agent_created(
                    agent_id=event.new_agent_id,
                    role=event.new_agent_role,
                    goal="",
                    llm="",
                )

        @crewai_event_bus.on(NewAgentConversationResetEvent)
        def _on_conversation_reset(
            source: Any, event: NewAgentConversationResetEvent
        ) -> None:
            logger.debug("NewAgent %s conversation reset", event.new_agent_id)
            tel = _get_tel(event.new_agent_id)
            if tel:
                tel.conversation_reset(agent_id=event.new_agent_id)

        # ── Messages ──────────────────────────────────────────────

        @crewai_event_bus.on(NewAgentMessageReceivedEvent)
        def _on_message_received(
            source: Any, event: NewAgentMessageReceivedEvent
        ) -> None:
            logger.debug(
                "NewAgent %s received message (%d chars)",
                event.new_agent_id,
                event.message_length,
            )
            tel = _get_tel(event.new_agent_id)
            if tel:
                tel.message_received(
                    agent_id=event.new_agent_id, message_length=event.message_length
                )

        @crewai_event_bus.on(NewAgentMessageSentEvent)
        def _on_message_sent(source: Any, event: NewAgentMessageSentEvent) -> None:
            logger.debug(
                "NewAgent %s sent message: %d in / %d out tokens",
                event.new_agent_role,
                event.input_tokens,
                event.output_tokens,
            )
            tel = _get_tel(event.new_agent_id)
            if tel:
                tel.message_sent(
                    agent_id=event.new_agent_id,
                    input_tokens=event.input_tokens,
                    output_tokens=event.output_tokens,
                    response_time_ms=event.response_time_ms,
                )

        # ── LLM Calls ────────────────────────────────────────────

        @crewai_event_bus.on(NewAgentLLMCallStartedEvent)
        def _on_llm_call_started(
            source: Any, event: NewAgentLLMCallStartedEvent
        ) -> None:
            logger.debug(
                "NewAgent %s LLM call started (model=%s)",
                event.new_agent_id,
                event.model,
            )
            tel = _get_tel(event.new_agent_id)
            if tel:
                tel.llm_call_started(agent_id=event.new_agent_id, model=event.model)

        @crewai_event_bus.on(NewAgentLLMCallCompletedEvent)
        def _on_llm_call_completed(
            source: Any, event: NewAgentLLMCallCompletedEvent
        ) -> None:
            logger.debug(
                "NewAgent %s LLM call completed: %d in / %d out tokens in %dms",
                event.new_agent_id,
                event.input_tokens,
                event.output_tokens,
                event.response_time_ms,
            )
            tel = _get_tel(event.new_agent_id)
            if tel:
                tel.llm_call_completed(
                    agent_id=event.new_agent_id,
                    model=event.model,
                    input_tokens=event.input_tokens,
                    output_tokens=event.output_tokens,
                    response_time_ms=event.response_time_ms,
                )

        @crewai_event_bus.on(NewAgentLLMCallFailedEvent)
        def _on_llm_call_failed(source: Any, event: NewAgentLLMCallFailedEvent) -> None:
            logger.warning(
                "NewAgent %s LLM call failed: %s", event.new_agent_id, event.error
            )
            tel = _get_tel(event.new_agent_id)
            if tel:
                tel.llm_call_failed(agent_id=event.new_agent_id, error=event.error)

        # ── Tool Usage ────────────────────────────────────────────

        @crewai_event_bus.on(NewAgentToolUsageStartedEvent)
        def _on_tool_started(source: Any, event: NewAgentToolUsageStartedEvent) -> None:
            logger.debug(
                "NewAgent %s using tool: %s", event.new_agent_id, event.tool_name
            )
            tel = _get_tel(event.new_agent_id)
            if tel:
                tel.tool_usage_started(
                    agent_id=event.new_agent_id, tool_name=event.tool_name
                )

        @crewai_event_bus.on(NewAgentToolUsageCompletedEvent)
        def _on_tool_completed(
            source: Any, event: NewAgentToolUsageCompletedEvent
        ) -> None:
            logger.debug(
                "NewAgent %s tool completed: %s", event.new_agent_id, event.tool_name
            )
            tel = _get_tel(event.new_agent_id)
            if tel:
                tel.tool_usage_completed_event(
                    agent_id=event.new_agent_id, tool_name=event.tool_name
                )

        @crewai_event_bus.on(NewAgentToolUsageFailedEvent)
        def _on_tool_failed(source: Any, event: NewAgentToolUsageFailedEvent) -> None:
            logger.warning(
                "NewAgent %s tool %s failed: %s",
                event.new_agent_id,
                event.tool_name,
                event.error,
            )
            tel = _get_tel(event.new_agent_id)
            if tel:
                tel.tool_usage_failed(
                    agent_id=event.new_agent_id,
                    tool_name=event.tool_name,
                    error=event.error,
                )

        # ── Delegation ────────────────────────────────────────────

        @crewai_event_bus.on(NewAgentDelegationStartedEvent)
        def _on_delegation_started(
            source: Any, event: NewAgentDelegationStartedEvent
        ) -> None:
            logger.debug(
                "NewAgent %s delegation started to %s",
                event.new_agent_id,
                event.coworker_role,
            )
            tel = _get_tel(event.new_agent_id)
            if tel:
                span = tel.delegation(
                    agent_id=event.new_agent_id,
                    coworker_role=event.coworker_role,
                    mode=event.delegation_mode,
                    source=event.coworker_source,
                )
                key = tel._span_key(
                    event.new_agent_id, "delegation", event.coworker_role
                )
                tel.store_span(key, span)

        @crewai_event_bus.on(NewAgentDelegationCompletedEvent)
        def _on_delegation_completed(
            source: Any, event: NewAgentDelegationCompletedEvent
        ) -> None:
            logger.debug(
                "NewAgent %s delegation to %s completed (%d tokens, %dms)",
                event.new_agent_id,
                event.coworker_role,
                event.tokens_consumed,
                event.response_time_ms,
            )
            tel = _get_tel(event.new_agent_id)
            if tel:
                key = tel._span_key(
                    event.new_agent_id, "delegation", event.coworker_role
                )
                span = tel.retrieve_span(key)
                tel.delegation_completed(
                    span,
                    tokens_consumed=event.tokens_consumed,
                    response_time_ms=event.response_time_ms,
                )

        @crewai_event_bus.on(NewAgentDelegationFailedEvent)
        def _on_delegation_failed(
            source: Any, event: NewAgentDelegationFailedEvent
        ) -> None:
            logger.warning(
                "NewAgent %s delegation to %s failed: %s",
                event.new_agent_id,
                event.coworker_role,
                event.error,
            )
            tel = _get_tel(event.new_agent_id)
            if tel:
                tel.delegation_failed(
                    agent_id=event.new_agent_id,
                    coworker_role=event.coworker_role,
                    error=event.error,
                )

        @crewai_event_bus.on(NewAgentFireAndForgetDispatchedEvent)
        def _on_fire_and_forget_dispatched(
            source: Any, event: NewAgentFireAndForgetDispatchedEvent
        ) -> None:
            logger.debug(
                "NewAgent %s fire-and-forget to %s",
                event.new_agent_id,
                event.coworker_role,
            )
            tel = _get_tel(event.new_agent_id)
            if tel:
                tel.fire_and_forget_dispatched(
                    agent_id=event.new_agent_id, coworker_role=event.coworker_role
                )

        @crewai_event_bus.on(NewAgentFireAndForgetCompletedEvent)
        def _on_fire_and_forget_completed(
            source: Any, event: NewAgentFireAndForgetCompletedEvent
        ) -> None:
            logger.debug(
                "NewAgent %s fire-and-forget to %s completed",
                event.new_agent_id,
                event.coworker_role,
            )
            tel = _get_tel(event.new_agent_id)
            if tel:
                tel.fire_and_forget_completed(
                    agent_id=event.new_agent_id, coworker_role=event.coworker_role
                )

        # ── Memory ────────────────────────────────────────────────

        @crewai_event_bus.on(NewAgentMemorySaveEvent)
        def _on_memory_save(source: Any, event: NewAgentMemorySaveEvent) -> None:
            logger.debug("NewAgent %s memory save", event.new_agent_id)
            tel = _get_tel(event.new_agent_id)
            if tel:
                tel.memory_save(agent_id=event.new_agent_id)

        @crewai_event_bus.on(NewAgentMemoryRecallEvent)
        def _on_memory_recall(source: Any, event: NewAgentMemoryRecallEvent) -> None:
            logger.debug(
                "NewAgent %s memory recall (%d results)",
                event.new_agent_id,
                event.results_count,
            )
            tel = _get_tel(event.new_agent_id)
            if tel:
                tel.memory_recall(
                    agent_id=event.new_agent_id, results_count=event.results_count
                )

        # ── Dreaming ──────────────────────────────────────────────

        @crewai_event_bus.on(NewAgentDreamingStartedEvent)
        def _on_dreaming_started(
            source: Any, event: NewAgentDreamingStartedEvent
        ) -> None:
            logger.debug("NewAgent %s dreaming started", event.new_agent_id)
            tel = _get_tel(event.new_agent_id)
            if tel:
                span = tel.dreaming(agent_id=event.new_agent_id)
                key = tel._span_key(event.new_agent_id, "dreaming")
                tel.store_span(key, span)

        @crewai_event_bus.on(NewAgentDreamingCompletedEvent)
        def _on_dreaming_completed(
            source: Any, event: NewAgentDreamingCompletedEvent
        ) -> None:
            logger.debug(
                "NewAgent %s dreaming: %d processed, %d canonical, %d workflows",
                event.new_agent_id,
                event.memories_processed,
                event.canonical_created,
                event.workflows_detected,
            )
            tel = _get_tel(event.new_agent_id)
            if tel:
                key = tel._span_key(event.new_agent_id, "dreaming")
                span = tel.retrieve_span(key)
                tel.dreaming_completed(
                    span,
                    memories_processed=event.memories_processed,
                    canonical_created=event.canonical_created,
                )

        # ── Planning ──────────────────────────────────────────────

        @crewai_event_bus.on(NewAgentPlanningStartedEvent)
        def _on_planning_started(
            source: Any, event: NewAgentPlanningStartedEvent
        ) -> None:
            logger.debug("NewAgent %s planning started", event.new_agent_id)
            tel = _get_tel(event.new_agent_id)
            if tel:
                span = tel.planning(agent_id=event.new_agent_id)
                key = tel._span_key(event.new_agent_id, "planning")
                tel.store_span(key, span)

        @crewai_event_bus.on(NewAgentPlanningCompletedEvent)
        def _on_planning_completed(
            source: Any, event: NewAgentPlanningCompletedEvent
        ) -> None:
            logger.debug(
                "NewAgent %s planned %d steps",
                event.new_agent_id,
                event.plan_steps_count,
            )
            tel = _get_tel(event.new_agent_id)
            if tel:
                key = tel._span_key(event.new_agent_id, "planning")
                span = tel.retrieve_span(key)
                tel.planning_completed(span, steps_count=event.plan_steps_count)

        # ── Guardrails ────────────────────────────────────────────

        @crewai_event_bus.on(NewAgentGuardrailPassedEvent)
        def _on_guardrail_passed(
            source: Any, event: NewAgentGuardrailPassedEvent
        ) -> None:
            logger.debug(
                "NewAgent %s guardrail passed (%s)",
                event.new_agent_id,
                event.guardrail_type,
            )
            tel = _get_tel(event.new_agent_id)
            if tel:
                tel.guardrail_passed(
                    agent_id=event.new_agent_id, guardrail_type=event.guardrail_type
                )

        @crewai_event_bus.on(NewAgentGuardrailRejectedEvent)
        def _on_guardrail_rejected(
            source: Any, event: NewAgentGuardrailRejectedEvent
        ) -> None:
            logger.warning(
                "NewAgent %s guardrail rejected (%s) after %d retries",
                event.new_agent_id,
                event.guardrail_type,
                event.retries,
            )
            tel = _get_tel(event.new_agent_id)
            if tel:
                tel.guardrail(
                    agent_id=event.new_agent_id, guardrail_type=event.guardrail_type
                )

        # ── Knowledge ─────────────────────────────────────────────

        @crewai_event_bus.on(NewAgentKnowledgeQueryEvent)
        def _on_knowledge_query(
            source: Any, event: NewAgentKnowledgeQueryEvent
        ) -> None:
            logger.debug("NewAgent %s knowledge query", event.new_agent_id)
            tel = _get_tel(event.new_agent_id)
            if tel:
                tel.knowledge_query(agent_id=event.new_agent_id)

        @crewai_event_bus.on(NewAgentKnowledgeSuggestedEvent)
        def _on_knowledge_suggested(
            source: Any, event: NewAgentKnowledgeSuggestedEvent
        ) -> None:
            logger.debug(
                "NewAgent %s knowledge suggested (type=%s)",
                event.new_agent_id,
                event.source_type,
            )
            tel = _get_tel(event.new_agent_id)
            if tel:
                tel.knowledge_suggested(
                    agent_id=event.new_agent_id, source_type=event.source_type
                )

        @crewai_event_bus.on(NewAgentKnowledgeConfirmedEvent)
        def _on_knowledge_confirmed(
            source: Any, event: NewAgentKnowledgeConfirmedEvent
        ) -> None:
            logger.debug(
                "NewAgent %s knowledge confirmed (type=%s)",
                event.new_agent_id,
                event.source_type,
            )
            tel = _get_tel(event.new_agent_id)
            if tel:
                tel.knowledge_confirmed(
                    agent_id=event.new_agent_id, source_type=event.source_type
                )

        @crewai_event_bus.on(NewAgentKnowledgeRejectedEvent)
        def _on_knowledge_rejected(
            source: Any, event: NewAgentKnowledgeRejectedEvent
        ) -> None:
            logger.debug("NewAgent %s knowledge rejected", event.new_agent_id)
            tel = _get_tel(event.new_agent_id)
            if tel:
                tel.knowledge_rejected(agent_id=event.new_agent_id)

        # ── Explain ───────────────────────────────────────────────

        @crewai_event_bus.on(NewAgentExplainRequestedEvent)
        def _on_explain_requested(
            source: Any, event: NewAgentExplainRequestedEvent
        ) -> None:
            logger.debug("NewAgent %s explain requested", event.new_agent_id)
            tel = _get_tel(event.new_agent_id)
            if tel:
                tel.explain_requested(agent_id=event.new_agent_id)

        # ── Spawn ─────────────────────────────────────────────────

        @crewai_event_bus.on(NewAgentSpawnStartedEvent)
        def _on_spawn_started(source: Any, event: NewAgentSpawnStartedEvent) -> None:
            logger.debug(
                "NewAgent %s spawn started (id=%s, depth=%d)",
                event.new_agent_id,
                event.spawn_id,
                event.spawn_depth,
            )
            tel = _get_tel(event.new_agent_id)
            if tel:
                span = tel.spawn(
                    agent_id=event.new_agent_id,
                    spawn_id=event.spawn_id,
                    depth=event.spawn_depth,
                )
                key = tel._span_key(event.new_agent_id, "spawn", event.spawn_id)
                tel.store_span(key, span)

        @crewai_event_bus.on(NewAgentSpawnCompletedEvent)
        def _on_spawn_completed(
            source: Any, event: NewAgentSpawnCompletedEvent
        ) -> None:
            logger.debug(
                "NewAgent %s spawn completed (id=%s)",
                event.new_agent_id,
                event.spawn_id,
            )
            tel = _get_tel(event.new_agent_id)
            if tel:
                key = tel._span_key(event.new_agent_id, "spawn", event.spawn_id)
                span = tel.retrieve_span(key)
                if span:
                    tel.spawn_completed(span)
                else:
                    tel.spawn_completed_event(
                        agent_id=event.new_agent_id, spawn_id=event.spawn_id
                    )

        @crewai_event_bus.on(NewAgentSpawnFailedEvent)
        def _on_spawn_failed(source: Any, event: NewAgentSpawnFailedEvent) -> None:
            logger.warning(
                "NewAgent %s spawn failed (id=%s): %s",
                event.new_agent_id,
                event.spawn_id,
                event.error,
            )
            tel = _get_tel(event.new_agent_id)
            if tel:
                tel.spawn_failed(
                    agent_id=event.new_agent_id,
                    spawn_id=event.spawn_id,
                    error=event.error,
                )

        # ── Narration ─────────────────────────────────────────────

        @crewai_event_bus.on(NewAgentNarrationGuardTriggeredEvent)
        def _on_narration_guard(
            source: Any, event: NewAgentNarrationGuardTriggeredEvent
        ) -> None:
            logger.debug(
                "NewAgent %s narration guard triggered (%d retries)",
                event.new_agent_id,
                event.retries,
            )
            tel = _get_tel(event.new_agent_id)
            if tel:
                tel.narration_guard_triggered(
                    agent_id=event.new_agent_id, retries=event.retries
                )

        # ── Context ───────────────────────────────────────────────

        @crewai_event_bus.on(NewAgentContextSummarizedEvent)
        def _on_context_summarized(
            source: Any, event: NewAgentContextSummarizedEvent
        ) -> None:
            logger.debug("NewAgent %s context summarized", event.new_agent_id)
            tel = _get_tel(event.new_agent_id)
            if tel:
                tel.context_summarized(agent_id=event.new_agent_id)

        # ── Status Updates ────────────────────────────────────────

        @crewai_event_bus.on(NewAgentStatusUpdateEvent)
        def _on_status_update(source: Any, event: NewAgentStatusUpdateEvent) -> None:
            logger.debug(
                "NewAgent status update: %s (%s)", event.state, event.detail or ""
            )

        # ── Workflow Events ───────────────────────────────────────

        @crewai_event_bus.on(NewAgentWorkflowDetectedEvent)
        def _on_workflow_detected(
            source: Any, event: NewAgentWorkflowDetectedEvent
        ) -> None:
            logger.debug(
                "NewAgent %s workflow detected: %s (%dx)",
                event.new_agent_id,
                event.tools,
                event.count,
            )
            tel = _get_tel(event.new_agent_id)
            if tel:
                tel.workflow_detected(
                    agent_id=event.new_agent_id, tools=event.tools, count=event.count
                )

        @crewai_event_bus.on(NewAgentWorkflowProposedEvent)
        def _on_workflow_proposed(
            source: Any, event: NewAgentWorkflowProposedEvent
        ) -> None:
            logger.debug("NewAgent %s workflow proposed", event.new_agent_id)
            tel = _get_tel(event.new_agent_id)
            if tel:
                tel.workflow_proposed(
                    agent_id=event.new_agent_id, description=event.workflow_description
                )

        @crewai_event_bus.on(NewAgentWorkflowConfirmedEvent)
        def _on_workflow_confirmed(
            source: Any, event: NewAgentWorkflowConfirmedEvent
        ) -> None:
            logger.debug("NewAgent %s workflow confirmed", event.new_agent_id)
            tel = _get_tel(event.new_agent_id)
            if tel:
                tel.workflow_confirmed(agent_id=event.new_agent_id)

        logger.debug("NewAgent event listeners registered (all event types)")

    except Exception as e:
        logger.debug("Failed to register NewAgent event listeners: %s", e)
