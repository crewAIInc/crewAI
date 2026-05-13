"""Event types for the NewAgent system."""

from __future__ import annotations

from crewai.events.base_events import BaseEvent


class NewAgentCreatedEvent(BaseEvent):
    """Emitted when a NewAgent instance is constructed."""

    type: str = "new_agent_created"
    new_agent_id: str = ""
    new_agent_role: str = ""


class NewAgentConversationStartedEvent(BaseEvent):
    type: str = "new_agent_conversation_started"
    conversation_id: str = ""
    new_agent_id: str = ""
    new_agent_role: str = ""


class NewAgentConversationResetEvent(BaseEvent):
    type: str = "new_agent_conversation_reset"
    conversation_id: str = ""
    new_agent_id: str = ""


class NewAgentMessageReceivedEvent(BaseEvent):
    type: str = "new_agent_message_received"
    conversation_id: str = ""
    new_agent_id: str = ""
    message_length: int = 0


class NewAgentMessageSentEvent(BaseEvent):
    type: str = "new_agent_message_sent"
    conversation_id: str = ""
    new_agent_id: str = ""
    new_agent_role: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    response_time_ms: int = 0
    model: str = ""


class NewAgentStatusUpdateEvent(BaseEvent):
    type: str = "new_agent_status_update"
    state: str = ""
    detail: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    elapsed_ms: int = 0
    new_agent_id: str = ""


class NewAgentLLMCallStartedEvent(BaseEvent):
    type: str = "new_agent_llm_call_started"
    new_agent_id: str = ""
    model: str = ""


class NewAgentLLMCallCompletedEvent(BaseEvent):
    type: str = "new_agent_llm_call_completed"
    new_agent_id: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    response_time_ms: int = 0


class NewAgentLLMCallFailedEvent(BaseEvent):
    type: str = "new_agent_llm_call_failed"
    new_agent_id: str = ""
    error: str = ""


class NewAgentToolUsageStartedEvent(BaseEvent):
    type: str = "new_agent_tool_usage_started"
    new_agent_id: str = ""
    tool_name: str = ""


class NewAgentToolUsageCompletedEvent(BaseEvent):
    type: str = "new_agent_tool_usage_completed"
    new_agent_id: str = ""
    tool_name: str = ""


class NewAgentToolUsageFailedEvent(BaseEvent):
    type: str = "new_agent_tool_usage_failed"
    new_agent_id: str = ""
    tool_name: str = ""
    error: str = ""


class NewAgentDelegationStartedEvent(BaseEvent):
    type: str = "new_agent_delegation_started"
    new_agent_id: str = ""
    coworker_role: str = ""
    delegation_mode: str = "sync"
    coworker_source: str = "local"


class NewAgentDelegationCompletedEvent(BaseEvent):
    type: str = "new_agent_delegation_completed"
    new_agent_id: str = ""
    coworker_role: str = ""
    tokens_consumed: int = 0
    response_time_ms: int = 0


class NewAgentDelegationFailedEvent(BaseEvent):
    type: str = "new_agent_delegation_failed"
    new_agent_id: str = ""
    coworker_role: str = ""
    error: str = ""


class NewAgentFireAndForgetDispatchedEvent(BaseEvent):
    type: str = "new_agent_fire_and_forget_dispatched"
    new_agent_id: str = ""
    coworker_role: str = ""


class NewAgentMemorySaveEvent(BaseEvent):
    type: str = "new_agent_memory_save"
    new_agent_id: str = ""
    scope: str = ""


class NewAgentMemoryRecallEvent(BaseEvent):
    type: str = "new_agent_memory_recall"
    new_agent_id: str = ""
    scope: str = ""
    results_count: int = 0


class NewAgentDreamingStartedEvent(BaseEvent):
    type: str = "new_agent_dreaming_started"
    new_agent_id: str = ""


class NewAgentDreamingCompletedEvent(BaseEvent):
    type: str = "new_agent_dreaming_completed"
    new_agent_id: str = ""
    memories_processed: int = 0
    canonical_created: int = 0
    workflows_detected: int = 0


class NewAgentPlanningStartedEvent(BaseEvent):
    type: str = "new_agent_planning_started"
    new_agent_id: str = ""


class NewAgentPlanningCompletedEvent(BaseEvent):
    type: str = "new_agent_planning_completed"
    new_agent_id: str = ""
    plan_steps_count: int = 0


class NewAgentGuardrailPassedEvent(BaseEvent):
    type: str = "new_agent_guardrail_passed"
    new_agent_id: str = ""
    guardrail_type: str = ""


class NewAgentGuardrailRejectedEvent(BaseEvent):
    type: str = "new_agent_guardrail_rejected"
    new_agent_id: str = ""
    guardrail_type: str = ""
    retries: int = 0


class NewAgentKnowledgeQueryEvent(BaseEvent):
    type: str = "new_agent_knowledge_query"
    new_agent_id: str = ""


class NewAgentKnowledgeSuggestedEvent(BaseEvent):
    type: str = "new_agent_knowledge_suggested"
    new_agent_id: str = ""
    source_type: str = ""


class NewAgentExplainRequestedEvent(BaseEvent):
    type: str = "new_agent_explain_requested"
    new_agent_id: str = ""


class NewAgentSpawnStartedEvent(BaseEvent):
    type: str = "new_agent_spawn_started"
    new_agent_id: str = ""
    spawn_id: str = ""
    parent_id: str = ""
    spawn_depth: int = 0


class NewAgentSpawnCompletedEvent(BaseEvent):
    type: str = "new_agent_spawn_completed"
    new_agent_id: str = ""
    spawn_id: str = ""


class NewAgentSpawnFailedEvent(BaseEvent):
    type: str = "new_agent_spawn_failed"
    new_agent_id: str = ""
    spawn_id: str = ""
    error: str = ""


class NewAgentFireAndForgetCompletedEvent(BaseEvent):
    type: str = "new_agent_fire_and_forget_completed"
    new_agent_id: str = ""
    coworker_role: str = ""


class NewAgentContextSummarizedEvent(BaseEvent):
    type: str = "new_agent_context_summarized"
    new_agent_id: str = ""


class NewAgentNarrationGuardTriggeredEvent(BaseEvent):
    type: str = "new_agent_narration_guard_triggered"
    new_agent_id: str = ""
    retries: int = 0


class NewAgentWorkflowDetectedEvent(BaseEvent):
    type: str = "new_agent_workflow_detected"
    new_agent_id: str = ""
    tools: list[str] = []
    count: int = 0


class NewAgentWorkflowProposedEvent(BaseEvent):
    type: str = "new_agent_workflow_proposed"
    new_agent_id: str = ""
    workflow_description: str = ""


class NewAgentWorkflowConfirmedEvent(BaseEvent):
    type: str = "new_agent_workflow_confirmed"
    new_agent_id: str = ""


class NewAgentKnowledgeConfirmedEvent(BaseEvent):
    type: str = "new_agent_knowledge_confirmed"
    new_agent_id: str = ""
    source_type: str = ""


class NewAgentKnowledgeRejectedEvent(BaseEvent):
    type: str = "new_agent_knowledge_rejected"
    new_agent_id: str = ""


class NewAgentSkillSuggestedEvent(BaseEvent):
    type: str = "new_agent_skill_suggested"
    new_agent_id: str = ""
    skill_name: str = ""
    source_type: str = ""


class NewAgentSkillConfirmedEvent(BaseEvent):
    type: str = "new_agent_skill_confirmed"
    new_agent_id: str = ""
    skill_name: str = ""


class NewAgentSkillRejectedEvent(BaseEvent):
    type: str = "new_agent_skill_rejected"
    new_agent_id: str = ""
    skill_name: str = ""


class NewAgentTokenUsageEvent(BaseEvent):
    """Emitted when token usage is recorded, for platform billing."""

    type: str = "new_agent_token_usage"
    new_agent_id: str = ""
    conversation_id: str = ""
    action: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
