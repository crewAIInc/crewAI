"""Checkpoint configuration for automatic state persistence."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, model_validator

from crewai.state.provider.json_provider import JsonProvider
from crewai.state.provider.sqlite_provider import SqliteProvider


CheckpointEventType = Literal[
    # Task
    "task_started",
    "task_completed",
    "task_failed",
    "task_evaluation",
    # Crew
    "crew_kickoff_started",
    "crew_kickoff_completed",
    "crew_kickoff_failed",
    "crew_train_started",
    "crew_train_completed",
    "crew_train_failed",
    "crew_test_started",
    "crew_test_completed",
    "crew_test_failed",
    "crew_test_result",
    # Agent
    "agent_execution_started",
    "agent_execution_completed",
    "agent_execution_error",
    "lite_agent_execution_started",
    "lite_agent_execution_completed",
    "lite_agent_execution_error",
    "agent_evaluation_started",
    "agent_evaluation_completed",
    "agent_evaluation_failed",
    # Flow
    "flow_created",
    "flow_started",
    "flow_finished",
    "flow_paused",
    "method_execution_started",
    "method_execution_finished",
    "method_execution_failed",
    "method_execution_paused",
    "human_feedback_requested",
    "human_feedback_received",
    "flow_input_requested",
    "flow_input_received",
    # LLM
    "llm_call_started",
    "llm_call_completed",
    "llm_call_failed",
    "llm_stream_chunk",
    "llm_thinking_chunk",
    # LLM Guardrail
    "llm_guardrail_started",
    "llm_guardrail_completed",
    "llm_guardrail_failed",
    # Tool
    "tool_usage_started",
    "tool_usage_finished",
    "tool_usage_error",
    "tool_validate_input_error",
    "tool_selection_error",
    "tool_execution_error",
    # Memory
    "memory_save_started",
    "memory_save_completed",
    "memory_save_failed",
    "memory_query_started",
    "memory_query_completed",
    "memory_query_failed",
    "memory_retrieval_started",
    "memory_retrieval_completed",
    "memory_retrieval_failed",
    # Knowledge
    "knowledge_search_query_started",
    "knowledge_search_query_completed",
    "knowledge_query_started",
    "knowledge_query_completed",
    "knowledge_query_failed",
    "knowledge_search_query_failed",
    # Reasoning
    "agent_reasoning_started",
    "agent_reasoning_completed",
    "agent_reasoning_failed",
    # MCP
    "mcp_connection_started",
    "mcp_connection_completed",
    "mcp_connection_failed",
    "mcp_tool_execution_started",
    "mcp_tool_execution_completed",
    "mcp_tool_execution_failed",
    "mcp_config_fetch_failed",
    # Observation
    "step_observation_started",
    "step_observation_completed",
    "step_observation_failed",
    "plan_refinement",
    "plan_replan_triggered",
    "goal_achieved_early",
    # Skill
    "skill_discovery_started",
    "skill_discovery_completed",
    "skill_loaded",
    "skill_activated",
    "skill_load_failed",
    # Logging
    "agent_logs_started",
    "agent_logs_execution",
    # A2A
    "a2a_delegation_started",
    "a2a_delegation_completed",
    "a2a_conversation_started",
    "a2a_conversation_completed",
    "a2a_message_sent",
    "a2a_response_received",
    "a2a_polling_started",
    "a2a_polling_status",
    "a2a_push_notification_registered",
    "a2a_push_notification_received",
    "a2a_push_notification_sent",
    "a2a_push_notification_timeout",
    "a2a_streaming_started",
    "a2a_streaming_chunk",
    "a2a_agent_card_fetched",
    "a2a_authentication_failed",
    "a2a_artifact_received",
    "a2a_connection_error",
    "a2a_server_task_started",
    "a2a_server_task_completed",
    "a2a_server_task_canceled",
    "a2a_server_task_failed",
    "a2a_parallel_delegation_started",
    "a2a_parallel_delegation_completed",
    "a2a_transport_negotiated",
    "a2a_content_type_negotiated",
    "a2a_context_created",
    "a2a_context_expired",
    "a2a_context_idle",
    "a2a_context_completed",
    "a2a_context_pruned",
    # System
    "SIGTERM",
    "SIGINT",
    "SIGHUP",
    "SIGTSTP",
    "SIGCONT",
    # Env
    "cc_env",
    "codex_env",
    "cursor_env",
    "default_env",
]


def _coerce_checkpoint(v: Any) -> Any:
    """BeforeValidator for checkpoint fields on Crew/Flow/Agent.

    Converts True to CheckpointConfig and triggers handler registration.
    """
    if v is True:
        v = CheckpointConfig()
    if isinstance(v, CheckpointConfig):
        from crewai.state.checkpoint_listener import _ensure_handlers_registered

        _ensure_handlers_registered()
    return v


class CheckpointConfig(BaseModel):
    """Configuration for automatic checkpointing.

    When set on a Crew, Flow, or Agent, checkpoints are written
    automatically whenever the specified event(s) fire.
    """

    location: str = Field(
        default="./.checkpoints",
        description="Storage destination. For JsonProvider this is a directory "
        "path; for SqliteProvider it is a database file path.",
    )
    on_events: list[CheckpointEventType | Literal["*"]] = Field(
        default=["task_completed"],
        description="Event types that trigger a checkpoint write. "
        'Use ["*"] to checkpoint on every event.',
    )
    provider: Annotated[
        JsonProvider | SqliteProvider,
        Field(discriminator="provider_type"),
    ] = Field(
        default_factory=JsonProvider,
        description="Storage backend. Defaults to JsonProvider.",
    )
    max_checkpoints: int | None = Field(
        default=None,
        description="Maximum checkpoints to keep. Oldest are pruned after "
        "each write. None means keep all.",
    )

    @model_validator(mode="after")
    def _register_handlers(self) -> CheckpointConfig:
        from crewai.state.checkpoint_listener import _ensure_handlers_registered

        _ensure_handlers_registered()
        return self

    @property
    def trigger_all(self) -> bool:
        return "*" in self.on_events

    @property
    def trigger_events(self) -> set[str]:
        return set(self.on_events)
