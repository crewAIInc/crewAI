"""Telemetry spans for the NewAgent system."""

from __future__ import annotations

import logging
from typing import Any


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GAP-47: Module-level registry mapping agent IDs to telemetry instances.
# Event handlers can look up the correct telemetry instance by agent ID.
# ---------------------------------------------------------------------------

_active_agents: dict[str, NewAgentTelemetry] = {}


def register_agent(agent_id: str, telemetry: NewAgentTelemetry) -> None:
    """Register an agent's telemetry instance for event-handler lookup."""
    _active_agents[agent_id] = telemetry


def unregister_agent(agent_id: str) -> None:
    """Remove an agent's telemetry instance from the registry."""
    _active_agents.pop(agent_id, None)


def get_telemetry_for_agent(agent_id: str) -> NewAgentTelemetry | None:
    """Look up the telemetry instance for a given agent ID."""
    return _active_agents.get(agent_id)


class NewAgentTelemetry:
    """Wraps the Telemetry singleton with NewAgent-specific span methods."""

    def __init__(self, share_data: bool = False) -> None:
        self._telemetry: Any = None
        self._share_data: bool = share_data
        # GAP-123: Store open duration spans keyed by (agent_id, operation, detail)
        self._pending_spans: dict[str, Any] = {}
        # GAP-124: Agent fingerprint (set once via set_fingerprint)
        self._agent_fingerprint: str = ""
        try:
            from crewai.telemetry.telemetry import Telemetry

            self._telemetry = Telemetry()
        except Exception:
            pass

    def set_fingerprint(self, fingerprint: str) -> None:
        """GAP-124: Store the agent's config fingerprint for span decoration."""
        self._agent_fingerprint = fingerprint

    def _span_key(self, agent_id: str, operation: str, detail: str = "") -> str:
        return f"{agent_id}:{operation}:{detail}"

    def store_span(self, key: str, span: Any) -> None:
        """Store an open span for later retrieval by a completed handler."""
        if span is not None:
            self._pending_spans[key] = span

    def retrieve_span(self, key: str) -> Any:
        """Pop and return a previously stored span, or None."""
        return self._pending_spans.pop(key, None)

    def _should_share_data(self) -> bool:
        """Check if the current agent opts into sharing sensitive data."""
        return self._share_data

    def _safe(self, fn: str, **kwargs: Any) -> None:
        """Call a telemetry method safely, swallowing errors."""
        if self._telemetry is None:
            return
        try:
            method = getattr(self._telemetry, fn, None)
            if method:
                method(**kwargs)
        except Exception:
            pass

    def agent_created(
        self,
        agent_id: str,
        role: str,
        goal: str,
        llm: str = "",
        tools_count: int = 0,
        coworkers_count: int = 0,
        memory_enabled: bool = True,
        planning_enabled: bool = True,
        # GAP-64: Additional metadata counts
        coworker_amp_count: int = 0,
        mcp_count: int = 0,
        apps_count: int = 0,
        knowledge_source_count: int = 0,
        tool_count: int = 0,
        **extra: Any,
    ) -> None:
        if self._telemetry is None:
            return
        try:
            import sys

            tracer = self._telemetry._tracer
            span = tracer.start_span("NewAgent Created")
            if span:
                # GAP-107: Include crewai_version and python_version
                try:
                    import crewai as _crewai_mod

                    span.set_attribute(
                        "crewai_version", getattr(_crewai_mod, "__version__", "unknown")
                    )
                except Exception:
                    span.set_attribute("crewai_version", "unknown")
                span.set_attribute("python_version", sys.version.split()[0])

                span.set_attribute("new_agent_id", agent_id)
                span.set_attribute("new_agent_role", role)
                # GAP-124: Agent fingerprint
                if self._agent_fingerprint:
                    span.set_attribute("agent_fingerprint", self._agent_fingerprint)
                # GAP-109: Only include goal when share_data is True
                if self._should_share_data():
                    span.set_attribute("new_agent_goal", goal)
                span.set_attribute("new_agent_llm", llm)
                span.set_attribute("new_agent_tools_count", tools_count)
                span.set_attribute("new_agent_coworkers_count", coworkers_count)
                span.set_attribute("new_agent_memory_enabled", memory_enabled)
                span.set_attribute("new_agent_planning_enabled", planning_enabled)
                # GAP-64: Metadata counts
                span.set_attribute("new_agent_coworker_amp_count", coworker_amp_count)
                span.set_attribute("new_agent_mcp_count", mcp_count)
                span.set_attribute("new_agent_apps_count", apps_count)
                span.set_attribute(
                    "new_agent_knowledge_source_count", knowledge_source_count
                )
                span.set_attribute("new_agent_tool_count", tool_count)
                # GAP-107: Forward extra keyword args as span attributes
                for key, val in extra.items():
                    span.set_attribute(key, str(val) if val is not None else "")
                tracer.end_span(span)
        except Exception:
            pass

    def execution_started(
        self, agent_id: str, conversation_id: str, model: str = ""
    ) -> Any:
        if self._telemetry is None:
            return None
        try:
            tracer = self._telemetry._tracer
            span = tracer.start_span("NewAgent Execution")
            if span:
                span.set_attribute("new_agent_id", agent_id)
                span.set_attribute("conversation_id", conversation_id)
                span.set_attribute("model", model)
                if self._agent_fingerprint:
                    span.set_attribute("agent_fingerprint", self._agent_fingerprint)
            return span
        except Exception:
            return None

    def execution_completed(
        self,
        span: Any,
        input_tokens: int = 0,
        output_tokens: int = 0,
        response_time_ms: int = 0,
    ) -> None:
        if span is None or self._telemetry is None:
            return
        try:
            tracer = self._telemetry._tracer
            span.set_attribute("input_tokens", input_tokens)
            span.set_attribute("output_tokens", output_tokens)
            span.set_attribute("response_time_ms", response_time_ms)
            tracer.end_span(span)
        except Exception:
            pass

    def tool_usage(self, agent_id: str, tool_name: str) -> Any:
        if self._telemetry is None:
            return None
        try:
            tracer = self._telemetry._tracer
            span = tracer.start_span("NewAgent Tool Usage")
            if span:
                span.set_attribute("new_agent_id", agent_id)
                span.set_attribute("tool_name", tool_name)
            return span
        except Exception:
            return None

    def tool_usage_error(self, span: Any, error: str = "") -> None:
        if span is None or self._telemetry is None:
            return
        try:
            tracer = self._telemetry._tracer
            span.set_attribute("error", error)
            tracer.end_span(span)
        except Exception:
            pass

    def tool_usage_completed(self, span: Any) -> None:
        if span is None or self._telemetry is None:
            return
        try:
            tracer = self._telemetry._tracer
            tracer.end_span(span)
        except Exception:
            pass

    def delegation(
        self,
        agent_id: str,
        coworker_role: str,
        mode: str = "sync",
        source: str = "local",
    ) -> Any:
        if self._telemetry is None:
            return None
        try:
            tracer = self._telemetry._tracer
            span = tracer.start_span("NewAgent Delegation")
            if span:
                span.set_attribute("new_agent_id", agent_id)
                span.set_attribute("coworker_role", coworker_role)
                span.set_attribute("delegation_mode", mode)
                span.set_attribute("coworker_source", source)
            return span
        except Exception:
            return None

    def delegation_completed(
        self, span: Any, tokens_consumed: int = 0, response_time_ms: int = 0
    ) -> None:
        if span is None or self._telemetry is None:
            return
        try:
            tracer = self._telemetry._tracer
            span.set_attribute("tokens_consumed", tokens_consumed)
            span.set_attribute("response_time_ms", response_time_ms)
            tracer.end_span(span)
        except Exception:
            pass

    def spawn(self, agent_id: str, spawn_id: str, depth: int = 0) -> Any:
        if self._telemetry is None:
            return None
        try:
            tracer = self._telemetry._tracer
            span = tracer.start_span("NewAgent Spawn")
            if span:
                span.set_attribute("new_agent_id", agent_id)
                span.set_attribute("spawn_id", spawn_id)
                span.set_attribute("spawn_depth", depth)
            return span
        except Exception:
            return None

    def spawn_completed(self, span: Any) -> None:
        if span is None or self._telemetry is None:
            return
        try:
            tracer = self._telemetry._tracer
            tracer.end_span(span)
        except Exception:
            pass

    def spawn_completed_event(self, agent_id: str, spawn_id: str = "") -> None:
        """GAP-123: Point span for spawn completion, used by event listener."""
        if self._telemetry is None:
            return
        try:
            tracer = self._telemetry._tracer
            span = tracer.start_span("NewAgent Spawn Completed")
            if span:
                span.set_attribute("new_agent_id", agent_id)
                span.set_attribute("spawn_id", spawn_id)
                tracer.end_span(span)
        except Exception:
            pass

    def dreaming(self, agent_id: str) -> Any:
        if self._telemetry is None:
            return None
        try:
            tracer = self._telemetry._tracer
            span = tracer.start_span("NewAgent Dreaming")
            if span:
                span.set_attribute("new_agent_id", agent_id)
            return span
        except Exception:
            return None

    def dreaming_completed(
        self, span: Any, memories_processed: int = 0, canonical_created: int = 0
    ) -> None:
        if span is None or self._telemetry is None:
            return
        try:
            tracer = self._telemetry._tracer
            span.set_attribute("memories_processed", memories_processed)
            span.set_attribute("canonical_created", canonical_created)
            tracer.end_span(span)
        except Exception:
            pass

    def planning(self, agent_id: str) -> Any:
        if self._telemetry is None:
            return None
        try:
            tracer = self._telemetry._tracer
            span = tracer.start_span("NewAgent Planning")
            if span:
                span.set_attribute("new_agent_id", agent_id)
            return span
        except Exception:
            return None

    def planning_completed(self, span: Any, steps_count: int = 0) -> None:
        if span is None or self._telemetry is None:
            return
        try:
            tracer = self._telemetry._tracer
            span.set_attribute("plan_steps_count", steps_count)
            tracer.end_span(span)
        except Exception:
            pass

    def guardrail(self, agent_id: str, guardrail_type: str = "") -> Any:
        if self._telemetry is None:
            return None
        try:
            tracer = self._telemetry._tracer
            span = tracer.start_span("NewAgent Guardrail")
            if span:
                span.set_attribute("new_agent_id", agent_id)
                span.set_attribute("guardrail_type", guardrail_type)
            return span
        except Exception:
            return None

    def guardrail_completed(self, span: Any, passed: bool = True) -> None:
        if span is None or self._telemetry is None:
            return
        try:
            tracer = self._telemetry._tracer
            span.set_attribute("guardrail_passed", passed)
            tracer.end_span(span)
        except Exception:
            pass

    def memory_save(self, agent_id: str) -> None:
        if self._telemetry is None:
            return
        try:
            tracer = self._telemetry._tracer
            span = tracer.start_span("NewAgent Memory Save")
            if span:
                span.set_attribute("new_agent_id", agent_id)
                tracer.end_span(span)
        except Exception:
            pass

    def memory_recall(self, agent_id: str, results_count: int = 0) -> None:
        if self._telemetry is None:
            return
        try:
            tracer = self._telemetry._tracer
            span = tracer.start_span("NewAgent Memory Recall")
            if span:
                span.set_attribute("new_agent_id", agent_id)
                span.set_attribute("results_count", results_count)
                tracer.end_span(span)
        except Exception:
            pass

    def knowledge_suggested(self, agent_id: str, source_type: str = "") -> None:
        if self._telemetry is None:
            return
        try:
            tracer = self._telemetry._tracer
            span = tracer.start_span("NewAgent Knowledge Suggested")
            if span:
                span.set_attribute("new_agent_id", agent_id)
                span.set_attribute("source_type", source_type)
                tracer.end_span(span)
        except Exception:
            pass

    # ── Additional span methods for GAP-47 / GAP-61 bridge ──────

    def conversation_reset(self, agent_id: str) -> None:
        if self._telemetry is None:
            return
        try:
            tracer = self._telemetry._tracer
            span = tracer.start_span("NewAgent Conversation Reset")
            if span:
                span.set_attribute("new_agent_id", agent_id)
                tracer.end_span(span)
        except Exception:
            pass

    def message_received(self, agent_id: str, message_length: int = 0) -> None:
        if self._telemetry is None:
            return
        try:
            tracer = self._telemetry._tracer
            span = tracer.start_span("NewAgent Message Received")
            if span:
                span.set_attribute("new_agent_id", agent_id)
                span.set_attribute("message_length", message_length)
                tracer.end_span(span)
        except Exception:
            pass

    def message_sent(
        self,
        agent_id: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        response_time_ms: int = 0,
    ) -> None:
        if self._telemetry is None:
            return
        try:
            tracer = self._telemetry._tracer
            span = tracer.start_span("NewAgent Message Sent")
            if span:
                span.set_attribute("new_agent_id", agent_id)
                span.set_attribute("input_tokens", input_tokens)
                span.set_attribute("output_tokens", output_tokens)
                span.set_attribute("response_time_ms", response_time_ms)
                tracer.end_span(span)
        except Exception:
            pass

    def llm_call_started(self, agent_id: str, model: str = "") -> None:
        if self._telemetry is None:
            return
        try:
            tracer = self._telemetry._tracer
            span = tracer.start_span("NewAgent LLM Call Started")
            if span:
                span.set_attribute("new_agent_id", agent_id)
                span.set_attribute("model", model)
                tracer.end_span(span)
        except Exception:
            pass

    def llm_call_completed(
        self,
        agent_id: str,
        model: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        response_time_ms: int = 0,
    ) -> None:
        if self._telemetry is None:
            return
        try:
            tracer = self._telemetry._tracer
            span = tracer.start_span("NewAgent LLM Call Completed")
            if span:
                span.set_attribute("new_agent_id", agent_id)
                span.set_attribute("model", model)
                span.set_attribute("input_tokens", input_tokens)
                span.set_attribute("output_tokens", output_tokens)
                span.set_attribute("response_time_ms", response_time_ms)
                tracer.end_span(span)
        except Exception:
            pass

    def llm_call_failed(self, agent_id: str, error: str = "") -> None:
        if self._telemetry is None:
            return
        try:
            tracer = self._telemetry._tracer
            span = tracer.start_span("NewAgent LLM Call Failed")
            if span:
                span.set_attribute("new_agent_id", agent_id)
                span.set_attribute("error", error)
                tracer.end_span(span)
        except Exception:
            pass

    def tool_usage_started(self, agent_id: str, tool_name: str = "") -> None:
        if self._telemetry is None:
            return
        try:
            tracer = self._telemetry._tracer
            span = tracer.start_span("NewAgent Tool Usage Started")
            if span:
                span.set_attribute("new_agent_id", agent_id)
                span.set_attribute("tool_name", tool_name)
                tracer.end_span(span)
        except Exception:
            pass

    def tool_usage_completed_event(self, agent_id: str, tool_name: str = "") -> None:
        """GAP-123: Point span for tool completion, used by event listener."""
        if self._telemetry is None:
            return
        try:
            tracer = self._telemetry._tracer
            span = tracer.start_span("NewAgent Tool Usage Completed")
            if span:
                span.set_attribute("new_agent_id", agent_id)
                span.set_attribute("tool_name", tool_name)
                if self._agent_fingerprint:
                    span.set_attribute("agent_fingerprint", self._agent_fingerprint)
                tracer.end_span(span)
        except Exception:
            pass

    def tool_usage_failed(
        self, agent_id: str, tool_name: str = "", error: str = ""
    ) -> None:
        if self._telemetry is None:
            return
        try:
            tracer = self._telemetry._tracer
            span = tracer.start_span("NewAgent Tool Usage Failed")
            if span:
                span.set_attribute("new_agent_id", agent_id)
                span.set_attribute("tool_name", tool_name)
                span.set_attribute("error", error)
                tracer.end_span(span)
        except Exception:
            pass

    def delegation_failed(
        self, agent_id: str, coworker_role: str = "", error: str = ""
    ) -> None:
        if self._telemetry is None:
            return
        try:
            tracer = self._telemetry._tracer
            span = tracer.start_span("NewAgent Delegation Failed")
            if span:
                span.set_attribute("new_agent_id", agent_id)
                span.set_attribute("coworker_role", coworker_role)
                span.set_attribute("error", error)
                tracer.end_span(span)
        except Exception:
            pass

    def fire_and_forget_dispatched(
        self, agent_id: str, coworker_role: str = ""
    ) -> None:
        if self._telemetry is None:
            return
        try:
            tracer = self._telemetry._tracer
            span = tracer.start_span("NewAgent Fire And Forget Dispatched")
            if span:
                span.set_attribute("new_agent_id", agent_id)
                span.set_attribute("coworker_role", coworker_role)
                tracer.end_span(span)
        except Exception:
            pass

    def fire_and_forget_completed(self, agent_id: str, coworker_role: str = "") -> None:
        if self._telemetry is None:
            return
        try:
            tracer = self._telemetry._tracer
            span = tracer.start_span("NewAgent Fire And Forget Completed")
            if span:
                span.set_attribute("new_agent_id", agent_id)
                span.set_attribute("coworker_role", coworker_role)
                tracer.end_span(span)
        except Exception:
            pass

    def spawn_failed(self, agent_id: str, spawn_id: str = "", error: str = "") -> None:
        if self._telemetry is None:
            return
        try:
            tracer = self._telemetry._tracer
            span = tracer.start_span("NewAgent Spawn Failed")
            if span:
                span.set_attribute("new_agent_id", agent_id)
                span.set_attribute("spawn_id", spawn_id)
                span.set_attribute("error", error)
                tracer.end_span(span)
        except Exception:
            pass

    def context_summarized(self, agent_id: str) -> None:
        if self._telemetry is None:
            return
        try:
            tracer = self._telemetry._tracer
            span = tracer.start_span("NewAgent Context Summarized")
            if span:
                span.set_attribute("new_agent_id", agent_id)
                tracer.end_span(span)
        except Exception:
            pass

    def narration_guard_triggered(self, agent_id: str, retries: int = 0) -> None:
        if self._telemetry is None:
            return
        try:
            tracer = self._telemetry._tracer
            span = tracer.start_span("NewAgent Narration Guard Triggered")
            if span:
                span.set_attribute("new_agent_id", agent_id)
                span.set_attribute("retries", retries)
                tracer.end_span(span)
        except Exception:
            pass

    def workflow_detected(
        self, agent_id: str, tools: list[str] | None = None, count: int = 0
    ) -> None:
        if self._telemetry is None:
            return
        try:
            tracer = self._telemetry._tracer
            span = tracer.start_span("NewAgent Workflow Detected")
            if span:
                span.set_attribute("new_agent_id", agent_id)
                span.set_attribute("workflow_tools", ",".join(tools or []))
                span.set_attribute("workflow_count", count)
                tracer.end_span(span)
        except Exception:
            pass

    def workflow_proposed(self, agent_id: str, description: str = "") -> None:
        if self._telemetry is None:
            return
        try:
            tracer = self._telemetry._tracer
            span = tracer.start_span("NewAgent Workflow Proposed")
            if span:
                span.set_attribute("new_agent_id", agent_id)
                span.set_attribute("workflow_description", description[:500])
                tracer.end_span(span)
        except Exception:
            pass

    def workflow_confirmed(self, agent_id: str) -> None:
        if self._telemetry is None:
            return
        try:
            tracer = self._telemetry._tracer
            span = tracer.start_span("NewAgent Workflow Confirmed")
            if span:
                span.set_attribute("new_agent_id", agent_id)
                tracer.end_span(span)
        except Exception:
            pass

    def knowledge_query(self, agent_id: str) -> None:
        if self._telemetry is None:
            return
        try:
            tracer = self._telemetry._tracer
            span = tracer.start_span("NewAgent Knowledge Query")
            if span:
                span.set_attribute("new_agent_id", agent_id)
                tracer.end_span(span)
        except Exception:
            pass

    def knowledge_confirmed(self, agent_id: str, source_type: str = "") -> None:
        if self._telemetry is None:
            return
        try:
            tracer = self._telemetry._tracer
            span = tracer.start_span("NewAgent Knowledge Confirmed")
            if span:
                span.set_attribute("new_agent_id", agent_id)
                span.set_attribute("source_type", source_type)
                tracer.end_span(span)
        except Exception:
            pass

    def knowledge_rejected(self, agent_id: str) -> None:
        if self._telemetry is None:
            return
        try:
            tracer = self._telemetry._tracer
            span = tracer.start_span("NewAgent Knowledge Rejected")
            if span:
                span.set_attribute("new_agent_id", agent_id)
                tracer.end_span(span)
        except Exception:
            pass

    def explain_requested(self, agent_id: str) -> None:
        if self._telemetry is None:
            return
        try:
            tracer = self._telemetry._tracer
            span = tracer.start_span("NewAgent Explain Requested")
            if span:
                span.set_attribute("new_agent_id", agent_id)
                tracer.end_span(span)
        except Exception:
            pass

    def guardrail_passed(self, agent_id: str, guardrail_type: str = "") -> None:
        if self._telemetry is None:
            return
        try:
            tracer = self._telemetry._tracer
            span = tracer.start_span("NewAgent Guardrail Passed")
            if span:
                span.set_attribute("new_agent_id", agent_id)
                span.set_attribute("guardrail_type", guardrail_type)
                tracer.end_span(span)
        except Exception:
            pass

    def status_update(self, state: str = "", detail: str = "") -> None:
        if self._telemetry is None:
            return
        try:
            tracer = self._telemetry._tracer
            span = tracer.start_span("NewAgent Status Update")
            if span:
                span.set_attribute("state", state)
                span.set_attribute("detail", detail or "")
                tracer.end_span(span)
        except Exception:
            pass
