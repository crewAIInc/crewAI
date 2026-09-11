from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
import json
import logging
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse


if TYPE_CHECKING:
    from crewai.flow.flow import Flow

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import (
    Link,
    NonRecordingSpan,
    Span,
    SpanContext,
    SpanKind,
    Status,
    StatusCode,
    TraceFlags,
    get_current_span,
    set_span_in_context,
)

from crewai.events import (
    AgentExecutionCompletedEvent,
    AgentExecutionErrorEvent,
    AgentExecutionStartedEvent,
    AgentReasoningCompletedEvent,
    AgentReasoningFailedEvent,
    AgentReasoningStartedEvent,
    CrewKickoffCompletedEvent,
    CrewKickoffFailedEvent,
    CrewKickoffStartedEvent,
    FlowCreatedEvent,
    FlowFailedEvent,
    FlowFinishedEvent,
    FlowStartedEvent,
    HumanFeedbackReceivedEvent,
    HumanFeedbackRequestedEvent,
    KnowledgeQueryCompletedEvent,
    KnowledgeQueryFailedEvent,
    KnowledgeQueryStartedEvent,
    KnowledgeRetrievalCompletedEvent,
    KnowledgeRetrievalStartedEvent,
    LLMCallCompletedEvent,
    LLMCallFailedEvent,
    LLMCallStartedEvent,
    MCPConnectionCompletedEvent,
    MCPConnectionFailedEvent,
    MCPConnectionStartedEvent,
    MCPToolExecutionCompletedEvent,
    MCPToolExecutionFailedEvent,
    MCPToolExecutionStartedEvent,
    MemoryQueryCompletedEvent,
    MemoryQueryFailedEvent,
    MemoryQueryStartedEvent,
    MemoryRetrievalCompletedEvent,
    MemoryRetrievalFailedEvent,
    MemoryRetrievalStartedEvent,
    MemorySaveCompletedEvent,
    MemorySaveFailedEvent,
    MemorySaveStartedEvent,
    MethodExecutionFailedEvent,
    MethodExecutionFinishedEvent,
    MethodExecutionStartedEvent,
    TaskCompletedEvent,
    TaskFailedEvent,
    TaskStartedEvent,
    ToolFailureDetectedEvent,
    ToolUsageErrorEvent,
    ToolUsageFinishedEvent,
    ToolUsageStartedEvent,
)
from crewai.events.types.a2a_events import (
    A2AConversationCompletedEvent,
    A2AConversationStartedEvent,
    A2ADelegationCompletedEvent,
    A2ADelegationStartedEvent,
    A2AParallelDelegationCompletedEvent,
    A2AParallelDelegationStartedEvent,
    A2AServerTaskCanceledEvent,
    A2AServerTaskCompletedEvent,
    A2AServerTaskFailedEvent,
    A2AServerTaskStartedEvent,
)
from crewai.events.types.agent_events import (
    LiteAgentExecutionCompletedEvent,
    LiteAgentExecutionErrorEvent,
    LiteAgentExecutionStartedEvent,
)
from crewai.events.types.flow_events import (
    FlowPausedEvent,
    MethodExecutionPausedEvent,
)
from crewai.events.types.knowledge_events import KnowledgeSearchQueryFailedEvent
from crewai.events.types.llm_guardrail_events import (
    LLMGuardrailCompletedEvent,
    LLMGuardrailStartedEvent,
)
from crewai.events.types.skill_events import (
    SkillActivatedEvent,
    SkillDiscoveryCompletedEvent,
    SkillDiscoveryStartedEvent,
    SkillLoadFailedEvent,
    SkillLoadedEvent,
    SkillUsedEvent,
)
from crewai.telemetry.tracing import semantic_conventions
from crewai.telemetry.tracing.context import PendingSpanEnd, TelemetryExecutionContext
from crewai.telemetry.tracing.session import TraceSession as TelemetryProviders
from crewai.utilities.serialization import to_serializable


logger = logging.getLogger(__name__)

UNKNOWN_MODEL = "unknown_model"


def _datetime_to_nanoseconds(dt: datetime) -> int:
    return int(dt.timestamp() * 1_000_000_000)


def _serialize(value: Any) -> str:
    return json.dumps(to_serializable(value))


def _set_span_attributes(span: Span, attributes: dict[str, Any]) -> None:
    for key, value in attributes.items():
        if value is not None:
            span.set_attribute(key, value)


def _get_parent_context(
    ctx: TelemetryExecutionContext, parent_event_id: str | None
) -> tuple[Any, bool]:
    """Resolve the OTel parent context for a child event.

    Returns ``(parent_context, used_root_fallback)``. ``used_root_fallback``
    is True only when the child's real parent span did not register
    and the child was attached to the execution root instead: the trace stays
    intact but the parent_span_id link is approximate.
    """
    if not parent_event_id:
        if ctx.root_span is not None:
            return set_span_in_context(ctx.root_span), False
        return None, False

    parent_span = ctx._span_refs.get(parent_event_id)
    if parent_span:
        return set_span_in_context(parent_span), False

    # Handlers run inline before emit returns, so waiting here cannot make a
    # missing parent appear. Keep the child in the execution's trace instead.
    if ctx.root_span is not None:
        return set_span_in_context(ctx.root_span), True
    return None, False


#: Link attribute (and value) marking a span as the OTel FOLLOWS_FROM
#: continuation of a previous execution segment — e.g. the resume side of a HITL
#: pause/resume. Wharf stores the link and crewai-plus groups segments by it.
LINK_TYPE_ATTRIBUTE = "crewai.link.type"
LINK_TYPE_FOLLOWS_FROM = "follows_from"


def _build_follows_from_link(otel_context: tuple[int, int]) -> Link:
    """Build a FOLLOWS_FROM span link to a previous segment's flow span.

    Each resume segment is its own trace root linked back to the segment that
    paused, rather than a *child* of it: a parent edge would imply the resume
    work is temporally contained within the previous segment, which the
    pause/resume gap (often minutes or hours) violates — producing the
    out-of-order, misnested flame graphs this replaces.
    """
    trace_id, span_id = otel_context
    return Link(
        SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            is_remote=True,
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
        ),
        attributes={LINK_TYPE_ATTRIBUTE: LINK_TYPE_FOLLOWS_FROM},
    )


def _start_span(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    name: str,
    event: Any,
    attributes: dict[str, Any],
    kind: Any = None,
    links: list[Link] | None = None,
) -> Span | None:
    try:
        tracer = providers.get_tracer()
        current = get_current_span()
        reuse_operation = (
            current.get_span_context().span_id in ctx.operation_spans
            and getattr(current, "name", None) == name
            and current not in ctx._span_refs.values()
        )
        parent_ctx, used_root_fallback = (
            (None, False)
            if reuse_operation
            else _get_parent_context(ctx, event.parent_event_id)
        )
        start_time_ns = _datetime_to_nanoseconds(event.timestamp)
        start_kwargs: dict[str, Any] = {
            "context": parent_ctx,
            "start_time": start_time_ns,
        }
        if kind is not None:
            start_kwargs["kind"] = kind
        if links:
            start_kwargs["links"] = links
        if reuse_operation:
            span = current
            for link in links or []:
                span.add_link(link.context, link.attributes)
        else:
            span = tracer.start_span(name, **start_kwargs)
        _set_span_attributes(span, attributes)
        span.set_attribute("event_id", event.event_id)
        if event.parent_event_id:
            span.set_attribute("parent_event_id", event.parent_event_id)
        if used_root_fallback:
            span.set_attribute("parent_span_approximate", True)
            logger.warning(
                "Parent span for '%s' (event_id=%s, parent_event_id=%s) did not "
                "register; attached to the execution root.",
                name,
                event.event_id,
                event.parent_event_id,
            )
        return span
    except Exception as e:
        logger.debug(f"Failed to start span '{name}': {e}")
        return None


def _get_span_duration_ms(span: Span, end_event: Any) -> float | None:
    if isinstance(span, ReadableSpan) and span.start_time is not None:
        end_ns = _datetime_to_nanoseconds(end_event.timestamp)
        return (end_ns - span.start_time) / 1_000_000
    return None


def _apply_error(
    span: Span, error: str | BaseException | None, attributes: dict[str, Any]
) -> Status:
    """Derive OTEL status from *error* and set standard error attributes on *span*.

    When *error* is an exception object, ``error.type`` is set to the class name
    and ``span.record_exception()`` records the full traceback.  When *error* is a
    plain string (the exception was already serialised before reaching us),
    ``error.type`` falls back to ``_OTHER``.
    """
    if error is None:
        return Status(StatusCode.OK)

    if isinstance(error, BaseException):
        attributes["error.type"] = type(error).__qualname__
        span.record_exception(error)
    else:
        attributes["error.type"] = "_OTHER"

    return Status(StatusCode.ERROR, str(error))


def _store_span(ctx: TelemetryExecutionContext, event_id: str, span: Span) -> None:
    """Store a span and apply any completion emitted before its start."""
    with ctx._span_lock:
        ctx.active_spans[event_id] = span
        ctx._span_refs[event_id] = span
        if ctx.root_span is None:
            ctx.root_span = span
        pending = ctx.pending_span_ends.pop(event_id, None)
    if pending:
        ctx.active_spans.pop(event_id, None)
        _finish_span(ctx, span, pending)


def _finish_span(
    ctx: TelemetryExecutionContext, span: Span, end: PendingSpanEnd
) -> None:
    if end.duration_attr and end.end_event:
        end.attributes[end.duration_attr] = _get_span_duration_ms(span, end.end_event)
    status = _apply_error(span, end.error, end.attributes)
    _set_span_attributes(span, end.attributes)
    span.set_status(status)
    span_id = span.get_span_context().span_id
    if span_id in ctx.operation_spans:
        ctx.completed_operations[span_id] = end.end_time_ns
    else:
        span.end(end_time=end.end_time_ns)


def _end_span(
    ctx: TelemetryExecutionContext,
    event_id: str | None,
    event: Any,
    attributes: dict[str, Any],
    error: str | BaseException | None = None,
    duration_attr: str | None = None,
) -> Span | None:
    if not event_id:
        return None

    end = PendingSpanEnd(
        attributes,
        error,
        _datetime_to_nanoseconds(event.timestamp),
        duration_attr,
        event,
    )

    with ctx._span_lock:
        span = ctx.active_spans.pop(event_id, None)
        if not span:
            ctx.pending_span_ends[event_id] = end
            return None
    _finish_span(ctx, span, end)
    return span


def _agent_llm_model(llm: Any) -> str:
    if hasattr(llm, "model"):
        return str(llm.model)
    return str(llm)


def _llm_provider_name(llm: Any) -> str | None:
    if hasattr(llm, "provider"):
        return str(llm.provider)
    return None


def _record_agent_llm_call(
    ctx: TelemetryExecutionContext, agent_id: str | None
) -> None:
    if not agent_id:
        return

    agent_key = f"{ctx.kickoff_id}::{agent_id}"
    with ctx._span_lock:
        ctx.agent_llm_call_counts[agent_key] = (
            ctx.agent_llm_call_counts.get(agent_key, 0) + 1
        )


def _conversation_id(event: Any, ctx: TelemetryExecutionContext) -> str | None:
    """Resolve gen_ai.conversation.id with a strict fallback chain.

    task_id (most specific) -> agent_id (lite-agent) -> kickoff_id (session-wide).
    Returns None when there is no active session at all (direct LLM.call outside
    any crew/flow context); the attribute is then omitted per spec.

    `agent_info["id"]` on the OSS side is a UUID object (LiteAgent.id is UUID4),
    and OTel attribute values must be primitive types — coerce to str at the
    boundary so the attribute isn't silently dropped by the SDK.
    """
    task_id = getattr(event, "task_id", None)
    agent_id = getattr(event, "agent_id", None)
    if not agent_id:
        info = getattr(event, "agent_info", None)
        if isinstance(info, dict):
            agent_id = info.get("id")
    resolved = task_id or agent_id or ctx.kickoff_id
    return str(resolved) if resolved is not None else None


def handle_crew_kickoff_started(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    crew: Any,
    event: CrewKickoffStartedEvent,
) -> None:
    serialized_inputs = _serialize(event.inputs)
    attrs = {
        **semantic_conventions.crewai_span(event_name=event.type, subject=crew.name),
        **semantic_conventions.crewai_crew(
            key=crew.key,
            name=crew.name,
            id=str(crew.id),
            inputs=serialized_inputs,
            process=str(crew.process),
            num_tasks=len(crew.tasks),
            num_agents=len(crew.agents),
        ),
        **semantic_conventions.gen_ai(
            operation_name=semantic_conventions.GEN_AI_OP_INVOKE_WORKFLOW,
            workflow_name=crew.name,
        ),
        **semantic_conventions.gen_ai_io(input_value=serialized_inputs),
    }

    span = _start_span(providers, ctx, "execute crew", event, attrs)
    if span:
        _store_span(ctx, event.event_id, span)

    providers.emit_log(
        f"Crew execution started: {crew.key}",
        level="INFO",
        span=span,
        attributes=attrs,
        ctx=ctx,
    )


def handle_task_started(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    task: Any,
    event: TaskStartedEvent,
) -> None:
    if not task.agent or not task.agent.crew:
        return
    crew = task.agent.crew

    attrs = {
        **semantic_conventions.crewai_span(
            event_name=event.type, subject=task.name or task.key
        ),
        **semantic_conventions.crewai_crew(
            key=crew.key, name=crew.name, id=str(crew.id)
        ),
        **semantic_conventions.crewai_task(
            key=task.key,
            id=str(task.id),
            name=task.name,
            description=task.description,
            expected_output=task.expected_output,
        ),
        **semantic_conventions.gen_ai(
            operation_name=semantic_conventions.GEN_AI_OP_EXECUTE_TASK,
        ),
        **semantic_conventions.gen_ai_io(input_value=task.description),
    }

    span = _start_span(providers, ctx, "execute task", event, attrs)
    if span:
        _store_span(ctx, event.event_id, span)

    providers.emit_log(
        f"Task started: {task.key}",
        level="INFO",
        span=span,
        attributes=attrs,
        ctx=ctx,
    )


def handle_task_completed(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    task: Any,
    event: TaskCompletedEvent,
) -> None:
    if not task.agent or not task.agent.crew:
        return
    crew = task.agent.crew
    span = (
        ctx.active_spans.get(event.started_event_id) if event.started_event_id else None
    )

    attrs: dict[str, Any] = {
        **semantic_conventions.crewai_span(
            event_name=event.type, subject=task.name or task.key
        ),
        **semantic_conventions.crewai_crew(
            key=crew.key, name=crew.name, id=str(crew.id)
        ),
        **semantic_conventions.crewai_task(
            key=task.key,
            id=str(task.id),
            name=task.name,
            output=task.output.raw if task.output else None,
        ),
        **semantic_conventions.gen_ai_io(
            output_value=task.output.raw if task.output else None
        ),
    }

    providers.emit_log(
        f"Task completed: {task.key}",
        level="INFO",
        span=span,
        attributes=attrs,
        ctx=ctx,
    )

    _end_span(ctx, event.started_event_id, event, attrs)


def handle_task_failed(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    task: Any,
    event: TaskFailedEvent,
) -> None:
    if not task.agent or not task.agent.crew:
        return
    crew = task.agent.crew

    attrs = {
        **semantic_conventions.crewai_span(
            event_name=event.type, subject=task.name or task.key
        ),
        **semantic_conventions.crewai_crew(
            key=crew.key, name=crew.name, id=str(crew.id)
        ),
        **semantic_conventions.crewai_task(
            key=task.key,
            id=str(task.id),
            name=task.name,
            description=task.description,
            expected_output=task.expected_output,
        ),
    }

    span = _end_span(ctx, event.started_event_id, event, attrs, error=event.error)

    providers.emit_log(
        f"Task failed: {task.key}",
        level="ERROR",
        span=span,
        attributes=attrs,
        ctx=ctx,
    )


def handle_crew_kickoff_failed(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    crew: Any,
    event: CrewKickoffFailedEvent,
) -> None:
    attrs = {
        **semantic_conventions.crewai_span(event_name=event.type, subject=crew.name),
        **semantic_conventions.crewai_crew(
            key=crew.key, name=crew.name, id=str(crew.id)
        ),
    }

    span = _end_span(ctx, event.started_event_id, event, attrs, error=event.error)

    providers.emit_log(
        f"Crew execution failed: {crew.key}",
        level="ERROR",
        span=span,
        attributes=attrs,
        ctx=ctx,
    )


def handle_crew_kickoff_completed(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    crew: Any,
    event: CrewKickoffCompletedEvent,
) -> None:
    final_string_output = event.output.raw

    if ctx.flow_crew_usage_metrics:
        with ctx._span_lock:
            for flow_metrics in ctx.flow_crew_usage_metrics.values():
                flow_metrics["crew_count"] += 1

    attrs: dict[str, Any] = {
        **semantic_conventions.crewai_span(event_name=event.type, subject=crew.name),
        **semantic_conventions.crewai_crew(
            key=crew.key,
            name=crew.name,
            id=str(crew.id),
            output=final_string_output,
            usage_metrics=_serialize(crew.usage_metrics),
        ),
        **semantic_conventions.gen_ai_io(output_value=final_string_output),
    }

    providers.emit_log(
        f"Crew execution completed: {crew.key}",
        level="INFO",
        attributes=attrs,
        ctx=ctx,
    )

    _end_span(
        ctx,
        event.started_event_id,
        event,
        attrs,
        duration_attr=semantic_conventions.CREW_EXECUTION_DURATION_MS,
    )


def handle_agent_execution_started(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    source: Any,
    event: AgentExecutionStartedEvent,
) -> None:
    agent = event.agent
    agent_id = str(getattr(agent, "id", ""))
    llm_model = _agent_llm_model(agent.llm)
    provider_name = _llm_provider_name(agent.llm)

    attrs: dict[str, Any] = {
        **semantic_conventions.gen_ai(
            operation_name="invoke_agent",
            agent_name=agent.key,
            agent_id=agent_id,
            agent_description=agent.goal,
            request_model=llm_model,
            provider_name=provider_name,
            tool_definitions=agent.tools,
            system_instructions=agent.backstory,
        ),
        **semantic_conventions.crewai_span(event_name=event.type, subject=agent.role),
        **semantic_conventions.crewai_agent(role=agent.role),
    }

    if (
        hasattr(agent, "crew")
        and agent.crew is not None
        and not isinstance(agent.crew, str)
    ):
        attrs.update(
            semantic_conventions.crewai_crew(
                name=agent.crew.name, id=str(agent.crew.id)
            )
        )

    span = _start_span(providers, ctx, "execute agent", event, attrs)
    if span:
        _store_span(ctx, event.event_id, span)

    providers.emit_log(
        f"Agent execution started: {agent.key}",
        level="INFO",
        span=span,
        attributes=attrs,
        ctx=ctx,
    )


def handle_agent_execution_completed(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    source: Any,
    event: AgentExecutionCompletedEvent,
) -> None:
    agent = event.agent
    agent_id = str(getattr(agent, "id", ""))
    llm_model = _agent_llm_model(agent.llm)
    provider_name = _llm_provider_name(agent.llm)
    agent_timing_key = f"{ctx.kickoff_id}::{agent_id}"

    attrs: dict[str, Any] = {
        **semantic_conventions.gen_ai(
            operation_name="invoke_agent",
            agent_name=agent.key,
            agent_id=agent_id,
            agent_description=agent.goal,
            request_model=llm_model,
            provider_name=provider_name,
            tool_definitions=agent.tools,
            system_instructions=agent.backstory,
        ),
        **semantic_conventions.crewai_span(event_name=event.type, subject=agent.role),
        **semantic_conventions.crewai_agent(role=agent.role),
    }

    if (
        hasattr(agent, "crew")
        and agent.crew is not None
        and not isinstance(agent.crew, str)
    ):
        attrs.update(
            semantic_conventions.crewai_crew(
                name=agent.crew.name, id=str(agent.crew.id)
            )
        )

    if agent_id:
        with ctx._span_lock:
            attrs[semantic_conventions.AGENT_LLM_CALLS_COUNT] = (
                ctx.agent_llm_call_counts.pop(agent_timing_key, 0)
            )

    providers.emit_log(
        f"Agent execution completed: {agent.key}",
        level="INFO",
        attributes=attrs,
        ctx=ctx,
    )

    _end_span(
        ctx,
        event.started_event_id,
        event,
        attrs,
        duration_attr=semantic_conventions.AGENT_EXECUTION_DURATION_MS,
    )


def handle_agent_execution_error(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    source: Any,
    event: AgentExecutionErrorEvent,
) -> None:
    agent = event.agent
    agent_id = str(getattr(agent, "id", ""))
    llm_model = _agent_llm_model(agent.llm)
    provider_name = _llm_provider_name(agent.llm)

    attrs: dict[str, Any] = {
        **semantic_conventions.gen_ai(
            operation_name="invoke_agent",
            agent_name=agent.key,
            agent_id=agent_id,
            request_model=llm_model,
            provider_name=provider_name,
        ),
        **semantic_conventions.crewai_span(event_name=event.type, subject=agent.role),
        **semantic_conventions.crewai_agent(role=agent.role),
    }

    if (
        hasattr(agent, "crew")
        and agent.crew is not None
        and not isinstance(agent.crew, str)
    ):
        attrs.update(
            semantic_conventions.crewai_crew(
                name=agent.crew.name, id=str(agent.crew.id)
            )
        )

    span = _end_span(
        ctx,
        event.started_event_id,
        event,
        attrs,
        error=event.error,
    )

    providers.emit_log(
        f"Agent execution error: {agent.key}",
        level="ERROR",
        span=span,
        attributes=attrs,
        ctx=ctx,
    )


def handle_tool_usage_started(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    source: Any,
    event: ToolUsageStartedEvent,
) -> None:
    attrs = {
        **semantic_conventions.gen_ai(
            operation_name="execute_tool",
            tool_name=event.tool_name,
            tool_type="function",
            tool_call_arguments=event.tool_args,
        ),
        **semantic_conventions.crewai_span(
            event_name=event.type, subject=event.tool_name
        ),
        **semantic_conventions.crewai_agent(key=event.agent_key, role=event.agent_role),
    }

    span = _start_span(providers, ctx, "call tool", event, attrs)
    if span:
        _store_span(ctx, event.event_id, span)

    providers.emit_log(
        f"Tool execution started: {event.tool_name}",
        level="INFO",
        span=span,
        attributes=attrs,
        ctx=ctx,
    )


def _instant_span(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    name: str,
    event: Any,
    attributes: dict[str, Any],
    error: str | None = None,
) -> Span | None:
    """Emit a zero-duration span for an event that has no start/end pair.

    Studio's trace tree is built from spans alone -- ``Wharf::SpansFetchService``
    fetches spans and nothing else, and a row's identity comes from the span's
    ``crewai.event_name`` attribute. An OTel *log* therefore never reaches the
    timeline, so anything that should appear as a row needs a span even when it
    represents an instant rather than an interval.
    """
    span = _start_span(providers, ctx, name, event, attributes)
    if span is None:
        return None

    status = _apply_error(span, error, attributes)
    _set_span_attributes(span, attributes)
    span.set_status(status)
    span.end(end_time=_datetime_to_nanoseconds(event.timestamp))
    return span


def _failure_message(failure: Any) -> str | None:
    """The one-line reason a tool call failed, or None when it succeeded."""
    if failure is None:
        return None
    return getattr(failure, "message", None) or str(failure)


def _tool_failure_attrs(failure: Any, policy: Any = None) -> dict[str, Any]:
    """Span attributes describing a declared tool failure.

    Empty for a successful call, so callers can splat it unconditionally.
    """
    if failure is None:
        return {}

    reason = getattr(failure, "reason", None)
    return semantic_conventions.crewai_tool_failure(
        message=_failure_message(failure),
        # Both are str-valued enums; OTel attributes must be primitives.
        reason=getattr(reason, "value", None) or (str(reason) if reason else None),
        code=getattr(failure, "code", None),
        retryable=getattr(failure, "retryable", None),
        policy=getattr(policy, "value", None) or (str(policy) if policy else None),
    )


def handle_tool_failure_detected(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    source: Any,
    event: Any,
) -> None:
    """Record a tool that ran but reported it did not do the work.

    ``handle_tool_usage_finished`` already marks the tool's own span failed;
    this adds the policy-carrying row next to it. It has to be a span rather
    than a log because the timeline is built from spans only.
    """
    failure = getattr(event, "failure", None)
    policy = getattr(event, "policy", None)

    attrs = {
        **semantic_conventions.gen_ai(
            operation_name="execute_tool",
            tool_name=event.tool_name,
            tool_type="function",
            tool_call_arguments=event.tool_args,
        ),
        **semantic_conventions.crewai_span(
            event_name=event.type, subject=event.tool_name
        ),
        **semantic_conventions.crewai_agent(key=event.agent_key, role=event.agent_role),
        **semantic_conventions.crewai_task(id=event.task_id, name=event.task_name),
        **_tool_failure_attrs(failure, policy),
    }

    span = _instant_span(
        providers,
        ctx,
        "tool failure",
        event,
        attrs,
        error=_failure_message(failure),
    )

    providers.emit_log(
        f"Tool failure detected: {event.tool_name}",
        level="ERROR",
        span=span,
        attributes=attrs,
        ctx=ctx,
    )


def handle_tool_usage_finished(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    source: Any,
    event: ToolUsageFinishedEvent,
) -> None:
    span = (
        ctx.active_spans.get(event.started_event_id) if event.started_event_id else None
    )

    # A tool can return normally and still have failed (Slack answering 200 with
    # ok=false, an MCP server setting isError). Without this the span closes OK
    # and the only evidence of the failure is prose inside tool_call_result.
    # getattr rather than event.failure: the pin guarantees the field, but this
    # handler is also reachable with events rebuilt from a replayed payload.
    failure = getattr(event, "failure", None)

    attrs = {
        **semantic_conventions.gen_ai(
            operation_name="execute_tool",
            tool_name=event.tool_name,
            tool_type="function",
            tool_call_arguments=event.tool_args,
            tool_call_result=event.output,
        ),
        **semantic_conventions.crewai_span(
            event_name=event.type, subject=event.tool_name
        ),
        **semantic_conventions.crewai_agent(key=event.agent_key, role=event.agent_role),
        **_tool_failure_attrs(failure),
    }

    providers.emit_log(
        f"Tool reported failure: {event.tool_name}"
        if failure
        else f"Tool execution completed: {event.tool_name}",
        level="ERROR" if failure else "INFO",
        span=span,
        attributes=attrs,
        ctx=ctx,
    )

    _end_span(
        ctx,
        event.started_event_id,
        event,
        attrs,
        error=_failure_message(failure),
    )


def handle_tool_usage_error(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    source: Any,
    event: ToolUsageErrorEvent,
) -> None:
    attrs = {
        **semantic_conventions.gen_ai(
            operation_name="execute_tool",
            tool_name=event.tool_name,
            tool_type="function",
            tool_call_arguments=event.tool_args,
        ),
        **semantic_conventions.crewai_span(
            event_name=event.type, subject=event.tool_name
        ),
        **semantic_conventions.crewai_agent(key=event.agent_key, role=event.agent_role),
    }

    span = _end_span(
        ctx,
        event.started_event_id,
        event,
        attrs,
        error=event.error,
    )

    providers.emit_log(
        f"Tool execution error: {event.tool_name}",
        level="ERROR",
        span=span,
        attributes=attrs,
        ctx=ctx,
    )


def handle_flow_created(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    source: Any,
    event: FlowCreatedEvent,
) -> None:
    flow_name = source.name or event.flow_name

    attrs = {
        **semantic_conventions.crewai_span(event_name=event.type, subject=flow_name),
        **semantic_conventions.crewai_flow(name=flow_name),
    }

    providers.emit_log(
        f"Flow created: {flow_name}",
        level="INFO",
        attributes=attrs,
        ctx=ctx,
    )


def _reparent_suppressed_flow_children(
    ctx: TelemetryExecutionContext, event: Any
) -> None:
    """Attach a suppressed flow's children to that flow's parent span.

    Infrastructure flows still open event scopes even when their spans are
    hidden. Alias the hidden event to its parent so children keep their real
    ancestry instead of falling back to the execution root.
    """
    parent_span: Span | None = None
    if event.parent_event_id:
        parent_span = ctx._span_refs.get(event.parent_event_id)
    if parent_span is None:
        parent_span = ctx.root_span

    with ctx._span_lock:
        if parent_span is not None:
            ctx._span_refs[event.event_id] = NonRecordingSpan(
                parent_span.get_span_context()
            )


def handle_flow_started(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    source: Flow[Any],
    event: FlowStartedEvent,
) -> None:
    if source.suppress_flow_events:
        _reparent_suppressed_flow_children(ctx, event)
        return

    flow_name = source.name or event.flow_name
    flow_id = str(source.flow_id)
    method_names = list(source._methods.keys())

    # Initialize flow metrics aggregation
    ctx.flow_crew_usage_metrics[flow_id] = {
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cached_prompt_tokens": 0,
        "reasoning_tokens": 0,
        "cache_creation_tokens": 0,
        "successful_requests": 0,
        "crew_count": 0,
    }

    serialized_inputs = _serialize(event.inputs)
    # A HITL resume re-emits FlowStartedEvent with no inputs, so surface the human feedback as the resume span's
    # GenAI input — otherwise the resume segment is input-less in LLM tooling.
    genai_input = serialized_inputs
    if not event.inputs and ctx.resume_feedback is not None:
        genai_input = _serialize(ctx.resume_feedback)
    attrs = {
        **semantic_conventions.crewai_span(event_name=event.type, subject=flow_name),
        **semantic_conventions.crewai_flow(
            name=flow_name,
            id=flow_id,
            method_names=_serialize(method_names),
            inputs=serialized_inputs,
        ),
        **semantic_conventions.gen_ai(
            operation_name=semantic_conventions.GEN_AI_OP_INVOKE_WORKFLOW,
            workflow_name=flow_name,
        ),
        **semantic_conventions.gen_ai_io(input_value=genai_input),
    }

    links = None
    if event.parent_event_id is None and ctx.parent_otel_context is not None:
        links = [_build_follows_from_link(ctx.parent_otel_context)]

    span = _start_span(providers, ctx, "execute flow", event, attrs, links=links)
    if span:
        _store_span(ctx, event.event_id, span)
        span_ctx = span.get_span_context()
        if span_ctx and span_ctx.is_valid:
            ctx.otel_resume_context = (span_ctx.trace_id, span_ctx.span_id)

    providers.emit_log(
        f"Flow started: {flow_name}",
        level="INFO",
        span=span,
        attributes=attrs,
        ctx=ctx,
    )


def handle_flow_finished(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    source: Flow[Any],
    event: FlowFinishedEvent,
) -> None:
    if source.suppress_flow_events:
        return

    flow_name = source.name or event.flow_name
    flow_id = str(source.flow_id)
    result = event.result

    with ctx._span_lock:
        aggregated_crew_usage_metrics = ctx.flow_crew_usage_metrics.pop(flow_id, None)

    if aggregated_crew_usage_metrics:
        # OSS Flow types this private attr as ``UsageMetrics``; we intentionally
        # stash the raw aggregation dict (it carries an enterprise-only
        # ``crew_count``). No one reads it back as a model, so the looser dict
        # is fine here.
        source._aggregated_usage_metrics = aggregated_crew_usage_metrics  # type: ignore[assignment]

    aggregated_metrics_str = (
        _serialize(aggregated_crew_usage_metrics)
        if aggregated_crew_usage_metrics
        and aggregated_crew_usage_metrics["crew_count"] > 0
        else None
    )

    serialized_result = _serialize(result)
    attrs: dict[str, Any] = {
        **semantic_conventions.crewai_span(event_name=event.type, subject=flow_name),
        **semantic_conventions.crewai_flow(
            name=flow_name,
            id=flow_id,
            result=serialized_result,
            aggregated_crew_usage_metrics=aggregated_metrics_str,
        ),
        **semantic_conventions.gen_ai_io(output_value=serialized_result),
    }

    providers.emit_log(
        f"Flow ended: {flow_name}",
        level="INFO",
        attributes=attrs,
        ctx=ctx,
    )

    _end_span(
        ctx,
        event.started_event_id,
        event,
        attrs,
        duration_attr=semantic_conventions.FLOW_EXECUTION_DURATION_MS,
    )


def handle_flow_failed(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    source: Flow[Any],
    event: FlowFailedEvent,
) -> None:
    if source.suppress_flow_events:
        return

    flow_name = source.name or event.flow_name
    flow_id = str(source.flow_id)

    with ctx._span_lock:
        aggregated_crew_usage_metrics = ctx.flow_crew_usage_metrics.pop(flow_id, None)

    if aggregated_crew_usage_metrics:
        source._aggregated_usage_metrics = aggregated_crew_usage_metrics  # type: ignore[assignment]

    aggregated_metrics_str = (
        _serialize(aggregated_crew_usage_metrics)
        if aggregated_crew_usage_metrics
        and aggregated_crew_usage_metrics["crew_count"] > 0
        else None
    )

    attrs: dict[str, Any] = {
        **semantic_conventions.crewai_span(event_name=event.type, subject=flow_name),
        **semantic_conventions.crewai_flow(
            name=flow_name,
            id=flow_id,
            aggregated_crew_usage_metrics=aggregated_metrics_str,
        ),
    }

    span = _end_span(
        ctx,
        event.started_event_id,
        event,
        attrs,
        error=event.error,
        duration_attr=semantic_conventions.FLOW_EXECUTION_DURATION_MS,
    )

    providers.emit_log(
        f"Flow failed: {flow_name}",
        level="ERROR",
        span=span,
        attributes=attrs,
        ctx=ctx,
    )


def handle_method_execution_started(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    source: Any,
    event: MethodExecutionStartedEvent,
) -> None:
    flow_name = source.name or event.flow_name

    serialized_params = _serialize(event.params)
    attrs = {
        **semantic_conventions.crewai_span(
            event_name=event.type, subject=event.method_name
        ),
        **semantic_conventions.crewai_flow(name=flow_name),
        **semantic_conventions.crewai_method(
            name=event.method_name,
            state=_serialize(event.state),
            params=serialized_params,
        ),
        **semantic_conventions.gen_ai(
            operation_name=semantic_conventions.GEN_AI_OP_EXECUTE_METHOD,
        ),
        **semantic_conventions.gen_ai_io(input_value=serialized_params),
    }

    span = _start_span(providers, ctx, "call method", event, attrs)
    if span:
        _store_span(ctx, event.event_id, span)

    providers.emit_log(
        f"Flow method started: {event.method_name}",
        level="INFO",
        span=span,
        attributes=attrs,
        ctx=ctx,
    )


def handle_method_execution_finished(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    source: Any,
    event: MethodExecutionFinishedEvent,
) -> None:
    flow_name = source.name or event.flow_name

    serialized_result = _serialize(event.result)
    attrs: dict[str, Any] = {
        **semantic_conventions.crewai_span(
            event_name=event.type, subject=event.method_name
        ),
        **semantic_conventions.crewai_flow(name=flow_name),
        **semantic_conventions.crewai_method(
            name=event.method_name,
            result=serialized_result,
            state=_serialize(event.state),
        ),
        **semantic_conventions.gen_ai_io(output_value=serialized_result),
    }

    providers.emit_log(
        f"Flow method ended: {event.method_name}",
        level="INFO",
        attributes=attrs,
        ctx=ctx,
    )

    _end_span(
        ctx,
        event.started_event_id,
        event,
        attrs,
        duration_attr=semantic_conventions.METHOD_DURATION_MS,
    )


def handle_method_execution_failed(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    source: Any,
    event: MethodExecutionFailedEvent,
) -> None:
    flow_name = source.name or event.flow_name
    flow_id = str(source.flow_id)

    with ctx._span_lock:
        aggregated_crew_usage_metrics = ctx.flow_crew_usage_metrics.pop(flow_id, None)

    if aggregated_crew_usage_metrics:
        source._aggregated_usage_metrics = aggregated_crew_usage_metrics

    attrs: dict[str, Any] = {
        **semantic_conventions.crewai_span(
            event_name=event.type, subject=event.method_name
        ),
        **semantic_conventions.crewai_flow(name=flow_name),
        **semantic_conventions.crewai_method(name=event.method_name),
    }

    span = _end_span(
        ctx,
        event.started_event_id,
        event,
        attrs,
        error=event.error,
        duration_attr=semantic_conventions.METHOD_DURATION_MS,
    )

    providers.emit_log(
        f"Flow method failed: {event.method_name}",
        level="ERROR",
        span=span,
        attributes=attrs,
        ctx=ctx,
    )


def handle_flow_paused(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    source: Flow[Any],
    event: FlowPausedEvent,
) -> None:
    if source.suppress_flow_events:
        return

    flow_name = source.name or event.flow_name
    flow_id = event.flow_id

    with ctx._span_lock:
        aggregated_crew_usage_metrics = ctx.flow_crew_usage_metrics.pop(flow_id, None)

    aggregated_metrics_str = (
        _serialize(aggregated_crew_usage_metrics)
        if aggregated_crew_usage_metrics
        and aggregated_crew_usage_metrics["crew_count"] > 0
        else None
    )

    attrs: dict[str, Any] = {
        **semantic_conventions.crewai_span(event_name=event.type, subject=flow_name),
        **semantic_conventions.crewai_flow(
            name=flow_name,
            id=flow_id,
            aggregated_crew_usage_metrics=aggregated_metrics_str,
        ),
        **semantic_conventions.crewai_human_feedback(
            method_name=event.method_name,
            message=event.message,
            emit=_serialize(event.emit) if event.emit else None,
        ),
    }

    providers.emit_log(
        f"Flow paused: {flow_name}",
        level="INFO",
        attributes=attrs,
        ctx=ctx,
    )

    _end_span(
        ctx,
        event.started_event_id,
        event,
        attrs,
        duration_attr=semantic_conventions.FLOW_EXECUTION_DURATION_MS,
    )

    flow_span = ctx._span_refs.get(event.started_event_id or "")
    if flow_span:
        span_ctx = flow_span.get_span_context()
        if span_ctx and span_ctx.is_valid:
            ctx.otel_resume_context = (span_ctx.trace_id, span_ctx.span_id)


def handle_method_execution_paused(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    source: Any,
    event: MethodExecutionPausedEvent,
) -> None:
    flow_name = source.name or event.flow_name

    attrs: dict[str, Any] = {
        **semantic_conventions.crewai_span(
            event_name=event.type, subject=event.method_name
        ),
        **semantic_conventions.crewai_flow(name=flow_name),
        **semantic_conventions.crewai_method(
            name=event.method_name,
            state=_serialize(event.state),
        ),
        **semantic_conventions.crewai_human_feedback(
            method_name=event.method_name,
            message=event.message,
            emit=_serialize(event.emit) if event.emit else None,
        ),
    }

    providers.emit_log(
        f"Flow method paused: {event.method_name}",
        level="INFO",
        attributes=attrs,
        ctx=ctx,
    )

    _end_span(
        ctx,
        event.started_event_id,
        event,
        attrs,
        duration_attr=semantic_conventions.METHOD_DURATION_MS,
    )


def handle_human_feedback_requested(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    source: Any,
    event: HumanFeedbackRequestedEvent,
) -> None:
    flow_name = source.name or event.flow_name

    request_id = getattr(event, "request_id", None)
    attrs: dict[str, Any] = {
        **semantic_conventions.crewai_span(
            event_name=event.type, subject=event.method_name
        ),
        **semantic_conventions.crewai_flow(name=flow_name),
        **semantic_conventions.crewai_human_feedback(
            method_name=event.method_name,
            message=event.message,
            emit=_serialize(event.emit) if event.emit else None,
            request_id=str(request_id) if request_id else None,
        ),
    }

    span = _start_span(providers, ctx, "request human feedback", event, attrs)
    if span:
        end_time_ns = _datetime_to_nanoseconds(event.timestamp)
        span.set_status(Status(StatusCode.OK))
        span.end(end_time=end_time_ns)

    providers.emit_log(
        f"Human feedback requested: {event.method_name}",
        level="INFO",
        span=span,
        attributes=attrs,
        ctx=ctx,
    )


def handle_human_feedback_received(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    source: Any,
    event: HumanFeedbackReceivedEvent,
) -> None:
    flow_name = source.name or event.flow_name

    attrs: dict[str, Any] = {
        **semantic_conventions.crewai_span(
            event_name=event.type, subject=event.method_name
        ),
        **semantic_conventions.crewai_flow(name=flow_name),
        **semantic_conventions.crewai_human_feedback(
            method_name=event.method_name,
            feedback=event.feedback,
            outcome=event.outcome,
        ),
    }

    span = _start_span(providers, ctx, "receive human feedback", event, attrs)
    if span:
        end_time_ns = _datetime_to_nanoseconds(event.timestamp)
        span.set_status(Status(StatusCode.OK))
        span.end(end_time=end_time_ns)

    providers.emit_log(
        f"Human feedback received: {event.method_name}",
        level="INFO",
        span=span,
        attributes=attrs,
        ctx=ctx,
    )


def _response_format_output_type(response_format: Any) -> str:
    """Map a `response_format` value to a `gen_ai.output.type` string.

    OpenAI/Azure shape: ``{"type": "text" | "json_object" | "json_schema", ...}``.
    A dict with ``type == "text"`` is plain text — only ``json_object`` /
    ``json_schema`` count as structured output. Non-dict values (Pydantic
    classes, dataclasses, etc.) imply structured output.
    """
    if isinstance(response_format, dict):
        return (
            "json"
            if response_format.get("type") in ("json_object", "json_schema")
            else "text"
        )
    return "json"


def _llm_output_type(event: Any) -> str:
    if getattr(event, "response_model", None) is not None:
        return "json"
    response_format = getattr(event, "response_format", None)
    if response_format is None:
        return "text"
    return _response_format_output_type(response_format)


def handle_llm_call_started(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    source: Any,
    event: LLMCallStartedEvent,
) -> None:
    _record_agent_llm_call(ctx, event.agent_id)
    model = event.model or UNKNOWN_MODEL
    provider_name = _llm_provider_name(source)

    attrs = {
        **semantic_conventions.gen_ai(
            operation_name="chat",
            request_model=model,
            provider_name=provider_name,
            input_messages=event.messages,
            tool_definitions=list(event.tools) if event.tools else None,
            output_type=_llm_output_type(event),
            temperature=getattr(event, "temperature", None),
            top_p=getattr(event, "top_p", None),
            max_tokens=getattr(event, "max_tokens", None),
            stream=getattr(event, "stream", None),
            seed=getattr(event, "seed", None),
            stop_sequences=getattr(event, "stop_sequences", None),
            frequency_penalty=getattr(event, "frequency_penalty", None),
            presence_penalty=getattr(event, "presence_penalty", None),
            choice_count=getattr(event, "n", None),
            conversation_id=_conversation_id(event, ctx),
        ),
        **semantic_conventions.crewai_span(event_name=event.type, subject=model),
        **semantic_conventions.crewai_llm(
            call_id=event.call_id,
            callbacks=_serialize(event.callbacks),
            available_functions=_serialize(event.available_functions),
        ),
    }

    span = _start_span(providers, ctx, "call llm", event, attrs, kind=SpanKind.CLIENT)
    if span:
        _store_span(ctx, event.event_id, span)

    providers.emit_log(
        f"LLM call started: {model}",
        level="INFO",
        span=span,
        attributes=attrs,
        ctx=ctx,
    )


def handle_llm_call_completed(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    source: Any,
    event: LLMCallCompletedEvent,
) -> None:
    model = event.model or UNKNOWN_MODEL
    provider_name = _llm_provider_name(source)
    span = (
        ctx.active_spans.get(event.started_event_id) if event.started_event_id else None
    )

    usage = event.usage or {}
    input_tokens = next(
        (
            v
            for v in (usage.get("prompt_tokens"), usage.get("input_tokens"))
            if v is not None
        ),
        0,
    )
    output_tokens = next(
        (
            v
            for v in (usage.get("completion_tokens"), usage.get("output_tokens"))
            if v is not None
        ),
        0,
    )

    # `gen_ai.output.type` is a request-shape attribute set on the start span;
    # the completed event lacks `response_model`/`response_format` so we leave
    # the existing attribute alone instead of overwriting it with "text".
    attrs = {
        **semantic_conventions.gen_ai(
            operation_name="chat",
            request_model=model,
            response_model=model,
            provider_name=provider_name,
            output_messages=event.response,
            finish_reason=getattr(event, "finish_reason", None),
            response_id=getattr(event, "response_id", None),
            conversation_id=_conversation_id(event, ctx),
        ),
        **semantic_conventions.crewai_span(event_name=event.type, subject=model),
        **semantic_conventions.crewai_llm(
            call_id=event.call_id,
            call_type=str(event.call_type.value),
        ),
    }

    if getattr(event, "task_id", None):
        attrs["crewai.task.id"] = event.task_id
    if getattr(event, "agent_role", None):
        attrs["crewai.agent.role"] = event.agent_role

    cached = usage.get("cached_prompt_tokens") or 0
    reasoning = usage.get("reasoning_tokens") or 0
    cache_creation = usage.get("cache_creation_tokens") or 0

    if input_tokens or output_tokens:
        attrs.update(
            semantic_conventions.gen_ai_usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_input_tokens=cached or None,
                reasoning_tokens=reasoning or None,
                cache_creation_tokens=cache_creation or None,
            )
        )

        if ctx.flow_crew_usage_metrics:
            with ctx._span_lock:
                for flow_metrics in ctx.flow_crew_usage_metrics.values():
                    flow_metrics["total_tokens"] += input_tokens + output_tokens
                    flow_metrics["prompt_tokens"] += input_tokens
                    flow_metrics["completion_tokens"] += output_tokens
                    flow_metrics["cached_prompt_tokens"] += cached
                    flow_metrics["reasoning_tokens"] += reasoning
                    flow_metrics["cache_creation_tokens"] += cache_creation
                    flow_metrics["successful_requests"] += 1

    providers.emit_log(
        f"LLM call ended: {model}",
        level="INFO",
        span=span,
        attributes=attrs,
        ctx=ctx,
    )

    _end_span(ctx, event.started_event_id, event, attrs)


def handle_llm_call_failed(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    source: Any,
    event: LLMCallFailedEvent,
) -> None:
    model = event.model or UNKNOWN_MODEL
    provider_name = _llm_provider_name(source)

    attrs = {
        **semantic_conventions.gen_ai(
            operation_name="chat",
            request_model=model,
            provider_name=provider_name,
            finish_reason="error",
            conversation_id=_conversation_id(event, ctx),
        ),
        **semantic_conventions.crewai_span(event_name=event.type, subject=model),
        **semantic_conventions.crewai_llm(call_id=event.call_id),
    }

    span = _end_span(ctx, event.started_event_id, event, attrs, error=event.error)

    providers.emit_log(
        f"LLM call failed: {model}",
        level="ERROR",
        span=span,
        attributes=attrs,
        ctx=ctx,
    )


# ---------------------------------------------------------------------------
# Lite Agent handlers
# ---------------------------------------------------------------------------


def _lite_agent_output_type(event: Any) -> str:
    """Lite-agent events don't carry a JSON-output marker today — default
    to "text". Read prospective fields with `getattr` so a future OSS addition
    (e.g. `output_pydantic`) flips this to "json" without an enterprise change.
    """
    if getattr(event, "output_pydantic", None) is not None:
        return "json"
    response_format = getattr(event, "response_format", None)
    if response_format is None:
        return "text"
    return _response_format_output_type(response_format)


def handle_lite_agent_execution_started(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    source: Any,
    event: LiteAgentExecutionStartedEvent,
) -> None:
    role = event.agent_info.get("role", "lite_agent")
    key = event.agent_info.get("key")
    agent_id = event.agent_info.get("id")

    attrs: dict[str, Any] = {
        **semantic_conventions.crewai_span(event_name=event.type, subject=role),
        **semantic_conventions.crewai_agent(role=role, key=key),
        **semantic_conventions.gen_ai(
            operation_name="invoke_agent",
            agent_name=key or role,
            agent_id=str(agent_id) if agent_id else None,
            tool_definitions=list(event.tools) if event.tools else None,
            input_messages=event.messages,
            output_type=_lite_agent_output_type(event),
            conversation_id=_conversation_id(event, ctx),
        ),
    }

    # NOTE: keep span name "execute lite agent" — there is a follow-up ticket
    # for the spec-compliant `invoke_agent <model>` rename.
    span = _start_span(
        providers, ctx, "execute lite agent", event, attrs, kind=SpanKind.CLIENT
    )
    if span:
        _store_span(ctx, event.event_id, span)

    providers.emit_log(
        f"Lite agent execution started: {role}",
        level="INFO",
        span=span,
        attributes=attrs,
        ctx=ctx,
    )


def handle_lite_agent_execution_completed(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    source: Any,
    event: LiteAgentExecutionCompletedEvent,
) -> None:
    role = event.agent_info.get("role", "lite_agent")
    key = event.agent_info.get("key")
    agent_id = event.agent_info.get("id")

    attrs: dict[str, Any] = {
        **semantic_conventions.crewai_span(event_name=event.type, subject=role),
        **semantic_conventions.crewai_agent(role=role, key=key),
        **semantic_conventions.gen_ai(
            operation_name="invoke_agent",
            agent_name=key or role,
            agent_id=str(agent_id) if agent_id else None,
            output_messages=event.output,
            conversation_id=_conversation_id(event, ctx),
        ),
    }

    providers.emit_log(
        f"Lite agent execution completed: {role}",
        level="INFO",
        attributes=attrs,
        ctx=ctx,
    )

    _end_span(ctx, event.started_event_id, event, attrs)


def handle_lite_agent_execution_error(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    source: Any,
    event: LiteAgentExecutionErrorEvent,
) -> None:
    role = event.agent_info.get("role", "lite_agent")
    key = event.agent_info.get("key")
    agent_id = event.agent_info.get("id")

    attrs: dict[str, Any] = {
        **semantic_conventions.crewai_span(event_name=event.type, subject=role),
        **semantic_conventions.crewai_agent(role=role, key=key),
        **semantic_conventions.gen_ai(
            operation_name="invoke_agent",
            agent_name=key or role,
            agent_id=str(agent_id) if agent_id else None,
            conversation_id=_conversation_id(event, ctx),
        ),
    }

    span = _end_span(ctx, event.started_event_id, event, attrs, error=event.error)

    providers.emit_log(
        f"Lite agent execution error: {role}",
        level="ERROR",
        span=span,
        attributes=attrs,
        ctx=ctx,
    )


# ---------------------------------------------------------------------------
# Agent Reasoning handlers
# ---------------------------------------------------------------------------


def handle_agent_reasoning_started(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    source: Any,
    event: AgentReasoningStartedEvent,
) -> None:
    attrs: dict[str, Any] = {
        **semantic_conventions.crewai_span(
            event_name=event.type, subject=event.agent_role
        ),
        **semantic_conventions.crewai_agent(role=event.agent_role),
        **semantic_conventions.crewai_reasoning(attempt=event.attempt),
        **semantic_conventions.crewai_task(id=event.task_id, name=event.task_name),
    }

    span = _start_span(providers, ctx, "agent reasoning", event, attrs)
    if span:
        _store_span(ctx, event.event_id, span)

    providers.emit_log(
        f"Agent reasoning started: {event.agent_role}",
        level="INFO",
        span=span,
        attributes=attrs,
        ctx=ctx,
    )


def handle_agent_reasoning_completed(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    source: Any,
    event: AgentReasoningCompletedEvent,
) -> None:
    attrs: dict[str, Any] = {
        **semantic_conventions.crewai_span(
            event_name=event.type, subject=event.agent_role
        ),
        **semantic_conventions.crewai_agent(role=event.agent_role),
        **semantic_conventions.crewai_reasoning(
            attempt=event.attempt, plan=event.plan, ready=event.ready
        ),
        **semantic_conventions.crewai_task(id=event.task_id, name=event.task_name),
    }

    providers.emit_log(
        f"Agent reasoning completed: {event.agent_role}",
        level="INFO",
        attributes=attrs,
        ctx=ctx,
    )

    _end_span(ctx, event.started_event_id, event, attrs)


def handle_agent_reasoning_failed(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    source: Any,
    event: AgentReasoningFailedEvent,
) -> None:
    attrs: dict[str, Any] = {
        **semantic_conventions.crewai_span(
            event_name=event.type, subject=event.agent_role
        ),
        **semantic_conventions.crewai_agent(role=event.agent_role),
        **semantic_conventions.crewai_reasoning(attempt=event.attempt),
        **semantic_conventions.crewai_task(id=event.task_id, name=event.task_name),
    }

    span = _end_span(ctx, event.started_event_id, event, attrs, error=event.error)

    providers.emit_log(
        f"Agent reasoning failed: {event.agent_role}",
        level="ERROR",
        span=span,
        attributes=attrs,
        ctx=ctx,
    )


# ---------------------------------------------------------------------------
# LLM Guardrail handlers
# ---------------------------------------------------------------------------


def _guardrail_subject(event: Any) -> str:
    """Build a human-readable subject for guardrail spans."""
    name = getattr(event, "guardrail_name", None)
    if isinstance(name, str) and name:
        return name[:100]
    desc = str(event.guardrail) if hasattr(event, "guardrail") else ""
    if desc:
        return desc[:100]
    return "guardrail"


def handle_llm_guardrail_started(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    source: Any,
    event: LLMGuardrailStartedEvent,
) -> None:
    attrs: dict[str, Any] = {
        **semantic_conventions.crewai_span(
            event_name=event.type, subject=_guardrail_subject(event)
        ),
        **semantic_conventions.crewai_guardrail(
            guardrail=str(event.guardrail),
            guardrail_type=getattr(event, "guardrail_type", None),
            retry_count=event.retry_count,
        ),
        **semantic_conventions.crewai_agent(role=event.agent_role),
    }

    span = _start_span(providers, ctx, "evaluate guardrail", event, attrs)
    if span:
        _store_span(ctx, event.event_id, span)

    providers.emit_log(
        "Guardrail evaluation started",
        level="INFO",
        span=span,
        attributes=attrs,
        ctx=ctx,
    )


def handle_llm_guardrail_completed(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    source: Any,
    event: LLMGuardrailCompletedEvent,
) -> None:
    attrs: dict[str, Any] = {
        **semantic_conventions.crewai_span(event_name=event.type),
        **semantic_conventions.crewai_guardrail(
            guardrail_type=getattr(event, "guardrail_type", None),
            retry_count=event.retry_count,
            success=event.success,
            result=_serialize(event.result),
            error=event.error,
        ),
        **semantic_conventions.crewai_agent(role=event.agent_role),
    }

    span = _end_span(
        ctx,
        event.started_event_id,
        event,
        attrs,
        error=event.error if event.error else None,
    )

    providers.emit_log(
        "Guardrail evaluation completed",
        level="ERROR" if event.error else "INFO",
        span=span,
        attributes=attrs,
        ctx=ctx,
    )


# ---------------------------------------------------------------------------
# Memory Query handlers
# ---------------------------------------------------------------------------


def handle_memory_query_started(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    source: Any,
    event: MemoryQueryStartedEvent,
) -> None:
    attrs: dict[str, Any] = {
        **semantic_conventions.crewai_span(
            event_name=event.type, subject="Memory Query"
        ),
        **semantic_conventions.crewai_memory(
            query=event.query,
            limit=event.limit,
            score_threshold=event.score_threshold,
        ),
        **semantic_conventions.crewai_agent(role=event.agent_role),
        **semantic_conventions.crewai_task(id=event.task_id, name=event.task_name),
    }

    span = _start_span(providers, ctx, "query memory", event, attrs)
    if span:
        _store_span(ctx, event.event_id, span)

    providers.emit_log(
        "Memory query started",
        level="INFO",
        span=span,
        attributes=attrs,
        ctx=ctx,
    )


def handle_memory_query_completed(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    source: Any,
    event: MemoryQueryCompletedEvent,
) -> None:
    attrs: dict[str, Any] = {
        **semantic_conventions.crewai_span(
            event_name=event.type, subject="Memory Query"
        ),
        **semantic_conventions.crewai_memory(
            query=event.query,
            results=_serialize(event.results),
            limit=event.limit,
            score_threshold=event.score_threshold,
        ),
        **semantic_conventions.crewai_agent(role=event.agent_role),
        **semantic_conventions.crewai_task(id=event.task_id, name=event.task_name),
    }

    providers.emit_log(
        "Memory query completed",
        level="INFO",
        attributes=attrs,
        ctx=ctx,
    )

    _end_span(
        ctx,
        event.started_event_id,
        event,
        attrs,
        duration_attr=semantic_conventions.MEMORY_QUERY_DURATION_MS,
    )


def handle_memory_query_failed(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    source: Any,
    event: MemoryQueryFailedEvent,
) -> None:
    attrs: dict[str, Any] = {
        **semantic_conventions.crewai_span(
            event_name=event.type, subject="Memory Query"
        ),
        **semantic_conventions.crewai_memory(
            query=event.query,
            limit=event.limit,
            score_threshold=event.score_threshold,
        ),
        **semantic_conventions.crewai_agent(role=event.agent_role),
        **semantic_conventions.crewai_task(id=event.task_id, name=event.task_name),
    }

    span = _end_span(
        ctx,
        event.started_event_id,
        event,
        attrs,
        error=event.error,
        duration_attr=semantic_conventions.MEMORY_QUERY_DURATION_MS,
    )

    providers.emit_log(
        "Memory query failed",
        level="ERROR",
        span=span,
        attributes=attrs,
        ctx=ctx,
    )


# ---------------------------------------------------------------------------
# Memory Retrieval handlers
# ---------------------------------------------------------------------------


def handle_memory_retrieval_started(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    source: Any,
    event: MemoryRetrievalStartedEvent,
) -> None:
    attrs: dict[str, Any] = {
        **semantic_conventions.crewai_span(
            event_name=event.type, subject="Memory Retrieval"
        ),
        **semantic_conventions.crewai_agent(role=event.agent_role),
        **semantic_conventions.crewai_task(id=event.task_id, name=event.task_name),
    }

    span = _start_span(providers, ctx, "retrieve memory", event, attrs)
    if span:
        _store_span(ctx, event.event_id, span)

    providers.emit_log(
        "Memory retrieval started",
        level="INFO",
        span=span,
        attributes=attrs,
        ctx=ctx,
    )


def handle_memory_retrieval_completed(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    source: Any,
    event: MemoryRetrievalCompletedEvent,
) -> None:
    attrs: dict[str, Any] = {
        **semantic_conventions.crewai_span(
            event_name=event.type, subject="Memory Retrieval"
        ),
        **semantic_conventions.crewai_memory(
            memory_content=event.memory_content,
        ),
        **semantic_conventions.crewai_agent(role=event.agent_role),
        **semantic_conventions.crewai_task(id=event.task_id, name=event.task_name),
    }

    providers.emit_log(
        "Memory retrieval completed",
        level="INFO",
        attributes=attrs,
        ctx=ctx,
    )

    _end_span(
        ctx,
        event.started_event_id,
        event,
        attrs,
        duration_attr=semantic_conventions.MEMORY_RETRIEVAL_DURATION_MS,
    )


def handle_memory_retrieval_failed(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    source: Any,
    event: MemoryRetrievalFailedEvent,
) -> None:
    attrs: dict[str, Any] = {
        **semantic_conventions.crewai_span(
            event_name=event.type, subject="Memory Retrieval"
        ),
        **semantic_conventions.crewai_agent(role=event.agent_role),
        **semantic_conventions.crewai_task(id=event.task_id, name=event.task_name),
    }

    span = _end_span(
        ctx,
        event.started_event_id,
        event,
        attrs,
        error=event.error,
        duration_attr=semantic_conventions.MEMORY_RETRIEVAL_DURATION_MS,
    )

    providers.emit_log(
        "Memory retrieval failed",
        level="ERROR",
        span=span,
        attributes=attrs,
        ctx=ctx,
    )


# ---------------------------------------------------------------------------
# Memory Save handlers
# ---------------------------------------------------------------------------


def handle_memory_save_started(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    source: Any,
    event: MemorySaveStartedEvent,
) -> None:
    attrs: dict[str, Any] = {
        **semantic_conventions.crewai_span(
            event_name=event.type, subject="Memory Save"
        ),
        **semantic_conventions.crewai_memory(
            value=event.value,
            metadata=_serialize(event.metadata) if event.metadata else None,
        ),
        **semantic_conventions.crewai_agent(role=event.agent_role),
        **semantic_conventions.crewai_task(id=event.task_id, name=event.task_name),
    }

    span = _start_span(providers, ctx, "save memory", event, attrs)
    if span:
        _store_span(ctx, event.event_id, span)

    providers.emit_log(
        "Memory save started",
        level="INFO",
        span=span,
        attributes=attrs,
        ctx=ctx,
    )


def handle_memory_save_completed(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    source: Any,
    event: MemorySaveCompletedEvent,
) -> None:
    attrs: dict[str, Any] = {
        **semantic_conventions.crewai_span(
            event_name=event.type, subject="Memory Save"
        ),
        **semantic_conventions.crewai_memory(
            value=event.value,
            metadata=_serialize(event.metadata) if event.metadata else None,
        ),
        **semantic_conventions.crewai_agent(role=event.agent_role),
        **semantic_conventions.crewai_task(id=event.task_id, name=event.task_name),
    }

    providers.emit_log(
        "Memory save completed",
        level="INFO",
        attributes=attrs,
        ctx=ctx,
    )

    _end_span(
        ctx,
        event.started_event_id,
        event,
        attrs,
        duration_attr=semantic_conventions.MEMORY_SAVE_DURATION_MS,
    )


def handle_memory_save_failed(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    source: Any,
    event: MemorySaveFailedEvent,
) -> None:
    attrs: dict[str, Any] = {
        **semantic_conventions.crewai_span(
            event_name=event.type, subject="Memory Save"
        ),
        **semantic_conventions.crewai_memory(
            value=event.value,
            metadata=_serialize(event.metadata) if event.metadata else None,
        ),
        **semantic_conventions.crewai_agent(role=event.agent_role),
        **semantic_conventions.crewai_task(id=event.task_id, name=event.task_name),
    }

    span = _end_span(
        ctx,
        event.started_event_id,
        event,
        attrs,
        error=event.error,
        duration_attr=semantic_conventions.MEMORY_SAVE_DURATION_MS,
    )

    providers.emit_log(
        "Memory save failed",
        level="ERROR",
        span=span,
        attributes=attrs,
        ctx=ctx,
    )


# ---------------------------------------------------------------------------
# Knowledge Query handlers
# ---------------------------------------------------------------------------


def handle_knowledge_query_started(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    source: Any,
    event: KnowledgeQueryStartedEvent,
) -> None:
    attrs: dict[str, Any] = {
        **semantic_conventions.crewai_span(
            event_name=event.type, subject="Knowledge Query"
        ),
        **semantic_conventions.crewai_knowledge(
            task_prompt=event.task_prompt,
        ),
        **semantic_conventions.crewai_agent(role=event.agent_role),
        **semantic_conventions.crewai_task(id=event.task_id, name=event.task_name),
    }

    span = _start_span(providers, ctx, "query knowledge", event, attrs)
    if span:
        _store_span(ctx, event.event_id, span)

    providers.emit_log(
        "Knowledge query started",
        level="INFO",
        span=span,
        attributes=attrs,
        ctx=ctx,
    )


def handle_knowledge_query_completed(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    source: Any,
    event: KnowledgeQueryCompletedEvent,
) -> None:
    attrs: dict[str, Any] = {
        **semantic_conventions.crewai_span(
            event_name=event.type, subject="Knowledge Query"
        ),
        **semantic_conventions.crewai_knowledge(
            query=event.query,
        ),
        **semantic_conventions.crewai_agent(role=event.agent_role),
        **semantic_conventions.crewai_task(id=event.task_id, name=event.task_name),
    }

    providers.emit_log(
        "Knowledge query completed",
        level="INFO",
        attributes=attrs,
        ctx=ctx,
    )

    _end_span(ctx, event.started_event_id, event, attrs)


def handle_knowledge_query_failed(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    source: Any,
    event: KnowledgeQueryFailedEvent,
) -> None:
    attrs: dict[str, Any] = {
        **semantic_conventions.crewai_span(
            event_name=event.type, subject="Knowledge Query"
        ),
        **semantic_conventions.crewai_agent(role=event.agent_role),
        **semantic_conventions.crewai_task(id=event.task_id, name=event.task_name),
    }

    span = _end_span(ctx, event.started_event_id, event, attrs, error=event.error)

    providers.emit_log(
        "Knowledge query failed",
        level="ERROR",
        span=span,
        attributes=attrs,
        ctx=ctx,
    )


# ---------------------------------------------------------------------------
# Knowledge Retrieval handlers
# ---------------------------------------------------------------------------


def handle_knowledge_retrieval_started(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    source: Any,
    event: KnowledgeRetrievalStartedEvent,
) -> None:
    attrs: dict[str, Any] = {
        **semantic_conventions.crewai_span(
            event_name=event.type, subject="Knowledge Retrieval"
        ),
        **semantic_conventions.crewai_agent(role=event.agent_role),
        **semantic_conventions.crewai_task(id=event.task_id, name=event.task_name),
    }

    span = _start_span(providers, ctx, "search knowledge", event, attrs)
    if span:
        _store_span(ctx, event.event_id, span)

    providers.emit_log(
        "Knowledge retrieval started",
        level="INFO",
        span=span,
        attributes=attrs,
        ctx=ctx,
    )


def handle_knowledge_retrieval_completed(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    source: Any,
    event: KnowledgeRetrievalCompletedEvent,
) -> None:
    attrs: dict[str, Any] = {
        **semantic_conventions.crewai_span(
            event_name=event.type, subject="Knowledge Retrieval"
        ),
        **semantic_conventions.crewai_knowledge(
            query=event.query,
            retrieved_knowledge=event.retrieved_knowledge,
        ),
        **semantic_conventions.crewai_agent(role=event.agent_role),
        **semantic_conventions.crewai_task(id=event.task_id, name=event.task_name),
    }

    providers.emit_log(
        "Knowledge retrieval completed",
        level="INFO",
        attributes=attrs,
        ctx=ctx,
    )

    _end_span(ctx, event.started_event_id, event, attrs)


def handle_knowledge_search_query_failed(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    source: Any,
    event: KnowledgeSearchQueryFailedEvent,
) -> None:
    attrs: dict[str, Any] = {
        **semantic_conventions.crewai_span(
            event_name=event.type, subject="Knowledge Retrieval"
        ),
        **semantic_conventions.crewai_knowledge(
            query=event.query,
        ),
        **semantic_conventions.crewai_agent(role=event.agent_role),
        **semantic_conventions.crewai_task(id=event.task_id, name=event.task_name),
    }

    span = _end_span(ctx, event.started_event_id, event, attrs, error=event.error)

    providers.emit_log(
        "Knowledge search query failed",
        level="ERROR",
        span=span,
        attributes=attrs,
        ctx=ctx,
    )


# ---------------------------------------------------------------------------
# Skill handlers
# ---------------------------------------------------------------------------


def _skill_attrs(event: Any, **extra: Any) -> dict[str, Any]:
    """Common attributes for every skill event.

    ``skill_path`` is a ``Path``; OTel attribute values must be primitives, so
    it is stringified here rather than at each call site.
    """
    skill_path = getattr(event, "skill_path", None)
    return {
        **semantic_conventions.crewai_span(
            event_name=event.type, subject=event.skill_name or "Skill"
        ),
        **semantic_conventions.crewai_skill(
            name=event.skill_name or None,
            path=str(skill_path) if skill_path else None,
            **extra,
        ),
        **semantic_conventions.crewai_agent(role=event.agent_role),
        **semantic_conventions.crewai_task(id=event.task_id, name=event.task_name),
    }


def handle_skill_used(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    source: Any,
    event: Any,
) -> None:
    """Record a skill's context being injected for a task.

    The runtime signal, unlike activation: it re-fires on every execution, so
    "which skills did this agent actually use, and how often" is answerable.
    """
    attrs = _skill_attrs(
        event, disclosure_level=getattr(event, "disclosure_level", None)
    )

    span = _instant_span(providers, ctx, "use skill", event, attrs)

    providers.emit_log(
        f"Skill used: {event.skill_name}",
        level="INFO",
        span=span,
        attributes=attrs,
        ctx=ctx,
    )


def handle_skill_discovery_started(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    source: Any,
    event: Any,
) -> None:
    attrs = _skill_attrs(event, search_path=str(event.search_path))

    span = _start_span(providers, ctx, "discover skills", event, attrs)
    if span:
        _store_span(ctx, event.event_id, span)

    providers.emit_log(
        "Skill discovery started",
        level="INFO",
        span=span,
        attributes=attrs,
        ctx=ctx,
    )


def handle_skill_discovery_completed(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    source: Any,
    event: Any,
) -> None:
    attrs = _skill_attrs(
        event,
        search_path=str(event.search_path),
        skills_found=event.skills_found,
        skill_names=", ".join(event.skill_names) if event.skill_names else None,
    )

    _end_span(ctx, event.started_event_id, event, attrs)

    providers.emit_log(
        f"Skill discovery completed: {event.skills_found} found",
        level="INFO",
        attributes=attrs,
        ctx=ctx,
    )


def handle_skill_loaded(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    source: Any,
    event: Any,
) -> None:
    attrs = _skill_attrs(
        event, disclosure_level=getattr(event, "disclosure_level", None)
    )

    span = _instant_span(providers, ctx, "load skill", event, attrs)

    providers.emit_log(
        f"Skill loaded: {event.skill_name}",
        level="INFO",
        span=span,
        attributes=attrs,
        ctx=ctx,
    )


def handle_skill_activated(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    source: Any,
    event: Any,
) -> None:
    attrs = _skill_attrs(
        event, disclosure_level=getattr(event, "disclosure_level", None)
    )

    span = _instant_span(providers, ctx, "activate skill", event, attrs)

    providers.emit_log(
        f"Skill activated: {event.skill_name}",
        level="INFO",
        span=span,
        attributes=attrs,
        ctx=ctx,
    )


def handle_skill_load_failed(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    source: Any,
    event: Any,
) -> None:
    attrs = _skill_attrs(event)

    span = _instant_span(providers, ctx, "load skill", event, attrs, error=event.error)

    providers.emit_log(
        f"Skill load failed: {event.skill_name} - {event.error}",
        level="ERROR",
        span=span,
        attributes=attrs,
        ctx=ctx,
    )


# ---------------------------------------------------------------------------
# MCP Connection handlers
# ---------------------------------------------------------------------------


def _mcp_server_label(server_name: str | None, server_url: str | None) -> str | None:
    raw = server_name or server_url
    if not raw:
        return None
    if raw.startswith(("http://", "https://")):
        host = urlparse(raw).hostname
        if host:
            return host
    return raw


def handle_mcp_connection_started(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    source: Any,
    event: MCPConnectionStartedEvent,
) -> None:
    attrs: dict[str, Any] = {
        **semantic_conventions.crewai_span(
            event_name=event.type,
            subject=_mcp_server_label(event.server_name, event.server_url)
            or event.server_name,
        ),
        **semantic_conventions.crewai_mcp(
            server_name=event.server_name,
            server_url=event.server_url,
            transport_type=event.transport_type,
            connect_timeout=event.connect_timeout,
            is_reconnect=event.is_reconnect,
        ),
        **semantic_conventions.crewai_agent(role=event.agent_role),
    }

    span = _start_span(providers, ctx, "connect mcp", event, attrs)
    if span:
        _store_span(ctx, event.event_id, span)

    providers.emit_log(
        f"MCP connection started: {event.server_name}",
        level="INFO",
        span=span,
        attributes=attrs,
        ctx=ctx,
    )


def handle_mcp_connection_completed(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    source: Any,
    event: MCPConnectionCompletedEvent,
) -> None:
    attrs: dict[str, Any] = {
        **semantic_conventions.crewai_span(
            event_name=event.type,
            subject=_mcp_server_label(event.server_name, event.server_url)
            or event.server_name,
        ),
        **semantic_conventions.crewai_mcp(
            server_name=event.server_name,
            server_url=event.server_url,
            transport_type=event.transport_type,
            is_reconnect=event.is_reconnect,
        ),
        **semantic_conventions.crewai_agent(role=event.agent_role),
    }

    providers.emit_log(
        f"MCP connection completed: {event.server_name}",
        level="INFO",
        attributes=attrs,
        ctx=ctx,
    )

    _end_span(
        ctx,
        event.started_event_id,
        event,
        attrs,
        duration_attr=semantic_conventions.MCP_CONNECTION_DURATION_MS,
    )


def handle_mcp_connection_failed(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    source: Any,
    event: MCPConnectionFailedEvent,
) -> None:
    attrs: dict[str, Any] = {
        **semantic_conventions.crewai_span(
            event_name=event.type,
            subject=_mcp_server_label(event.server_name, event.server_url)
            or event.server_name,
        ),
        **semantic_conventions.crewai_mcp(
            server_name=event.server_name,
            server_url=event.server_url,
            transport_type=event.transport_type,
            error_type=event.error_type,
        ),
        **semantic_conventions.crewai_agent(role=event.agent_role),
    }

    span = _end_span(ctx, event.started_event_id, event, attrs, error=event.error)

    providers.emit_log(
        f"MCP connection failed: {event.server_name}",
        level="ERROR",
        span=span,
        attributes=attrs,
        ctx=ctx,
    )


# ---------------------------------------------------------------------------
# MCP Tool Execution handlers
# ---------------------------------------------------------------------------


def handle_mcp_tool_execution_started(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    source: Any,
    event: MCPToolExecutionStartedEvent,
) -> None:
    attrs: dict[str, Any] = {
        **semantic_conventions.crewai_span(
            event_name=event.type, subject=event.tool_name
        ),
        **semantic_conventions.crewai_mcp(
            server_name=event.server_name,
            server_url=event.server_url,
            transport_type=event.transport_type,
            tool_name=event.tool_name,
            tool_args=_serialize(event.tool_args) if event.tool_args else None,
        ),
        **semantic_conventions.gen_ai(
            operation_name="execute_tool",
            tool_name=event.tool_name,
            tool_type="mcp",
            tool_call_arguments=event.tool_args,
        ),
        **semantic_conventions.crewai_agent(role=event.agent_role),
    }

    span = _start_span(providers, ctx, "execute mcp tool", event, attrs)
    if span:
        _store_span(ctx, event.event_id, span)

    providers.emit_log(
        f"MCP tool execution started: {event.tool_name}",
        level="INFO",
        span=span,
        attributes=attrs,
        ctx=ctx,
    )


def handle_mcp_tool_execution_completed(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    source: Any,
    event: MCPToolExecutionCompletedEvent,
) -> None:
    attrs: dict[str, Any] = {
        **semantic_conventions.crewai_span(
            event_name=event.type, subject=event.tool_name
        ),
        **semantic_conventions.crewai_mcp(
            server_name=event.server_name,
            server_url=event.server_url,
            transport_type=event.transport_type,
            tool_name=event.tool_name,
            tool_args=_serialize(event.tool_args) if event.tool_args else None,
            tool_result=_serialize(event.result) if event.result else None,
        ),
        **semantic_conventions.gen_ai(
            operation_name="execute_tool",
            tool_name=event.tool_name,
            tool_type="mcp",
            tool_call_arguments=event.tool_args,
            tool_call_result=event.result,
        ),
        **semantic_conventions.crewai_agent(role=event.agent_role),
    }

    providers.emit_log(
        f"MCP tool execution completed: {event.tool_name}",
        level="INFO",
        attributes=attrs,
        ctx=ctx,
    )

    _end_span(
        ctx,
        event.started_event_id,
        event,
        attrs,
        duration_attr=semantic_conventions.MCP_TOOL_EXECUTION_DURATION_MS,
    )


def handle_mcp_tool_execution_failed(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    source: Any,
    event: MCPToolExecutionFailedEvent,
) -> None:
    attrs: dict[str, Any] = {
        **semantic_conventions.crewai_span(
            event_name=event.type, subject=event.tool_name
        ),
        **semantic_conventions.crewai_mcp(
            server_name=event.server_name,
            server_url=event.server_url,
            transport_type=event.transport_type,
            tool_name=event.tool_name,
            tool_args=_serialize(event.tool_args) if event.tool_args else None,
            error_type=event.error_type,
        ),
        **semantic_conventions.gen_ai(
            operation_name="execute_tool",
            tool_name=event.tool_name,
            tool_type="mcp",
            tool_call_arguments=event.tool_args,
        ),
        **semantic_conventions.crewai_agent(role=event.agent_role),
    }

    span = _end_span(ctx, event.started_event_id, event, attrs, error=event.error)

    providers.emit_log(
        f"MCP tool execution failed: {event.tool_name}",
        level="ERROR",
        span=span,
        attributes=attrs,
        ctx=ctx,
    )


# ---------------------------------------------------------------------------
# A2A Delegation handlers
# ---------------------------------------------------------------------------


def handle_a2a_delegation_started(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    source: Any,
    event: A2ADelegationStartedEvent,
) -> None:
    attrs: dict[str, Any] = {
        **semantic_conventions.crewai_span(
            event_name=event.type, subject=event.a2a_agent_name or event.endpoint
        ),
        **semantic_conventions.crewai_a2a(
            endpoint=event.endpoint,
            task_description=event.task_description,
            agent_id=event.agent_id,
            context_id=event.context_id,
            is_multiturn=event.is_multiturn,
            turn_number=event.turn_number,
            a2a_agent_name=event.a2a_agent_name,
            agent_card=_serialize(event.agent_card) if event.agent_card else None,
            protocol_version=event.protocol_version,
            skill_id=event.skill_id,
            metadata=_serialize(event.metadata) if event.metadata else None,
        ),
    }

    span = _start_span(providers, ctx, "a2a delegate", event, attrs)
    if span:
        _store_span(ctx, event.event_id, span)

    providers.emit_log(
        f"A2A delegation started: {event.a2a_agent_name or event.endpoint}",
        level="INFO",
        span=span,
        attributes=attrs,
        ctx=ctx,
    )


def handle_a2a_delegation_completed(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    source: Any,
    event: A2ADelegationCompletedEvent,
) -> None:
    attrs: dict[str, Any] = {
        **semantic_conventions.crewai_span(
            event_name=event.type, subject=event.a2a_agent_name or event.endpoint
        ),
        **semantic_conventions.crewai_a2a(
            endpoint=event.endpoint,
            context_id=event.context_id,
            is_multiturn=event.is_multiturn,
            a2a_agent_name=event.a2a_agent_name,
            status=event.status,
            result=event.result,
            metadata=_serialize(event.metadata) if event.metadata else None,
        ),
    }

    error = (event.error or "Delegation failed") if event.status == "failed" else None

    span = _end_span(ctx, event.started_event_id, event, attrs, error=error)

    providers.emit_log(
        f"A2A delegation completed: {event.a2a_agent_name or event.endpoint}",
        level="ERROR" if error else "INFO",
        span=span,
        attributes=attrs,
        ctx=ctx,
    )


# ---------------------------------------------------------------------------
# A2A Conversation handlers
# ---------------------------------------------------------------------------


def handle_a2a_conversation_started(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    source: Any,
    event: A2AConversationStartedEvent,
) -> None:
    attrs: dict[str, Any] = {
        **semantic_conventions.crewai_span(
            event_name=event.type, subject=event.a2a_agent_name or event.endpoint
        ),
        **semantic_conventions.crewai_a2a(
            endpoint=event.endpoint,
            agent_id=event.agent_id,
            context_id=event.context_id,
            a2a_agent_name=event.a2a_agent_name,
            agent_card=_serialize(event.agent_card) if event.agent_card else None,
            protocol_version=event.protocol_version,
            skill_id=event.skill_id,
            metadata=_serialize(event.metadata) if event.metadata else None,
        ),
    }

    span = _start_span(providers, ctx, "a2a conversation", event, attrs)
    if span:
        _store_span(ctx, event.event_id, span)

    providers.emit_log(
        f"A2A conversation started: {event.a2a_agent_name or event.endpoint}",
        level="INFO",
        span=span,
        attributes=attrs,
        ctx=ctx,
    )


def handle_a2a_conversation_completed(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    source: Any,
    event: A2AConversationCompletedEvent,
) -> None:
    attrs: dict[str, Any] = {
        **semantic_conventions.crewai_span(
            event_name=event.type, subject=event.a2a_agent_name or event.endpoint
        ),
        **semantic_conventions.crewai_a2a(
            endpoint=event.endpoint,
            context_id=event.context_id,
            a2a_agent_name=event.a2a_agent_name,
            status=event.status,
            final_result=event.final_result,
            total_turns=event.total_turns,
            metadata=_serialize(event.metadata) if event.metadata else None,
        ),
    }

    error = (event.error or "Conversation failed") if event.status == "failed" else None

    span = _end_span(ctx, event.started_event_id, event, attrs, error=error)

    providers.emit_log(
        f"A2A conversation completed: {event.a2a_agent_name or event.endpoint}",
        level="ERROR" if error else "INFO",
        span=span,
        attributes=attrs,
        ctx=ctx,
    )


# ---------------------------------------------------------------------------
# A2A Server Task handlers
# ---------------------------------------------------------------------------


def handle_a2a_server_task_started(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    source: Any,
    event: A2AServerTaskStartedEvent,
) -> None:
    attrs: dict[str, Any] = {
        **semantic_conventions.crewai_span(
            event_name=event.type, subject="a2a server task"
        ),
        **semantic_conventions.crewai_a2a(
            task_id=event.task_id,
            context_id=event.context_id,
            metadata=_serialize(event.metadata) if event.metadata else None,
        ),
    }

    span = _start_span(providers, ctx, "a2a server task", event, attrs)
    if span:
        _store_span(ctx, event.event_id, span)

    providers.emit_log(
        f"A2A server task started: {event.task_id}",
        level="INFO",
        span=span,
        attributes=attrs,
        ctx=ctx,
    )


def handle_a2a_server_task_completed(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    source: Any,
    event: A2AServerTaskCompletedEvent,
) -> None:
    attrs: dict[str, Any] = {
        **semantic_conventions.crewai_span(
            event_name=event.type, subject="a2a server task"
        ),
        **semantic_conventions.crewai_a2a(
            task_id=event.task_id,
            context_id=event.context_id,
            result=event.result,
            metadata=_serialize(event.metadata) if event.metadata else None,
        ),
    }

    providers.emit_log(
        f"A2A server task completed: {event.task_id}",
        level="INFO",
        attributes=attrs,
        ctx=ctx,
    )

    _end_span(ctx, event.started_event_id, event, attrs)


def handle_a2a_server_task_canceled(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    source: Any,
    event: A2AServerTaskCanceledEvent,
) -> None:
    attrs: dict[str, Any] = {
        **semantic_conventions.crewai_span(
            event_name=event.type, subject="a2a server task"
        ),
        **semantic_conventions.crewai_a2a(
            task_id=event.task_id,
            context_id=event.context_id,
            metadata=_serialize(event.metadata) if event.metadata else None,
        ),
    }

    span = _end_span(ctx, event.started_event_id, event, attrs, error="Task canceled")

    providers.emit_log(
        f"A2A server task canceled: {event.task_id}",
        level="ERROR",
        span=span,
        attributes=attrs,
        ctx=ctx,
    )


def handle_a2a_server_task_failed(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    source: Any,
    event: A2AServerTaskFailedEvent,
) -> None:
    attrs: dict[str, Any] = {
        **semantic_conventions.crewai_span(
            event_name=event.type, subject="a2a server task"
        ),
        **semantic_conventions.crewai_a2a(
            task_id=event.task_id,
            context_id=event.context_id,
            metadata=_serialize(event.metadata) if event.metadata else None,
        ),
    }

    span = _end_span(ctx, event.started_event_id, event, attrs, error=event.error)

    providers.emit_log(
        f"A2A server task failed: {event.task_id}",
        level="ERROR",
        span=span,
        attributes=attrs,
        ctx=ctx,
    )


# ---------------------------------------------------------------------------
# A2A Parallel Delegation handlers
# ---------------------------------------------------------------------------


def handle_a2a_parallel_delegation_started(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    source: Any,
    event: A2AParallelDelegationStartedEvent,
) -> None:
    attrs: dict[str, Any] = {
        **semantic_conventions.crewai_span(
            event_name=event.type, subject="a2a parallel delegate"
        ),
        **semantic_conventions.crewai_a2a(
            endpoints=_serialize(event.endpoints),
            task_description=event.task_description,
        ),
    }

    span = _start_span(providers, ctx, "a2a parallel delegate", event, attrs)
    if span:
        _store_span(ctx, event.event_id, span)

    providers.emit_log(
        "A2A parallel delegation started",
        level="INFO",
        span=span,
        attributes=attrs,
        ctx=ctx,
    )


def handle_a2a_parallel_delegation_completed(
    providers: TelemetryProviders,
    ctx: TelemetryExecutionContext,
    source: Any,
    event: A2AParallelDelegationCompletedEvent,
) -> None:
    attrs: dict[str, Any] = {
        **semantic_conventions.crewai_span(
            event_name=event.type, subject="a2a parallel delegate"
        ),
        **semantic_conventions.crewai_a2a(
            endpoints=_serialize(event.endpoints),
            success_count=event.success_count,
            failure_count=event.failure_count,
            results=_serialize(event.results) if event.results else None,
        ),
    }

    error = (
        "All delegations failed"
        if event.success_count == 0 and event.failure_count > 0
        else None
    )

    span = _end_span(ctx, event.started_event_id, event, attrs, error=error)

    providers.emit_log(
        "A2A parallel delegation completed",
        level="ERROR" if error else "INFO",
        span=span,
        attributes=attrs,
        ctx=ctx,
    )


EVENT_HANDLERS: dict[type[Any], Callable[..., None]] = {
    CrewKickoffStartedEvent: handle_crew_kickoff_started,
    CrewKickoffCompletedEvent: handle_crew_kickoff_completed,
    CrewKickoffFailedEvent: handle_crew_kickoff_failed,
    TaskStartedEvent: handle_task_started,
    TaskCompletedEvent: handle_task_completed,
    TaskFailedEvent: handle_task_failed,
    AgentExecutionStartedEvent: handle_agent_execution_started,
    AgentExecutionCompletedEvent: handle_agent_execution_completed,
    AgentExecutionErrorEvent: handle_agent_execution_error,
    ToolUsageStartedEvent: handle_tool_usage_started,
    ToolUsageFinishedEvent: handle_tool_usage_finished,
    ToolUsageErrorEvent: handle_tool_usage_error,
    ToolFailureDetectedEvent: handle_tool_failure_detected,
    SkillUsedEvent: handle_skill_used,
    SkillDiscoveryStartedEvent: handle_skill_discovery_started,
    SkillDiscoveryCompletedEvent: handle_skill_discovery_completed,
    SkillLoadedEvent: handle_skill_loaded,
    SkillActivatedEvent: handle_skill_activated,
    SkillLoadFailedEvent: handle_skill_load_failed,
    FlowCreatedEvent: handle_flow_created,
    FlowStartedEvent: handle_flow_started,
    FlowFinishedEvent: handle_flow_finished,
    FlowFailedEvent: handle_flow_failed,
    FlowPausedEvent: handle_flow_paused,
    MethodExecutionStartedEvent: handle_method_execution_started,
    MethodExecutionFinishedEvent: handle_method_execution_finished,
    MethodExecutionFailedEvent: handle_method_execution_failed,
    MethodExecutionPausedEvent: handle_method_execution_paused,
    HumanFeedbackRequestedEvent: handle_human_feedback_requested,
    HumanFeedbackReceivedEvent: handle_human_feedback_received,
    LLMCallStartedEvent: handle_llm_call_started,
    LLMCallCompletedEvent: handle_llm_call_completed,
    LLMCallFailedEvent: handle_llm_call_failed,
    LiteAgentExecutionStartedEvent: handle_lite_agent_execution_started,
    LiteAgentExecutionCompletedEvent: handle_lite_agent_execution_completed,
    LiteAgentExecutionErrorEvent: handle_lite_agent_execution_error,
    AgentReasoningStartedEvent: handle_agent_reasoning_started,
    AgentReasoningCompletedEvent: handle_agent_reasoning_completed,
    AgentReasoningFailedEvent: handle_agent_reasoning_failed,
    LLMGuardrailStartedEvent: handle_llm_guardrail_started,
    LLMGuardrailCompletedEvent: handle_llm_guardrail_completed,
    MemoryQueryStartedEvent: handle_memory_query_started,
    MemoryQueryCompletedEvent: handle_memory_query_completed,
    MemoryQueryFailedEvent: handle_memory_query_failed,
    MemoryRetrievalStartedEvent: handle_memory_retrieval_started,
    MemoryRetrievalCompletedEvent: handle_memory_retrieval_completed,
    MemoryRetrievalFailedEvent: handle_memory_retrieval_failed,
    MemorySaveStartedEvent: handle_memory_save_started,
    MemorySaveCompletedEvent: handle_memory_save_completed,
    MemorySaveFailedEvent: handle_memory_save_failed,
    KnowledgeQueryStartedEvent: handle_knowledge_query_started,
    KnowledgeQueryCompletedEvent: handle_knowledge_query_completed,
    KnowledgeQueryFailedEvent: handle_knowledge_query_failed,
    KnowledgeRetrievalStartedEvent: handle_knowledge_retrieval_started,
    KnowledgeRetrievalCompletedEvent: handle_knowledge_retrieval_completed,
    KnowledgeSearchQueryFailedEvent: handle_knowledge_search_query_failed,
    MCPConnectionStartedEvent: handle_mcp_connection_started,
    MCPConnectionCompletedEvent: handle_mcp_connection_completed,
    MCPConnectionFailedEvent: handle_mcp_connection_failed,
    MCPToolExecutionStartedEvent: handle_mcp_tool_execution_started,
    MCPToolExecutionCompletedEvent: handle_mcp_tool_execution_completed,
    MCPToolExecutionFailedEvent: handle_mcp_tool_execution_failed,
    A2ADelegationStartedEvent: handle_a2a_delegation_started,
    A2ADelegationCompletedEvent: handle_a2a_delegation_completed,
    A2AConversationStartedEvent: handle_a2a_conversation_started,
    A2AConversationCompletedEvent: handle_a2a_conversation_completed,
    A2AServerTaskStartedEvent: handle_a2a_server_task_started,
    A2AServerTaskCompletedEvent: handle_a2a_server_task_completed,
    A2AServerTaskCanceledEvent: handle_a2a_server_task_canceled,
    A2AServerTaskFailedEvent: handle_a2a_server_task_failed,
    A2AParallelDelegationStartedEvent: handle_a2a_parallel_delegation_started,
    A2AParallelDelegationCompletedEvent: handle_a2a_parallel_delegation_completed,
}
