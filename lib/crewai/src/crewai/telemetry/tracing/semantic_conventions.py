"""Typed semantic convention attributes for OpenTelemetry spans.

Each function maps Python keyword arguments to standard or namespaced
OTEL attribute keys, returning a plain dict with None values filtered:

    attrs = {
        **semconv.gen_ai(operation_name="chat", request_model="gpt-4"),
        **semconv.crewai_span(event_name="llm_call_started", subject="gpt-4"),
    }

Standard conventions:
    gen_ai()        https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/

Custom conventions (crewai.* namespace):
    crewai_span()   Common attributes on every CrewAI span
    crewai_crew()   crewai.crew.*
    crewai_task()   crewai.task.*
    crewai_agent()  crewai.agent.*
    crewai_flow()   crewai.flow.*
    crewai_method()    crewai.method.*
    crewai_reasoning()  crewai.reasoning.*
    crewai_guardrail()  crewai.guardrail.*
    crewai_policy()     crewai.policy.*
    crewai_memory()     crewai.memory.*
    crewai_knowledge()  crewai.knowledge.*
    crewai_llm()        crewai.llm.*
    crewai_mcp()        crewai.mcp.*
    crewai_a2a()        crewai.a2a.*
    crewai_human_feedback()  crewai.human_feedback.*
"""

from __future__ import annotations

from collections.abc import Callable
import json
import logging
from typing import Any

from crewai.telemetry.tracing import gen_ai_shapes


logger = logging.getLogger(__name__)


# Attribute keys for duration and count metrics.
# Use these constants instead of hardcoding the strings to keep a single
# source of truth.  The dict-building functions below use the same keys.
CREW_EXECUTION_DURATION_MS = "crewai.crew.execution_duration_ms"
AGENT_EXECUTION_DURATION_MS = "crewai.agent.execution_duration_ms"
AGENT_LLM_CALLS_COUNT = "crewai.agent.llm_calls_count"
FLOW_EXECUTION_DURATION_MS = "crewai.flow.execution_duration_ms"
METHOD_DURATION_MS = "crewai.method.duration_ms"
MEMORY_QUERY_DURATION_MS = "crewai.memory.query_duration_ms"
MEMORY_RETRIEVAL_DURATION_MS = "crewai.memory.retrieval_duration_ms"
MEMORY_SAVE_DURATION_MS = "crewai.memory.save_duration_ms"
MCP_CONNECTION_DURATION_MS = "crewai.mcp.connection_duration_ms"
MCP_TOOL_EXECUTION_DURATION_MS = "crewai.mcp.tool_execution_duration_ms"
HUMAN_FEEDBACK_WAIT_DURATION_MS = "crewai.human_feedback.wait_duration_ms"
HUMAN_FEEDBACK_REQUEST_ID = "crewai.human_feedback.request_id"

GEN_AI_OP_INVOKE_WORKFLOW = "invoke_workflow"
GEN_AI_OP_EXECUTE_METHOD = "execute_method"
GEN_AI_OP_EXECUTE_TASK = "execute_task"


def _filter_none(attrs: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in attrs.items() if v is not None}


def _safe_shape(
    transform: Callable[[Any], Any], value: Any, attr: str
) -> tuple[str | None, dict[str, Any]]:
    try:
        normalized = transform(value)
        if normalized is None:
            return None, {}
        serialized = json.dumps(normalized, default=str)
        return gen_ai_shapes.truncate_attr(serialized, attr=attr)
    except Exception:
        logger.warning(
            "OTel %s serialization failed; dropping attribute", attr, exc_info=True
        )
        return None, {}


def _safe_output(
    response: Any,
    *,
    explicit_finish: str | None = None,
) -> tuple[str | None, list[str] | None, dict[str, Any]]:
    """Shape ``response`` into the OTel ``gen_ai.output.messages`` payload.

    When ``explicit_finish`` is supplied (the OSS event surfaced a real
    provider finish reason) every shaped message's ``finish_reason`` is
    overridden so the per-message field stays in lockstep with the
    span-level ``gen_ai.response.finish_reasons``. Without this the inferred
    default (typically ``"stop"``) would be embedded in the JSON payload
    while the span attribute shows e.g. ``length`` / ``content_filter``.

    The returned tuple also carries ``marker_attrs`` populated when the
    serialized payload exceeded the per-attribute byte cap; see
    :func:`crewai.telemetry.tracing.gen_ai_shapes.truncate_attr`. Any failure
    in shaping, serialization, or truncation drops the attribute and logs.
    """
    try:
        shaped = gen_ai_shapes.to_output_messages(response)
        if shaped is None:
            return None, None, {}
        if explicit_finish:
            for msg in shaped:
                if isinstance(msg, dict):
                    msg["finish_reason"] = explicit_finish
        finish_reasons = gen_ai_shapes.finish_reasons_from_messages(shaped)
        serialized = json.dumps(shaped, default=str)
        payload, markers = gen_ai_shapes.truncate_attr(
            serialized, attr="gen_ai.output.messages"
        )
        return payload, finish_reasons, markers
    except Exception:
        logger.warning(
            "OTel gen_ai.output.messages serialization failed; dropping attribute",
            exc_info=True,
        )
        return None, None, {}


def _payload_size(payload: str | None) -> int | None:
    """Return ``len(payload)`` for an already-serialized JSON string.

    Used for ``gen_ai.input.messages.size`` / ``gen_ai.output.messages.size``
    so consumers can budget storage / display without re-shaping the parts
    payload. Reflects the post-truncation size — when truncation fired,
    ``<attr>.original_size_bytes`` carries the pre-truncation byte count
    (see :func:`crewai.telemetry.tracing.gen_ai_shapes.truncate_attr`).
    """
    if payload is None:
        return None
    return len(payload)


def gen_ai(
    *,
    operation_name: str | None = None,
    agent_name: str | None = None,
    agent_id: str | None = None,
    agent_description: str | None = None,
    workflow_name: str | None = None,
    request_model: str | None = None,
    response_model: str | None = None,
    provider_name: str | None = None,
    system_instructions: str | None = None,
    tool_definitions: list[Any] | None = None,
    tool_name: str | None = None,
    tool_type: str | None = None,
    tool_call_arguments: dict[str, Any] | str | None = None,
    tool_call_result: Any = None,
    input_messages: Any = None,
    output_messages: Any = None,
    output_type: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    stream: bool | None = None,
    seed: int | None = None,
    stop_sequences: list[str] | None = None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
    choice_count: int | None = None,
    response_id: str | None = None,
    finish_reason: str | None = None,
    conversation_id: str | None = None,
) -> dict[str, Any]:
    """Build OTel GenAI span attributes from raw event values.

    Owns both the attribute keys (`gen_ai.input.messages`, `gen_ai.tool.*`, …)
    and the value shapes (parts arrays, finish_reason enums, …) defined in the
    OTel GenAI semantic convention. Spec-shape transforms live in
    `gen_ai_shapes` and are invoked through `_safe_shape` /  `_safe_output`,
    which swallow failures so instrumentation never raises into user code.

    Sampling params (``temperature``/``top_p``/…) and response identifiers
    (``finish_reason``/``response_id``) are optional so older OSS ``crewai``
    callers that don't surface those fields keep producing valid spans. When
    ``finish_reason`` is supplied it overrides the value inferred from the
    output payload shape.

    Spec: https://opentelemetry.io/docs/specs/semconv/gen-ai/
    """
    explicit_finish = gen_ai_shapes.coerce_finish_reason(finish_reason)
    output_payload, inferred_finish_reasons, output_markers = _safe_output(
        output_messages, explicit_finish=explicit_finish
    )
    response_finish_reasons = (
        [explicit_finish] if explicit_finish else inferred_finish_reasons
    )
    input_payload, input_markers = _safe_shape(
        gen_ai_shapes.to_input_messages,
        input_messages,
        "gen_ai.input.messages",
    )
    system_payload, system_markers = _safe_shape(
        gen_ai_shapes.to_system_instructions,
        system_instructions,
        "gen_ai.system_instructions",
    )
    tool_definitions_payload, tool_definitions_markers = _safe_shape(
        gen_ai_shapes.to_tool_definitions,
        tool_definitions,
        "gen_ai.tool.definitions",
    )
    tool_args_payload, tool_args_markers = _safe_shape(
        gen_ai_shapes.to_tool_call_arguments,
        tool_call_arguments,
        "gen_ai.tool.call.arguments",
    )
    tool_result_payload, tool_result_markers = _safe_shape(
        gen_ai_shapes.to_tool_call_result,
        tool_call_result,
        "gen_ai.tool.call.result",
    )

    return _filter_none(
        {
            "gen_ai.operation.name": operation_name,
            "gen_ai.agent.name": agent_name,
            "gen_ai.agent.id": agent_id,
            "gen_ai.agent.description": agent_description,
            "gen_ai.workflow.name": workflow_name,
            "gen_ai.request.model": request_model,
            "gen_ai.response.model": response_model,
            "gen_ai.provider.name": provider_name,
            "gen_ai.output.type": output_type,
            "gen_ai.request.temperature": temperature,
            "gen_ai.request.top_p": top_p,
            "gen_ai.request.max_tokens": max_tokens,
            "gen_ai.request.stream": stream,
            "gen_ai.request.seed": seed,
            "gen_ai.request.stop_sequences": stop_sequences,
            "gen_ai.request.frequency_penalty": frequency_penalty,
            "gen_ai.request.presence_penalty": presence_penalty,
            "gen_ai.request.choice.count": choice_count,
            "gen_ai.response.id": response_id,
            "gen_ai.system_instructions": system_payload,
            "gen_ai.tool.definitions": tool_definitions_payload,
            "gen_ai.tool.name": tool_name,
            "gen_ai.tool.type": tool_type,
            "gen_ai.tool.call.arguments": tool_args_payload,
            "gen_ai.tool.call.result": tool_result_payload,
            "gen_ai.input.messages": input_payload,
            "gen_ai.input.messages.size": _payload_size(input_payload),
            "gen_ai.output.messages": output_payload,
            "gen_ai.output.messages.size": _payload_size(output_payload),
            "gen_ai.response.finish_reasons": response_finish_reasons,
            "gen_ai.conversation.id": conversation_id,
            **input_markers,
            **output_markers,
            **system_markers,
            **tool_definitions_markers,
            **tool_args_markers,
            **tool_result_markers,
        }
    )


def gen_ai_usage(
    *,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    cached_input_tokens: int | None = None,
    reasoning_tokens: int | None = None,
    cache_creation_tokens: int | None = None,
) -> dict[str, Any]:
    return _filter_none(
        {
            "gen_ai.usage.input_tokens": input_tokens,
            "gen_ai.usage.output_tokens": output_tokens,
            "gen_ai.usage.cache_read.input_tokens": cached_input_tokens,
            "gen_ai.usage.cache_creation.input_tokens": cache_creation_tokens,
            "gen_ai.usage.reasoning_tokens": reasoning_tokens,
        }
    )


def gen_ai_io(
    *, input_value: str | None = None, output_value: str | None = None
) -> dict[str, Any]:
    """Input/output for non-LLM (workflow/orchestration) spans.

    The OTel GenAI spec requires ``gen_ai.input.messages`` /
    ``gen_ai.output.messages`` to follow the input-messages JSON schema (an
    array of ``{role, parts:[{type, content}]}``). Orchestration spans carry
    arbitrary already-serialized payloads (handlers call ``_serialize``), so they
    are shaped through the same ``gen_ai_shapes`` transforms ``gen_ai()`` uses —
    each value becomes a single text message that stays schema-conformant and
    renders across GenAI backends.
    """
    input_payload, input_markers = _safe_shape(
        gen_ai_shapes.to_input_messages, input_value, "gen_ai.input.messages"
    )
    output_payload, output_markers = _safe_shape(
        gen_ai_shapes.to_output_messages, output_value, "gen_ai.output.messages"
    )
    return _filter_none(
        {
            "gen_ai.input.messages": input_payload,
            "gen_ai.output.messages": output_payload,
            **input_markers,
            **output_markers,
        }
    )


def crewai_span(
    *,
    event_name: str | None = None,
    subject: str | None = None,
) -> dict[str, Any]:
    return _filter_none(
        {
            "crewai.event_name": event_name,
            "crewai.subject": subject,
        }
    )


def crewai_crew(
    *,
    key: str | None = None,
    name: str | None = None,
    id: str | None = None,
    inputs: str | None = None,
    process: str | None = None,
    num_tasks: int | None = None,
    num_agents: int | None = None,
    output: str | None = None,
    usage_metrics: str | None = None,
    execution_duration_ms: float | None = None,
) -> dict[str, Any]:
    return _filter_none(
        {
            "crewai.crew.key": key,
            "crewai.crew.name": name,
            "crewai.crew.id": id,
            "crewai.crew.inputs": inputs,
            "crewai.crew.process": process,
            "crewai.crew.num_tasks": num_tasks,
            "crewai.crew.num_agents": num_agents,
            "crewai.crew.output": output,
            "crewai.crew.usage_metrics": usage_metrics,
            CREW_EXECUTION_DURATION_MS: execution_duration_ms,
        }
    )


def crewai_task(
    *,
    key: str | None = None,
    id: str | None = None,
    name: str | None = None,
    description: str | None = None,
    expected_output: str | None = None,
    output: str | None = None,
) -> dict[str, Any]:
    return _filter_none(
        {
            "crewai.task.key": key,
            "crewai.task.id": id,
            "crewai.task.name": name,
            "crewai.task.description": description,
            "crewai.task.expected_output": expected_output,
            "crewai.task.output": output,
        }
    )


def crewai_agent(
    *,
    role: str | None = None,
    key: str | None = None,
    llm_calls_count: int | None = None,
    execution_duration_ms: float | None = None,
) -> dict[str, Any]:
    return _filter_none(
        {
            "crewai.agent.role": role,
            "crewai.agent.key": key,
            AGENT_LLM_CALLS_COUNT: llm_calls_count,
            AGENT_EXECUTION_DURATION_MS: execution_duration_ms,
        }
    )


def crewai_flow(
    *,
    name: str | None = None,
    id: str | None = None,
    method_names: str | None = None,
    inputs: str | None = None,
    result: str | None = None,
    execution_duration_ms: float | None = None,
    aggregated_crew_usage_metrics: str | None = None,
) -> dict[str, Any]:
    return _filter_none(
        {
            "crewai.flow.name": name,
            "crewai.flow.id": id,
            "crewai.flow.method_names": method_names,
            "crewai.flow.inputs": inputs,
            "crewai.flow.result": result,
            FLOW_EXECUTION_DURATION_MS: execution_duration_ms,
            "crewai.flow.aggregated_crew_usage_metrics": aggregated_crew_usage_metrics,
        }
    )


def crewai_method(
    *,
    name: str | None = None,
    state: str | None = None,
    params: str | None = None,
    result: str | None = None,
    duration_ms: float | None = None,
) -> dict[str, Any]:
    return _filter_none(
        {
            "crewai.method.name": name,
            "crewai.method.state": state,
            "crewai.method.params": params,
            "crewai.method.result": result,
            METHOD_DURATION_MS: duration_ms,
        }
    )


def crewai_reasoning(
    *,
    attempt: int | None = None,
    plan: str | None = None,
    ready: bool | None = None,
) -> dict[str, Any]:
    return _filter_none(
        {
            "crewai.reasoning.attempt": attempt,
            "crewai.reasoning.plan": plan,
            "crewai.reasoning.ready": ready,
        }
    )


def crewai_guardrail(
    *,
    guardrail: str | None = None,
    guardrail_type: str | None = None,
    retry_count: int | None = None,
    success: bool | None = None,
    result: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    return _filter_none(
        {
            "crewai.guardrail.description": guardrail,
            "crewai.guardrail.type": guardrail_type,
            "crewai.guardrail.retry_count": retry_count,
            "crewai.guardrail.success": success,
            "crewai.guardrail.result": result,
            "crewai.guardrail.error": error,
        }
    )


def crewai_policy(
    *,
    id: str | None = None,
    name: str | None = None,
    decision: str | None = None,
    mode: str | None = None,
    point: str | None = None,
    reason: str | None = None,
    blocking: bool | None = None,
    degradation_stage: str | None = None,
    degradation_error: str | None = None,
    absent_fields: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    return _filter_none(
        {
            "crewai.policy.id": id,
            "crewai.policy.name": name,
            "crewai.policy.decision": decision,
            "crewai.policy.mode": mode,
            "crewai.policy.point": point,
            "crewai.policy.reason": reason,
            "crewai.policy.blocking": blocking,
            "crewai.policy.degradation.stage": degradation_stage,
            "crewai.policy.degradation.error": degradation_error,
            "crewai.policy.absent_fields": (
                ",".join(absent_fields) if absent_fields else None
            ),
        }
    )


def crewai_memory(
    *,
    query: str | None = None,
    value: str | None = None,
    limit: int | None = None,
    score_threshold: float | None = None,
    results: str | None = None,
    memory_content: str | None = None,
    metadata: str | None = None,
) -> dict[str, Any]:
    return _filter_none(
        {
            "crewai.memory.query": query,
            "crewai.memory.value": value,
            "crewai.memory.limit": limit,
            "crewai.memory.score_threshold": score_threshold,
            "crewai.memory.results": results,
            "crewai.memory.content": memory_content,
            "crewai.memory.metadata": metadata,
        }
    )


def crewai_knowledge(
    *,
    task_prompt: str | None = None,
    query: str | None = None,
    retrieved_knowledge: str | None = None,
) -> dict[str, Any]:
    return _filter_none(
        {
            "crewai.knowledge.task_prompt": task_prompt,
            "crewai.knowledge.query": query,
            "crewai.knowledge.retrieved_knowledge": retrieved_knowledge,
        }
    )


def crewai_tool_failure(
    *,
    message: str | None = None,
    reason: str | None = None,
    code: str | None = None,
    retryable: bool | None = None,
    policy: str | None = None,
) -> dict[str, Any]:
    """Attributes for a tool that ran to completion but reported it failed.

    Distinct from a tool that *raised*: the call returned normally, so without
    these the span looks identical to a successful one.
    """
    return _filter_none(
        {
            "crewai.tool.failure.message": message,
            "crewai.tool.failure.reason": reason,
            "crewai.tool.failure.code": code,
            "crewai.tool.failure.retryable": retryable,
            "crewai.tool.failure.policy": policy,
        }
    )


def crewai_skill(
    *,
    name: str | None = None,
    path: str | None = None,
    disclosure_level: int | None = None,
    search_path: str | None = None,
    skills_found: int | None = None,
    skill_names: str | None = None,
) -> dict[str, Any]:
    return _filter_none(
        {
            "crewai.skill.name": name,
            "crewai.skill.path": path,
            "crewai.skill.disclosure_level": disclosure_level,
            "crewai.skill.search_path": search_path,
            "crewai.skill.skills_found": skills_found,
            "crewai.skill.skill_names": skill_names,
        }
    )


def crewai_llm(
    *,
    call_id: str | None = None,
    callbacks: str | None = None,
    available_functions: str | None = None,
    call_type: str | None = None,
) -> dict[str, Any]:
    return _filter_none(
        {
            "crewai.llm.call_id": call_id,
            "crewai.llm.callbacks": callbacks,
            "crewai.llm.available_functions": available_functions,
            "crewai.llm.call_type": call_type,
        }
    )


def crewai_mcp(
    *,
    server_name: str | None = None,
    server_url: str | None = None,
    transport_type: str | None = None,
    tool_name: str | None = None,
    tool_args: str | None = None,
    tool_result: str | None = None,
    connect_timeout: int | None = None,
    is_reconnect: bool | None = None,
    error_type: str | None = None,
) -> dict[str, Any]:
    return _filter_none(
        {
            "crewai.mcp.server_name": server_name,
            "crewai.mcp.server_url": server_url,
            "crewai.mcp.transport_type": transport_type,
            "crewai.mcp.tool_name": tool_name,
            "crewai.mcp.tool_args": tool_args,
            "crewai.mcp.tool_result": tool_result,
            "crewai.mcp.connect_timeout": connect_timeout,
            "crewai.mcp.is_reconnect": is_reconnect,
            "crewai.mcp.error_type": error_type,
        }
    )


def crewai_human_feedback(
    *,
    method_name: str | None = None,
    message: str | None = None,
    feedback: str | None = None,
    outcome: str | None = None,
    emit: str | None = None,
    wait_duration_ms: float | None = None,
    request_id: str | None = None,
) -> dict[str, Any]:
    return _filter_none(
        {
            "crewai.human_feedback.method_name": method_name,
            "crewai.human_feedback.message": message,
            "crewai.human_feedback.feedback": feedback,
            "crewai.human_feedback.outcome": outcome,
            "crewai.human_feedback.emit": emit,
            HUMAN_FEEDBACK_WAIT_DURATION_MS: wait_duration_ms,
            HUMAN_FEEDBACK_REQUEST_ID: request_id,
        }
    )


def crewai_a2a(
    *,
    endpoint: str | None = None,
    endpoints: str | None = None,
    task_description: str | None = None,
    agent_id: str | None = None,
    context_id: str | None = None,
    is_multiturn: bool | None = None,
    turn_number: int | None = None,
    a2a_agent_name: str | None = None,
    agent_card: str | None = None,
    protocol_version: str | None = None,
    skill_id: str | None = None,
    metadata: str | None = None,
    status: str | None = None,
    result: str | None = None,
    final_result: str | None = None,
    total_turns: int | None = None,
    task_id: str | None = None,
    success_count: int | None = None,
    failure_count: int | None = None,
    results: str | None = None,
) -> dict[str, Any]:
    return _filter_none(
        {
            "crewai.a2a.endpoint": endpoint,
            "crewai.a2a.endpoints": endpoints,
            "crewai.a2a.task_description": task_description,
            "crewai.a2a.agent_id": agent_id,
            "crewai.a2a.context_id": context_id,
            "crewai.a2a.is_multiturn": is_multiturn,
            "crewai.a2a.turn_number": turn_number,
            "crewai.a2a.agent_name": a2a_agent_name,
            "crewai.a2a.agent_card": agent_card,
            "crewai.a2a.protocol_version": protocol_version,
            "crewai.a2a.skill_id": skill_id,
            "crewai.a2a.metadata": metadata,
            "crewai.a2a.status": status,
            "crewai.a2a.result": result,
            "crewai.a2a.final_result": final_result,
            "crewai.a2a.total_turns": total_turns,
            "crewai.a2a.task_id": task_id,
            "crewai.a2a.success_count": success_count,
            "crewai.a2a.failure_count": failure_count,
            "crewai.a2a.results": results,
        }
    )
