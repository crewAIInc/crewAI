"""The call llm span reports token usage in every shape a provider emits.

Gemini and Bedrock do not use the OpenAI or Anthropic usage keys, so a span
that read only those keys reported zero input tokens for Gemini and no usage
at all for Bedrock, and the flow-level crew totals missed the same calls.
"""

from __future__ import annotations

from typing import Any

from crewai.events.types.llm_events import (
    LLMCallCompletedEvent,
    LLMCallStartedEvent,
    LLMCallType,
)
from crewai.telemetry.tracing import handlers
from crewai.telemetry.tracing.context import TelemetryExecutionContext
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Tracer
import pytest


@pytest.fixture(autouse=True)
def enable_otel_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """The suite otherwise runs with OTEL_SDK_DISABLED, which makes every
    assertion here pass vacuously against non-recording spans."""
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    monkeypatch.delenv("CREWAI_DISABLE_TELEMETRY", raising=False)
    monkeypatch.delenv("CREWAI_DISABLE_TRACKING", raising=False)


class _Providers:
    def __init__(self, tracer: Tracer) -> None:
        self._tracer = tracer

    def get_tracer(self, name: str | None = None) -> Tracer:
        return self._tracer

    def emit_log(self, *args: Any, **kwargs: Any) -> None:
        pass


@pytest.mark.parametrize(
    "usage",
    [
        {"prompt_tokens": 50, "completion_tokens": 25, "total_tokens": 75},
        # GeminiCompletion._extract_token_usage
        {
            "prompt_token_count": 50,
            "candidates_token_count": 25,
            "completion_tokens": 25,
            "total_token_count": 75,
            "total_tokens": 75,
            "cached_prompt_tokens": 0,
            "reasoning_tokens": 0,
        },
        # Bedrock Converse response["usage"], passed through unchanged
        {
            "inputTokens": 50,
            "outputTokens": 25,
            "totalTokens": 75,
            "cacheReadInputTokens": 0,
            "cacheWriteInputTokens": 0,
        },
    ],
    ids=["openai", "gemini", "bedrock"],
)
def test_call_llm_span_and_flow_totals_count_provider_usage(
    usage: dict[str, Any],
) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    ctx = TelemetryExecutionContext(
        kickoff_id="kickoff", automation_name="test", tracer=tracer
    )
    ctx.flow_crew_usage_metrics["flow"] = {
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cached_prompt_tokens": 0,
        "reasoning_tokens": 0,
        "cache_creation_tokens": 0,
        "successful_requests": 0,
        "crew_count": 0,
    }
    providers = _Providers(tracer)

    started = LLMCallStartedEvent(
        model="m", messages=[{"role": "user", "content": "hi"}], call_id="c1"
    )
    handlers.handle_llm_call_started(providers, ctx, None, started)
    completed = LLMCallCompletedEvent(
        model="m",
        response="hello",
        call_type=LLMCallType.LLM_CALL,
        call_id="c1",
        usage=usage,
    )
    completed.started_event_id = started.event_id
    handlers.handle_llm_call_completed(providers, ctx, None, completed)

    [span] = [s for s in exporter.get_finished_spans() if s.name == "call llm"]
    assert span.attributes["gen_ai.usage.input_tokens"] == 50
    assert span.attributes["gen_ai.usage.output_tokens"] == 25
    flow_metrics = ctx.flow_crew_usage_metrics["flow"]
    assert flow_metrics["prompt_tokens"] == 50
    assert flow_metrics["completion_tokens"] == 25
    assert flow_metrics["total_tokens"] == 75
    assert flow_metrics["successful_requests"] == 1


def test_fully_cached_bedrock_call_still_counts() -> None:
    """A Bedrock call can read its whole prompt from cache and return no
    output, so inputTokens and outputTokens are 0 while totalTokens is not."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    ctx = TelemetryExecutionContext(
        kickoff_id="kickoff", automation_name="test", tracer=tracer
    )
    ctx.flow_crew_usage_metrics["flow"] = {
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cached_prompt_tokens": 0,
        "reasoning_tokens": 0,
        "cache_creation_tokens": 0,
        "successful_requests": 0,
        "crew_count": 0,
    }
    providers = _Providers(tracer)

    started = LLMCallStartedEvent(
        model="m", messages=[{"role": "user", "content": "hi"}], call_id="c1"
    )
    handlers.handle_llm_call_started(providers, ctx, None, started)
    completed = LLMCallCompletedEvent(
        model="m",
        response="",
        call_type=LLMCallType.LLM_CALL,
        call_id="c1",
        usage={
            "inputTokens": 0,
            "outputTokens": 0,
            "totalTokens": 40,
            "cacheReadInputTokens": 40,
            "cacheWriteInputTokens": 0,
        },
    )
    completed.started_event_id = started.event_id
    handlers.handle_llm_call_completed(providers, ctx, None, completed)

    [span] = [s for s in exporter.get_finished_spans() if s.name == "call llm"]
    assert span.attributes["gen_ai.usage.cache_read.input_tokens"] == 40
    flow_metrics = ctx.flow_crew_usage_metrics["flow"]
    assert flow_metrics["total_tokens"] == 40
    assert flow_metrics["cached_prompt_tokens"] == 40
    assert flow_metrics["successful_requests"] == 1
