"""Offline checks for the manual Flow/Agent/MCP tracing runner."""

import asyncio
from datetime import datetime, timedelta, timezone
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from uuid import uuid4

from crewai.execution import get_execution_uuid
import pytest


@pytest.fixture
def runner(monkeypatch):
    path = Path(__file__).resolve().parents[4] / "scripts" / "wharf_runner.py"
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args: None)
    monkeypatch.setenv("OTEL_SDK_DISABLED", "false")
    monkeypatch.setenv("CREWAI_DISABLE_TELEMETRY", "true")
    monkeypatch.setenv("FIRECRAWL_API_KEY", "synthetic-firecrawl-key")
    spec = importlib.util.spec_from_file_location("wharf_runner", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "wharf_runner", module)
    spec.loader.exec_module(module)
    return module


def test_each_node_kicks_off_its_own_agent_with_remote_mcp(runner, monkeypatch):
    calls = []
    agents = []

    class OfflineAgent:
        def __init__(self, **config):
            self.config = config
            agents.append(self)

        async def kickoff_async(self, prompt):
            calls.append((prompt, get_execution_uuid()))
            return SimpleNamespace(raw=f"output-{len(calls)}")

    monkeypatch.setattr(runner, "FirecrawlAgent", OfflineAgent)
    monkeypatch.setenv("MODEL", "test-model")
    result = runner.WharfDemoFlow(tracing=False).kickoff(inputs={"topic": "test topic"})

    assert result == "output-3"
    assert len(agents) == 3
    assert "output-1" in calls[1][0] and "output-2" in calls[2][0]
    assert all("test topic" in prompt for prompt, _ in calls)
    assert calls[0][1] and len({execution_id for _, execution_id in calls}) == 1
    assert get_execution_uuid() is None
    for agent in agents:
        assert agent.config["llm"] == "test-model"
        (server,) = agent.config["mcps"]
        assert server.url == "https://mcp.firecrawl.dev/v2/mcp"
        assert server.streamable
        assert server.headers == {"Authorization": "Bearer synthetic-firecrawl-key"}
        assert server.tool_filter({"name": "firecrawl_search"})
        assert server.tool_filter({"name": "firecrawl_scrape"})
        assert not server.tool_filter({"name": "firecrawl_crawl"})


def test_runner_requires_firecrawl_key(runner, monkeypatch):
    monkeypatch.delenv("FIRECRAWL_API_KEY")
    with pytest.raises(ValueError, match="Set FIRECRAWL_API_KEY"):
        runner.firecrawl_mcp()


@pytest.mark.asyncio
async def test_firecrawl_agent_sends_only_basic_mcp_arguments(runner, monkeypatch):
    from crewai.llms.base_llm import BaseLLM
    from crewai.tools.tool_failure import (
        ToolExecutionFailedError,
        ToolFailure,
        handle_tool_failure,
    )
    from pydantic import BaseModel, ValidationError

    calls = []

    class OfflineLLM(BaseLLM):
        def call(self, messages, **kwargs):
            raise AssertionError("No LLM call expected")

    class Client:
        async def connect(self):
            pass

        async def disconnect(self):
            pass

        async def call_tool_result(self, name, args):
            calls.append((name, args))
            return SimpleNamespace(content="source text", is_error=False)

    tools = [
        runner.MCPNativeTool(
            client_factory=Client,
            tool_name=name,
            tool_schema={"args_schema": BaseModel},
            server_name="firecrawl",
        )
        for name in ("firecrawl_search", "firecrawl_scrape")
    ]
    monkeypatch.setattr(runner.Agent, "get_mcp_tools", lambda self, mcps: tools)
    agent = runner.FirecrawlAgent(
        role="tester", goal="test", backstory="test", llm=OfflineLLM(model="test")
    )
    search, scrape = agent.get_mcp_tools([runner.firecrawl_mcp()])
    args = search.args_schema(query="CrewAI Flow tracing").model_dump()
    assert await search._run_async(**args) == "source text"
    args = scrape.args_schema(url="https://docs.crewai.com").model_dump()
    assert await scrape._run_async(**args) == "source text"
    assert calls == [
        ("firecrawl_search", {"query": "CrewAI Flow tracing", "limit": 3}),
        (
            "firecrawl_scrape",
            {"url": "https://docs.crewai.com", "formats": ["markdown"]},
        ),
    ]
    with pytest.raises(ValidationError):
        search.args_schema(query="test", scrapeOptions={"queryOptions": {}})
    with pytest.raises(ValidationError):
        scrape.args_schema(url="https://docs.crewai.com", formats=["json"])
    with pytest.raises(ToolExecutionFailedError):
        handle_tool_failure(
            ToolFailure(message="HTTP 400"), tool_name=search.name, tool=search
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_one", [False, True])
async def test_concurrent_runs_keep_separate_trace_trees(runner, monkeypatch, fail_one):
    from crewai.events.event_bus import crewai_event_bus
    from crewai.events.types.llm_events import (
        LLMCallCompletedEvent,
        LLMCallStartedEvent,
        LLMCallType,
    )
    from crewai.llms.base_llm import BaseLLM
    from crewai.telemetry.tracing.context import get_trace_session
    from crewai.telemetry.tracing.grants import (
        GrantSpanExporter,
        TraceGrant,
        TraceGrantClient,
    )
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    recorders = {}
    entered = []
    ready = asyncio.Event()

    def create(client, execution_uuid):
        return TraceGrant(
            token="synthetic-grant",
            collector_url="https://collector.invalid/v1/traces",
            execution_uuid=execution_uuid,
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=15),
        )

    def exporter(grant):
        recorder = InMemorySpanExporter()
        recorders[grant.execution_uuid] = recorder
        return recorder

    class LocalLLM(BaseLLM):
        def call(self, messages, **kwargs):
            call_id = str(uuid4())
            crewai_event_bus.emit(
                self, LLMCallStartedEvent(call_id=call_id, messages=messages)
            )
            response = "Final Answer: Verified source https://docs.crewai.com"
            crewai_event_bus.emit(
                self,
                LLMCallCompletedEvent(
                    call_id=call_id, response=response, call_type=LLMCallType.LLM_CALL
                ),
            )
            return response

        async def acall(self, messages, **kwargs):
            return self.call(messages, **kwargs)

        def supports_function_calling(self):
            return False

        def supports_stop_words(self):
            return False

    class OfflineAgent(runner.FirecrawlAgent):
        async def kickoff_async(self, prompt):
            execution_uuid = get_execution_uuid()
            if execution_uuid not in entered:
                entered.append(execution_uuid)
                if len(entered) == 3:
                    ready.set()
            # All three real Flows must overlap before any finishes.
            await asyncio.wait_for(ready.wait(), timeout=10)
            if fail_one and execution_uuid == entered[0]:
                raise RuntimeError("deliberate test failure")
            return await super().kickoff_async(prompt)

    def offline_agent(**config):
        config.update(llm=LocalLLM(model="local-test"), verbose=False)
        return OfflineAgent(**config)

    monkeypatch.setenv("CREWAI_USER_PAT", "synthetic-pat")
    monkeypatch.setattr(TraceGrantClient, "create", create)
    monkeypatch.setattr(GrantSpanExporter, "_exporter", staticmethod(exporter))
    monkeypatch.setattr(runner.Agent, "get_mcp_tools", lambda self, mcps: [])
    monkeypatch.setattr(runner, "FirecrawlAgent", offline_agent)

    results = await runner.run_concurrent("same topic", runs=3)

    assert len(recorders) == len(entered) == len(results) == 3, results
    assert len({result.trace_id for result in results}) == 3
    assert sum(result.error is not None for result in results) == int(fail_one)
    for result in results:
        spans = recorders[result.execution_uuid].get_finished_spans()
        assert {span.attributes["crewai.execution_uuid"] for span in spans} == {
            result.execution_uuid
        }
        assert {format(span.context.trace_id, "032x") for span in spans} == {
            result.trace_id
        }
        assert [span.name for span in spans if span.parent is None] == ["execute flow"]
        span_ids = {span.context.span_id for span in spans}
        assert all(span.parent.span_id in span_ids for span in spans if span.parent)
        if not result.error:
            assert result.output == "Verified source https://docs.crewai.com"
            assert sum(span.name == "execute lite agent" for span in spans) == 3
            assert sum(span.name == "call llm" for span in spans) == 3
    assert get_execution_uuid() is None and get_trace_session() is None
