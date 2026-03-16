"""
Integration tests for AgentCoreRuntime.

Tests the full HTTP round-trip through BedrockAgentCoreApp using httpx's ASGI
transport (no real server needed, no AWS credentials needed). Uses a mock Crew
that returns predictable CrewOutput.

Validates:
  - Non-streaming JSON response via POST /invocations
  - Streaming SSE response via POST /invocations
  - GET /ping health check
  - 400 error for missing prompt
  - Various payload key formats (prompt, message, input, inputs)

Run:
    uv run pytest tests/tools/test_agentcore_runtime_e2e.py -v
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import httpx
import pytest

from crewai_tools.aws.bedrock.runtime.base import AgentCoreRuntime


# --- Helpers ---


def make_mock_crew(raw_result="mock crew output", stream_enabled=False):
    """Build a mock Crew that returns predictable CrewOutput."""
    crew_output = MagicMock()
    crew_output.raw = raw_result
    crew_output.__str__ = lambda self: self.raw
    crew_output.json_dict = None
    crew_output.tasks_output = []
    crew_output.token_usage = None

    crew = MagicMock()
    crew.stream = stream_enabled
    crew.kickoff.return_value = crew_output

    # For streaming: copy() returns a new mock that also has kickoff
    crew_copy = MagicMock()
    crew_copy.kickoff.return_value = crew_output
    crew.copy.return_value = crew_copy

    return crew


def _parse_sse(text: str) -> list[dict]:
    """Parse SSE text into list of JSON event dicts."""
    events = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if line.startswith("data: "):
            data = line[6:]
            try:
                events.append(json.loads(data))
            except json.JSONDecodeError:
                pass
    return events


# --- Fixtures ---


@pytest.fixture
def non_streaming_client():
    crew = make_mock_crew(raw_result="non-streaming result")
    runtime = AgentCoreRuntime(crew=crew, stream=False)
    transport = httpx.ASGITransport(app=runtime.app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


@pytest.fixture
def streaming_crew_and_client():
    """Return (crew, client) for streaming tests that need crew access."""
    from crewai.types.streaming import StreamChunkType

    # Build streaming chunks
    text_chunk = MagicMock()
    text_chunk.content = "streamed content"
    text_chunk.chunk_type = StreamChunkType.TEXT
    text_chunk.agent_role = "Writer"
    text_chunk.task_name = "write"
    text_chunk.tool_call = None

    crew_output = MagicMock()
    crew_output.raw = "streaming done"
    crew_output.__str__ = lambda self: self.raw
    crew_output.json_dict = None
    crew_output.tasks_output = []
    crew_output.token_usage = None

    streaming_output = MagicMock()
    streaming_output.__iter__ = MagicMock(return_value=iter([text_chunk]))
    streaming_output.result = crew_output
    streaming_output.get_full_text.return_value = "streamed content"

    crew_copy = MagicMock()
    crew_copy.kickoff.return_value = streaming_output

    crew = MagicMock()
    crew.stream = False
    crew.copy.return_value = crew_copy

    runtime = AgentCoreRuntime(crew=crew, stream=True)
    transport = httpx.ASGITransport(app=runtime.app)
    client = httpx.AsyncClient(transport=transport, base_url="http://testserver")
    return crew, client


# --- Tests ---


class TestPing:
    @pytest.mark.asyncio
    async def test_ping(self, non_streaming_client):
        resp = await non_streaming_client.get("/ping")
        assert resp.status_code == 200


class TestNonStreamingE2E:
    @pytest.mark.asyncio
    async def test_prompt_key(self, non_streaming_client):
        resp = await non_streaming_client.post(
            "/invocations", json={"prompt": "hello"}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "response" in body
        assert body["response"] == "non-streaming result"

    @pytest.mark.asyncio
    async def test_message_key(self, non_streaming_client):
        resp = await non_streaming_client.post(
            "/invocations", json={"message": "hello"}
        )
        assert resp.status_code == 200
        assert resp.json()["response"] == "non-streaming result"

    @pytest.mark.asyncio
    async def test_input_key(self, non_streaming_client):
        resp = await non_streaming_client.post(
            "/invocations", json={"input": "hello"}
        )
        assert resp.status_code == 200
        assert resp.json()["response"] == "non-streaming result"

    @pytest.mark.asyncio
    async def test_inputs_dict_key(self, non_streaming_client):
        resp = await non_streaming_client.post(
            "/invocations", json={"inputs": {"topic": "AI"}}
        )
        assert resp.status_code == 200
        assert resp.json()["response"] == "non-streaming result"

    @pytest.mark.asyncio
    async def test_missing_prompt_returns_400(self, non_streaming_client):
        resp = await non_streaming_client.post(
            "/invocations", json={"foo": "bar"}
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_empty_payload_returns_400(self, non_streaming_client):
        resp = await non_streaming_client.post("/invocations", json={})
        assert resp.status_code == 400


class TestStreamingE2E:
    @pytest.mark.asyncio
    async def test_streaming_returns_sse(self, streaming_crew_and_client):
        _, client = streaming_crew_and_client
        resp = await client.post("/invocations", json={"prompt": "hello"})
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

        events = _parse_sse(resp.text)
        assert len(events) > 0

        event_types = [e.get("event") for e in events]
        assert "done" in event_types

    @pytest.mark.asyncio
    async def test_streaming_text_events(self, streaming_crew_and_client):
        _, client = streaming_crew_and_client
        resp = await client.post("/invocations", json={"prompt": "hello"})
        events = _parse_sse(resp.text)

        text_events = [e for e in events if e.get("event") == "text"]
        assert len(text_events) >= 1
        assert text_events[0]["content"] == "streamed content"
        assert text_events[0]["agent_role"] == "Writer"

    @pytest.mark.asyncio
    async def test_streaming_done_event(self, streaming_crew_and_client):
        _, client = streaming_crew_and_client
        resp = await client.post("/invocations", json={"prompt": "hello"})
        events = _parse_sse(resp.text)

        done_events = [e for e in events if e.get("event") == "done"]
        assert len(done_events) == 1
        assert done_events[0]["response"] == "streaming done"
