"""A tool-using agent's final answer is requested in the task's response model."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from crewai import Agent, Crew, Task
from crewai.events import crewai_event_bus
from crewai.hooks.dispatch import HookAborted
from crewai.llms.base_llm import BaseLLM
from crewai.tools import BaseTool
from crewai.utilities.rpm_controller import RPMController

QUOTE = "The export job nightly-sync failed with code E_LOCK_42."
PROSE = "## Findings\n\nThe nightly export failed because of a lock."


class Findings(BaseModel):
    quote: str
    ticket_id: str


EXACT = Findings(quote=QUOTE, ticket_id="SUP-1234")


class LogSearchTool(BaseTool):
    name: str = "search_logs"
    description: str = "Search the support logs."

    def _run(self, query: str) -> str:
        return f"SUP-1234: {QUOTE}"


class FinalTool(BaseTool):
    name: str = "search_logs"
    description: str = "Search the support logs."
    result_as_answer: bool = True

    def _run(self, query: str) -> str:
        return f"SUP-1234: {QUOTE}"


@dataclass
class Call:
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None
    response_model: type[BaseModel] | None
    stream: bool
    from_agent: bool


class ScriptedLLM(BaseLLM):
    """Calls a tool once, then answers; replies to schema requests separately."""

    def __init__(
        self,
        final_answer: str,
        structured_reply: BaseModel | str | Exception = EXACT,
        native: bool = True,
    ) -> None:
        super().__init__(model="scripted")
        self.final_answer = final_answer
        self.structured_reply = structured_reply
        self.native = native
        self.calls: list[Call] = []

    def supports_function_calling(self) -> bool:
        return self.native

    def call(
        self,
        messages: Any,
        tools: list[dict[str, Any]] | None = None,
        response_model: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> Any:
        self.calls.append(
            Call(
                list(messages),
                tools,
                response_model,
                bool(self.stream),
                kwargs.get("from_agent") is not None,
            )
        )
        if response_model is not None:
            if isinstance(self.structured_reply, Exception):
                raise self.structured_reply
            return self.structured_reply
        if sum(1 for call in self.calls if call.response_model is None) == 1:
            if self.native:
                return [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "search_logs",
                            "arguments": json.dumps({"query": "nightly"}),
                        },
                    }
                ]
            return (
                "Thought: I should search the logs.\n"
                "Action: search_logs\n"
                'Action Input: {"query": "nightly"}'
            )
        if self.native:
            return self.final_answer
        return f"Thought: I know the answer.\nFinal Answer: {self.final_answer}"

    async def acall(self, messages: Any, **kwargs: Any) -> Any:
        return self.call(messages, **kwargs)

    @property
    def schema_calls(self) -> list[Call]:
        """Schema requests from the agent's loop; the output converter's are not."""
        return [
            call
            for call in self.calls
            if call.response_model is not None and call.from_agent
        ]


@pytest.fixture(autouse=True)
def _flush_events():
    yield
    crewai_event_bus.flush()


def _agent(
    llm: ScriptedLLM, tools: list[BaseTool] | None = None, max_rpm: int | None = None
) -> Agent:
    return Agent(
        role="Support investigator",
        goal="Find the failing job",
        backstory="Reads support logs",
        llm=llm,
        tools=[LogSearchTool()] if tools is None else tools,
        max_retry_limit=0,
        max_rpm=max_rpm,
        verbose=False,
    )


def _run_crew(
    llm: ScriptedLLM,
    mode: str,
    tools: list[BaseTool] | None = None,
    max_rpm: int | None = None,
) -> Any:
    agent = _agent(llm, tools, max_rpm)
    task = Task(
        description="Why did the nightly export fail?",
        expected_output="The failing job, quoted exactly",
        output_pydantic=Findings,
        agent=agent,
    )
    crew = Crew(
        agents=[agent], tasks=[task], stream=mode.startswith("stream"), verbose=False
    )
    if mode == "sync":
        return crew.kickoff()
    if mode == "async":
        return asyncio.run(crew.kickoff_async())
    if mode == "stream_sync":
        streaming = crew.kickoff()
        list(streaming)
        return streaming.result

    async def _stream_async() -> Any:
        streaming = await crew.kickoff_async()
        async for _ in streaming:
            pass
        return streaming.result

    return asyncio.run(_stream_async())


MODES = ["sync", "async", "stream_sync", "stream_async"]


@pytest.mark.parametrize("native", [True, False], ids=["native", "react"])
@pytest.mark.parametrize("mode", MODES)
def test_prose_final_answer_is_requested_again_in_the_response_model(
    mode: str, native: bool
) -> None:
    llm = ScriptedLLM(final_answer=PROSE, native=native)

    result = _run_crew(llm, mode)

    assert result.pydantic == EXACT
    assert len(llm.schema_calls) == 1
    schema_call = llm.schema_calls[0]
    assert schema_call.response_model is Findings
    assert schema_call.tools is None
    assert schema_call.stream is mode.startswith("stream")
    # The schema request repeats the follow-up turn after the tool result.
    final_answer_call = llm.calls[-2]
    assert final_answer_call.response_model is None
    assert schema_call.messages == final_answer_call.messages
    assert QUOTE in json.dumps(schema_call.messages)
    assert PROSE not in json.dumps(schema_call.messages)


@pytest.mark.parametrize("native", [True, False], ids=["native", "react"])
@pytest.mark.parametrize("mode", MODES)
def test_schema_request_waits_for_the_agent_rpm_limit(mode: str, native: bool) -> None:
    llm = ScriptedLLM(final_answer=PROSE, native=native)
    waited_after_calls: list[int] = []

    with patch.object(
        RPMController,
        "_wait_for_next_minute",
        lambda self: waited_after_calls.append(len(llm.calls)),
    ):
        result = _run_crew(llm, mode, max_rpm=2)

    assert result.pydantic == EXACT
    assert len(llm.schema_calls) == 1
    # The tool call and the text answer use the minute's two requests, so the
    # schema request is the one that has to wait.
    assert waited_after_calls == [2]


@pytest.mark.parametrize("native", [True, False], ids=["native", "react"])
@pytest.mark.parametrize("mode", MODES)
def test_final_answer_that_fits_makes_no_extra_call(mode: str, native: bool) -> None:
    llm = ScriptedLLM(final_answer=f"```json\n{EXACT.model_dump_json()}\n```", native=native)

    result = _run_crew(llm, mode)

    assert result.pydantic == EXACT
    assert llm.schema_calls == []
    assert len(llm.calls) == 2


@pytest.mark.parametrize("native", [True, False], ids=["native", "react"])
def test_json_text_from_the_schema_request_is_used(native: bool) -> None:
    """The LiteLLM fallback returns the structured instance as JSON text."""
    llm = ScriptedLLM(
        final_answer=PROSE, structured_reply=EXACT.model_dump_json(), native=native
    )

    result = _run_crew(llm, "sync")

    assert result.pydantic == EXACT
    assert len(llm.schema_calls) == 1


@pytest.mark.parametrize(
    "structured_reply",
    [PROSE + " (again)", ValueError("Failed to parse response into Findings")],
    ids=["provider_ignores_schema", "provider_rejects_reply"],
)
def test_text_answer_is_kept_when_the_schema_request_gives_no_instance(
    structured_reply: str | Exception,
) -> None:
    llm = ScriptedLLM(final_answer=PROSE, structured_reply=structured_reply)

    result = _agent(llm).kickoff(
        "Why did the nightly export fail?", response_format=Findings
    )

    assert result.raw == PROSE
    assert llm.schema_calls[0].tools is None


class ProviderBadRequest(Exception):
    """Stands in for a provider SDK's HTTP 400, which is not a ValueError."""


class RejectsToolHistoryLLM(ScriptedLLM):
    """Rejects a schema request carrying tool history and no tools, as Anthropic does."""

    def call(
        self,
        messages: Any,
        tools: list[dict[str, Any]] | None = None,
        response_model: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> Any:
        reply = super().call(
            messages, tools=tools, response_model=response_model, **kwargs
        )
        if (
            response_model is not None
            and tools is None
            and any(message.get("role") == "tool" for message in messages)
        ):
            raise ProviderBadRequest(
                "Requests which include tool_use or tool_result blocks must define tools."
            )
        return reply


@pytest.mark.parametrize("mode", MODES)
def test_provider_error_on_the_schema_request_keeps_the_task_answer(
    mode: str,
) -> None:
    llm = RejectsToolHistoryLLM(final_answer=PROSE)

    result = _run_crew(llm, mode)

    assert len(llm.schema_calls) == 1
    # The output converter still fits the text answer, as it does without tools.
    assert result.raw == PROSE
    assert result.pydantic == EXACT


def test_provider_error_on_the_schema_request_keeps_the_kickoff_answer() -> None:
    llm = ScriptedLLM(
        final_answer=PROSE,
        structured_reply=ProviderBadRequest("json_schema response_format unsupported"),
    )

    result = _agent(llm).kickoff(
        "Why did the nightly export fail?", response_format=Findings
    )

    assert result.raw == PROSE
    assert len(llm.schema_calls) == 1


class AbortsSchemaRequestLLM(ScriptedLLM):
    """A hook denies the agent's schema request; the converter's would succeed."""

    def call(self, messages: Any, **kwargs: Any) -> Any:
        reply = super().call(messages, **kwargs)
        if kwargs.get("response_model") is not None and kwargs.get("from_agent"):
            raise HookAborted("policy denied", source="test-policy")
        return reply


def test_hook_abort_on_the_schema_request_is_not_swallowed() -> None:
    llm = AbortsSchemaRequestLLM(final_answer=PROSE)

    with pytest.raises(HookAborted, match="policy denied"):
        _agent(llm).kickoff("Why did the nightly export fail?", response_format=Findings)


def test_kickoff_response_format_gets_the_structured_answer() -> None:
    llm = ScriptedLLM(final_answer=PROSE)

    result = _agent(llm).kickoff(
        "Why did the nightly export fail?", response_format=Findings
    )

    assert result.pydantic == EXACT
    assert len(llm.schema_calls) == 1


def test_result_as_answer_tool_output_is_not_requested_again() -> None:
    llm = ScriptedLLM(final_answer=PROSE)

    result = _run_crew(llm, "sync", tools=[FinalTool()])

    assert result.raw == f"SUP-1234: {QUOTE}"
    assert llm.schema_calls == []


def test_agent_without_tools_still_gets_the_schema_on_its_only_call() -> None:
    llm = ScriptedLLM(final_answer=PROSE)

    result = _run_crew(llm, "sync", tools=[])

    assert result.pydantic == EXACT
    assert len(llm.calls) == 1
    assert llm.calls[0].response_model is Findings


class ChunkingLLM(ScriptedLLM):
    """Streams each text or structured reply as one chunk, as a provider would."""

    def call(self, messages: Any, **kwargs: Any) -> Any:
        reply = super().call(messages, **kwargs)
        if self.stream and kwargs.get("from_agent") is not None:
            text = reply.model_dump_json() if isinstance(reply, BaseModel) else reply
            if isinstance(text, str):
                self._emit_stream_chunk_event(
                    text,
                    from_task=kwargs.get("from_task"),
                    from_agent=kwargs.get("from_agent"),
                )
        return reply


def test_stream_carries_the_text_answer_then_the_structured_one() -> None:
    llm = ChunkingLLM(final_answer=PROSE)
    agent = _agent(llm)
    task = Task(
        description="Why did the nightly export fail?",
        expected_output="The failing job, quoted exactly",
        output_pydantic=Findings,
        agent=agent,
    )
    streaming = Crew(agents=[agent], tasks=[task], stream=True, verbose=False).kickoff()

    chunks = [chunk.content for chunk in streaming]

    assert chunks == [PROSE, EXACT.model_dump_json()]
    assert streaming.result.pydantic == EXACT
