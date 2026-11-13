"""Hosted discovery uses the ordinary authorized CrewAI tool execution loop."""

from typing import Any
from unittest.mock import MagicMock

from crewai import Agent, Crew, Task
from crewai.crews.crew_output import CrewOutput
from crewai.llms.providers.openai.completion import OpenAICompletion
from crewai.tools import BaseTool
from openai.types.responses import Response
from pydantic import Field
import pytest


class LookupTool(BaseTool):
    name: str = "lookup"
    description: str = "Look up a value."
    calls: list[str] = Field(default_factory=list)

    def _run(self, query: str = "default") -> str:
        self.calls.append(query)
        if query == "fail":
            raise RuntimeError("lookup unavailable")
        return f"value:{query}"


def make_response(output: list[dict[str, Any]], response_id: str) -> Response:
    """Build SDK response objects without contacting OpenAI."""
    return Response.model_validate(
        {
            "id": response_id,
            "created_at": 0,
            "object": "response",
            "model": "gpt-5.5",
            "output": output,
            "status": "completed",
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
        }
    )


@pytest.mark.parametrize("hosted", [True, False], ids=["hosted", "eager"])
@pytest.mark.parametrize(
    ("name", "arguments", "expected_output", "expected_calls"),
    [
        ("lookup", '{"query":"hello"}', "value:hello", ["hello"]),
        ("unavailable", "{}", "Tool not found", []),
        ("lookup", '{"query":', "Failed to parse tool arguments as JSON", []),
        ("lookup", "null", "Tool arguments must be a JSON object", []),
        ("lookup", "[]", "Tool arguments must be a JSON object", []),
        ("lookup", '"hello"', "Tool arguments must be a JSON object", []),
        ("lookup", '{"query":"fail"}', "lookup unavailable", ["fail"]),
    ],
    ids=["success", "unknown", "malformed", "null", "array", "string", "failure"],
)
def test_discovered_function_uses_authorized_catalog(
    hosted: bool,
    name: str,
    arguments: str,
    expected_output: str,
    expected_calls: list[str],
) -> None:
    lookup = LookupTool()
    unused = LookupTool(name="unused")
    llm = OpenAICompletion(
        model="gpt-5.5", api="responses", api_key="test-key", tool_search=hosted
    )
    output: list[dict[str, Any]] = []
    if hosted:
        output.extend(
            [
                {
                    "id": "ts_1",
                    "type": "tool_search_call",
                    "execution": "server",
                    "call_id": None,
                    "status": "completed",
                    "arguments": {"paths": [name]},
                },
                {
                    "id": "tso_1",
                    "type": "tool_search_output",
                    "execution": "server",
                    "call_id": None,
                    "status": "completed",
                    "tools": [
                        {
                            "type": "function",
                            "name": name,
                            "description": "Discovered tool",
                            "parameters": {
                                "type": "object",
                                "properties": {"query": {"type": "string"}},
                            },
                            "strict": False,
                            "defer_loading": True,
                        }
                    ],
                },
            ]
        )
    output.append(
        {
            "id": "fc_1",
            "type": "function_call",
            "call_id": "call_lookup",
            "name": name,
            "arguments": arguments,
            "status": "completed",
        }
    )
    client = MagicMock()
    client.responses.create.side_effect = [
        make_response(output, "resp_1"),
        make_response(
            [
                {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": "Done", "annotations": []}
                    ],
                }
            ],
            "resp_2",
        ),
    ]
    llm._client = client
    agent = Agent(
        role="Tester",
        goal="Look up a value",
        backstory="Tests tool execution",
        llm=llm,
        tools=[lookup, unused],
        max_retry_limit=0,
        cache=False,
        verbose=False,
    )
    task = Task(description="Look up hello", expected_output="A value", agent=agent)

    result = Crew(agents=[agent], tasks=[task], cache=False, verbose=False).kickoff()

    assert isinstance(result, CrewOutput)
    assert result.raw == "Done"
    assert lookup.calls == expected_calls
    assert unused.calls == []
    assert client.responses.create.call_count == 2
    for request in client.responses.create.call_args_list:
        params = request.kwargs
        assert not params.get("stream", False)
        functions = [tool for tool in params["tools"] if tool["type"] == "function"]
        assert {tool["name"] for tool in functions} == {"lookup", "unused"}
        assert all(tool.get("defer_loading", False) is hosted for tool in functions)
        assert sum(tool["type"] == "tool_search" for tool in params["tools"]) == int(
            hosted
        )
    followup = client.responses.create.call_args.kwargs["input"]
    calls = [item for item in followup if item.get("type") == "function_call"]
    results = [item for item in followup if item.get("type") == "function_call_output"]
    assert len(calls) == len(results) == 1
    assert calls[0]["call_id"] == results[0]["call_id"] == "call_lookup"
    assert calls[0]["name"] == name
    assert calls[0]["arguments"] == arguments
    assert expected_output in results[0]["output"]
