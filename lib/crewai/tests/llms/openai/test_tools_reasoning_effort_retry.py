"""GPT-5.6 rejects function tools alongside `reasoning_effort` on completions.

The family applies a server-side `reasoning_effort` default, so a payload that
never mentions the parameter is still refused as soon as tools are attached --
measured against the live endpoint:

    gpt-5.6-sol  tools, no reasoning_effort key   -> 400
    gpt-5.6-sol  tools, reasoning_effort="none"   -> OK
    gpt-5.5      tools, no reasoning_effort key   -> OK

So an ordinary agent with a tool fails on this model family and nothing else. The
400 is recovered by resending with an explicit "none"; no model list is involved,
so a family OpenAI restricts later is handled without a release here.
"""

import json
from types import SimpleNamespace

import httpx
import pytest
from openai import BadRequestError

from crewai import Agent, Crew, Task
from crewai.llms.providers.openai.completion import OpenAICompletion
from crewai.tools import tool


MESSAGES = [{"role": "user", "content": "hi"}]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "Multiply two integers",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            },
        },
    }
]


def build(model: str = "gpt-5.6-sol", **kwargs) -> OpenAICompletion:
    return OpenAICompletion(model=model, api_key="sk-test", **kwargs)


def _bad_request(message: str, param: str = "reasoning_effort") -> BadRequestError:
    body = {
        "error": {
            "message": message,
            "type": "invalid_request_error",
            "param": param,
        }
    }
    return BadRequestError(
        message,
        response=httpx.Response(
            400,
            json=body,
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        ),
        body=body,
    )


def tools_effort_error(model: str = "gpt-5.6-sol") -> BadRequestError:
    """The 400 OpenAI returns for tools + reasoning_effort."""
    return _bad_request(
        f"Function tools with reasoning_effort are not supported for {model} in "
        "/v1/chat/completions. To use function tools, use /v1/responses or set "
        "reasoning_effort to 'none'."
    )


def unsupported_value_error() -> BadRequestError:
    """What o1/o3 return for reasoning_effort="none" -- not recoverable this way."""
    return _bad_request("Unsupported value: 'reasoning_effort' does not support 'none'.")


class TestErrorDetection:
    def test_matches_the_tools_effort_400(self):
        assert OpenAICompletion._rejects_reasoning_effort_with_tools(
            tools_effort_error()
        )

    def test_matches_a_flat_error_body(self):
        """The SDK populates `body` flat or nested depending on how it parsed."""
        message = (
            "Function tools with reasoning_effort are not supported for gpt-5.6-sol."
        )
        body = {"message": message, "param": "reasoning_effort"}
        error = BadRequestError(
            message,
            response=httpx.Response(
                400, json=body, request=httpx.Request("POST", "https://x/v1/c")
            ),
            body=body,
        )

        assert OpenAICompletion._rejects_reasoning_effort_with_tools(error)

    def test_ignores_the_unsupported_value_400(self):
        """o1/o3 reject reasoning_effort="none", so retrying with it can't help."""
        assert not OpenAICompletion._rejects_reasoning_effort_with_tools(
            unsupported_value_error()
        )

    def test_ignores_unrelated_exceptions(self):
        assert not OpenAICompletion._rejects_reasoning_effort_with_tools(
            RuntimeError("boom")
        )


class TestRetryParams:
    def test_sets_an_explicit_none(self):
        """Absence means "use the server default", which is what was rejected."""
        params = build()._reasoning_effort_none_params({"model": "gpt-5.6-sol"})

        assert params is not None
        assert params["reasoning_effort"] == "none"

    def test_overrides_a_requested_effort(self):
        params = build()._reasoning_effort_none_params({"reasoning_effort": "high"})

        assert params is not None
        assert params["reasoning_effort"] == "none"

    def test_returns_none_when_already_none(self):
        """Nothing left to try -- the retry must not loop."""
        assert (
            build()._reasoning_effort_none_params({"reasoning_effort": "none"}) is None
        )


class TestRetryBehaviour:
    def test_retries_with_none_and_succeeds(self, monkeypatch):
        llm = build()
        seen: list[dict] = []

        def fake_handle(params, **kwargs):
            seen.append(params)
            if params.get("reasoning_effort") != "none":
                raise tools_effort_error()
            return "ok"

        monkeypatch.setattr(llm, "_handle_completion", fake_handle)

        assert llm._call_completions(MESSAGES, tools=TOOLS) == "ok"
        assert len(seen) == 2, "expected one rejected call and one retry"
        assert "reasoning_effort" not in seen[0]
        assert seen[1]["reasoning_effort"] == "none"
        assert seen[1]["tools"], "tools must survive the retry"

    def test_does_not_retry_forever(self, monkeypatch):
        """A model that fails even with "none" must surface, not loop."""
        llm = build()
        calls: list[dict] = []

        def always_fail(params, **kwargs):
            calls.append(params)
            raise tools_effort_error()

        monkeypatch.setattr(llm, "_handle_completion", always_fail)

        with pytest.raises(BadRequestError):
            llm._call_completions(MESSAGES, tools=TOOLS)

        assert len(calls) == 2, "one original call plus exactly one retry"

    def test_does_not_retry_an_unrelated_400(self, monkeypatch):
        llm = build("o3-mini")
        calls: list[dict] = []

        def fail(params, **kwargs):
            calls.append(params)
            raise unsupported_value_error()

        monkeypatch.setattr(llm, "_handle_completion", fail)

        with pytest.raises(BadRequestError):
            llm._call_completions(MESSAGES, tools=TOOLS)

        assert len(calls) == 1

    def test_recovered_call_reports_no_failure(self, monkeypatch):
        """The retried attempt must not emit LLMCallFailedEvent."""
        llm = build()
        emitted: list[str] = []
        monkeypatch.setattr(
            llm,
            "_emit_call_failed_event",
            lambda **kwargs: emitted.append(kwargs.get("error", "")),
        )

        class FakeCompletions:
            def create(self, **kwargs):
                raise tools_effort_error()

        monkeypatch.setattr(
            llm,
            "_get_sync_client",
            lambda: SimpleNamespace(
                chat=SimpleNamespace(completions=FakeCompletions())
            ),
        )

        # The original BadRequestError must survive for _call_completions to match.
        with pytest.raises(BadRequestError):
            llm._handle_completion({"model": "gpt-5.6-sol", "messages": MESSAGES})

        assert emitted == []

    @pytest.mark.asyncio
    async def test_async_path_retries_too(self, monkeypatch):
        llm = build()
        seen: list[dict] = []

        async def fake_handle(params, **kwargs):
            seen.append(params)
            if params.get("reasoning_effort") != "none":
                raise tools_effort_error()
            return "ok"

        monkeypatch.setattr(llm, "_ahandle_completion", fake_handle)

        assert await llm._acall_completions(MESSAGES, tools=TOOLS) == "ok"
        assert seen[1]["reasoning_effort"] == "none"


class TestAgentDefinitions:
    """The bar: ordinary agent definitions run, with tools and with reasoning."""

    @staticmethod
    def _stub_transport(llm: OpenAICompletion, monkeypatch) -> list[dict]:
        """Reject like GPT-5.6 until an explicit reasoning_effort="none" arrives.

        The reasoning planner calls the LLM with its own `available_functions`, so
        the stub answers that request with the planner's JSON contract and every
        other request with the final answer.
        """
        sent: list[dict] = []

        def fake_handle(params, available_functions=None, **kwargs):
            sent.append(params)
            if params.get("tools") and params.get("reasoning_effort") != "none":
                raise tools_effort_error()
            if available_functions and "create_reasoning_plan" in available_functions:
                return json.dumps(
                    {"plan": "Multiply 17 by 23.", "steps": [], "ready": True}
                )
            return "391"

        monkeypatch.setattr(llm, "_handle_completion", fake_handle)
        return sent

    @staticmethod
    def _multiply_tool():
        @tool("multiply")
        def multiply(a: int, b: int) -> int:
            """Multiply two integers."""
            return a * b

        return multiply

    def test_agent_with_tools(self, monkeypatch):
        llm = build()
        sent = self._stub_transport(llm, monkeypatch)

        agent = Agent(
            role="Math",
            goal="Answer correctly",
            backstory="You are precise.",
            llm=llm,
            tools=[self._multiply_tool()],
        )
        task = Task(
            description="What is 17 times 23?",
            expected_output="A number",
            agent=agent,
        )

        assert "391" in str(Crew(agents=[agent], tasks=[task]).kickoff())
        assert any(p.get("reasoning_effort") == "none" for p in sent)

    def test_agent_with_tools_and_reasoning(self, monkeypatch):
        """`reasoning=True` adds a planning call, which must recover as well."""
        llm = build()
        sent = self._stub_transport(llm, monkeypatch)

        agent = Agent(
            role="Math",
            goal="Answer correctly",
            backstory="You are precise.",
            llm=llm,
            tools=[self._multiply_tool()],
            reasoning=True,
        )
        task = Task(
            description="What is 17 times 23?",
            expected_output="A number",
            agent=agent,
        )

        assert "391" in str(Crew(agents=[agent], tasks=[task]).kickoff())
        assert any(p.get("reasoning_effort") == "none" for p in sent)

    def test_agent_without_tools_is_untouched(self, monkeypatch):
        """No tools means no conflict, so nothing should be rewritten."""
        llm = build()
        sent = self._stub_transport(llm, monkeypatch)

        agent = Agent(
            role="Math",
            goal="Answer correctly",
            backstory="You are precise.",
            llm=llm,
        )
        task = Task(
            description="What is 17 times 23?",
            expected_output="A number",
            agent=agent,
        )

        assert "391" in str(Crew(agents=[agent], tasks=[task]).kickoff())
        assert all("reasoning_effort" not in p for p in sent)
