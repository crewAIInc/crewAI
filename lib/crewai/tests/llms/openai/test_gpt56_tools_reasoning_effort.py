"""Regression tests for models rejecting `reasoning_effort` alongside tools.

On /v1/chat/completions, gpt-5.4 and newer return a hard 400 when function tools
and a non-"none" `reasoning_effort` are sent together:

    Function tools with reasoning_effort are not supported for gpt-5.5 in
    /v1/chat/completions. To use function tools, use /v1/responses or set
    reasoning_effort to 'none'.

Measured against the live endpoint (2026-07):

    model                tools + "high"    tools + "none"
    gpt-5.6[-sol/...]    rejected          accepted
    gpt-5.5              rejected          accepted
    gpt-5.4[-mini/nano]  rejected          accepted
    gpt-5.2              accepted          accepted
    o1 / o3 / o4-mini    accepted          rejected

Known families are dropped before the request as a fast path. Anything else is
caught from the 400 and retried without the parameter, so a newly-restricted
model doesn't need a release here. The Responses API accepts
`reasoning: {"effort": ...}` with tools and is untouched.
"""

import httpx
import pytest
from openai import BadRequestError

from crewai.llms.providers.openai import completion as completion_module
from crewai.llms.providers.openai.completion import OpenAICompletion


MESSAGES = [{"role": "user", "content": "hi"}]

# Pre-converted OpenAI tool schema — passed through by
# _convert_tools_for_interference, keeping these tests focused on param handling.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]


def build(model: str, **kwargs) -> OpenAICompletion:
    return OpenAICompletion(model=model, api_key="sk-test", **kwargs)


def make_bad_request(
    model: str = "gpt-5.5", param: str = "reasoning_effort", message: str | None = None
) -> BadRequestError:
    """Build the 400 OpenAI returns for the tools + reasoning_effort combination."""
    if message is None:
        message = (
            f"Function tools with reasoning_effort are not supported for {model} "
            "in /v1/chat/completions. To use function tools, use /v1/responses "
            "or set reasoning_effort to 'none'."
        )
    body = {"error": {"message": message, "type": "invalid_request_error", "param": param}}
    response = httpx.Response(
        status_code=400,
        json=body,
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
    )
    return BadRequestError(message, response=response, body=body)


@pytest.fixture(autouse=True)
def _clear_learned_conflicts():
    """Keep the process-wide learned set from leaking between tests."""
    completion_module._LEARNED_TOOLS_REASONING_EFFORT_CONFLICTS.clear()
    yield
    completion_module._LEARNED_TOOLS_REASONING_EFFORT_CONFLICTS.clear()


class TestKnownFamiliesFastPath:
    """Models we've measured are stripped before the request goes out."""

    @pytest.mark.parametrize(
        "model",
        [
            "gpt-5.6",
            "openai/gpt-5.6",
            "gpt-5.6-sol",
            "GPT-5.6-Terra",
            "gpt-5.6-luna",
            "gpt-5.5",
            "gpt-5.5-2026-04-23",
            "gpt-5.4",
            "gpt-5.4-mini",
            "gpt-5.4-nano",
        ],
    )
    def test_drops_reasoning_effort_for_incompatible_models(self, model: str):
        llm = build(model, additional_params={"reasoning_effort": "high"})

        params = llm._prepare_completion_params(MESSAGES, tools=TOOLS)

        assert "reasoning_effort" not in params
        # Tools must survive — they are the reason the call exists.
        assert params["tools"]
        assert params["tool_choice"] == "auto"

    def test_drops_reasoning_effort_supplied_via_additional_params(self):
        """additional_params bypasses the typed field, so it must be checked too."""
        llm = build("gpt-5.5", additional_params={"reasoning_effort": "medium"})

        params = llm._prepare_completion_params(MESSAGES, tools=TOOLS)

        assert "reasoning_effort" not in params

    def test_keeps_reasoning_effort_without_tools(self):
        """Without tools the combination is legal, so nothing should change."""
        llm = build("gpt-5.5", additional_params={"reasoning_effort": "high"})

        params = llm._prepare_completion_params(MESSAGES, tools=None)

        assert params["reasoning_effort"] == "high"

    def test_keeps_explicit_none_effort_with_tools(self):
        """OpenAI explicitly allows reasoning_effort='none' with tools here."""
        llm = build("gpt-5.6", additional_params={"reasoning_effort": "none"})

        params = llm._prepare_completion_params(MESSAGES, tools=TOOLS)

        assert params["reasoning_effort"] == "none"

    @pytest.mark.parametrize("model", ["o1", "o3-mini", "gpt-5.2", "gpt-5"])
    def test_unaffected_models_keep_reasoning_effort_with_tools(self, model: str):
        """Measured as accepting the combination — must not be stripped."""
        llm = build(model, additional_params={"reasoning_effort": "high"})

        params = llm._prepare_completion_params(MESSAGES, tools=TOOLS)

        assert params["reasoning_effort"] == "high"

    def test_fast_path_actually_has_something_to_drop(self):
        """Guard against the drop tests passing vacuously.

        The typed `reasoning_effort` field is gated behind `is_o1_model`, so it
        never reaches the payload for gpt-5.x. If a test set it that way there
        would be nothing to strip and the assertion would pass for the wrong
        reason — `additional_params` is the path that actually leaks.
        """
        leaked = build(
            "gpt-5.5", additional_params={"reasoning_effort": "high"}
        )._prepare_completion_params(MESSAGES, tools=None)
        assert leaked["reasoning_effort"] == "high"

        typed = build(
            "gpt-5.5", reasoning_effort="high"
        )._prepare_completion_params(MESSAGES, tools=None)
        assert "reasoning_effort" not in typed


class TestErrorClassification:
    """Only the specific tools + reasoning_effort 400 should trigger a retry."""

    def test_matches_the_reported_error(self):
        assert OpenAICompletion._is_tools_reasoning_effort_error(make_bad_request())

    def test_ignores_other_bad_requests(self):
        error = make_bad_request(
            param="max_tokens",
            message="Unsupported parameter: 'max_tokens' is not supported.",
        )
        assert not OpenAICompletion._is_tools_reasoning_effort_error(error)

    def test_ignores_unsupported_effort_value(self):
        """o1/o3 reject reasoning_effort='none'; that is a different failure."""
        error = make_bad_request(
            message="Unsupported value: 'reasoning_effort' does not support 'none'."
        )
        assert not OpenAICompletion._is_tools_reasoning_effort_error(error)

    def test_ignores_unrelated_exceptions(self):
        assert not OpenAICompletion._is_tools_reasoning_effort_error(
            RuntimeError("boom")
        )


class TestRuntimeRetry:
    """Unmeasured models are recovered from the 400 rather than failing the run."""

    def test_retries_without_reasoning_effort_and_succeeds(self, monkeypatch):
        # gpt-5.9 is deliberately not in the known-families tuple.
        llm = build("gpt-5.9", additional_params={"reasoning_effort": "high"})
        seen: list[dict] = []

        def fake_handle(params, **kwargs):
            seen.append(params)
            if "reasoning_effort" in params:
                raise make_bad_request(model="gpt-5.9")
            return "ok"

        monkeypatch.setattr(llm, "_handle_completion", fake_handle)

        result = llm._call_completions(MESSAGES, tools=TOOLS)

        assert result == "ok"
        assert len(seen) == 2, "expected one failed call and one retry"
        assert seen[0]["reasoning_effort"] == "high"
        assert "reasoning_effort" not in seen[1]
        # Tools have to survive the retry.
        assert seen[1]["tools"]

    def test_remembers_the_conflict_for_later_calls(self, monkeypatch):
        llm = build("gpt-5.9", additional_params={"reasoning_effort": "high"})

        def fake_handle(params, **kwargs):
            if "reasoning_effort" in params:
                raise make_bad_request(model="gpt-5.9")
            return "ok"

        monkeypatch.setattr(llm, "_handle_completion", fake_handle)
        llm._call_completions(MESSAGES, tools=TOOLS)

        # Second call must strip up front instead of paying the 400 again.
        params = llm._prepare_completion_params(MESSAGES, tools=TOOLS)
        assert "reasoning_effort" not in params
        assert not llm._supports_reasoning_effort_with_tools("gpt-5.9")

    def test_does_not_retry_unrelated_bad_request(self, monkeypatch):
        llm = build("gpt-5.9", additional_params={"reasoning_effort": "high"})
        calls: list[dict] = []

        def fake_handle(params, **kwargs):
            calls.append(params)
            raise make_bad_request(param="max_tokens", message="Unsupported parameter")

        monkeypatch.setattr(llm, "_handle_completion", fake_handle)

        with pytest.raises(BadRequestError):
            llm._call_completions(MESSAGES, tools=TOOLS)

        assert len(calls) == 1, "must not retry errors it doesn't understand"

    def test_does_not_retry_when_nothing_to_strip(self, monkeypatch):
        """No reasoning_effort to remove means the retry can't help."""
        llm = build("gpt-5.9")
        calls: list[dict] = []

        def fake_handle(params, **kwargs):
            calls.append(params)
            raise make_bad_request(model="gpt-5.9")

        monkeypatch.setattr(llm, "_handle_completion", fake_handle)

        with pytest.raises(BadRequestError):
            llm._call_completions(MESSAGES, tools=TOOLS)

        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_async_path_retries_too(self, monkeypatch):
        llm = build("gpt-5.9", additional_params={"reasoning_effort": "high"})
        seen: list[dict] = []

        async def fake_handle(params, **kwargs):
            seen.append(params)
            if "reasoning_effort" in params:
                raise make_bad_request(model="gpt-5.9")
            return "ok"

        monkeypatch.setattr(llm, "_ahandle_completion", fake_handle)

        result = await llm._acall_completions(MESSAGES, tools=TOOLS)

        assert result == "ok"
        assert len(seen) == 2
        assert "reasoning_effort" not in seen[1]


class TestResponsesApiUntouched:
    def test_responses_api_keeps_effort_with_tools(self):
        """The Responses API has no such restriction — it takes reasoning.effort."""
        llm = build("gpt-5.6", api="responses", reasoning_effort="high")

        params = llm._prepare_responses_params(MESSAGES, tools=TOOLS)

        assert params["reasoning"] == {"effort": "high"}
