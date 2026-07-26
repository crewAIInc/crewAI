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


class TestLearnedFastPath:
    """Nothing is assumed up front; a 400 is what teaches us about a model."""

    def test_sends_reasoning_effort_for_an_unknown_model(self):
        """No model list, so the first request goes out as the caller asked.

        The 400 it may come back with is recovered in `_call_completions`.
        """
        llm = build("gpt-5.6", additional_params={"reasoning_effort": "high"})

        params = llm._prepare_completion_params(MESSAGES, tools=TOOLS)

        assert params["reasoning_effort"] == "high"

    def test_forces_none_once_the_model_is_known(self):
        """After a 400, skip the doomed request and go straight to "none".

        Explicit "none" rather than removing the key: gpt-5.6-* apply a
        server-side default, so omitting it is rejected the same way.
        """
        llm = build("gpt-5.6", additional_params={"reasoning_effort": "high"})
        llm._remember_reasoning_effort_conflict()

        params = llm._prepare_completion_params(MESSAGES, tools=TOOLS)

        assert "reasoning_effort" in params, "the key must be sent, not dropped"
        assert params["reasoning_effort"] == "none"
        # Tools must survive — they are the reason the call exists.
        assert params["tools"]
        assert params["tool_choice"] == "auto"

    def test_learning_applies_to_the_additional_params_leak(self):
        """additional_params bypasses the typed field, so it must be checked too."""
        llm = build("gpt-5.5", additional_params={"reasoning_effort": "medium"})
        llm._remember_reasoning_effort_conflict()

        params = llm._prepare_completion_params(MESSAGES, tools=TOOLS)

        assert params["reasoning_effort"] == "none"

    def test_keeps_reasoning_effort_without_tools(self):
        """Without tools the combination is legal, so nothing should change."""
        llm = build("gpt-5.5", additional_params={"reasoning_effort": "high"})
        llm._remember_reasoning_effort_conflict()

        params = llm._prepare_completion_params(MESSAGES, tools=None)

        assert params["reasoning_effort"] == "high"

    def test_learning_does_not_touch_other_models(self):
        """A conflict is remembered per model, not globally."""
        build("gpt-5.5")._remember_reasoning_effort_conflict()
        other = build("gpt-5.2", additional_params={"reasoning_effort": "high"})

        params = other._prepare_completion_params(MESSAGES, tools=TOOLS)

        assert params["reasoning_effort"] == "high"

    def test_fast_path_actually_has_something_to_drop(self):
        """Guard against the assertions above passing vacuously.

        The typed `reasoning_effort` field is gated behind `is_o1_model`, so it
        never reaches the payload for gpt-5.x. If a test set it that way there
        would be nothing to rewrite and the assertion would pass for the wrong
        reason — `additional_params` is the path that actually leaks.
        """
        leaked = build(
            "gpt-5.5", additional_params={"reasoning_effort": "high"}
        )._prepare_completion_params(MESSAGES, tools=None)
        assert leaked["reasoning_effort"] == "high"

        typed = build("gpt-5.5", reasoning_effort="high")._prepare_completion_params(
            MESSAGES, tools=None
        )
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
            if params.get("reasoning_effort") not in (None, "none"):
                raise make_bad_request(model="gpt-5.9")
            return "ok"

        monkeypatch.setattr(llm, "_handle_completion", fake_handle)

        result = llm._call_completions(MESSAGES, tools=TOOLS)

        assert result == "ok"
        assert len(seen) == 2, "expected one failed call and one retry"
        assert seen[0]["reasoning_effort"] == "high"
        assert seen[1]["reasoning_effort"] == "none"
        # Tools have to survive the retry.
        assert seen[1]["tools"]

    def test_remembers_the_conflict_for_later_calls(self, monkeypatch):
        llm = build("gpt-5.9", additional_params={"reasoning_effort": "high"})

        def fake_handle(params, **kwargs):
            if params.get("reasoning_effort") not in (None, "none"):
                raise make_bad_request(model="gpt-5.9")
            return "ok"

        monkeypatch.setattr(llm, "_handle_completion", fake_handle)
        llm._call_completions(MESSAGES, tools=TOOLS)

        # Second call must strip up front instead of paying the 400 again.
        params = llm._prepare_completion_params(MESSAGES, tools=TOOLS)
        assert params["reasoning_effort"] == "none"
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

    def test_learning_is_scoped_to_the_endpoint(self, monkeypatch):
        """A conflict learned against one endpoint must not leak to another.

        An OpenAI-compatible proxy may accept the combination that api.openai.com
        rejects, so stripping the parameter there would silently degrade it.
        """
        real = build("gpt-5.9", additional_params={"reasoning_effort": "high"})
        proxy = build(
            "gpt-5.9",
            base_url="https://proxy.internal/v1",
            additional_params={"reasoning_effort": "high"},
        )

        real._remember_reasoning_effort_conflict()

        assert (
            real._prepare_completion_params(MESSAGES, tools=TOOLS)["reasoning_effort"]
            == "none"
        )
        assert (
            proxy._prepare_completion_params(MESSAGES, tools=TOOLS)[
                "reasoning_effort"
            ]
            == "high"
        )

    def test_recoverable_error_is_not_reported_as_a_failure(self):
        """The retry must not surface a failure the user never experiences.

        `_handle_completion` logs "OpenAI API call failed" and emits
        LLMCallFailedEvent for anything it catches. For an error the caller is about
        to recover from, that produces a user-visible error panel for a call that
        ultimately succeeds.
        """
        llm = build("gpt-5.9")

        assert llm._is_recoverable_completion_error(
            ValueError("wrapped").with_traceback(None)
        ) is False

        recoverable = ValueError("wrapped")
        recoverable.__cause__ = make_bad_request()
        assert llm._is_recoverable_completion_error(recoverable)

    def test_does_not_retry_twice_when_already_none(self, monkeypatch):
        """Once reasoning_effort is "none" there is nothing left to try."""
        llm = build("gpt-5.9", additional_params={"reasoning_effort": "none"})
        calls: list[dict] = []

        def fake_handle(params, **kwargs):
            calls.append(params)
            raise make_bad_request(model="gpt-5.9")

        monkeypatch.setattr(llm, "_handle_completion", fake_handle)

        with pytest.raises(BadRequestError):
            llm._call_completions(MESSAGES, tools=TOOLS)

        assert len(calls) == 1, "must not retry when already at 'none'"

    @pytest.mark.asyncio
    async def test_async_path_retries_too(self, monkeypatch):
        llm = build("gpt-5.9", additional_params={"reasoning_effort": "high"})
        seen: list[dict] = []

        async def fake_handle(params, **kwargs):
            seen.append(params)
            if params.get("reasoning_effort") not in (None, "none"):
                raise make_bad_request(model="gpt-5.9")
            return "ok"

        monkeypatch.setattr(llm, "_ahandle_completion", fake_handle)

        result = await llm._acall_completions(MESSAGES, tools=TOOLS)

        assert result == "ok"
        assert len(seen) == 2
        assert seen[1]["reasoning_effort"] == "none"


class TestResponsesApiUntouched:
    def test_responses_api_keeps_effort_with_tools(self):
        """The Responses API has no such restriction — it takes reasoning.effort."""
        llm = build("gpt-5.6", api="responses", reasoning_effort="high")

        params = llm._prepare_responses_params(MESSAGES, tools=TOOLS)

        assert params["reasoning"] == {"effort": "high"}
