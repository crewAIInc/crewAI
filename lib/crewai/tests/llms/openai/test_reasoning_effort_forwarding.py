"""`reasoning_effort` reaches every reasoning model, not just o1.

The completions path used to gate the parameter behind
``is_o1_model = "o1" in model.lower()``, a literal substring test. gpt-5, o3 and
o4-mini contain no "o1", so an explicitly configured effort was dropped and the
model thought at the server default -- silently, since the request still
succeeded.

The gate could not simply be widened: ``is_o1_model`` also drives
``supports_function_calling``, ``supports_stop_words`` and the system->user
message rewrite, so marking gpt-5 as an o1 model would report that it cannot
call tools. The parameter is forwarded unconditionally instead, and a model that
genuinely does not support it says so in a 400 that is retried without the key.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from openai import BadRequestError

from crewai.llm import LLM
from crewai.llms.providers.openai.completion import OpenAICompletion


MESSAGES = [{"role": "user", "content": "hi"}]

REASONING_MODELS = ["gpt-5", "gpt-5-mini", "o3", "o3-mini", "o4-mini", "o1"]


def build(model: str = "gpt-5", **kwargs: Any) -> OpenAICompletion:
    return OpenAICompletion(model=model, api_key="sk-test", **kwargs)


def _bad_request(message: str, **source: Any) -> BadRequestError:
    body = {"error": {"message": message, "type": "invalid_request_error", **source}}
    return BadRequestError(
        message,
        response=httpx.Response(
            400,
            json=body,
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        ),
        body=body,
    )


def unsupported_parameter_error() -> BadRequestError:
    """What a non-reasoning model returns for the parameter itself."""
    return _bad_request(
        "Unsupported parameter: 'reasoning_effort' is not supported with this model.",
        param="reasoning_effort",
        code="unsupported_parameter",
    )


def unrecognized_argument_error() -> BadRequestError:
    """The other shape, where `param` is null."""
    return _bad_request(
        "Unrecognized request argument supplied: reasoning_effort", param=None
    )


def unsupported_value_error() -> BadRequestError:
    """A bad *value* -- the model does support the parameter."""
    return _bad_request(
        "Unsupported value: 'reasoning_effort' does not support 'none'.",
        param="reasoning_effort",
    )


class TestParameterReachesTheModel:
    @pytest.mark.parametrize("model", REASONING_MODELS)
    def test_forwarded_for_every_reasoning_model(self, model):
        params = build(model, reasoning_effort="high")._prepare_completion_params(
            MESSAGES
        )

        assert params["reasoning_effort"] == "high"

    def test_minimal_is_forwarded(self):
        """gpt-5's cheapest setting, and the one extraction workloads want."""
        params = build("gpt-5", reasoning_effort="minimal")._prepare_completion_params(
            MESSAGES
        )

        assert params["reasoning_effort"] == "minimal"

    @pytest.mark.parametrize("model", ["gpt-4o", "gpt-5", "o3"])
    def test_absent_when_not_configured(self, model):
        """Unset must stay off the wire, whatever the model."""
        assert "reasoning_effort" not in build(model)._prepare_completion_params(
            MESSAGES
        )


class TestO1FlagUntouched:
    """The gate was shared; widening it would have broken these."""

    @pytest.mark.parametrize("model", ["gpt-5", "o3", "o4-mini"])
    def test_reasoning_models_still_support_tools(self, model):
        assert build(model).supports_function_calling() is True

    @pytest.mark.parametrize("model", ["gpt-5", "o3"])
    def test_is_o1_model_still_only_matches_o1(self, model):
        assert build(model).is_o1_model is False
        assert build("o1").is_o1_model is True

    def test_system_messages_are_not_rewritten_for_gpt5(self):
        formatted = build("gpt-5")._format_messages(
            [{"role": "system", "content": "be terse"}]
        )

        assert formatted[0]["role"] == "system"


class TestErrorDetection:
    def test_matches_unsupported_parameter(self):
        assert OpenAICompletion._rejects_reasoning_effort_as_unsupported(
            unsupported_parameter_error()
        )

    def test_matches_unrecognized_argument(self):
        assert OpenAICompletion._rejects_reasoning_effort_as_unsupported(
            unrecognized_argument_error()
        )

    def test_ignores_an_unsupported_value(self):
        """Dropping the key here would silently restore the original bug."""
        assert not OpenAICompletion._rejects_reasoning_effort_as_unsupported(
            unsupported_value_error()
        )

    def test_ignores_a_400_about_another_parameter(self):
        assert not OpenAICompletion._rejects_reasoning_effort_as_unsupported(
            _bad_request(
                "Unsupported parameter: 'temperature' is not supported.",
                param="temperature",
                code="unsupported_parameter",
            )
        )

    def test_ignores_unrelated_exceptions(self):
        assert not OpenAICompletion._rejects_reasoning_effort_as_unsupported(
            RuntimeError("boom")
        )


class TestRetryParams:
    def test_removes_the_key(self):
        params = OpenAICompletion._without_reasoning_effort(
            {"model": "gpt-4o", "reasoning_effort": "high"}
        )

        assert params == {"model": "gpt-4o"}

    def test_returns_none_when_absent(self):
        """Nothing left to drop -- the retry must not loop."""
        assert OpenAICompletion._without_reasoning_effort({"model": "gpt-4o"}) is None


class TestRetryBehaviour:
    def test_retries_without_the_key_and_succeeds(self, monkeypatch):
        llm = build("gpt-4o", reasoning_effort="high")
        seen: list[dict] = []

        def fake_handle(params, **kwargs):
            seen.append(params)
            if "reasoning_effort" in params:
                raise unsupported_parameter_error()
            return "ok"

        monkeypatch.setattr(llm, "_handle_completion", fake_handle)

        assert llm._call_completions(MESSAGES) == "ok"
        assert len(seen) == 2, "expected one rejected call and one retry"
        assert seen[0]["reasoning_effort"] == "high"
        assert "reasoning_effort" not in seen[1]

    @pytest.mark.asyncio
    async def test_retries_on_the_async_path(self, monkeypatch):
        llm = build("gpt-4o", reasoning_effort="high")
        seen: list[dict] = []

        async def fake_handle(params, **kwargs):
            seen.append(params)
            if "reasoning_effort" in params:
                raise unsupported_parameter_error()
            return "ok"

        monkeypatch.setattr(llm, "_ahandle_completion", fake_handle)

        assert await llm._acall_completions(MESSAGES) == "ok"
        assert len(seen) == 2
        assert "reasoning_effort" not in seen[1]

    def test_retries_on_the_streaming_path(self, monkeypatch):
        llm = build("gpt-4o", reasoning_effort="high", stream=True)
        seen: list[dict] = []

        def fake_handle(params, **kwargs):
            seen.append(params)
            if "reasoning_effort" in params:
                raise unsupported_parameter_error()
            return "ok"

        monkeypatch.setattr(llm, "_handle_streaming_completion", fake_handle)

        assert llm._call_completions(MESSAGES) == "ok"
        assert len(seen) == 2
        assert "reasoning_effort" not in seen[1]

    @pytest.mark.asyncio
    async def test_retries_on_the_async_streaming_path(self, monkeypatch):
        llm = build("gpt-4o", reasoning_effort="high", stream=True)
        seen: list[dict] = []

        async def fake_handle(params, **kwargs):
            seen.append(params)
            if "reasoning_effort" in params:
                raise unsupported_parameter_error()
            return "ok"

        monkeypatch.setattr(llm, "_ahandle_streaming_completion", fake_handle)

        assert await llm._acall_completions(MESSAGES) == "ok"
        assert len(seen) == 2
        assert "reasoning_effort" not in seen[1]

    def test_does_not_retry_forever(self, monkeypatch):
        llm = build("gpt-4o", reasoning_effort="high")
        calls: list[dict] = []

        def always_fail(params, **kwargs):
            calls.append(params)
            raise unsupported_parameter_error()

        monkeypatch.setattr(llm, "_handle_completion", always_fail)

        with pytest.raises(BadRequestError, match="not supported with this model"):
            llm._call_completions(MESSAGES)

        assert len(calls) == 2, "one original call plus exactly one retry"

    def test_an_unsupported_value_surfaces(self, monkeypatch):
        """A bad value is the caller's mistake and must not be papered over."""
        llm = build("o3", reasoning_effort="none")
        calls: list[dict] = []

        def always_fail(params, **kwargs):
            calls.append(params)
            raise unsupported_value_error()

        monkeypatch.setattr(llm, "_handle_completion", always_fail)

        with pytest.raises(BadRequestError, match="does not support 'none'"):
            llm._call_completions(MESSAGES)

        assert len(calls) == 1, "a bad value must not be retried"


class TestLLMSurface:
    @pytest.mark.parametrize("effort", ["none", "minimal", "low", "medium", "high"])
    def test_llm_accepts_every_documented_effort(self, effort):
        llm = LLM(model="gpt-5", reasoning_effort=effort, is_litellm=True)

        assert llm.reasoning_effort == effort

    def test_reaches_the_wire_through_the_llm_factory(self):
        """End to end: LLM(...) -> native provider -> request params."""
        llm = LLM(model="gpt-5", reasoning_effort="minimal", is_litellm=False)
        params = llm._prepare_completion_params(MESSAGES)

        assert params["reasoning_effort"] == "minimal"
