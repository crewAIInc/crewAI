"""Tests for models that /v1/chat/completions doesn't serve at all.

The pro tier exists but is Responses-API-only. OpenAI reports it as a 404 that is
distinguishable from a genuine unknown model:

    responses-only   param="model", "only supported in v1/responses"
                     or "This is not a chat model"
    genuine typo     code="model_not_found", "does not exist"

Measured 2026-07 against the live endpoints: gpt-5-pro, gpt-5.2-pro, gpt-5.4-pro,
gpt-5.5-pro, o1-pro and o3-pro all 404 on chat completions and work on
/v1/responses. Rather than hardcoding that list, the 404 is caught and the call is
retried on the Responses API, then the model is remembered so the wasted round trip
is paid once per process.
"""

import httpx
import pytest
from openai import BadRequestError, NotFoundError

from crewai.llms.providers.openai import completion as completion_module
from crewai.llms.providers.openai.completion import OpenAICompletion


MESSAGES = [{"role": "user", "content": "hi"}]

RESPONSES_ONLY_MESSAGES = (
    "This model is only supported in v1/responses and not in /v1/chat/completions.",
    "This is not a chat model and thus not supported in the v1/chat/completions "
    "endpoint. Did you mean to use v1/completions?",
)


def build(model: str, **kwargs) -> OpenAICompletion:
    return OpenAICompletion(model=model, api_key="sk-test", **kwargs)


def make_not_found(message: str, code: str | None = None) -> NotFoundError:
    body = {
        "error": {
            "message": message,
            "type": "invalid_request_error",
            "param": None if code else "model",
            "code": code,
        }
    }
    response = httpx.Response(
        status_code=404,
        json=body,
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
    )
    return NotFoundError(message, response=response, body=body)


@pytest.fixture(autouse=True)
def _clear_learned_models():
    """Keep the process-wide learned set from leaking between tests."""
    completion_module._LEARNED_RESPONSES_ONLY_MODELS.clear()
    yield
    completion_module._LEARNED_RESPONSES_ONLY_MODELS.clear()


class TestErrorClassification:
    @pytest.mark.parametrize("message", RESPONSES_ONLY_MESSAGES)
    def test_detects_responses_only_404(self, message: str):
        assert OpenAICompletion._is_responses_only_error(make_not_found(message))

    def test_ignores_genuine_unknown_model(self):
        """A real typo must keep failing, not get retried on another endpoint."""
        error = make_not_found(
            "The model `gpt-5.99-fake` does not exist or you do not have access "
            "to it.",
            code="model_not_found",
        )
        assert not OpenAICompletion._is_responses_only_error(error)

    def test_ignores_unrelated_exceptions(self):
        assert not OpenAICompletion._is_responses_only_error(RuntimeError("boom"))


class TestFallback:
    def test_retries_on_responses_and_remembers_the_model(self, monkeypatch):
        llm = build("gpt-5-pro")
        calls: list[str] = []

        def fail_completion(**kwargs):
            calls.append("completions")
            raise ValueError("wrapped") from make_not_found(
                RESPONSES_ONLY_MESSAGES[0]
            )

        monkeypatch.setattr(llm, "_handle_completion", fail_completion)
        monkeypatch.setattr(
            llm, "_call_responses", lambda **kwargs: calls.append("responses") or "ok"
        )

        assert llm._call_completions(MESSAGES) == "ok"
        assert calls == ["completions", "responses"]
        # Second call must skip the doomed chat-completions attempt.
        assert llm._effective_api() == "responses"

    def test_does_not_retry_genuine_unknown_model(self, monkeypatch):
        llm = build("gpt-5.99-fake")
        calls: list[str] = []

        def fail_completion(**kwargs):
            calls.append("completions")
            raise ValueError("wrapped") from make_not_found(
                "The model does not exist.", code="model_not_found"
            )

        monkeypatch.setattr(llm, "_handle_completion", fail_completion)
        monkeypatch.setattr(
            llm, "_call_responses", lambda **kwargs: calls.append("responses") or "ok"
        )

        with pytest.raises(ValueError):
            llm._call_completions(MESSAGES)

        assert calls == ["completions"]
        assert not completion_module._LEARNED_RESPONSES_ONLY_MODELS

    def test_custom_endpoint_never_falls_back(self, monkeypatch):
        """An OpenAI-compatible server may not implement /v1/responses at all.

        Most (vLLM, LiteLLM proxies, Ollama) don't, so a self-hosted model must
        keep the endpoint the user chose rather than being silently rerouted.
        """
        llm = build(
            "o1-pro", custom_openai=True, base_url="https://my-vllm.internal/v1"
        )
        calls: list[str] = []

        def fail_completion(**kwargs):
            calls.append("completions")
            raise ValueError("wrapped") from make_not_found(
                RESPONSES_ONLY_MESSAGES[0]
            )

        monkeypatch.setattr(llm, "_handle_completion", fail_completion)
        monkeypatch.setattr(
            llm, "_call_responses", lambda **kwargs: calls.append("responses") or "ok"
        )

        with pytest.raises(ValueError):
            llm._call_completions(MESSAGES)

        assert calls == ["completions"]

    @pytest.mark.asyncio
    async def test_async_path_falls_back_too(self, monkeypatch):
        llm = build("gpt-5-pro")
        calls: list[str] = []

        async def fail_completion(**kwargs):
            calls.append("completions")
            raise ValueError("wrapped") from make_not_found(
                RESPONSES_ONLY_MESSAGES[0]
            )

        async def ok_responses(**kwargs):
            calls.append("responses")
            return "ok"

        monkeypatch.setattr(llm, "_ahandle_completion", fail_completion)
        monkeypatch.setattr(llm, "_acall_responses", ok_responses)

        assert await llm._acall_completions(MESSAGES) == "ok"
        assert calls == ["completions", "responses"]


class TestEffectiveApi:
    def test_defaults_to_completions_for_unknown_models(self):
        """Nothing is assumed up front; the 404 is what teaches us."""
        assert build("gpt-5-pro")._effective_api() == "completions"

    def test_explicit_responses_is_honoured(self):
        assert build("gpt-5.5", api="responses")._effective_api() == "responses"

    def test_learned_model_routes_directly(self):
        llm = build("gpt-5-pro")
        llm._remember_responses_only_model()

        assert llm._effective_api() == "responses"

    def test_learned_model_still_respects_custom_endpoint(self):
        llm = build(
            "gpt-5-pro", custom_openai=True, base_url="https://my-vllm.internal/v1"
        )
        completion_module._LEARNED_RESPONSES_ONLY_MODELS.add("gpt-5-pro")

        assert llm._effective_api() == "completions"


class TestNotFoundMessage:
    @pytest.mark.parametrize("message", RESPONSES_ONLY_MESSAGES)
    def test_points_at_responses_api(self, message: str):
        msg = build("gpt-5.5")._model_not_found_message(make_not_found(message))

        assert 'api="responses"' in msg
        assert "not available on /v1/chat/completions" in msg

    def test_keeps_plain_not_found_for_real_typos(self):
        msg = build("gpt-5.5")._model_not_found_message(
            make_not_found("The model does not exist.", code="model_not_found")
        )

        assert "not found" in msg
        assert 'api="responses"' not in msg


# The tools + reasoning_effort 400 from Bedrock's OpenAI-compatible endpoints
# (bedrock-runtime and bedrock-mantle /openai/v1). GPT-6 Sol offers "none" as a
# way out, like GPT-5.6 on OpenAI. GPT-6 Astra doesn't, and rejects "none"
# itself, so its function tools are only served on /v1/responses.
TOOLS_NONE_WAY_OUT = (
    "Function tools with reasoning_effort are not supported for us.openai.gpt-6-sol "
    "in /v1/chat/completions. To use function tools, use /v1/responses or set "
    "reasoning_effort to 'none'."
)
TOOLS_RESPONSES_ONLY = (
    "Function tools with reasoning_effort are not supported for "
    "us.openai.gpt-6-astra in /v1/chat/completions. To use function tools, use "
    "/v1/responses."
)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]


def make_tools_error(message: str) -> BadRequestError:
    body = {
        "error": {
            "message": message,
            "type": "invalid_request_error",
            "param": "reasoning_effort",
            "code": "validation_error",
        }
    }
    response = httpx.Response(
        status_code=400,
        json=body,
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
    )
    return BadRequestError(message, response=response, body=body)


class TestToolsOnlyOnResponses:
    def test_detects_tools_only_on_responses(self):
        assert OpenAICompletion._is_responses_only_error(
            make_tools_error(TOOLS_RESPONSES_ONLY)
        )

    def test_keeps_the_none_retry_when_offered(self):
        """GPT-5.6 / GPT-6 Sol still recover with reasoning_effort="none"."""
        assert not OpenAICompletion._is_responses_only_error(
            make_tools_error(TOOLS_NONE_WAY_OUT)
        )

    def test_keeps_the_old_path_when_responses_is_not_named(self):
        """A server that doesn't point at /v1/responses may not implement it."""
        assert not OpenAICompletion._is_responses_only_error(
            make_tools_error(
                "Function tools with reasoning_effort are not supported for this model."
            )
        )

    def test_falls_back_to_responses_without_a_none_retry(self, monkeypatch):
        llm = build("us.openai.gpt-6-astra")
        sent: list[dict] = []
        rerouted: list[dict] = []

        def fail_completion(params, **kwargs):
            sent.append(params)
            raise make_tools_error(TOOLS_RESPONSES_ONLY)

        monkeypatch.setattr(llm, "_handle_completion", fail_completion)
        monkeypatch.setattr(
            llm, "_call_responses", lambda **kwargs: rerouted.append(kwargs) or "ok"
        )

        assert llm._call_completions(MESSAGES, tools=TOOLS) == "ok"
        assert len(sent) == 1, "no reasoning_effort='none' retry"
        assert rerouted[0]["tools"] == TOOLS

    def test_custom_endpoint_falls_back_too(self, monkeypatch):
        """The 400 names /v1/responses, unlike a 404 from a server that may lack it."""
        llm = build(
            "us.openai.gpt-6-astra",
            custom_openai=True,
            base_url="https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1",
        )
        sent: list[dict] = []

        def fail_completion(params, **kwargs):
            sent.append(params)
            raise make_tools_error(TOOLS_RESPONSES_ONLY)

        monkeypatch.setattr(llm, "_handle_completion", fail_completion)
        monkeypatch.setattr(llm, "_call_responses", lambda **kwargs: "ok")

        assert llm._call_completions(MESSAGES, tools=TOOLS) == "ok"
        assert len(sent) == 1

    @pytest.mark.asyncio
    async def test_async_path_falls_back_too(self, monkeypatch):
        llm = build("us.openai.gpt-6-astra")
        sent: list[dict] = []

        async def fail_completion(params, **kwargs):
            sent.append(params)
            raise make_tools_error(TOOLS_RESPONSES_ONLY)

        async def ok_responses(**kwargs):
            return "ok"

        monkeypatch.setattr(llm, "_ahandle_completion", fail_completion)
        monkeypatch.setattr(llm, "_acall_responses", ok_responses)

        assert await llm._acall_completions(MESSAGES, tools=TOOLS) == "ok"
        assert len(sent) == 1
