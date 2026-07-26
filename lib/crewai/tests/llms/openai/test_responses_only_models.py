"""Tests for models that /v1/chat/completions doesn't serve at all.

The pro tier exists but is Responses-API-only. A chat completion returns 404 with
one of two messages, depending on the family:

    This model is only supported in v1/responses and not in v1/chat/completions.
    This is not a chat model and thus not supported in the v1/chat/completions
    endpoint. Did you mean to use v1/completions?

Measured 2026-07 against the live endpoints:

    model         /v1/chat/completions   /v1/responses
    gpt-5-pro     404                    OK
    gpt-5.5-pro   404                    OK
    gpt-5.4-pro   404                    OK
    gpt-5.2-pro   404                    OK
    o1-pro        404                    OK
    o3-pro        404                    OK

Since they work on the Responses API, requests are routed there rather than
failing with a misleading "model not found".
"""

import httpx
import pytest
from openai import NotFoundError

from crewai.llms.providers.openai.completion import OpenAICompletion


MESSAGES = [{"role": "user", "content": "hi"}]


def build(model: str, **kwargs) -> OpenAICompletion:
    return OpenAICompletion(model=model, api_key="sk-test", **kwargs)


def make_not_found(message: str) -> NotFoundError:
    body = {"error": {"message": message, "type": "invalid_request_error"}}
    response = httpx.Response(
        status_code=404,
        json=body,
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
    )
    return NotFoundError(message, response=response, body=body)


class TestModelNameNormalization:
    @pytest.mark.parametrize(
        ("configured", "expected"),
        [
            ("gpt-5-pro", "gpt-5-pro"),
            ("openai/gpt-5-pro", "gpt-5-pro"),
            ("GPT-5-Pro", "gpt-5-pro"),
            ("gpt-5-pro-2025-10-06", "gpt-5-pro"),
            ("openai/gpt-5.5-pro-2026-04-23", "gpt-5.5-pro"),
            ("gpt-5.5", "gpt-5.5"),
        ],
    )
    def test_strips_prefix_and_snapshot(self, configured: str, expected: str):
        assert OpenAICompletion._normalize_model_name(configured) == expected


class TestResponsesOnlyDetection:
    @pytest.mark.parametrize(
        "model",
        [
            "gpt-5-pro",
            "gpt-5.2-pro",
            "gpt-5.4-pro",
            "gpt-5.5-pro",
            "o1-pro",
            "o3-pro",
            "openai/gpt-5.5-pro",
            "gpt-5-pro-2025-10-06",
        ],
    )
    def test_detects_responses_only_models(self, model: str):
        assert OpenAICompletion._is_responses_only_model(model)

    @pytest.mark.parametrize(
        "model",
        ["gpt-5.5", "gpt-5.6-sol", "gpt-4o", "o3", "o3-mini", "gpt-5.2"],
    )
    def test_leaves_chat_models_alone(self, model: str):
        assert not OpenAICompletion._is_responses_only_model(model)

    def test_does_not_match_unrelated_pro_names(self):
        """The check is an exact model list, not a bare '-pro' substring."""
        assert not OpenAICompletion._is_responses_only_model("my-pro-deployment")
        assert not OpenAICompletion._is_responses_only_model("gpt-4-pro-custom")


class TestEffectiveApiRouting:
    def test_routes_responses_only_model_to_responses(self):
        llm = build("gpt-5-pro")

        assert llm.api == "completions"
        assert llm._effective_api() == "responses"

    def test_leaves_chat_models_on_completions(self):
        llm = build("gpt-5.5")

        assert llm._effective_api() == "completions"

    def test_explicit_responses_is_untouched(self):
        llm = build("gpt-5.5", api="responses")

        assert llm._effective_api() == "responses"

    @pytest.mark.parametrize("model", ["o1-pro", "openai/gpt-5-pro", "gpt-5.5-pro"])
    def test_custom_openai_endpoint_is_never_rerouted(self, model: str):
        """An OpenAI-compatible server may serve any model name on chat completions.

        The model list describes OpenAI's own deployment. Most compatible servers
        (vLLM, LiteLLM proxies, Ollama) don't implement /v1/responses at all, so
        rerouting a self-hosted "o1-pro" would break a working setup.
        """
        llm = build(
            model, custom_openai=True, base_url="https://my-vllm.internal/v1"
        )

        assert llm._effective_api() == "completions"

    def test_custom_openai_still_honours_explicit_responses(self):
        llm = build(
            "o1-pro",
            api="responses",
            custom_openai=True,
            base_url="https://my-vllm.internal/v1",
        )

        assert llm._effective_api() == "responses"

    def test_call_dispatches_pro_model_to_responses_handler(self, monkeypatch):
        """A pro model must reach the Responses path, not chat completions."""
        llm = build("gpt-5-pro")
        called: list[str] = []

        monkeypatch.setattr(
            llm, "_call_responses", lambda **kwargs: called.append("responses") or "ok"
        )
        monkeypatch.setattr(
            llm,
            "_call_completions",
            lambda **kwargs: called.append("completions") or "ok",
        )

        result = llm.call("hi")

        assert result == "ok"
        assert called == ["responses"]


class TestNotFoundMessage:
    @pytest.mark.parametrize(
        "message",
        [
            "This model is only supported in v1/responses and not in /v1/chat/completions.",
            "This is not a chat model and thus not supported in the v1/chat/completions endpoint.",
        ],
    )
    def test_points_at_responses_api(self, message: str):
        llm = build("gpt-5.5")

        msg = llm._model_not_found_message(make_not_found(message))

        assert 'api="responses"' in msg
        assert "not available on /v1/chat/completions" in msg

    def test_keeps_plain_not_found_for_real_typos(self):
        llm = build("gpt-5.5")

        msg = llm._model_not_found_message(
            make_not_found("The model `gpt-5.99` does not exist.")
        )

        assert "not found" in msg
        assert 'api="responses"' not in msg

    def test_known_pro_model_is_flagged_regardless_of_wording(self):
        """Even if OpenAI rewords the 404, a known pro model gets the hint."""
        llm = build("gpt-5-pro")

        msg = llm._model_not_found_message(make_not_found("Something went wrong."))

        assert 'api="responses"' in msg

    def test_custom_endpoint_does_not_get_name_based_hint(self):
        """Don't advise api="responses" for a server that may not implement it."""
        llm = build(
            "o1-pro", custom_openai=True, base_url="https://my-vllm.internal/v1"
        )

        msg = llm._model_not_found_message(make_not_found("Model does not exist."))

        assert 'api="responses"' not in msg

    def test_custom_endpoint_still_trusts_the_server_response(self):
        """If the server itself says responses-only, relay that regardless."""
        llm = build(
            "o1-pro", custom_openai=True, base_url="https://my-vllm.internal/v1"
        )

        msg = llm._model_not_found_message(
            make_not_found("This model is only supported in v1/responses.")
        )

        assert 'api="responses"' in msg
