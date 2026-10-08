"""`reasoning_effort` reaches every reasoning model, not just o1.

The completions path used to gate the parameter behind
``is_o1_model = "o1" in model.lower()``, a literal substring test. gpt-5, o3 and
o4-mini contain no "o1", so an explicitly configured effort was dropped and the
model thought at the server default -- silently, since the request still
succeeded.

The gate could not simply be widened: ``is_o1_model`` also drives
``supports_function_calling``, ``supports_stop_words`` and the system->user
message rewrite, so marking gpt-5 as an o1 model would report that it cannot
call tools. ``_supports_reasoning_effort`` is a separate predicate matched on
model *shape* -- the o-series, and GPT generation 5 onwards -- so a new member of
an existing family needs no release, and a non-reasoning model never pays a
wasted round trip. If the shape match is ever wrong for a future family, the
400 is retried without the key rather than surfacing.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from openai import APIConnectionError, BadRequestError, UnprocessableEntityError

from crewai.llm import LLM
from crewai.llms.providers.openai import completion as completion_module
from crewai.llms.providers.openai.completion import (
    OpenAICompletion,
    _supports_reasoning_effort,
)
from crewai.llms.providers.openai_compatible.completion import (
    OpenAICompatibleCompletion,
)


MESSAGES = [{"role": "user", "content": "hi"}]

REASONING_MODELS = ["gpt-5", "gpt-5-mini", "o3", "o3-mini", "o4-mini", "o1"]


def build(model: str = "gpt-5", **kwargs: Any) -> OpenAICompletion:
    return OpenAICompletion(model=model, api_key="sk-test", **kwargs)


GATEWAY = "https://gateway.example/v1"


def gateway(model: str = "gpt-4o", **kwargs: Any) -> OpenAICompletion:
    """An OpenAI-compatible server fronting whatever it calls `model`."""
    return build(model, base_url=GATEWAY, **kwargs)


def _status_error(cls: type, status: int, body: Any) -> Any:
    """A rejection whose body is not the OpenAI shape, as the SDK raises it."""
    message = f"Error code: {status} - {body}"
    return cls(
        message,
        response=httpx.Response(
            status, json=body, request=httpx.Request("POST", GATEWAY)
        ),
        body=body,
    )


@pytest.fixture(autouse=True)
def forget_rejecting_models():
    completion_module._LEARNED_NO_REASONING_EFFORT_MODELS.clear()
    yield
    completion_module._LEARNED_NO_REASONING_EFFORT_MODELS.clear()


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

    @pytest.mark.parametrize("model", ["gpt-4o", "gpt-4.1", "gpt-3.5-turbo"])
    def test_not_sent_to_non_reasoning_models(self, model):
        """No wasted round trip: these never reach the API with the parameter."""
        params = build(model, reasoning_effort="high")._prepare_completion_params(
            MESSAGES
        )

        assert "reasoning_effort" not in params


class TestSupportedModelShape:
    """Matched on shape so a new family member needs no release."""

    @pytest.mark.parametrize(
        "model",
        [
            "o1",
            "o1-mini",
            "o1-preview",
            "o3",
            "o3-mini",
            "o4-mini",
            "gpt-5",
            "gpt-5-mini",
            "gpt-5.1",
            "gpt-5.6-sol",
            "openai/gpt-5",
            "ft:o4-mini-2025-04-16:acme::abc123",
        ],
    )
    def test_supported(self, model):
        assert _supports_reasoning_effort(model) is True

    @pytest.mark.parametrize(
        "model",
        [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4.1",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
            "chatgpt-4o-latest",
            "omni-moderation-latest",
            "text-embedding-3-small",
            "ft:gpt-4o-mini-2024-07-18:acme::abc123",
        ],
    )
    def test_unsupported(self, model):
        assert _supports_reasoning_effort(model) is False

    @pytest.mark.parametrize("model", ["gpt-6", "gpt-7-turbo", "o5", "o9-mini"])
    def test_future_family_members_need_no_release(self, model):
        """The staleness this replaced: an unreleased family must still match."""
        assert _supports_reasoning_effort(model) is True


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
        assert build()._rejects_reasoning_effort_as_unsupported(
            unsupported_parameter_error()
        )

    def test_matches_unrecognized_argument(self):
        assert build()._rejects_reasoning_effort_as_unsupported(
            unrecognized_argument_error()
        )

    def test_ignores_an_unsupported_value(self):
        """Dropping the key here would silently restore the original bug."""
        assert not build()._rejects_reasoning_effort_as_unsupported(
            unsupported_value_error()
        )

    def test_ignores_a_400_about_another_parameter(self):
        assert not build()._rejects_reasoning_effort_as_unsupported(
            _bad_request(
                "Unsupported parameter: 'temperature' is not supported.",
                param="temperature",
                code="unsupported_parameter",
            )
        )

    def test_ignores_unrelated_exceptions(self):
        assert not build()._rejects_reasoning_effort_as_unsupported(
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
        llm = build("gpt-6-future", reasoning_effort="high")
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
        llm = build("gpt-6-future", reasoning_effort="high")
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
        llm = build("gpt-6-future", reasoning_effort="high", stream=True)
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
        llm = build("gpt-6-future", reasoning_effort="high", stream=True)
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
        llm = build("gpt-6-future", reasoning_effort="high")
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

    def test_a_model_that_rejected_the_parameter_is_not_sent_it_again(
        self, monkeypatch
    ):
        """The rejected call is paid once per process, not on every request."""
        llm = build("gpt-6-future", reasoning_effort="high")
        seen: list[dict] = []

        def fake_handle(params, **kwargs):
            seen.append(params)
            if "reasoning_effort" in params:
                raise unsupported_parameter_error()
            return "ok"

        monkeypatch.setattr(llm, "_handle_completion", fake_handle)

        llm._call_completions(MESSAGES)
        llm._call_completions(MESSAGES)

        assert ["reasoning_effort" in params for params in seen] == [
            True,
            False,
            False,
        ]

    def test_a_rejection_is_remembered_only_once_the_retry_succeeds(
        self, monkeypatch
    ):
        """A retry that dies for another reason has not established the fallback.

        Remembering before the retry would silently stop sending a configured
        effort for the rest of the process on the strength of one transient
        failure; the setting must survive until a call without it succeeds.
        """
        llm = build("gpt-6-future", reasoning_effort="high")
        seen: list[dict] = []
        retry_outcome: list[Exception | None] = [
            APIConnectionError(request=httpx.Request("POST", GATEWAY))
        ]

        def fake_handle(params, **kwargs):
            seen.append(params)
            if "reasoning_effort" in params:
                raise unsupported_parameter_error()
            if retry_outcome and (failure := retry_outcome.pop()):
                raise failure
            return "ok"

        monkeypatch.setattr(llm, "_handle_completion", fake_handle)

        with pytest.raises(APIConnectionError):
            llm._call_completions(MESSAGES)
        assert not completion_module._LEARNED_NO_REASONING_EFFORT_MODELS

        assert llm._call_completions(MESSAGES) == "ok"
        assert llm._reasoning_effort_key() in (
            completion_module._LEARNED_NO_REASONING_EFFORT_MODELS
        )
        assert ["reasoning_effort" in params for params in seen] == [
            True,
            False,
            True,
            False,
        ], "the effort is sent again after a failed retry, then dropped for good"

    @pytest.mark.asyncio
    async def test_a_rejection_is_remembered_only_once_the_async_retry_succeeds(
        self, monkeypatch
    ):
        llm = build("gpt-6-future", reasoning_effort="high")
        seen: list[dict] = []
        retry_outcome: list[Exception | None] = [
            APIConnectionError(request=httpx.Request("POST", GATEWAY))
        ]

        async def fake_handle(params, **kwargs):
            seen.append(params)
            if "reasoning_effort" in params:
                raise unsupported_parameter_error()
            if retry_outcome and (failure := retry_outcome.pop()):
                raise failure
            return "ok"

        monkeypatch.setattr(llm, "_ahandle_completion", fake_handle)

        with pytest.raises(APIConnectionError):
            await llm._acall_completions(MESSAGES)
        assert not completion_module._LEARNED_NO_REASONING_EFFORT_MODELS

        assert await llm._acall_completions(MESSAGES) == "ok"
        assert ["reasoning_effort" in params for params in seen] == [
            True,
            False,
            True,
            False,
        ]


class TestCompatibleServers:
    """The model name is the server's namespace, so it says nothing about support."""

    @pytest.mark.parametrize(
        "model", ["gpt-4o", "gpt-oss-120b", "qwen3-235b", "deepseek-r1"]
    )
    def test_explicit_setting_is_forwarded_whatever_the_name(self, model):
        params = gateway(model, reasoning_effort="low")._prepare_completion_params(
            MESSAGES
        )

        assert params["reasoning_effort"] == "low"

    def test_the_flag_marks_a_server_compatible(self):
        params = gateway(
            "gpt-4o", custom_openai=True, reasoning_effort="low"
        )._prepare_completion_params(MESSAGES)

        assert params["reasoning_effort"] == "low"

    def test_a_client_params_base_url_marks_a_server_compatible(self):
        """`client_params` win over the standard fields when the client is built."""
        llm = build("gpt-4o", client_params={"base_url": GATEWAY}, reasoning_effort="low")

        assert llm._prepare_completion_params(MESSAGES)["reasoning_effort"] == "low"

    def test_an_httpx_url_override_is_the_endpoint_the_client_calls(self):
        """The SDK takes `str | httpx.URL`; detection and memory must see both alike."""
        llm = build(
            "gpt-4o",
            client_params={"base_url": httpx.URL(GATEWAY)},
            reasoning_effort="low",
        )

        assert llm._prepare_completion_params(MESSAGES)["reasoning_effort"] == "low"
        assert llm._reasoning_effort_key() == (GATEWAY, "gpt-4o")

    def test_an_env_base_url_marks_a_server_compatible(self, monkeypatch):
        monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")

        params = build("qwen3", reasoning_effort="low")._prepare_completion_params(
            MESSAGES
        )

        assert params["reasoning_effort"] == "low"

    def test_openais_own_url_is_not_a_compatible_server(self):
        llm = build("gpt-4o", base_url="https://api.openai.com/v1", reasoning_effort="low")

        assert "reasoning_effort" not in llm._prepare_completion_params(MESSAGES)

    def test_reaches_the_wire_through_the_llm_factory(self):
        """gpt-oss looks like an OpenAI name to the factory, so it never flags it."""
        llm = LLM(
            model="gpt-oss-120b",
            base_url="http://localhost:8000/v1",
            api_key="sk-test",
            reasoning_effort="high",
        )

        assert llm._prepare_completion_params(MESSAGES)["reasoning_effort"] == "high"

    def test_forwarded_through_a_provider_subclass(self):
        llm = OpenAICompatibleCompletion(
            model="openai/gpt-oss-120b",
            provider="openrouter",
            api_key="sk-test",
            reasoning_effort="low",
        )

        assert llm._prepare_completion_params(MESSAGES)["reasoning_effort"] == "low"

    def test_unset_stays_off_the_wire(self):
        params = gateway("gpt-oss-120b")._prepare_completion_params(MESSAGES)

        assert "reasoning_effort" not in params

    def test_a_rejection_in_the_servers_own_words_is_recovered(self, monkeypatch):
        llm = gateway("qwen3-235b", reasoning_effort="low")
        seen: list[dict] = []

        def fake_handle(params, **kwargs):
            seen.append(params)
            if "reasoning_effort" in params:
                raise _bad_request(
                    "[{'loc': ('body', 'reasoning_effort'), "
                    "'msg': 'Extra inputs are not permitted'}]",
                    param=None,
                )
            return "ok"

        monkeypatch.setattr(llm, "_handle_completion", fake_handle)

        assert llm._call_completions(MESSAGES) == "ok"
        assert len(seen) == 2
        assert "reasoning_effort" not in seen[1]

    @pytest.mark.parametrize(
        "error",
        [
            _status_error(
                BadRequestError, 400, {"error": "unknown field `reasoning_effort`"}
            ),
            _status_error(
                UnprocessableEntityError,
                422,
                {
                    "detail": [
                        {
                            "loc": ["body", "reasoning_effort"],
                            "msg": "Extra inputs are not permitted",
                        }
                    ]
                },
            ),
        ],
        ids=["string-body", "fastapi-422"],
    )
    def test_a_rejection_outside_the_openai_shape_is_recovered(
        self, monkeypatch, error
    ):
        llm = gateway("qwen3-235b", reasoning_effort="low")
        seen: list[dict] = []

        def fake_handle(params, **kwargs):
            seen.append(params)
            if "reasoning_effort" in params:
                raise error
            return "ok"

        monkeypatch.setattr(llm, "_handle_completion", fake_handle)

        assert llm._call_completions(MESSAGES) == "ok"
        assert "reasoning_effort" not in seen[1]

    def test_a_complaint_about_the_value_surfaces_and_is_not_remembered(
        self, monkeypatch
    ):
        """The server takes the parameter; the value is the caller's mistake."""
        llm = gateway("o3", reasoning_effort="minimal")
        calls: list[dict] = []

        def always_fail(params, **kwargs):
            calls.append(params)
            raise _bad_request(
                "Invalid value 'minimal' for reasoning_effort; "
                "must be one of low, medium, high.",
                param="reasoning_effort",
            )

        monkeypatch.setattr(llm, "_handle_completion", always_fail)

        with pytest.raises(BadRequestError, match="Invalid value"):
            llm._call_completions(MESSAGES)

        assert len(calls) == 1
        assert not completion_module._LEARNED_NO_REASONING_EFFORT_MODELS

    @pytest.mark.parametrize(
        "error",
        [
            _status_error(
                UnprocessableEntityError,
                422,
                {
                    "detail": [
                        {
                            "loc": ["body", "reasoning_effort"],
                            "msg": "Input should be 'low', 'medium' or 'high'",
                            "type": "literal_error",
                        }
                    ]
                },
            ),
            _status_error(
                BadRequestError, 400, {"error": "reasoning_effort: something odd"}
            ),
        ],
        ids=["pydantic-enum", "ambiguous"],
    )
    def test_without_evidence_the_parameter_is_unknown_the_error_surfaces(
        self, monkeypatch, error
    ):
        """Only a rejection of the parameter itself is recovered; anything else
        naming it is the caller's to see."""
        llm = gateway("qwen3-235b", reasoning_effort="minimal")
        calls: list[dict] = []

        def always_fail(params, **kwargs):
            calls.append(params)
            raise error

        monkeypatch.setattr(llm, "_handle_completion", always_fail)

        with pytest.raises(type(error)):
            llm._call_completions(MESSAGES)

        assert len(calls) == 1
        assert not completion_module._LEARNED_NO_REASONING_EFFORT_MODELS

    @pytest.mark.parametrize(
        "gateway_config",
        [
            pytest.param({"base_url": GATEWAY}, id="base_url"),
            pytest.param(
                {"client_params": {"base_url": httpx.URL(GATEWAY)}}, id="httpx.URL"
            ),
        ],
    )
    def test_a_gateway_rejection_does_not_silence_the_model_on_openai(
        self, monkeypatch, gateway_config
    ):
        """However the gateway is configured, its rejection is keyed to it, not to OpenAI."""
        llm = build("gpt-5", reasoning_effort="high", **gateway_config)

        def reject(params, **kwargs):
            if "reasoning_effort" in params:
                raise unsupported_parameter_error()
            return "ok"

        monkeypatch.setattr(llm, "_handle_completion", reject)
        llm._call_completions(MESSAGES)

        params = build("gpt-5", reasoning_effort="high")._prepare_completion_params(
            MESSAGES
        )
        assert params["reasoning_effort"] == "high"

    def test_openai_itself_is_held_to_the_known_shapes(self, monkeypatch):
        """The lenient match is for other servers; OpenAI's 400s stay precise."""
        llm = build("gpt-5", reasoning_effort="low")
        calls: list[dict] = []

        def always_fail(params, **kwargs):
            calls.append(params)
            raise _bad_request("Something else about reasoning_effort.", param=None)

        monkeypatch.setattr(llm, "_handle_completion", always_fail)

        with pytest.raises(BadRequestError, match="Something else"):
            llm._call_completions(MESSAGES)

        assert len(calls) == 1


class TestLLMSurface:
    @pytest.mark.parametrize(
        "effort", ["none", "minimal", "low", "medium", "high", "xhigh"]
    )
    def test_llm_accepts_every_documented_effort(self, effort):
        llm = LLM(model="gpt-5", reasoning_effort=effort, is_litellm=True)

        assert llm.reasoning_effort == effort

    def test_reaches_the_wire_through_the_llm_factory(self):
        """End to end: LLM(...) -> native provider -> request params."""
        llm = LLM(model="gpt-5", reasoning_effort="minimal", is_litellm=False)
        params = llm._prepare_completion_params(MESSAGES)

        assert params["reasoning_effort"] == "minimal"
