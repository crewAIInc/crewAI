"""Retrying a LiteLLM model that rejects ``stop`` must not drop other params.

A model that reports ``Unsupported parameter: 'stop'`` is retried with ``stop``
added to ``additional_drop_params``. ``additional_params`` also carries the
caller's own request settings (endpoints, headers, retry counts), so the retry
may only extend that mapping, never replace it.
"""

from collections.abc import Callable
import importlib.util
from typing import Any

import pytest

from crewai.llm import LLM


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("litellm") is None,
    reason="the crewai[litellm] extra is not installed",
)


UNSUPPORTED_STOP_ERROR = (
    "litellm.BadRequestError: OpenAIException - Unsupported parameter: "
    "'stop' is not supported with this model."
)

USER_ADDITIONAL_PARAMS: dict[str, Any] = {
    "num_retries": 5,
    "extra_headers": {"X-Tenant": "acme"},
}


def _llm(additional_params: dict[str, Any]) -> LLM:
    return LLM(
        model="gpt-4o-mini",
        is_litellm=True,
        api_key="test-key",
        additional_params=additional_params,
    )


def _rejecting_completion(attempts: list[dict[str, Any]]) -> Callable[..., str]:
    """Provider stub: reject the first request, accept the retry."""

    def fake(**kwargs: Any) -> str:
        attempts.append(kwargs)
        if len(attempts) == 1:
            raise Exception(UNSUPPORTED_STOP_ERROR)
        return "ok"

    return fake


def _patch_prepare(monkeypatch: pytest.MonkeyPatch, llm: LLM) -> None:
    """The retry only needs the request bookkeeping, not real completion params."""
    monkeypatch.setattr(llm, "_prepare_completion_params", lambda *a, **kw: {})


def _assert_user_params_survive(llm: LLM) -> None:
    for key, value in USER_ADDITIONAL_PARAMS.items():
        assert llm.additional_params[key] == value
    assert llm.additional_params["additional_drop_params"] == ["stop"]


def test_unsupported_stop_retry_keeps_additional_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = _llm(dict(USER_ADDITIONAL_PARAMS))
    attempts: list[dict[str, Any]] = []

    _patch_prepare(monkeypatch, llm)
    monkeypatch.setattr(
        llm, "_handle_non_streaming_response", _rejecting_completion(attempts)
    )

    assert llm.call("hello") == "ok"
    assert len(attempts) == 2
    _assert_user_params_survive(llm)


@pytest.mark.asyncio
async def test_unsupported_stop_retry_keeps_additional_params_async(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = _llm(dict(USER_ADDITIONAL_PARAMS))
    attempts: list[dict[str, Any]] = []
    completion = _rejecting_completion(attempts)

    async def fake_completion(**kwargs: Any) -> str:
        return completion(**kwargs)

    _patch_prepare(monkeypatch, llm)
    monkeypatch.setattr(llm, "_ahandle_non_streaming_response", fake_completion)

    assert await llm.acall("hello") == "ok"
    assert len(attempts) == 2
    _assert_user_params_survive(llm)


def test_unsupported_stop_retry_extends_the_callers_drop_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    caller_drop_params = ["presence_penalty"]
    llm = _llm({"additional_drop_params": caller_drop_params})
    attempts: list[dict[str, Any]] = []

    _patch_prepare(monkeypatch, llm)
    monkeypatch.setattr(
        llm, "_handle_non_streaming_response", _rejecting_completion(attempts)
    )

    assert llm.call("hello") == "ok"
    assert llm.additional_params["additional_drop_params"] == [
        "presence_penalty",
        "stop",
    ]
    assert caller_drop_params == ["presence_penalty"]
