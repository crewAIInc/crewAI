"""Bedrock's side of the shared native-provider content and usage contract.

Issue #8128 extends the shared provider contracts (#8037 / #8039 / #8062) to
AWS Bedrock. Those suites are not on ``main`` yet, so Bedrock's adapter lives
here with the same shape (``response`` / ``stub_sync`` / ``stub_async`` /
``user_texts`` / ``system_text``) and can join their ``PROVIDERS`` list once
they land. Every test drives a fake Converse client: no AWS credentials, no
network, no paid call.

Shared expectations asserted here:

- string content reaches the model verbatim;
- a parts list is sent as content blocks, never as the list's ``repr``;
- ``None`` content keeps its message and never becomes the text ``"None"``;
- ``call`` and ``acall`` send the same request and return the same text;
- each call adds its prompt and completion tokens to the usage summary.

Bedrock-specific semantics, kept in ``_BedrockContract`` rather than shared:

- message content is a list of Converse blocks (``{"text": ...}``,
  ``{"image": ...}``), so a string becomes ``[{"text": s}]``;
- system text goes to the top-level ``system`` parameter, not the turns;
- usage arrives as ``inputTokens`` / ``outputTokens`` / ``totalTokens``, and
  ``LLMCallCompletedEvent.usage`` carries that raw dict, so no shared
  ``total_tokens`` key is asserted on the event here;
- a response with no content blocks returns an apology string rather than
  ``""``, so the shared "missing text becomes ``""``" rule is not asserted.

Known defects are strict ``xfail`` marks against the issue that tracks them,
so the mark fails loudly once the fix lands.
"""

from __future__ import annotations

import base64
import copy
from typing import Any

from crewai.llm import LLM
from crewai.llms.base_llm import BaseLLM
from crewai.llms.providers.bedrock import completion as bedrock_completion
from crewai.llms.providers.bedrock.completion import BedrockCompletion
from crewai.utilities.types import LLMMessage
import pytest


MODEL = "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"
PROMPT = "Name the capital of Italy."
ANSWER = "Rome."
PROMPT_TOKENS = 11
COMPLETION_TOKENS = 5
PNG_BYTES = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"

_IMAGE_BLOCK: dict[str, Any] = {
    "image": {"format": "png", "source": {"bytes": PNG_BYTES}}
}


class _Recorder:
    """Stands in for ``converse``, recording the request it received."""

    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.requests: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.requests.append(kwargs)
        return self.response

    @property
    def request(self) -> dict[str, Any]:
        assert len(self.requests) == 1, (
            f"expected exactly one Converse request, got {len(self.requests)}"
        )
        return self.requests[0]


class _AsyncRecorder(_Recorder):
    """``_Recorder`` for the aiobotocore client, which is awaited."""

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        self.requests.append(kwargs)
        return self.response


class _FakeAsyncClient:
    """The slice of the aiobotocore ``bedrock-runtime`` client Bedrock uses."""

    def __init__(self, converse: _AsyncRecorder) -> None:
        self.converse = converse


class _BedrockContract:
    """Adapter from Bedrock's Converse surface to the shared contract."""

    id = "bedrock"
    model = MODEL

    def build(self, **kwargs: Any) -> BedrockCompletion:
        llm = LLM(model=self.model, **kwargs)
        assert isinstance(llm, BedrockCompletion)
        return llm

    def response(self, text: str) -> dict[str, Any]:
        """A fake Converse response carrying ``text`` and fixed token usage."""
        return {
            "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
            "stopReason": "end_turn",
            "usage": {
                "inputTokens": PROMPT_TOKENS,
                "outputTokens": COMPLETION_TOKENS,
                "totalTokens": PROMPT_TOKENS + COMPLETION_TOKENS,
            },
        }

    def stub_sync(
        self,
        llm: BedrockCompletion,
        monkeypatch: pytest.MonkeyPatch,
        text: str = ANSWER,
    ) -> _Recorder:
        recorder = _Recorder(self.response(text))
        monkeypatch.setattr(llm._client, "converse", recorder)
        return recorder

    def stub_async(
        self,
        llm: BedrockCompletion,
        monkeypatch: pytest.MonkeyPatch,
        text: str = ANSWER,
    ) -> _AsyncRecorder:
        recorder = _AsyncRecorder(self.response(text))
        client = _FakeAsyncClient(recorder)

        async def ensure_async_client() -> _FakeAsyncClient:
            return client

        # Without aiobotocore, ``acall`` silently runs ``call`` in a thread;
        # force the native async path so this really tests it.
        monkeypatch.setattr(bedrock_completion, "AIOBOTOCORE_AVAILABLE", True)
        monkeypatch.setattr(llm, "_ensure_async_client", ensure_async_client)
        return recorder

    @staticmethod
    def user_blocks(request: dict[str, Any]) -> list[list[dict[str, Any]]]:
        """Content blocks of every ``user`` turn in a recorded request."""
        return [msg["content"] for msg in request["messages"] if msg["role"] == "user"]

    @staticmethod
    def user_texts(request: dict[str, Any]) -> list[str]:
        """Text of every ``user`` turn in a recorded request."""
        return [
            "".join(block.get("text", "") for block in blocks)
            for blocks in _BedrockContract.user_blocks(request)
        ]

    @staticmethod
    def system_text(request: dict[str, Any]) -> Any:
        """System text from the top-level ``system`` parameter, as sent."""
        system = request.get("system") or []
        assert len(system) <= 1, f"expected one system block, got {system!r}"
        return system[0]["text"] if system else ""


BEDROCK = _BedrockContract()


def _send(
    messages: list[LLMMessage], monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    llm = BEDROCK.build()
    recorder = BEDROCK.stub_sync(llm, monkeypatch)
    llm.call(messages)
    return recorder.request


# --- Message content -----------------------------------------------------------


def test_string_content_is_sent_verbatim_as_one_text_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _send([{"role": "user", "content": PROMPT}], monkeypatch)

    assert BEDROCK.user_blocks(request) == [[{"text": PROMPT}]]


def test_parts_list_is_sent_as_blocks_with_image_bytes_intact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parts: list[dict[str, Any]] = [{"text": "What is in this image?"}, _IMAGE_BLOCK]

    request = _send([{"role": "user", "content": parts}], monkeypatch)

    [blocks] = BEDROCK.user_blocks(request)
    assert blocks == parts
    assert blocks[1]["image"]["source"]["bytes"] == PNG_BYTES
    assert all(isinstance(block, dict) for block in blocks)


@pytest.mark.xfail(
    reason="#9095: OpenAI-style text parts are forwarded to Converse unconverted",
    strict=True,
)
def test_openai_style_text_part_is_sent_as_a_converse_text_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parts: list[dict[str, Any]] = [{"type": "text", "text": PROMPT}]

    request = _send([{"role": "user", "content": parts}], monkeypatch)

    assert BEDROCK.user_blocks(request) == [[{"text": PROMPT}]]


@pytest.mark.xfail(
    reason="#9095: image_url data URIs are forwarded to Converse unconverted",
    strict=True,
)
def test_openai_style_image_url_part_is_sent_as_a_converse_image_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_uri = "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode()
    parts: list[dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": data_uri}}
    ]

    request = _send([{"role": "user", "content": parts}], monkeypatch)

    assert BEDROCK.user_blocks(request) == [[_IMAGE_BLOCK]]


def test_none_user_content_keeps_the_message_and_never_becomes_text_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _send([{"role": "user", "content": None}], monkeypatch)

    assert len(BEDROCK.user_blocks(request)) == 1
    assert "None" not in BEDROCK.user_texts(request)[0]


def test_none_assistant_content_keeps_the_turn_and_never_becomes_text_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages: list[LLMMessage] = [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": None},
        {"role": "user", "content": "Second question"},
    ]

    request = _send(messages, monkeypatch)

    assert [msg["role"] for msg in request["messages"]] == [
        "user",
        "assistant",
        "user",
    ]
    assistant_blocks = request["messages"][1]["content"]
    assert all("None" not in block.get("text", "") for block in assistant_blocks)


def test_formatting_does_not_mutate_caller_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages: list[LLMMessage] = [
        {"role": "system", "content": "Answer briefly."},
        {"role": "user", "content": [{"text": "Describe it."}, _IMAGE_BLOCK]},
    ]
    original = copy.deepcopy(messages)

    _send(messages, monkeypatch)

    assert messages == original


def test_system_text_goes_to_the_system_parameter_not_the_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages: list[LLMMessage] = [
        {"role": "system", "content": "Answer briefly."},
        {"role": "user", "content": PROMPT},
    ]

    request = _send(messages, monkeypatch)

    assert BEDROCK.system_text(request) == "Answer briefly."
    assert [msg["role"] for msg in request["messages"]] == ["user"]
    assert BEDROCK.user_texts(request) == [PROMPT]


@pytest.mark.xfail(
    reason="#7920: a parts list is sent as the raw list in system[0].text",
    strict=True,
)
def test_parts_list_system_content_is_sent_as_its_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages: list[LLMMessage] = [
        {"role": "system", "content": [{"type": "text", "text": "Answer briefly."}]},
        {"role": "user", "content": PROMPT},
    ]

    request = _send(messages, monkeypatch)

    assert BEDROCK.system_text(request) == "Answer briefly."


@pytest.mark.xfail(
    reason="#7920: a later None system message is joined as the text 'None'",
    strict=True,
)
def test_none_system_content_never_becomes_text_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages: list[LLMMessage] = [
        {"role": "system", "content": "Answer briefly."},
        {"role": "system", "content": None},
        {"role": "user", "content": PROMPT},
    ]

    request = _send(messages, monkeypatch)

    assert "None" not in BEDROCK.system_text(request)


# --- Sync/async text and token usage ------------------------------------------


def _assert_usage(llm: BaseLLM, calls: int) -> None:
    usage = llm.get_token_usage_summary()
    assert usage.prompt_tokens == PROMPT_TOKENS * calls
    assert usage.completion_tokens == COMPLETION_TOKENS * calls
    assert usage.total_tokens == (PROMPT_TOKENS + COMPLETION_TOKENS) * calls
    assert usage.successful_requests == calls


def test_call_returns_the_text_and_counts_its_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = BEDROCK.build()
    recorder = BEDROCK.stub_sync(llm, monkeypatch)

    result = llm.call(PROMPT)

    assert result == ANSWER
    assert BEDROCK.user_texts(recorder.request) == [PROMPT]
    _assert_usage(llm, calls=1)


@pytest.mark.asyncio
async def test_acall_returns_the_text_and_counts_its_tokens_via_the_async_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = BEDROCK.build()
    sync_recorder = BEDROCK.stub_sync(llm, monkeypatch)
    async_recorder = BEDROCK.stub_async(llm, monkeypatch)

    result = await llm.acall(PROMPT)

    assert result == ANSWER
    assert BEDROCK.user_texts(async_recorder.request) == [PROMPT]
    assert sync_recorder.requests == []
    _assert_usage(llm, calls=1)


@pytest.mark.asyncio
async def test_acall_sends_the_same_request_as_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages: list[LLMMessage] = [
        {"role": "system", "content": "Answer briefly."},
        {"role": "user", "content": [{"text": "Describe it."}, _IMAGE_BLOCK]},
        {"role": "assistant", "content": None},
        {"role": "user", "content": PROMPT},
    ]
    sync_llm = BEDROCK.build()
    sync_recorder = BEDROCK.stub_sync(sync_llm, monkeypatch)
    async_llm = BEDROCK.build()
    async_recorder = BEDROCK.stub_async(async_llm, monkeypatch)

    sync_result = sync_llm.call(messages)
    async_result = await async_llm.acall(messages)

    assert async_result == sync_result == ANSWER
    assert async_recorder.request == sync_recorder.request


@pytest.mark.asyncio
async def test_usage_accumulates_across_call_and_acall(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = BEDROCK.build()
    BEDROCK.stub_sync(llm, monkeypatch)
    BEDROCK.stub_async(llm, monkeypatch)

    llm.call(PROMPT)
    _assert_usage(llm, calls=1)

    await llm.acall(PROMPT)
    _assert_usage(llm, calls=2)
