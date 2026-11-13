"""Shared contract for ``LLMMessage.content`` across native providers.

``LLMMessage.content`` is ``str | list[dict] | None``. These tests pin what
each native provider puts on the wire for each shape, using the provider's
own formatting and request-building code with no SDK calls or API keys.

Shared contract (asserted for every provider):

- string content is sent verbatim;
- text-only and assistant content parts are sent unchanged;
- an image part survives with its bytes, in the provider's own block format;
- ``None`` content never drops the message and never becomes the text ``"None"``.

Provider differences are recorded explicitly in ``PROVIDERS`` instead of
being normalised away: OpenAI forwards ``None`` and keeps system messages in
the list, Anthropic sends ``""`` and lifts system text into ``system``.

Gemini sends ``types.Content`` objects rather than message dicts, so it gets
its own section below that asserts the same contract on Gemini's shape:
``model`` for the assistant role, ``Part`` objects for content, and system
text in ``GenerateContentConfig.system_instruction``.
"""

from __future__ import annotations

import base64
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest

from crewai.llms.base_llm import BaseLLM
from crewai.llms.providers.anthropic.completion import AnthropicCompletion
from crewai.llms.providers.openai.completion import OpenAICompletion
from crewai.utilities.agent_utils import message_content_text
from crewai.utilities.types import LLMMessage


try:
    from google.genai import types as genai_types

    from crewai.llms.providers.gemini.completion import GeminiCompletion

    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

_PNG_B64 = "iVBORw0KGgo="
_IMAGE_URL_PART = {
    "type": "image_url",
    "image_url": {"url": f"data:image/png;base64,{_PNG_B64}"},
}


@dataclass(frozen=True)
class Provider:
    name: str
    build: Callable[[], BaseLLM]
    wire: Callable[[Any, list[LLMMessage]], tuple[list[dict[str, Any]], Any]]
    none_content_sent_as: Any
    system_in_message_list: bool
    image_data: Callable[[dict[str, Any]], str]


def _openai_wire(llm: Any, messages: list[LLMMessage]) -> tuple[list[dict[str, Any]], Any]:
    params = llm._prepare_completion_params(llm._format_messages(messages))
    return params["messages"], None


def _anthropic_wire(llm: Any, messages: list[LLMMessage]) -> tuple[list[dict[str, Any]], Any]:
    formatted, system = llm._format_messages_for_anthropic(messages)
    params = llm._prepare_completion_params(formatted, system)
    return params["messages"], params.get("system")


PROVIDERS = [
    Provider(
        name="openai",
        build=lambda: OpenAICompletion(model="gpt-4o", api_key="test-key"),
        wire=_openai_wire,
        none_content_sent_as=None,
        system_in_message_list=True,
        image_data=lambda part: part["image_url"]["url"].split(",", 1)[1],
    ),
    Provider(
        name="anthropic",
        build=lambda: AnthropicCompletion(model="claude-sonnet-4-5", api_key="test-key"),
        wire=_anthropic_wire,
        none_content_sent_as="",
        system_in_message_list=False,
        image_data=lambda part: part["source"]["data"],
    ),
]


@pytest.fixture(params=PROVIDERS, ids=lambda p: p.name)
def provider(request: pytest.FixtureRequest) -> Provider:
    return request.param


def _send(provider: Provider, messages: list[LLMMessage]) -> tuple[list[dict[str, Any]], Any]:
    return provider.wire(provider.build(), messages)


def test_string_content_is_sent_verbatim(provider: Provider) -> None:
    sent, _ = _send(provider, [{"role": "user", "content": "What is 2 + 2?"}])

    assert sent == [{"role": "user", "content": "What is 2 + 2?"}]


def test_text_only_parts_are_sent_unchanged(provider: Provider) -> None:
    parts = [{"type": "text", "text": "first"}, {"type": "text", "text": "second"}]

    sent, _ = _send(provider, [{"role": "user", "content": parts}])

    assert sent == [{"role": "user", "content": parts}]


def test_image_part_keeps_its_bytes_in_provider_format(provider: Provider) -> None:
    content = [{"type": "text", "text": "Describe this."}, _IMAGE_URL_PART]

    sent, _ = _send(provider, [{"role": "user", "content": content}])

    assert len(sent) == 1
    text_part, image_part = sent[0]["content"]
    assert text_part == {"type": "text", "text": "Describe this."}
    assert provider.image_data(image_part) == _PNG_B64


def test_assistant_parts_are_sent_unchanged(provider: Provider) -> None:
    parts = [{"type": "text", "text": "Earlier answer."}]
    messages: list[LLMMessage] = [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": parts},
        {"role": "user", "content": "Follow-up"},
    ]

    sent, _ = _send(provider, messages)

    assert [m["role"] for m in sent] == ["user", "assistant", "user"]
    assert sent[1]["content"] == parts


@pytest.mark.parametrize("role", ["user", "assistant"])
def test_none_content_keeps_the_message_and_never_becomes_text_none(
    provider: Provider, role: str
) -> None:
    messages: list[LLMMessage] = [
        {"role": "user", "content": "Question"},
        {"role": role, "content": None},  # type: ignore[typeddict-item]
        {"role": "user", "content": "Follow-up"},
    ]

    sent, system = _send(provider, messages)

    assert len(sent) == 3
    assert sent[1]["role"] == role
    assert sent[1]["content"] == provider.none_content_sent_as
    assert all(m["content"] != "None" for m in sent)
    assert system is None


def test_system_text_reaches_the_request(provider: Provider) -> None:
    messages: list[LLMMessage] = [
        {"role": "system", "content": "Answer tersely."},
        {"role": "user", "content": "Hi"},
    ]

    sent, system = _send(provider, messages)

    if provider.system_in_message_list:
        assert sent[0] == {"role": "system", "content": "Answer tersely."}
        assert system is None
    else:
        assert system == "Answer tersely."
        assert all(m["role"] != "system" for m in sent)
    assert sent[-1] == {"role": "user", "content": "Hi"}


def test_formatting_does_not_mutate_caller_content(provider: Provider) -> None:
    parts = [{"type": "text", "text": "look"}, dict(_IMAGE_URL_PART)]
    messages: list[LLMMessage] = [{"role": "user", "content": parts}]
    snapshot = repr(messages)

    _send(provider, messages)

    assert repr(messages) == snapshot


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("plain", "plain"),
        (None, ""),
        ([{"type": "text", "text": "a"}, _IMAGE_URL_PART, {"type": "text", "text": "b"}], "a b"),
    ],
    ids=["text", "none", "multimodal"],
)
def test_message_content_text_collapses_each_shape(content: Any, expected: str) -> None:
    """``message_content_text`` is the one place content may become a string."""
    assert message_content_text({"role": "user", "content": content}) == expected


# --- Gemini --------------------------------------------------------------

requires_gemini = pytest.mark.skipif(not HAS_GEMINI, reason="google-genai not installed")


def _gemini() -> Any:
    return GeminiCompletion(model="gemini-2.0-flash-001", api_key="test-key")


def _gemini_wire(llm: Any, messages: list[LLMMessage]) -> tuple[list[Any], Any]:
    """Gemini's request pieces: the ``contents`` turns and the system instruction."""
    contents, system = llm._format_messages_for_gemini(messages)
    config = llm._prepare_generation_config(system)
    return contents, config.system_instruction


def _part_texts(content: Any) -> list[str | None]:
    return [part.text for part in content.parts or []]


@requires_gemini
def test_gemini_string_content_is_sent_verbatim() -> None:
    contents, _ = _gemini_wire(_gemini(), [{"role": "user", "content": "What is 2 + 2?"}])

    assert contents == [
        genai_types.Content(role="user", parts=[genai_types.Part(text="What is 2 + 2?")])
    ]


@requires_gemini
def test_gemini_text_only_parts_are_sent_unchanged() -> None:
    parts = [{"type": "text", "text": "first"}, {"type": "text", "text": "second"}]

    contents, _ = _gemini_wire(_gemini(), [{"role": "user", "content": parts}])

    assert len(contents) == 1
    assert contents[0].role == "user"
    assert _part_texts(contents[0]) == ["first", "second"]


def _assert_image_part_keeps_its_bytes(image_part: dict[str, Any]) -> None:
    content = [{"type": "text", "text": "Describe this."}, image_part]

    contents, _ = _gemini_wire(_gemini(), [{"role": "user", "content": content}])

    assert len(contents) == 1
    text_part, sent_image = contents[0].parts
    assert text_part.text == "Describe this."
    assert sent_image.inline_data.data == base64.b64decode(_PNG_B64)
    assert sent_image.inline_data.mime_type == "image/png"


@requires_gemini
def test_gemini_inline_data_part_keeps_its_bytes() -> None:
    _assert_image_part_keeps_its_bytes(
        {"inlineData": {"mimeType": "image/png", "data": _PNG_B64}}
    )


@requires_gemini
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known defect (#8434): _format_messages_for_gemini silently drops "
        "OpenAI-style image_url parts, which step_executor emits for vision "
        "observations"
    ),
)
def test_gemini_image_url_part_keeps_its_bytes() -> None:
    _assert_image_part_keeps_its_bytes(_IMAGE_URL_PART)


@requires_gemini
def test_gemini_assistant_parts_are_sent_as_model_turn() -> None:
    messages: list[LLMMessage] = [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": [{"type": "text", "text": "Earlier answer."}]},
        {"role": "user", "content": "Follow-up"},
    ]

    contents, _ = _gemini_wire(_gemini(), messages)

    assert [c.role for c in contents] == ["user", "model", "user"]
    assert _part_texts(contents[1]) == ["Earlier answer."]


@requires_gemini
@pytest.mark.parametrize(("role", "gemini_role"), [("user", "user"), ("assistant", "model")])
def test_gemini_none_content_keeps_the_message_and_never_becomes_text_none(
    role: str, gemini_role: str
) -> None:
    messages: list[LLMMessage] = [
        {"role": "user", "content": "Question"},
        {"role": role, "content": None},  # type: ignore[typeddict-item]
        {"role": "user", "content": "Follow-up"},
    ]

    contents, system = _gemini_wire(_gemini(), messages)

    assert len(contents) == 3
    assert contents[1].role == gemini_role
    assert _part_texts(contents[1]) == [""]
    assert all("None" not in (text or "") for c in contents for text in _part_texts(c))
    assert system is None


@requires_gemini
def test_gemini_system_text_reaches_system_instruction() -> None:
    messages: list[LLMMessage] = [
        {"role": "system", "content": "Answer tersely."},
        {"role": "user", "content": "Hi"},
    ]

    contents, system = _gemini_wire(_gemini(), messages)

    # Gemini carries system text in the config, as a Content with role "user".
    assert system == genai_types.Content(
        role="user", parts=[genai_types.Part(text="Answer tersely.")]
    )
    assert [c.role for c in contents] == ["user"]
    assert _part_texts(contents[0]) == ["Hi"]


@requires_gemini
def test_gemini_formatting_does_not_mutate_caller_content() -> None:
    parts = [
        {"type": "text", "text": "look"},
        {"inlineData": {"mimeType": "image/png", "data": _PNG_B64}},
    ]
    messages: list[LLMMessage] = [{"role": "user", "content": parts}]
    snapshot = repr(messages)

    _gemini_wire(_gemini(), messages)

    assert repr(messages) == snapshot


@requires_gemini
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known defect: _format_messages_for_gemini sends str() of a non-string "
        "text block instead of skipping it"
    ),
)
def test_gemini_non_string_text_part_is_skipped() -> None:
    content = [
        {"type": "text", "text": {"nested": 1}},
        {"type": "text", "text": "kept"},
    ]

    contents, _ = _gemini_wire(_gemini(), [{"role": "user", "content": content}])

    assert _part_texts(contents[0]) == ["kept"]
