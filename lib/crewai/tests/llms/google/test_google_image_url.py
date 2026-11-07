"""Offline regression tests for OpenAI-style images in native Gemini requests."""

import base64
from collections.abc import AsyncIterator
from copy import deepcopy
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from crewai.agents.step_executor import StepExecutor
from crewai.llms.providers.gemini.completion import GeminiCompletion
from crewai.utilities.types import LLMMessage
from google.genai import types
import pytest


# Complete one-pixel PNG; no API-dependent image fixture.
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
)


def image_part(data: bytes = PNG, mime_type: str = "image/png") -> dict[str, Any]:
    """Build the same data-URI content shape used by StepExecutor."""
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{mime_type};base64,{base64.b64encode(data).decode('ascii')}",
            "detail": "auto",
        },
    }


@pytest.fixture
def llm() -> GeminiCompletion:
    """Construct a provider without creating a network client."""
    with patch("crewai.llms.providers.gemini.completion.genai.Client"):
        return GeminiCompletion(model="gemini-2.0-flash-001", api_key="test-key")


def test_gemini_image_url_part_keeps_its_bytes(llm: GeminiCompletion) -> None:
    """Reproduce #8434 through the real formatter and real Google Part objects."""
    contents, system = llm._format_messages_for_gemini(
        [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Describe this."}, image_part()],
            }
        ]
    )
    assert system is None
    assert contents == [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="Describe this."),
                types.Part.from_bytes(data=PNG, mime_type="image/png"),
            ],
        )
    ]


@pytest.mark.parametrize(
    "mime_type", ["image/png", "image/jpeg", "image/webp", "image/heic", "image/heif"]
)
def test_supported_image_mime_types(llm: GeminiCompletion, mime_type: str) -> None:
    """Conversion preserves arbitrary binary bytes without transcoding them."""
    data = bytes(range(256))
    contents, _ = llm._format_messages_for_gemini(
        [{"role": "user", "content": [image_part(data, mime_type)]}]
    )
    assert contents[0].parts == [types.Part.from_bytes(data=data, mime_type=mime_type)]


@pytest.mark.parametrize("content", ["Hello", [{"type": "text", "text": "Hello"}]])
def test_text_only_unchanged(llm: GeminiCompletion, content: Any) -> None:
    """Both existing text message representations remain unchanged."""
    contents, _ = llm._format_messages_for_gemini(
        [{"role": "user", "content": content}]
    )
    assert contents == [
        types.Content(role="user", parts=[types.Part.from_text(text="Hello")])
    ]


@pytest.mark.parametrize("image_first", [False, True])
def test_text_image_order(llm: GeminiCompletion, image_first: bool) -> None:
    """Text and images keep their relative positions."""
    content = [{"type": "text", "text": "caption"}, image_part()]
    expected = [
        types.Part.from_text(text="caption"),
        types.Part.from_bytes(data=PNG, mime_type="image/png"),
    ]
    if image_first:
        content.reverse()
        expected.reverse()
    contents, _ = llm._format_messages_for_gemini(
        [{"role": "user", "content": content}]
    )
    assert contents[0].parts == expected


def test_mixed_parts_roles_and_input_immutability(llm: GeminiCompletion) -> None:
    """Multiple images coexist with native parts, history and system prompts."""
    messages: list[LLMMessage] = [
        {"role": "system", "content": "Be concise."},
        {"role": "system", "content": [{"type": "text", "text": "Compare images."}]},
        {"role": "user", "content": "First question"},
        {
            "role": "assistant",
            "content": [image_part(), {"type": "text", "text": "Earlier"}],
        },
        {
            "role": "user",
            "content": [
                image_part(),
                {"type": "text", "text": "Compare"},
                {"inlineData": {"data": "AAEC/w==", "mimeType": "image/jpeg"}},
                {
                    "fileData": {
                        "fileUri": "gs://example/image.webp",
                        "mimeType": "image/webp",
                    }
                },
                image_part(b"\x00\xff\x01", "image/heif"),
            ],
        },
    ]
    original = deepcopy(messages)
    contents, system = llm._format_messages_for_gemini(messages)
    assert messages == original
    assert system == "Be concise.\n\nCompare images."
    assert [content.role for content in contents] == ["user", "model", "user"]
    assert contents[1].parts == [
        types.Part.from_bytes(data=PNG, mime_type="image/png"),
        types.Part.from_text(text="Earlier"),
    ]
    assert contents[2].parts == [
        types.Part.from_bytes(data=PNG, mime_type="image/png"),
        types.Part.from_text(text="Compare"),
        types.Part.from_bytes(data=b"\x00\x01\x02\xff", mime_type="image/jpeg"),
        types.Part.from_uri(file_uri="gs://example/image.webp", mime_type="image/webp"),
        types.Part.from_bytes(data=b"\x00\xff\x01", mime_type="image/heif"),
    ]


@pytest.mark.parametrize(
    "url",
    [
        "data:image/png;base64,iVBORw0KGgo=",
        "data:IMAGE/PNG;base64,iVBORw0KGgo=",
        "data:image/png;charset=utf-8;base64,iVBORw0KGgo=",
    ],
)
def test_data_uri_metadata(llm: GeminiCompletion, url: str) -> None:
    """Parse MIME metadata separately from the base64 data."""
    contents, _ = llm._format_messages_for_gemini(
        [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": url}}],
            }
        ]
    )
    assert contents[0].parts == [
        types.Part.from_bytes(data=b"\x89PNG\r\n\x1a\n", mime_type="image/png")
    ]


@pytest.mark.parametrize("encoded", ["+/8=", "%2B%2F8%3D", "+%2f8%3d"])
def test_percent_encoded_image_bytes(llm: GeminiCompletion, encoded: str) -> None:
    """URL escaping must not alter base64's plus, slash or padding characters."""
    contents, _ = llm._format_messages_for_gemini(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded}"},
                    }
                ],
            }
        ]
    )
    assert contents[0].parts == [
        types.Part.from_bytes(data=b"\xfb\xff", mime_type="image/png")
    ]


@pytest.mark.parametrize(
    "image_url",
    [
        None,
        "https://example.com/private.png",
        {},
        {"url": None},
        {"url": 42},
        {"url": []},
        {"url": "https://example.com/private.png?token=secret"},
        {"url": "file:///private/image.png"},
        {"url": "data:image/png,secret"},
        {"url": "data:image/png;base64"},
        {"url": "data:;base64,c2VjcmV0"},
        {"url": "data:text/plain;base64,c2VjcmV0"},
        {"url": "data:image/svg+xml;base64,c2VjcmV0"},
        {"url": "data:image/png;base64,!!!secret!!!"},
        {"url": "data:image/png;base64,a"},
        {"url": "data:image/png;base64,秘密"},
        {"url": "data:image/png;base64,"},
        {"url": "data:image/png;base64,c2VjcmV0\n"},
        {"url": "data:image/png;invalid;base64,c2VjcmV0"},
        {"url": "data:image/png;base64,c2VjcmV0%0A"},
        {"url": "data:image/png;base64,c2VjcmV0%ZZ"},
        {"url": "data:image/png;base64,c2VjcmV0%2"},
        {"url": "data:image/png;base64,%FF"},
        {"url": "data:image/png;base64,%252B%252F8%253D"},
    ],
)
def test_invalid_images_raise_without_payload_disclosure(
    llm: GeminiCompletion, image_url: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """Unsupported inputs must not silently become text-only requests."""
    messages: list[LLMMessage] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this."},
                {"type": "image_url", "image_url": image_url},
            ],
        }
    ]
    original = deepcopy(messages)
    with pytest.raises(ValueError, match="Gemini image_url") as exc:
        llm._format_messages_for_gemini(messages)
    assert messages == original
    for sensitive in ("secret", "c2VjcmV0", "private", "秘密"):
        assert sensitive not in str(exc.value) + caplog.text


def test_missing_image_url_raises(llm: GeminiCompletion) -> None:
    """An image declaration with no payload produces an actionable error."""
    with pytest.raises(ValueError, match="Gemini image_url"):
        llm._format_messages_for_gemini(
            [{"role": "user", "content": [{"type": "image_url"}]}]
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_sync_async_send_identical_image_bytes(
    llm: GeminiCompletion, stream: bool
) -> None:
    """Both public calls send the same multimodal payload to fake SDK clients."""
    response = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    role="model", parts=[types.Part.from_text(text="An image.")]
                )
            )
        ]
    )
    client = MagicMock()
    llm.stream = stream
    if stream:

        async def chunks() -> AsyncIterator[types.GenerateContentResponse]:
            yield response

        sync_generate = client.models.generate_content_stream
        sync_generate.return_value = iter([response])
        async_generate = AsyncMock(return_value=chunks())
        client.aio.models.generate_content_stream = async_generate
    else:
        sync_generate = client.models.generate_content
        sync_generate.return_value = response
        async_generate = AsyncMock(return_value=response)
        client.aio.models.generate_content = async_generate
    messages: list[LLMMessage] = [{"role": "user", "content": [image_part()]}]
    original = deepcopy(messages)
    with patch.object(GeminiCompletion, "_get_sync_client", return_value=client):
        assert llm.call(messages) == "An image."
        assert await llm.acall(messages) == "An image."
    sync_generate.assert_called_once()
    async_generate.assert_awaited_once()
    sync_contents = sync_generate.call_args.kwargs["contents"]
    async_contents = async_generate.call_args.kwargs["contents"]
    assert (
        sync_contents
        == async_contents
        == [
            types.Content(
                role="user",
                parts=[types.Part.from_bytes(data=PNG, mime_type="image/png")],
            )
        ]
    )
    assert messages == original


def test_step_executor_vision_observation_reaches_gemini(llm: GeminiCompletion) -> None:
    """The agent-generated observation retains its image through conversion."""
    observation = StepExecutor._build_observation_message(
        f"VISION_IMAGE:image/png:{base64.b64encode(PNG).decode('ascii')}"
    )
    contents, _ = llm._format_messages_for_gemini([observation])
    assert contents[0].parts == [
        types.Part.from_text(text="Observation: Here is the image:"),
        types.Part.from_bytes(data=PNG, mime_type="image/png"),
    ]


@pytest.mark.parametrize(
    "message",
    [
        {"role": "system", "content": [image_part()]},
        {"role": "tool", "tool_call_id": "call-1", "content": [image_part()]},
        {
            "role": "assistant",
            "tool_calls": [{"id": "call-1"}],
            "content": [image_part()],
        },
    ],
)
def test_images_in_text_only_turns_are_not_silently_dropped(
    llm: GeminiCompletion, message: LLMMessage
) -> None:
    """System instructions and function-call branches cannot carry image parts."""
    with pytest.raises(ValueError, match="Gemini image_url parts require"):
        llm._format_messages_for_gemini([message])


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/private.png?token=secret",
        "data:image/png;base64,!!!secret!!!",
    ],
)
async def test_invalid_images_abort_both_public_calls(
    llm: GeminiCompletion, url: str, stream: bool, caplog: pytest.LogCaptureFixture
) -> None:
    """Conversion errors reach callers before the SDK is called, without leaks."""
    llm.stream = stream
    messages: list[LLMMessage] = [
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]}
    ]
    with patch.object(GeminiCompletion, "_get_sync_client") as get_client:
        with pytest.raises(ValueError, match="Gemini image_url"):
            llm.call(messages)
        with pytest.raises(ValueError, match="Gemini image_url"):
            await llm.acall(messages)
        get_client.assert_not_called()
    assert "secret" not in caplog.text
    assert "private" not in caplog.text
