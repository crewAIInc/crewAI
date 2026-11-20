"""Second slice of the provider contract: what ``LLMMessage.content`` may hold.

``LLMMessage.content`` is ``str | list[dict] | None``, and ``AGENTS.md`` is explicit
that it must never be ``str()``-ed -- that puts a Python repr in front of the model.
These tests pin the three shapes at the SDK boundary, for OpenAI and Anthropic:

* a string stays a string,
* multimodal parts keep their text and are translated to the provider's own block
  format (OpenAI keeps ``image_url``, Anthropic rewrites it to a base64 ``image``
  block), never collapsed into a repr,
* ``None`` never becomes the string ``"None"``.

Where the providers legitimately differ -- OpenAI forwards ``None`` as ``null``,
Anthropic sends an empty string because its API has no null content -- the difference
is asserted rather than papered over, so a change on either side is visible.

Provider-specific suites keep owning everything else (tool calls, streaming, usage).
No network call is made.
"""

from __future__ import annotations

from typing import Any, Callable
from unittest.mock import MagicMock, patch

import httpx
import pytest
from anthropic.types import Message, TextBlock, Usage
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage

from crewai.llms.providers.anthropic.completion import AnthropicCompletion
from crewai.llms.providers.openai.completion import OpenAICompletion
from crewai.utilities.message_utils import message_content_text

OPENAI_MODEL = "gpt-4o"
ANTHROPIC_MODEL = "claude-sonnet-4-6"

PARTS: list[dict[str, Any]] = [
    {"type": "text", "text": "what is in this picture?"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
]


def _openai_raw_response(text: str = "ok") -> Any:
    """The ``with_raw_response`` wrapper the OpenAI provider parses."""
    completion = ChatCompletion(
        id="chatcmpl-content",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(role="assistant", content=text),
            )
        ],
        created=0,
        model=OPENAI_MODEL,
        object="chat.completion",
        usage=CompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )
    wrapper = MagicMock()
    wrapper.text = ""
    wrapper.http_response = httpx.Response(
        200, request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    )
    wrapper.parse.return_value = completion
    return wrapper


def _anthropic_message(text: str = "ok") -> Message:
    return Message(
        id="msg_content",
        content=[TextBlock(type="text", text=text)],
        model=ANTHROPIC_MODEL,
        role="assistant",
        stop_reason="end_turn",
        stop_sequence=None,
        type="message",
        usage=Usage(input_tokens=1, output_tokens=1),
    )


def _send_openai(content: Any) -> list[dict[str, Any]]:
    llm = OpenAICompletion(model=OPENAI_MODEL, api_key="placeholder-key")
    fake_client = MagicMock()
    fake_client.chat.completions.with_raw_response.create.return_value = (
        _openai_raw_response()
    )
    with patch.object(OpenAICompletion, "_get_sync_client", return_value=fake_client):
        llm.call([{"role": "user", "content": content}])
    return fake_client.chat.completions.with_raw_response.create.call_args.kwargs["messages"]


def _send_anthropic(content: Any) -> list[dict[str, Any]]:
    llm = AnthropicCompletion(model=ANTHROPIC_MODEL, api_key="placeholder-key")
    fake_client = MagicMock()
    fake_client.messages.create.return_value = _anthropic_message()
    with patch.object(AnthropicCompletion, "_get_sync_client", return_value=fake_client):
        llm.call([{"role": "user", "content": content}])
    return fake_client.messages.create.call_args.kwargs["messages"]


Senders = Callable[[Any], list[dict[str, Any]]]

SENDERS: list[Any] = [
    pytest.param(_send_openai, id="openai"),
    pytest.param(_send_anthropic, id="anthropic"),
]


@pytest.mark.parametrize("send", SENDERS)
def test_string_content_reaches_the_sdk_unchanged(send: Senders) -> None:
    """A string stays the same string; nothing is appended or re-encoded."""
    sent = send("plain text")
    assert sent[-1]["content"] == "plain text"


@pytest.mark.parametrize("send", SENDERS)
def test_multimodal_parts_stay_blocks_and_keep_their_text(send: Senders) -> None:
    """Parts reach the SDK as a list of blocks, and the text part survives."""
    sent = send(PARTS)
    content = sent[-1]["content"]

    assert isinstance(content, list), "multimodal content must stay a list of blocks"
    text_blocks = [block for block in content if block.get("type") == "text"]
    assert text_blocks, "the text part must survive the translation"
    assert text_blocks[0]["text"] == "what is in this picture?"
    # The failure mode AGENTS.md warns about: the whole list rendered as one string.
    assert content != str(PARTS)


def test_none_is_forwarded_as_null_by_openai_and_as_empty_by_anthropic() -> None:
    """The two providers legitimately differ; neither sends the string ``"None"``."""
    openai_sent = _send_openai(None)
    anthropic_sent = _send_anthropic(None)

    assert openai_sent[-1]["content"] is None
    assert anthropic_sent[-1]["content"] == ""

    for sent in (openai_sent, anthropic_sent):
        assert sent[-1]["content"] != "None"


def test_image_blocks_use_each_providers_own_shape() -> None:
    """OpenAI keeps ``image_url``; Anthropic has no such block and gets base64."""
    openai_content = _send_openai(PARTS)[-1]["content"]
    anthropic_content = _send_anthropic(PARTS)[-1]["content"]

    assert openai_content[1]["type"] == "image_url"
    assert anthropic_content[1]["type"] == "image"
    assert anthropic_content[1]["source"] == {
        "type": "base64",
        "media_type": "image/png",
        "data": "iVBORw0KGgo=",
    }


def test_message_content_text_handles_every_shape() -> None:
    """The text helper is used only where text is actually needed."""
    assert message_content_text({"role": "user", "content": "hi"}) == "hi"
    assert message_content_text({"role": "user", "content": None}) == ""
    assert (
        message_content_text({"role": "user", "content": PARTS})
        == "what is in this picture?"
    )
    assert (
        message_content_text(
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": "x"}}],
            }
        )
        == "[multimodal content]"
    )
    assert message_content_text({"role": "user", "content": PARTS}) != str(PARTS)
