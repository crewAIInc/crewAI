"""Shared Azure provider test fixtures."""

from collections.abc import Callable
from typing import Any

import pytest
from azure.ai.inference.models import (
    CompletionsUsage,
    StreamingChatChoiceUpdate,
    StreamingChatCompletionsUpdate,
    StreamingChatResponseMessageUpdate,
)


class _AsyncStream:
    def __init__(self, updates: list[StreamingChatCompletionsUpdate]) -> None:
        self._updates = iter(updates)

    def __aiter__(self) -> "_AsyncStream":
        return self

    async def __anext__(self) -> StreamingChatCompletionsUpdate:
        try:
            return next(self._updates)
        except StopIteration:
            raise StopAsyncIteration from None


class _AsyncStreamingClient:
    def __init__(self, content: str) -> None:
        self._content = content

    async def complete(self, **_: Any) -> _AsyncStream:
        return _AsyncStream(
            [
                StreamingChatCompletionsUpdate(
                    id="chatcmpl-test",
                    created=1,
                    model="gpt-4o-mini",
                    choices=[
                        StreamingChatChoiceUpdate(
                            index=0,
                            finish_reason=None,
                            delta=StreamingChatResponseMessageUpdate(
                                role="assistant",
                                content=self._content,
                                tool_calls=None,
                            ),
                        )
                    ],
                    usage=None,
                ),
                StreamingChatCompletionsUpdate(
                    id="chatcmpl-test",
                    created=1,
                    model="gpt-4o-mini",
                    choices=[
                        StreamingChatChoiceUpdate(
                            index=0,
                            finish_reason="stop",
                            delta=StreamingChatResponseMessageUpdate(
                                role=None,
                                content=None,
                                tool_calls=None,
                            ),
                        )
                    ],
                    usage=CompletionsUsage(
                        prompt_tokens=10,
                        completion_tokens=5,
                        total_tokens=15,
                    ),
                ),
            ]
        )


@pytest.fixture
def mock_azure_async_streaming_client() -> Callable[[Any, str], None]:
    """Install an offline async streaming client on an Azure completion."""

    def install(llm: Any, content: str) -> None:
        llm._async_client = _AsyncStreamingClient(content)

    return install
