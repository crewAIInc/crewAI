"""A multimodal content-part list must reach the model as its text."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

from crewai import Agent
from crewai.llms.base_llm import BaseLLM


class _Recorder(BaseLLM):
    """Records the exact message list handed to the provider."""

    def __init__(self) -> None:
        super().__init__(model="recorder")
        object.__setattr__(self, "seen", [])

    def call(self, messages: Any, **kwargs: Any) -> str:
        self.seen.append(list(messages) if isinstance(messages, list) else messages)
        return "ok"

    async def acall(self, messages: Any, **kwargs: Any) -> str:
        return self.call(messages, **kwargs)

    def supports_function_calling(self) -> bool:
        return False

    def supports_stop_words(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 8192


PARTS = [
    {"type": "text", "text": "what is in this photo"},
    {"type": "image_url", "image_url": {"url": "https://example/p.png"}},
]

IMAGE_ONLY = [{"type": "image_url", "image_url": {"url": "https://example/p.png"}}]


def _agent(llm: _Recorder, **kwargs: Any) -> Agent:
    return Agent(role="Support", goal="help", backstory="b", llm=llm, **kwargs)


def _request_text(seen: list[dict[str, Any]]) -> str:
    return str([m for m in seen if m.get("role") == "user"][-1]["content"])


def test_the_promoted_request_carries_the_text_not_a_repr() -> None:
    """`str()` on the parts list put `[{'type': 'text', ...}]` in Current Task."""
    llm = _Recorder()
    _agent(llm).kickoff([{"role": "user", "content": PARTS}])

    text = _request_text(llm.seen[0])

    assert "what is in this photo" in text
    assert "'type'" not in text
    assert "image_url" not in text


def test_the_async_path_agrees_with_the_sync_one() -> None:
    """`kickoff_async` shares `_prepare_kickoff`; assert it, don't assume it.

    Driven with ``asyncio.run`` rather than an async test: ``kickoff`` takes a
    different path when called from inside a running loop, so the sync half
    has to run outside one for the comparison to mean anything.
    """
    sync_llm = _Recorder()
    _agent(sync_llm).kickoff([{"role": "user", "content": PARTS}])

    async_llm = _Recorder()
    asyncio.run(_agent(async_llm).kickoff_async([{"role": "user", "content": PARTS}]))

    assert _request_text(async_llm.seen[0]) == _request_text(sync_llm.seen[0])
    assert "what is in this photo" in _request_text(async_llm.seen[0])


def test_an_image_only_request_is_named_not_serialized() -> None:
    """No text block to promote, so the placeholder stands in for the repr."""
    llm = _Recorder()
    _agent(llm).kickoff([{"role": "user", "content": IMAGE_ONLY}])

    text = _request_text(llm.seen[0])

    assert "[multimodal content]" in text
    assert "example/p.png" not in text


def test_history_keeps_its_parts_list_untouched() -> None:
    """Only the promoted turn collapses to text; the provider still gets parts."""
    llm = _Recorder()
    _agent(llm).kickoff(
        [
            {"role": "user", "content": PARTS},
            {"role": "assistant", "content": "a cat"},
            {"role": "user", "content": "and this one?"},
        ]
    )
    history = [m for m in llm.seen[0] if m.get("role") == "user"][0]

    assert history["content"] == PARTS


def test_a_plain_string_request_is_unchanged() -> None:
    llm = _Recorder()
    _agent(llm).kickoff([{"role": "user", "content": "just text"}])

    assert "just text" in _request_text(llm.seen[0])


def test_memory_recall_and_save_get_text_not_a_repr() -> None:
    """Both memory paths flattened the same way and stored the repr."""
    llm = _Recorder()
    memory = MagicMock()
    memory.recall.return_value = []
    memory.extract_memories.return_value = ["m"]

    agent = _agent(llm)
    object.__setattr__(agent, "memory", memory)
    agent.kickoff([{"role": "user", "content": PARTS}])

    recalled = memory.recall.call_args[0][0]
    assert "what is in this photo" in recalled
    assert "'type'" not in recalled

    saved = memory.extract_memories.call_args[0][0]
    assert "what is in this photo" in saved
    assert "'type'" not in saved
