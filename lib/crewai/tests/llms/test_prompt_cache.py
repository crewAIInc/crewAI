"""Regression tests for the provider-agnostic prompt-cache breakpoint flag."""

from __future__ import annotations

from crewai.llms.cache import (
    CACHE_BREAKPOINT_KEY,
    mark_cache_breakpoint,
    strip_cache_breakpoint,
)
from crewai.llms.providers.anthropic.completion import AnthropicCompletion
from crewai.llms.providers.openai.completion import OpenAICompletion


class TestCacheMarkerHelpers:
    def test_mark_returns_new_dict(self) -> None:
        original = {"role": "user", "content": "hi"}
        marked = mark_cache_breakpoint(original)
        assert marked[CACHE_BREAKPOINT_KEY] is True
        # Marker must NOT bleed back into the caller's dict — callers may
        # pass literal dicts and reuse them across calls.
        assert CACHE_BREAKPOINT_KEY not in original

    def test_strip_is_idempotent(self) -> None:
        msg = {"role": "user", "content": "hi", CACHE_BREAKPOINT_KEY: True}
        strip_cache_breakpoint(msg)
        assert CACHE_BREAKPOINT_KEY not in msg
        strip_cache_breakpoint(msg)
        assert CACHE_BREAKPOINT_KEY not in msg


class TestBaseFormatDoesNotMutate:
    """The strip-on-format pass must not erase markers from the caller's
    messages list — executors reuse a single list across many LLM calls,
    and mutating it would defeat caching on every iteration after the first.
    """

    def test_repeated_format_preserves_markers(self) -> None:
        llm = OpenAICompletion(model="gpt-4o-mini")
        messages = [
            mark_cache_breakpoint({"role": "system", "content": "stable system"}),
            mark_cache_breakpoint({"role": "user", "content": "stable user"}),
        ]
        # First call: provider strips markers from the returned (copied) list
        first = llm._format_messages(messages)
        assert all(CACHE_BREAKPOINT_KEY not in m for m in first)
        # Original list must STILL carry the markers
        assert messages[0][CACHE_BREAKPOINT_KEY] is True
        assert messages[1][CACHE_BREAKPOINT_KEY] is True
        # Second call from the same list still sees the markers
        second = llm._format_messages(messages)
        assert all(CACHE_BREAKPOINT_KEY not in m for m in second)
        assert messages[0][CACHE_BREAKPOINT_KEY] is True
        assert messages[1][CACHE_BREAKPOINT_KEY] is True


class TestAnthropicCacheStamping:
    def test_stamps_system_with_cache_control(self) -> None:
        llm = AnthropicCompletion(model="claude-sonnet-4-5")
        messages = [
            mark_cache_breakpoint({"role": "system", "content": "you are helpful"}),
            mark_cache_breakpoint({"role": "user", "content": "ping"}),
        ]
        formatted, system = llm._format_messages_for_anthropic(messages)
        assert isinstance(system, list)
        assert system[0]["cache_control"] == {"type": "ephemeral"}
        assert system[0]["text"] == "you are helpful"
        # First user block carries cache_control too
        last_block = formatted[0]["content"][-1]
        assert last_block["cache_control"] == {"type": "ephemeral"}

    def test_stamps_stable_user_not_tool_result(self) -> None:
        """Within a ReAct loop, tool results are flattened into a trailing
        user message. We must NOT stamp that volatile trailing block — we
        must stamp the original stable user prompt instead.
        """
        llm = AnthropicCompletion(model="claude-sonnet-4-5")
        messages = [
            mark_cache_breakpoint({"role": "system", "content": "you are helpful"}),
            mark_cache_breakpoint({"role": "user", "content": "stable task prompt"}),
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "tc_1",
                        "function": {"name": "ping", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "tc_1", "content": "volatile tool result"},
        ]
        formatted, _system = llm._format_messages_for_anthropic(messages)
        # Find the message that holds the stable prompt
        stable = next(
            fm
            for fm in formatted
            if fm["role"] == "user"
            and isinstance(fm["content"], list)
            and any(
                isinstance(b, dict)
                and b.get("type") == "text"
                and b.get("text") == "stable task prompt"
                for b in fm["content"]
            )
        )
        text_block = next(
            b for b in stable["content"] if isinstance(b, dict) and b.get("type") == "text"
        )
        assert text_block.get("cache_control") == {"type": "ephemeral"}
        # The tool_result-bearing user message must NOT be stamped
        tool_carrier = next(
            fm
            for fm in formatted
            if fm["role"] == "user"
            and isinstance(fm["content"], list)
            and any(
                isinstance(b, dict) and b.get("type") == "tool_result"
                for b in fm["content"]
            )
        )
        for block in tool_carrier["content"]:
            assert "cache_control" not in block

    def test_assistant_marker_is_ignored(self) -> None:
        """Markers on assistant messages have no stable stamp target after
        Anthropic's role coalescing, so they should be silently ignored
        rather than collected and then dropped on a mismatch.
        """
        llm = AnthropicCompletion(model="claude-sonnet-4-5")
        messages = [
            mark_cache_breakpoint({"role": "system", "content": "you are helpful"}),
            mark_cache_breakpoint(
                {"role": "assistant", "content": "I will help you out."}
            ),
            {"role": "user", "content": "ping"},
        ]
        formatted, system = llm._format_messages_for_anthropic(messages)
        # System still cached
        assert isinstance(system, list)
        # No user message was marked → no user message should carry cache_control
        for fm in formatted:
            if fm.get("role") != "user":
                continue
            content = fm.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        assert "cache_control" not in block

    def test_list_content_user_marker_matches(self) -> None:
        """A pre-formatted user message with a single text block should still
        match against the post-format user message.
        """
        llm = AnthropicCompletion(model="claude-sonnet-4-5")
        messages = [
            mark_cache_breakpoint(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "stable list prompt"}],
                }
            ),
        ]
        formatted, _system = llm._format_messages_for_anthropic(messages)
        user_msg = next(fm for fm in formatted if fm["role"] == "user")
        content = user_msg["content"]
        assert isinstance(content, list)
        text_block = next(b for b in content if isinstance(b, dict) and b.get("type") == "text")
        assert text_block.get("cache_control") == {"type": "ephemeral"}

    def test_unmarked_messages_get_no_cache_control(self) -> None:
        llm = AnthropicCompletion(model="claude-sonnet-4-5")
        messages = [
            {"role": "system", "content": "no caching here"},
            {"role": "user", "content": "no caching here either"},
        ]
        formatted, system = llm._format_messages_for_anthropic(messages)
        # No marker → system stays a plain string (no content-block conversion)
        assert isinstance(system, str)
        # No marker → no cache_control anywhere in formatted messages
        for fm in formatted:
            content = fm.get("content")
            if isinstance(content, list):
                for block in content:
                    assert "cache_control" not in block


class TestNonAnthropicStripsMarker:
    def test_openai_format_strips_marker_from_wire_payload(self) -> None:
        llm = OpenAICompletion(model="gpt-4o-mini")
        messages = [
            mark_cache_breakpoint({"role": "system", "content": "stable"}),
            mark_cache_breakpoint({"role": "user", "content": "hi"}),
        ]
        formatted = llm._format_messages(messages)
        for m in formatted:
            assert CACHE_BREAKPOINT_KEY not in m
