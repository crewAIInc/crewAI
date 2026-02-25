"""Tests for the double-buffered context window manager.

Reference: https://marklubin.me/posts/hopping-context-windows/
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from crewai.memory.context_buffer import (
    ContextBufferConfig,
    DoubleBufferContextManager,
    RenewalPolicy,
)
from crewai.utilities.agent_utils import _estimate_token_count


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_llm() -> MagicMock:
    """Create a mock LLM that returns canned summaries."""
    mock_llm = MagicMock()
    mock_llm.get_context_window_size.return_value = 1000
    mock_llm.call.return_value = "<summary>Summarized conversation.</summary>"
    return mock_llm


def _make_mock_i18n() -> MagicMock:
    """Create a mock i18n matching the patterns in test_agent_utils.py."""
    mock_i18n = MagicMock()
    mock_i18n.slice.side_effect = lambda key: {
        "summarizer_system_message": "You are a precise assistant that creates structured summaries.",
        "summarize_instruction": "Summarize the conversation:\n{conversation}",
        "summary": "<summary>\n{merged_summary}\n</summary>\nContinue the task.",
    }.get(key, "")
    return mock_i18n


def _make_message(content: str, role: str = "user") -> dict[str, Any]:
    """Build a minimal LLM message dict."""
    return {"role": role, "content": content}


def _make_system_message(content: str = "You are a helpful assistant.") -> dict[str, Any]:
    return {"role": "system", "content": content}


def _fake_summarize(
    messages: list[dict[str, Any]],
    llm: Any = None,
    callbacks: Any = None,
    i18n: Any = None,
    verbose: bool = True,
) -> None:
    """Fake summarize_messages that compresses in-place like the real one.

    Preserves system messages and replaces everything else with a short summary.
    This avoids hitting the real CrewAI event bus or LLM.
    """
    system_msgs = [m for m in messages if m.get("role") == "system"]
    non_system = [m for m in messages if m.get("role") != "system"]
    if not non_system:
        return
    first_content = str(non_system[0].get("content", ""))[:50]
    summary_text = f"Summary of {len(non_system)} messages: {first_content}..."
    messages.clear()
    messages.extend(system_msgs)
    messages.append({"role": "user", "content": summary_text})


# Patch path for summarize_messages as imported in context_buffer module
_SUMMARIZE_PATCH = "crewai.memory.context_buffer.summarize_messages"


# ---------------------------------------------------------------------------
# ContextBufferConfig tests
# ---------------------------------------------------------------------------

class TestContextBufferConfig:
    """Tests for ContextBufferConfig defaults and validation."""

    def test_default_values(self) -> None:
        cfg = ContextBufferConfig()
        assert cfg.checkpoint_threshold == 0.70
        assert cfg.swap_threshold == 0.95
        assert cfg.max_generations is None
        assert cfg.renewal_policy == RenewalPolicy.RECURSE

    def test_custom_values(self) -> None:
        cfg = ContextBufferConfig(
            checkpoint_threshold=0.5,
            swap_threshold=0.8,
            max_generations=3,
            renewal_policy=RenewalPolicy.DUMP,
        )
        assert cfg.checkpoint_threshold == 0.5
        assert cfg.swap_threshold == 0.8
        assert cfg.max_generations == 3
        assert cfg.renewal_policy == RenewalPolicy.DUMP

    def test_threshold_boundaries(self) -> None:
        """Thresholds must be between 0 and 1."""
        with pytest.raises(Exception):
            ContextBufferConfig(checkpoint_threshold=1.5)
        with pytest.raises(Exception):
            ContextBufferConfig(swap_threshold=-0.1)

    def test_max_generations_none_means_no_limit(self) -> None:
        cfg = ContextBufferConfig(max_generations=None)
        assert cfg.max_generations is None


# ---------------------------------------------------------------------------
# Constructor / validation tests
# ---------------------------------------------------------------------------

class TestDoubleBufferContextManagerInit:
    """Tests for DoubleBufferContextManager construction."""

    def test_initial_state(self) -> None:
        mgr = DoubleBufferContextManager(
            context_window_size=1000,
            llm=_make_mock_llm(),
            i18n=_make_mock_i18n(),
        )
        assert not mgr.has_back_buffer
        assert mgr.generation == 0
        assert mgr.active_buffer == []
        assert mgr.back_buffer == []
        assert mgr.accumulated_summaries == []

    def test_checkpoint_must_be_less_than_swap(self) -> None:
        cfg = ContextBufferConfig(checkpoint_threshold=0.95, swap_threshold=0.70)
        with pytest.raises(ValueError, match="checkpoint_threshold.*must be less than swap_threshold"):
            DoubleBufferContextManager(
                context_window_size=1000,
                llm=_make_mock_llm(),
                i18n=_make_mock_i18n(),
                config=cfg,
            )

    def test_equal_thresholds_rejected(self) -> None:
        cfg = ContextBufferConfig(checkpoint_threshold=0.80, swap_threshold=0.80)
        with pytest.raises(ValueError):
            DoubleBufferContextManager(
                context_window_size=1000,
                llm=_make_mock_llm(),
                i18n=_make_mock_i18n(),
                config=cfg,
            )

    def test_custom_config(self) -> None:
        cfg = ContextBufferConfig(
            checkpoint_threshold=0.5,
            swap_threshold=0.9,
            max_generations=3,
            renewal_policy=RenewalPolicy.DUMP,
        )
        mgr = DoubleBufferContextManager(
            context_window_size=2000,
            llm=_make_mock_llm(),
            i18n=_make_mock_i18n(),
            config=cfg,
        )
        assert mgr.config.checkpoint_threshold == 0.5
        assert mgr.config.max_generations == 3


# ---------------------------------------------------------------------------
# Token estimation tests
# ---------------------------------------------------------------------------

class TestTokenEstimation:
    """Tests for buffer token estimation."""

    def test_empty_buffer(self) -> None:
        mgr = DoubleBufferContextManager(
            context_window_size=1000,
            llm=_make_mock_llm(),
            i18n=_make_mock_i18n(),
        )
        assert mgr.estimate_buffer_tokens([]) == 0

    def test_single_message(self) -> None:
        mgr = DoubleBufferContextManager(
            context_window_size=1000,
            llm=_make_mock_llm(),
            i18n=_make_mock_i18n(),
        )
        # 100 chars -> 25 tokens
        msgs: list[dict[str, Any]] = [_make_message("a" * 100)]
        assert mgr.estimate_buffer_tokens(msgs) == 25

    def test_none_content_skipped(self) -> None:
        mgr = DoubleBufferContextManager(
            context_window_size=1000,
            llm=_make_mock_llm(),
            i18n=_make_mock_i18n(),
        )
        msgs: list[dict[str, Any]] = [{"role": "assistant", "content": None}]
        assert mgr.estimate_buffer_tokens(msgs) == 0

    def test_multiple_messages(self) -> None:
        mgr = DoubleBufferContextManager(
            context_window_size=1000,
            llm=_make_mock_llm(),
            i18n=_make_mock_i18n(),
        )
        msgs: list[dict[str, Any]] = [
            _make_message("a" * 40),   # 10 tokens
            _make_message("b" * 80),   # 20 tokens
        ]
        assert mgr.estimate_buffer_tokens(msgs) == 30

    def test_uses_crewai_estimate(self) -> None:
        """Verify we use the same heuristic as CrewAI's _estimate_token_count."""
        text = "The quick brown fox"
        expected = _estimate_token_count(text)
        mgr = DoubleBufferContextManager(
            context_window_size=1000,
            llm=_make_mock_llm(),
            i18n=_make_mock_i18n(),
        )
        assert mgr.estimate_buffer_tokens([_make_message(text)]) == expected


# ---------------------------------------------------------------------------
# Append behavior tests
# ---------------------------------------------------------------------------

class TestAppend:
    """Tests for message appending."""

    def test_append_adds_to_active_buffer(self) -> None:
        mgr = DoubleBufferContextManager(
            context_window_size=100_000,
            llm=_make_mock_llm(),
            i18n=_make_mock_i18n(),
        )
        msg = _make_message("hello")
        mgr.append(msg)
        assert len(mgr.active_buffer) == 1
        assert mgr.active_buffer[0]["content"] == "hello"

    def test_append_does_not_dual_write_in_idle(self) -> None:
        mgr = DoubleBufferContextManager(
            context_window_size=100_000,
            llm=_make_mock_llm(),
            i18n=_make_mock_i18n(),
        )
        mgr.append(_make_message("hello"))
        assert mgr.back_buffer == []

    @patch(_SUMMARIZE_PATCH, side_effect=_fake_summarize)
    def test_append_dual_writes_after_checkpoint(self, _mock_sum: Any) -> None:
        """After checkpoint, messages go to both buffers.

        Use context_window=1000 with checkpoint at 10% (100 tokens) so a
        single 400-char message (100 tokens) triggers checkpoint, but swap
        at 95% (950 tokens) is far away.
        """
        config = ContextBufferConfig(checkpoint_threshold=0.10, swap_threshold=0.95)
        mgr = DoubleBufferContextManager(
            context_window_size=1000,
            llm=_make_mock_llm(),
            i18n=_make_mock_i18n(),
            config=config,
        )
        # 400 chars = 100 tokens = 10% of 1000 -- triggers checkpoint
        mgr.append(_make_message("x" * 400))
        assert mgr.has_back_buffer

        back_len_before = len(mgr.back_buffer)
        mgr.append(_make_message("dual write"))
        # The new message should appear in the back buffer
        assert len(mgr.back_buffer) == back_len_before + 1

    @patch(_SUMMARIZE_PATCH, side_effect=_fake_summarize)
    def test_dual_append_while_back_buffer_exists(self, _mock_sum: Any) -> None:
        """After checkpoint, appending should dual-write to both buffers."""
        config = ContextBufferConfig(checkpoint_threshold=0.10, swap_threshold=0.95)
        mgr = DoubleBufferContextManager(
            context_window_size=1000,
            llm=_make_mock_llm(),
            i18n=_make_mock_i18n(),
            config=config,
        )
        mgr.append(_make_message("x" * 400))  # triggers checkpoint
        assert mgr.has_back_buffer
        back_len = len(mgr.back_buffer)
        mgr.append(_make_message("second"))     # dual-append
        assert mgr.has_back_buffer
        assert len(mgr.back_buffer) == back_len + 1


# ---------------------------------------------------------------------------
# Checkpoint phase tests
# ---------------------------------------------------------------------------

class TestCheckpoint:
    """Tests for the checkpoint phase."""

    @patch(_SUMMARIZE_PATCH, side_effect=_fake_summarize)
    def test_checkpoint_triggers_at_threshold(self, _mock_sum: Any) -> None:
        """Checkpoint should trigger when usage crosses checkpoint_threshold."""
        config = ContextBufferConfig(
            checkpoint_threshold=0.70,
            swap_threshold=0.95,
        )
        mgr = DoubleBufferContextManager(
            context_window_size=1000,
            llm=_make_mock_llm(),
            i18n=_make_mock_i18n(),
            config=config,
        )
        assert not mgr.has_back_buffer

        # Each message is 400 chars = 100 tokens. 6 msgs = 600 tokens = 60%.
        for _ in range(6):
            mgr.append(_make_message("x" * 400))
        assert not mgr.has_back_buffer

        # Push over 70%
        mgr.append(_make_message("x" * 400))  # 700 tokens
        assert mgr.has_back_buffer
        _mock_sum.assert_called()

    @patch(_SUMMARIZE_PATCH, side_effect=_fake_summarize)
    def test_checkpoint_seeds_back_buffer(self, _mock_sum: Any) -> None:
        """After checkpoint, back buffer should contain summarized content."""
        config = ContextBufferConfig(checkpoint_threshold=0.10, swap_threshold=0.95)
        mgr = DoubleBufferContextManager(
            context_window_size=1000,
            llm=_make_mock_llm(),
            i18n=_make_mock_i18n(),
            config=config,
        )
        mgr.append(_make_message("x" * 400))  # 100 tokens = 10%
        assert len(mgr.back_buffer) > 0

    @patch(_SUMMARIZE_PATCH, side_effect=_fake_summarize)
    def test_checkpoint_accumulates_summary(self, _mock_sum: Any) -> None:
        """Checkpoint should add to accumulated_summaries."""
        config = ContextBufferConfig(checkpoint_threshold=0.10, swap_threshold=0.95)
        mgr = DoubleBufferContextManager(
            context_window_size=1000,
            llm=_make_mock_llm(),
            i18n=_make_mock_i18n(),
            config=config,
        )
        mgr.append(_make_message("x" * 400))
        assert len(mgr.accumulated_summaries) >= 1

    @patch(_SUMMARIZE_PATCH, side_effect=_fake_summarize)
    def test_checkpoint_does_not_modify_active_buffer(self, _mock_sum: Any) -> None:
        """The active buffer should not be summarized during checkpoint --
        only the back buffer gets the summary."""
        config = ContextBufferConfig(checkpoint_threshold=0.10, swap_threshold=0.95)
        mgr = DoubleBufferContextManager(
            context_window_size=1000,
            llm=_make_mock_llm(),
            i18n=_make_mock_i18n(),
            config=config,
        )
        msg = _make_message("important content " + "x" * 400)
        mgr.append(msg)
        # The active buffer should still contain the original message
        contents = [m.get("content") for m in mgr.active_buffer]
        assert any("important content" in str(c) for c in contents)


# ---------------------------------------------------------------------------
# Swap phase tests
# ---------------------------------------------------------------------------

class TestSwap:
    """Tests for the swap phase."""

    @patch(_SUMMARIZE_PATCH, side_effect=_fake_summarize)
    def test_swap_triggers_and_increments_generation(self, _mock_sum: Any) -> None:
        """On swap, generation increments and back buffer clears."""
        config = ContextBufferConfig(
            checkpoint_threshold=0.50,
            swap_threshold=0.90,
        )
        mgr = DoubleBufferContextManager(
            context_window_size=1000,
            llm=_make_mock_llm(),
            i18n=_make_mock_i18n(),
            config=config,
        )

        for _ in range(10):
            mgr.append(_make_message("x" * 400))

        assert mgr.generation >= 1

    def test_swap_directly(self) -> None:
        """Test _swap directly with pre-set state."""
        mgr = DoubleBufferContextManager(
            context_window_size=1000,
            llm=_make_mock_llm(),
            i18n=_make_mock_i18n(),
            config=ContextBufferConfig(
                checkpoint_threshold=0.50,
                swap_threshold=0.90,
                max_generations=10,
            ),
        )
        mgr._active_buffer = [
            _make_message("x" * 400),
            _make_message("x" * 400),
            _make_message("x" * 400),
        ]
        mgr._back_buffer = [
            _make_message("summary of earlier conversation"),
            _make_message("x" * 400),
        ]
        mgr._generation = 0

        mgr._swap()

        assert mgr.generation == 1
        assert not mgr.has_back_buffer
        assert mgr.back_buffer == []
        assert len(mgr.active_buffer) == 2
        assert mgr.active_buffer[0]["content"] == "summary of earlier conversation"

    def test_swap_preserves_back_buffer_contents(self) -> None:
        """After swap, active buffer should equal what was in back buffer."""
        mgr = DoubleBufferContextManager(
            context_window_size=1000,
            llm=_make_mock_llm(),
            i18n=_make_mock_i18n(),
            config=ContextBufferConfig(
                checkpoint_threshold=0.50,
                swap_threshold=0.90,
                max_generations=10,
            ),
        )
        back_msgs = [
            _make_message("summary"),
            _make_message("post-checkpoint-1"),
            _make_message("post-checkpoint-2"),
        ]
        mgr._active_buffer = [_make_message("x" * 4000)]
        mgr._back_buffer = list(back_msgs)

        mgr._swap()

        assert mgr.active_buffer == back_msgs


# ---------------------------------------------------------------------------
# Renewal policy tests
# ---------------------------------------------------------------------------

class TestRenewalRecurse:
    """Tests for the RECURSE renewal policy (meta-summarize)."""

    @patch(_SUMMARIZE_PATCH, side_effect=_fake_summarize)
    def test_meta_summarize_triggers_at_max_generations(self, _mock_sum: Any) -> None:
        """Renewal should trigger when generation reaches max_generations."""
        config = ContextBufferConfig(
            checkpoint_threshold=0.50,
            swap_threshold=0.90,
            max_generations=2,
            renewal_policy=RenewalPolicy.RECURSE,
        )
        mgr = DoubleBufferContextManager(
            context_window_size=1000,
            llm=_make_mock_llm(),
            i18n=_make_mock_i18n(),
            config=config,
        )

        for _ in range(30):
            mgr.append(_make_message("x" * 400))

        # After enough swaps, renewal resets generation below max_generations
        assert mgr.generation < config.max_generations

    @patch(_SUMMARIZE_PATCH, side_effect=_fake_summarize)
    def test_meta_summarize_resets_generation(self, _mock_sum: Any) -> None:
        """After meta-summarize, generation should reset to 0."""
        mgr = DoubleBufferContextManager(
            context_window_size=1000,
            llm=_make_mock_llm(),
            i18n=_make_mock_i18n(),
            config=ContextBufferConfig(
                checkpoint_threshold=0.50,
                swap_threshold=0.90,
                max_generations=1,
                renewal_policy=RenewalPolicy.RECURSE,
            ),
        )

        for _ in range(15):
            mgr.append(_make_message("x" * 400))

        # max_generations=1: renewal fires on every swap, resetting to 0
        assert mgr.generation == 0

    def test_meta_summarize_directly(self) -> None:
        """Test _meta_summarize directly with pre-set state."""
        mgr = DoubleBufferContextManager(
            context_window_size=1000,
            llm=_make_mock_llm(),
            i18n=_make_mock_i18n(),
        )
        mgr._accumulated_summaries = ["Summary gen 1", "Summary gen 2", "Summary gen 3"]
        mgr._active_buffer = [
            _make_system_message(),
            _make_message("current data"),
        ]
        mgr._generation = 5

        with patch(_SUMMARIZE_PATCH, side_effect=_fake_summarize):
            mgr._meta_summarize()

        assert mgr.generation == 0
        assert not mgr.has_back_buffer
        assert len(mgr.accumulated_summaries) == 1

    def test_meta_summarize_condenses_summaries_directly(self) -> None:
        """Test that _meta_summarize condenses accumulated summaries to 1."""
        mgr = DoubleBufferContextManager(
            context_window_size=1000,
            llm=_make_mock_llm(),
            i18n=_make_mock_i18n(),
        )
        mgr._accumulated_summaries = ["Gen 1 summary", "Gen 2 summary", "Gen 3 summary"]
        mgr._active_buffer = [_make_system_message(), _make_message("current")]
        mgr._generation = 3

        with patch(_SUMMARIZE_PATCH, side_effect=_fake_summarize):
            mgr._meta_summarize()

        assert len(mgr.accumulated_summaries) == 1
        assert mgr.generation == 0


class TestRenewalDump:
    """Tests for the DUMP renewal policy (clean restart)."""

    @patch(_SUMMARIZE_PATCH, side_effect=_fake_summarize)
    def test_dump_resets_on_max_generations(self, _mock_sum: Any) -> None:
        """DUMP should reset generation when max_generations is reached."""
        config = ContextBufferConfig(
            checkpoint_threshold=0.50,
            swap_threshold=0.90,
            max_generations=1,
            renewal_policy=RenewalPolicy.DUMP,
        )
        mgr = DoubleBufferContextManager(
            context_window_size=1000,
            llm=_make_mock_llm(),
            i18n=_make_mock_i18n(),
            config=config,
        )

        for _ in range(15):
            mgr.append(_make_message("x" * 400))

        # After dump, generation resets
        assert mgr.generation == 0

    def test_dump_directly(self) -> None:
        """Test _dump directly with pre-set state."""
        mgr = DoubleBufferContextManager(
            context_window_size=1000,
            llm=_make_mock_llm(),
            i18n=_make_mock_i18n(),
        )
        mgr._accumulated_summaries = ["Summary 1", "Summary 2"]
        mgr._active_buffer = [
            _make_system_message("System prompt A"),
            _make_system_message("System prompt B"),
            _make_message("user data 1"),
            _make_message("user data 2"),
        ]
        mgr._back_buffer = [_make_message("stale")]
        mgr._generation = 5

        mgr._dump()

        assert mgr.generation == 0
        assert not mgr.has_back_buffer
        assert mgr.back_buffer == []
        assert mgr.accumulated_summaries == []
        # Only system messages survive
        assert len(mgr.active_buffer) == 2
        assert all(m["role"] == "system" for m in mgr.active_buffer)

    def test_dump_preserves_system_messages(self) -> None:
        """DUMP should preserve system messages in the active buffer."""
        mgr = DoubleBufferContextManager(
            context_window_size=1000,
            llm=_make_mock_llm(),
            i18n=_make_mock_i18n(),
        )
        mgr._active_buffer = [
            _make_system_message("System A"),
            _make_system_message("System B"),
            _make_message("data 1"),
            _make_message("data 2"),
            _make_message("data 3"),
        ]
        mgr._back_buffer = [_make_message("back")]
        mgr._accumulated_summaries = ["old summary"]
        mgr._generation = 3

        mgr._dump()

        assert mgr.generation == 0
        assert not mgr.has_back_buffer
        assert len(mgr.active_buffer) == 2
        assert mgr.active_buffer[0]["content"] == "System A"
        assert mgr.active_buffer[1]["content"] == "System B"
        assert mgr.back_buffer == []
        assert mgr.accumulated_summaries == []


# ---------------------------------------------------------------------------
# Full lifecycle tests
# ---------------------------------------------------------------------------

class TestFullLifecycle:
    """Integration-style tests for the full checkpoint -> concurrent -> swap cycle."""

    @patch(_SUMMARIZE_PATCH, side_effect=_fake_summarize)
    def test_idle_to_checkpoint_to_concurrent_to_swap(self, _mock_sum: Any) -> None:
        """Walk through all states in order."""
        config = ContextBufferConfig(
            checkpoint_threshold=0.50,
            swap_threshold=0.90,
        )
        mgr = DoubleBufferContextManager(
            context_window_size=1000,
            llm=_make_mock_llm(),
            i18n=_make_mock_i18n(),
            config=config,
        )

        # Phase 0: No back buffer yet
        assert not mgr.has_back_buffer
        for _ in range(4):  # 400 tokens = 40%
            mgr.append(_make_message("x" * 400))
        assert not mgr.has_back_buffer

        # Phase 1: Trigger checkpoint (>= 50%)
        mgr.append(_make_message("x" * 400))  # 500 tokens
        assert mgr.has_back_buffer

        # Phase 2: Dual-append (back buffer exists)
        mgr.append(_make_message("concurrent msg"))
        assert mgr.has_back_buffer

        # Phase 3: Fill to swap threshold
        while mgr.has_back_buffer:
            mgr.append(_make_message("x" * 400))

        # After swap, generation should have incremented
        assert mgr.generation >= 1

    @patch(_SUMMARIZE_PATCH, side_effect=_fake_summarize)
    def test_multiple_generations(self, _mock_sum: Any) -> None:
        """Multiple checkpoint-swap cycles should each increment generation."""
        config = ContextBufferConfig(
            checkpoint_threshold=0.50,
            swap_threshold=0.90,
            max_generations=10,
        )
        mgr = DoubleBufferContextManager(
            context_window_size=1000,
            llm=_make_mock_llm(),
            i18n=_make_mock_i18n(),
            config=config,
        )

        generations_seen: set[int] = set()
        for _ in range(30):
            mgr.append(_make_message("x" * 400))
            generations_seen.add(mgr.generation)

        # We should have seen multiple generations
        assert len(generations_seen) > 1

    def test_get_messages_returns_active_buffer_copy(self) -> None:
        """get_messages should return a copy, not the internal list."""
        mgr = DoubleBufferContextManager(
            context_window_size=100_000,
            llm=_make_mock_llm(),
            i18n=_make_mock_i18n(),
        )
        mgr.append(_make_message("hello"))
        messages = mgr.get_messages()
        assert messages == mgr.active_buffer
        assert messages is not mgr._active_buffer

    @patch(_SUMMARIZE_PATCH, side_effect=_fake_summarize)
    def test_reset_clears_all_state(self, _mock_sum: Any) -> None:
        """reset() should restore the manager to its initial state."""
        config = ContextBufferConfig(checkpoint_threshold=0.10, swap_threshold=0.95)
        mgr = DoubleBufferContextManager(
            context_window_size=1000,
            llm=_make_mock_llm(),
            i18n=_make_mock_i18n(),
            config=config,
        )
        mgr.append(_make_message("x" * 400))
        assert len(mgr.active_buffer) > 0

        mgr.reset()

        assert not mgr.has_back_buffer
        assert mgr.generation == 0
        assert mgr.active_buffer == []
        assert mgr.back_buffer == []
        assert mgr.accumulated_summaries == []


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge case and error handling tests."""

    def test_zero_context_window_size(self) -> None:
        """Zero context window should not cause division by zero."""
        mgr = DoubleBufferContextManager(
            context_window_size=0,
            llm=_make_mock_llm(),
            i18n=_make_mock_i18n(),
        )
        # Should not raise
        mgr.append(_make_message("hello"))
        assert len(mgr.active_buffer) == 1

    @patch(_SUMMARIZE_PATCH, side_effect=_fake_summarize)
    def test_system_messages_preserved_through_lifecycle(self, _mock_sum: Any) -> None:
        """System messages should survive checkpoint and swap."""
        config = ContextBufferConfig(
            checkpoint_threshold=0.50,
            swap_threshold=0.90,
        )
        mgr = DoubleBufferContextManager(
            context_window_size=1000,
            llm=_make_mock_llm(),
            i18n=_make_mock_i18n(),
            config=config,
        )

        sys_msg = _make_system_message("Persist this.")
        mgr.append(sys_msg)

        for _ in range(12):
            mgr.append(_make_message("x" * 400))

        assert len(mgr.active_buffer) > 0
        assert mgr.generation >= 1

    def test_none_content_messages_handled(self) -> None:
        """Messages with None content should not cause errors."""
        mgr = DoubleBufferContextManager(
            context_window_size=100_000,
            llm=_make_mock_llm(),
            i18n=_make_mock_i18n(),
        )
        mgr.append({"role": "assistant", "content": None})
        assert len(mgr.active_buffer) == 1
        assert mgr.estimate_buffer_tokens(mgr.active_buffer) == 0

    def test_multimodal_content_estimated(self) -> None:
        """List-based content should be estimated via str() representation."""
        mgr = DoubleBufferContextManager(
            context_window_size=100_000,
            llm=_make_mock_llm(),
            i18n=_make_mock_i18n(),
        )
        msg: dict[str, Any] = {
            "role": "user",
            "content": [{"type": "text", "text": "Describe this image"}],
        }
        mgr.append(msg)
        assert mgr.estimate_buffer_tokens(mgr.active_buffer) > 0

    def test_meta_summarize_with_no_summaries_falls_back_to_dump(self) -> None:
        """If meta_summarize is called with no accumulated summaries,
        it should fall back to dump."""
        mgr = DoubleBufferContextManager(
            context_window_size=1000,
            llm=_make_mock_llm(),
            i18n=_make_mock_i18n(),
        )
        mgr._accumulated_summaries.clear()
        mgr._active_buffer = [_make_system_message(), _make_message("data")]
        mgr._generation = 5

        mgr._meta_summarize()

        assert mgr.generation == 0
        assert not mgr.has_back_buffer

    def test_extract_summary_text_from_user_message(self) -> None:
        """_extract_summary_text should find the last user message content."""
        messages: list[dict[str, Any]] = [
            _make_system_message(),
            _make_message("The summary text"),
        ]
        result = DoubleBufferContextManager._extract_summary_text(messages)
        assert result == "The summary text"

    def test_extract_summary_text_empty_on_no_user_messages(self) -> None:
        """_extract_summary_text should return empty string if no user messages."""
        messages: list[dict[str, Any]] = [_make_system_message()]
        result = DoubleBufferContextManager._extract_summary_text(messages)
        assert result == ""

    def test_usage_ratio_calculation(self) -> None:
        """Usage ratio should correctly compute tokens / window size."""
        mgr = DoubleBufferContextManager(
            context_window_size=1000,
            llm=_make_mock_llm(),
            i18n=_make_mock_i18n(),
        )
        # 400 chars = 100 tokens -> ratio = 0.1
        mgr.append(_make_message("x" * 400))
        ratio = mgr._usage_ratio(mgr.active_buffer)
        assert abs(ratio - 0.1) < 0.01


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------

class TestEnums:
    """Tests for RenewalPolicy enum."""

    def test_renewal_policy_values(self) -> None:
        assert RenewalPolicy.RECURSE.value == "recurse"
        assert RenewalPolicy.DUMP.value == "dump"

    def test_renewal_policy_from_string(self) -> None:
        assert RenewalPolicy("recurse") == RenewalPolicy.RECURSE
        assert RenewalPolicy("dump") == RenewalPolicy.DUMP


# ---------------------------------------------------------------------------
# Stop-the-world fallback tests
# ---------------------------------------------------------------------------

class TestStopTheWorldFallback:
    """Tests for the stop-the-world fallback at swap time when no back buffer exists."""

    @patch(_SUMMARIZE_PATCH, side_effect=_fake_summarize)
    def test_swap_with_no_back_buffer_does_inline_checkpoint(self, _mock_sum: Any) -> None:
        """If _swap is called directly with an empty back buffer, it must
        perform a synchronous checkpoint first, NOT swap to an empty buffer."""
        mgr = DoubleBufferContextManager(
            context_window_size=1000,
            llm=_make_mock_llm(),
            i18n=_make_mock_i18n(),
            config=ContextBufferConfig(
                checkpoint_threshold=0.50,
                swap_threshold=0.90,
                max_generations=10,
            ),
        )
        # Pre-set state: active buffer has content, back buffer is empty
        mgr._active_buffer = [
            _make_system_message(),
            _make_message("msg 1"),
            _make_message("msg 2"),
            _make_message("msg 3"),
        ]
        mgr._back_buffer = []  # No back buffer!

        mgr._swap()

        # After stop-the-world: swap should have happened
        assert mgr.generation == 1
        assert not mgr.has_back_buffer
        assert mgr.back_buffer == []
        # Active buffer should NOT be empty (the inline checkpoint + swap
        # should have produced a summarized buffer)
        assert len(mgr.active_buffer) > 0
        _mock_sum.assert_called()

    @patch(_SUMMARIZE_PATCH, side_effect=_fake_summarize)
    def test_usage_jumps_past_swap_threshold_while_idle(self, _mock_sum: Any) -> None:
        """If a large message causes usage to jump past BOTH thresholds at
        once (from IDLE straight past swap threshold), the manager must NOT
        just skip compaction.  It should do a stop-the-world checkpoint+swap."""
        config = ContextBufferConfig(
            checkpoint_threshold=0.50,
            swap_threshold=0.90,
        )
        mgr = DoubleBufferContextManager(
            context_window_size=100,  # small window
            llm=_make_mock_llm(),
            i18n=_make_mock_i18n(),
            config=config,
        )

        # 400 chars = 100 tokens.  100 / 100 = 100% usage -> past swap threshold (90%)
        # This happens while still in IDLE state.
        mgr.append(_make_message("x" * 400))

        # Manager should have done a stop-the-world swap
        assert mgr.generation >= 1
        assert not mgr.has_back_buffer
        # Active buffer should have summarized content, not be empty
        assert len(mgr.active_buffer) > 0
        _mock_sum.assert_called()
