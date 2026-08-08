"""Double-buffered context window management for CrewAI.

Proactive context compaction using double buffering, checkpoint + WAL replay,
and hopping windows to eliminate stop-the-world pauses when the context window
fills up.

Algorithm (3 phases):
    1. **Checkpoint** — at ``checkpoint_threshold`` (default 70% capacity),
       summarize the active buffer and seed the back buffer with the summary.
    2. **Concurrent** — keep working; append every new message to BOTH the
       active buffer and the back buffer.
    3. **Swap** — when the active buffer hits ``swap_threshold`` (default 95%),
       swap to the back buffer seamlessly.

Each swap increments a *generation* counter. When ``max_generations`` is
reached the manager either recurses (meta-summarize all prior summaries) or
dumps (clean restart), controlled by ``renewal_policy``.  The default
``max_generations`` is ``None`` (renewal disabled) so that each checkpoint
cycle produces a fresh summary without incremental accumulation unless the
caller explicitly configures a finite value.
"""

from __future__ import annotations

import copy
from enum import Enum
import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from crewai.utilities.agent_utils import (
    _estimate_token_count,
    summarize_messages,
)
from crewai.utilities.types import LLMMessage


if TYPE_CHECKING:
    from crewai.utilities.i18n import I18N
    from crewai.utilities.token_counter_callback import TokenCalcHandler

logger = logging.getLogger(__name__)


class RenewalPolicy(str, Enum):
    """Strategy for handling maximum generation count.

    Attributes:
        RECURSE: Meta-summarize all accumulated summaries into one.
        DUMP: Discard all accumulated summaries and start clean.
    """

    RECURSE = "recurse"
    DUMP = "dump"



class ContextBufferConfig(BaseModel):
    """Configuration for the double-buffer context manager.

    All thresholds are expressed as fractions of the context window capacity
    (in estimated tokens).
    """

    checkpoint_threshold: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of context window capacity at which the checkpoint "
            "phase triggers (summarise and seed back buffer)."
        ),
    )
    swap_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of context window capacity at which the swap phase "
            "triggers (switch to back buffer)."
        ),
    )
    max_generations: int | None = Field(
        default=None,
        description=(
            "Maximum number of swap generations before a renewal is forced. "
            "Each swap increments the generation counter.  None means no limit "
            "(renewal disabled)."
        ),
    )
    renewal_policy: RenewalPolicy = Field(
        default=RenewalPolicy.RECURSE,
        description=(
            "What to do when max_generations is reached: 'recurse' "
            "meta-summarizes accumulated summaries; 'dump' does a clean restart."
        ),
    )


class DoubleBufferContextManager:
    """Proactive double-buffered context window manager.

    Instead of waiting for a context-length exception (reactive), this manager
    monitors estimated token usage and performs compaction *before* the window
    is full.

    Usage::

        mgr = DoubleBufferContextManager(
            context_window_size=128_000,
            llm=my_llm,
            i18n=my_i18n,
        )
        # After each message exchange:
        mgr.append(message)
        # The manager will checkpoint / swap automatically.

    Parameters:
        context_window_size: Total context window size in estimated tokens.
        llm: LLM instance used for summarization (must support ``call``).
        i18n: Internationalization instance for prompt templates.
        callbacks: Optional list of token-counting callbacks.
        config: Optional ``ContextBufferConfig`` for tuning thresholds.
    """

    def __init__(
        self,
        context_window_size: int,
        llm: Any,
        i18n: I18N,
        callbacks: list[TokenCalcHandler] | None = None,
        config: ContextBufferConfig | None = None,
    ) -> None:
        self._config = config or ContextBufferConfig()
        self._context_window_size = context_window_size
        self._llm = llm
        self._i18n = i18n
        self._callbacks: list[TokenCalcHandler] = callbacks or []

        # Buffers
        self._active_buffer: list[LLMMessage] = []
        self._back_buffer: list[LLMMessage] = []

        # State tracking
        self._generation: int = 0
        self._accumulated_summaries: list[str] = []

        # Validate thresholds
        if self._config.checkpoint_threshold >= self._config.swap_threshold:
            raise ValueError(
                f"checkpoint_threshold ({self._config.checkpoint_threshold}) "
                f"must be less than swap_threshold ({self._config.swap_threshold})"
            )

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def has_back_buffer(self) -> bool:
        """Whether a back buffer currently exists (checkpoint taken)."""
        return bool(self._back_buffer)

    @property
    def generation(self) -> int:
        """Current generation count (incremented on each swap)."""
        return self._generation

    @property
    def active_buffer(self) -> list[LLMMessage]:
        """Read-only view of the active message buffer."""
        return self._active_buffer

    @property
    def back_buffer(self) -> list[LLMMessage]:
        """Read-only view of the back buffer."""
        return self._back_buffer

    @property
    def config(self) -> ContextBufferConfig:
        """Current configuration."""
        return self._config

    @property
    def accumulated_summaries(self) -> list[str]:
        """List of summaries accumulated across generations."""
        return list(self._accumulated_summaries)

    # ------------------------------------------------------------------
    # Token estimation
    # ------------------------------------------------------------------

    def estimate_buffer_tokens(self, buffer: list[LLMMessage]) -> int:
        """Estimate the total token count for a buffer of messages.

        Uses CrewAI's ``_estimate_token_count`` heuristic (1 token per 4 chars).

        Args:
            buffer: List of LLM messages to estimate.

        Returns:
            Estimated total tokens across all messages.
        """
        total = 0
        for msg in buffer:
            content = msg.get("content")
            if content is None:
                continue
            if isinstance(content, list):
                text = str(content)
            else:
                text = str(content)
            total += _estimate_token_count(text)
        return total

    def _usage_ratio(self, buffer: list[LLMMessage]) -> float:
        """Return estimated token usage as a fraction of the context window."""
        if self._context_window_size <= 0:
            return 0.0
        return self.estimate_buffer_tokens(buffer) / self._context_window_size

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def append(self, message: LLMMessage) -> None:
        """Append a message and evaluate whether to checkpoint or swap.

        This is the main entry point. After every message exchange, call this
        method. It will:

        1. Always append to the active buffer.
        2. If in CONCURRENT state, also append to the back buffer.
        3. Check thresholds and trigger checkpoint or swap as needed.

        Args:
            message: The LLM message to append.
        """
        self._active_buffer.append(message)

        if self._back_buffer:
            self._back_buffer.append(message)

        self._evaluate_thresholds()

    def _evaluate_thresholds(self) -> None:
        """Check active buffer usage and trigger checkpoint or swap.

        If usage jumps past the swap threshold while still IDLE (no checkpoint
        has occurred), we trigger a stop-the-world swap which will internally
        perform a synchronous checkpoint first -- we NEVER just skip compaction.
        """
        usage = self._usage_ratio(self._active_buffer)

        if not self._back_buffer and usage >= self._config.swap_threshold:
            # Usage jumped past swap threshold without an intermediate
            # checkpoint.  Force a stop-the-world swap (which will checkpoint
            # inline since no back buffer exists).
            logger.warning(
                "DoubleBufferContextManager: usage %.1f%% jumped past swap "
                "threshold with no back buffer -- forcing stop-the-world swap "
                "(generation=%d)",
                usage * 100,
                self._generation,
            )
            self._swap()
        elif not self._back_buffer and usage >= self._config.checkpoint_threshold:
            self._checkpoint()
        elif self._back_buffer and usage >= self._config.swap_threshold:
            self._swap()

    def _checkpoint(self) -> None:
        """Phase 1: Summarize the active buffer and seed the back buffer.

        - Summarizes the current active buffer contents.
        - Seeds the back buffer with system messages + the summary.
        """
        logger.info(
            "DoubleBufferContextManager: checkpoint triggered at %.1f%% usage "
            "(generation=%d)",
            self._usage_ratio(self._active_buffer) * 100,
            self._generation,
        )

        # Take a snapshot for summarization (don't modify active buffer yet)
        snapshot = [copy.deepcopy(msg) for msg in self._active_buffer]

        # Summarize the snapshot — summarize_messages modifies in-place
        summarize_messages(
            messages=snapshot,
            llm=self._llm,
            callbacks=self._callbacks,
            i18n=self._i18n,
            verbose=False,
        )

        # Store the summary text for incremental accumulation
        summary_text = self._extract_summary_text(snapshot)
        if summary_text:
            self._accumulated_summaries.append(summary_text)

        # Seed the back buffer with the summarized snapshot
        self._back_buffer = list(snapshot)

        logger.info(
            "DoubleBufferContextManager: back buffer seeded with %d message(s) "
            "(generation=%d)",
            len(self._back_buffer),
            self._generation,
        )

    def _swap(self) -> None:
        """Phase 3: Swap the active buffer with the back buffer.

        - If no back buffer exists (checkpoint was never taken or failed),
          performs a synchronous stop-the-world checkpoint before swapping.
          We NEVER skip compaction -- the user was clear: if we hit swap time
          without a compacted back buffer, we must do it then.
        - Replaces the active buffer with the back buffer.
        - Clears the back buffer.
        - Increments the generation counter.
        - Checks if renewal is needed.
        """
        logger.info(
            "DoubleBufferContextManager: swap triggered at %.1f%% usage "
            "(generation=%d -> %d)",
            self._usage_ratio(self._active_buffer) * 100,
            self._generation,
            self._generation + 1,
        )

        # Stop-the-world fallback: if no back buffer exists, do an inline
        # checkpoint now so we never swap to an empty/missing buffer.
        if not self._back_buffer:
            logger.warning(
                "DoubleBufferContextManager: no back buffer at swap time! "
                "Performing synchronous stop-the-world checkpoint "
                "(generation=%d)",
                self._generation,
            )
            self._checkpoint()

        self._active_buffer = self._back_buffer
        self._back_buffer = []
        self._generation += 1

        logger.info(
            "DoubleBufferContextManager: swapped to back buffer with %d message(s) "
            "(generation=%d)",
            len(self._active_buffer),
            self._generation,
        )

        # Check if we need renewal
        if self._config.max_generations is not None and self._generation >= self._config.max_generations:
            self._renew()

    def _renew(self) -> None:
        """Handle max generation renewal based on the configured policy.

        RECURSE: meta-summarize all accumulated summaries into a single
                 compressed summary, then reset the generation counter.
        DUMP: discard everything and start with a clean active buffer.
        """
        logger.info(
            "DoubleBufferContextManager: renewal triggered (policy=%s, "
            "generation=%d, accumulated_summaries=%d)",
            self._config.renewal_policy.value,
            self._generation,
            len(self._accumulated_summaries),
        )

        if self._config.renewal_policy == RenewalPolicy.RECURSE:
            self._meta_summarize()
        elif self._config.renewal_policy == RenewalPolicy.DUMP:
            self._dump()

    def _meta_summarize(self) -> None:
        """Recurse: combine all accumulated summaries into one meta-summary.

        Builds a synthetic conversation from all accumulated summaries, runs
        summarization on it, then resets the active buffer to system messages
        plus the meta-summary.
        """
        if not self._accumulated_summaries:
            logger.warning(
                "DoubleBufferContextManager: meta-summarize called with no "
                "accumulated summaries; falling back to dump"
            )
            self._dump()
            return

        # Preserve system messages from active buffer
        system_messages = [
            msg for msg in self._active_buffer if msg.get("role") == "system"
        ]

        # Build a synthetic conversation from accumulated summaries
        combined = "\n\n---\n\n".join(self._accumulated_summaries)
        meta_messages: list[LLMMessage] = list(system_messages)
        meta_messages.append({"role": "user", "content": combined})  # type: ignore[typeddict-item]

        summarize_messages(
            messages=meta_messages,
            llm=self._llm,
            callbacks=self._callbacks,
            i18n=self._i18n,
            verbose=False,
        )

        # Replace active buffer with the meta-summarized content.
        # The meta-summary already covers the accumulated history so we
        # do not re-append previous non-system messages.
        self._active_buffer = list(meta_messages)

        # Extract the new meta-summary for accumulation tracking
        meta_summary_text = self._extract_summary_text(meta_messages)
        self._accumulated_summaries.clear()
        if meta_summary_text:
            self._accumulated_summaries.append(meta_summary_text)

        self._generation = 0

        logger.info(
            "DoubleBufferContextManager: meta-summarize complete; "
            "generation reset to 0"
        )

    def _dump(self) -> None:
        """Dump: clean restart — preserve only system messages."""
        system_messages = [
            msg for msg in self._active_buffer if msg.get("role") == "system"
        ]
        self._active_buffer = list(system_messages)
        self._back_buffer = []
        self._accumulated_summaries.clear()
        self._generation = 0

        logger.info(
            "DoubleBufferContextManager: dump complete; clean restart with "
            "%d system message(s)",
            len(system_messages),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_summary_text(messages: list[LLMMessage]) -> str:
        """Extract the summary text from a summarized message list.

        After ``summarize_messages`` runs, the result is [system...] + [user
        summary]. This extracts the user summary content.

        Args:
            messages: Messages after summarization.

        Returns:
            The summary text, or empty string if none found.
        """
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    return str(content)
        return ""

    def get_messages(self) -> list[LLMMessage]:
        """Return the current active buffer for use as the LLM context.

        Returns:
            A copy of the active buffer messages.
        """
        return list(self._active_buffer)

    def reset(self) -> None:
        """Fully reset the manager to its initial state."""
        self._active_buffer.clear()
        self._back_buffer.clear()
        self._accumulated_summaries.clear()
        self._generation = 0
