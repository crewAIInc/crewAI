"""Memory management for continuous operation mode.

This module provides utilities to prevent unbounded memory growth
during long-running continuous operations.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class MemoryStats(BaseModel):
    """Statistics about memory usage in continuous mode."""

    total_messages_processed: int = Field(
        default=0, description="Total messages processed since start"
    )
    current_message_count: int = Field(
        default=0, description="Current number of messages in memory"
    )
    summarizations_performed: int = Field(
        default=0, description="Number of times memory was summarized"
    )
    last_summarization: datetime | None = Field(
        default=None, description="Timestamp of last summarization"
    )
    trimmed_message_count: int = Field(
        default=0, description="Total number of messages trimmed"
    )


class ContinuousMemoryHandler:
    """Handler to prevent unbounded memory growth in continuous mode.

    This handler manages conversation history and observations to ensure
    memory doesn't grow indefinitely during long-running operations.

    Features:
    - Automatic trimming when message count exceeds threshold
    - Summarization of older messages to preserve context
    - Statistics tracking for monitoring
    - Configurable retention policies

    Example:
        ```python
        handler = ContinuousMemoryHandler(max_messages=100)

        # In continuous loop
        messages = handler.manage(messages)  # Auto-trims if needed
        ```
    """

    def __init__(
        self,
        max_messages: int = 100,
        retain_recent: int = 50,
        summarize_old: bool = True,
        summary_prefix: str = "[Previous context summary]",
    ) -> None:
        """Initialize the memory handler.

        Args:
            max_messages: Maximum messages before triggering cleanup
            retain_recent: Number of recent messages to always keep
            summarize_old: Whether to summarize old messages or just discard
            summary_prefix: Prefix for summary messages
        """
        self.max_messages = max_messages
        self.retain_recent = min(retain_recent, max_messages)
        self.summarize_old = summarize_old
        self.summary_prefix = summary_prefix

        self._stats = MemoryStats()
        self._summaries: list[str] = []

    @property
    def stats(self) -> MemoryStats:
        """Get current memory statistics."""
        return self._stats

    @property
    def summaries(self) -> list[str]:
        """Get all generated summaries."""
        return self._summaries.copy()

    def manage(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Manage message list to prevent unbounded growth.

        If the message list exceeds max_messages, older messages are
        either summarized or discarded based on configuration.

        Args:
            messages: List of message dictionaries

        Returns:
            Managed message list (potentially trimmed/summarized)
        """
        self._stats.total_messages_processed += len(messages)
        self._stats.current_message_count = len(messages)

        if len(messages) <= self.max_messages:
            return messages

        return self._summarize_and_trim(messages)

    def _summarize_and_trim(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Summarize older messages and trim the list.

        Args:
            messages: Full message list

        Returns:
            Trimmed message list with optional summary
        """
        # Calculate how many messages to trim
        trim_count = len(messages) - self.retain_recent
        old_messages = messages[:trim_count]
        recent_messages = messages[trim_count:]

        self._stats.trimmed_message_count += trim_count
        self._stats.summarizations_performed += 1
        self._stats.last_summarization = datetime.now()

        if self.summarize_old and old_messages:
            # Create a summary of the old messages
            summary = self._create_summary(old_messages)
            self._summaries.append(summary)

            # Prepend summary as a system message
            summary_message = {
                "role": "system",
                "content": f"{self.summary_prefix}\n{summary}",
            }

            # Keep max 5 summaries to prevent summary accumulation
            if len(self._summaries) > 5:
                self._summaries = self._summaries[-3:]

            return [summary_message] + recent_messages

        return recent_messages

    def _create_summary(self, messages: list[dict[str, Any]]) -> str:
        """Create a summary of messages.

        This creates a basic summary. For more sophisticated summarization,
        consider using an LLM-based summarizer.

        Args:
            messages: Messages to summarize

        Returns:
            Summary string
        """
        # Extract key information from messages
        actions: list[str] = []
        observations: list[str] = []
        thoughts: list[str] = []

        for msg in messages:
            content = msg.get("content", "")
            if not content:
                continue

            # Simple extraction based on content patterns
            if "Action:" in content:
                # Extract action
                action_start = content.find("Action:")
                action_end = content.find("\n", action_start)
                if action_end == -1:
                    action_end = len(content)
                action = content[action_start + 7 : action_end].strip()
                if action and action not in actions:
                    actions.append(action)

            if "Observation:" in content:
                # Extract observation
                obs_start = content.find("Observation:")
                obs_end = content.find("\n", obs_start)
                if obs_end == -1:
                    obs_end = min(obs_start + 200, len(content))
                obs = content[obs_start + 12 : obs_end].strip()
                if obs:
                    observations.append(obs[:100])

            if "Thought:" in content:
                # Extract thought
                thought_start = content.find("Thought:")
                thought_end = content.find("\n", thought_start)
                if thought_end == -1:
                    thought_end = min(thought_start + 150, len(content))
                thought = content[thought_start + 8 : thought_end].strip()
                if thought:
                    thoughts.append(thought[:80])

        # Build summary
        summary_parts: list[str] = [
            f"Processed {len(messages)} previous messages.",
        ]

        if actions:
            unique_actions = list(dict.fromkeys(actions))[:5]
            summary_parts.append(f"Actions taken: {', '.join(unique_actions)}")

        if observations:
            recent_obs = observations[-3:]
            summary_parts.append(f"Recent observations: {'; '.join(recent_obs)}")

        if thoughts:
            key_thoughts = thoughts[-2:]
            summary_parts.append(f"Key thoughts: {'; '.join(key_thoughts)}")

        return " ".join(summary_parts)

    def manage_observations(self, observations: list[str], max_count: int = 50) -> list[str]:
        """Manage observation list to prevent unbounded growth.

        Args:
            observations: List of observations
            max_count: Maximum observations to keep

        Returns:
            Trimmed observation list
        """
        if len(observations) <= max_count:
            return observations

        # Keep most recent observations
        return observations[-max_count:]

    def manage_tool_results(
        self,
        tool_results: list[dict[str, Any]],
        max_count: int = 30,
    ) -> list[dict[str, Any]]:
        """Manage tool results list to prevent unbounded growth.

        Args:
            tool_results: List of tool result dictionaries
            max_count: Maximum results to keep

        Returns:
            Trimmed tool results list
        """
        if len(tool_results) <= max_count:
            return tool_results

        # Keep most recent tool results
        return tool_results[-max_count:]

    def reset(self) -> None:
        """Reset memory handler state.

        Clears statistics and summaries for a fresh start.
        """
        self._stats = MemoryStats()
        self._summaries = []

    def get_stats_dict(self) -> dict[str, Any]:
        """Get statistics as a dictionary.

        Returns:
            Dictionary with memory statistics
        """
        return {
            "total_messages_processed": self._stats.total_messages_processed,
            "current_message_count": self._stats.current_message_count,
            "summarizations_performed": self._stats.summarizations_performed,
            "last_summarization": (
                self._stats.last_summarization.isoformat()
                if self._stats.last_summarization
                else None
            ),
            "trimmed_message_count": self._stats.trimmed_message_count,
            "summary_count": len(self._summaries),
        }


class ObservationBuffer:
    """Circular buffer for storing observations efficiently.

    Uses a fixed-size buffer to store observations, automatically
    overwriting oldest entries when full.

    Example:
        ```python
        buffer = ObservationBuffer(max_size=100)
        buffer.add("Price: $45,000")
        buffer.add("Volume: High")

        recent = buffer.get_recent(10)
        ```
    """

    def __init__(self, max_size: int = 100) -> None:
        """Initialize the observation buffer.

        Args:
            max_size: Maximum number of observations to store
        """
        self.max_size = max_size
        self._buffer: list[tuple[datetime, str]] = []
        self._total_added: int = 0

    def add(self, observation: str) -> None:
        """Add an observation to the buffer.

        Args:
            observation: Observation text to add
        """
        self._buffer.append((datetime.now(), observation))
        self._total_added += 1

        # Remove oldest if over capacity
        if len(self._buffer) > self.max_size:
            self._buffer.pop(0)

    def get_recent(self, count: int = 10) -> list[str]:
        """Get most recent observations.

        Args:
            count: Number of recent observations to return

        Returns:
            List of recent observation strings
        """
        recent = self._buffer[-count:]
        return [obs for _, obs in recent]

    def get_recent_with_timestamps(
        self, count: int = 10
    ) -> list[tuple[datetime, str]]:
        """Get most recent observations with timestamps.

        Args:
            count: Number of recent observations to return

        Returns:
            List of (timestamp, observation) tuples
        """
        return self._buffer[-count:]

    def get_since(self, since: datetime) -> list[str]:
        """Get observations since a specific time.

        Args:
            since: Datetime to filter from

        Returns:
            List of observations after the specified time
        """
        return [obs for ts, obs in self._buffer if ts > since]

    def clear(self) -> None:
        """Clear all observations from the buffer."""
        self._buffer = []

    @property
    def count(self) -> int:
        """Get current number of observations."""
        return len(self._buffer)

    @property
    def total_added(self) -> int:
        """Get total observations added since creation."""
        return self._total_added

    def __len__(self) -> int:
        """Get buffer length."""
        return len(self._buffer)

    def __iter__(self):
        """Iterate over observations."""
        return iter(obs for _, obs in self._buffer)
