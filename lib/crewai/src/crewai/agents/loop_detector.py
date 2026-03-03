"""Loop detection for agent execution.

Detects repetitive behavioral patterns in agent tool calls and provides
configurable intervention strategies to break out of loops.

Example usage:

    from crewai import Agent
    from crewai.agents.loop_detector import LoopDetector

    # Using default settings (window_size=5, repetition_threshold=3, inject_reflection)
    agent = Agent(
        role="Researcher",
        goal="Find novel insights",
        backstory="An experienced researcher",
        loop_detector=LoopDetector(),
    )

    # Custom configuration
    agent = Agent(
        role="Researcher",
        goal="Find novel insights",
        backstory="An experienced researcher",
        loop_detector=LoopDetector(
            window_size=10,
            repetition_threshold=4,
            on_loop="stop",
        ),
    )

    # With a custom callback
    def my_callback(detector: LoopDetector) -> str:
        return "You are repeating yourself. Try a completely different approach."

    agent = Agent(
        role="Researcher",
        goal="Find novel insights",
        backstory="An experienced researcher",
        loop_detector=LoopDetector(
            on_loop=my_callback,
        ),
    )
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
import json
import logging
from typing import Any, Literal

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class ToolCallRecord(BaseModel):
    """Record of a single tool call for loop detection."""

    tool_name: str
    tool_args: str  # Normalized JSON string of arguments

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ToolCallRecord):
            return NotImplemented
        return self.tool_name == other.tool_name and self.tool_args == other.tool_args

    def __hash__(self) -> int:
        return hash((self.tool_name, self.tool_args))


class LoopDetector(BaseModel):
    """Detects repetitive behavioral patterns in agent tool calls.

    Monitors a sliding window of recent tool calls and flags when
    the same tool is called repeatedly with the same arguments,
    indicating the agent is stuck in a loop.

    Args:
        window_size: Number of recent tool calls to track in the
            sliding window. Defaults to 5.
        repetition_threshold: How many identical tool calls within
            the window trigger loop detection. Defaults to 3.
        on_loop: Action to take when a loop is detected.
            - ``"inject_reflection"`` (default): Inject a meta-prompt
              asking the agent to try a different approach.
            - ``"stop"``: Force the agent to produce a final answer.
            - A callable ``(LoopDetector) -> str``: Custom callback
              that returns a message to inject into the conversation.

    Example::

        from crewai import Agent
        from crewai.agents.loop_detector import LoopDetector

        agent = Agent(
            role="Researcher",
            goal="Find novel insights about AI",
            backstory="Senior researcher",
            loop_detector=LoopDetector(
                window_size=5,
                repetition_threshold=3,
                on_loop="inject_reflection",
            ),
        )
    """

    window_size: int = Field(
        default=5,
        ge=2,
        description="Number of recent tool calls to track in the sliding window.",
    )
    repetition_threshold: int = Field(
        default=3,
        ge=2,
        description="How many identical tool calls within the window trigger loop detection.",
    )
    on_loop: Literal["inject_reflection", "stop"] | Callable[..., str] = Field(
        default="inject_reflection",
        description=(
            "Action when loop is detected: 'inject_reflection' to inject a reflection prompt, "
            "'stop' to force final answer, or a callable(LoopDetector) -> str for custom handling."
        ),
    )
    _history: deque[ToolCallRecord] = deque()

    def model_post_init(self, __context: Any) -> None:
        """Initialize the internal deque with the configured window size."""
        object.__setattr__(self, "_history", deque(maxlen=self.window_size))

    @staticmethod
    def _normalize_args(args: str | dict[str, Any]) -> str:
        """Normalize tool arguments to a canonical JSON string for comparison.

        Args:
            args: Tool arguments as a string or dict.

        Returns:
            A normalized JSON string with sorted keys.
        """
        if isinstance(args, str):
            try:
                parsed = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                return args.strip()
            return json.dumps(parsed, sort_keys=True, default=str)
        return json.dumps(args, sort_keys=True, default=str)

    def record_tool_call(self, tool_name: str, tool_args: str | dict[str, Any]) -> None:
        """Record a tool call in the sliding window.

        Args:
            tool_name: Name of the tool that was called.
            tool_args: Arguments passed to the tool.
        """
        record = ToolCallRecord(
            tool_name=tool_name,
            tool_args=self._normalize_args(tool_args),
        )
        self._history.append(record)

    def is_loop_detected(self) -> bool:
        """Check if a repetitive loop pattern is detected.

        Returns:
            True if any single tool call (same name + same args)
            appears at least ``repetition_threshold`` times within
            the current window.
        """
        if len(self._history) < self.repetition_threshold:
            return False

        counts: dict[ToolCallRecord, int] = {}
        for record in self._history:
            counts[record] = counts.get(record, 0) + 1
            if counts[record] >= self.repetition_threshold:
                return True
        return False

    def get_loop_message(self) -> str:
        """Get the intervention message based on the configured ``on_loop`` action.

        Returns:
            A string message to inject into the agent conversation.
        """
        if callable(self.on_loop) and not isinstance(self.on_loop, str):
            return self.on_loop(self)
        return ""

    def get_repeated_tool_info(self) -> str | None:
        """Get information about the repeated tool call, if any.

        Returns:
            A string describing the repeated tool and args, or None.
        """
        if len(self._history) < self.repetition_threshold:
            return None

        counts: dict[ToolCallRecord, int] = {}
        for record in self._history:
            counts[record] = counts.get(record, 0) + 1
            if counts[record] >= self.repetition_threshold:
                return f"{record.tool_name}({record.tool_args})"
        return None

    def reset(self) -> None:
        """Clear the tool call history."""
        self._history.clear()
