"""Base LLM abstract class for CrewAI.

This module provides the abstract base class for all LLM implementations
in CrewAI.
"""

from abc import ABC, abstractmethod
from typing import Any, Final

DEFAULT_CONTEXT_WINDOW_SIZE: Final[int] = 4096
DEFAULT_SUPPORTS_STOP_WORDS: Final[bool] = True


class BaseLLM(ABC):
    """Abstract base class for LLM implementations.

    This class defines the interface that all LLM implementations must follow.
    Users can extend this class to create custom LLM implementations that don't
    rely on litellm's authentication mechanism.

    Custom LLM implementations should handle error cases gracefully, including
    timeouts, authentication failures, and malformed responses. They should also
    implement proper validation for input parameters and provide clear error
    messages when things go wrong.

    Attributes:
        model: The model identifier/name.
        temperature: Optional temperature setting for response generation.
        stop: A list of stop sequences that the LLM should use to stop generation.
    """

    def __init__(
        self,
        model: str,
        temperature: float | None = None,
        stop: list[str] | None = None,
    ) -> None:
        """Initialize the BaseLLM with default attributes.

        Args:
            model: The model identifier/name.
            temperature: Optional temperature setting for response generation.
            stop: Optional list of stop sequences for generation.
        """
        self.model = model
        self.temperature = temperature
        self.stop: list[str] = stop or []

    @abstractmethod
    def call(
        self,
        messages: str | list[dict[str, str]],
        tools: list[dict] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
    ) -> str | Any:
        """Call the LLM with the given messages.

        Args:
            messages: Input messages for the LLM.
                     Can be a string or list of message dictionaries.
                     If string, it will be converted to a single user message.
                     If list, each dict must have 'role' and 'content' keys.
            tools: Optional list of tool schemas for function calling.
                  Each tool should define its name, description, and parameters.
            callbacks: Optional list of callback functions to be executed
                      during and after the LLM call.
            available_functions: Optional dict mapping function names to callables
                               that can be invoked by the LLM.
            from_task: Optional task caller to be used for the LLM call.
            from_agent: Optional agent caller to be used for the LLM call.

        Returns:
            Either a text response from the LLM (str) or
            the result of a tool function call (Any).

        Raises:
            ValueError: If the messages format is invalid.
            TimeoutError: If the LLM request times out.
            RuntimeError: If the LLM request fails for other reasons.
        """

    def supports_stop_words(self) -> bool:
        """Check if the LLM supports stop words.

        Returns:
            True if the LLM supports stop words, False otherwise.
        """
        return DEFAULT_SUPPORTS_STOP_WORDS

    def get_context_window_size(self) -> int:
        """Get the context window size for the LLM.

        Returns:
            The number of tokens/characters the model can handle.
        """
        # Default implementation - subclasses should override with model-specific values
        return DEFAULT_CONTEXT_WINDOW_SIZE
