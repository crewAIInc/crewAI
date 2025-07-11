from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union


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
        stop (list): A list of stop sequences that the LLM should use to stop generation.
            This is used by the CrewAgentExecutor and other components.
    """

    model: str
    temperature: Optional[float] = None
    stop: Optional[List[str]] = None

    def __init__(
        self,
        model: str,
        temperature: Optional[float] = None,
    ):
        """Initialize the BaseLLM with default attributes.

        This constructor sets default values for attributes that are expected
        by the CrewAgentExecutor and other components.

        All custom LLM implementations should call super().__init__() to ensure
        that these default attributes are properly initialized.
        """
        self.model = model
        self.temperature = temperature
        self.stop = []

    @abstractmethod
    def call(
        self,
        messages: Union[str, List[Dict[str, str]]],
        tools: Optional[List[dict]] = None,
        callbacks: Optional[List[Any]] = None,
        available_functions: Optional[Dict[str, Any]] = None,
        from_task: Optional[Any] = None,
        from_agent: Optional[Any] = None,
    ) -> Union[str, Any]:
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

        Returns:
            Either a text response from the LLM (str) or
            the result of a tool function call (Any).

        Raises:
            ValueError: If the messages format is invalid.
            TimeoutError: If the LLM request times out.
            RuntimeError: If the LLM request fails for other reasons.
        """
        pass

    def supports_stop_words(self) -> bool:
        """Check if the LLM supports stop words.

        Returns:
            bool: True if the LLM supports stop words, False otherwise.
        """
        return True  # Default implementation assumes support for stop words

    def get_context_window_size(self) -> int:
        """Get the context window size for the LLM.

        Returns:
            int: The number of tokens/characters the model can handle.
        """
        # Default implementation - subclasses should override with model-specific values
        return 4096
