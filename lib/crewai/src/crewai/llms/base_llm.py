"""Base LLM abstract class for CrewAI.

This module provides the abstract base class for all LLM implementations
in CrewAI, including common functionality for native SDK implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
import json
import logging
import re
from typing import TYPE_CHECKING, Any, Final

from pydantic import BaseModel

from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.llm_events import (
    LLMCallCompletedEvent,
    LLMCallFailedEvent,
    LLMCallStartedEvent,
    LLMCallType,
    LLMStreamChunkEvent,
)
from crewai.events.types.tool_usage_events import (
    ToolUsageErrorEvent,
    ToolUsageFinishedEvent,
    ToolUsageStartedEvent,
)
from crewai.types.usage_metrics import UsageMetrics


if TYPE_CHECKING:
    from crewai.agent.core import Agent
    from crewai.task import Task
    from crewai.tools.base_tool import BaseTool
    from crewai.utilities.types import LLMMessage


DEFAULT_CONTEXT_WINDOW_SIZE: Final[int] = 4096
DEFAULT_SUPPORTS_STOP_WORDS: Final[bool] = True
_JSON_EXTRACTION_PATTERN: Final[re.Pattern[str]] = re.compile(r"\{.*}", re.DOTALL)


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
        additional_params: Additional provider-specific parameters.
    """

    is_litellm: bool = False

    def __init__(
        self,
        model: str,
        temperature: float | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        provider: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the BaseLLM with default attributes.

        Args:
            model: The model identifier/name.
            temperature: Optional temperature setting for response generation.
            stop: Optional list of stop sequences for generation.
            **kwargs: Additional provider-specific parameters.
        """
        if not model:
            raise ValueError("Model name is required and cannot be empty")

        self.model = model
        self.temperature = temperature
        self.api_key = api_key
        self.base_url = base_url
        # Store additional parameters for provider-specific use
        self.additional_params = kwargs
        self._provider = provider or "openai"

        stop = kwargs.pop("stop", None)
        if stop is None:
            self.stop: list[str] = []
        elif isinstance(stop, str):
            self.stop = [stop]
        elif isinstance(stop, list):
            self.stop = stop
        else:
            self.stop = []

        self._token_usage = {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "successful_requests": 0,
            "cached_prompt_tokens": 0,
        }

    @property
    def provider(self) -> str:
        """Get the provider of the LLM."""
        return self._provider

    @provider.setter
    def provider(self, value: str) -> None:
        """Set the provider of the LLM."""
        self._provider = value

    @abstractmethod
    def call(
        self,
        messages: str | list[LLMMessage],
        tools: list[dict[str, BaseTool]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Task | None = None,
        from_agent: Agent | None = None,
        response_model: type[BaseModel] | None = None,
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
            response_model: Optional response model to be used for the LLM call.

        Returns:
            Either a text response from the LLM (str) or
            the result of a tool function call (Any).

        Raises:
            ValueError: If the messages format is invalid.
            TimeoutError: If the LLM request times out.
            RuntimeError: If the LLM request fails for other reasons.
        """

    def _convert_tools_for_interference(
        self, tools: list[dict[str, BaseTool]]
    ) -> list[dict[str, BaseTool]]:
        """Convert tools to a format that can be used for interference.

        Args:
            tools: List of tools to convert.

        Returns:
            List of converted tools (default implementation returns as-is)
        """
        return tools

    def supports_stop_words(self) -> bool:
        """Check if the LLM supports stop words.

        Returns:
            True if the LLM supports stop words, False otherwise.
        """
        return DEFAULT_SUPPORTS_STOP_WORDS

    def _supports_stop_words_implementation(self) -> bool:
        """Check if stop words are configured for this LLM instance.

        Native providers can override supports_stop_words() to return this value
        to ensure consistent behavior based on whether stop words are actually configured.

        Returns:
            True if stop words are configured and can be applied
        """
        return bool(self.stop)

    def _apply_stop_words(self, content: str) -> str:
        """Apply stop words to truncate response content.

        This method provides consistent stop word behavior across all native SDK providers.
        Native providers should call this method to post-process their responses.

        Args:
            content: The raw response content from the LLM

        Returns:
            Content truncated at the first occurrence of any stop word

        Example:
            >>> llm = MyNativeLLM(stop=["Observation:", "Final Answer:"])
            >>> response = (
            ...     "I need to search.\\n\\nAction: search\\nObservation: Found results"
            ... )
            >>> llm._apply_stop_words(response)
            "I need to search.\\n\\nAction: search"
        """
        if not self.stop or not content:
            return content

        # Find the earliest occurrence of any stop word
        earliest_stop_pos = len(content)
        found_stop_word = None

        for stop_word in self.stop:
            stop_pos = content.find(stop_word)
            if stop_pos != -1 and stop_pos < earliest_stop_pos:
                earliest_stop_pos = stop_pos
                found_stop_word = stop_word

        # Truncate at the stop word if found
        if found_stop_word is not None:
            truncated = content[:earliest_stop_pos].strip()
            logging.debug(
                f"Applied stop word '{found_stop_word}' at position {earliest_stop_pos}"
            )
            return truncated

        return content

    def get_context_window_size(self) -> int:
        """Get the context window size for the LLM.

        Returns:
            The number of tokens/characters the model can handle.
        """
        # Default implementation - subclasses should override with model-specific values
        return DEFAULT_CONTEXT_WINDOW_SIZE

    # Common helper methods for native SDK implementations

    def _emit_call_started_event(
        self,
        messages: str | list[LLMMessage],
        tools: list[dict[str, BaseTool]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Task | None = None,
        from_agent: Agent | None = None,
    ) -> None:
        """Emit LLM call started event."""
        if not hasattr(crewai_event_bus, "emit"):
            raise ValueError("crewai_event_bus does not have an emit method") from None

        crewai_event_bus.emit(
            self,
            event=LLMCallStartedEvent(
                messages=messages,
                tools=tools,
                callbacks=callbacks,
                available_functions=available_functions,
                from_task=from_task,
                from_agent=from_agent,
                model=self.model,
            ),
        )

    def _emit_call_completed_event(
        self,
        response: Any,
        call_type: LLMCallType,
        from_task: Task | None = None,
        from_agent: Agent | None = None,
        messages: str | list[dict[str, Any]] | None = None,
    ) -> None:
        """Emit LLM call completed event."""
        crewai_event_bus.emit(
            self,
            event=LLMCallCompletedEvent(
                messages=messages,
                response=response,
                call_type=call_type,
                from_task=from_task,
                from_agent=from_agent,
                model=self.model,
            ),
        )

    def _emit_call_failed_event(
        self,
        error: str,
        from_task: Task | None = None,
        from_agent: Agent | None = None,
    ) -> None:
        """Emit LLM call failed event."""
        if not hasattr(crewai_event_bus, "emit"):
            raise ValueError("crewai_event_bus does not have an emit method") from None

        crewai_event_bus.emit(
            self,
            event=LLMCallFailedEvent(
                error=error,
                from_task=from_task,
                from_agent=from_agent,
            ),
        )

    def _emit_stream_chunk_event(
        self,
        chunk: str,
        from_task: Task | None = None,
        from_agent: Agent | None = None,
        tool_call: dict[str, Any] | None = None,
    ) -> None:
        """Emit stream chunk event."""
        if not hasattr(crewai_event_bus, "emit"):
            raise ValueError("crewai_event_bus does not have an emit method") from None

        crewai_event_bus.emit(
            self,
            event=LLMStreamChunkEvent(
                chunk=chunk,
                tool_call=tool_call,
                from_task=from_task,
                from_agent=from_agent,
            ),
        )

    def _handle_tool_execution(
        self,
        function_name: str,
        function_args: dict[str, Any],
        available_functions: dict[str, Any],
        from_task: Task | None = None,
        from_agent: Agent | None = None,
    ) -> str | None:
        """Handle tool execution with proper event emission.

        Args:
            function_name: Name of the function to execute
            function_args: Arguments to pass to the function
            available_functions: Dict of available functions
            from_task: Optional task object
            from_agent: Optional agent object

        Returns:
            Result of function execution or None if function not found
        """
        if function_name not in available_functions:
            logging.warning(
                f"Function '{function_name}' not found in available functions"
            )
            return None

        try:
            # Emit tool usage started event
            started_at = datetime.now()

            crewai_event_bus.emit(
                self,
                event=ToolUsageStartedEvent(
                    tool_name=function_name,
                    tool_args=function_args,
                    from_agent=from_agent,
                    from_task=from_task,
                ),
            )

            # Execute the function
            fn = available_functions[function_name]
            result = fn(**function_args)

            # Emit tool usage finished event
            crewai_event_bus.emit(
                self,
                event=ToolUsageFinishedEvent(
                    output=result,
                    tool_name=function_name,
                    tool_args=function_args,
                    started_at=started_at,
                    finished_at=datetime.now(),
                    from_task=from_task,
                    from_agent=from_agent,
                ),
            )

            # Emit LLM call completed event for tool call
            self._emit_call_completed_event(
                response=result,
                call_type=LLMCallType.TOOL_CALL,
                from_task=from_task,
                from_agent=from_agent,
            )

            return str(result)

        except Exception as e:
            error_msg = f"Error executing function '{function_name}': {e!s}"
            logging.error(error_msg)

            # Emit tool usage error event
            if not hasattr(crewai_event_bus, "emit"):
                raise ValueError(
                    "crewai_event_bus does not have an emit method"
                ) from None

            crewai_event_bus.emit(
                self,
                event=ToolUsageErrorEvent(
                    tool_name=function_name,
                    tool_args=function_args,
                    error=error_msg,
                    from_task=from_task,
                    from_agent=from_agent,
                ),
            )

            # Emit LLM call failed event
            self._emit_call_failed_event(
                error=error_msg,
                from_task=from_task,
                from_agent=from_agent,
            )

            return None

    def _format_messages(self, messages: str | list[LLMMessage]) -> list[LLMMessage]:
        """Convert messages to standard format.

        Args:
            messages: Input messages (string or list of message dicts)

        Returns:
            List of message dictionaries with 'role' and 'content' keys

        Raises:
            ValueError: If message format is invalid
        """
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]

        # Validate message format
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                raise ValueError(f"Message at index {i} must be a dictionary")
            if "role" not in msg or "content" not in msg:
                raise ValueError(
                    f"Message at index {i} must have 'role' and 'content' keys"
                )

        return messages

    @staticmethod
    def _validate_structured_output(
        response: str,
        response_format: type[BaseModel] | None,
    ) -> str | BaseModel:
        """Validate and parse structured output.

        Args:
            response: Raw response string
            response_format: Optional Pydantic model for structured output

        Returns:
            Parsed response (BaseModel instance if response_format provided, otherwise string)

        Raises:
            ValueError: If structured output validation fails
        """
        if response_format is None:
            return response

        try:
            # Try to parse as JSON first
            if response.strip().startswith("{") or response.strip().startswith("["):
                data = json.loads(response)
                return response_format.model_validate(data)

            json_match = _JSON_EXTRACTION_PATTERN.search(response)
            if json_match:
                data = json.loads(json_match.group())
                return response_format.model_validate(data)

            raise ValueError("No JSON found in response")

        except (json.JSONDecodeError, ValueError) as e:
            logging.warning(f"Failed to parse structured output: {e}")
            raise ValueError(
                f"Failed to parse response into {response_format.__name__}: {e}"
            ) from e

    @staticmethod
    def _extract_provider(model: str) -> str:
        """Extract provider from model string.

        Args:
            model: Model string (e.g., 'openai/gpt-4' or 'gpt-4')

        Returns:
            Provider name (e.g., 'openai')
        """
        if "/" in model:
            return model.partition("/")[0]
        return "openai"  # Default provider

    def _track_token_usage_internal(self, usage_data: dict[str, Any]) -> None:
        """Track token usage internally in the LLM instance.

        Args:
            usage_data: Token usage data from the API response
        """
        # Extract tokens in a provider-agnostic way
        prompt_tokens = (
            usage_data.get("prompt_tokens")
            or usage_data.get("prompt_token_count")
            or usage_data.get("input_tokens")
            or 0
        )

        completion_tokens = (
            usage_data.get("completion_tokens")
            or usage_data.get("candidates_token_count")
            or usage_data.get("output_tokens")
            or 0
        )

        cached_tokens = (
            usage_data.get("cached_tokens")
            or usage_data.get("cached_prompt_tokens")
            or 0
        )

        self._token_usage["prompt_tokens"] += prompt_tokens
        self._token_usage["completion_tokens"] += completion_tokens
        self._token_usage["total_tokens"] += prompt_tokens + completion_tokens
        self._token_usage["successful_requests"] += 1
        self._token_usage["cached_prompt_tokens"] += cached_tokens

    def get_token_usage_summary(self) -> UsageMetrics:
        """Get summary of token usage for this LLM instance.

        Returns:
            Dictionary with token usage totals
        """
        return UsageMetrics(**self._token_usage)
