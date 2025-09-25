import json
import logging
import os
from typing import Any

from crewai.events.types.llm_events import LLMCallType
from crewai.llms.base_llm import BaseLLM
from crewai.utilities.agent_utils import is_context_length_exceeded
from crewai.utilities.exceptions.context_window_exceeding_exception import (
    LLMContextLengthExceededExceptionError,
)

try:
    from anthropic import Anthropic
    from anthropic.types import Message
    from anthropic.types.tool_use_block import ToolUseBlock
except ImportError:
    raise ImportError(
        "Anthropic native provider not available, to install: `uv add anthropic`"
    ) from None


class AnthropicCompletion(BaseLLM):
    """Anthropic native completion implementation.

    This class provides direct integration with the Anthropic Python SDK,
    offering native tool use, streaming support, and proper message formatting.
    """

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int = 2,
        temperature: float | None = None,
        max_tokens: int = 4096,  # Required for Anthropic
        top_p: float | None = None,
        stop_sequences: list[str] | None = None,
        stream: bool = False,
        **kwargs,
    ):
        """Initialize Anthropic chat completion client.

        Args:
            model: Anthropic model name (e.g., 'claude-3-5-sonnet-20241022')
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            base_url: Custom base URL for Anthropic API
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens in response (required for Anthropic)
            top_p: Nucleus sampling parameter
            stop_sequences: Stop sequences (Anthropic uses stop_sequences, not stop)
            stream: Enable streaming responses
            **kwargs: Additional parameters
        """
        super().__init__(
            model=model, temperature=temperature, stop=stop_sequences or [], **kwargs
        )

        # Initialize Anthropic client
        self.client = Anthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

        # Store completion parameters
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.stream = stream
        self.stop_sequences = stop_sequences or []

        # Model-specific settings
        self.is_claude_3 = "claude-3" in model.lower()
        self.supports_tools = self.is_claude_3  # Claude 3+ supports tool use

    def call(
        self,
        messages: str | list[dict[str, str]],
        tools: list[dict] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
    ) -> str | Any:
        """Call Anthropic messages API.

        Args:
            messages: Input messages for the chat completion
            tools: List of tool/function definitions
            callbacks: Callback functions (not used in native implementation)
            available_functions: Available functions for tool calling
            from_task: Task that initiated the call
            from_agent: Agent that initiated the call

        Returns:
            Chat completion response or tool call result
        """
        try:
            # Emit call started event
            self._emit_call_started_event(
                messages=messages,
                tools=tools,
                callbacks=callbacks,
                available_functions=available_functions,
                from_task=from_task,
                from_agent=from_agent,
            )

            # Format messages for Anthropic
            formatted_messages, system_message = self._format_messages_for_anthropic(
                messages
            )

            # Prepare completion parameters
            completion_params = self._prepare_completion_params(
                formatted_messages, system_message, tools
            )

            # Handle streaming vs non-streaming
            if self.stream:
                return self._handle_streaming_completion(
                    completion_params, available_functions, from_task, from_agent
                )

            return self._handle_completion(
                completion_params, available_functions, from_task, from_agent
            )

        except Exception as e:
            error_msg = f"Anthropic API call failed: {e!s}"
            logging.error(error_msg)
            self._emit_call_failed_event(
                error=error_msg, from_task=from_task, from_agent=from_agent
            )
            raise

    def _prepare_completion_params(
        self,
        messages: list[dict[str, str]],
        system_message: str | None = None,
        tools: list[dict] | None = None,
    ) -> dict[str, Any]:
        """Prepare parameters for Anthropic messages API.

        Args:
            messages: Formatted messages for Anthropic
            system_message: Extracted system message
            tools: Tool definitions

        Returns:
            Parameters dictionary for Anthropic API
        """
        params = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "stream": self.stream,
        }

        # Add system message if present
        if system_message:
            params["system"] = system_message

        # Add optional parameters if set
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.top_p is not None:
            params["top_p"] = self.top_p
        if self.stop_sequences:
            params["stop_sequences"] = self.stop_sequences

        # Handle tools for Claude 3+
        if tools and self.supports_tools:
            params["tools"] = self._convert_tools_for_interference(tools)

        return params

    def _convert_tools_for_interference(self, tools: list[dict]) -> list[dict]:
        """Convert CrewAI tool format to Anthropic tool use format."""
        from crewai.llms.providers.utils.common import safe_tool_conversion

        anthropic_tools = []

        for tool in tools:
            name, description, parameters = safe_tool_conversion(tool, "Anthropic")

            anthropic_tool = {
                "name": name,
                "description": description,
            }

            if parameters and isinstance(parameters, dict):
                anthropic_tool["input_schema"] = parameters  # type: ignore

            anthropic_tools.append(anthropic_tool)

        return anthropic_tools

    def _format_messages_for_anthropic(
        self, messages: str | list[dict[str, str]]
    ) -> tuple[list[dict[str, str]], str | None]:
        """Format messages for Anthropic API.

        Anthropic has specific requirements:
        - System messages are separate from conversation messages
        - Messages must alternate between user and assistant
        - First message must be from user

        Args:
            messages: Input messages

        Returns:
            Tuple of (formatted_messages, system_message)
        """
        # Use base class formatting first
        base_formatted = super()._format_messages(messages)

        formatted_messages = []
        system_message = None

        for message in base_formatted:
            role = message.get("role")
            content = message.get("content", "")

            if role == "system":
                # Extract system message - Anthropic handles it separately
                if system_message:
                    system_message += f"\n\n{content}"
                else:
                    system_message = content
            else:
                # Add user/assistant messages - ensure both role and content are str, not None
                role_str = role if role is not None else "user"
                content_str = content if content is not None else ""
                formatted_messages.append({"role": role_str, "content": content_str})

        # Ensure first message is from user (Anthropic requirement)
        if not formatted_messages:
            # If no messages, add a default user message
            formatted_messages.append({"role": "user", "content": "Hello"})
        elif formatted_messages[0]["role"] != "user":
            # If first message is not from user, insert a user message at the beginning
            formatted_messages.insert(0, {"role": "user", "content": "Hello"})

        return formatted_messages, system_message

    def _handle_completion(
        self,
        params: dict[str, Any],
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
    ) -> str | Any:
        """Handle non-streaming message completion."""
        try:
            response: Message = self.client.messages.create(**params)

        except Exception as e:
            if is_context_length_exceeded(e):
                logging.error(f"Context window exceeded: {e}")
                raise LLMContextLengthExceededExceptionError(str(e)) from e
            raise e from e

        usage = self._extract_anthropic_token_usage(response)
        self._track_token_usage_internal(usage)

        if response.content and available_functions:
            for content_block in response.content:
                if isinstance(content_block, ToolUseBlock):
                    function_name = content_block.name
                    function_args = content_block.input

                    result = self._handle_tool_execution(
                        function_name=function_name,
                        function_args=function_args,  # type: ignore
                        available_functions=available_functions,
                        from_task=from_task,
                        from_agent=from_agent,
                    )

                    if result is not None:
                        return result

        # Extract text content
        content = ""
        if response.content:
            for content_block in response.content:
                if hasattr(content_block, "text"):
                    content += content_block.text

        content = self._apply_stop_words(content)

        self._emit_call_completed_event(
            response=content,
            call_type=LLMCallType.LLM_CALL,
            from_task=from_task,
            from_agent=from_agent,
            messages=params["messages"],
        )

        if usage.get("total_tokens", 0) > 0:
            logging.info(f"Anthropic API usage: {usage}")

        return content

    def _handle_streaming_completion(
        self,
        params: dict[str, Any],
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
    ) -> str:
        """Handle streaming message completion."""
        full_response = ""
        tool_uses = {}

        # Make streaming API call
        with self.client.messages.stream(**params) as stream:
            for event in stream:
                # Handle content delta events
                if hasattr(event, "delta") and hasattr(event.delta, "text"):
                    text_delta = event.delta.text
                    full_response += text_delta
                    self._emit_stream_chunk_event(
                        chunk=text_delta,
                        from_task=from_task,
                        from_agent=from_agent,
                    )

                # Handle tool use events
                elif hasattr(event, "delta") and hasattr(event.delta, "partial_json"):
                    # Tool use streaming - accumulate JSON
                    tool_id = getattr(event, "index", "default")
                    if tool_id not in tool_uses:
                        tool_uses[tool_id] = {
                            "name": "",
                            "input": "",
                        }

                    if hasattr(event.delta, "name"):
                        tool_uses[tool_id]["name"] = event.delta.name
                    if hasattr(event.delta, "partial_json"):
                        tool_uses[tool_id]["input"] += event.delta.partial_json

        # Handle completed tool uses
        if tool_uses and available_functions:
            for tool_data in tool_uses.values():
                function_name = tool_data["name"]

                try:
                    function_args = json.loads(tool_data["input"])
                except json.JSONDecodeError as e:
                    logging.error(f"Failed to parse streamed tool arguments: {e}")
                    continue

                # Execute tool
                result = self._handle_tool_execution(
                    function_name=function_name,
                    function_args=function_args,
                    available_functions=available_functions,
                    from_task=from_task,
                    from_agent=from_agent,
                )

                if result is not None:
                    return result

        # Apply stop words to full response
        full_response = self._apply_stop_words(full_response)

        # Emit completion event and return full response
        self._emit_call_completed_event(
            response=full_response,
            call_type=LLMCallType.LLM_CALL,
            from_task=from_task,
            from_agent=from_agent,
            messages=params["messages"],
        )

        return full_response

    def supports_function_calling(self) -> bool:
        """Check if the model supports function calling."""
        return self.supports_tools

    def supports_stop_words(self) -> bool:
        """Check if the model supports stop words."""
        return True  # All Claude models support stop sequences

    def get_context_window_size(self) -> int:
        """Get the context window size for the model."""
        from crewai.llm import CONTEXT_WINDOW_USAGE_RATIO

        # Context window sizes for Anthropic models
        context_windows = {
            "claude-3-5-sonnet": 200000,
            "claude-3-5-haiku": 200000,
            "claude-3-opus": 200000,
            "claude-3-sonnet": 200000,
            "claude-3-haiku": 200000,
            "claude-3-7-sonnet": 200000,
            "claude-2.1": 200000,
            "claude-2": 100000,
            "claude-instant": 100000,
        }

        # Find the best match for the model name
        for model_prefix, size in context_windows.items():
            if self.model.startswith(model_prefix):
                return int(size * CONTEXT_WINDOW_USAGE_RATIO)

        # Default context window size for Claude models
        return int(200000 * CONTEXT_WINDOW_USAGE_RATIO)

    def _extract_anthropic_token_usage(self, response: Message) -> dict[str, Any]:
        """Extract token usage from Anthropic response."""
        if hasattr(response, "usage") and response.usage:
            usage = response.usage
            input_tokens = getattr(usage, "input_tokens", 0)
            output_tokens = getattr(usage, "output_tokens", 0)
            return {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            }
        return {"total_tokens": 0}
