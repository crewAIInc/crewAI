from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, Field, PrivateAttr, model_validator
from typing_extensions import Self

from crewai.events.types.llm_events import LLMCallType
from crewai.llm.base_llm import BaseLLM
from crewai.llm.core import CONTEXT_WINDOW_USAGE_RATIO
from crewai.llm.hooks.transport import HTTPTransport
from crewai.llm.providers.utils.common import safe_tool_conversion
from crewai.utilities.agent_utils import is_context_length_exceeded
from crewai.utilities.exceptions.context_window_exceeding_exception import (
    LLMContextLengthExceededError,
)
from crewai.utilities.types import LLMMessage


if TYPE_CHECKING:
    from crewai.agent.core import Agent
    from crewai.task import Task


try:
    from anthropic import Anthropic
    from anthropic.types import Message
    from anthropic.types.tool_use_block import ToolUseBlock
    import httpx
except ImportError:
    raise ImportError(
        'Anthropic native provider not available, to install: uv add "crewai[anthropic]"'
    ) from None


class AnthropicCompletion(BaseLLM):
    """Anthropic native completion implementation.

    This class provides direct integration with the Anthropic Python SDK,
    offering native tool use, streaming support, and proper message formatting.

    Attributes:
        model: Anthropic model name (e.g., 'claude-3-5-sonnet-20241022')
        base_url: Custom base URL for Anthropic API
        timeout: Request timeout in seconds
        max_retries: Maximum number of retries
        max_tokens: Maximum tokens in response (required for Anthropic)
        top_p: Nucleus sampling parameter
        stream: Enable streaming responses
        client_params: Additional parameters for the Anthropic client
        interceptor: HTTP interceptor for modifying requests/responses at transport level
    """

    base_url: str | None = Field(
        default=None, description="Custom base URL for Anthropic API"
    )
    timeout: float | None = Field(
        default=None, description="Request timeout in seconds"
    )
    max_retries: int = Field(default=2, description="Maximum number of retries")
    max_tokens: int = Field(
        default=4096, description="Maximum tokens in response (required for Anthropic)"
    )
    top_p: float | None = Field(default=None, description="Nucleus sampling parameter")
    stream: bool = Field(default=False, description="Enable streaming responses")
    client_params: dict[str, Any] | None = Field(
        default=None, description="Additional Anthropic client parameters"
    )
    client: Anthropic = Field(
        default_factory=Anthropic, exclude=True, description="Anthropic client instance"
    )

    _is_claude_3: bool = PrivateAttr(default=False)
    _supports_tools: bool = PrivateAttr(default=False)

    @model_validator(mode="after")
    def setup_client(self) -> Self:
        """Initialize the Anthropic client and model-specific settings."""
        self.client = Anthropic(**self._get_client_params())

        self._is_claude_3 = "claude-3" in self.model.lower()
        self._supports_tools = self._is_claude_3

        return self

    @property
    def is_claude_3(self) -> bool:
        """Check if model is Claude 3."""
        return self._is_claude_3

    @property
    def supports_tools(self) -> bool:
        """Check if model supports tools."""
        return self._supports_tools

    def _get_client_params(self) -> dict[str, Any]:
        """Get client parameters."""

        if self.api_key is None:
            self.api_key = os.getenv("ANTHROPIC_API_KEY")
            if self.api_key is None:
                raise ValueError("ANTHROPIC_API_KEY is required")

        client_params = {
            "api_key": self.api_key,
            "base_url": self.base_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }

        if self.interceptor:
            transport = HTTPTransport(interceptor=self.interceptor)
            http_client = httpx.Client(transport=transport)
            client_params["http_client"] = http_client  # type: ignore[assignment]

        if self.client_params:
            client_params.update(self.client_params)

        return client_params

    def call(
        self,
        messages: str | list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Task | None = None,
        from_agent: Agent | None = None,
        response_model: type[BaseModel] | None = None,
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
                    completion_params,
                    available_functions,
                    from_task,
                    from_agent,
                    response_model,
                )

            return self._handle_completion(
                completion_params,
                available_functions,
                from_task,
                from_agent,
                response_model,
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
        messages: list[LLMMessage],
        system_message: str | None = None,
        tools: list[dict[str, Any]] | None = None,
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
        if self.stop:
            params["stop_sequences"] = self.stop

        # Handle tools for Claude 3+
        if tools and self.supports_tools:
            params["tools"] = self._convert_tools_for_interference(tools)

        return params

    def _convert_tools_for_interference(
        self, tools: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Convert CrewAI tool format to Anthropic tool use format."""
        anthropic_tools = []

        for tool in tools:
            if "input_schema" in tool and "name" in tool and "description" in tool:
                anthropic_tools.append(tool)
                continue

            try:
                name, description, parameters = safe_tool_conversion(tool, "Anthropic")
            except (ImportError, KeyError, ValueError) as e:
                logging.error(f"Error converting tool to Anthropic format: {e}")
                raise e

            anthropic_tool = {
                "name": name,
                "description": description,
            }

            if parameters and isinstance(parameters, dict):
                anthropic_tool["input_schema"] = parameters  # type: ignore[assignment]
            else:
                anthropic_tool["input_schema"] = {  # type: ignore[assignment]
                    "type": "object",
                    "properties": {},
                    "required": [],
                }

            anthropic_tools.append(anthropic_tool)

        return anthropic_tools

    def _format_messages_for_anthropic(
        self, messages: str | list[LLMMessage]
    ) -> tuple[list[LLMMessage], str | None]:
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

        formatted_messages: list[LLMMessage] = []
        system_message: str | None = None

        for message in base_formatted:
            role = message.get("role")
            content = message.get("content", "")

            if role == "system":
                if system_message:
                    system_message += f"\n\n{content}"
                else:
                    system_message = cast(str, content)
            else:
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
        from_task: Task | None = None,
        from_agent: Agent | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        """Handle non-streaming message completion."""
        if response_model:
            structured_tool = {
                "name": "structured_output",
                "description": "Returns structured data according to the schema",
                "input_schema": response_model.model_json_schema(),
            }

            params["tools"] = [structured_tool]
            params["tool_choice"] = {"type": "tool", "name": "structured_output"}

        try:
            response: Message = self.client.messages.create(**params)

        except Exception as e:
            if is_context_length_exceeded(e):
                logging.error(f"Context window exceeded: {e}")
                raise LLMContextLengthExceededError(str(e)) from e
            raise e from e

        usage = self._extract_anthropic_token_usage(response)
        self._track_token_usage_internal(usage)

        if response_model and response.content:
            tool_uses = [
                block for block in response.content if isinstance(block, ToolUseBlock)
            ]
            if tool_uses and tool_uses[0].name == "structured_output":
                structured_data = tool_uses[0].input
                structured_json = json.dumps(structured_data)

                self._emit_call_completed_event(
                    response=structured_json,
                    call_type=LLMCallType.LLM_CALL,
                    from_task=from_task,
                    from_agent=from_agent,
                    messages=params["messages"],
                )

                return structured_json

        # Check if Claude wants to use tools
        if response.content and available_functions:
            tool_uses = [
                block for block in response.content if isinstance(block, ToolUseBlock)
            ]

            if tool_uses:
                # Handle tool use conversation flow
                return self._handle_tool_use_conversation(
                    response,
                    tool_uses,
                    params,
                    available_functions,
                    from_task,
                    from_agent,
                )

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
        from_task: Task | None = None,
        from_agent: Agent | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str:
        """Handle streaming message completion."""
        if response_model:
            structured_tool = {
                "name": "structured_output",
                "description": "Returns structured data according to the schema",
                "input_schema": response_model.model_json_schema(),
            }

            params["tools"] = [structured_tool]
            params["tool_choice"] = {"type": "tool", "name": "structured_output"}

        full_response = ""

        # Remove 'stream' parameter as messages.stream() doesn't accept it
        # (the SDK sets it internally)
        stream_params = {k: v for k, v in params.items() if k != "stream"}

        # Make streaming API call
        with self.client.messages.stream(**stream_params) as stream:
            for event in stream:
                if hasattr(event, "delta") and hasattr(event.delta, "text"):
                    text_delta = event.delta.text
                    full_response += text_delta
                    self._emit_stream_chunk_event(
                        chunk=text_delta,
                        from_task=from_task,
                        from_agent=from_agent,
                    )

            final_message: Message = stream.get_final_message()

        usage = self._extract_anthropic_token_usage(final_message)
        self._track_token_usage_internal(usage)

        if response_model and final_message.content:
            tool_uses = [
                block
                for block in final_message.content
                if isinstance(block, ToolUseBlock)
            ]
            if tool_uses and tool_uses[0].name == "structured_output":
                structured_data = tool_uses[0].input
                structured_json = json.dumps(structured_data)

                self._emit_call_completed_event(
                    response=structured_json,
                    call_type=LLMCallType.LLM_CALL,
                    from_task=from_task,
                    from_agent=from_agent,
                    messages=params["messages"],
                )

                return structured_json

        if final_message.content and available_functions:
            tool_uses = [
                block
                for block in final_message.content
                if isinstance(block, ToolUseBlock)
            ]

            if tool_uses:
                # Handle tool use conversation flow
                return self._handle_tool_use_conversation(
                    final_message,
                    tool_uses,
                    params,
                    available_functions,
                    from_task,
                    from_agent,
                )

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

    def _handle_tool_use_conversation(
        self,
        initial_response: Message,
        tool_uses: list[ToolUseBlock],
        params: dict[str, Any],
        available_functions: dict[str, Any],
        from_task: Task | None = None,
        from_agent: Agent | None = None,
    ) -> str:
        """Handle the complete tool use conversation flow.

        This implements the proper Anthropic tool use pattern:
        1. Claude requests tool use
        2. We execute the tools
        3. We send tool results back to Claude
        4. Claude processes results and generates final response
        """
        # Execute all requested tools and collect results
        tool_results = []

        for tool_use in tool_uses:
            function_name = tool_use.name
            function_args = tool_use.input

            # Execute the tool
            result = self._handle_tool_execution(
                function_name=function_name,
                function_args=function_args,  # type: ignore
                available_functions=available_functions,
                from_task=from_task,
                from_agent=from_agent,
            )

            # Create tool result in Anthropic format
            tool_result = {
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": str(result)
                if result is not None
                else "Tool execution completed",
            }
            tool_results.append(tool_result)

        # Prepare follow-up conversation with tool results
        follow_up_params = params.copy()

        # Add Claude's tool use response to conversation
        assistant_message = {"role": "assistant", "content": initial_response.content}

        # Add user message with tool results
        user_message = {"role": "user", "content": tool_results}

        # Update messages for follow-up call
        follow_up_params["messages"] = params["messages"] + [
            assistant_message,
            user_message,
        ]

        try:
            # Send tool results back to Claude for final response
            final_response: Message = self.client.messages.create(**follow_up_params)

            # Track token usage for follow-up call
            follow_up_usage = self._extract_anthropic_token_usage(final_response)
            self._track_token_usage_internal(follow_up_usage)

            # Extract final text content
            final_content = ""
            if final_response.content:
                for content_block in final_response.content:
                    if hasattr(content_block, "text"):
                        final_content += content_block.text

            final_content = self._apply_stop_words(final_content)

            # Emit completion event for the final response
            self._emit_call_completed_event(
                response=final_content,
                call_type=LLMCallType.LLM_CALL,
                from_task=from_task,
                from_agent=from_agent,
                messages=follow_up_params["messages"],
            )

            # Log combined token usage
            total_usage = {
                "input_tokens": follow_up_usage.get("input_tokens", 0),
                "output_tokens": follow_up_usage.get("output_tokens", 0),
                "total_tokens": follow_up_usage.get("total_tokens", 0),
            }

            if total_usage.get("total_tokens", 0) > 0:
                logging.info(f"Anthropic API tool conversation usage: {total_usage}")

            return final_content

        except Exception as e:
            if is_context_length_exceeded(e):
                logging.error(f"Context window exceeded in tool follow-up: {e}")
                raise LLMContextLengthExceededError(str(e)) from e

            logging.error(f"Tool follow-up conversation failed: {e}")
            # Fallback: return the first tool result if follow-up fails
            if tool_results:
                return tool_results[0]["content"]
            raise e

    def supports_function_calling(self) -> bool:
        """Check if the model supports function calling."""
        return self.supports_tools

    def supports_stop_words(self) -> bool:
        """Check if the model supports stop words."""
        return True  # All Claude models support stop sequences

    def get_context_window_size(self) -> int:
        """Get the context window size for the model."""

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
