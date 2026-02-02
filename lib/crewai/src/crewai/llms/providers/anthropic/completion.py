from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Any, Final, Literal, TypeGuard, cast

from pydantic import BaseModel

from crewai.events.types.llm_events import LLMCallType
from crewai.llms.base_llm import BaseLLM
from crewai.llms.hooks.transport import AsyncHTTPTransport, HTTPTransport
from crewai.utilities.agent_utils import is_context_length_exceeded
from crewai.utilities.exceptions.context_window_exceeding_exception import (
    LLMContextLengthExceededError,
)
from crewai.utilities.types import LLMMessage


if TYPE_CHECKING:
    from crewai.llms.hooks.base import BaseInterceptor

try:
    from anthropic import Anthropic, AsyncAnthropic, transform_schema
    from anthropic.types import Message, TextBlock, ThinkingBlock, ToolUseBlock
    from anthropic.types.beta import BetaMessage, BetaTextBlock, BetaToolUseBlock
    import httpx
except ImportError:
    raise ImportError(
        'Anthropic native provider not available, to install: uv add "crewai[anthropic]"'
    ) from None


ANTHROPIC_FILES_API_BETA: Final = "files-api-2025-04-14"
ANTHROPIC_STRUCTURED_OUTPUTS_BETA: Final = "structured-outputs-2025-11-13"

NATIVE_STRUCTURED_OUTPUT_MODELS: Final[
    tuple[
        Literal["claude-sonnet-4-5"],
        Literal["claude-sonnet-4.5"],
        Literal["claude-opus-4-5"],
        Literal["claude-opus-4.5"],
        Literal["claude-opus-4-1"],
        Literal["claude-opus-4.1"],
        Literal["claude-haiku-4-5"],
        Literal["claude-haiku-4.5"],
    ]
] = (
    "claude-sonnet-4-5",
    "claude-sonnet-4.5",
    "claude-opus-4-5",
    "claude-opus-4.5",
    "claude-opus-4-1",
    "claude-opus-4.1",
    "claude-haiku-4-5",
    "claude-haiku-4.5",
)


def _supports_native_structured_outputs(model: str) -> bool:
    """Check if the model supports native structured outputs.

    Native structured outputs are only available for Claude 4.5 models
    (Sonnet 4.5, Opus 4.5, Opus 4.1, Haiku 4.5).
    Other models require the tool-based fallback approach.

    Args:
        model: The model name/identifier.

    Returns:
        True if the model supports native structured outputs.
    """
    model_lower = model.lower()
    return any(prefix in model_lower for prefix in NATIVE_STRUCTURED_OUTPUT_MODELS)


def _is_pydantic_model_class(obj: Any) -> TypeGuard[type[BaseModel]]:
    """Check if an object is a Pydantic model class.

    This distinguishes between Pydantic model classes that support structured
    outputs (have model_json_schema) and plain dicts like {"type": "json_object"}.

    Args:
        obj: The object to check.

    Returns:
        True if obj is a Pydantic model class.
    """
    return isinstance(obj, type) and issubclass(obj, BaseModel)


def _contains_file_id_reference(messages: list[dict[str, Any]]) -> bool:
    """Check if any message content contains a file_id reference.

    Anthropic's Files API is in beta and requires a special header when
    file_id references are used in content blocks.

    Args:
        messages: List of message dicts to check.

    Returns:
        True if any content block contains a file_id reference.
    """
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    source = block.get("source", {})
                    if isinstance(source, dict) and source.get("type") == "file":
                        return True
    return False


class AnthropicThinkingConfig(BaseModel):
    type: Literal["enabled", "disabled"]
    budget_tokens: int | None = None


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
        client_params: dict[str, Any] | None = None,
        interceptor: BaseInterceptor[httpx.Request, httpx.Response] | None = None,
        thinking: AnthropicThinkingConfig | None = None,
        response_format: type[BaseModel] | None = None,
        **kwargs: Any,
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
            client_params: Additional parameters for the Anthropic client
            interceptor: HTTP interceptor for modifying requests/responses at transport level.
            response_format: Pydantic model for structured output. When provided, responses
                will be validated against this model schema.
            **kwargs: Additional parameters
        """
        super().__init__(
            model=model, temperature=temperature, stop=stop_sequences or [], **kwargs
        )

        # Client params
        self.interceptor = interceptor
        self.client_params = client_params
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries

        self.client = Anthropic(**self._get_client_params())

        async_client_params = self._get_client_params()
        if self.interceptor:
            async_transport = AsyncHTTPTransport(interceptor=self.interceptor)
            async_http_client = httpx.AsyncClient(transport=async_transport)
            async_client_params["http_client"] = async_http_client

        self.async_client = AsyncAnthropic(**async_client_params)

        # Store completion parameters
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.stream = stream
        self.stop_sequences = stop_sequences or []
        self.thinking = thinking
        self.previous_thinking_blocks: list[ThinkingBlock] = []
        self.response_format = response_format
        # Model-specific settings
        self.is_claude_3 = "claude-3" in model.lower()
        self.supports_tools = True

    @property
    def stop(self) -> list[str]:
        """Get stop sequences sent to the API."""
        return self.stop_sequences

    @stop.setter
    def stop(self, value: list[str] | str | None) -> None:
        """Set stop sequences.

        Synchronizes stop_sequences to ensure values set by CrewAgentExecutor
        are properly sent to the Anthropic API.

        Args:
            value: Stop sequences as a list, single string, or None
        """
        if value is None:
            self.stop_sequences = []
        elif isinstance(value, str):
            self.stop_sequences = [value]
        elif isinstance(value, list):
            self.stop_sequences = value
        else:
            self.stop_sequences = []

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
        from_task: Any | None = None,
        from_agent: Any | None = None,
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

            if not self._invoke_before_llm_call_hooks(formatted_messages, from_agent):
                raise ValueError("LLM call blocked by before_llm_call hook")

            # Prepare completion parameters
            completion_params = self._prepare_completion_params(
                formatted_messages, system_message, tools
            )

            effective_response_model = response_model or self.response_format

            # Handle streaming vs non-streaming
            if self.stream:
                return self._handle_streaming_completion(
                    completion_params,
                    available_functions,
                    from_task,
                    from_agent,
                    effective_response_model,
                )

            return self._handle_completion(
                completion_params,
                available_functions,
                from_task,
                from_agent,
                effective_response_model,
            )

        except Exception as e:
            error_msg = f"Anthropic API call failed: {e!s}"
            logging.error(error_msg)
            self._emit_call_failed_event(
                error=error_msg, from_task=from_task, from_agent=from_agent
            )
            raise

    async def acall(
        self,
        messages: str | list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        """Async call to Anthropic messages API.

        Args:
            messages: Input messages for the chat completion
            tools: List of tool/function definitions
            callbacks: Callback functions (not used in native implementation)
            available_functions: Available functions for tool calling
            from_task: Task that initiated the call
            from_agent: Agent that initiated the call
            response_model: Optional response model.

        Returns:
            Chat completion response or tool call result
        """
        try:
            self._emit_call_started_event(
                messages=messages,
                tools=tools,
                callbacks=callbacks,
                available_functions=available_functions,
                from_task=from_task,
                from_agent=from_agent,
            )

            formatted_messages, system_message = self._format_messages_for_anthropic(
                messages
            )

            completion_params = self._prepare_completion_params(
                formatted_messages, system_message, tools
            )

            effective_response_model = response_model or self.response_format

            if self.stream:
                return await self._ahandle_streaming_completion(
                    completion_params,
                    available_functions,
                    from_task,
                    from_agent,
                    effective_response_model,
                )

            return await self._ahandle_completion(
                completion_params,
                available_functions,
                from_task,
                from_agent,
                effective_response_model,
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
        if self.stop_sequences:
            params["stop_sequences"] = self.stop_sequences

        # Handle tools for Claude 3+
        if tools and self.supports_tools:
            params["tools"] = self._convert_tools_for_interference(tools)

        if self.thinking:
            if isinstance(self.thinking, AnthropicThinkingConfig):
                params["thinking"] = self.thinking.model_dump()
            else:
                params["thinking"] = self.thinking

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
                from crewai.llms.providers.utils.common import safe_tool_conversion

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

    def _extract_thinking_block(
        self, content_block: Any
    ) -> ThinkingBlock | dict[str, Any] | None:
        """Extract and format thinking block from content block.

        Args:
            content_block: Content block from Anthropic response

        Returns:
            Dictionary with thinking block data including signature, or None if not a thinking block
        """
        if content_block.type == "thinking":
            thinking_block = {
                "type": "thinking",
                "thinking": content_block.thinking,
            }
            if hasattr(content_block, "signature"):
                thinking_block["signature"] = content_block.signature
            return thinking_block
        if content_block.type == "redacted_thinking":
            redacted_block = {"type": "redacted_thinking"}
            if hasattr(content_block, "thinking"):
                redacted_block["thinking"] = content_block.thinking
            if hasattr(content_block, "signature"):
                redacted_block["signature"] = content_block.signature
            return redacted_block
        return None

    def _format_messages_for_anthropic(
        self, messages: str | list[LLMMessage]
    ) -> tuple[list[LLMMessage], str | None]:
        """Format messages for Anthropic API.

        Anthropic has specific requirements:
        - System messages are separate from conversation messages
        - Messages must alternate between user and assistant
        - First message must be from user
        - Tool results must be in user messages with tool_result content blocks
        - When thinking is enabled, assistant messages must start with thinking blocks

        Args:
            messages: Input messages

        Returns:
            Tuple of (formatted_messages, system_message)
        """
        # Use base class formatting first
        base_formatted = super()._format_messages(messages)

        formatted_messages: list[LLMMessage] = []
        system_message: str | None = None
        pending_tool_results: list[dict[str, Any]] = []

        for message in base_formatted:
            role = message.get("role")
            content = message.get("content", "")

            if role == "system":
                if system_message:
                    system_message += f"\n\n{content}"
                else:
                    system_message = cast(str, content)
            elif role == "tool":
                tool_call_id = message.get("tool_call_id", "")
                if not tool_call_id:
                    raise ValueError("Tool message missing required tool_call_id")
                tool_result = {
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": content if content else "",
                }
                pending_tool_results.append(tool_result)
            elif role == "assistant":
                # First, flush any pending tool results as a user message
                if pending_tool_results:
                    formatted_messages.append(
                        {"role": "user", "content": pending_tool_results}
                    )
                    pending_tool_results = []

                # Handle assistant message with tool_calls (convert to Anthropic format)
                tool_calls = message.get("tool_calls", [])
                if tool_calls:
                    assistant_content: list[dict[str, Any]] = []
                    for tc in tool_calls:
                        if isinstance(tc, dict):
                            func = tc.get("function", {})
                            tool_use = {
                                "type": "tool_use",
                                "id": tc.get("id", ""),
                                "name": func.get("name", ""),
                                "input": json.loads(func.get("arguments", "{}"))
                                if isinstance(func.get("arguments"), str)
                                else func.get("arguments", {}),
                            }
                            assistant_content.append(tool_use)
                    if assistant_content:
                        formatted_messages.append(
                            {"role": "assistant", "content": assistant_content}
                        )
                elif isinstance(content, list):
                    formatted_messages.append({"role": "assistant", "content": content})
                elif self.thinking and self.previous_thinking_blocks:
                    structured_content = cast(
                        list[dict[str, Any]],
                        [
                            *self.previous_thinking_blocks,
                            {"type": "text", "text": content if content else ""},
                        ],
                    )
                    formatted_messages.append(
                        LLMMessage(role="assistant", content=structured_content)
                    )
                else:
                    content_str = content if content is not None else ""
                    formatted_messages.append(
                        LLMMessage(role="assistant", content=content_str)
                    )
            else:
                # User message - first flush any pending tool results
                if pending_tool_results:
                    formatted_messages.append(
                        {"role": "user", "content": pending_tool_results}
                    )
                    pending_tool_results = []

                role_str = role if role is not None else "user"
                if isinstance(content, list):
                    formatted_messages.append({"role": role_str, "content": content})
                else:
                    content_str = content if content is not None else ""
                    formatted_messages.append(
                        LLMMessage(role=role_str, content=content_str)
                    )

        # Flush any remaining pending tool results
        if pending_tool_results:
            formatted_messages.append({"role": "user", "content": pending_tool_results})

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
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        """Handle non-streaming message completion."""
        uses_file_api = _contains_file_id_reference(params.get("messages", []))
        betas: list[str] = []
        use_native_structured_output = False

        if uses_file_api:
            betas.append(ANTHROPIC_FILES_API_BETA)

        extra_body: dict[str, Any] | None = None
        if _is_pydantic_model_class(response_model):
            schema = transform_schema(response_model.model_json_schema())
            if _supports_native_structured_outputs(self.model):
                use_native_structured_output = True
                betas.append(ANTHROPIC_STRUCTURED_OUTPUTS_BETA)
                extra_body = {
                    "output_format": {
                        "type": "json_schema",
                        "schema": schema,
                    }
                }
            else:
                structured_tool = {
                    "name": "structured_output",
                    "description": "Output the structured response",
                    "input_schema": schema,
                }
                params["tools"] = [structured_tool]
                params["tool_choice"] = {"type": "tool", "name": "structured_output"}

        try:
            if betas:
                params["betas"] = betas
                response = self.client.beta.messages.create(
                    **params, extra_body=extra_body
                )
            else:
                response = self.client.messages.create(**params)

        except Exception as e:
            if is_context_length_exceeded(e):
                logging.error(f"Context window exceeded: {e}")
                raise LLMContextLengthExceededError(str(e)) from e
            raise e from e

        usage = self._extract_anthropic_token_usage(response)
        self._track_token_usage_internal(usage)

        if _is_pydantic_model_class(response_model) and response.content:
            if use_native_structured_output:
                for block in response.content:
                    if isinstance(block, (TextBlock, BetaTextBlock)):
                        structured_data = response_model.model_validate_json(block.text)
                        self._emit_call_completed_event(
                            response=structured_data.model_dump_json(),
                            call_type=LLMCallType.LLM_CALL,
                            from_task=from_task,
                            from_agent=from_agent,
                            messages=params["messages"],
                        )
                        return structured_data
            else:
                for block in response.content:
                    if (
                        isinstance(block, (ToolUseBlock, BetaToolUseBlock))
                        and block.name == "structured_output"
                    ):
                        structured_data = response_model.model_validate(block.input)
                        self._emit_call_completed_event(
                            response=structured_data.model_dump_json(),
                            call_type=LLMCallType.LLM_CALL,
                            from_task=from_task,
                            from_agent=from_agent,
                            messages=params["messages"],
                        )
                        return structured_data

        # Check if Claude wants to use tools
        if response.content:
            tool_uses = [
                block
                for block in response.content
                if isinstance(block, (ToolUseBlock, BetaToolUseBlock))
            ]

            if tool_uses:
                # If no available_functions, return tool calls for executor to handle
                # This allows the executor to manage tool execution with proper
                # message history and post-tool reasoning prompts
                if not available_functions:
                    self._emit_call_completed_event(
                        response=list(tool_uses),
                        call_type=LLMCallType.TOOL_CALL,
                        from_task=from_task,
                        from_agent=from_agent,
                        messages=params["messages"],
                    )
                    return list(tool_uses)

                # Handle tool use conversation flow internally
                return self._handle_tool_use_conversation(
                    response,
                    tool_uses,
                    params,
                    available_functions,
                    from_task,
                    from_agent,
                )

        content = ""
        thinking_blocks: list[ThinkingBlock] = []

        if response.content:
            for content_block in response.content:
                if hasattr(content_block, "text"):
                    content += content_block.text
                else:
                    thinking_block = self._extract_thinking_block(content_block)
                    if thinking_block:
                        thinking_blocks.append(cast(ThinkingBlock, thinking_block))

        if thinking_blocks:
            self.previous_thinking_blocks = thinking_blocks

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

        return self._invoke_after_llm_call_hooks(
            params["messages"], content, from_agent
        )

    def _handle_streaming_completion(
        self,
        params: dict[str, Any],
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        """Handle streaming message completion."""
        betas: list[str] = []
        use_native_structured_output = False

        extra_body: dict[str, Any] | None = None
        if _is_pydantic_model_class(response_model):
            schema = transform_schema(response_model.model_json_schema())
            if _supports_native_structured_outputs(self.model):
                use_native_structured_output = True
                betas.append(ANTHROPIC_STRUCTURED_OUTPUTS_BETA)
                extra_body = {
                    "output_format": {
                        "type": "json_schema",
                        "schema": schema,
                    }
                }
            else:
                structured_tool = {
                    "name": "structured_output",
                    "description": "Output the structured response",
                    "input_schema": schema,
                }
                params["tools"] = [structured_tool]
                params["tool_choice"] = {"type": "tool", "name": "structured_output"}

        full_response = ""

        # Remove 'stream' parameter as messages.stream() doesn't accept it
        # (the SDK sets it internally)
        stream_params = {k: v for k, v in params.items() if k != "stream"}

        if betas:
            stream_params["betas"] = betas

        current_tool_calls: dict[int, dict[str, Any]] = {}

        stream_context = (
            self.client.beta.messages.stream(**stream_params, extra_body=extra_body)
            if betas
            else self.client.messages.stream(**stream_params)
        )
        with stream_context as stream:
            response_id = None
            for event in stream:
                if hasattr(event, "message") and hasattr(event.message, "id"):
                    response_id = event.message.id

                if hasattr(event, "delta") and hasattr(event.delta, "text"):
                    text_delta = event.delta.text
                    full_response += text_delta
                    self._emit_stream_chunk_event(
                        chunk=text_delta,
                        from_task=from_task,
                        from_agent=from_agent,
                        response_id=response_id,
                    )

                if event.type == "content_block_start":
                    block = event.content_block
                    if block.type == "tool_use":
                        block_index = event.index
                        current_tool_calls[block_index] = {
                            "id": block.id,
                            "name": block.name,
                            "arguments": "",
                            "index": block_index,
                        }
                        self._emit_stream_chunk_event(
                            chunk="",
                            from_task=from_task,
                            from_agent=from_agent,
                            tool_call={
                                "id": block.id,
                                "function": {
                                    "name": block.name,
                                    "arguments": "",
                                },
                                "type": "function",
                                "index": block_index,
                            },
                            call_type=LLMCallType.TOOL_CALL,
                            response_id=response_id,
                        )
                elif event.type == "content_block_delta":
                    if event.delta.type == "input_json_delta":
                        block_index = event.index
                        partial_json = event.delta.partial_json
                        if block_index in current_tool_calls and partial_json:
                            current_tool_calls[block_index]["arguments"] += partial_json
                            self._emit_stream_chunk_event(
                                chunk=partial_json,
                                from_task=from_task,
                                from_agent=from_agent,
                                tool_call={
                                    "id": current_tool_calls[block_index]["id"],
                                    "function": {
                                        "name": current_tool_calls[block_index]["name"],
                                        "arguments": current_tool_calls[block_index][
                                            "arguments"
                                        ],
                                    },
                                    "type": "function",
                                    "index": block_index,
                                },
                                call_type=LLMCallType.TOOL_CALL,
                                response_id=response_id,
                            )

            final_message = stream.get_final_message()

        thinking_blocks: list[ThinkingBlock] = []
        if final_message.content:
            for content_block in final_message.content:
                thinking_block = self._extract_thinking_block(content_block)
                if thinking_block:
                    thinking_blocks.append(cast(ThinkingBlock, thinking_block))

        if thinking_blocks:
            self.previous_thinking_blocks = thinking_blocks

        usage = self._extract_anthropic_token_usage(final_message)
        self._track_token_usage_internal(usage)

        if _is_pydantic_model_class(response_model):
            if use_native_structured_output:
                structured_data = response_model.model_validate_json(full_response)
                self._emit_call_completed_event(
                    response=structured_data.model_dump_json(),
                    call_type=LLMCallType.LLM_CALL,
                    from_task=from_task,
                    from_agent=from_agent,
                    messages=params["messages"],
                )
                return structured_data
            for block in final_message.content:
                if (
                    isinstance(block, ToolUseBlock)
                    and block.name == "structured_output"
                ):
                    structured_data = response_model.model_validate(block.input)
                    self._emit_call_completed_event(
                        response=structured_data.model_dump_json(),
                        call_type=LLMCallType.LLM_CALL,
                        from_task=from_task,
                        from_agent=from_agent,
                        messages=params["messages"],
                    )
                    return structured_data

        if final_message.content:
            tool_uses = [
                block
                for block in final_message.content
                if isinstance(block, (ToolUseBlock, BetaToolUseBlock))
            ]

            if tool_uses:
                if not available_functions:
                    return list(tool_uses)

                return self._handle_tool_use_conversation(
                    final_message,
                    tool_uses,
                    params,
                    available_functions,
                    from_task,
                    from_agent,
                )

        full_response = self._apply_stop_words(full_response)

        self._emit_call_completed_event(
            response=full_response,
            call_type=LLMCallType.LLM_CALL,
            from_task=from_task,
            from_agent=from_agent,
            messages=params["messages"],
        )

        return self._invoke_after_llm_call_hooks(
            params["messages"], full_response, from_agent
        )

    def _execute_tools_and_collect_results(
        self,
        tool_uses: list[ToolUseBlock | BetaToolUseBlock],
        available_functions: dict[str, Any],
        from_task: Any | None = None,
        from_agent: Any | None = None,
    ) -> list[dict[str, Any]]:
        """Execute tools and collect results in Anthropic format.

        Args:
            tool_uses: List of tool use blocks from Claude's response (regular or beta API)
            available_functions: Available functions for tool calling
            from_task: Task that initiated the call
            from_agent: Agent that initiated the call

        Returns:
            List of tool result dictionaries in Anthropic format
        """
        tool_results = []

        for tool_use in tool_uses:
            function_name = tool_use.name
            function_args = tool_use.input

            result = self._handle_tool_execution(
                function_name=function_name,
                function_args=cast(dict[str, Any], function_args),
                available_functions=available_functions,
                from_task=from_task,
                from_agent=from_agent,
            )

            tool_result = {
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": str(result)
                if result is not None
                else "Tool execution completed",
            }
            tool_results.append(tool_result)

        return tool_results

    def _handle_tool_use_conversation(
        self,
        initial_response: Message | BetaMessage,
        tool_uses: list[ToolUseBlock | BetaToolUseBlock],
        params: dict[str, Any],
        available_functions: dict[str, Any],
        from_task: Any | None = None,
        from_agent: Any | None = None,
    ) -> str:
        """Handle the complete tool use conversation flow.

        This implements the proper Anthropic tool use pattern:
        1. Claude requests tool use
        2. We execute the tools
        3. We send tool results back to Claude
        4. Claude processes results and generates final response
        """
        tool_results = self._execute_tools_and_collect_results(
            tool_uses, available_functions, from_task, from_agent
        )

        follow_up_params = params.copy()

        # Add Claude's tool use response to conversation
        assistant_content: list[
            ThinkingBlock | ToolUseBlock | TextBlock | dict[str, Any]
        ] = []
        for block in initial_response.content:
            thinking_block = self._extract_thinking_block(block)
            if thinking_block:
                assistant_content.append(thinking_block)
            elif block.type == "tool_use":
                assistant_content.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )
            elif hasattr(block, "text"):
                assistant_content.append({"type": "text", "text": block.text})

        assistant_message = {"role": "assistant", "content": assistant_content}

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

            final_content = ""
            thinking_blocks: list[ThinkingBlock] = []

            if final_response.content:
                for content_block in final_response.content:
                    if hasattr(content_block, "text"):
                        final_content += content_block.text
                    else:
                        thinking_block = self._extract_thinking_block(content_block)
                        if thinking_block:
                            thinking_blocks.append(cast(ThinkingBlock, thinking_block))

            if thinking_blocks:
                self.previous_thinking_blocks = thinking_blocks

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
                return cast(str, tool_results[0]["content"])
            raise e

    async def _ahandle_completion(
        self,
        params: dict[str, Any],
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        """Handle non-streaming async message completion."""
        uses_file_api = _contains_file_id_reference(params.get("messages", []))
        betas: list[str] = []
        use_native_structured_output = False

        if uses_file_api:
            betas.append(ANTHROPIC_FILES_API_BETA)

        extra_body: dict[str, Any] | None = None
        if _is_pydantic_model_class(response_model):
            schema = transform_schema(response_model.model_json_schema())
            if _supports_native_structured_outputs(self.model):
                use_native_structured_output = True
                betas.append(ANTHROPIC_STRUCTURED_OUTPUTS_BETA)
                extra_body = {
                    "output_format": {
                        "type": "json_schema",
                        "schema": schema,
                    }
                }
            else:
                structured_tool = {
                    "name": "structured_output",
                    "description": "Output the structured response",
                    "input_schema": schema,
                }
                params["tools"] = [structured_tool]
                params["tool_choice"] = {"type": "tool", "name": "structured_output"}

        try:
            if betas:
                params["betas"] = betas
                response = await self.async_client.beta.messages.create(
                    **params, extra_body=extra_body
                )
            else:
                response = await self.async_client.messages.create(**params)

        except Exception as e:
            if is_context_length_exceeded(e):
                logging.error(f"Context window exceeded: {e}")
                raise LLMContextLengthExceededError(str(e)) from e
            raise e from e

        usage = self._extract_anthropic_token_usage(response)
        self._track_token_usage_internal(usage)

        if _is_pydantic_model_class(response_model) and response.content:
            if use_native_structured_output:
                for block in response.content:
                    if isinstance(block, (TextBlock, BetaTextBlock)):
                        structured_data = response_model.model_validate_json(block.text)
                        self._emit_call_completed_event(
                            response=structured_data.model_dump_json(),
                            call_type=LLMCallType.LLM_CALL,
                            from_task=from_task,
                            from_agent=from_agent,
                            messages=params["messages"],
                        )
                        return structured_data
            else:
                for block in response.content:
                    if (
                        isinstance(block, ToolUseBlock)
                        and block.name == "structured_output"
                    ):
                        structured_data = response_model.model_validate(block.input)
                        self._emit_call_completed_event(
                            response=structured_data.model_dump_json(),
                            call_type=LLMCallType.LLM_CALL,
                            from_task=from_task,
                            from_agent=from_agent,
                            messages=params["messages"],
                        )
                        return structured_data

        # Handle both ToolUseBlock (regular API) and BetaToolUseBlock (beta API features)
        if response.content:
            tool_uses = [
                block
                for block in response.content
                if isinstance(block, (ToolUseBlock, BetaToolUseBlock))
            ]

            if tool_uses:
                # If no available_functions, return tool calls for executor to handle
                if not available_functions:
                    self._emit_call_completed_event(
                        response=list(tool_uses),
                        call_type=LLMCallType.TOOL_CALL,
                        from_task=from_task,
                        from_agent=from_agent,
                        messages=params["messages"],
                    )
                    return list(tool_uses)

                return await self._ahandle_tool_use_conversation(
                    response,
                    tool_uses,
                    params,
                    available_functions,
                    from_task,
                    from_agent,
                )

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

    async def _ahandle_streaming_completion(
        self,
        params: dict[str, Any],
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        """Handle async streaming message completion."""
        betas: list[str] = []
        use_native_structured_output = False

        extra_body: dict[str, Any] | None = None
        if _is_pydantic_model_class(response_model):
            schema = transform_schema(response_model.model_json_schema())
            if _supports_native_structured_outputs(self.model):
                use_native_structured_output = True
                betas.append(ANTHROPIC_STRUCTURED_OUTPUTS_BETA)
                extra_body = {
                    "output_format": {
                        "type": "json_schema",
                        "schema": schema,
                    }
                }
            else:
                structured_tool = {
                    "name": "structured_output",
                    "description": "Output the structured response",
                    "input_schema": schema,
                }
                params["tools"] = [structured_tool]
                params["tool_choice"] = {"type": "tool", "name": "structured_output"}

        full_response = ""

        stream_params = {k: v for k, v in params.items() if k != "stream"}

        if betas:
            stream_params["betas"] = betas

        current_tool_calls: dict[int, dict[str, Any]] = {}

        stream_context = (
            self.async_client.beta.messages.stream(
                **stream_params, extra_body=extra_body
            )
            if betas
            else self.async_client.messages.stream(**stream_params)
        )
        async with stream_context as stream:
            response_id = None
            async for event in stream:
                if hasattr(event, "message") and hasattr(event.message, "id"):
                    response_id = event.message.id

                if hasattr(event, "delta") and hasattr(event.delta, "text"):
                    text_delta = event.delta.text
                    full_response += text_delta
                    self._emit_stream_chunk_event(
                        chunk=text_delta,
                        from_task=from_task,
                        from_agent=from_agent,
                        response_id=response_id,
                    )

                if event.type == "content_block_start":
                    block = event.content_block
                    if block.type == "tool_use":
                        block_index = event.index
                        current_tool_calls[block_index] = {
                            "id": block.id,
                            "name": block.name,
                            "arguments": "",
                            "index": block_index,
                        }
                        self._emit_stream_chunk_event(
                            chunk="",
                            from_task=from_task,
                            from_agent=from_agent,
                            tool_call={
                                "id": block.id,
                                "function": {
                                    "name": block.name,
                                    "arguments": "",
                                },
                                "type": "function",
                                "index": block_index,
                            },
                            call_type=LLMCallType.TOOL_CALL,
                            response_id=response_id,
                        )
                elif event.type == "content_block_delta":
                    if event.delta.type == "input_json_delta":
                        block_index = event.index
                        partial_json = event.delta.partial_json
                        if block_index in current_tool_calls and partial_json:
                            current_tool_calls[block_index]["arguments"] += partial_json
                            self._emit_stream_chunk_event(
                                chunk=partial_json,
                                from_task=from_task,
                                from_agent=from_agent,
                                tool_call={
                                    "id": current_tool_calls[block_index]["id"],
                                    "function": {
                                        "name": current_tool_calls[block_index]["name"],
                                        "arguments": current_tool_calls[block_index][
                                            "arguments"
                                        ],
                                    },
                                    "type": "function",
                                    "index": block_index,
                                },
                                call_type=LLMCallType.TOOL_CALL,
                                response_id=response_id,
                            )

            final_message = await stream.get_final_message()

        usage = self._extract_anthropic_token_usage(final_message)
        self._track_token_usage_internal(usage)

        if _is_pydantic_model_class(response_model):
            if use_native_structured_output:
                structured_data = response_model.model_validate_json(full_response)
                self._emit_call_completed_event(
                    response=structured_data.model_dump_json(),
                    call_type=LLMCallType.LLM_CALL,
                    from_task=from_task,
                    from_agent=from_agent,
                    messages=params["messages"],
                )
                return structured_data
            for block in final_message.content:
                if (
                    isinstance(block, ToolUseBlock)
                    and block.name == "structured_output"
                ):
                    structured_data = response_model.model_validate(block.input)
                    self._emit_call_completed_event(
                        response=structured_data.model_dump_json(),
                        call_type=LLMCallType.LLM_CALL,
                        from_task=from_task,
                        from_agent=from_agent,
                        messages=params["messages"],
                    )
                    return structured_data

        if final_message.content:
            tool_uses = [
                block
                for block in final_message.content
                if isinstance(block, (ToolUseBlock, BetaToolUseBlock))
            ]

            if tool_uses:
                if not available_functions:
                    return list(tool_uses)

                return await self._ahandle_tool_use_conversation(
                    final_message,
                    tool_uses,
                    params,
                    available_functions,
                    from_task,
                    from_agent,
                )

        full_response = self._apply_stop_words(full_response)

        self._emit_call_completed_event(
            response=full_response,
            call_type=LLMCallType.LLM_CALL,
            from_task=from_task,
            from_agent=from_agent,
            messages=params["messages"],
        )

        return full_response

    async def _ahandle_tool_use_conversation(
        self,
        initial_response: Message | BetaMessage,
        tool_uses: list[ToolUseBlock | BetaToolUseBlock],
        params: dict[str, Any],
        available_functions: dict[str, Any],
        from_task: Any | None = None,
        from_agent: Any | None = None,
    ) -> str:
        """Handle the complete async tool use conversation flow.

        This implements the proper Anthropic tool use pattern:
        1. Claude requests tool use
        2. We execute the tools
        3. We send tool results back to Claude
        4. Claude processes results and generates final response
        """
        tool_results = self._execute_tools_and_collect_results(
            tool_uses, available_functions, from_task, from_agent
        )

        follow_up_params = params.copy()

        assistant_message = {"role": "assistant", "content": initial_response.content}

        user_message = {"role": "user", "content": tool_results}

        follow_up_params["messages"] = params["messages"] + [
            assistant_message,
            user_message,
        ]

        try:
            final_response: Message = await self.async_client.messages.create(
                **follow_up_params
            )

            follow_up_usage = self._extract_anthropic_token_usage(final_response)
            self._track_token_usage_internal(follow_up_usage)

            final_content = ""
            if final_response.content:
                for content_block in final_response.content:
                    if hasattr(content_block, "text"):
                        final_content += content_block.text

            final_content = self._apply_stop_words(final_content)

            self._emit_call_completed_event(
                response=final_content,
                call_type=LLMCallType.LLM_CALL,
                from_task=from_task,
                from_agent=from_agent,
                messages=follow_up_params["messages"],
            )

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
            if tool_results:
                return cast(str, tool_results[0]["content"])
            raise e

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

    @staticmethod
    def _extract_anthropic_token_usage(
        response: Message | BetaMessage,
    ) -> dict[str, Any]:
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

    def supports_multimodal(self) -> bool:
        """Check if the model supports multimodal inputs.

        All Claude 3+ models support vision and PDFs.

        Returns:
            True if the model supports images and PDFs.
        """
        return "claude-3" in self.model.lower() or "claude-4" in self.model.lower()

    def get_file_uploader(self) -> Any:
        """Get an Anthropic file uploader using this LLM's clients.

        Returns:
            AnthropicFileUploader instance with pre-configured sync and async clients.
        """
        try:
            from crewai_files.uploaders.anthropic import AnthropicFileUploader

            return AnthropicFileUploader(
                client=self.client,
                async_client=self.async_client,
            )
        except ImportError:
            return None
