from __future__ import annotations

import json
import logging
import os
from typing import Any, Final, Literal, TypeGuard, cast

from pydantic import BaseModel, PrivateAttr, model_validator

from crewai.events.types.llm_events import LLMCallType
from crewai.llms.base_llm import BaseLLM, JsonResponseFormat, llm_call_context
from crewai.llms.hooks.base import BaseInterceptor
from crewai.llms.hooks.transport import AsyncHTTPTransport, HTTPTransport
from crewai.utilities.agent_utils import is_context_length_exceeded
from crewai.utilities.exceptions.context_window_exceeding_exception import (
    LLMContextLengthExceededError,
)
from crewai.utilities.types import LLMMessage


try:
    from anthropic import Anthropic, AsyncAnthropic, transform_schema
    from anthropic.types import (
        Message,
        TextBlock,
        ThinkingBlock,
        ToolUseBlock,
    )
    from anthropic.types.beta import BetaMessage, BetaTextBlock, BetaToolUseBlock
    import httpx
except ImportError:
    raise ImportError(
        'Anthropic native provider not available, to install: uv add "crewai[anthropic]"'
    ) from None


TOOL_SEARCH_TOOL_TYPES: Final[tuple[str, ...]] = (
    "tool_search_tool_regex_20251119",
    "tool_search_tool_bm25_20251119",
)

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


class AnthropicToolSearchConfig(BaseModel):
    """Configuration for Anthropic's server-side tool search.

    When enabled, tools marked with defer_loading=True are not loaded into
    context immediately. Instead, Claude uses the tool search tool to
    dynamically discover and load relevant tools on-demand.

    Attributes:
        type: The tool search variant to use.
            - "regex": Claude constructs regex patterns to search tool names/descriptions.
            - "bm25": Claude uses natural language queries to search tools.
    """

    type: Literal["regex", "bm25"] = "bm25"


class AnthropicCompletion(BaseLLM):
    """Anthropic native completion implementation.

    This class provides direct integration with the Anthropic Python SDK,
    offering native tool use, streaming support, and proper message formatting.
    """

    llm_type: Literal["anthropic"] = "anthropic"
    model: str = "claude-3-5-sonnet-20241022"
    timeout: float | None = None
    max_retries: int = 2
    max_tokens: int = 4096
    top_p: float | None = None
    stream: bool = False
    client_params: dict[str, Any] | None = None
    interceptor: BaseInterceptor[httpx.Request, httpx.Response] | None = None
    thinking: AnthropicThinkingConfig | None = None
    response_format: JsonResponseFormat | type[BaseModel] | None = None
    tool_search: AnthropicToolSearchConfig | None = None
    is_claude_3: bool = False
    supports_tools: bool = True

    _client: Any = PrivateAttr(default=None)
    _async_client: Any = PrivateAttr(default=None)
    _previous_thinking_blocks: list[Any] = PrivateAttr(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalize_anthropic_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        # Anthropic uses stop_sequences; normalize from stop kwarg
        popped = data.pop("stop_sequences", None)
        seqs = popped if popped is not None else (data.get("stop") or [])
        if isinstance(seqs, str):
            seqs = [seqs]
        data["stop"] = seqs
        data["is_claude_3"] = "claude-3" in data.get("model", "").lower()
        # Normalize tool_search
        ts = data.get("tool_search")
        if ts is True:
            data["tool_search"] = AnthropicToolSearchConfig()
        elif ts is not None and not isinstance(ts, AnthropicToolSearchConfig):
            data["tool_search"] = None
        return data

    @model_validator(mode="after")
    def _init_clients(self) -> AnthropicCompletion:
        self._client = Anthropic(**self._get_client_params())

        async_client_params = self._get_client_params()
        if self.interceptor:
            async_transport = AsyncHTTPTransport(interceptor=self.interceptor)
            async_http_client = httpx.AsyncClient(transport=async_transport)
            async_client_params["http_client"] = async_http_client

        self._async_client = AsyncAnthropic(**async_client_params)
        return self

    def to_config_dict(self) -> dict[str, Any]:
        """Extend base config with Anthropic-specific fields."""
        config = super().to_config_dict()
        if self.max_tokens != 4096:  # non-default
            config["max_tokens"] = self.max_tokens
        if self.max_retries != 2:  # non-default
            config["max_retries"] = self.max_retries
        if self.top_p is not None:
            config["top_p"] = self.top_p
        if self.timeout is not None:
            config["timeout"] = self.timeout
        return config

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
        with llm_call_context():
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
                formatted_messages, system_message = (
                    self._format_messages_for_anthropic(messages)
                )

                if not self._invoke_before_llm_call_hooks(
                    formatted_messages, from_agent
                ):
                    raise ValueError("LLM call blocked by before_llm_call hook")

                # Prepare completion parameters
                completion_params = self._prepare_completion_params(
                    formatted_messages, system_message, tools, available_functions
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
        with llm_call_context():
            try:
                self._emit_call_started_event(
                    messages=messages,
                    tools=tools,
                    callbacks=callbacks,
                    available_functions=available_functions,
                    from_task=from_task,
                    from_agent=from_agent,
                )

                formatted_messages, system_message = (
                    self._format_messages_for_anthropic(messages)
                )

                completion_params = self._prepare_completion_params(
                    formatted_messages, system_message, tools, available_functions
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
        available_functions: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Prepare parameters for Anthropic messages API.

        Args:
            messages: Formatted messages for Anthropic
            system_message: Extracted system message
            tools: Tool definitions
            available_functions: Available functions for tool calling. When provided
                with a single tool, tool_choice is automatically set to force tool use.

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
            converted_tools = self._convert_tools_for_interference(tools)

            # When tool_search is enabled and there are 2+ regular tools,
            # inject the search tool and mark regular tools with defer_loading.
            # With only 1 tool there's nothing to search — skip tool search
            # entirely so the normal forced tool_choice optimisation still works.
            regular_tools = [
                t
                for t in converted_tools
                if t.get("type", "") not in TOOL_SEARCH_TOOL_TYPES
            ]
            if self.tool_search is not None and len(regular_tools) >= 2:
                converted_tools = self._apply_tool_search(converted_tools)

            params["tools"] = converted_tools

            if available_functions and len(regular_tools) == 1:
                tool_name = regular_tools[0].get("name")
                if tool_name and tool_name in available_functions:
                    params["tool_choice"] = {"type": "tool", "name": tool_name}

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
            # Pass through tool search tool definitions unchanged
            tool_type = tool.get("type", "")
            if tool_type in TOOL_SEARCH_TOOL_TYPES:
                anthropic_tools.append(tool)
                continue

            if "input_schema" in tool and "name" in tool and "description" in tool:
                anthropic_tools.append(tool)
                continue

            try:
                from crewai.llms.providers.utils.common import safe_tool_conversion

                name, description, parameters = safe_tool_conversion(tool, "Anthropic")
            except (ImportError, KeyError, ValueError) as e:
                logging.error(f"Error converting tool to Anthropic format: {e}")
                raise e

            anthropic_tool: dict[str, Any] = {
                "name": name,
                "description": description,
            }

            if parameters and isinstance(parameters, dict):
                anthropic_tool["input_schema"] = parameters
            else:
                anthropic_tool["input_schema"] = {
                    "type": "object",
                    "properties": {},
                    "required": [],
                }

            anthropic_tools.append(anthropic_tool)

        return anthropic_tools

    def _apply_tool_search(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Inject tool search tool and mark regular tools with defer_loading.

        When tool_search is enabled, this method:
        1. Adds the appropriate tool search tool definition (regex or bm25)
        2. Marks all regular tools with defer_loading=True so they are only
           loaded when Claude discovers them via search

        Args:
            tools: Converted tool definitions in Anthropic format.

        Returns:
            Updated tools list with tool search tool prepended and
            regular tools marked as deferred.
        """
        if self.tool_search is None:
            return tools

        # Check if a tool search tool is already present (user passed one manually)
        has_search_tool = any(
            t.get("type", "") in TOOL_SEARCH_TOOL_TYPES for t in tools
        )

        result: list[dict[str, Any]] = []

        if not has_search_tool:
            # Map config type to API type identifier
            type_map = {
                "regex": "tool_search_tool_regex_20251119",
                "bm25": "tool_search_tool_bm25_20251119",
            }
            tool_type = type_map[self.tool_search.type]
            # Tool search tool names follow the convention: tool_search_tool_{variant}
            tool_name = f"tool_search_tool_{self.tool_search.type}"
            result.append({"type": tool_type, "name": tool_name})

        for tool in tools:
            # Don't modify tool search tools
            if tool.get("type", "") in TOOL_SEARCH_TOOL_TYPES:
                result.append(tool)
                continue

            # Mark regular tools as deferred if not already set
            if "defer_loading" not in tool:
                tool = {**tool, "defer_loading": True}
            result.append(tool)

        return result

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

    @staticmethod
    def _convert_image_blocks(content: Any) -> Any:
        """Convert OpenAI-style image_url blocks to Anthropic image blocks.

        Upstream code (e.g. StepExecutor) uses the standard ``image_url``
        format with a ``data:`` URI.  Anthropic rejects that — it requires
        ``{"type": "image", "source": {"type": "base64", ...}}``.

        Non-list content and blocks that are not ``image_url`` are passed
        through unchanged.
        """
        if not isinstance(content, list):
            return content

        converted: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "image_url":
                converted.append(block)
                continue

            image_info = block.get("image_url", {})
            url = image_info.get("url", "") if isinstance(image_info, dict) else ""
            if url.startswith("data:") and ";base64," in url:
                # Parse  data:<media_type>;base64,<data>
                header, b64_data = url.split(";base64,", 1)
                media_type = (
                    header.split("data:", 1)[1] if "data:" in header else "image/png"
                )
                converted.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64_data,
                        },
                    }
                )
            else:
                # Non-data URI — pass through as-is (Anthropic supports url source)
                converted.append(block)

        return converted

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
                tool_content = self._convert_image_blocks(content) if content else ""
                tool_result = {
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": tool_content,
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
                elif self.thinking and self._previous_thinking_blocks:
                    structured_content = cast(
                        list[dict[str, Any]],
                        [
                            *self._previous_thinking_blocks,
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
                    formatted_messages.append(
                        {
                            "role": role_str,
                            "content": self._convert_image_blocks(content),
                        }
                    )
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
        response_model: JsonResponseFormat | type[BaseModel] | None = None,
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
                response = self._client.beta.messages.create(
                    **params, extra_body=extra_body
                )
            else:
                response = self._client.messages.create(**params)

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
                            usage=usage,
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
                            usage=usage,
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
                        usage=usage,
                    )
                    return list(tool_uses)

                result = self._execute_first_tool(
                    tool_uses, available_functions, from_task, from_agent
                )
                if result is not None:
                    return result

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
            self._previous_thinking_blocks = thinking_blocks

        content = self._apply_stop_words(content)
        self._emit_call_completed_event(
            response=content,
            call_type=LLMCallType.LLM_CALL,
            from_task=from_task,
            from_agent=from_agent,
            messages=params["messages"],
            usage=usage,
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
        response_model: JsonResponseFormat | type[BaseModel] | None = None,
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
            self._client.beta.messages.stream(**stream_params, extra_body=extra_body)
            if betas
            else self._client.messages.stream(**stream_params)
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
            self._previous_thinking_blocks = thinking_blocks

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
                    usage=usage,
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
                        usage=usage,
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

                # Execute first tool and return result directly
                result = self._execute_first_tool(
                    tool_uses, available_functions, from_task, from_agent
                )
                if result is not None:
                    return result

        full_response = self._apply_stop_words(full_response)

        self._emit_call_completed_event(
            response=full_response,
            call_type=LLMCallType.LLM_CALL,
            from_task=from_task,
            from_agent=from_agent,
            messages=params["messages"],
            usage=usage,
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

    def _execute_first_tool(
        self,
        tool_uses: list[ToolUseBlock | BetaToolUseBlock],
        available_functions: dict[str, Any],
        from_task: Any | None = None,
        from_agent: Any | None = None,
    ) -> Any | None:
        """Execute the first tool from the tool_uses list and return its result.

        This is used when available_functions is provided, to directly execute
        the tool and return its result (matching OpenAI behavior for use cases
        like reasoning_handler).

        Args:
            tool_uses: List of tool use blocks from Claude's response
            available_functions: Available functions for tool calling
            from_task: Task that initiated the call
            from_agent: Agent that initiated the call

        Returns:
            The result of the first tool execution, or None if execution failed
        """
        tool_use = tool_uses[0]
        function_name = tool_use.name
        function_args = cast(dict[str, Any], tool_use.input)

        return self._handle_tool_execution(
            function_name=function_name,
            function_args=function_args,
            available_functions=available_functions,
            from_task=from_task,
            from_agent=from_agent,
        )

    # TODO: we drop this
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
            final_response: Message = self._client.messages.create(**follow_up_params)

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
                self._previous_thinking_blocks = thinking_blocks

            final_content = self._apply_stop_words(final_content)

            # Emit completion event for the final response
            self._emit_call_completed_event(
                response=final_content,
                call_type=LLMCallType.LLM_CALL,
                from_task=from_task,
                from_agent=from_agent,
                messages=follow_up_params["messages"],
                usage=follow_up_usage,
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
        response_model: JsonResponseFormat | type[BaseModel] | None = None,
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
                response = await self._async_client.beta.messages.create(
                    **params, extra_body=extra_body
                )
            else:
                response = await self._async_client.messages.create(**params)

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
                            usage=usage,
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
                            usage=usage,
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
                        usage=usage,
                    )
                    return list(tool_uses)

                result = self._execute_first_tool(
                    tool_uses, available_functions, from_task, from_agent
                )
                if result is not None:
                    return result

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
            usage=usage,
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
        response_model: JsonResponseFormat | type[BaseModel] | None = None,
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
            self._async_client.beta.messages.stream(
                **stream_params, extra_body=extra_body
            )
            if betas
            else self._async_client.messages.stream(**stream_params)
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
                    usage=usage,
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
                        usage=usage,
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

                result = self._execute_first_tool(
                    tool_uses, available_functions, from_task, from_agent
                )
                if result is not None:
                    return result

        full_response = self._apply_stop_words(full_response)

        self._emit_call_completed_event(
            response=full_response,
            call_type=LLMCallType.LLM_CALL,
            from_task=from_task,
            from_agent=from_agent,
            messages=params["messages"],
            usage=usage,
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
            final_response: Message = await self._async_client.messages.create(
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
                usage=follow_up_usage,
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
            cache_read_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0
            return {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "cached_prompt_tokens": cache_read_tokens,
            }
        return {"total_tokens": 0}

    def supports_multimodal(self) -> bool:
        """Check if the model supports multimodal inputs.

        All Claude 3+ models support vision and PDFs.

        Returns:
            True if the model supports images and PDFs.
        """
        model_lower = self.model.lower()
        return (
            "claude-3" in model_lower
            or "claude-4" in model_lower
            or "claude-sonnet-4" in model_lower
            or "claude-opus-4" in model_lower
            or "claude-haiku-4" in model_lower
        )

    def get_file_uploader(self) -> Any:
        """Get an Anthropic file uploader using this LLM's clients.

        Returns:
            AnthropicFileUploader instance with pre-configured sync and async clients.
        """
        try:
            from crewai_files.uploaders.anthropic import AnthropicFileUploader

            return AnthropicFileUploader(
                client=self._client,
                async_client=self._async_client,
            )
        except ImportError:
            return None
