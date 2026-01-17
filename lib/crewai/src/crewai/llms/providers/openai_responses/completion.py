"""OpenAI Responses API completion implementation for CrewAI.

This module provides native integration with OpenAI's Responses API (/v1/responses),
offering advantages for agent-based workflows including simpler input format,
built-in conversation management via previous_response_id, and native support
for o-series reasoning models.
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Any, Literal

import httpx
from openai import AsyncOpenAI, OpenAI
from openai.types.responses import Response
from pydantic import BaseModel

from crewai.events.types.llm_events import LLMCallType
from crewai.llms.base_llm import BaseLLM
from crewai.llms.hooks.transport import AsyncHTTPTransport, HTTPTransport
from crewai.utilities.agent_utils import is_context_length_exceeded
from crewai.utilities.exceptions.context_window_exceeding_exception import (
    LLMContextLengthExceededError,
)


if TYPE_CHECKING:
    from crewai.agent.core import Agent
    from crewai.llms.hooks.base import BaseInterceptor
    from crewai.task import Task
    from crewai.tools.base_tool import BaseTool
    from crewai.utilities.types import LLMMessage


# Context window sizes for models supported by Responses API
RESPONSES_API_CONTEXT_WINDOWS: dict[str, int] = {
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4.1": 1047576,
    "gpt-4.1-mini": 1047576,
    "gpt-4.1-nano": 1047576,
    "o1": 200000,
    "o1-mini": 128000,
    "o1-preview": 128000,
    "o3": 200000,
    "o3-mini": 200000,
    "o4-mini": 200000,
}


class OpenAIResponsesCompletion(BaseLLM):
    """OpenAI Responses API completion implementation.

    This class provides native integration with OpenAI's Responses API,
    offering advantages over the Chat Completions API for agent workflows:

    - Simpler input format: Use plain strings or structured input instead of
      complex message arrays
    - Built-in conversation management: Stateful interactions with
      previous_response_id for multi-turn conversations
    - Native tool support: Cleaner function calling semantics
    - Streaming support: Real-time token streaming with simpler event handling
    - Better support for o-series reasoning models: reasoning_effort parameter

    Usage:
        ```python
        from crewai import Agent, LLM

        # Option 1: Using provider parameter
        llm = LLM(model="gpt-4o", provider="openai_responses")

        # Option 2: Using model prefix
        llm = LLM(model="openai_responses/gpt-4o")

        # With o-series reasoning models
        llm = LLM(
            model="o3-mini",
            provider="openai_responses",
            reasoning_effort="high"
        )

        agent = Agent(
            role="Research Analyst",
            goal="Find and summarize information",
            backstory="Expert researcher",
            llm=llm,
        )
        ```
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        project: str | None = None,
        timeout: float | None = None,
        max_retries: int = 2,
        default_headers: dict[str, str] | None = None,
        default_query: dict[str, Any] | None = None,
        client_params: dict[str, Any] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_output_tokens: int | None = None,
        stream: bool = False,
        reasoning_effort: Literal["low", "medium", "high"] | None = None,
        previous_response_id: str | None = None,
        provider: str | None = None,
        interceptor: BaseInterceptor[httpx.Request, httpx.Response] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize OpenAI Responses API client.

        Args:
            model: Model identifier (e.g., "gpt-4o", "o3-mini")
            api_key: OpenAI API key (falls back to OPENAI_API_KEY env var)
            base_url: Custom API base URL
            organization: OpenAI organization ID
            project: OpenAI project ID
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            default_headers: Default headers for all requests
            default_query: Default query parameters for all requests
            client_params: Additional client configuration parameters
            temperature: Sampling temperature (0-2)
            top_p: Nucleus sampling parameter
            max_output_tokens: Maximum tokens in response
            stream: Enable streaming responses
            reasoning_effort: Effort level for o-series reasoning models
                ("low", "medium", "high")
            previous_response_id: ID of previous response for multi-turn
                conversations
            provider: Provider identifier (typically "openai_responses")
            interceptor: HTTP interceptor for request/response modification
            **kwargs: Additional provider-specific parameters
        """
        if provider is None:
            provider = kwargs.pop("provider", "openai_responses")

        self.interceptor = interceptor

        # Client configuration
        self.organization = organization
        self.project = project
        self.max_retries = max_retries
        self.default_headers = default_headers
        self.default_query = default_query
        self.client_params = client_params
        self.timeout = timeout
        self.base_url = base_url
        self.api_base = kwargs.pop("api_base", None)

        super().__init__(
            model=model,
            temperature=temperature,
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url,
            timeout=timeout,
            provider=provider,
            **kwargs,
        )

        # Initialize sync client
        client_config = self._get_client_params()
        if self.interceptor:
            transport = HTTPTransport(interceptor=self.interceptor)
            http_client = httpx.Client(transport=transport)
            client_config["http_client"] = http_client

        self.client = OpenAI(**client_config)

        # Initialize async client
        async_client_config = self._get_client_params()
        if self.interceptor:
            async_transport = AsyncHTTPTransport(interceptor=self.interceptor)
            async_http_client = httpx.AsyncClient(transport=async_transport)
            async_client_config["http_client"] = async_http_client

        self.async_client = AsyncOpenAI(**async_client_config)

        # Responses API specific parameters
        self.top_p = top_p
        self.max_output_tokens = max_output_tokens
        self.stream = stream
        self.reasoning_effort = reasoning_effort
        self.previous_response_id = previous_response_id

        # Model type detection
        self.is_o_series = any(
            model.lower().startswith(prefix)
            for prefix in ["o1", "o3", "o4"]
        )

    def _get_client_params(self) -> dict[str, Any]:
        """Get OpenAI client initialization parameters."""
        if self.api_key is None:
            self.api_key = os.getenv("OPENAI_API_KEY")
            if self.api_key is None:
                raise ValueError("OPENAI_API_KEY is required")

        base_params = {
            "api_key": self.api_key,
            "organization": self.organization,
            "project": self.project,
            "base_url": self.base_url
            or self.api_base
            or os.getenv("OPENAI_BASE_URL")
            or None,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "default_headers": self.default_headers,
            "default_query": self.default_query,
        }

        client_params = {k: v for k, v in base_params.items() if v is not None}

        if self.client_params:
            client_params.update(self.client_params)

        return client_params

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
        """Call OpenAI Responses API.

        Args:
            messages: Input messages (string or list of message dicts).
                System messages are converted to the `instructions` parameter.
                Other messages become the `input` parameter.
            tools: List of tool/function definitions
            callbacks: Callback functions (not used in native implementation)
            available_functions: Available functions for tool calling
            from_task: Task that initiated the call
            from_agent: Agent that initiated the call
            response_model: Pydantic model for structured output

        Returns:
            Response text or tool call result
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

            formatted_messages = self._format_messages(messages)

            if not self._invoke_before_llm_call_hooks(formatted_messages, from_agent):
                raise ValueError("LLM call blocked by before_llm_call hook")

            params = self._prepare_responses_params(
                messages=formatted_messages,
                tools=tools,
                response_model=response_model,
            )

            if self.stream:
                return self._handle_streaming_response(
                    params=params,
                    formatted_messages=formatted_messages,
                    available_functions=available_functions,
                    from_task=from_task,
                    from_agent=from_agent,
                    response_model=response_model,
                )

            return self._handle_response(
                params=params,
                formatted_messages=formatted_messages,
                available_functions=available_functions,
                from_task=from_task,
                from_agent=from_agent,
                response_model=response_model,
            )

        except Exception as e:
            if is_context_length_exceeded(e):
                logging.error(f"Context window exceeded: {e}")
                raise LLMContextLengthExceededError(str(e)) from e

            error_msg = f"OpenAI Responses API call failed: {e!s}"
            logging.error(error_msg)
            self._emit_call_failed_event(
                error=error_msg, from_task=from_task, from_agent=from_agent
            )
            raise

    async def acall(
        self,
        messages: str | list[LLMMessage],
        tools: list[dict[str, BaseTool]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Task | None = None,
        from_agent: Agent | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        """Async call to OpenAI Responses API.

        Args:
            messages: Input messages (string or list of message dicts)
            tools: List of tool/function definitions
            callbacks: Callback functions
            available_functions: Available functions for tool calling
            from_task: Task that initiated the call
            from_agent: Agent that initiated the call
            response_model: Pydantic model for structured output

        Returns:
            Response text or tool call result
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

            formatted_messages = self._format_messages(messages)

            if not self._invoke_before_llm_call_hooks(formatted_messages, from_agent):
                raise ValueError("LLM call blocked by before_llm_call hook")

            params = self._prepare_responses_params(
                messages=formatted_messages,
                tools=tools,
                response_model=response_model,
            )

            if self.stream:
                return await self._ahandle_streaming_response(
                    params=params,
                    formatted_messages=formatted_messages,
                    available_functions=available_functions,
                    from_task=from_task,
                    from_agent=from_agent,
                    response_model=response_model,
                )

            return await self._ahandle_response(
                params=params,
                formatted_messages=formatted_messages,
                available_functions=available_functions,
                from_task=from_task,
                from_agent=from_agent,
                response_model=response_model,
            )

        except Exception as e:
            if is_context_length_exceeded(e):
                logging.error(f"Context window exceeded: {e}")
                raise LLMContextLengthExceededError(str(e)) from e

            error_msg = f"OpenAI Responses API call failed: {e!s}"
            logging.error(error_msg)
            self._emit_call_failed_event(
                error=error_msg, from_task=from_task, from_agent=from_agent
            )
            raise

    def _prepare_responses_params(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, BaseTool]] | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> dict[str, Any]:
        """Prepare parameters for Responses API call.

        Converts CrewAI message format to Responses API format:
        - System messages become `instructions`
        - User/assistant messages become `input`

        Args:
            messages: List of message dictionaries
            tools: Optional tool definitions
            response_model: Optional Pydantic model for structured output

        Returns:
            Parameters dict for responses.create()
        """
        # Extract system messages as instructions
        instructions_parts: list[str] = []
        input_messages: list[dict[str, str]] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                instructions_parts.append(content)
            else:
                input_messages.append({"role": role, "content": content})

        # Build input - can be a string for simple cases or structured
        if len(input_messages) == 1 and input_messages[0]["role"] == "user":
            # Simple case: single user message as string
            api_input: str | list[dict[str, str]] = input_messages[0]["content"]
        else:
            # Complex case: multiple messages as list
            api_input = input_messages

        params: dict[str, Any] = {
            "model": self.model,
            "input": api_input,
        }

        # Add instructions if we have system messages
        if instructions_parts:
            params["instructions"] = "\n\n".join(instructions_parts)

        # Generation parameters
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.top_p is not None:
            params["top_p"] = self.top_p
        if self.max_output_tokens is not None:
            params["max_output_tokens"] = self.max_output_tokens

        # Reasoning effort for o-series models
        if self.is_o_series and self.reasoning_effort:
            params["reasoning"] = {"effort": self.reasoning_effort}

        # Stateful conversation
        if self.previous_response_id:
            params["previous_response_id"] = self.previous_response_id

        # Tools
        if tools:
            params["tools"] = self._convert_tools_for_responses(tools)
            params["tool_choice"] = "auto"

        # Structured output via text format
        if response_model:
            params["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": response_model.__name__,
                    "schema": response_model.model_json_schema(),
                    "strict": True,
                }
            }

        # Streaming
        if self.stream:
            params["stream"] = True

        return params

    def _convert_tools_for_responses(
        self, tools: list[dict[str, BaseTool]]
    ) -> list[dict[str, Any]]:
        """Convert CrewAI tools to Responses API format.

        The Responses API uses a similar tool format to Chat Completions,
        with `strict: true` set by default for better reliability.

        Args:
            tools: CrewAI tool definitions

        Returns:
            List of Responses API tool definitions
        """
        from crewai.llms.providers.utils.common import safe_tool_conversion

        responses_tools: list[dict[str, Any]] = []

        for tool in tools:
            name, description, parameters = safe_tool_conversion(tool, "OpenAI")

            responses_tool: dict[str, Any] = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "strict": True,  # Enable strict mode by default
                },
            }

            if parameters:
                if isinstance(parameters, dict):
                    responses_tool["function"]["parameters"] = parameters
                else:
                    responses_tool["function"]["parameters"] = dict(parameters)

            responses_tools.append(responses_tool)

        return responses_tools

    def _handle_response(
        self,
        params: dict[str, Any],
        formatted_messages: list[LLMMessage],
        available_functions: dict[str, Any] | None = None,
        from_task: Task | None = None,
        from_agent: Agent | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        """Handle non-streaming Responses API call.

        Args:
            params: Responses API parameters
            available_functions: Available functions for tool execution
            from_task: Task context
            from_agent: Agent context
            response_model: Optional response model for structured output

        Returns:
            Response text or tool call result
        """
        response: Response = self.client.responses.create(**params)

        # Store response ID for potential follow-up calls
        self.previous_response_id = response.id

        # Track token usage
        if response.usage:
            self._track_token_usage_internal({
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.total_tokens,
            })

        # Handle tool calls
        if response.output and available_functions:
            for output_item in response.output:
                if hasattr(output_item, "type") and output_item.type == "function_call":
                    function_name = output_item.name
                    try:
                        function_args = json.loads(output_item.arguments)
                    except json.JSONDecodeError as e:
                        logging.error(f"Failed to parse tool arguments: {e}")
                        continue

                    result = self._handle_tool_execution(
                        function_name=function_name,
                        function_args=function_args,
                        available_functions=available_functions,
                        from_task=from_task,
                        from_agent=from_agent,
                    )

                    if result is not None:
                        return result

        # Extract text content
        content = response.output_text or ""
        content = self._apply_stop_words(content)

        # Handle structured output
        if response_model and content:
            try:
                parsed = response_model.model_validate_json(content)
                structured_json = parsed.model_dump_json()
                self._emit_call_completed_event(
                    response=structured_json,
                    call_type=LLMCallType.LLM_CALL,
                    from_task=from_task,
                    from_agent=from_agent,
                )
                structured_json = self._invoke_after_llm_call_hooks(
                    formatted_messages, structured_json, from_agent
                )
                return structured_json
            except Exception as e:
                logging.warning(f"Structured output parsing failed: {e}")

        self._emit_call_completed_event(
            response=content,
            call_type=LLMCallType.LLM_CALL,
            from_task=from_task,
            from_agent=from_agent,
        )

        content = self._invoke_after_llm_call_hooks(
            formatted_messages, content, from_agent
        )

        return content

    async def _ahandle_response(
        self,
        params: dict[str, Any],
        formatted_messages: list[LLMMessage],
        available_functions: dict[str, Any] | None = None,
        from_task: Task | None = None,
        from_agent: Agent | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        """Handle async non-streaming Responses API call.

        Args:
            params: Responses API parameters
            available_functions: Available functions for tool execution
            from_task: Task context
            from_agent: Agent context
            response_model: Optional response model for structured output

        Returns:
            Response text or tool call result
        """
        response: Response = await self.async_client.responses.create(**params)

        # Store response ID
        self.previous_response_id = response.id

        # Track token usage
        if response.usage:
            self._track_token_usage_internal({
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.total_tokens,
            })

        # Handle tool calls
        if response.output and available_functions:
            for output_item in response.output:
                if hasattr(output_item, "type") and output_item.type == "function_call":
                    function_name = output_item.name
                    try:
                        function_args = json.loads(output_item.arguments)
                    except json.JSONDecodeError as e:
                        logging.error(f"Failed to parse tool arguments: {e}")
                        continue

                    result = self._handle_tool_execution(
                        function_name=function_name,
                        function_args=function_args,
                        available_functions=available_functions,
                        from_task=from_task,
                        from_agent=from_agent,
                    )

                    if result is not None:
                        return result

        # Extract text content
        content = response.output_text or ""
        content = self._apply_stop_words(content)

        # Handle structured output
        if response_model and content:
            try:
                parsed = response_model.model_validate_json(content)
                structured_json = parsed.model_dump_json()
                self._emit_call_completed_event(
                    response=structured_json,
                    call_type=LLMCallType.LLM_CALL,
                    from_task=from_task,
                    from_agent=from_agent,
                )
                structured_json = self._invoke_after_llm_call_hooks(
                    formatted_messages, structured_json, from_agent
                )
                return structured_json
            except Exception as e:
                logging.warning(f"Structured output parsing failed: {e}")

        self._emit_call_completed_event(
            response=content,
            call_type=LLMCallType.LLM_CALL,
            from_task=from_task,
            from_agent=from_agent,
        )

        return self._invoke_after_llm_call_hooks(
            formatted_messages, content, from_agent
        )

    def _handle_streaming_response(
        self,
        params: dict[str, Any],
        formatted_messages: list[LLMMessage],
        available_functions: dict[str, Any] | None = None,
        from_task: Task | None = None,
        from_agent: Agent | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        """Handle streaming Responses API call.

        Args:
            params: Responses API parameters
            available_functions: Available functions for tool execution
            from_task: Task context
            from_agent: Agent context
            response_model: Optional response model for structured output

        Returns:
            Complete response text or tool call result
        """
        full_response = ""
        tool_calls: dict[str, dict[str, Any]] = {}
        response_id: str | None = None
        usage_data: dict[str, int] = {}

        with self.client.responses.stream(**params) as stream:
            for event in stream:
                event_type = getattr(event, "type", None)

                # Handle response created event
                if event_type == "response.created":
                    if hasattr(event, "response") and event.response:
                        response_id = event.response.id

                # Handle text delta events
                elif event_type == "response.output_text.delta":
                    delta_text = getattr(event, "delta", "")
                    if delta_text:
                        full_response += delta_text
                        self._emit_stream_chunk_event(
                            chunk=delta_text,
                            from_task=from_task,
                            from_agent=from_agent,
                        )

                # Handle function call argument delta
                elif event_type == "response.function_call_arguments.delta":
                    item_id = getattr(event, "item_id", "default")
                    delta = getattr(event, "delta", "")

                    if item_id not in tool_calls:
                        tool_calls[item_id] = {
                            "name": "",
                            "arguments": "",
                        }

                    tool_calls[item_id]["arguments"] += delta

                    self._emit_stream_chunk_event(
                        chunk=delta,
                        from_task=from_task,
                        from_agent=from_agent,
                        tool_call={
                            "id": item_id,
                            "function": tool_calls[item_id],
                            "type": "function",
                        },
                        call_type=LLMCallType.TOOL_CALL,
                    )

                # Handle function call name
                elif event_type == "response.output_item.added":
                    item = getattr(event, "item", None)
                    if item and hasattr(item, "type") and item.type == "function_call":
                        item_id = getattr(item, "id", "default")
                        if item_id not in tool_calls:
                            tool_calls[item_id] = {
                                "name": getattr(item, "name", ""),
                                "arguments": "",
                            }
                        else:
                            tool_calls[item_id]["name"] = getattr(item, "name", "")

                # Handle completion event
                elif event_type == "response.completed":
                    if hasattr(event, "response") and event.response:
                        resp = event.response
                        response_id = resp.id
                        if resp.usage:
                            usage_data = {
                                "prompt_tokens": resp.usage.input_tokens,
                                "completion_tokens": resp.usage.output_tokens,
                                "total_tokens": resp.usage.total_tokens,
                            }

        # Store response ID
        if response_id:
            self.previous_response_id = response_id

        # Track token usage
        if usage_data:
            self._track_token_usage_internal(usage_data)

        # Handle tool calls
        if tool_calls and available_functions:
            for item_id, call_data in tool_calls.items():
                function_name = call_data.get("name", "")
                arguments = call_data.get("arguments", "")

                if not function_name or not arguments:
                    continue

                if function_name not in available_functions:
                    logging.warning(
                        f"Function '{function_name}' not found in available functions"
                    )
                    continue

                try:
                    function_args = json.loads(arguments)
                except json.JSONDecodeError as e:
                    logging.error(f"Failed to parse streamed tool arguments: {e}")
                    continue

                result = self._handle_tool_execution(
                    function_name=function_name,
                    function_args=function_args,
                    available_functions=available_functions,
                    from_task=from_task,
                    from_agent=from_agent,
                )

                if result is not None:
                    return result

        # Apply stop words
        full_response = self._apply_stop_words(full_response)

        # Handle structured output
        if response_model and full_response:
            try:
                parsed = response_model.model_validate_json(full_response)
                structured_json = parsed.model_dump_json()
                self._emit_call_completed_event(
                    response=structured_json,
                    call_type=LLMCallType.LLM_CALL,
                    from_task=from_task,
                    from_agent=from_agent,
                )
                structured_json = self._invoke_after_llm_call_hooks(
                    formatted_messages, structured_json, from_agent
                )
                return structured_json
            except Exception as e:
                logging.warning(f"Structured output parsing failed: {e}")

        self._emit_call_completed_event(
            response=full_response,
            call_type=LLMCallType.LLM_CALL,
            from_task=from_task,
            from_agent=from_agent,
        )

        return self._invoke_after_llm_call_hooks(
            formatted_messages, full_response, from_agent
        )

    async def _ahandle_streaming_response(
        self,
        params: dict[str, Any],
        formatted_messages: list[LLMMessage],
        available_functions: dict[str, Any] | None = None,
        from_task: Task | None = None,
        from_agent: Agent | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        """Handle async streaming Responses API call.

        Args:
            params: Responses API parameters
            available_functions: Available functions for tool execution
            from_task: Task context
            from_agent: Agent context
            response_model: Optional response model for structured output

        Returns:
            Complete response text or tool call result
        """
        full_response = ""
        tool_calls: dict[str, dict[str, Any]] = {}
        response_id: str | None = None
        usage_data: dict[str, int] = {}

        async with self.async_client.responses.stream(**params) as stream:
            async for event in stream:
                event_type = getattr(event, "type", None)

                # Handle response created event
                if event_type == "response.created":
                    if hasattr(event, "response") and event.response:
                        response_id = event.response.id

                # Handle text delta events
                elif event_type == "response.output_text.delta":
                    delta_text = getattr(event, "delta", "")
                    if delta_text:
                        full_response += delta_text
                        self._emit_stream_chunk_event(
                            chunk=delta_text,
                            from_task=from_task,
                            from_agent=from_agent,
                        )

                # Handle function call argument delta
                elif event_type == "response.function_call_arguments.delta":
                    item_id = getattr(event, "item_id", "default")
                    delta = getattr(event, "delta", "")

                    if item_id not in tool_calls:
                        tool_calls[item_id] = {
                            "name": "",
                            "arguments": "",
                        }

                    tool_calls[item_id]["arguments"] += delta

                    self._emit_stream_chunk_event(
                        chunk=delta,
                        from_task=from_task,
                        from_agent=from_agent,
                        tool_call={
                            "id": item_id,
                            "function": tool_calls[item_id],
                            "type": "function",
                        },
                        call_type=LLMCallType.TOOL_CALL,
                    )

                # Handle function call name
                elif event_type == "response.output_item.added":
                    item = getattr(event, "item", None)
                    if item and hasattr(item, "type") and item.type == "function_call":
                        item_id = getattr(item, "id", "default")
                        if item_id not in tool_calls:
                            tool_calls[item_id] = {
                                "name": getattr(item, "name", ""),
                                "arguments": "",
                            }
                        else:
                            tool_calls[item_id]["name"] = getattr(item, "name", "")

                # Handle completion event
                elif event_type == "response.completed":
                    if hasattr(event, "response") and event.response:
                        resp = event.response
                        response_id = resp.id
                        if resp.usage:
                            usage_data = {
                                "prompt_tokens": resp.usage.input_tokens,
                                "completion_tokens": resp.usage.output_tokens,
                                "total_tokens": resp.usage.total_tokens,
                            }

        # Store response ID
        if response_id:
            self.previous_response_id = response_id

        # Track token usage
        if usage_data:
            self._track_token_usage_internal(usage_data)

        # Handle tool calls
        if tool_calls and available_functions:
            for item_id, call_data in tool_calls.items():
                function_name = call_data.get("name", "")
                arguments = call_data.get("arguments", "")

                if not function_name or not arguments:
                    continue

                if function_name not in available_functions:
                    logging.warning(
                        f"Function '{function_name}' not found in available functions"
                    )
                    continue

                try:
                    function_args = json.loads(arguments)
                except json.JSONDecodeError as e:
                    logging.error(f"Failed to parse streamed tool arguments: {e}")
                    continue

                result = self._handle_tool_execution(
                    function_name=function_name,
                    function_args=function_args,
                    available_functions=available_functions,
                    from_task=from_task,
                    from_agent=from_agent,
                )

                if result is not None:
                    return result

        # Apply stop words
        full_response = self._apply_stop_words(full_response)

        # Handle structured output
        if response_model and full_response:
            try:
                parsed = response_model.model_validate_json(full_response)
                structured_json = parsed.model_dump_json()
                self._emit_call_completed_event(
                    response=structured_json,
                    call_type=LLMCallType.LLM_CALL,
                    from_task=from_task,
                    from_agent=from_agent,
                )
                structured_json = self._invoke_after_llm_call_hooks(
                    formatted_messages, structured_json, from_agent
                )
                return structured_json
            except Exception as e:
                logging.warning(f"Structured output parsing failed: {e}")

        self._emit_call_completed_event(
            response=full_response,
            call_type=LLMCallType.LLM_CALL,
            from_task=from_task,
            from_agent=from_agent,
        )

        return self._invoke_after_llm_call_hooks(
            formatted_messages, full_response, from_agent
        )

    def supports_function_calling(self) -> bool:
        """Check if the model supports function calling."""
        # Most models support function calling, but o1-preview doesn't
        return "o1-preview" not in self.model.lower()

    def supports_stop_words(self) -> bool:
        """Check if the model supports stop words."""
        # Responses API doesn't have native stop word support
        # We apply stop words manually in _apply_stop_words
        return self._supports_stop_words_implementation()

    def get_context_window_size(self) -> int:
        """Get the context window size for the model."""
        from crewai.llm import CONTEXT_WINDOW_USAGE_RATIO

        # Find matching context window size
        for model_prefix, size in RESPONSES_API_CONTEXT_WINDOWS.items():
            if self.model.lower().startswith(model_prefix):
                return int(size * CONTEXT_WINDOW_USAGE_RATIO)

        # Default context window
        return int(128000 * CONTEXT_WINDOW_USAGE_RATIO)
