"""OpenAI Responses API implementation for CrewAI.

This module provides integration with OpenAI's Responses API (/v1/responses),
which offers advantages over the traditional Chat Completions API for agent-based
workflows including simpler input format, built-in conversation management,
and native tool support.
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Any

import httpx
from openai import APIConnectionError, AsyncOpenAI, NotFoundError, OpenAI
from openai.types.responses import Response
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from pydantic import BaseModel

from crewai.events.types.llm_events import LLMCallType
from crewai.llms.base_llm import BaseLLM
from crewai.llms.hooks.transport import AsyncHTTPTransport, HTTPTransport
from crewai.utilities.agent_utils import is_context_length_exceeded
from crewai.utilities.exceptions.context_window_exceeding_exception import (
    LLMContextLengthExceededError,
)
from crewai.utilities.pydantic_schema_utils import generate_model_description
from crewai.utilities.types import LLMMessage


if TYPE_CHECKING:
    from crewai.agent.core import Agent
    from crewai.llms.hooks.base import BaseInterceptor
    from crewai.task import Task
    from crewai.tools.base_tool import BaseTool


class OpenAIResponsesCompletion(BaseLLM):
    """OpenAI Responses API implementation.

    This class provides direct integration with OpenAI's Responses API,
    offering simpler input format, built-in conversation management,
    native tool support, and better support for o-series reasoning models.

    Example usage:
        # Option 1: Using provider parameter
        llm = LLM(model="gpt-4o", provider="openai_responses")

        # Option 2: Using model prefix
        llm = LLM(model="openai_responses/gpt-4o")

        # Works with all CrewAI components
        agent = Agent(
            role="Research Analyst",
            goal="Find and summarize information",
            backstory="Expert researcher",
            llm=llm,
        )
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
        reasoning_effort: str | None = None,
        previous_response_id: str | None = None,
        store: bool | None = None,
        stream: bool = False,
        response_format: dict[str, Any] | type[BaseModel] | None = None,
        provider: str | None = None,
        interceptor: BaseInterceptor[httpx.Request, httpx.Response] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize OpenAI Responses API client.

        Args:
            model: The model to use (e.g., "gpt-4o", "gpt-4o-mini", "o1", "o3-mini")
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            base_url: Custom base URL for API requests
            organization: OpenAI organization ID
            project: OpenAI project ID
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            default_headers: Default headers to include in requests
            default_query: Default query parameters
            client_params: Additional client configuration parameters
            temperature: Sampling temperature (0-2)
            top_p: Nucleus sampling parameter
            max_output_tokens: Maximum tokens in the response
            reasoning_effort: Reasoning effort for o-series models ("low", "medium", "high")
            previous_response_id: ID of previous response for multi-turn conversations
            store: Whether to store the response for later retrieval
            stream: Whether to stream the response
            response_format: Response format (dict or Pydantic model for structured output)
            provider: Provider name (defaults to "openai_responses")
            interceptor: HTTP interceptor for request/response modification
            **kwargs: Additional parameters
        """
        if provider is None:
            provider = kwargs.pop("provider", "openai_responses")

        self.interceptor = interceptor
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

        client_config = self._get_client_params()
        if self.interceptor:
            transport = HTTPTransport(interceptor=self.interceptor)
            http_client = httpx.Client(transport=transport)
            client_config["http_client"] = http_client

        self.client = OpenAI(**client_config)

        async_client_config = self._get_client_params()
        if self.interceptor:
            async_transport = AsyncHTTPTransport(interceptor=self.interceptor)
            async_http_client = httpx.AsyncClient(transport=async_transport)
            async_client_config["http_client"] = async_http_client

        self.async_client = AsyncOpenAI(**async_client_config)

        self.top_p = top_p
        self.max_output_tokens = max_output_tokens
        self.reasoning_effort = reasoning_effort
        self.previous_response_id = previous_response_id
        self.store = store
        self.stream = stream
        self.response_format = response_format
        self.is_o_model = any(
            model.lower().startswith(prefix) for prefix in ["o1", "o3", "o4"]
        )
        self.last_response_id: str | None = None

    def _get_client_params(self) -> dict[str, Any]:
        """Get OpenAI client parameters."""
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
            messages: Input messages for the response
            tools: List of tool/function definitions
            callbacks: Callback functions (not used in native implementation)
            available_functions: Available functions for tool calling
            from_task: Task that initiated the call
            from_agent: Agent that initiated the call
            response_model: Response model for structured output

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

            params = self._prepare_response_params(
                messages=formatted_messages, tools=tools, response_model=response_model
            )

            if self.stream:
                return self._handle_streaming_response(
                    params=params,
                    available_functions=available_functions,
                    from_task=from_task,
                    from_agent=from_agent,
                    response_model=response_model,
                )

            return self._handle_response(
                params=params,
                available_functions=available_functions,
                from_task=from_task,
                from_agent=from_agent,
                response_model=response_model,
                formatted_messages=formatted_messages,
            )

        except Exception as e:
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
            messages: Input messages for the response
            tools: List of tool/function definitions
            callbacks: Callback functions (not used in native implementation)
            available_functions: Available functions for tool calling
            from_task: Task that initiated the call
            from_agent: Agent that initiated the call
            response_model: Response model for structured output

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

            params = self._prepare_response_params(
                messages=formatted_messages, tools=tools, response_model=response_model
            )

            if self.stream:
                return await self._ahandle_streaming_response(
                    params=params,
                    available_functions=available_functions,
                    from_task=from_task,
                    from_agent=from_agent,
                    response_model=response_model,
                )

            return await self._ahandle_response(
                params=params,
                available_functions=available_functions,
                from_task=from_task,
                from_agent=from_agent,
                response_model=response_model,
                formatted_messages=formatted_messages,
            )

        except Exception as e:
            error_msg = f"OpenAI Responses API call failed: {e!s}"
            logging.error(error_msg)
            self._emit_call_failed_event(
                error=error_msg, from_task=from_task, from_agent=from_agent
            )
            raise

    def _prepare_response_params(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, BaseTool]] | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> dict[str, Any]:
        """Prepare parameters for OpenAI Responses API."""
        instructions, input_content = self._convert_messages_to_responses_format(
            messages
        )

        params: dict[str, Any] = {
            "model": self.model,
            "input": input_content,
        }

        if instructions:
            params["instructions"] = instructions

        if self.stream:
            params["stream"] = self.stream

        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.top_p is not None:
            params["top_p"] = self.top_p
        if self.max_output_tokens is not None:
            params["max_output_tokens"] = self.max_output_tokens

        if self.is_o_model and self.reasoning_effort:
            params["reasoning"] = {"effort": self.reasoning_effort}

        if self.previous_response_id is not None:
            params["previous_response_id"] = self.previous_response_id

        if self.store is not None:
            params["store"] = self.store

        if response_model is not None:
            params["text"] = {
                "format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_model.__name__,
                        "schema": response_model.model_json_schema(),
                        "strict": True,
                    },
                }
            }
        elif self.response_format is not None:
            if isinstance(self.response_format, type) and issubclass(
                self.response_format, BaseModel
            ):
                params["text"] = {
                    "format": generate_model_description(self.response_format)
                }
            elif isinstance(self.response_format, dict):
                params["text"] = {"format": self.response_format}

        if tools:
            params["tools"] = self._convert_tools_for_responses(tools)

        return params

    def _convert_messages_to_responses_format(
        self, messages: list[LLMMessage]
    ) -> tuple[str | None, str | list[dict[str, Any]]]:
        """Convert CrewAI messages to Responses API format.

        The Responses API uses 'instructions' for system messages and 'input'
        for user/assistant messages.

        Args:
            messages: List of messages in CrewAI format

        Returns:
            Tuple of (instructions, input_content)
        """
        instructions_parts: list[str] = []
        input_items: list[dict[str, Any]] = []

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            if role == "system":
                instructions_parts.append(str(content))
            else:
                input_items.append(
                    {
                        "type": "message",
                        "role": role,
                        "content": str(content),
                    }
                )

        instructions = "\n\n".join(instructions_parts) if instructions_parts else None

        if len(input_items) == 1 and input_items[0]["role"] == "user":
            return instructions, input_items[0]["content"]

        return instructions, input_items if input_items else ""

    def _convert_tools_for_responses(
        self, tools: list[dict[str, BaseTool]]
    ) -> list[dict[str, Any]]:
        """Convert CrewAI tool format to Responses API function tool format."""
        from crewai.llms.providers.utils.common import safe_tool_conversion

        responses_tools = []

        for tool in tools:
            name, description, parameters = safe_tool_conversion(tool, "OpenAI")

            responses_tool: dict[str, Any] = {
                "type": "function",
                "name": name,
                "strict": True,
            }

            if description:
                responses_tool["description"] = description

            if parameters:
                if isinstance(parameters, dict):
                    responses_tool["parameters"] = parameters
                else:
                    responses_tool["parameters"] = dict(parameters)

            responses_tools.append(responses_tool)

        return responses_tools

    def _handle_response(
        self,
        params: dict[str, Any],
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[BaseModel] | None = None,
        formatted_messages: list[LLMMessage] | None = None,
    ) -> str | Any:
        """Handle non-streaming response."""
        try:
            response: Response = self.client.responses.create(**params)

            self.last_response_id = response.id

            usage = self._extract_responses_token_usage(response)
            self._track_token_usage_internal(usage)

            for output_item in response.output:
                if isinstance(output_item, ResponseFunctionToolCall):
                    if available_functions:
                        function_name = output_item.name
                        try:
                            function_args = json.loads(output_item.arguments)
                        except json.JSONDecodeError as e:
                            logging.error(f"Failed to parse tool arguments: {e}")
                            function_args = {}

                        result = self._handle_tool_execution(
                            function_name=function_name,
                            function_args=function_args,
                            available_functions=available_functions,
                            from_task=from_task,
                            from_agent=from_agent,
                        )

                        if result is not None:
                            return result

            content = response.output_text or ""
            content = self._apply_stop_words(content)

            if response_model:
                try:
                    structured_result = self._validate_structured_output(
                        content, response_model
                    )
                    self._emit_call_completed_event(
                        response=structured_result,
                        call_type=LLMCallType.LLM_CALL,
                        from_task=from_task,
                        from_agent=from_agent,
                        messages=formatted_messages,
                    )
                    return (
                        structured_result.model_dump_json()
                        if isinstance(structured_result, BaseModel)
                        else structured_result
                    )
                except ValueError as e:
                    logging.warning(f"Structured output validation failed: {e}")

            self._emit_call_completed_event(
                response=content,
                call_type=LLMCallType.LLM_CALL,
                from_task=from_task,
                from_agent=from_agent,
                messages=formatted_messages,
            )

            if usage.get("total_tokens", 0) > 0:
                logging.info(f"OpenAI Responses API usage: {usage}")

            content = self._invoke_after_llm_call_hooks(
                formatted_messages or [], content, from_agent
            )

        except NotFoundError as e:
            error_msg = f"Model {self.model} not found: {e}"
            logging.error(error_msg)
            self._emit_call_failed_event(
                error=error_msg, from_task=from_task, from_agent=from_agent
            )
            raise ValueError(error_msg) from e
        except APIConnectionError as e:
            error_msg = f"Failed to connect to OpenAI API: {e}"
            logging.error(error_msg)
            self._emit_call_failed_event(
                error=error_msg, from_task=from_task, from_agent=from_agent
            )
            raise ConnectionError(error_msg) from e
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

        return content

    async def _ahandle_response(
        self,
        params: dict[str, Any],
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[BaseModel] | None = None,
        formatted_messages: list[LLMMessage] | None = None,
    ) -> str | Any:
        """Handle non-streaming async response."""
        try:
            response: Response = await self.async_client.responses.create(**params)

            self.last_response_id = response.id

            usage = self._extract_responses_token_usage(response)
            self._track_token_usage_internal(usage)

            for output_item in response.output:
                if isinstance(output_item, ResponseFunctionToolCall):
                    if available_functions:
                        function_name = output_item.name
                        try:
                            function_args = json.loads(output_item.arguments)
                        except json.JSONDecodeError as e:
                            logging.error(f"Failed to parse tool arguments: {e}")
                            function_args = {}

                        result = self._handle_tool_execution(
                            function_name=function_name,
                            function_args=function_args,
                            available_functions=available_functions,
                            from_task=from_task,
                            from_agent=from_agent,
                        )

                        if result is not None:
                            return result

            content = response.output_text or ""
            content = self._apply_stop_words(content)

            if response_model:
                try:
                    structured_result = self._validate_structured_output(
                        content, response_model
                    )
                    self._emit_call_completed_event(
                        response=structured_result,
                        call_type=LLMCallType.LLM_CALL,
                        from_task=from_task,
                        from_agent=from_agent,
                        messages=formatted_messages,
                    )
                    return (
                        structured_result.model_dump_json()
                        if isinstance(structured_result, BaseModel)
                        else structured_result
                    )
                except ValueError as e:
                    logging.warning(f"Structured output validation failed: {e}")

            self._emit_call_completed_event(
                response=content,
                call_type=LLMCallType.LLM_CALL,
                from_task=from_task,
                from_agent=from_agent,
                messages=formatted_messages,
            )

            if usage.get("total_tokens", 0) > 0:
                logging.info(f"OpenAI Responses API usage: {usage}")

        except NotFoundError as e:
            error_msg = f"Model {self.model} not found: {e}"
            logging.error(error_msg)
            self._emit_call_failed_event(
                error=error_msg, from_task=from_task, from_agent=from_agent
            )
            raise ValueError(error_msg) from e
        except APIConnectionError as e:
            error_msg = f"Failed to connect to OpenAI API: {e}"
            logging.error(error_msg)
            self._emit_call_failed_event(
                error=error_msg, from_task=from_task, from_agent=from_agent
            )
            raise ConnectionError(error_msg) from e
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

        return content

    def _handle_streaming_response(
        self,
        params: dict[str, Any],
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str:
        """Handle streaming response."""
        full_response = ""
        tool_calls: dict[str, dict[str, str]] = {}

        with self.client.responses.stream(**params) as stream:
            for event in stream:
                if hasattr(event, "type"):
                    if event.type == "response.output_text.delta":
                        delta = getattr(event, "delta", "")
                        if delta:
                            full_response += delta
                            self._emit_stream_chunk_event(
                                chunk=delta,
                                from_task=from_task,
                                from_agent=from_agent,
                            )
                    elif event.type == "response.function_call_arguments.delta":
                        call_id = getattr(event, "call_id", "default")
                        if call_id not in tool_calls:
                            tool_calls[call_id] = {"name": "", "arguments": ""}
                        delta = getattr(event, "delta", "")
                        if delta:
                            tool_calls[call_id]["arguments"] += delta
                    elif event.type == "response.output_item.added":
                        item = getattr(event, "item", None)
                        if item and hasattr(item, "type") and item.type == "function_call":
                            call_id = getattr(item, "call_id", "default")
                            if call_id not in tool_calls:
                                tool_calls[call_id] = {"name": "", "arguments": ""}
                            tool_calls[call_id]["name"] = getattr(item, "name", "")
                    elif event.type == "response.completed":
                        response = getattr(event, "response", None)
                        if response:
                            self.last_response_id = response.id
                            usage = self._extract_responses_token_usage(response)
                            self._track_token_usage_internal(usage)

        if tool_calls and available_functions:
            for call_data in tool_calls.values():
                function_name = call_data["name"]
                arguments = call_data["arguments"]

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

        full_response = self._apply_stop_words(full_response)

        self._emit_call_completed_event(
            response=full_response,
            call_type=LLMCallType.LLM_CALL,
            from_task=from_task,
            from_agent=from_agent,
        )

        return full_response

    async def _ahandle_streaming_response(
        self,
        params: dict[str, Any],
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str:
        """Handle async streaming response."""
        full_response = ""
        tool_calls: dict[str, dict[str, str]] = {}

        async with self.async_client.responses.stream(**params) as stream:
            async for event in stream:
                if hasattr(event, "type"):
                    if event.type == "response.output_text.delta":
                        delta = getattr(event, "delta", "")
                        if delta:
                            full_response += delta
                            self._emit_stream_chunk_event(
                                chunk=delta,
                                from_task=from_task,
                                from_agent=from_agent,
                            )
                    elif event.type == "response.function_call_arguments.delta":
                        call_id = getattr(event, "call_id", "default")
                        if call_id not in tool_calls:
                            tool_calls[call_id] = {"name": "", "arguments": ""}
                        delta = getattr(event, "delta", "")
                        if delta:
                            tool_calls[call_id]["arguments"] += delta
                    elif event.type == "response.output_item.added":
                        item = getattr(event, "item", None)
                        if item and hasattr(item, "type") and item.type == "function_call":
                            call_id = getattr(item, "call_id", "default")
                            if call_id not in tool_calls:
                                tool_calls[call_id] = {"name": "", "arguments": ""}
                            tool_calls[call_id]["name"] = getattr(item, "name", "")
                    elif event.type == "response.completed":
                        response = getattr(event, "response", None)
                        if response:
                            self.last_response_id = response.id
                            usage = self._extract_responses_token_usage(response)
                            self._track_token_usage_internal(usage)

        if tool_calls and available_functions:
            for call_data in tool_calls.values():
                function_name = call_data["name"]
                arguments = call_data["arguments"]

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

        full_response = self._apply_stop_words(full_response)

        self._emit_call_completed_event(
            response=full_response,
            call_type=LLMCallType.LLM_CALL,
            from_task=from_task,
            from_agent=from_agent,
        )

        return full_response

    def supports_function_calling(self) -> bool:
        """Check if the model supports function calling."""
        return True

    def supports_stop_words(self) -> bool:
        """Check if the model supports stop words."""
        return not self.is_o_model

    def get_context_window_size(self) -> int:
        """Get the context window size for the model."""
        from crewai.llm import CONTEXT_WINDOW_USAGE_RATIO

        context_windows = [
            ("gpt-4.1-mini-2025-04-14", 1047576),
            ("gpt-4.1-nano-2025-04-14", 1047576),
            ("gpt-4.1", 1047576),
            ("gpt-4o-mini", 200000),
            ("gpt-4o", 128000),
            ("gpt-4-turbo", 128000),
            ("gpt-4", 8192),
            ("o1-preview", 128000),
            ("o1-mini", 128000),
            ("o3-mini", 200000),
            ("o4-mini", 200000),
        ]

        for model_prefix, size in context_windows:
            if self.model.startswith(model_prefix):
                return int(size * CONTEXT_WINDOW_USAGE_RATIO)

        return int(8192 * CONTEXT_WINDOW_USAGE_RATIO)

    def _extract_responses_token_usage(self, response: Response) -> dict[str, Any]:
        """Extract token usage from Responses API response."""
        if hasattr(response, "usage") and response.usage:
            usage = response.usage
            return {
                "prompt_tokens": getattr(usage, "input_tokens", 0),
                "completion_tokens": getattr(usage, "output_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
            }
        return {"total_tokens": 0}

    def _format_messages(self, messages: str | list[LLMMessage]) -> list[LLMMessage]:
        """Format messages for Responses API."""
        base_formatted = super()._format_messages(messages)

        formatted_messages: list[LLMMessage] = []

        for message in base_formatted:
            if self.is_o_model and message.get("role") == "system":
                formatted_messages.append(
                    {"role": "user", "content": f"System: {message['content']}"}
                )
            else:
                formatted_messages.append(message)

        return formatted_messages
