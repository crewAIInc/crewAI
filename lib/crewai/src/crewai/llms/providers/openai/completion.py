from __future__ import annotations

from collections.abc import AsyncIterator
import json
import logging
import os
from typing import TYPE_CHECKING, Any

import httpx
from openai import APIConnectionError, AsyncOpenAI, NotFoundError, OpenAI, Stream
from openai.lib.streaming.chat import ChatCompletionStream
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import ChoiceDelta
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


class OpenAICompletion(BaseLLM):
    """OpenAI native completion implementation.

    This class provides direct integration with the OpenAI Python SDK,
    offering native structured outputs, function calling, and streaming support.
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
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        max_tokens: int | None = None,
        max_completion_tokens: int | None = None,
        seed: int | None = None,
        stream: bool = False,
        response_format: dict[str, Any] | type[BaseModel] | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
        reasoning_effort: str | None = None,
        provider: str | None = None,
        interceptor: BaseInterceptor[httpx.Request, httpx.Response] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize OpenAI chat completion client."""

        if provider is None:
            provider = kwargs.pop("provider", "openai")

        self.interceptor = interceptor
        # Client configuration attributes
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

        # Completion parameters
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.max_tokens = max_tokens
        self.max_completion_tokens = max_completion_tokens
        self.seed = seed
        self.stream = stream
        self.response_format = response_format
        self.logprobs = logprobs
        self.top_logprobs = top_logprobs
        self.reasoning_effort = reasoning_effort
        self.is_o1_model = "o1" in model.lower()
        self.is_gpt4_model = "gpt-4" in model.lower()

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
        """Call OpenAI chat completion API.

        Args:
            messages: Input messages for the chat completion
            tools: list of tool/function definitions
            callbacks: Callback functions (not used in native implementation)
            available_functions: Available functions for tool calling
            from_task: Task that initiated the call
            from_agent: Agent that initiated the call
            response_model: Response model for structured output.

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

            formatted_messages = self._format_messages(messages)

            if not self._invoke_before_llm_call_hooks(formatted_messages, from_agent):
                raise ValueError("LLM call blocked by before_llm_call hook")

            completion_params = self._prepare_completion_params(
                messages=formatted_messages, tools=tools
            )

            if self.stream:
                return self._handle_streaming_completion(
                    params=completion_params,
                    available_functions=available_functions,
                    from_task=from_task,
                    from_agent=from_agent,
                    response_model=response_model,
                )

            return self._handle_completion(
                params=completion_params,
                available_functions=available_functions,
                from_task=from_task,
                from_agent=from_agent,
                response_model=response_model,
            )

        except Exception as e:
            error_msg = f"OpenAI API call failed: {e!s}"
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
        """Async call to OpenAI chat completion API.

        Args:
            messages: Input messages for the chat completion
            tools: list of tool/function definitions
            callbacks: Callback functions (not used in native implementation)
            available_functions: Available functions for tool calling
            from_task: Task that initiated the call
            from_agent: Agent that initiated the call
            response_model: Response model for structured output.

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

            formatted_messages = self._format_messages(messages)

            completion_params = self._prepare_completion_params(
                messages=formatted_messages, tools=tools
            )

            if self.stream:
                return await self._ahandle_streaming_completion(
                    params=completion_params,
                    available_functions=available_functions,
                    from_task=from_task,
                    from_agent=from_agent,
                    response_model=response_model,
                )

            return await self._ahandle_completion(
                params=completion_params,
                available_functions=available_functions,
                from_task=from_task,
                from_agent=from_agent,
                response_model=response_model,
            )

        except Exception as e:
            error_msg = f"OpenAI API call failed: {e!s}"
            logging.error(error_msg)
            self._emit_call_failed_event(
                error=error_msg, from_task=from_task, from_agent=from_agent
            )
            raise

    def _prepare_completion_params(
        self, messages: list[LLMMessage], tools: list[dict[str, BaseTool]] | None = None
    ) -> dict[str, Any]:
        """Prepare parameters for OpenAI chat completion."""
        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if self.stream:
            params["stream"] = self.stream
            params["stream_options"] = {"include_usage": True}

        params.update(self.additional_params)

        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.top_p is not None:
            params["top_p"] = self.top_p
        if self.frequency_penalty is not None:
            params["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty is not None:
            params["presence_penalty"] = self.presence_penalty
        if self.max_completion_tokens is not None:
            params["max_completion_tokens"] = self.max_completion_tokens
        elif self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        if self.seed is not None:
            params["seed"] = self.seed
        if self.logprobs is not None:
            params["logprobs"] = self.logprobs
        if self.top_logprobs is not None:
            params["top_logprobs"] = self.top_logprobs

        # Handle o1 model specific parameters
        if self.is_o1_model and self.reasoning_effort:
            params["reasoning_effort"] = self.reasoning_effort

        if self.response_format is not None:
            if isinstance(self.response_format, type) and issubclass(
                self.response_format, BaseModel
            ):
                params["response_format"] = generate_model_description(
                    self.response_format
                )
            elif isinstance(self.response_format, dict):
                params["response_format"] = self.response_format

        if tools:
            params["tools"] = self._convert_tools_for_interference(tools)
            params["tool_choice"] = "auto"

        # Filter out CrewAI-specific parameters that shouldn't go to the API
        crewai_specific_params = {
            "callbacks",
            "available_functions",
            "from_task",
            "from_agent",
            "provider",
            "api_key",
            "base_url",
            "api_base",
            "timeout",
        }

        return {k: v for k, v in params.items() if k not in crewai_specific_params}

    def _convert_tools_for_interference(
        self, tools: list[dict[str, BaseTool]]
    ) -> list[dict[str, Any]]:
        """Convert CrewAI tool format to OpenAI function calling format."""
        from crewai.llms.providers.utils.common import safe_tool_conversion

        openai_tools = []

        for tool in tools:
            name, description, parameters = safe_tool_conversion(tool, "OpenAI")

            openai_tool = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                },
            }

            if parameters:
                if isinstance(parameters, dict):
                    openai_tool["function"]["parameters"] = parameters  # type: ignore
                else:
                    openai_tool["function"]["parameters"] = dict(parameters)

            openai_tools.append(openai_tool)
        return openai_tools

    def _handle_completion(
        self,
        params: dict[str, Any],
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        """Handle non-streaming chat completion."""
        try:
            if response_model:
                parse_params = {
                    k: v for k, v in params.items() if k != "response_format"
                }
                parsed_response = self.client.beta.chat.completions.parse(
                    **parse_params,
                    response_format=response_model,
                )
                math_reasoning = parsed_response.choices[0].message

                if math_reasoning.refusal:
                    pass

                usage = self._extract_openai_token_usage(parsed_response)
                self._track_token_usage_internal(usage)

                parsed_object = parsed_response.choices[0].message.parsed
                if parsed_object:
                    structured_json = parsed_object.model_dump_json()
                    self._emit_call_completed_event(
                        response=structured_json,
                        call_type=LLMCallType.LLM_CALL,
                        from_task=from_task,
                        from_agent=from_agent,
                        messages=params["messages"],
                    )
                    return structured_json

            response: ChatCompletion = self.client.chat.completions.create(**params)

            usage = self._extract_openai_token_usage(response)

            self._track_token_usage_internal(usage)

            choice: Choice = response.choices[0]
            message = choice.message

            if message.tool_calls and available_functions:
                tool_call = message.tool_calls[0]
                function_name = tool_call.function.name

                try:
                    function_args = json.loads(tool_call.function.arguments)
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

            content = message.content or ""
            content = self._apply_stop_words(content)

            if self.response_format and isinstance(self.response_format, type):
                try:
                    structured_result = self._validate_structured_output(
                        content, self.response_format
                    )
                    self._emit_call_completed_event(
                        response=structured_result,
                        call_type=LLMCallType.LLM_CALL,
                        from_task=from_task,
                        from_agent=from_agent,
                        messages=params["messages"],
                    )
                    return structured_result
                except ValueError as e:
                    logging.warning(f"Structured output validation failed: {e}")

            self._emit_call_completed_event(
                response=content,
                call_type=LLMCallType.LLM_CALL,
                from_task=from_task,
                from_agent=from_agent,
                messages=params["messages"],
            )

            if usage.get("total_tokens", 0) > 0:
                logging.info(f"OpenAI API usage: {usage}")

            content = self._invoke_after_llm_call_hooks(
                params["messages"], content, from_agent
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
            # Handle context length exceeded and other errors
            if is_context_length_exceeded(e):
                logging.error(f"Context window exceeded: {e}")
                raise LLMContextLengthExceededError(str(e)) from e

            error_msg = f"OpenAI API call failed: {e!s}"
            logging.error(error_msg)
            self._emit_call_failed_event(
                error=error_msg, from_task=from_task, from_agent=from_agent
            )
            raise e from e

        return content

    def _handle_streaming_completion(
        self,
        params: dict[str, Any],
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str:
        """Handle streaming chat completion."""
        full_response = ""
        tool_calls = {}

        if response_model:
            parse_params = {
                k: v
                for k, v in params.items()
                if k not in ("response_format", "stream")
            }

            stream: ChatCompletionStream[BaseModel]
            with self.client.beta.chat.completions.stream(
                **parse_params, response_format=response_model
            ) as stream:
                for chunk in stream:
                    if chunk.type == "content.delta":
                        delta_content = chunk.delta
                        if delta_content:
                            self._emit_stream_chunk_event(
                                chunk=delta_content,
                                from_task=from_task,
                                from_agent=from_agent,
                            )

                final_completion = stream.get_final_completion()
                if final_completion:
                    usage = self._extract_openai_token_usage(final_completion)
                    self._track_token_usage_internal(usage)
                    if final_completion.choices:
                        parsed_result = final_completion.choices[0].message.parsed
                        if parsed_result:
                            structured_json = parsed_result.model_dump_json()
                            self._emit_call_completed_event(
                                response=structured_json,
                                call_type=LLMCallType.LLM_CALL,
                                from_task=from_task,
                                from_agent=from_agent,
                                messages=params["messages"],
                            )
                            return structured_json

            logging.error("Failed to get parsed result from stream")
            return ""

        completion_stream: Stream[ChatCompletionChunk] = (
            self.client.chat.completions.create(**params)
        )

        usage_data = {"total_tokens": 0}

        for completion_chunk in completion_stream:
            if hasattr(completion_chunk, "usage") and completion_chunk.usage:
                usage_data = self._extract_openai_token_usage(completion_chunk)
                continue

            if not completion_chunk.choices:
                continue

            choice = completion_chunk.choices[0]
            chunk_delta: ChoiceDelta = choice.delta

            if chunk_delta.content:
                full_response += chunk_delta.content
                self._emit_stream_chunk_event(
                    chunk=chunk_delta.content,
                    from_task=from_task,
                    from_agent=from_agent,
                )

            if chunk_delta.tool_calls:
                for tool_call in chunk_delta.tool_calls:
                    call_id = tool_call.id or "default"
                    if call_id not in tool_calls:
                        tool_calls[call_id] = {
                            "name": "",
                            "arguments": "",
                        }

                    if tool_call.function and tool_call.function.name:
                        tool_calls[call_id]["name"] = tool_call.function.name
                    if tool_call.function and tool_call.function.arguments:
                        tool_calls[call_id]["arguments"] += tool_call.function.arguments

        self._track_token_usage_internal(usage_data)

        if tool_calls and available_functions:
            for call_data in tool_calls.values():
                function_name = call_data["name"]
                arguments = call_data["arguments"]

                # Skip if function name is empty or arguments are empty
                if not function_name or not arguments:
                    continue

                # Check if function exists in available functions
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
            messages=params["messages"],
        )

        return self._invoke_after_llm_call_hooks(
            params["messages"], full_response, from_agent
        )

    async def _ahandle_completion(
        self,
        params: dict[str, Any],
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        """Handle non-streaming async chat completion."""
        try:
            if response_model:
                parse_params = {
                    k: v for k, v in params.items() if k != "response_format"
                }
                parsed_response = await self.async_client.beta.chat.completions.parse(
                    **parse_params,
                    response_format=response_model,
                )
                math_reasoning = parsed_response.choices[0].message

                if math_reasoning.refusal:
                    pass

                usage = self._extract_openai_token_usage(parsed_response)
                self._track_token_usage_internal(usage)

                parsed_object = parsed_response.choices[0].message.parsed
                if parsed_object:
                    structured_json = parsed_object.model_dump_json()
                    self._emit_call_completed_event(
                        response=structured_json,
                        call_type=LLMCallType.LLM_CALL,
                        from_task=from_task,
                        from_agent=from_agent,
                        messages=params["messages"],
                    )
                    return structured_json

            response: ChatCompletion = await self.async_client.chat.completions.create(
                **params
            )

            usage = self._extract_openai_token_usage(response)

            self._track_token_usage_internal(usage)

            choice: Choice = response.choices[0]
            message = choice.message

            if message.tool_calls and available_functions:
                tool_call = message.tool_calls[0]
                function_name = tool_call.function.name

                try:
                    function_args = json.loads(tool_call.function.arguments)
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

            content = message.content or ""
            content = self._apply_stop_words(content)

            if self.response_format and isinstance(self.response_format, type):
                try:
                    structured_result = self._validate_structured_output(
                        content, self.response_format
                    )
                    self._emit_call_completed_event(
                        response=structured_result,
                        call_type=LLMCallType.LLM_CALL,
                        from_task=from_task,
                        from_agent=from_agent,
                        messages=params["messages"],
                    )
                    return structured_result
                except ValueError as e:
                    logging.warning(f"Structured output validation failed: {e}")

            self._emit_call_completed_event(
                response=content,
                call_type=LLMCallType.LLM_CALL,
                from_task=from_task,
                from_agent=from_agent,
                messages=params["messages"],
            )

            if usage.get("total_tokens", 0) > 0:
                logging.info(f"OpenAI API usage: {usage}")
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

            error_msg = f"OpenAI API call failed: {e!s}"
            logging.error(error_msg)
            self._emit_call_failed_event(
                error=error_msg, from_task=from_task, from_agent=from_agent
            )
            raise e from e

        return content

    async def _ahandle_streaming_completion(
        self,
        params: dict[str, Any],
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str:
        """Handle async streaming chat completion."""
        full_response = ""
        tool_calls = {}

        if response_model:
            completion_stream: AsyncIterator[
                ChatCompletionChunk
            ] = await self.async_client.chat.completions.create(**params)

            accumulated_content = ""
            usage_data = {"total_tokens": 0}
            async for chunk in completion_stream:
                if hasattr(chunk, "usage") and chunk.usage:
                    usage_data = self._extract_openai_token_usage(chunk)
                    continue

                if not chunk.choices:
                    continue

                choice = chunk.choices[0]
                delta: ChoiceDelta = choice.delta

                if delta.content:
                    accumulated_content += delta.content
                    self._emit_stream_chunk_event(
                        chunk=delta.content,
                        from_task=from_task,
                        from_agent=from_agent,
                    )

            self._track_token_usage_internal(usage_data)

            try:
                parsed_object = response_model.model_validate_json(accumulated_content)
                structured_json = parsed_object.model_dump_json()

                self._emit_call_completed_event(
                    response=structured_json,
                    call_type=LLMCallType.LLM_CALL,
                    from_task=from_task,
                    from_agent=from_agent,
                    messages=params["messages"],
                )

                return structured_json
            except Exception as e:
                logging.error(f"Failed to parse structured output from stream: {e}")
                self._emit_call_completed_event(
                    response=accumulated_content,
                    call_type=LLMCallType.LLM_CALL,
                    from_task=from_task,
                    from_agent=from_agent,
                    messages=params["messages"],
                )
                return accumulated_content

        stream: AsyncIterator[
            ChatCompletionChunk
        ] = await self.async_client.chat.completions.create(**params)

        usage_data = {"total_tokens": 0}

        async for chunk in stream:
            if hasattr(chunk, "usage") and chunk.usage:
                usage_data = self._extract_openai_token_usage(chunk)
                continue

            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            chunk_delta: ChoiceDelta = choice.delta

            if chunk_delta.content:
                full_response += chunk_delta.content
                self._emit_stream_chunk_event(
                    chunk=chunk_delta.content,
                    from_task=from_task,
                    from_agent=from_agent,
                )

            if chunk_delta.tool_calls:
                for tool_call in chunk_delta.tool_calls:
                    call_id = tool_call.id or "default"
                    if call_id not in tool_calls:
                        tool_calls[call_id] = {
                            "name": "",
                            "arguments": "",
                        }

                    if tool_call.function and tool_call.function.name:
                        tool_calls[call_id]["name"] = tool_call.function.name
                    if tool_call.function and tool_call.function.arguments:
                        tool_calls[call_id]["arguments"] += tool_call.function.arguments

        self._track_token_usage_internal(usage_data)

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
            messages=params["messages"],
        )

        return full_response

    def supports_function_calling(self) -> bool:
        """Check if the model supports function calling."""
        return not self.is_o1_model

    def supports_stop_words(self) -> bool:
        """Check if the model supports stop words."""
        return not self.is_o1_model

    def get_context_window_size(self) -> int:
        """Get the context window size for the model."""
        from crewai.llm import CONTEXT_WINDOW_USAGE_RATIO, LLM_CONTEXT_WINDOW_SIZES

        min_context = 1024
        max_context = 2097152

        for key, value in LLM_CONTEXT_WINDOW_SIZES.items():
            if value < min_context or value > max_context:
                raise ValueError(
                    f"Context window for {key} must be between {min_context} and {max_context}"
                )

        # Context window sizes for OpenAI models
        context_windows = {
            "gpt-4": 8192,
            "gpt-4o": 128000,
            "gpt-4o-mini": 200000,
            "gpt-4-turbo": 128000,
            "gpt-4.1": 1047576,
            "gpt-4.1-mini-2025-04-14": 1047576,
            "gpt-4.1-nano-2025-04-14": 1047576,
            "o1-preview": 128000,
            "o1-mini": 128000,
            "o3-mini": 200000,
            "o4-mini": 200000,
        }

        # Find the best match for the model name
        for model_prefix, size in context_windows.items():
            if self.model.startswith(model_prefix):
                return int(size * CONTEXT_WINDOW_USAGE_RATIO)

        # Default context window size
        return int(8192 * CONTEXT_WINDOW_USAGE_RATIO)

    def _extract_openai_token_usage(
        self, response: ChatCompletion | ChatCompletionChunk
    ) -> dict[str, Any]:
        """Extract token usage from OpenAI ChatCompletion or ChatCompletionChunk response."""
        if hasattr(response, "usage") and response.usage:
            usage = response.usage
            return {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
            }
        return {"total_tokens": 0}

    def _format_messages(self, messages: str | list[LLMMessage]) -> list[LLMMessage]:
        """Format messages for OpenAI API."""
        base_formatted = super()._format_messages(messages)

        # Apply OpenAI-specific formatting
        formatted_messages: list[LLMMessage] = []

        for message in base_formatted:
            if self.is_o1_model and message.get("role") == "system":
                formatted_messages.append(
                    {"role": "user", "content": f"System: {message['content']}"}
                )
            else:
                formatted_messages.append(message)

        return formatted_messages
