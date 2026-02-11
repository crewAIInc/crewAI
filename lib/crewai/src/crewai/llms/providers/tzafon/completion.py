"""Tzafon native completion implementation for CrewAI.

Tzafon provides an OpenAI-compatible chat completions API, so this
implementation uses the OpenAI Python SDK with a custom base_url.
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Any

from openai import APIConnectionError, AsyncOpenAI, NotFoundError, OpenAI, Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from pydantic import BaseModel

from crewai.events.types.llm_events import LLMCallType
from crewai.llms.base_llm import BaseLLM, llm_call_context
from crewai.utilities.agent_utils import is_context_length_exceeded
from crewai.utilities.exceptions.context_window_exceeding_exception import (
    LLMContextLengthExceededError,
)
from crewai.utilities.pydantic_schema_utils import generate_model_description


if TYPE_CHECKING:
    from crewai.agent.core import Agent
    from crewai.task import Task
    from crewai.tools.base_tool import BaseTool
    from crewai.utilities.types import LLMMessage


TZAFON_DEFAULT_BASE_URL = "https://api.tzafon.ai/v1"


class TzafonCompletion(BaseLLM):
    """Tzafon native completion implementation.

    Uses the OpenAI Python SDK pointed at the Tzafon API endpoint.

    Args:
        model: Tzafon model identifier (e.g. "tzafon.sm-1").
        api_key: Tzafon API key. Falls back to TZAFON_API_KEY env var.
        base_url: Override the Tzafon API URL. Falls back to TZAFON_BASE_URL
            env var, then defaults to https://api.tzafon.ai/v1.
        temperature: Sampling temperature (0-2).
        max_tokens: Maximum tokens to generate.
        stream: Whether to stream responses.
        stop: Stop sequences.
    """

    def __init__(
        self,
        model: str = "tzafon.sm-1",
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int = 2,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
        response_format: dict[str, Any] | type[BaseModel] | None = None,
        provider: str | None = None,
        **kwargs: Any,
    ) -> None:
        if provider is None:
            provider = kwargs.pop("provider", "tzafon")

        self.timeout = timeout
        self.max_retries = max_retries

        resolved_api_key = api_key or os.getenv("TZAFON_API_KEY")
        resolved_base_url = (
            base_url or os.getenv("TZAFON_BASE_URL") or TZAFON_DEFAULT_BASE_URL
        )

        super().__init__(
            model=model,
            temperature=temperature,
            api_key=resolved_api_key,
            base_url=resolved_base_url,
            timeout=timeout,
            provider=provider,
            **kwargs,
        )

        if self.api_key is None:
            raise ValueError(
                "Tzafon API key is required. Set TZAFON_API_KEY environment "
                "variable or pass api_key parameter."
            )

        client_params: dict[str, Any] = {
            "api_key": self.api_key,
            "base_url": resolved_base_url,
        }
        if timeout is not None:
            client_params["timeout"] = timeout
        if max_retries is not None:
            client_params["max_retries"] = max_retries

        self.client = OpenAI(**client_params)
        self.async_client = AsyncOpenAI(**client_params)

        self.max_tokens = max_tokens
        self.stream = stream
        self.response_format = response_format

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
        """Call Tzafon chat completions API.

        Args:
            messages: Input messages for the completion.
            tools: List of tool/function definitions.
            callbacks: Callback functions (unused).
            available_functions: Available functions for tool calling.
            from_task: Task that initiated the call.
            from_agent: Agent that initiated the call.
            response_model: Pydantic model for structured output.

        Returns:
            Completion response text or tool call result.
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

                formatted_messages = self._format_messages(messages)

                if not self._invoke_before_llm_call_hooks(
                    formatted_messages, from_agent
                ):
                    raise ValueError("LLM call blocked by before_llm_call hook")

                params = self._prepare_params(messages=formatted_messages, tools=tools)

                if self.stream:
                    return self._handle_streaming(
                        params=params,
                        available_functions=available_functions,
                        from_task=from_task,
                        from_agent=from_agent,
                        response_model=response_model,
                    )

                return self._handle_completion(
                    params=params,
                    available_functions=available_functions,
                    from_task=from_task,
                    from_agent=from_agent,
                    response_model=response_model,
                )

            except Exception as e:
                error_msg = f"Tzafon API call failed: {e!s}"
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
        """Async call to Tzafon chat completions API."""
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

                formatted_messages = self._format_messages(messages)

                params = self._prepare_params(messages=formatted_messages, tools=tools)

                return await self._ahandle_completion(
                    params=params,
                    available_functions=available_functions,
                    from_task=from_task,
                    from_agent=from_agent,
                    response_model=response_model,
                )

            except Exception as e:
                error_msg = f"Tzafon API call failed: {e!s}"
                logging.error(error_msg)
                self._emit_call_failed_event(
                    error=error_msg, from_task=from_task, from_agent=from_agent
                )
                raise

    def _prepare_params(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, BaseTool]] | None = None,
    ) -> dict[str, Any]:
        """Prepare parameters for chat completion request."""
        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }

        if self.stream:
            params["stream"] = True
            params["stream_options"] = {"include_usage": True}

        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        if self.stop:
            params["stop"] = self.stop

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

        return params

    def _convert_tools_for_interference(
        self, tools: list[dict[str, BaseTool]]
    ) -> list[dict[str, Any]]:
        """Convert CrewAI tool format to OpenAI function calling format."""
        from crewai.llms.providers.utils.common import safe_tool_conversion

        openai_tools = []
        for tool in tools:
            name, description, parameters = safe_tool_conversion(tool, "Tzafon")

            openai_tool: dict[str, Any] = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                },
            }

            if parameters:
                params_dict = (
                    parameters if isinstance(parameters, dict) else dict(parameters)
                )
                openai_tool["function"]["parameters"] = params_dict

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

                usage = self._extract_token_usage(parsed_response)
                self._track_token_usage_internal(usage)

                parsed_object = parsed_response.choices[0].message.parsed
                if parsed_object:
                    self._emit_call_completed_event(
                        response=parsed_object.model_dump_json(),
                        call_type=LLMCallType.LLM_CALL,
                        from_task=from_task,
                        from_agent=from_agent,
                        messages=params["messages"],
                    )
                    return parsed_object

            response: ChatCompletion = self.client.chat.completions.create(**params)

            usage = self._extract_token_usage(response)
            self._track_token_usage_internal(usage)

            choice: Choice = response.choices[0]
            message = choice.message

            # Tool calls without available_functions: return raw tool calls
            if message.tool_calls and not available_functions:
                self._emit_call_completed_event(
                    response=list(message.tool_calls),
                    call_type=LLMCallType.TOOL_CALL,
                    from_task=from_task,
                    from_agent=from_agent,
                    messages=params["messages"],
                )
                return list(message.tool_calls)

            # Tool calls with available_functions: execute the tool
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

            content = self._apply_stop_words(content)

            self._emit_call_completed_event(
                response=content,
                call_type=LLMCallType.LLM_CALL,
                from_task=from_task,
                from_agent=from_agent,
                messages=params["messages"],
            )

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
            error_msg = f"Failed to connect to Tzafon API: {e}"
            logging.error(error_msg)
            self._emit_call_failed_event(
                error=error_msg, from_task=from_task, from_agent=from_agent
            )
            raise ConnectionError(error_msg) from e
        except Exception as e:
            if is_context_length_exceeded(e):
                logging.error(f"Context window exceeded: {e}")
                raise LLMContextLengthExceededError(str(e)) from e

            error_msg = f"Tzafon API call failed: {e!s}"
            logging.error(error_msg)
            self._emit_call_failed_event(
                error=error_msg, from_task=from_task, from_agent=from_agent
            )
            raise e from e

        return content

    def _handle_streaming(
        self,
        params: dict[str, Any],
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | BaseModel:
        """Handle streaming chat completion."""
        full_response = ""
        tool_calls: dict[int, dict[str, Any]] = {}

        if response_model:
            parse_params = {
                k: v
                for k, v in params.items()
                if k not in ("response_format", "stream", "stream_options")
            }
            with self.client.beta.chat.completions.stream(
                **parse_params, response_format=response_model
            ) as stream:
                for chunk in stream:
                    response_id = chunk.id if hasattr(chunk, "id") else None
                    if chunk.type == "content.delta":
                        delta_content = chunk.delta
                        if delta_content:
                            self._emit_stream_chunk_event(
                                chunk=delta_content,
                                from_task=from_task,
                                from_agent=from_agent,
                                response_id=response_id,
                            )

                final_completion = stream.get_final_completion()
                if final_completion:
                    usage = self._extract_token_usage(final_completion)
                    self._track_token_usage_internal(usage)
                    if final_completion.choices:
                        parsed_result = final_completion.choices[0].message.parsed
                        if parsed_result:
                            self._emit_call_completed_event(
                                response=parsed_result.model_dump_json(),
                                call_type=LLMCallType.LLM_CALL,
                                from_task=from_task,
                                from_agent=from_agent,
                                messages=params["messages"],
                            )
                            return parsed_result

            return ""

        completion_stream: Stream[ChatCompletionChunk] = (
            self.client.chat.completions.create(**params)
        )

        usage_data: dict[str, Any] = {"total_tokens": 0}

        for completion_chunk in completion_stream:
            response_id = (
                completion_chunk.id if hasattr(completion_chunk, "id") else None
            )

            if hasattr(completion_chunk, "usage") and completion_chunk.usage:
                usage_data = self._extract_token_usage(completion_chunk)
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
                    response_id=response_id,
                )

            if chunk_delta.tool_calls:
                for tool_call in chunk_delta.tool_calls:
                    tool_index = tool_call.index if tool_call.index is not None else 0
                    if tool_index not in tool_calls:
                        tool_calls[tool_index] = {
                            "id": tool_call.id,
                            "name": "",
                            "arguments": "",
                            "index": tool_index,
                        }
                    elif tool_call.id and not tool_calls[tool_index]["id"]:
                        tool_calls[tool_index]["id"] = tool_call.id

                    if tool_call.function and tool_call.function.name:
                        tool_calls[tool_index]["name"] = tool_call.function.name
                    if tool_call.function and tool_call.function.arguments:
                        tool_calls[tool_index]["arguments"] += (
                            tool_call.function.arguments
                        )

                    self._emit_stream_chunk_event(
                        chunk=tool_call.function.arguments
                        if tool_call.function and tool_call.function.arguments
                        else "",
                        from_task=from_task,
                        from_agent=from_agent,
                        tool_call={
                            "id": tool_calls[tool_index]["id"],
                            "function": {
                                "name": tool_calls[tool_index]["name"],
                                "arguments": tool_calls[tool_index]["arguments"],
                            },
                            "type": "function",
                            "index": tool_calls[tool_index]["index"],
                        },
                        call_type=LLMCallType.TOOL_CALL,
                        response_id=response_id,
                    )

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
        """Handle async non-streaming chat completion."""
        try:
            if response_model:
                parse_params = {
                    k: v for k, v in params.items() if k != "response_format"
                }
                parsed_response = await self.async_client.beta.chat.completions.parse(
                    **parse_params,
                    response_format=response_model,
                )

                usage = self._extract_token_usage(parsed_response)
                self._track_token_usage_internal(usage)

                parsed_object = parsed_response.choices[0].message.parsed
                if parsed_object:
                    self._emit_call_completed_event(
                        response=parsed_object.model_dump_json(),
                        call_type=LLMCallType.LLM_CALL,
                        from_task=from_task,
                        from_agent=from_agent,
                        messages=params["messages"],
                    )
                    return parsed_object

            response: ChatCompletion = await self.async_client.chat.completions.create(
                **params
            )

            usage = self._extract_token_usage(response)
            self._track_token_usage_internal(usage)

            choice: Choice = response.choices[0]
            message = choice.message

            if message.tool_calls and not available_functions:
                self._emit_call_completed_event(
                    response=list(message.tool_calls),
                    call_type=LLMCallType.TOOL_CALL,
                    from_task=from_task,
                    from_agent=from_agent,
                    messages=params["messages"],
                )
                return list(message.tool_calls)

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

            self._emit_call_completed_event(
                response=content,
                call_type=LLMCallType.LLM_CALL,
                from_task=from_task,
                from_agent=from_agent,
                messages=params["messages"],
            )

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
            error_msg = f"Failed to connect to Tzafon API: {e}"
            logging.error(error_msg)
            self._emit_call_failed_event(
                error=error_msg, from_task=from_task, from_agent=from_agent
            )
            raise ConnectionError(error_msg) from e
        except Exception as e:
            if is_context_length_exceeded(e):
                raise LLMContextLengthExceededError(str(e)) from e
            raise

        return content

    def supports_stop_words(self) -> bool:
        """Check if the model supports stop words."""
        return self._supports_stop_words_implementation()

    def get_context_window_size(self) -> int:
        """Get the context window size for the model."""
        from crewai.llm import CONTEXT_WINDOW_USAGE_RATIO, LLM_CONTEXT_WINDOW_SIZES

        # Check the global context window sizes map first
        for model_key, size in LLM_CONTEXT_WINDOW_SIZES.items():
            if self.model.startswith(model_key):
                return int(size * CONTEXT_WINDOW_USAGE_RATIO)

        # Conservative default (Tzafon doesn't document context window sizes)
        return int(8192 * CONTEXT_WINDOW_USAGE_RATIO)

    def supports_multimodal(self) -> bool:
        """Tzafon does not currently document multimodal support."""
        return False

    def _extract_token_usage(
        self, response: ChatCompletion | ChatCompletionChunk
    ) -> dict[str, Any]:
        """Extract token usage from a chat completion response."""
        if hasattr(response, "usage") and response.usage:
            usage = response.usage
            return {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
            }
        return {"total_tokens": 0}
