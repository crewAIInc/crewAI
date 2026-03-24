"""MiniMax native completion implementation for CrewAI.

MiniMax provides an OpenAI-compatible API, so this provider uses the
``openai`` Python SDK with MiniMax's base URL and API key.

Key constraints:
- temperature must be in (0.0, 1.0]; zero is rejected by the API.
- response_format (JSON mode / structured outputs) is not supported;
  it is silently stripped to avoid API errors.
- Available chat models:
    * MiniMax-M2.7  (default, latest flagship with enhanced reasoning and coding)
    * MiniMax-M2.7-highspeed  (high-speed version of M2.7 for low-latency scenarios)
    * MiniMax-M2.5  (previous generation, 204 800-token context)
    * MiniMax-M2.5-highspeed  (same context, faster)
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Any

import httpx
from openai import APIConnectionError, AsyncOpenAI, NotFoundError, OpenAI, Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from pydantic import BaseModel

from crewai.events.types.llm_events import LLMCallType
from crewai.llms.base_llm import BaseLLM, llm_call_context
from crewai.llms.hooks.transport import AsyncHTTPTransport, HTTPTransport
from crewai.utilities.agent_utils import is_context_length_exceeded
from crewai.utilities.exceptions.context_window_exceeding_exception import (
    LLMContextLengthExceededError,
)
from crewai.utilities.types import LLMMessage


if TYPE_CHECKING:
    from crewai.agent.core import Agent
    from crewai.llms.hooks.base import BaseInterceptor
    from crewai.task import Task
    from crewai.tools.base_tool import BaseTool


# MiniMax-specific defaults
MINIMAX_DEFAULT_BASE_URL = "https://api.minimax.io/v1"
MINIMAX_CONTEXT_WINDOW = 204_800
MINIMAX_DEFAULT_MODEL = "MiniMax-M2.7"


class MiniMaxCompletion(BaseLLM):
    """MiniMax native completion implementation.

    Uses the OpenAI Python SDK pointed at MiniMax's OpenAI-compatible
    endpoint (``https://api.minimax.io/v1`` by default).

    Args:
        model: Model identifier (default ``MiniMax-M2.7``).
        api_key: MiniMax API key.  Falls back to ``MINIMAX_API_KEY`` env var.
        base_url: Override the default MiniMax API endpoint.
        temperature: Sampling temperature, must be in (0.0, 1.0].
            Defaults to 1.0.  Values <= 0 are clamped to 0.01.
    """

    def __init__(
        self,
        model: str = MINIMAX_DEFAULT_MODEL,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int = 2,
        default_headers: dict[str, str] | None = None,
        client_params: dict[str, Any] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        max_completion_tokens: int | None = None,
        seed: int | None = None,
        stream: bool = False,
        provider: str | None = None,
        interceptor: BaseInterceptor[httpx.Request, httpx.Response] | None = None,
        **kwargs: Any,
    ) -> None:
        if provider is None:
            provider = kwargs.pop("provider", "minimax")

        self.interceptor = interceptor
        self.max_retries = max_retries
        self.default_headers = default_headers
        self.client_params = client_params
        self.timeout = timeout

        # MiniMax temperature constraint: must be in (0.0, 1.0]
        if temperature is not None and temperature <= 0:
            temperature = 0.01
        if temperature is None:
            temperature = 1.0

        resolved_base_url = (
            base_url
            or os.getenv("MINIMAX_BASE_URL")
            or MINIMAX_DEFAULT_BASE_URL
        )

        super().__init__(
            model=model,
            temperature=temperature,
            api_key=api_key or os.getenv("MINIMAX_API_KEY"),
            base_url=resolved_base_url,
            timeout=timeout,
            provider=provider,
            **kwargs,
        )

        client_cfg = self._get_client_params()
        if self.interceptor:
            transport = HTTPTransport(interceptor=self.interceptor)
            http_client = httpx.Client(transport=transport)
            client_cfg["http_client"] = http_client

        self.client = OpenAI(**client_cfg)

        async_cfg = self._get_client_params()
        if self.interceptor:
            async_transport = AsyncHTTPTransport(interceptor=self.interceptor)
            async_http_client = httpx.AsyncClient(transport=async_transport)
            async_cfg["http_client"] = async_http_client

        self.async_client = AsyncOpenAI(**async_cfg)

        # Completion parameters
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.max_completion_tokens = max_completion_tokens
        self.seed = seed
        self.stream = stream

    # ------------------------------------------------------------------
    # Client helpers
    # ------------------------------------------------------------------

    def _get_client_params(self) -> dict[str, Any]:
        if self.api_key is None:
            self.api_key = os.getenv("MINIMAX_API_KEY")
            if self.api_key is None:
                raise ValueError(
                    "MINIMAX_API_KEY is required. "
                    "Set it as an environment variable or pass api_key."
                )

        base_params: dict[str, Any] = {
            "api_key": self.api_key,
            "base_url": self.base_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "default_headers": self.default_headers,
        }
        cfg = {k: v for k, v in base_params.items() if v is not None}

        if self.client_params:
            cfg.update(self.client_params)

        return cfg

    # ------------------------------------------------------------------
    # Synchronous call
    # ------------------------------------------------------------------

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

                params = self._prepare_params(formatted_messages, tools)

                if self.stream:
                    return self._handle_streaming(
                        params, available_functions, from_task, from_agent
                    )

                return self._handle_completion(
                    params, available_functions, from_task, from_agent
                )

            except Exception as e:
                error_msg = f"MiniMax API call failed: {e!s}"
                logging.error(error_msg)
                self._emit_call_failed_event(
                    error=error_msg, from_task=from_task, from_agent=from_agent
                )
                raise

    # ------------------------------------------------------------------
    # Asynchronous call
    # ------------------------------------------------------------------

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

                params = self._prepare_params(formatted_messages, tools)

                if self.stream:
                    return await self._ahandle_streaming(
                        params, available_functions, from_task, from_agent
                    )

                return await self._ahandle_completion(
                    params, available_functions, from_task, from_agent
                )

            except Exception as e:
                error_msg = f"MiniMax API call failed: {e!s}"
                logging.error(error_msg)
                self._emit_call_failed_event(
                    error=error_msg, from_task=from_task, from_agent=from_agent
                )
                raise

    # ------------------------------------------------------------------
    # Parameter preparation
    # ------------------------------------------------------------------

    def _prepare_params(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, BaseTool]] | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }

        if self.stream:
            params["stream"] = True
            params["stream_options"] = {"include_usage": True}

        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.top_p is not None:
            params["top_p"] = self.top_p
        if self.max_completion_tokens is not None:
            params["max_completion_tokens"] = self.max_completion_tokens
        elif self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        if self.seed is not None:
            params["seed"] = self.seed

        # MiniMax does not support response_format; intentionally omitted.

        if tools:
            params["tools"] = self._convert_tools_for_interference(tools)
            params["tool_choice"] = "auto"

        return params

    def _convert_tools_for_interference(
        self, tools: list[dict[str, BaseTool]]
    ) -> list[dict[str, Any]]:
        from crewai.llms.providers.utils.common import safe_tool_conversion

        openai_tools: list[dict[str, Any]] = []
        for tool in tools:
            name, description, parameters = safe_tool_conversion(tool, "MiniMax")
            tool_def: dict[str, Any] = {
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
                tool_def["function"]["parameters"] = params_dict
            openai_tools.append(tool_def)
        return openai_tools

    # ------------------------------------------------------------------
    # Non-streaming completion
    # ------------------------------------------------------------------

    def _handle_completion(
        self,
        params: dict[str, Any],
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
    ) -> str | Any:
        try:
            response: ChatCompletion = self.client.chat.completions.create(**params)
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
            error_msg = f"Failed to connect to MiniMax API: {e}"
            logging.error(error_msg)
            self._emit_call_failed_event(
                error=error_msg, from_task=from_task, from_agent=from_agent
            )
            raise ConnectionError(error_msg) from e
        except Exception as e:
            if is_context_length_exceeded(e):
                logging.error(f"Context window exceeded: {e}")
                raise LLMContextLengthExceededError(str(e)) from e
            error_msg = f"MiniMax API call failed: {e!s}"
            logging.error(error_msg)
            self._emit_call_failed_event(
                error=error_msg, from_task=from_task, from_agent=from_agent
            )
            raise

        return content

    # ------------------------------------------------------------------
    # Streaming completion
    # ------------------------------------------------------------------

    def _handle_streaming(
        self,
        params: dict[str, Any],
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
    ) -> str | list[dict[str, Any]]:
        full_response = ""
        tool_calls: dict[int, dict[str, Any]] = {}

        completion_stream: Stream[ChatCompletionChunk] = (
            self.client.chat.completions.create(**params)
        )

        usage_data: dict[str, int] = {"total_tokens": 0}

        for chunk in completion_stream:
            response_id = chunk.id if hasattr(chunk, "id") else None

            if hasattr(chunk, "usage") and chunk.usage:
                usage_data = self._extract_token_usage(chunk)
                continue

            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            delta: ChoiceDelta = choice.delta

            if delta.content:
                full_response += delta.content
                self._emit_stream_chunk_event(
                    chunk=delta.content,
                    from_task=from_task,
                    from_agent=from_agent,
                    response_id=response_id,
                )

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index if tc.index is not None else 0
                    if idx not in tool_calls:
                        tool_calls[idx] = {
                            "id": tc.id,
                            "name": "",
                            "arguments": "",
                            "index": idx,
                        }
                    elif tc.id and not tool_calls[idx]["id"]:
                        tool_calls[idx]["id"] = tc.id
                    if tc.function and tc.function.name:
                        tool_calls[idx]["name"] = tc.function.name
                    if tc.function and tc.function.arguments:
                        tool_calls[idx]["arguments"] += tc.function.arguments

                    self._emit_stream_chunk_event(
                        chunk=(
                            tc.function.arguments
                            if tc.function and tc.function.arguments
                            else ""
                        ),
                        from_task=from_task,
                        from_agent=from_agent,
                        tool_call={
                            "id": tool_calls[idx]["id"],
                            "function": {
                                "name": tool_calls[idx]["name"],
                                "arguments": tool_calls[idx]["arguments"],
                            },
                        },
                        response_id=response_id,
                    )

        return self._finalize_streaming(
            full_response, tool_calls, usage_data, params,
            available_functions, from_task, from_agent,
        )

    def _finalize_streaming(
        self,
        full_response: str,
        tool_calls: dict[int, dict[str, Any]],
        usage_data: dict[str, int],
        params: dict[str, Any],
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
    ) -> str | list[dict[str, Any]]:
        self._track_token_usage_internal(usage_data)

        if tool_calls and not available_functions:
            tool_calls_list = [
                {
                    "id": d["id"],
                    "type": "function",
                    "function": {"name": d["name"], "arguments": d["arguments"]},
                    "index": d["index"],
                }
                for d in tool_calls.values()
            ]
            self._emit_call_completed_event(
                response=tool_calls_list,
                call_type=LLMCallType.TOOL_CALL,
                from_task=from_task,
                from_agent=from_agent,
                messages=params["messages"],
            )
            return tool_calls_list

        if tool_calls and available_functions:
            for d in tool_calls.values():
                fn_name = d["name"]
                args_str = d["arguments"]
                if not fn_name or not args_str:
                    continue
                if fn_name not in available_functions:
                    logging.warning(f"Function '{fn_name}' not found")
                    continue
                try:
                    fn_args = json.loads(args_str)
                except json.JSONDecodeError as e:
                    logging.error(f"Failed to parse streamed tool arguments: {e}")
                    continue
                result = self._handle_tool_execution(
                    function_name=fn_name,
                    function_args=fn_args,
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

    # ------------------------------------------------------------------
    # Async non-streaming
    # ------------------------------------------------------------------

    async def _ahandle_completion(
        self,
        params: dict[str, Any],
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
    ) -> str | Any:
        try:
            response: ChatCompletion = (
                await self.async_client.chat.completions.create(**params)
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

        except Exception as e:
            if is_context_length_exceeded(e):
                raise LLMContextLengthExceededError(str(e)) from e
            raise

        return content

    # ------------------------------------------------------------------
    # Async streaming
    # ------------------------------------------------------------------

    async def _ahandle_streaming(
        self,
        params: dict[str, Any],
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
    ) -> str | list[dict[str, Any]]:
        full_response = ""
        tool_calls: dict[int, dict[str, Any]] = {}
        usage_data: dict[str, int] = {"total_tokens": 0}

        stream = await self.async_client.chat.completions.create(**params)

        async for chunk in stream:
            response_id = chunk.id if hasattr(chunk, "id") else None

            if hasattr(chunk, "usage") and chunk.usage:
                usage_data = self._extract_token_usage(chunk)
                continue

            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            delta: ChoiceDelta = choice.delta

            if delta.content:
                full_response += delta.content
                self._emit_stream_chunk_event(
                    chunk=delta.content,
                    from_task=from_task,
                    from_agent=from_agent,
                    response_id=response_id,
                )

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index if tc.index is not None else 0
                    if idx not in tool_calls:
                        tool_calls[idx] = {
                            "id": tc.id,
                            "name": "",
                            "arguments": "",
                            "index": idx,
                        }
                    elif tc.id and not tool_calls[idx]["id"]:
                        tool_calls[idx]["id"] = tc.id
                    if tc.function and tc.function.name:
                        tool_calls[idx]["name"] = tc.function.name
                    if tc.function and tc.function.arguments:
                        tool_calls[idx]["arguments"] += tc.function.arguments

        return self._finalize_streaming(
            full_response, tool_calls, usage_data, params,
            available_functions, from_task, from_agent,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_token_usage(
        response: ChatCompletion | ChatCompletionChunk,
    ) -> dict[str, Any]:
        if hasattr(response, "usage") and response.usage:
            usage = response.usage
            result: dict[str, Any] = {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
            }
            prompt_details = getattr(usage, "prompt_tokens_details", None)
            if prompt_details:
                result["cached_prompt_tokens"] = (
                    getattr(prompt_details, "cached_tokens", 0) or 0
                )
            return result
        return {"total_tokens": 0}

    def supports_stop_words(self) -> bool:
        return self._supports_stop_words_implementation()

    def get_context_window_size(self) -> int:
        from crewai.llm import CONTEXT_WINDOW_USAGE_RATIO, LLM_CONTEXT_WINDOW_SIZES

        size = LLM_CONTEXT_WINDOW_SIZES.get(self.model, MINIMAX_CONTEXT_WINDOW)
        return int(size * CONTEXT_WINDOW_USAGE_RATIO)

    def supports_multimodal(self) -> bool:
        return False
