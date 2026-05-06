"""Cerebras native completion implementation.

Uses the official ``cerebras-cloud-sdk`` (:class:`Cerebras` / :class:`AsyncCerebras`)
directly for ``chat.completions.create``. This class subclasses
:class:`~crewai.llms.base_llm.BaseLLM` only — it does not inherit from the OpenAI
provider — while following the same chat-completion request shape the Cerebras API
expects (OpenAI-compatible HTTP surface).

Install the optional dependency: ``uv add "crewai[cerebras]"``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
import json
import logging
import os
from typing import TYPE_CHECKING, Any, Literal

import httpx
from pydantic import BaseModel, PrivateAttr, model_validator

from crewai.events.types.llm_events import LLMCallType
from crewai.llms.base_llm import BaseLLM, JsonResponseFormat, llm_call_context
from crewai.llms.hooks.base import BaseInterceptor
from crewai.llms.hooks.transport import AsyncHTTPTransport, HTTPTransport
from crewai.llms.providers.utils.common import safe_tool_conversion
from crewai.utilities.agent_utils import is_context_length_exceeded
from crewai.utilities.exceptions.context_window_exceeding_exception import (
    LLMContextLengthExceededError,
)
from crewai.utilities.pydantic_schema_utils import (
    generate_model_description,
    sanitize_tool_params_for_openai_strict,
)
from crewai.utilities.types import LLMMessage


if TYPE_CHECKING:
    from crewai.agents.agent_builder.base_agent import BaseAgent
    from crewai.task import Task
    from crewai.tools.base_tool import BaseTool

try:
    from cerebras.cloud.sdk import (
        APIConnectionError,
        AsyncCerebras,
        Cerebras,
        NotFoundError,
    )
except ImportError:
    raise ImportError(
        'Cerebras native provider not available, to install: uv add "crewai[cerebras]"'
    ) from None


CEREBRAS_BASE_URL_ENV = "CEREBRAS_BASE_URL"
CEREBRAS_API_KEY_ENV = "CEREBRAS_API_KEY"


def _extract_chat_usage(response: Any) -> dict[str, Any]:
    """Best-effort usage extraction; works for Cerebras and OpenAI-shaped responses."""
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
        completion_details = getattr(usage, "completion_tokens_details", None)
        if completion_details:
            result["reasoning_tokens"] = (
                getattr(completion_details, "reasoning_tokens", 0) or 0
            )
        return result
    return {"total_tokens": 0}


def _first_tool_call_function(message: Any) -> tuple[str, dict[str, Any]] | None:
    """Resolve the first function tool call using duck typing (Cerebras SDK types)."""
    tool_calls = getattr(message, "tool_calls", None) or []
    if not tool_calls:
        return None
    tc = tool_calls[0]
    fn = getattr(tc, "function", None)
    if fn is None:
        return None
    name = getattr(fn, "name", None) or ""
    if not name:
        return None
    raw_args = getattr(fn, "arguments", None) or "{}"
    try:
        args: dict[str, Any] = (
            json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        args = {}
    return name, args


class CerebrasCompletion(BaseLLM):
    """Cerebras chat completions via ``cerebras-cloud-sdk``.

    Reads ``CEREBRAS_API_KEY`` and optional ``CEREBRAS_BASE_URL``. Only the chat
    completions API is supported (``api`` is always ``\"completions\"``).
    """

    llm_type: Literal["cerebras"] = "cerebras"

    timeout: float | None = None
    max_retries: int = 2
    default_headers: dict[str, str] | None = None
    default_query: dict[str, Any] | None = None
    client_params: dict[str, Any] | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    seed: int | None = None
    stream: bool = False
    response_format: JsonResponseFormat | type[BaseModel] | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    reasoning_effort: Literal["none", "low", "medium", "high"] | None = None
    interceptor: BaseInterceptor[httpx.Request, httpx.Response] | None = None
    api: Literal["completions"] = "completions"
    api_base: str | None = None

    service_tier: Literal["priority", "default", "auto", "flex"] | None = None
    prompt_cache_key: str | None = None
    clear_thinking: bool | None = None

    _client: Any = PrivateAttr(default=None)
    _async_client: Any = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def _normalize_cerebras_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data["provider"] = "cerebras"
        data["api_key"] = data.get("api_key") or os.getenv(CEREBRAS_API_KEY_ENV)
        if not data.get("base_url") and not data.get("api_base"):
            env_base_url = os.getenv(CEREBRAS_BASE_URL_ENV)
            if env_base_url:
                data["base_url"] = env_base_url
        if "api_base" not in data:
            data["api_base"] = None
        data["api"] = "completions"
        return data

    @model_validator(mode="after")
    def _init_clients(self) -> CerebrasCompletion:
        try:
            self._client = self._build_sync_client()
            self._async_client = self._build_async_client()
        except ValueError:
            pass
        return self

    def _get_client_params(self) -> dict[str, Any]:
        if self.api_key is None:
            self.api_key = os.getenv(CEREBRAS_API_KEY_ENV)
            if self.api_key is None:
                raise ValueError(
                    "CEREBRAS_API_KEY is required. Set it in the environment "
                    "or pass api_key= when constructing the LLM."
                )

        base_url = self.base_url or self.api_base or os.getenv(CEREBRAS_BASE_URL_ENV)

        base_params: dict[str, Any] = {
            "api_key": self.api_key,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "default_headers": self.default_headers,
            "default_query": self.default_query,
        }
        if base_url:
            base_params["base_url"] = base_url

        client_params = {k: v for k, v in base_params.items() if v is not None}

        if self.client_params:
            client_params.update(self.client_params)

        return client_params

    def _build_sync_client(self) -> Any:
        client_config = self._get_client_params()
        if self.interceptor:
            transport = HTTPTransport(interceptor=self.interceptor)
            client_config["http_client"] = httpx.Client(transport=transport)
        return Cerebras(**client_config)

    def _build_async_client(self) -> Any:
        client_config = self._get_client_params()
        if self.interceptor:
            transport = AsyncHTTPTransport(interceptor=self.interceptor)
            client_config["http_client"] = httpx.AsyncClient(transport=transport)
        return AsyncCerebras(**client_config)

    def _get_sync_client(self) -> Any:
        if self._client is None:
            self._client = self._build_sync_client()
        return self._client

    def _get_async_client(self) -> Any:
        if self._async_client is None:
            self._async_client = self._build_async_client()
        return self._async_client

    def to_config_dict(self) -> dict[str, Any]:
        config = super().to_config_dict()
        if self.timeout is not None:
            config["timeout"] = self.timeout
        if self.max_retries != 2:
            config["max_retries"] = self.max_retries
        if self.top_p is not None:
            config["top_p"] = self.top_p
        if self.frequency_penalty is not None:
            config["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty is not None:
            config["presence_penalty"] = self.presence_penalty
        if self.max_tokens is not None:
            config["max_tokens"] = self.max_tokens
        if self.max_completion_tokens is not None:
            config["max_completion_tokens"] = self.max_completion_tokens
        if self.seed is not None:
            config["seed"] = self.seed
        if self.reasoning_effort is not None:
            config["reasoning_effort"] = self.reasoning_effort
        if self.stream:
            config["stream"] = True
        if self.service_tier is not None:
            config["service_tier"] = self.service_tier
        if self.prompt_cache_key is not None:
            config["prompt_cache_key"] = self.prompt_cache_key
        if self.clear_thinking is not None:
            config["clear_thinking"] = self.clear_thinking
        return config

    def _convert_tools_for_interference(
        self, tools: list[dict[str, BaseTool]]
    ) -> list[dict[str, Any]]:
        openai_tools: list[dict[str, Any]] = []

        for tool in tools:
            name, description, parameters = safe_tool_conversion(tool, "OpenAI")

            openai_tool: dict[str, Any] = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "strict": True,
                },
            }

            if parameters:
                params_dict = (
                    parameters if isinstance(parameters, dict) else dict(parameters)
                )
                openai_tool["function"]["parameters"] = (
                    sanitize_tool_params_for_openai_strict(params_dict)
                )

            openai_tools.append(openai_tool)
        return openai_tools

    def _prepare_completion_params(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, BaseTool]] | None = None,
    ) -> dict[str, Any]:
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

        if self.reasoning_effort is not None:
            params["reasoning_effort"] = self.reasoning_effort
        if self.service_tier is not None:
            params["service_tier"] = self.service_tier
        if self.prompt_cache_key is not None:
            params["prompt_cache_key"] = self.prompt_cache_key
        if self.clear_thinking is not None:
            params["clear_thinking"] = self.clear_thinking

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

    def _finalize_streaming_response(
        self,
        full_response: str,
        tool_calls: dict[int, dict[str, Any]],
        usage_data: dict[str, Any] | None,
        params: dict[str, Any],
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
    ) -> str | list[dict[str, Any]]:
        if usage_data:
            self._track_token_usage_internal(usage_data)

        if tool_calls and not available_functions:
            tool_calls_list = [
                {
                    "id": call_data["id"],
                    "type": "function",
                    "function": {
                        "name": call_data["name"],
                        "arguments": call_data["arguments"],
                    },
                    "index": call_data["index"],
                }
                for call_data in tool_calls.values()
            ]
            self._emit_call_completed_event(
                response=tool_calls_list,
                call_type=LLMCallType.TOOL_CALL,
                from_task=from_task,
                from_agent=from_agent,
                messages=params["messages"],
                usage=usage_data,
            )
            return tool_calls_list

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
            usage=usage_data,
        )

        return full_response

    def _handle_completion(
        self,
        params: dict[str, Any],
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        try:
            if response_model:
                structured_params = {
                    k: v for k, v in params.items() if k != "response_format"
                }
                structured_params["response_format"] = generate_model_description(
                    response_model
                )
                parsed_response = self._get_sync_client().chat.completions.create(
                    **structured_params
                )
                usage = _extract_chat_usage(parsed_response)
                self._track_token_usage_internal(usage)
                message = parsed_response.choices[0].message
                content = getattr(message, "content", None) or ""
                structured = self._validate_structured_output(content, response_model)
                self._emit_call_completed_event(
                    response=structured.model_dump_json()
                    if isinstance(structured, BaseModel)
                    else structured,
                    call_type=LLMCallType.LLM_CALL,
                    from_task=from_task,
                    from_agent=from_agent,
                    messages=params["messages"],
                    usage=usage,
                )
                return structured

            response = self._get_sync_client().chat.completions.create(**params)

            usage = _extract_chat_usage(response)
            self._track_token_usage_internal(usage)

            message = response.choices[0].message

            if message.tool_calls and not available_functions:
                self._emit_call_completed_event(
                    response=list(message.tool_calls),
                    call_type=LLMCallType.TOOL_CALL,
                    from_task=from_task,
                    from_agent=from_agent,
                    messages=params["messages"],
                    usage=usage,
                )
                return list(message.tool_calls)

            if message.tool_calls and available_functions:
                parsed_tool = _first_tool_call_function(message)
                if not parsed_tool:
                    return message.content or ""
                function_name, function_args = parsed_tool

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
                        usage=usage,
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
                usage=usage,
            )

            if usage.get("total_tokens", 0) > 0:
                logging.info(f"Cerebras API usage: {usage}")

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
            error_msg = f"Failed to connect to Cerebras API: {e}"
            logging.error(error_msg)
            self._emit_call_failed_event(
                error=error_msg, from_task=from_task, from_agent=from_agent
            )
            raise ConnectionError(error_msg) from e
        except Exception as e:
            if is_context_length_exceeded(e):
                logging.error(f"Context window exceeded: {e}")
                raise LLMContextLengthExceededError(str(e)) from e

            error_msg = f"Cerebras API call failed: {e!s}"
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
    ) -> str | list[dict[str, Any]] | BaseModel:
        full_response = ""
        tool_calls: dict[int, dict[str, Any]] = {}
        usage_data: dict[str, Any] | None = None

        if response_model:
            completion_stream = self._get_sync_client().chat.completions.create(
                **params
            )

            accumulated_content = ""
            for chunk in completion_stream:
                response_id_stream = chunk.id if hasattr(chunk, "id") else None

                if hasattr(chunk, "usage") and chunk.usage:
                    usage_data = _extract_chat_usage(chunk)
                    continue

                if not chunk.choices:
                    continue

                choice = chunk.choices[0]
                delta = choice.delta

                if delta.content:
                    accumulated_content += delta.content
                    self._emit_stream_chunk_event(
                        chunk=delta.content,
                        from_task=from_task,
                        from_agent=from_agent,
                        response_id=response_id_stream,
                    )

            if usage_data:
                self._track_token_usage_internal(usage_data)

            try:
                parsed_object = response_model.model_validate_json(accumulated_content)

                self._emit_call_completed_event(
                    response=parsed_object.model_dump_json(),
                    call_type=LLMCallType.LLM_CALL,
                    from_task=from_task,
                    from_agent=from_agent,
                    messages=params["messages"],
                    usage=usage_data,
                )

                return parsed_object
            except Exception as e:
                logging.error(f"Failed to parse structured output from stream: {e}")
                self._emit_call_completed_event(
                    response=accumulated_content,
                    call_type=LLMCallType.LLM_CALL,
                    from_task=from_task,
                    from_agent=from_agent,
                    messages=params["messages"],
                    usage=usage_data,
                )
                return accumulated_content

        completion_stream = self._get_sync_client().chat.completions.create(**params)

        for completion_chunk in completion_stream:
            response_id_stream = (
                completion_chunk.id if hasattr(completion_chunk, "id") else None
            )

            if hasattr(completion_chunk, "usage") and completion_chunk.usage:
                usage_data = _extract_chat_usage(completion_chunk)
                continue

            if not completion_chunk.choices:
                continue

            choice = completion_chunk.choices[0]
            chunk_delta = choice.delta

            if chunk_delta.content:
                full_response += chunk_delta.content
                self._emit_stream_chunk_event(
                    chunk=chunk_delta.content,
                    from_task=from_task,
                    from_agent=from_agent,
                    response_id=response_id_stream,
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
                        response_id=response_id_stream,
                    )

        result = self._finalize_streaming_response(
            full_response=full_response,
            tool_calls=tool_calls,
            usage_data=usage_data,
            params=params,
            available_functions=available_functions,
            from_task=from_task,
            from_agent=from_agent,
        )
        if isinstance(result, str):
            return self._invoke_after_llm_call_hooks(
                params["messages"], result, from_agent
            )
        return result

    async def _ahandle_completion(
        self,
        params: dict[str, Any],
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        try:
            if response_model:
                structured_params = {
                    k: v for k, v in params.items() if k != "response_format"
                }
                structured_params["response_format"] = generate_model_description(
                    response_model
                )
                parsed_response = (
                    await self._get_async_client().chat.completions.create(
                        **structured_params
                    )
                )
                usage = _extract_chat_usage(parsed_response)
                self._track_token_usage_internal(usage)
                message = parsed_response.choices[0].message
                content = getattr(message, "content", None) or ""
                structured = self._validate_structured_output(content, response_model)
                self._emit_call_completed_event(
                    response=structured.model_dump_json()
                    if isinstance(structured, BaseModel)
                    else structured,
                    call_type=LLMCallType.LLM_CALL,
                    from_task=from_task,
                    from_agent=from_agent,
                    messages=params["messages"],
                    usage=usage,
                )
                return structured

            response = await self._get_async_client().chat.completions.create(**params)

            usage = _extract_chat_usage(response)
            self._track_token_usage_internal(usage)

            message = response.choices[0].message

            if message.tool_calls and not available_functions:
                self._emit_call_completed_event(
                    response=list(message.tool_calls),
                    call_type=LLMCallType.TOOL_CALL,
                    from_task=from_task,
                    from_agent=from_agent,
                    messages=params["messages"],
                    usage=usage,
                )
                return list(message.tool_calls)

            if message.tool_calls and available_functions:
                parsed_tool = _first_tool_call_function(message)
                if not parsed_tool:
                    return message.content or ""
                function_name, function_args = parsed_tool

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
                        usage=usage,
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
                usage=usage,
            )

            if usage.get("total_tokens", 0) > 0:
                logging.info(f"Cerebras API usage: {usage}")
        except NotFoundError as e:
            error_msg = f"Model {self.model} not found: {e}"
            logging.error(error_msg)
            self._emit_call_failed_event(
                error=error_msg, from_task=from_task, from_agent=from_agent
            )
            raise ValueError(error_msg) from e
        except APIConnectionError as e:
            error_msg = f"Failed to connect to Cerebras API: {e}"
            logging.error(error_msg)
            self._emit_call_failed_event(
                error=error_msg, from_task=from_task, from_agent=from_agent
            )
            raise ConnectionError(error_msg) from e
        except Exception as e:
            if is_context_length_exceeded(e):
                logging.error(f"Context window exceeded: {e}")
                raise LLMContextLengthExceededError(str(e)) from e

            error_msg = f"Cerebras API call failed: {e!s}"
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
    ) -> str | list[dict[str, Any]] | BaseModel:
        full_response = ""
        tool_calls: dict[int, dict[str, Any]] = {}
        usage_data: dict[str, Any] | None = None

        if response_model:
            completion_stream: AsyncIterator[
                Any
            ] = await self._get_async_client().chat.completions.create(**params)

            accumulated_content = ""
            async for chunk in completion_stream:
                response_id_stream = chunk.id if hasattr(chunk, "id") else None

                if hasattr(chunk, "usage") and chunk.usage:
                    usage_data = _extract_chat_usage(chunk)
                    continue

                if not chunk.choices:
                    continue

                choice = chunk.choices[0]
                delta = choice.delta

                if delta.content:
                    accumulated_content += delta.content
                    self._emit_stream_chunk_event(
                        chunk=delta.content,
                        from_task=from_task,
                        from_agent=from_agent,
                        response_id=response_id_stream,
                    )

            if usage_data:
                self._track_token_usage_internal(usage_data)

            try:
                parsed_object = response_model.model_validate_json(accumulated_content)

                self._emit_call_completed_event(
                    response=parsed_object.model_dump_json(),
                    call_type=LLMCallType.LLM_CALL,
                    from_task=from_task,
                    from_agent=from_agent,
                    messages=params["messages"],
                    usage=usage_data,
                )

                return parsed_object
            except Exception as e:
                logging.error(f"Failed to parse structured output from stream: {e}")
                self._emit_call_completed_event(
                    response=accumulated_content,
                    call_type=LLMCallType.LLM_CALL,
                    from_task=from_task,
                    from_agent=from_agent,
                    messages=params["messages"],
                    usage=usage_data,
                )
                return accumulated_content

        stream: AsyncIterator[
            Any
        ] = await self._get_async_client().chat.completions.create(**params)

        async for chunk in stream:
            response_id_stream = chunk.id if hasattr(chunk, "id") else None

            if hasattr(chunk, "usage") and chunk.usage:
                usage_data = _extract_chat_usage(chunk)
                continue

            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            chunk_delta = choice.delta

            if chunk_delta.content:
                full_response += chunk_delta.content
                self._emit_stream_chunk_event(
                    chunk=chunk_delta.content,
                    from_task=from_task,
                    from_agent=from_agent,
                    response_id=response_id_stream,
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
                        response_id=response_id_stream,
                    )

        result = self._finalize_streaming_response(
            full_response=full_response,
            tool_calls=tool_calls,
            usage_data=usage_data,
            params=params,
            available_functions=available_functions,
            from_task=from_task,
            from_agent=from_agent,
        )
        if isinstance(result, str):
            return self._invoke_after_llm_call_hooks(
                params["messages"], result, from_agent
            )
        return result

    def _call_completions(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, BaseTool]] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Task | None = None,
        from_agent: BaseAgent | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        completion_params = self._prepare_completion_params(
            messages=messages, tools=tools
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

    async def _acall_completions(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, BaseTool]] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Task | None = None,
        from_agent: BaseAgent | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        completion_params = self._prepare_completion_params(
            messages=messages, tools=tools
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

    def call(
        self,
        messages: str | list[LLMMessage],
        tools: list[dict[str, BaseTool]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Task | None = None,
        from_agent: BaseAgent | None = None,
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

                return self._call_completions(
                    messages=formatted_messages,
                    tools=tools,
                    available_functions=available_functions,
                    from_task=from_task,
                    from_agent=from_agent,
                    response_model=response_model,
                )

            except Exception as e:
                error_msg = f"Cerebras API call failed: {e!s}"
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
        from_agent: BaseAgent | None = None,
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

                return await self._acall_completions(
                    messages=formatted_messages,
                    tools=tools,
                    available_functions=available_functions,
                    from_task=from_task,
                    from_agent=from_agent,
                    response_model=response_model,
                )

            except Exception as e:
                error_msg = f"Cerebras API call failed: {e!s}"
                logging.error(error_msg)
                self._emit_call_failed_event(
                    error=error_msg, from_task=from_task, from_agent=from_agent
                )
                raise

    def supports_function_calling(self) -> bool:
        return True

    def get_context_window_size(self) -> int:
        from crewai.llm import CONTEXT_WINDOW_USAGE_RATIO

        return int(8192 * CONTEXT_WINDOW_USAGE_RATIO)
