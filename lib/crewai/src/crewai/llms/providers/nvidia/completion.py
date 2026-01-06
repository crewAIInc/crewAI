from __future__ import annotations

from collections.abc import AsyncIterator
import json
import logging
import os
import re
import time
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

# Cache TTL for model metadata (1 hour)
METADATA_CACHE_TTL = 3600

# Valid model name pattern: alphanumeric, hyphens, underscores, slashes, dots
# Examples: meta/llama-3.1-70b-instruct, nvidia/nemo-retriever-embedding-v1
MODEL_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_.\-/]+$")


class NvidiaCompletion(BaseLLM):
    """NVIDIA native completion implementation.

    This class provides direct integration with NVIDIA using the OpenAI-compatible API,
    offering native structured outputs, function calling, and streaming support.

    NVIDIA uses the OpenAI Python SDK since their API is OpenAI-compatible.
    Default base URL: https://integrate.api.nvidia.com/v1
    """

    def __init__(
        self,
        model: str = "meta/llama-3.1-70b-instruct",
        api_key: str | None = None,
        base_url: str | None = None,
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
        seed: int | None = None,
        stream: bool = False,
        response_format: dict[str, Any] | type[BaseModel] | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
        provider: str | None = None,
        interceptor: BaseInterceptor[httpx.Request, httpx.Response] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize NVIDIA chat completion client.

        Args:
            model: NVIDIA model name (e.g., 'meta/llama-3.1-70b-instruct')
            api_key: NVIDIA API key (defaults to NVIDIA_API_KEY env var)
            base_url: NVIDIA base URL (defaults to https://integrate.api.nvidia.com/v1)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            temperature: Sampling temperature (0-1)
            top_p: Nucleus sampling parameter
            frequency_penalty: Frequency penalty
            presence_penalty: Presence penalty
            max_tokens: Maximum tokens in response
            seed: Random seed for reproducibility
            stream: Enable streaming responses
            response_format: Response format configuration
            logprobs: Include log probabilities
            top_logprobs: Number of top log probabilities
            provider: Provider name (defaults to 'nvidia')
            interceptor: HTTP interceptor for transport-level modifications
            **kwargs: Additional parameters
        """

        if provider is None:
            provider = kwargs.pop("provider", "nvidia")

        # Validate model name to prevent injection attacks
        if not MODEL_NAME_PATTERN.match(model):
            raise ValueError(
                f"Invalid model name: '{model}'. Model names must only contain "
                "alphanumeric characters, hyphens, underscores, slashes, and dots."
            )

        self.interceptor = interceptor
        # Client configuration attributes
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
            api_key=api_key or os.getenv("NVIDIA_API_KEY"),
            base_url=base_url,
            timeout=timeout,
            provider=provider,
            **kwargs,
        )

        # Initialize clients without requiring API key (deferred to actual API calls)
        client_config = self._get_client_params(require_key=False)
        if self.interceptor:
            transport = HTTPTransport(interceptor=self.interceptor)
            http_client = httpx.Client(transport=transport)
            client_config["http_client"] = http_client

        self.client = OpenAI(**client_config)

        async_client_config = self._get_client_params(require_key=False)
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
        self.seed = seed
        self.stream = stream
        self.response_format = response_format
        self.logprobs = logprobs
        self.top_logprobs = top_logprobs

        # Detect model capabilities
        model_lower = model.lower()
        self.is_vision_model = any(indicator in model_lower for indicator in [
            "-vl",              # Vision-Language suffix
            "deplot",           # Google chart understanding
            "fuyu",             # Adept multimodal
            "kosmos",           # Microsoft multimodal grounding
            "mistral-large-3",  # Mistral Large 3 vision support
            "multimodal",       # Explicit multimodal designation
            "nemotron-nano",    # NVIDIA Nemotron Nano series
            "neva",             # NVIDIA vision-language
            "nvclip",           # NVIDIA CLIP
            "paligemma",        # Google vision-language
            "streampetr",       # NVIDIA perception model
            "vila",             # NVIDIA Visual Language Assistant
            "vision",           # Contains 'vision' in name
            "vlm",              # Vision-Language Model designation
        ])
        self.supports_tools = self._check_tool_support(model)

        # Cache for model metadata from API with TTL
        self._model_metadata: dict[str, Any] | None = None
        self._model_metadata_timestamp: float = 0.0

    def close(self) -> None:
        """Close HTTP clients to release resources.

        This method should be called when the NvidiaCompletion instance is no longer needed
        to properly clean up HTTP connections and release resources.

        Usage:
            completion = NvidiaCompletion(model="meta/llama-3.1-70b-instruct")
            try:
                # Use completion
                result = completion.call(messages)
            finally:
                completion.close()
        """
        try:
            if hasattr(self, "client") and self.client:
                self.client.close()
        except Exception as e:
            logging.debug(f"Error closing sync client: {e}")

        try:
            if hasattr(self, "async_client") and self.async_client:
                # AsyncOpenAI client needs to be awaited, but __del__ is sync
                # So we just close the underlying HTTP client if it exists
                if hasattr(self.async_client, "_client"):
                    self.async_client._client.close()
        except Exception as e:
            logging.debug(f"Error closing async client: {e}")

    def __del__(self) -> None:
        """Destructor to ensure HTTP clients are closed."""
        self.close()

    def __enter__(self) -> Self:
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager and close clients."""
        self.close()

    def _get_client_params(self, require_key: bool = True) -> dict[str, Any]:
        """Get NVIDIA client parameters.

        Args:
            require_key: If True, raises error when API key is missing.
                        If False, returns params with None API key (for non-API operations).
        """

        if self.api_key is None:
            self.api_key = os.getenv("NVIDIA_API_KEY")
            if self.api_key is None and require_key:
                raise ValueError(
                    "NVIDIA_API_KEY is required. Get your API key from https://build.nvidia.com/"
                )

        # Default to NVIDIA's integrated API endpoint
        default_base_url = "https://integrate.api.nvidia.com/v1"

        base_params = {
            "api_key": self.api_key or "placeholder",  # Placeholder for initialization
            "base_url": self.base_url
            or self.api_base
            or os.getenv("NVIDIA_BASE_URL")
            or default_base_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "default_headers": self.default_headers,
            "default_query": self.default_query,
        }

        client_params = {k: v for k, v in base_params.items() if v is not None}

        if self.client_params:
            client_params.update(self.client_params)

        return client_params

    def _check_tool_support(self, model: str) -> bool:
        """Check if the model supports tool/function calling.

        Most modern NVIDIA models support tools. If a model doesn't support tools,
        the API will return an appropriate error which we handle gracefully.
        """
        return True  # Default to True, let API handle unsupported models

    def _fetch_model_metadata(self) -> dict[str, Any] | None:
        """Fetch model metadata from NVIDIA API with TTL-based caching.

        Queries the /v1/models endpoint to get model information including max_model_len.
        Results are cached with a TTL to prevent cache poisoning and stale data.

        Returns:
            Model metadata dict with fields like max_model_len, or None if fetch fails
        """
        current_time = time.time()

        # Check if cache is valid (exists and not expired)
        if self._model_metadata is not None:
            cache_age = current_time - self._model_metadata_timestamp
            if cache_age < METADATA_CACHE_TTL:
                return self._model_metadata
            else:
                logging.debug(
                    f"Model metadata cache expired ({cache_age:.1f}s > {METADATA_CACHE_TTL}s), refreshing..."
                )

        try:
            # Query /v1/models endpoint to get model list
            models = self.client.models.list()

            # Find our specific model in the list
            for model_obj in models.data:
                if model_obj.id == self.model:
                    # Convert to dict
                    if hasattr(model_obj, "model_dump"):
                        metadata = model_obj.model_dump()
                    else:
                        metadata = model_obj.__dict__

                    # Validate metadata structure before caching
                    if not isinstance(metadata, dict):
                        logging.warning(
                            f"Invalid metadata type for {self.model}: {type(metadata)}"
                        )
                        self._model_metadata = {}
                        self._model_metadata_timestamp = current_time
                        return None

                    # Cache validated metadata with timestamp
                    self._model_metadata = metadata
                    self._model_metadata_timestamp = current_time

                    logging.debug(
                        f"Fetched and cached metadata for {self.model}: {self._model_metadata}"
                    )
                    return self._model_metadata

            # Model not found in list - cache empty dict to avoid repeated lookups
            logging.debug(f"Model {self.model} not found in /v1/models response")
            self._model_metadata = {}
            self._model_metadata_timestamp = current_time
            return None

        except Exception as e:
            # API call failed - cache empty dict and return None
            logging.debug(f"Failed to fetch model metadata: {e}")
            self._model_metadata = {}
            self._model_metadata_timestamp = current_time
            return None

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
        """Call NVIDIA NIM chat completion API.

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
        # Validate API key before making actual API call
        if not self.api_key and not os.getenv("NVIDIA_API_KEY"):
            raise ValueError(
                "NVIDIA_API_KEY is required for API calls. Get your API key from https://build.nvidia.com/"
            )

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
            error_msg = f"NVIDIA NIM API call failed: {e!s}"
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
        """Async call to NVIDIA NIM chat completion API.

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
        # Validate API key before making actual API call
        if not self.api_key and not os.getenv("NVIDIA_API_KEY"):
            raise ValueError(
                "NVIDIA_API_KEY is required for API calls. Get your API key from https://build.nvidia.com/"
            )

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
            error_msg = f"NVIDIA NIM API call failed: {e!s}"
            logging.error(error_msg)
            self._emit_call_failed_event(
                error=error_msg, from_task=from_task, from_agent=from_agent
            )
            raise

    def _prepare_completion_params(
        self, messages: list[LLMMessage], tools: list[dict[str, BaseTool]] | None = None
    ) -> dict[str, Any]:
        """Prepare parameters for NVIDIA NIM chat completion."""
        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if self.stream:
            params["stream"] = self.stream
            # Note: stream_options not supported by all NVIDIA models

        params.update(self.additional_params)

        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.top_p is not None:
            params["top_p"] = self.top_p
        if self.frequency_penalty is not None:
            params["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty is not None:
            params["presence_penalty"] = self.presence_penalty
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        elif self._is_reasoning_model():
            # Reasoning models (DeepSeek R1, V3, etc.) request entire context window
            # when max_tokens is not specified, causing API errors. Set sensible default.
            params["max_tokens"] = 4096
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

        if tools and self.supports_tools:
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

    def _is_reasoning_model(self) -> bool:
        """Detect if the current model is a reasoning model (DeepSeek R1, V3, etc.).

        Reasoning models have special behavior where they request the entire context window
        when max_tokens is not specified, which can cause API errors.

        Returns:
            True if the model is a reasoning model, False otherwise.
        """
        model_lower = self.model.lower()
        reasoning_patterns = [
            "deepseek-r1",
            "deepseek-v3",
            "deepseek-ai/deepseek-r1",
            "deepseek-ai/deepseek-v3",
            "gpt-oss",  # OpenAI GPT-OSS models exhibit similar behavior
        ]
        return any(pattern in model_lower for pattern in reasoning_patterns)

    def _convert_tools_for_interference(
        self, tools: list[dict[str, BaseTool]]
    ) -> list[dict[str, Any]]:
        """Convert CrewAI tool format to OpenAI function calling format (NVIDIA NIM compatible)."""
        from crewai.llms.providers.utils.common import safe_tool_conversion

        nvidia_tools = []

        for tool in tools:
            name, description, parameters = safe_tool_conversion(tool, "NVIDIA NIM")

            nvidia_tool = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                },
            }

            if parameters:
                if isinstance(parameters, dict):
                    nvidia_tool["function"]["parameters"] = parameters  # type: ignore
                else:
                    nvidia_tool["function"]["parameters"] = dict(parameters)

            nvidia_tools.append(nvidia_tool)
        return nvidia_tools

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

                usage = self._extract_token_usage(parsed_response)
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

            usage = self._extract_token_usage(response)

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

            # Check reasoning_content first (for reasoning models like DeepSeek R1)
            # then fall back to regular content
            content = getattr(message, 'reasoning_content', None) or message.content or ""
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
                logging.info(f"NVIDIA NIM API usage: {usage}")

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
            error_msg = f"Failed to connect to NVIDIA NIM API: {e}"
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

            error_msg = f"NVIDIA NIM API call failed: {e!s}"
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
        tool_calls: dict[int, dict[str, Any]] = {}

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
                    usage = self._extract_token_usage(final_completion)
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
                    )

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

                usage = self._extract_token_usage(parsed_response)
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

            usage = self._extract_token_usage(response)

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

            # Check reasoning_content first (for reasoning models like DeepSeek R1)
            # then fall back to regular content
            content = getattr(message, 'reasoning_content', None) or message.content or ""
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
                logging.info(f"NVIDIA NIM API usage: {usage}")
        except NotFoundError as e:
            error_msg = f"Model {self.model} not found: {e}"
            logging.error(error_msg)
            self._emit_call_failed_event(
                error=error_msg, from_task=from_task, from_agent=from_agent
            )
            raise ValueError(error_msg) from e
        except APIConnectionError as e:
            error_msg = f"Failed to connect to NVIDIA NIM API: {e}"
            logging.error(error_msg)
            self._emit_call_failed_event(
                error=error_msg, from_task=from_task, from_agent=from_agent
            )
            raise ConnectionError(error_msg) from e
        except Exception as e:
            if is_context_length_exceeded(e):
                logging.error(f"Context window exceeded: {e}")
                raise LLMContextLengthExceededError(str(e)) from e

            error_msg = f"NVIDIA NIM API call failed: {e!s}"
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
        tool_calls: dict[int, dict[str, Any]] = {}

        if response_model:
            completion_stream: AsyncIterator[
                ChatCompletionChunk
            ] = await self.async_client.chat.completions.create(**params)

            accumulated_content = ""
            usage_data = {"total_tokens": 0}
            async for chunk in completion_stream:
                if hasattr(chunk, "usage") and chunk.usage:
                    usage_data = self._extract_token_usage(chunk)
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
                usage_data = self._extract_token_usage(chunk)
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

        return full_response

    async def astream(
        self,
        messages: str | list[LLMMessage],
        tools: list[dict[str, BaseTool]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Task | None = None,
        from_agent: Agent | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> AsyncIterator[str]:
        """Stream responses from NVIDIA NIM chat completion API.

        This method provides an async generator that yields text chunks as they
        are received from the NVIDIA API, enabling real-time streaming responses.

        Args:
            messages: Input messages for the chat completion
            tools: list of tool/function definitions
            callbacks: Callback functions (not used in native implementation)
            available_functions: Available functions for tool calling
            from_task: Task that initiated the call
            from_agent: Agent that initiated the call
            response_model: Response model for structured output

        Yields:
            Text chunks as they are received from the API

        Raises:
            ValueError: If API key is missing or if LLM call is blocked by hook
            NotFoundError: If the model is not found
            APIConnectionError: If connection to NVIDIA API fails
            LLMContextLengthExceededError: If context window is exceeded
        """
        # Validate API key before making actual API call
        if not self.api_key and not os.getenv("NVIDIA_API_KEY"):
            raise ValueError(
                "NVIDIA_API_KEY is required for API calls. Get your API key from https://build.nvidia.com/"
            )

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

            # Force streaming mode for this method
            completion_params["stream"] = True

            # Handle structured output with response_model
            if response_model:
                completion_stream: AsyncIterator[
                    ChatCompletionChunk
                ] = await self.async_client.chat.completions.create(**completion_params)

                accumulated_content = ""
                usage_data = {"total_tokens": 0}

                async for chunk in completion_stream:
                    if hasattr(chunk, "usage") and chunk.usage:
                        usage_data = self._extract_token_usage(chunk)
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
                        yield delta.content

                self._track_token_usage_internal(usage_data)

                # Validate accumulated content against response_model
                try:
                    parsed_object = response_model.model_validate_json(accumulated_content)
                    structured_json = parsed_object.model_dump_json()

                    self._emit_call_completed_event(
                        response=structured_json,
                        call_type=LLMCallType.LLM_CALL,
                        from_task=from_task,
                        from_agent=from_agent,
                        messages=completion_params["messages"],
                    )
                except Exception as e:
                    logging.error(f"Failed to parse structured output from stream: {e}")
                    self._emit_call_completed_event(
                        response=accumulated_content,
                        call_type=LLMCallType.LLM_CALL,
                        from_task=from_task,
                        from_agent=from_agent,
                        messages=completion_params["messages"],
                    )

                return

            # Standard streaming without response_model
            stream: AsyncIterator[
                ChatCompletionChunk
            ] = await self.async_client.chat.completions.create(**completion_params)

            full_response = ""
            tool_calls: dict[int, dict[str, Any]] = {}
            usage_data = {"total_tokens": 0}

            async for chunk in stream:
                if hasattr(chunk, "usage") and chunk.usage:
                    usage_data = self._extract_token_usage(chunk)
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
                    yield chunk_delta.content

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
                        )

            self._track_token_usage_internal(usage_data)

            # Handle tool calls if present
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
                        yield str(result)
                        return

            # Apply stop words and emit completion event
            full_response = self._apply_stop_words(full_response)

            self._emit_call_completed_event(
                response=full_response,
                call_type=LLMCallType.LLM_CALL,
                from_task=from_task,
                from_agent=from_agent,
                messages=completion_params["messages"],
            )

        except NotFoundError as e:
            error_msg = f"Model {self.model} not found: {e}"
            logging.error(error_msg)
            self._emit_call_failed_event(
                error=error_msg, from_task=from_task, from_agent=from_agent
            )
            raise ValueError(error_msg) from e
        except APIConnectionError as e:
            error_msg = f"Failed to connect to NVIDIA NIM API: {e}"
            logging.error(error_msg)
            self._emit_call_failed_event(
                error=error_msg, from_task=from_task, from_agent=from_agent
            )
            raise ConnectionError(error_msg) from e
        except Exception as e:
            if is_context_length_exceeded(e):
                logging.error(f"Context window exceeded: {e}")
                raise LLMContextLengthExceededError(str(e)) from e

            error_msg = f"NVIDIA NIM API call failed: {e!s}"
            logging.error(error_msg)
            self._emit_call_failed_event(
                error=error_msg, from_task=from_task, from_agent=from_agent
            )
            raise

    def supports_function_calling(self) -> bool:
        """Check if the model supports function calling."""
        return self.supports_tools

    def supports_stop_words(self) -> bool:
        """Check if the model supports stop words."""
        return True  # NVIDIA models support stop sequences

    def get_context_window_size(self) -> int:
        """Get the context window size for the model.

        Tries to fetch max_model_len from NVIDIA API, falls back to pattern-based
        defaults if API is unavailable or doesn't return the information.
        """
        from crewai.llm import CONTEXT_WINDOW_USAGE_RATIO

        # Try to get from API first
        metadata = self._fetch_model_metadata()
        if metadata and "max_model_len" in metadata:
            max_len = metadata["max_model_len"]
            logging.debug(
                f"Using API-provided context window for {self.model}: {max_len}"
            )
            return int(max_len * CONTEXT_WINDOW_USAGE_RATIO)

        # Fallback to pattern-based defaults
        model_lower = self.model.lower()

        # Modern models with large context windows (128K)
        if any(
            indicator in model_lower
            for indicator in ["llama-3.1", "llama-3.2", "llama-3.3", "qwen3", "phi-3"]
        ):
            logging.debug(f"Using pattern-based context window (128K) for {self.model}")
            return int(128000 * CONTEXT_WINDOW_USAGE_RATIO)

        # Default for all NVIDIA models (32K - safe for most modern models)
        logging.debug(f"Using default context window (32K) for {self.model}")
        return int(32768 * CONTEXT_WINDOW_USAGE_RATIO)

    def _extract_token_usage(
        self, response: ChatCompletion | ChatCompletionChunk
    ) -> dict[str, Any]:
        """Extract token usage from NVIDIA NIM ChatCompletion or ChatCompletionChunk response."""
        if hasattr(response, "usage") and response.usage:
            usage = response.usage
            return {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
            }
        return {"total_tokens": 0}

    def _format_messages(self, messages: str | list[LLMMessage]) -> list[LLMMessage]:
        """Format messages for NVIDIA NIM API."""
        return super()._format_messages(messages)
