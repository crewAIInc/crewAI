from collections.abc import Iterator
import json
import logging
import os
from typing import Any

from crewai.events.types.llm_events import LLMCallType
from crewai.llms.base_llm import BaseLLM
from crewai.utilities.agent_utils import is_context_length_exceeded
from crewai.utilities.exceptions.context_window_exceeding_exception import (
    LLMContextLengthExceededError,
)
from openai import APIConnectionError, NotFoundError, OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from pydantic import BaseModel


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
        **kwargs,
    ):
        """Initialize OpenAI chat completion client."""

        if provider is None:
            provider = kwargs.pop("provider", "openai")

        # Client configuration attributes
        self.organization = organization
        self.project = project
        self.max_retries = max_retries
        self.default_headers = default_headers
        self.default_query = default_query
        self.client_params = client_params
        self.timeout = timeout
        self.base_url = base_url

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
        self.client = OpenAI(**client_config)

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
            "base_url": self.base_url,
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
        messages: str | list[dict[str, str]],
        tools: list[dict] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
    ) -> str | Any:
        """Call OpenAI chat completion API.

        Args:
            messages: Input messages for the chat completion
            tools: list of tool/function definitions
            callbacks: Callback functions (not used in native implementation)
            available_functions: Available functions for tool calling
            from_task: Task that initiated the call
            from_agent: Agent that initiated the call

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
                formatted_messages, tools
            )

            if self.stream:
                return self._handle_streaming_completion(
                    completion_params, available_functions, from_task, from_agent
                )

            return self._handle_completion(
                completion_params, available_functions, from_task, from_agent
            )

        except Exception as e:
            error_msg = f"OpenAI API call failed: {e!s}"
            logging.error(error_msg)
            self._emit_call_failed_event(
                error=error_msg, from_task=from_task, from_agent=from_agent
            )
            raise

    def _prepare_completion_params(
        self, messages: list[dict[str, str]], tools: list[dict] | None = None
    ) -> dict[str, Any]:
        """Prepare parameters for OpenAI chat completion."""
        params = {
            "model": self.model,
            "messages": messages,
            "stream": self.stream,
        }

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

        # Handle response format for structured outputs
        if self.response_format:
            if isinstance(self.response_format, type) and issubclass(
                self.response_format, BaseModel
            ):
                # Convert Pydantic model to OpenAI response format
                params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": self.response_format.__name__,
                        "schema": self.response_format.model_json_schema(),
                    },
                }
            else:
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
            "timeout",
        }

        return {k: v for k, v in params.items() if k not in crewai_specific_params}

    def _convert_tools_for_interference(self, tools: list[dict]) -> list[dict]:
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
    ) -> str | Any:
        """Handle non-streaming chat completion."""
        try:
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
    ) -> str:
        """Handle streaming chat completion."""
        full_response = ""
        tool_calls = {}

        # Make streaming API call
        stream: Iterator[ChatCompletionChunk] = self.client.chat.completions.create(
            **params
        )

        for chunk in stream:
            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            delta: ChoiceDelta = choice.delta

            # Handle content streaming
            if delta.content:
                full_response += delta.content
                self._emit_stream_chunk_event(
                    chunk=delta.content,
                    from_task=from_task,
                    from_agent=from_agent,
                )

            # Handle tool call streaming
            if delta.tool_calls:
                for tool_call in delta.tool_calls:
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

    def _extract_openai_token_usage(self, response: ChatCompletion) -> dict[str, Any]:
        """Extract token usage from OpenAI ChatCompletion response."""
        if hasattr(response, "usage") and response.usage:
            usage = response.usage
            return {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
            }
        return {"total_tokens": 0}

    def _format_messages(
        self, messages: str | list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """Format messages for OpenAI API."""
        # Use base class formatting first
        base_formatted = super()._format_messages(messages)

        # Apply OpenAI-specific formatting
        formatted_messages = []

        for message in base_formatted:
            if self.is_o1_model and message.get("role") == "system":
                formatted_messages.append(
                    {"role": "user", "content": f"System: {message['content']}"}
                )
            else:
                formatted_messages.append(message)

        return formatted_messages
