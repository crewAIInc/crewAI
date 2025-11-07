from __future__ import annotations

from collections.abc import Iterator
import json
import logging
import os
from typing import TYPE_CHECKING, Any, Final

import httpx
from openai import APIConnectionError, NotFoundError, OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    model_validator,
)
from typing_extensions import Self

from crewai.events.types.llm_events import LLMCallType
from crewai.llm import CONTEXT_WINDOW_USAGE_RATIO, LLM_CONTEXT_WINDOW_SIZES
from crewai.llms.base_llm import BaseLLM
from crewai.llms.hooks.transport import HTTPTransport
from crewai.utilities.agent_utils import is_context_length_exceeded
from crewai.utilities.exceptions.context_window_exceeding_exception import (
    LLMContextLengthExceededError,
)
from crewai.utilities.types import LLMMessage


if TYPE_CHECKING:
    from crewai.agent.core import Agent
    from crewai.task import Task
    from crewai.tools.base_tool import BaseTool


OPENAI_CONTEXT_WINDOWS: dict[str, int] = {
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

MIN_CONTEXT_WINDOW: Final[int] = 1024
MAX_CONTEXT_WINDOW: Final[int] = 2097152


class OpenAICompletion(BaseLLM):
    """OpenAI native completion implementation.

    This class provides direct integration with the OpenAI Python SDK,
    offering native structured outputs, function calling, and streaming support.
    """

    model: str = Field(
        default="gpt-4o",
        description="OpenAI model name (e.g., 'gpt-4o')",
    )
    organization: str | None = Field(
        default=None,
        description="Name of the OpenAI organization",
    )
    project: str | None = Field(
        default=None,
        description="Name of the OpenAI project",
    )
    api_base: str | None = Field(
        default=os.getenv("OPENAI_BASE_URL"),
        description="Base URL for OpenAI API",
    )
    default_headers: dict[str, str] | None = Field(
        default=None,
        description="Default headers for OpenAI API requests",
    )
    default_query: dict[str, Any] | None = Field(
        default=None,
        description="Default query parameters for OpenAI API requests",
    )
    top_p: float | None = Field(
        default=None,
        description="Top-p sampling parameter",
    )
    frequency_penalty: float | None = Field(
        default=None,
        description="Frequency penalty parameter",
    )
    presence_penalty: float | None = Field(
        default=None,
        description="Presence penalty parameter",
    )
    max_completion_tokens: int | None = Field(
        default=None,
        description="Maximum tokens for completion",
    )
    seed: int | None = Field(
        default=None,
        description="Random seed for reproducibility",
    )
    response_format: dict[str, Any] | type[BaseModel] | None = Field(
        default=None,
        description="Response format for structured output",
    )
    logprobs: bool | None = Field(
        default=None,
        description="Whether to include log probabilities",
    )
    top_logprobs: int | None = Field(
        default=None,
        description="Number of top log probabilities to return",
    )
    reasoning_effort: str | None = Field(
        default=None,
        description="Reasoning effort level for o1 models",
    )
    supports_function_calling: bool = Field(
        default=True,
        description="Whether the model supports function calling",
    )
    is_o1_model: bool = Field(
        default=False,
        description="Whether the model is an o1 model",
    )
    is_gpt4_model: bool = Field(
        default=False,
        description="Whether the model is a GPT-4 model",
    )
    _client: OpenAI = PrivateAttr(
        default_factory=OpenAI,
    )

    @model_validator(mode="after")
    def initialize_client(self) -> Self:
        """Initialize the Anthropic client after Pydantic validation.

        This runs after all field validation is complete, ensuring that:
        - All BaseLLM fields are set (model, temperature, stop_sequences, etc.)
        - Field validators have run (stop_sequences is normalized to set[str])
        - API key and other configuration is ready
        """
        if self.api_key is None:
            self.api_key = os.getenv("OPENAI_API_KEY")
            if self.api_key is None:
                raise ValueError("OPENAI_API_KEY is required")

        self.is_o1_model = "o1" in self.model.lower()
        self.supports_function_calling = not self.is_o1_model
        self.is_gpt4_model = "gpt-4" in self.model.lower()
        self.supports_stop_words = not self.is_o1_model

        params = self.model_dump(
            include={
                "api_key",
                "organization",
                "project",
                "base_url",
                "timeout",
                "max_retries",
                "default_headers",
                "default_query",
            },
            exclude_none=True,
        )

        if self.interceptor:
            transport = HTTPTransport(interceptor=self.interceptor)
            http_client = httpx.Client(transport=transport)
            params["http_client"] = http_client

        if self.client_params:
            params.update(self.client_params)

        self._client = OpenAI(**params)
        return self

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

    def _prepare_completion_params(
        self, messages: list[LLMMessage], tools: list[dict[str, BaseTool]] | None = None
    ) -> dict[str, Any]:
        """Prepare parameters for OpenAI chat completion."""
        params = self.model_dump(
            include={
                "model",
                "stream",
                "temperature",
                "top_p",
                "frequency_penalty",
                "presence_penalty",
                "max_completion_tokens",
                "max_tokens",
                "seed",
                "logprobs",
                "top_logprobs",
                "reasoning_effort",
            },
            exclude_none=True,
        )
        params["messages"] = messages
        params.update(self.additional_params)

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
        from_task: Task | None = None,
        from_agent: Agent | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        """Handle non-streaming chat completion."""
        try:
            if response_model:
                parsed_response = self._client.beta.chat.completions.parse(
                    **params,
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

            response: ChatCompletion = self._client.chat.completions.create(**params)

            usage = self._extract_openai_token_usage(response)

            self._track_token_usage_internal(usage)

            choice: Choice = response.choices[0]
            message = choice.message

            if message.tool_calls and available_functions:
                tool_call = message.tool_calls[0]
                function_name = tool_call.function.name  # type: ignore[union-attr]

                try:
                    function_args = json.loads(tool_call.function.arguments)  # type: ignore[union-attr]
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
        from_task: Task | None = None,
        from_agent: Agent | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str:
        """Handle streaming chat completion."""
        full_response = ""
        tool_calls = {}

        if response_model:
            completion_stream: Iterator[ChatCompletionChunk] = (
                self._client.chat.completions.create(**params)
            )

            accumulated_content = ""
            for chunk in completion_stream:
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

        stream: Iterator[ChatCompletionChunk] = self._client.chat.completions.create(
            **params
        )

        for chunk in stream:
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

        return full_response

    def get_context_window_size(self) -> int:
        """Get the context window size for the model."""
        for key, value in LLM_CONTEXT_WINDOW_SIZES.items():
            if value < MIN_CONTEXT_WINDOW or value > MAX_CONTEXT_WINDOW:
                raise ValueError(
                    f"Context window for {key} must be between {MIN_CONTEXT_WINDOW} and {MAX_CONTEXT_WINDOW}"
                )

        # Find the best match for the model name
        for model_prefix, size in OPENAI_CONTEXT_WINDOWS.items():
            if self.model.startswith(model_prefix):
                return int(size * CONTEXT_WINDOW_USAGE_RATIO)

        # Default context window size
        return int(8192 * CONTEXT_WINDOW_USAGE_RATIO)

    @staticmethod
    def _extract_openai_token_usage(response: ChatCompletion) -> dict[str, Any]:
        """Extract token usage from OpenAI ChatCompletion response."""
        if response.usage:
            usage = response.usage
            return {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
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
