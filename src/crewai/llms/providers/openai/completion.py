import json
import logging
import os
from collections.abc import Iterator
from typing import Any

from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from pydantic import BaseModel

from crewai.events.types.llm_events import LLMCallType
from crewai.llms.base_llm import BaseLLM


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
        temperature: float | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        max_tokens: int | None = None,
        max_completion_tokens: int | None = None,
        seed: int | None = None,
        stop: str | list[str] | None = None,
        stream: bool = False,
        response_format: dict[str, Any] | type[BaseModel] | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
        reasoning_effort: str | None = None,  # For o1 models
        **kwargs,
    ):
        """Initialize OpenAI chat completion client."""
        # Convert stop to list[str] | None for BaseLLM compatibility
        stop_list: list[str] | None = None
        if stop is not None:
            if isinstance(stop, str):
                stop_list = [stop]
            else:
                stop_list = stop

        super().__init__(model=model, temperature=temperature, stop=stop_list, **kwargs)

        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url,
            organization=organization,
            project=project,
            timeout=timeout,
            max_retries=max_retries,
        )

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
        """Prepare parameters for OpenAI chat completion.

        Args:
            messages: Formatted messages
            tools: Tool definitions

        Returns:
            Parameters dictionary for OpenAI API
        """
        params = {
            "model": self.model,
            "messages": messages,
            "stream": self.stream,
        }

        # Add optional parameters if set
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
        if self.stop:
            params["stop"] = self.stop
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

        return params

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

                try:
                    function_args = json.loads(call_data["arguments"])
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
                return size

        # Default context window size
        return 8192

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
