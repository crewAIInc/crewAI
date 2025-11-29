from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from crewai.utilities.agent_utils import is_context_length_exceeded
from crewai.utilities.converter import generate_model_description
from crewai.utilities.exceptions.context_window_exceeding_exception import (
    LLMContextLengthExceededError,
)
from crewai.utilities.types import LLMMessage


if TYPE_CHECKING:
    from crewai.llms.hooks.base import BaseInterceptor
    from crewai.tools.base_tool import BaseTool


try:
    from azure.ai.inference import (
        ChatCompletionsClient,
    )
    from azure.ai.inference.models import (
        ChatCompletions,
        ChatCompletionsToolCall,
        JsonSchemaFormat,
        StreamingChatCompletionsUpdate,
    )
    from azure.core.credentials import (
        AzureKeyCredential,
    )
    from azure.core.exceptions import (
        HttpResponseError,
    )

    from crewai.events.types.llm_events import LLMCallType
    from crewai.llms.base_llm import BaseLLM

except ImportError:
    raise ImportError(
        'Azure AI Inference native provider not available, to install: uv add "crewai[azure-ai-inference]"'
    ) from None


class AzureCompletion(BaseLLM):
    """Azure AI Inference native completion implementation.

    This class provides direct integration with the Azure AI Inference Python SDK,
    offering native function calling, streaming support, and proper Azure authentication.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        endpoint: str | None = None,
        api_version: str | None = None,
        timeout: float | None = None,
        max_retries: int = 2,
        temperature: float | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        stream: bool = False,
        interceptor: BaseInterceptor[Any, Any] | None = None,
        **kwargs: Any,
    ):
        """Initialize Azure AI Inference chat completion client.

        Args:
            model: Azure deployment name or model name
            api_key: Azure API key (defaults to AZURE_API_KEY env var)
            endpoint: Azure endpoint URL (defaults to AZURE_ENDPOINT env var)
            api_version: Azure API version (defaults to AZURE_API_VERSION env var)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            temperature: Sampling temperature (0-2)
            top_p: Nucleus sampling parameter
            frequency_penalty: Frequency penalty (-2 to 2)
            presence_penalty: Presence penalty (-2 to 2)
            max_tokens: Maximum tokens in response
            stop: Stop sequences
            stream: Enable streaming responses
            interceptor: HTTP interceptor (not yet supported for Azure).
            **kwargs: Additional parameters
        """
        if interceptor is not None:
            raise NotImplementedError(
                "HTTP interceptors are not yet supported for Azure AI Inference provider. "
                "Interceptors are currently supported for OpenAI and Anthropic providers only."
            )

        super().__init__(
            model=model, temperature=temperature, stop=stop or [], **kwargs
        )

        self.api_key = api_key or os.getenv("AZURE_API_KEY")
        self.endpoint = (
            endpoint
            or os.getenv("AZURE_ENDPOINT")
            or os.getenv("AZURE_OPENAI_ENDPOINT")
            or os.getenv("AZURE_API_BASE")
        )
        self.api_version = api_version or os.getenv("AZURE_API_VERSION") or "2024-06-01"
        self.timeout = timeout
        self.max_retries = max_retries

        if not self.api_key:
            raise ValueError(
                "Azure API key is required. Set AZURE_API_KEY environment variable or pass api_key parameter."
            )
        if not self.endpoint:
            raise ValueError(
                "Azure endpoint is required. Set AZURE_ENDPOINT environment variable or pass endpoint parameter."
            )

        # Validate and potentially fix Azure OpenAI endpoint URL
        self.endpoint = self._validate_and_fix_endpoint(self.endpoint, model)

        # Build client kwargs
        client_kwargs = {
            "endpoint": self.endpoint,
            "credential": AzureKeyCredential(self.api_key),
        }

        # Add api_version if specified (primarily for Azure OpenAI endpoints)
        if self.api_version:
            client_kwargs["api_version"] = self.api_version

        self.client = ChatCompletionsClient(**client_kwargs)  # type: ignore[arg-type]

        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.max_tokens = max_tokens
        self.stream = stream

        self.is_openai_model = any(
            prefix in model.lower() for prefix in ["gpt-", "o1-", "text-"]
        )

        self.is_azure_openai_endpoint = (
            "openai.azure.com" in self.endpoint
            and "/openai/deployments/" in self.endpoint
        )

    def _validate_and_fix_endpoint(self, endpoint: str, model: str) -> str:
        """Validate and fix Azure endpoint URL format.

        Azure OpenAI endpoints should be in the format:
        https://<resource-name>.openai.azure.com/openai/deployments/<deployment-name>

        Args:
            endpoint: The endpoint URL
            model: The model/deployment name

        Returns:
            Validated and potentially corrected endpoint URL
        """
        if "openai.azure.com" in endpoint and "/openai/deployments/" not in endpoint:
            endpoint = endpoint.rstrip("/")

            if not endpoint.endswith("/openai/deployments"):
                deployment_name = model.replace("azure/", "")
                endpoint = f"{endpoint}/openai/deployments/{deployment_name}"
                logging.info(f"Constructed Azure OpenAI endpoint URL: {endpoint}")

        return endpoint

    def call(
        self,
        messages: str | list[LLMMessage],
        tools: list[dict[str, BaseTool]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        """Call Azure AI Inference chat completions API.

        Args:
            messages: Input messages for the chat completion
            tools: List of tool/function definitions
            callbacks: Callback functions (not used in native implementation)
            available_functions: Available functions for tool calling
            from_task: Task that initiated the call
            from_agent: Agent that initiated the call

        Returns:
            Chat completion response or tool call result
        """
        try:
            # Emit call started event
            self._emit_call_started_event(
                messages=messages,
                tools=tools,
                callbacks=callbacks,
                available_functions=available_functions,
                from_task=from_task,
                from_agent=from_agent,
            )

            # Format messages for Azure
            formatted_messages = self._format_messages_for_azure(messages)

            # Prepare completion parameters
            completion_params = self._prepare_completion_params(
                formatted_messages, tools, response_model
            )

            # Handle streaming vs non-streaming
            if self.stream:
                return self._handle_streaming_completion(
                    completion_params,
                    available_functions,
                    from_task,
                    from_agent,
                    response_model,
                )

            return self._handle_completion(
                completion_params,
                available_functions,
                from_task,
                from_agent,
                response_model,
            )

        except HttpResponseError as e:
            if e.status_code == 401:
                error_msg = "Azure authentication failed. Check your API key."
            elif e.status_code == 404:
                error_msg = (
                    f"Azure endpoint not found. Check endpoint URL: {self.endpoint}"
                )
            elif e.status_code == 429:
                error_msg = "Azure API rate limit exceeded. Please retry later."
            else:
                error_msg = f"Azure API HTTP error: {e.status_code} - {e.message}"

            logging.error(error_msg)
            self._emit_call_failed_event(
                error=error_msg, from_task=from_task, from_agent=from_agent
            )
            raise
        except Exception as e:
            error_msg = f"Azure API call failed: {e!s}"
            logging.error(error_msg)
            self._emit_call_failed_event(
                error=error_msg, from_task=from_task, from_agent=from_agent
            )
            raise

    def _prepare_completion_params(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> dict[str, Any]:
        """Prepare parameters for Azure AI Inference chat completion.

        Args:
            messages: Formatted messages for Azure
            tools: Tool definitions
            response_model: Pydantic model for structured output

        Returns:
            Parameters dictionary for Azure API
        """
        params = {
            "messages": messages,
            "stream": self.stream,
        }

        if response_model and self.is_openai_model:
            model_description = generate_model_description(response_model)
            json_schema_info = model_description["json_schema"]
            json_schema_name = json_schema_info["name"]

            params["response_format"] = JsonSchemaFormat(
                name=json_schema_name,
                schema=json_schema_info["schema"],
                description=f"Schema for {json_schema_name}",
                strict=json_schema_info["strict"],
            )

        # Only include model parameter for non-Azure OpenAI endpoints
        # Azure OpenAI endpoints have the deployment name in the URL
        if not self.is_azure_openai_endpoint:
            params["model"] = self.model

        # Add optional parameters if set
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
        if self.stop:
            params["stop"] = self.stop

        # Handle tools/functions for Azure OpenAI models
        if tools and self.is_openai_model:
            params["tools"] = self._convert_tools_for_interference(tools)
            params["tool_choice"] = "auto"

        additional_params = self.additional_params
        additional_drop_params = additional_params.get("additional_drop_params")
        drop_params = additional_params.get("drop_params")

        if drop_params and isinstance(additional_drop_params, list):
            for drop_param in additional_drop_params:
                params.pop(drop_param, None)

        return params

    def _convert_tools_for_interference(
        self, tools: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Convert CrewAI tool format to Azure OpenAI function calling format."""

        from crewai.llms.providers.utils.common import safe_tool_conversion

        azure_tools = []

        for tool in tools:
            name, description, parameters = safe_tool_conversion(tool, "Azure")

            azure_tool = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                },
            }

            if parameters:
                if isinstance(parameters, dict):
                    azure_tool["function"]["parameters"] = parameters  # type: ignore
                else:
                    azure_tool["function"]["parameters"] = dict(parameters)

            azure_tools.append(azure_tool)

        return azure_tools

    def _format_messages_for_azure(
        self, messages: str | list[LLMMessage]
    ) -> list[LLMMessage]:
        """Format messages for Azure AI Inference API.

        Args:
            messages: Input messages

        Returns:
            List of dict objects with 'role' and 'content' keys
        """
        # Use base class formatting first
        base_formatted = super()._format_messages(messages)

        azure_messages: list[LLMMessage] = []

        for message in base_formatted:
            role = message.get("role", "user")  # Default to user if no role
            content = message.get("content", "")

            # Azure AI Inference requires both 'role' and 'content'
            azure_messages.append({"role": role, "content": content})

        return azure_messages

    def _handle_completion(
        self,
        params: dict[str, Any],
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        """Handle non-streaming chat completion."""
        # Make API call
        try:
            response: ChatCompletions = self.client.complete(**params)

            if not response.choices:
                raise ValueError("No choices returned from Azure API")

            choice = response.choices[0]
            message = choice.message

            # Extract and track token usage
            usage = self._extract_azure_token_usage(response)
            self._track_token_usage_internal(usage)

            if response_model and self.is_openai_model:
                content = message.content or ""
                try:
                    structured_data = response_model.model_validate_json(content)
                    structured_json = structured_data.model_dump_json()

                    self._emit_call_completed_event(
                        response=structured_json,
                        call_type=LLMCallType.LLM_CALL,
                        from_task=from_task,
                        from_agent=from_agent,
                        messages=params["messages"],
                    )

                    return structured_json
                except Exception as e:
                    error_msg = f"Failed to validate structured output with model {response_model.__name__}: {e}"
                    logging.error(error_msg)
                    raise ValueError(error_msg) from e

            # Handle tool calls
            if message.tool_calls and available_functions:
                tool_call = message.tool_calls[0]  # Handle first tool call
                if isinstance(tool_call, ChatCompletionsToolCall):
                    function_name = tool_call.function.name

                    try:
                        function_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError as e:
                        logging.error(f"Failed to parse tool arguments: {e}")
                        function_args = {}

                    # Execute tool
                    result = self._handle_tool_execution(
                        function_name=function_name,
                        function_args=function_args,
                        available_functions=available_functions,
                        from_task=from_task,
                        from_agent=from_agent,
                    )

                    if result is not None:
                        return result

            # Extract content
            content = message.content or ""

            # Apply stop words
            content = self._apply_stop_words(content)

            # Emit completion event and return content
            self._emit_call_completed_event(
                response=content,
                call_type=LLMCallType.LLM_CALL,
                from_task=from_task,
                from_agent=from_agent,
                messages=params["messages"],
            )

        except Exception as e:
            if is_context_length_exceeded(e):
                logging.error(f"Context window exceeded: {e}")
                raise LLMContextLengthExceededError(str(e)) from e

            error_msg = f"Azure API call failed: {e!s}"
            logging.error(error_msg)
            self._emit_call_failed_event(
                error=error_msg, from_task=from_task, from_agent=from_agent
            )
            raise e

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

        # Make streaming API call
        for update in self.client.complete(**params):
            if isinstance(update, StreamingChatCompletionsUpdate):
                if update.choices:
                    choice = update.choices[0]
                    if choice.delta and choice.delta.content:
                        content_delta = choice.delta.content
                        full_response += content_delta
                        self._emit_stream_chunk_event(
                            chunk=content_delta,
                            from_task=from_task,
                            from_agent=from_agent,
                        )

                    # Handle tool call streaming
                    if choice.delta and choice.delta.tool_calls:
                        for tool_call in choice.delta.tool_calls:
                            call_id = tool_call.id or "default"
                            if call_id not in tool_calls:
                                tool_calls[call_id] = {
                                    "name": "",
                                    "arguments": "",
                                }

                            if tool_call.function and tool_call.function.name:
                                tool_calls[call_id]["name"] = tool_call.function.name
                            if tool_call.function and tool_call.function.arguments:
                                tool_calls[call_id]["arguments"] += (
                                    tool_call.function.arguments
                                )

        # Handle completed tool calls
        if tool_calls and available_functions:
            for call_data in tool_calls.values():
                function_name = call_data["name"]

                try:
                    function_args = json.loads(call_data["arguments"])
                except json.JSONDecodeError as e:
                    logging.error(f"Failed to parse streamed tool arguments: {e}")
                    continue

                # Execute tool
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
        # Azure OpenAI models support function calling
        return self.is_openai_model

    def supports_stop_words(self) -> bool:
        """Check if the model supports stop words."""
        return True  # Most Azure models support stop sequences

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

        # Context window sizes for common Azure models
        context_windows = {
            "gpt-4": 8192,
            "gpt-4o": 128000,
            "gpt-4o-mini": 200000,
            "gpt-4-turbo": 128000,
            "gpt-35-turbo": 16385,
            "gpt-3.5-turbo": 16385,
            "text-embedding": 8191,
        }

        # Find the best match for the model name
        for model_prefix, size in sorted(
            context_windows.items(), key=lambda x: len(x[0]), reverse=True
        ):
            if self.model.startswith(model_prefix):
                return int(size * CONTEXT_WINDOW_USAGE_RATIO)

        # Default context window size
        return int(8192 * CONTEXT_WINDOW_USAGE_RATIO)

    def _extract_azure_token_usage(self, response: ChatCompletions) -> dict[str, Any]:
        """Extract token usage from Azure response."""
        if hasattr(response, "usage") and response.usage:
            usage = response.usage
            return {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
            }
        return {"total_tokens": 0}
