from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Any, Literal, TypedDict

from pydantic import BaseModel
from typing_extensions import Self

from crewai.utilities.agent_utils import is_context_length_exceeded
from crewai.utilities.exceptions.context_window_exceeding_exception import (
    LLMContextLengthExceededError,
)
from crewai.utilities.pydantic_schema_utils import generate_model_description
from crewai.utilities.types import LLMMessage


if TYPE_CHECKING:
    from crewai.llms.hooks.base import BaseInterceptor


try:
    from azure.ai.inference import (
        ChatCompletionsClient,
    )
    from azure.ai.inference.aio import (
        ChatCompletionsClient as AsyncChatCompletionsClient,
    )
    from azure.ai.inference.models import (
        ChatCompletions,
        ChatCompletionsToolCall,
        ChatCompletionsToolDefinition,
        FunctionDefinition,
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
    from crewai.llms.base_llm import BaseLLM, llm_call_context

except ImportError:
    raise ImportError(
        'Azure AI Inference native provider not available, to install: uv add "crewai[azure-ai-inference]"'
    ) from None


class AzureCompletionParams(TypedDict, total=False):
    """Type definition for Azure chat completion parameters."""

    messages: list[LLMMessage]
    stream: bool
    model_extras: dict[str, Any]
    response_format: JsonSchemaFormat
    model: str
    temperature: float
    top_p: float
    frequency_penalty: float
    presence_penalty: float
    max_tokens: int
    stop: list[str]
    tools: list[ChatCompletionsToolDefinition]
    tool_choice: str


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
        max_completion_tokens: int | None = None,
        stop: list[str] | None = None,
        stream: bool = False,
        interceptor: BaseInterceptor[Any, Any] | None = None,
        response_format: type[BaseModel] | None = None,
        api: Literal["completions", "responses"] = "completions",
        instructions: str | None = None,
        store: bool | None = None,
        previous_response_id: str | None = None,
        include: list[str] | None = None,
        builtin_tools: list[str] | None = None,
        parse_tool_outputs: bool = False,
        auto_chain: bool = False,
        auto_chain_reasoning: bool = False,
        seed: int | None = None,
        reasoning_effort: str | None = None,
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
            max_completion_tokens: Maximum completion tokens (used by Responses API)
            stop: Stop sequences
            stream: Enable streaming responses
            interceptor: HTTP interceptor (not yet supported for Azure).
            response_format: Pydantic model for structured output. Used as default when
                           response_model is not passed to call()/acall() methods.
                           Only works with OpenAI models deployed on Azure.
            api: Which API to use - 'completions' for Chat Completions (default),
                 'responses' for OpenAI Responses API (requires Azure OpenAI endpoint).
            instructions: System instructions for Responses API.
            store: Whether to store the response for multi-turn (Responses API).
            previous_response_id: Response ID for multi-turn conversations (Responses API).
            include: Additional output types to include (Responses API).
            builtin_tools: Built-in tools like 'web_search', 'file_search' (Responses API).
            parse_tool_outputs: Return structured ResponsesAPIResult (Responses API).
            auto_chain: Automatically chain responses using response IDs (Responses API).
            auto_chain_reasoning: Auto-chain with reasoning items for ZDR (Responses API).
            seed: Random seed for deterministic outputs.
            reasoning_effort: Reasoning effort level for reasoning models.
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

        # Responses API parameters
        self.api = api
        self.instructions = instructions
        self.store = store
        self.previous_response_id = previous_response_id
        self.include = include
        self.builtin_tools = builtin_tools
        self.parse_tool_outputs = parse_tool_outputs
        self.auto_chain = auto_chain
        self.auto_chain_reasoning = auto_chain_reasoning
        self.max_completion_tokens = max_completion_tokens
        self.seed = seed
        self.reasoning_effort = reasoning_effort

        # Auto-chain state tracking
        self._last_response_id: str | None = None
        self._last_reasoning_items: list[Any] = []

        # Built-in tool type mapping (same as OpenAI)
        self.BUILTIN_TOOL_TYPES: dict[str, str] = {
            "web_search": "web_search_preview",
            "file_search": "file_search",
            "code_interpreter": "code_interpreter",
            "computer_use": "computer_use_preview",
        }

        # Validate and potentially fix Azure OpenAI endpoint URL
        self.endpoint = self._validate_and_fix_endpoint(self.endpoint, model)

        # Build client kwargs for Azure AI Inference (Chat Completions)
        client_kwargs = {
            "endpoint": self.endpoint,
            "credential": AzureKeyCredential(self.api_key),
        }

        # Add api_version if specified (primarily for Azure OpenAI endpoints)
        if self.api_version:
            client_kwargs["api_version"] = self.api_version

        self.client = ChatCompletionsClient(**client_kwargs)  # type: ignore[arg-type]

        self.async_client = AsyncChatCompletionsClient(**client_kwargs)  # type: ignore[arg-type]

        # If using Responses API, also create OpenAI AzureOpenAI clients
        if self.api == "responses":
            self._init_responses_clients()

        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.max_tokens = max_tokens
        self.stream = stream
        self.response_format = response_format

        self.is_openai_model = any(
            prefix in model.lower() for prefix in ["gpt-", "o1-", "text-"]
        )

        self.is_azure_openai_endpoint = (
            "openai.azure.com" in self.endpoint
            and "/openai/deployments/" in self.endpoint
        )

    def _init_responses_clients(self) -> None:
        """Initialize OpenAI AzureOpenAI clients for Responses API.

        The Responses API requires the OpenAI SDK's AzureOpenAI client,
        which supports the same responses.create() interface as the
        regular OpenAI client.
        """
        try:
            from openai import AsyncAzureOpenAI, AzureOpenAI
        except ImportError:
            raise ImportError(
                "The 'openai' package is required for Azure Responses API support. "
                "Install it with: uv add 'crewai[openai]' or pip install openai"
            ) from None

        # Extract the base Azure endpoint (without /openai/deployments/...)
        azure_endpoint = self.endpoint
        if "/openai/deployments/" in azure_endpoint:
            azure_endpoint = azure_endpoint.split("/openai/deployments/")[0]

        responses_kwargs: dict[str, Any] = {
            "api_key": self.api_key,
            "azure_endpoint": azure_endpoint,
            "api_version": self.api_version or "2025-03-01-preview",
        }

        if self.timeout is not None:
            responses_kwargs["timeout"] = self.timeout
        responses_kwargs["max_retries"] = self.max_retries

        self.responses_client = AzureOpenAI(**responses_kwargs)
        self.async_responses_client = AsyncAzureOpenAI(**responses_kwargs)

    @staticmethod
    def _validate_and_fix_endpoint(endpoint: str, model: str) -> str:
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

    def _handle_api_error(
        self,
        error: Exception,
        from_task: Any | None = None,
        from_agent: Any | None = None,
    ) -> None:
        """Handle API errors with appropriate logging and events.

        Args:
            error: The exception that occurred
            from_task: Task that initiated the call
            from_agent: Agent that initiated the call

        Raises:
            The original exception after logging and emitting events
        """
        if isinstance(error, HttpResponseError):
            if error.status_code == 401:
                error_msg = "Azure authentication failed. Check your API key."
            elif error.status_code == 404:
                error_msg = (
                    f"Azure endpoint not found. Check endpoint URL: {self.endpoint}"
                )
            elif error.status_code == 429:
                error_msg = "Azure API rate limit exceeded. Please retry later."
            else:
                error_msg = (
                    f"Azure API HTTP error: {error.status_code} - {error.message}"
                )
        else:
            error_msg = f"Azure API call failed: {error!s}"

        logging.error(error_msg)
        self._emit_call_failed_event(
            error=error_msg, from_task=from_task, from_agent=from_agent
        )
        raise error

    def _handle_completion_error(
        self,
        error: Exception,
        from_task: Any | None = None,
        from_agent: Any | None = None,
    ) -> None:
        """Handle completion-specific errors including context length checks.

        Args:
            error: The exception that occurred
            from_task: Task that initiated the call
            from_agent: Agent that initiated the call

        Raises:
            LLMContextLengthExceededError if context window exceeded, otherwise the original exception
        """
        if is_context_length_exceeded(error):
            logging.error(f"Context window exceeded: {error}")
            raise LLMContextLengthExceededError(str(error)) from error

        error_msg = f"Azure API call failed: {error!s}"
        logging.error(error_msg)
        self._emit_call_failed_event(
            error=error_msg, from_task=from_task, from_agent=from_agent
        )
        raise error

    @property
    def last_response_id(self) -> str | None:
        """Get the last response ID for auto-chaining."""
        return self._last_response_id

    def reset_chain(self) -> None:
        """Reset the auto-chain state."""
        self._last_response_id = None

    @property
    def last_reasoning_items(self) -> list[Any]:
        """Get the last reasoning items for ZDR auto-chaining."""
        return self._last_reasoning_items

    def reset_reasoning_chain(self) -> None:
        """Reset the reasoning auto-chain state."""
        self._last_reasoning_items = []

    def call(
        self,
        messages: str | list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
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
            response_model: Response model

        Returns:
            Chat completion response or tool call result
        """
        with llm_call_context():
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

                effective_response_model = response_model or self.response_format

                # Format messages for Azure
                formatted_messages = self._format_messages_for_azure(messages)

                if not self._invoke_before_llm_call_hooks(
                    formatted_messages, from_agent
                ):
                    raise ValueError("LLM call blocked by before_llm_call hook")

                # Route to Responses API if configured
                if self.api == "responses":
                    return self._call_responses(
                        messages=formatted_messages,
                        tools=tools,
                        available_functions=available_functions,
                        from_task=from_task,
                        from_agent=from_agent,
                        response_model=effective_response_model,
                    )

                # Prepare completion parameters
                completion_params = self._prepare_completion_params(
                    formatted_messages, tools, effective_response_model
                )

                # Handle streaming vs non-streaming
                if self.stream:
                    return self._handle_streaming_completion(
                        completion_params,
                        available_functions,
                        from_task,
                        from_agent,
                        effective_response_model,
                    )

                return self._handle_completion(
                    completion_params,
                    available_functions,
                    from_task,
                    from_agent,
                    effective_response_model,
                )

            except Exception as e:
                return self._handle_api_error(e, from_task, from_agent)  # type: ignore[func-returns-value]

    async def acall(  # type: ignore[return]
        self,
        messages: str | list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        """Call Azure AI Inference chat completions API asynchronously.

        Args:
            messages: Input messages for the chat completion
            tools: List of tool/function definitions
            callbacks: Callback functions (not used in native implementation)
            available_functions: Available functions for tool calling
            from_task: Task that initiated the call
            from_agent: Agent that initiated the call
            response_model: Pydantic model for structured output

        Returns:
            Chat completion response or tool call result
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

                effective_response_model = response_model or self.response_format

                formatted_messages = self._format_messages_for_azure(messages)

                # Route to Responses API if configured
                if self.api == "responses":
                    return await self._acall_responses(
                        messages=formatted_messages,
                        tools=tools,
                        available_functions=available_functions,
                        from_task=from_task,
                        from_agent=from_agent,
                        response_model=effective_response_model,
                    )

                completion_params = self._prepare_completion_params(
                    formatted_messages, tools, effective_response_model
                )

                if self.stream:
                    return await self._ahandle_streaming_completion(
                        completion_params,
                        available_functions,
                        from_task,
                        from_agent,
                        effective_response_model,
                    )

                return await self._ahandle_completion(
                    completion_params,
                    available_functions,
                    from_task,
                    from_agent,
                    effective_response_model,
                )

            except Exception as e:
                self._handle_api_error(e, from_task, from_agent)

    def _prepare_completion_params(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> AzureCompletionParams:
        """Prepare parameters for Azure AI Inference chat completion.

        Args:
            messages: Formatted messages for Azure
            tools: Tool definitions
            response_model: Pydantic model for structured output

        Returns:
            Parameters dictionary for Azure API
        """
        params: AzureCompletionParams = {
            "messages": messages,
            "stream": self.stream,
        }

        model_extras: dict[str, Any] = {}
        if self.stream:
            model_extras["stream_options"] = {"include_usage": True}

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
        if self.stop and self.supports_stop_words():
            params["stop"] = self.stop

        # Handle tools/functions for Azure OpenAI models
        if tools and self.is_openai_model:
            params["tools"] = self._convert_tools_for_interference(tools)
            params["tool_choice"] = "auto"

        prompt_cache_key = self.additional_params.get("prompt_cache_key")
        if prompt_cache_key:
            model_extras["prompt_cache_key"] = prompt_cache_key

        if model_extras:
            params["model_extras"] = model_extras

        additional_params = self.additional_params
        additional_drop_params = additional_params.get("additional_drop_params")
        drop_params = additional_params.get("drop_params")

        if drop_params and isinstance(additional_drop_params, list):
            for drop_param in additional_drop_params:
                if isinstance(drop_param, str):
                    params.pop(drop_param, None)  # type: ignore[misc]

        return params

    def _convert_tools_for_interference(  # type: ignore[override]
        self, tools: list[dict[str, Any]]
    ) -> list[ChatCompletionsToolDefinition]:
        """Convert CrewAI tool format to Azure OpenAI function calling format.

        Args:
            tools: List of CrewAI tool definitions

        Returns:
            List of Azure ChatCompletionsToolDefinition objects
        """
        from crewai.llms.providers.utils.common import safe_tool_conversion

        azure_tools: list[ChatCompletionsToolDefinition] = []

        for tool in tools:
            name, description, parameters = safe_tool_conversion(tool, "Azure")

            function_def = FunctionDefinition(
                name=name,
                description=description,
                parameters=parameters
                if isinstance(parameters, dict)
                else dict(parameters)
                if parameters
                else None,
            )

            tool_def = ChatCompletionsToolDefinition(function=function_def)

            azure_tools.append(tool_def)

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
            # Handle None content - Azure requires string content
            content = message.get("content") or ""

            if role == "tool":
                tool_call_id = message.get("tool_call_id", "")
                if not tool_call_id:
                    raise ValueError("Tool message missing required tool_call_id")
                azure_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": content,
                    }
                )
            # Handle assistant messages with tool_calls
            elif role == "assistant" and message.get("tool_calls"):
                tool_calls = message.get("tool_calls", [])
                azure_msg: LLMMessage = {
                    "role": "assistant",
                    "content": content,  # Already defaulted to "" above
                    "tool_calls": tool_calls,
                }
                azure_messages.append(azure_msg)
            else:
                # Azure AI Inference requires both 'role' and 'content'
                azure_messages.append({"role": role, "content": content})

        return azure_messages

    def _validate_and_emit_structured_output(
        self,
        content: str,
        response_model: type[BaseModel],
        params: AzureCompletionParams,
        from_task: Any | None = None,
        from_agent: Any | None = None,
    ) -> BaseModel:
        """Validate content against response model and emit completion event.

        Args:
            content: Response content to validate
            response_model: Pydantic model for validation
            params: Completion parameters containing messages
            from_task: Task that initiated the call
            from_agent: Agent that initiated the call

        Returns:
            Validated Pydantic model instance

        Raises:
            ValueError: If validation fails
        """
        try:
            structured_data = response_model.model_validate_json(content)

            self._emit_call_completed_event(
                response=structured_data.model_dump_json(),
                call_type=LLMCallType.LLM_CALL,
                from_task=from_task,
                from_agent=from_agent,
                messages=params["messages"],
            )

            return structured_data
        except Exception as e:
            error_msg = f"Failed to validate structured output with model {response_model.__name__}: {e}"
            logging.error(error_msg)
            raise ValueError(error_msg) from e

    def _process_completion_response(
        self,
        response: ChatCompletions,
        params: AzureCompletionParams,
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        """Process completion response with usage tracking, tool execution, and events.

        Args:
            response: Chat completion response from Azure API
            params: Completion parameters containing messages
            available_functions: Available functions for tool calling
            from_task: Task that initiated the call
            from_agent: Agent that initiated the call
            response_model: Pydantic model for structured output

        Returns:
            Response content or structured output
        """
        if not response.choices:
            raise ValueError("No choices returned from Azure API")

        choice = response.choices[0]
        message = choice.message

        # Extract and track token usage
        usage = self._extract_azure_token_usage(response)
        self._track_token_usage_internal(usage)

        # If there are tool_calls but no available_functions, return the tool_calls
        # This allows the caller (e.g., executor) to handle tool execution
        if message.tool_calls and not available_functions:
            self._emit_call_completed_event(
                response=list(message.tool_calls),
                call_type=LLMCallType.TOOL_CALL,
                from_task=from_task,
                from_agent=from_agent,
                messages=params["messages"],
            )
            return list(message.tool_calls)

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

        if response_model and self.is_openai_model:
            return self._validate_and_emit_structured_output(
                content=content,
                response_model=response_model,
                params=params,
                from_task=from_task,
                from_agent=from_agent,
            )

        content = self._apply_stop_words(content)

        # Emit completion event and return content
        self._emit_call_completed_event(
            response=content,
            call_type=LLMCallType.LLM_CALL,
            from_task=from_task,
            from_agent=from_agent,
            messages=params["messages"],
        )

        return self._invoke_after_llm_call_hooks(
            params["messages"], content, from_agent
        )

    def _handle_completion(
        self,
        params: AzureCompletionParams,
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        """Handle non-streaming chat completion."""
        try:
            # Cast params to Any to avoid type checking issues with TypedDict unpacking
            response: ChatCompletions = self.client.complete(**params)  # type: ignore[assignment,arg-type]
            return self._process_completion_response(
                response=response,
                params=params,
                available_functions=available_functions,
                from_task=from_task,
                from_agent=from_agent,
                response_model=response_model,
            )
        except Exception as e:
            return self._handle_completion_error(e, from_task, from_agent)  # type: ignore[func-returns-value]

    def _process_streaming_update(
        self,
        update: StreamingChatCompletionsUpdate,
        full_response: str,
        tool_calls: dict[int, dict[str, Any]],
        from_task: Any | None = None,
        from_agent: Any | None = None,
    ) -> str:
        """Process a single streaming update chunk.

        Args:
            update: Streaming update from Azure API
            full_response: Accumulated response content
            tool_calls: Dictionary of accumulated tool calls
            from_task: Task that initiated the call
            from_agent: Agent that initiated the call

        Returns:
            Updated full_response string
        """
        if update.choices:
            choice = update.choices[0]
            response_id = update.id if hasattr(update, "id") else None
            if choice.delta and choice.delta.content:
                content_delta = choice.delta.content
                full_response += content_delta
                self._emit_stream_chunk_event(
                    chunk=content_delta,
                    from_task=from_task,
                    from_agent=from_agent,
                    response_id=response_id,
                )

            if choice.delta and choice.delta.tool_calls:
                for idx, tool_call in enumerate(choice.delta.tool_calls):
                    if idx not in tool_calls:
                        tool_calls[idx] = {
                            "id": tool_call.id,
                            "name": "",
                            "arguments": "",
                        }
                    elif tool_call.id and not tool_calls[idx]["id"]:
                        tool_calls[idx]["id"] = tool_call.id

                    if tool_call.function and tool_call.function.name:
                        tool_calls[idx]["name"] = tool_call.function.name
                    if tool_call.function and tool_call.function.arguments:
                        tool_calls[idx]["arguments"] += tool_call.function.arguments

                    self._emit_stream_chunk_event(
                        chunk=tool_call.function.arguments
                        if tool_call.function and tool_call.function.arguments
                        else "",
                        from_task=from_task,
                        from_agent=from_agent,
                        tool_call={
                            "id": tool_calls[idx]["id"],
                            "function": {
                                "name": tool_calls[idx]["name"],
                                "arguments": tool_calls[idx]["arguments"],
                            },
                            "type": "function",
                            "index": idx,
                        },
                        call_type=LLMCallType.TOOL_CALL,
                        response_id=response_id,
                    )

        return full_response

    def _finalize_streaming_response(
        self,
        full_response: str,
        tool_calls: dict[int, dict[str, Any]],
        usage_data: dict[str, int],
        params: AzureCompletionParams,
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        """Finalize streaming response with usage tracking, tool execution, and events.

        Args:
            full_response: The complete streamed response content
            tool_calls: Dictionary of tool calls accumulated during streaming
            usage_data: Token usage data from the stream
            params: Completion parameters containing messages
            available_functions: Available functions for tool calling
            from_task: Task that initiated the call
            from_agent: Agent that initiated the call
            response_model: Pydantic model for structured output validation

        Returns:
            Final response content after processing, or structured output
        """
        self._track_token_usage_internal(usage_data)

        # Handle structured output validation
        if response_model and self.is_openai_model:
            return self._validate_and_emit_structured_output(
                content=full_response,
                response_model=response_model,
                params=params,
                from_task=from_task,
                from_agent=from_agent,
            )

        # If there are tool_calls but no available_functions, return them
        # in OpenAI-compatible format for executor to handle
        if tool_calls and not available_functions:
            formatted_tool_calls = [
                {
                    "id": call_data.get("id", f"call_{idx}"),
                    "type": "function",
                    "function": {
                        "name": call_data["name"],
                        "arguments": call_data["arguments"],
                    },
                }
                for idx, call_data in tool_calls.items()
            ]
            self._emit_call_completed_event(
                response=formatted_tool_calls,
                call_type=LLMCallType.TOOL_CALL,
                from_task=from_task,
                from_agent=from_agent,
                messages=params["messages"],
            )
            return formatted_tool_calls

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

        return self._invoke_after_llm_call_hooks(
            params["messages"], full_response, from_agent
        )

    def _handle_streaming_completion(
        self,
        params: AzureCompletionParams,
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        """Handle streaming chat completion."""
        full_response = ""
        tool_calls: dict[int, dict[str, Any]] = {}

        usage_data = {"total_tokens": 0}
        for update in self.client.complete(**params):  # type: ignore[arg-type]
            if isinstance(update, StreamingChatCompletionsUpdate):
                if update.usage:
                    usage = update.usage
                    usage_data = {
                        "prompt_tokens": usage.prompt_tokens,
                        "completion_tokens": usage.completion_tokens,
                        "total_tokens": usage.total_tokens,
                    }
                    continue

                full_response = self._process_streaming_update(
                    update=update,
                    full_response=full_response,
                    tool_calls=tool_calls,
                    from_task=from_task,
                    from_agent=from_agent,
                )

        return self._finalize_streaming_response(
            full_response=full_response,
            tool_calls=tool_calls,
            usage_data=usage_data,
            params=params,
            available_functions=available_functions,
            from_task=from_task,
            from_agent=from_agent,
            response_model=response_model,
        )

    async def _ahandle_completion(
        self,
        params: AzureCompletionParams,
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        """Handle non-streaming chat completion asynchronously."""
        try:
            # Cast params to Any to avoid type checking issues with TypedDict unpacking
            response: ChatCompletions = await self.async_client.complete(**params)  # type: ignore[assignment,arg-type]
            return self._process_completion_response(
                response=response,
                params=params,
                available_functions=available_functions,
                from_task=from_task,
                from_agent=from_agent,
                response_model=response_model,
            )
        except Exception as e:
            return self._handle_completion_error(e, from_task, from_agent)  # type: ignore[func-returns-value]

    async def _ahandle_streaming_completion(
        self,
        params: AzureCompletionParams,
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        """Handle streaming chat completion asynchronously."""
        full_response = ""
        tool_calls: dict[int, dict[str, Any]] = {}

        usage_data = {"total_tokens": 0}

        stream = await self.async_client.complete(**params)  # type: ignore[arg-type]
        async for update in stream:  # type: ignore[union-attr]
            if isinstance(update, StreamingChatCompletionsUpdate):
                if hasattr(update, "usage") and update.usage:
                    usage = update.usage
                    usage_data = {
                        "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                        "completion_tokens": getattr(usage, "completion_tokens", 0),
                        "total_tokens": getattr(usage, "total_tokens", 0),
                    }
                    continue

                full_response = self._process_streaming_update(
                    update=update,
                    full_response=full_response,
                    tool_calls=tool_calls,
                    from_task=from_task,
                    from_agent=from_agent,
                )

        return self._finalize_streaming_response(
            full_response=full_response,
            tool_calls=tool_calls,
            usage_data=usage_data,
            params=params,
            available_functions=available_functions,
            from_task=from_task,
            from_agent=from_agent,
            response_model=response_model,
        )

    # =========================================================================
    # Responses API methods
    # =========================================================================

    def _call_responses(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        """Call Azure OpenAI Responses API."""
        params = self._prepare_responses_params(
            messages=messages, tools=tools, response_model=response_model
        )

        if self.stream:
            return self._handle_streaming_responses(
                params=params,
                available_functions=available_functions,
                from_task=from_task,
                from_agent=from_agent,
                response_model=response_model,
            )

        return self._handle_responses(
            params=params,
            available_functions=available_functions,
            from_task=from_task,
            from_agent=from_agent,
            response_model=response_model,
        )

    async def _acall_responses(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        """Async call to Azure OpenAI Responses API."""
        params = self._prepare_responses_params(
            messages=messages, tools=tools, response_model=response_model
        )

        if self.stream:
            return await self._ahandle_streaming_responses(
                params=params,
                available_functions=available_functions,
                from_task=from_task,
                from_agent=from_agent,
                response_model=response_model,
            )

        return await self._ahandle_responses(
            params=params,
            available_functions=available_functions,
            from_task=from_task,
            from_agent=from_agent,
            response_model=response_model,
        )

    def _prepare_responses_params(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> dict[str, Any]:
        """Prepare parameters for Azure OpenAI Responses API.

        The Responses API uses a different structure than Chat Completions:
        - `input` instead of `messages`
        - `instructions` for system-level guidance (extracted from system messages)
        - `text.format` instead of `response_format` for structured outputs
        - Internally-tagged tool format (flat structure)
        """
        instructions: str | None = self.instructions
        input_messages: list[LLMMessage] = []

        for message in messages:
            if message.get("role") == "system":
                content = message.get("content", "")
                content_str = content if isinstance(content, str) else str(content)
                if instructions:
                    instructions = f"{instructions}\n\n{content_str}"
                else:
                    instructions = content_str
            else:
                input_messages.append(message)

        # Prepare input with optional reasoning items for ZDR chaining
        final_input: list[Any] = []
        if self.auto_chain_reasoning and self._last_reasoning_items:
            final_input.extend(self._last_reasoning_items)
        final_input.extend(input_messages if input_messages else messages)

        params: dict[str, Any] = {
            "model": self.model,
            "input": final_input,
        }

        if instructions:
            params["instructions"] = instructions

        if self.stream:
            params["stream"] = True

        if self.store is not None:
            params["store"] = self.store

        # Handle response chaining: explicit previous_response_id takes precedence
        if self.previous_response_id:
            params["previous_response_id"] = self.previous_response_id
        elif self.auto_chain and self._last_response_id:
            params["previous_response_id"] = self._last_response_id

        # Handle include parameter with auto_chain_reasoning support
        include_items: list[str] = list(self.include) if self.include else []
        if self.auto_chain_reasoning:
            if "reasoning.encrypted_content" not in include_items:
                include_items.append("reasoning.encrypted_content")
        if include_items:
            params["include"] = include_items

        params.update(self.additional_params)

        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.top_p is not None:
            params["top_p"] = self.top_p
        if self.max_completion_tokens is not None:
            params["max_output_tokens"] = self.max_completion_tokens
        elif self.max_tokens is not None:
            params["max_output_tokens"] = self.max_tokens
        if self.seed is not None:
            params["seed"] = self.seed

        if self.reasoning_effort:
            params["reasoning"] = {"effort": self.reasoning_effort}

        if response_model or self.response_format:
            format_model = response_model or self.response_format
            if isinstance(format_model, type) and issubclass(format_model, BaseModel):
                schema_output = generate_model_description(format_model)
                json_schema = schema_output.get("json_schema", {})
                params["text"] = {
                    "format": {
                        "type": "json_schema",
                        "name": json_schema.get("name", format_model.__name__),
                        "strict": json_schema.get("strict", True),
                        "schema": json_schema.get("schema", {}),
                    }
                }
            elif isinstance(format_model, dict):
                params["text"] = {"format": format_model}

        all_tools: list[dict[str, Any]] = []

        if self.builtin_tools:
            for tool_name in self.builtin_tools:
                tool_type = self.BUILTIN_TOOL_TYPES.get(tool_name, tool_name)
                all_tools.append({"type": tool_type})

        if tools:
            all_tools.extend(self._convert_tools_for_responses(tools))

        if all_tools:
            params["tools"] = all_tools

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

    def _convert_tools_for_responses(
        self, tools: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Convert CrewAI tools to Responses API format (internally-tagged).

        Responses API uses flat structure:
        {"type": "function", "name": "...", "description": "...", "parameters": {...}}
        """
        from crewai.llms.providers.utils.common import safe_tool_conversion

        responses_tools = []

        for tool in tools:
            name, description, parameters = safe_tool_conversion(tool, "Azure")

            responses_tool: dict[str, Any] = {
                "type": "function",
                "name": name,
                "description": description,
            }

            if parameters:
                if isinstance(parameters, dict):
                    responses_tool["parameters"] = parameters
                else:
                    responses_tool["parameters"] = dict(parameters)

            responses_tools.append(responses_tool)

        return responses_tools

    def _handle_responses(
        self,
        params: dict[str, Any],
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        """Handle non-streaming Responses API call."""
        from openai.types.responses import Response

        try:
            response: Response = self.responses_client.responses.create(**params)

            # Track response ID for auto-chaining
            if self.auto_chain and response.id:
                self._last_response_id = response.id

            # Track reasoning items for ZDR auto-chaining
            if self.auto_chain_reasoning:
                reasoning_items = self._extract_reasoning_items(response)
                if reasoning_items:
                    self._last_reasoning_items = reasoning_items

            usage = self._extract_responses_token_usage(response)
            self._track_token_usage_internal(usage)

            # If parse_tool_outputs is enabled, return structured result
            if self.parse_tool_outputs:
                parsed_result = self._extract_builtin_tool_outputs(response)
                parsed_result.text = self._apply_stop_words(parsed_result.text)

                self._emit_call_completed_event(
                    response=parsed_result.text,
                    call_type=LLMCallType.LLM_CALL,
                    from_task=from_task,
                    from_agent=from_agent,
                    messages=params.get("input", []),
                )

                return parsed_result

            function_calls = self._extract_function_calls_from_response(response)
            if function_calls and not available_functions:
                self._emit_call_completed_event(
                    response=function_calls,
                    call_type=LLMCallType.TOOL_CALL,
                    from_task=from_task,
                    from_agent=from_agent,
                    messages=params.get("input", []),
                )
                return function_calls

            if function_calls and available_functions:
                for call in function_calls:
                    function_name = call.get("name", "")
                    function_args = call.get("arguments", {})
                    if isinstance(function_args, str):
                        try:
                            function_args = json.loads(function_args)
                        except json.JSONDecodeError:
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
                        messages=params.get("input", []),
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
                messages=params.get("input", []),
            )

            content = self._invoke_after_llm_call_hooks(
                params.get("input", []), content, from_agent
            )

        except Exception as e:
            if is_context_length_exceeded(e):
                logging.error(f"Context window exceeded: {e}")
                raise LLMContextLengthExceededError(str(e)) from e

            error_msg = f"Azure Responses API call failed: {e!s}"
            logging.error(error_msg)
            self._emit_call_failed_event(
                error=error_msg, from_task=from_task, from_agent=from_agent
            )
            raise

        return content

    async def _ahandle_responses(
        self,
        params: dict[str, Any],
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        """Handle async non-streaming Responses API call."""
        from openai.types.responses import Response

        try:
            response: Response = await self.async_responses_client.responses.create(**params)

            # Track response ID for auto-chaining
            if self.auto_chain and response.id:
                self._last_response_id = response.id

            # Track reasoning items for ZDR auto-chaining
            if self.auto_chain_reasoning:
                reasoning_items = self._extract_reasoning_items(response)
                if reasoning_items:
                    self._last_reasoning_items = reasoning_items

            usage = self._extract_responses_token_usage(response)
            self._track_token_usage_internal(usage)

            # If parse_tool_outputs is enabled, return structured result
            if self.parse_tool_outputs:
                parsed_result = self._extract_builtin_tool_outputs(response)
                parsed_result.text = self._apply_stop_words(parsed_result.text)

                self._emit_call_completed_event(
                    response=parsed_result.text,
                    call_type=LLMCallType.LLM_CALL,
                    from_task=from_task,
                    from_agent=from_agent,
                    messages=params.get("input", []),
                )

                return parsed_result

            function_calls = self._extract_function_calls_from_response(response)
            if function_calls and not available_functions:
                self._emit_call_completed_event(
                    response=function_calls,
                    call_type=LLMCallType.TOOL_CALL,
                    from_task=from_task,
                    from_agent=from_agent,
                    messages=params.get("input", []),
                )
                return function_calls

            if function_calls and available_functions:
                for call in function_calls:
                    function_name = call.get("name", "")
                    function_args = call.get("arguments", {})
                    if isinstance(function_args, str):
                        try:
                            function_args = json.loads(function_args)
                        except json.JSONDecodeError:
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
                        messages=params.get("input", []),
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
                messages=params.get("input", []),
            )

        except Exception as e:
            if is_context_length_exceeded(e):
                logging.error(f"Context window exceeded: {e}")
                raise LLMContextLengthExceededError(str(e)) from e

            error_msg = f"Azure Responses API call failed: {e!s}"
            logging.error(error_msg)
            self._emit_call_failed_event(
                error=error_msg, from_task=from_task, from_agent=from_agent
            )
            raise

        return content

    def _handle_streaming_responses(
        self,
        params: dict[str, Any],
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        """Handle streaming Responses API call."""
        from openai.types.responses import Response

        full_response = ""
        function_calls: list[dict[str, Any]] = []
        final_response: Response | None = None

        stream = self.responses_client.responses.create(**params)
        response_id_stream = None

        for event in stream:
            if event.type == "response.created":
                response_id_stream = event.response.id

            if event.type == "response.output_text.delta":
                delta_text = event.delta or ""
                full_response += delta_text
                self._emit_stream_chunk_event(
                    chunk=delta_text,
                    from_task=from_task,
                    from_agent=from_agent,
                    response_id=response_id_stream,
                )

            elif event.type == "response.function_call_arguments.delta":
                pass

            elif event.type == "response.output_item.done":
                item = event.item
                if item.type == "function_call":
                    function_calls.append(
                        {
                            "id": item.call_id,
                            "name": item.name,
                            "arguments": item.arguments,
                        }
                    )

            elif event.type == "response.completed":
                final_response = event.response
                if self.auto_chain and event.response and event.response.id:
                    self._last_response_id = event.response.id
                if self.auto_chain_reasoning and event.response:
                    reasoning_items = self._extract_reasoning_items(event.response)
                    if reasoning_items:
                        self._last_reasoning_items = reasoning_items
                if event.response and event.response.usage:
                    usage = self._extract_responses_token_usage(event.response)
                    self._track_token_usage_internal(usage)

        # If parse_tool_outputs is enabled, return structured result
        if self.parse_tool_outputs and final_response:
            parsed_result = self._extract_builtin_tool_outputs(final_response)
            parsed_result.text = self._apply_stop_words(parsed_result.text)

            self._emit_call_completed_event(
                response=parsed_result.text,
                call_type=LLMCallType.LLM_CALL,
                from_task=from_task,
                from_agent=from_agent,
                messages=params.get("input", []),
            )

            return parsed_result

        if function_calls and available_functions:
            for call in function_calls:
                function_name = call.get("name", "")
                function_args = call.get("arguments", {})
                if isinstance(function_args, str):
                    try:
                        function_args = json.loads(function_args)
                    except json.JSONDecodeError:
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

        if response_model:
            try:
                structured_result = self._validate_structured_output(
                    full_response, response_model
                )
                self._emit_call_completed_event(
                    response=structured_result,
                    call_type=LLMCallType.LLM_CALL,
                    from_task=from_task,
                    from_agent=from_agent,
                    messages=params.get("input", []),
                )
                return structured_result
            except ValueError as e:
                logging.warning(f"Structured output validation failed: {e}")

        full_response = self._apply_stop_words(full_response)

        self._emit_call_completed_event(
            response=full_response,
            call_type=LLMCallType.LLM_CALL,
            from_task=from_task,
            from_agent=from_agent,
            messages=params.get("input", []),
        )

        return self._invoke_after_llm_call_hooks(
            params.get("input", []), full_response, from_agent
        )

    async def _ahandle_streaming_responses(
        self,
        params: dict[str, Any],
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        """Handle async streaming Responses API call."""
        from openai.types.responses import Response

        full_response = ""
        function_calls: list[dict[str, Any]] = []
        final_response: Response | None = None

        stream = await self.async_responses_client.responses.create(**params)
        response_id_stream = None

        async for event in stream:
            if event.type == "response.created":
                response_id_stream = event.response.id

            if event.type == "response.output_text.delta":
                delta_text = event.delta or ""
                full_response += delta_text
                self._emit_stream_chunk_event(
                    chunk=delta_text,
                    from_task=from_task,
                    from_agent=from_agent,
                    response_id=response_id_stream,
                )

            elif event.type == "response.function_call_arguments.delta":
                pass

            elif event.type == "response.output_item.done":
                item = event.item
                if item.type == "function_call":
                    function_calls.append(
                        {
                            "id": item.call_id,
                            "name": item.name,
                            "arguments": item.arguments,
                        }
                    )

            elif event.type == "response.completed":
                final_response = event.response
                if self.auto_chain and event.response and event.response.id:
                    self._last_response_id = event.response.id
                if self.auto_chain_reasoning and event.response:
                    reasoning_items = self._extract_reasoning_items(event.response)
                    if reasoning_items:
                        self._last_reasoning_items = reasoning_items
                if event.response and event.response.usage:
                    usage = self._extract_responses_token_usage(event.response)
                    self._track_token_usage_internal(usage)

        # If parse_tool_outputs is enabled, return structured result
        if self.parse_tool_outputs and final_response:
            parsed_result = self._extract_builtin_tool_outputs(final_response)
            parsed_result.text = self._apply_stop_words(parsed_result.text)

            self._emit_call_completed_event(
                response=parsed_result.text,
                call_type=LLMCallType.LLM_CALL,
                from_task=from_task,
                from_agent=from_agent,
                messages=params.get("input", []),
            )

            return parsed_result

        if function_calls and available_functions:
            for call in function_calls:
                function_name = call.get("name", "")
                function_args = call.get("arguments", {})
                if isinstance(function_args, str):
                    try:
                        function_args = json.loads(function_args)
                    except json.JSONDecodeError:
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

        if response_model:
            try:
                structured_result = self._validate_structured_output(
                    full_response, response_model
                )
                self._emit_call_completed_event(
                    response=structured_result,
                    call_type=LLMCallType.LLM_CALL,
                    from_task=from_task,
                    from_agent=from_agent,
                    messages=params.get("input", []),
                )
                return structured_result
            except ValueError as e:
                logging.warning(f"Structured output validation failed: {e}")

        full_response = self._apply_stop_words(full_response)

        self._emit_call_completed_event(
            response=full_response,
            call_type=LLMCallType.LLM_CALL,
            from_task=from_task,
            from_agent=from_agent,
            messages=params.get("input", []),
        )

        return full_response

    def _extract_function_calls_from_response(self, response: Any) -> list[dict[str, Any]]:
        """Extract function calls from Responses API output."""
        return [
            {
                "id": item.call_id,
                "name": item.name,
                "arguments": item.arguments,
            }
            for item in response.output
            if item.type == "function_call"
        ]

    def _extract_responses_token_usage(self, response: Any) -> dict[str, Any]:
        """Extract token usage from Responses API response."""
        if response.usage:
            result = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            input_details = getattr(response.usage, "input_tokens_details", None)
            if input_details:
                result["cached_prompt_tokens"] = (
                    getattr(input_details, "cached_tokens", 0) or 0
                )
            return result
        return {"total_tokens": 0}

    def _extract_builtin_tool_outputs(self, response: Any) -> Any:
        """Extract and parse all built-in tool outputs from Responses API.

        Returns a ResponsesAPIResult from the OpenAI provider module.
        """
        from crewai.llms.providers.openai.completion import (
            CodeInterpreterFileResult,
            CodeInterpreterLogResult,
            CodeInterpreterResult,
            ComputerUseResult,
            FileSearchResult,
            FileSearchResultItem,
            ReasoningSummary,
            ResponsesAPIResult,
            WebSearchResult,
        )

        result = ResponsesAPIResult(
            text=response.output_text or "",
            response_id=response.id,
        )

        for item in response.output:
            item_type = item.type

            if item_type == "web_search_call":
                result.web_search_results.append(
                    WebSearchResult(
                        id=item.id,
                        status=item.status,  # type: ignore[union-attr]
                        type=item_type,
                    )
                )

            elif item_type == "file_search_call":
                file_results: list[FileSearchResultItem] = (
                    [
                        FileSearchResultItem(
                            file_id=r.file_id,  # type: ignore[union-attr]
                            filename=r.filename,  # type: ignore[union-attr]
                            text=r.text,  # type: ignore[union-attr]
                            score=r.score,  # type: ignore[union-attr]
                            attributes=r.attributes,  # type: ignore[union-attr]
                        )
                        for r in item.results  # type: ignore[union-attr]
                    ]
                    if item.results  # type: ignore[union-attr]
                    else []
                )
                result.file_search_results.append(
                    FileSearchResult(
                        id=item.id,
                        status=item.status,  # type: ignore[union-attr]
                        type=item_type,
                        queries=list(item.queries),  # type: ignore[union-attr]
                        results=file_results,
                    )
                )

            elif item_type == "code_interpreter_call":
                code_results: list[
                    CodeInterpreterLogResult | CodeInterpreterFileResult
                ] = []
                for r in item.results:  # type: ignore[union-attr]
                    if r.type == "logs":  # type: ignore[union-attr]
                        code_results.append(
                            CodeInterpreterLogResult(type="logs", logs=r.logs)  # type: ignore[union-attr]
                        )
                    elif r.type == "files":  # type: ignore[union-attr]
                        files_data = [
                            {"file_id": f.file_id, "mime_type": f.mime_type}
                            for f in r.files  # type: ignore[union-attr]
                        ]
                        code_results.append(
                            CodeInterpreterFileResult(type="files", files=files_data)
                        )
                result.code_interpreter_results.append(
                    CodeInterpreterResult(
                        id=item.id,
                        status=item.status,  # type: ignore[union-attr]
                        type=item_type,
                        code=item.code,  # type: ignore[union-attr]
                        container_id=item.container_id,  # type: ignore[union-attr]
                        results=code_results,
                    )
                )

            elif item_type == "computer_call":
                action_dict = item.action.model_dump() if item.action else {}  # type: ignore[union-attr]
                safety_checks = [
                    {"id": c.id, "code": c.code, "message": c.message}
                    for c in item.pending_safety_checks  # type: ignore[union-attr]
                ]
                result.computer_use_results.append(
                    ComputerUseResult(
                        id=item.id,
                        status=item.status,  # type: ignore[union-attr]
                        type=item_type,
                        call_id=item.call_id,  # type: ignore[union-attr]
                        action=action_dict,
                        pending_safety_checks=safety_checks,
                    )
                )

            elif item_type == "reasoning":
                summaries = [{"type": s.type, "text": s.text} for s in item.summary]  # type: ignore[union-attr]
                result.reasoning_summaries.append(
                    ReasoningSummary(
                        id=item.id,
                        status=item.status,  # type: ignore[union-attr]
                        type=item_type,
                        summary=summaries,
                        encrypted_content=item.encrypted_content,  # type: ignore[union-attr]
                    )
                )

            elif item_type == "function_call":
                result.function_calls.append(
                    {
                        "id": item.call_id,  # type: ignore[union-attr]
                        "name": item.name,  # type: ignore[union-attr]
                        "arguments": item.arguments,  # type: ignore[union-attr]
                    }
                )

        return result

    def _extract_reasoning_items(self, response: Any) -> list[Any]:
        """Extract reasoning items with encrypted content from response."""
        return [item for item in response.output if item.type == "reasoning"]

    def _validate_structured_output(
        self, content: str, response_model: type[BaseModel]
    ) -> BaseModel:
        """Validate and parse structured output content against a Pydantic model.

        Args:
            content: JSON string content from the response
            response_model: Pydantic model class for validation

        Returns:
            Validated Pydantic model instance

        Raises:
            ValueError: If content cannot be parsed/validated
        """
        try:
            return response_model.model_validate_json(content)
        except Exception as e:
            raise ValueError(
                f"Failed to validate structured output with model "
                f"{response_model.__name__}: {e}"
            ) from e

    def supports_function_calling(self) -> bool:
        """Check if the model supports function calling."""
        # Azure OpenAI models support function calling
        return self.is_openai_model

    def supports_stop_words(self) -> bool:
        """Check if the model supports stop words.

        Models using the Responses API (GPT-5 family, o-series reasoning models,
        computer-use-preview) do not support stop sequences.
        See: https://learn.microsoft.com/en-us/azure/ai-foundry/foundry-models/concepts/models-sold-directly-by-azure
        """
        model_lower = self.model.lower() if self.model else ""

        if "gpt-5" in model_lower:
            return False

        o_series_models = ["o1", "o3", "o4", "o1-mini", "o3-mini", "o4-mini"]

        responses_api_models = ["computer-use-preview"]

        unsupported_stop_models = o_series_models + responses_api_models

        for unsupported in unsupported_stop_models:
            if unsupported in model_lower:
                return False

        return True

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

    @staticmethod
    def _extract_azure_token_usage(response: ChatCompletions) -> dict[str, Any]:
        """Extract token usage from Azure response."""
        if hasattr(response, "usage") and response.usage:
            usage = response.usage
            cached_tokens = 0
            prompt_details = getattr(usage, "prompt_tokens_details", None)
            if prompt_details:
                cached_tokens = getattr(prompt_details, "cached_tokens", 0) or 0
            return {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
                "cached_prompt_tokens": cached_tokens,
            }
        return {"total_tokens": 0}

    async def aclose(self) -> None:
        """Close the async client and clean up resources.

        This ensures proper cleanup of the underlying aiohttp session
        to avoid unclosed connector warnings.
        """
        if hasattr(self.async_client, "close"):
            await self.async_client.close()

    async def __aenter__(self) -> Self:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.aclose()

    def supports_multimodal(self) -> bool:
        """Check if the model supports multimodal inputs.

        Azure OpenAI vision-enabled models include GPT-4o and GPT-4 Turbo with Vision.

        Returns:
            True if the model supports images.
        """
        vision_models = ("gpt-4o", "gpt-4-turbo", "gpt-4-vision", "gpt-4v")
        return any(self.model.lower().startswith(m) for m in vision_models)
