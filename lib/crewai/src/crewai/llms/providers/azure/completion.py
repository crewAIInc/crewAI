from __future__ import annotations

import json
import logging
import os
from typing import Any, Literal, TypedDict
from urllib.parse import urlparse

from pydantic import BaseModel, PrivateAttr, model_validator
from typing_extensions import Self

from crewai.llms.hooks.base import BaseInterceptor
from crewai.utilities.agent_utils import is_context_length_exceeded
from crewai.utilities.exceptions.context_window_exceeding_exception import (
    LLMContextLengthExceededError,
)
from crewai.utilities.pydantic_schema_utils import generate_model_description
from crewai.utilities.types import LLMMessage


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

    llm_type: Literal["azure"] = "azure"
    endpoint: str | None = None
    api_version: str | None = None
    timeout: float | None = None
    max_retries: int = 2
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    interceptor: BaseInterceptor[Any, Any] | None = None
    response_format: type[BaseModel] | None = None
    is_openai_model: bool = False
    is_azure_openai_endpoint: bool = False

    _client: Any = PrivateAttr(default=None)
    _async_client: Any = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def _normalize_azure_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        if data.get("interceptor") is not None:
            raise NotImplementedError(
                "HTTP interceptors are not yet supported for Azure AI Inference provider. "
                "Interceptors are currently supported for OpenAI and Anthropic providers only."
            )

        # Resolve env vars
        data["api_key"] = data.get("api_key") or os.getenv("AZURE_API_KEY")
        data["endpoint"] = (
            data.get("endpoint")
            or os.getenv("AZURE_ENDPOINT")
            or os.getenv("AZURE_OPENAI_ENDPOINT")
            or os.getenv("AZURE_API_BASE")
        )
        data["api_version"] = (
            data.get("api_version") or os.getenv("AZURE_API_VERSION") or "2024-06-01"
        )

        # Credentials and endpoint are validated lazily in `_init_clients`
        # so the LLM can be constructed before deployment env vars are set.
        model = data.get("model", "")
        if data["endpoint"]:
            data["endpoint"] = AzureCompletion._validate_and_fix_endpoint(
                data["endpoint"], model
            )
        data["is_azure_openai_endpoint"] = AzureCompletion._is_azure_openai_endpoint(
            data["endpoint"]
        )
        data["is_openai_model"] = any(
            prefix in model.lower() for prefix in ["gpt-", "o1-", "text-"]
        )
        return data

    @staticmethod
    def _is_azure_openai_endpoint(endpoint: str | None) -> bool:
        if not endpoint:
            return False
        hostname = urlparse(endpoint).hostname or ""
        return (
            hostname == "openai.azure.com" or hostname.endswith(".openai.azure.com")
        ) and "/openai/deployments/" in endpoint

    @model_validator(mode="after")
    def _init_clients(self) -> AzureCompletion:
        """Eagerly build clients when credentials are available, otherwise
        defer so ``LLM(model="azure/...")`` can be constructed at module
        import time even before deployment env vars are set.
        """
        try:
            self._client = self._build_sync_client()
            self._async_client = self._build_async_client()
        except ValueError:
            pass
        return self

    def _build_sync_client(self) -> Any:
        return ChatCompletionsClient(**self._make_client_kwargs())

    def _build_async_client(self) -> Any:
        return AsyncChatCompletionsClient(**self._make_client_kwargs())

    def _make_client_kwargs(self) -> dict[str, Any]:
        # Re-read env vars so that a deferred build can pick up credentials
        # that weren't set at instantiation time (e.g. LLM constructed at
        # module import before deployment env vars were injected).
        if not self.api_key:
            self.api_key = os.getenv("AZURE_API_KEY")
        if not self.endpoint:
            endpoint = (
                os.getenv("AZURE_ENDPOINT")
                or os.getenv("AZURE_OPENAI_ENDPOINT")
                or os.getenv("AZURE_API_BASE")
            )
            if endpoint:
                self.endpoint = AzureCompletion._validate_and_fix_endpoint(
                    endpoint, self.model
                )
                # Recompute the routing flag now that the endpoint is known —
                # _prepare_completion_params uses it to decide whether to
                # include `model` in the request body (Azure OpenAI endpoints
                # embed the deployment name in the URL and reject it).
                self.is_azure_openai_endpoint = (
                    AzureCompletion._is_azure_openai_endpoint(self.endpoint)
                )

        if not self.endpoint:
            raise ValueError(
                "Azure endpoint is required. Set AZURE_ENDPOINT environment "
                "variable or pass endpoint parameter."
            )
        client_kwargs: dict[str, Any] = {
            "endpoint": self.endpoint,
            "credential": self._resolve_credential(),
        }
        if self.api_version:
            client_kwargs["api_version"] = self.api_version
        return client_kwargs

    def _resolve_credential(self) -> Any:
        """Return an Azure credential, preferring the API key when set.

        Without an API key, fall back to ``DefaultAzureCredential`` from
        ``azure-identity``. That chain auto-detects the standard keyless
        paths the customer's environment may provide — OIDC Workload
        Identity Federation (``AZURE_FEDERATED_TOKEN_FILE`` +
        ``AZURE_TENANT_ID`` + ``AZURE_CLIENT_ID``), Managed Identity on
        AKS/Azure VMs, environment-configured service principals, and
        developer tools like the Azure CLI. Installing ``azure-identity``
        is what enables these paths; without it we raise the existing
        API-key error.
        """
        if self.api_key:
            return AzureKeyCredential(self.api_key)

        try:
            from azure.identity import DefaultAzureCredential
        except ImportError:
            raise ValueError(
                "Azure API key is required when azure-identity is not "
                "installed. Set AZURE_API_KEY, or install azure-identity "
                'for keyless auth: uv add "crewai[azure-ai-inference]"'
            ) from None

        return DefaultAzureCredential()

    def _get_sync_client(self) -> Any:
        if self._client is None:
            self._client = self._build_sync_client()
        return self._client

    def _get_async_client(self) -> Any:
        if self._async_client is None:
            self._async_client = self._build_async_client()
        return self._async_client

    def to_config_dict(self) -> dict[str, Any]:
        """Extend base config with Azure-specific fields."""
        config = super().to_config_dict()
        if self.endpoint:
            config["endpoint"] = self.endpoint
        if self.api_version and self.api_version != "2024-06-01":
            config["api_version"] = self.api_version
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
        return config

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
        ep_host = urlparse(endpoint).hostname or ""
        is_azure_openai = ep_host == "openai.azure.com" or ep_host.endswith(
            ".openai.azure.com"
        )
        if is_azure_openai and "/openai/deployments/" not in endpoint:
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
        usage: dict[str, Any] | None = None,
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
                usage=usage,
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
                usage=usage,
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
                usage=usage,
            )

        content = self._apply_stop_words(content)

        # Emit completion event and return content
        self._emit_call_completed_event(
            response=content,
            call_type=LLMCallType.LLM_CALL,
            from_task=from_task,
            from_agent=from_agent,
            messages=params["messages"],
            usage=usage,
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
            response: ChatCompletions = self._get_sync_client().complete(**params)
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
        usage_data: dict[str, Any] | None,
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
            usage_data: Token usage data from the stream, or None if unavailable
            params: Completion parameters containing messages
            available_functions: Available functions for tool calling
            from_task: Task that initiated the call
            from_agent: Agent that initiated the call
            response_model: Pydantic model for structured output validation

        Returns:
            Final response content after processing, or structured output
        """
        if usage_data:
            self._track_token_usage_internal(usage_data)

        # Handle structured output validation
        if response_model and self.is_openai_model:
            return self._validate_and_emit_structured_output(
                content=full_response,
                response_model=response_model,
                params=params,
                from_task=from_task,
                from_agent=from_agent,
                usage=usage_data,
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
                usage=usage_data,
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
            usage=usage_data,
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

        usage_data: dict[str, Any] | None = None
        for update in self._get_sync_client().complete(**params):
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
            response: ChatCompletions = await self._get_async_client().complete(
                **params
            )
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

        usage_data: dict[str, Any] | None = None

        stream = await self._get_async_client().complete(**params)
        async for update in stream:
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
        """Extract token usage and response metadata from Azure response."""
        if hasattr(response, "usage") and response.usage:
            usage = response.usage
            cached_tokens = 0
            prompt_details = getattr(usage, "prompt_tokens_details", None)
            if prompt_details:
                cached_tokens = getattr(prompt_details, "cached_tokens", 0) or 0
            reasoning_tokens = 0
            completion_details = getattr(usage, "completion_tokens_details", None)
            if completion_details:
                reasoning_tokens = (
                    getattr(completion_details, "reasoning_tokens", 0) or 0
                )
            result: dict[str, Any] = {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
                "cached_prompt_tokens": cached_tokens,
                "reasoning_tokens": reasoning_tokens,
            }
            return result
        return {"total_tokens": 0}

    async def aclose(self) -> None:
        """Close the async client and clean up resources.

        This ensures proper cleanup of the underlying aiohttp session
        to avoid unclosed connector warnings. Accesses the cached client
        directly rather than going through `_get_async_client` so a
        cleanup on an uninitialized LLM is a harmless no-op rather than
        a credential-required error.
        """
        if self._async_client is not None and hasattr(self._async_client, "close"):
            await self._async_client.close()

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
