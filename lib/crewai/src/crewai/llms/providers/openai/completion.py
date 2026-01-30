from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
import json
import logging
import os
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypedDict

import httpx
from openai import APIConnectionError, AsyncOpenAI, NotFoundError, OpenAI, Stream
from openai.lib.streaming.chat import ChatCompletionStream
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from openai.types.responses import Response
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


class WebSearchResult(TypedDict, total=False):
    """Result from web search built-in tool."""

    id: str | None
    status: str | None
    type: str


class FileSearchResultItem(TypedDict, total=False):
    """Individual file search result."""

    file_id: str | None
    filename: str | None
    text: str | None
    score: float | None
    attributes: dict[str, str | float | bool] | None


class FileSearchResult(TypedDict, total=False):
    """Result from file search built-in tool."""

    id: str | None
    status: str | None
    type: str
    queries: list[str]
    results: list[FileSearchResultItem]


class CodeInterpreterLogResult(TypedDict):
    """Log output from code interpreter."""

    type: str
    logs: str


class CodeInterpreterFileResult(TypedDict):
    """File output from code interpreter."""

    type: str
    files: list[dict[str, Any]]


class CodeInterpreterResult(TypedDict, total=False):
    """Result from code interpreter built-in tool."""

    id: str | None
    status: str | None
    type: str
    code: str | None
    container_id: str | None
    results: list[CodeInterpreterLogResult | CodeInterpreterFileResult]


class ComputerUseResult(TypedDict, total=False):
    """Result from computer use built-in tool."""

    id: str | None
    status: str | None
    type: str
    call_id: str | None
    action: dict[str, Any]
    pending_safety_checks: list[dict[str, Any]]


class ReasoningSummary(TypedDict, total=False):
    """Summary from model reasoning."""

    id: str | None
    status: str | None
    type: str
    summary: list[dict[str, Any]]
    encrypted_content: str | None


@dataclass
class ResponsesAPIResult:
    """Result from OpenAI Responses API including text and tool outputs.

    Attributes:
        text: The text content from the response.
        web_search_results: Results from web_search built-in tool calls.
        file_search_results: Results from file_search built-in tool calls.
        code_interpreter_results: Results from code_interpreter built-in tool calls.
        computer_use_results: Results from computer_use built-in tool calls.
        reasoning_summaries: Reasoning/thinking summaries from the model.
        function_calls: Custom function tool calls.
        response_id: The response ID for multi-turn conversations.
    """

    text: str = ""
    web_search_results: list[WebSearchResult] = field(default_factory=list)
    file_search_results: list[FileSearchResult] = field(default_factory=list)
    code_interpreter_results: list[CodeInterpreterResult] = field(default_factory=list)
    computer_use_results: list[ComputerUseResult] = field(default_factory=list)
    reasoning_summaries: list[ReasoningSummary] = field(default_factory=list)
    function_calls: list[dict[str, Any]] = field(default_factory=list)
    response_id: str | None = None

    def has_tool_outputs(self) -> bool:
        """Check if there are any built-in tool outputs."""
        return bool(
            self.web_search_results
            or self.file_search_results
            or self.code_interpreter_results
            or self.computer_use_results
        )

    def has_reasoning(self) -> bool:
        """Check if there are reasoning summaries."""
        return bool(self.reasoning_summaries)


class OpenAICompletion(BaseLLM):
    """OpenAI native completion implementation.

    This class provides direct integration with the OpenAI Python SDK,
    supporting both Chat Completions API and Responses API.

    The Responses API is OpenAI's newer API primitive with built-in tools
    (web search, file search, code interpreter), stateful conversations,
    and improved reasoning model support.

    Args:
        api: Which OpenAI API to use - "completions" (default) or "responses".
        instructions: System-level instructions (Responses API only).
        store: Whether to store responses for multi-turn (Responses API only).
        previous_response_id: ID of previous response for multi-turn (Responses API only).
        include: Additional data to include in response (Responses API only).
        builtin_tools: List of OpenAI built-in tools to enable (Responses API only).
            Supported: "web_search", "file_search", "code_interpreter", "computer_use".
        parse_tool_outputs: Whether to return structured ResponsesAPIResult with
            parsed built-in tool outputs instead of just text (Responses API only).
        auto_chain: Automatically track and use response IDs for multi-turn
            conversations (Responses API only). When True, each response ID is saved
            and used as previous_response_id in subsequent calls.
        auto_chain_reasoning: Automatically track and pass encrypted reasoning items
            for ZDR (Zero Data Retention) compliance (Responses API only). When True,
            adds "reasoning.encrypted_content" to include, captures reasoning items
            from responses, and passes them back in subsequent calls to preserve
            chain-of-thought without storing data on OpenAI servers.
    """

    BUILTIN_TOOL_TYPES: ClassVar[dict[str, str]] = {
        "web_search": "web_search_preview",
        "file_search": "file_search",
        "code_interpreter": "code_interpreter",
        "computer_use": "computer_use_preview",
    }

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
        interceptor: BaseInterceptor[httpx.Request, httpx.Response] | None = None,
        api: Literal["completions", "responses"] = "completions",
        instructions: str | None = None,
        store: bool | None = None,
        previous_response_id: str | None = None,
        include: list[str] | None = None,
        builtin_tools: list[str] | None = None,
        parse_tool_outputs: bool = False,
        auto_chain: bool = False,
        auto_chain_reasoning: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize OpenAI completion client."""

        if provider is None:
            provider = kwargs.pop("provider", "openai")

        self.interceptor = interceptor
        # Client configuration attributes
        self.organization = organization
        self.project = project
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
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url,
            timeout=timeout,
            provider=provider,
            **kwargs,
        )

        client_config = self._get_client_params()
        if self.interceptor:
            transport = HTTPTransport(interceptor=self.interceptor)
            http_client = httpx.Client(transport=transport)
            client_config["http_client"] = http_client

        self.client = OpenAI(**client_config)

        async_client_config = self._get_client_params()
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
        self.max_completion_tokens = max_completion_tokens
        self.seed = seed
        self.stream = stream
        self.response_format = response_format
        self.logprobs = logprobs
        self.top_logprobs = top_logprobs
        self.reasoning_effort = reasoning_effort
        self.is_o1_model = "o1" in model.lower()
        self.is_gpt4_model = "gpt-4" in model.lower()

        # API selection and Responses API parameters
        self.api = api
        self.instructions = instructions
        self.store = store
        self.previous_response_id = previous_response_id
        self.include = include
        self.builtin_tools = builtin_tools
        self.parse_tool_outputs = parse_tool_outputs
        self.auto_chain = auto_chain
        self.auto_chain_reasoning = auto_chain_reasoning
        self._last_response_id: str | None = None
        self._last_reasoning_items: list[Any] | None = None

    @property
    def last_response_id(self) -> str | None:
        """Get the last response ID from auto-chaining.

        Returns:
            The response ID from the most recent Responses API call,
            or None if no calls have been made or auto_chain is disabled.
        """
        return self._last_response_id

    def reset_chain(self) -> None:
        """Reset the auto-chain state to start a new conversation.

        Clears the stored response ID so the next call starts fresh
        without linking to previous responses.
        """
        self._last_response_id = None

    @property
    def last_reasoning_items(self) -> list[Any] | None:
        """Get the last reasoning items from auto-chain reasoning.

        Returns:
            The reasoning items from the most recent Responses API call
            containing encrypted content, or None if no calls have been made
            or auto_chain_reasoning is disabled.
        """
        return self._last_reasoning_items

    def reset_reasoning_chain(self) -> None:
        """Reset the reasoning chain state to start fresh.

        Clears the stored reasoning items so the next call starts without
        preserving previous chain-of-thought context. Useful when starting
        a new reasoning task that shouldn't reference previous reasoning.
        """
        self._last_reasoning_items = None

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
            "base_url": self.base_url
            or self.api_base
            or os.getenv("OPENAI_BASE_URL")
            or None,
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
        messages: str | list[LLMMessage],
        tools: list[dict[str, BaseTool]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Task | None = None,
        from_agent: Agent | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        """Call OpenAI API (Chat Completions or Responses based on api setting).

        Args:
            messages: Input messages for the completion.
            tools: List of tool/function definitions.
            callbacks: Callback functions (not used in native implementation).
            available_functions: Available functions for tool calling.
            from_task: Task that initiated the call.
            from_agent: Agent that initiated the call.
            response_model: Response model for structured output.

        Returns:
            Completion response or tool call result.
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

            if not self._invoke_before_llm_call_hooks(formatted_messages, from_agent):
                raise ValueError("LLM call blocked by before_llm_call hook")

            if self.api == "responses":
                return self._call_responses(
                    messages=formatted_messages,
                    tools=tools,
                    available_functions=available_functions,
                    from_task=from_task,
                    from_agent=from_agent,
                    response_model=response_model,
                )

            return self._call_completions(
                messages=formatted_messages,
                tools=tools,
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

    def _call_completions(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, BaseTool]] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Task | None = None,
        from_agent: Agent | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        """Call OpenAI Chat Completions API."""
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
        """Async call to OpenAI API (Chat Completions or Responses).

        Args:
            messages: Input messages for the completion.
            tools: List of tool/function definitions.
            callbacks: Callback functions (not used in native implementation).
            available_functions: Available functions for tool calling.
            from_task: Task that initiated the call.
            from_agent: Agent that initiated the call.
            response_model: Response model for structured output.

        Returns:
            Completion response or tool call result.
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

            if self.api == "responses":
                return await self._acall_responses(
                    messages=formatted_messages,
                    tools=tools,
                    available_functions=available_functions,
                    from_task=from_task,
                    from_agent=from_agent,
                    response_model=response_model,
                )

            return await self._acall_completions(
                messages=formatted_messages,
                tools=tools,
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

    async def _acall_completions(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, BaseTool]] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Task | None = None,
        from_agent: Agent | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        """Async call to OpenAI Chat Completions API."""
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

    def _call_responses(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, BaseTool]] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Task | None = None,
        from_agent: Agent | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        """Call OpenAI Responses API."""
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
        tools: list[dict[str, BaseTool]] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Task | None = None,
        from_agent: Agent | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        """Async call to OpenAI Responses API."""
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
        tools: list[dict[str, BaseTool]] | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> dict[str, Any]:
        """Prepare parameters for OpenAI Responses API.

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
                # System messages should always have string content
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
        self, tools: list[dict[str, BaseTool]]
    ) -> list[dict[str, Any]]:
        """Convert CrewAI tools to Responses API format.

        Responses API uses internally-tagged format (flat structure):
        {
            "type": "function",
            "name": "get_weather",
            "description": "...",
            "parameters": {...}
        }

        Unlike Chat Completions which uses externally-tagged:
        {
            "type": "function",
            "function": {"name": "...", "description": "...", "parameters": {...}}
        }
        """
        from crewai.llms.providers.utils.common import safe_tool_conversion

        responses_tools = []

        for tool in tools:
            name, description, parameters = safe_tool_conversion(tool, "OpenAI")

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
    ) -> str | ResponsesAPIResult | Any:
        """Handle non-streaming Responses API call."""
        try:
            response: Response = self.client.responses.create(**params)

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
            if is_context_length_exceeded(e):
                logging.error(f"Context window exceeded: {e}")
                raise LLMContextLengthExceededError(str(e)) from e

            error_msg = f"OpenAI Responses API call failed: {e!s}"
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
    ) -> str | ResponsesAPIResult | Any:
        """Handle async non-streaming Responses API call."""
        try:
            response: Response = await self.async_client.responses.create(**params)

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
            if is_context_length_exceeded(e):
                logging.error(f"Context window exceeded: {e}")
                raise LLMContextLengthExceededError(str(e)) from e

            error_msg = f"OpenAI Responses API call failed: {e!s}"
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
    ) -> str | ResponsesAPIResult | Any:
        """Handle streaming Responses API call."""
        full_response = ""
        function_calls: list[dict[str, Any]] = []
        final_response: Response | None = None

        stream = self.client.responses.create(**params)
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
                # Track response ID for auto-chaining
                if self.auto_chain and event.response and event.response.id:
                    self._last_response_id = event.response.id
                # Track reasoning items for ZDR auto-chaining
                if self.auto_chain_reasoning and event.response:
                    reasoning_items = self._extract_reasoning_items(event.response)
                    if reasoning_items:
                        self._last_reasoning_items = reasoning_items
                if event.response and event.response.usage:
                    usage = {
                        "prompt_tokens": event.response.usage.input_tokens,
                        "completion_tokens": event.response.usage.output_tokens,
                        "total_tokens": event.response.usage.total_tokens,
                    }
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
    ) -> str | ResponsesAPIResult | Any:
        """Handle async streaming Responses API call."""
        full_response = ""
        function_calls: list[dict[str, Any]] = []
        final_response: Response | None = None

        stream = await self.async_client.responses.create(**params)
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
                # Track response ID for auto-chaining
                if self.auto_chain and event.response and event.response.id:
                    self._last_response_id = event.response.id
                # Track reasoning items for ZDR auto-chaining
                if self.auto_chain_reasoning and event.response:
                    reasoning_items = self._extract_reasoning_items(event.response)
                    if reasoning_items:
                        self._last_reasoning_items = reasoning_items
                if event.response and event.response.usage:
                    usage = {
                        "prompt_tokens": event.response.usage.input_tokens,
                        "completion_tokens": event.response.usage.output_tokens,
                        "total_tokens": event.response.usage.total_tokens,
                    }
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

    def _extract_function_calls_from_response(
        self, response: Response
    ) -> list[dict[str, Any]]:
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

    def _extract_responses_token_usage(self, response: Response) -> dict[str, Any]:
        """Extract token usage from Responses API response."""
        if response.usage:
            return {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        return {"total_tokens": 0}

    def _extract_builtin_tool_outputs(self, response: Response) -> ResponsesAPIResult:
        """Extract and parse all built-in tool outputs from Responses API.

        Parses web_search, file_search, code_interpreter, computer_use,
        and reasoning outputs into structured types.

        Args:
            response: The OpenAI Response object.

        Returns:
            ResponsesAPIResult containing parsed outputs.
        """
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

    def _extract_reasoning_items(self, response: Response) -> list[Any]:
        """Extract reasoning items with encrypted content from response.

        Used for ZDR (Zero Data Retention) compliance to capture encrypted
        reasoning tokens that can be passed back in subsequent requests.

        Args:
            response: The OpenAI Response object.

        Returns:
            List of reasoning items from the response output that have
            encrypted_content, suitable for passing back in future requests.
        """
        return [item for item in response.output if item.type == "reasoning"]

    def _prepare_completion_params(
        self, messages: list[LLMMessage], tools: list[dict[str, BaseTool]] | None = None
    ) -> dict[str, Any]:
        """Prepare parameters for OpenAI chat completion."""
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

        # Handle o1 model specific parameters
        if self.is_o1_model and self.reasoning_effort:
            params["reasoning_effort"] = self.reasoning_effort

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
                    "strict": True,
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

                usage = self._extract_openai_token_usage(parsed_response)
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

            usage = self._extract_openai_token_usage(response)

            self._track_token_usage_internal(usage)

            choice: Choice = response.choices[0]
            message = choice.message

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

            # If there are tool_calls and available_functions, execute the tools
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

            if usage.get("total_tokens", 0) > 0:
                logging.info(f"OpenAI API usage: {usage}")

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
        response_model: type[BaseModel] | None = None,
    ) -> str | BaseModel:
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
                    response_id_stream = chunk.id if hasattr(chunk, "id") else None

                    if chunk.type == "content.delta":
                        delta_content = chunk.delta
                        if delta_content:
                            self._emit_stream_chunk_event(
                                chunk=delta_content,
                                from_task=from_task,
                                from_agent=from_agent,
                                response_id=response_id_stream,
                            )

                final_completion = stream.get_final_completion()
                if final_completion:
                    usage = self._extract_openai_token_usage(final_completion)
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

            logging.error("Failed to get parsed result from stream")
            return ""

        completion_stream: Stream[ChatCompletionChunk] = (
            self.client.chat.completions.create(**params)
        )

        usage_data = {"total_tokens": 0}

        for completion_chunk in completion_stream:
            response_id_stream = (
                completion_chunk.id if hasattr(completion_chunk, "id") else None
            )

            if hasattr(completion_chunk, "usage") and completion_chunk.usage:
                usage_data = self._extract_openai_token_usage(completion_chunk)
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

                usage = self._extract_openai_token_usage(parsed_response)
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

            usage = self._extract_openai_token_usage(response)

            self._track_token_usage_internal(usage)

            choice: Choice = response.choices[0]
            message = choice.message

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

            # If there are tool_calls and available_functions, execute the tools
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

    async def _ahandle_streaming_completion(
        self,
        params: dict[str, Any],
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | BaseModel:
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
                response_id_stream = chunk.id if hasattr(chunk, "id") else None

                if hasattr(chunk, "usage") and chunk.usage:
                    usage_data = self._extract_openai_token_usage(chunk)
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
                        response_id=response_id_stream,
                    )

            self._track_token_usage_internal(usage_data)

            try:
                parsed_object = response_model.model_validate_json(accumulated_content)

                self._emit_call_completed_event(
                    response=parsed_object.model_dump_json(),
                    call_type=LLMCallType.LLM_CALL,
                    from_task=from_task,
                    from_agent=from_agent,
                    messages=params["messages"],
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
                )
                return accumulated_content

        stream: AsyncIterator[
            ChatCompletionChunk
        ] = await self.async_client.chat.completions.create(**params)

        usage_data = {"total_tokens": 0}

        async for chunk in stream:
            response_id_stream = chunk.id if hasattr(chunk, "id") else None

            if hasattr(chunk, "usage") and chunk.usage:
                usage_data = self._extract_openai_token_usage(chunk)
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
            "gpt-5": 1047576,
            "gpt-5-mini": 1047576,
            "gpt-5-nano": 1047576,
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

    def _extract_openai_token_usage(
        self, response: ChatCompletion | ChatCompletionChunk
    ) -> dict[str, Any]:
        """Extract token usage from OpenAI ChatCompletion or ChatCompletionChunk response."""
        if hasattr(response, "usage") and response.usage:
            usage = response.usage
            return {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
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

    def supports_multimodal(self) -> bool:
        """Check if the model supports multimodal inputs.

        OpenAI vision-enabled models include GPT-4o, GPT-4.1, GPT-5, and o-series.

        Returns:
            True if the model supports images.
        """
        vision_models = (
            "gpt-4o",
            "gpt-4.1",
            "gpt-4-turbo",
            "gpt-4-vision",
            "gpt-5",
            "o1",
            "o3",
            "o4",
        )
        return any(self.model.lower().startswith(m) for m in vision_models)

    def get_file_uploader(self) -> Any:
        """Get an OpenAI file uploader using this LLM's clients.

        Returns:
            OpenAIFileUploader instance with pre-configured sync and async clients.
        """
        try:
            from crewai_files.uploaders.openai import OpenAIFileUploader

            return OpenAIFileUploader(
                client=self.client,
                async_client=self.async_client,
            )
        except ImportError:
            return None
