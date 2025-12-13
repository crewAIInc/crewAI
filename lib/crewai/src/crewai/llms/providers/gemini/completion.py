from __future__ import annotations

import logging
import os
import re
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import BaseModel

from crewai.events.types.llm_events import LLMCallType
from crewai.llms.base_llm import BaseLLM
from crewai.utilities.agent_utils import is_context_length_exceeded
from crewai.utilities.exceptions.context_window_exceeding_exception import (
    LLMContextLengthExceededError,
)
from crewai.utilities.types import LLMMessage


if TYPE_CHECKING:
    from crewai.llms.hooks.base import BaseInterceptor


try:
    from google import genai
    from google.genai import types
    from google.genai.errors import APIError
    from google.genai.types import GenerateContentResponse, Schema
except ImportError:
    raise ImportError(
        'Google Gen AI native provider not available, to install: uv add "crewai[google-genai]"'
    ) from None


class GeminiCompletion(BaseLLM):
    """Google Gemini native completion implementation.

    This class provides direct integration with the Google Gen AI Python SDK,
    offering native function calling, streaming support, and proper Gemini formatting.
    """

    def __init__(
        self,
        model: str = "gemini-2.0-flash-001",
        api_key: str | None = None,
        project: str | None = None,
        location: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        max_output_tokens: int | None = None,
        stop_sequences: list[str] | None = None,
        stream: bool = False,
        safety_settings: dict[str, Any] | None = None,
        client_params: dict[str, Any] | None = None,
        interceptor: BaseInterceptor[Any, Any] | None = None,
        **kwargs: Any,
    ):
        """Initialize Google Gemini chat completion client.

        Args:
            model: Gemini model name (e.g., 'gemini-2.0-flash-001', 'gemini-1.5-pro')
            api_key: Google API key (defaults to GOOGLE_API_KEY or GEMINI_API_KEY env var)
            project: Google Cloud project ID (for Vertex AI)
            location: Google Cloud location (for Vertex AI, defaults to 'us-central1')
            temperature: Sampling temperature (0-2)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            max_output_tokens: Maximum tokens in response
            stop_sequences: Stop sequences
            stream: Enable streaming responses
            safety_settings: Safety filter settings
            client_params: Additional parameters to pass to the Google Gen AI Client constructor.
                          Supports parameters like http_options, credentials, debug_config, etc.
            interceptor: HTTP interceptor (not yet supported for Gemini).
            **kwargs: Additional parameters
        """
        if interceptor is not None:
            raise NotImplementedError(
                "HTTP interceptors are not yet supported for Google Gemini provider. "
                "Interceptors are currently supported for OpenAI and Anthropic providers only."
            )

        super().__init__(
            model=model, temperature=temperature, stop=stop_sequences or [], **kwargs
        )

        # Store client params for later use
        self.client_params = client_params or {}

        # Get API configuration with environment variable fallbacks
        self.api_key = (
            api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        )
        self.project = project or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.location = location or os.getenv("GOOGLE_CLOUD_LOCATION") or "us-central1"

        use_vertexai = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() == "true"

        self.client = self._initialize_client(use_vertexai)

        # Store completion parameters
        self.top_p = top_p
        self.top_k = top_k
        self.max_output_tokens = max_output_tokens
        self.stream = stream
        self.safety_settings = safety_settings or {}
        self.stop_sequences = stop_sequences or []
        self.tools: list[dict[str, Any]] | None = None

        # Model-specific settings
        version_match = re.search(r"gemini-(\d+(?:\.\d+)?)", model.lower())
        self.supports_tools = bool(
            version_match and float(version_match.group(1)) >= 1.5
        )

    @property
    def stop(self) -> list[str]:
        """Get stop sequences sent to the API."""
        return self.stop_sequences

    @stop.setter
    def stop(self, value: list[str] | str | None) -> None:
        """Set stop sequences.

        Synchronizes stop_sequences to ensure values set by CrewAgentExecutor
        are properly sent to the Gemini API.

        Args:
            value: Stop sequences as a list, single string, or None
        """
        if value is None:
            self.stop_sequences = []
        elif isinstance(value, str):
            self.stop_sequences = [value]
        elif isinstance(value, list):
            self.stop_sequences = value
        else:
            self.stop_sequences = []

    def _initialize_client(self, use_vertexai: bool = False) -> genai.Client:
        """Initialize the Google Gen AI client with proper parameter handling.

        Args:
            use_vertexai: Whether to use Vertex AI (from environment variable)

        Returns:
            Initialized Google Gen AI Client
        """
        client_params = {}

        if self.client_params:
            client_params.update(self.client_params)

        if use_vertexai or self.project:
            client_params.update(
                {
                    "vertexai": True,
                    "project": self.project,
                    "location": self.location,
                }
            )

            client_params.pop("api_key", None)

        elif self.api_key:
            client_params["api_key"] = self.api_key

            client_params.pop("vertexai", None)
            client_params.pop("project", None)
            client_params.pop("location", None)

        else:
            try:
                return genai.Client(**client_params)
            except Exception as e:
                raise ValueError(
                    "Either GOOGLE_API_KEY/GEMINI_API_KEY (for Gemini API) or "
                    "GOOGLE_CLOUD_PROJECT (for Vertex AI) must be set"
                ) from e

        return genai.Client(**client_params)

    def _get_client_params(self) -> dict[str, Any]:
        """Get client parameters for compatibility with base class.

        Note: This method is kept for compatibility but the Google Gen AI SDK
        uses a different initialization pattern via the Client constructor.
        """
        params = {}

        if (
            hasattr(self, "client")
            and hasattr(self.client, "vertexai")
            and self.client.vertexai
        ):
            # Vertex AI configuration
            params.update(
                {
                    "vertexai": True,
                    "project": self.project,
                    "location": self.location,
                }
            )
        elif self.api_key:
            params["api_key"] = self.api_key

        if self.client_params:
            params.update(self.client_params)

        return params

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
        """Call Google Gemini generate content API.

        Args:
            messages: Input messages for the chat completion
            tools: List of tool/function definitions
            callbacks: Callback functions (not used as token counts are handled by the response)
            available_functions: Available functions for tool calling
            from_task: Task that initiated the call
            from_agent: Agent that initiated the call
            response_model: Response model to use.

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
            self.tools = tools

            formatted_content, system_instruction = self._format_messages_for_gemini(
                messages
            )

            messages_for_hooks = self._convert_contents_to_dict(formatted_content)

            if not self._invoke_before_llm_call_hooks(messages_for_hooks, from_agent):
                raise ValueError("LLM call blocked by before_llm_call hook")

            config = self._prepare_generation_config(
                system_instruction, tools, response_model
            )

            if self.stream:
                return self._handle_streaming_completion(
                    formatted_content,
                    config,
                    available_functions,
                    from_task,
                    from_agent,
                    response_model,
                )

            return self._handle_completion(
                formatted_content,
                config,
                available_functions,
                from_task,
                from_agent,
                response_model,
            )

        except APIError as e:
            error_msg = f"Google Gemini API error: {e.code} - {e.message}"
            logging.error(error_msg)
            self._emit_call_failed_event(
                error=error_msg, from_task=from_task, from_agent=from_agent
            )
            raise
        except Exception as e:
            error_msg = f"Google Gemini API call failed: {e!s}"
            logging.error(error_msg)
            self._emit_call_failed_event(
                error=error_msg, from_task=from_task, from_agent=from_agent
            )
            raise

    async def acall(
        self,
        messages: str | list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        """Async call to Google Gemini generate content API.

        Args:
            messages: Input messages for the chat completion
            tools: List of tool/function definitions
            callbacks: Callback functions (not used as token counts are handled by the response)
            available_functions: Available functions for tool calling
            from_task: Task that initiated the call
            from_agent: Agent that initiated the call
            response_model: Response model to use.

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
            self.tools = tools

            formatted_content, system_instruction = self._format_messages_for_gemini(
                messages
            )

            config = self._prepare_generation_config(
                system_instruction, tools, response_model
            )

            if self.stream:
                return await self._ahandle_streaming_completion(
                    formatted_content,
                    config,
                    available_functions,
                    from_task,
                    from_agent,
                    response_model,
                )

            return await self._ahandle_completion(
                formatted_content,
                config,
                available_functions,
                from_task,
                from_agent,
                response_model,
            )

        except APIError as e:
            error_msg = f"Google Gemini API error: {e.code} - {e.message}"
            logging.error(error_msg)
            self._emit_call_failed_event(
                error=error_msg, from_task=from_task, from_agent=from_agent
            )
            raise
        except Exception as e:
            error_msg = f"Google Gemini API call failed: {e!s}"
            logging.error(error_msg)
            self._emit_call_failed_event(
                error=error_msg, from_task=from_task, from_agent=from_agent
            )
            raise

    def _prepare_generation_config(
        self,
        system_instruction: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> types.GenerateContentConfig:
        """Prepare generation config for Google Gemini API.

        Args:
            system_instruction: System instruction for the model
            tools: Tool definitions
            response_model: Pydantic model for structured output

        Returns:
            GenerateContentConfig object for Gemini API
        """
        self.tools = tools
        config_params: dict[str, Any] = {}

        # Add system instruction if present
        if system_instruction:
            # Convert system instruction to Content format
            system_content = types.Content(
                role="user", parts=[types.Part.from_text(text=system_instruction)]
            )
            config_params["system_instruction"] = system_content

        # Add generation config parameters
        if self.temperature is not None:
            config_params["temperature"] = self.temperature
        if self.top_p is not None:
            config_params["top_p"] = self.top_p
        if self.top_k is not None:
            config_params["top_k"] = self.top_k
        if self.max_output_tokens is not None:
            config_params["max_output_tokens"] = self.max_output_tokens
        if self.stop_sequences:
            config_params["stop_sequences"] = self.stop_sequences

        if response_model:
            config_params["response_mime_type"] = "application/json"
            config_params["response_schema"] = response_model.model_json_schema()

        # Handle tools for supported models
        if tools and self.supports_tools:
            config_params["tools"] = self._convert_tools_for_interference(tools)

        if self.safety_settings:
            config_params["safety_settings"] = self.safety_settings

        return types.GenerateContentConfig(**config_params)

    def _convert_tools_for_interference(  # type: ignore[override]
        self, tools: list[dict[str, Any]]
    ) -> list[types.Tool]:
        """Convert CrewAI tool format to Gemini function declaration format."""
        gemini_tools = []

        from crewai.llms.providers.utils.common import safe_tool_conversion

        for tool in tools:
            name, description, parameters = safe_tool_conversion(tool, "Gemini")

            function_declaration = types.FunctionDeclaration(
                name=name,
                description=description,
            )

            # Add parameters if present - ensure parameters is a dict
            if parameters and isinstance(parameters, Schema):
                function_declaration.parameters = parameters

            gemini_tool = types.Tool(function_declarations=[function_declaration])
            gemini_tools.append(gemini_tool)

        return gemini_tools

    def _format_messages_for_gemini(
        self, messages: str | list[LLMMessage]
    ) -> tuple[list[types.Content], str | None]:
        """Format messages for Gemini API.

        Gemini has specific requirements:
        - System messages are separate system_instruction
        - Content is organized as Content objects with Parts
        - Roles are 'user' and 'model' (not 'assistant')

        Args:
            messages: Input messages

        Returns:
            Tuple of (formatted_contents, system_instruction)
        """
        # Use base class formatting first
        base_formatted = super()._format_messages(messages)

        contents: list[types.Content] = []
        system_instruction: str | None = None

        for message in base_formatted:
            role = message["role"]
            content = message["content"]

            # Convert content to string if it's a list
            if isinstance(content, list):
                text_content = " ".join(
                    str(item.get("text", "")) if isinstance(item, dict) else str(item)
                    for item in content
                )
            else:
                text_content = str(content) if content else ""

            if role == "system":
                # Extract system instruction - Gemini handles it separately
                if system_instruction:
                    system_instruction += f"\n\n{text_content}"
                else:
                    system_instruction = text_content
            else:
                # Convert role for Gemini (assistant -> model)
                gemini_role = "model" if role == "assistant" else "user"

                # Create Content object
                gemini_content = types.Content(
                    role=gemini_role, parts=[types.Part.from_text(text=text_content)]
                )
                contents.append(gemini_content)

        return contents, system_instruction

    def _validate_and_emit_structured_output(
        self,
        content: str,
        response_model: type[BaseModel],
        messages_for_event: list[LLMMessage],
        from_task: Any | None = None,
        from_agent: Any | None = None,
    ) -> str:
        """Validate content against response model and emit completion event.

        Args:
            content: Response content to validate
            response_model: Pydantic model for validation
            messages_for_event: Messages to include in event
            from_task: Task that initiated the call
            from_agent: Agent that initiated the call

        Returns:
            Validated and serialized JSON string

        Raises:
            ValueError: If validation fails
        """
        try:
            structured_data = response_model.model_validate_json(content)
            structured_json = structured_data.model_dump_json()

            self._emit_call_completed_event(
                response=structured_json,
                call_type=LLMCallType.LLM_CALL,
                from_task=from_task,
                from_agent=from_agent,
                messages=messages_for_event,
            )

            return structured_json
        except Exception as e:
            error_msg = f"Failed to validate structured output with model {response_model.__name__}: {e}"
            logging.error(error_msg)
            raise ValueError(error_msg) from e

    def _finalize_completion_response(
        self,
        content: str,
        contents: list[types.Content],
        response_model: type[BaseModel] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
    ) -> str:
        """Finalize completion response with validation and event emission.

        Args:
            content: The response content
            contents: Original contents for event conversion
            response_model: Pydantic model for structured output validation
            from_task: Task that initiated the call
            from_agent: Agent that initiated the call

        Returns:
            Final response content after processing
        """
        messages_for_event = self._convert_contents_to_dict(contents)

        # Handle structured output validation
        if response_model:
            return self._validate_and_emit_structured_output(
                content=content,
                response_model=response_model,
                messages_for_event=messages_for_event,
                from_task=from_task,
                from_agent=from_agent,
            )

        self._emit_call_completed_event(
            response=content,
            call_type=LLMCallType.LLM_CALL,
            from_task=from_task,
            from_agent=from_agent,
            messages=messages_for_event,
        )

        return self._invoke_after_llm_call_hooks(
            messages_for_event, content, from_agent
        )

    def _process_response_with_tools(
        self,
        response: GenerateContentResponse,
        contents: list[types.Content],
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        """Process response, execute function calls, and finalize completion.

        Args:
            response: The completion response
            contents: Original contents for event conversion
            available_functions: Available functions for function calling
            from_task: Task that initiated the call
            from_agent: Agent that initiated the call
            response_model: Pydantic model for structured output validation

        Returns:
            Final response content or function call result
        """
        if response.candidates and (self.tools or available_functions):
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if hasattr(part, "function_call") and part.function_call:
                        function_name = part.function_call.name
                        if function_name is None:
                            continue
                        function_args = (
                            dict(part.function_call.args)
                            if part.function_call.args
                            else {}
                        )

                        result = self._handle_tool_execution(
                            function_name=function_name,
                            function_args=function_args,
                            available_functions=available_functions or {},
                            from_task=from_task,
                            from_agent=from_agent,
                        )

                        if result is not None:
                            return result

        content = response.text or ""
        content = self._apply_stop_words(content)

        return self._finalize_completion_response(
            content=content,
            contents=contents,
            response_model=response_model,
            from_task=from_task,
            from_agent=from_agent,
        )

    def _process_stream_chunk(
        self,
        chunk: GenerateContentResponse,
        full_response: str,
        function_calls: dict[str, dict[str, Any]],
        usage_data: dict[str, int],
        from_task: Any | None = None,
        from_agent: Any | None = None,
    ) -> tuple[str, dict[str, dict[str, Any]], dict[str, int]]:
        """Process a single streaming chunk.

        Args:
            chunk: The streaming chunk response
            full_response: Accumulated response text
            function_calls: Accumulated function calls
            usage_data: Accumulated usage data
            from_task: Task that initiated the call
            from_agent: Agent that initiated the call

        Returns:
            Tuple of (updated full_response, updated function_calls, updated usage_data)
        """
        if chunk.usage_metadata:
            usage_data = self._extract_token_usage(chunk)

        if chunk.text:
            full_response += chunk.text
            self._emit_stream_chunk_event(
                chunk=chunk.text,
                from_task=from_task,
                from_agent=from_agent,
            )

        if chunk.candidates:
            candidate = chunk.candidates[0]
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if hasattr(part, "function_call") and part.function_call:
                        call_id = part.function_call.name or "default"
                        if call_id not in function_calls:
                            function_calls[call_id] = {
                                "name": part.function_call.name,
                                "args": dict(part.function_call.args)
                                if part.function_call.args
                                else {},
                            }

        return full_response, function_calls, usage_data

    def _finalize_streaming_response(
        self,
        full_response: str,
        function_calls: dict[str, dict[str, Any]],
        usage_data: dict[str, int],
        contents: list[types.Content],
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str:
        """Finalize streaming response with usage tracking, function execution, and events.

        Args:
            full_response: The complete streamed response content
            function_calls: Dictionary of function calls accumulated during streaming
            usage_data: Token usage data from the stream
            contents: Original contents for event conversion
            available_functions: Available functions for function calling
            from_task: Task that initiated the call
            from_agent: Agent that initiated the call
            response_model: Pydantic model for structured output validation

        Returns:
            Final response content after processing
        """
        self._track_token_usage_internal(usage_data)

        # Handle completed function calls
        if function_calls and available_functions:
            for call_data in function_calls.values():
                function_name = call_data["name"]
                function_args = call_data["args"]

                # Skip if function_name is None
                if not isinstance(function_name, str):
                    continue

                # Ensure function_args is a dict
                if not isinstance(function_args, dict):
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

        return self._finalize_completion_response(
            content=full_response,
            contents=contents,
            response_model=response_model,
            from_task=from_task,
            from_agent=from_agent,
        )

    def _handle_completion(
        self,
        contents: list[types.Content],
        config: types.GenerateContentConfig,
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        """Handle non-streaming content generation."""
        try:
            # The API accepts list[Content] but mypy is overly strict about variance
            contents_for_api: Any = contents
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents_for_api,
                config=config,
            )

            usage = self._extract_token_usage(response)
        except Exception as e:
            if is_context_length_exceeded(e):
                logging.error(f"Context window exceeded: {e}")
                raise LLMContextLengthExceededError(str(e)) from e
            raise e from e

        self._track_token_usage_internal(usage)

        return self._process_response_with_tools(
            response=response,
            contents=contents,
            available_functions=available_functions,
            from_task=from_task,
            from_agent=from_agent,
            response_model=response_model,
        )

    def _handle_streaming_completion(
        self,
        contents: list[types.Content],
        config: types.GenerateContentConfig,
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str:
        """Handle streaming content generation."""
        full_response = ""
        function_calls: dict[str, dict[str, Any]] = {}
        usage_data = {"total_tokens": 0}

        # The API accepts list[Content] but mypy is overly strict about variance
        contents_for_api: Any = contents
        for chunk in self.client.models.generate_content_stream(
            model=self.model,
            contents=contents_for_api,
            config=config,
        ):
            full_response, function_calls, usage_data = self._process_stream_chunk(
                chunk=chunk,
                full_response=full_response,
                function_calls=function_calls,
                usage_data=usage_data,
                from_task=from_task,
                from_agent=from_agent,
            )

        return self._finalize_streaming_response(
            full_response=full_response,
            function_calls=function_calls,
            usage_data=usage_data,
            contents=contents,
            available_functions=available_functions,
            from_task=from_task,
            from_agent=from_agent,
            response_model=response_model,
        )

    async def _ahandle_completion(
        self,
        contents: list[types.Content],
        config: types.GenerateContentConfig,
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        """Handle async non-streaming content generation."""
        try:
            # The API accepts list[Content] but mypy is overly strict about variance
            contents_for_api: Any = contents
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=contents_for_api,
                config=config,
            )

            usage = self._extract_token_usage(response)
        except Exception as e:
            if is_context_length_exceeded(e):
                logging.error(f"Context window exceeded: {e}")
                raise LLMContextLengthExceededError(str(e)) from e
            raise e from e

        self._track_token_usage_internal(usage)

        return self._process_response_with_tools(
            response=response,
            contents=contents,
            available_functions=available_functions,
            from_task=from_task,
            from_agent=from_agent,
            response_model=response_model,
        )

    async def _ahandle_streaming_completion(
        self,
        contents: list[types.Content],
        config: types.GenerateContentConfig,
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str:
        """Handle async streaming content generation."""
        full_response = ""
        function_calls: dict[str, dict[str, Any]] = {}
        usage_data = {"total_tokens": 0}

        # The API accepts list[Content] but mypy is overly strict about variance
        contents_for_api: Any = contents
        stream = await self.client.aio.models.generate_content_stream(
            model=self.model,
            contents=contents_for_api,
            config=config,
        )
        async for chunk in stream:
            full_response, function_calls, usage_data = self._process_stream_chunk(
                chunk=chunk,
                full_response=full_response,
                function_calls=function_calls,
                usage_data=usage_data,
                from_task=from_task,
                from_agent=from_agent,
            )

        return self._finalize_streaming_response(
            full_response=full_response,
            function_calls=function_calls,
            usage_data=usage_data,
            contents=contents,
            available_functions=available_functions,
            from_task=from_task,
            from_agent=from_agent,
            response_model=response_model,
        )

    def supports_function_calling(self) -> bool:
        """Check if the model supports function calling."""
        return self.supports_tools

    def supports_stop_words(self) -> bool:
        """Check if the model supports stop words."""
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

        context_windows = {
            "gemini-3-pro-preview": 1048576,  # 1M tokens
            "gemini-2.0-flash": 1048576,  # 1M tokens
            "gemini-2.0-flash-thinking": 32768,
            "gemini-2.0-flash-lite": 1048576,
            "gemini-2.5-flash": 1048576,
            "gemini-2.5-pro": 1048576,
            "gemini-1.5-pro": 2097152,  # 2M tokens
            "gemini-1.5-flash": 1048576,
            "gemini-1.5-flash-8b": 1048576,
            "gemini-1.0-pro": 32768,
            "gemma-3-1b": 32000,
            "gemma-3-4b": 128000,
            "gemma-3-12b": 128000,
            "gemma-3-27b": 128000,
        }

        # Find the best match for the model name
        for model_prefix, size in context_windows.items():
            if self.model.startswith(model_prefix):
                return int(size * CONTEXT_WINDOW_USAGE_RATIO)

        # Default context window size for Gemini models
        return int(1048576 * CONTEXT_WINDOW_USAGE_RATIO)  # 1M tokens

    @staticmethod
    def _extract_token_usage(response: GenerateContentResponse) -> dict[str, Any]:
        """Extract token usage from Gemini response."""
        if response.usage_metadata:
            usage = response.usage_metadata
            return {
                "prompt_token_count": getattr(usage, "prompt_token_count", 0),
                "candidates_token_count": getattr(usage, "candidates_token_count", 0),
                "total_token_count": getattr(usage, "total_token_count", 0),
                "total_tokens": getattr(usage, "total_token_count", 0),
            }
        return {"total_tokens": 0}

    @staticmethod
    def _convert_contents_to_dict(
        contents: list[types.Content],
    ) -> list[LLMMessage]:
        """Convert contents to dict format."""
        result: list[LLMMessage] = []
        for content_obj in contents:
            role = content_obj.role
            if role == "model":
                role = "assistant"
            elif role is None:
                role = "user"

            parts = content_obj.parts or []
            content = " ".join(
                part.text for part in parts if hasattr(part, "text") and part.text
            )

            result.append(
                LLMMessage(
                    role=cast(Literal["user", "assistant", "system"], role),
                    content=content,
                )
            )
        return result
