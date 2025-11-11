import logging
import os
from typing import Any, ClassVar, cast

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator
from typing_extensions import Self

from crewai.events.types.llm_events import LLMCallType
from crewai.llm.base_llm import BaseLLM
from crewai.llm.core import CONTEXT_WINDOW_USAGE_RATIO, LLM_CONTEXT_WINDOW_SIZES
from crewai.llm.providers.utils.common import safe_tool_conversion
from crewai.utilities.agent_utils import is_context_length_exceeded
from crewai.utilities.exceptions.context_window_exceeding_exception import (
    LLMContextLengthExceededError,
)
from crewai.utilities.types import LLMMessage


try:
    from google import genai  # type: ignore[import-untyped]
    from google.genai import types  # type: ignore[import-untyped]
    from google.genai.errors import APIError  # type: ignore[import-untyped]
except ImportError:
    raise ImportError(
        'Google Gen AI native provider not available, to install: uv add "crewai[google-genai]"'
    ) from None


class GeminiCompletion(BaseLLM):
    """Google Gemini native completion implementation.

    This class provides direct integration with the Google Gen AI Python SDK,
    offering native function calling, streaming support, and proper Gemini formatting.

    Attributes:
        model: Gemini model name (e.g., 'gemini-2.0-flash-001', 'gemini-1.5-pro')
        project: Google Cloud project ID (for Vertex AI)
        location: Google Cloud location (for Vertex AI, defaults to 'us-central1')
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        max_output_tokens: Maximum tokens in response
        stream: Enable streaming responses
        safety_settings: Safety filter settings
        client_params: Additional parameters for Google Gen AI Client constructor
        interceptor: HTTP interceptor (not yet supported for Gemini)
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(
        ignored_types=(property,), arbitrary_types_allowed=True
    )

    project: str | None = Field(
        default=None, description="Google Cloud project ID (for Vertex AI)"
    )
    location: str = Field(
        default="us-central1",
        description="Google Cloud location (for Vertex AI, defaults to 'us-central1')",
    )
    top_p: float | None = Field(default=None, description="Nucleus sampling parameter")
    top_k: int | None = Field(default=None, description="Top-k sampling parameter")
    max_output_tokens: int | None = Field(
        default=None, description="Maximum tokens in response"
    )
    stream: bool = Field(default=False, description="Enable streaming responses")
    safety_settings: dict[str, Any] = Field(
        default_factory=dict, description="Safety filter settings"
    )
    client_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional parameters for Google Gen AI Client constructor",
    )
    client: Any = Field(
        default=None, exclude=True, description="Gemini client instance"
    )

    _is_gemini_2: bool = PrivateAttr(default=False)
    _is_gemini_1_5: bool = PrivateAttr(default=False)
    _supports_tools: bool = PrivateAttr(default=False)

    @model_validator(mode="after")
    def setup_client(self) -> Self:
        """Initialize the Gemini client and validate configuration."""
        if self.interceptor is not None:
            raise NotImplementedError(
                "HTTP interceptors are not yet supported for Google Gemini provider. "
                "Interceptors are currently supported for OpenAI and Anthropic providers only."
            )

        if self.api_key is None:
            self.api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

        if self.project is None:
            self.project = os.getenv("GOOGLE_CLOUD_PROJECT")

        if self.location == "us-central1":
            env_location = os.getenv("GOOGLE_CLOUD_LOCATION")
            if env_location:
                self.location = env_location

        use_vertexai = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() == "true"

        self.client = self._initialize_client(use_vertexai)

        self._is_gemini_2 = "gemini-2" in self.model.lower()
        self._is_gemini_1_5 = "gemini-1.5" in self.model.lower()
        self._supports_tools = self._is_gemini_1_5 or self._is_gemini_2

        return self

    @property
    def is_gemini_2(self) -> bool:
        """Check if model is Gemini 2."""
        return self._is_gemini_2

    @property
    def is_gemini_1_5(self) -> bool:
        """Check if model is Gemini 1.5."""
        return self._is_gemini_1_5

    @property
    def supports_tools(self) -> bool:
        """Check if model supports tools."""
        return self._supports_tools

    def _initialize_client(self, use_vertexai: bool = False) -> Any:
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
            callbacks: Callback functions (not used as token counts are handled by the reponse)
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
            self.tools = tools

            formatted_content, system_instruction = self._format_messages_for_gemini(
                messages
            )

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
                system_instruction,
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

    def _prepare_generation_config(  # type: ignore[no-any-unimported]
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
        config_params = {}

        if system_instruction:
            system_content = types.Content(
                role="user", parts=[types.Part.from_text(text=system_instruction)]
            )
            config_params["system_instruction"] = system_content

        if self.temperature is not None:
            config_params["temperature"] = self.temperature
        if self.top_p is not None:
            config_params["top_p"] = self.top_p
        if self.top_k is not None:
            config_params["top_k"] = self.top_k
        if self.max_output_tokens is not None:
            config_params["max_output_tokens"] = self.max_output_tokens
        if self.stop:
            config_params["stop_sequences"] = self.stop

        if response_model:
            config_params["response_mime_type"] = "application/json"
            config_params["response_schema"] = response_model.model_json_schema()

        if tools and self.supports_tools:
            config_params["tools"] = self._convert_tools_for_interference(tools)

        if self.safety_settings:
            config_params["safety_settings"] = self.safety_settings

        return types.GenerateContentConfig(**config_params)

    def _convert_tools_for_interference(  # type: ignore[no-any-unimported]
        self, tools: list[dict[str, Any]]
    ) -> list[types.Tool]:
        """Convert CrewAI tool format to Gemini function declaration format."""
        gemini_tools = []

        for tool in tools:
            name, description, parameters = safe_tool_conversion(tool, "Gemini")

            function_declaration = types.FunctionDeclaration(
                name=name,
                description=description,
            )

            if parameters and isinstance(parameters, dict):
                function_declaration.parameters = parameters

            gemini_tool = types.Tool(function_declarations=[function_declaration])
            gemini_tools.append(gemini_tool)

        return gemini_tools

    def _format_messages_for_gemini(  # type: ignore[no-any-unimported]
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

        contents = []
        system_instruction: str | None = None

        for message in base_formatted:
            role = message.get("role")
            content = message.get("content", "")

            if role == "system":
                if system_instruction:
                    system_instruction += f"\n\n{content}"
                else:
                    system_instruction = cast(str, content)
            else:
                gemini_role = "model" if role == "assistant" else "user"
                gemini_content = types.Content(
                    role=gemini_role, parts=[types.Part.from_text(text=content)]
                )
                contents.append(gemini_content)

        return contents, system_instruction

    def _handle_completion(  # type: ignore[no-any-unimported]
        self,
        contents: list[types.Content],
        system_instruction: str | None,
        config: types.GenerateContentConfig,
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        """Handle non-streaming content generation."""
        api_params = {
            "model": self.model,
            "contents": contents,
            "config": config,
        }

        try:
            response = self.client.models.generate_content(**api_params)

            usage = self._extract_token_usage(response)
        except Exception as e:
            if is_context_length_exceeded(e):
                logging.error(f"Context window exceeded: {e}")
                raise LLMContextLengthExceededError(str(e)) from e
            raise e from e

        self._track_token_usage_internal(usage)

        if response.candidates and (self.tools or available_functions):
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if hasattr(part, "function_call") and part.function_call:
                        function_name = part.function_call.name
                        function_args = (
                            dict(part.function_call.args)
                            if part.function_call.args
                            else {}
                        )

                        result = self._handle_tool_execution(
                            function_name=function_name,
                            function_args=function_args,
                            available_functions=available_functions,  # type: ignore
                            from_task=from_task,
                            from_agent=from_agent,
                        )

                        if result is not None:
                            return result

        content = response.text if hasattr(response, "text") else ""
        content = self._apply_stop_words(content)

        messages_for_event = self._convert_contents_to_dict(contents)

        self._emit_call_completed_event(
            response=content,
            call_type=LLMCallType.LLM_CALL,
            from_task=from_task,
            from_agent=from_agent,
            messages=messages_for_event,
        )

        return content

    def _handle_streaming_completion(  # type: ignore[no-any-unimported]
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
        function_calls = {}

        api_params = {
            "model": self.model,
            "contents": contents,
            "config": config,
        }

        for chunk in self.client.models.generate_content_stream(**api_params):
            if hasattr(chunk, "text") and chunk.text:
                full_response += chunk.text
                self._emit_stream_chunk_event(
                    chunk=chunk.text,
                    from_task=from_task,
                    from_agent=from_agent,
                )

            if hasattr(chunk, "candidates") and chunk.candidates:
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

        if function_calls and available_functions:
            for call_data in function_calls.values():
                function_name = call_data["name"]
                function_args = call_data["args"]

                result = self._handle_tool_execution(
                    function_name=function_name,
                    function_args=function_args,
                    available_functions=available_functions,
                    from_task=from_task,
                    from_agent=from_agent,
                )

                if result is not None:
                    return result

        messages_for_event = self._convert_contents_to_dict(contents)

        self._emit_call_completed_event(
            response=full_response,
            call_type=LLMCallType.LLM_CALL,
            from_task=from_task,
            from_agent=from_agent,
            messages=messages_for_event,
        )

        return full_response

    def supports_function_calling(self) -> bool:
        """Check if the model supports function calling."""
        return self.supports_tools

    def supports_stop_words(self) -> bool:
        """Check if the model supports stop words."""
        return True

    def get_context_window_size(self) -> int:
        """Get the context window size for the model."""

        min_context = 1024
        max_context = 2097152

        for key, value in LLM_CONTEXT_WINDOW_SIZES.items():
            if value < min_context or value > max_context:
                raise ValueError(
                    f"Context window for {key} must be between {min_context} and {max_context}"
                )

        context_windows = {
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

        for model_prefix, size in context_windows.items():
            if self.model.startswith(model_prefix):
                return int(size * CONTEXT_WINDOW_USAGE_RATIO)

        return int(1048576 * CONTEXT_WINDOW_USAGE_RATIO)

    def _extract_token_usage(self, response: dict[str, Any]) -> dict[str, Any]:
        """Extract token usage from Gemini response."""
        if hasattr(response, "usage_metadata"):
            usage = response.usage_metadata
            return {
                "prompt_token_count": getattr(usage, "prompt_token_count", 0),
                "candidates_token_count": getattr(usage, "candidates_token_count", 0),
                "total_token_count": getattr(usage, "total_token_count", 0),
                "total_tokens": getattr(usage, "total_token_count", 0),
            }
        return {"total_tokens": 0}

    def _convert_contents_to_dict(  # type: ignore[no-any-unimported]
        self,
        contents: list[types.Content],
    ) -> list[dict[str, str]]:
        """Convert contents to dict format."""
        return [
            {
                "role": "assistant"
                if content_obj.role == "model"
                else content_obj.role,
                "content": " ".join(
                    part.text
                    for part in content_obj.parts
                    if hasattr(part, "text") and part.text
                ),
            }
            for content_obj in contents
        ]
