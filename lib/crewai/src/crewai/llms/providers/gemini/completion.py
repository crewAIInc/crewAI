import logging
import os
from typing import Any

from crewai.events.types.llm_events import LLMCallType
from crewai.llms.base_llm import BaseLLM
from crewai.utilities.agent_utils import is_context_length_exceeded
from crewai.utilities.exceptions.context_window_exceeding_exception import (
    LLMContextLengthExceededExceptionError,
)

try:
    from google import genai  # type: ignore
    from google.genai import types  # type: ignore
    from google.genai.errors import APIError  # type: ignore
except ImportError:
    raise ImportError(
        "Google Gen AI native provider not available, to install: `uv add google-genai`"
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
        **kwargs,
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
            **kwargs: Additional parameters
        """
        super().__init__(
            model=model, temperature=temperature, stop=stop_sequences or [], **kwargs
        )

        # Get API configuration
        self.api_key = (
            api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        )
        self.project = project or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.location = location or os.getenv("GOOGLE_CLOUD_LOCATION") or "us-central1"

        # Initialize client based on available configuration
        if self.project:
            # Use Vertex AI
            self.client = genai.Client(
                vertexai=True,
                project=self.project,
                location=self.location,
            )
        elif self.api_key:
            # Use Gemini Developer API
            self.client = genai.Client(api_key=self.api_key)
        else:
            raise ValueError(
                "Either GOOGLE_API_KEY/GEMINI_API_KEY (for Gemini API) or "
                "GOOGLE_CLOUD_PROJECT (for Vertex AI) must be set"
            )

        # Store completion parameters
        self.top_p = top_p
        self.top_k = top_k
        self.max_output_tokens = max_output_tokens
        self.stream = stream
        self.safety_settings = safety_settings or {}
        self.stop_sequences = stop_sequences or []

        # Model-specific settings
        self.is_gemini_2 = "gemini-2" in model.lower()
        self.is_gemini_1_5 = "gemini-1.5" in model.lower()
        self.supports_tools = self.is_gemini_1_5 or self.is_gemini_2

    def call(
        self,
        messages: str | list[dict[str, str]],
        tools: list[dict] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
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

            config = self._prepare_generation_config(system_instruction, tools)

            if self.stream:
                return self._handle_streaming_completion(
                    formatted_content,
                    config,
                    available_functions,
                    from_task,
                    from_agent,
                )

            return self._handle_completion(
                formatted_content,
                system_instruction,
                config,
                available_functions,
                from_task,
                from_agent,
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
        tools: list[dict] | None = None,
    ) -> types.GenerateContentConfig:
        """Prepare generation config for Google Gemini API.

        Args:
            system_instruction: System instruction for the model
            tools: Tool definitions

        Returns:
            GenerateContentConfig object for Gemini API
        """
        self.tools = tools
        config_params = {}

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

        # Handle tools for supported models
        if tools and self.supports_tools:
            config_params["tools"] = self._convert_tools_for_interference(tools)

        if self.safety_settings:
            config_params["safety_settings"] = self.safety_settings

        return types.GenerateContentConfig(**config_params)

    def _convert_tools_for_interference(self, tools: list[dict]) -> list[types.Tool]:
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
            if parameters and isinstance(parameters, dict):
                function_declaration.parameters = parameters

            gemini_tool = types.Tool(function_declarations=[function_declaration])
            gemini_tools.append(gemini_tool)

        return gemini_tools

    def _format_messages_for_gemini(
        self, messages: str | list[dict[str, str]]
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
        system_instruction = None

        for message in base_formatted:
            role = message.get("role")
            content = message.get("content", "")

            if role == "system":
                # Extract system instruction - Gemini handles it separately
                if system_instruction:
                    system_instruction += f"\n\n{content}"
                else:
                    system_instruction = content
            else:
                # Convert role for Gemini (assistant -> model)
                gemini_role = "model" if role == "assistant" else "user"

                # Create Content object
                gemini_content = types.Content(
                    role=gemini_role, parts=[types.Part.from_text(text=content)]
                )
                contents.append(gemini_content)

        return contents, system_instruction

    def _handle_completion(
        self,
        contents: list[types.Content],
        system_instruction: str | None,
        config: types.GenerateContentConfig,
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
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
                raise LLMContextLengthExceededExceptionError(str(e)) from e
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

    def _handle_streaming_completion(
        self,
        contents: list[types.Content],
        config: types.GenerateContentConfig,
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
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

        # Handle completed function calls
        if function_calls and available_functions:
            for call_data in function_calls.values():
                function_name = call_data["name"]
                function_args = call_data["args"]

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
        return self._supports_stop_words_implementation()

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

    def _convert_contents_to_dict(
        self, contents: list[types.Content]
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
