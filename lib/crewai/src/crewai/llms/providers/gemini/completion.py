from __future__ import annotations

import base64
import json
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
from crewai.utilities.pydantic_schema_utils import generate_model_description
from crewai.utilities.types import LLMMessage


if TYPE_CHECKING:
    from crewai.llms.hooks.base import BaseInterceptor


try:
    from google import genai
    from google.genai import types
    from google.genai.errors import APIError
    from google.genai.types import GenerateContentResponse
except ImportError:
    raise ImportError(
        'Google Gen AI native provider not available, to install: uv add "crewai[google-genai]"'
    ) from None


STRUCTURED_OUTPUT_TOOL_NAME = "structured_output"


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
        use_vertexai: bool | None = None,
        response_format: type[BaseModel] | None = None,
        **kwargs: Any,
    ):
        """Initialize Google Gemini chat completion client.

        Args:
            model: Gemini model name (e.g., 'gemini-2.0-flash-001', 'gemini-1.5-pro')
            api_key: Google API key for Gemini API authentication.
                    Defaults to GOOGLE_API_KEY or GEMINI_API_KEY env var.
                    NOTE: Cannot be used with Vertex AI (project parameter). Use Gemini API instead.
            project: Google Cloud project ID for Vertex AI with ADC authentication.
                    Requires Application Default Credentials (gcloud auth application-default login).
                    NOTE: Vertex AI does NOT support API keys, only OAuth2/ADC.
                    If both api_key and project are set, api_key takes precedence.
            location: Google Cloud location (for Vertex AI with ADC, defaults to 'us-central1')
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
            use_vertexai: Whether to use Vertex AI instead of Gemini API.
                         - True: Use Vertex AI (with ADC or Express mode with API key)
                         - False: Use Gemini API (explicitly override env var)
                         - None (default): Check GOOGLE_GENAI_USE_VERTEXAI env var
                         When using Vertex AI with API key (Express mode), http_options with
                         api_version="v1" is automatically configured.
            response_format: Pydantic model for structured output. Used as default when
                           response_model is not passed to call()/acall() methods.
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

        if use_vertexai is None:
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
        self.response_format = response_format

        # Model-specific settings
        version_match = re.search(r"gemini-(\d+(?:\.\d+)?)", model.lower())
        self.supports_tools = bool(
            version_match and float(version_match.group(1)) >= 1.5
        )
        self.is_gemini_2_0 = bool(
            version_match and float(version_match.group(1)) >= 2.0
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

        Note:
            Google Gen AI SDK has two distinct endpoints with different auth requirements:
            - Gemini API (generativelanguage.googleapis.com): Supports API key authentication
            - Vertex AI (aiplatform.googleapis.com): Only supports OAuth2/ADC, NO API keys

            When vertexai=True is set, it routes to aiplatform.googleapis.com which rejects
            API keys. Use Gemini API endpoint for API key authentication instead.
        """
        client_params = {}

        if self.client_params:
            client_params.update(self.client_params)

        # Determine authentication mode based on available credentials
        has_api_key = bool(self.api_key)
        has_project = bool(self.project)

        if has_api_key and has_project:
            logging.warning(
                "Both API key and project provided. Using API key authentication. "
                "Project/location parameters are ignored when using API keys. "
                "To use Vertex AI with ADC, remove the api_key parameter."
            )
            has_project = False

        # Vertex AI with ADC (project without API key)
        if (use_vertexai or has_project) and not has_api_key:
            client_params.update(
                {
                    "vertexai": True,
                    "project": self.project,
                    "location": self.location,
                }
            )

        # API key authentication (works with both Gemini API and Vertex AI Express)
        elif has_api_key:
            client_params["api_key"] = self.api_key

            # Vertex AI Express mode: API key + vertexai=True + http_options with api_version="v1"
            # See: https://cloud.google.com/vertex-ai/generative-ai/docs/start/quickstart?usertype=apikey
            if use_vertexai:
                client_params["vertexai"] = True
                client_params["http_options"] = types.HttpOptions(api_version="v1")
            else:
                # This ensures we use the Gemini API (generativelanguage.googleapis.com)
                client_params["vertexai"] = False

            # Clean up project/location (not allowed with API key)
            client_params.pop("project", None)
            client_params.pop("location", None)

        else:
            try:
                return genai.Client(**client_params)
            except Exception as e:
                raise ValueError(
                    "Authentication required. Provide one of:\n"
                    "  1. API key via GOOGLE_API_KEY or GEMINI_API_KEY environment variable\n"
                    "     (use_vertexai=True is optional for Vertex AI with API key)\n"
                    "  2. For Vertex AI with ADC: Set GOOGLE_CLOUD_PROJECT and run:\n"
                    "     gcloud auth application-default login\n"
                    "  3. Pass api_key parameter directly to LLM constructor\n"
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
            if self.api_key:
                params["api_key"] = self.api_key
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
            effective_response_model = response_model or self.response_format

            formatted_content, system_instruction = self._format_messages_for_gemini(
                messages
            )

            messages_for_hooks = self._convert_contents_to_dict(formatted_content)

            if not self._invoke_before_llm_call_hooks(messages_for_hooks, from_agent):
                raise ValueError("LLM call blocked by before_llm_call hook")

            config = self._prepare_generation_config(
                system_instruction, tools, effective_response_model
            )

            if self.stream:
                return self._handle_streaming_completion(
                    formatted_content,
                    config,
                    available_functions,
                    from_task,
                    from_agent,
                    effective_response_model,
                )

            return self._handle_completion(
                formatted_content,
                config,
                available_functions,
                from_task,
                from_agent,
                effective_response_model,
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
            effective_response_model = response_model or self.response_format

            formatted_content, system_instruction = self._format_messages_for_gemini(
                messages
            )

            config = self._prepare_generation_config(
                system_instruction, tools, effective_response_model
            )

            if self.stream:
                return await self._ahandle_streaming_completion(
                    formatted_content,
                    config,
                    available_functions,
                    from_task,
                    from_agent,
                    effective_response_model,
                )

            return await self._ahandle_completion(
                formatted_content,
                config,
                available_functions,
                from_task,
                from_agent,
                effective_response_model,
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

        Note:
            Structured output support varies by model version:
            - Gemini 1.5 and earlier: Uses response_schema (Pydantic model)
            - Gemini 2.0+: Uses response_json_schema (JSON Schema) with propertyOrdering

            When both tools AND response_model are present, we add a structured_output
            pseudo-tool since Gemini doesn't support tools + response_schema together.
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

        if tools and self.supports_tools:
            gemini_tools = self._convert_tools_for_interference(tools)

            if response_model:
                schema_output = generate_model_description(response_model)
                schema = schema_output.get("json_schema", {}).get("schema", {})
                if self.is_gemini_2_0:
                    schema = self._add_property_ordering(schema)

                structured_output_tool = types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name=STRUCTURED_OUTPUT_TOOL_NAME,
                            description=(
                                "Use this tool to provide your final structured response. "
                                "Call this tool when you have gathered all necessary information "
                                "and are ready to provide the final answer in the required format."
                            ),
                            parameters_json_schema=schema,
                        )
                    ]
                )
                gemini_tools.append(structured_output_tool)

            config_params["tools"] = gemini_tools
        elif response_model:
            config_params["response_mime_type"] = "application/json"
            schema_output = generate_model_description(response_model)
            schema = schema_output.get("json_schema", {}).get("schema", {})

            if self.is_gemini_2_0:
                schema = self._add_property_ordering(schema)
                config_params["response_json_schema"] = schema
            else:
                config_params["response_schema"] = response_model

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
                parameters_json_schema=parameters if parameters else None,
            )

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

            # Build parts list from content
            parts: list[types.Part] = []
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if "text" in item:
                            parts.append(types.Part.from_text(text=str(item["text"])))
                        elif "inlineData" in item:
                            inline = item["inlineData"]
                            parts.append(
                                types.Part.from_bytes(
                                    data=base64.b64decode(inline["data"]),
                                    mime_type=inline["mimeType"],
                                )
                            )
                    else:
                        parts.append(types.Part.from_text(text=str(item)))
            else:
                parts.append(types.Part.from_text(text=str(content) if content else ""))

            text_content: str = " ".join(p.text for p in parts if p.text is not None)

            if role == "system":
                # Extract system instruction - Gemini handles it separately
                if system_instruction:
                    system_instruction += f"\n\n{text_content}"
                else:
                    system_instruction = text_content
            elif role == "tool":
                tool_call_id = message.get("tool_call_id")
                if not tool_call_id:
                    raise ValueError("Tool message missing required tool_call_id")

                tool_name = message.get("name", "")

                response_data: dict[str, Any]
                try:
                    parsed = json.loads(text_content) if text_content else {}
                    if isinstance(parsed, dict):
                        response_data = parsed
                    else:
                        response_data = {"result": parsed}
                except (json.JSONDecodeError, TypeError):
                    response_data = {"result": text_content}

                function_response_part = types.Part.from_function_response(
                    name=tool_name, response=response_data
                )
                contents.append(
                    types.Content(role="user", parts=[function_response_part])
                )
            elif role == "assistant" and message.get("tool_calls"):
                raw_parts: list[Any] | None = message.get("raw_tool_call_parts")
                if raw_parts and all(isinstance(p, types.Part) for p in raw_parts):
                    tool_parts: list[types.Part] = list(raw_parts)
                    if text_content:
                        tool_parts.insert(0, types.Part.from_text(text=text_content))
                else:
                    tool_parts = []
                    if text_content:
                        tool_parts.append(types.Part.from_text(text=text_content))

                    tool_calls: list[dict[str, Any]] = message.get("tool_calls") or []
                    for tool_call in tool_calls:
                        func: dict[str, Any] = tool_call.get("function") or {}
                        func_name: str = str(func.get("name") or "")
                        func_args_raw: str | dict[str, Any] = (
                            func.get("arguments") or {}
                        )

                        func_args: dict[str, Any]
                        if isinstance(func_args_raw, str):
                            try:
                                func_args = (
                                    json.loads(func_args_raw) if func_args_raw else {}
                                )
                            except (json.JSONDecodeError, TypeError):
                                func_args = {}
                        else:
                            func_args = func_args_raw

                        tool_parts.append(
                            types.Part.from_function_call(
                                name=func_name, args=func_args
                            )
                        )

                contents.append(types.Content(role="model", parts=tool_parts))
            else:
                # Convert role for Gemini (assistant -> model)
                gemini_role = "model" if role == "assistant" else "user"

                # Create Content object
                gemini_content = types.Content(role=gemini_role, parts=parts)
                contents.append(gemini_content)

        return contents, system_instruction

    def _validate_and_emit_structured_output(
        self,
        content: str,
        response_model: type[BaseModel],
        messages_for_event: list[LLMMessage],
        from_task: Any | None = None,
        from_agent: Any | None = None,
    ) -> BaseModel:
        """Validate content against response model and emit completion event.

        Args:
            content: Response content to validate
            response_model: Pydantic model for validation
            messages_for_event: Messages to include in event
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
                messages=messages_for_event,
            )

            return structured_data
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
    ) -> str | BaseModel:
        """Finalize completion response with validation and event emission.

        Args:
            content: The response content
            contents: Original contents for event conversion
            response_model: Pydantic model for structured output validation
            from_task: Task that initiated the call
            from_agent: Agent that initiated the call

        Returns:
            Final response content after processing (str or Pydantic model if response_model provided)
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

    def _handle_structured_output_tool_call(
        self,
        structured_data: dict[str, Any],
        response_model: type[BaseModel],
        contents: list[types.Content],
        from_task: Any | None = None,
        from_agent: Any | None = None,
    ) -> BaseModel:
        """Validate and emit event for structured_output tool call.

        Args:
            structured_data: The arguments passed to the structured_output tool
            response_model: Pydantic model to validate against
            contents: Original contents for event conversion
            from_task: Task that initiated the call
            from_agent: Agent that initiated the call

        Returns:
            Validated Pydantic model instance

        Raises:
            ValueError: If validation fails
        """
        try:
            validated_data = response_model.model_validate(structured_data)
            self._emit_call_completed_event(
                response=validated_data.model_dump_json(),
                call_type=LLMCallType.LLM_CALL,
                from_task=from_task,
                from_agent=from_agent,
                messages=self._convert_contents_to_dict(contents),
            )
            return validated_data
        except Exception as e:
            error_msg = (
                f"Failed to validate {STRUCTURED_OUTPUT_TOOL_NAME} tool response "
                f"with model {response_model.__name__}: {e}"
            )
            logging.error(error_msg)
            raise ValueError(error_msg) from e

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
                # Collect function call parts
                function_call_parts = [
                    part for part in candidate.content.parts if part.function_call
                ]

                # Check for structured_output pseudo-tool call (used when tools + response_model)
                if response_model and function_call_parts:
                    for part in function_call_parts:
                        if (
                            part.function_call
                            and part.function_call.name == STRUCTURED_OUTPUT_TOOL_NAME
                        ):
                            structured_data = (
                                dict(part.function_call.args)
                                if part.function_call.args
                                else {}
                            )
                            return self._handle_structured_output_tool_call(
                                structured_data=structured_data,
                                response_model=response_model,
                                contents=contents,
                                from_task=from_task,
                                from_agent=from_agent,
                            )

                # Filter out structured_output from function calls returned to executor
                non_structured_output_parts = [
                    part
                    for part in function_call_parts
                    if not (
                        part.function_call
                        and part.function_call.name == STRUCTURED_OUTPUT_TOOL_NAME
                    )
                ]

                # If there are function calls but no available_functions,
                # return them for the executor to handle (like OpenAI/Anthropic)
                if non_structured_output_parts and not available_functions:
                    self._emit_call_completed_event(
                        response=non_structured_output_parts,
                        call_type=LLMCallType.TOOL_CALL,
                        from_task=from_task,
                        from_agent=from_agent,
                        messages=self._convert_contents_to_dict(contents),
                    )
                    return non_structured_output_parts

                # Otherwise execute the tools internally
                for part in candidate.content.parts:
                    if part.function_call:
                        function_name = part.function_call.name
                        if function_name is None:
                            continue
                        # Skip structured_output - it's handled above
                        if function_name == STRUCTURED_OUTPUT_TOOL_NAME:
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

        content = self._extract_text_from_response(response)

        effective_response_model = None if self.tools else response_model
        if not effective_response_model:
            content = self._apply_stop_words(content)

        return self._finalize_completion_response(
            content=content,
            contents=contents,
            response_model=effective_response_model,
            from_task=from_task,
            from_agent=from_agent,
        )

    def _process_stream_chunk(
        self,
        chunk: GenerateContentResponse,
        full_response: str,
        function_calls: dict[int, dict[str, Any]],
        usage_data: dict[str, int],
        from_task: Any | None = None,
        from_agent: Any | None = None,
    ) -> tuple[str, dict[int, dict[str, Any]], dict[str, int]]:
        """Process a single streaming chunk.

        Args:
            chunk: The streaming chunk response
            full_response: Accumulated response text
            function_calls: Accumulated function calls keyed by sequential index
            usage_data: Accumulated usage data
            from_task: Task that initiated the call
            from_agent: Agent that initiated the call

        Returns:
            Tuple of (updated full_response, updated function_calls, updated usage_data)
        """
        response_id = chunk.response_id if hasattr(chunk, "response_id") else None
        if chunk.usage_metadata:
            usage_data = self._extract_token_usage(chunk)

        if chunk.text:
            full_response += chunk.text
            self._emit_stream_chunk_event(
                chunk=chunk.text,
                from_task=from_task,
                from_agent=from_agent,
                response_id=response_id,
            )

        if chunk.candidates:
            candidate = chunk.candidates[0]
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if part.function_call:
                        call_index = len(function_calls)
                        call_id = f"call_{call_index}"
                        args_dict = (
                            dict(part.function_call.args)
                            if part.function_call.args
                            else {}
                        )
                        args_json = json.dumps(args_dict)

                        function_calls[call_index] = {
                            "id": call_id,
                            "name": part.function_call.name,
                            "args": args_dict,
                        }

                        self._emit_stream_chunk_event(
                            chunk=args_json,
                            from_task=from_task,
                            from_agent=from_agent,
                            tool_call={
                                "id": call_id,
                                "function": {
                                    "name": part.function_call.name or "",
                                    "arguments": args_json,
                                },
                                "type": "function",
                                "index": call_index,
                            },
                            call_type=LLMCallType.TOOL_CALL,
                            response_id=response_id,
                        )

        return full_response, function_calls, usage_data

    def _finalize_streaming_response(
        self,
        full_response: str,
        function_calls: dict[int, dict[str, Any]],
        usage_data: dict[str, int],
        contents: list[types.Content],
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | BaseModel | list[dict[str, Any]]:
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

        if response_model and function_calls:
            for call_data in function_calls.values():
                if call_data.get("name") == STRUCTURED_OUTPUT_TOOL_NAME:
                    structured_data = call_data.get("args", {})
                    return self._handle_structured_output_tool_call(
                        structured_data=structured_data,
                        response_model=response_model,
                        contents=contents,
                        from_task=from_task,
                        from_agent=from_agent,
                    )

        non_structured_output_calls = {
            idx: call_data
            for idx, call_data in function_calls.items()
            if call_data.get("name") != STRUCTURED_OUTPUT_TOOL_NAME
        }

        # If there are function calls but no available_functions,
        # return them for the executor to handle
        if non_structured_output_calls and not available_functions:
            formatted_function_calls = [
                {
                    "id": call_data["id"],
                    "function": {
                        "name": call_data["name"],
                        "arguments": json.dumps(call_data["args"]),
                    },
                    "type": "function",
                }
                for call_data in non_structured_output_calls.values()
            ]
            self._emit_call_completed_event(
                response=formatted_function_calls,
                call_type=LLMCallType.TOOL_CALL,
                from_task=from_task,
                from_agent=from_agent,
                messages=self._convert_contents_to_dict(contents),
            )
            return formatted_function_calls

        # Handle completed function calls (excluding structured_output)
        if non_structured_output_calls and available_functions:
            for call_data in non_structured_output_calls.values():
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

        # When tools are present, structured output should come via the structured_output
        # pseudo-tool, not via direct text response. If we reach here with tools present,
        # the LLM chose to return plain text instead of calling structured_output.
        effective_response_model = None if self.tools else response_model

        return self._finalize_completion_response(
            content=full_response,
            contents=contents,
            response_model=effective_response_model,
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
    ) -> str | BaseModel | list[dict[str, Any]] | Any:
        """Handle streaming content generation."""
        full_response = ""
        function_calls: dict[int, dict[str, Any]] = {}
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
    ) -> str | Any:
        """Handle async streaming content generation."""
        full_response = ""
        function_calls: dict[int, dict[str, Any]] = {}
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
    def _extract_text_from_response(response: GenerateContentResponse) -> str:
        """Extract text content from Gemini response without triggering warnings.

        This method directly accesses the response parts to extract text content,
        avoiding the warning that occurs when using response.text on responses
        containing non-text parts (e.g., 'thought_signature' from thinking models).

        Args:
            response: The Gemini API response

        Returns:
            Concatenated text content from all text parts
        """
        if not response.candidates:
            return ""

        candidate = response.candidates[0]
        if not candidate.content or not candidate.content.parts:
            return ""

        text_parts = [
            part.text
            for part in candidate.content.parts
            if hasattr(part, "text") and part.text
        ]

        return "".join(text_parts)

    @staticmethod
    def _add_property_ordering(schema: dict[str, Any]) -> dict[str, Any]:
        """Add propertyOrdering to JSON schema for Gemini 2.0 compatibility.

        Gemini 2.0 models require an explicit propertyOrdering list to define
        the preferred structure of JSON objects. This recursively adds
        propertyOrdering to all objects in the schema.

        Args:
            schema: JSON schema dictionary.

        Returns:
            Modified schema with propertyOrdering added to all objects.
        """
        if isinstance(schema, dict):
            if schema.get("type") == "object" and "properties" in schema:
                properties = schema["properties"]
                if properties and "propertyOrdering" not in schema:
                    schema["propertyOrdering"] = list(properties.keys())

            for value in schema.values():
                if isinstance(value, dict):
                    GeminiCompletion._add_property_ordering(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            GeminiCompletion._add_property_ordering(item)

        return schema

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

    def supports_multimodal(self) -> bool:
        """Check if the model supports multimodal inputs.

        Gemini models support images, audio, video, and PDFs.

        Returns:
            True if the model supports multimodal inputs.
        """
        return True

    def format_text_content(self, text: str) -> dict[str, Any]:
        """Format text as a Gemini content block.

        Gemini uses {"text": "..."} format instead of {"type": "text", "text": "..."}.

        Args:
            text: The text content to format.

        Returns:
            A content block in Gemini's expected format.
        """
        return {"text": text}

    def get_file_uploader(self) -> Any:
        """Get a Gemini file uploader using this LLM's client.

        Returns:
            GeminiFileUploader instance with pre-configured client.
        """
        try:
            from crewai_files.uploaders.gemini import GeminiFileUploader

            return GeminiFileUploader(client=self.client)
        except ImportError:
            return None
