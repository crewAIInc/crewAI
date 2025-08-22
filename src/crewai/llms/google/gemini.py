import os
from typing import Any, Dict, List, Optional, Union, Type, Literal, TYPE_CHECKING
from pydantic import BaseModel

if TYPE_CHECKING:
    from google import genai
    from google.genai import types

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

from crewai.llms.base_llm import BaseLLM
from crewai.utilities.events import crewai_event_bus
from crewai.utilities.events.llm_events import (
    LLMCallCompletedEvent,
    LLMCallFailedEvent,
    LLMCallStartedEvent,
    LLMCallType,
)
from crewai.utilities.events.tool_usage_events import (
    ToolUsageStartedEvent,
    ToolUsageFinishedEvent,
    ToolUsageErrorEvent,
)
from datetime import datetime


class GeminiLLM(BaseLLM):
    """Google Gemini LLM implementation using the official Google Gen AI Python SDK."""

    def __init__(
        self,
        model: str = "gemini-1.5-pro",
        timeout: Optional[Union[float, int]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        n: Optional[int] = None,  # Not supported by Gemini but kept for compatibility
        stop: Optional[Union[str, List[str]]] = None,
        max_completion_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        presence_penalty: Optional[
            float
        ] = None,  # Not supported but kept for compatibility
        frequency_penalty: Optional[
            float
        ] = None,  # Not supported but kept for compatibility
        logit_bias: Optional[
            Dict[int, float]
        ] = None,  # Not supported but kept for compatibility
        response_format: Optional[Type[BaseModel]] = None,
        seed: Optional[int] = None,  # Not supported but kept for compatibility
        logprobs: Optional[int] = None,  # Not supported but kept for compatibility
        top_logprobs: Optional[int] = None,  # Not supported but kept for compatibility
        base_url: Optional[str] = None,  # Not used by Gemini
        api_base: Optional[str] = None,  # Not used by Gemini
        api_version: Optional[str] = None,  # Not used by Gemini
        api_key: Optional[str] = None,
        callbacks: List[Any] = [],
        reasoning_effort: Optional[
            Literal["none", "low", "medium", "high"]
        ] = None,  # Not used by Gemini
        stream: bool = False,
        max_retries: int = 2,
        # Gemini-specific parameters
        top_k: Optional[int] = None,  # Gemini top-k sampling parameter
        candidate_count: int = 1,  # Number of response candidates
        safety_settings: Optional[
            List[Dict[str, Any]]
        ] = None,  # Gemini safety settings
        generation_config: Optional[
            Dict[str, Any]
        ] = None,  # Additional generation config
        # Vertex AI parameters
        use_vertex_ai: bool = False,
        project_id: Optional[str] = None,
        location: str = "us-central1",
        **kwargs,
    ):
        """Initialize Gemini LLM with the official Google Gen AI SDK.

        Args:
            model: Gemini model name (e.g., 'gemini-1.5-pro', 'gemini-2.0-flash-001')
            timeout: Request timeout in seconds
            temperature: Sampling temperature (0-2 for Gemini)
            top_p: Nucleus sampling parameter
            n: Number of completions (not supported by Gemini, kept for compatibility)
            stop: Stop sequences
            max_completion_tokens: Maximum tokens in completion
            max_tokens: Maximum tokens (legacy parameter)
            presence_penalty: Not supported by Gemini, kept for compatibility
            frequency_penalty: Not supported by Gemini, kept for compatibility
            logit_bias: Not supported by Gemini, kept for compatibility
            response_format: Pydantic model for structured output
            seed: Not supported by Gemini, kept for compatibility
            logprobs: Not supported by Gemini, kept for compatibility
            top_logprobs: Not supported by Gemini, kept for compatibility
            base_url: Not used by Gemini
            api_base: Not used by Gemini
            api_version: Not used by Gemini
            api_key: Google AI API key
            callbacks: List of callback functions
            reasoning_effort: Not used by Gemini, kept for compatibility
            stream: Whether to stream responses
            max_retries: Number of retries for failed requests
            top_k: Gemini-specific top-k sampling parameter
            candidate_count: Number of response candidates to generate
            safety_settings: Gemini safety settings configuration
            generation_config: Additional Gemini generation configuration
            use_vertex_ai: Whether to use Vertex AI instead of Gemini API
            project_id: Google Cloud project ID (required for Vertex AI)
            location: Google Cloud region (default: us-central1)
            **kwargs: Additional parameters
        """
        # Check if Google Gen AI SDK is available
        if genai is None or types is None:
            raise ImportError(
                "Google Gen AI Python SDK is required. Please install it with: "
                "pip install google-genai"
            )

        super().__init__(model=model, temperature=temperature)

        # Store all parameters for compatibility
        self.timeout = timeout
        self.top_p = top_p
        self.n = n
        self.max_completion_tokens = max_completion_tokens
        self.max_tokens = max_tokens or max_completion_tokens
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.logit_bias = logit_bias
        self.response_format = response_format
        self.seed = seed
        self.logprobs = logprobs
        self.top_logprobs = top_logprobs
        self.api_base = api_base
        self.base_url = base_url
        self.api_version = api_version
        self.callbacks = callbacks
        self.reasoning_effort = reasoning_effort
        self.stream = stream
        self.additional_params = kwargs
        self.context_window_size = 0
        self.max_retries = max_retries

        # Gemini-specific parameters
        self.top_k = top_k
        self.candidate_count = candidate_count
        self.safety_settings = safety_settings or []
        self.generation_config = generation_config or {}

        # Vertex AI parameters
        self.use_vertex_ai = use_vertex_ai
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.location = location

        # API key handling
        self.api_key = (
            api_key
            or os.getenv("GOOGLE_AI_API_KEY")
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
        )

        # Normalize stop parameter to match LLM class behavior
        if stop is None:
            self.stop: List[str] = []
        elif isinstance(stop, str):
            self.stop = [stop]
        else:
            self.stop = stop

        # Initialize client attribute
        self.client: Any = None

        # Initialize the Google Gen AI client
        self._initialize_client()
        self.model_config = self._get_model_config()

    def _initialize_client(self):
        """Initialize the Google Gen AI client."""
        if genai is None or types is None:
            return

        try:
            if self.use_vertex_ai:
                if not self.project_id:
                    raise ValueError(
                        "project_id is required when use_vertex_ai=True. "
                        "Set it directly or via GOOGLE_CLOUD_PROJECT environment variable."
                    )
                self.client = genai.Client(
                    vertexai=True,
                    project=self.project_id,
                    location=self.location,
                )
            else:
                if not self.api_key:
                    raise ValueError(
                        "API key is required for Gemini Developer API. "
                        "Set it via api_key parameter or GOOGLE_AI_API_KEY/GEMINI_API_KEY environment variable."
                    )
                self.client = genai.Client(api_key=self.api_key)
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize Google Gen AI client: {str(e)}"
            ) from e

    def _get_model_config(self) -> Dict[str, Any]:
        """Get model-specific configuration for Gemini models."""
        # Gemini model configurations based on Google's documentation
        model_configs = {
            # Gemini 2.0 Flash (latest)
            "gemini-2.0-flash": {
                "context_window": 1048576,
                "supports_tools": True,
                "supports_vision": True,
            },
            "gemini-2.0-flash-001": {
                "context_window": 1048576,
                "supports_tools": True,
                "supports_vision": True,
            },
            "gemini-2.0-flash-exp": {
                "context_window": 1048576,
                "supports_tools": True,
                "supports_vision": True,
            },
            # Gemini 1.5 Pro
            "gemini-1.5-pro": {
                "context_window": 2097152,
                "supports_tools": True,
                "supports_vision": True,
            },
            "gemini-1.5-pro-002": {
                "context_window": 2097152,
                "supports_tools": True,
                "supports_vision": True,
            },
            "gemini-1.5-pro-001": {
                "context_window": 2097152,
                "supports_tools": True,
                "supports_vision": True,
            },
            "gemini-1.5-pro-exp-0827": {
                "context_window": 2097152,
                "supports_tools": True,
                "supports_vision": True,
            },
            # Gemini 1.5 Flash
            "gemini-1.5-flash": {
                "context_window": 1048576,
                "supports_tools": True,
                "supports_vision": True,
            },
            "gemini-1.5-flash-002": {
                "context_window": 1048576,
                "supports_tools": True,
                "supports_vision": True,
            },
            "gemini-1.5-flash-001": {
                "context_window": 1048576,
                "supports_tools": True,
                "supports_vision": True,
            },
            "gemini-1.5-flash-8b": {
                "context_window": 1048576,
                "supports_tools": True,
                "supports_vision": True,
            },
            "gemini-1.5-flash-8b-exp-0827": {
                "context_window": 1048576,
                "supports_tools": True,
                "supports_vision": True,
            },
            # Legacy Gemini Pro
            "gemini-pro": {
                "context_window": 30720,
                "supports_tools": True,
                "supports_vision": False,
            },
            "gemini-pro-vision": {
                "context_window": 16384,
                "supports_tools": False,
                "supports_vision": True,
            },
            # Gemini Ultra (when available)
            "gemini-ultra": {
                "context_window": 30720,
                "supports_tools": True,
                "supports_vision": True,
            },
        }

        # Default config if model not found
        default_config = {
            "context_window": 1048576,
            "supports_tools": True,
            "supports_vision": True,
        }

        # Try exact match first
        if self.model in model_configs:
            return model_configs[self.model]

        # Try prefix match for versioned models
        for model_prefix, config in model_configs.items():
            if self.model.startswith(model_prefix):
                return config

        return default_config

    def _format_messages(self, messages: Union[str, List[Dict[str, str]]]) -> List[Any]:
        """Format messages for Google Gen AI SDK.

        Args:
            messages: Input messages as string or list of dicts

        Returns:
            List of properly formatted Content objects
        """
        if genai is None or types is None:
            return []

        if isinstance(messages, str):
            return [
                types.Content(role="user", parts=[types.Part.from_text(text=messages)])
            ]

        # Validate message format
        for msg in messages:
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                raise ValueError(
                    "Each message must be a dict with 'role' and 'content' keys"
                )

        # Convert to Google Gen AI SDK format
        formatted_messages = []
        system_instruction = None

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                # System instruction will be handled separately
                system_instruction = content
            elif role == "user":
                formatted_messages.append(
                    types.Content(
                        role="user", parts=[types.Part.from_text(text=content)]
                    )
                )
            elif role == "assistant":
                formatted_messages.append(
                    types.Content(
                        role="model", parts=[types.Part.from_text(text=content)]
                    )
                )

        # Store system instruction for later use
        self._system_instruction = system_instruction

        return formatted_messages

    def _format_tools(self, tools: Optional[List[dict]]) -> Optional[List[Any]]:
        """Format tools for Google Gen AI SDK function calling.

        Args:
            tools: List of tool definitions

        Returns:
            Google Gen AI SDK formatted tool definitions
        """
        if genai is None or types is None:
            return None

        if not tools or not self.model_config.get("supports_tools", True):
            return None

        formatted_tools = []
        for tool in tools:
            # Convert to Google Gen AI SDK function declaration format
            function_declaration = types.FunctionDeclaration(
                name=tool.get("name", ""),
                description=tool.get("description", ""),
                parameters=tool.get("parameters", {}),
            )
            formatted_tools.append(
                types.Tool(function_declarations=[function_declaration])
            )

        return formatted_tools

    def _build_generation_config(
        self,
        system_instruction: Optional[str] = None,
        tools: Optional[List[Any]] = None,
    ) -> Any:
        """Build Google Gen AI SDK generation config from parameters."""
        if genai is None or types is None:
            return {}
        config_dict = self.generation_config.copy()

        # Add parameters that map to Gemini's generation config
        if self.temperature is not None:
            config_dict["temperature"] = self.temperature

        if self.top_p is not None:
            config_dict["top_p"] = self.top_p

        if self.top_k is not None:
            config_dict["top_k"] = self.top_k

        if self.max_tokens is not None:
            config_dict["max_output_tokens"] = self.max_tokens

        if self.candidate_count is not None:
            config_dict["candidate_count"] = self.candidate_count

        if self.stop:
            config_dict["stop_sequences"] = self.stop

        if self.stream:
            config_dict["stream"] = True

        # Add safety settings
        if self.safety_settings:
            config_dict["safety_settings"] = self.safety_settings

        # Add response format if specified
        if self.response_format:
            config_dict["response_modalities"] = ["TEXT"]

        # Add system instruction if present
        if system_instruction:
            config_dict["system_instruction"] = system_instruction

        # Add tools if present
        if tools:
            config_dict["tools"] = tools

        return types.GenerateContentConfig(**config_dict)

    def _handle_tool_calls(
        self,
        response,
        available_functions: Optional[Dict[str, Any]] = None,
        from_task: Optional[Any] = None,
        from_agent: Optional[Any] = None,
    ) -> Any:
        """Handle tool calls from Google Gen AI SDK response.

        Args:
            response: Google Gen AI SDK response
            available_functions: Dict mapping function names to callables
            from_task: Optional task context
            from_agent: Optional agent context

        Returns:
            Result of function execution or error message
        """
        # Check if response has function calls
        if (
            not available_functions
            or not hasattr(response, "candidates")
            or not response.candidates
        ):
            return response.text if hasattr(response, "text") else str(response)

        candidate = response.candidates[0] if response.candidates else None
        if (
            not candidate
            or not hasattr(candidate, "content")
            or not hasattr(candidate.content, "parts")
        ):
            return response.text if hasattr(response, "text") else str(response)

        # Look for function call parts
        for part in candidate.content.parts:
            if hasattr(part, "function_call"):
                function_call = part.function_call
                function_name = function_call.name
                function_args = {}

                if function_name not in available_functions:
                    return f"Error: Function '{function_name}' not found in available functions"

                try:
                    # Google Gen AI SDK provides arguments as a struct
                    function_args = (
                        dict(function_call.args)
                        if hasattr(function_call, "args")
                        else {}
                    )
                    fn = available_functions[function_name]

                    # Execute function with event tracking
                    assert hasattr(crewai_event_bus, "emit")
                    started_at = datetime.now()
                    crewai_event_bus.emit(
                        self,
                        event=ToolUsageStartedEvent(
                            tool_name=function_name,
                            tool_args=function_args,
                        ),
                    )

                    result = fn(**function_args)

                    crewai_event_bus.emit(
                        self,
                        event=ToolUsageFinishedEvent(
                            output=result,
                            tool_name=function_name,
                            tool_args=function_args,
                            started_at=started_at,
                            finished_at=datetime.now(),
                        ),
                    )

                    # Emit success event
                    event_data = {
                        "response": result,
                        "call_type": LLMCallType.TOOL_CALL,
                        "model": self.model,
                    }
                    if from_task is not None:
                        event_data["from_task"] = from_task
                    if from_agent is not None:
                        event_data["from_agent"] = from_agent

                    crewai_event_bus.emit(
                        self,
                        event=LLMCallCompletedEvent(**event_data),
                    )

                    return result

                except Exception as e:
                    error_msg = f"Error executing function '{function_name}': {e}"
                    crewai_event_bus.emit(
                        self,
                        event=ToolUsageErrorEvent(
                            tool_name=function_name,
                            tool_args=function_args,
                            error=error_msg,
                        ),
                    )
                    return error_msg

        # If no function calls, return text content
        return response.text if hasattr(response, "text") else str(response)

    def call(
        self,
        messages: Union[str, List[Dict[str, str]]],
        tools: Optional[List[dict]] = None,
        callbacks: Optional[List[Any]] = None,
        available_functions: Optional[Dict[str, Any]] = None,
        from_task: Optional[Any] = None,
        from_agent: Optional[Any] = None,
    ) -> Union[str, Any]:
        """Call Google Gen AI SDK with the given messages.

        Args:
            messages: Input messages for the LLM
            tools: Optional list of tool schemas
            callbacks: Optional callbacks to execute
            available_functions: Optional dict of available functions
            from_task: Optional task context
            from_agent: Optional agent context

        Returns:
            LLM response or tool execution result

        Raises:
            ValueError: If messages format is invalid
            RuntimeError: If API call fails
        """
        # Emit call started event
        print("calling from native gemini", messages)
        assert hasattr(crewai_event_bus, "emit")

        # Prepare event data
        started_event_data = {
            "messages": messages,
            "tools": tools,
            "callbacks": callbacks,
            "available_functions": available_functions,
            "model": self.model,
        }
        if from_task is not None:
            started_event_data["from_task"] = from_task
        if from_agent is not None:
            started_event_data["from_agent"] = from_agent

        crewai_event_bus.emit(
            self,
            event=LLMCallStartedEvent(**started_event_data),
        )

        retry_count = 0
        last_error = None

        while retry_count <= self.max_retries:
            try:
                # Format messages
                formatted_messages = self._format_messages(messages)
                system_instruction = getattr(self, "_system_instruction", None)

                # Format tools if provided and supported
                formatted_tools = self._format_tools(tools)

                # Build generation config
                generation_config = self._build_generation_config(
                    system_instruction, formatted_tools
                )

                # Execute callbacks before API call
                if callbacks:
                    for callback in callbacks:
                        if hasattr(callback, "on_llm_start"):
                            callback.on_llm_start(
                                serialized={"name": self.__class__.__name__},
                                prompts=[str(formatted_messages)],
                            )

                # Prepare the API call parameters
                api_params = {
                    "model": self.model,
                    "contents": formatted_messages,
                    "config": generation_config,
                }

                # Make API call
                if self.stream:
                    # Streaming response
                    response_stream = self.client.models.generate_content(**api_params)

                    full_response = ""
                    try:
                        for chunk in response_stream:
                            if hasattr(chunk, "text") and chunk.text:
                                full_response += chunk.text
                    except Exception as e:
                        print(
                            f"Streaming error (continuing with partial response): {e}"
                        )

                    result = full_response or "No response content"
                else:
                    # Non-streaming response
                    response = self.client.models.generate_content(**api_params)

                    # Handle tool calls if present
                    result = self._handle_tool_calls(
                        response, available_functions, from_task, from_agent
                    )

                # Execute callbacks after API call
                if callbacks:
                    for callback in callbacks:
                        if hasattr(callback, "on_llm_end"):
                            callback.on_llm_end(response=result)

                # Emit completion event
                completion_event_data = {
                    "messages": messages,  # Use original messages, not formatted_messages
                    "response": result,
                    "call_type": LLMCallType.LLM_CALL,
                    "model": self.model,
                }
                if from_task is not None:
                    completion_event_data["from_task"] = from_task
                if from_agent is not None:
                    completion_event_data["from_agent"] = from_agent

                crewai_event_bus.emit(
                    self,
                    event=LLMCallCompletedEvent(**completion_event_data),
                )

                return result

            except Exception as e:
                last_error = e
                retry_count += 1

                if retry_count <= self.max_retries:
                    print(
                        f"Gemini API call failed (attempt {retry_count}/{self.max_retries + 1}): {e}"
                    )
                    continue

                # All retries exhausted
                # Execute error callbacks
                if callbacks:
                    for callback in callbacks:
                        if hasattr(callback, "on_llm_error"):
                            callback.on_llm_error(error=e)

                # Emit failed event
                crewai_event_bus.emit(
                    self,
                    event=LLMCallFailedEvent(error=str(e)),
                )

                raise RuntimeError(
                    f"Gemini API call failed after {self.max_retries + 1} attempts: {str(e)}"
                ) from e

    def supports_stop_words(self) -> bool:
        """Check if Gemini models support stop words."""
        return True

    def get_context_window_size(self) -> int:
        """Get the context window size for the current model."""
        if self.context_window_size != 0:
            return self.context_window_size

        # Use 85% of the context window like the original LLM class
        context_window = self.model_config.get("context_window", 1048576)
        self.context_window_size = int(context_window * 0.85)
        return self.context_window_size

    def supports_function_calling(self) -> bool:
        """Check if the current model supports function calling."""
        return self.model_config.get("supports_tools", True)

    def supports_vision(self) -> bool:
        """Check if the current model supports vision capabilities."""
        return self.model_config.get("supports_vision", False)
