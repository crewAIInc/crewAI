import json
import os
from typing import Any, Dict, List, Optional, Union, Type, Literal
from openai import OpenAI
from pydantic import BaseModel

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


class OpenAILLM(BaseLLM):
    """OpenAI LLM implementation with full LLM class compatibility."""

    def __init__(
        self,
        model: str = "gpt-4",
        timeout: Optional[Union[float, int]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        n: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        max_completion_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[Dict[int, float]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        seed: Optional[int] = None,
        logprobs: Optional[int] = None,
        top_logprobs: Optional[int] = None,
        base_url: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        api_key: Optional[str] = None,
        callbacks: List[Any] = [],
        reasoning_effort: Optional[Literal["none", "low", "medium", "high"]] = None,
        stream: bool = False,
        max_retries: int = 2,
        **kwargs,
    ):
        """Initialize OpenAI LLM with full compatibility.

        Args:
            model: OpenAI model name (e.g., 'gpt-4', 'gpt-3.5-turbo')
            timeout: Request timeout in seconds
            temperature: Sampling temperature (0-2)
            top_p: Nucleus sampling parameter
            n: Number of completions to generate
            stop: Stop sequences
            max_completion_tokens: Maximum tokens in completion
            max_tokens: Maximum tokens (legacy parameter)
            presence_penalty: Presence penalty (-2 to 2)
            frequency_penalty: Frequency penalty (-2 to 2)
            logit_bias: Logit bias dictionary
            response_format: Pydantic model for structured output
            seed: Random seed for deterministic output
            logprobs: Whether to return log probabilities
            top_logprobs: Number of most likely tokens to return
            base_url: Custom API base URL
            api_base: Legacy API base parameter
            api_version: API version (for Azure)
            api_key: OpenAI API key
            callbacks: List of callback functions
            reasoning_effort: Reasoning effort for o1 models
            stream: Whether to stream responses
            max_retries: Number of retries for failed requests
            **kwargs: Additional parameters
        """
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
        self.api_base = api_base or base_url
        self.base_url = base_url or api_base
        self.api_version = api_version
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.callbacks = callbacks
        self.reasoning_effort = reasoning_effort
        self.stream = stream
        self.additional_params = kwargs
        self.context_window_size = 0

        # Normalize stop parameter to match LLM class behavior
        if stop is None:
            self.stop: List[str] = []
        elif isinstance(stop, str):
            self.stop = [stop]
        else:
            self.stop = stop

        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=max_retries,
            **kwargs,
        )

        self.model_config = self._get_model_config()

    def _get_model_config(self) -> Dict[str, Any]:
        """Get model-specific configuration."""
        # Enhanced model configurations matching current LLM_CONTEXT_WINDOW_SIZES
        model_configs = {
            "gpt-4": {"context_window": 8192, "supports_tools": True},
            "gpt-4o": {"context_window": 128000, "supports_tools": True},
            "gpt-4o-mini": {"context_window": 200000, "supports_tools": True},
            "gpt-4-turbo": {"context_window": 128000, "supports_tools": True},
            "gpt-4.1": {"context_window": 1047576, "supports_tools": True},
            "gpt-4.1-mini": {"context_window": 1047576, "supports_tools": True},
            "gpt-4.1-nano": {"context_window": 1047576, "supports_tools": True},
            "gpt-3.5-turbo": {"context_window": 16385, "supports_tools": True},
            "o1-preview": {"context_window": 128000, "supports_tools": False},
            "o1-mini": {"context_window": 128000, "supports_tools": False},
            "o3-mini": {"context_window": 200000, "supports_tools": False},
            "o4-mini": {"context_window": 200000, "supports_tools": False},
        }

        # Default config if model not found
        default_config = {"context_window": 4096, "supports_tools": True}

        for model_prefix, config in model_configs.items():
            if self.model.startswith(model_prefix):
                return config

        return default_config

    def _format_messages(
        self, messages: Union[str, List[Dict[str, str]]]
    ) -> List[Dict[str, str]]:
        """Format messages for OpenAI API.

        Args:
            messages: Input messages as string or list of dicts

        Returns:
            List of properly formatted message dicts
        """
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]

        # Validate message format
        for msg in messages:
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                raise ValueError(
                    "Each message must be a dict with 'role' and 'content' keys"
                )

        # Handle O1 model special case (system messages not supported)
        if "o1" in self.model.lower():
            formatted_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    # Convert system messages to assistant messages for O1
                    formatted_messages.append(
                        {"role": "assistant", "content": msg["content"]}
                    )
                else:
                    formatted_messages.append(msg)
            return formatted_messages

        return messages

    def _format_tools(self, tools: Optional[List[dict]]) -> Optional[List[dict]]:
        """Format tools for OpenAI function calling.

        Args:
            tools: List of tool definitions

        Returns:
            OpenAI-formatted tool definitions
        """
        if not tools or not self.model_config.get("supports_tools", True):
            return None

        formatted_tools = []
        for tool in tools:
            # Convert to OpenAI tool format
            formatted_tool = {
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {}),
                },
            }
            formatted_tools.append(formatted_tool)

        return formatted_tools

    def _handle_tool_calls(
        self,
        response,
        available_functions: Optional[Dict[str, Any]] = None,
        from_task: Optional[Any] = None,
        from_agent: Optional[Any] = None,
    ) -> Any:
        """Handle tool calls from OpenAI response.

        Args:
            response: OpenAI API response
            available_functions: Dict mapping function names to callables
            from_task: Optional task context
            from_agent: Optional agent context

        Returns:
            Result of function execution or error message
        """
        message = response.choices[0].message

        if not message.tool_calls or not available_functions:
            return message.content

        # Execute the first tool call
        tool_call = message.tool_calls[0]
        function_name = tool_call.function.name
        function_args = {}

        if function_name not in available_functions:
            return f"Error: Function '{function_name}' not found in available functions"

        try:
            # Parse function arguments
            function_args = json.loads(tool_call.function.arguments)
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

        except json.JSONDecodeError as e:
            error_msg = f"Error parsing function arguments: {e}"
            crewai_event_bus.emit(
                self,
                event=ToolUsageErrorEvent(
                    tool_name=function_name,
                    tool_args=function_args,
                    error=error_msg,
                ),
            )
            return error_msg
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

    def call(
        self,
        messages: Union[str, List[Dict[str, str]]],
        tools: Optional[List[dict]] = None,
        callbacks: Optional[List[Any]] = None,
        available_functions: Optional[Dict[str, Any]] = None,
        from_task: Optional[Any] = None,
        from_agent: Optional[Any] = None,
    ) -> Union[str, Any]:
        """Call OpenAI API with the given messages.

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
        print("calling from native openai", messages)
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

        try:
            # Format messages
            formatted_messages = self._format_messages(messages)

            # Prepare API call parameters
            api_params = {
                "model": self.model,
                "messages": formatted_messages,
            }

            # Add optional parameters
            if self.temperature is not None:
                api_params["temperature"] = self.temperature

            if self.top_p is not None:
                api_params["top_p"] = self.top_p

            if self.n is not None:
                api_params["n"] = self.n

            if self.max_tokens is not None:
                api_params["max_tokens"] = self.max_tokens

            if self.presence_penalty is not None:
                api_params["presence_penalty"] = self.presence_penalty

            if self.frequency_penalty is not None:
                api_params["frequency_penalty"] = self.frequency_penalty

            if self.logit_bias is not None:
                api_params["logit_bias"] = self.logit_bias

            if self.seed is not None:
                api_params["seed"] = self.seed

            if self.logprobs is not None:
                api_params["logprobs"] = self.logprobs

            if self.top_logprobs is not None:
                api_params["top_logprobs"] = self.top_logprobs

            if self.stop:
                api_params["stop"] = self.stop

            if self.response_format is not None:
                # Handle structured output for Pydantic models
                if hasattr(self.response_format, "model_json_schema"):
                    api_params["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": self.response_format.__name__,
                            "schema": self.response_format.model_json_schema(),
                            "strict": True,
                        },
                    }
                else:
                    api_params["response_format"] = self.response_format

            if self.reasoning_effort is not None and "o1" in self.model:
                api_params["reasoning_effort"] = self.reasoning_effort

            # Add tools if provided and supported
            formatted_tools = self._format_tools(tools)
            if formatted_tools:
                api_params["tools"] = formatted_tools
                api_params["tool_choice"] = "auto"

            # Execute callbacks before API call
            if callbacks:
                for callback in callbacks:
                    if hasattr(callback, "on_llm_start"):
                        callback.on_llm_start(
                            serialized={"name": self.__class__.__name__},
                            prompts=[str(formatted_messages)],
                        )

            # Make API call
            if self.stream:
                response = self.client.chat.completions.create(
                    stream=True, **api_params
                )
                # Handle streaming (simplified for now)
                full_response = ""
                for chunk in response:
                    if (
                        hasattr(chunk.choices[0].delta, "content")
                        and chunk.choices[0].delta.content
                    ):
                        full_response += chunk.choices[0].delta.content
                result = full_response
            else:
                response = self.client.chat.completions.create(**api_params)
                # Handle tool calls if present
                result = self._handle_tool_calls(
                    response, available_functions, from_task, from_agent
                )

                # If no tool calls, return text content
                if result == response.choices[0].message.content:
                    result = response.choices[0].message.content or ""

            # Execute callbacks after API call
            if callbacks:
                for callback in callbacks:
                    if hasattr(callback, "on_llm_end"):
                        callback.on_llm_end(response=result)

            # Emit completion event
            completion_event_data = {
                "messages": formatted_messages,
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
            # Execute error callbacks
            if callbacks:
                for callback in callbacks:
                    if hasattr(callback, "on_llm_error"):
                        callback.on_llm_error(error=e)

            # Emit failed event
            failed_event_data = {
                "error": str(e),
            }
            if from_task is not None:
                failed_event_data["from_task"] = from_task
            if from_agent is not None:
                failed_event_data["from_agent"] = from_agent

            crewai_event_bus.emit(
                self,
                event=LLMCallFailedEvent(**failed_event_data),
            )

            raise RuntimeError(f"OpenAI API call failed: {str(e)}") from e

    def supports_stop_words(self) -> bool:
        """Check if OpenAI models support stop words."""
        return True

    def get_context_window_size(self) -> int:
        """Get the context window size for the current model."""
        if self.context_window_size != 0:
            return self.context_window_size

        # Use 85% of the context window like the original LLM class
        context_window = self.model_config.get("context_window", 4096)
        self.context_window_size = int(context_window * 0.85)
        return self.context_window_size

    def supports_function_calling(self) -> bool:
        """Check if the current model supports function calling."""
        return self.model_config.get("supports_tools", True)
