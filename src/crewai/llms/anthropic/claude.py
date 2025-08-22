import os
from typing import Any, Dict, List, Optional, Union, Type, Literal
from anthropic import Anthropic
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


class ClaudeLLM(BaseLLM):
    """Anthropic Claude LLM implementation with full LLM class compatibility."""

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        timeout: Optional[Union[float, int]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        n: Optional[int] = None,  # Not supported by Claude but kept for compatibility
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
        base_url: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,  # Not used by Anthropic
        api_key: Optional[str] = None,
        callbacks: List[Any] = [],
        reasoning_effort: Optional[
            Literal["none", "low", "medium", "high"]
        ] = None,  # Not used by Claude
        stream: bool = False,
        max_retries: int = 2,
        # Claude-specific parameters
        thinking_mode: bool = False,  # Enable Claude's thinking mode
        top_k: Optional[int] = None,  # Claude-specific sampling parameter
        **kwargs,
    ):
        """Initialize Claude LLM with full compatibility.

        Args:
            model: Claude model name (e.g., 'claude-3-5-sonnet-20241022')
            timeout: Request timeout in seconds
            temperature: Sampling temperature (0-1 for Claude)
            top_p: Nucleus sampling parameter
            n: Number of completions (not supported by Claude, kept for compatibility)
            stop: Stop sequences
            max_completion_tokens: Maximum tokens in completion
            max_tokens: Maximum tokens (legacy parameter)
            presence_penalty: Not supported by Claude, kept for compatibility
            frequency_penalty: Not supported by Claude, kept for compatibility
            logit_bias: Not supported by Claude, kept for compatibility
            response_format: Pydantic model for structured output
            seed: Not supported by Claude, kept for compatibility
            logprobs: Not supported by Claude, kept for compatibility
            top_logprobs: Not supported by Claude, kept for compatibility
            base_url: Custom API base URL
            api_base: Legacy API base parameter
            api_version: Not used by Anthropic
            api_key: Anthropic API key
            callbacks: List of callback functions
            reasoning_effort: Not used by Claude, kept for compatibility
            stream: Whether to stream responses
            max_retries: Number of retries for failed requests
            thinking_mode: Enable Claude's thinking mode (if supported)
            top_k: Claude-specific top-k sampling parameter
            **kwargs: Additional parameters
        """
        super().__init__(model=model, temperature=temperature)

        # Store all parameters for compatibility
        self.timeout = timeout
        self.top_p = top_p
        self.n = n  # Claude doesn't support n>1, but we store it for compatibility
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
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.callbacks = callbacks
        self.reasoning_effort = reasoning_effort
        self.stream = stream
        self.additional_params = kwargs
        self.context_window_size = 0

        # Claude-specific parameters
        self.thinking_mode = thinking_mode
        self.top_k = top_k

        # Normalize stop parameter to match LLM class behavior
        if stop is None:
            self.stop: List[str] = []
        elif isinstance(stop, str):
            self.stop = [stop]
        else:
            self.stop = stop

        # Initialize Anthropic client
        client_kwargs = {}
        if self.api_key:
            client_kwargs["api_key"] = self.api_key
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        if self.timeout:
            client_kwargs["timeout"] = self.timeout
        if max_retries:
            client_kwargs["max_retries"] = max_retries

        # Add any additional kwargs that might be relevant to the client
        for key, value in kwargs.items():
            if key not in ["thinking_mode", "top_k"]:  # Exclude our custom params
                client_kwargs[key] = value

        self.client = Anthropic(**client_kwargs)
        self.model_config = self._get_model_config()

    def _get_model_config(self) -> Dict[str, Any]:
        """Get model-specific configuration for Claude models."""
        # Claude model configurations based on Anthropic's documentation
        model_configs = {
            # Claude 3.5 Sonnet
            "claude-3-5-sonnet-20241022": {
                "context_window": 200000,
                "supports_tools": True,
                "supports_vision": True,
            },
            "claude-3-5-sonnet-20240620": {
                "context_window": 200000,
                "supports_tools": True,
                "supports_vision": True,
            },
            # Claude 3.5 Haiku
            "claude-3-5-haiku-20241022": {
                "context_window": 200000,
                "supports_tools": True,
                "supports_vision": True,
            },
            # Claude 3 Opus
            "claude-3-opus-20240229": {
                "context_window": 200000,
                "supports_tools": True,
                "supports_vision": True,
            },
            # Claude 3 Sonnet
            "claude-3-sonnet-20240229": {
                "context_window": 200000,
                "supports_tools": True,
                "supports_vision": True,
            },
            # Claude 3 Haiku
            "claude-3-haiku-20240307": {
                "context_window": 200000,
                "supports_tools": True,
                "supports_vision": True,
            },
            # Claude 2.1
            "claude-2.1": {
                "context_window": 200000,
                "supports_tools": False,
                "supports_vision": False,
            },
            "claude-2": {
                "context_window": 100000,
                "supports_tools": False,
                "supports_vision": False,
            },
            # Claude Instant
            "claude-instant-1.2": {
                "context_window": 100000,
                "supports_tools": False,
                "supports_vision": False,
            },
        }

        # Default config if model not found
        default_config = {
            "context_window": 200000,
            "supports_tools": True,
            "supports_vision": False,
        }

        # Try exact match first
        if self.model in model_configs:
            return model_configs[self.model]

        # Try prefix match for versioned models
        for model_prefix, config in model_configs.items():
            if self.model.startswith(model_prefix):
                return config

        return default_config

    def _format_messages(
        self, messages: Union[str, List[Dict[str, str]]]
    ) -> List[Dict[str, str]]:
        """Format messages for Anthropic API.

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

        # Claude requires alternating user/assistant messages and cannot start with system
        formatted_messages = []
        system_message = None

        for msg in messages:
            if msg["role"] == "system":
                # Store system message separately - Claude handles it differently
                if system_message is None:
                    system_message = msg["content"]
                else:
                    system_message += "\n\n" + msg["content"]
            else:
                formatted_messages.append(msg)

        # Ensure messages alternate and start with user
        if formatted_messages and formatted_messages[0]["role"] != "user":
            formatted_messages.insert(0, {"role": "user", "content": "Hello"})

        # Store system message for later use
        self._system_message = system_message

        return formatted_messages

    def _format_tools(self, tools: Optional[List[dict]]) -> Optional[List[dict]]:
        """Format tools for Claude function calling.

        Args:
            tools: List of tool definitions

        Returns:
            Claude-formatted tool definitions
        """
        if not tools or not self.model_config.get("supports_tools", True):
            return None

        formatted_tools = []
        for tool in tools:
            # Convert to Claude tool format
            formatted_tool = {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "input_schema": tool.get("parameters", {}),
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
        """Handle tool calls from Claude response.

        Args:
            response: Claude API response
            available_functions: Dict mapping function names to callables
            from_task: Optional task context
            from_agent: Optional agent context

        Returns:
            Result of function execution or error message
        """
        # Claude returns tool use in content blocks
        if not hasattr(response, "content") or not available_functions:
            return response.content[0].text if response.content else ""

        # Look for tool use blocks
        for content_block in response.content:
            if hasattr(content_block, "type") and content_block.type == "tool_use":
                function_name = content_block.name
                function_args = {}

                if function_name not in available_functions:
                    return f"Error: Function '{function_name}' not found in available functions"

                try:
                    # Claude provides arguments as a dict
                    function_args = content_block.input
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

        # If no tool calls, return text content
        return response.content[0].text if response.content else ""

    def call(
        self,
        messages: Union[str, List[Dict[str, str]]],
        tools: Optional[List[dict]] = None,
        callbacks: Optional[List[Any]] = None,
        available_functions: Optional[Dict[str, Any]] = None,
        from_task: Optional[Any] = None,
        from_agent: Optional[Any] = None,
    ) -> Union[str, Any]:
        """Call Claude API with the given messages.

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
        print("calling from native claude", messages)
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
            system_message = getattr(self, "_system_message", None)

            # Prepare API call parameters
            api_params = {
                "model": self.model,
                "messages": formatted_messages,
                "max_tokens": self.max_tokens or 4000,  # Claude requires max_tokens
            }

            # Add system message if present
            if system_message:
                api_params["system"] = system_message

            # Add optional parameters that Claude supports
            if self.temperature is not None:
                api_params["temperature"] = self.temperature

            if self.top_p is not None:
                api_params["top_p"] = self.top_p

            if self.top_k is not None:
                api_params["top_k"] = self.top_k

            if self.stop:
                api_params["stop_sequences"] = self.stop

            # Add tools if provided and supported
            formatted_tools = self._format_tools(tools)
            if formatted_tools:
                api_params["tools"] = formatted_tools

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
                response = self.client.messages.create(stream=True, **api_params)
                # Handle streaming (simplified implementation)
                full_response = ""
                try:
                    for event in response:
                        if hasattr(event, "type"):
                            if event.type == "content_block_delta":
                                if hasattr(event, "delta") and hasattr(
                                    event.delta, "text"
                                ):
                                    full_response += event.delta.text
                except Exception as e:
                    # If streaming fails, fall back to the response we have
                    print(f"Streaming error (continuing with partial response): {e}")
                result = full_response or "No response content"
            else:
                response = self.client.messages.create(**api_params)
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

            raise RuntimeError(f"Claude API call failed: {str(e)}") from e

    def supports_stop_words(self) -> bool:
        """Check if Claude models support stop words."""
        return True

    def get_context_window_size(self) -> int:
        """Get the context window size for the current model."""
        if self.context_window_size != 0:
            return self.context_window_size

        # Use 85% of the context window like the original LLM class
        context_window = self.model_config.get("context_window", 200000)
        self.context_window_size = int(context_window * 0.85)
        return self.context_window_size

    def supports_function_calling(self) -> bool:
        """Check if the current model supports function calling."""
        return self.model_config.get("supports_tools", True)

    def supports_vision(self) -> bool:
        """Check if the current model supports vision capabilities."""
        return self.model_config.get("supports_vision", False)
