import json
import logging
import os
import sys
import threading
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, List, Literal, Optional, Type, Union, cast

from dotenv import load_dotenv
from pydantic import BaseModel

from crewai.utilities.events.llm_events import (
    LLMCallCompletedEvent,
    LLMCallFailedEvent,
    LLMCallStartedEvent,
    LLMCallType,
)
from crewai.utilities.events.tool_usage_events import ToolExecutionErrorEvent

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    import litellm
    from litellm import Choices
    from litellm.types.utils import ModelResponse
    from litellm.utils import get_supported_openai_params, supports_response_schema


from crewai.utilities.events import crewai_event_bus
from crewai.utilities.exceptions.context_window_exceeding_exception import (
    LLMContextLengthExceededException,
)

load_dotenv()


class LLM(ABC):
    """Base class for LLM implementations.
    
    This class defines the interface that all LLM implementations must follow.
    Users can extend this class to create custom LLM implementations that don't
    rely on litellm's authentication mechanism.
    
    Custom LLM implementations should handle error cases gracefully, including
    timeouts, authentication failures, and malformed responses. They should also
    implement proper validation for input parameters and provide clear error
    messages when things go wrong.
    
    Attributes:
        stop (list): A list of stop sequences that the LLM should use to stop generation.
            This is used by the CrewAgentExecutor and other components.
    """
    
    def __new__(cls, *args, **kwargs):
        """Create a new LLM instance.
        
        This method handles backward compatibility by creating a DefaultLLM instance
        when the LLM class is instantiated directly with parameters.
        
        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
            
        Returns:
            Either a new LLM instance or a DefaultLLM instance for backward compatibility.
        """
        if cls is LLM and (args or kwargs.get('model') is not None):
            from crewai.llm import DefaultLLM
            return DefaultLLM(*args, **kwargs)
        return super().__new__(cls)
    
    def __init__(self):
        """Initialize the LLM with default attributes.
        
        This constructor sets default values for attributes that are expected
        by the CrewAgentExecutor and other components.
        
        All custom LLM implementations should call super().__init__() to ensure
        that these default attributes are properly initialized.
        """
        self.stop = []
    
    @classmethod
    def create(
        cls,
        model: str,
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
        **kwargs,
    ) -> 'DefaultLLM':
        """Create a default LLM instance using litellm.
        
        This factory method creates a default LLM instance using litellm as the backend.
        It's the recommended way to create LLM instances for most users.
        
        Args:
            model: The model name (e.g., "gpt-4").
            timeout: Optional timeout for the LLM call.
            temperature: Optional temperature for the LLM call.
            top_p: Optional top_p for the LLM call.
            n: Optional n for the LLM call.
            stop: Optional stop sequences for the LLM call.
            max_completion_tokens: Optional max_completion_tokens for the LLM call.
            max_tokens: Optional max_tokens for the LLM call.
            presence_penalty: Optional presence_penalty for the LLM call.
            frequency_penalty: Optional frequency_penalty for the LLM call.
            logit_bias: Optional logit_bias for the LLM call.
            response_format: Optional response_format for the LLM call.
            seed: Optional seed for the LLM call.
            logprobs: Optional logprobs for the LLM call.
            top_logprobs: Optional top_logprobs for the LLM call.
            base_url: Optional base_url for the LLM call.
            api_base: Optional api_base for the LLM call.
            api_version: Optional api_version for the LLM call.
            api_key: Optional api_key for the LLM call.
            callbacks: Optional callbacks for the LLM call.
            reasoning_effort: Optional reasoning_effort for the LLM call.
            **kwargs: Additional keyword arguments for the LLM call.
            
        Returns:
            A DefaultLLM instance configured with the provided parameters.
        """
        from crewai.llm import DefaultLLM
        
        return DefaultLLM(
            model=model,
            timeout=timeout,
            temperature=temperature,
            top_p=top_p,
            n=n,
            stop=stop,
            max_completion_tokens=max_completion_tokens,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            response_format=response_format,
            seed=seed,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            base_url=base_url,
            api_base=api_base,
            api_version=api_version,
            api_key=api_key,
            callbacks=callbacks,
            reasoning_effort=reasoning_effort,
            **kwargs,
        )
    
    def call(
        self,
        messages: Union[str, List[Dict[str, str]]],
        tools: Optional[List[dict]] = None,
        callbacks: Optional[List[Any]] = None,
        available_functions: Optional[Dict[str, Any]] = None,
    ) -> Union[str, Any]:
        """Call the LLM with the given messages.
        
        Args:
            messages: Input messages for the LLM.
                     Can be a string or list of message dictionaries.
                     If string, it will be converted to a single user message.
                     If list, each dict must have 'role' and 'content' keys.
            tools: Optional list of tool schemas for function calling.
                  Each tool should define its name, description, and parameters.
            callbacks: Optional list of callback functions to be executed
                      during and after the LLM call.
            available_functions: Optional dict mapping function names to callables
                               that can be invoked by the LLM.
            
        Returns:
            Either a text response from the LLM (str) or
            the result of a tool function call (Any).
            
        Raises:
            ValueError: If the messages format is invalid.
            TimeoutError: If the LLM request times out.
            RuntimeError: If the LLM request fails for other reasons.
            NotImplementedError: If this method is not implemented by a subclass.
        """
        raise NotImplementedError("Subclasses must implement call()")
        
    def supports_function_calling(self) -> bool:
        """Check if the LLM supports function calling.
        
        This method should return True if the LLM implementation supports
        function calling (tools), and False otherwise. If this method returns
        True, the LLM should be able to handle the 'tools' parameter in the
        call() method.
        
        Returns:
            True if the LLM supports function calling, False otherwise.
            
        Raises:
            NotImplementedError: If this method is not implemented by a subclass.
        """
        raise NotImplementedError("Subclasses must implement supports_function_calling()")
        
    def supports_stop_words(self) -> bool:
        """Check if the LLM supports stop words.
        
        This method should return True if the LLM implementation supports
        stop words, and False otherwise. If this method returns True, the
        LLM should respect the 'stop' attribute when generating responses.
        
        Returns:
            True if the LLM supports stop words, False otherwise.
            
        Raises:
            NotImplementedError: If this method is not implemented by a subclass.
        """
        raise NotImplementedError("Subclasses must implement supports_stop_words()")
        
    def get_context_window_size(self) -> int:
        """Get the context window size of the LLM.
        
        This method should return the maximum number of tokens that the LLM
        can process in a single request. This is used by CrewAI to ensure
        that messages don't exceed the LLM's context window.
        
        Returns:
            The context window size as an integer.
            
        Raises:
            NotImplementedError: If this method is not implemented by a subclass.
        """
        raise NotImplementedError("Subclasses must implement get_context_window_size()")


class FilteredStream:
    def __init__(self, original_stream):
        self._original_stream = original_stream
        self._lock = threading.Lock()

    def write(self, s) -> int:
        with self._lock:
            # Filter out extraneous messages from LiteLLM
            if (
                "Give Feedback / Get Help: https://github.com/BerriAI/litellm/issues/new"
                in s
                or "LiteLLM.Info: If you need to debug this error, use `litellm.set_verbose=True`"
                in s
            ):
                return 0
            return self._original_stream.write(s)

    def flush(self):
        with self._lock:
            return self._original_stream.flush()


LLM_CONTEXT_WINDOW_SIZES = {
    # openai
    "gpt-4": 8192,
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4-turbo": 128000,
    "o1-preview": 128000,
    "o1-mini": 128000,
    "o3-mini": 200000,  # Based on official o3-mini specifications
    # gemini
    "gemini-2.0-flash": 1048576,
    "gemini-1.5-pro": 2097152,
    "gemini-1.5-flash": 1048576,
    "gemini-1.5-flash-8b": 1048576,
    # deepseek
    "deepseek-chat": 128000,
    # groq
    "gemma2-9b-it": 8192,
    "gemma-7b-it": 8192,
    "llama3-groq-70b-8192-tool-use-preview": 8192,
    "llama3-groq-8b-8192-tool-use-preview": 8192,
    "llama-3.1-70b-versatile": 131072,
    "llama-3.1-8b-instant": 131072,
    "llama-3.2-1b-preview": 8192,
    "llama-3.2-3b-preview": 8192,
    "llama-3.2-11b-text-preview": 8192,
    "llama-3.2-90b-text-preview": 8192,
    "llama3-70b-8192": 8192,
    "llama3-8b-8192": 8192,
    "mixtral-8x7b-32768": 32768,
    "llama-3.3-70b-versatile": 128000,
    "llama-3.3-70b-instruct": 128000,
    # sambanova
    "Meta-Llama-3.3-70B-Instruct": 131072,
    "QwQ-32B-Preview": 8192,
    "Qwen2.5-72B-Instruct": 8192,
    "Qwen2.5-Coder-32B-Instruct": 8192,
    "Meta-Llama-3.1-405B-Instruct": 8192,
    "Meta-Llama-3.1-70B-Instruct": 131072,
    "Meta-Llama-3.1-8B-Instruct": 131072,
    "Llama-3.2-90B-Vision-Instruct": 16384,
    "Llama-3.2-11B-Vision-Instruct": 16384,
    "Meta-Llama-3.2-3B-Instruct": 4096,
    "Meta-Llama-3.2-1B-Instruct": 16384,
}

DEFAULT_CONTEXT_WINDOW_SIZE = 8192
CONTEXT_WINDOW_USAGE_RATIO = 0.75


@contextmanager
def suppress_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        warnings.filterwarnings(
            "ignore", message="open_text is deprecated*", category=DeprecationWarning
        )

        # Redirect stdout and stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = FilteredStream(old_stdout)
        sys.stderr = FilteredStream(old_stderr)
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


class DefaultLLM(LLM):
    """Default LLM implementation using litellm.
    
    This class provides a concrete implementation of the LLM interface
    using litellm as the backend. It's the default implementation used
    by CrewAI when no custom LLM is provided.
    """
    
    def __init__(
        self,
        model: str,
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
        **kwargs,
    ):
        super().__init__()  # Initialize the base class
        
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.top_p = top_p
        self.n = n
        self.max_completion_tokens = max_completion_tokens
        self.max_tokens = max_tokens
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.logit_bias = logit_bias
        self.response_format = response_format
        self.seed = seed
        self.logprobs = logprobs
        self.top_logprobs = top_logprobs
        self.base_url = base_url
        self.api_base = api_base
        self.api_version = api_version
        self.api_key = api_key
        self.callbacks = callbacks
        self.context_window_size = 0
        self.reasoning_effort = reasoning_effort
        self.additional_params = kwargs
        self.is_anthropic = self._is_anthropic_model(model)

        litellm.drop_params = True

        # Normalize self.stop to always be a List[str]
        if stop is None:
            self.stop = []  # Already initialized in base class
        elif isinstance(stop, str):
            self.stop = [stop]
        else:
            self.stop = stop

        self.set_callbacks(callbacks)
        self.set_env_callbacks()

    def _is_anthropic_model(self, model: str) -> bool:
        """Determine if the model is from Anthropic provider.

        Args:
            model: The model identifier string.

        Returns:
            bool: True if the model is from Anthropic, False otherwise.
        """
        ANTHROPIC_PREFIXES = ("anthropic/", "claude-", "claude/")
        return any(prefix in model.lower() for prefix in ANTHROPIC_PREFIXES)

    def call(
        self,
        messages: Union[str, List[Dict[str, str]]],
        tools: Optional[List[dict]] = None,
        callbacks: Optional[List[Any]] = None,
        available_functions: Optional[Dict[str, Any]] = None,
    ) -> Union[str, Any]:
        """High-level LLM call method.

        Args:
            messages: Input messages for the LLM.
                     Can be a string or list of message dictionaries.
                     If string, it will be converted to a single user message.
                     If list, each dict must have 'role' and 'content' keys.
            tools: Optional list of tool schemas for function calling.
                  Each tool should define its name, description, and parameters.
            callbacks: Optional list of callback functions to be executed
                      during and after the LLM call.
            available_functions: Optional dict mapping function names to callables
                               that can be invoked by the LLM.

        Returns:
            Union[str, Any]: Either a text response from the LLM (str) or
                           the result of a tool function call (Any).

        Raises:
            TypeError: If messages format is invalid
            ValueError: If response format is not supported
            LLMContextLengthExceededException: If input exceeds model's context limit

        Examples:
            # Example 1: Simple string input
            >>> response = llm.call("Return the name of a random city.")
            >>> print(response)
            "Paris"

            # Example 2: Message list with system and user messages
            >>> messages = [
            ...     {"role": "system", "content": "You are a geography expert"},
            ...     {"role": "user", "content": "What is France's capital?"}
            ... ]
            >>> response = llm.call(messages)
            >>> print(response)
            "The capital of France is Paris."
        """
        crewai_event_bus.emit(
            self,
            event=LLMCallStartedEvent(
                messages=messages,
                tools=tools,
                callbacks=callbacks,
                available_functions=available_functions,
            ),
        )
        # Validate parameters before proceeding with the call.
        self._validate_call_params()

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        # For O1 models, system messages are not supported.
        # Convert any system messages into assistant messages.
        if "o1" in self.model.lower():
            for message in messages:
                if message.get("role") == "system":
                    message["role"] = "assistant"

        with suppress_warnings():
            if callbacks and len(callbacks) > 0:
                self.set_callbacks(callbacks)

            try:
                # --- 1) Format messages according to provider requirements
                formatted_messages = self._format_messages_for_provider(messages)

                # --- 2) Prepare the parameters for the completion call
                params = {
                    "model": self.model,
                    "messages": formatted_messages,
                    "timeout": self.timeout,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "n": self.n,
                    "stop": self.stop,
                    "max_tokens": self.max_tokens or self.max_completion_tokens,
                    "presence_penalty": self.presence_penalty,
                    "frequency_penalty": self.frequency_penalty,
                    "logit_bias": self.logit_bias,
                    "response_format": self.response_format,
                    "seed": self.seed,
                    "logprobs": self.logprobs,
                    "top_logprobs": self.top_logprobs,
                    "api_base": self.api_base,
                    "base_url": self.base_url,
                    "api_version": self.api_version,
                    "api_key": self.api_key,
                    "stream": False,
                    "tools": tools,
                    "reasoning_effort": self.reasoning_effort,
                    **self.additional_params,
                }

                # Remove None values from params
                params = {k: v for k, v in params.items() if v is not None}

                # --- 2) Make the completion call
                response = litellm.completion(**params)
                response_message = cast(Choices, cast(ModelResponse, response).choices)[
                    0
                ].message
                text_response = response_message.content or ""
                tool_calls = getattr(response_message, "tool_calls", [])

                # --- 3) Handle callbacks with usage info
                if callbacks and len(callbacks) > 0:
                    for callback in callbacks:
                        if hasattr(callback, "log_success_event"):
                            usage_info = getattr(response, "usage", None)
                            if usage_info:
                                callback.log_success_event(
                                    kwargs=params,
                                    response_obj={"usage": usage_info},
                                    start_time=0,
                                    end_time=0,
                                )

                # --- 4) If no tool calls, return the text response
                if not tool_calls or not available_functions:
                    self._handle_emit_call_events(text_response, LLMCallType.LLM_CALL)
                    return text_response

                # --- 5) Handle the tool call
                tool_call = tool_calls[0]
                function_name = tool_call.function.name

                if function_name in available_functions:
                    try:
                        function_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError as e:
                        logging.warning(f"Failed to parse function arguments: {e}")
                        return text_response

                    fn = available_functions[function_name]
                    try:
                        # Call the actual tool function
                        result = fn(**function_args)
                        self._handle_emit_call_events(result, LLMCallType.TOOL_CALL)
                        return result

                    except Exception as e:
                        logging.error(
                            f"Error executing function '{function_name}': {e}"
                        )
                        crewai_event_bus.emit(
                            self,
                            event=ToolExecutionErrorEvent(
                                tool_name=function_name,
                                tool_args=function_args,
                                tool_class=fn,
                                error=str(e),
                            ),
                        )
                        crewai_event_bus.emit(
                            self,
                            event=LLMCallFailedEvent(
                                error=f"Tool execution error: {str(e)}"
                            ),
                        )
                        return text_response

                else:
                    logging.warning(
                        f"Tool call requested unknown function '{function_name}'"
                    )
                    return text_response

            except Exception as e:
                crewai_event_bus.emit(
                    self,
                    event=LLMCallFailedEvent(error=str(e)),
                )
                if not LLMContextLengthExceededException(
                    str(e)
                )._is_context_limit_error(str(e)):
                    logging.error(f"LiteLLM call failed: {str(e)}")
                raise

    def _handle_emit_call_events(self, response: Any, call_type: LLMCallType):
        """Handle the events for the LLM call.

        Args:
            response (str): The response from the LLM call.
            call_type (str): The type of call, either "tool_call" or "llm_call".
        """
        crewai_event_bus.emit(
            self,
            event=LLMCallCompletedEvent(response=response, call_type=call_type),
        )

    def _format_messages_for_provider(
        self, messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Format messages according to provider requirements.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
                     Can be empty or None.

        Returns:
            List of formatted messages according to provider requirements.
            For Anthropic models, ensures first message has 'user' role.

        Raises:
            TypeError: If messages is None or contains invalid message format.
        """
        if messages is None:
            raise TypeError("Messages cannot be None")

        # Validate message format first
        for msg in messages:
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                raise TypeError(
                    "Invalid message format. Each message must be a dict with 'role' and 'content' keys"
                )

        if not self.is_anthropic:
            return messages

        # Anthropic requires messages to start with 'user' role
        if not messages or messages[0]["role"] == "system":
            # If first message is system or empty, add a placeholder user message
            return [{"role": "user", "content": "."}, *messages]

        return messages

    def _get_custom_llm_provider(self) -> str:
        """
        Derives the custom_llm_provider from the model string.
        - For example, if the model is "openrouter/deepseek/deepseek-chat", returns "openrouter".
        - If the model is "gemini/gemini-1.5-pro", returns "gemini".
        - If there is no '/', defaults to "openai".
        """
        if "/" in self.model:
            return self.model.split("/")[0]
        return "openai"

    def _validate_call_params(self) -> None:
        """
        Validate parameters before making a call. Currently this only checks if
        a response_format is provided and whether the model supports it.
        The custom_llm_provider is dynamically determined from the model:
          - E.g., "openrouter/deepseek/deepseek-chat" yields "openrouter"
          - "gemini/gemini-1.5-pro" yields "gemini"
          - If no slash is present, "openai" is assumed.
        """
        provider = self._get_custom_llm_provider()
        if self.response_format is not None and not supports_response_schema(
            model=self.model,
            custom_llm_provider=provider,
        ):
            raise ValueError(
                f"The model {self.model} does not support response_format for provider '{provider}'. "
                "Please remove response_format or use a supported model."
            )

    def supports_function_calling(self) -> bool:
        try:
            params = get_supported_openai_params(model=self.model)
            return params is not None and "tools" in params
        except Exception as e:
            logging.error(f"Failed to get supported params: {str(e)}")
            return False

    def supports_stop_words(self) -> bool:
        try:
            params = get_supported_openai_params(model=self.model)
            return params is not None and "stop" in params
        except Exception as e:
            logging.error(f"Failed to get supported params: {str(e)}")
            return False

    def get_context_window_size(self) -> int:
        """
        Returns the context window size, using 75% of the maximum to avoid
        cutting off messages mid-thread.

        Raises:
            ValueError: If a model's context window size is outside valid bounds (1024-2097152)
        """
        if self.context_window_size != 0:
            return self.context_window_size

        MIN_CONTEXT = 1024
        MAX_CONTEXT = 2097152  # Current max from gemini-1.5-pro

        # Validate all context window sizes
        for key, value in LLM_CONTEXT_WINDOW_SIZES.items():
            if value < MIN_CONTEXT or value > MAX_CONTEXT:
                raise ValueError(
                    f"Context window for {key} must be between {MIN_CONTEXT} and {MAX_CONTEXT}"
                )

        self.context_window_size = int(
            DEFAULT_CONTEXT_WINDOW_SIZE * CONTEXT_WINDOW_USAGE_RATIO
        )
        for key, value in LLM_CONTEXT_WINDOW_SIZES.items():
            if self.model.startswith(key):
                self.context_window_size = int(value * CONTEXT_WINDOW_USAGE_RATIO)
        return self.context_window_size

    def set_callbacks(self, callbacks: List[Any]):
        """
        Attempt to keep a single set of callbacks in litellm by removing old
        duplicates and adding new ones.
        """
        with suppress_warnings():
            callback_types = [type(callback) for callback in callbacks]
            for callback in litellm.success_callback[:]:
                if type(callback) in callback_types:
                    litellm.success_callback.remove(callback)

            for callback in litellm._async_success_callback[:]:
                if type(callback) in callback_types:
                    litellm._async_success_callback.remove(callback)

            litellm.callbacks = callbacks

    def set_env_callbacks(self):
        """
        Sets the success and failure callbacks for the LiteLLM library from environment variables.

        This method reads the `LITELLM_SUCCESS_CALLBACKS` and `LITELLM_FAILURE_CALLBACKS`
        environment variables, which should contain comma-separated lists of callback names.
        It then assigns these lists to `litellm.success_callback` and `litellm.failure_callback`,
        respectively.

        If the environment variables are not set or are empty, the corresponding callback lists
        will be set to empty lists.

        Example:
            LITELLM_SUCCESS_CALLBACKS="langfuse,langsmith"
            LITELLM_FAILURE_CALLBACKS="langfuse"

        This will set `litellm.success_callback` to ["langfuse", "langsmith"] and
        `litellm.failure_callback` to ["langfuse"].
        """
        with suppress_warnings():
            success_callbacks_str = os.environ.get("LITELLM_SUCCESS_CALLBACKS", "")
            success_callbacks = []
            if success_callbacks_str:
                success_callbacks = [
                    cb.strip() for cb in success_callbacks_str.split(",") if cb.strip()
                ]

            failure_callbacks_str = os.environ.get("LITELLM_FAILURE_CALLBACKS", "")
            failure_callbacks = []
            if failure_callbacks_str:
                failure_callbacks = [
                    cb.strip() for cb in failure_callbacks_str.split(",") if cb.strip()
                ]

                litellm.success_callback = success_callbacks
                litellm.failure_callback = failure_callbacks


class BaseLLM(LLM):
    """Deprecated: Use LLM instead.
    
    This class is kept for backward compatibility and will be removed in a future release.
    It inherits from LLM and provides the same interface, but emits a deprecation warning
    when instantiated.
    """
    
    def __init__(self):
        """Initialize the BaseLLM with a deprecation warning.
        
        This constructor emits a deprecation warning and then calls the parent class's
        constructor to initialize the LLM.
        """
        import warnings
        warnings.warn(
            "BaseLLM is deprecated and will be removed in a future release. "
            "Use LLM instead for custom implementations.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__()
