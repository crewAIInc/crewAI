import json
import logging
import os
import sys
import threading
import warnings
from contextlib import contextmanager
from typing import Any, Dict, List, Literal, Optional, Type, Union, cast

from dotenv import load_dotenv
from pydantic import BaseModel

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    import litellm
    from litellm import Choices, get_supported_openai_params
    from litellm.types.utils import ModelResponse
    from litellm.utils import supports_response_schema


from crewai.utilities.exceptions.context_window_exceeding_exception import (
    LLMContextLengthExceededException,
)

load_dotenv()


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


class LLM:
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
            self.stop: List[str] = []
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
        ANTHROPIC_PREFIXES = ('anthropic/', 'claude-', 'claude/')
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
                        return result

                    except Exception as e:
                        logging.error(
                            f"Error executing function '{function_name}': {e}"
                        )
                        return text_response

                else:
                    logging.warning(
                        f"Tool call requested unknown function '{function_name}'"
                    )
                    return text_response

            except Exception as e:
                if not LLMContextLengthExceededException(
                    str(e)
                )._is_context_limit_error(str(e)):
                    logging.error(f"LiteLLM call failed: {str(e)}")
                raise

    def _format_messages_for_provider(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
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
                raise TypeError("Invalid message format. Each message must be a dict with 'role' and 'content' keys")
            
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
            return "response_format" in params
        except Exception as e:
            logging.error(f"Failed to get supported params: {str(e)}")
            return False

    def supports_stop_words(self) -> bool:
        try:
            params = get_supported_openai_params(model=self.model)
            return "stop" in params
        except Exception as e:
            logging.error(f"Failed to get supported params: {str(e)}")
            return False

    def get_context_window_size(self) -> int:
        """
        Returns the context window size, using 75% of the maximum to avoid
        cutting off messages mid-thread.
        """
        if self.context_window_size != 0:
            return self.context_window_size

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
