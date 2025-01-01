import logging
import os
import sys
import threading
import warnings
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    import litellm
    from litellm import get_supported_openai_params

from crewai.utilities.exceptions.context_window_exceeding_exception import (
    LLMContextLengthExceededException,
)


class FilteredStream:
    def __init__(self, original_stream):
        self._original_stream = original_stream
        self._lock = threading.Lock()

    def write(self, s) -> int:
        with self._lock:
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
}

DEFAULT_CONTEXT_WINDOW_SIZE = 8192
CONTEXT_WINDOW_USAGE_RATIO = 0.75


@contextmanager
def suppress_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        # Redirect stdout and stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = FilteredStream(old_stdout)
        sys.stderr = FilteredStream(old_stderr)

        try:
            yield
        finally:
            # Restore stdout and stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr


class LLM(BaseModel):
    model: str = "gpt-4"  # Set default model
    timeout: Optional[Union[float, int]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    max_completion_tokens: Optional[int] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Dict[int, float]] = None
    response_format: Optional[Dict[str, Any]] = None
    seed: Optional[int] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    base_url: Optional[str] = None
    api_version: Optional[str] = None
    api_key: Optional[str] = None
    callbacks: Optional[List[Any]] = None
    context_window_size: Optional[int] = None
    kwargs: Dict[str, Any] = Field(default_factory=dict)
    logger: Optional[logging.Logger] = Field(default_factory=lambda: logging.getLogger(__name__))

    def __init__(
        self,
        model: Optional[Union[str, 'LLM']] = "gpt-4",
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
        response_format: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        base_url: Optional[str] = None,
        api_version: Optional[str] = None,
        api_key: Optional[str] = None,
        callbacks: Optional[List[Any]] = None,
        context_window_size: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        # Initialize with default values
        init_dict = {
            "model": model if isinstance(model, str) else "gpt-4",
            "timeout": timeout,
            "temperature": temperature,
            "top_p": top_p,
            "n": n,
            "stop": stop,
            "max_completion_tokens": max_completion_tokens,
            "max_tokens": max_tokens,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "logit_bias": logit_bias,
            "response_format": response_format,
            "seed": seed,
            "logprobs": logprobs,
            "top_logprobs": top_logprobs,
            "base_url": base_url,
            "api_version": api_version,
            "api_key": api_key,
            "callbacks": callbacks,
            "context_window_size": context_window_size,
            "kwargs": kwargs,
        }
        super().__init__(**init_dict)
        
        # Initialize model with default value
        self.model = "gpt-4"  # Default fallback
        
        # Extract and validate model name
        if isinstance(model, LLM):
            # Extract and validate model name from LLM instance
            if hasattr(model, 'model'):
                if isinstance(model.model, str):
                    self.model = model.model
                else:
                    # Try to extract string model name from nested LLM
                    if isinstance(model.model, LLM):
                        self.model = str(model.model.model) if hasattr(model.model, 'model') else "gpt-4"
                    else:
                        self.model = "gpt-4"
                        if self.logger:
                            self.logger.warning("Nested LLM model is not a string, using default: gpt-4")
            else:
                self.model = "gpt-4"
                if self.logger:
                    self.logger.warning("LLM instance has no model attribute, using default: gpt-4")
        else:
            # Extract and validate model name for non-LLM instances
            if not isinstance(model, str):
                if self.logger:
                    self.logger.debug(f"Model is not a string, attempting to extract name. Type: {type(model)}")
                if model is not None:
                    if hasattr(model, 'model_name'):
                        model_name = getattr(model, 'model_name', None)
                        self.model = str(model_name) if model_name is not None else "gpt-4"
                    elif hasattr(model, 'model'):
                        model_attr = getattr(model, 'model', None)
                        self.model = str(model_attr) if model_attr is not None else "gpt-4"
                    elif hasattr(model, '_model_name'):
                        model_name = getattr(model, '_model_name', None)
                        self.model = str(model_name) if model_name is not None else "gpt-4"
                    else:
                        self.model = "gpt-4"  # Default fallback
                        if self.logger:
                            self.logger.warning(f"Could not extract model name from {type(model)}, using default: {self.model}")
                else:
                    self.model = "gpt-4"  # Default fallback for None
                    if self.logger:
                        self.logger.warning("Model is None, using default: gpt-4")
            else:
                self.model = str(model)  # Ensure it's a string

        # If model is an LLM instance, copy its configuration
        if isinstance(model, LLM):
            # Extract and validate model name first
            if hasattr(model, 'model'):
                if isinstance(model.model, str):
                    self.model = model.model
                else:
                    # Try to extract string model name from nested LLM
                    if isinstance(model.model, LLM):
                        self.model = str(model.model.model) if hasattr(model.model, 'model') else "gpt-4"
                    else:
                        self.model = "gpt-4"
                        if self.logger:
                            self.logger.warning("Nested LLM model is not a string, using default: gpt-4")
            else:
                self.model = "gpt-4"
                if self.logger:
                    self.logger.warning("LLM instance has no model attribute, using default: gpt-4")

            # Copy other configuration
            self.timeout = model.timeout
            self.temperature = model.temperature
            self.top_p = model.top_p
            self.n = model.n
            self.stop = model.stop
            self.max_completion_tokens = model.max_completion_tokens
            self.max_tokens = model.max_tokens
            self.presence_penalty = model.presence_penalty
            self.frequency_penalty = model.frequency_penalty
            self.logit_bias = model.logit_bias
            self.response_format = model.response_format
            self.seed = model.seed
            self.logprobs = model.logprobs
            self.top_logprobs = model.top_logprobs
            self.base_url = model.base_url
            self.api_version = model.api_version
            self.api_key = model.api_key
            self.callbacks = model.callbacks
            self.context_window_size = model.context_window_size
            self.kwargs = model.kwargs

            # Final validation of model name
            if not isinstance(self.model, str):
                self.model = "gpt-4"
                if self.logger:
                    self.logger.warning("Model name is still not a string after LLM copy, using default: gpt-4")
        else:
            # Extract and validate model name for non-LLM instances
            if not isinstance(model, str):
                if self.logger:
                    self.logger.debug(f"Model is not a string, attempting to extract name. Type: {type(model)}")
                if model is not None:
                    if hasattr(model, 'model_name'):
                        model_name = getattr(model, 'model_name', None)
                        self.model = str(model_name) if model_name is not None else "gpt-4"
                    elif hasattr(model, 'model'):
                        model_attr = getattr(model, 'model', None)
                        self.model = str(model_attr) if model_attr is not None else "gpt-4"
                    elif hasattr(model, '_model_name'):
                        model_name = getattr(model, '_model_name', None)
                        self.model = str(model_name) if model_name is not None else "gpt-4"
                    else:
                        self.model = "gpt-4"  # Default fallback
                        if self.logger:
                            self.logger.warning(f"Could not extract model name from {type(model)}, using default: {self.model}")
                else:
                    self.model = "gpt-4"  # Default fallback for None
                    if self.logger:
                        self.logger.warning("Model is None, using default: gpt-4")
            else:
                self.model = str(model)  # Ensure it's a string

            # Final validation
            if not isinstance(self.model, str):
                self.model = "gpt-4"
                if self.logger:
                    self.logger.warning("Model name is still not a string after extraction, using default: gpt-4")

            self.timeout = timeout
            self.temperature = temperature
            self.top_p = top_p
            self.n = n
            self.stop = stop
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
            self.api_version = api_version
            self.api_key = api_key
            self.callbacks = callbacks
            self.context_window_size = 0
            self.kwargs = kwargs

            # Ensure model is a string after initialization
            if not isinstance(self.model, str):
                self.model = "gpt-4"
                self.logger.warning(f"Model is still not a string after initialization, using default: {self.model}")

        litellm.drop_params = True

        self.set_callbacks(callbacks)
        self.set_env_callbacks()

    def call(
        self,
        messages: List[Dict[str, str]],
        callbacks: Optional[List[Any]] = None
    ) -> str:
        with suppress_warnings():
            if callbacks and len(callbacks) > 0:
                self.set_callbacks(callbacks)

            # Store original model to restore later
            original_model = self.model

            try:
                # Ensure model is a string before making the call
                if not isinstance(self.model, str):
                    if self.logger:
                        self.logger.warning(f"Model is not a string in call method: {type(self.model)}. Attempting to convert...")
                    if isinstance(self.model, LLM):
                        self.model = self.model.model if isinstance(self.model.model, str) else "gpt-4"
                    elif hasattr(self.model, 'model_name'):
                        self.model = str(self.model.model_name)
                    elif hasattr(self.model, 'model'):
                        if isinstance(self.model.model, str):
                            self.model = str(self.model.model)
                        elif hasattr(self.model.model, 'model_name'):
                            self.model = str(self.model.model.model_name)
                        else:
                            self.model = "gpt-4"
                            if self.logger:
                                self.logger.warning("Could not extract model name from nested model object, using default: gpt-4")
                    else:
                        self.model = "gpt-4"
                        if self.logger:
                            self.logger.warning("Could not extract model name, using default: gpt-4")
                
                if self.logger:
                    self.logger.debug(f"Using model: {self.model} (type: {type(self.model)}) for LiteLLM call")

                # Create base params with validated model name
                # Extract model name string
                model_name = None
                if isinstance(self.model, str):
                    model_name = self.model
                elif hasattr(self.model, 'model_name'):
                    model_name = str(self.model.model_name)
                elif hasattr(self.model, 'model'):
                    if isinstance(self.model.model, str):
                        model_name = str(self.model.model)
                    elif hasattr(self.model.model, 'model_name'):
                        model_name = str(self.model.model.model_name)
                
                if not model_name:
                    model_name = "gpt-4"
                    if self.logger:
                        self.logger.warning("Could not extract model name, using default: gpt-4")

                params = {
                    "model": model_name,
                    "messages": messages,
                    "stream": False,
                    "api_key": self.api_key or os.getenv("OPENAI_API_KEY"),
                    "api_base": self.base_url,
                    "api_version": self.api_version,
                }
                
                if self.logger:
                    self.logger.debug(f"Using model parameters: {params}")

                # Add API configuration if available
                api_key = self.api_key or os.getenv("OPENAI_API_KEY")
                if api_key:
                    params["api_key"] = api_key

                # Try to get supported parameters for the model
                try:
                    supported_params = get_supported_openai_params(self.model)
                    optional_params = {}
                    
                    if supported_params:
                        param_mapping = {
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
                            "top_logprobs": self.top_logprobs
                        }
                        
                        # Only add parameters that are supported and not None
                        optional_params = {
                            k: v for k, v in param_mapping.items()
                            if k in supported_params and v is not None
                        }
                        if "logprobs" in supported_params and self.logprobs is not None:
                            optional_params["logprobs"] = self.logprobs
                        if "top_logprobs" in supported_params and self.top_logprobs is not None:
                            optional_params["top_logprobs"] = self.top_logprobs
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Failed to get supported params for model {self.model}: {str(e)}")
                    # If we can't get supported params, just add non-None parameters
                    param_mapping = {
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
                        "top_logprobs": self.top_logprobs
                    }
                    optional_params = {k: v for k, v in param_mapping.items() if v is not None}

                # Update params with optional parameters
                params.update(optional_params)
                
                # Add API endpoint configuration if available
                if self.base_url:
                    params["api_base"] = self.base_url
                if self.api_version:
                    params["api_version"] = self.api_version

                # Final validation of model parameter
                if not isinstance(params["model"], str):
                    if self.logger:
                        self.logger.error(f"Model is still not a string after all conversions: {type(params['model'])}")
                    params["model"] = "gpt-4"
                
                # Update params with non-None optional parameters
                params.update({k: v for k, v in optional_params.items() if v is not None})
                
                # Add any additional kwargs
                if self.kwargs:
                    params.update(self.kwargs)

                # Remove None values to avoid passing unnecessary parameters
                params = {k: v for k, v in params.items() if v is not None}

                response = litellm.completion(**params)
                content = response["choices"][0]["message"]["content"]
                
                # Extract usage metrics
                usage = response.get("usage", {})
                if callbacks:
                    for callback in callbacks:
                        if hasattr(callback, "update_token_usage"):
                            callback.update_token_usage(usage)
                
                return content
            except Exception as e:
                if not LLMContextLengthExceededException(
                    str(e)
                )._is_context_limit_error(str(e)):
                    logging.error(f"LiteLLM call failed: {str(e)}")
                raise  # Re-raise the exception after logging
            finally:
                # Always restore the original model object
                self.model = original_model

    def supports_function_calling(self) -> bool:
        """Check if the LLM supports function calling.
        
        Returns:
            bool: True if the model supports function calling, False otherwise
        """
        try:
            params = get_supported_openai_params(model=self.model)
            return "response_format" in params
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to get supported params: {str(e)}")
            return False

    def supports_stop_words(self) -> bool:
        """Check if the LLM supports stop words.
        Returns False if the LLM is not properly initialized."""
        if not hasattr(self, 'model') or self.model is None:
            return False
        try:
            params = get_supported_openai_params(model=self.model)
            return "stop" in params
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to get supported params: {str(e)}")
            return False

    def get_context_window_size(self) -> int:
        """Get the context window size for the current model.
        
        Returns:
            int: The context window size in tokens
        """
        # Only using 75% of the context window size to avoid cutting the message in the middle
        if self.context_window_size is not None and self.context_window_size != 0:
            return int(self.context_window_size)

        window_size = DEFAULT_CONTEXT_WINDOW_SIZE
        if isinstance(self.model, str):
            for key, value in LLM_CONTEXT_WINDOW_SIZES.items():
                if self.model.startswith(key):
                    window_size = value
                    break

        self.context_window_size = int(window_size * CONTEXT_WINDOW_USAGE_RATIO)
        return self.context_window_size

    def set_callbacks(self, callbacks: Optional[List[Any]] = None) -> None:
        """Set callbacks for the LLM.
        
        Args:
            callbacks: Optional list of callback functions. If None, no callbacks will be set.
        """
        if callbacks is not None:
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
        success_callbacks_str = os.environ.get("LITELLM_SUCCESS_CALLBACKS", "")
        success_callbacks = []
        if success_callbacks_str:
            success_callbacks = [
                callback.strip() for callback in success_callbacks_str.split(",")
            ]

        failure_callbacks_str = os.environ.get("LITELLM_FAILURE_CALLBACKS", "")
        failure_callbacks = []
        if failure_callbacks_str:
            failure_callbacks = [
                callback.strip() for callback in failure_callbacks_str.split(",")
            ]

        litellm.success_callback = success_callbacks
        litellm.failure_callback = failure_callbacks
